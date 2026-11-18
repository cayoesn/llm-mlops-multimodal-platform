"""Feast Feature Store: Offline Parquet PIT joins + Online Redis/Memory serving (Unit II.3)."""

from __future__ import annotations

import bisect
import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.core.exceptions import FeatureStoreSkewError
from app.core.observability import log_event, log_info, log_warning
from feature_repo.features import CUSTOMER_RISK_FEATURE_VIEW

_LOGGER = logging.getLogger("omnipulse.mlops.feast")


class FeastFeatureRecord(BaseModel):
    """Canonical entity-timestamped feature row stored in Offline Parquet and Online Redis."""

    model_config = ConfigDict(frozen=True)

    customer_id: str = Field(min_length=1)
    event_timestamp: int = Field(ge=0)
    tx_velocity_24h: float
    credit_utilization: float
    chargeback_ratio: float
    device_trust_score: float

    def to_feature_vector(self) -> list[float]:
        """Return ordered floating-point feature vector guaranteed identical offline/online."""
        return [
            self.tx_velocity_24h,
            self.credit_utilization,
            self.chargeback_ratio,
            self.device_trust_score,
        ]


class EntityObservationQuery(BaseModel):
    """Point-in-time historical lookup query pairing an entity ID with label timestamp."""

    model_config = ConfigDict(frozen=True)

    customer_id: str = Field(min_length=1)
    request_timestamp: int = Field(ge=0)


class OnlineKVBackend(Protocol):
    """Protocol abstracting Redis or in-memory online key-value feature serving."""

    def put_entity(self, key: str, payload_json: str) -> None:
        """Persist serialized feature vector for an entity key."""

    def get_entity(self, key: str) -> str | None:
        """Fetch serialized feature vector for an entity key in O(1)."""


class InMemoryOnlineBackend:
    """Thread-safe O(1) online feature store fallback for hermetic execution."""

    def __init__(self) -> None:
        """Initialize RLock-protected dictionary."""
        self._lock = threading.RLock()
        self._store: dict[str, str] = {}

    def put_entity(self, key: str, payload_json: str) -> None:
        """Store serialized JSON feature payload under entity key."""
        with self._lock:
            self._store[key] = payload_json

    def get_entity(self, key: str) -> str | None:
        """Retrieve serialized JSON feature payload in O(1)."""
        with self._lock:
            return self._store.get(key)


class RedisOnlineBackend:
    """Redis-backed Feast online feature store adapter with automatic fallback."""

    def __init__(self, redis_url: str) -> None:
        """Configure Redis client and local fallback mirror."""
        import redis

        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.15,
            socket_timeout=0.15,
        )
        self._fallback = InMemoryOnlineBackend()

    def put_entity(self, key: str, payload_json: str) -> None:
        """Write to Redis and mirror locally if Redis daemon is unreachable."""
        self._fallback.put_entity(key, payload_json)
        try:
            self._client.set(f"feast:{CUSTOMER_RISK_FEATURE_VIEW.name}:{key}", payload_json)
        except Exception:
            log_warning(_LOGGER, "redis_online_write_fallback", entity_key=key)

    def get_entity(self, key: str) -> str | None:
        """Read from Redis in O(1) with seamless fallback."""
        try:
            raw = self._client.get(f"feast:{CUSTOMER_RISK_FEATURE_VIEW.name}:{key}")
            if isinstance(raw, str):
                return raw
        except Exception:
            pass
        return self._fallback.get_entity(key)


class FeastFeatureService:
    """Manages Feast Offline Parquet ingestion, O(N log M) PIT joins, and Online materialization."""

    def __init__(
        self,
        offline_parquet_path: Path | None = None,
        online_backend: OnlineKVBackend | None = None,
    ) -> None:
        """Initialize offline Parquet store path and online serving backend."""
        settings = get_settings()
        default_dir = Path(settings.OMNIPULSE_ARTIFACT_DIR)
        self._parquet_path = (
            offline_parquet_path
            if offline_parquet_path is not None
            else default_dir / "offline_features.parquet"
        )
        self._online: OnlineKVBackend = (
            online_backend if online_backend is not None else InMemoryOnlineBackend()
        )
        self._lock = threading.RLock()

    @property
    def offline_parquet_path(self) -> Path:
        """Return path to the Parquet offline feature table."""
        return self._parquet_path

    def ingest_offline_parquet(self, records: Sequence[FeastFeatureRecord]) -> int:
        """Persist validated feature records into columnar Apache Parquet storage."""
        if not records:
            raise FeatureStoreSkewError("Cannot ingest an empty feature batch into Feast.")
        with self._lock:
            self._parquet_path.parent.mkdir(parents=True, exist_ok=True)
            columns: dict[str, list[Any]] = {
                "customer_id": [r.customer_id for r in records],
                "event_timestamp": [r.event_timestamp for r in records],
                "tx_velocity_24h": [r.tx_velocity_24h for r in records],
                "credit_utilization": [r.credit_utilization for r in records],
                "chargeback_ratio": [r.chargeback_ratio for r in records],
                "device_trust_score": [r.device_trust_score for r in records],
            }
            table = pa.table(columns)
            pq.write_table(table, self._parquet_path, compression="snappy")
        log_event(
            _LOGGER,
            "feast_offline_ingested",
            rows=len(records),
            parquet_path=str(self._parquet_path),
        )
        return len(records)

    def _load_sorted_entity_timeline(
        self,
    ) -> dict[str, tuple[list[int], list[FeastFeatureRecord]]]:
        """Load offline Parquet records grouped by entity and sorted ascending by timestamp."""
        if not self._parquet_path.exists():
            raise FeatureStoreSkewError(
                f"Offline Parquet feature store missing at {self._parquet_path}"
            )
        table = pq.read_table(self._parquet_path)
        rows = table.to_pylist()
        grouped: dict[str, list[FeastFeatureRecord]] = {}
        for row in rows:
            rec = FeastFeatureRecord(**row)
            grouped.setdefault(rec.customer_id, []).append(rec)

        indexed: dict[str, tuple[list[int], list[FeastFeatureRecord]]] = {}
        for cust_id, items in grouped.items():
            items.sort(key=lambda r: r.event_timestamp)
            timestamps = [r.event_timestamp for r in items]
            indexed[cust_id] = (timestamps, items)
        return indexed

    def get_historical_features_pit(
        self,
        queries: Sequence[EntityObservationQuery],
    ) -> list[FeastFeatureRecord]:
        """Execute O(Q log M) bisect point-in-time joins preventing future data leakage."""
        timeline = self._load_sorted_entity_timeline()
        joined: list[FeastFeatureRecord] = []
        for query in queries:
            entry = timeline.get(query.customer_id)
            if entry is None:
                raise FeatureStoreSkewError(
                    f"Entity '{query.customer_id}' absent from offline feature store."
                )
            timestamps, records = entry
            pos = bisect.bisect_right(timestamps, query.request_timestamp) - 1
            if pos < 0:
                raise FeatureStoreSkewError(
                    f"No point-in-time feature snapshot <= {query.request_timestamp} "
                    f"for '{query.customer_id}'."
                )
            joined.append(records[pos])
        return joined

    def materialize_to_online_store(self) -> int:
        """Materialize the latest feature vector per entity from Offline Parquet to Online Store."""
        timeline = self._load_sorted_entity_timeline()
        count = 0
        for cust_id, (_, records) in timeline.items():
            latest = records[-1]
            self._online.put_entity(cust_id, latest.model_dump_json())
            count += 1
        log_info(_LOGGER, "feast_materialized_online", entities=count)
        return count

    def get_online_features(self, customer_id: str) -> FeastFeatureRecord:
        """Retrieve real-time inference features for customer_id in O(1)."""
        raw = self._online.get_entity(customer_id)
        if raw is None:
            raise FeatureStoreSkewError(f"Entity '{customer_id}' missing from Feast online store.")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise FeatureStoreSkewError("Corrupted Feast online feature payload.")
        return FeastFeatureRecord(**parsed)
