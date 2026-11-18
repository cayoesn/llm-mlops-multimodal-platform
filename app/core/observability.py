"""Structured PII-safe logging, ContextVar tracing, OpenTelemetry, Prometheus & Forecasting."""

from __future__ import annotations

import collections
import contextvars
import hashlib
import hmac
import json
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from app.config import get_settings

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "omnipulse_request_id",
    default="req-unbound",
)
trace_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "omnipulse_trace_id",
    default="trace-unbound",
)
request_start_ns_ctx: contextvars.ContextVar[int] = contextvars.ContextVar(
    "omnipulse_request_start_ns",
    default=0,
)

_PII_MASTER_PATTERN = re.compile(
    r"(?P<bearer>Bearer\s+[A-Za-z0-9._\-]{8,})"
    r"|(?P<card>\b(?:\d{4}[-\s]?){3}\d{4}\b)"
    r"|(?P<cpf>\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b)"
    r"|(?P<email>\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b)"
)

_REDACTION_TOKENS: dict[str, str] = {
    "bearer": "Bearer [REDACTED_TOKEN]",
    "card": "[REDACTED_CARD]",
    "cpf": "[REDACTED_CPF]",
    "email": "[REDACTED_EMAIL]",
}

_TRACER_LOCK = threading.RLock()
_TRACER_INITIALIZED = False


def _ensure_otel_tracer() -> trace.Tracer:
    """Initialize OpenTelemetry TracerProvider once and return the platform tracer."""
    global _TRACER_INITIALIZED
    with _TRACER_LOCK:
        if not _TRACER_INITIALIZED:
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            _TRACER_INITIALIZED = True
        return trace.get_tracer("omnipulse.platform", "1.0.0")


def scrub_pii_text(raw: str) -> str:
    """Scrub tokens, cards, CPFs, and emails in a single O(|T|) regex pass."""

    def _replace(match: re.Match[str]) -> str:
        group_name = match.lastgroup
        if group_name is None:
            return "[REDACTED]"
        return _REDACTION_TOKENS[group_name]

    return _PII_MASTER_PATTERN.sub(_replace, raw)


def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings inside structured log fields."""
    if isinstance(value, str):
        return scrub_pii_text(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return value


def bind_request_context(request_id: str | None = None, trace_id: str | None = None) -> str:
    """Bind request_id, trace_id, and start timestamp into coroutine-local ContextVars."""
    resolved_req = (
        request_id.strip() if request_id and request_id.strip() else f"req-{uuid.uuid4().hex[:12]}"
    )
    resolved_trace = trace_id.strip() if trace_id and trace_id.strip() else uuid.uuid4().hex
    request_id_ctx.set(resolved_req)
    trace_id_ctx.set(resolved_trace)
    request_start_ns_ctx.set(time.perf_counter_ns())
    return resolved_req


def get_elapsed_latency_ms() -> float:
    """Return elapsed milliseconds from the ContextVar monotonic request start."""
    started = request_start_ns_ctx.get()
    if started <= 0:
        return 0.05
    return max(0.01, (time.perf_counter_ns() - started) / 1_000_000.0)


def _format_structured_payload(event: str, fields: dict[str, Any]) -> str:
    """Serialize structured log payload with ContextVar correlation and PII redaction."""
    envelope: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": scrub_pii_text(event),
        "request_id": request_id_ctx.get(),
        "trace_id": trace_id_ctx.get(),
    }
    for key, val in fields.items():
        envelope[key] = _sanitize_value(val)
    return json.dumps(envelope, separators=(",", ":"), ensure_ascii=True, default=str)


def log_info(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Emit a structured PII-safe INFO log entry."""
    logger.info(_format_structured_payload(message, fields))


def log_warning(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Emit a structured PII-safe WARNING log entry."""
    logger.warning(_format_structured_payload(message, fields))


def log_debug(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Emit a structured PII-safe DEBUG log entry."""
    logger.debug(_format_structured_payload(message, fields))


def log_error(
    logger: logging.Logger,
    message: str,
    *,
    exc: BaseException | None = None,
    **fields: Any,
) -> None:
    """Emit a structured PII-safe ERROR log entry preserving sanitized exception metadata."""
    enriched = dict(fields)
    if exc is not None:
        enriched["error_type"] = type(exc).__name__
        enriched["error_detail"] = scrub_pii_text(str(exc))
    logger.error(_format_structured_payload(message, enriched))


def log_event(logger: logging.Logger, event_name: str, **fields: Any) -> None:
    """Emit a domain lifecycle event log entry."""
    enriched = {"category": "domain_event", **fields}
    logger.info(_format_structured_payload(event_name, enriched))


def log_tool_result(
    logger: logging.Logger,
    tool_name: str,
    *,
    success: bool,
    latency_ms: float,
    **fields: Any,
) -> None:
    """Emit a standardized tool/stage execution result log entry."""
    enriched = {
        "category": "tool_result",
        "tool": tool_name,
        "success": success,
        "latency_ms": round(latency_ms, 3),
        **fields,
    }
    logger.info(_format_structured_payload(f"tool:{tool_name}", enriched))


@dataclass(frozen=True, slots=True)
class AuditChainRecord:
    """Immutable tamper-evident link in the canonical HMAC-SHA256 audit chain."""

    seq: int
    timestamp_utc: str
    request_id: str
    pillar: str
    action: str
    payload_hash: str
    confidence: float
    previous_hash: str
    signature: str


class CryptographicAuditChain:
    """Collision-free canonical JSON HMAC-SHA256 audit chain with O(1) append and O(M) verify."""

    GENESIS_DIGEST = "0" * 64

    def __init__(self, secret_key: str | None = None) -> None:
        """Initialize the audit chain with an HMAC secret of at least 8 characters."""
        raw_secret = (
            secret_key if secret_key is not None else get_settings().OMNIPULSE_AUDIT_HMAC_SECRET
        )
        if len(raw_secret.strip()) < 8:
            raise ValueError("HMAC secret_key must be at least 8 characters long.")
        self._secret_bytes = raw_secret.encode("utf-8")
        self._lock = threading.RLock()
        self._records: list[AuditChainRecord] = []

    def __len__(self) -> int:
        """Return total number of audit records."""
        with self._lock:
            return len(self._records)

    def records(self) -> list[AuditChainRecord]:
        """Return a thread-safe shallow copy of all audit records."""
        with self._lock:
            return list(self._records)

    def _compute_canonical_mac(
        self,
        seq: int,
        timestamp_utc: str,
        request_id: str,
        pillar: str,
        action: str,
        payload_hash: str,
        confidence: float,
        previous_hash: str,
    ) -> str:
        """Compute collision-free HMAC-SHA256 over canonical JSON array tuple."""
        canonical_tuple = [
            seq,
            previous_hash,
            timestamp_utc,
            request_id,
            pillar,
            action,
            payload_hash,
            round(confidence, 6),
        ]
        encoded = json.dumps(
            canonical_tuple,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hmac.new(self._secret_bytes, encoded, hashlib.sha256).hexdigest()

    def append(
        self,
        *,
        request_id: str,
        pillar: str,
        action: str,
        payload_hash: str,
        confidence: float,
    ) -> AuditChainRecord:
        """Append a signed audit record in O(1) under RLock."""
        if isinstance(confidence, bool) or not math.isfinite(confidence):
            raise ValueError("Audit confidence must be a finite float.")
        with self._lock:
            seq = len(self._records) + 1
            prev_hash = self._records[-1].signature if self._records else self.GENESIS_DIGEST
            ts = datetime.now(UTC).isoformat()
            mac = self._compute_canonical_mac(
                seq=seq,
                timestamp_utc=ts,
                request_id=request_id,
                pillar=pillar,
                action=action,
                payload_hash=payload_hash,
                confidence=confidence,
                previous_hash=prev_hash,
            )
            record = AuditChainRecord(
                seq=seq,
                timestamp_utc=ts,
                request_id=request_id,
                pillar=pillar,
                action=action,
                payload_hash=payload_hash,
                confidence=round(confidence, 6),
                previous_hash=prev_hash,
                signature=mac,
            )
            self._records.append(record)
            return record

    def verify(self) -> tuple[bool, int, str]:
        """Verify strict sequence monotonicity, hash linkage, and HMAC signatures in O(M)."""
        with self._lock:
            snapshot = list(self._records)
        expected_prev = self.GENESIS_DIGEST
        for idx, rec in enumerate(snapshot, start=1):
            if rec.seq != idx:
                return False, idx, f"Sequence discontinuity at index {idx}: got {rec.seq}"
            if rec.previous_hash != expected_prev:
                return False, idx, f"Broken previous_hash link at seq {rec.seq}"
            expected_mac = self._compute_canonical_mac(
                seq=rec.seq,
                timestamp_utc=rec.timestamp_utc,
                request_id=rec.request_id,
                pillar=rec.pillar,
                action=rec.action,
                payload_hash=rec.payload_hash,
                confidence=rec.confidence,
                previous_hash=rec.previous_hash,
            )
            if not hmac.compare_digest(rec.signature, expected_mac):
                return False, idx, f"Invalid HMAC signature at seq {rec.seq}"
            expected_prev = rec.signature
        return True, len(snapshot), f"Verified {len(snapshot)} canonical HMAC-SHA256 records."


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    """Time-series Holt double-exponential smoothing forecast result."""

    observations: int
    current_level: float
    trend_slope: float
    next_value: float
    horizon_value: float
    horizon_steps: int


class HoltLinearForecaster:
    """Bounded-window Holt linear trend forecaster in O(1) observe and O(1) forecast."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        window_size: int = 64,
    ) -> None:
        """Initialize smoothing weights and bounded FIFO observation deque."""
        if not (0.0 < alpha <= 1.0 and 0.0 < beta <= 1.0):
            raise ValueError("Smoothing factors alpha and beta must be in (0, 1].")
        self._alpha = alpha
        self._beta = beta
        self._history: collections.deque[float] = collections.deque(maxlen=max(4, window_size))
        self._level: float = 0.0
        self._trend: float = 0.0
        self._lock = threading.RLock()

    def observe(self, value: float) -> None:
        """Ingest one finite scalar metric sample in O(1)."""
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("Observed metric value must be a finite number.")
        with self._lock:
            n = len(self._history)
            self._history.append(float(value))
            if n == 0:
                self._level = float(value)
                self._trend = 0.0
            elif n == 1:
                prev = self._level
                self._level = float(value)
                self._trend = float(value) - prev
            else:
                prev_level = self._level
                self._level = self._alpha * float(value) + (1.0 - self._alpha) * (
                    prev_level + self._trend
                )
                self._trend = (
                    self._beta * (self._level - prev_level) + (1.0 - self._beta) * self._trend
                )

    def forecast(self, horizon: int = 5) -> ForecastSnapshot:
        """Project future metric trajectory at 1 step and horizon steps."""
        steps = max(1, horizon)
        with self._lock:
            count = len(self._history)
            if count == 0:
                return ForecastSnapshot(0, 0.0, 0.0, 0.0, 0.0, steps)
            return ForecastSnapshot(
                observations=count,
                current_level=round(self._level, 4),
                trend_slope=round(self._trend, 4),
                next_value=round(self._level + self._trend, 4),
                horizon_value=round(self._level + steps * self._trend, 4),
                horizon_steps=steps,
            )


class ObservabilityHub:
    """Unified 3D Observability manager: Infrastructure, Inference Service, and Model Quality."""

    def __init__(self) -> None:
        """Create isolated Prometheus registry, OpenTelemetry tracer, forecasters, and audit log."""
        self.registry = CollectorRegistry(auto_describe=True)
        self.tracer = _ensure_otel_tracer()
        self.audit_chain = CryptographicAuditChain()
        self.rps_forecaster = HoltLinearForecaster()
        self.drift_forecaster = HoltLinearForecaster()
        self._lock = threading.RLock()
        self._quality_latest: dict[str, float] = {}
        self._infra_latest: dict[str, float] = {
            "cpu_utilization_pct": 18.5,
            "memory_used_mb": 412.0,
            "gpu_vram_simulated_mb": 1024.0,
            "parquet_storage_mb": 24.0,
        }

        self._req_counter = Counter(
            "omnipulse_inference_requests_total",
            "Total inference requests processed per pillar and outcome",
            ["pillar", "status"],
            registry=self.registry,
        )
        self._latency_hist = Histogram(
            "omnipulse_inference_latency_seconds",
            "Inference service latency distribution in seconds",
            ["pillar"],
            registry=self.registry,
        )
        self._quality_gauge = Gauge(
            "omnipulse_model_quality_score",
            "Statistical and semantic quality metrics across MLOps, LLMOps, and Multimodal pillars",
            ["pillar", "metric"],
            registry=self.registry,
        )
        self._infra_gauge = Gauge(
            "omnipulse_infrastructure_resource",
            "Infrastructure telemetry across CPU, RAM, GPU VRAM, and storage",
            ["resource"],
            registry=self.registry,
        )
        for res_name, val in self._infra_latest.items():
            self._infra_gauge.labels(resource=res_name).set(val)

    def record_inference(self, pillar: str, *, latency_seconds: float, success: bool) -> None:
        """Record Dimension-2 service reliability & latency metrics."""
        status = "success" if success else "error"
        safe_latency = max(0.0, float(latency_seconds))
        self._req_counter.labels(pillar=pillar, status=status).inc()
        self._latency_hist.labels(pillar=pillar).observe(safe_latency)
        self.rps_forecaster.observe(1.0 / max(0.001, safe_latency))

    def record_model_quality(self, *, pillar: str, metric_name: str, value: float) -> None:
        """Record Dimension-3 model statistical or semantic quality metric."""
        if isinstance(value, bool) or not math.isfinite(value):
            return
        self._quality_gauge.labels(pillar=pillar, metric=metric_name).set(float(value))
        with self._lock:
            self._quality_latest[f"{pillar}.{metric_name}"] = round(float(value), 6)
        if "drift" in metric_name or "psi" in metric_name:
            self.drift_forecaster.observe(float(value))

    def update_infrastructure(self, **metrics: float) -> None:
        """Record Dimension-1 infrastructure telemetry gauges."""
        with self._lock:
            for name, val in metrics.items():
                if not isinstance(val, bool) and math.isfinite(val):
                    self._infra_latest[name] = round(float(val), 3)
                    self._infra_gauge.labels(resource=name).set(float(val))

    def snapshot_3d(self) -> dict[str, Any]:
        """Return a consolidated 3D observability snapshot across all three dimensions."""
        with self._lock:
            infra_copy = dict(self._infra_latest)
            quality_copy = dict(self._quality_latest)
        valid, count, _ = self.audit_chain.verify()
        return {
            "dimension_1_infrastructure": infra_copy,
            "dimension_2_service_forecast": {
                "throughput_forecast": self.rps_forecaster.forecast(horizon=5).__dict__,
                "drift_forecast": self.drift_forecaster.forecast(horizon=5).__dict__,
            },
            "dimension_3_model_quality": quality_copy,
            "audit_chain": {
                "valid": valid,
                "records": count,
            },
        }

    def export_prometheus_text(self) -> str:
        """Serialize the Prometheus registry into OpenMetrics text format."""
        return generate_latest(self.registry).decode("utf-8")


_hub_lock = threading.RLock()
_global_hub: ObservabilityHub | None = None


def get_observability_hub() -> ObservabilityHub:
    """Return the thread-safe global ObservabilityHub singleton."""
    global _global_hub
    with _hub_lock:
        if _global_hub is None:
            _global_hub = ObservabilityHub()
        return _global_hub


def reset_observability_hub() -> None:
    """Reset the global ObservabilityHub singleton for isolated unit tests."""
    global _global_hub
    with _hub_lock:
        _global_hub = None
