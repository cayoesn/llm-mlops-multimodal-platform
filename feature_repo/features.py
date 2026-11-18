"""Canonical Feast entity and feature view metadata for customer risk scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureViewMetadata:
    """Immutable Feast feature view schema descriptor."""

    name: str
    entity_key: str
    ttl_seconds: int
    features: tuple[str, ...]
    offline_format: str = "parquet"
    online_backend: str = "redis"


CUSTOMER_RISK_FEATURE_VIEW = FeatureViewMetadata(
    name="customer_transaction_risk_v1",
    entity_key="customer_id",
    ttl_seconds=86_400,
    features=(
        "tx_velocity_24h",
        "credit_utilization",
        "chargeback_ratio",
        "device_trust_score",
    ),
)
