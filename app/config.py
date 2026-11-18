"""Typed platform configuration with defensive empty-string environment fallback."""

from __future__ import annotations

import threading
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable runtime settings for OmniPulse MLOps, LLMOps, and Multimodal pipelines."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    OMNIPULSE_ENV: str = "production"
    OMNIPULSE_API_PORT: int = 8017
    OMNIPULSE_ARTIFACT_DIR: str = "artifacts/runtime"
    OMNIPULSE_AUDIT_HMAC_SECRET: str = "omnipulse-lgpd-hmac-secret-key-2026"

    OMNIPULSE_MLFLOW_TRACKING_URI: str = "http://localhost:5005"
    OMNIPULSE_REDIS_URL: str = "redis://localhost:6379/0"
    OMNIPULSE_QDRANT_URL: str = "http://localhost:6333"
    OMNIPULSE_OLLAMA_BASE_URL: str = "http://localhost:11434"
    OMNIPULSE_VLLM_BASE_URL: str = "http://localhost:8001"
    OMNIPULSE_LLM_BACKEND: str = "local_deterministic"

    OMNIPULSE_PSI_DRIFT_THRESHOLD: float = 0.20
    OMNIPULSE_KS_DRIFT_ALPHA: float = 0.05
    OMNIPULSE_MIN_CHAMPION_ROC_AUC: float = 0.80
    OMNIPULSE_AB_CHALLENGER_TRAFFIC_RATIO: float = 0.20

    OMNIPULSE_SEMANTIC_CACHE_SIMILARITY: float = 0.90
    OMNIPULSE_SEMANTIC_CACHE_MAX_SIZE: int = 512
    OMNIPULSE_SEMANTIC_CACHE_TTL_SECONDS: float = 900.0
    OMNIPULSE_MIN_RAGAS_FAITHFULNESS: float = 0.70
    OMNIPULSE_MAX_TOXICITY_SCORE: float = 0.35

    OMNIPULSE_MAX_IMAGE_PIXELS: int = 16_777_216
    OMNIPULSE_MAX_IMAGE_SIDE: int = 4096
    OMNIPULSE_MIN_CROSS_MODAL_SYNERGY: float = 0.50

    OMNIPULSE_GPU_HOUR_RATE_USD: float = 2.40
    OMNIPULSE_STORAGE_GB_MONTH_USD: float = 0.023
    OMNIPULSE_RETRAIN_RUN_COST_USD: float = 4.50

    OMNIPULSE_SIMULATOR_ADMIN_TOKEN: str = ""
    OMNIPULSE_SIMULATOR_RETRAIN_COOLDOWN_SECONDS: float = 10.0
    OMNIPULSE_MAX_AUDIT_RECORDS: int = 2000
    OMNIPULSE_MLFLOW_MAX_ARCHIVED_VERSIONS: int = 8

    @field_validator(
        "OMNIPULSE_API_PORT",
        "OMNIPULSE_SEMANTIC_CACHE_MAX_SIZE",
        "OMNIPULSE_MAX_IMAGE_PIXELS",
        "OMNIPULSE_MAX_IMAGE_SIDE",
        "OMNIPULSE_MAX_AUDIT_RECORDS",
        "OMNIPULSE_MLFLOW_MAX_ARCHIVED_VERSIONS",
        mode="before",
    )
    @classmethod
    def coerce_empty_int(cls, value: Any, info: Any) -> Any:
        """Replace blank string environment overrides with the field default."""
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].default
        return value

    @field_validator(
        "OMNIPULSE_PSI_DRIFT_THRESHOLD",
        "OMNIPULSE_KS_DRIFT_ALPHA",
        "OMNIPULSE_MIN_CHAMPION_ROC_AUC",
        "OMNIPULSE_AB_CHALLENGER_TRAFFIC_RATIO",
        "OMNIPULSE_SEMANTIC_CACHE_SIMILARITY",
        "OMNIPULSE_SEMANTIC_CACHE_TTL_SECONDS",
        "OMNIPULSE_MIN_RAGAS_FAITHFULNESS",
        "OMNIPULSE_MAX_TOXICITY_SCORE",
        "OMNIPULSE_MIN_CROSS_MODAL_SYNERGY",
        "OMNIPULSE_GPU_HOUR_RATE_USD",
        "OMNIPULSE_STORAGE_GB_MONTH_USD",
        "OMNIPULSE_RETRAIN_RUN_COST_USD",
        "OMNIPULSE_SIMULATOR_RETRAIN_COOLDOWN_SECONDS",
        mode="before",
    )
    @classmethod
    def coerce_empty_float(cls, value: Any, info: Any) -> Any:
        """Replace blank string environment overrides with the float field default."""
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].default
        return value


_settings_lock = threading.RLock()
_cached_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the thread-safe cached Settings singleton."""
    global _cached_settings
    with _settings_lock:
        if _cached_settings is None:
            _cached_settings = Settings()
        return _cached_settings


def reset_settings_cache() -> None:
    """Clear the cached Settings singleton for hermetic test isolation."""
    global _cached_settings
    with _settings_lock:
        _cached_settings = None
