"""Domain exception hierarchy for OmniPulse MLOps, LLMOps, and Multimodal subsystems."""

from __future__ import annotations


class OmniPulseError(RuntimeError):
    """Base exception for all OmniPulse operational and domain errors."""


class ValidationLayerError(OmniPulseError):
    """Raised when input validation, numeric finiteness, or media bounds fail."""


class GuardrailViolationError(OmniPulseError):
    """Raised when prompt injection, toxicity, or LGPD/GDPR policy checks block a payload."""


class FeatureStoreSkewError(OmniPulseError):
    """Raised when Feast entity keys, point-in-time bounds, or schemas violate parity."""


class ModelRegistryStageError(OmniPulseError):
    """Raised when a model fails SLO quality gates during MLflow stage transition."""


class ModalitySyncError(OmniPulseError):
    """Raised when multimodal inputs fail temporal, schema, or decompression safety checks."""


class ResilienceError(OmniPulseError):
    """Base exception for resilience, rate-limiting, and SLA budget enforcement."""


class CircuitBreakerOpenError(ResilienceError):
    """Raised fail-fast when a downstream circuit breaker is in the OPEN state."""


class RateLimitExceededError(ResilienceError):
    """Raised when a TokenBucketRateLimiter lacks sufficient tokens."""


class ResilienceTimeoutError(ResilienceError):
    """Raised when an operation exceeds its monotonic total SLA budget."""


class DownstreamHTTPStatusError(ResilienceError):
    """Structured HTTP status exception enabling semantic 4xx/429 vs 5xx classification."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with explicit HTTP status code and diagnostic message."""
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
