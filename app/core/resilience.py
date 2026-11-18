"""CircuitBreaker, NoOpCircuitBreaker, TokenBucketRateLimiter, and SLA-aware ResilientExecutor."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, Protocol

from app.core.exceptions import (
    CircuitBreakerOpenError,
    DownstreamHTTPStatusError,
    RateLimitExceededError,
    ResilienceTimeoutError,
)
from app.core.observability import log_error, log_warning

CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]

_LOGGER = logging.getLogger("omnipulse.resilience")
_NON_TRIPPING_CLIENT_STATUSES = frozenset({400, 401, 403, 404, 409, 422})


class CircuitBreakerProtocol(Protocol):
    """Structural protocol implemented by active and no-op circuit breakers."""

    @property
    def state(self) -> CircuitState:
        """Return current circuit state."""

    @property
    def consecutive_failures(self) -> int:
        """Return consecutive failure count."""

    def allow_request(self) -> bool:
        """Return True if an outbound call is permitted."""

    def before_call(self) -> None:
        """Verify circuit allows execution or raise CircuitBreakerOpenError."""

    def record_success(self) -> None:
        """Record a healthy upstream response."""

    def record_failure(self) -> None:
        """Record a terminal upstream failure."""

    def reset(self) -> None:
        """Reset circuit breaker state to CLOSED."""


class NoOpCircuitBreaker:
    """O(1) null-object circuit breaker for bypassed internal components."""

    @property
    def state(self) -> CircuitState:
        """Always report CLOSED."""
        return "CLOSED"

    @property
    def consecutive_failures(self) -> int:
        """Always report 0 failures."""
        return 0

    def allow_request(self) -> bool:
        """Always permit execution."""
        return True

    def before_call(self) -> None:
        """Never raise."""
        return None

    def record_success(self) -> None:
        """No-op success handler."""
        return None

    def record_failure(self) -> None:
        """No-op failure handler."""
        return None

    def reset(self) -> None:
        """No-op reset handler."""
        return None


class CircuitBreaker:
    """Thread-safe RLock tri-state (CLOSED, OPEN, HALF_OPEN) circuit breaker with probe watchdog."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
    ) -> None:
        """Initialize failure threshold and monotonic OPEN->HALF_OPEN cooldown."""
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_seconds <= 0.0:
            raise ValueError("recovery_seconds must be > 0")
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._lock = threading.RLock()
        self._state: CircuitState = "CLOSED"
        self._consecutive_failures = 0
        self._opened_at_monotonic = 0.0
        self._half_open_probe_in_flight = False
        self._half_open_started_monotonic = 0.0

    def _refresh_time_transitions(self, now: float) -> None:
        """Transition expired OPEN states or hung HALF_OPEN probes under RLock."""
        if self._state == "OPEN" and (now - self._opened_at_monotonic) >= self._recovery_seconds:
            self._state = "HALF_OPEN"
            self._half_open_probe_in_flight = False
        elif (
            self._state == "HALF_OPEN"
            and self._half_open_probe_in_flight
            and (now - self._half_open_started_monotonic) >= self._recovery_seconds
        ):
            self._half_open_probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        """Return time-adjusted circuit breaker state."""
        with self._lock:
            self._refresh_time_transitions(time.monotonic())
            return self._state

    @property
    def consecutive_failures(self) -> int:
        """Return consecutive failure count."""
        with self._lock:
            return self._consecutive_failures

    def allow_request(self) -> bool:
        """Check whether an attempt may proceed and acquire HALF_OPEN probe slot if needed."""
        with self._lock:
            now = time.monotonic()
            self._refresh_time_transitions(now)
            if self._state == "CLOSED":
                return True
            if self._state == "HALF_OPEN" and not self._half_open_probe_in_flight:
                self._half_open_probe_in_flight = True
                self._half_open_started_monotonic = now
                return True
            return False

    def before_call(self) -> None:
        """Permit execution or fail-fast with CircuitBreakerOpenError."""
        if not self.allow_request():
            raise CircuitBreakerOpenError("Downstream circuit breaker is OPEN.")

    def record_success(self) -> None:
        """Close the circuit and clear failure counters on healthy upstream response."""
        with self._lock:
            self._state = "CLOSED"
            self._consecutive_failures = 0
            self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        """Increment failure count once per exhausted request and open circuit at threshold."""
        with self._lock:
            now = time.monotonic()
            self._consecutive_failures += 1
            self._half_open_probe_in_flight = False
            if self._state == "HALF_OPEN" or self._consecutive_failures >= self._failure_threshold:
                self._state = "OPEN"
                self._opened_at_monotonic = now

    def release_probe_without_trip(self) -> None:
        """Release HALF_OPEN probe slot on HTTP 429 without opening or resetting failures."""
        with self._lock:
            self._half_open_probe_in_flight = False

    def reset(self) -> None:
        """Force circuit breaker back to pristine CLOSED state."""
        with self._lock:
            self._state = "CLOSED"
            self._consecutive_failures = 0
            self._opened_at_monotonic = 0.0
            self._half_open_probe_in_flight = False


class TokenBucketRateLimiter:
    """Thread-safe continuous fractional Token Bucket rate limiter in O(1) time and space."""

    def __init__(self, capacity: float, refill_rate_per_second: float) -> None:
        """Initialize bucket capacity and token refill rate per second."""
        if capacity <= 0.0 or refill_rate_per_second <= 0.0:
            raise ValueError("capacity and refill_rate_per_second must be positive.")
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate_per_second)
        self._tokens = float(capacity)
        self._last_refill_monotonic = time.monotonic()
        self._lock = threading.RLock()

    def try_consume(self, tokens: float = 1.0) -> bool:
        """Refill elapsed tokens and consume if available in O(1)."""
        cost = max(0.0, float(tokens))
        with self._lock:
            now = time.monotonic()
            elapsed = max(0.0, now - self._last_refill_monotonic)
            self._last_refill_monotonic = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    def consume(self, tokens: float = 1.0) -> None:
        """Consume tokens or raise RateLimitExceededError."""
        if not self.try_consume(tokens):
            raise RateLimitExceededError("Rate limit token bucket exhausted.")


class ResilientExecutor:
    """Centralized retry + circuit-breaker + monotonic SLA budget executor."""

    def __init__(
        self,
        breaker: CircuitBreakerProtocol | None = None,
        max_retries: int = 2,
        base_backoff_seconds: float = 0.02,
    ) -> None:
        """Configure circuit breaker, retry ceiling, and exponential backoff base."""
        self._breaker: CircuitBreakerProtocol = (
            breaker if breaker is not None else NoOpCircuitBreaker()
        )
        self._max_retries = max(0, max_retries)
        self._base_backoff = max(0.0, base_backoff_seconds)

    async def execute_async[T](
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        total_sla_seconds: float = 5.0,
        operation_name: str = "downstream_call",
    ) -> T:
        """Execute an async callable under monotonic SLA deadline and anti-skew breaker rules."""
        self._breaker.before_call()
        deadline = time.monotonic() + max(0.001, total_sla_seconds)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._breaker.record_failure()
                raise ResilienceTimeoutError(
                    f"Operation '{operation_name}' exhausted {total_sla_seconds:.3f}s SLA."
                )
            try:
                result = await asyncio.wait_for(operation(), timeout=remaining)
                self._breaker.record_success()
                return result
            except TimeoutError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    log_warning(
                        _LOGGER,
                        "retryable_timeout",
                        operation=operation_name,
                        attempt=attempt + 1,
                    )
                    continue
                self._breaker.record_failure()
                log_error(_LOGGER, "sla_timeout_terminal", exc=exc, operation=operation_name)
                raise ResilienceTimeoutError(
                    f"Operation '{operation_name}' timed out under {total_sla_seconds:.3f}s SLA."
                ) from exc
            except DownstreamHTTPStatusError as exc:
                if exc.status_code in _NON_TRIPPING_CLIENT_STATUSES:
                    self._breaker.record_success()
                    raise
                if exc.status_code == 429:
                    if isinstance(self._breaker, CircuitBreaker):
                        self._breaker.release_probe_without_trip()
                    raise
                last_exc = exc
                if attempt < self._max_retries:
                    log_warning(
                        _LOGGER,
                        "retryable_http_5xx",
                        operation=operation_name,
                        status_code=exc.status_code,
                        attempt=attempt + 1,
                    )
                    backoff = min(
                        self._base_backoff * (2**attempt),
                        max(0.0, deadline - time.monotonic()),
                    )
                    if backoff > 0.0:
                        await asyncio.sleep(backoff)
                    continue
                self._breaker.record_failure()
                log_error(_LOGGER, "downstream_5xx_terminal", exc=exc, operation=operation_name)
                raise
            except CircuitBreakerOpenError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    log_warning(
                        _LOGGER,
                        "retryable_transport_error",
                        operation=operation_name,
                        attempt=attempt + 1,
                        error_type=type(exc).__name__,
                    )
                    backoff = min(
                        self._base_backoff * (2**attempt),
                        max(0.0, deadline - time.monotonic()),
                    )
                    if backoff > 0.0:
                        await asyncio.sleep(backoff)
                    continue
                self._breaker.record_failure()
                log_error(
                    _LOGGER,
                    "downstream_transport_terminal",
                    exc=exc,
                    operation=operation_name,
                )
                raise

        assert last_exc is not None
        raise last_exc


async def resilient_gather[T](
    coroutines: Sequence[Awaitable[T]],
    *,
    fallback_value: T,
) -> list[T]:
    """Execute concurrent tasks via asyncio.gather substituting fallback_value on exceptions."""
    raw_outcomes = await asyncio.gather(*coroutines, return_exceptions=True)
    resolved: list[T] = []
    for item in raw_outcomes:
        if isinstance(item, BaseException):
            if isinstance(item, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise item
            log_warning(_LOGGER, "resilient_gather_fallback", error_type=type(item).__name__)
            resolved.append(fallback_value)
        else:
            resolved.append(item)
    return resolved


_registry_lock = threading.RLock()
_breaker_registry: dict[str, CircuitBreaker] = {}
_NOOP_SINGLETON = NoOpCircuitBreaker()


def get_circuit_breaker(
    service_name: str,
    *,
    enabled: bool = True,
    failure_threshold: int = 5,
    recovery_seconds: float = 30.0,
) -> CircuitBreakerProtocol:
    """Return a named CircuitBreaker or O(1) NoOpCircuitBreaker when disabled."""
    if not enabled or service_name.startswith("saori_"):
        return _NOOP_SINGLETON
    with _registry_lock:
        existing = _breaker_registry.get(service_name)
        if existing is None:
            existing = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_seconds=recovery_seconds,
            )
            _breaker_registry[service_name] = existing
        return existing


def reset_circuit_breaker_registry() -> None:
    """Reset and clear all registered circuit breakers for hermetic test runs."""
    with _registry_lock:
        for breaker in _breaker_registry.values():
            breaker.reset()
        _breaker_registry.clear()
