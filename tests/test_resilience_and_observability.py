"""TDD tests for Component 1: Config, Resilience, 3D Observability, Forecaster & FinOps ROI."""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.config import Settings, get_settings
from app.core.exceptions import (
    CircuitBreakerOpenError,
    DownstreamHTTPStatusError,
    RateLimitExceededError,
    ResilienceTimeoutError,
)
from app.core.finops_roi import FinOpsROIEngine, FinOpsWorkloadInput
from app.core.observability import (
    CryptographicAuditChain,
    HoltLinearForecaster,
    bind_request_context,
    get_observability_hub,
    log_debug,
    log_error,
    log_event,
    log_info,
    log_tool_result,
    log_warning,
    request_id_ctx,
)
from app.core.resilience import (
    CircuitBreaker,
    NoOpCircuitBreaker,
    ResilientExecutor,
    TokenBucketRateLimiter,
    get_circuit_breaker,
    resilient_gather,
)


def test_settings_handles_empty_env_strings_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty string environment variables fall back to typed defaults instead of ValueError."""
    monkeypatch.setenv("OMNIPULSE_API_PORT", "")
    monkeypatch.setenv("OMNIPULSE_PSI_DRIFT_THRESHOLD", "")
    settings = Settings()
    assert settings.OMNIPULSE_API_PORT == 8017
    assert settings.OMNIPULSE_PSI_DRIFT_THRESHOLD == pytest.approx(0.20)
    assert get_settings().OMNIPULSE_API_PORT == 8017


def test_noop_circuit_breaker_always_closed() -> None:
    """NoOpCircuitBreaker operates in O(1) and never blocks execution."""
    breaker = get_circuit_breaker("omnipulse_noop_bypass_service", enabled=False)
    assert isinstance(breaker, NoOpCircuitBreaker)
    assert breaker.state == "CLOSED"
    assert breaker.consecutive_failures == 0
    breaker.record_failure()
    assert breaker.state == "CLOSED"
    assert breaker.allow_request() is True


@pytest.mark.asyncio
async def test_circuit_breaker_records_single_failure_after_retries_and_ignores_4xx_429() -> None:
    """ResilientExecutor avoids threshold skew on retries and does not trip on HTTP 404 or 429."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=0.05)
    executor = ResilientExecutor(breaker=breaker, max_retries=2, base_backoff_seconds=0.001)

    attempts = 0

    async def transient_then_ok() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("upstream reset")
        return "recovered"

    result = await executor.execute_async(transient_then_ok, total_sla_seconds=1.0)
    assert result == "recovered"
    assert attempts == 3
    assert breaker.consecutive_failures == 0
    assert breaker.state == "CLOSED"

    async def always_503() -> str:
        raise DownstreamHTTPStatusError(status_code=503, message="service unavailable")

    with pytest.raises(DownstreamHTTPStatusError):
        await executor.execute_async(always_503, total_sla_seconds=1.0)
    assert breaker.consecutive_failures == 1
    assert breaker.state == "CLOSED"

    async def rate_limited_429() -> str:
        raise DownstreamHTTPStatusError(status_code=429, message="too many requests")

    with pytest.raises(DownstreamHTTPStatusError):
        await executor.execute_async(rate_limited_429, total_sla_seconds=1.0)
    assert breaker.consecutive_failures == 1
    assert breaker.state == "CLOSED"

    with pytest.raises(DownstreamHTTPStatusError):
        await executor.execute_async(always_503, total_sla_seconds=1.0)
    assert breaker.consecutive_failures == 2
    assert str(breaker.state) == "OPEN"

    with pytest.raises(CircuitBreakerOpenError):
        await executor.execute_async(always_503, total_sla_seconds=1.0)

    await asyncio.sleep(0.06)
    assert str(breaker.state) == "HALF_OPEN"

    async def not_found_404() -> str:
        raise DownstreamHTTPStatusError(status_code=404, message="entity missing")

    with pytest.raises(DownstreamHTTPStatusError):
        await executor.execute_async(not_found_404, total_sla_seconds=1.0)
    assert breaker.state == "CLOSED"
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_resilient_executor_enforces_monotonic_global_sla() -> None:
    """Slow upstream calls abort under total_sla_seconds without retry multiplication overrun."""
    executor = ResilientExecutor(
        breaker=NoOpCircuitBreaker(), max_retries=3, base_backoff_seconds=0.05
    )

    async def hung_call() -> str:
        await asyncio.sleep(0.25)
        return "never"

    with pytest.raises(ResilienceTimeoutError):
        await executor.execute_async(hung_call, total_sla_seconds=0.04)


@pytest.mark.asyncio
async def test_token_bucket_and_resilient_gather_concurrency() -> None:
    """TokenBucketRateLimiter enforces O(1) capacity and resilient_gather handles fallbacks."""
    limiter = TokenBucketRateLimiter(capacity=3.0, refill_rate_per_second=10.0)
    limiter.consume(2.0)
    limiter.consume(1.0)
    with pytest.raises(RateLimitExceededError):
        limiter.consume(1.0)

    async def ok_task(val: int) -> int:
        await asyncio.sleep(0.005)
        return val * 2

    async def failing_task() -> int:
        raise RuntimeError("boom")

    results = await resilient_gather(
        [ok_task(4), failing_task(), ok_task(7)],
        fallback_value=-1,
    )
    assert results == [8, -1, 14]


@pytest.mark.asyncio
async def test_contextvar_isolation_and_pii_scrubbing_in_structured_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Structured log helpers preserve per-coroutine ContextVar IDs and scrub PII in O(|T|)."""
    logger = logging.getLogger("omnipulse.test")
    caplog.set_level(logging.DEBUG, logger="omnipulse.test")

    async def worker(req_id: str, cpf: str) -> str:
        bind_request_context(req_id)
        await asyncio.sleep(0.005)
        log_info(logger, "processing_customer", cpf=cpf, email="alice@example.com")
        return request_id_ctx.get()

    id_a, id_b = await asyncio.gather(
        worker("req-alpha", "123.456.789-01"),
        worker("req-beta", "98765432100"),
    )
    assert id_a == "req-alpha"
    assert id_b == "req-beta"

    log_debug(logger, "debug_trace", token="Bearer sk-secret-token-9999")
    log_warning(logger, "retry_attempt", attempt=1)
    log_event(logger, "pipeline_stage", stage="feast_materialize", rows=128)
    log_tool_result(logger, "whisper_transcribe", success=True, latency_ms=12.4)
    try:
        raise ValueError("card 4532-1111-2222-3333 rejected")
    except ValueError as exc:
        log_error(logger, "payment_validation_failed", exc=exc)

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "123.456.789-01" not in combined
    assert "98765432100" not in combined
    assert "alice@example.com" not in combined
    assert "sk-secret-token-9999" not in combined
    assert "4532-1111-2222-3333" not in combined
    assert "[REDACTED_CPF]" in combined
    assert "[REDACTED_EMAIL]" in combined


def test_cryptographic_audit_chain_collision_free_and_tamper_evident() -> None:
    """Canonical JSON serialization prevents pipe-delimiter collisions and detects tampering."""
    chain_a = CryptographicAuditChain(secret_key="audit-secret-key-omnipulse")
    chain_b = CryptographicAuditChain(secret_key="audit-secret-key-omnipulse")

    rec_a = chain_a.append(
        request_id="req-1",
        pillar="mlops|fraud",
        action="promote",
        payload_hash="sha256:abc",
        confidence=0.92,
    )
    rec_b = chain_b.append(
        request_id="req-1",
        pillar="mlops",
        action="fraud|promote",
        payload_hash="sha256:abc",
        confidence=0.92,
    )
    assert rec_a.signature != rec_b.signature
    assert rec_a.seq == 1
    assert rec_a.previous_hash == CryptographicAuditChain.GENESIS_DIGEST

    chain_a.append(
        request_id="req-2",
        pillar="multimodal",
        action="inspect_approved",
        payload_hash="sha256:def",
        confidence=0.88,
    )
    valid, count, message = chain_a.verify()
    assert valid is True
    assert count == 2
    assert "verified" in message.lower()


def test_holt_forecaster_and_3d_observability_hub() -> None:
    """HoltLinearForecaster predicts upward trends and ObservabilityHub exports Prometheus."""
    forecaster = HoltLinearForecaster(alpha=0.6, beta=0.4, window_size=32)
    for step in range(1, 11):
        forecaster.observe(float(step * 10))
    prediction = forecaster.forecast(horizon=3)
    assert prediction.next_value > 100.0
    assert prediction.horizon_value > prediction.next_value
    assert prediction.trend_slope > 0.0

    hub = get_observability_hub()
    hub.record_inference("mlops", latency_seconds=0.012, success=True)
    hub.record_model_quality(
        pillar="mlops",
        metric_name="roc_auc",
        value=0.91,
    )
    metrics_payload = hub.export_prometheus_text()
    assert "omnipulse_inference_requests_total" in metrics_payload
    assert "omnipulse_model_quality_score" in metrics_payload


def test_finops_roi_engine_quantifies_savings_and_flags_negative_roi() -> None:
    """FinOpsROIEngine models Unit 1 Fig. 17 & Unit 5.5 ROI and catches negative-ROI over-spend."""
    engine = FinOpsROIEngine()
    healthy = engine.evaluate(
        FinOpsWorkloadInput(
            monthly_requests=500_000,
            business_value_per_1k_requests_usd=18.0,
            gpu_hours_baseline=240.0,
            quantization_bits=4,
            semantic_cache_hit_ratio=0.45,
            autoscaling_efficiency_ratio=0.30,
            raw_storage_gb=500.0,
            use_parquet_compression=True,
            retraining_runs_per_month=4,
        )
    )
    assert healthy.roi_ratio > 1.0
    assert healthy.negative_roi_risk is False
    assert healthy.cost_savings_pct > 50.0
    assert healthy.effective_storage_gb < 150.0

    over_engineered = engine.evaluate(
        FinOpsWorkloadInput(
            monthly_requests=1_000,
            business_value_per_1k_requests_usd=2.0,
            gpu_hours_baseline=400.0,
            quantization_bits=16,
            semantic_cache_hit_ratio=0.0,
            autoscaling_efficiency_ratio=0.0,
            raw_storage_gb=2_000.0,
            use_parquet_compression=False,
            retraining_runs_per_month=30,
        )
    )
    assert over_engineered.roi_ratio < 0.0
    assert over_engineered.negative_roi_risk is True
