"""5-Layer FastAPI Application Factory, ASGI Guards, Lifespan & Exception Mapping."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.api.routes import PlatformRuntimeContainer, router
from app.core.exceptions import (
    CircuitBreakerOpenError,
    FeatureStoreSkewError,
    GuardrailViolationError,
    ModalitySyncError,
    ModelRegistryStageError,
    OmniPulseError,
    RateLimitExceededError,
    ResilienceTimeoutError,
    ValidationLayerError,
)
from app.core.observability import (
    bind_request_context,
    get_elapsed_latency_ms,
    log_error,
    log_info,
    log_warning,
)

_LOGGER = logging.getLogger(__name__)


def create_app(*, max_request_body_bytes: int = 4_194_304) -> FastAPI:
    """Create and configure the OmniPulse MLOps, LLMOps & Multimodal FastAPI service."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        container = PlatformRuntimeContainer()
        await asyncio.to_thread(container.bootstrap_mlops)
        application.state.container = container
        log_info(_LOGGER, "omnipulse_platform_started", max_body_bytes=max_request_body_bytes)
        yield
        log_info(_LOGGER, "omnipulse_platform_shutdown")

    application = FastAPI(
        title="OmniPulse AI — Enterprise MLOps, LLMOps & Multimodal LLMOps Platform",
        version="1.0.0",
        description=(
            "Reference Level-2 MLOps, Text LLMOps, and Audio/Image/Video Multimodal LLMOps "
            "platform built with Feast, DVC/CML, MLflow, Airflow, Qdrant, Whisper, and ONNX."
        ),
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def correlation_and_payload_guard(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = bind_request_context(
            request_id=request.headers.get("X-Request-ID"),
            trace_id=request.headers.get("X-Trace-ID"),
        )
        content_len_header = request.headers.get("content-length")
        if content_len_header is not None:
            try:
                declared_bytes = int(content_len_header)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid_content_length"},
                    headers={"X-Request-ID": req_id},
                )
            if declared_bytes > max_request_body_bytes:
                log_warning(
                    _LOGGER,
                    "request_payload_rejected_413",
                    declared_bytes=declared_bytes,
                    limit_bytes=max_request_body_bytes,
                )
                return JSONResponse(
                    status_code=413,
                    content={"error": "payload_too_large", "max_bytes": max_request_body_bytes},
                    headers={"X-Request-ID": req_id},
                )

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Latency-Ms"] = f"{get_elapsed_latency_ms():.2f}"
        return response

    @application.exception_handler(GuardrailViolationError)
    async def handle_guardrail_error(_request: Request, exc: GuardrailViolationError) -> Response:
        log_warning(_LOGGER, "guardrail_violation_blocked", exc=exc)
        return JSONResponse(
            status_code=400, content={"error": "guardrail_violation", "detail": str(exc)}
        )

    @application.exception_handler(ValidationLayerError)
    async def handle_validation_error(_request: Request, exc: ValidationLayerError) -> Response:
        log_warning(_LOGGER, "validation_layer_rejected", exc=exc)
        return JSONResponse(
            status_code=422, content={"error": "validation_error", "detail": str(exc)}
        )

    @application.exception_handler(ModalitySyncError)
    async def handle_modality_sync_error(_request: Request, exc: ModalitySyncError) -> Response:
        log_warning(_LOGGER, "modality_sync_rejected", exc=exc)
        return JSONResponse(
            status_code=422, content={"error": "modality_sync_error", "detail": str(exc)}
        )

    @application.exception_handler(FeatureStoreSkewError)
    async def handle_feast_error(_request: Request, exc: FeatureStoreSkewError) -> Response:
        log_warning(_LOGGER, "feast_entity_lookup_failed", exc=exc)
        return JSONResponse(
            status_code=404, content={"error": "feature_store_miss", "detail": str(exc)}
        )

    @application.exception_handler(ModelRegistryStageError)
    async def handle_registry_error(_request: Request, exc: ModelRegistryStageError) -> Response:
        log_warning(_LOGGER, "mlflow_registry_gate_failed", exc=exc)
        return JSONResponse(
            status_code=409, content={"error": "registry_stage_conflict", "detail": str(exc)}
        )

    @application.exception_handler(RateLimitExceededError)
    async def handle_rate_limit_error(_request: Request, exc: RateLimitExceededError) -> Response:
        log_warning(_LOGGER, "rate_limit_exceeded", exc=exc)
        return JSONResponse(
            status_code=429, content={"error": "rate_limit_exceeded", "detail": str(exc)}
        )

    @application.exception_handler(CircuitBreakerOpenError)
    @application.exception_handler(ResilienceTimeoutError)
    async def handle_resilience_error(_request: Request, exc: OmniPulseError) -> Response:
        log_error(_LOGGER, "downstream_resilience_circuit_open", exc=exc)
        return JSONResponse(
            status_code=503, content={"error": "service_unavailable", "detail": str(exc)}
        )

    application.include_router(router)
    return application


app = create_app()
