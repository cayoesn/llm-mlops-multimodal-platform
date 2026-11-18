"""5-Layer FastAPI Application Factory, Streaming ASGI Guards, Lifespan & Exception Mapping."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.api.routes import PlatformRuntimeContainer, router
from app.core.exceptions import (
    AuthorizationError,
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


class _StreamingBodyLimitMiddleware:
    """ASGI middleware enforcing strict byte ceilings on Content-Length and chunked streams."""

    def __init__(self, app_callable: Any, *, max_bytes: int) -> None:
        """Bind downstream ASGI callable and maximum allowed request body bytes."""
        self._app = app_callable
        self._max_bytes = max_bytes

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """Validate Content-Length header and bound streamed http.request chunks pre-dispatch."""
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        raw_headers: list[tuple[bytes, bytes]] = list(scope.get("headers", []))
        header_map = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in raw_headers}
        req_id = bind_request_context(
            request_id=header_map.get("x-request-id"),
            trace_id=header_map.get("x-trace-id"),
        )

        async def _send_with_correlation(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                hdrs = list(message.get("headers", []))
                existing = {k.lower() for k, _ in hdrs}
                if b"x-request-id" not in existing:
                    hdrs.append((b"x-request-id", req_id.encode("latin-1")))
                if b"x-process-latency-ms" not in existing:
                    latency_bytes = f"{get_elapsed_latency_ms():.2f}".encode("latin-1")
                    hdrs.append((b"x-process-latency-ms", latency_bytes))
                message = {**message, "headers": hdrs}
            await send(message)

        cl_value = header_map.get("content-length")
        if cl_value is not None:
            try:
                declared_bytes = int(cl_value)
            except ValueError:
                bad_cl = JSONResponse(
                    status_code=400,
                    content={"error": "invalid_content_length"},
                )
                await bad_cl(scope, receive, _send_with_correlation)
                return
            if declared_bytes < 0:
                neg_cl = JSONResponse(
                    status_code=400,
                    content={"error": "negative_content_length"},
                )
                await neg_cl(scope, receive, _send_with_correlation)
                return
            if declared_bytes > self._max_bytes:
                log_warning(
                    _LOGGER,
                    "request_payload_rejected_413",
                    declared_bytes=declared_bytes,
                    limit_bytes=self._max_bytes,
                )
                too_big = JSONResponse(
                    status_code=413,
                    content={"error": "payload_too_large", "max_bytes": self._max_bytes},
                )
                await too_big(scope, receive, _send_with_correlation)
                return

        if scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self._app(scope, receive, _send_with_correlation)
            return

        buffered_chunks: list[bytes] = []
        streamed_bytes = 0
        while True:
            message = await receive()
            msg_type = message.get("type")
            if msg_type == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if isinstance(chunk, (bytes, bytearray)) and chunk:
                streamed_bytes += len(chunk)
                if streamed_bytes > self._max_bytes:
                    log_warning(
                        _LOGGER,
                        "chunked_request_payload_rejected_413",
                        streamed_bytes=streamed_bytes,
                        limit_bytes=self._max_bytes,
                    )
                    too_big_stream = JSONResponse(
                        status_code=413,
                        content={"error": "payload_too_large", "max_bytes": self._max_bytes},
                    )
                    await too_big_stream(scope, receive, _send_with_correlation)
                    return
                buffered_chunks.append(bytes(chunk))
            if not message.get("more_body", False):
                break

        full_body = b"".join(buffered_chunks)
        body_delivered = False

        async def _replay_receive() -> MutableMapping[str, Any]:
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {"type": "http.request", "body": full_body, "more_body": False}
            return await receive()

        await self._app(scope, _replay_receive, _send_with_correlation)


def create_app(*, max_request_body_bytes: int = 4_194_304) -> FastAPI:
    """Create and configure the OmniPulse MLOps, LLMOps & Multimodal FastAPI service."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        container = PlatformRuntimeContainer()
        await asyncio.to_thread(container.bootstrap_mlops)
        application.state.container = container
        log_info(_LOGGER, "omnipulse_platform_started", max_body_bytes=max_request_body_bytes)
        yield
        await container.simulator.stop()
        await container.llm_gateway.aclose()
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

    application.add_middleware(_StreamingBodyLimitMiddleware, max_bytes=max_request_body_bytes)

    @application.exception_handler(GuardrailViolationError)
    async def handle_guardrail_error(_request: Request, exc: GuardrailViolationError) -> Response:
        log_warning(_LOGGER, "guardrail_violation_blocked", exc=exc)
        return JSONResponse(
            status_code=400,
            content={"error": "guardrail_violation", "detail": str(exc)},
        )

    @application.exception_handler(AuthorizationError)
    async def handle_authorization_error(_request: Request, exc: AuthorizationError) -> Response:
        log_warning(_LOGGER, "authorization_rejected", exc=exc)
        return JSONResponse(
            status_code=403,
            content={"error": "forbidden", "detail": str(exc)},
        )

    @application.exception_handler(ValidationLayerError)
    async def handle_validation_error(_request: Request, exc: ValidationLayerError) -> Response:
        log_warning(_LOGGER, "validation_layer_rejected", exc=exc)
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": str(exc)},
        )

    @application.exception_handler(ModalitySyncError)
    async def handle_modality_sync_error(_request: Request, exc: ModalitySyncError) -> Response:
        log_warning(_LOGGER, "modality_sync_rejected", exc=exc)
        return JSONResponse(
            status_code=422,
            content={"error": "modality_sync_error", "detail": str(exc)},
        )

    @application.exception_handler(FeatureStoreSkewError)
    async def handle_feast_error(_request: Request, exc: FeatureStoreSkewError) -> Response:
        log_warning(_LOGGER, "feast_entity_lookup_failed", exc=exc)
        return JSONResponse(
            status_code=404,
            content={"error": "feature_store_miss", "detail": str(exc)},
        )

    @application.exception_handler(ModelRegistryStageError)
    async def handle_registry_error(_request: Request, exc: ModelRegistryStageError) -> Response:
        log_warning(_LOGGER, "mlflow_registry_gate_failed", exc=exc)
        return JSONResponse(
            status_code=409,
            content={"error": "registry_stage_conflict", "detail": str(exc)},
        )

    @application.exception_handler(RateLimitExceededError)
    async def handle_rate_limit_error(_request: Request, exc: RateLimitExceededError) -> Response:
        log_warning(_LOGGER, "rate_limit_exceeded", exc=exc)
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "detail": str(exc)},
        )

    @application.exception_handler(CircuitBreakerOpenError)
    @application.exception_handler(ResilienceTimeoutError)
    async def handle_resilience_error(_request: Request, exc: OmniPulseError) -> Response:
        log_error(_LOGGER, "downstream_resilience_circuit_open", exc=exc)
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "detail": str(exc)},
        )

    application.include_router(router)
    return application


app = create_app()
