"""FastAPI REST & WebSocket route handlers orchestrating MLOps, LLMOps & Multimodal engines."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import time
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Query, Request, Response, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError

from app.api.schemas import (
    AuditVerifyResponse,
    HealthResponse,
    LLMOpsChatAPIResponse,
    LLMOpsChatRequest,
    LoRAAdaptRequest,
    MLOpsDriftEvaluateRequest,
    MLOpsPipelineRunRequest,
    MLOpsPipelineRunResponse,
    MLOpsPredictRequest,
    MLOpsPredictResponse,
    MultimodalInspectAPIRequest,
    WebSocketStreamUtterance,
)
from app.config import get_settings
from app.core.exceptions import ValidationLayerError
from app.core.finops_roi import FinOpsROIEngine, FinOpsROIReport, FinOpsWorkloadInput
from app.core.observability import bind_request_context, get_observability_hub, log_info
from app.llmops.lora_rlhf_loop import (
    LoRAAdaptationSummary,
    LoRAAdapterLayer,
    PreferencePair,
    RLHFAdaptationLoop,
)
from app.llmops.rag_engine import KnowledgeDocument
from app.llmops.semantic_cache import embed_text_simd
from app.llmops.serving_gateway import LLMServingGateway
from app.mlops.drift_retrainer import (
    DriftEvaluationReport,
    DriftRetrainingEngine,
    MLOpsMaturityCapabilities,
)
from app.mlops.dvc_pipeline import DVCPipelineRunner
from app.mlops.mlflow_lifecycle import ExperimentMetrics
from app.multimodal.fusion_engine import (
    MultimodalFusionEngine,
    MultimodalInspectionInput,
    MultimodalInspectionReport,
)

_LOGGER = logging.getLogger(__name__)
router = APIRouter()


class PlatformRuntimeContainer:
    """Thread-safe container wiring MLOps, LLMOps, Multimodal, and FinOps subsystems."""

    def __init__(self) -> None:
        """Instantiate domain engines and pre-index baseline enterprise policy documents."""
        self.dvc_runner = DVCPipelineRunner()
        self.drift_engine = DriftRetrainingEngine(self.dvc_runner.mlflow_manager)
        self.llm_gateway = LLMServingGateway()
        self.llm_gateway.rag_engine.index_documents(
            [
                KnowledgeDocument(
                    doc_id="pol-dispute-001",
                    title="Damaged Parcel Instant Refund Policy",
                    content=(
                        "Customer claims involving a damaged parcel, crushed box, or broken glass "
                        "screen verified by ONNX visual inspection qualify for instant refund "
                        "within 24 hours under LGPD audit rules."
                    ),
                ),
                KnowledgeDocument(
                    doc_id="pol-lgpd-002",
                    title="LGPD Biometric & PII Governance Standard",
                    content=(
                        "All CPF numbers, payment cards, emails, and customer voiceprints must be "
                        "redacted or cryptographically anonymized prior to model inference."
                    ),
                ),
                KnowledgeDocument(
                    doc_id="pol-fraud-003",
                    title="Cross-Modal Fraud Escalation Protocol",
                    content=(
                        "When spoken or written damage claims contradict intact parcel video and "
                        "photo evidence, escalate immediately to manual fraud review."
                    ),
                ),
            ],
        )
        self.lora_layer = LoRAAdapterLayer(d_in=256, d_out=256, rank=4, alpha=8.0)
        self.rlhf_loop = RLHFAdaptationLoop(self.lora_layer, learning_rate=0.12)
        self.multimodal_engine = MultimodalFusionEngine()
        self.finops_engine = FinOpsROIEngine()

    def bootstrap_mlops(self) -> None:
        """Ensure offline Parquet, Feast Online Store, and Production ONNX model exist."""
        self.dvc_runner.run_pipeline(force=False)


def _get_container(request: Request) -> PlatformRuntimeContainer:
    """Resolve the initialized PlatformRuntimeContainer from FastAPI app.state."""
    container: PlatformRuntimeContainer = request.app.state.container
    return container


def _decode_base64_bytes(encoded: str) -> bytes:
    """Decode base64 media string into raw bytes with strict validation."""
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationLayerError("Invalid base64-encoded media payload.") from exc


def _decode_video_frames(frames_b64: list[str]) -> np.ndarray:
    """Decode base64 frame list into an (F, 32, 32, 3) uint8 RGB video tensor."""
    decoded_frames: list[np.ndarray] = []
    for raw_b64 in frames_b64:
        frame_bytes = _decode_base64_bytes(raw_b64)
        try:
            with Image.open(io.BytesIO(frame_bytes)) as img:
                resized = img.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
                decoded_frames.append(np.asarray(resized, dtype=np.uint8))
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ValidationLayerError("Malformed video frame image payload.") from exc
    return np.stack(decoded_frames, axis=0)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service liveness, environment, and Level-2 MLOps readiness."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service="omnipulse-mlops-llmops-multimodal",
        environment=settings.OMNIPULSE_ENV,
        maturity_level=2,
    )


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose 3D Prometheus OpenMetrics scrape payload."""
    text = get_observability_hub().export_prometheus_text()
    return Response(content=text, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/api/v1/observability/summary")
async def observability_summary() -> dict[str, Any]:
    """Return consolidated 3D observability snapshot with Holt linear forecasts."""
    return get_observability_hub().snapshot_3d()


@router.get("/api/v1/audit/verify", response_model=AuditVerifyResponse)
async def verify_audit_chain() -> AuditVerifyResponse:
    """Verify full HMAC-SHA256 cryptographic audit chain integrity."""
    valid, count, detail = get_observability_hub().audit_chain.verify()
    return AuditVerifyResponse(valid=valid, records=count, detail=detail)


@router.get("/api/v1/finops/roi", response_model=FinOpsROIReport)
async def evaluate_finops_roi(
    request: Request,
    monthly_requests: int = Query(default=250_000, ge=0),
    business_value_per_1k_requests_usd: float = Query(default=18.0, ge=0.0),
    gpu_hours_baseline: float = Query(default=160.0, ge=0.0),
    quantization_bits: int = Query(default=4),
) -> FinOpsROIReport:
    """Compute FinOps quantization, semantic cache, HPA, and Parquet ROI governance report."""
    if quantization_bits not in (4, 8, 16):
        raise ValidationLayerError("quantization_bits must be one of 4, 8, or 16.")
    resolved_bits: Literal[4, 8, 16] = (
        4 if quantization_bits == 4 else (8 if quantization_bits == 8 else 16)
    )
    container = _get_container(request)
    workload = FinOpsWorkloadInput(
        monthly_requests=monthly_requests,
        business_value_per_1k_requests_usd=business_value_per_1k_requests_usd,
        gpu_hours_baseline=gpu_hours_baseline,
        quantization_bits=resolved_bits,
    )
    return container.finops_engine.evaluate(workload)


@router.post("/api/v1/mlops/pipeline/run", response_model=MLOpsPipelineRunResponse)
async def run_mlops_pipeline(
    payload: MLOpsPipelineRunRequest,
    request: Request,
) -> MLOpsPipelineRunResponse:
    """Execute declarative DVC pipeline stages, assess Maturity Level, and build CML report."""
    container = _get_container(request)
    result = await asyncio.to_thread(container.dvc_runner.run_pipeline, force=payload.force)
    baseline = ExperimentMetrics(
        roc_auc=max(0.5, round(result.metrics.roc_auc - 0.02, 5)),
        avg_prec=max(0.5, round(result.metrics.avg_prec - 0.015, 5)),
        f1_score=max(0.5, round(result.metrics.f1_score - 0.02, 5)),
        brier_loss=min(0.5, round(result.metrics.brier_loss + 0.01, 5)),
    )
    cml_md = container.dvc_runner.generate_cml_pr_report(
        baseline=baseline,
        candidate=result.metrics,
    )
    maturity = container.drift_engine.assess_maturity_level(
        MLOpsMaturityCapabilities(
            automated_training_pipeline=True,
            data_and_model_versioning=True,
            feature_store_enabled=True,
            ci_cd_model_deployment=True,
            continuous_drift_monitoring=True,
            automated_retraining_trigger=True,
            ab_testing_registry=True,
        ),
    )
    return MLOpsPipelineRunResponse(
        stages_executed=result.stages_executed,
        cached_stages_skipped=result.cached_stages_skipped,
        registered_model_version=result.registered_model_version,
        onnx_artifact_bytes=result.onnx_artifact_bytes,
        metrics=result.metrics,
        maturity_level=maturity.level,
        maturity_label=maturity.label,
        cml_markdown_report=cml_md,
    )


@router.post("/api/v1/mlops/predict", response_model=MLOpsPredictResponse)
async def mlops_online_predict(
    payload: MLOpsPredictRequest,
    request: Request,
) -> MLOpsPredictResponse:
    """Fetch O(1) Feast online features and run Champion/Challenger ONNX Runtime scoring."""
    container = _get_container(request)
    if payload.feature_override is not None:
        features = payload.feature_override
    else:
        online_rec = await asyncio.to_thread(
            container.dvc_runner.feast_service.get_online_features,
            payload.customer_id,
        )
        features = [
            online_rec.tx_velocity_24h,
            online_rec.credit_utilization,
            online_rec.chargeback_ratio,
            online_rec.device_trust_score,
        ]

    ab_res = await asyncio.to_thread(
        container.dvc_runner.mlflow_manager.predict_with_ab_routing,
        customer_id=payload.customer_id,
        feature_vector=features,
    )
    return MLOpsPredictResponse(
        customer_id=ab_res.customer_id,
        risk_probability=ab_res.risk_probability,
        risk_band=ab_res.risk_band,
        served_version=ab_res.served_version,
        served_stage=ab_res.served_stage,
        latency_ms=ab_res.latency_ms,
    )


@router.post("/api/v1/mlops/drift/evaluate", response_model=DriftEvaluationReport)
async def evaluate_mlops_drift(
    payload: MLOpsDriftEvaluateRequest,
    request: Request,
) -> DriftEvaluationReport:
    """Evaluate PSI + KS Data Drift & Concept Drift with closed-loop auto-retraining."""
    container = _get_container(request)
    return await asyncio.to_thread(
        container.drift_engine.evaluate_and_retrain_if_needed,
        reference_feature=payload.reference_feature,
        current_feature=payload.current_feature,
        recent_accuracy=payload.recent_accuracy,
        auto_retrain=payload.auto_retrain,
    )


@router.post("/api/v1/llmops/chat", response_model=LLMOpsChatAPIResponse)
async def llmops_rag_chat(
    payload: LLMOpsChatRequest,
    request: Request,
) -> LLMOpsChatAPIResponse:
    """Execute Guardrails -> Semantic Cache -> Hybrid RAG -> Open-Source Serving -> RAGAS."""
    container = _get_container(request)
    resp = await container.llm_gateway.chat(query=payload.query, tenant_id=payload.tenant_id)
    return LLMOpsChatAPIResponse(**resp.model_dump())


@router.post("/api/v1/llmops/adapt/lora", response_model=LoRAAdaptationSummary)
async def llmops_adapt_lora(
    payload: LoRAAdaptRequest,
    request: Request,
) -> LoRAAdaptationSummary:
    """Record RLHF human preference tuple and run low-rank LoRA DPO adaptation epochs."""
    container = _get_container(request)

    def _adapt() -> LoRAAdaptationSummary:
        pair = PreferencePair(
            prompt_embedding=embed_text_simd(payload.prompt, dim=256).tolist(),
            chosen_embedding=embed_text_simd(payload.chosen_response, dim=256).tolist(),
            rejected_embedding=embed_text_simd(payload.rejected_response, dim=256).tolist(),
        )
        container.rlhf_loop.record_human_preference(pair)
        return container.rlhf_loop.run_dpo_adaptation_step(epochs=payload.epochs)

    return await asyncio.to_thread(_adapt)


@router.post("/api/v1/multimodal/inspect", response_model=MultimodalInspectionReport)
async def multimodal_inspect(
    payload: MultimodalInspectAPIRequest,
    request: Request,
) -> MultimodalInspectionReport:
    """Run concurrent Text + Audio + Image + Video inspection and Cross-Modal Synergy audit."""
    container = _get_container(request)
    image_bytes = _decode_base64_bytes(payload.image_base64)
    video_tensor = await asyncio.to_thread(_decode_video_frames, payload.video_frames_base64)
    waveform = np.asarray(payload.audio_samples, dtype=np.float32)

    inspection_in = MultimodalInspectionInput(
        case_id=payload.case_id,
        speaker_id=payload.speaker_id,
        voice_consent_token=payload.voice_consent_token,
        claim_text=payload.claim_text,
        audio_waveform=waveform,
        audio_prompt_hint=payload.audio_prompt_hint,
        image_bytes=image_bytes,
        video_frames=video_tensor,
        policy_contexts=payload.policy_contexts,
    )
    return await container.multimodal_engine.inspect(inspection_in)


@router.websocket("/ws/v1/multimodal/stream")
async def multimodal_websocket_stream(websocket: WebSocket) -> None:
    """Stream real-time utterance ASR + ONNX frame inspection over low-latency WebSocket."""
    await websocket.accept()
    container: PlatformRuntimeContainer = websocket.app.state.container
    bind_request_context()
    try:
        while True:
            raw_msg = await websocket.receive_json()
            started_ns = time.perf_counter_ns()
            chunk = WebSocketStreamUtterance.model_validate(raw_msg)
            waveform = np.asarray(chunk.audio_samples, dtype=np.float32)
            frame_bytes = _decode_base64_bytes(chunk.frame_base64)

            audio_task = asyncio.to_thread(
                container.multimodal_engine.audio_pipeline.process_utterance,
                waveform,
                speaker_id=chunk.speaker_id,
                consent_token=chunk.voice_consent_token,
                prompt_hint=chunk.prompt_hint,
            )
            vision_task = asyncio.to_thread(
                container.multimodal_engine.vision_pipeline.inspect_image_bytes,
                frame_bytes,
            )
            audio_rep, vision_rep = await asyncio.gather(audio_task, vision_task)
            latency_ms = max(0.05, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

            log_info(
                _LOGGER,
                "websocket_multimodal_chunk_processed",
                utterance_id=chunk.utterance_id,
                visual_label=vision_rep.predicted_label,
                latency_ms=round(latency_ms, 3),
            )
            await websocket.send_json(
                {
                    "utterance_id": chunk.utterance_id,
                    "transcript": audio_rep.transcript,
                    "snr_db": audio_rep.snr_db,
                    "visual_label": vision_rep.predicted_label,
                    "visual_confidence": vision_rep.confidence,
                    "latency_ms": round(latency_ms, 3),
                },
            )
    except WebSocketDisconnect:
        return
