"""Immutable Pydantic v2 schemas for the 5-Layer REST & WebSocket API (Unit IV)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llmops.ragas_evaluator import RAGASMetricsReport
from app.mlops.mlflow_lifecycle import ExperimentMetrics


class HealthResponse(BaseModel):
    """Liveness and readiness payload."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy"]
    service: str
    environment: str
    maturity_level: int


class AuditVerifyResponse(BaseModel):
    """Cryptographic HMAC-SHA256 audit verification response."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    records: int
    detail: str


class MLOpsPipelineRunRequest(BaseModel):
    """Request to trigger or verify the 6-stage DVC + Feast + MLflow pipeline."""

    model_config = ConfigDict(frozen=True)

    force: bool = False


class MLOpsPipelineRunResponse(BaseModel):
    """Summary of DVC execution, CML Markdown PR table, and Maturity Level assessment."""

    model_config = ConfigDict(frozen=True)

    stages_executed: int
    cached_stages_skipped: int
    registered_model_version: int
    onnx_artifact_bytes: int
    metrics: ExperimentMetrics
    maturity_level: int
    maturity_label: str
    cml_markdown_report: str


class MLOpsPredictRequest(BaseModel):
    """Real-time risk inference request resolved via Feast Online Store."""

    model_config = ConfigDict(frozen=True)

    customer_id: str = Field(min_length=1)
    feature_override: list[float] | None = None


class MLOpsPredictResponse(BaseModel):
    """Champion/Challenger A/B risk prediction response."""

    model_config = ConfigDict(frozen=True)

    customer_id: str
    risk_probability: float
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    served_version: int
    served_stage: str
    latency_ms: float


class MLOpsDriftEvaluateRequest(BaseModel):
    """Statistical PSI/KS data drift and live accuracy concept drift evaluation request."""

    model_config = ConfigDict(frozen=True)

    reference_feature: list[float] = Field(min_length=10)
    current_feature: list[float] = Field(min_length=10)
    recent_accuracy: float = Field(ge=0.0, le=1.0)
    auto_retrain: bool = True


class LLMOpsChatRequest(BaseModel):
    """Enterprise policy RAG copilot request."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=8000)
    tenant_id: str = Field(default="default", min_length=1)


class LLMOpsChatAPIResponse(BaseModel):
    """Grounded RAG response with cache, guardrail, and inline RAGAS metrics."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[str]
    cache_hit: bool
    cache_hit_type: str | None
    pii_entities_redacted: int
    toxicity_score: float
    prompt_version_hash: str
    ragas_metrics: RAGASMetricsReport
    latency_ms: float


class LoRAAdaptRequest(BaseModel):
    """Human preference pair submission for continuous LoRA/RLHF parameter-efficient updates."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    chosen_response: str = Field(min_length=1)
    rejected_response: str = Field(min_length=1)
    epochs: int = Field(default=6, ge=1, le=50)


class MultimodalInspectAPIRequest(BaseModel):
    """Omnichannel REST payload carrying Text, Audio waveform, Image, and Video keyframes."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    voice_consent_token: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    audio_samples: list[float] = Field(min_length=64)
    audio_prompt_hint: str = ""
    image_base64: str = Field(min_length=8)
    video_frames_base64: list[str] = Field(min_length=1)
    policy_contexts: list[str] = Field(default_factory=list)


class WebSocketStreamUtterance(BaseModel):
    """Low-latency WebSocket multimodal chunk carrying live speech and camera frame."""

    model_config = ConfigDict(frozen=True)

    utterance_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    voice_consent_token: str = Field(min_length=1)
    audio_samples: list[float] = Field(min_length=64)
    prompt_hint: str = ""
    frame_base64: str = Field(min_length=8)
