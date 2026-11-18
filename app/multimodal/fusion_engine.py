"""Cross-Modal Fusion, Sync Version Registry, Synergy Monitor & Multimodal RAGAS (Unit V.4)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.core.exceptions import ModalitySyncError
from app.core.observability import (
    get_observability_hub,
    log_event,
    log_tool_result,
    request_id_ctx,
)
from app.llmops.ragas_evaluator import RAGASEvaluator
from app.llmops.semantic_cache import embed_text_simd
from app.multimodal.audio_pipeline import AudioMLOpsPipeline, AudioUtteranceReport
from app.multimodal.vision_video_pipeline import (
    ImageInspectionReport,
    VideoInspectionReport,
    VisionVideoMLOpsPipeline,
)

_LOGGER = logging.getLogger(__name__)

_VISUAL_SEMANTIC_DESCRIPTORS: dict[str, str] = {
    "damaged_parcel": "damaged parcel crushed broken glass screen structural impact fracture",
    "intact_parcel": "intact pristine sealed undamaged clean parcel packaging box",
}


class MultimodalSyncManifest(BaseModel):
    """Synchronization metadata tying dataset & checkpoint versions across modalities."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(min_length=1)
    modality_timestamps: dict[str, float]
    modality_schemas: dict[str, str]
    fusion_backbone: str = Field(min_length=1)


class RegisteredMultimodalBundle(BaseModel):
    """Immutable versioned multimodal checkpoint and dataset synchronization entry."""

    model_config = ConfigDict(frozen=True)

    bundle_hash: str
    registered_at_utc: str
    max_observed_skew_seconds: float
    manifest: MultimodalSyncManifest


class MultimodalCheckpointRegistry:
    """Validates cross-modal temporal & schema synchronization before registering versions."""

    def __init__(self, *, max_timestamp_skew_seconds: float = 5.0) -> None:
        """Set maximum allowed capture skew across synchronized modalities."""
        self._max_skew = max(0.0, max_timestamp_skew_seconds)
        self._bundles: list[RegisteredMultimodalBundle] = []
        self._lock = threading.RLock()

    def register_bundle(self, manifest: MultimodalSyncManifest) -> RegisteredMultimodalBundle:
        """Verify timestamp & schema alignment and record a content-addressable bundle hash."""
        if len(manifest.modality_timestamps) < 2:
            raise ModalitySyncError(
                "Multimodal bundle requires at least 2 synchronized modalities."
            )
        if set(manifest.modality_timestamps) != set(manifest.modality_schemas):
            raise ModalitySyncError("Modality timestamp keys must match modality schema keys.")

        ts_vals = list(manifest.modality_timestamps.values())
        skew = max(ts_vals) - min(ts_vals)
        if skew > self._max_skew:
            raise ModalitySyncError(
                f"Cross-modal timestamp skew {skew:.2f}s exceeds limit {self._max_skew:.2f}s.",
            )

        canonical = json.dumps(
            manifest.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        entry = RegisteredMultimodalBundle(
            bundle_hash=digest,
            registered_at_utc=datetime.now(UTC).isoformat(),
            max_observed_skew_seconds=round(skew, 4),
            manifest=manifest,
        )
        with self._lock:
            self._bundles.append(entry)
        return entry

    def latest_bundle(self) -> RegisteredMultimodalBundle | None:
        """Return most recently registered multimodal checkpoint bundle."""
        with self._lock:
            return self._bundles[-1] if self._bundles else None


class MultimodalRAGASReport(BaseModel):
    """Semantic quality metrics across multimodal evidence grounding and synthesis."""

    model_config = ConfigDict(frozen=True)

    cross_modal_faithfulness: float
    modality_grounding_coverage: float
    answer_relevance: float
    overall_score: float


class MultimodalInspectionInput(BaseModel):
    """Omnichannel case payload combining Text, Audio waveform, Image bytes & Video stream."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    case_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    voice_consent_token: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    audio_waveform: np.ndarray
    audio_prompt_hint: str = ""
    image_bytes: bytes
    video_frames: np.ndarray
    policy_contexts: list[str] = Field(default_factory=list)


class MultimodalInspectionReport(BaseModel):
    """Consolidated report emitted by the concurrent Multimodal Fusion Engine."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    synthesized_summary: str
    recommended_decision: str
    cross_modal_synergy: float
    pairwise_modality_similarity: dict[str, float]
    contradiction_alert: bool
    modality_collapse_detected: bool
    audio_transcript: str
    audio_snr_db: float
    image_label: str
    video_dominant_label: str
    video_keyframes: list[int]
    multimodal_ragas: MultimodalRAGASReport
    latency_ms: float


class MultimodalFeedbackRecord(BaseModel):
    """Human reviewer annotation feeding continuous multimodal dataset curation."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    verified_outcome: str = Field(min_length=1)
    synergy_override: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class MultimodalFusionEngine:
    """Runs Text, Audio, Image & Video pipelines concurrently and audits Cross-Modal Synergy."""

    def __init__(
        self,
        *,
        audio_pipeline: AudioMLOpsPipeline | None = None,
        vision_pipeline: VisionVideoMLOpsPipeline | None = None,
        ragas_evaluator: RAGASEvaluator | None = None,
        min_synergy: float | None = None,
    ) -> None:
        """Wire modality sub-pipelines, RAGAS evaluator, and human curation queue."""
        self._audio = audio_pipeline or AudioMLOpsPipeline()
        self._vision = vision_pipeline or VisionVideoMLOpsPipeline()
        self._ragas = ragas_evaluator or RAGASEvaluator()
        self._min_synergy = (
            min_synergy
            if min_synergy is not None
            else get_settings().OMNIPULSE_MIN_CROSS_MODAL_SYNERGY
        )
        self._feedback_queue: list[MultimodalFeedbackRecord] = []
        self._lock = threading.RLock()

    async def inspect(self, payload: MultimodalInspectionInput) -> MultimodalInspectionReport:
        """Execute concurrent multimodal encoding, compute synergy matrix, and score RAGAS."""
        started_ns = time.perf_counter_ns()

        audio_task = asyncio.to_thread(
            self._audio.process_utterance,
            payload.audio_waveform,
            speaker_id=payload.speaker_id,
            consent_token=payload.voice_consent_token,
            prompt_hint=payload.audio_prompt_hint,
        )
        image_task = asyncio.to_thread(self._vision.inspect_image_bytes, payload.image_bytes)
        video_task = asyncio.to_thread(self._vision.inspect_video_stream, payload.video_frames)
        text_vec_task = asyncio.to_thread(embed_text_simd, payload.claim_text)

        audio_rep, img_rep, vid_rep, text_vec = await asyncio.gather(
            audio_task,
            image_task,
            video_task,
            text_vec_task,
        )

        return self._fuse_outputs(
            payload=payload,
            audio_rep=audio_rep,
            img_rep=img_rep,
            vid_rep=vid_rep,
            text_vec=text_vec,
            started_ns=started_ns,
        )

    def _fuse_outputs(
        self,
        *,
        payload: MultimodalInspectionInput,
        audio_rep: AudioUtteranceReport,
        img_rep: ImageInspectionReport,
        vid_rep: VideoInspectionReport,
        text_vec: np.ndarray,
        started_ns: int,
    ) -> MultimodalInspectionReport:
        """Project modality descriptors into shared space and evaluate synergy & RAGAS."""
        audio_vec = embed_text_simd(audio_rep.transcript)
        image_semantic = _VISUAL_SEMANTIC_DESCRIPTORS[img_rep.predicted_label]
        video_semantic = _VISUAL_SEMANTIC_DESCRIPTORS[vid_rep.dominant_label]
        image_vec = embed_text_simd(image_semantic)
        video_vec = embed_text_simd(video_semantic)

        names = ("text", "audio", "image", "video")
        mat = np.vstack([text_vec, audio_vec, image_vec, video_vec])
        sim_matrix = mat @ mat.T

        pairwise: dict[str, float] = {}
        off_diag: list[float] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = max(0.0, min(1.0, float(sim_matrix[i, j])))
                pairwise[f"{names[i]}_{names[j]}"] = round(score, 4)
                off_diag.append(score)

        synergy = round(sum(off_diag) / max(1, len(off_diag)), 6)
        modality_energies = np.linalg.norm(mat, axis=1)
        energy_share = modality_energies / max(1e-9, float(np.sum(modality_energies)))
        collapse_detected = bool(
            float(np.max(energy_share)) > 0.65 or np.min(modality_energies) < 0.1
        )
        contradiction = bool(synergy < self._min_synergy)

        if not contradiction and img_rep.predicted_label == "damaged_parcel":
            decision = "approve_expedited_claim"
        elif contradiction:
            decision = "escalate_manual_fraud_review"
        else:
            decision = "standard_policy_resolution"

        evidence_contexts = [
            payload.claim_text,
            audio_rep.transcript,
            image_semantic,
            video_semantic,
            *payload.policy_contexts,
        ]
        summary = (
            f"Text claim and Whisper transcript ({audio_rep.transcript}) evaluated alongside "
            f"image evidence ({image_semantic}) and video keyframes ({video_semantic})."
        )
        base_ragas = self._ragas.evaluate(
            query=payload.claim_text,
            retrieved_contexts=evidence_contexts,
            answer=summary,
        )
        covered_modalities = sum(
            1
            for phrase in (
                audio_rep.transcript.split()[0],
                image_semantic.split()[0],
                video_semantic.split()[0],
            )
            if phrase.lower() in summary.lower()
        )
        grounding_cov = round((covered_modalities + 1) / 4.0, 4)
        mm_overall = round(
            (base_ragas.faithfulness + grounding_cov + base_ragas.answer_relevance) / 3.0,
            4,
        )
        mm_ragas = MultimodalRAGASReport(
            cross_modal_faithfulness=base_ragas.faithfulness,
            modality_grounding_coverage=grounding_cov,
            answer_relevance=base_ragas.answer_relevance,
            overall_score=mm_overall,
        )

        latency_ms = max(0.05, (time.perf_counter_ns() - started_ns) / 1_000_000.0)
        hub = get_observability_hub()
        hub.record_inference("multimodal_fusion", latency_seconds=latency_ms / 1000.0, success=True)
        hub.record_model_quality(
            pillar="multimodal_fusion",
            metric_name="cross_modal_synergy",
            value=synergy,
        )
        hub.record_model_quality(
            pillar="multimodal_fusion",
            metric_name="multimodal_ragas",
            value=mm_overall,
        )
        payload_hash = hashlib.sha256(
            f"{payload.case_id}:{decision}:{synergy}".encode(),
        ).hexdigest()
        hub.audit_chain.append(
            request_id=request_id_ctx.get(),
            pillar="multimodal_fusion",
            action=decision,
            payload_hash=payload_hash,
            confidence=synergy,
        )
        log_tool_result(
            _LOGGER,
            "multimodal_fusion_inspect",
            success=True,
            latency_ms=latency_ms,
            case_id=payload.case_id,
            synergy=synergy,
            contradiction=contradiction,
        )

        return MultimodalInspectionReport(
            case_id=payload.case_id,
            synthesized_summary=summary,
            recommended_decision=decision,
            cross_modal_synergy=synergy,
            pairwise_modality_similarity=pairwise,
            contradiction_alert=contradiction,
            modality_collapse_detected=collapse_detected,
            audio_transcript=audio_rep.transcript,
            audio_snr_db=audio_rep.snr_db,
            image_label=img_rep.predicted_label,
            video_dominant_label=vid_rep.dominant_label,
            video_keyframes=vid_rep.keyframe_indices,
            multimodal_ragas=mm_ragas,
            latency_ms=round(latency_ms, 3),
        )

    def record_human_feedback(self, feedback: MultimodalFeedbackRecord) -> int:
        """Append verified human operator feedback to the multimodal dataset curation queue."""
        with self._lock:
            self._feedback_queue.append(feedback)
            queued = len(self._feedback_queue)
        get_observability_hub().record_model_quality(
            pillar="multimodal_fusion",
            metric_name="human_verified_synergy",
            value=feedback.synergy_override,
        )
        log_event(
            _LOGGER,
            "multimodal_feedback_recorded",
            case_id=feedback.case_id,
            verified_outcome=feedback.verified_outcome,
            queue_depth=queued,
        )
        return queued

    def pending_curation_records(self) -> list[MultimodalFeedbackRecord]:
        """Return a thread-safe copy of queued multimodal human curation records."""
        with self._lock:
            return list(self._feedback_queue)
