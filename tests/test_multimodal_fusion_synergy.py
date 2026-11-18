"""TDD suite for Pillar 3 Cross-Modal Fusion, Version Sync, Synergy Monitor & Multimodal RAGAS."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core.exceptions import ModalitySyncError
from app.core.observability import get_observability_hub
from app.multimodal.audio_pipeline import issue_voice_consent_token
from app.multimodal.fusion_engine import (
    MultimodalCheckpointRegistry,
    MultimodalFeedbackRecord,
    MultimodalFusionEngine,
    MultimodalInspectionInput,
    MultimodalSyncManifest,
)


def _make_png(damaged: bool) -> bytes:
    """Generate deterministic damaged vs intact RGB image bytes."""
    rng = np.random.default_rng(19 if damaged else 23)
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    if damaged:
        arr[:, :, 0] = rng.integers(190, 255, size=(32, 32), dtype=np.uint8)
        arr[8:24, 8:24, :] = 5
    else:
        arr[:, :, 1] = rng.integers(175, 225, size=(32, 32), dtype=np.uint8)
        arr[:, :, 2] = rng.integers(180, 235, size=(32, 32), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _make_video(damaged: bool) -> np.ndarray:
    """Generate an 8-frame 32x32x3 video array matching damaged or intact state."""
    rng = np.random.default_rng(31 if damaged else 37)
    frames: list[np.ndarray] = []
    for _ in range(8):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        if damaged:
            arr[:, :, 0] = rng.integers(185, 250, size=(32, 32), dtype=np.uint8)
            arr[8:24, 8:24, :] = 6
        else:
            arr[:, :, 1] = rng.integers(175, 225, size=(32, 32), dtype=np.uint8)
            arr[:, :, 2] = rng.integers(180, 235, size=(32, 32), dtype=np.uint8)
        frames.append(arr)
    return np.stack(frames, axis=0)


def _make_speech_wave() -> np.ndarray:
    """Generate a clean 16kHz synthetic speech burst."""
    sr = 16_000
    t = np.linspace(0.0, 0.8, int(sr * 0.8), endpoint=False, dtype=np.float32)
    wave = 0.02 * np.sin(2.0 * np.pi * 60.0 * t)
    active = slice(int(0.15 * sr), int(0.65 * sr))
    wave[active] += 0.55 * np.sin(2.0 * np.pi * 240.0 * t[active])
    return np.clip(wave, -1.0, 1.0).astype(np.float32)


def test_multimodal_version_registry_enforces_timestamp_and_schema_sync() -> None:
    """Ensure multimodal checkpoints reject desynchronized modality capture timestamps."""
    registry = MultimodalCheckpointRegistry(max_timestamp_skew_seconds=5.0)

    valid_manifest = MultimodalSyncManifest(
        dataset_name="omnipulse_dispute_evidence_v1",
        modality_timestamps={
            "text": 1_700_000_000.0,
            "audio": 1_700_000_001.5,
            "image": 1_700_000_002.2,
            "video": 1_700_000_003.0,
        },
        modality_schemas={
            "text": "utf8_claim_v1",
            "audio": "pcm16k_mono_v1",
            "image": "rgb_png_v1",
            "video": "rgb_frames_v1",
        },
        fusion_backbone="omnipulse-crossmodal-onnx-int8",
    )
    entry = registry.register_bundle(valid_manifest)
    assert len(entry.bundle_hash) == 64
    assert registry.latest_bundle() == entry

    skewed_manifest = MultimodalSyncManifest(
        dataset_name="omnipulse_dispute_evidence_v1",
        modality_timestamps={
            "text": 1_700_000_000.0,
            "audio": 1_700_000_180.0,
            "image": 1_700_000_001.0,
        },
        modality_schemas={
            "text": "utf8_claim_v1",
            "audio": "pcm16k_mono_v1",
            "image": "rgb_png_v1",
        },
        fusion_backbone="omnipulse-crossmodal-onnx-int8",
    )
    with pytest.raises(ModalitySyncError):
        registry.register_bundle(skewed_manifest)


@pytest.mark.asyncio
async def test_concurrent_multimodal_fusion_detects_synergy_vs_cross_modal_contradiction() -> None:
    """Verify asyncio.gather multimodal inspection separates consistent vs contradictory claims."""
    engine = MultimodalFusionEngine()
    consent = issue_voice_consent_token("cust-901")

    consistent_payload = MultimodalInspectionInput(
        case_id="case-consistent-001",
        speaker_id="cust-901",
        voice_consent_token=consent,
        claim_text="Damaged parcel arrived crushed with broken glass screen and structural impact.",
        audio_waveform=_make_speech_wave(),
        audio_prompt_hint="damaged parcel crushed broken glass screen structural impact",
        image_bytes=_make_png(damaged=True),
        video_frames=_make_video(damaged=True),
        policy_contexts=[
            (
                "Damaged parcel claims with crushed packaging or broken screen "
                "qualify for instant refund."
            ),
        ],
    )
    consistent_report = await engine.inspect(consistent_payload)
    assert consistent_report.cross_modal_synergy >= 0.65
    assert consistent_report.contradiction_alert is False
    assert consistent_report.modality_collapse_detected is False
    assert consistent_report.multimodal_ragas.overall_score >= 0.65
    assert consistent_report.recommended_decision == "approve_expedited_claim"

    contradictory_payload = MultimodalInspectionInput(
        case_id="case-contradict-002",
        speaker_id="cust-901",
        voice_consent_token=consent,
        claim_text="Damaged parcel arrived crushed with broken glass screen and structural impact.",
        audio_waveform=_make_speech_wave(),
        audio_prompt_hint="damaged parcel crushed broken glass screen structural impact",
        image_bytes=_make_png(damaged=False),
        video_frames=_make_video(damaged=False),
        policy_contexts=[
            (
                "Intact sealed parcel claims contradicting visual damage evidence "
                "require fraud review."
            ),
        ],
    )
    contradictory_report = await engine.inspect(contradictory_payload)
    assert contradictory_report.cross_modal_synergy < 0.50
    assert contradictory_report.contradiction_alert is True
    assert contradictory_report.recommended_decision == "escalate_manual_fraud_review"

    valid_chain, count, _ = get_observability_hub().audit_chain.verify()
    assert valid_chain is True and count >= 2


def test_multimodal_human_feedback_loop_updates_curation_buffer() -> None:
    """Ensure human operator reviews feed back into the multimodal retraining queue."""
    engine = MultimodalFusionEngine()
    queued = engine.record_human_feedback(
        MultimodalFeedbackRecord(
            case_id="case-contradict-002",
            reviewer_id="auditor-17",
            verified_outcome="confirmed_visual_contradiction",
            synergy_override=0.22,
            notes="Customer uploaded intact box photo despite claiming crushed box on call.",
        ),
    )
    assert queued == 1
    assert len(engine.pending_curation_records()) == 1
