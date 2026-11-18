"""TDD suite for Pillar 3 Audio DSP/Whisper/WER/LGPD and Vision/Video ONNX/mAP/Drift pipelines."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core.exceptions import GuardrailViolationError, ModalitySyncError, ValidationLayerError
from app.multimodal.audio_pipeline import (
    AudioMLOpsPipeline,
    compute_snr_db,
    compute_word_error_rate,
    issue_voice_consent_token,
)
from app.multimodal.vision_video_pipeline import (
    BoundingBoxDetection,
    VisionVideoMLOpsPipeline,
    compute_iou_matrix,
    compute_mean_average_precision,
)


def _synthesize_utterance_waveform(
    sample_rate: int = 16_000,
    noise_std: float = 0.04,
) -> np.ndarray:
    """Create a 1.0-second 16kHz signal with silence margins and speech-like harmonic bursts."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    waveform = rng.normal(0.0, noise_std, size=sample_rate).astype(np.float32)

    active = slice(int(0.20 * sample_rate), int(0.80 * sample_rate))
    carrier = (
        0.55 * np.sin(2.0 * np.pi * 220.0 * t[active])
        + 0.30 * np.sin(2.0 * np.pi * 440.0 * t[active])
        + 0.15 * np.sin(2.0 * np.pi * 660.0 * t[active])
    )
    waveform[active] += carrier.astype(np.float32)
    return np.clip(waveform, -1.0, 1.0)


def _encode_png_bytes(rgb_array: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as in-memory PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(rgb_array, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _make_parcel_frame(*, damaged: bool, seed: int = 7) -> np.ndarray:
    """Synthesize deterministic intact vs damaged RGB 32x32 frames."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    if damaged:
        frame[:, :, 0] = rng.integers(180, 255, size=(32, 32), dtype=np.uint8)
        frame[:, :, 1] = rng.integers(10, 45, size=(32, 32), dtype=np.uint8)
        frame[:, :, 2] = rng.integers(10, 45, size=(32, 32), dtype=np.uint8)
        frame[8:24, 8:24, :] = 5
    else:
        frame[:, :, 0] = rng.integers(30, 60, size=(32, 32), dtype=np.uint8)
        frame[:, :, 1] = rng.integers(170, 220, size=(32, 32), dtype=np.uint8)
        frame[:, :, 2] = rng.integers(180, 230, size=(32, 32), dtype=np.uint8)
    return frame


def test_audio_dsp_denoise_vad_snr_whisper_wer_and_lgpd_governance() -> None:
    """Verify spectral gating, VAD, SNR gain, Whisper latency, DP WER, and LGPD voice privacy."""
    pipeline = AudioMLOpsPipeline(sample_rate=16_000)
    raw_wave = _synthesize_utterance_waveform()

    with pytest.raises(GuardrailViolationError):
        pipeline.process_utterance(
            raw_wave,
            speaker_id="cust-882",
            consent_token="invalid-forged-token",
            prompt_hint="damaged screen claim",
        )

    valid_token = issue_voice_consent_token("cust-882")
    result = pipeline.process_utterance(
        raw_wave,
        speaker_id="cust-882",
        consent_token=valid_token,
        prompt_hint="crushed box broken glass screen",
        reference_transcript="customer reports crushed box broken glass screen",
    )

    raw_snr = compute_snr_db(raw_wave, result.vad_mask)
    assert result.snr_db > raw_snr
    assert 0.35 <= result.speech_activity_ratio <= 0.85
    assert result.latency_ms > 0.0
    assert "broken glass screen" in result.transcript
    assert result.wer is not None and 0.0 <= result.wer <= 0.30
    assert result.biometric_anonymized is True
    assert not np.allclose(raw_wave, result.anonymized_waveform)
    assert len(pipeline.recent_observability_samples()) == 1

    assert compute_word_error_rate("exact match test", "exact match test") == 0.0
    assert compute_word_error_rate(
        "four word reference sentence", "four word changed"
    ) == pytest.approx(
        0.5,
    )
    assert compute_word_error_rate("", "non empty") == 1.0


def test_image_header_bomb_guard_preprocessing_and_augmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure lazy PIL header checks block oversized images before pixel allocation."""
    pipeline = VisionVideoMLOpsPipeline()

    with pytest.raises(ValidationLayerError):
        pipeline.inspect_image_bytes(b"not-an-image-payload")

    oversized = np.zeros((96, 96, 3), dtype=np.uint8)
    oversized_bytes = _encode_png_bytes(oversized)
    monkeypatch.setattr(pipeline, "_max_side", 64)
    with pytest.raises(ModalitySyncError):
        pipeline.inspect_image_bytes(oversized_bytes)

    monkeypatch.setattr(pipeline, "_max_side", 4096)
    valid_bytes = _encode_png_bytes(_make_parcel_frame(damaged=True))
    tensor, aug_views = pipeline.preprocess_and_augment(valid_bytes)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == np.float32
    assert len(aug_views) == 2
    assert not np.allclose(aug_views[0], aug_views[1])


def test_video_keyframe_extraction_onnx_runtime_inference_map_and_visual_drift() -> None:
    """Verify video scene keyframes, real ONNX Runtime session, IoU/mAP, and JS Visual Drift."""
    pipeline = VisionVideoMLOpsPipeline()

    frames = [_make_parcel_frame(damaged=(idx >= 5), seed=100 + idx) for idx in range(12)]
    video_array = np.stack(frames, axis=0)

    video_report = pipeline.inspect_video_stream(
        video_array,
        uniform_stride=4,
        scene_delta_threshold=0.18,
    )
    assert 0 in video_report.keyframe_indices
    assert 5 in video_report.keyframe_indices
    assert video_report.compression_ratio >= 2.0
    assert video_report.dominant_label == "damaged_parcel"
    assert video_report.onnx_quantization_profile.compression_ratio >= 3.5
    assert video_report.onnx_latency_ms > 0.0

    pristine_img = _encode_png_bytes(_make_parcel_frame(damaged=False, seed=21))
    img_report = pipeline.inspect_image_bytes(pristine_img)
    assert img_report.predicted_label == "intact_parcel"
    assert img_report.confidence > 0.55

    gt_boxes = [
        BoundingBoxDetection(label="damage", confidence=1.0, x1=10.0, y1=10.0, x2=50.0, y2=50.0),
    ]
    accurate_preds = [
        BoundingBoxDetection(label="damage", confidence=0.94, x1=12.0, y1=11.0, x2=49.0, y2=51.0),
    ]
    disjoint_preds = [
        BoundingBoxDetection(
            label="damage", confidence=0.90, x1=120.0, y1=120.0, x2=160.0, y2=160.0
        ),
    ]
    iou_mat = compute_iou_matrix(accurate_preds, gt_boxes)
    assert iou_mat.shape == (1, 1) and float(iou_mat[0, 0]) > 0.75
    assert compute_mean_average_precision(accurate_preds, gt_boxes) == pytest.approx(1.0)
    assert compute_mean_average_precision(disjoint_preds, gt_boxes) == pytest.approx(0.0)

    ref_batch = np.stack([_make_parcel_frame(damaged=False, seed=s) for s in range(6)], axis=0)
    stable_batch = np.stack(
        [_make_parcel_frame(damaged=False, seed=50 + s) for s in range(6)],
        axis=0,
    )
    shifted_batch = np.stack(
        [_make_parcel_frame(damaged=True, seed=90 + s) for s in range(6)],
        axis=0,
    )

    stable_drift = pipeline.evaluate_visual_drift(ref_batch, stable_batch)
    shifted_drift = pipeline.evaluate_visual_drift(ref_batch, shifted_batch)
    assert stable_drift.drift_detected is False
    assert shifted_drift.drift_detected is True
    assert shifted_drift.jensen_shannon_divergence > stable_drift.jensen_shannon_divergence
