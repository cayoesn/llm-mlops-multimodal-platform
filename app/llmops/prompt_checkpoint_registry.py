"""Versioned Prompt Registry, Context Compression & Quantized Checkpoint Sharding Planner."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ValidationLayerError

QuantizationMode = Literal["FP16", "INT8", "INT4"]

_BYTES_PER_PARAM: dict[QuantizationMode, float] = {
    "FP16": 2.0,
    "INT8": 1.0,
    "INT4": 0.55,
}


class VersionedPromptTemplate(BaseModel):
    """Immutable SHA-256 versioned prompt template registered for LLMOps reproducibility."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    version_hash: str
    system_template: str
    max_context_tokens: int
    created_at_utc: str


class QuantizedCheckpointSpec(BaseModel):
    """Hardware and model checkpoint descriptor for LLM serving capacity planning."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(min_length=1)
    parameter_billions: float = Field(gt=0.0)
    quantization: QuantizationMode = "INT4"
    context_length: int = Field(default=8192, ge=512)
    gpu_vram_gb_per_device: float = Field(default=24.0, gt=0.0)


class CheckpointDeploymentPlan(BaseModel):
    """Calculated VRAM footprint, KV-cache overhead, and tensor-parallel GPU shard count."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    quantization: QuantizationMode
    estimated_weights_gb: float
    kv_cache_overhead_gb: float
    total_required_vram_gb: float
    recommended_gpu_shards: int
    vram_savings_vs_fp16_pct: float


class PromptCheckpointRegistry:
    """Thread-safe prompt version store, token budget trimmer, and GPU shard planner."""

    def __init__(self) -> None:
        """Initialize RLock-protected prompt version catalog."""
        self._lock = threading.RLock()
        self._prompts: dict[str, list[VersionedPromptTemplate]] = {}

    def register_prompt(
        self,
        *,
        name: str,
        system_template: str,
        max_context_tokens: int = 2048,
    ) -> VersionedPromptTemplate:
        """Register an immutable SHA-256 hashed prompt version."""
        if not name.strip() or not system_template.strip():
            raise ValidationLayerError("Prompt name and system_template must be non-empty.")
        if max_context_tokens < 8:
            raise ValidationLayerError("max_context_tokens must be >= 8.")

        canonical = json.dumps(
            [name.strip(), system_template.strip(), max_context_tokens],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()

        with self._lock:
            history = self._prompts.setdefault(name.strip(), [])
            version_num = len(history) + 1
            record = VersionedPromptTemplate(
                name=name.strip(),
                version=version_num,
                version_hash=digest,
                system_template=system_template.strip(),
                max_context_tokens=max_context_tokens,
                created_at_utc=datetime.now(UTC).isoformat(),
            )
            history.append(record)
            return record

    def get_latest_prompt(self, name: str) -> VersionedPromptTemplate:
        """Return the active latest version for prompt name."""
        with self._lock:
            history = self._prompts.get(name)
            if not history:
                raise ValidationLayerError(f"Prompt '{name}' is not registered.")
            return history[-1]

    def fit_context_budget(self, prompt_name: str, ranked_chunks: list[str]) -> str:
        """Pack highest-priority retrieved context chunks within max_context_tokens in O(L)."""
        spec = self.get_latest_prompt(prompt_name)
        budget = spec.max_context_tokens
        accepted_words: list[str] = []
        for chunk in ranked_chunks:
            words = chunk.split()
            remaining = budget - len(accepted_words)
            if remaining <= 0:
                break
            accepted_words.extend(words[:remaining])
        return " ".join(accepted_words)

    @staticmethod
    def plan_checkpoint_deployment(spec: QuantizedCheckpointSpec) -> CheckpointDeploymentPlan:
        """Calculate quantized model VRAM requirements and tensor-parallel GPU sharding."""
        fp16_weights_gb = spec.parameter_billions * _BYTES_PER_PARAM["FP16"]
        quant_weights_gb = spec.parameter_billions * _BYTES_PER_PARAM[spec.quantization]
        kv_cache_gb = (spec.context_length / 4096.0) * (spec.parameter_billions * 0.04)
        total_vram_gb = (quant_weights_gb + kv_cache_gb) * 1.12
        usable_per_gpu = spec.gpu_vram_gb_per_device * 0.88
        shards = max(1, math.ceil(total_vram_gb / usable_per_gpu))
        savings_pct = (
            ((fp16_weights_gb - quant_weights_gb) / fp16_weights_gb) * 100.0
            if fp16_weights_gb > 0.0
            else 0.0
        )
        return CheckpointDeploymentPlan(
            model_id=spec.model_id,
            quantization=spec.quantization,
            estimated_weights_gb=round(quant_weights_gb, 3),
            kv_cache_overhead_gb=round(kv_cache_gb, 3),
            total_required_vram_gb=round(total_vram_gb, 3),
            recommended_gpu_shards=shards,
            vram_savings_vs_fp16_pct=round(savings_pct, 3),
        )
