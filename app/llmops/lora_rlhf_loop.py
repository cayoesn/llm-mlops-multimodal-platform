"""Parameter-Efficient LoRA Rank-r Adapter & RLHF/DPO Preference Update Loop (Unit V.1)."""

from __future__ import annotations

import threading

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PreferencePair(BaseModel):
    """Human feedback preference pair (prompt, chosen response, rejected response)."""

    model_config = ConfigDict(frozen=True)

    prompt_embedding: list[float] = Field(min_length=4)
    chosen_embedding: list[float] = Field(min_length=4)
    rejected_embedding: list[float] = Field(min_length=4)


class LoRAAdaptationSummary(BaseModel):
    """Telemetry report produced after an incremental LoRA DPO/RLHF update step."""

    model_config = ConfigDict(frozen=True)

    rank: int
    trainable_parameters: int
    frozen_base_parameters: int
    trainable_parameter_ratio: float
    pairs_evaluated: int
    initial_margin: float
    post_margin: float
    preference_accuracy: float


class LoRAAdapterLayer:
    """Low-Rank Adaptation layer W_eff = W_0 + (alpha / r) * (B @ A) with frozen W_0."""

    def __init__(
        self,
        *,
        d_in: int = 64,
        d_out: int = 64,
        rank: int = 4,
        alpha: float = 8.0,
        seed: int = 42,
    ) -> None:
        """Initialize frozen base matrix W_0, Gaussian A, and zero-initialized B."""
        if rank < 1 or rank >= min(d_in, d_out):
            raise ValueError("LoRA rank must satisfy 1 <= rank < min(d_in, d_out).")
        rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.scaling = alpha / float(rank)

        w0 = rng.normal(0.0, 0.05, size=(d_out, d_in)).astype(np.float32)
        w0.flags.writeable = False
        self.base_weights = w0

        self.lora_a = rng.normal(0.0, 0.02, size=(rank, d_in)).astype(np.float32)
        self.lora_b = np.zeros((d_out, rank), dtype=np.float32)
        self._lock = threading.RLock()

    @property
    def trainable_parameters(self) -> int:
        """Return count of trainable parameters in low-rank matrices A and B."""
        return int(self.lora_a.size + self.lora_b.size)

    @property
    def frozen_base_parameters(self) -> int:
        """Return count of frozen pre-trained parameters in W_0."""
        return int(self.base_weights.size)

    @property
    def trainable_parameter_ratio(self) -> float:
        """Return ratio of LoRA trainable parameters to frozen base parameters."""
        return round(self.trainable_parameters / max(1, self.frozen_base_parameters), 6)

    @property
    def adapter_lock(self) -> threading.RLock:
        """Expose re-entrant adapter lock for synchronized RLHF/DPO parameter updates."""
        return self._lock

    def project(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass y = W_0 x + (alpha / r) B (A x)."""
        with self._lock:
            base_out: np.ndarray = self.base_weights @ x
            delta_out: np.ndarray = self.scaling * (self.lora_b @ (self.lora_a @ x))
            projected: np.ndarray = np.asarray(base_out + delta_out, dtype=np.float32)
            return projected


class RLHFAdaptationLoop:
    """Curates human preference pairs and updates LoRA matrices via Direct Preference margin."""

    def __init__(self, adapter: LoRAAdapterLayer, learning_rate: float = 0.05) -> None:
        """Bind LoRA adapter and step size."""
        self._adapter = adapter
        self._lr = learning_rate
        self._buffer: list[PreferencePair] = []
        self._lock = threading.RLock()

    def record_human_preference(self, pair: PreferencePair) -> int:
        """Append a validated human feedback preference tuple to the RLHF buffer."""
        if (
            len(pair.prompt_embedding) != self._adapter.d_in
            or len(pair.chosen_embedding) != self._adapter.d_out
            or len(pair.rejected_embedding) != self._adapter.d_out
        ):
            raise ValueError("PreferencePair embedding dimensions must match LoRAAdapterLayer.")
        with self._lock:
            self._buffer.append(pair)
            return len(self._buffer)

    def _mean_margin(
        self,
        prompts: np.ndarray,
        diffs: np.ndarray,
    ) -> tuple[float, float]:
        """Compute average preference reward margin and win-rate accuracy."""
        margins: list[float] = []
        wins = 0
        for p_vec, diff_vec in zip(prompts, diffs, strict=True):
            projected = self._adapter.project(p_vec)
            margin = float(np.dot(projected, diff_vec))
            margins.append(margin)
            if margin > 0.0:
                wins += 1
        return sum(margins) / len(margins), wins / len(margins)

    def run_dpo_adaptation_step(self, epochs: int = 10) -> LoRAAdaptationSummary:
        """Optimize LoRA A & B on buffered RLHF pairs keeping W_0 strictly frozen."""
        with self._lock, self._adapter.adapter_lock:
            if not self._buffer:
                raise ValueError("No human preference pairs recorded in RLHF buffer.")
            snapshot = list(self._buffer)

            prompts = np.asarray([p.prompt_embedding for p in snapshot], dtype=np.float32)
            chosen = np.asarray([p.chosen_embedding for p in snapshot], dtype=np.float32)
            rejected = np.asarray([p.rejected_embedding for p in snapshot], dtype=np.float32)
            diffs = chosen - rejected

            initial_margin, _ = self._mean_margin(prompts, diffs)

            for _ in range(max(1, epochs)):
                for p_vec, diff_vec in zip(prompts, diffs, strict=True):
                    projected = self._adapter.project(p_vec)
                    margin = float(np.dot(projected, diff_vec))
                    sigmoid_weight = float(1.0 / (1.0 + np.exp(np.clip(margin, -20.0, 20.0))))
                    scale = self._lr * sigmoid_weight * self._adapter.scaling

                    hidden = self._adapter.lora_a @ p_vec
                    grad_b = np.outer(diff_vec, hidden)
                    grad_a = np.outer(self._adapter.lora_b.T @ diff_vec, p_vec)

                    self._adapter.lora_b += scale * grad_b
                    self._adapter.lora_a += scale * grad_a

            post_margin, acc = self._mean_margin(prompts, diffs)
            return LoRAAdaptationSummary(
                rank=self._adapter.rank,
                trainable_parameters=self._adapter.trainable_parameters,
                frozen_base_parameters=self._adapter.frozen_base_parameters,
                trainable_parameter_ratio=self._adapter.trainable_parameter_ratio,
                pairs_evaluated=len(snapshot),
                initial_margin=round(initial_margin, 5),
                post_margin=round(post_margin, 5),
                preference_accuracy=round(acc, 5),
            )
