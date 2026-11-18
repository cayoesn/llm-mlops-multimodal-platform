"""TDD tests for Pillar 2 RAGAS Evaluation Suite and LoRA + RLHF Continuous Adaptation Loop."""

from __future__ import annotations

import numpy as np

from app.llmops.lora_rlhf_loop import LoRAAdapterLayer, PreferencePair, RLHFAdaptationLoop
from app.llmops.ragas_evaluator import RAGASEvaluator


def test_ragas_evaluator_distinguishes_faithful_vs_hallucinated_answers() -> None:
    """RAGASEvaluator scores grounded answers high and penalizes ungrounded hallucinations."""
    evaluator = RAGASEvaluator()
    contexts = [
        "Chargeback disputes must be filed within 7 business days with signed delivery proof.",
        "Refunds for verified fraud are credited within 48 hours to the customer account.",
    ]

    grounded = evaluator.evaluate(
        query="When must chargeback disputes be filed and how fast are fraud refunds?",
        retrieved_contexts=contexts,
        answer=(
            "Chargeback disputes must be filed within 7 business days with signed delivery proof. "
            "Refunds for verified fraud are credited within 48 hours."
        ),
    )
    assert grounded.faithfulness >= 0.85
    assert grounded.contextual_relevance >= 0.60
    assert grounded.answer_relevance >= 0.60
    assert grounded.passed_quality_gate is True

    hallucinated = evaluator.evaluate(
        query="When must chargeback disputes be filed?",
        retrieved_contexts=contexts,
        answer=(
            "Customers receive automatic cryptocurrency airdrops after 90 lunar cycles "
            "without submitting any documentation."
        ),
    )
    assert hallucinated.faithfulness < 0.35
    assert hallucinated.passed_quality_gate is False


def test_lora_adapter_freezes_base_weights_and_improves_rlhf_preference_margin() -> None:
    """LoRA rank-r adapter trains <3% parameters and strictly increases RLHF preference margin."""
    lora = LoRAAdapterLayer(d_in=64, d_out=64, rank=4, alpha=8.0, seed=42)
    base_snapshot = lora.base_weights.copy()
    assert lora.trainable_parameter_ratio < 0.15

    loop = RLHFAdaptationLoop(adapter=lora, learning_rate=0.08)
    rng = np.random.default_rng(7)
    prompt_vec = rng.normal(0.0, 1.0, size=64).astype(np.float32)
    chosen_vec = rng.normal(0.5, 1.0, size=64).astype(np.float32)
    rejected_vec = -chosen_vec

    pair = PreferencePair(
        prompt_embedding=prompt_vec.tolist(),
        chosen_embedding=chosen_vec.tolist(),
        rejected_embedding=rejected_vec.tolist(),
    )
    loop.record_human_preference(pair)
    summary = loop.run_dpo_adaptation_step(epochs=12)

    assert np.array_equal(lora.base_weights, base_snapshot)
    assert summary.post_margin > summary.initial_margin
    assert summary.preference_accuracy == 1.0
