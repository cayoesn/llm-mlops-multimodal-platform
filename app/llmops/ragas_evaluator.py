"""RAGAS Semantic Evaluation Suite: Faithfulness, Contextual & Answer Relevance (Unit V.1)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.core.observability import get_observability_hub
from app.llmops.semantic_cache import embed_text_simd

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_CONTENT_WORD_RE = re.compile(r"[A-Za-z0-9]{3,}")
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "are",
        "from",
        "must",
        "within",
        "when",
        "how",
        "what",
    }
)


def _content_tokens(text: str) -> set[str]:
    """Extract informative lowercase content tokens of length >= 3."""
    return {t.lower() for t in _CONTENT_WORD_RE.findall(text) if t.lower() not in _STOP}


class RAGASMetricsReport(BaseModel):
    """Immutable RAGAS assessment scores for RAG pipeline quality gating."""

    model_config = ConfigDict(frozen=True)

    faithfulness: float
    contextual_relevance: float
    contextual_precision: float
    answer_relevance: float
    overall_ragas_score: float
    passed_quality_gate: bool


class RAGASEvaluator:
    """Computes atomic-claim Faithfulness, Contextual Relevance/Precision & Answer Relevance."""

    def __init__(self, min_faithfulness: float | None = None) -> None:
        """Bind minimum faithfulness gate."""
        self._min_faithfulness = (
            min_faithfulness
            if min_faithfulness is not None
            else get_settings().OMNIPULSE_MIN_RAGAS_FAITHFULNESS
        )

    def evaluate(
        self,
        *,
        query: str,
        retrieved_contexts: Sequence[str],
        answer: str,
    ) -> RAGASMetricsReport:
        """Evaluate generated RAG answer against retrieved contexts and user query."""
        claims = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
        context_union = _content_tokens(" ".join(retrieved_contexts))

        if not claims or not context_union:
            faithfulness = 0.0
        else:
            claim_scores: list[float] = []
            for claim in claims:
                c_tokens = _content_tokens(claim)
                if not c_tokens:
                    continue
                overlap = len(c_tokens & context_union) / len(c_tokens)
                claim_scores.append(overlap)
            faithfulness = sum(claim_scores) / max(1, len(claim_scores))

        q_tokens = _content_tokens(query)
        if not retrieved_contexts or not q_tokens:
            contextual_relevance = 0.0
            contextual_precision = 0.0
        else:
            chunk_relevances: list[float] = []
            relevant_flags: list[int] = []
            for ctx in retrieved_contexts:
                ctx_toks = _content_tokens(ctx)
                tok_cov = len(q_tokens & ctx_toks) / len(q_tokens)
                sim = max(0.0, float(embed_text_simd(query) @ embed_text_simd(ctx)))
                score = min(1.0, 0.65 * tok_cov + 0.55 * sim)
                chunk_relevances.append(score)
                relevant_flags.append(1 if score >= 0.25 else 0)

            contextual_relevance = max(chunk_relevances)
            prec_at_k: list[float] = []
            running_hits = 0
            for rank_idx, flag in enumerate(relevant_flags, start=1):
                if flag == 1:
                    running_hits += 1
                    prec_at_k.append(running_hits / rank_idx)
            contextual_precision = sum(prec_at_k) / len(prec_at_k) if prec_at_k else 0.0

        ans_tokens = _content_tokens(answer)
        q_cov = (len(q_tokens & ans_tokens) / len(q_tokens)) if q_tokens else 0.0
        ans_cosine = max(0.0, float(embed_text_simd(query) @ embed_text_simd(answer)))
        answer_relevance = min(1.0, 0.55 * q_cov + 0.60 * ans_cosine)

        overall = (
            0.40 * faithfulness
            + 0.20 * contextual_relevance
            + 0.20 * contextual_precision
            + 0.20 * answer_relevance
        )
        passed = faithfulness >= self._min_faithfulness

        hub = get_observability_hub()
        hub.record_model_quality(
            pillar="llmops",
            metric_name="ragas_faithfulness",
            value=faithfulness,
        )
        hub.record_model_quality(
            pillar="llmops",
            metric_name="ragas_overall",
            value=overall,
        )

        return RAGASMetricsReport(
            faithfulness=round(faithfulness, 5),
            contextual_relevance=round(contextual_relevance, 5),
            contextual_precision=round(contextual_precision, 5),
            answer_relevance=round(answer_relevance, 5),
            overall_ragas_score=round(overall, 5),
            passed_quality_gate=passed,
        )
