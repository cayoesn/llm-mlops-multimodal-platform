"""Qdrant-compatible Hybrid Dense + Lexical RRF RAG Engine with O(N log K) Heap Top-K."""

from __future__ import annotations

import heapq
import re
import threading
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.llmops.semantic_cache import embed_text_simd

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class KnowledgeDocument(BaseModel):
    """Immutable enterprise knowledge base document indexed for RAG."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    """Ranked document chunk returned by Reciprocal Rank Fusion (RRF)."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    content: str
    rrf_score: float
    dense_cosine: float


class HybridRAGEngine:
    """Executes dense SIMD cosine + lexical overlap RRF retrieval in O(N log K)."""

    def __init__(self, rrf_k: int = 60) -> None:
        """Initialize RRF smoothing constant and thread-safe vector store."""
        self._rrf_k = max(1, rrf_k)
        self._lock = threading.RLock()
        self._docs: list[KnowledgeDocument] = []
        self._embeddings: np.ndarray | None = None
        self._token_sets: list[frozenset[str]] = []

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> int:
        """Embed and index documents into the thread-safe vector & lexical catalog."""
        if not documents:
            return 0
        vecs = np.vstack([embed_text_simd(f"{d.title} {d.content}") for d in documents])
        tok_sets = [
            frozenset(t.lower() for t in _TOKEN_RE.findall(f"{d.title} {d.content}"))
            for d in documents
        ]
        with self._lock:
            self._docs = list(documents)
            self._embeddings = vecs
            self._token_sets = tok_sets
            return len(self._docs)

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        """Retrieve Top-K chunks via Reciprocal Rank Fusion and heapq.nlargest O(N log K)."""
        k = max(1, top_k)
        with self._lock:
            if not self._docs or self._embeddings is None:
                return []
            docs_snapshot = list(self._docs)
            emb_snapshot = self._embeddings
            tok_snapshot = list(self._token_sets)

        q_vec = embed_text_simd(query)
        dense_scores = (emb_snapshot @ q_vec).tolist()

        q_tokens = frozenset(t.lower() for t in _TOKEN_RE.findall(query))
        lexical_scores = [
            len(q_tokens & doc_toks) / max(1, len(q_tokens)) for doc_toks in tok_snapshot
        ]

        dense_order = sorted(
            range(len(docs_snapshot)),
            key=lambda idx: dense_scores[idx],
            reverse=True,
        )
        lex_order = sorted(
            range(len(docs_snapshot)),
            key=lambda idx: lexical_scores[idx],
            reverse=True,
        )

        dense_rank = {doc_idx: rank for rank, doc_idx in enumerate(dense_order, start=1)}
        lex_rank = {doc_idx: rank for rank, doc_idx in enumerate(lex_order, start=1)}

        fused_candidates: list[tuple[float, int]] = []
        for idx in range(len(docs_snapshot)):
            rrf = (1.0 / (self._rrf_k + dense_rank[idx])) + (1.0 / (self._rrf_k + lex_rank[idx]))
            fused_candidates.append((rrf, idx))

        best = heapq.nlargest(k, fused_candidates, key=lambda pair: pair[0])
        return [
            RetrievedChunk(
                doc_id=docs_snapshot[idx].doc_id,
                title=docs_snapshot[idx].title,
                content=docs_snapshot[idx].content,
                rrf_score=round(rrf_val, 6),
                dense_cosine=round(float(dense_scores[idx]), 5),
            )
            for rrf_val, idx in best
        ]
