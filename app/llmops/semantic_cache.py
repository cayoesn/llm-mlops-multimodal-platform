"""O(1) OrderedDict LRU + Vectorized NumPy BLAS Cosine Semantic Cache (Unit I.5 & V.1)."""

from __future__ import annotations

import collections
import hashlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "the",
        "timeframe",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


def embed_text_simd(text: str, dim: int = 256) -> np.ndarray:
    """Project text into an L2-normalized float32 feature vector using specificity weighting."""
    vec = np.zeros(dim, dtype=np.float32)
    raw_tokens = _WORD_RE.findall(text)
    for raw_tok in raw_tokens:
        low = raw_tok.lower()
        if low in _STOPWORDS:
            weight = 0.15
        else:
            acronym_bonus = 0.8 if (raw_tok.isupper() and len(raw_tok) >= 2) else 0.0
            weight = 1.0 + 0.15 * min(12, len(low)) + acronym_bonus
        digest = hashlib.blake2b(low.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[bucket] += sign * weight

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


@dataclass(slots=True)
class _CacheEntry:
    """Internal mutable cache record tracked inside OrderedDict."""

    query_text: str
    response_text: str
    embedding: np.ndarray
    expires_at_monotonic: float


class SemanticCacheHit(BaseModel):
    """Immutable cache hit response detailing exact vs cosine match similarity."""

    model_config = ConfigDict(frozen=True)

    response_text: str
    similarity: float
    hit_type: Literal["exact_lru", "cosine_semantic"]


class SemanticCache:
    """Bounded two-tier LRU + vectorized BLAS cosine similarity cache with TTL expiration."""

    def __init__(
        self,
        max_size: int = 512,
        similarity_threshold: float = 0.90,
        ttl_seconds: float = 900.0,
    ) -> None:
        """Configure bounded LRU capacity, cosine similarity gate, and TTL seconds."""
        self._max_size = max(1, max_size)
        self._threshold = similarity_threshold
        self._ttl_seconds = max(0.1, ttl_seconds)
        self._entries: collections.OrderedDict[str, _CacheEntry] = collections.OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Return active cache entry count."""
        with self._lock:
            return len(self._entries)

    @property
    def hit_ratio(self) -> float:
        """Return cumulative cache hit ratio in [0.0, 1.0]."""
        with self._lock:
            total = self._hits + self._misses
            return (self._hits / total) if total > 0 else 0.0

    @staticmethod
    def _normalize_key(query: str) -> str:
        """Compute canonical SHA-256 exact-match key."""
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _purge_expired_locked(self, now: float) -> None:
        """Remove expired entries lazily under RLock."""
        expired_keys = [k for k, v in self._entries.items() if v.expires_at_monotonic <= now]
        for key in expired_keys:
            self._entries.pop(key, None)

    def get(self, query: str) -> SemanticCacheHit | None:
        """Lookup exact key in O(1) first, then evaluate vectorized BLAS cosine similarity."""
        key = self._normalize_key(query)
        now = time.monotonic()
        with self._lock:
            self._purge_expired_locked(now)
            exact_entry = self._entries.get(key)
            if exact_entry is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return SemanticCacheHit(
                    response_text=exact_entry.response_text,
                    similarity=1.0,
                    hit_type="exact_lru",
                )

            if not self._entries:
                self._misses += 1
                return None

            keys_snapshot = list(self._entries.keys())
            matrix = np.vstack([self._entries[k].embedding for k in keys_snapshot])
            query_vec = embed_text_simd(query)
            scores = matrix @ query_vec
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= self._threshold:
                matched_key = keys_snapshot[best_idx]
                matched = self._entries[matched_key]
                self._entries.move_to_end(matched_key)
                self._hits += 1
                return SemanticCacheHit(
                    response_text=matched.response_text,
                    similarity=round(best_score, 5),
                    hit_type="cosine_semantic",
                )

            self._misses += 1
            return None

    def put(self, query: str, response_text: str) -> None:
        """Insert or refresh a query-response pair enforcing max_size LRU eviction."""
        key = self._normalize_key(query)
        entry = _CacheEntry(
            query_text=query,
            response_text=response_text,
            embedding=embed_text_simd(query),
            expires_at_monotonic=time.monotonic() + self._ttl_seconds,
        )
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = entry
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)
