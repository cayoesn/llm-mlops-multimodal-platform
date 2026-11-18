"""Self-Hosted Open-Source LLM Serving Gateway (Ollama, vLLM & Local Deterministic)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.core.exceptions import DownstreamHTTPStatusError
from app.core.observability import get_observability_hub, log_info
from app.core.resilience import ResilientExecutor, TokenBucketRateLimiter, get_circuit_breaker
from app.llmops.guardrails import GuardrailsEngine
from app.llmops.prompt_checkpoint_registry import PromptCheckpointRegistry
from app.llmops.rag_engine import HybridRAGEngine, RetrievedChunk
from app.llmops.ragas_evaluator import RAGASEvaluator, RAGASMetricsReport
from app.llmops.semantic_cache import SemanticCache

_LOGGER = logging.getLogger("omnipulse.llmops.serving")


class LLMBackendProtocol(Protocol):
    """Protocol implemented by Ollama, vLLM, and hermetic deterministic CPU synthesizers."""

    async def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        """Generate grounded text completion asynchronously."""


class LocalGroundedSynthesizer:
    """Deterministic extractive-grounded CPU generator for zero-flake local & CI execution."""

    async def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        """Synthesize a concise policy-grounded response directly from retrieved context."""
        if not context.strip():
            return "Insufficient verified policy context available to answer this inquiry."
        return f"Grounded Policy Response: {context.strip()}"


class OllamaHTTPBackend:
    """Async self-hosted Ollama (/api/generate) serving adapter with connection pooling."""

    def __init__(
        self,
        base_url: str,
        model: str = "llama3.2:3b",
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure Ollama endpoint, model tag, and reusable AsyncClient pool."""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=4.0)

    async def aclose(self) -> None:
        """Close the owned AsyncClient connection pool on shutdown."""
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        """Invoke Ollama HTTP generation API mapping non-2xx responses to semantic errors."""
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_query}"
        resp = await self._client.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False},
        )
        if resp.status_code >= 400:
            raise DownstreamHTTPStatusError(resp.status_code, resp.text[:200])
        body = resp.json()
        if isinstance(body, dict) and isinstance(body.get("response"), str):
            return str(body["response"])
        return "Empty response from Ollama."


class VLLMHTTPBackend:
    """Async OpenAI-compatible vLLM (/v1/chat/completions) serving adapter with pooling."""

    def __init__(
        self,
        base_url: str,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure vLLM endpoint, model identifier, and reusable AsyncClient pool."""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=4.0)

    async def aclose(self) -> None:
        """Close the owned AsyncClient connection pool on shutdown."""
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        """Invoke vLLM OpenAI-compatible chat completions endpoint."""
        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {user_query}",
                    },
                ],
                "temperature": 0.0,
            },
        )
        if resp.status_code >= 400:
            raise DownstreamHTTPStatusError(resp.status_code, resp.text[:200])
        body = resp.json()
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {})
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return str(msg["content"])
        return "Empty response from vLLM."


class LLMChatResponse(BaseModel):
    """Immutable end-to-end LLMOps gateway response including RAGAS & Guardrail telemetry."""

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


class LLMServingGateway:
    """Unified LLMOps pipeline: RateLimit -> Guardrails -> Cache -> RAG -> Serving -> RAGAS."""

    DEFAULT_PROMPT_NAME = "omnipulse_enterprise_copilot"

    def __init__(
        self,
        *,
        rag_engine: HybridRAGEngine | None = None,
        backend: LLMBackendProtocol | None = None,
        cache: SemanticCache | None = None,
        guardrails: GuardrailsEngine | None = None,
        ragas: RAGASEvaluator | None = None,
    ) -> None:
        """Wire RAG, Semantic Cache, Guardrails, Prompt Registry, and ResilientExecutor."""
        settings = get_settings()
        self.rag_engine = rag_engine if rag_engine is not None else HybridRAGEngine()
        if backend is not None:
            self.backend: LLMBackendProtocol = backend
        elif settings.OMNIPULSE_LLM_BACKEND == "ollama":
            self.backend = OllamaHTTPBackend(settings.OMNIPULSE_OLLAMA_BASE_URL)
        elif settings.OMNIPULSE_LLM_BACKEND == "vllm":
            self.backend = VLLMHTTPBackend(settings.OMNIPULSE_VLLM_BASE_URL)
        else:
            self.backend = LocalGroundedSynthesizer()
        self.cache = (
            cache
            if cache is not None
            else SemanticCache(
                max_size=settings.OMNIPULSE_SEMANTIC_CACHE_MAX_SIZE,
                similarity_threshold=settings.OMNIPULSE_SEMANTIC_CACHE_SIMILARITY,
                ttl_seconds=settings.OMNIPULSE_SEMANTIC_CACHE_TTL_SECONDS,
            )
        )
        self.guardrails = guardrails if guardrails is not None else GuardrailsEngine()
        self.ragas = ragas if ragas is not None else RAGASEvaluator()
        self.prompt_registry = PromptCheckpointRegistry()
        self.prompt_registry.register_prompt(
            name=self.DEFAULT_PROMPT_NAME,
            system_template=(
                "You are OmniPulse Enterprise AI Copilot. Answer strictly using retrieved "
                "corporate policies and comply with LGPD/GDPR privacy rules."
            ),
            max_context_tokens=256,
        )
        self._rate_limiter = TokenBucketRateLimiter(capacity=120.0, refill_rate_per_second=40.0)
        self._executor = ResilientExecutor(
            breaker=get_circuit_breaker("llm_serving_backend"),
            max_retries=1,
        )
        self._lock = threading.RLock()
        self._backend_invocations = 0

    async def aclose(self) -> None:
        """Close pooled HTTP client resources if the active backend owns a connection pool."""
        if isinstance(self.backend, (OllamaHTTPBackend, VLLMHTTPBackend)):
            await self.backend.aclose()

    @property
    def backend_invocations(self) -> int:
        """Return total non-cached backend LLM generation invocations."""
        with self._lock:
            return self._backend_invocations

    async def chat(self, *, query: str, tenant_id: str = "default") -> LLMChatResponse:
        """Execute full LLMOps chat lifecycle with guardrails, caching, RAG, and RAGAS."""
        started_ns = time.perf_counter_ns()
        self._rate_limiter.consume(1.0)

        sanitized_session = self.guardrails.sanitize_inbound(query)
        clean_query = sanitized_session.sanitized_text
        prompt_spec = self.prompt_registry.get_latest_prompt(self.DEFAULT_PROMPT_NAME)

        chunks: list[RetrievedChunk] = self.rag_engine.retrieve(clean_query, top_k=2)
        citations = [c.doc_id for c in chunks]
        context_list = [c.content for c in chunks]
        packed_context = self.prompt_registry.fit_context_budget(
            self.DEFAULT_PROMPT_NAME,
            context_list,
        )

        cached = self.cache.get(clean_query)
        if cached is not None:
            raw_answer = cached.response_text
            cache_hit = True
            hit_type: str | None = cached.hit_type
        else:
            cache_hit = False
            hit_type = None

            async def _call_llm() -> str:
                return await self.backend.generate(
                    prompt_spec.system_template,
                    clean_query,
                    packed_context,
                )

            raw_answer = await self._executor.execute_async(
                _call_llm,
                total_sla_seconds=4.5,
                operation_name="llm_generate",
            )
            with self._lock:
                self._backend_invocations += 1
            self.cache.put(clean_query, raw_answer)

        toxicity = self.guardrails.validate_outbound_safety(raw_answer)
        ragas_report = self.ragas.evaluate(
            query=clean_query,
            retrieved_contexts=context_list,
            answer=raw_answer,
        )
        final_answer = self.guardrails.restore_outbound(raw_answer, sanitized_session.vault_map)

        elapsed_ms = max(0.01, (time.perf_counter_ns() - started_ns) / 1_000_000.0)
        get_observability_hub().record_inference(
            "llmops",
            latency_seconds=elapsed_ms / 1_000.0,
            success=True,
        )
        log_info(
            _LOGGER,
            "llmops_chat_completed",
            tenant_id=tenant_id,
            cache_hit=cache_hit,
            faithfulness=ragas_report.faithfulness,
        )
        return LLMChatResponse(
            answer=final_answer,
            citations=citations,
            cache_hit=cache_hit,
            cache_hit_type=hit_type,
            pii_entities_redacted=sanitized_session.redacted_entities_count,
            toxicity_score=toxicity,
            prompt_version_hash=prompt_spec.version_hash,
            ragas_metrics=ragas_report,
            latency_ms=round(elapsed_ms, 4),
        )
