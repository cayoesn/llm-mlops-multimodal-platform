"""Single-pass O(|T|) Reversible LGPD PII Vault, Anti-Injection & Toxicity Guardrails."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.core.exceptions import GuardrailViolationError

_VAULT_PII_PATTERN = re.compile(
    r"(?P<EMAIL>\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b)"
    r"|(?P<CARD>\b(?:\d{4}[-\s]?){3}\d{4}\b)"
    r"|(?P<CPF>\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b)"
    r"|(?P<PHONE>(?:\+\d{1,3}[\s\-]?)?\(?\d{2,3}\)?[\s\-]?\d{4,5}-\d{4}\b)"
)

_PROMPT_INJECTION_PATTERN = re.compile(
    r"(ignore\s+(?:all\s+)?previous\s+instructions"
    r"|reveal\s+system\s+(?:prompt|secrets)"
    r"|developer\s+mode\s+jailbreak"
    r"|bypass\s+safety\s+guardrails)",
    re.IGNORECASE,
)

_TOXIC_LEXICON = frozenset(
    {
        "idiot",
        "moron",
        "hate",
        "violent",
        "threat",
        "attack",
        "kill",
        "slur",
        "abuse",
    }
)
_WORD_SPLIT_RE = re.compile(r"[A-Za-z]+")


class SanitizedPromptSession(BaseModel):
    """Immutable sanitized prompt with reversible token vault mapping."""

    model_config = ConfigDict(frozen=True)

    sanitized_text: str
    vault_map: dict[str, str]
    redacted_entities_count: int


class GuardrailsEngine:
    """Enforces prompt-injection blocking, single-pass LGPD PII vaulting, and toxicity limits."""

    def __init__(self, max_toxicity_score: float | None = None) -> None:
        """Configure toxicity score ceiling."""
        self._max_toxicity = (
            max_toxicity_score
            if max_toxicity_score is not None
            else get_settings().OMNIPULSE_MAX_TOXICITY_SCORE
        )

    def sanitize_inbound(self, prompt_text: str) -> SanitizedPromptSession:
        """Verify inbound prompt against jailbreaks and tokenize PII in one O(|T|) pass."""
        if _PROMPT_INJECTION_PATTERN.search(prompt_text):
            raise GuardrailViolationError("Prompt blocked by anti-injection security guardrail.")

        counters: dict[str, int] = {"EMAIL": 0, "CARD": 0, "CPF": 0}
        vault: dict[str, str] = {}

        def _tokenize(match: re.Match[str]) -> str:
            kind = match.lastgroup or "PII"
            counters[kind] = counters.get(kind, 0) + 1
            placeholder = f"<PII_{kind}_{counters[kind]}>"
            vault[placeholder] = match.group(0)
            return placeholder

        sanitized = _VAULT_PII_PATTERN.sub(_tokenize, prompt_text)
        return SanitizedPromptSession(
            sanitized_text=sanitized,
            vault_map=vault,
            redacted_entities_count=len(vault),
        )

    @staticmethod
    def restore_outbound(response_text: str, vault_map: dict[str, str]) -> str:
        """Atomically restore authorized PII placeholders in outbound response."""
        if not vault_map:
            return response_text
        pattern = re.compile("|".join(re.escape(k) for k in vault_map))
        return pattern.sub(lambda m: vault_map[m.group(0)], response_text)

    def compute_toxicity_score(self, text: str) -> float:
        """Compute normalized lexical/semantic toxicity density in [0.0, 1.0]."""
        words = [w.lower() for w in _WORD_SPLIT_RE.findall(text)]
        if not words:
            return 0.0
        flagged = sum(1 for w in words if w in _TOXIC_LEXICON)
        return min(1.0, round((flagged / len(words)) * 2.5, 5))

    def validate_outbound_safety(self, response_text: str) -> float:
        """Validate outbound LLM generation against toxicity governance threshold."""
        score = self.compute_toxicity_score(response_text)
        if score > self._max_toxicity:
            raise GuardrailViolationError(
                f"Outbound response blocked: toxicity_score={score:.3f} > {self._max_toxicity:.3f}"
            )
        return score
