"""Shared pytest fixtures ensuring hermetic singleton and ContextVar resets."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_global_platform_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Reset settings, registries, observability state, and isolate artifact paths per test."""
    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OMNIPULSE_AUDIT_HMAC_SECRET", "test-hmac-secret-key-omnipulse-2026")
    try:
        from app.config import reset_settings_cache
        from app.core.observability import reset_observability_hub
        from app.core.resilience import reset_circuit_breaker_registry

        reset_settings_cache()
        reset_observability_hub()
        reset_circuit_breaker_registry()
    except ImportError:
        pass
    yield
    try:
        from app.config import reset_settings_cache
        from app.core.observability import reset_observability_hub
        from app.core.resilience import reset_circuit_breaker_registry

        reset_settings_cache()
        reset_observability_hub()
        reset_circuit_breaker_registry()
    except ImportError:
        pass
