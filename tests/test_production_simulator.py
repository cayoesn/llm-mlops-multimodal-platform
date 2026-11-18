"""TDD suite for Continuous Production Traffic Simulator engine, REST/SSE API, and CLI."""

from __future__ import annotations

import asyncio
import io
import json
import socket
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import PlatformRuntimeContainer
from app.main import create_app
from app.simulator import (
    ProductionTrafficSimulator,
    main,
    run_api_client_simulation,
    run_local_cli_simulation,
)


async def test_simulator_executes_all_pillars_drift_wave_and_caps_history(tmp_path: Path) -> None:
    """Verify 8-scenario rotation + PSI drift wave + bounded O(1) deque history ceiling."""
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(container, max_history=5)

    for _ in range(9):
        await sim.execute_tick()

    st = sim.status(limit_recent=10)
    assert st.running is False
    assert st.total_ticks == 9
    assert set(st.pillar_counts.keys()) == {"mlops", "llmops", "multimodal"}
    assert st.outcome_counts.get("retrained", 0) == 1
    assert st.outcome_counts.get("scored", 0) >= 1
    assert st.outcome_counts.get("grounded", 0) >= 1
    assert st.outcome_counts.get("cache_hit", 0) >= 1
    assert st.outcome_counts.get("pii_redacted", 0) >= 1
    assert st.outcome_counts.get("blocked_guardrail", 0) >= 1
    assert st.outcome_counts.get("lora_adapted", 0) >= 1
    assert st.outcome_counts.get("consistent_approved", 0) >= 1
    assert st.outcome_counts.get("contradiction_escalated", 0) >= 1
    assert len(st.recent_events) == 5
    assert [ev.tick for ev in st.recent_events] == [5, 6, 7, 8, 9]


async def test_simulator_non_starvation_under_even_drift_wave_divisor(tmp_path: Path) -> None:
    """Verify even drift_wave_every=2 does not starve any normal scenario slot."""
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(container, max_history=50)
    await sim.start(interval_seconds=60.0, drift_wave_every=2)
    await sim.stop()

    observed_scenarios: set[str] = set()
    for _ in range(16):
        ev = await sim.execute_tick()
        observed_scenarios.add(ev.scenario)

    assert len(observed_scenarios) == 9
    assert "mlops_drift_wave" in observed_scenarios
    assert "multimodal_contradiction_fraud" in observed_scenarios


async def test_simulator_idempotent_start_and_immediate_stop_wakeup(tmp_path: Path) -> None:
    """Verify repeated start() reuses active task and stop() wakes 60s sleep in < 0.25s."""
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(container, max_history=20)

    st1 = await sim.start(interval_seconds=0.05, drift_wave_every=6)
    assert st1.running is True
    first_task = sim.active_task
    assert first_task is not None

    collected = []
    async for event in sim.stream_events(max_events=2, poll_interval=0.02):
        collected.append(event)
    assert len(collected) == 2

    st2 = await sim.start(interval_seconds=60.0, drift_wave_every=10)
    assert st2.running is True
    assert st2.interval_seconds == 60.0
    assert st2.drift_wave_every == 10
    assert sim.active_task is first_task

    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    stopped = await sim.stop()
    elapsed = time.perf_counter() - t0
    assert stopped.running is False
    assert elapsed < 0.25
    assert sim.active_task is None


async def test_simulator_records_unexpected_tick_exception_without_crashing(tmp_path: Path) -> None:
    """Verify unexpected runtime error in a scenario logs and increments error outcome safely."""
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(container, max_history=10)

    with patch.object(
        container.dvc_runner.feast_service,
        "get_online_features",
        side_effect=RuntimeError("synthetic_feast_outage"),
    ):
        ev = await sim.execute_tick()

    assert ev.outcome == "error"
    assert ev.detail.get("error_type") == "RuntimeError"
    assert sim.status().outcome_counts.get("error") == 1


def test_simulate_api_start_status_sse_stream_stop_and_audit_trail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /api/v1/simulate/{status,start,stream,stop} and live audit trail updates."""
    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "sim_api"))
    from app.config import reset_settings_cache

    reset_settings_cache()
    app = create_app(max_request_body_bytes=512_000)
    with TestClient(app) as client:
        st0 = client.get("/api/v1/simulate/status")
        assert st0.status_code == 200
        assert st0.json()["running"] is False

        bad_start = client.post("/api/v1/simulate/start", json={"interval_seconds": 0.001})
        assert bad_start.status_code == 422

        started = client.post(
            "/api/v1/simulate/start",
            json={"interval_seconds": 0.05, "drift_wave_every": 4},
        )
        assert started.status_code == 200
        assert started.json()["running"] is True

        with client.stream("GET", "/api/v1/simulate/stream", params={"max_events": 4}) as sse:
            assert sse.status_code == 200
            assert "text/event-stream" in sse.headers["content-type"]
            events = [
                json.loads(line.removeprefix("data: "))
                for line in sse.iter_lines()
                if line.startswith("data: ")
            ]
            assert len(events) == 4
            assert {ev["pillar"] for ev in events}.issubset({"mlops", "llmops", "multimodal"})

        stopped = client.post("/api/v1/simulate/stop")
        assert stopped.status_code == 200
        assert stopped.json()["running"] is False
        assert stopped.json()["total_ticks"] >= 4

        audit = client.get("/api/v1/audit/verify")
        assert audit.status_code == 200
        assert audit.json()["valid"] is True
        assert audit.json()["records"] >= 1


def test_simulate_lifespan_shutdown_stops_active_background_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify FastAPI lifespan exit cleanly stops an unstopped 60s simulation task in < 0.5s."""
    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "sim_lifespan"))
    from app.config import reset_settings_cache

    reset_settings_cache()
    app = create_app()
    assert isinstance(app, FastAPI)
    t_exit = 0.0
    with TestClient(app) as client:
        resp = client.post("/api/v1/simulate/start", json={"interval_seconds": 60.0})
        assert resp.status_code == 200
        assert resp.json()["running"] is True
        t_exit = time.perf_counter()
    shutdown_elapsed = time.perf_counter() - t_exit
    assert shutdown_elapsed < 0.5
    assert app.state.container.simulator.status().running is False


async def test_cli_local_and_asgi_api_modes_execute_bounded_cycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_local_cli_simulation, run_api_client_simulation, and CLI main() entrypoint."""
    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "sim_cli"))
    from app.config import reset_settings_cache

    reset_settings_cache()
    local_buf = io.StringIO()
    local_summary = await run_local_cli_simulation(
        interval_seconds=0.05,
        drift_wave_every=2,
        max_cycles=3,
        output_stream=local_buf,
    )
    assert local_summary.total_ticks == 3
    assert "tick=1" in local_buf.getvalue()

    app = create_app(max_request_body_bytes=512_000)
    assert isinstance(app, FastAPI)
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    app.state.container = container

    api_buf = io.StringIO()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        completed = await run_api_client_simulation(
            base_url="http://testserver",
            interval_seconds=0.01,
            drift_wave_every=8,
            max_cycles=9,
            http_client=client,
            enable_websocket=False,
            output_stream=api_buf,
        )
    await container.llm_gateway.aclose()
    assert completed == 9
    out_text = api_buf.getvalue()
    assert "mlops_drift_wave" in out_text
    assert "llmops_prompt_injection_blocked" in out_text
    assert "multimodal_contradiction_fraud" in out_text

    rc = await asyncio.to_thread(
        main,
        ["--mode", "local", "--interval", "0.05", "--drift-every", "3", "--max-cycles", "2"],
    )
    assert rc == 0


async def test_cli_api_mode_live_tcp_and_websocket_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify external API simulator exercises real HTTP + WebSocket over ephemeral TCP."""
    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "sim_tcp"))
    from app.config import reset_settings_cache

    reset_settings_cache()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    app = create_app(max_request_body_bytes=512_000)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)

    buf = io.StringIO()
    try:
        completed = await run_api_client_simulation(
            base_url=f"http://127.0.0.1:{port}",
            interval_seconds=0.01,
            drift_wave_every=4,
            max_cycles=2,
            enable_websocket=True,
            output_stream=buf,
        )
    finally:
        server.should_exit = True
        await server_task

    assert completed == 2
    assert "ws_stream_visual=" in buf.getvalue()
