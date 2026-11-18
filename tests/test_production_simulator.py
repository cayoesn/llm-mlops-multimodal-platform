"""TDD suite for Continuous Production Traffic Simulator engine, REST/SSE API, and CLI."""

from __future__ import annotations

import asyncio
import io
import json
import socket
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
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


async def test_concurrent_stop_and_start_never_orphans_background_task(tmp_path: Path) -> None:
    """Regression C1: overlapping stop() awaiting slow tick + start() serializes cleanly."""
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(container, max_history=20)

    tick_entered = asyncio.Event()
    allow_tick_finish = asyncio.Event()
    orig_predict = container.dvc_runner.feast_service.get_online_features

    def _slow_first_tick(customer_id: str) -> object:
        if not tick_entered.is_set():
            tick_entered.set()
            deadline = time.perf_counter() + 0.15
            while not allow_tick_finish.is_set() and time.perf_counter() < deadline:
                time.sleep(0.01)
        return orig_predict(customer_id)

    with patch.object(
        container.dvc_runner.feast_service,
        "get_online_features",
        side_effect=_slow_first_tick,
    ):
        await sim.start(interval_seconds=60.0, drift_wave_every=8)
        await asyncio.wait_for(tick_entered.wait(), timeout=2.0)

        stop_task = asyncio.create_task(sim.stop())
        await asyncio.sleep(0.02)
        start_task = asyncio.create_task(sim.start(interval_seconds=60.0, drift_wave_every=8))
        allow_tick_finish.set()
        await asyncio.gather(stop_task, start_task)

    assert sim.status().running is True
    assert sim.active_task is not None
    await sim.stop()
    assert sim.status().running is False
    assert sim.active_task is None
    await container.llm_gateway.aclose()


async def test_stream_events_exits_cleanly_when_simulator_stopped_and_disconnect_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression I3: stream_events returns immediately when stopped; ASGI disconnect works."""
    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "sim_sse_stop"))
    from app.config import reset_settings_cache

    reset_settings_cache()
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)

    async def _collect_stopped() -> list[object]:
        return [ev async for ev in container.simulator.stream_events(poll_interval=0.01)]

    drained = await asyncio.wait_for(_collect_stopped(), timeout=0.25)
    assert drained == []
    await container.llm_gateway.aclose()

    app = create_app()
    observed_disconnected: list[bool] = []

    @app.get("/_test/check-disconnect")
    async def _check_disc(request: Request) -> dict[str, bool]:
        flag = await request.is_disconnected()
        observed_disconnected.append(flag)
        return {"disconnected": flag}

    async def _mock_receive() -> MutableMapping[str, Any]:
        return {"type": "http.disconnect"}

    sent_messages: list[MutableMapping[str, Any]] = []

    async def _mock_send(msg: MutableMapping[str, Any]) -> None:
        sent_messages.append(msg)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/_test/check-disconnect",
        "raw_path": b"/_test/check-disconnect",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8017),
    }
    await app(scope, _mock_receive, _mock_send)
    assert observed_disconnected == [True]


async def test_api_client_handles_non_json_500_ws_errors_bounded_buffers_and_exc_pii(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression C2, I4, I5: non-JSON 500 resilience, RLHF/ONNX bounds, and log_warning PII."""
    import logging

    from app.core.observability import log_warning
    from app.llmops.lora_rlhf_loop import LoRAAdapterLayer, PreferencePair, RLHFAdaptationLoop
    from app.simulator import _http_to_ws_url

    assert _http_to_ws_url("ws://localhost:8017") == "ws://localhost:8017/ws/v1/multimodal/stream"
    assert _http_to_ws_url("wss://example.com/") == "wss://example.com/ws/v1/multimodal/stream"

    test_logger = logging.getLogger("omnipulse.test.pii")
    with caplog.at_level(logging.WARNING, logger="omnipulse.test.pii"):
        log_warning(
            test_logger,
            "warn_with_pii_exception",
            exc=ValueError("Leaked CPF 123.456.789-01 and mail a.b@omnipulse.ai"),
        )
    assert "123.456.789-01" not in caplog.text
    assert "a.b@omnipulse.ai" not in caplog.text

    layer = LoRAAdapterLayer(d_in=16, d_out=16, rank=2)
    rlhf = RLHFAdaptationLoop(layer, max_buffer_size=4)
    pair = PreferencePair(
        prompt_embedding=[0.1] * 16,
        chosen_embedding=[0.2] * 16,
        rejected_embedding=[-0.1] * 16,
    )
    for _ in range(10):
        buf_len = rlhf.record_human_preference(pair)
    assert buf_len == 4

    def _failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error Plaintext")

    err_buf = io.StringIO()
    mock_transport = httpx.MockTransport(_failing_handler)
    async with httpx.AsyncClient(transport=mock_transport, base_url="http://broken") as bad_client:
        cycles = await run_api_client_simulation(
            base_url="http://broken",
            interval_seconds=0.01,
            drift_wave_every=2,
            max_cycles=6,
            http_client=bad_client,
            enable_websocket=False,
            output_stream=err_buf,
        )
    assert cycles == 6
    assert "http=500" in err_buf.getvalue()
    assert "guardrail_blocked" not in err_buf.getvalue()

    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(container, max_history=25)
    for _ in range(18):
        await sim.execute_tick()
    rag_events = [ev for ev in sim.status().recent_events if ev.scenario == "llmops_rag_chat"]
    assert len(rag_events) >= 2
    assert all(ev.detail.get("cache_hit") is False for ev in rag_events)
    active_paths = {
        v.onnx_artifact_path
        for v in container.dvc_runner.mlflow_manager.list_versions()
        if v.stage != "Archived"
    }
    assert set(container.dvc_runner.mlflow_manager._ort_sessions.keys()).issubset(active_paths)
    await container.llm_gateway.aclose()

    def _net_err_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    net_buf = io.StringIO()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_net_err_handler),
        base_url="http://127.0.0.1:1",
    ) as net_client:
        net_cycles = await run_api_client_simulation(
            base_url="http://127.0.0.1:1",
            interval_seconds=0.01,
            max_cycles=2,
            http_client=net_client,
            enable_websocket=True,
            output_stream=net_buf,
        )
    assert net_cycles == 2
    assert "transport_error=ConnectError" in net_buf.getvalue()
    assert "ws_stream_error=" in net_buf.getvalue()


async def test_simulator_admin_token_guard_and_retrain_cooldown_throttle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify X-Simulator-Token 403 guard and monotonic retrain cooldown anti-amplification."""
    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "sim_guard"))
    monkeypatch.setenv("OMNIPULSE_SIMULATOR_ADMIN_TOKEN", "omnipulse-sim-admin-secret")
    from app.config import reset_settings_cache

    reset_settings_cache()
    app = create_app()
    with TestClient(app) as client:
        denied = client.post("/api/v1/simulate/start", json={"interval_seconds": 1.0})
        assert denied.status_code == 403

        wrong_tok = client.post(
            "/api/v1/simulate/start",
            json={"interval_seconds": 1.0},
            headers={"X-Simulator-Token": "wrong-token"},
        )
        assert wrong_tok.status_code == 403

        allowed = client.post(
            "/api/v1/simulate/start",
            json={"interval_seconds": 1.0},
            headers={"X-Simulator-Token": "omnipulse-sim-admin-secret"},
        )
        assert allowed.status_code == 200
        assert allowed.json()["running"] is True

        stop_denied = client.post("/api/v1/simulate/stop")
        assert stop_denied.status_code == 403

        stop_ok = client.post(
            "/api/v1/simulate/stop",
            headers={"X-Simulator-Token": "omnipulse-sim-admin-secret"},
        )
        assert stop_ok.status_code == 200
        assert stop_ok.json()["running"] is False

    monkeypatch.delenv("OMNIPULSE_SIMULATOR_ADMIN_TOKEN", raising=False)
    reset_settings_cache()

    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(
        container,
        max_history=10,
        min_retrain_cooldown_seconds=30.0,
    )
    await sim.start(interval_seconds=60.0, drift_wave_every=1)
    await sim.stop()

    ev1 = await sim.execute_tick()
    ev2 = await sim.execute_tick()
    ev3 = await sim.execute_tick()
    assert ev1.scenario == "mlops_drift_wave"
    assert ev1.outcome == "retrained"
    assert ev1.detail.get("cooldown_throttled") is False
    assert ev2.outcome == "drift_checked"
    assert ev2.detail.get("cooldown_throttled") is True
    assert ev3.outcome == "drift_checked"
    assert ev3.detail.get("cooldown_throttled") is True
    await container.llm_gateway.aclose()


async def test_mlflow_concurrent_predict_and_archive_toctou_and_disk_pruning(
    tmp_path: Path,
) -> None:
    """Verify zero leaked archived ONNX sessions under concurrency and bounded disk pruning."""
    from app.mlops.mlflow_lifecycle import MLflowLifecycleManager

    art_root = tmp_path / "mlflow_pruning"
    manager = MLflowLifecycleManager(
        artifact_root=art_root,
        max_archived_versions=2,
        max_tracked_runs=6,
    )
    first_run = manager.train_and_track_experiment(run_name="v1", sample_size=120, random_seed=1)
    v1 = manager.register_model_version(first_run.run_id)
    manager.transition_model_stage(v1.version, target_stage="Production")

    async def _predict_burst() -> None:
        for idx in range(30):
            await asyncio.to_thread(
                manager.predict_with_ab_routing,
                customer_id=f"cust-{idx:04d}",
                feature_vector=[0.4, 0.3, 0.05, 0.85],
            )

    async def _promote_sequence() -> None:
        for step in range(2, 7):
            run = await asyncio.to_thread(
                manager.train_and_track_experiment,
                run_name=f"v{step}",
                sample_size=120,
                random_seed=step,
            )
            reg = await asyncio.to_thread(manager.register_model_version, run.run_id)
            await asyncio.to_thread(
                manager.transition_model_stage,
                reg.version,
                target_stage="Production",
            )

    await asyncio.gather(_predict_burst(), _promote_sequence())

    versions = manager.list_versions()
    archived = [v for v in versions if v.stage == "Archived"]
    assert len(archived) <= 2
    non_archived_paths = {v.onnx_artifact_path for v in versions if v.stage != "Archived"}
    assert set(manager._ort_sessions.keys()).issubset(non_archived_paths)
    run_dirs_on_disk = list((art_root / "runs").glob("run-*"))
    assert len(run_dirs_on_disk) <= 3


async def test_anchored_audit_chain_bounded_rotation_and_nonblocking_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify O(1) anchored sliding-window audit rotation and off-event-loop telemetry routes."""
    import dataclasses

    from app.core.observability import CryptographicAuditChain

    chain = CryptographicAuditChain(secret_key="bounded-secret-key-2026", max_records=5)
    for idx in range(1, 13):
        chain.append(
            request_id=f"req-{idx}",
            pillar="mlops",
            action="predict",
            payload_hash=f"sha256:{idx:04d}",
            confidence=0.9,
        )

    assert len(chain) == 5
    retained = chain.records()
    assert [r.seq for r in retained] == [8, 9, 10, 11, 12]
    valid, count, detail = chain.verify()
    assert valid is True
    assert count == 5
    assert "anchor_seq=7" in detail

    tampered = dataclasses.replace(retained[2], action="forged_action")
    chain._records[2] = tampered
    valid_after, _, _ = chain.verify()
    assert valid_after is False

    monkeypatch.setenv("OMNIPULSE_ARTIFACT_DIR", str(tmp_path / "nonblock_api"))
    from app.config import reset_settings_cache

    reset_settings_cache()
    app = create_app()
    with (
        patch("app.api.routes.asyncio.to_thread", wraps=asyncio.to_thread) as spy_to_thread,
        TestClient(app) as client,
    ):
        before = spy_to_thread.call_count
        resp_audit = client.get("/api/v1/audit/verify")
        resp_obs = client.get("/api/v1/observability/summary")
        resp_metrics = client.get("/metrics")
        resp_sim_status = client.get("/api/v1/simulate/status")
        assert resp_audit.status_code == 200
        assert resp_obs.status_code == 200
        assert resp_metrics.status_code == 200
        assert resp_sim_status.status_code == 200
        assert spy_to_thread.call_count - before == 4
