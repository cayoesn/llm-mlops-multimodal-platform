# Continuous Production Traffic Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid continuous production traffic simulator (`app/simulator.py`, `/api/v1/simulate/*` REST + SSE endpoints, and `make simulate` / `make simulate-api` CLI modes) that exercises MLOps, LLMOps, and Multimodal pipelines end-to-end with live structured logs, Prometheus/Grafana metrics, and HMAC-SHA256 audit records.

**Architecture:** `ProductionTrafficSimulator` runs an `asyncio.Task` controlled by a late-bound `asyncio.Event` and a single `threading.RLock` inside `PlatformRuntimeContainer`, cycling an 8-scenario synthetic workload plus periodic PSI Data Drift retraining waves into a bounded `collections.deque(maxlen=200)` ring buffer. FastAPI exposes `/start`, `/stop`, `/status`, and `/stream` (SSE) while cleaning up on `lifespan` shutdown, and `python -m app.simulator` supports both in-process streaming and external HTTP + WebSocket load generation.

**Tech Stack:** Python 3.12, FastAPI, AsyncIO, Pydantic v2, NumPy, Pillow, `httpx`, `websockets`, Prometheus/OpenMetrics, `pytest` / `pytest-asyncio`, `ruff`, `mypy --strict`.

**Spec:** `docs/superpowers/specs/2026-11-18-continuous-production-simulator-design.md`

## Global Constraints

- Python 3.12+, `mypy --strict`, `ruff` (`line-length = 100`, rules `E, F, I, UP, B`).
- Zero inline `#` comments in any `.py` file; formal English docstrings only.
- Zero direct `logger.*` calls outside `app/core/observability.py`; use `log_info`, `log_warning`, `log_error(..., exc=exc)`, `log_event`.
- Zero unbounded memory growth: recent events stored in `collections.deque(maxlen=max_history)` (`default=200`), synchronized with a single `threading.RLock` never held across `await`.
- Disabled by default on boot (`running=False`); graceful cancellation guaranteed on `POST /api/v1/simulate/stop` and FastAPI `lifespan` exit.
- Overall `pytest --cov=app` coverage `>= 95%`.

## Review Focus

1. Calling `stop()` (or exiting FastAPI `lifespan`) while `interval_seconds=60.0` must wake `stop_event.wait()` immediately (`< 0.2s`) instead of hanging on `asyncio.sleep`.
2. Calling `POST /api/v1/simulate/start` repeatedly while `running=True` must update `interval_seconds` and `drift_wave_every` in-place without spawning duplicate background `asyncio.Task`s.
3. Setting `drift_wave_every` to an even divisor (e.g., `2` or `4`) must advance a separate `_scenario_cursor` so none of the 8 normal scenario slots are starved.
4. Running more ticks than `max_history` must cap `len(recent_events) == max_history` in `O(1)` time and `O(max_history)` space.
5. `GET /api/v1/simulate/stream?max_events=N` must emit valid SSE `data: {...}\n\n` frames and close cleanly once `N` events are delivered without busy-looping CPU.

---

### Task 1: Simulator Schemas & Core `ProductionTrafficSimulator` Engine

**Files:**
- Modify: `app/api/schemas.py`
- Create: `app/simulator.py`
- Test: `tests/test_production_simulator.py`

**Interfaces:**
- Consumes: `PlatformRuntimeContainer` protocol/duck-type (`dvc_runner`, `drift_engine`, `llm_gateway`, `rlhf_loop`, `multimodal_engine`), `issue_voice_consent_token` (`app/multimodal/audio_pipeline.py`), `GuardrailViolationError` (`app/core/exceptions.py`).
- Produces:
  - In `app/api/schemas.py`:
    - `SimulationEventRecord(BaseModel)`: `tick: int`, `timestamp_utc: str`, `pillar: Literal["mlops", "llmops", "multimodal"]`, `scenario: str`, `outcome: str`, `latency_ms: float`, `detail: dict[str, Any]`
    - `SimulationStartRequest(BaseModel)`: `interval_seconds: float = Field(default=1.5, ge=0.05, le=60.0)`, `drift_wave_every: int = Field(default=8, ge=1, le=1000)`
    - `SimulationStatusResponse(BaseModel)`: `running: bool`, `interval_seconds: float`, `drift_wave_every: int`, `total_ticks: int`, `pillar_counts: dict[str, int]`, `outcome_counts: dict[str, int]`, `recent_events: list[SimulationEventRecord]`
  - In `app/simulator.py`:
    - `class ProductionTrafficSimulator`:
      - `__init__(self, container: Any, *, max_history: int = 200) -> None`
      - `async def execute_tick(self) -> SimulationEventRecord`
      - `async def start(self, *, interval_seconds: float = 1.5, drift_wave_every: int = 8) -> SimulationStatusResponse`
      - `async def stop(self) -> SimulationStatusResponse`
      - `def status(self, *, limit_recent: int = 25) -> SimulationStatusResponse`
      - `async def stream_events(self, *, max_events: int | None = None, poll_interval: float = 0.05) -> AsyncIterator[SimulationEventRecord]`

- [ ] **Step 1: Write failing unit tests in `tests/test_production_simulator.py` for engine lifecycle, all 8 scenarios + drift wave, idempotent `start()`, immediate `stop()` wakeup, non-starvation, and `deque` cap**

```python
async def test_simulator_executes_all_pillars_drift_wave_and_caps_history(tmp_path: Path) -> None:
    container = PlatformRuntimeContainer()
    await asyncio.to_thread(container.bootstrap_mlops)
    sim = ProductionTrafficSimulator(container, max_history=5)
    for _ in range(9):
        await sim.execute_tick()
    st = sim.status(limit_recent=10)
    assert st.total_ticks == 9
    assert set(st.pillar_counts.keys()) == {"mlops", "llmops", "multimodal"}
    assert st.outcome_counts.get("retrained", 0) >= 1
    assert st.outcome_counts.get("blocked_guardrail", 0) >= 1
    assert len(st.recent_events) == 5
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run: `.venv/bin/pytest tests/test_production_simulator.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError: No module named 'app.simulator'`

- [ ] **Step 3: Implement schemas in `app/api/schemas.py` and `ProductionTrafficSimulator` in `app/simulator.py`**

Use `@functools.lru_cache(maxsize=4)` for synthetic RGB PNG and 3-frame video fixtures, `threading.RLock` for state snapshots/counters, `collections.deque(maxlen=max_history)` for event history, separate `_scenario_cursor` vs `_tick_counter`, and `asyncio.wait_for(stop_event.wait(), timeout=interval)` for `< 1ms` stop wakeup.

- [ ] **Step 4: Run `pytest` on `tests/test_production_simulator.py` to verify GREEN**

Run: `.venv/bin/pytest tests/test_production_simulator.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add app/api/schemas.py app/simulator.py tests/test_production_simulator.py
git commit -m "feat(simulator): implement async multi-pillar production traffic simulator engine"
```

---

### Task 2: FastAPI `/api/v1/simulate/*` Control & SSE Endpoints + Lifespan Cleanup

**Files:**
- Modify: `app/api/routes.py`
- Modify: `app/main.py`
- Test: `tests/test_production_simulator.py`

**Interfaces:**
- Consumes: `ProductionTrafficSimulator`, `SimulationStartRequest`, `SimulationStatusResponse` from Task 1.
- Produces:
  - `PlatformRuntimeContainer.simulator: ProductionTrafficSimulator` in `app/api/routes.py`
  - `POST /api/v1/simulate/start` -> `SimulationStatusResponse`
  - `POST /api/v1/simulate/stop` -> `SimulationStatusResponse`
  - `GET /api/v1/simulate/status?limit_recent=25` -> `SimulationStatusResponse`
  - `GET /api/v1/simulate/stream?max_events=...` -> `StreamingResponse` (`media_type="text/event-stream"`)
  - `await container.simulator.stop()` in `app/main.py` `lifespan` teardown before `await container.llm_gateway.aclose()`.

- [ ] **Step 1: Add failing API + SSE + lifespan teardown tests to `tests/test_production_simulator.py`**

```python
def test_simulate_api_start_status_sse_stream_stop_and_lifespan_cleanup(api_client: TestClient) -> None:
    st0 = api_client.get("/api/v1/simulate/status")
    assert st0.status_code == 200
    assert st0.json()["running"] is False

    started = api_client.post(
        "/api/v1/simulate/start",
        json={"interval_seconds": 0.05, "drift_wave_every": 4},
    )
    assert started.status_code == 200
    assert started.json()["running"] is True

    with api_client.stream("GET", "/api/v1/simulate/stream", params={"max_events": 3}) as sse:
        assert sse.status_code == 200
        assert "text/event-stream" in sse.headers["content-type"]
        data_lines = [line for line in sse.iter_lines() if line.startswith("data: ")]
        assert len(data_lines) == 3

    stopped = api_client.post("/api/v1/simulate/stop")
    assert stopped.status_code == 200
    assert stopped.json()["running"] is False
    assert stopped.json()["total_ticks"] >= 3
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run: `.venv/bin/pytest tests/test_production_simulator.py -v`
Expected: FAIL (`404 Not Found` on `/api/v1/simulate/status`)

- [ ] **Step 3: Wire `simulator` in `PlatformRuntimeContainer`, add `/api/v1/simulate/{start,stop,status,stream}` in `app/api/routes.py`, and add `await container.simulator.stop()` in `app/main.py`**

Ensure `GET /api/v1/simulate/stream` checks `await request.is_disconnected()` and formats each `SimulationEventRecord` as `f"data: {event.model_dump_json()}\n\n"`.

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run: `.venv/bin/pytest tests/test_production_simulator.py tests/test_api_and_websocket.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add app/api/routes.py app/main.py tests/test_production_simulator.py
git commit -m "feat(api): expose live simulation start/stop/status/SSE routes and lifespan teardown"
```

---

### Task 3: CLI Modes (`local` & `api` HTTP+WebSocket), `Makefile` Targets & Documentation

**Files:**
- Modify: `app/simulator.py`
- Modify: `Makefile`
- Modify: `README.md`
- Test: `tests/test_production_simulator.py`

**Interfaces:**
- Consumes: `ProductionTrafficSimulator`, FastAPI REST & WebSocket endpoints (`/api/v1/*`, `/ws/v1/multimodal/stream`).
- Produces:
  - `async def run_local_cli_simulation(*, interval_seconds: float = 1.0, drift_wave_every: int = 8, max_cycles: int = 0, output_stream: TextIO | None = None) -> SimulationStatusResponse`
  - `async def run_api_client_simulation(*, base_url: str = "http://localhost:8017", interval_seconds: float = 1.0, max_cycles: int = 0, http_client: httpx.AsyncClient | None = None, enable_websocket: bool = True, output_stream: TextIO | None = None) -> int`
  - `def main(argv: Sequence[str] | None = None) -> int` (`python -m app.simulator --mode {local,api} ...`)
  - `Makefile` targets: `simulate` and `simulate-api`
  - `README.md` section documenting live traffic simulation via Swagger/SSE, CLI, and Docker Compose + Grafana.

- [ ] **Step 1: Add failing CLI (`local`, `api`, and `main(argv)`) tests to `tests/test_production_simulator.py`**

```python
async def test_cli_local_and_api_modes_execute_bounded_cycles(tmp_path: Path) -> None:
    buf = io.StringIO()
    summary = await run_local_cli_simulation(
        interval_seconds=0.05,
        drift_wave_every=2,
        max_cycles=3,
        output_stream=buf,
    )
    assert summary.total_ticks == 3
    assert "[tick=1]" in buf.getvalue()
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run: `.venv/bin/pytest tests/test_production_simulator.py -v`
Expected: FAIL (`ImportError: cannot import name 'run_local_cli_simulation'`)

- [ ] **Step 3: Implement CLI functions + `if __name__ == "__main__":` entrypoint in `app/simulator.py`, add `simulate` & `simulate-api` to `Makefile`, and document usage in `README.md`**

Use `httpx.AsyncClient` connection pooling and `websockets.connect` (when `enable_websocket=True`) for `--mode api`, formatting human-readable progress lines to `output_stream` (`sys.stdout` default) while emitting structured `log_info` telemetry.

- [ ] **Step 4: Run full verification suite (`ruff`, `mypy`, zero-comments check, zero-direct-loggers check, `pytest --cov=app`, and `docker build`)**

Run:
```bash
.venv/bin/ruff format app tests dags feature_repo
.venv/bin/ruff check app tests dags feature_repo
.venv/bin/mypy app dags feature_repo
.venv/bin/pytest tests -v --cov=app --cov-report=term-missing --cov-fail-under=95
docker build -t llm-mlops-multimodal-platform:latest .
```
Expected: 0 lint/type issues, 0 inline comments, 0 direct logger calls, all tests PASS with `>= 95%` coverage, Docker image builds cleanly.

- [ ] **Step 5: Commit & Push to `origin/main`**

```bash
git add app/simulator.py Makefile README.md docs/superpowers/plans/2026-11-18-continuous-production-simulator.md tests/test_production_simulator.py
git commit -m "feat(cli): add local and HTTP/WebSocket traffic simulator CLI, Makefile targets, and docs"
git push origin main
```
