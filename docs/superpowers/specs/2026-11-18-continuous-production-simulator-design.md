# Short Design — Continuous Production Traffic Simulator (OmniPulse AI)

- **Scoped Size:** Small, self-contained change built on top of the existing `llm-mlops-multimodal-platform` architecture.

## 1. Intent & Why
Enable live end-to-end observation of the OmniPulse AI platform operating as a production-like environment — emitting structured JSON logs (with LGPD PII scrubbing), 3D observability metrics in Prometheus/Grafana, an HMAC-SHA256 cryptographic audit chain, and live decisions across all three pillars without manually crafting request payloads.

## 2. Goals, Non-Goals & Anti-Goals
- **Goals:** Generate realistic synthetic traffic in a continuous loop covering MLOps (Feast + ONNX + Data Drift waves with automated retraining), LLMOps (RAG, semantic cache, PII redaction, prompt-injection blocking, and LoRA/RLHF adaptation), and Multimodal (consistent vs. contradictory audio + image + video inspections), with real-time control via API/Swagger and CLI (`make simulate`).
- **Non-Goals:** Does not replace distributed stress-benchmarking tools (such as Locust or k6) and does not introduce dedicated web frontends beyond Swagger, SSE, and Grafana.
- **Anti-Goals:** Never leak memory during long-running executions (history strictly bounded in an `O(1)` ring buffer), never leave orphaned tasks on FastAPI shutdown, and never auto-start hidden traffic during tests or default server boot without an explicit command.

## 3. Technical Constraints
- Strict adherence to repository engineering standards: Python 3.12+, `mypy --strict`, `ruff` (line length <= 100), zero inline `#` comments in Python code, zero direct `logger.*` calls outside `app/core/observability.py`, and overall test coverage `>= 95%`.

## 4. Evaluated Approaches
1. **Approach 1 (Selected) — Hybrid Engine: Controllable Async Task in the API (`/api/v1/simulate/*` + SSE Stream) + `make simulate` CLI:** Non-blocking async loop (`asyncio.Task` + `asyncio.Event`) wired into `PlatformRuntimeContainer`, controllable via Swagger (`/start`, `/stop`, `/status`, `/stream`), paired with a CLI (`python -m app.simulator`) capable of running in-process or issuing live HTTP + WebSocket (`/ws/v1/multimodal/stream`) traffic against a running server. Delivers full Docker Compose/Grafana observability without requiring container restarts.
2. **Approach 2 — External Load Script/Container Only:** Keeps the API untouched and requires a separate process sending traffic over the network; rejected because it provides no interactive control through Swagger `/docs`.
3. **Approach 3 — Fixed Internal Boot Loop via Environment Variable:** Enables the generator solely via an `ENV` flag during `lifespan`; rejected because pausing or adjusting cadence requires restarting the service.

## 5. User-Defined Decisions
- Execution mode: **continuous loop generator** (rather than a single batch burst).
- Architecture: **Approach 1** (async engine controllable through API endpoints plus CLI support).

## 6. Builder-Defined Defaults (Configurable)
- **Initial state:** Disabled by default on boot (`running=False`), starting only via `POST /api/v1/simulate/start` or the CLI.
- **Cadence & anomaly waves:** Default interval of `1.5s` per cycle (`0.05s` to `60.0s`), triggering a synthetic Data Drift wave (`PSI > 0.20` -> automated MLflow retraining) every `8` cycles (`drift_wave_every=8`).
- **Memory & lifecycle:** Circular buffer `collections.deque(maxlen=200)` for recent events, thread-safe aggregate counters by pillar/outcome, and guaranteed graceful cancellation on both `POST /api/v1/simulate/stop` and `lifespan` teardown in `app/main.py`.
- **CLI modes (`Makefile`):** `make simulate` (runs in-process and streams formatted events to the terminal) and `make simulate-api` (external client exercising REST + WebSocket `/ws/v1/multimodal/stream` against `http://localhost:8017`).
