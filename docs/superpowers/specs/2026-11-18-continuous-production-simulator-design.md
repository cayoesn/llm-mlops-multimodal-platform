# Design Curto — Simulador Contínuo de Tráfego de Produção (OmniPulse AI)

- **Escopo dimensionado:** Mudança pequena e autocontida (*small change*) sobre a arquitetura existente do `llm-mlops-multimodal-platform`.

## 1. Intenção e Motivação (Intent & Why)
Permitir observar a plataforma OmniPulse AI operando fim a fim como um ambiente vivo de produção — gerando logs JSON estruturados (com scrubbing LGPD), métricas 3D no Prometheus/Grafana, cadeia criptográfica HMAC-SHA256 e decisões reais nos 3 pilares sem precisar montar payloads manualmente.

## 2. Objetivos, Não-Objetivos e Anti-Objetivos
- **Objetivos (Goals):** Gerar tráfego sintético realista em loop contínuo cobrindo MLOps (Feast + ONNX + ondas de Data Drift com retreino automático), LLMOps (RAG, cache semântico, redação PII, bloqueio de *prompt injection* e adaptação LoRA/RLHF) e Multimodal (áudio + imagem + vídeo consistentes vs. contraditórios), com controle em tempo real via API/Swagger e via CLI (`make simulate`).
- **Não-Objetivos (Non-Goals):** Não substitui ferramentas de *stress benchmark* distribuído (como Locust/k6) nem cria frontends web dedicados fora do Swagger, SSE e Grafana já existentes.
- **Anti-Objetivos (Anti-Goals):** Jamais vazar memória em execuções longas (histórico estritamente limitado em buffer circular `O(1)`), jamais deixar tasks órfãs no shutdown da FastAPI e jamais iniciar tráfego oculto automaticamente em testes ou no boot padrão sem comando explícito.

## 3. Restrições Técnicas (Constraints)
- Conformidade estrita com o padrão do repositório: Python 3.12+, `mypy --strict`, `ruff` (linha <= 100), zero comentários inline `#` em código Python, zero chamadas diretas a `logger.*` fora de `app/core/observability.py`, e cobertura global de testes `>= 95%`.

## 4. Abordagens Avaliadas
1. **Abordagem 1 (Escolhida) — Motor Híbrido: Task Assíncrona Controlável na API (`/api/v1/simulate/*` + Stream SSE) + CLI `make simulate`:** Loop assíncrono não-bloqueante (`asyncio.Task` + `asyncio.Event`) acoplado ao `PlatformRuntimeContainer`, controlável via Swagger (`/start`, `/stop`, `/status`, `/stream`) e acompanhado de CLI (`python -m app.simulator`) capaz de operar em modo local ou disparando requisições HTTP + WebSocket (`/ws/v1/multimodal/stream`) contra um servidor ativo. Entrega máxima observabilidade no Docker Compose/Grafana sem exigir reinício de containers.
2. **Abordagem 2 — Apenas Script/Container Externo de Carga:** Mantém a API intocada e exige processo separado batendo via rede; descartada por não oferecer controle interativo pelo Swagger `/docs`.
3. **Abordagem 3 — Loop Interno Fixo por Variável de Ambiente no Boot:** Liga o gerador apenas via `ENV` no `lifespan`; descartada por exigir reiniciar o serviço para pausar ou alterar a cadência.

## 5. Decisões Definidas pelo Usuário
- Modo de execução: **gerador contínuo em loop** (em vez de disparo único em lote).
- Arquitetura: **Abordagem 1** (motor assíncrono controlável via endpoints da API + suporte CLI).

## 6. Padrões Definidos pelo Construtor (Defaults Ajustáveis)
- **Estado inicial:** Desligado por padrão no boot (`running=False`), iniciando apenas via `POST /api/v1/simulate/start` ou CLI.
- **Cadência e ondas de anomalia:** Intervalo padrão de `1.5s` por ciclo (`0.1s` a `60.0s`), disparando uma onda sintética de *Data Drift* (`PSI > 0.20` -> retreino automático MLflow) a cada `8` ciclos (`drift_wave_every=8`).
- **Memória e ciclo de vida:** Buffer circular `collections.deque(maxlen=200)` para os últimos eventos, contadores agregados por pilar/status protegidos e cancelamento gracioso garantido tanto em `POST /api/v1/simulate/stop` quanto no encerramento do `lifespan` em `app/main.py`.
- **Modos do CLI (`Makefile`):** `make simulate` (executa e exibe o fluxo contínuo formatado no terminal) e `make simulate-api` (cliente externo exercitando REST + WebSocket `/ws/v1/multimodal/stream` em `http://localhost:8017`).
