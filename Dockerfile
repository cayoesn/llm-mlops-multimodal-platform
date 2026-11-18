FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMNIPULSE_ENV=production \
    OMNIPULSE_API_PORT=8017 \
    OMNIPULSE_ARTIFACT_DIR=/var/lib/omnipulse/runtime

RUN groupadd --gid 10001 omnipulse && \
    useradd --uid 10001 --gid omnipulse --create-home --shell /usr/sbin/nologin omnipulse && \
    mkdir -p /app /var/lib/omnipulse/runtime && \
    chown -R omnipulse:omnipulse /app /var/lib/omnipulse

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

COPY --chown=omnipulse:omnipulse pyproject.toml dvc.yaml params.yaml MLproject ./
COPY --chown=omnipulse:omnipulse app ./app
COPY --chown=omnipulse:omnipulse feature_repo ./feature_repo
COPY --chown=omnipulse:omnipulse dags ./dags

USER omnipulse

EXPOSE 8017

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8017/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8017", "--workers", "1"]
