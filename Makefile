.PHONY: install run test coverage lint typecheck pipeline simulate simulate-api docker-up docker-down docker-logs

PYTHON ?= .venv/bin/python

install:
	$(PYTHON) -m pip install -e ".[dev]"

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8017 --reload

test:
	$(PYTHON) -m pytest -v

coverage:
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing --cov-fail-under=95

lint:
	.venv/bin/ruff check app tests dags feature_repo && .venv/bin/ruff format --check app tests dags feature_repo

typecheck:
	.venv/bin/mypy app dags feature_repo

pipeline:
	$(PYTHON) -m app.mlops.dvc_pipeline

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

simulate:
	$(PYTHON) -m app.simulator --mode local --interval 1.0 --drift-every 8

simulate-api:
	$(PYTHON) -m app.simulator --mode api --base-url http://localhost:8017 --interval 1.0 --drift-every 8
