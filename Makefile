.PHONY: install lock data preprocess train mlflow-ui simulate detect causal serve up down test lint

install:
	uv sync

lock:
	uv lock

data:
	uv run python -m credit_scoring.data.download

preprocess:
	uv run python -m credit_scoring.data.preprocess

train:
	uv run python -m credit_scoring.model.train

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

simulate:
	uv run python -m credit_scoring.drift.simulate

detect:
	uv run python -m credit_scoring.drift.detect

causal:
	uv run python -m credit_scoring.causal.analysis

serve:
	uv run uvicorn credit_scoring.serving.app:app --reload --port 8000

up:
	docker compose up -d --build

down:
	docker compose down

test:
	uv run pytest --cov=credit_scoring

lint:
	uv run ruff check src tests
