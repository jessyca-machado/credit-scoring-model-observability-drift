.PHONY: install lock data preprocess ingest demo-contract train mlflow-ui simulate predict detect \
        causal serve traffic pipeline up down observability logs test lint

# Mesmo RUN_ID para todas as etapas de um `make pipeline` (correlação nos logs).
# Atribuição imediata (:=) para o `date` ser avaliado uma única vez.
ifndef RUN_ID
RUN_ID := $(shell date +%Y%m%d-%H%M%S)
endif
export RUN_ID

install:
	uv sync

lock:
	uv lock

data:
	uv run python -m credit_scoring.data.download

preprocess:
	uv run python -m credit_scoring.data.preprocess

ingest:
	uv run python -m credit_scoring.data.ingest $(FILE)

demo-contract:
	uv run python -m credit_scoring.contracts.demo

train:
	uv run python -m credit_scoring.model.train

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

simulate:
	uv run python -m credit_scoring.drift.simulate

predict:
	uv run python -m credit_scoring.model.predict

detect:
	uv run python -m credit_scoring.drift.detect

causal:
	uv run python -m credit_scoring.causal.analysis

serve:
	uv run uvicorn credit_scoring.serving.app:app --reload --port 8000

traffic:
	uv run python -m credit_scoring.serving.traffic --url http://localhost:8000

# Treina apenas se ainda não houver modelo (re-treino é decisão explícita: `make train`)
models/model.joblib:
	$(MAKE) train

# Pipeline completo, instrumentado (logs em logs/pipeline.jsonl, métricas no Pushgateway)
pipeline: data preprocess models/model.joblib simulate predict detect demo-contract
	@echo "Pipeline $(RUN_ID) concluído. Dashboard: http://localhost:3000"

# Sobe a stack, roda o pipeline e gera tráfego na API
observability: up pipeline traffic
	@echo "Grafana: http://localhost:3000 | Prometheus: http://localhost:9090 | Alertas: http://localhost:9090/alerts"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	tail -n 20 -f logs/pipeline.jsonl

test:
	uv run pytest --cov=credit_scoring

lint:
	uv run ruff check src tests
