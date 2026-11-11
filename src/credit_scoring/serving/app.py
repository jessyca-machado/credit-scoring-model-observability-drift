"""API de credit scoring com métricas Prometheus.

Endpoints:

- `POST /predict`: recebe uma ou mais solicitações, valida contra o
  `INFERENCE_CONTRACT` e retorna PD, score e decisão;
- `GET /health`: estado da API e versão do modelo;
- `GET /metrics`: métricas no formato Prometheus.

Métricas expostas:

- `api_requests_total{endpoint, status_code}` e
  `api_request_latency_seconds{endpoint}` (histograma);
- `model_predictions_total{decision}`;
- `model_score` e `model_default_proba` (histogramas de saída do modelo);
- `api_contract_violations_total{check}`: solicitações barradas pelo contrato;
- `model_info{version, threshold}`: versão do modelo carregado.

Uso:
    uv run uvicorn credit_scoring.serving.app:app --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from credit_scoring.contracts.schemas import INFERENCE_CONTRACT, contract_report
from credit_scoring.data.load import CATEGORY_MAPS, category_dtype
from credit_scoring.data.preprocess import add_features
from credit_scoring.model.predict import LoadedModel, load_model, predict
from credit_scoring.observability.logs import setup_logging

logger = logging.getLogger(__name__)

CATEGORICAL_COLUMNS: list[str] = [*CATEGORY_MAPS, "sex", "marital_status"]
CallNext = Callable[[Request], Awaitable[Response]]

REQUESTS = Counter(
    "api_requests_total", "Requisições recebidas", ["endpoint", "status_code"]
)
LATENCY = Histogram(
    "api_request_latency_seconds", "Latência das requisições", ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
PREDICTIONS = Counter("model_predictions_total", "Decisões do modelo", ["decision"])
SCORE = Histogram(
    "model_score", "Distribuição do score (0-1000)", buckets=tuple(range(0, 1001, 100))
)
PROBA = Histogram(
    "model_default_proba", "Distribuição da PD prevista",
    buckets=(0.05, 0.1, 0.11, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0),
)
CONTRACT_VIOLATIONS = Counter(
    "api_contract_violations_total", "Violações do contrato de dados na API", ["check"]
)
MODEL_INFO = Gauge("model_info", "Modelo carregado (valor sempre 1)", ["version", "threshold"])


class Application(BaseModel):
    """Solicitação de crédito. `monthly_installment` é calculado pela API."""

    application_id: str = Field(examples=["APP-90000001"])
    duration_months: int = Field(examples=[24])
    credit_amount: int = Field(examples=[3500])
    installment_rate: int = Field(examples=[3])
    residence_since: int = Field(examples=[2])
    existing_credits: int = Field(examples=[1])
    num_dependents: int = Field(examples=[1])
    checking_status: str = Field(examples=["no checking account"])
    credit_history: str = Field(examples=["existing credits paid duly"])
    purpose: str = Field(examples=["car (used)"])
    savings_status: str = Field(examples=["< 100 DM"])
    employment_since: str = Field(examples=["1 <= ... < 4 years"])
    other_debtors: str = Field(examples=["none"])
    property: str = Field(examples=["car or other"])
    other_installment_plans: str = Field(examples=["none"])
    housing: str = Field(examples=["own"])
    job: str = Field(examples=["skilled employee / official"])
    own_telephone: str = Field(examples=["yes"])
    foreign_worker: str = Field(examples=["no"])
    age: int = Field(examples=[35])
    sex: str = Field(examples=["female"])
    marital_status: str = Field(examples=["single"])


class Prediction(BaseModel):
    """Resultado da avaliação de crédito."""

    application_id: str
    default_proba: float
    score: int
    decision: str
    model_version: str


def to_frame(applications: list[Application]) -> pd.DataFrame:
    """Converte as solicitações no formato exigido pelo contrato.

    Colunas categóricas recebem o dtype do treino apenas quando todos os
    valores pertencem ao domínio; caso contrário permanecem texto, para que
    o contrato reporte o valor inválido (ex.: `purpose = crypto`).

    Args:
        applications: Solicitações recebidas.

    Returns:
        DataFrame no formato do `INFERENCE_CONTRACT`.
    """
    df = add_features(pd.DataFrame([a.model_dump() for a in applications]))
    for col in CATEGORICAL_COLUMNS:
        dtype = category_dtype(col)
        if df[col].isin(dtype.categories).all():
            df[col] = df[col].astype(dtype)
    return df


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo uma única vez na subida da API.

    Args:
        app: Aplicação FastAPI.

    Yields:
        Controle para a aplicação em execução.
    """
    if not logging.getLogger().handlers:
        setup_logging("api.jsonl")
    model = load_model(os.environ.get("MODELS_DIR", "models"))
    app.state.model = model
    MODEL_INFO.labels(version=model.version, threshold=str(model.threshold)).set(1)
    logger.info("Modelo carregado: %s", model.version, extra={"event": "model_loaded"})
    yield


app = FastAPI(title="Credit Scoring API", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe(request: Request, call_next: CallNext) -> Response:
    """Mede latência e contabiliza requisições por endpoint e status.

    Args:
        request: Requisição recebida.
        call_next: Próximo handler.

    Returns:
        Resposta do handler.
    """
    endpoint = request.url.path
    if endpoint.startswith("/metrics"):
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    REQUESTS.labels(endpoint=endpoint, status_code=str(response.status_code)).inc()
    return response


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Informa se a API está pronta e qual modelo está carregado.

    Args:
        request: Requisição (dá acesso ao modelo carregado).

    Returns:
        Status e versão do modelo.
    """
    model: LoadedModel = request.app.state.model
    return {"status": "ok", "model_version": model.version, "threshold": model.threshold}


@app.post("/predict")
def predict_endpoint(applications: list[Application], request: Request) -> list[Prediction]:
    """Avalia solicitações de crédito.

    Args:
        applications: Uma ou mais solicitações.
        request: Requisição (dá acesso ao modelo carregado).

    Returns:
        Uma predição por solicitação.

    Raises:
        HTTPException: 422 se as solicitações violarem o contrato de dados.
    """
    model: LoadedModel = request.app.state.model
    df = to_frame(applications)
    report = contract_report(df, INFERENCE_CONTRACT)
    if not report.empty:
        for check, n in report["check"].map(lambda c: str(c)[:60]).value_counts().items():
            CONTRACT_VIOLATIONS.labels(check=check).inc(n)
        violations = report.assign(failure_case=report["failure_case"].map(str)).to_dict("records")
        logger.warning(
            "Requisição bloqueada pelo contrato: %d violações", len(report),
            extra={"event": "api_contract_violation", "violations": len(report)},
        )
        raise HTTPException(status_code=422, detail={"contract_violations": violations})

    preds = predict(df, model)
    for row in preds.itertuples():
        PREDICTIONS.labels(decision=row.decision).inc()
        SCORE.observe(row.score)
        PROBA.observe(row.default_proba)
    logger.info(
        "%d solicitações avaliadas", len(preds),
        extra={
            "event": "predictions_served",
            "rows": len(preds),
            "approved": int((preds["decision"] == "approve").sum()),
        },
    )
    return [
        Prediction(
            application_id=r.application_id,
            default_proba=r.default_proba,
            score=r.score,
            decision=r.decision,
            model_version=r.model_version,
        )
        for r in preds.itertuples()
    ]
