"""Geração de predições do modelo de credit scoring.

Para cada solicitação, produz:

- `default_proba`: probabilidade de inadimplência (PD) estimada pelo modelo;
- `score`: score de crédito de 0 a 1000 (`(1 - PD) * 1000`; maior = melhor);
- `decision`: `approve` ou `reject`, aplicando o threshold de custo mínimo;
- `model_version`: versão do modelo que tomou a decisão (rastreabilidade).

O log de predições guarda apenas o `application_id` e a saída do modelo -
sem features nem dados pessoais (LGPD, art. 6º, III - minimização). As
features ficam no lote de origem e podem ser recuperadas pelo ID, por exemplo
para atender a um pedido de revisão de decisão automatizada (LGPD, art. 20).

Uso:
    uv run python -m credit_scoring.model.predict
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_scoring.config import load_config
from credit_scoring.contracts.schemas import INFERENCE_CONTRACT, validate
from credit_scoring.data.load import ID_COLUMN
from credit_scoring.data.preprocess import FEATURES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    """Modelo treinado com os metadados necessários para decidir.

    Attributes:
        pipeline: Pipeline scikit-learn (pré-processamento + XGBoost).
        threshold: Ponto de corte: PD `>=` threshold resulta em recusa.
        version: Identificador da versão do modelo.
    """

    pipeline: Pipeline
    threshold: float
    version: str


def load_model(models_dir: Path | str) -> LoadedModel:
    """Carrega o pipeline e os metadados salvos por `make train`.

    Args:
        models_dir: Diretório com `model.joblib` e `metadata.json`.

    Returns:
        `LoadedModel` pronto para predição.

    Raises:
        FileNotFoundError: Se o modelo ainda não foi treinado.
    """
    models_dir = Path(models_dir)
    meta: dict[str, Any] = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    version = f"{meta['model_name']}@{meta['trained_at']}+{meta['data']['train_sha256'][:8]}"
    return LoadedModel(
        pipeline=joblib.load(models_dir / "model.joblib"),
        threshold=float(meta["threshold"]),
        version=version,
    )


def to_score(default_proba: np.ndarray) -> np.ndarray:
    """Converte probabilidade de default em score de crédito.

    Args:
        default_proba: Probabilidades de inadimplência (0 a 1).

    Returns:
        Scores inteiros de 0 a 1000 (maior = menor risco).
    """
    return np.round((1 - default_proba) * 1000).astype("int64")


def predict(df: pd.DataFrame, model: LoadedModel) -> pd.DataFrame:
    """Gera predições para um lote de solicitações.

    O lote é validado contra o `INFERENCE_CONTRACT` antes da predição:
    o modelo nunca decide sobre dados fora do contrato.

    Args:
        df: Lote no formato do `INFERENCE_CONTRACT`.
        model: Modelo carregado por `load_model`.

    Returns:
        DataFrame com `application_id`, `default_proba`, `score`,
        `decision`, `threshold`, `model_version` e `predicted_at`.

    Raises:
        pandera.errors.SchemaErrors: Se o lote violar o contrato de dados.
    """
    validate(df, INFERENCE_CONTRACT)
    proba = model.pipeline.predict_proba(df[FEATURES])[:, 1].astype("float64")
    return pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].to_numpy(),
            "default_proba": proba.round(6),
            "score": to_score(proba),
            "decision": np.where(proba >= model.threshold, "reject", "approve"),
            "threshold": model.threshold,
            "model_version": model.version,
            "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )


def predict_batches(
    accepted_dir: Path | str, output_dir: Path | str, model: LoadedModel, pattern: str = "prod_*"
) -> pd.DataFrame:
    """Gera predições para todos os lotes aceitos e salva um arquivo por lote.

    Args:
        accepted_dir: Diretório com os lotes aceitos pela ingestão.
        output_dir: Diretório de saída das predições.
        model: Modelo carregado.
        pattern: Padrão (glob, sem extensão) dos lotes a processar.

    Returns:
        Resumo por lote: `batch`, `n`, `mean_proba`, `mean_score` e
        `approval_rate`.

    Raises:
        FileNotFoundError: Se nenhum lote for encontrado.
    """
    accepted_dir, output_dir = Path(accepted_dir), Path(output_dir)
    batches = sorted(accepted_dir.glob(f"{pattern}.parquet"))
    if not batches:
        raise FileNotFoundError(f"Nenhum lote '{pattern}' em {accepted_dir}. Rode `make simulate`.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in batches:
        preds = predict(pd.read_parquet(path), model)
        preds.to_parquet(output_dir / path.name, index=False)
        summary.append(
            {
                "batch": path.stem,
                "n": len(preds),
                "mean_proba": preds["default_proba"].mean(),
                "mean_score": preds["score"].mean(),
                "approval_rate": (preds["decision"] == "approve").mean(),
            }
        )
    return pd.DataFrame(summary)


def main() -> None:
    """Gera predições para os lotes de produção conforme `configs/config.yaml`."""
    cfg = load_config()
    prod = Path(cfg["paths"]["production"])
    model = load_model(cfg["paths"]["models"])
    logger.info("Modelo %s | threshold %.2f", model.version, model.threshold)
    summary = predict_batches(prod / "accepted", prod / "predictions", model)
    table = summary.round(3).to_string(index=False)
    logger.info("Predições salvas em %s\n%s", prod / "predictions", table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
