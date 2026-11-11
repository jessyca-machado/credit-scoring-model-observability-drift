"""Demonstração do contrato de dados bloqueando um lote com erros.

Gera dois lotes de produção a partir do conjunto de teste:

- `batch_ok.parquet`: amostra limpa, deve ser **aceita**;
- `batch_corrupted.parquet`: mesma amostra com erros injetados, deve ser
  **bloqueada** e enviada para quarentena.

Uso:
    uv run python -m credit_scoring.contracts.demo
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from credit_scoring.data.ingest import IngestionResult, ingest_batch
from credit_scoring.data.load import ID_COLUMN, TARGET
from credit_scoring.observability.logs import setup_logging
from credit_scoring.observability.metrics import StageMetrics, track_stage

logger = logging.getLogger(__name__)

TEST_PATH: Final[Path] = Path("data/processed/test.parquet")
INCOMING_DIR: Final[Path] = Path("data/production/incoming")

# Erros injetados: (descrição, regra do contrato que deve barrar)
INJECTED_ERRORS: Final[list[tuple[str, str]]] = [
    ("cliente menor de idade (age = 16)", "age: in_range(18, 100)"),
    ("renda/valor do crédito nulo (credit_amount = NaN)", "credit_amount: not_nullable"),
    ("valor do crédito negativo (credit_amount = -500)", "credit_amount: greater_than(0)"),
    ("categoria inexistente (purpose = 'crypto')", "purpose: isin"),
    ("ID duplicado", "application_id: field_uniqueness"),
    ("solicitação duplicada com outro ID", "<row>: no_duplicate_records"),
    ("parcela inconsistente com valor / prazo", "<row>: monthly_installment_consistent"),
    ("sexo + estado civil inexistente", "<row>: valid_sex_marital_status_pair"),
    ("coluna não prevista com dado pessoal (cpf)", "column_in_schema"),
]


def make_clean_batch(n: int = 50, seed: int = 7) -> pd.DataFrame:
    """Amostra um lote limpo do conjunto de teste, sem o alvo (rótulo tardio).

    Args:
        n: Quantidade de solicitações.
        seed: Semente da amostragem.

    Returns:
        Lote no formato esperado pelo `INFERENCE_CONTRACT`.
    """
    test = pd.read_parquet(TEST_PATH)
    return test.sample(n, random_state=seed).drop(columns=TARGET).reset_index(drop=True)


def corrupt(batch: pd.DataFrame) -> pd.DataFrame:
    """Injeta um erro de cada tipo em linhas distintas do lote.

    Args:
        batch: Lote limpo (saída de `make_clean_batch`), com ao menos 10 linhas.

    Returns:
        Novo lote com os erros descritos em `INJECTED_ERRORS`.
    """
    bad = batch.copy()
    bad["credit_amount"] = bad["credit_amount"].astype("float64")  # permite NaN
    bad["purpose"] = bad["purpose"].astype(str)  # permite categoria fora do domínio

    bad.loc[0, "age"] = 16
    bad.loc[1, "credit_amount"] = np.nan
    bad.loc[2, "credit_amount"] = -500
    bad.loc[3, "purpose"] = "crypto"
    bad.loc[4, ID_COLUMN] = bad.loc[5, ID_COLUMN]
    bad.loc[6, "monthly_installment"] = bad.loc[6, "monthly_installment"] * 10
    bad.loc[7, ["sex", "marital_status"]] = ["female", "married/widowed"]

    duplicate = bad.iloc[[8]].assign(**{ID_COLUMN: "APP-99999"})
    bad = pd.concat([bad, duplicate], ignore_index=True)
    bad["cpf"] = "000.000.000-00"
    return bad


def run(metrics: StageMetrics | None = None) -> list[IngestionResult]:
    """Gera os dois lotes e executa a ingestão de ambos.

    Args:
        metrics: Se informado, registra as métricas de ingestão dos lotes.

    Returns:
        Resultados da ingestão do lote limpo e do lote corrompido.
    """
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    clean = make_clean_batch()
    ok_path = INCOMING_DIR / "batch_ok.parquet"
    bad_path = INCOMING_DIR / "batch_corrupted.parquet"
    clean.to_parquet(ok_path, index=False)
    corrupt(clean).to_parquet(bad_path, index=False)

    logger.info("Erros injetados em %s:", bad_path.name)
    for desc, rule in INJECTED_ERRORS:
        logger.info("  - %-50s -> %s", desc, rule)

    return [ingest_batch(ok_path, metrics=metrics), ingest_batch(bad_path, metrics=metrics)]


def main() -> None:
    """Executa a demonstração e exibe o resultado final."""
    with track_stage("contract_demo") as metrics:
        results = run(metrics)
    for r in results:
        status = "ACEITO   " if r.accepted else "BLOQUEADO"
        logger.info(
            "%s | %-24s | %3d linhas | %2d violações | %s",
            status,
            r.batch,
            r.n_rows,
            r.n_violations,
            r.report or r.destination,
        )


if __name__ == "__main__":
    setup_logging()
    main()
