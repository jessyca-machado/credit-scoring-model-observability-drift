"""Ingestão de lotes de produção protegida pelo contrato de dados.

Todo lote que chega em produção passa pelo `INFERENCE_CONTRACT` antes de ser
disponibilizado para o modelo:

- **válido** → movido para `data/production/accepted/`;
- **inválido** → movido para `data/production/quarantine/`, junto com um
  relatório de violações (`<lote>.violations.csv`). O lote inteiro é
  bloqueado: nenhuma linha dele chega ao modelo.

Uso:
    uv run python -m credit_scoring.data.ingest data/production/incoming/lote.parquet

O processo termina com código de saída 1 se o lote for rejeitado, permitindo
que orquestradores (Airflow, CI, Makefile) interrompam o pipeline.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from credit_scoring.contracts.schemas import INFERENCE_CONTRACT, contract_report
from credit_scoring.observability.logs import setup_logging
from credit_scoring.observability.metrics import StageMetrics, track_stage

logger = logging.getLogger(__name__)

PRODUCTION_DIR: Final[Path] = Path("data/production")


@dataclass(frozen=True)
class IngestionResult:
    """Resultado da ingestão de um lote.

    Attributes:
        batch: Nome do arquivo do lote.
        accepted: Se o lote passou no contrato.
        n_rows: Quantidade de linhas do lote.
        n_violations: Quantidade de violações encontradas.
        destination: Caminho final do lote (`accepted/` ou `quarantine/`).
        report: Caminho do relatório de violações, se houver.
    """

    batch: str
    accepted: bool
    n_rows: int
    n_violations: int
    destination: Path
    report: Path | None = None


def summarize(report: pd.DataFrame, max_examples: int = 3) -> pd.DataFrame:
    """Resume as violações por coluna e regra.

    Args:
        report: Saída de `contract_report`.
        max_examples: Quantidade de valores inválidos exibidos por regra.

    Returns:
        DataFrame com `column`, `check`, `violations` e `examples`.
    """
    checks = report["check"].map(str)
    return (
        report.assign(
            failure_case=report["failure_case"].map(str),
            check=checks.where(checks.str.len() <= 40, checks.str[:37] + "..."),
        )
        .groupby(["column", "check"], sort=False)["failure_case"]
        .agg(violations="size", examples=lambda s: ", ".join(s.unique()[:max_examples]))
        .reset_index()
    )


def record_ingestion(
    metrics: StageMetrics, result: IngestionResult, report: pd.DataFrame
) -> None:
    """Registra as métricas de ingestão de um lote.

    Args:
        metrics: Registro de métricas da etapa em andamento.
        result: Resultado da ingestão.
        report: Violações encontradas (vazio se aceito).
    """
    batch = Path(result.batch).stem
    metrics.set(
        "ingestion_batch_accepted", int(result.accepted),
        "1 se o lote passou no contrato de dados, 0 se foi bloqueado", batch=batch,
    )
    metrics.set("ingestion_batch_rows", result.n_rows, "Linhas recebidas no lote", batch=batch)
    metrics.set(
        "ingestion_batch_violations", result.n_violations,
        "Total de violações do contrato no lote", batch=batch,
    )
    if not report.empty:
        counts = report.groupby(["column", "check"]).size()
        for (column, check), n in counts.items():
            metrics.set(
                "ingestion_contract_violations", int(n),
                "Violações do contrato por coluna e regra",
                batch=batch, column=str(column), check=str(check)[:60],
            )


def ingest_batch(
    path: Path | str,
    production_dir: Path | str = PRODUCTION_DIR,
    metrics: StageMetrics | None = None,
) -> IngestionResult:
    """Valida um lote contra o contrato e o encaminha para aceite ou quarentena.

    Args:
        path: Caminho do lote em Parquet.
        production_dir: Diretório base de produção.
        metrics: Se informado, registra as métricas de ingestão do lote.

    Returns:
        `IngestionResult` com a decisão e os caminhos de destino.

    Raises:
        FileNotFoundError: Se o lote não existir.
    """
    path, production_dir = Path(path), Path(production_dir)
    df = pd.read_parquet(path)
    report = contract_report(df, INFERENCE_CONTRACT)

    if report.empty:
        dest_dir = production_dir / "accepted"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        shutil.move(path, dest)
        logger.info(
            "Lote %s ACEITO (%d linhas) -> %s", path.name, len(df), dest,
            extra={"event": "batch_accepted", "batch": path.stem, "rows": len(df)},
        )
        result = IngestionResult(path.name, True, len(df), 0, dest)
        if metrics is not None:
            record_ingestion(metrics, result, report)
        return result

    dest_dir = production_dir / "quarantine"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    report_path = dest_dir / f"{path.stem}.violations.csv"
    shutil.move(path, dest)
    report.to_csv(report_path, index=False)

    logger.error(
        "Lote %s BLOQUEADO: %d violações do contrato '%s'. Movido para %s\n%s",
        path.name,
        len(report),
        INFERENCE_CONTRACT.name,
        dest,
        summarize(report).to_string(index=False),
        extra={
            "event": "batch_rejected",
            "batch": path.stem,
            "rows": len(df),
            "violations": len(report),
            "violated_rules": sorted(set(report["check"].map(str).str[:60])),
        },
    )
    result = IngestionResult(path.name, False, len(df), len(report), dest, report_path)
    if metrics is not None:
        record_ingestion(metrics, result, report)
    return result


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da linha de comando.

    Args:
        argv: Argumentos (padrão: `sys.argv[1:]`).

    Returns:
        0 se o lote foi aceito, 1 se foi bloqueado.
    """
    parser = argparse.ArgumentParser(description="Ingere um lote validando o contrato de dados")
    parser.add_argument("batch", type=Path, help="Caminho do lote (.parquet)")
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION_DIR)
    args = parser.parse_args(argv)
    with track_stage("ingest") as metrics:
        result = ingest_batch(args.batch, args.production_dir, metrics)
    return 0 if result.accepted else 1


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
