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


def ingest_batch(path: Path | str, production_dir: Path | str = PRODUCTION_DIR) -> IngestionResult:
    """Valida um lote contra o contrato e o encaminha para aceite ou quarentena.

    Args:
        path: Caminho do lote em Parquet.
        production_dir: Diretório base de produção.

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
        logger.info("Lote %s ACEITO (%d linhas) -> %s", path.name, len(df), dest)
        return IngestionResult(path.name, True, len(df), 0, dest)

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
    )
    return IngestionResult(path.name, False, len(df), len(report), dest, report_path)


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
    result = ingest_batch(args.batch, args.production_dir)
    return 0 if result.accepted else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
