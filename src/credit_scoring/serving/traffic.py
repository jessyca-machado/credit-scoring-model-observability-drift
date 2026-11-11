"""Gerador de tráfego para a API, a partir dos lotes de produção.

Envia as solicitações dos lotes aceitos para `POST /predict` em pequenos
grupos e, a cada `--bad-every` requisições, uma requisição com erros
(menor de idade, categoria inexistente), para exercitar as métricas de
violação de contrato.

Uso:
    uv run python -m credit_scoring.serving.traffic --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from credit_scoring.observability.logs import setup_logging

logger = logging.getLogger(__name__)


def to_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converte um lote no corpo JSON aceito pela API.

    Args:
        df: Lote no formato do `INFERENCE_CONTRACT`.

    Returns:
        Lista de solicitações (sem `monthly_installment`, calculado pela API).
    """
    records = df.drop(columns=["monthly_installment"]).astype(object).to_dict("records")
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in records]


def post(url: str, payload: list[dict[str, Any]]) -> int:
    """Envia uma requisição para `/predict`.

    Args:
        url: URL base da API.
        payload: Solicitações.

    Returns:
        Código HTTP da resposta.
    """
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def run(url: str, accepted_dir: Path, chunk: int, delay: float, bad_every: int) -> dict[int, int]:
    """Envia todos os lotes aceitos para a API.

    Args:
        url: URL base da API.
        accepted_dir: Diretório com os lotes aceitos.
        chunk: Solicitações por requisição.
        delay: Pausa entre requisições (segundos).
        bad_every: A cada quantas requisições enviar uma com erros (0 = nunca).

    Returns:
        Contagem de respostas por código HTTP.
    """
    status: dict[int, int] = {}
    n = 0
    for path in sorted(accepted_dir.glob("prod_*.parquet")):
        payload = to_payload(pd.read_parquet(path))
        for i in range(0, len(payload), chunk):
            body = payload[i : i + chunk]
            n += 1
            if bad_every and n % bad_every == 0:
                body = [dict(body[0], age=16, purpose="crypto"), *body[1:]]
            code = post(url, body)
            status[code] = status.get(code, 0) + 1
            time.sleep(delay)
        logger.info("Lote %s enviado | respostas até agora: %s", path.stem, status)
    return status


def main() -> None:
    """Ponto de entrada da linha de comando."""
    parser = argparse.ArgumentParser(description="Gera tráfego para a API de credit scoring")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--accepted-dir", type=Path, default=Path("data/production/accepted"))
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--bad-every", type=int, default=25)
    args = parser.parse_args()
    status = run(args.url, args.accepted_dir, args.chunk, args.delay, args.bad_every)
    logger.info("Tráfego finalizado: %s", status)


if __name__ == "__main__":
    setup_logging("traffic.jsonl")
    main()
