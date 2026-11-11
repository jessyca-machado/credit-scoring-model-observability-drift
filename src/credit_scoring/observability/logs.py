"""Logs estruturados (JSON Lines) para centralização no Loki.

Cada processo grava dois destinos:

- **console**: formato legível (`NÍVEL mensagem`) para quem roda localmente;
- **arquivo** `logs/<arquivo>.jsonl`: um JSON por linha, coletado pelo
  Grafana Alloy e enviado ao Loki.

Todo registro JSON carrega `run_id` (identifica uma execução de ponta a
ponta) e `stage` (etapa do pipeline em andamento), além de quaisquer campos
passados em `extra=` - por exemplo `event`, `batch`, `rows`, `violations`.

Exemplo:
    >>> logger.info("Lote aceito", extra={"event": "batch_accepted", "batch": "prod_2025_01"})
    {"timestamp": "...", "level": "INFO", "message": "Lote aceito", "run_id": "...",
     "stage": "ingest", "event": "batch_accepted", "batch": "prod_2025_01"}
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

RUN_ID: Final[str] = os.environ.get("RUN_ID") or uuid4().hex[:12]
current_stage: ContextVar[str | None] = ContextVar("current_stage", default=None)

_RESERVED: Final[frozenset[str]] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys() | {"message", "asctime"}
)


class ContextFilter(logging.Filter):
    """Injeta `run_id` e `stage` em todos os registros de log."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Adiciona o contexto da execução ao registro.

        Args:
            record: Registro de log.

        Returns:
            Sempre True (o filtro só enriquece, não descarta).
        """
        record.run_id = RUN_ID
        record.stage = current_stage.get()
        return True


class JsonFormatter(logging.Formatter):
    """Formata registros de log como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa o registro com campos padrão e extras.

        Args:
            record: Registro de log.

        Returns:
            Linha JSON.
        """
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(filename: str = "pipeline.jsonl", level: int = logging.INFO) -> Path:
    """Configura o logging do processo: console legível + arquivo JSON.

    O diretório dos logs vem da variável de ambiente `LOG_DIR` (padrão `logs`).

    Args:
        filename: Nome do arquivo JSON Lines.
        level: Nível mínimo de log.

    Returns:
        Caminho do arquivo de log JSON.
    """
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / filename

    context = ContextFilter()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    file = logging.FileHandler(path, encoding="utf-8")
    file.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for handler in (console, file):
        handler.addFilter(context)
        root.addHandler(handler)
    for noisy in ("urllib3", "httpx", "mlflow", "alembic", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path
