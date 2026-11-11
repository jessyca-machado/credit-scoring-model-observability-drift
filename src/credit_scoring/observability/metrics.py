"""Métricas das etapas do pipeline, enviadas ao Prometheus via Pushgateway.

Etapas de pipeline são processos curtos (rodam e terminam), então não podem
ser coletadas por *scrape*. Cada etapa empacota suas métricas num registro
próprio e as envia ao Pushgateway ao final, que o Prometheus coleta.

Uso:
    >>> with track_stage("ingest") as stage:
    ...     stage.set("ingestion_batch_rows", 300, "Linhas do lote", batch="prod_2025_01")

Toda etapa publica automaticamente:

- `pipeline_stage_success{stage}`: 1 = sucesso, 0 = falha;
- `pipeline_stage_duration_seconds{stage}`;
- `pipeline_stage_last_run_timestamp_seconds{stage}`;
- `pipeline_stage_last_success_timestamp_seconds{stage}` (só em sucesso).

A observabilidade nunca derruba o pipeline: se o Pushgateway estiver fora do
ar, a falha de envio é apenas registrada em log.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

from credit_scoring.observability.logs import RUN_ID, current_stage

logger = logging.getLogger(__name__)

PUSH_JOB: Final[str] = "credit_scoring_pipeline"
DEFAULT_PUSHGATEWAY: Final[str] = "localhost:9091"
PUSH_TIMEOUT_SECONDS: Final[float] = 2.0


def pushgateway_url() -> str | None:
    """Endereço do Pushgateway.

    Lido da variável de ambiente `PUSHGATEWAY_URL`. Se ela existir e estiver
    vazia, o envio é desativado (útil em testes).

    Returns:
        Endereço `host:porta` ou None se o envio estiver desativado.
    """
    url = os.environ.get("PUSHGATEWAY_URL", DEFAULT_PUSHGATEWAY)
    return url or None


class StageMetrics:
    """Registro de métricas de uma etapa do pipeline.

    Attributes:
        stage: Nome da etapa.
        registry: Registro Prometheus isolado da etapa.
    """

    def __init__(self, stage: str) -> None:
        """Cria o registro da etapa.

        Args:
            stage: Nome da etapa.
        """
        self.stage = stage
        self.registry = CollectorRegistry()
        self._gauges: dict[str, Gauge] = {}

    def set(self, name: str, value: float, documentation: str = "", **labels: str) -> None:
        """Define o valor de um gauge, criando-o na primeira chamada.

        Args:
            name: Nome da métrica.
            value: Valor.
            documentation: Descrição da métrica (usada na criação).
            **labels: Labels da série. Devem ser sempre os mesmos por métrica.
        """
        if name not in self._gauges:
            self._gauges[name] = Gauge(
                name, documentation or name, list(labels), registry=self.registry
            )
        gauge = self._gauges[name]
        (gauge.labels(**labels) if labels else gauge).set(value)

    def push(self) -> bool:
        """Envia as métricas ao Pushgateway.

        Usa `pushadd` (POST): substitui apenas as métricas enviadas, de modo
        que métricas de negócio da última execução bem-sucedida permanecem
        visíveis mesmo quando uma execução falha antes de publicá-las.

        Returns:
            True se o envio foi feito, False se desativado ou indisponível.
        """
        url = pushgateway_url()
        if url is None:
            return False
        try:
            pushadd_to_gateway(
                url,
                job=PUSH_JOB,
                grouping_key={"stage": self.stage},
                registry=self.registry,
                timeout=PUSH_TIMEOUT_SECONDS,
            )
        except OSError as exc:
            logger.warning(
                "Pushgateway indisponível em %s: métricas da etapa %s não enviadas (%s)",
                url,
                self.stage,
                exc,
                extra={"event": "metrics_push_failed"},
            )
            return False
        return True


@contextmanager
def track_stage(stage: str) -> Iterator[StageMetrics]:
    """Instrumenta uma etapa do pipeline com logs e métricas.

    Registra em log o início, o fim (com status e duração) e eventuais
    falhas com stack trace, e publica as métricas no Pushgateway ao final -
    inclusive quando a etapa falha. Exceções são relançadas.

    Args:
        stage: Nome da etapa (ex.: `ingest`, `detect`).

    Yields:
        `StageMetrics` para a etapa registrar métricas de negócio.
    """
    token = current_stage.set(stage)
    metrics = StageMetrics(stage)
    start = time.perf_counter()
    logger.info("Etapa '%s' iniciada (run %s)", stage, RUN_ID, extra={"event": "stage_started"})
    success = False
    try:
        yield metrics
        success = True
    except Exception as exc:
        logger.exception(
            "Etapa '%s' falhou: %s",
            stage,
            exc,
            extra={"event": "stage_failed", "error_type": type(exc).__name__},
        )
        raise
    finally:
        duration = time.perf_counter() - start
        now = time.time()
        metrics.set(
            "pipeline_stage_success", int(success), "1 se a última execução teve sucesso",
            stage=stage,
        )
        metrics.set(
            "pipeline_stage_duration_seconds", duration, "Duração da última execução",
            stage=stage,
        )
        metrics.set(
            "pipeline_stage_last_run_timestamp_seconds", now, "Início da última execução",
            stage=stage,
        )
        if success:
            metrics.set(
                "pipeline_stage_last_success_timestamp_seconds", now,
                "Fim da última execução bem-sucedida", stage=stage,
            )
        status = "success" if success else "failure"
        logger.log(
            logging.INFO if success else logging.ERROR,
            "Etapa '%s' finalizada: %s em %.2fs",
            stage,
            status,
            duration,
            extra={"event": "stage_finished", "status": status, "duration_s": round(duration, 3)},
        )
        metrics.push()
        current_stage.reset(token)
