"""Detecção de drift e degradação com Evidently AI.

Para cada lote mensal de produção, gera dois relatórios Evidently:

1. **Data drift** (`DataDriftPreset`) - Dataset de Referência (treino) x
   lote de produção. Compara a distribuição de cada feature e da predição
   (`default_proba`) usando PSI. Uma coluna é considerada em drift quando o
   PSI passa de `drift.psi_critical`.
2. **Desempenho** (`ClassificationPreset`) - holdout de teste x lote de
   produção, usando os rótulos tardios. O teste é a referência de desempenho
   porque métricas no próprio treino seriam otimistas.

Os relatórios HTML ficam em `reports/evidently/` e as métricas de todos os
meses são consolidadas em `reports/drift_summary.csv` - a série temporal que
mostra quando e como o modelo degradou.

Uso:
    uv run python -m credit_scoring.drift.detect
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import ClassificationPreset, DataDriftPreset

from credit_scoring.config import load_config
from credit_scoring.data.load import ID_COLUMN, TARGET
from credit_scoring.data.preprocess import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from credit_scoring.model.evaluate import classification_report
from credit_scoring.model.predict import LoadedModel, load_model, predict
from credit_scoring.observability.logs import setup_logging
from credit_scoring.observability.metrics import StageMetrics, track_stage

logger = logging.getLogger(__name__)

PREDICTION: str = "default_proba"
MONITORED_COLUMNS: list[str] = [*FEATURES, PREDICTION]


def with_predictions(df: pd.DataFrame, model: LoadedModel) -> pd.DataFrame:
    """Anexa a probabilidade de default prevista pelo modelo.

    Args:
        df: Lote com features (e opcionalmente o alvo).
        model: Modelo carregado.

    Returns:
        Cópia do lote com a coluna `default_proba`.
    """
    preds = predict(df.drop(columns=TARGET, errors="ignore"), model)
    return df.merge(preds[[ID_COLUMN, PREDICTION]], on=ID_COLUMN, validate="one_to_one")


def data_definition() -> DataDefinition:
    """Define os tipos de coluna e a tarefa de classificação para o Evidently.

    Returns:
        `DataDefinition` com features numéricas, categóricas e a
        classificação binária (`default` x `default_proba`).
    """
    return DataDefinition(
        id_column=ID_COLUMN,
        numerical_columns=[*NUMERIC_FEATURES, PREDICTION],
        categorical_columns=CATEGORICAL_FEATURES,
        classification=[
            BinaryClassification(target=TARGET, prediction_probas=PREDICTION, pos_label=1)
        ],
    )


def _dataset(df: pd.DataFrame) -> Dataset:
    """Converte um DataFrame em `Dataset` do Evidently.

    Args:
        df: DataFrame com features, predição e (opcionalmente) alvo.

    Returns:
        `Dataset` com a definição de dados do projeto.
    """
    return Dataset.from_pandas(df, data_definition=data_definition())


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, psi_threshold: float) -> Snapshot:
    """Executa o relatório de data drift (features + predição) com PSI.

    Args:
        reference: Dataset de Referência com `default_proba`.
        current: Lote de produção com `default_proba`.
        psi_threshold: PSI a partir do qual a coluna é considerada em drift.

    Returns:
        `Snapshot` do Evidently com o resultado.
    """
    preset = DataDriftPreset(columns=MONITORED_COLUMNS, method="psi", threshold=psi_threshold)
    report = Report([preset])
    return report.run(_dataset(current), _dataset(reference))


def performance_report(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float
) -> Snapshot:
    """Executa o relatório de desempenho de classificação.

    Args:
        reference: Holdout de teste com alvo e `default_proba`.
        current: Lote de produção com rótulos tardios e `default_proba`.
        threshold: Ponto de corte do modelo (decisão de recusa).

    Returns:
        `Snapshot` do Evidently com o resultado.
    """
    report = Report([ClassificationPreset(probas_threshold=threshold)])
    return report.run(_dataset(current), _dataset(reference))


def _metric_type(metric: dict[str, Any]) -> str:
    """Extrai o nome curto do tipo de métrica (ex.: `ValueDrift`).

    Args:
        metric: Item de `snapshot.dict()["metrics"]`.

    Returns:
        Nome curto da métrica.
    """
    return str(metric["config"]["type"]).rsplit(":", 1)[-1]


def extract_drift(snapshot: Snapshot) -> dict[str, float]:
    """Extrai PSI por coluna e o total de colunas em drift.

    Args:
        snapshot: Resultado de `drift_report`.

    Returns:
        Dicionário com `drifted_columns`, `drift_share` e `psi_<coluna>`.
    """
    out: dict[str, float] = {}
    for metric in snapshot.dict()["metrics"]:
        kind = _metric_type(metric)
        if kind == "DriftedColumnsCount":
            out["drifted_columns"] = int(metric["value"]["count"])
            out["drift_share"] = float(metric["value"]["share"])
        elif kind == "ValueDrift":
            out[f"psi_{metric['config']['column']}"] = float(metric["value"])
    return out


def extract_performance(snapshot: Snapshot) -> dict[str, float]:
    """Extrai as métricas globais de classificação do relatório.

    Args:
        snapshot: Resultado de `performance_report`.

    Returns:
        Dicionário com `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
        e `log_loss` calculados pelo Evidently.
    """
    names = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1Score": "f1",
        "RocAuc": "roc_auc",
        "LogLoss": "log_loss",
    }
    return {
        names[_metric_type(m)]: float(m["value"])
        for m in snapshot.dict()["metrics"]
        if _metric_type(m) in names
    }


def load_production_batch(prod_dir: Path, batch: str) -> pd.DataFrame:
    """Carrega um lote aceito junto com seus rótulos tardios.

    Args:
        prod_dir: Diretório base de produção.
        batch: Nome do lote (`prod_AAAA_MM`).

    Returns:
        Lote com features e alvo `default`.
    """
    features = pd.read_parquet(prod_dir / "accepted" / f"{batch}.parquet")
    labels = pd.read_parquet(prod_dir / "labels" / f"{batch}.parquet")
    return features.merge(labels, on=ID_COLUMN, validate="one_to_one")


def classify_status(row: dict[str, Any], drift_cfg: dict[str, Any]) -> str:
    """Classifica a saúde do modelo no mês.

    Args:
        row: Métricas do mês (saída consolidada de `detect`).
        drift_cfg: Seção `drift` do config.

    Returns:
        `ok`, `data_drift` (distribuição mudou, desempenho preservado) ou
        `degraded` (desempenho abaixo do mínimo aceitável).
    """
    if row["roc_auc"] < drift_cfg["auc_min"]:
        return "degraded"
    if row["drifted_columns"] > 0 or row[f"psi_{PREDICTION}"] > drift_cfg["psi_critical"]:
        return "data_drift"
    return "ok"


STATUS_CODES: dict[str, int] = {"ok": 0, "data_drift": 1, "degraded": 2}

PERFORMANCE_METRICS: dict[str, str] = {
    "roc_auc": "model_roc_auc",
    "ks": "model_ks",
    "recall": "model_recall",
    "precision": "model_precision",
    "approval_rate": "model_approval_rate",
    "expected_cost": "model_expected_cost",
    "default_rate": "production_default_rate",
    "mean_default_proba": "model_mean_default_proba",
}


def export_metrics(metrics: StageMetrics, summary: pd.DataFrame) -> None:
    """Publica os resultados dos testes de drift como métricas Prometheus.

    Duas visões:

    - **histórico** (label `month`): uma série por mês, para os painéis de
      evolução no Grafana;
    - **atual** (sem `month`): valores do mês mais recente, usados pelas
      regras de alerta (`feature_psi`, `prediction_psi`, `model_auc`,
      `drifted_columns`, `model_health_status`).

    Args:
        metrics: Registro de métricas da etapa.
        summary: Saída de `detect`.
    """
    psi_cols = [c for c in summary.columns if c.startswith("psi_")]
    for row in summary.to_dict("records"):
        month = row["month"]
        for col in psi_cols:
            metrics.set(
                "drift_feature_psi", row[col], "PSI referência x produção por mês",
                feature=col.removeprefix("psi_"), month=month,
            )
        metrics.set(
            "drift_drifted_columns", row["drifted_columns"],
            "Colunas com PSI acima do limite crítico", month=month,
        )
        metrics.set(
            "model_health_status_code", STATUS_CODES[row["status"]],
            "0 = ok, 1 = data drift, 2 = degradado", month=month,
        )
        for col, name in PERFORMANCE_METRICS.items():
            metrics.set(name, row[col], f"{col} do modelo em produção por mês", month=month)

    latest = summary.iloc[-1]
    for col in psi_cols:
        feature = col.removeprefix("psi_")
        if feature == PREDICTION:
            metrics.set("prediction_psi", latest[col], "PSI da predição no mês mais recente")
        else:
            metrics.set(
                "feature_psi", latest[col], "PSI da feature no mês mais recente", feature=feature
            )
    metrics.set("drifted_columns", latest["drifted_columns"], "Colunas em drift no último mês")
    metrics.set("model_auc", latest["roc_auc"], "AUC do modelo no mês mais recente")
    metrics.set(
        "model_health_status", STATUS_CODES[latest["status"]],
        "Saúde do modelo no último mês: 0 = ok, 1 = data drift, 2 = degradado",
    )
    metrics.set(
        "drift_last_evaluated_month_timestamp_seconds",
        pd.Period(latest["month"], "M").to_timestamp().timestamp(),
        "Mês de referência da última avaliação de drift",
    )


def detect(cfg: dict[str, Any], save_html: bool = True) -> pd.DataFrame:
    """Compara referência e produção mês a mês e consolida as métricas.

    Args:
        cfg: Configurações do projeto.
        save_html: Se True, salva os relatórios HTML do Evidently.

    Returns:
        DataFrame com uma linha por mês: PSI por coluna, colunas em drift,
        métricas de desempenho (Evidently + KS, custo e aprovação) e status.

    Raises:
        FileNotFoundError: Se não houver lotes de produção.
    """
    paths, drift_cfg = cfg["paths"], cfg["drift"]
    prod = Path(paths["production"])
    html_dir = Path(paths["reports"]) / "evidently"
    model = load_model(paths["models"])

    reference = with_predictions(
        pd.read_parquet(Path(paths["reference"]) / "reference.parquet"), model
    )
    holdout = with_predictions(pd.read_parquet(Path(paths["processed"]) / "test.parquet"), model)

    batches = sorted(p.stem for p in (prod / "accepted").glob("prod_*.parquet"))
    if not batches:
        raise FileNotFoundError("Nenhum lote de produção encontrado. Rode `make simulate`.")
    if save_html:
        html_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for batch in batches:
        current = with_predictions(load_production_batch(prod, batch), model)
        drift = drift_report(reference, current, drift_cfg["psi_critical"])
        perf = performance_report(holdout, current, model.threshold)
        if save_html:
            drift.save_html(str(html_dir / f"data_drift_{batch}.html"))
            perf.save_html(str(html_dir / f"performance_{batch}.html"))

        credit = classification_report(current[TARGET], current[PREDICTION], model.threshold)
        row: dict[str, Any] = {
            "batch": batch,
            "month": batch.removeprefix("prod_").replace("_", "-"),
            "n": len(current),
            "default_rate": float(current[TARGET].mean()),
            "mean_default_proba": float(current[PREDICTION].mean()),
            **extract_drift(drift),
            **extract_performance(perf),
            "ks": credit["ks"],
            "expected_cost": credit["expected_cost"],
            "approval_rate": credit["approval_rate"],
        }
        row["status"] = classify_status(row, drift_cfg)
        rows.append(row)
        drifted = [
            k.removeprefix("psi_") for k, v in row.items()
            if k.startswith("psi_") and v > drift_cfg["psi_critical"]
        ]
        level = logging.WARNING if row["status"] != "ok" else logging.INFO
        logger.log(
            level,
            "%s: status %s | AUC %.3f | colunas em drift: %s",
            batch, row["status"], row["roc_auc"], ", ".join(drifted) or "nenhuma",
            extra={
                "event": "drift_evaluated",
                "batch": batch,
                "month": row["month"],
                "status": row["status"],
                "roc_auc": round(row["roc_auc"], 4),
                "drifted_features": drifted,
            },
        )

    summary = pd.DataFrame(rows)
    summary_path = Path(paths["reports"]) / "drift_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Resumo salvo em %s | relatórios HTML em %s", summary_path, html_dir)
    return summary


def main() -> None:
    """Executa a detecção e exibe a evolução mensal das principais métricas."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    with track_stage("detect") as metrics:
        summary = detect(load_config())
        export_metrics(metrics, summary)
    cols = [
        "month",
        "drifted_columns",
        f"psi_{PREDICTION}",
        "psi_credit_amount",
        "psi_checking_status",
        "default_rate",
        "roc_auc",
        "ks",
        "recall",
        "approval_rate",
        "expected_cost",
        "status",
    ]
    logger.info("\n%s", summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    setup_logging()
    main()
