import json
import logging

import pandas as pd
import pytest
from prometheus_client import generate_latest

from credit_scoring.observability.logs import RUN_ID, setup_logging
from credit_scoring.observability.metrics import StageMetrics, pushgateway_url, track_stage


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    path = setup_logging("test.jsonl")
    yield path
    logging.getLogger().handlers.clear()


def read_events(path) -> list[dict]:
    for handler in logging.getLogger().handlers:
        handler.flush()
    return [json.loads(line) for line in path.read_text().splitlines()]


def exposition(metrics: StageMetrics) -> str:
    return generate_latest(metrics.registry).decode()


def test_json_log_has_context_and_extras(log_file):
    logging.getLogger("x").info("olá", extra={"event": "batch_accepted", "rows": 10})
    event = read_events(log_file)[-1]
    assert event["message"] == "olá"
    assert event["level"] == "INFO"
    assert event["run_id"] == RUN_ID
    assert event["event"] == "batch_accepted"
    assert event["rows"] == 10


def test_track_stage_success(log_file):
    with track_stage("unit") as metrics:
        metrics.set("custom_metric", 3, "teste", batch="b1")
        logging.getLogger("x").info("dentro da etapa")

    text = exposition(metrics)
    assert 'pipeline_stage_success{stage="unit"} 1.0' in text
    assert 'custom_metric{batch="b1"} 3.0' in text
    assert "pipeline_stage_last_success_timestamp_seconds" in text

    events = read_events(log_file)
    assert [e.get("event") for e in events if e.get("event")] == ["stage_started", "stage_finished"]
    assert all(e["stage"] == "unit" for e in events)
    assert events[-1]["status"] == "success"


def test_track_stage_failure_is_logged_and_reraised(log_file):
    with pytest.raises(ValueError, match="boom"):
        with track_stage("unit") as metrics:
            raise ValueError("boom")

    text = exposition(metrics)
    assert 'pipeline_stage_success{stage="unit"} 0.0' in text
    assert "pipeline_stage_last_success_timestamp_seconds" not in text

    failed = [e for e in read_events(log_file) if e.get("event") == "stage_failed"]
    assert failed and failed[0]["error_type"] == "ValueError"
    assert "Traceback" in failed[0]["exception"]


def test_push_disabled_by_env():
    assert pushgateway_url() is None
    assert StageMetrics("unit").push() is False


def test_push_failure_does_not_raise(monkeypatch, caplog):
    monkeypatch.setenv("PUSHGATEWAY_URL", "127.0.0.1:1")
    metrics = StageMetrics("unit")
    metrics.set("x", 1)
    assert metrics.push() is False
    assert "Pushgateway indisponível" in caplog.text


def test_export_drift_metrics():
    from credit_scoring.drift.detect import export_metrics

    summary = pd.DataFrame(
        [
            {"month": "2025-01", "psi_credit_amount": 0.05, "psi_default_proba": 0.1,
             "drifted_columns": 0, "status": "ok", "roc_auc": 0.80, "ks": 0.5,
             "recall": 0.9, "precision": 0.4, "approval_rate": 0.3, "expected_cost": 0.6,
             "default_rate": 0.3, "mean_default_proba": 0.3},
            {"month": "2025-12", "psi_credit_amount": 0.40, "psi_default_proba": 0.5,
             "drifted_columns": 2, "status": "degraded", "roc_auc": 0.65, "ks": 0.3,
             "recall": 0.8, "precision": 0.5, "approval_rate": 0.2, "expected_cost": 0.8,
             "default_rate": 0.5, "mean_default_proba": 0.5},
        ]
    )
    metrics = StageMetrics("detect")
    export_metrics(metrics, summary)
    text = exposition(metrics)
    assert 'drift_feature_psi{feature="credit_amount",month="2025-01"} 0.05' in text
    assert 'model_roc_auc{month="2025-12"} 0.65' in text
    assert 'feature_psi{feature="credit_amount"} 0.4' in text
    assert "prediction_psi 0.5" in text
    assert "model_auc 0.65" in text
    assert "model_health_status 2.0" in text
