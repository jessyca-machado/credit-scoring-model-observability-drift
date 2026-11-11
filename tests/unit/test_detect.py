from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.data.load import TARGET
from credit_scoring.drift.detect import (
    PREDICTION,
    classify_status,
    drift_report,
    extract_drift,
    extract_performance,
    performance_report,
    with_predictions,
)
from credit_scoring.drift.simulate import apply_data_drift
from credit_scoring.model.predict import load_model

pytestmark = [
    pytest.mark.skipif(
        not Path("models/model.joblib").exists(), reason="rode `make train` antes"
    ),
    pytest.mark.filterwarnings("ignore"),
]

DRIFT_CFG = {"psi_critical": 0.25, "auc_min": 0.70}


@pytest.fixture(scope="module")
def model():
    return load_model("models")


@pytest.fixture(scope="module")
def reference(model) -> pd.DataFrame:
    return with_predictions(pd.read_parquet("data/reference/reference.parquet"), model)


@pytest.fixture(scope="module")
def holdout(model) -> pd.DataFrame:
    return with_predictions(pd.read_parquet("data/processed/test.parquet"), model)


def test_with_predictions_adds_probability(holdout):
    assert holdout[PREDICTION].between(0, 1).all()
    assert TARGET in holdout.columns


def test_no_drift_against_itself(reference):
    result = extract_drift(drift_report(reference, reference, psi_threshold=0.25))
    assert result["drifted_columns"] == 0
    assert result["psi_credit_amount"] == pytest.approx(0, abs=1e-6)


def test_detects_inflation_drift(reference, model):
    inflated = apply_data_drift(
        reference.drop(columns=PREDICTION), 2.0, 1.0, 0.0, np.random.default_rng(0)
    )
    current = with_predictions(inflated, model)
    result = extract_drift(drift_report(reference, current, psi_threshold=0.25))
    assert result["psi_credit_amount"] > 0.25
    assert result["drifted_columns"] >= 1


def test_performance_metrics(holdout, model):
    result = extract_performance(performance_report(holdout, holdout, model.threshold))
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"} <= result.keys()
    assert result["roc_auc"] > 0.7


@pytest.mark.parametrize(
    ("auc", "drifted", "psi_pred", "expected"),
    [
        (0.80, 0, 0.10, "ok"),
        (0.80, 2, 0.10, "data_drift"),
        (0.80, 0, 0.40, "data_drift"),
        (0.65, 3, 0.50, "degraded"),
    ],
)
def test_classify_status(auc, drifted, psi_pred, expected):
    row = {"roc_auc": auc, "drifted_columns": drifted, f"psi_{PREDICTION}": psi_pred}
    assert classify_status(row, DRIFT_CFG) == expected
