import numpy as np
import pandas as pd
import pytest

from credit_scoring.data.load import category_dtype
from credit_scoring.data.preprocess import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from credit_scoring.model.evaluate import (
    classification_report,
    expected_cost,
    find_best_threshold,
    ks_statistic,
)
from credit_scoring.model.train import build_pipeline


def test_ks_perfect_separation():
    assert ks_statistic([0, 0, 1, 1], np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_expected_cost_weights_false_negatives():
    y_true = np.array([1, 0])
    assert expected_cost(y_true, np.array([0, 0])) == 2.5  # 1 FN * 5 / 2
    assert expected_cost(y_true, np.array([1, 1])) == 0.5  # 1 FP * 1 / 2


def test_threshold_is_low_when_fn_is_expensive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.3 * y + rng.uniform(0, 0.7, 2000), 0, 1)
    assert find_best_threshold(y, proba) < 0.5


def test_classification_report_keys():
    report = classification_report(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6]), 0.5)
    assert report["auc"] == 1.0
    assert report["approval_rate"] == 0.5
    assert {"gini", "ks", "brier", "expected_cost"} <= report.keys()


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({c: rng.integers(1, 50, n) for c in NUMERIC_FEATURES})
    for c in CATEGORICAL_FEATURES:
        cats = category_dtype(c).categories
        X[c] = pd.Categorical(rng.choice(cats[:2], n), dtype=category_dtype(c))
    return X[FEATURES], pd.Series(rng.integers(0, 2, n))


def test_pipeline_handles_unseen_category(toy_data):
    X, y = toy_data
    model = build_pipeline(min_frequency=5).fit(X, y)
    new = X.head(1).copy()
    new["purpose"] = pd.Categorical(["vacation"], dtype=category_dtype("purpose"))
    proba = model.predict_proba(new)[:, 1]
    assert 0 <= proba[0] <= 1
