from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from credit_scoring.data.load import ID_COLUMN, TARGET
from credit_scoring.model.predict import load_model, predict, predict_batches, to_score

MODELS_DIR = Path("models")
TEST_PATH = Path("data/processed/test.parquet")

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "model.joblib").exists() or not TEST_PATH.exists(),
    reason="rode `make preprocess` e `make train` antes",
)


@pytest.fixture(scope="module")
def model():
    return load_model(MODELS_DIR)


@pytest.fixture
def batch() -> pd.DataFrame:
    return pd.read_parquet(TEST_PATH).head(30).drop(columns=TARGET)


def test_to_score():
    assert to_score(np.array([0.0, 0.25, 1.0])).tolist() == [1000, 750, 0]


def test_predict_output(batch, model):
    preds = predict(batch, model)
    assert list(preds[ID_COLUMN]) == list(batch[ID_COLUMN])
    assert preds["default_proba"].between(0, 1).all()
    assert preds["score"].between(0, 1000).all()
    reject = preds["default_proba"] >= model.threshold
    assert (preds.loc[reject, "decision"] == "reject").all()
    assert (preds.loc[~reject, "decision"] == "approve").all()
    assert preds["model_version"].str.startswith("credit-scoring-xgboost@").all()


def test_predictions_do_not_store_personal_data(batch, model):
    preds = predict(batch, model)
    assert not {"age", "sex", "marital_status", "foreign_worker"} & set(preds.columns)


def test_predict_rejects_batch_outside_contract(batch, model):
    batch.loc[0, "age"] = 16
    with pytest.raises(SchemaErrors):
        predict(batch, model)


def test_predict_batches(batch, model, tmp_path):
    accepted = tmp_path / "accepted"
    accepted.mkdir()
    batch.to_parquet(accepted / "prod_2025_01.parquet", index=False)
    summary = predict_batches(accepted, tmp_path / "predictions", model)
    assert (tmp_path / "predictions" / "prod_2025_01.parquet").exists()
    assert summary.loc[0, "n"] == len(batch)


def test_predict_batches_without_batches(model, tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_batches(tmp_path, tmp_path / "out", model)
