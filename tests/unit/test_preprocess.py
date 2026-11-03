import pandas as pd
import pytest

from credit_scoring.data.download import DEFAULT_PATH
from credit_scoring.data.load import SENSITIVE_COLUMNS, TARGET, category_dtype
from credit_scoring.data.preprocess import (
    FEATURES,
    ID_COLUMN,
    add_features,
    add_id,
    build_dataset,
    clean,
    split_xy,
    train_test,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "credit_amount": [1200, 3000, 1200],
            "duration_months": [12, 24, 12],
            TARGET: pd.Series([0, 1, 0], dtype="int8"),
        }
    )


def test_clean_removes_duplicates(df):
    out = clean(df)
    assert len(out) == 2
    assert out["credit_amount"].dtype == "int16"


def test_clean_fails_on_nulls(df):
    df.loc[0, "credit_amount"] = None
    with pytest.raises(ValueError, match="nulos"):
        clean(df)


def test_add_features_monthly_installment(df):
    out = add_features(df)
    assert out["monthly_installment"].tolist() == [100.0, 125.0, 100.0]


def test_add_id_is_unique_and_first(df):
    out = add_id(df)
    assert out.columns[0] == ID_COLUMN
    assert out[ID_COLUMN].is_unique


def test_category_dtype_keeps_unseen_categories():
    assert "vacation" in category_dtype("purpose").categories


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="rode `make data` antes")
class TestFullDataset:
    def test_build_dataset(self):
        data = build_dataset()
        assert len(data) == 1000
        assert not data.isna().any().any()

    def test_sensitive_not_in_features(self):
        X, y, sensitive = split_xy(build_dataset())
        assert not set(SENSITIVE_COLUMNS) & set(X.columns)
        assert list(X.columns) == FEATURES
        assert y.name == TARGET
        assert list(sensitive.columns) == SENSITIVE_COLUMNS

    def test_train_test_is_stratified(self):
        split = train_test(build_dataset(), test_size=0.2, seed=42)
        assert len(split.train) == 800 and len(split.test) == 200
        assert split.train[TARGET].mean() == pytest.approx(0.3, abs=0.01)
        assert split.test[TARGET].mean() == pytest.approx(0.3, abs=0.01)
        assert not set(split.train[ID_COLUMN]) & set(split.test[ID_COLUMN])
