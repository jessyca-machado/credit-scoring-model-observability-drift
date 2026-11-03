import pandas as pd
import pytest

from credit_scoring.data.download import DEFAULT_PATH
from credit_scoring.data.load import (
    CATEGORY_MAPS,
    RAW_COLUMNS,
    TARGET,
    decode,
    load_german_credit,
)

SAMPLE = [
    "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1",
    "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2",
]


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    rows = [line.split(" ") for line in SAMPLE]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    num = ["duration_months", "credit_amount", "installment_rate", "residence_since",
           "age", "existing_credits", "num_dependents", "class"]
    df[num] = df[num].astype(int)
    return df


def test_decode_target(raw_sample):
    out = decode(raw_sample)
    assert out[TARGET].tolist() == [0, 1]


def test_decode_splits_personal_status(raw_sample):
    out = decode(raw_sample)
    assert "personal_status" not in out.columns
    assert out["sex"].tolist() == ["male", "female"]
    assert out["marital_status"].tolist() == ["single", "divorced/separated/married"]


def test_decode_has_no_unmapped_codes(raw_sample):
    out = decode(raw_sample)
    for col in CATEGORY_MAPS:
        assert out[col].notna().all(), col


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="rode `make data` antes")
def test_full_dataset():
    df = load_german_credit()
    assert df.shape == (1000, 22)
    assert df[TARGET].sum() == 300
    assert not df.isna().any().any()
