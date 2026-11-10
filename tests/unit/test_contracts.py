import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from credit_scoring.contracts.schemas import (
    INFERENCE_CONTRACT,
    TRAINING_CONTRACT,
    contract_report,
    validate,
)
from credit_scoring.data.download import DEFAULT_PATH
from credit_scoring.data.load import TARGET, category_dtype
from credit_scoring.data.preprocess import build_dataset

pytestmark = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="rode `make data` antes")


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return build_dataset()


@pytest.fixture
def batch(dataset) -> pd.DataFrame:
    return dataset.head(20).copy()


def failed_checks(df: pd.DataFrame, contract=TRAINING_CONTRACT) -> set[tuple[str, str]]:
    report = contract_report(df, contract)
    return set(zip(report["column"].astype(str), report["check"].astype(str), strict=True))


def test_full_dataset_respects_contract(dataset):
    assert contract_report(dataset, TRAINING_CONTRACT).empty


def test_contract_columns_match_pipeline_output(dataset):
    assert set(TRAINING_CONTRACT.columns) == set(dataset.columns)


def test_rejects_underage(batch):
    batch.loc[0, "age"] = 17
    assert ("age", "in_range(18, 100)") in failed_checks(batch)


def test_rejects_null_credit_amount(batch):
    batch["credit_amount"] = batch["credit_amount"].astype("float64")
    batch.loc[0, "credit_amount"] = None
    cols = {c for c, _ in failed_checks(batch)}
    assert "credit_amount" in cols


def test_rejects_non_positive_credit_amount(batch):
    batch.loc[0, "credit_amount"] = 0
    assert ("credit_amount", "greater_than(0)") in failed_checks(batch)


def test_rejects_duplicate_records(batch):
    dup = batch.iloc[[0]].copy()
    dup["application_id"] = "APP-99999"
    checks = {chk for _, chk in failed_checks(pd.concat([batch, dup], ignore_index=True))}
    assert "no_duplicate_records" in checks


def test_rejects_duplicate_id(batch):
    batch.loc[1, "application_id"] = batch.loc[0, "application_id"]
    assert ("application_id", "field_uniqueness") in failed_checks(batch)


def test_rejects_unknown_category(batch):
    batch["purpose"] = batch["purpose"].astype(str)
    batch.loc[0, "purpose"] = "crypto"
    cols = {c for c, _ in failed_checks(batch)}
    assert "purpose" in cols


def test_rejects_inconsistent_installment(batch):
    batch.loc[0, "monthly_installment"] = 1.0
    checks = {chk for _, chk in failed_checks(batch)}
    assert "monthly_installment_consistent" in checks


def test_rejects_invalid_sex_marital_pair(batch):
    batch.loc[0, "sex"] = "female"
    batch.loc[0, "marital_status"] = "married/widowed"
    checks = {chk for _, chk in failed_checks(batch)}
    assert "valid_sex_marital_status_pair" in checks


def test_rejects_unexpected_column(batch):
    batch["cpf"] = "000.000.000-00"
    report = contract_report(batch, TRAINING_CONTRACT)
    assert "cpf" in set(report["failure_case"])


def test_accepts_inflated_values(batch):
    """Drift econômico não é erro de dado: valores maiores devem passar."""
    batch["credit_amount"] = batch["credit_amount"] * 3
    batch["monthly_installment"] = (batch["credit_amount"] / batch["duration_months"]).round(2)
    assert contract_report(batch, TRAINING_CONTRACT).empty


def test_inference_contract_allows_missing_target(batch):
    assert contract_report(batch.drop(columns=TARGET), INFERENCE_CONTRACT).empty
    assert not contract_report(batch.drop(columns=TARGET), TRAINING_CONTRACT).empty


def test_validate_raises_with_all_violations(batch):
    batch.loc[0, "age"] = 10
    batch.loc[1, "credit_amount"] = -5
    batch["purpose"] = pd.Categorical(batch["purpose"], dtype=category_dtype("purpose"))
    with pytest.raises(SchemaErrors) as exc:
        validate(batch, TRAINING_CONTRACT)
    assert {"age", "credit_amount"} <= set(exc.value.failure_cases["column"])
