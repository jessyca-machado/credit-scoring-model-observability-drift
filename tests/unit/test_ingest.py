from pathlib import Path

import pandas as pd
import pytest

from credit_scoring.contracts.demo import TEST_PATH, corrupt, make_clean_batch
from credit_scoring.data.ingest import ingest_batch, main

pytestmark = pytest.mark.skipif(not TEST_PATH.exists(), reason="rode `make preprocess` antes")


@pytest.fixture
def clean_batch() -> pd.DataFrame:
    return make_clean_batch(n=20)


def write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def test_clean_batch_is_accepted(clean_batch, tmp_path):
    src = write(clean_batch, tmp_path / "incoming" / "ok.parquet")
    result = ingest_batch(src, tmp_path)
    assert result.accepted
    assert (tmp_path / "accepted" / "ok.parquet").exists()
    assert not src.exists(), "lote deve sair da área de entrada"
    assert not (tmp_path / "quarantine").exists()


def test_corrupted_batch_is_blocked(clean_batch, tmp_path):
    src = write(corrupt(clean_batch), tmp_path / "incoming" / "bad.parquet")
    result = ingest_batch(src, tmp_path)

    assert not result.accepted
    assert not src.exists(), "lote deve sair da área de entrada"
    assert not (tmp_path / "accepted").exists(), "nenhuma linha pode ser aceita"
    assert (tmp_path / "quarantine" / "bad.parquet").exists()

    report = pd.read_csv(result.report)
    checks = set(report["check"])
    assert {
        "in_range(18, 100)",
        "not_nullable",
        "greater_than(0)",
        "field_uniqueness",
        "column_in_schema",
        "no_duplicate_records",
        "monthly_installment_consistent",
        "valid_sex_marital_status_pair",
    } <= checks
    assert "crypto" in set(report["failure_case"].astype(str))


def test_cli_exit_code(clean_batch, tmp_path):
    ok = write(clean_batch, tmp_path / "incoming" / "ok.parquet")
    bad = write(corrupt(clean_batch), tmp_path / "incoming" / "bad.parquet")
    assert main([str(ok), "--production-dir", str(tmp_path)]) == 0
    assert main([str(bad), "--production-dir", str(tmp_path)]) == 1
