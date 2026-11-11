import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import load_config
from credit_scoring.contracts.schemas import INFERENCE_CONTRACT, contract_report
from credit_scoring.data.load import ID_COLUMN, TARGET
from credit_scoring.drift.simulate import (
    apply_concept_drift,
    apply_data_drift,
    generate_batch,
    ramp,
    sample_weights,
)

TEST_PATH = "data/processed/test.parquet"


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    try:
        return pd.read_parquet(TEST_PATH)
    except FileNotFoundError:
        pytest.skip("rode `make preprocess` antes")


@pytest.fixture(scope="module")
def sim() -> dict:
    return load_config()["simulation"]


def test_ramp():
    assert ramp(2, start=3, n_batches=12) == 0.0
    assert ramp(3, start=3, n_batches=12) == pytest.approx(1 / 9)
    assert ramp(11, start=3, n_batches=12) == 1.0


def test_sample_weights_favor_new_profile(pool):
    stressed = pool["checking_status"] == "< 0 DM"
    neutral = sample_weights(pool, 0.0, 3.0)
    drifted = sample_weights(pool, 1.0, 3.0)
    assert np.allclose(neutral, 1 / len(pool))
    assert drifted[stressed.to_numpy()].sum() > neutral[stressed.to_numpy()].sum()


def test_data_drift_inflates_and_keeps_installment_consistent(pool):
    rng = np.random.default_rng(0)
    out = apply_data_drift(pool, inflation_factor=1.3, intensity=1.0, extend_prob=0.4, rng=rng)
    assert out["credit_amount"].mean() > 1.2 * pool["credit_amount"].mean()
    assert (out["duration_months"] >= pool["duration_months"]).all()
    expected = out["credit_amount"] / out["duration_months"]
    assert (out["monthly_installment"] - expected).abs().max() <= 0.01


def test_concept_drift_only_turns_good_into_bad(pool):
    rng = np.random.default_rng(0)
    out, flipped = apply_concept_drift(pool, 1.0, 0.5, 0.5, rng)
    assert flipped > 0
    assert (out[TARGET] >= pool[TARGET]).all(), "nenhum mau pagador vira bom"
    changed = out[TARGET] != pool[TARGET]
    targeted = (pool["checking_status"] == "no checking account") | (
        pool["credit_history"] == "critical account / other credits"
    )
    assert targeted[changed].all(), "só perfis afetados pela crise mudam de rótulo"
    assert out[pool.columns.drop(TARGET)].equals(pool[pool.columns.drop(TARGET)])


def test_no_concept_drift_when_intensity_zero(pool):
    out, flipped = apply_concept_drift(pool, 0.0, 0.5, 0.5, np.random.default_rng(0))
    assert flipped == 0
    assert out[TARGET].equals(pool[TARGET])


@pytest.mark.parametrize("t", [0, 11])
def test_generated_batch_respects_inference_contract(pool, sim, t):
    batch, manifest = generate_batch(pool, t, sim, np.random.default_rng(t))
    assert contract_report(batch.drop(columns=TARGET), INFERENCE_CONTRACT).empty
    assert batch[ID_COLUMN].str.startswith(f"APP-{t + 1}").all()
    assert manifest.n_rows == len(batch)


def test_drifted_batch_differs_from_stable(pool, sim):
    stable, m0 = generate_batch(pool, 0, sim, np.random.default_rng(1))
    drifted, m11 = generate_batch(pool, 11, sim, np.random.default_rng(1))
    assert m0.data_drift == 0 and m11.data_drift == 1
    assert drifted["credit_amount"].mean() > stable["credit_amount"].mean()
    neg = "< 0 DM"
    assert (drifted["checking_status"] == neg).mean() > (stable["checking_status"] == neg).mean()
    assert m11.default_rate > m0.default_rate
