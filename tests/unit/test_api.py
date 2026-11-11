from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_scoring.data.load import TARGET

pytestmark = pytest.mark.skipif(
    not Path("models/model.joblib").exists(), reason="rode `make train` antes"
)


@pytest.fixture(scope="module")
def client():
    from credit_scoring.serving.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def payload() -> list[dict]:
    df = pd.read_parquet("data/processed/test.parquet").head(3)
    df = df.drop(columns=[TARGET, "monthly_installment"])
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            for r in df.astype(object).to_dict("records")]


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("credit-scoring-xgboost@")


def test_predict(client, payload):
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    preds = resp.json()
    assert [p["application_id"] for p in preds] == [a["application_id"] for a in payload]
    assert all(p["decision"] in {"approve", "reject"} for p in preds)
    assert all(0 <= p["score"] <= 1000 for p in preds)


def test_predict_blocked_by_contract(client, payload):
    payload[0]["age"] = 16
    payload[1]["purpose"] = "crypto"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    violations = resp.json()["detail"]["contract_violations"]
    assert {"16", "crypto"} <= {v["failure_case"] for v in violations}


def test_metrics_exposed(client, payload):
    client.post("/predict", json=payload)
    text = client.get("/metrics/").text
    assert 'api_requests_total{endpoint="/predict",status_code="200"}' in text
    assert "model_predictions_total" in text
    assert "model_score_bucket" in text
    assert "api_contract_violations_total" in text
    assert 'model_info{threshold="0.11"' in text
