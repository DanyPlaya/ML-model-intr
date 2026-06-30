from app.api import app
from app.model_handler import FEATURE_NAMES


def sample_payload():
    return {
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 34,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 12000,
        "BILL_AMT2": 11000,
        "BILL_AMT3": 10500,
        "BILL_AMT4": 9800,
        "BILL_AMT5": 9000,
        "BILL_AMT6": 8500,
        "PAY_AMT1": 3000,
        "PAY_AMT2": 2500,
        "PAY_AMT3": 2000,
        "PAY_AMT4": 2000,
        "PAY_AMT5": 2000,
        "PAY_AMT6": 2000,
    }


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["message"] == "Service is running"
    assert payload["model_versions"]


def test_metadata_endpoint_exposes_features():
    client = app.test_client()
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.get_json()["features"] == FEATURE_NAMES


def test_predict_endpoint():
    client = app.test_client()
    response = client.post("/predict", json=sample_payload())
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["prediction"] in [0, 1]
    assert 0 <= payload["probability"] <= 1
    assert payload["probability_default"] == payload["probability"]
    assert payload["model_version"] in ["v1", "v2"]
    assert payload["model_artifact"].startswith("model_v")
    assert payload["model_alias"]
    assert payload["request_id"]
    assert payload["assignment"] == "default"


def test_predict_accepts_explicit_model_version():
    client = app.test_client()
    payload = sample_payload()
    payload["model_version"] = "v1"

    response = client.post("/predict", json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert data["model_alias"] == "v1"
    assert data["model_version"] == "v1"
    assert data["assignment"] == "explicit"


def test_predict_accepts_explicit_v2_model_version():
    client = app.test_client()
    payload = sample_payload()
    payload["model_version"] = "v2"

    response = client.post("/predict", json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert data["model_alias"] == "v2"
    assert data["model_version"] == "v2"
    assert data["assignment"] == "explicit"


def test_predict_accepts_customer_id_for_ab_assignment():
    client = app.test_client()
    payload = sample_payload()
    payload["customer_id"] = "client-42"

    response = client.post("/predict", json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert data["model_alias"] in ["v1", "v2"]
    assert data["assignment"] in ["ab_hash", "default"]

    repeated_response = client.post("/predict", json=payload)
    assert repeated_response.get_json()["model_alias"] == data["model_alias"]


def test_predict_rejects_unknown_model_version():
    client = app.test_client()
    payload = sample_payload()
    payload["model_version"] = "v999"

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "Unknown model_version" in response.get_json()["error"]


def test_predict_rejects_missing_features():
    client = app.test_client()
    data = sample_payload()
    data.pop("AGE")

    response = client.post("/predict", json=data)
    assert response.status_code == 400
    assert "AGE" in response.get_json()["error"]


def test_predict_rejects_non_object_json():
    client = app.test_client()
    response = client.post("/predict", json=[1, 2, 3])

    assert response.status_code == 400
    assert "JSON object" in response.get_json()["error"]


def test_predict_rejects_non_numeric_feature():
    client = app.test_client()
    payload = sample_payload()
    payload["AGE"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "AGE must be numeric" in response.get_json()["error"]


def test_predict_rejects_wrong_method():
    client = app.test_client()
    response = client.get("/predict")

    assert response.status_code == 405
