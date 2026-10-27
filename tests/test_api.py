"""API integration tests for the FastAPI serving endpoints."""

import json
from pathlib import Path
import sys
from typing import Iterator

from fastapi.testclient import TestClient
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Create a module-scoped TestClient so the CatBoost model loads once."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """GET /health should return 200 OK with model loaded and version hash."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0
    assert data["model_version"] != "unavailable"
    assert data["model_hash"] == data["model_version"]


def test_predict_happy_path(client: TestClient) -> None:
    """POST /predict should return a valid Agtron score for valid roast inputs."""
    payload = {
        "Consumed": 500.0,
        "Temp": 440.0,
        "Quench": 73.0,
        "Month": "1",
        "Blend": "284",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_agtron" in data
    assert 15.0 <= data["predicted_agtron"] <= 65.0
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0


def test_predict_normalizes_blend_and_accepts_lowercase_fields(client: TestClient) -> None:
    """POST /predict should normalize float-like blend strings and accept lowercase keys."""
    canonical = client.post(
        "/predict",
        json={"Consumed": 500.0, "Temp": 440.0, "Quench": 73.0, "Month": "1", "Blend": "284"},
    ).json()
    normalized = client.post(
        "/predict",
        json={"consumed": 500.0, "temp": 440.0, "quench": 73.0, "month": 1, "blend": " 284.0 "},
    ).json()

    assert normalized["predicted_agtron"] == canonical["predicted_agtron"]


def test_recommend_happy_path(client: TestClient) -> None:
    """POST /recommend should return optimal Temp/Quench settings and top 5 candidates."""
    payload = {
        "Consumed": 500.0,
        "Temp": 440.0,
        "Quench": 73.0,
        "Month": "1",
        "Blend": "284",
        "target_agtron": 45.0,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert 15.0 <= data["current_predicted_agtron"] <= 65.0
    assert data["target_agtron"] == 45.0
    assert 300.0 <= data["recommended_temp"] <= 600.0
    assert 0.0 <= data["recommended_quench"] <= 150.0
    assert 15.0 <= data["expected_agtron"] <= 65.0
    assert data["distance_from_target"] >= 0.0

    top_recs = data["top_recommendations"]
    assert len(top_recs) == 5
    errors = [candidate["error"] for candidate in top_recs]
    assert errors == sorted(errors)
    assert top_recs[0]["temp"] == data["recommended_temp"]
    assert top_recs[0]["quench"] == data["recommended_quench"]


@pytest.mark.parametrize(
    "endpoint,invalid_payload",
    [
        # Out-of-range temperature (too low)
        (
            "/predict",
            {"Consumed": 500.0, "Temp": 150.0, "Quench": 73.0, "Month": "1", "Blend": "284"},
        ),
        # Out-of-range temperature (too high)
        (
            "/predict",
            {"Consumed": 500.0, "Temp": 750.0, "Quench": 73.0, "Month": "1", "Blend": "284"},
        ),
        # Negative quench
        (
            "/predict",
            {"Consumed": 500.0, "Temp": 440.0, "Quench": -5.0, "Month": "1", "Blend": "284"},
        ),
        # Zero or negative consumed weight
        (
            "/predict",
            {"Consumed": 0.0, "Temp": 440.0, "Quench": 73.0, "Month": "1", "Blend": "284"},
        ),
        # Unknown blend not in training data
        (
            "/predict",
            {"Consumed": 500.0, "Temp": 440.0, "Quench": 73.0, "Month": "1", "Blend": "99999"},
        ),
        # Non-numeric blend string
        (
            "/predict",
            {"Consumed": 500.0, "Temp": 440.0, "Quench": 73.0, "Month": "1", "Blend": "invalid"},
        ),
        # Invalid month (> 12)
        (
            "/predict",
            {"Consumed": 500.0, "Temp": 440.0, "Quench": 73.0, "Month": "13", "Blend": "284"},
        ),
        # Out-of-range target_agtron on /recommend
        (
            "/recommend",
            {
                "Consumed": 500.0,
                "Temp": 440.0,
                "Quench": 73.0,
                "Month": "1",
                "Blend": "284",
                "target_agtron": 95.0,
            },
        ),
    ],
)
def test_invalid_inputs_return_422(
    client: TestClient, endpoint: str, invalid_payload: dict
) -> None:
    """Invalid roast inputs should be rejected with 422 Unprocessable Entity."""
    response = client.post(endpoint, json=invalid_payload)
    assert response.status_code == 422


def test_predict_logs_structured_jsonl_record(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /predict should append a structured JSON record with inputs, output, timestamp, and latency."""
    log_file = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_file))

    payload = {
        "Consumed": 500.0,
        "Temp": 440.0,
        "Quench": 73.0,
        "Month": "1",
        "Blend": "284",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "prediction"
    assert record["endpoint"] == "/predict"
    assert record["Consumed"] == 500.0
    assert record["Temp"] == 440.0
    assert record["Quench"] == 73.0
    assert record["Month"] == "1"
    assert record["Blend"] == "284"
    assert record["predicted_agtron"] == response.json()["predicted_agtron"]
    assert record["model_version"] == response.json()["model_version"]
    assert "timestamp" in record
    assert isinstance(record["latency_ms"], float)
    assert record["latency_ms"] >= 0.0


