import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_readiness_check():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_missing_idempotency_key():
    response = client.post("/api/v1/broll/jobs", json={"script": "Test script"})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "MISSING_IDEMPOTENCY_KEY"
    assert data["error"]["retryable"] is False

def test_missing_input_content():
    response = client.post(
        "/api/v1/broll/jobs",
        json={"script": ""},
        headers={"Idempotency-Key": "test_idem_empty_123"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "MISSING_INPUT_CONTENT"

def test_idempotency_caching():
    key = "idem_test_cache_repeat_99"
    payload = {"script": "This is a repeated submission test.", "aspect_ratios": ["16:9"]}
    
    # First submission
    res1 = client.post("/api/v1/broll/jobs", json=payload, headers={"Idempotency-Key": key})
    assert res1.status_code == 202
    job_id_1 = res1.json()["job_id"]

    # Second submission with same key
    res2 = client.post("/api/v1/broll/jobs", json=payload, headers={"Idempotency-Key": key})
    assert res2.status_code == 202
    assert res2.json()["job_id"] == job_id_1
    assert res2.headers.get("X-Cache") == "HIT"
