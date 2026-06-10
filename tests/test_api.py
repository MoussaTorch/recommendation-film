"""Tests de l'API FastAPI (Swagger auto via /docs)."""
from fastapi.testclient import TestClient

from interface.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "data_loaded" in body
    assert "models_loaded" in body


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) >= 4


def test_openapi_docs():
    response = client.get("/docs")
    assert response.status_code == 200
