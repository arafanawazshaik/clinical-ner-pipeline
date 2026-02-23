"""Tests for FastAPI endpoints."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestExtractEndpoint:
    def test_extract_entities(self):
        response = client.post("/extract", json={
            "text": "Patient has type 2 diabetes mellitus on metformin 500mg daily.",
            "document_id": "test_001",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test_001"
        assert len(data["entities"]) > 0

    def test_extract_empty_text(self):
        response = client.post("/extract", json={
            "text": "",
            "document_id": "test_002",
        })
        assert response.status_code == 422

    def test_extract_has_metadata(self):
        response = client.post("/extract", json={
            "text": "Patient takes lisinopril 10mg daily.",
            "document_id": "test_003",
        })
        data = response.json()
        assert "processing_metadata" in data
        assert data["processing_metadata"]["processing_time_ms"] > 0


class TestBatchEndpoint:
    def test_batch_extract(self):
        response = client.post("/batch", json={
            "notes": [
                {"text": "Patient has diabetes.", "document_id": "d1"},
                {"text": "Patient takes aspirin.", "document_id": "d2"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2