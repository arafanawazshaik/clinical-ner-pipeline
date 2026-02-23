"""Tests for Pydantic schema validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError
from api.models.requests import ExtractRequest, BatchRequest
from api.models.responses import ExtractResponse, HealthResponse
import pytest


class TestRequestModels:
    def test_valid_extract_request(self):
        req = ExtractRequest(text="Patient has diabetes", document_id="doc_001")
        assert req.text == "Patient has diabetes"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            ExtractRequest(text="", document_id="doc_001")

    def test_default_document_id(self):
        req = ExtractRequest(text="Patient has diabetes")
        assert req.document_id == "doc_001"

    def test_valid_batch_request(self):
        req = BatchRequest(notes=[
            ExtractRequest(text="Note 1", document_id="d1"),
            ExtractRequest(text="Note 2", document_id="d2"),
        ])
        assert len(req.notes) == 2


class TestResponseModels:
    def test_health_response(self):
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version="v1.0",
            uptime_seconds=100.0,
        )
        assert resp.status == "healthy"