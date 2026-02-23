"""Health check endpoint."""

import time
from fastapi import APIRouter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.models.responses import HealthResponse
from api.dependencies import get_predictor

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    try:
        predictor = get_predictor()
        model_loaded = True
        model_version = "biobert-ner-v1.0"
    except Exception:
        model_loaded = False
        model_version = "not loaded"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=round(time.time() - _start_time, 2),
    )