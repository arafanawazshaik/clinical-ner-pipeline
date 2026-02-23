"""
FastAPI Application.

Serves the clinical NER pipeline as a REST API
with Prometheus metrics for monitoring.
"""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import extract, batch, health

# ── Prometheus Metrics ──
REQUEST_COUNT = Counter(
    "ner_requests_total",
    "Total NER API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "ner_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

ENTITIES_EXTRACTED = Counter(
    "ner_entities_extracted_total",
    "Total entities extracted",
    ["entity_type"],
)

# ── App Setup ──
app = FastAPI(
    title="Clinical NER Pipeline",
    description="Extract medical entities from clinical notes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract.router)
app.include_router(batch.router)
app.include_router(health.router)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request count and latency for all endpoints."""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)

    return response


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )