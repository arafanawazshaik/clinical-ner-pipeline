"""
FastAPI Application.

Serves the clinical NER pipeline as a REST API.
"""

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import extract, batch, health

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