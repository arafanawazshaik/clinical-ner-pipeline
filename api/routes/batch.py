"""Batch extraction endpoint."""

import time
from fastapi import APIRouter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.models.requests import BatchRequest
from api.models.responses import ExtractResponse
from api.dependencies import get_predictor
from src.preprocessing.pipeline import preprocess_note
from src.rules.rule_extractor import extract_entities_by_rules
from src.inference.postprocessor import postprocess
from src.inference.schema_formatter import format_output

router = APIRouter()


@router.post("/batch", response_model=list[ExtractResponse])
async def batch_extract(request: BatchRequest):
    """Extract entities from multiple clinical notes."""
    predictor = get_predictor()
    results = []

    for note in request.notes:
        start = time.time()

        preprocessed = preprocess_note(note.text)
        ml_entities = predictor.predict(preprocessed.cleaned_text)
        rule_entities = extract_entities_by_rules(preprocessed.cleaned_text)
        accepted, flagged = postprocess(
            preprocessed.cleaned_text,
            ml_entities,
            rule_entities,
            preprocessed.sections,
        )

        elapsed_ms = (time.time() - start) * 1000
        result = format_output(
            document_id=note.document_id,
            accepted_entities=accepted,
            flagged_entities=flagged,
            processing_time_ms=elapsed_ms,
        )
        results.append(result)

    return results