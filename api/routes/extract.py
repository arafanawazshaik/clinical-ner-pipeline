"""Single document extraction endpoint."""

import time
from fastapi import APIRouter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.models.requests import ExtractRequest
from api.models.responses import ExtractResponse
from api.dependencies import get_predictor
from src.preprocessing.pipeline import preprocess_note
from src.rules.rule_extractor import extract_entities_by_rules
from src.inference.postprocessor import postprocess
from src.inference.schema_formatter import format_output

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    """Extract entities from a single clinical note."""
    start = time.time()

    # Preprocess
    preprocessed = preprocess_note(request.text)

    # ML prediction
    predictor = get_predictor()
    ml_entities = predictor.predict(preprocessed.cleaned_text)

    # Rule-based extraction
    rule_entities = extract_entities_by_rules(preprocessed.cleaned_text)

    # Post-process and merge
    accepted, flagged = postprocess(
        preprocessed.cleaned_text,
        ml_entities,
        rule_entities,
        preprocessed.sections,
    )

    elapsed_ms = (time.time() - start) * 1000

    result = format_output(
        document_id=request.document_id,
        accepted_entities=accepted,
        flagged_entities=flagged,
        processing_time_ms=elapsed_ms,
    )

    return result