"""
Schema Formatter.

Converts merged entities into Pydantic-validated JSON output.
Ensures every output document follows a strict schema.
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class EntityOutput(BaseModel):
    """Single entity in the output."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    negated: bool
    section: str
    source: str
    context_snippet: str


class ProcessingMetadata(BaseModel):
    """Metadata about the extraction run."""
    total_entities: int
    auto_accepted: int
    flagged: int
    fallback_used: int
    processing_time_ms: float


class ExtractionResult(BaseModel):
    """Full output document schema."""
    document_id: str
    note_type: str = "Unknown"
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_version: str = "biobert-ner-v1.0"
    entities: list[EntityOutput]
    flagged_for_review: list[EntityOutput]
    processing_metadata: ProcessingMetadata


def format_output(
    document_id: str,
    accepted_entities: list,
    flagged_entities: list,
    processing_time_ms: float = 0.0,
    note_type: str = "Unknown",
) -> dict:
    """
    Build and validate the final JSON output.

    Args:
        document_id: Unique ID for this document.
        accepted_entities: Entities that passed confidence routing.
        flagged_entities: Entities flagged for review.
        processing_time_ms: Total processing time.
        note_type: Detected note type.

    Returns:
        Validated dict matching ExtractionResult schema.
    """
    # Convert entities to Pydantic models
    entity_outputs = [
        EntityOutput(
            text=e.text,
            label=e.label,
            start=e.start,
            end=e.end,
            confidence=e.confidence,
            negated=e.negated,
            section=e.section,
            source=e.source,
            context_snippet=e.context_snippet,
        )
        for e in accepted_entities
    ]

    flagged_outputs = [
        EntityOutput(
            text=e.text,
            label=e.label,
            start=e.start,
            end=e.end,
            confidence=e.confidence,
            negated=e.negated,
            section=e.section,
            source=e.source,
            context_snippet=e.context_snippet,
        )
        for e in flagged_entities
    ]

    # Count by status
    fallback_count = sum(1 for e in accepted_entities if e.status == "fallback")

    metadata = ProcessingMetadata(
        total_entities=len(entity_outputs) + len(flagged_outputs),
        auto_accepted=len(entity_outputs) - fallback_count,
        flagged=len(flagged_outputs),
        fallback_used=fallback_count,
        processing_time_ms=round(processing_time_ms, 2),
    )

    result = ExtractionResult(
        document_id=document_id,
        note_type=note_type,
        entities=entity_outputs,
        flagged_for_review=flagged_outputs,
        processing_metadata=metadata,
    )

    return result.model_dump()