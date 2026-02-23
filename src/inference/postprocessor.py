"""
Post-Processor for NER Results.

Merges ML model and rule-based entities, handles deduplication,
overlap resolution, and confidence-based routing.
"""

from dataclasses import dataclass, asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import AUTO_ACCEPT_THRESHOLD, FLAG_THRESHOLD
from src.rules.negation_detector import detect_negation


@dataclass
class MergedEntity:
    """Final merged entity with all metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    negated: bool
    section: str
    source: str          # "model", "rules", or "merged"
    context_snippet: str
    status: str          # "auto_accepted", "flagged", "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


def postprocess(
    text: str,
    ml_entities: list,
    rule_entities: list,
    sections: list = None,
) -> tuple[list[MergedEntity], list[MergedEntity]]:
    """
    Merge ML + rule entities into final output.

    Steps:
    1. Deduplicate exact matches
    2. Resolve overlapping spans
    3. Apply confidence routing
    4. Add negation status
    5. Add section labels and context

    Args:
        text: Original clinical note text.
        ml_entities: Entities from BioBERT predictor.
        rule_entities: Entities from rule extractor.
        sections: Detected sections from preprocessing.

    Returns:
        Tuple of (accepted_entities, flagged_entities).
    """
    # Step 1: Convert to common format
    all_entities = []
    for e in ml_entities:
        all_entities.append({
            "text": e.text,
            "label": e.label,
            "start": e.start,
            "end": e.end,
            "confidence": e.confidence,
            "source": "model",
        })
    for e in rule_entities:
        all_entities.append({
            "text": e.text,
            "label": e.label,
            "start": e.start,
            "end": e.end,
            "confidence": e.confidence,
            "source": "rules",
        })

    # Step 2: Deduplicate
    all_entities = _deduplicate(all_entities)

    # Step 3: Resolve overlaps
    all_entities = _resolve_overlaps(all_entities)

    # Step 4: Build final entities with metadata
    accepted = []
    flagged = []

    for e in all_entities:
        # Negation check
        neg = detect_negation(text, e["text"], e["start"], e["end"])

        # Section assignment
        section = _find_section(e["start"], sections) if sections else "Unknown"

        # Context snippet
        snippet = _get_context(text, e["start"], e["end"])

        # Confidence routing
        if e["confidence"] >= AUTO_ACCEPT_THRESHOLD:
            status = "auto_accepted"
        elif e["confidence"] >= FLAG_THRESHOLD:
            status = "flagged"
        else:
            status = "fallback"

        merged = MergedEntity(
            text=e["text"],
            label=e["label"],
            start=e["start"],
            end=e["end"],
            confidence=round(e["confidence"], 4),
            negated=neg.negated,
            section=section,
            source=e["source"],
            context_snippet=snippet,
            status=status,
        )

        if status == "flagged":
            flagged.append(merged)
        else:
            accepted.append(merged)

    return accepted, flagged


def _deduplicate(entities: list[dict]) -> list[dict]:
    """Remove exact-match duplicates, preferring model over rules."""
    seen = {}
    for e in entities:
        key = (e["text"].lower(), e["label"], e["start"], e["end"])
        if key not in seen:
            seen[key] = e
        elif e["source"] == "model":
            seen[key] = e  # Prefer model
    return list(seen.values())


def _resolve_overlaps(entities: list[dict]) -> list[dict]:
    """Resolve overlapping spans — prefer higher confidence."""
    if not entities:
        return entities

    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

    filtered = [entities[0]]
    for e in entities[1:]:
        prev = filtered[-1]
        if e["start"] >= prev["end"]:
            filtered.append(e)
        elif e["confidence"] > prev["confidence"]:
            filtered[-1] = e

    return filtered


def _find_section(position: int, sections: list) -> str:
    """Find which section an entity falls in."""
    if not sections:
        return "Unknown"
    for section in sections:
        if section.start <= position < section.end:
            return section.name
    return "Unknown"


def _get_context(text: str, start: int, end: int, window: int = 40) -> str:
    """Get a text snippet around the entity for context."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    snippet = text[ctx_start:ctx_end]
    if ctx_start > 0:
        snippet = "..." + snippet
    if ctx_end < len(text):
        snippet = snippet + "..."
    return snippet