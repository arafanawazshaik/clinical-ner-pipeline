"""
Rule-Based Entity Extractor.

Extracts clinical entities using pattern matching and dictionary lookup.
Complements the ML model by providing deterministic extraction for
well-defined patterns (medications, dosages, dates) and serves as
a fallback when model confidence is low.
"""

import re
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rules.negation_detector import detect_negation


@dataclass
class RuleEntity:
    """An entity extracted by rules."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    negated: bool
    source: str = "rules"


# ── Entity Patterns ───────────────────────────────────────

DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|mL|units?|mEq)"
    r"(?:\s+(?:PO|IV|IM|SQ|SubQ|SC|SL|PR|INH|TOP))?"
    r"(?:\s+(?:daily|once daily|twice daily|BID|TID|QID|QD|"
    r"QHS|PRN|every\s+\d+\s+hours?|Q\d+H))?"
    r"(?:\s+(?:PRN|as needed))?",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s*\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}"
    r")\b"
)

# Common medication names for dictionary lookup
MEDICATION_LIST = [
    "metformin", "lisinopril", "atorvastatin", "amlodipine",
    "metoprolol", "omeprazole", "furosemide", "warfarin",
    "levothyroxine", "albuterol", "prednisone", "gabapentin",
    "sertraline", "acetaminophen", "aspirin", "clopidogrel",
    "enoxaparin", "insulin glargine", "amoxicillin", "ciprofloxacin",
    "hydrochlorothiazide", "pantoprazole", "apixaban",
    "ceftriaxone", "vancomycin", "heparin", "morphine",
    "lorazepam", "ondansetron", "famotidine",
]

# Common diagnosis patterns
DIAGNOSIS_LIST = [
    "type 2 diabetes mellitus", "type 1 diabetes mellitus",
    "essential hypertension", "congestive heart failure",
    "chronic obstructive pulmonary disease", "atrial fibrillation",
    "coronary artery disease", "acute kidney injury",
    "chronic kidney disease", "deep vein thrombosis",
    "pulmonary embolism", "urinary tract infection",
    "community-acquired pneumonia", "myocardial infarction",
    "gastroesophageal reflux disease", "iron deficiency anemia",
    "major depressive disorder", "obstructive sleep apnea",
    "peripheral artery disease", "sepsis",
]


def extract_entities_by_rules(text: str) -> list[RuleEntity]:
    """
    Extract all entities from text using rule-based patterns.

    Combines:
    - Regex patterns for dosages and dates
    - Dictionary lookup for medications and diagnoses
    - Negation detection for each found entity

    Args:
        text: Clinical note text (preprocessed).

    Returns:
        List of RuleEntity objects.
    """
    entities = []

    # Extract dosages
    for match in DOSAGE_PATTERN.finditer(text):
        entities.append(_make_entity(text, match, "DOSAGE"))

    # Extract dates
    for match in DATE_PATTERN.finditer(text):
        entities.append(_make_entity(text, match, "DATE"))

    # Extract medications
    for med in MEDICATION_LIST:
        for match in re.finditer(re.escape(med), text, re.IGNORECASE):
            entities.append(_make_entity(text, match, "MEDICATION"))

    # Extract diagnoses
    for dx in DIAGNOSIS_LIST:
        for match in re.finditer(re.escape(dx), text, re.IGNORECASE):
            entities.append(_make_entity(text, match, "DIAGNOSIS"))

    # Remove duplicates and overlaps
    entities = _remove_overlapping(entities)

    return entities


def _make_entity(text: str, match: re.Match, label: str) -> RuleEntity:
    """Create a RuleEntity from a regex match with negation check."""
    neg_result = detect_negation(
        text=text,
        entity_text=match.group(),
        entity_start=match.start(),
        entity_end=match.end(),
    )

    return RuleEntity(
        text=match.group(),
        label=label,
        start=match.start(),
        end=match.end(),
        confidence=0.80,
        negated=neg_result.negated,
    )


def _remove_overlapping(entities: list[RuleEntity]) -> list[RuleEntity]:
    """Remove overlapping entities, keeping the longest match."""
    if not entities:
        return entities

    # Sort by start position, then by length (longest first)
    entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

    filtered = [entities[0]]
    for entity in entities[1:]:
        if entity.start >= filtered[-1].end:
            filtered.append(entity)
        elif (entity.end - entity.start) > (filtered[-1].end - filtered[-1].start):
            filtered[-1] = entity

    return filtered