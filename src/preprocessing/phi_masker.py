"""
PHI Masker for Clinical Notes.

Detects and masks Protected Health Information using regex patterns.
Fast, deterministic, and auditable — no ML required.
Masks: names, dates, MRNs, SSNs, phone numbers, emails, ages over 89.
"""

import re
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PHI_PLACEHOLDER


@dataclass
class PHISpan:
    """A detected PHI span with its type and position."""
    text: str
    phi_type: str
    start: int
    end: int


# ── PHI Regex Patterns ────────────────────────────────────

PHI_PATTERNS = {
    "MRN": re.compile(
        r"\bMRN[-:\s]?\d{6,10}\b"
    ),
    "SSN": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "PHONE": re.compile(
        r"\b(?:\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b"
    ),
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "DATE": re.compile(
        r"\b(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"       # MM/DD/YYYY or MM-DD-YY
        r"|\d{4}-\d{2}-\d{2}"                    # YYYY-MM-DD
        r"|(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s*\d{4}"  # Month DD, YYYY
        r"|(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}"              # Month YYYY
        r")\b"
    ),
    "AGE_OVER_89": re.compile(
        r"\b(?:9[0-9]|1[0-9]{2})\s*(?:year|yr|y/?o|y\.o\.)"
    ),
    "ZIP": re.compile(
        r"\b\d{5}(?:-\d{4})?\b"
    ),
}

# Common name prefixes that indicate a person's name follows
NAME_PREFIXES = re.compile(
    r"(?:Patient|Pt|Dr|Mr|Mrs|Ms|Miss)[.:]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)


def detect_phi(text: str) -> list[PHISpan]:
    """
    Detect all PHI spans in text using regex patterns.

    Returns:
        List of PHISpan objects sorted by position.
    """
    spans = []

    # Pattern-based detection
    for phi_type, pattern in PHI_PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append(PHISpan(
                text=match.group(),
                phi_type=phi_type,
                start=match.start(),
                end=match.end(),
            ))

    # Name detection
    for match in NAME_PREFIXES.finditer(text):
        name = match.group(1)
        name_start = match.start(1)
        name_end = match.end(1)
        spans.append(PHISpan(
            text=name,
            phi_type="NAME",
            start=name_start,
            end=name_end,
        ))

    # Sort by position and remove overlaps
    spans.sort(key=lambda s: (s.start, -s.end))
    spans = _remove_overlaps(spans)

    return spans


def _remove_overlaps(spans: list[PHISpan]) -> list[PHISpan]:
    """Remove overlapping spans, keeping the longest match."""
    if not spans:
        return spans

    filtered = [spans[0]]
    for span in spans[1:]:
        if span.start >= filtered[-1].end:
            filtered.append(span)
        elif (span.end - span.start) > (filtered[-1].end - filtered[-1].start):
            filtered[-1] = span

    return filtered


def mask_phi(text: str, placeholder: str = PHI_PLACEHOLDER) -> tuple[str, list[PHISpan]]:
    """
    Detect and mask all PHI in text.

    Args:
        text: Clinical note text.
        placeholder: Replacement string for PHI (default: "[PHI]").

    Returns:
        Tuple of (masked_text, list of detected PHI spans).
    """
    spans = detect_phi(text)

    # Replace from end to start to preserve positions
    masked = text
    for span in reversed(spans):
        typed_placeholder = f"{placeholder}"
        masked = masked[:span.start] + typed_placeholder + masked[span.end:]

    return masked, spans