"""
Clinical-Aware Sentence Splitter.

Splits clinical notes into sentences, handling medical formatting
that breaks standard sentence splitters (numbered lists, vitals,
abbreviations with periods, section headers).
"""

import re


def split_sentences(text: str) -> list[str]:
    """
    Split clinical text into sentences with medical-aware rules.

    Handles:
        - Standard period/question/exclamation endings
        - Numbered lists (1. Item  2. Item)
        - Section headers (ASSESSMENT AND PLAN:)
        - Vitals lines (T 98.6F, HR 72, BP 120/80)
        - Abbreviations with periods (Dr. Smith, e.g., etc.)

    Args:
        text: Clinical note text.

    Returns:
        List of sentence strings.
    """
    # Protect abbreviations with periods from splitting
    protected = _protect_abbreviations(text)

    # Split on sentence boundaries
    raw_sentences = _split_on_boundaries(protected)

    # Restore protected abbreviations and clean up
    sentences = []
    for sent in raw_sentences:
        sent = _restore_abbreviations(sent)
        sent = sent.strip()
        if sent:
            sentences.append(sent)

    return sentences


# ── Abbreviations that contain periods ────────────────────

PROTECTED_ABBREVIATIONS = [
    "Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.",
    "vs.", "etc.", "e.g.", "i.e.", "approx.",
    "dept.", "hosp.", "pt.", "Pt.",
    "a.m.", "p.m.", "y.o.", "h/o.",
    "Temp.", "temp.",
]

# Placeholder that won't appear in real text
_PERIOD_PLACEHOLDER = "\x00PERIOD\x00"


def _protect_abbreviations(text: str) -> str:
    """Replace periods in known abbreviations with placeholder."""
    for abbrev in PROTECTED_ABBREVIATIONS:
        safe = abbrev.replace(".", _PERIOD_PLACEHOLDER)
        text = text.replace(abbrev, safe)
    return text


def _restore_abbreviations(text: str) -> str:
    """Restore placeholders back to periods."""
    return text.replace(_PERIOD_PLACEHOLDER, ".")


def _split_on_boundaries(text: str) -> list[str]:
    """
    Split text on sentence boundaries.

    Boundaries:
        - Period/question/exclamation followed by space + uppercase
        - Newlines (clinical notes use newlines as boundaries)
        - Numbered list items
    """
    sentences = []
    current = []

    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            # Empty line = paragraph break, flush current sentence
            if current:
                sentences.append(" ".join(current))
                current = []
            continue

        # Check if line is a section header (ALL CAPS with colon)
        if _is_section_header(line):
            if current:
                sentences.append(" ".join(current))
                current = []
            sentences.append(line)
            continue

        # Check if line is a numbered list item
        if _is_list_item(line):
            if current:
                sentences.append(" ".join(current))
                current = []
            sentences.append(line)
            continue

        # Check if line is a vitals line
        if _is_vitals_line(line):
            if current:
                sentences.append(" ".join(current))
                current = []
            sentences.append(line)
            continue

        # Split within the line on sentence-ending punctuation
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
        for part in parts:
            current.append(part)

    # Flush remaining
    if current:
        sentences.append(" ".join(current))

    return sentences


def _is_section_header(line: str) -> bool:
    """Check if line is a clinical section header."""
    # ALL CAPS with optional colon: "ASSESSMENT AND PLAN:" or "MEDICATIONS"
    if re.match(r"^[A-Z][A-Z\s&/]+:?\s*$", line):
        return True
    # Title case with colon: "History of Present Illness:"
    if re.match(r"^[A-Z][a-zA-Z\s]+:\s*$", line):
        return True
    return False


def _is_list_item(line: str) -> bool:
    """Check if line is a numbered or bulleted list item."""
    return bool(re.match(r"^\s*(?:\d+[.)\-]|[-*#])\s+", line))


def _is_vitals_line(line: str) -> bool:
    """Check if line contains vital signs."""
    vitals_pattern = r"(?:Vitals?|T\s+\d|HR\s+\d|BP\s+\d|RR\s+\d|SpO2|Temp\s+\d)"
    return bool(re.search(vitals_pattern, line))