"""
Clinical Section Detector.

Identifies clinical note sections (HPI, Medications, Assessment, etc.)
using keyword/header matching. Returns section boundaries so downstream
components know which part of the note they're processing.
"""

import re
from dataclasses import dataclass


@dataclass
class Section:
    """A detected clinical section."""
    name: str
    start: int
    end: int
    text: str


# ── Section Header Patterns ───────────────────────────────

SECTION_HEADERS = {
    "Chief Complaint": [
        r"CHIEF\s+COMPLAINT",
        r"CC:",
        r"Reason\s+for\s+(?:Visit|Consultation)",
    ],
    "History of Present Illness": [
        r"HISTORY\s+OF\s+PRESENT\s+ILLNESS",
        r"HPI:",
        r"SUBJECTIVE:",
    ],
    "Past Medical History": [
        r"PAST\s+MEDICAL\s+HISTORY",
        r"PMH:",
        r"MEDICAL\s+HISTORY",
    ],
    "Medications": [
        r"MEDICATIONS?(?:\s+ON\s+ADMISSION)?",
        r"CURRENT\s+MEDICATIONS?",
        r"HOME\s+MEDICATIONS?",
        r"DISCHARGE\s+MEDICATIONS?",
    ],
    "Allergies": [
        r"ALLERGIES",
        r"DRUG\s+ALLERGIES",
    ],
    "Review of Systems": [
        r"REVIEW\s+OF\s+SYSTEMS",
        r"ROS:",
    ],
    "Physical Examination": [
        r"PHYSICAL\s+EXAM(?:INATION)?",
        r"OBJECTIVE:",
        r"PE:",
    ],
    "Assessment": [
        r"ASSESSMENT(?:\s+AND\s+PLAN)?",
        r"A(?:&|/)P:",
        r"IMPRESSION(?:\s+AND\s+PLAN)?",
    ],
    "Plan": [
        r"PLAN:",
        r"TREATMENT\s+PLAN",
        r"RECOMMENDATIONS?:",
    ],
    "Procedures": [
        r"PROCEDURES?:",
        r"OPERATIONS?\s+PERFORMED",
    ],
    "Hospital Course": [
        r"HOSPITAL\s+COURSE",
        r"CLINICAL\s+COURSE",
    ],
    "Discharge Instructions": [
        r"DISCHARGE\s+INSTRUCTIONS?",
    ],
    "Follow-Up": [
        r"FOLLOW[\s-]?UP",
    ],
    "Operative Findings": [
        r"OPERATIVE\s+FINDINGS?",
        r"FINDINGS?:",
        r"INTRAOPERATIVE\s+FINDINGS?",
    ],
    "Preoperative Diagnosis": [
        r"PRE[\s-]?OPERATIVE\s+DIAGNOS[IE]S",
    ],
    "Postoperative Diagnosis": [
        r"POST[\s-]?OPERATIVE\s+DIAGNOS[IE]S",
    ],
}

# Compile all patterns
_COMPILED_PATTERNS = {}
for section_name, patterns in SECTION_HEADERS.items():
    combined = "|".join(patterns)
    _COMPILED_PATTERNS[section_name] = re.compile(
        r"^\s*(?:" + combined + r")\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


def detect_sections(text: str) -> list[Section]:
    """
    Detect clinical sections in text.

    Scans for section headers and returns boundaries.
    Each section runs from its header to the next header (or end of text).

    Args:
        text: Clinical note text.

    Returns:
        List of Section objects sorted by position.
    """
    matches = []

    for section_name, pattern in _COMPILED_PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append({
                "name": section_name,
                "header_start": match.start(),
                "header_end": match.end(),
            })

    # Sort by position
    matches.sort(key=lambda m: m["header_start"])

    # Build sections with boundaries
    sections = []
    for i, match in enumerate(matches):
        start = match["header_end"]

        # Section ends at next header or end of text
        if i + 1 < len(matches):
            end = matches[i + 1]["header_start"]
        else:
            end = len(text)

        section_text = text[start:end].strip()

        sections.append(Section(
            name=match["name"],
            start=match["header_start"],
            end=end,
            text=section_text,
        ))

    return sections


def get_section_map(text: str) -> dict[str, str]:
    """
    Get a simple section name → text mapping.

    Args:
        text: Clinical note text.

    Returns:
        Dict mapping section names to their text content.
    """
    sections = detect_sections(text)
    return {s.name: s.text for s in sections}