"""
Shared test fixtures.

Provides sample clinical notes, expected entities, and mock data
reusable across all test modules.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_note():
    """A simple clinical note for testing."""
    return (
        "HISTORY OF PRESENT ILLNESS:\n"
        "Patient John Smith (MRN-123456) is a 65-year-old male with "
        "type 2 diabetes mellitus on metformin 500mg twice daily.\n"
        "No evidence of pulmonary embolism.\n"
        "Admitted on 01/15/2022.\n\n"
        "MEDICATIONS:\n"
        "1. metformin 500mg twice daily\n"
        "2. lisinopril 10mg daily"
    )


@pytest.fixture
def empty_note():
    """An empty clinical note."""
    return ""


@pytest.fixture
def negated_note():
    """A note with negated entities."""
    return "Patient denies chest pain. No history of diabetes. COPD ruled out."


@pytest.fixture
def sample_entities():
    """Expected entity labels from sample_note."""
    return [
        {"text": "type 2 diabetes mellitus", "label": "DIAGNOSIS"},
        {"text": "metformin", "label": "MEDICATION"},
        {"text": "500mg twice daily", "label": "DOSAGE"},
        {"text": "pulmonary embolism", "label": "DIAGNOSIS"},
    ]