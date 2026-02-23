"""
Shared Dependencies.

Loads the NER model once at startup and shares it across requests.
"""

from src.inference.predictor import NERPredictor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Singleton — loaded once, shared across all requests
_predictor = None


def get_predictor() -> NERPredictor:
    """Get or create the shared NER predictor."""
    global _predictor
    if _predictor is None:
        _predictor = NERPredictor()
    return _predictor