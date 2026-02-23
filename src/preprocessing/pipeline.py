"""
Preprocessing Pipeline Orchestrator.

Runs the full preprocessing flow on a clinical note:
clean → expand abbreviations → mask PHI → split sentences → detect sections.
Each step is independent and can be run individually or as a full pipeline.
"""

import time
from dataclasses import dataclass, field

from src.preprocessing.cleaner import clean_text
from src.preprocessing.abbreviation_expander import expand_abbreviations
from src.preprocessing.phi_masker import mask_phi
from src.preprocessing.sentence_splitter import split_sentences
from src.preprocessing.section_detector import detect_sections


@dataclass
class PreprocessedNote:
    """Output of the preprocessing pipeline."""
    original_text: str
    cleaned_text: str
    sentences: list[str] = field(default_factory=list)
    sections: list = field(default_factory=list)
    phi_spans: list = field(default_factory=list)
    abbreviations_expanded: list = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cleaned_text": self.cleaned_text,
            "num_sentences": len(self.sentences),
            "num_sections": len(self.sections),
            "sections_found": [s.name for s in self.sections],
            "phi_masked_count": len(self.phi_spans),
            "abbreviations_expanded_count": len(self.abbreviations_expanded),
            "processing_time_ms": self.processing_time_ms,
        }


def preprocess_note(
    text: str,
    expand_abbrevs: bool = True,
    mask_phi_flag: bool = True,
    preserve_abbrev_original: bool = True,
) -> PreprocessedNote:
    """
    Run the full preprocessing pipeline on a single clinical note.

    Args:
        text: Raw clinical note text.
        expand_abbrevs: Whether to expand abbreviations.
        mask_phi_flag: Whether to mask PHI.
        preserve_abbrev_original: If True, keeps "HTN (hypertension)".

    Returns:
        PreprocessedNote with all processing results.
    """
    start_time = time.time()

    result = PreprocessedNote(original_text=text, cleaned_text=text)

    # Step 1: Clean text
    result.cleaned_text = clean_text(text)

    # Step 2: Expand abbreviations
    if expand_abbrevs:
        result.cleaned_text, result.abbreviations_expanded = expand_abbreviations(
            result.cleaned_text,
            preserve_original=preserve_abbrev_original,
        )

    # Step 3: Mask PHI
    if mask_phi_flag:
        result.cleaned_text, result.phi_spans = mask_phi(result.cleaned_text)

    # Step 4: Split sentences
    result.sentences = split_sentences(result.cleaned_text)

    # Step 5: Detect sections
    result.sections = detect_sections(result.cleaned_text)

    # Record processing time
    elapsed = (time.time() - start_time) * 1000
    result.processing_time_ms = round(elapsed, 2)

    return result


def preprocess_batch(
    texts: list[str],
    **kwargs,
) -> list[PreprocessedNote]:
    """
    Preprocess a batch of clinical notes.

    Args:
        texts: List of raw clinical note texts.
        **kwargs: Passed to preprocess_note.

    Returns:
        List of PreprocessedNote objects.
    """
    results = []
    for text in texts:
        result = preprocess_note(text, **kwargs)
        results.append(result)
    return results