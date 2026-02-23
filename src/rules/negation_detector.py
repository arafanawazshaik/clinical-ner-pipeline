"""
NegEx-Style Negation Detector.

Detects whether clinical entities are negated (absent) or affirmed (present).
Uses pre-negation and post-negation trigger patterns with a sliding window
to determine negation scope.

Examples:
    "no evidence of pneumonia"         → NEGATED
    "denies chest pain"                → NEGATED
    "patient has diabetes"             → PRESENT
    "pneumonia ruled out"              → NEGATED
"""

import re
from dataclasses import dataclass


@dataclass
class NegationResult:
    """Result of negation detection for an entity."""
    entity_text: str
    negated: bool
    trigger: str        # The negation trigger word/phrase
    trigger_type: str   # "pre" or "post" or "none"


# ── Negation Trigger Patterns ─────────────────────────────

PRE_NEGATION_TRIGGERS = [
    "no evidence of",
    "no signs of",
    "no history of",
    "no known",
    "not consistent with",
    "not suggestive of",
    "no further",
    "without evidence of",
    "without signs of",
    "without",
    "denies any",
    "denies",
    "denied",
    "does not have",
    "did not have",
    "not have",
    "no",
    "not",
    "never",
    "none",
    "absence of",
    "absent",
    "negative for",
    "free of",
    "unremarkable for",
]

POST_NEGATION_TRIGGERS = [
    "ruled out",
    "has been ruled out",
    "was ruled out",
    "is ruled out",
    "unlikely",
    "is unlikely",
    "was excluded",
    "has been excluded",
    "not found",
    "was not found",
    "not seen",
    "was not seen",
    "not identified",
    "not present",
    "not demonstrated",
    "not detected",
]

# Sort by length (longest first) to match most specific trigger
PRE_NEGATION_TRIGGERS.sort(key=len, reverse=True)
POST_NEGATION_TRIGGERS.sort(key=len, reverse=True)

# Default negation window: how many words from trigger to look for entity
DEFAULT_WINDOW = 7


def detect_negation(
    text: str,
    entity_text: str,
    entity_start: int,
    entity_end: int,
    window: int = DEFAULT_WINDOW,
) -> NegationResult:
    """
    Determine if an entity mention is negated in context.

    Checks:
    1. Pre-negation: trigger appears BEFORE entity within window
    2. Post-negation: trigger appears AFTER entity within window

    Args:
        text: Full clinical note text.
        entity_text: The entity string to check.
        entity_start: Character start position of entity.
        entity_end: Character end position of entity.
        window: Number of words to search for triggers.

    Returns:
        NegationResult with negation status.
    """
    # Get text window before entity
    pre_start = max(0, entity_start - 150)
    pre_text = text[pre_start:entity_start].lower().strip()

    # Get text window after entity
    post_end = min(len(text), entity_end + 150)
    post_text = text[entity_end:post_end].lower().strip()

    # Check pre-negation triggers
    pre_words = pre_text.split()
    pre_window_text = " ".join(pre_words[-window:]) if pre_words else ""

    for trigger in PRE_NEGATION_TRIGGERS:
        if trigger in pre_window_text:
            return NegationResult(
                entity_text=entity_text,
                negated=True,
                trigger=trigger,
                trigger_type="pre",
            )

    # Check post-negation triggers
    post_words = post_text.split()
    post_window_text = " ".join(post_words[:window]) if post_words else ""

    for trigger in POST_NEGATION_TRIGGERS:
        if trigger in post_window_text:
            return NegationResult(
                entity_text=entity_text,
                negated=True,
                trigger=trigger,
                trigger_type="post",
            )

    # No negation found
    return NegationResult(
        entity_text=entity_text,
        negated=False,
        trigger="",
        trigger_type="none",
    )


def detect_negations_batch(
    text: str,
    entities: list[dict],
    window: int = DEFAULT_WINDOW,
) -> list[NegationResult]:
    """
    Check negation for a list of entities.

    Args:
        text: Full clinical note text.
        entities: List of entity dicts with text, start, end keys.
        window: Negation window size.

    Returns:
        List of NegationResult objects.
    """
    results = []
    for entity in entities:
        result = detect_negation(
            text=text,
            entity_text=entity["text"],
            entity_start=entity["start"],
            entity_end=entity["end"],
            window=window,
        )
        results.append(result)
    return results