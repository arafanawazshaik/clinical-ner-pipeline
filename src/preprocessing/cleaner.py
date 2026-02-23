"""
Text Cleaner for Clinical Notes.

Fixes encoding issues, normalizes whitespace, strips artifacts,
and prepares raw clinical text for downstream processing.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single clinical note.

    Steps:
        1. Fix Unicode encoding issues
        2. Normalize whitespace
        3. Remove control characters
        4. Fix common OCR/EHR artifacts
        5. Normalize punctuation
        6. Strip leading/trailing whitespace

    Args:
        text: Raw clinical note text.

    Returns:
        Cleaned text string.
    """
    text = fix_encoding(text)
    text = remove_control_chars(text)
    text = fix_ehr_artifacts(text)
    text = normalize_whitespace(text)
    text = normalize_punctuation(text)
    text = text.strip()
    return text


def fix_encoding(text: str) -> str:
    """Fix common Unicode encoding problems."""
    # Normalize to NFC form (composed characters)
    text = unicodedata.normalize("NFC", text)

    # Fix common mojibake patterns (wrong encoding interpretations)
    replacements = {
        "\ufeff": "",     # BOM character
        "\x00": "",       # Null bytes
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters, keeping newlines and tabs."""
    cleaned = []
    for char in text:
        if char in ("\n", "\t", "\r"):
            cleaned.append(char)
        elif unicodedata.category(char) == "Cc":
            continue  # Skip control characters
        else:
            cleaned.append(char)
    return "".join(cleaned)


def fix_ehr_artifacts(text: str) -> str:
    """
    Fix common artifacts from EHR copy-paste and OCR.
    Clinical notes often have junk from system exports.
    """
    # Remove page headers/footers like "Page 1 of 3"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text)

    # Remove repeated separator lines (===, ---, ___)
    text = re.sub(r"[=\-_]{3,}", "", text)

    # Remove timestamps stuck to text like "[12:34:56]"
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", text)

    # Remove system tags like "*** DRAFT ***" or "<<UNSIGNED>>"
    text = re.sub(r"\*{2,}.*?\*{2,}", "", text)
    text = re.sub(r"<<.*?>>", "", text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving clinical note structure.
    Keeps single newlines (section breaks) but removes excessive blank lines.
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces on each line
    text = re.sub(r" +\n", "\n", text)

    return text


def normalize_punctuation(text: str) -> str:
    """Normalize fancy quotes and dashes to standard ASCII."""
    replacements = {
        "\u2018": "'",   # Left single quote
        "\u2019": "'",   # Right single quote
        "\u201c": '"',   # Left double quote
        "\u201d": '"',   # Right double quote
        "\u2013": "-",   # En dash
        "\u2014": "-",   # Em dash
        "\u2026": "...", # Ellipsis
    }
    for fancy, plain in replacements.items():
        text = text.replace(fancy, plain)
    return text