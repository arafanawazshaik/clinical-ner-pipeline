"""Tests for rule-based extraction and negation detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rules.negation_detector import detect_negation
from src.rules.rule_extractor import extract_entities_by_rules


class TestNegationDetector:
    def test_pre_negation(self):
        text = "No evidence of pneumonia"
        result = detect_negation(text, "pneumonia", 15, 24)
        assert result.negated is True
        assert result.trigger_type == "pre"

    def test_post_negation(self):
        text = "Pneumonia was ruled out"
        result = detect_negation(text, "Pneumonia", 0, 9)
        assert result.negated is True
        assert result.trigger_type == "post"

    def test_no_negation(self):
        text = "Patient has pneumonia"
        result = detect_negation(text, "pneumonia", 12, 21)
        assert result.negated is False

    def test_denies_pattern(self):
        text = "Patient denies chest pain"
        result = detect_negation(text, "chest pain", 15, 25)
        assert result.negated is True


class TestRuleExtractor:
    def test_extracts_medication(self):
        entities = extract_entities_by_rules("Patient takes metformin daily")
        labels = [e.label for e in entities]
        assert "MEDICATION" in labels

    def test_extracts_dosage(self):
        entities = extract_entities_by_rules("Take 500mg twice daily")
        labels = [e.label for e in entities]
        assert "DOSAGE" in labels

    def test_extracts_date(self):
        entities = extract_entities_by_rules("Admitted on 01/15/2022")
        labels = [e.label for e in entities]
        assert "DATE" in labels

    def test_extracts_diagnosis(self):
        entities = extract_entities_by_rules("Diagnosed with type 2 diabetes mellitus")
        labels = [e.label for e in entities]
        assert "DIAGNOSIS" in labels

    def test_negated_entity(self):
        entities = extract_entities_by_rules("No evidence of pulmonary embolism")
        negated = [e for e in entities if e.negated]
        assert len(negated) > 0

    def test_empty_text(self):
        entities = extract_entities_by_rules("")
        assert len(entities) == 0