"""
Label schema for Clinical NER Pipeline.
Entity types, BIO tag mappings, and output JSON schema definitions.
"""

# ── Entity Types ──────────────────────────────────────────
ENTITY_TYPES = [
    "DIAGNOSIS",
    "MEDICATION",
    "PROCEDURE",
    "DOSAGE",
    "DATE",
]

# ── BIO Tag Set ───────────────────────────────────────────
BIO_TAGS = ["O"]
for entity_type in ENTITY_TYPES:
    BIO_TAGS.append(f"B-{entity_type}")
    BIO_TAGS.append(f"I-{entity_type}")

# Tag ↔ ID mappings
TAG_TO_ID = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
ID_TO_TAG = {idx: tag for idx, tag in enumerate(BIO_TAGS)}
NUM_LABELS = len(BIO_TAGS)

# ── Clinical Sections ─────────────────────────────────────
CLINICAL_SECTIONS = [
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Medications",
    "Allergies",
    "Review of Systems",
    "Physical Examination",
    "Assessment",
    "Plan",
    "Procedures",
    "Hospital Course",
    "Discharge Medications",
    "Discharge Instructions",
    "Follow-Up",
    "Operative Findings",
    "Preoperative Diagnosis",
    "Postoperative Diagnosis",
]

# ── Section → Likely Entity Mappings ──────────────────────
SECTION_ENTITY_MAP = {
    "Chief Complaint":            ["DIAGNOSIS"],
    "History of Present Illness": ["DIAGNOSIS", "MEDICATION", "DATE", "PROCEDURE"],
    "Past Medical History":       ["DIAGNOSIS", "DATE", "PROCEDURE"],
    "Medications":                ["MEDICATION", "DOSAGE"],
    "Allergies":                  ["MEDICATION"],
    "Assessment":                 ["DIAGNOSIS"],
    "Plan":                       ["MEDICATION", "PROCEDURE", "DOSAGE"],
    "Procedures":                 ["PROCEDURE", "DATE"],
    "Hospital Course":            ["DIAGNOSIS", "MEDICATION", "PROCEDURE", "DATE"],
    "Discharge Medications":      ["MEDICATION", "DOSAGE"],
    "Operative Findings":         ["DIAGNOSIS", "PROCEDURE"],
    "Preoperative Diagnosis":     ["DIAGNOSIS"],
    "Postoperative Diagnosis":    ["DIAGNOSIS"],
}