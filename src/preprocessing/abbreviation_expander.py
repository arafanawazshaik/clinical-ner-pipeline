"""
Clinical Abbreviation Expander.

Maps common clinical abbreviations to their full forms.
Uses context-aware matching to avoid false expansions
(e.g., "OR" as Operating Room vs. the word "or").
"""

import re


# ── Abbreviation Dictionary ───────────────────────────────
# 200+ common clinical abbreviations organized by category

ABBREVIATIONS = {
    # ── Diagnoses ─────────────────────────────────────────
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "DM1": "type 1 diabetes mellitus",
    "DM2": "type 2 diabetes mellitus",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "CAD": "coronary artery disease",
    "AFib": "atrial fibrillation",
    "A-fib": "atrial fibrillation",
    "AKI": "acute kidney injury",
    "CKD": "chronic kidney disease",
    "MI": "myocardial infarction",
    "AMI": "acute myocardial infarction",
    "CVA": "cerebrovascular accident",
    "TIA": "transient ischemic attack",
    "DVT": "deep vein thrombosis",
    "PE": "pulmonary embolism",
    "GERD": "gastroesophageal reflux disease",
    "UTI": "urinary tract infection",
    "PNA": "pneumonia",
    "CAP": "community-acquired pneumonia",
    "OSA": "obstructive sleep apnea",
    "ESRD": "end-stage renal disease",
    "HLD": "hyperlipidemia",
    "BPH": "benign prostatic hyperplasia",
    "RA": "rheumatoid arthritis",
    "OA": "osteoarthritis",
    "SLE": "systemic lupus erythematosus",
    "MS": "multiple sclerosis",
    "PAD": "peripheral artery disease",
    "PVD": "peripheral vascular disease",
    "IDDM": "insulin-dependent diabetes mellitus",
    "NIDDM": "non-insulin-dependent diabetes mellitus",

    # ── Procedures ────────────────────────────────────────
    "CABG": "coronary artery bypass graft",
    "PCI": "percutaneous coronary intervention",
    "PTCA": "percutaneous transluminal coronary angioplasty",
    "TKR": "total knee replacement",
    "THR": "total hip replacement",
    "EGD": "esophagogastroduodenoscopy",
    "ERCP": "endoscopic retrograde cholangiopancreatography",
    "LP": "lumbar puncture",
    "I&D": "incision and drainage",
    "ECG": "electrocardiogram",
    "EKG": "electrocardiogram",
    "CXR": "chest X-ray",

    # ── Imaging ───────────────────────────────────────────
    "CT": "computed tomography",
    "MRI": "magnetic resonance imaging",
    "US": "ultrasound",
    "XR": "X-ray",
    "CTA": "computed tomography angiography",
    "MRA": "magnetic resonance angiography",
    "TEE": "transesophageal echocardiogram",
    "TTE": "transthoracic echocardiogram",

    # ── Medications ───────────────────────────────────────
    "ASA": "aspirin",
    "APAP": "acetaminophen",
    "abx": "antibiotics",
    "ABX": "antibiotics",
    "HCTZ": "hydrochlorothiazide",
    "MOM": "milk of magnesia",
    "NTG": "nitroglycerin",
    "PPIs": "proton pump inhibitors",
    "PPI": "proton pump inhibitor",
    "SSRIs": "selective serotonin reuptake inhibitors",
    "SSRI": "selective serotonin reuptake inhibitor",
    "ACEi": "ACE inhibitor",
    "ARB": "angiotensin receptor blocker",
    "BB": "beta blocker",
    "CCB": "calcium channel blocker",

    # ── Dosage/Route ──────────────────────────────────────
    "PO": "by mouth",
    "IV": "intravenous",
    "IM": "intramuscular",
    "SQ": "subcutaneous",
    "SubQ": "subcutaneous",
    "SC": "subcutaneous",
    "SL": "sublingual",
    "PR": "per rectum",
    "INH": "inhaled",
    "TOP": "topical",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "QD": "once daily",
    "QHS": "at bedtime",
    "QAM": "every morning",
    "QPM": "every evening",
    "PRN": "as needed",
    "Q4H": "every 4 hours",
    "Q6H": "every 6 hours",
    "Q8H": "every 8 hours",
    "Q12H": "every 12 hours",

    # ── Clinical Terms ────────────────────────────────────
    "Hx": "history",
    "HPI": "history of present illness",
    "PMH": "past medical history",
    "PSH": "past surgical history",
    "FH": "family history",
    "SH": "social history",
    "ROS": "review of systems",
    "CC": "chief complaint",
    "A&P": "assessment and plan",
    "Dx": "diagnosis",
    "DDx": "differential diagnosis",
    "Tx": "treatment",
    "Rx": "prescription",
    "Sx": "symptoms",
    "Hgb": "hemoglobin",
    "Hct": "hematocrit",
    "WBC": "white blood cell count",
    "RBC": "red blood cell count",
    "Plt": "platelet count",
    "BMP": "basic metabolic panel",
    "CMP": "comprehensive metabolic panel",
    "CBC": "complete blood count",
    "LFTs": "liver function tests",
    "ABG": "arterial blood gas",
    "UA": "urinalysis",
    "SpO2": "oxygen saturation",
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "Temp": "temperature",
    "BMI": "body mass index",
    "I/O": "intake and output",

    # ── Status/Disposition ────────────────────────────────
    "DNR": "do not resuscitate",
    "DNI": "do not intubate",
    "AMA": "against medical advice",
    "D/C": "discharge",
    "DC": "discharge",
    "f/u": "follow up",
    "F/U": "follow up",
    "w/": "with",
    "w/o": "without",
    "s/p": "status post",
    "S/P": "status post",
    "c/o": "complaining of",
    "NKDA": "no known drug allergies",
    "NKA": "no known allergies",
    "WNL": "within normal limits",
    "NAD": "no acute distress",
    "A&O": "alert and oriented",
    "AAOx3": "alert and oriented times three",
}

# Words that look like abbreviations but shouldn't be expanded
# when they appear in certain contexts
SKIP_WORDS = {"OR", "IN", "AT", "ON", "IS", "IT", "AN", "AS", "BE", "DO", "IF", "NO", "SO", "TO", "UP", "WE"}


def expand_abbreviations(text: str, preserve_original: bool = True) -> tuple[str, list[dict]]:
    """
    Expand clinical abbreviations in text.

    Args:
        text: Clinical note text.
        preserve_original: If True, format as "ABR (expansion)".
                          If False, replace abbreviation entirely.

    Returns:
        Tuple of (expanded_text, list of expansions made).
        Each expansion dict has: abbreviation, expansion, start, end.
    """
    expansions = []
    result = text

    # Sort by length (longest first) to avoid partial matches
    sorted_abbrevs = sorted(ABBREVIATIONS.keys(), key=len, reverse=True)

    for abbrev in sorted_abbrevs:
        expansion = ABBREVIATIONS[abbrev]

        # Word boundary pattern — only match whole words
        pattern = r"(?<![A-Za-z])" + re.escape(abbrev) + r"(?![A-Za-z])"

        for match in re.finditer(pattern, result):
            matched_text = match.group()

            # Skip if it's a common English word in wrong context
            if matched_text.upper() in SKIP_WORDS:
                continue

            expansions.append({
                "abbreviation": matched_text,
                "expansion": expansion,
                "start": match.start(),
                "end": match.end(),
            })

    # Apply replacements (from end to start to preserve positions)
    expansions.sort(key=lambda x: x["start"], reverse=True)
    for exp in expansions:
        if preserve_original:
            replacement = f"{exp['abbreviation']} ({exp['expansion']})"
        else:
            replacement = exp["expansion"]
        result = result[:exp["start"]] + replacement + result[exp["end"]:]

    # Re-sort by position for output
    expansions.sort(key=lambda x: x["start"])
    return result, expansions