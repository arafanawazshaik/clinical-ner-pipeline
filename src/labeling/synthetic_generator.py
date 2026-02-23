"""
Synthetic Clinical Note Generator.

Generates realistic clinical notes with known ground-truth entity annotations.
Uses template-based generation with randomized entity insertion to produce
BIO-tagged training data across 5 clinical note types.
"""

import random
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# ── Entity Pools ──────────────────────────────────────────
# Representative samples from ICD-10, RxNorm, CPT

DIAGNOSES = [
    "type 2 diabetes mellitus",
    "essential hypertension",
    "congestive heart failure",
    "chronic obstructive pulmonary disease",
    "atrial fibrillation",
    "coronary artery disease",
    "acute kidney injury",
    "major depressive disorder",
    "community-acquired pneumonia",
    "deep vein thrombosis",
    "pulmonary embolism",
    "iron deficiency anemia",
    "chronic kidney disease stage 3",
    "gastroesophageal reflux disease",
    "urinary tract infection",
    "cellulitis of the right lower extremity",
    "acute myocardial infarction",
    "sepsis",
    "hypothyroidism",
    "osteoarthritis of the right knee",
    "peripheral artery disease",
    "hepatitis C",
    "systemic lupus erythematosus",
    "obstructive sleep apnea",
    "alcohol use disorder",
]

MEDICATIONS = [
    "metformin",
    "lisinopril",
    "atorvastatin",
    "amlodipine",
    "metoprolol succinate",
    "omeprazole",
    "furosemide",
    "warfarin",
    "levothyroxine",
    "albuterol",
    "prednisone",
    "gabapentin",
    "sertraline",
    "acetaminophen",
    "aspirin",
    "clopidogrel",
    "enoxaparin",
    "insulin glargine",
    "amoxicillin",
    "ciprofloxacin",
    "hydrochlorothiazide",
    "pantoprazole",
    "apixaban",
    "ceftriaxone",
    "vancomycin",
]

PROCEDURES = [
    "coronary artery bypass graft",
    "MRI of the brain",
    "CT scan of the chest",
    "chest X-ray",
    "echocardiogram",
    "cardiac catheterization",
    "colonoscopy",
    "upper endoscopy",
    "total knee replacement",
    "appendectomy",
    "cholecystectomy",
    "lumbar puncture",
    "blood transfusion",
    "dialysis",
    "intubation and mechanical ventilation",
    "electrocardiogram",
    "carotid endarterectomy",
    "CT scan of the abdomen and pelvis",
    "bone marrow biopsy",
    "central line placement",
]

DOSAGES = [
    "500mg twice daily",
    "10mg PO daily",
    "20mg once daily",
    "40mg IV every 8 hours",
    "25mg PO BID",
    "100mg PO at bedtime",
    "5mg sublingual PRN",
    "1000mg IV every 12 hours",
    "81mg PO daily",
    "2.5mg PO daily",
    "50mg PO TID",
    "250mg PO every 6 hours",
    "75mg PO daily",
    "10 units subcutaneous at bedtime",
    "325mg PO every 4 hours PRN",
    "40mg PO once daily",
    "200mg PO BID",
    "1mg IV every 6 hours",
    "12.5mg PO daily",
    "5mg PO twice daily",
]

DATES = [
    "01/15/2022",
    "03/22/2021",
    "November 2020",
    "June 15, 2023",
    "2022-08-10",
    "December 2019",
    "07/03/2022",
    "April 2023",
    "02/28/2021",
    "September 10, 2022",
    "2021-11-05",
    "January 2022",
    "05/18/2023",
    "March 2020",
    "10/12/2021",
    "August 25, 2022",
    "2023-01-30",
    "February 2021",
    "08/07/2023",
    "July 4, 2022",
]

PATIENT_NAMES = [
    "John Smith", "Maria Garcia", "Robert Johnson", "Linda Williams",
    "Michael Brown", "Elizabeth Jones", "David Miller", "Jennifer Davis",
    "James Wilson", "Patricia Moore", "Charles Taylor", "Barbara Anderson",
]

DOCTOR_NAMES = [
    "Dr. Sarah Thompson", "Dr. James Chen", "Dr. Amanda Patel",
    "Dr. Michael Rodriguez", "Dr. Lisa Washington", "Dr. Robert Kim",
    "Dr. Emily Parker", "Dr. David Nguyen", "Dr. Rachel Foster",
]

MRNS = [f"MRN-{random.randint(100000, 999999)}" for _ in range(50)]

# ── Data Classes ──────────────────────────────────────────

@dataclass
class EntityAnnotation:
    """A single entity found in text with its character positions."""
    text: str       # the actual entity text, e.g. "type 2 diabetes mellitus"
    label: str      # entity type, e.g. "DIAGNOSIS"
    start: int      # character position where entity starts
    end: int        # character position where entity ends

    def to_dict(self):
        return asdict(self)


@dataclass
class SyntheticNote:
    """A complete generated clinical note with its annotations."""
    document_id: str
    note_type: str
    text: str
    entities: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "note_type": self.note_type,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "sections": self.sections,
            }
        
    # ── Template Fragments ────────────────────────────────────

class TemplateLibrary:
        """Clinical note section templates with entity placeholders."""

        # ── HPI (History of Present Illness) ──────────────────
        HPI_TEMPLATES = [
            "The patient is a {age}-year-old {gender} who presents with {DIAGNOSIS}. "
            "Symptoms began on {DATE}. The patient reports worsening symptoms over the "
            "past {duration}. Current medications include {MEDICATION} {DOSAGE}.",

            "This is a {age}-year-old {gender} with a history of {DIAGNOSIS} who was "
            "admitted on {DATE} with complaints of {symptom}. The patient was previously "
            "on {MEDICATION} {DOSAGE} but reports poor compliance.",

            "{age}-year-old {gender} with known {DIAGNOSIS} presenting with acute "
            "exacerbation. The patient was last seen in clinic on {DATE} and was taking "
            "{MEDICATION} {DOSAGE}. A {PROCEDURE} was performed at that time.",

            "The patient is a {age}-year-old {gender} admitted on {DATE} for evaluation "
            "of {symptom}. Past medical history is significant for {DIAGNOSIS}. "
            "Home medications include {MEDICATION} {DOSAGE} and {MEDICATION2} {DOSAGE2}.",
        ]

        # ── Past Medical History ──────────────────────────────
        PMH_TEMPLATES = [
            "1. {DIAGNOSIS}\n2. {DIAGNOSIS2}\n3. {DIAGNOSIS3}",
            "{DIAGNOSIS}, diagnosed {DATE}. {DIAGNOSIS2}. {DIAGNOSIS3}, status post {PROCEDURE}.",
            "Significant for {DIAGNOSIS} and {DIAGNOSIS2}. History of {PROCEDURE} on {DATE}.",
        ]

        # ── Medications section ───────────────────────────────
        MED_TEMPLATES = [
            "1. {MEDICATION} {DOSAGE}\n2. {MEDICATION2} {DOSAGE2}\n3. {MEDICATION3} {DOSAGE3}",
            "{MEDICATION} {DOSAGE}, {MEDICATION2} {DOSAGE2}, {MEDICATION3} {DOSAGE3}.",
            "Home medications:\n- {MEDICATION} {DOSAGE}\n- {MEDICATION2} {DOSAGE2}\n"
            "- {MEDICATION3} {DOSAGE3}\n- {MEDICATION4} {DOSAGE4}",
        ]

        # ── Assessment & Plan ─────────────────────────────────
        AP_TEMPLATES = [
            "1. {DIAGNOSIS} - continue {MEDICATION} {DOSAGE}. Schedule {PROCEDURE}.\n"
            "2. {DIAGNOSIS2} - stable, continue current management.\n"
            "3. {DIAGNOSIS3} - start {MEDICATION2} {DOSAGE2}.",

            "Assessment: {age}-year-old {gender} with {DIAGNOSIS} and {DIAGNOSIS2}.\n"
            "Plan:\n- Continue {MEDICATION} {DOSAGE}\n- Order {PROCEDURE}\n"
            "- Follow up in {follow_up} weeks",

            "Problem list:\n"
            "# {DIAGNOSIS}: Initiate {MEDICATION} {DOSAGE}. {PROCEDURE} pending.\n"
            "# {DIAGNOSIS2}: Continue {MEDICATION2} {DOSAGE2}. Stable.\n"
            "# {DIAGNOSIS3}: Monitor. Repeat labs on {DATE}.",
        ]

        # ── Hospital Course ───────────────────────────────────
        COURSE_TEMPLATES = [
            "The patient was admitted on {DATE} with {DIAGNOSIS}. On hospital day 1, "
            "{PROCEDURE} was performed which showed {finding}. The patient was started "
            "on {MEDICATION} {DOSAGE}. By hospital day {hd}, symptoms improved. "
            "{MEDICATION2} {DOSAGE2} was added on {DATE2} for {DIAGNOSIS2}.",

            "Hospital course was notable for management of {DIAGNOSIS}. The patient "
            "underwent {PROCEDURE} on {DATE}. Post-procedure, the patient was started "
            "on {MEDICATION} {DOSAGE}. The patient's {DIAGNOSIS2} remained stable on "
            "{MEDICATION2} {DOSAGE2}.",
        ]

        # ── Operative Note ────────────────────────────────────
        OPERATIVE_TEMPLATES = [
            "Preoperative Diagnosis: {DIAGNOSIS}\n"
            "Postoperative Diagnosis: {DIAGNOSIS}\n"
            "Procedure: {PROCEDURE}\n"
            "Date of Surgery: {DATE}\n\n"
            "The patient was brought to the operating room and placed under general "
            "anesthesia. {procedure_detail}. The patient tolerated the procedure well "
            "and was transferred to the recovery room in stable condition.",
        ]

        # ── Discharge Medications ─────────────────────────────
        DC_MED_TEMPLATES = [
            "Discharge Medications:\n"
            "1. {MEDICATION} {DOSAGE}\n"
            "2. {MEDICATION2} {DOSAGE2}\n"
            "3. {MEDICATION3} {DOSAGE3}\n"
            "4. {MEDICATION4} {DOSAGE4}",
        ]

        # ── Negation (for creating negated examples) ──────────
        NEGATION_TEMPLATES = [
            "The patient denies any history of {DIAGNOSIS}.",
            "No evidence of {DIAGNOSIS} on imaging.",
            "Patient is not currently on {MEDICATION}.",
            "{DIAGNOSIS} has been ruled out.",
            "Without signs of {DIAGNOSIS}.",
            "There is no {DIAGNOSIS}.",
        ]    
    # ── Helper Functions ──────────────────────────────────────

def _pick(pool: list, exclude: set = None) -> str:
        """Pick a random item from pool, avoiding already-used items."""
        available = [x for x in pool if x not in (exclude or set())]
        if not available:
            available = pool
        return random.choice(available)


def _random_age() -> int:
        return random.randint(25, 92)


def _random_gender() -> str:
        return random.choice(["male", "female"])


def _random_duration() -> str:
        n = random.randint(1, 14)
        unit = random.choice(["days", "weeks", "months"])
        return f"{n} {unit}"


def _random_symptom() -> str:
        return random.choice([
            "chest pain", "shortness of breath", "abdominal pain",
            "fever and chills", "dizziness", "lower extremity swelling",
            "nausea and vomiting", "fatigue", "confusion",
            "productive cough", "headache", "back pain",
            "palpitations", "syncope", "hemoptysis",
        ])


def _random_finding() -> str:
        return random.choice([
            "moderate stenosis", "no acute abnormalities",
            "bilateral infiltrates", "a 2cm mass in the right lower lobe",
            "elevated troponin levels", "mild cardiomegaly",
            "normal ejection fraction", "scattered opacities",
        ])


def _random_procedure_detail() -> str:
        return random.choice([
            "A midline incision was made and the tissue was carefully dissected",
            "The scope was advanced without difficulty and the area was visualized",
            "Hemostasis was achieved and the wound was closed in layers",
            "The graft was anastomosed in standard end-to-side fashion",
        ])
        
# ── Entity Tracking Builder ───────────────────────────────

class NoteBuilder:
    """
    Builds a clinical note string while tracking entity positions.
    As text is added, it records character offsets for every entity
    so the BIO converter knows exactly which tokens to tag.
    """

    def __init__(self):
        self.text = ""
        self.entities: list[EntityAnnotation] = []
        self._used_diagnoses = set()
        self._used_medications = set()
        self._used_procedures = set()
        self._used_dosages = set()
        self._used_dates = set()

    @property
    def cursor(self) -> int:
        """Current position in the text — where the next character will go."""
        return len(self.text)

    def add_raw(self, text: str):
        """Add plain text with no entity tracking."""
        self.text += text

    def add_entity(self, label: str) -> str:
        """Pick a random entity, append it to text, and record its position."""
        pool_map = {
            "DIAGNOSIS": (DIAGNOSES, self._used_diagnoses),
            "MEDICATION": (MEDICATIONS, self._used_medications),
            "PROCEDURE": (PROCEDURES, self._used_procedures),
            "DOSAGE": (DOSAGES, self._used_dosages),
            "DATE": (DATES, self._used_dates),
        }
        pool, used = pool_map[label]
        entity_text = _pick(pool, used)
        used.add(entity_text)

        start = self.cursor
        self.text += entity_text
        end = self.cursor

        self.entities.append(EntityAnnotation(
            text=entity_text,
            label=label,
            start=start,
            end=end,
        ))
        return entity_text

    def fill_template(self, template: str) -> str:
        """
        Process a template string. Replaces:
        - {DIAGNOSIS}, {MEDICATION}, etc. → tracked entities with positions
        - {age}, {gender}, {symptom}, etc. → random filler text (not tracked)
        - {MEDICATION2}, {DOSAGE3}, etc. → different item from same pool
        """
        result_parts = []
        i = 0
        while i < len(template):
            if template[i] == "{":
                j = template.index("}", i)
                placeholder = template[i + 1:j]

                # Strip trailing digits: MEDICATION2 → MEDICATION
                base_label = placeholder.rstrip("0123456789")

                if base_label in ("DIAGNOSIS", "MEDICATION", "PROCEDURE", "DOSAGE", "DATE"):
                    # Flush accumulated raw text first
                    raw_chunk = "".join(result_parts)
                    result_parts.clear()
                    self.add_raw(raw_chunk)
                    # Add tracked entity
                    self.add_entity(base_label)
                else:
                    # Non-entity placeholder → random filler
                    value = self._resolve_placeholder(base_label)
                    result_parts.append(value)

                i = j + 1
            else:
                result_parts.append(template[i])
                i += 1

        # Flush any remaining raw text
        if result_parts:
            self.add_raw("".join(result_parts))

    def _resolve_placeholder(self, name: str) -> str:
        """Replace non-entity placeholders with random values."""
        resolvers = {
            "age": lambda: str(_random_age()),
            "gender": _random_gender,
            "duration": _random_duration,
            "symptom": _random_symptom,
            "finding": _random_finding,
            "procedure_detail": _random_procedure_detail,
            "follow_up": lambda: str(random.randint(1, 8)),
            "hd": lambda: str(random.randint(2, 7)),
        }
        resolver = resolvers.get(name)
        if resolver:
            return resolver()
        return f"[{name}]"
    
    # ── Note Type Generators ──────────────────────────────────

def _generate_discharge_summary(builder: NoteBuilder):
    """Generate a Discharge Summary note."""
    builder.add_raw("DISCHARGE SUMMARY\n\n")

    builder.add_raw(f"Patient: {random.choice(PATIENT_NAMES)}\n")
    builder.add_raw(f"MRN: {random.choice(MRNS)}\n")
    builder.add_raw(f"Attending: {random.choice(DOCTOR_NAMES)}\n")
    builder.add_raw("Admission Date: ")
    builder.add_entity("DATE")
    builder.add_raw("\nDischarge Date: ")
    builder.add_entity("DATE")
    builder.add_raw("\n\n")

    builder.add_raw("HISTORY OF PRESENT ILLNESS:\n")
    builder.fill_template(random.choice(TemplateLibrary.HPI_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("PAST MEDICAL HISTORY:\n")
    builder.fill_template(random.choice(TemplateLibrary.PMH_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("MEDICATIONS ON ADMISSION:\n")
    builder.fill_template(random.choice(TemplateLibrary.MED_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("HOSPITAL COURSE:\n")
    builder.fill_template(random.choice(TemplateLibrary.COURSE_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("ASSESSMENT AND PLAN:\n")
    builder.fill_template(random.choice(TemplateLibrary.AP_TEMPLATES))
    builder.add_raw("\n\n")

    builder.fill_template(random.choice(TemplateLibrary.DC_MED_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("PERTINENT NEGATIVES:\n")
    for _ in range(random.randint(1, 2)):
        builder.fill_template(random.choice(TemplateLibrary.NEGATION_TEMPLATES))
        builder.add_raw("\n")


def _generate_progress_note(builder: NoteBuilder):
    """Generate a Progress Note."""
    builder.add_raw("PROGRESS NOTE\n\n")

    builder.add_raw("Date: ")
    builder.add_entity("DATE")
    builder.add_raw(f"\nProvider: {random.choice(DOCTOR_NAMES)}\n\n")

    builder.add_raw("SUBJECTIVE:\n")
    builder.fill_template(
        "Patient reports {symptom}. Currently taking {MEDICATION} {DOSAGE}. "
        "Last dose taken this morning."
    )
    builder.add_raw("\n\n")

    builder.add_raw("OBJECTIVE:\n")
    builder.add_raw(f"Vitals: T {random.uniform(97.0, 101.5):.1f}F, "
                    f"HR {random.randint(60, 110)}, "
                    f"BP {random.randint(100, 180)}/{random.randint(60, 100)}, "
                    f"RR {random.randint(12, 24)}, "
                    f"SpO2 {random.randint(92, 100)}%\n")
    builder.add_raw("General: Alert, oriented, in no acute distress.\n\n")

    builder.add_raw("ASSESSMENT AND PLAN:\n")
    builder.fill_template(random.choice(TemplateLibrary.AP_TEMPLATES))
    builder.add_raw("\n")


def _generate_history_physical(builder: NoteBuilder):
    """Generate a History and Physical (H&P) note."""
    builder.add_raw("HISTORY AND PHYSICAL\n\n")

    builder.add_raw(f"Patient: {random.choice(PATIENT_NAMES)}\n")
    builder.add_raw("Date of Evaluation: ")
    builder.add_entity("DATE")
    builder.add_raw(f"\nReferring Physician: {random.choice(DOCTOR_NAMES)}\n\n")

    builder.add_raw("CHIEF COMPLAINT:\n")
    builder.fill_template("{symptom} for {duration}.")
    builder.add_raw("\n\n")

    builder.add_raw("HISTORY OF PRESENT ILLNESS:\n")
    builder.fill_template(random.choice(TemplateLibrary.HPI_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("PAST MEDICAL HISTORY:\n")
    builder.fill_template(random.choice(TemplateLibrary.PMH_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("CURRENT MEDICATIONS:\n")
    builder.fill_template(random.choice(TemplateLibrary.MED_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("ALLERGIES:\n")
    builder.add_raw(random.choice([
        "No known drug allergies.",
        "Penicillin (rash).",
        "Sulfa drugs (anaphylaxis).",
        "NKDA.",
    ]))
    builder.add_raw("\n\n")

    builder.add_raw("REVIEW OF SYSTEMS:\n")
    builder.fill_template("Positive for {symptom}. ")
    builder.fill_template(random.choice(TemplateLibrary.NEGATION_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("PHYSICAL EXAMINATION:\n")
    builder.add_raw(f"Vitals: T {random.uniform(97.0, 101.5):.1f}F, "
                    f"HR {random.randint(60, 110)}, "
                    f"BP {random.randint(100, 180)}/{random.randint(60, 100)}\n")
    builder.add_raw("General: Well-appearing, in no acute distress.\n")
    builder.add_raw("HEENT: Normocephalic, atraumatic. PERRL.\n")
    builder.add_raw("Cardiovascular: Regular rate and rhythm, no murmurs.\n")
    builder.add_raw("Lungs: Clear to auscultation bilaterally.\n\n")

    builder.add_raw("ASSESSMENT AND PLAN:\n")
    builder.fill_template(random.choice(TemplateLibrary.AP_TEMPLATES))
    builder.add_raw("\n")

def _generate_consult_note(builder: NoteBuilder):
    """Generate a Consult Note."""
    specialty = random.choice([
        "Cardiology", "Pulmonology", "Nephrology",
        "Gastroenterology", "Endocrinology", "Infectious Disease",
    ])

    builder.add_raw(f"{specialty.upper()} CONSULT NOTE\n\n")
    builder.add_raw(f"Consulting Physician: {random.choice(DOCTOR_NAMES)}\n")
    builder.add_raw("Date: ")
    builder.add_entity("DATE")
    builder.add_raw("\n\n")

    builder.add_raw("REASON FOR CONSULTATION:\n")
    builder.fill_template(
        "Consultation requested for management of {DIAGNOSIS} "
        "in the setting of {DIAGNOSIS2}."
    )
    builder.add_raw("\n\n")

    builder.add_raw("HISTORY OF PRESENT ILLNESS:\n")
    builder.fill_template(random.choice(TemplateLibrary.HPI_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("RELEVANT HISTORY:\n")
    builder.fill_template(random.choice(TemplateLibrary.PMH_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("CURRENT MEDICATIONS:\n")
    builder.fill_template(random.choice(TemplateLibrary.MED_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("RECOMMENDATIONS:\n")
    builder.fill_template(
        "1. Recommend {PROCEDURE} to further evaluate.\n"
        "2. Start {MEDICATION} {DOSAGE}.\n"
        "3. Follow up with {specialty} in {follow_up} weeks.\n"
    )
    builder.add_raw("4. Will continue to follow.\n\n")

    builder.fill_template(random.choice(TemplateLibrary.NEGATION_TEMPLATES))
    builder.add_raw("\n")


def _generate_operative_note(builder: NoteBuilder):
    """Generate an Operative Note."""
    builder.add_raw("OPERATIVE NOTE\n\n")

    builder.add_raw(f"Surgeon: {random.choice(DOCTOR_NAMES)}\n")
    builder.add_raw(f"Patient: {random.choice(PATIENT_NAMES)}\n")
    builder.add_raw("Date of Surgery: ")
    builder.add_entity("DATE")
    builder.add_raw("\n\n")

    builder.fill_template(random.choice(TemplateLibrary.OPERATIVE_TEMPLATES))
    builder.add_raw("\n\n")

    builder.add_raw("ESTIMATED BLOOD LOSS: ")
    builder.add_raw(f"{random.choice(['50', '100', '150', '200', '250', '300'])} mL\n\n")

    builder.add_raw("SPECIMENS: ")
    builder.add_raw(random.choice([
        "Sent to pathology.", "None.", "Tissue specimen sent to pathology.",
    ]))
    builder.add_raw("\n\n")

    builder.add_raw("POSTOPERATIVE PLAN:\n")
    builder.fill_template(
        "1. {MEDICATION} {DOSAGE} for pain management.\n"
        "2. {MEDICATION2} {DOSAGE2} for DVT prophylaxis.\n"
        "3. Follow up on {DATE}.\n"
        "4. Monitor for signs of {DIAGNOSIS}.\n"
    )


# ── Generator Registry ────────────────────────────────────
# Maps note type names to their generator functions

NOTE_GENERATORS = {
    "Discharge Summary": _generate_discharge_summary,
    "Progress Note": _generate_progress_note,
    "History and Physical": _generate_history_physical,
    "Consult Note": _generate_consult_note,
    "Operative Note": _generate_operative_note,
}


# ── Public API ────────────────────────────────────────────

def generate_note(note_type: Optional[str] = None) -> SyntheticNote:
    """
    Generate a single synthetic clinical note with entity annotations.

    Args:
        note_type: Type of note to generate. If None, picks randomly.

    Returns:
        SyntheticNote with text, entity annotations, and metadata.
    """
    if note_type is None:
        note_type = random.choice(list(NOTE_GENERATORS.keys()))

    generator = NOTE_GENERATORS[note_type]
    builder = NoteBuilder()
    generator(builder)

    return SyntheticNote(
        document_id=f"synth_{uuid.uuid4().hex[:8]}",
        note_type=note_type,
        text=builder.text,
        entities=builder.entities,
    )


def generate_dataset(
    num_notes: int = 500,
    seed: int = 42,
    note_types: Optional[list[str]] = None,
) -> list[SyntheticNote]:
    """
    Generate a full synthetic dataset.

    Args:
        num_notes: Number of notes to generate.
        seed: Random seed for reproducibility.
        note_types: List of note types. If None, uses all types evenly.

    Returns:
        List of SyntheticNote objects.
    """
    random.seed(seed)
    if note_types is None:
        note_types = list(NOTE_GENERATORS.keys())

    notes = []
    for i in range(num_notes):
        nt = note_types[i % len(note_types)]
        note = generate_note(nt)
        notes.append(note)

    return notes


def save_dataset(notes: list[SyntheticNote], output_dir: str | Path):
    """
    Save generated notes to JSONL format.

    Writes two files:
    - notes.jsonl: Full note text + metadata
    - annotations.jsonl: Entity annotations aligned to notes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    notes_path = output_dir / "notes.jsonl"
    annotations_path = output_dir / "annotations.jsonl"

    with open(notes_path, "w") as nf, open(annotations_path, "w") as af:
        for note in notes:
            nf.write(json.dumps({
                "document_id": note.document_id,
                "note_type": note.note_type,
                "text": note.text,
            }) + "\n")

            af.write(json.dumps({
                "document_id": note.document_id,
                "entities": [e.to_dict() for e in note.entities],
            }) + "\n")

    print(f"Saved {len(notes)} notes to {notes_path}")
    print(f"Saved annotations to {annotations_path}")

    # Print entity distribution stats
    entity_counts = {}
    for note in notes:
        for e in note.entities:
            entity_counts[e.label] = entity_counts.get(e.label, 0) + 1

    print(f"\nEntity distribution:")
    for label, count in sorted(entity_counts.items()):
        print(f"  {label}: {count}")    