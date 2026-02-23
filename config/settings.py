"""
Global configuration for the Clinical NER Pipeline.
Paths, thresholds, model parameters, and runtime settings.
"""

from pathlib import Path


# ── Project Paths ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
SPLITS_DIR = DATA_DIR / "splits"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ── Synthetic Data ────────────────────────────────────────
SYNTHETIC_NUM_NOTES = 500
SYNTHETIC_NOTE_TYPES = [
    "Discharge Summary",
    "Progress Note",
    "History and Physical",
    "Consult Note",
    "Operative Note",
]
SYNTHETIC_SEED = 42

# ── Preprocessing ─────────────────────────────────────────
PHI_PLACEHOLDER = "[PHI]"
MAX_NOTE_LENGTH = 10000

# ── Model Training ────────────────────────────────────────
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3

# ── Data Splits ───────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Confidence Routing ────────────────────────────────────
CONFIDENCE_AUTO_ACCEPT = 0.85
CONFIDENCE_FLAG_THRESHOLD = 0.60

# ── Negation Detection ────────────────────────────────────
NEGATION_WINDOW_SIZE = 5

# ── MLflow ────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "clinical-ner"
MLFLOW_TRACKING_URI = "mlruns"

# ── Batch Inference ───────────────────────────────────────
BATCH_SIZE_INFERENCE = 50
MAX_RETRIES = 3
# ── Confidence Routing ──
AUTO_ACCEPT_THRESHOLD = 0.85
FLAG_THRESHOLD = 0.60
