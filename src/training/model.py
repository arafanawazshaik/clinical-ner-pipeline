"""
BioBERT Token Classification Model.

Wraps HuggingFace's AutoModelForTokenClassification with BioBERT
as the base model. Adds convenience methods for loading and saving.
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MODEL_NAME
from config.label_schema import TAG_TO_ID, ID_TO_TAG


def load_ner_model(
    model_name: str = MODEL_NAME,
    num_labels: int = None,
    from_checkpoint: str = None,
):
    if model_name is None:
        model_name = MODEL_NAME
    """
    Load BioBERT for token classification.

    Args:
        model_name: HuggingFace model name (default: BioBERT).
        num_labels: Number of BIO tags. If None, uses label schema.
        from_checkpoint: Path to saved checkpoint. If provided, loads from disk.

    Returns:
        Tuple of (model, tokenizer).
    """
    if num_labels is None:
        num_labels = len(TAG_TO_ID)

    if from_checkpoint:
        model = AutoModelForTokenClassification.from_pretrained(
            from_checkpoint,
            num_labels=num_labels,
            id2label=ID_TO_TAG,
            label2id=TAG_TO_ID,
        )
        tokenizer = AutoTokenizer.from_pretrained(from_checkpoint)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID_TO_TAG,
            label2id=TAG_TO_ID,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def save_model(model, tokenizer, output_dir: str | Path):
    """
    Save model and tokenizer to disk.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        output_dir: Directory to save to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")