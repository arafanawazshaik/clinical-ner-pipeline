"""
NER Evaluation Metrics.

Computes entity-level precision, recall, and F1 using seqeval.
Converts tag IDs back to tag strings and filters out -100 (ignored tokens).
"""

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.label_schema import ID_TO_TAG


def compute_ner_metrics(
    all_labels: list[list[int]],
    all_preds: list[list[int]],
) -> dict:
    """
    Compute entity-level NER metrics.

    Converts tag IDs to strings, filters out -100 tokens,
    then computes P/R/F1 using seqeval (entity-level, not token-level).

    Args:
        all_labels: List of sequences of true tag IDs.
        all_preds: List of sequences of predicted tag IDs.

    Returns:
        Dict with overall_precision, overall_recall, overall_f1,
        and per-entity-type metrics.
    """
    true_tags = []
    pred_tags = []

    for label_seq, pred_seq in zip(all_labels, all_preds):
        true_seq = []
        pred_seq_filtered = []

        for label_id, pred_id in zip(label_seq, pred_seq):
            if label_id == -100:
                continue
            true_seq.append(ID_TO_TAG.get(label_id, "O"))
            pred_seq_filtered.append(ID_TO_TAG.get(pred_id, "O"))

        if true_seq:
            true_tags.append(true_seq)
            pred_tags.append(pred_seq_filtered)

    if not true_tags:
        return {
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "overall_f1": 0.0,
            "report": "No valid sequences to evaluate.",
        }

    return {
        "overall_precision": precision_score(true_tags, pred_tags),
        "overall_recall": recall_score(true_tags, pred_tags),
        "overall_f1": f1_score(true_tags, pred_tags),
        "report": classification_report(true_tags, pred_tags),
    }


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print("NER EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Precision: {metrics['overall_precision']:.4f}")
    print(f"  Recall:    {metrics['overall_recall']:.4f}")
    print(f"  F1 Score:  {metrics['overall_f1']:.4f}")
    print(f"\n{metrics['report']}") 