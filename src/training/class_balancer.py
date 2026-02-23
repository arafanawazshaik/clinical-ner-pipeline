"""
Class Balancer for NER Training.

Computes class weights inversely proportional to tag frequency.
Handles the heavy class imbalance in NER where O tags dominate
(~65% of tokens) and rare entity tags need boosting.
"""

import json
import torch
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.label_schema import TAG_TO_ID, BIO_TAGS


def compute_class_weights(data_path: str | Path, smoothing: float = 0.1) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy.

    Formula: weight_i = total_tags / (num_classes * count_i + smoothing)

    The smoothing factor prevents division by zero for unseen tags
    and reduces extreme weights for very rare tags.

    Args:
        data_path: Path to BIO-tagged JSONL (train split).
        smoothing: Smoothing factor added to counts.

    Returns:
        Tensor of shape (num_classes,) with weight per BIO tag.
    """
    tag_counts = count_tags(data_path)
    num_classes = len(TAG_TO_ID)
    total = sum(tag_counts.values())

    weights = []
    for tag in BIO_TAGS:
        tag_id = TAG_TO_ID[tag]
        count = tag_counts.get(tag, 0)
        weight = total / (num_classes * count + smoothing)
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Normalize so average weight = 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()

    return weights_tensor


def count_tags(data_path: str | Path) -> dict[str, int]:
    """
    Count BIO tag occurrences in a JSONL dataset.

    Args:
        data_path: Path to BIO-tagged JSONL file.

    Returns:
        Dict mapping tag strings to their counts.
    """
    counts = Counter()
    with open(data_path) as f:
        for line in f:
            record = json.loads(line)
            counts.update(record["tags"])
    return dict(counts)


def print_class_weights(data_path: str | Path):
    """Print tag counts and computed weights for inspection."""
    tag_counts = count_tags(data_path)
    weights = compute_class_weights(data_path)

    print(f"{'Tag':<20} {'Count':>8} {'Weight':>8}")
    print("-" * 38)
    for i, tag in enumerate(BIO_TAGS):
        count = tag_counts.get(tag, 0)
        weight = weights[i].item()
        print(f"{tag:<20} {count:>8} {weight:>8.3f}")
        