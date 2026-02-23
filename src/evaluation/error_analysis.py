"""
Error Analysis for NER Model.

Analyzes prediction errors by:
- Entity type confusion matrix
- False positive / false negative patterns
- Error distribution by section and note type
- Common misclassification patterns
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.label_schema import ID_TO_TAG, BIO_TAGS


@dataclass
class ErrorCase:
    """A single prediction error."""
    text: str
    true_label: str
    pred_label: str
    context: str
    error_type: str  # "false_positive", "false_negative", "misclassified"


class NERErrorAnalyzer:
    """
    Analyzes NER prediction errors to identify patterns
    and guide model improvement.
    """

    def __init__(self):
        self.errors = []
        self.confusion = defaultdict(Counter)
        self.entity_stats = defaultdict(lambda: {
            "tp": 0, "fp": 0, "fn": 0,
        })

    def analyze(
        self,
        texts: list[str],
        true_labels: list[list[int]],
        pred_labels: list[list[int]],
    ) -> dict:
        """
        Run full error analysis.

        Args:
            texts: Original text strings.
            true_labels: Ground truth tag ID sequences.
            pred_labels: Predicted tag ID sequences.

        Returns:
            Dict with error analysis results.
        """
        self.errors = []
        self.confusion = defaultdict(Counter)
        self.entity_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for text, true_seq, pred_seq in zip(texts, true_labels, pred_labels):
            words = text.split()
            self._analyze_sequence(words, true_seq, pred_seq)

        return self._compile_report()

    def _analyze_sequence(
        self,
        words: list[str],
        true_seq: list[int],
        pred_seq: list[int],
    ):
        """Analyze errors in a single sequence."""
        for i, (true_id, pred_id) in enumerate(zip(true_seq, pred_seq)):
            if true_id == -100:
                continue

            true_tag = ID_TO_TAG.get(true_id, "O")
            pred_tag = ID_TO_TAG.get(pred_id, "O")

            # Build confusion matrix
            self.confusion[true_tag][pred_tag] += 1

            if true_tag == pred_tag:
                # Correct prediction
                if true_tag != "O":
                    entity_type = true_tag.split("-")[-1]
                    self.entity_stats[entity_type]["tp"] += 1
                continue

            # Get word and context
            word = words[i] if i < len(words) else "<UNK>"
            start = max(0, i - 3)
            end = min(len(words), i + 4)
            context = " ".join(words[start:end])

            # Classify error type
            if true_tag == "O" and pred_tag != "O":
                error_type = "false_positive"
                entity_type = pred_tag.split("-")[-1]
                self.entity_stats[entity_type]["fp"] += 1
            elif true_tag != "O" and pred_tag == "O":
                error_type = "false_negative"
                entity_type = true_tag.split("-")[-1]
                self.entity_stats[entity_type]["fn"] += 1
            else:
                error_type = "misclassified"
                true_entity = true_tag.split("-")[-1]
                pred_entity = pred_tag.split("-")[-1]
                self.entity_stats[true_entity]["fn"] += 1
                self.entity_stats[pred_entity]["fp"] += 1

            self.errors.append(ErrorCase(
                text=word,
                true_label=true_tag,
                pred_label=pred_tag,
                context=context,
                error_type=error_type,
            ))

    def _compile_report(self) -> dict:
        """Compile all analysis into a report dict."""
        # Error type distribution
        error_types = Counter(e.error_type for e in self.errors)

        # Most confused pairs
        confused_pairs = []
        for true_tag, preds in self.confusion.items():
            for pred_tag, count in preds.items():
                if true_tag != pred_tag and true_tag != "O":
                    confused_pairs.append({
                        "true": true_tag,
                        "predicted": pred_tag,
                        "count": count,
                    })
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)

        # Per-entity metrics
        entity_metrics = {}
        for entity_type, stats in self.entity_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            entity_metrics[entity_type] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }

        # Most common false positives
        fp_words = Counter(
            e.text for e in self.errors if e.error_type == "false_positive"
        ).most_common(10)

        # Most common false negatives
        fn_words = Counter(
            e.text for e in self.errors if e.error_type == "false_negative"
        ).most_common(10)

        return {
            "total_errors": len(self.errors),
            "error_distribution": dict(error_types),
            "entity_metrics": entity_metrics,
            "top_confused_pairs": confused_pairs[:10],
            "top_false_positives": fp_words,
            "top_false_negatives": fn_words,
            "sample_errors": [
                {
                    "text": e.text,
                    "true": e.true_label,
                    "pred": e.pred_label,
                    "context": e.context,
                    "type": e.error_type,
                }
                for e in self.errors[:20]
            ],
        }


def print_error_report(report: dict):
    """Pretty-print the error analysis report."""
    print(f"\n{'='*60}")
    print("NER ERROR ANALYSIS REPORT")
    print(f"{'='*60}")

    print(f"\nTotal Errors: {report['total_errors']}")
    print(f"\nError Distribution:")
    for etype, count in report["error_distribution"].items():
        print(f"  {etype}: {count}")

    print(f"\nPer-Entity Metrics:")
    print(f"  {'Entity':<15} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'-'*59}")
    for entity, metrics in report["entity_metrics"].items():
        print(f"  {entity:<15} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} "
              f"{metrics['f1']:>8.4f} {metrics['tp']:>6} {metrics['fp']:>6} {metrics['fn']:>6}")

    print(f"\nTop Confused Pairs:")
    for pair in report["top_confused_pairs"][:5]:
        print(f"  {pair['true']} → {pair['predicted']}: {pair['count']}")

    print(f"\nTop False Positives (words wrongly tagged as entities):")
    for word, count in report["top_false_positives"][:5]:
        print(f"  '{word}': {count}")

    print(f"\nTop False Negatives (entities missed by model):")
    for word, count in report["top_false_negatives"][:5]:
        print(f"  '{word}': {count}")

    if report["sample_errors"]:
        print(f"\nSample Errors (first 10):")
        for err in report["sample_errors"][:10]:
            print(f"  [{err['type']}] '{err['text']}' "
                  f"true={err['true']} pred={err['pred']}")
            print(f"    context: ...{err['context']}...")