"""
Evaluate NER Model + Error Analysis.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --split test
"""

import argparse
import json
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SPLITS_DIR, MODEL_DIR, MAX_SEQ_LENGTH
from config.label_schema import ID_TO_TAG
from src.training.dataset import NERDataset
from src.evaluation.metrics import compute_ner_metrics, print_metrics
from src.evaluation.error_analysis import NERErrorAnalyzer, print_error_report
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER model")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None, help="Save report as JSON")
    args = parser.parse_args()

    data_path = SPLITS_DIR / f"{args.split}.jsonl"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # Load data
    print(f"Loading {args.split} data...")
    dataset = NERDataset(data_path, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run predictions
    all_preds = []
    all_labels = []
    all_texts = []

    print("Running predictions...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Load original texts for error analysis
    with open(data_path) as f:
        for line in f:
            record = json.loads(line)
            all_texts.append(" ".join(record["tokens"]))

    # Compute metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    metrics = compute_ner_metrics(all_labels, all_preds)
    print_metrics(metrics)

    # Error analysis
    analyzer = NERErrorAnalyzer()
    report = analyzer.analyze(all_texts, all_labels, all_preds)
    print_error_report(report)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        full_report = {
            "split": args.split,
            "metrics": {
                "precision": metrics["overall_precision"],
                "recall": metrics["overall_recall"],
                "f1": metrics["overall_f1"],
            },
            "error_analysis": {
                "total_errors": report["total_errors"],
                "error_distribution": report["error_distribution"],
                "entity_metrics": report["entity_metrics"],
                "top_confused_pairs": report["top_confused_pairs"],
                "top_false_positives": report["top_false_positives"],
                "top_false_negatives": report["top_false_negatives"],
            },
        }
        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()