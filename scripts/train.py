"""
Train BioBERT NER Model.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 5 --batch-size 8
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    SPLITS_DIR, MODEL_DIR, NUM_EPOCHS,
    TRAIN_BATCH_SIZE, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
)
from src.training.trainer import NERTrainer


def main():
    parser = argparse.ArgumentParser(description="Train BioBERT NER model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--output-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    train_path = SPLITS_DIR / "train.jsonl"
    val_path = SPLITS_DIR / "val.jsonl"

    if not train_path.exists():
        print(f"Error: {train_path} not found. Run generate_synthetic_data.py first.")
        sys.exit(1)

    trainer = NERTrainer(
        train_path=train_path,
        val_path=val_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )

    results = trainer.train()

    print(f"\nFinal Results:")
    print(f"  Best F1: {results['best_f1']:.4f}")
    print(f"  Epochs trained: {results['epochs_trained']}")


if __name__ == "__main__":
    main()