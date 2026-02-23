"""
Generate Synthetic Clinical Dataset.

Produces labeled clinical notes in BIO format for NER model training.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --num-notes 1000 --seed 123
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    SYNTHETIC_DIR,
    LABELED_DIR,
    SPLITS_DIR,
    SYNTHETIC_NUM_NOTES,
    SYNTHETIC_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from src.labeling.synthetic_generator import generate_dataset, save_dataset
from src.labeling.bio_converter import convert_dataset_to_bio, create_train_val_test_splits


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical NER dataset")
    parser.add_argument("--num-notes", type=int, default=SYNTHETIC_NUM_NOTES,
                        help=f"Number of notes to generate (default: {SYNTHETIC_NUM_NOTES})")
    parser.add_argument("--seed", type=int, default=SYNTHETIC_SEED,
                        help=f"Random seed (default: {SYNTHETIC_SEED})")
    args = parser.parse_args()

    print("=" * 60)
    print("CLINICAL NER — SYNTHETIC DATA GENERATOR")
    print("=" * 60)

    # Step 1: Generate synthetic notes
    print(f"\n[1/3] Generating {args.num_notes} synthetic clinical notes...")
    notes = generate_dataset(num_notes=args.num_notes, seed=args.seed)

    # Step 2: Save notes + annotations
    print(f"\n[2/3] Saving to {SYNTHETIC_DIR}...")
    save_dataset(notes, SYNTHETIC_DIR)

    # Step 3: Convert to BIO and create splits
    print(f"\n[3/3] Converting to BIO format...")
    notes_path = Path(SYNTHETIC_DIR) / "notes.jsonl"
    annotations_path = Path(SYNTHETIC_DIR) / "annotations.jsonl"
    bio_path = Path(LABELED_DIR) / "bio_tagged.jsonl"

    convert_dataset_to_bio(notes_path, annotations_path, bio_path)

    print(f"\nCreating train/val/test splits...")
    create_train_val_test_splits(
        bio_path=bio_path,
        output_dir=SPLITS_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=args.seed,
    )

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Synthetic notes:  {notes_path}")
    print(f"  Annotations:      {annotations_path}")
    print(f"  BIO-tagged:       {bio_path}")
    print(f"  Train split:      {SPLITS_DIR / 'train.jsonl'}")
    print(f"  Val split:        {SPLITS_DIR / 'val.jsonl'}")
    print(f"  Test split:       {SPLITS_DIR / 'test.jsonl'}")

    total_entities = sum(len(n.entities) for n in notes)
    print(f"\n  Total entities: {total_entities}")
    print(f"  Avg per note: {total_entities / len(notes):.1f}")


if __name__ == "__main__":
    main()