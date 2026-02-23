"""
BIO Tag Converter.

Converts character-level entity annotations into token-level BIO tags.
Supports whitespace tokenization for rule-based use and will later
support subword tokenization alignment for BioBERT training.
"""

import random
import json
import re
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.label_schema import TAG_TO_ID, ID_TO_TAG, BIO_TAGS

@dataclass
class BIOToken:
    """A single token with its BIO tag."""
    token: str
    tag: str
    tag_id: int
    char_start: int
    char_end: int


def _whitespace_tokenize(text: str) -> list[tuple[str, int, int]]:
    """
    Tokenize text by whitespace, returning (token, start, end) tuples.
    Preserves character offsets for alignment with entity annotations.
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens
def annotate_bio_tags(text: str, entities: list[dict]) -> list[BIOToken]:
    """
    Convert entity annotations to BIO-tagged token sequence.

    For each token, assigns:
    - B-{TYPE} if the token starts an entity
    - I-{TYPE} if the token continues an entity
    - O if the token is outside any entity

    Args:
        text: The full clinical note text.
        entities: List of entity dicts with keys: text, label, start, end.

    Returns:
        List of BIOToken objects with tags assigned.
    """
    tokens = _whitespace_tokenize(text)
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    bio_tokens = []
    for token_text, tok_start, tok_end in tokens:
        tag = "O"

        for entity in sorted_entities:
            e_start = entity["start"]
            e_end = entity["end"]
            e_label = entity["label"]

            # Token fully inside entity span
            if tok_start >= e_start and tok_end <= e_end:
                if tok_start == e_start:
                    tag = f"B-{e_label}"
                else:
                    tag = f"I-{e_label}"
                break

            # Partial overlap — assign if majority overlaps
            elif tok_start < e_end and tok_end > e_start:
                overlap = min(tok_end, e_end) - max(tok_start, e_start)
                tok_len = tok_end - tok_start
                if overlap > tok_len * 0.5:
                    if tok_start <= e_start:
                        tag = f"B-{e_label}"
                    else:
                        tag = f"I-{e_label}"
                    break

        tag_id = TAG_TO_ID.get(tag, TAG_TO_ID["O"])
        bio_tokens.append(BIOToken(
            token=token_text,
            tag=tag,
            tag_id=tag_id,
            char_start=tok_start,
            char_end=tok_end,
        ))

    return bio_tokens

def convert_note_to_bio(note_dict: dict, annotation_dict: dict) -> dict:
    """
    Convert a single note + annotations pair to BIO format.
    """
    assert note_dict["document_id"] == annotation_dict["document_id"], \
        f"ID mismatch: {note_dict['document_id']} vs {annotation_dict['document_id']}"

    bio_tokens = annotate_bio_tags(
        text=note_dict["text"],
        entities=annotation_dict["entities"],
    )

    return {
        "document_id": note_dict["document_id"],
        "note_type": note_dict["note_type"],
        "tokens": [t.token for t in bio_tokens],
        "tags": [t.tag for t in bio_tokens],
        "tag_ids": [t.tag_id for t in bio_tokens],
        "char_offsets": [(t.char_start, t.char_end) for t in bio_tokens],
    }


def convert_dataset_to_bio(notes_path: str | Path, annotations_path: str | Path, output_path: str | Path):
    """
    Convert a full JSONL dataset (notes + annotations) to BIO-tagged JSONL.
    """
    notes_path = Path(notes_path)
    annotations_path = Path(annotations_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    notes = {}
    with open(notes_path) as f:
        for line in f:
            d = json.loads(line)
            notes[d["document_id"]] = d

    annotations = {}
    with open(annotations_path) as f:
        for line in f:
            d = json.loads(line)
            annotations[d["document_id"]] = d

    count = 0
    entity_tag_counts = {}
    with open(output_path, "w") as out:
        for doc_id, note in notes.items():
            ann = annotations.get(doc_id)
            if ann is None:
                print(f"Warning: no annotations for {doc_id}, skipping")
                continue

            bio_record = convert_note_to_bio(note, ann)
            out.write(json.dumps(bio_record) + "\n")
            count += 1

            for tag in bio_record["tags"]:
                entity_tag_counts[tag] = entity_tag_counts.get(tag, 0) + 1

    print(f"Converted {count} notes to BIO format → {output_path}")
    print(f"\nTag distribution:")
    for tag in BIO_TAGS:
        c = entity_tag_counts.get(tag, 0)
        print(f"  {tag}: {c}")


def create_train_val_test_splits(
    bio_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split BIO-tagged JSONL into train/val/test sets.
    """
    random.seed(seed)

    bio_path = Path(bio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(bio_path) as f:
        for line in f:
            records.append(json.loads(line))

    random.shuffle(records)

    n = len(records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    for split_name, split_records in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for rec in split_records:
                f.write(json.dumps(rec) + "\n")
        print(f"  {split_name}: {len(split_records)} notes → {path}")

