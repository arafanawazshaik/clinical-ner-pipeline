"""
PyTorch Dataset for BIO-tagged Clinical NER.

Loads BIO-tagged sequences from JSONL, tokenizes with BioBERT tokenizer,
and aligns BIO labels to subword tokens. First subword gets the label,
remaining subwords get -100 (ignored by loss function).
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MODEL_NAME, MAX_SEQ_LENGTH
from config.label_schema import TAG_TO_ID


class NERDataset(Dataset):
    """
    PyTorch Dataset for token-level NER.

    Each sample is a clinical note tokenized into subword tokens
    with aligned BIO tag IDs.
    """

    def __init__(self, data_path: str | Path, tokenizer: AutoTokenizer = None, max_length: int = MAX_SEQ_LENGTH):
        """
        Args:
            data_path: Path to BIO-tagged JSONL file.
            tokenizer: HuggingFace tokenizer. If None, loads BioBERT.
            max_length: Maximum sequence length for tokenizer.
        """
        self.data_path = Path(data_path)
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        self.samples = self._load_data()

    def _load_data(self) -> list[dict]:
        """Load BIO-tagged records from JSONL."""
        samples = []
        with open(self.data_path) as f:
            for line in f:
                record = json.loads(line)
                samples.append(record)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenize a sample and align labels to subword tokens.

        Returns dict with:
            - input_ids: token IDs for BioBERT
            - attention_mask: 1 for real tokens, 0 for padding
            - labels: BIO tag IDs aligned to subwords (-100 for non-first subwords)
            - document_id: for tracking
        """
        sample = self.samples[idx]
        tokens = sample["tokens"]
        tags = sample["tags"]

        # Tokenize each word and align labels
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels to subword tokens
        labels = self._align_labels(encoding, tags)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "document_id": sample["document_id"],
        }

    def _align_labels(self, encoding, tags: list[str]) -> list[int]:
        """
        Align BIO tags to subword tokens.

        Rules:
            - First subword of a word → gets the word's BIO tag
            - Subsequent subwords → get -100 (ignored in loss)
            - Special tokens ([CLS], [SEP], [PAD]) → get -100
        """
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word → assign the tag
                if word_id < len(tags):
                    tag = tags[word_id]
                    aligned_labels.append(TAG_TO_ID.get(tag, TAG_TO_ID["O"]))
                else:
                    aligned_labels.append(-100)
            else:
                # Continuation subword → ignore
                aligned_labels.append(-100)

            previous_word_id = word_id

        return aligned_labels