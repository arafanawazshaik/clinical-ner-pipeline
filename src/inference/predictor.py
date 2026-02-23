"""
Single-Document NER Predictor.

Runs the trained BioBERT model on a preprocessed clinical note
and returns entity predictions with confidence scores.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MODEL_DIR, MAX_SEQ_LENGTH
from config.label_schema import ID_TO_TAG


@dataclass
class PredictedEntity:
    """An entity predicted by the ML model."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    source: str = "model"


class NERPredictor:
    """
    Runs BioBERT NER inference on clinical text.

    Loads a trained checkpoint, tokenizes input, runs inference,
    and converts subword predictions back to entity spans.
    """

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> list[PredictedEntity]:
        """
        Run NER on a single text and return predicted entities.

        Args:
            text: Preprocessed clinical note text.

        Returns:
            List of PredictedEntity objects with spans and confidence.
        """
        # Tokenize
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=False,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1).values

        # Convert subword predictions to word-level
        word_ids = encoding.word_ids(batch_index=0)
        word_preds = self._align_predictions(word_ids, preds[0], confidences[0])

        # Build character-level offsets from words
        word_spans = self._get_word_spans(text, words)

        # Merge BIO tags into entity spans
        entities = self._merge_bio_tags(words, word_preds, word_spans)

        return entities

    def _align_predictions(
        self,
        word_ids: list,
        preds: torch.Tensor,
        confidences: torch.Tensor,
    ) -> list[tuple[str, float]]:
        """
        Align subword predictions back to words.
        Takes only the first subword's prediction for each word.
        """
        word_preds = []
        previous_word_id = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                tag = ID_TO_TAG.get(preds[i].item(), "O")
                conf = confidences[i].item()
                word_preds.append((tag, conf))
            previous_word_id = word_id

        return word_preds

    def _get_word_spans(self, text: str, words: list[str]) -> list[tuple[int, int]]:
        """Get character start/end positions for each whitespace-split word."""
        spans = []
        pos = 0
        for word in words:
            start = text.index(word, pos)
            end = start + len(word)
            spans.append((start, end))
            pos = end
        return spans

    def _merge_bio_tags(
        self,
        words: list[str],
        word_preds: list[tuple[str, float]],
        word_spans: list[tuple[int, int]],
    ) -> list[PredictedEntity]:
        """
        Merge consecutive BIO tags into entity spans.

        B-DIAGNOSIS I-DIAGNOSIS I-DIAGNOSIS → one entity span.
        """
        entities = []
        current_entity = None

        for i, (tag, conf) in enumerate(word_preds):
            if i >= len(words):
                break

            if tag.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entities.append(current_entity)

                label = tag[2:]
                start = word_spans[i][0]
                end = word_spans[i][1]

                current_entity = PredictedEntity(
                    text=words[i],
                    label=label,
                    start=start,
                    end=end,
                    confidence=conf,
                )

            elif tag.startswith("I-") and current_entity:
                label = tag[2:]
                if label == current_entity.label:
                    # Extend current entity
                    current_entity.text += " " + words[i]
                    current_entity.end = word_spans[i][1]
                    current_entity.confidence = min(current_entity.confidence, conf)
                else:
                    # Label mismatch — save current and start new
                    entities.append(current_entity)
                    current_entity = None

            else:
                # O tag — save current entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        return entities