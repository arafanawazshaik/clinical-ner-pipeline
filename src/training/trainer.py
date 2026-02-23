"""
NER Training Loop.

Trains BioBERT for token classification with:
- Weighted cross-entropy loss (handles class imbalance)
- AdamW optimizer with linear warmup
- Early stopping on validation F1
- Per-epoch evaluation and metric logging
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, EARLY_STOPPING_PATIENCE, MODEL_DIR,
)
from src.training.dataset import NERDataset
from src.training.model import load_ner_model, save_model
from src.training.class_balancer import compute_class_weights
from src.evaluation.metrics import compute_ner_metrics


class NERTrainer:
    """
    Trainer for BioBERT NER model.

    Handles the full training cycle: data loading, training loop,
    evaluation, early stopping, and checkpoint saving.
    """

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        model_name: str = None,
        output_dir: str | Path = MODEL_DIR,
        num_epochs: int = NUM_EPOCHS,
        train_batch_size: int = TRAIN_BATCH_SIZE,
        eval_batch_size: int = EVAL_BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        patience: int = EARLY_STOPPING_PATIENCE,
        use_class_weights: bool = True,
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.use_class_weights = use_class_weights

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model, self.tokenizer = load_ner_model(model_name=model_name)
        self.model.to(self.device)

        # Load datasets
        print("Loading training data...")
        self.train_dataset = NERDataset(self.train_path, tokenizer=self.tokenizer)
        print("Loading validation data...")
        self.val_dataset = NERDataset(self.val_path, tokenizer=self.tokenizer)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )

        # Class weights for loss function
        if self.use_class_weights:
            weights = compute_class_weights(self.train_path)
            self.class_weights = weights.to(self.device)
            print("Using weighted cross-entropy loss")
        else:
            self.class_weights = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
        }

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dict with training history and best metrics.
        """
        print(f"\n{'='*60}")
        print(f"TRAINING START")
        print(f"{'='*60}")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples:   {len(self.val_dataset)}")
        print(f"  Epochs:        {self.num_epochs}")
        print(f"  Batch size:    {self.train_batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"{'='*60}\n")

        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            train_loss = self._train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Evaluate
            val_loss, val_metrics = self._evaluate()
            val_f1 = val_metrics["overall_f1"]
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)

            elapsed = time.time() - epoch_start

            print(f"Epoch {epoch}/{self.num_epochs} "
                  f"| Train Loss: {train_loss:.4f} "
                  f"| Val Loss: {val_loss:.4f} "
                  f"| Val F1: {val_f1:.4f} "
                  f"| Time: {elapsed:.1f}s")

            # Early stopping check
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                save_model(self.model, self.tokenizer, self.output_dir)
                print(f"  → New best F1! Model saved to {self.output_dir}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
        return {
            "best_f1": best_f1,
            "history": self.history,
            "epochs_trained": epoch,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Use weighted loss if class weights are set
            if self.class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    ignore_index=-100,
                )
                logits = outputs.logits
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
            else:
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _evaluate(self) -> tuple[float, dict]:
        """Evaluate on validation set. Returns (loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                num_batches += 1

                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / num_batches
        metrics = compute_ner_metrics(all_labels, all_preds)

        return avg_loss, metrics