"""
Batch Runner.

Processes multiple clinical notes with retry handling,
incremental writes, and structured logging.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.pipeline import preprocess_note
from src.inference.predictor import NERPredictor
from src.rules.rule_extractor import extract_entities_by_rules
from src.inference.postprocessor import postprocess
from src.inference.schema_formatter import format_output
from src.utils.logger import get_logger, log_event, log_error


@dataclass
class BatchStats:
    """Statistics for a batch run."""
    total: int = 0
    success: int = 0
    failed: int = 0
    retried: int = 0
    total_time_ms: float = 0.0


def run_batch(
    notes: list[dict],
    output_path: str | Path,
    max_retries: int = 3,
    log_file: str | Path = None,
) -> BatchStats:
    """
    Process a batch of clinical notes.

    Each note dict should have 'document_id' and 'text' keys.
    Results are written incrementally as each document completes.

    Args:
        notes: List of dicts with document_id and text.
        output_path: Path for output JSONL file.
        max_retries: Max retries per failed document.
        log_file: Optional log file path.

    Returns:
        BatchStats with processing summary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger("batch-runner", log_file=log_file)
    stats = BatchStats(total=len(notes))

    log_event(logger, "batch_started", total_documents=len(notes))

    # Load model once for entire batch
    predictor = NERPredictor()

    with open(output_path, "w") as out_file:
        for i, note in enumerate(notes):
            doc_id = note.get("document_id", f"doc_{i}")
            text = note.get("text", "")

            if not text.strip():
                log_event(logger, "document_skipped",
                          document_id=doc_id, reason="empty text")
                stats.failed += 1
                continue

            result = _process_with_retry(
                predictor=predictor,
                doc_id=doc_id,
                text=text,
                max_retries=max_retries,
                logger=logger,
                stats=stats,
            )

            if result:
                # Incremental write
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()
                stats.success += 1

                log_event(logger, "document_processed",
                          document_id=doc_id,
                          entities_found=result["processing_metadata"]["total_entities"],
                          processing_time_ms=result["processing_metadata"]["processing_time_ms"])
            else:
                stats.failed += 1

            # Progress update every 10 docs
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(notes)} documents...")

    log_event(logger, "batch_completed",
              total=stats.total,
              success=stats.success,
              failed=stats.failed,
              retried=stats.retried,
              total_time_ms=stats.total_time_ms)

    return stats


def _process_with_retry(
    predictor: NERPredictor,
    doc_id: str,
    text: str,
    max_retries: int,
    logger,
    stats: BatchStats,
) -> dict | None:
    """Process a single document with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()

            # Preprocess
            preprocessed = preprocess_note(text)

            # ML prediction
            ml_entities = predictor.predict(preprocessed.cleaned_text)

            # Rule-based extraction
            rule_entities = extract_entities_by_rules(preprocessed.cleaned_text)

            # Merge and post-process
            accepted, flagged = postprocess(
                preprocessed.cleaned_text,
                ml_entities,
                rule_entities,
                preprocessed.sections,
            )

            elapsed_ms = (time.time() - start) * 1000
            stats.total_time_ms += elapsed_ms

            # Format output
            result = format_output(
                document_id=doc_id,
                accepted_entities=accepted,
                flagged_entities=flagged,
                processing_time_ms=elapsed_ms,
            )

            return result

        except Exception as e:
            if attempt < max_retries:
                stats.retried += 1
                log_error(logger, "document_retry",
                          error=e,
                          document_id=doc_id,
                          attempt=attempt)
            else:
                log_error(logger, "document_failed",
                          error=e,
                          document_id=doc_id,
                          attempts=max_retries)
                return None