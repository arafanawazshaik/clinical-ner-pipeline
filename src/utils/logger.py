"""
Structured Logger.

JSON-formatted logging for production monitoring and audit trails.
Each log entry includes document_id, model_version, and timestamps.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


def get_logger(
    name: str = "clinical-ner",
    log_file: str | Path = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a structured JSON logger.

    Args:
        name: Logger name.
        log_file: Optional file path for log output.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = JSONFormatter()

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_event(logger: logging.Logger, event: str, **kwargs):
    """Log a structured event with extra data."""
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.extra_data = kwargs
    logger.handle(record)


def log_error(logger: logging.Logger, event: str, error: Exception, **kwargs):
    """Log an error with full details."""
    record = logger.makeRecord(
        name=logger.name,
        level=logging.ERROR,
        fn="",
        lno=0,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.extra_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **kwargs,
    }
    logger.handle(record)