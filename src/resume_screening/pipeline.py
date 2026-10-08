"""End-to-end screening pipeline.

Stages: INGEST -> PARSE -> EXTRACT -> HARD FILTER -> LLM -> SCORE ->
GITHUB -> FINAL SCORE -> RANK -> OUTPUT

Each stage is isolated so a failure in one resume never aborts the batch.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .utils import get_logger

log = get_logger("pipeline")


def run_pipeline(settings: Settings) -> dict[str, Any]:
    """Execute the screening pipeline and return a JSON-serialisable result."""
    log.info("Pipeline run starting (input=%s, output=%s)", settings.input_dir, settings.output_path)
    if not settings.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {settings.input_dir}")

    return {
        "summary": {
            "total_resumes": 0,
            "parsed": 0,
            "failed": 0,
            "eligible": 0,
            "rejected": 0,
            "status": "skeleton",
        },
        "candidates": [],
    }
