"""End-to-end screening pipeline.

Stages: INGEST -> PARSE -> EXTRACT -> HARD FILTER -> LLM -> SCORE ->
GITHUB -> FINAL SCORE -> RANK -> OUTPUT

Each stage is isolated so a failure in one resume never aborts the batch.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .extractor import extract_candidate
from .models import Candidate
from .parser import ingest_directory
from .utils import get_logger

log = get_logger("pipeline")


def run_pipeline(settings: Settings) -> dict[str, Any]:
    """Execute the screening pipeline and return a JSON-serialisable result."""
    log.info("Pipeline run starting (input=%s, output=%s)", settings.input_dir, settings.output_path)
    if not settings.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {settings.input_dir}")

    ingestion = ingest_directory(settings.input_dir)
    counts = ingestion.counts()

    candidates: list[Candidate] = []
    extraction_failures: list[dict[str, str]] = []
    for resume in ingestion.parsed:
        try:
            candidates.append(extract_candidate(resume))
        except Exception as exc:  # noqa: BLE001 - isolate failures per resume
            log.exception("Extraction failed for %s", resume.filename)
            extraction_failures.append(
                {"file": resume.filename, "error": f"{type(exc).__name__}: {exc}"}
            )

    log.info("Extracted %d candidates (%d extraction failures)", len(candidates), len(extraction_failures))

    return {
        "summary": {
            "total_resumes": counts["total_files"],
            "parsed": counts["parsed"],
            "failed": counts["failed"],
            "duplicates": counts["duplicates"],
            "extracted": len(candidates),
            "eligible": 0,
            "rejected": 0,
            "status": "extracted",
        },
        "extraction_failures": extraction_failures,
        "parse_issues": [
            r.to_dict() for r in ingestion.resumes if not r.ok
        ],
        "candidates": [
            {
                "candidate_name": c.name,
                "source_file": c.source_file,
                "email": c.email,
                "github": c.github_url,
                "skills": c.skills[:15],
                "projects": len(c.projects),
                "experiences": len(c.experiences),
            }
            for c in candidates
        ],
    }
