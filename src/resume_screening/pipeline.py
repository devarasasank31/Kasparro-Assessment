"""End-to-end screening pipeline.

Stages: INGEST -> PARSE -> EXTRACT -> HARD FILTER -> LLM -> SCORE ->
GITHUB -> FINAL SCORE -> RANK -> OUTPUT

Each stage is isolated so a failure in one resume never aborts the batch.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .eligibility import check_eligibility
from .extractor import extract_candidate
from .models import Candidate, EligibilityResult
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
    eligibility: dict[str, EligibilityResult] = {}
    extraction_failures: list[dict[str, str]] = []
    for resume in ingestion.parsed:
        try:
            candidate = extract_candidate(resume)
            eligibility[candidate.source_file] = check_eligibility(candidate)
            candidates.append(candidate)
        except Exception as exc:  # noqa: BLE001 - isolate failures per resume
            log.exception("Extraction failed for %s", resume.filename)
            extraction_failures.append(
                {"file": resume.filename, "error": f"{type(exc).__name__}: {exc}"}
            )

    eligible = [c for c in candidates if eligibility[c.source_file].eligible]
    rejected = [c for c in candidates if not eligibility[c.source_file].eligible]
    log.info(
        "Extracted %d candidates (%d extraction failures) -> %d eligible, %d rejected",
        len(candidates),
        len(extraction_failures),
        len(eligible),
        len(rejected),
    )

    return {
        "summary": {
            "total_resumes": counts["total_files"],
            "parsed": counts["parsed"],
            "failed": counts["failed"],
            "duplicates": counts["duplicates"],
            "extracted": len(candidates),
            "eligible": len(eligible),
            "rejected": len(rejected),
            "status": "filtered",
        },
        "extraction_failures": extraction_failures,
        "parse_issues": [r.to_dict() for r in ingestion.resumes if not r.ok],
        "candidates": [
            {
                "candidate_name": c.name,
                "source_file": c.source_file,
                "eligible": eligibility[c.source_file].eligible,
                "rejection_reasons": eligibility[c.source_file].rejection_reasons,
                "matched_skills": eligibility[c.source_file].matched_skills[:15],
                "python_evidence": eligibility[c.source_file].python_evidence[:2],
                "ai_evidence": eligibility[c.source_file].ai_evidence[:3],
                "github": c.github_url,
            }
            for c in candidates
        ],
    }
