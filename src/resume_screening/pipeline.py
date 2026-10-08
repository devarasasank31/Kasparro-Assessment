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
from .scoring import ScoreResult, score_candidate
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

    scores: dict[str, ScoreResult] = {}
    scoring_failures: list[dict[str, str]] = []
    for candidate in eligible:
        try:
            scores[candidate.source_file] = score_candidate(candidate, eligibility[candidate.source_file])
        except Exception as exc:  # noqa: BLE001 - scoring must never abort the batch
            log.exception("Scoring failed for %s", candidate.source_file)
            scoring_failures.append(
                {"file": candidate.source_file, "error": f"{type(exc).__name__}: {exc}"}
            )

    ranked = sorted(
        (c for c in eligible if c.source_file in scores),
        key=lambda c: (-scores[c.source_file].total(), c.name.lower()),
    )
    unscored = [c for c in eligible if c.source_file not in scores]

    log.info(
        "Extracted %d candidates (%d extraction failures) -> %d eligible, %d rejected, scored %d",
        len(candidates),
        len(extraction_failures),
        len(eligible),
        len(rejected),
        len(scores),
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
            "scored": len(scores),
            "status": "scored",
        },
        "extraction_failures": extraction_failures,
        "scoring_failures": scoring_failures,
        "parse_issues": [r.to_dict() for r in ingestion.resumes if not r.ok],
        "candidates": [
            {
                "rank": rank + 1,
                "candidate_name": c.name,
                "source_file": c.source_file,
                "eligible": eligibility[c.source_file].eligible,
                "total_score": scores[c.source_file].total(),
                "score_breakdown": scores[c.source_file].breakdown.as_dict(),
                "rejection_reasons": eligibility[c.source_file].rejection_reasons,
                "matched_skills": eligibility[c.source_file].matched_skills[:15],
                "project_summary": scores[c.source_file].project_summary,
                "strengths": scores[c.source_file].strengths,
                "concerns": scores[c.source_file].concerns,
                "score_evidence": scores[c.source_file].evidence_by_category(),
                "github": c.github_url,
            }
            for rank, c in enumerate(ranked)
        ]
        + [
            {
                "rank": None,
                "candidate_name": c.name,
                "source_file": c.source_file,
                "eligible": True,
                "total_score": None,
                "rejection_reasons": [],
                "matched_skills": eligibility[c.source_file].matched_skills[:15],
                "concerns": ["Scoring failed for this candidate"],
                "github": c.github_url,
            }
            for c in unscored
        ]
        + [
            {
                "rank": None,
                "candidate_name": c.name,
                "source_file": c.source_file,
                "eligible": False,
                "total_score": None,
                "rejection_reasons": eligibility[c.source_file].rejection_reasons,
                "matched_skills": eligibility[c.source_file].matched_skills[:15],
                "github": c.github_url,
            }
            for c in rejected
        ],
    }
