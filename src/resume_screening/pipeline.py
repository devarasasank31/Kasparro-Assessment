"""End-to-end screening pipeline.

Stages: INGEST -> PARSE -> EXTRACT -> HARD FILTER -> LLM -> SCORE ->
GITHUB -> FINAL SCORE -> RANK -> OUTPUT

Each stage is isolated so a failure in one resume (or one network call) never
aborts the batch.
"""

from __future__ import annotations

import hashlib
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from .cache import RunCache
from .config import Settings
from .eligibility import check_eligibility
from .extractor import extract_candidate
from .github import enrich_candidates
from .llm import LLMAnalysis, LLMError, build_client
from .models import Candidate, EligibilityResult
from .output import build_records
from .parser import ingest_directory
from .ranking import rank_candidates, score_stats
from .scoring import ScoreResult, score_candidate
from .utils import get_logger

log = get_logger("pipeline")


@contextmanager
def _timed(stages: dict[str, float], name: str):
    """Record how long a pipeline stage took, in milliseconds."""
    started = time.perf_counter()
    try:
        yield
    finally:
        stages[name] = round((time.perf_counter() - started) * 1000, 1)


def _run_metadata(settings: Settings, started_at: datetime, duration_ms: float) -> dict[str, Any]:
    """Reproducibility block: what ran, when, with which configuration."""
    from . import __version__

    return {
        "tool_version": __version__,
        "python": platform.python_version(),
        "started_at": started_at.isoformat(),
        "duration_ms": duration_ms,
        "settings": {
            "input_dir": str(settings.input_dir),
            "output_path": str(settings.output_path),
            "use_llm": settings.use_llm,
            "llm_enabled": settings.llm_enabled,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "use_github": settings.use_github,
            "github_max_workers": settings.github_max_workers,
            "use_cache": settings.use_cache,
            "limit": settings.limit,
            "weights": dict(settings.weights),
        },
    }


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------
def _extract_all(resumes: list[Any]) -> tuple[list[Candidate], dict[str, EligibilityResult], list[dict[str, str]]]:
    candidates: list[Candidate] = []
    eligibility: dict[str, EligibilityResult] = {}
    failures: list[dict[str, str]] = []
    for resume in resumes:
        try:
            candidate = extract_candidate(resume)
            eligibility[candidate.source_file] = check_eligibility(candidate)
            candidates.append(candidate)
        except Exception as exc:  # broad by design: isolate failures per resume
            log.exception("Extraction failed for %s", resume.filename)
            failures.append({"file": resume.filename, "error": f"{type(exc).__name__}: {exc}"})
    return candidates, eligibility, failures


def _enrich_with_llm(
    candidates: list[Candidate],
    eligibility: dict[str, EligibilityResult],
    settings: Settings,
    store: RunCache | None = None,
) -> tuple[dict[str, LLMAnalysis], list[dict[str, str]]]:
    """Optional structured model pass over eligible resumes only.

    Successful responses are cached so a re-run does not re-bill the model.
    """
    analyses: dict[str, LLMAnalysis] = {}
    failures: list[dict[str, str]] = []
    if not settings.use_llm or not settings.llm_enabled:
        log.info("LLM enrichment skipped (disabled or missing credentials)")
        return analyses, failures
    client = build_client(settings)
    if client is None:
        return analyses, failures

    targets = [c for c in candidates if eligibility[c.source_file].eligible]
    log.info("LLM enrichment: %d eligible resumes via %s", len(targets), settings.llm_provider)
    for candidate in targets:
        cache_key = _llm_cache_key(candidate, settings)
        cached = store.get("llm", cache_key) if store is not None else None
        if isinstance(cached, dict):
            try:
                analyses[candidate.source_file] = LLMAnalysis.model_validate(cached)
                continue
            except Exception:  # noqa: BLE001 - stale/corrupt entry is just a miss
                log.warning("Discarding unreadable LLM cache entry for %s", candidate.source_file)
        try:
            analysis = client.analyse(candidate.full_text)
        except LLMError as exc:
            log.warning("LLM analysis failed for %s: %s", candidate.source_file, exc)
            failures.append({"file": candidate.source_file, "error": str(exc)})
            continue
        except Exception as exc:  # broad by design: a model quirk must not kill the run
            log.exception("Unexpected LLM failure for %s", candidate.source_file)
            failures.append(
                {"file": candidate.source_file, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        analyses[candidate.source_file] = analysis
        if store is not None:
            store.set("llm", cache_key, analysis.model_dump())
    log.info("LLM enrichment complete: %d ok, %d failed", len(analyses), len(failures))
    return analyses, failures


def _llm_cache_key(candidate: Candidate, settings: Settings) -> str:
    digest = hashlib.sha256(
        f"{settings.llm_provider}|{settings.llm_model}|{candidate.full_text}".encode()
    ).hexdigest()
    return f"analysis:{digest}"


def _score_all(
    candidates: list[Candidate],
    eligibility: dict[str, EligibilityResult],
    analyses: dict[str, LLMAnalysis],
    github_results: dict[str, Any],
) -> tuple[dict[str, ScoreResult], list[dict[str, str]]]:
    scores: dict[str, ScoreResult] = {}
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        source = candidate.source_file
        if not eligibility[source].eligible or source not in eligibility:
            continue
        analysis = analyses.get(source)
        adjustment = 0.0
        summary_override = None
        if analysis is not None:
            baseline = 0.0
            try:
                baseline = score_candidate(candidate, eligibility[source]).breakdown.ai_project_depth
            except Exception:  # broad by design: baseline is only used for the nudge
                log.exception("Baseline scoring failed for %s", source)
            adjustment = analysis.set_deterministic_depth(baseline)
            if analysis.project_summaries:
                summary_override = analysis.project_summaries[0].summary or None
        try:
            scores[source] = score_candidate(
                candidate,
                eligibility[source],
                github_results.get(source),
                llm_adjustment=adjustment,
                project_summary_override=summary_override,
            )
        except Exception as exc:  # broad by design: scoring must never abort the batch
            log.exception("Scoring failed for %s", source)
            failures.append({"file": source, "error": f"{type(exc).__name__}: {exc}"})
    return scores, failures


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(settings: Settings) -> dict[str, Any]:
    """Execute the screening pipeline and return a JSON-serialisable result."""
    log.info("Pipeline run starting (input=%s, output=%s)", settings.input_dir, settings.output_path)
    if not settings.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {settings.input_dir}")

    started_at = datetime.now(timezone.utc)
    clock = time.perf_counter()
    stages: dict[str, float] = {}

    with _timed(stages, "ingest"):
        ingestion = ingest_directory(settings.input_dir)
        if settings.limit and settings.limit < len(ingestion.resumes):
            ingestion.resumes = ingestion.resumes[: settings.limit]
            ingestion.total_files = len(ingestion.resumes)
            log.info("Limiting run to the first %d resumes", settings.limit)
        counts = ingestion.counts()

    store = RunCache(settings.cache_dir, enabled=settings.use_cache)
    try:
        with _timed(stages, "extract_and_filter"):
            candidates, eligibility, extraction_failures = _extract_all(ingestion.parsed)
        with _timed(stages, "llm"):
            analyses, llm_failures = _enrich_with_llm(candidates, eligibility, settings, store)
        with _timed(stages, "github"):
            github_results, github_failures = enrich_candidates(
                candidates, settings, store=store
            )
    finally:
        store.flush()

    with _timed(stages, "score"):
        scores, scoring_failures = _score_all(candidates, eligibility, analyses, github_results)

    eligible = [c for c in candidates if eligibility[c.source_file].eligible]
    rejected = [c for c in candidates if not eligibility[c.source_file].eligible]
    ranked = rank_candidates(eligible, scores)
    unscored = [c for c in eligible if c.source_file not in scores]

    duration_ms = round((time.perf_counter() - clock) * 1000, 1)
    stages["total"] = duration_ms

    log.info(
        "Batch complete: %d files, %d parsed, %d eligible, %d rejected, %d scored (%.0f ms)",
        counts["total_files"],
        counts["parsed"],
        len(eligible),
        len(rejected),
        len(scores),
        duration_ms,
    )

    return {
        "run": _run_metadata(settings, started_at, duration_ms),
        "summary": {
            "total_resumes": counts["total_files"],
            "parsed": counts["parsed"],
            "failed": counts["failed"],
            "duplicates": counts["duplicates"],
            "extracted": len(candidates),
            "eligible": len(eligible),
            "rejected": len(rejected),
            "scored": len(scores),
            "llm_used": bool(analyses),
            "llm_failures": len(llm_failures),
            "github_enriched": sum(1 for g in github_results.values() if g.status == "ok"),
            "github_failures": len(github_failures),
            "cache": store.summary(),
            "score_stats": score_stats(ranked, scores),
            "duration_ms": duration_ms,
            "stages_ms": stages,
            "status": "complete",
        },
        "parse_issues": [
            r.to_dict() for r in ingestion.resumes if r.status not in {"parsed", "duplicate"}
        ],
        "duplicates": [r.filename for r in ingestion.duplicates],
        "failures": {
            "extraction": extraction_failures,
            "llm": llm_failures,
            "github": github_failures,
            "scoring": scoring_failures,
        },
        "candidates": build_records(
            candidates, eligibility, scores, ranked, unscored, rejected, analyses, github_results
        ),
    }
