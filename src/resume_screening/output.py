"""Turn pipeline stage results into the documented JSON shape.

Kept separate from :mod:`resume_screening.pipeline` so the output contract is
easy to inspect and test on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ranking import fit_tier

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .eligibility import EligibilityResult
    from .github import GitHubEnrichment
    from .llm import LLMAnalysis
    from .models import Candidate
    from .scoring import ScoreResult

# Fields that every record carries, even rejected ones.
BASE_FIELDS = (
    "candidate_name",
    "source_file",
    "eligible",
    "matched_skills",
    "rejection_reasons",
    "email",
    "github_url",
    "github_status",
    "github_summary",
    "llm_used",
)


def base_record(
    candidate: "Candidate",
    eligibility: "EligibilityResult",
    github: "GitHubEnrichment | None",
    analysis: "LLMAnalysis | None",
) -> dict[str, Any]:
    return {
        "candidate_name": candidate.name,
        "source_file": candidate.source_file,
        "eligible": eligibility.eligible,
        "matched_skills": eligibility.matched_skills[:20],
        "rejection_reasons": eligibility.rejection_reasons,
        "email": candidate.email,
        "github_url": candidate.github_url,
        "github_status": github.status if github else "no_profile",
        "github_summary": github.summary if github else "",
        "llm_used": analysis is not None,
    }


def score_record(
    candidate: "Candidate",
    score: "ScoreResult",
    rank: int,
    analysis: "LLMAnalysis | None",
) -> dict[str, Any]:
    total = score.total()
    return {
        "rank": rank,
        "fit_tier": fit_tier(total),
        "total_score": total,
        "score_breakdown": score.breakdown.as_dict(),
        "project_summary": score.project_summary,
        "strengths": score.strengths,
        "concerns": score.concerns,
        "evidence": score.evidence_by_category(),
        **_llm_fields(analysis),
    }


def _llm_fields(analysis: "LLMAnalysis | None") -> dict[str, Any]:
    if analysis is None:
        return {}
    return {
        "llm_rationale": analysis.rationale,
        "llm_ai_project_depth": analysis.ai_project_depth,
        "llm_adjustment": analysis.adjustment,
    }


def build_records(
    candidates: list["Candidate"],
    eligibility: dict[str, "EligibilityResult"],
    scores: dict[str, "ScoreResult"],
    ranked: list["Candidate"],
    unscored: list["Candidate"],
    rejected: list["Candidate"],
    analyses: dict[str, "LLMAnalysis"],
    github_results: dict[str, "GitHubEnrichment"],
) -> list[dict[str, Any]]:
    """Ranked candidates first, then unscored, then rejected - stable order."""
    output: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, start=1):
        source = candidate.source_file
        record = base_record(
            candidate, eligibility[source], github_results.get(source), analyses.get(source)
        )
        record.update(
            score_record(candidate, scores[source], rank, analyses.get(source))
        )
        output.append(record)

    for candidate in unscored:
        source = candidate.source_file
        record = base_record(
            candidate, eligibility[source], github_results.get(source), analyses.get(source)
        )
        record.update(
            {"rank": None, "fit_tier": None, "total_score": None,
             "concerns": ["Scoring failed for this candidate"]}
        )
        output.append(record)

    for candidate in rejected:
        source = candidate.source_file
        record = base_record(
            candidate, eligibility[source], github_results.get(source), analyses.get(source)
        )
        record.update({"rank": None, "fit_tier": None, "total_score": None})
        output.append(record)

    return output
