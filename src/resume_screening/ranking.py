"""Ranking, fit tiers and batch statistics.

Ranking is fully deterministic: score desc, then candidate name, then source
file, so a re-run over the same inputs always produces the same ordering.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import Candidate
    from .scoring import ScoreResult

log = get_logger("ranking")

#: Score at or above which a candidate is considered a strong fit.
TIER_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (80.0, "strong_fit"),
    (65.0, "good_fit"),
    (50.0, "possible_fit"),
    (0.0, "weak_fit"),
)


def fit_tier(total: float) -> str:
    """Map a 0-100 total score onto a human readable fit tier."""
    for threshold, label in TIER_THRESHOLDS:
        if total >= threshold:
            return label
    return "weak_fit"  # pragma: no cover - guarded by the 0.0 floor


def rank_candidates(
    candidates: Iterable[Candidate],
    scores: dict[str, ScoreResult],
) -> list[Candidate]:
    """Return candidates ordered best-first with deterministic tie-breaking."""

    def sort_key(candidate: Candidate) -> tuple[float, str, str]:
        score = scores.get(candidate.source_file)
        total = score.total() if score is not None else float("-inf")
        return (-total, candidate.name.lower(), candidate.source_file)

    ranked = sorted((c for c in candidates if c.source_file in scores), key=sort_key)
    log.debug("Ranked %d candidates", len(ranked))
    return ranked


def score_stats(ranked: Iterable[Candidate], scores: dict[str, ScoreResult]) -> dict[str, Any]:
    """Batch level distribution of total scores (handy for the report)."""
    values = sorted(scores[c.source_file].total() for c in ranked)
    if not values:
        return {"count": 0}

    def percentile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        index = (len(values) - 1) * p
        low = int(index)
        high = min(low + 1, len(values) - 1)
        fraction = index - low
        return round(values[low] + (values[high] - values[low]) * fraction, 2)

    return {
        "count": len(values),
        "min": values[0],
        "max": values[-1],
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "strong_fit": sum(1 for v in values if fit_tier(v) == "strong_fit"),
        "good_fit": sum(1 for v in values if fit_tier(v) == "good_fit"),
        "possible_fit": sum(1 for v in values if fit_tier(v) == "possible_fit"),
        "weak_fit": sum(1 for v in values if fit_tier(v) == "weak_fit"),
    }
