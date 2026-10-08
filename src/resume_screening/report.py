"""Human readable terminal report for a completed run (bonus feature)."""

from __future__ import annotations

from typing import Any

LINE_WIDTH = 100


def _bar(value: float, maximum: float, width: int = 12) -> str:
    if maximum <= 0:
        return " " * width
    filled = round((value / maximum) * width)
    return "#" * filled + "-" * (width - filled)


def format_report(result: dict[str, Any], top: int = 10) -> str:
    """Render a compact summary plus the top candidates as a fixed-width table."""
    summary = result.get("summary", {})
    stats = summary.get("score_stats", {}) or {}
    candidates = result.get("candidates", [])
    ranked = [c for c in candidates if c.get("rank") is not None]
    rejected = [c for c in candidates if c.get("eligible") is False]

    lines: list[str] = ["=" * LINE_WIDTH, "AI RESUME SCREENING - RUN REPORT", "=" * LINE_WIDTH]

    lines.append("Batch")
    batch_rows = [
        ("resumes found", summary.get("total_resumes")),
        ("parsed", summary.get("parsed")),
        ("failed / unreadable", summary.get("failed")),
        ("duplicates skipped", summary.get("duplicates")),
        ("eligible", summary.get("eligible")),
        ("rejected", summary.get("rejected")),
        ("scored", summary.get("scored")),
        ("llm used", summary.get("llm_used")),
        ("github enriched", summary.get("github_enriched")),
        ("github issues", summary.get("github_failures")),
        ("duration (ms)", summary.get("duration_ms")),
    ]
    for label, value in batch_rows:
        lines.append(f"  {label:<24} {value}")

    stages = summary.get("stages_ms", {}) or {}
    if stages:
        lines.append(
            "  stages (ms)               "
            + "  ".join(f"{name}={value}" for name, value in stages.items())
        )

    if stats.get("count"):
        lines.append("")
        lines.append(
            "Scores  "
            f"min={stats['min']}  median={stats['median']}  mean={stats['mean']}  max={stats['max']}  |  "
            f"strong={stats['strong_fit']} good={stats['good_fit']} "
            f"possible={stats['possible_fit']} weak={stats['weak_fit']}"
        )

    lines.append("")
    lines.append(f"Top {min(top, len(ranked))} of {len(ranked)} ranked candidates")
    lines.append("-" * LINE_WIDTH)
    lines.append(
        f"{'#':>3}  {'candidate':<24} {'total':>6} {'tier':<15} "
        f"{'ai/40':<13} {'py/30':<13} {'github':<10} profile"
    )
    for record in ranked[:top]:
        breakdown = record.get("score_breakdown", {}) or {}
        lines.append(
            f"{record.get('rank'):>3}  "
            f"{_truncate(str(record.get('candidate_name', '')), 24):<24} "
            f"{record.get('total_score'):>6} "
            f"{record.get('fit_tier', '')!s:<15} "
            f"{_bar(float(breakdown.get('ai_project_depth', 0)), 40)} "
            f"{_bar(float(breakdown.get('python_backend', 0)), 30)} "
            f"{record.get('github_status', 'no_profile'):<10}"
        )

    if rejected:
        lines.append("")
        lines.append(f"Rejected ({len(rejected)})")
        lines.append("-" * LINE_WIDTH)
        for record in rejected[:top]:
            reasons = ", ".join(record.get("rejection_reasons", []) or []) or "n/a"
            lines.append(f"  {_truncate(str(record.get('candidate_name', '')), 26):<26} {reasons}")

    failures = result.get("failures", {}) or {}
    issue_count = sum(len(v) for v in failures.values() if isinstance(v, list))
    if issue_count:
        lines.append("")
        lines.append(f"Failures ({issue_count}) - see the 'failures' block in results.json")

    lines.append("=" * LINE_WIDTH)
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"
