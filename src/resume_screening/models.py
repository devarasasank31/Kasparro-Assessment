"""Typed internal models shared across the pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Project(BaseModel):
    """A single project entry parsed from a resume."""

    name: str
    technologies: list[str] = Field(default_factory=list)
    summary: str = ""
    bullets: list[str] = Field(default_factory=list)
    raw: str = ""

    @property
    def text(self) -> str:
        return " ".join([self.name, self.summary, " ".join(self.bullets)]).strip()


class Experience(BaseModel):
    """Work / internship experience block."""

    title: str = ""
    organization: str = ""
    period: str = ""
    bullets: list[str] = Field(default_factory=list)
    raw: str = ""

    @property
    def text(self) -> str:
        return " ".join([self.title, self.organization, " ".join(self.bullets)]).strip()


class Candidate(BaseModel):
    """Everything we could deterministically extract from one resume."""

    source_file: str
    name: str
    email: str | None = None
    phone: str | None = None
    github_url: str | None = None
    linkedin_url: str | None = None

    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    skill_groups: dict[str, list[str]] = Field(default_factory=dict)
    projects: list[Project] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)

    #: Raw text per detected section, keyed by normalised section name.
    sections: dict[str, str] = Field(default_factory=dict)
    #: Full normalised resume text (excluded from JSON serialisation).
    full_text: str = Field(default="", exclude=True)

    extraction_warnings: list[str] = Field(default_factory=list)

    @property
    def github_username(self) -> str | None:
        if not self.github_url:
            return None
        return self.github_url.rstrip("/").split("/")[-1] or None

    def section_text(self, *names: str) -> str:
        """Concatenate the raw text of the requested sections."""
        parts = [self.sections[n] for n in names if n in self.sections and self.sections[n]]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scoring / ranking models
# ---------------------------------------------------------------------------
class ScoreEvidence(BaseModel):
    """One explainable contribution to a score component."""

    label: str
    points: float
    detail: str = ""
    category: str = ""


class ScoreBreakdown(BaseModel):
    """Per-category points plus the evidence that produced them."""

    ai_project_depth: float = 0
    python_backend: float = 0
    cloud_fullstack: float = 0
    github: float = 0
    engineering_depth: float = 0
    penalties: float = 0

    def total(self) -> float:
        return (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
            + self.penalties
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "ai_project_depth": round(self.ai_project_depth, 1),
            "python_backend": round(self.python_backend, 1),
            "cloud_fullstack": round(self.cloud_fullstack, 1),
            "github": round(self.github, 1),
            "engineering_depth": round(self.engineering_depth, 1),
            "penalties": round(self.penalties, 1),
        }


class EligibilityResult(BaseModel):
    """Outcome of the deterministic hard filter."""

    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)


class GitHubEnrichment(BaseModel):
    """Public GitHub signal for one candidate (never fatal)."""

    username: str | None = None
    profile_url: str | None = None
    status: Literal["ok", "not_found", "rate_limited", "disabled", "error", "no_profile"] = "no_profile"
    activity_score: float = 0
    repository_score: float = 0
    total_score: float = 0
    summary: str = ""
    public_repos: int = 0
    recent_repos: int = 0
    last_activity: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ScreeningResult(BaseModel):
    """Final, JSON-serialisable record for one candidate."""

    rank: int | None = None
    candidate_name: str
    source_file: str
    eligible: bool
    total_score: float | None = None
    score_breakdown: dict[str, float] | None = None
    matched_skills: list[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    github_status: str = "no_profile"
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    score_evidence: list[dict[str, Any]] = Field(default_factory=list)
    enrichment: dict[str, Any] = Field(default_factory=dict)
