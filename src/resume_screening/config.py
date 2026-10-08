"""Central configuration.

All tunable values (weights, thresholds, timeouts, credentials) live here so
that business logic never hard-codes numbers or secrets. Credentials are only
ever read from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

# --------------------------------------------------------------------------
# Scoring weights (total = 100)
# --------------------------------------------------------------------------
SCORE_WEIGHTS: dict[str, int] = {
    "ai_project_depth": 40,
    "python_backend": 30,
    "cloud_fullstack": 15,
    "github": 10,
    "engineering_depth": 5,
}
TOTAL_SCORE = sum(SCORE_WEIGHTS.values())  # 100

# --------------------------------------------------------------------------
# Hard eligibility
# --------------------------------------------------------------------------
#: Minimum number of distinct AI evidence hits required to pass the AI gate.
MIN_AI_EVIDENCE_HITS = 1

# --------------------------------------------------------------------------
# GitHub enrichment (max 10 points, split 5 activity + 5 repositories)
# --------------------------------------------------------------------------
GITHUB_ACTIVITY_MAX = 5
GITHUB_REPOS_MAX = 5
GITHUB_MAX_REPOS_INSPECTED = 30
GITHUB_EVENT_WINDOW_DAYS = 90
GITHUB_REQUEST_TIMEOUT = 10.0
GITHUB_MAX_PAGES = 3

# --------------------------------------------------------------------------
# Network / reliability
# --------------------------------------------------------------------------
LLM_TIMEOUT_SECONDS = 30.0
HTTP_MAX_RETRIES = 2
HTTP_BACKOFF_SECONDS = 0.5

# --------------------------------------------------------------------------
# Deterministic heuristics knobs
# --------------------------------------------------------------------------
#: A project is considered "recent" for activity scoring when updated within
#: this many days of the run.
RECENT_ACTIVITY_DAYS = 180
#: Penalty applied for shallow / tutorial-style AI projects.
SHALLOW_PROJECT_PENALTY = 10
#: Maximum total penalty applied under "project quality" deductions.
MAX_PROJECT_PENALTY = 15


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for a single screening run."""

    input_dir: Path = Path("./resumes")
    output_path: Path = Path("./output/results.json")

    # Feature toggles
    use_llm: bool = True
    use_github: bool = True

    # LLM (never hard-coded; empty string == unavailable)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout: float = LLM_TIMEOUT_SECONDS

    # GitHub
    github_token: str = ""
    github_timeout: float = GITHUB_REQUEST_TIMEOUT

    # Misc
    verbose: bool = False
    log_level: str = "INFO"
    cache_dir: Path = Path(".cache")
    use_cache: bool = True
    #: 0 = process every resume; N > 0 stops after N files (smoke tests).
    limit: int = 0

    # Scoring weights (kept here so tests can override)
    weights: Mapping[str, int] = field(default_factory=lambda: dict(SCORE_WEIGHTS))

    # ------------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        """LLM is only truly usable when toggled on *and* keyed."""
        return bool(self.use_llm and self.llm_api_key)

    @property
    def github_enabled(self) -> bool:
        return self.use_github

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        """Build settings from environment variables plus explicit overrides."""
        base = cls(
            llm_provider=os.environ.get("LLM_PROVIDER", "openai").strip() or "openai",
            llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            llm_api_key=os.environ.get("LLM_API_KEY", "").strip(),
            llm_base_url=os.environ.get("LLM_BASE_URL", "").strip(),
            llm_timeout=_env_float("LLM_TIMEOUT_SECONDS", LLM_TIMEOUT_SECONDS),
            github_token=os.environ.get("GITHUB_TOKEN", "").strip(),
            github_timeout=_env_float("GITHUB_TIMEOUT_SECONDS", GITHUB_REQUEST_TIMEOUT),
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip() or "INFO",
            use_cache=_env_bool("USE_CACHE", True),
        )
        if overrides:
            base = replace(base, **overrides)  # type: ignore[arg-type]
        return base

    # ------------------------------------------------------------------
    @classmethod
    def from_args(
        cls,
        *,
        input_dir: str | Path,
        output_path: str | Path,
        model: str | None = None,
        no_llm: bool = False,
        no_github: bool = False,
        no_cache: bool = False,
        limit: int = 0,
        verbose: bool = False,
        **env_overrides: object,
    ) -> "Settings":
        """Combine CLI arguments (highest precedence) with environment values."""
        overrides: dict[str, object] = {
            "input_dir": Path(input_dir),
            "output_path": Path(output_path),
            "use_llm": not no_llm,
            "use_github": not no_github,
            "use_cache": not no_cache,
            "limit": max(int(limit), 0),
            "verbose": verbose,
        }
        if model:
            overrides["llm_model"] = model
        overrides.update(env_overrides)
        return cls.from_env(**overrides)


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny dependency-free `.env` loader (no-op when the file is missing).

    Existing environment variables always win so real env config is preserved.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
    except OSError:
        return
