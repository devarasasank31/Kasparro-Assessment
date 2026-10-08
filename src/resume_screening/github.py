"""Public GitHub enrichment.

Uses only the anonymous/public GitHub REST API. Credentials, when present, come
from ``GITHUB_TOKEN`` via ``Settings`` - never from source code.

Guarantees:
- a missing, private or malformed profile never fails the candidate
- rate limits and network errors are recorded as a status, not an exception
- requests are bounded (profile + one repository page per username) and cached
  for the duration of the run
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import requests

from .cache import RunCache
from .config import (
    GITHUB_ACTIVITY_MAX,
    GITHUB_MAX_PAGES,
    GITHUB_MAX_REPOS_INSPECTED,
    GITHUB_REPOS_MAX,
    GITHUB_REQUEST_TIMEOUT,
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    RECENT_ACTIVITY_DAYS,
    Settings,
)
from .extractor import GITHUB_RESERVED
from .models import Candidate, GitHubEnrichment
from .utils import get_logger

log = get_logger("github")

API_ROOT = "https://api.github.com"
USER_AGENT = "resume-screening-bot/1.0"
RELEVANT_LANGUAGE_RE = re.compile(r"python|jupyter|jupyter notebook", re.I)
RELEVANT_TOPIC_RE = re.compile(
    r"ai|ml|llm|rag|agent|langchain|langgraph|nlp|deep-learning|machine-learning|"
    r"transformers|embeddings|openai|gpt|pytorch|tensorflow",
    re.I,
)


def extract_username(profile_url: str | None) -> str | None:
    """Pull a GitHub username out of a profile URL."""
    if not profile_url:
        return None
    match = re.search(r"github\.com/([A-Za-z0-9-]+)", profile_url, re.I)
    if not match:
        return None
    username = match.group(1)
    if username.lower() in GITHUB_RESERVED or username.lower() == "settings":
        return None
    return username


class GitHubClient:
    """Thin, timeout-bounded wrapper over the public GitHub REST API."""

    def __init__(
        self,
        token: str = "",
        timeout: float = GITHUB_REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.token = token
        self.timeout = timeout
        self.session = session or requests.Session()
        self.rate_limited = False

    # ------------------------------------------------------------------
    @property
    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a GitHub API path with bounded retries.

        Transient network errors are retried; 404s and rate limits are not.
        Raises :class:`GitHubAPIError` on failure.
        """
        if self.rate_limited:
            raise GitHubAPIError("rate_limited", "run already rate limited")
        url = f"{API_ROOT}{path}"
        response: requests.Response | None = None
        last_error: Exception | None = None
        for attempt in range(HTTP_MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    url, headers=self.headers, params=params, timeout=self.timeout
                )
                last_error = None
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < HTTP_MAX_RETRIES:
                    time.sleep(HTTP_BACKOFF_SECONDS * (attempt + 1))
        if last_error is not None or response is None:
            raise GitHubAPIError("error", f"network error: {last_error}") from last_error

        if response.status_code == 404:
            raise GitHubAPIError("not_found", "profile or resource not found")
        if response.status_code in {403, 429}:
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining == "0" or response.status_code == 429:
                self.rate_limited = True
                raise GitHubAPIError("rate_limited", "GitHub API rate limit reached")
            raise GitHubAPIError("error", f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise GitHubAPIError("error", f"HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubAPIError("error", "response was not valid JSON") from exc


class GitHubAPIError(Exception):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _activity_score(recent_push_days: list[int], now: datetime) -> float:
    """0-5: how recently and how often the candidate pushes public code."""
    if not recent_push_days:
        return 0.0
    newest = min(recent_push_days)
    if newest <= 30:
        recency = 3.0
    elif newest <= 90:
        recency = 2.0
    elif newest <= RECENT_ACTIVITY_DAYS:
        recency = 1.0
    else:
        recency = 0.0

    active_count = sum(1 for age in recent_push_days if age <= 90)
    if active_count >= 5:
        volume = 2.0
    elif active_count >= 2:
        volume = 1.5
    elif active_count >= 1:
        volume = 1.0
    else:
        volume = 0.0
    return round(min(GITHUB_ACTIVITY_MAX, recency + volume), 1)


def _repository_score(repos: list[dict[str, Any]], now: datetime) -> float:
    """0-5: maintained, relevant public repositories."""
    usable = [r for r in repos if not r.get("fork") and not r.get("archived")]
    usable = usable[:GITHUB_MAX_REPOS_INSPECTED]
    if not usable:
        return 0.0

    owned = len(usable)
    if owned >= 10:
        points = 2.0
    elif owned >= 5:
        points = 1.5
    elif owned >= 3:
        points = 1.0
    else:
        points = 0.5

    relevant = 0
    for repo in usable:
        haystack = " ".join(
            [
                str(repo.get("language") or ""),
                " ".join(repo.get("topics") or []),
                str(repo.get("description") or ""),
                str(repo.get("name") or ""),
            ]
        )
        if RELEVANT_LANGUAGE_RE.search(haystack) or RELEVANT_TOPIC_RE.search(haystack):
            relevant += 1
    if relevant >= 3:
        points += 2.0
    elif relevant >= 1:
        points += 1.0

    maintained = sum(
        1
        for repo in usable
        if (_parse_date(repo.get("pushed_at")) or datetime.min.replace(tzinfo=timezone.utc))
        >= now - timedelta(days=RECENT_ACTIVITY_DAYS)
    )
    if maintained >= 2:
        points += 1.0
    elif maintained >= 1:
        points += 0.5

    return round(min(GITHUB_REPOS_MAX, points), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def enrich_candidate(
    candidate: Candidate,
    client: GitHubClient | None,
    cache: dict[str, GitHubEnrichment],
    enabled: bool = True,
    store: RunCache | None = None,
) -> GitHubEnrichment:
    """Enrich one candidate. Always returns a result - never raises."""
    if not enabled:
        return GitHubEnrichment(status="disabled", profile_url=candidate.github_url)

    username = extract_username(candidate.github_url)
    if not username:
        return GitHubEnrichment(status="no_profile", profile_url=candidate.github_url)
    if username in cache:
        return cache[username]

    profile_url = f"https://github.com/{username}"
    if store is not None:
        cached = store.get("github", f"enrichment:{username}")
        if isinstance(cached, dict):
            try:
                result = GitHubEnrichment.model_validate(cached)
            except Exception:  # noqa: BLE001 - a stale/corrupt entry is just a miss
                result = None  # type: ignore[assignment]
            if result is not None:
                cache[username] = result
                log.debug("GitHub cache hit for %s", username)
                return result

    if client is None:
        result = GitHubEnrichment(username=username, profile_url=profile_url, status="disabled")
        cache[username] = result
        return result

    now = datetime.now(timezone.utc)
    try:
        profile = client.get(f"/users/{username}")
        repos = client.get(
            f"/users/{username}/repos",
            params={"per_page": 100, "sort": "pushed", "type": "owner"},
        )
    except GitHubAPIError as exc:
        result = GitHubEnrichment(
            username=username,
            profile_url=profile_url,
            status=exc.status,  # type: ignore[arg-type]
            error=str(exc),
            summary=f"GitHub enrichment failed: {exc}",
        )
        cache[username] = result
        log.info("GitHub enrichment %s for %s: %s", exc.status, username, exc)
        return result

    if not isinstance(repos, list):
        repos = []

    ages: list[int] = []
    for repo in repos:
        pushed = _parse_date(repo.get("pushed_at") or repo.get("updated_at"))
        if pushed:
            ages.append(max(0, (now - pushed).days))

    activity = _activity_score(ages, now)
    repository = _repository_score(repos, now)
    public_repos = int(profile.get("public_repos") or 0)
    recent = sum(1 for age in ages if age <= 90)
    relevant = sum(
        1
        for repo in repos[:GITHUB_MAX_REPOS_INSPECTED]
        if not repo.get("fork")
        and (
            RELEVANT_LANGUAGE_RE.search(str(repo.get("language") or ""))
            or RELEVANT_TOPIC_RE.search(
                " ".join([str(repo.get("description") or ""), " ".join(repo.get("topics") or [])])
            )
        )
    )
    last_activity_days = min(ages) if ages else None
    last_activity_text = (
        f"last push {last_activity_days}d ago"
        if last_activity_days is not None
        else "no recent pushes"
    )
    summary = (
        f"{public_repos} public repos, {recent} pushed in the last 90 days, "
        f"{relevant} Python/AI relevant; {last_activity_text}"
    )

    result = GitHubEnrichment(
        username=username,
        profile_url=profile_url,
        status="ok",
        activity_score=activity,
        repository_score=repository,
        total_score=round(activity + repository, 1),
        summary=summary,
        public_repos=public_repos,
        recent_repos=recent,
        last_activity=(f"{last_activity_days} days ago" if last_activity_days is not None else None),
    )
    if store is not None:
        store.set("github", f"enrichment:{username}", result.model_dump())
    cache[username] = result
    return result


def enrich_candidates(
    candidates: Iterable[Candidate],
    settings: Settings,
    client: GitHubClient | None = None,
    store: RunCache | None = None,
) -> tuple[dict[str, GitHubEnrichment], list[dict[str, str]]]:
    """Enrich every candidate that exposes a GitHub profile."""
    candidates = list(candidates)
    results: dict[str, GitHubEnrichment] = {}
    failures: list[dict[str, str]] = []
    if not settings.use_github:
        log.info("GitHub enrichment disabled by configuration")
        for candidate in candidates:
            results[candidate.source_file] = GitHubEnrichment(
                status="disabled", profile_url=candidate.github_url
            )
        return results, failures

    client = client or GitHubClient(token=settings.github_token, timeout=settings.github_timeout)
    cache: dict[str, GitHubEnrichment] = {}
    for candidate in candidates:
        enrichment = enrich_candidate(candidate, client, cache, enabled=True, store=store)
        results[candidate.source_file] = enrichment
        if enrichment.status in {"error", "rate_limited", "not_found"} and candidate.github_url:
            failures.append(
                {
                    "file": candidate.source_file,
                    "username": enrichment.username or "",
                    "status": enrichment.status,
                    "error": enrichment.error or "",
                }
            )
        if client.rate_limited:
            # Stop calling the API for the rest of the run, but keep recording
            # statuses so the output stays honest.
            for remaining in candidates:
                if remaining.source_file not in results:
                    results[remaining.source_file] = GitHubEnrichment(
                        username=extract_username(remaining.github_url),
                        profile_url=remaining.github_url,
                        status="rate_limited",
                        error="run stopped early after GitHub rate limit",
                    )
            break

    ok = sum(1 for r in results.values() if r.status == "ok")
    log.info("GitHub enrichment complete: %d ok, %d issues", ok, len(failures))
    return results, failures
