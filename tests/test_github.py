import json
import threading
import time

import pytest
import requests

from resume_screening.cache import RunCache
from resume_screening.config import Settings
from resume_screening.extractor import extract_candidate
from resume_screening.github import (
    GitHubAPIError,
    GitHubClient,
    enrich_candidates,
    extract_username,
)
from resume_screening.models import Candidate
from resume_screening.parser import ParsedResume


def _candidate(github_url: str | None, filename: str = "candidate_00.pdf") -> Candidate:
    text = f"Resume\nGitHub: {github_url}\n" if github_url else "Resume\n"
    return extract_candidate(ParsedResume(path=None, filename=filename, text=text))  # type: ignore[arg-type]


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, headers=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text or json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(url)
        return self.responses.pop(0)


def test_extract_username_variants() -> None:
    assert extract_username("https://github.com/ada-l") == "ada-l"
    assert extract_username("github.com/ada-l/") == "ada-l"
    assert extract_username("https://www.github.com/ada-l?tab=repos") == "ada-l"
    assert extract_username(None) is None
    assert extract_username("https://github.com/features") is None


def test_missing_profile_is_recorded_not_raised() -> None:
    candidate = _candidate("https://github.com/nobody-here-xyz")
    session = FakeSession([FakeResponse(status_code=404)])
    client = GitHubClient(session=session)

    results, failures = enrich_candidates([candidate], Settings(use_github=True), client=client)

    outcome = results[candidate.source_file]
    assert outcome.status == "not_found"
    assert outcome.total_score == 0
    assert failures and failures[0]["status"] == "not_found"


def test_successful_enrichment_is_capped_at_ten_points() -> None:
    candidate = _candidate("https://github.com/ada-l")
    profile = {"public_repos": 12}
    repos = [
        {
            "name": f"repo-{i}",
            "language": "Python",
            "fork": False,
            "archived": False,
            "pushed_at": "2026-10-01T00:00:00Z",
            "description": "LLM RAG agent",
            "topics": ["langchain"],
        }
        for i in range(12)
    ]
    session = FakeSession(
        [
            FakeResponse(payload=profile),
            FakeResponse(payload=repos),
        ]
    )
    client = GitHubClient(session=session)

    results, failures = enrich_candidates([candidate], Settings(use_github=True), client=client)
    outcome = results[candidate.source_file]

    assert outcome.status == "ok"
    assert outcome.total_score <= 10
    assert outcome.activity_score <= 5
    assert outcome.repository_score <= 5
    assert outcome.total_score == outcome.activity_score + outcome.repository_score
    assert "public repos" in outcome.summary
    assert not failures


def test_same_username_is_only_requested_once() -> None:
    first = _candidate("https://github.com/ada-l", filename="a.pdf")
    second = _candidate("https://github.com/ada-l", filename="b.pdf")
    session = FakeSession(
        [
            FakeResponse(payload={"public_repos": 2}),
            FakeResponse(payload=[]),
        ]
    )
    client = GitHubClient(session=session)

    results, _ = enrich_candidates([first, second], Settings(use_github=True), client=client)

    assert len(session.calls) == 2  # one profile + one repos call for both candidates
    assert results["a.pdf"].total_score == results["b.pdf"].total_score


def test_rate_limit_stops_further_requests() -> None:
    first = _candidate("https://github.com/ada-l", filename="a.pdf")
    second = _candidate("https://github.com/bob-l", filename="b.pdf")
    session = FakeSession([FakeResponse(status_code=403, headers={"X-RateLimit-Remaining": "0"})])
    client = GitHubClient(session=session)

    results, failures = enrich_candidates([first, second], Settings(use_github=True), client=client)

    assert results["a.pdf"].status == "rate_limited"
    assert results["b.pdf"].status == "rate_limited"
    assert [f["status"] for f in failures] == ["rate_limited", "rate_limited"]
    assert len(session.calls) == 1  # no hammering after the limit


def test_network_failure_is_recorded() -> None:
    candidate = _candidate("https://github.com/ada-l")

    class BoomSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    client = GitHubClient(session=BoomSession())
    results, failures = enrich_candidates([candidate], Settings(use_github=True), client=client)

    assert results[candidate.source_file].status == "error"
    assert results[candidate.source_file].error
    assert failures


def test_disabled_github_makes_no_network_calls() -> None:
    candidate = _candidate("https://github.com/ada-l")

    class ExplodingSession:
        def get(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("must not be called")

    client = GitHubClient(session=ExplodingSession())
    results, failures = enrich_candidates([candidate], Settings(use_github=False), client=client)

    assert results[candidate.source_file].status == "disabled"
    assert not failures


def test_candidate_without_profile_needs_no_client() -> None:
    candidate = _candidate(None)
    results, failures = enrich_candidates([candidate], Settings(use_github=True))
    assert results[candidate.source_file].status == "no_profile"
    assert not failures


def test_github_api_error_carries_status() -> None:
    err = GitHubAPIError("rate_limited", "slow down")
    assert err.status == "rate_limited"
    assert str(err) == "slow down"


def test_second_run_serves_github_from_cache(tmp_path) -> None:
    candidate = _candidate("https://github.com/ada-l")
    session = FakeSession([FakeResponse(payload={"public_repos": 3}), FakeResponse(payload=[])])
    client = GitHubClient(session=session)
    store = RunCache(tmp_path)

    first, _ = enrich_candidates([candidate], Settings(use_github=True), client=client, store=store)
    store.flush()
    assert first[candidate.source_file].status == "ok"
    assert len(session.calls) == 2

    # Second run: a fresh cache read must avoid any further HTTP calls.
    second, failures = enrich_candidates(
        [candidate], Settings(use_github=True), client=client, store=RunCache(tmp_path)
    )
    assert second[candidate.source_file].status == "ok"
    assert second[candidate.source_file].total_score == first[candidate.source_file].total_score
    assert len(session.calls) == 2
    assert not failures


def test_rate_limited_results_are_not_cached(tmp_path) -> None:
    candidate = _candidate("https://github.com/ada-l")
    limiter = FakeSession([FakeResponse(status_code=403, headers={"X-RateLimit-Remaining": "0"})])
    store = RunCache(tmp_path)
    enrich_candidates(
        [candidate], Settings(use_github=True), client=GitHubClient(session=limiter), store=store
    )
    store.flush()

    # The failure must not poison the cache: a later run retries the API.
    retrier = FakeSession([FakeResponse(payload={"public_repos": 1}), FakeResponse(payload=[])])
    results, _ = enrich_candidates(
        [candidate], Settings(use_github=True), client=GitHubClient(session=retrier), store=RunCache(tmp_path)
    )
    assert results[candidate.source_file].status == "ok"
    assert len(retrier.calls) == 2


class RoutingSession:
    """URL addressed fake GitHub API that is safe to call from many threads."""

    def __init__(self, profiles: dict, repos: dict, delay: float = 0.0, explode: str | None = None):
        self.profiles = profiles
        self.repos = repos
        self.delay = delay
        self.explode = explode
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.in_flight = 0
        self.peak = 0

    def get(self, url, headers=None, params=None, timeout=None):
        with self._lock:
            self.calls.append(url)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if url.endswith("/repos"):
                username = url.split("/users/")[1].split("/")[0]
                if self.explode == username:
                    raise RuntimeError("unexpected worker failure")
                return FakeResponse(payload=self.repos.get(username, []))
            username = url.rsplit("/", 1)[1]
            if self.explode == username:
                raise RuntimeError("unexpected worker failure")
            if username not in self.profiles:
                return FakeResponse(status_code=404)
            return FakeResponse(payload=self.profiles[username])
        finally:
            with self._lock:
                self.in_flight -= 1


def _batch(count: int) -> list[Candidate]:
    return [
        _candidate(f"https://github.com/user-{i}", filename=f"candidate_{i:02d}.pdf")
        for i in range(count)
    ]


def test_concurrent_enrichment_is_bounded_and_matches_sequential() -> None:
    profiles = {f"user-{i}": {"public_repos": 4} for i in range(6)}
    repos = {f"user-{i}": [] for i in range(6)}
    batch = _batch(6)

    sequential_session = RoutingSession(profiles, repos)
    sequential, _ = enrich_candidates(
        batch,
        Settings(use_github=True, github_max_workers=4),
        client=GitHubClient(session=sequential_session),
        max_workers=1,
    )

    parallel_session = RoutingSession(profiles, repos, delay=0.1)
    parallel, _ = enrich_candidates(
        batch,
        Settings(use_github=True, github_max_workers=4),
        client=GitHubClient(session=parallel_session),
        max_workers=3,
    )

    assert sequential == parallel
    assert all(r.status == "ok" for r in parallel.values())
    assert parallel_session.peak <= 3  # bounded pool, never more than asked
    assert parallel_session.peak >= 2  # work really was overlapped


def test_worker_failure_is_isolated_to_one_profile() -> None:
    profiles = {"good": {"public_repos": 2}, "boom": {"public_repos": 2}}
    batch = [
        _candidate("https://github.com/good", filename="a.pdf"),
        _candidate("https://github.com/boom", filename="b.pdf"),
        _candidate("https://github.com/good", filename="c.pdf"),
    ]
    session = RoutingSession(profiles, {"good": [], "boom": []}, explode="boom")

    results, failures = enrich_candidates(
        batch,
        Settings(use_github=True),
        client=GitHubClient(session=session),
        max_workers=3,
    )

    assert results["b.pdf"].status == "error"
    assert results["b.pdf"].error
    assert results["a.pdf"].status == "ok"
    assert results["c.pdf"].status == "ok"
    assert [f["file"] for f in failures] == ["b.pdf"]


def test_default_worker_setting_is_positive_and_small() -> None:
    settings = Settings.from_env()
    assert 1 <= settings.github_max_workers <= 8


def test_client_requires_valid_json() -> None:
    client = GitHubClient(session=FakeSession([FakeResponse(payload=None)]))
    with pytest.raises(GitHubAPIError) as excinfo:
        client.get("/users/ada-l")
    assert excinfo.value.status == "error"
