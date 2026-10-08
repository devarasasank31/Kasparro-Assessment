import json
import time

from resume_screening.cache import DEFAULT_TTL, RunCache
from resume_screening.config import Settings


def test_miss_then_hit(tmp_path) -> None:
    cache = RunCache(tmp_path)
    assert cache.get("github", "k") is None
    cache.set("github", "k", {"a": 1})
    assert cache.get("github", "k") == {"a": 1}
    stats = cache.summary()["github"]
    assert stats == {"hits": 1, "misses": 1, "writes": 1}


def test_entries_survive_a_reload(tmp_path) -> None:
    writer = RunCache(tmp_path)
    writer.set("github", "ada", {"status": "ok"})
    writer.flush()

    reader = RunCache(tmp_path)
    assert reader.get("github", "ada") == {"status": "ok"}
    assert reader.summary()["github"]["hits"] == 1


def test_expired_entries_are_treated_as_misses(tmp_path) -> None:
    cache = RunCache(tmp_path)
    cache.set("github", "ada", {"status": "ok"})
    cache.flush()

    # Backdate the stored timestamp beyond the namespace TTL.
    path = tmp_path / "github.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ada"]["stored_at"] = time.time() - DEFAULT_TTL["github"] - 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert RunCache(tmp_path).get("github", "ada") is None


def test_namespaces_are_isolated(tmp_path) -> None:
    cache = RunCache(tmp_path)
    cache.set("llm", "key", "value")
    assert cache.get("github", "key") is None
    assert cache.get("llm", "key") == "value"


def test_disabled_cache_never_reads_or_writes(tmp_path) -> None:
    cache = RunCache(tmp_path, enabled=False)
    cache.set("github", "k", 1)
    cache.flush()
    assert cache.get("github", "k") is None
    assert cache.summary() == {}
    assert not (tmp_path / "github.json").exists()


def test_corrupt_cache_file_is_ignored(tmp_path) -> None:
    (tmp_path / "github.json").write_text("{not json", encoding="utf-8")
    cache = RunCache(tmp_path)
    assert cache.get("github", "k") is None
    cache.set("github", "k", {"ok": True})
    cache.flush()
    assert RunCache(tmp_path).get("github", "k") == {"ok": True}


def test_flush_does_not_raise_when_the_path_is_a_directory(tmp_path) -> None:
    (tmp_path / "github.json").mkdir()
    cache = RunCache(tmp_path)
    cache.set("github", "k", 1)
    cache.flush()  # must not raise
    assert "github" not in [p for p in cache._dirty]  # noqa: SLF001


def test_cache_is_never_required(tmp_path) -> None:
    # A cache directory that does not exist yet is created on flush.
    target = tmp_path / "nested" / "deeper"
    cache = RunCache(target)
    cache.set("llm", "k", {"x": 1})
    cache.flush()
    assert (target / "llm.json").is_file()


def test_settings_expose_cache_controls() -> None:
    settings = Settings.from_args(input_dir="in", output_path="out.json", no_cache=True)
    assert settings.use_cache is False
    assert Settings.from_args(input_dir="in", output_path="out.json").use_cache is True


LLM_RESUME = """Jane Doe
Email: jane@example.com

Skills: Python, FastAPI, LangChain, Docker

Projects
- AI Resume Screener - Built an agentic LangChain pipeline in Python that parses resumes
  and ranks candidates with FastAPI endpoints and a Postgres store.
"""


def _eligible_candidate():
    from resume_screening.eligibility import check_eligibility
    from resume_screening.extractor import extract_candidate
    from resume_screening.parser import ParsedResume

    candidate = extract_candidate(
        ParsedResume(path=None, filename="jane.pdf", text=LLM_RESUME)  # type: ignore[arg-type]
    )
    return candidate, check_eligibility(candidate)


def test_llm_analysis_is_only_requested_once(tmp_path, monkeypatch) -> None:
    from resume_screening import pipeline as pipeline_mod
    from resume_screening.llm import LLMAnalysis

    candidate, eligibility = _eligible_candidate()
    assert eligibility.eligible

    class StubClient:
        calls = 0

        def analyse(self, text: str) -> LLMAnalysis:
            StubClient.calls += 1
            return LLMAnalysis(ai_project_depth=30, rationale="real agentic project")

    stub = StubClient()
    monkeypatch.setattr(pipeline_mod, "build_client", lambda settings: stub)
    settings = Settings(use_llm=True, llm_api_key="test-key", cache_dir=tmp_path)

    first_store = RunCache(tmp_path)
    first, failures = pipeline_mod._enrich_with_llm(
        [candidate], {candidate.source_file: eligibility}, settings, first_store
    )
    first_store.flush()

    second, _ = pipeline_mod._enrich_with_llm(
        [candidate], {candidate.source_file: eligibility}, settings, RunCache(tmp_path)
    )

    assert not failures
    assert stub.calls == 1
    assert first[candidate.source_file].ai_project_depth == 30
    assert second[candidate.source_file].ai_project_depth == 30
    assert second[candidate.source_file].rationale == "real agentic project"


def test_llm_stage_makes_no_calls_when_disabled(tmp_path, monkeypatch) -> None:
    from resume_screening import pipeline as pipeline_mod

    candidate, eligibility = _eligible_candidate()

    def _fail(settings):  # pragma: no cover - must never run
        raise AssertionError("build_client should not be consulted when use_llm is off")

    monkeypatch.setattr(pipeline_mod, "build_client", _fail)
    settings = Settings(use_llm=False, cache_dir=tmp_path)
    analyses, failures = pipeline_mod._enrich_with_llm(
        [candidate], {candidate.source_file: eligibility}, settings, RunCache(tmp_path)
    )
    assert analyses == {}
    assert failures == []
