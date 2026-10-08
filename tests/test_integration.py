"""End-to-end batch tests over synthetic resumes in tests/fixtures/resumes."""

import json
from pathlib import Path

import pytest

from resume_screening.config import Settings
from resume_screening.pipeline import run_pipeline

FIXTURES = Path(__file__).parent / "fixtures" / "resumes"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        input_dir=FIXTURES,
        output_path=tmp_path / "results.json",
        use_llm=False,
        use_github=False,
        use_cache=False,
    )


def test_batch_survives_a_malformed_and_a_duplicate_file(settings: Settings) -> None:
    result = run_pipeline(settings)
    summary = result["summary"]

    assert summary["total_resumes"] == 6
    assert summary["parsed"] == 4
    assert summary["failed"] == 1
    assert summary["duplicates"] == 1
    assert summary["eligible"] == 2
    assert summary["rejected"] == 2
    assert summary["scored"] == 2
    assert summary["status"] == "complete"
    assert result["failures"]["extraction"] == []
    assert result["failures"]["scoring"] == []
    assert [p["file"] for p in result["parse_issues"]] == ["broken.pdf"]


def test_ranked_candidates_lead_the_output(settings: Settings) -> None:
    result = run_pipeline(settings)
    ranked = [c for c in result["candidates"] if c["rank"] is not None]

    assert [c["rank"] for c in ranked] == [1, 2]
    assert ranked[0]["candidate_name"] == "Asha Rao"
    assert ranked[1]["candidate_name"] == "Ravi Kumar"
    assert ranked[0]["total_score"] > ranked[1]["total_score"]
    assert ranked[0]["fit_tier"] in {"strong_fit", "good_fit"}
    for record in ranked:
        breakdown = record["score_breakdown"]
        assert set(breakdown) == {
            "ai_project_depth",
            "python_backend",
            "cloud_fullstack",
            "github",
            "engineering_depth",
            "penalties",
        }
        assert 0 <= record["total_score"] <= 100
        assert record["project_summary"] or record["strengths"]


def test_rejected_candidates_carry_explicit_reasons(settings: Settings) -> None:
    result = run_pipeline(settings)
    rejected = {c["candidate_name"]: c for c in result["candidates"] if not c["eligible"]}

    assert set(rejected) == {"Meera Shah", "Karan Patel"}
    assert rejected["Meera Shah"]["rejection_reasons"] == [
        "No evidence of Python stack",
        "No AI/agentic project evidence",
    ]
    assert rejected["Karan Patel"]["rejection_reasons"] == [
        "No AI/agentic project evidence"
    ]
    assert "Java" in rejected["Meera Shah"]["matched_skills"]
    for record in rejected.values():
        assert record["rank"] is None
        assert record["total_score"] is None


def test_every_parsed_resume_appears_exactly_once(settings: Settings) -> None:
    result = run_pipeline(settings)
    files = [c["source_file"] for c in result["candidates"]]
    assert sorted(files) == ["alpha.txt", "beta.txt", "delta.txt", "gamma.txt"]


def _without_timings(payload) -> dict:
    """Copy of a result with wall-clock fields removed for comparison."""
    clone = json.loads(json.dumps(payload))
    clone["run"].pop("started_at", None)
    clone["run"].pop("duration_ms", None)
    clone["summary"].pop("duration_ms", None)
    clone["summary"].pop("stages_ms", None)
    return clone


def test_result_is_json_serialisable_and_stable(settings: Settings) -> None:
    first = run_pipeline(settings)
    second = run_pipeline(settings)

    serialised = json.dumps(first, ensure_ascii=False)
    assert json.loads(serialised)["summary"]["total_resumes"] == 6
    assert _without_timings(first) == _without_timings(second)  # deterministic


def test_run_metadata_documents_the_configuration(settings: Settings) -> None:
    result = run_pipeline(settings)
    run = result["run"]

    assert run["tool_version"]
    assert run["python"].startswith("3.")
    assert run["started_at"].endswith("+00:00")
    assert run["settings"]["use_github"] is False
    assert run["settings"]["use_llm"] is False
    assert run["settings"]["weights"]["ai_project_depth"] == 40
    assert run["settings"]["weights"]["python_backend"] == 30

    summary = result["summary"]
    assert summary["duration_ms"] > 0
    assert set(summary["stages_ms"]) == {
        "ingest",
        "extract_and_filter",
        "llm",
        "github",
        "score",
        "total",
    }
    assert summary["stages_ms"]["total"] == summary["duration_ms"]


def test_github_is_honoured_as_optional_signal(tmp_path: Path) -> None:
    offline = run_pipeline(
        Settings(
            input_dir=FIXTURES,
            output_path=tmp_path / "a.json",
            use_llm=False,
            use_github=False,
            use_cache=False,
        )
    )
    assert offline["summary"]["github_enriched"] == 0
    records = [c for c in offline["candidates"] if c["rank"] is not None]
    assert records
    assert all(r["github_status"] == "disabled" for r in records)
