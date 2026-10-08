from resume_screening.models import Candidate, EligibilityResult, ScoreBreakdown
from resume_screening.output import BASE_FIELDS, build_records
from resume_screening.ranking import TIER_THRESHOLDS, fit_tier, rank_candidates, score_stats
from resume_screening.scoring import ScoreResult


def _candidate(filename: str, name: str) -> Candidate:
    return Candidate(source_file=filename, name=name)


def _score(ai: float = 0, py: float = 0, cloud: float = 0, gh: float = 0, eng: float = 0,
           penalties: float = 0) -> ScoreResult:
    return ScoreResult(
        breakdown=ScoreBreakdown(
            ai_project_depth=ai,
            python_backend=py,
            cloud_fullstack=cloud,
            github=gh,
            engineering_depth=eng,
            penalties=penalties,
        )
    )


def _eligible() -> EligibilityResult:
    return EligibilityResult(eligible=True, matched_skills=["Python", "LangChain"])


def test_fit_tier_boundaries() -> None:
    assert fit_tier(100) == "strong_fit"
    assert fit_tier(80) == "strong_fit"
    assert fit_tier(79.9) == "good_fit"
    assert fit_tier(65) == "good_fit"
    assert fit_tier(64.9) == "possible_fit"
    assert fit_tier(50) == "possible_fit"
    assert fit_tier(49.9) == "weak_fit"
    assert fit_tier(0) == "weak_fit"
    # thresholds are declared in descending order so the walk is correct
    assert list(TIER_THRESHOLDS) == sorted(TIER_THRESHOLDS, key=lambda t: -t[0])


def test_rank_candidates_sorts_by_score_then_name() -> None:
    low = _candidate("b.pdf", "Zoe")
    mid = _candidate("a.pdf", "anna")
    tie_a = _candidate("1.pdf", "Ada")
    tie_b = _candidate("0.pdf", "Ada")

    scores = {
        "b.pdf": _score(ai=10),
        "a.pdf": _score(ai=20),
        "1.pdf": _score(ai=40),
        "0.pdf": _score(ai=40),
    }
    ranked = rank_candidates([low, mid, tie_a, tie_b], scores)

    assert [c.source_file for c in ranked] == ["0.pdf", "1.pdf", "a.pdf", "b.pdf"]


def test_rank_candidates_ignores_candidates_without_scores() -> None:
    scored = _candidate("a.pdf", "Ada")
    unscored = _candidate("b.pdf", "Bob")
    ranked = rank_candidates([unscored, scored], {"a.pdf": _score(ai=30)})
    assert [c.source_file for c in ranked] == ["a.pdf"]


def test_score_stats_reports_distribution() -> None:
    candidates = [_candidate(f"{i}.pdf", f"C{i}") for i in range(4)]
    scores = {
        "0.pdf": _score(ai=10),
        "1.pdf": _score(ai=20),
        "2.pdf": _score(ai=30),
        "3.pdf": _score(ai=90),
    }
    stats = score_stats(candidates, scores)

    assert stats["count"] == 4
    assert stats["min"] == 10
    assert stats["max"] == 90
    assert stats["mean"] == 37.5
    assert stats["median"] == 25.0
    assert stats["strong_fit"] == 1  # only the 90
    assert stats["good_fit"] == 0
    assert stats["possible_fit"] == 0
    assert stats["weak_fit"] == 3


def test_score_stats_handles_an_empty_batch() -> None:
    assert score_stats([], {}) == {"count": 0}


def _records(analyses=None, github=None):
    ranked = [_candidate("a.pdf", "Ada"), _candidate("b.pdf", "Ben")]
    rejected = [_candidate("c.pdf", "Cara")]
    scores = {"a.pdf": _score(ai=40, py=30, cloud=15, gh=10, eng=3),
              "b.pdf": _score(ai=10, py=10)}
    eligibility = {"a.pdf": _eligible(), "b.pdf": _eligible(),
                   "c.pdf": EligibilityResult(eligible=False, rejection_reasons=["No AI/agentic project evidence"])}
    return build_records(
        candidates=ranked + rejected,
        eligibility=eligibility,
        scores=scores,
        ranked=ranked,
        unscored=[],
        rejected=rejected,
        analyses=analyses or {},
        github_results=github or {},
    )


def test_records_carry_rank_tier_and_breakdown() -> None:
    records = _records()

    assert [r["rank"] for r in records] == [1, 2, None]
    assert records[0]["candidate_name"] == "Ada"
    assert records[0]["fit_tier"] == "strong_fit"
    assert records[0]["total_score"] == 98.0
    assert records[1]["fit_tier"] == "weak_fit"
    assert set(records[0]["score_breakdown"]) == {
        "ai_project_depth", "python_backend", "cloud_fullstack",
        "github", "engineering_depth", "penalties",
    }


def test_every_record_includes_the_base_fields() -> None:
    for record in _records():
        for field in BASE_FIELDS:
            assert field in record, f"missing {field}"


def test_rejected_records_expose_their_reasons() -> None:
    records = _records()
    rejected = records[-1]
    assert rejected["eligible"] is False
    assert rejected["rejection_reasons"] == ["No AI/agentic project evidence"]
    assert rejected["rank"] is None
    assert rejected["total_score"] is None


def test_llm_fields_only_appear_when_a_model_ran() -> None:
    from resume_screening.llm import LLMAnalysis

    plain = _records()[0]
    assert "llm_rationale" not in plain
    assert plain["llm_used"] is False

    with_llm = _records(analyses={"a.pdf": LLMAnalysis(ai_project_depth=35, rationale="solid")})[0]
    assert with_llm["llm_used"] is True
    assert with_llm["llm_rationale"] == "solid"
    assert with_llm["llm_ai_project_depth"] == 35


def test_github_status_defaults_to_no_profile() -> None:
    records = _records()
    assert records[0]["github_status"] == "no_profile"
    assert records[0]["github_summary"] == ""
