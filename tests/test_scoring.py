from resume_screening.eligibility import check_eligibility
from resume_screening.extractor import extract_candidate
from resume_screening.models import GitHubEnrichment
from resume_screening.parser import ParsedResume
from resume_screening.scoring import build_corpora, project_quality_penalties, score_candidate


def _candidate(text: str, filename: str = "candidate_00.pdf"):
    return extract_candidate(ParsedResume(path=None, filename=filename, text=text))  # type: ignore[arg-type]


STRONG = """\
Asha Rao
asha@example.com | github.com/asha

SKILLS
Languages: Python, SQL
AI: LangChain, LangGraph, RAG, Embeddings, Docker, GCP
Backend: FastAPI, PostgreSQL, Redis

PROJECTS
Stateful Support Agent | Python, FastAPI, LangGraph, pgvector | 2025
- Built a multi-agent RAG workflow with tool calling, conversation state and citation checks.
- Added an evaluation harness (RAGAS) and cut hallucination rate by 38%.
- Deployed on GCP with Docker Compose and GitHub Actions; pytest suite covers retrieval.

Ticket Triage Agent | Python, LangGraph, Redis | 2024
- Implemented a tool-calling agent that routes tickets and writes back to PostgreSQL.

WORK EXPERIENCE
AI Engineer Intern | Acme Corp | Jan 2025 - Present
- Built an LLM document pipeline over 40k PDFs using embeddings and vector search.
"""

THIN = """\
Rohit Sen
rohit@example.com

SKILLS
Languages: Python
AI: OpenAI, LangChain, ChatGPT

PROJECTS
PDF Chatbot
- Built a simple chatbot that calls the OpenAI API and returns answers.
"""


def test_strong_candidate_scores_near_top_and_within_bounds() -> None:
    candidate = _candidate(STRONG)
    eligibility = check_eligibility(candidate)
    assert eligibility.eligible

    result = score_candidate(candidate, eligibility, GitHubEnrichment())
    total = result.total()

    assert 0 <= total <= 100
    assert result.breakdown.ai_project_depth >= 30
    assert result.breakdown.python_backend >= 20
    assert result.breakdown.cloud_fullstack >= 8
    assert result.breakdown.penalties == 0
    assert result.project_summary
    assert result.strengths
    assert sum(e.points for e in result.evidence if e.category == "ai_project_depth") >= 30


def test_skills_only_ai_scores_lower_than_project_evidence() -> None:
    skills_only = """\
    Neha Rao
    SKILLS
    Languages: Python
    AI: LangChain, RAG, Embeddings, LangGraph
    """
    project_based = """\
    Neha Rao
    SKILLS
    Languages: Python
    AI: LangChain, RAG, Embeddings, LangGraph
    PROJECTS
    Doc Assistant | Python, LangChain, pgvector
    - Built a RAG pipeline with embeddings, chunking and vector search.
    """
    skills_result = score_candidate(_candidate(skills_only), check_eligibility(_candidate(skills_only)))
    project_result = score_candidate(
        _candidate(project_based), check_eligibility(_candidate(project_based))
    )

    assert project_result.breakdown.ai_project_depth > skills_result.breakdown.ai_project_depth


def test_shallow_ai_project_is_penalised() -> None:
    candidate = _candidate(THIN)
    corpora = build_corpora(candidate)
    penalty = project_quality_penalties(candidate, corpora)

    assert penalty.points < 0
    assert penalty.reasons
    assert abs(penalty.points) <= 15

    result = score_candidate(candidate, check_eligibility(candidate))
    assert result.breakdown.penalties < 0
    assert any("thin LLM" in concern or "tutorial" in concern.lower() for concern in result.concerns)


def test_github_component_is_capped_at_ten() -> None:
    candidate = _candidate(STRONG)
    eligibility = check_eligibility(candidate)
    github = GitHubEnrichment(
        status="ok",
        activity_score=5,
        repository_score=5,
        total_score=10,
        summary="Very active",
    )
    result = score_candidate(candidate, eligibility, github)
    assert result.breakdown.github == 10

    over = GitHubEnrichment(status="ok", activity_score=9, repository_score=9, total_score=18)
    assert score_candidate(candidate, eligibility, over).breakdown.github == 10


def test_missing_github_scores_zero_and_is_not_fatal() -> None:
    candidate = _candidate(STRONG)
    result = score_candidate(candidate, check_eligibility(candidate), None)
    assert result.breakdown.github == 0
    assert 0 <= result.total() <= 100


def test_llm_adjustment_is_bounded() -> None:
    candidate = _candidate(STRONG)
    eligibility = check_eligibility(candidate)
    baseline = score_candidate(candidate, eligibility).breakdown.ai_project_depth
    boosted = score_candidate(candidate, eligibility, llm_adjustment=4).breakdown.ai_project_depth
    clamped = score_candidate(candidate, eligibility, llm_adjustment=99).breakdown.ai_project_depth

    assert boosted == min(baseline + 4, 40)
    assert clamped == 40


def test_total_is_sum_of_categories_minus_penalties() -> None:
    candidate = _candidate(STRONG)
    result = score_candidate(candidate, check_eligibility(candidate))
    breakdown = result.breakdown

    expected = (
        breakdown.ai_project_depth
        + breakdown.python_backend
        + breakdown.cloud_fullstack
        + breakdown.github
        + breakdown.engineering_depth
        + breakdown.penalties
    )
    assert round(expected, 1) == result.total()
