from resume_screening.eligibility import check_eligibility, find_ai_evidence, find_python_evidence
from resume_screening.extractor import extract_candidate
from resume_screening.parser import ParsedResume


def _candidate(text: str, filename: str = "candidate_00.pdf"):
    return extract_candidate(ParsedResume(path=None, filename=filename, text=text))  # type: ignore[arg-type]


PYTHON_AND_AI = """\
Ravi Kumar
ravi@example.com | github.com/ravi-dev

SKILLS
Languages: Python, SQL
AI: LangChain, RAG, Embeddings, Docker

PROJECTS
Support Agent | Python, FastAPI, LangGraph
- Built a multi-agent support workflow with retrieval and tool calling.
"""

PYTHON_ONLY = """\
Ravi Kumar
ravi@example.com

SKILLS
Languages: Python, SQL, Django
Backend: FastAPI, PostgreSQL, Redis

PROJECTS
Inventory API | Python, FastAPI
- Built a REST backend with async workers and PostgreSQL.
"""

AI_ONLY = """\
Ravi Kumar
ravi@example.com

SKILLS
Languages: JavaScript, TypeScript
AI: LangChain, RAG, Embeddings

PROJECTS
Doc Chat | React, LangChain
- Built a retrieval chat interface for internal documentation.
"""

JAVA_REACT_ONLY = """\
Anjali Patil
anjali@example.com

SKILLS
Languages: Java
Frameworks: Spring Boot, React.js, Hibernate
Databases: MySQL, Redis

PROJECTS
Learn Sphere | Java, Spring Boot
- Built an online learning platform with payment integration.
"""

JAVA_PLUS_PYTHON_AND_AI = """\
Anjali Patil
anjali@example.com

SKILLS
Languages: Java, Python
Frameworks: Spring Boot
AI: LangGraph, RAG

PROJECTS
Ticket Triage Agent | Python, LangGraph, Redis
- Built an agentic triage workflow with retrieval and tool calling.
"""


def test_python_and_ai_is_eligible() -> None:
    result = check_eligibility(_candidate(PYTHON_AND_AI))
    assert result.eligible is True
    assert result.rejection_reasons == []
    assert "Python" in result.matched_skills
    assert result.python_evidence
    assert any("LangChain" in item or "RAG" in item for item in result.ai_evidence)


def test_python_without_ai_is_rejected() -> None:
    result = check_eligibility(_candidate(PYTHON_ONLY))
    assert result.eligible is False
    assert result.rejection_reasons == ["No AI/agentic project evidence"]
    assert result.python_evidence
    assert result.matched_skills  # skills are still reported for rejected profiles


def test_ai_without_python_is_rejected() -> None:
    result = check_eligibility(_candidate(AI_ONLY))
    assert result.eligible is False
    assert result.rejection_reasons == ["No evidence of Python stack"]
    assert result.ai_evidence


def test_java_react_only_is_rejected() -> None:
    result = check_eligibility(_candidate(JAVA_REACT_ONLY))
    assert result.eligible is False
    assert "No evidence of Python stack" in result.rejection_reasons


def test_java_with_python_and_ai_is_eligible() -> None:
    """Extra languages must never disqualify a Python + AI candidate."""
    result = check_eligibility(_candidate(JAVA_PLUS_PYTHON_AND_AI))
    assert result.eligible is True
    assert result.rejection_reasons == []


def test_ai_claim_without_implementation_context_does_not_pass() -> None:
    """A bare 'Artificial Intelligence' course entry is not AI evidence."""
    text = """\
    Neha Verma
    SKILLS
    Languages: Python
    Coursework: Artificial Intelligence, Machine Learning
    """
    candidate = _candidate(text)
    assert find_ai_evidence(candidate) == []
    assert check_eligibility(candidate).eligible is False


def test_python_word_inside_other_words_does_not_count() -> None:
    candidate = _candidate("SKILLS\nFrameworks: Javascript, Djangoism\n")
    assert find_python_evidence(candidate) == []


def test_glued_together_python_is_still_detected() -> None:
    candidate = _candidate("Built tools inpython3 with FastAPI and LangGraph agents")
    assert find_python_evidence(candidate)
