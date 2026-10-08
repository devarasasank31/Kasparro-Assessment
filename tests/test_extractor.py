from resume_screening.extractor import (
    extract_candidate,
    extract_github,
    parse_projects,
    parse_skills_section,
    split_sections,
)
from resume_screening.parser import ParsedResume


def _parsed(text: str, filename: str = "candidate_01.pdf") -> ParsedResume:
    return ParsedResume(path=None, filename=filename, text=text)  # type: ignore[arg-type]


RESUME = """\
Ada Lovelace
London, UK | +44 7700 900123 | ada@example.com
GitHub: github.com/ada-l
LinkedIn: linkedin.com/in/ada-l

PROFESSIONAL SUMMARY
Mathematician building analytical engines.

TECHNICAL SKILLS
Languages: Python, SQL
AI / ML: LangChain, RAG, Embeddings, Vector Search
DevOps: Docker, GCP

WORK EXPERIENCE
Software Engineer | Analytical Engines Ltd | Jan 2023 - Present
- Built a RAG pipeline over internal documentation with citation checks.
- Designed FastAPI services with async workers.

PROJECTS
Doc Q&A System | Python, FastAPI, pgvector | 2024
- Retrieval augmented chatbot with evaluation harness and caching.
- Deployed on GCP with Docker Compose.

EDUCATION
B.A. Mathematics, University of London, 2020
"""


def test_split_sections_uses_synonyms() -> None:
    sections = split_sections(RESUME)
    assert "skills" in sections
    assert "experience" in sections
    assert "projects" in sections
    assert "education" in sections
    assert "Ada Lovelace" in sections["header"]


def test_merged_heading_falls_back_to_subsection_match() -> None:
    sections = split_sections("Header line\nWORK EXPERIENCE PROFESSIONAL SUMMARY\nDid things")
    assert "experience" in sections
    assert "Did things" in sections["experience"]


def test_parse_skills_section_groups_categories() -> None:
    groups = parse_skills_section("Languages: Python, SQL\nAI / ML: LangChain, RAG\nDocker, GCP")
    flat = {item for items in groups.values() for item in items}
    assert {"Python", "SQL", "LangChain", "RAG", "Docker", "GCP"} <= flat


def test_parse_projects_extracts_title_and_technologies() -> None:
    projects = parse_projects(
        "Doc Q&A System | Python, FastAPI, pgvector | 2024\n"
        "- Retrieval augmented chatbot with evaluation harness.\n"
        "- Deployed on GCP."
    )
    assert len(projects) == 1
    project = projects[0]
    assert project.name == "Doc Q&A System"
    assert "FastAPI" in project.technologies
    assert len(project.bullets) == 2
    assert "evaluation harness" in project.summary


def test_parse_projects_handles_plain_title_blocks() -> None:
    projects = parse_projects(
        "AI Document Intelligence\n"
        "Built a platform using RAG, FastAPI and Redis.\n"
        "Implemented async processing with embeddings.\n"
        "Job Market ETL Platform\n"
        "Built a backend system using FastAPI and PostgreSQL."
    )
    assert len(projects) == 2
    assert projects[0].name == "AI Document Intelligence"


def test_extract_candidate_full_record() -> None:
    candidate = extract_candidate(_parsed(RESUME))
    assert candidate.name == "Ada Lovelace"
    assert candidate.email == "ada@example.com"
    assert candidate.phone is not None
    assert candidate.github_url == "https://github.com/ada-l"
    assert candidate.linkedin_url == "https://www.linkedin.com/in/ada-l"
    assert "Python" in candidate.skills
    assert "LangChain" in candidate.skills
    assert candidate.projects and candidate.projects[0].name == "Doc Q&A System"
    assert candidate.experiences
    assert candidate.education
    assert "No GitHub profile found" not in candidate.extraction_warnings


def test_extract_github_ignores_reserved_paths() -> None:
    assert extract_github("see github.com/features and github.com/ada-l") == "https://github.com/ada-l"
    assert extract_github("nothing here") is None


def test_extract_candidate_without_text_does_not_crash() -> None:
    candidate = extract_candidate(_parsed("", filename="broken.pdf"))
    assert candidate.name == "broken"
    assert candidate.projects == []
    assert candidate.extraction_warnings
