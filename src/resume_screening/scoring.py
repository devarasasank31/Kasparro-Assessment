"""Deterministic, explainable 100-point scoring engine.

Weights (see ``config.SCORE_WEIGHTS``):

    AI / Agentic / RAG project depth ....... 40
    Python & backend engineering ........... 30
    Cloud / deployment / full stack ......... 15
    GitHub activity ......................... 10
    Engineering depth signals ............... 5

Every point is attached to a :class:`ScoreEvidence` row so the final ranking can
be justified line by line. Evidence found only in a skills list scores half of
evidence found in a project or internship description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import MAX_PROJECT_PENALTY, SCORE_WEIGHTS, SHALLOW_PROJECT_PENALTY
from .models import (
    Candidate,
    EligibilityResult,
    GitHubEnrichment,
    Project,
    ScoreBreakdown,
    ScoreEvidence,
)
from .utils import clamp, dedupe_preserve_order, get_logger

log = get_logger("scoring")

# ---------------------------------------------------------------------------
# Signal rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """One scored signal."""

    label: str
    points: float
    pattern: re.Pattern[str]
    detail: str = ""


def _rule(label: str, points: float, pattern: str, detail: str = "") -> Rule:
    return Rule(label=label, points=points, pattern=re.compile(pattern, re.I), detail=detail)


AI_RULES: list[Rule] = [
    _rule("Agentic workflow (LangGraph / agents / state)", 8,
          r"lang\s*graph|\bagentic\b|state graph|multi[- ]agent|agent (workflow|orchestration|graph)|autogen|crew\s*ai",
          "stateful agent orchestration"),
    _rule("RAG / retrieval pipeline", 7,
          r"\brag\b|retrieval[- ]augmented|retrieval pipeline|context window retrieval",
          "retrieval over documents"),
    _rule("Embeddings & vector search", 6,
          r"\bembeddings?\b|vector (search|database|db|store)|pgvector|chroma|pinecone|qdrant|weaviate|similarity search|semantic search",
          "vector retrieval layer"),
    _rule("LLM provider integration", 6,
          r"openai|chatgpt|gpt-[0-9]|\bgpt4\b|anthropic|claude|gemini|groq|mistral|deepseek|hugging\s?face|\bllms?\b|large language model",
          "production LLM call path"),
    _rule("Tool calling / tool use", 5,
          r"tool[- ]calling|tool use|function calling|mcp\b|tools?= *\[\?",
          "agent tool interface"),
    _rule("Evaluation / guardrails", 5,
          r"(evaluation|eval) (pipeline|harness|framework|dataset)|ragas|guardrail|human[- ]in[- ]the[- ]loop|a/?b test",
          "measured quality loop"),
    _rule("Framework implementation (LangChain / LlamaIndex / Haystack)", 5,
          r"lang\s*chain|llama\s*index|llamaindex|\bhaystack\b",
          "AI framework used in code"),
    _rule("Applied ML / DL / NLP / CV model work", 5,
          r"deep learning|machine learning|neural networks?|\bpytorch\b|\btensorflow\b|\bkeras\b|\bscikit[- ]?learn|\bsklearn\b|\bnlp\b|computer vision|\bopencv\b|xgboost|lightgbm",
          "model trained or applied"),
    _rule("Quantified impact on AI work", 4, r"\d+\s?%|\d+x (?:faster|improvement|accuracy)|latency|throughput",
          "measured outcomes"),
    _rule("Voice / multimodal AI", 3, r"speech recognition|text-to-speech|\btts\b|\bstt\b|whisper|multimodal|\bocr\b",
          "non-text modality"),
]

#: Marker used to decide whether a block of text is AI-related work at all.
AI_MARKER_RE = re.compile(
    r"\b(llm|llms|rag|agents?|agentic|langchain|lang\s?graph|llamaindex|embeddings?|"
    r"machine learning|deep learning|nlp|computer vision|pytorch|tensorflow|"
    r"scikit[- ]?learn|chatbots?|generative ai|genai|openai|gpt|claude|gemini|"
    r"transformers?|prompt engineering|vector search|semantic search)\b",
    re.I,
)

PYTHON_RULES: list[Rule] = [
    _rule("Python used in projects / internships", 9,
          r"python", "implementation language"),
    _rule("Backend framework (FastAPI / Django / Flask)", 7,
          r"fastapi|fast api|django|flask", "service layer"),
    _rule("Async / background processing", 5,
          r"\basyncio\b|async/await|celery|background (tasks|jobs)|websockets?|concurrent\.futures|threading|multiprocessing",
          "concurrency or queueing"),
    _rule("SQL / NoSQL datastores", 5,
          r"postgresql|postgres|\bmysql\b|mongodb|\bredis\b|\bsqlite\b|database",
          "persistent storage"),
    _rule("API & backend engineering signals", 5,
          r"rest apis?|restful|authentication|jwt|oauth|middleware|dependency injection|pydantic|sqlalchemy|alembic",
          "backend craft"),
    _rule("Typed / tested Python codebase", 4,
          r"pytest|mypy|typing|type hints|unittest|coverage", "quality practices"),
]

CLOUD_RULES: list[Rule] = [
    _rule("Cloud platform (GCP / AWS / Azure)", 6,
          r"\bgcp\b|google cloud|amazon web services|\baws\b|\bazure\b|compute engine|cloud run|lambda\b",
          "hosted infrastructure"),
    _rule("Containerisation & deployment", 5,
          r"docker|dockerfile|kubernetes|\bk8s\b|terraform|cloud run|render\.com|vercel|netlify|deploy(ed|ment)?\b",
          "shipped artefacts"),
    _rule("CI/CD automation", 4,
          r"ci/cd|cicd|github actions|gitlab ci|jenkins|workflow (runs|yaml)", "repeatable releases"),
    _rule("End-to-end full stack system", 4,
          r"frontend|react|next\.?js|full[- ]stack|end[- ]to[- ]end", "whole product surface"),
    _rule("Frontend framework used with a backend", 2,
          r"react|next\.?js|vue|angular|typescript", "UI layer"),
]

ENGINEERING_RULES: list[Rule] = [
    _rule("Automated testing", 1.5,
          r"pytest|unit test\w*|integration test\w*|test coverage|jest|junit|tdd|mock(ing|s)?\b",
          "regression safety"),
    _rule("Caching / queues / background jobs", 1,
          r"\bredis\b|celery|rabbitmq|kafka|queue|cache[sd]? (hit|layer|strategy)", "throughput design"),
    _rule("Observability & failure handling", 1,
          r"prometheus|grafana|opentelemetry|structured logging|monitoring|circuit breaker|retry (logic|mechanism)|graceful (degradation|failure)|health checks?",
          "operability"),
    _rule("Architecture & scalability", 1,
          r"microservices?|event[- ]driven|design patterns?|solid\b|scalab\w+|load balanc\w+|rate limit\w+|caching strategy",
          "system design"),
    _rule("Concurrency", 0.5,
          r"\basyncio\b|async/await|concurrent|threading|multiprocessing|parallelis\w+",
          "parallel execution"),
]

# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
@dataclass
class Corpora:
    """Text split by how much it proves implementation ownership."""

    rich: str      # projects + internships + summary (strongest evidence)
    skills: str    # skills section (weakest evidence)
    full: str      # whole resume
    project_count: int = 0
    ai_project_count: int = 0


def build_corpora(candidate: Candidate) -> Corpora:
    rich_parts = [project.raw for project in candidate.projects]
    rich_parts += [exp.raw for exp in candidate.experiences]
    if candidate.summary:
        rich_parts.append(candidate.summary)
    rich = "\n".join(part for part in rich_parts if part)
    if not rich.strip():
        rich = candidate.full_text
    skills = candidate.sections.get("skills", "")
    return Corpora(
        rich=rich,
        skills=skills,
        full=candidate.full_text,
        project_count=len(candidate.projects),
        ai_project_count=_count_ai_projects(candidate),
    )


def _count_ai_projects(candidate: Candidate) -> int:
    count = sum(1 for project in candidate.projects if AI_MARKER_RE.search(project.text))
    if count == 0 and any(AI_MARKER_RE.search(exp.text) for exp in candidate.experiences):
        count = 1
    return count


def _evaluate(rule: Rule, corpora: Corpora) -> tuple[float, str]:
    """Return (points, evidence detail) for one rule.

    Implementation-context matches score in full; a match that only exists in
    the skills list scores half, because a keyword is not proof of ownership.
    """
    rich_match = rule.pattern.search(corpora.rich)
    if rich_match:
        snippet = " ".join(rich_match.group(0).split())
        return rule.points, f"in projects/experience: {snippet}"

    full_match = rule.pattern.search(corpora.full)
    if full_match:
        if corpora.skills and rule.pattern.search(corpora.skills):
            return rule.points * 0.5, "skills list only"
        snippet = " ".join(full_match.group(0).split())
        return rule.points * 0.5, f"mentioned elsewhere: {snippet}"
    return 0.0, ""


def _score_rules(
    rules: list[Rule], corpora: Corpora, cap: float
) -> tuple[float, list[ScoreEvidence], list[str]]:
    evidence: list[ScoreEvidence] = []
    matched_labels: list[str] = []
    total = 0.0
    for rule in rules:
        points, detail = _evaluate(rule, corpora)
        if points <= 0:
            continue
        total += points
        evidence.append(ScoreEvidence(label=rule.label, points=round(points, 1), detail=detail))
        matched_labels.append(rule.label)
    return clamp(total, 0, cap), evidence, matched_labels


def _score_ai(candidate: Candidate, corpora: Corpora) -> tuple[float, list[ScoreEvidence], list[str]]:
    points, evidence, labels = _score_rules(
        AI_RULES, corpora, cap=SCORE_WEIGHTS["ai_project_depth"]
    )

    # Aggregate signal: more than one AI project beats a single demo.
    if corpora.ai_project_count >= 2:
        bonus = 5.0
        evidence.append(
            ScoreEvidence(
                label="Multiple distinct AI projects",
                points=bonus,
                detail=f"{corpora.ai_project_count} AI-related projects detected",
            )
        )
        labels.append("Multiple distinct AI projects")
        points += bonus

    # AI work inside real employment history beats hobby code.
    experience_text = "\n".join(exp.raw for exp in candidate.experiences)
    if experience_text and AI_MARKER_RE.search(experience_text):
        bonus = 5.0
        evidence.append(
            ScoreEvidence(
                label="AI used in professional experience",
                points=bonus,
                detail="AI/ML work described in an internship or job",
            )
        )
        labels.append("AI used in professional experience")
        points += bonus

    return round(clamp(points, 0, SCORE_WEIGHTS["ai_project_depth"]), 1), evidence, labels


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------
_TUTORIAL_RE = re.compile(
    r"tutorial|youtube (video|course)|clone of|to-?do app|portfolio template|"
    r"copied from|followed along|step[- ]by[- ]step guide",
    re.I,
)
_THIN_WRAPPER_RE = re.compile(
    r"chatbot|chat bot|ask (the |a )?(pdf|document|resume)|simple (api )?wrapper|"
    r"api call to (openai|gpt|claude)|playground",
    re.I,
)
_RICH_AI_RE = re.compile(
    r"\brag\b|retrieval|embedding|vector|agent|tool[- ]calling|state|workflow|"
    r"evaluat|pipeline|fastapi|django|flask|celery|database|postgres|redis|cache|"
    r"chunk|rerank|ingest|memory|orchestrat",
    re.I,
)


@dataclass
class PenaltyResult:
    points: float = 0.0
    reasons: list[str] = field(default_factory=list)


def project_quality_penalties(candidate: Candidate, corpora: Corpora) -> PenaltyResult:
    """Deduct 5-15 points for shallow or tutorial-style AI projects."""
    result = PenaltyResult()
    if not corpora.ai_project_count:
        return result

    text = corpora.rich

    if _TUTORIAL_RE.search(text):
        result.points -= 5
        result.reasons.append("Tutorial-style project wording without implementation detail")

    thin = _THIN_WRAPPER_RE.search(text)
    rich = _RICH_AI_RE.search(text)
    if thin and not rich:
        result.points -= SHALLOW_PROJECT_PENALTY
        result.reasons.append(
            "AI project looks like a thin LLM/API wrapper without retrieval, state, "
            "backend or evaluation"
        )

    described = [p for p in candidate.projects if p.bullets or len(p.summary) > 60]
    if candidate.projects and not described:
        result.points -= 5
        result.reasons.append("Projects listed without implementation detail or ownership")

    # Never let penalties push below zero and never exceed the documented cap.
    result.points = -clamp(abs(result.points), 0, MAX_PROJECT_PENALTY)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class ScoreResult:
    breakdown: ScoreBreakdown
    evidence: list[ScoreEvidence] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    project_summary: str = ""

    def total(self) -> float:
        return round(clamp(self.breakdown.total(), 0, 100), 1)

    def evidence_by_category(self) -> dict[str, list[dict[str, object]]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for item in self.evidence:
            grouped.setdefault(item.category, []).append(
                {"label": item.label, "points": item.points, "detail": item.detail}
            )
        return grouped


def _truncate(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "..."


def _default_project_summary(candidate: Candidate) -> str:
    """Pick the most AI-heavy project and summarise it in one sentence."""
    best: Project | None = None
    best_hits = 0
    for project in candidate.projects:
        hits = len(AI_MARKER_RE.findall(project.text))
        if hits > best_hits:
            best, best_hits = project, hits
    if best is not None:
        body = best.summary or " ".join(best.bullets) or "AI project"
        return _truncate(f"{best.name}: {body}")
    if candidate.summary:
        return _truncate(candidate.summary)
    return "No project description could be summarised."


def _strengths(
    ai_labels: list[str],
    py_labels: list[str],
    cloud_labels: list[str],
    eng_labels: list[str],
    github: GitHubEnrichment,
    breakdown: ScoreBreakdown,
) -> list[str]:
    strengths = [label for label in ai_labels[:3]]
    if breakdown.python_backend >= 18:
        strengths.append("Solid Python & backend evidence")
    elif py_labels:
        strengths.append("Python used in real project work")
    if breakdown.cloud_fullstack >= 8:
        strengths.append("Cloud / deployment experience")
    if breakdown.engineering_depth >= 3:
        strengths.append("Engineering depth signals (testing, caching, observability)")
    if github.status == "ok" and github.total_score >= 6:
        strengths.append("Active, maintained GitHub presence")
    return dedupe_preserve_order(strengths)[:6]


def _concerns(
    candidate: Candidate,
    corpora: Corpora,
    github: GitHubEnrichment,
    breakdown: ScoreBreakdown,
    penalties: PenaltyResult,
) -> list[str]:
    concerns: list[str] = list(penalties.reasons)
    if corpora.ai_project_count == 0:
        concerns.append("No clear AI project implementation found")
    if breakdown.cloud_fullstack < 6:
        concerns.append("Limited cloud or deployment evidence")
    if breakdown.engineering_depth < 2:
        concerns.append("Little evidence of testing or observability practices")
    if github.status in {"no_profile", "not_found", "error", "rate_limited"}:
        concerns.append(f"GitHub enrichment unavailable ({github.status})")
    if not candidate.projects:
        concerns.append("No projects section detected in the resume")
    if breakdown.python_backend >= 22 and breakdown.ai_project_depth <= 15:
        concerns.append("Strong Python profile but shallow AI implementation depth")
    return dedupe_preserve_order(concerns)[:6]


def score_candidate(
    candidate: Candidate,
    eligibility: EligibilityResult,
    github: GitHubEnrichment | None = None,
    *,
    llm_adjustment: float = 0.0,
    project_summary_override: str | None = None,
) -> ScoreResult:
    """Score one eligible candidate across the five weighted categories."""
    corpora = build_corpora(candidate)
    github = github or GitHubEnrichment()

    ai_points, ai_evidence, ai_labels = _score_ai(candidate, corpora)
    ai_points = clamp(ai_points + llm_adjustment, 0, SCORE_WEIGHTS["ai_project_depth"])
    if llm_adjustment:
        ai_evidence.append(
            ScoreEvidence(
                label="LLM project-quality adjustment",
                points=round(llm_adjustment, 1),
                detail="structured model assessment",
                category="ai_project_depth",
            )
        )

    py_points, py_evidence, py_labels = _score_rules(
        PYTHON_RULES, corpora, cap=SCORE_WEIGHTS["python_backend"]
    )
    cloud_points, cloud_evidence, cloud_labels = _score_rules(
        CLOUD_RULES, corpora, cap=SCORE_WEIGHTS["cloud_fullstack"]
    )
    eng_points, eng_evidence, eng_labels = _score_rules(
        ENGINEERING_RULES, corpora, cap=SCORE_WEIGHTS["engineering_depth"]
    )

    github_points = clamp(
        github.activity_score + github.repository_score,
        0,
        SCORE_WEIGHTS["github"],
    )

    penalties = project_quality_penalties(candidate, corpora)

    breakdown = ScoreBreakdown(
        ai_project_depth=round(ai_points, 1),
        python_backend=round(py_points, 1),
        cloud_fullstack=round(cloud_points, 1),
        github=round(github_points, 1),
        engineering_depth=round(eng_points, 1),
        penalties=round(penalties.points, 1),
    )

    def tagged(category: str, items: list[ScoreEvidence]) -> list[ScoreEvidence]:
        for item in items:
            item.category = category
        return items

    evidence: list[ScoreEvidence] = []
    evidence += tagged("ai_project_depth", ai_evidence)
    evidence += tagged("python_backend", py_evidence)
    evidence += tagged("cloud_fullstack", cloud_evidence)
    evidence += tagged(
        "github",
        [
            ScoreEvidence(
                label="GitHub activity",
                points=round(github_points, 1),
                detail=github.summary or github.status,
            )
        ]
        if github_points
        else [],
    )
    evidence += tagged("engineering_depth", eng_evidence)
    evidence += tagged("penalties", penalties_reasons_to_evidence(penalties))

    return ScoreResult(
        breakdown=breakdown,
        evidence=evidence,
        strengths=_strengths(ai_labels, py_labels, cloud_labels, eng_labels, github, breakdown),
        concerns=_concerns(candidate, corpora, github, breakdown, penalties),
        project_summary=project_summary_override or _default_project_summary(candidate),
    )


def penalties_reasons_to_evidence(penalty: PenaltyResult) -> list[ScoreEvidence]:
    if not penalty.reasons:
        return []
    share = round(penalty.points / len(penalty.reasons), 1)
    return [
        ScoreEvidence(label="Project quality penalty", points=share, detail=reason)
        for reason in penalty.reasons
    ]
