"""Deterministic hard eligibility filter.

A candidate is eligible only when BOTH hold:

A. Python evidence  - Python appears as a genuine skill / project / work
   technology / implementation language.
B. AI evidence      - at least one meaningful AI, LLM, RAG or agentic
   implementation (framework, project or custom build).

The rules are intentionally rule-based and live entirely outside the LLM so the
same resume always produces the same decision, with auditable evidence.
"""

from __future__ import annotations

import re

from .config import MIN_AI_EVIDENCE_HITS
from .extractor import _is_heading
from .models import Candidate, EligibilityResult
from .skills import scan_skills
from .utils import dedupe_preserve_order

# ---------------------------------------------------------------------------
# Python detection
# ---------------------------------------------------------------------------
PYTHON_RE = re.compile(r"(?<![a-z0-9_])python\s*[\d.]*(?![a-z0-9])", re.I)

# ---------------------------------------------------------------------------
# AI evidence terms
# ---------------------------------------------------------------------------
# (label, tier, regex)
#   strong   -> the term itself is proof of a meaningful AI implementation
#   moderate -> needs project/experience context or an implementation verb
#   weak     -> needs an implementation verb nearby (e.g. "Built an AI chatbot")
AI_TERMS: list[tuple[str, str, str]] = [
    # --- frameworks / architectures ------------------------------------
    ("LangChain", "strong", r"lang\s*chain"),
    ("LangGraph", "strong", r"lang\s*graph"),
    ("LlamaIndex", "strong", r"llama\s*index|llamaindex"),
    ("Google ADK", "strong", r"google adk|agent development kit|\badk\b"),
    ("AutoGen", "strong", r"\bautogen\b"),
    ("CrewAI", "strong", r"crew\s*ai"),
    ("Haystack", "strong", r"\bhaystack\b"),
    ("RAG pipeline", "strong", r"\brag\b|retrieval[- ]augmented"),
    ("embeddings", "strong", r"\bembeddings?\b|text-embedding"),
    ("vector search", "strong", r"vector (search|database|db|store)|similarity search|semantic search|cosine similarity|vector index"),
    ("agentic system", "strong", r"\bagentic\b|multi[- ]agent|tool[- ]calling|tool use|tool-calling|state machine|agent (workflow|orchestration|framework|system|graph)"),
    ("AI agent", "strong", r"\bai[- ]?agents?\b|\bagents?\b(?= *(?:built|system|framework|workflow|pipeline))"),
    ("LLM", "strong", r"\bllms?\b|large language model|language model"),
    ("LLM provider", "moderate", r"openai|chatgpt|gpt-[0-9]|\bgpt4\b|anthropic|claude|gemini api|google gemini|hugging\s?face|groq|mistral|deepseek|llama\s?[23]"),
    ("prompt engineering", "strong", r"prompt (engineering|design|optimisation|optimization)"),
    ("fine-tuning", "strong", r"fine[- ]tun|adapter tuning|\blora\b|\bpeft\b"),
    ("evaluation pipeline", "strong", r"(evaluation|eval) (pipeline|harness|framework)|ragas|llm (evaluation|judge)|human[- ]in[- ]the[- ]loop"),
    ("multi-agent workflow", "strong", r"multi[- ]agent|swarm|orchestrat\w+ agents"),
    ("voice/speech AI", "strong", r"speech recognition|text-to-speech|\btts\b|\bstt\b|whisper model"),
    # --- classical / applied AI (context required) ----------------------
    ("deep learning", "moderate", r"deep learning|neural networks?|\bcnn\b|\brnn\b|\blstm\b|convolutional"),
    ("machine learning", "moderate", r"machine learning|supervised learning|unsupervised learning|scikit[- ]?learn|\bsklearn\b|\bxgboost\b|\blightgbm\b|random forest|regression model|classification model"),
    ("NLP", "moderate", r"\bnlp\b|natural language processing|sentiment analys|text classif|named entity"),
    ("computer vision", "moderate", r"computer vision|\bopencv\b|object detection|image segmentation|image classif|yolo"),
    ("ML framework", "moderate", r"\bpytorch\b|\btensorflow\b|\bkeras\b"),
    # --- weak signals (need an implementation verb nearby) --------------
    ("AI feature", "weak", r"ai[- ]powered|ai[- ]driven|generative ai|\bgenai\b|artificial intelligence|\bai/ml\b|\bai models?\b|\bai assistant\b|\bai system\b|\bai chatbot\b|\bchatbot\b"),
]

_COMPILED_AI_TERMS = [(label, tier, re.compile(pattern, re.I)) for label, tier, pattern in AI_TERMS]

#: Verbs that indicate the candidate actually implemented something.
IMPLEMENTATION_RE = re.compile(
    r"\b(built|build|building|develop\w*|implement\w*|design\w*|train\w*|deploy\w*|"
    r"integrat\w*|creat\w*|architect\w*|engineered|productionis\w*|productioniz\w*|"
    r"prototype\w*|experiment\w*|worked on|hands[- ]on|fine[- ]tuned|tuned)\b",
    re.I,
)

#: Sections where merely *mentioning* a technology is not evidence of a build.
_NON_EVIDENCE_SECTIONS = {"skills", "certifications", "achievements", "strengths", "education"}

_CONTEXT_WINDOW = 180


def _snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    """Return the line containing ``[start, end)`` plus a little context."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = " ".join(text[line_start:line_end].split())
    if len(line) > radius * 2:
        local_start = max(0, start - line_start - radius)
        line = ("..." if local_start else "") + line[local_start : local_start + radius * 2] + "..."
    return line.strip()


def find_python_evidence(candidate: Candidate) -> list[str]:
    """Short snippets proving Python appears in the resume."""
    text = candidate.full_text
    if not text:
        return []
    evidence: list[str] = []
    for match in PYTHON_RE.finditer(text):
        evidence.append(_snippet(text, match.start(), match.end()))
        if len(evidence) >= 3:
            break
    if not evidence:
        # pdf extraction occasionally glues words together; "python" is long
        # enough that a space-stripped lookup is still unambiguous.
        compact = re.sub(r"\s+", "", text.lower())
        if "python" in compact:
            idx = compact.index("python")
            evidence.append("..." + compact[max(0, idx - 40) : idx + 60] + "...")
    return dedupe_preserve_order(evidence)


def _section_spans(text: str) -> list[tuple[int, int, str]]:
    """Map character ranges of ``text`` to section names (heading excluded)."""
    spans: list[tuple[int, int, str]] = []
    current = "header"
    start = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        key = _is_heading(line.rstrip("\n"))
        if key:
            spans.append((start, offset, current))
            current = key
            start = offset + len(line)
        offset += len(line)
    spans.append((start, offset, current))
    return spans


def _section_at(spans: list[tuple[int, int, str]], position: int) -> str | None:
    for start, end, key in spans:
        if start <= position <= end:
            return key
    return None


def find_ai_evidence(candidate: Candidate) -> list[tuple[str, str]]:
    """Return ``(label, snippet)`` pairs for meaningful AI evidence.

    Strong terms always count. Moderate terms need project/experience context
    or an implementation verb. Weak terms ("AI-powered ...") need an
    implementation verb close by, which stops a bare skills-list mention from
    passing the filter on its own.
    """
    text = candidate.full_text
    if not text:
        return []

    spans = _section_spans(text)
    hits: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for label, tier, pattern in _COMPILED_AI_TERMS:
        if label in seen_labels:
            continue
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - _CONTEXT_WINDOW) : match.end() + _CONTEXT_WINDOW]
            has_implementation_context = bool(IMPLEMENTATION_RE.search(window))
            section = _section_at(spans, match.start())
            in_project_context = section in {"projects", "experience"}

            if tier == "strong":
                ok = True
            elif tier == "moderate":
                ok = in_project_context or (
                    has_implementation_context and section not in _NON_EVIDENCE_SECTIONS
                )
            else:
                ok = has_implementation_context and section not in _NON_EVIDENCE_SECTIONS

            if not ok:
                continue
            seen_labels.add(label)
            hits.append((label, _snippet(text, match.start(), match.end())))
            break
        if len(hits) >= 8:
            break
    return hits


def check_eligibility(candidate: Candidate) -> EligibilityResult:
    """Apply the deterministic Python + AI hard filter."""
    python_evidence = find_python_evidence(candidate)
    ai_hits = find_ai_evidence(candidate)

    rejection_reasons: list[str] = []
    if not python_evidence:
        rejection_reasons.append("No evidence of Python stack")
    if len(ai_hits) < MIN_AI_EVIDENCE_HITS:
        rejection_reasons.append("No AI/agentic project evidence")

    matched_skills = scan_skills(candidate.full_text)
    ai_evidence = [f"{label}: {snippet}" for label, snippet in ai_hits]

    return EligibilityResult(
        eligible=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        matched_skills=matched_skills[:25],
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
    )
