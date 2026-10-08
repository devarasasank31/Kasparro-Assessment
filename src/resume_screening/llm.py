"""Provider-independent LLM adapter with strict structured output.

Design rules enforced here:

- credentials only ever come from environment variables (via ``Settings``)
- the model returns JSON that is validated against :class:`LLMAnalysis`
- the model is **never** consulted for hard eligibility
- every failure raises :class:`LLMError`, which the pipeline records and
  survives - one bad response must never abort a batch run
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

import requests
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from .config import Settings
from .utils import get_logger

log = get_logger("llm")

#: Resume text handed to the model is truncated to keep prompts bounded.
PROMPT_MAX_CHARS = 7000
DEFAULT_TEMPERATURE = 0.0


class LLMError(RuntimeError):
    """Raised for any transport, protocol or schema failure."""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class LLMProjectSummary(BaseModel):
    """One project as judged by the model."""

    name: str
    summary: str = ""
    is_ai_project: bool = False
    quality: Literal["shallow", "solid", "strong"] = "solid"


class LLMAnalysis(BaseModel):
    """Strict schema for a resume assessment."""

    project_summaries: list[LLMProjectSummary] = Field(default_factory=list)
    #: Model's own estimate of AI/agentic/RAG depth (0-40).
    ai_project_depth: int = Field(default=0, ge=0, le=40)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    shallow_project: bool = False
    rationale: str = ""

    #: Set by :meth:`set_deterministic_depth`; never serialised.
    _adjustment: float = PrivateAttr(default=0.0)

    @property
    def adjustment(self) -> float:
        """Bounded delta between model judgement and the deterministic score.

        The deterministic engine stays in charge; the model only nudges it.
        """
        return self._adjustment

    def set_deterministic_depth(self, value: float) -> float:
        delta = (self.ai_project_depth - value) * 0.5
        self._adjustment = max(-6.0, min(6.0, round(delta, 1)))
        return self._adjustment


SYSTEM_PROMPT = """\
You are a senior engineer reviewing one resume for an SDE internship that needs
strong Python fundamentals plus practical AI/agentic/RAG experience.

You will receive the resume text. Respond with ONE JSON object only - no prose,
no markdown fences - using exactly this schema:

{
  "project_summaries": [
    {"name": "...", "summary": "...", "is_ai_project": true, "quality": "strong|solid|shallow"}
  ],
  "ai_project_depth": <integer 0-40>,
  "strengths": ["..."],
  "concerns": ["..."],
  "evidence": ["short verbatim quotes supporting your judgement"],
  "shallow_project": <true|false>,
  "rationale": "one or two sentences"
}

Rules:
- ai_project_depth measures real AI implementation depth: agents, retrieval,
  RAG, tool calling, evaluation, state, orchestration, data processing.
  A thin wrapper around an LLM API with no workflow, retrieval, state or
  evaluation must score low (0-15).
- Do NOT decide eligibility. Python/AI hard filtering happens elsewhere.
- Be evidence based: quote or closely paraphrase what the resume actually says.
- If the resume text is truncated, judge only what is present.
"""


class LLMClient(Protocol):
    """Minimal surface the rest of the pipeline depends on."""

    available: bool

    def analyse(self, resume_text: str) -> LLMAnalysis:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class HTTPChatClient:
    """Chat-completions style client covering OpenAI-compatible APIs, Anthropic
    and Ollama behind one small adapter."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        base_url: str = "",
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.provider = provider.strip().lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # -- availability ---------------------------------------------------
    @property
    def available(self) -> bool:
        if self.provider == "echo":
            return True
        if self.provider == "ollama":
            return True
        return bool(self.api_key)

    # -- public API -----------------------------------------------------
    def analyse(self, resume_text: str) -> LLMAnalysis:
        raw = self._complete(resume_text)
        return parse_llm_response(raw)

    # -- internals ------------------------------------------------------
    def _complete(self, resume_text: str) -> str:
        if not self.available:
            raise LLMError(f"LLM provider '{self.provider}' is not configured (missing API key)")
        user_prompt = f"Resume:\n{resume_text[:PROMPT_MAX_CHARS]}"
        if self.provider == "echo":
            return json.dumps(_echo_payload(resume_text))
        try:
            if self.provider == "anthropic":
                return self._call_anthropic(user_prompt)
            if self.provider == "ollama":
                return self._call_ollama(user_prompt)
            return self._call_openai_compatible(user_prompt)
        except LLMError:
            raise
        except requests.RequestException as exc:
            raise LLMError(f"network error: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        if self.provider == "ollama":
            return {"content-type": "application/json"}
        return {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def _call_openai_compatible(self, user_prompt: str) -> str:
        url = self.base_url or "https://api.openai.com/v1"
        payload = {
            "model": self.model,
            "temperature": DEFAULT_TEMPERATURE,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        data = self._post(f"{url}/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc

    def _call_anthropic(self, user_prompt: str) -> str:
        url = self.base_url or "https://api.anthropic.com/v1"
        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": DEFAULT_TEMPERATURE,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = self._post(f"{url}/messages", payload)
        try:
            return "".join(block.get("text", "") for block in data["content"])
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc

    def _call_ollama(self, user_prompt: str) -> str:
        url = self.base_url or "http://localhost:11434"
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self._post(f"{url}/api/chat", payload)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                url, headers=self._headers(), json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"request failed: {exc}") from exc

        if response.status_code in {401, 403}:
            raise LLMError(f"authentication rejected (HTTP {response.status_code})")
        if response.status_code == 429:
            raise LLMError("rate limited (HTTP 429)")
        if response.status_code >= 400:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError("response was not valid JSON") from exc


def build_client(settings: Settings) -> HTTPChatClient | None:
    """Create a client for the configured provider, or ``None`` when unusable."""
    if not settings.use_llm:
        return None
    client = HTTPChatClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )
    if not client.available:
        log.info("LLM disabled: no credentials for provider '%s'", settings.llm_provider)
        return None
    return client


# ---------------------------------------------------------------------------
# Response parsing / validation
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.M)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    if not cleaned:
        raise LLMError("empty model response")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("no JSON object found in model response")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("model response was not a JSON object")
    return parsed


def parse_llm_response(raw: str) -> LLMAnalysis:
    """Validate a raw model response against :class:`LLMAnalysis`."""
    payload = extract_json(raw)
    try:
        return LLMAnalysis.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"structured output failed validation: {exc.errors()[:3]}") from exc


def _echo_payload(resume_text: str) -> dict[str, Any]:
    """Deterministic offline payload used by tests and dry runs."""
    first_line = next((line.strip() for line in resume_text.splitlines() if line.strip()), "resume")
    return {
        "project_summaries": [
            {
                "name": "Primary project",
                "summary": f"Offline placeholder derived from: {first_line[:80]}",
                "is_ai_project": True,
                "quality": "solid",
            }
        ],
        "ai_project_depth": 20,
        "strengths": ["offline echo provider"],
        "concerns": ["echo provider does not inspect the resume"],
        "evidence": ["echo"],
        "shallow_project": False,
        "rationale": "Offline deterministic provider for tests and dry runs.",
    }
