import json

import pytest

from resume_screening.config import Settings
from resume_screening.llm import (
    HTTPChatClient,
    LLMAnalysis,
    LLMError,
    build_client,
    extract_json,
    parse_llm_response,
)

VALID = json.dumps(
    {
        "project_summaries": [
            {
                "name": "Support Agent",
                "summary": "Multi-agent RAG workflow with tool calling.",
                "is_ai_project": True,
                "quality": "strong",
            }
        ],
        "ai_project_depth": 34,
        "strengths": ["Agents with evaluation"],
        "concerns": ["No Redis evidence"],
        "evidence": ["Built a multi-agent RAG workflow"],
        "shallow_project": False,
        "rationale": "Real retrieval and orchestration.",
    }
)


def test_parse_valid_structured_output() -> None:
    analysis = parse_llm_response(VALID)
    assert analysis.ai_project_depth == 34
    assert analysis.project_summaries[0].quality == "strong"
    assert analysis.shallow_project is False
    assert analysis.strengths == ["Agents with evaluation"]


def test_parse_output_wrapped_in_markdown_fences() -> None:
    analysis = parse_llm_response(f"```json\n{VALID}\n```")
    assert analysis.ai_project_depth == 34


def test_parse_output_with_surrounding_prose() -> None:
    analysis = parse_llm_response(f"Here is my assessment:\n{VALID}\nHope that helps!")
    assert analysis.rationale.startswith("Real retrieval")


def test_out_of_range_depth_is_rejected() -> None:
    payload = json.loads(VALID)
    payload["ai_project_depth"] = 90
    with pytest.raises(LLMError):
        parse_llm_response(json.dumps(payload))


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(LLMError):
        parse_llm_response("definitely not json")


def test_empty_response_is_rejected() -> None:
    with pytest.raises(LLMError):
        parse_llm_response("   ")


def test_extract_json_handles_nested_objects() -> None:
    payload = extract_json('noise {"a": {"b": 1}} tail')
    assert payload == {"a": {"b": 1}}


def test_adjustment_is_bounded_and_damped() -> None:
    analysis = LLMAnalysis(ai_project_depth=40)
    assert analysis.set_deterministic_depth(20) == 6.0
    assert analysis.set_deterministic_depth(40) == 0.0
    assert analysis.adjustment == 0.0
    assert LLMAnalysis(ai_project_depth=0).set_deterministic_depth(40) == -6.0


def test_client_without_credentials_is_unavailable() -> None:
    client = HTTPChatClient(provider="openai", model="gpt-4o-mini", api_key="")
    assert client.available is False
    with pytest.raises(LLMError):
        client.analyse("some resume")


def test_build_client_returns_none_when_llm_disabled() -> None:
    assert build_client(Settings(use_llm=False, llm_api_key="sk-test")) is None
    assert build_client(Settings(use_llm=True, llm_api_key="")) is None
    assert build_client(Settings(use_llm=True, llm_api_key="sk-test")) is not None


def test_echo_provider_produces_valid_analysis() -> None:
    client = HTTPChatClient(provider="echo", model="none")
    assert client.available is True
    analysis = client.analyse("Ada Lovelace\nPython and LangGraph")
    assert 0 <= analysis.ai_project_depth <= 40
    assert analysis.rationale


def test_failed_http_call_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    client = HTTPChatClient(provider="openai", model="gpt-4o-mini", api_key="sk-test")

    def _req_boom(*args, **kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(client.session, "post", _req_boom)
    with pytest.raises(LLMError):
        client.analyse("resume text")


def test_server_error_status_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 500
        text = "boom"

        def json(self):  # pragma: no cover - never reached
            return {}

    client = HTTPChatClient(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    monkeypatch.setattr(client.session, "post", lambda *a, **k: _Response())
    with pytest.raises(LLMError) as excinfo:
        client.analyse("resume text")
    assert "500" in str(excinfo.value)
