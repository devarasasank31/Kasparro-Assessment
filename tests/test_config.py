from pathlib import Path

from resume_screening import __version__
from resume_screening.config import SCORE_WEIGHTS, TOTAL_SCORE, Settings


def test_version_is_semver_like() -> None:
    assert __version__.count(".") == 2


def test_score_weights_total_100() -> None:
    assert TOTAL_SCORE == 100
    assert sum(SCORE_WEIGHTS.values()) == 100
    assert SCORE_WEIGHTS == {
        "ai_project_depth": 40,
        "python_backend": 30,
        "cloud_fullstack": 15,
        "github": 10,
        "engineering_depth": 5,
    }


def test_settings_from_args_toggles(tmp_path: Path) -> None:
    settings = Settings.from_args(
        input_dir=tmp_path / "in",
        output_path=tmp_path / "out.json",
        no_llm=True,
        no_github=True,
        verbose=True,
    )
    assert settings.input_dir == tmp_path / "in"
    assert settings.output_path == tmp_path / "out.json"
    assert settings.use_llm is False
    assert settings.use_github is False
    assert settings.verbose is True


def test_llm_enabled_requires_api_key() -> None:
    assert Settings(llm_api_key="").llm_enabled is False
    assert Settings(llm_api_key="sk-test").llm_enabled is True
    assert Settings(llm_api_key="sk-test", use_llm=False).llm_enabled is False


def test_keyless_providers_run_without_a_key() -> None:
    assert Settings(llm_provider="echo").llm_enabled is True
    assert Settings(llm_provider="ollama").llm_enabled is True
    assert Settings(llm_provider="echo", use_llm=False).llm_enabled is False
    assert Settings(llm_provider="openai", llm_api_key="").llm_enabled is False
