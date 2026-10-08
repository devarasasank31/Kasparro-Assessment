import json
from pathlib import Path

import pytest

from resume_screening.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "resumes"


def _run(tmp_path: Path, *extra: str) -> int:
    output = tmp_path / "results.json"
    return main(
        [
            "--input", str(FIXTURES),
            "--output", str(output),
            "--no-github",
            "--no-cache",
            *extra,
        ]
    )


def test_cli_writes_results_and_exits_zero(tmp_path: Path) -> None:
    assert _run(tmp_path) == 0
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["summary"]["eligible"] == 2
    assert payload["candidates"][0]["rank"] == 1


def test_cli_prints_a_report_on_request(tmp_path: Path, capsys) -> None:
    assert _run(tmp_path, "--report", "--top", "1") == 0
    out = capsys.readouterr().out
    assert "AI RESUME SCREENING - RUN REPORT" in out
    assert "Asha Rao" in out
    assert "Rejected (2)" in out


def test_cli_limit_restricts_the_batch(tmp_path: Path) -> None:
    assert _run(tmp_path, "--limit", "2") == 0
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["summary"]["total_resumes"] == 2


def test_cli_missing_input_directory_exits_with_code_2(tmp_path: Path) -> None:
    code = main(
        ["--input", str(tmp_path / "nowhere"), "--output", str(tmp_path / "r.json")]
    )
    assert code == 2


def test_cli_rejects_unknown_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--nonsense"])
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "main.py" in capsys.readouterr().out
