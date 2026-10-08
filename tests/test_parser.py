from pathlib import Path

from resume_screening.parser import (
    discover_resume_files,
    ingest_directory,
    parse_resume,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_discover_finds_supported_files_sorted(tmp_path: Path) -> None:
    _write(tmp_path / "b.pdf", "x")
    _write(tmp_path / "A.txt", "y")
    _write(tmp_path / "notes.md", "z")
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "sub" / "nested.pdf", "n")

    found = discover_resume_files(tmp_path)
    assert [p.name for p in found] == ["A.txt", "b.pdf"]
    assert all(p.suffix in {".pdf", ".txt", ".docx"} for p in found)


def test_discover_missing_directory_raises(tmp_path: Path) -> None:
    try:
        discover_resume_files(tmp_path / "nope")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_parse_txt_normalises_whitespace(tmp_path: Path) -> None:
    path = _write(tmp_path / "resume.txt", "John   Doe\n\n\n\nPython  Developer")
    parsed = parse_resume(path)
    assert parsed.status == "parsed"
    assert parsed.text == "John Doe\n\nPython Developer"
    assert parsed.error is None


def test_parse_corrupt_pdf_reports_failure_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf body")

    parsed = parse_resume(path)
    assert parsed.status == "failed"
    assert parsed.text == ""
    assert parsed.error


def test_parse_blank_pdf_reports_empty(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with path.open("wb") as handle:
        writer.write(handle)

    parsed = parse_resume(path)
    assert parsed.status == "empty"
    assert parsed.error


def test_ingestion_continues_past_bad_files_and_deduplicates(tmp_path: Path) -> None:
    _write(tmp_path / "good.txt", "Ada Lovelace\nPython and LangChain")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 broken")
    _write(tmp_path / "good_copy.txt", "Ada Lovelace\nPython and LangChain")

    result = ingest_directory(tmp_path)

    assert result.total_files == 3
    assert {r.filename for r in result.parsed} == {"good.txt"}
    assert {r.filename for r in result.failed} == {"bad.pdf"}
    dup = result.duplicates[0]
    assert dup.filename == "good_copy.txt"
    assert dup.duplicate_of == "good.txt"
    assert result.counts() == {"total_files": 3, "parsed": 1, "failed": 1, "duplicates": 1}
