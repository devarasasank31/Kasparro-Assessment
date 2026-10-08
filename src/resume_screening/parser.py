"""Resume ingestion and text extraction.

Responsibilities:
- discover resume files in a directory (PDF required, TXT/DOCX bonus)
- deduplicate identical files by content hash
- extract and normalise text without ever letting one bad file kill the batch
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal
from xml.etree import ElementTree

from .utils import get_logger, normalise_text

log = get_logger("parser")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

#: Safety valve for absurdly long documents.
MAX_PAGES = 15
MAX_TEXT_CHARS = 60_000

ParseStatus = Literal["parsed", "failed", "empty", "duplicate", "unsupported"]


@dataclass
class ParsedResume:
    """Result of reading a single resume file."""

    path: Path
    filename: str
    file_hash: str = ""
    text: str = ""
    page_count: int = 0
    status: ParseStatus = "parsed"
    error: str | None = None
    duplicate_of: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "parsed" and bool(self.text)

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.filename,
            "status": self.status,
            "page_count": self.page_count,
            "char_count": len(self.text),
            **({"error": self.error} if self.error else {}),
            **({"duplicate_of": self.duplicate_of} if self.duplicate_of else {}),
        }


@dataclass
class IngestionResult:
    """Aggregate outcome of scanning an input directory."""

    resumes: list[ParsedResume] = field(default_factory=list)
    total_files: int = 0

    @property
    def parsed(self) -> list[ParsedResume]:
        return [r for r in self.resumes if r.ok]

    @property
    def failed(self) -> list[ParsedResume]:
        return [r for r in self.resumes if r.status in {"failed", "empty"}]

    @property
    def duplicates(self) -> list[ParsedResume]:
        return [r for r in self.resumes if r.status == "duplicate"]

    def counts(self) -> dict[str, int]:
        return {
            "total_files": self.total_files,
            "parsed": len(self.parsed),
            "failed": len(self.failed),
            "duplicates": len(self.duplicates),
        }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_resume_files(input_dir: Path) -> list[Path]:
    """Return supported resume files inside ``input_dir`` (non-recursive, sorted)."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = [
        path
        for path in sorted(input_dir.iterdir(), key=lambda p: p.name.lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def compute_file_hash(path: Path) -> str:
    """Sha256 of file content, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Text extraction per format
# ---------------------------------------------------------------------------
def _extract_pdf(path: Path) -> tuple[str, int]:
    import pdfplumber  # imported lazily so unit tests can stub it

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages[:MAX_PAGES]:
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - one bad page must not fail the file
                log.warning("Page extraction failed in %s: %s", path.name, exc)
                pages.append("")
    return "\n".join(pages), page_count


def _extract_docx(path: Path) -> tuple[str, int]:
    """Minimal DOCX reader using only the standard library (bonus support)."""
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for para in root.iter(f"{namespace}p"):
        text = "".join(node.text or "" for node in para.iter(f"{namespace}t"))
        paragraphs.append(text)
    return "\n".join(paragraphs), 1


def _extract_txt(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding), 1
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), 1


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_resume(path: Path, file_hash: str | None = None) -> ParsedResume:
    """Parse one resume file. Never raises for a malformed document."""
    parsed = ParsedResume(
        path=path,
        filename=path.name,
        file_hash=file_hash or "",
    )
    extractor = _EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        parsed.status = "unsupported"
        parsed.error = f"Unsupported file type: {path.suffix}"
        return parsed

    try:
        text, page_count = extractor(path)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for batch safety
        parsed.status = "failed"
        parsed.error = f"{type(exc).__name__}: {exc}"
        log.warning("Failed to parse %s (%s)", path.name, parsed.error)
        return parsed

    parsed.page_count = page_count
    parsed.text = normalise_text(text)[:MAX_TEXT_CHARS]
    if not parsed.text:
        parsed.status = "empty"
        parsed.error = "No extractable text (scanned or image-only PDF?)"
        log.warning("%s yielded no extractable text", path.name)
    return parsed


def ingest_directory(
    input_dir: Path,
    skip_hashes: Iterable[str] | None = None,
) -> IngestionResult:
    """Discover and parse every resume in ``input_dir``.

    Identical file contents (same sha256) are processed once and reported as
    duplicates rather than being parsed again.
    """
    files = discover_resume_files(input_dir)
    result = IngestionResult(total_files=len(files))
    seen: dict[str, str] = {h: name for h, name in _hash_pairs(skip_hashes)}

    for path in files:
        try:
            digest = compute_file_hash(path)
        except OSError as exc:
            result.resumes.append(
                ParsedResume(
                    path=path,
                    filename=path.name,
                    status="failed",
                    error=f"Unreadable file: {exc}",
                )
            )
            continue

        if digest in seen:
            result.resumes.append(
                ParsedResume(
                    path=path,
                    filename=path.name,
                    file_hash=digest,
                    status="duplicate",
                    duplicate_of=seen[digest],
                )
            )
            log.info("Skipping duplicate file %s (same content as %s)", path.name, seen[digest])
            continue

        seen[digest] = path.name
        result.resumes.append(parse_resume(path, file_hash=digest))

    log.info(
        "Ingestion complete: %d files -> %d parsed, %d failed, %d duplicates",
        result.total_files,
        len(result.parsed),
        len(result.failed),
        len(result.duplicates),
    )
    return result


def _hash_pairs(skip_hashes: Iterable[str] | None) -> list[tuple[str, str]]:
    if not skip_hashes:
        return []
    return [(digest, f"<prior:{digest[:8]}>") for digest in skip_hashes]
