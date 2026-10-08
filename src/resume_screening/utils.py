"""Shared helpers: logging, text normalisation, small utilities."""

from __future__ import annotations

import logging
import re
import unicodedata

LOGGER_NAME = "resume_screening"

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_logging(verbose: bool = False, level: str | None = None) -> None:
    """Configure root logging exactly once."""
    resolved = level or ("DEBUG" if verbose else "INFO")
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(resolved)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
    root.propagate = False
    # Third-party parsers are chatty about cosmetic PDF issues; keep them quiet
    # unless we are actually debugging.
    logging.getLogger("pdfminer").setLevel(logging.CRITICAL if not verbose else logging.DEBUG)


def normalise_text(text: str) -> str:
    """Normalise extracted PDF text for downstream processing.

    - strips exotic unicode padding / control characters
    - converts non-breaking spaces to regular spaces
    - collapses horizontal whitespace and excessive blank lines
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # pdfminer emits "(cid:123)" placeholders for glyphs it cannot map.
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order (case-insensitive)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out
