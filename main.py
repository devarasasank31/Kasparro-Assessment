#!/usr/bin/env python3
"""Entry point for the AI Resume Screening & Ranking System.

    python main.py --input ./resumes --output ./output/results.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resume_screening.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
