"""Command line interface for the resume screening pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Settings, load_dotenv
from .pipeline import run_pipeline
from .report import format_report
from .utils import get_logger, setup_logging

log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AI Resume Screening & Ranking System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="./resumes", help="Directory containing resume PDFs")
    parser.add_argument("--output", default="./output/results.json", help="Path for results.json")
    parser.add_argument("--model", default=None, help="Override the LLM model name")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enrichment entirely")
    parser.add_argument("--no-github", action="store_true", help="Disable GitHub enrichment entirely")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and do not write the .cache run cache (forces fresh API calls)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N resumes (0 = all); handy for smoke tests",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a terminal report with the top candidates after the run",
    )
    parser.add_argument("--top", type=int, default=10, help="Rows shown by --report")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    load_dotenv()
    setup_logging(verbose=args.verbose)

    settings = Settings.from_args(
        input_dir=args.input,
        output_path=args.output,
        model=args.model,
        no_llm=args.no_llm,
        no_github=args.no_github,
        no_cache=args.no_cache,
        limit=args.limit,
        verbose=args.verbose,
    )

    try:
        result = run_pipeline(settings)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except Exception:  # broad by design: the CLI must stay informative
        log.exception("Pipeline failed")
        return 1

    output_path = Path(settings.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = result.get("summary", {})
    log.info(
        "Done: total=%s parsed=%s eligible=%s rejected=%s failed=%s -> %s",
        summary.get("total_resumes"),
        summary.get("parsed"),
        summary.get("eligible"),
        summary.get("rejected"),
        summary.get("failed"),
        output_path,
    )
    if args.report:
        print(format_report(result, top=max(args.top, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
