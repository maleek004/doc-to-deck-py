"""`doctodeck-eval`: run the golden set, resume a stopped run, and report on runs.

    uv run doctodeck-eval run --provider fake --prompt v1
    uv run doctodeck-eval run --provider gemini --prompt v1         # real calls, through the response cache
    uv run doctodeck-eval run --resume evals/runs/<id>.jsonl         # after a rate limit, carry on
    uv run doctodeck-eval report evals/runs/<a>.jsonl evals/runs/<b>.jsonl --out evals/report.md

Keys come from the environment or `.env` in the current folder, like the app. Real providers always run behind
the response cache (LLM_CACHE, or data/llm-cache.sqlite3 when unset) unless --no-cache: a rerun of an unchanged
prompt then costs nothing, and quota goes only to what changed.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from doctodeck.costs.ledger import Ledger
from doctodeck.costs.prices import PRICES
from doctodeck.evals.golden import GOLDEN_DIR, load_golden_set
from doctodeck.evals.report import markdown_report
from doctodeck.evals.runner import EvalSettings, RunFile, RunFileError, read_run, resume_eval, start_eval
from doctodeck.llm.cache import ResponseCache
from doctodeck.llm.contracts import LlmError
from doctodeck.llm.registry import build_registry
from doctodeck.pipeline.prompts import PROMPT_SETS

RUNS_DIR = GOLDEN_DIR.parent / "runs"
DEFAULT_CACHE = Path("data/llm-cache.sqlite3")
EXIT_INCOMPLETE = 2  # the run stopped on a provider outage: resume it later


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doctodeck-eval", description="Evaluate deck generation on the golden set.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the golden set (or resume a stopped run)")
    run.add_argument("--provider", default="fake", help="fake, gemini or claude (default: fake)")
    run.add_argument("--prompt", default="v1", choices=sorted(PROMPT_SETS), help="prompt version (default: v1)")
    run.add_argument("--cases", help="comma-separated case ids (default: all)")
    run.add_argument("--resume", type=Path, metavar="RUN_FILE", help="carry on a stopped run")
    run.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    run.add_argument("--no-cache", action="store_true", help="call real providers even for repeat requests")

    report = commands.add_parser("report", help="Markdown report of one or more run files")
    report.add_argument("runs", nargs="+", type=Path, metavar="RUN_FILE")
    report.add_argument("--out", type=Path, help="write the report here instead of printing it")
    return parser


def _load_env() -> None:
    """`.env` into the environment without overriding what is already set (uvicorn's loader does the same)."""
    if Path(".env").is_file():
        from dotenv import load_dotenv  # installed with uvicorn[standard]

        load_dotenv(".env", override=False)


def _cache() -> ResponseCache:
    return ResponseCache(os.environ.get("LLM_CACHE", "").strip() or DEFAULT_CACHE)


async def _run(args: argparse.Namespace) -> int:
    cases = load_golden_set()
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = sorted(set(wanted) - {case.id for case in cases})
        if unknown:
            print(f"Unknown case(s): {', '.join(unknown)}", file=sys.stderr)
            return 1
        cases = [case for case in cases if case.id in wanted]

    prompts = PROMPT_SETS[args.prompt]
    provider = args.provider
    if args.resume:  # the run file decides what this run is
        header = read_run(args.resume).header
        provider, prompts = header.provider, PROMPT_SETS.get(header.prompt_version, prompts)

    cache = None if args.no_cache else _cache()
    async with httpx.AsyncClient() as http:
        llms = build_registry(http, cache, Ledger.from_env())  # eval calls cost money too
        try:
            # Only real clients are wrapped in the cache (registry.py): a fake run is never "cached".
            cached = cache is not None and llms.get(provider).is_real
            today = datetime.now(UTC).date()
            settings = EvalSettings(provider=provider, prompts=prompts, cached=cached, prices=PRICES.prices_on(today))
            if args.resume:
                run = await resume_eval(args.resume, cases, llms, settings, progress=print)
            else:
                run = await start_eval(cases, llms, settings, args.runs_dir, progress=print)
        finally:
            await llms.aclose()
    return _finish(run)


def _finish(run: RunFile) -> int:
    print()
    print(markdown_report([run]))
    print(f"Run file: {run.path.as_posix()}")
    if not run.complete:
        print(
            f"Incomplete ({len(run.missing)} case(s) left). Resume: doctodeck-eval run --resume {run.path.as_posix()}"
        )
        return EXIT_INCOMPLETE
    return 0


def _report(args: argparse.Namespace) -> int:
    text = markdown_report([read_run(path) for path in args.runs])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {args.out.as_posix()}")
    else:
        print(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _load_env()
    # Windows consoles default to a legacy code page; the report has ✅ and ⚠️ in it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        return asyncio.run(_run(args)) if args.command == "run" else _report(args)
    except (RunFileError, LlmError) as ex:  # LlmError here: e.g. an unknown provider, before anything ran
        print(f"Error: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
