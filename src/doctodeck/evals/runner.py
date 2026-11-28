"""The eval runner (3.4): the golden set through the real pipeline, one JSON line per case.

A run file (`evals/runs/<run id>.jsonl`, git-ignored) is a log, written as it goes:

    {"kind": "run", ...}          the header: provider, model, prompt version and fingerprint, the cases
    {"kind": "deck", ...}         one per finished case: its modules, cards and metrics
    {"kind": "stopped", ...}      only if the provider became unavailable: where the run stopped and why
    {"kind": "summary", ...}      the SetSummary over every finished case, and whether the run is complete

Each line is flushed as soon as its case finishes, so a crash, a Ctrl+C or a quota running out keeps everything
done so far. `resume_eval` re-reads the file, checks it is the same experiment, and runs only the missing cases.

Two kinds of failure, treated differently:
- The *model's* fault (invalid output, refusal, cut-off answer): that IS the result for this case. It is scored
  (no cards: coverage 0, and the schema-valid rate shows it) and the run goes on.
- The *provider* is unavailable (rate limit, timeout, network, bad key): every later case would fail too, and
  scoring them 0 would blame the prompt for an outage. The run stops, marked incomplete, ready to resume.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from doctodeck.costs.ledger import deck_scope
from doctodeck.evals.golden import GoldenCase
from doctodeck.evals.metrics import CallRecord, DeckMetrics, DeckRun, SetSummary, TokenPrice, deck_metrics, summarize
from doctodeck.llm.contracts import LlmError, LlmErrorCode
from doctodeck.llm.registry import LlmRegistry
from doctodeck.models import ApiModel
from doctodeck.pipeline.generator import CardGenerator
from doctodeck.pipeline.models import GeneratedCard, GenerateRequest, PipelineOptions, PlannedModule, PlanRequest
from doctodeck.pipeline.planner import DeckPlanner
from doctodeck.pipeline.prompts import PromptSet

# The provider can't be reached or won't serve us: stop the run rather than score an outage as a bad deck.
STOPPING_ERRORS = frozenset(
    {LlmErrorCode.RATE_LIMITED, LlmErrorCode.TIMEOUT, LlmErrorCode.PROVIDER_ERROR, LlmErrorCode.UNKNOWN_PROVIDER}
)


# ---------- The lines of a run file ----------


class RunHeader(ApiModel):
    kind: Literal["run"] = "run"
    run_id: str
    started_at: str
    provider: str
    model: str
    prompt_version: str
    prompt_fingerprint: str
    cached: bool  # whether real providers ran behind the response cache
    cases: list[str]


class DeckRecord(ApiModel):
    kind: Literal["deck"] = "deck"
    case_id: str
    status: Literal["ok", "model_error"]
    error_code: str | None = None
    error_message: str | None = None
    modules: list[PlannedModule]
    cards: list[GeneratedCard]  # kept so the judge (3.3) and hand-labelling can work from the run file
    metrics: DeckMetrics


class StopRecord(ApiModel):
    kind: Literal["stopped"] = "stopped"
    case_id: str
    error_code: str
    message: str


class SummaryRecord(ApiModel):
    kind: Literal["summary"] = "summary"
    complete: bool
    summary: SetSummary


@dataclass
class RunFile:
    """A run file read back. Later lines win: a resumed run appends a new summary, and the last one counts."""

    path: Path
    header: RunHeader
    decks: dict[str, DeckRecord] = field(default_factory=dict)
    stopped: StopRecord | None = None
    summary: SummaryRecord | None = None

    @property
    def complete(self) -> bool:
        return all(case_id in self.decks for case_id in self.header.cases)

    @property
    def missing(self) -> list[str]:
        return [case_id for case_id in self.header.cases if case_id not in self.decks]


class RunFileError(Exception):
    pass


def read_run(path: Path) -> RunFile:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RunFileError(f"{path.name} is empty")
    try:
        run = RunFile(path, RunHeader.model_validate_json(lines[0]))
        for line in lines[1:]:
            kind = json.loads(line).get("kind")
            if kind == "deck":
                deck = DeckRecord.model_validate_json(line)
                run.decks[deck.case_id] = deck
                run.stopped = None  # a deck after a stop means the run was resumed past it
            elif kind == "stopped":
                run.stopped = StopRecord.model_validate_json(line)
            elif kind == "summary":
                run.summary = SummaryRecord.model_validate_json(line)
            else:
                raise RunFileError(f"{path.name}: unknown line kind {kind!r}")
    except ValueError as ex:  # bad JSON, or a line of the wrong shape (pydantic.ValidationError is a ValueError)
        raise RunFileError(f"{path.name}: {ex}") from ex
    return run


def latest_run(runs_dir: Path) -> Path | None:
    """Run ids start with a UTC timestamp, so the newest file sorts last."""
    files = sorted(runs_dir.glob("*.jsonl")) if runs_dir.is_dir() else []
    return files[-1] if files else None


# ---------- Running ----------


@dataclass(frozen=True)
class EvalSettings:
    provider: str
    prompts: PromptSet
    cached: bool = False
    options: PipelineOptions = field(default_factory=PipelineOptions)
    prices: dict[str, TokenPrice] = field(default_factory=dict)


def new_run_id(provider: str, prompt_version: str, now: datetime) -> str:
    return f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{provider}-{prompt_version}"


async def run_case(case: GoldenCase, llms: LlmRegistry, settings: EvalSettings) -> DeckRecord:
    """One golden case through plan → generate → metrics. A model fault becomes a scored `model_error` record;
    a provider outage raises LlmError for the caller to stop on."""
    document = case.parse()
    planner = DeckPlanner(llms, settings.options, settings.prompts)
    generator = CardGenerator(llms, settings.options, settings.prompts)
    calls: list[CallRecord] = []
    modules: list[PlannedModule] = []
    cards: list[GeneratedCard] = []
    repairs: list[str] = []
    truncated = False
    failure: LlmError | None = None

    try:
        plan = await planner.plan(
            PlanRequest(
                provider=settings.provider, source_title=case.id, sections=document.sections,
                word_count=document.word_count, page_count=document.page_count,
            )
        )  # fmt: skip
        calls.append(CallRecord("plan", plan.usage))
        modules, repairs = plan.modules, list(plan.repairs)

        generated = await generator.generate(
            GenerateRequest(
                provider=settings.provider, source_title=case.id, sections=document.sections,
                clean_text=document.clean_text, modules=modules,
            )
        )  # fmt: skip
        calls.append(CallRecord("cards", generated.usage))
        cards, truncated = generated.cards, generated.truncated
        repairs += generated.repairs
    except LlmError as ex:
        if ex.code in STOPPING_ERRORS:
            raise
        failure = ex
        calls.append(CallRecord("plan" if not modules else "cards", error=ex.code))

    run = DeckRun(case.id, case.expect, document, modules, cards, calls, truncated, repairs)
    return DeckRecord(
        case_id=case.id,
        status="model_error" if failure else "ok",
        error_code=failure.code if failure else None,
        error_message=failure.message if failure else None,
        modules=modules,
        cards=cards,
        metrics=deck_metrics(run, settings.prices),
    )


def _append(path: Path, record: ApiModel) -> None:
    # Open, write one line, close: each line is on disk before the next case starts.
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(record.model_dump_json(by_alias=True) + "\n")


async def _run_cases(path: Path, cases: Sequence[GoldenCase], llms: LlmRegistry, settings: EvalSettings,
                     progress: Callable[[str], None]) -> RunFile:  # fmt: skip
    for case in cases:
        try:
            with deck_scope(f"{path.stem}/{case.id}"):  # the ledger groups each case's calls
                record = await run_case(case, llms, settings)
        except LlmError as ex:
            _append(path, StopRecord(case_id=case.id, error_code=ex.code, message=ex.message))
            progress(f"{case.id}: stopped, {ex.code}: {ex.message}")
            break
        _append(path, record)
        m = record.metrics
        progress(f"{case.id}: {record.status}, {m.card_count} cards, facts {m.fact_coverage:.0%}")

    run = read_run(path)
    if run.decks:
        complete = run.complete
        _append(path, SummaryRecord(complete=complete, summary=summarize([d.metrics for d in run.decks.values()])))
    return read_run(path)


async def start_eval(
    cases: Sequence[GoldenCase],
    llms: LlmRegistry,
    settings: EvalSettings,
    runs_dir: Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[str], None] = lambda _: None,
) -> RunFile:
    client = llms.get(settings.provider)  # an unknown provider fails here, before a file is written
    started = now()
    run_id = new_run_id(settings.provider, settings.prompts.version, started)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_id}.jsonl"
    header = RunHeader(
        run_id=run_id,
        started_at=started.astimezone(UTC).isoformat(timespec="seconds"),
        provider=client.provider,
        model=client.model,
        prompt_version=settings.prompts.version,
        prompt_fingerprint=settings.prompts.fingerprint(),
        cached=settings.cached,
        cases=[case.id for case in cases],
    )
    path.write_text(header.model_dump_json(by_alias=True) + "\n", encoding="utf-8", newline="\n")
    return await _run_cases(path, cases, llms, settings, progress)


async def resume_eval(
    path: Path,
    cases: Sequence[GoldenCase],
    llms: LlmRegistry,
    settings: EvalSettings,
    progress: Callable[[str], None] = lambda _: None,
) -> RunFile:
    """Run the cases the file is still missing, after checking it is the same experiment: same provider, model,
    prompt version and prompt text. Mixing two experiments in one file would make its summary meaningless."""
    run = read_run(path)
    header, client = run.header, llms.get(settings.provider)
    now_is = (client.provider, client.model, settings.prompts.version, settings.prompts.fingerprint())
    was = (header.provider, header.model, header.prompt_version, header.prompt_fingerprint)
    if now_is != was:
        raise RunFileError(
            f"{path.name} was run as {'/'.join(was)}; this is {'/'.join(now_is)}. Start a new run instead."
        )
    by_id = {case.id: case for case in cases}
    unknown = [case_id for case_id in run.missing if case_id not in by_id]
    if unknown:
        raise RunFileError(f"{path.name} needs cases that are no longer in the golden set: {', '.join(unknown)}")
    if not run.missing:
        progress("nothing to resume: every case is done")
        return run
    return await _run_cases(path, [by_id[case_id] for case_id in run.missing], llms, settings, progress)
