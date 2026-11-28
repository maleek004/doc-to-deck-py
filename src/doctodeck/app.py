"""The FastAPI app: serves the page and the JSON API.

Run it with `uv run doctodeck` (or `uv run uvicorn doctodeck.app:app --reload --port 8000`).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from doctodeck.costs.ledger import DeckCost, Ledger, LedgerEntry, deck_scope
from doctodeck.errors import DocumentErrorCode, DocumentParsingError
from doctodeck.evals.report import run_summary, summary_rows
from doctodeck.evals.runner import RunFileError, latest_run, read_run
from doctodeck.llm.cache import ResponseCache
from doctodeck.llm.contracts import LlmError, LlmErrorCode
from doctodeck.llm.registry import LlmRegistry, build_registry
from doctodeck.models import ApiModel, ParsedDocument
from doctodeck.parser import ParserOptions, parse_document
from doctodeck.pipeline.generator import CardGenerator
from doctodeck.pipeline.models import (
    INVALID_REQUEST,
    GenerateRequest,
    GenerateResult,
    PipelineOptions,
    PipelineRequestError,
    PlanRequest,
    PlanResult,
)
from doctodeck.pipeline.planner import DeckPlanner

STATIC_DIR = Path(__file__).parent / "static"
NO_FILE = "NO_FILE"


class ParseResponse(ApiModel):
    """The response of POST /api/parse: the file's name plus its outline and clean text."""

    file_name: str
    document: ParsedDocument


class ProviderInfo(ApiModel):
    """One entry in the page's provider dropdown."""

    provider: str
    model: str
    is_real: bool


class ApiError(ApiModel):
    """The body of every error this API returns: a stable code for code, a message for people."""

    error: str
    message: str


def get_parser_options() -> ParserOptions:
    """A dependency, so a test can swap in tiny limits with `app.dependency_overrides`.
    (The Python counterpart of re-registering a service in ASP.NET's container.)"""
    return ParserOptions()


def get_llm_registry(request: Request) -> LlmRegistry:
    """The registry built once at startup (see lifespan). Tests override this to add a scripted fake."""
    return request.app.state.llms


class EvalCaseRow(ApiModel):
    case_id: str
    status: str
    cards: int
    fact_coverage: str  # formatted, e.g. "62% (5/8)"
    missed_facts: list[str]
    must_not_hits: list[str]


class EvalMetricRow(ApiModel):
    label: str
    value: str


class EvalReport(ApiModel):
    """GET /api/evals/latest: the newest run file, already formatted, for panel 3."""

    run_id: str
    provider: str
    model: str
    prompt_version: str
    started_at: str
    complete: bool
    stopped: str | None  # why the run stopped early, if it did
    summary: list[EvalMetricRow]
    cases: list[EvalCaseRow]


NO_EVAL_RUNS = "NO_EVAL_RUNS"
EVAL_RUN_UNREADABLE = "EVAL_RUN_UNREADABLE"


def get_eval_runs_dir() -> Path:
    """Where `doctodeck-eval run` writes. A dependency, so tests point it at a temporary folder."""
    from doctodeck.evals.cli import RUNS_DIR

    return RUNS_DIR


def build_eval_report(path: Path) -> EvalReport:
    run = read_run(path)
    summary = run_summary(run)
    cases = []
    for case_id in run.header.cases:
        deck = run.decks.get(case_id)
        if deck is None:
            cases.append(EvalCaseRow(case_id=case_id, status="not run", cards=0, fact_coverage="",
                                     missed_facts=[], must_not_hits=[]))  # fmt: skip
            continue
        m = deck.metrics
        total = len(m.covered_facts) + len(m.missed_facts)
        cases.append(
            EvalCaseRow(
                case_id=case_id,
                status=deck.status if deck.status == "ok" else f"{deck.status}: {deck.error_code}",
                cards=m.card_count,
                fact_coverage=f"{m.fact_coverage:.0%} ({len(m.covered_facts)}/{total})",
                missed_facts=m.missed_facts,
                must_not_hits=[hit.rule_id for hit in m.must_not_hits],
            )
        )
    header = run.header
    return EvalReport(
        run_id=header.run_id,
        provider=header.provider,
        model=header.model,
        prompt_version=header.prompt_version,
        started_at=header.started_at,
        complete=run.complete,
        stopped=f"{run.stopped.case_id}: {run.stopped.error_code}" if run.stopped else None,
        summary=[EvalMetricRow(label=label, value=value) for label, value in summary_rows(summary)] if summary else [],
        cases=cases,
    )


def get_ledger(request: Request) -> Ledger:
    return request.app.state.ledger


class LedgerView(ApiModel):
    """GET /api/ledger: recent calls, cost per deck, and what real providers cost today and ever."""

    entries: list[LedgerEntry]
    decks: list[DeckCost]
    spent_today_usd: float
    spent_total_usd: float


def get_pipeline_options() -> PipelineOptions:
    return PipelineOptions()


# Sub-dependencies: FastAPI resolves the registry and the options first, then builds the step from them.
# A new planner per request is cheap; the expensive part (the registry and, from 2.2, its HTTP clients) is shared.
def get_planner(
    llms: Annotated[LlmRegistry, Depends(get_llm_registry)],
    options: Annotated[PipelineOptions, Depends(get_pipeline_options)],
) -> DeckPlanner:
    return DeckPlanner(llms, options)


def get_generator(
    llms: Annotated[LlmRegistry, Depends(get_llm_registry)],
    options: Annotated[PipelineOptions, Depends(get_pipeline_options)],
) -> CardGenerator:
    return CardGenerator(llms, options)


# Translate the parser's error code into the matching HTTP status. Anything else that goes wrong
# is a bug, so it is left to bubble up and become a 500.
_STATUS_BY_CODE = {
    DocumentErrorCode.FILE_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    DocumentErrorCode.UNSUPPORTED_FILE_TYPE: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}


# Which side is at fault decides the status: 4xx = the request, 5xx = us or the provider upstream.
_LLM_STATUS_BY_CODE = {
    LlmErrorCode.UNKNOWN_PROVIDER: status.HTTP_400_BAD_REQUEST,
    LlmErrorCode.REFUSED: status.HTTP_422_UNPROCESSABLE_CONTENT,  # the content was declined
    LlmErrorCode.RATE_LIMITED: status.HTTP_503_SERVICE_UNAVAILABLE,  # busy even after retries; try later
    LlmErrorCode.TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
}  # anything else: 502 Bad Gateway, the upstream provider misbehaved or sent bad output


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(ApiError(error=code, message=message).model_dump(by_alias=True), status_code=status_code)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once around the app's whole life. One shared httpx.AsyncClient keeps a pool of open
    connections to the providers; opening a new client per call would redo the TCP + TLS handshake
    every time. `async with` closes the pool cleanly at shutdown."""
    async with httpx.AsyncClient() as http:
        # LLM_CACHE set: real providers answer repeat requests from a SQLite file (llm/cache.py).
        app.state.ledger = Ledger.from_env()  # every call is recorded, with its cost (Layer 4)
        app.state.llms = build_registry(http, ResponseCache.from_env(), app.state.ledger)
        try:
            yield
        finally:
            await app.state.llms.aclose()  # clients that own a pool (the Claude SDK) close it here


def create_app() -> FastAPI:
    app = FastAPI(title="doc-to-deck", lifespan=lifespan)

    @app.exception_handler(DocumentParsingError)
    async def on_parsing_error(_: Request, ex: DocumentParsingError) -> JSONResponse:
        return error_response(_STATUS_BY_CODE.get(ex.code, status.HTTP_400_BAD_REQUEST), ex.code, ex.message)

    @app.exception_handler(LlmError)
    async def on_llm_error(_: Request, ex: LlmError) -> JSONResponse:
        return error_response(_LLM_STATUS_BY_CODE.get(ex.code, status.HTTP_502_BAD_GATEWAY), ex.code, ex.message)

    @app.exception_handler(PipelineRequestError)
    async def on_pipeline_request_error(_: Request, ex: PipelineRequestError) -> JSONResponse:
        return error_response(status.HTTP_400_BAD_REQUEST, INVALID_REQUEST, str(ex))

    @app.exception_handler(RequestValidationError)
    async def on_invalid_body(_: Request, ex: RequestValidationError) -> JSONResponse:
        # FastAPI's default is a 422 with its own body shape; keep every error as { error, message }.
        problems = "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in ex.errors()[:3])
        return error_response(status.HTTP_400_BAD_REQUEST, INVALID_REQUEST, problems)

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/parse", response_model=ParseResponse, responses={400: {"model": ApiError}})
    async def parse(
        options: Annotated[ParserOptions, Depends(get_parser_options)],
        file: Annotated[UploadFile | None, File()] = None,
    ) -> ParseResponse | JSONResponse:
        """POST a multipart form with a field named "file" (.md, .markdown or .pdf)."""
        if file is None or not file.filename:
            return error_response(
                status.HTTP_400_BAD_REQUEST, NO_FILE, "Attach a .md or .pdf file in a form field named 'file'."
            )

        # Parsing is CPU-bound and synchronous; run it on a worker thread so this event loop keeps
        # serving other requests meanwhile. file.file is the spooled temp file Starlette wrote.
        document = await run_in_threadpool(parse_document, file.file, file.filename, options)
        return ParseResponse(file_name=file.filename, document=document)

    @app.get("/api/providers")
    async def providers(llms: Annotated[LlmRegistry, Depends(get_llm_registry)]) -> list[ProviderInfo]:
        clients = sorted(llms.all, key=lambda c: c.is_real)  # the fake first: it always works
        return [ProviderInfo(provider=c.provider, model=c.model, is_real=c.is_real) for c in clients]

    # JSON bodies: FastAPI validates them against the Pydantic model before the function runs.
    @app.post("/api/plan", responses={400: {"model": ApiError}, 502: {"model": ApiError}})
    async def plan(body: PlanRequest, planner: Annotated[DeckPlanner, Depends(get_planner)]) -> PlanResult:
        with deck_scope(body.deck_id):  # the ledger files this call under the page's deck
            return await planner.plan(body)

    @app.post("/api/generate", responses={400: {"model": ApiError}, 502: {"model": ApiError}})
    async def generate(
        body: GenerateRequest, generator: Annotated[CardGenerator, Depends(get_generator)]
    ) -> GenerateResult:
        with deck_scope(body.deck_id):
            return await generator.generate(body)

    @app.get("/api/ledger")
    async def ledger_view(ledger: Annotated[Ledger, Depends(get_ledger)], limit: int = 50) -> LedgerView:
        from datetime import UTC, datetime

        limit = max(1, min(limit, 500))
        return LedgerView(
            entries=await ledger.recent(limit),
            decks=await ledger.decks(20),
            spent_today_usd=await ledger.spent(datetime.now(UTC).date().isoformat()),
            spent_total_usd=await ledger.spent(),
        )

    @app.get("/api/evals/latest", response_model=EvalReport, responses={404: {"model": ApiError}})
    async def latest_eval(runs_dir: Annotated[Path, Depends(get_eval_runs_dir)]) -> EvalReport | JSONResponse:
        """The newest `doctodeck-eval` run (panel 3). Reading a file is blocking I/O: done on a worker thread."""
        path = latest_run(runs_dir)
        if path is None:
            return error_response(status.HTTP_404_NOT_FOUND, NO_EVAL_RUNS,
                                  "No eval runs yet. Run: uv run doctodeck-eval run --provider fake")  # fmt: skip
        try:
            return await run_in_threadpool(build_eval_report, path)
        except RunFileError as ex:
            return error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, EVAL_RUN_UNREADABLE, str(ex))

    return app


app = create_app()
