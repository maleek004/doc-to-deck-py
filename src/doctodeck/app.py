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

from doctodeck.errors import DocumentErrorCode, DocumentParsingError
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
        app.state.llms = build_registry(http)
        yield


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
        return await planner.plan(body)

    @app.post("/api/generate", responses={400: {"model": ApiError}, 502: {"model": ApiError}})
    async def generate(
        body: GenerateRequest, generator: Annotated[CardGenerator, Depends(get_generator)]
    ) -> GenerateResult:
        return await generator.generate(body)

    return app


app = create_app()
