"""The FastAPI app: serves the page and the JSON API.

Run it with `uv run doctodeck` (or `uv run uvicorn doctodeck.app:app --reload --port 8000`).
"""

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from doctodeck.errors import DocumentErrorCode, DocumentParsingError
from doctodeck.models import ApiModel, ParsedDocument
from doctodeck.parser import ParserOptions, parse_document

STATIC_DIR = Path(__file__).parent / "static"
NO_FILE = "NO_FILE"


class ParseResponse(ApiModel):
    """The response of POST /api/parse: the file's name plus its outline and clean text."""

    file_name: str
    document: ParsedDocument


class ApiError(ApiModel):
    """The body of every error this API returns: a stable code for code, a message for people."""

    error: str
    message: str


def get_parser_options() -> ParserOptions:
    """A dependency, so a test can swap in tiny limits with `app.dependency_overrides`.
    (The Python counterpart of re-registering a service in ASP.NET's container.)"""
    return ParserOptions()


# Translate the parser's error code into the matching HTTP status. Anything else that goes wrong
# is a bug, so it is left to bubble up and become a 500.
_STATUS_BY_CODE = {
    DocumentErrorCode.FILE_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    DocumentErrorCode.UNSUPPORTED_FILE_TYPE: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(ApiError(error=code, message=message).model_dump(by_alias=True), status_code=status_code)


def create_app() -> FastAPI:
    app = FastAPI(title="doc-to-deck")

    @app.exception_handler(DocumentParsingError)
    async def on_parsing_error(_: Request, ex: DocumentParsingError) -> JSONResponse:
        return error_response(_STATUS_BY_CODE.get(ex.code, status.HTTP_400_BAD_REQUEST), ex.code, ex.message)

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

    return app


app = create_app()
