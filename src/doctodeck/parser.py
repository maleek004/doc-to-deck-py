"""Turns an uploaded file into an outline of sections. Mirrors AnkiX's DocumentParserService.

Supports Markdown and PDF. The order of checks is cheapest first: extension, then size (read in
chunks, stopping as soon as the limit is crossed), then file signature, then the real parse.

Parsing is synchronous on purpose: it is CPU-bound work. The API runs it in a worker thread
(`run_in_threadpool`) so a big PDF never blocks the event loop that serves other requests.
"""

import io
import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import BinaryIO

import pdfplumber

from doctodeck.errors import DocumentErrorCode, DocumentParsingError
from doctodeck.models import ParsedDocument
from doctodeck.outline import OutlineBuilder

SUPPORTED_EXTENSIONS = (".md", ".markdown", ".pdf")
PDF_SIGNATURE = b"%PDF-"
_CHUNK_SIZE = 64 * 1024

# Splits on \r\n, \r or \n only. str.splitlines() would also split on form feeds, \x1c-\x1e and
# Unicode line separators, which can sit inside real text.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")

# Markdown: 1-6 '#' at the very start of the line, at least one whitespace, then the title.
_MD_HEADING = re.compile(r"(#{1,6})\s+(.+)")

# PDF: "1. Introduction", "2.3 Async streams", "1.2.1 Scope". The title must start with a capital
# letter; Python's re has no \p{Lu}, so that part is checked with str.isupper() in code.
_PDF_NUMBERED = re.compile(r"(\d+(?:\.\d+)*)\.?\s+(\S.*)")

# PDF: "Chapter 3 ...", "Module IV ...".
_PDF_NAMED = re.compile(r"(?:Chapter|Section|Part|Module|Topic)\s+[\dIVXLCDM]+\b", re.IGNORECASE)

# PDF: "3", "Page 3", "3 / 10", "Page 3 of 10".
_PAGE_NUMBER = re.compile(r"\s*(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?\s*", re.IGNORECASE)


@dataclass(frozen=True)
class ParserOptions:
    """Safety limits for uploads. Defaults match AnkiX (25 MB, 300 PDF pages); tests pass tiny
    limits so they can prove the limits work without building a 25 MB file."""

    max_file_size_bytes: int = 25 * 1024 * 1024
    max_pdf_pages: int = 300


def parse_document(stream: BinaryIO, filename: str, options: ParserOptions | None = None) -> ParsedDocument:
    options = options or ParserOptions()
    if not filename or not filename.strip():
        raise ValueError("filename is required")

    # 1. Is this a file type we handle at all? (cheapest check first)
    path = PurePath(filename)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(
            DocumentErrorCode.UNSUPPORTED_FILE_TYPE,
            f"Unsupported file type '{extension}'. Supported: .md, .markdown, .pdf.",
        )

    # 2. Read the bytes, refusing to read past the size limit.
    data = _read_with_limit(stream, options.max_file_size_bytes)
    if not data:
        raise DocumentParsingError(DocumentErrorCode.FILE_EMPTY, "The file is empty (0 bytes).")

    # 3. Do the bytes look like what the extension claims?
    _validate_signature(data, extension)

    # 4. Parse.
    title = path.stem
    if extension == ".pdf":
        return _parse_pdf(data, title, options)
    return _parse_markdown(data, title)


def _read_with_limit(stream: BinaryIO, max_bytes: int) -> bytes:
    buffer = bytearray()
    while chunk := stream.read(_CHUNK_SIZE):
        buffer += chunk
        if len(buffer) > max_bytes:
            # Stop as soon as the limit is crossed, without reading the rest of a huge upload.
            raise DocumentParsingError(
                DocumentErrorCode.FILE_TOO_LARGE,
                f"The file is larger than the {max_bytes:,}-byte limit.",
            )
    return bytes(buffer)


def _validate_signature(data: bytes, extension: str) -> None:
    if extension == ".pdf":
        # Every real PDF starts with the bytes "%PDF-". A renamed .exe or .txt does not.
        if not data.startswith(PDF_SIGNATURE):
            raise DocumentParsingError(
                DocumentErrorCode.INVALID_FILE_SIGNATURE,
                "The file is named .pdf but its contents are not a PDF.",
            )
        return

    # Text files never contain a NUL byte; binary files almost always do (checked in the first 512 bytes).
    if b"\x00" in data[:512]:
        raise DocumentParsingError(
            DocumentErrorCode.INVALID_FILE_SIGNATURE,
            "The file is named as Markdown but contains binary data.",
        )


def _parse_markdown(data: bytes, title: str) -> ParsedDocument:
    # "utf-8-sig" drops a UTF-8 byte order mark if there is one, so the first '#' is still seen.
    # errors="replace" turns invalid bytes into U+FFFD instead of failing the whole upload.
    text = data.decode("utf-8-sig", errors="replace")

    outline = OutlineBuilder(title)
    in_code_fence = False

    for raw_line in _LINE_BREAK.split(text):
        line = raw_line.rstrip()
        start = line.lstrip()

        # ``` or ~~~ opens or closes a code block. Inside one, '#' is code, not a heading.
        if start.startswith(("```", "~~~")):
            in_code_fence = not in_code_fence
            outline.add_line(line)
            continue

        heading = None if in_code_fence else _MD_HEADING.fullmatch(line)
        if heading:
            heading_title = heading.group(2).strip().rstrip("#").strip()  # "# Title ##" -> "Title"
            outline.add_heading(heading_title, level=len(heading.group(1)))
        else:
            outline.add_line(line)

    return outline.build(1, DocumentErrorCode.EMPTY_DOCUMENT, "The document contains no text.")


def _parse_pdf(data: bytes, title: str, options: ParserOptions) -> ParsedDocument:
    try:
        pages = _extract_pdf_lines(data, options.max_pdf_pages)
    except DocumentParsingError:
        raise  # our own errors (e.g. too many pages) pass through unchanged
    except Exception as ex:
        # Anything pdfplumber/pdfminer raises means the file is broken; callers only ever see our
        # error type. `from ex` keeps the original error as __cause__ for the logs.
        raise DocumentParsingError(
            DocumentErrorCode.MALFORMED_DOCUMENT, "The PDF could not be read. It may be corrupted."
        ) from ex

    _remove_page_noise(pages)

    outline = OutlineBuilder(title)
    for line in (line for page in pages for line in page):
        heading = _pdf_heading(line)
        if heading:
            outline.add_heading(*heading)
        else:
            outline.add_line(line)

    return outline.build(
        len(pages),
        DocumentErrorCode.NO_EXTRACTABLE_TEXT,
        "The PDF has no extractable text. Scanned or image-only PDFs need OCR, which is not supported.",
    )


def _extract_pdf_lines(data: bytes, max_pages: int) -> list[list[str]]:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        page_count = len(pdf.pages)
        if page_count == 0:
            raise DocumentParsingError(DocumentErrorCode.MALFORMED_DOCUMENT, "The PDF has no pages.")
        if page_count > max_pages:
            raise DocumentParsingError(
                DocumentErrorCode.TOO_MANY_PAGES,
                f"The PDF has {page_count} pages; the limit is {max_pages}.",
            )

        pages: list[list[str]] = []
        for page in pdf.pages:
            # pdfplumber groups characters into lines by their vertical position, so line breaks
            # survive. (AnkiX's PdfPig `page.Text` loses them: see Pay It Back #1 in plan.md.)
            text = page.extract_text() or ""
            pages.append([s for s in (line.strip() for line in _LINE_BREAK.split(text)) if s])
        return pages


def _remove_page_noise(pages: list[list[str]]) -> None:
    """Removes page numbers, then any line that sits at the top (header) or bottom (footer) of at
    least half the pages. Page numbers go first: they change on every page, so while they are the
    last line they would hide a repeated footer sitting just above them."""
    for i, lines in enumerate(pages):
        pages[i] = [line for line in lines if not _PAGE_NUMBER.fullmatch(line)]

    if len(pages) < 2:
        return  # one page: nothing can "repeat"

    threshold = max(2, -(-len(pages) // 2))  # ceil(n / 2) with integers only

    first_lines = [p[0] for p in pages if p]
    last_lines = [p[-1] for p in pages if len(p) > 1]
    headers = {line for line in first_lines if first_lines.count(line) >= threshold}
    footers = {line for line in last_lines if last_lines.count(line) >= threshold}

    for lines in pages:
        # Only remove them where they sit; the same words in the middle of a page are content.
        if lines and lines[0] in headers:
            del lines[0]
        if lines and lines[-1] in footers:
            del lines[-1]


def _pdf_heading(line: str) -> tuple[str, int] | None:
    """PDFs have no '#', so headings are guessed from the text, as AnkiX does: numbered ("1.",
    "2.3"), named ("Chapter 3"), or a short line ending in ':'. Returns (title, level) or None."""
    if len(line) > 90:
        return None  # headings are short; long lines are sentences

    title = line.rstrip(":").strip()

    numbered = _PDF_NUMBERED.fullmatch(line)
    if numbered and numbered.group(2)[0].isupper():
        level = min(numbered.group(1).count(".") + 1, 6)  # "1"→1, "2.3"→2, "1.2.1"→3
        return title, level

    if _PDF_NAMED.match(line):
        return title, 1

    if line.endswith(":") and len(line) < 60 and "." not in line:
        return title, 2

    return None
