"""Collects headings and body lines into sections.

Both the Markdown and the PDF parser feed it, so "how a section is packaged" lives in one place.
Each parser only decides what is a heading.
"""

import re

from doctodeck.errors import DocumentErrorCode, DocumentParsingError
from doctodeck.models import DocumentSection, ParsedDocument

SNIPPET_LENGTH = 250

_WHITESPACE = re.compile(r"\s+")


def count_words(text: str) -> int:
    # str.split() with no argument splits on any run of whitespace and drops empty strings.
    return len(text.split())


def build_snippet(text: str) -> str:
    flat = _WHITESPACE.sub(" ", text).strip()
    if len(flat) <= SNIPPET_LENGTH:
        return flat
    # Python strings are sequences of code points, so slicing can never split an emoji in half
    # (C#'s UTF-16 strings can, which is why the C# kata had a surrogate-pair check here).
    return flat[:SNIPPET_LENGTH] + "…"


class OutlineBuilder:
    def __init__(self, fallback_title: str) -> None:
        self._fallback_title = fallback_title
        self._sections: list[DocumentSection] = []
        self._body: list[str] = []  # the current section's lines
        self._all_lines: list[str] = []  # every line, becomes clean_text
        self._title: str | None = None  # None = before the first heading
        self._level = 1

    def add_heading(self, title: str, level: int) -> None:
        self._close_section()
        self._title = title
        self._level = level
        self._all_lines.append(title)

    def add_line(self, line: str) -> None:
        self._body.append(line)
        self._all_lines.append(line)

    def build(self, page_count: int, empty_code: DocumentErrorCode, empty_message: str) -> ParsedDocument:
        """Closes the last section and returns the outline, or raises if there is no text at all."""
        self._close_section()  # the last section has no heading after it to close it

        clean_text = "\n".join(self._all_lines).strip()
        total_words = count_words(clean_text)
        if total_words == 0:
            raise DocumentParsingError(empty_code, empty_message)

        return ParsedDocument(
            sections=list(self._sections),
            clean_text=clean_text,
            word_count=total_words,
            page_count=page_count,
        )

    def _close_section(self) -> None:
        section_text = "\n".join(self._body).strip()
        self._body.clear()

        # Blank space before the first heading is not a section.
        if self._title is None and not section_text:
            return

        # A heading with no body is still kept (0 words): it is part of the outline.
        self._sections.append(
            DocumentSection(
                title=self._title if self._title is not None else self._fallback_title,
                level=self._level,
                word_count=count_words(section_text),
                snippet=build_snippet(section_text),
            )
        )
