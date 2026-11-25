"""Everything the parser refuses before (or instead of) parsing."""

import io

import pytest

from doctodeck.errors import DocumentErrorCode, DocumentParsingError
from doctodeck.parser import ParserOptions, parse_document
from tests.layer1.helpers import parse_bytes
from tests.pdf_builder import build_pdf


def rejects(data: bytes, filename: str, options: ParserOptions | None = None) -> DocumentParsingError:
    with pytest.raises(DocumentParsingError) as caught:
        parse_bytes(data, filename, options)
    return caught.value


def test_unsupported_extension_is_rejected():
    assert rejects(b"# Hi\ntext", "notes.docx").code == DocumentErrorCode.UNSUPPORTED_FILE_TYPE


def test_extension_check_ignores_case():
    doc = parse_bytes(b"# Hi\ntext", "NOTES.MD")

    assert doc.sections[0].title == "Hi"


def test_empty_file_is_rejected():
    assert rejects(b"", "notes.pdf").code == DocumentErrorCode.FILE_EMPTY


def test_file_over_the_size_limit_is_rejected():
    error = rejects(b"a" * 101, "notes.md", ParserOptions(max_file_size_bytes=100))

    assert error.code == DocumentErrorCode.FILE_TOO_LARGE


def test_file_exactly_at_the_size_limit_is_accepted():
    doc = parse_bytes(b"a" * 100, "notes.md", ParserOptions(max_file_size_bytes=100))

    assert doc.word_count == 1


def test_reading_stops_at_the_limit_instead_of_reading_everything():
    class EndlessStream(io.RawIOBase):
        """A stream that never ends, like a malicious upload. Reading it all would never finish."""

        def readable(self) -> bool:
            return True

        def readinto(self, buffer) -> int:
            buffer[:] = b"a" * len(buffer)
            return len(buffer)

    with pytest.raises(DocumentParsingError) as caught:
        parse_document(io.BufferedReader(EndlessStream()), "notes.md", ParserOptions(max_file_size_bytes=1_000_000))

    assert caught.value.code == DocumentErrorCode.FILE_TOO_LARGE


def test_pdf_extension_with_non_pdf_contents_is_rejected():
    assert rejects(b"This is just text renamed to .pdf", "fake.pdf").code == DocumentErrorCode.INVALID_FILE_SIGNATURE


def test_markdown_extension_with_binary_contents_is_rejected():
    exe_start = b"MZ\x00\x00\x90"  # the start of a Windows .exe

    assert rejects(exe_start, "notes.md").code == DocumentErrorCode.INVALID_FILE_SIGNATURE


def test_corrupted_pdf_is_rejected_as_malformed():
    error = rejects(b"%PDF-1.7\nthis is not really a pdf", "broken.pdf")

    assert error.code == DocumentErrorCode.MALFORMED_DOCUMENT


def test_library_error_is_kept_as_the_cause():
    # A valid signature, then a truncated body: pdfminer itself raises, and we keep its error.
    truncated = build_pdf(["one line"])[:60]

    error = rejects(truncated, "broken.pdf")

    assert error.code == DocumentErrorCode.MALFORMED_DOCUMENT
    assert error.__cause__ is not None  # set by `raise ... from ex`, kept for the logs


def test_pdf_with_too_many_pages_is_rejected():
    error = rejects(build_pdf(["one"], ["two"], ["three"]), "long.pdf", ParserOptions(max_pdf_pages=2))

    assert error.code == DocumentErrorCode.TOO_MANY_PAGES
