"""Every reason the parser can reject a file.

A StrEnum instead of loose strings: a typo is an AttributeError at import time, and because each
member *is* a str, it serialises to JSON as "FILE_TOO_LARGE" with no extra code. The page switches
on the same values.
"""

from enum import StrEnum


class DocumentErrorCode(StrEnum):
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_EMPTY = "FILE_EMPTY"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_SIGNATURE = "INVALID_FILE_SIGNATURE"
    TOO_MANY_PAGES = "TOO_MANY_PAGES"
    MALFORMED_DOCUMENT = "MALFORMED_DOCUMENT"
    NO_EXTRACTABLE_TEXT = "NO_EXTRACTABLE_TEXT"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"


class DocumentParsingError(Exception):
    """A document the parser refuses to process.

    `code` is stable and machine-readable; the message is for people. When a library (pdfplumber)
    caused it, raise this `from` the original error so the traceback keeps the real cause for logs.
    """

    def __init__(self, code: DocumentErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
