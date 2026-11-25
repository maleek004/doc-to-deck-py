import io

from doctodeck.models import ParsedDocument
from doctodeck.parser import ParserOptions, parse_document


def parse_bytes(data: bytes, filename: str, options: ParserOptions | None = None) -> ParsedDocument:
    return parse_document(io.BytesIO(data), filename, options)
