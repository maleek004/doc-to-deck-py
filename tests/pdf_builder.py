"""Builds small real PDFs in memory for tests, with no PDF library.

Each list of strings is one page; each string is one line of text, written top to bottom in
Helvetica 12 pt. Writing the format by hand shows what a PDF actually is: numbered objects, a
content stream of drawing operators, and a cross-reference table of byte offsets at the end.
"""


def _escape(text: str) -> str:
    # In a PDF string literal ( ... ), backslash and parentheses must be escaped.
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(*pages: list[str]) -> bytes:
    page_count = len(pages)
    # Object numbers: 1 catalog, 2 page tree, 3 font, then (page, content stream) pairs.
    page_ids = [4 + 2 * i for i in range(page_count)]

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {page_count} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }

    for page_id, lines in zip(page_ids, pages, strict=True):
        content_id = page_id + 1
        y = 800
        ops = []
        for line in lines:
            # BT/ET = begin/end text; Tf = font and size; Td = move to (x, y); Tj = show text.
            ops.append(f"BT /F1 12 Tf 50 {y} Td ({_escape(line)}) Tj ET")
            y -= 18
        stream = "\n".join(ops).encode("cp1252")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "  # A4 in points (1/72 inch)
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"

    # The xref table: one 20-byte entry per object giving its byte offset, so a reader can jump to it.
    xref_at = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n" % size
    out += b"0000000000 65535 f \n"
    for number in range(1, size):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_at)
    return bytes(out)
