"""Builds each PDF golden case's document.pdf from its source.txt.

    uv run python -m tests.layer3.build_golden_pdfs

source.txt is one line of PDF text per line, with pages separated by a line reading `=== page ===`. The PDF is
built with the test suite's hand-written builder, which writes no timestamps, so the same source always gives the
same bytes: a test rebuilds every PDF and fails if a committed one is out of date.
"""

from pathlib import Path

from doctodeck.evals.golden import GOLDEN_DIR
from tests.pdf_builder import build_pdf

PAGE_BREAK = "=== page ==="


def pdf_from_source(source: str) -> bytes:
    pages: list[list[str]] = [[]]
    for line in source.splitlines():
        if line.strip() == PAGE_BREAK:
            pages.append([])
        elif line.strip():
            pages[-1].append(line)
    return build_pdf(*pages)


def pdf_sources(root: Path = GOLDEN_DIR) -> list[Path]:
    return sorted(root.glob("*/source.txt"))


def main() -> None:
    for source in pdf_sources():
        target = source.with_name("document.pdf")
        target.write_bytes(pdf_from_source(source.read_text(encoding="utf-8")))
        print(f"wrote {target.relative_to(GOLDEN_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
