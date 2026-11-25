import pytest

from doctodeck.errors import DocumentErrorCode, DocumentParsingError
from doctodeck.models import ParsedDocument
from doctodeck.outline import SNIPPET_LENGTH
from tests.layer1.helpers import parse_bytes


def parse(markdown: str, filename: str = "notes.md") -> ParsedDocument:
    return parse_bytes(markdown.encode("utf-8"), filename)


def test_two_top_level_headings_become_two_sections_with_their_titles():
    doc = parse(
        "# Async and await\n"
        "An async method returns a coroutine that completes later.\n"
        "\n"
        "# Cancellation\n"
        "Cancelling a task raises CancelledError inside it.\n"
    )

    assert [s.title for s in doc.sections] == ["Async and await", "Cancellation"]


def test_heading_levels_are_recorded():
    doc = parse("# Chapter\nintro\n## Section\ndetail\n### Subsection\nfine print\n")

    assert [s.level for s in doc.sections] == [1, 2, 3]


def test_hash_inside_a_code_fence_is_not_a_heading():
    doc = parse("""\
# Python comments
Use a hash to start a comment:
```python
# this is a comment, not a heading
x = 1
```
""")

    assert [s.title for s in doc.sections] == ["Python comments"]
    assert "# this is a comment" in doc.clean_text


def test_hashtag_without_a_space_is_not_a_heading():
    doc = parse("# Tags\n#python and #fastapi are tags, not headings.\n")

    assert [s.title for s in doc.sections] == ["Tags"]


def test_text_before_the_first_heading_becomes_a_section_named_after_the_file():
    doc = parse("These notes cover the basics.\n\n# Next\nMore text.\n", "python-basics.md")

    assert [s.title for s in doc.sections] == ["python-basics", "Next"]


def test_heading_with_no_body_is_kept_with_zero_words():
    doc = parse("# Overview\n# Details\nSome text here.\n")

    assert [(s.title, s.word_count) for s in doc.sections] == [("Overview", 0), ("Details", 3)]


def test_closing_hashes_are_trimmed_from_the_title():
    doc = parse("# Title ##\nbody\n")

    assert doc.sections[0].title == "Title"


def test_windows_and_old_mac_line_endings_do_not_leak_into_titles():
    doc = parse("# First\r\nbody one\r\n# Second\rbody two\r")

    assert [s.title for s in doc.sections] == ["First", "Second"]


def test_long_section_snippet_is_cut_to_the_limit():
    long_body = " ".join(["word"] * 200)  # ~1000 chars

    doc = parse(f"# Long\n{long_body}")

    (section,) = doc.sections
    assert section.word_count == 200
    assert len(section.snippet) == SNIPPET_LENGTH + 1  # + the "…"
    assert section.snippet.endswith("…")


def test_snippet_never_splits_an_emoji():
    body = "a" * (SNIPPET_LENGTH - 1) + "😀" + " tail" * 10

    doc = parse(f"# Emoji\n{body}")

    assert doc.sections[0].snippet == "a" * (SNIPPET_LENGTH - 1) + "😀…"


def test_document_with_only_whitespace_is_rejected():
    with pytest.raises(DocumentParsingError) as caught:
        parse("   \n\n\t\n")

    assert caught.value.code == DocumentErrorCode.EMPTY_DOCUMENT


def test_utf8_byte_order_mark_does_not_hide_the_first_heading():
    bom = b"\xef\xbb\xbf"  # what Windows Notepad puts at the start of "UTF-8 with BOM" files

    doc = parse_bytes(bom + b"# First\nbody", "notes.md")

    assert [s.title for s in doc.sections] == ["First"]


def test_invalid_utf8_is_replaced_not_fatal():
    doc = parse_bytes(b"# Caf\xe9\nlatin-1 bytes in a utf-8 file", "notes.md")

    assert doc.sections[0].title == "Caf�"
