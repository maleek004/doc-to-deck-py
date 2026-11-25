import pytest

from doctodeck.errors import DocumentErrorCode, DocumentParsingError
from doctodeck.models import ParsedDocument
from tests.layer1.helpers import parse_bytes
from tests.pdf_builder import build_pdf


def parse(*pages: list[str], filename: str = "notes.pdf") -> ParsedDocument:
    return parse_bytes(build_pdf(*pages), filename)


def titles(doc: ParsedDocument) -> list[str]:
    return [s.title for s in doc.sections]


def test_numbered_headings_become_sections():
    doc = parse(["1. Introduction", "Async functions return coroutines.", "2. Cancellation", "Tasks can be cancelled."])

    assert titles(doc) == ["1. Introduction", "2. Cancellation"]
    assert doc.sections[0].word_count == 4
    assert doc.page_count == 1


def test_sub_numbering_sets_the_heading_level():
    doc = parse(["1. Basics", "text", "1.1 Variables", "text", "1.2.1 Scope", "text"])

    assert [s.level for s in doc.sections] == [1, 2, 3]


def test_chapter_and_colon_headings_are_detected():
    doc = parse(["Chapter 2 Concurrency", "Threads share memory.", "Key ideas:", "Locks protect shared state."])

    assert titles(doc) == ["Chapter 2 Concurrency", "Key ideas"]
    assert [s.level for s in doc.sections] == [1, 2]


def test_a_section_continues_across_a_page_break():
    doc = parse(["1. Introduction", "first part of the text"], ["second part of the text"])

    (section,) = doc.sections
    assert section.word_count == 10
    assert doc.page_count == 2


def test_repeated_headers_footers_and_page_numbers_are_removed():
    doc = parse(
        ["ACME Course Notes", "1. Arrays", "Arrays hold items in order.", "Confidential", "Page 1 of 3"],
        ["ACME Course Notes", "2. Lists", "Lists can grow.", "Confidential", "Page 2 of 3"],
        ["ACME Course Notes", "3. Maps", "Maps link keys to values.", "Confidential", "Page 3 of 3"],
    )

    assert titles(doc) == ["1. Arrays", "2. Lists", "3. Maps"]
    assert "ACME" not in doc.clean_text
    assert "Confidential" not in doc.clean_text
    assert "Page 1 of 3" not in doc.clean_text
    assert "Maps link keys to values." in doc.clean_text


def test_header_text_in_the_middle_of_a_page_is_kept():
    doc = parse(
        ["ACME Course Notes", "1. About", "Written by ACME Course Notes staff."],
        ["ACME Course Notes", "2. More", "ACME Course Notes", "is also a mid-page line here."],
    )

    # The top-of-page copies go; the mid-page one is content and stays.
    assert "ACME Course Notes\nis also a mid-page line here." in doc.clean_text


def test_a_single_page_keeps_its_first_and_last_lines():
    doc = parse(["ACME Course Notes", "Just one page of text.", "Confidential"])

    assert "ACME Course Notes" in doc.clean_text
    assert "Confidential" in doc.clean_text


def test_text_without_headings_becomes_one_section_named_after_the_file():
    doc = parse(["Plain paragraph with no headings at all."], filename="lecture-03.pdf")

    assert titles(doc) == ["lecture-03"]


def test_a_numbered_sentence_starting_lowercase_is_not_a_heading():
    doc = parse(["1. Basics", "2 cups of flour make one loaf."])

    assert titles(doc) == ["1. Basics"]


def test_text_with_parentheses_survives_the_round_trip():
    doc = parse(["1. Calls", "print(len(items)) shows the count."])

    assert "print(len(items)) shows the count." in doc.clean_text


def test_a_pdf_with_no_text_is_rejected():
    with pytest.raises(DocumentParsingError) as caught:
        parse([])

    assert caught.value.code == DocumentErrorCode.NO_EXTRACTABLE_TEXT
