"""The golden set (3.1): the cases themselves are checked, so a broken case fails here, not halfway through an eval.

Three kinds of test:
- the set as a whole (size, formats, domains);
- each case, parametrized by folder: it parses through Layer 1, every fact is really in the source (a fact the
  document doesn't contain could never be covered, and would silently cap the score), every must-not is too
  (they name source material that shouldn't become cards), and its module range fits its outline;
- the loader and the term matcher, on small inputs.
"""

import re
from pathlib import Path

import pytest

from doctodeck.evals.golden import (
    GOLDEN_DIR,
    GoldenSetError,
    TermRule,
    compile_term,
    load_case,
    load_golden_set,
)
from tests.layer3.build_golden_pdfs import pdf_from_source, pdf_sources

CASES = load_golden_set()
BY_ID = {case.id: case for case in CASES}


def case_ids(cases=CASES) -> list[str]:
    return [case.id for case in cases]


# ---------- The set ----------


def test_the_set_has_four_to_six_cases_in_both_formats_and_distinct_domains():
    assert 4 <= len(CASES) <= 6
    assert {case.document_path.suffix for case in CASES} == {".md", ".pdf"}
    domains = [case.expect.domain for case in CASES]
    assert len(set(domains)) == len(domains)


# ---------- Each case ----------


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_the_document_parses_and_is_short(case):
    document = case.parse()

    assert 150 <= document.word_count <= 1_500  # short enough to read and label by hand (3.3)
    assert len(document.sections) >= 3


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_every_fact_is_in_the_source(case):
    text = case.parse().clean_text

    missing = [fact.id for fact in case.expect.facts if not fact.matches(text)]

    assert missing == [], "a fact the document doesn't contain can never be covered"


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_every_must_not_is_in_the_source(case):
    text = case.parse().clean_text

    absent = [rule.id for rule in case.expect.must_not if not rule.matches(text)]

    assert absent == [], "must_not names source material that shouldn't become cards; it must be there"


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_the_module_range_fits_the_outline(case):
    sections = len(case.parse().sections)
    modules, cards = case.expect.modules, case.expect.cards

    assert modules.min >= 1 and modules.max <= sections  # a module covers at least one section
    assert cards.min >= modules.max  # room for at least one card per module


@pytest.mark.parametrize("source", pdf_sources(), ids=lambda path: path.parent.name)
def test_each_committed_pdf_is_up_to_date_with_its_source(source):
    committed = source.with_name("document.pdf").read_bytes()

    assert committed == pdf_from_source(source.read_text(encoding="utf-8")), (
        "rebuild with: uv run python -m tests.layer3.build_golden_pdfs"
    )


def test_the_pdf_case_gets_its_numbered_and_colon_headings_and_loses_its_page_furniture():
    document = BY_ID["roman-republic"].parse()

    titles = [section.title for section in document.sections]
    assert titles[0] == "roman-republic"  # the untitled opening, named after the file
    assert titles[1:7] == [
        "1. Founding of the Republic",
        "2. Magistrates",
        "3. The Senate and the assemblies",
        "4. The Conflict of the Orders",
        "5. The Punic Wars",
        "6. The fall of the Republic",
    ]
    assert titles[7] == "About these notes"  # a short line ending in ':' is a heading too
    assert document.page_count == 2
    assert "Study notes" not in document.clean_text  # the running header on both pages
    assert not re.search(r"Page \d of \d", document.clean_text)
    assert "509 BC" in document.clean_text and "27 BC" in document.clean_text  # dates mid-sentence, not headings


def test_hash_comments_in_the_awkward_cases_code_block_are_not_headings():
    document = BY_ID["sql-joins"].parse()

    assert all(not section.title.startswith("sqlite3") for section in document.sections)
    assert "# sqlite3 is in the standard library" in document.clean_text


# ---------- The term matcher ----------


@pytest.mark.parametrize(
    ("term", "text", "found"),
    [
        ("Vary", "the vary header", True),  # case-insensitive
        ("no-cache", "Cache-Control: no-cache, private", True),
        ("revalidat*", "it must be revalidated first", True),  # trailing * = any word ending
        ("revalidat*", "a revalidation", True),
        ("respiration", "wasteful photorespiration", False),  # whole words only
        ("respiration", "Cellular respiration is elsewhere", True),
        ("Not Modified", "304 Not\n   Modified", True),  # any whitespace matches any whitespace
        ("12.68%", "an APY of 12.68%.", True),  # punctuation at the edges still matches
        ("12.68%", "an APY of 112.68%", False),
        ("Pe^(rt)", "uses A = Pe^(rt) and", True),  # regex characters are literal
        ("ETag|If-None-Match", "send If-None-Match", True),  # | = alternatives
        ("ETag|If-None-Match", "send If-Modified-Since", False),
    ],
)
def test_term_matching(term, text, found):
    assert bool(compile_term(term).search(text)) is found


def test_a_rule_needs_all_of_its_terms():
    rule = TermRule(id="etag-304", text="t", terms=["ETag|If-None-Match", "304|Not Modified"])

    assert rule.matches("An ETag check gets a 304.")
    assert not rule.matches("An ETag check.")


def test_an_empty_alternative_is_rejected():
    with pytest.raises(ValueError):
        TermRule(id="x", text="t", terms=["ETag||304"])


# ---------- The loader ----------


GOOD_EXPECT = """
title = "T"
domain = "d"
why = "w"
modules = { min = 1, max = 2 }
cards = { min = 2, max = 9 }
[[facts]]
id = "a"
text = "t"
terms = ["alpha"]
[[facts]]
id = "b"
text = "t"
terms = ["beta"]
[[facts]]
id = "c"
text = "t"
terms = ["gamma"]
"""


def write_case(root: Path, expect: str = GOOD_EXPECT, documents: tuple[str, ...] = ("document.md",)) -> Path:
    folder = root / "case"
    folder.mkdir()
    for name in documents:
        (folder / name).write_text("# A\n\nalpha beta gamma\n", encoding="utf-8")
    (folder / "expect.toml").write_text(expect, encoding="utf-8")
    return folder


def test_a_well_formed_case_loads(tmp_path):
    case = load_case(write_case(tmp_path))

    assert case.id == "case"
    assert 2 in case.expect.cards and 10 not in case.expect.cards
    assert [fact.id for fact in case.expect.facts] == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s + "must_nt = []\n", "must_nt"),  # a typo'd key is an error, not ignored
        (lambda s: s.replace("min = 1, max = 2", "min = 3, max = 2"), "greater than max"),
        (lambda s: s.replace('id = "b"', 'id = "a"'), "duplicate ids: a"),
        (lambda s: s.replace('id = "b"', 'id = "Not Kebab"'), "id"),
        (lambda s: s.replace('[[facts]]\nid = "c"\ntext = "t"\nterms = ["gamma"]\n', ""), "facts"),
        (lambda s: s.replace('title = "T"', 'title = "T'), "case/expect.toml"),  # not valid TOML
    ],
    ids=["unknown-key", "min-over-max", "duplicate-id", "bad-id", "too-few-facts", "bad-toml"],
)
def test_a_broken_expect_file_is_reported_with_its_case(tmp_path, change, message):
    folder = write_case(tmp_path, change(GOOD_EXPECT))

    with pytest.raises(GoldenSetError, match="case") as caught:
        load_case(folder)

    assert message in str(caught.value)


@pytest.mark.parametrize("documents", [(), ("document.md", "document.pdf")], ids=["none", "two"])
def test_a_case_needs_exactly_one_document(tmp_path, documents):
    with pytest.raises(GoldenSetError, match="exactly one"):
        load_case(write_case(tmp_path, documents=documents))


def test_the_default_location_is_the_repositorys_evals_golden_folder():
    assert GOLDEN_DIR.parts[-2:] == ("evals", "golden")
    assert (GOLDEN_DIR / "http-caching" / "expect.toml").exists()
