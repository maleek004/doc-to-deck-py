"""The golden set (3.1): a handful of documents, each with hand-written expectations of a good deck.

Layout, one folder per case under `evals/golden/`:

    evals/golden/http-caching/
        document.md        the document (exactly one document.md or document.pdf)
        expect.toml        what a good deck from it must and must not contain
        source.txt         PDF cases only: the text the PDF is built from (see tests/layer3/build_golden_pdfs.py)

Why TOML and not YAML: Python reads TOML with the standard library (`tomllib`, 3.11+), so the eval needs no
extra dependency, and TOML has no YAML-style surprises (`no` becoming False, `1.10` becoming 1.1).

Facts are matched by *terms*, not by exact sentences, because a card can phrase a fact a hundred ways. A fact's
`terms` must ALL appear (in a card, in 3.2); within one term, `|` separates alternatives and a trailing `*` matches
any word ending ("revalidat*" matches "revalidate", "revalidation"). Matching is case-insensitive, on whole words,
and any run of whitespace matches any other. It is deliberately simple: a deterministic floor that 3.3's LLM judge
complements, not a replacement for it.
"""

import re
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from doctodeck.models import ParsedDocument
from doctodeck.parser import ParserOptions, parse_document

# src/doctodeck/evals/golden.py → the repository root is three folders up from this file's folder.
GOLDEN_DIR = Path(__file__).resolve().parents[3] / "evals" / "golden"
DOCUMENT_NAMES = ("document.md", "document.pdf")


class _Strict(BaseModel):
    # extra="forbid": a typo in expect.toml ("must_nt") is an error, not a silently ignored key.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Range(_Strict):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) is greater than max ({self.max})")
        return self

    def __contains__(self, value: int) -> bool:
        return self.min <= value <= self.max


def compile_term(term: str) -> re.Pattern[str]:
    """One term → one regex. "s-maxage|shared|CDN*" → (?<!\\w)(?:s\\-maxage|shared|CDN\\w*)(?!\\w), ignoring case.

    (?<!\\w) and (?!\\w) are "not preceded / followed by a word character": whole-word matching that also works
    when a term starts or ends with punctuation ("12.68%", "Pe^(rt)"), where \\b would fail."""
    alternatives = []
    for alternative in term.split("|"):
        alternative = alternative.strip()
        prefix = alternative.endswith("*")
        words = alternative.removesuffix("*").split()
        if not words:
            raise ValueError(f"empty alternative in term {term!r}")
        pattern = r"\s+".join(re.escape(word) for word in words)
        alternatives.append(pattern + (r"\w*" if prefix else ""))
    return re.compile(rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)", re.IGNORECASE)


class TermRule(_Strict):
    """A fact the deck must cover, or something it must not contain. `text` is for people; `terms` decide."""

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")  # kebab-case: stable keys for reports
    text: str = Field(min_length=1)
    terms: list[str] = Field(min_length=1)

    @field_validator("terms")
    @classmethod
    def _compiles(cls, terms: list[str]) -> list[str]:
        for term in terms:
            compile_term(term)  # raises ValueError, which Pydantic reports against this field
        return terms

    # Compiled once per rule. Pydantic leaves cached_property alone, and it writes straight into the instance's
    # __dict__, so it works even on a frozen model.
    @cached_property
    def patterns(self) -> list[re.Pattern[str]]:
        return [compile_term(term) for term in self.terms]

    def matches(self, text: str) -> bool:
        """True when every term is found in `text`."""
        return all(pattern.search(text) for pattern in self.patterns)


class GoldenExpect(_Strict):
    """The contents of one expect.toml."""

    title: str
    domain: str
    why: str  # what this case is in the set to test
    modules: Range
    cards: Range
    facts: list[TermRule] = Field(min_length=3)
    must_not: list[TermRule] = []

    @model_validator(mode="after")
    def _unique_ids(self) -> "GoldenExpect":
        ids = [rule.id for rule in (*self.facts, *self.must_not)]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate ids: {', '.join(duplicates)}")
        return self


@dataclass(frozen=True)
class GoldenCase:
    id: str  # the folder name, e.g. "http-caching"
    document_path: Path
    expect: GoldenExpect

    @property
    def is_pdf(self) -> bool:
        return self.document_path.suffix == ".pdf"

    def parse(self, options: ParserOptions | None = None) -> ParsedDocument:
        """Through the same Layer 1 parser an upload goes through, so the eval sees what users' decks see."""
        with self.document_path.open("rb") as stream:
            # Named after the case ("roman-republic.pdf"): a PDF's untitled opening section takes the file name.
            return parse_document(stream, self.id + self.document_path.suffix, options)


class GoldenSetError(Exception):
    pass


def load_case(folder: Path) -> GoldenCase:
    documents = [folder / name for name in DOCUMENT_NAMES if (folder / name).exists()]
    if len(documents) != 1:
        raise GoldenSetError(f"{folder.name}: needs exactly one of {', '.join(DOCUMENT_NAMES)}, found {len(documents)}")
    expect_path = folder / "expect.toml"
    if not expect_path.exists():
        raise GoldenSetError(f"{folder.name}: no expect.toml")
    try:
        with expect_path.open("rb") as file:  # tomllib wants bytes: TOML is always UTF-8
            raw = tomllib.load(file)
        expect = GoldenExpect.model_validate(raw)
    except (tomllib.TOMLDecodeError, ValueError) as ex:  # pydantic.ValidationError is a ValueError
        raise GoldenSetError(f"{folder.name}/expect.toml: {ex}") from ex
    return GoldenCase(id=folder.name, document_path=documents[0], expect=expect)


def load_golden_set(root: Path = GOLDEN_DIR) -> list[GoldenCase]:
    """Every case folder under `root`, sorted by name so runs are always in the same order."""
    folders = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith((".", "_")))
    if not folders:
        raise GoldenSetError(f"No golden cases in {root}")
    return [load_case(folder) for folder in folders]
