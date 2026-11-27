"""Deterministic metrics (3.2): numbers about a generated deck that need no LLM to compute.

Pure functions: a `DeckRun` (what one golden case produced) goes in, a `DeckMetrics` comes out; `summarize` folds
many of those into one `SetSummary` for a report row. No I/O and no clock, so every number here is reproducible
and unit-tested on hand-made card sets. The runner (3.4) builds the DeckRuns; the judge (3.3) adds what rules
can't see (is the card *true* to the source? is it worth remembering?).

What each metric is for:
- fact coverage: did the deck teach what the document is about? (the golden facts; the headline number)
- must-not hits: did it make cards from boilerplate or out-of-scope material?
- gate flag rates: how often the quality gate's rules fired, per rule
- near-duplicate rate: the same fact asked twice, compared across the whole deck (the gate only compares fronts)
- section coverage: how much of the document's outline ended up with cards
- cards per 1k words: deck density, comparable across documents of different length
- schema-valid rate: of the answers the model gave, how many matched the output schema
- tokens, cost, latency (p50/p95): what the deck cost and how long the user waited
"""

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from doctodeck.evals.golden import GoldenExpect
from doctodeck.llm.contracts import LlmErrorCode, LlmUsage
from doctodeck.models import ApiModel, ParsedDocument
from doctodeck.pipeline.models import GeneratedCard, PlannedModule, QualityFlag, QualityScore

# Two cards are near-duplicates when their front+back word sets overlap this much (Jaccard). Lower than the gate's
# 0.80 on fronts alone: front+back adds words, so the same fact asked two ways scores lower than two equal fronts.
DUPLICATE_JACCARD = 0.6
_WORD = re.compile(r"\w+")
# Words too common to say anything about whether two cards share a fact.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "at", "for", "by", "with", "and", "or",
        "is", "are", "was", "were", "be", "it", "its", "this", "that",
        "what", "which", "who", "how", "why", "when", "does", "do", "did", "can",
    }
)  # fmt: skip


# ---------- Inputs ----------


@dataclass(frozen=True)
class CallRecord:
    """One LLM call made while building a deck: its usage if it answered, its error code if it failed."""

    purpose: str  # "plan" or "cards"
    usage: LlmUsage | None = None
    error: LlmErrorCode | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class DeckRun:
    """What one golden case produced through the pipeline."""

    case_id: str
    expect: GoldenExpect
    document: ParsedDocument
    modules: Sequence[PlannedModule]
    cards: Sequence[GeneratedCard]
    calls: Sequence[CallRecord]
    truncated: bool = False
    repairs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class TokenPrice:
    """US dollars per million tokens. Layer 4 owns the real price table; evals only need to be handed one."""

    input_per_million: float
    output_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_million + completion_tokens * self.output_per_million) / 1_000_000


# ---------- Outputs ----------


class MustNotHit(ApiModel):
    rule_id: str
    card_index: int  # 0-based position in the deck


class DeckMetrics(ApiModel):
    """One golden case's numbers. An ApiModel, so the runner can write it as one JSON line (camelCase)."""

    case_id: str
    card_count: int
    module_count: int
    card_count_in_range: bool
    module_count_in_range: bool

    fact_coverage: float  # 0..1: facts covered by at least one card / all facts
    covered_facts: list[str]
    missed_facts: list[str]
    must_not_hits: list[MustNotHit]

    flag_rates: dict[QualityFlag, float]  # share of cards with each flag (every flag present, 0.0 when unseen)
    score_rates: dict[QualityScore, float]  # share of cards per gate score
    near_duplicate_rate: float  # share of cards that repeat an earlier card
    section_coverage: float  # share of non-empty sections that ended up with at least one card
    cards_per_1k_words: float

    calls: int
    failed_calls: int
    answered_calls: int  # calls where the model produced an answer (valid or not)
    schema_valid_calls: int
    cached_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None  # None when no price is known for a model used
    latency_ms: int  # the user's wait for this deck: the calls run one after another, so it's their sum

    truncated: bool
    repair_count: int


class SetSummary(ApiModel):
    """Many DeckMetrics folded into one report row (3.4 compares rows: prompt v1 vs v2, Gemini vs Claude)."""

    cases: int
    cards: int
    mean_fact_coverage: float  # macro: the average of per-case coverage, so every document counts the same
    facts_covered: int  # micro: across all cases
    facts_total: int
    must_not_hits: int
    cases_with_card_count_in_range: int
    cases_with_module_count_in_range: int
    flag_rates: dict[QualityFlag, float]  # pooled over all cards
    needs_review_rate: float
    near_duplicate_rate: float  # pooled over all cards
    mean_section_coverage: float
    mean_cards_per_1k_words: float
    schema_valid_rate: float | None  # pooled over answered calls; None when no call got an answer
    failed_calls: int
    cached_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    deck_latency_p50_ms: int | None
    deck_latency_p95_ms: int | None


# ---------- Building blocks ----------


def card_text(card: GeneratedCard) -> str:
    """What a fact must be found in: the card's question and answer together."""
    return f"{card.front}\n{card.back}"


def fact_coverage(expect: GoldenExpect, cards: Sequence[GeneratedCard]) -> tuple[list[str], list[str]]:
    """(covered fact ids, missed fact ids). A fact is covered when ONE card holds all its terms: cards should be
    atomic, so a fact whose terms are scattered over three cards hasn't been taught by any of them."""
    texts = [card_text(card) for card in cards]
    covered, missed = [], []
    for fact in expect.facts:
        (covered if any(fact.matches(text) for text in texts) else missed).append(fact.id)
    return covered, missed


def must_not_hits(expect: GoldenExpect, cards: Sequence[GeneratedCard]) -> list[MustNotHit]:
    return [
        MustNotHit(rule_id=rule.id, card_index=index)
        for index, card in enumerate(cards)
        for rule in expect.must_not
        if rule.matches(card_text(card))
    ]


def rates[K](values: Iterable[K], keys: Iterable[K], total: int) -> dict[K, float]:
    """Share of `total` for every key, including the keys never seen (0.0), so report columns never go missing."""
    counts = dict.fromkeys(keys, 0)
    for value in values:
        counts[value] += 1
    return {key: (count / total if total else 0.0) for key, count in counts.items()}


def content_words(text: str) -> frozenset[str]:
    return frozenset(word for word in _WORD.findall(text.casefold()) if word not in _STOP_WORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """len(A & B) / len(A | B): 1.0 for the same set, 0.0 for nothing shared. Two empty sets count as 0, not 1:
    two empty cards aren't evidence of a repeated fact."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def near_duplicate_count(cards: Sequence[GeneratedCard], threshold: float = DUPLICATE_JACCARD) -> int:
    """Cards whose front+back is at least `threshold` similar to an EARLIER card. The first of a pair is the
    original and isn't counted, so a deck with one fact asked twice has one duplicate, not two.
    O(n²) pairs: fine for decks of tens of cards; MinHash would be the answer for millions."""
    seen: list[frozenset[str]] = []
    duplicates = 0
    for card in cards:
        words = content_words(card_text(card))
        if any(jaccard(words, earlier) >= threshold for earlier in seen):
            duplicates += 1
        seen.append(words)
    return duplicates


def section_coverage(
    document: ParsedDocument, modules: Sequence[PlannedModule], cards: Sequence[GeneratedCard]
) -> float:
    """Share of the document's non-empty sections that belong to a module which got at least one card.

    Measured through the plan, because cards don't say which section they came from. Empty sections (a heading
    with no text) are left out: there is nothing in them to make cards from. Boilerplate sections ("About these
    notes") stay in, so a perfect deck scores a little under 1.0; compare runs on the same set, not to 1.0."""
    eligible = {i for i, section in enumerate(document.sections) if section.word_count > 0}
    if not eligible:
        return 0.0
    modules_with_cards = {card.module_index for card in cards}
    covered = {
        i for module in modules if module.module_index in modules_with_cards for i in module.section_indices
    } & eligible
    return len(covered) / len(eligible)


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p% of the values at or below it.

    Nearest-rank always returns a value that was actually observed (no interpolation), which is the honest choice
    for a handful of samples: with 5 decks, p95 is simply the slowest one. None for no values."""
    if not values:
        return None
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))  # 1-based
    return ordered[rank - 1]


def calls_cost(calls: Iterable[CallRecord], prices: Mapping[str, TokenPrice]) -> float | None:
    """What the calls cost at list price, cached ones included (a cached answer still cost this once; what was
    *spent this run* is Layer 4's ledger). None if any model used has no price: a partial sum would look cheap."""
    total = 0.0
    for call in calls:
        if call.usage is None:
            continue
        price = prices.get(call.usage.model)
        if price is None:
            return None
        total += price.cost(call.usage.prompt_tokens, call.usage.completion_tokens)
    return total


# ---------- Per deck and per set ----------


def deck_metrics(run: DeckRun, prices: Mapping[str, TokenPrice] | None = None) -> DeckMetrics:
    cards, expect, document = run.cards, run.expect, run.document
    n = len(cards)
    covered, missed = fact_coverage(expect, cards)
    usages = [call.usage for call in run.calls if call.usage is not None]
    answered = [call for call in run.calls if call.ok or call.error == LlmErrorCode.INVALID_OUTPUT]

    return DeckMetrics(
        case_id=run.case_id,
        card_count=n,
        module_count=len(run.modules),
        card_count_in_range=n in expect.cards,
        module_count_in_range=len(run.modules) in expect.modules,
        fact_coverage=len(covered) / len(expect.facts),
        covered_facts=covered,
        missed_facts=missed,
        must_not_hits=must_not_hits(expect, cards),
        flag_rates=rates((flag for card in cards for flag in card.quality_flags), QualityFlag, n),
        score_rates=rates((card.quality_score for card in cards), QualityScore, n),
        near_duplicate_rate=near_duplicate_count(cards) / n if n else 0.0,
        section_coverage=section_coverage(document, run.modules, cards),
        cards_per_1k_words=n / document.word_count * 1000 if document.word_count else 0.0,
        calls=len(run.calls),
        failed_calls=sum(not call.ok for call in run.calls),
        answered_calls=len(answered),
        schema_valid_calls=sum(call.ok for call in answered),
        cached_calls=sum(usage.cached for usage in usages),
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        cost_usd=calls_cost(run.calls, prices or {}),
        latency_ms=sum(usage.latency_ms for usage in usages),
        truncated=run.truncated,
        repair_count=len(run.repairs),
    )


def summarize(decks: Sequence[DeckMetrics]) -> SetSummary:
    if not decks:
        raise ValueError("nothing to summarize")
    cards = sum(deck.card_count for deck in decks)
    answered = sum(deck.answered_calls for deck in decks)
    costs = [deck.cost_usd for deck in decks]
    latencies = [deck.latency_ms for deck in decks]

    def pooled(per_deck_rate: float, deck: DeckMetrics) -> float:
        return per_deck_rate * deck.card_count  # back from a share to a count, to pool over all cards

    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values)

    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    return SetSummary(
        cases=len(decks),
        cards=cards,
        mean_fact_coverage=mean(deck.fact_coverage for deck in decks),
        facts_covered=sum(len(deck.covered_facts) for deck in decks),
        facts_total=sum(len(deck.covered_facts) + len(deck.missed_facts) for deck in decks),
        must_not_hits=sum(len(deck.must_not_hits) for deck in decks),
        cases_with_card_count_in_range=sum(deck.card_count_in_range for deck in decks),
        cases_with_module_count_in_range=sum(deck.module_count_in_range for deck in decks),
        flag_rates={
            flag: (sum(pooled(deck.flag_rates[flag], deck) for deck in decks) / cards if cards else 0.0)
            for flag in QualityFlag
        },
        needs_review_rate=(
            sum(pooled(deck.score_rates[QualityScore.NEEDS_REVIEW], deck) for deck in decks) / cards if cards else 0.0
        ),
        near_duplicate_rate=sum(pooled(deck.near_duplicate_rate, deck) for deck in decks) / cards if cards else 0.0,
        mean_section_coverage=mean(deck.section_coverage for deck in decks),
        mean_cards_per_1k_words=mean(deck.cards_per_1k_words for deck in decks),
        schema_valid_rate=sum(deck.schema_valid_calls for deck in decks) / answered if answered else None,
        failed_calls=sum(deck.failed_calls for deck in decks),
        cached_calls=sum(deck.cached_calls for deck in decks),
        prompt_tokens=sum(deck.prompt_tokens for deck in decks),
        completion_tokens=sum(deck.completion_tokens for deck in decks),
        cost_usd=None if any(cost is None for cost in costs) else sum(costs),
        deck_latency_p50_ms=None if p50 is None else round(p50),
        deck_latency_p95_ms=None if p95 is None else round(p95),
    )
