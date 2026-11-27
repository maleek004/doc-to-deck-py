"""Deterministic metrics (3.2), on hand-made card sets whose right answers can be worked out on paper."""

import json

import pytest

from doctodeck.evals.golden import GoldenExpect, load_golden_set
from doctodeck.evals.metrics import (
    CallRecord,
    DeckMetrics,
    DeckRun,
    TokenPrice,
    calls_cost,
    content_words,
    deck_metrics,
    fact_coverage,
    jaccard,
    must_not_hits,
    near_duplicate_count,
    percentile,
    rates,
    section_coverage,
    summarize,
)
from doctodeck.llm.contracts import LlmErrorCode, LlmUsage
from doctodeck.models import DocumentSection, ParsedDocument
from doctodeck.pipeline.models import GeneratedCard, PlannedModule, QualityFlag, QualityScore

# ---------- Builders ----------


def card(front: str, back: str = "An answer.", module: int = 1, flags=(), score=QualityScore.EXCELLENT):
    return GeneratedCard(
        front=front, back=back, module_index=module, module_title=f"M{module}", topic_tag="t", rationale="r",
        quality_score=score, quality_flags=list(flags),
    )  # fmt: skip


def module(index: int, sections: list[int]) -> PlannedModule:
    return PlannedModule(module_index=index, title=f"M{index}", description="d", estimated_cards=3,
                         section_indices=sections)  # fmt: skip


def document(*section_words: int) -> ParsedDocument:
    sections = [DocumentSection(title=f"S{i}", level=1, word_count=w, snippet="") for i, w in enumerate(section_words)]
    return ParsedDocument(sections=sections, clean_text="x", word_count=sum(section_words), page_count=1)


def usage(prompt=100, completion=50, latency=1000, model="m", cached=False) -> LlmUsage:
    return LlmUsage(provider="p", model=model, prompt_tokens=prompt, completion_tokens=completion,
                    latency_ms=latency, cached=cached)  # fmt: skip


EXPECT = GoldenExpect.model_validate(
    {
        "title": "T", "domain": "d", "why": "w",
        "modules": {"min": 1, "max": 2}, "cards": {"min": 2, "max": 4},
        "facts": [
            {"id": "etag", "text": "t", "terms": ["ETag", "304"]},
            {"id": "vary", "text": "t", "terms": ["Vary"]},
            {"id": "no-store", "text": "t", "terms": ["no-store"]},
        ],
        "must_not": [{"id": "boilerplate", "text": "t", "terms": ["golden set"]}],
    }
)  # fmt: skip


def run(cards, calls=(), modules=None, doc=None, **kwargs) -> DeckRun:
    return DeckRun(
        case_id="case",
        expect=EXPECT,
        document=doc or document(400, 600),
        modules=modules if modules is not None else [module(1, [0]), module(2, [1])],
        cards=cards,
        calls=calls or [CallRecord("plan", usage()), CallRecord("cards", usage())],
        **kwargs,
    )


# ---------- Fact coverage and must-nots ----------


def test_a_fact_is_covered_when_one_card_holds_all_its_terms():
    deck = [
        card("What does an ETag check return?", "A 304 if unchanged."),
        card("What does the vary header do?", "Adds headers to the key."),
    ]

    covered, missed = fact_coverage(EXPECT, deck)

    assert covered == ["etag", "vary"]  # in expect order; matching ignores case ("vary")
    assert missed == ["no-store"]


def test_terms_spread_over_two_cards_do_not_cover_a_fact():
    _, missed = fact_coverage(EXPECT, [card("What is an ETag?", "A version tag."),
                                       card("What does 304 mean?", "Not Modified.")])  # fmt: skip

    assert "etag" in missed  # no single card teaches "ETag revalidation gets a 304"


def test_must_not_hits_name_the_rule_and_the_card():
    hits = must_not_hits(EXPECT, [card("What is Vary?"), card("Who wrote these notes?", "The golden set authors.")])

    assert [(hit.rule_id, hit.card_index) for hit in hits] == [("boilerplate", 1)]


def test_the_real_http_caching_case_on_a_good_deck_and_a_bad_card():
    case = next(c for c in load_golden_set() if c.id == "http-caching")
    deck = [
        card("What does Cache-Control: max-age=600 mean?", "The response is fresh for 600 seconds."),
        card("What does s-maxage control?", "Freshness in shared caches such as a CDN."),
        card("What does revalidating with If-None-Match return when nothing changed?", "304 Not Modified, no body."),
        card("Which header does a cache send with a Last-Modified date?", "If-Modified-Since."),
        card("What does no-cache require?", "The cache must revalidate before every use."),
        card("Which directive forbids storing a response?", "no-store."),
        card("Who may store a private response?", "Only the user's browser, not shared caches."),
        card("What does the Vary header change?", "It adds request headers to the cache key."),
        card("Why can fingerprinted file names be cached for a year?", "A change makes a new URL."),
        card("When were these notes last revised?", "November 2026."),
    ]

    covered, missed = fact_coverage(case.expect, deck)
    hits = must_not_hits(case.expect, deck)

    assert missed == [] and len(covered) == 9
    assert [(hit.rule_id, hit.card_index) for hit in hits] == [("about-these-notes", 9)]


# ---------- Rates, duplicates, sections ----------


def test_rates_list_every_key_even_unseen_ones_and_survive_an_empty_deck():
    shares = rates([QualityFlag.ANSWER_LEAK, QualityFlag.ANSWER_LEAK, QualityFlag.AMBIGUOUS_CUE], QualityFlag, 4)

    assert shares[QualityFlag.ANSWER_LEAK] == 0.5
    assert shares[QualityFlag.AMBIGUOUS_CUE] == 0.25
    assert shares[QualityFlag.EMPTY_BACK] == 0.0
    assert set(shares) == set(QualityFlag)
    assert rates([], QualityFlag, 0) == dict.fromkeys(QualityFlag, 0.0)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [({"x", "y"}, {"x", "y"}, 1.0), ({"x"}, {"y"}, 0.0), ({"x", "y", "z"}, {"y", "z", "w"}, 0.5), (set(), set(), 0.0)],
)
def test_jaccard(a, b, expected):
    assert jaccard(frozenset(a), frozenset(b)) == expected


def test_content_words_fold_case_and_drop_stop_words():
    assert content_words("What does the ETag do?") == {"etag"}


def test_the_same_fact_asked_twice_is_one_duplicate_not_two():
    deck = [
        card("What does no-store forbid?", "Storing the response at all."),
        card("What is forbidden by no-store?", "Storing the response at all."),  # reworded, same fact
        card("What does no-store forbid?", "Storing the response at all."),  # a third copy
        card("What does Vary add to the cache key?", "Request headers."),  # a different fact
    ]

    assert near_duplicate_count(deck) == 2  # the first of each group is the original


def test_cards_on_the_same_topic_but_different_facts_are_not_duplicates():
    deck = [
        card("What does no-cache require?", "Revalidation with the server before every use."),
        card("What does no-store forbid?", "Storing the response anywhere, for sensitive data."),
    ]

    assert near_duplicate_count(deck) == 0


def test_section_coverage_counts_sections_of_modules_that_got_cards():
    doc = document(100, 200, 0, 50)  # section 2 is an empty heading: not counted
    modules = [module(1, [0, 1]), module(2, [3]), module(3, [2])]

    assert section_coverage(doc, modules, [card("q", module=1)]) == pytest.approx(2 / 3)
    assert section_coverage(doc, modules, [card("q", module=1), card("q", module=2)]) == 1.0
    assert section_coverage(doc, modules, []) == 0.0


def test_section_coverage_ignores_indices_outside_the_outline():
    assert section_coverage(document(100), [module(1, [0, 7, -1])], [card("q")]) == 1.0


# ---------- Percentiles and cost ----------


@pytest.mark.parametrize(
    ("p", "expected"),
    [(30, 20), (40, 20), (50, 35), (95, 50), (100, 50)],  # the textbook nearest-rank example
)
def test_nearest_rank_percentile(p, expected):
    assert percentile([35, 20, 15, 50, 40], p) == expected  # order of input doesn't matter


def test_percentile_edges():
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7  # one deck: p50 and p95 are that deck
    assert percentile(list(range(1, 11)), 95) == 10  # with 10 samples, p95 is the slowest
    with pytest.raises(ValueError):
        percentile([1], 0)


def test_cost_uses_the_price_of_each_model_and_skips_failed_calls():
    prices = {"flash": TokenPrice(0.30, 2.50), "haiku": TokenPrice(1.00, 5.00)}
    calls = [
        CallRecord("plan", usage(prompt=1_000_000, completion=0, model="flash")),  # $0.30
        CallRecord("cards", usage(prompt=200_000, completion=100_000, model="haiku")),  # $0.20 + $0.50
        CallRecord("cards", error=LlmErrorCode.RATE_LIMITED),  # no answer, no tokens
    ]

    assert calls_cost(calls, prices) == pytest.approx(1.00)


def test_an_unpriced_model_makes_the_cost_unknown_not_cheaper():
    calls = [CallRecord("plan", usage(model="flash")), CallRecord("cards", usage(model="mystery"))]

    assert calls_cost(calls, {"flash": TokenPrice(1, 1)}) is None


# ---------- One deck ----------


def test_deck_metrics_on_a_known_deck():
    cards = [
        card("What does an ETag revalidation return?", "304 when unchanged.", module=1),
        card("What does Vary do?", "Adds headers.", module=1, flags=[QualityFlag.AMBIGUOUS_CUE],
             score=QualityScore.GOOD),
        card("Who wrote this?", "The golden set team.", module=1,
             flags=[QualityFlag.AMBIGUOUS_CUE, QualityFlag.ANSWER_LEAK], score=QualityScore.NEEDS_REVIEW),
    ]  # fmt: skip
    calls = [
        CallRecord("plan", usage(prompt=300, completion=40, latency=1200)),
        CallRecord("cards", error=LlmErrorCode.INVALID_OUTPUT),  # answered, but not to the schema
        CallRecord("cards", error=LlmErrorCode.TIMEOUT),  # no answer at all
        CallRecord("cards", usage(prompt=900, completion=400, latency=3800, cached=True)),
    ]

    m = deck_metrics(run(cards, calls, repairs=["Card 4 was empty and was skipped."], truncated=True))

    assert (m.card_count, m.module_count) == (3, 2)
    assert m.card_count_in_range and m.module_count_in_range
    assert m.fact_coverage == pytest.approx(2 / 3)
    assert m.missed_facts == ["no-store"]
    assert [hit.card_index for hit in m.must_not_hits] == [2]
    assert m.flag_rates[QualityFlag.AMBIGUOUS_CUE] == pytest.approx(2 / 3)
    assert m.flag_rates[QualityFlag.ANSWER_LEAK] == pytest.approx(1 / 3)
    assert m.score_rates[QualityScore.NEEDS_REVIEW] == pytest.approx(1 / 3)
    assert m.section_coverage == 0.5  # only module 1 (section 0) got cards; section 1 has words but no cards
    assert m.cards_per_1k_words == 3.0  # 3 cards over 1,000 words
    assert (m.calls, m.failed_calls, m.answered_calls, m.schema_valid_calls) == (4, 2, 3, 2)
    assert (m.prompt_tokens, m.completion_tokens, m.latency_ms) == (1200, 440, 5000)
    assert m.cached_calls == 1
    assert m.cost_usd is None  # no prices given
    assert m.truncated and m.repair_count == 1


def test_counts_outside_the_expected_ranges_are_reported():
    many = [card(f"What is fact number {i} about?", f"Answer {i}.") for i in range(6)]

    m = deck_metrics(run(many, modules=[module(i, [0]) for i in range(1, 4)]))

    assert not m.card_count_in_range  # 6 > max 4
    assert not m.module_count_in_range  # 3 > max 2


def test_an_empty_deck_scores_zero_without_dividing_by_zero():
    m = deck_metrics(run([]))

    assert (m.fact_coverage, m.near_duplicate_rate, m.section_coverage, m.cards_per_1k_words) == (0, 0, 0, 0)


def test_deck_metrics_round_trip_as_one_camel_case_json_line():
    m = deck_metrics(run([card("What does Vary do?", "Adds headers.")]))

    line = m.model_dump_json(by_alias=True)

    assert "\n" not in line
    assert json.loads(line)["factCoverage"] == pytest.approx(1 / 3)
    assert DeckMetrics.model_validate_json(line) == m


# ---------- The set ----------


def test_macro_and_micro_coverage_differ_when_cases_differ_in_size():
    small = deck_metrics(run([card("What does Vary do?", "Adds headers.")]))  # 1 of 3 facts
    big = small.model_copy(update={"case_id": "big", "fact_coverage": 1.0,
                                   "covered_facts": [f"f{i}" for i in range(10)], "missed_facts": []})  # fmt: skip

    summary = summarize([small, big])

    assert summary.mean_fact_coverage == pytest.approx((1 / 3 + 1.0) / 2)  # each document counts the same
    assert (summary.facts_covered, summary.facts_total) == (11, 13)  # each fact counts the same


def test_flag_rates_are_pooled_over_all_cards_not_averaged_per_deck():
    one_flagged = deck_metrics(run([card("What does Vary do?", flags=[QualityFlag.ANSWER_LEAK])]))
    clean = deck_metrics(run([card(f"What is clean fact {i}?", f"Answer {i}.") for i in range(3)]))

    summary = summarize([one_flagged, clean])

    assert summary.flag_rates[QualityFlag.ANSWER_LEAK] == pytest.approx(1 / 4)  # not (1.0 + 0.0) / 2


def test_summary_latency_percentiles_and_totals():
    decks = [
        deck_metrics(run([card("q")], calls=[CallRecord("cards", usage(latency=ms, prompt=10, completion=5))]))
        for ms in (4000, 1000, 2000, 3000, 9000)
    ]

    summary = summarize(decks)

    assert (summary.deck_latency_p50_ms, summary.deck_latency_p95_ms) == (3000, 9000)
    assert (summary.prompt_tokens, summary.completion_tokens) == (50, 25)
    assert summary.schema_valid_rate == 1.0


def test_summary_unknowns_stay_unknown():
    failed = deck_metrics(run([], calls=[CallRecord("plan", error=LlmErrorCode.RATE_LIMITED)]))

    summary = summarize([failed])

    assert summary.schema_valid_rate is None  # no call got an answer: not 0%, unknown
    assert summary.cost_usd == 0.0  # no answer, no tokens: the cost is known, and it is nothing
    assert summary.failed_calls == 1
    with pytest.raises(ValueError):
        summarize([])


def test_summary_cost_adds_up_when_every_model_is_priced():
    prices = {"m": TokenPrice(1.0, 2.0)}
    deck = deck_metrics(run([card("q")], calls=[CallRecord("cards", usage(prompt=500_000, completion=250_000))]),
                        prices)  # fmt: skip

    assert deck.cost_usd == pytest.approx(1.0)
    assert summarize([deck, deck]).cost_usd == pytest.approx(2.0)
