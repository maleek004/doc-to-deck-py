"""Markdown reports of eval runs: one column per run, and a change column when exactly two are compared.

Built for the one question an eval answers: "did this change make the decks better, and what did it cost?" So the
first table puts the headline metrics side by side, the second shows fact coverage per document (an average can
hide one document getting much worse), and the last lists what the newest run still misses.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from doctodeck.evals.metrics import SetSummary, summarize
from doctodeck.evals.runner import RunFile
from doctodeck.pipeline.models import QualityFlag


@dataclass(frozen=True)
class _Row:
    label: str
    value: Callable[[SetSummary], float | int | None]
    show: Callable[[float | int], str]
    better: str = ""  # "up", "down" or "" (neither): sets the arrow on a change


def _pct(value: float | int) -> str:
    return f"{value:.0%}"


def _int(value: float | int) -> str:
    return f"{value:,.0f}"


def _usd(value: float | int) -> str:
    return f"${value:.4f}"


def _ms(value: float | int) -> str:
    return f"{value / 1000:.1f} s"


ROWS = [
    _Row("Fact coverage (mean per document)", lambda s: s.mean_fact_coverage, _pct, "up"),
    _Row("Facts covered", lambda s: s.facts_covered, _int, "up"),
    _Row("Must-not hits", lambda s: s.must_not_hits, _int, "down"),
    _Row("Cards", lambda s: s.cards, _int),
    _Row("Card count in range (cases)", lambda s: s.cases_with_card_count_in_range, _int, "up"),
    _Row("Module count in range (cases)", lambda s: s.cases_with_module_count_in_range, _int, "up"),
    _Row("Needs Review (gate)", lambda s: s.needs_review_rate, _pct, "down"),
    _Row("Answer leak (gate)", lambda s: s.flag_rates[QualityFlag.ANSWER_LEAK], _pct, "down"),
    _Row("Ambiguous cue (gate)", lambda s: s.flag_rates[QualityFlag.AMBIGUOUS_CUE], _pct, "down"),
    _Row("Near-duplicates", lambda s: s.near_duplicate_rate, _pct, "down"),
    _Row("Section coverage (mean)", lambda s: s.mean_section_coverage, _pct, "up"),
    _Row("Cards per 1k words (mean)", lambda s: s.mean_cards_per_1k_words, lambda v: f"{v:.1f}"),
    _Row("Schema-valid answers", lambda s: s.schema_valid_rate, _pct, "up"),
    _Row("Failed calls", lambda s: s.failed_calls, _int, "down"),
    _Row("Prompt tokens", lambda s: s.prompt_tokens, _int, "down"),
    _Row("Completion tokens", lambda s: s.completion_tokens, _int, "down"),
    _Row("Cost (list price)", lambda s: s.cost_usd, _usd, "down"),
    _Row("Deck latency p50", lambda s: s.deck_latency_p50_ms, _ms, "down"),
    _Row("Deck latency p95", lambda s: s.deck_latency_p95_ms, _ms, "down"),
]


def run_summary(run: RunFile) -> SetSummary | None:
    """Recomputed from the deck lines rather than trusting the last summary line: the decks are the data."""
    return summarize([deck.metrics for deck in run.decks.values()]) if run.decks else None


def _cell(value: float | int | None, row: _Row) -> str:
    return "n/a" if value is None else row.show(value)


def _change(a: float | int | None, b: float | int | None, row: _Row) -> str:
    if a is None or b is None:
        return ""
    delta = b - a
    if abs(delta) < 1e-9:
        return "="
    if row.show is _pct:
        text = f"{delta * 100:+.0f} pts"  # percentage points, not percent: 40% → 50% is +10 pts
    elif row.show is _ms:
        text = f"{delta / 1000:+.1f} s"
    elif row.show is _usd:
        text = f"{delta:+.4f}"
    else:
        text = f"{delta:+,.1f}".removesuffix(".0")
    good = (delta > 0) == (row.better == "up")
    return f"{text} {'✅' if good else '⚠️'}" if row.better else text


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return lines


def _name(run: RunFile) -> str:
    h = run.header
    return f"{h.provider} {h.prompt_version}" + ("" if run.complete else " (incomplete)")


def summary_rows(summary: SetSummary) -> list[tuple[str, str]]:
    """(label, formatted value) for every report row: the page's panel 3 shows the same rows as the Markdown."""
    return [(row.label, _cell(row.value(summary), row)) for row in ROWS]


def markdown_report(runs: Sequence[RunFile]) -> str:
    if not runs:
        raise ValueError("no runs to report")
    summaries = [run_summary(run) for run in runs]
    compare = len(runs) == 2
    out = ["# Eval report", ""]

    out += _table(
        ["Run", "Model", "Prompt", "Started (UTC)", "Cases done", "Cache"],
        [
            [run.header.run_id, run.header.model, f"{run.header.prompt_version} ({run.header.prompt_fingerprint})",
             run.header.started_at, f"{len(run.decks)}/{len(run.header.cases)}", "on" if run.header.cached else "off"]
            for run in runs
        ],
    )  # fmt: skip
    for run in runs:
        if run.stopped:
            out += ["", f"⚠️ `{run.header.run_id}` stopped at **{run.stopped.case_id}**: {run.stopped.error_code}. "
                    f"Resume it with `doctodeck-eval run --resume {run.path.as_posix()}`."]  # fmt: skip

    out += ["", "## Summary", ""]
    header = ["Metric", *(_name(run) for run in runs)] + (["Change"] if compare else [])
    rows = []
    for row in ROWS:
        values = [None if s is None else row.value(s) for s in summaries]
        rows.append([row.label, *(_cell(v, row) for v in values)] + ([_change(*values, row)] if compare else []))
    out += _table(header, rows)

    out += ["", "## Fact coverage by document", ""]
    case_ids = list(dict.fromkeys(case_id for run in runs for case_id in run.header.cases))
    rows = []
    for case_id in case_ids:
        cells = []
        for run in runs:
            deck = run.decks.get(case_id)
            if deck is None:
                cells.append("not run")
            else:
                m = deck.metrics
                total = len(m.covered_facts) + len(m.missed_facts)
                note = "" if deck.status == "ok" else f" ({deck.error_code})"
                cells.append(f"{m.fact_coverage:.0%} ({len(m.covered_facts)}/{total}){note}")
        rows.append([case_id, *cells])
    out += _table(["Document", *(_name(run) for run in runs)], rows)

    newest = runs[-1]
    missed = [
        (case_id, deck.metrics.missed_facts) for case_id, deck in newest.decks.items() if deck.metrics.missed_facts
    ]
    hits = [(case_id, hit.rule_id) for case_id, deck in newest.decks.items() for hit in deck.metrics.must_not_hits]
    out += ["", f"## Still missed in {newest.header.run_id}", ""]
    out += [f"- **{case_id}**: {', '.join(facts)}" for case_id, facts in missed] or ["Nothing: every fact covered."]
    if hits:
        out += ["", "Must-not hits: " + ", ".join(f"{case_id}/{rule}" for case_id, rule in hits) + "."]
    return "\n".join(out) + "\n"
