"""The eval runner, report, CLI and panel-3 endpoint (3.4), all on the fake provider: no key, no network.

`Flaky` wraps the fake and fails on a chosen call, to play an outage or a model fault partway through a run.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from doctodeck.app import create_app, get_eval_runs_dir
from doctodeck.evals import cli
from doctodeck.evals.golden import load_golden_set
from doctodeck.evals.report import markdown_report
from doctodeck.evals.runner import (
    EvalSettings,
    RunFileError,
    latest_run,
    read_run,
    resume_eval,
    start_eval,
)
from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest, LlmResult
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline import prompts
from doctodeck.pipeline.prompts import DEFAULT_PROMPTS, PromptSet

CASES = load_golden_set()
NOW = datetime(2026, 11, 28, 9, 30, 0, tzinfo=UTC)
SETTINGS = EvalSettings(provider="fake", prompts=DEFAULT_PROMPTS)


class Flaky:
    """The fake, except that call number `fail_on` (1-based, counting plan and cards calls) raises `code`."""

    def __init__(self, fail_on: int, code: LlmErrorCode, provider: str = "fake") -> None:
        self._inner = FakeLlmClient(provider=provider)
        self.provider, self.model, self.is_real = provider, self._inner.model, False
        self.fail_on, self.code, self.calls = fail_on, code, 0

    async def complete[T](self, request: LlmRequest[T]) -> LlmResult[T]:
        self.calls += 1
        if self.calls == self.fail_on:
            raise LlmError(self.code, f"injected {self.code} on call {self.calls}")
        return await self._inner.complete(request)


def registry(client=None) -> LlmRegistry:
    return LlmRegistry([client or FakeLlmClient()])


async def start(tmp_path: Path, client=None, cases=CASES, settings=SETTINGS, progress=lambda _: None):
    return await start_eval(cases, registry(client), settings, tmp_path, now=lambda: NOW, progress=progress)


def kinds(path: Path) -> list[str]:
    return [json.loads(line)["kind"] for line in path.read_text(encoding="utf-8").splitlines()]


# ---------- A full run ----------


async def test_a_run_writes_a_header_a_line_per_case_and_a_summary(tmp_path):
    run = await start(tmp_path)

    assert run.path == tmp_path / "20261128T093000Z-fake-v1.jsonl"
    assert kinds(run.path) == ["run", "deck", "deck", "deck", "deck", "deck", "summary"]
    header = run.header
    assert (header.provider, header.model, header.prompt_version) == ("fake", "fake-demo", "v1")
    assert header.prompt_fingerprint == DEFAULT_PROMPTS.fingerprint()
    assert header.cases == [case.id for case in CASES]
    assert run.complete and run.stopped is None
    assert run.summary.complete
    assert run.summary.summary.facts_total == 44


async def test_deck_lines_keep_the_cards_and_modules_for_the_judge(tmp_path):
    run = await start(tmp_path)

    deck = run.decks["http-caching"]
    assert deck.status == "ok"
    assert deck.cards and deck.modules
    assert deck.metrics.card_count == len(deck.cards)


async def test_an_unknown_provider_fails_before_any_file_is_written(tmp_path):
    with pytest.raises(LlmError) as caught:
        await start(tmp_path, settings=replace(SETTINGS, provider="nope"))

    assert caught.value.code == LlmErrorCode.UNKNOWN_PROVIDER
    assert list(tmp_path.iterdir()) == []


# ---------- Failures partway ----------


async def test_an_outage_stops_the_run_keeps_finished_cases_and_resume_finishes_it(tmp_path):
    # Calls 1-4 are cases 1 and 2 (plan + cards each); call 5 is case 3's plan.
    run = await start(tmp_path, Flaky(fail_on=5, code=LlmErrorCode.RATE_LIMITED))

    assert kinds(run.path) == ["run", "deck", "deck", "stopped", "summary"]
    assert not run.complete and not run.summary.complete
    assert run.stopped.case_id == CASES[2].id
    assert run.missing == [case.id for case in CASES[2:]]

    resumed = await resume_eval(run.path, CASES, registry(), SETTINGS)

    assert resumed.complete and resumed.stopped is None
    assert set(resumed.decks) == {case.id for case in CASES}
    assert kinds(resumed.path)[-4:] == ["deck", "deck", "deck", "summary"]
    assert resumed.summary.complete


async def test_a_crash_keeps_every_case_finished_before_it(tmp_path):
    def crash_after_two(message: str) -> None:
        if message.startswith(CASES[1].id):
            raise KeyboardInterrupt  # Ctrl+C, or the machine going down

    with pytest.raises(KeyboardInterrupt):
        await start(tmp_path, progress=crash_after_two)

    path = latest_run(tmp_path)
    assert kinds(path) == ["run", "deck", "deck"]  # both on disk before the crash
    assert (await resume_eval(path, CASES, registry(), SETTINGS)).complete


async def test_a_model_fault_is_scored_and_the_run_goes_on(tmp_path):
    # Call 4 is case 2's cards call: the model answers, but not in the schema.
    run = await start(tmp_path, Flaky(fail_on=4, code=LlmErrorCode.INVALID_OUTPUT))

    assert run.complete
    failed = run.decks[CASES[1].id]
    assert (failed.status, failed.error_code) == ("model_error", LlmErrorCode.INVALID_OUTPUT)
    assert failed.cards == [] and failed.metrics.fact_coverage == 0.0
    assert (failed.metrics.answered_calls, failed.metrics.schema_valid_calls) == (2, 1)
    assert run.summary.summary.schema_valid_rate == pytest.approx(9 / 10)


async def test_a_model_fault_on_the_plan_records_a_failed_plan_call(tmp_path):
    run = await start(tmp_path, Flaky(fail_on=1, code=LlmErrorCode.REFUSED), cases=CASES[:1])

    deck = run.decks[CASES[0].id]
    assert deck.status == "model_error" and deck.modules == []
    assert deck.metrics.calls == 1 and deck.metrics.failed_calls == 1


# ---------- Resume safety ----------


async def test_resume_refuses_another_experiment(tmp_path):
    run = await start(tmp_path, Flaky(fail_on=5, code=LlmErrorCode.TIMEOUT))
    edited = PromptSet(
        "v1", lambda data: replace(prompts.plan_request(data), system_prompt="edited"), prompts.cards_request
    )

    with pytest.raises(RunFileError, match="Start a new run"):
        await resume_eval(run.path, CASES, registry(), replace(SETTINGS, prompts=edited))  # same name, new text
    with pytest.raises(RunFileError, match="Start a new run"):
        await resume_eval(
            run.path, CASES, registry(FakeLlmClient(provider="gemini")), replace(SETTINGS, provider="gemini")
        )


async def test_resuming_a_complete_run_adds_nothing(tmp_path):
    run = await start(tmp_path)
    before = run.path.read_text(encoding="utf-8")

    again = await resume_eval(run.path, CASES, registry(), SETTINGS)

    assert again.complete and run.path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("content", "message"),
    [("", "empty"), ("not json\n", "jsonl"), ('{"kind": "run"}\n', "jsonl")],
    ids=["empty", "not-json", "bad-header"],
)
def test_unreadable_run_files_are_reported(tmp_path, content, message):
    path = tmp_path / "broken.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(RunFileError, match=message):
        read_run(path)


async def test_an_unknown_line_kind_is_reported(tmp_path):
    run = await start(tmp_path, cases=CASES[:1])
    with run.path.open("a", encoding="utf-8") as file:
        file.write('{"kind": "mystery"}\n')

    with pytest.raises(RunFileError, match="mystery"):
        read_run(run.path)


def test_latest_run_is_the_newest_file(tmp_path):
    assert latest_run(tmp_path) is None
    assert latest_run(tmp_path / "missing") is None
    for name in ("20261127T100000Z-fake-v1.jsonl", "20261128T090000Z-fake-v1.jsonl", "notes.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    assert latest_run(tmp_path).name == "20261128T090000Z-fake-v1.jsonl"


# ---------- Prompt versions ----------


def test_the_fingerprint_is_stable_and_changes_with_the_prompt_text():
    edited = PromptSet(
        "v1", lambda data: replace(prompts.plan_request(data), system_prompt="edited"), prompts.cards_request
    )

    assert DEFAULT_PROMPTS.fingerprint() == DEFAULT_PROMPTS.fingerprint()
    assert len(DEFAULT_PROMPTS.fingerprint()) == 12
    assert edited.fingerprint() != DEFAULT_PROMPTS.fingerprint()


async def test_the_pipeline_sends_the_prompt_version_it_is_given(tmp_path):
    fake = FakeLlmClient()
    shouty = PromptSet(
        "v2",
        lambda data: replace(prompts.plan_request(data), system_prompt="PLAN v2"),
        lambda data: replace(prompts.cards_request(data), system_prompt="CARDS v2"),
    )

    run = await start(tmp_path, fake, cases=CASES[:1], settings=replace(SETTINGS, prompts=shouty))

    assert [request.system_prompt for request in fake.requests] == ["PLAN v2", "CARDS v2"]
    assert run.header.prompt_version == "v2" and run.path.name.endswith("-fake-v2.jsonl")


# ---------- The report ----------


async def test_a_two_run_report_compares_side_by_side(tmp_path):
    before = await start(tmp_path, Flaky(fail_on=4, code=LlmErrorCode.INVALID_OUTPUT))  # one case lost
    after = await start_eval(CASES, registry(), SETTINGS, tmp_path / "b", now=lambda: NOW)

    text = markdown_report([before, after])

    assert "| Metric | fake v1 | fake v1 | Change |" in text
    coverage = next(line for line in text.splitlines() if line.startswith("| Fact coverage (mean"))
    assert "pts ✅" in coverage  # more coverage is better
    failed = next(line for line in text.splitlines() if line.startswith("| Schema-valid"))
    assert "90%" in failed and "100%" in failed
    assert "0% (0/" in text and "(INVALID_MODEL_OUTPUT)" in text  # the lost case is named, not hidden


async def test_a_report_on_an_incomplete_run_says_how_to_resume(tmp_path):
    run = await start(tmp_path, Flaky(fail_on=5, code=LlmErrorCode.RATE_LIMITED))

    text = markdown_report([run])

    assert "fake v1 (incomplete)" in text
    assert f"stopped at **{CASES[2].id}**: PROVIDER_RATE_LIMITED" in text
    assert "doctodeck-eval run --resume" in text
    assert "not run" in text


async def test_a_worse_must_not_count_is_flagged(tmp_path):
    good = await start(tmp_path, cases=CASES[:1])
    bad = read_run(good.path)
    bad.decks[CASES[0].id] = bad.decks[CASES[0].id].model_copy(
        update={"metrics": bad.decks[CASES[0].id].metrics.model_copy(update={"must_not_hits": []})}
    )

    hits = next(line for line in markdown_report([bad, good]).splitlines() if line.startswith("| Must-not hits"))

    assert "+1 ⚠️" in hits  # more must-not hits is worse


# ---------- The CLI ----------


@pytest.fixture
def in_tmp(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no .env here, and the default cache file lands in tmp
    return tmp_path


def test_cli_run_then_report(in_tmp, capsys):
    assert cli.main(["run", "--provider", "fake", "--runs-dir", "runs", "--cases", "sql-joins,photosynthesis"]) == 0
    out = capsys.readouterr().out
    assert "sql-joins: ok" in out and "# Eval report" in out

    [path] = (in_tmp / "runs").glob("*.jsonl")
    assert read_run(path).header.cases == ["photosynthesis", "sql-joins"]  # golden-set order, not typed order

    assert cli.main(["report", str(path), "--out", "report.md"]) == 0
    assert (in_tmp / "report.md").read_text(encoding="utf-8").startswith("# Eval report")


def test_cli_rejects_unknown_cases_and_providers(in_tmp, capsys):
    assert cli.main(["run", "--cases", "nope", "--runs-dir", "runs"]) == 1
    assert "Unknown case(s): nope" in capsys.readouterr().err

    assert cli.main(["run", "--provider", "nope", "--runs-dir", "runs"]) == 1
    assert "Unknown provider 'nope'" in capsys.readouterr().err


def test_cli_rejects_an_unknown_prompt_version(in_tmp):
    with pytest.raises(SystemExit):  # argparse's choices
        cli.main(["run", "--prompt", "v9"])


# ---------- Panel 3 ----------


async def test_latest_eval_endpoint_shows_the_newest_run(tmp_path):
    await start(tmp_path, Flaky(fail_on=5, code=LlmErrorCode.RATE_LIMITED))
    app = create_app()
    app.dependency_overrides[get_eval_runs_dir] = lambda: tmp_path

    with TestClient(app) as client:
        body = client.get("/api/evals/latest").json()

    assert body["runId"] == "20261128T093000Z-fake-v1"
    assert body["complete"] is False and body["stopped"] == f"{CASES[2].id}: PROVIDER_RATE_LIMITED"
    labels = [row["label"] for row in body["summary"]]
    assert labels[0] == "Fact coverage (mean per document)"
    assert [c["status"] for c in body["cases"]][:3] == ["ok", "ok", "not run"]
    assert "(" in body["cases"][0]["factCoverage"]  # e.g. "62% (5/8)"


def test_latest_eval_endpoint_without_runs_or_with_a_broken_one(tmp_path):
    app = create_app()
    app.dependency_overrides[get_eval_runs_dir] = lambda: tmp_path

    with TestClient(app) as client:
        none = client.get("/api/evals/latest")
        (tmp_path / "20261128T000000Z-fake-v1.jsonl").write_text("garbage\n", encoding="utf-8")
        broken = client.get("/api/evals/latest")

    assert (none.status_code, none.json()["error"]) == (404, "NO_EVAL_RUNS")
    assert (broken.status_code, broken.json()["error"]) == (500, "EVAL_RUN_UNREADABLE")
