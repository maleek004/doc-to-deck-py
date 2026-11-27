---
kata: doc-to-deck-py
status: locked           # draft | locked | on-hold | complete
pace: fast               # full pre-verified code per milestone; decks, drills and articles may trail by one layer
ankix-target: >-
  src/backend/AnkiX.Api/Services/Ai/DocumentParserService.cs#L13-L15, #L119-L163, #L165-L365, #L599-L720;
  src/backend/AnkiX.Api/Services/Ai/StructuredLlmClientBase.cs#L281-L412;
  src/backend/AnkiX.Api/Services/Ai/GeminiClient.cs#L61-L210, #L212-L320;
  src/backend/AnkiX.Api/Services/Ai/ClaudeClient.cs#L37-L84, #L86-L165, #L167-L220, #L222-L270;
  src/backend/AnkiX.Api/Extensions/AiModuleExtensions.cs#L10-L90;
  src/backend/AnkiX.Api/Services/Ai/GeminiSchemas.cs#L172-L295;
  src/backend/AnkiX.Api/Services/Ai/DeckDecompositionService.cs#L22-L102;
  src/backend/AnkiX.Api/Services/Ai/BulkCardGenerationService.cs#L42-L257, #L259-L294;
  src/backend/AnkiX.Api/Services/AiCreditService.cs#L40-L177, #L179-L250;
  src/backend/AnkiX.Api/Constants/AiPricingTiers.cs#L5-L44;
  src/backend/AnkiX.Api/Controllers/DocumentsAiController.cs#L320-L444
locked: 2026-11-24       # anchors checked against ankiX master 0443924
study-group: not created
predecessor: ../doc-to-deck-pipeline (C#, on hold after milestone 2.2)
---

# doc-to-deck-py: upload a document, get a measured, cost-tracked flashcard deck

## Why this kata
AnkiX turns an uploaded Markdown or PDF file into flashcards. It splits the file into sections, asks an LLM
(Gemini or Claude) to plan thematic modules, asks again for atomic cards under a strict JSON schema, filters
every card through a rule-based quality gate, and charges credits only when the call succeeds. This kata rebuilds
that pipeline in Python, the language AI engineering interviews run in, and adds what AnkiX lacks: an **eval
harness** that shows whether a prompt or provider change made the cards better or worse. The interview story
goes from "I integrated an LLM" to "I built every stage, I can say what each one prevents, and I measured it."

## The 60-second interview pitch (fill in the numbers as layers land)
- Problem: turn arbitrary documents into good flashcards, reliably and within a cost budget.
- Pipeline: parse → plan → generate (structured output, two providers) → quality gate → charge.
- Reliability: hand-rolled retry with backoff, jitter and `Retry-After` for Gemini, versus the SDK's built-in retries for Claude; schema validation; repair of the model's plan.
- Measurement: golden set of __ documents; prompt v2 raised key-fact coverage from __% to __% and cut the flag rate by __ points; Claude vs Gemini: __ vs __ quality score, at __ vs __ cents per deck, p95 latency __ vs __ s.
- Cost: every call logged with tokens and cost; the daily budget cap; an atomic credit deduction proven with 20 concurrent requests.

## Shape of the app
- Python 3.12+, managed with `uv`. Package `doctodeck/` (src layout), tests in `tests/`.
- **FastAPI** API that also serves `static/index.html` (one unstyled page, vanilla JS, no build step; one panel per layer).
- **Pydantic v2** models are the single source of truth for the card and plan schemas. `model_json_schema()` produces the JSON Schema for Claude, and a small hand-written translator produces Gemini's OpenAPI subset.
- **LLM providers** behind one `LlmClient` `Protocol` (async), chosen per request (`provider=fake|gemini|claude`):
  - **Gemini**: hand-rolled REST over `httpx.AsyncClient`, mirroring AnkiX's `GeminiClient`. You write the retry loop and the response parsing yourself. No `tenacity`; name it in the interview as the production choice.
  - **Claude**: the official `anthropic` Python SDK with structured outputs, default `claude-haiku-5-5` (matching AnkiX), `stop_reason` checked (`refusal`, `max_tokens`) before parsing, SDK retries left on. Contrast the SDK's retries with your Gemini loop.
  - Prompts live once in a shared module; each provider only "sends prompt + schema, returns parsed JSON + usage".
- **No LangChain or LlamaIndex.** Deliberate: the point is to own every stage.
- Storage: SQLite through the stdlib `sqlite3` module with raw SQL, so the atomic `UPDATE ... WHERE balance >= ?` stays visible.
- Tests: `pytest` + `pytest-asyncio`, `respx` for faking `httpx` traffic. **Every test runs on `FakeLlmClient` or `respx`; no test touches the network or a key.** Live runs (evals against real providers) are opt-in through `uv run doctodeck-eval ...` and never part of `pytest`.
- Keys: `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` from env or a git-ignored `.env`. A provider with no key falls back to the fake, as AnkiX does, and the dropdown shows "(fake)".
- Tooling: `ruff` (lint + format), `pyright` in basic mode.

## Study Group scope
1. Chunking as context shaping: Markdown heading detection with code-fence state; PDF line extraction with repeated header and footer removal.
2. Schema-constrained output behind a provider abstraction: one Pydantic model rendered as JSON Schema (Claude) and as Gemini's OpenAPI subset; defensive parsing of each provider's response.
3. Validating model output against reality: clamp counts, repair the plan (out-of-range or forgotten sections), rule-based card flags.
4. Transient-failure retry in asyncio: exponential backoff + jitter, `Retry-After`, a timeout per attempt, never retrying a cancellation (`asyncio.CancelledError`).
5. Evals: golden sets, deterministic metrics vs LLM-as-judge, judge calibration against human labels, comparing variants.
6. Cost control: token accounting, pricing tables, budget caps, atomic compare-and-swap deduction, charging only on success.

## Layers

### Layer 1: Ingest (Markdown + PDF → outline)
- Anchor: `DocumentParserService.cs` (size/page limits, magic-byte check, PDF sections and header/footer noise, Markdown heading regex with code-fence state)
- Builds: `parse_document(bytes, filename) -> ParsedDocument(sections, clean_text, word_count, page_count)`; size and page caps; PDF magic-byte check; `pdfplumber` line extraction (it returns real lines, so the heading bug found in AnkiX stays fixed); `POST /api/parse`; page panel 1.
- Port the C# kata's test cases one for one: same samples, same expected outlines. Budget: about one day.
- Lab command: `uv run pytest -k layer1` then `curl -F file=@samples/sample.pdf http://localhost:8000/api/parse`
- [x] CLI challenge (2026-11-24: Markdown + PDF parser, input guards, `POST /api/parse`, page panel 1; 43 tests green, ruff clean; lab command returns 5 sections from the 3-page sample with headers, footers and page numbers stripped; checked in a browser, including an uploaded heading that tries to inject HTML and a fake `.pdf`)
- Python-specific notes: test PDFs are written by hand (`tests/pdf_builder.py`, no PDF library), so the format itself is part of the lesson; parsing runs in `run_in_threadpool` so it never blocks the event loop; limits are swapped in tests with `app.dependency_overrides`.
- Known gap (Python-specific; Kestrel caps bodies at 30 MB by default, uvicorn does not): uvicorn has no request-body limit, so Starlette spools the whole upload to a temp file before the parser's 25 MB check runs. The parser's check protects memory and parsing time, not disk or bandwidth; in production the limit belongs in the reverse proxy (e.g. nginx `client_max_body_size`) or an ASGI middleware that checks `Content-Length` and counts streamed bytes.
- Test-client warning: Starlette now warns that `TestClient` on `httpx` is deprecated in favour of `httpx2`. Left as is for now; revisit before Layer 2, which uses `httpx` for the Gemini client.
- [ ] Study Group deck & drill (exercise ids: )
- [ ] Azimuth checkpoint
- [ ] Field report (draft: docs/publications/layer-1.md, URL: )

### Layer 2: Plan → generate with Gemini or Claude → quality gate
- Anchor: `StructuredLlmClientBase.cs` (shared prompts); `GeminiClient.cs` (request, retry loop, transient codes, parse, safety, tokens); `ClaudeClient.cs` (effort, structured outputs, stop reasons, schema conversion); `AiModuleExtensions.cs` (provider selection, fake fallback); `GeminiSchemas.cs`; `DeckDecompositionService.cs` (output validation); `BulkCardGenerationService.cs` (`EvaluateCardQuality`)
- Milestones:
  - [x] 2.1 Contract + fake (2026-11-25: 91 tests green (48 new), ruff clean; parse → plan → generate checked in a browser on the fake, including an HTML-injection heading and the no-modules error): `LlmClient` Protocol returning `LlmResult(data, usage{prompt_tokens, completion_tokens, provider, model, latency_ms})`; Pydantic `DeckPlan` / `CardBatch` (built as `PlanOutput` / `CardsOutput`); shared prompts; scripted `FakeLlmClient` (canned JSON, injected 429/503/timeout/refusal); `DeckPlanner` with plan repair; `CardGenerator` with `module_index` mapping and a truncation flag; `CardQualityGate`; `/api/providers`, `/api/plan`, `/api/generate`; page panel 2.
  - [x] 2.2 Gemini (2026-11-27: 128 tests green (37 new), ruff clean; live run on `samples/sample.md` with `gemini-2.5-flash`: 1 module, 6 cards, plan 1.3 s, cards 3.7 s): `GeminiClient` over `httpx` (key in the `x-goog-api-key` header), JSON Schema → Gemini dialect translator, retries on 408/429/500/502/503/504 and on network errors and timeouts, backoff + jitter + the server's wait (`Retry-After` header or the body's `RetryInfo`), failing at once when it is longer than the cap, no retry on other 4xx or on cancellation, handling of safety blocks, `MAX_TOKENS` and thought parts. Tested with `respx`.
  - [x] 2.3 Claude (2026-11-27: 147 tests green (19 new), ruff clean; live run pending: no `ANTHROPIC_API_KEY` in `.env` yet): `ClaudeClient` over the `anthropic` SDK, structured outputs from the same Pydantic models, `stop_reason` handling, SDK retries, refusal fallback only on models that support it.
  - [ ] 2.4 Side by side: one sample through both providers on the page (cards, flags, tokens, latency). This is the hand-off to Layer 3. **Deferred (2026-11-27) until Layer 3's response cache exists** (3.0), so repeat runs cost no free-tier quota.
- Python-specific design (2.1): each `LlmRequest` carries its Pydantic output type, so every client returns a *validated* model (`LlmResult[T]`) instead of raw JSON, the way the Anthropic SDK's `messages.parse` works; the same class is the JSON Schema sent to the model. Output models are strict on shape (every field required) and loose on values (no `ge`/`le`): the schema guarantees shape, the planner and generator repair meaning. `LlmClient` is a `Protocol` (structural typing, no base class). Pipeline steps are FastAPI sub-dependencies (`get_planner` depends on `get_llm_registry` and `get_pipeline_options`); request-body validation errors are reshaped to `400 INVALID_REQUEST` so every error stays `{error, message}`.
- Python-specific design (2.2): `to_gemini_schema()` inlines Pydantic's `$defs`/`$ref`, capitalises types and drops `title`/`default`/`additionalProperties` keywords (but not properties *named* `title`), and refuses recursive models. Each attempt runs under `asyncio.timeout` (one overall deadline; httpx's timeouts are per phase); an outside cancellation stays `CancelledError` and is never caught or retried. One `httpx.AsyncClient` is opened and closed in the app's `lifespan` and shared by the registry. Thinking tokens (`thoughtsTokenCount`) are counted as completion tokens, since they bill at the output rate. `tests/conftest.py` deletes provider keys from the environment for every test (the C# `TestAppFactory` equivalent). `.env` is loaded by uvicorn in `main()` only.
- Live-run findings (2.2):
  - 2026-11-25: every flash model (3.8, 3.7, `flash-latest`) answered 503 "high demand" to a two-word prompt. The client behaved as designed (4 attempts, then `PROVIDER_RATE_LIMITED` → our 503), but the error hid Gemini's own reason. Fix: 429 (quota: waiting may not help) and 503 (overload: it passes) now get separate messages carrying Gemini's text.
  - 2026-11-27: on `gemini-2.5-flash` the cards call timed out 4 × 30 s. The thinking setting only covered Gemini 3 (`thinkingLevel`); 2.5 uses `thinkingBudget`, so it thought without limit (821 output tokens for a tiny plan). Setting `thinkingBudget: 0` for 2.5 Flash: 80 tokens, plan 1.3 s, cards 3.7 s. Pay It Back candidate: check whether AnkiX's Gemini client has the same gap on 2.5 models.
  - 2026-11-27 (the "504s"): a 30-card generation from a 21,000-character document timed out at 30 s per attempt, and each retry regenerated everything. The free tier allows 5 requests per minute and **20 per day** per model, so the retries ended in 429s and then the daily limit. Gemini puts the wait in the 429's body (`google.rpc.RetryInfo.retryDelay`, e.g. `"18s"`), not in a `Retry-After` header, so the client retried after 0.5/1/2 s, which was certain to fail. Fixes: read `retryDelay` from the body; when the server asks for longer than `max_delay` (10 s), fail at once and say when to retry, in readable units, naming a daily limit as such; `attempt_timeout` 30 → 90 s; timeouts retried only once (`max_timeout_retries`), since a slow request is likely slow again and every attempt spends quota. 157 tests green (10 new). Live re-run not possible today: `gemini-2.5-flash` was out of daily requests (the fixed client stopped after 1 attempt: "retry in about 3.5 h"), and `gemini-3.8-flash` was answering 503 "high demand", with 2 × 90 s timeouts on the plan call.
  - What this means for 2.4: on the free tier one full run (1 plan + 1 cards call) costs 2 of the 20 daily requests, so the comparison is ~10 runs a day per model, paced at least 12 s apart. Options: compare on the small sample first; use a paid tier or another model for the big document; or wait for Layer 3's response cache, which makes re-runs free.
  - Input for Layer 3: 5 of 6 live cards were flagged `ANSWER_LEAK` because the answers restate the question's long words ("LLM API failures are…"). The gate may be too strict; measure it against human labels in 3.3 before changing it.
- Python-specific design (2.3): `anthropic` 2.14 (on `httpx2`, so it keeps its own connection pool and `respx` can't see it; tests give the real SDK a `DefaultAsyncHttpxClient` over `httpx2.MockTransport`, with `retry-after-ms: 1` so the SDK's real retries run fast). Calls `messages.create()` with `output_config.format` (schema from the SDK's public `transform_schema`, which adds `additionalProperties: false`; `$ref` is accepted, no inlining) instead of `messages.parse()`: `parse()` validates text while building the response, so a refusal or `max_tokens` cut-off would raise a Pydantic error before `stop_reason` could be read. Stop reasons are checked first (`refusal` → REFUSED with its category, `max_tokens` → TRUNCATED). No `temperature` (current models reject non-default sampling) and `effort: low`. Retries are the SDK's (`max_retries=3`); claude.py only maps the final typed error, most specific first (`APITimeoutError` before `APIConnectionError`; 529 `OverloadedError` → RATE_LIMITED). Server-side refusal fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`) only for models that support it; Haiku 5.5 has none. `usage.model` reports the model that answered; prompt tokens include cache reads and writes. The registry's `aclose()` closes the SDK at shutdown. Side effects: installing `httpx2` removed Starlette's TestClient deprecation warning; importing `anthropic` costs about 7 s, most of the suite's 10 s.
- Possible simplification to revisit: newer Gemini API versions also accept standard JSON Schema in `responseJsonSchema`, which would make the translator unnecessary; kept the translator to mirror AnkiX.
- Deliberate differences from AnkiX (as in the C# kata): composition over a base class; plan repair records every fix; cards carry `module_index`; truncation is reported, not silent.
- Lab command: `uv run pytest -k layer2`, then with keys set, generate from `samples/sample.md` once per provider on the page
- [ ] CLI challenge
- [ ] Study Group deck & drill (exercise ids: )
- [ ] Azimuth checkpoint
- [ ] Field report (draft: docs/publications/layer-2.md, URL: )

### Layer 3: Evals (new; AnkiX has none)
- Anchor: none in AnkiX. `BulkCardGenerationService.cs` `EvaluateCardQuality` is the only quality signal there, and it judges one card at a time with no reference document. This whole layer is a Pay It Back candidate.
- Milestones:
  - [x] 3.0 Response cache (2026-11-27: 169 tests green (12 new), ruff clean; not yet checked live, as both Gemini models were blocked today; moved to the front on 2026-11-27, so 2.4 and every later live run spend quota only once): a `CachingLlmClient` wrapper over any real `LlmClient`, keyed by a hash of provider, model, client settings, prompts and output schema; stored in SQLite (`LLM_CACHE` path, off when unset); only successful, validated answers are stored; a hit reports the original usage with `cached: true`. Then do 2.4.
  - [x] 3.1 Golden set (2026-11-27: 217 tests green (48 new), ruff clean; 5 cases: HTTP caching, photosynthesis, SQL joins (awkward), compound interest, Roman Republic (PDF); 44 facts): 4–6 short documents in `evals/golden/` (Markdown and PDF, different domains, one awkward one: tables or code-heavy), each with a hand-written `expect.toml` (TOML, not YAML: see the design note): the key facts a good deck must cover, things it must not contain, and the expected module count range.
  - [x] 3.2 Deterministic metrics (2026-11-27: 248 tests green (31 new), ruff clean; fake-provider baseline on the golden set: 20/44 facts, 47% mean coverage, 5 must-not hits, module count never in range): schema-valid rate, gate flag rate by rule, key-fact coverage (fuzzy match of facts against card text), near-duplicate rate, section coverage, cards per 1k words, tokens, cost, latency p50/p95. Pure functions, unit-tested on fixed card sets.
  - [ ] 3.3 LLM-as-judge: a rubric (faithful to the source, atomic, unambiguous, worth remembering) scored 1–5 per card under a Pydantic schema. The judge model differs from the generator (self-preference bias). **Calibrate**: you label about 30 cards by hand and record the judge's agreement with you; the judge isn't trusted until it is measured.
  - [ ] 3.4 Runner + report: `uv run doctodeck-eval --provider gemini --prompt v2` runs the golden set (through the 3.0 cache, so reruns are free), writes `evals/runs/<timestamp>.jsonl` and a Markdown comparison table. Use it once for real: change one prompt, show the before and after, and record the numbers in the pitch above. Page panel 3: latest report.
- Python-specific design (3.0): `llm/cache.py`. `CachingLlmClient` is a decorator over any `LlmClient` (same provider, model and `is_real`); the registry wraps real clients only, when `LLM_CACHE` is set. Key: SHA-256 of canonical JSON (`sort_keys`, fixed separators) of `CACHE_VERSION`, provider, model, the client's `settings` (Gemini: temperature + thinking; Claude: effort + max_tokens), both prompts and the output schema. Stored: the validated answer and the original `LlmUsage`; a hit returns that usage with `cached: true` (new field on `LlmUsage`, shown on the page). Errors and wrong-shape answers are never stored; an entry that no longer validates is deleted and the provider asked. SQLite via stdlib `sqlite3` in `asyncio.to_thread`, one short connection per operation, WAL mode. Not done: single-flight for identical concurrent misses, a TTL, a per-request bypass.
- Python-specific design (3.1): `doctodeck/evals/golden.py`. One folder per case: `document.md` or `document.pdf`, `expect.toml`, and for PDFs a `source.txt` that `python -m tests.layer3.build_golden_pdfs` turns into the PDF (no timestamps, so a test checks the committed bytes are up to date). TOML via stdlib `tomllib` instead of YAML: no dependency, no type surprises. Strict Pydantic models (`extra="forbid"`, kebab-case ids, unique ids, min <= max, at least 3 facts). Facts and must-nots are matched by terms: all must appear; `|` alternatives; trailing `*` prefix; case-insensitive whole words via `(?<!\w)…(?!\w)` (works at punctuation edges, unlike `\b`); any whitespace. Tests check that every fact and must-not is in its own parsed source, the module range fits the outline, the PDF's headings and page furniture, and that `#` in a code block stays text. must_not holds source material that shouldn't become cards (boilerplate, setup, out of scope), not misconceptions: term matching can't tell a wrong claim from a card correcting it; that's the judge's job (3.3).
- Python-specific design (3.2): `doctodeck/evals/metrics.py`, pure functions. `DeckRun` (case, parsed document, planned modules, cards, `CallRecord`s, truncated, repairs) → `deck_metrics()` → `DeckMetrics`; `summarize()` → `SetSummary`. Both are ApiModels, so 3.4 writes them as JSON lines. A fact is covered only when one card holds all its terms (atomic cards). Coverage is reported macro (mean per case) and micro (facts pooled); flag, score and duplicate rates are pooled over cards, and list every key even at 0.0. Near-duplicates: content-word Jaccard of front + back >= 0.6 against any earlier card in the deck (the gate: fronts, one batch, 0.80). Section coverage goes through the plan, since cards don't name sections. Schema-valid rate counts only calls that got an answer. Unknowns stay unknown: no answered calls → `None` rate; an unpriced model → `None` cost (prices are passed in as `TokenPrice`; the table is Layer 4's). Latency: a deck's calls are summed (sequential); p50/p95 by nearest rank, which always returns an observed value. Not done: semantic coverage (embeddings) and faithfulness, which are 3.3's judge.
- Failure injection: a judge response that fails the schema is recorded as a judge error, never as a score of 0; a provider outage partway through a run leaves the finished documents' results intact and resumable.
- Lab command: `uv run pytest -k layer3` (harness on the fake), then `uv run doctodeck-eval --provider fake --prompt v1`
- [ ] CLI challenge
- [ ] Study Group deck & drill (exercise ids: )
- [ ] Azimuth checkpoint
- [ ] Field report (draft: docs/publications/layer-3.md, URL: )

### Layer 4: Cost accounting and guardrails (slimmed from the C# kata's billing layer)
- Anchor: `AiPricingTiers.cs` (prices; ingest fee on the text actually sent); `AiCreditService.cs` (atomic deduction + ledger, idempotent grant); `DocumentsAiController.cs` (pre-flight → call → deduct on success, charge at generation, no cancellation for the charge)
- Builds:
  1. Usage ledger: one row per LLM call (provider, model, prompt/completion tokens, cost computed from a price table, latency, outcome). Panel 4 shows the ledger and the cost per deck.
  2. Credit wallet: a single local wallet (no invite links, no admin page). Pre-flight check → LLM call → atomic `UPDATE wallet SET balance = balance - ? WHERE id = ? AND balance >= ?` only on success. The charge runs under `asyncio.shield` so a client disconnect can't skip it (Python's version of AnkiX's `CancellationToken.None`).
  3. Guardrails: an input token budget per request (truncate and report, as in Pay It Back #3); a global daily spend cap that switches real providers off while fake mode keeps working.
- Failure injection: 20 concurrent generate calls on a balance that covers 5 → exactly 5 succeed and the balance never goes negative; a provider error charges nothing; a client disconnect after a successful call still charges; reaching the daily cap refuses real providers but the fake still answers.
- Lab command: `uv run pytest -k layer4`
- [ ] CLI challenge
- [ ] Study Group deck & drill (exercise ids: )
- [ ] Azimuth checkpoint
- [ ] Field report (draft: docs/publications/layer-4.md, URL: )

## Stretch (only if time allows; not required to finish)
- Claude prompt caching on the shared system prompt, measured in the Layer 4 ledger (cached vs uncached input tokens).
- Fan out generation per module with `asyncio.gather` and a concurrency limit (`asyncio.Semaphore`), merged afterwards, instead of truncating the source (Pay It Back #3).
- OpenTelemetry spans around each pipeline stage.

## Parked Azimuth Questions
<!-- Raised in expedition mode; answered in home mode against live ankiX. -->

## Pay It Back to AnkiX
Carried over from the C# kata (`../doc-to-deck-pipeline/plan.md`); tick each in only one of the two plans.
- [ ] PDF outline detection never fires: `DocumentParserService.cs` splits PdfPig's `page.Text`, which has no line breaks. Fix: `ContentOrderTextExtractor.GetText(page)`, plus a test asserting section titles.
- [ ] Cards are assigned to modules round-robin, ignoring the model's intent (`BulkCardGenerationService.cs`). Fix: `moduleIndex` in the schema.
- [ ] Batch generation silently drops source text past `MaxIngestSourceChars` (`StructuredLlmClientBase.cs`, `AiPricingTiers.cs`). Fix: fan out per module, or at least surface a `truncated` flag.
- [ ] New: AnkiX has no offline eval of its prompts. A prompt or model change ships unmeasured. Fix: port the Layer 3 golden set and runner (or call the kata's runner against AnkiX's prompts).

## Decisions
- 2026-11-24: Rebuild doc-to-deck in Python instead of finishing the C# kata, because AI engineering interviews run in Python (Pydantic, async, SDKs, evals). The C# kata goes on hold after milestone 2.2 (87 tests green) and stays available as "the same system in .NET".
- 2026-11-24: Replace the C# kata's billing layer (invite links, per-link wallets, admin page) with an **evals** layer (L3) and a slim **cost and guardrails** layer (L4). Interviewers rarely ask about access links; they always ask how you know the output is good and what it costs. The atomic credit deduction stays, as the one concurrency story.
- 2026-11-24: Stack: Python 3.12+, uv, FastAPI, Pydantic v2, httpx, anthropic SDK, pdfplumber, stdlib sqlite3, pytest + pytest-asyncio + respx, ruff. No LangChain, LlamaIndex or tenacity: every stage is hand-built so it can be explained.
- 2026-11-24: Front end: one plain `index.html` with vanilla JS served by FastAPI, as in the C# kata.
- 2026-11-24: Pace: the navigator supplies complete, pre-verified code per milestone (tests + implementation) plus a flashcard deck; the user reviews, runs and asks questions, and follow-up questions become flashcards.
- 2026-11-24: Its own sibling repo `ankix-subprojects/doc-to-deck-py`, public; keys and `.env` never committed.
- 2026-11-24: Layer 1 is a fast port: same samples and test cases as the C# kata. Time goes to Layers 2–3.
- 2026-11-27: Defer 2.4 (Gemini vs Claude side by side) until Layer 3's response cache lands. The Gemini free tier (20 requests per model per day, 2 per run) leaves too few runs, and a cache makes re-runs free. The cache moves from 3.4 to a new first milestone, 3.0; 2.4 follows it.
- 2026-11-24: No vector search or RAG layer. AnkiX does no retrieval, and the kata copies how AnkiX works; retrieval stays out of scope, stretch goals included.

## Sync log
- 2026-11-24: plan drafted and locked; repo initialised locally with Layer 1 (not yet pushed)
- 2026-11-25: Layer 1 committed (`7d7a383`, branch `main`, `.gitattributes` keeps PDFs binary); 2.1 built and pushed to github.com/maleek004/doc-to-deck-py (`fb0c036`)
- 2026-11-27: 2.2 finished (error messages, 2.5 thinking budget, live run) and committed
- 2026-11-27: 2.2 pushed (`52bd862`); 2.3 built (Claude client), live run pending a key
- 2026-11-27: 2.3 pushed (`0ed4d29`); Gemini quota and timeout fix (body `retryDelay`, fail fast on long waits, 90 s deadline, one timeout retry), not committed
- 2026-11-27: 2.4 deferred until the Layer 3 response cache; next is Layer 3
- 2026-11-27: 3.0 response cache built (`LLM_CACHE`), not committed; next is 2.4 once Gemini has quota again (and with a Claude key)
- 2026-11-27: 3.1 golden set built (5 cases, 44 facts), not committed; next is 3.2 deterministic metrics
- 2026-11-27: 3.1 pushed (`a451f88`); 3.2 deterministic metrics built, not committed; next is 3.3 (judge, needs live calls) or 3.4 (runner, buildable on the fake)
