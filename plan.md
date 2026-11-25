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
  - [ ] 2.2 Gemini: `GeminiClient` over `httpx` (key in the `x-goog-api-key` header), JSON Schema → Gemini dialect translator, retries on 408/429/500/502/503/504 and on network errors and timeouts, backoff + jitter + capped `Retry-After`, no retry on other 4xx or on cancellation, handling of safety blocks, `MAX_TOKENS` and thought parts. Tested with `respx`.
  - [ ] 2.3 Claude: `ClaudeClient` over the `anthropic` SDK, structured outputs from the same Pydantic models, `stop_reason` handling, SDK retries, refusal fallback only on models that support it.
  - [ ] 2.4 Side by side: one sample through both providers on the page (cards, flags, tokens, latency). This is the hand-off to Layer 3.
- Python-specific design (2.1): each `LlmRequest` carries its Pydantic output type, so every client returns a *validated* model (`LlmResult[T]`) instead of raw JSON, the way the Anthropic SDK's `messages.parse` works; the same class is the JSON Schema sent to the model. Output models are strict on shape (every field required) and loose on values (no `ge`/`le`): the schema guarantees shape, the planner and generator repair meaning. `LlmClient` is a `Protocol` (structural typing, no base class). Pipeline steps are FastAPI sub-dependencies (`get_planner` depends on `get_llm_registry` and `get_pipeline_options`); request-body validation errors are reshaped to `400 INVALID_REQUEST` so every error stays `{error, message}`.
- Note for 2.2: `model_json_schema()` nests models under `$defs` with `$ref`; the Gemini translator must inline them.
- Deliberate differences from AnkiX (as in the C# kata): composition over a base class; plan repair records every fix; cards carry `module_index`; truncation is reported, not silent.
- Lab command: `uv run pytest -k layer2`, then with keys set, generate from `samples/sample.md` once per provider on the page
- [ ] CLI challenge
- [ ] Study Group deck & drill (exercise ids: )
- [ ] Azimuth checkpoint
- [ ] Field report (draft: docs/publications/layer-2.md, URL: )

### Layer 3: Evals (new; AnkiX has none)
- Anchor: none in AnkiX. `BulkCardGenerationService.cs` `EvaluateCardQuality` is the only quality signal there, and it judges one card at a time with no reference document. This whole layer is a Pay It Back candidate.
- Milestones:
  - [ ] 3.1 Golden set: 4–6 short documents in `evals/golden/` (Markdown and PDF, different domains, one awkward one: tables or code-heavy), each with a hand-written `expect.yaml`: the key facts a good deck must cover, things it must not contain, and the expected module count range.
  - [ ] 3.2 Deterministic metrics: schema-valid rate, gate flag rate by rule, key-fact coverage (fuzzy match of facts against card text), near-duplicate rate, section coverage, cards per 1k words, tokens, cost, latency p50/p95. Pure functions, unit-tested on fixed card sets.
  - [ ] 3.3 LLM-as-judge: a rubric (faithful to the source, atomic, unambiguous, worth remembering) scored 1–5 per card under a Pydantic schema. The judge model differs from the generator (self-preference bias). **Calibrate**: you label about 30 cards by hand and record the judge's agreement with you; the judge isn't trusted until it is measured.
  - [ ] 3.4 Runner + report: `uv run doctodeck-eval --provider gemini --prompt v2` runs the golden set (cached responses so reruns are free), writes `evals/runs/<timestamp>.jsonl` and a Markdown comparison table. Use it once for real: change one prompt, show the before and after, and record the numbers in the pitch above. Page panel 3: latest report.
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
- 2026-11-24: No vector search or RAG layer. AnkiX does no retrieval, and the kata copies how AnkiX works; retrieval stays out of scope, stretch goals included.

## Sync log
- 2026-11-24: plan drafted and locked; repo initialised locally with Layer 1 (not yet pushed)
- 2026-11-25: Layer 1 committed (`7d7a383`, branch `main`, `.gitattributes` keeps PDFs binary); 2.1 built; still local only, no GitHub repo yet
