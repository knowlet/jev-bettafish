# System One / decision model integration

## Endpoint

System One defaults to the **OpenRouter Decisions API** with the
`inception/mercury-decide:free` model. Configure the provider as a matching key,
endpoint, and model:

| Role | Canonical env | Default when unset |
|---|---|---|
| key | `DECISION_MODEL_API_KEY` | — (fail-open to legacy LLM path) |
| url | `DECISION_MODEL_BASE_URL` | `https://openrouter.ai/api/alpha/decisions` |
| model | `DECISION_MODEL_NAME` | `inception/mercury-decide:free` |

Despite its name, `DECISION_MODEL_BASE_URL` is the **complete POST endpoint**.
Do not use the OpenAI-compatible `/api/v1` base URL or the TypeSafe SDK's
`/api` base URL here. The client sends only the explicitly configured decision
key to this endpoint and does not follow HTTP redirects. Custom endpoints are
supported; their key and model must belong to that service.

### Breaking migration

There are no legacy environment-variable fallbacks:

| Previous configuration | Required action |
|---|---|
| `TYPESAFE_API_KEY` with a TypeSafe-issued key | Obtain an **OpenRouter-issued** key for the default endpoint and set `DECISION_MODEL_API_KEY`. Renaming the old credential does not make it valid for OpenRouter. |
| `OPENROUTER_API_KEY` with an OpenRouter-issued key | Set that OpenRouter credential as `DECISION_MODEL_API_KEY`; the old variable name is ignored. |
| `SYSTEM_ONE_URL` / `SYSTEM_ONE_MODEL` | Configure the matching `DECISION_MODEL_BASE_URL` / `DECISION_MODEL_NAME`, or use the new OpenRouter defaults. |
| `MODEL_API_KEY` / `MODEL_BASE_URL` / `MODEL_NAME` in Actions or headless runs | Move the matching LLM provider configuration to `LLM_MODEL_API_KEY` / `LLM_MODEL_BASE_URL` / `LLM_MODEL_NAME`. |

Decision credentials are never borrowed from LLM, engine, OpenRouter legacy,
or TypeSafe legacy variables. A deployment with only the old decision variable
names falls back to the LLM path without making a decision request. A successful
research run alone therefore does not demonstrate that migration is complete.
If retaining the direct TypeSafe service, explicitly configure its matching
endpoint, model, and credential under all three canonical `DECISION_MODEL_*` names.

### Configuration precedence

The client resolves explicit constructor arguments first, then the process
environment, then the selected `.env`, then built-in defaults. The selected
file is the working-directory `.env` if present; otherwise it is the project
root's `.env`. The two files are not merged. Empty process values still override
the file; an empty or whitespace-only decision key disables the client. An
explicit `api_key=""` also disables it instead of borrowing an ambient key.

System One reads `.env` without exporting its contents into the process
environment. Creating a decision client therefore cannot promote unrelated
engine credentials above later `.env` edits. Restart the process after changing
decision configuration; the shared client holds its resolved configuration.
The key remains optional in both client and workflow preflight.

`LLM_MODEL_*` supplies GitHub Actions/headless defaults only. The direct
headless runner loads the same selected `.env` before preflight and chooses
the QueryEngine provider configuration as a group. If any `QUERY_ENGINE_*`
key, URL, or model value is configured, supply that group's own key and model;
the URL may be empty to use the OpenAI SDK default. It never fills a partial
engine group with another provider's generic values. If all three engine fields
are empty or absent, it uses the complete `LLM_MODEL_*` group. Clear the sample
QueryEngine URL/model in `.env.example` when choosing generic headless defaults.
Local and Docker `app.py` users must configure each engine separately.

The OpenAI SDK's default also honors a process-level `OPENAI_BASE_URL`.
Set the canonical LLM/engine base URL explicitly to pin the intended provider;
System One does not export an `OPENAI_BASE_URL` from `.env` into the process.

The request/response schema (`state` + typed `questions` -> `answers`) is the
System One schema; mercury-decide serves it natively.

## Audit result

BettaFish mixes open-ended generation with bounded judgment. System One is a
good replacement for the latter, not for arbitrary prose generation.

| Decision site | Fit for the decision model | Status | Why |
| --- | --- | --- | --- |
| QueryEngine search-tool routing | Strong | Implemented | Closed Tavily tool set |
| QueryEngine reflection continue/stop | Strong | Implemented | Value-of-information Noul gate |
| Query/Media evidence triage | Strong | Implemented | Batched relevance/value/novelty Scores |
| MediaEngine tool routing | Strong | Implemented | Provider-aware closed tool set |
| MediaEngine reflection continue/stop | Strong | Implemented | Value-of-information Noul gate |
| InsightEngine DB routing + parameters | Strong | Implemented | Tool/platform/time Choice + sentiment Noul in one fan-out |
| InsightEngine reflection continue/stop | Strong | Implemented | Value-of-information Noul gate |
| Insight keyword expansion | Strong | Implemented gate | Noul skips keyword-generation LLM for concrete queries |
| Forum host intervention | Strong | Implemented gate | Noul + reason Choice before host LLM |
| ReportEngine template selection | Strong | Implemented | Closed set of local templates |
| ReportEngine word budget | Strong | Implemented | Batched Scores + deterministic allocator replace planner LLM |
| ReportEngine SWOT/PEST applicability | Strong | Implemented | Batched Noul judgments; code enforces max one each |
| Search-query generation | Poor | Keep LLM | Arbitrary text generation |
| First/reflection summaries | Poor | Keep LLM | Evidence synthesis |
| Report structure/title/hero copy | Poor | Keep LLM | Open-ended structured generation |
| Chapter generation | Poor | Keep LLM | Long-form synthesis |
| Forum host speech | Poor | Keep LLM | New prose and synthesis |
| Free-form keyword generation | Mixed | Conditional LLM | Needed only when the decision model says expansion is useful |

### Existing bug fixed by the split

The QueryEngine prompt asked the LLM for `search_tool`, `start_date`, and
`end_date`, but `search_node.py` only retained `search_query` and
`reasoning`. The caller therefore usually fell back to
`basic_search_news`. Tool routing is now owned by System One; the LLM prompt
only generates the open-ended query.

## Runtime behavior

The integration is fail-open to the legacy path:

- no decision-model key (`DECISION_MODEL_API_KEY`), an empty key, or
  `SYSTEM_ONE_ENABLED=false` -> no decision request; existing LLM behavior;
- request/response error -> existing behavior;
- Query search-tool and report-template Choice confidence below
  `SYSTEM_ONE_CHOICE_CONFIDENCE` -> fallback; Media/Insight routing currently
  validates the selected option but does not apply this confidence threshold;
- reflection Noul below `SYSTEM_ONE_STOP_THRESHOLD` -> skip the next expensive
  reflection query + search + summary chain.

Defaults: `DECISION_MODEL_NAME=inception/mercury-decide:free`,
`DECISION_MODEL_BASE_URL=https://openrouter.ai/api/alpha/decisions`,
`SYSTEM_ONE_CHOICE_CONFIDENCE=0.45`,
`SYSTEM_ONE_STOP_THRESHOLD=0.30`, `SYSTEM_ONE_TIMEOUT=30`.

Set `SYSTEM_ONE_TRACE_PATH` to write JSONL traces. By default traces contain a
state hash, answers, and latency rather than the full state.

HTTP 401/403, 429, 5xx, timeout and connection failures return to the legacy
path. The client does not retry, sleep for `Retry-After`, or implement a circuit
breaker: later decision sites can attempt another request. The fallback LLM
and search providers can still incur their normal costs. Offline failure tests
verify control flow, not live credentials, quotas, answer quality or service
availability.

### Research data leaving the process

The state is sent as JSON to the configured decision endpoint; the trace's
state hash does not anonymize or replace the outbound state. With the default
configuration, OpenRouter and its routed decision-model provider receive it.
Compared with Jev, the payload construction is unchanged but its recipients
change. There is no automatic sensitive-data redaction.

- Search routing: generated query and complete section input, including any
  accumulated paragraph state present in that input.
- Reflection: section title/content, accumulated summary and reflection index.
- Evidence triage: query, section and up to 10 candidates by default; per
  candidate, title up to 300 characters, content up to 900, URL up to 500,
  publication date and platform.
- Insight keyword gate: original query and context; Forum gate: last five
  nonempty agent speeches, without a per-speech truncation.
- Report template: query, template names/descriptions, first four report
  excerpts up to 1,200 characters each and 1,200 characters of forum text.
- Word budget: query, section outlines, up to 1,800 characters per supplied
  report and 1,200 characters of forum text; layout: query and section outlines.

Local traces additionally contain answers and usage. `SYSTEM_ONE_TRACE_STATE`
opts into the first 4,000 characters of serialized state. The manual workflow
uploads research artifacts (including traces) with 14-day retention; review
artifact access independently of provider data policies.

### Deployment acceptance checklist

1. Confirm canonical secret/variable names exist in the actual deployment;
   inspect names/presence only, never print key values. Workflow files and a
   passing offline CI run cannot establish repository secret configuration.
2. Verify the key issuer matches the explicit endpoint and model. Preserve
   per-engine settings for local/Docker `app.py`; generic LLM names alone do
   not configure those deployments.
3. Review provider data handling and current free-model availability/limits
   before enabling private or sensitive research. A `:free` model name does
   not make the entire research workflow free.
4. Compare representative non-sensitive, explicitly authorized samples with
   Jev before relying on the existing thresholds. Compatible response types
   do not establish identical decisions or confidence calibration. Routing,
   evidence ordering, early-stop, host and report decisions may change.
5. Distinguish configuration availability, attempted HTTP success and an
   answer actually accepted by the decision helper. Keep System One disabled
   if migration or data-sharing requirements are not satisfied.

As checked on 2026-10-18, OpenRouter's [free-model limits](https://openrouter.ai/docs/api_reference/limits)
are generally 20 requests/minute and 50/day below 10 purchased credits, rising
to 1,000/day at that purchase tier (the limits documentation allows one credit
of tolerance for fees/rounding, with possible account/endpoint exceptions). These are shared-account
limits, not a reserved allowance per research job. Each decision-site call is
one HTTP request, even when it batches multiple questions; quota exhaustion
can therefore move a research run back onto paid LLM paths.

The [public ZDR catalog](https://openrouter.ai/api/v1/endpoints/zdr) listed paid
`inception/mercury-decide`, but not `inception/mercury-decide:free`, on that date.
Absence does not prove training, but zero retention for this free route is not
established. Review [provider-specific policies](https://openrouter.ai/docs/guides/privacy/provider-logging)
and [OpenRouter collection settings](https://openrouter.ai/docs/guides/privacy/data-collection).
The current client sends only model/state/questions and does not request a
ZDR routing constraint. Do not assume it enforces a zero-retention policy.

## Manual GitHub Action

Workflow: **Manual Research (System One)**

Inputs: `query`, `max_reflections` (0..3), and `use_system_one`.

Configuration wired by the workflow:

Repository secrets:
- `LLM_MODEL_API_KEY` (required; issued by the configured LLM provider)
- `TAVILY_API_KEY` (required)
- `ANSPIRE_API_KEY` (optional for this QueryEngine-only job)
- `DECISION_MODEL_API_KEY` (optional; absent means legacy LLM fallback)

Repository variables:
- `LLM_MODEL_BASE_URL` (empty means the OpenAI SDK default endpoint)
- `LLM_MODEL_NAME` (required)
- `DECISION_MODEL_BASE_URL` (default: OpenRouter Decisions endpoint)
- `DECISION_MODEL_NAME` (default: `inception/mercury-decide:free`)

The current headless job runs QueryEngine, so Tavily is the active search API.
Anspire is wired for the rest of BettaFish but is not invoked by this job.
The workflow reads the new secret names only; it cannot rename or provision
repository secrets for you. With System One requested but no decision secret,
preflight reports that the run will use the LLM fallback and continues.
Artifacts contain `result.md`, `metadata.json`, and (when used)
`system_one_trace.jsonl`.

For direct headless runs, `--output-dir` selects the artifact directory and
defaults QueryEngine's working directory to its `query_reports` subdirectory.
Only a nonblank process-level `OUTPUT_DIR` overrides that QueryEngine path;
the runner ignores a dotenv `OUTPUT_DIR`, and blank process values use the CLI
default. `result.md` and `metadata.json` always use `--output-dir`.

`metadata.json` separates configuration from per-run request outcomes:

| Field | Meaning |
|---|---|
| `system_one_requested` | The resolved enable toggle. |
| `system_one_available` | A nonempty decision key is configured, including a key read from `.env`; this is not a connectivity or authentication check. |
| `system_one_enabled` | The client has a key and the enable toggle permits requests. |
| `system_one_model` | The model requested by the shared decision client. |
| `system_one_requests_attempted` | Decision HTTP requests attempted during this research run. |
| `system_one_responses_succeeded` | Requests that returned a successful HTTP status and a JSON object; this does not mean a decision was accepted. |
| `system_one_requests_failed` | Transport, HTTP, or response-decoding failures during this run. |
| `llm_model` / `max_reflections` | Values from the research agent's effective settings. |

For example, missing credentials yield requested=true, available=false,
enabled=false, and zero requests when the toggle was left on. A configured
client receiving HTTP 429 remains available/enabled but records a failed
request and falls back. A low-confidence answer can count as a successful
response while the decision site still uses the LLM fallback. The counters
exclude any earlier requests made by the shared client before this run.

Use `system_one_trace.jsonl` to inspect request outcomes and returned answers.
Each record's `model` is the configured model; `response_model` and `provider`,
when supplied by the service, describe the response. Trace write failures do
not discard valid responses or prevent fallback, and request counters do not
depend on trace-file writes. Neither the counters nor a successful trace
record measures accepted/applied decisions. The metadata never claims that
every decision used System One.


## Current decision-model coverage

The following bounded decisions are now owned by System One:

- QueryEngine initial and reflection search-tool routing (Choice).
- QueryEngine reflection early-stop (Noul).
- QueryEngine evidence triage: relevance / evidence value / novelty (batched Scores).
- MediaEngine initial and reflection search-tool routing with provider-aware options (Choice).
- MediaEngine reflection early-stop (Noul).
- MediaEngine evidence triage: relevance / evidence value / novelty (batched Scores).
- InsightEngine database-tool routing plus platform, time window and sentiment decision in one speculative fan-out request (Choice + Noul).
- InsightEngine reflection early-stop (Noul).
- InsightEngine KeywordOptimizer expansion gate (Noul): concrete entity/event queries skip the free-form keyword-expansion LLM; abstract queries keep the legacy LLM expansion path.
- ForumEngine host-intervention gate (Noul + reason Choice).
- ReportEngine template selection (Choice).
- ReportEngine word-budget planning: chapter importance / evidence density / analytical complexity (batched Scores) followed by deterministic allocation.
- ReportEngine SWOT / PEST applicability (batched Noul), with code enforcing at most one chapter for each framework.

Open-ended generation remains with LLMs: search-query text, report structure/title/hero copy, summaries, reflection query text, chapter writing, final prose, Forum host speech, and free-form keyword generation.

### Evidence triage

For Query/Media search results, System One evaluates up to `SYSTEM_ONE_EVIDENCE_CANDIDATES` candidates in one request. Each result gets three independent Score questions:

- relevance: 50%
- evidence value: 35%
- novelty: 15%

Only the top `SYSTEM_ONE_EVIDENCE_MAX_RESULTS` are sent to the expensive summary LLM. System One failure preserves original ordering.

### Insight fan-out

Insight routing asks tool, platform, recency window and whether sentiment is useful in one request. The answers are independent and code uses only the relevant parameters for the selected tool.

### Forum host gate

Every five agent speeches are judged first. If `host_needed < SYSTEM_ONE_HOST_THRESHOLD`, the five speeches are consumed without paying for a host LLM turn. Decision-model failure preserves the legacy host behavior.

### Word budget

WordBudgetNode first asks three Score questions per chapter (importance, evidence density, complexity) in one request. Python combines them with 0.50 / 0.30 / 0.20 weights and allocates the total word budget deterministically. The old LLM planner is retained only as fail-open fallback.


### KeywordOptimizer expansion gate

Before the free-form keyword-expansion LLM runs, the decision model judges whether expansion is
materially useful. Concrete event/person/organization/product queries use direct
deterministic tokens and skip that LLM call. Abstract or compound analytical queries
retain the existing LLM expansion path. System One failure is fail-open to legacy behavior.


### Report-structure refusal guard

The report-structure LLM is upstream of every search and System One decision. A provider
refusal or malformed outline must therefore never silently turn into a generic
unrelated research topic. Query/Media/Insight now share a guard with this policy:

1. detect common refusal responses;
2. retry once with the benign scope made explicit: public-information research
   and fact checking only, with no operational harmful guidance;
3. if the retry still fails, build five deterministic sections that each retain
   the exact original query;
4. malformed/non-JSON output uses the same topic-preserving fallback.

This fixes the failure mode observed in run 35323864180 where the structure model
returned a refusal for "广州大学城随机捅人" and the old fallback replaced it with
generic "研究概述 / 深度分析", causing every downstream search to drift off-topic.
