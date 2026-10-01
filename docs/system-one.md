# System One / decision model integration

## Endpoint

System One talks to the **OpenRouter Decisions API** with the
`inception/mercury-decide:free` model:

- URL: `https://openrouter.ai/api/alpha/decisions` (override with `SYSTEM_ONE_URL`)
- Model: `inception/mercury-decide:free` (override with `SYSTEM_ONE_MODEL`)
- Key: `OPENROUTER_API_KEY` (`TYPESAFE_API_KEY` still accepted as a legacy
  alias so existing deployments keep working)

The request/response schema (`state` + typed `questions` -> `answers`) is the
System One schema; mercury-decide serves it natively.

## Audit result

BettaFish mixes open-ended generation with bounded judgment. System One is a
good replacement for the latter, not for arbitrary prose generation.

| Decision site | Fit for Jev | Status | Why |
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
| Free-form keyword generation | Mixed | Conditional LLM | Needed only when Jev says expansion is useful |

### Existing bug fixed by the split

The QueryEngine prompt asked the LLM for `search_tool`, `start_date`, and
`end_date`, but `search_node.py` only retained `search_query` and
`reasoning`. The caller therefore usually fell back to
`basic_search_news`. Tool routing is now owned by System One; the LLM prompt
only generates the open-ended query.

## Runtime behavior

The integration is fail-open to the legacy path:

- no `OPENROUTER_API_KEY` (and no legacy `TYPESAFE_API_KEY`) -> existing behavior;
- request/response error -> existing behavior;
- Choice confidence below `SYSTEM_ONE_CHOICE_CONFIDENCE` -> fallback;
- reflection Noul below `SYSTEM_ONE_STOP_THRESHOLD` -> skip the next expensive
  reflection query + search + summary chain.

Defaults: `SYSTEM_ONE_MODEL=inception/mercury-decide:free`,
`SYSTEM_ONE_URL=https://openrouter.ai/api/alpha/decisions`,
`SYSTEM_ONE_CHOICE_CONFIDENCE=0.45`,
`SYSTEM_ONE_STOP_THRESHOLD=0.30`, `SYSTEM_ONE_TIMEOUT=30`.

Set `SYSTEM_ONE_TRACE_PATH` to write JSONL traces. By default traces contain a
state hash, answers, and latency rather than the full state.

## Manual GitHub Action

Workflow: **Manual Research (System One)**

Inputs: `query`, `max_reflections` (0..3), and `use_system_one`.

Configuration wired by the workflow:

Repository secrets:
- `MODEL_API_KEY`
- `TAVILY_API_KEY`
- `ANSPIRE_API_KEY`
- `OPENROUTER_API_KEY` (legacy `TYPESAFE_API_KEY` still accepted)

Repository variables:
- `MODEL_BASE_URL`
- `MODEL_NAME`

The current headless job runs QueryEngine, so Tavily is the active search API.
Anspire is wired for the rest of BettaFish but is not invoked by this job.
Artifacts contain `result.md`, `metadata.json`, and (when used)
`system_one_trace.jsonl`.


## Current Jev coverage

The following bounded decisions are now owned by System One / Jev:

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

Before the free-form keyword-expansion LLM runs, Jev judges whether expansion is
materially useful. Concrete event/person/organization/product queries use direct
deterministic tokens and skip that LLM call. Abstract or compound analytical queries
retain the existing LLM expansion path. System One failure is fail-open to legacy behavior.


### Report-structure refusal guard

The report-structure LLM is upstream of every search and Jev decision. A provider
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
