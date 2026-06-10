<!-- scope: AutomationBench execution engine — runner, toolset modes, BM25, execute, world, cost
     deps: automationbench-overview
     see-also: automationbench-tasks-grading, benchmark-comparison
-->

# AutomationBench — Harness / Execution Engine

- **Core Insight:** The whole "world" is **one in-process Pydantic object** passed by
  reference into every tool; there is no HTTP server, no DB, no subprocess — which is
  exactly what buys determinism and <1% run variance.
- **Guideline:** If you want a reproducible agent benchmark, simulate the world in-process
  with typed state and inject it into tools *behind* the model-facing schema.
- **Source:** `automationbench/{runner.py, tool_wrapper.py, clients.py, pricing.py,
  usage.py, utils/bm25.py}`, `tools/zapier/meta.py`, `tools/api/{search,fetch}.py`,
  `schema/world.py`; cross-checked against `tests/`.
- **Relevant chapters:** ch-02, ch-03, ch-04, ch-08.

## Episode lifecycle

`AutomationBenchEnv` subclasses `vf.StatefulToolEnv` from the `verifiers` library
(`runner.py`). One episode:

1. **Dataset build** — `domains/<d>/tasks.py` emits rows `{prompt, info}`; `info` carries
   `initial_state`, `zapier_tools`, `assertions`. Noise is injected here (see
   [[automationbench-tasks-grading]]).
2. **`setup_state`** (`runner.py:136`) — deserialize `info`, strip HF-injected `None`s,
   build `WorldState(**initial_state)`, deep-copy the initial state for later
   free-assertion detection, set the per-task tool list.
3. **Agent loop** (bounded by `max_turns`, default **25**) — model is called with history +
   tool defs; each tool call is dispatched through `update_tool_args` (which re-injects
   `world`) and executed; results return as `ToolMessage`s. Loop ends on a no-tool-call
   response or `max_turns`.
4. **Grading** — `partial_credit` / `task_completed_correctly` from the rubric.
5. **Export** — `export_results` (`export.py:28`) writes scores, per-assertion results,
   end state, messages, usage, cost for the visualizer.

## The three toolset modes (the doc "2 tools" is mode `zapier`)

- **`zapier`** — exactly two meta-tools: `search_tools` + `execute_tool` (`runner.py:83`).
  Discovery is required; this is the headline mode.
- **`limited_zapier`** — all ~400 named tools, filtered per task to `info.zapier_tools`.
  Isolates execution skill from discovery skill (ablation).
- **`api`** — `api_search`, `api_fetch`, `base64_encode` (REST-shaped; model emits URLs).

## The `world` injection trick

Tools take `world: WorldState` as a first arg. `tool_wrapper.py` lists `world` in
`args_to_skip`, so it never appears in the JSON schema the model sees; it is re-injected at
dispatch (`runner.py:118`). The model calls `execute_tool(tool_name, arguments)`; the
harness adds `world`.

## Tool discovery: a from-scratch BM25

`utils/bm25.py` implements BM25 (no library): `k1=1.5, b=0.75`, Robertson–Sparck-Jones IDF
`log((N-df+0.5)/(df+0.5)+1)`, tokenizer `re.findall(r"[a-z0-9]+", text.lower().replace("_"," "))`
so `gmail_send_email → [gmail, send, email]`.

- **`search_tools(query, top_k=5)`** (`tools/zapier/meta.py:134`) → JSON list of up to
  `top_k` `{name, description (docstring minus world), parameters (schema)}`. Corpus = one
  doc per tool: `"name: docstring param param_desc ..."`. `make_search_tools(max_top_k=N)`
  caps it.
- **`api_search`** (`tools/api/search.py`) → BM25 over a tab-separated endpoint index
  (`schemas/index.txt`), lazily rebuilt when any `.jsonc` schema is newer.
- **`_compress_meta_messages`** (`runner.py:234`) rewrites *previous-turn* search results to
  `[Previously found: name1, name2]` once `execute_tool` runs, to save context — but leaves
  *current-turn* results intact so the model doesn't hallucinate args it just searched.

## Execute → simulated backend

- **`execute_tool(tool_name, arguments)`** (`meta.py:189`) → `registry.execute`: JSON-parse
  `arguments`, merge injected `world`, call `func(**merged)`, return string / `json.dumps`.
- **`api_fetch(method, url, params, body)`** (`fetch.py`) → `_url_to_internal_path(url)`
  routes via static + dynamic tables (special-cases for graph.facebook.com fan-out, Jira/
  Confluence `.atlassian.net` split, BambooHR gateway). Unknown URL → `{"error":{"code":404}}`.
  **URL-hallucination tolerance**: ~15 known model mistakes (e.g. `slack.googleapis.com`)
  are normalized to the right router so a URL typo isn't scored as a task failure.
- **State**: `WorldState` (`schema/world.py:70`) is one Pydantic `BaseModel`,
  `extra="forbid"`, with 44 app sub-states; each app holds typed record lists. Tools mutate
  `world.<app>.<collection>` in memory. `to_display_dict()` omits `None` fields → sparse,
  realistic records. Pagination, required fields, and common 4xx codes are simulated.

## Cost metric

`_extract_usage_and_debug` (`runner.py:185`) accumulates prompt/completion (and reasoning)
tokens per turn into `state["_usage"]`. `pricing.py` resolves model → per-token price via
exact → normalized → alias → normalized-query lookup, sourcing `llm-prices.com`
(24h cache) with a hardcoded fallback; CLI `--input-cost/--output-cost` override.
Cost `= in_tok·in_price + out_tok·out_price`, summed per run (`usage.py`).

## Notable design choices

Generic Search+Execute (discovery tested, not just use); three modes for ablation;
free-assertion anti-reward-hacking (see [[automationbench-tasks-grading]]);
**no conversational user simulator** (task is self-contained in the trigger + initial
state — the key structural contrast with [[taubench]]); all state in-process Pydantic;
streaming Anthropic client with interleaved-thinking for long episodes (`clients.py`).
