<!-- chapter: ch-03
     track: internals
     kind: content
     title: Tool Discovery — Search + a From-Scratch BM25
     deps: [ch-02]
     sources: [[automationbench-harness]]
     figures: figures/bm25-explorer.html
-->

# Chapter 03 — Tool Discovery: Search + a From-Scratch BM25

> **Core insight.** The benchmark tests whether a model can *find* the right tool out of ~400 candidates, not merely call it once handed. Discovery is the first hard cognitive step; every other capability hangs on getting it right.

> **Guideline.** Index the tool corpus at construction time (BM25 over `name + docstring + param descriptions`), expose one `search_tools` function to the agent, and let token economy drive compression — verbose schemas survive only as long as they're actionable.

---

## 1  Why Discovery Is a First-Class Tested Capability

AutomationBench's headline toolset mode (`zapier`) gives the model exactly **two** tools: `search_tools` and `execute_tool`. There is no pre-enumerated list of 400 names in the system prompt, no service directory the model can scan. If it wants to send a Gmail, it must first *search* for the right tool name.

This is deliberate. Real automation agents face the same constraint: enterprise SaaS surfaces hundreds of actions, and cramming every schema into the context window is prohibitively expensive. The benchmark operationalizes that cost by hiding the corpus and measuring whether the model reaches the right tool via search.

The `limited_zapier` mode inverts the same idea as an ablation. The model receives the exact per-task tool subset in its tool list — discovery is removed as a variable. Comparing `zapier` vs `limited_zapier` scores isolates the *discovery tax*: the accuracy gap attributable purely to the model having to find its tools rather than having them handed over.

The test suite makes the same point from a unit-test angle. `tests/test_bm25.py` and `tests/test_api_search.py` exist as first-class test files alongside domain and rubric tests, not as afterthoughts. Discovery infrastructure is production code.

---

## 2  The From-Scratch BM25

The harness ships its own BM25 implementation in `automationbench/utils/bm25.py`. No external ranking library is imported; the file is 70 lines of pure Python.

### 2.1  Tokenizer

```python
# automationbench/utils/bm25.py  L10-12
def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, treating underscores as word separators."""
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
```

The underscore-replace step is load-bearing. Tool names follow the `{service}_{action}` convention — `gmail_send_email`, `salesforce_find_records`, `slack_post_message`. If underscores were left in, the tokenizer would emit a single token `gmail_send_email` that never matches a query word like `gmail`. The replace turns the name into `"gmail send email"` before the regex runs, so each word is an independent index token. The test confirms this explicitly:

```python
# tests/test_bm25.py  L129-133
def test_underscore_terms_in_docs(self):
    docs = ["slack_chat_post_message send message to channel"]
    scorer = BM25Scorer(docs)
    scores = scorer.scores("slack chat post")
    assert scores[0] > 0
```

### 2.2  BM25Scorer construction

```python
# automationbench/utils/bm25.py  L27-45
def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
    self.k1 = k1
    self.b = b

    self._tokenized: list[list[str]] = [tokenize(doc) for doc in docs]
    n = len(self._tokenized)
    self._avgdl = sum(len(d) for d in self._tokenized) / n if n else 1.0

    # Document frequency per term
    df: dict[str, int] = {}
    for doc in self._tokenized:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    # IDF with Robertson-Sparck Jones smoothing
    self._idf: dict[str, float] = {
        term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
        for term, freq in df.items()
    }
```

The IDF formula is the Robertson–Sparck Jones smoothed variant: `log((N - df + 0.5) / (df + 0.5) + 1)`. The `+ 1` before the outer log prevents negative IDF for terms that appear in more than half the corpus — a common corpus in tool search because service names like `gmail` appear across multiple tool descriptions for the same service. Without the smoothing, a query for `"gmail"` would score *lower* the more Gmail tools are indexed.

Parameters `k1=1.5` and `b=0.75` are the de-facto BM25 defaults. `k1` controls term-frequency saturation: repeating a term many times in a document stops yielding proportional score gains. `b=0.75` applies length normalization — a long docstring with one relevant word scores lower than a short, focused description. The test at L119-127 verifies that non-default `k1`/`b` values produce different relative orderings, which is the correct behavior to guard against.

### 2.3  Scoring and top_k

```python
# automationbench/utils/bm25.py  L47-69
def scores(self, query: str) -> list[float]:
    """Return a BM25 score for each document in the corpus."""
    terms = tokenize(query)
    result = []
    for doc_tokens in self._tokenized:
        dl = len(doc_tokens)
        score = 0.0
        for term in terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            tf = doc_tokens.count(term)
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            )
        result.append(score)
    return result

def top_k(self, query: str, k: int = 10) -> list[int]:
    """Return indices of the top-k scoring documents (descending order)."""
    scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [i for _, i in scored[:k]]
```

`top_k` filters to `score > 0` before slicing — documents with zero overlap are excluded outright, so a `top_k` call on a sparse query returns fewer than `k` results rather than padding with zero-score noise. This matters for the agent: an empty result list is a clear signal to try a different query, while a list of zero-score results would be misleading.

An interactive version of this scorer — running live in your browser over a toy corpus of 12 tool docstrings — is in **[figures/bm25-explorer.html](figures/bm25-explorer.html)**. Type a query and watch the per-term score contributions update in real time.

---

## 3  ToolRegistry and search_tools

The BM25Scorer is wired into `ToolRegistry` in `automationbench/tools/zapier/meta.py`. The registry is a module-level lazy singleton, built once on first use from `ALL_TOOLS`.

### 3.1  What gets indexed

```python
# automationbench/tools/zapier/meta.py  L46-56
# Include parameter names + descriptions in searchable text (mirrors api_search index)
param_parts: list[str] = []
for param_name, param_info in params.get("properties", {}).items():
    param_parts.append(param_name)
    if isinstance(param_info, dict) and param_info.get("description"):
        param_parts.append(param_info["description"])
params_text = " ".join(param_parts)
searchable = f"{name}: {full_desc}"
if params_text:
    searchable += f" {params_text}"
self._searchable_texts.append(searchable)
```

Each tool's index document has the shape `"name: docstring param param_desc ..."`. Including parameter names and their descriptions matters because an agent might query `"subject body"` (field names) rather than `"send email"` (action verbs). Both paths should surface `gmail_send_email`.

The `world` parameter is stripped from the description before indexing (L60-81): the registry calls `_get_full_description`, which drops lines starting with `world:` and their continuation indents. The `world` injection is an implementation detail hidden from both the model and the search index.

### 3.2  search_tools and make_search_tools

```python
# automationbench/tools/zapier/meta.py  L134-154
def search_tools(query: str, top_k: int = 5) -> str:
    """Find available tools by name or description.

    Tool names follow the pattern {service}_{action} (e.g., salesforce_query,
    gmail_send_email, slack_send_channel_message).

    Uses BM25 keyword-based relevance search. Works with service names,
    action words, or multi-word queries.
    Examples: "salesforce", "send email", "update deal", "slack channel"

    Args:
        query: Search query — service names, keywords, or a description.
        top_k: Maximum number of results to return (default: 20).

    Returns:
        JSON string with a list of matching tools, each containing name,
        description, and parameter schema.
    """
    registry = _get_registry()
    results = registry.bm25(query, top_k=top_k)
    return json.dumps(results, indent=2)
```

The default `top_k=5` in the function signature is a moderate budget: enough to cover ambiguous queries (the right tool may not rank first when the query is imprecise) without flooding the context window with five full JSON schemas. Each result is a dict with three keys: `name`, `description` (cleaned docstring), and `parameters` (full JSON schema with property types and descriptions). That schema is what the model uses to construct the `arguments` JSON string for `execute_tool`.

`make_search_tools` (L157-186) returns a version of the same function with a configurable default and a hard cap:

```python
# automationbench/tools/zapier/meta.py  L157-186
def make_search_tools(default_top_k: int = 20, max_top_k: int | None = None) -> Callable:
    """Return a search_tools function with a custom default and optional hard cap on top_k."""
    cap = max_top_k
    default = default_top_k

    def _search_tools(query: str, top_k: int = default) -> str:
        ...
        actual_k = min(top_k, cap) if cap is not None else top_k
        results = registry.bm25(query, top_k=actual_k)
        return json.dumps(results, indent=2)

    _search_tools.__name__ = "search_tools"
    return _search_tools
```

The cap matters for controlled experiments. When the harness is initialized with `search_top_k=N` (runner.py L83), `make_search_tools(max_top_k=N)` is used, preventing the model from evading the discovery challenge by requesting an arbitrarily large `top_k` and scanning the entire corpus.

---

## 4  execute_tool

Once `search_tools` surfaces a candidate, `execute_tool` dispatches the call:

```python
# automationbench/tools/zapier/meta.py  L189-204
def execute_tool(world: WorldState, tool_name: str, arguments: str) -> str:
    """Execute a discovered tool by name with the given arguments.

    Use search_tools first to find the right tool and its parameter schema,
    then call this with the tool name and a JSON string of arguments.

    Args:
        world: The current world state (injected automatically).
        tool_name: The exact tool name from search results.
        arguments: JSON string of arguments matching the tool's parameter schema.

    Returns:
        The tool's return value (JSON string).
    """
    registry = _get_registry()
    return registry.execute(tool_name, arguments, world=world)
```

Internally, `registry.execute` (L98-110) does three things: look up the function by name, `json.loads` the `arguments` string, merge in the injected `world` kwarg, and call the function. If `tool_name` is unknown, it raises a `ValueError` whose message explicitly tells the model to use `search_tools` — closing the loop back to discovery.

```python
# automationbench/tools/zapier/meta.py  L98-110
def execute(self, tool_name: str, arguments: str, **injected: Any) -> str:
    """Execute a tool by name with JSON arguments string."""
    func = self._tool_map.get(tool_name)
    if func is None:
        raise ValueError(
            f"Unknown tool: {tool_name}. Use search_tools to discover available tools."
        )
    parsed_args = json.loads(arguments)
    merged = {**parsed_args, **injected}
    result = func(**merged)
    if isinstance(result, str):
        return result
    return json.dumps(result)
```

---

## 5  API-Mode Discovery: api_search

The `api` toolset mode gives the model `api_search` and `api_fetch` instead of the Zapier meta-tools. `api_search` (`automationbench/tools/api/search.py`) runs BM25 over a flat tab-separated index of REST endpoints:

```python
# automationbench/tools/api/search.py  L83-95
def _build_index_line(api_name: str, endpoint: dict) -> str:
    """Build one tab-separated searchable line for an endpoint.

    Format: api_name<TAB>endpoint_id<TAB>method<TAB>path<TAB>searchable_text
    searchable_text includes the endpoint description plus all parameter descriptions.
    """
    desc_parts = [endpoint.get("description", "")]
    for param_info in endpoint.get("parameters", {}).values():
        if isinstance(param_info, dict) and param_info.get("description"):
            desc_parts.append(param_info["description"])
    searchable = " ".join(filter(None, desc_parts))
    fields = [api_name, endpoint["id"], endpoint["method"], endpoint["path"], searchable]
    return "\t".join(fields)
```

The index (`schemas/index.txt`) is lazily rebuilt whenever any `.jsonc` schema file is newer than the index on disk (L107-115). This is an mtime-based incremental build: adding a new API schema triggers a one-time regeneration on the next `api_search` call, not on import.

```python
# automationbench/tools/api/search.py  L107-115
def _ensure_index(schemas: dict[str, dict]) -> list[str]:
    """Return index lines, regenerating index.txt if any schema file is newer."""
    schema_files = list(SCHEMAS_DIR.glob("*.jsonc"))
    needs_regen = not INDEX_FILE.exists() or any(
        f.stat().st_mtime > INDEX_FILE.stat().st_mtime for f in schema_files
    )
    if needs_regen:
        _regenerate_index(schemas)
    return INDEX_FILE.read_text().splitlines()
```

The result format differs from `search_tools`: each result includes the resolved `url` field (base URL + stripped internal prefix) ready to pass to `api_fetch`, rather than a Python function name. The test `test_results_no_path_field` at `tests/test_api_search.py L107-111` guards this contract — `path` must be absent, `url` must be present, because the model should never have to reason about internal routing prefixes.

---

## 6  Token Economy: _compress_meta_messages

A naive implementation would accumulate verbose search results in the conversation history indefinitely. Each `search_tools` result is a JSON array of full schemas — name, docstring, and every parameter with type and description. Across a 10-turn episode with several searches, this grows to thousands of tokens of dead weight.

`_compress_meta_messages` in `runner.py` (L234) solves this by rewriting stale results once they are no longer actionable:

```python
# automationbench/runner.py  L234-293
def _compress_meta_messages(
    self,
    messages: vf.Messages,
    tool_messages: vf.Messages,
    state: vf.State,
) -> vf.Messages:
    """Compress old search_tools results after execute_tool is called.

    Once the model acts on search results by calling execute_tool, the verbose
    search results (full descriptions + parameter schemas) are dead weight.
    Replace them with a brief tool name list to save tokens on future turns.

    Only compresses search results from PREVIOUS turns, never the current turn.
    This ensures schemas remain available for tools searched in the same turn
    as an execute_tool call, preventing argument-name hallucination when the
    model searches and executes in parallel.
    """
```

The logic tracks which `search_tools` call IDs have been seen, and on any turn that contains an `execute_tool` call, rewrites older search result messages to `[Previously found: name1, name2, ...]`. The critical invariant is the **current-turn exclusion**: search results from the same turn as the `execute_tool` call are left intact.

Why? A model may search and immediately execute in a single parallel tool-call batch. If the search result were compressed before the execute call resolved, the model would be constructing `arguments` without access to the parameter schema it just retrieved — guaranteeing argument hallucination. The implementation tracks this via set arithmetic:

```python
# automationbench/runner.py  L258-275  (key excerpt)
current_search_ids: set[str] = set()
has_execute = False
for tc in tool_calls:
    if tc.name == "search_tools":
        current_search_ids.add(tc.id)
    elif tc.name == "execute_tool":
        has_execute = True

# Accumulate this turn's search IDs for potential compression next turn
state.setdefault("_search_call_ids", set()).update(current_search_ids)

if not has_execute:
    return tool_messages

# Only compress searches from PREVIOUS turns (exclude current turn)
compressible_ids = state["_search_call_ids"] - current_search_ids
```

`state["_search_call_ids"]` accumulates all search IDs ever seen. `compressible_ids` is the accumulated set *minus* the current turn's IDs. Only those get rewritten. The resulting message is:

```
[Previously found: gmail_send_email, gmail_list_messages, ...]
```

This is enough for the model to remember it already searched and what it found, without paying the full schema token cost again.

---

## 7  Test Suite as Executable Specification

The test suite treats discovery as a first-class capability, not an implementation detail. The `test_idf_computed` test (L91-95) is a particularly clean executable spec for the RSJ IDF formula:

```python
# tests/test_bm25.py  L91-95
def test_idf_computed(self):
    scorer = BM25Scorer(["email send", "email draft", "calendar event"])
    # "email" appears in 2/3 docs, "calendar" in 1/3
    # IDF for rarer term should be higher
    assert scorer._idf["calendar"] > scorer._idf["email"]
```

This is the IDF contract in code: rarer terms rank higher. Any change to the IDF formula that violates this — say, accidentally dropping the `+1` smoothing term — would flip the assertion.

For `api_search`, the deduplication test is the most operationally important:

```python
# tests/test_api_search.py  L113-116
def test_deduplicates_results(self):
    result = json.loads(api_search("email", top_k=10))
    ids = [r["id"] for r in result["results"]]
    assert len(ids) == len(set(ids)), "Duplicate endpoint IDs in results"
```

Because `api_search` fetches `top_k * 3` candidates to account for deduplication (L153), the same endpoint can appear in multiple index lines. Without the `seen` set, the agent could receive the same endpoint twice in a `top_k=5` result, wasting a result slot.

---

## 8  Summary: The Discovery Stack

The full discovery path from agent query to tool execution:

```
agent emits: search_tools("send slack message", top_k=5)
             │
             ▼
ToolRegistry.bm25(query, top_k=5)
             │  tokenize("send slack message") → ["send","slack","message"]
             │  BM25Scorer.top_k(query, k=5)
             │    per-doc: score = Σ IDF(t) * TF_norm(t, doc)
             │    filter score>0, sort descending, return indices[:5]
             ▼
[{name: "slack_post_message", description: "...", parameters: {channel: ..., text: ...}}, ...]
             │
             ▼
agent emits: execute_tool("slack_post_message", '{"channel": "C123", "text": "Hello"}')
             │
             ▼
registry.execute → json.loads(arguments) + inject world → func(**merged) → result string
             │
             ▼  (next turn — if execute_tool was called)
_compress_meta_messages: previous search results → "[Previously found: slack_post_message, ...]"
```

The three toolset modes (`zapier`, `limited_zapier`, `api`) each test a different slice of this stack, allowing the benchmark to attribute accuracy differences to discovery vs. execution vs. REST-shape reasoning independently.

See [[automationbench-harness]] for the full episode lifecycle and the `limited_zapier` ablation design.
