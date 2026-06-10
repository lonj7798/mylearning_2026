<!-- chapter: ch-02
     track: internals
     kind: content
     title: Architecture Overview: WorldState, the Episode Loop, and Toolset Modes
     deps: [ch-01]
     sources: [[automationbench-harness]], [[automationbench-overview]]
     figures: figures/episode-flow.html
-->

# Chapter 02 — Architecture Overview: WorldState, the Episode Loop, and Toolset Modes

> **Core insight.** AutomationBench achieves deterministic, sub-1% run variance by simulating
> every SaaS backend as a single in-process Pydantic object — no HTTP server, no database, no
> subprocess. The entire "world" is a typed Python value passed by reference into every tool
> call. Because there are no network faults, no race conditions, and no external state, the
> same initial state always produces the same trajectory under the same model outputs.

> **Guideline.** If you want a reproducible agent benchmark, simulate the world in-process
> with typed state and inject it into tools *behind* the model-facing schema. The model never
> sees the `world` parameter; the harness injects it at dispatch time. This decoupling is what
> lets you ablate discovery, execution, and API-style access as three clean modes without
> changing any tool implementation.

---

## 1. What `verifiers` Provides and What AutomationBench Adds

AutomationBench is built on the `verifiers` library. The entry point is:

```python
# automationbench/runner.py
class AutomationBenchEnv(vf.StatefulToolEnv):
```

`vf.StatefulToolEnv` supplies the scaffolding that most agent benchmarks need: a dataset
iteration loop, a tool registry with `add_tool`, a `max_turns` guard on the agent loop, and
the `env_response` method that dispatches tool calls and returns `ToolMessage`s. What
AutomationBenchEnv adds on top is:

- **WorldState** — the in-process Pydantic simulation of 44 SaaS backends (see §2).
- **Three toolset modes** — `zapier`, `limited_zapier`, and `api` (see §4).
- **The `world` injection trick** — hiding the internal `world` argument from the model-facing
  JSON schema via `args_to_skip` (see §5).
- **Per-task tool filtering** — `setup_state` narrows the tool list from the full registry to
  only the tools allowed by `info.zapier_tools` for the current task.
- **Meta-message compression** — `_compress_meta_messages` rewrites stale search results to
  short summaries once `execute_tool` is called, preventing context bloat across long episodes.

The constructor signature captures the key configuration knobs:

```python
# automationbench/runner.py  L49-59
def __init__(
    self,
    dataset: Dataset,
    rubric: vf.Rubric,
    tools: list[Callable] | None = None,
    max_turns: int = 25,
    allow_all_tools: bool = False,  # Enforce per-task tool restrictions
    toolset: str = "zapier",
    use_meta_tools: bool | None = None,  # None = infer from toolset
    search_top_k: int | None = None,  # Hard cap on search_tools top_k (None = no cap)
    **kwargs,
):
```

`max_turns` defaults to **25**, not 50. The system prompt instructs the model to complete
tasks within 50 turns, but that is a soft prompt-level budget; the hard guard that terminates
the episode is `max_turns=25`. See §6 for a full reconciliation with the documentation.

---

## 2. WorldState: One In-Process Pydantic Root Model

The simulated world is a single Pydantic `BaseModel` defined in `schema/world.py`. Its root:

```python
# automationbench/schema/world.py  L70-73
class WorldState(BaseModel):
    """Root world state containing all app states."""

    model_config = ConfigDict(extra="forbid")
```

`extra="forbid"` is load-bearing. It means that constructing a `WorldState` from a dataset
row's `initial_state` dict will raise a `ValidationError` immediately if the dict contains any
key that does not match a declared field. This is the first safety check that keeps benchmark
tasks honest — a misspelled app name is caught at episode start, not silently ignored.

Every field is declared with a typed sub-state class and a `default_factory`:

```python
# automationbench/schema/world.py  L76-128 (selected lines)
    meta: WorldMeta = Field(default_factory=WorldMeta)
    airtable: AirtableState = Field(default_factory=AirtableState)
    asana: AsanaState = Field(default_factory=AsanaState)
    ...
    gmail: GmailState = Field(default_factory=GmailState)
    google_calendar: GoogleCalendarState = Field(default_factory=GoogleCalendarState)
    google_sheets: GoogleSheetsState = Field(default_factory=GoogleSheetsState)
    hubspot: HubSpotState = Field(default_factory=HubSpotState)
    salesforce: SalesforceState = Field(default_factory=SalesforceState)
    slack: SlackState = Field(default_factory=SlackState)
    ...
    zendesk: ZendeskState = Field(default_factory=ZendeskState)
```

There are 44 app sub-states covering SaaS categories: CRM (Salesforce, HubSpot), productivity
(Notion, Trello, Asana, Monday, Jira, Confluence, Airtable), communication (Slack, Gmail,
Twilio, Zoom), marketing (Mailchimp, Buffer, Facebook, LinkedIn, Google Ads), HR (BambooHR,
Recruitee), finance (QuickBooks, Xero, Wave), and support (Zendesk, Freshdesk, Intercom,
Gorgias, HelpScout, Reamaze, Zoho Desk, Hiver, HelpCrunch).

Each sub-state holds typed record lists (e.g. `SalesforceState.contacts: list[Contact]`).
Tools mutate `world.<app>.<collection>` in memory — there is no persistence layer. Because the
object is passed by reference into every tool call, a mutation inside `salesforce_contact_update`
is immediately visible to the next tool call reading `world.salesforce.contacts`. This is what
makes the simulation deterministic and what makes rubric evaluation trivially cheap: after the
episode ends the grader reads the final `world` object directly.

The `WorldMeta` sub-model carries a `current_time` timestamp (defaulting to `datetime.now(utc)`)
and a `no_same_sender_noise` flag used by the noise-injection layer. Both share the same
`extra="forbid"` configuration.

---

## 3. Episode Lifecycle

See the [episode-flow diagram](figures/episode-flow.html) for a visual walkthrough of the full
pipeline.

### 3.1 Dataset Row: prompt + info

Each task row in the HuggingFace dataset has the shape:

```
{
  "prompt":  "<trigger message, e.g. an email notification>",
  "info": {
    "initial_state":  { ... WorldState dict ... },
    "zapier_tools":   [ "gmail_send_email", "slack_send_message", ... ],
    "assertions":     [ { "type": "...", ... }, ... ],
    "invariants":     [ ... ]
  }
}
```

HuggingFace normalizes dataset schemas across all rows: every key that exists in any row is
added to every row with a `None` default. Pydantic's `default_factory` pattern breaks when
`None` is explicitly passed — the factory is bypassed and `None` propagates into a typed field.
AutomationBench fixes this before touching Pydantic:

```python
# automationbench/runner.py  L22-35
def strip_none_values(obj):
    """
    Recursively strip None values from nested dicts and lists.

    HuggingFace Dataset normalizes schemas across rows, adding all possible keys
    and setting missing values to None. This breaks Pydantic's default_factory
    since None is passed instead of the field being omitted.
    """
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    else:
        return obj
```

This function is applied to both `initial_state` and `assertions` before any Pydantic
construction (runner.py L149-153).

### 3.2 setup_state

`setup_state` is called once per episode before the agent loop begins. It does three things:

```python
# automationbench/runner.py  L136-183
async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
    """Initialize per-task world state and filter tools."""
    state = await super().setup_state(state, **kwargs)

    # Get task info (deserialize from JSON if it's a string)
    info = state.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
        state["info"] = info

    # Initialize world state
    initial_state_dict = strip_none_values(info.get("initial_state", {}))

    # Also strip None values from assertions (same HuggingFace normalization issue)
    if "assertions" in info:
        info["assertions"] = [strip_none_values(a) for a in info["assertions"]]
    world = WorldState(**initial_state_dict)
    state["world"] = world
    state["initial_state"] = copy.deepcopy(initial_state_dict)
    ...
```

The `copy.deepcopy(initial_state_dict)` on L156 is a guard for the rubric grader: it needs
the original state to detect "free assertions" (assertions that were already true before the
agent did anything — a potential reward-hacking vector). Mutating `state["world"]` during the
episode must not corrupt this baseline copy.

After building `WorldState`, `setup_state` filters the tool list:

```python
# automationbench/runner.py  L159-181
    if self.use_meta_tools:
        # Meta-tools mode: model always gets the discovery tools
        filtered_tools = self._all_tool_defs
    elif self.allow_all_tools or self.toolset == "api":
        filtered_tools = self._all_tool_defs
    else:
        # If tools not specified, model gets NO tools (empty array)
        allowed_tools = info.get("zapier_tools", [])

        # Validate tool names - fail loudly if unknown tool specified
        all_tool_names = {t.name for t in self._all_tool_defs}
        unknown_tools = set(allowed_tools) - all_tool_names
        if unknown_tools:
            raise ValueError(
                f"Unknown tools specified in task: {unknown_tools}. Available: {all_tool_names}"
            )

        filtered_tools = [
            tool for tool in self._all_tool_defs if tool.name in allowed_tools
        ]

    state["tool_defs"] = filtered_tools
```

The test suite confirms the filtering contract: an empty `zapier_tools` list yields an empty
tool array (test_runner.py L65-82), and an unknown tool name raises `ValueError` with
`"Unknown tools"` in the message (L85-102).

### 3.3 Agent Loop

The loop is driven by `vf.StatefulToolEnv`. On each turn:

1. The model is called with the full message history plus `state["tool_defs"]`.
2. If the model emits tool calls, `env_response` is invoked.
3. `env_response` calls `_extract_usage_and_debug` (accumulating `prompt_tokens` and
   `completion_tokens` into `state["_usage"]`), then delegates to `super().env_response`.
4. Before dispatch, `update_tool_args` re-injects the hidden `world` argument (see §5).
5. In `zapier` mode, `_compress_meta_messages` post-processes the resulting `ToolMessage`s:
   previous-turn `search_tools` results that are now dead weight are rewritten to
   `[Previously found: name1, name2]`, saving tokens on long episodes.
6. The loop ends when the model produces a response with no tool calls, or when `max_turns`
   (default 25) is reached.

The `env_response` override in full:

```python
# automationbench/runner.py  L296-308
async def env_response(
    self,
    messages: vf.Messages,
    state: vf.State,
    **kwargs: Any,
) -> vf.Messages:
    """Process tool calls. In meta-tools mode, compress old search results."""
    # Extract usage/debug from the latest model response before processing tool calls
    self._extract_usage_and_debug(state)

    tool_messages = await super().env_response(messages, state, **kwargs)

    return tool_messages
```

### 3.4 Rubric Grading

After the loop terminates, the `verifiers` framework calls the rubric. The AutomationBench
rubric reads `state["world"]` (the final mutated state) and `state["initial_state"]` (the
deep-copied baseline) and evaluates each assertion. Partial credit is supported: a task
with five assertions where the model completed three returns a score of 0.6.

### 3.5 export_results

After all episodes in the run complete, `export_results` is called:

```python
# automationbench/export.py  L28-37
def export_results(
    outputs: list[dict[str, Any]],
    usage: RunUsage,
    model: str,
    domains: list[str],
    output_path: Path | str | None = None,
    duration_seconds: float | None = None,
    toolset: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
```

It collects per-task scores, per-assertion results, full message histories, token counts, and
cost data into a single JSON structure. The output powers the benchmark visualizer. Token counts
prefer `state["_usage"]` (accumulated per-turn by `_extract_usage_and_debug`) over a fallback
character-count estimate. The `benchmark_version` field is read from `pyproject.toml` — a
single source of truth that prevents version drift between runs.

---

## 4. The Three Toolset Modes

The toolset determines which tools the model sees and what they test. This is the primary
ablation axis in AutomationBench.

### 4.1 `zapier` — Two Meta-Tools (Default)

```python
# automationbench/runner.py  L79-85
    if self.use_meta_tools:
        from automationbench.tools.zapier.meta import execute_tool, make_search_tools, search_tools

        # Register only the 2 meta-tools for tool discovery
        actual_search = make_search_tools(max_top_k=search_top_k) if search_top_k is not None else search_tools
        self.add_tool(actual_search)
        self.add_tool(execute_tool, args_to_skip=["world"])
```

The model sees exactly **two** tools:

- `search_tools(query, top_k=5)` — BM25 search over the tool corpus, returning a JSON list of
  `{name, description, parameters}` entries. The model must call this first to discover what
  tools exist.
- `execute_tool(tool_name, arguments)` — executes any named tool by dispatching through the
  full registry.

This is the headline mode: it tests both *discovery* (can the model find the right tool?) and
*execution* (can it call it correctly?). When the paper says "two tools," it means this mode.

`use_meta_tools` is automatically `True` when `toolset == "zapier"` and `False` for
`limited_zapier` and `api` (runner.py L73-77).

### 4.2 `limited_zapier` — Named Tools Filtered Per Task

```python
# automationbench/runner.py  L86-93
    else:
        # Register tools based on toolset selection
        tool_list = API_TOOLS if toolset == "api" else ALL_TOOLS  # limited_zapier also uses ALL_TOOLS (filtered per-task in setup_state)
        for tool in tool_list:
            # Auto-detect args_to_skip: skip 'world' only if the function accepts it
            sig = inspect.signature(tool)
            args_to_skip = ["world"] if "world" in sig.parameters else []
            self.add_tool(tool, args_to_skip=args_to_skip)
```

In `limited_zapier` mode all ~400 named tools are registered at env construction time, but
`setup_state` filters the per-task view to exactly those listed in `info.zapier_tools`. The
model sees named functions with full signatures — no discovery step required.

This mode isolates **execution skill** from **discovery skill**: if a model scores much lower
on `zapier` than on `limited_zapier`, the bottleneck is tool discovery, not argument
construction. The test `test_setup_state_filters_tools` directly validates this:

```python
# tests/test_runner.py  L41-62
async def test_setup_state_filters_tools(self):
    env = AutomationBenchEnv(dataset=dataset, rubric=rubric, allow_all_tools=False, toolset="limited_zapier")
    state = cast(vf.State, {
        "info": {
            "initial_state": {},
            "zapier_tools": ["salesforce_query"],  # Only one tool
        }
    })
    state = await env.setup_state(state)
    tool_names = [t.name for t in state["tool_defs"]]
    assert tool_names == ["salesforce_query"]
```

### 4.3 `api` — Three REST-Shaped Tools

```python
# automationbench/tools/api/__init__.py
API_TOOLS = [api_search, api_fetch, base64_encode]
```

The model sees three tools that mirror a REST API interface:

- `api_search` — BM25 search over a tab-separated endpoint index (`schemas/index.txt`), lazily
  rebuilt from `.jsonc` schema files. The model searches for an endpoint by keyword.
- `api_fetch(method, url, params, body)` — executes REST-style calls. The harness routes the
  URL via `_url_to_internal_path()` to the correct in-process world mutation. Critically, the
  routing layer normalizes ~15 known model URL hallucinations (e.g. `slack.googleapis.com` →
  correct Slack router) so a URL typo isn't scored as a task failure.
- `base64_encode` — utility for constructing payloads that require base64 encoding.

The `api` mode tests a different skill profile: the model must reason about HTTP methods,
URL structure, and request/response shapes rather than function names and typed parameter
schemas. A model that is strong on Zapier-style function calling but weak on REST semantics
(or vice versa) will show a systematic gap between modes.

### 4.4 What Each Mode Ablates

| Mode | Tools visible | Tests | Isolates |
|------|---------------|-------|---------|
| `zapier` | `search_tools` + `execute_tool` | Discovery + Execution | Neither: joint score |
| `limited_zapier` | ~N named functions (per-task) | Execution only | Discovery (difference vs zapier) |
| `api` | `api_search` + `api_fetch` + `base64_encode` | REST semantics | Abstraction style |

---

## 5. The `world` Injection Trick

Every tool implementation takes `world: WorldState` as a parameter — it is how the tool reads
and mutates the simulated backend. But exposing a complex Pydantic model in the JSON schema
that the model sees would be both useless (the model cannot pass a Python object) and
potentially confusing. AutomationBench removes `world` from the model-facing schema entirely.

The mechanism lives in `tool_wrapper.py`:

```python
# automationbench/tool_wrapper.py  L13-38
def _create_tool_wrapper(func: Callable, args_to_skip: list[str]) -> Callable:
    """Create a wrapper function with skipped args removed from signature.

    This is needed because convert_func_to_oai_tool uses the function signature
    to generate JSON schema, and the strict schema validation fails on complex
    types like WorldState before we can strip them.
    """
    original_sig = inspect.signature(func)
    original_hints = get_type_hints(func)

    # Build new parameters without skipped args
    new_params = [p for name, p in original_sig.parameters.items() if name not in args_to_skip]
    new_sig = original_sig.replace(parameters=new_params)

    # Build new type hints without skipped args
    new_hints = {k: v for k, v in original_hints.items() if k not in args_to_skip}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Replace signature and annotations
    wrapper.__signature__ = new_sig  # type: ignore
    wrapper.__annotations__ = new_hints

    return wrapper
```

The key operations are:
1. Strip `world` from `inspect.signature` — so `convert_func_to_oai_tool` (the verifiers
   utility that generates the JSON schema) never sees it.
2. Strip `world` from `__annotations__` — so `get_type_hints` used for schema generation
   also produces a clean result.
3. The underlying function is still called with all arguments via `functools.wraps`.

The re-injection happens at dispatch time in `update_tool_args`:

```python
# automationbench/runner.py  L118-134
def update_tool_args(
    self,
    tool_name: str,
    tool_args: dict,
    messages: vf.Messages,
    state: vf.State,
    **kwargs,
) -> dict:
    """Auto-inject skipped args into tool calls."""
    updated_args = dict(tool_args)

    # Auto-inject any skipped args that exist in state
    skipped = self.skipped_args.get(tool_name, [])
    if "world" in skipped:
        updated_args["world"] = state["world"]

    return updated_args
```

The test that directly validates this contract:

```python
# tests/test_runner.py  L138-154
def test_update_tool_args_injects_world(self):
    world = WorldState()
    state = cast(vf.State, {"world": world})

    updated = env.update_tool_args(
        "salesforce_query",
        {"object_type": "Contact", "where_clause": "Email = 'test@example.com'"},
        [], state,
    )
    assert updated["world"] is world
```

Note the `is` check: the injected `world` is the exact same object, not a copy. This is
intentional — mutations inside the tool must be visible to subsequent tool calls in the same
episode.

The auto-detection of which tools need `args_to_skip` happens at construction:

```python
# automationbench/runner.py  L90-93
            sig = inspect.signature(tool)
            args_to_skip = ["world"] if "world" in sig.parameters else []
            self.add_tool(tool, args_to_skip=args_to_skip)
```

Only tools whose Python signature actually includes a `world` parameter get the skip treatment.
Utility tools (like `base64_encode`) that don't touch world state pass through unmodified.

---

## 6. Reconciling Documentation vs. Code

Two specific tensions exist between the paper-level description and the actual implementation.

### 6.1 "Two tools" refers only to `zapier` mode

The paper and README describe AutomationBench as giving the agent "two tools." This is
accurate only for the default `zapier` mode (`search_tools` + `execute_tool`). The benchmark
ships with three distinct modes, each with a different tool interface:

| Claim | Accurate scope |
|-------|---------------|
| "Two tools" | `toolset="zapier"` only |
| "~400 named tools" | `toolset="limited_zapier"`, per-task filtered |
| "REST-style tools" | `toolset="api"` |

When you read experimental results, always check which `toolset` was used. The difference
in score between `zapier` and `limited_zapier` is the estimated cost of tool discovery.

### 6.2 "Max 50 steps" is a prompt hint, not the hard guard

The system prompt instructs the model to complete tasks within 50 turns. The hard guard in the
code is `max_turns=25` (runner.py L54). These are two different knobs:

- `max_turns=25` — enforced by `vf.StatefulToolEnv`; the episode terminates unconditionally
  when 25 turns elapse. Increasing this requires passing `max_turns=N` to `AutomationBenchEnv`.
- The system prompt budget — a soft instruction that influences model behavior (many models
  try to be efficient if told they have limited steps) but has no enforcement mechanism.

For a model that calls `search_tools` once and `execute_tool` once per subtask, 25 turns
supports roughly 12 subtasks per episode. For complex multi-step workflows (search → validate
→ execute → verify → chain) the 25-turn ceiling is tighter than the 50-turn prompt suggests.

---

## 7. Meta-Message Compression in `zapier` Mode

In `zapier` mode, `search_tools` returns verbose JSON containing tool names, docstrings,
and full parameter schemas. After the model acts on those results by calling `execute_tool`,
the verbose payloads are dead weight in the context window. `_compress_meta_messages` handles
this:

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

The compression is turn-aware: it only rewrites search results from *previous* turns, never
from the current turn. The comment explains why — if the model issues `search_tools` and
`execute_tool` in the same turn (a valid parallel call pattern), the schema for the searched
tool must still be in context when the argument names are validated. Compressing it
prematurely would cause hallucinated argument names in the `execute_tool` call.

Previous-turn search results are rewritten from:

```json
[{"name": "gmail_send_email", "description": "...", "parameters": {...}}, ...]
```

to:

```
[Previously found: gmail_send_email, slack_send_message]
```

This is only applied when the content is ≥200 characters (short results are not worth
rewriting) and is only parseable as a JSON list of objects.

---

## Summary

AutomationBench builds its determinism and reproducibility guarantee on one structural choice:
the entire world is a single in-process Pydantic object with `extra="forbid"`. This choice
propagates into every other design decision:

- `setup_state` constructs `WorldState(**initial_state_dict)` once per episode and passes it
  into the harness state dict.
- `update_tool_args` re-injects `world` at dispatch time after stripping it from the
  model-facing schema via `tool_wrapper.py`.
- The rubric reads `state["world"]` at episode end — no round-trips, no serialization.
- `export_results` bundles the end state, assertion results, and token costs into a JSON
  visualizer feed.

The three toolset modes (`zapier`, `limited_zapier`, `api`) are ablations over this same
infrastructure, varying only which tool interface the model sees — not the underlying world
simulation, not the grading logic, not the episode lifecycle. That clean separation is what
makes the ablation results interpretable.

See also: [[automationbench-harness]], [[automationbench-overview]]
