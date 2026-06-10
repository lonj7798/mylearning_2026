<!-- chapter: ch-04
track: internals
kind: content
title: Execute and the Simulated World
deps: [ch-03]
sources: [[automationbench-harness]], [[automationbench-tasks-grading]]
-->

# Chapter 04 — Execute and the Simulated World

> **Core insight.** Every tool call in AutomationBench resolves to a Python
> function that reads from and writes to a single in-process Pydantic object.
> There is no HTTP server, no database connection, no subprocess, and no
> randomness beyond what is seeded at episode start. That design choice is what
> makes variance across repeated runs below 1 % and makes grading purely
> deterministic.

> **Guideline.** When building an agentic benchmark, keep the world state
> in-process and validated at write time (`extra="forbid"`). Push all API
> surface fidelity — pagination, 4xx codes, required-field checks — into the
> routing layer, not into the grading layer. Grading must never compensate for
> a loose world model.

---

## 1. The execution path from tool call to world mutation

Ch-02 introduced the `world` injection trick and Ch-03 showed how an agent
finds a tool. This chapter covers what happens when the agent actually calls
one.

There are two execution paths depending on the toolset mode:

| Toolset mode | Entry point | Route |
|---|---|---|
| `zapier` / `limited_zapier` | `execute_tool(world, tool_name, arguments)` | `ToolRegistry.execute` → named Python function |
| `api` | `api_fetch(world, method, url, params, body)` | `_url_to_internal_path` → `route_<app>` → named Python function |

Both paths converge on the same underlying Python functions in
`automationbench/tools/api/impl/<app>.py`. The routing layer is purely a
dispatch concern.

---

## 2. `execute_tool` → `ToolRegistry.execute`

`execute_tool` is the agent-facing entry point for Zapier mode
(`automationbench/tools/zapier/meta.py`, L189–204):

```python
# automationbench/tools/zapier/meta.py  L189–204
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

`world` is typed in the signature but never appears in the JSON schema the
agent sees — Ch-02's `args_to_skip=["world"]` strips it before schema
generation. The agent supplies `tool_name` and `arguments` (a JSON string);
`world` arrives via the harness.

`ToolRegistry.execute` at L98–110 does three things:

```python
# automationbench/tools/zapier/meta.py  L98–110
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

1. **Lookup** — `tool_name` is checked against `_tool_map` (built at registry
   init from `ALL_TOOLS`). An unknown name raises `ValueError` with an
   explicit hint to call `search_tools` first. This is intentional: the agent
   cannot guess-and-execute.

2. **JSON-parse + inject** — `arguments` is decoded with `json.loads`. Then
   `world` (and any other injected kwargs) is merged in via `{**parsed_args,
   **injected}`. Agent-supplied args can never shadow `world` because
   `args_to_skip` already removed it from the published schema.

3. **Dispatch + coerce** — The function is called with the merged kwargs. If
   the return value is already a `str` (the common case: tools return
   `json.dumps(...)`) it is passed through unchanged. Non-strings are
   serialized with `json.dumps`. The agent always receives a JSON string.

The `_create_tool_wrapper` in `automationbench/tool_wrapper.py` (L13–38) is
responsible for the schema-facing signature:

```python
# automationbench/tool_wrapper.py  L13–38
def _create_tool_wrapper(func: Callable, args_to_skip: list[str]) -> Callable:
    """Create a wrapper function with skipped args removed from signature."""
    original_sig = inspect.signature(func)
    original_hints = get_type_hints(func)

    new_params = [p for name, p in original_sig.parameters.items() if name not in args_to_skip]
    new_sig = original_sig.replace(parameters=new_params)
    new_hints = {k: v for k, v in original_hints.items() if k not in args_to_skip}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__signature__ = new_sig
    wrapper.__annotations__ = new_hints
    return wrapper
```

The wrapper is only used at schema-generation time (`_get_parameter_schema`,
L84–92). At execution time, `func(**merged)` calls the real function with
`world` already re-injected. The wrapper is never invoked during an actual
tool call.

---

## 3. API mode: `api_fetch` → `_url_to_internal_path` → `route_<app>`

When the agent is in `api` mode it calls `api_fetch` with a full REST URL. The
routing is a two-stage lookup (`automationbench/tools/api/fetch.py`):

```python
# automationbench/tools/api/fetch.py  L252–283
def api_fetch(
    world: WorldState,
    method: str,
    url: str,
    params: Optional[str] = None,
    body: Optional[str] = None,
) -> str:
    try:
        parsed_params = _coerce_to_dict(params)
        body_dict = _coerce_to_dict(body)
    except json.JSONDecodeError as e:
        return json.dumps({"error": {"code": 400, "message": f"Invalid JSON: {e}"}})

    internal_path, router = _url_to_internal_path(url)
    if router is not None:
        return router(world, method, internal_path, parsed_params, body_dict)

    return json.dumps({"error": {"code": 404, "message": f"Unknown API URL: {url}"}})
```

**Stage 1 — `_url_to_internal_path`** translates a full URL into an
`(internal_path, router_fn)` pair. The function maintains three lookup tables
(L74–248):

- `_STATIC_URL_ROUTERS` — a list of `(url_prefix, internal_prefix, router)`
  triples. Each entry strips the real-world URL prefix and prepends a
  canonical internal prefix. Example:
  `("https://api.hubapi.com/", "hubspot/", route_hubspot)` transforms
  `https://api.hubapi.com/crm/v3/objects/contacts` →
  `hubspot/crm/v3/objects/contacts`.

- `_DYNAMIC_HOST_ROUTERS` — hostname-suffix matching for services where the
  subdomain encodes the customer's account (Salesforce, Zendesk, Freshdesk,
  Mailchimp). `https://mycompany.my.salesforce.com/services/data/v60/query`
  matches `.salesforce.com` and becomes `salesforce/services/data/v60/query`.

- Special-case blocks for services that fan out across multiple routers on the
  same host. `graph.facebook.com` (L143–157) splits into four separate routers
  — Instagram, Facebook Pages, Facebook Conversions, Facebook Lead Ads — based
  on path structure (whether the path ends with `/events`, `/leads`,
  `/leadgen_forms`, contains `act_`, has a single segment, etc.). Similarly,
  `api.linkedin.com/rest/` (L159–165) branches on whether the path starts with
  `conversionEvents`. And `*.atlassian.net` (L167–172) branches on `rest/`
  (→ Jira) vs `wiki/` (→ Confluence).

**Stage 2 — `route_<app>`** receives `(world, method, internal_path, params,
body_dict)` and dispatches to the correct impl function. Each router is built
with `make_router` from `automationbench/utils/routing.py`:

```python
# automationbench/utils/routing.py  L20–53
def make_router(
    routes: list[tuple[str, str, str]],
    handlers: dict[str, Callable],
) -> Callable:
    def _route(world, method, path, params, body_dict) -> str:
        method_upper = method.upper()
        for route_method, pattern, handler_key in routes:
            if route_method != method_upper:
                continue
            match = re.match(pattern, path)
            if not match:
                continue
            ids = list(match.groups())
            return handlers[handler_key](world, ids, params, body_dict)
        return json.dumps({"error": {"code": 404, "message": f"No handler for {method} {path}"}})
    return _route
```

Routes are regex triples `(method, pattern, handler_key)`. Capture groups in
the pattern become the `ids` list passed to the handler, so
`r"slack/conversations\.history$"` (no groups) yields `ids=[]`, while
`r"items/([^/]+)$"` yields `ids=["abc123"]`. First-match-wins; method
comparison is case-insensitive (L42). An unmatched request returns a 404
JSON — never raises.

The Slack router illustrates the pattern
(`automationbench/tools/api/routes/slack.py`, L35–92):

```python
# automationbench/tools/api/routes/slack.py  L35–92
_ROUTES: list[tuple[str, str, str]] = [
    ("GET",  r"slack/conversations\.list$",    "conversations_list"),
    ("POST", r"slack/chat\.postMessage$",      "chat_post_message"),
    ("GET",  r"slack/conversations\.history$", "conversations_history"),
    # Model hallucinations: channels/C_XXX/messages → conversations.history
    ("GET",  r"slack/channels/([^/]+)/messages$", "channels_messages_alias"),
    # Model hallucinations: channels.C_XXX/messages (dot) → conversations.history
    ("GET",  r"slack/channels\.([^/]+)/messages$", "channels_dot_messages_alias"),
    # Model hallucinations: channels.history → conversations.history
    ("GET",  r"slack/channels\.history$",      "conversations_history"),
    # ... (19 total routes)
]

_HANDLERS = {
    "conversations_list":    lambda w, ids, p, b: slack_conversations_list(w, **p),
    "chat_post_message":     lambda w, ids, p, b: slack_chat_post_message(w, **b),
    "conversations_history": lambda w, ids, p, b: slack_conversations_history(w, **p),
    "channels_messages_alias":
        lambda w, ids, p, b: slack_conversations_history(w, channel=ids[0], **p),
    "channels_dot_messages_alias":
        lambda w, ids, p, b: slack_conversations_history(w, channel=ids[0], **p),
    # ...
}

route_slack = make_router(_ROUTES, _HANDLERS)
```

Handlers are lambdas that adapt the `(world, ids, params, body_dict)` calling
convention to the underlying impl function's kwargs. GET endpoints read from
`params`; POST/PATCH endpoints read from `body_dict`. The capture-group trick
is used in `channels_messages_alias`: `ids[0]` carries the channel ID parsed
from the URL segment, which is forwarded as `channel=` to
`slack_conversations_history`.

---

## 4. URL-hallucination tolerance: the deliberate over-forgiveness tradeoff

The routing tables contain ~15 known model mistakes, normalized silently to
the correct router. A curated sample from `fetch.py`:

| Hallucinated URL prefix | Canonical intent | Line |
|---|---|---|
| `https://slack.googleapis.com/` | Slack Web API | L89 |
| `https://www.slack.com/api/` | Slack Web API (extra `www.`) | L86 |
| `https://slack.com/` | Slack Web API (missing `/api/`) | L88 |
| `https://www.googleapis.com/gmail/` | Gmail API | L76 |
| `https://www.googleapis.com/sheets/` | Google Sheets | L80 |
| `https://www.sheets.googleapis.com/` | Google Sheets (hallucinated domain) | L81 |
| `https://calendar.googleapis.com/` | Google Calendar | L79 |
| `https://sandbox-quickbooks.api.intuit.com/` | QuickBooks (sandbox alias) | L115 |
| `https://3.basecamp.com/` | Basecamp 3 (alias) | L113 |

The intent is explicit in the inline comments: `# models sometimes use
www.googleapis.com for Gmail` (L76), `# models hallucinate this domain` (L81,
L89).

**The design tradeoff.** Normalizing hallucinated URLs is a deliberate
eval-robustness choice: it removes noise from the measurement. If an agent
gets the right operation but produces `slack.googleapis.com` instead of
`slack.com/api/`, the benchmark would punish it with a 404 for a mistake that
says nothing about reasoning or workflow capability. The benchmark's thesis is
that the interesting difficulty is in cross-app coordination and policy
adherence — not in URL trivia that a documentation lookup would immediately
resolve.

The counterargument is that over-forgiveness could mask a real capability gap.
A production Zapier automation that talks to a wrong base URL actually fails;
the benchmark would not catch an agent that works in simulation but hallucinates
URLs in deployment. The benchmark trades sim-to-real fidelity on URL
correctness for reduced measurement variance on the capabilities it actually
cares about. Researchers who want to measure URL accuracy should run in api mode
with hallucination tolerance disabled — but that option is not currently exposed
as a CLI flag.

---

## 5. App state: Pydantic with `extra="forbid"` and `to_display_dict()`

Every app schema uses `model_config = ConfigDict(extra="forbid")`. This is a
hard guarantee: any key not declared on the model triggers a `ValidationError`
at write time. There is no silent field drop, no `extra="ignore"` permissiveness
that would let a task seed or a tool implementation smuggle in undeclared data.

`HubSpotContact` in `automationbench/schema/hubspot.py` illustrates the
pattern:

```python
# automationbench/schema/hubspot.py  L29–72
class HubSpotContact(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    id: str = Field(default_factory=generate_hubspot_id)
    email: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    jobtitle: Optional[str] = None
    lifecyclestage: str = Field(default="lead", validation_alias="lifecycle_stage")
    lead_score: Optional[int] = None
    properties: Dict[str, str] = Field(default_factory=dict)
    # ... (UTM attribution, open_ticket, industry, NPS, billing, notes ...)
    utm_source: Optional[str] = None
    utm_campaign: Optional[str] = None
    nps_score: Optional[int] = None
    payment_status: Optional[str] = None
    lifetime_value: Optional[float] = None
    notes: Optional[str] = None
```

Almost every field is `Optional[T] = None`. A freshly seeded contact carries
only the fields relevant to its task. This is not laziness — it is a
deliberate realism choice: real CRM records are sparse. Most contacts do not
have an NPS score, a UTM attribution chain, and a payment status
simultaneously.

`to_display_dict()` (L74–143) enforces the sparse presentation:

```python
# automationbench/schema/hubspot.py  L74–143
def to_display_dict(self) -> dict:
    result = {
        "id": self.id,
        "email": self.email,
        "firstname": self.firstname,
        "lastname": self.lastname,
        "phone": self.phone,
        "company": self.company,
        "jobtitle": self.jobtitle,
        "lifecyclestage": self.lifecyclestage,
        "hs_object_id": self.id,
        "createdAt": self.created_at.isoformat(),
        "updatedAt": self.updated_at.isoformat(),
    }
    if self.lead_score is not None:
        result["lead_score"] = self.lead_score
    if self.utm_source:
        result["utm_source"] = self.utm_source
    if self.nps_score is not None:
        result["nps_score"] = self.nps_score
    if self.payment_status is not None:
        result["payment_status"] = self.payment_status
    if self.lifetime_value is not None:
        result["lifetime_value"] = self.lifetime_value
    if self.notes is not None:
        result["notes"] = self.notes
    return result
```

The core identity fields (`id`, `email`, names, `lifecyclestage`,
timestamps) are always present. All optional enrichments are added only when
non-None. The agent sees the same JSON shape a real HubSpot API response would
return for a lightly-populated contact: a handful of fields, not 30 null-padded
keys.

This matters for agent behavior. An agent that pattern-matches on field
presence (e.g., branches on whether `payment_status` is present) behaves
correctly under this model. An agent that always reads `record["payment_status"]`
without checking would raise a `KeyError` — the same failure mode as a live API.

The same `extra="forbid"` + sparse `to_display_dict()` pattern recurs across
all 44 app schemas. `WorldState` itself applies it at the root level
(`automationbench/schema/world.py`, L70–73):

```python
# automationbench/schema/world.py  L70–73
class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: WorldMeta = Field(default_factory=WorldMeta)
    airtable: AirtableState = Field(default_factory=AirtableState)
    # ... (44 app-state fields)
```

Every layer of the hierarchy enforces this constraint. There is no gap between
`WorldState` and a leaf record model where an unvalidated write could slip
through.

---

## 6. Real-API fidelity: pagination, required fields, and 4xx codes

The simulation does not pretend HTTP mechanics away. Route handlers emit
structured error responses for the same conditions a live API would reject,
and list endpoints paginate their output.

**4xx codes.** `test_api_fetch.py` (L224–251) codifies the contract:

```python
# tests/test_api_fetch.py  L224–251
class TestApiFetch:
    def test_unknown_url_returns_404(self):
        world = WorldState()
        result = json.loads(api_fetch(world, "GET", "https://unknown.example.com/data"))
        assert result["error"]["code"] == 404

    def test_invalid_json_params(self):
        world = WorldState()
        result = json.loads(
            api_fetch(world, "GET",
                      "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                      params="{bad")
        )
        assert result["error"]["code"] == 400
        assert "Invalid JSON" in result["error"]["message"]

    def test_valid_gmail_route(self):
        world = WorldState()
        result = json.loads(
            api_fetch(world, "GET",
                      "https://gmail.googleapis.com/gmail/v1/users/me/messages")
        )
        assert result.get("error", {}).get("code") != 404
```

The same pattern applies within routers. `test_api_routes.py` (L88–93,
L147–151, L226–231) checks that unknown paths within a recognized host still
return 404:

```python
# tests/test_api_routes.py  L88–93
def test_unknown_route_404(self):
    world = WorldState()
    result = json.loads(route_gmail(
        world, "GET", "gmail/v1/nonexistent", {}, {}
    ))
    assert result["error"]["code"] == 404
```

**Update operations and not-found 4xx.** The Gorgias impl tests
(`tests/test_api_impl_gorgias.py`, L77–80) verify that an update on a
non-existent ticket returns an error rather than silently creating or
discarding:

```python
# tests/test_api_impl_gorgias.py  L77–80
def test_update_not_found(self):
    world = WorldState()
    result = json.loads(gorgias_tickets_update(world, ticket_id="fake"))
    assert "error" in result
```

**End-to-end state mutations.** The same test file (L31–34) confirms that a
successful create both returns the created record and appends it to
`world.gorgias.tickets`:

```python
# tests/test_api_impl_gorgias.py  L31–34
def test_create_basic(self):
    world = WorldState()
    result = json.loads(gorgias_tickets_create(world, subject="Help"))
    assert result["subject"] == "Help"
    assert len(world.gorgias.tickets) == 1
```

This is the core loop: the agent calls a tool, the tool mutates `world`, the
grader reads `world` after the episode ends. The test above is the minimal
proof that all three steps work correctly together.

**Route dispatch fidelity.** `test_routing.py` (L54–64) specifies the
`make_router` failure modes explicitly: no matching route → 404, method
mismatch → 404, case-insensitive method → succeeds:

```python
# tests/test_routing.py  L54–64
def test_no_matching_route_returns_404(self):
    routes = [("GET", r"items$", "list_items")]
    router = make_router(routes, {"list_items": lambda w, i, p, b: "[]"})
    result = json.loads(router(WorldState(), "GET", "nonexistent", {}, {}))
    assert result["error"]["code"] == 404

def test_method_mismatch(self):
    routes = [("GET", r"items$", "list_items")]
    router = make_router(routes, {"list_items": lambda w, i, p, b: "[]"})
    result = json.loads(router(WorldState(), "POST", "items", {}, {}))
    assert result["error"]["code"] == 404
```

The fidelity level is deliberate. The benchmark targets the friction of a real
API (correct method, correct path, required fields present, valid JSON body)
without replicating its flakiness (rate limits, auth token expiry, transient
5xx, network latency). That is the dividing line between a useful simulation
and a production client.

---

## 7. Why in-process simulation buys determinism and <1% variance

The core design decision — everything runs inside a single Python process — has
four concrete consequences.

**No I/O non-determinism.** A live sandbox introduces HTTP round-trips,
authentication state, rate-limit headers, and server-side side effects. Any of
these can differ between two runs of the same episode. In-process, the only
source of non-determinism is Python's random number generator, which
AutomationBench seeds deterministically per `example_id` (see Ch-07 for the
noise seeding). Two runs of episode 501 on the same model will hit identical
`WorldState` snapshots.

**Instant state inspection.** The grader reads `world` directly after the
agent loop ends. There is no need to query a DB, poll an API, or deserialize
a log. `world.hubspot.contacts[0].lifecyclestage` is a Python attribute access.
Assertion logic (`rubric/assertions/<app>.py`) is pure field comparison.

**Atomic rollback.** Each episode gets a fresh `WorldState()`. There is no
shared mutable state between episodes. Parallelizing evaluation across tasks
is safe by construction — no locking, no teardown, no fixture cleanup.

**Reproducibility for RL.** The `partial_credit` score (Ch-06) is a dense
reward signal usable directly for RL training. Because the environment is
deterministic, a training loop can re-run the same episode with the same seed
and get the same reward signal. This is the property that makes AutomationBench
double as a training environment (Ch-10).

The cost is sim-to-real transfer. An agent that scores 60 % in-process may
score lower against a live Salesforce instance for reasons that have nothing
to do with reasoning: a stale OAuth token, an API endpoint moved in a minor
version bump, a field name that changed in a spring product update. The
benchmark measures reasoning about tool-using workflows; it explicitly does
not measure the DevOps overhead of maintaining real API credentials at scale.
For the purposes AutomationBench targets — comparing model capability across
releases, measuring the effect of toolset mode, generating RL training episodes
— the tradeoff is correct.

---

## Key takeaways

- `execute_tool` → `ToolRegistry.execute`: three steps — lookup by name
  (raises on unknown), `json.loads` the argument string, merge in `world` via
  `**injected`, call the real function. String results pass through; others are
  `json.dumps`'d.
- `api_fetch` → `_url_to_internal_path` → `make_router`: two dispatch stages.
  First stage maps a full URL to a `(canonical_path, router_fn)` pair using
  static prefix tables, dynamic hostname-suffix tables, and special-case fan-out
  blocks (Facebook, LinkedIn, Atlassian). Second stage regex-matches
  `(method, path)` against a per-app route table and calls the impl function.
- URL-hallucination tolerance normalizes ~15 known model mistakes to the correct
  router. This deliberately trades sim-to-real URL fidelity for reduced
  measurement noise on the capabilities the benchmark actually targets.
- `extra="forbid"` on every Pydantic model class prevents undeclared writes at
  every layer. `to_display_dict()` omits `None` fields, producing sparse
  records that mimic real API responses.
- In-process execution gives <1 % run-to-run variance, instant state
  inspection, trivial parallelism, and RL-compatible determinism. The cost is
  explicit: URL accuracy and DevOps friction are not measured.
