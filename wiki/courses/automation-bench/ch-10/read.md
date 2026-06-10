<!-- chapter: ch-10
     track: extension
     kind: lab
     title: Lab: Run It, Extend It, Reuse It as an RL Environment
     deps: [ch-09]
     sources: [[automationbench-harness]], [[automationbench-tasks-grading]], [[benchmark-comparison]]
     capstone_for: automation-bench
-->

# Chapter 10 — Lab: Run It, Extend It, Reuse It as an RL Environment

> **Core insight.** AutomationBench is not just an evaluator — it is a dense-reward RL
> environment. `partial_credit = passed / total` (weight 1.0 in `create_rubric`) is a
> continuous signal that the `verifiers` training loop can use directly. Every design choice
> in the harness — typed in-process world, deterministic seeded noise, must-not-occur guards,
> free-assertion exclusion — was made precisely so that reward is honest and hackable-resistant
> at training time, not just at eval time.
>
> **Guideline.** Run before you extend; extend before you design. Stage 1 teaches you what
> a 25-turn episode actually looks like. Stage 2 teaches you what the extension surface costs.
> Stage 3 teaches you why `partial_credit` is a better training signal than a sparse 0/1.
> Stage 4 turns those lessons into a benchmark design for your own agent. Stage 5 closes the
> loop: the Lina TMR sales-call simulator inherits AutomationBench's assertion engineering —
> τ-style conversation + AB-style end-state grading + pass^k reliability.

---

## Goal

Three artifacts, each reproducible from your repo:

1. **A run artifact.** A `visualizer/runs/local/<model>-<timestamp>.json` produced by a
   real evaluation — at minimum the `simple` domain (200 tasks, ~$0.50 at haiku-class
   pricing) but ideally one scored domain (`sales` or `hr`). Include the visualizer
   screenshot or terminal output in your memo.
2. **An extension.** One of: (a) a new task added to an existing domain, OR (b) a new
   assertion type registered in the rubric. Must run cleanly under `uv run pytest tests/`.
3. **A deliverable memo.** Either `ab-task-spec.md` (new task + rubric) or
   `lina-bench-spec.md` (Lina TMR benchmark spec). See Stage 5 for the template.

---

## Full-budget path

Target: standard cloud API (~$5–15 for one domain), local Python env.

- **Run.** All six scored domains (600 tasks). Use `--max-concurrent 100` (the default)
  and `--reasoning-effort medium` for a thinking model. Budget ~$10 with `claude-haiku-4-5`.
- **Visualize.** Open `compare.html` with two runs — baseline vs `--toolset limited_zapier`
  — to see the tool-discovery vs tool-execution gap.
- **Extend.** Add a new `sales` task (cross-app, at least 4 assertions including one
  negative guard) and a new `@AssertionRegistry.register` type.
- **Memo.** The full Lina TMR benchmark spec (see Stage 5).

## Resource-constrained path

Target: minimal API spend (~$0.50–2), no GPU needed.

- **Run.** `simple` domain only (`--domains simple`). ~200 tasks, haiku-class model.
  Add `--num-examples 20` for a smoke test first.
- **Extend.** One new assertion type only — skip the full task constructor if time is short.
- **Memo.** Either deliverable option.

---

## Stage 1 — Run a real evaluation

### Install and first smoke test

```bash
# automationbench/README.md — Quick start
git clone https://github.com/zapier/AutomationBench.git
cd AutomationBench

# Install dependencies
uv sync

# Set your API key (or create a .env file)
export OPENAI_API_KEY=sk-...
# Anthropic: auto-detected via `claude-*` prefix
export ANTHROPIC_API_KEY=sk-ant-...

# Smoke test: 5 examples, simple domain
uv run auto-bench --model claude-haiku-4-5-20251001 \
  --domains simple \
  --num-examples 5

# Full simple domain
uv run auto-bench --model claude-haiku-4-5-20251001 --domains simple

# Single scored domain
uv run auto-bench --model claude-haiku-4-5-20251001 --domains sales
```

The default `--toolset` is `api` (REST-shaped tools: `api_search`, `api_fetch`,
`base64_encode`). Switch to `zapier` for the discovery-tested headline mode — the agent
must call `search_tools(query)` over ~400 tools to find what it needs
([[automationbench-harness]] §The three toolset modes):

```bash
# zapier toolset: discovery + execution tested together
uv run auto-bench --model claude-haiku-4-5-20251001 \
  --domains sales \
  --toolset zapier \
  --export-json visualizer/runs/local/haiku-sales-zapier.json
```

### The CLI option surface

`automationbench/scripts/eval.py` exposes the full option set. Key flags quoted verbatim:

```
--model          Model name (default: gpt-5-mini)
--domains        Comma-separated domains or "all"
--toolset        api | zapier | limited_zapier
--num-examples   Number of examples (-1 for all)
--max-steps      Max model response steps per task (default: 50)
--max-concurrent Max concurrent tasks (default: 100)
--reasoning-effort  low / medium / high / xhigh / max
--input-cost     Per-token input cost in USD (overrides lookup)
--output-cost    Per-token output cost in USD (overrides lookup)
--export-json    Path to export results JSON
--save-every     Save incremental results every N tasks (default: 1)
--skip           Skip first N tasks
--tasks          Comma-separated task names to run
```

Cost override is useful when running a local model or a fine-tuned checkpoint behind a
proxy — pass `--input-cost 0.000001 --output-cost 0.000002` to force accurate $/task
reporting without a pricing-DB entry (`pricing.py` resolves by exact → normalized → alias
lookup with 24h llm-prices.com cache, hardcoded fallback).

### What to watch during an episode

`eval.py:248` passes `state_columns=["_usage", "_debug", "_assertion_results", "_end_state"]`
to `env.evaluate`. After the run the JSON export contains per-task:

```json
{
  "task": "sales.multi_hop_lookup",
  "reward": 0.6,
  "_assertion_results": [
    {"type": "salesforce_field_equals", "passed": true,  "excluded": false, "params": {...}},
    {"type": "gmail_message_sent_to_with_body_contains", "passed": false, "excluded": false, "params": {...}},
    {"type": "gmail_message_not_sent_to", "passed": true, "excluded": false, "params": {...}}
  ],
  "_end_state": { ... }
}
```

`reward` here is `partial_credit` — the fraction of non-excluded assertions that passed.
`task_completed_correctly` is 1.0 only when `partial_credit == 1.0` (`rubric/__init__.py:150`).

### Open the visualizer

```bash
# visualizer/README.md — Quick Start §2
python3 visualizer/serve.py
# → http://localhost:8000 (redirects to /compare.html)
```

`serve.py:99` redirects `/` to `compare.html` automatically. Load two JSON files from
`visualizer/runs/local/` to compare `api` vs `zapier` toolset on the same domain. The
comparison view (`compare.html`) ranks runs by **Average Score** (pass rate is secondary),
with a cost vs score scatter plot. The `index.html` single-run view shows the score
distribution histogram, token usage by task, and per-assertion pass counts.

---

## Stage 2 — The extension surface

The harness has three natural extension points: tasks, apps, and assertion types.

### 2a — Add a task

Each domain's `tasks.py` exports constructor functions that return a plain `dict` with keys
`example_id`, `task`, `prompt`, `answer`, and `info`. The `info` dict carries
`zapier_tools`, `initial_state`, and `assertions`.

Real example from `automationbench/domains/simple/tasks.py` (task 3001, the sanity baseline):

```python
# automationbench/domains/simple/tasks.py
def get_simple_email_sf_contact_phone_update() -> dict:
    return {
        "example_id": 3001,
        "task": "simple.email_sf_contact_phone_update",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Jordan Lee just emailed us with a new phone number. "
                    "Can you find that email and update her phone number in Salesforce?"
                ),
            },
        ],
        "answer": "",          # always empty — grading is assertion-based
        "info": {
            "zapier_tools": [  # per-task allowlist for limited_zapier mode
                "gmail_find_email",
                "gmail_get_email_by_id",
                "salesforce_find_records",
                "salesforce_contact_update",
            ],
            "initial_state": {
                "gmail": { "messages": [ { ... } ], "labels": [], "drafts": [] },
                "salesforce": { "contacts": [ { ... } ] },
            },
            "assertions": [
                {"type": "salesforce_contact_phone_equals",
                 "contact_id": "003xx000003JORDAN",
                 "phone": "+15550101"},
            ],
        },
    }
```

For a non-trivial sales task add at minimum: one positive assertion that requires
cross-app traversal, one count-lock (`salesforce_collection_count_equals`) to block
duplicate-create, and one negative guard (`gmail_message_not_sent_to`) to block shotgun
routing. The `initial_state` only needs to populate the apps the task actually touches —
all other app states default to empty via `WorldState`'s `Field(default_factory=...)`.

Seed noise in the `initial_state` by hand for new tasks, or call the domain's
`apply_noise(world_dict, seed=example_id)` from `_noise.py`. The sales noise pool
uses the `099` ID range (`001xx000099NA001`…) so it never collides with task-critical
records (`_noise.py` comment: "Noise IDs use the 099 range … to avoid conflicts").

### 2b — Add an app (Pydantic *State + tools + route)

Adding a new SaaS app involves three files:

1. **`automationbench/schema/<app>.py`** — a Pydantic `*State` with `extra="forbid"`:

```python
# automationbench/schema/hubspot.py (excerpt — HubSpotContact as the record shape)
class HubSpotContact(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    id: str = Field(default_factory=generate_hubspot_id)
    email: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    lifecyclestage: str = Field(default="lead", validation_alias="lifecycle_stage")
    lead_score: Optional[int] = None
    # ... typed fields; None = absent, not zero
```

2. **`automationbench/tools/zapier/<app>/`** — one Python module per tool, each function
   taking `world: WorldState` as the first positional argument (injected by
   `tool_wrapper.py`, hidden from the model's JSON schema via `args_to_skip`).

3. **`automationbench/tools/api/routes/<app>.py`** — REST route handlers. Register by
   adding a `route_<app>` function in the routes package; `api_fetch` dispatches via
   `_url_to_internal_path` (`fetch.py`).

4. Register the new state field in `WorldState` (`schema/world.py:70`):

```python
# automationbench/schema/world.py (pattern — all 44 apps follow this)
class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # ... existing apps ...
    hubspot: HubSpotState = Field(default_factory=HubSpotState)
    # add your app here:
    # myapp: MyAppState = Field(default_factory=MyAppState)
```

### 2c — Add an assertion type

New assertion types are one decorated function registered via `AssertionRegistry.register`.
The pattern from `automationbench/rubric/assertions/salesforce.py`:

```python
# automationbench/rubric/assertions/salesforce.py
@AssertionRegistry.register("salesforce_contact_phone_equals")
def salesforce_contact_phone_equals(world: WorldState, assertion: dict) -> bool:
    """Check if a Salesforce contact's phone matches the expected value."""
    contact = world.salesforce.get_by_id("contacts", assertion["contact_id"])
    if contact is None:
        return False
    actual = _normalize_phone(getattr(contact, "phone", "") or "")
    expected = _normalize_phone(assertion["phone"])
    return actual == expected
```

Negative (must-not-occur) assertions additionally carry `@negative_assertion()`:

```python
# automationbench/rubric/registry.py — the decorator
@AssertionRegistry.register("gmail_message_not_sent_to")
@negative_assertion("gmail")
def gmail_message_not_sent_to(world: WorldState, assertion: dict) -> bool:
    # returns True if the forbidden message was NOT sent
    ...
```

`negative_assertion` marks `fn._negative_assertion = True` (`registry.py:103`). The
`partial_credit` scorer in `rubric/__init__.py` uses this flag to enforce: negative
assertions only receive credit when all positive assertions in the task pass. The decorator
takes optional app-name arguments for documentation only — they have no runtime effect.

After adding a new assertion file, import it in `automationbench/rubric/assertions/__init__.py`
so the side-effect registration runs. Verify: `uv run pytest tests/test_assertions.py`.

---

## Stage 3 — AutomationBench as a verifiers / RL environment

### The dense reward

`automationbench/rubric/__init__.py:153` quotes the design intent directly:

```python
def create_rubric():
    """
    ``partial_credit`` is the primary reward (weight 1.0) — denser signal for
    training and iterative development. ``task_completed_correctly`` is the
    strict 0/1 benchmark metric, weighted 0.0 so it doesn't affect ``reward``
    but is still surfaced in the eval output.
    """
    import verifiers as vf
    return vf.Rubric(
        funcs=[partial_credit, task_completed_correctly],
        weights=[1.0, 0.0],
    )
```

`partial_credit` is `passed / total` (0.0–1.0). It is the `reward` field that the
`verifiers` training loop accumulates. `task_completed_correctly` is logged as a metric
but its weight is 0.0 — it does not shape gradients. This design separates "is the model
improving?" (partial credit, dense) from "did the model succeed?" (binary pass rate, honest).

The free-assertion exclusion rule (`rubric/__init__.py:52–120`) is the anti-reward-hacking
mechanism: any assertion that was already satisfied in `initial_state` is excluded from
the denominator unless the agent broke it (in which case it counts as a failure). This
prevents a policy that does nothing from earning partial credit on pre-seeded "free"
assertions.

### The verifiers base class

`runner.py:38` shows the inheritance:

```python
# automationbench/runner.py
class AutomationBenchEnv(vf.StatefulToolEnv):
    def __init__(self, dataset, rubric, tools=None, max_turns=25,
                 toolset="zapier", search_top_k=None, **kwargs):
        ...
```

`vf.StatefulToolEnv` is from the `verifiers` library (Prime Intellect / Will Brown).
`AutomationBenchEnv` overrides `setup_state` (deserializes `WorldState`), the tool
dispatch (`update_tool_args` re-injects `world`), and the grading path. The `evaluate`
call in `eval.py:243` is the same call a training loop would make — the only difference is
that training would set `rollouts_per_example > 1` and feed `reward` to a PPO/GRPO
optimizer rather than printing it.

### Prime Intellect Environments Hub

The bench is registered on Prime Intellect's hosted runner:

```bash
# README.md — Prime Intellect Environments Hub
prime env install zapier/AutomationBench
prime eval run zapier/AutomationBench

# Smoke test with 5 examples
prime eval run zapier/AutomationBench --num-examples 5

# Run a single domain
prime eval run zapier/AutomationBench --env-args '{"domains": "sales"}'
```

This means you can use AutomationBench as an RL training target without hosting the
environment yourself — connect a policy model via the Prime Intellect client and let
`partial_credit` flow back as reward.

### Why dense reward matters for multi-step tasks

A task with 8 assertions and a 25-turn budget has a sparse 0/1 signal at the episode
boundary. The dense reward tells the optimizer *which sub-goals the policy already reaches*
(e.g., it found the right account and updated the stage, but failed to send the
escalation email). Without partial credit, every incomplete episode is equally wrong.
With it, the policy gradient can distinguish "got 5/8 right" from "got 0/8 right" — the
same insight that motivates process reward models vs outcome reward models in math RL.

---

## Stage 4 — Designing your own agent benchmark

AutomationBench's design choices are a template. Applied to a new domain, the checklist is:

**1. Type the world.** Every mutable object the agent can touch is a Pydantic model with
`extra="forbid"`. No schemaless dicts. This makes assertions trivially correct and state
inspection free. [[automationbench-harness]] calls this the key that "buys determinism."

**2. End-state assertions, not output parsing.** Grade what the world looks like *after*
the episode, not what the agent said. `answer` is always `""` in every AutomationBench
task. No LLM-judge where a programmatic check exists ([[automationbench-tasks-grading]]).

**3. Negative guards are not optional.** A benchmark with only positive assertions is
reward-hackable by shotgun behavior (create everything, send to everyone). Every task that
involves routing or selective action needs at least one `*_not_sent_to` or
`*_not_exists` guard, plus a count-lock assertion
(`salesforce_collection_count_equals`) to block duplicate-create strategies.

**4. Seed noise by `example_id`.** Deterministic noise injection (`_noise.py:apply_noise`)
means the same `example_id` always produces the same world — run variance <1%.
This makes pass^k cheap: 10 repeats of the same task cost 10× the API budget, not 10× the
engineering effort of building 10 different tasks.

**5. A sanity domain is a harness-validity control.** AutomationBench ships `simple` (200
tasks, frontier models score ~97%). A low main-domain score is real agent failure, not a
broken harness. [[insights]] calls this "a sanity domain is a harness-validity control."
Every benchmark should ship an equivalent.

**6. Cost as a first-class metric.** `_extract_usage_and_debug` accumulates tokens per
turn; `pricing.py` resolves model → price. Cost per task is in every exported JSON. The
visualizer's default scatter plot is **cost vs score**. A model that scores 65% at $0.05/task
may be better than one that scores 70% at $0.50/task, depending on the deployment.

**7. Two metrics from one rubric.** `partial_credit` for training signal and developer
iteration; `task_completed_correctly` (binary) for honest reporting. Never conflate them.

---

## Stage 5 — Transfer to Lina TMR

### The structural mismatch

From [[benchmark-comparison]]: AutomationBench is a back-office automator (no user, many
apps, buried policy). Lina TMR is a conversational sales agent (multi-turn prospect, one
domain, explicit goals). The evaluation engineering to borrow is not the toolset — it is
the assertion framework.

Lina's eval problem is exactly the parked sim-to-real / end-model-eval question from
llm-training ch-29: a τ-style user simulator generates the conversation side, but the
simulator itself can drift from real prospects, and the grader must not add another
LLM-judge layer on top of it. AutomationBench's programmatic end-state rubric closes this
gap: the conversation ends, the world is inspected, assertions fire.

### The recipe

Combine three components:

```
τ-style user simulator   ←→   Lina agent   →   typed CRM world
                                                      ↓
                                          AssertionRegistry.check(world, assertions)
                                                      ↓
                                          partial_credit (RL reward)
                                          task_completed_correctly (eval metric)
                                          pass^k (reliability, k=10 cheap due to determinism)
```

**τ-style user simulator**: an LLM prompted as a prospect with a hidden `goal_state` (e.g.,
`{"budget": 50000, "timeline": "Q3", "pain_point": "manual_reporting"}`). The simulator
reveals information only when asked, simulates objections, and ends the call. This provides
the conversation variance that a static trigger cannot.

**Typed CRM world**: a Pydantic `LinaCRMState` with `contacts`, `opportunities`, `call_logs`,
`scheduled_followups`, `disqualified_leads`. Every action Lina can take (create a note,
schedule a follow-up, update deal stage, add to sequence) mutates this world in-process.

**Assertion rubric**: end-state assertions grade the world after the call ends. Example task:

```
Task: Qualify a cold prospect (VP of Engineering, 50-person SaaS startup).
Goal: Identify MEDDIC fields, advance to demo stage if qualified, log a follow-up.

Assertions (must-pass):
  {"type": "crm_opportunity_stage_equals", "opp_id": "opp_lina_001",
   "stage": "Demo Scheduled"}
  {"type": "crm_call_log_exists", "contact_id": "contact_001",
   "contains": ["budget", "timeline"]}
  {"type": "crm_followup_scheduled", "contact_id": "contact_001"}

Assertions (must-not-occur / negative guards):
  {"type": "crm_lead_not_disqualified_without_reason", "contact_id": "contact_001"}
  {"type": "crm_no_duplicate_opportunity", "contact_id": "contact_001"}

Noise:
  Three other contacts in the CRM with similar company profiles (seeded by example_id).
  One with a compliance hold note: "Do not contact — pending legal review."
```

**Sanity domain**: 20 tasks where the prospect immediately reveals budget and timeline
in the first turn (zero elicitation needed). A working Lina agent should score >90%.
Low sanity scores = broken harness, not broken agent.

**Metrics**:
- `partial_credit` (dense, 0–1) as RL reward for GRPO/PPO training.
- `task_completed_correctly` (binary) as the eval headline.
- `pass^k` at k=10: run each `example_id` 10 times (cheap — deterministic world, only the
  simulator introduces variance). A 70% pass rate with pass^10=0.02 is disqualifying for
  production; pass^10=0.45 is acceptable. [[benchmark-comparison]] §Two metric philosophies.
- Cost per call (tokens × price), surfaced in the same JSON export format as AutomationBench.

---

## Deliverable memo

Ship **one** of the two options below. Include it in your repo as `ch-10-memo.md`.

### Option A — New AutomationBench task + rubric

A fully runnable task constructor. Required sections:

1. **Task name and domain.** `sales.my_new_task` or `hr.my_new_task`.
2. **Trigger prompt.** Natural language, no answer in the prompt.
3. **Initial world.** The `initial_state` dict — only populated apps, realistic seed data,
   at least one noise contact/record from the domain's noise pool.
4. **Assertion set.** At minimum: 2 positive assertions (one cross-app), 1 count-lock,
   1 negative guard. Justify each.
5. **Run evidence.** Terminal output or JSON snippet showing the task ran without assertion
   errors under `uv run pytest tests/test_assertions.py`.

### Option B — Lina TMR benchmark spec

A 1–2 page spec. Required sections:

1. **Typed world.** The `LinaCRMState` schema: which collections, which fields, which
   `extra="forbid"` Pydantic models. Name every app state the agent can touch.
2. **Assertion set.** At least 4 positive + 2 negative guards. For each, give the dict shape
   and justify why it cannot be gamed.
3. **Seeded noise.** What noise pool looks like — at minimum a compliance-hold decoy and a
   near-match entity trap (similar name, different contact_id).
4. **Sanity domain.** 3 example tasks with pass criteria a working agent should trivially hit.
5. **Metrics.** `partial_credit`, `task_completed_correctly`, pass^k at k=10, cost/call.
   Explain why pass^k is the right headline for a sales-call scenario.

### Acceptance criteria (both options)

Hard gates:

1. Every `"type"` in your assertion set maps to a registered handler (or you've added the
   registration). Running `AssertionRegistry.check(world, assertion)` does not raise
   `ValueError: Unknown assertion type`.
2. At least one negative guard is present and would fire if the agent acted on all contacts
   (shotgun behavior).
3. The count-lock or equivalent prevents duplicate-create strategies from earning partial
   credit.
4. The sanity domain / sanity tasks are easy enough that a correct agent scores ≥90%.
5. Noise is seeded by `example_id` — two runs of the same task see the same world.
6. The memo names one specific failure mode the rubric cannot catch (i.e., an honest
   acknowledgment of the benchmark's blind spot).

---

## Connections

- **ch-02** — `AutomationBenchEnv(vf.StatefulToolEnv)` is the concrete instantiation of
  the `verifiers` base class first encountered there.
- **ch-03** — the typed in-process world (`WorldState`, 44 Pydantic app states) is the
  design pattern from ch-03's environment architecture discussion.
- **ch-05 / ch-06** — task anatomy (trigger + initial state + assertion rubric) and
  hardening (decoys, negative guards, free-assertion exclusion) are ch-05/06 concepts
  instantiated here.
- **ch-09** — [[benchmark-comparison]] structural contrast (AB vs τ-bench) motivates Stage 5.
- **llm-training ch-29** — the Lina TMR sim-to-real / end-model-eval problem parked there
  is closed in Stage 5 of this lab.

## Further reading

- [[automationbench-harness]] — episode lifecycle, world injection, BM25 tool discovery,
  cost metric.
- [[automationbench-tasks-grading]] — task anatomy, noise mechanisms, scoring logic,
  free-assertion exclusion.
- [[benchmark-comparison]] — AB vs τ-bench structural comparison; pass-rate + cost vs
  pass^k metric philosophies.
- [[insights]] — cross-source insight index; "sanity domain is a harness-validity control."
- `verifiers` library (Prime Intellect / Will Brown) — `StatefulToolEnv`, `Rubric`,
  `evaluate`; the training-loop interface `AutomationBenchEnv` plugs into.
