<!-- chapter: ch-08
     track: internals
     kind: content
     title: Metrics, Cost, and Reproducibility
     deps: [ch-07]
     sources: [[automationbench-harness]], [[benchmark-comparison]], [[taubench]]
-->

# Chapter 08 — Metrics, Cost, and Reproducibility

> **Core insight.** AutomationBench reports two coordinated axes — binary pass-rate and
> cost-per-task — and buys near-zero run variance by simulating the entire world in one
> deterministic in-process Pydantic object. That combination lets you answer "can this model
> do the job, and what will it cost?" without running each task dozens of times.

> **Guideline.** Use pass-rate as the capability gate and cost-per-task as the deployment
> filter. The two axes together distinguish a cheap unreliable model from an expensive
> reliable one; either axis alone leaves the purchase decision half-made.

---

## 1. The official metric: binary pass-rate

The public score you see on the leaderboard is a **binary pass-rate**: a task either passes
or it does not. Partial progress earns nothing in the headline number.

This is enforced at the rubric layer. `rubric/__init__.py` defines two functions and weights
them explicitly:

```python
# automationbench/rubric/__init__.py  L143–L166

def task_completed_correctly(state: Any, **kwargs) -> float:
    """Binary pass/fail metric: 1.0 iff every scored assertion passed, else 0.0.

    This is the official benchmark pass-rate signal. It reads the cached
    `partial_credit` value stored by that function, so it avoids re-running
    every assertion.
    """
    return float(state.get("partial_credit", 0.0) == 1.0)


def create_rubric():
    """Create the rubric for AutomationBench task evaluation.

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

Two rubric functions run; only one drives the `reward` field. `partial_credit` (weight 1.0)
gives a fractional score — "what fraction of assertions passed?" — and feeds back into
training pipelines and RL reward signals (the denser gradient is more useful during learning
than a sparse 0/1). `task_completed_correctly` (weight 0.0) is the strict gate: it returns
1.0 only when `partial_credit` is already exactly 1.0. Weight 0.0 means this function does
not affect `reward` at all; it is surface-only, reported in output but never rewarded.

The `pass_rate` the CLI prints and `export_results` stores is computed from
`task_completed_correctly`, not from `reward`:

```python
# automationbench/scripts/eval.py  L288–L290

binary_scores = [float(ro.get("metrics", {}).get("task_completed_correctly", 0.0)) for ro in raw_outputs]
pass_rate = sum(binary_scores) / len(binary_scores) if binary_scores else None
print_avg_reward(avg_reward, pass_rate)
```

And in `export.py` (L168–L172):

```python
# automationbench/export.py  L167–L171

"pass_rate": (
    sum(1 for t in task_results if t["passed"]) / len(task_results)
    if task_results else 0.0
),
```

where `t["passed"]` is `reward == 1.0` (export.py L57). The public dataset has 606 tasks
across six domains (sales 106, the other five 100 each), plus a held-out private set of 600+
tasks on the official leaderboard. SOTA pass-rate as of 2026-06 sits at 12–17%
([[automationbench-overview]]).

### Partial credit is not discarded — it is re-routed

The `partial_credit` value lives in `state["partial_credit"]` and appears in the export's
per-task `score` field (`export.py L55`). Its purpose is twofold: it provides denser signal
for any RL fine-tuning pipeline that uses the harness directly as a reward environment, and
it enables post-hoc analysis ("the model completed 3 of 4 assertions; which one did it
miss?"). The binary gate and the fractional score coexist — they answer different questions.

---

## 2. Cost-per-task as a first-class second axis

### Token accumulation across turns

Each task is multi-turn: the model may issue up to 25 tool calls (the default `max_turns`)
before the episode ends. Token counts must be summed across all turns, not sampled from the
last one.

`_extract_usage_and_debug` in `runner.py` is called inside `env_response` at the end of
every turn, immediately after the model responds and before the tool results are returned:

```python
# automationbench/runner.py  L185–L213

def _extract_usage_and_debug(self, state: vf.State) -> None:
    """Extract token usage and debug info from the latest trajectory step.

    Called at env_response time to process the most recent model response.
    Reads from state["trajectory"][-1]["response"] (a vf.Response object).
    """
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return

    step = trajectory[-1]
    response = step.get("response")
    if response is None:
        return

    # Extract usage from vf.Response
    usage = getattr(response, "usage", None)
    if usage is not None:
        if "_usage" not in state:
            state["_usage"] = {"input_tokens": 0, "output_tokens": 0}
        state["_usage"]["input_tokens"] += getattr(usage, "prompt_tokens", 0)
        state["_usage"]["output_tokens"] += getattr(usage, "completion_tokens", 0)
```

The result is an accumulator dict `state["_usage"]` that grows turn-by-turn. A companion in
`usage.py` (`extract_usage_from_state`, L43–L74) handles reasoning tokens from xAI Grok and
OpenAI o-series models — those are billed as output tokens but reported in a separate
`completion_tokens_details.reasoning_tokens` field. The accumulator pattern ensures that a
25-turn episode does not undercount even when early turns generate the most tokens.

`calculate_run_usage` (`usage.py` L77) aggregates across all tasks in a run:

```python
# automationbench/usage.py  L102–L126

for task_name, state in rollout_items:
    # Prefer the _usage field we accumulated in add_model_response (via state_columns).
    # Fall back to the framework's token_usage if available, then to trajectory scanning.
    custom_usage = state.get("_usage") or {}
    if custom_usage.get("input_tokens", 0) or custom_usage.get("output_tokens", 0):
        input_tokens = int(custom_usage.get("input_tokens", 0) or 0)
        output_tokens = int(custom_usage.get("output_tokens", 0) or 0)
    else:
        token_usage = state.get("token_usage") or {}
        input_tokens = int(token_usage.get("input_tokens", 0) or 0)
        output_tokens = int(token_usage.get("output_tokens", 0) or 0)

    total_input += input_tokens
    total_output += output_tokens

    # Calculate costs if pricing available
    if pricing is not None:
        input_cost = input_tokens * pricing.input_cost_per_token
        output_cost = output_tokens * pricing.output_cost_per_token
        total_cost = input_cost + output_cost
```

The formula is `in_tok * in_price + out_tok * out_price`, applied per task and then summed.
`RunUsage.total_cost` (`usage.py` L140–L142) uses `pricing.calculate_cost` for the run total:

```python
# automationbench/pricing.py  L199–L201

def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
    """Calculate total cost for given token counts."""
    return input_tokens * self.input_cost_per_token + output_tokens * self.output_cost_per_token
```

### Pricing DB: four-strategy lookup and 24-hour cache

Model names arrive in many formats: `openai/gpt-4o`, `claude-sonnet-4-5-20251001`,
`vertex_ai/gemini-3-flash-preview`. `PricingDatabase.get_pricing` normalizes them before
hitting the live price feed.

```python
# automationbench/pricing.py  L293–L344

def get_pricing(self, model: str) -> ModelPricing | None:
    """
    Get pricing for a model.

    Uses multiple lookup strategies:
    1. Exact match
    2. Normalized query against exact keys (strips provider prefixes, date suffixes)
    3. Alias lookup for known naming variations
    4. Normalized query against normalized keys

    Returns None if model pricing is unknown and no CLI override provided.
    """
    # CLI override takes precedence
    if self._input_cost_override is not None and self._output_cost_override is not None:
        return ModelPricing(
            input_cost_per_token=self._input_cost_override,
            output_cost_per_token=self._output_cost_override,
            source="cli-override",
        )

    pricing_data = self._get_pricing_data()

    # Strategy 1: Exact match
    if model in pricing_data:
        result = self._make_pricing(pricing_data[model], pricing_data)
        if result:
            return result

    # Strategy 2: Normalized query against exact keys
    normalized = normalize_model_name(model)
    if normalized in pricing_data:
        result = self._make_pricing(pricing_data[normalized], pricing_data)
        if result:
            return result

    # Strategy 3: Check aliases for known naming variations
    aliased = MODEL_ALIASES.get(normalized)
    if aliased and aliased in pricing_data:
        result = self._make_pricing(pricing_data[aliased], pricing_data)
        if result:
            return result

    # Strategy 4: Normalized query against normalized keys
    for key, entry in pricing_data.items():
        if normalize_model_name(key) == normalized:
            result = self._make_pricing(entry, pricing_data)
            if result:
                return result

    return None
```

Normalization (`pricing.py` L50–L88) strips provider prefixes (`openrouter/anthropic/`,
`vertex_ai/`, `gemini-dev/`, etc.), region prefixes (`us-east-1/`), date suffixes
(`-20251101`, `@20251001`), version tags (`-v1:0`), and `-preview` suffixes. Examples from
the docstring: `"openai/gpt-4o"→"gpt-4o"`, `"claude-opus-4-5-20251101"→"claude-opus-4-5"`,
`"vertex_ai/gemini-3-flash-preview"→"gemini-3-flash"`.

The data source is `llm-prices.com` (the constant `LLM_PRICES_URL =
"https://www.llm-prices.com/current-v1.json"` at L14), fetched at most once per 24 hours and
cached at `~/.cache/automationbench/model_prices.json` (`CACHE_TTL_SECONDS = 24 * 60 * 60`,
L18). On cache miss or network failure the code falls back to a hardcoded `FALLBACK_PRICING`
dict (`pricing.py` L93–L172) with entries for every current-generation model family —
itself updated in Mar 2026 per the comment. The `source` field on `ModelPricing` records
which path was used: `"llm-prices"`, `"fallback"`, or `"cli-override"`.

### CLI overrides take unconditional precedence

Two flags let the caller bypass the pricing DB entirely:

```python
# automationbench/scripts/eval.py  L369–L377  (argparse section)

parser.add_argument(
    "--input-cost",
    type=float,
    default=None,
    help="Per-token input cost in USD (overrides pricing lookup)",
)
parser.add_argument(
    "--output-cost",
    type=float,
    default=None,
    help="Per-token output cost in USD (overrides pricing lookup)",
)
```

When both `--input-cost` and `--output-cost` are supplied, `get_pricing` returns immediately
with `source="cli-override"` before any DB lookup (pricing.py L306–L311). This is the right
override for local vLLM models, negotiated enterprise pricing, or inference providers not yet
in `llm-prices.com`.

---

## 3. Reproducibility: deterministic world, seeded noise, <1% variance

τ-bench (115 retail tasks) requires 10+ repeats because a simulated LLM user injects
stochastic variance every run — turn-taking, information volunteered, termination signal
([[taubench]]). AutomationBench achieves a fundamentally different guarantee.

The entire "world" is **one in-process Pydantic object** (`WorldState`, `schema/world.py`).
No HTTP server, no database process, no subprocess — every tool call mutates `world.<app>
.<collection>` directly in memory. Randomness (distractor records, noise phone numbers)
is seeded in the task definition at authoring time, not at runtime. The same task dict
produces the same world on every run.

`setup_state` (`runner.py` L136`) deep-copies the initial world dict for free-assertion
detection (`state["initial_state"] = copy.deepcopy(initial_state_dict)`), but this copy
contains the same deterministic seed data. There is no sampling from a stochastic user
simulator. Result: run-to-run variance is typically **<1%** ([[automationbench-harness]],
[[automationbench-overview]]).

The practical consequence: a single pass over the 606 public tasks gives you a statistically
stable estimate. You do not need to repeat the suite 10 times to trust the number. At 606
tasks, a 1-percentage-point movement is real signal, not noise.

---

## 4. The `simple` domain as a harness-validity control

Low pass-rates could mean two things: the tasks are hard, or the harness is broken (wrong
tool schemas, bad assertions, broken world mutations). AutomationBench separates these with a
**200-task `simple` domain**.

A sample task from `domains/simple/tasks.py` (L23–80) illustrates the structure:

```python
# automationbench/domains/simple/tasks.py  L23–80  (excerpt)

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
        "info": {
            "zapier_tools": [
                "gmail_find_email",
                "gmail_get_email_by_id",
                "salesforce_find_records",
                "salesforce_contact_update",
            ],
            "initial_state": { ... }
        }
    }
```

Simple tasks require the same tool infrastructure as main-benchmark tasks — the same
`AutomationBenchEnv`, the same `WorldState`, the same rubric — but they use only one or two
tools and have no policy constraints, no distractors, and no noise injection. Any correctly
working model should complete them trivially.

The observed result: even small models achieve **~97% pass-rate on `simple`**
([[automationbench-overview]]). That number serves as a dual guarantee. First, if your run
shows 97% on `simple` and 8% on the main domains, the harness is working — low scores are
genuine orchestration difficulty, not a bug. Second, if your run shows 50% on `simple`, stop
reading the main score: something is wrong with your API key, tool dispatch, or environment
setup.

The `simple` domain is not included in `DEFAULT_DOMAINS` (`domains/__init__.py` L34):

```python
# automationbench/domains/__init__.py  L33–L34

PUBLIC_DOMAINS = ["sales", "marketing", "operations", "support", "finance", "hr"]
DEFAULT_DOMAINS = list(PUBLIC_DOMAINS)
```

You must pass `--domains simple` explicitly. It is separate by design — its easy tasks would
inflate the headline pass-rate if mixed into the default run.

---

## 5. Reading a run: export JSON, per-assertion results, end state

`export_results` (`export.py` L28) is the single egress point from a completed run to the
visualizer. Every task produces a structured record:

```python
# automationbench/export.py  L128–L152

task_result = {
    "id": i + 1,
    "name": task_name,
    "score": float(reward),          # partial_credit (0.0–1.0)
    "passed": reward == 1.0,         # binary gate
    "assertions_total": assertions_total,
    "assertions_passed": assertions_passed,
    "assertion_results": assertion_results,  # per-assertion pass/fail + excluded flag
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "cost": task_cost,
    "steps": steps,
    "messages": messages,            # full chat completion
    "end_state": output.get("_end_state"),
}
```

The `assertion_results` list is the key diagnostic field. Each entry carries `type`,
`passed`, `excluded`, and `params`. When `passed=False` on an assertion that was not already
satisfied before the agent acted (`excluded=False`), you have a concrete failure: a named
assertion type, the exact parameters it expected, and the end-state world that did not
satisfy it. There is no ambiguity about what went wrong.

`_end_state` is a full `model_dump()` of the `WorldState` at episode termination
(`rubric/__init__.py` L126). Comparing it to `initial_state` in the same export record lets
you replay the episode mentally: what did the agent change, what did it leave untouched, what
did it corrupt?

The top-level `summary` block in the export JSON (`export.py` L154–L196) collects:
`avg_score` (mean partial credit), `pass_rate` (binary), `passed_count`, `failed_count`,
`total_input_tokens`, `total_output_tokens`, `total_cost`, `cost_formatted`, plus debug
counters for empty responses and zero-output tasks that indicate API or streaming issues.

### What "false confidence in incorrect tool calls" looks like

The dominant failure mode — **72–91% of failures** ([[automationbench-overview]]) — is that
the model calls a tool with confidently-stated but wrong arguments, the tool returns an error
or silently wrong result, and the model declares success anyway. The export surfaces this via
two paths:

1. `assertion_results` shows the specific assertion that failed (e.g.,
   `contact_phone_equals` for task `simple.email_sf_contact_phone_update`). The phone
   in the end-state doesn't match the expected value even though the agent called
   `salesforce_contact_update`.

2. `finish_reasons` and `empty_responses` in `_debug` (`runner.py` L215–L231) catch cases
   where the model stops generating mid-task or emits an empty completion — a signal that
   it misread a tool result and interpreted it as completion.

Looking at these two fields together distinguishes three failure classes: wrong args (correct
tool, wrong data), wrong tool (right intent, wrong tool name), and premature termination
(correct so far, but stopped before all steps).

---

## 6. pass-rate + cost vs τ-bench pass^k: two deployment questions

AutomationBench and τ-bench report different headline metrics because they are answering
different questions about deployment readiness ([[benchmark-comparison]]).

### pass^k defined

τ-bench's pass^k measures **reliability over k independent trials** of the same task,
averaged over the task set [[taubench]]:

```
pass^k = (1/|T|) Σ_i p̂_i^k
```

with unbiased estimator `ρ(n, c, k) = 1 − C(n−c, k) / C(n, k)` where `n` is the number of
trials run and `c` is the number of passes observed. A task where the model succeeds 90% of
the time contributes only `0.9^8 ≈ 0.43` to pass^8. Reliability collapses fast. The
original τ-bench numbers: GPT-4o at ~61% pass^1 retail drops to below 25% pass^8 — the same
model, the same tasks, just measured at k=8 instead of k=1.

τ-bench **needs** pass^k because the stochastic user simulator introduces substantial
run-to-run variance. The same task can succeed or fail depending on which phrasing the
simulated user happens to choose. Running k=10 or more repeats is the only way to get a
stable estimate of how the agent actually behaves in repeated deployments.

### Why AutomationBench uses pass-rate + cost instead

AutomationBench's determinism makes the incentive calculus different. With <1% variance, a
single run over 606 tasks gives a stable estimate — you already know how the model performs
on each task. Running k repeats would cost k× as much compute and produce nearly identical
numbers. pass^k is statistically redundant given determinism.

The second axis — **cost-per-task** — fills the gap. Given two models at similar pass-rates,
cost-per-task answers which one is deployable. A model that costs $0.08/task at 15% pass-rate
sits in a different category from one that costs $0.40/task at 17% pass-rate. This axis has
no τ-bench analogue because τ-bench does not export token usage or cost as a first-class
metric.

### What AutomationBench *could* do

The harness could adopt pass^k cheaply: its determinism means repeating a task 10 times is
equivalent to running one task with k identical seeds. But it would add almost no information
— the variance the metric is designed to catch is not present. The design choice is
intentional: report capability-at-a-price (the back-office deployment question), not
reliability-over-trials (the customer-service deployment question)
([[benchmark-comparison]]).

The two questions map to deployment contexts:

| Question | Metric | When it matters |
|----------|--------|-----------------|
| "Can this model do the job, and what does it cost?" | pass-rate + cost/task | Back-office automation; each task is unique; throughput and economics drive the decision |
| "Will it succeed every time the same customer asks?" | pass^k | Customer-service agents; same task recurs; a 20% failure rate is disqualifying |

Choosing the wrong metric for the wrong deployment context gives false confidence in either
direction: a model optimized for pass^k may be expensive and slow; a model optimized for
pass-rate + cost may be unreliable at any individual task when that task repeats.

---

## Summary

The measurement system of AutomationBench is internally consistent: the binary pass-rate is
the right headline for a benchmark where each task is a unique business event; cost-per-task
is the right second axis for evaluating deployment economics; determinism (<1% variance)
makes both metrics stable without repetition; the `simple` domain validates the harness
before you trust any main-domain score; and per-assertion export gives you the specific
failure, not just a 0. The contrast with τ-bench's pass^k is not a gap — it is a choice
about which deployment question each benchmark is designed to answer.

**Next:** [[ch-09]] — the structural head-to-head with τ-bench and τ²-bench in full.
