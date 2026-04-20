# === CALLING SPEC ===
# PURPOSE: Walkthroughs of concrete @check rule implementations
# CALLED BY: read.md (ch-10 index)
# CALLS: nothing
# PURE: yes
# DETERMINISTIC: yes

---
chapter: ch-10
course: boson-agent
phase: read
excerpt_of: example-rule
source_file: >
  boson-agent/agents/demo-gateway/layers/01-guard/rules/spam_filter.py,
  boson-agent/agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py,
  boson-agent/agents/test-lina-gateway/layers/03-orchestrator/rules/transition_detector.py
created_at: "2026-04-19"
---

# Excerpt: Concrete Rule Implementations

> Three rules from two agents — covering the full range from the simplest
> possible deterministic guard to a production LLM-backed parallel check with
> stateful session caching and a two-strike escalation counter.

---

## Example 1: `spam_filter.py` — Simple Deterministic Sequential Guard

```python
# agents/demo-gateway/layers/01-guard/rules/spam_filter.py, lines 1-29

"""Demo Layer 01: Guard — Filter spam/abuse + Respond to greetings."""
from gateway.rules.check import check
from gateway.schemas.actions import Filter, Respond, Pass

SPAM_WORDS = {"spam", "buy now", "click here", "free money"}
GREETING_WORDS = {"hi", "hello", "hey"}


@check("spam_filter", mode="sequential", priority=10)        # (A)
def spam_filter(messages, user_message, session):
    """Filter spam messages."""
    lower = user_message.strip().lower()
    for spam in SPAM_WORDS:
        if spam in lower:
            return Filter(reason=f"spam:{spam}")             # (B)
    return Pass()


@check("greeting_responder", mode="sequential", priority=20) # (C)
def greeting_responder(messages, user_message, session):
    """Auto-respond to simple greetings."""
    lower = user_message.strip().lower()
    if lower in GREETING_WORDS and len(messages) == 0:       # (D)
        return Respond(text="Hello! I'm a demo assistant. ...")
    return Pass()
```

### Annotations

**(A)** `priority=10` — runs before `greeting_responder` (priority=20). If
spam is detected, `greeting_responder` never sees the message. The priority
gap of 10 is a convention that leaves room to insert a check at priority 15.

**(B)** `Filter(reason=...)` — a blocking action. The message is rejected
and no response is sent to the user (or a generic rejection response, depending
on ActionExecutor configuration). Notice that `Filter` is distinct from
`Respond`: `Filter` silently drops the message; `Respond` returns a message.

**(C)** `priority=20` — only reached if `spam_filter` returned `Pass()`.
This demonstrates the sequential short-circuit: priority ordering creates an
implicit dependency chain.

**(D)** `len(messages) == 0` — the check reads `messages` (the full
conversation history) to detect a first-turn greeting. This is the typical
pattern for stateless checks that use conversation context without writing
to `session`. The check is pure: same input always produces same output.

### Notice

Both functions are `def`, not `async def`. They run synchronously inside the
`_run_check` wrapper. Because they do no I/O, there is no event-loop impact.
Using `async def` here would add coroutine overhead with no benefit.

---

## Example 2: `stage_manager.py` — Stateful Sequential Check with Session Writes

```python
# agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py, lines 1-51

@check("turn_counter", mode="sequential", priority=1)        # (A)
def count_turns(messages, user_message, session):
    """Track turn count per session."""
    if not hasattr(session, "turn_count"):
        session.turn_count = 0
    session.turn_count += 1                                  # (B)
    return Pass()


@check("auto_stage_transition", mode="sequential", priority=10)
def auto_transition(messages, user_message, session):
    """Transition stages based on ctx.data intent from Layer 02."""
    intent = getattr(session, "data", {}).get("intent")     # (C)
    active = getattr(session, "active_stage", None)

    if intent == "closing" and active != "closing":
        return StageTransition("closing")

    if active == "welcome" and session.turn_count > 1:       # (D)
        return StageTransition("main")

    return Pass()


@check("compact_trigger", mode="sequential", priority=20)
def trigger_compact(messages, user_message, session):
    """Trigger compact when message count is high."""
    if len(messages) > 30:
        return Compact()                                     # (E)
    return Continue()
```

### Annotations

**(A)** `priority=1` — `turn_counter` runs before everything else in this
layer. It must run first because `auto_transition` (priority=10) reads
`session.turn_count`. This is an example of **inter-check data dependency
expressed through priority ordering**.

**(B)** `session.turn_count += 1` — writing a custom attribute to `session`.
The `SessionState` dataclass does not define `turn_count`; Python allows
arbitrary attribute assignment on dataclass instances. This is the pattern
for stateful checks: initialise with `hasattr` guard, then mutate. The value
persists across turns because `session` is the same object for the entire
conversation lifetime.

**(C)** `getattr(session, "data", {}).get("intent")` — reading `session.data`,
a dict that Layer 02 writes. This demonstrates inter-layer data passing: an
earlier layer (02 — analyst) annotates the session, and a later layer
(03 — orchestrator) reads that annotation. The `getattr` with default `{}`
is defensive: if Layer 02 did not run or did not set `data`, this check
returns `Pass()` gracefully.

**(D)** `session.turn_count > 1` — using the value set by `turn_counter`
(priority=1) in the same turn. This only works because `turn_counter` runs
first. If the priorities were reversed, `turn_count` might not exist yet.

**(E)** `Compact()` — triggers the async compaction pipeline. This is not
a blocking action: the engine adds it to `collected`, and `ActionExecutor`
schedules a background `asyncio.gather` task. The response to the current
turn proceeds normally while compaction runs in the background.

### Notice

`turn_counter` always returns `Pass()`. It is a **side-effect-only check**:
its sole purpose is to mutate `session.turn_count`. This is a legitimate
pattern — not every check needs to return a meaningful action. But because it
is sequential and returns `Pass`, it does not short-circuit; subsequent checks
always run.

---

## Example 3: `transition_detector.py` (Lina) — Production LLM Parallel Check

```python
# agents/test-lina-gateway/layers/03-orchestrator/rules/transition_detector.py,
# lines 345-396

@check("stage_transition", mode="parallel", priority=20, check_type="llm")  # (A)
async def detect_stage_transition(messages, user_message, session):
    """Detect stage transitions: deterministic first, then LLM fallback."""
    stage = getattr(session, "active_stage", None)
    if not stage or stage in ("end", "close"):
        return Continue()                                    # (B)

    lower = user_message.lower().strip()

    # 1. Try deterministic keyword match (fast, free)
    target = _deterministic_check(stage, lower, messages)   # (C)

    # Count escalation requests — only transition after 2+ requests
    if target == "escalate_to_human":
        if not hasattr(session, "escalate_count"):
            session.escalate_count = 0
        session.escalate_count += 1
        if session.escalate_count < 2:
            return Inject(                                   # (D)
                content="[Customer requested human agent (%d/2). "
                "Acknowledge the request, but try to help them first.]"
                % session.escalate_count
            )
        return StageTransition("escalate_to_human")

    if target:
        return StageTransition(target)                      # (E)

    # 2a. Product checklist — all items must be checked
    if stage == "product_focused":
        target = await _llm_checklist(stage, messages, user_message, session)
        if target:
            return StageTransition(target)
        return Continue()

    # 2b. Fall back to LLM evaluation
    target = await _llm_check(stage, messages, user_message) # (F)
    if target:
        return StageTransition(target)

    return Continue()
```

### Annotations

**(A)** `mode="parallel", check_type="llm"` — this check runs concurrently
with other parallel checks in the same layer. Marking it `check_type="llm"`
signals to operators that it issues LLM calls (using a cheap Haiku model via
`TRANSITION_LLM_MODEL` env var). The parallel mode is correct here: stage
transition detection is independent of other parallel checks (e.g., an
`Inject` check that adds user profile context).

**(B)** Early return for terminal stages — if `active_stage` is `"end"` or
`"close"`, no transition is possible. This guard prevents LLM calls after the
conversation has concluded, saving cost and latency.

**(C)** Deterministic-first pattern: `_deterministic_check()` uses keyword
matching (no LLM) to detect common transitions. This is fast and free. The
LLM fallback (steps 2a/2b) is only reached if deterministic matching fails.
This is an explicit latency/cost optimisation baked into the rule logic, not
the framework.

**(D)** `Inject(content=...)` with stateful `session.escalate_count` — the
two-strike escalation pattern. The first escalation request injects a
reminder into the LLM's context ("try to help them first"). Only on the
second request does the rule return `StageTransition("escalate_to_human")`.
This demonstrates that `Inject` can be used as a soft intervention that
changes the LLM's behaviour without routing control away from it.

**(E)** `StageTransition(target)` — instructs `ActionExecutor` to change
`session.active_stage` and inject the new stage's system prompt. This is a
routing action, not a terminal action: the LLM still runs after the
transition, but with the updated stage context.

**(F)** `await _llm_check(stage, messages, user_message)` — the LLM fallback
issues a network call to a cheap model (Haiku by default). The result is a
stage name or `None`. This call has latency (typically 200–800ms) and cost.
The deterministic-first pattern at (C) exists specifically to avoid reaching
this line on common inputs.

### Notice: LLM Inside a Check — Not Inside the Agent

This check calls an LLM directly from rule evaluation, before the main agent
LLM runs. The call is to a **different, cheaper model** (Haiku) configured
via environment variables. This is an important architectural point: the rule
engine can use its own LLM inference independently of the agent's model. The
agent might be using GPT-4o; the transition detector uses Haiku. This
separation lets operators control cost and latency for rule evaluation
separately from response generation.

This also clarifies the @check vs @hook distinction carryforward from ch-05:
hooks are fired by the framework around the agent loop (ON_TURN_START,
PRE_LLM_CALL, etc.). Checks are framework-fired too, but they gate the agent
entirely — a rule that returns `StageTransition` or `Respond` means the main
agent LLM may not run at all for this turn.

See also: [[../read.md]] §Key Concepts §5 for the full synthesis,
[[engine-parallel.md]] for how `asyncio.gather` runs this check concurrently.
