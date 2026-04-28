# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of the Action dataclass and all constructor functions
# CALLED BY: ch-11/read.md (wikilink)
# PHASE: read
# CHAPTER: ch-11

---
chapter: ch-11
course: boson-agent
phase: read
sub_page: true
title: "Action Types — the unified vocabulary"
source: boson-agent/packages/gateway/gateway/schemas/actions.py
---

# Action Types — the Unified Vocabulary

This sub-page walks through every line of `gateway/schemas/actions.py`, the single
source of truth for every decision the Gateway can communicate from a rule check to
the dispatch site.

---

## The Base Dataclass

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 17-28

ActionType = Literal[
    "continue", "respond", "inject", "compact", "pre_tool",
    "stage_transition", "filter", "pass",
]

@dataclass
class Action:
    """A single action produced by a rule check."""

    type: ActionType
    payload: dict = field(default_factory=dict)
```

`Action` is a two-field dataclass. `type` is a `Literal` string — not an enum, not an
integer, just a plain lowercase string. `payload` is a plain `dict` with a default
empty factory. There is no inheritance, no generic, no protocol. The entire contract
between a rule function and the executor is: produce an object that has `.type` and
`.payload`.

**Notice:** The `type` field uses a `Literal` union rather than an `enum`. This means
any dict-based serialisation (JSON, logging) reads naturally without extra conversion,
and the exhaustiveness is checked by mypy at the call site, not at runtime. The
executor dispatches on `action.type` with a plain `dict` lookup — pattern matching
would have been equally valid, but the dict-of-handlers is more easily extensible at
runtime (you can register new handlers after construction).

---

## Flow-Control Constructors

### Continue and Pass — the No-Op Pair

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 33-41

def Continue() -> Action:
    """Proceed to next rule/layer (default action)."""
    return Action(type="continue")


def Pass() -> Action:
    """Forward message unchanged to next layer (alias for Continue in layers)."""
    return Action(type="pass")
```

Both functions return an `Action` with an empty payload. They exist as two separate
names for the same semantic: "I have nothing to say; proceed". `Continue` is the inner
rule vocabulary; `Pass` reads more naturally in a layer guard that is explicitly
forwarding without change. The RuleEngine filters both out identically:

```python
# boson-agent/packages/gateway/gateway/rules/engine.py, lines 56-57
non_continue = [a for a in actions if a.type not in ("continue", "pass")]
```

**Notice:** Having two names for one semantic is an explicit documentation choice, not
a refactoring oversight. The `v0.6 compatibility stub` comment in `layers/engine.py`
explains that `Pass` was added for readability when the unified action set merged inner
and layer vocabs. A rule author who writes `Pass()` signals "I am a filter layer,
deliberately letting this through", whereas `Continue()` signals "I have no opinion".

---

### Filter — the Silent Drop

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 44-46

def Filter(reason: str = "") -> Action:
    """Drop message. Do not forward to next layer. Queue to SignalQueue."""
    return Action(type="filter", payload={"reason": reason})
```

`Filter` carries a `reason` string which the pipeline writes to the `SignalQueue` for
analytics. The message is discarded — no text is sent back to the client. This is the
only action type that writes to the `SignalQueue`; all others either short-circuit,
mutate history, or proceed.

**Notice:** `Filter` and `Respond` look similar (both stop the pipeline), but their
client-visible behaviour is opposite. `Filter` = silence (no bytes to client). `Respond`
= a fixed string to the client. The difference matters for abuse filtering, where sending
any response (even "blocked") can confirm to a spammer that the system is running.

---

### Respond — Skip the LLM Entirely

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 49-51

def Respond(text: str) -> Action:
    """Send fixed response to client, skip agent."""
    return Action(type="respond", payload={"text": text})
```

One mandatory argument: the text to yield. The executor sets `result.should_continue =
False` and `result.response = text`. `GatewayCore.handle_message` checks
`result.should_continue` before constructing the agent runtime — if `False`, it yields
`result.response` and returns immediately. The LLM is never instantiated.

---

### Inject — Augment Without Skipping

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 54-56

def Inject(content: str) -> Action:
    """Add content to conversation history, then continue."""
    return Action(type="inject", payload={"content": content})
```

`Inject` wraps `content` in a `<system-reminder>` block and appends it as a `user`-role
message before the LLM call. The agent loop then sees this extra context alongside the
original user message. Execution continues normally — this is an augmentation, not a
stop.

---

## Orchestration Constructors

### Compact — Fire-and-Forget Background Task

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 61-63

def Compact() -> Action:
    """Trigger async LLM compact in background."""
    return Action(type="compact")
```

No payload. The executor calls `compact_pipeline.trigger(session)`, which starts an
`asyncio.create_task` and returns immediately. The summarization result is stored in
`session.pending_compact` and applied on the *next* turn's `apply_pending()` call.
Returning `Compact()` from a rule never blocks the current turn.

**Notice:** `Compact()` takes no arguments because all configuration (threshold,
keep-recent count, model, system prompt) lives in `CompactConfig`, which is wired into
`AsyncCompactPipeline` at setup time. The rule author cannot override compaction
parameters per-call; that would make compaction behaviour dependent on rule authorship,
not on the operator's config file.

---

### PreTool — Seed Context Before the LLM

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 66-70

def PreTool(tool_name: str, arguments: dict | None = None) -> Action:
    """Pre-execute a tool, add result to history, then continue."""
    return Action(
        type="pre_tool",
        payload={"tool_name": tool_name, "arguments": arguments or {}},
    )
```

`PreTool` names a tool registered in the `ToolRegistry` and optional arguments. The
executor calls `execute_tool(registry, tool_name, arguments)` synchronously (within the
async handler, but awaited), then appends the `ToolResult` as a `user`-role message.
The LLM therefore sees a "pre-answered" tool call in its history at turn start, without
having to request it.

---

### StageTransition — Signal the State Machine

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 73-76

def StageTransition(target_stage: str) -> Action:
    """Transition to a new stage. Gateway enforces via StageMachine."""
    return Action(type="stage_transition", payload={"target_stage": target_stage})
```

`StageTransition` only carries the target name. It does not directly mutate
`session.active_stage`. Instead, the executor stores the target in
`result.pending_transition`, and `GatewayCore._apply_stage_transition()` calls
`StageMachine.transition()`, which validates the edge before committing. Rules that
return `StageTransition("closing")` are making a *request*, not a *command* — the state
machine can reject invalid transitions.

---

## The Priority Table

The `LayerPipeline` defines a numeric priority for each action type used when resolving
conflicts across a multi-layer result:

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 32-41

ACTION_PRIORITY = {
    "filter": 0,
    "respond": 1,
    "inject": 2,
    "stage_transition": 3,
    "compact": 3,
    "pre_tool": 3,
    "pass": 4,
    "continue": 4,
}
```

Lower number = higher precedence. `filter` beats everything. `respond` beats inject.
`inject`, `stage_transition`, `compact`, and `pre_tool` are all "orchestration" tier
with equal priority. `pass`/`continue` win only when nothing else fires. This table is
the contract that makes multi-layer composition safe: a guard layer's `Filter` cannot
be overridden by an orchestrator's `Inject` on the same turn.
