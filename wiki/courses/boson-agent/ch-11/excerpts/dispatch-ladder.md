# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of ActionExecutor and GatewayCore dispatch — the single dispatch site
# CALLED BY: ch-11/read.md (wikilink)
# PHASE: read
# CHAPTER: ch-11

---
chapter: ch-11
course: boson-agent
phase: read
sub_page: true
title: "The Dispatch Ladder — ActionExecutor and GatewayCore"
sources:
  - boson-agent/packages/gateway/gateway/router/executor.py
  - boson-agent/packages/gateway/gateway/core.py
  - boson-agent/packages/gateway/gateway/layers/pipeline.py
  - boson-agent/packages/gateway/gateway/rules/engine.py
---

# The Dispatch Ladder — ActionExecutor and GatewayCore

This sub-page traces exactly what happens to an `Action` from the moment `RuleEngine`
returns it to the moment the Gateway either sends a response, fires a side effect, or
calls the agent loop. There are two dispatch sites: `ActionExecutor.execute()` (used
when there are no layers) and `LayerPipeline.process()` (used when layers are
configured). Both converge on the same six handler functions.

---

## RuleEngine Output

Before the dispatch, `RuleEngine.evaluate()` produces the action list:

```python
# boson-agent/packages/gateway/gateway/rules/engine.py, lines 51-75

collected: list[Action] = []

# --- sequential phase ---
for check_fn in self._sequential:
    actions = await self._run_check(check_fn, messages, user_message, session)
    non_continue = [a for a in actions if a.type not in ("continue", "pass")]
    if non_continue:
        collected.extend(non_continue)
        # Short-circuit: stop sequential pipeline on first non-CONTINUE
        break

# --- parallel phase ---
if self._parallel:
    results = await asyncio.gather(
        *[
            self._run_check(fn, messages, user_message, session)
            for fn in self._parallel
        ],
        return_exceptions=False,
    )
    for actions in results:
        non_continue = [a for a in actions if a.type not in ("continue", "pass")]
        collected.extend(non_continue)

return collected if collected else [Continue()]
```

Key observations:

1. `Continue` and `Pass` are filtered out — they never reach the executor. The executor
   only sees actions that actually do something.
2. Sequential checks short-circuit on the first non-continue result. If a guard rule
   returns `Filter`, no subsequent sequential rule runs.
3. Parallel checks all run concurrently and all results are collected. A parallel
   `StageTransition` and a parallel `Compact` both appear in the output.
4. If nothing fires, `[Continue()]` is returned as the default, ensuring the executor
   always has at least one item to process.

---

## ActionExecutor: The Handler Table

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 36-45

self._handlers: dict = {
    "continue": self._handle_continue,
    "respond": self._handle_respond,
    "inject": self._handle_inject,
    "compact": self._handle_compact,
    "pre_tool": self._handle_pre_tool,
    "stage_transition": self._handle_stage_transition,
}
```

The executor's dispatch mechanism is a plain dict keyed by `action.type`. This is the
entire "match/if ladder" — there is no `match` statement, no `isinstance` chain, no
visitor pattern. The dict was populated in `__init__`, so handler lookup is O(1) at
dispatch time.

**Notice:** `"filter"` and `"pass"` are absent from `self._handlers`. `filter` is
handled only by `LayerPipeline` (which logs to the `SignalQueue` and returns without
yielding). `pass` is filtered out by `RuleEngine` before the action list reaches the
executor. By the time `execute()` is called, neither type should appear; if one does,
the `if handler is None` guard logs a warning and skips it.

---

## ActionExecutor.execute(): The Main Loop

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 47-67

async def execute(
    self, actions: list[Action], session: SessionState
) -> ExecutionResult:
    """Execute actions in order. RESPOND stops the pipeline."""
    result = ExecutionResult()

    for action in actions:
        handler = self._handlers.get(action.type)
        if handler is None:
            logger.warning("Unknown action type: %s", action.type)
            continue

        await handler(action, session, result)

        if not result.should_continue:
            break

    return result
```

The loop is sequential. Each handler receives the mutable `ExecutionResult` object and
can set `result.should_continue = False` to break the loop. Only `_handle_respond` does
this. All other handlers leave `should_continue` as `True`. The implication: if a
`Respond` and a `StageTransition` are both in the action list, and `Respond` appears
first, the `StageTransition` is never executed. Action ordering matters. The
`RuleEngine` returns actions in collection order (sequential first, parallel second);
the executor processes them in that order with no re-sorting.

---

## ExecutionResult: The Carrier Object

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 23-30

@dataclass
class ExecutionResult:
    """Result of executing a list of actions."""

    should_continue: bool = True
    response: str | None = None
    pending_transition: str | None = None
```

`ExecutionResult` has exactly three fields:

- `should_continue`: controls whether `GatewayCore` calls the agent loop. Default `True`.
- `response`: the fixed string to yield when `should_continue` is `False`.
- `pending_transition`: the target stage name, set by `_handle_stage_transition`.

There is no field for injections or pre-tool results because those are applied directly
to `session.messages` inside the handler — they are committed immediately, not deferred
via the result object. Only side effects that need `GatewayCore`'s involvement
(responding to the client, transitioning the stage machine) travel via `ExecutionResult`.

---

## GatewayCore.handle_message(): The Full Dispatch Sequence

```python
# boson-agent/packages/gateway/gateway/core.py, lines 149-168

# 5. Run rule engine
if self._rule_engine is not None:
    actions = await self._rule_engine.evaluate(
        session.messages, content, session
    )
else:
    actions = [Continue()]

# 6. Execute actions
result = await self._action_executor.execute(actions, session)

# v0.4: Execute pending stage transition from rules
if result.pending_transition:
    skill_fillers = await self._apply_stage_transition(session, result.pending_transition)
    for sf in skill_fillers:
        yield f"\n{sf}\n"

# 7. Fixed response — skip agent
if not result.should_continue:
    yield result.response or ""
    return
```

Reading this in sequence:

**Step 5 (line 149)**: The rule engine is optional. If no rules are configured, the
default is `[Continue()]` — the agent always runs. This is the correct fail-open
default for a gateway without rules.

**Step 6 (line 157)**: `execute()` runs all handlers. By the time this line returns,
injections are already in `session.messages`, compaction may be running in the
background, and `result` holds whatever flags were set.

**Step 7a (line 160)**: Stage transition is applied *after* the executor loop and
*before* the agent check. This ordering ensures the stage prompt is in the history
before the LLM is called, even if a `Compact` action also fired in the same turn.

**Step 7b (line 165)**: `should_continue` check. If `False` (only from `Respond`), the
fixed response is yielded and the method returns. The `_build_agent_runtime()` call and
`run_agent_loop()` at lines 175+ are never reached.

If `should_continue` is `True` (all cases except `Respond`), execution falls through to
the agent loop — no explicit `else` branch needed.

---

## LayerPipeline: The Multi-Layer Variant

When the Gateway is configured with named layers (`LayerPipeline`), the dispatch site
shifts. Each layer's `RuleEngine` is evaluated in turn, and the pipeline dispatches
immediately rather than accumulating into `ExecutionResult`:

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 89-137

for layer_name, engine in self._layers:
    ctx.layer_name = layer_name
    actions = await engine.evaluate(session.messages, content, ctx)
    decision = self._resolve_actions(actions)

    if decision == "filter":
        # ...log, return (drop message)
        return

    if decision == "respond":
        for a in actions:
            if self._action_type(a) == "respond":
                yield text
                return  # Short-circuit

    # Process all non-flow-control actions
    for a in actions:
        atype = self._action_type(a)
        if atype == "inject":
            proposed_injections.append(inject_content)  # staged
        elif atype == "stage_transition":
            fillers = await self._on_stage_transition(session, target)
        elif atype == "compact":
            await self._on_compact(session)
        elif atype == "pre_tool":
            await self._on_pre_tool(session, tool_name, tool_args)
```

Key differences from `ActionExecutor`:

1. **Staged injections**: injections are held in `proposed_injections` and committed only
   after all layers pass. `ActionExecutor` commits injections immediately per action.
2. **Priority resolution**: `_resolve_actions()` picks the highest-priority action type
   across all actions from a layer (e.g. `filter` beats `inject`). `ActionExecutor`
   processes actions sequentially in list order.
3. **No `ExecutionResult`**: the pipeline dispatches via callbacks (`_on_stage_transition`,
   `_on_compact`, `_on_pre_tool`) rather than via the `ExecutionResult` carrier. `filter`
   and `respond` short-circuit the generator directly with `return`.
4. **`filter` is handled**: unlike `ActionExecutor`, the pipeline dispatches `filter`
   explicitly (logging to `SignalQueue`). `ActionExecutor` never sees filter because it
   is only used without the layer pipeline.

---

## Full Dispatch Flow (Sequence Summary)

```
RuleEngine.evaluate()
    sequential checks (priority-ordered, short-circuit on first non-continue)
    parallel checks (asyncio.gather, all results collected)
    → list[Action]  (Continue/Pass filtered out; [Continue()] if nothing fired)

ActionExecutor.execute(actions, session)
    for each action:
        handler = handlers[action.type]
        await handler(action, session, result)
        if not result.should_continue: break
    → ExecutionResult(should_continue, response, pending_transition)

GatewayCore.handle_message()
    if result.pending_transition:
        _apply_stage_transition()     # mutates session.active_stage, injects stage prompt
    if not result.should_continue:
        yield result.response         # done — no LLM
        return
    # else: fall through to run_agent_loop()
```

Every action type flows through this same funnel. The rule author returns an `Action`
dataclass; the executor resolves it to a handler by dict lookup; the handler either
mutates `session.messages` in-place (inject, pre_tool), fires a background task
(compact), sets a deferred flag (stage_transition → pending_transition), or gates the
agent call (respond → should_continue=False). One dispatch site, six outcomes.
