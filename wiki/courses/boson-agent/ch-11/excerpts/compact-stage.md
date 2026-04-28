# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of Compact and StageTransition — the orchestration-level actions
# CALLED BY: ch-11/read.md (wikilink)
# PHASE: read
# CHAPTER: ch-11

---
chapter: ch-11
course: boson-agent
phase: read
sub_page: true
title: "Compact and StageTransition — orchestration-level actions"
sources:
  - boson-agent/packages/gateway/gateway/schemas/actions.py
  - boson-agent/packages/gateway/gateway/router/executor.py
  - boson-agent/packages/gateway/gateway/core.py
  - boson-agent/packages/gateway/gateway/compact/pipeline.py
  - boson-agent/agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py
  - boson-agent/agents/test-lina-gateway/layers/03-orchestrator/rules/transition_detector.py
---

# Compact and StageTransition — Orchestration-Level Actions

`Compact` and `StageTransition` are the two actions that reach outside the
request-response cycle and mutate durable subsystems: the `AsyncCompactPipeline` (which
manages history summarisation) and the `StageMachine` (which governs conversation flow).
Neither stops the turn — both fire a side effect and proceed to the LLM.

---

## Compact: Async Background Summarisation

### Definition

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 61-63

def Compact() -> Action:
    """Trigger async LLM compact in background."""
    return Action(type="compact")
```

`Compact()` takes no arguments. There is no `payload`. All compaction configuration
(threshold, model, keep-recent count, system prompt) is fixed in `CompactConfig` at
`GatewayCore.setup()` time. A rule returning `Compact()` says "compact now" — it cannot
say "compact with these settings".

### Executor Handler

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 97-105

async def _handle_compact(
    self, action: Action, session: SessionState, result: ExecutionResult
) -> None:
    """COMPACT — trigger async compaction if pipeline is available."""
    if self._compact_pipeline is not None:
        try:
            self._compact_pipeline.trigger(session)
        except Exception as exc:
            logger.warning("compact_pipeline.trigger raised: %s", exc)
```

`_handle_compact` calls `self._compact_pipeline.trigger(session)` — note the absence of
`await`. `trigger` is itself an async method, but it is **not awaited here**. Let's look
at what `trigger` does:

```python
# boson-agent/packages/gateway/gateway/compact/pipeline.py, lines 52-65

async def trigger(self, session: SessionState) -> bool:
    """Start a background compact task if conditions are met."""
    if session.compact_in_progress:
        return False
    if not self.should_compact(session):
        return False

    session.compact_in_progress = True
    asyncio.create_task(self._compact_task(session))
    return True
```

`trigger` is a coroutine that *schedules* `_compact_task` via `asyncio.create_task` and
returns immediately. By not awaiting `trigger` in `_handle_compact`, the executor is
relying on the fact that `trigger` only does two guard checks and one task creation
before returning — it never suspends on I/O. This is correct but subtle: if `trigger`
were ever changed to do real async work before the `create_task`, the executor would
miss it by not awaiting. The current implementation is safe; the pattern is slightly
fragile if `trigger` evolves.

**Notice:** `session.compact_in_progress = True` is set synchronously inside `trigger`,
before the background task starts. This prevents a second `Compact()` action (from a
later rule or a later turn) from spawning a duplicate task while the first is running.
The flag is reset to `False` in the `finally` block of `_compact_task`. This is the
only concurrency guard — there are no locks, because the Gateway is single-threaded
asyncio per session.

### Background Task and Deferred Apply

```python
# boson-agent/packages/gateway/gateway/compact/pipeline.py, lines 67-86

async def _compact_task(self, session: SessionState) -> None:
    """Background task: summarise old messages and queue the result."""
    try:
        keep = self._config.keep_recent
        messages_to_compact = list(session.messages[:-keep]) if keep else list(session.messages)

        summary = await self.strategy.summarize(
            messages_to_compact,
            session.system_prompt,
        )

        session.pending_compact = {
            "summary": summary,
            "keep_recent": keep,
        }
    except Exception as exc:
        logger.error("compact_task failed: %s", exc)
        session.pending_compact = None
    finally:
        session.compact_in_progress = False
```

The task summarises all messages except the most recent `keep_recent`, then stores the
result in `session.pending_compact`. The current turn's LLM call is not affected. On the
**next** turn, `GatewayCore.handle_message` calls `apply_pending()` at the very start
(line 119-121 of `core.py`):

```python
# boson-agent/packages/gateway/gateway/core.py, lines 119-121

# 3. Apply any pending compact from previous turn
if self._compact_pipeline is not None:
    self._compact_pipeline.apply_pending(session, shared_history)
```

`apply_pending` calls `shared_history.swap_compact(summary, keep_recent)`, which
replaces the old messages with the summary. The compaction is applied exactly once, at
the start of the turn after it completes.

### Example: Turn-Count Trigger

```python
# boson-agent/agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py, lines 42-49

@check("compact_trigger", mode="sequential", priority=20)
def trigger_compact(messages, user_message, session):
    """Trigger compact when message count is high."""
    if len(messages) > 30:
        return Compact()
    return Continue()
```

The simplest possible policy: count messages, trigger compaction above a threshold. In
production, you might also check `session.active_stage` (avoid compacting mid-purchase)
or check `session.compact_in_progress` explicitly. The rule here delegates both guards
to `AsyncCompactPipeline.trigger()`, which performs them itself.

---

## StageTransition: Request a State Machine Advance

### Definition

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 73-76

def StageTransition(target_stage: str) -> Action:
    """Transition to a new stage. Gateway enforces via StageMachine."""
    return Action(type="stage_transition", payload={"target_stage": target_stage})
```

`StageTransition` carries only the target stage name. The rule author names a stage
defined in `stage_config.py` — the action does not carry the full stage definition, only
the transition request.

### Executor Handler: Deferred via pending_transition

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 123-132

async def _handle_stage_transition(
    self, action: Action, session: SessionState, result: ExecutionResult
) -> None:
    """STAGE_TRANSITION — store target for core to execute after actions."""
    target = action.payload.get("target_stage", "")
    if not target:
        logger.warning("stage_transition: missing target_stage")
        return
    result.pending_transition = target
```

The executor does **not** call `StageMachine.transition()` directly. It stores the
target in `result.pending_transition`. This is a two-phase commit pattern: collect
intent during action execution, then apply it after all actions are processed. The
reason is ordering: if `Compact()` and `StageTransition("closing")` are both returned
from the same rule set, the executor processes all actions in sequence, and then
`GatewayCore` applies the transition cleanly after the action loop finishes.

### GatewayCore Applies the Transition

```python
# boson-agent/packages/gateway/gateway/core.py, lines 159-163

# v0.4: Execute pending stage transition from rules
if result.pending_transition:
    skill_fillers = await self._apply_stage_transition(session, result.pending_transition)
    for sf in skill_fillers:
        yield f"\n{sf}\n"
```

`_apply_stage_transition` is called only when `result.pending_transition` is set. It
calls `self._stage_machine.transition(from_stage=session.active_stage, to_stage=target)`,
which validates the edge in the stage graph. If the transition is not allowed, it
returns `result.success = False` and nothing changes. The stage machine is the authority
— the rule's `StageTransition` is a request, not a command.

### Full Transition Path

After the machine validates the edge:

```python
# boson-agent/packages/gateway/gateway/core.py, lines 310-330

async def _apply_stage_transition(self, session: SessionState, target: str) -> list[str]:
    if not self._stage_machine or not session.active_stage:
        return []
    result = self._stage_machine.transition(from_stage=session.active_stage, to_stage=target)
    if not result.success:
        return []
    session.active_stage = target
    self._inject_stage(session, result.new_stage)

    # Defer preloads when in pipeline (run after user message is appended)
    if getattr(session, "_in_pipeline", False):
        session._pending_preload_stage = target
        return []

    return await self._run_stage_preloads(session, target)
```

Steps:
1. `StageMachine.transition()` validates the edge.
2. `session.active_stage = target` commits the new stage.
3. `_inject_stage()` appends the new stage's system prompt into the current user message
   (or as a `_pending_stage_injection` if in-pipeline).
4. `_run_stage_preloads()` fires configured tool and skill preloads for the new stage.

The LLM therefore sees, at the start of its call: the new stage prompt injected into the
current user message, plus any preloaded tool results. It knows it has transitioned
before generating its first token.

### Example: Intent-Based Transition (Demo)

```python
# boson-agent/agents/demo-gateway/layers/03-orchestrator/rules/stage_manager.py, lines 22-39

@check("auto_stage_transition", mode="sequential", priority=10)
def auto_transition(messages, user_message, session):
    """Transition stages based on ctx.data intent from Layer 02."""
    intent = getattr(session, "data", {}).get("intent")
    active = getattr(session, "active_stage", None)

    if intent == "closing" and active != "closing":
        return StageTransition("closing")

    # Auto-transition from welcome to main on first real message
    if active == "welcome" and session.turn_count > 1:
        return StageTransition("main")

    return Pass()
```

This rule reads `session.data.intent`, which was written by a previous layer (Layer 02),
demonstrating inter-layer data passing via `ctx.data`. It returns `StageTransition`
when the intent signals a closing or when the welcome stage should advance. Note the
`active != "closing"` guard — without it, every turn in the closing stage would re-fire
the transition and re-inject the closing stage prompt.

### Example: LLM-Driven Transition (Lina)

```python
# boson-agent/agents/test-lina-gateway/layers/03-orchestrator/rules/transition_detector.py, lines 376-380

    if target:
        logger.info("Deterministic transition: %s -> %s", stage, target)
        session.checklist_state = {}
        return StageTransition(target)
```

In the Lina production agent, `StageTransition` is returned after either deterministic
keyword matching or a lightweight LLM call. `session.checklist_state = {}` clears the
checklist cache so the new stage starts fresh. The rule returns exactly one
`StageTransition` — the executor stores it in `pending_transition` and the state machine
validates it on the same turn.

---

## Compact vs StageTransition: Side-Effect Topology

| Property | Compact | StageTransition |
|---|---|---|
| Immediate effect on turn | None (fires background task) | Stage prompt injected into user message |
| Durable state changed | `session.pending_compact` | `session.active_stage` |
| Validated by subsystem? | No (threshold check only) | Yes (StageMachine validates edge) |
| Applied when? | Next turn's `apply_pending()` | Same turn (before LLM call) |
| Can be rejected? | No (if threshold met) | Yes (invalid transition → no-op) |
| LLM sees effect? | Next turn only | Same turn (new stage prompt in context) |

The asymmetry is deliberate: compaction is a background optimisation that should not
affect the current turn's quality, while a stage transition must take effect immediately
so the LLM's instructions match the conversation state.
