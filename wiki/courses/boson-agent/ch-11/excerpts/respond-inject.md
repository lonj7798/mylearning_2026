# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of Respond and Inject — the LLM-skipping / LLM-augmenting actions
# CALLED BY: ch-11/read.md (wikilink)
# PHASE: read
# CHAPTER: ch-11

---
chapter: ch-11
course: boson-agent
phase: read
sub_page: true
title: "Respond and Inject — LLM-skipping and LLM-augmenting actions"
sources:
  - boson-agent/packages/gateway/gateway/schemas/actions.py
  - boson-agent/packages/gateway/gateway/router/executor.py
  - boson-agent/packages/gateway/gateway/core.py
  - boson-agent/agents/demo-gateway/layers/01-guard/rules/spam_filter.py
  - boson-agent/agents/test-lina-gateway/layers/03-orchestrator/rules/transition_detector.py
---

# Respond and Inject — LLM-Skipping and LLM-Augmenting Actions

`Respond` and `Inject` are the two actions that directly control whether the LLM
participates in a turn. `Respond` eliminates the LLM call. `Inject` enriches the
context the LLM receives. Both are triggered from rule checks and dispatched by
`ActionExecutor` before `run_agent_loop` is ever called.

---

## Respond: Fixed Reply, No Agent

### Definition

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 49-51

def Respond(text: str) -> Action:
    """Send fixed response to client, skip agent."""
    return Action(type="respond", payload={"text": text})
```

`Respond(text)` packages one string into the payload. Nothing else. The rule author
decides the entire client-visible response at authoring time (or at check-evaluation
time if building `text` dynamically).

### Executor Handler

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 78-83

async def _handle_respond(
    self, action: Action, session: SessionState, result: ExecutionResult
) -> None:
    """RESPOND — return fixed text to client, stop pipeline."""
    result.response = action.payload.get("text", "")
    result.should_continue = False
```

Two assignments. `result.response` captures the text; `result.should_continue = False`
is the gate that stops the executor loop (`if not result.should_continue: break`) and
prevents any subsequent actions from running. The `ExecutionResult` object travels back
to `GatewayCore.handle_message`.

### GatewayCore Short-Circuit

```python
# boson-agent/packages/gateway/gateway/core.py, lines 166-168

# 7. Fixed response — skip agent
if not result.should_continue:
    yield result.response or ""
    return
```

Lines 166-168 are the only place `result.should_continue` is tested in `core.py`.
If `False`, the method yields the fixed string and returns — the `_build_agent_runtime`
call and `run_agent_loop` async loop that follow are never reached. From a latency
perspective, this is the cheapest possible turn: no LLM tokens, no tool execution, one
string yield.

**Notice:** The `yield result.response or ""` guard (`or ""`) exists because `response`
is typed `str | None` and can be `None` if the executor logic ever reaches this branch
without setting a response (defensive coding). In practice, `_handle_respond` always
sets a string, but the `or ""` ensures the generator never yields `None`, which would
break the WebSocket framing layer.

### Example: Greeting Responder

```python
# boson-agent/agents/demo-gateway/layers/01-guard/rules/spam_filter.py, lines 22-28

@check("greeting_responder", mode="sequential", priority=20)
def greeting_responder(messages, user_message, session):
    """Auto-respond to simple greetings. Demonstrates Respond action."""
    lower = user_message.strip().lower()
    if lower in GREETING_WORDS and len(messages) == 0:
        return Respond(text="Hello! I'm a demo assistant. Ask me to calculate, check weather, search docs, or get time!")
    return Pass()
```

This rule fires only on the very first message (`len(messages) == 0`) and only when the
content is a bare greeting. The `Respond` short-circuits the agent; the LLM never sees
the message. On any subsequent turn, or any non-greeting message, `Pass()` is returned
and the pipeline continues. This is the canonical pattern for FAQ-style short-circuits.

---

## Inject: Add Context, Keep the LLM

### Definition

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 54-56

def Inject(content: str) -> Action:
    """Add content to conversation history, then continue."""
    return Action(type="inject", payload={"content": content})
```

`Inject` is structurally identical to `Respond` at the dataclass level — a type string
and a single payload key. The difference is entirely in the executor handler and in
what happens to `result.should_continue`.

### Executor Handler

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 85-94

async def _handle_inject(
    self, action: Action, session: SessionState, result: ExecutionResult
) -> None:
    """INJECT — prepend system reminder to conversation history."""
    content = action.payload.get("content", "")
    session.messages.append(
        Message(
            role="user",
            content=f"<system-reminder>{content}</system-reminder>",
        )
    )
```

`_handle_inject` appends a new `Message` with `role="user"` to `session.messages`. The
content is wrapped in `<system-reminder>` tags. `result.should_continue` is never
touched, so the executor loop continues and `GatewayCore` proceeds to the agent.

**Notice:** The injected message is appended as `role="user"`, not `role="system"`.
Most LLM APIs treat `system` as a preamble, not a conversational turn. Using `user`
role with `<system-reminder>` tags is a prompt-engineering convention that places the
injection in the flow of the conversation so the model sees it as a "just-in-time
reminder" at the correct turn position, without overwriting the static system prompt.
Compare [[ch-05]] where `inject_system_reminder` uses the same `<system-reminder>`
wrapper pattern — this is the same mechanism surfaced at the gateway rule layer.

### LayerPipeline Staged-Inject Variant

When running through `LayerPipeline` (the multi-layer case), inject is handled
differently — it is *staged* rather than immediately committed:

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 119-124

for a in actions:
    atype = self._action_type(a)
    if atype == "inject":
        inject_content = a.payload.get("content", "")
        proposed_injections.append(inject_content)
        logger.info("Layer %s: INJECT (staged) '%s'", layer_name, inject_content[:50])
```

And later, only committed after all layers pass:

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 149-151

# Staged injections from layers
for inj in proposed_injections:
    parts.append(f"<system-reminder>{inj}</system-reminder>")
```

The injection is held in `proposed_injections` until every layer has been evaluated. If
any later layer returns `Filter` or `Respond`, the staged injections are discarded.
This staged-commit pattern prevents partial mutations: you never get a history with an
injected context reminder but no actual agent response because a guard layer fired.

### Example: Escalation Inject in Lina

```python
# boson-agent/agents/test-lina-gateway/layers/03-orchestrator/rules/transition_detector.py, lines 367-373

if target == "escalate_to_human":
    if not hasattr(session, "escalate_count"):
        session.escalate_count = 0
    session.escalate_count += 1
    if session.escalate_count < 2:
        logger.info("Escalation request %d (need 2 to transfer)", session.escalate_count)
        return Inject(
            content="[Customer requested human agent (%d/2). "
            "Acknowledge the request, but try to help them first. "
            "If they insist, transfer will happen.]" % session.escalate_count
        )
```

This is a real production use of `Inject`. On the first escalation request, rather than
transitioning to `escalate_to_human` immediately, the rule injects a bracketed note into
the conversation history. The LLM reads this note in the next turn and responds
appropriately ("I understand you'd like to speak with a human — let me first see if I
can help…"). Only on the second request does the rule return `StageTransition`.

**Notice:** The injected `content` uses Python `%` formatting rather than f-strings.
This is incidental style, but it shows that `Inject` can build dynamic strings at check
evaluation time — the content is not a static template. The rule composes the reminder
string from session state (`session.escalate_count`) at runtime.

---

## Respond vs Inject: Decision Table

| Criterion | Respond | Inject |
|---|---|---|
| LLM called? | No | Yes |
| Text in response? | Yes (the `text` arg) | No (LLM generates it) |
| History mutated? | No | Yes (+1 user message) |
| `should_continue` | False | True (unchanged) |
| Latency | Minimum (no LLM) | Normal (LLM runs) |
| Typical use | FAQ, greetings, abuse blocks | Context reminders, policy nudges |

The choice reduces to one question: **do you need the LLM to generate a response, or do
you already know exactly what to say?** If you know, use `Respond`. If you want to steer
the LLM, use `Inject`.
