---
calling_spec:
  purpose: "Full walkthrough of stage preloads: tool synthetic history + skill injection"
  parent: "[[../read.md]]"
  phase: read
  chapter: ch-12
  course: boson-agent
---

# Stage Preloads — Full Walkthrough

> Sub-page of [[../read.md]]. Covers `GatewayCore._run_stage_preloads()` and the
> mechanism that synthesizes tool call history on stage entry.

---

## Why Preloads Exist

When a stage transition fires mid-conversation, the LLM enters a new mode with
different tools. But the LLM has no data yet — it would need to make a tool call
on its very first turn in the new stage just to bootstrap its context. Preloads
eliminate that wasted turn by running tool calls *before the LLM's turn starts*
and injecting the results directly into message history. The LLM arrives at the
new stage already informed.

There are two preload types:

| Type | Mechanism | History representation |
|------|-----------|----------------------|
| Tool preload | `execute_tool()` called synchronously | `[assistant] ToolUseBlock + [user] ToolResultBlock` pair |
| Skill preload | `skill.prompt_template` retrieved | `[user] <system-reminder>…</system-reminder>` message |

---

## Core Implementation

```python
# boson-agent/packages/gateway/gateway/core.py, lines 332-369

async def _run_stage_preloads(self, session, target):
    """Execute tool + skill preloads. Returns skill fillers to yield.

    Preload tools imitate the full tool calling format:
      [assistant] tool_use block
      [user]      tool_result block
    So the LLM sees a natural tool call exchange.
    """
    if self._tool_registry:
        from uuid import uuid4
        from basement.schemas.message_schema import ToolUseBlock
        for name, args in self._stage_preloads.get(target, []):
            try:
                tool_use_id = f"toolu_{uuid4().hex[:12]}"
                # Assistant message: tool call
                session.messages.append(Message(
                    role="assistant",
                    content=[ToolUseBlock(id=tool_use_id, name=name, input=args)],
                ))
                # Execute and get result
                res = await execute_tool(self._tool_registry, name, args)
                res.tool_use_id = tool_use_id
                # User message: tool result
                session.messages.append(Message(role="user", content=[res]))
            except Exception as e:
                logger.warning("Tool preload %s: %s", name, e)
    fillers = []
    if self._skill_registry:
        for name in self._skill_preloads.get(target, []):
            try:
                skill = self._skill_registry.get(name)
                session.messages.append(Message(
                    role="user",
                    content=f"<system-reminder>{skill.prompt_template}</system-reminder>"))
                raw = self._skill_fillers.get(
                    name, self._skill_fillers.get("_default", ""))
                if raw:
                    fillers.append(random.choice(raw) if isinstance(raw, list) else raw)
            except Exception as e:
                logger.warning("Skill preload %s: %s", name, e)
    return fillers
```

**Line-by-line for tool preloads (lines 340-358):**

- **Line 343** `tool_use_id = f"toolu_{uuid4().hex[:12]}"` — generates a
  fresh ID in the same format the Anthropic API uses for real tool calls
  (`toolu_` prefix + hex suffix). This is important: the API validates that
  every `tool_result` block references a `tool_use` block with a matching ID.
  Synthetic preload calls must be indistinguishable from real ones.

- **Lines 345–348** Appends an `[assistant]` message containing a single
  `ToolUseBlock`. This mimics what the LLM would produce if it decided to call
  the tool itself. From the next LLM turn's perspective, "I called this tool."

- **Lines 349–352** Calls `execute_tool()` synchronously (awaited), gets the
  `ToolResultBlock`, patches its `tool_use_id`, and appends a `[user]` message
  containing the result. From the LLM's perspective, "The tool responded with X."

- **Lines 353–355** Exception handler is intentionally non-fatal: a preload
  failure logs a warning and skips that entry. The stage transition still
  completes; the LLM just won't have that preloaded datum.

**Line-by-line for skill preloads (lines 358–368):**

- **Lines 362–364** Retrieves the skill's `prompt_template` and injects it as a
  `[user]` message with `<system-reminder>` wrapping. This is the same mechanism
  used by the `use_skill` meta-tool at runtime — making skill injection on stage
  entry identical in form to skill injection on demand.

- **Lines 365–366** Looks up a "filler" string for the skill — a display message
  shown to the end-user UI while the agent is internally processing ("Let me
  check the schedule options for you…"). Fillers are cosmetic; they are wrapped
  in `[FILLER]…[/FILLER]` tags by the caller and stripped by the UI layer.

---

## The Pipeline Deferral Problem

Preloads cannot always run immediately when a `StageTransition` action fires.
If the transition happens *inside the layer pipeline* (while the pipeline is
assembling the final user message), the user message has not yet been appended
to `session.messages`. Running preloads at that moment would produce history
in the wrong order: preload pairs would land before the user message that
triggered the transition.

The solution is a two-phase approach:

```python
# boson-agent/packages/gateway/gateway/core.py, lines 326-330

async def _apply_stage_transition(self, session, target):
    # ...validate, set active_stage, inject prompt...

    # Defer preloads when in pipeline (run after user message is appended)
    if getattr(session, "_in_pipeline", False):
        session._pending_preload_stage = target
        return []

    return await self._run_stage_preloads(session, target)
```

And in the pipeline, after the user message is committed:

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 170-175

pending_preload = getattr(session, "_pending_preload_stage", None)
if pending_preload and self._on_run_preloads:
    preload_fillers = await self._on_run_preloads(session, pending_preload)
    skill_fillers.extend(preload_fillers)
    session._pending_preload_stage = None
```

**Notice:** `_pending_preload_stage` is a transient sentinel attribute — set
on the session object during the pipeline run and cleared immediately after.
It is never persisted or read after the turn completes. This is the same
pattern used for `_pending_stage_injection`, `_in_pipeline`, and
`_pipeline_appended` — a set of ephemeral per-turn flags that coordinate
timing between `LayerPipeline` and `GatewayCore`.

---

## What the LLM History Looks Like After Preloads

After a transition to `product_focused` with `preloads: [("check_product_summary", {})]`:

```
[user]       "Customer: I'd like to hear more about your product"
             + <system-reminder>[Stage: product_focused] Available tools: …</system-reminder>
[assistant]  ToolUseBlock(name="check_product_summary", input={}, id="toolu_abc123")
[user]       ToolResultBlock(tool_use_id="toolu_abc123", content="Product: Premium Cancer Coverage…")
[assistant]  "Great, let me walk you through the key benefits…"  ← first real LLM turn
```

The LLM enters its first turn in `product_focused` already holding the product
summary. It never needs to ask for it. The two preload messages are synthetic
but structurally identical to a real tool call exchange — the LLM treats them
as prior conversation history.
