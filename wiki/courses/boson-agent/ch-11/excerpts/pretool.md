# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of PreTool — the pre-agent tool execution action
# CALLED BY: ch-11/read.md (wikilink)
# PHASE: read
# CHAPTER: ch-11

---
chapter: ch-11
course: boson-agent
phase: read
sub_page: true
title: "PreTool — seeding context before the LLM turn"
sources:
  - boson-agent/packages/gateway/gateway/schemas/actions.py
  - boson-agent/packages/gateway/gateway/router/executor.py
  - boson-agent/packages/gateway/gateway/layers/pipeline.py
---

# PreTool — Seeding Context Before the LLM Turn

`PreTool` is the action that lets a rule author say: "before the LLM gets this message,
run this tool and put the result into history." The LLM then sees a completed tool
exchange in its context at the start of the turn, without having to request it.

---

## Definition

```python
# boson-agent/packages/gateway/gateway/schemas/actions.py, lines 66-70

def PreTool(tool_name: str, arguments: dict | None = None) -> Action:
    """Pre-execute a tool, add result to history, then continue."""
    return Action(
        type="pre_tool",
        payload={"tool_name": tool_name, "arguments": arguments or {}},
    )
```

`PreTool` takes a tool name (must be registered in the `ToolRegistry`) and an optional
arguments dict. The `arguments or {}` guard normalises `None` to `{}` so the executor
always has a dict to pass to `execute_tool`. There is no return-value slot in the action
— the result is written directly to `session.messages` by the executor.

---

## Executor Handler: The Four-Step Sequence

```python
# boson-agent/packages/gateway/gateway/router/executor.py, lines 107-121

async def _handle_pre_tool(
    self, action: Action, session: SessionState, result: ExecutionResult
) -> None:
    """PRE_TOOL — execute a tool and append the result to history."""
    if self._tool_registry is None:
        logger.warning("pre_tool action received but no tool_registry configured")
        return

    tool_name = action.payload.get("tool_name", "")
    arguments = action.payload.get("arguments", {})

    tool_result = await execute_tool(self._tool_registry, tool_name, arguments)
    session.messages.append(
        Message(role="user", content=[tool_result])
    )
```

Walk through this line by line:

1. **Guard**: if no `tool_registry` is configured (possible in unit tests or minimal
   setups), the handler logs a warning and returns cleanly. `result.should_continue` is
   never set to `False`, so the turn proceeds to the LLM without the pre-tool result.
   Fail-open design.

2. **Extract payload**: `tool_name` and `arguments` are pulled from the action payload.
   These are plain strings/dicts — no type coercion needed.

3. **Execute**: `execute_tool(registry, tool_name, arguments)` is the same function the
   agent loop uses to execute tools during the think-act-observe cycle. `PreTool` reuses
   the exact same execution path — there is no special pre-tool executor.

4. **Append**: the `ToolResult` is wrapped in a list and appended as a `user`-role
   message with list content. Using `content=[tool_result]` (list form) rather than
   `content=str(tool_result)` (string form) preserves the structured `ToolResult` type
   so the LLM API can render it correctly as a tool result block in the conversation.

**Notice:** The history entry created here is only a tool *result* message (`role="user"`
with `content=[ToolResult]`). There is no corresponding `role="assistant"` tool-use
block before it. Compare this to the stage preload path in `GatewayCore._run_stage_preloads`
(lines 343-356 of `core.py`), which inserts *both* an assistant tool-use block and a
user tool-result block to create a natural exchange. The `ActionExecutor._handle_pre_tool`
omits the assistant turn — the result appears in history as if it arrived from a
background data source, not from a model-requested tool call.

---

## LayerPipeline: Pre-Tool in Multi-Layer Context

```python
# boson-agent/packages/gateway/gateway/layers/pipeline.py, lines 133-137

elif atype == "pre_tool" and self._on_pre_tool:
    tool_name = a.payload.get("tool_name", "")
    tool_args = a.payload.get("arguments", {})
    logger.info("Layer %s: PRE_TOOL %s", layer_name, tool_name)
    await self._on_pre_tool(session, tool_name, tool_args)
```

In the layered pipeline, `pre_tool` is dispatched via a callback (`self._on_pre_tool`)
rather than directly to the executor handler. This callback is wired at construction
time and points back to the same `execute_tool` path. The layer-pipeline variant runs
the tool *before* the user message is appended to history (the user message commit
happens at lines 162-164, after all layers complete). This means the pre-tool result
arrives in history before the current turn's user message — the LLM sees: `[pre-tool
result] → [user message]`, which is the intended order for context seeding.

---

## The Latency Argument

The docstring says "reduce latency or seed context". Here is what this means concretely:

**Latency reduction case**: Suppose the agent almost always calls `get_user_profile` on
the first message in a session. Without `PreTool`, the sequence is:
```
user message → LLM decides to call get_user_profile → tool executes → LLM generates
```
That is two LLM roundtrips for the first response. With `PreTool`, a rule fires
`PreTool("get_user_profile", {"user_id": session.user_id})` before the turn, and the
sequence becomes:
```
[rule fires] → tool executes (no LLM) → user message → LLM generates (profile already in context)
```
One LLM roundtrip. The tool result is in history at the start of the turn.

**Context seeding case**: Some tools produce deterministic results (current time, user
tier, feature flags). There is no value in the LLM "deciding" to call them — the rule
always knows the result will be needed. `PreTool` makes the tool call deterministic and
immediate.

---

## Comparison: PreTool vs Stage Preloads

Both `PreTool` (rule action) and `_run_stage_preloads` (stage transition callback) call
`execute_tool` before the LLM. The key differences:

| Aspect | PreTool action | Stage preloads |
|---|---|---|
| Triggered by | Rule returning `PreTool(...)` | Stage transition committing |
| Timing | Before user message append (pipeline) or after (executor) | After user message append |
| History format | tool_result only (no assistant tool-use block) | Full exchange: assistant tool-use + user tool-result |
| Configuration | Inline in rule code | `stage_preloads` dict in `GatewayCore` setup |
| Operator control | Rule author decides | Operator config decides |

Stage preloads are the declarative, operator-configured path; `PreTool` is the
programmatic, rule-author path. Both ultimately call the same `execute_tool` function.
