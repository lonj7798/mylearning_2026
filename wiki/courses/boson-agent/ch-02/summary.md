# ch-02 Summary — The Think-Act-Observe Agent Loop

## 1. `run_agent_loop()` — what is it?

`run_agent_loop` handles one full turn of the agent. A "turn" = one user message → multiple LLM calls → final text response.

It's an **async generator** — uses `yield` to stream events out. Gateway consumes those events with `async for`. A single turn may contain many LLM calls because of tool chaining, so the loop itself is bounded by `max_turns`.

At the top it aliases `ctx = runtime.context_manager`, `api = runtime.conversation_api`, `hooks = runtime.hook_registry` — a readability idiom used across LOD orchestrators.

## 2. Turn lifecycle — ordered events

The actual code order in agent_loop.py:

0. **append user message** (skip if `runtime.skip_user_append` is True — Gateway may have already appended)
1. **Fire `ON_TURN_START`** (after append, so hooks can read the new user input)
2. `while turn_count < max_turns:`
    - **Fire `PRE_LLM_CALL`**
    - `async for event in provider.stream(...)`:
        - `TextDelta` → append to `text_parts`, yield to Gateway
        - `ToolUseStart` → append dict to `tool_uses`, yield to Gateway (filler hook)
        - `InputJsonDelta` → concatenate into `tool_uses[-1]["input_json"]`
        - `ToolUseEnd` → pass (accumulation complete)
        - `MessageEnd` → yield
    - Branch:
        - If `tool_uses` is non-empty → build assistant message (text + tool blocks) + `_execute_tool_uses(...)` + `continue`
        - Else → add text assistant message + **Fire `POST_LLM_CALL`** + `break`
3. `else:` (while...else — fires only if max_turns hit without break)
    - `yield TextDelta("[Max turns exceeded — stopping]")`
    - `yield MessageEnd(stop_reason="max_turns")`
4. **Fire `ON_TURN_END`**
5. `await api.flush_pending()`

### Event count per turn

| event | count | condition |
|-------|-------|-----------|
| `ON_TURN_START` | exactly 1 | every turn |
| `PRE_LLM_CALL` | 1..N | N = number of LLM calls in this turn (up to `max_turns`) |
| `POST_LLM_CALL` | 0 or 1 | only when the LLM ends with text (not tool_use). At most once per turn. |
| `PRE_TOOL_CALL` | 0..K | K = total number of tool calls across all iterations |
| `POST_TOOL_CALL` | 0..K | paired with PRE_TOOL_CALL |
| `ON_TURN_END` | exactly 1 | every turn, even on max_turns bail |

The key insight I got wrong the first time: **`POST_LLM_CALL` is a "turn success" marker, not a "per-LLM-call" marker.** A turn that hits max_turns never fires `POST_LLM_CALL` at all.

## 3. StreamEvent — the 5 types

All are pydantic `BaseModel` with minimal fields:

```python
TextDelta(text: str)
ToolUseStart(id: str, name: str)
InputJsonDelta(partial_json: str)
ToolUseEnd(id: str)
MessageEnd(stop_reason: str)
```

Roles:
- **TextDelta** = partial text tokens (streamed live to user)
- **ToolUseStart** = "tool call begins here", carries id + name
- **InputJsonDelta** = partial JSON arguments (provider streams chunks like `'{"cit'` + `'y":"서울"}'`)
- **ToolUseEnd** = "that tool's arguments are complete", carries the matching id
- **MessageEnd** = final event, carries `stop_reason`

`stop_reason` values the loop cares about:
- `"end_turn"` = LLM finished with text only → loop should `break`
- `"tool_use"` = LLM ended with a tool call → loop should execute tools and `continue`
- `"max_turns"` = our custom value we yield when the safety bound trips (not a real provider value)

### How the loop detects "LLM called a tool"

Not a declaration — **by stream shape**. If `ToolUseStart` appeared during the stream and `tool_uses` list is non-empty after the `async for` exits, the loop takes the tool branch. The `stop_reason` confirms: `"tool_use"` means "do tools and come back", `"end_turn"` means "done".

## 4. Provider abstraction — two layers of wrapping

The 5 types above are **boson-agent's common shape**. The raw provider APIs are different:

### Anthropic raw events (from Anthropic SDK)

```json
{"type": "content_block_start", "content_block": {"type": "tool_use", "id": "...", "name": "..."}}
{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
{"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "..."}}
{"type": "content_block_stop"}
{"type": "message_delta", "delta": {"stop_reason": "tool_use"}}
```

Event types are `content_block_start / _delta / _stop` and `message_delta`. Inside `delta` is a typed sub-event. Tool boundaries are **explicit** (`content_block_start` for tool_use, `content_block_stop` closes it).

### OpenAI raw events (from OpenAI SDK)

```json
{"choices": [{"delta": {"content": "네"}}]}
{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "...", "function": {"name": "...", "arguments": "{\"cit"}}]}}]}
{"choices": [{"finish_reason": "tool_calls"}]}
```

Everything is flat inside `delta`. `content` (text) and `tool_calls` (array of tool objects) are **separate fields**, not a typed block list. Tool boundaries are implicit — derived from `index` changes. `finish_reason="tool_calls"` instead of `"tool_use"`.

### boson-agent normalization

`anthropic_provider.py` and `openai_provider.py` convert their raw SDK events into the same 5 StreamEvent types. The internal model follows Anthropic's **block list** style (ordered list of TextBlock + ToolUseBlock), because it preserves text/tool ordering — OpenAI's flat fields can't express ordering.

Agent loop sees only the 5 common types, never the raw SDK events. Adding a new provider means writing a new adapter; agent_loop stays untouched.

## 5. Excerpt 3 — the tool execution branch

When `tool_uses` is non-empty after the stream ends, this block runs (agent_loop.py:172-195):

1. **Strip system-reminder echoes** from `text_parts`. The LLM sometimes echoes the `<system-reminder>...</system-reminder>` tags it saw; a regex removes them before saving the text to history. Without this, the echoed reminder would get baked into the assistant message and confuse the next turn.
2. **JSON-parse each tool's accumulated string.** `tu["input_json"]` is the concatenation of all `InputJsonDelta.partial_json` values for one tool. `json.loads(...)` turns it into a dict. Empty string → `{}` (tools with no arguments).
3. **Build `assistant_blocks`** = `[TextBlock?, ToolUseBlock*]`. Text is optional (LLM may call a tool with no preamble), tool blocks are one per tool call. This matches Anthropic's `content: List[Block]` shape, which the API requires for the assistant message.
4. **Append the assistant message** to `ctx`. This is required by the API contract: every `tool_use` block in history must be paired with a `tool_result` block in the next user message. Without the append, the next LLM call would error.
5. **`_execute_tool_uses(...)`** — iterates each tool: fires `PRE_TOOL_CALL`, runs the handler (via ToolRouter or direct registry), catches exceptions (fires `ON_ERROR`), fires `POST_TOOL_CALL`, appends a `ToolResultBlock` inside a user message.
6. **`continue`** — jump to the top of the `while` loop. Next iteration fires `PRE_LLM_CALL` again and calls the provider with history now containing the tool_use + tool_result pair. LLM sees its own earlier call and the result → decides next action.

`continue` is the heart of tool chaining: it's what lets the LLM call `get_weather("서울")`, see the result, then call `get_weather("도쿄")`, see that, then finally respond in text.

## 6. `max_turns` safety bound

This is a **controlled termination**, not an error. No exception is raised. Python's `while...else` construct is used:

```python
while turn_count < max_turns:
    ...
    if text_only: break  # ← normal exit
else:
    # this else fires only when the while condition becomes false
    # (break skips this block entirely)
    yield TextDelta("[Max turns exceeded — stopping]")
    yield MessageEnd(stop_reason="max_turns")
```

### Normal exit vs max_turns exit

| | normal exit | max_turns exit |
|---|---|---|
| trigger | `break` after text-only response | while condition becomes false |
| yielded before cleanup | last TextDelta + MessageEnd(end_turn) | `[Max turns...]` TextDelta + MessageEnd(max_turns) |
| POST_LLM_CALL fires? | yes | no |
| ON_TURN_END fires? | yes | yes (same cleanup path) |

So both exits run the same post-loop cleanup. The only difference is the message the user sees and whether `POST_LLM_CALL` ran.

### Recovery idea from discussion

Instead of bailing with `[Max turns...]`, we could make one more LLM call with `tools=None` to force a graceful text summary based on tool results already gathered. Trade-offs:

- **Pros**: better UX (natural response), reuses the tool work that already cost tokens.
- **Cons**:
    - extra API call when something is already broken
    - risks masking bugs — max_turns hits are a signal that the agent is stuck. Auto-recovery hides that signal.
    - LLM may still deflect ("I need more info...") — no guarantee of a useful response.

## 7. Why `tools=None` = force text-only

Ties back to the meta-tool vs native discussion in ch-01. If the API request doesn't include a `tools` parameter, the LLM literally doesn't know any tools exist → it must respond with text.

boson-agent's `LLMProvider.stream()` signature is `(messages, system, tools)`. No `tool_choice` parameter. `tools=None` is the only knob for forcing text — and it's sufficient, because the provider adapters pass it straight through to the API.

Correction to my first draft: `tools=None` is **not** the default per turn. Normal turns pass the full tool list. `tools=None` is specifically for cases like max_turns recovery, or a stage where the LLM should only explain and not act.

## 8. Why we track hook events — 5 reasons

1. **Extension** — add features without editing `agent_loop.py`.
2. **Separation of concerns** — cross-cutting code (logging, permissions, metrics) lives in its own files. LOD rule.
3. **Policy enforcement** — `PRE_*` hooks can block before the action happens.
4. **Observability** — `POST_*` hooks can record after the action completes.
5. **Customization** — each agent registers its own hook set; demo-gateway and Lina have very different needs.

One-line: LLM agents are unpredictable, so the core stays clean while hook points open at every important moment for observing, controlling, and extending.

---

## Open questions / what I'm still fuzzy on

- `AgentRuntime` AD3 pattern — never really dug into it in discussion.
- Tool handler raising mid-execution — what exactly the loop does after `_execute_tool_uses` catches the exception. I know `ON_ERROR` fires, but I'm not sure whether the loop continues to the next LLM call or bails.
- Async generator yielding vs event-loop scheduling — I understand the user-facing effect (streaming feels live) but the actual Python event-loop mechanics when `yield` is hit inside an `async for` I still treat as magic.
- `POST_LLM_CALL`'s "text-only" condition: I now understand WHEN it fires, but I'm not 100% sure what context fields are available inside it (usage stats, token counts, the text itself?).

---

## Carryover from ch-01 (still open)

- `GatewayCore.handle_message` full per-turn flow — will return in ch-07.
- `InterruptHandler` Basement SIGINT vs Gateway barge-in — will return in ch-07–09.
- system-reminder immutability invariant — partially surfaced in ch-02 tool execution (regex strip at line 178).
