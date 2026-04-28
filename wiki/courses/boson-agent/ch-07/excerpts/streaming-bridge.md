---
chapter: ch-07
course: boson-agent
phase: read
excerpt_of: gateway/core.py lines 196-256 — streaming bridge
created_at: "2026-04-19"
---

# Excerpt: Streaming Bridge — the async generator yield chain

**Source:** `boson-agent/packages/gateway/gateway/core.py`, lines 196–256
**Context:** Inside `GatewayCore.handle_message`, after `_build_agent_runtime`

---

## The full yield chain

```
WebSocket handler
    async for chunk in message_handler(session_id, content):
        await websocket.send(chunk)
            │
            ▼
GatewayCore.handle_message   [async generator]
    async for event in run_agent_loop(runtime, content):
        if isinstance(event, TextDelta):
            yield event.text          ──► WebSocket
            │
            ▼
run_agent_loop               [async generator]
    async for event in runtime.provider.stream(...):
        if isinstance(event, TextDelta):
            yield event               ──► handle_message
            │
            ▼
provider.stream              [async generator]
    # reads from HTTP/2 or anthropic SDK streaming response
    yield TextDelta(text=chunk)       ──► run_agent_loop
```

There are three async generators chained together. Each level consumes the generator below it with `async for`, transforms or filters the events, and `yield`s selected output to the level above. The WebSocket handler at the top converts text chunks to wire bytes.

---

## The streaming bridge code, line by line

```python
# boson-agent/packages/gateway/gateway/core.py, lines 196-256
from basement.llm.base import TextDelta, ToolUseStart
filler_count = 0
# Buffer initial text to catch system-reminder echoes (always at response start)
# Once cleared, switch to direct streaming for low latency
initial_buf: list[str] = []
streaming = False

async for event in run_agent_loop(runtime, content):
```

**Line 196-197:** Import the two event types at function scope (deferred import to keep module load fast). `TextDelta` carries a text fragment; `ToolUseStart` signals the beginning of a tool call.

**Lines 198-201:** State initialisation. `initial_buf` is the guard buffer. `streaming` is the gate flag. Both are local to this invocation of `handle_message` — they reset every turn. `filler_count` tracks how many tool filler strings have been yielded this turn.

**Line 203:** Begin consuming the async generator. `run_agent_loop` yields events of several types; this loop only cares about `TextDelta` and `ToolUseStart`.

---

```python
    if isinstance(event, TextDelta):
        if streaming:
            yield event.text
        else:
            initial_buf.append(event.text)
            combined = ''.join(initial_buf)
            # Check if system-reminder tag is opening
            if '<system-reminder>' in combined:
                if '</system-reminder>' in combined:
                    # Complete tag — strip and flush remainder
                    clean = _SR_RE.sub('', combined)
                    clean = _TOOL_CALL_RE.sub('', clean).strip()
                    if clean:
                        yield clean
                    initial_buf = []
                    streaming = True
                # else: tag still open, keep buffering
            elif len(combined) > 30 or '\n' in combined:
                # No tag after enough text — safe to stream
                clean = _TOOL_CALL_RE.sub('', combined).strip()
                if clean:
                    yield clean
                initial_buf = []
                streaming = True
```

**`if streaming: yield event.text`** — the hot path. Once the guard has cleared, every fragment is forwarded immediately with zero buffering. This is critical for perceived latency: the client sees tokens as they arrive from the LLM.

**`initial_buf.append(event.text); combined = ''.join(initial_buf)`** — accumulate and rejoin. The LLM streams text in variable-sized chunks; a `<system-reminder>` tag boundary could fall mid-chunk, so the full accumulated string is checked each time rather than checking individual fragments.

**Three-branch decision:**
1. `'<system-reminder>' in combined and '</system-reminder>' in combined` — a complete tag is present. Strip it with `_SR_RE` (which handles DOTALL — the tag content may span multiple chunks). Also strip any `use_tool(...)` artifacts with `_TOOL_CALL_RE`. Yield the remainder if non-empty. Clear buf, open gate.
2. `'<system-reminder>' in combined` (opening tag but no closing tag yet) — fall through, keep buffering. The closing tag will arrive in a future chunk.
3. `len(combined) > 30 or '\n' in combined` — no reminder tag after enough content. The LLM is producing normal text. Flush (with `_TOOL_CALL_RE` strip only, no `_SR_RE`), open gate.

**Why 30 characters?** A `<system-reminder>` tag is 17 characters. By 30 characters of accumulated text, if no `<` has appeared, the response is clearly not a reminder echo. The newline heuristic catches shorter responses (e.g., "Yes.\n") before waiting for 30 chars.

---

```python
    elif isinstance(event, ToolUseStart):
        # Flush any buffered text as filler
        if initial_buf:
            raw = ''.join(initial_buf)
            clean = _SR_RE.sub('', raw)
            clean = _TOOL_CALL_RE.sub('', clean).strip()
            if clean:
                yield f"[FILLER]{clean}[/FILLER]"
            initial_buf = []
        streaming = False  # reset — post-tool text needs buffering again
        limit = self._max_fillers_per_turn  # 0=off, -1=unlimited, N=cap
        if limit != 0 and (limit < 0 or filler_count < limit):
            raw = self._tool_fillers.get(
                event.name, self._tool_fillers.get("_default", ""))
            if raw:
                filler = random.choice(raw) if isinstance(raw, list) else raw
                yield f"\n{filler}\n"
                filler_count += 1
        if self._show_tool_calls:
            yield f"[tool: {event.name}]"
```

**`ToolUseStart` handling — three actions in sequence:**

1. **Flush `initial_buf` as filler.** Any text the LLM produced before the tool call (e.g., "Let me check that for you...") is tagged as `[FILLER]...[/FILLER]`. This is UI metadata: the client can choose to display this as a loading indicator or discard it. It is tagged separately because it is not part of the final response.

2. **Reset `streaming = False`.** After tool execution, `run_agent_loop` loops back to the LLM call with tool results injected. The LLM may again open with a system-reminder echo or a new tool call. The guard must re-apply. This reset is what makes the state machine per-tool-call, not just per-turn.

3. **Yield configured filler text.** `self._tool_fillers` is a dict of `tool_name → str | list[str]` configured by the operator. If a list, one item is chosen at random. This is the "thinking..." / "searching..." UX message seen in production deployments. The `_default` key is a catch-all for unconfigured tool names.

**`limit` semantics:**
- `0` — fillers disabled globally
- `-1` — unlimited
- `N > 0` — cap at N fillers per turn (prevents a multi-tool turn from spamming filler strings)

---

```python
# Flush any remaining buffered text
if initial_buf:
    raw = ''.join(initial_buf)
    clean = _SR_RE.sub('', raw)
    clean = _TOOL_CALL_RE.sub('', clean).strip()
    if clean:
        yield clean
```

**Post-loop flush.** If the entire response was short enough to stay in `initial_buf` (under 30 chars, no newline, no system-reminder tag), it would never have been yielded during the loop. This final block ensures it is not lost. The same cleaning pipeline applies.

---

## Why `streaming=False` resets on every `ToolUseStart`

Consider a turn with two tool calls:

```
LLM: "Let me look that up."  [TextDelta × N]  → initial_buf accumulates
LLM: [ToolUseStart: search]                   → buf flushed as FILLER; streaming=False
...tool executes, result injected...
LLM: "Now let me also check." [TextDelta × M] → initial_buf accumulates again
LLM: [ToolUseStart: lookup]                   → buf flushed as FILLER; streaming=False
...tool executes, result injected...
LLM: "Here is your answer: ..." [TextDelta × K]
```

Without the `streaming=False` reset, the second and third bursts of text would bypass the guard, potentially forwarding reminder echoes or tool-call syntax to the client. The reset ensures the guard is re-applied after each tool boundary.

**Notice:** `run_agent_loop` yields `ToolUseStart` events synchronously from within its inner `async for event in runtime.provider.stream(...)` loop. These events appear *before* the tool is actually executed. The tool execution happens in `_execute_tool_uses`, which is called after the streaming inner loop completes. So by the time `handle_message` sees `ToolUseStart`, the tool has not yet run — the gateway's filler text is yielded while the tool is still pending.

---

## The async generator mechanics

`handle_message` is declared `async def` with `yield` statements. Python treats this as an **async generator function**. Calling `handle_message(session_id, content)` does not execute any code — it returns an async generator object. Execution begins only when the caller does `async for chunk in handle_message(...)`.

Each `yield` suspends `handle_message` and delivers the yielded value to the caller. Control resumes at the `yield` site when the caller requests the next value. This means:

- While the caller is processing a chunk (e.g., sending it over the WebSocket), `handle_message` is suspended.
- While `handle_message` is suspended inside `async for event in run_agent_loop(...)`, `run_agent_loop` is suspended inside `async for event in provider.stream(...)`.
- The asyncio event loop fills the gaps with I/O readiness callbacks from the HTTP/2 connection to the LLM API.

No threads are involved. No queues. The backpressure from the WebSocket send propagates up through the generator chain naturally.
