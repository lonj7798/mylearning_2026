---
chapter: ch-08
course: boson-agent
phase: read
excerpt_of: packages/gateway/gateway/core.py + packages/basement/basement/loop/agent_loop.py
created_at: "2026-04-19"
---

# Excerpt: Streaming Translation — `gateway/core.py` + `basement/loop/agent_loop.py`

One-line description: This is the full async-generator chain that converts
a single LLM `TextDelta` event emitted deep inside `run_agent_loop` into
a `text_delta` JSON frame on the WebSocket wire — every `await` and `yield`
point annotated.

---

## The full yield chain, layer by layer

The path from provider token to wire frame passes through four async
generator boundaries. Reading bottom-up (provider → wire):

```
LLM provider.stream()           — yields StreamEvent (TextDelta, ToolUseStart, …)
    ↓  async for event in provider.stream(...)
run_agent_loop(runtime, input)  — yields StreamEvent unchanged for TextDelta
    ↓  async for event in run_agent_loop(runtime, content)
GatewayCore.handle_message()    — filters, strips, yields str chunks
    ↓  async for chunk in self._message_handler(session_id, content)
_process_message()              — serializes each chunk to JSON, awaits websocket.send()
    → wire
```

---

## Layer 1: LLM provider — `provider.stream()` (agent_loop.py, lines 112–131)

```python
# packages/basement/basement/loop/agent_loop.py, lines 112-131

async for event in runtime.provider.stream(
    messages=ctx.get_messages(),
    system=ctx.get_system_prompt(),
    tools=tools,
):
    if isinstance(event, TextDelta):
        text_parts.append(event.text)
        yield event                      # (1) re-yield TextDelta to caller
    elif isinstance(event, ToolUseStart):
        tool_uses.append(
            {"id": event.id, "name": event.name, "input_json": ""}
        )
        yield event                      # (2) re-yield ToolUseStart to caller
    elif isinstance(event, InputJsonDelta):
        if tool_uses:
            tool_uses[-1]["input_json"] += event.partial_json
    elif isinstance(event, ToolUseEnd):
        pass
    elif isinstance(event, MessageEnd):
        yield event
```

`provider.stream()` is itself an async generator (the concrete
implementation calls the Anthropic API with `stream=True` and yields
SSE events as typed objects). Each `TextDelta(text="…")` is yielded
immediately without buffering — low latency is the priority at this layer.

`run_agent_loop` is also an async generator (it uses `yield` internally).
When `yield event` executes at point (1), the loop suspends and the
`TextDelta` object travels up to whoever is iterating `run_agent_loop`.

**Notice:** `text_parts` accumulates a local copy of every text chunk for
the tool-chaining logic (line 139: build the assistant message after the
loop). This means each token is stored in memory as well as yielded
upward — a deliberate tradeoff (correctness of history over memory
minimalism).

---

## Layer 2: GatewayCore — `handle_message()` filtering logic (core.py, lines 203–255)

```python
# packages/gateway/gateway/core.py, lines 196-255

from basement.llm.base import TextDelta, ToolUseStart
filler_count = 0
initial_buf: list[str] = []    # (A) system-reminder stripping buffer
streaming = False

async for event in run_agent_loop(runtime, content):  # (B) consume agent loop
    if isinstance(event, TextDelta):
        if streaming:
            yield event.text               # (C) fast path: direct yield
        else:
            initial_buf.append(event.text)
            combined = ''.join(initial_buf)
            if '<system-reminder>' in combined:
                if '</system-reminder>' in combined:
                    # Complete tag — strip and flush remainder
                    clean = _SR_RE.sub('', combined)
                    clean = _TOOL_CALL_RE.sub('', clean).strip()
                    if clean:
                        yield clean        # (D) deferred yield after stripping
                    initial_buf = []
                    streaming = True
                # else: tag still open, keep buffering
            elif len(combined) > 30 or '\n' in combined:
                # No tag after enough text — safe to stream
                clean = _TOOL_CALL_RE.sub('', combined).strip()
                if clean:
                    yield clean            # (E) buffered flush, no tag found
                initial_buf = []
                streaming = True
    elif isinstance(event, ToolUseStart):
        if initial_buf:
            raw = ''.join(initial_buf)
            clean = _SR_RE.sub('', raw)
            clean = _TOOL_CALL_RE.sub('', clean).strip()
            if clean:
                yield f"[FILLER]{clean}[/FILLER]"   # (F) pre-tool text as filler
            initial_buf = []
        streaming = False   # reset — post-tool text needs buffering again
        ...
        if self._show_tool_calls:
            yield f"[tool: {event.name}]"

# Flush any remaining buffered text
if initial_buf:
    raw = ''.join(initial_buf)
    clean = _SR_RE.sub('', raw)
    clean = _TOOL_CALL_RE.sub('', clean).strip()
    if clean:
        yield clean                        # (G) tail flush
```

This is the most complex yield point in the chain. `handle_message` is
itself an async generator — it uses `yield` to pass text chunks up to the
WebSocket layer. Key annotations:

- **(A)** `initial_buf` exists to catch `<system-reminder>` tags that the
  LLM occasionally echoes at the start of a response. The buffer holds
  tokens until it can confirm no tag is opening (>30 chars or a newline
  arrived without a tag) or until it has a complete tag to strip.
- **(B)** `async for event in run_agent_loop(...)` — this is the suspension
  point. Every time the agent loop yields a `TextDelta`, this coroutine
  resumes here.
- **(C)** Once `streaming = True`, every subsequent `TextDelta.text` is
  yielded immediately — the buffer is bypassed for low latency.
- **(D)** and **(E)** are the two deferred-yield cases: either a complete
  `<system-reminder>` tag was found and stripped, or enough text arrived to
  confirm no tag is present.
- **(F)** Pre-tool text (text before a tool call) is wrapped in
  `[FILLER]...[/FILLER]` markers so the client can display it as a
  transitional message while the tool runs.
- **(G)** The tail flush handles the case where the response ended while
  text was still in `initial_buf` (short responses that never triggered the
  30-char threshold).

**Notice:** `streaming = False` is reset on every `ToolUseStart` (line 237).
This means after every tool call, the system-reminder detection buffer is
re-armed. This handles multi-tool chains: the LLM might inject another
`<system-reminder>` at the start of its post-tool response.

---

## Layer 3: `_process_message()` — wire encoding (websocket.py, lines 205–217)

```python
# packages/gateway/gateway/server/websocket.py, lines 205-217

async for chunk in self._message_handler(session_id, content):  # (H)
    delta = serialize_server_message(
        ServerMessage(
            session_id=session_id, type="text_delta", content=chunk
        )
    )
    await websocket.send(delta)                                  # (I)

turn_end = serialize_server_message(
    ServerMessage(session_id=session_id, type="turn_end")
)
await websocket.send(turn_end)                                   # (J)
```

- **(H)** Resumes whenever `handle_message` yields a `str`. At this point
  the string is already cleaned (no `<system-reminder>`, no tool-call
  syntax).
- **(I)** `await websocket.send(delta)` is the only true I/O call in the
  chain. It suspends until the frame is handed to the OS TCP send buffer.
  This is where backpressure surfaces: if the client is slow to read,
  `websocket.send` will block here, which naturally throttles the
  `async for chunk` loop above, which in turn holds the
  `async for event in run_agent_loop(...)` loop, which stops pulling from
  `provider.stream()`.
- **(J)** After the generator is exhausted, `turn_end` is sent. No further
  chunks will arrive until the next `user_message`.

---

## Annotated full call trace for one token ("Hello")

```
provider.stream()           yields  TextDelta(text="Hello")
  └─ agent_loop.py:118      yield event                     → TextDelta travels up
       └─ core.py:205        async for event in run_agent_loop(...)  resumes
            len("Hello") < 30, no newline → initial_buf = ["Hello"]
            (buffering; no yield yet)
       ... next event: MessageEnd → agent_loop exits
       core.py:249  tail flush: initial_buf = ["Hello"]
            clean = "Hello"
            yield "Hello"                                   → str travels up
  └─ websocket.py:206  async for chunk in self._message_handler(...)  resumes
       chunk = "Hello"
       serialize → '{"session_id":"s1","type":"text_delta","content":"Hello"}'
       await websocket.send(...)                            → bytes on wire
```

In a longer response where `streaming = True` is reached (>30 chars before
first newline), each subsequent token travels the fast path:
`TextDelta.text` is re-yielded by `agent_loop`, immediately yielded by
`handle_message` (core.py line 206: `yield event.text`), immediately
serialized and sent by `_process_message`. Three `yield`/`await` handoffs
per token — no intermediate buffering.

Connection to universal pattern: this excerpt is the core of step 5 in the
universal pattern — the per-turn translation loop that converts async-generator
events into protocol frames.
