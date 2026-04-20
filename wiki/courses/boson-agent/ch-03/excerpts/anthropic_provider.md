---
chapter: ch-03
course: boson-agent
phase: read
excerpt_for: packages/basement/basement/llm/anthropic_provider.py
created_at: "2026-04-17"
---

# Source: `basement/llm/anthropic_provider.py` — Anthropic Adapter

**One-line description:** Wraps the `anthropic.AsyncAnthropic` SDK client, converts internal types to Anthropic API shapes, and translates Anthropic SSE events into the framework's `StreamEvent` union.

---

## Code Excerpt A — Tool Conversion

```python
# packages/basement/basement/llm/anthropic_provider.py, lines 40-49

def _convert_tools(self, tools: list[ToolSpec]) -> list[dict]:
    """Convert ToolSpec list to Anthropic tool format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in tools
    ]
```

This shows the before/after of adapter conversion in its simplest form. A `ToolSpec` has `name`, `description`, and `input_schema` (a JSON Schema dict). The Anthropic API happens to want exactly the same three fields — so `_convert_tools` is a one-to-one projection. The field name `input_schema` is Anthropic's terminology, and `ToolSpec` was designed to match it. This alignment is intentional: Anthropic was the primary target when the framework was built.

Compare this to OpenAI's `_convert_tools` (in `openai_provider.py`): OpenAI wraps the same data in an extra `{"type": "function", "function": {...}}` envelope and renames `input_schema` to `parameters`. Same data, different wrapping — the adapter absorbs the difference.

---

## Code Excerpt B — The `stream` Method (Core of the Adapter)

```python
# packages/basement/basement/llm/anthropic_provider.py, lines 64-126

async def stream(
    self,
    messages: list[Message],
    system: str,
    tools: list[ToolSpec] | None = None,
) -> AsyncIterator[StreamEvent]:
    kwargs = {
        "model": self._config.model,
        "max_tokens": self._config.max_tokens,
        "system": system,
        "messages": self._convert_messages(messages),
    }
    if self._config.temperature is not None:
        kwargs["temperature"] = self._config.temperature
    if tools:
        kwargs["tools"] = self._convert_tools(tools)

    try:
        current_tool_id: str | None = None

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        yield ToolUseStart(id=block.id, name=block.name)

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif delta.type == "input_json_delta":
                        yield InputJsonDelta(partial_json=delta.partial_json)

                elif event.type == "content_block_stop":
                    if current_tool_id is not None:
                        yield ToolUseEnd(id=current_tool_id)
                        current_tool_id = None

                elif event.type == "message_stop":
                    pass  # handled below via get_final_message()

            final = await stream.get_final_message()
            yield MessageEnd(stop_reason=final.stop_reason)

    except anthropic.APIError as e:
        raise ProviderError(f"Anthropic API error: {e}") from e
```

---

## Explanation

The `stream` method is an **async generator** — the `yield` keywords inside the `async with` block are where the framework actually produces `StreamEvent` values. When the agent loop does `async for event in runtime.provider.stream(...)`, Python drives this coroutine forward one `yield` at a time. Nothing is buffered; each `yield TextDelta(...)` immediately reaches the agent loop's `isinstance` check. This is the mechanism the learner called "magic" in ch-02 — the magic is just Python's async generator protocol: each `yield` suspends the function and hands its value to whoever is iterating.

The Anthropic SSE stream produces events with string `type` fields: `"content_block_start"`, `"content_block_delta"`, `"content_block_stop"`, `"message_stop"`. The adapter maps these to framework types:

| Anthropic SSE event | Condition | Framework type emitted |
|---|---|---|
| `content_block_start` | `block.type == "tool_use"` | `ToolUseStart` |
| `content_block_delta` | `delta.type == "text_delta"` | `TextDelta` |
| `content_block_delta` | `delta.type == "input_json_delta"` | `InputJsonDelta` |
| `content_block_stop` | `current_tool_id is not None` | `ToolUseEnd` |
| After loop | via `get_final_message()` | `MessageEnd` |

The `current_tool_id` variable is a one-element buffer tracking which tool block is currently open. When `content_block_stop` arrives, the adapter emits `ToolUseEnd` with that ID and clears the variable. This state is local to the `stream` call — no class-level mutation.

The `tools` guard at line 88 (`if tools:`) means if `tools=None`, the `kwargs` dict never gets a `"tools"` key, so the Anthropic API call omits the field entirely. This is how `tools=None` forces a text-only response at the API layer: without the `tools` field in the request, the model cannot produce a `tool_use` block regardless of what the system prompt says.

**Notice:** `message_stop` is explicitly caught but does nothing (`pass`). The `stop_reason` is obtained via `await stream.get_final_message()` after the SSE loop finishes — the SDK buffers the final message object separately from the event stream. This is an Anthropic SDK implementation detail the adapter hides.

**Connection to universal pattern:** This is Step 3–4 of the pattern — format conversion and event translation. The `yield` statements are the literal points where Anthropic's SSE vocabulary becomes the framework's `StreamEvent` vocabulary.
