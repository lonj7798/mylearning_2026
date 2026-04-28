---
chapter: ch-03
course: boson-agent
phase: read
excerpt_for: packages/basement/basement/llm/openai_provider.py
created_at: "2026-04-17"
---

# Source: `basement/llm/openai_provider.py` — OpenAI Adapter

**One-line description:** The most complex of the three adapters; wraps the `openai.AsyncOpenAI` chat-completions API and must reconstruct discrete tool boundaries from index-based chunk fragments that arrive without explicit open/close events.

---

## Code Excerpt A — Tool Conversion (API Shape Difference)

```python
# packages/basement/basement/llm/openai_provider.py, lines 45-57

def _convert_tools(self, tools: list[ToolSpec]) -> list[dict]:
    """Convert ToolSpec list to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]
```

The OpenAI API requires the `{"type": "function", "function": {...}}` wrapper. This does not exist in Anthropic's format. `ToolSpec.input_schema` also gets renamed to `parameters`. Same data, two extra levels of nesting. The adapter is the only place this translation lives.

---

## Code Excerpt B — Message Conversion (System Prompt Placement)

```python
# packages/basement/basement/llm/openai_provider.py, lines 59-106

def _convert_messages(
    self, messages: list[Message], system: str
) -> list[dict]:
    """Convert Message list to OpenAI message format.

    OpenAI uses a system message in the messages array
    (unlike Anthropic's separate system parameter).
    """
    result = [{"role": "system", "content": system}]
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
        else:
            # Handle tool_use and tool_result blocks
            for block in msg.content:
                if block.type == "tool_use":
                    import json
                    result.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }],
                    })
                elif block.type == "tool_result":
                    result.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": block.content,
                    })
                elif block.type == "text":
                    result.append({"role": msg.role, "content": block.text})
    return result
```

The framework's internal `Message` type stores the system prompt separately from conversation history (matching Anthropic's API shape). OpenAI requires the system prompt as the first element of the messages array. The adapter's `_convert_messages` takes `system: str` as an extra parameter and prepends it as `{"role": "system", "content": system}` before the rest of the history.

The tool result block mapping is also different: OpenAI uses `role: "tool"` with `tool_call_id`, while Anthropic uses a `"tool_result"` block inside a user message. The adapter converts both directions.

---

## Code Excerpt C — Stream Loop (The Hard Part)

```python
# packages/basement/basement/llm/openai_provider.py, lines 134-187

        try:
            active_tool_ids: dict[int, str] = {}
            finish_reason: str | None = None

            response = await self._client.chat.completions.create(**kwargs)

            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                # Text content
                if delta and delta.content:
                    yield TextDelta(text=delta.content)

                # Tool calls
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index

                        # New tool call detected
                        if idx not in active_tool_ids and tc.id:
                            active_tool_ids[idx] = tc.id
                            name = (
                                tc.function.name if tc.function else "unknown"
                            )
                            yield ToolUseStart(id=tc.id, name=name)

                        # Argument fragments
                        if tc.function and tc.function.arguments:
                            yield InputJsonDelta(
                                partial_json=tc.function.arguments
                            )

            # Close any remaining active tools
            for idx in sorted(active_tool_ids.keys()):
                yield ToolUseEnd(id=active_tool_ids[idx])

            # Emit MessageEnd
            stop = "end_turn"
            if finish_reason == "tool_calls":
                stop = "tool_use"
            elif finish_reason == "stop":
                stop = "end_turn"
            elif finish_reason == "length":
                stop = "max_tokens"
            yield MessageEnd(stop_reason=stop)
```

---

## Explanation

This is why the source file has a `# AD4: OpenAI 250 LOC Budget` note — the OpenAI streaming protocol is significantly more complex than Anthropic's.

**No explicit open/close events.** Anthropic's SSE stream has `content_block_start` and `content_block_stop` events that act as explicit delimiters for tool blocks. OpenAI does not. Instead, each chunk contains a `delta.tool_calls` list where each entry has an integer `index`. The first chunk for a new tool call carries `tc.id` and `tc.function.name`; subsequent chunks for the same tool carry only `tc.function.arguments` fragments.

The adapter tracks which tool indexes have been seen using `active_tool_ids: dict[int, str]` (index → tool_id). When a chunk arrives with `idx not in active_tool_ids and tc.id`, that is the start of a new tool call, and the adapter emits `ToolUseStart`. Since OpenAI has no `content_block_stop`, the close events are only emitted **after** the async loop ends — `for idx in sorted(active_tool_ids.keys()): yield ToolUseEnd(...)`. The agent loop receives all `ToolUseEnd` events at the end, not interleaved with text.

The `finish_reason` translation maps OpenAI's three stop reasons (`"tool_calls"`, `"stop"`, `"length"`) to the framework's two (`"tool_use"`, `"end_turn"`, `"max_tokens"`). The agent loop checks `MessageEnd.stop_reason` to decide whether to enter the tool execution branch.

**Notice:** The OpenAI adapter calls `await self._client.chat.completions.create(**kwargs)` with `"stream": True` in kwargs (line 127). This returns an async iterable directly, not a context manager like Anthropic's `.messages.stream()`. The difference in SDK design means the adapter can't use `async with` — the cleanup is handled automatically when the async for loop exhausts.

**Connection to universal pattern:** Same Step 3–4 as the Anthropic adapter, but the translation is substantially harder. The `active_tool_ids` dict is the state machine that reconstructs the open/close structure that OpenAI's protocol leaves implicit.
