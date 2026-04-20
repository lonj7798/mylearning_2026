---
chapter: ch-03
course: boson-agent
phase: read
excerpt_for: packages/basement/basement/llm/google_provider.py
created_at: "2026-04-17"
---

# Source: `basement/llm/google_provider.py` — Google Gemini Adapter

**One-line description:** Wraps the `google-genai` SDK's synchronous `generate_content_stream`, bridges its part-based content model to `StreamEvent`, and synthesizes tool IDs that Gemini does not provide natively.

---

## Code Excerpt A — System Prompt Placement (a third approach)

```python
# packages/basement/basement/llm/google_provider.py, lines 60-98

def _convert_messages(
    self, messages: list[Message], system: str
) -> tuple[str | None, list[types.Content]]:
    """Convert Message list to Gemini content format.

    Returns (system_instruction, contents).
    """
    contents = []
    for msg in messages:
        role = "user" if msg.role == "user" else "model"

        if isinstance(msg.content, str):
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content)],
                )
            )
        else:
            parts = []
            for block in msg.content:
                if block.type == "text":
                    parts.append(types.Part.from_text(text=block.text))
                elif block.type == "tool_use":
                    parts.append(
                        types.Part.from_function_call(
                            name=block.name,
                            args=block.input,
                        )
                    )
                elif block.type == "tool_result":
                    parts.append(
                        types.Part.from_function_response(
                            name="tool_result",
                            response={"result": block.content},
                        )
                    )
            if parts:
                contents.append(types.Content(role=role, parts=parts))

    return system, contents
```

Three different system prompt conventions across three providers: Anthropic takes it as a top-level `system=` kwarg to the API call; OpenAI prepends it as a `{"role": "system"}` dict to the messages list; Gemini accepts it as `system_instruction=` in `GenerateContentConfig`. The return type `tuple[str | None, list[types.Content]]` makes explicit that the system string travels separately from the contents list — the caller (`stream`) routes them to the right places.

Role naming also differs: Gemini uses `"user"` and `"model"` (not `"assistant"`). The adapter converts inline: `role = "user" if msg.role == "user" else "model"`.

---

## Code Excerpt B — Stream Loop (Synchronous SDK, Tool ID Synthesis)

```python
# packages/basement/basement/llm/google_provider.py, lines 100-154

    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        system_instruction, contents = self._convert_messages(messages, system)

        config = types.GenerateContentConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
            system_instruction=system_instruction,
        )
        if tools:
            config.tools = self._convert_tools(tools)

        try:
            response = self._client.models.generate_content_stream(
                model=self._config.model,
                contents=contents,
                config=config,
            )

            has_tool_calls = False
            for chunk in response:
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    if part.text:
                        yield TextDelta(text=part.text)
                    elif part.function_call:
                        has_tool_calls = True
                        fc = part.function_call
                        tool_id = f"toolu_{uuid4().hex[:12]}"
                        yield ToolUseStart(id=tool_id, name=fc.name)
                        args = dict(fc.args) if fc.args else {}
                        yield InputJsonDelta(partial_json=json.dumps(args))
                        yield ToolUseEnd(id=tool_id)

            stop_reason = "tool_use" if has_tool_calls else "end_turn"
            yield MessageEnd(stop_reason=stop_reason)

        except Exception as e:
            raise ProviderError(f"Google API error: {e}") from e
```

---

## Explanation

The Google adapter has two characteristics not seen in the other two providers.

**Synchronous SDK, async generator wrapper.** The google-genai SDK's `generate_content_stream` returns a synchronous iterator (plain `for chunk in response`), not an async one. The adapter method is still declared `async def stream(...)` to satisfy the `LLMProvider` Protocol, and it uses `yield` to make it an async generator — but the inner loop is a regular `for`. This works because Python allows `yield` inside an `async def`, turning it into an async generator regardless of whether the inner iteration is sync or async. The caller (`agent_loop`) uses `async for`, which drives the async generator protocol.

**Tool ID synthesis.** Gemini's function call API does not assign a unique ID to each tool call — it returns `part.function_call.name` and args but no `id` field. The framework's `ToolUseBlock` and `ToolUseEnd` both require an `id` to match start and end events and correlate with tool results. The adapter synthesizes one: `tool_id = f"toolu_{uuid4().hex[:12]}"`. Since the ID is invented here and used consistently within the same part, the agent loop's matching still works. The ID never leaves the framework; Gemini's API doesn't know about it.

**No argument streaming.** Unlike Anthropic and OpenAI, Gemini delivers function arguments as a complete dict in a single part (`fc.args`), not as JSON fragments. The adapter still emits `InputJsonDelta` (with the full JSON in one call) and `ToolUseEnd` immediately after `ToolUseStart` — making the event sequence identical to what the other adapters produce, even though Gemini produced no actual deltas.

**Notice:** The three events `ToolUseStart → InputJsonDelta → ToolUseEnd` are emitted for every function call inside a single `for part in candidate.content.parts` iteration. The agent loop's `tool_uses[-1]["input_json"] += event.partial_json` accumulation still works — it just concatenates a single fragment that happens to be the complete JSON.

**Connection to universal pattern:** Google illustrates that the universal pattern's output (the five-event stream) can be produced from very different underlying protocols, including ones that deliver complete tool calls atomically rather than incrementally. The adapter synthesizes the streaming shape even when the underlying API is not streaming tool arguments.
