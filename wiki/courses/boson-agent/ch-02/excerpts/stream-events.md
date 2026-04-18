---
chapter: ch-02
course: boson-agent
phase: read
kind: excerpt
source: packages/basement/basement/llm/base.py
created_at: 2026-04-17T00:00:00Z
---

# Excerpt: `llm/base.py` — StreamEvent Types and LLMProvider Protocol

← Back to [[../read]]

---

## Source description

`llm/base.py` is 64 lines. It defines two things: the five `StreamEvent` Pydantic models that the loop consumes, and the `LLMProvider` Protocol that any streaming LLM backend must implement. Nothing else. This is LOD Pattern 7 (zero hallucination): all types are explicit Pydantic models, not dicts or ad-hoc strings.

---

## Excerpt 1 — The five StreamEvent types (lines 20-51)

```python
# packages/basement/basement/llm/base.py, lines 20-51
class TextDelta(BaseModel):
    """Incremental text token from LLM response."""
    text: str


class ToolUseStart(BaseModel):
    """Start of a tool_use block in LLM response."""
    id: str
    name: str


class InputJsonDelta(BaseModel):
    """Incremental JSON fragment for tool input arguments."""
    partial_json: str


class ToolUseEnd(BaseModel):
    """End of a tool_use block."""
    id: str


class MessageEnd(BaseModel):
    """End of LLM response message."""
    stop_reason: str  # "end_turn" | "tool_use"


StreamEvent = TextDelta | ToolUseStart | InputJsonDelta | ToolUseEnd | MessageEnd
```

**What this shows mechanically.** The five classes map directly to the Anthropic streaming event taxonomy. The Anthropic API emits events in this sequence for a tool-use response:

```
content_block_start  (type="tool_use", id=..., name=...)  → ToolUseStart
content_block_delta  (type="input_json_delta", partial=)  → InputJsonDelta  (repeated)
content_block_stop                                         → ToolUseEnd
message_delta        (stop_reason="tool_use")              → MessageEnd
```

And for a text response:
```
content_block_start  (type="text")
content_block_delta  (type="text_delta", text=...)        → TextDelta  (repeated)
content_block_stop
message_delta        (stop_reason="end_turn")              → MessageEnd
```

The framework normalizes all three providers (Anthropic, OpenAI, Google) to emit this same five-type vocabulary. The loop only ever sees these five types, regardless of which provider is in use.

`StreamEvent = TextDelta | ToolUseStart | InputJsonDelta | ToolUseEnd | MessageEnd` is a type alias for the union. The loop's `isinstance` dispatching is exhaustive over this union — if a provider emits an unknown type, all five `isinstance` branches fail and the event is silently dropped (since there is no `else` clause).

**Notice.** `ToolUseEnd` carries the `id` of the tool call it closes. The loop body at `agent_loop.py` line 129 does `pass` — it does not use `event.id`. The id is available for callers who `yield event` upstream and need to correlate tool-end events, but the loop itself does not require it because it tracks tool state via list position (`tool_uses[-1]`) rather than id lookup.

**Connection to universal pattern.** These five types are the complete vocabulary of the "THINK" phase of the universal pattern. Understanding their sequencing is equivalent to understanding what the LLM wire protocol looks like from the framework's perspective.

---

## Excerpt 2 — The LLMProvider Protocol (lines 54-63)

```python
# packages/basement/basement/llm/base.py, lines 54-63
@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider implementations."""

    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
```

**What this shows mechanically.** `LLMProvider` is a structural Protocol (PEP 544): any class with an `async def stream(messages, system, tools)` method satisfies it, without inheriting from it. The `@runtime_checkable` decorator allows `isinstance(provider, LLMProvider)` to work at runtime, though the loop does not check this — it just calls `runtime.provider.stream()` and trusts the type.

The method signature is minimal: `messages` (full conversation history), `system` (system prompt string), `tools` (list of `ToolSpec` or `None`). No streaming configuration, no temperature, no API key — those are baked into the provider instance at construction time from `AgentConfig.llm`. The call site is clean.

The return type `AsyncIterator[StreamEvent]` means `stream()` is an async generator function. Callers iterate it with `async for event in runtime.provider.stream(...)`, which is exactly what the loop does at `agent_loop.py` lines 112-131.

**Notice.** `tools: list[ToolSpec] | None = None` defaults to `None`, not `[]`. This matters: sending `tools=[]` to some LLM APIs triggers an error or a warning ("empty tools list"), whereas sending `tools=None` or omitting the parameter entirely means "no tool calling." The loop converts `get_all_specs()` results to `None` via `or None` before passing to `stream()`.

**Connection to universal pattern.** The Protocol is the seam that makes the universal pattern provider-agnostic. The loop is written against the Protocol, not against the Anthropic SDK. Swapping providers — or using a `FakeProvider` in tests — requires only that the new class implement `stream()` with this signature.

---

## How the five events interleave in a tool-chain turn

This timeline traces a single turn where the LLM calls two tools sequentially (`search`, then `summarize`), then responds with text:

```
Iteration 1 of while loop:
  LLM stream:
    ToolUseStart(id="tu_1", name="search")
    InputJsonDelta(partial_json='{"q')
    InputJsonDelta(partial_json='uery": "quantum"}')
    ToolUseEnd(id="tu_1")
    MessageEnd(stop_reason="tool_use")
  → tool_uses = [{"id":"tu_1","name":"search","input_json":'{"query":"quantum"}'}]
  → execute search → ToolResultBlock appended as user msg
  → continue

Iteration 2 of while loop:
  LLM stream:
    ToolUseStart(id="tu_2", name="summarize")
    InputJsonDelta(partial_json='{"text": "…"}')
    ToolUseEnd(id="tu_2")
    MessageEnd(stop_reason="tool_use")
  → tool_uses = [{"id":"tu_2","name":"summarize","input_json":'{"text":"…"}'}]
  → execute summarize → ToolResultBlock appended as user msg
  → continue

Iteration 3 of while loop:
  LLM stream:
    TextDelta(text="Here is")
    TextDelta(text=" the summary:")
    TextDelta(text=" quantum computing …")
    MessageEnd(stop_reason="end_turn")
  → tool_uses = []  (empty — text-only branch)
  → ctx.add_message("assistant", "Here is the summary: …")
  → POST_LLM_CALL fires
  → break
```

The `turn_count` increments to 3. If `max_turns` were 2, the loop would have exhausted after iteration 2 — yielding the warning text and stopping before the LLM could give its final response. This is the exact scenario `max_turns` prevents: an infinite tool chain.
