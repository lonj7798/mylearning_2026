---
title: "Two Agent Loops: Who Owns the Context, the Tools, and the End of a Turn"
chapter: ch-09
phase: collision
course: pipecat
sources:
  - llm-service-context
  - function-calling
  - boson-agent-loop
  - boson-tool-router
  - rtv-pipeline-session
deps:
  - ch-03
  - ch-08
figure: figures/two-loops.html
---

# Chapter 9 — Two Agent Loops: Who Owns the Context, the Tools, and the End of a Turn

> **Scope, stated up front and enforced for the whole chapter.**
>
> **One idea: loop ownership.** Three questions, and only three — who holds the message list, who
> dispatches a tool, who decides the turn ended. Everything below serves one of those three.
>
> **Context compaction is NOT in this chapter.** `LLMContextSummarizer`
> (`src/pipecat/processors/aggregators/llm_context_summarizer.py:57`) versus boson's
> `gateway/compact/` is a different subsystem with different failure modes and a different trigger,
> and it is on [[ch-13/read]]'s give-back list. If you catch yourself reasoning about *when history
> gets shortened*, you have left this chapter. Come back to *who is allowed to touch history at all*.
>
> **No comparative verdict.** This chapter says what each design **does** and what each choice
> **costs**. It does not say which is better, which wins, which you should adopt, or which is "the
> right choice." Three resolutions are developed and costed at §11 and **none of them is marked
> recommended**. [[ch-13/read]] is the only place in this course where anything gets scored, and it
> gets to do that only because it will have seen all twelve subsystems. This invariant was broken
> twice while this outline was under review. It is not broken here.

---

## 왜 이 챕터인가

[[ch-03/read]] ended §7.6 with a sentence it deliberately refused to finish:

> There is no Pipecat analogue of a "text-in / text-out agent slot that is forbidden to know about
> tools." Whether the two contracts can coexist, and what it would take, is [[ch-09/read]]'s entire
> subject. Do not resolve it here.

This is that chapter, and the collision is deeper than that sentence made it sound. It is not that
Pipecat lacks a text-only agent slot. It is that **Pipecat and boson-agent disagree about who is the
subject of the sentence "the agent runs a turn."**

In boson, the subject is a function. `run_agent_loop(runtime, user_input)` is 561 lines that own the
message list, the provider call, the tool dispatch, the turn counter, the hooks, and the history
repair after cancellation — all six, in one lexical scope, where you can read the whole turn
top-to-bottom.

In Pipecat, there is no subject. The turn is a **circulation**: a plain object holds the messages,
two processors on either side of the LLM mutate it, the LLM service reads it and reads nothing else,
and the loop closes because the processor at the *end* of the pipeline pushes a frame **backwards**
into the processor in the *middle* of it. Nobody is running the loop. The loop is the shape of the
graph.

That difference has three consequences you will spend this chapter measuring, and every one of them
is a real cost you would pay in Lina TMR:

1. Pipecat's `get_messages()` hands out the **live list**, on purpose, because the assistant
   aggregator rewrites tool-result dicts **in place**. boson's `ContextManager.get_messages()`
   returns `deepcopy(self._messages)`, on purpose, to prevent exactly that. A one-line "defensive"
   port silently breaks Pipecat's tool path — no exception, no log, just tool results that never
   reach the model.
2. Pipecat has **no turn cap**. Not a large one — none. `grep -rn "max_turns\|max_iterations\|max_tool_calls" src/pipecat/` returns zero hits. boson's `max_turns=50` and its `while/else`
   exhaustion path have nothing to port onto.
3. boson keeps **three separate gates** on a tool — exposure, availability, permission — and
   Pipecat's registry is a `dict[str | None, FunctionCallRegistryItem]` with slots for none of them.

Two of those three are things Pipecat *does not have*. That is not a criticism; a framework can
decline to have a policy. It is an inventory item, and the inventory is what [[ch-13/read]] prices.

---

## 0. How to read the evidence in this chapter

Same two-class rule as [[ch-03/read]] §0. It matters more here than anywhere else in the course,
because this chapter is where an earlier draft of the course outline was **wrong four separate
times**, and every one of those errors came from trusting a summary instead of opening a file.

| Class | Source of truth | How you check it |
|---|---|---|
| **Pipecat claims** — paths, line numbers, class names, greps, LOC | `wiki/raw-data/pipecat/pipecat-src` at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` | Open the file. Every line number below was re-read against that tree on 2026-08-25. Where an excerpt and the source disagreed, the source won and I say so. |
| **boson-agent / realtime_voice claims** | the `boson-*` and `rtv-*` excerpts under `wiki/raw-data/pipecat/excerpts/`, read from the private repos | Check against your own repo. Not checkable from this wiki, and nothing here pretends otherwise. Every boson code block carries its repo path **and** the excerpt wikilink. |

boson-agent excerpts are from commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb` (2026-08-20).
realtime_voice is from branch `voice-chat-dev`, commit `034ce4ca09a2f109e6c248a43bc989f8d26a6abf`
(2026-07-29). If either has moved, the shapes below are what they were then.

### 0.1 Four facts the plan for this chapter had wrong

Read these first. They are not trivia — two of them are load-bearing for the whole argument, and one
of them is an argument that *only works* because of a single line number.

**(a) `create_context_aggregator()` does not exist.** Every Pipecat tutorial older than about a
year opens with `context_aggregator = llm.create_context_aggregator(context)`. That method is gone
at this commit:

```console
$ cd wiki/raw-data/pipecat/pipecat-src
$ grep -rn "create_context_aggregator" src/ examples/ | wc -l
0
```

The only hits in the whole tree are in `CHANGELOG.md`, documenting its removal:

```markdown
# CHANGELOG.md:4278-4286
- ⚠️ Removed deprecated service-specific context and aggregator machinery,
  which was superseded by the universal `LLMContext` system.

  Service-specific classes removed: `AnthropicLLMContext`,
  `AnthropicContextAggregatorPair`, `AWSBedrockLLMContext`,
  `AWSBedrockContextAggregatorPair`, `OpenAIContextAggregatorPair`, and their
  user/assistant aggregators. Also removed `create_context_aggregator()` from
  `LLMService`, `OpenAILLMService`, `AnthropicLLMService`, and
  `AWSBedrockLLMService`.
```

The replacement is `LLMContext(messages, tools)` + `LLMContextAggregatorPair(context)`. Any tutorial
showing the old form will not run. This matters beyond typing: the removal is *the* signal that the
context stopped being a thing the service manufactures and became a thing **the application owns and
hands in**. That is the whole of §2.

**(b) `register_direct_function` is deprecated.** It still exists, but:

```python
# src/pipecat/services/llm_service.py:982-998
    @deprecated(
        "`LLMService.register_direct_function` is deprecated since 1.4.0 and will be removed in "
        "2.0.0. Use `LLMContext(tools=[...])` instead."
    )
    def register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool | None = None,
        timeout_secs: float | None = None,
    ):
        """Register a direct function handler for LLM function calls.

        .. deprecated:: 1.4.0
            Direct functions are now registered automatically. List them in
            ``LLMContext(tools=[...])`` for tools available at session start, or
            push an :class:`LLMSetToolsFrame` to change tools mid-session.
            Will be removed in 2.0.0.
```

Same signal as (a), in the tool dimension: registration moved off the service and onto the context.

**(c) Pipecat has no tool loop and no turn cap.** Both are absences, both verified by grep, both in
§5 and §6.

**(d) The completion starts at `base_llm.py:605`, not `:604`.** An earlier draft of this outline
cited `:604` and built its central claim on it. That is wrong, and the correction is not cosmetic —
the four consecutive lines do four different things, and the argument "the service performs exactly
one inference per frame and then returns" is an argument about which line the inference is on:

```python
# src/pipecat/services/openai/base_llm.py:599-605
        await super().process_frame(frame, direction)          # :599  mandatory, non-negotiable

        if isinstance(frame, LLMContextFrame):                 # :601  the only trigger
            try:
                await self.push_frame(LLMFullResponseStartFrame())   # :603  bracket opens
                await self.start_processing_metrics()                # :604  metrics clock starts
                await self._process_context(frame.context)           # :605  ← the completion
```

`:601` is the `isinstance` test. `:603` pushes the opening bracket frame. `:604` starts the metrics
clock. `:605` is where a token is generated. Verify it yourself before you read another paragraph:

```console
$ awk 'NR>=599 && NR<=605 {printf "%d\t%s\n", NR, $0}' src/pipecat/services/openai/base_llm.py
```

I am pointing at this hard because it is the shape of every error the outline made: a summary said
"the service pushes a start frame and processes the context," a reader collapsed that into one line
number, and an argument got built on the collapsed version.

---

## 1. The three ownership questions, and the three answers on the table

Before any code, fix the questions. Everything in this chapter is an answer to exactly one of them.

| | **Q1. Who holds the message list?** | **Q2. Who dispatches a tool?** | **Q3. Who decides the turn ended?** |
|---|---|---|---|
| **Pipecat** | `LLMContext`, a plain object the *application* constructs (`llm_context.py:83`). Two aggregators mutate it in place. The LLM service holds nothing. | `LLMService.run_function_calls()` → `_run_function_call()` (`llm_service.py:1437`, `:1547`), from a registry the context populates. | Nobody. The turn ends when no `LLMContextFrame` is pushed upstream — an absence, not a decision. |
| **boson-agent** | `ContextManager`, reached as `runtime.context_manager` and mutated only from inside `run_agent_loop` (`agent_loop.py:184`). | `ToolRouter.dispatch()` (`metatool/router.py:103`), through `_execute_tool_uses` (`agent_loop.py:393`). | `run_agent_loop` itself, at `break  # Done — text response means end of turn` (`agent_loop.py:363`). |
| **realtime_voice** | Nothing in the voice package. `StreamingConversationAgent` never sees a message list. | Nothing in the voice package. Tools are behind `GatewayConversationAgent`. | The agent slot, by ending its `AsyncIterator[AgentTextDelta]`. |

Read the third row carefully. realtime_voice's answer to all three questions is **"not me."** That is
not an omission; it is the contract, quoted from boson's own `CLAUDE.md` in [[rtv-vs-pipecat-gap]]
and reproduced in [[ch-03/read]] §7.6: *"Keep Basement and the dental business logic text-native."*
The voice package is forbidden to know what a tool is.

That third row is why this chapter is not hypothetical. §9 comes back to it.

---

## 2. Pipecat, Q1: the message list is a plain object the application hands in

### 2.1 `LLMContext` is three fields

Open it. It is 510 lines, and the state is three attributes:

```python
# src/pipecat/processors/aggregators/llm_context.py:83, 91-112
class LLMContext:
    """Manages conversation context for LLM interactions.
    ...
    """

    def __init__(
        self,
        messages: list[LLMContextMessage] | None = None,
        tools: ToolsSchema | list[FunctionSchema | DirectFunction] | NotGiven = NOT_GIVEN,
        tool_choice: LLMContextToolChoice | NotGiven = NOT_GIVEN,
    ):
        ...
        self._messages: list[LLMContextMessage] = messages if messages else []
        self._tools: ToolsSchema | NotGiven = LLMContext._normalize_and_validate_tools(tools)
        self._tool_choice: LLMContextToolChoice | NotGiven = tool_choice
```

Note what is **not** there. No provider. No client. No model name. No system prompt field. No turn
counter. No lock. No `asyncio` anything. `LLMContext` is a bag of three values with methods, and it
is not a `FrameProcessor` — it never appears in a `Pipeline` list, it is passed *to* the things that
do.

A detail that will matter in §8 when you compare message schemas: `LLMContextMessage` is a union,
and its "standard" arm is OpenAI's type — but only when a type checker is looking:

```python
# src/pipecat/processors/aggregators/llm_context.py:50-65
# The aliases resolve under type checking only. Every LLM service reaches this
# module, so importing the OpenAI SDK here would put its load on the startup
# path of pipelines that never talk to OpenAI. At runtime both are structurally
# dicts or strings, which is all the annotations need them to be.
if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolChoiceOptionParam,
    )

    LLMStandardMessage: TypeAlias = ChatCompletionMessageParam
    LLMContextToolChoice: TypeAlias = ChatCompletionToolChoiceOptionParam
else:
    LLMStandardMessage: TypeAlias = Any
    LLMContextToolChoice: TypeAlias = Any
```

So at runtime a Pipecat message is a **plain dict**, in OpenAI's shape, with zero validation. There
is no `Message` class, no pydantic model, no `ContentBlock`. Hold that thought until §8.4 — boson's
entire `basement.schemas.message_schema` layer has no landing site here.

### 2.2 The mutators, and the one that mutates in place

```python
# src/pipecat/processors/aggregators/llm_context.py:361-394
    def add_message(self, message: LLMContextMessage):
        """Add a single message to the context.
        ...
        """
        self._messages.append(message)

    def add_messages(self, messages: list[LLMContextMessage]):
        """Add multiple messages to the context.
        ...
        """
        self._messages.extend(messages)

    def set_messages(self, messages: list[LLMContextMessage]):
        """Replace all messages in the context.
        ...
        """
        self._messages[:] = messages

    def transform_messages(
        self, transform: Callable[[list[LLMContextMessage]], list[LLMContextMessage]]
    ):
        """Transform the current messages using the provided function.
        ...
        """
        self.set_messages(transform(self._messages))
```

`set_messages` is `self._messages[:] = messages`, not `self._messages = messages`. That is a slice
assignment: it mutates the existing list object rather than rebinding the attribute. Anyone else
holding a reference to that same list — and in a running pipeline there are several — sees the
replacement. If it were a rebind, a processor that had cached `ctx.get_messages()` would keep
looking at the old list forever.

This is the first sighting of the design principle that runs through the whole chapter: **Pipecat
optimises for many holders of one object.** boson optimises for one holder and defensive copies out.

### 2.3 `get_messages()` returns the live list, and that is deliberate

Here is the method that a careful engineer will be tempted to "fix":

```python
# src/pipecat/processors/aggregators/llm_context.py:221-258
    def get_messages(
        self,
        llm_specific_filter: str | None = None,
        *,
        truncate_large_values: bool = False,
    ) -> list[LLMContextMessage]:
        """Get the current messages list.
        ...
        """
        if llm_specific_filter is None:
            messages = self._messages
        else:
            messages = [
                msg
                for msg in self._messages
                if not isinstance(msg, LLMSpecificMessage) or msg.llm == llm_specific_filter
            ]
            if len(messages) < len(self._messages):
                logger.error(
                    f"Attempted to use incompatible LLMSpecificMessages with LLM '{llm_specific_filter}'."
                )

        if truncate_large_values:
            messages = LLMContext._truncate_large_values_from_messages(messages)

        return messages
```

Three paths, and only one of them copies:

| Call | What comes back |
|---|---|
| `get_messages()` | `self._messages` — **the live list object itself**. Mutating a dict inside it mutates the context. |
| `get_messages("openai")` | A new list, but the **same dict objects** — a shallow filter. Mutating a dict inside it still mutates the context. |
| `get_messages(truncate_large_values=True)` | Deep copies with base64 blobs replaced by placeholders. Safe, and used for logging. |

There is exactly one deep-copying path and it exists for log output, not for safety. If you came from
a codebase where a getter returning internal state is a code smell, this looks like a bug. It is
load-bearing, and §2.4 is the proof.

### 2.4 Why: `_update_function_call_result` rewrites the dicts in place

The assistant aggregator writes the tool half of history in **two stages**, separated in time by
however long the tool takes to run.

**Stage one**, the moment a call starts. Two messages are appended: an assistant message carrying
the `tool_calls` array, and a placeholder `tool` message whose content is the literal string
`"IN_PROGRESS"`.

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1747-1781
    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        logger.debug(
            f"{self} FunctionCallInProgressFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )

        # Update context with the in-progress function call
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments, ensure_ascii=False),
                        },
                        "type": "function",
                    }
                ],
            }
        )

        is_async = not frame.cancel_on_interruption
        if is_async:
            self._context.add_message(async_tool_messages.build_started_message(frame.tool_call_id))
        else:
            self._context.add_message(
                {
                    "role": "tool",
                    "content": "IN_PROGRESS",
                    "tool_call_id": frame.tool_call_id,
                }
            )

        self._function_calls_in_progress[frame.tool_call_id] = frame
```

`ensure_ascii=False` on line 1761 is not decorative, for you specifically: Korean tool arguments
survive as Korean rather than becoming `\uXXXX` escapes in the context the model re-reads.

**Stage two**, whenever the result lands:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1908-1929
    async def _handle_function_call_finished(
        self, frame: FunctionCallResultFrame, in_progress_frame: FunctionCallInProgressFrame
    ):
        """Handle the final result of a function call.

        Removes the call from the in-progress map, updates the context, and
        triggers LLM inference when appropriate.
        """
        is_async = not in_progress_frame.cancel_on_interruption
        del self._function_calls_in_progress[frame.tool_call_id]

        result = json.dumps(frame.result, ensure_ascii=False) if frame.result else "COMPLETED"

        if is_async:
            # For async function calls inject a developer message so the LLM is
            # notified of the completed result instead of updating the IN_PROGRESS
            # tool message.
            self._context.add_message(
                async_tool_messages.build_final_result_message(frame.tool_call_id, result)
            )
        else:
            self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
```

And here is the six-line method the whole design rests on:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:2157-2165
    def _update_function_call_result(self, function_name: str, tool_call_id: str, result: Any):
        for message in self._context.get_messages():
            if (
                not isinstance(message, LLMSpecificMessage)
                and message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result
```

It calls `get_messages()`, walks the dicts, finds the one whose `tool_call_id` matches, and does
`message["content"] = result`. There is no write-back. There is no `set_messages`. The assignment
*is* the write-back — and it only works because `get_messages()` handed over the real dicts inside
the real list.

Note also the sync/async asymmetry, because it is a genuine behavioural fork you would hit in Lina:
a **sync** tool (`cancel_on_interruption=True`, the default) gets its placeholder rewritten; an
**async** tool (`cancel_on_interruption=False`) never gets rewritten at all — its result arrives as a
*new* developer message appended later. Two different history shapes from one flag.

### 2.5 The collision: `deepcopy` would break this silently

boson's `ContextManager` does the opposite, on purpose:

```python
# packages/basement/basement/context/manager.py:47-52
# (boson-agent, private; excerpt-attested via [[llm-service-context]])
def get_messages(self):
    """Return a copy of the messages to prevent external mutation."""
    return deepcopy(self._messages)
```

That is textbook-correct defensive programming, and in boson it is correct: `run_agent_loop` is the
only writer, it writes through `ctx.add_message(...)`, and a copy handed to a provider adapter cannot
be corrupted by it.

Now port it. You are writing a `BosonLLMContext(LLMContext)` and you keep the docstring habit:

```python
class BosonLLMContext(LLMContext):
    def get_messages(self, llm_specific_filter=None, *, truncate_large_values=False):
        """Return a copy of the messages to prevent external mutation."""
        return deepcopy(super().get_messages(llm_specific_filter,
                                             truncate_large_values=truncate_large_values))
```

Trace what happens on a single tool call, step by step:

1. Model emits `tool_calls: [{id: "call_a1", function: {name: "lookup_policy", ...}}]`.
2. `_handle_function_call_in_progress` appends two messages. Context now ends with
   `{"role": "tool", "content": "IN_PROGRESS", "tool_call_id": "call_a1"}`.
3. `lookup_policy` runs, awaits `params.result_callback({"premium": 43000})`.
4. `_handle_function_call_finished` computes `result = '{"premium": 43000}'`.
5. `_update_function_call_result("lookup_policy", "call_a1", '{"premium": 43000}')` iterates over
   **a deep copy**, finds the matching dict **in the copy**, and sets `content` **on the copy**.
6. The copy is discarded when the loop ends. Nothing raises. Nothing logs.
7. `_maybe_push_context_after_function_result` pushes `LLMContextFrame` upstream anyway.
8. The LLM re-runs with a context whose tool message still reads `"IN_PROGRESS"`.

The observable symptom is a bot that says *"잠시만요, 확인 중입니다"* forever, or hallucinates a
premium, or calls `lookup_policy` again — and there is no error anywhere in your logs, because
nothing errored. That is the sharpest single incompatibility between the two codebases, and it is
one line of well-intentioned code.

> **Rule to carry:** in Pipecat, `LLMContext` is a **shared mutable object with multiple concurrent
> writers by design**. Any hardening you add to it must preserve identity of the dicts inside
> `_messages`. If you need a snapshot, take it explicitly at your call site
> (`copy.deepcopy(ctx.get_messages())`) — do not make the getter safe.

### 2.6 One object, two processors: `LLMContextAggregatorPair`

The pair is not a processor. It is a two-line factory whose only real job is to hand the *same*
context object to both halves:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:2290-2314
        user_params = user_params or LLMUserAggregatorParams()
        assistant_params = assistant_params or LLMAssistantAggregatorParams()
        if add_tool_change_messages is not None:
            user_params.add_tool_change_messages = add_tool_change_messages
            assistant_params.add_tool_change_messages = add_tool_change_messages

        self._user = LLMUserAggregator(
            context,
            params=user_params,
            _realtime_service_mode=realtime_service_mode,
        )
        # Wire the assistant→user back-reference unconditionally: realtime mode
        # may be auto-configured later (realtime_service_mode=None), so the
        # reference must already exist when it flips on. ...
        self._assistant = LLMAssistantAggregator(
            context,
            params=assistant_params,
            _realtime_service_mode=realtime_service_mode,
            _paired_user_aggregator=self._user,
        )
```

Note `context` appears twice with no copy between. `__iter__` (`:2332`) is what lets you write
`user_agg, assistant_agg = LLMContextAggregatorPair(context)`.

The **user half** turns a finished transcript into a message and a frame:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:856-873
    async def push_aggregation(self) -> str:
        """Push the current aggregation."""
        if len(self._aggregation) == 0:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()
        self._context.add_message(
            cast(LLMContextMessage, {"role": self.role, "content": aggregation})
        )
        await self.push_context_frame()

        message = UserTurnMessageAddedMessage(
            content=aggregation, timestamp=self._user_turn_start_timestamp
        )
        await self._call_event_handler("on_user_turn_message_added", message)

        return aggregation
```

Two lines of substance: append to the shared context, then push a frame that carries **the context
itself**, not the text:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:481-496
    def _get_context_frame(self) -> LLMContextFrame:
        """Create a context frame with the current context.
        ...
        """
        return LLMContextFrame(context=self._context)

    async def push_context_frame(self, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a context frame in the specified direction.
        ...
        """
        frame = self._get_context_frame()
        await self.push_frame(frame, direction)
```

And the frame is one field:

```python
# src/pipecat/frames/frames.py:550-561
@dataclass
class LLMContextFrame(Frame):
    """Frame containing a universal LLM context.

    Used as a signal to LLM services to ingest the provided context and
    generate a response based on it.

    Parameters:
        context: The LLM context containing messages, tools, and configuration.
    """

    context: LLMContext
```

Recall [[ch-02/read]] §3: `LLMContextFrame` subclasses `Frame` **directly**, in none of the three
branches. It is neither data, nor control, nor system. That is not sloppiness — it is the honest
type for a frame that carries a *reference to shared mutable state* rather than a value. Two
`LLMContextFrame`s in flight at once point at the same object; the frame is a doorbell, not a parcel.

The **assistant half** does the mirror image for the text it heard get spoken:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1677-1694
    async def push_aggregation(self) -> str:
        """Push the current assistant aggregation with timestamp."""
        if not self._aggregation:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()

        self._context.add_message({"role": "assistant", "content": aggregation})

        # Push context frame
        await self.push_context_frame()

        # Push timestamp frame with current time
        timestamp_frame = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(timestamp_frame)

        return aggregation
```

### 2.7 Where the two halves physically sit

```python
# examples/function-calling/function-calling-openai.py:101-117
    context = LLMContext(tools=[get_current_weather, get_restaurant_recommendation])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
```

Read the list positions and nothing else, and you already know three things:

- The user aggregator is **before** the LLM, so its `push_context_frame()` default (DOWNSTREAM)
  reaches the LLM as the *next* processor.
- The assistant aggregator is **after `transport.output()`** — dead last, behind the thing that
  plays audio. That placement is [[ch-08/read]]'s subject (it is why the assistant history contains
  only text that was actually audible), and it is also why the assistant's re-prompt has to travel
  **upstream through four processors** to get back to the LLM.
- `Pipeline.__init__` wraps the list: `self._processors = [self._source, *processors, self._sink]`
  (`src/pipecat/pipeline/pipeline.py:119`), so the assistant aggregator's downstream neighbour is a
  `PipelineSink`, not nothing. [[ch-04/read]] §6.3 taught this; it means a downstream
  `push_context_frame()` from the assistant half goes to the sink and dies quietly.

---

## 3. Pipecat, Q1 continued: the LLM service holds nothing

### 3.1 The base service's `process_frame` touches the context exactly once

```python
# src/pipecat/services/llm_service.py:679-723
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.
        ...
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
        elif isinstance(frame, LLMConfigureOutputFrame):
            self._skip_tts = frame.skip_tts
        elif isinstance(frame, LLMUpdateSettingsFrame):
            ...
        elif isinstance(frame, LLMContextSummaryRequestFrame):
            await self._handle_summary_request(frame)

        if isinstance(frame, LLMContextFrame):
            # Sync the registered handlers with the tools advertised in the
            # context: register any newly advertised handler, drop the ones we
            # auto-registered that are no longer advertised. The context carries
            # the current tool set on every inference, so this is the single place
            # tool changes take effect for text LLMs.
            #
            # Realtime (speech-to-speech) services run continuously and don't get a
            # fresh context frame per turn, so they additionally call
            # _sync_registered_tool_handlers on their own LLMSetToolsFrame
            # handling.
            self._sync_registered_tool_handlers(frame.context.tools)
```

Four `elif` branches about *settings*, and exactly one statement about the context: sync the tool
handlers. No message list is read here. No message list is stored. `LLMService` is 2,220 lines and
none of them keeps a history.

(`LLMContextSummaryRequestFrame` at `:709` is compaction's entry point. Out of scope — see the scope
box. It is on [[ch-13/read]]'s give-back list.)

### 3.2 The concrete service: one frame in, one completion, done

```python
# src/pipecat/services/openai/base_llm.py:590-615
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Handles LLMContextFrame to trigger LLM completions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except httpx.TimeoutException as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)
```

Twenty-six lines. That is the entire "agent loop" on the service side. Four observations, each of
which you can check against the block above:

1. **`:599` is mandatory.** [[ch-01/read]] §7.2 taught why: skip `super().process_frame` and the
   base class never sees `StartFrame`, never starts your process task, and the processor is inert.
   Every quoted `process_frame` in this chapter opens with it.
2. **`:601` does not check direction.** This is the single most consequential omission in the file.
   An `LLMContextFrame` triggers a completion whether it arrived from the user aggregator
   (downstream) or from the assistant aggregator (**upstream**). §5 is built on this line.
3. **The context frame is consumed.** In the `if` branch, `frame` is never re-pushed. Only the
   `else` at `:614` forwards. So an `LLMContextFrame` **stops at the LLM service**, in both
   directions. Nothing after the LLM ever sees one travelling downstream; nothing before the LLM
   ever sees one travelling upstream.
4. **`finally` always pushes `LLMFullResponseEndFrame`** — except when the task is cancelled, which
   is [[ch-08/read]]'s territory and not re-taught here.

And where does `_process_context` end? Not in a loop:

```python
# src/pipecat/services/openai/base_llm.py:563-588
        if function_name:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments or "{}")
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: Failed to parse function call arguments: {arguments}")
                    continue
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)
```

`await self.run_function_calls(function_calls)` and then the function returns. There is no
`while`. There is no `continue`. There is no second call to `_process_context`. The service fires
the tools and goes home.

> **State this to yourself in one sentence before continuing:** *a Pipecat LLM service performs
> exactly one inference per `LLMContextFrame`, and firing tool calls is the last thing it does.*

---

## 4. Pipecat, Q2: who dispatches a tool

[[function-calling]] summarises this whole section in one sentence — *"Pipecat has no multi-turn tool
loop inside the LLM service"* — and the sections below are that sentence taken apart into its
mechanical parts, each re-read against the source. Every line number below was re-measured; the
excerpt and the tree agree, and where the *outline* had drifted from both — the adapter LOC figure,
§4.7 — the tree wins and I show the command.

### 4.1 Three registration routes, one dictionary

Every route ends in `self._functions: dict[str | None, FunctionCallRegistryItem]`
(`llm_service.py:380`). The item is a six-field dataclass:

```python
# src/pipecat/services/llm_service.py:200-205
    function_name: str | None
    handler: FunctionCallHandler | DirectFunctionWrapper
    cancel_on_interruption: bool
    timeout_secs: float | None = None
    cancellable_by_llm: bool = False
    auto_registered: bool = False
```

**Route 1 — a direct function.** A plain async callable whose first parameter is `params` and whose
**docstring is the schema**:

```python
# examples/function-calling/function-calling-openai.py:36-52
async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    await params.result_callback({"conditions": "nice", "temperature": "75"})


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"name": "The Golden Dragon"})
```

The wrapping happens inside `ToolsSchema.__init__`, which is where "a callable in a list" becomes
"a schema plus a wrapper":

```python
# src/pipecat/adapters/schemas/tools_schema.py:54-69
        def _map_standard_tools(tools):
            schemas = []
            direct_functions = []
            for tool in tools:
                if isinstance(tool, FunctionSchema):
                    schemas.append(tool)
                elif callable(tool):
                    wrapper = DirectFunctionWrapper(tool)
                    schemas.append(wrapper.to_function_schema())
                    direct_functions.append(wrapper)
                else:
                    raise TypeError(f"Unsupported tool type: {type(tool)}")
            return schemas, direct_functions

        self._standard_tools, self._direct_functions = _map_standard_tools(standard_tools)
        self._custom_tools = custom_tools
```

and invocation is one line:

```python
# src/pipecat/adapters/schemas/direct_function.py:279-289
    async def invoke(self, args: Mapping[str, Any], params: "FunctionCallParams"):
        """Invoke the wrapped function with the provided arguments.
        ...
        """
        return await self.function(params=params, **args)
```

If you are coming from boson's `@tool` decorator, this is the same trick — docstring as description,
signature as JSON Schema — with the decorator removed. `@tool` raises if the docstring is missing
(`tools/decorator.py:42-45`, per [[boson-tool-router]]); `DirectFunctionWrapper` derives the schema
from the same two sources.

**Route 2 — a `FunctionSchema` carrying a handler.**

```python
# src/pipecat/adapters/schemas/function_schema.py:32-58
    def __init__(
        self,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
        handler: "FunctionCallHandler | None" = None,
    ) -> None:
        """Initialize the function schema.

        Args:
            name: Name of the function to be called.
            description: Description of what the function does.
            properties: Dictionary defining parameter types, descriptions, and constraints.
            required: List of property names that are required parameters.
            handler: Optional handler for this function. When provided, the LLM
                service registers it automatically wherever the schema is
                advertised in the `LLMContext`, making a separate
                ``register_function`` call unnecessary. ...
        """
        self._name = name
        self._description = description
        self._properties = properties
        self._required = required
        self._handler = handler
```

A `FunctionSchema` with `handler=None` is **advertise-only**: the model sees the tool, and calling it
falls through to the catch-all or the missing-handler path. That asymmetry is a hook you will want in
§10.

**Route 3 — explicit `register_function`, including the catch-all.**

```python
# src/pipecat/services/llm_service.py:879-897
    def register_function(
        self,
        function_name: str | None,
        handler: Any,
        *,
        cancel_on_interruption: bool | None = None,
        timeout_secs: float | None = None,
        cancellable_by_llm: bool | None = None,
    ):
        """Register a function handler for LLM function calls.

        Call options resolve with the precedence **explicit argument >
        ``@tool_options`` decorator > default**. ``None`` (the default) means
        "not provided" — the option falls back to the ``@tool_options`` value on
        the handler, then to the documented default.

        Args:
            function_name: The name of the function to handle. Use None to handle
                all function calls with a catch-all handler.
```

`function_name=None` installs a **catch-all**: one handler that receives every call the registry
does not otherwise match. Remember that; it is the entire mechanical basis of the "wrap" resolution
in §10.2.

### 4.2 The registry is re-synced on every context frame

This is the mechanism that makes the context — not the service — the source of truth about tools:

```python
# src/pipecat/services/llm_service.py:1196-1212
        # Register direct functions.
        for wrapper in normalized.direct_functions:
            if wrapper.name in self._functions:
                continue
            if wrapper.name in self._explicitly_unregistered_function_names:
                # Explicitly unregistered while still advertised — leave it gone so
                # calls hit the missing-handler recovery path.
                continue
            self._register_direct_function(wrapper.function)
            # Mark the entry as advertised-tool-set-managed so it can be pruned on a
            # later sync that stops advertising it. Names already in _functions are
            # skipped above, so explicit registrations keep their default
            # auto_registered=False and are never pruned.
            self._functions[wrapper.name].auto_registered = True
            logger.debug(
                f"{self}: auto-registered handler for advertised direct function '{wrapper.name}'"
            )
```

and the pruning half:

```python
# src/pipecat/services/llm_service.py:1339-1354
    def _unregister_unadvertised_tool_handlers(self, advertised: set[str | None]) -> None:
        """Drop auto-registered handlers for tools no longer advertised.

        Only entries with ``auto_registered=True`` are eligible; explicit
        registrations, the catch-all handler, and built-in tools are untouched.
        ...
        """
        stale = [
            name
            for name, item in self._functions.items()
            if item.auto_registered and name not in advertised
        ]
        for name in stale:
            del self._functions[name]
```

Two-tier ownership, and it is worth naming because it is exactly the seam a stage machine wants:

| Registration | `auto_registered` | Pruned when un-advertised? |
|---|---|---|
| Advertised direct function / `FunctionSchema(handler=...)` | `True` | Yes, on the next `LLMContextFrame` |
| `register_function(name, handler)` | `False` | Never |
| Catch-all `register_function(None, handler)` | `False` | Never |

So "what the model can see" (`context.tools`) drives "what the service can run" (`_functions`) —
automatically, per inference, in one direction. That is one gate where boson keeps three. §8.5.

### 4.3 Dispatch: `run_function_calls` → `_run_function_call` → `result_callback`

```python
# src/pipecat/services/llm_service.py:1446-1487
        if len(function_calls) == 0:
            return

        # Exclude the built-in cancel tool — it's an internal mechanism and
        # should not be surfaced to user-facing event handlers or frames.
        user_visible_calls = [
            fc for fc in function_calls if fc.function_name not in self._cancel_tool_names
        ]
        if user_visible_calls:
            await self._call_event_handler("on_function_calls_started", user_visible_calls)
            await self.broadcast_frame(FunctionCallsStartedFrame, function_calls=user_visible_calls)

        # When group_parallel_tools is True all calls share a group_id so the
        # aggregator triggers the LLM exactly once after the last one completes.
        # When False, group_id is None and each result triggers inference independently.
        group_id = str(uuid.uuid4()) if self._group_parallel_tools else None

        runner_items = []
        for function_call in function_calls:
            if function_call.function_name in self._functions.keys():
                item = self._functions[function_call.function_name]
            elif None in self._functions.keys():
                item = self._functions[None]
            else:
                self._log_missing_function_call(function_call.function_name, function_call.context)
                item = self._build_missing_function_call_registry_item(function_call.function_name)

            runner_items.append(
                FunctionCallRunnerItem(
                    registry_item=item,
                    function_name=function_call.function_name,
                    tool_call_id=function_call.tool_call_id,
                    arguments=function_call.arguments,
                    context=function_call.context,
                    group_id=group_id,
                )
            )

        if self._run_in_parallel:
            await self._run_parallel_function_calls(runner_items)
        else:
            await self._run_sequential_function_calls(runner_items)
```

Three-level name resolution — exact name, then the `None` catch-all, then a synthesized
missing-handler item that settles the call with an error message so the turn still terminates.

`broadcast_frame` at `:1456` is worth expanding, because §9.3's turn-cap processor depends on
knowing exactly where that frame goes:

```python
# src/pipecat/processors/frame_processor.py:1038-1054
    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.

        This method creates two instances of the given frame class using the
        provided keyword arguments (without deep-copying them) and pushes them
        upstream and downstream.
        ...
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)
```

**Two instances**, cross-linked by `broadcast_sibling_id`, one each way. Same mechanism
[[ch-08/read]] traced for `InterruptionFrame`. So a `FunctionCallsStartedFrame` is visible to *every*
processor in the pipeline — but any single processor sees exactly **one** of the two, whichever one
travels past it.

Then per call:

```python
# src/pipecat/services/llm_service.py:1547-1583
    async def _run_function_call(self, runner_item: FunctionCallRunnerItem):
        # Re-resolve the registry item at execution time. The function may have
        # been unregistered between queuing and execution, in which case we
        # fall back to the missing-function handler so the call still terminates
        # with a normal tool result.
        if runner_item.function_name in self._functions.keys():
            item = self._functions[runner_item.function_name]
        elif None in self._functions.keys():
            item = self._functions[None]
        ...
        # Broadcast function call in-progress. This frame will let our assistant
        # context aggregator know that we are in the middle of a function
        # call. Some contexts/aggregators may not need this. But some definitely
        # do (Anthropic, for example).
        await self.broadcast_frame(
            FunctionCallInProgressFrame,
            function_name=runner_item.function_name,
            tool_call_id=runner_item.tool_call_id,
            arguments=runner_item.arguments,
            cancel_on_interruption=item.cancel_on_interruption,
            group_id=runner_item.group_id,
        )
```

and the actual invocation, with the two handler shapes:

```python
# src/pipecat/services/llm_service.py:1650-1683
        if item.timeout_secs or self._function_call_timeout_secs:
            timeout_task = self.create_task(timeout_handler())

        try:
            if isinstance(item.handler, DirectFunctionWrapper):
                # Handler is a DirectFunctionWrapper
                await item.handler.invoke(
                    args=runner_item.arguments,
                    params=FunctionCallParams(
                        function_name=runner_item.function_name,
                        tool_call_id=runner_item.tool_call_id,
                        arguments=runner_item.arguments,
                        llm=self,
                        pipeline_worker=self.pipeline_worker,
                        context=runner_item.context,
                        result_callback=function_call_result_callback,
                        app_resources=self.pipeline_worker.app_resources,
                        worker_runner=self.pipeline_worker.worker_runner,
                    ),
                )
            else:
                # Handler is a FunctionCallHandler
                params = FunctionCallParams(
                    ...
                )
                await item.handler(params)
```

### 4.4 The handler contract: never return a value

```python
# src/pipecat/services/llm_service.py:142-155
    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    # `LLMService[Any]` so any concrete subclass (regardless of how — or
    # whether — it parameterizes the adapter type) can be assigned here.
    # Plain `LLMService` would invoke the TypeVar default and pyright would
    # treat it invariantly, rejecting `LLMService[XAdapter]` at the call
    # sites that build FunctionCallParams.
    llm: LLMService[Any]
    pipeline_worker: PipelineWorker
    context: LLMContext
    result_callback: FunctionCallResultCallback
    app_resources: Any = None
    worker_runner: WorkerRunner | None = None
```

A Pipecat tool handler takes **one** argument and returns **nothing**. It reports by awaiting
`params.result_callback(result)`. If it returns a value instead, the value is discarded and the call
never settles — it hangs until `function_call_timeout_secs` fires, if you set one, and forever if you
did not.

Compare the boson contract, per [[boson-tool-router]]:

```python
# packages/basement/basement/tools/executor.py:67 — _invoke_handler
# (boson-agent, private; excerpt-attested via [[boson-tool-router]])
#   async handlers:  await spec.handler(**arguments)
#   return contract: a returned ToolResultBlock passes through unchanged;
#                    anything else is str()-ified into ToolResultBlock(...)
#                    exceptions become "Tool error: {type}: {msg}" with is_error=True
```

Two differences, both mechanical, both affecting every tool you own:

| | boson | Pipecat |
|---|---|---|
| Signature | `handler(**arguments)` — kwargs sprayed | `handler(params: FunctionCallParams)` — one object, args inside `params.arguments` |
| Result | `return value` | `await params.result_callback(value)` |

[[boson-tool-router]] counts **22 tools under `agents/*/tools/`**. Each needs a signature change or a
shim. That is a number for §10.1.

Notice also what `FunctionCallParams` gives a handler that boson's `handler(**arguments)` does not:
`params.context` — the *live* `LLMContext`. A Pipecat tool handler can read and rewrite conversation
history from inside the tool. That is either a very useful hook or a footgun, depending on your
discipline; it has no boson equivalent, because boson tools receive only their own arguments.

### 4.5 Both function-call frames are uninterruptible, on purpose

```python
# src/pipecat/frames/frames.py:769-791
@dataclass
class FunctionCallResultFrame(DataFrame, UninterruptibleFrame):
    """Frame containing the result of an LLM function call.

    This is an uninterruptible frame because once a result is generated we
    always want to update the context.

    Parameters:
        function_name: Name of the function that was executed.
        tool_call_id: Unique identifier for the function call.
        arguments: Arguments that were passed to the function.
        result: The result returned by the function.
        run_llm: Whether to run the LLM after this result.
        properties: Additional properties for result handling.

    """

    function_name: str
    tool_call_id: str
    arguments: Any
    result: Any
    run_llm: bool | None = None
    properties: FunctionCallResultProperties | None = None
```

```python
# src/pipecat/frames/frames.py:2163-2187
@dataclass
class FunctionCallInProgressFrame(ControlFrame, UninterruptibleFrame):
    """Frame signaling that a function call is currently executing.

    This is an uninterruptible frame because we always want to update the
    context.
    ...
        group_id: Identifier shared by all function calls originating from the
            same LLM response batch. Used to determine when the last call in a
            group completes so the LLM can be triggered exactly once.
    """
```

The mixin's own docstring says what it buys:

```python
# src/pipecat/frames/frames.py:146-157
@dataclass
class UninterruptibleFrame:
    """A marker for data or control frames that must not be interrupted.

    Frames with this mixin are still ordered normally, but unlike other frames,
    they are preserved during interruptions: they remain in internal queues and
    any task processing them will not be cancelled. This ensures the frame is
    always delivered and processed to completion.

    """

    pass
```

Read the pair together: a tool call that has started **will** be recorded in history, and a tool
result that has been produced **will** land in history, even if the customer barges in mid-call.
That is exactly the invariant boson spends ~80 lines maintaining by hand —
`_reconcile_cancelled_tool_uses` (`agent_loop.py:56`) and `_await_tool_boundary` (`:113`), which pair
every unanswered `tool_use` with a synthetic `ToolResultBlock(content=f"canceled: {tu['name']}", is_error=True)` before re-raising, per [[boson-agent-loop]]. Pipecat gets the same invariant from a
mixin on a dataclass plus a scheduling rule. [[ch-08/read]] owns the cascade; here the only point is
**who wrote the code**: a frame flag versus a reconciliation routine.

### 4.6 The only bounds are per-call timeouts

```python
# src/pipecat/services/llm_service.py:1632-1648
        # Start a timeout task for deferred function calls
        async def timeout_handler():
            effective_timeout = item.timeout_secs or self._function_call_timeout_secs
            # This task is only started when one of the two is set.
            assert effective_timeout is not None
            await asyncio.sleep(effective_timeout)
            logger.warning(
                f"{self} Function call [{runner_item.function_name}:{runner_item.tool_call_id}] timed out after {effective_timeout} seconds and is being cancelled."
                f" You can increase this timeout by passing `timeout_secs` to `register_function()`,"
                f" or set a global default via `function_call_timeout_secs` on the LLM constructor."
            )
            # Settle the call before cancelling, so a result racing in while the
            # handler unwinds is rejected instead of broadcast.
            runner_item.settled = True
            # Cancelling goes through a detached task because it awaits the
            # function call task this handler runs inside of.
            self.create_task(self._timeout_function_call(runner_item))
```

Both bounds are **per call**: `timeout_secs` on one tool, `function_call_timeout_secs` on the
service. Neither bounds *how many calls happen*. Hold that until §6.

### 4.7 Per-provider tool shaping lives behind one abstract method

```python
# src/pipecat/adapters/base_llm_adapter.py:113-123
    @abstractmethod
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[Any]:
        """Convert tools schema to the provider's specific format.

        Args:
            tools_schema: The standardized tools schema to convert.

        Returns:
            List of tools in the provider's expected format.
        """
        pass
```

Two implementations, so you can see the shape difference that boson hand-writes six times:

```python
# src/pipecat/adapters/services/open_ai_adapter.py:206-220
        functions_schema = tools_schema.standard_tools
        # `function=...` expects a `FunctionDefinition` TypedDict; the dict
        # produced by `to_default_dict()` is structurally compatible. Cast at
        # the boundary.
        formatted_standard_tools: list[ChatCompletionToolParam] = [
            ChatCompletionToolParam(type="function", function=cast(Any, func.to_default_dict()))
            for func in functions_schema
        ]
        custom_openai_tools: list[ChatCompletionToolParam] = []
        if tools_schema.custom_tools:
            custom_openai_tools = cast(
                list[ChatCompletionToolParam],
                tools_schema.custom_tools.get(AdapterType.OPENAI, []),
            )
        return formatted_standard_tools + custom_openai_tools
```

```python
# src/pipecat/adapters/services/anthropic_adapter.py:499-500
        functions_schema = tools_schema.standard_tools
        return [self._to_anthropic_function_format(func) for func in functions_schema]
```

**Attach the number to the right path**, because the outline got this wrong once and it is the kind
of error that survives review:

```console
$ find src/pipecat/adapters/services -name "*.py" | xargs wc -l | tail -1
    3647 total
$ find src/pipecat/adapters/services -name "*.py" | wc -l
13          # 12 concrete adapter modules + an empty __init__.py

$ find src/pipecat/adapters -name "*.py" | xargs wc -l | tail -1
    4549 total
$ find src/pipecat/adapters -name "*.py" | wc -l
19
```

**3,647 L across 12 provider adapter modules in `adapters/services/`. 4,549 L across 19 files for
all of `adapters/`.** The extra ~900 lines are the base adapter, the schema layer
(`function_schema.py`, `tools_schema.py`, `direct_function.py`), and the registry. Quote whichever
you mean and say which one you meant.

The 12 modules, largest first: `gemini_adapter.py` 802, `anthropic_adapter.py` 500,
`bedrock_adapter.py` 375, `open_ai_responses_adapter.py` 294, `open_ai_adapter.py` 269,
`grok_realtime_adapter.py` 265, `inworld_realtime_adapter.py` 261,
`open_ai_realtime_adapter.py` 244, `aws_nova_sonic_adapter.py` 240, `perplexity_adapter.py` 173,
`mistral_adapter.py` 135, `gemini_live_adapter.py` 89.

Against that, per [[boson-tool-router]], boson has **6 provider factories** in `PROVIDER_REGISTRY`
(`llm/registry.py:61-68`): `anthropic`, `openai`, `google`, `boson`, `xai`, `openrouter` — with the
last three being OpenAI-compatible subclasses. Each hand-writes its own tool shaping: Anthropic's
`{"name","description","input_schema"}` versus OpenAI's
`{"type":"function","function":{"name","description","parameters"}}` — `input_schema` versus
`parameters`, which is precisely the difference `to_provider_tools_format` exists to absorb.

---

## 5. Pipecat, Q3: the loop closes because a frame goes backwards

You now have every piece except the one that makes it a loop. Here it is.

### 5.1 The upstream push

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1859-1889
    async def _maybe_push_context_after_function_result(self) -> None:
        """Decide whether to push a context frame after a function call settles.

        Push an ``LLMContextFrame`` upstream (with care to avoid duplicate
        pushes while results are queued or the bot is still speaking).
        Cascade LLMs use the context frame to re-run inference with the
        new tool result in scope. Realtime LLMs read the new tool result
        out of the context the same way — they don't get function results
        from ``FunctionCallResultFrame`` directly — so the same push is
        load-bearing for both modes.
        """
        if self.has_queued_frame(FunctionCallResultFrame):
            # Another FunctionCallResultFrame is already queued. Defer the context push
            # to bundle all results into a single LLM call instead of triggering one
            # inference pass per result. The context will be pushed once the last
            # function call in the queue is processed.
            logger.debug(
                f"{self}: More FunctionCallResultFrames queued — deferring context frame push."
            )
        elif self._bot_speaking:
            # Defer the context frame push until the bot finishes speaking. If multiple
            # function call results arrive while the bot is speaking, they all accumulate
            # in the context and a single push is performed once speaking stops, preventing
            # the LLM from running multiple times and producing duplicated responses.
            # This should be an edge case, since it would require a FunctionCallResultFrame
            # being queued between an LLM response start and end frame.
            logger.debug(f"{self}: Bot is speaking — deferring context frame push.")
            self._push_context_on_bot_stopped_speaking = True
        else:
            logger.debug(f"{self}: Pushing context frame!")
            await self.push_context_frame(FrameDirection.UPSTREAM)
```

`FrameDirection.UPSTREAM`, from the **last processor in the pipeline**, carrying a reference to the
context the LLM will read. That single line at `:1889` is the entire tool loop.

### 5.2 The gate in front of it

Before that method is even called, `_handle_function_call_result` decides whether a re-prompt is
warranted at all:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1811-1848
        run_llm = False

        # Append any images that were generated by function calls.
        if frame.tool_call_id in self._function_calls_image_results:
            image_frame = self._function_calls_image_results[frame.tool_call_id]

            del self._function_calls_image_results[frame.tool_call_id]

            # If an image frame has been added to the context, let's run inference.
            run_llm = await self._maybe_append_image_to_context(image_frame)

        # Run inference if the function call result requires it.
        if frame.result:
            if properties and properties.run_llm is not None:
                # If the tool call result has a run_llm property, use it.
                run_llm = properties.run_llm
            elif frame.run_llm is not None:
                # If the frame is indicating we should run the LLM, do it.
                run_llm = frame.run_llm
            else:
                # Run the LLM when this is the last function call in the group
                # to complete. If group_id is set, only consider sibling calls;
                # otherwise always execute as soon as we receive the result.
                if group_id:
                    run_llm = not any(
                        f is not None
                        and f.group_id == group_id
                        # We are now able to receive "updates", so the current
                        # frame can still be in the in progress list, and we need to
                        # ignore it.
                        and f.tool_call_id != frame.tool_call_id
                        for f in self._function_calls_in_progress.values()
                    )
                else:
                    run_llm = True

        if run_llm and not self._user_speaking:
            await self._maybe_push_context_after_function_result()
```

So the re-prompt decision is a four-way precedence:

1. `properties.run_llm` set by the handler → obey it.
2. `frame.run_llm` set on the result frame → obey it.
3. `group_id` set → re-prompt only when this is the last sibling of the batch still outstanding.
4. Otherwise → re-prompt immediately.

Plus two suppressors: `self._user_speaking` (do not re-prompt over a talking customer) and
`self._bot_speaking` (defer until the bot's current utterance ends).

A handler can therefore end the turn from inside itself by settling with
`FunctionCallResultProperties(run_llm=False)` — no re-prompt, no reply, silence. That is your hook
for a tool like `end_call` or `transfer_to_human`, and it is the closest thing Pipecat has to a
deliberate "the turn is over" statement. It is still not a turn *cap*; it is one tool declining to
continue.

### 5.3 Trace the cycle as a walk on the linked list

Take the canonical pipeline from §2.7 and follow one tool-using turn. Every hop is a real
`push_frame` you can find above.

```
transport.input  →  stt  →  user_agg  →  llm  →  tts  →  transport.output  →  assistant_agg
                                          ↑                                        │
                                          └────────────── UPSTREAM ────────────────┘
```

| # | Where | What happens | Line |
|---|---|---|---|
| 1 | `user_agg` | turn ends; `add_message({"role":"user", ...})`; `push_context_frame()` DOWNSTREAM | `:863-866` |
| 2 | `llm` | `isinstance(frame, LLMContextFrame)` → push `LLMFullResponseStartFrame`, start metrics, `_process_context()` | `:601-605` |
| 3 | `llm` | model emits `tool_calls`; build `FunctionCallFromLLM`s; `run_function_calls()` | `base_llm.py:579-588` |
| 4 | `llm` | broadcast `FunctionCallsStartedFrame` **both ways** | `llm_service.py:1456` |
| 5 | `llm` | per call: broadcast `FunctionCallInProgressFrame` both ways | `llm_service.py:1576-1583` |
| 6 | `assistant_agg` | sees in-progress frame → appends `tool_calls` message + `"IN_PROGRESS"` placeholder | `:1753-1779` |
| 7 | `llm` | `finally:` push `LLMFullResponseEndFrame`. **`process_frame` returns. The service is now idle.** | `base_llm.py:611-613` |
| 8 | handler | `await params.result_callback({...})` → broadcast `FunctionCallResultFrame` | `llm_service.py:1589+` |
| 9 | `assistant_agg` | rewrites the placeholder **in place** | `:2157-2165` |
| 10 | `assistant_agg` | `run_llm` resolves True → `push_context_frame(UPSTREAM)` | `:1847-1889` |
| 11 | `transport.output` → `tts` → `llm` | frame walks upstream three hops | `frame_processor.py:1183-1194` |
| 12 | `llm` | `isinstance(frame, LLMContextFrame)` — **direction not checked** → another completion | `base_llm.py:601` |
| 13 | `llm` | model replies in text; `LLMTextFrame`s go downstream to `tts` | — |
| 14 | `assistant_agg` | `push_aggregation()` appends `{"role":"assistant","content": ...}` | `:1685` |
| 15 | — | no `FunctionCallResultFrame`, so nothing is pushed upstream. **The turn ends by absence.** | — |

Step 12 is the hinge. Step 15 is the answer to Q3.

> **Q3, Pipecat's answer, in one sentence:** *nothing decides the turn ended; the turn ends when no
> processor pushes an `LLMContextFrame` at the LLM, and the only thing that ever pushes one upstream
> is a settled tool result.*

There is no `break`. There is no state machine. There is no terminal event — compare
realtime_voice, which enforces **exactly one terminal event per generation** with
`self._terminal: dict[GenerationId, VoiceEventKind]` and `_terminal_event()` at
`pipeline/session.py:516`, per [[rtv-pipeline-session]]. Pipecat's turn boundary is not represented
in the system at all.

### 5.4 Use the figure here

→ **[Open the ch-09 two-loops viewer](./figures/two-loops.html)** and step both columns through one
tool-using turn.

Do these three things with it, in order, and do not skim the third:

1. **Step the Pipecat column to the in-place rewrite.** Watch the shared message list flash when
   `_update_function_call_result` assigns `message["content"] = result`, and watch the boson column's
   `deepcopy` fork the list at the same instant. That flash is §2.5 — the failure that produces no
   error.
2. **Toggle the runaway-tool counter.** Let the model keep calling tools. Pipecat's counter climbs
   with no ceiling because there is none; boson's stops at 50 and emits
   `MessageEnd(stop_reason="max_turns")`. This is §6.
3. **Switch the resolution toggle through adopt / wrap / bypass** and read what each one breaks. The
   figure marks none of them as recommended — it shows three cost columns. That is the same
   discipline as §10 and it is deliberate.

---

## 6. Pipecat, the absence: no tool loop, no turn cap

Two greps, both re-run against the tree at commit `0cbf9c5b` on 2026-08-25.

```console
$ cd wiki/raw-data/pipecat/pipecat-src
$ grep -rn "max_turns\|max_iterations\|max_tool_calls" src/pipecat/ | wc -l
0
```

Zero hits. Not a different name — the concept is absent. Search `src/pipecat/` for anything that
counts inferences within a turn and you will find the same nothing. [[function-calling]] states it
in the same words: *"Think-act-observe is a topology, not a `while` block — and consequently it has
**no turn cap**."*

Now put that next to §5.3 step 12 and imagine the failure mode concretely, in Lina's domain:

> The customer asks *"제 보험료가 얼마죠?"*. The model calls `lookup_policy`. The tool returns
> `{"error": "policy_id required"}`. The model, seeing an error, calls `lookup_policy` again with a
> guessed id. Same error. Again. Again.

Each cycle is: result frame → placeholder rewrite → `run_llm=True` → upstream `LLMContextFrame` →
completion → `run_function_calls` → result frame. Nothing in that cycle counts. Nothing in it stops.
Your bounds are:

| Bound | Where | What it actually limits |
|---|---|---|
| `timeout_secs` | per tool, `register_function(..., timeout_secs=...)` | how long **one** call may run |
| `function_call_timeout_secs` | `LLMService.__init__` `:307` | how long **any** call may run |
| `FunctionCallResultProperties(run_llm=False)` | per result, set by the handler | whether **this one** result re-prompts |
| — | — | **nothing limits how many cycles happen** |

Each cycle costs one inference and, in a voice call, keeps the customer listening to silence or
filler. At 700 ms per cycle a hundred cycles is over a minute of dead air on a live phone call, and
the only thing that ends it is the customer hanging up or a `PipelineWorker` idle timeout
([[ch-04/read]] §9) firing.

Against that, boson bounds it in two lines:

```python
# packages/basement/basement/loop/agent_loop.py:207-209
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
turn_count = 0
while turn_count < runtime.config.max_turns:
    ...
```

with `max_turns: int = Field(default=50, ge=1, le=1000)` (`schemas/config_schema.py:62`) and an
explicit exhaustion path on the `while/else` at `:365`, which yields
`TextDelta(text="\n[Max turns exceeded — stopping]")` and `MessageEnd(stop_reason="max_turns")`.

> **Say it plainly, because softening it would be a disservice:** boson's `max_turns` guard and its
> `while/else` exhaustion path **have no counterpart in Pipecat**. They cannot be ported. They must
> be rebuilt as a counting `FrameProcessor`, or consciously dropped with the risk written down. §9.3
> builds the processor.

---

## 7. Interlude: what Pipecat's ownership model buys

Two paragraphs, no code, and then straight back to the collision. This is here because a chapter that
only lists absences is a biased chapter.

Splitting ownership three ways — context in a plain object, dispatch in the service, loop closure in
the topology — makes each piece independently replaceable, which is [[ch-01/read]]'s substitutability
claim cashed at the LLM layer. You can swap `OpenAILLMService` for `AnthropicLLMService` without
touching the context, because the context knows nothing about providers. You can gate the loop
without touching either, by splicing a processor: `GatedLLMContextAggregator`
(`gated_llm_context.py:14`) is 82 lines that hold the **latest** `LLMContextFrame` and release it
when a notifier fires, dropping stale ones rather than queueing them — a hold-the-turn primitive
that exists only because the loop runs through a splice point instead of through a call stack.

And you can observe the whole loop without instrumenting any of it. Every step in §5.3's table is a
frame crossing a processor boundary, which means [[ch-11/read]]'s observer plane sees it for free.
In boson, the equivalent visibility is the six hook events fired by hand from inside
`run_agent_loop` (`ON_TURN_START`, `PRE_LLM_CALL`, `PRE_TOOL_CALL`, `POST_TOOL_CALL`/`ON_ERROR`,
`POST_LLM_CALL`, `ON_TURN_END`, per [[boson-agent-loop]]) — a fixed set, chosen at authoring time,
extensible only by editing the loop.

That is what the topology buys. §8 is what it costs against your existing code.

---

## 8. boson's answer: one function that owns all three

### 8.1 The entry point and the three aliases

```python
# packages/basement/basement/loop/agent_loop.py:176, 184-186
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
async def run_agent_loop(runtime: AgentRuntime, user_input: str) -> AsyncIterator[StreamEvent]:
    ...
    ctx = runtime.context_manager
    api = runtime.conversation_api
    hooks = runtime.hook_registry
```

Three locals alias the whole world. `AgentRuntime` (`schemas/runtime.py:18`) is a dataclass of 15
fields including `provider`, `tool_registry`, `tool_router`, `permissions`, `cancellation_flag`,
`skip_user_append`, `exposed_meta_tools`, `on_tool_start`, `on_tool_end`.

Compare the Pipecat side: `LLMContext.__init__` takes three arguments and stores three fields. The
asymmetry is the chapter in miniature — boson's turn has one object that reaches everything; Pipecat's
turn has one object that reaches nothing and a graph that reaches everything.

### 8.2 The single inference, and the two branches after it

```python
# packages/basement/basement/loop/agent_loop.py:251-255
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
async for event in runtime.provider.stream(
    messages=ctx.get_messages(),
    system=ctx.get_system_prompt(),
    tools=tools,
):
```

`ctx.get_messages()` is the `deepcopy` from §2.5, handed to the provider. `ctx.get_system_prompt()`
is a **separate accessor** — and per [[llm-service-context]] it has no `LLMContext` equivalent at
all. Pipecat's system prompt is either `Service.Settings(system_instruction=...)` or a leading
`{"role": "system"}` message that adapters pull out via `_extract_initial_system`
(`src/pipecat/adapters/base_llm_adapter.py:208`, called from the anthropic, bedrock and gemini
adapters). So boson's clean split of *system* from *messages* becomes, on the Pipecat side, either a
service setting or a message the adapter has to re-extract.

Then, per [[boson-agent-loop]], two branches decide the turn with no arbiter between them:

```python
# packages/basement/basement/loop/agent_loop.py:293, 347, 355-363
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
if tool_uses:
    ...                      # build the assistant tool_use message, execute the batch
    continue                 # :347 — back to the LLM
else:
    ctx.add_message("assistant", text)   # :355
    ...                                  # fire POST_LLM_CALL
    break  # Done — text response means end of turn      # :363
```

plus a third exit, cooperative cancellation after a tool batch:
`if cancellation_flag is not None and cancellation_flag.is_set: break` (`:343-345`).

> **Q3, boson's answer, in one sentence:** *`run_agent_loop` decides, at `agent_loop.py:363`, and the
> rule is "a text-only response means the turn is over."*

Both frameworks reach the same behaviour — a text-only reply ends the turn. Pipecat reaches it
because a text-only reply produces no `FunctionCallResultFrame` and therefore nobody pushes anything
upstream. boson reaches it because a `break` statement says so. The behaviour matches; the *locus of
the decision* does not, and that is the thing you are porting.

### 8.3 The tool array is recomputed every iteration

```python
# packages/basement/basement/loop/agent_loop.py:224-237
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
if runtime.tool_router:
    from basement.metatool.registry import meta_tool_names
    exposed = (runtime.exposed_meta_tools
               if runtime.exposed_meta_tools is not None else meta_tool_names())
    tools = [s for s in runtime.tool_registry.get_all_specs() if s.name in exposed] or None
else:
    tools = runtime.tool_registry.get_all_specs() or None
```

Inside the `while`, so on **every** iteration. Pipecat's structural counterpart is
`_sync_registered_tool_handlers(frame.context.tools)` at `llm_service.py:723` — also on every
inference, because every inference arrives on a context frame. Same cadence, opposite direction:
boson recomputes what the model **sees** from a registry; Pipecat recomputes what the service can
**run** from what the model sees.

To change the advertised set mid-session in Pipecat you push an `LLMSetToolsFrame`, which the user
aggregator handles:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:822-834
        elif isinstance(frame, LLMSetToolsFrame):
            # Normalize and validate (a plain list of direct functions / FunctionSchema
            # objects becomes a ToolsSchema) so the tool-change diff and
            # set_tools see a consistent type.
            normalized_tools = LLMContext._normalize_and_validate_tools(frame.tools)
            self._maybe_add_tool_change_messages(normalized_tools)
            self.set_tools(normalized_tools)
            # Push the LLMSetToolsFrame as well, since speech-to-speech LLM
            # services (like OpenAI Realtime) may need to know about tool
            # changes; unlike text-based LLM services they won't just "pick up
            # the change" on the next LLM run, as the LLM is continuously
            # running.
            await self.push_frame(frame, direction)
```

`_maybe_add_tool_change_messages` is the `add_tool_change_messages=True` feature from
`LLMContextAggregatorPair` (`:2260`, `:2292-2294`): when the tool set changes mid-conversation, an
announcement message is added so the model does not keep calling a tool that has vanished. That is
Pipecat's built-in answer to the same hallucination problem the `use_tool` indirection was invented
to dodge — the pair-level flag exists precisely so both halves participate and the shared context
guarantees the announcement lands exactly once.

### 8.4 The message schema collision: there is no `tool` role in boson

Per [[boson-agent-loop]]:

```python
# packages/basement/basement/schemas/message_schema.py:46
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
role: Literal["user", "assistant"]
```

Two roles. That is the whole enum. Tool results ride **inside a user message** as content blocks:

```python
# packages/basement/basement/loop/agent_loop.py:540
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
ctx.add_message("user", [result])     # result is a ToolResultBlock(tool_use_id, content, is_error)
```

and the assistant tool-use message is assembled by hand at `:316-328`: strip echoed
`<system-reminder>` blocks with `re.sub`, add a `TextBlock(text=text)` if there was text, then one
`ToolUseBlock(id, name, input)` per call.

That is Anthropic's message shape. Pipecat's is OpenAI's. Put one tool call side by side and count
what has to change:

```python
# boson — Anthropic-shaped, two roles, blocks inside content
[
  {"role": "assistant", "content": [
      {"type": "text",     "text": "확인해 보겠습니다"},
      {"type": "tool_use", "id": "toolu_01", "name": "lookup_policy",
       "input": {"policy_id": "P-9931"}},
  ]},
  {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "toolu_01",
       "content": "{\"premium\": 43000}", "is_error": False},
  ]},
]
```

```python
# Pipecat — OpenAI-shaped, three roles, tool_calls array + a separate tool message
[
  {"role": "assistant", "content": "확인해 보겠습니다"},        # push_aggregation, :1685
  {"role": "assistant", "tool_calls": [                          # _handle_..._in_progress, :1753
      {"id": "call_01", "type": "function",
       "function": {"name": "lookup_policy",
                    "arguments": "{\"policy_id\": \"P-9931\"}"}},
  ]},
  {"role": "tool", "content": "{\"premium\": 43000}",            # placeholder, then rewritten
   "tool_call_id": "call_01"},                                   # in place at :2165
]
```

Differences that are not renames:

| | boson | Pipecat |
|---|---|---|
| Roles | 2 (`user`, `assistant`) | 3+ (`user`, `assistant`, `tool`, `system`, developer messages) |
| Tool result carrier | a `user` message containing `ToolResultBlock` | a `tool` message with `tool_call_id` |
| Tool call carrier | `ToolUseBlock` inside assistant `content` | `tool_calls` array **beside** `content` |
| Arguments | a dict (`input`) | a **JSON string** (`arguments`) |
| Text + tool call | one assistant message with both blocks | **two** assistant messages |
| Error signalling | `is_error=True` on the block | no field — the string content is all there is |
| Validation | pydantic `Message` / `ContentBlock` | none; plain dicts |

The last row of the table is the one that bites during a port: there is no place to put `is_error`.
Pipecat's error convention is the string itself —
`FUNCTION_CALL_ERROR_MESSAGE_TEMPLATE = "The function \`{function_name}\` failed and returned no result."` (`llm_service.py:299-301`). Any boson tool that relies on `is_error` for downstream branching
loses that signal unless you encode it into the result payload yourself.

And note the strict-alternation constraint boson maintains deliberately, from
`_check_cancellation_and_emit` (`agent_loop.py:139-156`): cancel entries are **merged into the same
user message** as the tool results, *"mirroring the merged-blocks shape of
`InterruptHandler.handle_barge_in`, which preserves strict assistant→user role alternation (some
models reject consecutive same-role messages)"*. Pipecat's shape emits two consecutive `assistant`
messages as a matter of course (text, then `tool_calls`). Under an OpenAI-family model that is fine.
Under a strict-alternation model it is not — and boson's provider list includes `anthropic` first.

> **Say it plainly:** porting boson's history to `LLMContext` is a **schema rewrite**, not a rename.
> The `Message` / `ContentBlock` pydantic layer has no landing site — `LLMContext` stores plain
> dicts and pushes provider shaping down into the adapters.

### 8.5 Three gates, and Pipecat has slots for one

This is the second-largest structural absence in the chapter, after the turn cap. From
[[boson-tool-router]], boson keeps three independent decisions apart:

| Gate | Where it lives | What it decides | Why it is separate |
|---|---|---|---|
| **Exposure** | `agent_loop.py:224-237`, pinned at `gateway/core.py:280-303` | what the model **sees** in the tools array | pinned to `{"use_tool","use_skill"}` under `maximize_caching`, because *"the prompt-cache prefix is ordered tools → system → messages, so changing exposure on a stage transition invalidates the entire cached conversation"* |
| **Availability** | `_allowed_tools_var: ContextVar[set[str] | None]` (`metatool/router.py:32`), checked inside `dispatch` | what may **run in the current stage** | a `ContextVar` is per-asyncio-task, so one router instance serves concurrent sessions |
| **Permission** | `PermissionChecker.check_tool(tool_name)`, first thing in `dispatch` (`router.py:103`) | what this **caller** is allowed to run at all | raises `PermissionDeniedError`, independent of stage |

The dispatch order is, verbatim from [[boson-tool-router]]: (1) permission check → raises; (2) stage
gate — if `_allowed_tools_var.get()` is not `None` and the name is neither in it nor a meta-tool,
return an `is_error=True` `ToolResultBlock` with content
`f"Tool '{tool_name}' is not available in the current stage."`; (3) `ToolNotFoundError` if absent;
(4) `return await _invoke_handler(spec, arguments)`.

The exposure gate is why `use_tool` exists at all. Its advertised schema is deliberately generic:

```json
{"type":"object",
 "properties":{"tool_name":{"type":"string","description":"Name of the tool to call"},
               "arguments":{"type":"object","description":"Arguments to pass to the tool"}},
 "required":["tool_name","arguments"]}
```

Two entries in the tools array, byte-stable across every stage transition, so the prompt-cache prefix
never invalidates — while the allowlist underneath changes freely.

Now look at Pipecat's registry item again (`llm_service.py:200-205`): `function_name`, `handler`,
`cancel_on_interruption`, `timeout_secs`, `cancellable_by_llm`, `auto_registered`. **No permission
field. No allowlist field. No caller identity.** And `_run_function_call` (`:1552-1566`) resolves a
name to a handler and invokes it — there is no interception point between resolution and invocation.

Pipecat's one gate is exposure, and it is fused to registration: what the context advertises is what
the registry holds (§4.2). Availability and permission have to be re-created, and there are exactly
two places to put them:

1. **Inside each handler** — 22 copies of the same guard, or one decorator applied 22 times.
2. **In a catch-all** — `register_function(None, guard_handler)`, which receives every call, checks
   permission and the allowlist, and then dispatches to your own table. This is §10.2, and it is
   structurally the same move as `use_tool`.

Note that the `use_tool` indirection itself survives a port unchanged: it becomes
`LLMContext(tools=[use_tool, use_skill])` — a two-entry advertised set with the real dispatch table
behind it. The meta-tool trick is portable. The three-gate separation is what has no slot.

One more finding, reported because absence is evidence and [[boson-tool-router]] flagged it:
`register_meta_tool(name)` exists in `metatool/registry.py` and `grep -rn "register_meta_tool"` finds
**no call site anywhere in the boson repo**. The extensibility point was built and never used. Do
not port dead extension points into a new framework; port the two that are live.

### 8.6 Sequential versus parallel, which is a correctness issue for you

```python
# packages/basement/basement/loop/agent_loop.py:393
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
async def _execute_tool_uses(runtime, tool_uses, hooks, api, ctx):
    for idx, tu in enumerate(tool_uses):
        ...
```

A plain `for`. Strictly sequential, no parallelism, per [[boson-agent-loop]]. And per
[[boson-tool-router]], `tools/executor.py:30` holds a module-level `_SYNC_HANDLER_LOCK = threading.Lock()` because *"production tools do read-modify-write on shared YAML/JSON files."*

Pipecat's default:

```python
# src/pipecat/services/llm_service.py:303-308
    def __init__(
        self,
        run_in_parallel: bool = True,
        group_parallel_tools: bool = True,
        function_call_timeout_secs: float | None = None,
        enable_async_tool_cancellation: bool = False,
        settings: LLMSettings | None = None,
        **kwargs,
    ):
```

`run_in_parallel=True`. If the model emits two tool calls in one response,
`_run_parallel_function_calls` (`:1534`) creates a task per call and they interleave at every
`await`. Port 22 read-modify-write tools onto that default and you have a data race the day a model
decides to call two of them at once.

The fix is one keyword — `OpenAILLMService(..., run_in_parallel=False)` — which routes through
`_run_sequential_function_calls` (`:1542`), a queue drained by a single runner task. It is one word,
and it is one word that has to be **re-asserted**, because the default is the opposite of what your
tools assume. Write it on the migration checklist, not in a comment.

---

## 9. The four collisions, priced individually

Everything above compresses to four. Each gets a cost in the unit that actually matters: **how much
new code, and what breaks if you skip it.**

### 9.1 Live list versus `deepcopy`

- **The collision:** `LLMContext.get_messages()` returns `self._messages` (`:245`);
  `ContextManager.get_messages()` returns `deepcopy(self._messages)` (`manager.py:47-51`).
- **Why each is right in its own home:** Pipecat needs identity because
  `_update_function_call_result` writes through the returned reference (`:2158-2165`). boson needs
  isolation because a copy handed to a provider adapter cannot be corrupted.
- **Failure mode if ported naively:** tool results silently never reach the model. No exception, no
  log line. §2.5 traces the eight steps.
- **Cost to resolve:** zero lines *if* you know. The whole cost is knowledge. Write it in the
  migration doc as a hard rule: **never override `get_messages`, never copy-on-read, take snapshots
  at the call site.**
- **Test that would catch it:** call a tool, then assert
  `any(m.get("content") != "IN_PROGRESS" for m in ctx.get_messages() if m.get("role") == "tool")`
  after the result settles.

### 9.2 Message schema rewrite

- **The collision:** two roles and blocks-in-content versus three-plus roles and a `tool_calls`
  array; dict arguments versus JSON-string arguments; `is_error` versus no field. §8.4 has the table.
- **Cost to resolve:** a converter in each direction if you need history to survive a migration
  (existing sessions, transcript archives, evaluation sets), plus a decision about `is_error`. If you
  do not need old history to load, the cost is only the 22 tools' result shapes.
- **What you lose:** pydantic validation on messages. `LLMContext` stores plain dicts with no
  schema. A typo in a hand-built message dict is discovered by the provider's 400, not by your type
  checker.
- **What you gain:** the adapter layer (§4.7) does the six provider shapings you currently hand-write.
- **Not a rename.** If someone on the team scopes this as "s/user/tool/", correct them.

### 9.3 No turn cap → you build a counting processor

This one you can actually price in lines, so here it is in lines. Build it from the frame facts
established above, not from intuition:

- The **first** inference of a turn arrives at the LLM as a **downstream** `LLMContextFrame` from the
  user aggregator, and the LLM **consumes** it (§3.2, observation 3). So a processor placed *after*
  the LLM never sees it.
- Every **re-prompt** arrives as an **upstream** `LLMContextFrame` from the assistant aggregator,
  travelling `assistant_agg → transport.output → tts → llm`. A processor placed **between `llm` and
  `tts`** sits on that path and sees every one of them, before the LLM does.
- `UserStartedSpeakingFrame` is a `SystemFrame` (`frames.py:1154`) travelling downstream from
  `transport.input`; the LLM forwards it through its `else` branch (`base_llm.py:614-615`), so the
  same processor sees it and can use it to reset.

That gives one placement that sees both the thing to count and the thing to reset on:

```python
# your code — e.g. lina/processors/tool_turn_cap.py
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ToolTurnCap(FrameProcessor):
    """Bound the number of tool→inference cycles inside a single user turn.

    Place BETWEEN the LLM service and the TTS service:

        [... user_agg, llm, ToolTurnCap(max_turns=8), tts, transport.output(), assistant_agg]

    Every re-prompt after a tool result travels UPSTREAM through this position on
    its way back to the LLM (llm_response_universal.py:1889), so counting them here
    counts exactly the cycles that Pipecat itself does not bound.
    """

    def __init__(self, *, max_turns: int = 8, **kwargs):
        super().__init__(**kwargs)
        self._max_turns = max_turns
        self._turns = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)   # non-negotiable; ch-01 §7.2

        if isinstance(frame, UserStartedSpeakingFrame):
            self._turns = 0

        elif isinstance(frame, LLMContextFrame) and direction == FrameDirection.UPSTREAM:
            self._turns += 1
            if self._turns > self._max_turns:
                logger.warning(
                    f"{self}: tool cycle cap hit ({self._turns} > {self._max_turns}); "
                    "dropping the re-prompt and closing the turn."
                )
                # Drop the frame: the LLM never runs again this turn.
                await self.push_frame(
                    TTSSpeakFrame("죄송합니다, 확인이 오래 걸리네요. 다시 말씀해 주시겠어요?"),
                    FrameDirection.DOWNSTREAM,
                )
                return

        await self.push_frame(frame, direction)
```

**Read the costs of this, not just the code.** They are real and you should not adopt it without
them:

1. **It counts re-prompts, not inferences.** The first inference of the turn is invisible from this
   position. `max_turns=8` here means "8 tool cycles", not boson's "8 total iterations". Off by one,
   by construction.
2. **Dropping the frame ends the turn by starvation.** Nothing else will push an `LLMContextFrame` at
   the LLM, so without the `TTSSpeakFrame` the bot simply goes silent. The spoken line is not
   decoration; it is the only thing that keeps the call alive.
3. **The context is left mid-tool.** The last tool result sits in history with no assistant reply
   after it. The next user turn appends a `user` message directly after a `tool` message. OpenAI
   accepts that. Confirm your provider does before shipping.
4. **`UserStartedSpeakingFrame` as the reset is a choice, not the only one.** [[ch-06/read]] taught
   the turn-strategy chain; if you use a different turn-start signal, reset on that instead. Reset on
   the wrong frame and the counter either never resets (every turn after the first is capped
   immediately) or resets mid-turn (the cap never fires).
5. **A gentler variant exists.** Instead of dropping, forward the frame after appending a nudge
   message and clearing the tool choice:
   `frame.context.add_message({"role": "system", "content": "[Tool budget exhausted — answer the customer directly, without calling tools.]"})`.
   That spends one more inference and gets you a real sentence instead of a canned one. It is
   OpenAI-shaped, so check what your adapter does with a mid-history `system` message before relying
   on it.

Roughly 30 lines plus a test. Against boson's two-line `while` condition. That is the honest price of
loop closure by topology: the bound has to be re-expressed as a processor, because there is no loop
header to put it in.

### 9.4 Three gates → one slot

- **The collision:** §8.5. Pipecat's `FunctionCallRegistryItem` has no permission field, no allowlist
  field, and `_run_function_call` has no interception point between name resolution and invocation.
- **Cost to resolve:** either 22 decorated handlers, or one catch-all
  (`register_function(None, ...)`) plus your own dispatch table — which is `ToolRouter` again,
  re-hosted. The `use_tool` / `use_skill` meta-tool indirection ports unchanged as a two-entry
  `LLMContext(tools=[...])`.
- **What does not port:** `ToolRegistry.discover_tools()` — filesystem discovery of `@tool`-decorated
  functions — has **no Pipecat equivalent**. It stays as boson glue that produces the list handed to
  `LLMContext(tools=[...])`.
- **A subtlety worth catching now:** the `_allowed_tools_var` `ContextVar` is per-asyncio-task, which
  is how one router serves concurrent sessions. Pipecat's per-session isolation is per
  `PipelineWorker` ([[ch-04/read]] §3), and a tool handler reaches its session through
  `params.pipeline_worker` / `params.app_resources` rather than through a `ContextVar`. Whether a
  `ContextVar` set outside a Pipecat task is still visible inside `_run_function_call`'s task depends
  on where the task was created — `create_task` copies the current context at creation time. Do not
  assume; test it before relying on it.

---

## 10. The third answer, already shipped: realtime_voice

Before the resolutions, one fact, stated as a fact.

[[ch-03/read]] §7.6 recorded that realtime_voice's agent slot is one Protocol that yields exactly one
type:

```python
# packages/realtime_voice/realtime_voice/protocols.py — shape as recorded in [[rtv-pipeline-session]]
class StreamingConversationAgent(Protocol):
    def stream(request) -> AsyncIterator[AgentTextDelta]
    async def cancel / close
```

`AgentTextDelta` and nothing else. Not tool calls. Not context objects. Not messages. Per
[[rtv-pipeline-session]] and [[rtv-vs-pipecat-gap]], tools live in `packages/basement` and
`packages/gateway`, reached through `GatewayConversationAgent.stream()` →
`bridge.dispatch_transcript()` (`agents/dental-w-tool-gateway/voice_server.py:163`). The excerpt's
own summary of the slot: *"Pipecat assumes it owns the LLM call, whereas boson deliberately delegates
to `GatewayConversationAgent` so stages/rules/tools stay text-native."*

Map that onto §1's three questions and the answer to all three is **"not the voice package."** The
voice layer holds no context, dispatches no tools, and learns the turn is over when the iterator
stops.

That is structurally the third resolution below — **bypass** — already implemented, already running,
already carrying Korean phone calls. So when you read §11.3, you are not reading a hypothetical: you
are reading a description of code that exists on branch `voice-chat-dev`.

**No claim is being made here about whether that was a good decision.** It is a fact about the state
of your repo, and it changes what "adopt" and "wrap" would cost — because both of them mean undoing
something that works today, and that undoing is a line item. [[ch-13/read]] prices it.

---

## 11. Three resolutions, costed. None recommended.

Ground rules for this section, so there is no ambiguity about what it is doing:

- Each resolution gets: **what you write**, **what you delete**, **what you lose**, **what you must
  rebuild**, and **the open questions that would decide it**.
- No option is marked recommended. No option is called better, cleaner, safer, or right.
- The costs are stated in the same units so [[ch-13/read]] can compare them.

### 11.1 Adopt — port the tools onto Pipecat's loop

**The shape.** `LLMContext` becomes the message list. `LLMContextAggregatorPair` straddles the LLM
service. `run_agent_loop` is deleted. Each boson tool becomes a direct function or a
`FunctionSchema(handler=...)`. The tool loop closes through the pipeline.

**What you write**

| Item | Size | Anchor |
|---|---|---|
| 22 tool signature changes: `handler(**args) → return v` becomes `handler(params) → await params.result_callback(v)` | 22 functions, or 1 shim decorator applied 22 times | §4.4; count from [[boson-tool-router]] |
| `run_in_parallel=False` on the LLM service constructor | 1 keyword | `llm_service.py:305` |
| `ToolTurnCap` processor to restore `max_turns` | ~30 lines + test | §9.3 |
| Permission + allowlist re-hosting (decorator or catch-all) | 1 catch-all + a dispatch table, or 22 decorations | §9.4 |
| Message-history converter, if old sessions must load | 2 functions | §8.4 |
| System-reminder injection processor (`pop_pending_reminders`) | 1 `FrameProcessor` | [[llm-service-context]] migration note; see [[custom-processor-guide]] |
| Hook re-hosting: 6 events → `llm.event_handler(...)` + processor placement | 6 sites | [[boson-agent-loop]] |

**What you delete.** `run_agent_loop` (561 L). `_reconcile_cancelled_tool_uses` (`:56`) and
`_await_tool_boundary` (`:113`) largely evaporate — per [[boson-agent-loop]], `FunctionCallCancelFrame`
plus `runner_item.settled` plus the two `UninterruptibleFrame`s cover what those ~80 lines maintained
by hand. The six hand-written provider tool-shapers in `basement/llm/*` are replaced by 12 adapters
(§4.7).

**What you lose**

- The turn is no longer readable in one place. Debugging "why did it call that tool twice" means
  reading a pipeline and a frame log instead of one function.
- `max_turns` until you build §9.3.
- The three-gate separation until you build §9.4.
- `is_error` as a first-class field (§8.4).
- Pydantic validation on messages.
- `AsyncIterator[StreamEvent]` as the public contract — `TextDelta` / `ToolUseStart` / `MessageEnd`
  become `LLMTextFrame` / `FunctionCallInProgressFrame` / `LLMFullResponseEndFrame`, so
  `gateway/core.py:323`'s `async for event in run_agent_loop(...)` — including its system-reminder
  tail-buffer scrubber at `:305-389` — becomes a `FrameProcessor` between LLM and TTS.

**What you gain**

- 12 provider adapters instead of 6 hand-written shapers.
- The interruption behaviour of §4.5 without `_reconcile_cancelled_tool_uses`.
- Every step of §5.3 visible to the observer plane ([[ch-11/read]]) for free.
- `params.context` — tool handlers can read and rewrite history.

**Open questions that would decide it**

1. Does any live Lina tool depend on `is_error` for control flow, or is it only ever surfaced to the
   model as text?
2. Is `ctx.get_system_prompt()` used anywhere that a `system_instruction` setting or a leading
   `system` message cannot serve?
3. Do you need old sessions to load, or is a clean cut acceptable?

### 11.2 Wrap — one custom `FrameProcessor` containing boson's loop

**The shape.** Pipecat owns transport, VAD, STT, TTS, and the interruption cascade. One
`FrameProcessor` sits where the LLM service would be, receives the accumulated user text, calls
`run_agent_loop` internally, and emits `LLMTextFrame`s downstream. boson keeps the context, the
tools, the gates, and the turn counter.

**What you write**

| Item | Size | Anchor |
|---|---|---|
| `BosonAgentProcessor(FrameProcessor)` — consume the turn-end signal, call `run_agent_loop`, translate `StreamEvent` → frames | ~150–250 lines | [[custom-processor-guide]]; contract in [[ch-01/read]] §7 |
| Interruption handling inside it: reset accumulators on `InterruptionFrame`, cancel the in-flight loop | ~30 lines | [[ch-08/read]] |
| Bridge from `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` bracketing to boson's `MessageEnd` | ~20 lines | `base_llm.py:603, 613` |

**What you delete.** Nothing in boson. On the Pipecat side you do not use `LLMContext`,
`LLMContextAggregatorPair`, `LLMService`, the adapter layer, or `run_function_calls`.

**What you lose**

- Everything §11.1 gains. The adapter layer, `FunctionCallParams`, the uninterruptible result frames,
  the frame-level visibility of tool calls. Your tool calls are invisible to [[ch-11/read]]'s
  observer plane unless you emit frames for them yourself.
- The aggregator pair's turn-boundary machinery — [[ch-06/read]]'s turn-strategy chain is wired into
  `LLMUserAggregator`, so bypassing it means re-deciding where "the user's turn ended" comes from.
- Compatibility with Pipecat Flows ([[ch-10/read]]), which drives conversation state by manipulating
  the `LLMContext` you are not using.

**What you keep.** `max_turns`. The three gates. The `Message` schema. The hooks. The 22 tool
signatures, untouched. `deepcopy` in `get_messages()` — which stays correct, because you are not
using the aggregator that depends on the live list.

**The specific hazards of this option, named**

- **The cancellation seam is the risk.** [[boson-agent-loop]] records that `cancellation_flag` is
  read at exactly two places — `:344` after a tool batch and `:513` after one tool completes — and
  **never between `TextDelta`s**. `gateway/interrupt/cancellation.py:171` says so verbatim:
  `NOTE: Cooperative — tool runs to completion, then flag is checked.` Pipecat's interruption
  cancels your `process_frame` task at whatever `await` it is parked on ([[ch-08/read]]). If that is
  inside `run_agent_loop`'s provider stream, you get a hard `CancelledError` in the middle of a
  561-line function whose repair apparatus was designed for exactly that case. It should work. It is
  the seam to test first, hardest, and with Korean audio.
- **You are inside one `process_frame` for the whole turn.** Every tool call, every re-prompt.
  [[ch-04/read]] §4 taught what that means for the processor's two-task model: a long-running
  `process_frame` occupies the cancellable task for its entire duration.

**Open questions that would decide it**

1. Does a hard `CancelledError` mid-`run_agent_loop` leave history repairable in practice, on real
   barge-in traffic, not in a unit test?
2. Do you want Flows ([[ch-10/read]])? If yes, this option makes it expensive.
3. Who owns the turn-end signal if `LLMUserAggregator` is not in the pipeline?

### 11.3 Bypass — use Pipecat for voice I/O only

**The shape.** Pipecat runs `transport.input()`, VAD, STT, TTS, `transport.output()` and the
interruption cascade. No `LLMContext`, no aggregator pair, no LLM service. A processor forwards final
transcripts out to the gateway over your existing text contract and feeds returned text into TTS.
boson keeps everything about the agent.

**This is the shape realtime_voice already implements** (§10) — with realtime_voice, rather than
Pipecat, as the voice layer.

**Which Pipecat processors remain in the pipeline** — concretely, since the figure asks you to
enumerate them:

```python
pipeline = Pipeline([
    transport.input(),          # WebRTC / WebSocket / telephony      — ch-05
    stt,                        # STTService                          — ch-06
    turn_detection_or_vad,      # if not folded into the aggregator   — ch-06
    boson_bridge,               # YOUR processor: transcript out, text in
    tts,                        # TTSService                          — ch-07
    transport.output(),         # MediaSender + clock task            — ch-07, ch-08
])
```

Gone from the canonical §2.7 pipeline: `user_aggregator`, `llm`, `assistant_aggregator`. Three
positions out of seven.

**What you write**

| Item | Size | Anchor |
|---|---|---|
| `boson_bridge` processor: `TranscriptionFrame` → gateway; returned text → `LLMTextFrame`/`TTSSpeakFrame` | ~100–150 lines | [[custom-processor-guide]] |
| Turn-end decision, since `LLMUserAggregator` is absent | depends on the strategy chosen | [[ch-06/read]] |
| Your own interruption forwarding to the gateway's `CancellationFlag` | ~20 lines | [[ch-08/read]] |

**What you lose**

- Everything in §11.1's gain column, plus Flows ([[ch-10/read]]) entirely, plus any frame-level
  visibility into the agent's behaviour.
- Rule layers as pipeline middleware ([[ch-12/read]]) apply only to the voice half; agent-side rules
  stay in boson.

**What you keep.** All of boson, unchanged. The text-native contract. The gates. `max_turns`. The
schema. And a voice layer with the transport breadth, the serializers, and the interruption cascade
that [[ch-05/read]] through [[ch-08/read]] catalogued.

**The specific cost, named.** You are running two frameworks. Pipecat's frame model does not reach
into boson, and boson's `StreamEvent` model does not reach into Pipecat, so the bridge processor is
a translation layer you own forever. Every new frame type that matters to the agent — a rule
violation, a stage transition, a barge-in classification — has to be plumbed through it by hand.

**Open questions that would decide it**

1. What is the actual latency cost of the extra hop out to the gateway and back, measured, on a real
   Korean call? [[ch-11/read]] builds the budget that answers this.
2. Does the gateway's `CancellationFlag` respond fast enough to a Pipecat `InterruptionFrame` to
   satisfy the barge-in targets from [[ch-08/read]]?
3. If realtime_voice already implements this shape, what specifically does swapping in Pipecat's
   voice layer buy — and that is a [[ch-13/read]] question, not this chapter's.

### 11.4 The three costs, side by side

| | **Adopt** | **Wrap** | **Bypass** |
|---|---|---|---|
| New Pipecat-side code | ~30 L cap + gates + reminder processor | ~200–300 L in one processor | ~120–170 L bridge |
| boson code deleted | `run_agent_loop` (561 L) + ~80 L cancellation repair + 6 provider shapers | none | none |
| Tool signatures changed | 22 | 0 | 0 |
| `max_turns` | rebuilt as a processor | kept | kept |
| Three gates | rebuilt as catch-all or decorators | kept | kept |
| Message schema | rewritten | kept | kept |
| Adapter layer used | yes (12 adapters) | no | no |
| Flows ([[ch-10/read]]) available | yes | expensive | no |
| Observer plane sees tool calls | yes | only what you emit | no |
| Interruption of the agent | framework-provided | your seam to prove | your seam to prove |
| Already implemented in your repo | no | no | **yes, with realtime_voice as the voice layer** |

**No row of this table is a verdict, and the last row is a fact about your repository, not an
argument.** [[ch-13/read]] scores this. This chapter hands it over unresolved, which is the whole
job.

---

## 12. Three framework-extension moves for Lina

Your strongest mode, applied to the mechanisms above. Each of these is a design you could sketch this
week; none of them requires choosing a resolution first.

**Move 1 — the catch-all as a permission kernel.** `register_function(None, handler)` installs a
handler that receives **every** call the registry does not match by name (`llm_service.py:1467-1468`,
and again at execution time `:1554-1555`). Combine that with `FunctionSchema(handler=None)` —
advertise-only schemas (§4.1, route 2) — and you get a Pipecat-native `ToolRouter`: advertise N
handler-less schemas so the model sees real tool names with real parameter schemas, register exactly
one catch-all, and do permission → allowlist → dispatch inside it, in boson's order (§8.5). You get
the three gates back **and** you keep real tool names in the prompt instead of `use_tool` indirection.
Cost: you lose per-tool `timeout_secs` and `cancel_on_interruption`, because those live on the
registry item and there is only one registry item. Design question to answer before building: is
per-tool timeout worth more to Lina than gate separation?

**Move 2 — a stage machine driven by `LLMSetToolsFrame` with a pinned prefix.** boson pins the
advertised set to two entries under `maximize_caching` because *"the prompt-cache prefix is ordered
tools → system → messages"* (§8.5). Pipecat's `LLMSetToolsFrame` path (§8.3) changes the advertised
set mid-session, and `add_tool_change_messages=True` on the aggregator pair (`:2260`) appends an
announcement so the model stops calling what vanished. Those two facts are in tension: announcing a
tool change **appends a message**, which is *after* the tools block in the cache prefix — so an
announcement costs you nothing in cache terms, while changing `tools` costs you the whole prefix.
That gives a design you can state precisely: **advertise a byte-stable superset, announce
availability changes as messages, enforce with the allowlist.** Prompt cache preserved, model kept
honest, gates intact. Write it up as a one-page design and check the tokens.

**Move 3 — `run_llm=False` as the end-of-call primitive.** §5.2 showed a handler can settle with
`FunctionCallResultProperties(run_llm=False)` and no re-prompt happens. That is the only place in
Pipecat where a *tool* gets to say "the turn is over." For a tele-sales agent, `end_call`,
`transfer_to_human`, and `schedule_callback` are exactly the tools that should end a turn without a
reply — and today, in boson, each of them ends the turn only by convincing the model to stop calling
tools. Sketch the three handlers with `run_llm=False` plus the `EndFrame` / `CancelFrame` shutdown
paths from [[ch-04/read]] §8, and you have a deterministic end-of-call that does not depend on model
behaviour. Mechanically, boson has no equivalent hook today — its turn ends at `agent_loop.py:363`
on a text-only response, which the model must be persuaded to produce. That difference is a fact
about the two mechanisms, not a score; it belongs on the [[ch-13/read]] give-back list as a
candidate under all three resolutions.

---

## 13. What to hold in your head

Twelve facts. If you remember nothing else from this chapter, remember these, and remember that every
one of them is checkable with a command you have already seen.

1. `LLMContext` is three fields (`llm_context.py:110-112`) and the application owns it.
2. `create_context_aggregator()` **does not exist** at this commit. Zero hits in `src/` and
   `examples/`.
3. `register_direct_function` is `@deprecated` since 1.4.0 (`llm_service.py:982-984`).
4. `get_messages()` returns the **live list** (`:245`); only `truncate_large_values=True` copies.
5. It returns the live list **because** `_update_function_call_result` assigns
   `message["content"] = result` through it (`:2158-2165`).
6. boson's `get_messages()` returns `deepcopy(...)`. Porting that defence breaks the tool path
   silently.
7. One completion per `LLMContextFrame`. `:599` super, `:601` isinstance, `:603` start frame,
   `:604` metrics, **`:605` the completion**.
8. `:601` **does not check direction**, which is why an upstream frame re-prompts the same service.
9. The loop closes at `llm_response_universal.py:1889` —
   `await self.push_context_frame(FrameDirection.UPSTREAM)` from the last processor in the pipeline.
10. `grep -rn "max_turns\|max_iterations\|max_tool_calls" src/pipecat/` → **zero**. The only bounds
    are per-call timeouts.
11. boson keeps three gates — exposure, availability (`_allowed_tools_var`), permission
    (`PermissionChecker`). `FunctionCallRegistryItem` has slots for none.
12. `run_in_parallel` defaults to `True`. boson's tools assume sequential and hold a
    `threading.Lock`. Re-assert it.

And one sentence per system, for Q3:

- **Pipecat:** the turn ends when nobody pushes an `LLMContextFrame` upstream. An absence.
- **boson:** the turn ends at `agent_loop.py:363`, `break  # Done — text response means end of turn`.
  A statement.
- **realtime_voice:** the turn ends when the agent's `AsyncIterator[AgentTextDelta]` stops. A
  delegation.

---

## 다음 챕터로

This chapter hands forward three things.

**A resolved mechanism.** You now know, at file-and-line resolution, who owns the context (a plain
object the application constructs), who dispatches tools (the service, from a registry the context
populates and prunes), and who decides the turn ended (nobody — it ends by the absence of an upstream
frame). That is not an opinion about Pipecat; it is a reading of `llm_context.py`,
`llm_response_universal.py`, `llm_service.py` and `base_llm.py`.

**Two absences and one inversion, priced.** No turn cap (~30 lines to rebuild, with four costs named
in §9.3). No permission or allowlist slot in the registry (a catch-all or 22 decorations, §9.4). And
the `deepcopy`-versus-live-list inversion that costs zero lines and everything in attention (§9.1).

**An unresolved choice.** Adopt, wrap, bypass — costed in §11, **none recommended**, with the fact
recorded that realtime_voice already implements the bypass shape. [[ch-13/read]] scores it, and it
gets to do that only after [[ch-10/read]] through [[ch-12/read]] have shown it the rest.

Next is [[ch-10/read]] — **Pipecat Flows: a state machine that lives OUTSIDE the pipeline.** It
depends on this chapter for one reason you can already predict from §2: Flows drives a conversation
by manipulating the very `LLMContext` you just watched two aggregators fight over, and by swapping
the advertised tool set with the `LLMSetToolsFrame` path from §8.3. Every node transition is a
`set_messages` / `set_tools` pair against the shared object. If you did not fully believe §2.3's
claim that the live list is load-bearing, [[ch-10/read]] will make you believe it — because a state
machine that mutates a context it does not own is either elegant or terrifying depending entirely on
whether you know who else is holding a reference.

Bring §8.5's three gates with you. Flows has a stage concept too, and the comparison is the point.
