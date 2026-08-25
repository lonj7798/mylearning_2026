---
title: "Rule Layers as Middleware: Designing boson's Rules onto Pipecat"
chapter: ch-12
phase: collision
course: pipecat
sources:
  - design-boson-rules-on-pipecat
  - boson-layers-rules
  - boson-script-engine
  - custom-processor-guide
  - processor-vocabulary
  - bus-and-extensions
  - flows-state-machine
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# Rule Layers as Middleware: Designing boson's Rules onto Pipecat

## 왜 이 챕터인가

You asked for this chapter. It sat behind [[ch-11/read]] on purpose, because the central
trade-off in it is a **latency** trade-off, and a latency trade-off that cannot be priced is
just an opinion. ch-11 built the budget. This chapter spends it.

It is also the chapter where the course stops explaining Pipecat and starts making you
design against it. So it is written differently from every chapter before it, and the
difference is not stylistic — it is the point.

**This chapter withholds its answers.** It is organised in four steps, and they run in this
order and no other:

| | what happens | what is deliberately absent |
|---|---|---|
| **STEP 1** | Four verified facts. One permission, three constraints. Each with a file and a line you can `awk` yourself. | any design, any position, any processor name, any code |
| **STEP 2** | Two questions, posed and **left open**. | any sentence that answers either |
| **STEP 3** | You open `figures/rule-processor-placement.html` and place two blocks yourself. | the figure's reveal button stays disabled until you have |
| **STEP 4** | The two conclusions, an eleven-row mapping table, the pipeline listing, and ~40 real lines of `process_frame` — **framed as the check on your derivation, not as the lesson** | nothing |

If you read STEP 4 first you will learn the answer and not the derivation, and the answer is
worth much less than the derivation, because the answer is specific to Pipecat at commit
`0cbf9c5b` and the derivation is specific to *you*. There are three constraints and two
conclusions, and the conclusions follow from the constraints tightly enough that you can get
there. That is why the constraints come with line numbers and the conclusions come with a
"if your answer disagrees with this row, one of us is wrong and here is the line that
settles it."

Two things this chapter is **not**. It is not a keep-or-replace argument — that is
[[ch-13/read]]'s job and nowhere else's. And it is not a claim that Pipecat is missing
something. Pipecat is not missing anything; it made a different bet about where transactions
live, and this chapter finds the exact place where that bet costs you.

One preliminary: this chapter assumes you already have [[ch-02/read]]'s three-way frame test,
[[ch-09/read]]'s ownership model of `LLMContext` (especially §2.2 and §2.3 — the slice
assignment and the live list), [[ch-10/read]]'s reading of `flows/` as a machine that lives
*outside* the pipeline, and [[ch-11/read]]'s latency budget with its empty slot for rule
evaluation. It will not re-derive any of them.

---
---

# STEP 1 — THE FACTS

Four facts. One is a **permission** — it removes a blocker you would otherwise assume exists.
Three are **constraints** — they are walls, and the design has to be built inside them.

Nothing in STEP 1 is a design. Nothing in STEP 1 names a processor, a position, or a
pipeline. If you find yourself reading a conclusion here, I have failed and you should tell
me so in the discuss phase.

---

## 1. FACT ZERO (the permission): Flows does not require transitions to be LLM function calls

This one is not a constraint, it is a lock that turns out to be open. It goes first because
if you believe the opposite — and the course's own earlier working assumption believed the
opposite, see [[boson-stage-machine]]'s migration note — then boson's entire deterministic
stage machine looks unportable and you will design around a wall that is not there.

boson's core invariant, stated in [[boson-stage-machine]]: *"the transition is performed by
deterministic rule code, not by the model … The LLM is told which stage it is in and never
gets to choose the next one."* Nine registered Lina stages, a `transitions` whitelist per
stage, and `StageMachine.transition()` rejecting anything not on the list.

Pipecat Flows has **two** independent producers of a `NodeConfig`, and only one of them
involves the model.

**Path A — an LLM function call.** A tool handler returns `(result, next_node)`; the tuple's
second element *is* the transition.

**`src/pipecat/flows/manager.py:443-447`**

```python
    async def _create_transition_func(
        self,
        name: str,
        handler: Callable | FlowsDirectFunctionWrapper,
    ) -> Callable:
```

That path is gated by a concurrency guard, which is worth seeing because it tells you Flows
takes in-flight tool batches seriously:

**`src/pipecat/flows/manager.py:525-537`**

```python
    async def _check_and_execute_transition(self) -> None:
        """Check if all functions are complete and execute transition if so."""
        if not self._pending_transition:
            return

        # Check if all function calls are complete using Pipecat's state
        assistant_aggregator = self._context_aggregator.assistant()
        if not assistant_aggregator.has_function_calls_in_progress:
            # All functions complete, execute transition
            transition_info = self._pending_transition
            self._pending_transition = None

            await self._execute_transition(transition_info)
```

`has_function_calls_in_progress` is `llm_response_universal.py:1503`, and its body is
`return bool(self._function_calls_in_progress)` at `:1509`. A parallel tool batch cannot
transition mid-batch. Good engineering, and irrelevant to you, because you are not going to
use Path A.

**Path B — plain code, no model involved.** The method is public:

**`src/pipecat/flows/manager.py:588-591`**

```python
    async def set_node_from_config(self, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Used to manually transition between nodes in a flow.
```

`async def`, no leading underscore, docstring says *"Used to manually transition."* And it is
not a theoretical API — it is called from ordinary Python twice in-tree, from two different
kinds of callback:

**`examples/flows/warm_transfer.py:259-261`**

```python
async def start_human_agent_interaction(flow_manager: FlowManager):
    """Transition to the "human_agent_interaction" node."""
    await flow_manager.set_node_from_config(create_human_agent_interaction_node())
```

driven from a Daily transport event handler guarded on flow state:

**`examples/flows/warm_transfer.py:656-658`**

```python
            user_id = participant.get("info", {}).get("userId")
            if user_id == "agent" and flow_manager.current_node == "transferring_to_human_agent":
                await start_human_agent_interaction(flow_manager=flow_manager)
```

and again, from a worker lifecycle event:

**`examples/flows/multi_worker_handoff.py:345-352`**

```python
    @worker.event_handler("on_activated")
    async def on_activated(worker, args):
        if not initialized["done"]:
            initialized["done"] = True
            await flow_manager.initialize(party_size_node())
        else:
            # Control was handed back to us; restart the reservation flow.
            await flow_manager.set_node_from_config(party_size_node())
```

Neither call site is a tool handler. Neither involves the model at any point.

> **Source correction.** [[design-boson-rules-on-pipecat]] cites Path B's proof as
> `warm_transfer.py:658`, and [[flows-state-machine]] cites both `:658` and `:259-261`.
> Opening the file settles it: **`:261` is the `set_node_from_config` call**, and **`:658`
> is the transport callback that invokes the function containing it.** Both lines are real
> and they are two different things. Cite `:261` when you mean "code calls the public API"
> and `:658` when you mean "and the caller is a transport event." The chapter spec's `:261`
> is the tighter citation and I use it.

**What Fact Zero gives you, stated as narrowly as it deserves:** boson's chain —
`RuleEngine.evaluate` → `Action` → `ExecutionResult.pending_transition` →
`core._apply_stage_transition` → `StageMachine.transition` ([[boson-stage-machine]]) — has a
sink in Pipecat that the model cannot reach. Whatever object ends up holding a `flow_manager`
reference can call `set_node_from_config(node_for(target))` and the LLM never gains control
of the stage machine.

**What Fact Zero does not give you:** it says nothing about *where* that object lives, nothing
about *when* in a turn it may act, and nothing about whether it can be more than one object.
Those are the two questions in STEP 2, and Fact Zero answers neither. Check that claim
yourself when you get there.

---

## 2. CONSTRAINT ONE — boson's rules provably require a FINISHED user utterance

Not "prefer". Not "work better with". **Require**, structurally, and the proof is in three
independent places in boson's own code. All three come from [[boson-layers-rules]] and
[[design-boson-rules-on-pipecat]] (boson-agent is private; these are the pre-read excerpts and
I do not open that repo).

**(a) The type signature.** `LayerPipeline.process` (`gateway/layers/pipeline.py:87-94`):

```python
async def process(
    self,
    session_id: str,
    content: str,
    session: SessionState,
    *,
    user_message_appended: bool = False,
)
```

`content: str`. Not `AsyncIterator[str]`, not a stream handle, not a partial-with-a-final-flag.
One whole utterance, one call. There is no API through which a rule could receive a prefix.

**(b) The gateway refuses to deliver partials.** Every branch of the `partial_transcript`
handler in `server/websocket.py:293-317` ends in `continue`. Partials land in
`self._partial_transcripts[session_id]` and never reach `_replace_active_task(...)` at `:374`,
which is the *sole* path to `_message_handler`. The comment at `:288-292` is explicit, and
quoted verbatim in [[boson-layers-rules]]: *"Incremental ASR: keep only the latest hypothesis.
A partial may stop an in-flight response promptly, but only after the same filler/policy gate
used by explicit final frames has authorized the interruption. **Rules/LLMs/tools still do not
see incomplete text.**"*

**(c) The rules exploit it, so weakening it breaks them silently.** This is the part that
matters, because (a) and (b) are policy and could in principle be relaxed; (c) cannot.

`end_signal.py` does whole-string `kw in lower` matching over
`user_message.lower().strip()`, across 5 categories × ~10 Korean/English keywords, gated by a
`STAGE_SIGNALS` table at `:54-62`. `_detect_natural_close` reaches back over `messages[-4:]`.
Both are whole-string operations. Feed either a prefix and it does not error — it **fires
early on the wrong thing**, which is worse.

The filler filter is the same shape:

```python
@check("korean_filler_filter", mode="sequential", priority=10)
def filter_fillers(messages, user_message, session):
    """Filter filler words ONLY during agent streaming."""
    status = getattr(session, "pre_turn_status", None) or session.get_agent_status()
    if status not in ("generating", "tool_processing"):
        return Pass()
    if _is_filler_text(user_message):
        return Filter(
            reason=f"filler_word:{_normalize(user_message) or user_message.strip()}"
            f" | agent_status:{status}"
        )
    return Pass()
```

(`agents/test-lina-gateway/layers/01-filler-filter/rules/korean_fillers.py:53-75`, via
[[boson-layers-rules]].) `_is_filler_text` tests membership in
`KOREAN_FILLERS = ["네","예","아","음","어","그","응","아아","음음","네네","예예","아하","흠","그래요"]`
after normalising away non-Hangul characters. `"네"` is in that list. So is the first
character of `"네 그런데 제가 지난번에 여쭤본 게…"`. A `TranscriptionFrame` carrying an early
hypothesis of that sentence would be filtered as a backchannel, and the customer's actual
question would be discarded.

And there is one more layer of this in Pipecat's own vocabulary that you have already met.
`InterimTranscriptionFrame` (`frames.py:476`) is a `TextFrame` subclass, so *any* code that
tests `isinstance(frame, TextFrame)` and reads `.text` sees interim hypotheses unless it
opts out. Pipecat's own `SentenceAggregator` opts out with a bare `return`
(`aggregators/sentence.py:50-51`, quoted in [[custom-processor-guide]]).

**Constraint One, in one line:** every boson rule is a pure Python function whose input
contract is *one complete utterance as a `str`*, and three of the thirteen live checks would
produce wrong answers — not errors, wrong answers — on a prefix.

---

## 3. CONSTRAINT TWO — `push_frame` is irreversible

There is no un-push. There is no retraction frame. There is no `pop_frame`, no
`recall_frame`, no `cancel_pushed`. Grep for them; they do not exist.

What exists is this:

**`src/pipecat/processors/frame_processor.py:1004-1015`**

```python
    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame to the next processor in the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await self._call_event_handler("on_before_push_frame", frame)

        await self.__internal_push_frame(frame, direction)

        await self._call_event_handler("on_after_push_frame", frame)
```

Three statements. A hook before, the enqueue-on-neighbour, a hook after. [[ch-01/read]]
established what `__internal_push_frame` does: it puts the frame on the *next processor's*
input queue. The moment that returns, the frame is in another processor's queue and that
processor's task may already be running it. You do not own it any more. You never owned it.

The closest thing to a retraction in the whole framework is broadcast interruption:

**`src/pipecat/processors/frame_processor.py:1017-1022`**

```python
    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        self.__reset_process_task()
        await self.stop_all_metrics()
        await self.broadcast_frame(InterruptionFrame)
```

Read what that is and what it is not. It resets *this* processor's process task, stops
metrics, and sends an `InterruptionFrame` in both directions. Downstream processors drain
their queues in response ([[ch-08/read]] traced the full cascade). It is a **best-effort
global abort**, not an undo: it does not tell anyone what to un-do, it does not restore any
state anyone mutated, and by the time it arrives, side effects that were going to happen have
happened — audio may already be on the wire.

Contrast with what boson does. [[boson-layers-rules]] describes `LayerPipeline._process_active`
(`:128-335`) as a genuine two-phase commit. **Phase 1** (`:178-248`) evaluates every layer and
only *stages* actions — nothing escapes. On `decision == "filter"` (`:205-239`) it walks
`session.messages` **backwards by object identity** and deletes the exact
`pipeline_user_message` object it appended, clears `_pending_stage_injection`, and returns.
**Phase 2** (`:254-316`) replays the staged actions only if Phase 1 committed. And the
flow-control precedence that makes the arbitration deterministic:

```python
ACTION_PRIORITY = {
    "filter": 0, "respond": 1, "inject": 2,
    "stage_transition": 3, "compact": 3, "pre_tool": 3,
    "pass": 4, "continue": 4,
}
```

(`gateway/layers/pipeline.py:42-51`, via [[boson-layers-rules]].) `filter` is 0. It beats
everything, from any layer, at any time before the commit.

**Constraint Two, in one line:** boson's veto is a transaction rollback over staged effects;
Pipecat has exactly one write verb, it takes effect on call, and the only global abort it
offers restores nothing.

---

## 4. CONSTRAINT THREE — the data dependency, and the two lines inside it that matter

This is the constraint the whole chapter rests on, so I am giving it four sub-sections and
telling you to run the `awk` yourself. The chapter spec flags that an earlier draft cited the
wrong evidence line here and built the derivation on it. [[ch-09/read]] §0(d) flagged the same
class of error. Do not take this one from me.

### 4.1 The aggregator side: write, then push, then notify

**`src/pipecat/processors/aggregators/llm_response_universal.py:856-873`**

```python
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

Verify:

```console
$ awk 'NR>=856 && NR<=873 {printf "%d\t%s\n", NR, $0}' \
    src/pipecat/processors/aggregators/llm_response_universal.py
```

Three line numbers, three different events, in this order:

| line | statement | what it means |
|---|---|---|
| **`:863`** | `self._context.add_message({"role": self.role, "content": aggregation})` | the user's finished turn is now **in the shared `LLMContext`** |
| **`:866`** | `await self.push_context_frame()` | an `LLMContextFrame` leaves the aggregator, travelling downstream |
| **`:871`** | `await self._call_event_handler("on_user_turn_message_added", message)` | an event fires |

`:863` before `:866` before `:871`. Not a convention — a straight-line function body.

`push_context_frame()` builds the frame from the same context object
(`_get_context_frame` → `LLMContextFrame(context=self._context)`, `llm_response_universal.py:564-568`,
read in [[ch-09/read]] §3). The frame is a doorbell, not a parcel: `LLMContextFrame` carries a
reference, and [[ch-02/read]] §3 already noted it subclasses `Frame` directly, in none of the
three branches.

### 4.2 The service side: four consecutive lines that do four different things

**`src/pipecat/services/openai/base_llm.py:599-605`**

```python
        await super().process_frame(frame, direction)          # :599

        if isinstance(frame, LLMContextFrame):                 # :601
            try:
                await self.push_frame(LLMFullResponseStartFrame())   # :603
                await self.start_processing_metrics()                # :604
                await self._process_context(frame.context)           # :605
```

Verify:

```console
$ awk 'NR>=599 && NR<=605 {printf "%d\t%s\n", NR, $0}' \
    src/pipecat/services/openai/base_llm.py
```

| line | statement | what it means |
|---|---|---|
| **`:601`** | `isinstance(frame, LLMContextFrame)` | the **only** trigger. No other frame starts a completion. |
| **`:603`** | `await self.push_frame(LLMFullResponseStartFrame())` | **a frame leaves this processor, downstream** |
| **`:604`** | `await self.start_processing_metrics()` | the metrics clock starts |
| **`:605`** | `await self._process_context(frame.context)` | **the completion begins here** |

### 4.3 The `:601 → :605` window is not empty, and that is the whole point

Read `:603` again. Before any token is generated, before `_process_context` is even entered,
`LLMFullResponseStartFrame` has already been pushed downstream. That is a `push_frame` call,
and Constraint Two says a `push_frame` call is irreversible.

So the window between "the service recognised the frame" and "the service started generating"
is **four lines wide and already contaminated**. You cannot reach into it. There is no hook
between `:601` and `:605`, and even if you invented one, by `:604` an irreversible push has
already happened and downstream processors are already reacting to the opening bracket of a
response that may not exist.

This is why the "let generation start and cancel it" idea has to be evaluated as *paying
twice*, not as *free*. But that evaluation is yours, in STEP 2.

### 4.4 One more line, because your rollback surface depends on it

`LLMContext` offers exactly one replace verb:

**`src/pipecat/processors/aggregators/llm_context.py:361-383`**

```python
    def add_message(self, message: LLMContextMessage):
        """Add a single message to the context.

        Args:
            message: The message to add to the conversation history.
        """
        self._messages.append(message)

    def add_messages(self, messages: list[LLMContextMessage]):
        """Add multiple messages to the context.

        Args:
            messages: List of messages to add to the conversation history.
        """
        self._messages.extend(messages)

    def set_messages(self, messages: list[LLMContextMessage]):
        """Replace all messages in the context.

        Args:
            messages: New list of messages to replace the current history.
        """
        self._messages[:] = messages
```

`set_messages(list)`. It takes a **list**. It does not take a message, an index, or an object
handle. There is no `remove_message`, no `delete_message_by_identity`, no
`pop_message`. [[ch-09/read]] §2.2 already explained why `self._messages[:] = messages` is a
slice assignment rather than a rebind, and §2.3 already established that `get_messages()`
returns the **live list**. Both facts are load-bearing here and I will cash them in STEP 4.

**Constraint Three, in one line:** the write at `:863` happens before the push at `:866`, the
push at `:866` reaches a service whose *only* trigger is `:601` and whose completion starts at
`:605` — with an irreversible downstream push already spent at `:603` — and the only rollback
verb anywhere on that path is `set_messages(list)`.

---

## STOP.

That is STEP 1. One permission, three constraints, eleven verified line numbers.

Do not scroll to STEP 4 yet. If you already have an answer forming, write it down somewhere
before you read STEP 2 — an answer you wrote before you saw the questions is a much better
diagnostic than one you wrote after.

---
---

# STEP 2 — THE TWO QUESTIONS

Two questions. I am posing them and leaving them open. Nothing between here and the end of
STEP 3 answers either one, and if you catch me answering one early, that is a defect in the
chapter and I want to hear about it in discuss.

Both are answerable from STEP 1 alone. You do not need any fact I have not already given you.

---

## Q1 — WHERE IS THE SEAM?

> **At which single position in a Pipecat pipeline does a complete user turn exist while
> inference has not yet begun?**

Precision matters in how you answer this. "Somewhere in the middle" is not an answer. The
answer has the form *"immediately downstream of X and immediately upstream of Y"*, and both
X and Y must be justified by a **data dependency** — a line number that writes something you
need, or a line number that consumes something you must get in front of — not by taste, not
by convention, and not by "that's where middleware usually goes."

Two additional demands on your answer:

**(i) It must be a single position, or you must prove there is more than one.** Constraint
Three gives you a chain of line numbers. Walk it and see how many candidate positions
survive both halves of the question at once — *complete* and *not yet inferring*.

**(ii) You must knock out two specific rivals by name**, because both are plausible and both
are things a competent engineer proposes on day one:

- **Rival B — the event handler.** `on_user_turn_message_added` fires at `:871` and it hands
  you the complete aggregated turn as a `UserTurnMessageAddedMessage`. That is exactly the
  data a rule wants. Why is it not the seam? (Constraint Three, §4.1 has the answer. One line
  number decides it.)
- **Rival C — start and cancel.** Let the LLM begin, run the rules concurrently, and if a rule
  vetoes, call `broadcast_interruption()`. Zero added latency on the pass path, which is most
  turns. Why is that not free? (Constraints Two and Three, §4.3. Two costs, and one of them
  is not a latency cost at all.)

---

## Q2 — MUST THE LAYERS COLLAPSE?

> **Can boson's cross-layer veto survive being spread across N adjacent `FrameProcessor`s, or
> not?**

boson has four live layers — `01-filler-filter`, `02-analyzer`, `03-orchestrator`,
`04-committer` ([[boson-layers-rules]]). Pipecat's whole aesthetic, established in
[[ch-01/read]], is that a pipeline is a list and you splice things into it. Four layers, four
processors, in order. It is the obvious shape and it is the shape [[custom-processor-guide]]
recommends in so many words:

> *"porting it as a single monolithic processor keeps it working but forfeits the
> per-processor queueing, observability, and interruption semantics Pipecat gives each stage
> — the honest port splits its four layers … into four `FrameProcessor`s."*

That is a real recommendation from the excerpt library, written by someone who had read
`frame_processor.py`. Decide whether it is right.

To answer, you need to be concrete about one scenario:

> Layer `03-orchestrator` stages an `Inject` and a `PreTool`. Layer `04-committer` then
> returns `Filter(reason=...)`. Per `ACTION_PRIORITY`, `filter` is 0 and wins: **every**
> staged effect from **every** layer is discarded, including the user message that was
> appended before layer 01 ran, deleted by object identity.

Now run that scenario across four separate processors, where each processor's only way to
communicate with the next is `push_frame`, and answer:

- **(a)** What has already left processor 03 by the time processor 04 decides to veto?
- **(b)** Constraint Three §4.4 gives you exactly one rollback verb. Does it have the shape
  boson's rollback needs? Look at what boson deletes *by* (object identity) and what
  `set_messages` accepts (a list). Is one expressible in terms of the other, in general?
- **(c)** If your answer to (a) is "nothing has left yet, because processor 03 could just
  *not push*", then say what processor 04 is doing at that moment. Where does it get the frame
  it is evaluating?

There is a right answer to (c) and it is the hinge of the whole question.

---

**Write both answers down before STEP 3.** Not in your head. STEP 3 is a bench that reports
what actually happens at a position, and a bench is only useful if you brought a prediction
to it.

---
---

# STEP 3 — GO WORK THE FIGURE

Open it now:

**→ [`figures/rule-processor-placement.html`](figures/rule-processor-placement.html)**

It is a **derivation bench, not an answer key**. Read top to bottom without touching anything
and it gives you nothing — that is deliberate and it is enforced in the figure's own code.

Here is what to do with it, in order.

**PANEL ONE — place both blocks.** You get an empty pipeline containing only
`transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator`
and two unplaced blocks. Drop them anywhere. The tool will never tell you "correct." What it
does instead is replay one Korean utterance and report **what each processor actually
received at the position you chose** — which frame types arrived, and what the rule code
would have been handed. Put a block before `stt` and watch a text-matching rule receive
`InputAudioRawFrame`. Put one after `llm` and watch the model finish generating before your
`Inject` gets a vote. Wrong positions are more instructive than right ones here, so try at
least four.

**The reveal button is disabled until both blocks are placed.** That is not a bug and there
is no keyboard shortcut around it. The mapping table sits behind that gate, and when it opens
it is labelled **CHECK YOUR DERIVATION** — because by then you will have made one.

**PANEL TWO — drag the interception marker.** This is Q1 rendered as a timeline: `add_message`
at `:863`, `push_context_frame` at `:866`, `on_user_turn_message_added` at `:871`, and on the
service side `:601`, `:603`, `:604`, `:605` — with the `:601 → :605` window drawn containing
the frame that already escaped at `:603`. Position the marker yourself. The panel computes
**CAN VETO / CANNOT VETO** from where you dropped it. Drop it deliberately on `:871` and watch
Rival B die. **You must compute a veto verdict here at least once**, because PANEL FOUR's code
view stays locked until you have.

**PANEL THREE — run the two-phase commit twice.** Four layers stage actions, the last one
fires `Filter`, and you watch boson's identity-based rollback delete the appended user
message. Then re-run the identical round as four separate Pipecat processors. Every frame that
already escaped `push_frame` is marked in red. Count the red. That count is your answer to Q2,
and it is a number, not an opinion.

**PANELS FOUR, FIVE, SIX** are the check — the live `process_frame` body with a
"delete the `super()` call" switch that reproduces [[ch-01/read]]'s black hole, the latency
bill dropped into ch-11's empty slot, and the transition race counted at the `llm` input.
They are the interactive twin of STEP 4. Do them after you have read STEP 4, or alongside it.

**Before you scroll past this line, you should have on paper:** one seam position with two
named justifications, two knocked-out rivals with the line number that kills each, and a
yes/no on the collapse with the reason.

---
---

# STEP 4 — CHECK YOUR DERIVATION

Everything from here is a **check**, not a lesson.

The standing rule for the rest of the chapter: **if your answer disagrees with something
below, one of the two is wrong, and the text names the source line that settles it.** Go to
the line. Not to me, and not to the excerpt — the excerpts are pre-read summaries and this
chapter corrects four of them.

---

## 5. CHECK — Conclusion (a): the seam

**The seam is downstream of the user aggregator and upstream of the LLM service.**

There is exactly one such position, and both walls are data dependencies.

**The downstream wall (why not earlier).** Before the aggregator, the frames in flight are
`TranscriptionFrame` and `InterimTranscriptionFrame` — fragments. Constraint One says a rule
handed a fragment does not fail, it answers wrong: `kw in user_message.lower()` in
`end_signal.py` fires on a prefix, and the LLM intent-matcher's prompt — anchored on
*"Most recent turn (PRIMARY SIGNAL — evaluate against THESE)"*
(`intent_matcher.py:205-271`, via [[design-boson-rules-on-pipecat]]) — is handed half a
sentence and scores it. The complete turn does not exist until `:863` writes it into the
context.

**The upstream wall (why not later).** After the LLM service, `:605` has run. `Inject` exists
to steer the generation it precedes; after the generation there is nothing left to steer.
`Respond` and `Filter` degrade into cancel-and-redo. And per §4.3, `:603` already pushed.

**So: after `:866` pushes, before `:601` tests.** In pipeline terms, that is one processor
slot, between the user aggregator and the LLM service.

**Rollback is real at that position, and only at that position.** You hold a reference to the
same `LLMContext` the aggregator wrote to at `:863` — `LLMContextFrame.context` is that object
([[ch-09/read]] §3: the frame is a doorbell, not a parcel). Snapshot the message list,
run the rules, and on veto call `set_messages(snapshot)`. Nothing downstream has seen anything
yet, because you have not pushed.

### 5.1 Rival B is dead, and one line kills it

`on_user_turn_message_added` fires at **`:871`**. `push_context_frame()` runs at **`:866`**.

Five lines earlier the frame left. By the time your event handler is invoked, the
`LLMContextFrame` is on the LLM service's input queue and quite possibly already past `:603`.
An event handler at `:871` is a **notification**, not a gate. It can observe; it can never
veto.

That is not a stylistic objection. It is arithmetic on two line numbers in one function body,
and it is why [[design-boson-rules-on-pipecat]] calls the rejection *"decisive rather than
stylistic."*

**What Rival B is still good for**, and this is worth keeping: the observe-only rules.
`end_signal`, `turn_counter`, `stage_round_tracker` do not veto — they classify and write
findings onto the session. Those could legitimately live in an `on_user_turn_message_added`
handler and cost the critical path zero. If you ever need to shave the bill in §11, that is
where the knife goes first.

### 5.2 Rival C is dead, and it costs two different things

Letting generation start and cancelling looks free because on the pass path — most turns — it
adds nothing. It is not free, and the two costs are different in kind:

**Cost 1, latency.** `broadcast_interruption()` (`frame_processor.py:1017-1022`) aborts and
you start over. You pay the first generation's TTFB and then pay a second one. Against
ch-11's budget that is not a small tax; it is roughly a doubled LLM leg on every vetoed turn.

**Cost 2, and this one is not a latency cost at all.** Between `:605` and your veto, tokens
have been generated, streamed to TTS, and possibly rendered to audio on the wire. For a
Korean insurance tele-sales agent reading regulated consent script text, "the bot said half a
sentence it should not have said, then stopped" is not a latency regression. It is a
compliance event. [[boson-script-engine]] is explicit about why the script bypasses the model
at all: *"Korean insurance-consent script text is legally fixed."* A veto that arrives after
audio has played is not a veto.

> **If your derivation put the seam anywhere else**, the two lines that settle it are
> `llm_response_universal.py:866` (the push, which is what makes anything after it a
> notification) and `openai/base_llm.py:601` (the test, which is what makes anything after it
> a cancellation). Everything between those two is the seam. Nothing outside them is.

---

## 6. CHECK — Conclusion (b): all layers must collapse into one processor

**Yes, the layers must collapse. Cross-layer veto cannot survive being spread across N
processors.** [[custom-processor-guide]]'s "the honest port splits its four layers into four
`FrameProcessor`s" is wrong, and Constraints Two and Three are why.

Here is the proof in the shape of Q2's scenario.

**(a) What has already left processor 03?** For processor 04 to be evaluating anything at all,
processor 03 must have pushed. That is the only way a frame gets from 03 to 04 — `push_frame`,
which is enqueue-on-neighbour and irreversible (Constraint Two). So the answer to Q2(c) —
the hinge — is: **processor 04 cannot be running unless processor 03 has already committed.**
The two events are not merely ordered, they are causally chained. "Layer 3 stages and waits for
layer 4's verdict" has no expression in a linked list of processors, because the mechanism that
gets data to layer 4 *is* the commit.

Worse, layer 03's actions are not only frames. `Inject` mutates the shared `LLMContext` by
calling `add_message` — [[ch-09/read]]'s central finding is that Pipecat optimises for **many
holders of one object**, so that mutation is visible to everyone the instant it happens, with
no frame involved and no push to withhold. `PreTool` appends synthetic tool-call history. By
the time 04 says `Filter`, those writes are in the shared context and there is nothing to
un-push.

**(b) Does the rollback verb have the right shape?** No, and this is the sharper half.

| | boson | Pipecat |
|---|---|---|
| what is deleted | *this exact object*, found by walking `session.messages` backwards comparing identity | — |
| the verb | identity delete of `pipeline_user_message` (`gateway/layers/pipeline.py:205-239`) | `set_messages(list)` (`llm_context.py:377`) |
| handle available | a Python object reference held since `:154-156` | none — the API takes a list, returns nothing, and hands out no handles |

`set_messages` is *replace the whole list*. It is expressible as "restore this snapshot,"
which is a coarser operation than "delete this object." The coarse version is only equivalent
when **one** party is mutating between snapshot and restore. Spread across four processors,
between processor 01's snapshot and processor 04's restore you have three other processors, an
LLM service that may already hold the same context object, and the assistant aggregator. A
whole-list restore there does not undo layer 04's mistake — it clobbers everyone's writes
back to a moment that is no longer meaningful.

**(c) The count from PANEL THREE.** Run the four-processor version and count the red frames.
Every red frame is one thing `set_messages` cannot recall.

### 6.1 What the collapse actually costs you

Be honest about the price rather than pretending there isn't one.
[[custom-processor-guide]] is right about the cost even though it is wrong about the
conclusion. Collapsing into one processor forfeits:

- **Per-layer queueing.** Four processors give you four `FrameProcessorQueue`s and four
  independent process tasks. One processor gives you one. Layer 03's LLM checks block layers
  02 and 04 in the same coroutine.
- **Per-layer observability.** [[ch-11/read]]'s observer plane fires `on_process_frame` per
  *processor*. Four processors give you four named rows in a trace; one gives you one row
  called `BosonRuleProcessor` and the rest is your own logging.
- **Per-layer interruption semantics.** `super().process_frame` clears state on
  `InterruptionFrame` per processor (`frame_processor.py:839-841`). One processor means one
  reset, and any per-layer state you wanted independently cleared is your problem.

Those are real costs. They are not *correctness* costs. Cross-layer veto is a correctness
property, and there is exactly one way to keep it.

> **If your derivation said "four processors, and I'll coordinate them with an
> `EventNotifier`"** — that is the right instinct and the wrong problem. §13 shows where the
> notifier pattern genuinely is Pipecat's answer. It coordinates *notification*. It does not
> give processor 03 a way to un-push.

---

## 7. CHECK — the mapping table

**Eleven rows.** Each is a claim you can disagree with; the right-hand column names the line
that settles the disagreement.

| # | boson mechanism | Pipecat home | what is lost | line that settles it |
|---|---|---|---|---|
| 1 | `@check(mode="sequential")` + first-non-continue short-circuit | processor internals, **ported verbatim** | nothing | Pipecat ships no rule scheduler at all — there is nothing to fight. `rules/engine.py:68-69` runs unchanged inside `process_frame`. |
| 2 | `@check(mode="parallel")` under one `asyncio.gather` | same processor, **the same `asyncio.gather`** | nothing | `rules/engine.py:74-80`. You are inside a coroutine; `gather` works exactly as it did. |
| 3 | `Respond(text)` | push `TTSSpeakFrame(text)`, **swallow** the context frame | nothing | `frames.py:795-809` — `TTSSpeakFrame(text: str, append_to_context: bool = True)`. Flows' `tts_say` action is the wrong tool: it fires only at node-set time (`actions.py:104`), which is far too coarse for a per-turn scripted line. |
| 4 | `Inject(content)` | `context.add_message({...})` **before** pushing | the merge-into-the-last-user-message option | `llm_context.py:361` `add_message` appends. boson's `_merge_system_reminder` (`gateway/layers/pipeline.py:341-372`) folds `<system-reminder>…</system-reminder>` *into* the most recent user message — there is no frame or context verb that does that. Append or rewrite the whole list; there is no third move. |
| 5 | `PreTool(name, args, preamble)` | a Flows `function` action in `pre_actions` — **exactly right** | the preamble-as-first-stream-chunk | `actions.py:279-285`: `elif previous_action_type == "function": … needs_wait = True`, unconditionally. `"function"` actions **always** wait, which is precisely boson's synchronous-before-generation semantics. The preamble (`["감사합니다!"]`) demotes to a separate `tts_say` ordered before it. |
| 6 | `Compact()` | a `function` pre-action calling **boson's** compactor | background-async compaction; a pre-action blocks | Do **not** use Flows' summary path. `manager.py:801-822`: `asyncio.wait_for(..., timeout=5.0)`, and on `TimeoutError` it logs a warning and sets `update_config.strategy = ContextStrategy.APPEND` — it **silently degrades to append**. A compaction that silently doesn't compact is worse than no compaction. |
| 7 | `_GLOBAL_TOOLS` | `FlowManager(global_functions=[...])` | nothing — exact match | `manager.py:100` declares the kwarg; `manager.py:654` mixes it in: `functions_list = self._global_functions + node_config.get("functions", [])`. Global-then-node, on every node. |
| 8 | `ScriptEngine.process_turn(state, msg, registry)` | runs **unchanged** inside the processor | nothing — the cleanest port in the system | It is all `@staticmethod`, takes a dict and returns `(new_state, Action)`, and never mutates its input ([[boson-script-engine]]). Zero gateway coupling. Put `script_state` on the processor (or `flow_manager.state`, `manager.py:157`) and call it. |
| 9 | `Continue()` | **no home as an action** | — | An action runs *after* the round resolves, and by then the user message is in the context (`:863`). "Keep going" is not an effect, it is the absence of one — it belongs in the processor's control flow, not in an action list. |
| 10 | `Pass()` | **no home as an action** | — | Same reason as row 9. `Pass()` is this layer declining to vote. There is no frame that means "I decline"; you express it by not branching. |
| 11 | `Filter(reason)` | **no home as an action** — it is a processor verdict | cannot un-broadcast a VAD interruption already sent | This is the important one. A `Filter` is a *pre-LLM routing verdict*: don't push, restore the snapshot, return. Modelling it as an action means it executes after the round, and per `:863` the message is already committed by then. Also: do **not** reach for `FunctionFilter` (`filters/function_filter.py:21`) — it decides pass/block on a predicate and has no way to do the context rollback. |

**Read rows 9, 10 and 11 together.** The three actions with no home are exactly the three
that are *routing verdicts* rather than effects. That is not a coincidence and it is the
generalisation worth carrying: **boson's `Action` type conflates "what to do" with "whether to
proceed," and the Pipecat port splits them.** Effects become frames or context writes;
verdicts become control flow inside one `process_frame`.

**What is deliberately not a row.** `StageTransition(target)` is absent, because §1 already
settled it — Fact Zero, `manager.py:588`, called from plain code. It is not a mapping
question any more; it is a *sequencing* question, and §9.4 handles it.

Also absent, and you should know they are absent rather than discover it in month three:
`skills` (`StageDefinition.skills`) has no Pipecat concept whatsoever; per-session attribute
namespaces (`SharedLayerContext.__getattr__`/`__setattr__`) have none, since `flow_manager.state`
is a bare `dict[str, Any]` (`manager.py:157`); and `TOOL_PROCESSING` has no frame that means
it, so boson's `AgentStatusTracker` three-state model degrades to two states driven by
`BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame`.

---

## 8. CHECK — the pipeline

```python
pipeline = Pipeline([
    transport.input(),
    stt,                       # Korean 8 kHz telephony STT
    BosonFillerGate(),         # boson layer 01
    user_aggregator,           # LLMContextAggregatorPair(context).user()
    BosonRuleProcessor(...),   # boson layers 02 / 03 / 04, Tier 1 + Tier 2
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])

# FlowManager(llm=llm, context_aggregator=pair, worker=worker,
#             global_functions=[...])  is NOT in this list.
```

Nine entries. Every non-obvious one is justified by a data dependency, not by taste. If your
list differs, the argument is below and it is falsifiable.

### 8.1 `BosonFillerGate` between `stt` and `user_aggregator`

Two walls again.

**Not earlier.** Before `stt`, the only frames are `InputAudioRawFrame`. `_is_filler_text()`
takes a string. Put the gate at position 1 and the rule has no input — it is not degraded, it
is **dead**. PANEL ONE shows you exactly this if you drop it there.

**Not later.** After the aggregator, `:863` has already run: `"네"` is `add_message`d into the
shared context and `:866` has already pushed. A one-line string membership test has become a
context rollback plus a swallowed frame. You have converted the cheapest rule in the system
into the most expensive one for no reason.

**And it must drop `InterimTranscriptionFrame` unconditionally**, or every keyword rule
downstream fires on prefixes (Constraint One(c)). Pipecat's own `SentenceAggregator` does this
with a bare `return` at `aggregators/sentence.py:50-51`; copy that.

### 8.2 `BosonRuleProcessor` between `user_aggregator` and `llm`

This is Conclusion (a) rendered as a list index. `push_aggregation` writes at `:863` and
pushes at `:866`; `base_llm.py:601` consumes. Your processor goes in the gap. §5 is the
argument; this is just where it lands.

### 8.3 `assistant_aggregator` after `transport.output()`

This is the Pipecat house pattern and [[ch-09/read]] already explained Pipecat's reason. What
is worth naming is that **boson has the same invariant for a different reason**: a `Respond()`
that was interrupted mid-TTS must not be recorded as having been spoken. Move the assistant
aggregator before `transport.output()` and the history records text the customer never heard —
which is exactly the drift boson's identity-based rollback exists to prevent, arriving through
a different door. Two systems, two rationales, one ordering. That agreement is a good sign.

### 8.4 On a transition turn, the rule processor SWALLOWS the context frame

This is the counterintuitive one and it reverses the advice you would give from first
principles.

The naive fix for "two things might trigger inference" is *ordering*: call
`set_node_from_config()` first, then push your frame, and let the node's `LLMRunFrame` win the
race. **That does not work**, and the reason is structural:

**`src/pipecat/flows/manager.py:838-841`**

```python
            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))

            await self._worker.queue_frames(frames)
```

`queue_frames` on the **worker**. [[ch-10/read]]'s central finding is that `FlowManager` is not
a `FrameProcessor` — it drives the pipeline from outside by injecting at the **head**. So the
node's `LLMMessagesAppendFrame`, `LLMSetToolsFrame` and `LLMRunFrame` enter at
`transport.input()` and must traverse `stt` → `BosonFillerGate` → `user_aggregator` before they
reach `llm`. Your pushed `LLMContextFrame` is already adjacent to `llm` and reaches it
immediately.

Ordering your two calls does not order their arrivals. They travel different distances.

**Swallowing removes the race by construction.** On a transition turn: mutate the context,
call `set_node_from_config`, and `return` **without pushing**. Now exactly one
inference-triggering frame exists in the system, and it is the node's `LLMRunFrame` at
`manager.py:707-709`:

```python
            respond_immediately = node_config.get("respond_immediately", True)
            if respond_immediately:
                await self._worker.queue_frames([LLMRunFrame()])
```

On non-transition turns, push normally.

The failure modes to test for are **both** directions — §14's second prototype counts them:
swallow both paths and you get **zero** generations (the bot goes silent mid-call); push and
transition and you get **two** (the bot answers itself).

---

## 9. CHECK — the code

Naming a class is not designing one. Here is the body, and it is the shape
[[custom-processor-guide]] has been building toward for eleven chapters — *"four lines of
ceremony around one `if isinstance(...)` chain."*

### 9.1 The companion gate

**`BosonFillerGate` — boson layer 01, ~16 lines**

```python
class BosonFillerGate(FrameProcessor):
    """boson layer 01: drop Korean backchannels before they reach the aggregator."""

    def __init__(self, *, session: SessionState, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)           # 1  never optional

        if isinstance(frame, InterimTranscriptionFrame):        # 2  Constraint One(c)
            return                                              #    unconditional drop

        if isinstance(frame, TranscriptionFrame):
            status = self._session.pre_turn_status               # 3  korean_fillers.py:66
            if status in ("generating", "tool_processing") and _is_filler_text(frame.text):
                self._signals.add(Signal(reason=f"filler_word:{frame.text}"))
                return                                          # 4  drop == not pushing

        await self.push_frame(frame, direction)                 # 5  everything else
```

Line 2 is not optional and it is not a performance tweak. `InterimTranscriptionFrame`
subclasses `TextFrame` (`frames.py:476`), so without that `return` the interim hypothesis
`"네"` of the sentence `"네 그런데…"` is filtered as a backchannel and the customer's question
disappears. Constraint One(c), enforced in one line.

Line 3 is the `pre_turn_status` read, and it is why the gate is stateful rather than a
`FunctionFilter`. `korean_fillers.py:66` gates on it, and if you drop that gate the filter
self-filters — it eats genuine `"네"` answers to consent questions, which in a Korean
insurance sale is the single worst thing this file could do.

### 9.2 The rule processor

**`BosonRuleProcessor.process_frame` — ~40 lines**

```python
class BosonRuleProcessor(FrameProcessor):

    def can_generate_metrics(self) -> bool:                      # see §11.4 — not optional
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)            #  1  literally first

        if not isinstance(frame, LLMContextFrame):               #  2  the only frame we own
            await self.push_frame(frame, direction)              #     everything else passes
            return

        ctx = frame.context                                      #  3  the SAME object :863 wrote
        snapshot = list(ctx.get_messages())                      #  4  COPY — see §9.3
        user_message = _text_of(snapshot[-1])                    #  5  the turn :863 appended

        await self.start_ttfb_metrics()                          #  6  the bill — see §11.4

        staged: list[Action] = []                                #  7  boson Phase 1
        for engine in self._engines:                             #     02-analyzer, 03, 04
            decision = await engine.evaluate_sequential(         #  8  TIER 1: 11 checks
                snapshot, user_message, self._ctx_for(engine),   #     pure Python, sub-ms
            )
            staged.extend(decision.actions)
            if decision.short_circuit:                           #  9  engine.py:68-69, verbatim
                break

        if not _has_veto(staged):                                # 10  TIER 2: the 2 llm checks
            results = await asyncio.gather(*[                    # 11  engine.py:74-80, verbatim
                check(snapshot, user_message, self._ctx_for(check))
                for check in self._llm_checks                    #     intent_rules, sentiment
            ])                                                   #     wall clock = max, not sum
            staged.extend(_flatten(results))

        await self.stop_ttfb_metrics()
        verdict = resolve_actions(staged)                        # 12  ACTION_PRIORITY, unchanged

        if verdict.filter is not None:                           # 13  THE VETO
            ctx.set_messages(snapshot[:-1])                      # 14  rollback — see §9.3
            self._signals.add(Signal(reason=verdict.filter.reason))
            return                                               # 15  no push == no inference

        if verdict.respond is not None:                          # 16  script text, verbatim
            ctx.set_messages(snapshot[:-1])                      #     the LLM never sees this turn
            ctx.add_message({"role": "assistant", "content": verdict.respond.text})
            await self.push_frame(TTSSpeakFrame(verdict.respond.text,
                                                append_to_context=False))
            return                                               # 17  swallow the context frame

        for inject in verdict.injects:                           # 18  mutate in place, then fall
            ctx.add_message({"role": "developer", "content": inject.content})

        if verdict.transition is not None:                       # 19  Fact Zero, cashed
            result = self._stage_machine.transition(             # 20  boson legality PRE-CHECK
                self._session.active_stage, verdict.transition.target,
            )
            if result.success:
                self._session.active_stage = result.new_stage.name
                await self._flow_manager.set_node_from_config(   # 21  manager.py:588
                    self._node_for(result.new_stage.name),
                )
                return                                           # 22  SWALLOW — see §8.4
            logger.warning(f"illegal transition rejected: {result.error}")

        await self.push_frame(frame, direction)                  # 23  the ordinary turn
```

### 9.3 Two lines in that body that are not what the excerpts say

> **Source correction, and it is the difference between a rollback and a no-op.**
> [[design-boson-rules-on-pipecat]] §4 writes the snapshot as *"snapshot then
> `context.set_messages(snapshot)`"*, and the figure spec writes it as
> `snapshot = frame.context.get_messages()`. **Open `llm_context.py` and that is a bug.**
>
> **`src/pipecat/processors/aggregators/llm_context.py:244-260`**
>
> ```python
>         if llm_specific_filter is None:
>             messages = self._messages
>         else:
>             messages = [ ... ]
>         if truncate_large_values:
>             messages = LLMContext._truncate_large_values_from_messages(messages)
>
>         return messages
> ```
>
> With no filter and no truncation, `get_messages()` returns `self._messages` itself — the
> **live list**, exactly as [[ch-09/read]] §2.3 established, and deliberately so. And
> `set_messages` is `self._messages[:] = messages` (`:383`).
>
> Put those together. `snapshot = ctx.get_messages()` binds `snapshot` to the same list object
> as `ctx._messages`. Any `add_message` during your round appends to *the snapshot too*. Then
> `ctx.set_messages(snapshot)` slice-assigns a list onto itself. **The rollback silently
> restores nothing** — no exception, no warning, a veto that does not veto.
>
> `list(...)` at line 4 is not defensive style. It is the difference between working code and
> a two-phase commit that quietly commits.

Second: **line 14 is `snapshot[:-1]`, not `snapshot`.** boson's `Filter` deletes the appended
user message by object identity (`gateway/layers/pipeline.py:205-239`). Your processor first sees the
context *after* `:863` appended it, so `snapshot[-1]` **is** that message — position recovers
what identity gave boson. `set_messages(snapshot)` would undo your round but leave the user's
utterance in the history; `set_messages(snapshot[:-1])` matches boson.

Be honest about what that costs: the positional recovery is correct only because you are the
only writer between `:863` and here, which is true *because the layers collapsed*. Row (b) of
§6 said identity is not expressible as a list replace **in general**. This is the special
case where it is — and the collapse is what creates the special case. If you had kept four
processors, `snapshot[-1]` would be a guess.

### 9.4 The two-tier split, and why line 10 is there

Line 8 is **Tier 1**: eleven deterministic checks — `korean_filler_filter`, `end_signal`,
`response_classifier`, `hesitation_hook`, `turn_counter`, `stage_round_tracker`,
`preload_on_question`, `script_flow`, `tool_gate`, `help_responder`, `auto_compact`. Pure
Python over strings and dicts. Sub-millisecond. Free.

Line 11 is **Tier 2**: the two `check_type="llm"` checks — `intent_rules` (priority 30) and
`sentiment_tracker` (priority 10), both `mode="parallel"` in boson today
([[design-boson-rules-on-pipecat]] §1). Under one `gather`, so wall clock is `max`, not `sum`.
That is a boson design decision you inherit for free, and it halves the bill.

Line 10 — `if not _has_veto(staged)` — is not in boson. It is an optimisation the port makes
available: if Tier 1 already produced a `Filter` or a `Respond` (`ACTION_PRIORITY` 0 and 1),
Tier 2's result cannot change the outcome, so skip it and save 250-400 ms. On a filler turn —
which for Lina is a large fraction of turns, since customers say `"네"` constantly — you pay
zero. Verify against boson's arbitration before you ship it: `RuleEngine.evaluate` keeps every
non-continue parallel result and arbitrates transitions across phases (`engine.py:100-127`),
so skipping Tier 2 changes *which signals get written*, not which action wins. If a later
layer reads `session.sentiment`, you have changed behaviour. Decide deliberately.

### 9.5 Line 1, one more time

`await super().process_frame(frame, direction)` is the literal first statement, and skipping
it is not a style violation — it is a black hole. The base implementation is not a no-op:

**`src/pipecat/processors/frame_processor.py:827-847`**

```python
        observer = self._setup.observer if self._setup else None
        if observer:
            data = FrameProcessed(
                processor=self,
                frame=frame,
                direction=direction,
                timestamp=self.get_clock().get_time(),
            )
            await observer.on_process_frame(data)

        if isinstance(frame, StartFrame):
            await self.__start(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
        elif isinstance(frame, CancelFrame):
            await self.__cancel(frame)
        elif isinstance(frame, (FrameProcessorPauseFrame, FrameProcessorPauseUrgentFrame)):
            await self.__pause(frame)
        elif isinstance(frame, (FrameProcessorResumeFrame, FrameProcessorResumeUrgentFrame)):
            await self.__resume(frame)
```

Skip it and: your processor never starts its internal tasks (`StartFrame` → `__start`), never
clears state on barge-in (`InterruptionFrame` → `_start_interruption`), never stops metrics,
and is **invisible to every observer** — which means invisible to ch-11's entire observability
plane. Note also what it does *not* do: **it does not push the frame.** Forwarding is entirely
your job, which is why line 2's `else`-shaped branch exists.

PANEL FOUR of the figure has a "delete the `super()` call" switch. Flip it and watch
[[ch-01/read]]'s black hole reproduce in your own processor.

---

## 10. Where the code is honest about what is missing

Three things in §9.2 are placeholders for boson code that has no Pipecat counterpart, and you
should recognise them as such rather than as helper functions I did not bother to write.

**`self._ctx_for(engine)` — `SharedLayerContext` has no home.** boson's per-turn proxy
(`layers/context.py:17`) forwards unknown attribute reads at `:40` and writes at `:54`
straight through to the live `SessionState`, so `session.checklist_state`,
`session.fired_rules`, `session.script_state` are read/write inside a rule and persist. Pipecat
offers `flow_manager.state`, a bare `dict[str, Any]` (`manager.py:157`). **The zero-edit choice
is to keep a real `SessionState` object on the processor and pass it as the `session`
argument.** Rewriting thirteen rules' `getattr(session, …)` to dict lookups buys you nothing
and costs you thirteen files of diff.

**`self._signals` — `SignalQueue` stays a plain object.** `get_recent(seconds, source_layer)`
(`layers/signals.py:63`) is a **query** over an append-only list. §13 explains at length why
that verb has no bus equivalent.

**`self._stage_machine` — kept as a pure validator.** Lines 19-22 call
`StageMachine.transition()` *before* `set_node_from_config`. Flows has no from→to check
anywhere: `_validate_node_config` (`manager.py:867-898`) checks exactly two things — that
`task_messages` is present and that each `functions` entry is callable or a
`FlowsFunctionSchema`. There is no node registry, and `get_or_generate_node_name` falls back to
`str(uuid.uuid4())` for an unnamed node. Drop boson's `StageMachine` and you drop **all**
transition legality, silently. §12.1 has the detail.

---

## 11. THE LATENCY BILL — now priceable, because ch-11 built the budget

[[ch-11/read]] left a slot in the budget labelled *rule evaluation* and left it empty because
nothing had filled it yet. Fill it.

### 11.1 Tier 1 is free

Eleven pure-Python checks over a string and a list of dicts. Substring tests, dict lookups,
a `re.split`. Sub-millisecond, and it does not appear in a budget denominated in hundreds of
milliseconds. Ignore it.

### 11.2 Tier 2 is the bill: ~250-400 ms on every turn

Two checks, both `check_type="llm"`, both `mode="parallel"`, both under one `gather`.
`evaluate_intent_rules` (`transition_detector.py:82`) calls `_llm_match_descs`
(`intent_matcher.py:205-271`) — **one batched call**, model `("boson", "Qwen3.6-27B-FP8")` at
`temperature=0.1` (`llm_config.py:20,34`), output a comma-separated index list or `"none"`
(~5 tokens). `sentiment_tracker` fires concurrently. So the cost is **one Qwen3.6-27B TTFB
plus a handful of output tokens ≈ 250-400 ms**, and because they are gathered it is `max`, not
`sum`.

### 11.3 Scale it against something real

Pipecat ships measured STT references:

**`src/pipecat/services/stt_latency.py:37-46`**

```python
# Conservative fallback for services without measured values
DEFAULT_TTFS_P99: float = 1.0

# Measured P99 TTFS latency values (in seconds)
ASSEMBLYAI_TTFS_P99: float = 0.42
AWS_TRANSCRIBE_TTFS_P99: float = 1.90
AZURE_TTFS_P99: float = 1.80
CARTESIA_TTFS_P99: float = 0.81
DEEPGRAM_TTFS_P99: float = 0.35
DEEPGRAM_SAGEMAKER_TTFS_P99: float = 0.35
```

> **Source correction.** [[design-boson-rules-on-pipecat]] §3 cites *"0.45 s for Deepgram."*
> The **shipped constant is `0.35`** (`stt_latency.py:45`). `0.45` appears exactly once in
> that file, at `:27`, inside a docstring showing how to pass a *measured* value:
> `stt = DeepgramSTTService(api_key="...", ttfs_p99_latency=0.45)` — an illustration, not a
> reference. Use **0.35**. The chapter spec's "0.35-0.45 s" range covers both; the number the
> code will actually use if you do nothing is 0.35.

Now the arithmetic. On the pre-LLM half of the budget:

| leg | cost | source |
|---|---|---|
| STT finalisation (Deepgram reference) | 0.35 s | `stt_latency.py:45` |
| Korean 8 kHz telephony STT | **no entry in the table** | must be benchmarked — [[ch-06/read]] §19 |
| Tier 1 rules | < 0.001 s | 11 pure-Python checks |
| **Tier 2 rules** | **0.25 – 0.40 s** | one Qwen3.6-27B TTFB, gathered |

**Tier 2 roughly doubles the pre-LLM half.** Not a rounding error — a second STT's worth of
delay, added to every single turn, before the sales LLM has seen a token.

### 11.4 Make it show up in `LatencyBreakdown`, and note what the excerpt got wrong here

The chapter spec says to wrap the processor in
`start_processing_metrics()`/`stop_processing_metrics()` "or the bill never appears in
`LatencyBreakdown`." **That is half right and the half that is wrong will cost you an
afternoon.** Two independent gotchas, both verified:

**Gotcha 1 — `LatencyBreakdown` never reads processing metrics.**

**`src/pipecat/observers/user_bot_latency_observer.py:322-341`**

```python
        now = time.time()
        for metrics_data in frame.data:
            if isinstance(metrics_data, TTFBMetricsData) and metrics_data.value > 0:
                self._ttfb.append(
                    TTFBBreakdownMetrics(
                        processor=metrics_data.processor,
                        model=metrics_data.model,
                        start_time=now - metrics_data.value,
                        duration_secs=metrics_data.value,
                    )
                )
            elif isinstance(metrics_data, TextAggregationMetricsData):
```

`TTFBMetricsData` and `TextAggregationMetricsData`. That is the whole list.
`ProcessingMetricsData` exists (`metrics/metrics.py:99`) and is never read by this observer.
`LatencyBreakdown`'s own fields confirm it (`:107-111`): `ttfb`, `text_aggregation`,
`user_turn_start_time`, `user_turn_secs`, `function_calls`. **There is no processing slot.**

So use `start_ttfb_metrics()` / `stop_ttfb_metrics()` — which is what §9.2 line 6 does. It is
a slight abuse of the name (nothing is being byte-streamed) but it is the only channel the
breakdown observer listens on, and the result is that your rule processor appears in
`chronological_events()` (`:113-140`) alongside `stt`, `llm` and `tts`, sorted by start time,
with a `processor` label you chose.

**Gotcha 2 — a plain `FrameProcessor` emits no metrics at all.**

**`src/pipecat/processors/frame_processor.py:488-494`**

```python
    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False
```

`False`. And every metrics method is guarded on it — `start_ttfb_metrics` at `:511`,
`stop_ttfb_metrics` at `:528`, `start_processing_metrics` at `:570`, all with
`if self.can_generate_metrics() and self.metrics_enabled:`. Without the override in §9.2, your
`start_ttfb_metrics()` call is a **silent no-op** and the bill never appears anywhere.

Three things, all required, or you are flying blind:

1. `def can_generate_metrics(self) -> bool: return True` on your processor.
2. `enable_metrics=True` in `PipelineParams` (`metrics_enabled` reads
   `self._setup.enable_metrics`, `frame_processor.py:419`).
3. `start_ttfb_metrics` / `stop_ttfb_metrics`, not the processing pair.

### 11.5 The sentence you owe the product owner

The alternative to paying is running Tier 2 *concurrently with the LLM's first tokens* and
calling `set_node_from_config()` when it completes — which makes every stage change land
**one turn late**. You cannot veto a turn you have already answered, so that variant also
gives up `Filter` and `Respond` in-turn.

State the trade in one sentence and do not decorate it:

> **In-turn veto and in-turn steering cost 250-400 ms on every turn. Next-turn transitions
> cost 0 ms and give up the veto. boson pays the 250-400 ms today.**

PANEL FIVE of the figure has the switch. Flip it and watch the bar go to zero and the stage
change slide one turn right.

Two ways to shave it that do not require choosing:

- **§5.1's leftover.** Move the three observe-only rules (`end_signal`, `turn_counter`,
  `stage_round_tracker`) into an `on_user_turn_message_added` handler. They cannot veto
  anyway, and off the critical path they cost zero.
- **§9.4's line 10.** Skip Tier 2 when Tier 1 already vetoed. On filler-heavy Korean
  tele-sales traffic that is a large fraction of turns at zero behavioural cost — *if* you have
  verified nothing downstream reads the signals Tier 2 would have written.

---

## 12. FOUR REAL FRICTIONS, none fatal

These are the things that will bite in month two if you do not write them down in month zero.

### 12.1 Transition legality is lost

boson: `StageDefinition.transitions` is a whitelist of legal successors, and
`StageMachine.transition()` returns
`TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")`
(`stage/machine.py:57-60`, via [[boson-stage-machine]]).

Flows: nothing. `_validate_node_config` (`manager.py:867-898`) checks two things and neither
is a from→to test. There is no node registry. `self._current_functions: set[str]`
(`manager.py:148`) is assigned at `:704` and — per [[flows-state-machine]] — **never read
anywhere in the codebase**; it is dead state, not a gate.

**Mitigation, and it is cheap:** keep boson's `StageMachine` as a pure pre-check validator
sitting in front of `set_node_from_config` (§9.2 lines 19-22). Throw away its prompt and tool
plumbing, keep its edge whitelist. [[boson-stage-machine]]'s guideline is worth repeating
verbatim here: *"port the edge whitelist first, not the prompts."* And note the failure mode
it names — every `v0.7.5 (#12)` regression comment in `stage_config.py` is a rule emitting a
transition the whitelist silently rejected, *"invisible in logs unless you check
`TransitionResult.error`."* Line 22's `logger.warning` is not decoration.

### 12.2 Every Flows transition is an inference trigger by default

`respond_immediately` defaults to `True` (`flows/types.py:203-204, 237`). So a rule-driven
transition fired on the user's turn queues an `LLMRunFrame()` at `manager.py:707-709` and the
bot **speaks**. boson's transition is silent bookkeeping that happens *before* the agent loop
runs.

**Mitigation:** ported nodes need `respond_immediately=False` — except on the turns where you
deliberately want the node to be the sole inference trigger (§8.4). That is a per-node
decision, not a global setting, and getting it wrong in either direction is §14's second
prototype.

### 12.3 There is no silent transition at all

Even with `respond_immediately=False`, a node set is never free:

**`src/pipecat/flows/manager.py:838-839`**

```python
            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))
```

`LLMSetToolsFrame` is appended **unconditionally**. And if the node declares no functions,
`formatted_tools` is `NOT_GIVEN` (`manager.py:670-672`) and the node **clears the tool set**.
Meanwhile `task_messages` is a `Required[list[dict]]` key (`types.py:224`).

The closest thing to a no-op node is `task_messages=[]`, which passes validation because
`_validate_node_config` only checks key *presence*. It still emits an
`LLMMessagesAppendFrame(messages=[...])` and an `LLMSetToolsFrame`.

**Consequence for Lina:** this collides head-on with boson's `_allowed_tools_var` ContextVar
tool gate in `metatool/router.py` ([[boson-stage-machine]] flags the collision). Two systems
both authoritative about the advertised tool array, one of them re-emitting on every node set.
**Pick one.** If Flows owns tools, delete the ContextVar gate; if the router owns tools, never
put `functions` on a node.

### 12.4 Ordering becomes queue ordering

boson calls `core._apply_stage_transition` **inline**, before `run_agent_loop`. Call ordering
is execution ordering. Flows queues frames at the pipeline head (`manager.py:841`), so an
in-pipeline processor and the flow's own frames race through the same queue rather than
executing in call order.

This is the general form of §8.4's specific race, and it is why "just order the two calls
correctly" is not a fix anywhere in this design. Any time you find yourself reasoning about
ordering between something your processor pushes and something `FlowManager` queues, stop and
check the *distance* each one travels.

---

## 13. THE BUS AS A SIDE CHANNEL — and when it is the wrong tool

You will look at `src/pipecat/bus/` and think it is the natural home for
`SharedLayerContext` and `SignalQueue`. It is not, and the reason is a missing verb.

### 13.1 What the bus is

Name-addressed pub/sub **between workers**. The unit is a `BusMessage` dataclass; delivery is
a one-way callback — `async def on_bus_message(self, message) -> None` (`bus/subscriber.py:25`);
`BusSubscriber` (`bus/subscriber.py:12`) is a `name` property plus that one method, and that is
the entire receive-side contract. Fan-out at `bus/bus.py:153-160` gives every subscriber every
message, with `target` filtered by the *receiver*.

**There is no read API.** No `bus.get()`, no snapshot, no query. You cannot ask the bus what
the current state is; you can only be told when something happened. Plus two queue hops of
staleness on every non-system message — `_router_task` (`bus/bus.py:169`) shunts data to
`data_queue` and `_data_dispatch_task` (`:187`) drains it.

### 13.2 What boson's layers actually need

Three verbs, and the bus has none of them:

| boson | verb | where |
|---|---|---|
| `SharedLayerContext.__getattr__` / `__setattr__` | **synchronous read/write of live state** | `layers/context.py:40,54` — proxies straight through to the real `SessionState` |
| `AgentStatusTracker.get_status()` | **poll** | `layers/status.py:59` |
| `SignalQueue.get_recent(seconds, source_layer)` | **query** over an append-only list | `layers/signals.py:63` |

Read, poll, query. The bus offers push. Porting `SharedLayerContext` onto it would force every
layer to maintain its own replica and would **silently** turn `session.counter += 1` — which
today writes through to the live `SessionState` — into a write to a local copy that nobody
else ever sees. Silently. No error, no warning, wrong answers.

### 13.3 What Pipecat's own answer is

Not the bus. Look at what the shipped voicemail extension does when four processors in two
different branches of a `ParallelPipeline` need to coordinate:

**`src/pipecat/extensions/voicemail/voicemail_detector.py:634-648`**

```python
        # Create notification system for coordinating between components
        self._gate_notifier = EventNotifier()  # Signals classification completion
        self._conversation_notifier = EventNotifier()  # Signals conversation detected
        self._voicemail_notifier = EventNotifier()  # Signals voicemail detected

        # Create the processor components
        self._classifier_gate = ClassifierGate(self._gate_notifier, self._conversation_notifier)
        self._conversation_gate = ConversationGate(self._voicemail_notifier)
        self._classification_processor = ClassificationProcessor(
            gate_notifier=self._gate_notifier,
            conversation_notifier=self._conversation_notifier,
            voicemail_notifier=self._voicemail_notifier,
            voicemail_response_delay=voicemail_response_delay,
        )
        self._voicemail_gate = TTSGate(self._conversation_notifier, self._voicemail_notifier)
```

Three plain `EventNotifier()` objects (`utils/sync/event_notifier.py:14`), constructed in
`__init__` and handed to four processors — one of which, `TTSGate`, is spliced into a
completely different point of the *outer* pipeline via `.gate()`. That is Pipecat's own answer
to non-adjacency, and it is a shared object, not a message.

Three placements, then:

| need | tool | why |
|---|---|---|
| layers read live session state | **a shared plain object** — the `LLMContext` pattern from [[ch-09/read]] | synchronous reads; many holders of one object |
| "layer A tells non-adjacent layer B something happened" | **a shared `EventNotifier`** | `voicemail_detector.py:635-637` is the precedent |
| cross-cutting observation of every frame at every processor | **an Observer** ([[ch-11/read]]) | observers see everything *without being spliced into the pipe* — exactly the non-adjacency property `SignalQueue` needs |
| a classifier or analytics worker **out of process** | **the bus** (`RedisBus` / `PgmqBus` + `BusJobRequestMessage`) | genuine process boundary; request/response with status |
| "wait until component X is ready" | **`WorkerRegistry.watch(name, handler)`** | `registry/registry.py:80` — idempotent, and fires immediately if already registered (`:100-102`), which closes the startup race for free; boson hand-rolls this today |

The last row is a small free win. Take it.

**One warning if you do reach for the bus.** `BusBridgeProcessor` (`bus/bridge_processor.py:41`)
**consumes the local frame**: the normal path at `:114-121` builds a `BusFrameMessage`, calls
`self._bus.send(msg)`, and never calls `push_frame`. It is a *terminator* mid-pipeline —
everything downstream of it receives only what comes back from the bus. Only
`_LIFECYCLE_FRAMES` (`:37`), `_PASSTHROUGH_FRAMES` (`:38`) and your `exclude_frames` pass
through locally. Splice one in without knowing that and your pipeline goes quiet.

---

## 14. THREE OPEN RISKS, each with a named prototype

None of these are settled by reading. Each is an experiment, and each has a pass/fail
condition you can write before you run it.

### Risk 1 — filler filter versus energy-based barge-in

**The gap.** boson filters `"네"` by *content* and by `pre_turn_status`. Pipecat interrupts on
**VAD energy**, upstream of STT and content-blind. So by the time `BosonFillerGate` sees any
text at all, the bot has already been interrupted. The gate can stop the backchannel from
becoming a turn; it cannot un-interrupt an interruption that was broadcast before the
transcript existed.

**Measure this, do not estimate it.** Timestamp-diff the interruption broadcast against
`TranscriptionFrame` arrival, over a corpus of **lone Korean backchannels on 8 kHz telephony
audio**. The gap will be positive — the question is by how much, and whether the bot's TTS has
produced audible output inside it.

**What [[ch-06/read]] already settled, so you do not redo it.** §19.3 checked the two candidate
strategies:

- `TranscriptionUserTurnStartStrategy` (`turns/user_start/transcription_user_turn_start_strategy.py:14`)
  exists and is a *supported* Pipecat config, but it is the wrong direction: it makes turn-start
  **more** eager (`InterimTranscriptionFrame` → `trigger_user_turn_started()` at `:38-40`). It
  adds a text trigger; it does not withhold a VAD one.
- `MinWordsUserTurnStartStrategy` withholds, but counts the wrong unit:

  **`src/pipecat/turns/user_start/min_words_user_turn_start_strategy.py:108-111`**

  ```python
          min_words = self._min_words if self._bot_speaking else 1

          word_count = len(frame.text.split())
          should_trigger = word_count >= min_words
  ```

  `"네"` is **one word** and clears `min_words=1`; boson's `WordFilterPolicy(max_chars=3)`
  catches it on **characters**. Setting `min_words=2` also suppresses `"잠깐만요"` — a genuine
  floor claim from a customer. Note `:108` too: the threshold applies **only while the bot is
  speaking**, dropping to 1 in silence. That asymmetry is right and you would have to remember
  to build it.

**So the prototype is [[ch-06/read]] §20 Probe 1, and this chapter's contribution is that it
is now mandatory rather than interesting.** A positive measured gap means a custom
`BaseUserTurnStartStrategy` subclass that withholds turn-start until a transcript exists and
passes a Korean-backchannel test. And it costs the unmeasured Korean STT TTFS
(§11.3: no entry in `stt_latency.py`), which is the same number Risk 1's measurement produces.
Do the measurement once and it answers two questions.

### Risk 2 — the transition frame race

**Prototype.** Fire a rule-driven transition. Put a `FrameLogger` (`processors/logger.py:23`)
at the `llm` **input** and assert two things on the same trace:

1. **Exactly one** inference-triggering frame arrives per turn.
2. `LLMSetToolsFrame` **precedes** it.

Watch for **both** failure directions, because they are symmetric and one of them is easy to
miss:

| you did | frames at `llm` | symptom |
|---|---|---|
| push the context frame **and** call `set_node_from_config` | **two** | the bot answers itself; two TTFBs; doubled cost |
| swallow both paths | **zero** | the bot goes silent mid-call and nothing errors |
| swallow only the context frame (§8.4) | **one** | correct |

The zero case is the dangerous one — a silent bot on a live sales call produces no exception,
no log line, and a customer hanging up.

**A trap in the instrument itself**, from [[processor-vocabulary]]: `FrameLogger.__init__`
takes `ignored_frame_types` defaulting to a four-type tuple, and the guard at `:64` is
`if self._ignored_frame_types and not isinstance(...)`. Passing `ignored_frame_types=()`
disables **all** logging rather than logging everything — the empty tuple is falsy. Do not
debug a silent bot with a logger you silenced.

PANEL SIX of the figure runs all three rows.

### Risk 3 — two-phase-commit blast radius

**The gap.** boson rolls back by object identity over `session.messages`. `LLMContext` offers
only `set_messages(list)` and the aggregator has already written at `:863`. §9.3 showed the
positional stand-in `snapshot[:-1]` and why it is sound *only* because the layers collapsed.
"Sound in the argument" and "sound over the real corpus" are different claims.

**Prototype.** Snapshot/restore around the whole rule round and replay the Lina e2e suite
(`agents/test-lina-gateway/tests/`, `e2e_runner.py`), counting divergences against boson's
current behaviour turn by turn.

**Where the divergences will be, specifically:** turns where a `PreTool` appended synthetic
tool-call history **before** a later layer filtered. That is the case where the list you
restore and the list boson's identity-walk would have produced are not the same list, because
`PreTool` writes more than one message and `snapshot[:-1]` removes exactly one. Write that test
first.

**Pass condition:** zero divergences on the `PreTool`-then-`Filter` subset, or a documented and
deliberate difference for each one. "Close enough" is not a pass condition for a system that
speaks regulated consent text.

---

## 15. Source corrections, collected

Four, all verified by opening the file at `0cbf9c5b`. The excerpts are pre-read summaries and
they were right about far more than they were wrong about, but where they disagree with the
source, **the source wins**.

| # | claim in the excerpt / spec | what the file says |
|---|---|---|
| 1 | Path B is proven at `warm_transfer.py:658` ([[design-boson-rules-on-pipecat]] §2) | `:261` is the `set_node_from_config` call; `:658` is the transport callback that reaches it. Two different lines, both real. §1 |
| 2 | rollback is `snapshot = ctx.get_messages()` then `set_messages(snapshot)` ([[design-boson-rules-on-pipecat]] §3, and the figure spec) | `get_messages()` returns the **live list** (`llm_context.py:244-245,260`) and `set_messages` is `self._messages[:] = messages` (`:383`). Without `list(...)` the rollback is a self-assignment and silently restores nothing. §9.3 |
| 3 | "Pipecat's own STT TTFS P99 reference is 0.45 s for Deepgram" ([[design-boson-rules-on-pipecat]] §3) | `DEEPGRAM_TTFS_P99 = 0.35` (`stt_latency.py:45`). `0.45` appears only in a docstring example at `:27`. §11.3 |
| 4 | wrap in `start_processing_metrics()`/`stop_processing_metrics()` "or the bill never appears in `LatencyBreakdown`" (chapter spec) | `UserBotLatencyObserver._handle_metrics_frame` reads only `TTFBMetricsData` and `TextAggregationMetricsData` (`:322-341`); `LatencyBreakdown` has no processing field (`:107-111`). And `FrameProcessor.can_generate_metrics()` returns `False` (`:494`), so on a plain custom processor *all* metrics calls are silent no-ops. §11.4 |

And one correction to an excerpt's *recommendation* rather than its facts:
[[custom-processor-guide]]'s *"the honest port splits its four layers into four
`FrameProcessor`s"* is knocked out by §6. Its facts about what four processors would give you
are all correct; the conclusion does not survive Constraints Two and Three.

---

## 16. What this chapter did not decide

Three things, named so you do not mistake silence for agreement.

**Whether to do the migration at all.** [[ch-13/read]] owns that. This chapter answered "if you
port the rule layers, what shape must they take and what does it cost." It did not answer
"should you."

**Whether Tier 2 belongs on the critical path forever.** §11.5 states the trade and boson's
current position. It does not tell you that 250-400 ms is worth it for Lina. That is a product
decision informed by a number you now have, plus a Korean STT TTFS number you do not yet have.

**Whether the collapse is acceptable operationally.** §6.1 priced it in lost per-layer
queueing, observability and interruption semantics. Those are real, and if your team debugs by
reading per-processor traces they may hurt more than the argument suggests. The correctness
conclusion stands regardless; the operational verdict is yours.

---

## 다음 챕터로

This chapter hands three things to [[ch-13/read]], which is the capstone and has to decide
keep-or-replace subsystem by subsystem.

**A priced row for the rules subsystem.** Not "portable" or "hard" — **one `FrameProcessor`,
zero edits to the thirteen rule files, +250-400 ms on every turn, and cross-layer veto
preserved only because the layers collapsed.** That is a row you can put next to the transport
row and the STT row and compare.

**A short list of things that genuinely have no Pipecat home**, so ch-13 does not have to
rediscover them: transition legality (`StageDefinition.transitions`), `skills`, per-session
attribute namespaces (`SharedLayerContext`), the `TOOL_PROCESSING` agent status, and the
`Inject` merge-into-the-last-user-message option. Every one of those stays boson code or
disappears.

**And a short list of things Pipecat gives back**, which ch-13 should credit on the other side
of the ledger: Flows nodes for stages, `global_functions` for `_GLOBAL_TOOLS`, `function`
pre-actions with their always-wait semantics for `PreTool`, `WorkerRegistry.watch` for the
startup race boson hand-rolls, and — from §13 — the `EventNotifier` pattern for every
"non-adjacent components must coordinate" problem you were about to solve with a message bus.

Two sentences to carry out of this chapter.

**The seam is not a matter of taste.** `:863` writes, `:866` pushes, `:601` tests, `:603`
pushes irreversibly, `:605` generates. There is exactly one gap in that sequence where a
complete turn exists and nothing has been spent, and every design choice in this chapter falls
out of where that gap is.

**And the collapse is the price of the veto.** Pipecat's one write verb takes effect on call.
A transaction that spans processors therefore cannot roll back, so a system that needs
cross-layer veto has to be one processor — which is not Pipecat being deficient, it is
Pipecat's bet about where transactions live, presenting you with its bill.
