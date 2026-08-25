---
title: "Pipecat Flows: A State Machine That Lives OUTSIDE the Pipeline"
chapter: ch-10
phase: collision
course: pipecat
sources:
  - flows-state-machine
  - flows-node-types
  - flows-actions
  - flows-insurance-example
  - boson-stage-machine
  - theory-narrow-waist
deps:
  - ch-02
  - ch-09
figure: figures/flow-node-transition.html
---

# Chapter 10 — Pipecat Flows: A State Machine That Lives OUTSIDE the Pipeline

> **Scope, stated up front and enforced for the whole chapter.**
>
> **One idea: the inversion.** `FlowManager` is not a `FrameProcessor`. It is not in the `Pipeline`
> list. It drives the pipeline from outside by pushing frames into the head of it. Every other fact
> in this chapter — the frame batch, the string node ID, the missing validation, the three action
> verbs, the insurance graph — is a consequence of that one structural choice, and each section says
> which consequence it is.
>
> **The rule-layer design is NOT in this chapter.** You will learn here that Flows accepts a node
> transition from plain code as well as from a tool call. That is a *mechanism*. Turning it into
> boson's `RuleEngine → StageTransition → set_node_from_config` seam — where the processor stands,
> what it costs in milliseconds, whether the layers collapse — is [[ch-12/read]]'s entire subject and
> [[ch-11/read]] supplies the denominator. If you catch yourself sketching a `BosonRuleProcessor`,
> you have left this chapter.
>
> **No comparative verdict.** This chapter says what Flows **does**, what it **costs**, and what it
> **does not have**. It does not say whether to adopt it, whether it beats what you already shipped,
> or whether boson's stage machine should be replaced. [[ch-13/read]] is the only place in this
> course where anything is scored, and it earns that only after seeing all twelve subsystems.

---

## 왜 이 챕터인가

[[ch-02/read]] §11 used `flows/` as a frame budget and then deliberately refused to open it:

> *(How `flows/` actually works — `set_node`, the transition mechanics, the pre/post-action
> ordering, and the finding that transitions do not have to be LLM function calls — is
> [[ch-10/read]]'s subject. This section takes exactly one thing from it: the frame budget.)*

[[ch-09/read]] ended by naming what you would have to believe to read this chapter correctly:

> a state machine that mutates a context it does not own is either elegant or terrifying depending
> entirely on whether you know who else is holding a reference.

This is that chapter, and the answer to the "who else is holding a reference" question turns out to
be the whole design rather than a footnote to it.

Here is the thing worth your attention. Pipecat is a framework whose organising principle is that
**everything is a processor in a list** — [[ch-01/read]] proved the splice algebra, [[ch-04/read]]
proved the runtime, [[ch-02/read]] proved the waist. If you were told "Pipecat ships a conversation
state machine" and asked to guess its shape, you would guess a `FrameProcessor` subclass that you
put in the `Pipeline` list somewhere between the user aggregator and the LLM, intercepting frames,
holding a current-state field, deciding what to forward.

That guess is wrong, and it is wrong in a way that is worth more than the fact itself. `FlowManager`
is a **plain Python class**. It is not in the list. It has no `process_frame`. It never sees a frame
travel past it. It sits beside the pipeline holding a reference to the `PipelineWorker` and drives
the conversation by **queueing frames into the head of the pipe** — the same public API a transport
event handler uses when it makes Lina speak first ([[ch-04/read]] §11).

For you specifically, this matters twice.

**First, boson's stage machine is the same shape and you did not choose that shape deliberately.**
`StageMachine` (`gateway/stage/machine.py`) is stateless and shared, `session.active_stage` is a
string on the session, and the transition is bookkeeping performed by rule code before the agent
loop runs ([[boson-stage-machine]]). Nothing about it is a pipeline node. Pipecat's own team, given
a conversation state machine to build inside a frame-based framework, arrived at the same
architecture — state in a plain manager object, effects on existing frames. That convergence is
evidence, and this chapter tells you exactly what it is evidence *of* and what it costs.

**Second, the in-tree example is an insurance quote bot.** `examples/flows/insurance_quote.py` is
380 lines of five node factories collecting an age, a marital status, a premium, and a coverage
adjustment, over voice. You sell insurance by phone in Korean. §11 reads that file line by line and
extracts the decomposition pattern that transfers, and §13 maps it onto the nine Lina stages you
already have.

**How to read the code in this chapter.** Every path, line number, class name and count below was
re-read against `wiki/raw-data/pipecat/pipecat-src` at commit
`0cbf9c5b031eef06e53f0a193b9a67d60230e6be` while writing. Where a curated excerpt disagreed with the
source, the source wins and the disagreement is stated inline rather than quietly fixed. §0 lists
those up front, because four of them are corrections to things a first reading of this package
naturally gets wrong.

---

## 0. Six corrections, stated before anything is built on them

A conversation state machine is exactly the kind of subsystem where you arrive with expectations
from other frameworks and then read the code through them. Here are the six places that happens with
`pipecat.flows`, each checked by opening the file.

### 0.1 `FlowConfig` does not exist at this commit

Every tutorial-shaped mental model of a flow framework has a document: a JSON or YAML object with
nodes, edges, and an `initial_node` key. Pipecat Flows does not have one.

```
$ cd wiki/raw-data/pipecat/pipecat-src
$ grep -rn "FlowConfig" . | wc -l
       0
$ grep -rni "static flow\|static_flow" . | wc -l
       0
$ grep -rni "dynamic flow\|dynamic_flow" . | wc -l
       0
$ grep -rn '"initial_node"' . | wc -l
       0
```

Four greps over the entire repository — source, tests, examples, docs — and all four return zero.
There is no `FlowConfig`, no declarative flow document, no edge table, no `initial_node` key, and
**no static-versus-dynamic distinction**. If you have read older material about Pipecat Flows that
contrasts "static flows" (a `FlowConfig` dict) with "dynamic flows" (Python functions), that
vocabulary is gone from this tree. Only the runtime-determined style survives, and the package
docstring states it as the design rather than as a mode:

**`src/pipecat/flows/__init__.py:6-14`**
```python
"""Pipecat Flows - Structured conversation framework for Pipecat.

This package provides a framework for building structured conversations in Pipecat.
The FlowManager handles conversation flows with support for state management,
function calling, and cross-provider compatibility.

Pipecat Flows determines conversation structure at runtime, supporting function
calling, action execution, and seamless transitions between conversation states.
"""
```

*"determines conversation structure at runtime."* A node is a Python function that returns a
`NodeConfig`. An edge is the second element of a tuple that a handler returned. That is the entire
graph representation, and §6.4 gives it a type.

### 0.2 There are exactly THREE built-in actions

Not a verb library. Three, registered in three consecutive lines, and §9 quotes them.

### 0.3 `_validate_node_config` performs no transition-legality check

It checks two things and neither of them is a from→to edge. §10 quotes the whole method so you can
count the checks yourself.

### 0.4 `self._current_functions` is dead in `src/` — but not "nowhere"

The tempting sentence is "it is assigned and never read anywhere in the codebase." That sentence is
wrong, and the corrected version is more interesting:

```
$ grep -rn "_current_functions" src/ tests/ examples/
src/pipecat/flows/manager.py:148:        self._current_functions: set[str] = set()  # Track registered functions
src/pipecat/flows/manager.py:704:            self._current_functions = new_functions
tests/test_flows_manager.py:191:        self.assertEqual(flow_manager._current_functions, set())
tests/test_flows_manager.py:319:        self.assertEqual(len(flow_manager._current_functions), 3)
tests/test_flows_manager.py:758:        self.assertIn("test1", flow_manager._current_functions)
tests/test_flows_manager.py:759:        self.assertIn("test2", flow_manager._current_functions)
tests/test_flows_manager.py:1174:        self.assertEqual(flow_manager._current_functions, set())
tests/test_flows_manager.py:1203:        self.assertEqual(flow_manager._current_functions, set())
```

Two sites in `src/` — one declaration, one assignment. **Zero reads in `src/`.** Six reads in
`tests/test_flows_manager.py`. So the precise claim is: *`_current_functions` is dead as a runtime
gate and live as a test-assertion surface.* Nothing in the running system consults it before
dispatching a function; the test suite consults it to check that `_set_node` tracked what it was
supposed to track. §5.4 explains why the distinction changes what you can build on it.

### 0.5 `FlowResult` is deprecated with **"No replacement."**

**`src/pipecat/flows/types.py:40-53`**
```python
@deprecated("`FlowResult` is deprecated since 1.5.0 and will be removed in 2.0.0. No replacement.")
class FlowResult(TypedDict, total=False):
    """Optional convention TypedDict for ``status``/``error`` results.

    .. deprecated:: 1.5.0
        No replacement. ``FlowResult`` is no longer required or referenced by
        any handler type, and Pipecat's upstream function-call-result contract
        is ``Any`` — define your own ``TypedDict`` or return any
        JSON-serializable value. Will be removed in 2.0.0.

    Parameters:
        status: Status of the function execution.
        error: Optional error message if execution failed.
    """
```

Read the deprecation body carefully, because it is unusual. Most Pipecat deprecations name a
replacement in the first sentence — the repo's own `AGENTS.md` requires it. This one says **"No
replacement."** and then explains why: the upstream contract for a function-call result is plain
`Any`. `FlowResult` was a convention, the convention was dropped, and nothing took its place.
`insurance_quote.py` accordingly declares its own bare `TypedDict`s (`AgeCollectionResult`,
`QuoteCalculationResult`) that inherit from nothing.

### 0.6 `NodeConfig`'s key set includes `role_messages`, and `task_messages` is the only `Required` one

Both plural and singular forms exist. `role_message` (singular, `str`) is current; `role_messages`
(plural, `list[dict]`) is deprecated since 1.5.0 and takes a completely different path through the
pipeline — §4.3 shows that the two do not merely differ in shape, they emit different frames. And of
the nine keys, exactly one is `Required[...]`. §6.1 has the table.

---

## 1. The one structural fact

### 1.1 The class declaration

**`src/pipecat/flows/manager.py:80-89`**
```python
class FlowManager:
    """Manages conversation flows.

    The FlowManager orchestrates conversation flows by managing state transitions,
    function registration, and message handling across different LLM providers,
    with comprehensive action handling and error management.

    The manager coordinates all aspects of a conversation including LLM context
    management, function registration, state transitions, and action execution.
    """
```

`class FlowManager:` — bare. No base class. Not `FrameProcessor`, not `BasePipeline`, not
`BaseObject`. It does not inherit the event system, the task manager, the two-queue runtime from
[[ch-04/read]] §4, or `push_frame`. It has no `process_frame` method, which means the uniform
interface [[ch-01/read]] built the whole splice algebra on — `async def process_frame(self, frame:
Frame, direction: FrameDirection)` at `processors/frame_processor.py:820` — is simply absent here.

**You cannot `link()` a `FlowManager` to anything.** It is not a member of the algebra.

### 1.2 It is not in the `Pipeline` list — read `hello_world.py`

The 192-line `hello_world.py` is the smallest complete Flows program in the tree, and the important
thing about it is what its pipeline list does *not* contain.

**`examples/flows/hello_world.py:135-167`**
```python
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )
```

Count the pipeline entries: **seven**, and every one of them is from [[ch-04/read]] §11's canonical
voice bot. `transport.input()`, `stt`, `context_aggregator.user()`, `llm`, `tts`,
`transport.output()`, `context_aggregator.assistant()`. This is the identical list you would write
for a bot with no state machine at all. Adding a conversation state machine to a Pipecat bot changes
**zero** entries in the `Pipeline` constructor.

Now read the construction order, because it is load-bearing:

1. `Pipeline([...])` is built — the flow does not exist yet.
2. `PipelineWorker(pipeline, ...)` wraps it.
3. `WorkerRunner(...)` is constructed.
4. `await runner.add_workers(worker)` registers it.
5. **Only then** `FlowManager(worker=worker, llm=llm, context_aggregator=context_aggregator,
   transport=transport)`.

The `FlowManager` is constructed *after* the pipeline is fully assembled and registered, because it
is a **consumer** of the assembled pipeline rather than a **part** of it. It takes the already-built
`worker` as a constructor argument. There is no possible ordering in which it could be spliced in,
because there is nothing to splice.

And then the kickoff:

**`examples/flows/hello_world.py:169-180`**
```python
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()
```

The conversation starts from a **transport event handler**, not from a frame arriving anywhere. This
is exactly the shape [[ch-04/read]] §11 identified as the reason a voice agent speaks first: the
pipeline is machinery, and something outside it has to start the conversation.
`flow_manager.initialize(create_initial_node())` is that something.

`insurance_quote.py:350-361` is byte-for-byte the same pattern with a different initial node. So is
`restaurant_reservation.py`. So is `patient_intake.py`. The shape is not an accident of the hello
world.

### 1.3 What it holds instead of a position in the list

Since it has no position, it must hold references. Here is the whole state, verbatim:

**`src/pipecat/flows/manager.py:134-154`**
```python
        self._worker = worker
        self._llm = llm
        self._action_manager = ActionManager(worker, flow_manager=self)
        self._adapter = LLMAdapter()
        self._initialized = False
        self._context_aggregator = context_aggregator
        self._pending_transition: dict[str, Any] | None = None
        self._context_strategy = context_strategy or ContextStrategyConfig(
            strategy=ContextStrategy.APPEND
        )
        self._transport = transport
        self._global_functions = global_functions or []

        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None

        self._showed_deprecation_warning_for_role_messages = False
        self._showed_deprecation_warning_for_reset_with_summary = False
        self._showed_deprecation_warning_for_zero_arg_handler = False
        self._showed_deprecation_warning_for_legacy_handler = False
```

Three references outward (`_worker`, `_llm`, `_context_aggregator`, plus the optional `_transport`),
one owned sub-object (`_action_manager`), and three fields of actual conversation state:
`_state` is a bare `dict`, `_current_functions` is a `set[str]` that nothing in `src/` reads, and
`_current_node` is a `str | None`.

Note also the four `_showed_deprecation_warning_for_*` booleans. That is one flag per deprecated
code path, each existing so a warning fires once per `FlowManager` rather than once per node
transition. Four of them in a 898-line module is a measurement of how much of this API is on its way
out, and §6.7 lists what is scheduled for removal.

The constructor is **keyword-only** (`def __init__(self, *, llm, context_aggregator, worker=None,
task=None, ...)`, `manager.py:91-101`), and `task` is the deprecated 1.5.0 spelling of `worker`.
Passing both raises:

**`src/pipecat/flows/manager.py:121-132`**
```python
        if worker is not None and task is not None:
            raise ValueError("Pass either 'worker' or 'task' (deprecated), not both.")
        if task is not None:
            warnings.warn(
                "The 'task' parameter is deprecated since 1.5.0 and will be removed "
                "in 2.0.0. Use 'worker' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            worker = task
        if worker is None:
            raise ValueError("FlowManager requires a 'worker' (PipelineWorker).")
```

The last three lines are the point of this section restated as a runtime check: **a `FlowManager`
without a `PipelineWorker` cannot exist.** A `FrameProcessor` has no such dependency — you can
construct one, hold it, and link it later. `FlowManager` requires the assembled thing it will drive,
at construction, or it refuses to be built.

### 1.4 This is [[ch-02/read]]'s rule applied at full scale

[[ch-02/read]] §12 stated a three-way test for boson's own state: **(a)** state → not a frame,
**(b)** effects → existing frames, **(c)** genuinely new in-band signals → a subsystem-local
`frames.py`, budget two to four.

`flows/` is Pipecat's own team obeying its own rule on the hardest possible case, and now you can
see the full shape of the obedience rather than just the frame count:

| Test | What `flows/` did | Evidence |
|---|---|---|
| (a) state → not a frame | `_current_node: str`, `_state: dict`, `NodeConfig` as `TypedDict` | `manager.py:147-149`; `types.py:182` |
| (b) effects → existing frames | `LLMUpdateSettingsFrame`, `LLMMessagesAppendFrame`/`UpdateFrame`, `LLMSetToolsFrame`, `LLMRunFrame`, `TTSSpeakFrame`, `EndFrame` | `manager.py:768, 838, 839, 709`; `actions.py:323, 359` |
| (c) new in-band signals | exactly two, both `ControlFrame`, both in `flows/actions.py` | `actions.py:49-66` |
| **(d) — the part ch-02 could not see** | **the manager itself is not in the pipeline at all** | `manager.py:80`; `hello_world.py:135-145` |

Row (d) is what this chapter adds. [[theory-narrow-waist]] §4 counted the two frames and drew the
three-way test; [[ch-02/read]] read the frame budget and correctly concluded that
`flows/` kept its private vocabulary private. What it could not show, because it was reading
`frames.py`, is that the *component* stayed private too. The frame budget is two because the
component budget is **zero**.

---

## 2. Two touch points, and only two

If a thing outside the pipeline is going to drive the pipeline, there are exactly two questions:
how do effects get in, and how do observations get out. Flows answers each in one place.

### 2.1 Out: `queue_frames`, twice

```
$ grep -n "queue_frames\|queue_frame(" src/pipecat/flows/manager.py src/pipecat/flows/actions.py
src/pipecat/flows/actions.py:322:            await self._worker.queue_frame(
src/pipecat/flows/actions.py:329:            await self._worker.queue_frame(ActionFinishedFrame())
src/pipecat/flows/actions.py:353:            await self._worker.queue_frame(
src/pipecat/flows/actions.py:359:        await self._worker.queue_frame(EndFrame())
src/pipecat/flows/actions.py:385:            await self._worker.queue_frame(FunctionActionFrame(action=action, function=handler))
src/pipecat/flows/manager.py:269:                    await flow_manager.worker.queue_frame(
src/pipecat/flows/manager.py:709:                await self._worker.queue_frames([LLMRunFrame()])
src/pipecat/flows/manager.py:841:            await self._worker.queue_frames(frames)
```

`manager.py:269` is inside a docstring example, not code. So the manager's real write surface is
**two lines**: `:709` queues the inference trigger, `:841` queues the settings-and-context batch.
Everything else is `ActionManager`, and its five writes are all inside the three built-in action
handlers.

That is the total. A subsystem of 898 + 400 + 518 + 68 + 62 = 1,946 lines drives a real-time voice
pipeline through **seven `queue_frame*` call sites**.

### 2.2 In: one event, filtered to three frame types

Reading back is harder, because a thing outside the pipeline cannot observe frames by being adjacent
to anything. Flows uses the worker's downstream-arrival event:

**`src/pipecat/flows/actions.py:103-127`**
```python
        # Register built-in actions
        self._register_action("tts_say", self._handle_tts_action)
        self._register_action("end_conversation", self._handle_end_action)
        self._register_action("function", self._handle_function_action)

        # Add pipeline observation
        worker.set_reached_downstream_filter(
            (ActionFinishedFrame, FunctionActionFrame, BotStoppedSpeakingFrame)
        )

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame_reached_downstream(worker, frame):
            if isinstance(frame, FunctionActionFrame):
                # Run function action
                await frame.function(frame.action, flow_manager)
                self._decrement_ongoing_actions_count()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                # Execute deferred post-actions if the bot's turn is over.
                # A BotStoppedSpeakingFrame only indicates that the bot's turn is over if there are
                # no ongoing actions (otherwise one of those actions may have been responsible for it).
                if self._ongoing_actions_count == 0:
                    await self._execute_deferred_post_actions()
            elif isinstance(frame, ActionFinishedFrame):
                # Handle action finished
                self._decrement_ongoing_actions_count()
```

Three frame types, three behaviours:

- `FunctionActionFrame` — **this is where a `function` action's handler actually runs.** Not when the
  action was scheduled. The handler rides inside the frame as a field
  (`function: FlowActionHandler`, `actions.py:59`) and is invoked when the frame reaches the tail of
  the pipeline. §9.2 explains why that placement is the whole point of the `function` verb.
- `BotStoppedSpeakingFrame` — the trigger for deferred post-actions, guarded on the ongoing-action
  count for the reason the comment states.
- `ActionFinishedFrame` — the completion signal for `tts_say`, decrementing the counter that
  §9.6's wait table blocks on.

The mechanism on the worker side:

**`src/pipecat/pipeline/worker.py:1320-1329`**
```python
    async def _sink_push_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames coming downstream from the pipeline.

        This tasks process frames coming downstream from the pipeline. For
        example, heartbeat frames or an EndFrame which would indicate all
        processors have handled the EndFrame and therefore we can exit the worker
        cleanly.
        """
        if isinstance(frame, tuple(self._reached_downstream_types)):
            await self._call_event_handler("on_frame_reached_downstream", frame)
```

The worker's internal sink — the processor [[ch-04/read]] §6.3 showed it wraps around your pipeline —
fires the event for any frame whose type is in the registered set. Flows is not adjacent to
anything; it is subscribed to the far end.

### 2.3 Honest finding: `set_reached_downstream_filter` REPLACES, and Flows is the only caller

Look at the two methods the worker exposes:

**`src/pipecat/pipeline/worker.py:695-717`**
```python
    def set_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
        """Set which frame types trigger the on_frame_reached_downstream event.

        Args:
            types: Tuple of frame types to monitor for downstream events.
        """
        self._reached_downstream_types = set(types)

    def add_reached_upstream_filter(self, types: tuple[type[Frame], ...]):
        """Add frame types to trigger the on_frame_reached_upstream event.

        Args:
            types: Tuple of frame types to add to upstream monitoring.
        """
        self._reached_upstream_types.update(types)

    def add_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
        """Add frame types to trigger the on_frame_reached_downstream event.

        Args:
            types: Tuple of frame types to add to downstream monitoring.
        """
        self._reached_downstream_types.update(types)
```

`set_...` does `= set(types)`. `add_...` does `.update(types)`. The default is empty
(`worker.py:563`: `self._reached_downstream_types: set[type[Frame]] = set()`), so on a fresh worker
the two are equivalent — which is presumably why `ActionManager` uses `set_`.

But:

```
$ grep -rn "set_reached_downstream_filter\|add_reached_downstream_filter" src/ examples/
src/pipecat/pipeline/worker.py:695:    def set_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
src/pipecat/pipeline/worker.py:711:    def add_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
src/pipecat/flows/actions.py:109:        worker.set_reached_downstream_filter(
```

`ActionManager.__init__` is the **only** caller of either method in the whole tree, and
`add_reached_downstream_filter` has zero callers. So this is a latent, order-dependent conflict that
nothing in the repo currently exercises:

- If your application calls `worker.set_reached_downstream_filter((MyFrame,))` and **then**
  constructs a `FlowManager`, your filter is silently wiped and your handler stops firing.
- If you construct the `FlowManager` and **then** call `set_reached_downstream_filter(...)`, you wipe
  Flows' filter — and the failure mode is not an exception. `function` actions stop executing,
  because the handler only ever runs from `on_frame_reached_downstream`. Deferred post-actions never
  fire. `tts_say` never decrements its counter, so §9.6's wait table blocks forever on the next
  batch that needs to wait.

The fix is one word: use `add_reached_downstream_filter` on your side, always, and construct the
`FlowManager` before anything else registers a filter. Write it down now, because the symptom
("post-actions randomly don't run") points nowhere near the cause.

This is not in any excerpt; it came out of grepping the worker while checking §2.2. It is exactly
the class of coupling that appears when a component drives a pipeline it is not part of: the
component has to reach into shared, single-slot configuration on the worker, and single-slot
configuration does not compose.

---

## 3. What "outside" costs: the frames enter at the HEAD

This is the section that makes the inversion concrete rather than architectural, and it is the one
[[ch-12/read]] will build its central hazard on.

### 3.1 The direction split, recalled in three lines

[[ch-04/read]] §10 established the mechanism; here is the one sentence you need. `queue_frame` with
the default `FrameDirection.DOWNSTREAM` puts the frame on `self._push_queue`, which feeds
`self._pipeline.queue_frame(...)` — so **the frame enters at the head and traverses every processor
in order.** Upstream frames enter at the tail instead. And `queue_frames` is a plain loop:

**`src/pipecat/pipeline/worker.py:810-829`**
```python
    async def queue_frames(
        self,
        frames: Iterable[Frame] | AsyncIterable[Frame],
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        """Queue multiple frames to be pushed through the pipeline.

        Downstream frames are pushed from the beginning of the pipeline.
        Upstream frames are pushed from the end of the pipeline.

        Args:
            frames: An iterable or async iterable of frames to be processed.
            direction: The direction to push the frames. Defaults to downstream.
        """
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame, direction)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame, direction)
```

No batching. No atomicity. **A "frame batch" from `FlowManager` is a batch only in the sense that a
`for` loop wrote it.** Frames from elsewhere can interleave.

Flows never passes a direction. Every one of its seven writes is downstream, at the head.

### 3.2 Trace one `LLMRunFrame` and find out where inference actually starts

Here is a question worth answering precisely, because the obvious answer is wrong. `_set_node` does
`await self._worker.queue_frames([LLMRunFrame()])` at `manager.py:709`. In the seven-processor
pipeline from §1.2, **which processor turns that into an inference call?**

The obvious answer is `llm`. It is not.

**`src/pipecat/processors/aggregators/llm_response_universal.py:814-821`** — inside
`LLMUserAggregator.process_frame`, the third processor in the list:
```python
        elif isinstance(frame, LLMRunFrame):
            await self._handle_llm_run(frame)
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_llm_messages_append(frame)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            await self._handle_llm_messages_update(frame)
        elif isinstance(frame, LLMMessagesTransformFrame):
            await self._handle_llm_messages_transform(frame)
```

**`src/pipecat/processors/aggregators/llm_response_universal.py:1173-1184`**
```python
    async def _handle_llm_run(self, frame: LLMRunFrame):
        await self.push_context_frame()

    async def _handle_llm_messages_append(self, frame: LLMMessagesAppendFrame):
        self.add_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame()

    async def _handle_llm_messages_update(self, frame: LLMMessagesUpdateFrame):
        self.set_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame()
```

And the tail of that same `elif` chain:

**`src/pipecat/processors/aggregators/llm_response_universal.py:849-850`**
```python
        else:
            await self.push_frame(frame, direction)
```

Read those three fragments together. `LLMRunFrame`, `LLMMessagesAppendFrame` and
`LLMMessagesUpdateFrame` each match an `elif` branch that calls a handler **and does not call
`push_frame`**. The chain's terminal `else` is the only pass-through. Therefore:

> **`LLMRunFrame`, `LLMMessagesAppendFrame` and `LLMMessagesUpdateFrame` are CONSUMED by the user
> aggregator. They never reach the LLM service. They never reach the assistant aggregator.**

What reaches the LLM is an `LLMContextFrame`, pushed downstream by `push_context_frame()` — carrying
the shared `LLMContext` object that [[ch-09/read]] §2 spent a section proving is a live reference
rather than a copy. The node's messages were written into that object by `add_messages` /
`set_messages` a few microseconds earlier, by the same processor.

So the real answer to "where does a node transition become an inference" is: **at the third
processor in the list, two hops before the LLM**, and the mechanism is a mutation of a shared object
followed by a pointer to it.

This also settles a question you might not have thought to ask. `LLMAssistantAggregator` has its own
`_handle_llm_run` (`llm_response_universal.py:1705-1706`) that pushes the context frame **UPSTREAM**:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1705-1716`**
```python
    async def _handle_llm_run(self, frame: LLMRunFrame):
        await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_llm_messages_append(self, frame: LLMMessagesAppendFrame):
        self.add_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_llm_messages_update(self, frame: LLMMessagesUpdateFrame):
        self.set_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame(FrameDirection.UPSTREAM)
```

In the canonical pipeline that code is unreachable for a downstream `LLMRunFrame`, because the user
aggregator ate it four processors earlier. It exists for the case where the frame is queued
**upstream** — entering at the tail — and for pipelines that have an assistant aggregator without a
user aggregator ahead of it. Two symmetric handlers, one live per direction. That is the direction
split from [[ch-04/read]] §10 showing up as duplicated code in a real processor.

### 3.3 `LLMSetToolsFrame` is the one that goes all the way — and is handled twice

The tool frame behaves differently, and the difference is instructive.

**`src/pipecat/processors/aggregators/llm_response_universal.py:822-834`** — user aggregator:
```python
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

**`src/pipecat/processors/aggregators/llm_response_universal.py:1585-1591`** — assistant aggregator:
```python
        elif isinstance(frame, LLMSetToolsFrame):
            # Normalize and validate (a plain list of direct functions / FunctionSchema
            # objects becomes a ToolsSchema) so the tool-change diff and
            # set_tools see a consistent type.
            normalized_tools = LLMContext._normalize_and_validate_tools(frame.tools)
            self._maybe_add_tool_change_messages(normalized_tools)
            self.set_tools(normalized_tools)
```

Same three lines, minus the push. So the frame is handled at processor 3, forwarded, seen by the LLM
service at processor 4, and handled again at processor 7 where it stops.

Two aggregators calling `set_tools` on the **same shared `LLMContext`** is idempotent. Two
aggregators calling `_maybe_add_tool_change_messages` would not be — it appends a developer message
announcing the delta — except that the method's docstring is unusually explicit about having thought
this through:

**`src/pipecat/processors/aggregators/llm_response_universal.py:413-428`**
```python
    def _maybe_add_tool_change_messages(self, new_tools: ToolsSchema | NotGiven) -> None:
        """Append a developer message describing tool add/remove deltas.

        No-op unless ``add_tool_change_messages`` was enabled on the aggregator,
        and no-op when the diff against the currently advertised tools is empty.
        Custom (LLM-specific) tools are ignored — only standard tools are diffed.

        Both aggregators call this on every ``LLMSetToolsFrame`` they handle.
        Whichever aggregator handles the frame first computes a real diff
        against the shared context and adds the announcement; by the time
        the other aggregator sees it (if at all), the context already
        reflects the new tools, so its diff is empty and no duplicate
        message is added. This is order-independent: it works whether the
        frame flows downstream (user aggregator first) or upstream
        (assistant aggregator first, and consumed without being forwarded).
        """
```

*"the context already reflects the new tools, so its diff is empty."* The de-duplication is not a
guard; it is a consequence of both aggregators diffing against the same live object. This is the
sharpest single illustration of [[ch-09/read]] §2.3's claim that the live list is load-bearing — the
correctness of a duplicate-suppression behaviour rests entirely on the fact that the two processors
are looking at one object and not two copies.

And where does the tool change actually take effect for a text LLM? Not on the `LLMSetToolsFrame`:

**`src/pipecat/services/llm_service.py:712-723`**
```python
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

*"the single place tool changes take effect for text LLMs."* So the node's tool swap lands because
the user aggregator wrote it into the shared context before the run, and the LLM read the context.
The `LLMSetToolsFrame` travelling past the LLM is for speech-to-speech services only. This is
[[ch-09/read]] §3's "the LLM service holds nothing" restated from the Flows side, and it means
`FlowManager` never calls `register_function` — it advertises handlers on the schema and lets the
service auto-register them (§8.4).

### 3.4 Assemble the trace

Put §3.1–§3.3 together. One node transition, one canonical pipeline, and the actual path each frame
takes:

| Frame (in queue order) | Enters at | Consumed / stops at | What it did |
|---|---|---|---|
| `LLMUpdateSettingsFrame` *(conditional)* | head | `llm` (`llm_service.py:692`) | sets the system instruction on the service |
| `LLMMessagesAppendFrame` **or** `LLMMessagesUpdateFrame` | head | `context_aggregator.user()` (proc 3) | `add_messages` / `set_messages` on the shared context |
| `LLMSetToolsFrame` | head | `context_aggregator.assistant()` (proc 7) | `set_tools` on the shared context, twice, idempotently |
| — *separate `queue_frames` call* — | | | |
| `LLMRunFrame` *(conditional)* | head | `context_aggregator.user()` (proc 3) | pushes `LLMContextFrame` downstream → inference |

Three things to take from that table into [[ch-12/read]]:

1. **Every flow frame must traverse `transport.input()` and `stt` before it does anything.** Those
   two processors do not handle any of these types, so they fall to their terminal
   `push_frame` — but they are real queue hops on a real event loop, and any processor you insert
   at the head for rule evaluation sits *in front of the state machine's own traffic*.
2. **The batch is not a transaction.** Four frames, two `queue_frames` calls, a plain `for` loop
   inside each, and no ordering guarantee against anything else on the queue.
3. **The tool swap completes at processor 7 while the inference trigger completes at processor 3.**
   They are queued in the order tools-then-run, and each is processed in order *at each processor*,
   so at processor 3 the tools frame is seen before the run frame and the context is correct. But
   the two effects finish at different depths, and §4.6 shows what happens when a barge-in lands in
   the middle.

---

## 4. `_set_node` is the whole machine

Everything Flows does at a transition happens in one 122-line method. Read it in pieces.

### 4.1 The docstring's order, and the real order

**`src/pipecat/flows/manager.py:602-623`**
```python
    async def _set_node(self, node_id: str, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Handles the complete node transition process in the following order:
        1. Execute pre-actions (if any)
        2. Set up messages (role and task)
        3. Register node functions
        4. Update LLM context with messages and tools
        5. Update state (current node and functions)
        6. Trigger LLM completion with new context
        7. Execute post-actions (if any)

        Args:
            node_id: Identifier for the new node.
            node_config: Complete configuration for the node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        if not self._initialized:
            raise FlowTransitionError(f"{self.__class__.__name__} must be initialized first")
```

Seven steps, and the docstring is accurate. But notice what step 1 being *first* means, because it
is the answer to a question §9.5 will ask: **pre-actions run before the new node's messages and
tools are installed.** "Pre" does not mean "before the transition"; the transition is already
committed to. It means "before `_update_llm_context`."

### 4.2 Setup and function-schema construction

**`src/pipecat/flows/manager.py:625-672`**
```python
        try:
            # Clear any pending transition state when starting a new node
            # This ensures clean state regardless of how we arrived here:
            # - Normal transition flow (already cleared in _check_and_execute_transition)
            # - Direct calls to set_node/set_node_from_config
            self._pending_transition = None

            self._validate_node_config(node_id, node_config)
            logger.debug(f"Setting node: {node_id}")

            # Clear any deferred post-actions from previous node
            self._action_manager.clear_deferred_post_actions()

            # Register action handlers from config
            for action_list in [
                node_config.get("pre_actions", []),
                node_config.get("post_actions", []),
            ]:
                for action in action_list:
                    self._register_action_from_config(action)

            # Execute pre-actions if any
            if pre_actions := node_config.get("pre_actions"):
                await self._execute_actions(pre_actions=pre_actions)

            # Build the node's function schemas (carrying handlers)
            new_functions: set[str] = set()

            # Mix in global functions that should be available at every node
            functions_list = self._global_functions + node_config.get("functions", [])

            standard_functions: list[FunctionSchema] = []
            for func_config in functions_list:
                if callable(func_config):
                    tool = FlowsDirectFunctionWrapper(function=func_config)
                elif isinstance(func_config, FlowsFunctionSchema):
                    tool = func_config
                else:
                    raise InvalidFunctionError(
                        f"Invalid function format in node '{node_id}'. "
                        "Use FlowsFunctionSchema or direct functions."
                    )
                standard_functions.append(await self._create_function_schema(tool))
                new_functions.add(tool.name)

            formatted_tools = (
                ToolsSchema(standard_tools=standard_functions) if standard_functions else NOT_GIVEN
            )
```

Four things to notice.

**`self._global_functions + node_config.get("functions", [])`** at `:654` — global functions are
**prepended** to every node's list. This is the mixin channel, set once in the constructor
(`global_functions=[...]`), and it maps directly onto boson's `_GLOBAL_TOOLS` (§13.2).

**`ToolsSchema(...) if standard_functions else NOT_GIVEN`** at `:670-672` — a node with no functions
produces `NOT_GIVEN`, not an empty schema. §4.3 shows what that does.

**`FlowsDirectFunctionWrapper(function=func_config)` for anything `callable`** — the direct-function
path. A bare `async def` in a node's `functions` list is wrapped and its schema derived from the
signature and docstring (§8.4).

**No `register_function` call anywhere.** The handler is stamped onto the `FunctionSchema` and the
LLM service picks it up from the context, per §3.3.

### 4.3 The role-message fork, and the context update

**`src/pipecat/flows/manager.py:674-700`**
```python
            role_message = node_config.get("role_message")
            role_messages = node_config.get("role_messages")

            if role_message and role_messages:
                logger.warning(
                    "Both 'role_message' and 'role_messages' specified; using 'role_message'"
                )

            if role_messages and not role_message:
                if not self._showed_deprecation_warning_for_role_messages:
                    self._showed_deprecation_warning_for_role_messages = True
                    warnings.warn(
                        "'role_messages' is deprecated and will be removed in 2.0.0. "
                        "Use 'role_message' (singular, str) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            # Update LLM context
            await self._update_llm_context(
                role_message=role_message,
                role_messages=role_messages if not role_message else None,
                task_messages=node_config["task_messages"],
                functions=formatted_tools,
                strategy=node_config.get("context_strategy"),
            )
            logger.debug("Updated LLM context")
```

`node_config["task_messages"]` — direct subscript, no `.get`. This is where the `Required` key
becomes load-bearing, and it would `KeyError` if `_validate_node_config` had not already raised a
`FlowError` for the same reason (§10.1).

And now the batch itself.

### 4.4 The emitted frame batch — memorize this

**`src/pipecat/flows/manager.py:762-771`**
```python
        try:
            frames = []

            # New path: role_message as LLM system instruction (persists until changed)
            if role_message:
                frames.append(
                    LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=role_message))
                )

            messages = []
```

**`src/pipecat/flows/manager.py:773-777`**
```python
            # Legacy path: role_messages prepended to context messages
            if role_messages:
                messages.extend(role_messages)

            update_config = strategy or self._context_strategy
```

**`src/pipecat/flows/manager.py:824-845`**
```python
            # Add task messages
            messages.extend(task_messages)

            # Use an "update" (replace) frame for the RESET/RESET_WITH_SUMMARY
            # strategies; otherwise append. (Note that even the first node follows
            # the same rule: appending ensures any prior context contributions,
            # such as by tts_say pre-actions, is preserved rather than replaced).
            frame_type = (
                LLMMessagesUpdateFrame
                if update_config.strategy
                in [ContextStrategy.RESET, ContextStrategy.RESET_WITH_SUMMARY]
                else LLMMessagesAppendFrame
            )

            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))

            await self._worker.queue_frames(frames)

            logger.debug(
                f"Updated LLM context using {frame_type.__name__} with strategy {update_config.strategy}"
            )
```

**Four rules. Learn them as rules, because every operational surprise in Flows is one of them
firing.**

**Rule 1 — `LLMUpdateSettingsFrame` is conditional and PERSISTENT.** It appears only
`if role_message`. There is no "clear the system instruction" path. Once node A sets a
`role_message`, the system instruction stays on the LLM service for nodes B, C and D that omit it,
until node E sets a different one. In `insurance_quote.py` exactly one node sets it — the initial
one — and the persona survives all four transitions. That is the intended usage, and
`patient_intake.py` shows the exception that proves it: `create_prescriptions_node` re-declares the
identical `role_message` string, because it also sets `ContextStrategy.RESET` and the author wanted
the persona re-asserted alongside a wiped message list.

**Rule 2 — the Append-versus-Update choice IS the `ContextStrategy`.** It is not "a strategy that
influences a frame choice"; the enum's only runtime effect in the entire package is this ternary.
`RESET` and `RESET_WITH_SUMMARY` map to `LLMMessagesUpdateFrame` (which calls `set_messages` —
replace); anything else maps to `LLMMessagesAppendFrame` (`add_messages` — add). §7 is entirely
about this line.

Read the comment above it too, because it explains a design decision you would otherwise call a bug:
*"even the first node follows the same rule: appending ensures any prior context contributions, such
as by tts_say pre-actions, is preserved rather than replaced."* The very first node of a
conversation appends into an empty context, which is identical to replacing it — unless a pre-action
already spoke a line and appended it. Append is the safe default at every node including the first.

**Rule 3 — `LLMSetToolsFrame` is UNCONDITIONAL.** No `if`. Line `:839`, every single transition,
without exception. And when a node declares no functions, `functions` is `NOT_GIVEN` (§4.2), whose
docstring in `frames.py:700-704` says it plainly: *"or ``NOT_GIVEN`` to clear tools."*

> **A node with no `functions` key does not "leave the tools alone." It CLEARS the tool set.**

This is the single most consequential asymmetry in the whole design and §13.5 is about what it
collides with in boson. There is no such thing as a tool-neutral node.

**Rule 4 — `LLMRunFrame` is conditional and QUEUED SEPARATELY.** It is not in `frames`. It is queued
at `:709`, back in `_set_node`, after `_update_llm_context` has already returned and already flushed
its batch:

**`src/pipecat/flows/manager.py:702-719`**
```python
            # Update state
            self._current_node = node_id
            self._current_functions = new_functions

            # Trigger completion with new context
            respond_immediately = node_config.get("respond_immediately", True)
            if respond_immediately:
                await self._worker.queue_frames([LLMRunFrame()])

            # Execute post-actions if any
            if post_actions := node_config.get("post_actions"):
                if respond_immediately:
                    await self._execute_actions(post_actions=post_actions)
                else:
                    # Schedule post-actions for execution after first LLM response in this node
                    self._schedule_deferred_post_actions(post_actions=post_actions)

            logger.debug(f"Successfully set node: {node_id}")
```

Why separate rather than a fifth element of `frames`? Two independent reasons, and the second is the
non-obvious one.

The first is structural: `respond_immediately` is read in `_set_node`, and `_update_llm_context` does
not receive it.

The second is that **the context frames genuinely cannot trigger inference on their own**:

**`src/pipecat/frames/frames.py:644-657`**
```python
@dataclass
class LLMMessagesAppendFrame(DataFrame):
    """Frame containing LLM messages to append to current context.

    A frame containing a list of LLM messages that need to be added to the
    current context.

    Parameters:
        messages: List of context messages to append.
        run_llm: Whether the context update should be sent to the LLM.
    """

    messages: list[LLMContextMessage]
    run_llm: bool | None = None
```

`run_llm: bool | None = None`. Flows constructs `frame_type(messages=messages)` — positional
messages, `run_llm` left at its default of `None`. And the aggregator's handler is
`if frame.run_llm:` (§3.2), which `None` fails. So the messages land in the context and **nothing
runs**. The separate `LLMRunFrame` is not stylistic; it is the only thing in the batch that starts an
inference, and `respond_immediately=False` genuinely produces a node that installs a prompt and a
tool set and then waits silently for the user to speak.

That flag is the outbound-versus-inbound switch, and §12.2 shows `restaurant_reservation.py` using
it as exactly one line.

→ **[Open the ch-10 `_set_node` emitter](./figures/flow-node-transition.html)** and drive the four
rules before reading further. Compose a `NodeConfig` with the toggles, and do these three things in
order: (1) set a `role_message` on node 1, then advance two nodes that omit it, and watch the
`LLMUpdateSettingsFrame` *not* reappear while the system instruction stays set — that is Rule 1;
(2) flip `context_strategy` from `APPEND` to `RESET` and watch the frame class change while the
running message list stops growing and starts being replaced — Rule 2; (3) delete every entry from
`functions` and watch `LLMSetToolsFrame` fire anyway with an empty set — Rule 3. The layout is the
argument: `FlowManager` is drawn outside the `Pipeline` box, and each frame animates from the head
through stt and the user aggregator before anything reaches the llm.

### 4.5 The batch is not a transaction

Say it once, precisely, because [[ch-12/read]] needs it. A node transition emits up to four frames
through **two** `queue_frames` calls, each of which is a `for` loop over `queue_frame` (§3.1), each
of which does `await self._push_queue.put(frame)`. Between any two of those puts, the event loop can
run something else — a transcription arriving, a rule processor pushing, a VAD event, another
coroutine calling `set_node_from_config`.

There is no lock, no sequence number, and no atomic batch primitive anywhere in this path.

What saves you in practice is that within a single processor, frames are dequeued in order
([[ch-04/read]] §4.2), so a `LLMSetToolsFrame` queued before an `LLMRunFrame` is *seen* before it at
processor 3. What does not save you is anything about global ordering across two producers, which
[[ch-04/read]] §10 already stated has no guarantee at all.

### 4.6 Honest finding: the batch is only PARTLY interruption-proof

Here is something no excerpt notes, and it falls straight out of [[ch-08/read]] §3.

[[ch-08/read]] established that on a barge-in, `FrameQueue.reset()` (`utils/frame_queue.py:84-95`)
drains every processor's queue and re-enqueues **only** frames carrying the `UninterruptibleFrame`
mixin. Now check the four frames a node transition emits:

| Frame | Declaration | `UninterruptibleFrame`? |
|---|---|---|
| `LLMUpdateSettingsFrame` | `frames.py:2283`, extends `ServiceUpdateSettingsFrame` | **YES** — `class ServiceUpdateSettingsFrame(ControlFrame, UninterruptibleFrame, Generic[TSettings])`, `frames.py:2251` |
| `LLMMessagesAppendFrame` | `frames.py:645`, `class LLMMessagesAppendFrame(DataFrame)` | no |
| `LLMMessagesUpdateFrame` | `frames.py:661`, `class LLMMessagesUpdateFrame(DataFrame)` | no |
| `LLMSetToolsFrame` | `frames.py:694`, `class LLMSetToolsFrame(DataFrame)` | no |
| `LLMRunFrame` | `frames.py:634`, `class LLMRunFrame(DataFrame)` | no |

The persona change survives a barge-in. The messages, the tool set, and the inference trigger do
not.

Concretely: if the customer starts speaking while a node transition's frames are still queued in
front of, say, the STT processor, `reset()` keeps the `LLMUpdateSettingsFrame` and discards the other
three. `_set_node` has already returned successfully. `self._current_node` has already been assigned
(`:703`). `FlowManager` believes it is in the new node. The LLM has the new persona, the **old**
messages, and the **old** tool set.

There is no reconciliation path. `FlowManager` never reads back what landed; §2.2 showed its only
inbound channel is three action-related frame types, none of which report on the context batch.

Is this reachable in practice? It requires a barge-in inside the window between queueing and the
frames clearing the front of the pipeline, which is short. `respond_immediately=False` nodes widen
it, because they are entered on the user's turn — precisely when the user is likely to be speaking.
A rule-driven transition fired from a transcription (the shape [[ch-12/read]] will design) is
entered during the user's turn by construction.

Write it on the constraints list for [[ch-12/read]]: **a transition fired during the user's turn can
be partially lost, and the framework will not tell you.** The mitigation is not exotic — re-assert
the node after the interruption settles, using the same public `set_node_from_config` path §12.3
shows `warm_transfer.py` using from a transport callback — but you have to know to build it.

---

## 5. Node state is a string

### 5.1 The declaration and the assignment

**`src/pipecat/flows/manager.py:147-149`**
```python
        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None
```

**`src/pipecat/flows/manager.py:702-704`**
```python
            # Update state
            self._current_node = node_id
            self._current_functions = new_functions
```

```
$ grep -n "_current_node" src/pipecat/flows/manager.py
149:        self._current_node: str | None = None
248:        return self._current_node
703:            self._current_node = node_id
```

Three sites in the whole module: the declaration, the `current_node` property's return, and the
single assignment.
`node_id` is a `str`. **`self._current_node` never holds a `NodeConfig`.**

This matters more than it sounds. It means `FlowManager` has no memory of the node it is in, only of
what that node was *called*. It cannot tell you the current node's `functions`, its
`task_messages`, or its `context_strategy`. Ask "what tools are advertised right now" and Flows has
no answer — the answer lives in the shared `LLMContext` that the aggregators wrote (§3.3), which is
a completely different object owned by completely different code.

The node object itself is garbage after `_set_node` returns. It was a `NodeConfig` dict; its
contents were decomposed into frames and pushed; the dict is dropped.

[[ch-02/read]] §11 quoted these same three lines as evidence for the frame budget and said "the
current node is a `str`." This chapter adds why that is a *design*, not a shortcut: a node in Flows
is an **event**, not an **entity**. It happens; it does not persist.

### 5.2 Node identity is decided in one line, and defaults to a UUID

**`src/pipecat/flows/types.py:509-518`**
```python
def get_or_generate_node_name(node_config: NodeConfig) -> str:
    """Get the node name from configuration or generate a UUID if not set.

    Args:
        node_config: Node configuration dictionary.

    Returns:
        Node name from config or generated UUID string.
    """
    return node_config.get("name", str(uuid.uuid4()))
```

One line of body. `name` is optional (§6.1), and when it is absent every entry into that node
produces a **fresh throwaway UUID**.

The consequence is not cosmetic:

```python
# A node without "name"
if flow_manager.current_node == "collecting_consent":   # never True
    ...
```

Enter the same unnamed node twice and `current_node` holds two different strings. Any guard, any log
correlation, any dashboard grouping by node — all broken, silently, with no warning emitted anywhere.

**Rule: set `name` on every node.** `insurance_quote.py` sets it on all five. `hello_world.py` sets
it on both — though note its end node is named `"create_end_node"`, the factory's name rather than
the node's, which is a small inconsistency in the example and a reminder that nothing validates it.

### 5.3 The guard pattern, in-tree

`warm_transfer.py` is the file that uses `current_node` for real, and it is the only concurrency
control in that example:

**`examples/flows/warm_transfer.py:650-658`**
```python
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport: DailyTransport, participant: dict[str, Any]):
            """Handle the human agent maybe having joined the call:
            - If the participant who joined is the human agent and we're currently in the "transferring_to_human_agent" node, go to the "human_agent_interaction" node.
            - Otherwise...nothing, for the purposes of this demo. We're assuming the human agent won't join while the conversation flow is any other node.
            """
            user_id = participant.get("info", {}).get("userId")
            if user_id == "agent" and flow_manager.current_node == "transferring_to_human_agent":
                await start_human_agent_interaction(flow_manager=flow_manager)
```

*(The [[flows-state-machine]] excerpt cites this guard at `warm_transfer.py:658`; the source has it
at **657**. Minor, but the rule is that the source wins.)*

`flow_manager.current_node == "transferring_to_human_agent"` is a string comparison against a
hand-written literal, in application code, with no constant and no enum. That is the whole guard.
The docstring is candid about the rest — *"Otherwise...nothing, for the purposes of this demo. We're
assuming the human agent won't join while the conversation flow is any other node."*

For a production tele-sales system this is where you supply your own discipline: an enum or a module
of constants for node names, and a guard on every out-of-band transition. Flows will not help.

### 5.4 `_current_functions`: dead in `src/`, live in tests — and why the distinction matters

§0.4 gave the grep. Here is why the precise scope changes what you can conclude.

If it were read nowhere at all, you would say: dead code, ignore it. Because it is read at six sites
in `tests/test_flows_manager.py`, you have to say something more careful, and the careful version is
more useful.

Look at what the tests assert:

**`tests/test_flows_manager.py:186-191`**
```python
        # Test valid config
        valid_config = {"name": "test", "task_messages": []}
        await flow_manager.set_node_from_config(valid_config)

        self.assertEqual(flow_manager._current_node, "test")
        self.assertEqual(flow_manager._current_functions, set())
```

**`tests/test_flows_manager.py:316-319`**
```python
        # Verify all functions were advertised (each carrying a handler) and tracked
        handlers = get_advertised_tool_handlers(self.mock_worker)
        self.assertEqual(set(handlers), {"func_0", "func_1", "func_2"})
        self.assertEqual(len(flow_manager._current_functions), 3)
```

The field is a **bookkeeping mirror the test suite uses to confirm that `_set_node` processed the
right function list**. It is asserted alongside `get_advertised_tool_handlers(self.mock_worker)`,
which reads what actually went out on the wire. `_current_functions` is the cheap internal check;
the mock-worker read is the real one.

So the two claims you may make:

- ✅ *"No runtime code path consults `_current_functions` before dispatching a function. It is not a
  gate, not an allowlist, and not a source of truth for what the LLM can call."* — provable, zero
  reads in `src/`.
- ❌ *"It is never read anywhere in the codebase."* — false, and it is the kind of false that gets
  caught in a design review.

And the practical consequence: **do not build a permission check on it.** If you are porting boson's
per-stage tool allowlist ([[boson-tool-router]]'s `_allowed_tools_var` ContextVar gate), there is no
Flows field to hang it on. What the model may call is decided entirely by what is in the shared
`LLMContext.tools` at inference time, and enforcement — if you want enforcement distinct from
exposure — has to live in your handlers or in a wrapping catch-all. §13.5 returns to this.

---

## 6. The type vocabulary

`flows/types.py` is 518 lines and contains no class hierarchy. It is `TypedDict`s, one `Enum`, three
`@dataclass`es, a pile of `Callable` aliases, and one wrapper class. The vocabulary *is* the design,
so read it as such — [[flows-node-types]] is the pre-read summary; everything below was re-checked
against the file.

### 6.1 `NodeConfig` is a `TypedDict(total=False)`

**`src/pipecat/flows/types.py:224-237`**
```python
    task_messages: Required[list[dict]]
    name: str
    role_message: str
    role_messages: list[dict[str, Any]]
    # ``FlowsFunctionSchema`` and ``FlowsDirectFunction`` are defined below
    # (see the note above ``ConsolidatedFunctionResult``); string forward
    # references keep ``NodeConfig`` definable here without re-introducing the
    # cross-module forward reference that ``ConsolidatedFunctionResult`` used
    # to require.
    functions: "list[FlowsFunctionSchema | FlowsDirectFunction]"
    pre_actions: list[ActionConfig]
    post_actions: list[ActionConfig]
    context_strategy: ContextStrategyConfig
    respond_immediately: bool
```

Nine keys, `total=False`, so everything is optional **except** the one marked `Required`.

| key | type | what it controls | notes |
|---|---|---|---|
| `task_messages` | `Required[list[dict]]` | the node's objective | **the only required key**; subscripted directly at `manager.py:696` |
| `name` | `str` | node label | absent → fresh UUID per entry (§5.2) |
| `role_message` | `str` | persona, as the LLM **system instruction** | emits `LLMUpdateSettingsFrame`; **persists across nodes** |
| `role_messages` | `list[dict[str, Any]]` | persona, as **context messages** | deprecated 1.5.0; prepended into `messages`, not a settings frame |
| `functions` | `list[FlowsFunctionSchema \| FlowsDirectFunction]` | advertised tool set | absent → `NOT_GIVEN` → **clears tools** |
| `pre_actions` | `list[ActionConfig]` | side effects before `_update_llm_context` | a raise here aborts the node (§9.5) |
| `post_actions` | `list[ActionConfig]` | side effects after the run trigger | deferred when `respond_immediately=False` |
| `context_strategy` | `ContextStrategyConfig` | per-node override of the manager default | the Append/Update fork (§7) |
| `respond_immediately` | `bool` | queue `LLMRunFrame()` on entry | default `True` |

Note the `role_message` / `role_messages` asymmetry properly, because the table hides how different
they are. `role_message` (singular) becomes a **system instruction on the LLM service**, set via
`LLMUpdateSettingsFrame`, living outside the message list, persisting until overwritten.
`role_messages` (plural, deprecated) becomes **ordinary messages prepended to `messages`**, which
means they are subject to `ContextStrategy` — a `RESET` wipes them — and they are re-added on every
node that declares them. Two different storage locations with two different lifetimes. The migration
from plural to singular was not a spelling change.

### 6.2 `ContextStrategyConfig` is a `@dataclass`, not a `TypedDict`

**`src/pipecat/flows/types.py:155-179`**
```python
@dataclass
class ContextStrategyConfig:
    """Configuration for context management.

    Parameters:
        strategy: Strategy to use for context management.
        summary_prompt: Required prompt text when using RESET_WITH_SUMMARY.

            .. deprecated:: 1.5.0
                Use ``LLMContextSummaryConfig.summarization_prompt`` instead.
                Deprecated together with ``RESET_WITH_SUMMARY``. Will be removed
                in 2.0.0.
    """

    strategy: ContextStrategy
    summary_prompt: str | None = None

    def __post_init__(self):
        """Validate configuration.

        Raises:
            ValueError: If summary_prompt is missing when using RESET_WITH_SUMMARY.
        """
        if self.strategy == ContextStrategy.RESET_WITH_SUMMARY and not self.summary_prompt:
            raise ValueError("summary_prompt is required when using RESET_WITH_SUMMARY strategy")
```

Of the two shapes you write by hand when authoring a node, **only `NodeConfig` is a `TypedDict`.**
`ContextStrategyConfig` is a real class with a real constructor and a real `__post_init__`.

The asymmetry is not decorative — it is a difference in *when you learn you were wrong*:

- `ContextStrategyConfig(strategy=RESET_WITH_SUMMARY)` with no prompt → **`ValueError` at
  construction**, at the line you wrote, with a message naming the field.
- `NodeConfig(task_mesages=[...])` with a typo → **nothing at construction**, a `FlowError` later
  from `_validate_node_config`, phrased as a missing-field error at a node you have to identify from
  the ID string.
- `NodeConfig(respond_immediatly=False)` with a typo → **nothing, ever.** The key sits in the dict
  and is never looked at. `node_config.get("respond_immediately", True)` returns `True` and your
  node speaks first when you designed it to wait.

That last one is worth an operational note because of how it presents in a voice agent. You wrote a
node meant to enter silently and wait for the customer. It talks over them instead. There is no
error, no warning, no log line. The only way to find it is to read the key name character by
character or to test the behaviour.

`ActionConfig` (`types.py:112-131`) is also a `TypedDict(total=False)` with one `Required` key
(`type`), and its docstring makes the openness explicit: *"Additional fields are allowed and passed
to the handler."* So arbitrary payload keys ride along in the same dict — which means a typo in an
action key is not just unchecked, it is *by design* indistinguishable from a payload field.

### 6.3 The edge type

**`src/pipecat/flows/types.py:260-280`**
```python
# ``ConsolidatedFunctionResult`` is the public return-type alias for "direct"
# functions. It must be defined **after** ``NodeConfig`` and without a string
# forward reference: ``get_type_hints()`` on a user-defined direct function
# resolves names against the user's module globals, not this module's, so a
# ``"NodeConfig"`` forward reference here would fail unless the user happened
# to import ``NodeConfig`` themselves.
ConsolidatedFunctionResult = tuple[Any, NodeConfig | None | _NoResponse]
"""Return type for "consolidated" functions.

Return type for "consolidated" functions that do either or both of:
- doing some work
- specifying the next node to transition to after the work is done

The first tuple element is the function-call result delivered to the LLM.
Any JSON-serializable value is accepted (matching Pipecat's upstream
``FunctionCallResultCallback`` contract). The second element is the next node
to transition to, ``None`` to stay on the current node and respond, or
:data:`NO_RESPONSE` to finish without transitioning or responding. Pass a
``None`` *result* to signal a transition-only handler; FlowManager substitutes
an acknowledgement result.
"""

```

**This one-line type alias is the entire graph representation of Pipecat Flows.**

`tuple[Any, NodeConfig | None | _NoResponse]` — element 0 goes to the LLM as the tool result,
element 1 is the edge. There is no `Edge` class, no adjacency list, no registry. An edge exists at
the moment a handler returns, and not before.

The three outcomes, and the vocabulary the source uses for them:

**`src/pipecat/flows/manager.py:496-514`**
```python
                is_no_response = next_node is NO_RESPONSE
                if is_no_response or not next_node:
                    # Node function: stay on the current node.
                    properties = FunctionCallResultProperties(
                        run_llm=not is_no_response,
                        on_context_updated=None,
                    )
                else:
                    # Edge function: transition to the returned node.
                    self._pending_transition = {
                        "next_node": next_node,
                        "function_name": name,
                        "arguments": params.arguments,
                        "result": result,
                    }
                    properties = FunctionCallResultProperties(
                        run_llm=False,
                        on_context_updated=self._check_and_execute_transition,
                    )
```

| returns | called | behaviour |
|---|---|---|
| `(result, some_node_config)` | **edge function** | transition; `run_llm=False` because the *new node* will trigger the run |
| `(result, None)` | **node function** | stay put, `run_llm=True` — do the work, let the model talk about it |
| `(result, NO_RESPONSE)` | — | stay put, `run_llm=False` — result lands in context, model stays silent |

The terms "node function" and "edge function" are the source's own (`manager.py:498, 504`), echoed in
`examples/flows/README.md:34`. Use them; they are the vocabulary the docs and the code agree on.

### 6.4 `NO_RESPONSE` is compared by identity

**`src/pipecat/flows/types.py:240-257`**
```python
class _NoResponse:
    """Type of the :data:`NO_RESPONSE` sentinel."""

    def __repr__(self) -> str:
        return "NO_RESPONSE"


NO_RESPONSE = _NoResponse()
"""Function return value (in the "next node" slot) indicating "don't immediately respond".

Return ``(result, NO_RESPONSE)`` from a "consolidated" function when the bot
should remain silent after the function finishes, rather than immediately
responding. The function result will make it into the context, but the next
response will be triggered by something else, like a user utterance.

For functions that transition to another node,
``NodeConfig.respond_immediately`` provides the equivalent control.
"""
```

A one-method private class, instantiated once at module level, and the check at `manager.py:496` is
`next_node is NO_RESPONSE` — **identity, not equality, not truthiness.** You must return the exact
imported singleton. A different `_NoResponse()` instance would fall through to `not next_node`,
which for an object with no `__bool__` is `False`, so it would be treated as an edge function and
`_set_node` would be handed an instance of `_NoResponse` as a `NodeConfig`. That fails inside
`_validate_node_config` with a `FlowError` about a missing `task_messages` field — an error message
pointing nowhere near the cause.

The docstring's last sentence names the symmetry precisely: `NO_RESPONSE` is "stay here and be
silent"; `respond_immediately=False` is "go there and be silent." Same intent, two mechanisms,
because one is a per-call decision and the other is a per-node property.

The one in-tree use is a genuine handoff:

**`examples/flows/multi_worker_handoff.py:330-334`**
```python
        # The router is now responsible for the next turn, so hand off with
        # NO_RESPONSE to avoid running the LLM. Note that we don't need to
        # transition to any next node: on_activated re-seeds party_size_node
        # when control returns to the reservation worker.
        return {"status": "transferred"}, NO_RESPONSE
```

Another worker owns the next turn. Recording the result but not speaking is exactly right. For Lina,
the shape maps onto a tool that hands control to an outside system — a payment step, an SMS send —
where the customer will be spoken to by something other than this pipeline.

### 6.5 The forward-reference trap

Read the comment at `types.py:260-265` again, quoted in §6.3. It is a real constraint on your code,
not an internal note:

> `get_type_hints()` on a user-defined direct function resolves names against the **user's** module
> globals, not this module's.

Practical rule: if you annotate a direct function `-> tuple[AgeCollectionResult, NodeConfig]`, then
`NodeConfig` must be importable **in your module**. That is why every example file has
`from pipecat.flows import FlowManager, NodeConfig` at the top even when it never constructs a
`FlowManager` by hand. If schema extraction breaks with a `NameError` mentioning a type you did not
write, this is why.

### 6.6 The other type shapes, briefly

`FlowsFunctionSchema` (`types.py:354-379`) is a `@dataclass` with `name`, `description`,
`properties`, `required`, `handler`, plus `cancel_on_interruption: bool = False` and
`timeout_secs: float | None = None`. Use it when the JSON-Schema `properties` need a shape a Python
signature cannot express — enums, nested objects, provider-specific constraints. Otherwise the
examples README states the default: *"The examples define their functions as 'direct
functions' — async functions whose schema is derived from the signature and docstring — which is the
recommended pattern"* (`examples/flows/README.md:47`).

`FlowArgs = dict[str, Any]` (`types.py:59`), with a note that 2.0.0 plans to widen it to
`Mapping[str, Any]`.

The exception hierarchy is five classes in 62 lines, all descending from one base:
`FlowError` ← `FlowInitializationError`, `FlowTransitionError`, `InvalidFunctionError`,
`ActionError` (`exceptions.py:15-61`). Every failure inside `_set_node` is re-wrapped:
`raise FlowError(f"Failed to set node {node_id}: {str(e)}") from e` (`manager.py:723`). So a single
`except FlowError` catches everything the subsystem can throw, at the cost of the original type being
one `__cause__` hop away.

### 6.7 Six exported names are already scheduled for removal in 2.0.0

`__init__.py:48-73` exports twenty-one names. These are on the way out:

| deprecated (since 1.5.0) | replacement |
|---|---|
| `FlowResult` | **none** — "No replacement."; return any JSON-serializable value |
| `role_messages` (NodeConfig key) | `role_message` (singular `str`) |
| `ContextStrategy.RESET_WITH_SUMMARY` + `ContextStrategyConfig.summary_prompt` | `LLMSummarizeContextFrame` from a pre-action |
| `ZeroArgFunctionHandler`, `LegacyFunctionHandler` | `FlowFunctionHandler` — two-arg `(args, flow_manager)` |
| `LegacyActionHandler` | `FlowActionHandler` — two-arg `(action, flow_manager)` |
| `flows_direct_function` | `flows_tool_options` |
| `FlowManager(task=...)` and `FlowManager.task` | `worker=` / `.worker` |

**Target surface for anything you build:** `NodeConfig` with a string `role_message`;
`FlowsDirectFunction` or `FlowsFunctionSchema`; two-argument handlers; `flows_tool_options`;
`ContextStrategy.APPEND` and `ContextStrategy.RESET`. Nothing else.

That list is short enough to be encouraging and long enough to be a warning: roughly a third of the
public API of this subsystem is scheduled for deletion. It is a young package inside a mature
framework, and [[ch-13/read]] should price that as an adoption cost rather than pretend it away.

---

## 7. `ContextStrategy` is the only knob in Flows that forgets

### 7.1 The whole enum, and its single effect

**`src/pipecat/flows/types.py:134-152`**
```python
class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Parameters:
        APPEND: Append new messages to existing context (default).
        RESET: Reset context with new messages only.
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary.

            .. deprecated:: 1.5.0
                Use :class:`LLMSummarizeContextFrame` instead — push it in a
                pre-action to trigger on-demand summarization during a node
                transition. See
                https://docs.pipecat.ai/guides/fundamentals/context-summarization.
                Will be removed in 2.0.0.
    """

    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"
```

Three members. The enum's only runtime consumer is the ternary at `manager.py:831-836` quoted in
§4.4. `RESET` → `LLMMessagesUpdateFrame` → `set_messages` (replace). `APPEND` →
`LLMMessagesAppendFrame` → `add_messages` (add).

That is the whole mechanism, and it is worth stating what it therefore *is*: **`ContextStrategy` is
per-transition context truncation.** A node with `RESET` starts the LLM's message list over from its
own `task_messages` and nothing else. Everything the customer said, everything the bot said, every
tool result — gone from what the model sees on the next inference.

Selection is per-node with a manager-wide default:
`update_config = strategy or self._context_strategy` (`manager.py:777`), where
`self._context_strategy` defaults to `ContextStrategyConfig(strategy=ContextStrategy.APPEND)`
(`manager.py:141-143`).

`patient_intake.py` shows the intended use — a long structured intake where each collection stage is
independent:

**`examples/flows/patient_intake.py:233-246`**
```python
def create_prescriptions_node() -> NodeConfig:
    """Create the prescriptions collection node."""
    return NodeConfig(
        name="get_prescriptions",
        role_message="You are Jessica, an agent for Tri-County Health Services. You must ALWAYS use one of the available functions to progress the conversation. Be professional but friendly.",
        task_messages=[
            {
                "role": "developer",
                "content": "This step is for collecting prescriptions. Ask them what prescriptions they're taking, including the dosage. Get to the point by saying 'Thanks for confirming that. First up, what prescriptions are you currently taking, including the dosage for each medication?'. After recording prescriptions (or confirming none), proceed to allergies.",
            }
        ],
        context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        functions=[record_prescriptions],
    )
```

Look at what `RESET` forced: the `role_message` is re-declared, verbatim, identical to the initial
node's. That is not redundancy — §6.1 established that `role_message` lives *outside* the message
list as a system instruction, so a `RESET` would not have wiped it. Re-declaring it is the author
being defensive, and it costs one extra `LLMUpdateSettingsFrame` per entry. Read it as a hint that
the persona/message split is subtle enough that even the reference example hedges.

**What survives a `RESET`:** the system instruction (`role_message`), the advertised tool set (a
separate frame writing a separate field), and `flow_manager.state` (a plain dict on the manager,
never in the context). **What does not:** every message.

That third item is the important one for §11.3: `flow_manager.state` is the channel that survives
context truncation, which is exactly why the examples put collected data there rather than relying
on the transcript.

### 7.2 `RESET_WITH_SUMMARY`: do not build on it

The deprecated third member has a mechanism worth seeing, because the failure mode is silent.

**`src/pipecat/flows/manager.py:792-822`**
```python
            if (
                update_config.strategy == ContextStrategy.RESET_WITH_SUMMARY
                and self._context_aggregator
                and self._context_aggregator.user()._context
            ):
                # We know summary_prompt exists because of __post_init__ validation in ContextStrategyConfig
                summary_prompt = cast(str, update_config.summary_prompt)
                try:
                    # Try to get summary with 5 second timeout
                    summary = await asyncio.wait_for(
                        self._create_conversation_summary(
                            summary_prompt,
                            self._context_aggregator.user()._context,
                        ),
                        timeout=5.0,
                    )

                    if summary:
                        summary_message = self._adapter.format_summary_message(summary)
                        messages.append(summary_message)
                        logger.debug(f"Added conversation summary to context: {summary_message}")
                    else:
                        # Fall back to APPEND strategy if summary fails
                        logger.warning(
                            "Failed to generate summary, falling back to APPEND strategy"
                        )
                        update_config.strategy = ContextStrategy.APPEND

                except TimeoutError:
                    logger.warning("Summary generation timed out, falling back to APPEND strategy")
                    update_config.strategy = ContextStrategy.APPEND
```

Three things, in ascending order of how much they should bother you.

**A hard 5.0-second `asyncio.wait_for`, inline in the transition path.** Not configurable. A node
transition using this strategy can block for five seconds before a single frame is queued. In a voice
conversation that is not a latency regression, it is a dead line. [[ch-11/read]] will put a number on
the whole budget; five seconds is larger than all of it.

**Silent fallback to `APPEND` on timeout or empty summary.** A `logger.warning`, and then the
conversation proceeds with the *opposite* context semantics from the one you configured. You asked
for "start clean with a summary" and got "keep everything." For a long intake this is the difference
between a 2,000-token prompt and a 30,000-token one, at runtime, decided by whether a side LLM call
came back in time.

**`update_config.strategy = ContextStrategy.APPEND` mutates the config object.** `update_config` is
`strategy or self._context_strategy` — and when the node supplied a `context_strategy`, that is *the
node factory's own object*. If your factory returns a module-level `ContextStrategyConfig` instance
rather than constructing a fresh one per call, one timeout permanently downgrades that node for the
rest of the process. Nothing restores it.

And `flows/adapters.py` is not what its name suggests. It is 68 lines containing one two-method
class that exists solely for this strategy:

**`src/pipecat/flows/adapters.py:22-37`**
```python
class LLMAdapter:
    """Helpers for generating and formatting conversation summaries."""

    def format_summary_message(self, summary: str) -> dict:
        """Format a summary as a developer message.

        Summary messages use the LLMContextMessage format (OpenAI-style),
        as summarization triggers an LLMMessagesUpdateFrame.

        Args:
            summary: The generated summary text.

        Returns:
            A developer message containing the summary.
        """
        return {"role": "developer", "content": f"Here's a summary of the conversation:\n{summary}"}
```

If you expected a per-provider function-schema adapter — the thing [[ch-09/read]] §4 found in
`src/pipecat/adapters/` — this is not it. Flows sidesteps provider shaping entirely by emitting a
provider-neutral `ToolsSchema(standard_tools=[...])` (`manager.py:670-672`) and letting the LLM
service's own adapter convert it. `flows/adapters.py` does summaries and nothing else.

**What to do instead**, per the deprecation body itself: `ContextStrategy.RESET` on the node, plus a
pre-action pushing `LLMSummarizeContextFrame` if you actually want a summary. That moves the
summarization onto Pipecat's native, out-of-band path instead of blocking the transition.

### 7.3 Why this is the one thing boson gains for free

Hold this for §13.6, but note it now while the mechanism is fresh. boson's `_inject_stage` only ever
**appends** a `<system-reminder>`-wrapped stage prompt ([[boson-stage-machine]]). There is no reset
path anywhere in the stage layer. So a long Lina call that walks introduction → product_focused →
consultation → informed_consent → purchase drags every prior stage's prompt through every later
inference, forever, growing.

`ContextStrategyConfig(strategy=ContextStrategy.RESET)` on a node is that problem solved in one key.
It is the single clearest thing Flows offers that boson does not currently have.

---

## 8. Handler dispatch is arity introspection, not typing

### 8.1 Three branches on `len(sig.parameters)`

**`src/pipecat/flows/manager.py:409-441`**
```python
        # Get the function signature
        sig = inspect.signature(handler)

        # Calculate effective parameter count
        effective_param_count = len(sig.parameters)

        # Handle different function signatures. inspect.signature has already
        # proven the shape, so each cast narrows the union to the branch we know
        # we're in.
        if effective_param_count == 0:
            if not self._showed_deprecation_warning_for_zero_arg_handler:
                self._showed_deprecation_warning_for_zero_arg_handler = True
                warnings.warn(
                    "Zero-argument function handlers are deprecated and will be "
                    "removed in 2.0.0. Update handlers to accept "
                    "(args: FlowArgs, flow_manager: FlowManager) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return await cast(ZeroArgFunctionHandler, handler)()
        elif effective_param_count == 1:
            if not self._showed_deprecation_warning_for_legacy_handler:
                self._showed_deprecation_warning_for_legacy_handler = True
                warnings.warn(
                    "Single-argument (legacy) function handlers are deprecated "
                    "and will be removed in 2.0.0. Update handlers to accept "
                    "(args: FlowArgs, flow_manager: FlowManager) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return await cast(LegacyFunctionHandler, handler)(args)
        else:
            return await cast(FlowFunctionHandler, handler)(args, self)
```

`0` / `1` / `else`. Not a `Protocol`, not `isinstance`, not a registration flag — a parameter count.
The `cast()` calls are pure type-checker appeasement; nothing is verified at runtime beyond the
count. The one-shot deprecation booleans from §1.3 are here.

The consequence to hold onto: **the union `FunctionHandler = ZeroArgFunctionHandler |
LegacyFunctionHandler | FlowFunctionHandler` is discriminated by counting, so a handler with three
parameters lands in `else` and is called with two.** `TypeError`. Nothing in Flows checks the upper
bound.

### 8.2 `actions.py` does the same thing with a different threshold

**`src/pipecat/flows/actions.py:178-209`**
```python
                # Determine if handler can accept flow_manager argument by inspecting its signature
                # Handlers can either take (action) or (action, flow_manager)
                try:
                    sig = inspect.signature(handler)
                    can_handle_flow_manager_arg = len(sig.parameters) > 1
                except (ValueError, TypeError):
                    logger.warning(
                        f"Unable to determine handler signature for action type '{action_type}', "
                        "falling back to legacy single-parameter call"
                    )
                    can_handle_flow_manager_arg = False

                # Invoke handler appropriately, with async and flow_manager arg as needed
                if can_handle_flow_manager_arg:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(action, self._flow_manager)
                    else:
                        handler(action, self._flow_manager)
                else:
                    if not self._showed_deprecation_warning_for_legacy_action_handler:
                        self._showed_deprecation_warning_for_legacy_action_handler = True
                        warnings.warn(
                            "Single-argument (legacy) action handlers are deprecated "
                            "and will be removed in 2.0.0. Update handlers to accept "
                            "(action: dict, flow_manager: FlowManager) instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                    if asyncio.iscoroutinefunction(handler):
                        await handler(action)
                    else:
                        handler(action)
```

Two-way (`> 1`) instead of three-way, plus a sync/async branch — action handlers may be plain `def`.
Note the `except (ValueError, TypeError)` fallback: `inspect.signature` fails on some C builtins and
on certain `functools.partial` shapes, and the recovery is to assume the legacy one-arg form.

### 8.3 Pipecat's own built-ins trip Pipecat's own deprecation warning

Now put §8.2 next to `actions.py:104-106`:

```python
        self._register_action("tts_say", self._handle_tts_action)
```

`self._handle_tts_action` is a **bound method**. Its declaration is
`async def _handle_tts_action(self, action: dict) -> None:` (`actions.py:302`), and once bound,
`self` is gone from the signature. `len(inspect.signature(handler).parameters)` is therefore **1**.

So `can_handle_flow_manager_arg` is `False`, the legacy branch runs, and the first `tts_say` action
in any Flows program emits:

```
DeprecationWarning: Single-argument (legacy) action handlers are deprecated and will be
removed in 2.0.0. Update handlers to accept (action: dict, flow_manager: FlowManager) instead.
```

...about Pipecat's own built-in. All three built-ins have this shape (`_handle_tts_action(self,
action)`, `_handle_end_action(self, action)`, `_handle_function_action(self, action)`).

It is harmless — the one-shot flag means it fires once per `ActionManager` — but it is worth knowing
for two reasons. First, if you turn `DeprecationWarning` into an error in tests (a reasonable thing
to do), Flows will fail on its own defaults. Second, it is a small proof that arity sniffing does not
distinguish "legacy handler" from "bound method that happens to have one visible parameter." Your own
bound-method handlers will hit the same branch, silently, and be called without the `flow_manager`
argument you expected — which surfaces as a `TypeError` if you declared `(self, action,
flow_manager)`, because that binds to two visible parameters and takes the *modern* branch. The two
cases are one parameter apart and behave completely differently.

**Rule: register plain module-level `async def` handlers, not bound methods.** If you must use a
method, make it a `@staticmethod` so the visible parameter count matches the declared shape.

### 8.4 Direct functions: `flow_manager` by keyword, and the name is literal

**`src/pipecat/flows/types.py:461-468`**
```python
    @classmethod
    def special_first_param_name(cls) -> str:
        """Get the special first parameter name for Flows direct functions.

        Returns:
            The string "flow_manager" which is expected as the first parameter.
        """
        return "flow_manager"
```

**`src/pipecat/flows/types.py:496-506`**
```python
    async def invoke(self, args: Mapping[str, Any], flow_manager: "FlowManager"):
        """Invoke the wrapped function with the provided arguments.

        Args:
            args: Arguments to pass to the function.
            flow_manager: FlowManager instance for function execution context.

        Returns:
            The result of the function call.
        """
        return await self.function(flow_manager=flow_manager, **args)
```

`self.function(flow_manager=flow_manager, **args)` — **by keyword.** The first parameter of a direct
function must be *named* `flow_manager`. Not `fm`, not `manager`, not `flow`. Rename it and you get a
`TypeError` about an unexpected keyword argument at call time, not at registration time.

And direct functions must return a tuple:

**`src/pipecat/flows/manager.py:483-490`**
```python
                else:
                    result = handler_response
                    next_node = None
                    # FlowsDirectFunctions should always be "consolidated" functions that return a tuple
                    if isinstance(handler, FlowsDirectFunctionWrapper):
                        raise InvalidFunctionError(
                            f"Direct function {name} expected to return a tuple (result, next_node) but got {type(result)}"
                        )
```

A `FlowsFunctionSchema` handler may return a bare result; a direct function may not. Two shapes, two
contracts.

Finally, note the error containment in `transition_func`:

**`src/pipecat/flows/manager.py:518-521`**
```python
            except Exception as e:
                logger.error(f"Error in transition function {name}: {str(e)}")
                error_result = {"status": "error", "error": str(e)}
                await params.result_callback(error_result)
```

**A raising handler never crashes the turn.** The exception becomes `{"status": "error", "error":
"..."}` delivered through the ordinary result callback, so the LLM sees a failed tool call and can
talk about it. No transition happens — `_pending_transition` was never set. For a sales call this is
the right default: a database timeout during `verify_personal_info` becomes "죄송합니다, 잠시만요"
rather than a dropped call. But it also means **your handler's exceptions are invisible unless you
watch the logs**, since nothing upstream is told.

### 8.5 The transition is gated on in-flight tool calls

One more mechanism that only exists because Flows is outside the pipeline. An edge function does not
transition immediately; it stores the intent and registers a callback:

**`src/pipecat/flows/manager.py:525-550`**
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

    async def _execute_transition(self, transition_info: dict[str, Any]) -> None:
        """Execute the stored transition."""
        next_node = transition_info.get("next_node")

        try:
            if next_node:
                node_name = get_or_generate_node_name(next_node)
                logger.debug(f"Transition to function-returned node: {node_name}")
                await self._set_node(node_name, next_node)
        except Exception as e:
            logger.error(f"Error executing transition: {str(e)}")
            raise
```

`if not assistant_aggregator.has_function_calls_in_progress` is the concurrency guard, and it reads a
field on a **processor inside the pipeline** — the assistant aggregator [[ch-09/read]] §5 showed
owns the function-call lifecycle. `FlowManager` cannot know about in-flight tool calls by observing
frames; it has no position from which to observe. So it reaches into the aggregator's public state
directly.

What this buys: when the model emits a parallel batch of three tool calls and one of them returns an
edge, the transition waits for all three. Without it, `_set_node` would fire `LLMSetToolsFrame` and
clear the tool set while two calls were still resolving.

What it costs: a second hard reference from outside the pipeline into a specific processor's internal
state, on top of the `_worker` reference. `FlowManager` is coupled to the aggregator pair by
`get_current_context()` too (`manager.py:349`: `context = self._context_aggregator.user()._context` —
note the underscore-prefixed attribute). Being outside the pipeline does not mean being decoupled
from it; it means the coupling is by attribute access instead of by frame flow.

---

## 9. Actions: exactly three verbs, and the maintainers want you to use one

### 9.1 The complete built-in vocabulary

**`src/pipecat/flows/actions.py:103-106`**
```python
        # Register built-in actions
        self._register_action("tts_say", self._handle_tts_action)
        self._register_action("end_conversation", self._handle_end_action)
        self._register_action("function", self._handle_function_action)
```

Three lines. That is the entire built-in action vocabulary of Pipecat Flows ([[flows-actions]]).

There is **no** `transition` action, no `set_node`, no `inject_message`, no `log`, no `webhook`, no
`http`, no `wait`, no `branch`. I looked for each; none exist. Everything that is not literally
speaking a line of text or hanging up is expressed as `{"type": "function", "handler": fn}`.

### 9.2 What each one does

**`tts_say`** — `actions.py:302-332`. Requires `text`; optional `append_text_to_context` (default
`True`). Queues `TTSSpeakFrame(text=..., append_to_context=...)` then `ActionFinishedFrame()`. A
missing `text` **logs an error and returns** — it does not raise, so a malformed `tts_say` is a
silent no-op that still counts as a successfully executed action for §9.6's wait table.

**`end_conversation`** — `actions.py:334-362`. Optional `text` goodbye line, then `EndFrame()`. Note
the comment:

**`src/pipecat/flows/actions.py:359-362`**
```python
        await self._worker.queue_frame(EndFrame())

        # NOTE: there's no point queueing an ActionFinishedFrame here, since the previously-queued
        # EndFrame ensures that it'll never get delivered to our observer
```

And `execute_actions` breaks out of the loop after it:

**`src/pipecat/flows/actions.py:215-219`**
```python
                # If action was end_conversation, break
                # (If we didn't, we could end up waiting for the next actions to finish, and...they
                # never would)
                if action_type == "end_conversation":
                    break
```

Anything you list after `end_conversation` in a `post_actions` array **never runs**. Not a bug — the
comment explains that continuing would deadlock the wait table against frames the `EndFrame` has
already made undeliverable. But it does mean `post_actions=[{"type": "end_conversation"}, {"type":
"function", "handler": log_call_outcome}]` silently drops your logging.

The terminal-node pattern works because of a mechanism [[ch-04/read]] §8 already established:
`EndFrame` is a `ControlFrame` processed in order, so it queues *behind* the `LLMRunFrame` and the
speech it produces, and the pipeline tears down after the goodbye has played out rather than during
it. Both `hello_world.py:112` and `insurance_quote.py:297` rely on that ordering.

**`function`** — `actions.py:364-388`. Requires `handler`; queues a `FunctionActionFrame`.

**`src/pipecat/flows/actions.py:380-388`**
```python
        # Mark that we're starting the action
        self._increment_ongoing_actions_count()

        # Queue the action frame (we're queueing rather than running it here to ensure it happens
        # at the appropriate time in the pipeline, like when the bot's turn is over, for example).
        await self._worker.queue_frame(FunctionActionFrame(action=action, function=handler))

        # NOTE: we do NOT queue an ActionFinishedFrame here; instead, we will decrement the ongoing
        # actions count when the function has finished executing (the function may take some time)
```

Read the parenthetical: *"we're queueing rather than running it here to ensure it happens at the
appropriate time in the pipeline."* **This is the entire reason `function` exists as an action type
rather than as "just call the function."** The handler does not run when the action is scheduled; it
runs when the frame reaches the pipeline's downstream end (§2.2), which is *after* everything queued
before it has been processed by every processor — including the TTS audio. That is how
`warm_transfer.py` starts hold music at the moment the bot finishes saying "please hold" rather than
while it is still speaking.

This is also the third of [[ch-02/read]]'s two frames earning its place: `FunctionActionFrame` exists
because ordering a side effect against speech requires being *in* the stream, and a plain method call
is not in the stream.

### 9.3 The maintainers de-emphasize custom actions, in a comment

**`src/pipecat/flows/actions.py:286-297`**
```python
        else:
            # Either previous action was:
            # - None (the upcoming action is the first one), so there's nothing to wait for.
            # - A fully custom action, where we don't wait, like we've always done. Note that we
            #   could, in the future, add new API affordances for users to tell us to wait for the
            #   the action to finish before moving on to the next one along with a way for them to
            #   tell us when the action is done. But let's hold off on doing that since we're
            #   de-emphasizing custom actions in favor of "function" actions, which should meet most
            #   needs.
            # Note that it should not be possible for the previous action to be "end_conversation",
            # since we stop processing actions after that one.
            pass
```

*"we're de-emphasizing custom actions in favor of `function` actions, which should meet most needs."*
A design direction stated in a source comment rather than in docs. Take it: register a custom action
type only when you need the same verb across many nodes and want the config to read declaratively.
Otherwise `{"type": "function", "handler": fn}` and be done.

The trade-off is concrete and lives in §9.6's table: a custom action type gets **no ordering
support** — the wait table never waits for one, in either direction. A `function` action always
blocks. So the escape hatch is also the one with correct ordering, and the "nicer" API is the one
that silently races.

### 9.4 What actions cannot do

Three absences, and they are absences of the right kind:

1. **An action cannot choose the next node.** There is no action type that transitions. Node
   selection lives entirely in `ConsolidatedFunctionResult`'s second element (§6.3), or in an
   out-of-band `set_node_from_config` call.
2. **An action cannot veto a transition.** There is no return value the framework inspects; the
   modern signature is `Callable[[dict, FlowManager], Awaitable[None]]` — it returns `None`.
3. **An action cannot return a value to anything.** Not to the LLM, not to the node, not to
   `FlowManager`. It communicates only by side effect: mutating `flow_manager.state`, queueing
   frames, calling out to your own systems.

For boson this is the single largest vocabulary mismatch, because
`ActionType = Literal["continue", "respond", "inject", "compact", "pre_tool", "stage_transition",
"filter", "pass"]` includes `stage_transition`, `filter` and `pass` — a transition verb and two
routing verdicts. None of the three has an action-shaped home in Flows. §13.3 and [[ch-12/read]]
handle where they go instead.

### 9.5 A raising PRE-action aborts the node; a post-action failure cannot

There is no veto API, but there **is** an ordering fact that behaves like one.

**`src/pipecat/flows/actions.py:220-224`**
```python
            except Exception as e:
                # Undo any increment of ongoing actions count that happened during this action
                if self._ongoing_actions_count > ongoing_actions_count:
                    self._decrement_ongoing_actions_count()  # Assumption: on increment per action
                raise ActionError(f"Failed to execute action {action_type}: {str(e)}") from e
```

An exception in a handler becomes `ActionError`, which propagates out of `_execute_actions`, out of
`_set_node`'s `try`, and is re-wrapped at `manager.py:721-723` as
`FlowError(f"Failed to set node {node_id}: ...")`.

Now recall the ordering from §4.1:

```
pre-actions (:647-648)  →  _update_llm_context (:693)  →  _current_node = node_id (:703)
                        →  LLMRunFrame (:709)          →  post-actions (:712-714)
```

- **A raising pre-action** happens before `_update_llm_context`. No frames were queued.
  `self._current_node` was not assigned. **The node is effectively aborted** and the caller gets a
  `FlowError`. This is a real, usable guard: put a validation check in a pre-action and a failure
  keeps the conversation where it was.
- **A raising post-action** happens after all of it. The frames are queued and travelling, the node
  ID is set, the inference is running. **The node cannot be un-set.** You get a `FlowError` describing
  a transition that has already, irreversibly, happened.

Note also that the pre-action guard is *only* reliable for `respond_immediately=True` semantics of
"before the context changed." It does not roll anything back; it just runs early enough that there is
nothing to roll back.

### 9.6 The wait table is hard-coded and not user-configurable

**`src/pipecat/flows/actions.py:264-285`**
```python
        needs_wait = False
        if previous_action_type == "tts_say":
            # "tts_say" enqueues a TTSSpeakFrame, which has an effect when it hits the TTS node in
            # the pipeline.
            # As long as the upcoming action enqueues a frame with an effect at the same point or
            # later in the pipeline, we don't need to wait.
            # If the upcoming action is:
            # - "tts_say": no need to wait (effect happens at the same point)
            # - "end_conversation": no need to wait (effect happens at the end of the pipeline)
            # - "function": no need to wait (effect happens at the end of the pipeline)
            #  - None: wait (we're done with this set of actions; the next thing to occur may be a
            #    node change/LLM context update, which has an effect earlier in the pipeline)
            # - custom action: wait (we don't know what it will do)
            if upcoming_action_type not in ["tts_say", "end_conversation", "function"]:
                needs_wait = True  # None or custom action
        elif previous_action_type == "function":
            # "function" enqueues a FunctionActionFrame, which has an effect at the end of the
            # pipeline.
            # Functions can take some time to execute (and don't hold up the pipeline as they're
            # doing so), so we need to wait for them to finish before proceeding with the next
            # action or moving on from the current set of actions.
            needs_wait = True
```

| previous | upcoming | waits? | why |
|---|---|---|---|
| `None` (first action) | anything | no | nothing in flight |
| `tts_say` | `tts_say` / `end_conversation` / `function` | no | effect lands at the same depth or later |
| `tts_say` | custom type, or `None` (end of batch) | **yes** | unknown depth, or a context update is next and lands *earlier* |
| `function` | anything, including `None` | **yes, always** | may take time, and the pipeline does not hold for it |
| custom type | anything | no | the deliberate gap from §9.3 |

The reasoning is genuinely about **pipeline depth**, not about time. A frame's effect lands at a
fixed position in the pipe, so you only need to wait when the next thing acts *earlier* than the
thing in flight. That is a compositional argument about a pipes-and-filters system —
[[ch-01/read]]'s topology showing up as a scheduling rule.

The bookkeeping is a counter plus an event:

**`src/pipecat/flows/actions.py:390-400`**
```python
    def _increment_ongoing_actions_count(self) -> None:
        """Increment the count of ongoing actions and reset the finished event if this is the first action."""
        self._ongoing_actions_count += 1
        if self._ongoing_actions_count == 1:
            self._ongoing_actions_finished_event.clear()

    def _decrement_ongoing_actions_count(self) -> None:
        """Decrement the count of ongoing actions and set the finished event if this was the last action."""
        self._ongoing_actions_count = max(0, self._ongoing_actions_count - 1)
        if self._ongoing_actions_count == 0:
            self._ongoing_actions_finished_event.set()
```

**The decrement is driven by frames arriving downstream, not by the handler returning** — that is
§2.2's `on_frame_reached_downstream`. Which loops back to §2.3: wipe the downstream filter and this
counter never decrements, and `await self._ongoing_actions_finished_event.wait()` at `actions.py:300`
hangs forever.

### 9.7 Built-ins are overridable by name

`_register_action` is a plain dict write:

**`src/pipecat/flows/actions.py:129-142`**
```python
    def _register_action(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say").
            handler: Async or sync function that handles the action.

        Raises:
            ValueError: If handler is not callable.
        """
        if not callable(handler):
            raise ValueError("Action handler must be callable")
        self._action_handlers[action_type] = handler
        logger.debug(f"Registered handler for action type: {action_type}")
```

No collision check. So `flow_manager.register_action("tts_say", my_handler)` silently replaces the
built-in, and the in-tree example does exactly that with `end_conversation`:

**`examples/flows/multi_worker_handoff.py:338-341`**
```python
    async def end_conversation_action(action: dict) -> None:
        await worker.end(reason=action.get("reason"))

    flow_manager.register_action("end_conversation", end_conversation_action)
```

In a multi-worker setup, "end the conversation" must end *this worker*, not queue an `EndFrame` into
a shared pipeline. Overriding by name is the supported way to say so.

But note what the override loses: `_maybe_wait_for_ongoing_actions_to_finish` still special-cases the
*string* `"end_conversation"` (`actions.py:218`, and the note at `:295-296`), so the `break` behaviour
and the ordering assumptions still apply to your replacement even though it does entirely different
things. The wait table keys on names, not on behaviour.

The implicit registration path is the other route:

**`src/pipecat/flows/manager.py:379-392`**
```python
        action_type = action.get("type")
        handler = action.get("handler")

        # Register action if not already registered
        if action_type and action_type not in self._action_manager._action_handlers:
            # Register handler if provided
            if handler and callable(handler):
                self.register_action(action_type, handler)
                logger.debug(f"Registered action handler from config: {action_type}")
            else:
                raise ActionError(
                    f"Action '{action_type}' not registered. "
                    "Provide handler in action config or register manually."
                )
```

An inline `handler` in an `ActionConfig` self-registers on first use. Note `if action_type not in
self._action_manager._action_handlers` — **first registration wins**, so a second node declaring the
same type with a different handler is silently ignored. If you use the inline pattern for a verb that
means different things in different nodes, only the first node's handler ever runs.

---

## 10. What Flows does not give you

This is the section to read twice if you are considering a port, because the absences are structural
rather than missing-feature-shaped.

### 10.1 `_validate_node_config` checks exactly two things

**`src/pipecat/flows/manager.py:867-898`**
```python
    def _validate_node_config(self, node_id: str, config: NodeConfig) -> None:
        """Validate the configuration of a conversation node.

        This method ensures that:
        1. Required fields (task_messages) are present.
        2. Each function is either a ``FlowsFunctionSchema`` or a valid direct
           function.

        Args:
            node_id: Identifier for the node being validated.
            config: Complete node configuration to validate.

        Raises:
            FlowError: If required fields are missing.
            InvalidFunctionError: If function format is invalid.
        """
        # Check required fields
        if "task_messages" not in config:
            raise FlowError(f"Node '{node_id}' missing required 'task_messages' field")

        # Get functions list with default empty list if not provided
        functions_list = config.get("functions", [])

        # Validate each function configuration if there are any
        for func in functions_list:
            if callable(func):
                FlowsDirectFunctionWrapper.validate_function(func)
            elif not isinstance(func, FlowsFunctionSchema):
                raise InvalidFunctionError(
                    f"Invalid function format in node '{node_id}'. "
                    "Use FlowsFunctionSchema or direct functions."
                )
```

That is the whole method, and the whole validation surface of Pipecat Flows.

1. Is `"task_messages"` a key? (Not: is it a non-empty list. Not: are its entries well-formed
   messages. `task_messages=[]` passes.)
2. Is each entry of `functions` callable or a `FlowsFunctionSchema`?

**There is no check of the current node against the target node. Anywhere.** Not here, not in
`_set_node`, not in `_execute_transition`, not in `set_node_from_config`.

```
$ grep -rn "not allowed\|illegal transition\|invalid transition" src/pipecat/flows/
(no output)
```

Which means: from the terminal `end` node of `insurance_quote.py`, a handler returning
`create_initial_node()` transitions cleanly back to the introduction. Flows will validate that the
initial node has `task_messages`, find that it does, and go. No error, no warning, no log line above
`debug`.

### 10.2 There is no node registry

`_set_node` takes a `NodeConfig` **object**. It does not look anything up. There is no dict of known
nodes, no `register_node`, no `has_node(name)`, no `get_node(name)`, no enumeration.

So the following are all impossible with the API as shipped:

- **Enumerate the graph.** You cannot ask "what nodes exist" — the answer lives in whichever Python
  functions you happened to write.
- **Validate an initial node.** Nothing can check that a configured starting node exists, because
  existence is not a property Flows can evaluate.
- **Draw the graph.** The edges are `return` statements inside handler bodies. Static extraction
  would require parsing your source.
- **Assert reachability or termination.** No structure to analyse.

Compare boson, from [[boson-stage-machine]]: `load_stages(config, prompts)` builds a
`dict[str, StageDefinition]` keyed by name from `agents/*/stage_config.py` plus `stages/*.md`, and
`StageMachine.transition(from_stage, to_stage)` returns
`TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")` when the target is not
in the current stage's `transitions` list. The graph is a data structure you can read, print, diff in
a pull request, and test.

Neither of those two properties — the registry and the edge check — has any counterpart in
`flows/types.py` or `flows/manager.py`.

### 10.3 What this means concretely

State the trade in both directions, because it genuinely goes both ways.

**What Flows gives you that is hard:** the atomic-ish swap of system instruction, message list, and
tool array, correctly ordered, correctly sequenced against in-flight function calls (§8.5), with the
inference trigger and the pre/post-action ordering handled. That is real work and it is the part that
is tedious to get right.

**What Flows does not give you at all:** transition legality, node registration, graph
introspection, and any notion of a node knowing its successors.

So the shape of a serious port is not "replace the stage machine with `FlowManager`." It is: **keep
your validator, throw away your prompt-and-tool plumbing.** `StageMachine.transition()` stays as a
pure pre-check in front of `set_node_from_config`; `build_stage_injection` and
`StageContext.filter_tools` are what Flows replaces. §13 works through the mapping field by field.

---

## 11. `examples/flows/insurance_quote.py` — read the real graph

380 lines, five node factories, and a customer who has to say their age and marital status out loud
to a bot. This is the closest thing in the Pipecat tree to what you are building, so read it as
source material rather than as a demo.

### 11.1 The graph

Five factories, four tool edges plus one self-loop:

```
create_initial_node()
   └─ collect_age(age) ──────────────────────────────▶ marital_status

create_marital_status_node()
   └─ collect_marital_status(marital_status) ────────▶ quote_calculation

create_quote_calculation_node(age, marital_status)
   └─ calculate_quote(age, marital_status) ──────────▶ quote_results

create_quote_results_node(quote)
   ├─ update_coverage(coverage_amount, deductible) ──▶ quote_results   (self-loop)
   └─ end_quote() ───────────────────────────────────▶ end

create_end_node()
   post_actions=[{"type": "end_conversation"}], no functions
```

Note the arities of the factories, because they are the design. `create_initial_node()` and
`create_marital_status_node()` take nothing. `create_quote_calculation_node(age, marital_status)`
takes two. `create_quote_results_node(quote)` takes a dict. **A node factory's parameters are the
data that node needs baked into its prompt.**

### 11.2 The initial node sets the persona once

**`examples/flows/insurance_quote.py:211-237`**
```python
def create_initial_node() -> NodeConfig:
    """Create the initial node asking for age."""
    return NodeConfig(
        name="initial",
        role_message="You are a friendly insurance agent. Your responses will be converted to audio, so avoid special characters. Always use the available functions to progress the conversation naturally.",
        task_messages=[
            {
                "role": "developer",
                "content": "Start by asking for the customer's age.",
            }
        ],
        functions=[collect_age],
    )


def create_marital_status_node() -> NodeConfig:
    """Create node for collecting marital status."""
    return NodeConfig(
        name="marital_status",
        task_messages=[
            {
                "role": "developer",
                "content": "Ask about the customer's marital status for premium calculation.",
            }
        ],
        functions=[collect_marital_status],
    )
```

The persona is one string on node 1 and **appears nowhere else in the file**. Rule 1 from §4.4 is why
that works: `role_message` becomes a system instruction that persists across every later transition.

Three things about the persona string are voice-agent discipline rather than prompt style, and all
three transfer to Korean:

- *"Your responses will be converted to audio, so avoid special characters."* — the model does not
  otherwise know it is speaking.
- *"Always use the available functions to progress the conversation naturally."* — with only one
  function advertised, this is how the model is told the function *is* the way forward.
- Everything else about what to do lives in `task_messages`, not here.

And `create_marital_status_node` is the pattern for every non-initial node: a name, one imperative
developer message, one function. That is it. Eleven lines.

### 11.3 The results node — the most instructive block in the file

**`examples/flows/insurance_quote.py:258-281`**
```python
def create_quote_results_node(
    quote: QuoteCalculationResult | CoverageUpdateResult,
) -> NodeConfig:
    """Create node for showing quote and adjustment options."""
    return NodeConfig(
        name="quote_results",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Quote details:\n"
                    f"Monthly Premium: ${quote['monthly_premium']:.2f}\n"
                    f"Coverage Amount: ${quote['coverage_amount']:,}\n"
                    f"Deductible: ${quote['deductible']:,}\n\n"
                    "Explain these quote details to the customer. When they request changes, "
                    "use update_coverage to recalculate their quote. Explain how their "
                    "changes affected the premium and compare it to their previous quote. "
                    "Ask if they'd like to make any other adjustments or if they're ready "
                    "to end the quote process."
                ),
            }
        ],
        functions=[update_coverage, end_quote],
    )
```

*(The [[flows-insurance-example]] excerpt cites this block at `L262-281`; the `def` actually starts at
**258**. The source wins.)*

Four things, and the first is the one to steal.

**`${quote['monthly_premium']:.2f}` — the number is formatted in Python, not by the model.** The
premium arrives as a float: `rates["base_rate"] * rates["risk_multiplier"]` = `150 * 1.5` = `225.0`.
Interpolated raw, the model would receive `225.0` and might well say "two hundred twenty five point
zero." `:.2f` makes it `225.00` and `:,` makes coverage `250,000`. **Any number a voice agent speaks
should be pre-formatted in the prompt.** For Korean this is more urgent, not less: 월 보험료
`45,000원` versus `45000.0`, and 만/억 grouping that a model will get wrong at some rate you cannot
measure and cannot fix after the customer has heard it.

**The self-loop is just a function that returns the node it was called from.** `update_coverage`
returns `create_quote_results_node(result)` — a *new* `NodeConfig` with the same `name` and a
different prompt. There is no loop construct. Re-entering a node means calling the factory again with
new data, and every re-entry re-runs the entire `_set_node` batch: same tools re-advertised, new
`task_messages` appended (default `APPEND`, so the *previous* quote's numbers are still in the
context — which is what makes *"compare it to their previous quote"* possible at all).

**`functions=[update_coverage, end_quote]` is the rail guard.** This node has exactly two legal
moves. The customer cannot be routed anywhere else because nothing else is advertised — §10.1 showed
there is no legality check, but there does not need to be one for the *model-driven* path, because a
node's function list is the model's entire menu. The legality gap only opens on the out-of-band path
(§12.3, [[ch-12/read]]).

**Terminal nodes carry `post_actions` and no functions.**

**`examples/flows/insurance_quote.py:284-298`**
```python
def create_end_node() -> NodeConfig:
    """Create the final node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Thank the customer for their time and end the conversation. "
                    "Mention that a representative will contact them about the quote."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )
```

No `functions` key at all. Per Rule 3, this **clears the tool set** — which is exactly right for a
terminal node, and is the one case where the unconditional `LLMSetToolsFrame` is a feature rather
than a hazard. The model physically cannot call anything while saying goodbye.

### 11.4 The three channels that carry data forward

The handlers:

**`examples/flows/insurance_quote.py:114-146`**
```python
async def collect_age(
    flow_manager: FlowManager, age: int
) -> tuple[AgeCollectionResult, NodeConfig]:
    """Record customer's age.

    Args:
        age (int): The customer's age.
    """
    logger.debug(f"collect_age handler executing with age: {age}")

    flow_manager.state["age"] = age
    result = AgeCollectionResult(age=age)

    next_node = create_marital_status_node()

    return result, next_node


async def collect_marital_status(
    flow_manager: FlowManager, marital_status: str
) -> tuple[MaritalStatusResult, NodeConfig]:
    """Record marital status after customer provides it.

    Args:
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    logger.debug(f"collect_marital_status handler executing with status: {marital_status}")

    result = MaritalStatusResult(marital_status=marital_status)

    next_node = create_quote_calculation_node(flow_manager.state["age"], marital_status)

    return result, next_node
```

Three channels, used for three different jobs:

| channel | visible to the LLM? | survives `ContextStrategy.RESET`? | job |
|---|---|---|---|
| the function's **return value** (tuple element 0) | **yes** — delivered as the tool result | it is a message, so **no** | tell the model the call succeeded and what it produced |
| **`flow_manager.state`** | **no** — a plain dict on the manager | **yes** — never in the context | hold the application's data |
| the **f-string in `task_messages`** | **yes** — it *is* the next prompt | it is the new node's message, so it is what remains | steer the next utterance |

The third one is the one people underuse. The only channel that reliably controls **what the bot says
next** is the text you interpolate into the next node's `task_messages`. A tool result is a message
the model may or may not lean on; a `task_messages` directive is the node's objective.

Also note the docstring convention. `"""Record customer's age.\n\n    Args:\n        age (int): The
customer's age.\n    """` — that docstring **is** the JSON Schema. The summary line becomes the tool
description; the `Args:` block becomes `properties`. Which means a docstring is also a routing hint
the model reads, and `patient_intake.py` exploits that deliberately: *"Record the user's
prescriptions. **Once confirmed, the next step is to collect allergy information.**"*
(`patient_intake.py:142`). The next step is documented to the model in the tool description of the
current step.

### 11.5 The round trip: `state → prompt → model → args`

Follow one value all the way through, because it is the pattern's one real weakness and you need to
see it before you copy the pattern.

1. `collect_age` writes `flow_manager.state["age"] = age` (`:124`). This is the file's **only**
   `state` write.
2. `collect_marital_status` reads it back to construct the next node:
   `create_quote_calculation_node(flow_manager.state["age"], marital_status)` (`:144`).
3. That factory interpolates it into the prompt:

**`examples/flows/insurance_quote.py:240-255`**
```python
def create_quote_calculation_node(age: int, marital_status: str) -> NodeConfig:
    """Create node for calculating initial quote."""
    return NodeConfig(
        name="quote_calculation",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Calculate a quote for {age} year old {marital_status} customer. "
                    "First, call calculate_quote with their information. "
                    "Then explain the quote details and ask if they'd like to adjust coverage."
                ),
            }
        ],
        functions=[calculate_quote],
    )
```

4. And `calculate_quote(flow_manager, age, marital_status)` receives them **as LLM-supplied
   arguments** — the model reads `age` out of the prompt and puts it back into the tool call.

**`examples/flows/insurance_quote.py:149-158`**
```python
async def calculate_quote(
    flow_manager: FlowManager, age: int, marital_status: str
) -> tuple[QuoteCalculationResult, NodeConfig]:
    """Calculate initial insurance quote.

    Args:
        age (int): The customer's age.
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    logger.debug(f"calculate_quote handler executing with age: {age}, status: {marital_status}")
```

So a value the application already had in a Python dict travels: **`state["age"] → the next node's
prompt → the model → back as a function argument.**

It works. It also passes the value through a language model, and that is a laundering step nobody
asked for. `36` can come back as `36`, or as `35`, or as `"mid-thirties"` if the prompt phrasing
drifts. There is no schema validation between the prompt and the argument beyond the JSON type.

**The rule for Lina, stated plainly:** anything the model must not paraphrase — a 주민등록번호, a
policy code, a 계좌번호, a consent timestamp, a product ID like
`saedam_355_m35_plan_default` — must be **read from `flow_manager.state` inside the handler**, never
declared as a function parameter. Declare only what the customer is actually saying in this turn.

```python
# WRONG for identifiers — the model relays the value
async def save_payment_info(
    flow_manager: FlowManager, resident_number: str, account_number: str
) -> tuple[dict, NodeConfig]: ...

# RIGHT — the model triggers the action, the handler reads the data
async def save_payment_info(
    flow_manager: FlowManager, confirmed: bool
) -> tuple[dict, NodeConfig]:
    """Save the customer's payment information once they have confirmed.

    Args:
        confirmed (bool): Whether the customer confirmed the details read back to them.
    """
    rrn = flow_manager.state["resident_number"]        # never left Python
    account = flow_manager.state["account_number"]
    ...
```

This is also a compliance shape, not only a correctness one. A 주민등록번호 that never enters
`task_messages` and never becomes a tool argument never enters the LLM provider's request body.

### 11.6 The transferable decomposition pattern

Extracted from the file, stated as a recipe:

1. **One node = one thing you are extracting or accomplishing.** `insurance_quote.py` splits age and
   marital status into two nodes even though one node with a two-parameter function would work. The
   split is what makes each `task_messages` a single unambiguous instruction.
2. **`task_messages` is imperative and addressed to the model**, `role: "developer"`, one message.
   *"Ask about the customer's marital status for premium calculation."* Not a description of the
   node; an order.
3. **`functions` is the rail guard.** A node's function list is the model's complete menu. Omit
   `end_conversation`-shaped functions from nodes where hanging up would be wrong.
4. **`role_message` once, on the initial node.** It persists.
5. **Data the *next* prompt needs → f-string it into the factory's parameters.** Data the
   *application* needs → `flow_manager.state`. Data the *model* needs to acknowledge → the return
   value.
6. **Pre-format every number in Python.**
7. **Terminal nodes: `post_actions=[{"type": "end_conversation"}]`, no `functions`.**
8. **Always set `name`.** (§5.2 — this one is not in the example's own discipline, since
   `hello_world.py` names a node after its factory. Do it anyway.)

Mapped onto one Lina stage as a shape check — `informed_consent`, which has four Korean consent
questions (개인정보 수집·이용, 신용정보집중기관 조회, 주민번호 처리, 건강정보 처리):

- Either four nodes, one per question, each with one `record_consent` function edging to the next;
- Or one node with a `record_consent(consent_item: str, agreed: bool)` function returning
  `(result, create_consent_node())` — a self-loop like `update_coverage` — with `flow_manager.state`
  accumulating the four answers and the factory re-rendering *"이미 동의하신 항목: …. 다음
  항목: …"* into `task_messages` each time.

The second is `patient_intake.py`'s shape and the first is `insurance_quote.py`'s. Which is right
depends on whether a customer answering all four at once ("네 네 다 동의합니다") should be one tool
call or four — and that is a product question this chapter cannot answer for you.

---

## 12. Three more examples, three more patterns

### 12.1 `patient_intake.py` — the LLM gets an acknowledgement, the app gets the payload

**`examples/flows/patient_intake.py:142-155`**
```python
async def record_prescriptions(
    flow_manager: FlowManager, prescriptions: list[dict]
) -> tuple[PrescriptionRecordResult, NodeConfig]:
    """Record the user's prescriptions. Once confirmed, the next step is to collect allergy information.

    Args:
        prescriptions (list[dict]): List of prescription objects, each with "medication" (str, the medication's name) and "dosage" (str, the prescription's dosage).
    """
    # Store prescriptions in flow state
    flow_manager.state["prescriptions"] = prescriptions

    # In a real app, this would store in patient records
    return PrescriptionRecordResult(count=len(prescriptions)), create_allergies_node()
```

Two things worth copying.

**A `list[dict]` parameter described entirely in the docstring.** No `FlowsFunctionSchema`, no
explicit JSON Schema — the `Args:` line spells out the object shape in prose and the schema
extraction turns it into `properties`. This is the answer to "how do I collect several structured
items in one turn" without leaving the direct-function path.

**The return value is `count`, not the payload.** `PrescriptionRecordResult(count=len(prescriptions))`
goes to the model; the actual list goes to `flow_manager.state`. The model gets *"3"* and can say
"세 가지 약을 기록했습니다"; it never re-reads the medication names, so it never gets a chance to
paraphrase one. That is §11.5's rule applied to the *return* channel rather than the *argument*
channel, and for a health or finance domain it is the correct default.

`patient_intake.py` also has the two backward edges Flows makes trivially cheap: `revise_information`
returns `create_prescriptions_node()` (a loop back to an earlier node) and `confirm_information`
returns `create_confirmation_node()`. Since a node is just a `NodeConfig` a function returned, going
backwards costs exactly as much as going forwards — which is a real advantage over an edge-whitelist
design where every backward edge is a row you have to remember to add.

### 12.2 `restaurant_reservation.py` — outcome branching, and the outbound/inbound flag

**`examples/flows/restaurant_reservation.py:150-166`**
```python
        party_size (int): Number of people in the party.
    """
    # Check availability with mock API
    is_available, alternative_times = await reservation_system.check_availability(party_size, time)

    # Result: availability status and alternative times, if any
    result = TimeResult(
        status="success", time=time, available=is_available, alternative_times=alternative_times
    )

    # Next node: confirmation or no availability
    if is_available:
        next_node = create_confirmation_node()
    else:
        next_node = create_no_availability_node(alternative_times)

    return result, next_node
```

**The branch is an `if` statement in Python.** Not a condition on an edge, not a guard expression in
a config file — an `if`. This is the single biggest expressive advantage of edges-as-return-values:
the routing logic is ordinary code with access to the API response, the database, the time of day,
and `flow_manager.state`. No DSL can compete with that.

And the failure branch closes over its data:

**`examples/flows/restaurant_reservation.py:227-243`**
```python
    times_list = ", ".join(alternative_times)
    return NodeConfig(
        name="no_availability",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Apologize that the requested time is not available. "
                    f"Suggest these alternative times: {times_list}. "
                    "Ask if they'd like to try one of these times. If they pick a time, check "
                    "its availability. If they'd rather not book after all, call the "
                    "end_conversation function to wrap up."
                ),
            }
        ],
        functions=[check_availability, end_conversation],
    )
```

`", ".join(alternative_times)` before the f-string — §11.3's pre-formatting rule applied to a list.

Then the flag that matters most to you:

**`examples/flows/restaurant_reservation.py:174-188`**
```python
# Node configurations
def create_initial_node(wait_for_user: bool) -> NodeConfig:
    """Create initial node for party size collection."""
    return NodeConfig(
        name="initial",
        role_message="You are a restaurant reservation assistant for La Maison, an upscale French restaurant. Be casual and friendly. This is a voice conversation, so avoid special characters and emojis.",
        task_messages=[
            {
                "role": "developer",
                "content": "Warmly greet the customer and ask how many people are in their party. This is your only job for now; if the customer asks for something else, politely remind them you can't do it.",
            }
        ],
        functions=[collect_party_size],
        respond_immediately=not wait_for_user,
    )
```

`respond_immediately=not wait_for_user`. **One flag switches between an outbound call (the bot
speaks first) and an inbound one (the bot waits).** Everything else about the node is identical.

Lina is an outbound tele-sales agent, so `respond_immediately=True` is right for her introduction
node — she dials, the customer picks up, she speaks. But §13.7 will show that the *default* being
`True` is a hazard for every *other* node in a rule-driven design, because a transition fired on the
customer's turn will make her talk over them.

Note also the rail-guard sentence in prose: *"This is your only job for now; if the customer asks for
something else, politely remind them you can't do it."* With only `collect_party_size` advertised,
the model has no other move — but it can still ramble. The prompt closes the gap the tool list leaves
open. For a Korean tele-sales script where 이탈 방지 matters, write this sentence into every
collection node.

### 12.3 `warm_transfer.py` — the escalation recipe, and the transition that is not a tool call

715 lines, and the closest thing in the repository to a shippable feature for a tele-sales product.
Its five-node graph is documented in a header comment:

**`examples/flows/warm_transfer.py:82-92`**
```python
# Flow nodes:
#
# 1. initial_customer_interaction
#    The initial node, where the bot interacts with the customer and tries to help with their requests.
#    Functions:
#    - check_store_location_and_hours_of_operation (always succeeds)
#    - start_order (always fails)
#    - end_customer_conversation
#    Transitions to either:
#    - continued_customer_interaction
#    - transferring_to_human_agent
```

Three mechanisms to take.

**(a) Escalation is the failure branch of every task function.**

**`examples/flows/warm_transfer.py:249-255`**
```python
# Helpers
def next_node_after_customer_task(result: Mapping[str, Any]) -> NodeConfig:
    """Transition to either the "continued_customer_interaction" node or "transferring_to_human_agent" node, depending on the outcome of the previous customer task"""
    if result.get("status") == "success":
        return create_continued_customer_interaction_node()
    else:
        return create_transferring_to_human_agent_node()
```

A plain helper, called by every task handler, so that every failure routes to escalation without
each handler repeating the branch. For Lina this is `escalate_to_human` reached from anywhere a tool
fails, rather than only from an explicit intent rule.

**(b) A node can be pure side effect, with no functions at all.**

**`examples/flows/warm_transfer.py:321-342`**
```python
def create_transferring_to_human_agent_node() -> NodeConfig:
    """Create the "transferring_to_human_agent" node.
    This is the node where the customer is asked to please hold while the bot transfers them to a human agent. Hold music plays while the customer waits.
    """
    return NodeConfig(
        name="transferring_to_human_agent",
        task_messages=[
            {
                "role": "developer",
                "content": "Start by apologizing to the customer that there was an issue fulfilling their last request, then inform them that they are being transferred to a human agent. Tell them to please hold while you connect them, and thank them for their patience.",
            }
        ],
        pre_actions=[
            ActionConfig(type="function", handler=mute_customer),
        ],
        post_actions=[
            ActionConfig(type="function", handler=start_hold_music),
            ActionConfig(type="function", handler=make_customer_hear_only_hold_music),
            ActionConfig(type="function", handler=print_human_agent_join_url),
        ],
    )
```

Four actions, all `type="function"` — §9.3's guidance followed by the framework's own most
sophisticated example. And the ordering is exactly why `function` actions queue rather than call:
`mute_customer` runs before the node's messages are installed (§4.1), and the three post-actions
run at the pipeline tail *after* the "please hold" line has been spoken (§9.2). Start the hold music
one frame earlier and it plays over the apology.

**(c) The transition that is not a tool call.** A human agent joining the Daily room is not something
the model can call a function about, so the transition is pushed from a transport event handler —
`warm_transfer.py:657`, quoted in §5.3, calling:

**`examples/flows/warm_transfer.py:258-261`**
```python
# Transitions
async def start_human_agent_interaction(flow_manager: FlowManager):
    """Transition to the "human_agent_interaction" node."""
    await flow_manager.set_node_from_config(create_human_agent_interaction_node())
```

**`set_node_from_config` is PUBLIC** (`manager.py:588`) and callable from any coroutine:

**`src/pipecat/flows/manager.py:588-600`**
```python
    async def set_node_from_config(self, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Used to manually transition between nodes in a flow.

        Args:
            node_config: Configuration for the new node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        await self._set_node(get_or_generate_node_name(node_config), node_config)
```

One line of body, straight into `_set_node`. **Flows does not force transitions through the model.**
There are exactly two producers of a `NodeConfig` — an LLM function call whose handler returns one,
and any coroutine calling this method — and the second is proven in-tree twice
(`warm_transfer.py:261` from a transport callback, `multi_worker_handoff.py:352` from
`@worker.event_handler("on_activated")`).

That fact is the hinge of [[ch-12/read]] and it is stated here as a **mechanism only**. What it makes
possible for boson's rule layers — where the processor stands, what the queue race costs, whether
the layers collapse — is not this chapter's business.

The briefing node also shows the intended use of a context reset at a handoff:

**`examples/flows/warm_transfer.py:360-365`**
```python
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY,
            summary_prompt=(
                "Summarize the conversation with the customer, including what they were trying to accomplish and what, if anything, went wrong while trying to fulfill their requests. Include specific error details."
            ),
        ),
```

The human agent gets a summary rather than a transcript. The *intent* is exactly right for Lina's
`escalate_to_human`; the *mechanism* is the deprecated one from §7.2, so build it as
`ContextStrategy.RESET` plus a pre-action pushing `LLMSummarizeContextFrame`.

---

## 13. Mapping onto boson's stage machine

boson already has this graph. It is declarative, it is in one file, and it has nine nodes.

### 13.1 What is already there

From [[boson-stage-machine]]: `agents/test-lina-gateway/stage_config.py` declares
`initial_stage = "introduction"` and nine registered stages — `introduction`, `product_focused`,
`escalate_to_human`, `consultation`, `purchase`, `reschedule`, `dnc_processing`,
`informed_consent`, `end` — each a dict with three fields:

```python
@dataclass
class StageDefinition:
    name: str; prompt: str = ""; tools: list[str] = []; skills: list[str] = []; transitions: list[str] = []
```

And the whitelist, verbatim per [[boson-stage-machine]]:

| stage | allowed successors | stage tools |
|---|---|---|
| `introduction` (initial) | product_focused, purchase, dnc_processing, reschedule, escalate_to_human, end | — |
| `product_focused` | consultation, purchase, reschedule, dnc_processing, escalate_to_human, end | check_product_detail, check_product_summary, lookup_faq |
| `consultation` | purchase, informed_consent, reschedule, dnc_processing, escalate_to_human, end | + check_available_products; skill `product_manager` |
| `informed_consent` | consultation, end, reschedule, dnc_processing, escalate_to_human | record_consent, get_consent_status |
| `purchase` | end, escalate_to_human | agreement_record, agreement_status, check_product_detail, verify_personal_info, save_payment_info, save_address; skill `payment_manager` |
| `reschedule` | consultation, end, escalate_to_human | reschedule |
| `dnc_processing` | end, escalate_to_human | register_dnc, check_dnc_status |
| `escalate_to_human` | end | escalate_to_human |
| `end` | *(none — terminal)* | — |

Now map the three fields.

### 13.2 `tools` → `NodeConfig["functions"]` — direct

`stages[X]["tools"]` is a list of tool names; `NodeConfig["functions"]` is a list of direct functions
or `FlowsFunctionSchema`s. The port is mechanical: the name list becomes an import list.

`_GLOBAL_TOOLS` (currently `[]`) maps onto `FlowManager(global_functions=[...])`, which Flows mixes
in at every node — `manager.py:654`, quoted in §4.2:

```python
            functions_list = self._global_functions + node_config.get("functions", [])
```

Prepended, at every node, unconditionally. Exactly the semantics `_GLOBAL_TOOLS` has.

boson's `@tool` handlers map onto `FlowsDirectFunction`s — the docstring already carries the
description, which is the same convention both systems use. The two contract mismatches
[[boson-tool-router]] flags are real but they are [[ch-09/read]] §9.4's subject, not this
chapter's: Pipecat handlers receive one `FunctionCallParams` and settle via
`await params.result_callback(result)`, while boson's are `handler(**arguments)` returning a value.
Flows sits on top of that, so it inherits the mismatch without adding to it — a Flows direct function
is `async def f(flow_manager, **params) -> tuple[Any, NodeConfig | None]`, which is a *third* shape.

### 13.3 `transitions` → **stops being data**

This is the structural rewrite, and it is worth being precise about what is gained and lost.

`stages[X]["transitions"]` is a per-stage allowlist. `StageMachine.transition()` checks it and
returns `TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")`. Flows has
**no counterpart at any level** — not in `types.py` (there is no edge type and no node type,
§10.2), not in `_validate_node_config` (§10.1), not in `set_node_from_config` (§12.3).

In Flows the legal successors are implicit in which `NodeConfig` each function can return. They are
distributed across handler bodies. There is no single place to read them.

**What you gain.** A whole class of bug becomes structurally impossible. `stage_config.py` documents
it twice in its own comments — the `v0.7.5 (#12)` note where `transition_detector.py:157` emitted
`StageTransition("purchase")` and the stage machine rejected it because the whitelist omitted
`"purchase"`. In Flows a handler that returns `create_purchase_node()` transitions to the purchase
node. Full stop. There is no second list to keep in sync, so there is nothing to fall out of sync.

And look at how that bug fails today, from [[boson-stage-machine]]'s reading of
`core._apply_stage_transition`:

```python
result = self._stage_machine.transition(from_stage=session.active_stage, to_stage=target)
if not result.success:
    return                      # ← silent no-op on a rejected edge
session.active_stage = target
self._inject_stage(session, result.new_stage)
```

`return` on failure. The `error` string is constructed and discarded. A rule fires, the transition is
rejected, and nothing anywhere records that it happened. That failure mode is invisible unless
someone checks `TransitionResult.error`, and the code path that would check it does not.

**What you lose.** The single-file, readable, diffable transition table. Today a reviewer can open
`stage_config.py` and see in one screen that `purchase` can only go to `end` or
`escalate_to_human` — which is a compliance-relevant property of an insurance sales call, not a
developer convenience. After a naive port that property is not stated anywhere; it is an emergent
consequence of which factories the purchase node's handlers happen to call.

**The shape that keeps both.** Keep `StageMachine` as a **pure validator** in front of
`set_node_from_config`, and throw away only its prompt and tool plumbing:

```python
NODES: dict[str, Callable[[], NodeConfig]] = {
    "introduction": create_introduction_node,
    "product_focused": create_product_focused_node,
    # ... nine entries
}

async def go(flow_manager: FlowManager, target: str) -> None:
    result = stage_machine.transition(from_stage=flow_manager.current_node, to_stage=target)
    if not result.success:
        logger.warning(result.error)          # ← the line that is missing today
        return
    await flow_manager.set_node_from_config(NODES[target]())
```

Two things that dict buys you beyond the check. It is the **node registry** §10.2 says Flows does
not have, so `has_stage(name)` and initial-stage validation come back. And `NodeConfig["name"]`
must be set to the boson stage name, or `flow_manager.current_node` — which is now the left-hand
side of the legality check — holds a UUID and the check compares garbage (§5.2).

That is as far as this chapter goes. Where the call to `go()` comes from is [[ch-12/read]].

### 13.4 `skills` → nothing

`stages[X]["skills"]` (`product_manager` on `consultation`, `payment_manager` on `purchase`) has
**no Flows concept at all.** There is no `skills` key in `NodeConfig`, nothing in `types.py`, no
second-tier tool notion anywhere in the package.

Two options, neither of which Flows helps with: flatten each skill into the functions it exposes, or
keep skills as boson meta-tools behind a `use_skill` direct function ([[boson-tool-router]]). The
second preserves the two-tier structure but means the node's `functions` list is
`[use_tool, use_skill]` at every node — which interacts badly with §13.5.

Also with no counterpart: boson's `<system-reminder>` protocol,
`ContextManager.pop_pending_reminders()`, and the per-turn
`<system-reminder>Active stage: {session.active_stage}</system-reminder>` re-assertion from
`session/history.py`. Flows does not re-state the current node to the model on every turn; the node's
`task_messages` were appended once at transition and then recede into history as the conversation
grows. For a long `consultation` stage that is a meaningful behavioural difference, and `APPEND`
makes it worse rather than better: the stage prompt gets further away every turn.

### 13.5 The collision: unconditional `LLMSetToolsFrame` versus a byte-stable tool array

This is the sharpest conflict between the two designs, and it does not appear until you put two
facts side by side.

**Fact one, from §4.4 Rule 3.** `LLMSetToolsFrame(tools=functions)` is emitted at `manager.py:839`
on **every** node transition, unconditionally, with the full advertised set.

**Fact two, from [[boson-tool-router]].** boson deliberately keeps the advertised tool array
**byte-stable across stages** to preserve prompt caching. The model is shown only `use_tool` and
`use_skill`; per-stage access control happens underneath, at dispatch time, via a ContextVar
allowlist:

> *"Do not collapse 'the model can see it' into 'the model may call it' — boson deliberately keeps
> the advertised array byte-stable across stages to preserve prompt caching while the allowlist
> changes underneath."*

Three gates, kept distinct: **exposure** (what the model sees), **availability** (what the stage
allows), **permission** (what may run at all).

Flows has **one** gate, and it is exposure. `NodeConfig["functions"]` is the advertised array *and*
the availability list *and* — since there is no permission hook — the permission model. §5.4 showed
there is no field to hang a separate check on.

So a naive port has to pick:

**Option A — port the node's `functions` list literally.** Each node advertises its own stage tools.
The tool array now changes on every transition, which changes the prompt prefix, which invalidates
the provider-side prompt cache at every stage boundary. For a call that walks four or five stages
that is four or five cache misses on a prefix that includes the whole system prompt. [[ch-11/read]]
will let you price it; the point here is that it is a *cost boson currently does not pay*, introduced
by a mechanism that has no off switch.

**Option B — keep the two-entry meta-tool array.** `functions=[use_tool, use_skill]` at every node,
identical everywhere, and keep the ContextVar allowlist as the availability gate. The array is
byte-stable, so the cache survives — but now `NodeConfig["functions"]` is the same on all nine nodes
and carries no information, and the per-node rail guard from §11.3 is gone. The thing that made
`insurance_quote.py`'s design safe — the model's menu *is* the node's legal moves — does not exist,
because the menu is always the same two entries.

**Option C — keep both gates.** Advertise per-node functions (Option A's array) *and* keep the
dispatch-time allowlist as a second check. You pay the cache cost and keep the defence in depth.

There is no Option D where you get boson's caching behaviour and Flows' per-node rail guard, because
they are the same knob turned in opposite directions. **Pick one, deliberately, and write down which
one.** [[ch-13/read]] scores it; this chapter's job is to make sure the choice is visible rather than
accidental.

### 13.6 `ContextStrategy.RESET` is the one clear gain

§7.3 stated it; here it is with the boson side attached.

boson's `_inject_stage` appends `f"[Stage: {stage.name}]\n\n{stage.prompt}"` as a
`<system-reminder>`-wrapped message ([[boson-stage-machine]], `stage/context.py:73`). Only appends.
There is no reset path in the stage layer.

Consequence: a Lina call that reaches `purchase` has, in its context, the introduction prompt, the
product_focused prompt, the consultation prompt, the informed_consent prompt, and the purchase
prompt — plus every turn of conversation in between — on every single inference from that point on.
The prompt grows monotonically for the length of the call, and the *earliest* instructions are the
ones the model has been reading longest.

`ContextStrategyConfig(strategy=ContextStrategy.RESET)` on the `purchase` node replaces the message
list with that node's `task_messages`. The persona survives (it is a system instruction, §6.1). The
tool set survives (separate frame, separate field). `flow_manager.state` survives (never in the
context). What goes is the conversation.

Which is a real trade, not a free win: `purchase` needs to know what product was discussed in
`consultation`. That is what `flow_manager.state` and the f-string channel are for — §11.4's table
is the design pattern that makes `RESET` usable, and `patient_intake.py:244` is the working example.

Do **not** build on `RESET_WITH_SUMMARY` (§7.2 — deprecated, 5-second blocking cap, silent fallback,
config mutation). Use `RESET` plus a pre-action pushing `LLMSummarizeContextFrame` when you need the
summary.

### 13.7 `respond_immediately=False` is mandatory for rule-driven transitions

The default is `True` (`manager.py:707`). Every `_set_node` call therefore queues an `LLMRunFrame`
and the bot speaks.

boson's stage transition is **silent bookkeeping performed before the agent loop runs**
([[boson-stage-machine]]): a rule emits `StageTransition(target)`, `_apply_stage_transition` moves
`session.active_stage` and injects the prompt, and *then* the loop runs once and produces one
response. The transition does not produce an utterance; it changes the conditions under which the
next utterance is produced.

Port that naively and every rule-driven transition — including the ones that fire mid-turn while the
customer is still talking — queues an `LLMRunFrame` and Lina interrupts the customer to respond to a
stage change they did not know happened.

So: **every node in a rule-driven port carries `respond_immediately=False`**, except the
introduction node on an outbound call, which is §12.2's `wait_for_user=False` case.

And note what that flag also changes, from §4.4: post-actions stop running inline and become
deferred to the next `BotStoppedSpeakingFrame` with `_ongoing_actions_count == 0` (§2.2, §9.6). A
node that both sets `respond_immediately=False` and carries post-actions has its side effects fire at
a time determined by the *bot's* speech, which may be several turns later than you expect. Check
every ported node for that combination.

### 13.8 What does not port, and what this chapter refuses to design

**Does not port:** `stages[X]["skills"]` (§13.4). The `<system-reminder>` protocol. The per-turn
active-stage re-assertion. `Continue()`, `Pass()` and `Filter(reason)` from
`schemas/actions.py`'s eight-verb vocabulary — those are pre-LLM routing verdicts, and by the time a
Flows action runs the message is already in the context (§9.4).

**Ports cleanly:** `Respond(text)` → `{"type": "tts_say", ...}`. `PreTool(...)` →
`pre_actions=[{"type": "function", "handler": fn}]`, with the caveat from §9.6 that a preceding
`tts_say` does **not** serialize against a following `function`. Terminal stages (`end`,
`dnc_processing`) → `post_actions=[{"type": "end_conversation"}]` and no functions.
`escalate_to_human` → the whole `warm_transfer.py` recipe (§12.3).

**And here is where this chapter stops.** You now know that `set_node_from_config` is public, that it
accepts a `NodeConfig` from any coroutine, that Flows imposes no legality check, and that a
`StageMachine` can be kept in front of it as a pure validator. That is a mechanism and a constraint
set.

Turning it into a design — where the rule processor stands in the pipeline, whether a complete user
turn exists at that position before inference begins, whether boson's cross-layer veto survives being
spread across adjacent processors, and what the queue race between an in-pipeline processor and
`FlowManager`'s head-injected frames costs — is [[ch-12/read]]'s work, and it needs
[[ch-11/read]]'s millisecond budget before it can price any of it. Do not sketch it here.

---

## 14. What to hold in your head

| # | Fact | Where |
|---|---|---|
| 1 | `FlowManager` is a plain class, not a `FrameProcessor`, not in the `Pipeline` list | `manager.py:80`; `hello_world.py:135-145` |
| 2 | It is constructed *after* the worker and requires one; it drives the pipe from outside | `manager.py:121-134`; `hello_world.py:147-167` |
| 3 | Two touch points: `queue_frames` out (`:709`, `:841`), one filtered downstream event in | `manager.py:709, 841`; `actions.py:109-127` |
| 4 | `set_reached_downstream_filter` **replaces**; Flows is its only caller; wipe it and actions die silently | `worker.py:695-701`; `actions.py:109` |
| 5 | Flow frames enter at the **head** and traverse everything; `LLMRunFrame` is consumed at the user aggregator | `worker.py:793-808`; `llm_response_universal.py:814, 1173` |
| 6 | The batch: `LLMUpdateSettingsFrame`? → `Append`/`Update` → `SetTools` (always) → then separately `LLMRun`? | `manager.py:762-841, 709` |
| 7 | A node with no `functions` **clears** the tool set (`NOT_GIVEN`) | `manager.py:670-672, 839` |
| 8 | The batch is not atomic — two `queue_frames` calls, a plain loop, no ordering guarantee | `worker.py:810-829` |
| 9 | Only `LLMUpdateSettingsFrame` is `UninterruptibleFrame`; a barge-in can drop the rest | `frames.py:2251` vs `634, 645, 661, 694` |
| 10 | `_current_node` is a `str`, never a `NodeConfig`; unnamed nodes get a fresh UUID | `manager.py:149, 703`; `types.py:518` |
| 11 | `_current_functions`: zero reads in `src/`, six in `tests/test_flows_manager.py` | grep, §0.4 |
| 12 | `FlowConfig` does not exist; no static/dynamic split; structure is determined at runtime | grep = 0; `__init__.py:12-13` |
| 13 | `NodeConfig` is a `TypedDict(total=False)`; `task_messages` is the only `Required` key; typos are silent | `types.py:224-237` |
| 14 | `ContextStrategyConfig` is a `@dataclass` and validates in `__post_init__` — the asymmetry with `NodeConfig` | `types.py:155-179` |
| 15 | An edge is `ConsolidatedFunctionResult`'s second element: node / `None` / `NO_RESPONSE` (by identity) | `types.py:266, 247`; `manager.py:496` |
| 16 | `ContextStrategy` maps onto `Update` vs `Append` and is the only knob that forgets | `manager.py:831-836` |
| 17 | `RESET_WITH_SUMMARY`: deprecated, 5.0 s blocking cap, silent fallback, mutates the config | `manager.py:799-822` |
| 18 | `FlowResult` is deprecated with **"No replacement."**; six exported names go in 2.0.0 | `types.py:40-53`; §6.7 |
| 19 | Handler dispatch is `len(sig.parameters)`; the built-ins are bound methods and trip the legacy branch | `manager.py:409-441`; `actions.py:180-188` |
| 20 | Exactly **three** built-in actions: `tts_say`, `end_conversation`, `function` | `actions.py:104-106` |
| 21 | The maintainers de-emphasize custom actions in favour of `function` | `actions.py:292-294` |
| 22 | Actions cannot pick a node, veto a transition, or return a value; a raising **pre**-action aborts | `actions.py:220-224`; `manager.py:647, 693` |
| 23 | `_validate_node_config` checks two things; **no** from→to check exists anywhere | `manager.py:867-898` |
| 24 | There is no node registry — nodes are constructed, never registered; the graph is not enumerable | §10.2 |
| 25 | `set_node_from_config` is public: transitions do **not** have to come from the model | `manager.py:588`; `warm_transfer.py:261` |
| 26 | Data forward: return value (model sees), `state` (survives `RESET`), f-string in `task_messages` (steers) | `insurance_quote.py:124, 144, 248` |
| 27 | The `state → prompt → model → args` round trip is real; read identifiers from `state` in the handler | `insurance_quote.py:144-158`; §11.5 |
| 28 | Pre-format every number in Python; the model must never render a premium | `insurance_quote.py:269-271` |
| 29 | `respond_immediately=not wait_for_user` is the outbound/inbound switch | `restaurant_reservation.py:187` |
| 30 | boson mapping: `tools`→`functions` direct; `transitions`→**stops being data**; `skills`→nothing | §13.2–13.4 |

---

## 다음 챕터로

This chapter hands forward four things.

**A resolved architecture.** `FlowManager` is a plain object beside the pipeline. It writes through
`queue_frames` at two lines and reads through one filtered event. Its node state is a string. Its
graph is a set of Python functions and its edges are tuple elements. That is not a characterisation;
it is `manager.py:80`, `hello_world.py:135-167`, `manager.py:149`, and `types.py:266`.

**A frame batch to memorize.** Four frames, four rules: `LLMUpdateSettingsFrame` conditional and
persistent, `Append`-versus-`Update` **is** the `ContextStrategy`, `LLMSetToolsFrame`
**unconditional** so an empty node clears the tools, `LLMRunFrame` conditional and queued
separately. Every operational surprise in Flows is one of those four firing.

**Three absences, named precisely.** No transition legality. No node registry. No permission gate
distinct from exposure — and `_current_functions`, the field that looks like it might be one, has
zero reads in `src/` and six in the test suite. Whatever guard layer boson keeps, it keeps because
Flows does not have one.

**Two open constraints for the chapters that follow.** The batch enters at the **head** of the
pipeline and must traverse every upstream processor before it reaches the LLM — including any
processor you insert for rule evaluation. And only one of its four frames survives a barge-in, so a
transition fired during the customer's turn can be partially lost with no error and no way for
`FlowManager` to notice.

Next is [[ch-11/read]] — **the latency budget and the observer plane.** It comes before the rule
design on purpose. Two numbers from this chapter are already waiting for a denominator:
`RESET_WITH_SUMMARY`'s hard 5.0-second cap, and the prompt-cache cost of §13.5's unconditional tool
re-advertisement. Neither can be argued as cheap or expensive until there is a budget to measure them
against, and the observer plane is the instrument that makes the budget knowable at all — a
read-only second plane over the frame graph, which is exactly the non-adjacency property a component
outside the pipeline needs.

Then [[ch-12/read]] takes §12.3's public `set_node_from_config` and turns it into a design. Bring
§4.5's non-atomicity, §4.6's interruptibility split, and §3.4's head-injection trace with you —
those three are the constraints the rule seam has to be built against, and [[ch-12/read]] will hand
them back to you as a derivation before it shows you any answer.
