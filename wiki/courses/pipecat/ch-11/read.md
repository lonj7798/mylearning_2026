---
title: "The Latency Budget and the Observer Plane"
chapter: ch-11
phase: read
course: pipecat
sources:
  - latency-budget-voice
  - rtvi-observability
  - endpointing-turn-boundary
  - transport-telephony
figure: figures/latency-waterfall.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# The Latency Budget and the Observer Plane

## 왜 이 챕터인가

This chapter is one position earlier than it looks like it should be, and the reason is a
denominator.

[[ch-12/read]] has exactly one load-bearing decision: whether boson's rule layers run *blocking*
between the aggregated transcript and the LLM call, or *concurrently* with the LLM's first tokens.
That decision is denominated in milliseconds. It is not a structural question — structurally both
are one `FrameProcessor` in one position — it is an arithmetic question of the form *"is N ms an
acceptable fraction of the turn?"* You cannot answer that before you know what the turn costs, what
the parts of the turn cost, and which of those parts you are allowed to spend against. So the budget
comes first.

And the budget cannot come first on its own, because a budget you cannot measure is not a budget, it
is a guess with a table around it. That is why this chapter has two subjects and treats them as one:

1. **The voice-to-voice latency budget** — every millisecond from the last voiced user sample to the
   first audible assistant sample, as a sum of named, separately-timed stages.
2. **The observer plane** — the second, read-only plane over the frame graph that is the reason any
   of those stages has a number at all.

You have already met an accounting problem shaped exactly like this. Your GPU training-memory course
was one ledger: parameters + gradients + optimizer state + activations, each term named, each term
attributable, and the discipline being that nothing is allowed to be "overhead." This is the same
exercise on the time axis. The two differences are that (a) some of these terms overlap and some do
not, and getting that wrong is the single most common way voice-latency arithmetic goes bad, and (b)
one of the terms is **deliberately left empty at the end of this chapter**, because it is yours to
price and I am not going to price it for you.

### What this chapter does not contain

Deployment, process topology, worker-per-call versus worker-per-process, autoscaling, cold-start
economics — none of it is here. Latency accounting and deployment are different subjects that got
merged in an earlier draft of the outline. Process topology belongs to [[ch-04/read]] §13, which you
already have; deployment belongs to [[ch-13/read]]. Nothing below depends on either.

Also not here: **any verdict about realtime_voice versus Pipecat.** §11 states what each of the
three systems measures. It contains no "better," no "worse," no "should adopt." [[ch-13/read]] is
the only place anything gets scored, and [[ch-08/read]] §9's standing rule applies here too.

### What you are expected to already own

This chapter *consumes* four earlier chapters and re-teaches none of them:

- **[[ch-04/read]] §4** — the per-processor two-queue / two-task runtime and the `SystemFrame`
  priority split. You need it in §6 to understand why an observer hook does not sit on the data
  path, and in §7 to understand why a `MetricsFrame` cannot be lost to a barge-in in flight.
- **[[ch-06/read]] §9, §11, §15** — the STT TTFB redefinition
  (`speech_end_time = frame.timestamp - frame.stop_secs`), the 23-constant `stt_latency.py` table,
  and the `effective_stt_wait = max(0.0, stt_timeout - stop_secs)` safety net. **The table was built
  once, there, where provider selection actually happens.** §3.1 below spends it as a single term.
  It does not re-render it, and neither does this chapter's figure.
- **[[ch-07/read]] §1, §4** — `TTFA = TTFB + leading_silence`, `detect_speech_onset`, and sentence
  aggregation as a named cost. §3.4 and §3.5 consume both as budget lines.
- **[[ch-08/read]] §6** — `audio_out_10ms_chunks = 4` → 40 ms per written chunk, and the two-queue
  drain at the output transport. §3.6 spends both numbers.

### How to read the evidence

Every Pipecat line number below was opened at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`
(2026-08-25; `CHANGELOG` head `[1.7.0] - 2026-08-01`) in
`wiki/raw-data/pipecat/pipecat-src/`. Every claim about `boson-agent` and about `realtime_voice`
comes from the excerpt library — [[latency-budget-voice]], [[rtvi-observability]],
[[boson-gateway-server]], [[boson-layers-rules]], [[rtv-vs-pipecat-gap]], [[rtv-pipeline-session]],
[[rtv-webrtc-transport]], [[rtv-vad-chunking]] — which was read from your private repos on
2026-08-25. Those repos are not opened here.

---

## 0. Four corrections before we start

All four were in the material this chapter was assembled from. All four are wrong, or wrong in
degree, against the tree at this commit. Where an excerpt disagrees with the source, **the source
wins**, and I say so in the text rather than quietly picking one.

### 0.1 Metrics are OFF by default

[[rtvi-observability]]'s guideline reads *"Turn on `PipelineParams(enable_metrics=True,
enable_usage_metrics=True)` from day one."* That is good advice, and it is phrased as if the default
were a matter of taste. It is not. Read the defaults:

**`src/pipecat/pipeline/worker.py:184-195`**
```python
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_heartbeats: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    heartbeats_period_secs: float = HEARTBEAT_SECS
    heartbeats_monitor_secs: float = HEARTBEAT_MONITOR_SECS
    report_only_initial_ttfb: bool = False
    send_initial_empty_metrics: bool = True
    start_metadata: dict[str, Any] = Field(default_factory=dict)
```

`enable_metrics: bool = False`. A default Pipecat bot measures **nothing**. Not TTFB, not TTFA, not
TTFAT, not processing time, not text aggregation. Every metric in this chapter is behind that one
boolean, because every emit site is gated on it (§6.4). If you take exactly one operational
instruction from this chapter, it is that line and its sibling `enable_usage_metrics`.

### 0.2 `UserBotLatencyObserver` is not auto-wired by default either

[[rtvi-observability]] says `PipelineWorker` auto-wires the latency observer when
`enable_tracing=True`. True — but `enable_tracing` is not the default, and it is not the only
condition:

**`src/pipecat/pipeline/worker.py:287-289`**
```python
        enable_tracing: bool = False,
        enable_turn_tracking: bool = True,
        enable_rtvi: bool = True,
```

**`src/pipecat/pipeline/worker.py:412-413`**
```python
        self._enable_tracing = enable_tracing and is_tracing_available()
        self._enable_turn_tracking = enable_turn_tracking
```

Two gates, ANDed. `enable_tracing` defaults `False`, *and* it is silently downgraded to `False` when
OpenTelemetry is not importable. Then a third gate:

**`src/pipecat/pipeline/worker.py:426-443`**
```python
        if self._enable_turn_tracking:
            self._turn_tracking_observer = TurnTrackingObserver()
            observers.append(self._turn_tracking_observer)
        if self._enable_tracing and self._turn_tracking_observer:
            # Create pipeline-scoped tracing context
            self._tracing_context = TracingContext()
            # Create latency observer for tracing
            self._user_bot_latency_observer = UserBotLatencyObserver()
            observers.append(self._user_bot_latency_observer)
            # Create turn trace observer with latency tracking
            self._turn_trace_observer = TurnTraceObserver(
                self._turn_tracking_observer,
                latency_tracker=self._user_bot_latency_observer,
                conversation_id=self._conversation_id,
                additional_span_attributes=self._additional_span_attributes,
                tracing_context=self._tracing_context,
            )
            observers.append(self._turn_trace_observer)
```

So the full condition for the framework to hand you a `LatencyBreakdown` unasked is:
`enable_tracing=True` **and** OpenTelemetry installed **and** `enable_turn_tracking` left on
**and** `PipelineParams(enable_metrics=True)`. Four things. Miss any one and
`on_latency_breakdown` never fires.

The practical consequence, and it is not a small one: **you should construct
`UserBotLatencyObserver()` yourself and pass it in `observers=[...]`** rather than relying on the
tracing path, unless you actually want OTel spans. It is a plain `BaseObserver`; nothing about it
requires tracing. Tracing requires *it*, not the other way round.

### 0.3 The observer plane is read-only, but it is not synchronous

The natural reading of "`BaseObserver` sees every frame transfer" is that each processor calls each
observer inline, which would mean a slow observer directly extends the frame path. That is not what
happens. Processors call **one** observer — a proxy — and the proxy only enqueues:

**`src/pipecat/pipeline/worker.py:551-554`**
```python
        # The worker observer acts as a proxy to the provided observers. This way,
        # we only need to pass a single observer (using the StartFrame) which
        # then just acts as a proxy.
        self._observer = WorkerObserver(observers=observers)
```

**`src/pipecat/pipeline/worker_observer.py:188-192`**
```python
    async def _send_to_proxy(self, data: Any):
        if not self._proxies:
            return
        for proxy in self._proxies.values():
            await proxy.queue.put(data)
```

One `asyncio.Queue` and one task **per registered observer**
(`_create_proxy`, `worker_observer.py:173-178`). The class docstring says it outright:

**`src/pipecat/pipeline/worker_observer.py:57-63`**
```python
    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a worker for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each worker.
```

This matters twice. Once as reassurance: your custom observer can do slow work without paying for it
on the customer's critical path. And once as a warning, in §10.3: those queues are unbounded, and
the observer computes its own timestamps with `time.time()` *after* dequeuing, not from the
pipeline-clock timestamp carried on the event. Both facts have measurement consequences.

### 0.4 `stt_latency.py` is 68 lines and 23 measured constants, and one number in the excerpt list is
easy to miscount

[[latency-budget-voice]] lists the table and it is correct, but two entries are easy to drop.
`DEEPGRAM_SAGEMAKER_TTFS_P99` is a *separate* constant that happens to share Deepgram's 0.35, and
`NVIDIA` / `WHISPER` are not measured values at all — they are aliases of the fallback:

**`src/pipecat/services/stt_latency.py:66-68`**
```python
# These services run locally and should be replaced with measured values
NVIDIA_TTFS_P99: float = DEFAULT_TTFS_P99
WHISPER_TTFS_P99: float = DEFAULT_TTFS_P99
```

So the file's 68 lines hold **23 measured provider constants**, one `DEFAULT_TTFS_P99 = 1.0`, and
two aliases of the default. [[ch-06/read]] §11.1 prints all 23. This chapter uses exactly two of
them — the extremes — and never re-renders the table.

---

## 1. What a latency budget actually is

Before any code: the concept, stated so that §3's formula is a summary of something you already
believe rather than a thing to memorise.

### 1.1 One number, and why one number is useless

The number the customer experiences is one interval:

```
[ last voiced sample of the customer's speech ]  ──────►  [ first AUDIBLE sample of Lina's reply ]
```

Call it the **voice-to-voice interval**. It is the only number a product owner cares about and the
only number that appears in your own `CLAUDE.md` target (§2.1). It is also, on its own, useless for
engineering, for the same reason "the model uses 47 GB" is useless: it tells you the total and
nothing about which lever moves it.

The useful object is the decomposition. Pipecat's decomposition is not a documentation artefact —
it is *materialised in the type system*, as eight Pydantic classes in one 237-line file:

**`src/pipecat/metrics/metrics.py:19-38`**
```python
class MetricsData(BaseModel):
    """Base class for all metrics data.

    Parameters:
        processor: Name of the processor generating the metrics.
        model: Optional model name associated with the metrics.
    """

    processor: str
    model: str | None = None


class TTFBMetricsData(MetricsData):
    """Time To First Byte (TTFB) metrics data.

    Parameters:
        value: TTFB measurement in seconds.
    """

    value: float
```

Every subclass carries `processor` and `model`. That is the whole attribution mechanism: a metric is
never a bare float, it is a float *plus who produced it plus which model produced it*. Which is
exactly what a ledger needs, and exactly what a `logger.info(f"took {t}ms")` does not give you.

The eight timing/usage classes, all in `metrics.py`:

| Class | Fields | What it prices |
|---|---|---|
| `TTFBMetricsData` (:31) | `value` | request → first output *of any kind* |
| `TTFAMetricsData` (:41) | `ttfa, ttfb, leading_silence` | request → first **audible** sample |
| `TTFATMetricsData` (:64) | `ttfat, ttfb, thinking_time` | request → first **answer token** |
| `ProcessingMetricsData` (:99) | `value` | an explicit `start`/`stop` window you opened |
| `LLMUsageMetricsData` (:143) | `value: LLMTokenUsage` | tokens (cost, not latency) |
| `STTUsageMetricsData` (:173) | `value: STTUsage(audio_seconds)` | audio seconds submitted |
| `TTSUsageMetricsData` (:183) | `value: int` | characters synthesised |
| `TextAggregationMetricsData` (:193) | `value` | first LLM token → first complete sentence |
| `TurnMetricsData` (:206) | `is_complete, probability, e2e_processing_time_ms` | the turn model's verdict + its own cost |

(`SmartTurnMetricsData` at `:225` is `@deprecated` since 0.0.104, removal 2.0.0 — do not build on
it.)

Read that table as a claim about what is worth separating. Three different "time to first
something" classes exist because three different first-somethings matter and they are routinely
confused. §3.3 and §3.5 are about exactly that confusion.

### 1.2 The two failure modes of voice-latency arithmetic

Both are worth naming now, because §4 is about avoiding them.

**Failure mode 1 — summing terms that overlap.** TTS synthesis wall time is not on the budget.
[[ch-07/read]] §1.1 made this concrete: audio plays at 1× wall clock, so once the first audible
sample is out, every remaining millisecond of synthesis happens *underneath* playback that is
already running. Add "TTS total synthesis time" to your budget and you will produce a number that is
2–3× the interval the customer actually experiences, then optimise the wrong thing.

**Failure mode 2 — mistaking a tail term for a typical term.** [[ch-06/read]] §15.3 already told you
the STT safety net is *"a ceiling you almost never pay."* `effective_stt_wait` is armed at every VAD
stop and cancelled the moment a finalized transcript arrives. Choosing xAI at 2.14 s over Deepgram
at 0.35 s does not add 1.79 s to your median turn; it adds it to your **worst** turns. If your
target is a P50 *and* a P95 — and yours is — those are two different budgets built from the same
terms with different terms dominating. §5 does this arithmetic explicitly.

The figure exists to make both failure modes visible at once:

> **[The voice-to-voice latency waterfall](figures/latency-waterfall.html)** — drag each stage knob
> and watch the total move against your own P50 = 1.0 s and P95 = 1.5 s lines. Do one thing with it
> before reading further: set the STT provider to Deepgram, note the total, switch to xAI, and
> observe that the bar moves by 1.79 s while the *serial* stages do not change at all. That is
> failure mode 2 rendered as a picture.

---

## 2. The only target numbers that exist, and where they come from

### 2.1 Yours

Your own `CLAUDE.md` states the target, and it is the only stated perceptual target anywhere in this
course's evidence base. From [[rtv-vs-pipecat-gap]]'s citation, quoting boson-agent's `CLAUDE.md`
mission:

> "P50 at or below 1.0 seconds and P95 at or below 1.5 seconds", measured "from the last voiced user
> sample to the first audible assistant sample, including end-of-turn/VAD time"

Three things in that sentence do real work and you should not let them blur:

1. **"last voiced user sample"** — not "when VAD decided." Those differ by `stop_secs`. Pipecat
   agrees with you here and does the subtraction in two independent places (§3.1, §7.2).
2. **"first audible assistant sample"** — not "first byte of audio." Those differ by
   `leading_silence`. Pipecat has a metric for exactly that difference (`TTFAMetricsData`) and, as
   §7.4 shows, its headline latency observer *does not use it*.
3. **"including end-of-turn/VAD time"** — the endpointing wait is inside the budget, not before it.
   This is the clause that makes the budget honest, and it is the clause most vendor latency
   marketing quietly drops.

The same `CLAUDE.md` also states an engineering rule — *"Instrument before optimizing… Report
P50/P95/P99"* ([[rtv-vs-pipecat-gap]]) — which is the sentence this entire chapter is a mechanism
for.

### 2.2 Pipecat's: there are none, and I want you to check that yourself

The repo states **no perceptual latency target anywhere.** The README says one thing, and it is a
marketing adjective:

**`README.md:29`**
```markdown
- **Real-Time**: Ultra-low latency interaction with different transports (e.g. WebSockets or WebRTC)
```

`AGENTS.md` states none. `CLAUDE.md` is one line, `@AGENTS.md`. `docs/` is a Sphinx config. The
single latency-shaped constant in the whole eval harness is a **test timeout**, and its comment says
so:

**`src/pipecat/evals/harness.py:110-113`**
```python
# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000
```

Sixty seconds, explicitly chosen so that evals do *not* fail on latency. And the shipped release-eval
suite backs that up: of the **38** scenario files in `scripts/release-evals/scenarios/`, exactly
**2** set `within_ms` at all —
`filter_incomplete_turns_user_idle.yaml:34` (`within_ms: 45000`) and
`multi_worker_handoff_back_and_forth.yaml:53` (`within_ms: 40000`). Both are 40+ second timeouts
guarding a state machine, not latency assertions.

You can verify the whole claim in two commands:

```bash
# run in wiki/raw-data/pipecat/pipecat-src at commit 0cbf9c5b
grep -rn -i "latency" README.md AGENTS.md
ls scripts/release-evals/scenarios/*.yaml | wc -l && grep -rln "within_ms" scripts/release-evals/scenarios/ | wc -l
```

**External knowledge, flagged as external.** The two figures everyone quotes in voice UX — roughly
~200 ms as the natural human turn-taking gap, and roughly ~800 ms voice-to-voice as the "feels live"
ceiling — come from the wider voice-UX literature and vendor marketing. They are not in this
repository and must not be cited as Pipecat claims. If you put them on a slide, cite them as
industry convention, or drop them and cite your own `CLAUDE.md`, which is a real internal
commitment and a better argument anyway.

### 2.3 So what is the line every bar is drawn against?

Yours. P50 ≤ 1.0 s, P95 ≤ 1.5 s, last voiced user sample → first audible assistant sample. That is
the only number in this chapter with an owner, and the figure draws both lines because a single
line would hide failure mode 2.

---

## 3. The formula, term by term

Stated plainly first, as [[latency-budget-voice]] gives it:

```
voice_to_voice
    =  stop_secs
     + max(0, TTFS_p99 − stop_secs)      # STT finalization safety net
     + rule evaluation                   # ← TBD ms. §3.2. This chapter does not fill it in.
     + LLM TTFB (+ thinking time)
     + text aggregation
     + TTS TTFA
     + transport
```

That is the shape. Now each term, with the source that produces it, and — where the source
disagrees with the one-line formula — the correction.

### 3.1 Term 1+2: the endpointing wait

This is the term [[ch-06/read]] built, and it is one of the two biggest in the budget. It has two
parts, and both are already yours:

- `VADParams.stop_secs = 0.2` — the silence the VAD must observe before it will declare the turn
  over. [[ch-06/read]] §2 derived that this is **6 chunks** at 16 kHz / 512 samples
  (`round(0.2 / 0.032)` = `round(6.25)` = 6), i.e. 192 ms of actual audio, not 200.
- `max(0, TTFS_p99 − stop_secs)` — the safety net, sized from the STT service's own published P99
  time-to-final-segment. [[ch-06/read]] §15.2 worked this for four providers.

The subtraction exists so you do not pay for the same silence twice: the provider's P99 is measured
from the customer's *physical* speech end, but the strategy only hears about it `stop_secs` later.

**Correction, and it is a real one.** The plain formula matches the **default** stop strategy and
only that one. `UserTurnStrategies` defaults to
`stop = [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]`
([[endpointing-turn-boundary]], `turns/user_turn_strategies.py` L43), and that strategy anchors the
wait to an **absolute deadline**:

**`src/pipecat/turns/user_stop/turn_analyzer_user_turn_stop_strategy.py:231-245`**
```python
        # The STT p99 budget is measured from when the user actually stopped
        # speaking, which VAD only reports stop_secs later. Anchoring the
        # safety net to that absolute deadline keeps the wait fixed no matter
        # how long end-of-turn analysis (e.g. ML inference) takes before the
        # timer is armed.
        stt_deadline = frame.timestamp - frame.stop_secs + self._stt_timeout

        state, prediction = await self._turn_analyzer.analyze_end_of_turn()
        await self._handle_prediction_result(prediction)

        # The user stopped speaking and the turn is complete, we now need to
        # wait for transcriptions.
        self._turn_complete = state == EndOfTurnState.COMPLETE

        timeout = max(0, stt_deadline - time.time())
```

Read `timeout = max(0, stt_deadline - time.time())` carefully, because it is a piece of budget
design, not bookkeeping. `analyze_end_of_turn()` runs the smart-turn ONNX inference **between** the
deadline being computed and the timer being armed. Because the deadline is absolute, the inference
time is *absorbed into* the STT wait rather than added after it. Smart-turn inference is free as long
as it finishes inside the STT's P99 window. That is a genuinely elegant piece of accounting and it is
the reason the default strategy's endpointing term really is `stop_secs + max(0, TTFS_p99 −
stop_secs)` and not `stop_secs + inference + max(...)`.

The *other* timing strategy does not match the plain formula. `SpeechTimeoutUserTurnStopStrategy`
runs two timers that must **both** complete:

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py:216-230`**
```python
        # user_speech_timeout is the policy floor and always runs. A prior
        # fallback-mode run of the same timer is superseded here.
        await self._restart_user_speech_timer()

        # stt_timeout is a safety net. Short-circuit it if the transcript is
        # already finalized, or if the VAD stop_secs already covered it.
        self._stt_wait_done = False
        effective_stt_wait = max(0.0, self._stt_timeout - self._stop_secs)
        if self._transcript_finalized or effective_stt_wait <= 0:
            self._stt_wait_done = True
        else:
            self._stt_timeout_task = self.task_manager.create_task(
                self._stt_timeout_handler(effective_stt_wait),
                f"{self}::_stt_timeout_handler",
            )
```

Both timers are armed at the same instant and run concurrently, so on that path the endpointing term
is:

```
stop_secs + max(user_speech_timeout, max(0, TTFS_p99 − stop_secs))
          = 0.2 + max(0.6, effective_stt_wait)     with the defaults
```

With Deepgram (`effective_stt_wait = 0.15`) the **0.6 s policy floor dominates completely** and the
STT constant is irrelevant. That is a result worth internalising before you go shopping for a fast
STT: on the VAD-only path, buying an STT faster than 0.8 s TTFS buys you nothing at all unless you
also lower `user_speech_timeout`, which is a product decision about how long a Korean speaker's
mid-sentence pause may be — not a latency decision.

**Which term goes on your sheet, then?** Whichever strategy you configure. Write down the strategy
name next to the number; a budget line with no strategy attached is not checkable.

**The one thing this chapter adds to ch-06's table.** Provider choice moves this term by
**2.14 − 0.35 = 1.79 s** at the extremes (`XAI_TTFS_P99` vs `DEEPGRAM_TTFS_P99` /
`DEEPGRAM_SAGEMAKER_TTFS_P99` / `SONIOX_TTFS_P99`, `stt_latency.py:45,46,61,63`). That is larger than
your **entire** P50 budget. It is the single biggest lever in the file, and it is a config argument.
The figure's provider control is a read-only dropdown of those names and constants for exactly this
demonstration — it does not rebuild ch-06's table, it *spends* it.

### 3.2 Term 3: rule evaluation — TBD ms

Here is the empty slot, and it stays empty.

Any processor you place serially between the aggregated user transcript and the LLM service lands on
the critical path. [[boson-layers-rules]] establishes that boson's `LayerPipeline` is exactly such a
thing: a two-phase-commit vote over one already-complete user utterance, 1,206 LOC across
`gateway/layers/` and `gateway/rules/`, which *must* see a finished `str` and therefore *must* sit
after the aggregator. Its migration note is unambiguous — the whole transaction has to collapse into
**one** `FrameProcessor`, because `push_frame()` is irreversible and a veto spread across processors
cannot be rolled back.

So the budget line exists, it is real, and it is serial. What it costs is a function of how many
checks run, whether any of them call an LLM, and whether they block the pipeline or race it. All of
that is [[ch-12/read]]'s subject. It has an actual answer with an actual number, and giving it to you
here would hand you the conclusion of the chapter whose entire purpose is to make you derive it.

So the line on your sheet reads, verbatim:

```
rule evaluation ................................ TBD ms   (ch-12 prices it; ch-13 measures it)
```

The figure's rule knob is draggable and has **no default value** and a label saying so. Drag it and
watch what happens to the total against your P50 line. That is the exercise: arrive at ch-12 owning
the denominator, not the answer.

One thing this chapter *does* give you, because it is a mechanism and not a number — the instruction
for how that term becomes visible instead of becoming an unexplained gap. It is §12.

### 3.3 Term 4: LLM time-to-first-token, and why TTFB is the wrong clock

The LLM term is the one everybody optimises and, as §5 shows, rarely the one that dominates. It is
also the term where Pipecat's naming will mislead you if you skim.

`TTFBMetricsData` does **not** measure "time until the model starts answering." It measures time
until the model produces *any* output. The docstring is explicit and it is the most important
docstring in this chapter:

**`src/pipecat/processors/metrics/frame_processor_metrics.py:137-156`**
```python
    async def stop_ttfb_metrics(self, *, end_time: float | None = None):
        """Stop TTFB measurement and generate metrics frame.

        TTFB ends at the first output the service produces, of any kind —
        including content a caller never sees, such as an LLM's reasoning. Events
        that merely acknowledge the request (an HTTP response head, a stream-open
        or keepalive event) carry no output and must not stop it, or TTFB
        measures connection setup rather than the service's response.

        Only the first call per measurement takes effect, so services can call
        this from every branch that handles output rather than tracking which
        arrived first.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.

        Returns:
            MetricsFrame containing TTFB data, or None if not measuring.
        """
```

Two rules in one paragraph. First: **reasoning tokens stop the TTFB clock.** A reasoning model that
thinks for 900 ms and then answers will report a *flattering* TTFB and a customer who waited. Second:
an HTTP response head or a keepalive must **not** stop it, or you would be measuring connection
setup.

The number that actually corresponds to "the customer's wait" is TTFAT:

**`src/pipecat/metrics/metrics.py:64-96`**
```python
class TTFATMetricsData(MetricsData):
    """Time To First Answer Token (TTFAT) metrics data.

    Measures the time from an LLM request to the first token of the answer the
    caller sees, i.e. time-to-first-byte plus any reasoning the model streamed
    first. ``ttfat`` is reported with its breakdown so consumers can see how much
    of the perceived latency is the model thinking rather than responding,
    without correlating a separate ``TTFBMetricsData``.

    A turn that answers with a tool call rather than text ends the measurement at
    the call, taken where it first appears in the stream. Answering from a tool
    result takes a second inference, so such a turn reports twice — once for the
    call, once for the answer built from its result. Each figure covers one
    inference; consumers wanting one per user turn keep the first, which is the
    one that measures how quickly the model began responding at all.

    Reported only by LLM services that answer in text. Speech-to-speech services
    answer in audio, which has no answer token to measure to.

    Parameters:
        ttfat: TTFAT measurement in seconds (``ttfb`` plus ``thinking_time``).
        ttfb: Time-to-first-byte that TTFAT builds on, in seconds. This mirrors
            the standalone ``TTFBMetricsData`` (emitted earlier) for convenience;
            it is not a separate measurement, so don't aggregate both.
        thinking_time: Time between the model's first output and the first answer
            token, in seconds (``ttfat`` minus ``ttfb``). Reasoning is the usual
            reason a model streams something before it starts answering, though a
            model that reasons none still spends a little time here.
    """

    ttfat: float
    ttfb: float
    thinking_time: float
```

`ttfat = ttfb + thinking_time`, and — exactly as with TTFA in [[ch-07/read]] §1.2 — the mirrored
`ttfb` field is a **copy for convenience**, *"not a separate measurement, so don't aggregate both."*
If your Lina dashboard sums `TTFBMetricsData.value` across a call *and* sums `TTFATMetricsData.ttfat`,
you double-count the model's response time. This is the second time the same trap appears in the
same file; treat it as a house rule: **the mirrored field is for display, never for aggregation.**

The call sites are where you check the claim. In the OpenAI base LLM:

**`src/pipecat/services/openai/base_llm.py:436-446`**
```python
    async def _process_context(self, context: LLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()
```

**`src/pipecat/services/openai/base_llm.py:502-513`**
```python
                    if chunk.choices is None or len(chunk.choices) == 0:
                        continue

                    await self.stop_ttfb_metrics()

                    if not chunk.choices[0].delta:
                        continue

                    if chunk.choices[0].delta.tool_calls:
                        # A turn that only calls tools produces no answer text, so
                        # the call itself is what the caller gets and TTFAT ends
                        # here rather than going unmeasured.
                        await self.stop_ttfat_metrics()
```

TTFB stops on the first chunk with choices — reasoning included. TTFAT stops separately, either at
the first tool-call delta (above) or at the first pushed text token, in the shared base class:

**`src/pipecat/services/llm_service.py:737-755`**
```python
    async def _push_llm_text(self, text: str):
        """Push LLM text, using turn completion detection if enabled.

        This helper method simplifies text pushing in LLM implementations by
        handling the conditional logic for turn completion internally.

        Args:
            text: The text content from the LLM to push.
        """
        # Measured before turn-completion filtering, which can hold text back or
        # drop it entirely — neither says anything about how fast the model
        # answered.
        if self.reports_ttfat:
            await self.stop_ttfat_metrics()
```

Note where that sits: **before** turn-completion filtering. The metric measures the *model*, not the
pipeline's decision to hold the model's output back. And `reports_ttfat` is derived, not configured:

**`src/pipecat/services/llm_service.py:459-472`**
```python
    def reports_ttfat(self) -> bool:
        """Whether this service reports time-to-first-answer-token.

        Speech-to-speech services answer in audio, which has no answer token to
        measure to, so only text-answering services report it. Derived from
        :meth:`service_metadata_frame` and cached, since it is read once per
        streamed token.

        Returns:
            True if this service reports TTFAT.
        """
        if self._reports_ttfat is None:
            self._reports_ttfat = not self.service_metadata_frame().is_realtime_service
        return self._reports_ttfat
```

**Consequence for Lina, stated plainly:** if you ever move to a speech-to-speech model, `TTFAT`
silently stops existing for that service, and your budget loses its LLM line. Nothing errors. The
field just stops appearing. Plan for that before it happens, not after.

**The "reports twice" case is not an edge case for you.** Lina calls tools. A turn that answers *from
a tool result* runs two inferences and therefore emits two `TTFATMetricsData`. The docstring tells
you which one to keep — *"consumers wanting one per user turn keep the first"* — and separately, the
tool execution itself is timed as its own line (`FunctionCallMetrics`, §7.3). So a tool turn's LLM
cost on your sheet is three numbers, not one: first inference TTFAT, function duration, second
inference TTFAT.

### 3.4 Term 5: text aggregation

[[ch-07/read]] §4 built this. Recall the definition and move on:

**`src/pipecat/metrics/metrics.py:193-203`**
```python
class TextAggregationMetricsData(MetricsData):
    """Text aggregation time metrics data.

    Measures the time from the first LLM token to the first complete sentence,
    representing the latency cost of sentence aggregation in the TTS pipeline.

    Parameters:
        value: Aggregation time in seconds.
    """

    value: float
```

The reason it is on the budget as its own line rather than folded into "TTS" is that it is *not* the
TTS's cost — it is the cost of a policy decision (`TextAggregationMode.SENTENCE`, the default) made
in your pipeline, which you can change. `TextAggregationMode`'s own docstring prices it at
"~200-300 ms per sentence" ([[ch-07/read]] §4.1).

Two mechanical facts that matter for the budget and were not in ch-07:

**Only the first aggregation of a turn is on the budget.** The window opens on the first non-
transcription `TextFrame` and closes on the first non-`TOKEN` aggregate:

**`src/pipecat/services/tts_service.py:766-772`**
```python
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, InterimTranscriptionFrame)
            and not isinstance(frame, TranscriptionFrame)
        ):
            await self.start_text_aggregation_metrics()
            await self._process_text_frame(frame)
```

**`src/pipecat/services/tts_service.py:1099-1101`**
```python
            if aggregate.type != AggregationType.TOKEN:
                # Stop the aggregation metric on the first sentence only.
                await self.stop_text_aggregation_metrics()
```

Second and third sentences aggregate too, but they aggregate *under* the playback of the first — 
failure mode 1 again. The observer enforces the same rule when it builds the breakdown:

**`src/pipecat/observers/user_bot_latency_observer.py:333-341`**
```python
            elif isinstance(metrics_data, TextAggregationMetricsData):
                # Only keep the first measurement — it's the one that
                # impacts the initial speaking latency.
                if self._text_aggregation is None:
                    self._text_aggregation = TextAggregationBreakdownMetrics(
                        processor=metrics_data.processor,
                        start_time=now - metrics_data.value,
                        duration_secs=metrics_data.value,
                    )
```

**And the term is invisible to every exporter.** Flagged here, proved in §8.3: `RTVIObserver` does
not bucket `TextAggregationMetricsData`, `MetricsLogObserver` does not import it, and `SentryMetrics`
does not override anything that produces it. The **only** consumer in the tree is
`UserBotLatencyObserver`. If you instrument Lina via RTVI or Sentry and not via the latency observer,
this budget line does not exist for you.

### 3.5 Term 6: TTS time-to-first-AUDIBLE-sample

[[ch-07/read]] §1 built this too, and it is the second of the two biggest terms. Recall the identity —
`ttfa = ttfb + leading_silence` — and the fact that `leading_silence` is *measured*, not declared,
by running an energy-based onset detector over the streaming PCM:

**`src/pipecat/processors/metrics/frame_processor_metrics.py:208-227`**
```python
        if not self._ttfa_active or sample_rate <= 0:
            return None

        self._ttfa_buffer += audio
        onset = detect_speech_onset(self._ttfa_buffer, sample_rate, num_channels)
        if onset is None:
            # No confirmed onset yet. Bound memory against pathologically long
            # (or never-arriving) silence by giving up past a sane limit.
            max_bytes = int(self._TTFA_MAX_BUFFER_SECONDS * sample_rate * max(num_channels, 1) * 2)
            if len(self._ttfa_buffer) >= max_bytes:
                logger.debug(
                    f"{self._processor_name()} TTFA: no onset within "
                    f"{self._TTFA_MAX_BUFFER_SECONDS:.0f}s of audio; not reporting"
                )
                self._ttfa_active = False
                self._ttfa_buffer = b""
            return None

        silence_duration = onset / sample_rate
        value = self._last_ttfb_time + silence_duration
```

`_TTFA_MAX_BUFFER_SECONDS = 3.0` (`:43`). Past three seconds of audio with no confirmed onset the
measurement is abandoned with a debug log and **nothing is reported**. Not a zero, not a warning at
`error` level — a silently missing line. Worth knowing before you go looking for a TTFA number that
is not there.

The budget-relevant addition this chapter makes: the onset detector's own parameters set the
resolution of this line.

**`src/pipecat/audio/utils.py:326-335`**
```python
def detect_speech_onset(
    pcm_bytes: bytes,
    sample_rate: int,
    num_channels: int = 1,
    *,
    frame_ms: float = 10.0,
    hop_ms: float = 1.0,
    threshold_db: float = -40.0,
    min_voiced_ms: float = 50.0,
) -> int | None:
```

`hop_ms = 1.0` is the onset resolution — so `leading_silence` is quantised to 1 ms, which is far
finer than anything else on this sheet. `threshold_db = -40.0` is documented as sitting *"above
typical TTS noise-floor padding and below voiced onset"* (`:358-359`), and `min_voiced_ms = 50.0`
rejects transients. The gate is fixed; there is no per-provider tuning. For a Korean TTS whose voiced
onset begins with a low-energy consonant, a -40 dBFS gate is an assumption you have not tested.
That is a measurement item, not a fault.

### 3.6 Term 7: transport

This is the term Pipecat instruments **least**, and you should know that before you budget it.

The part it does define is the output chunking granularity, which [[ch-08/read]] §6 already spent as
an interrupt-granularity decision. Same number, different account:

**`src/pipecat/transports/base_transport.py:72`**
```python
    audio_out_10ms_chunks: int = 4
```

**`src/pipecat/transports/base_output.py:132-136`**
```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

40 ms per written chunk at the default. That is a floor on how fast the first audio can leave the
process, and a floor on interrupt granularity, and the two are the same number — which is the
trade-off: shrink it and barge-in gets crisper while syscall overhead rises.

For Lina the transport term has a second component that Pipecat does not measure at all: the
telephony serializer. [[transport-telephony]] establishes that a Pipecat phone call is a WebSocket
transport plus a `FrameSerializer`, and that five of six shipped serializers are 8 kHz μ-law. So the
outbound path is a permanent resample pair — `_input_resampler` and `_output_resampler`, both from
`create_stream_resampler(clear_after_secs=...)` — sitting between your 24 kHz TTS output and an
8 kHz wire. There is **no metric class for serialization or resampling**, and no serializer calls
`start_processing_metrics()`. If you want that number you wrap it yourself (§12).

And the genuinely uninstrumented piece: network transit and jitter between your process and the
customer's handset. Nothing in this repository measures it. `TransportTimingReport` (§9.3) measures
*connection* milestones, not per-turn transit. For a Korean PSTN call with a CPaaS in between, that
is not a rounding error, and the only honest thing to put on the sheet is a row marked *measured
externally*.

---

## 4. Which stages are serial and which overlap

This is §1.2's failure mode 1 turned into a rule you can apply.

### 4.1 The serial spine

These stages are strictly ordered, each waiting on the previous, and every millisecond in them is a
millisecond the customer waits:

```
customer stops talking
   │
   ├─ stop_secs ..................... VAD hysteresis (ch-06 §2)              SERIAL
   ├─ max(0, TTFS_p99 − stop_secs) .. STT finalization safety net (ch-06 §15) SERIAL (tail-weighted)
   ├─ rule evaluation ............... TBD ms (§3.2)                          SERIAL
   ├─ LLM TTFB + thinking_time ...... first answer token (§3.3)              SERIAL
   ├─ text aggregation .............. first complete sentence (§3.4)         SERIAL
   ├─ TTS ttfb + leading_silence .... first audible sample (§3.5)            SERIAL
   └─ transport ..................... chunking + serializer + network (§3.6) SERIAL
   │
first audible sample reaches the customer
```

Seven serial terms. Their sum is the budget.

### 4.2 What overlaps, and therefore costs nothing

Four things that look like costs and are not:

**(a) TTS synthesis after the first audible sample.** [[ch-07/read]] §1.1. Playback runs at 1×; the
rest of synthesis hides under it. This is why the TTS *service* being slow in aggregate can be
completely invisible while its `leading_silence` is a disaster.

**(b) Second and subsequent sentence aggregation.** §3.4. Same reason, and it is why the observer
keeps only the first measurement.

**(c) Smart-turn ONNX inference.** §3.1. Absorbed into the absolute STT deadline. Free unless it
overruns the STT P99 window.

**(d) The observer plane itself.** §0.3. `WorkerObserver` enqueues; observers run on their own tasks.
Your custom `LatencyBudgetObserver` writing to a database does not extend the turn — it extends a
queue.

### 4.3 The one that is neither: function calls

A tool call is serial *and* it multiplies the LLM term. The sequence is: first inference produces a
tool call (TTFAT #1) → the function executes (`FunctionCallMetrics.duration_secs`) → a second
inference produces the answer (TTFAT #2) → then aggregation and TTS. Every part of that is on the
critical path.

The observer times the middle piece by pairing two frames on `tool_call_id`:

**`src/pipecat/observers/user_bot_latency_observer.py:260-275`**
```python
        elif isinstance(data.frame, FunctionCallInProgressFrame):
            self._function_call_starts[data.frame.tool_call_id] = (
                data.frame.function_name,
                time.time(),
            )
        elif isinstance(data.frame, FunctionCallResultFrame):
            start = self._function_call_starts.pop(data.frame.tool_call_id, None)
            if start is not None:
                function_name, start_time = start
                self._function_call_metrics.append(
                    FunctionCallMetrics(
                        function_name=function_name,
                        start_time=start_time,
                        duration_secs=time.time() - start_time,
                    )
                )
```

For Lina this is the line that will surprise you. An insurance quote lookup that takes 400 ms against
a carrier API is 400 ms of customer silence, sitting on top of two LLM inferences instead of one. If
your P95 is blowing out, look here before you look at the model.

---

## 5. Why the LLM is rarely the term that dominates

The claim in [[latency-budget-voice]]'s core insight is *"the two biggest terms are not the LLM."*
Do not take it on faith; do the arithmetic with the constants this course has verified.

### 5.1 A worked P50, with real numbers and one honest blank

Assume the default turn-analyzer stop strategy, Deepgram STT, a fast non-reasoning LLM, sentence
aggregation on, a streaming TTS with modest silence padding, and a WebSocket telephony transport.

| Term | Value | Where it comes from |
|---|---|---|
| `stop_secs` | 0.200 s | `VADParams`, ch-06 §2 (192 ms of actual audio) |
| STT safety net (typical) | ~0.000 s | ch-06 §15.3 — finalized transcript cancels the timer |
| rule evaluation | **TBD** | §3.2 |
| LLM `ttfat` | ~0.350 s | vendor-measured; no repo constant exists |
| text aggregation | ~0.250 s | `TextAggregationMode` docstring, ch-07 §4.1 |
| TTS `ttfa` | ~0.300 s | `ttfb` ~0.18 + `leading_silence` ~0.12, ch-07 §1.1 shape |
| transport | ~0.060 s | 40 ms chunk + serializer + local network |
| **total (excl. rules)** | **~1.16 s** | |

Already over your P50 = 1.0 s target with the rule line still blank. That is the honest starting
position and it is why §3.2's slot matters so much.

Now look at where the mass sits. The LLM is 0.35 of 1.16 — **30 %**. The endpointing floor plus
aggregation plus TTFA is 0.75 — **65 %**. Halving the LLM term (a genuinely hard engineering project:
smaller model, better provider, prompt caching) buys 0.175 s. Switching `text_aggregation_mode` to
`TOKEN` (a one-line config change with a documented quality caveat, ch-07 §4.1) buys roughly the
same. Fixing a TTS with 300 ms of leading silence buys more than either.

### 5.2 The same budget at P95

Now change one assumption: the transcript does not finalize before the safety net expires. With
Deepgram, `effective_stt_wait = max(0, 0.35 − 0.2) = 0.15 s`. Total goes to ~1.31 s. Still inside
your P95 = 1.5 s, with the rule line blank.

Change the provider to xAI: `effective_stt_wait = max(0, 2.14 − 0.2) = 1.94 s`, total ~3.10 s. Your
P95 target is gone by a factor of two, and **nothing about your LLM changed.**

That is the entire argument, and it is the reason [[ch-06/read]] put provider selection in ch-06
rather than here. The lever that dominates your tail is a keyword argument in a constructor.

### 5.3 The reasoning-model trap, priced

Suppose you enable a reasoning model to improve objection handling. `TTFBMetricsData.value` might
report 0.22 s — *better* than before, because the model starts emitting reasoning quickly. Meanwhile
`TTFATMetricsData.thinking_time` is 0.9 s and the customer waits 1.12 s for the LLM term alone.

If your dashboard graphs `ttfb` — which is what `MetricsLogObserver` prints most prominently, what
RTVI's `"ttfb"` bucket carries, and what `SentryMetrics` sends as a transaction (§8) — the regression
is **invisible and looks like an improvement**. `ttfat` is the only field that catches it, and §8.3
shows exactly which of the three exporters carry it.

Use the figure's TTFB-versus-TTFAT toggle on this scenario specifically. It is the fastest way to
convince yourself that the metric name you graph is a design decision.

---

## 6. The observer plane

Everything above assumed those numbers exist. This section is why they do.

### 6.1 The claim, stated precisely

Pipecat's instrumentation is not a logging convention. It is a **second, read-only plane over the
frame graph**: a thing that sees every frame transfer between every pair of processors, without
being a processor, without being adjacent to anything, and without being able to modify what it
sees.

That property — call it **non-adjacency** — is worth naming now, because [[ch-12/read]] reaches for
it directly. [[ch-01/read]] taught that Pipecat's splice algebra works because `FrameProcessor`
presents a uniform interface and processors link into a doubly-linked list. Everything that wants to
*participate* in the pipeline must occupy a position in that list. The observer plane is the one
mechanism that gets to watch without occupying a position. boson's `SignalQueue`
([[boson-layers-rules]], `layers/signals.py`) — the append-only log any later layer reads via
`get_recent(seconds, source_layer=None)` — needs exactly that property, and this is where it comes
from.

### 6.2 The contract: four hooks, three event dataclasses

The whole base class is 142 lines and there is nothing hidden in it.

**`src/pipecat/observers/base_observer.py:24-67`**
```python
@dataclass
class FrameProcessed:
    """Event data for frame processing in the pipeline.

    Represents an event where a frame is being processed by a processor. This
    data structure is typically used by observers to track the flow of frames
    through the pipeline for logging, debugging, or analytics purposes.

    Parameters:
        processor: The processor processing the frame.
        frame: The frame being processed.
        direction: The direction of the frame (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.

    """

    processor: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


@dataclass
class FramePushed:
    """Event data for frame transfers between processors in the pipeline.

    Represents an event where a frame is pushed from one processor to another
    within the pipeline. This data structure is typically used by observers
    to track the flow of frames through the pipeline for logging, debugging,
    or analytics purposes.

    Parameters:
        source: The processor sending the frame.
        destination: The processor receiving the frame.
        frame: The frame being transferred.
        direction: The direction of the transfer (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.
    """

    source: "FrameProcessor"
    destination: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int
```

`FramePushed` carries **both endpoints**. That is the difference between the two events and it is
the whole reason both exist: `FrameProcessed` tells you *a processor is handling this frame*;
`FramePushed` tells you *this frame is moving from A to B*. A latency observer wants the second,
because a hop is an edge, not a node.

The third event is about startup, and its docstring explains a clock subtlety you will trip over:

**`src/pipecat/observers/base_observer.py:70-87`**
```python
@dataclass
class ProcessorSetUp:
    """Event data for a processor having been set up.

    Processors are set up concurrently and before any frame flows, so this is
    what a timing observer measures the work a processor does to get ready by.
    The times come from :func:`time.monotonic_ns`, since the pipeline clock
    only starts once the pipeline does.

    Parameters:
        processor: The processor that was set up.
        started_at_ns: When the processor's ``setup()`` began.
        finished_at_ns: When the processor's ``setup()`` returned.
    """

    processor: "FrameProcessor"
    started_at_ns: int
    finished_at_ns: int
```

And the base class itself:

**`src/pipecat/observers/base_observer.py:90-97`**
```python
class BaseObserver(BaseObject):
    """Base class for pipeline frame observers.

    Observers can view all frames that flow through the pipeline without
    needing to inject processors into the pipeline structure. This enables
    non-intrusive monitoring capabilities such as frame logging, debugging,
    performance analysis, and analytics collection.
    """
```

Four hooks, all `async`, all `pass` in the base: `on_process_frame(FrameProcessed)` (`:99`),
`on_push_frame(FramePushed)` (`:111`), `on_processor_setup(ProcessorSetUp)` (`:123`),
`on_pipeline_started()` (`:136`). Subclass, override the one you need, ignore the rest. There is no
registration, no filtering API, no subscription-by-frame-type. You get everything and you branch on
`isinstance` yourself.

### 6.3 Where the hooks actually fire

Two call sites in `frame_processor.py`, and reading both is the difference between believing the
plane is read-only and knowing it.

**`src/pipecat/processors/frame_processor.py:820-841`**
```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
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
```

Two things in that one block. First: the observer is notified **before** the frame is acted on, and
the observer receives the frame object itself — it *could* mutate it, and nothing stops it. "Read-
only" is a convention enforced by nobody. Do not mutate frames in an observer; the plane's entire
value is that its presence changes nothing.

Second, and this is a budget fact: `await self.stop_all_metrics()` on every `InterruptionFrame`, in
every processor. Any TTFB / processing / text-aggregation window left open by a cancelled turn is
closed right there. Interrupted turns do not leak half-open measurements into the next turn.

The push site:

**`src/pipecat/processors/frame_processor.py:1160-1194`**
```python
    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        """Internal method to push frames to adjacent processors.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        observer = self._setup.observer if self._setup else None
        try:
            timestamp = self.get_clock().get_time() if self._setup else 0
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")

                if observer:
                    data = FramePushed(
                        source=self,
                        destination=self._next,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await observer.on_push_frame(data)
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                if observer:
                    data = FramePushed(
                        source=self,
                        destination=self._prev,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await observer.on_push_frame(data)
                await self._prev.queue_frame(frame, direction)
```

`await observer.on_push_frame(data)` happens **before** `await self._next.queue_frame(...)`. Per
§0.3 that await resolves to a queue put, so the cost is one enqueue per registered observer per
frame transfer. Cheap, but not zero — and it happens for *every* frame, including every 40 ms audio
chunk in both directions.

The `timestamp` is the **pipeline clock**:

**`src/pipecat/clocks/system_clock.py:30-38`**
```python
    def get_time(self) -> int:
        """Get the elapsed time since the clock was started.

        Returns:
            The elapsed time in nanoseconds since start() was called.
            Returns 0 if the clock has not been started yet.
        """
        return time.monotonic_ns() - self._time if self._time > 0 else 0
```

Monotonic nanoseconds since the pipeline started. Remember this in §10.3, because
`UserBotLatencyObserver` does not use it.

### 6.4 `can_generate_metrics()` — the opt-in that greys half the pipeline out

The plane sees every frame. It does not follow that every processor produces metrics. The gate is a
method that returns `False` on the base class:

**`src/pipecat/processors/frame_processor.py:488-494`**
```python
    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False
```

Every metric emitter is double-gated on it:

**`src/pipecat/processors/frame_processor.py:504-531`**
```python
    async def start_ttfb_metrics(self, *, start_time: float | None = None):
        """Start time-to-first-byte metrics collection.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_ttfb_metrics(
                start_time=start_time, report_only_initial_ttfb=self.report_only_initial_ttfb
            )

    async def cancel_ttfb_metrics(self):
        """Abandon the current time-to-first-byte measurement without reporting it."""
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.cancel_ttfb_metrics()

    async def stop_ttfb_metrics(self, *, end_time: float | None = None):
        """Stop time-to-first-byte metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfb_metrics(end_time=end_time)
            if frame:
                await self.push_frame(frame)
```

`can_generate_metrics() and self.metrics_enabled` — the processor must opt in **and** the pipeline
must have metrics on (§0.1). The same double gate wraps all twelve wrapper methods at
`frame_processor.py:504-628`, with the three usage methods checking `usage_metrics_enabled`
instead.

How many processors opt in? Count it:

```bash
# run in wiki/raw-data/pipecat/pipecat-src at commit 0cbf9c5b
grep -rn "def can_generate_metrics" src/pipecat/ | wc -l          # 97
grep -rn -A6 "def can_generate_metrics" src/pipecat/ | grep -c "return True"    # 96
grep -rn -A6 "def can_generate_metrics" src/pipecat/ | grep -c "return False"   # 1
```

**97 definitions in the tree. 96 return `True`. Exactly one returns `False` — the base class.** So
the rule in practice is: *services* opt in, near-universally, and *everything you write yourself*
does not. Your Korean phrase chunker, your rule-layer processor, your DTMF handler, your ledger port
— all silent by default, all contributing to what §12 calls an unexplained gap.

The roster is even published at startup. When metrics are on, the worker walks the pipeline and emits
one zero-valued entry per instrumented processor:

**`src/pipecat/pipeline/worker.py:1030-1037`**
```python
    def _initial_metrics_frame(self) -> MetricsFrame:
        """Create an initial metrics frame with zero values for all processors."""
        processors = self._pipeline.processors_with_metrics()
        data = []
        for p in processors:
            data.append(TTFBMetricsData(processor=p.name, value=0.0))
            data.append(ProcessingMetricsData(processor=p.name, value=0.0))
        return MetricsFrame(data=data)
```

**`src/pipecat/pipeline/pipeline.py:153-167`**
```python
    def processors_with_metrics(self):
        """Return processors that can generate metrics.

        Recursively collects all processors that support metrics generation,
        including those from nested pipelines.

        Returns:
            List of frame processors that can generate metrics.
        """
        services = []
        for p in self.processors:
            if p.can_generate_metrics():
                services.append(p)
            services.extend(p.processors_with_metrics())
        return services
```

Sent once, gated on `enable_metrics and send_initial_empty_metrics` (`worker.py:1232`). An RTVI
client therefore learns the exact roster of instrumented processors before the first turn — and a
processor absent from that list will never appear later. That list *is* the answer to "what can I
measure in this pipeline?", and you can print it in three lines.

### 6.5 Metrics travel as frames, and the frame class matters

**`src/pipecat/frames/frames.py:1317-1326`**
```python
class MetricsFrame(SystemFrame):
    """Frame containing performance metrics data.

    Emitted by processors that can compute metrics like latencies.

    Parameters:
        data: List of metrics data collected by the processor.
    """

    data: list[MetricsData]
```

`MetricsFrame(SystemFrame)`. Per [[ch-04/read]] §4 and [[ch-08/read]] §3, `SystemFrame`s are handled
on the input task rather than relayed to the cancellable process task — which means a metrics frame
in flight when a barge-in lands is **not** discarded with the rest of the turn's work. The
measurement of a turn survives the cancellation of that turn. Given that [[ch-08/read]] taught you
how aggressively the cascade throws work away, that is a deliberate exemption and a good one:
cancelled turns are exactly the turns whose numbers you want.

`FrameProcessorMetrics` (`processors/metrics/frame_processor_metrics.py:31`, 390 L) owns every timer
and every method returns a `MetricsFrame` rather than pushing one — the `FrameProcessor` wrapper
decides whether to push. That separation is what makes `SentryMetrics` (§8.4) possible as a drop-in
subclass.

### 6.6 The bundled observers

`src/pipecat/observers/` holds three concrete observers plus the base; `observers/loggers/` holds
four more; and one lives outside the package in `pipeline/worker.py`:

| Observer | File | What it produces |
|---|---|---|
| `TurnTrackingObserver` | `turn_tracking_observer.py:29` (199 L) | `on_turn_started(n)` / `on_turn_ended(n, duration, was_interrupted)` |
| `UserBotLatencyObserver` | `user_bot_latency_observer.py:143` (350 L) | `on_latency_measured`, `on_latency_breakdown`, `on_first_bot_speech_latency` |
| `StartupTimingObserver` | `startup_timing_observer.py` (386 L) | `on_startup_timing_report`, `on_transport_timing_report` |
| `IdleFrameObserver` | `pipeline/worker.py:106` | sets the idle event (ch-04 §9) |
| `MetricsLogObserver` | `loggers/metrics_log_observer.py` | console lines per `MetricsData` type |
| `LLMLogObserver` | `loggers/llm_log_observer.py` | LLM frame trace |
| `DebugLogObserver` | `loggers/debug_log_observer.py` | `frame_types={Frame: (Processor, FrameEndpoint)}` filter |
| `TranscriptionLogObserver` | `loggers/transcription_log_observer.py` | transcript trace |

`TurnTrackingObserver`'s turn boundary is worth one paragraph because it is *not* the same boundary
as the latency observer's:

**`src/pipecat/observers/turn_tracking_observer.py:36-41`**
```python
    - The first turn starts immediately when the pipeline starts (StartFrame)
    - Subsequent turns start when the user starts speaking
    - A turn ends when the bot stops speaking and either:

      - The user starts speaking again
      - A timeout period elapses with no more bot speech
```

with `turn_end_timeout_secs=2.5` (`:52`). So "turn duration" from this observer includes the bot's
entire speech plus up to 2.5 s of trailing silence. It is a *conversation-structure* metric, not a
latency metric. Do not put it on the budget sheet.

---

## 7. `LatencyBreakdown`: the budget as a typed object

This is where the plane and the budget become one thing. `UserBotLatencyObserver` is 350 lines and
produces exactly the decomposition §3 described.

### 7.1 The object

**`src/pipecat/observers/user_bot_latency_observer.py:83-111`**
```python
class LatencyBreakdown(BaseModel):
    """Per-service latency breakdown for a single user-to-bot cycle.

    Collected between ``VADUserStoppedSpeakingFrame`` and
    ``BotStartedSpeakingFrame`` when ``enable_metrics=True`` in
    :class:`~pipecat.pipeline.worker.PipelineParams`.

    Parameters:
        ttfb: Time-to-first-byte metrics from each service in the pipeline.
        text_aggregation: First text aggregation measurement, representing
            the latency cost of sentence aggregation in the TTS pipeline.
        user_turn_start_time: Unix timestamp when the user turn started
            (actual user silence, adjusted for VAD stop_secs). ``None`` if
            no ``VADUserStoppedSpeakingFrame`` was observed.
        user_turn_secs: Duration in seconds of the user's turn, measured
            from when the user actually stopped speaking to when the turn
            was released (``UserStoppedSpeakingFrame``). This includes
            VAD silence detection, STT finalization, and any turn analyzer
            wait. ``None`` if no ``UserStoppedSpeakingFrame`` was observed
            (e.g. no turn analyzer configured).
        function_calls: Latency for each function call executed during
            this cycle. Empty if no function calls occurred.
    """

    ttfb: list[TTFBBreakdownMetrics] = Field(default_factory=list)
    text_aggregation: TextAggregationBreakdownMetrics | None = None
    user_turn_start_time: float | None = None
    user_turn_secs: float | None = None
    function_calls: list[FunctionCallMetrics] = Field(default_factory=list)
```

Map it onto §3 directly:

| `LatencyBreakdown` field | Budget terms it covers |
|---|---|
| `user_turn_secs` | §3.1 — `stop_secs` + STT finalization + turn-analyzer wait, **as one fused number** |
| `ttfb: list[...]` | §3.3 (LLM) and §3.5's `ttfb` half, one entry per instrumented service, each tagged with `processor` and `model` |
| `text_aggregation` | §3.4, first measurement only |
| `function_calls` | §4.3 |
| — | §3.2 rule evaluation: **absent unless you emit it.** §12. |
| — | §3.5's `leading_silence`: **absent.** §7.4. |
| — | §3.6 transport: **absent.** |

That table is the honest state of the instrument. It covers four of the seven serial terms, one of
them only in part, and two of the remaining three are things you must add yourself. That is not a
criticism; it is the specification you build against.

Note also that `user_turn_secs` **fuses** two of §3's terms into one. The docstring says so — *"This
includes VAD silence detection, STT finalization, and any turn analyzer wait."* If you want
`stop_secs` and the safety net separated on your dashboard, the observer will not do it for you; you
subtract the known `stop_secs` yourself, or you add a `TurnMetricsData` listener for
`e2e_processing_time_ms` (`metrics.py:206-218`, which is measured *"from VAD speech-to-silence
transition to turn completion"*).

### 7.2 The two clock edges

**`src/pipecat/observers/user_bot_latency_observer.py:236-256`**
```python
        if isinstance(data.frame, VADUserStartedSpeakingFrame):
            # Reset when user starts speaking
            self._user_stopped_time = None
            self._user_turn_start_time = None
            self._user_turn = None
            self._reset_accumulators()
            # If user speaks before the bot's first speech, abandon the
            # first-bot-speech measurement — it's only meaningful for greetings.
            self._first_bot_speech_measured = True
        elif isinstance(data.frame, VADUserStoppedSpeakingFrame):
            # Record the actual time the user stopped speaking, which is
            # the VAD determination time minus the stop_secs silence duration
            # that had to elapse before the VAD confirmed speech ended.
            self._user_stopped_time = data.frame.timestamp - data.frame.stop_secs
            self._user_turn_start_time = self._user_stopped_time
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            # Measure the user turn duration: from actual user silence to
            # turn release. Includes VAD silence detection, STT finalization,
            # and any turn analyzer wait.
            if self._user_stopped_time is not None:
                self._user_turn = time.time() - self._user_stopped_time
```

`self._user_stopped_time = data.frame.timestamp - data.frame.stop_secs` — the same subtraction
[[ch-06/read]] §9 showed inside `STTService`, computed independently here. Two subsystems doing the
same rewind, from the same frame fields:

**`src/pipecat/frames/frames.py:1241-1252`**
```python
class VADUserStoppedSpeakingFrame(SystemFrame):
    """Frame emitted when VAD definitively detects user stopped speaking.

    Parameters:
        stop_secs: The VAD stop_secs duration that was used to confirm the user
            stopped speaking. This represents the silence duration that had to
            elapse before the VAD determined speech ended.
        timestamp: Wall-clock time when the VAD made its determination.
    """

    stop_secs: float = 0.0
    timestamp: float = field(default_factory=time.time)
```

The frame carries the parameter it was decided under. That is the design principle worth stealing:
**a measurement event should carry the configuration it was measured under**, so a consumer can
normalise without reading config. Your rule-layer processor's own metric should do the same.

The closing edge:

**`src/pipecat/observers/user_bot_latency_observer.py:281-307`**
```python
    async def _handle_bot_started_speaking(self):
        """Handle BotStartedSpeakingFrame to emit latency and breakdown."""
        emit_breakdown = False

        # One-time first bot speech measurement (client connect → first speech)
        if self._client_connected_time is not None and not self._first_bot_speech_measured:
            self._first_bot_speech_measured = True
            latency = time.time() - self._client_connected_time
            await self._call_event_handler("on_first_bot_speech_latency", latency)
            emit_breakdown = True

        if self._user_stopped_time is not None:
            latency = time.time() - self._user_stopped_time
            self._user_stopped_time = None
            await self._call_event_handler("on_latency_measured", latency)
            emit_breakdown = True

        if emit_breakdown:
            breakdown = LatencyBreakdown(
                ttfb=list(self._ttfb),
                text_aggregation=self._text_aggregation,
                user_turn_start_time=self._user_turn_start_time,
                user_turn_secs=self._user_turn,
                function_calls=list(self._function_call_metrics),
            )
            await self._call_event_handler("on_latency_breakdown", breakdown)
            self._reset_accumulators()
```

Also note the greeting measurement: `on_first_bot_speech_latency` runs from `ClientConnectedFrame`
to the first `BotStartedSpeakingFrame`, **once**, and is abandoned if the customer speaks first
(`:244`). For an outbound tele-sales call that is a real product number — how long after the
customer picks up does Lina start talking — and it is one event handler away.

### 7.3 What resets, and what deliberately does not

**`src/pipecat/observers/user_bot_latency_observer.py:257-259`**
```python
        elif isinstance(data.frame, InterruptionFrame):
            # Discard stale metrics from cancelled LLM/TTS cycles
            self._reset_accumulators()
```

**`src/pipecat/observers/user_bot_latency_observer.py:343-350`**
```python
    def _reset_accumulators(self):
        """Clear per-cycle metric accumulators."""
        self._ttfb = []
        self._text_aggregation = None
        self._user_turn_start_time = None
        self._user_turn = None
        self._function_call_starts = {}
        self._function_call_metrics = []
```

Read the list of what it clears and, more importantly, what it does not. `_user_stopped_time` is
**not** in it. Only `VADUserStartedSpeakingFrame` clears that (`:238`).

So on a barge-in: the per-service breakdown is thrown away, but the top-level clock keeps running
from the *original* speech-end. Is that a bug? No — trace it. A barge-in arrives because the customer
started speaking, which means `VADUserStartedSpeakingFrame` has already been seen (that is what fired
the interruption in the first place, [[ch-08/read]] §1), and *that* handler cleared
`_user_stopped_time` to `None`. The `InterruptionFrame` branch exists for the other interruption
sources — [[ch-08/read]] counted nine originators — where no VAD start preceded it. In those cases
keeping `_user_stopped_time` is correct: the customer is still waiting for a reply to the utterance
they finished, and the clock should still be running.

Subtle, and worth having traced once, because it is the difference between "cancelled cycles do not
pollute the numbers" (true, for the breakdown) and "cancelled cycles are not measured" (false, and
you would not want it to be true).

There is also frame deduplication, because the same frame is observed at multiple hops:

**`src/pipecat/observers/user_bot_latency_observer.py:215-227`**
```python
        # Only process downstream frames
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        # Skip already processed frames (bounded deque + set)
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)
        self._frame_history.append(data.frame.id)

        if len(self._processed_frames) > len(self._frame_history):
            self._processed_frames = set(self._frame_history)
```

Downstream only, and a bounded `deque(maxlen=100)` + `set` pair for id dedup. Necessary because
`on_push_frame` fires once per *hop*, not once per frame, so a frame crossing eight processors would
otherwise be counted eight times. If you write your own observer that reacts to frames rather than
to hops, copy this pattern — it is four lines and it is the single most common bug in hand-rolled
observers.

### 7.4 The honest limit that changes what the headline number means

Here is the finding that matters most in this section, and it is not in any excerpt — it comes from
opening the output transport.

`on_latency_measured` stops at `BotStartedSpeakingFrame`. Where does that frame come from?

**`src/pipecat/transports/base_output.py:793-811`**
```python
        async def _handle_bot_speech(self, frame: Frame):
            # TTS case.
            if isinstance(frame, TTSAudioRawFrame):
                # We will only trigger bot stopped speaking based on the TTSStoppedFrame,
                # if we have received audio from TTS
                self._tts_audio_received = True
                await self._bot_currently_speaking()
            # Speech stream case.
            elif isinstance(frame, SpeechOutputAudioRawFrame):
                await self._maybe_bot_currently_speaking(frame)

        async def _handle_frame(self, frame: Frame):
            """Handle various frame types with appropriate processing.

            Args:
                frame: The frame to handle.
            """
            if isinstance(frame, OutputAudioRawFrame):
                await self._handle_bot_speech(frame)
```

**`src/pipecat/transports/base_output.py:896-916`**
```python
        async def _audio_task_handler(self):
            """Main audio processing task handler."""
            async for frame in self._next_frame():
                # No need to push EndFrame, it's pushed from process_frame().
                if isinstance(frame, EndFrame):
                    # Send some final silence so words don't cut out.
                    await self._send_silence(self._params.audio_out_end_silence_secs)
                    break

                # Handle frame.
                await self._handle_frame(frame)

                # If we are not able to write to the transport we shouldn't
                # push downstream.
                push_downstream = True

                # Try to send audio to the transport.
                try:
                    if isinstance(frame, OutputAudioRawFrame):
                        push_downstream = await self._internal_write_audio_frame(frame)
```

Three facts, in order:

1. `_handle_frame(frame)` runs **before** `_internal_write_audio_frame(frame)`. So
   `BotStartedSpeakingFrame` is pushed *before* the bytes go to the transport, let alone to the
   network.
2. On the TTS path (`TTSAudioRawFrame`), `_bot_currently_speaking()` is called **unconditionally**
   on the first chunk. There is no silence check. The silence check exists only on the
   speech-to-speech path, via `_maybe_bot_currently_speaking` → `if not is_silence(frame.audio)`
   (`:785-787`).
3. Therefore `BotStartedSpeakingFrame` fires on the **first byte of TTS audio dequeued at the output
   transport, including when that byte is silence.**

Which gives the conclusion:

> **`on_latency_measured` measures to first *byte*, not to first *audible sample*, and it stops
> before the network write.** It is missing `leading_silence` (§3.5) and it is missing transport
> (§3.6). Against your `CLAUDE.md` definition — *"to the first audible assistant sample"* — it is
> systematically **optimistic**.

By how much? Exactly `TTFAMetricsData.leading_silence` plus your transport transit. Both are
knowable: the first is already measured and sitting in a `MetricsFrame` your observer is already
receiving — it is just not one of the two types `_handle_metrics_frame` accumulates
(`:324-341` handles `TTFBMetricsData` and `TextAggregationMetricsData`, and nothing else). The
second is not measured by anything.

**This is a ~20-line fix and it is your first framework-extension move (§13.1).** Subclass
`UserBotLatencyObserver`, accumulate `TTFAMetricsData`, and report
`on_latency_measured + leading_silence` as the number you hold against 1.0 s / 1.5 s. Until you do,
your dashboard and your `CLAUDE.md` are measuring two different intervals and the dashboard is
winning arguments it should be losing.

The figure's TTFB-versus-TTFA toggle draws exactly this gap as a separate hatched segment at the end
of the waterfall. Toggle it once and note how far past the P50 line the bar moves.

### 7.5 `chronological_events()`

The one convenience method, and it is the right shape for a log line:

**`src/pipecat/observers/user_bot_latency_observer.py:113-140`**
```python
    def chronological_events(self) -> list[str]:
        """Return human-readable event labels sorted by start time.

        Collects all sub-metrics into a flat list, sorts by ``start_time``,
        and returns formatted strings suitable for logging.

        Returns:
            List of formatted strings, one per event, in chronological order.
        """
        events: list[tuple] = []

        if self.user_turn_start_time is not None and self.user_turn_secs is not None:
            events.append((self.user_turn_start_time, f"User turn: {self.user_turn_secs:.3f}s"))

        for t in self.ttfb:
            events.append((t.start_time, f"{t.processor}: TTFB {t.duration_secs:.3f}s"))

        for fc in self.function_calls:
            events.append((fc.start_time, f"{fc.function_name}: {fc.duration_secs:.3f}s"))

        if self.text_aggregation:
            ta = self.text_aggregation
            events.append(
                (ta.start_time, f"{ta.processor}: text aggregation {ta.duration_secs:.3f}s")
            )

        events.sort(key=lambda e: e[0])
        return [label for _, label in events]
```

Note how `start_time` is reconstructed for the TTFB and aggregation entries — `now -
metrics_data.value` at the moment the `MetricsFrame` is observed (`:329`, `:339`). It is a
back-computed origin, not a recorded one. Good enough to order a waterfall; not good enough to
correlate across processes. Which is what §10.3 is about.

---

## 8. Where the numbers go: four consumers, and what each one drops

The same `MetricsFrame` stream feeds four sinks. They do not carry the same information, and the
differences are the thing to learn.

### 8.1 Console — `MetricsLogObserver`

**`src/pipecat/observers/loggers/metrics_log_observer.py:33-47`**
```python
class MetricsLogObserver(BaseObserver):
    """Observer to log metrics activity to the console.

    Monitors and logs all MetricsFrame instances, including:

    - TTFBMetricsData (Time To First Byte)
    - TTFAMetricsData (Time To First Audio)
    - TTFATMetricsData (Time To First Answer Token)
    - ProcessingMetricsData (General processing time)
    - LLMUsageMetricsData (Token usage statistics)
    - STTUsageMetricsData (Speech-to-Text audio seconds)
    - TTSUsageMetricsData (Text-to-Speech character counts)
    - TurnMetricsData (Turn prediction metrics)
```

Eight types listed. `TextAggregationMetricsData` is not among them, and it is not imported by the
file. Filterable via `include_metrics: set[type[MetricsData]] | None` (`:66-68`).

### 8.2 The client wire — `RTVIObserver`

**`src/pipecat/processors/frameworks/rtvi/observer.py:839-873`**
```python
    async def _handle_metrics(self, frame: MetricsFrame):
        """Handle metrics frames and convert to RTVI metrics messages."""
        metrics = {}
        for d in frame.data:
            if isinstance(d, TTFBMetricsData):
                if "ttfb" not in metrics:
                    metrics["ttfb"] = []
                metrics["ttfb"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, TTFAMetricsData):
                if "ttfa" not in metrics:
                    metrics["ttfa"] = []
                metrics["ttfa"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, TTFATMetricsData):
                if "ttfat" not in metrics:
                    metrics["ttfat"] = []
                metrics["ttfat"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, ProcessingMetricsData):
                if "processing" not in metrics:
                    metrics["processing"] = []
                metrics["processing"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, LLMUsageMetricsData):
                if "tokens" not in metrics:
                    metrics["tokens"] = []
                metrics["tokens"].append(d.value.model_dump(exclude_none=True))
            elif isinstance(d, STTUsageMetricsData):
                if "stt_usage" not in metrics:
                    metrics["stt_usage"] = []
                metrics["stt_usage"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, TTSUsageMetricsData):
                if "characters" not in metrics:
                    metrics["characters"] = []
                metrics["characters"].append(d.model_dump(exclude_none=True))

        message = RTVI.MetricsMessage(data=metrics)
        await self.send_rtvi_message(message)
```

Seven buckets: `ttfb`, `ttfa`, `ttfat`, `processing`, `tokens`, `stt_usage`, `characters`. On by
default (`RTVIObserverParams.metrics_enabled: bool = True`, `observer.py:178`) — but it carries
nothing at all until `PipelineParams.enable_metrics=True` (§0.1), because nothing produces a
`MetricsFrame` before then.

Not bucketed: `TextAggregationMetricsData`, `TurnMetricsData`. Both fall through the chain silently.

### 8.3 The gap, named

Line the three exporters up against the eight metric classes:

| Metric class | `MetricsLogObserver` | `RTVIObserver` | `SentryMetrics` | `UserBotLatencyObserver` |
|---|:---:|:---:|:---:|:---:|
| `TTFBMetricsData` | yes | `"ttfb"` | yes | yes |
| `TTFAMetricsData` | yes | `"ttfa"` | no | **no** |
| `TTFATMetricsData` | yes | `"ttfat"` | no | no |
| `ProcessingMetricsData` | yes | `"processing"` | yes | no |
| `TextAggregationMetricsData` | **no** | **no** | **no** | yes (first only) |
| `LLMUsageMetricsData` | yes | `"tokens"` | no | no |
| `STTUsageMetricsData` | yes | `"stt_usage"` | no | no |
| `TTSUsageMetricsData` | yes | `"characters"` | no | no |
| `TurnMetricsData` | yes | **no** | no | no |

Two rows deserve your attention.

**`TextAggregationMetricsData` reaches exactly one consumer**, and it is the one that is off by
default (§0.2). Your §3.4 budget line — 200–300 ms, a config change away from being much smaller —
is the least observable number in the entire budget.

**`TTFAMetricsData` reaches the log and the client but not the latency observer.** That is §7.4
restated from the other direction: the leading-silence number exists, and it exists on the wire, and
the thing computing your headline latency does not read it.

### 8.4 Sentry, and what a drop-in exporter costs you

**`src/pipecat/processors/metrics/sentry.py:25-31`**
```python
class SentryMetrics(FrameProcessorMetrics):
    """Frame processor metrics integration with Sentry monitoring.

    Extends FrameProcessorMetrics to send time-to-first-byte (TTFB) and
    processing metrics as Sentry transactions for performance monitoring
    and debugging.
    """
```

Passed per service as `metrics=SentryMetrics()` (see `examples/observability/
observability-sentry-metrics.py:69,77,82`). It overrides exactly four methods —
`start_ttfb_metrics` (`:71`), `stop_ttfb_metrics` (`:93`), `start_processing_metrics` (`:112`),
`stop_processing_metrics` (`:129`) — and nothing else. So a Sentry-instrumented Lina reports TTFB
and processing time and is blind to TTFA, TTFAT, text aggregation and all usage metrics. Per §5.3,
that is precisely the configuration in which a reasoning-model regression reads as an improvement.

### 8.5 OpenTelemetry

`utils/tracing/` carries the span machinery: `setup_tracing(service_name, exporter,
console_export)`, `is_tracing_available()`, and `TurnTraceObserver`, which creates one span per turn
under a conversation span. Its one latency attribute:

**`src/pipecat/utils/tracing/turn_trace_observer.py:88-105`**
```python
        @latency_tracker.event_handler("on_latency_measured")
        async def on_latency_measured(tracker, latency_seconds):
            await self._handle_latency_measured(latency_seconds)

    async def _handle_latency_measured(self, latency_seconds: float):
        """Handle latency measurement events.

        Called when the latency tracker measures user-to-bot latency.
        Adds the latency as an attribute to the current turn span.

        Args:
            latency_seconds: The measured latency in seconds.
        """
        if self._current_span and is_tracing_available():
            self._current_span.set_attribute("turn.user_bot_latency_seconds", latency_seconds)
```

Read the subscription: `TurnTraceObserver` does not compute latency itself, it **subscribes to
`UserBotLatencyObserver`**. That is why §0.2's wiring is a chain rather than a switch, and it is why
turning tracing on without metrics on gives you spans with no latency attribute.

`turn.user_bot_latency_seconds` — the headline number from §7.4, with all of §7.4's caveats, and
**not** the breakdown. Service-level spans come from the `@traced_llm` / `@traced_stt` /
`@traced_tts` decorators.

### 8.6 Collection is provided; aggregation is left to you

**`examples/observability/` holds exactly three files**, totalling 396 lines:

```
observability-observer.py         193 L
observability-sentry-metrics.py   146 L
observability-heartbeats.py        57 L
```

The largest one is the shape of everything you will write:

**`examples/observability/observability-observer.py:142-161`**
```python
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[
            CustomObserver(),
            LLMLogObserver(),
            DebugLogObserver(
                frame_types={
                    TTSTextFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
                    UserStartedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
                    EndFrame: None,
                }
            ),
        ],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
```

Note what is **not** in that list: `UserBotLatencyObserver`. The repo's own observability example
does not instrument latency. It logs frames.

There is **no dashboard, no exporter beyond Sentry/OTel, and no aggregation example anywhere in the
tree.** No P50/P95 computation, no histogram, no rollup, no storage. Pipecat gives you typed events
with attribution and a delivery mechanism, and stops. Your `CLAUDE.md` says *"Report P50/P95/P99"* —
that reporting layer is code you write, and it should be on the migration estimate as a line item,
not assumed.

For scale: `examples/observability/` is 396 lines. A minimum honest Lina observability layer — a
`LatencyBudgetObserver` that corrects for §7.4, writes one row per turn, and computes rolling
percentiles — is roughly the same order. It is a day, not a week. But it is not zero, and nothing in
this repository does it for you.

---

## 9. Auto-wiring, precisely

Putting §0.1, §0.2 and §8 together into the operational statement.

### 9.1 What a default `PipelineWorker` gives you

| Feature | Default | Effect |
|---|---|---|
| `enable_turn_tracking` | `True` | `TurnTrackingObserver` appended → `on_turn_started` / `on_turn_ended` available |
| `enable_rtvi` | `True` | `RTVIProcessor()` constructed and prepended; `create_rtvi_observer(...)` appended |
| `enable_tracing` | `False` | no `UserBotLatencyObserver`, no `TurnTraceObserver`, no spans |
| `PipelineParams.enable_metrics` | `False` | **no `MetricsFrame` is ever produced** |
| `PipelineParams.enable_usage_metrics` | `False` | no token / audio-second / character counts |
| `idle_timeout_secs` | `IDLE_TIMEOUT_SECS = 300` | idle observer wired (ch-04 §9) |
| `enable_heartbeats` | `False` | `HEARTBEAT_SECS = 1.0`, `HEARTBEAT_MONITOR_SECS = 10.0` unused |

So the true out-of-the-box state is: **turn events yes, RTVI transport for metrics yes, metrics
themselves no, latency breakdown no.** The RTVI metrics channel is open and empty. That is a
confusing failure mode to debug from the client side — the client is correctly subscribed to a
message type the server will never send — and it is worth knowing before you spend an afternoon on
it.

### 9.2 The minimum wiring for Lina

Four things, and they are all constructor arguments:

```python
# Not a full bot — the instrumentation surface only.
# Everything else (transport, turn strategies, TTS block) is ch-05 / ch-06 / ch-07.
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.worker import PipelineParams, PipelineWorker

latency = UserBotLatencyObserver()          # 1. construct it yourself; do NOT rely on enable_tracing (§0.2)

worker = PipelineWorker(
    pipeline,
    params=PipelineParams(
        enable_metrics=True,                # 2. without this nothing is measured at all (§0.1)
        enable_usage_metrics=True,          # 3. tokens / audio-seconds / characters for cost, not latency
    ),
    observers=[latency, MetricsLogObserver()],   # 4. the plane is opt-in per observer
)

@latency.event_handler("on_latency_breakdown")
async def _on_breakdown(observer, breakdown):
    for line in breakdown.chronological_events():   # §7.5
        logger.info(line)
```

`report_only_initial_ttfb` stays `False`: you want every TTFB in a turn, not just the first, because
a tool-calling turn has two LLM inferences (§4.3) and reporting only the first hides the second.

### 9.3 Cold start is a separate observer

`StartupTimingObserver` answers a different question — not "how fast is a turn" but "how long before
the first turn is possible":

**`src/pipecat/observers/startup_timing_observer.py:76-92`**
```python
class ProcessorStartupTiming(BaseModel):
    """Startup timing for a single processor.

    Parameters:
        processor_name: The name of the processor.
        start_offset_secs: Offset in seconds from the StartFrame to when this
            processor's start() began.
        duration_secs: What the processor cost to get ready, in seconds: its
            setup() and its start() together.
        setup_duration_secs: How long the processor's setup() took, in seconds,
            which is the part of ``duration_secs`` spent connecting.
    """

    processor_name: str
    start_offset_secs: float
    duration_secs: float
    setup_duration_secs: float
```

with `StartupTimingReport.total_duration_secs` documented as *"Processors are set up concurrently, so
this is the span rather than the sum of what each cost"* (`:100-103`) — the same overlap discipline
as §4.2, applied to startup. And `TransportTimingReport(bot_connected_secs, client_connected_secs)`
for the connection milestones.

For outbound tele-sales this is not a footnote. The customer picks up and the pipeline must already
be warm; `setup_duration_secs` per service tells you which provider connection is making them wait.
Pair it with `on_first_bot_speech_latency` (§7.2) and you have the greeting path fully accounted.

---

## 10. Three honest limits

### 10.1 No perceptual target exists in this repository

Established in §2.2 with the grep. Restated here because it is the kind of thing that quietly becomes
folklore: if you write "Pipecat targets sub-800 ms" in a design document, you are inventing a claim.
Write "our CLAUDE.md targets P50 ≤ 1.0 s / P95 ≤ 1.5 s, measured last-voiced-sample to
first-audible-sample" instead. That one has an owner and a definition.

### 10.2 Collection is provided; aggregation is not

Established in §8.6. Three example files, 396 lines, no dashboard, no percentile computation, no
storage. Budget the reporting layer.

### 10.3 The instrument has error bars, and they are systematic

Three separate sources of error, all verifiable, all pointing the same direction:

**(a) Two clocks.** `FramePushed.timestamp` is the pipeline clock — monotonic nanoseconds since
pipeline start (`system_clock.py:30-38`). `UserBotLatencyObserver` computes every one of its
durations with `time.time()` — wall clock, non-monotonic — at `:232`, `:256`, `:263`, `:273`,
`:288`, `:293`, `:322`. It never reads `data.timestamp`. So the plane offers a monotonic clock and the headline
latency observer does not use it. On a machine whose wall clock steps (NTP correction mid-call), a
measurement can be wrong by the step. `stop_ttfb_metrics` guards against the negative case explicitly
— *"a wall clock that stepped backwards mid measurement"* (`frame_processor_metrics.py:163-166`) —
which tells you the authors have seen it happen.

**(b) Observer queue lag.** Per §0.3, `time.time()` in the observer runs when the event is
*dequeued*, not when it was pushed. Under load, every duration the observer computes is biased
**upward** by the queue lag. The queues are `asyncio.Queue()` with no maxsize
(`worker_observer.py:175`), so lag is unbounded and there is no drop policy and no metric for the
lag itself. An observer that falls behind grows memory and skews numbers rather than reporting that
it fell behind.

**(c) Back-computed start times.** §7.5: `start_time = now - metrics_data.value`. The origin of every
TTFB and aggregation bar in the waterfall is inferred from its duration, not recorded. Fine for
ordering; not fine for cross-process correlation.

None of this makes the instrument useless — it makes it an instrument with a known bias, which is
strictly better than a hand-rolled `time.perf_counter()` with an unknown one. But when you set an
alert threshold at 1.0 s, know that you are alerting on a number that is optimistic by
`leading_silence + transport` (§7.4) and pessimistic by observer lag, and that those two errors do
not cancel in any principled way. The correction in §13.1 fixes the first. The second you monitor by
watching queue depth.

---

## 11. Where both boson stacks stand today

Mechanism only. No scoring — [[ch-13/read]] owns that.

### 11.1 boson-agent's gateway

Per [[rtvi-observability]] and [[boson-gateway-server]], the gateway has two pieces of timing code
and neither is a budget:

1. `packages/gateway/gateway/debug/log_decorator.py` — `time.perf_counter()` around a call, printed
   as `[TRACE …] EXIT (…ms)`. A debug aid. Per-call, unaggregated, not typed, not attributed to a
   turn.
2. The ad-hoc `elapsed_ms` threaded from `bootstrap.py:222`
   (`(_time.monotonic() - started_at) * 1000`) into the barge-in path — `core.py:166
   should_interrupt(session_id, content, elapsed_ms)` → `InterruptHandler.check_barge_in` →
   `policy.evaluate(...)`, where `DurationPolicy(min_ms=500)` allows a barge-in only after 500 ms of
   agent streaming ([[boson-interrupt-subsystem]]). That is a *policy input*, not a measurement: it
   is read once, compared to a threshold, and discarded.

Mapping onto this chapter: `elapsed_ms` is "time since the agent started streaming," which is a
suffix of the LLM term and nothing else. There is no equivalent of `user_turn_secs`, no per-service
attribution, no `MetricsFrame`, no aggregation. `gateway/layers/status.py` (`AgentStatusTracker`,
generating/settling/idle) is the structural analogue of `TurnTrackingObserver` — a turn-phase state
machine — and, like it, produces conversation structure rather than latency.

And the one that matters for [[ch-12/read]]: **`gateway/layers/` and `gateway/rules/` emit no metrics
today.** 1,206 LOC ([[boson-layers-rules]]) sitting serially between the finished utterance and the
LLM, entirely untimed. That is §3.2's slot on the boson side: it is not that the number is unknown,
it is that no mechanism exists to know it.

### 11.2 realtime_voice

Per [[rtv-vs-pipecat-gap]] and [[rtv-webrtc-transport]], three timing-adjacent things exist:

1. `provider_latency_ms` / `endpoint_latency_ms` fields on events. Per [[rtv-vad-chunking]],
   `endpoint_latency_ms = self._silence_frames * frame.duration_seconds * 1000` (`energy.py` L79,
   `silero.py` L89), described there as *"the only self-measured latency in the VAD layer"* — the
   VAD's own reconstruction of how much silence it consumed, i.e. the `stop_secs` term of §3.1,
   computed inside the VAD rather than carried on the frame the way
   `VADUserStoppedSpeakingFrame.stop_secs` is (§7.2).
2. A `VoiceEvent` stream (14 `VoiceEventKind` values) fanned out to the WebRTC data channel — an
   event log, not a metrics plane; no `MetricsData` types, no aggregation.
3. `BoundedAudioOutput.discarded_frames` — *"exposed but never read by anything."*

And two structural facts with direct budget consequences. First, [[rtv-pipeline-session]] records
that realtime_voice has *"any observer plane, any frame-level metrics"* among the things absent by
construction: with no `Frame` base class and no processor abstraction, there is no edge for a
`FramePushed` event to describe. Second, [[rtv-webrtc-transport]] notes the two resample points
(browser Opus → 16 kHz in, 24 kHz → 48 kHz out) are *"neither measured"* while both sit on the
critical path — the same §3.6 blind spot, in a different codebase.

Third, the term that is *structurally* different rather than merely unmeasured:
`OpenAICompatibleUnaryASR` buffers the whole utterance into a WAV and does one
`audio.transcriptions.create` at `finalize()` (`openai_compat.py` L194-242, `timeout_seconds=1.5`).
On the accounting of §3.1, that replaces `max(0, TTFS_p99 − stop_secs)` — a tail term that is usually
cancelled by a finalized transcript — with a **full transcription round-trip that is paid on every
turn, after VAD stop**, with no short-circuit available because there are no interim results to
finalize against. It is a different line on the sheet with a different distribution, and that is the
statement; what to do about it is [[ch-13/read]].

### 11.3 What the migration adds to the budget

boson today has no server-side STT, TTS or VAD ([[latency-budget-voice]]): the turn starts at a
client-delivered text partial. Moving to a server-side voice pipeline therefore **adds** §3's terms
1, 2 and 6 to a budget that previously did not contain them — `stop_secs + TTFS_p99` alone is
0.55 s–2.34 s depending on provider, and TTFA is on top of that. New cost, which must be reclaimed
elsewhere or accepted.

The one term that moves the *other* way is boson's current end-of-turn mechanism.
[[endpointing-turn-boundary]] records it precisely: `PartialDetector.should_finalize
(elapsed_since_last_ms)` is `>= 2000` — a **2000 ms text-silence timer**, and it is the entire boson
end-of-turn mechanism. Against §3.1's default path, `0.2 + max(0, TTFS_p99 − 0.2)` with a fast
provider is roughly 0.35 s. That is the arithmetic to write down; [[ch-06/read]] §15 gives the
mechanism behind it.

The compact/summarization path ([[boson-compact-session]]) runs off-turn and does not enter this
budget — **provided it stays off the critical path.** That proviso is a design constraint you now own
and can check, because the mechanism to check it is `start_processing_metrics()` (§12).

---

## 12. The handoff to ch-12, stated as a mechanism

This is the one instruction this chapter gives about the rule layer, and it is not a number.

Any processor you place serially between the aggregated transcript and the LLM service is on the
critical path. If it does not open a metrics window, its cost does not disappear — it reappears as an
**unexplained gap** between two `TTFBBreakdownMetrics` entries in your `LatencyBreakdown`, and you
will spend a day blaming the network.

The mechanism is two calls, and they are the same two calls `OpenAILLMService` uses on itself:

**`src/pipecat/services/openai/base_llm.py:601-613`**
```python
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
```

Three things to copy exactly:

1. `start_processing_metrics()` before the work, `stop_processing_metrics()` in a **`finally`**. An
   exception must not leave the window open; `stop_ttfb_metrics`'s sibling guard exists precisely
   because open windows get measured against the next unrelated output.
2. Override `can_generate_metrics()` to return `True` on your processor. Per §6.4 the base returns
   `False`, so without the override both calls are no-ops and you will see nothing and suspect the
   framework.
3. The result arrives as `ProcessingMetricsData(processor=<your processor name>, value=<seconds>)`,
   which per §8 reaches the console log and the RTVI `"processing"` bucket — but **not**
   `LatencyBreakdown`, which only accumulates `TTFBMetricsData` and `TextAggregationMetricsData`
   (`user_bot_latency_observer.py:324-341`). So getting your rule cost into the breakdown object
   itself is a fourth step: subclass the observer and accumulate `ProcessingMetricsData` too. That is
   §13.1's other half.

So the budget line for ch-12 is now well-formed even though it is empty. It has a name, a producer, a
metric class, a delivery path, and a known hole in the delivery path:

```
rule evaluation ......... TBD ms   producer: BosonRuleProcessor (ch-12)
                                   metric:   ProcessingMetricsData
                                   visible in: console log, RTVI "processing"
                                   NOT visible in: LatencyBreakdown (unless you subclass, §13.1)
```

Fill in `TBD` in ch-12. Measure it in ch-13.

---

## 13. Framework-extension moves for Lina

Not a summary — four things to build, each an application of a mechanism above to a problem this
chapter did not pose.

### 13.1 `LinaLatencyObserver` — make the headline number match your own definition

The gap in §7.4 is the highest-value fix in this chapter, and it is a subclass. The design, not the
code:

- Subclass `UserBotLatencyObserver`. Override `_handle_metrics_frame` to *also* accumulate
  `TTFAMetricsData` (capturing `leading_silence`) and `ProcessingMetricsData` (capturing §12's rule
  term), then call `super()`.
- Extend `LatencyBreakdown` with two fields: `leading_silence: float | None` and
  `processing: list[ProcessingBreakdownMetrics]`.
- Emit a corrected headline: `on_latency_measured + leading_silence`. That is the number to hold
  against P50 = 1.0 s / P95 = 1.5 s, because it is the number your `CLAUDE.md` defines.
- Keep the uncorrected number too, under a different name, so you can see the size of the correction
  over time. If it is stable you can stop worrying about it; if it drifts, your TTS vendor changed
  something.

The genuinely non-obvious part is not the code, it is the ordering constraint: `TTFAMetricsData` is
emitted by the **TTS service** when it scans the first audio chunk (`tts_service.py:1741` →
`process_ttfa_metrics`), while `BotStartedSpeakingFrame` is emitted by the **output transport**
(`base_output.py:710`) — a later position. So the TTFA frame is pushed from an *earlier* processor
but may be observed after the bot-started frame, depending on hop counts and queue depths. Your
accumulator must therefore be able to attach a late-arriving `leading_silence` to a breakdown it has
already emitted, or hold the breakdown one beat. Deciding which is the real design work, and it is
the kind of thing that only shows up when you trace the two emit sites.

### 13.2 The 8 kHz Korean TTFS benchmark — the number that does not exist

State it plainly, because it is the single most important open number in this course:

> **Korean STT on 8 kHz μ-law telephony audio has no entry in `stt_latency.py`.** Not a bad entry, not
> a stale entry — no entry. Every one of the 23 constants was measured on the provider's standard
> benchmark conditions with `VADParams.stop_secs=0.2`, and none of them was measured on Korean, and
> none of them was measured on 8 kHz companded telephony audio.

The consequence follows from [[transport-telephony]]: five of six shipped serializers are 8 kHz μ-law,
the Nyquist ceiling on the wire is 4 kHz, and μ-law is 8-bit companded. Upsampling 8 k → 16 k before
STT satisfies the model's input contract and restores no bandwidth. For Korean specifically, the
fricative/sibilant band (ㅅ/ㅆ/ㅊ) and much of the cue for 받침 discrimination sits at or above 4 kHz
and is simply not in the signal. A provider that endpoints on acoustic confidence will behave
differently on that signal than on studio audio — possibly slower, possibly faster and wronger. You
do not know which, and neither does this table.

The deliverable: run the benchmark at <https://github.com/pipecat-ai/stt-benchmark> against real
8 kHz μ-law Korean audio with your VAD settings, and pass the result explicitly:

```python
stt = SomeKoreanSTTService(api_key=..., ttfs_p99_latency=<your measured number>)
```

The constructor argument exists precisely for this (`stt_service.py:119-124`), and if you do not pass
it the service warns and falls back to `DEFAULT_TTFS_P99 = 1.0`:

**`src/pipecat/services/stt_service.py:568-575`**
```python
        if not self.supports_ttfs:
            ttfs = 0.0
        else:
            ttfs = self._ttfs_p99_latency
            if ttfs is None:
                ttfs = DEFAULT_TTFS_P99
                logger.warning(f"{self.name}: ttfs_p99_latency not set, using default {ttfs}s")
        return STTMetadataFrame(service_name=self.name, ttfs_p99_latency=ttfs)
```

A silent 1.0 s assumption inside your safety net is a bad place for a guess to live. **Until that
number is measured, every P95 claim about Lina is uncheckable.** The figure draws this row as a
visibly empty bar with exactly that annotation.

### 13.3 A `ProcessingMetricsProcessor` wrapper

Generalise §12. Write one small `FrameProcessor` that takes another processor (or a callable) and
wraps its `process_frame` in `start_processing_metrics()` / `stop_processing_metrics()`, with
`can_generate_metrics()` returning `True`. Then wrap the things Pipecat does not instrument and that
you care about:

- The rule-layer processor (§12) — the ch-12 term.
- Your Korean text aggregator ([[ch-07/read]] §10.1) — currently invisible; it sits inside the TTS
  service's aggregation window, so its cost is folded into `TextAggregationMetricsData` and cannot be
  separated from the sentence-boundary wait.
- The telephony serializer's resample pair (§3.6) — currently invisible, and on every audio frame in
  both directions.

The design question worth thinking about before you write it: a wrapper that measures *per frame*
produces one `ProcessingMetricsData` per frame, which for audio at 40 ms chunks is 25 metrics frames
per second per direction. That is a real load on the observer queues (§10.3b). The right shape is
almost certainly to measure per *turn* — open the window on the frame type that starts the work,
close it on the one that ends it — which is exactly what the LLM service does in §12. Copy the
pattern, not the granularity.

### 13.4 A budget regression test, using the eval harness for what it is not designed for

§2.2 established that `pipecat eval` deliberately does not assert on latency — `within_ms` defaults
to 60 s and only 2 of 38 shipped scenarios set it at all. But the field exists
(`evals/scenario.py:287`, `within_ms: int | None = None`) and the harness enforces it
(`harness.py:1093`, `budget_ms = expectation.within_ms or self._default_timeout_ms`).

So: write one Lina scenario whose expectations carry `within_ms` set to your actual budget rather
than to a timeout, run it in CI against a real bot with `-t eval`, and you have a latency regression
gate the framework does not ship. The honest caveat, and you must state it in the scenario's own
comment: the harness measures from *the turn's user send* over the RTVI websocket, not from the last
voiced sample, so the number it enforces is not the same number as §2.1's. It is a **regression**
gate — it catches "this got 300 ms worse" — not a compliance gate for the P50/P95 target. Those are
different instruments and conflating them is how a green CI ends up covering a product that feels
slow.

---

## 다음 챕터로

What this chapter hands forward, named so later chapters cite it instead of re-deriving it:

- **The budget is seven serial terms and the LLM is one of them.** `stop_secs` → STT finalization →
  **rule evaluation (TBD)** → LLM `ttfat` → text aggregation → TTS `ttfa` → transport. §5's worked
  P50 puts the LLM at ~30 % and the endpointing/aggregation/TTFA block at ~65 %. Everything after the
  first audible sample is free (§4.2); adding TTS synthesis wall time to the budget is the standard
  way to get this wrong.

- **The two biggest levers are keyword arguments.** `ttfs_p99_latency=` moves the tail by up to
  1.79 s between the extremes of `stt_latency.py`'s 23 constants ([[ch-06/read]] §11 owns that
  table); `text_aggregation_mode=` moves the median by 200–300 ms per sentence ([[ch-07/read]] §4).
  Neither is a model change.

- **The observer plane is non-adjacency, and that is why [[ch-12/read]] can use it.** `BaseObserver`
  (`base_observer.py:90`), four hooks, three event dataclasses, notified at
  `frame_processor.py:827-835` and `:1160-1194`, fanned out through `WorkerObserver`'s one-queue-
  one-task-per-observer proxy so a slow observer grows a queue instead of a turn. This is the one
  mechanism in Pipecat that watches the pipeline without occupying a position in it.

- **Nothing is measured until you say so, twice.** `PipelineParams.enable_metrics = False` by
  default (`worker.py:189`), and `can_generate_metrics()` returns `False` on the base class
  (`frame_processor.py:488-494`) — 97 definitions in the tree, 96 return `True`, all of them
  services. Anything you write is silent until you override it. §9.2 is the four-line minimum wiring.

- **`LatencyBreakdown` covers four of the seven terms, and the headline number is optimistic.**
  `on_latency_measured` stops at `BotStartedSpeakingFrame`, which the output transport pushes on the
  first TTS chunk dequeued — before the write, and with no silence check on the TTS path
  (`base_output.py:793-799, 896-916`). So it excludes `leading_silence` and transport, and is
  therefore *not* the interval your `CLAUDE.md` defines. §13.1 is the ~20-line correction.

- **`TextAggregationMetricsData` reaches exactly one consumer**, and `TTFAMetricsData` reaches every
  consumer except the latency observer (§8.3's table). Which metric name you graph is a design
  decision, and graphing `ttfb` makes a reasoning-model regression look like an improvement (§5.3).

- **Collection is provided; aggregation is left to you.** `examples/observability/` is three files
  and 396 lines, with no dashboard, no exporter beyond Sentry/OTel, and no percentile computation
  anywhere in the tree. Your `CLAUDE.md`'s "Report P50/P95/P99" is a line item on the migration
  estimate.

- **Pipecat states no perceptual target.** `README.md:29` says "Ultra-low latency interaction";
  `evals/harness.py:113` sets a deliberately generous 60 s test timeout; 2 of 38 shipped scenarios
  set `within_ms` at all, both above 40 s. The ~200 ms / ~800 ms figures are external voice-UX
  convention, not repo claims. The one target with an owner is yours: **P50 ≤ 1.0 s, P95 ≤ 1.5 s,
  last voiced user sample → first audible assistant sample, including end-of-turn time.**

- **Both boson stacks are near-blind here** (§11): the gateway has a trace decorator and an
  `elapsed_ms` barge-in input; realtime_voice has an unread `discarded_frames` counter, two
  latency fields on events, and no observer plane by construction. Mechanism stated; no verdict —
  [[ch-13/read]] scores it.

[[ch-12/read]] takes the slot this chapter left empty. It now has everything it needs to be argued
rather than asserted: a denominator (§5's ~1.16 s with the rule line blank), a target with an owner
(§2.1), a mechanism to make the term visible (§12), and the knowledge that the term is serial and
irreducible if it wants to veto. The question ch-12 answers is not "where does the rule processor
go" — [[boson-layers-rules]] already forces that, after `LLMUserAggregator` and before the LLM
service. It is "what does it cost there, and is that cost worth in-turn veto?"

Open questions parked here so they are not lost:

- **The 8 kHz Korean TTFS number.** §13.2. Nothing in this course closes it; it is a measurement and
  it needs real telephony audio and a chosen provider. [[ch-13/read]] cannot make a credible P95
  claim without it, so it is on ch-13's critical path, not ch-13's nice-to-have list.
- **The transport term.** §3.6. Serialization, the permanent 8 kHz ↔ 24 kHz resample pair, and
  network transit are unmeasured by anything in this repository. §13.3 gives the wrapper for the
  first two; the third is external instrumentation.
- **Whether the compact/summarization path stays off the critical path.** §11.3. Currently an
  assumption ([[boson-compact-session]] says it runs off-turn); §12's mechanism turns it into
  something checkable. Check it before ch-13 rather than assuming it.
- **[[ch-08/read]] §7.7's stray-fragment race.** Parked there with the note *"run it before ch-11
  builds the observer plane, because the observer plane is where you would watch for it."* The plane
  now exists: a `DebugLogObserver(frame_types={TTSTextFrame: (BaseOutputTransport,
  FrameEndpoint.SOURCE)})` — the exact filter the repo's own example uses
  (`observability-observer.py:152-157`) — is the instrument for it. One run, one day of logs.
