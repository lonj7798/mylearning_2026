---
title: "When Does the User's Turn End? VAD, Streaming STT, and the Turn-Strategy Chain"
chapter: ch-06
phase: voice-io
course: pipecat
sources:
  - vad-silero
  - stt-service-interface
  - stt-korean-providers
  - endpointing-turn-boundary
  - transport-telephony
  - rtv-vad-chunking
deps:
  - ch-03
  - ch-05
figure: figures/turn-boundary.html
---

# Chapter 6 — When Does the User's Turn End?

> **Scope, stated up front and enforced for the whole chapter.** Same rule as [[ch-03/read]]: this
> chapter is **mechanism and evidence only**. Where Pipecat and realtime_voice both implement
> something, this chapter states what each one *does* and stops. It casts no vote, uses no
> comparative adjective, and reaches no verdict. Scoring is [[ch-13/read]]'s job.
>
> Two things this chapter *does* do, because they are statements of fact rather than preference:
> it names the product situation in which each behaviour is a liability — symmetrically, for both
> designs — and it corrects the source excerpts where the source code disagrees with them. Three
> such corrections appear below, flagged as **SOURCE CORRECTION**.

---

## 왜 이 챕터인가

One question organises this entire chapter:

> **When is the user's turn over?**

Everything below is an answer to it, at a different layer, and the layers are taught in dependency
order. That ordering is not stylistic. The transcript layer's timers are denominated in a VAD
quantity; the strategy layer's safety net is denominated in an STT quantity. Learn them out of
order and every number you meet is a magic constant.

- **Part one — the audio-level answer.** Silero says "there is voice in this 32 ms chunk." A
  four-state hysteresis machine turns a sequence of those into "the user is speaking." That is
  §1–§5.
- **Part two — the transcript-level answer.** A streaming STT says "here is what they said, and I
  have nothing more to send." That is §6–§12.
- **Part three — the arbiter.** Neither of the above declares a turn. A chain of pluggable
  strategies consumes both and negotiates a verdict, and any link in the chain can veto. That is
  §13–§18.

Nothing later in this course may be denominated in a VAD quantity that has not been taught by the
end of this chapter. [[ch-08/read]] spends `start_secs` on barge-in latency; [[ch-11/read]] spends
`stop_secs` and the P99 table on the latency budget. Both cite this chapter rather than re-deriving.

**What you already have, and what this chapter will not repeat.** [[ch-03/read]] §4 already put
`VADParams`, `VADState`, the `_run_analyzer` state machine and `num_frames_required()` on the page,
and already ran the two-frame blip through both machines. [[ch-03/read]] §5 already put the
`STTService` / `SegmentedSTTService` docstrings side by side. [[ch-05/read]] established that a
Pipecat phone call is a WebSocket transport plus a `FrameSerializer`, and that five of the six
telephony serializers sit at 8 kHz μ-law on the wire. This chapter does not re-teach any of that.
It goes *underneath* it: the three-way split ch-03 did not name, the controller ch-03 never showed,
the arithmetic denominated in both sample rates, and the entire third layer — the strategy chain —
that ch-03 could only gesture at.

---

## 0. The shape of the answer, before any code

Here is the thing that surprises people who have built one of these before, and it is worth having
in your head before you read a single line:

**Pipecat has no component whose job is "decide the turn ended."** There is no `endpointing.py`.
There is no `EndpointingConfig`. There is no class named `TurnDetector`. I checked:

```bash
# run in wiki/raw-data/pipecat/pipecat-src at commit 0cbf9c5b
$ find src -name "endpointing*"
$ grep -rn "EndpointingConfig" src/
$
```

Both return nothing. The word "endpointing" survives in this tree only as a *provider* setting —
Deepgram's `endpointing` parameter, Speechmatics' `TurnDetectionMode`, Sarvam's
`endpointing: Literal["vad", "manual"]`. The framework itself refuses to own the concept.

Instead the decision is decomposed across three layers that are deliberately kept ignorant of each
other:

```
  audio ──► SileroOnnxModel          "0.87 confidence on this 512-sample chunk"
              │
              ▼
            VADAnalyzer              "QUIET → STARTING → SPEAKING → STOPPING → QUIET"
              │
              ▼
            VADController            fires on_speech_started / on_speech_stopped
              │                      (edge-triggered; STARTING and STOPPING never escape)
              ▼
            VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame
              │
              │       ┌──── TranscriptionFrame / InterimTranscriptionFrame  (from STT)
              │       │
              ▼       ▼
            UserTurnController       start strategies, then stop strategies, in list order
              │                      any link may return STOP and break the chain
              ▼
            UserStartedSpeakingFrame / UserStoppedSpeakingFrame   ← the actual turn
```

Read the two frame families in that diagram carefully, because the naming is the whole design and
it is easy to skim past:

| Frame | Meaning | Who emits |
|---|---|---|
| `VADUserStartedSpeakingFrame` | *there is voice* | `VADController`, via the analyzer |
| `UserStartedSpeakingFrame` | *this is a turn* | `UserTurnProcessor`, via a start strategy |

Silero decides the first. A **strategy** decides the second. They are different words for different
propositions and Pipecat gives them different types. Every confusion in this layer comes from
collapsing them.

---

# PART ONE — THE AUDIO-LEVEL ANSWER: VAD

## 1. The three-way split

[[vad-silero]] names the split, and the source bears it out: VAD in Pipecat is three objects in
three files, and each one is replaceable without touching the others.

| Object | File | Line | Job |
|---|---|---|---|
| `SileroOnnxModel` | `audio/vad/silero.py` | 34 | per-chunk confidence, ONNX inference |
| `SileroVADAnalyzer(VADAnalyzer)` | `audio/vad/silero.py` | 130 | model adapter: chunk size, int16→float32, state reset |
| `VADAnalyzer(ABC)` | `audio/vad/vad_analyzer.py` | 63 | the hysteresis machine — four states, two counters |
| `VADController(BaseObject)` | `audio/vad/vad_controller.py` | 31 | edge-triggered event emitter |

`vad_analyzer.py` is 255 lines, `silero.py` is 226, `vad_controller.py` is 244. That is the entire
VAD surface of the framework: 725 lines.

### 1.1 The model — what it actually is

`SileroOnnxModel` is a thin ONNX Runtime wrapper, and two of its constructor decisions matter for a
telephony host:

**`src/pipecat/audio/vad/silero.py` L42–61**

```python
    def __init__(self, path, force_onnx_cpu=True):
        """Initialize the Silero ONNX model.

        Args:
            path: Path to the ONNX model file.
            force_onnx_cpu: Whether to force CPU execution provider.
        """
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]
```

Both thread counts are pinned to 1 and CPU is forced by default. That is a deliberate concurrency
decision, and it is the right one to notice given [[ch-04/read]] §13's topology: one
`PipelineWorker` per call, every session an `asyncio.Task` on one loop. If the VAD model spun up its
own thread pool per session you would have N × cores threads at 40 concurrent calls. It does not.
Each analyzer gets exactly one worker thread, and the model inside it is single-threaded:

**`src/pipecat/audio/vad/vad_analyzer.py` L90–92**

```python
        # Thread executor that will run the model. We only need one thread per
        # analyzer because one analyzer just handles one audio stream.
        self._executor = ThreadPoolExecutor(max_workers=1)
```

`self.sample_rates = [8000, 16000]` on the last line of `__init__` is the fact that [[ch-05/read]]'s
telephony chapter has been waiting for. Hold it; §2.3 spends it.

The second model-level decision is a forced amnesia:

**`src/pipecat/audio/vad/silero.py` L208–226**

```python
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            # Divide by 32768 because we have signed 16-bit data.
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            new_confidence = self._model(audio_float32, self.sample_rate)[0]

            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.time()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= _MODEL_RESET_STATES_TIME:
                self._model.reset_states()
                self._last_reset_time = curr_time

            return new_confidence
        except Exception as e:
            # This comes from an empty audio array
            logger.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0
```

`_MODEL_RESET_STATES_TIME = 5.0` (L23). Every five seconds the recurrent state is zeroed. And the
`except` clause returns `0` — the lowest possible confidence — on *any* exception. A malformed
buffer does not crash the call; it reads as silence. For a 20-minute insurance sales call that is a
failure mode you should know the shape of: if something upstream starts feeding the analyzer bad
buffers, the symptom is not an exception in your logs, it is a bot that stops hearing the customer.

### 1.2 The analyzer — the two things ch-03 did not say

[[ch-03/read]] §4.1 already quoted `VADParams`, `VADState`, and the `_run_analyzer` match block. I
am not re-quoting them. Two properties of that machine went unstated there and both matter here.

**First: the gate is a per-chunk AND, and only one half of it is a *model* output.**

**`src/pipecat/audio/vad/vad_analyzer.py` L206–211**

```python
            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume
```

`confidence` comes from Silero. `volume` does not — it comes from an `AudioVolumeTracker` put
through exponential smoothing with a fixed factor:

**`src/pipecat/audio/vad/vad_analyzer.py` L173–176**

```python
    def _get_smoothed_volume(self, audio: bytes) -> float:
        """Calculate smoothed audio volume using exponential smoothing."""
        self._volume_tracker.update(audio, self.sample_rate)
        return exp_smoothing(self._volume_tracker.volume, self._prev_volume, self._smoothing_factor)
```

with `self._smoothing_factor = 0.2` (L87). So `min_volume = 0.6` is a threshold on a *smoothed,
stateful* signal, not on the instantaneous RMS of the current chunk. A single loud transient does
not clear it; sustained energy does. That is what makes the AND a noise gate rather than a second
threshold on the same evidence.

This is also the reason [[vad-silero]]'s guideline says to tune barge-in with the timing params and
not with `confidence`: `confidence` and `min_volume` are both *per-chunk* gates over one 32 ms
window, and no per-chunk gate can distinguish a cough from a syllable. Only the timing params —
which count *consecutive* chunks — buy you that discrimination.

**Second: `set_params` resets the state machine but deliberately does not reset the volume tracker.**

**`src/pipecat/audio/vad/vad_analyzer.py` L166–171**

```python
        # VAD state resets, but volume state doesn't: the rolling window and its
        # smoothing follow the audio stream, which is continuous across
        # parameter changes.
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET
```

The comment states the invariant precisely: parameters are a property of the *policy*, the smoothed
volume is a property of the *stream*, and the stream did not stop just because you sent a
`VADParamsUpdateFrame`. If you ever build a mid-call VAD retune for Lina — say, loosening the gate
after detecting a noisy caller — this is the line that tells you the retune will not blank your
noise floor estimate.

### 1.3 The controller — the object ch-03 never showed

`VADController` is where the state machine becomes events, and it does one thing that changes what
the rest of the pipeline can even observe.

**`src/pipecat/audio/vad/vad_controller.py` L176–190**

```python
    async def _handle_vad(self, audio: bytes, vad_state: VADState) -> VADState:
        """Handle Voice Activity Detection results and trigger appropriate events."""
        new_vad_state = await self._vad_analyzer.analyze_audio(audio)
        if (
            new_vad_state != vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            if new_vad_state == VADState.SPEAKING:
                await self._call_event_handler("on_speech_started")
            elif new_vad_state == VADState.QUIET:
                await self._call_event_handler("on_speech_stopped")

            vad_state = new_vad_state
        return vad_state
```

Three conditions, ANDed. The first is edge-detection: only *transitions* fire. The second and third
are the interesting ones — `STARTING` and `STOPPING` are explicitly filtered out.

**`STARTING` and `STOPPING` are not observable outside the analyzer.** No frame carries them. No
event fires on them. They exist purely as internal hypothesis-holding states. From the pipeline's
point of view, the four-state machine has exactly two *observable* states, and the extra two are the
private machinery that decides which of the two you get told about.

That is the precise sense in which the four-state machine "discards a false start silently": there
is no filtering step downstream, because there was never a downstream signal to filter.

All five events are registered `sync=True`:

**`src/pipecat/audio/vad/vad_controller.py` L106–110**

```python
        self._register_event_handler("on_speech_started", sync=True)
        self._register_event_handler("on_speech_stopped", sync=True)
        self._register_event_handler("on_speech_activity", sync=True)
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
```

Per [[ch-04/read]] §2, `sync=True` means the handler runs inline rather than being scheduled as a
background task. Speech-start is a barge-in trigger; scheduling it behind the loop's task queue
would add exactly the kind of latency [[ch-04/read]] §5.1's `N/r` arithmetic warns about. This is
that lesson being applied by the framework itself.

### 1.4 `audio_idle_timeout` — the watchdog for audio that stops arriving

**`src/pipecat/audio/vad/vad_controller.py` L192–213**

```python
    async def _audio_idle_handler(self):
        """Monitor for an idle audio stream while in SPEAKING state.

        When no audio frames arrive for `audio_idle_timeout` seconds
        (e.g. user mutes mic mid-speech), forces a transition to QUIET and
        emits `on_speech_stopped`.
        """
        while True:
            deadline = self._last_audio_time + self._audio_idle_timeout
            remaining = deadline - time.monotonic()
            if remaining > 0:
                # Audio is still recent; sleep only for the remaining window.
                await asyncio.sleep(remaining)
                continue

            if self._vad_state == VADState.SPEAKING:
                logger.warning(f"{self}: no audio received while speaking, forcing speech stop")
                self._vad_state = VADState.QUIET
                await self._call_event_handler("on_speech_stopped")

            # Wait for the next potential idle window.
            await asyncio.sleep(self._audio_idle_timeout)
```

Default `audio_idle_timeout: float = 1.0` (L75). The failure mode it covers is structural, not
exotic: the hysteresis machine only advances when a chunk arrives. If the audio stops *while the
state is `SPEAKING`*, there is no chunk to carry the state to `STOPPING`, so the turn never ends and
the customer waits forever. Note the state it forces is `QUIET`, bypassing `STOPPING` entirely —
which is exactly what makes the `on_speech_stopped` event fire past the L179–183 filter.

For Lina this is not a mic-mute scenario, it is a carrier scenario. A Twilio media stream that stops
delivering `media` events mid-utterance — network blip, carrier hiccup, one-way audio — leaves the
analyzer pinned in `SPEAKING`. One second later the watchdog closes the turn and the pipeline moves
on. Without it, the call is silently dead but still billing.

---

## 2. The arithmetic, done by hand at both sample rates

This is the section the rest of the course is denominated in. Do it by hand. It is four lines of
division and it is easy to get wrong — the excerpts got it wrong, and so did the outline, twice.

### 2.1 The conversion

**`src/pipecat/audio/vad/vad_analyzer.py` L151–171**

```python
    def set_params(self, params: VADParams):
        """Set VAD parameters and recalculate internal values.

        Args:
            params: VAD parameters for detection configuration.
        """
        logger.debug(f"Setting VAD params to: {params}")
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        # VAD state resets, but volume state doesn't: the rolling window and its
        # smoothing follow the audio stream, which is continuous across
        # parameter changes.
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET
```

Four line numbers to memorise, because [[ch-08/read]] and [[ch-11/read]] both cite them:

- `def set_params` — **L151**
- `vad_frames_per_sec = self._vad_frames / self.sample_rate` — **L162**
- `self._vad_start_frames = round(...)` — **L164**
- `self._vad_stop_frames = round(...)` — **L165**

`vad_frames_per_sec` is a misleading name. It is **seconds per chunk**, not chunks per second.
[[ch-03/read]] §4.2 already flagged this; it is worth flagging twice because the name will mislead
you every time you re-read the file.

### 2.2 At 16 kHz

`num_frames_required()` returns 512 (`silero.py` L191–197, already quoted in [[ch-03/read]] §4.1).

```
vad_frames_per_sec = 512 / 16000
                   = 0.032                       seconds per chunk

_vad_start_frames  = round(start_secs / 0.032)
                   = round(0.2 / 0.032)
                   = round(6.25)
                   = 6                           consecutive chunks

_vad_stop_frames   = round(stop_secs / 0.032)
                   = round(0.2 / 0.032)
                   = round(6.25)
                   = 6                           consecutive chunks
```

**6, not 7.** Python's `round()` is banker's rounding — it rounds half to even — and 6.25 is not a
half case anyway; `round(6.25)` is 6 because 6.25 is nearer to 6 than to 7. Verify it yourself:

```bash
$ python3 -c "print(round(0.2 / (512/16000)))"
6
```

> **SOURCE CORRECTION #1.** [[vad-silero]] states *"`start_secs=0.2` → **7 consecutive chunks**
> (`round(6.25)`)"*, and [[rtv-vad-chunking]]'s comparison table repeats **7** in two rows. Both are
> wrong. The code at `vad_analyzer.py` L164 computes 6. [[ch-03/read]] §4.2 already made this
> correction; I am restating it because every downstream number in this chapter and in
> [[ch-11/read]] hangs off it, and a 32 ms error compounds through a latency budget.

So the real quantities, which you should now be able to state without looking anything up:

| Quantity | Value at 16 kHz |
|---|---|
| chunk duration | 32 ms |
| chunks to confirm speech start | 6 |
| **onset detection lag** | **192 ms** |
| chunks to confirm speech stop | 6 |
| **offset detection lag** | **192 ms** |

192 ms, not 200 ms. The rounding loses you 8 ms on each side. Nobody will ever notice, but you
should know the number you are actually running is not the number in the config.

### 2.3 At 8 kHz — the fact Lina TMR runs on

`num_frames_required()` returns 256 at 8 kHz. Redo the division:

```
vad_frames_per_sec = 256 / 8000
                   = 0.032                       seconds per chunk   ← IDENTICAL

_vad_start_frames  = round(0.2 / 0.032) = 6      ← IDENTICAL
_vad_stop_frames   = round(0.2 / 0.032) = 6      ← IDENTICAL
```

```bash
$ python3 -c "print(256/8000, round(0.2/(256/8000)))"
0.032 6
```

**The chunk count is the same, and so is the wall-clock lag.** Silero's 8 kHz chunk is half the
samples of its 16 kHz chunk, and 8 kHz is half the rate, so the two cancel exactly. 512/16000 and
256/8000 are both 0.032.

This is the payoff of the `self.sample_rates = [8000, 16000]` line in §1.1. [[ch-05/read]]
established that five of the six telephony serializers put 8 kHz μ-law on the wire —
`twilio_sample_rate: int = 8000`, `telnyx_sample_rate: int = 8000`,
`plivo_sample_rate: int = 8000`, `exotel_sample_rate: int = 8000`,
`genesys_sample_rate: int = 8000`, with `VonageFrameSerializer` the outlier at
`vonage_sample_rate: int = 16000` ([[transport-telephony]], all six verified in the serializer
sources). So the concrete consequence for you:

> **Lina TMR's VAD tuning does not change when you move from a browser demo to a phone call.**
> `start_secs=0.2` and `stop_secs=0.2` mean the same six chunks and the same 192 ms at both rates.
> There is no telephony retune to budget for in this layer.

That is a genuinely narrow claim and I want to fence it precisely, because the fence is where the
work is. What transfers is the **timing arithmetic**. What does not transfer is the **acoustics**:
[[transport-telephony]] is explicit that 8 kHz μ-law has a 4 kHz Nyquist ceiling and 8-bit
companding, so Silero's *confidence values* on telephony audio are a different distribution than on
studio audio, and `confidence = 0.7` / `min_volume = 0.6` are thresholds on that distribution. The
frame counts need no retune. The thresholds are an open question and nothing in this repository
answers it. §13 records that as a measurement you owe yourself, not as a fact.

The guard that enforces the two-rate world:

**`src/pipecat/audio/vad/silero.py` L175–189**

```python
    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000 Hz).

        Raises:
            ValueError: If sample rate is not 8000 or 16000 Hz.
        """
        if sample_rate != 16000 and sample_rate != 8000:
            raise ValueError(
                f"Silero VAD sample rate needs to be 16000 or 8000 (sample rate: {sample_rate})"
            )

        super().set_sample_rate(sample_rate)
```

→ **Open [`figures/turn-boundary.html`](figures/turn-boundary.html) and use panel one now.** The
frame-count arithmetic is shown there as a live calculation rather than an answer — the L162
division, the L164 `round()`, and a sample-rate switch that recomputes 256/8000 to the same 6. Do
one thing with it before moving on: drag `start_secs` and find the value where the chunk count
changes. It is not where you expect, because `round()` steps at the half-chunk boundary — 0.208 s
still gives 6, 0.209 s gives 7. That step function is why `start_secs` does not behave like a
continuous latency dial.

---

## 3. What VAD emits is not a turn

**`src/pipecat/frames/frames.py` L1226–1237**

```python
class VADUserStartedSpeakingFrame(SystemFrame):
    """Frame emitted when VAD definitively detects user started speaking.

    Parameters:
        start_secs: The VAD start_secs duration that was used to confirm the user
            started speaking. This represents the speech duration that had to
            elapse before the VAD determined speech began.
        timestamp: Wall-clock time when the VAD made its determination.
    """

    start_secs: float = 0.0
    timestamp: float = field(default_factory=time.time)
```

**`src/pipecat/frames/frames.py` L1241–1252**

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

Two design decisions are visible here, and both get spent later in this chapter.

**The frame carries the parameter that produced it.** `stop_secs` is not read from config by
downstream consumers; it rides on the frame. That is what lets a consumer reconstruct *when the user
actually stopped talking* rather than when the VAD noticed — §9 and §19 both do exactly that
subtraction, and they can only do it because the frame is self-describing.

**Both are `SystemFrame`s.** Per [[ch-04/read]] §4.1, system frames travel the priority path and are
processed inline on the input task rather than queued behind data on the cancellable process task.
A turn signal that queued behind buffered audio would arrive after the audio it was supposed to
describe. It does not.

The conversion from "voice" to "turn" is one `isinstance` in a 35-line file:

**`src/pipecat/turns/user_start/vad_user_turn_start_strategy.py` L22–35**

```python
    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to detect user turn start.

        Args:
            frame: The frame to be analyzed.

        Returns:
            STOP if the user started speaking, CONTINUE otherwise.
        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        return ProcessFrameResult.CONTINUE
```

That is the entire `VADUserTurnStartStrategy`. Its smallness is the point: the audio layer's job
ends at `VADUserStartedSpeakingFrame`, and a *policy object* — replaceable, stackable, vetoable —
turns it into a turn. Swap this file for `MinWordsUserTurnStartStrategy` and the VAD does not know
or care.

---

## 4. Where the analyzer mounts, and why it is not where you would guess

This is the single most surprising wiring fact in the voice-I/O phase, and it is a direct
consequence of a removal that [[ch-05/read]] already recorded.

```bash
$ grep -n "vad_analyzer\|vad_enabled\|turn_analyzer" src/pipecat/transports/base_transport.py
$
```

Nothing. `TransportParams` has **no `vad_analyzer` field and no `vad_enabled` field**. If you have
read any Pipecat tutorial written before this commit, every one of them passes
`TransportParams(vad_enabled=True, vad_analyzer=SileroVADAnalyzer())`. That API is gone:

**`CHANGELOG.md` L4402–4406**

```
- ⚠️ Removed `vad_analyzer` and `turn_analyzer` parameters from
  `TransportParams` and all transport input classes, along with all deprecated
  VAD/turn analysis logic in `BaseInputTransport`. VAD and turn detection are
  now handled entirely by `LLMUserAggregator`.
  (PR [#4229](https://github.com/pipecat-ai/pipecat/pull/4229))
```

There are now exactly two mount points.

**Mount point 1 — the aggregator params.** This is the canonical one:

**`src/pipecat/processors/aggregators/llm_response_universal.py` L169–177**

```python
    add_tool_change_messages: bool = False
    audio_idle_timeout: float = 1.0
    user_turn_strategies: UserTurnStrategies | None = None
    user_mute_strategies: list[BaseUserMuteStrategy] = field(default_factory=list)
    user_turn_stop_timeout: float = 5.0
    user_idle_timeout: float = 0
    vad_analyzer: VADAnalyzer | None = None
    filter_incomplete_user_turns: bool = False
    user_turn_completion_config: UserTurnCompletionConfig | None = None
```

`@dataclass class LLMUserAggregatorParams` is declared at **L119–120**. Every knob this chapter
teaches is on that one dataclass: the analyzer, the strategies, the mute strategies, the stop
watchdog, the idle timeout, and the controller's `audio_idle_timeout` from §1.4. The aggregator
constructs the `VADController` itself:

**`src/pipecat/processors/aggregators/llm_response_universal.py` L748–761**

```python
        # VAD controller
        self._vad_controller: VADController | None = None
        if self._params.vad_analyzer:
            self._vad_controller = VADController(
                self._params.vad_analyzer,
                audio_idle_timeout=self._params.audio_idle_timeout,
            )
            self._vad_controller.add_event_handler("on_speech_started", self._on_vad_speech_started)
            self._vad_controller.add_event_handler("on_speech_stopped", self._on_vad_speech_stopped)
            self._vad_controller.add_event_handler(
                "on_speech_activity", self._on_vad_speech_activity
            )
            self._vad_controller.add_event_handler("on_push_frame", self._on_push_frame)
            self._vad_controller.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
```

Note the conditional at L750: **no analyzer, no controller, no VAD frames at all.** Forgetting
`vad_analyzer=` does not produce an error or a warning. It produces a bot that never detects speech
onset and relies entirely on `TranscriptionUserTurnStartStrategy` — which is a *supported*
configuration (§17), just not the one you thought you had configured. That is the single most
likely misconfiguration in this layer and it fails silently.

The canonical wiring, verbatim:

**`examples/getting-started/06-voice-agent.py` L75–91**

```python
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )
```

**Mount point 2 — the standalone processor.** For a pipeline with no LLM aggregator:

**`src/pipecat/processors/audio/vad_processor.py` L26–48**

```python
class VADProcessor(FrameProcessor):
    """Processes audio frames through voice activity detection.

    This processor wraps a VADController to detect speech in audio streams
    and push VAD frames into the pipeline:

    - ``VADUserStartedSpeakingFrame``: Pushed when speech begins.
    - ``VADUserStoppedSpeakingFrame``: Pushed when speech ends.
    - ``UserSpeakingFrame``: Pushed periodically while speech is detected.

    Example::

        vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    """

    def __init__(
        self,
        *,
        vad_analyzer: VADAnalyzer,
        speech_activity_period: float = 0.2,
        audio_idle_timeout: float = 1.0,
        **kwargs,
    ):
```

Same `VADController`, same defaults, different host. Transcription-only bots use this.

**Why the aggregator and not the transport?** State the reason mechanically rather than as a
judgement. The turn boundary is a decision that needs *both* audio evidence and transcript
evidence — §17's default start list has one strategy for each. The transport sees only audio. The
aggregator sits downstream of the STT (position 3 in the canonical chain, right after `stt` at
position 2), so it is the first processor in the pipeline that sees both. Putting the decision where
both inputs are available is the structural reason the parameter moved.

---

## 5. Mechanism differential against realtime_voice

[[ch-03/read]] §4.2 already built the parameter-for-parameter table and §4.4 already traced the
two-frame blip through both machines. I am not repeating either. What follows is the part ch-03
could not cover, because it needed §1 and §4 first: the *structural* differential.

**This is a differential, not a ranking.** No comparative adjective appears below and no
recommendation is made. [[ch-13/read]] is the only place either design is scored.

### 5.1 What each design has, structurally

| Structural element | Pipecat | realtime_voice |
|---|---|---|
| model / analyzer / controller as three replaceable objects | yes — `silero.py:34`, `vad_analyzer.py:63`, `vad_controller.py:31` | no — `SileroVAD` is model + state machine in one class ([[rtv-vad-chunking]]) |
| observable states | 2 (`SPEAKING`, `QUIET`); `STARTING`/`STOPPING` filtered at `vad_controller.py:179-183` | 2 (`self._speaking` bool) |
| internal hypothesis states | 2 (`STARTING`, `STOPPING`) | 0 |
| gate | AND of Silero confidence ≥ `0.7` and smoothed volume ≥ `0.6` (`vad_analyzer.py:211`) | Silero confidence ≥ `threshold = 0.5`; no volume term ([[rtv-vad-chunking]]) |
| onset/offset unit | seconds (`start_secs` / `stop_secs`), converted to chunks at `vad_analyzer.py:164-165` | frames (`min_speech_frames = 2`, `min_silence_frames = 6`) |
| analysis chunk size | fixed by the analyzer — `num_frames_required()` | whatever the transport delivered |
| 8 kHz path | `num_frames_required() -> 256`; `sample_rates = [8000, 16000]` | none — `ValueError("SileroVAD requires 16 kHz mono PCM")` at `vad/silero.py:58` ([[rtv-vad-chunking]]) |
| idle-audio watchdog | `VADController(audio_idle_timeout=1.0)`, `vad_controller.py:192` | absent ([[rtv-vad-chunking]]) |
| self-measured latency | none in this layer | `endpoint_latency_ms = self._silence_frames * frame.duration_seconds * 1000` (`energy.py:79`, `silero.py:89`) |
| pre-speech replay buffer | not exposed here — lives inside `SegmentedSTTService` (§7) | `VoiceSessionConfig.vad_prefix_frames = 5`, replayed into ASR on `SPEECH_STARTED` |

Two rows deserve their names said out loud rather than left in a table cell.

**The hypothesis-state row.** A two-state machine has no representation for "maybe speech." It must
commit on every chunk. A four-state machine holds the hypothesis in `STARTING` and can retract it,
and because the controller filters `STARTING`, the retraction costs nothing downstream. Fed the same
two-frame blip: Pipecat's machine goes `QUIET → STARTING → QUIET` and emits nothing;
realtime_voice's flips `self._speaking` to `True` and emits `SPEECH_STARTED`, which
`VoiceSession._on_speech_started` turns into a generation advance and an assistant cancellation
([[rtv-vad-chunking]], [[rtv-pipeline-session]]). Both machines are doing what they were written to
do.

**The self-measured-latency row.** realtime_voice computes and reports its own endpoint latency at
the VAD layer. Pipecat does not; that number surfaces through the observer plane instead
([[ch-11/read]]). This is one place where realtime_voice has a mechanism Pipecat has no counterpart
for at this layer, and it belongs in the differential for exactly that reason.

### 5.2 The product situation each behaviour is a liability in

Stated symmetrically. These are situations, not scores.

**Pipecat's behaviour is a liability when the product needs sub-192 ms floor-yield.** Six chunks of
confirmed speech is 192 ms of onset lag before anything downstream learns the customer opened their
mouth, and it is 192 ms that a machine with `min_speech_frames = 2` at 20 ms frames (40 ms) does not
spend. In a product where the assistant's monologue must stop the instant the customer starts —
a rapid-fire objection-handling exchange — that difference is audible, and the four-state machine
has no configuration that removes it without also removing the false-start rejection, because they
are the same counter.

**realtime_voice's behaviour is a liability when the input has sustained non-speech energy.** With
no `min_volume` conjunct and a 0.5 confidence threshold, two frames over threshold produce a real
`SPEECH_STARTED`, and per [[rtv-vad-chunking]] that immediately advances the generation and cancels
the assistant. In a tele-sales call the sustained-energy sources are not hypothetical: hold music,
a TV in the customer's room, a second person talking nearby, and Korean backchannels ("네", "아",
"음") that the customer emits *while listening* and does not intend as a floor claim. Each one costs
a cancelled assistant turn.

**And a third, which belongs to neither design.** Neither system filters backchannels at the VAD
layer. Pipecat has no "ignore 네/응" mechanism in `audio/vad/`; boson's lives in the Gateway as
`WordFilterPolicy` ([[boson-interrupt-subsystem]]). That semantic gate is a `BaseUserTurnStartStrategy`
subclass you write yourself — §20 makes it a probe.

**One asymmetry that is a fact rather than a trade-off:** realtime_voice has no 8 kHz path at all.
`ValueError("SileroVAD requires 16 kHz mono PCM")` at `vad/silero.py:58` is a hard raise, not a
degraded mode. Telephony audio arriving at 8 kHz must be resampled to 16 kHz before the VAD sees it,
or the call fails. Pipecat's analyzer accepts 8 kHz natively. That is a capability difference, not a
tuning difference, and §2.3 quantified what it buys: nothing changes in the timing.

→ **Panel one of [`figures/turn-boundary.html`](figures/turn-boundary.html) has a blip injector.**
Run the same two-frame input through both machines and watch where the second one's `SPEECH_STARTED`
lands. The panel reports both as behaviour with no badge and no winner; that is deliberate.

---

# PART TWO — THE TRANSCRIPT-LEVEL ANSWER: STREAMING STT

## 6. Three base classes and exactly one abstract method

`src/pipecat/services/stt_service.py` is **1,040 lines**.

> **SOURCE CORRECTION #2.** [[stt-service-interface]] says 1,041 L and 69 L for `stt_latency.py`;
> `wc -l` at this commit gives **1,040** and **68**. Off-by-one, harmless, but it tells you the
> excerpt was written against a slightly different read and you should trust the file.

Three classes, at three line numbers you will want:

| Class | Line | Shape |
|---|---|---|
| `class STTService(AIService)` | 51 | continuous streaming — every chunk goes to the provider |
| `class SegmentedSTTService(STTService)` | 797 | request/response — one call per utterance |
| `class WebsocketSTTService(STTService, WebsocketService)` | 929 | streaming over a socket, with reconnect |

And one abstract method for all of them:

**`src/pipecat/services/stt_service.py` L334–335**

```python
    @abstractmethod
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
```

Note the line numbers precisely: **L334 is the `@abstractmethod` decorator, L335 is the signature.**
The outline flagged this because it is the kind of off-by-one that survives three review passes.

The return type is the whole interface. `AsyncGenerator[Frame | None, None]` — a provider yields
*zero or more frames per audio chunk*, and `None` is a legal yield. A streaming provider yields
interims as they arrive and a final when the utterance closes; a segmented provider yields exactly
one frame per call; a provider with nothing to say yields nothing. One signature, both shapes, and
the difference is entirely in the yield pattern.

The constructor carries the latency contract:

**`src/pipecat/services/stt_service.py` L94–105**

```python
    def __init__(
        self,
        *,
        audio_passthrough=True,
        sample_rate: int | None = None,
        stt_ttfb_timeout: float = 2.0,
        ttfs_p99_latency: float | None = None,
        keepalive_timeout: float | None = None,
        keepalive_interval: float = 5.0,
        settings: STTSettings | None = None,
        **kwargs,
    ):
```

`ttfs_p99_latency` is the field §11's table populates and §18/§19's timers consume. It is a
*per-deployment* number that the class takes as a constructor argument, with the benchmark constant
as the default. Hold that; it is the design decision that makes the whole third part work.

---

## 7. Streaming versus segmented, mechanically

### 7.1 How audio reaches the service

**`src/pipecat/services/stt_service.py` L474–480**

```python
        if isinstance(frame, AudioRawFrame):
            # In this service we accumulate audio internally and at the end we
            # push a TextFrame. We also push audio downstream in case someone
            # else needs it.
            await self.process_audio_frame(frame, direction)
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
```

`audio_passthrough=True` by default: the STT consumes the audio *and* forwards it. This matters
because the turn analyzer in §19 is downstream of the STT and needs the same audio the STT just ate.
Set `audio_passthrough=False` and you silently starve `TurnAnalyzerUserTurnStopStrategy`.

Mute is checked before the provider ever sees a byte:

**`src/pipecat/services/stt_service.py` L437–438**

```python
        if self._muted:
            return
```

That is two lines up from `_last_audio_time` and above `process_generator(self.run_stt(...))`. Mute
in Pipecat means *the provider is not billed and produces no transcript*, not *the transcript is
discarded*. §16's `user_mute_strategies` are what drive it.

### 7.2 Streaming — the default path

`STTService.process_audio_frame` ends at:

**`src/pipecat/services/stt_service.py` L461–463**

```python
        self._record_stt_audio_usage(frame.audio)

        await self.process_generator(self.run_stt(frame.audio))
```

Every audio frame — roughly every 20 ms of wire audio — is handed straight to `run_stt`, which for a
websocket service means "write to the socket." Transcription overlaps the customer's own speech.
By the time VAD fires `VADUserStoppedSpeakingFrame`, the provider has already computed most of the
utterance and needs only to close it out. That is the entire latency argument for streaming, and
§11's table is a measurement of how well each provider closes it out.

### 7.3 Segmented — buffer, trim, wrap, one call

**`src/pipecat/services/stt_service.py` L901–926**

```python
    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Process audio frames by buffering them for segmented transcription.

        Continuously buffers audio, growing the buffer while user is speaking and
        maintaining a small buffer when not speaking to account for VAD delay.

        If the frame has a user_id, it is stored for later use in transcription.

        Args:
            frame: The audio frame to process.
            direction: The direction of frame processing.
        """
        # UserAudioRawFrame contains a user_id (e.g. Daily, Livekit)
        if isinstance(frame, UserAudioRawFrame):
            self._user_id = frame.user_id
        # AudioRawFrame does not have a user_id (e.g. SmallWebRTCTransport, websockets)
        else:
            self._user_id = ""

        # If the user is speaking the audio buffer will keep growing.
        self._audio_buffer += frame.audio

        # If the user is not speaking we keep just a little bit of audio.
        if not self._user_speaking and len(self._audio_buffer) > self._audio_buffer_size_1s:
            discarded = len(self._audio_buffer) - self._audio_buffer_size_1s
            self._audio_buffer = self._audio_buffer[discarded:]
```

The trim window:

**`src/pipecat/services/stt_service.py` L836–837**

```python
        await super().setup(setup)
        self._audio_buffer_size_1s = self.sample_rate * 2
```

Work the arithmetic, because it connects directly to §2:

```
16 kHz:  16000 samples/s × 2 bytes/sample = 32000 bytes = 1.000 s of pre-roll
 8 kHz:   8000 samples/s × 2 bytes/sample = 16000 bytes = 1.000 s of pre-roll
```

One second, at either rate. And §2 told you the onset detection lag is **192 ms**. So the pre-roll
covers the VAD's blind spot **5.2 times over**. The buffer is not a tuning knob — it is a fixed,
generous over-provision against a lag the analyzer already bounded. This is the Pipecat counterpart
to realtime_voice's `vad_prefix_frames = 5` deque ([[ch-03/read]] §4.5), with the difference already
stated there: one is a config field, the other is a constant inside the STT.

Then, on VAD stop, exactly one call:

**`src/pipecat/services/stt_service.py` L876–899**

```python
    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._user_speaking = False

        # A service that can no longer work can't transcribe this segment.
        if not self.is_usable:
            self._audio_buffer.clear()
            return

        # Report usage for the raw segment before transcription so tracing can
        # attach it to the STT span the resulting TranscriptionFrame closes.
        self._record_stt_audio_usage(self._audio_buffer)
        await self.emit_stt_usage_metrics()

        if self.wants_wav_segments:
            audio = pcm_to_wav(self._audio_buffer, self.sample_rate)
        else:
            # Local models read the buffer as raw 16-bit PCM; wrapping it in a
            # WAV container would make them misread the 44-byte header as audio.
            audio = bytes(self._audio_buffer)

        # Start clean.
        self._audio_buffer.clear()

        await self.process_generator(self.run_stt(audio))
```

**`SegmentedSTTService` is triggered by `VADUserStoppedSpeakingFrame`.** Read that again with §4 in
mind: no `vad_analyzer` on the aggregator params means no `VADUserStoppedSpeakingFrame`, which means
a segmented STT service that **never transcribes anything at all**. The docstring says
*"Requires VAD to be enabled in the pipeline to function properly"* (L803) and it means it literally.
That is the second silent-failure mode in this chapter, and it has the same root cause as the first.

Finally, segmented services stamp every transcript as final by construction:

**`src/pipecat/services/stt_service.py` L850–862**

```python
    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame, marking TranscriptionFrames as finalized.

        Segmented STT services process complete speech segments and return a single
        TranscriptionFrame per segment, so every transcription is inherently finalized.

        Args:
            frame: The frame to push.
            direction: The direction of frame flow in the pipeline.
        """
        if isinstance(frame, TranscriptionFrame):
            frame.finalized = True
        await super().push_frame(frame, direction)
```

`finalized = True` is what short-circuits §18's STT safety-net timer. So a segmented service
degrades gracefully in the strategy chain: it is slow to produce the transcript, but the moment it
does, the chain stops waiting. The cost lands entirely in the round trip, not in the timers.

---

## 8. The base class never emits an interim

This is small and load-bearing:

```bash
$ grep -c "InterimTranscriptionFrame" src/pipecat/services/stt_service.py
0
```

**Zero references.** The 1,040-line base class does not import, construct, or match
`InterimTranscriptionFrame` even once. Interims are entirely the provider's business:

```bash
$ grep -rl "InterimTranscriptionFrame(" $(find src/pipecat/services -name '*.py') | wc -l
25
```

25 modules construct it directly. Three of those are realtime LLM services
(`inworld/realtime/llm.py`, `openai/realtime/llm.py`, `xai/realtime/llm.py`); the other 22 are STT
modules. What the base class *does* do is intercept the final:

**`src/pipecat/services/stt_service.py` L531–544**

```python
        if isinstance(frame, TranscriptionFrame):
            # Store the transcript time for TTFB calculation
            self._last_transcript_time = time.time()

            # Set finalized from pending state and auto-reset
            if self._finalize_pending:
                frame.finalized = True
                self._finalize_pending = False

            # If this is a finalized transcription, report TTFB immediately
            if frame.finalized:
                await self.stop_ttfb_metrics()
                # Cancel the timeout since we've already reported
                await self._cancel_ttfb_timeout()
```

`push_frame` is at L519. The asymmetry — base class owns the final, providers own the interim — is
the reason `TranscriptionFrame` has a `finalized: bool` field and `InterimTranscriptionFrame` does
not ([[ch-03/read]] §5.1 cited both at `frames.py` L450 and L476). One type means "possibly not
done"; the other carries an explicit flag saying whether it is.

---

## 9. TTFB, redefined

Pipecat does not measure STT latency the way you measure LLM latency. The docstring states the
redefinition in the class body:

**`src/pipecat/services/stt_service.py` L64–70**

```
    A streaming STT reports latency through TTFB — speech end to final transcript —
    and not through processing metrics. Audio arrives continuously, so there is no
    discrete request whose duration a
    :meth:`~pipecat.processors.frame_processor.FrameProcessor.start_processing_metrics`
    window could measure; anchoring one to a speech or turn boundary measures how
    long the user talked. :class:`SegmentedSTTService` does issue a discrete
    request per utterance, so its subclasses time that call and report both.
```

The argument is airtight and worth internalising: a streaming STT has no request, so "time to first
byte of the response" is not a thing that exists. The only interval that *is* meaningful — and the
only one a voice product cares about — is **speech end → final transcript.**

And here is the subtraction that §3 set up:

**`src/pipecat/services/stt_service.py` L627–652**

```python
    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle VAD user stopped speaking frame.

        Calculates the actual speech end time and starts a timeout task to wait
        for the final transcription before reporting TTFB.

        Args:
            frame: The VAD user stopped speaking frame.
        """
        self._user_speaking = False

        # Skip TTFB measurement if stop_secs is not set
        if frame.stop_secs == 0.0:
            return

        # Calculate the actual speech end time (current time minus VAD stop delay).
        # This approximates when the last user audio was sent to the STT service,
        # which we use to measure against the eventual transcription response.
        speech_end_time = frame.timestamp - frame.stop_secs
        await self.start_ttfb_metrics(start_time=speech_end_time)

        # Start timeout task (any previous timeout was cancelled by VADUserStartedSpeakingFrame
        # or InterruptionFrame)
        self._ttfb_timeout_task = self.create_task(
            self._ttfb_timeout_handler(), name="stt_ttfb_timeout"
        )
```

`def _handle_vad_user_stopped_speaking` is at **L627**; the subtraction is at **L645**.

```
speech_end_time = frame.timestamp - frame.stop_secs
```

That line only means something now. `frame.timestamp` is when the VAD *decided*. `frame.stop_secs`
is the silence window it had to observe first. Their difference is when the customer actually
stopped talking. Without §2 you would read that as arbitrary; with §2 you can put a number on it:
at the default it rewinds the clock by 0.2 s of configured silence, which the analyzer actually
spent as 192 ms of six chunks. The metric is anchored to physical reality rather than to the
detector's reaction time — which is the only way a cross-provider latency table means anything,
because otherwise every provider's number would include your VAD config.

The 2-second backstop:

**`src/pipecat/services/stt_service.py` L666–677**

```python
        try:
            await asyncio.sleep(self._stt_ttfb_timeout)
            if self._last_transcript_time > 0:
                await self.stop_ttfb_metrics(end_time=self._last_transcript_time)
            else:
                # No transcript at all, so there is no end time to measure to.
                # Close the measurement rather than leave it open for the next
                # transcript to be measured against.
                await self.cancel_ttfb_metrics()
        except asyncio.CancelledError:
            # Task was cancelled (new utterance or interruption), which is expected behavior
            pass
```

`stt_ttfb_timeout=2.0`. Note the `else`: if no transcript arrives at all, the measurement is
*cancelled*, not left open. A dropped utterance does not corrupt the next one's number.

---

## 10. `STTMetadataFrame` — the service publishes its own latency

**`src/pipecat/frames/frames.py` L1655–1667**

```python
class STTMetadataFrame(ServiceMetadataFrame):
    """Metadata from STT service.

    Broadcast by STT services to inform downstream processors (like turn
    strategies) about STT latency characteristics.

    Parameters:
        ttfs_p99_latency: Time to final segment P99 latency in seconds.
            This is the expected time from when speech ends to when the
            final transcript is received, at the 99th percentile.
    """

    ttfs_p99_latency: float
```

**`src/pipecat/services/stt_service.py` L559–575**

```python
    def service_metadata_frame(self) -> STTMetadataFrame:
        """Build the STT metadata frame broadcast at start.

        Overrides :meth:`AIService.service_metadata_frame` to return an
        :class:`~pipecat.frames.frames.STTMetadataFrame` carrying the service's TTFS
        P99 latency. A service that does its own server-side end-of-turn detection
        overrides this (calling ``super()``) to set ``user_turn_strategies`` on the
        returned frame.
        """
        if not self.supports_ttfs:
            ttfs = 0.0
        else:
            ttfs = self._ttfs_p99_latency
            if ttfs is None:
                ttfs = DEFAULT_TTFS_P99
                logger.warning(f"{self.name}: ttfs_p99_latency not set, using default {ttfs}s")
        return STTMetadataFrame(service_name=self.name, ttfs_p99_latency=ttfs)
```

Two branches worth naming.

`supports_ttfs` (L548–557) returns `True` by default and `False` for services where the *server*
owns the turn boundary — because then "speech end → final transcript" has no separate existence to
measure. When it is `False` the frame carries `ttfs = 0.0`, and the docstring at L554–555 tells you
what downstream does with that: *"Downstream turn-stop strategies that consume `STTMetadataFrame`
treat a 0 latency as 'no extra wait.'"* Zero is not missing data. Zero is an instruction.

The `ttfs is None` branch is the escape hatch: an unmeasured service falls back to
`DEFAULT_TTFS_P99 = 1.0` **and logs a warning**. If you write a custom STT service for a Korean
provider and forget the constructor argument, you get a one-second safety net you did not choose and
a warning you will probably never read. Note it now; §21 makes it a decision.

Also read the last sentence of that docstring — *"A service that does its own server-side end-of-turn
detection overrides this (calling `super()`) to set `user_turn_strategies` on the returned frame"* —
because it is the mechanism §21 counts nine instances of.

---

## 11. The benchmark table — built once, here

`src/pipecat/services/stt_latency.py` is **68 lines** and it is the only hard benchmark table in the
repository. **Provider selection happens in this chapter and nowhere else in this course.**
[[ch-11/read]]'s latency waterfall consumes a selected value from this table; it does not re-render
the table.

The module docstring states the measurement conditions, and they are the reason this section had to
come after §2:

**`src/pipecat/services/stt_latency.py` L7–35**

```python
"""STT service latency defaults.

This module contains P99 time-to-final-segment (TTFS) latency values for STT
services. TTFS measures the time from when speech ends to when the final
transcript is received.

These values are used by turn stop strategies to optimize timing. Each STT
service publishes its latency via STTMetadataFrame at pipeline start.

All built-in values were measured with VADParams.stop_secs=0.2, the recommended
default. If you change stop_secs, re-run the benchmark with your VAD settings
and pass the measured value to your STT service constructor.

To measure latency for your specific deployment (region, network conditions,
self-hosted instances), use the STT benchmark tool:
https://github.com/pipecat-ai/stt-benchmark

Run the TTFS benchmark for your service and configuration, then pass the
measured value to your STT service constructor:

    stt = DeepgramSTTService(api_key="...", ttfs_p99_latency=0.45)

Turn-based STT services (e.g. ``CartesiaTurnsSTTService``,
``DeepgramFluxSTTService``) have no meaningful TTFS metric — the server
defines the turn boundary directly, so there is no separate "speech end →
final transcript" interval to measure. Those services override the
``STTService.supports_ttfs`` property to return False rather than supplying
a constant here.
"""
```

**Every constant in the table below was measured at `VADParams.stop_secs = 0.2`** — the value §2
derived six chunks and 192 ms from. That is not a footnote. It is the reason §18 and §19 both emit a
warning when you change `stop_secs`, and it is the reason a chapter that taught this table before
teaching VAD would be teaching a number with no denominator.

### 11.1 The complete table — all 23 measured constants

`stt_latency.py` holds **23 measured provider constants**, not the dozen usually quoted. Here is
every one, verbatim from the source, sorted by latency:

**`src/pipecat/services/stt_latency.py` L37–68**

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
ELEVENLABS_TTFS_P99: float = 2.01
ELEVENLABS_REALTIME_TTFS_P99: float = 0.41
FAL_TTFS_P99: float = 2.07
GLADIA_TTFS_P99: float = 1.49
GOOGLE_TTFS_P99: float = 1.57
GRADIUM_TTFS_P99: float = 0.62
GROQ_TTFS_P99: float = 1.54
MISTRAL_TTFS_P99: float = 1.89
OPENAI_TTFS_P99: float = 2.01
OPENAI_REALTIME_TTFS_P99: float = 1.66
SARVAM_TTFS_P99: float = 1.17
# Provisional until benchmarked against the realtime endpoint.
SARVAM_REALTIME_TTFS_P99: float = 1.00
SMALLEST_TTFS_P99: float = 1.59
SONIOX_TTFS_P99: float = 0.35
SPEECHMATICS_TTFS_P99: float = 0.74
XAI_TTFS_P99: float = 2.14
TOGETHER_TTFS_P99: float = 1.00

# These services run locally and should be replaced with measured values
NVIDIA_TTFS_P99: float = DEFAULT_TTFS_P99
WHISPER_TTFS_P99: float = DEFAULT_TTFS_P99
```

Sorted, with the Korean column §12 will fill in:

| # | Constant | P99 (s) | Verified `Language.KO`? |
|---|---|---|---|
| 1 | `DEEPGRAM_TTFS_P99` | 0.35 | passthrough — no map |
| 2 | `DEEPGRAM_SAGEMAKER_TTFS_P99` | 0.35 | passthrough — no map |
| 3 | `SONIOX_TTFS_P99` | 0.35 | **yes** — `soniox/stt.py:163` |
| 4 | `ELEVENLABS_REALTIME_TTFS_P99` | 0.41 | no — see §12.4 |
| 5 | `ASSEMBLYAI_TTFS_P99` | 0.42 | **no — documented exclusion** |
| 6 | `GRADIUM_TTFS_P99` | 0.62 | no `Language.KO` in `gradium/` |
| 7 | `SPEECHMATICS_TTFS_P99` | 0.74 | **yes** — `speechmatics/stt.py:1159` |
| 8 | `CARTESIA_TTFS_P99` | 0.81 | **no — English-only at launch** |
| 9 | `SARVAM_REALTIME_TTFS_P99` | 1.00 (provisional) | no — Indian languages |
| 10 | `TOGETHER_TTFS_P99` | 1.00 | no `Language.KO` |
| 11 | `SARVAM_TTFS_P99` | 1.17 | no — `KOK_IN` is Konkani |
| 12 | `GLADIA_TTFS_P99` | 1.49 | **yes** — `gladia/stt.py:113` |
| 13 | `GROQ_TTFS_P99` | 1.54 | no `Language.KO` |
| 14 | `GOOGLE_TTFS_P99` | 1.57 | **yes** — `google/stt.py:233` |
| 15 | `SMALLEST_TTFS_P99` | 1.59 | no `Language.KO` |
| 16 | `OPENAI_REALTIME_TTFS_P99` | 1.66 | passthrough |
| 17 | `AZURE_TTFS_P99` | 1.80 | **yes** — `azure/common.py:200` |
| 18 | `MISTRAL_TTFS_P99` | 1.89 | no `Language.KO` |
| 19 | `AWS_TRANSCRIBE_TTFS_P99` | 1.90 | **yes** — `aws/stt.py:458` |
| 20 | `ELEVENLABS_TTFS_P99` | 2.01 | **yes** — `elevenlabs/stt.py:109` (`"kor"`) |
| 21 | `OPENAI_TTFS_P99` | 2.01 | passthrough |
| 22 | `FAL_TTFS_P99` | 2.07 | **yes** — `fal/stt.py:89` |
| 23 | `XAI_TTFS_P99` | 2.14 | **yes** — `xai/stt.py:67` |
| — | `DEFAULT_TTFS_P99` | 1.00 | fallback for local models |
| — | `NVIDIA_TTFS_P99` | = `DEFAULT_TTFS_P99` | **yes** — `nvidia/stt.py:92` (unmeasured) |
| — | `WHISPER_TTFS_P99` | = `DEFAULT_TTFS_P99` | **yes** — `whisper/base_stt.py:90` (unmeasured) |

Three things to read off it.

**The spread is 6.1×.** 0.35 s to 2.14 s. That is not a tuning difference; it is a different
product. At P99, a customer talking to a bot on `XAI` waits nearly two full seconds longer for the
turn to close than one on `SONIOX`, before the LLM has produced a single token.

**`NVIDIA` and `WHISPER` are aliases, not measurements.** They are literally
`= DEFAULT_TTFS_P99` — the same conservative 1.0 the `ttfs is None` branch in §10 falls back to. The
comment says so: *"These services run locally and should be replaced with measured values."* Never
quote 1.0 as a Whisper benchmark. It is a placeholder.

**`SARVAM_REALTIME` carries its own in-source disclaimer** — `# Provisional until benchmarked
against the realtime endpoint.` The repo is telling you which of its own numbers it does not trust.

**The absent rows are as informative as the present ones.** `CartesiaTurnsSTTService` and
`DeepgramFluxSTTService` have no constant here at all, by design: per the module docstring they
override `supports_ttfs` to `False` because the server defines the turn boundary. §21 explains what
that costs and buys.

→ **Panel two of [`figures/turn-boundary.html`](figures/turn-boundary.html) renders this table as
the interactive provider selector** — sorted bar chart from 0.35 to 2.14, with the two local aliases
drawn as aliases rather than measurements. This is the only place in the course that control is
built. Select a provider there and carry the number into §18's `effective_stt_wait` arithmetic.

---

## 12. Korean, with the open unknown stated rather than guessed

`src/pipecat/transcriptions/language.py` L310–311 defines the two enum members:

```python
    KO = "ko"
    KO_KR = "ko-KR"
```

"Does Pipecat support Korean STT?" has **three different kinds of answer** depending on the service,
and only one of them is evidence ([[stt-korean-providers]]):

1. **Verified** — `Language.KO` appears as a key in the service's own hand-curated map.
2. **Passthrough** — no map at all; whatever string the enum holds goes out on the wire and the repo
   takes no position on whether the provider accepts it.
3. **Documented exclusion** — a map exists and Korean is deliberately not in it.

### 12.1 The twelve verified services

Ground truth, from a grep for `Language.KO` restricted to STT modules:

| Service | file:line | code sent | base class | P99 (s) |
|---|---|---|---|---|
| `SonioxSTTService` | `soniox/stt.py:163` | `"ko"` | `WebsocketSTTService` (`soniox/stt.py:264`) | 0.35 |
| `SpeechmaticsSTTService` | `speechmatics/stt.py:1159` | `"ko"` | `STTService` (`speechmatics/stt.py:166`) | 0.74 |
| `GladiaSTTService` | `gladia/stt.py:113` | `"ko"` | `WebsocketSTTService` (`gladia/stt.py:202`) | 1.49 |
| `GoogleSTTService` | `google/stt.py:233-234` | `"ko-KR"` | `STTService` (`google/stt.py:489`) | 1.57 |
| `AzureSTTService` | `azure/common.py:200-201` | `"ko-KR"` | `STTService` (`azure/stt.py:93`) | 1.80 |
| `AWSTranscribeSTTService` | `aws/stt.py:458-459` | `"ko-KR"` | `WebsocketSTTService` (`aws/stt.py:56`) | 1.90 |
| `NvidiaSTTService` (Riva) | `nvidia/stt.py:92-93` | `"ko-KR"` | `STTService` (`nvidia/stt.py:229`) | 1.0 (alias) |
| `ElevenLabsSTTService` | `elevenlabs/stt.py:109` | `"kor"` — three letters | `SegmentedSTTService` (`elevenlabs/stt.py:210`) | 2.01 |
| `FalSTTService` | `fal/stt.py:89` | `"ko"` | `SegmentedSTTService` (`fal/stt.py:155`) | 2.07 |
| `XAISTTService` | `xai/stt.py:67` | `"ko"` | `WebsocketSTTService` (`xai/stt.py:102`) | 2.14 |
| `WhisperSTTService` (local) | `whisper/stt.py:155`, `whisper/base_stt.py:90` | `"ko"` | `SegmentedSTTService` (`whisper/base_stt.py:123`) | 1.0 (alias) |
| `MoonshineSTTService` (local) | `moonshine/stt.py:67` | `"ko"` | `SegmentedSTTService` (`moonshine/stt.py:128`) | — |

Twelve. `ElevenLabsSTTService` sending three-letter `"kor"` rather than `"ko"` is a real quirk worth
remembering — if you ever write a config-driven service selector, a naive ISO-639-1 assumption
breaks on exactly this row.

Moonshine's docstring is the strongest in-prose statement of support in the tree:

**`src/pipecat/services/moonshine/stt.py` L122–124**

```
        language: Language for transcription. Moonshine publishes models for
            Arabic, Chinese, English, Japanese, Korean, Spanish, Ukrainian, and
            Vietnamese; regional variants resolve to their base code.
```

### 12.2 FunASR — a thirteenth, by a different mechanism

FunASR is Korean-capable and has **no `Language.KO` mapping at all.** It uses a plain string set:

**`src/pipecat/services/funasr/stt.py` L40–58**

```python
# Language codes natively supported by SenseVoice; anything else falls back to
# automatic language detection.
_FUNASR_LANGUAGES = {"zh", "en", "ja", "ko", "yue", "nospeech"}


def language_to_funasr_language(language: Language | str | None) -> str:
    """Map a language value to a SenseVoice language code.

    Args:
        language: A pipecat language, raw language code, or ``None`` for
            auto-detection.

    Returns:
        A SenseVoice language code (e.g. ``"zh"``), or ``"auto"``.
    """
    if language is None:
        return "auto"
    if isinstance(language, Language):
        code = str(language.value).split("-")[0].lower()
    else:
        code = str(language).split("-")[0].lower()
```

The set is at **L42**. Anything outside it coerces to `"auto"`. `Language.KO` has value `"ko"`, so
`str(language.value).split("-")[0].lower()` yields `"ko"`, which *is* in the set — Korean works, but
by string membership rather than by curated mapping. `FunASRSTTService` is a `SegmentedSTTService`
(`funasr/stt.py:87`) and has no TTFS constant.

> **SOURCE CORRECTION #3.** [[stt-korean-providers]]'s "Verified Korean" table lists FunASR as a
> thirteenth row at `funasr/stt.py:41` with code `"ko"`, in the same table as the twelve services
> that have a `Language.KO` key. The source disagrees: `funasr/stt.py` contains **no `Language.KO`
> mapping**, only `_FUNASR_LANGUAGES` at L42. The distinction is not pedantic — a curated map is a
> maintainer asserting the provider accepts that code, while a string-set membership check is a
> coincidence of ISO codes lining up. Count it as **12 verified + 1 by a different mechanism**, and
> when you shortlist, treat FunASR's evidence class as its own.

### 12.3 Passthroughs — no map, therefore no in-repo verification

**Deepgram** is the important one, because it has the joint-best measured latency in the whole table
and the repo takes no position on Korean:

```bash
$ grep -n "LANGUAGE_MAP\|language_to_service_language" src/pipecat/services/deepgram/stt.py
$
```

No map, no override. It serialises whatever it is holding:

**`src/pipecat/services/deepgram/stt.py` L579–582**

```python
        if is_given(s.model) and s.model is not None:
            kwargs["model"] = str(s.model)
        if is_given(s.language) and s.language is not None:
            kwargs["language"] = str(s.language)
```

`Language.KO` would go out as `"ko"`. Whether Deepgram accepts it is a question this repository does
not answer. **0.35 s and unknown Korean support** is the shape of that row, and you close it with a
benchmark, not with an inference.

`OpenAISTTService` and `OpenAIRealtimeSTTService` are the same shape — language passed straight
through, Korean riding on Whisper's own language set, nothing in-repo confirming it.
`DeepgramFluxSTTService` documents only `model="flux-general-multi"` plus `language_hints` and has
`supports_ttfs → False`.

### 12.4 Documented exclusions — read these before shortlisting

**`AssemblyAISTTService` has no Korean.** The map is exhaustive and Korean is not in it:

**`src/pipecat/services/assemblyai/stt.py` L129–149**

```python
    LANGUAGE_MAP = {
        Language.AR: "ar",
        Language.DA: "da",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.HE: "he",
        Language.HI: "hi",
        Language.IT: "it",
        Language.JA: "ja",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PT: "pt",
        Language.SV: "sv",
        Language.TR: "tr",
        Language.VI: "vi",
        Language.ZH: "zh",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)
```

Eighteen languages. Japanese and Chinese are there; Korean is not. `use_base_code=True` means
`Language.KO` falls through to `"ko"` **with a warning** — a fallback, not support. AssemblyAI's
0.42 s is the fourth-best number in the table and it is unavailable to you on the evidence in this
repo.

**`CartesiaTurnsSTTService` is excluded outright:**

**`src/pipecat/services/cartesia/turns/stt.py` L157–158**

```python
        # ink-2 is English-only at launch; language on emitted frames is fixed.
        self._language = Language.EN
```

**`SarvamSTTService`'s only `KO*` key is Konkani.** This is the misread the outline warns about:

**`src/pipecat/services/sarvam/stt.py` L845–852**

```python
        Language.EN_IN: "en-IN",
        Language.GU_IN: "gu-IN",
        Language.HI_IN: "hi-IN",
        Language.KN_IN: "kn-IN",
        Language.KOK_IN: "kok-IN",
        Language.MAI_IN: "mai-IN",
        Language.ML_IN: "ml-IN",
        Language.MR_IN: "mr-IN",
```

`KOK_IN` → `"kok-IN"` is **Konkani**, an Indo-Aryan language of Goa. Sarvam is an Indian-language
service. A grep for `KO` in that file finds it; a careless reader files Sarvam as Korean-capable.

**And one exclusion I found that the excerpts do not record.** The 0.41 s row in §11 is
`ELEVENLABS_REALTIME_TTFS_P99`, and it belongs to `ElevenLabsRealtimeSTTService`
(`elevenlabs/stt.py:452`) — **a different class from the `ElevenLabsSTTService` that carries the
`Language.KO: "kor"` mapping.** `language_to_service_language` is defined only once in that file, at
L322, inside the segmented class. The realtime class takes a raw
`language_code: str | None = None` (L494) and never consults the map. So:

| Class | Line | Base | P99 | Korean evidence |
|---|---|---|---|---|
| `ElevenLabsSTTService` | 210 | `SegmentedSTTService` | 2.01 | **verified** — `Language.KO: "kor"` |
| `ElevenLabsRealtimeSTTService` | 452 | `WebsocketSTTService` | 0.41 | passthrough — raw `language_code` string |

The fast ElevenLabs and the Korean-verified ElevenLabs are not the same service. If you read the
excerpt's table and the latency table together without opening the file, you would conclude
ElevenLabs offers verified Korean at 0.41 s streaming. It does not.

### 12.5 The open unknown, stated rather than guessed

Two greps, both returning nothing:

```bash
$ grep -rniE "\bWER\b|word error rate" $(find src/pipecat -name '*.py')
$
```

**There is no accuracy number of any kind, for any service, in any language, anywhere in this
repository.** `stt_latency.py` records *latency only*, is silent on the language and the sample rate
of the benchmark audio, and states only that the values were measured at `stop_secs=0.2`.

**And there is no 8 kHz telephony number for any STT service.** The only `8000` values in the tree
are the telephony serializer defaults [[ch-05/read]] enumerated. No STT module documents its
behaviour at that rate.

Put those two absences next to [[transport-telephony]]'s acoustics and the size of the gap is
concrete. Lina TMR's audio is 8 kHz μ-law: a 4 kHz Nyquist ceiling, 8-bit companding. Korean
fricatives and sibilants (ㅅ/ㅆ/ㅊ) and much of the acoustic cue for 받침 discrimination sit at or
above 4 kHz and are **not in the signal**. Upsampling 8 k → 16 k before the STT satisfies the model's
input contract; it does not restore the band. Therefore:

> **Every number in §11's table is an English-assumed, sample-rate-unstated latency measurement, and
> the entire Korean shortlist in §12.1 is currently ordered by it.** No accuracy evidence exists.
> The blocking item before committing to a provider is your own benchmark on real Lina TMR μ-law
> 8 kHz Korean audio, using <https://github.com/pipecat-ai/stt-benchmark>.

That is the honest state of the evidence. Do not let the precision of "0.35" fool you into thinking
the shortlist is decided.

---

# PART THREE — THE ARBITER: THE TURN-STRATEGY CHAIN

## 13. What is actually there

§0 established the absences. Here is the presence: **`src/pipecat/turns/` — 4,429 lines.**

```bash
$ find src/pipecat/turns -name "*.py" -exec cat {} + | wc -l
4429
```

| Sub-package | Lines | What lives there |
|---|---|---|
| `user_start/` | 1,076 | `VAD`, `Transcription`, `MinWords`, `WakePhrase`, `External`, `KrispVivaIP` start strategies |
| `user_stop/` | 1,219 | `SpeechTimeout`, `TurnAnalyzer`, `External`, `ExternalCompletion`, `LLMTurnCompletion`, `Deferred` |
| `user_mute/` | 259 | `Always`, `FirstSpeech`, `FunctionCall`, `MuteUntilFirstBotComplete` |
| `user_turn_completion_mixin.py` | 604 | the LLM ✓/○/◐ gate |
| `user_turn_controller.py` | 398 | runs the chains, owns the veto and the watchdog |
| `user_turn_processor.py` | 241 | the `FrameProcessor` that broadcasts the real turn frames |
| `user_idle_controller.py` | 161 | `on_user_turn_idle` |
| `user_turn_strategies.py` | 163 | the containers and the defaults |
| `types.py` | 24 | `ProcessFrameResult` |

Four and a half thousand lines to answer one boolean. That ratio is the design: the question is
genuinely hard, the evidence comes from three different subsystems, and Pipecat's response is to
make every piece of it swappable rather than to pick one heuristic.

---

## 14. The defaults, and what they commit you to

**`src/pipecat/turns/user_turn_strategies.py` L27–51**

```python
def default_user_turn_start_strategies() -> list[BaseUserTurnStartStrategy]:
    """Return the default user turn start strategies.

    Returns ``[VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]``.
    Useful when building a custom strategy list that extends the defaults.

    Example::

        start_strategies = [
            WakePhraseUserTurnStartStrategy(phrases=["hey pipecat"]),
            *default_user_turn_start_strategies(),
        ]
    """
    return [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]


def default_user_turn_stop_strategies() -> list[BaseUserTurnStopStrategy]:
    """Return the default user turn stop strategies.

    Returns ``[TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)]``.
    Useful when building a custom strategy list that extends the defaults.
    """
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    return [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
```

**The default start list is exactly one strategy per part of this chapter.** `VADUserTurnStartStrategy`
is part one's answer; `TranscriptionUserTurnStartStrategy` is part two's. They are ORed by chain
order: whichever fires first claims the turn and returns `STOP`, breaking the loop.

The transcript one exists specifically to cover VAD's failure mode:

**`src/pipecat/turns/user_start/transcription_user_turn_start_strategy.py` L14–44**

```python
class TranscriptionUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy based on transcriptions.

    This strategy signals the start of a user turn when a transcription is
    received while the bot is speaking. It is useful as a fallback in scenarios
    where VAD-based detection fails (for example, when the user speaks very
    softly) but the STT service still produces transcriptions.

    """

    def __init__(self, *, use_interim: bool = True, **kwargs):
        """Initialize transcription-based user turn start strategy."""
        super().__init__(**kwargs)
        self._use_interim = use_interim

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to detect the start of a user turn.

        Args:
            frame: The frame to be processed.

        Returns:
            STOP if a transcription was received, CONTINUE otherwise.
        """
        if isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        elif isinstance(frame, TranscriptionFrame):
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        return ProcessFrameResult.CONTINUE
```

`use_interim=True` by default — an *interim* transcript is enough to claim the floor. Remember §5's
liability analysis: a soft speaker whose energy never clears `min_volume = 0.6` is exactly who this
covers. And note the corollary: **boson's current text-triggered barge-in is a supported Pipecat
configuration**, not a thing Pipecat replaces. §19 spends that.

### 14.1 The default stop strategy is a machine-learning model

This is the fact people miss:

```python
    return [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
```

**Out of the box, Pipecat runs an ONNX end-of-turn classifier over the audio at every VAD stop.**
There is no configuration flag to turn it on; it is what you get if you configure nothing.

**`src/pipecat/audio/turn/smart_turn/local_smart_turn_v3.py`** — the concrete facts:

- L16–17: `import onnxruntime as ort`, `import soxr`
- L25: `_MODEL_SAMPLE_RATE = 16000`
- L28: `class LocalSmartTurnAnalyzerV3(BaseSmartTurn)`
- L35: `__init__(self, *, smart_turn_model_path: str | None = None, cpu_count: int = 1, **kwargs)`
- L50: `model_name = "smart-turn-v3.2-cpu.onnx"` — bundled with the package
- L73: `so.intra_op_num_threads = cpu_count` — one thread by default, same discipline as §1.1
- L136: `soxr.resample(audio_array, actual_rate, _MODEL_SAMPLE_RATE, quality="HQ")`
- L164: `log_mel = compute_whisper_log_mel_features(audio_array, do_normalize=True)`

**It is audio-only.** Whisper log-mel features in, complete/incomplete out. It never reads transcript
text. [[endpointing-turn-boundary]] draws the consequence and it is the right one for a Korean
product: *"smart-turn-v3 is acoustic, so it carries no Korean-language risk but gets no help from
Korean sentence-final endings."*

Sit with both halves of that. **No risk:** an English-trained *text* model has language-specific
segmentation behaviour; an acoustic model reads prosody, which is not tied to a script or a lexicon.
**No help:** Korean marks utterance completion explicitly in the morphology — `-습니다`, `-어요`,
`-거든요`, `-는데…` — and a final `-는데` trailing off is a *syntactically explicit* "I am not
finished." An acoustic model sees only the pitch contour and the trailing energy. The single most
reliable end-of-turn cue in the language is invisible to the default stop strategy. That is a
concrete, buildable gap and §20 makes it a probe.

Note also the resample at L136: the model runs at 16 kHz, so Lina's 8 kHz telephony audio gets
upsampled by `soxr` before inference — the same "restores sample count, not bandwidth" caveat from
[[transport-telephony]] applies to the turn model as much as to the STT.

**The VAD-only path is the downgrade you must explicitly select.** If you want cheap CPU and
predictable timing, you pass `SpeechTimeoutUserTurnStopStrategy` yourself. Nothing selects it for
you.

### 14.2 The asymmetry in `SmartTurnParams`

**`src/pipecat/audio/turn/smart_turn/base_smart_turn.py` L27–43**

```python
STOP_SECS = 3
PRE_SPEECH_MS = 500
MAX_DURATION_SECONDS = 8  # Max allowed segment duration
...
class SmartTurnParams(BaseTurnParams):
    """Configuration parameters for smart turn analysis.

    Parameters:
        stop_secs: Maximum silence duration in seconds before ending turn.
        pre_speech_ms: Milliseconds of audio to include before speech starts.
        max_duration_secs: Maximum duration in seconds for audio segments.
    """

    stop_secs: float = STOP_SECS
    pre_speech_ms: float = PRE_SPEECH_MS
    max_duration_secs: float = MAX_DURATION_SECONDS
```

`SmartTurnParams.stop_secs = 3` next to `VADParams.stop_secs = 0.2` looks like a contradiction. It
is not — they are two different clocks and only one of them is the normal path.

- `SmartTurnParams.stop_secs = 3` is the **analyzer's own silence fallback**: if three seconds of
  silence accumulate inside `append_audio` without any external trigger, the analyzer declares
  `COMPLETE` on its own.
- `VADParams.stop_secs = 0.2` is when the analyzer gets **consulted**. `_handle_vad_user_stopped_speaking`
  calls `analyze_end_of_turn()` at every VAD stop — every 192 ms of silence.

**So the ML verdict is the normal path, not the exception.** The 3-second constant fires only when
the VAD path failed to fire at all. Anyone who reads `stop_secs = 3` and budgets three seconds of
dead air per turn has inverted the design.

`pre_speech_ms = 500` is the analyzer's own pre-roll — a third one, after VAD's six-chunk lag and
the segmented STT's one second. Each layer buffers against the layer above it independently.

---

## 15. The two timing strategies, and the shared safety net

Both stop strategies that reason about time compute the same quantity from §11's table. Learn it
once.

### 15.1 `SpeechTimeoutUserTurnStopStrategy` — the VAD-only path (328 L)

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py` L27–53**

```python
class SpeechTimeoutUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy using two independent timers after VAD stop.

    After the user stops speaking (detected by VAD), this strategy runs two
    independent timers. The user turn stop is triggered only when both have
    finished and at least one transcript has been received:

    - user_speech_timeout: Policy floor — the window in which the user may
      resume speaking after a pause. Always runs to completion.
    - stt_timeout: Safety net for STT latency — the P99 time for the STT
      service to return a final transcript after VAD stop, adjusted by the
      VAD stop_secs. Short-circuited when the STT service emits a finalized
      transcript (TranscriptionFrame.finalized=True), since finalization
      means STT has nothing more to send.
    """

    def __init__(
        self,
        *,
        user_speech_timeout: float = 0.6,
        wait_for_transcript: bool = True,
        **kwargs,
    ):
```

Two timers, and the docstring names their roles precisely: one is a **policy floor**, one is a
**safety net**. They are not redundant.

- `user_speech_timeout = 0.6` — the grace window in which the customer may resume. It is a product
  decision about how long a Korean speaker's mid-sentence pause is allowed to be, and it always runs
  to completion regardless of what the STT does.
- `stt_timeout` — comes from the frame, not from config:

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py` L159–161**

```python
        if isinstance(frame, STTMetadataFrame):
            self._stt_timeout = frame.ttfs_p99_latency
            self._stop_secs_warned = False
```

That is §10's broadcast landing. The service publishes its own P99, and the turn strategy sizes its
timer from it. **This is the join between part two and part three**, and it is one line.

### 15.2 The arithmetic — worked before the formula

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py` L216–230**

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

`effective_stt_wait = max(0.0, self._stt_timeout - self._stop_secs)` is at **L223**.

Work it concretely before reading the formula as a formula. Pick two rows off §11's table and plug
them in with the default `stop_secs = 0.2`:

```
Soniox        (0.35 s):  max(0, 0.35 - 0.2) = 0.15 s of safety net
Speechmatics  (0.74 s):  max(0, 0.74 - 0.2) = 0.54 s
Google        (1.57 s):  max(0, 1.57 - 0.2) = 1.37 s
xAI           (2.14 s):  max(0, 2.14 - 0.2) = 1.94 s
```

Now say what the subtraction *means*. The provider's P99 is measured from the moment the customer
physically stopped talking (§9's `speech_end_time` anchoring). But the strategy only learns about it
`stop_secs` later, when the VAD frame arrives. Those 0.2 seconds have already been spent waiting.
Subtracting them stops you from paying for the same silence twice.

And the worst case, which the `max(0.0, ...)` guards: set `stop_secs = 0.5` with Soniox and
`0.35 - 0.5` is negative — the safety net collapses to zero. The code warns about exactly that:

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py` L196–214**

```python
        if not self._stop_secs_warned:
            if self._stop_secs != VAD_STOP_SECS:
                self._stop_secs_warned = True
                logger.warning(
                    f"{self}: VAD stop_secs ({self._stop_secs}s) differs from the "
                    f"recommended default ({VAD_STOP_SECS}s). Built-in p99 latency "
                    f"values assume stop_secs={VAD_STOP_SECS}. Re-run "
                    f"https://github.com/pipecat-ai/stt-benchmark with your settings "
                    f"and pass the TTFS P99 latency result as ttfs_p99_latency to "
                    f"your STT service."
                )
            if self._stt_timeout > 0 and self._stop_secs >= self._stt_timeout:
                self._stop_secs_warned = True
                logger.warning(
                    f"{self}: VAD stop_secs ({self._stop_secs}s) >= STT p99 latency "
                    f"({self._stt_timeout}s). STT wait timeout collapsed to 0s, which "
                    f"may cause delayed turn detection specified by the "
                    f"user_turn_stop_timeout parameter in the LLMUserAggregatorParams."
                )
```

Two warnings, and the first one is the loop closing on §2 and §11. `VAD_STOP_SECS` is imported
directly from `vad_analyzer.py` (L13 of this file). **Move `stop_secs` off 0.2 and the framework
tells you, at runtime, that every constant in the 23-entry table just became invalid for you.** That
is a rare and honest piece of engineering: a library that knows the provenance of its own defaults
and complains when you invalidate it.

### 15.3 The short-circuit

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py` L232–244**

```python
    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        self._text += frame.text

        if frame.finalized:
            self._transcript_finalized = True
            # Short-circuit the stt_timeout safety net: STT has told us
            # there's nothing more coming.
            if not self._stt_wait_done:
                self._stt_wait_done = True
                if self._stt_timeout_task:
                    await self.task_manager.cancel_task(self._stt_timeout_task)
                    self._stt_timeout_task = None
```

**The P99 timer is a ceiling you almost never pay.** In the normal case the transcript finalizes
well before P99 and the timer is cancelled. You pay `effective_stt_wait` only in the 1% tail, or
with a provider that never sets `finalized`. This is why picking `xAI` at 2.14 s does not mean
adding 1.94 s to every turn — it means your *worst* turns are that much worse.

And the anti-staleness rule, which is subtle:

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py` L170–181**

```python
        elif isinstance(frame, InterimTranscriptionFrame):
            # An interim means more transcription is still on the way, so an
            # earlier finalized transcript no longer covers all of the user's
            # speech.
            # Without this, a transcript finalized during a pause too short for
            # VAD to report a stop (and thus a new start, which is what normally
            # clears the flag) would leave the flag stale and skip the STT
            # safety net at the next VAD stop while the tail of the utterance is
            # still in flight. This can happen when the STT endpointer finalizes
            # on silences shorter than the VAD stop_secs — e.g. an aggressive
            # STT endpoint or a manually raised stop_secs.
            self._transcript_finalized = False
```

An interim *revokes* an earlier finalize. Read the comment's scenario: the STT's own endpointer
finalizes on a 100 ms pause that is shorter than VAD's 192 ms, so the VAD never reported a stop and
never cleared the flag. Then the customer keeps talking. Without this line the strategy would skip
the safety net at the next VAD stop while the tail of the utterance was still in flight, and cut the
customer off mid-sentence. **This is a bug class that only exists because two independent
endpointers with different thresholds are running at once**, and it is worth remembering when you
select a provider whose server-side endpointing you cannot disable.

---

## 16. `TurnAnalyzerUserTurnStopStrategy` — the default path (364 L)

Same two-clock structure, one crucial difference in how the deadline is computed.

**`src/pipecat/turns/user_stop/turn_analyzer_user_turn_stop_strategy.py` L225–245**

```python
    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        self._stop_secs = frame.stop_secs
        self._vad_stopped = True

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

`stt_deadline` at **L236**, `timeout` at **L245**. Compare with §15.2's relative
`effective_stt_wait` and the difference is the point of the comment:

```
SpeechTimeout:   effective_stt_wait = max(0, stt_timeout - stop_secs)      # relative duration
TurnAnalyzer:    stt_deadline       = timestamp - stop_secs + stt_timeout  # absolute wall-clock
                 timeout            = max(0, stt_deadline - time.time())   # remaining, computed AFTER inference
```

The order of the three statements is load-bearing. `stt_deadline` is computed **before**
`await self._turn_analyzer.analyze_end_of_turn()`, and `timeout` is computed **after**. ONNX
inference takes time. If the timer were armed with a relative duration after inference, that
inference time would be *added* to the STT budget rather than *spent from* it. Anchoring to an
absolute deadline makes the safety net a fixed wall-clock window that model latency eats into
instead of extending.

That is a genuinely nice piece of engineering and it is exactly the kind of detail that decides
whether a P50 latency target holds under load: at 40 concurrent calls the ONNX inference is
contending for CPU, and the design guarantees the contention does not multiply through the turn
budget.

Both analyzer shapes are handled by the same strategy:

**`src/pipecat/turns/user_stop/turn_analyzer_user_turn_stop_strategy.py` L203–216**

```python
    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Handle input audio to check if the turn is completed."""
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

        # Streaming analyzers (e.g. KrispVivaTurn) detect turn completion
        # frame-by-frame inside append_audio, so COMPLETE is returned here
        # rather than in analyze_end_of_turn. Batch analyzers (BaseSmartTurn)
        # return COMPLETE here only on a silence timeout. In either case we
        # consume and push metrics immediately while they're fresh.
        if state == EndOfTurnState.COMPLETE:
            _, prediction = await self._turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
            self._turn_complete = True
            await self._maybe_trigger_user_turn_stopped()
```

`append_audio` is called on **every** `InputAudioRawFrame`, and note it is passed
`self._vad_user_speaking` — the analyzer needs to know whether the audio it is buffering is speech.
This is the line that requires `audio_passthrough=True` on the STT (§7.1): the strategy sits
downstream of the STT service in the pipeline and only sees the audio because the STT forwarded it.

`EndOfTurnState` is `COMPLETE = 1` / `INCOMPLETE = 2` — a two-valued verdict from
`audio/turn/base_turn_analyzer.py`, whose ABC is five methods: `speech_triggered`, `params`,
`append_audio`, `analyze_end_of_turn`, `clear`. Writing a custom analyzer means implementing five
methods. §20 makes that a probe.

---

## 17. The controller: order, veto, and the watchdog

**`src/pipecat/turns/user_turn_controller.py` L203–211**

```python
        for strategy in self._user_turn_strategies.start or []:
            result = await strategy.process_frame(frame)
            if result == ProcessFrameResult.STOP:
                break

        for strategy in self._user_turn_strategies.stop or []:
            result = await strategy.process_frame(frame)
            if result == ProcessFrameResult.STOP:
                break
```

Start chain first, then stop chain, both in list order, both breaking on `STOP`. That is the entire
dispatch. `ProcessFrameResult` has two values (`turns/types.py`, 24 lines) and `STOP` means "I have
handled this frame; do not let the strategies after me see it."

**Order in the list is policy.** Put `WakePhraseUserTurnStartStrategy` before
`VADUserTurnStartStrategy` and the wake phrase gets first refusal on every frame. Put it after and
VAD claims the turn before the phrase matcher ever runs. There is no priority field — the list index
*is* the priority.

The veto:

**`src/pipecat/turns/user_turn_controller.py` L354–371**

```python
    async def _trigger_user_turn_stop(
        self, strategy: BaseUserTurnStopStrategy | None, params: UserTurnStoppedParams
    ):
        # Prevent two consecutive user turn stops.
        if not self._user_turn:
            return

        # Never finalize while the user is audibly speaking. A stop strategy can
        # finalize on a latent signal (e.g. an LLM ✓ that resolves after the
        # user resumed), which is stale by the time it arrives. Keep the turn
        # open so the next inference re-evaluates; the watchdog still finalizes
        # if the user then falls silent. Detector strategies only finalize once
        # the user has stopped, so this is a no-op for them.
        if self._user_speaking:
            return

        self._user_turn = False
        self._user_turn_stop_timeout_event.set()
```

Two guards. **L358** drops a duplicate stop for a turn already closed. **L367** is the one worth
naming: *no strategy may end the turn while the user is audibly speaking*, no matter what it
concluded. A stop verdict is always computed from evidence that is at least slightly stale — an LLM
completion check takes hundreds of milliseconds, ONNX inference takes tens — and the customer may
have resumed in the interval. The controller re-checks the live physical state at the moment of
finalization and refuses.

And the backstop, because a chain of vetoing strategies can deadlock:

**`src/pipecat/turns/user_turn_controller.py` L385–398**

```python
    async def _user_turn_stop_timeout_task_handler(self):
        while True:
            try:
                await asyncio.wait_for(
                    self._user_turn_stop_timeout_event.wait(),
                    timeout=self._user_turn_stop_timeout,
                )
                self._user_turn_stop_timeout_event.clear()
            except TimeoutError:
                if self._user_turn and not self._user_speaking:
                    await self._call_event_handler("on_user_turn_stop_timeout")
                    await self._trigger_user_turn_stop(
                        None, UserTurnStoppedParams(enable_user_speaking_frames=True)
                    )
```

`user_turn_stop_timeout: float = 5.0` from `LLMUserAggregatorParams`. Five seconds after the last
activity, if the turn is still open and the customer is silent, it force-stops with
`strategy=None` — nobody's verdict, the framework's. **Five seconds is a very long time on a sales
call.** §21 decides whether to keep it.

Downstream, `UserTurnProcessor` is what actually publishes:

**`src/pipecat/turns/user_turn_processor.py` L196–212**

```python
    async def _on_user_turn_started(
        self,
        controller: UserTurnController,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        logger.debug(f"{self}: User started speaking (strategy: {strategy})")

        if params.enable_user_speaking_frames:
            await self.broadcast_frame(UserStartedSpeakingFrame)

        await self._user_idle_controller.process_frame(UserStartedSpeakingFrame())

        if params.enable_interruptions:
            await self.broadcast_interruption()

        await self._call_event_handler("on_user_turn_started", strategy)
```

`broadcast_interruption()` at **L210** is where the turn decision becomes a barge-in.
[[ch-08/read]] owns everything downstream of that call; this chapter's job ends at producing it.

One more mechanism, because it prevents a class of double-decision bug:

**`src/pipecat/turns/user_turn_processor.py` L162–170**

```python
        elif isinstance(frame, ProposedUserStartedSpeakingFrame):
            # A proposal is resolved once. Forwarding one our own strategies
            # resolve would let a resolver further down the pipeline decide the
            # same turn a second time.
            if not self._user_turn_controller.resolves_proposed_turn_start_frames:
                await self.push_frame(frame, direction)
```

A `ProposedUserStartedSpeakingFrame` is **consumed**, not forwarded, when a local strategy resolved
it. That is the mechanism §21's external-endpointing services depend on.

---

## 18. Nine assignment sites, eight files, two unconditional

Some providers take the turn decision away from the chain entirely. The mechanism is the last
sentence of §10's docstring: a service overrides `service_metadata_frame()` and sets
`user_turn_strategies` on the frame it returns.

**Count it yourself:**

```bash
$ grep -rn "user_turn_strategies = ExternalUserTurnStrategies" $(find src/pipecat -name '*.py') | sort
src/pipecat/services/assemblyai/stt.py:669
src/pipecat/services/cartesia/turns/stt.py:198
src/pipecat/services/deepgram/flux/stt_base.py:250
src/pipecat/services/gladia/stt.py:370
src/pipecat/services/openai/stt.py:414
src/pipecat/services/sarvam/stt.py:395
src/pipecat/services/sarvam/stt.py:1034
src/pipecat/services/soniox/stt.py:422
src/pipecat/services/speechmatics/stt.py:561
```

**Nine sites across eight files.** `sarvam/stt.py` has two — one per service class,
`SarvamSTTService` at L173 and `SarvamRealtimeSTTService` at L891. And **only two of the nine are
unconditional.**

### 18.1 The two unconditional sites

**`src/pipecat/services/cartesia/turns/stt.py` L189–201**

```python
    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend external turn strategies: this service detects turns server-side.

        Cartesia's turn-detection STT defines turn boundaries on the server and
        emits ``ProposedUserStarted/StoppedSpeakingFrame``, so the user aggregator
        resolves those rather than running local VAD/smart-turn. Applied unless
        the user passed their own ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        frame.user_turn_strategies = ExternalUserTurnStrategies(
            enable_interruptions=self._should_interrupt,
        )
        return frame
```

**`src/pipecat/services/deepgram/flux/stt_base.py` L241–253**

```python
    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend external turn strategies: Flux detects turns server-side.

        Flux emits its own start-of-turn and end-of-turn events (as
        ``ProposedUserStarted/StoppedSpeakingFrame``), so the user aggregator
        resolves those rather than running local VAD/smart-turn. Applied unless
        the user passed their own ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        frame.user_turn_strategies = ExternalUserTurnStrategies(
            enable_interruptions=self._should_interrupt,
        )
        return frame
```

No guard. Choosing either service *is* choosing external endpointing. Both also return
`supports_ttfs → False` (§10), which is why neither appears in §11's table: the server owns the
boundary, so there is no "speech end → final transcript" interval to measure.

### 18.2 The seven flag-gated sites, with exact guards and defaults

Every one of the remaining seven sits under an `if`. Here is each guard verbatim, with the flag's
default resolved from its constructor:

| Service | Site | Guard | Flag default | Fires by default? |
|---|---|---|---|---|
| `SpeechmaticsSTTService` | `speechmatics/stt.py:561` | `if is_given(mode) and mode != TurnDetectionMode.EXTERNAL` | `turn_detection_mode = TurnDetectionMode.EXTERNAL` (L308) | **no** |
| `GladiaSTTService` | `gladia/stt.py:370` | `if self._settings.enable_vad` | `enable_vad=False` (L279) | **no** |
| `SonioxSTTService` | `soniox/stt.py:422` | `if not self._vad_force_turn_endpoint` | `vad_force_turn_endpoint: bool = True` (L287) | **no** |
| `AssemblyAISTTService` | `assemblyai/stt.py:669` | `if not self._vad_force_turn_endpoint` | `vad_force_turn_endpoint: bool = True` (L319) | **no** |
| `SarvamSTTService` | `sarvam/stt.py:395` | `if self._settings.vad_signals` | `vad_signals=None` (L269) | **no** |
| `SarvamRealtimeSTTService` | `sarvam/stt.py:1034` | `if self._endpointing == "vad"` | `endpointing: Literal["vad", "manual"] = "vad"` (L922) | **YES** |
| `OpenAISTTService` | `openai/stt.py:414` | `if self._server_vad_enabled` | `turn_detection: dict \| Literal[False] \| None = False` (L248) → `_server_vad_enabled = turn_detection is not False` (L367) | **no** |

Four of those rows have traps in them.

**Speechmatics' `EXTERNAL` means the opposite of what it looks like.** Read the enum docstring:

**`src/pipecat/services/speechmatics/stt.py` L71–85**

```python
class TurnDetectionMode(StrEnum):
    """Endpoint and turn detection handling mode.

    How the STT engine handles the endpointing of speech. If using Pipecat's built-in endpointing,
    then use `TurnDetectionMode.EXTERNAL` (default).

    To use the STT engine's built-in endpointing, then use `TurnDetectionMode.ADAPTIVE` for simple
    voice activity detection or `TurnDetectionMode.SMART_TURN` for more advanced ML-based
    endpointing.
    """

    FIXED = "fixed"
    EXTERNAL = "external"
    ADAPTIVE = "adaptive"
    SMART_TURN = "smart_turn"
```

`EXTERNAL` is named from **Speechmatics'** point of view — "the endpointer is external to me, i.e.
Pipecat's." From Pipecat's point of view it means *internal*. So the guard
`mode != TurnDetectionMode.EXTERNAL` reads backwards until you know that, and the default `EXTERNAL`
means Speechmatics does **not** take the turn decision. Four modes, three of which hand it over.

**Soniox and AssemblyAI are gated on the identical flag.** Both `vad_force_turn_endpoint: bool = True`,
both guarded `if not self._vad_force_turn_endpoint`, and both docstrings say so in the same words:

**`src/pipecat/services/soniox/stt.py` L413–418**

```
        With ``vad_force_turn_endpoint=False`` Soniox's endpoint detection decides
        turn endings and this service proposes turn boundaries, so the user
        aggregator resolves those rather than running local VAD/smart-turn. In the
        default Pipecat mode (``vad_force_turn_endpoint=True``) the STT proposes
        no turns, so the defaults are left in place. Applied unless the user
        passed their own ``user_turn_strategies``.
```

**`src/pipecat/services/assemblyai/stt.py` L660–665**

```
        With ``vad_force_turn_endpoint=False`` AssemblyAI's model decides turn
        endings and emits ``ProposedUserStarted/StoppedSpeakingFrame``, so the user
        aggregator resolves those rather than running local VAD/smart-turn. In the
        default Pipecat mode (``vad_force_turn_endpoint=True``) the STT proposes no
        turns, so the defaults are left in place. Applied unless the user passed
        their own ``user_turn_strategies``.
```

Same flag, same default, same behaviour. **Do not classify them differently.** Any table that puts
one in a different group from the other is wrong. (AssemblyAI adds one constraint the other does not:
`assemblyai/stt.py` L436–440 raises unless the model is `u3-pro` when you set the flag to `False`.)

**`SarvamRealtimeSTTService` is the one flag-gated site that behaves unconditionally out of the box.**
`endpointing` defaults to `"vad"`, so the guard passes unless you explicitly pass `"manual"`. If you
are counting "services that take the turn decision by default," the answer is **three**, not two.

**OpenAI's guard has an off-by-one-value hazard.** `_server_vad_enabled = turn_detection is not False`
— so `turn_detection=False` (the default) disables it, but `turn_detection=None` **enables** it,
because `None is not False`. A caller who writes `turn_detection=None` meaning "unset" gets server
VAD. Worth knowing if you ever wire this from a config file where absent fields deserialise to `None`.

### 18.3 The summary you should be able to reproduce

> **9 assignment sites · 8 service files · 2 unconditional · 7 provider-flag-gated · 3 firing by
> default.**

And the payload is always the same container:

**`src/pipecat/turns/user_turn_strategies.py` L81–109**

```python
@dataclass
class ExternalUserTurnStrategies(UserTurnStrategies):
    """Container for turn strategies driven by another component in the pipeline.

    Preconfigures :class:`UserTurnStrategies` with
    :class:`~pipecat.turns.user_start.ExternalUserTurnStartStrategy` and
    :class:`~pipecat.turns.user_stop.ExternalUserTurnStopStrategy`, so a service
    with its own turn detection — or a shared
    :class:`~pipecat.turns.user_turn_processor.UserTurnProcessor` — controls when
    user turns start and stop.

    What the aggregator emits depends on which signal drives the turn.
    ``ProposedUserStarted/StoppedSpeakingFrame`` leaves the decision here, so the
    aggregator pushes the turn frames and broadcasts interruptions.
    ``UserStarted/StoppedSpeakingFrame`` means the emitter already announced the
    turn, so the aggregator emits nothing and the parameter below doesn't apply.

    Parameters:
        enable_interruptions: Whether to broadcast an interruption when a
            proposal starts a turn. Services route their ``should_interrupt``
            setting here.

    """

    enable_interruptions: bool = True

    def __post_init__(self):
        self.start = [ExternalUserTurnStartStrategy(enable_interruptions=self.enable_interruptions)]
        self.stop = [ExternalUserTurnStopStrategy()]
```

`__post_init__` **replaces** both lists unconditionally. The local VAD and smart-turn strategies are
not augmented; they are gone. That is the mechanism §19 cares about.

### 18.4 A separate and different mechanism — not part of the nine

Three realtime **LLM** services pass the same container as a **constructor kwarg** rather than
assigning it on a metadata frame:

```bash
$ grep -rn "user_turn_strategies=ExternalUserTurnStrategies" $(find src/pipecat -name '*.py') | sort
src/pipecat/extensions/voicemail/voicemail_detector.py:631
src/pipecat/services/inworld/realtime/llm.py:445
src/pipecat/services/openai/realtime/llm.py:551
src/pipecat/services/xai/realtime/llm.py:394
```

The three LLM ones share a pattern —
`user_turn_strategies=ExternalUserTurnStrategies() if emits_turn_frames else None` — and are not STT
services, so they are **not part of the nine**. (The fourth hit is
`extensions/voicemail/voicemail_detector.py:631`, which constructs an `LLMUserAggregatorParams`
directly; it is neither an STT service nor a realtime LLM, and it is not part of any of these counts.
The outline named three sites for this mechanism; the source has four, of which three are services.)

---

## 19. Re-founding boson on real audio

Now put boson beside all three parts. **This is a mechanism differential, not a verdict.**
[[ch-13/read]] scores.

### 19.1 What boson decides a turn from today

boson-agent has **no audio path in its interruption subsystem at all.** [[boson-interrupt-subsystem]]
states it flatly: *"Every interruption decision in the codebase — `PartialDetector.is_partial`,
`WordFilterPolicy.evaluate`, `fillers.is_filler`, `InterruptionGate.allows` — takes `text: str` as
its primary argument. There is no audio path, no energy threshold, no VAD."*

And a grep for `deepgram|whisper|speech_to_text|vad` across `boson-agent/packages/**/*.py` returns
**zero hits** ([[stt-service-interface]], [[stt-korean-providers]]). Part two of this chapter is
therefore a **net addition** to boson, not a port. There is nothing to migrate.

The live end-of-turn mechanism is a silence timer in the Gateway:

- `gateway/server/websocket.py:616` — `_start_silence_timer` sleeps `silence_timeout_ms / 1000`,
  default **2000 ms**, then calls `_finalize_partial` (`:661`) ([[boson-gateway-server]]).
- `gateway/interrupt/policy.py` — `default_bargein_policy()` is
  `CompositePolicy([DurationPolicy(min_ms=500), WordFilterPolicy(...)], mode="all")`
  ([[boson-interrupt-subsystem]]).

> **A correction the outline needs.** [[endpointing-turn-boundary]] calls
> `PartialDetector.should_finalize` *"the entire boson end-of-turn mechanism"* and the outline says
> `ExternalUserTurnStrategies` would let `PartialDetector` *"be deleted outright rather than merely
> replaced."* [[boson-interrupt-subsystem]] contradicts both: `PartialDetector` is constructed
> exactly once at `bootstrap.py:316`, stored by `core.py:175`, and **`self._partial_detector` is
> never read anywhere** — every remaining reference is in `packages/gateway/tests/`. It is already
> dead code. What actually runs is `_start_silence_timer` / `_finalize_partial` in
> `gateway/server/websocket.py`. So: `PartialDetector` can be deleted today, for reasons that have
> nothing to do with Pipecat. The live 2000 ms timer is the thing this chapter's part three
> replaces. Do not conflate them.

### 19.2 The three shapes on one timeline

Mechanism, not judgement. All three measured from the customer's last voiced sample.

```
boson today (text-only, no audio path):
  [last voiced sample]
    → client-side ASR emits its final partial          ← interval NOT bounded by boson
    → _start_silence_timer sleeps 2000 ms              ← websocket.py:616
    → _finalize_partial                                ← websocket.py:661
    → first LLM token
  floor: 2000 ms of text silence, plus the client ASR's own emission interval

realtime_voice (audio VAD, unary ASR):
  [last voiced sample]
    → min_silence_frames(6) × frame_duration           ← 120 ms @20 ms frames … 600 ms @100 ms
    → finalize(): whole-utterance WAV upload, one blocking HTTPS call
      openai_compat.py L194-242, timeout_seconds=1.5   ← [[rtv-vs-pipecat-gap]]
    → first LLM token
  floor: VAD offset + full transcription RTT, SERIAL

Pipecat, default configuration:
  [last voiced sample]
    → stop_secs 0.2 s = 6 chunks = 192 ms              ← §2
    → LocalSmartTurnAnalyzerV3 ONNX inference          ← consulted at EVERY vad stop, §14.2
    → transcript already ~computed (streaming, §7.2); finalized short-circuits the net, §15.3
    → first LLM token
  floor: 192 ms + inference, transcription OVERLAPPED

Pipecat, VAD-only downgrade (SpeechTimeoutUserTurnStopStrategy):
    → 192 ms + user_speech_timeout 0.6 s = ~0.79 s, plus max(0, ttfs_p99 - 0.2) in the tail
```

The structural difference is one word: **serial versus overlapped.** realtime_voice's transcription
round trip begins after the endpoint decision; a streaming STT's has been running throughout the
customer's utterance and needs only to close out. [[ch-03/read]] §5.3 put the same two shapes on a
timeline; part two of this chapter is what makes the second line's terms nameable.

`CLAUDE.md`'s target for boson, quoted in [[rtv-vs-pipecat-gap]], is *"P50 at or below 1.0 seconds
and P95 at or below 1.5 seconds"* measured *"from the last voiced user sample to the first audible
assistant sample, including end-of-turn/VAD time."* Every term above is inside that window.
[[ch-11/read]] does the budget arithmetic; this chapter's job was to make each term a quantity with a
file and a line behind it.

### 19.3 What maps, what does not

| boson piece | Pipecat counterpart | Faithful? |
|---|---|---|
| `_start_silence_timer(2000 ms)` | a stop strategy — `SpeechTimeout(0.6)` or the smart-turn default | yes, and denominated differently |
| `DurationPolicy(min_ms=500)` | `VADParams.start_secs = 0.2` → 6 chunks | yes |
| text-triggered barge-in | `TranscriptionUserTurnStartStrategy(use_interim=True)` | **exactly** — boson's behaviour is a supported Pipecat config |
| `WordFilterPolicy(ignore_words=[…], max_chars=3)` | `MinWordsUserTurnStartStrategy(min_words=…)` | **no** — Pipecat counts *words*, boson counts *characters*, and Korean backchannels are 1 word / 2 chars |
| `fillers` registry (agent-registered callback) | none | write a `BaseUserTurnStartStrategy` subclass |
| `PartialDetector` | — | already dead code; delete regardless |

The `WordFilterPolicy` row is the one to internalise. `MinWordsUserTurnStartStrategy` does this:

**`src/pipecat/turns/user_start/min_words_user_turn_start_strategy.py` L108–111**

```python
        min_words = self._min_words if self._bot_speaking else 1

        word_count = len(frame.text.split())
        should_trigger = word_count >= min_words
```

`len(frame.text.split())`. "네" is one token by that count and would clear `min_words=1`; it is two
*characters* and would be caught by boson's `max_chars=3`. Setting `min_words=2` to catch it would
also suppress "잠깐만요" — a genuine floor claim. **Pipecat has no Korean backchannel filter and the
nearest thing counts the wrong unit.** That gap is real, it is yours to close, and §20 makes it the
first probe.

Note the L108 conditional too: `min_words` applies **only while the bot is speaking**, dropping to 1
otherwise. The strategy is stricter about interruptions than about turn-taking in silence, which is
the right asymmetry and one you would have to remember to build yourself.

---

## 20. Framework-extension probes

Three moves. Each applies a mechanism from this chapter to a domain the chapter did not cover. Write
the answers before [[ch-08/read]].

**Probe 1 — the Korean backchannel strategy.** §19.3 established that `MinWordsUserTurnStartStrategy`
counts words where boson's `WordFilterPolicy` counts characters, and that neither system filters
backchannels at the VAD layer. Sketch a `KoreanBackchannelUserTurnStartStrategy(BaseUserTurnStartStrategy)`.
Three design questions to answer explicitly: (a) does it sit *before* or *after*
`VADUserTurnStartStrategy` in the start list, and what does the §17 break-on-`STOP` semantics mean
for your answer? (b) `VADUserTurnStartStrategy` fires on `VADUserStartedSpeakingFrame`, which carries
no text at all — so can a text-based filter suppress it, or must you drop the VAD strategy entirely
and go transcript-only? (c) if you go transcript-only, what does §5.2's onset-latency analysis say
you have given up? There is a right answer to (b) and it is not comfortable.

**Probe 2 — a Korean-morphology turn analyzer.** §14.1 established that `LocalSmartTurnAnalyzerV3` is
audio-only and never reads transcript text, and that Korean marks utterance completion explicitly in
the morphology. `BaseTurnAnalyzer` is a five-method ABC (`speech_triggered`, `params`, `append_audio`,
`analyze_end_of_turn`, `clear`). Design a `KoreanEndingTurnAnalyzer` that returns `INCOMPLETE` on a
trailing `-는데` / `-고` / `-지만` and `COMPLETE` on `-습니다` / `-어요`. Then answer the hard part:
`append_audio(buffer, is_speech)` receives **audio**, not text, so where does your analyzer get the
transcript from — and what does §16's `stt_deadline` arithmetic say about the latency of an analyzer
that has to wait for one? Is the right shape an analyzer at all, or a second stop strategy stacked
after `TurnAnalyzerUserTurnStopStrategy`? Justify from the §17 chain semantics.

**Probe 3 — the provider decision, made as a decision.** Using §11's table and §12's evidence
classes, produce the Lina TMR shortlist as a *ranked* list with the evidence class named for each
row. Constraints you must respect: verified Korean, streaming (not segmented), and the §12.5
absence of any accuracy or 8 kHz number. Then answer the question that makes it real: Deepgram sits
at 0.35 s — joint-best in the table — with **zero** in-repo Korean evidence, while Soniox sits at
the same 0.35 s with a verified mapping *and* optional server-side endpointing. Under what
circumstance would you still run the Deepgram benchmark first? Name the circumstance, do not name a
preference.

---

## 21. Deliverable: the Lina TMR turn-boundary configuration

Every value below is decided from a section above, with the section named. This is a configuration,
not a recommendation about Pipecat-versus-boson — [[ch-13/read]] owns that question and this
deliverable is conditional on its answer.

```python
# Conditional on adopting Pipecat; see [[ch-13/read]] for that decision.

user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        # ---- PART ONE: the audio layer ----------------------------------
        vad_analyzer=SileroVADAnalyzer(          # §4: mounts HERE, not on the transport
            params=VADParams(
                confidence=0.7,                  # §2.3: UNVERIFIED on 8 kHz μ-law — measure
                start_secs=0.2,                  # §2: 6 chunks = 192 ms, identical at 8 kHz
                stop_secs=0.2,                   # §15.2: DO NOT MOVE — invalidates the P99 table
                min_volume=0.6,                  # §1.2: smoothed volume, UNVERIFIED on μ-law
            ),
        ),
        audio_idle_timeout=1.0,                  # §1.4: carrier one-way-audio watchdog

        # ---- PART THREE: the arbiter ------------------------------------
        user_turn_strategies=UserTurnStrategies(
            start=[
                VADUserTurnStartStrategy(),                    # §14
                TranscriptionUserTurnStartStrategy(),          # §14: soft-speaker fallback
                # KoreanBackchannelUserTurnStartStrategy(),    # §20 probe 1 — MUST BUILD
            ],
            stop=[
                TurnAnalyzerUserTurnStopStrategy(              # §14.1: the default, kept
                    turn_analyzer=LocalSmartTurnAnalyzerV3(),
                ),
            ],
        ),
        user_turn_stop_timeout=5.0,              # §17 — see the note below
        user_idle_timeout=0,
    ),
)

stt = <ProviderSTTService>(
    ttfs_p99_latency=<YOUR MEASURED VALUE>,      # §11 + §12.5: do NOT ship the constant
)
```

**Five decisions and their reasons.**

1. **`stop_secs` stays at 0.2.** §15.2: every constant in §11's 23-entry table was measured at that
   value, and both timing strategies emit a runtime warning the moment you move it. If you have a
   reason to move it, you also have a reason to re-run the benchmark — the two are the same
   decision.

2. **`ttfs_p99_latency` is passed explicitly, never left to the constant.** §11's numbers are
   English-assumed and sample-rate-unstated; §12.5 says no Korean or 8 kHz number exists anywhere.
   Shipping the default means shipping someone else's measurement as if it were yours. And §10's
   `ttfs is None` branch means forgetting it entirely gets you a silent 1.0 s.

3. **The smart-turn default is kept.** §14.1: it is acoustic, so it carries no Korean-language risk.
   It also gets no help from Korean sentence-final endings, which is what probe 2 exists to close.
   Keeping it is the low-risk position until that probe produces something measurable.

4. **`confidence` and `min_volume` are flagged UNVERIFIED, not tuned.** §2.3 fenced this carefully:
   the frame-count arithmetic transfers to 8 kHz exactly, the *thresholds* are conditioned on a
   signal distribution that μ-law companding changes, and no evidence in this repository speaks to
   it. Ship the defaults, instrument them, and treat the first hundred calls as the measurement.

5. **`user_turn_stop_timeout=5.0` is kept but flagged.** §17: five seconds of a customer's silence
   before the framework force-closes the turn is a long time on a sales call. It fires only when the
   strategy chain deadlocks, which should be never — so lowering it trades a rare bad experience for
   a more common premature cut. Instrument `on_user_turn_stop_timeout` before touching the number.
   If it never fires, the value does not matter; if it fires regularly, you have a chain bug, not a
   timeout bug.

**Two things this configuration does not do, stated so they are not lost.**

- **It does not use `ExternalUserTurnStrategies`.** §18 counted three services that would take the
  turn decision by default. Handing the boundary to a provider means the §14–§17 chain — including
  the §17 veto and watchdog — is replaced wholesale by `__post_init__` (§18.3), and it means
  `supports_ttfs → False` for the turn-based ones. That is a bigger decision than a provider choice
  and it should be made after probe 3, not before.

- **It does not decide the provider.** Probe 3 does, and §12.5 says the shortlist cannot honestly be
  ordered until the 8 kHz μ-law Korean benchmark is run.

→ **Panel three of [`figures/turn-boundary.html`](figures/turn-boundary.html) makes this
configuration live.** The `stop_secs` slider and the provider selector move the turn boundary on a
shared time axis, the smart-turn verdict and the `max(0, ttfs_p99 - stop_secs)` safety net run as
competing clocks against the absolute `stt_deadline`, and the external-endpointing strip lists all
nine sites with a toggle per guard — including one shared toggle driving both the Soniox and
AssemblyAI rows, so §18.2's identical gating is visible rather than asserted. Move `stop_secs` off
0.2 and the panel raises the same warning the source does.

---

## 다음 챕터로

What this chapter hands forward, named so later chapters cite it rather than re-deriving it:

- **The VAD quantities** (§2) — `start_secs=0.2` and `stop_secs=0.2` are **6 chunks and 192 ms**, at
  16 kHz *and* at 8 kHz, because 512/16000 and 256/8000 are both 0.032. `round(6.25) = 6`, not 7 —
  the excerpts say 7 and they are wrong. [[ch-08/read]] spends the onset number on barge-in latency;
  [[ch-11/read]] spends the offset number on the budget. Neither re-derives it.

- **The three-way split and the two frame families** (§1, §3) — model / analyzer / controller, and
  `VADUserStartedSpeakingFrame` ("there is voice") versus `UserStartedSpeakingFrame` ("this is a
  turn"). [[ch-08/read]]'s cascade begins at `broadcast_interruption()` in
  `user_turn_processor.py:210`, which is where this chapter's chain ends.

- **The mount point** (§4) — `LLMUserAggregatorParams`, not `TransportParams`, per CHANGELOG L4402.
  Two silent failure modes hang off it: no `vad_analyzer` means no VAD frames at all, and no VAD
  frames means a `SegmentedSTTService` that never transcribes.

- **The 23-constant `stt_latency.py` table** (§11) — built once, here, with `DEFAULT_TTFS_P99 = 1.0`
  and `NVIDIA` / `WHISPER` as aliases rather than measurements, a 0.35–2.14 s spread, and every value
  measured at `stop_secs=0.2`. [[ch-11/read]]'s waterfall consumes a selected value and must not
  re-render the table.

- **The Korean evidence classes** (§12) — 12 verified `Language.KO` mappings, FunASR as a thirteenth
  by plain string set, Deepgram as a passthrough the repo takes no position on, AssemblyAI as a
  documented exclusion, Cartesia English-only, `KOK_IN` as Konkani, and the ElevenLabs class split
  where the fast service and the Korean-verified service are different classes. Plus the absence:
  **zero accuracy numbers and zero 8 kHz numbers anywhere in the tree.** [[ch-07/read]] faces the
  mirror-image question for TTS.

- **`effective_stt_wait = max(0, ttfs_p99 - stop_secs)`** (§15.2) and the absolute
  `stt_deadline = timestamp - stop_secs + ttfs_p99` (§16). Both terms come from different layers of
  this chapter, which is why this chapter had to teach all three.

- **The 9 / 8 / 2 / 7 count** (§18) — nine `ExternalUserTurnStrategies` assignment sites across eight
  service files, two unconditional (`cartesia/turns/stt.py:198`, `deepgram/flux/stt_base.py:250`),
  seven flag-gated, **three firing by default** once `SarvamRealtimeSTTService`'s default-on
  `endpointing="vad"` is counted. Soniox and AssemblyAI share the identical
  `vad_force_turn_endpoint` flag and belong in the same group.

[[ch-07/read]] takes the other half of the voice loop: streaming TTS, first audible sample, word
timestamps, and the Korean provider question again from the synthesis side. It needs §11's table
shape as a model — it will build its own — and it needs [[ch-05/read]]'s 8 kHz μ-law constraint,
which bites the output path as hard as it bites this one.

Open questions parked here so they are not lost:

- **The Korean 8 kHz μ-law STT benchmark.** §12.5. It is the blocking item for provider selection and
  nothing in this repository can substitute for it. It also silently blocks §21's `confidence` /
  `min_volume` decision, because both are thresholds on a distribution μ-law changes.
- **Whether Korean morphology belongs in the turn decision at all.** §14.1 and probe 2. The default
  stop strategy cannot see the most reliable end-of-turn cue in the language. Whether that costs
  anything measurable on real calls is unknown and testable.
- **Whether to hand the boundary to a provider.** §18. `ExternalUserTurnStrategies.__post_init__`
  replaces the entire chain, including the §17 veto. Decide after probe 3, and score it in
  [[ch-13/read]].
- **boson's `fillers` registry.** [[boson-interrupt-subsystem]] records an agent-registered
  language-specific ignore callback with no Pipecat hook. Probe 1 covers the backchannel half;
  the registry's *dynamic* half — an agent changing its own ignore list mid-call — has no counterpart
  in a strategy list that is fixed at aggregator construction. [[ch-12/read]] owns it.
