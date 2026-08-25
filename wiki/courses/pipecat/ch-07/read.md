---
title: "Streaming TTS: First Audible Sample, Word Timestamps, and Korean"
chapter: ch-07
phase: voice-io
course: pipecat
sources:
  - tts-service-interface
  - tts-korean-providers
  - canonical-voice-bot
  - rtv-vad-chunking
deps:
  - ch-03
  - ch-04
  - ch-05
figure: figures/tts-streaming.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# Chapter 7 — Streaming TTS: First Audible Sample, Word Timestamps, and Korean

> **Scope, stated up front and enforced for the whole chapter.** This chapter describes what two
> designs **do**. It does not rank them. When realtime_voice's `KoreanPhraseChunker` appears in §9
> it appears as an algorithm with inputs and outputs, next to Pipecat's algorithm with its inputs
> and outputs, and the only comparisons made are *measurements* — same input, both paths, printed
> result. No "better", no "should adopt", no "the right choice". [[ch-13/read]] is the only chapter
> that scores anything, and it needs the mechanics of ch-05 through ch-12 in hand before it can.
>
> Two things in this chapter **are** verdicts, and both are verdicts about the Pipecat source rather
> than about a design: the outline that generated this chapter made two factual claims that the code
> at commit `0cbf9c5b` contradicts. Those are corrected in §0, plainly, before anything is built on
> top of them.

---

## 왜 이 챕터인가

[[ch-04/read]] left you at the point where a `PipelineWorker` is running, four exits are wired, and
`transport.output()` is playing audio at the end of a seven-processor chain. It deliberately did not
open the two boxes labelled `stt` and `tts` — it treated them as processors that produce and consume
frames, which for runtime purposes is all they are.

This chapter opens the second box. Position 5 in the canonical pipeline ([[canonical-voice-bot]],
`examples/getting-started/06-voice-agent.py:81-91`) is one object, `CartesiaTTSService`, and it
turns out to be the single most latency-sensitive processor in the whole chain — not because it is
slow, but because it is the **last** thing that happens before the customer hears anything. Every
millisecond upstream of it is invisible; every millisecond inside it is audible as dead air.

Concretely, for Lina: 고객님 finishes a sentence, Silero fires `UserStoppedSpeakingFrame`, the LLM
starts streaming Korean tokens, and then there is a gap before the first 소리 comes out of the phone.
That gap is what the customer experiences as "이 상담원 좀 느리네." This chapter is about the three
things that make up that gap, all three of which live inside `TTSService`, and about the one of them
that byte-level monitoring cannot see.

It is also the chapter where Korean stops being a configuration string and starts being a
constraint. There are 12 language maps in this tree with a Korean entry, 12 services that emit word
timestamps, and the intersection has 6 members. If you want barge-in that truncates the assistant
context at the last **spoken** word — the mechanism [[ch-08/read]] is built on — you are choosing
from those 6 and no others. And if you wanted on-prem Korean TTS, this chapter is where you find out
that it is not available in this tree at all.

Three questions run through everything below:

1. **Why does the whole design optimise one number, and which number is it?** Not TTFB. §1.
2. **Where does word-level timing actually live?** There is no `WordBoundaryFrame`. There is no
   `TTSWordFrame`. §6 tells you what there is instead, and why the answer is a `pts` field.
3. **What does Korean cost you here?** A specific provider shortlist, a silent-failure mode, and a
   code path that was written for zh/ja and that Korean deliberately falls *out* of. §7 and §8.

---

## 0. Two corrections to make before we start

The outline that produced this chapter, and one of the excerpts feeding it, each state something the
source contradicts. The standing rule for this course is that **the source wins**. Both corrections
change conclusions, so they go first rather than in a footnote.

### 0.1 The live ladder has three classes, not four. `AudioContextTTSService` IS deprecated.

The outline says, verbatim:

> the 'everything with Word in the name is deprecated' escape clause does NOT collapse the tree,
> because `AudioContextTTSService(WebsocketTTSService)` at :2083 has no 'Word' in its name, is not
> deprecated, and is a fourth live class

Open the file.

**`src/pipecat/services/tts_service.py:2079-2094`**
```python
@deprecated(
    "`AudioContextTTSService` is deprecated since 0.0.105 and will be removed in 2.0.0. "
    "Use `WebsocketTTSService` instead."
)
class AudioContextTTSService(WebsocketTTSService):
    """Deprecated. Inherit from WebsocketTTSService directly instead.

    Audio context management (previously the main purpose of this class) is now
    built into TTSService. This class is kept only for backwards compatibility.

    .. deprecated:: 0.0.105
        Subclass :class:`WebsocketTTSService` directly and pass
        ``reuse_context_id_within_turn`` as
        keyword arguments to its ``__init__``.
        Will be removed in 2.0.0.
    """
```

It carries the `@deprecated` decorator, the `.. deprecated:: 0.0.105` directive, and a docstring
whose first line is "Deprecated." The outline's claim is false. The excerpt
[[tts-service-interface]] got the *list* right (it includes `AudioContextTTSService` among the
deprecated classes) while getting the *headline* sloppy ("everything with 'Word' in the name is
deprecated" — true but not exhaustive; `AudioContextTTSService` has no "Word" in its name and is
deprecated anyway).

The consequence matters. The outline concluded "the live ladder a subclass author actually chooses
from is those four, and the axes are two — websocket transport, and per-turn audio contexts."
**Neither half survives.** The live ladder is three classes:

```
TTSService(AIService)                          tts_service.py:109
├── WordTTSService                             :1882   @deprecated 0.0.105
└── WebsocketTTSService(TTSService,             :1899
    │                   WebsocketService)
    ├── InterruptibleTTSService                :1969
    │   └── InterruptibleWordTTSService        :2062   @deprecated 0.0.105
    ├── WebsocketWordTTSService                :2040   @deprecated 0.0.105
    └── AudioContextTTSService                 :2083   @deprecated 0.0.105
        └── AudioContextWordTTSService         :2121   @deprecated 0.0.105
```

Eight classes declared, five deprecated, **three live**: `TTSService`, `WebsocketTTSService`,
`InterruptibleTTSService`. And there is exactly **one** axis, not two — websocket transport — because
per-turn audio contexts moved *into* `TTSService` itself. The deprecation docstring says so in as
many words: *"Audio context management (previously the main purpose of this class) is now built into
TTSService."* You will see the audio-context machinery in §5 sitting on the root class, right where
that sentence says it is.

The inheritance shape is still a two-branch tree rather than a chain — `WordTTSService` hangs off the
root and `WebsocketTTSService` uses multiple inheritance from `TTSService` **and** `WebsocketService`
— so the outline was right that "the usual shorthand does not cover the tree." It was wrong about
which classes are alive in it.

### 0.2 The Korean word-grouping path is not "an untested assumption". There are four tests.

The outline says:

> Korean is deliberately excluded from the CJK word-grouping paths (both Cartesia and ElevenLabs
> test `base_lang in {"zh", "ja"}`), so it falls through to the space-separated branch and gets
> per-token frames — which matches 어절 spacing but is an **untested assumption in this code**.

The first clause is correct and verified in §8.1. The last clause is false. `tests/test_cartesia_tts.py`
contains four Korean assertions, one of which is exactly the 어절-spacing round trip:

**`tests/test_cartesia_tts.py:69-84`**
```python
def test_cartesia_korean_word_timestamps_preserve_words_and_timestamps():
    assert _process_word_timestamps(
        words=["안녕하세요", "반갑습니다"],
        starts=[0.0, 0.2],
        language="ko",
    ) == [("안녕하세요", 0.0), ("반갑습니다", 0.2)]


def test_cartesia_korean_word_timestamps_do_not_join_latin_and_hangul():
    assert _process_word_timestamps(
        words=["AI", "어시스턴트입니다."],
        starts=[3.7026982, 4.1999383],
        language="ko",
    ) == [("AI", 3.7026982), ("어시스턴트입니다.", 4.1999383)]
```

**`tests/test_cartesia_tts.py:111-124`**
```python
def test_cartesia_korean_timestamp_groups_reassemble_with_spaces():
    assert (
        _concatenate_processed_timestamps(
            [
                (["저는"], [1.6]),
                (["여러분의"], [1.8]),
                (["AI", "어시스턴트입니다."], [3.7, 4.2]),
            ],
            language="ko",
        )
        == "저는 여러분의 AI 어시스턴트입니다."
    )
```

There are two more in `tests/test_aggregated_frame_sequencer.py` (a `TestCJKLanguages` class at `:765` with a
`# --- Korean ---` section, `:774-805`) and a parallel pair in `tests/test_word_completion_tracker.py:1866+`.
Korean word grouping and Korean slot completion are unit-tested at the frame level.

What is genuinely untested — and this is the claim the outline *should* have made — is anything
**behavioural**. No test in this repo sends Korean text to a real TTS provider and listens to the
result. No test asserts that a service declaring `Language.KO: "ko"` actually produces intelligible
Korean, or that its timestamps line up with the audio. §7.4 restates the six-service intersection
with that provenance attached, because the distinction between "the frame plumbing is tested" and
"the audio was never checked" is the whole difference between a claim you can rely on and a claim
you have to verify yourself.

---

## 1. The one number this layer optimises

### 1.1 Start with the arithmetic, not the concept

Say Lina's LLM finishes deciding on a Korean sentence at t=0. The sentence is 4.2 seconds of speech
when spoken. Three timings, all real:

| | Provider A | Provider B |
|---|---|---|
| First byte of audio arrives | 180 ms | 180 ms |
| Leading silence in that audio | 20 ms | 210 ms |
| Total synthesis wall time | 900 ms | 380 ms |
| Playback duration | 4,200 ms | 4,200 ms |

Provider B synthesises the whole sentence in less than half the time. On any byte-level dashboard it
looks identical to A on first-byte and dramatically better on throughput.

The customer hears silence for **200 ms** with A and **390 ms** with B. B is nearly twice as slow to
the ear.

And the 900 ms vs 380 ms synthesis difference does not exist for the listener at all. Audio plays at
1× wall clock. A's remaining 720 ms of synthesis happens *underneath* the 4,200 ms of playback that
is already running. As long as synthesis stays faster than real time — which for any of these
providers it is, by a wide margin — everything after the first audible sample is free.

That is the whole thesis of this layer, and it is why the file spends its complexity budget on
ordering and buffering rather than on speed. Only one interval is exposed:

```
[ LLM decided ] ────────────────────────────────► [ first AUDIBLE sample ]
                        everything after this point is hidden by playback
```

### 1.2 The metric that measures that interval is TTFA, and it is not TTFB

Two metric types exist and they are not interchangeable.

**`src/pipecat/metrics/metrics.py:31-61`**
```python
class TTFBMetricsData(MetricsData):
    """Time To First Byte (TTFB) metrics data.

    Parameters:
        value: TTFB measurement in seconds.
    """

    value: float


class TTFAMetricsData(MetricsData):
    """Time To First Audio (TTFA) metrics data.

    Measures the time from a TTS request to the first audible audio sample,
    i.e. time-to-first-byte plus any leading silence the service pads onto the
    start of its response. ``ttfa`` is reported with its breakdown so consumers
    can see how much of the perceived latency is silence padding rather than
    service response time, without correlating a separate ``TTFBMetricsData``.

    Parameters:
        ttfa: TTFA measurement in seconds (``ttfb`` plus ``leading_silence``).
        ttfb: Time-to-first-byte that TTFA builds on, in seconds. This mirrors
            the standalone ``TTFBMetricsData`` (emitted earlier) for convenience;
            it is not a separate measurement, so don't aggregate both.
        leading_silence: Silence padding before the first audible sample, in
            seconds (``ttfa`` minus ``ttfb``).
    """

    ttfa: float
    ttfb: float
    leading_silence: float
```

`ttfa = ttfb + leading_silence`, and the docstring is explicit that the `ttfb` field is a **copy**
for convenience, not a second measurement — *"don't aggregate both."* If you build a Lina dashboard
that sums TTFB across the call and also sums TTFA, you will double-count the response time.

The log observer prints exactly this breakdown:

**`src/pipecat/observers/loggers/metrics_log_observer.py:153-157`**
```python
        elif isinstance(metrics_data, TTFAMetricsData):
            logger.debug(
                f"📊 {processor_info} TTFA{model_info}: {metrics_data.ttfa}s "
                f"({metrics_data.leading_silence}s leading silence) at {time_sec:.3f}s"
            )
```

That parenthetical is the number Provider B in §1.1 is bad at, and it is the number that never
appears anywhere else.

### 1.3 `leading_silence` is measured, not declared

This is not a provider-supplied field. Pipecat computes it by running an energy-based onset detector
over the audio bytes as they stream in.

**`src/pipecat/processors/metrics/frame_processor_metrics.py:190-238`** (abridged to the arithmetic)
```python
    async def process_ttfa_metrics(self, *, audio: bytes, sample_rate: int, num_channels: int):
        if not self._ttfa_active or sample_rate <= 0:
            return None

        self._ttfa_buffer += audio
        onset = detect_speech_onset(self._ttfa_buffer, sample_rate, num_channels)
        if onset is None:
            # No confirmed onset yet. Bound memory against pathologically long
            # (or never-arriving) silence by giving up past a sane limit.
            max_bytes = int(self._TTFA_MAX_BUFFER_SECONDS * sample_rate * max(num_channels, 1) * 2)
            if len(self._ttfa_buffer) >= max_bytes:
                ...
            return None

        silence_duration = onset / sample_rate
        value = self._last_ttfb_time + silence_duration
```

`_ttfa_active` is set when TTFB stops — i.e. the TTFA scan begins at exactly the moment the first
byte lands:

**`src/pipecat/processors/metrics/frame_processor_metrics.py:179-182`**
```python
        # The first byte has arrived; begin scanning leading silence so TTFA can
        # be reported as TTFB plus the silence duration (see process_ttfa_metrics).
        self._ttfa_active = True
        self._ttfa_buffer = b""
```

And the onset detector itself is a real short-time-energy scan, not a threshold on max amplitude:

**`src/pipecat/audio/utils.py:326-346`**
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
    """Detect the first sample of sustained audible speech in PCM audio.

    Measures short-time RMS energy: the signal is downmixed to mono, scanned
    with a ``frame_ms`` window hopping every ``hop_ms``, and each window's RMS
    is computed with its mean removed (a DC offset carries no energy). Onset is
    the start of the first run of windows whose RMS stays above ``threshold_db``
    for at least ``min_voiced_ms``.
```

Three defaults worth writing down because you will tune against them: onset resolution is **1 ms**
(`hop_ms`), the gate is **−40 dBFS**, and an onset only counts if energy stays above the gate for
**50 ms**. A provider that pads its response with a low-level noise floor rather than digital silence
will still be measured correctly, because −40 dBFS sits above typical noise-floor padding — the
docstring says exactly that: *"Sits above typical TTS noise-floor padding and below voiced onset."*

The call site is one line in the TTS service, run on **every** audio frame until a measurement lands:

**`src/pipecat/services/tts_service.py:1734-1741`**
```python
                elif isinstance(frame, TTSAudioRawFrame):
                    received_audio = True
                    # Set the word-timestamp baseline once, on the first audio chunk.
                    if not timestamps_started:
                        await self.stop_ttfb_metrics()
                        await self.start_word_timestamps()
                        timestamps_started = True
                    await self.process_ttfa_metrics(frame)
```

Note the ordering in those eight lines, because it is dense and it matters for §6: the **first**
audio chunk of a context stops TTFB, sets the word-timestamp PTS baseline, and starts the TTFA scan
— all three anchored to the same event. Word timestamps and audible-onset measurement share a clock
origin by construction.

### 1.4 The three additive components, and who owns each

`TTFA` as the customer experiences it decomposes into three intervals, and this layer controls all
three:

| Component | Owner | Where it is measured | Typical size |
|---|---|---|---|
| (a) aggregation delay | `TTSService` text aggregator | `TextAggregationMetricsData` | 200–300 ms per the docstring (§4.1) |
| (b) provider TTFB | the provider | `TTFBMetricsData` | 100–400 ms |
| (c) leading silence | the provider | `TTFAMetricsData.leading_silence` | 0–250 ms |

(a) is the one people forget, and it is the one you can change without switching vendors. It has its
own metric type:

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

"From the first LLM token to the first complete sentence." That is dead air that has nothing to do
with your TTS provider and everything to do with how you configured `TextAggregationMode`. §4 is
about it.

### 1.5 Where the 1× assumption is enforced

"Audio plays at 1× wall clock" is not a hand-wave; it is the output transport's chunking contract.

**`src/pipecat/transports/base_output.py:130-136`**
```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

Audio leaves the pipeline in fixed 10 ms multiples; the concrete transport's `write_audio_frame`
paces them against the far end ([[ch-05/read]] owns that). Two consequences you should hold on to:

1. Synthesis running faster than real time buys you nothing beyond the first sample — it just fills
   a queue.
2. Synthesis running *slower* than real time is a different failure entirely: the queue drains, and
   the customer hears a gap mid-sentence. Nothing in `TTSService` detects that. It is not a metric
   in this tree.

→ **[tts-streaming.html](figures/tts-streaming.html)** — open it now and stay in the first panel for
§1. Drag the provider-padding slider and watch the two marks separate: TTFB stays pinned while TTFA
slides right. That gap is `leading_silence`, and the panel is the fastest way to internalise why a
provider can be "fast" and sound slow. Come back to the panel's second and third views at §4.5 and
§6.5 respectively.

---

## 2. The class ladder, read correctly

### 2.1 What you subclass, and what the base gives you

`TTSService` is 2,136 lines and declares eight classes. After §0.1, three are live. Here is the root:

**`src/pipecat/services/tts_service.py:109-114`**
```python
class TTSService(AIService):
    """Base class for text-to-speech services.

    Provides common functionality for TTS services including text aggregation,
    filtering, audio generation, and frame management. Supports configurable
    sentence aggregation, silence insertion, and frame processing control.
```

Note what that docstring does *not* say: it does not say "converts text to audio." Text-to-audio is
one abstract method. Everything else the class does is buffering, ordering, and bookkeeping:

- a text aggregator that decides *when* to call the provider (§4)
- a per-context audio FIFO and a single serialization queue that decide *in what order* frames leave
  (§5)
- word-timestamp bookkeeping that decides *when downstream sees each word* (§6)
- three failure guards: zero-audio detection, interruption teardown, text-transform error handling

Read it as a reordering buffer with a synthesis call in the middle, not as a codec.

### 2.2 The abstract method, and the `None` contract

There is exactly one abstract method.

**`src/pipecat/services/tts_service.py:537-554`**
```python
    # Converts the text to audio.
    @abstractmethod
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Run text-to-speech synthesis on the provided text.

        This method must be implemented by subclasses to provide actual TTS functionality.
        The base class logs the synthesized text before invoking this method, so
        implementations should not log it again.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        raise NotImplementedError
        yield  # pragma: no cover
```

The return type is `AsyncGenerator[Frame | None, None]` and the `| None` is the entire HTTP-versus-
websocket distinction. The consumer explains it:

**`src/pipecat/services/tts_service.py:1347-1374`**
```python
    async def tts_process_generator(
        self, context_id: str, generator: AsyncGenerator[Frame | None, None]
    ):
        """Process frames from an async generator, routing them through the audio context.

        All non-None frames yielded by the generator are appended to the audio context
        identified by context_id. The audio context must be created by run_tts (via
        create_audio_context) before the first frame is yielded.

        WebSocket services yield None to signal that audio will arrive via a separate
        receive loop; those services manage context lifetime themselves (via remove_audio_context
        in the receive loop on "done"). HTTP services never yield None and do NOT call
        remove_audio_context in run_tts — the caller (_synthesize_text) closes the context
        after appending any remaining frames (e.g. TTSTextFrame).
        """
        is_yielding_frames = False
        async for frame in generator:
            if frame:
                await self.append_to_audio_context(context_id, frame)
                if isinstance(frame, TTSAudioRawFrame):
                    is_yielding_frames = True

        self._is_yielding_frames_synchronously = is_yielding_frames
```

So a websocket subclass's `run_tts` is: send the text over the socket, `yield None`, return. The
audio arrives later on a receive task that appends to the audio context itself. An HTTP subclass's
`run_tts` yields real `TTSAudioRawFrame`s and the caller closes the context after.

### 2.3 The real HTTP/websocket split is a property, not a class

`supports_processing_metrics` is where the distinction is actually declared, and it is worth reading
the reasoning because it tells you which metric to trust for which service:

**`src/pipecat/services/tts_service.py:422-433`**
```python
    @property
    def supports_processing_metrics(self) -> bool:
        """Whether this service has a meaningful processing-time metric.

        Processing time is measured around :meth:`run_tts`, so it only means
        something when synthesis finishes before ``run_tts`` returns. Services
        that hand the text off and receive audio elsewhere — anything holding a
        persistent connection with its own receive task — return False, since
        the measurement would cover the send and nothing else. TTFB and TTFA
        carry the latency for those.
        """
        return True
```

**`src/pipecat/services/tts_service.py:1925-1934`**
```python
    @property
    def supports_processing_metrics(self) -> bool:
        """Whether this service has a meaningful processing-time metric.

        False: ``run_tts`` sends the text and returns, and audio arrives later
        on the receive task, so there is no synthesis inside the measured
        window. A subclass that instead waits for the server to signal the end
        of synthesis before returning can override this back to True.
        """
        return False
```

Practical consequence for a Lina dashboard: if you pick a websocket provider (and §7 will show you
that five of the six Korean-plus-timestamps services are websocket), `ProcessingMetricsData` for
your TTS service will be **absent**, not zero. Do not build a panel that expects it. TTFB and TTFA
are what you have.

For chunked-HTTP services there is one more constant that shapes TTFA:

**`src/pipecat/services/tts_service.py:473-488`**
```python
    @property
    def chunk_size(self) -> int:
        """Get the recommended chunk size for audio streaming.

        This property indicates how much audio we download (from TTS services
        that require chunking) before we start pushing the first audio
        frame. This will make sure we download the rest of the audio while audio
        is being played without causing audio glitches (specially at the
        beginning). Of course, this will also depend on how fast the TTS service
        generates bytes.

        Returns:
            The recommended chunk size in bytes.
        """
        CHUNK_SECONDS = 0.5
        return int(self.sample_rate * CHUNK_SECONDS * 2)  # 2 bytes/sample
```

That is a **500 ms pre-buffer**, added directly to TTFA, taken deliberately to avoid the glitch
described in the docstring. It is the second-largest single latency decision in the file after
sentence aggregation, and it applies only to the HTTP path. When you see §7's table listing HTTP
variants next to websocket variants of the same provider, this constant is one of the things you are
choosing between.

### 2.4 `InterruptibleTTSService` — the one live subclass that adds behaviour

`WebsocketTTSService` adds connectivity. `InterruptibleTTSService` adds exactly one thing, and it is
a thing you need to understand before [[ch-08/read]]:

**`src/pipecat/services/tts_service.py:1969-2002`**
```python
class InterruptibleTTSService(WebsocketTTSService):
    """Websocket-based TTS service that handles interruptions without word timestamps.

    Designed for TTS services that don't support word timestamps. Handles interruptions
    by reconnecting the websocket when the bot is speaking and gets interrupted.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # True once run_tts has been invoked (TTSStartedFrame pushed) for the
        # current turn but before BotStartedSpeakingFrame confirms playback —
        # the narrow window where _bot_speaking (which only reflects confirmed
        # playback) can't yet tell a reconnect is needed. Kept separate from
        # _bot_speaking so this early, unconfirmed marker never makes a turn
        # that produces no audio look like one that played.
        self._tts_started: bool = False

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        # If the bot is not speaking we don't need to reconnect when the user
        # speaks. If the bot is speaking and the user interrupts we need to
        # reconnect. Captured before calling super(), which clears
        # _bot_speaking as part of its own interruption handling.
        should_reconnect = self._bot_speaking or self._tts_started
        self._tts_started = False
        await super()._handle_interruption(frame, direction)
        if should_reconnect:
            await self._disconnect()
            await self._connect()
```

A full websocket teardown and reconnect on every barge-in. That is expensive — a TLS handshake and a
provider-side session setup in the middle of a conversation — and the class docstring says why it is
necessary: *"Designed for TTS services that don't support word timestamps."* A half-spoken
server-side context that you cannot address by ID cannot be rewound; killing the socket is the only
cancel primitive available.

Which is your first concrete reason to care about §7's six-service list. Of the Korean-capable
services, exactly one — `LmntTTSService` (`lmnt/tts.py:78`) — inherits `InterruptibleTTSService`,
and it is the one Korean-capable websocket service that does *not* emit word timestamps. The two
facts are the same fact.

---

## 3. How text gets in

Before aggregation there is a routing question: which frames does `TTSService` even look at?

**`src/pipecat/services/tts_service.py:759-776`**
```python
        if (
            isinstance(frame, (TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame))
            and frame.skip_tts
        ):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AggregatedTextFrame):
            await self._push_tts_frames(frame)
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, InterimTranscriptionFrame)
            and not isinstance(frame, TranscriptionFrame)
        ):
            await self.start_text_aggregation_metrics()
            await self._process_text_frame(frame)
```

Three branches, in priority order:

1. **`skip_tts` set** → pass through untouched. This is how a processor upstream marks text that
   should reach the context but never be spoken.
2. **Already an `AggregatedTextFrame`** → straight to synthesis, **no aggregation**. Someone
   upstream already decided the boundary.
3. **A plain `TextFrame` that is not a transcription** → aggregate it internally.

Branch 2 is the extension point. `LLMTextProcessor` is a standalone `FrameProcessor` that converts
`LLMTextFrame` → `AggregatedTextFrame` using any `BaseTextAggregator` you hand it:

**`src/pipecat/processors/aggregators/llm_text_processor.py:39-51`**
```python
    def __init__(self, *, text_aggregator: BaseTextAggregator | None = None, **kwargs):
        """Initialize the LLM text processor.

        Args:
            text_aggregator: An optional text aggregator to use for processing LLM text frames. By
                default, a SimpleTextAggregator aggregating by sentence will be used.
            **kwargs: Additional arguments passed to parent class.

        TODO: Allow transformations per aggregation type or all (and deprecate the TTS filters).
        """
        super().__init__(**kwargs)
        self._text_aggregator: BaseTextAggregator = text_aggregator or SimpleTextAggregator()
```

**`src/pipecat/processors/aggregators/llm_text_processor.py:83-94`**
```python
    async def _handle_llm_text(self, in_frame: LLMTextFrame):
        async for aggregation in self._text_aggregator.aggregate(in_frame.text):
            out_frame = AggregatedTextFrame(
                text=aggregation.text,
                aggregated_by=aggregation.type,
                raw_text=aggregation.full_match
                if isinstance(aggregation, PatternMatch)
                else aggregation.text,
            )
            out_frame.append_to_context = True
            out_frame.skip_tts = in_frame.skip_tts
            await self.push_frame(out_frame)
```

Write down that this exists. It is the seam through which a Korean-aware chunker enters a Pipecat
pipeline **without** subclassing any TTS service: implement `BaseTextAggregator`, hand it to
`LLMTextProcessor`, place the processor between `llm` and `tts`, and `TTSService` takes branch 2 and
never runs its own aggregator. §10 and §11 use this.

Two of the classes involved:

**`src/pipecat/utils/text/base_text_aggregator.py:20-26`**
```python
class AggregationType(StrEnum):
    """Built-in aggregation strings."""

    SENTENCE = "sentence"
    TOKEN = "token"
    WORD = "word"
```

**`src/pipecat/frames/frames.py:386-401`**
```python
@dataclass
class AggregatedTextFrame(TextFrame):
    """Text frame representing an aggregation of TextFrames.

    This frame contains multiple TextFrames aggregated together for processing
    or output along with a field to indicate how they are aggregated.

    Parameters:
        aggregated_by: Method used to aggregate the text frames.
        context_id: Unique identifier for the TTS context that generated this text.
        raw_text: The full matched text including start/end pattern delimiters, set when
            this frame was produced from a PatternMatch (e.g. a ``<code>...</code>`` block).
            None for ordinary sentence aggregations.
        will_be_spoken: Whether the TTS service will speak this frame. Set to ``True``
            by the TTS service just before synthesis. Defaults to ``False``.
    """
```

`aggregated_by` is typed `AggregationType | str` — an open string, not a closed enum. You can invent
`"eojeol"` or `"phrase"` as an aggregation type and it will flow through the whole system. That is
the same open-sum-type bet [[ch-02/read]] identified at the frame level, appearing again one level
down.

---

## 4. Sentence aggregation as a named budget line

### 4.1 The mode, and the number in its docstring

**`src/pipecat/services/tts_service.py:82-96`**
```python
class TextAggregationMode(StrEnum):
    """Controls how incoming text is aggregated before TTS synthesis.

    Parameters:
        SENTENCE: Buffer text until sentence boundaries are detected before synthesis.
            Produces more natural speech but adds latency (~200-300ms per sentence).
        TOKEN: Stream text tokens directly to TTS as they arrive.
            Reduces latency but may affect speech quality depending on the TTS provider.
    """

    SENTENCE = "sentence"
    TOKEN = "token"
```

`SENTENCE` is the default, resolved in `__init__`:

**`src/pipecat/services/tts_service.py:296-299`**
```python
        if text_aggregation_mode is None:
            text_aggregation_mode = TextAggregationMode.SENTENCE

        self._text_aggregation_mode: TextAggregationMode = text_aggregation_mode
```

Cartesia's `__init__` repeats the budget with an extra sentence of honesty:

**`src/pipecat/services/cartesia/tts.py:312-319`**
```python
        # By default, we aggregate sentences before sending to TTS. This adds
        # ~200-300ms of latency per sentence (waiting for the sentence-ending
        # punctuation token from the LLM). Setting
        # text_aggregation_mode=TextAggregationMode.TOKEN streams tokens
        # directly, which reduces latency. Streaming quality is good but less
        # tested than sentence aggregation.
        # TODO: Consider making TOKEN the default for Cartesia in 1.0.
```

*"Streaming quality is good but less tested than sentence aggregation."* That is the framework
telling you the honest state of TOKEN mode. Take it at face value: TOKEN is real, it is supported,
and it has less mileage.

### 4.2 The lookahead rule and its exact cost

`SENTENCE` mode does not call the boundary detector when it sees punctuation. It calls it when it
sees the first **non-whitespace character after** the punctuation.

**`src/pipecat/utils/text/simple_text_aggregator.py:78-121`**
```python
    async def _check_sentence_with_lookahead(self, char: str) -> Aggregation | None:
        """Check for sentence boundaries using lookahead logic.

        This method implements the core sentence detection logic with lookahead.
        When sentence-ending punctuation is detected, it waits for the next
        non-whitespace character before calling NLTK. This disambiguates cases
        like "$29." (not a sentence) vs "$29. Next" (sentence ends at period).
        Whitespace alone is not meaningful lookahead since it appears in both
        cases. Instead, the first non-whitespace character after the punctuation
        is used to confirm the sentence boundary.
        """
        # If we need lookahead, check if we now have non-whitespace
        if self._needs_lookahead:
            # Check if the new character is non-whitespace
            if char.strip():
                # We have meaningful lookahead, call NLTK
                self._needs_lookahead = False
                eos_marker = match_endofsentence(self._text)

                if eos_marker:
                    # NLTK confirmed a sentence - return it
                    result = self._text[:eos_marker]
                    self._text = self._text[eos_marker:]
                    return Aggregation(text=result.strip(" "), type=AggregationType.SENTENCE)
                # No sentence found - keep accumulating
                return None
            # Still whitespace, keep waiting
            return None

        # Check if we just added sentence-ending punctuation
        if self._text and self._text[-1] in SENTENCE_ENDING_PUNCTUATION:
            # Mark that we need lookahead (don't call NLTK yet)
            self._needs_lookahead = True

        return None
```

The cost is **one LLM token per sentence**, always, by design. If your LLM streams at 40 tokens/sec,
that is 25 ms. If it streams at 8 tokens/sec on a long Korean generation, that is 125 ms. It is
additive with the provider's TTFB and it is invisible unless you are looking at
`TextAggregationMetricsData`.

What it buys is not a "nicer" split; it is the ability to distinguish a decimal point from a full
stop when the two are the same character. The examples in the docstring are `$29.` versus `$29. Next`
— and if you have been reading [[rtv-vad-chunking]], you already recognise that as the same problem
`KoreanPhraseChunker._is_safe_period` solves with an explicit guard. §9.3 runs both through the same
inputs.

### 4.3 The end-of-turn flush closes the loop

The last sentence of an LLM turn has no following token to confirm it. That case is handled
explicitly at the frame level:

**`src/pipecat/services/tts_service.py:787-801`**
```python
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            # Flush any remaining text (including text waiting for lookahead)
            remaining = await self._text_aggregator.flush()
            # Stop the aggregation metric (no-op if already stopped on first sentence).
            await self.stop_text_aggregation_metrics()
            if remaining:
                await self._push_tts_frames(
                    AggregatedTextFrame(
                        remaining.text,
                        remaining.type,
                        raw_text=remaining.full_match
                        if isinstance(remaining, PatternMatch)
                        else remaining.text,
                    )
                )
```

So the lookahead tax is paid on sentences 1..n−1 of a turn and **not** on sentence n. For Lina's
typical 2–4 sentence responses that is 1–3 extra tokens of latency per turn, all of them before the
first audible sample only if they occur in sentence 1 — which they do not, because sentence 1's
boundary confirmation *is* the lookahead. Read that again: the lookahead tax lands **exactly on the
number you care about**, TTFA, because it delays sentence 1.

### 4.4 `match_endofsentence` and the Korean fallback

**`src/pipecat/utils/string.py:151-202`**
```python
def match_endofsentence(text: str) -> int:
    """Find the position of the end of a sentence in the provided text.

    This function uses NLTK's sentence tokenizer to detect sentence boundaries
    in the input text, combined with punctuation verification to ensure that
    single tokens without proper sentence endings aren't considered complete sentences.
    """
    text = text.rstrip()

    if not text:
        return 0

    # Use NLTK's sentence tokenizer to find sentence boundaries
    sentences = _sent_tokenizer()(text)

    if not sentences:
        return 0

    first_sentence = sentences[0]

    if len(sentences) == 1 and first_sentence == text:
        if text and text[-1] in SENTENCE_ENDING_PUNCTUATION:
            return len(text)
        # Fallback for languages not supported by NLTK (e.g., Japanese, Chinese,
        # Korean, Hindi, Arabic). NLTK returned the entire text as a single
        # sentence, and the last character is not sentence-ending punctuation
        # (it's a lookahead character). Scan for unambiguous non-Latin sentence-
        # ending punctuation that doesn't need NLTK's disambiguation.
        for i, ch in enumerate(text):
            if ch in UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION:
                return i + 1
        return 0

    # If there are multiple sentences, the first one is complete by definition
    # (NLTK found a boundary, so there must be proper punctuation)
    if len(sentences) > 1:
        return len(first_sentence)

    # Single sentence that doesn't equal the full text means incomplete
    return 0
```

Korean is named in that comment, and the set it falls back to is derived by subtraction:

**`src/pipecat/utils/string.py:118-127`**
```python
# Latin punctuation that NLTK handles well — these need NLTK's disambiguation
# because "." can appear in abbreviations, decimals, etc.
_LATIN_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = frozenset({".", "!", "?", ";", "…"})

# Non-Latin sentence-ending punctuation that is always unambiguous and never needs
# NLTK's disambiguation logic. Used as a fallback when NLTK doesn't support the
# language (e.g., Japanese, Chinese, Korean, Hindi, Arabic).
UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = (
    SENTENCE_ENDING_PUNCTUATION - _LATIN_SENTENCE_ENDING_PUNCTUATION
)
```

The full-width Korean-relevant members come from `SENTENCE_ENDING_PUNCTUATION` at `:73-116`, which
groups them under the comment *"East Asian punctuation (Chinese (Traditional & Simplified), Japanese,
Korean)"*: `。？！；．｡`.

**The trap.** That fallback fires only when the ASCII period path fails. Korean written with ASCII
`.` `?` `!` — which is how essentially every Korean LLM writes and how Lina's LLM will write —
goes through NLTK's English Punkt model, not through the fallback. So the sentence in the docstring
("used as a fallback when NLTK doesn't support the language") describes a path your Korean bot will
mostly not take. Which raises the obvious question: does English Punkt actually work on Korean?

### 4.5 Measured, not assumed: six Korean inputs through the real aggregator

The repo's own Korean assertions in `tests/test_utils_string.py` only cover full-width punctuation:

**`tests/test_utils_string.py:94-99`**
```python
        korean_sentences = [
            "안녕하세요。",  # Korean with ideographic period
            "어떻게 지내세요？",  # Korean question
        ]
        for sentence in korean_sentences:
            assert match_endofsentence(sentence), f"Failed for Korean: {sentence}"
```

**`tests/test_utils_string.py:176-177`**
```python
        # Korean: sentence + lookahead character
        assert match_endofsentence("안녕하세요。다") == 6
```

Nothing there covers Korean with ASCII punctuation, which is the case that matters. So I ran it. The
script below reimplements `match_endofsentence` and `SimpleTextAggregator._check_sentence_with_lookahead`
verbatim (same NLTK call, same lookahead state machine) and feeds it token streams that look like
Lina's output. You can run it yourself; it needs only `nltk` with `punkt_tab`.

```python
# reimplementation of src/pipecat/utils/string.py:151 and
# src/pipecat/utils/text/simple_text_aggregator.py:78, fed synthetic token streams
from nltk.tokenize import sent_tokenize

SENTENCE_ENDING_PUNCTUATION = set(".!?;…。？！；．｡।॥؟؛۔؏၊။។៕໌།༎։՜՞።፧፨")
UNAMB = SENTENCE_ENDING_PUNCTUATION - set(".!?;…")

def match_endofsentence(text):
    text = text.rstrip()
    if not text: return 0
    s = sent_tokenize(text)
    if not s: return 0
    if len(s) == 1 and s[0] == text:
        if text[-1] in SENTENCE_ENDING_PUNCTUATION: return len(text)
        for i, ch in enumerate(text):
            if ch in UNAMB: return i + 1
        return 0
    if len(s) > 1: return len(s[0])
    return 0
```

Result, one line per stream, showing which token triggered the emit:

```
--- decimal 29.99
   tokens: "월","보험","료는"," 29",".","99","만","원입니다","."," 다","음","은"
   token 10 ' 다' -> EMIT ['월보험료는 29.99만원입니다.']
   flush -> '다음은'
--- greeting
   token 3 ' 리나' -> EMIT ['안녕하세요.']
   flush -> '리나입니다.'
--- korean decimal 1.5
   token 7 ' 가' -> EMIT ['월 1.5만원입니다.']
   flush -> '가입'
--- gpt-4.1
   token 9 ' 네' -> EMIT ['저는 gpt-4.1을 씁니다.']
   flush -> '네'
--- 1,000
   token 7 ' 다' -> EMIT ['보험료는 1,000원입니다.']
   flush -> '다'
--- fullwidth
   token 2 '고객님' -> EMIT ['안녕하세요。']
   flush -> '고객님'
```

Four things fall out of that, and they are all facts about this code, not opinions:

1. **The lookahead cost is visible.** In every case, the emit fires on the token *after* the sentence
   is complete — token 10, not token 8; token 3, not token 2. That is the 200–300 ms.
2. **`1.5`, `gpt-4.1` and `1,000` do not split.** Pipecat has no explicit numeric or identifier
   guard, and it does not need one for these inputs: NLTK's Punkt abbreviation/number model plus the
   non-whitespace lookahead already refuses the cut. `1.5` is protected because `match_endofsentence`
   is never called until a non-whitespace character follows the `.`, and by then the buffer reads
   `1.5` and Punkt returns a single sentence not ending in punctuation → 0.
3. **The full-width path works and is 1 token cheaper** — `안녕하세요。` emits on token 2, not token 3,
   because the fallback scan does not need Punkt to agree.
4. **English Punkt handles Korean-with-ASCII-punctuation on these inputs.** That is not a guarantee;
   Punkt is an unsupervised model trained on English, and its abbreviation list has no Korean
   knowledge. I found a contrived counterexample — `match_endofsentence("월 보험료는 29. 000원")`
   returns 10, cutting after `29.` — but it takes a digit-space-digit pattern to produce it. Treat
   "Punkt works on Korean" as *unrefuted on realistic inputs*, not proven.

→ Second panel of **[tts-streaming.html](figures/tts-streaming.html)**. It runs the same Korean
sentence through both paths side by side and marks the cut points. Use it after you have read §9,
not before — the panel is a comparison of two algorithms and it only means something once you know
what each one is doing.

### 4.6 The decoy: `processors/aggregators/sentence.py`

Search this repo for "sentence aggregator" and you will find a file that has nothing to do with any
of the above.

**`src/pipecat/processors/aggregators/sentence.py:18-31`**
```python
class SentenceAggregator(FrameProcessor):
    """Aggregates text frames into complete sentences.

    This processor accumulates incoming text frames until a sentence-ending
    pattern is detected, then outputs the complete sentence as a single frame.
    Useful for ensuring downstream processors receive coherent, complete sentences
    rather than fragmented text.

    Frame input/output::

        TextFrame("Hello,") -> None
        TextFrame(" world.") -> TextFrame("Hello, world.")
    """
```

63 lines, a standalone `FrameProcessor`, no lookahead, no `AggregationType`, no `PatternMatch`
handling, and — critically — **`TTSService` never touches it**. Grep confirms: `tts_service.py`
imports `SimpleTextAggregator` from `pipecat.utils.text.simple_text_aggregator` (`:59`) and nothing
from `processors.aggregators.sentence`.

Its `process_frame` calls `match_endofsentence` the moment punctuation appears:

**`src/pipecat/processors/aggregators/sentence.py:54-58`**
```python
        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
            if match_endofsentence(self._aggregation):
                await self.push_frame(TextFrame(self._aggregation))
                self._aggregation = ""
```

No lookahead state. Which means it *will* split `$29.` mid-price, and it *will* split `1.5` if Punkt
is fooled. If an LLM writes you Pipecat code that inserts a `SentenceAggregator` before your TTS to
"improve chunking," it has handed you a second, worse aggregator in front of the good one. Delete it.

### 4.7 The metric that makes aggregation delay visible

**`src/pipecat/services/tts_service.py:448-462`**
```python
    async def start_text_aggregation_metrics(self):
        """Start text aggregation metrics if not already started.

        Only starts the metric once per LLM response. Skipped when streaming
        tokens since per-token aggregation time is not meaningful.
        """
        if self._is_streaming_tokens or self._text_aggregation_metrics_started:
            return
        self._text_aggregation_metrics_started = True
        await super().start_text_aggregation_metrics()

    async def stop_text_aggregation_metrics(self):
        """Stop text aggregation metrics and reset the started flag."""
        self._text_aggregation_metrics_started = False
        await super().stop_text_aggregation_metrics()
```

Started on the first `TextFrame` of a response (`:771`), stopped on the first non-TOKEN aggregate
(`:1099-1101`). One measurement per LLM turn, covering exactly "first token → first complete
sentence." In TOKEN mode it is skipped entirely, which is correct — there is no aggregation to
measure — but it also means **you cannot A/B SENTENCE against TOKEN on this metric**. Switching modes
makes the number disappear rather than go to zero. To compare, you have to watch TTFA.

---

## 5. Ordering: one FIFO, and the two cursors that are not the same cursor

This section is the "reordering buffer" claim made concrete. Skip it and §6 will not make sense.

### 5.1 The serialization queue holds three kinds of thing

**`src/pipecat/services/tts_service.py:395-407`**
```python
        # Single FIFO queue that serializes everything the TTS service emits downstream.
        # Items can be:
        #   str   – an audio context ID: process the per-context audio queue in full before
        #           moving on (see _handle_audio_context).
        #   Frame – a non-system downstream frame (e.g. AggregatedTextFrame, FooFrame) that
        #           must be emitted in-order relative to surrounding audio contexts.
        #   None  – shutdown sentinel (sent by stop()).
        # Created once here so it survives interruptions: on interruption we call reset()
        # which drops non-UninterruptibleFrame items while keeping uninterruptible ones
        # (e.g. FunctionCallResultFrame) that must not be lost mid-flight.
        self._serialization_queue: FrameQueue = FrameQueue(
            frame_getter=lambda item: item if isinstance(item, Frame) else None
        )
```

And the single consumer:

**`src/pipecat/services/tts_service.py:1624-1656`**
```python
    async def _audio_context_task_handler(self):
        """Drain the serialization queue, preserving downstream frame order.

        The queue carries three kinds of items (see _create_audio_context_task):

        * str  – audio context ID: block until all audio for that context has been
                 pushed downstream, then call on_audio_context_completed().
        * Frame – a non-system downstream frame that must be emitted at this exact
                  position in the output stream (e.g. AggregatedTextFrame preceding
                  its audio, or an arbitrary frame that arrived between two speak frames).
        * None – shutdown sentinel; exit the loop once reached.
        """
        running = True
        while running:
            context_value = await self._serialization_queue.get()
            if isinstance(context_value, Frame):
                await self.push_frame(context_value)
            elif isinstance(context_value, str):
                context_id = context_value
                self._playing_context_id = context_id

                # Process the audio context until the context doesn't have more
                # audio available (i.e. we find None).
                await self._handle_audio_context(context_id)

                # We just finished processing the context, so we can safely remove it.
                del self._audio_contexts[context_id]
                await self.on_audio_context_completed(context_id=context_id)
                self.reset_active_audio_context()

            self._serialization_queue.task_done()
```

Note the blocking `await self._handle_audio_context(context_id)` inside the loop. **One** consumer
task, draining **one** context to completion before touching the next. That is the ordering
guarantee, and it is why a `FooFrame` you push between two `TTSSpeakFrame`s cannot overtake the first
utterance's audio — the `else` branch of `process_frame` routes it into the same FIFO:

**`src/pipecat/services/tts_service.py:901-910`**
```python
        else:
            if direction == FrameDirection.DOWNSTREAM and not isinstance(frame, SystemFrame):
                # Route non-system downstream frames through the serialization queue so they
                # are emitted in the same order they arrive relative to any audio contexts that
                # are already queued (e.g. a FooFrame sent right after a TTSSpeakFrame must
                # not overtake the TTSStartedFrame / TTSAudioRawFrame / TTSStoppedFrame
                # sequence from that speak frame).
                await self._serialization_queue.put(frame)
            else:
                await self.push_frame(frame, direction)
```

`SystemFrame` is excluded — which is the out-of-band priority path [[ch-04/read]] §4 spent a whole
section on, showing up here as the exception clause in an `if`. An `InterruptionFrame` is a
`SystemFrame` and does not queue behind three seconds of Korean audio.

### 5.2 Two cursors, cleared at different times

**`src/pipecat/services/tts_service.py:375-393`**
```python
        # _turn_context_id:
        #   Set on LLMFullResponseStartFrame and cleared after LLMFullResponseEndFrame
        #   is processed (i.e. after flush). All sentences within one LLM turn share
        #   this ID so the TTS service groups them into a single audio context.
        #   Temporarily set to None for TTSSpeakFrame utterances, which are standalone.
        #
        # _playing_context_id (playback-side cursor):
        #   Set by _audio_context_task_handler as it dequeues contexts for playback.
        #   Cleared by reset_active_audio_context() on interruption. Used by
        #   has_active_audio_context() and get_active_audio_context_id().
        #
        # Both fields may hold the same value during a turn, but
        # they clear at different times: _turn_context_id is cleared when the LLM turn
        # ends (synthesis done) while _playing_context_id remains set until the audio
        # finishes playing. Merging them would null out the playback cursor prematurely.
        self._playing_context_id: str | None = None
        self._turn_context_id: str | None = None
```

Synthesis-side cursor and playback-side cursor. They diverge for the entire duration of the audio
tail: the LLM finished, `_turn_context_id` is `None`, and there are still 2.8 seconds of Korean in
the pipe with `_playing_context_id` still set.

That divergence is the reason Pipecat can answer "what has the customer actually heard?" — and
[[ch-03/read]] §7.2 showed you realtime_voice answering the same question with a different mechanism
(`AudioTextPlayoutLedger`'s `_next_sample` / `_played_sample` pair, per [[rtv-vad-chunking]]). Two
cursors in both systems. Different units: context IDs here, sample counts there.

### 5.3 Context reuse: one context per LLM turn

**`src/pipecat/services/tts_service.py:526-535`**
```python
    def create_context_id(self) -> str:
        """Generate or reuse a context ID based on concurrent TTS support.

        Returns:
            A context ID string for the TTS request.
        """
        if self._reuse_context_id_within_turn and self._turn_context_id:
            self._refresh_audio_context(self._turn_context_id)
            return self._turn_context_id
        return str(uuid.uuid4())
```

`reuse_context_id_within_turn` defaults to `True` (`:189`). So every sentence of a single LLM turn
lands in **one** audio context, which is what lets the word-timestamp PTS baseline stay continuous
across sentence boundaries within a turn (§6.3).

`_refresh_audio_context` is a keepalive against the idle timeout:

**`src/pipecat/services/tts_service.py:1609-1612`**
```python
    def _refresh_audio_context(self, context_id: str):
        """Signal that the audio context is still in use, resetting the timeout."""
        if self.audio_context_available(context_id):
            self._audio_contexts[context_id].put_nowait(TTSService._CONTEXT_KEEPALIVE)
```

### 5.4 `_handle_audio_context` — the loop where everything lands

This is the densest 70 lines in the file. It is worth reading whole.

**`src/pipecat/services/tts_service.py:1710-1781`**
```python
    async def _handle_audio_context(self, context_id: str):
        """Process items from an audio context queue until it is exhausted."""
        queue = self._audio_contexts[context_id]
        running = True
        timestamps_started = False
        received_audio = False
        should_push_stop_frame = False
        while running:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=self._stop_frame_timeout_s)
                if frame is TTSService._CONTEXT_KEEPALIVE:
                    # Context is still in use, reset the timeout.
                    continue
                elif frame is None:
                    running = False
                elif isinstance(frame, _WordTimestampEntry):
                    # Route word timestamps through _add_word_timestamps so they are
                    # processed in playback order alongside audio frames.
                    await self._add_word_timestamps(
                        [(frame.word, frame.timestamp)],
                        frame.context_id,
                        includes_inter_frame_spaces=frame.includes_inter_frame_spaces,
                    )
                    continue
                elif isinstance(frame, TTSAudioRawFrame):
                    received_audio = True
                    # Set the word-timestamp baseline once, on the first audio chunk.
                    if not timestamps_started:
                        await self.stop_ttfb_metrics()
                        await self.start_word_timestamps()
                        timestamps_started = True
                    await self.process_ttfa_metrics(frame)

                if frame:
                    if isinstance(frame, TTSStartedFrame):
                        should_push_stop_frame = self._push_stop_frames
                        # Stamp appropriate append_to_context value onto every
                        # TTSStartedFrame here — the single point both
                        # base-class- and subclass-emitted started frames pass
                        # through.
                        tts_context = self._tts_contexts.get(context_id)
                        if tts_context is not None:
                            frame.append_to_context = tts_context.append_to_context
                    elif isinstance(frame, TTSStoppedFrame):
                        # Checking if we have any remaining spoken slots before pushing the TTSStoppedFrame
                        await self._apply_force_complete(context_id)

                        should_push_stop_frame = False
                        # Setting the last word timestamp as the TTSStoppedFrame PTS
                        if not frame.pts:
                            frame.pts = self._word_last_pts
```

Five things happen in that one queue, interleaved in **playback order**: audio frames, word-timestamp
entries, the started/stopped brackets, the keepalive sentinel, and the `None` end-of-context marker.
The interleaving is the point. A word timestamp queued behind three audio chunks is emitted after
those three audio chunks, not when the provider's socket happened to deliver it.

The timeout branch is the silent-provider path:

**`src/pipecat/services/tts_service.py:1766-1781`**
```python
            except TimeoutError:
                # We didn't get audio, so let's consider this context finished.
                logger.trace(f"{self} time out on audio context {context_id}")
                if should_push_stop_frame and self._push_stop_frames:
                    await self.push_frame(TTSStoppedFrame(context_id=context_id))
                    should_push_stop_frame = False
                break

        await self._apply_force_complete(context_id)

        if should_push_stop_frame and self._push_stop_frames:
            await self.push_frame(TTSStoppedFrame(context_id=context_id))

        await self._maybe_reset_word_timestamps(context_id)

        await self._record_context_audio_outcome(context_id, received_audio)
```

`self._stop_frame_timeout_s` defaults to **3.0 seconds** (`:158`). A context that produces nothing for
three seconds is declared finished. That constant is one of the two numbers in this file you should
consider changing for a telephony bot, and §11 says why.

### 5.5 `TTSSpeakFrame` — the out-of-band utterance

`TTSSpeakFrame` is how you make the bot say something that did not come from the LLM — a filler, a
hold message, a DTMF prompt.

**`src/pipecat/frames/frames.py:794-810`**
```python
@dataclass
class TTSSpeakFrame(DataFrame):
    """Frame containing text that should be spoken by TTS.

    A frame that contains text that should be spoken by the TTS service
    in the pipeline (if any).

    Parameters:
        text: The text to be spoken.
        append_to_context: Whether the spoken text should be appended to the LLM
            context. Defaults to True. (Note that, as of version 1.4.0, ``None`` —
            the previous default — is no longer a supported value.)
    """

    text: str
    append_to_context: bool = True
```

Its handling does something subtle — it *temporarily nulls the turn cursor* so the utterance gets a
fresh UUID and does not join the LLM turn's context:

**`src/pipecat/services/tts_service.py:842-861`**
```python
        elif isinstance(frame, TTSSpeakFrame):
            # Store if we were processing text or not so we can set it back.
            processing_text = self._processing_text
            saved_sent_non_whitespace = self._sent_non_whitespace_in_context
            self._sent_non_whitespace_in_context = False
            # TTSSpeakFrame is independent — temporarily clear the turn context
            # so create_context_id() generates a fresh UUID for this utterance.
            saved_turn_context_id = self._turn_context_id
            self._turn_context_id = None
            # Creating a new context_id for the TTS request.
            self._turn_context_id = self.create_context_id()
            await self.on_turn_context_created(self._turn_context_id)
            # If we are not receiving text from the LLM, we can assume that the SpeakFrame should be automatically added to the context
            push_assistant_aggregation = frame.append_to_context and not self._llm_response_started
```

...and restores the saved cursor at the end (`:874-876`). This is the mechanism you would use for
Lina's 추임새 — "네, 잠시만요" while a tool call runs — with `append_to_context=False` so the filler
does not pollute the conversation history.

### 5.6 What an interruption clears

**`src/pipecat/services/tts_service.py:1030-1056`**
```python
    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        self._processing_text = False
        self._sent_non_whitespace_in_context = False
        self._bot_speaking = False
        await self._text_aggregator.handle_interruption()
        for filter in self._text_filters:
            await filter.handle_interruption()

        self._llm_response_started = False
        self._streamed_text = ""
        self._text_aggregation_metrics_started = False
        self._aggregated_frame_sequencer.clear()  # discard all pending slots on interruption
        self._pending_llm_response_end_frames.clear()
        await self.reset_word_timestamps()

        await self._stop_audio_context_task()
        # Drops non-UninterruptibleFrame items while keeping uninterruptible ones
        # (e.g. FunctionCallResultFrame) that must not be lost mid-flight.
        self._serialization_queue.reset()
        audio_contexts = self.get_audio_contexts()
        if audio_contexts:
            for ctx_id in audio_contexts:
                await self.on_audio_context_interrupted(context_id=ctx_id)
        self.reset_active_audio_context()
        self._turn_context_id = None
        self._word_last_pts = 0
        self._create_audio_context_task()
```

Thirteen pieces of state reset, one hook (`on_audio_context_interrupted`) fired per live context so
a websocket subclass can send a provider-side cancel, and the consumer task destroyed and recreated.
The `_serialization_queue.reset()` preserving `UninterruptibleFrame`s is the same mechanism
[[ch-04/read]] §4.3 described at the processor level, applied to this service's private queue.

[[ch-08/read]] owns what happens *around* this method. For now note only the last line of the
docstring above `on_audio_context_interrupted`:

**`src/pipecat/services/tts_service.py:1848-1861`**
```python
    async def on_audio_context_interrupted(self, context_id: str):
        """Called when an audio context is cancelled due to an interruption.

        Override this in a subclass to perform provider-specific cleanup (e.g.
        sending a cancel/close message over the WebSocket) when the bot is
        interrupted mid-speech.  The audio context task has already been stopped
        and the active context has **not** yet been reset when this is called,
        so ``context_id`` reflects the context that was cut short.
        """
        pass
```

That is the barge-in extension point for a custom Korean provider.

---

## 6. Word timestamps: what actually exists

### 6.1 There is no word frame. State it plainly.

Grep `src/` for `WordBoundaryFrame` or `TTSWordFrame` at commit `0cbf9c5b` and you get **nothing**.
`frames.py` declares 132 classes with "Frame" in the name and none of them is a word-boundary frame.

If you read a blog post or an LLM-generated migration plan that mentions either name, it is
hallucinating. Word-level information is carried by exactly two things.

### 6.2 Thing one: `TTSTextFrame` with a `pts`

**`src/pipecat/frames/frames.py:416-421`**
```python
@dataclass
class TTSTextFrame(AggregatedTextFrame):
    """Text frame generated by Text-to-Speech services."""

    pass
```

Five lines. All the machinery is inherited: `text`, `aggregated_by`, `context_id`,
`append_to_context`, `includes_inter_frame_spaces` from `AggregatedTextFrame` / `TextFrame`, and
`pts` from the root `Frame`:

**`src/pipecat/frames/frames.py:65-89`**
```python
class Frame:
    """Base frame class for all frames in the Pipecat pipeline.

    Parameters:
        id: Unique identifier for the frame instance.
        name: Human-readable name combining class name and instance count.
        pts: Presentation timestamp in nanoseconds.
        ...
    """

    id: int = field(init=False)
    name: str = field(init=False)
    pts: int | None = field(init=False)
```

**Presentation timestamp, in nanoseconds, on the universal base frame.** That is the design decision.
Rather than inventing a word-boundary type, Pipecat reuses the field every frame already has and
emits one `TTSTextFrame` per word token, stamped with when that word will be *spoken*.

The frames are built here, and every one is stamped:

**`src/pipecat/utils/context/aggregated_frame_sequencer.py:859-882`**
```python
    def _build_word_frame(
        self,
        text: str,
        pts: int,
        context_id: str | None,
        raw_text: str | None = None,
        suppress_in_context: bool = False,
        includes_inter_frame_spaces: bool = False,
    ) -> Frame:
        """Build a TTSTextFrame with all standard word-timestamp attributes set."""
        frame = TTSTextFrame(text, aggregated_by=AggregationType.WORD)
        frame.pts = pts
        frame.context_id = context_id
        if suppress_in_context:
            frame.append_to_context = False
        else:
            frame.append_to_context = (
                self._context_append_to_context.get(context_id, True)
                if context_id is not None
                else True
            )
        frame.raw_text = raw_text
        frame.includes_inter_frame_spaces = includes_inter_frame_spaces
        return frame
```

`aggregated_by=AggregationType.WORD` — the third enum member from §3, and this is its only producer.

### 6.3 Thing two: `AggregatedTextProgressFrame`

**`src/pipecat/frames/frames.py:423-444`**
```python
@dataclass
class AggregatedTextProgressFrame(DataFrame):
    """Progress frame emitted alongside each TTSTextFrame during word-timestamp playback.

    Carries the spoken-so-far / remaining-text breakdown for the active
    AggregatedTextFrame slot, enabling downstream consumers (e.g. the RTVI
    observer) to do word-level highlighting without coupling to internal
    sequencer state.

    Parameters:
        segment_id: ID of the AggregatedTextFrame being spoken.
        context_id: TTS context this frame belongs to.
        text: Full original text of the source AggregatedTextFrame.
        aggregated_by: Aggregation type of the source AggregatedTextFrame.
        accumulated_text: Text already spoken in this slot (including the current word).
        remaining_text: Text not yet spoken in this slot.
    """

    segment_id: int
    context_id: str | None
    text: str
    aggregated_by: AggregationType | str
    accumulated_text: str
    remaining_text: str
```

`accumulated_text` / `remaining_text` is the split you want for a live transcript UI, and — more
importantly for Lina — it is a **ready-made "what has the customer heard so far"** payload emitted
per word, without anyone having to compute it.

Compare the shape, not the quality, against realtime_voice's answer from [[rtv-vad-chunking]] as
recorded in [[ch-03/read]] §7.2: `AudioTextPlayoutLedger.audible_text()` computes
`text[: int(len(text) * ratio)]` from a sample-count ratio. Both produce a prefix of the spoken text.
One derives it from a per-word timestamp emitted by the provider; the other derives it from a linear
character-per-sample interpolation over a sample counter. Different inputs, different resolutions,
same output type. What each requires from the TTS is the thing to hold on to: one needs a provider
that emits timestamps, the other needs only a byte count.

### 6.4 The internal entry that never leaves the service

**`src/pipecat/services/tts_service.py:99-107`**
```python
@dataclass
class _WordTimestampEntry:
    """Internal: word timestamp routed through an audio context queue."""

    word: str
    timestamp: float
    context_id: str
    includes_inter_frame_spaces: bool = False
```

Underscore-prefixed, never pushed. Its whole job is to be **queueable** — to sit in the same
`asyncio.Queue` as the audio frames so that §5.4's loop pulls it out in playback order:

**`src/pipecat/services/tts_service.py:1434-1454`**
```python
        if pre_merge_tokens:
            word_times = merge_punct_tokens(word_times)

        ifs = bool(includes_inter_frame_spaces)
        if context_id and self.audio_context_available(context_id):
            for word, timestamp in word_times:
                await self.append_to_audio_context(
                    context_id,
                    _WordTimestampEntry(
                        word=word,
                        timestamp=timestamp,
                        context_id=context_id,
                        includes_inter_frame_spaces=ifs,
                    ),
                )
        elif context_id is not None:
            logger.trace(f"Dropping stale words for context {context_id}; word times: {word_times}")
        else:
            await self._add_word_timestamps(
                word_times=word_times, context_id=context_id, includes_inter_frame_spaces=ifs
            )
```

Note the middle branch: a word timestamp arriving for a context that no longer exists is **dropped
with a trace log**. That is the late-delivery case — a provider's socket delivering timestamps for a
context you cancelled two seconds ago. Silent, correct, and invisible unless you enable trace logging.

### 6.5 The PTS baseline

**`src/pipecat/services/tts_service.py:1380-1397`**
```python
    async def start_word_timestamps(self):
        """Start tracking word timestamps from the current time."""
        if self._initial_word_timestamp == -1:
            current_time = self.get_clock().get_time()
            # Initialize word timestamp tracking. Use the last emitted timestamp if it's ahead
            # of current time to maintain continuity across overlapping audio contexts.
            self._initial_word_timestamp = (
                self._word_last_pts if self._word_last_pts > current_time else current_time
            )
            # If we cached some initial word times (because we didn't receive
            # audio), let's add them now.
            if self._initial_word_times:
                cached = self._initial_word_times.copy()
                self._initial_word_times = []
                for word, timestamp_seconds, ctx_id, ifs in cached:
                    await self._add_word_timestamps(
                        [(word, timestamp_seconds)], ctx_id, includes_inter_frame_spaces=ifs
                    )
```

Called once per context, on the first audio chunk (§1.3). The provider gives seconds-since-start-of-
*its* response; Pipecat converts to absolute pipeline nanoseconds by adding this baseline:

**`src/pipecat/services/tts_service.py:1472-1490`**
```python
        for word, timestamp in word_times:
            ts_ns = seconds_to_nanoseconds(timestamp)
            if self._initial_word_timestamp == -1:
                # Cache until we have audio and can compute PTS.
                self._initial_word_times.append(
                    (word, timestamp, context_id, includes_inter_frame_spaces)
                )
            else:
                pts = self._initial_word_timestamp + ts_ns
                # Build TTSTextFrame(s) for this word token, advancing the active
                # slot's tracker and flushing any skipped frames now unblocked.
                for f in self._aggregated_frame_sequencer.process_word(
                    word, pts, context_id, includes_inter_frame_spaces
                ):
                    if isinstance(f, TTSTextFrame):
                        # The sequencer stamps every word frame it builds.
                        assert f.pts is not None
                        self._word_last_pts = f.pts
                    await self.push_frame(f)
```

The caching branch handles timestamps that arrive *before* the first audio chunk — some providers
send alignment first. They are held and flushed once the baseline exists.

### 6.6 The hook ch-08 needs: the output transport's clock queue

Here is what makes `pts` more than metadata. `transport.output()` sorts frames by presentation
timestamp and **sleeps until each one is due**.

**`src/pipecat/transports/base_output.py:394-398`**
```python
        elif isinstance(frame, TTSStoppedFrame):
            await sender.handle_tts_stopped(frame)
        elif frame.pts:
            await sender.handle_timed_frame(frame)
        else:
            await sender.handle_sync_frame(frame)
```

**`src/pipecat/transports/base_output.py:642-656`**
```python
        async def handle_timed_frame(self, frame: Frame):
            """Handle frames with presentation timestamps.

            Args:
                frame: The frame with timing information to handle.
            """
            await self._clock_queue.put((frame.pts, next(self._clock_queue_counter), frame))

        async def handle_sync_frame(self, frame: Frame):
            """Handle frames that need synchronized processing.

            Args:
                frame: The frame to handle synchronously.
            """
            await self._audio_queue.put(frame)
```

**`src/pipecat/transports/base_output.py:1079-1100`**
```python
        async def _clock_task_handler(self):
            """Main clock/timing task handler for timed frame delivery."""
            running = True
            while running:
                timestamp, _, frame = await self._clock_queue.get()

                # If we hit an EndFrame, we can finish right away.
                running = not isinstance(frame, EndFrame)

                # If we have a frame we check it's presentation timestamp. If it
                # has already passed we process it, otherwise we wait until it's
                # time to process it.
                if running:
                    current_time = self._transport.get_clock().get_time()
                    if timestamp > current_time:
                        wait_time = nanoseconds_to_seconds(timestamp - current_time)
                        await asyncio.sleep(wait_time)

                    # Push frame downstream.
                    await self._transport.push_frame(frame)

                self._clock_queue.task_done()
```

An `asyncio.PriorityQueue` keyed on `pts`, with a real `asyncio.sleep` to the presentation time.

Read the consequence carefully, because it is the whole payoff of this chapter and the premise of the
next one:

> A word-timestamped `TTSTextFrame` reaches processors **downstream of `transport.output()`** at the
> moment its word is played, not at the moment it was synthesised. Position 7 of the canonical
> pipeline — the assistant context aggregator ([[canonical-voice-bot]]) — is downstream of
> `transport.output()`. So the assistant context is built out of words **as the customer hears them**.
> A barge-in that arrives at t=1.4 s into a 4.2 s sentence has, by construction, only delivered the
> words with `pts <= 1.4 s`. Truncating the context is not a computation. It is just: stop.

That is what [[ch-08/read]] spends. It is available to you only if your TTS provider emits word
timestamps — which for Korean means the six services in §7.4.

There is a subtlety the code has to work around: frames *without* a PTS take the audio queue instead
of the clock queue, and could therefore overtake clock-queued word frames. Two places stamp a PTS
purely to force a frame onto the clock queue:

**`src/pipecat/services/tts_service.py:926-933`**
```python
                    # When word-level TTSTextFrames are routed through the
                    # transport's clock queue (PTS-based), the aggregation frame
                    # would otherwise take the audio (sync) queue path and
                    # could overtake the final word frames. Stamping it with a
                    # PTS just past the last word forces it through the clock
                    # queue too, so the assistant aggregator sees every word
                    # before flushing.
                    if self._word_last_pts:
                        aggregation_frame.pts = self._word_last_pts + 1
```

**`src/pipecat/services/tts_service.py:950-964`**
```python
        # The will_be_spoken AggregatedTextFrame is the "new segment" announcement whose
        # segment_id the per-word progress frames reference. Those progress frames carry a
        # PTS and travel the transport's clock queue; the announcement itself has no PTS and
        # would take the audio (sync) queue, so for a context whose audio is delayed behind
        # another context's, its clock-queued progress can be delivered before it. Stamp it
        # with the same baseline the first word will use so it rides the clock queue too,
        # sorted immediately before its first progress frame (ties broken by push order).
        # Only on the word-timestamp path (push_text_frames services have no progress frames).
        if (
            isinstance(frame, AggregatedTextFrame)
            and frame.will_be_spoken
            and frame.pts is None
            and not self._push_text_frames
        ):
            frame.pts = max(self._word_last_pts, self.get_clock().get_time())
```

`pts` here is being used as a **queue selector**, not just a timestamp. Worth knowing when you debug
an ordering bug and find frames arriving in an order the pipeline diagram says is impossible.

→ Third panel of **[tts-streaming.html](figures/tts-streaming.html)**. Scrub the playhead and watch
word frames release from the clock queue at their `pts`. Then drop a barge-in marker mid-sentence and
read off which words had already been delivered — that set is exactly what the assistant context will
contain. Do this before starting ch-08.

### 6.7 The Korean test that already proves the truncation

`AggregatedFrameSequencer.force_complete` is what runs when a provider drops timestamp events, and
the repo tests it on a Korean sentence:

**`tests/test_aggregated_frame_sequencer.py:793-805`**
```python
    async def test_korean_force_complete_emits_correct_remaining_text(self):
        """After one Korean word, force_complete emits the correct unspoken suffix."""
        seq = _seq()
        sentence = "저는 여러분의 AI 어시스턴트입니다."
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        seq.process_word("저는", pts=10, context_id="ctx1")

        result = seq.force_complete("ctx1", last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "여러분의 AI 어시스턴트입니다.")
        self.assertEqual(tts_frames[0].pts, 50)
```

One Korean 어절 spoken, and the machinery correctly identifies `여러분의 AI 어시스턴트입니다.` as the
unspoken remainder — with the 어절 spacing reconstructed. That is the Korean 어절-level split working
end to end at the frame layer, and it is the concrete refutation of the outline claim §0.2 corrected.

---

## 7. Korean, provider by provider

The survey below re-walks the ground mapped in [[tts-korean-providers]], opening every file it
names. Where the excerpt and the source disagree the source is quoted and the disagreement is
flagged.

### 7.1 The grep, and its exact output

```
$ grep -rn "Language.KO" src/pipecat/services/*/tts*.py | sort
src/pipecat/services/aws/tts.py:98:        Language.KO: "ko-KR",
src/pipecat/services/camb/tts.py:82:        Language.KO: "ko-kr",
src/pipecat/services/cartesia/tts.py:112:        Language.KO: "ko",
src/pipecat/services/elevenlabs/tts_base.py:259:        Language.KO: "ko",
src/pipecat/services/google/tts.py:146:        Language.KO: "ko-KR",
src/pipecat/services/google/tts.py:363:        Language.KO: "ko-KR",
src/pipecat/services/inworld/tts.py:78:        Language.KO: "ko-KR",
src/pipecat/services/lmnt/tts.py:54:        Language.KO: "ko",
src/pipecat/services/minimax/tts.py:69:        Language.KO: "Korean",
src/pipecat/services/soniox/tts.py:89:        Language.KO: "ko",
src/pipecat/services/xai/tts.py:72:        Language.KO: "ko",
src/pipecat/services/xtts/tts.py:69:        Language.KO: "ko",
```

Eleven provider directories. Azure is the twelfth and does not show up here because its map lives in
a shared module:

**`src/pipecat/services/azure/common.py:199-201`**
```python
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
```

**Twelve provider modules declare Korean.** Say "modules", not "services", because each module ships
one to three service classes (a websocket one, an HTTP one, sometimes a third), and they share the
map. The full class-level table, verified class by class:

| Module | Class(es) | Korean code | Transport | Word ts |
|---|---|---|---|---|
| `azure` | `AzureTTSService` (`azure/tts.py:271`) | `ko-KR` | SDK push-stream | **yes** |
| | `AzureHttpTTSService` (`:832`) | `ko-KR` | HTTP | no |
| `cartesia` | `CartesiaTTSService` (`cartesia/tts.py:219`) | `ko` | WS | **yes** |
| | `CartesiaHttpTTSService` (`:786`) | `ko` | HTTP | no |
| `elevenlabs` | `ElevenLabsTTSService` (`elevenlabs/tts.py:200`) | `ko` | WS | **yes** |
| | `ElevenLabsHttpTTSService` (`:637`) | `ko` | HTTP | yes |
| `inworld` | `InworldTTSService` (`inworld/tts.py:554`) | `ko-KR` | WS | **yes** |
| | `InworldHttpTTSService` (`:123`) | `ko-KR` | HTTP | yes |
| `soniox` | `SonioxTTSService` (`soniox/tts.py:143`) | `ko` | WS | **yes** |
| `xai` | `XAITTSService` (`xai/tts.py:323`) | `ko` | WS | **yes** (gated) |
| | `XAIHttpTTSService` (`:150`) | `ko` | HTTP | no |
| `google` | `GoogleTTSService` (`google/tts.py:1023`) | `ko-KR` | gRPC streaming | no |
| | `GoogleHttpTTSService` (`:550`), `GeminiTTSService` (`:1205`) | `ko-KR` | HTTP | no |
| `lmnt` | `LmntTTSService` (`lmnt/tts.py:78`, `InterruptibleTTSService`) | `ko` | WS | no |
| `minimax` | `MiniMaxHttpTTSService` (`minimax/tts.py:137`) | `Korean` | HTTP | no |
| `aws` | `AWSPollyTTSService` (`aws/tts.py:148`) | `ko-KR` | HTTP | no |
| `camb` | `CambTTSService` (`camb/tts.py:157`) | `ko-kr` | HTTP | no |
| `xtts` | `XTTSService` (`xtts/tts.py:93`) | `ko` | local HTTP | no |

`MiniMaxHttpTTSService` is the odd one — its "code" is the English word `"Korean"`, fed to a
`language_boost` field rather than a language parameter. Not a locale at all.

### 7.2 An unmapped language does not raise. It warns.

**`src/pipecat/transcriptions/language.py:583-630`** (the fallback tail)
```python
    # Check if language is in the verified map
    result = language_map.get(language)

    if result is not None:
        return result

    # Not in map - fall back with warning
    lang_str = str(language)

    if use_base_code:
        # Extract base code (e.g., "en" from "en-US")
        base_code = lang_str.split("-")[0].lower()
        logger.warning(f"Language {language} not verified. Using base code '{base_code}'.")
        return base_code
    else:
        logger.warning(f"Language {language} not verified. Using '{lang_str}'.")
        return lang_str
```

No exception. A `logger.warning`, and the raw code goes out on the wire.

That is the design decision the rest of this section is about. Pipecat treats language maps as a
*verification list*, not a capability contract — the docstring says *"not verified"*, not
"unsupported". The framework's position is that it should not know better than you which languages a
provider supports, so it passes your request through and tells you it could not confirm it.

### 7.3 The Rime trap, in full

**`src/pipecat/services/rime/tts.py:46-63`**
```python
def language_to_rime_language(language: Language) -> str:
    """Convert pipecat Language to Rime language code.

    Args:
        language: The pipecat Language enum value.

    Returns:
        Three-letter language code used by Rime (e.g., 'eng' for English).
    """
    LANGUAGE_MAP = {
        Language.DE: "ger",
        Language.FR: "fra",
        Language.EN: "eng",
        Language.ES: "spa",
        Language.HI: "hin",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)
```

Five languages, three-letter codes, no Korean. And `use_base_code=False`, so `Language.KO` becomes
the literal string `"ko"` — a two-letter code sent to an API that only understands
`eng/ger/fra/spa/hin`.

Trace the whole failure:

1. You write `RimeTTSService(settings=..., language=Language.KO)`. Rime is on your shortlist because
   it emits word timestamps (it is in the `add_word_timestamps` caller list) and it markets low
   latency.
2. `TTSService.__init__` converts at construction time:

   **`src/pipecat/services/tts_service.py:273-276`**
   ```python
        if isinstance(self._settings.language, Language):
            converted = self.language_to_service_language(self._settings.language)
            if converted is not None:
                self._settings.language = converted
   ```
3. `resolve_language` logs `Language ko not verified. Using 'ko'.` — one WARNING line, at startup,
   among however many others your app emits.
4. The bot starts. The pipeline builds. Nothing raises. `StartFrame` propagates. Every health check
   passes.
5. The customer says 여보세요. The LLM produces Korean. `run_tts` sends it with `"ko"`. Rime returns
   nothing usable.
6. Three contexts later — §7.8 — the service writes itself off.

Everything up to step 6 looks healthy. That is the shape of the failure: **a Korean bot pointed at a
non-Korean provider configures fine.**

The same shape applies to every service in §7.5.

### 7.4 The intersection is six, and here is exactly what that number means

```
$ grep -rn "add_word_timestamps" src/pipecat/services/*/tts*.py | cut -d: -f1 | sort -u
src/pipecat/services/azure/tts.py
src/pipecat/services/cartesia/tts.py
src/pipecat/services/elevenlabs/tts.py
src/pipecat/services/gradium/tts.py
src/pipecat/services/hume/tts.py
src/pipecat/services/inworld/tts.py
src/pipecat/services/resembleai/tts.py
src/pipecat/services/rime/tts.py
src/pipecat/services/smallest/tts.py
src/pipecat/services/soniox/tts.py
src/pipecat/services/speechify/tts.py
src/pipecat/services/xai/tts.py
```

(Plus `src/pipecat/services/elevenlabs/dialogue/tts.py:378`, a thirteenth call site inside the
ElevenLabs module.)

Intersect the twelve Korean-mapping modules with the twelve timestamp-emitting modules:

```
Korean:      aws azure camb cartesia elevenlabs google inworld lmnt minimax soniox xai xtts
Timestamps:  azure cartesia elevenlabs gradium hume inworld resembleai rime smallest soniox speechify xai
             ────────────────────────────────────────────────────────────────────────────────
Intersection: azure  cartesia  elevenlabs  inworld  soniox  xai
```

**Six** — the same six [[tts-korean-providers]] names, reproduced here from the greps rather than
taken on trust. Now the provenance, stated as honestly as I can:

> This number is the intersection of **two greps over this source tree**: services whose language map
> contains a `Language.KO` entry, and services that call `add_word_timestamps`. It is a claim about
> what the **code declares**, not a claim verified by listening to any output.
>
> Specifically, nothing here establishes that:
> - any of the six synthesises *good* Korean, or even intelligible Korean;
> - the timestamps any of them emit are accurate, or do not drift over a long utterance;
> - a Korean voice ID exists for the account you will actually use.
>
> No test in the repo listens to audio. `pipecat.evals` (`src/pipecat/evals/`) can drive a real bot
> end to end with an LLM judge, and `scripts/release-evals/` ships a manifest — but there is no
> Korean scenario in it. The Korean tests that exist (§0.2, §6.7) are frame-level unit tests over
> synthetic token lists.
>
> Two of these six could be bad at Korean and this repo would not know.

So: use the six as a **shortlist to test**, not a list of six working options. That distinction is
the difference between a two-week and a two-month integration.

One of the six has a further gate. xAI only emits timestamps when you ask:

**`src/pipecat/services/xai/tts.py:139-150`**
```python
        with_timestamps: Whether to request character timings. When enabled, the
            service converts them into per-word ``TTSTextFrame`` objects.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    optimize_streaming_latency: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    with_timestamps: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
```

`NOT_GIVEN` by default. If you pick xAI and forget to set `with_timestamps=True`, you get a service
that appears on the six-service list and behaves like one of the other six.

### 7.5 The services that do NOT support Korean — checked, not assumed

Every one of these was opened:

- **Rime** (`rime/tts.py:46`) — five languages, no Korean. §7.3.
- **Neuphonic** (`neuphonic/tts.py:40`) — `de, en, es, nl, ar, fr, pt, ru, hi, zh`. No Korean.
- **Kokoro** (`kokoro/tts.py:66`) — maps espeak-ng voice names, verified by opening the map:
  `en-us, en-gb, es, fr-fr, fr-be, fr-ca, fr-ch, hi, it, ja, pt, pt-br, cmn, yue, zh, zh-CN, zh-HK,
  zh-TW`. **No Korean**, and the docstring is blunt about what happens if you send one anyway:

  **`src/pipecat/services/kokoro/tts.py:66-78`**
  ```python
  def language_to_kokoro_language(language: Language) -> str:
      """Convert a Language enum to kokoro-onnx language code.

      kokoro-onnx phonemizes through espeak-ng, so these are espeak-ng voice names
      rather than ISO codes. They differ for Mandarin (``cmn``, not ``zh``) and
      French, which espeak-ng only offers per-region (``fr-fr``, no bare ``fr``);
      an unsupported name fails at synthesis time.
      """
  ```
  *"an unsupported name fails at synthesis time"* — i.e. at 14:03 on a live call, not at startup.
- **Piper** (`piper/tts.py:44`, `:207`) — no `language_to_service_language` override at all. Both
  classes set `default_settings = self.Settings(model=None, voice=None, language=None)` (`:91`,
  `:243`). Language is whatever `.onnx` voice file you loaded; Pipecat has no Korean knowledge here
  and cannot warn you about anything.
- **OpenAI** (`openai/tts.py:81`) — no language parameter exists.
- **Fish, Deepgram, Gradium, Hume, ResembleAI** — `language=None` in default settings, no map.

### 7.6 There is no maintained self-hosted Korean TTS in this tree

The only local service with a Korean map is deprecated:

**`src/pipecat/services/xtts/tts.py:89-92`**
```python
@deprecated(
    "`XTTSService` is deprecated since 1.7.0 and will be removed in 2.0.0. No replacement. "
    "`KokoroTTSService` and `PiperTTSService` are the maintained local TTS services."
)
```

**`src/pipecat/services/xtts/tts.py:10-16`**
```
text-to-speech synthesis using local Docker deployment.

.. deprecated:: 1.7.0
    No replacement. :class:`~pipecat.services.kokoro.tts.KokoroTTSService` and
    :class:`~pipecat.services.piper.tts.PiperTTSService` are the maintained
    local TTS services. Will be removed in 2.0.0.
```

*"No replacement."* And the two named replacements are exactly the two local services that do not map
Korean (§7.5).

Add to that: there is **no Korean-native provider integrated at all**. No `typecast/`, no
`supertone/`, no `clova/` or `naver/` directory exists under `src/pipecat/services/`.

Both facts are gaps, not choices. If Lina has an on-prem or data-residency requirement — and for
Korean insurance tele-sales that is not a hypothetical — the options in this tree are: write a custom
`TTSService` subclass against a Korean vendor's API, or do not use Pipecat's TTS layer for that
deployment. Nothing in the current tree covers it. That is a build item, and §11 lists it as one.

### 7.7 Config details that will bite, gathered in one place

**Azure needs a locale-matched voice.** The default is English on both axes:

**`src/pipecat/services/azure/tts.py:331-343`**
```python
        default_settings = self.Settings(
            model=None,
            voice="en-US-SaraNeural",
            language="en-US",
            emphasis=None,
            force_locale=False,
```

Setting `language=Language.KO` while leaving `voice="en-US-SaraNeural"` is precisely the
unknown-voice-for-this-language case that §7.8's guard exists to catch. Azure also hard-codes a
sentence-boundary pause and offers a locale wrapper:

**`src/pipecat/services/azure/tts.py:197-206`**
```python
        ssml = (
            f"<speak version='1.0' xml:lang='{language}' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{self._settings.voice}'>"
            "<mstts:silence type='Sentenceboundary' value='20ms' />"
        )

        if self._settings.force_locale:
            ssml += f"<lang xml:lang='{language}'>"
```

`20ms` at every sentence boundary, not configurable through settings.

**ElevenLabs silently drops your language code if the model does not cover it.**

**`src/pipecat/services/elevenlabs/tts_base.py:173-205`**
```python
def elevenlabs_language_code(model: str | None, language: str | None) -> str | None:
    """Resolve the ``language_code`` to send for a model.

    Returns:
        The language code to send, or None if it can't be used - either because
        the model takes no language code or because it doesn't cover this
        language. Both cases are logged.
    """
    if not language:
        return None

    supported = ELEVENLABS_MODEL_LANGUAGES.get(model or "")
    if supported is None:
        logger.warning(
            f"Language code [{language}] not applied. Language codes can only be used with: "
            f"{', '.join(sorted(ELEVENLABS_MODEL_LANGUAGES))}"
        )
        return None

    if language not in supported:
        logger.warning(
            f"Language code [{language}] not applied. {model} supports "
            f"{len(supported)} languages, which don't include [{language}]."
        )
        return None
```

`"ko"` is present in both `ELEVENLABS_V2_5_LANGUAGES` (32 codes, `tts_base.py:39`) and
`ELEVENLABS_V3_LANGUAGES` (74 codes, `:78`) — I counted both. The default model is
`"eleven_flash_v2_5"`, which is in the map. But `ELEVENLABS_MODEL_LANGUAGES` has only four keys
(`:165-170`), so choosing `eleven_multilingual_v2` — a plausible choice for a Korean bot — drops your
language code entirely with a warning and falls back to the model's own auto-detection.

**Soniox caps concurrency per connection.**

**`src/pipecat/services/soniox/tts.py:143-153`**
```python
class SonioxTTSService(WebsocketTTSService):
    """Soniox WebSocket TTS service with streaming text-in, streaming audio-out.

    Streams text incrementally to Soniox's real-time TTS endpoint and routes the
    returned base64-encoded audio back as :class:`TTSAudioRawFrame` frames.
    Multiple concurrent streams are multiplexed over a single WebSocket
    connection via Pipecat's audio-context mechanism (mapped to Soniox's
    ``stream_id``). Supports up to 5 concurrent streams per connection.
    """
```

Five streams per connection. For a single-call Lina worker that is irrelevant; for a
process-per-N-calls deployment shape ([[ch-04/read]] §13) it is a hard constraint on how you pool
connections. Its defaults are also English: `model="tts-rt-v2", voice="Bryce"` (`:189-193`).

**Inworld emits punctuation and spaces as separate tokens.** Handled with a flag:

**`src/pipecat/services/inworld/tts.py:1084`**
```python
                    await self.add_word_timestamps(word_times, ctx_id, pre_merge_tokens=True)
```

which runs:

**`src/pipecat/utils/text/word_timestamp_utils.py:12-40`**
```python
def merge_punct_tokens(
    word_times: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge punctuation/space-only tokens into the preceding word.

    Some TTS services (e.g. Inworld) emit spaces and punctuation as separate
    word-timestamp tokens rather than attaching them to the adjacent word.
    This function collapses those tokens so downstream consumers always receive
    words with trailing punctuation already attached — identical to the format
    produced by ElevenLabs or Cartesia.

    A token is considered punct/space-only when its text contains no alphanumeric
    characters after stripping XML/HTML tags.
    """
```

Its Korean behaviour is worth one line of thought, because it hinges on a Python detail:

**`src/pipecat/utils/text/word_timestamp_utils.py:42-53`**
```python
    merged: list[tuple[str, float]] = []
    for word, ts in word_times:
        stripped = re.sub(r"<[^>]+>", "", word)
        has_alnum = any(c.isalnum() for c in stripped)
        if not has_alnum:
            if merged:
                prev_word, prev_ts = merged[-1]
                merged[-1] = (prev_word + word, prev_ts)
            # else: leading punct/space with no preceding word → discard
        else:
            merged.append((word, ts))
    return [(word.strip(), ts) for word, ts in merged]
```

`str.isalnum()` is True for Hangul syllables — they are Unicode letters. So Korean tokens survive the
filter and only real punctuation/whitespace tokens get merged. The function was written for English
and works on Korean for a reason nobody wrote down. Worth checking if you ever change it.

### 7.8 The only catch: three silent contexts

**`src/pipecat/services/tts_service.py:224-231`**
```python
            max_consecutive_zero_audio_contexts: How many consecutive TTS contexts may
                complete without producing any audio before the service is reported unable
                to do its job. Catches a provider that accepts requests and stays silent —
                an unknown voice ID, say — which no error ever surfaces. On reaching the
                limit the service reports a permanent error, stops being given work, and
                the pipeline worker applies its
                :class:`~pipecat.pipeline.worker.ProcessorUnusablePolicy`. Set to 0 to let
                silent contexts go unchecked.
```

Default 3 (`:168`). The implementation:

**`src/pipecat/services/tts_service.py:1801-1834`**
```python
        if received_audio:
            self._consecutive_zero_audio_contexts = 0
            return

        # This context played nothing, so the transport will never send the
        # BotStoppedSpeakingFrame that lifts a pause taken for it, which can
        # happen when the pause was taken while the context was still open. A
        # bot still speaking is playing audio from another context, whose own
        # BotStoppedSpeakingFrame is still to come.
        if not self._bot_speaking:
            await self._maybe_resume_frame_processing()

        if not self._max_consecutive_zero_audio_contexts:
            return

        # An unusable service is deliberately not given work (see
        # _synthesize_text), so its silent contexts say nothing new.
        if not self.is_usable:
            return

        self._consecutive_zero_audio_contexts += 1
        logger.warning(
            f"{self} audio context {context_id} completed with no audio "
            f"({self._consecutive_zero_audio_contexts} in a row)"
        )

        if self._consecutive_zero_audio_contexts >= self._max_consecutive_zero_audio_contexts:
            await self.push_error(
                error_msg=(
                    f"{self._consecutive_zero_audio_contexts} consecutive TTS contexts "
                    "completed with no audio"
                ),
                force_treat_as_permanent=True,
            )
```

This is well tested — eight cases in `tests/test_tts_zero_audio_contexts.py`, whose module docstring
states the failure mode exactly:

**`tests/test_tts_zero_audio_contexts.py:7-13`**
```
"""Tests for writing off a TTS service that stops producing audio.

A provider can accept every request and return no audio at all — an unknown
voice ID, say — without reporting an error. TTSService counts the contexts that
complete in silence and, past a configurable limit, reports itself unable to do
its job so the pipeline worker and any ServiceSwitcher can act on it.
"""
```

Now put the timeline together for the Rime scenario in §7.3. `_stop_frame_timeout_s` is 3.0 s
(§5.4); a silent context is only declared finished when it times out. Three consecutive silent
contexts is therefore **on the order of nine seconds of dead air on a live call** before anything is
reported — and only *then* does `ProcessorUnusablePolicy` ([[ch-04/read]]) get to act.

For Lina, on a cold outbound dial to a 고객 who just picked up, nine seconds of silence is the entire
call. If you want a faster write-off you set `max_consecutive_zero_audio_contexts=1` and accept that
a genuinely empty utterance (a turn that is only a function call, say) will trip it. There is no
middle setting that is both fast and safe. §11 makes a call.

---

## 8. Korean word grouping: the branch Korean falls out of

### 8.1 The zh/ja test, in two places

**`src/pipecat/services/cartesia/tts.py:445-456`**
```python
    def _is_chinese_or_japanese_language(self, language: str) -> bool:
        """Check if the given language is Chinese or Japanese.

        Args:
            language: The language code to check.

        Returns:
            True if the language is Chinese or Japanese.
        """
        base_lang = language.split("-")[0].lower()
        return base_lang in {"zh", "ja"}
```

**`src/pipecat/services/elevenlabs/tts_base.py:329-337`**
```python
def _is_chinese_or_japanese_language(language: str) -> bool:
    """Check if the given language is Chinese or Japanese."""
    base_lang = language.split("-")[0].lower()
    return base_lang in {"zh", "ja"}


def _word_timestamps_include_inter_frame_spaces(language: str | None) -> bool:
    """Whether timestamp text should be treated as carrying its own spacing."""
    return bool(language and _is_chinese_or_japanese_language(language))
```

`{"zh", "ja"}`. Korean is not in the set, and its absence is deliberate rather than an oversight —
the CJK branch exists because zh/ja have no inter-word spaces, and Korean does.

What the branch does when taken:

**`src/pipecat/services/cartesia/tts.py:505-520`**
```python
        if current_language and self._is_chinese_or_japanese_language(current_language):
            # For Chinese/Japanese, combine all characters in this message into one word
            # using the first character's start time.
            if words and starts:
                combined_word = "".join(self._strip_cartesia_tags(w) for w in words)
                first_start = starts[0]
                return [(combined_word, first_start)] if combined_word else []
            else:
                return []
        else:
            result = []
            for word, start in zip(words, starts):
                cleaned = self._strip_cartesia_tags(word)
                if cleaned:
                    result.append((cleaned, start))
            return result
```

For zh/ja, a whole timestamp message collapses to **one** word — a full utterance chunk with a single
timestamp, which is exactly the resolution loss you would expect and which makes mid-utterance
truncation coarse. Korean takes the `else` branch and gets one entry per token, keeping full
resolution.

`includes_inter_frame_spaces` is the flag that rides along:

**`src/pipecat/services/cartesia/tts.py:522-526`**
```python
    def _word_timestamps_include_inter_frame_spaces(self) -> bool:
        """Whether timestamp text should be treated as carrying its own spacing."""
        current_language = assert_given(self._settings.language)
        return bool(current_language and self._is_chinese_or_japanese_language(current_language))
```

False for Korean → the downstream concatenator inserts a space between consecutive word frames. That
is the behaviour the test in §0.2 asserts: `["저는"], ["여러분의"], ["AI", "어시스턴트입니다."]`
reassembles as `"저는 여러분의 AI 어시스턴트입니다."`.

### 8.2 ElevenLabs splits on the space character, which for Korean means 어절

ElevenLabs sends character-level alignment; Pipecat converts it to words:

**`src/pipecat/services/elevenlabs/tts_base.py:442-464`**
```python
    # Build words and track their start positions
    words = []
    word_start_times = []
    current_word = partial_word  # Start with any partial word from previous chunk
    word_start_time = partial_word_start_time if partial_word else None

    for i, char in enumerate(chars):
        if char == " ":
            # End of current word
            if current_word:  # Only add non-empty words
                words.append(current_word)
                word_start_times.append(word_start_time)
                current_word = ""
                word_start_time = None
        else:
            # Building a word
            if word_start_time is None:  # First character of new word
                # Convert from milliseconds to seconds and add cumulative offset
                word_start_time = cumulative_time + (char_start_times_ms[i] / 1000.0)
            current_word += char

    # Build result for complete words
    word_times = list(zip(words, word_start_times))
```

`if char == " "` — that is the entire word-boundary definition. For English it is words. For Korean
it is **어절**: `보험료는`, `한달에`, `이만구천원입니다.` each become one `TTSTextFrame` with its own
`pts`.

Which lands the truncation unit on 어절 rather than on characters. Cutting a Korean assistant
turn mid-어절 produces a fragment like `보험료` that reads as a broken word; cutting at 어절
boundaries produces `보험료는 한달에` which reads as an interrupted sentence. Nobody wrote a comment
saying so — this is what the code does, and it happens to line up with Korean orthography.

For zh/ja it does not line up at all: with no spaces, the whole utterance would be one "word", which
is why the `includes_inter_frame_spaces` branch exists to handle those separately.

### 8.3 What is tested, and what is not — for the last time

| Claim | Status at `0cbf9c5b` |
|---|---|
| Korean word timestamps stay per-token, not merged | **tested** (`test_cartesia_tts.py:69-75`) |
| Latin and Hangul tokens are not joined | **tested** (`:78-84`) |
| Korean groups reassemble with 어절 spaces | **tested** (`:111-124`) |
| The sequencer completes a Korean slot word by word | **tested** (`test_aggregated_frame_sequencer.py:776-791`) |
| `force_complete` emits the correct Korean remainder | **tested** (`:793-805`) |
| Korean full-width punctuation ends a sentence | **tested** (`test_utils_string.py:94-99`, `:176-177`) |
| Korean with **ASCII** punctuation ends a sentence | **not tested**; measured in §4.5, holds on realistic inputs |
| Any provider produces intelligible Korean audio | **not tested, not testable in this repo** |
| Korean word timestamps align with the actual audio | **not tested, not testable in this repo** |

That table is the honest summary. The frame plumbing for Korean is real and covered. The audio is
entirely on you.

---

## 9. What realtime_voice does at this layer

Mechanism only. §9.3 is a measurement, not a ranking. All realtime_voice facts here come from
[[rtv-vad-chunking]]; the private repo is not opened.

### 9.1 `KoreanPhraseChunker` — the 1 → 2 → tail schedule

283 lines, no Pipecat counterpart. Its constructor and docstring, as recorded in the excerpt and
already summarised in [[ch-03/read]] §7.1:

```
__init__(*, min_chars=12, max_chars=60, hard_max_chars=None,
         batch_max_chars=320, adaptive_batching=True)
# hard_max_chars=None resolves to min(batch_max_chars, max_chars * 2)   (L56-60)
```

> *"Adaptive mode emits the first complete sentence immediately, batches the next two complete
> sentences, then holds the remaining response as one final group until `flush`. `max_chars` is a
> soft latency target rather than an immediate cut point."* (docstring L28-34)

`_batch_phase` 0/1/2 in `_accept_adaptive` (L115-149). Read in TTFA terms — the frame §1 gave you —
the schedule is a direct statement about which sentence's latency is exposed:

- **Phase 0** (first sentence, emitted alone): sentence 1 is the only one on the critical path, so it
  ships as small as possible.
- **Phase 1** (next two sentences, batched in pairs): playback of sentence 1 is already running, so
  sentences 2 and 3 are being produced *underneath* it and amortise request overhead instead.
- **Phase 2** (bounded tail, one group until flush): stop fragmenting once the customer is listening.

Pipecat has no phase counter. Its aggregation mode is a constant for the life of the service —
`SENTENCE` or `TOKEN` — with the same boundary rule applied to sentence 1 and sentence 9.

### 9.2 The three guards, and the span preservation

From the excerpt, verbatim on the identifiers:

- `_is_safe_period` (L255) refuses to split a dot inside `1.5`, inside `...`, or between ASCII token
  characters. Comment L266-269: *"A dot between ASCII token characters belongs to a model name,
  hostname, abbreviation, or identifier rather than ending a Korean sentence."*
- `_is_numeric_separator` (L277) protects `1,000` from the `_SOFT_END` comma set
  (`frozenset(",，;；:")`).
- `_INTERNAL_TAG = re.compile(r"\[(?:interruption|system|tool|objection|customer|assistant)[^\]]*\]")`
  strips gateway control tags out of the spoken text while `start_char` / `end_char` keep the
  **source** span intact.

The third is the structurally distinctive one, and it is worth being precise about what it is for.
Boson's gateway injects control tags into the assistant text stream — `[interruption]`,
`[objection ...]`, `[tool ...]` — which must not be spoken but must remain addressable, because
`AudioTextPlayoutLedger` maps *heard characters* back onto the **original** string. Strip-and-forget
would break that mapping. Strip-and-keep-span does not.

Pipecat's analogue of "text that reaches the context but is not spoken" is a different mechanism
entirely: the `skip_tts` flag on `TextFrame` (`frames.py:303-330`) plus `skip_aggregator_types`:

**`src/pipecat/services/tts_service.py:1159-1163`**
```python
        # Skip sending to TTS if the aggregation type is in the skip list. Simply
        # push the original frame downstream.
        if type in self._skip_aggregator_types:
            await self._push_frame_respecting_previous_aggregated_frame(src_frame, context_id)
            return
```

Frame-level rather than character-level. There is no character-span concept anywhere in Pipecat's
TTS path.

### 9.3 The same inputs, both paths — measurement, not verdict

`_is_safe_period` and `_is_numeric_separator` are explicit guards against splitting `1.5`, `gpt-4.1`,
`1,000`. Pipecat has no such guard. The obvious question is what Pipecat's path actually does with
those inputs, and §4.5 answered it by running them:

| Input (as a token stream, inside a Korean sentence) | Pipecat: does it split? | realtime_voice guard |
|---|---|---|
| `월 1.5만원입니다.` | no — emits the whole sentence | `_is_safe_period` |
| `저는 gpt-4.1을 씁니다.` | no — emits the whole sentence | `_is_safe_period` |
| `보험료는 1,000원입니다.` | no — emits the whole sentence | `_is_numeric_separator` |
| `월보험료는 29.99만원입니다.` | no — emits the whole sentence | `_is_safe_period` |
| `안녕하세요。고객님` | splits after `。`, 1 token earlier | `_STRONG_END` includes `。` |
| `월 보험료는 29. 000원` | **splits after `29.`** | `_is_safe_period` would refuse |

Two mechanisms — an explicit hand-written guard list, and an unsupervised statistical tokenizer plus a
one-character lookahead — reaching the same output on five of six inputs and differing on a contrived
sixth. That is the measurement. What to conclude from it is [[ch-13/read]]'s job, and the inputs that
would actually settle it are your own logged Korean transcripts, not either of these tables.

One structural difference is not a measurement and can be stated flatly: **realtime_voice's guards
are inspectable and Pipecat's is not.** `_is_safe_period` is a function you can read and add a rule
to. NLTK's Punkt is a trained model whose behaviour on Korean you can only characterise empirically,
as §4.5 did. If you need to *guarantee* a split never happens on a particular pattern — a policy
number, a 주민등록번호 fragment, an insurance product code — the aggregator interface from §3 is where
you would add that guarantee, not the model.

### 9.4 `interrupt/fillers.py` — the name is misleading and it survives untouched

40 lines, and despite living under `interrupt/`, it does not suppress the *bot's* fillers. From
[[boson-interrupt-subsystem]]:

- `FillerCheck = Callable[[str, str], bool]` over `(text, agent_status)`, with
  `set_filler_check` / `get_filler_check` / `is_filler` / `clear_filler_check`.
- Docstring: *"The gateway has zero language knowledge — it only calls the registered callback."*
- `is_filler` returns `False` when unset.
- Lina's actual implementation lives outside the package, at
  `agents/test-lina-gateway/layers/01-filler-filter/rules/korean_fillers.py`.

It is **user-side** filler suppression: the customer says 음, 어, 네네 while Lina is speaking, and this
callback decides that it is not a real barge-in. `InterruptionGate.allows` consults it before the
barge-in policy ([[boson-interrupt-subsystem]] `server/interruption.py:36`, step 3).

Nothing in this chapter replaces it. Nothing in Pipecat's TTS layer touches it. It is orthogonal to
everything in §1–§8, it is Korean-language business logic, and the migration answer for it is "port
it, unchanged" — as a custom `BaseUserTurnStartStrategy` subclass, per [[boson-interrupt-subsystem]],
which is [[ch-06/read]]'s and [[ch-08/read]]'s territory rather than this chapter's.

I flag it here because the name invites exactly one mistake — reading "fillers" next to "TTS" and
assuming it is about the bot's 추임새. It is not. The bot-side equivalent is `TTSSpeakFrame` (§5.5),
which is a different mechanism for a different problem.

### 9.5 The ledger, one paragraph, because ch-03 has it

`AudioTextPlayoutLedger` (110 L) computes `audible_text()` from `ratio = (cursor - sample_start) /
(sample_end - sample_start)` and `text[: int(len(text) * ratio)]` — [[ch-03/read]] §7.2 has the full
treatment. The one thing to add now that you have §6: its **input requirement** is a sample count and
nothing else, whereas Pipecat's word-frame path requires the provider to emit timestamps. That is why
§7.4's six-service shortlist matters for a Pipecat migration and would not matter for a ledger-based
one. Two designs with different dependencies on the vendor. [[ch-13/read]] weighs that; this chapter
records it.

---

## 10. Framework-extension probes

Three moves. Each applies a mechanism from this chapter to something outside it. Do them before
[[ch-08/read]].

### 10.1 A Korean-aware `BaseTextAggregator`

You have all the pieces: `BaseTextAggregator.aggregate()` returns an `AsyncIterator[Aggregation]`
(`base_text_aggregator.py:103`); `Aggregation.type` is an **open string** (`:46`); `LLMTextProcessor`
takes any aggregator (`llm_text_processor.py:39-51`); `TTSService.process_frame` takes branch 2 on
`AggregatedTextFrame` and skips its own aggregation entirely (`tts_service.py:764-765`).

So the `1 → 2 → bounded-tail` schedule from §9.1 is implementable as a `BaseTextAggregator` subclass
with a `_batch_phase` counter, mounted on `LLMTextProcessor`, with **no** subclassing of any TTS
service.

The design question to answer for yourself, and it is not answered in either codebase: `Aggregation`
carries only `text` and `type`, and phase 2 emits a *group* of sentences as one aggregate. What
`type` string do you give it, and what does the downstream `AggregatedTextProgressFrame.segment_id`
then refer to — the group or the sentence? Trace it through `AggregatedFrameSequencer.register_spoken`
(`aggregated_frame_sequencer.py:292`) and decide what a "segment" means for a batched group. Write
down the answer; it determines whether your live-transcript UI highlights sentence by sentence or
paragraph by paragraph.

### 10.2 `text_transforms` as the Korean-number rule layer

You have not been shown this hook yet. It is a per-aggregation-type text rewriter that runs *after*
aggregation and *before* synthesis:

**`src/pipecat/services/tts_service.py:642-656`**
```python
    def add_text_transformer(
        self,
        transform_function: Callable[[str, AggregationType | str], Awaitable[str]],
        aggregation_type: AggregationType | str = "*",
    ):
        """Transform text for a specific aggregation type.

        Args:
            transform_function: The function to apply for transformation. This function should take
                the text and aggregation type as input and return the transformed text.
                Ex.: async def my_transform(text: str, aggregation_type: str) -> str:
            aggregation_type: The type of aggregation to transform. This value defaults to "*" indicating
                the function should handle all text before sending to TTS.
        """
        self._text_transforms.append((aggregation_type, transform_function))
```

And its call site, with a comment that names exactly the trap:

**`src/pipecat/services/tts_service.py:1242-1262`**
```python
        # Note: Text transformations are meant to only affect the text sent to the TTS for
        # TTS-specific purposes. This allows for explicit TTS modifications (e.g., inserting
        # TTS supported tags for spelling or emotion or replacing an @ with "at"). For TTS
        # services that support word-level timestamps, this CAN affect the resulting context
        # since the TTSTextFrames are generated from the TTS output stream
        transformed_text = text
        for aggregation_type, transform in self._text_transforms:
            if aggregation_type == type or aggregation_type == "*":
                try:
                    transformed_text = await transform(transformed_text, type)
                except Exception as e:
                    # Pushing the error with category "APPLICATION" since the failure came
                    # from the application's text transformer. Speaking the untransformed
                    # text isn't safe — a transformer may exist to remove something — so this
                    # turn produces no audio.
                    await self.push_error(
                        error_msg=f"Error transforming text for TTS [{transformed_text}]: {e}",
                        exception=e,
                        category=ErrorCategory.APPLICATION,
                    )
                    return
```

For Lina this is where `29000` → `이만 구천` lives, and where `010-1234-5678` becomes a
digit-by-digit reading. Two things to internalise from those twenty lines:

1. **A transform that raises kills the turn silently.** No audio, an `ErrorFrame` with category
   `APPLICATION`, and — because the context produced no audio — a tick on the zero-audio counter from
   §7.8. Three buggy transforms in a row and your TTS service writes itself off. Wrap yours.
2. **On a word-timestamp provider, the transform changes the assistant context**, because
   `TTSTextFrame`s are built from the provider's output stream, which is the *transformed* text. So
   if you transform `29000` → `이만 구천`, the LLM's memory of the turn says `이만 구천`, not `29000`.
   Whether that is what you want for a sales call where the number matters later is a decision, not a
   detail.

### 10.3 Read the zero-audio guard as a service-switch trigger

`_record_context_audio_outcome` calls `push_error(..., force_treat_as_permanent=True)`, which makes
the service unusable, which — per its own docstring — lets a `ServiceSwitcher` fail over. Build the
timeline for Lina's worst case: primary Korean provider goes silent at 14:03:22 on a live call.

- t+0.0 s: sentence 1 sent, context opens, no audio.
- t+3.0 s: `_stop_frame_timeout_s` fires, context 1 declared complete, counter = 1.
- t+6.0 s: counter = 2.
- t+9.0 s: counter = 3 → `push_error(force_treat_as_permanent=True)` → service unusable → switch.

Nine seconds. Now decide what you actually want, and note that the levers are not independent:
lowering `_stop_frame_timeout_s` also changes how quickly a *legitimately slow* provider gets its
context closed out from under it, and lowering `max_consecutive_zero_audio_contexts` to 1 means a
single function-call-only turn (which produces no audio by design) trips the write-off. Find the
third lever — there is one, in how you construct contexts — or accept one of the two costs. §11
takes a position.

---

## 11. Deliverable: the TTS block of the Lina pipeline

Not a recommendation between systems — a set of decisions inside Pipecat, with the ones that cannot
be decided from this chapter marked as such.

### 11.1 Decided from the source

| Decision | Value | Because |
|---|---|---|
| Base class to subclass, if custom | `WebsocketTTSService` | §0.1 — `AudioContextTTSService` is deprecated; audio contexts are on the root |
| Provider shortlist | Azure, Cartesia, ElevenLabs, Inworld, Soniox, xAI | §7.4 — Korean map ∩ word timestamps |
| Shortlist status | **candidates to A/B on real Korean audio**, not options | §7.4 provenance — grep, not behaviour |
| `text_aggregation_mode` | `SENTENCE` to start | §4.1 — TOKEN's own comment says it is less tested |
| Where a Korean chunker goes | `LLMTextProcessor` between `llm` and `tts` | §3, §10.1 — no TTS subclassing needed |
| Assistant aggregator position | after `transport.output()` | §6.6 — otherwise word-paced truncation has no evidence |
| Metrics to alert on | `TTFAMetricsData.leading_silence`, `TextAggregationMetricsData` | §1.2, §1.4 — neither is visible in TTFB |
| Do not aggregate | `TTFBMetricsData` + `TTFAMetricsData.ttfb` | §1.2 — the docstring says they are the same measurement |
| On-prem Korean TTS | not available; build item | §7.6 — XTTS deprecated "No replacement", Kokoro/Piper have no Korean |
| Delete on sight | any `SentenceAggregator` in the pipeline | §4.6 — decoy; no lookahead, not in the TTS path |

### 11.2 Not decided here, and what would decide it

| Open question | What would settle it |
|---|---|
| Which of the six providers | A/B on recorded Korean sales dialogue: TTFA distribution, timestamp drift over 10 s, 이름/숫자 pronunciation |
| `max_consecutive_zero_audio_contexts` | Measured base rate of legitimately-silent turns (function-call-only turns) in Lina's traffic |
| `_stop_frame_timeout_s` | Provider p99 first-audio latency under real network conditions |
| SENTENCE vs TOKEN | TTFA delta measured on the chosen provider; §4.7 says the aggregation metric cannot answer it |
| Whether to port the 1→2→tail schedule | TTFA delta between phase-0 single sentence and default sentence aggregation, on the chosen provider |

Note that four of those five need the same experiment: TTFA measured on real Korean traffic against a
chosen provider. Run that once and most of the table collapses.

### 11.3 The sketch

```python
# --- the TTS block only; transports and turn strategies are ch-05 / ch-06 ---

from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.services.tts_service import TextAggregationMode

# 1. Korean chunking lives OUTSIDE the TTS service (§3, §10.1).
#    KoreanAggregator implements BaseTextAggregator; TTSService then takes the
#    AggregatedTextFrame branch at tts_service.py:764 and never runs its own.
korean_text = LLMTextProcessor(text_aggregator=KoreanAggregator())

# 2. One of the six from §7.4. Cartesia shown because it is the canonical
#    example's choice; the shortlist is not resolved (§11.2).
tts = CartesiaTTSService(
    api_key=...,
    settings=CartesiaTTSService.Settings(
        language=Language.KO,          # ko  (cartesia/tts.py:112)
        voice=KOREAN_VOICE_ID,         # MUST be Korean — §7.7 Azure trap applies here too
    ),
    text_aggregation_mode=TextAggregationMode.SENTENCE,
    # §10.3: three silent contexts × 3 s timeout = ~9 s of dead air before failover.
    # Left at the default until Lina's rate of legitimately-silent turns is measured.
    max_consecutive_zero_audio_contexts=3,
)

# 3. Korean number/phone reading — after aggregation, before synthesis (§10.2).
#    MUST NOT raise: an exception here produces no audio for the turn AND ticks
#    the zero-audio counter (tts_service.py:1252-1262).
async def read_korean_numbers(text: str, aggregation_type) -> str:
    try:
        return korean_numeral_rules(text)
    except Exception:
        return text          # speak it untransformed rather than lose the turn
tts.add_text_transformer(read_korean_numbers)

pipeline = Pipeline([
    transport.input(),
    stt,
    user_aggregator,
    llm,
    korean_text,             # <- new: LLMTextFrame -> AggregatedTextFrame
    tts,
    transport.output(),
    assistant_aggregator,    # <- AFTER output(): word-paced context (§6.6)
])
```

Four things that sketch is deliberately doing:

1. **Korean logic is not inside the TTS service.** It is a processor. Swapping providers does not
   touch it.
2. **The transformer cannot kill a turn.** §10.2's failure mode is handled at the call site.
3. **The assistant aggregator is last.** That is the [[canonical-voice-bot]] position and it is what
   makes ch-08's truncation work.
4. **The provider is a placeholder.** Six candidates, none verified on Korean audio by anything in
   this repo.

### 11.4 The three things you must build yourself

1. **Korean TTS on-prem**, if the deployment needs it. A custom `WebsocketTTSService` subclass
   against a Korean vendor. Nothing in the tree does this (§7.6).
2. **A Korean TTS acceptance test.** The repo has none and cannot have one. `pipecat.evals`
   (`src/pipecat/evals/`) gives you the harness — a real bot, scripted turns, an LLM judge — and no
   Korean scenario exists in `scripts/release-evals/`. Writing one is the only way §7.4's shortlist
   becomes a choice.
3. **Under-real-time detection.** If synthesis falls below 1×, the customer hears a mid-sentence gap
   and no metric in this tree reports it (§1.5). Queue depth at `transport.output()` is the signal;
   nothing exposes it today.

---

## 12. What this layer does not give you

Stated plainly, because absence is evidence:

- **No mid-utterance underrun detection.** §1.5.
- **No cross-provider TTFA comparison harness.** You get the metric per service; comparing two means
  running two bots.
- **No prosody control beyond what the provider's settings expose.** `text_transforms` operate on
  text; SSML support is per-provider (Azure has it, Cartesia has its own tag dialect at
  `cartesia/tts.py:457` — `_CARTESIA_TAG_RE` over `spell|emotion|break|volume|speed`).
- **No Korean-native provider integration.** §7.6.
- **No behavioural verification of any language claim.** §7.4.
- **No character-level span tracking through the TTS path.** The unit is the frame. If you need
  "which characters of the original string were heard", either the word-frame path (§6.3) at 어절
  resolution, or something you build.
- **No back-pressure from TTS to the LLM.** `pause_frame_processing` (`:164`) halts *inbound frame
  processing* while audio is still to play, which is not the same thing — the LLM keeps generating
  into a queue.

---

## 다음 챕터로

This chapter hands forward exactly three things.

**A number and its decomposition.** TTFA, not TTFB, with `leading_silence` measured by an onset
detector rather than reported by a vendor, and split three ways — aggregation delay, provider TTFB,
leading silence — with a metric type for each. [[ch-11/read]] builds the full latency budget and this
is one of its four legs; when it draws a waterfall, the three components above are three of the bars,
and `TextAggregationMetricsData` is the one that will surprise you.

**The word-timestamp hook, with its price.** `TTSTextFrame.pts` in nanoseconds, released by
`transport.output()`'s clock priority queue at presentation time, delivered to an assistant aggregator
sitting downstream of the transport. That is the whole mechanism [[ch-08/read]] uses to truncate the
assistant context at the last **spoken** word rather than the last generated token — and §7.4 is the
price: six Korean-capable providers, from a grep, none of them verified by ear. Go into ch-08 knowing
that its central capability is conditional on a vendor choice you have not yet made.

**Two corrections and one honest gap.** `AudioContextTTSService` is deprecated (§0.1) — the live
ladder is three classes on one axis, not four on two. Korean word grouping is unit-tested, four times
over (§0.2) — what is untested is anything you could hear. Both corrections came from opening files
the outline described from memory, and both changed a conclusion. That is the habit this course is
trying to build, and it is why §7.4 states its provenance in a block quote instead of a footnote.

What is deliberately **not** here: any judgement about whether `KoreanPhraseChunker`'s explicit guards
or NLTK-plus-lookahead is the thing to run in production. §9.3 ran the same six inputs through both
and printed the results. Five agree. One differs, on an input you would have to try to produce. That
is a measurement and it stays a measurement until [[ch-13/read]], which will have ch-08's interruption
mechanics, ch-09's context ownership and ch-11's latency budget in hand — none of which you have yet.

Next is [[ch-08/read]]. It takes the barge-in cascade apart frame by frame, and it opens on the
question this chapter set up and did not answer: when 고객님 cuts Lina off 1.4 seconds into a 4.2
second sentence, what exactly does the assistant context end up containing — and how many distinct
mechanisms had to agree for it to be the right thing?
