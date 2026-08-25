---
title: "사용자의 turn은 언제 끝나는가: VAD, Streaming STT, 그리고 turn-strategy chain"
chapter: ch-06
phase: voice-io
course: pipecat
kind: korean-companion
source: [[read]]
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
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# ch-06 — 사용자의 turn은 언제 끝나는가

> 이 문서는 [[read]] (`ch-06/read.md`)의 한국어 companion입니다. section 번호가 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 전부 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 — frame, processor, pipeline, queue, aggregator, endpointing,
> back-pressure, interruption, timestamp, strategy, chain, veto, watchdog, hysteresis 등.

> **범위(scope), 미리 밝히고 chapter 전체에 강제합니다.** [[ch-03/read]]과 같은 규칙입니다: 이 chapter는
> **mechanism과 evidence만** 다룹니다. Pipecat과 realtime_voice가 둘 다 무언가를 구현한 곳에서는, 각각이
> *무엇을 하는지*만 진술하고 거기서 멈춥니다. 표를 던지지 않고, 비교급 형용사를 쓰지 않으며, 판정에
> 이르지 않습니다. 점수 매기기는 [[ch-13/read]]의 일입니다.
>
> 다만 이 chapter가 *하는* 두 가지가 있습니다. 선호가 아니라 사실의 진술이기 때문입니다: 각 동작이
> 어떤 제품 상황에서 부채(liability)가 되는지를 — 두 design 모두에 대해 대칭적으로 — 이름 붙이는 것,
> 그리고 source code가 source excerpt와 어긋나는 지점에서 excerpt를 정정하는 것. 그런 정정이 아래에
> 세 개 나오며 **SOURCE CORRECTION**으로 표시됩니다.

---

## 왜 이 챕터인가

이 chapter 전체를 조직하는 질문은 하나입니다:

> **사용자의 turn은 언제 끝나는가?**

아래의 모든 내용이 그 질문에 대한 답이며, 각각 다른 layer에서의 답입니다. 그리고 layer들은 dependency
순서로 가르쳐집니다. 그 순서는 문체상의 선택이 아닙니다. transcript layer의 timer들은 VAD의 양(quantity)으로
계량되고, strategy layer의 safety net은 STT의 양으로 계량됩니다. 순서를 뒤집어 배우면 마주치는 모든 숫자가
magic constant가 됩니다.

- **1부 — audio-level 답.** Silero가 "이 32 ms chunk에 voice가 있다"고 말합니다. 네 개 state를 가진
  hysteresis machine이 그 sequence를 "사용자가 말하고 있다"로 바꿉니다. 그것이 §1–§5입니다.
- **2부 — transcript-level 답.** streaming STT가 "이게 사용자가 말한 내용이고, 더 보낼 것이 없다"고
  말합니다. 그것이 §6–§12입니다.
- **3부 — 중재자(arbiter).** 위의 어느 것도 turn을 선언하지 않습니다. 꽂아 넣을 수 있는(pluggable)
  strategy들의 chain이 둘 다를 소비해서 판정을 협상하고, chain의 어떤 link든 veto할 수 있습니다.
  그것이 §13–§18입니다.

이 course에서 이 chapter가 끝나는 시점까지 가르쳐지지 않은 VAD 양으로 계량되는 것은 이후에 아무것도
나올 수 없습니다. [[ch-08/read]]은 `start_secs`를 barge-in latency에 씁니다. [[ch-11/read]]은 `stop_secs`와
P99 표를 latency budget에 씁니다. 둘 다 다시 유도하지 않고 이 chapter를 인용합니다.

**이미 가지고 있는 것, 그리고 이 chapter가 반복하지 않을 것.** [[ch-03/read]] §4는 이미 `VADParams`,
`VADState`, `_run_analyzer` state machine, `num_frames_required()`를 지면에 올렸고, two-frame blip을 두
machine에 각각 통과시켜 보았습니다. [[ch-03/read]] §5는 이미 `STTService` / `SegmentedSTTService`의
docstring을 나란히 놓았습니다. [[ch-05/read]]은 Pipecat의 전화 통화가 WebSocket transport + `FrameSerializer`
라는 것, 그리고 여섯 개의 telephony serializer 중 다섯이 wire 위에서 8 kHz μ-law라는 것을 확립했습니다.
이 chapter는 그중 어느 것도 다시 가르치지 않습니다. 그 *밑으로* 들어갑니다: ch-03이 이름 붙이지 않은
three-way split, ch-03이 한 번도 보여주지 않은 controller, 두 sample rate 모두로 계량된 산술, 그리고
ch-03이 손짓만 할 수 있었던 세 번째 layer 전체 — strategy chain.

---

## 0. 코드를 보기 전에, 답의 모양

이런 걸 한 번 만들어 본 사람들을 놀라게 하는 사실이 있고, 한 줄이라도 읽기 전에 머릿속에 넣어 둘
가치가 있습니다:

**Pipecat에는 "turn이 끝났다고 결정하는" 것이 일인 component가 없습니다.** `endpointing.py`가 없습니다.
`EndpointingConfig`가 없습니다. `TurnDetector`라는 이름의 class가 없습니다. 확인했습니다:

```bash
# run in wiki/raw-data/pipecat/pipecat-src at commit 0cbf9c5b
$ find src -name "endpointing*"
$ grep -rn "EndpointingConfig" src/
$
```

둘 다 아무것도 반환하지 않습니다. 이 tree에서 "endpointing"이라는 단어는 오직 *provider* setting으로만
살아남아 있습니다 — Deepgram의 `endpointing` parameter, Speechmatics의 `TurnDetectionMode`, Sarvam의
`endpointing: Literal["vad", "manual"]`. framework 자신은 그 개념을 소유하기를 거부합니다.

대신 그 결정은 의도적으로 서로를 모르도록 유지된 세 개의 layer로 분해됩니다:

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

저 diagram의 두 frame family를 주의 깊게 읽으십시오. naming이 곧 design 전체이고, 그냥 훑고 지나가기
쉽습니다:

| Frame | Meaning | Who emits |
|---|---|---|
| `VADUserStartedSpeakingFrame` | *there is voice* | `VADController`, via the analyzer |
| `UserStartedSpeakingFrame` | *this is a turn* | `UserTurnProcessor`, via a start strategy |

첫 번째는 Silero가 결정합니다. 두 번째는 **strategy**가 결정합니다. 서로 다른 명제에 대한 서로 다른
단어이고, Pipecat은 그것들에 서로 다른 type을 부여합니다. 이 layer의 모든 혼란은 그 둘을 뭉개는 데서
나옵니다.

> 💡 **쉬운 설명 — "voice가 있다"와 "turn이다"가 왜 다른 명제인가요?**
> 고객이 통화 중에 "아…" 하고 헛기침을 합니다. 마이크에는 분명히 voice가 있습니다 — Silero의 confidence는
> 높게 나옵니다. 하지만 그것은 발언권(floor)을 가져가겠다는 주장이 아닙니다. 반대로, 아주 조용히 말하는
> 고객은 `min_volume` gate를 못 넘어서 VAD가 침묵으로 읽을 수 있지만, STT는 transcript를 뱉습니다 —
> voice 신호는 없는데 turn은 있는 겁니다. 두 명제가 어긋날 수 있기 때문에 type을 나눈 것이고,
> `TranscriptionUserTurnStartStrategy`(§14)가 존재하는 이유가 바로 두 번째 경우입니다.

---

# PART ONE — THE AUDIO-LEVEL ANSWER: VAD

## 1. three-way split

[[vad-silero]]가 그 split에 이름을 붙이고, source가 그것을 뒷받침합니다: Pipecat의 VAD는 세 개 파일에
있는 세 개의 object이며, 각각은 나머지를 건드리지 않고 교체 가능합니다.

| Object | File | Line | Job |
|---|---|---|---|
| `SileroOnnxModel` | `audio/vad/silero.py` | 34 | per-chunk confidence, ONNX inference |
| `SileroVADAnalyzer(VADAnalyzer)` | `audio/vad/silero.py` | 130 | model adapter: chunk size, int16→float32, state reset |
| `VADAnalyzer(ABC)` | `audio/vad/vad_analyzer.py` | 63 | the hysteresis machine — four states, two counters |
| `VADController(BaseObject)` | `audio/vad/vad_controller.py` | 31 | edge-triggered event emitter |

`vad_analyzer.py`는 255줄, `silero.py`는 226줄, `vad_controller.py`는 244줄입니다. 그것이 framework의 VAD
surface 전부입니다: 725줄.

### 1.1 model — 그것이 실제로 무엇인가

`SileroOnnxModel`은 얇은 ONNX Runtime wrapper이고, 그 constructor의 결정 중 두 가지가 telephony host에게
중요합니다:

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

두 thread count가 모두 1로 고정되어 있고 CPU가 기본으로 강제됩니다. 그것은 의도적인 concurrency 결정이며,
[[ch-04/read]] §13의 topology — 통화 하나당 `PipelineWorker` 하나, 모든 session이 하나의 loop 위의
`asyncio.Task` — 를 생각하면 알아차릴 가치가 있는 옳은 결정입니다. 만약 VAD model이 session마다 자기
thread pool을 띄운다면 동시 통화 40건에서 N × cores 개의 thread가 생깁니다. 그렇게 하지 않습니다.
각 analyzer는 정확히 하나의 worker thread를 갖고, 그 안의 model은 single-threaded입니다:

**`src/pipecat/audio/vad/vad_analyzer.py` L90–92**

```python
        # Thread executor that will run the model. We only need one thread per
        # analyzer because one analyzer just handles one audio stream.
        self._executor = ThreadPoolExecutor(max_workers=1)
```

`__init__`의 마지막 줄에 있는 `self.sample_rates = [8000, 16000]`은 [[ch-05/read]]의 telephony chapter가
기다려 온 사실입니다. 붙들고 계십시오. §2.3이 그것을 씁니다.

두 번째 model-level 결정은 강제된 건망증(forced amnesia)입니다:

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

`_MODEL_RESET_STATES_TIME = 5.0` (L23). 5초마다 recurrent state가 0으로 초기화됩니다. 그리고 `except`
절은 *어떤* exception에 대해서든 `0` — 가능한 가장 낮은 confidence — 을 반환합니다. 잘못된 buffer는
통화를 crash시키지 않고, **침묵으로 읽힙니다.** 20분짜리 보험 sales call에서 이것은 모양을 알아 둬야 할
failure mode입니다: upstream의 무언가가 analyzer에게 나쁜 buffer를 먹이기 시작하면, 증상은 log의
exception이 아니라 **고객의 말을 더 이상 듣지 못하는 bot**입니다.

> 💡 **쉬운 설명 — 왜 이 `return 0`이 특히 위험한가요?**
> "실패하면 예외를 던진다"면 monitoring이 잡습니다. "실패하면 0을 반환한다"는 *유효한 답처럼 보이는*
> 값을 돌려주는 것이고, 0은 이 시스템에서 "완전한 침묵"과 구별되지 않습니다. resampler 버그나
> 잘린 μ-law frame 때문에 buffer가 계속 망가지면, 통화는 정상적으로 이어지고 log는 조용하며,
> bot만 영원히 고객 차례를 기다립니다. §1.4의 `audio_idle_timeout`도 이걸 못 잡습니다 —
> audio frame은 계속 *도착하고* 있으니까요. 잡히는 건 `logger.error` 줄뿐이니, 그 line을
> alert에 걸어 두십시오.

### 1.2 analyzer — ch-03이 말하지 않은 두 가지

[[ch-03/read]] §4.1이 이미 `VADParams`, `VADState`, `_run_analyzer` match block을 인용했습니다. 다시
인용하지 않습니다. 그 machine의 두 가지 속성이 거기서 진술되지 않았고, 둘 다 여기서 중요합니다.

**첫째: gate는 per-chunk AND이고, 그중 한쪽만 *model* 출력입니다.**

**`src/pipecat/audio/vad/vad_analyzer.py` L206–211**

```python
            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume
```

`confidence`는 Silero에서 옵니다. `volume`은 아닙니다 — 그것은 `AudioVolumeTracker`를 고정된 factor의
exponential smoothing에 통과시킨 값입니다:

**`src/pipecat/audio/vad/vad_analyzer.py` L173–176**

```python
    def _get_smoothed_volume(self, audio: bytes) -> float:
        """Calculate smoothed audio volume using exponential smoothing."""
        self._volume_tracker.update(audio, self.sample_rate)
        return exp_smoothing(self._volume_tracker.volume, self._prev_volume, self._smoothing_factor)
```

`self._smoothing_factor = 0.2` (L87)입니다. 따라서 `min_volume = 0.6`은 현재 chunk의 순간 RMS가 아니라
**smoothing된, stateful한** 신호에 대한 threshold입니다. 한 번의 큰 transient는 이 gate를 통과시키지
못하고, 지속되는 energy만 통과시킵니다. 그것이 이 AND를 "같은 증거에 대한 두 번째 threshold"가 아니라
noise gate로 만들어 주는 요소입니다.

이것이 또한 [[vad-silero]]의 guideline이 barge-in을 `confidence`가 아니라 timing param으로 tuning하라고
말하는 이유이기도 합니다: `confidence`와 `min_volume`은 둘 다 하나의 32 ms window에 대한 *per-chunk*
gate이고, 어떤 per-chunk gate도 기침과 음절을 구별할 수 없습니다. *연속된* chunk를 세는 timing param만이
그 구별을 사줍니다.

> 💡 **쉬운 설명 — exponential smoothing이 왜 "noise gate"가 되나요?**
> `exp_smoothing`은 `new = α·현재값 + (1-α)·직전값` 형태이고 여기서 α = 0.2입니다. 즉 현재 chunk의
> 기여는 20%뿐입니다. 문 닫히는 "쾅" 소리처럼 한 chunk만 크게 튀는 신호는 smoothed 값을 20%만
> 밀어 올리므로 0.6을 넘기 어렵습니다. 반면 사람이 계속 말하면 매 chunk마다 밀려 올라가 몇 chunk 만에
> threshold를 넘습니다. Silero는 "이게 목소리처럼 들리나"를 보고, volume gate는 "이게 *지속되고* 있나"를
> 봅니다. 두 개가 AND로 묶여야 순간적인 소음이 걸러집니다.

**둘째: `set_params`는 state machine을 reset하지만 volume tracker는 의도적으로 reset하지 않습니다.**

**`src/pipecat/audio/vad/vad_analyzer.py` L166–171**

```python
        # VAD state resets, but volume state doesn't: the rolling window and its
        # smoothing follow the audio stream, which is continuous across
        # parameter changes.
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET
```

주석이 invariant를 정확하게 진술합니다: parameter는 *policy*의 속성이고, smoothed volume은 *stream*의
속성이며, `VADParamsUpdateFrame`을 보냈다고 해서 stream이 멈춘 것은 아닙니다. 언젠가 Lina를 위해
mid-call VAD 재튜닝을 만든다면 — 예컨대 시끄러운 발신자를 감지한 뒤 gate를 느슨하게 하는 식 — 이 줄이
그 재튜닝이 당신의 noise floor 추정치를 날려버리지 않는다는 것을 말해 줍니다.

### 1.3 controller — ch-03이 한 번도 보여주지 않은 object

`VADController`는 state machine이 event가 되는 곳이고, 나머지 pipeline이 *관찰할 수 있는 것 자체*를
바꾸는 일을 하나 합니다.

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

세 조건이 AND로 묶여 있습니다. 첫 번째는 edge-detection입니다: *transition*에서만 발화합니다. 두 번째와
세 번째가 흥미로운 것들입니다 — `STARTING`과 `STOPPING`이 명시적으로 걸러집니다.

**`STARTING`과 `STOPPING`은 analyzer 바깥에서 관찰 불가능합니다.** 어떤 frame도 그것들을 나르지 않습니다.
어떤 event도 그것들에 대해 발화하지 않습니다. 그것들은 순전히 내부의 가설 보류(hypothesis-holding)
state로만 존재합니다. pipeline의 관점에서 이 four-state machine은 정확히 두 개의 *관찰 가능한* state를
가지며, 나머지 둘은 그 둘 중 어느 쪽을 통보받을지를 결정하는 사적인 기계 장치입니다.

그것이 four-state machine이 "false start를 조용히 버린다"는 말의 정확한 의미입니다: downstream에 filtering
단계가 있는 게 아니라, 애초에 filtering할 downstream 신호가 존재한 적이 없습니다.

다섯 개 event 전부가 `sync=True`로 등록됩니다:

**`src/pipecat/audio/vad/vad_controller.py` L106–110**

```python
        self._register_event_handler("on_speech_started", sync=True)
        self._register_event_handler("on_speech_stopped", sync=True)
        self._register_event_handler("on_speech_activity", sync=True)
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
```

[[ch-04/read]] §2에 따르면 `sync=True`는 handler가 background task로 schedule되는 대신 inline으로
실행된다는 뜻입니다. speech-start는 barge-in trigger입니다. 그것을 loop의 task queue 뒤에 schedule하는
것은 [[ch-04/read]] §5.1의 `N/r` 산술이 경고하는 바로 그 종류의 latency를 더하는 일입니다. 이것은 그
교훈을 framework 자신이 적용하고 있는 모습입니다.

### 1.4 `audio_idle_timeout` — audio가 도착을 멈출 때를 위한 watchdog

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

기본값은 `audio_idle_timeout: float = 1.0` (L75)입니다. 이것이 커버하는 failure mode는 이국적인 게 아니라
구조적인 것입니다: hysteresis machine은 chunk가 도착할 때만 전진합니다. *state가 `SPEAKING`인 동안*
audio가 멈추면, state를 `STOPPING`으로 옮겨 줄 chunk가 없으므로 turn이 영원히 끝나지 않고 고객은 영원히
기다립니다. 강제되는 state가 `QUIET`이고 `STOPPING`을 통째로 건너뛴다는 점에 주목하십시오 — 그것이
바로 `on_speech_stopped` event가 L179–183의 filter를 통과해 발화하게 만드는 요소입니다.

Lina에게 이것은 mic-mute 시나리오가 아니라 carrier 시나리오입니다. 발화 도중 `media` event 전달을 멈춘
Twilio media stream — network blip, carrier 딸꾹질, one-way audio — 은 analyzer를 `SPEAKING`에 고정시킵니다.
1초 뒤 watchdog이 turn을 닫고 pipeline이 전진합니다. 이것이 없으면 통화는 조용히 죽어 있으면서 과금은
계속됩니다.

---
## 2. 두 sample rate 모두에서, 손으로 하는 산술

이것이 나머지 course가 계량되는 기준이 되는 section입니다. 손으로 하십시오. 나눗셈 네 줄이고 틀리기
쉽습니다 — excerpt들이 틀렸고, outline도 두 번 틀렸습니다.

### 2.1 변환

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

외워 둘 line number 네 개입니다. [[ch-08/read]]과 [[ch-11/read]]이 둘 다 이것들을 인용하기 때문입니다:

- `def set_params` — **L151**
- `vad_frames_per_sec = self._vad_frames / self.sample_rate` — **L162**
- `self._vad_start_frames = round(...)` — **L164**
- `self._vad_stop_frames = round(...)` — **L165**

`vad_frames_per_sec`는 오해를 부르는 이름입니다. 그것은 chunk 당 초 수(**seconds per chunk**)이지, 초당
chunk 수가 아닙니다. [[ch-03/read]] §4.2가 이미 이것을 지적했습니다. 두 번 지적할 가치가 있습니다.
파일을 다시 읽을 때마다 그 이름이 당신을 속일 것이기 때문입니다.

### 2.2 16 kHz에서

`num_frames_required()`는 512를 반환합니다 (`silero.py` L191–197, [[ch-03/read]] §4.1에서 이미 인용).

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

**7이 아니라 6입니다.** Python의 `round()`는 banker's rounding — 절반을 짝수로 반올림 — 이고, 어차피
6.25는 절반 케이스도 아닙니다. `round(6.25)`가 6인 이유는 6.25가 7보다 6에 더 가깝기 때문입니다.
직접 확인하십시오:

```bash
$ python3 -c "print(round(0.2 / (512/16000)))"
6
```

> **SOURCE CORRECTION #1.** [[vad-silero]]는 *"`start_secs=0.2` → **7 consecutive chunks**
> (`round(6.25)`)"*라고 진술하고, [[rtv-vad-chunking]]의 비교 표는 두 행에서 **7**을 반복합니다. 둘 다
> 틀렸습니다. `vad_analyzer.py` L164의 코드는 6을 계산합니다. [[ch-03/read]] §4.2가 이미 이 정정을
> 했습니다. 여기서 다시 진술하는 이유는 이 chapter와 [[ch-11/read]]의 downstream 숫자 전부가 여기에
> 매달려 있고, 32 ms 오차는 latency budget을 통과하며 누적되기 때문입니다.

그래서 진짜 값들은 이렇습니다. 이제 아무것도 찾아보지 않고 진술할 수 있어야 합니다:

| Quantity | Value at 16 kHz |
|---|---|
| chunk duration | 32 ms |
| chunks to confirm speech start | 6 |
| **onset detection lag** | **192 ms** |
| chunks to confirm speech stop | 6 |
| **offset detection lag** | **192 ms** |

200 ms가 아니라 192 ms입니다. rounding이 양쪽에서 각각 8 ms를 앗아갑니다. 아무도 알아채지 못하겠지만,
당신이 실제로 돌리고 있는 숫자가 config에 적힌 숫자가 아니라는 것은 알아야 합니다.

### 2.3 8 kHz에서 — Lina TMR이 그 위에서 돌아가는 사실

8 kHz에서 `num_frames_required()`는 256을 반환합니다. 나눗셈을 다시 하십시오:

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

**chunk 개수도 같고, wall-clock lag도 같습니다.** Silero의 8 kHz chunk는 16 kHz chunk의 절반 sample이고,
8 kHz는 절반 rate이므로 둘이 정확히 상쇄됩니다. 512/16000과 256/8000은 둘 다 0.032입니다.

이것이 §1.1의 `self.sample_rates = [8000, 16000]` 줄의 payoff입니다. [[ch-05/read]]은 여섯 개 telephony
serializer 중 다섯이 wire 위에 8 kHz μ-law를 올린다는 것을 확립했습니다 —
`twilio_sample_rate: int = 8000`, `telnyx_sample_rate: int = 8000`,
`plivo_sample_rate: int = 8000`, `exotel_sample_rate: int = 8000`,
`genesys_sample_rate: int = 8000`, 그리고 `VonageFrameSerializer`가 `vonage_sample_rate: int = 16000`으로
예외 ([[transport-telephony]], 여섯 개 모두 serializer source에서 검증됨). 따라서 당신에게 주는 구체적인
결과는:

> **Lina TMR의 VAD tuning은 browser demo에서 전화 통화로 옮겨갈 때 바뀌지 않습니다.**
> `start_secs=0.2`와 `stop_secs=0.2`는 두 rate 모두에서 같은 여섯 chunk, 같은 192 ms를 뜻합니다.
> 이 layer에서 telephony 재튜닝으로 잡아 둘 예산은 없습니다.

이것은 정말로 좁은 주장이고, 저는 그 울타리를 정확히 치고 싶습니다. 울타리가 바로 일이 있는 곳이기
때문입니다. 이전되는 것은 **timing 산술**입니다. 이전되지 *않는* 것은 **acoustics**입니다:
[[transport-telephony]]는 8 kHz μ-law가 4 kHz Nyquist 천장과 8-bit companding을 가진다고 명시하므로,
telephony audio에 대한 Silero의 *confidence 값*은 studio audio에서와 다른 분포이고, `confidence = 0.7` /
`min_volume = 0.6`은 그 분포에 대한 threshold입니다. frame 개수는 재튜닝이 필요 없습니다. threshold는
열린 질문이고 이 repository의 어느 것도 답하지 않습니다. §13은 그것을 사실이 아니라 당신이 스스로에게
빚진 측정으로 기록합니다.

> 💡 **쉬운 설명 — "sample 수는 복원되지만 대역폭은 복원되지 않는다"**
> 8 kHz로 녹음된 신호에는 4 kHz 이상의 성분이 물리적으로 존재하지 않습니다(Nyquist). upsampling은
> sample 개수를 16000개/초로 늘려 줄 뿐, 사라진 고주파를 만들어내지 못합니다. 한국어에서 ㅅ/ㅆ/ㅊ
> 같은 마찰음·파찰음의 에너지는 상당 부분 4 kHz 위에 있습니다. 그래서 "16 kHz model에 넣을 수 있다"와
> "16 kHz 성능이 나온다"는 완전히 다른 문장입니다. 이 구분은 §12.5와 §14.1에서 두 번 더 나옵니다.

두 rate만 허용하는 세계를 강제하는 guard:

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

→ **지금 [`figures/turn-boundary.html`](figures/turn-boundary.html)을 열고 panel one을 사용하십시오.**
frame-count 산술이 답이 아니라 살아 있는 계산으로 그려져 있습니다 — L162의 나눗셈, L164의 `round()`,
그리고 256/8000을 같은 6으로 재계산하는 sample-rate 스위치. 넘어가기 전에 한 가지만 해 보십시오:
`start_secs`를 드래그해서 chunk 개수가 바뀌는 값을 찾으십시오. 예상한 곳이 아닙니다. `round()`가
half-chunk 경계에서 계단을 밟기 때문입니다 — 0.208 s는 여전히 6, 0.209 s는 7입니다. 그 계단 함수가
`start_secs`가 연속적인 latency 다이얼처럼 굴지 않는 이유입니다.

---

## 3. VAD가 내보내는 것은 turn이 아니다

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

여기서 두 개의 design 결정이 보이고, 둘 다 이 chapter의 뒷부분에서 쓰입니다.

**frame이 자기를 만들어 낸 parameter를 나릅니다.** `stop_secs`는 downstream consumer가 config에서 읽는
값이 아니라 frame에 실려 옵니다. 그것이 consumer로 하여금 VAD가 *알아챈* 시점이 아니라 *사용자가 실제로
말을 멈춘* 시점을 재구성할 수 있게 해 줍니다 — §9와 §19가 정확히 그 뺄셈을 하고, frame이 자기 자신을
기술하기 때문에만 그렇게 할 수 있습니다.

**둘 다 `SystemFrame`입니다.** [[ch-04/read]] §4.1에 따르면 system frame은 priority path를 타고, 취소
가능한 process task에서 data 뒤에 queue되는 대신 input task에서 inline으로 처리됩니다. buffer된 audio
뒤에 줄 서는 turn 신호는 자기가 기술하기로 되어 있던 audio보다 늦게 도착할 것입니다. 그렇게 되지
않습니다.

"voice"에서 "turn"으로의 변환은 35줄짜리 파일 안의 `isinstance` 하나입니다:

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

그것이 `VADUserTurnStartStrategy` 전부입니다. 그 작음이 요점입니다: audio layer의 일은
`VADUserStartedSpeakingFrame`에서 끝나고, *policy object* — 교체 가능하고, 쌓을 수 있고, veto 가능한 —
가 그것을 turn으로 바꿉니다. 이 파일을 `MinWordsUserTurnStartStrategy`로 바꿔치기해도 VAD는 알지도
못하고 신경 쓰지도 않습니다.

---

## 4. analyzer가 어디에 mount되는가, 그리고 왜 당신이 짐작할 곳이 아닌가

이것은 voice-I/O phase에서 가장 놀라운 배선 사실이고, [[ch-05/read]]이 이미 기록한 제거의 직접적인
결과입니다.

```bash
$ grep -n "vad_analyzer\|vad_enabled\|turn_analyzer" src/pipecat/transports/base_transport.py
$
```

아무것도 없습니다. `TransportParams`에는 **`vad_analyzer` field도 `vad_enabled` field도 없습니다.**
이 commit 이전에 쓰인 Pipecat tutorial을 읽어 본 적이 있다면, 그 전부가
`TransportParams(vad_enabled=True, vad_analyzer=SileroVADAnalyzer())`를 넘깁니다. 그 API는 사라졌습니다:

**`CHANGELOG.md` L4402–4406**

```
- ⚠️ Removed `vad_analyzer` and `turn_analyzer` parameters from
  `TransportParams` and all transport input classes, along with all deprecated
  VAD/turn analysis logic in `BaseInputTransport`. VAD and turn detection are
  now handled entirely by `LLMUserAggregator`.
  (PR [#4229](https://github.com/pipecat-ai/pipecat/pull/4229))
```

이제 mount point는 정확히 두 개입니다.

**Mount point 1 — aggregator params.** 이것이 정석입니다:

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

`@dataclass class LLMUserAggregatorParams`는 **L119–120**에 선언되어 있습니다. 이 chapter가 가르치는 모든
knob이 그 하나의 dataclass 위에 있습니다: analyzer, strategy들, mute strategy들, stop watchdog, idle
timeout, 그리고 §1.4의 controller `audio_idle_timeout`. aggregator가 `VADController`를 직접 만듭니다:

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

L750의 조건문에 주목하십시오: **analyzer가 없으면, controller도 없고, VAD frame도 전혀 없습니다.**
`vad_analyzer=`를 잊는다고 error나 warning이 나지 않습니다. 나오는 것은 speech onset을 절대 감지하지 못하고
전적으로 `TranscriptionUserTurnStartStrategy`에 의존하는 bot입니다 — 그것은 *지원되는* 구성이지만(§17),
당신이 구성했다고 생각한 그것은 아닙니다. 이 layer에서 가장 일어나기 쉬운 misconfiguration이고, 조용히
실패합니다.

정석 배선, 그대로:

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

**Mount point 2 — 독립 processor.** LLM aggregator가 없는 pipeline을 위해:

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

같은 `VADController`, 같은 default, 다른 host. transcription 전용 bot이 이것을 씁니다.

**왜 transport가 아니라 aggregator인가?** 판단이 아니라 기계적으로 이유를 진술하십시오. turn 경계는
audio 증거와 transcript 증거가 *둘 다* 필요한 결정입니다 — §17의 기본 start list에는 각각을 위한
strategy가 하나씩 있습니다. transport는 audio만 봅니다. aggregator는 STT의 downstream에 앉습니다(정석
chain에서 position 3, position 2의 `stt` 바로 뒤). 따라서 pipeline에서 둘 다를 보는 첫 processor입니다.
두 입력이 모두 available한 곳에 결정을 놓은 것이 parameter가 이동한 구조적 이유입니다.

---

## 5. realtime_voice에 대한 mechanism differential

[[ch-03/read]] §4.2가 이미 parameter 대 parameter 표를 만들었고 §4.4가 이미 two-frame blip을 두 machine에
통과시켰습니다. 둘 다 반복하지 않습니다. 아래는 ch-03이 다룰 수 없었던 부분입니다. §1과 §4가 먼저
필요했기 때문입니다: *구조적* differential.

**이것은 differential이지 ranking이 아닙니다.** 아래에 비교급 형용사는 나오지 않고 추천도 하지 않습니다.
[[ch-13/read]]가 두 design 중 어느 것이든 점수가 매겨지는 유일한 곳입니다.

### 5.1 각 design이 구조적으로 가진 것

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

두 행은 표 칸에 남겨 두지 말고 소리 내어 이름을 불러 줄 가치가 있습니다.

**hypothesis-state 행.** two-state machine에는 "어쩌면 speech"를 담을 표현이 없습니다. 매 chunk마다
확정해야 합니다. four-state machine은 그 가설을 `STARTING`에 담아 두고 철회할 수 있으며, controller가
`STARTING`을 걸러내기 때문에 그 철회는 downstream에 아무 비용도 들지 않습니다. 같은 two-frame blip을
먹였을 때: Pipecat의 machine은 `QUIET → STARTING → QUIET`로 가고 아무것도 내보내지 않습니다.
realtime_voice의 machine은 `self._speaking`을 `True`로 뒤집고 `SPEECH_STARTED`를 내보내며,
`VoiceSession._on_speech_started`가 그것을 generation 전진과 assistant 취소로 바꿉니다
([[rtv-vad-chunking]], [[rtv-pipeline-session]]). 두 machine 모두 자기가 쓰인 대로 하고 있습니다.

**self-measured-latency 행.** realtime_voice는 VAD layer에서 자기 자신의 endpoint latency를 계산하고
보고합니다. Pipecat은 그러지 않습니다. 그 숫자는 대신 observer plane을 통해 드러납니다([[ch-11/read]]).
이곳은 realtime_voice가 이 layer에서 Pipecat에 대응물이 없는 mechanism을 가진 한 곳이고, 정확히 그
이유로 differential에 들어갑니다.

### 5.2 각 동작이 부채가 되는 제품 상황

대칭적으로 진술합니다. 이것들은 상황이지 점수가 아닙니다.

**Pipecat의 동작은 제품이 sub-192 ms floor-yield를 필요로 할 때 부채입니다.** 확인된 speech 여섯 chunk는
downstream의 무엇이든 고객이 입을 열었다는 것을 알기까지 192 ms의 onset lag이고, 20 ms frame에서
`min_speech_frames = 2`(40 ms)인 machine은 쓰지 않는 192 ms입니다. assistant의 독백이 고객이 말을
시작하는 순간 멈춰야 하는 제품 — 빠르게 오가는 반론 처리(objection-handling) — 에서 그 차이는 귀에
들리고, four-state machine에는 false-start 거부를 함께 제거하지 않고 그것을 없애는 구성이 없습니다.
**둘이 같은 counter**이기 때문입니다.

**realtime_voice의 동작은 입력에 지속적인 비-speech energy가 있을 때 부채입니다.** `min_volume` 항이
없고 confidence threshold가 0.5이므로, threshold를 넘는 두 frame이면 진짜 `SPEECH_STARTED`를 만들고,
[[rtv-vad-chunking]]에 따르면 그것은 즉시 generation을 전진시키고 assistant를 취소합니다. tele-sales
통화에서 지속적 energy 원천은 가설이 아닙니다: 통화 대기음악, 고객 방의 TV, 근처에서 말하는 두 번째
사람, 그리고 고객이 *듣는 동안* 내뱉으며 발언권 주장으로 의도하지 않은 한국어 backchannel("네", "아",
"음"). 각각이 취소된 assistant turn 하나씩을 대가로 치릅니다.

**그리고 세 번째, 어느 design에도 속하지 않는 것.** 두 시스템 모두 VAD layer에서 backchannel을 걸러내지
않습니다. Pipecat에는 `audio/vad/` 안에 "네/응을 무시하라"는 mechanism이 없습니다. boson의 그것은
Gateway 안에 `WordFilterPolicy`로 있습니다([[boson-interrupt-subsystem]]). 그 semantic gate는 당신이 직접
쓰는 `BaseUserTurnStartStrategy` subclass입니다 — §20이 그것을 probe로 만듭니다.

**trade-off가 아니라 사실인 비대칭 하나:** realtime_voice에는 8 kHz path가 아예 없습니다.
`vad/silero.py:58`의 `ValueError("SileroVAD requires 16 kHz mono PCM")`은 저하된 모드가 아니라 hard raise
입니다. 8 kHz로 도착하는 telephony audio는 VAD가 보기 전에 16 kHz로 resample되어야 하고, 아니면 통화가
실패합니다. Pipecat의 analyzer는 8 kHz를 native로 받습니다. 그것은 tuning 차이가 아니라 capability
차이이고, §2.3이 그것이 무엇을 사주는지 정량화했습니다: timing에서는 아무것도 바뀌지 않습니다.

→ **[`figures/turn-boundary.html`](figures/turn-boundary.html)의 panel one에 blip injector가 있습니다.**
같은 two-frame 입력을 두 machine에 흘려 보내고, 두 번째 machine의 `SPEECH_STARTED`가 어디에 떨어지는지
보십시오. panel은 둘 다 badge도 승자도 없이 동작으로만 보고합니다. 그것은 의도된 것입니다.

---

# PART TWO — THE TRANSCRIPT-LEVEL ANSWER: STREAMING STT

## 6. 세 개의 base class와 정확히 하나의 abstract method

`src/pipecat/services/stt_service.py`는 **1,040줄**입니다.

> **SOURCE CORRECTION #2.** [[stt-service-interface]]는 1,041 L이라고, `stt_latency.py`는 69 L이라고
> 말합니다. 이 commit에서 `wc -l`은 **1,040**과 **68**을 줍니다. off-by-one이고 무해하지만, excerpt가
> 약간 다른 read를 대상으로 쓰였다는 것과 당신이 파일을 믿어야 한다는 것을 말해 줍니다.

세 개 class, 원할 만한 세 개 line number:

| Class | Line | Shape |
|---|---|---|
| `class STTService(AIService)` | 51 | continuous streaming — every chunk goes to the provider |
| `class SegmentedSTTService(STTService)` | 797 | request/response — one call per utterance |
| `class WebsocketSTTService(STTService, WebsocketService)` | 929 | streaming over a socket, with reconnect |

그리고 셋 전부에 대해 abstract method는 하나입니다:

**`src/pipecat/services/stt_service.py` L334–335**

```python
    @abstractmethod
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
```

line number를 정확히 기록하십시오: **L334가 `@abstractmethod` decorator, L335가 signature입니다.**
outline이 이것을 표시한 이유는 review 세 번을 살아남는 종류의 off-by-one이기 때문입니다.

return type이 곧 interface 전부입니다. `AsyncGenerator[Frame | None, None]` — provider는 *audio chunk 당
0개 이상의 frame*을 yield하고, `None`도 합법적인 yield입니다. streaming provider는 interim이 도착하는
대로 yield하고 utterance가 닫히면 final을 yield합니다. segmented provider는 호출당 정확히 하나의 frame을
yield합니다. 할 말이 없는 provider는 아무것도 yield하지 않습니다. 하나의 signature, 두 개의 shape, 그리고
차이는 전적으로 yield pattern에 있습니다.

constructor가 latency contract를 나릅니다:

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

`ttfs_p99_latency`는 §11의 표가 채우고 §18/§19의 timer가 소비하는 field입니다. 그것은 class가 constructor
argument로 받는 *per-deployment* 숫자이고, benchmark 상수가 default입니다. 붙들고 계십시오. 그것이 3부
전체를 작동하게 만드는 design 결정입니다.

---

## 7. streaming 대 segmented, 기계적으로

### 7.1 audio가 service에 도달하는 방식

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

기본값 `audio_passthrough=True`: STT는 audio를 소비하고 *동시에* 전달합니다. 이것이 중요한 이유는 §19의
turn analyzer가 STT의 downstream에 있으면서 STT가 방금 먹은 것과 같은 audio를 필요로 하기 때문입니다.
`audio_passthrough=False`로 두면 `TurnAnalyzerUserTurnStopStrategy`를 조용히 굶기게 됩니다.

mute는 provider가 단 1바이트도 보기 전에 검사됩니다:

**`src/pipecat/services/stt_service.py` L437–438**

```python
        if self._muted:
            return
```

그것은 `_last_audio_time`보다 두 줄 위이고 `process_generator(self.run_stt(...))`보다 위입니다. Pipecat에서
mute는 *transcript가 버려진다*가 아니라 *provider에게 과금되지 않고 transcript가 생성되지 않는다*는
뜻입니다. §16의 `user_mute_strategies`가 그것을 구동하는 것입니다.

### 7.2 streaming — 기본 경로

`STTService.process_audio_frame`은 여기서 끝납니다:

**`src/pipecat/services/stt_service.py` L461–463**

```python
        self._record_stt_audio_usage(frame.audio)

        await self.process_generator(self.run_stt(frame.audio))
```

모든 audio frame — 대략 wire audio 20 ms마다 — 이 곧바로 `run_stt`에 넘겨지고, websocket service에게 그것은
"socket에 write하라"는 뜻입니다. transcription이 고객 자신의 발화와 겹칩니다. VAD가
`VADUserStoppedSpeakingFrame`을 발화시킬 즈음이면 provider는 이미 utterance의 대부분을 계산해 두었고
마무리만 하면 됩니다. 그것이 streaming의 latency 논증 전부이고, §11의 표는 각 provider가 그 마무리를
얼마나 잘 하는지에 대한 측정입니다.

### 7.3 segmented — buffer, trim, wrap, 한 번의 호출

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

trim window:

**`src/pipecat/services/stt_service.py` L836–837**

```python
        await super().setup(setup)
        self._audio_buffer_size_1s = self.sample_rate * 2
```

산술을 해 보십시오. §2와 직접 연결되기 때문입니다:

```
16 kHz:  16000 samples/s × 2 bytes/sample = 32000 bytes = 1.000 s of pre-roll
 8 kHz:   8000 samples/s × 2 bytes/sample = 16000 bytes = 1.000 s of pre-roll
```

두 rate 모두에서 1초입니다. 그리고 §2는 onset detection lag이 **192 ms**라고 말했습니다. 따라서 pre-roll은
VAD의 사각지대를 **5.2배**로 덮습니다. 이 buffer는 tuning knob이 아닙니다 — analyzer가 이미 상한을
지어 놓은 lag에 대한 고정된, 넉넉한 과잉 확보입니다. 이것이 realtime_voice의 `vad_prefix_frames = 5`
deque([[ch-03/read]] §4.5)에 대한 Pipecat 쪽 대응물이고, 차이는 거기서 이미 진술했습니다: 하나는 config
field, 다른 하나는 STT 안의 상수입니다.

그리고 VAD stop에서, 정확히 한 번의 호출:

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

**`SegmentedSTTService`는 `VADUserStoppedSpeakingFrame`으로 trigger됩니다.** §4를 염두에 두고 다시
읽으십시오: aggregator params에 `vad_analyzer`가 없으면 `VADUserStoppedSpeakingFrame`이 없고, 그것은
segmented STT service가 **아무것도 전혀 transcribe하지 않는다**는 뜻입니다. docstring은
*"Requires VAD to be enabled in the pipeline to function properly"* (L803)라고 말하고, 문자 그대로의
뜻입니다. 이것이 이 chapter의 두 번째 silent-failure mode이고, 첫 번째와 root cause가 같습니다.

마지막으로, segmented service는 구성상 모든 transcript를 final로 도장 찍습니다:

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

`finalized = True`가 §18의 STT safety-net timer를 short-circuit시키는 것입니다. 그래서 segmented service는
strategy chain 안에서 우아하게 저하됩니다: transcript를 만들어 내는 것은 느리지만, 만들어 내는 순간
chain은 기다리기를 멈춥니다. 비용은 전적으로 round trip에 떨어지지 timer에 떨어지지 않습니다.

---

## 8. base class는 interim을 절대 내보내지 않는다

작지만 하중을 받는 사실입니다:

```bash
$ grep -c "InterimTranscriptionFrame" src/pipecat/services/stt_service.py
0
```

**0개 참조.** 1,040줄짜리 base class는 `InterimTranscriptionFrame`을 import하지도, 생성하지도, 매치하지도
않습니다 — 단 한 번도. interim은 전적으로 provider의 일입니다:

```bash
$ grep -rl "InterimTranscriptionFrame(" $(find src/pipecat/services -name '*.py') | wc -l
25
```

25개 module이 그것을 직접 생성합니다. 그중 셋은 realtime LLM service(`inworld/realtime/llm.py`,
`openai/realtime/llm.py`, `xai/realtime/llm.py`)이고, 나머지 22개는 STT module입니다. base class가 *하는*
일은 final을 가로채는 것입니다:

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

`push_frame`은 L519에 있습니다. 그 비대칭 — base class가 final을 소유하고 provider가 interim을 소유한다 —
이 `TranscriptionFrame`에는 `finalized: bool` field가 있고 `InterimTranscriptionFrame`에는 없는 이유입니다
([[ch-03/read]] §5.1이 `frames.py` L450과 L476에서 둘 다 인용했습니다). 한 type은 "아직 안 끝났을 수도
있음"을 뜻하고, 다른 type은 끝났는지 여부를 말하는 명시적인 flag를 나릅니다.

---

## 9. TTFB, 재정의되다

Pipecat은 STT latency를 LLM latency를 재는 방식으로 재지 않습니다. docstring이 class body 안에서 그
재정의를 진술합니다:

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

논증은 빈틈이 없고 내면화할 가치가 있습니다: streaming STT에는 request가 없으므로 "response의 첫 byte까지
걸린 시간"은 존재하지 않는 것입니다. *의미가 있는* 유일한 구간 — 그리고 voice 제품이 신경 쓰는 유일한
구간 — 은 **speech end → final transcript**입니다.

그리고 여기가 §3이 준비해 둔 뺄셈입니다:

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

`def _handle_vad_user_stopped_speaking`은 **L627**에 있고, 뺄셈은 **L645**에 있습니다.

```
speech_end_time = frame.timestamp - frame.stop_secs
```

저 줄은 이제서야 의미를 갖습니다. `frame.timestamp`는 VAD가 *결정한* 시각입니다. `frame.stop_secs`는
그전에 관찰해야 했던 silence window입니다. 둘의 차이가 고객이 실제로 말을 멈춘 시각입니다. §2 없이는
그것을 임의의 값으로 읽었을 것입니다. §2가 있으면 숫자를 붙일 수 있습니다: default에서 그것은 설정된
silence 0.2 s만큼 시계를 되감고, analyzer는 그것을 실제로 여섯 chunk의 192 ms로 소비했습니다. metric이
detector의 반응 시간이 아니라 물리적 현실에 anchor됩니다 — 그것이 cross-provider latency 표가 의미를
갖는 유일한 방법입니다. 그렇지 않으면 모든 provider의 숫자에 당신의 VAD config가 섞여 들어갈 테니까요.

> 💡 **쉬운 설명 — anchoring이 왜 표를 의미 있게 만드나요?**
> 시계를 되감지 않는다면 측정 구간은 "VAD가 알아챈 시각 → final transcript"가 됩니다. 여기에는 당신이
> `stop_secs`를 어떻게 잡았는지가 그대로 섞여 들어갑니다. `stop_secs=0.5`로 놓은 사람의 표와
> `0.2`로 놓은 사람의 표가 같은 provider에 대해 다른 숫자를 냅니다. 물리적 순간(고객이 입을 다문 시각)에
> 고정시키면 그 숫자는 오직 provider의 속성이 됩니다. 그래서 §11의 표가 "모두 `stop_secs=0.2`에서
> 측정됨"이라고 굳이 밝히는 것이고, §15.2에서 `stop_secs`를 옮기면 framework가 경고하는 것입니다.

2초짜리 backstop:

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

`stt_ttfb_timeout=2.0`입니다. `else`에 주목하십시오: transcript가 아예 도착하지 않으면 측정은 열린 채로
남는 것이 아니라 *취소*됩니다. 유실된 utterance 하나가 다음 utterance의 숫자를 오염시키지 않습니다.

---

## 10. `STTMetadataFrame` — service가 자기 latency를 스스로 공표한다

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

이름 붙일 가치가 있는 branch가 둘입니다.

`supports_ttfs` (L548–557)는 기본으로 `True`를 반환하고, *server*가 turn 경계를 소유하는 service에 대해
`False`를 반환합니다 — 그런 경우 "speech end → final transcript"에는 측정할 독립된 존재가 없기
때문입니다. `False`일 때 frame은 `ttfs = 0.0`을 나르고, L554–555의 docstring이 downstream이 그것으로
무엇을 하는지 말해 줍니다: *"Downstream turn-stop strategies that consume `STTMetadataFrame` treat a 0
latency as 'no extra wait.'"* 0은 결측치가 아닙니다. 0은 지시입니다.

`ttfs is None` branch는 escape hatch입니다: 측정되지 않은 service는 `DEFAULT_TTFS_P99 = 1.0`으로
떨어지고 **warning을 log합니다**. 한국어 provider용 custom STT service를 쓰면서 constructor argument를
잊으면, 당신이 선택하지 않은 1초짜리 safety net과 아마 절대 읽지 않을 warning을 얻습니다. 지금
기록해 두십시오. §21이 그것을 결정으로 만듭니다.

그 docstring의 마지막 문장도 읽으십시오 — *"A service that does its own server-side end-of-turn
detection overrides this (calling `super()`) to set `user_turn_strategies` on the returned frame"* —
그것이 §21이 아홉 개의 instance를 세는 mechanism이기 때문입니다.

---

## 11. benchmark 표 — 여기서 한 번만 만들어진다

`src/pipecat/services/stt_latency.py`는 **68줄**이고 repository에서 유일한 hard benchmark 표입니다.
**provider 선택은 이 chapter에서 일어나고 이 course의 다른 어디에서도 일어나지 않습니다.**
[[ch-11/read]]의 latency waterfall은 이 표에서 선택된 값 하나를 소비합니다. 표를 다시 그리지 않습니다.

module docstring이 측정 조건을 진술하며, 그것이 이 section이 §2 뒤에 와야만 했던 이유입니다:

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

**아래 표의 모든 상수는 `VADParams.stop_secs = 0.2`에서 측정되었습니다** — §2가 여섯 chunk와 192 ms를
유도해 낸 그 값입니다. 그것은 각주가 아닙니다. `stop_secs`를 바꿀 때 §18과 §19가 둘 다 warning을 내보내는
이유이고, VAD를 가르치기 전에 이 표를 가르치는 chapter는 분모 없는 숫자를 가르치는 셈이 되는 이유입니다.

### 11.1 완전한 표 — 측정된 상수 23개 전부

`stt_latency.py`는 **측정된 provider 상수 23개**를 담고 있습니다. 보통 인용되는 열두어 개가 아닙니다.
여기 전부가, source에서 그대로, latency 순으로:

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

정렬한 것, §12가 채울 한국어 열과 함께:

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

여기서 읽어 낼 것이 셋입니다.

**폭이 6.1배입니다.** 0.35 s에서 2.14 s. 그것은 tuning 차이가 아니라 다른 제품입니다. P99에서 `XAI`를
쓰는 bot과 대화하는 고객은 `SONIOX`를 쓰는 bot의 고객보다 turn이 닫히기까지 거의 2초를 더 기다립니다.
LLM이 토큰 하나 만들어 내기도 전에 말입니다.

**`NVIDIA`와 `WHISPER`는 측정이 아니라 alias입니다.** 문자 그대로 `= DEFAULT_TTFS_P99`입니다 — §10의
`ttfs is None` branch가 떨어지는 그 보수적인 1.0과 같은 값입니다. 주석이 그렇게 말합니다:
*"These services run locally and should be replaced with measured values."* 1.0을 Whisper benchmark로
절대 인용하지 마십시오. 그것은 placeholder입니다.

**`SARVAM_REALTIME`은 자기 자신의 in-source 면책 조항을 달고 있습니다** — `# Provisional until benchmarked
against the realtime endpoint.` repo가 자기 숫자 중 어느 것을 자기도 믿지 않는지 말해 주고 있습니다.

**빠져 있는 행이 있는 행만큼이나 많은 정보를 줍니다.** `CartesiaTurnsSTTService`와
`DeepgramFluxSTTService`는 여기에 상수가 아예 없고, 그것은 설계입니다: module docstring에 따르면 그들은
`supports_ttfs`를 `False`로 override합니다. server가 turn 경계를 정의하기 때문입니다. §21이 그것이 무엇을
치르고 무엇을 사는지 설명합니다.

→ **[`figures/turn-boundary.html`](figures/turn-boundary.html)의 panel two가 이 표를 interactive provider
selector로 렌더링합니다** — 0.35에서 2.14까지 정렬된 bar chart이고, 두 개의 local alias는 측정이 아니라
alias로 그려집니다. 이 control이 만들어지는 곳은 course에서 여기뿐입니다. 거기서 provider를 선택하고
그 숫자를 §18의 `effective_stt_wait` 산술로 가져가십시오.

---

## 12. 한국어, 추측 대신 열린 미지수를 진술하기

`src/pipecat/transcriptions/language.py` L310–311이 두 개의 enum member를 정의합니다:

```python
    KO = "ko"
    KO_KR = "ko-KR"
```

"Pipecat이 한국어 STT를 지원하나?"에는 service에 따라 **세 가지 서로 다른 종류의 답**이 있고, 그중 하나만
evidence입니다 ([[stt-korean-providers]]):

1. **Verified** — `Language.KO`가 그 service 자신의 손으로 큐레이션한 map에 key로 나타납니다.
2. **Passthrough** — map이 아예 없고, enum이 담고 있는 문자열이 그대로 wire로 나가며, provider가 그것을
   받아들이는지에 대해 repo는 아무 입장도 취하지 않습니다.
3. **Documented exclusion** — map이 있고 한국어가 의도적으로 그 안에 없습니다.

### 12.1 검증된 12개 service

ground truth. STT module로 제한한 `Language.KO` grep 결과:

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

열둘입니다. `ElevenLabsSTTService`가 `"ko"`가 아니라 세 글자 `"kor"`를 보낸다는 것은 기억해 둘 가치가
있는 진짜 quirk입니다 — 언젠가 config 기반 service selector를 쓴다면, 순진한 ISO-639-1 가정이 정확히
이 행에서 깨집니다.

Moonshine의 docstring은 이 tree에서 가장 강한 산문상의 지원 진술입니다:

**`src/pipecat/services/moonshine/stt.py` L122–124**

```
        language: Language for transcription. Moonshine publishes models for
            Arabic, Chinese, English, Japanese, Korean, Spanish, Ukrainian, and
            Vietnamese; regional variants resolve to their base code.
```

### 12.2 FunASR — 다른 mechanism에 의한 열세 번째

FunASR은 한국어가 가능하고 **`Language.KO` mapping이 전혀 없습니다.** 평범한 문자열 집합을 씁니다:

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

집합은 **L42**에 있습니다. 그 바깥의 무엇이든 `"auto"`로 강제됩니다. `Language.KO`의 값은 `"ko"`이므로
`str(language.value).split("-")[0].lower()`는 `"ko"`를 내고, 그것은 집합 *안에* 있습니다 — 한국어는
동작하지만, 큐레이션된 mapping이 아니라 문자열 membership에 의해서입니다. `FunASRSTTService`는
`SegmentedSTTService`(`funasr/stt.py:87`)이고 TTFS 상수가 없습니다.

> **SOURCE CORRECTION #3.** [[stt-korean-providers]]의 "Verified Korean" 표는 FunASR을 `funasr/stt.py:41`,
> code `"ko"`로 열세 번째 행에 올려놓습니다 — `Language.KO` key를 가진 열두 service와 같은 표에 말입니다.
> source는 이에 동의하지 않습니다: `funasr/stt.py`에는 **`Language.KO` mapping이 없고** L42의
> `_FUNASR_LANGUAGES`만 있습니다. 이 구별은 현학적인 게 아닙니다 — 큐레이션된 map은 maintainer가
> provider가 그 code를 받아들인다고 단언하는 것이고, 문자열 집합 membership 검사는 ISO code가 우연히
> 맞아떨어진 것입니다. **12 verified + 1 by a different mechanism**으로 세고, shortlist를 만들 때
> FunASR의 evidence class를 그 자체로 취급하십시오.

### 12.3 Passthrough — map이 없으므로 in-repo 검증도 없다

**Deepgram**이 중요한 경우입니다. 표 전체에서 공동 1위의 측정 latency를 가졌는데 repo는 한국어에 대해
아무 입장도 취하지 않기 때문입니다:

```bash
$ grep -n "LANGUAGE_MAP\|language_to_service_language" src/pipecat/services/deepgram/stt.py
$
```

map도 없고 override도 없습니다. 들고 있는 것을 그대로 직렬화합니다:

**`src/pipecat/services/deepgram/stt.py` L579–582**

```python
        if is_given(s.model) and s.model is not None:
            kwargs["model"] = str(s.model)
        if is_given(s.language) and s.language is not None:
            kwargs["language"] = str(s.language)
```

`Language.KO`는 `"ko"`로 나갈 것입니다. Deepgram이 그것을 받아들이는지는 이 repository가 답하지 않는
질문입니다. **0.35 s이면서 한국어 지원은 미지**인 것이 그 행의 모양이고, 그것은 추론이 아니라 benchmark로
닫는 것입니다.

`OpenAISTTService`와 `OpenAIRealtimeSTTService`도 같은 모양입니다 — language가 그대로 통과하고, 한국어는
Whisper 자신의 language set에 얹혀 가며, in-repo에서 확인해 주는 것은 없습니다.
`DeepgramFluxSTTService`는 `model="flux-general-multi"`와 `language_hints`만 문서화하고
`supports_ttfs → False`입니다.

### 12.4 Documented exclusion — shortlist 만들기 전에 읽으십시오

**`AssemblyAISTTService`에는 한국어가 없습니다.** map은 exhaustive하고 한국어는 그 안에 없습니다:

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

열여덟 개 언어. 일본어와 중국어는 있고 한국어는 없습니다. `use_base_code=True`는 `Language.KO`가
**warning과 함께** `"ko"`로 떨어진다는 뜻입니다 — 지원이 아니라 fallback입니다. AssemblyAI의 0.42 s는
표에서 네 번째로 좋은 숫자이고, 이 repo의 evidence상 당신에게는 사용 불가능합니다.

**`CartesiaTurnsSTTService`는 대놓고 배제됩니다:**

**`src/pipecat/services/cartesia/turns/stt.py` L157–158**

```python
        # ink-2 is English-only at launch; language on emitted frames is fixed.
        self._language = Language.EN
```

**`SarvamSTTService`의 유일한 `KO*` key는 Konkani입니다.** outline이 경고하는 오독이 이것입니다:

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

`KOK_IN` → `"kok-IN"`은 **Konkani**, 고아(Goa) 지역의 인도아리아어입니다. Sarvam은 인도 언어 service입니다.
그 파일에서 `KO`를 grep하면 이것이 걸리고, 부주의한 독자는 Sarvam을 한국어 가능으로 분류합니다.

**그리고 excerpt들이 기록하지 않은, 제가 찾은 exclusion 하나.** §11의 0.41 s 행은
`ELEVENLABS_REALTIME_TTFS_P99`이고, 그것은 `ElevenLabsRealtimeSTTService`(`elevenlabs/stt.py:452`)의
것입니다 — **`Language.KO: "kor"` mapping을 나르는 `ElevenLabsSTTService`와는 다른 class입니다.**
`language_to_service_language`는 그 파일에서 딱 한 번, L322에, segmented class 안에 정의되어 있습니다.
realtime class는 raw `language_code: str | None = None`(L494)을 받고 map을 절대 참조하지 않습니다. 따라서:

| Class | Line | Base | P99 | Korean evidence |
|---|---|---|---|---|
| `ElevenLabsSTTService` | 210 | `SegmentedSTTService` | 2.01 | **verified** — `Language.KO: "kor"` |
| `ElevenLabsRealtimeSTTService` | 452 | `WebsocketSTTService` | 0.41 | passthrough — raw `language_code` string |

빠른 ElevenLabs와 한국어가 검증된 ElevenLabs는 같은 service가 아닙니다. excerpt의 표와 latency 표를
파일을 열어 보지 않고 함께 읽었다면, ElevenLabs가 0.41 s streaming에서 검증된 한국어를 제공한다고
결론 내렸을 것입니다. 그렇지 않습니다.

### 12.5 열린 미지수, 추측 대신 진술하기

grep 두 개, 둘 다 아무것도 반환하지 않습니다:

```bash
$ grep -rniE "\bWER\b|word error rate" $(find src/pipecat -name '*.py')
$
```

**이 repository 어디에도, 어떤 service에 대해서도, 어떤 언어에 대해서도, 어떤 종류의 정확도 숫자도
없습니다.** `stt_latency.py`는 *latency만* 기록하고, benchmark audio의 언어와 sample rate에 대해서는
침묵하며, 값들이 `stop_secs=0.2`에서 측정되었다는 것만 진술합니다.

**그리고 어떤 STT service에 대해서도 8 kHz telephony 숫자가 없습니다.** tree에 있는 유일한 `8000` 값들은
[[ch-05/read]]이 열거한 telephony serializer default들입니다. 어떤 STT module도 그 rate에서의 자기 동작을
문서화하지 않습니다.

이 두 부재를 [[transport-telephony]]의 acoustics 옆에 놓으면 gap의 크기가 구체적이 됩니다. Lina TMR의
audio는 8 kHz μ-law입니다: 4 kHz Nyquist 천장, 8-bit companding. 한국어 마찰음과 치찰음(ㅅ/ㅆ/ㅊ), 그리고
받침 구별에 필요한 acoustic cue의 상당 부분이 4 kHz 이상에 있고 **신호에 들어 있지 않습니다.** STT 앞에서
8 k → 16 k로 upsampling하는 것은 model의 입력 contract를 만족시킬 뿐, 대역을 복원하지 않습니다. 따라서:

> **§11 표의 모든 숫자는 영어를 가정한, sample rate가 명시되지 않은 latency 측정치이고, §12.1의 한국어
> shortlist 전체가 현재 그 숫자로 정렬되어 있습니다.** 정확도 evidence는 존재하지 않습니다. provider를
> 확정하기 전의 blocking item은 실제 Lina TMR μ-law 8 kHz 한국어 audio에 대한 당신 자신의 benchmark이며,
> <https://github.com/pipecat-ai/stt-benchmark>를 사용하십시오.

그것이 evidence의 정직한 상태입니다. "0.35"의 정밀함에 속아서 shortlist가 결정되었다고 생각하지 마십시오.

---

# PART THREE — THE ARBITER: THE TURN-STRATEGY CHAIN

## 13. 실제로 거기 있는 것

§0은 부재를 확립했습니다. 여기 존재가 있습니다: **`src/pipecat/turns/` — 4,429줄.**

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

boolean 하나에 답하려고 4천 5백 줄입니다. 그 비율이 곧 design입니다: 질문이 진짜로 어렵고, 증거가
서로 다른 세 subsystem에서 오며, Pipecat의 대응은 heuristic 하나를 고르는 대신 모든 조각을 교체 가능하게
만드는 것입니다.

---

## 14. default들, 그리고 그것이 당신을 무엇에 묶어 두는가

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

**기본 start list는 정확히 이 chapter의 각 부에 하나씩입니다.** `VADUserTurnStartStrategy`가 1부의 답이고,
`TranscriptionUserTurnStartStrategy`가 2부의 답입니다. 그것들은 chain 순서에 의해 OR로 묶입니다:
먼저 발화하는 쪽이 turn을 주장하고 `STOP`을 반환해서 loop를 끊습니다.

transcript 쪽은 VAD의 failure mode를 덮으려고 존재합니다:

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

`use_interim=True`가 default입니다 — *interim* transcript만으로도 발언권을 주장하기에 충분합니다. §5의
liability 분석을 기억하십시오: energy가 `min_volume = 0.6`을 절대 넘지 못하는 조용한 화자가 정확히 이것이
커버하는 대상입니다. 그리고 따름정리에 주목하십시오: **boson의 현재 text-triggered barge-in은 지원되는
Pipecat 구성입니다** — Pipecat이 대체하는 무언가가 아닙니다. §19가 그것을 씁니다.

### 14.1 기본 stop strategy는 machine-learning model이다

사람들이 놓치는 사실이 이것입니다:

```python
    return [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
```

**아무것도 건드리지 않으면 Pipecat은 모든 VAD stop마다 audio 위에서 ONNX end-of-turn classifier를
돌립니다.** 그것을 켜는 configuration flag가 없습니다. 아무것도 구성하지 않으면 얻게 되는 것입니다.

**`src/pipecat/audio/turn/smart_turn/local_smart_turn_v3.py`** — 구체적인 사실들:

- L16–17: `import onnxruntime as ort`, `import soxr`
- L25: `_MODEL_SAMPLE_RATE = 16000`
- L28: `class LocalSmartTurnAnalyzerV3(BaseSmartTurn)`
- L35: `__init__(self, *, smart_turn_model_path: str | None = None, cpu_count: int = 1, **kwargs)`
- L50: `model_name = "smart-turn-v3.2-cpu.onnx"` — package에 번들됨
- L73: `so.intra_op_num_threads = cpu_count` — 기본 한 thread, §1.1과 같은 규율
- L136: `soxr.resample(audio_array, actual_rate, _MODEL_SAMPLE_RATE, quality="HQ")`
- L164: `log_mel = compute_whisper_log_mel_features(audio_array, do_normalize=True)`

**audio 전용입니다.** Whisper log-mel feature가 들어가고 complete/incomplete가 나옵니다. transcript text를
절대 읽지 않습니다. [[endpointing-turn-boundary]]가 결론을 도출하고, 한국어 제품에 대해 그것은 옳은
결론입니다: *"smart-turn-v3 is acoustic, so it carries no Korean-language risk but gets no help from
Korean sentence-final endings."*

그 두 절반을 모두 곱씹으십시오. **위험 없음:** 영어로 훈련된 *text* model은 언어별 segmentation 동작을
가집니다. acoustic model은 prosody를 읽고, prosody는 문자 체계나 어휘에 묶여 있지 않습니다.
**도움도 없음:** 한국어는 발화 완결을 형태론에서 명시적으로 표시합니다 — `-습니다`, `-어요`, `-거든요`,
`-는데…` — 그리고 끝을 흐리는 `-는데`는 *통사적으로 명시적인* "아직 안 끝났음"입니다. acoustic model은
pitch contour와 꼬리 energy만 봅니다. 이 언어에서 가장 신뢰할 만한 단 하나의 end-of-turn cue가 기본
stop strategy에게는 보이지 않습니다. 그것은 구체적이고 만들어 낼 수 있는 gap이며 §20이 그것을 probe로
만듭니다.

> 💡 **쉬운 설명 — "위험 없음"과 "도움 없음"을 예시로**
> 고객: "그게 좀 비싸긴 한데…" — 여기서 `-는데`는 "말이 아직 안 끝났다"는 문법적 신호입니다.
> 하지만 acoustic model이 보는 것은 끝에서 떨어지는 pitch와 줄어드는 energy뿐이고, 그것은
> 완결된 문장의 음향 패턴과 구별이 어렵습니다. 그래서 bot이 끼어듭니다.
> 반대로 이 model에는 "한국어라서 오작동한다"는 위험은 없습니다. 영어 텍스트로 훈련된 punctuation
> model이라면 한국어 어절 경계를 엉뚱하게 자르겠지만, mel spectrogram에는 언어별 편향이 그만큼
> 직접적으로 실리지 않습니다. 요약: **깨지지는 않지만, 공짜 정보 하나를 못 씁니다.**

L136의 resample도 함께 보십시오: model은 16 kHz에서 돌아가므로 Lina의 8 kHz telephony audio는 inference
전에 `soxr`로 upsampling됩니다 — [[transport-telephony]]의 "sample 수는 복원하지만 대역폭은 복원하지 않는다"는
단서가 STT에 적용되는 만큼 turn model에도 그대로 적용됩니다.

**VAD 전용 경로는 당신이 명시적으로 선택해야 하는 downgrade입니다.** 저렴한 CPU와 예측 가능한 timing을
원한다면 `SpeechTimeoutUserTurnStopStrategy`를 직접 넘기십시오. 아무것도 그것을 대신 골라 주지 않습니다.

### 14.2 `SmartTurnParams`의 비대칭

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

`SmartTurnParams.stop_secs = 3`이 `VADParams.stop_secs = 0.2` 옆에 있으면 모순처럼 보입니다. 아닙니다 —
그것들은 두 개의 서로 다른 시계이고, 그중 하나만 정상 경로입니다.

- `SmartTurnParams.stop_secs = 3`은 **analyzer 자신의 silence fallback**입니다: 외부 trigger 없이
  `append_audio` 안에서 3초의 silence가 쌓이면 analyzer가 스스로 `COMPLETE`를 선언합니다.
- `VADParams.stop_secs = 0.2`는 analyzer가 **자문받는** 시점입니다. `_handle_vad_user_stopped_speaking`이
  모든 VAD stop에서 — silence 192 ms마다 — `analyze_end_of_turn()`을 호출합니다.

**따라서 ML 판정이 정상 경로이지 예외가 아닙니다.** 3초 상수는 VAD 경로가 아예 발화하지 못했을 때에만
발화합니다. `stop_secs = 3`을 읽고 turn당 3초의 정적을 예산에 잡는 사람은 design을 거꾸로 이해한
것입니다.

`pre_speech_ms = 500`은 analyzer 자신의 pre-roll입니다 — VAD의 여섯 chunk lag과 segmented STT의 1초에
이어 세 번째입니다. 각 layer가 자기 위 layer에 대해 독립적으로 buffering합니다.

---

## 15. 두 개의 timing strategy, 그리고 공유되는 safety net

시간을 다루는 두 stop strategy 모두 §11의 표에서 같은 양을 계산합니다. 한 번에 배우십시오.

### 15.1 `SpeechTimeoutUserTurnStopStrategy` — VAD 전용 경로 (328 L)

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

두 개의 timer이고, docstring이 그 역할을 정확하게 이름 붙입니다: 하나는 **policy floor**, 하나는
**safety net**. 둘은 중복이 아닙니다.

- `user_speech_timeout = 0.6` — 고객이 다시 말을 이어갈 수 있는 유예 window입니다. 그것은 한국어 화자의
  문장 중간 pause가 얼마나 길어도 되는지에 대한 제품 결정이고, STT가 무엇을 하든 상관없이 항상 끝까지
  돕니다.
- `stt_timeout` — config가 아니라 frame에서 옵니다:

**`src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py` L159–161**

```python
        if isinstance(frame, STTMetadataFrame):
            self._stt_timeout = frame.ttfs_p99_latency
            self._stop_secs_warned = False
```

그것이 §10의 broadcast가 착지하는 지점입니다. service가 자기 P99를 공표하고, turn strategy가 그것으로
자기 timer의 크기를 정합니다. **이것이 2부와 3부의 접합부**이고, 한 줄입니다.

### 15.2 산술 — 공식 이전에 손으로

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

`effective_stt_wait = max(0.0, self._stt_timeout - self._stop_secs)`는 **L223**에 있습니다.

공식을 공식으로 읽기 전에 구체적으로 계산해 보십시오. §11 표에서 두 행을 골라 default `stop_secs = 0.2`와
함께 대입합니다:

```
Soniox        (0.35 s):  max(0, 0.35 - 0.2) = 0.15 s of safety net
Speechmatics  (0.74 s):  max(0, 0.74 - 0.2) = 0.54 s
Google        (1.57 s):  max(0, 1.57 - 0.2) = 1.37 s
xAI           (2.14 s):  max(0, 2.14 - 0.2) = 1.94 s
```

이제 그 뺄셈이 *무엇을 뜻하는지* 말하십시오. provider의 P99는 고객이 물리적으로 말을 멈춘 순간부터
측정됩니다(§9의 `speech_end_time` anchoring). 그런데 strategy는 그것을 `stop_secs`만큼 뒤에, VAD frame이
도착할 때에야 알게 됩니다. 그 0.2초는 이미 기다림에 쓰인 것입니다. 그것을 빼는 것이 같은 침묵에 대해
두 번 값을 치르지 않게 해 줍니다.

그리고 최악의 경우 — `max(0.0, ...)`가 막아 주는 것: Soniox에서 `stop_secs = 0.5`로 설정하면
`0.35 - 0.5`가 음수가 되어 safety net이 0으로 붕괴합니다. 코드가 정확히 그것을 경고합니다:

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

두 개의 warning이고, 첫 번째가 §2와 §11 위로 loop가 닫히는 지점입니다. `VAD_STOP_SECS`는
`vad_analyzer.py`에서 직접 import됩니다(이 파일의 L13). **`stop_secs`를 0.2에서 옮기면 framework가
runtime에, 23개짜리 표의 모든 상수가 방금 당신에게 무효가 되었다고 말해 줍니다.** 그것은 드물고 정직한
engineering입니다: 자기 default의 출처를 알고 있고, 당신이 그것을 무효화하면 항의하는 library.

### 15.3 short-circuit

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

**P99 timer는 당신이 거의 치르지 않는 천장입니다.** 정상적인 경우 transcript는 P99보다 한참 전에
finalize되고 timer는 취소됩니다. `effective_stt_wait`를 치르는 것은 1%의 꼬리에서, 또는 `finalized`를
절대 설정하지 않는 provider에서뿐입니다. 이것이 2.14 s의 `xAI`를 고른다고 해서 모든 turn에 1.94 s를
더한다는 뜻이 아닌 이유입니다 — *최악의* turn들이 그만큼 더 나빠진다는 뜻입니다.

그리고 미묘한 anti-staleness 규칙:

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

interim이 앞선 finalize를 *철회*합니다. 주석의 시나리오를 읽으십시오: STT 자신의 endpointer가 VAD의
192 ms보다 짧은 100 ms pause에서 finalize하므로, VAD는 stop을 보고한 적이 없고 flag를 지운 적도 없습니다.
그러고 나서 고객이 계속 말합니다. 이 줄이 없으면 strategy는 발화의 꼬리가 아직 날아오는 중인데 다음 VAD
stop에서 safety net을 건너뛰고, 고객의 말을 문장 중간에 잘라 버릴 것입니다. **이것은 서로 다른 threshold를
가진 두 개의 독립적인 endpointer가 동시에 돌고 있기 때문에만 존재하는 bug class**이고, server-side
endpointing을 끌 수 없는 provider를 고를 때 기억할 가치가 있습니다.

> 💡 **쉬운 설명 — 두 endpointer가 왜 충돌하나요?**
> Pipecat의 VAD는 "192 ms 침묵 = 멈춤"이라고 봅니다. Deepgram 같은 provider는 서버에서 자기만의
> endpointing을 돌리며 100 ms 침묵에서 final transcript를 뱉을 수 있습니다. 고객이
> "제가 생각을 좀 [100 ms] 해봤는데요" 라고 말하면: STT는 "제가 생각을 좀"을 final로 표시하고,
> VAD는 아무 일도 없었다고 봅니다(192 ms에 못 미치니까). `_transcript_finalized`는 True로 남습니다.
> 문장이 진짜 끝난 뒤 VAD stop이 오면, strategy는 "이미 final 받았지"라며 safety net을 건너뛰고
> "제가 생각을 좀"만 들고 LLM으로 넘어갑니다. interim 하나가 flag를 다시 False로 되돌리는 것이
> 그 시나리오 전체를 막습니다.

---

## 16. `TurnAnalyzerUserTurnStopStrategy` — 기본 경로 (364 L)

같은 two-clock 구조이고, deadline을 계산하는 방식에서 결정적인 차이가 하나 있습니다.

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

`stt_deadline`은 **L236**, `timeout`은 **L245**입니다. §15.2의 상대적인 `effective_stt_wait`와 비교하면
차이가 주석의 요점입니다:

```
SpeechTimeout:   effective_stt_wait = max(0, stt_timeout - stop_secs)      # relative duration
TurnAnalyzer:    stt_deadline       = timestamp - stop_secs + stt_timeout  # absolute wall-clock
                 timeout            = max(0, stt_deadline - time.time())   # remaining, computed AFTER inference
```

세 statement의 순서가 하중을 받습니다. `stt_deadline`은 `await self._turn_analyzer.analyze_end_of_turn()`
**이전에** 계산되고, `timeout`은 **이후에** 계산됩니다. ONNX inference는 시간이 걸립니다. 만약 timer가
inference 이후에 상대적 duration으로 무장된다면, 그 inference 시간이 STT 예산에서 *차감*되는 대신
*추가*될 것입니다. 절대 deadline에 anchor하면 safety net은 model latency가 갉아먹는 고정된 wall-clock
window가 되지, 늘어나는 window가 되지 않습니다.

그것은 진짜로 좋은 engineering이고, 부하 상황에서 P50 latency 목표가 유지되는지를 결정하는 딱 그런
종류의 디테일입니다: 동시 통화 40건에서 ONNX inference는 CPU를 두고 경쟁하고, 이 design은 그 경쟁이 turn
예산을 통해 곱해지지 않는다는 것을 보장합니다.

두 analyzer shape 모두 같은 strategy가 처리합니다:

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

`append_audio`는 **모든** `InputAudioRawFrame`에 대해 호출되고, `self._vad_user_speaking`이 함께 넘어간다는
점에 주목하십시오 — analyzer는 자기가 buffering하는 audio가 speech인지 알아야 합니다. 이 줄이 STT에
`audio_passthrough=True`를 요구하는 줄입니다(§7.1): strategy는 pipeline에서 STT service의 downstream에
앉아 있고, STT가 전달해 주었기 때문에만 audio를 봅니다.

`EndOfTurnState`는 `COMPLETE = 1` / `INCOMPLETE = 2` — `audio/turn/base_turn_analyzer.py`에서 오는 두 값짜리
판정이고, 그 ABC는 다섯 개 method입니다: `speech_triggered`, `params`, `append_audio`,
`analyze_end_of_turn`, `clear`. custom analyzer를 쓴다는 것은 다섯 method를 구현한다는 뜻입니다. §20이
그것을 probe로 만듭니다.

---

## 17. controller: 순서, veto, 그리고 watchdog

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

start chain 먼저, 그다음 stop chain, 둘 다 list 순서로, 둘 다 `STOP`에서 break. 그것이 dispatch 전부입니다.
`ProcessFrameResult`는 값이 둘입니다(`turns/types.py`, 24줄). `STOP`은 "이 frame은 내가 처리했다. 내 뒤의
strategy들이 이것을 보게 하지 마라"는 뜻입니다.

**list에서의 순서가 policy입니다.** `WakePhraseUserTurnStartStrategy`를 `VADUserTurnStartStrategy` 앞에
놓으면 wake phrase가 모든 frame에 대해 우선 거절권을 갖습니다. 뒤에 놓으면 phrase matcher가 돌기도 전에
VAD가 turn을 주장합니다. priority field는 없습니다 — list index가 *곧* priority입니다.

veto:

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

guard가 둘입니다. **L358**은 이미 닫힌 turn에 대한 중복 stop을 버립니다. **L367**이 이름 붙일 가치가 있는
것입니다: *어떤 strategy도 사용자가 들리게 말하고 있는 동안에는 turn을 끝낼 수 없습니다.* 무엇을
결론지었든 상관없습니다. stop 판정은 언제나 최소한 약간은 오래된 증거로부터 계산됩니다 — LLM completion
check는 수백 밀리초가 걸리고 ONNX inference는 수십 밀리초가 걸립니다 — 그리고 그 사이에 고객이 다시
말하기 시작했을 수 있습니다. controller는 finalize하는 그 순간에 살아 있는 물리적 state를 다시 확인하고
거부합니다.

그리고 backstop. veto하는 strategy들의 chain은 deadlock할 수 있기 때문입니다:

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

`LLMUserAggregatorParams`에서 오는 `user_turn_stop_timeout: float = 5.0`입니다. 마지막 활동 이후 5초가
지나고 turn이 여전히 열려 있고 고객이 조용하면, `strategy=None`으로 강제 종료합니다 — 누구의 판정도
아닌, framework의 판정입니다. **sales call에서 5초는 아주 긴 시간입니다.** §21이 그것을 유지할지
결정합니다.

downstream에서 실제로 공표하는 것은 `UserTurnProcessor`입니다:

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

**L210**의 `broadcast_interruption()`이 turn 결정이 barge-in이 되는 지점입니다. [[ch-08/read]]이 그 호출의
downstream 전부를 소유합니다. 이 chapter의 일은 그것을 만들어 내는 데서 끝납니다.

double-decision bug 부류를 막아 주기 때문에 mechanism 하나만 더:

**`src/pipecat/turns/user_turn_processor.py` L162–170**

```python
        elif isinstance(frame, ProposedUserStartedSpeakingFrame):
            # A proposal is resolved once. Forwarding one our own strategies
            # resolve would let a resolver further down the pipeline decide the
            # same turn a second time.
            if not self._user_turn_controller.resolves_proposed_turn_start_frames:
                await self.push_frame(frame, direction)
```

로컬 strategy가 해소한 `ProposedUserStartedSpeakingFrame`은 forward되지 않고 **소비됩니다**. 그것이
§21의 external-endpointing service들이 의존하는 mechanism입니다.

---

## 18. 아홉 개의 assignment site, 여덟 개 파일, 두 개는 무조건

어떤 provider들은 turn 결정을 chain에서 통째로 빼앗아 갑니다. mechanism은 §10 docstring의 마지막
문장입니다: service가 `service_metadata_frame()`을 override하고, 반환하는 frame에 `user_turn_strategies`를
설정합니다.

**직접 세어 보십시오:**

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

**여덟 개 파일에 걸친 아홉 개 site.** `sarvam/stt.py`에 두 개가 있습니다 — service class 하나당 하나씩,
`SarvamSTTService`가 L173, `SarvamRealtimeSTTService`가 L891. 그리고 **아홉 중 둘만이 무조건입니다.**

### 18.1 무조건인 두 site

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

guard가 없습니다. 둘 중 하나를 고르는 것이 *곧* external endpointing을 고르는 것입니다. 둘 다
`supports_ttfs → False`를 반환하기도 합니다(§10). 그것이 둘 다 §11의 표에 나타나지 않는 이유입니다:
server가 경계를 소유하므로 측정할 "speech end → final transcript" 구간이 없습니다.

### 18.2 flag로 gate된 일곱 site, 정확한 guard와 default와 함께

나머지 일곱 전부가 `if` 아래에 앉아 있습니다. 각 guard가 그대로 있고, flag의 default는 constructor에서
해소한 값입니다:

| Service | Site | Guard | Flag default | Fires by default? |
|---|---|---|---|---|
| `SpeechmaticsSTTService` | `speechmatics/stt.py:561` | `if is_given(mode) and mode != TurnDetectionMode.EXTERNAL` | `turn_detection_mode = TurnDetectionMode.EXTERNAL` (L308) | **no** |
| `GladiaSTTService` | `gladia/stt.py:370` | `if self._settings.enable_vad` | `enable_vad=False` (L279) | **no** |
| `SonioxSTTService` | `soniox/stt.py:422` | `if not self._vad_force_turn_endpoint` | `vad_force_turn_endpoint: bool = True` (L287) | **no** |
| `AssemblyAISTTService` | `assemblyai/stt.py:669` | `if not self._vad_force_turn_endpoint` | `vad_force_turn_endpoint: bool = True` (L319) | **no** |
| `SarvamSTTService` | `sarvam/stt.py:395` | `if self._settings.vad_signals` | `vad_signals=None` (L269) | **no** |
| `SarvamRealtimeSTTService` | `sarvam/stt.py:1034` | `if self._endpointing == "vad"` | `endpointing: Literal["vad", "manual"] = "vad"` (L922) | **YES** |
| `OpenAISTTService` | `openai/stt.py:414` | `if self._server_vad_enabled` | `turn_detection: dict \| Literal[False] \| None = False` (L248) → `_server_vad_enabled = turn_detection is not False` (L367) | **no** |

저 행들 중 넷에 함정이 있습니다.

**Speechmatics의 `EXTERNAL`은 보이는 것과 반대를 뜻합니다.** enum docstring을 읽으십시오:

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

`EXTERNAL`은 **Speechmatics의** 관점에서 붙여진 이름입니다 — "endpointer가 나에게 외부에 있다, 즉
Pipecat의 것이다." Pipecat의 관점에서 그것은 *내부*를 뜻합니다. 그래서 guard
`mode != TurnDetectionMode.EXTERNAL`은 그 사실을 알기 전까지 거꾸로 읽히고, default `EXTERNAL`은
Speechmatics가 turn 결정을 가져가지 **않는다**는 뜻입니다. 네 개 mode 중 셋이 그것을 넘겨줍니다.

**Soniox와 AssemblyAI는 동일한 flag로 gate됩니다.** 둘 다 `vad_force_turn_endpoint: bool = True`이고,
둘 다 `if not self._vad_force_turn_endpoint`로 guard되며, 두 docstring이 같은 표현으로 그렇게 말합니다:

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

같은 flag, 같은 default, 같은 동작. **둘을 다르게 분류하지 마십시오.** 하나를 다른 group에 넣는 표는
전부 틀린 것입니다. (AssemblyAI는 다른 쪽에 없는 제약을 하나 더 붙입니다: `assemblyai/stt.py` L436–440은
flag를 `False`로 설정할 때 model이 `u3-pro`가 아니면 raise합니다.)

**`SarvamRealtimeSTTService`는 flag로 gate되어 있으면서 기본 상태에서 무조건처럼 동작하는 유일한
site입니다.** `endpointing`의 default가 `"vad"`이므로, 명시적으로 `"manual"`을 넘기지 않는 한 guard가
통과합니다. "기본값으로 turn 결정을 가져가는 service" 수를 센다면, 답은 둘이 아니라 **셋**입니다.

**OpenAI의 guard에는 off-by-one-value 위험이 있습니다.** `_server_vad_enabled = turn_detection is not False`
— 따라서 `turn_detection=False`(default)는 그것을 끄지만, `turn_detection=None`은 **켭니다**.
`None is not False`이기 때문입니다. "설정 안 함"의 의미로 `turn_detection=None`을 쓰는 호출자는 server VAD를
얻습니다. 없는 field가 `None`으로 역직렬화되는 config 파일에서 이것을 배선한다면 알아 둘 가치가 있습니다.

### 18.3 재현할 수 있어야 하는 요약

> **9 assignment sites · 8 service files · 2 unconditional · 7 provider-flag-gated · 3 firing by
> default.**

그리고 payload는 언제나 같은 container입니다:

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

`__post_init__`이 두 list를 무조건 **교체합니다**. 로컬 VAD와 smart-turn strategy는 보강되는 것이 아니라
사라집니다. 그것이 §19가 신경 쓰는 mechanism입니다.

### 18.4 별개의, 다른 mechanism — 아홉에 속하지 않음

세 개의 realtime **LLM** service가 같은 container를 metadata frame에 assign하는 대신 **constructor
kwarg**로 넘깁니다:

```bash
$ grep -rn "user_turn_strategies=ExternalUserTurnStrategies" $(find src/pipecat -name '*.py') | sort
src/pipecat/extensions/voicemail/voicemail_detector.py:631
src/pipecat/services/inworld/realtime/llm.py:445
src/pipecat/services/openai/realtime/llm.py:551
src/pipecat/services/xai/realtime/llm.py:394
```

세 개의 LLM 쪽은 패턴을 공유합니다 —
`user_turn_strategies=ExternalUserTurnStrategies() if emits_turn_frames else None` — 그리고 STT service가
아니므로 **아홉에 속하지 않습니다.** (네 번째 hit은 `extensions/voicemail/voicemail_detector.py:631`이고,
`LLMUserAggregatorParams`를 직접 생성합니다. STT service도 realtime LLM도 아니며 이 세기들 어디에도
속하지 않습니다. outline은 이 mechanism에 대해 세 site를 지목했지만, source에는 넷이 있고 그중 셋이
service입니다.)

---

## 19. boson을 실제 audio 위에 다시 세우기

이제 boson을 세 부분 모두 옆에 놓으십시오. **이것은 mechanism differential이지 판정이 아닙니다.**
[[ch-13/read]]이 점수를 매깁니다.

### 19.1 boson이 오늘 turn을 무엇으로 결정하는가

boson-agent에는 **interruption subsystem에 audio path가 아예 없습니다.** [[boson-interrupt-subsystem]]이
단호하게 진술합니다: *"Every interruption decision in the codebase — `PartialDetector.is_partial`,
`WordFilterPolicy.evaluate`, `fillers.is_filler`, `InterruptionGate.allows` — takes `text: str` as
its primary argument. There is no audio path, no energy threshold, no VAD."*

그리고 `boson-agent/packages/**/*.py` 전체에 대한 `deepgram|whisper|speech_to_text|vad` grep은 **hit이
0개**입니다 ([[stt-service-interface]], [[stt-korean-providers]]). 따라서 이 chapter의 2부는 boson에
port가 아니라 **순수 추가(net addition)**입니다. 이관할 것이 없습니다.

살아 있는 end-of-turn mechanism은 Gateway 안의 silence timer입니다:

- `gateway/server/websocket.py:616` — `_start_silence_timer`가 `silence_timeout_ms / 1000`만큼 sleep합니다.
  default **2000 ms**. 그다음 `_finalize_partial`(`:661`)을 호출합니다 ([[boson-gateway-server]]).
- `gateway/interrupt/policy.py` — `default_bargein_policy()`는
  `CompositePolicy([DurationPolicy(min_ms=500), WordFilterPolicy(...)], mode="all")`입니다
  ([[boson-interrupt-subsystem]]).

> **outline에 필요한 정정 하나.** [[endpointing-turn-boundary]]는 `PartialDetector.should_finalize`를
> *"the entire boson end-of-turn mechanism"*이라고 부르고, outline은 `ExternalUserTurnStrategies`가
> `PartialDetector`를 *"be deleted outright rather than merely replaced"* 하게 해 줄 것이라고 말합니다.
> [[boson-interrupt-subsystem]]은 둘 다에 반박합니다: `PartialDetector`는 `bootstrap.py:316`에서 정확히
> 한 번 생성되고, `core.py:175`가 보관하며, **`self._partial_detector`는 어디에서도 읽히지 않습니다** —
> 남아 있는 모든 참조는 `packages/gateway/tests/`에 있습니다. 이미 dead code입니다. 실제로 도는 것은
> `gateway/server/websocket.py`의 `_start_silence_timer` / `_finalize_partial`입니다. 따라서:
> `PartialDetector`는 오늘 삭제할 수 있고, 그 이유는 Pipecat과 아무 상관이 없습니다. 이 chapter의 3부가
> 대체하는 것은 살아 있는 2000 ms timer입니다. 둘을 뭉치지 마십시오.

### 19.2 하나의 timeline 위의 세 가지 모양

판단이 아니라 mechanism입니다. 셋 모두 고객의 마지막 유성음 sample에서부터 측정합니다.

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

구조적 차이는 한 단어입니다: **serial 대 overlapped.** realtime_voice의 transcription round trip은 endpoint
결정 *이후에* 시작됩니다. streaming STT의 그것은 고객이 발화하는 내내 돌고 있었고 마무리만 하면 됩니다.
[[ch-03/read]] §5.3이 같은 두 모양을 timeline 위에 올렸습니다. 이 chapter의 2부는 두 번째 줄의 항들을
이름 부를 수 있게 만드는 것입니다.

`CLAUDE.md`의 boson 목표는, [[rtv-vs-pipecat-gap]]에 인용된 대로 *"P50 at or below 1.0 seconds and P95
at or below 1.5 seconds"*이며 *"from the last voiced user sample to the first audible assistant sample,
including end-of-turn/VAD time"*으로 측정됩니다. 위의 모든 항이 그 window 안에 있습니다. [[ch-11/read]]이
예산 산술을 합니다. 이 chapter의 일은 각 항을 파일과 줄 번호가 뒤에 있는 양으로 만드는 것이었습니다.

### 19.3 무엇이 mapping되고 무엇이 되지 않는가

| boson piece | Pipecat counterpart | Faithful? |
|---|---|---|
| `_start_silence_timer(2000 ms)` | a stop strategy — `SpeechTimeout(0.6)` or the smart-turn default | yes, and denominated differently |
| `DurationPolicy(min_ms=500)` | `VADParams.start_secs = 0.2` → 6 chunks | yes |
| text-triggered barge-in | `TranscriptionUserTurnStartStrategy(use_interim=True)` | **exactly** — boson's behaviour is a supported Pipecat config |
| `WordFilterPolicy(ignore_words=[…], max_chars=3)` | `MinWordsUserTurnStartStrategy(min_words=…)` | **no** — Pipecat counts *words*, boson counts *characters*, and Korean backchannels are 1 word / 2 chars |
| `fillers` registry (agent-registered callback) | none | write a `BaseUserTurnStartStrategy` subclass |
| `PartialDetector` | — | already dead code; delete regardless |

`WordFilterPolicy` 행이 내면화할 것입니다. `MinWordsUserTurnStartStrategy`는 이렇게 합니다:

**`src/pipecat/turns/user_start/min_words_user_turn_start_strategy.py` L108–111**

```python
        min_words = self._min_words if self._bot_speaking else 1

        word_count = len(frame.text.split())
        should_trigger = word_count >= min_words
```

`len(frame.text.split())`. "네"는 그 셈법으로 token 하나이고 `min_words=1`을 통과합니다. 그것은
*문자*로는 두 글자이고 boson의 `max_chars=3`에는 걸립니다. 그것을 잡으려고 `min_words=2`로 설정하면
"잠깐만요" — 진짜 발언권 주장 — 도 함께 억눌립니다. **Pipecat에는 한국어 backchannel filter가 없고, 가장
가까운 것은 잘못된 단위를 셉니다.** 그 gap은 실재하고, 당신이 닫을 몫이며, §20이 그것을 첫 번째 probe로
만듭니다.

> 💡 **쉬운 설명 — 왜 word count가 한국어에서 잘못된 단위인가요?**
> `"네".split()` → `["네"]`, 길이 1. `"잠깐만요".split()` → `["잠깐만요"]`, 길이 1. 한국어는 교착어라
> 영어에서 여러 단어로 쓸 내용이 하나의 어절에 붙습니다. 영어라면 "yeah"(1 word)와 "hold on a
> second"(4 words)가 word count로 깔끔하게 갈리지만, 한국어에서는 둘 다 1입니다. boson이 문자 수를
> 세는 것은 우연이 아니라 언어에 맞춘 선택이었습니다. Pipecat의 단위를 그대로 쓰면 이 구별이
> 통째로 사라집니다.

L108의 조건문에도 주목하십시오: `min_words`는 **bot이 말하고 있는 동안에만** 적용되고, 그 외에는 1로
떨어집니다. strategy는 침묵 중의 turn-taking보다 interruption에 대해 더 엄격하며, 그것은 옳은 비대칭이고
직접 만들었다면 기억해서 넣어야 했을 것입니다.

---

## 20. Framework-extension probe

세 개의 수(move)입니다. 각각이 이 chapter의 mechanism을 chapter가 다루지 않은 domain에 적용합니다.
[[ch-08/read]] 전에 답을 쓰십시오.

**Probe 1 — 한국어 backchannel strategy.** §19.3은 `MinWordsUserTurnStartStrategy`가 boson의
`WordFilterPolicy`가 문자를 세는 곳에서 단어를 센다는 것, 그리고 두 시스템 모두 VAD layer에서 backchannel을
걸러내지 않는다는 것을 확립했습니다.
`KoreanBackchannelUserTurnStartStrategy(BaseUserTurnStartStrategy)`를 스케치하십시오. 명시적으로 답할
design 질문 셋: (a) 그것이 start list에서 `VADUserTurnStartStrategy` *앞*에 앉는가 *뒤*에 앉는가, 그리고
§17의 break-on-`STOP` semantics가 당신의 답에 대해 무엇을 뜻하는가? (b) `VADUserTurnStartStrategy`는
text를 전혀 나르지 않는 `VADUserStartedSpeakingFrame`에서 발화합니다 — 그렇다면 text 기반 filter가 그것을
억제할 수 있는가, 아니면 VAD strategy를 통째로 빼고 transcript 전용으로 가야 하는가? (c) transcript
전용으로 간다면, §5.2의 onset-latency 분석은 당신이 무엇을 포기했다고 말하는가? (b)에는 정답이 있고,
그것은 편안한 답이 아닙니다.

**Probe 2 — 한국어 형태론 turn analyzer.** §14.1은 `LocalSmartTurnAnalyzerV3`이 audio 전용이고 transcript
text를 절대 읽지 않는다는 것, 그리고 한국어가 발화 완결을 형태론에서 명시적으로 표시한다는 것을
확립했습니다. `BaseTurnAnalyzer`는 다섯 method짜리 ABC입니다(`speech_triggered`, `params`, `append_audio`,
`analyze_end_of_turn`, `clear`). 꼬리의 `-는데` / `-고` / `-지만`에 `INCOMPLETE`를, `-습니다` / `-어요`에
`COMPLETE`를 반환하는 `KoreanEndingTurnAnalyzer`를 설계하십시오. 그다음 어려운 부분에 답하십시오:
`append_audio(buffer, is_speech)`는 text가 아니라 **audio**를 받습니다. 그러면 당신의 analyzer는 transcript를
어디에서 얻는가 — 그리고 §16의 `stt_deadline` 산술은 transcript를 기다려야 하는 analyzer의 latency에 대해
무엇을 말하는가? 옳은 모양이 analyzer이기는 한가, 아니면 `TurnAnalyzerUserTurnStopStrategy` 뒤에 쌓인
두 번째 stop strategy인가? §17의 chain semantics로부터 정당화하십시오.

**Probe 3 — provider 결정을, 결정으로서 내리기.** §11의 표와 §12의 evidence class를 사용해서, Lina TMR
shortlist를 각 행의 evidence class를 이름 붙인 *ranked* list로 만드십시오. 지켜야 할 제약: 검증된 한국어,
streaming(segmented 아님), 그리고 §12.5의 어떤 정확도나 8 kHz 숫자도 없다는 부재. 그다음 그것을 실제로
만드는 질문에 답하십시오: Deepgram은 0.35 s — 표에서 공동 1위 — 에 있으면서 in-repo 한국어 evidence가
**전혀** 없고, Soniox는 같은 0.35 s에 검증된 mapping *그리고* 선택적 server-side endpointing을 함께
가지고 있습니다. 어떤 상황에서 그래도 Deepgram benchmark를 먼저 돌리겠습니까? 선호가 아니라 상황을
이름 붙이십시오.

---

## 21. 산출물: Lina TMR turn-boundary configuration

아래의 모든 값은 위의 어떤 section에서 결정되었고, 그 section을 이름으로 밝힙니다. 이것은 configuration
이지 Pipecat 대 boson에 대한 권고가 아닙니다 — [[ch-13/read]]이 그 질문을 소유하고 이 산출물은 그 답에
조건부입니다.

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

**다섯 개의 결정과 그 이유.**

1. **`stop_secs`는 0.2에 머뭅니다.** §15.2: §11의 23개짜리 표에 있는 모든 상수가 그 값에서 측정되었고,
   두 timing strategy 모두 당신이 그것을 옮기는 순간 runtime warning을 냅니다. 그것을 옮길 이유가 있다면,
   benchmark를 다시 돌릴 이유도 있는 것입니다 — 그 둘은 같은 결정입니다.

2. **`ttfs_p99_latency`는 명시적으로 넘기고, 절대 상수에 맡기지 않습니다.** §11의 숫자들은 영어를
   가정했고 sample rate가 명시되지 않았습니다. §12.5는 한국어나 8 kHz 숫자가 어디에도 존재하지 않는다고
   말합니다. default를 ship하는 것은 남의 측정치를 당신 것인 양 ship하는 것입니다. 그리고 §10의
   `ttfs is None` branch는 아예 잊어버리면 조용히 1.0 s를 얻는다는 뜻입니다.

3. **smart-turn default를 유지합니다.** §14.1: 그것은 acoustic이므로 한국어 언어 위험을 나르지 않습니다.
   동시에 한국어 문말 어미로부터 아무 도움도 받지 못하고, probe 2가 그것을 닫으려고 존재합니다. 그
   probe가 측정 가능한 무언가를 내놓기 전까지는 유지하는 것이 저위험 입장입니다.

4. **`confidence`와 `min_volume`은 tuning되지 않고 UNVERIFIED로 표시됩니다.** §2.3이 이 울타리를 조심스럽게
   쳤습니다: frame-count 산술은 8 kHz로 정확히 이전되고, *threshold*는 μ-law companding이 바꾸는 신호
   분포에 조건부이며, 이 repository의 어떤 evidence도 그것에 대해 말하지 않습니다. default를 ship하고,
   계측하고, 첫 100통을 측정으로 취급하십시오.

5. **`user_turn_stop_timeout=5.0`은 유지하되 표시해 둡니다.** §17: framework가 turn을 강제 종료하기 전에
   고객이 5초 동안 침묵한다는 것은 sales call에서 긴 시간입니다. 그것은 strategy chain이 deadlock할 때에만
   발화하고, 그런 일은 없어야 합니다 — 따라서 값을 낮추는 것은 드문 나쁜 경험을 더 흔한 조기 절단과
   맞바꾸는 것입니다. 숫자를 건드리기 전에 `on_user_turn_stop_timeout`을 계측하십시오. 절대 발화하지
   않는다면 값은 상관없습니다. 정기적으로 발화한다면 당신에게 있는 것은 timeout bug가 아니라 chain
   bug입니다.

**이 configuration이 하지 않는 두 가지, 잊히지 않도록 진술합니다.**

- **`ExternalUserTurnStrategies`를 쓰지 않습니다.** §18은 기본값으로 turn 결정을 가져가는 service를 셋
  세었습니다. 경계를 provider에게 넘긴다는 것은 §14–§17의 chain 전체 — §17의 veto와 watchdog을 포함해서 —
  가 `__post_init__`(§18.3)에 의해 통째로 교체된다는 뜻이고, turn 기반 service들에 대해서는
  `supports_ttfs → False`라는 뜻입니다. 그것은 provider 선택보다 큰 결정이고, probe 3 이전이 아니라
  이후에 내려야 합니다.

- **provider를 결정하지 않습니다.** probe 3이 그것을 하고, §12.5는 8 kHz μ-law 한국어 benchmark를 돌리기
  전까지 shortlist를 정직하게 정렬할 수 없다고 말합니다.

→ **[`figures/turn-boundary.html`](figures/turn-boundary.html)의 panel three가 이 configuration을 살아
있게 만듭니다.** `stop_secs` slider와 provider selector가 공유된 시간 축 위에서 turn 경계를 움직이고,
smart-turn 판정과 `max(0, ttfs_p99 - stop_secs)` safety net이 절대적인 `stt_deadline`에 맞서 경쟁하는
시계로 돌아가며, external-endpointing strip이 아홉 site 전부를 guard별 toggle과 함께 나열합니다 —
Soniox와 AssemblyAI 두 행을 함께 구동하는 공유 toggle 하나를 포함해서, §18.2의 동일한 gating이 주장이
아니라 눈에 보이게 합니다. `stop_secs`를 0.2에서 옮기면 panel은 source가 내는 것과 같은 warning을 냅니다.

---

## 다음 챕터로

이 chapter가 앞으로 넘기는 것들. 이후 chapter들이 다시 유도하지 않고 인용할 수 있도록 이름을 붙입니다:

- **VAD 양들** (§2) — `start_secs=0.2`와 `stop_secs=0.2`는 16 kHz *에서도* 8 kHz *에서도* **6 chunk,
  192 ms**입니다. 512/16000과 256/8000이 둘 다 0.032이기 때문입니다. `round(6.25) = 6`이지 7이 아닙니다 —
  excerpt들은 7이라고 말하고 틀렸습니다. [[ch-08/read]]은 onset 숫자를 barge-in latency에 씁니다.
  [[ch-11/read]]은 offset 숫자를 예산에 씁니다. 둘 다 다시 유도하지 않습니다.

- **three-way split과 두 frame family** (§1, §3) — model / analyzer / controller, 그리고
  `VADUserStartedSpeakingFrame`("voice가 있다") 대 `UserStartedSpeakingFrame`("이것은 turn이다").
  [[ch-08/read]]의 cascade는 `user_turn_processor.py:210`의 `broadcast_interruption()`에서 시작하고,
  그곳이 이 chapter의 chain이 끝나는 곳입니다.

- **mount point** (§4) — CHANGELOG L4402에 따라 `TransportParams`가 아니라 `LLMUserAggregatorParams`.
  거기에 두 개의 silent failure mode가 매달려 있습니다: `vad_analyzer`가 없으면 VAD frame이 전혀 없고,
  VAD frame이 없으면 `SegmentedSTTService`는 아무것도 transcribe하지 않습니다.

- **23개 상수의 `stt_latency.py` 표** (§11) — 여기서 한 번만 만들어지고, `DEFAULT_TTFS_P99 = 1.0`,
  `NVIDIA` / `WHISPER`는 측정이 아니라 alias, 0.35–2.14 s의 폭, 그리고 모든 값이 `stop_secs=0.2`에서
  측정됨. [[ch-11/read]]의 waterfall은 선택된 값 하나를 소비하고 표를 다시 그려서는 안 됩니다.

- **한국어 evidence class** (§12) — 검증된 `Language.KO` mapping 12개, 평범한 문자열 집합에 의한 열세
  번째로서의 FunASR, repo가 아무 입장도 취하지 않는 passthrough로서의 Deepgram, documented exclusion으로서의
  AssemblyAI, 영어 전용인 Cartesia, Konkani인 `KOK_IN`, 그리고 빠른 service와 한국어가 검증된 service가
  서로 다른 class인 ElevenLabs class split. 그리고 부재: **tree 어디에도 정확도 숫자 0개, 8 kHz 숫자
  0개.** [[ch-07/read]]이 TTS에 대해 거울상 질문을 마주합니다.

- **`effective_stt_wait = max(0, ttfs_p99 - stop_secs)`** (§15.2)와 절대적인
  `stt_deadline = timestamp - stop_secs + ttfs_p99` (§16). 두 항 모두 이 chapter의 서로 다른 layer에서
  오고, 그것이 이 chapter가 세 layer 전부를 가르쳐야 했던 이유입니다.

- **9 / 8 / 2 / 7 세기** (§18) — 여덟 개 service 파일에 걸친 아홉 개 `ExternalUserTurnStrategies`
  assignment site, 무조건 둘(`cartesia/turns/stt.py:198`, `deepgram/flux/stt_base.py:250`), flag로 gate된
  일곱, 그리고 `SarvamRealtimeSTTService`의 기본 활성 `endpointing="vad"`를 세면 **기본값으로 발화하는
  것 셋**. Soniox와 AssemblyAI는 동일한 `vad_force_turn_endpoint` flag를 공유하며 같은 group에 속합니다.

[[ch-07/read]]은 voice loop의 나머지 절반을 가져갑니다: streaming TTS, 첫 가청 sample, word timestamp,
그리고 합성 쪽에서 다시 한번 한국어 provider 질문. 그것은 §11 표의 모양을 model로 필요로 하고 — 자기
표를 직접 만들 것입니다 — [[ch-05/read]]의 8 kHz μ-law 제약을 필요로 합니다. 그 제약은 이쪽 path를 무는
만큼 출력 path도 세게 뭅니다.

잊히지 않도록 여기에 세워 두는 열린 질문들:

- **한국어 8 kHz μ-law STT benchmark.** §12.5. provider 선택의 blocking item이고 이 repository의 어떤
  것도 그것을 대신할 수 없습니다. 그것은 또한 §21의 `confidence` / `min_volume` 결정을 조용히
  막고 있습니다. 둘 다 μ-law가 바꾸는 분포에 대한 threshold이기 때문입니다.
- **한국어 형태론이 turn 결정에 속하는가 자체.** §14.1과 probe 2. 기본 stop strategy는 이 언어에서 가장
  신뢰할 만한 end-of-turn cue를 볼 수 없습니다. 그것이 실제 통화에서 측정 가능한 비용을 치르는지는
  미지이고 테스트 가능합니다.
- **경계를 provider에게 넘길지 여부.** §18. `ExternalUserTurnStrategies.__post_init__`은 §17의 veto를
  포함한 chain 전체를 교체합니다. probe 3 이후에 결정하고, [[ch-13/read]]에서 점수를 매기십시오.
- **boson의 `fillers` registry.** [[boson-interrupt-subsystem]]은 agent가 등록하는 언어별 무시 callback을
  기록하고 있고 Pipecat에는 hook이 없습니다. probe 1이 backchannel 쪽 절반을 다룹니다. registry의
  *동적인* 절반 — agent가 통화 중에 자기 무시 목록을 바꾸는 것 — 은 aggregator 생성 시점에 고정되는
  strategy list에 대응물이 없습니다. [[ch-12/read]]이 그것을 소유합니다.

