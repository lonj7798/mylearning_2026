---
title: "Streaming TTS: 첫 가청 sample, word timestamp, 그리고 한국어"
chapter: ch-07
phase: voice-io
course: pipecat
lang: ko
companion_of: read.md
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
verified: 2026-08-25
---

# ch-07 — Streaming TTS: 첫 가청 sample, word timestamp, 그리고 한국어

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 (frame, processor, pipeline, queue, aggregator, endpointing,
> back-pressure, interruption, timestamp, context, lookahead 등).

> **범위(scope). 미리 밝히고 이 chapter 전체에 강제합니다.** 이 chapter는 두 design이 **무엇을 하는지**를
> 기술합니다. 순위를 매기지 않습니다. §9에서 realtime_voice의 `KoreanPhraseChunker`가 등장할 때, 그것은
> 입력과 출력을 가진 하나의 algorithm으로서, 자기 입력과 출력을 가진 Pipecat의 algorithm 옆에 놓일
> 뿐입니다. 그리고 유일하게 이루어지는 비교는 *측정*입니다 — 같은 input, 두 경로, 출력된 결과. "더 낫다"
> 없음, "채택해야 한다" 없음, "옳은 선택" 없음. [[ch-13/read]]가 무언가를 채점하는 유일한 chapter이고,
> 그러기 위해서는 ch-05부터 ch-12까지의 mechanics를 손에 쥐고 있어야 합니다.
>
> 이 chapter에서 두 가지는 **판정(verdict)이 맞습니다**. 그리고 둘 다 어떤 design에 대한 판정이 아니라
> Pipecat source에 대한 판정입니다: 이 chapter를 생성한 outline이 두 개의 사실 주장을 했는데, commit
> `0cbf9c5b`의 코드가 그것들과 모순됩니다. 그 위에 무언가를 쌓아 올리기 전에, §0에서 평범하게
> 정정합니다.

---

## 왜 이 챕터인가

[[ch-04/read]]는 `PipelineWorker`가 돌고 있고, 네 개의 exit이 배선되어 있고, 일곱 개 processor chain의
끝에서 `transport.output()`이 audio를 재생하고 있는 지점에서 당신을 남겨두었습니다. 그것은 `stt`와 `tts`
라고 이름 붙은 두 상자를 의도적으로 열지 않았습니다 — 그것들을 frame을 produce하고 consume하는 processor로
취급했고, runtime 목적에서는 그게 그것들의 전부이기 때문입니다.

이 chapter는 두 번째 상자를 엽니다. canonical pipeline([[canonical-voice-bot]],
`examples/getting-started/06-voice-agent.py:81-91`)의 position 5는 하나의 object, `CartesiaTTSService`
이고, 알고 보면 그것은 chain 전체에서 **가장 latency에 민감한** processor입니다 — 느려서가 아니라,
고객이 무언가를 듣기 전에 일어나는 **마지막** 것이기 때문입니다. 그것보다 upstream에 있는 모든
millisecond는 보이지 않고, 그것 **내부**의 모든 millisecond는 dead air로 들립니다.

구체적으로, Lina의 경우: 고객님이 한 문장을 마치고, Silero가 `UserStoppedSpeakingFrame`을 발사하고, LLM이
한국어 token을 streaming하기 시작하고, 그러고 나서 전화에서 첫 소리가 나오기까지 gap이 있습니다. 그 gap이
고객이 "이 상담원 좀 느리네."로 경험하는 것입니다. 이 chapter는 그 gap을 구성하는 세 가지에 대한 것이고,
셋 다 `TTSService` 안에 살며, 그중 하나는 byte 수준 monitoring이 볼 수 없는 것입니다.

또한 이 chapter는 한국어가 configuration string이기를 멈추고 constraint가 되기 시작하는 chapter입니다.
이 tree에는 Korean entry를 가진 language map이 12개 있고, word timestamp를 emit하는 service가 12개 있으며,
교집합은 6개입니다. 마지막으로 **발화된(spoken)** 단어에서 assistant context를 잘라내는 barge-in을 원한다면
— [[ch-08/read]]가 그 위에 세워지는 mechanism — 당신은 그 6개 중에서 고르는 것이고 다른 선택지는 없습니다.
그리고 on-prem 한국어 TTS를 원했다면, 이 chapter가 그것이 이 tree에 아예 없다는 사실을 알게 되는 곳입니다.

아래 전체를 관통하는 세 개의 질문:

1. **왜 이 design 전체가 하나의 숫자를 optimise하며, 그 숫자는 무엇인가?** TTFB가 아닙니다. §1.
2. **word 수준 timing은 실제로 어디에 사는가?** `WordBoundaryFrame`은 없습니다. `TTSWordFrame`도 없습니다.
   §6이 대신 무엇이 있는지, 그리고 왜 답이 `pts` field인지 알려줍니다.
3. **한국어는 여기서 무엇을 비용으로 요구하는가?** 특정한 provider shortlist 하나, silent-failure mode
   하나, 그리고 zh/ja를 위해 쓰였고 한국어가 의도적으로 그 *바깥으로* 떨어지는 code path 하나. §7과 §8.

---

## 0. 시작하기 전에 해야 할 두 개의 정정

이 chapter를 만들어낸 outline, 그리고 그것을 먹인 excerpt 중 하나는 각각 source가 모순되는 무언가를
진술합니다. 이 course의 standing rule은 **source가 이긴다**는 것입니다. 두 정정 모두 결론을 바꾸므로,
각주가 아니라 맨 앞에 옵니다.

### 0.1 live ladder는 네 개가 아니라 세 개의 class다. `AudioContextTTSService`는 deprecated **맞습니다**.

outline은 이렇게 말합니다, 그대로:

> the 'everything with Word in the name is deprecated' escape clause does NOT collapse the tree,
> because `AudioContextTTSService(WebsocketTTSService)` at :2083 has no 'Word' in its name, is not
> deprecated, and is a fourth live class

파일을 여십시오.

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

`@deprecated` decorator, `.. deprecated:: 0.0.105` directive, 그리고 첫 줄이 "Deprecated."인 docstring을
달고 있습니다. outline의 주장은 거짓입니다. excerpt [[tts-service-interface]]는 *목록*은 맞게 가져갔지만
(deprecated class 중에 `AudioContextTTSService`를 포함시켰습니다) *헤드라인*을 엉성하게 썼습니다
("everything with 'Word' in the name is deprecated" — 참이지만 전부는 아님; `AudioContextTTSService`는
이름에 "Word"가 없는데도 deprecated입니다).

그 귀결이 중요합니다. outline은 "the live ladder a subclass author actually chooses from is those four,
and the axes are two — websocket transport, and per-turn audio contexts"라고 결론지었습니다.
**두 절반 모두 살아남지 못합니다.** live ladder는 세 개의 class입니다:

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

선언된 class는 여덟 개, deprecated는 다섯 개, **live는 세 개**: `TTSService`, `WebsocketTTSService`,
`InterruptibleTTSService`. 그리고 axis는 둘이 아니라 정확히 **하나** — websocket transport — 입니다.
per-turn audio context가 `TTSService` 자체 *안으로* 옮겨갔기 때문입니다. deprecation docstring이 그대로
그렇게 말합니다: *"Audio context management (previously the main purpose of this class) is now built into
TTSService."* audio-context machinery는 §5에서 root class 위에, 저 문장이 말하는 바로 그 자리에 앉아
있는 것을 보게 됩니다.

inheritance shape는 여전히 chain이 아니라 two-branch tree입니다 — `WordTTSService`가 root에 매달려 있고
`WebsocketTTSService`는 `TTSService` **와** `WebsocketService`로부터 multiple inheritance를 씁니다 —
그래서 outline이 "the usual shorthand does not cover the tree"라고 한 것은 맞았습니다. 틀린 것은 그
안에서 어떤 class가 살아 있느냐였습니다.

### 0.2 한국어 word-grouping 경로는 "검증되지 않은 가정"이 아니다. test가 네 개 있다.

outline은 이렇게 말합니다:

> Korean is deliberately excluded from the CJK word-grouping paths (both Cartesia and ElevenLabs
> test `base_lang in {"zh", "ja"}`), so it falls through to the space-separated branch and gets
> per-token frames — which matches 어절 spacing but is an **untested assumption in this code**.

첫 절은 맞고 §8.1에서 검증됩니다. 마지막 절은 거짓입니다. `tests/test_cartesia_tts.py`는 네 개의 한국어
assertion을 담고 있고, 그중 하나는 정확히 어절 spacing round trip입니다:

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

`tests/test_aggregated_frame_sequencer.py`에 두 개가 더 있고(`:765`의 `TestCJKLanguages` class에
`# --- Korean ---` section, `:774-805`), `tests/test_word_completion_tracker.py:1866+`에 평행한 한 쌍이
있습니다. 한국어 word grouping과 한국어 slot completion은 frame 수준에서 unit-test되어 있습니다.

진짜로 검증되지 않은 것은 — 그리고 이것이 outline이 *했어야 할* 주장인데 — **behavioural**한 무엇입니다.
이 repo의 어떤 test도 한국어 text를 실제 TTS provider에 보내고 결과를 들어보지 않습니다. `Language.KO: "ko"`
를 선언하는 service가 실제로 알아들을 수 있는 한국어를 만들어내는지, 또는 그 timestamp가 audio와 맞아
떨어지는지를 assert하는 test는 없습니다. §7.4는 그 provenance를 붙여서 6-service 교집합을 다시 진술합니다.
"frame plumbing은 test되어 있다"와 "audio는 한 번도 확인된 적 없다"의 구별이야말로, 당신이 기댈 수 있는
주장과 당신이 직접 검증해야 하는 주장 사이의 차이 전부이기 때문입니다.

> 💡 **쉬운 설명 — "frame plumbing은 test됨 / audio는 미검증"이 실무에서 뜻하는 것**
> repo의 한국어 test들은 전부 *가짜 입력*으로 돌아갑니다. `["안녕하세요", "반갑습니다"]`라는 문자열
> list와 `[0.0, 0.2]`라는 숫자 list를 함수에 넣고, 나오는 tuple list가 맞는지 봅니다. 즉 "provider가
> 이런 모양의 데이터를 주면 우리 코드가 올바르게 처리한다"는 것만 증명합니다. provider가 애초에
> 그런 데이터를 주는지, 그 `0.2`초가 실제 오디오의 그 단어 위치와 일치하는지는 아무도 확인하지
> 않았습니다. 그래서 §7.4의 "6개"는 **테스트할 후보 목록**이지 **작동하는 목록**이 아닙니다.

---

## 1. 이 layer가 optimise하는 단 하나의 숫자

### 1.1 개념이 아니라 산수부터 시작하십시오

Lina의 LLM이 t=0에 한국어 문장 하나를 결정했다고 합시다. 그 문장은 말로 하면 4.2초 분량입니다. 세 개의
timing, 전부 실제로 있는 것들:

| | Provider A | Provider B |
|---|---|---|
| First byte of audio arrives | 180 ms | 180 ms |
| Leading silence in that audio | 20 ms | 210 ms |
| Total synthesis wall time | 900 ms | 380 ms |
| Playback duration | 4,200 ms | 4,200 ms |

Provider B는 문장 전체를 절반도 안 되는 시간에 synthesise합니다. 어떤 byte 수준 dashboard에서 보든
first-byte는 A와 동일해 보이고 throughput은 극적으로 더 좋아 보입니다.

고객은 A에서 **200 ms**, B에서 **390 ms** 동안 침묵을 듣습니다. B는 귀에 대고는 거의 두 배 느립니다.

그리고 900 ms 대 380 ms라는 synthesis 차이는 청자에게는 아예 존재하지 않습니다. audio는 1× wall clock으로
재생됩니다. A의 남은 720 ms의 synthesis는 이미 돌아가고 있는 4,200 ms의 playback *아래에서* 일어납니다.
synthesis가 real time보다 빠르게 유지되는 한 — 이 provider들 중 어느 것이든 큰 여유로 그렇습니다 —
첫 가청 sample 이후의 모든 것은 공짜입니다.

그것이 이 layer의 thesis 전부이고, 그래서 이 파일이 자기 complexity 예산을 speed가 아니라 ordering과
buffering에 쓰는 것입니다. 노출되는 interval은 단 하나입니다:

```
[ LLM decided ] ────────────────────────────────► [ first AUDIBLE sample ]
                        everything after this point is hidden by playback
```

> 💡 **쉬운 설명 — 왜 synthesis 속도가 "공짜"가 되나요?**
> 4.2초짜리 오디오는 어차피 4.2초 동안 재생됩니다. 그 4.2초 동안 뒤쪽 오디오를 만들 시간은 4.2초나
> 있습니다. Provider A가 나머지를 720 ms 안에 다 만들든 3초에 걸쳐 만들든, 재생은 끊기지 않습니다.
> **단 하나의 예외**가 §1.5입니다: 만드는 속도가 재생 속도보다 느려지면(1× 미만) queue가 말라서 문장
> 중간에 침묵이 들립니다. 그것만 아니면 "빠른 synthesis"는 그냥 queue를 채울 뿐입니다.

### 1.2 그 interval을 재는 metric은 TTFA이고, TTFB가 아니다

두 개의 metric type이 존재하고 둘은 서로 교환 가능하지 않습니다.

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

`ttfa = ttfb + leading_silence`이고, docstring은 `ttfb` field가 두 번째 측정이 아니라 편의를 위한
**복사본**이라고 명시합니다 — *"don't aggregate both."* Lina dashboard를 만들면서 통화 전체의 TTFB를
합산하고 TTFA도 합산하면, response time을 두 번 세게 됩니다.

log observer는 정확히 이 breakdown을 출력합니다:

**`src/pipecat/observers/loggers/metrics_log_observer.py:153-157`**
```python
        elif isinstance(metrics_data, TTFAMetricsData):
            logger.debug(
                f"📊 {processor_info} TTFA{model_info}: {metrics_data.ttfa}s "
                f"({metrics_data.leading_silence}s leading silence) at {time_sec:.3f}s"
            )
```

저 괄호 안의 숫자가 §1.1의 Provider B가 못하는 그 숫자이고, 다른 어디에도 나타나지 않는 그 숫자입니다.

### 1.3 `leading_silence`는 선언되는 게 아니라 측정된다

이것은 provider가 제공하는 field가 아닙니다. Pipecat이 audio byte가 streaming되어 들어오는 동안 그 위에
energy 기반 onset detector를 돌려서 계산합니다.

**`src/pipecat/processors/metrics/frame_processor_metrics.py:190-238`** (산수 부분만 발췌)
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

`_ttfa_active`는 TTFB가 멈출 때 set됩니다 — 즉 TTFA scan은 첫 byte가 도착하는 바로 그 순간에 시작합니다:

**`src/pipecat/processors/metrics/frame_processor_metrics.py:179-182`**
```python
        # The first byte has arrived; begin scanning leading silence so TTFA can
        # be reported as TTFB plus the silence duration (see process_ttfa_metrics).
        self._ttfa_active = True
        self._ttfa_buffer = b""
```

그리고 onset detector 자체는 max amplitude에 대한 threshold가 아니라 진짜 short-time-energy scan입니다:

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

적어 둘 가치가 있는 default 세 개, 당신이 이것들에 맞춰 tuning하게 될 것이기 때문입니다: onset resolution은
**1 ms**(`hop_ms`), gate는 **−40 dBFS**, 그리고 energy가 gate 위에 **50 ms** 이상 머물러야만 onset으로
쳐 줍니다. digital silence 대신 낮은 수준의 noise floor로 응답을 padding하는 provider도 여전히 올바르게
측정됩니다. −40 dBFS가 전형적인 noise-floor padding보다 위에 있기 때문이고, docstring이 정확히 그렇게
말합니다: *"Sits above typical TTS noise-floor padding and below voiced onset."*

> 💡 **쉬운 설명 — short-time RMS energy scan이란**
> PCM byte를 10 ms짜리 창(window)으로 자르고, 그 창을 1 ms씩 밀면서 각 창의 RMS(제곱평균제곱근, 대충
> "이 구간이 얼마나 시끄러운가")를 잽니다. RMS를 dBFS로 바꿔서 −40 dBFS를 넘는 창이 **연속 50 ms 이상**
> 이어지는 첫 지점을 onset으로 봅니다. "50 ms 연속" 조건이 중요합니다 — 그게 없으면 클릭 잡음 하나에
> onset이 잡힙니다. 그리고 RMS를 계산할 때 평균(DC offset)을 빼는데, 일정한 offset은 에너지를 가지지
> 않기 때문입니다. 결과: provider가 "무음"이라고 주장하는 구간이 실제로는 −60 dBFS 잡음이어도
> "아직 침묵"으로 올바르게 분류됩니다.

call site는 TTS service 안의 한 줄이고, 측정값이 나올 때까지 **모든** audio frame에 대해 실행됩니다:

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

저 여덟 줄의 순서에 주목하십시오. 밀도가 높고 §6에서 중요해지기 때문입니다: 한 context의 **첫** audio
chunk가 TTFB를 멈추고, word-timestamp PTS baseline을 설정하고, TTFA scan을 시작합니다 — 셋 다 같은
event에 anchor됩니다. word timestamp와 가청 onset 측정은 구성상 clock origin을 공유합니다.

### 1.4 세 개의 가산적 component, 그리고 각각의 주인

고객이 경험하는 `TTFA`는 세 개의 interval로 분해되고, 이 layer가 셋 다 통제합니다:

| Component | Owner | Where it is measured | Typical size |
|---|---|---|---|
| (a) aggregation delay | `TTSService` text aggregator | `TextAggregationMetricsData` | 200–300 ms per the docstring (§4.1) |
| (b) provider TTFB | the provider | `TTFBMetricsData` | 100–400 ms |
| (c) leading silence | the provider | `TTFAMetricsData.leading_silence` | 0–250 ms |

(a)는 사람들이 잊어버리는 것이고, 벤더를 바꾸지 않고 당신이 바꿀 수 있는 것입니다. 자기 metric type도
있습니다:

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

"From the first LLM token to the first complete sentence." 그것은 당신의 TTS provider와는 아무 상관이
없고 당신이 `TextAggregationMode`를 어떻게 설정했는지와 전적으로 상관 있는 dead air입니다. §4가 그것에
대한 것입니다.

### 1.5 1× 가정은 어디서 강제되는가

"Audio plays at 1× wall clock"은 손짓이 아닙니다. output transport의 chunking contract입니다.

**`src/pipecat/transports/base_output.py:130-136`**
```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

audio는 고정된 10 ms 배수로 pipeline을 떠나고, 구체적인 transport의 `write_audio_frame`이 그것들을 반대편
끝에 맞춰 pacing합니다([[ch-05/read]]가 그것을 소유합니다). 붙들고 있어야 할 귀결 두 개:

1. real time보다 빠르게 도는 synthesis는 첫 sample 이후로는 아무것도 사주지 않습니다 — 그냥 queue를
   채울 뿐입니다.
2. real time보다 *느리게* 도는 synthesis는 완전히 다른 failure입니다: queue가 마르고, 고객은 문장 중간에
   gap을 듣습니다. `TTSService`의 어떤 것도 그것을 감지하지 않습니다. 이 tree에 그런 metric은 없습니다.

→ **[tts-streaming.html](figures/tts-streaming.html)** — 지금 열고 §1 동안은 첫 번째 panel에 머무십시오.
provider-padding slider를 끌면서 두 mark가 벌어지는 것을 보십시오: TTFB는 고정된 채로 TTFA가 오른쪽으로
미끄러집니다. 그 gap이 `leading_silence`이고, 이 panel은 왜 provider가 "빠른데" 느리게 들릴 수 있는지를
체득하는 가장 빠른 방법입니다. panel의 두 번째와 세 번째 view로는 각각 §4.5와 §6.5에서 돌아오십시오.

---

## 2. class ladder, 제대로 읽기

### 2.1 무엇을 subclass하며, base는 무엇을 주는가

`TTSService`는 2,136줄이고 여덟 개의 class를 선언합니다. §0.1 이후, 살아 있는 것은 셋입니다. root는
이렇습니다:

**`src/pipecat/services/tts_service.py:109-114`**
```python
class TTSService(AIService):
    """Base class for text-to-speech services.

    Provides common functionality for TTS services including text aggregation,
    filtering, audio generation, and frame management. Supports configurable
    sentence aggregation, silence insertion, and frame processing control.
```

저 docstring이 말하지 *않는* 것에 주목하십시오: "converts text to audio"라고 말하지 않습니다. text에서
audio로 가는 것은 abstract method 하나입니다. 이 class가 하는 나머지 전부는 buffering, ordering,
bookkeeping입니다:

- *언제* provider를 호출할지 결정하는 text aggregator (§4)
- frame이 *어떤 순서로* 떠날지 결정하는 per-context audio FIFO와 단일 serialization queue (§5)
- *downstream이 각 단어를 언제 보는지* 결정하는 word-timestamp bookkeeping (§6)
- 세 개의 failure guard: zero-audio detection, interruption teardown, text-transform error handling

codec이 아니라, **가운데에 synthesis 호출이 하나 들어 있는 reordering buffer**로 읽으십시오.

### 2.2 abstract method, 그리고 `None` contract

abstract method는 정확히 하나입니다.

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

return type은 `AsyncGenerator[Frame | None, None]`이고 그 `| None`이 HTTP 대 websocket 구분의 전부입니다.
consumer가 설명해 줍니다:

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

그래서 websocket subclass의 `run_tts`는: text를 socket으로 보내고, `yield None` 하고, return합니다.
audio는 나중에 자기가 직접 audio context에 append하는 receive task 위에서 도착합니다. HTTP subclass의
`run_tts`는 진짜 `TTSAudioRawFrame`들을 yield하고 caller가 나중에 context를 닫습니다.

> 💡 **쉬운 설명 — `| None` 하나가 왜 두 아키텍처를 가르나요?**
> generator가 `None`을 yield한다는 것은 "나는 지금 줄 게 없다. 하지만 실패한 것도 아니다"라는 신호입니다.
> websocket service는 요청을 소켓으로 밀어 넣고 바로 끝납니다 — audio는 완전히 다른 task(receive loop)
> 에서 도착해 직접 context에 쌓입니다. 그러니 generator로 돌려줄 frame이 없습니다. HTTP service는
> 반대로 자기가 응답을 다 받아서 frame으로 yield합니다. 같은 abstract method 하나로 두 모양을 다
> 담기 위해 return type에 `| None`을 넣은 것이고, 이 차이가 §2.3의 metric 가용성 차이로 그대로
> 이어집니다.

### 2.3 진짜 HTTP/websocket 구분은 class가 아니라 property다

`supports_processing_metrics`가 그 구분이 실제로 선언되는 곳이고, 어느 service에 대해 어느 metric을
믿어야 하는지 알려주므로 그 논리를 읽을 가치가 있습니다:

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

Lina dashboard에 대한 실무적 귀결: websocket provider를 고르면(그리고 §7이 한국어 + timestamp를 갖춘 여섯
service 중 다섯이 websocket임을 보여줄 것입니다), 당신의 TTS service에 대한 `ProcessingMetricsData`는
0이 아니라 **부재**할 것입니다. 그것을 기대하는 panel을 만들지 마십시오. 당신이 가진 것은 TTFB와 TTFA입니다.

chunked-HTTP service에는 TTFA를 형성하는 상수가 하나 더 있습니다:

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

그것은 **500 ms pre-buffer**이고, TTFA에 직접 더해지며, docstring이 기술하는 glitch를 피하기 위해
의도적으로 취해진 것입니다. sentence aggregation 다음으로 이 파일에서 두 번째로 큰 단일 latency 결정이고,
HTTP 경로에만 적용됩니다. §7의 table에서 같은 provider의 HTTP variant가 websocket variant 옆에 나열된
것을 볼 때, 이 상수가 당신이 고르고 있는 것 중 하나입니다.

### 2.4 `InterruptibleTTSService` — 동작을 추가하는 유일한 live subclass

`WebsocketTTSService`는 연결성을 추가합니다. `InterruptibleTTSService`는 정확히 한 가지를 추가하고,
그것은 [[ch-08/read]] 전에 이해해 두어야 하는 것입니다:

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

barge-in마다 websocket을 통째로 teardown하고 reconnect합니다. 그것은 비쌉니다 — 대화 도중의 TLS handshake와
provider 쪽 session setup — 그리고 class docstring이 왜 그것이 필요한지 말해 줍니다:
*"Designed for TTS services that don't support word timestamps."* ID로 주소 지정할 수 없는, 반쯤 발화된
server 쪽 context는 되감을 수 없습니다. socket을 죽이는 것이 유일하게 가용한 cancel primitive입니다.

이것이 §7의 6-service 목록에 관심을 가져야 할 첫 번째 구체적 이유입니다. 한국어 가능 service 중 정확히
하나 — `LmntTTSService` (`lmnt/tts.py:78`) — 만이 `InterruptibleTTSService`를 상속하고, 그것이 word
timestamp를 emit하지 *않는* 유일한 한국어 가능 websocket service입니다. 두 사실은 같은 사실입니다.

---

## 3. text는 어떻게 들어오는가

aggregation 이전에 routing 질문이 있습니다: `TTSService`는 애초에 어떤 frame을 쳐다보는가?

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

세 개의 branch, 우선순위 순서로:

1. **`skip_tts`가 set됨** → 손대지 않고 통과. upstream processor가 "context에는 도달해야 하지만 절대
   발화되면 안 되는" text를 표시하는 방법입니다.
2. **이미 `AggregatedTextFrame`임** → 곧장 synthesis로, **aggregation 없음**. upstream의 누군가가 이미
   경계를 결정했습니다.
3. **transcription이 아닌 평범한 `TextFrame`** → 내부에서 aggregate.

branch 2가 extension point입니다. `LLMTextProcessor`는 당신이 건네주는 아무 `BaseTextAggregator`나 써서
`LLMTextFrame` → `AggregatedTextFrame` 변환을 하는 독립 `FrameProcessor`입니다:

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

이것이 존재한다는 것을 적어 두십시오. 이것이 한국어 인지 chunker가 **어떤 TTS service도 subclass하지 않고**
Pipecat pipeline에 들어가는 seam입니다: `BaseTextAggregator`를 구현하고, `LLMTextProcessor`에 건네주고,
그 processor를 `llm`과 `tts` 사이에 놓으면, `TTSService`는 branch 2를 타고 자기 aggregator를 절대 돌리지
않습니다. §10과 §11이 이것을 씁니다.

관련된 class 두 개:

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

`aggregated_by`는 `AggregationType | str`로 typing되어 있습니다 — closed enum이 아니라 열린 string입니다.
`"eojeol"`이나 `"phrase"`를 aggregation type으로 발명해도 system 전체를 그대로 흘러갑니다. 그것은
[[ch-02/read]]가 frame 수준에서 지목했던 것과 같은 open-sum-type 베팅이, 한 단계 아래에서 다시 나타난
것입니다.

---

## 4. 이름 붙은 예산 항목으로서의 sentence aggregation

### 4.1 mode, 그리고 그 docstring 속의 숫자

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

`SENTENCE`가 default이고, `__init__`에서 결정됩니다:

**`src/pipecat/services/tts_service.py:296-299`**
```python
        if text_aggregation_mode is None:
            text_aggregation_mode = TextAggregationMode.SENTENCE

        self._text_aggregation_mode: TextAggregationMode = text_aggregation_mode
```

Cartesia의 `__init__`은 그 예산을 한 문장의 솔직함을 덧붙여 반복합니다:

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

*"Streaming quality is good but less tested than sentence aggregation."* 그것은 framework가 TOKEN mode의
정직한 상태를 당신에게 말해 주는 것입니다. 액면 그대로 받아들이십시오: TOKEN은 실재하고, 지원되며,
주행거리가 더 짧습니다.

### 4.2 lookahead rule과 그 정확한 비용

`SENTENCE` mode는 구두점을 볼 때 boundary detector를 호출하지 않습니다. 구두점 **뒤의** 첫 **비공백**
문자를 볼 때 호출합니다.

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

비용은 **문장당 LLM token 하나**이고, 항상, 설계상 그렇습니다. LLM이 초당 40 token으로 streaming하면 그건
25 ms입니다. 긴 한국어 생성에서 초당 8 token으로 streaming하면 그건 125 ms입니다. provider의 TTFB와
가산적이고, `TextAggregationMetricsData`를 보고 있지 않으면 보이지 않습니다.

그것이 사주는 것은 "더 예쁜" 분할이 아닙니다. 소수점과 마침표가 같은 문자일 때 그 둘을 구별할 수 있는
능력입니다. docstring의 예시는 `$29.` 대 `$29. Next`입니다 — 그리고 [[rtv-vad-chunking]]을 읽어 왔다면,
그것이 `KoreanPhraseChunker._is_safe_period`가 명시적 guard로 푸는 것과 같은 문제라는 것을 이미 알아볼
것입니다. §9.3이 둘을 같은 입력으로 돌립니다.

> 💡 **쉬운 설명 — lookahead 상태 기계를 손으로 돌려보기**
> buffer가 `"$29"`인 상태에서 `.`이 들어옵니다. 마지막 문자가 문장 종결 구두점이므로 코드는 NLTK를
> 부르지 않고 `_needs_lookahead = True`만 켭니다. 다음 문자가 `" "`(공백)이면 여전히 기다립니다 —
> 공백은 `$29. Next`에도 `$29.99`에도 나올 수 있어서 정보가 없습니다. 다음 문자가 `9`이면 buffer는
> `"$29. 9"`가 되고 NLTK를 부르는데, Punkt가 문장 경계를 인정하지 않아 계속 쌓습니다. 다음 문자가
> `N`이면 buffer `"$29. N"`에서 Punkt가 경계를 확인하고 `$29.`를 emit합니다. **핵심**: 판단이 항상
> 한 token 늦게 일어나고, 그 한 token이 §4.1의 200–300 ms입니다.

### 4.3 end-of-turn flush가 loop를 닫는다

LLM turn의 마지막 문장에는 그것을 확인해 줄 후속 token이 없습니다. 그 경우는 frame 수준에서 명시적으로
처리됩니다:

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

그래서 lookahead 세금은 한 turn의 문장 1..n−1에 대해 지불되고 문장 n에 대해서는 **지불되지 않습니다**.
Lina의 전형적인 2–4문장 응답이면 turn당 1–3 token의 추가 latency이고, 그중 첫 가청 sample 이전에 오는
것은 문장 1에서 발생하는 경우뿐인데 — 그런데 그건 발생하지 *않는* 게 아닙니다. 문장 1의 경계 확인 그
자체가 lookahead이기 때문입니다. 다시 읽으십시오: lookahead 세금은 **당신이 신경 쓰는 바로 그 숫자**,
TTFA에 정확히 떨어집니다. 문장 1을 지연시키기 때문입니다.

### 4.4 `match_endofsentence`와 한국어 fallback

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

한국어가 저 주석에 이름으로 나오고, 그것이 떨어지는 집합은 뺄셈으로 유도됩니다:

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

한국어와 관련된 full-width member들은 `:73-116`의 `SENTENCE_ENDING_PUNCTUATION`에서 오고, 거기서
*"East Asian punctuation (Chinese (Traditional & Simplified), Japanese, Korean)"* 라는 주석 아래 묶여
있습니다: `。？！；．｡`.

**함정.** 저 fallback은 ASCII period 경로가 실패할 때만 발동합니다. ASCII `.` `?` `!`로 쓰인 한국어 —
사실상 모든 한국어 LLM이 쓰는 방식이고 Lina의 LLM이 쓸 방식 — 는 fallback이 아니라 NLTK의 영어 Punkt
model을 통과합니다. 그러니 docstring의 문장("used as a fallback when NLTK doesn't support the language")은
당신의 한국어 bot이 대체로 타지 *않을* 경로를 기술합니다. 그러면 명백한 질문이 나옵니다: 영어 Punkt가
한국어에서 실제로 작동하는가?

### 4.5 가정이 아니라 측정: 진짜 aggregator를 통과한 여섯 개의 한국어 입력

repo 자신의 한국어 assertion들(`tests/test_utils_string.py`)은 full-width 구두점만 다룹니다:

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

거기 어디에도 ASCII 구두점을 쓴 한국어를 다루는 것은 없는데, 그것이 정작 중요한 경우입니다. 그래서 제가
돌려봤습니다. 아래 script는 `match_endofsentence`와
`SimpleTextAggregator._check_sentence_with_lookahead`를 그대로 재구현하고(같은 NLTK 호출, 같은 lookahead
상태 기계) Lina의 출력처럼 생긴 token stream을 먹입니다. 직접 돌려볼 수 있습니다. `punkt_tab`이 있는
`nltk`만 있으면 됩니다.

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

결과, stream당 한 줄씩, 어떤 token이 emit을 촉발했는지 표시:

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

거기서 네 가지가 떨어져 나오고, 전부 의견이 아니라 이 코드에 대한 사실입니다:

1. **lookahead 비용이 눈에 보입니다.** 모든 경우에서, emit은 문장이 완성된 *다음* token에 발화합니다 —
   token 8이 아니라 10, token 2가 아니라 3. 그것이 200–300 ms입니다.
2. **`1.5`, `gpt-4.1`, `1,000`은 쪼개지지 않습니다.** Pipecat에는 명시적 숫자/식별자 guard가 없고, 이
   입력들에 대해서는 필요하지도 않습니다: NLTK Punkt의 abbreviation/number model에 비공백 lookahead가
   더해져 이미 절단을 거부합니다. `1.5`가 보호되는 이유는 `.` 뒤에 비공백 문자가 오기 전까지
   `match_endofsentence`가 아예 호출되지 않고, 그때쯤이면 buffer가 `1.5`로 읽히며 Punkt가 구두점으로
   끝나지 않는 단일 문장을 반환하기 때문입니다 → 0.
3. **full-width 경로는 작동하고 token 하나만큼 더 쌉니다** — `안녕하세요。`는 token 3이 아니라 2에서
   emit됩니다. fallback scan은 Punkt의 동의를 필요로 하지 않기 때문입니다.
4. **영어 Punkt는 이 입력들에서 ASCII-구두점 한국어를 처리해 냅니다.** 그것은 보장이 아닙니다. Punkt는
   영어로 훈련된 unsupervised model이고, 그 abbreviation 목록에는 한국어 지식이 없습니다. 저는 억지스러운
   반례를 하나 찾았습니다 — `match_endofsentence("월 보험료는 29. 000원")`이 10을 반환하며 `29.` 뒤에서
   자릅니다 — 하지만 그것을 만들려면 숫자-공백-숫자 pattern이 필요합니다. "Punkt는 한국어에서 작동한다"를
   증명된 것이 아니라 *현실적인 입력에서 반증되지 않았다*로 취급하십시오.

→ **[tts-streaming.html](figures/tts-streaming.html)**의 두 번째 panel. 같은 한국어 문장을 두 경로에
나란히 통과시키고 절단점을 표시합니다. §9를 읽은 **다음에** 쓰십시오, 그 전이 아니라 — 그 panel은 두
algorithm의 비교이고, 각각이 무엇을 하는지 알아야만 의미가 생깁니다.

### 4.6 미끼: `processors/aggregators/sentence.py`

이 repo에서 "sentence aggregator"를 검색하면 위의 어떤 것과도 상관없는 파일을 찾게 됩니다.

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

63줄, 독립 `FrameProcessor`, lookahead 없음, `AggregationType` 없음, `PatternMatch` 처리 없음, 그리고 —
결정적으로 — **`TTSService`는 이것을 절대 건드리지 않습니다.** grep이 확인해 줍니다: `tts_service.py`는
`pipecat.utils.text.simple_text_aggregator`에서 `SimpleTextAggregator`를 import하고(`:59`),
`processors.aggregators.sentence`에서는 아무것도 import하지 않습니다.

그 `process_frame`은 구두점이 나타나는 순간 `match_endofsentence`를 호출합니다:

**`src/pipecat/processors/aggregators/sentence.py:54-58`**
```python
        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
            if match_endofsentence(self._aggregation):
                await self.push_frame(TextFrame(self._aggregation))
                self._aggregation = ""
```

lookahead state가 없습니다. 즉 그것은 `$29.`를 가격 중간에서 쪼갤 *것이고*, Punkt가 속으면 `1.5`도 쪼갤
*것입니다*. 어떤 LLM이 "chunking을 개선한다"며 TTS 앞에 `SentenceAggregator`를 끼워 넣는 Pipecat 코드를
써 준다면, 그것은 좋은 aggregator 앞에 두 번째의 더 나쁜 aggregator를 하나 건네준 것입니다. 지우십시오.

### 4.7 aggregation delay를 보이게 만드는 metric

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

응답의 첫 `TextFrame`에서 시작하고(`:771`), 첫 비-TOKEN aggregate에서 멈춥니다(`:1099-1101`). LLM turn당
하나의 측정이고, 정확히 "first token → first complete sentence"를 덮습니다. TOKEN mode에서는 통째로
skip되는데 그건 옳습니다 — 측정할 aggregation이 없으니까요 — 하지만 그것은 동시에 **이 metric으로는
SENTENCE 대 TOKEN을 A/B할 수 없다**는 뜻이기도 합니다. mode를 바꾸면 숫자가 0이 되는 게 아니라 사라집니다.
비교하려면 TTFA를 봐야 합니다.

---

## 5. Ordering: 하나의 FIFO, 그리고 서로 같은 cursor가 아닌 두 cursor

이 section이 "reordering buffer"라는 주장을 구체화합니다. 건너뛰면 §6이 말이 되지 않습니다.

### 5.1 serialization queue는 세 종류의 것을 담는다

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

그리고 단일 consumer:

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

loop 안의 blocking `await self._handle_audio_context(context_id)`에 주목하십시오. **하나**의 consumer
task가, 다음 것을 건드리기 전에 **하나**의 context를 완료까지 drain합니다. 그것이 ordering 보장이고,
그래서 두 `TTSSpeakFrame` 사이에 push한 `FooFrame`이 첫 발화의 audio를 추월할 수 없는 것입니다 —
`process_frame`의 `else` branch가 그것을 같은 FIFO로 라우팅합니다:

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

`SystemFrame`은 제외됩니다 — 그것이 [[ch-04/read]] §4가 한 section을 통째로 쓴 out-of-band priority
path이고, 여기서는 `if` 안의 예외 절로 등장합니다. `InterruptionFrame`은 `SystemFrame`이고 3초짜리
한국어 audio 뒤에 줄 서지 않습니다.

### 5.2 두 개의 cursor, 서로 다른 시점에 clear된다

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

synthesis 쪽 cursor와 playback 쪽 cursor. 둘은 audio tail 전체 기간 동안 갈라집니다: LLM은 끝났고,
`_turn_context_id`는 `None`인데, pipe 안에는 아직 2.8초 분량의 한국어가 있고 `_playing_context_id`는
여전히 set되어 있습니다.

그 divergence가 Pipecat이 "고객이 실제로 무엇을 들었는가?"에 답할 수 있는 이유이고 — [[ch-03/read]]
§7.2는 realtime_voice가 같은 질문에 다른 mechanism으로 답하는 것을 보여주었습니다
(`AudioTextPlayoutLedger`의 `_next_sample` / `_played_sample` 쌍, [[rtv-vad-chunking]] 기준). 두 system
모두 cursor가 둘입니다. 단위가 다릅니다: 여기서는 context ID, 저기서는 sample count.

> 💡 **쉬운 설명 — 왜 cursor 하나로 합치면 안 되나요?**
> LLM이 문장을 다 만든 시점과 고객이 그 문장을 다 들은 시점은 몇 초 차이가 납니다. 하나로 합치면
> "합성 끝"에서 cursor가 null이 되고, 그 순간 `has_active_audio_context()`는 False를 반환합니다.
> 그러면 그 2.8초 동안 들어온 barge-in은 "봇이 말하고 있지 않았다"로 처리되어 아무 context도 취소하지
> 않습니다. 두 cursor는 **생산 시간축**과 **재생 시간축**이 다른 시스템에서는 서로 다른 물리량이고,
> §6.6의 word-paced truncation 전체가 이 분리 위에 서 있습니다.

### 5.3 context 재사용: LLM turn당 하나의 context

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

`reuse_context_id_within_turn`은 기본값이 `True`입니다(`:189`). 그래서 한 LLM turn의 모든 문장이
**하나의** audio context에 떨어지고, 그것이 turn 내부의 문장 경계를 가로질러 word-timestamp PTS
baseline이 연속적으로 유지되게 해 주는 것입니다(§6.3).

`_refresh_audio_context`는 idle timeout에 대한 keepalive입니다:

**`src/pipecat/services/tts_service.py:1609-1612`**
```python
    def _refresh_audio_context(self, context_id: str):
        """Signal that the audio context is still in use, resetting the timeout."""
        if self.audio_context_available(context_id):
            self._audio_contexts[context_id].put_nowait(TTSService._CONTEXT_KEEPALIVE)
```

### 5.4 `_handle_audio_context` — 모든 것이 도착하는 loop

이 파일에서 가장 밀도 높은 70줄입니다. 통째로 읽을 가치가 있습니다.

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

그 하나의 queue 안에서 다섯 가지 일이 **playback 순서로** 뒤섞여 일어납니다: audio frame, word-timestamp
entry, started/stopped bracket, keepalive sentinel, 그리고 `None` end-of-context marker. 그 뒤섞임이
요점입니다. audio chunk 세 개 뒤에 queue된 word timestamp는 provider의 socket이 그것을 언제 전달했든
상관없이 그 세 audio chunk 다음에 emit됩니다.

timeout branch가 silent-provider 경로입니다:

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

`self._stop_frame_timeout_s`의 기본값은 **3.0초**입니다(`:158`). 3초 동안 아무것도 만들지 않은 context는
끝난 것으로 선언됩니다. 그 상수는 telephony bot을 위해 바꾸는 것을 고려해야 할, 이 파일의 두 숫자 중
하나이고, §11이 왜인지 말합니다.

### 5.5 `TTSSpeakFrame` — out-of-band 발화

`TTSSpeakFrame`은 LLM에서 나오지 않은 것을 bot이 말하게 만드는 방법입니다 — filler, hold message,
DTMF prompt.

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

그 처리는 미묘한 일을 합니다 — 그 발화가 새 UUID를 받고 LLM turn의 context에 합류하지 않도록
*turn cursor를 일시적으로 null로 만듭니다*:

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

...그리고 끝에서 저장해 둔 cursor를 복원합니다(`:874-876`). 이것이 Lina의 추임새 — tool call이 도는 동안
"네, 잠시만요" — 에 쓸 mechanism이고, filler가 대화 history를 오염시키지 않도록 `append_to_context=False`
를 함께 씁니다.

### 5.6 interruption이 clear하는 것

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

열세 조각의 state가 reset되고, live context마다 hook(`on_audio_context_interrupted`)이 하나씩 발화되어
websocket subclass가 provider 쪽 cancel을 보낼 수 있게 하며, consumer task가 파괴되고 재생성됩니다.
`UninterruptibleFrame`을 보존하는 `_serialization_queue.reset()`은 [[ch-04/read]] §4.3이 processor
수준에서 기술한 것과 같은 mechanism을, 이 service의 private queue에 적용한 것입니다.

[[ch-08/read]]가 이 method *주변*에서 일어나는 일을 소유합니다. 지금은 위의 `on_audio_context_interrupted`
docstring 마지막 줄만 기억하십시오:

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

그것이 custom 한국어 provider를 위한 barge-in extension point입니다.

---

## 6. Word timestamp: 실제로 존재하는 것

### 6.1 word frame은 없다. 평이하게 진술하십시오.

commit `0cbf9c5b`에서 `src/`를 `WordBoundaryFrame`이나 `TTSWordFrame`으로 grep하면 **아무것도** 안
나옵니다. `frames.py`는 이름에 "Frame"이 들어간 class를 132개 선언하고, 그중 어느 것도 word-boundary
frame이 아닙니다.

둘 중 하나를 언급하는 blog post나 LLM이 생성한 migration plan을 읽었다면, 그건 hallucination입니다.
word 수준 정보는 정확히 두 가지가 실어 나릅니다.

### 6.2 첫 번째: `pts`를 가진 `TTSTextFrame`

**`src/pipecat/frames/frames.py:416-421`**
```python
@dataclass
class TTSTextFrame(AggregatedTextFrame):
    """Text frame generated by Text-to-Speech services."""

    pass
```

다섯 줄. machinery는 전부 상속됩니다: `AggregatedTextFrame` / `TextFrame`에서 `text`, `aggregated_by`,
`context_id`, `append_to_context`, `includes_inter_frame_spaces`를, 그리고 root `Frame`에서 `pts`를:

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

**Presentation timestamp, 나노초 단위, 보편적 base frame 위에.** 그것이 design 결정입니다. word-boundary
type을 발명하는 대신, Pipecat은 모든 frame이 이미 가지고 있는 field를 재사용하고, word token 하나당
`TTSTextFrame` 하나를 emit하되 그 단어가 언제 *발화될지*를 stamp합니다.

frame은 여기서 만들어지고, 전부 stamp됩니다:

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

`aggregated_by=AggregationType.WORD` — §3의 세 번째 enum member이고, 여기가 그것의 유일한 producer입니다.

### 6.3 두 번째: `AggregatedTextProgressFrame`

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

`accumulated_text` / `remaining_text`는 live transcript UI를 위해 당신이 원하는 분할이고 — Lina에게 더
중요하게는 — 아무도 계산할 필요 없이 단어마다 emit되는 **기성품 "고객이 지금까지 무엇을 들었는가"**
payload입니다.

품질이 아니라 shape를, [[ch-03/read]] §7.2에 기록된 [[rtv-vad-chunking]]의 realtime_voice 답과 비교해
보십시오: `AudioTextPlayoutLedger.audible_text()`는 sample-count 비율로부터
`text[: int(len(text) * ratio)]`를 계산합니다. 둘 다 발화된 text의 prefix를 만들어냅니다. 하나는 provider가
emit한 단어별 timestamp에서 유도하고, 다른 하나는 sample counter 위의 선형 문자-당-sample interpolation에서
유도합니다. 다른 입력, 다른 해상도, 같은 출력 type. 각각이 TTS에게 **무엇을 요구하는지**가 붙들고 있을
지점입니다: 하나는 timestamp를 emit하는 provider가 필요하고, 다른 하나는 byte count만 있으면 됩니다.

### 6.4 service를 절대 떠나지 않는 내부 entry

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

밑줄 prefix, 절대 push되지 않음. 그 일 전체는 **queue에 넣을 수 있는 것**이 되는 것입니다 — audio frame과
같은 `asyncio.Queue`에 앉아 §5.4의 loop가 그것을 playback 순서로 꺼내도록:

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

가운데 branch에 주목하십시오: 더 이상 존재하지 않는 context에 대해 도착한 word timestamp는 **trace log와
함께 버려집니다**. 그것이 late-delivery 경우입니다 — provider의 socket이 2초 전에 당신이 취소한 context의
timestamp를 배달하는 것. 조용하고, 올바르며, trace logging을 켜지 않는 한 보이지 않습니다.

### 6.5 PTS baseline

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

context당 한 번, 첫 audio chunk에서 호출됩니다(§1.3). provider는 *자기* 응답 시작으로부터의 초를 줍니다.
Pipecat은 이 baseline을 더해서 절대적인 pipeline 나노초로 변환합니다:

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

caching branch는 첫 audio chunk *이전에* 도착하는 timestamp를 처리합니다 — 일부 provider는 alignment를
먼저 보냅니다. 그것들은 붙들려 있다가 baseline이 생기면 flush됩니다.

> 💡 **쉬운 설명 — 상대 시간을 절대 시간으로 바꾸는 산수**
> provider는 "이 단어는 내 응답 시작 후 1.2초"라고 말합니다. 그런데 pipeline은 "지금 몇 시인가"를
> 나노초 단위 clock으로 셉니다. 첫 audio chunk가 도착한 pipeline 시각이 `T`라면,
> `pts = T + 1.2초를 나노초로` 가 그 단어의 재생 예정 시각입니다. `T`를 첫 audio chunk에 anchor하는
> 것이 핵심입니다 — 요청을 보낸 시각이 아니라 **소리가 나오기 시작한 시각**이어야 재생 시각과
> 맞아떨어지기 때문입니다. 그리고 `_word_last_pts`가 현재 시각보다 앞서 있으면 그것을 쓰는데,
> context가 겹칠 때 시간이 뒤로 가지 않게 하기 위해서입니다.

### 6.6 ch-08이 필요로 하는 hook: output transport의 clock queue

`pts`를 metadata 이상으로 만드는 것이 여기 있습니다. `transport.output()`은 frame을 presentation
timestamp로 정렬하고 **각각이 due될 때까지 잠듭니다**.

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

`pts`를 key로 하는 `asyncio.PriorityQueue`, 그리고 presentation time까지의 진짜 `asyncio.sleep`.

귀결을 조심스럽게 읽으십시오. 이 chapter의 payoff 전체이자 다음 chapter의 전제이기 때문입니다:

> word-timestamp가 찍힌 `TTSTextFrame`은 **`transport.output()`의 downstream에 있는** processor에,
> 그것이 synthesise된 순간이 아니라 그 단어가 재생되는 순간에 도달합니다. canonical pipeline의
> position 7 — assistant context aggregator([[canonical-voice-bot]]) — 는 `transport.output()`의
> downstream입니다. 그래서 assistant context는 **고객이 듣는 대로의** 단어들로 지어집니다.
> 4.2초짜리 문장의 1.4초 지점에 도착한 barge-in은, 구성상, `pts <= 1.4 s`인 단어들만 전달했습니다.
> context를 자르는 것은 계산이 아닙니다. 그냥: 멈추는 것입니다.

그것이 [[ch-08/read]]가 쓰는 것입니다. 당신의 TTS provider가 word timestamp를 emit할 때에만 가용하고 —
한국어에서 그것은 §7.4의 여섯 service를 뜻합니다.

> 💡 **쉬운 설명 — "자르는 게 계산이 아니라 그냥 멈추는 것"의 의미**
> 순진한 구현은 이렇게 합니다: 문장 전체를 assistant context에 미리 넣어 두고, barge-in이 오면
> "1.4초까지면 몇 글자쯤 말했을까"를 역산해서 뒤를 지웁니다. 역산은 추정이고 틀립니다.
> Pipecat 구현은 애초에 문장 전체를 넣은 적이 없습니다 — 단어 frame이 재생 시각에 하나씩 도착해서
> 하나씩 append됩니다. barge-in 시점에 aggregator에 들어가 있는 것은 이미 재생된 단어들뿐입니다.
> 지울 것이 없습니다. **자료구조를 시간축에 정렬시켜 두어서 truncation 문제를 없애 버린 것**입니다.

코드가 우회해야 하는 미묘한 점이 하나 있습니다: PTS가 *없는* frame은 clock queue 대신 audio queue를 타고,
따라서 clock queue에 들어간 word frame을 추월할 수 있습니다. frame을 clock queue로 강제로 밀어 넣기 위해
순전히 그 목적으로 PTS를 stamp하는 곳이 두 군데 있습니다:

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

여기서 `pts`는 단순한 timestamp가 아니라 **queue selector**로 쓰이고 있습니다. ordering bug를 debug하다가
pipeline diagram상 불가능하다고 하는 순서로 frame이 도착하는 것을 발견할 때 알아 둘 가치가 있습니다.

→ **[tts-streaming.html](figures/tts-streaming.html)**의 세 번째 panel. playhead를 scrub하면서 word
frame이 자기 `pts`에서 clock queue로부터 release되는 것을 보십시오. 그런 다음 문장 중간에 barge-in marker를
떨어뜨리고 어떤 단어들이 이미 전달되었는지 읽으십시오 — 그 집합이 정확히 assistant context가 담게 될
것입니다. ch-08을 시작하기 전에 하십시오.

### 6.7 truncation을 이미 증명하는 한국어 test

`AggregatedFrameSequencer.force_complete`는 provider가 timestamp event를 누락할 때 돌아가는 것이고,
repo는 그것을 한국어 문장으로 test합니다:

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

한국어 어절 하나가 발화되었고, machinery는 `여러분의 AI 어시스턴트입니다.`를 발화되지 않은 나머지로 —
어절 spacing을 재구성해서 — 올바르게 식별합니다. 그것이 frame layer에서 end to end로 작동하는 한국어
어절 수준 분할이고, §0.2가 정정한 outline 주장에 대한 구체적 반증입니다.

---

## 7. 한국어, provider 하나씩

아래 survey는 [[tts-korean-providers]]가 지도로 그린 땅을 다시 걷되, 그것이 이름 부르는 모든 파일을
직접 엽니다. excerpt와 source가 불일치하는 곳에서는 source를 인용하고 불일치를 표시합니다.

### 7.1 grep, 그리고 그 정확한 출력

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

열한 개의 provider directory. Azure가 열두 번째이고 여기 안 나오는 이유는 그 map이 공유 module에 살기
때문입니다:

**`src/pipecat/services/azure/common.py:199-201`**
```python
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
```

**열두 개의 provider module이 한국어를 선언합니다.** "service"가 아니라 "module"이라고 말하십시오. 각
module이 service class를 하나에서 셋까지 ship하고(websocket 하나, HTTP 하나, 때로는 세 번째), 그것들이
map을 공유하기 때문입니다. class 수준의 전체 table, class 하나씩 검증한 것:

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

`MiniMaxHttpTTSService`가 이상한 놈입니다 — 그 "code"는 영어 단어 `"Korean"`이고, language parameter가
아니라 `language_boost` field에 먹입니다. locale이 아예 아닙니다.

### 7.2 mapping되지 않은 언어는 raise하지 않는다. warning을 낸다.

**`src/pipecat/transcriptions/language.py:583-630`** (fallback 꼬리 부분)
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

exception 없음. `logger.warning` 하나, 그리고 raw code가 그대로 wire를 타고 나갑니다.

그것이 이 section의 나머지가 다루는 design 결정입니다. Pipecat은 language map을 capability contract가
아니라 *verification list*로 취급합니다 — docstring은 "unsupported"가 아니라 *"not verified"*라고
말합니다. framework의 입장은, 어떤 provider가 어떤 언어를 지원하는지를 자기가 당신보다 더 잘 알아서는
안 된다는 것이고, 그래서 당신의 요청을 통과시키면서 확인할 수 없었다고 알려줍니다.

> 💡 **쉬운 설명 — verification list vs capability contract**
> capability contract라면 `Language.KO`가 map에 없을 때 `raise`해야 합니다: "이 provider는 한국어를
> 지원하지 않는다." verification list는 다르게 말합니다: "우리는 이 조합을 확인해 본 적이 없다.
> 그래도 원하면 보내주겠다." 후자를 고른 이유는 실무적입니다 — provider는 Pipecat 릴리스보다 훨씬
> 자주 언어를 추가하고, 하드 실패는 어제 지원 시작된 언어를 쓰려는 사용자를 막아버립니다.
> **대가**: 지원하지 않는 조합도 부팅에 성공합니다. §7.3의 Rime 시나리오가 그 대가를 통째로
> 보여줍니다.

### 7.3 Rime 함정, 전부

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

다섯 언어, 세 글자 코드, 한국어 없음. 그리고 `use_base_code=False`라서 `Language.KO`는 문자 그대로의
string `"ko"`가 됩니다 — `eng/ger/fra/spa/hin`만 이해하는 API에 보내지는 두 글자 코드.

failure 전체를 추적하십시오:

1. 당신은 `RimeTTSService(settings=..., language=Language.KO)`라고 씁니다. Rime이 shortlist에 있는 이유는
   word timestamp를 emit하고(`add_word_timestamps` caller 목록에 있습니다) 저지연을 마케팅하기 때문입니다.
2. `TTSService.__init__`이 생성 시점에 변환합니다:

   **`src/pipecat/services/tts_service.py:273-276`**
   ```python
        if isinstance(self._settings.language, Language):
            converted = self.language_to_service_language(self._settings.language)
            if converted is not None:
                self._settings.language = converted
   ```
3. `resolve_language`가 `Language ko not verified. Using 'ko'.`를 log합니다 — WARNING 한 줄, 시작
   시점에, 당신의 app이 뿜는 다른 몇 개인지 모를 줄들 사이에서.
4. bot이 시작됩니다. pipeline이 지어집니다. 아무것도 raise하지 않습니다. `StartFrame`이 전파됩니다.
   모든 health check가 통과합니다.
5. 고객이 여보세요라고 말합니다. LLM이 한국어를 만들어냅니다. `run_tts`가 그것을 `"ko"`로 보냅니다.
   Rime은 쓸 만한 것을 아무것도 반환하지 않습니다.
6. 세 context 뒤 — §7.8 — service가 자기 자신을 write off합니다.

step 6까지 모든 것이 건강해 보입니다. 그것이 이 failure의 모양입니다: **한국어가 아닌 provider를 가리키는
한국어 bot은 문제없이 설정된다.**

같은 모양이 §7.5의 모든 service에 적용됩니다.

### 7.4 교집합은 여섯이고, 그 숫자가 정확히 뜻하는 것은 이렇습니다

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

(더해서 `src/pipecat/services/elevenlabs/dialogue/tts.py:378`, ElevenLabs module 내부의 열세 번째 call
site.)

한국어를 mapping하는 열두 module과 timestamp를 emit하는 열두 module을 교집합:

```
Korean:      aws azure camb cartesia elevenlabs google inworld lmnt minimax soniox xai xtts
Timestamps:  azure cartesia elevenlabs gradium hume inworld resembleai rime smallest soniox speechify xai
             ────────────────────────────────────────────────────────────────────────────────
Intersection: azure  cartesia  elevenlabs  inworld  soniox  xai
```

**여섯** — [[tts-korean-providers]]와 같은 여섯 개 이름이지만, 믿고 가져온 것이 아니라 grep에서 재생산한
것입니다. 이제 provenance를, 제가 할 수 있는 한 정직하게 진술합니다:

> 이 숫자는 **이 source tree에 대한 두 개의 grep**의 교집합입니다: language map에 `Language.KO` entry가
> 있는 service, 그리고 `add_word_timestamps`를 호출하는 service. 그것은 **코드가 무엇을 선언하는지**에
> 대한 주장이지, 어떤 출력을 들어보고 검증한 주장이 아닙니다.
>
> 구체적으로, 여기 있는 어떤 것도 다음을 확립하지 않습니다:
> - 여섯 중 어느 것이든 *좋은* 한국어를, 아니 알아들을 수 있는 한국어라도 synthesise한다는 것;
> - 그중 어느 것이 emit하는 timestamp가 정확하다는 것, 또는 긴 발화에서 drift하지 않는다는 것;
> - 당신이 실제로 쓸 계정에 한국어 voice ID가 존재한다는 것.
>
> repo의 어떤 test도 audio를 듣지 않습니다. `pipecat.evals` (`src/pipecat/evals/`)는 LLM judge로 실제
> bot을 end to end로 구동할 수 있고 `scripts/release-evals/`가 manifest를 ship합니다 — 그런데 거기에
> 한국어 scenario는 없습니다. 존재하는 한국어 test들(§0.2, §6.7)은 합성 token list에 대한 frame 수준
> unit test입니다.
>
> 이 여섯 중 둘이 한국어를 못해도 이 repo는 알지 못할 것입니다.

그러니: 여섯 개를 **test할 shortlist**로 쓰십시오, 여섯 개의 작동하는 선택지 목록이 아니라. 그 구별이
2주짜리 integration과 2개월짜리 integration의 차이입니다.

여섯 중 하나에는 gate가 하나 더 있습니다. xAI는 당신이 요청할 때만 timestamp를 emit합니다:

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

기본값은 `NOT_GIVEN`입니다. xAI를 고르고 `with_timestamps=True` 설정을 잊으면, 6-service 목록에는 올라
있으면서 다른 여섯이 아닌 것처럼 행동하는 service를 얻습니다.

### 7.5 한국어를 지원하지 **않는** service들 — 가정이 아니라 확인한 것

이들 전부 직접 열어봤습니다:

- **Rime** (`rime/tts.py:46`) — 다섯 언어, 한국어 없음. §7.3.
- **Neuphonic** (`neuphonic/tts.py:40`) — `de, en, es, nl, ar, fr, pt, ru, hi, zh`. 한국어 없음.
- **Kokoro** (`kokoro/tts.py:66`) — espeak-ng voice 이름을 mapping하며, map을 열어서 검증했습니다:
  `en-us, en-gb, es, fr-fr, fr-be, fr-ca, fr-ch, hi, it, ja, pt, pt-br, cmn, yue, zh, zh-CN, zh-HK,
  zh-TW`. **한국어 없음**, 그리고 docstring은 그래도 보냈을 때 무슨 일이 생기는지에 대해 노골적입니다:

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
  *"an unsupported name fails at synthesis time"* — 즉 시작 시점이 아니라 실통화 14:03에 실패합니다.
- **Piper** (`piper/tts.py:44`, `:207`) — `language_to_service_language` override 자체가 없습니다. 두
  class 모두 `default_settings = self.Settings(model=None, voice=None, language=None)`을 설정합니다
  (`:91`, `:243`). 언어는 당신이 load한 `.onnx` voice file이 무엇이냐에 달렸고, Pipecat은 여기서 한국어
  지식이 전혀 없어서 아무것도 경고해 줄 수 없습니다.
- **OpenAI** (`openai/tts.py:81`) — language parameter가 존재하지 않습니다.
- **Fish, Deepgram, Gradium, Hume, ResembleAI** — default settings에서 `language=None`, map 없음.

### 7.6 이 tree에는 유지보수되는 self-hosted 한국어 TTS가 없다

한국어 map을 가진 유일한 local service는 deprecated입니다:

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

*"No replacement."* 그리고 지목된 두 대체재는 정확히 한국어를 mapping하지 않는 두 local service입니다
(§7.5).

여기에 더해: **한국어 native provider가 아예 통합되어 있지 않습니다.** `src/pipecat/services/` 아래에
`typecast/`도, `supertone/`도, `clova/`나 `naver/` directory도 존재하지 않습니다.

두 사실 모두 선택이 아니라 gap입니다. Lina에 on-prem이나 data-residency 요구가 있다면 — 그리고 한국
보험 tele-sales에서 그것은 가정이 아닙니다 — 이 tree에서의 선택지는: 한국 벤더의 API에 대고 custom
`TTSService` subclass를 쓰거나, 그 배포에서는 Pipecat의 TTS layer를 쓰지 않는 것입니다. 현재 tree에서
그것을 커버하는 것은 아무것도 없습니다. 그것은 build item이고, §11이 그것을 하나로 열거합니다.

### 7.7 물게 될 config 세부사항들, 한곳에 모아서

**Azure는 locale이 맞는 voice를 필요로 합니다.** 기본값은 두 축 모두 영어입니다:

**`src/pipecat/services/azure/tts.py:331-343`**
```python
        default_settings = self.Settings(
            model=None,
            voice="en-US-SaraNeural",
            language="en-US",
            emphasis=None,
            force_locale=False,
```

`voice="en-US-SaraNeural"`를 그대로 둔 채 `language=Language.KO`를 설정하는 것이 정확히 §7.8의 guard가
잡으려고 존재하는 unknown-voice-for-this-language 경우입니다. Azure는 또한 문장 경계 pause를 하드코딩하고
locale wrapper를 제공합니다:

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

모든 문장 경계에 `20ms`, settings로는 설정 불가능합니다.

**ElevenLabs는 model이 그 언어를 커버하지 않으면 당신의 language code를 조용히 버립니다.**

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

`"ko"`는 `ELEVENLABS_V2_5_LANGUAGES`(32개 code, `tts_base.py:39`)와 `ELEVENLABS_V3_LANGUAGES`(74개 code,
`:78`) 양쪽에 다 있습니다 — 둘 다 제가 세어 봤습니다. 기본 model은 `"eleven_flash_v2_5"`이고, map에
있습니다. 하지만 `ELEVENLABS_MODEL_LANGUAGES`에는 key가 넷뿐이라(`:165-170`), `eleven_multilingual_v2`를
고르면 — 한국어 bot에게 그럴듯한 선택인데 — 당신의 language code가 warning과 함께 통째로 버려지고 model
자신의 auto-detection으로 fallback합니다.

**Soniox는 연결당 concurrency에 상한이 있습니다.**

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

연결당 다섯 stream. 단일 통화짜리 Lina worker에게는 상관없지만, process-per-N-calls 배포 형태
([[ch-04/read]] §13)에서는 connection을 어떻게 pool할지에 대한 hard constraint입니다. 기본값도 영어입니다:
`model="tts-rt-v2", voice="Bryce"` (`:189-193`).

**Inworld는 구두점과 공백을 별도 token으로 emit합니다.** flag로 처리합니다:

**`src/pipecat/services/inworld/tts.py:1084`**
```python
                    await self.add_word_timestamps(word_times, ctx_id, pre_merge_tokens=True)
```

이것이 다음을 돌립니다:

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

그 한국어 동작은 한 줄 생각할 가치가 있습니다. Python의 세부사항 하나에 달려 있기 때문입니다:

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

`str.isalnum()`은 한글 음절에 대해 True입니다 — 그것들은 Unicode letter이기 때문입니다. 그래서 한국어
token은 filter를 살아남고 진짜 구두점/공백 token만 merge됩니다. 이 함수는 영어를 위해 쓰였고 아무도
적어두지 않은 이유로 한국어에서 작동합니다. 이것을 언제든 바꾸게 되면 확인할 가치가 있습니다.

### 7.8 유일한 안전망: 세 개의 silent context

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

기본값 3 (`:168`). 구현:

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

이것은 잘 test되어 있습니다 — `tests/test_tts_zero_audio_contexts.py`에 여덟 case가 있고, 그 module
docstring이 failure mode를 정확히 진술합니다:

**`tests/test_tts_zero_audio_contexts.py:7-13`**
```
"""Tests for writing off a TTS service that stops producing audio.

A provider can accept every request and return no audio at all — an unknown
voice ID, say — without reporting an error. TTSService counts the contexts that
complete in silence and, past a configurable limit, reports itself unable to do
its job so the pipeline worker and any ServiceSwitcher can act on it.
"""
```

이제 §7.3의 Rime 시나리오에 대해 timeline을 맞춰 보십시오. `_stop_frame_timeout_s`는 3.0초이고(§5.4),
silent context는 timeout이 나야만 끝난 것으로 선언됩니다. 따라서 세 개의 연속된 silent context는
무언가가 보고되기까지 **실통화에서 9초 규모의 dead air**이고 — 그러고 나서야 `ProcessorUnusablePolicy`
([[ch-04/read]])가 행동할 기회를 얻습니다.

Lina에게, 방금 전화를 받은 고객에게 거는 cold outbound dial에서 9초의 침묵은 통화 전체입니다. 더 빠른
write-off를 원하면 `max_consecutive_zero_audio_contexts=1`로 설정하고, 정당하게 비어 있는 발화(예를
들어 function call만 있는 turn)가 그것을 발동시킨다는 것을 받아들여야 합니다. 빠르면서 동시에 안전한
중간 설정은 없습니다. §11이 입장을 정합니다.

---

## 8. 한국어 word grouping: 한국어가 그 바깥으로 떨어지는 branch

### 8.1 zh/ja test, 두 곳에서

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

`{"zh", "ja"}`. 한국어는 그 집합에 없고, 그 부재는 실수가 아니라 의도적입니다 — CJK branch가 존재하는
이유는 zh/ja에 단어 사이 공백이 없기 때문이고, 한국어에는 있습니다.

그 branch를 탔을 때 하는 일:

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

zh/ja에서는 timestamp message 하나 전체가 **한** 단어로 접힙니다 — timestamp 하나를 가진 발화 chunk
전체이고, 정확히 예상할 만한 해상도 손실이며, 발화 중간 truncation을 거칠게 만듭니다. 한국어는 `else`
branch를 타고 token당 entry 하나를 얻어 완전한 해상도를 유지합니다.

`includes_inter_frame_spaces`가 함께 딸려 가는 flag입니다:

**`src/pipecat/services/cartesia/tts.py:522-526`**
```python
    def _word_timestamps_include_inter_frame_spaces(self) -> bool:
        """Whether timestamp text should be treated as carrying its own spacing."""
        current_language = assert_given(self._settings.language)
        return bool(current_language and self._is_chinese_or_japanese_language(current_language))
```

한국어에는 False → downstream concatenator가 연속된 word frame 사이에 공백을 삽입합니다. 그것이 §0.2의
test가 assert하는 동작입니다: `["저는"], ["여러분의"], ["AI", "어시스턴트입니다."]`가
`"저는 여러분의 AI 어시스턴트입니다."`로 재조립됩니다.

> 💡 **쉬운 설명 — `includes_inter_frame_spaces`가 왜 필요한가요?**
> word frame을 다시 하나의 문자열로 이어 붙일 때 "사이에 공백을 넣어야 하나?"를 결정해야 합니다.
> 영어/한국어는 단어 사이에 공백이 있으니 넣어야 하고(`False` → concatenator가 넣음), 중국어/일본어는
> 애초에 공백이 없으니 넣으면 안 됩니다(`True` → token 자체가 이미 필요한 spacing을 다 담고 있다고
> 간주). flag 이름이 헷갈리는데, "timestamp text가 자기 spacing을 이미 포함하고 있는가"라는 뜻입니다.
> True면 그대로 붙이고, False면 사이에 공백을 끼웁니다.

### 8.2 ElevenLabs는 공백 문자로 분할하고, 한국어에서 그것은 어절을 뜻한다

ElevenLabs는 character 수준 alignment를 보냅니다. Pipecat이 그것을 단어로 변환합니다:

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

`if char == " "` — 그것이 word-boundary 정의의 전부입니다. 영어에서는 단어입니다. 한국어에서는
**어절**입니다: `보험료는`, `한달에`, `이만구천원입니다.` 각각이 자기 `pts`를 가진 `TTSTextFrame` 하나가
됩니다.

그것이 truncation 단위를 character가 아니라 어절 위에 놓습니다. 한국어 assistant turn을 어절 중간에서
자르면 `보험료` 같은 깨진 단어로 읽히는 조각이 나오고, 어절 경계에서 자르면 `보험료는 한달에`가 나와서
중단된 문장으로 읽힙니다. 아무도 그렇다고 주석을 달아 두지 않았습니다 — 이것은 코드가 하는 일이고,
마침 한국어 정서법과 맞아떨어지는 것입니다.

zh/ja에서는 전혀 맞아떨어지지 않습니다: 공백이 없으니 발화 전체가 하나의 "단어"가 될 것이고, 그래서
그것들을 따로 처리하는 `includes_inter_frame_spaces` branch가 존재하는 것입니다.

### 8.3 무엇이 test되었고 무엇이 안 되었는가 — 마지막으로

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

저 table이 정직한 요약입니다. 한국어를 위한 frame plumbing은 실재하고 커버되어 있습니다. audio는 전적으로
당신의 몫입니다.

---

## 9. 이 layer에서 realtime_voice가 하는 일

mechanism만. §9.3은 순위가 아니라 측정입니다. 여기 나오는 realtime_voice 사실은 전부
[[rtv-vad-chunking]]에서 오고, private repo는 열지 않습니다.

### 9.1 `KoreanPhraseChunker` — 1 → 2 → tail schedule

283줄, Pipecat 대응물 없음. 그 constructor와 docstring, excerpt에 기록되었고 [[ch-03/read]] §7.1에서 이미
요약된 대로:

```
__init__(*, min_chars=12, max_chars=60, hard_max_chars=None,
         batch_max_chars=320, adaptive_batching=True)
# hard_max_chars=None resolves to min(batch_max_chars, max_chars * 2)   (L56-60)
```

> *"Adaptive mode emits the first complete sentence immediately, batches the next two complete
> sentences, then holds the remaining response as one final group until `flush`. `max_chars` is a
> soft latency target rather than an immediate cut point."* (docstring L28-34)

`_accept_adaptive`의 `_batch_phase` 0/1/2 (L115-149). §1이 준 frame인 TTFA 관점으로 읽으면, 이 schedule은
어느 문장의 latency가 노출되는지에 대한 직접적인 진술입니다:

- **Phase 0** (첫 문장, 혼자 emit): 문장 1만이 critical path 위에 있으므로, 가능한 한 작게 내보냅니다.
- **Phase 1** (다음 두 문장, 쌍으로 batch): 문장 1의 playback이 이미 돌고 있으므로, 문장 2와 3은 그
  *아래에서* 생산되며 대신 request overhead를 amortise합니다.
- **Phase 2** (경계 있는 tail, flush까지 한 group): 고객이 듣고 있으면 파편화를 멈춥니다.

Pipecat에는 phase counter가 없습니다. 그 aggregation mode는 service 수명 내내 상수입니다 — `SENTENCE`
아니면 `TOKEN` — 그리고 같은 경계 규칙이 문장 1에도 문장 9에도 적용됩니다.

### 9.2 세 개의 guard, 그리고 span 보존

excerpt에서, identifier는 그대로:

- `_is_safe_period` (L255)는 `1.5` 안의, `...` 안의, 또는 ASCII token 문자들 사이의 점에서 분할하기를
  거부합니다. 주석 L266-269: *"A dot between ASCII token characters belongs to a model name,
  hostname, abbreviation, or identifier rather than ending a Korean sentence."*
- `_is_numeric_separator` (L277)는 `_SOFT_END` comma 집합(`frozenset(",，;；:")`)으로부터 `1,000`을
  보호합니다.
- `_INTERNAL_TAG = re.compile(r"\[(?:interruption|system|tool|objection|customer|assistant)[^\]]*\]")`
  은 gateway control tag를 발화 text에서 걷어내되 `start_char` / `end_char`가 **source** span을 온전히
  유지합니다.

세 번째가 구조적으로 특이한 것이고, 그것이 무엇을 위한 것인지 정확히 짚을 가치가 있습니다. Boson의
gateway는 assistant text stream에 control tag를 주입합니다 — `[interruption]`, `[objection ...]`,
`[tool ...]` — 이것들은 발화되면 안 되지만 주소 지정 가능한 상태로 남아 있어야 합니다.
`AudioTextPlayoutLedger`가 *들린 문자*를 **원본** 문자열 위로 되매핑하기 때문입니다. strip-and-forget은
그 mapping을 깨뜨립니다. strip-and-keep-span은 깨뜨리지 않습니다.

"context에는 도달하지만 발화되지 않는 text"에 대한 Pipecat의 대응물은 완전히 다른 mechanism입니다:
`TextFrame`의 `skip_tts` flag(`frames.py:303-330`)에 `skip_aggregator_types`를 더한 것:

**`src/pipecat/services/tts_service.py:1159-1163`**
```python
        # Skip sending to TTS if the aggregation type is in the skip list. Simply
        # push the original frame downstream.
        if type in self._skip_aggregator_types:
            await self._push_frame_respecting_previous_aggregated_frame(src_frame, context_id)
            return
```

character 수준이 아니라 frame 수준입니다. Pipecat의 TTS path 어디에도 character-span 개념은 없습니다.

### 9.3 같은 입력, 두 경로 — 판정이 아니라 측정

`_is_safe_period`와 `_is_numeric_separator`는 `1.5`, `gpt-4.1`, `1,000`을 쪼개는 것에 대한 명시적
guard입니다. Pipecat에는 그런 guard가 없습니다. 명백한 질문은 Pipecat의 경로가 그 입력들로 실제로 무엇을
하느냐이고, §4.5가 그것들을 돌려서 답했습니다:

| Input (as a token stream, inside a Korean sentence) | Pipecat: does it split? | realtime_voice guard |
|---|---|---|
| `월 1.5만원입니다.` | no — emits the whole sentence | `_is_safe_period` |
| `저는 gpt-4.1을 씁니다.` | no — emits the whole sentence | `_is_safe_period` |
| `보험료는 1,000원입니다.` | no — emits the whole sentence | `_is_numeric_separator` |
| `월보험료는 29.99만원입니다.` | no — emits the whole sentence | `_is_safe_period` |
| `안녕하세요。고객님` | splits after `。`, 1 token earlier | `_STRONG_END` includes `。` |
| `월 보험료는 29. 000원` | **splits after `29.`** | `_is_safe_period` would refuse |

두 mechanism — 손으로 쓴 명시적 guard 목록 하나, 그리고 unsupervised 통계 tokenizer에 한 글자 lookahead를
더한 것 하나 — 이 여섯 입력 중 다섯에서 같은 출력에 도달하고 억지스러운 여섯 번째에서 갈립니다. 그것이
측정입니다. 거기서 무엇을 결론지을지는 [[ch-13/read]]의 일이고, 그것을 실제로 결판낼 입력은 이 두 table
중 어느 것도 아니라 당신이 직접 로깅한 한국어 transcript입니다.

측정이 아니면서 평이하게 진술할 수 있는 구조적 차이가 하나 있습니다: **realtime_voice의 guard는 들여다볼
수 있고 Pipecat의 것은 그렇지 않다.** `_is_safe_period`는 당신이 읽고 규칙을 추가할 수 있는 함수입니다.
NLTK의 Punkt는 한국어에서의 동작을 §4.5처럼 경험적으로만 특성화할 수 있는 훈련된 model입니다. 특정
pattern에서 분할이 절대 일어나지 않는다고 *보장*해야 한다면 — 증권번호, 주민등록번호 조각, 보험 상품
코드 — §3의 aggregator interface가 그 보장을 추가할 자리이지, model이 아닙니다.

### 9.4 `interrupt/fillers.py` — 이름이 오해를 부르고, 손대지 않은 채 살아남는다

40줄이고, `interrupt/` 아래 살면서도 *bot의* filler를 억제하지 않습니다.
[[boson-interrupt-subsystem]]에서:

- `(text, agent_status)`에 대한 `FillerCheck = Callable[[str, str], bool]`, 그리고
  `set_filler_check` / `get_filler_check` / `is_filler` / `clear_filler_check`.
- Docstring: *"The gateway has zero language knowledge — it only calls the registered callback."*
- 설정되지 않았을 때 `is_filler`는 `False`를 반환합니다.
- Lina의 실제 구현은 package 바깥,
  `agents/test-lina-gateway/layers/01-filler-filter/rules/korean_fillers.py`에 있습니다.

그것은 **user 쪽** filler 억제입니다: Lina가 말하는 동안 고객이 음, 어, 네네라고 하면, 이 callback이 그것이
진짜 barge-in이 아니라고 결정합니다. `InterruptionGate.allows`가 barge-in policy 앞에 그것을 참조합니다
([[boson-interrupt-subsystem]] `server/interruption.py:36`, step 3).

이 chapter의 어떤 것도 그것을 대체하지 않습니다. Pipecat의 TTS layer 어떤 것도 그것을 건드리지 않습니다.
그것은 §1–§8의 모든 것과 직교하고, 한국어 business logic이며, 그것에 대한 migration 답은 "그대로
포팅하라"입니다 — [[boson-interrupt-subsystem]]에 따라 custom `BaseUserTurnStartStrategy` subclass로,
그리고 그것은 이 chapter가 아니라 [[ch-06/read]]와 [[ch-08/read]]의 영역입니다.

여기서 언급하는 이유는 이름이 정확히 하나의 실수를 부르기 때문입니다 — "fillers"를 "TTS" 옆에서 읽고
그것이 bot의 추임새에 대한 것이라고 가정하는 것. 아닙니다. bot 쪽 대응물은 `TTSSpeakFrame`(§5.5)이고,
그것은 다른 문제를 위한 다른 mechanism입니다.

### 9.5 ledger, 한 문단, ch-03이 가지고 있으므로

`AudioTextPlayoutLedger`(110줄)는 `ratio = (cursor - sample_start) / (sample_end - sample_start)`와
`text[: int(len(text) * ratio)]`로부터 `audible_text()`를 계산합니다 — [[ch-03/read]] §7.2가 전체 다룸을
가지고 있습니다. §6을 가진 지금 덧붙일 한 가지: 그것의 **입력 요구사항**은 sample count 하나뿐인 반면,
Pipecat의 word-frame 경로는 provider가 timestamp를 emit할 것을 요구합니다. 그래서 §7.4의 6-service
shortlist가 Pipecat migration에서는 중요하고 ledger 기반 migration에서는 중요하지 않을 것입니다. 벤더에
대한 의존성이 다른 두 design. [[ch-13/read]]가 그것을 저울질하고, 이 chapter는 그것을 기록합니다.

---

## 10. Framework-extension probe

세 개의 수. 각각이 이 chapter의 mechanism을 이 chapter 바깥의 무언가에 적용합니다. [[ch-08/read]] 전에
하십시오.

### 10.1 한국어를 아는 `BaseTextAggregator`

조각은 전부 가지고 있습니다: `BaseTextAggregator.aggregate()`는 `AsyncIterator[Aggregation]`을 반환하고
(`base_text_aggregator.py:103`), `Aggregation.type`은 **열린 string**이며(`:46`), `LLMTextProcessor`는
아무 aggregator나 받고(`llm_text_processor.py:39-51`), `TTSService.process_frame`은
`AggregatedTextFrame`에 대해 branch 2를 타고 자기 aggregation을 통째로 건너뜁니다(`tts_service.py:764-765`).

그래서 §9.1의 `1 → 2 → bounded-tail` schedule은 `_batch_phase` counter를 가진 `BaseTextAggregator`
subclass로 구현 가능하고, `LLMTextProcessor` 위에 장착되며, 어떤 TTS service도 subclass하지 **않습니다**.

당신이 스스로 답해야 하는 design 질문, 그리고 이것은 두 codebase 어디에서도 답해지지 않습니다:
`Aggregation`은 `text`와 `type`만 실어 나르고, phase 2는 문장들의 *group*을 하나의 aggregate로 emit합니다.
거기에 무슨 `type` string을 줄 것이며, 그러면 downstream의 `AggregatedTextProgressFrame.segment_id`는
무엇을 가리키게 됩니까 — group입니까 문장입니까? `AggregatedFrameSequencer.register_spoken`
(`aggregated_frame_sequencer.py:292`)을 통해 추적하고 batch된 group에 대해 "segment"가 무엇을 뜻하는지
결정하십시오. 답을 적어 두십시오. 그것이 당신의 live-transcript UI가 문장 단위로 highlight할지 문단
단위로 할지를 결정합니다.

### 10.2 한국어 숫자 rule layer로서의 `text_transforms`

이 hook은 아직 보여드리지 않았습니다. aggregation *이후*, synthesis *이전*에 돌아가는 aggregation-type별
text rewriter입니다:

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

그리고 그 call site, 함정을 정확히 이름 부르는 주석과 함께:

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

Lina에게 이곳이 `29000` → `이만 구천`이 사는 자리이고, `010-1234-5678`이 한 자리씩 읽히게 되는 자리입니다.
저 20줄에서 체득할 것 두 가지:

1. **raise하는 transform은 turn을 조용히 죽입니다.** audio 없음, category `APPLICATION`의 `ErrorFrame`,
   그리고 — 그 context가 audio를 만들지 않았으므로 — §7.8의 zero-audio counter에 tick 하나. 연속으로
   버그 있는 transform 세 번이면 당신의 TTS service가 자기를 write off합니다. 당신 것은 감싸십시오.
2. **word-timestamp provider에서 transform은 assistant context를 바꿉니다.** `TTSTextFrame`이 provider의
   output stream — 즉 *변환된* text — 로부터 지어지기 때문입니다. 그래서 `29000` → `이만 구천`으로
   변환하면, 그 turn에 대한 LLM의 기억은 `29000`이 아니라 `이만 구천`이라고 말합니다. 나중에 숫자가
   중요해지는 sales call에서 그것이 당신이 원하는 것인지 여부는 detail이 아니라 결정입니다.

> 💡 **쉬운 설명 — 2번이 왜 놀라운 결과인가요?**
> 순진한 기대는 이렇습니다: "transform은 TTS에 보내는 문자열만 바꾸고, LLM이 기억하는 원본은 그대로겠지."
> word timestamp를 쓰지 않는 provider에서는 실제로 그렇습니다. 그런데 word-timestamp 경로에서는
> assistant context가 **provider가 되돌려준 단어들**로 지어집니다(§6.6). provider는 변환된 문자열을
> 받았으니 변환된 단어들을 돌려줍니다. 결과: `"보험료는 이만 구천 원입니다"`가 대화 history에
> 들어가고, 다음 turn에 LLM이 "아까 말씀드린 29000원"을 참조할 근거가 사라집니다. 요약하면
> **word-timestamp를 켜는 순간 transform은 렌더링 계층이 아니라 기억 계층을 건드립니다.**

### 10.3 zero-audio guard를 service-switch trigger로 읽기

`_record_context_audio_outcome`은 `push_error(..., force_treat_as_permanent=True)`를 호출하고, 그것이
service를 unusable로 만들며, 그것이 — 자기 docstring에 따라 — `ServiceSwitcher`가 failover하게 해 줍니다.
Lina의 최악 경우에 대한 timeline을 만드십시오: primary 한국어 provider가 실통화 14:03:22에 침묵합니다.

- t+0.0 s: 문장 1 전송, context 열림, audio 없음.
- t+3.0 s: `_stop_frame_timeout_s` 발화, context 1 완료 선언, counter = 1.
- t+6.0 s: counter = 2.
- t+9.0 s: counter = 3 → `push_error(force_treat_as_permanent=True)` → service unusable → switch.

9초. 이제 당신이 실제로 무엇을 원하는지 결정하십시오. 그리고 lever들이 독립적이지 않다는 점에
주의하십시오: `_stop_frame_timeout_s`를 낮추면 *정당하게 느린* provider의 context가 얼마나 빨리 밑에서
닫혀 버리는지도 바뀌고, `max_consecutive_zero_audio_contexts`를 1로 낮추면 function-call만 있는 turn
하나가(설계상 audio를 만들지 않습니다) write-off를 발동시킵니다. 세 번째 lever를 찾으십시오 — 하나
있습니다, context를 구성하는 방식 안에 — 아니면 두 비용 중 하나를 받아들이십시오. §11이 입장을 정합니다.

---

## 11. 산출물: Lina pipeline의 TTS block

system 간의 추천이 아니라 — Pipecat 안에서의 결정 집합이고, 이 chapter에서 결정할 수 없는 것들은 그렇게
표시했습니다.

### 11.1 source로부터 결정된 것

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

### 11.2 여기서 결정되지 않은 것, 그리고 무엇이 그것을 결정할 것인가

| Open question | What would settle it |
|---|---|
| Which of the six providers | A/B on recorded Korean sales dialogue: TTFA distribution, timestamp drift over 10 s, 이름/숫자 pronunciation |
| `max_consecutive_zero_audio_contexts` | Measured base rate of legitimately-silent turns (function-call-only turns) in Lina's traffic |
| `_stop_frame_timeout_s` | Provider p99 first-audio latency under real network conditions |
| SENTENCE vs TOKEN | TTFA delta measured on the chosen provider; §4.7 says the aggregation metric cannot answer it |
| Whether to port the 1→2→tail schedule | TTFA delta between phase-0 single sentence and default sentence aggregation, on the chosen provider |

저 다섯 중 넷이 같은 실험을 필요로 한다는 데 주목하십시오: 선택한 provider에 대해 실제 한국어 traffic에서
측정한 TTFA. 그것을 한 번 돌리면 table 대부분이 붕괴합니다.

### 11.3 스케치

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

저 스케치가 의도적으로 하고 있는 네 가지:

1. **한국어 logic이 TTS service 안에 있지 않습니다.** 그것은 processor입니다. provider를 바꿔도 손대지
   않습니다.
2. **transformer가 turn을 죽일 수 없습니다.** §10.2의 failure mode가 call site에서 처리됩니다.
3. **assistant aggregator가 마지막입니다.** 그것이 [[canonical-voice-bot]] position이고 ch-08의 truncation을
   작동하게 만드는 것입니다.
4. **provider는 placeholder입니다.** 후보 여섯, 그중 어느 것도 이 repo의 무엇에 의해서도 한국어 audio로
   검증되지 않았습니다.

### 11.4 당신이 직접 만들어야 하는 세 가지

1. **on-prem 한국어 TTS**, 배포가 그것을 필요로 한다면. 한국 벤더에 대고 만드는 custom
   `WebsocketTTSService` subclass. tree의 어떤 것도 이것을 하지 않습니다(§7.6).
2. **한국어 TTS acceptance test.** repo에는 없고 있을 수도 없습니다. `pipecat.evals`
   (`src/pipecat/evals/`)가 harness를 줍니다 — 실제 bot, script된 turn, LLM judge — 그리고
   `scripts/release-evals/`에는 한국어 scenario가 없습니다. 그것을 쓰는 것만이 §7.4의 shortlist를
   선택으로 만드는 유일한 길입니다.
3. **under-real-time detection.** synthesis가 1× 아래로 떨어지면 고객은 문장 중간의 gap을 듣고 이 tree의
   어떤 metric도 그것을 보고하지 않습니다(§1.5). `transport.output()`에서의 queue depth가 신호입니다.
   오늘 그것을 노출하는 것은 아무것도 없습니다.

---

## 12. 이 layer가 주지 않는 것

평이하게 진술합니다. 부재도 증거이기 때문입니다:

- **발화 중간 underrun 감지 없음.** §1.5.
- **cross-provider TTFA 비교 harness 없음.** metric은 service별로 얻지만, 둘을 비교하려면 bot을 둘
  돌려야 합니다.
- **provider settings가 노출하는 것 이상의 prosody 제어 없음.** `text_transforms`는 text 위에서
  작동합니다. SSML 지원은 provider별입니다(Azure는 있고, Cartesia는 `cartesia/tts.py:457`에 자기 tag
  dialect가 있습니다 — `spell|emotion|break|volume|speed`에 대한 `_CARTESIA_TAG_RE`).
- **한국어 native provider 통합 없음.** §7.6.
- **어떤 언어 주장에 대해서도 behavioural 검증 없음.** §7.4.
- **TTS path를 관통하는 character 수준 span 추적 없음.** 단위는 frame입니다. "원본 문자열의 어느 문자가
  들렸는가"가 필요하다면, 어절 해상도의 word-frame 경로(§6.3)를 쓰거나, 직접 만드는 것 중 하나입니다.
- **TTS에서 LLM으로 가는 back-pressure 없음.** `pause_frame_processing`(`:164`)은 아직 재생할 audio가
  있는 동안 *inbound frame processing*을 멈추는데, 그것은 같은 것이 아닙니다 — LLM은 queue를 향해 계속
  생성합니다.

---

## 다음 챕터로

이 chapter는 정확히 세 가지를 앞으로 넘깁니다.

**하나의 숫자와 그 분해.** TTFB가 아니라 TTFA, 벤더가 보고하는 것이 아니라 onset detector로 측정되는
`leading_silence`와 함께, 그리고 세 갈래로 쪼개져서 — aggregation delay, provider TTFB, leading silence —
각각에 metric type이 하나씩 붙어서. [[ch-11/read]]가 전체 latency budget을 세우고 이것이 그 네 다리 중
하나입니다. 거기서 waterfall을 그릴 때 위의 세 component가 세 개의 bar가 되고,
`TextAggregationMetricsData`가 당신을 놀라게 할 그것입니다.

**word-timestamp hook, 그 대가와 함께.** 나노초 단위의 `TTSTextFrame.pts`, `transport.output()`의 clock
priority queue가 presentation time에 release, transport의 downstream에 앉은 assistant aggregator에 전달.
그것이 [[ch-08/read]]가 assistant context를 마지막 생성된 token이 아니라 마지막 **발화된** 단어에서
잘라내는 데 쓰는 mechanism 전부이고 — §7.4가 그 대가입니다: 한국어 가능 provider 여섯, grep에서 나왔고,
그중 어느 것도 귀로 검증되지 않음. ch-08의 핵심 capability가 당신이 아직 내리지 않은 벤더 선택에
조건부라는 것을 알고 들어가십시오.

**두 개의 정정과 하나의 정직한 gap.** `AudioContextTTSService`는 deprecated입니다(§0.1) — live ladder는
축 둘에 넷이 아니라 축 하나에 세 class입니다. 한국어 word grouping은 네 번에 걸쳐 unit-test되어
있습니다(§0.2) — 검증되지 않은 것은 당신이 **들을 수 있는** 무엇입니다. 두 정정 모두 outline이 기억에
의존해 기술한 파일들을 직접 열어서 나왔고, 둘 다 결론을 바꿨습니다. 그것이 이 course가 세우려는
습관이고, 그래서 §7.4가 자기 provenance를 각주가 아니라 block quote로 진술하는 것입니다.

의도적으로 여기 **없는** 것: `KoreanPhraseChunker`의 명시적 guard와 NLTK-플러스-lookahead 중 무엇을
production에서 돌려야 하는지에 대한 어떤 판단도. §9.3은 같은 여섯 입력을 양쪽으로 돌리고 결과를
출력했습니다. 다섯은 일치합니다. 하나는 갈리는데, 그것은 만들려고 애써야 나오는 입력에서입니다. 그것은
측정이고, [[ch-13/read]]까지 측정으로 남습니다. 그때는 ch-08의 interruption mechanics, ch-09의 context
ownership, ch-11의 latency budget을 손에 쥐고 있을 텐데 — 지금은 그중 아무것도 없습니다.

다음은 [[ch-08/read]]입니다. barge-in cascade를 frame 하나씩 분해하고, 이 chapter가 세워 두고 답하지 않은
질문 위에서 시작합니다: 고객님이 4.2초짜리 문장의 1.4초 지점에서 Lina를 끊었을 때, assistant context는
정확히 무엇을 담게 되는가 — 그리고 그것이 옳은 것이 되기 위해 몇 개의 서로 다른 mechanism이 합의해야
했는가?
