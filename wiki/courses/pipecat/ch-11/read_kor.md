---
title: "The Latency Budget and the Observer Plane"
chapter: ch-11
phase: read
course: pipecat
lang: ko
companion_of: read.md
sources:
  - latency-budget-voice
  - rtvi-observability
  - endpointing-turn-boundary
  - transport-telephony
figure: figures/latency-waterfall.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# ch-11 — The Latency Budget and the Observer Plane

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 (frame, processor, node, transition, action, aggregator, latency,
> observer, falsifier, budget, term, serial, overlap 등).

## 왜 이 챕터인가

이 chapter는 있어야 할 자리보다 한 칸 앞에 놓여 있고, 그 이유는 **분모(denominator)** 때문입니다.

[[ch-12/read]]에는 하중을 받는 decision이 정확히 하나 있습니다: boson의 rule layer가 aggregate된
transcript와 LLM call 사이에서 *blocking*으로 도는가, 아니면 LLM의 첫 token들과 *concurrent*하게
도는가. 그 decision은 millisecond 단위로 값이 매겨집니다. 구조(structural) 질문이 아닙니다 —
구조적으로는 둘 다 하나의 position에 놓인 하나의 `FrameProcessor`입니다 — 그것은 *"N ms가 한 turn의
비율로서 받아들일 만한가?"* 형태의 산술(arithmetic) 질문입니다. 한 turn이 얼마인지, 그 turn의 부분들이
각각 얼마인지, 그리고 그중 어느 부분을 지출해도 되는지를 알기 전에는 그 질문에 답할 수 없습니다.
그래서 budget이 먼저 옵니다.

그리고 budget은 혼자서는 먼저 올 수 없습니다. 측정할 수 없는 budget은 budget이 아니라 표를 두른
추측이기 때문입니다. 그래서 이 chapter에는 subject가 둘이고, 그 둘을 하나로 다룹니다:

1. **voice-to-voice latency budget** — 사용자의 마지막 유성음(voiced) sample부터 assistant의 첫
   가청(audible) sample까지의 모든 millisecond를, 이름이 붙고 따로따로 측정되는 stage들의 합으로.
2. **observer plane** — frame graph 위에 얹힌 두 번째의 read-only plane. 위 stage들이 애초에 숫자를
   가질 수 있는 이유가 바로 이것입니다.

당신은 이것과 정확히 같은 shape의 회계(accounting) 문제를 이미 만난 적이 있습니다. 당신의 GPU
training-memory course는 하나의 ledger였습니다: parameters + gradients + optimizer state +
activations, 각 term에 이름이 있고, 각 term이 귀속(attributable)되며, 어떤 것도 "overhead"라고
불리는 것이 허용되지 않는 규율. 이것은 같은 연습을 시간 축(time axis) 위에서 하는 것입니다. 차이는
둘뿐입니다: (a) 이 term들 중 일부는 overlap하고 일부는 하지 않는데, 그것을 틀리는 것이 voice-latency
산술이 망가지는 가장 흔한 방식이고, (b) term 하나는 이 chapter의 끝에서 **일부러 비워 둡니다**.
그것은 당신이 값을 매길 몫이고, 제가 대신 매기지 않을 것이기 때문입니다.

### 이 chapter에 들어 있지 않은 것

Deployment, process topology, worker-per-call 대 worker-per-process, autoscaling, cold-start
경제성 — 전부 여기 없습니다. latency accounting과 deployment는 서로 다른 subject인데 outline의 이전
draft에서 합쳐져 있었습니다. process topology는 이미 당신이 가지고 있는 [[ch-04/read]] §13에 속하고,
deployment는 [[ch-13/read]]에 속합니다. 아래의 어떤 것도 그 둘에 의존하지 않습니다.

또한 여기 없는 것: **realtime_voice 대 Pipecat에 대한 어떤 판정도**. §11은 세 시스템 각각이 무엇을
측정하는지를 진술합니다. "더 낫다"도, "더 나쁘다"도, "채택해야 한다"도 없습니다. [[ch-13/read]]가
무언가에 점수를 매기는 유일한 곳이고, [[ch-08/read]] §9의 상시 규칙이 여기에도 적용됩니다.

### 이미 가지고 있어야 하는 것

이 chapter는 앞선 네 chapter를 *소비(consume)*하고, 그중 어느 것도 다시 가르치지 않습니다:

- **[[ch-04/read]] §4** — processor마다의 two-queue / two-task runtime과 `SystemFrame` priority
  split. §6에서 observer hook이 왜 data path 위에 앉아 있지 않은지를 이해하는 데 필요하고, §7에서
  `MetricsFrame`이 왜 진행 중인 barge-in에 의해 유실될 수 없는지를 이해하는 데 필요합니다.
- **[[ch-06/read]] §9, §11, §15** — STT TTFB 재정의
  (`speech_end_time = frame.timestamp - frame.stop_secs`), 23개 상수짜리 `stt_latency.py` 표, 그리고
  `effective_stt_wait = max(0.0, stt_timeout - stop_secs)` 안전망. **그 표는 provider 선택이 실제로
  일어나는 그곳에서 한 번 만들어졌습니다.** 아래 §3.1은 그것을 단일 term으로 *지출*합니다. 다시
  렌더링하지 않고, 이 chapter의 figure도 마찬가지입니다.
- **[[ch-07/read]] §1, §4** — `TTFA = TTFB + leading_silence`, `detect_speech_onset`, 그리고 이름이
  붙은 비용으로서의 sentence aggregation. §3.4와 §3.5가 둘 다 budget line으로 소비합니다.
- **[[ch-08/read]] §6** — `audio_out_10ms_chunks = 4` → written chunk당 40 ms, 그리고 output
  transport에서의 two-queue drain. §3.6이 두 숫자를 모두 지출합니다.

### evidence를 읽는 법

아래의 모든 Pipecat 줄 번호는 commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`
(2026-08-25; `CHANGELOG` head `[1.7.0] - 2026-08-01`) 시점의
`wiki/raw-data/pipecat/pipecat-src/`에서 열어 확인한 것입니다. `boson-agent`와 `realtime_voice`에
대한 모든 주장은 excerpt library — [[latency-budget-voice]], [[rtvi-observability]],
[[boson-gateway-server]], [[boson-layers-rules]], [[rtv-vs-pipecat-gap]], [[rtv-pipeline-session]],
[[rtv-webrtc-transport]], [[rtv-vad-chunking]] — 에서 나왔고, 그것은 2026-08-25에 당신의 private
repo에서 읽어 온 것입니다. 그 repo들은 여기서 열지 않습니다.

---

## 0. 시작하기 전 네 개의 정정

넷 다 이 chapter가 조립된 자료 안에 있던 내용입니다. 넷 다 이 commit의 tree에 비추어 틀렸거나, 정도가
틀렸습니다. excerpt가 source와 어긋나는 곳에서는 **source가 이깁니다**. 그리고 저는 조용히 한쪽을
고르는 대신 본문에서 그렇게 말합니다.

### 0.1 Metrics는 기본값이 OFF다

[[rtvi-observability]]의 guideline은 이렇게 적혀 있습니다: *"Turn on `PipelineParams(enable_metrics=True,
enable_usage_metrics=True)` from day one."* 좋은 조언이고, 마치 default가 취향의 문제인 것처럼 쓰여
있습니다. 아닙니다. default를 읽으십시오:

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

`enable_metrics: bool = False`. 기본 Pipecat bot은 **아무것도** 측정하지 않습니다. TTFB도, TTFA도,
TTFAT도, processing time도, text aggregation도. 이 chapter의 모든 metric은 저 boolean 하나 뒤에 있고,
모든 emit site가 그것에 gate되어 있기 때문입니다(§6.4). 이 chapter에서 딱 하나의 운영 지시만
가져간다면, 그것은 저 줄과 그 형제인 `enable_usage_metrics`입니다.

### 0.2 `UserBotLatencyObserver`도 기본으로 auto-wire되지 않는다

[[rtvi-observability]]는 `enable_tracing=True`일 때 `PipelineWorker`가 latency observer를
auto-wire한다고 말합니다. 사실입니다 — 하지만 `enable_tracing`은 default가 아니고, 조건도 그것
하나가 아닙니다:

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

두 개의 gate가 AND로 묶여 있습니다. `enable_tracing`의 default는 `False`이고, *게다가* OpenTelemetry를
import할 수 없으면 조용히 `False`로 강등됩니다. 그다음 세 번째 gate:

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

그래서 framework가 요청하지 않아도 `LatencyBreakdown`을 건네주게 만드는 전체 조건은:
`enable_tracing=True` **그리고** OpenTelemetry 설치 **그리고** `enable_turn_tracking`을 켠 채로 둠
**그리고** `PipelineParams(enable_metrics=True)`. 네 가지입니다. 하나라도 놓치면
`on_latency_breakdown`은 절대 발화하지 않습니다.

실무적 귀결이고, 작은 것이 아닙니다: **당신이 직접 `UserBotLatencyObserver()`를 만들어
`observers=[...]`로 넘기십시오.** 실제로 OTel span을 원하는 게 아니라면 tracing 경로에 기대지
마십시오. 그것은 평범한 `BaseObserver`이고, tracing을 필요로 하는 구석이 전혀 없습니다. tracing이
*그것을* 필요로 하는 것이지, 그 반대가 아닙니다.

### 0.3 observer plane은 read-only이지만, synchronous하지는 않다

"`BaseObserver`가 모든 frame transfer를 본다"는 문장의 자연스러운 독해는, 각 processor가 각 observer를
inline으로 호출한다는 것 — 즉 느린 observer가 frame path를 직접 늘린다는 것입니다. 실제로 일어나는
일은 그게 아닙니다. processor는 **하나의** observer — proxy — 만 호출하고, 그 proxy는 enqueue만
합니다:

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

**등록된 observer마다** `asyncio.Queue` 하나와 task 하나
(`_create_proxy`, `worker_observer.py:173-178`). class docstring이 대놓고 말합니다:

**`src/pipecat/pipeline/worker_observer.py:57-63`**
```python
    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a worker for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each worker.
```

이것은 두 번 중요합니다. 한 번은 안심으로: 당신의 custom observer는 느린 일을 해도 고객의 critical
path에서 그 값을 치르지 않습니다. 그리고 한 번은 경고로, §10.3에서: 그 queue들은 unbounded이고,
observer는 dequeue *이후에* `time.time()`으로 자기 자신의 timestamp를 계산합니다 — event에 실려 온
pipeline-clock timestamp가 아니라. 두 사실 모두 measurement 상의 귀결이 있습니다.

> 💡 **쉬운 설명 — proxy 한 겹이 왜 그렇게 중요한가요?**
> 만약 processor가 observer 5개를 직접 `await`했다면, frame 하나가 A→B로 넘어갈 때마다 observer 5개의
> 작업이 *그 자리에서* 끝나야 다음 processor로 넘어갑니다. observer 하나가 DB에 쓰느라 20 ms 걸리면
> 그 20 ms는 고객이 기다리는 시간에 그대로 더해집니다. Pipecat은 그 사이에 queue를 끼워서, processor가
> 하는 일을 "queue에 넣기"로 줄였습니다. 그래서 느린 observer는 **turn을 늘리는 대신 queue를 늘립니다**.
> 대가는 두 가지입니다: (1) queue에 언제 넣었는지가 아니라 언제 꺼냈는지 기준으로 시간을 재면 숫자가
> 밀리고, (2) queue에 maxsize가 없으니 밀린 만큼 메모리가 자랍니다.

### 0.4 `stt_latency.py`는 68줄에 측정 상수 23개이고, excerpt 목록의 숫자 하나는 잘못 세기 쉽다

[[latency-budget-voice]]가 그 표를 나열하고 있고 표 자체는 맞지만, 두 항목은 흘리기 쉽습니다.
`DEEPGRAM_SAGEMAKER_TTFS_P99`는 Deepgram의 0.35와 값이 같을 뿐인 *별개의* 상수이고,
`NVIDIA` / `WHISPER`는 측정값이 전혀 아닙니다 — fallback의 alias입니다:

**`src/pipecat/services/stt_latency.py:66-68`**
```python
# These services run locally and should be replaced with measured values
NVIDIA_TTFS_P99: float = DEFAULT_TTFS_P99
WHISPER_TTFS_P99: float = DEFAULT_TTFS_P99
```

그러므로 이 파일의 68줄은 **측정된 provider 상수 23개**, `DEFAULT_TTFS_P99 = 1.0` 하나, 그리고 그
default의 alias 두 개를 담고 있습니다. [[ch-06/read]] §11.1이 23개를 전부 출력합니다. 이 chapter는
그중 정확히 두 개 — 양 극단 — 만 쓰고, 표는 절대 다시 렌더링하지 않습니다.

---

## 1. latency budget이란 정확히 무엇인가

코드에 들어가기 전에 개념부터. §3의 공식이 외워야 할 무언가가 아니라 이미 믿고 있는 것의 요약이
되도록 진술합니다.

### 1.1 하나의 숫자, 그리고 하나의 숫자가 왜 쓸모없는가

고객이 경험하는 숫자는 하나의 interval입니다:

```
[ last voiced sample of the customer's speech ]  ──────►  [ first AUDIBLE sample of Lina's reply ]
```

이것을 **voice-to-voice interval**이라고 부릅시다. product owner가 신경 쓰는 유일한 숫자이고, 당신
자신의 `CLAUDE.md` target(§2.1)에 등장하는 유일한 숫자입니다. 동시에 그것 하나만으로는 engineering에
쓸모가 없습니다 — "이 model은 47 GB를 쓴다"가 쓸모없는 것과 같은 이유로: 총합은 알려주지만 어떤
lever가 그것을 움직이는지는 알려주지 않습니다.

쓸모 있는 대상은 decomposition입니다. Pipecat의 decomposition은 문서상의 산물이 아닙니다 — 그것은
**type system에 물질화(materialise)되어** 있고, 237줄짜리 파일 하나 안의 Pydantic class 여덟 개입니다:

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

모든 subclass가 `processor`와 `model`을 지니고 다닙니다. 그것이 attribution mechanism의 전부입니다:
metric은 결코 맨 float가 아니라, float *더하기 누가 만들었는지 더하기 어느 model이 만들었는지*
입니다. 그것이 정확히 ledger가 필요로 하는 것이고, `logger.info(f"took {t}ms")`가 주지 않는
것입니다.

`metrics.py` 안의 timing/usage class 여덟 개:

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

(`:225`의 `SmartTurnMetricsData`는 0.0.104부터 `@deprecated`이고 2.0.0에서 제거 예정입니다 — 그
위에 무언가를 쌓지 마십시오.)

저 표를 **무엇을 분리할 가치가 있는가에 대한 주장**으로 읽으십시오. "time to first something" class가
셋인 이유는 서로 다른 first-something 셋이 중요하고, 그 셋이 일상적으로 혼동되기 때문입니다. §3.3과
§3.5가 정확히 그 혼동에 관한 것입니다.

### 1.2 voice-latency 산술의 두 가지 failure mode

지금 이름을 붙여 둘 가치가 있습니다. §4가 그 둘을 피하는 방법에 관한 것이기 때문입니다.

**Failure mode 1 — overlap하는 term들을 더하기.** TTS synthesis wall time은 budget에 없습니다.
[[ch-07/read]] §1.1이 이것을 구체화했습니다: audio는 1× wall clock으로 재생되므로, 첫 가청 sample이
나간 순간부터 남은 synthesis의 모든 millisecond는 *이미 돌아가고 있는* playback **아래에서**
일어납니다. "TTS total synthesis time"을 budget에 더하면 고객이 실제로 경험하는 interval의 2–3배인
숫자를 만들어내고, 그다음 엉뚱한 것을 최적화하게 됩니다.

**Failure mode 2 — tail term을 typical term으로 착각하기.** [[ch-06/read]] §15.3이 이미 STT 안전망은
*"거의 지불하지 않는 천장(a ceiling you almost never pay)"*이라고 말했습니다. `effective_stt_wait`은
모든 VAD stop에서 무장(arm)되고 finalize된 transcript가 도착하는 순간 취소됩니다. Deepgram의 0.35 s
대신 xAI의 2.14 s를 고르는 것은 median turn에 1.79 s를 더하지 않습니다. **최악의** turn들에 더합니다.
당신의 target이 P50 *과* P95라면 — 그리고 당신 것이 그렇습니다 — 그 둘은 같은 term들로 만들어졌지만
지배하는 term이 다른, 서로 다른 두 개의 budget입니다. §5가 그 산술을 명시적으로 합니다.

figure는 두 failure mode를 한 번에 보이게 만들기 위해 존재합니다:

> **[The voice-to-voice latency waterfall](figures/latency-waterfall.html)** — 각 stage knob을 드래그해
> 총합이 당신 자신의 P50 = 1.0 s / P95 = 1.5 s 선에 대해 어떻게 움직이는지 보십시오. 더 읽기 전에
> 딱 하나만 해 보십시오: STT provider를 Deepgram으로 두고 총합을 적어 두고, xAI로 바꾼 뒤, bar가
> 1.79 s 움직이는데 *serial* stage들은 전혀 변하지 않는다는 것을 관찰하십시오. 그것이 failure mode 2를
> 그림으로 그린 것입니다.

> 💡 **쉬운 설명 — 두 failure mode를 한 문장으로**
> failure mode 1은 **동시에 일어나는 것을 줄 세워 더하는** 실수이고(TTS가 계속 만들어내는 동안 이미
> 소리가 나고 있음), failure mode 2는 **가끔만 내는 돈을 매번 내는 돈으로 적는** 실수입니다(STT
> 안전망은 transcript가 늦게 올 때만 지불됨). 첫째는 총합을 부풀리고, 둘째는 P50과 P95를 뒤섞습니다.
> ledger 용어로 말하면 첫째는 "double counting", 둘째는 "평균과 분포 꼬리의 혼동"입니다.

---

## 2. 존재하는 유일한 target 수치들과, 그것들이 어디서 왔는가

### 2.1 당신의 것

당신 자신의 `CLAUDE.md`가 target을 진술하고 있고, 그것이 이 course의 evidence base 전체에서 유일하게
명시된 perceptual target입니다. [[rtv-vs-pipecat-gap]]의 인용에서, boson-agent의 `CLAUDE.md` mission을
그대로 옮기면:

> "P50 at or below 1.0 seconds and P95 at or below 1.5 seconds", measured "from the last voiced user
> sample to the first audible assistant sample, including end-of-turn/VAD time"

저 문장에서 실제로 일을 하는 세 가지가 있고, 흐려지게 두면 안 됩니다:

1. **"last voiced user sample"** — "VAD가 결정한 시점"이 아닙니다. 둘은 `stop_secs`만큼 다릅니다.
   Pipecat은 여기서 당신에게 동의하고, 서로 독립적인 두 곳에서 그 뺄셈을 합니다(§3.1, §7.2).
2. **"first audible assistant sample"** — "audio의 첫 byte"가 아닙니다. 둘은 `leading_silence`만큼
   다릅니다. Pipecat에는 정확히 그 차이를 위한 metric(`TTFAMetricsData`)이 있고, §7.4가 보이듯
   headline latency observer는 *그것을 쓰지 않습니다*.
3. **"including end-of-turn/VAD time"** — endpointing wait은 budget 안에 있지, 그 앞에 있지 않습니다.
   이 절이 budget을 정직하게 만드는 절이고, 대부분의 vendor latency 마케팅이 조용히 빼먹는 절입니다.

같은 `CLAUDE.md`가 engineering rule도 진술합니다 — *"Instrument before optimizing… Report
P50/P95/P99"* ([[rtv-vs-pipecat-gap]]) — 이 chapter 전체가 그 문장을 위한 mechanism입니다.

### 2.2 Pipecat의 것: 없습니다. 그리고 그것을 직접 확인하기를 바랍니다

repo는 **어디에도 perceptual latency target을 진술하지 않습니다.** README가 한 가지를 말하는데,
그것은 마케팅 형용사입니다:

**`README.md:29`**
```markdown
- **Real-Time**: Ultra-low latency interaction with different transports (e.g. WebSockets or WebRTC)
```

`AGENTS.md`는 아무것도 진술하지 않습니다. `CLAUDE.md`는 `@AGENTS.md` 한 줄입니다. `docs/`는 Sphinx
config입니다. eval harness 전체에서 latency 모양을 한 유일한 상수는 **test timeout**이고, 주석이
그렇게 말합니다:

**`src/pipecat/evals/harness.py:110-113`**
```python
# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000
```

60초. eval이 latency 때문에 실패하지 *않도록* 명시적으로 고른 값입니다. 그리고 배포되는
release-eval suite가 이를 뒷받침합니다: `scripts/release-evals/scenarios/`의 시나리오 파일 **38개**
중 `within_ms`를 설정하는 것은 정확히 **2개** —
`filter_incomplete_turns_user_idle.yaml:34` (`within_ms: 45000`) 과
`multi_worker_handoff_back_and_forth.yaml:53` (`within_ms: 40000`). 둘 다 40초 이상의 timeout이고,
latency assertion이 아니라 state machine을 지키는 것입니다.

이 주장 전체를 명령 두 줄로 검증할 수 있습니다:

```bash
# run in wiki/raw-data/pipecat/pipecat-src at commit 0cbf9c5b
grep -rn -i "latency" README.md AGENTS.md
ls scripts/release-evals/scenarios/*.yaml | wc -l && grep -rln "within_ms" scripts/release-evals/scenarios/ | wc -l
```

**외부 지식이며, 외부라고 표시합니다.** voice UX에서 모두가 인용하는 두 수치 — 사람의 자연스러운
turn-taking gap이 대략 ~200 ms, "살아 있는 것처럼 느껴지는" voice-to-voice 천장이 대략 ~800 ms — 는
더 넓은 voice-UX 문헌과 vendor 마케팅에서 나옵니다. 이 repository에 있지 않으며 Pipecat의 주장으로
인용해서는 **안 됩니다**. 슬라이드에 올린다면 업계 관행(industry convention)으로 인용하거나, 아니면
빼고 당신 자신의 `CLAUDE.md`를 인용하십시오. 그것은 실제 내부 commitment이고 더 나은 논거입니다.

### 2.3 그래서 모든 bar가 그어지는 기준선은 무엇인가?

당신 것입니다. P50 ≤ 1.0 s, P95 ≤ 1.5 s, 마지막 유성 user sample → 첫 가청 assistant sample. 이
chapter에서 주인이 있는 유일한 숫자이고, figure가 두 선을 모두 그리는 이유는 선이 하나면 failure
mode 2를 가려버리기 때문입니다.

---
## 3. 공식, term 하나씩

[[latency-budget-voice]]가 주는 그대로 먼저 평이하게 진술하면:

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

그것이 shape입니다. 이제 각 term을, 그것을 만들어내는 source와 함께 — 그리고 source가 한 줄짜리
공식과 어긋나는 곳에서는 정정과 함께 — 봅니다.

### 3.1 Term 1+2: endpointing wait

이것이 [[ch-06/read]]가 만든 term이고, budget에서 가장 큰 둘 중 하나입니다. 부분이 둘이고, 둘 다
이미 당신 것입니다:

- `VADParams.stop_secs = 0.2` — VAD가 turn이 끝났다고 선언하기 전에 관찰해야 하는 침묵.
  [[ch-06/read]] §2는 이것이 16 kHz / 512 samples에서 **6 chunk**임을 도출했습니다
  (`round(0.2 / 0.032)` = `round(6.25)` = 6), 즉 200이 아니라 실제 audio 192 ms입니다.
- `max(0, TTFS_p99 − stop_secs)` — 안전망. STT service 자신이 공표한 P99
  time-to-final-segment로부터 크기가 정해집니다. [[ch-06/read]] §15.2가 provider 넷에 대해
  계산했습니다.

뺄셈이 존재하는 이유는 같은 침묵의 값을 두 번 치르지 않기 위해서입니다: provider의 P99는 고객의
*물리적* 발화 종료 시점부터 측정되는데, strategy는 그것을 `stop_secs`만큼 늦게 듣기 때문입니다.

**정정, 그리고 진짜 정정입니다.** 저 평이한 공식은 **default** stop strategy에만 맞고, 오직 그것에만
맞습니다. `UserTurnStrategies`의 default는
`stop = [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]`
([[endpointing-turn-boundary]], `turns/user_turn_strategies.py` L43)이고, 그 strategy는 wait를
**절대적 deadline**에 고정합니다:

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

`timeout = max(0, stt_deadline - time.time())`을 주의 깊게 읽으십시오. 이것은 장부 기입이 아니라
budget design의 한 조각이기 때문입니다. `analyze_end_of_turn()`은 deadline이 계산된 시점과 timer가
무장되는 시점 **사이에서** smart-turn ONNX inference를 돌립니다. deadline이 절대적이기 때문에,
inference 시간은 STT wait 뒤에 더해지는 대신 그 안으로 *흡수됩니다*. smart-turn inference는 STT의
P99 window 안에서 끝나는 한 공짜입니다. 진정으로 우아한 회계 조각이고, default strategy의
endpointing term이 `stop_secs + inference + max(...)`가 아니라 정말로
`stop_secs + max(0, TTFS_p99 − stop_secs)`인 이유입니다.

> 💡 **쉬운 설명 — "절대 deadline이 inference를 흡수한다"가 무슨 뜻인가요?**
> 두 가지 방식으로 타이머를 걸 수 있습니다.
> (a) 상대적: "지금부터 0.15초 기다려라" → inference가 0.05초 걸렸다면 총 대기는 0.05 + 0.15 = 0.20초.
> (b) 절대적: "발화 종료 시각 + 0.35초까지 기다려라" → inference가 0.05초 걸렸어도 종료 시각은 그대로이므로
> 남은 대기는 0.10초로 줄고, 총 대기는 여전히 0.35초.
> Pipecat은 (b)를 씁니다. 그래서 ONNX inference가 아무리 오래 걸려도 — STT P99 window를 넘기지만
> 않는다면 — budget에 한 줄도 추가되지 않습니다. **inference가 빨라지는 게 아니라, 어차피 기다릴
> 시간 안에 숨는 것**입니다.

*다른* timing strategy는 평이한 공식과 맞지 않습니다. `SpeechTimeoutUserTurnStopStrategy`는 **둘 다**
완료되어야 하는 timer 두 개를 돌립니다:

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

두 timer는 같은 순간에 무장되어 concurrent하게 돌기 때문에, 그 경로에서 endpointing term은:

```
stop_secs + max(user_speech_timeout, max(0, TTFS_p99 − stop_secs))
          = 0.2 + max(0.6, effective_stt_wait)     with the defaults
```

Deepgram(`effective_stt_wait = 0.15`)이면 **0.6 s policy floor가 완전히 지배하고** STT 상수는
무관해집니다. 빠른 STT를 사러 가기 전에 내재화할 가치가 있는 결과입니다: VAD-only 경로에서는
0.8 s TTFS보다 빠른 STT를 사도 `user_speech_timeout`을 함께 낮추지 않는 한 아무것도 얻지 못하고,
그것을 낮추는 것은 한국어 화자의 문장 중간 pause가 얼마나 길어도 되느냐에 대한 product decision이지
latency decision이 아닙니다.

**그래서 당신 시트에는 어느 term이 올라가는가?** 당신이 설정한 strategy의 것입니다. 숫자 옆에
strategy 이름을 적어 두십시오. strategy가 붙어 있지 않은 budget line은 검증 가능하지 않습니다.

**이 chapter가 ch-06의 표에 더하는 단 한 가지.** provider 선택이 이 term을 극단 사이에서
**2.14 − 0.35 = 1.79 s** 움직입니다 (`XAI_TTFS_P99` 대 `DEEPGRAM_TTFS_P99` /
`DEEPGRAM_SAGEMAKER_TTFS_P99` / `SONIOX_TTFS_P99`, `stt_latency.py:45,46,61,63`). 그것은 당신의
**전체** P50 budget보다 큽니다. 그 파일에서 가장 큰 단일 lever이고, config argument입니다. figure의
provider control은 정확히 이 시연을 위해 그 이름과 상수들의 read-only dropdown입니다 — ch-06의 표를
다시 만드는 것이 아니라 *지출*하는 것입니다.

### 3.2 Term 3: rule evaluation — TBD ms

여기가 빈 slot이고, 비어 있는 채로 둡니다.

aggregate된 user transcript와 LLM service 사이에 serial하게 놓는 어떤 processor든 critical path에
올라탑니다. [[boson-layers-rules]]는 boson의 `LayerPipeline`이 정확히 그런 것임을 확립합니다: 이미
완성된 하나의 user utterance에 대한 two-phase-commit 투표, `gateway/layers/`와 `gateway/rules/`에
걸친 1,206 LOC이고, 완성된 `str`을 *반드시* 봐야 하므로 aggregator 뒤에 *반드시* 앉아야 합니다. 그
migration note는 명확합니다 — transaction 전체가 **하나의** `FrameProcessor`로 붕괴해야 합니다.
`push_frame()`은 되돌릴 수 없고, 여러 processor에 흩뿌려진 veto는 rollback될 수 없기 때문입니다.

그러므로 budget line은 존재하고, 실재하며, serial합니다. 그것이 얼마인가는 몇 개의 check가 도는지,
그중 LLM을 호출하는 것이 있는지, 그리고 그것이 pipeline을 block하는지 아니면 race하는지의 함수입니다.
그 전부가 [[ch-12/read]]의 subject입니다. 실제 숫자가 딸린 실제 답이 있고, 여기서 그것을 건네주는
것은 그것을 당신이 직접 도출하게 만드는 것이 존재 목적인 chapter의 결론을 미리 넘겨주는 일입니다.

그래서 당신 시트의 그 줄은 문자 그대로 이렇게 적힙니다:

```
rule evaluation ................................ TBD ms   (ch-12 prices it; ch-13 measures it)
```

figure의 rule knob은 드래그 가능하고 **default 값이 없으며** 그렇다고 말하는 label이 붙어 있습니다.
드래그해서 총합이 당신의 P50 선에 대해 어떻게 되는지 보십시오. 그것이 연습입니다: 답이 아니라
분모를 소유한 채로 ch-12에 도착하기.

이 chapter가 *주는* 것 하나는, 숫자가 아니라 mechanism이기 때문입니다 — 그 term이 설명 불가능한
gap이 되는 대신 눈에 보이게 만드는 방법. 그것이 §12입니다.

### 3.3 Term 4: LLM time-to-first-token, 그리고 왜 TTFB가 잘못된 시계인가

LLM term은 모두가 최적화하는 term이고, §5가 보이듯 지배하는 경우는 드뭅니다. 또한 대충 읽으면
Pipecat의 이름 짓기가 당신을 오도할 term이기도 합니다.

`TTFBMetricsData`는 "model이 답하기 시작할 때까지의 시간"을 측정하지 **않습니다**. model이 *어떤*
output이든 만들어낼 때까지의 시간을 측정합니다. docstring이 명시적이고, 이 chapter에서 가장 중요한
docstring입니다:

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

한 문단에 규칙이 둘. 첫째: **reasoning token이 TTFB 시계를 멈춥니다.** 900 ms 동안 생각한 뒤 답하는
reasoning model은 *보기 좋은* TTFB와 기다린 고객을 함께 보고할 것입니다. 둘째: HTTP response head나
keepalive는 그것을 멈춰서는 **안 됩니다**. 그러지 않으면 connection setup을 재게 됩니다.

"고객의 기다림"에 실제로 대응하는 숫자는 TTFAT입니다:

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

`ttfat = ttfb + thinking_time`이고 — [[ch-07/read]] §1.2의 TTFA와 정확히 마찬가지로 — 거울처럼 복사된
`ttfb` field는 **편의를 위한 복사본**입니다. *"not a separate measurement, so don't aggregate both."*
당신의 Lina dashboard가 한 통화에서 `TTFBMetricsData.value`를 합산하고 *동시에*
`TTFATMetricsData.ttfat`도 합산한다면, model의 응답 시간을 이중 계산하는 것입니다. 같은 파일에서 같은
함정이 두 번째로 나온 것이니, 집안 규칙으로 삼으십시오: **mirrored field는 표시용이지, 절대
aggregation용이 아니다.**

주장을 확인하는 곳은 call site입니다. OpenAI base LLM에서:

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

TTFB는 choices가 들어 있는 첫 chunk에서 멈춥니다 — reasoning 포함. TTFAT는 따로 멈추는데, 위처럼 첫
tool-call delta에서 멈추거나, 아니면 공유 base class에서 처음 push되는 text token에서 멈춥니다:

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

그것이 어디에 앉아 있는지 보십시오: turn-completion filtering **이전**입니다. metric은 *model*을
재는 것이지, model의 output을 붙잡아 두겠다는 pipeline의 결정을 재는 것이 아닙니다. 그리고
`reports_ttfat`은 설정되는 것이 아니라 파생되는 것입니다:

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

**Lina에 대한 귀결, 평이하게 진술하면:** 언젠가 speech-to-speech model로 옮기면 그 service에 대해
`TTFAT`은 조용히 존재하기를 그치고, 당신의 budget은 LLM line을 잃습니다. 아무 error도 나지 않습니다.
field가 그냥 나타나지 않게 될 뿐입니다. 그 일이 벌어진 뒤가 아니라 벌어지기 전에 계획하십시오.

**"두 번 보고한다"는 경우는 당신에게 edge case가 아닙니다.** Lina는 tool을 호출합니다. tool 결과로부터
답하는 turn은 inference를 두 번 돌리고 따라서 `TTFATMetricsData`를 두 개 emit합니다. docstring이 어느
것을 남길지 말해 줍니다 — *"consumers wanting one per user turn keep the first"* — 그리고 별개로 tool
실행 자체가 자기만의 line으로 측정됩니다(`FunctionCallMetrics`, §7.3). 그러므로 tool turn의 LLM 비용은
당신 시트에서 하나가 아니라 세 숫자입니다: 첫 inference TTFAT, function duration, 두 번째 inference
TTFAT.

### 3.4 Term 5: text aggregation

[[ch-07/read]] §4가 이것을 만들었습니다. 정의만 되새기고 넘어갑니다:

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

이것이 "TTS"에 접혀 들어가지 않고 자기 line으로 budget에 오르는 이유는, 그것이 TTS의 비용이 *아니기*
때문입니다 — 그것은 당신의 pipeline에서 내린 policy decision(`TextAggregationMode.SENTENCE`, default)의
비용이고, 당신이 바꿀 수 있습니다. `TextAggregationMode` 자신의 docstring이 그것에 "~200-300 ms per
sentence"라는 값을 매깁니다([[ch-07/read]] §4.1).

budget에 중요하고 ch-07에는 없었던 기계적 사실 둘:

**한 turn에서 첫 번째 aggregation만 budget에 오릅니다.** window는 transcription이 아닌 첫
`TextFrame`에서 열리고 `TOKEN`이 아닌 첫 aggregate에서 닫힙니다:

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

두 번째, 세 번째 문장도 aggregate되지만, 그것들은 첫 문장의 playback *아래에서* aggregate됩니다 —
다시 failure mode 1입니다. observer도 breakdown을 만들 때 같은 규칙을 강제합니다:

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

**그리고 이 term은 모든 exporter에게 보이지 않습니다.** 여기서 표시해 두고 §8.3에서 증명합니다:
`RTVIObserver`는 `TextAggregationMetricsData`를 bucket에 넣지 않고, `MetricsLogObserver`는 그것을
import하지 않으며, `SentryMetrics`는 그것을 만들어내는 어떤 것도 override하지 않습니다. tree 전체에서
**유일한** consumer는 `UserBotLatencyObserver`입니다. RTVI나 Sentry로 Lina를 instrument하고 latency
observer로는 하지 않는다면, 이 budget line은 당신에게 존재하지 않습니다.

### 3.5 Term 6: TTS time-to-first-AUDIBLE-sample

[[ch-07/read]] §1이 이것도 만들었고, 가장 큰 두 term 중 두 번째입니다. 항등식 —
`ttfa = ttfb + leading_silence` — 과, `leading_silence`가 선언되는 것이 아니라 streaming PCM 위에
energy 기반 onset detector를 돌려 *측정*된다는 사실을 되새기십시오:

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

`_TTFA_MAX_BUFFER_SECONDS = 3.0` (`:43`). 확정된 onset 없이 audio 3초가 지나면 측정은 debug log와 함께
포기되고 **아무것도 보고되지 않습니다**. 0도 아니고, `error` level 경고도 아니고 — 조용히 사라진
line입니다. 없는 TTFA 숫자를 찾아 나서기 전에 알아 둘 가치가 있습니다.

이 chapter가 더하는 budget 관련 사항: onset detector 자신의 parameter가 이 line의 해상도를 정합니다.

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

`hop_ms = 1.0`이 onset 해상도입니다 — 그래서 `leading_silence`는 1 ms로 양자화되고, 이 시트의 다른
어떤 것보다 훨씬 곱습니다. `threshold_db = -40.0`은 *"above typical TTS noise-floor padding and below
voiced onset"*에 놓인다고 문서화되어 있고(`:358-359`), `min_voiced_ms = 50.0`은 transient를
기각합니다. 이 gate는 고정이고 provider별 튜닝이 없습니다. 유성 onset이 저에너지 자음으로 시작하는
한국어 TTS에 대해 -40 dBFS gate는 당신이 시험해 보지 않은 가정입니다. 그것은 결함이 아니라 측정
항목입니다.

### 3.6 Term 7: transport

이것이 Pipecat이 **가장 적게** instrument하는 term이고, budget을 잡기 전에 알아야 합니다.

정의되어 있는 부분은 output chunking granularity이고, [[ch-08/read]] §6이 이미 그것을 interrupt
granularity decision으로 지출했습니다. 같은 숫자, 다른 계정:

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

default에서 written chunk당 40 ms. 그것은 첫 audio가 process를 얼마나 빨리 떠날 수 있는가의 하한이고,
동시에 interrupt granularity의 하한이며, 그 둘은 같은 숫자입니다 — 그것이 trade-off입니다: 줄이면
barge-in이 더 또렷해지고 syscall overhead가 오릅니다.

Lina에게 transport term에는 Pipecat이 전혀 측정하지 않는 두 번째 성분이 있습니다: telephony
serializer. [[transport-telephony]]는 Pipecat 전화 통화가 WebSocket transport 더하기
`FrameSerializer`이고, 배포되는 serializer 여섯 중 다섯이 8 kHz μ-law임을 확립합니다. 그러므로
outbound path에는 영구적인 resample 쌍 — `_input_resampler`와 `_output_resampler`, 둘 다
`create_stream_resampler(clear_after_secs=...)`에서 — 이 당신의 24 kHz TTS output과 8 kHz wire 사이에
앉아 있습니다. **serialization이나 resampling을 위한 metric class는 없고**, 어떤 serializer도
`start_processing_metrics()`를 호출하지 않습니다. 그 숫자를 원하면 직접 감싸야 합니다(§12).

그리고 진정으로 instrument되지 않은 조각: 당신의 process와 고객 단말 사이의 network transit과 jitter.
이 repository의 어떤 것도 그것을 측정하지 않습니다. `TransportTimingReport`(§9.3)는 *connection*
milestone을 측정하지, turn별 transit이 아닙니다. 사이에 CPaaS가 낀 한국 PSTN 통화에서 그것은 반올림
오차가 아니고, 시트에 적을 수 있는 유일하게 정직한 것은 *measured externally*라고 표시된 행입니다.

---

## 4. 어떤 stage가 serial이고 어떤 것이 overlap하는가

이것은 §1.2의 failure mode 1을 적용 가능한 규칙으로 바꾼 것입니다.

### 4.1 serial한 척추(spine)

이 stage들은 엄격히 순서가 있고, 각각이 앞의 것을 기다리며, 그 안의 모든 millisecond가 고객이
기다리는 millisecond입니다:

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

serial term 일곱 개. 그 합이 budget입니다.

### 4.2 overlap하는 것들, 따라서 비용이 0인 것들

비용처럼 보이지만 아닌 것 넷:

**(a) 첫 가청 sample 이후의 TTS synthesis.** [[ch-07/read]] §1.1. playback은 1×로 돌고, 나머지
synthesis는 그 아래에 숨습니다. 그래서 TTS *service*가 총량으로 느린 것은 완전히 보이지 않을 수
있는데, 그 `leading_silence`는 재앙일 수 있습니다.

**(b) 두 번째 이후 문장의 aggregation.** §3.4. 같은 이유이고, observer가 첫 measurement만 남기는
이유입니다.

**(c) Smart-turn ONNX inference.** §3.1. 절대적 STT deadline 안으로 흡수됩니다. STT P99 window를
넘기지 않는 한 공짜입니다.

**(d) observer plane 자체.** §0.3. `WorkerObserver`는 enqueue하고, observer들은 자기 task에서 돕니다.
database에 쓰는 당신의 custom `LatencyBudgetObserver`는 turn을 늘리지 않습니다 — queue를 늘립니다.

### 4.3 둘 다 아닌 것: function call

tool call은 serial *이면서* LLM term을 배가합니다. 순서는: 첫 inference가 tool call을 만들고
(TTFAT #1) → function이 실행되고(`FunctionCallMetrics.duration_secs`) → 두 번째 inference가 답을
만들고(TTFAT #2) → 그다음 aggregation과 TTS. 그 모든 부분이 critical path 위에 있습니다.

observer는 `tool_call_id`로 frame 두 개를 짝지어 중간 조각의 시간을 잽니다:

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

Lina에게 이것은 당신을 놀라게 할 line입니다. 보험사 API를 상대로 400 ms 걸리는 견적 조회는 고객의
침묵 400 ms이고, 그것이 inference 하나가 아니라 둘 위에 얹힙니다. P95가 터지고 있다면 model을 보기
전에 여기를 보십시오.

---

## 5. 왜 LLM이 지배하는 term인 경우가 드문가

[[latency-budget-voice]]의 core insight에 있는 주장은 *"the two biggest terms are not the LLM"*
입니다. 믿음으로 받아들이지 말고, 이 course가 검증한 상수로 산술을 하십시오.

### 5.1 실제 숫자와 정직한 빈칸 하나로 계산한 P50

default turn-analyzer stop strategy, Deepgram STT, 빠른 non-reasoning LLM, sentence aggregation 켬,
적당한 silence padding을 가진 streaming TTS, 그리고 WebSocket telephony transport를 가정합니다.

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

rule line이 아직 빈 채로 이미 당신의 P50 = 1.0 s target을 넘었습니다. 그것이 정직한 출발 위치이고,
§3.2의 slot이 그토록 중요한 이유입니다.

이제 질량이 어디에 있는지 보십시오. LLM은 1.16 중 0.35 — **30 %**입니다. endpointing floor 더하기
aggregation 더하기 TTFA가 0.75 — **65 %**입니다. LLM term을 절반으로 줄이면(더 작은 model, 더 나은
provider, prompt caching — 진짜로 어려운 engineering project입니다) 0.175 s를 삽니다.
`text_aggregation_mode`를 `TOKEN`으로 바꾸면(문서화된 품질 caveat이 딸린 한 줄짜리 config 변경,
ch-07 §4.1) 대략 같은 양을 삽니다. leading silence가 300 ms인 TTS를 고치면 그 둘 중 어느 것보다 더
많이 삽니다.

### 5.2 같은 budget을 P95에서

이제 가정 하나를 바꿉니다: 안전망이 만료되기 전에 transcript가 finalize되지 않습니다. Deepgram이면
`effective_stt_wait = max(0, 0.35 − 0.2) = 0.15 s`. 총합은 ~1.31 s가 됩니다. rule line이 빈 채로,
여전히 당신의 P95 = 1.5 s 안입니다.

provider를 xAI로 바꿉니다: `effective_stt_wait = max(0, 2.14 − 0.2) = 1.94 s`, 총합 ~3.10 s. 당신의
P95 target은 두 배 차이로 사라졌고, **당신의 LLM에 대해서는 아무것도 바뀌지 않았습니다.**

그것이 논증 전부이고, [[ch-06/read]]가 provider 선택을 여기가 아니라 ch-06에 둔 이유입니다. 당신의
꼬리를 지배하는 lever는 constructor의 keyword argument입니다.

### 5.3 reasoning-model 함정에 값을 매기면

objection handling을 개선하려고 reasoning model을 켠다고 합시다. `TTFBMetricsData.value`는 0.22 s를
보고할 수도 있습니다 — 이전보다 *더 좋은* 값입니다. model이 reasoning을 빠르게 뱉기 시작하기
때문입니다. 그동안 `TTFATMetricsData.thinking_time`은 0.9 s이고 고객은 LLM term만으로 1.12 s를
기다립니다.

당신의 dashboard가 `ttfb`를 그린다면 — 그것이 `MetricsLogObserver`가 가장 눈에 띄게 출력하는 것이고,
RTVI의 `"ttfb"` bucket이 나르는 것이며, `SentryMetrics`가 transaction으로 보내는 것입니다(§8) — 그
regression은 **보이지 않고 개선처럼 보입니다**. `ttfat`이 그것을 잡아내는 유일한 field이고, §8.3이
세 exporter 중 어느 것이 그것을 나르는지 정확히 보여줍니다.

figure의 TTFB 대 TTFAT 토글을 특히 이 시나리오에 써 보십시오. 당신이 그리는 metric의 *이름*이 design
decision이라는 것을 스스로 납득하는 가장 빠른 방법입니다.

> 💡 **쉬운 설명 — 왜 "이름"이 design decision인가요?**
> 같은 turn에 대해 세 숫자가 동시에 존재합니다: `ttfb`(첫 출력), `ttfat`(첫 *답변* token),
> `ttfa`(첫 *가청* sample). 셋 다 "빠르기"를 재는 것처럼 들리지만, 셋 중 고객의 체감과 같은 것은
> `ttfa`뿐이고 나머지는 그것의 부분 합입니다. 대시보드는 세 개 중 하나를 골라야 하고, `ttfb`를 고른
> 순간 reasoning model 도입은 *지표상 개선*으로 기록됩니다. metric 이름을 고르는 일은 알림
> threshold를 고르는 일보다 앞서고, 더 중요합니다.

---
## 6. observer plane

위의 모든 것은 그 숫자들이 존재한다고 가정했습니다. 이 section이 그것들이 존재하는 이유입니다.

### 6.1 주장을 정확히 진술하면

Pipecat의 instrumentation은 logging convention이 아닙니다. 그것은 **frame graph 위에 얹힌 두 번째의
read-only plane**입니다: 모든 processor 쌍 사이의 모든 frame transfer를 보는 것인데, processor가
아니면서, 무엇과도 인접(adjacent)하지 않으면서, 자기가 본 것을 수정할 수 없으면서 그렇게 합니다.

그 property — **non-adjacency**라고 부릅시다 — 는 지금 이름을 붙여 둘 가치가 있습니다.
[[ch-12/read]]가 그것에 직접 손을 뻗기 때문입니다. [[ch-01/read]]는 Pipecat의 splice algebra가
작동하는 이유가 `FrameProcessor`가 uniform interface를 제시하고 processor들이 doubly-linked list로
link되기 때문임을 가르쳤습니다. pipeline에 *참여*하고 싶은 모든 것은 그 list에서 position을 차지해야
합니다. observer plane은 position을 차지하지 않고 지켜볼 수 있는 유일한 mechanism입니다. boson의
`SignalQueue`([[boson-layers-rules]], `layers/signals.py`) — 이후의 어떤 layer든
`get_recent(seconds, source_layer=None)`로 읽는 append-only log — 는 정확히 그 property를 필요로 하고,
그것이 나오는 곳이 여기입니다.

> 💡 **쉬운 설명 — non-adjacency가 왜 특별한 성질인가요?**
> pipe-and-filter에서 어떤 것을 보려면 보통 **그 자리에 서야** 합니다. STT와 LLM 사이의 frame을 보고
> 싶으면 둘 사이에 processor를 끼워야 하고, 그 순간 당신은 관찰자가 아니라 참가자가 됩니다 — 지연을
> 더하고, frame을 떨어뜨릴 수 있고, 순서를 바꿀 수 있습니다. observer plane은 "모든 간선(edge)을 동시에
> 보는 자리"를 만들어 줍니다. 그래서 boson의 signal log처럼 **여러 layer가 서로의 활동을 뒤늦게
> 참조해야 하는 구조**를 pipeline 위상을 건드리지 않고 얹을 수 있습니다.

### 6.2 contract: hook 네 개, event dataclass 세 개

base class 전체가 142줄이고, 숨겨진 것이 없습니다.

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

`FramePushed`는 **양쪽 끝점을 모두** 지니고 다닙니다. 그것이 두 event의 차이이고 둘 다 존재하는 이유
전부입니다: `FrameProcessed`는 *어떤 processor가 이 frame을 다루고 있다*를 말해 주고, `FramePushed`는
*이 frame이 A에서 B로 움직이고 있다*를 말해 줍니다. latency observer는 두 번째를 원합니다. hop은
node가 아니라 edge이기 때문입니다.

세 번째 event는 startup에 관한 것이고, 그 docstring이 당신이 걸려 넘어질 clock의 미묘함을 설명합니다:

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

그리고 base class 자체:

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

hook 네 개, 전부 `async`, base에서는 전부 `pass`: `on_process_frame(FrameProcessed)` (`:99`),
`on_push_frame(FramePushed)` (`:111`), `on_processor_setup(ProcessorSetUp)` (`:123`),
`on_pipeline_started()` (`:136`). subclass하고, 필요한 것 하나를 override하고, 나머지는 무시하십시오.
registration도, filtering API도, frame type별 subscription도 없습니다. 전부를 받고 `isinstance`로
직접 분기합니다.

### 6.3 hook이 실제로 발화하는 곳

`frame_processor.py`의 call site 둘이고, 그 둘을 읽는 것이 plane이 read-only라고 *믿는 것*과 *아는
것*의 차이입니다.

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

저 block 하나에 두 가지가 있습니다. 첫째: observer는 frame에 대해 행동이 취해지기 **전에** 통지받고,
observer는 frame object 자체를 받습니다 — 그것을 *변형할 수 있고*, 아무것도 막지 않습니다. "Read-
only"는 아무도 강제하지 않는 convention입니다. observer에서 frame을 mutate하지 마십시오. 이 plane의
가치 전부는 그 존재가 아무것도 바꾸지 않는다는 데 있습니다.

둘째, 그리고 이것은 budget 사실입니다: 모든 processor에서, 모든 `InterruptionFrame`마다
`await self.stop_all_metrics()`. 취소된 turn이 열어 둔 TTFB / processing / text-aggregation window는
바로 거기서 닫힙니다. 중단된 turn이 반쯤 열린 measurement를 다음 turn으로 흘리지 않습니다.

push site:

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

`await observer.on_push_frame(data)`는 `await self._next.queue_frame(...)` **이전에** 일어납니다.
§0.3에 따라 그 await은 queue put으로 귀결되므로, 비용은 frame transfer마다 등록된 observer마다 enqueue
하나입니다. 싸지만 0은 아닙니다 — 그리고 *모든* frame에 대해 일어납니다. 양방향의 40 ms audio chunk
하나하나까지 포함해서.

`timestamp`는 **pipeline clock**입니다:

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

pipeline이 시작된 이후의 monotonic nanoseconds. §10.3에서 이것을 기억하십시오.
`UserBotLatencyObserver`가 그것을 쓰지 않기 때문입니다.

### 6.4 `can_generate_metrics()` — pipeline의 절반을 회색으로 만드는 opt-in

plane은 모든 frame을 봅니다. 그렇다고 모든 processor가 metric을 만들어내는 것은 아닙니다. gate는
base class에서 `False`를 반환하는 method입니다:

**`src/pipecat/processors/frame_processor.py:488-494`**
```python
    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False
```

모든 metric emitter가 그것에 이중으로 gate되어 있습니다:

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

`can_generate_metrics() and self.metrics_enabled` — processor가 opt in해야 하고 **그리고** pipeline이
metrics를 켠 상태여야 합니다(§0.1). 같은 이중 gate가 `frame_processor.py:504-628`의 wrapper method
열두 개를 전부 감싸고, usage method 셋은 대신 `usage_metrics_enabled`를 검사합니다.

몇 개의 processor가 opt in할까요? 세어 보십시오:

```bash
# run in wiki/raw-data/pipecat/pipecat-src at commit 0cbf9c5b
grep -rn "def can_generate_metrics" src/pipecat/ | wc -l          # 97
grep -rn -A6 "def can_generate_metrics" src/pipecat/ | grep -c "return True"    # 96
grep -rn -A6 "def can_generate_metrics" src/pipecat/ | grep -c "return False"   # 1
```

**tree에 definition 97개. 96개가 `True`를 반환. 정확히 하나가 `False`를 반환 — base class.** 그래서
실무의 규칙은: *service*는 거의 보편적으로 opt in하고, *당신이 직접 쓰는 모든 것*은 하지 않습니다.
당신의 한국어 phrase chunker, 당신의 rule-layer processor, 당신의 DTMF handler, 당신의 ledger port —
전부 기본으로 침묵하고, 전부 §12가 unexplained gap이라고 부르는 것에 기여합니다.

명단은 심지어 startup에 공표됩니다. metrics가 켜져 있으면 worker가 pipeline을 순회하며 instrument된
processor마다 값이 0인 entry 하나씩을 emit합니다:

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

한 번만 보내지고, `enable_metrics and send_initial_empty_metrics`에 gate됩니다(`worker.py:1232`).
그러므로 RTVI client는 첫 turn 전에 instrument된 processor의 정확한 명단을 배웁니다 — 그리고 그
list에 없는 processor는 나중에도 절대 나타나지 않습니다. 그 list가 곧 "이 pipeline에서 나는 무엇을
측정할 수 있는가?"에 대한 답이고, 세 줄이면 출력할 수 있습니다.

### 6.5 metric은 frame으로 이동하고, frame class가 중요하다

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

`MetricsFrame(SystemFrame)`. [[ch-04/read]] §4와 [[ch-08/read]] §3에 따라 `SystemFrame`은 취소
가능한 process task로 넘겨지는 대신 input task에서 처리됩니다 — 즉 barge-in이 도착한 시점에 비행
중이던 metrics frame은 그 turn의 나머지 작업과 함께 **버려지지 않습니다**. turn의 measurement가 그
turn의 취소를 살아남습니다. [[ch-08/read]]가 cascade가 얼마나 공격적으로 작업을 버리는지 가르쳤다는
점을 고려하면, 이것은 의도된 예외이고 좋은 예외입니다: 취소된 turn이야말로 정확히 그 숫자를 원하는
turn입니다.

`FrameProcessorMetrics`(`processors/metrics/frame_processor_metrics.py:31`, 390 L)가 모든 timer를
소유하고 모든 method가 `MetricsFrame`을 push하는 대신 *반환*합니다 — push할지는 `FrameProcessor`
wrapper가 결정합니다. 그 분리가 `SentryMetrics`(§8.4)를 drop-in subclass로 가능하게 만드는
것입니다.

### 6.6 번들된 observer들

`src/pipecat/observers/`에 concrete observer 셋과 base가 있고, `observers/loggers/`에 넷이 더 있으며,
하나는 package 밖 `pipeline/worker.py`에 삽니다:

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

`TurnTrackingObserver`의 turn 경계는 한 문단 값어치가 있습니다. latency observer의 경계와 *같지
않기* 때문입니다:

**`src/pipecat/observers/turn_tracking_observer.py:36-41`**
```python
    - The first turn starts immediately when the pipeline starts (StartFrame)
    - Subsequent turns start when the user starts speaking
    - A turn ends when the bot stops speaking and either:

      - The user starts speaking again
      - A timeout period elapses with no more bot speech
```

`turn_end_timeout_secs=2.5` (`:52`)와 함께. 그러므로 이 observer의 "turn duration"은 bot의 발화 전체와
최대 2.5 s의 뒤따르는 침묵을 포함합니다. 그것은 *conversation-structure* metric이지 latency metric이
아닙니다. budget 시트에 올리지 마십시오.

---

## 7. `LatencyBreakdown`: typed object로서의 budget

여기가 plane과 budget이 한 몸이 되는 곳입니다. `UserBotLatencyObserver`는 350줄이고 §3이 기술한
decomposition을 정확히 만들어냅니다.

### 7.1 그 object

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

§3에 그대로 대응시키면:

| `LatencyBreakdown` field | Budget terms it covers |
|---|---|
| `user_turn_secs` | §3.1 — `stop_secs` + STT finalization + turn-analyzer wait, **as one fused number** |
| `ttfb: list[...]` | §3.3 (LLM) and §3.5's `ttfb` half, one entry per instrumented service, each tagged with `processor` and `model` |
| `text_aggregation` | §3.4, first measurement only |
| `function_calls` | §4.3 |
| — | §3.2 rule evaluation: **absent unless you emit it.** §12. |
| — | §3.5's `leading_silence`: **absent.** §7.4. |
| — | §3.6 transport: **absent.** |

저 표가 이 instrument의 정직한 상태입니다. 일곱 serial term 중 넷을 덮고, 그중 하나는 일부만 덮으며,
남은 셋 중 둘은 당신이 직접 더해야 하는 것들입니다. 비판이 아닙니다. 당신이 그것에 맞춰 만들어야 할
specification입니다.

또한 `user_turn_secs`가 §3의 term 두 개를 하나로 **융합(fuse)**한다는 점에 유의하십시오. docstring이
그렇게 말합니다 — *"This includes VAD silence detection, STT finalization, and any turn analyzer
wait."* dashboard에서 `stop_secs`와 안전망을 분리하고 싶다면 observer가 대신 해 주지 않습니다. 알고
있는 `stop_secs`를 직접 빼거나, `e2e_processing_time_ms`를 위해 `TurnMetricsData` listener를 다십시오
(`metrics.py:206-218`, 이것은 *"from VAD speech-to-silence transition to turn completion"*으로
측정됩니다).

### 7.2 두 개의 clock edge

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

`self._user_stopped_time = data.frame.timestamp - data.frame.stop_secs` — [[ch-06/read]] §9가
`STTService` 안에서 보여준 것과 같은 뺄셈인데, 여기서 독립적으로 계산됩니다. 두 subsystem이 같은
frame field들로부터 같은 되감기를 합니다:

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

frame이 자기가 어떤 parameter 아래에서 결정되었는지를 지니고 다닙니다. 훔쳐 갈 만한 design 원칙이
그것입니다: **measurement event는 자기가 측정된 configuration을 함께 지니고 다녀야 한다.** 그래야
consumer가 config를 읽지 않고도 정규화할 수 있습니다. 당신의 rule-layer processor의 metric도 같은
일을 해야 합니다.

닫는 edge:

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

greeting measurement도 눈여겨보십시오: `on_first_bot_speech_latency`는 `ClientConnectedFrame`부터 첫
`BotStartedSpeakingFrame`까지를 **한 번** 재고, 고객이 먼저 말하면 포기됩니다(`:244`). outbound
tele-sales 통화에서 그것은 진짜 product 숫자입니다 — 고객이 전화를 받은 뒤 Lina가 말하기
시작하기까지 얼마나 걸리는가 — 그리고 event handler 하나 거리에 있습니다.

### 7.3 무엇이 reset되고, 무엇이 의도적으로 reset되지 않는가

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

무엇을 지우는지 목록을 읽고, 더 중요하게는 무엇을 지우지 *않는지*를 읽으십시오. `_user_stopped_time`은
거기에 **없습니다**. 오직 `VADUserStartedSpeakingFrame`만 그것을 지웁니다(`:238`).

그래서 barge-in에서: service별 breakdown은 버려지지만, top-level clock은 *원래의* 발화 종료 시점부터
계속 돕니다. 버그일까요? 아닙니다 — 추적해 보십시오. barge-in이 도착하는 이유는 고객이 말하기
시작했기 때문이고, 그것은 `VADUserStartedSpeakingFrame`이 이미 관찰되었다는 뜻이며(그것이 애초에
interruption을 발화시킨 것입니다, [[ch-08/read]] §1), *그* handler가 `_user_stopped_time`을 `None`으로
지웠습니다. `InterruptionFrame` branch는 다른 interruption source들을 위해 존재합니다 —
[[ch-08/read]]가 originator 아홉 개를 세었습니다 — 그 경우들에는 앞선 VAD start가 없습니다. 그런
경우 `_user_stopped_time`을 유지하는 것이 옳습니다: 고객은 자기가 끝낸 발화에 대한 답을 여전히
기다리고 있고, 시계는 여전히 돌아야 합니다.

미묘하고, 한 번 추적해 볼 가치가 있습니다. 그것이 "취소된 cycle이 숫자를 오염시키지 않는다"(참,
breakdown에 대해)와 "취소된 cycle은 측정되지 않는다"(거짓, 그리고 참이기를 원하지도 않을 것)의
차이이기 때문입니다.

> 💡 **쉬운 설명 — 왜 `_user_stopped_time`만 남기나요?**
> 두 종류의 interruption을 구분하면 됩니다.
> (1) **고객이 말을 끊은 경우**: VAD start가 먼저 왔고, 그 handler가 이미 시계를 초기화했습니다.
> 새 turn이 시작되었으니 옛 시계는 의미가 없습니다.
> (2) **다른 이유로 turn이 취소된 경우**(함수 오류, 서버 신호 등): 고객은 여전히 아까 한 말에 대한
> 답을 기다리고 있습니다. 이때 시계를 리셋하면 "고객이 실제로 기다린 시간"을 잃어버립니다.
> `InterruptionFrame` branch가 세부 breakdown만 지우고 top-level 시계는 두는 이유가 이것입니다.
> 코드가 (1)을 명시적으로 처리하지 않고, (1)에서는 이미 다른 handler가 처리했다는 사실에 의존한다는
> 점이 이 설계의 미묘한 부분입니다.

frame deduplication도 있습니다. 같은 frame이 여러 hop에서 관찰되기 때문입니다:

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

downstream만, 그리고 id dedup을 위한 bounded `deque(maxlen=100)` + `set` 쌍. 필요합니다.
`on_push_frame`은 frame당 한 번이 아니라 *hop*당 한 번 발화하므로, processor 여덟 개를 지나는 frame은
그러지 않으면 여덟 번 세어질 것이기 때문입니다. hop이 아니라 frame에 반응하는 자기만의 observer를
쓴다면 이 pattern을 복사하십시오 — 네 줄이고, 손으로 만든 observer에서 가장 흔한 단 하나의
버그입니다.

### 7.4 headline 숫자의 의미를 바꾸는 정직한 한계

이 section에서 가장 중요한 발견이 여기 있고, 어떤 excerpt에도 없습니다 — output transport를 열어
보아야 나옵니다.

`on_latency_measured`는 `BotStartedSpeakingFrame`에서 멈춥니다. 그 frame은 어디서 올까요?

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

순서대로 세 가지 사실:

1. `_handle_frame(frame)`이 `_internal_write_audio_frame(frame)` **이전에** 돕니다. 그러므로
   `BotStartedSpeakingFrame`은 byte가 transport로 가기 *전에*, network는 말할 것도 없이, push됩니다.
2. TTS 경로(`TTSAudioRawFrame`)에서는 첫 chunk에 대해 `_bot_currently_speaking()`이 **무조건**
   호출됩니다. silence check가 없습니다. silence check는 speech-to-speech 경로에만 있고,
   `_maybe_bot_currently_speaking` → `if not is_silence(frame.audio)`를 통합니다(`:785-787`).
3. 따라서 `BotStartedSpeakingFrame`은 **output transport에서 dequeue된 TTS audio의 첫 byte에,
   그 byte가 침묵일 때에도** 발화합니다.

여기서 결론이 나옵니다:

> **`on_latency_measured`는 첫 *가청 sample*이 아니라 첫 *byte*까지를 재고, network write 이전에
> 멈춥니다.** 그것은 `leading_silence`(§3.5)를 빠뜨리고 transport(§3.6)를 빠뜨립니다. 당신의
> `CLAUDE.md` 정의 — *"to the first audible assistant sample"* — 에 비추어 그것은 체계적으로
> **낙관적(optimistic)**입니다.

얼마나? 정확히 `TTFAMetricsData.leading_silence` 더하기 당신의 transport transit만큼. 둘 다 알 수
있습니다: 첫째는 이미 측정되어 있고, 당신의 observer가 이미 받고 있는 `MetricsFrame` 안에 앉아
있습니다 — 단지 `_handle_metrics_frame`이 축적하는 두 type 중 하나가 아닐 뿐입니다
(`:324-341`이 `TTFBMetricsData`와 `TextAggregationMetricsData`를 다루고, 그 외에는 아무것도 다루지
않습니다). 둘째는 어떤 것도 측정하지 않습니다.

**이것은 ~20줄짜리 수정이고 당신의 첫 framework-extension move입니다(§13.1).**
`UserBotLatencyObserver`를 subclass하고, `TTFAMetricsData`를 축적하고,
`on_latency_measured + leading_silence`를 1.0 s / 1.5 s에 대고 볼 숫자로 보고하십시오. 그렇게 하기
전까지 당신의 dashboard와 당신의 `CLAUDE.md`는 서로 다른 두 interval을 재고 있고, dashboard가 져야 할
논쟁을 이기고 있습니다.

figure의 TTFB 대 TTFA 토글이 정확히 이 gap을 waterfall 끝의 별도 빗금 구간으로 그립니다. 한 번
토글하고 bar가 P50 선을 얼마나 넘어가는지 적어 두십시오.

### 7.5 `chronological_events()`

유일한 편의 method이고, log line에 맞는 shape입니다:

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

TTFB와 aggregation entry의 `start_time`이 어떻게 재구성되는지 보십시오 — `MetricsFrame`이 관찰되는
순간의 `now - metrics_data.value`입니다(`:329`, `:339`). 기록된 원점이 아니라 역산된 원점입니다.
waterfall의 순서를 매기기에는 충분하고, process를 넘나들며 correlate하기에는 충분하지 않습니다. 그것이
§10.3의 주제입니다.

---

## 8. 숫자는 어디로 가는가: 네 개의 consumer, 그리고 각자가 흘리는 것

같은 `MetricsFrame` stream이 sink 넷을 먹입니다. 그들은 같은 정보를 나르지 않고, 그 차이가 배워야 할
것입니다.

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

여덟 type이 나열되어 있습니다. `TextAggregationMetricsData`는 그중에 없고, 이 파일은 그것을 import
조차 하지 않습니다. `include_metrics: set[type[MetricsData]] | None`으로 필터링 가능합니다(`:66-68`).

### 8.2 client wire — `RTVIObserver`

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

bucket 일곱 개: `ttfb`, `ttfa`, `ttfat`, `processing`, `tokens`, `stt_usage`, `characters`. 기본으로
켜져 있지만(`RTVIObserverParams.metrics_enabled: bool = True`, `observer.py:178`) —
`PipelineParams.enable_metrics=True`가 되기 전까지는 아무것도 나르지 않습니다(§0.1). 그전에는
`MetricsFrame`을 만드는 것이 아무것도 없기 때문입니다.

bucket에 들어가지 않는 것: `TextAggregationMetricsData`, `TurnMetricsData`. 둘 다 chain을 조용히
통과해 떨어집니다.

### 8.3 gap에 이름 붙이기

exporter 셋을 metric class 여덟 개에 대해 줄 세우면:

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

두 행이 당신의 주목을 받을 만합니다.

**`TextAggregationMetricsData`는 정확히 하나의 consumer에게 도달하고**, 그 하나가 기본으로 꺼져 있는
것입니다(§0.2). 200–300 ms이고 config 변경 하나로 훨씬 작아질 수 있는 당신의 §3.4 budget line은
budget 전체에서 가장 관측 불가능한 숫자입니다.

**`TTFAMetricsData`는 log와 client에는 도달하지만 latency observer에는 도달하지 않습니다.** 그것이
§7.4를 반대 방향에서 다시 말한 것입니다: leading-silence 숫자는 존재하고, wire 위에도 존재하는데,
당신의 headline latency를 계산하는 그것이 그것을 읽지 않습니다.

### 8.4 Sentry, 그리고 drop-in exporter의 대가

**`src/pipecat/processors/metrics/sentry.py:25-31`**
```python
class SentryMetrics(FrameProcessorMetrics):
    """Frame processor metrics integration with Sentry monitoring.

    Extends FrameProcessorMetrics to send time-to-first-byte (TTFB) and
    processing metrics as Sentry transactions for performance monitoring
    and debugging.
    """
```

service마다 `metrics=SentryMetrics()`로 넘깁니다(`examples/observability/
observability-sentry-metrics.py:69,77,82` 참조). 정확히 네 개의 method를 override합니다 —
`start_ttfb_metrics` (`:71`), `stop_ttfb_metrics` (`:93`), `start_processing_metrics` (`:112`),
`stop_processing_metrics` (`:129`) — 그리고 그 외에는 없습니다. 그러므로 Sentry로 instrument된 Lina는
TTFB와 processing time을 보고하고 TTFA, TTFAT, text aggregation, 그리고 모든 usage metric에
장님입니다. §5.3에 따라 그것은 정확히 reasoning-model regression이 개선으로 읽히는 configuration
입니다.

### 8.5 OpenTelemetry

`utils/tracing/`이 span machinery를 담고 있습니다: `setup_tracing(service_name, exporter,
console_export)`, `is_tracing_available()`, 그리고 conversation span 아래에 turn마다 span 하나를
만드는 `TurnTraceObserver`. 그것의 유일한 latency attribute:

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

subscription을 읽으십시오: `TurnTraceObserver`는 latency를 스스로 계산하지 않고,
**`UserBotLatencyObserver`를 구독합니다**. 그것이 §0.2의 wiring이 스위치가 아니라 chain인 이유이고,
metrics를 켜지 않은 채 tracing만 켜면 latency attribute가 없는 span을 얻는 이유입니다.

`turn.user_bot_latency_seconds` — §7.4의 headline 숫자이고, §7.4의 모든 caveat이 붙은 채이며,
breakdown은 **아닙니다**. service 수준 span은 `@traced_llm` / `@traced_stt` / `@traced_tts`
decorator에서 나옵니다.

### 8.6 collection은 제공되고, aggregation은 당신 몫이다

**`examples/observability/`에는 정확히 파일 세 개**가 있고, 합쳐서 396줄입니다:

```
observability-observer.py         193 L
observability-sentry-metrics.py   146 L
observability-heartbeats.py        57 L
```

그중 가장 큰 것이 당신이 쓰게 될 모든 것의 shape입니다:

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

저 목록에 **없는** 것을 보십시오: `UserBotLatencyObserver`. repo 자신의 observability 예제는 latency를
instrument하지 않습니다. frame을 log할 뿐입니다.

**dashboard도, Sentry/OTel 너머의 exporter도, aggregation 예제도 tree 어디에도 없습니다.** P50/P95
계산도, histogram도, rollup도, storage도 없습니다. Pipecat은 attribution이 붙은 typed event와 전달
mechanism을 주고 거기서 멈춥니다. 당신의 `CLAUDE.md`는 *"Report P50/P95/P99"*라고 말합니다 — 그
reporting layer는 당신이 쓰는 코드이고, 가정할 것이 아니라 migration 추정치의 line item으로 올라가야
합니다.

규모 감각을 위해: `examples/observability/`는 396줄입니다. 최소한으로 정직한 Lina observability
layer — §7.4를 보정하고, turn당 한 행을 쓰고, rolling percentile을 계산하는 `LatencyBudgetObserver` —
는 대략 같은 자릿수입니다. 일주일이 아니라 하루짜리입니다. 하지만 0은 아니고, 이 repository의 어떤
것도 당신을 대신해 주지 않습니다.

---
## 9. auto-wiring, 정확하게

§0.1, §0.2, §8을 합쳐 운영상의 진술로 만듭니다.

### 9.1 기본 `PipelineWorker`가 주는 것

| Feature | Default | Effect |
|---|---|---|
| `enable_turn_tracking` | `True` | `TurnTrackingObserver` appended → `on_turn_started` / `on_turn_ended` available |
| `enable_rtvi` | `True` | `RTVIProcessor()` constructed and prepended; `create_rtvi_observer(...)` appended |
| `enable_tracing` | `False` | no `UserBotLatencyObserver`, no `TurnTraceObserver`, no spans |
| `PipelineParams.enable_metrics` | `False` | **no `MetricsFrame` is ever produced** |
| `PipelineParams.enable_usage_metrics` | `False` | no token / audio-second / character counts |
| `idle_timeout_secs` | `IDLE_TIMEOUT_SECS = 300` | idle observer wired (ch-04 §9) |
| `enable_heartbeats` | `False` | `HEARTBEAT_SECS = 1.0`, `HEARTBEAT_MONITOR_SECS = 10.0` unused |

그러므로 진짜 out-of-the-box 상태는: **turn event 예, metric을 위한 RTVI transport 예, metric 자체
아니오, latency breakdown 아니오.** RTVI metric channel은 열려 있고 비어 있습니다. 그것은 client
쪽에서 debug하기에 혼란스러운 failure mode이고 — client는 server가 절대 보내지 않을 message type을
정확하게 구독하고 있습니다 — 오후 한나절을 태우기 전에 알아 둘 가치가 있습니다.

### 9.2 Lina를 위한 최소 wiring

네 가지이고, 전부 constructor argument입니다:

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

`report_only_initial_ttfb`는 `False`로 둡니다: 한 turn의 첫 TTFB만이 아니라 모든 TTFB를 원하기
때문입니다. tool을 호출하는 turn은 LLM inference를 두 번 하고(§4.3), 첫 번째만 보고하면 두 번째가
숨습니다.

### 9.3 cold start는 별개의 observer다

`StartupTimingObserver`는 다른 질문에 답합니다 — "turn이 얼마나 빠른가"가 아니라 "첫 turn이 가능해질
때까지 얼마나 걸리는가":

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

그리고 `StartupTimingReport.total_duration_secs`는 *"Processors are set up concurrently, so
this is the span rather than the sum of what each cost"*로 문서화되어 있습니다(`:100-103`) — §4.2와
같은 overlap 규율을 startup에 적용한 것입니다. 그리고 connection milestone을 위한
`TransportTimingReport(bot_connected_secs, client_connected_secs)`.

outbound tele-sales에서 이것은 각주가 아닙니다. 고객이 전화를 받으면 pipeline은 이미 warm해야 하고,
service별 `setup_duration_secs`가 어느 provider connection이 고객을 기다리게 만드는지 알려 줍니다.
`on_first_bot_speech_latency`(§7.2)와 짝지으면 greeting 경로가 완전히 회계 처리됩니다.

---

## 10. 세 가지 정직한 한계

### 10.1 이 repository에 perceptual target은 존재하지 않는다

§2.2에서 grep으로 확립했습니다. 여기서 다시 진술하는 이유는 이런 것이 조용히 통설(folklore)이 되기
때문입니다: design document에 "Pipecat은 sub-800 ms를 목표한다"라고 쓰면, 당신은 주장을 발명하는
것입니다. 대신 "우리 CLAUDE.md는 P50 ≤ 1.0 s / P95 ≤ 1.5 s를, 마지막 유성 sample부터 첫 가청
sample까지로 측정해 목표한다"라고 쓰십시오. 그것에는 주인과 정의가 있습니다.

### 10.2 collection은 제공되고, aggregation은 아니다

§8.6에서 확립했습니다. 예제 파일 셋, 396줄, dashboard 없음, percentile 계산 없음, storage 없음.
reporting layer를 budget에 넣으십시오.

### 10.3 instrument에는 error bar가 있고, 그것은 systematic하다

세 개의 별개 오차 원인이고, 전부 검증 가능하며, 전부 같은 방향을 가리킵니다:

**(a) 두 개의 clock.** `FramePushed.timestamp`는 pipeline clock입니다 — pipeline 시작 이후의 monotonic
nanoseconds(`system_clock.py:30-38`). `UserBotLatencyObserver`는 자기 duration을 전부 `time.time()`으로
계산합니다 — wall clock이고 monotonic이 아닙니다 — `:232`, `:256`, `:263`, `:273`, `:288`, `:293`,
`:322`에서. 그것은 `data.timestamp`를 한 번도 읽지 않습니다. 그래서 plane은 monotonic clock을
제공하는데 headline latency observer가 그것을 쓰지 않습니다. wall clock이 튀는 기계에서(통화 중
NTP 보정), measurement는 그 튄 만큼 틀릴 수 있습니다. `stop_ttfb_metrics`는 음수 경우를 명시적으로
방어합니다 — *"a wall clock that stepped backwards mid measurement"*
(`frame_processor_metrics.py:163-166`) — 저자들이 그것을 겪어 봤다는 뜻입니다.

**(b) observer queue lag.** §0.3에 따라, observer의 `time.time()`은 event가 push된 때가 아니라
*dequeue*될 때 돕니다. 부하가 걸리면 observer가 계산하는 모든 duration이 queue lag만큼 **위로**
편향됩니다. queue는 maxsize 없는 `asyncio.Queue()`이므로(`worker_observer.py:175`) lag은 무제한이고,
drop policy도 없고, lag 자체에 대한 metric도 없습니다. 뒤처진 observer는 뒤처졌다고 보고하는 대신
메모리를 키우고 숫자를 왜곡합니다.

**(c) 역산된 start time.** §7.5: `start_time = now - metrics_data.value`. waterfall의 모든 TTFB와
aggregation bar의 원점은 기록된 것이 아니라 자기 duration으로부터 추론된 것입니다. 순서를 매기기에는
괜찮고, process 간 correlation에는 괜찮지 않습니다.

이 중 어느 것도 instrument를 쓸모없게 만들지 않습니다 — 그것을 **알려진 bias를 가진 instrument**로
만들 뿐이고, 그것은 알려지지 않은 bias를 가진 손수 만든 `time.perf_counter()`보다 엄격히 낫습니다.
다만 alert threshold를 1.0 s로 잡을 때, 당신이 alert하는 숫자가 `leading_silence + transport`만큼
낙관적이고(§7.4) observer lag만큼 비관적이며, 그 두 오차가 어떤 원칙적인 방식으로도 상쇄되지 않는다는
것을 아십시오. §13.1의 보정이 첫째를 고칩니다. 둘째는 queue depth를 지켜보며 monitor합니다.

> 💡 **쉬운 설명 — 두 오차가 "상쇄되지 않는다"는 게 왜 중요한가요?**
> "하나는 작게 재고 하나는 크게 재니 대충 맞겠지"라고 생각하기 쉽습니다. 아닙니다. 낙관 오차
> (`leading_silence` + transport)는 **TTS vendor와 network에 의존**하고 부하와 무관하게 거의 일정합니다.
> 비관 오차(queue lag)는 **부하에 의존**해서 바쁠 때만 커집니다. 그래서 한가할 때는 숫자가 실제보다
> 작고, 바쁠 때는 두 오차가 어느 쪽으로 기울지 예측할 수 없습니다. 정확히 P95를 봐야 하는 상황
> (= 바쁠 때)에서 오차 방향을 모르게 되는 것이 문제입니다. 먼저 §13.1로 상수 오차를 제거하고,
> 남은 변동 오차만 queue depth로 감시하는 것이 순서입니다.

---

## 11. 두 boson stack의 현재 위치

mechanism만입니다. 채점 없음 — [[ch-13/read]]가 그것을 소유합니다.

### 11.1 boson-agent의 gateway

[[rtvi-observability]]와 [[boson-gateway-server]]에 따르면, gateway에는 timing 코드가 두 조각 있고
둘 다 budget이 아닙니다:

1. `packages/gateway/gateway/debug/log_decorator.py` — call 주위의 `time.perf_counter()`,
   `[TRACE …] EXIT (…ms)`로 출력. debug 보조 도구입니다. call별이고, aggregate되지 않고, typed가
   아니고, turn에 귀속되지 않습니다.
2. `bootstrap.py:222`에서 (`(_time.monotonic() - started_at) * 1000`) barge-in 경로로 실려 들어가는
   임시 `elapsed_ms` — `core.py:166 should_interrupt(session_id, content, elapsed_ms)` →
   `InterruptHandler.check_barge_in` → `policy.evaluate(...)`, 여기서
   `DurationPolicy(min_ms=500)`이 agent streaming 500 ms 이후에만 barge-in을 허용합니다
   ([[boson-interrupt-subsystem]]). 그것은 measurement가 아니라 *policy input*입니다: 한 번 읽히고,
   threshold와 비교되고, 버려집니다.

이 chapter에 대응시키면: `elapsed_ms`는 "agent가 streaming을 시작한 이후의 시간"이고, 그것은 LLM
term의 접미(suffix)일 뿐 그 이상이 아닙니다. `user_turn_secs`의 등가물도, service별 attribution도,
`MetricsFrame`도, aggregation도 없습니다. `gateway/layers/status.py`(`AgentStatusTracker`,
generating/settling/idle)가 `TurnTrackingObserver`의 구조적 대응물입니다 — turn-phase state machine —
그리고 그것처럼 latency가 아니라 conversation structure를 만들어냅니다.

그리고 [[ch-12/read]]에 중요한 것: **`gateway/layers/`와 `gateway/rules/`는 오늘 아무 metric도
emit하지 않습니다.** 완성된 utterance와 LLM 사이에 serial하게 앉은 1,206 LOC([[boson-layers-rules]])이
통째로 측정되지 않습니다. 그것이 boson 쪽의 §3.2 slot입니다: 숫자를 모른다는 것이 아니라, 알
mechanism이 존재하지 않는다는 것입니다.

### 11.2 realtime_voice

[[rtv-vs-pipecat-gap]]와 [[rtv-webrtc-transport]]에 따르면, timing에 인접한 것이 셋 존재합니다:

1. event 위의 `provider_latency_ms` / `endpoint_latency_ms` field. [[rtv-vad-chunking]]에 따르면
   `endpoint_latency_ms = self._silence_frames * frame.duration_seconds * 1000` (`energy.py` L79,
   `silero.py` L89)이고, 거기서 *"the only self-measured latency in the VAD layer"*로 기술됩니다 —
   VAD 자신이 소비한 침묵이 얼마인지에 대한 자체 재구성, 즉 §3.1의 `stop_secs` term인데,
   `VADUserStoppedSpeakingFrame.stop_secs`처럼 frame에 실려 다니는 대신 VAD 안에서 계산됩니다(§7.2).
2. WebRTC data channel로 fan-out되는 `VoiceEvent` stream(`VoiceEventKind` 값 14개) — metrics plane이
   아니라 event log입니다. `MetricsData` type도 없고 aggregation도 없습니다.
3. `BoundedAudioOutput.discarded_frames` — *"exposed but never read by anything."*

그리고 budget에 직접 귀결이 있는 구조적 사실 둘. 첫째, [[rtv-pipeline-session]]은 realtime_voice가
구성상 없는 것들 가운데 *"any observer plane, any frame-level metrics"*를 기록합니다: `Frame` base
class도 없고 processor abstraction도 없으니, `FramePushed` event가 기술할 edge 자체가 없습니다.
둘째, [[rtv-webrtc-transport]]는 resample 지점 둘(browser Opus → 16 kHz 입력, 24 kHz → 48 kHz 출력)이
*"neither measured"*이면서 둘 다 critical path에 앉아 있다고 적습니다 — 다른 codebase에서의 같은
§3.6 사각지대입니다.

셋째, 단지 측정되지 않은 것이 아니라 *구조적으로* 다른 term:
`OpenAICompatibleUnaryASR`은 발화 전체를 WAV로 buffering하고 `finalize()`에서
`audio.transcriptions.create`를 한 번 합니다(`openai_compat.py` L194-242, `timeout_seconds=1.5`).
§3.1의 회계로 보면 그것은 `max(0, TTFS_p99 − stop_secs)` — 보통 finalize된 transcript에 의해 취소되는
tail term — 을 **VAD stop 이후 매 turn마다 지불되는 완전한 transcription round-trip**으로 대체하고,
finalize할 interim 결과가 없으므로 short-circuit도 불가능합니다. 시트 위의 다른 line이고 분포도
다릅니다. 여기까지가 진술이고, 그것에 대해 무엇을 할지는 [[ch-13/read]]입니다.

### 11.3 migration이 budget에 더하는 것

boson은 오늘 server-side STT, TTS, VAD가 없습니다([[latency-budget-voice]]): turn은 client가 전달한
text partial에서 시작합니다. server-side voice pipeline으로 옮기는 것은 따라서 §3의 term 1, 2, 6을
그것들을 담고 있지 않던 budget에 **더합니다** — `stop_secs + TTFS_p99`만으로도 provider에 따라
0.55 s–2.34 s이고, TTFA는 그 위에 얹힙니다. 새 비용이고, 다른 곳에서 되찾거나 받아들여야 합니다.

*반대* 방향으로 움직이는 유일한 term은 boson의 현재 end-of-turn mechanism입니다.
[[endpointing-turn-boundary]]가 정확히 기록합니다: `PartialDetector.should_finalize
(elapsed_since_last_ms)`가 `>= 2000` — **2000 ms text-silence timer**이고, 그것이 boson end-of-turn
mechanism의 전부입니다. §3.1의 default 경로에 대고 보면, 빠른 provider로 `0.2 + max(0, TTFS_p99 − 0.2)`는
대략 0.35 s입니다. 그것이 적어 둘 산술이고, [[ch-06/read]] §15가 그 뒤의 mechanism을 줍니다.

compact/summarization 경로([[boson-compact-session]])는 turn 밖에서 돌고 이 budget에 들어오지
않습니다 — **critical path 밖에 머무는 한.** 그 단서 조항은 이제 당신이 소유하고 검사할 수 있는
design constraint입니다. 검사할 mechanism이 `start_processing_metrics()`(§12)이기 때문입니다.

---

## 12. ch-12로의 handoff, mechanism으로 진술하면

이것이 이 chapter가 rule layer에 대해 주는 유일한 지시이고, 숫자가 아닙니다.

aggregate된 transcript와 LLM service 사이에 serial하게 놓는 어떤 processor든 critical path 위에
있습니다. 그것이 metrics window를 열지 않으면 그 비용은 사라지지 않습니다 — 당신의
`LatencyBreakdown`에서 두 `TTFBBreakdownMetrics` entry 사이의 **설명 불가능한 gap**으로 다시
나타나고, 당신은 하루를 network 탓하는 데 쓰게 됩니다.

mechanism은 호출 두 개이고, `OpenAILLMService`가 자기 자신에게 쓰는 바로 그 두 호출입니다:

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

정확히 복사할 것 셋:

1. 작업 전에 `start_processing_metrics()`, **`finally`** 안에서 `stop_processing_metrics()`.
   exception이 window를 열어 둔 채로 두어서는 안 됩니다. `stop_ttfb_metrics`의 형제 guard가 존재하는
   이유가 정확히 그것입니다 — 열린 window는 다음의 무관한 output에 대해 측정되어 버립니다.
2. 당신의 processor에서 `can_generate_metrics()`가 `True`를 반환하도록 override하십시오. §6.4에 따라
   base는 `False`를 반환하므로, override 없이는 두 호출 모두 no-op이고 당신은 아무것도 보지 못한 채
   framework를 의심하게 됩니다.
3. 결과는 `ProcessingMetricsData(processor=<your processor name>, value=<seconds>)`로 도착하고, §8에
   따라 그것은 console log와 RTVI `"processing"` bucket에 도달하지만 — `LatencyBreakdown`에는
   **도달하지 않습니다**. 그것은 `TTFBMetricsData`와 `TextAggregationMetricsData`만 축적하기
   때문입니다(`user_bot_latency_observer.py:324-341`). 그래서 당신의 rule 비용을 breakdown object
   자체에 넣는 것은 네 번째 단계입니다: observer를 subclass하고 `ProcessingMetricsData`도 축적하기.
   그것이 §13.1의 나머지 절반입니다.

그래서 ch-12를 위한 budget line은 비어 있으면서도 이제 형식이 갖춰졌습니다. 이름이 있고, producer가
있고, metric class가 있고, 전달 경로가 있고, 그 전달 경로에 알려진 구멍이 있습니다:

```
rule evaluation ......... TBD ms   producer: BosonRuleProcessor (ch-12)
                                   metric:   ProcessingMetricsData
                                   visible in: console log, RTVI "processing"
                                   NOT visible in: LatencyBreakdown (unless you subclass, §13.1)
```

`TBD`는 ch-12에서 채웁니다. ch-13에서 측정합니다.

---

## 13. Lina를 위한 framework-extension move

요약이 아닙니다 — 만들 것 넷이고, 각각은 위의 mechanism을 이 chapter가 제기하지 않은 문제에 적용한
것입니다.

### 13.1 `LinaLatencyObserver` — headline 숫자를 당신 자신의 정의에 맞추기

§7.4의 gap은 이 chapter에서 가치가 가장 높은 수정이고, subclass 하나입니다. 코드가 아니라 설계로:

- `UserBotLatencyObserver`를 subclass하십시오. `_handle_metrics_frame`을 override해서
  `TTFAMetricsData`(`leading_silence` 포착)와 `ProcessingMetricsData`(§12의 rule term 포착)도
  축적하게 한 뒤, `super()`를 호출하십시오.
- `LatencyBreakdown`을 field 둘로 확장하십시오: `leading_silence: float | None`과
  `processing: list[ProcessingBreakdownMetrics]`.
- 보정된 headline을 emit하십시오: `on_latency_measured + leading_silence`. 그것이 P50 = 1.0 s /
  P95 = 1.5 s에 대고 볼 숫자입니다. 당신의 `CLAUDE.md`가 정의하는 숫자이기 때문입니다.
- 보정되지 않은 숫자도 다른 이름으로 함께 유지하십시오. 그래야 보정의 크기를 시간에 따라 볼 수
  있습니다. 안정적이면 걱정을 멈춰도 되고, 흘러가면 TTS vendor가 무언가를 바꾼 것입니다.

진정으로 자명하지 않은 부분은 코드가 아니라 순서 제약(ordering constraint)입니다. `TTFAMetricsData`는
**TTS service**가 첫 audio chunk를 훑을 때 emit하고(`tts_service.py:1741` → `process_ttfa_metrics`),
`BotStartedSpeakingFrame`은 **output transport**가 emit합니다(`base_output.py:710`) — 더 뒤의
position입니다. 그러므로 TTFA frame은 *더 앞선* processor에서 push되지만, hop 개수와 queue 깊이에
따라 bot-started frame보다 **나중에 관찰될 수 있습니다.** 따라서 당신의 accumulator는 늦게 도착한
`leading_silence`를 이미 emit해 버린 breakdown에 붙일 수 있어야 하거나, breakdown을 한 박자 붙들고
있어야 합니다. 어느 쪽인지 결정하는 것이 진짜 설계 작업이고, 그것은 두 emit site를 추적해야만
드러나는 종류의 것입니다.

> 💡 **쉬운 설명 — "앞선 processor가 먼저 push했는데 왜 늦게 관찰되나요?"**
> observer가 보는 것은 push 시점이 아니라 **자기 queue에서 꺼낸 시점**입니다(§0.3). TTS service가
> `TTFAMetricsData`를 push한 뒤, 그 audio frame이 output transport까지 가서
> `BotStartedSpeakingFrame`이 push되기까지는 hop 몇 개가 더 필요합니다. 하지만 observer queue는
> 그 사이 순서를 보장하지 않고, `MetricsFrame`은 `SystemFrame`이라 처리 경로도 다릅니다(§6.5).
> 결과적으로 "논리적으로 먼저인 event"가 "관찰 순서상 나중"이 될 수 있고, breakdown을 닫는 시점에
> `leading_silence`가 아직 없을 수 있습니다. 그래서 **닫는 시점을 늦추거나, 닫은 뒤 붙일 수 있게
> 만들거나** 둘 중 하나를 골라야 합니다.

### 13.2 8 kHz 한국어 TTFS benchmark — 존재하지 않는 숫자

평이하게 진술합니다. 이 course에서 가장 중요한 미결 숫자이기 때문입니다:

> **8 kHz μ-law telephony audio 위의 한국어 STT는 `stt_latency.py`에 entry가 없습니다.** 나쁜 entry도
> 아니고 오래된 entry도 아니고 — entry가 없습니다. 23개 상수 전부가 provider의 표준 benchmark 조건에서
> `VADParams.stop_secs=0.2`로 측정되었고, 그중 어느 것도 한국어로 측정되지 않았으며, 어느 것도
> 8 kHz companded telephony audio로 측정되지 않았습니다.

귀결은 [[transport-telephony]]에서 따라옵니다: 배포되는 serializer 여섯 중 다섯이 8 kHz μ-law이고,
wire 위의 Nyquist 천장은 4 kHz이며, μ-law는 8-bit companded입니다. STT 전에 8 k → 16 k로 upsample하는
것은 model의 input contract를 만족시킬 뿐 bandwidth를 복원하지 않습니다. 특히 한국어에서
마찰음/치찰음 대역(ㅅ/ㅆ/ㅊ)과 받침 판별 단서의 상당 부분이 4 kHz 이상에 있고 신호에 그냥 없습니다.
acoustic confidence로 endpoint를 잡는 provider는 그 신호에서 스튜디오 audio와 다르게 행동할 것입니다 —
더 느릴 수도, 더 빠르면서 더 틀릴 수도 있습니다. 어느 쪽인지 당신은 모르고, 이 표도 모릅니다.

산출물: <https://github.com/pipecat-ai/stt-benchmark> 의 benchmark를 당신의 VAD 설정으로 실제
8 kHz μ-law 한국어 audio에 대해 돌리고, 결과를 명시적으로 넘기십시오:

```python
stt = SomeKoreanSTTService(api_key=..., ttfs_p99_latency=<your measured number>)
```

constructor argument는 정확히 이것을 위해 존재하고(`stt_service.py:119-124`), 넘기지 않으면 service가
경고하고 `DEFAULT_TTFS_P99 = 1.0`으로 되돌아갑니다:

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

당신의 안전망 안에 있는 조용한 1.0 s 가정은 추측이 살기에 나쁜 자리입니다. **그 숫자가 측정되기
전까지, Lina에 대한 모든 P95 주장은 검증 불가능합니다.** figure는 이 행을 정확히 그 주석과 함께
눈에 띄게 빈 bar로 그립니다.

### 13.3 `ProcessingMetricsProcessor` wrapper

§12를 일반화하십시오. 다른 processor(또는 callable)를 받아 그 `process_frame`을
`start_processing_metrics()` / `stop_processing_metrics()`로 감싸고 `can_generate_metrics()`가 `True`를
반환하는 작은 `FrameProcessor` 하나를 쓰십시오. 그런 다음 Pipecat이 instrument하지 않으면서 당신이
신경 쓰는 것들을 감싸십시오:

- rule-layer processor(§12) — ch-12의 term.
- 당신의 한국어 text aggregator([[ch-07/read]] §10.1) — 현재 보이지 않습니다. 그것은 TTS service의
  aggregation window 안에 앉아 있어서 비용이 `TextAggregationMetricsData`에 접혀 들어가고 문장 경계
  대기와 분리될 수 없습니다.
- telephony serializer의 resample 쌍(§3.6) — 현재 보이지 않고, 양방향의 모든 audio frame에서
  일어납니다.

쓰기 전에 생각해 볼 가치가 있는 설계 질문: *frame당* 측정하는 wrapper는 frame당
`ProcessingMetricsData` 하나를 만들어내고, 40 ms chunk의 audio라면 방향당 초당 25개의 metrics frame
입니다. 그것은 observer queue에 진짜 부하입니다(§10.3b). 올바른 shape는 거의 확실히 *turn당*
측정하는 것입니다 — 작업을 시작하는 frame type에서 window를 열고, 끝내는 것에서 닫기 — 그리고 그것이
바로 §12에서 LLM service가 하는 일입니다. granularity가 아니라 pattern을 복사하십시오.

### 13.4 budget regression test, eval harness를 설계 의도가 아닌 용도로 쓰기

§2.2는 `pipecat eval`이 의도적으로 latency에 대해 assert하지 않음을 확립했습니다 — `within_ms`의
default는 60 s이고 배포되는 38개 시나리오 중 2개만 그것을 설정합니다. 하지만 field는 존재하고
(`evals/scenario.py:287`, `within_ms: int | None = None`) harness가 그것을 강제합니다
(`harness.py:1093`, `budget_ms = expectation.within_ms or self._default_timeout_ms`).

그러므로: expectation들이 timeout이 아니라 당신의 실제 budget으로 설정된 `within_ms`를 지니는 Lina
시나리오를 하나 쓰고, CI에서 실제 bot에 대해 `-t eval`로 돌리면, framework가 배포하지 않는 latency
regression gate를 얻습니다. 정직한 caveat이고 시나리오 자신의 주석에 진술해야 합니다: harness는
*turn의 user send*로부터 RTVI websocket 위에서 측정하지, 마지막 유성 sample부터가 아닙니다. 그래서
그것이 강제하는 숫자는 §2.1의 숫자와 같은 숫자가 아닙니다. 그것은 **regression** gate입니다 —
"이게 300 ms 나빠졌다"를 잡아냅니다 — P50/P95 target을 위한 compliance gate가 아닙니다. 그 둘은 서로
다른 instrument이고, 그것을 뒤섞는 것이 초록색 CI가 느리게 느껴지는 제품을 덮게 되는 방식입니다.

---

## 다음 챕터로

이 chapter가 앞으로 넘기는 것들이고, 이후 chapter가 다시 도출하는 대신 인용하도록 이름을 붙였습니다:

- **budget은 serial term 일곱 개이고 LLM은 그중 하나입니다.** `stop_secs` → STT finalization →
  **rule evaluation (TBD)** → LLM `ttfat` → text aggregation → TTS `ttfa` → transport. §5의 계산된
  P50은 LLM을 ~30 %에, endpointing/aggregation/TTFA 묶음을 ~65 %에 둡니다. 첫 가청 sample 이후의
  모든 것은 공짜이고(§4.2), TTS synthesis wall time을 budget에 더하는 것이 이것을 틀리는 표준
  방식입니다.

- **가장 큰 두 lever는 keyword argument입니다.** `ttfs_p99_latency=`는 `stt_latency.py`의 23개 상수
  양 극단 사이에서 꼬리를 최대 1.79 s 움직이고([[ch-06/read]] §11이 그 표를 소유합니다),
  `text_aggregation_mode=`는 median을 문장당 200–300 ms 움직입니다([[ch-07/read]] §4). 둘 다 model
  변경이 아닙니다.

- **observer plane은 non-adjacency이고, 그래서 [[ch-12/read]]가 그것을 쓸 수 있습니다.**
  `BaseObserver`(`base_observer.py:90`), hook 넷, event dataclass 셋,
  `frame_processor.py:827-835`와 `:1160-1194`에서 통지되며, `WorkerObserver`의 observer마다
  queue-하나-task-하나 proxy를 통해 fan-out되므로 느린 observer는 turn 대신 queue를 키웁니다. 이것이
  Pipecat에서 position을 차지하지 않고 pipeline을 지켜보는 유일한 mechanism입니다.

- **당신이 두 번 말하기 전까지 아무것도 측정되지 않습니다.** `PipelineParams.enable_metrics`는
  기본이 `False`이고(`worker.py:189`), `can_generate_metrics()`는 base class에서 `False`를 반환합니다
  (`frame_processor.py:488-494`) — tree에 definition 97개, 96개가 `True`를 반환, 전부 service입니다.
  당신이 쓰는 것은 override하기 전까지 침묵합니다. §9.2가 네 줄짜리 최소 wiring입니다.

- **`LatencyBreakdown`은 일곱 term 중 넷을 덮고, headline 숫자는 낙관적입니다.**
  `on_latency_measured`는 `BotStartedSpeakingFrame`에서 멈추는데, 그것은 output transport가 dequeue한
  첫 TTS chunk에서 push합니다 — write 이전이고, TTS 경로에는 silence check가 없습니다
  (`base_output.py:793-799, 896-916`). 그래서 그것은 `leading_silence`와 transport를 제외하고,
  따라서 당신의 `CLAUDE.md`가 정의하는 interval이 *아닙니다*. §13.1이 ~20줄짜리 보정입니다.

- **`TextAggregationMetricsData`는 정확히 하나의 consumer에게 도달하고**, `TTFAMetricsData`는 latency
  observer를 제외한 모든 consumer에게 도달합니다(§8.3의 표). 어느 metric 이름을 그리는가는 design
  decision이고, `ttfb`를 그리면 reasoning-model regression이 개선처럼 보입니다(§5.3).

- **collection은 제공되고, aggregation은 당신 몫입니다.** `examples/observability/`는 파일 셋에
  396줄이고, dashboard도 없고, Sentry/OTel 너머의 exporter도 없고, tree 어디에도 percentile 계산이
  없습니다. 당신의 `CLAUDE.md`의 "Report P50/P95/P99"는 migration 추정치의 line item입니다.

- **Pipecat은 perceptual target을 진술하지 않습니다.** `README.md:29`가 "Ultra-low latency
  interaction"이라고 말하고, `evals/harness.py:113`은 의도적으로 넉넉한 60 s test timeout을 두며,
  배포되는 38개 시나리오 중 2개만 `within_ms`를 설정하는데 둘 다 40 s 이상입니다. ~200 ms / ~800 ms
  수치는 외부 voice-UX 관행이지 repo의 주장이 아닙니다. 주인이 있는 유일한 target은 당신 것입니다:
  **P50 ≤ 1.0 s, P95 ≤ 1.5 s, 마지막 유성 user sample → 첫 가청 assistant sample, end-of-turn time
  포함.**

- **두 boson stack 모두 여기서는 거의 장님입니다**(§11): gateway에는 trace decorator와 `elapsed_ms`
  barge-in input이 있고, realtime_voice에는 읽히지 않는 `discarded_frames` counter, event 위의
  latency field 둘, 그리고 구성상 존재하지 않는 observer plane이 있습니다. mechanism은 진술했고
  판정은 없습니다 — [[ch-13/read]]가 채점합니다.

[[ch-12/read]]가 이 chapter가 비워 둔 slot을 가져갑니다. 이제 그것은 주장(assert)이 아니라 논증
(argue)될 수 있는 모든 것을 갖췄습니다: 분모(§5의 rule line이 빈 채 ~1.16 s), 주인이 있는
target(§2.1), 그 term을 보이게 만드는 mechanism(§12), 그리고 veto를 하고 싶다면 그 term이 serial하고
축소 불가능하다는 지식. ch-12가 답하는 질문은 "rule processor가 어디에 가는가"가 아닙니다 —
[[boson-layers-rules]]가 이미 그것을 강제합니다, `LLMUserAggregator` 뒤 그리고 LLM service 앞.
질문은 "거기서 그것이 얼마이고, 그 비용이 in-turn veto의 값어치를 하는가"입니다.

잃어버리지 않도록 여기에 세워 둔 미결 질문들:

- **8 kHz 한국어 TTFS 숫자.** §13.2. 이 course의 어떤 것도 그것을 닫지 않습니다. 그것은 측정이고 실제
  telephony audio와 선택된 provider가 필요합니다. [[ch-13/read]]는 그것 없이는 신뢰할 만한 P95 주장을
  할 수 없으므로, 그것은 ch-13의 nice-to-have 목록이 아니라 critical path 위에 있습니다.
- **transport term.** §3.6. serialization, 영구적인 8 kHz ↔ 24 kHz resample 쌍, 그리고 network
  transit은 이 repository의 어떤 것에도 측정되지 않습니다. §13.3이 앞의 둘을 위한 wrapper를 주고,
  셋째는 외부 instrumentation입니다.
- **compact/summarization 경로가 critical path 밖에 머무는가.** §11.3. 현재는 가정입니다
  ([[boson-compact-session]]이 turn 밖에서 돈다고 말합니다). §12의 mechanism이 그것을 검사 가능한
  것으로 바꿉니다. ch-13 전에 가정하는 대신 검사하십시오.
- **[[ch-08/read]] §7.7의 stray-fragment race.** 거기에 *"run it before ch-11 builds the observer
  plane, because the observer plane is where you would watch for it"*이라는 메모와 함께 세워 두었던
  것입니다. 이제 plane이 존재합니다: `DebugLogObserver(frame_types={TTSTextFrame: (BaseOutputTransport,
  FrameEndpoint.SOURCE)})` — repo 자신의 예제가 쓰는 바로 그 filter
  (`observability-observer.py:152-157`) — 가 그것을 위한 instrument입니다. 한 번의 실행, 하루치 log.

