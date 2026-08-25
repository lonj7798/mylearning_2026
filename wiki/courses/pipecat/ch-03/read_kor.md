---
title: "What You Already Built: realtime_voice as the Baseline (한국어 companion)"
chapter: ch-03
phase: composition
course: pipecat
kind: korean-companion
companion_of: read.md
sources:
  - rtv-vs-pipecat-gap
  - rtv-pipeline-session
  - rtv-vad-chunking
  - rtv-webrtc-transport
  - theory-narrow-waist
deps:
  - ch-01
  - ch-02
figure: figures/rtv-baseline.html
---

# 3장 — 이미 네가 만든 것: baseline으로서의 realtime_voice

> **이 파일은 [[read]] (`read.md`)의 한국어 companion입니다.** section 번호는 원문과 1:1로 같습니다.
> 나란히 놓고 읽으세요. code block, 파일 경로, 줄 번호, 숫자는 원문과 **글자 그대로 동일**합니다.
> CS/ML 용어는 영어 그대로 둡니다 (frame, processor, pipeline, queue, back-pressure,
> uniform interface, sum type, narrow waist).

> **범위(scope). 챕터 맨 앞에서 선언하고, 챕터 전체에서 강제합니다.** 이 chapter는 **mechanism과
> evidence만** 다룹니다. keep / replace / hybrid 중 어디에도 표를 던지지 않습니다. 선호를 밝히지
> 않습니다. verdict에 도달하지 않고, 어느 한쪽으로 기울지도 않습니다. 아래의 모든 문장은
> measurement이거나, 인용이거나, 질문입니다. 점수를 매기는 것은 [[ch-13/read]]의 일이고, 그것을
> 미루는 것은 **의도적**입니다 — 아직 mechanism을 본 적도 없는 subsystem에 점수를 매길 수는
> 없으며, 그런 subsystem이 아직 여덟 개 남아 있습니다 ([[ch-04/read]]부터 [[ch-12/read]]까지).
>
> **이 chapter가 앞으로 넘기는 것은 fact sheet이지 recommendation이 아닙니다.**

---

## 왜 이 챕터인가

[[ch-01/read]]은 Pipes-and-Filters를 줬습니다. 모든 boundary에서의 uniform interface가 splicing(끼워
넣기)을 가능하게 만드는 바로 그것이라는 이야기였죠. [[ch-02/read]]는 narrow waist와 그 대가를 줬습니다.
Pipecat의 uniform interface는 **하나의 open sum type**, 즉 `Frame`이고, open sum type은 새로운
*function*(processor)은 싸게 만들지만 새로운 *case*(frame type)는 비싸게 만듭니다 — Wadler의
Expression Problem이, production codebase에서 **136개 파일에 걸친 577개의
`isinstance(frame, ...)` call site**라는 측정 가능한 비율로 실제로 벌어지고 있는 것입니다.

이건 trade-off에 대한 교훈입니다. 그런데 trade-off는, 실제로 출시해야만 했던 누군가가 양쪽을 다
구현해 놓은 것을 보기 전까지는 추상적입니다. 당신은 그걸 봤습니다. 브랜치 `voice-chat-dev`의
`packages/realtime_voice/`는 동작하는 full-duplex 한국어 voice stack이고, 그 data의 단위는 **정반대의
베팅**입니다: 한 줄로 선언된 *closed* union.

```python
# packages/realtime_voice/realtime_voice/types.py:201
# (boson-agent, private; excerpt-attested via [[rtv-pipeline-session]])
VoiceRuntimeEvent = VADEvent | ASREvent | AgentTextDelta | VoiceEvent
```

그래서 이 chapter는 ch-02의 교훈을 당신 자신의 code 위에서 구체화하면서 composition phase를 닫습니다.
framing question — 이후 모든 chapter가 끌어다 쓸 질문 — 은 "realtime_voice에 frame이 있는가?"가
아닙니다(없습니다). 질문은 이것입니다:

> **각 베팅은 무엇을 샀고, 무엇을 지불했으며, 그것은 어떤 author 집단(population)에 대해서인가?**

이건 framework-extension 질문이고, 당신이 가장 강한 모드입니다. 이 chapter의 나머지는 그 질문에 답하기
위해 필요한 evidence이며, subsystem별로 양방향으로 배치됩니다: realtime_voice가 구현했지만 Pipecat이
안 한 것, 그리고 Pipecat이 구현했지만 realtime_voice가 안 한 것. 둘 중 어느 목록도 scorecard가
아닙니다. 둘 다 input입니다.

---

## 0. 이 chapter의 evidence를 읽는 방법

이 chapter에는 **두 종류의 claim**이 있고, 둘은 검증 가능성(verifiability)이 다릅니다. 이 둘을 구분할
줄 알아야 합니다. [[ch-13/read]]가 둘을 다르게 가중치 매길 것이기 때문입니다.

| Class | Source of truth | 어떻게 확인하는가 |
|---|---|---|
| **Pipecat claim** — 파일 경로, 줄 번호, class 이름, count, LOC, grep 결과 | `wiki/raw-data/pipecat/pipecat-src`, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` | 파일을 열어보면 됩니다. 아래의 모든 숫자는 2026-08-25에 그 tree에 대해 재측정되었고, 정확한 command가 옆에 인쇄되어 있습니다. |
| **realtime_voice / boson-agent claim** — 3,886 lines, 561-line `VoiceSession`, 60개 test, `types.py:201` | `wiki/raw-data/pipecat/excerpts/` 아래의 `rtv-*` excerpt들. private repo `boson-agent`, 브랜치 `voice-chat-dev`, commit `034ce4ca09a2f109e6c248a43bc989f8d26a6abf` (2026-07-29)에서 직접 읽은 것 | 당신 자신의 repo에 대고 확인해야 합니다. 이 wiki에서는 **확인 불가능**하며, 이 chapter는 그런 척하지 않습니다. |

아래의 모든 boson 쪽 code block은 boson 경로 **와** 출처 excerpt wikilink를 함께 달고 있습니다. 모든
Pipecat 쪽 code block은 repo 기준 상대 경로와 지금 당장 `sed -n` 걸 수 있는 line range를 달고 있습니다.
둘이 서로 어긋날 때는 매끄럽게 덮지 않고 그렇다고 말합니다 — 그런 불일치가 §9에 하나 있고, 표시해
뒀습니다.

**숫자가 시작되기 전에 솔직한 caveat 하나.** 두 snapshot은 6주 차이가 납니다. boson excerpt는
2026-07-29, Pipecat tree는 2026-08-25입니다. 만약 `realtime_voice`가 2026-07-29 이후에 움직였다면 —
그리고 그건 당신의 active repo이니 움직였다고 가정하세요 — 여기 있는 모든 boson 숫자는 현재 값이 아니라
**하한(floor)**입니다. [[ch-13/read]] 전에 재측정하세요.

---

## 1. scale strip, 측정된 값

거친 숫자부터 시작합니다. 그게 모든 것의 frame이 되기도 하고, 동시에 가장 과잉 해석되기 쉬운 숫자이기도
하기 때문입니다.

```bash
# run in wiki/raw-data/pipecat/pipecat-src
$ find src   -name '*.py' | xargs wc -l | tail -1
  168847 total
$ find tests -name '*.py' | xargs wc -l | tail -1
   92538 total
$ grep -rno "def test_" tests | wc -l
    4236
$ find tests -name 'test*.py' | wc -l
     236
```

이에 대비해서, [[rtv-vs-pipecat-gap]]에서 (excerpt-attested):

| | realtime_voice | Pipecat |
|---|---|---|
| source lines | 3,886 | 168,847 |
| test lines | 1,504 | 92,538 |
| test function / `def test_` match 수 | 60 | 4,236 |
| test 파일 수 | 6 | 236 |
| test:src line ratio | 0.39 | 0.55 |

168,847 / 3,886 ≈ **43×**. 이 비율은 실제 값이지만, 그 자체로는 거의 정보가 없습니다. 두 숫자가 같은
대상을 재고 있지 않기 때문입니다. Pipecat의 168,847 lines 중 아주 큰 비중은 **breadth**입니다 — 62개의
service directory, 11개의 transport package, 9개의 serializer, CLI, registry, template — 그리고 그중
어느 것도 realtime_voice는 시도하지 않았습니다. realtime_voice는 정확히 하나의 ASR vendor, 하나의 TTS
vendor, 하나의 transport, 하나의 언어를 상대로 쓰였기 때문입니다.

그러니 43×를 "capability가 43배"로 읽지 마세요. 이렇게 읽으세요: *이 두 artifact는 서로 다른 질문에
답하려고 만들어졌다.* 이 chapter의 나머지는 정확히 어떤 질문들인지에 관한 것입니다.

이 raw 숫자가 실제로 무게를 갖는 유일한 자리는 §9의 마지막 줄입니다. 거기서는 60개 test의 *이름* 이 —
ratio가 아니라 — evidence입니다.

> **💡 쉬운 설명 — 43×를 어떻게 읽어야 하나**
> 두 숫자를 "한 사람이 만든 자동차 1대 vs 공장이 만든 43대"로 읽으면 틀립니다. 정확한 비유는
> "한국 시장용 세단 1종 vs 세단·트럭·버스·전기차·수출용 우핸들까지 62개 라인업"입니다. 43배의 대부분은
> **같은 기능을 더 잘 하는 것**이 아니라 **다른 vendor / 다른 transport / 다른 나라를 지원하는 breadth**
> 입니다. 그래서 43×는 "누가 더 낫냐"가 아니라 "무엇을 상대로 쓰였냐"의 지표입니다.

---

## 2. data의 단위: open sum type 대 closed union

### 2.1 Pipecat이 선언하는 것

[[ch-02/read]]와 [[theory-narrow-waist]]에서 상기하세요: waist는 payload를 전혀 나르지 않습니다.

**`src/pipecat/frames/frames.py` L64–89**

```python
@dataclass
class Frame:
    """Base frame class for all frames in the Pipecat pipeline.

    All frames inherit from this base class and automatically receive
    unique identifiers, names, and metadata support.

    Parameters:
        id: Unique identifier for the frame instance.
        name: Human-readable name combining class name and instance count.
        pts: Presentation timestamp in nanoseconds.
        broadcast_sibling_id: ID of the paired frame when this frame was
            broadcast in both directions. Set automatically by
            ``broadcast_frame()`` and ``broadcast_frame_instance()``.
        metadata: Dictionary for arbitrary frame metadata.
        transport_source: Name of the transport source that created this frame.
        transport_destination: Name of the transport destination for this frame.
    """

    id: int = field(init=False)
    name: str = field(init=False)
    pts: int | None = field(init=False)
    broadcast_sibling_id: int | None = field(init=False)
    metadata: dict[str, Any] = field(init=False)
    transport_source: str | None = field(init=False)
    transport_destination: str | None = field(init=False)
```

그리고 그 아래의 3-way branch. 이 docstring들은 content taxonomy가 아니라 **scheduling contract**
입니다:

**`src/pipecat/frames/frames.py` L104–138**

```python
@dataclass
class SystemFrame(Frame):
    """System frame class for immediate processing.

    A frame that takes higher priority than other frames. System frames are
    handled in order and are not affected by user interruptions.
    """

    pass


@dataclass
class DataFrame(Frame):
    """Data frame class for processing data in order.

    A frame that is processed in order and usually contains data such as LLM
    context, text, audio or images. Data frames are cancelled by user
    interruptions.
    """

    pass


@dataclass
class ControlFrame(Frame):
    """Control frame class for processing control information in order.

    A frame that, similar to data frames, is processed in order and usually
    contains control information such as update settings or to end the pipeline
    after everything is flushed. Control frames are cancelled by user
    interruptions.

    """

    pass
```

그 근처에 네 번째 이름이 하나 더 있고, 이건 정확하게 짚고 갈 가치가 있습니다. 네 번째 branch로
잘못 분류하기 쉽거든요:

**`src/pipecat/frames/frames.py` L147–157**

```python
class UninterruptibleFrame:
    """A marker for data or control frames that must not be interrupted.

    Frames with this mixin are still ordered normally, but unlike other frames,
    they are preserved during interruptions: they remain in internal queues and
    any task processing them will not be cancelled. This ensures the frame is
    always delivered and processed to completion.

    """

    pass
```

`UninterruptibleFrame`은 `Frame`을 상속하지 **않습니다**. 두 번째 base로 적용되는 mixin입니다 —
L1899의 `class EndFrame(ControlFrame, UninterruptibleFrame)`처럼요. 그러니 branch 구조는 네 개의 peer가
아니라 **세 개의 class + 직교하는 marker 하나**입니다.

### 2.2 "narrow" waist는 실제로 얼마나 넓은가

frame class를 세는 일은 정직한 사람들끼리도 방법에 따라 다른 숫자가 나오는 영역입니다. 그래서 제가 쓴
방법을 명시해서 인쇄합니다.

```python
# AST walk over src/pipecat/frames/frames.py at commit 0cbf9c5b
# ---------------------------------------------------------------
# total classes declared in the file ....................... 133
# classes whose name ends in "Frame" ....................... 131
# transitive descendants of `Frame` ........................ 123
#   subtree under SystemFrame (excl. the root) .............  48
#   subtree under DataFrame   (excl. the root) .............  33
#   subtree under ControlFrame(excl. the root) .............  39
#   ------------------------------------------------- sum ... 120
#   UNION of the three subtrees, deduplicated .............. 119
#   carrying the UninterruptibleFrame mixin ................  13
```

숫자가 안 맞아 보이는 세 줄이 오히려 흥미로운 지점이고, 정확히 화해됩니다(reconcile):

- branch subtree들의 **합(sum)**은 120인데 **합집합(union)**은 119입니다. `InputTextRawFrame`이
  `class InputTextRawFrame(SystemFrame, TextFrame)`으로 선언되어 있어서 — `SystemFrame` 아래에
  *그리고* (`TextFrame`을 통해) `DataFrame` 아래에 동시에 있고, 따라서 두 번 세어지기 때문입니다;
- L551의 `LLMContextFrame`은 `Frame`을 **직접** subclass합니다. 어떤 branch에도 속하지 않죠 — 이
  파일에서 유일한 그런 class입니다 — 그래서 123에는 들어가지만 119에도 120에도 안 들어갑니다;
- 119 (union) + 1 (`LLMContextFrame`) + 3 (branch root 자체들) = **123**.

[[theory-narrow-waist]]는 같은 파일을 123 descendants / 120 "concrete"로 보고하며 같은 두 anomaly를
짚습니다. 이 chapter에 대한 course outline의 figure spec은 ch-02의 AST walk를 `129 → 122 → 119`로
기술합니다. 여기서 `119`는 위의 dedup된 branch union이고 `122`는 거기에 세 개의 root를 더한 값이라 세
숫자 중 둘은 화해됩니다. **`129`는 제가 시도한 어떤 counting rule로도 재현하지 못했기에, 주장하지
않습니다.** [[ch-02/read]]에 갔을 때 대조해 보세요. 거기서도 재현이 안 되면 그건 틀린 값이고, 반복할 게
아니라 고쳐야 합니다.

실제로 중요한 숫자인 dispatch tax:

```bash
$ grep -rn "isinstance(frame," src | wc -l        # 577
$ grep -rln "isinstance(frame," src | wc -l       # 136
```

**577개 site, 136개 파일.** 새로운 `Frame` subclass 하나하나가 그 전부에 대한 새로운 의무입니다.

### 2.3 handle되지 않은 frame이 왜 raise가 아니라 사라지는가

이건 open sum type의 mechanical한 귀결이고, 요약을 믿기보다 실제 method를 읽을 가치가 있습니다.
`FrameProcessor.process_frame` — 모든 processor가 `super()`로 호출하는 base 구현 — 은 정확히 다섯 가지
frame 모양만 handle하고 **`else`가 없습니다**:

**`src/pipecat/processors/frame_processor.py` L820–847**

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
        elif isinstance(frame, CancelFrame):
            await self.__cancel(frame)
        elif isinstance(frame, (FrameProcessorPauseFrame, FrameProcessorPauseUrgentFrame)):
            await self.__pause(frame)
        elif isinstance(frame, (FrameProcessorResumeFrame, FrameProcessorResumeUrgentFrame)):
            await self.__resume(frame)
```

거기 **없는 것**에 주목하세요: fallthrough branch도 없고, 맨 아래에
`await self.push_frame(frame, direction)`도 없습니다. forwarding은 *subclass*가 직접 호출해야 하는 별개
method입니다:

**`src/pipecat/processors/frame_processor.py` L1004–1015**

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

그러니: frame type을 하나 추가하면, 그것을 예상하지 못한 author가 쓴 모든 processor가 그걸 떨어뜨립니다
— 조용히, exception 없이, error level log 한 줄 없이, type error 없이. frame은 processor에 도달하고,
다섯 개 `isinstance` arm 중 어느 것에도 매치되지 않고, method는 return하고, frame은 사라집니다. 그것이
open sum type이 기본 탑재하고 출시되는 failure mode입니다.

> **💡 쉬운 설명 — silent drop이 왜 이렇게 나쁜가**
> 우체국에 새 종류의 소포(`RuleViolation`)를 하나 도입했다고 합시다. 중간 집배원 136명은 각자 자기
> 매뉴얼에 적힌 5종류 소포만 처리하도록 훈련돼 있고, 매뉴얼에 없는 소포가 오면 **반송도 신고도 하지 않고
> 그냥 버립니다.** 문제는 버려졌다는 사실 자체가 아무 데도 기록되지 않는다는 겁니다. 당신이 보게 되는
> 증상은 3-4 hop 뒤에서 "assistant 음성이 안 나옴" 또는 "turn이 멈춤"이고, 원인 지점과 증상 지점이
> 완전히 떨어져 있습니다. 반대로 closed union은 우체국 입구에서 "이런 소포 규격 없음" 하고 소리를
> 지릅니다(§2.5의 `TypeError`).

### 2.4 realtime_voice가 대신 선언하는 것

**`packages/realtime_voice/realtime_voice/types.py:201`** — boson-agent, private,
[[rtv-pipeline-session]] 경유 excerpt-attested:

```python
VoiceRuntimeEvent = VADEvent | ASREvent | AgentTextDelta | VoiceEvent
```

저 한 줄이 frame taxonomy 전체입니다. 123개 descendant를 가진 base class가 아니라, **네 개의 frozen
dataclass에 대한 PEP 604 union**이고, 구성상 닫혀 있습니다(closed by construction).

[[rtv-pipeline-session]]에 따르면 이 union이 감싸는 payload type들은 전부
`@dataclass(frozen=True, slots=True)`입니다: `AudioFrame` (L48, PCM16 전용 — 다른 `AudioFormat`이나
`2 * channels`에 정렬되지 않은 byte 길이에 대해 `__post_init__`이 `InvalidAudioFrameError`를 raise),
`VADEvent` (L95), `ASREvent` (L113), `AgentRequest` (L126), `AgentTextDelta` (L134), `TextChunk`
(L142), `TTSRequest` (L149), `SynthesizedAudio` (L161), `VoiceEvent` (L188).

그리고 Pipecat에 대응물이 전혀 없는 구조적 디테일 하나: correlation이 **모든 payload에** 붙어 있습니다.
`SessionId / TurnId / GenerationId / PhraseId`는 L16–19에서 `NewType("...", str)`이고, 모든 frame이
`CorrelationIds(session_id, turn_id, generation_id)` (L27)를 나릅니다. Pipecat에서 base `Frame`은 `id`,
`name`, `pts`, `broadcast_sibling_id`, `metadata`, `transport_source`, `transport_destination`를
나르고 — 위 L64–101 인용 참조 — **conversational correlation field는 없습니다**. session / turn /
generation identity는 Pipecat code가 그것이 필요할 때 열려 있는 `metadata: dict[str, Any]`에 얹혀
갑니다. [[theory-narrow-waist]]는 이걸 ossification(경직화)의 escape hatch로 지목합니다: base field
set은 사실상 얼어 있으므로, 새로운 것은 전부 type 없는 dict로 들어갑니다.

그리고 그냥 평범하게 이름 붙일 구조적 부재(absence)가 하나 더 있습니다. Pipecat의 `SystemFrame` /
`DataFrame` / `ControlFrame` 분할은 **scheduling priority를 datum의 type 안에 encode**합니다.
`VoiceRuntimeEvent`에는 그런 분할이 없습니다. realtime_voice에서 priority는 datum의 속성이 전혀
아닙니다 — 그 datum이 어떤 queue에 올려졌는가의 속성이고, queue는 네 개이며 overflow policy는 세 가지가
있습니다(§3.3). 같은 관심사(concern), 다른 위치.

### 2.5 closed union이 boundary에서 드러나는 지점

closed union에서 흥미로운 건 선언 자체가 아닙니다. system의 가장자리에서, union이 이름 붙이지 않은
무언가가 도착했을 때 무슨 일이 벌어지는가입니다. realtime_voice에서 그 가장자리는 WebRTC transport의
wire-mapping function이고, [[rtv-webrtc-transport]]에서 나옵니다:

**`packages/realtime_voice/realtime_voice/transport/webrtc/transport.py` L118–156** —
`_control_event()`, closed union에서 wire type으로 가는 손으로 쓴 mapping (excerpt-attested):

- `VoiceEvent` → `event.kind.value` (이미 dot로 구분된 형태, 예: `"assistant.audio_committed"`)
- `AgentTextDelta` → `"text_delta"`
- `ASREvent` → `"transcript.interim" | "transcript.final" | "asr.end_of_turn" | "asr.error"`
- `VADEvent` → `"vad.speech_started" | "vad.speech_stopped"`
- 그 외 전부 → `TypeError(f"unsupported voice event: {type(event).__name__}")`

두 failure mode를 나란히 놓으세요. 이게 trade의 전부입니다:

| | 아무도 handle하지 않는 새 datum |
|---|---|
| **Pipecat, open sum type** | processor에 도달하고, `process_frame`의 어떤 `isinstance` arm에도 매치되지 않고, method가 return하고, **frame이 조용히 drop됨**; pipeline은 계속 돌아가고 증상은 downstream 어딘가에서 audio 누락이나 멈춘 turn으로 나타남 |
| **realtime_voice, closed union** | `_control_event`에 도달하고, 어떤 arm에도 매치되지 않고, **wire boundary에서 `TypeError` raise**; 그리고 그 전에 `mypy`가 union 선언 지점에서 거부할 기회를 이미 가졌음 |

### 2.6 각 베팅이 사는 것 — 진짜 질문

저울에 손가락 얹지 말고, 양방향으로 단도직입적으로 말합시다.

**open sum type은 core를 건드리지 않는 third-party extension을 삽니다.** 누구든 — STT integration을
출시하는 vendor든, 한국어 rules processor를 쓰는 당신이든 — 자기 module 안에서 `Frame` subclass를
선언하면 그것이 기존 pipeline 전체와 compose됩니다. Pipecat 자신의 `flows/` package가 정확히 이렇게
합니다: `frames.py`가 아니라 `flows/actions.py`에서 **두 개**의 frame,
`FunctionActionFrame(ControlFrame)`과 `ActionFinishedFrame(ControlFrame)`을 정의합니다
([[theory-narrow-waist]], 그리고 [[ch-10/read]] 참조). core 파일은 전혀 바뀌지 않았습니다. 대가는 577개
site와 silent drop입니다: frame의 집합이 compile time에 알 수 없기 때문에, 모든 processor가 모든
frame을 handle한다는 것을 그 어디에서도 증명할 수 없습니다.

**closed union은 exhaustiveness(전수성)를 삽니다.** case의 집합이 한 줄에 쓰여 있기 때문에, type
checker가 `VoiceRuntimeEvent`에 대한 `match`를 걸어보고 당신이 어떤 case를 빠뜨렸는지 check time에
말해 줄 수 있습니다. 알 수 없는 event는 알 수 없는 hop에서 frame이 사라지는 것이 아니라 이름 있는
boundary에서 터지는 시끄러운 `TypeError`입니다. 대가는 **extension이 일어날 수 있는 곳이 core뿐**이라는
것입니다: event type을 추가한다는 건 `types.py:201`을 고치고 그 union을 소비하는 모든 `isinstance`
chain을 고친다는 뜻이며, third party는 당신 package에 patch를 넣지 않고는 아예 할 수가 없습니다.

이 둘이 한 질문에 대한 두 답이 아니라는 점에 주목하세요. 이 둘은 **다음 component를 누가 쓸 것인가에
대한 서로 다른 두 질문**에 대한 답입니다:

- Pipecat은 **자기가 만날 일 없는 external author 집단**을 최적화하고 있습니다 — 매주 도착하는, 62개
  service directory 어치의 vendor들. 그 집단에게 "core에 PR 없이 확장할 수 있다"는 요구사항이고,
  exhaustiveness는 case set이 설계상 열려 있으므로 어차피 달성 불가능합니다.
- realtime_voice는 **파일 전체를 소유한 한 팀** — 당신 — 집단을 최적화하고 있습니다. 그 집단에게
  `types.py:201`을 고치는 건 2분짜리 비용이고, 그 대가로 얻는 것은 package 안의 모든 event-handling
  site에 대해 type checker가 두 번째 reviewer가 되어 준다는 것입니다.

둘 다 방어 가능합니다. 어느 쪽이 옳은지는 *다음 component를 누가 쓰는가*의 함수이며, 그건 두 codebase에
관한 사실이 아니라 당신 팀과 당신 roadmap에 관한 사실입니다. 그리고 그것이 정확히 [[ch-13/read]]가
결정해야 하는 것이고, 그래서 이 chapter는 결정을 거부합니다.

> **💡 쉬운 설명 — Expression Problem을 표 하나로**
> 두 축이 있습니다: **새 case(데이터 종류) 추가**와 **새 function(처리기) 추가**.
>
> | | 새 case 추가 | 새 function 추가 |
> |---|---|---|
> | open sum type (Pipecat `Frame`) | 비쌈 — 577 site가 잠재적 의무 | **쌈** — 아무나 자기 module에서 processor 선언 |
> | closed union (`VoiceRuntimeEvent`) | **쌈이자 안전** — 한 줄 + type checker가 나머지 site를 찾아줌 | 비쌈 — core를 소유한 사람만 가능 |
>
> 어느 열이 중요한지는 코드가 아니라 **조직도**가 결정합니다. 외부 vendor가 매주 오면 오른쪽 위 칸이
> 지배적이고, 팀 하나가 파일 전체를 소유하면 왼쪽 아래 칸이 지배적입니다.

### 2.7 figure를 여기서 쓰세요

→ **[ch-03 baseline viewer 열기](./figures/rtv-baseline.html)** 를 열고, 더 읽기 전에 *headline panel*
을 직접 조작해 보세요: Pipecat 쪽에서 **"add a frame type"** 을 눌러 어떤 processor들이 silent-drop
site로 불이 켜지는지 보고, 그다음 realtime_voice 쪽에서 **"add an event type"** 을 눌러 `types.py:201`
에서 시작해 모든 `isinstance` chain을 거쳐 `_control_event`가 `TypeError`를 raise하는 데까지 강제되는
edit을 따라가 보세요. 각 column 아래의 checker strip이 요점입니다 — type checker가 오른쪽에서는 무엇을
증명할 수 있고 왼쪽에서는 무엇을 증명할 수 없는지를 보여줍니다. 의견을 형성하기 전에 양방향을 다 해
보세요. 이 panel은 의도적으로 대칭이고 투표 버튼이 없습니다.

### 2.8 framework-extension probe #1

당신은 `RuleViolation` — boson의 rule layer([[ch-12/read]])가 내는 verdict로, TTS가 소리를 내기
**전에** downstream guard가 반드시 봐야 하는 것 — 을 두 system 각각에 추가하려고 합니다. 산문으로
답하지 말고, diff로 답하세요.

1. Pipecat에서: 새 class는 어느 파일로 가고, 어떤 base를 subclass하며, 그리고 — §2.3의
   `process_frame` 인용을 감안할 때 — guard가 실제로 그걸 보려면 rule layer와 `transport.output()`
   사이의 processor 중 몇 개가 바뀌어야 합니까?
2. realtime_voice에서: 같은 질문. edit을 세어 보세요. `types.py:201`이 하나입니다. 나머지 목록은
   무엇이고, 그중 어느 edit을 type checker가 대신 찾아줬을까요?
3. 이제 집단(population)을 바꿉니다: 어떤 third party가 한국어 존댓말 일관성 checker를 pip-설치 가능한
   package로 각 system에 대해 출시하고 싶어 합니다. 위의 두 edit 중 그들에게 *가능하기라도 한* 것은
   어느 쪽입니까?

---

## 3. work의 단위: processor abstraction 대 supervisor

### 3.1 Pipecat이 compose하는 것

Pipecat의 work 단위는 `FrameProcessor`이고, composition은 하나의 method가 만드는 doubly-linked list
입니다:

**`src/pipecat/processors/frame_processor.py` L671–679**

```python
    def link(self, processor: FrameProcessor):
        """Link this processor to the next processor in the pipeline.

        Args:
            processor: The processor to link to.
        """
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")
```

pointer 두 개, `_next`와 `_prev`. pointer가 두 개라는 것이 frame이 두 방향으로 이동할 수 있는 이유
입니다:

**`src/pipecat/processors/frame_processor.py` L60–69**

```python
class FrameDirection(Enum):
    """Direction of frame flow in the processing pipeline.

    Parameters:
        DOWNSTREAM: Frames flowing from input to output.
        UPSTREAM: Frames flowing back from output to input.
    """

    DOWNSTREAM = 1
    UPSTREAM = 2
```

그리고 push method는 정확히 그것으로 dispatch합니다:

**`src/pipecat/processors/frame_processor.py` L1170–1183**

```python
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
```

그러면 `Pipeline`은 거의 아무것도 아닙니다 — source와 sink로 감싼 list를, 쌍쌍이 link한 것:

**`src/pipecat/pipeline/pipeline.py` L113–121, L197–202**

```python
        super().__init__(enable_direct_mode=True)

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = source or PipelineSource(self.push_frame, name=f"{self}::Source")
        self._sink = sink or PipelineSink(self.push_frame, name=f"{self}::Sink")
        self._processors: list[FrameProcessor] = [self._source, *processors, self._sink]

        self._link_processors()
```

```python
    def _link_processors(self):
        """Link all processors in sequence and set their parent."""
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.link(curr)
            prev = curr
```

저 여섯 줄짜리 loop이 topology mechanism의 전부입니다. `processors`에 원소 하나를 끼워 넣으면 pipeline이
달라집니다. 그것이 [[ch-01/read]]이 지목한 Pipes-and-Filters의 성질입니다: **topology는 data다.**

runtime에 각 processor *안에서* 무슨 일이 일어나는지 — task 두 개, queue 두 개, priority scheduling —
는 [[ch-04/read]]의 주제이고, 여기서는 의도적으로 쓰지 않습니다.

### 3.2 realtime_voice가 supervise하는 것

[[rtv-pipeline-session]]에서, `packages/realtime_voice/realtime_voice/pipeline/session.py`:

- `VoiceSession`은 **L89**에 선언되어 있고 **561 lines**입니다.
- 그 docstring, L90: *"Supervise audio -> VAD -> ASR -> text agent -> TTS -> queued audio."*
- constructor는 네 개의 stage를 keyword arg로 받고(`vad`, `asr`, `agent`, `tts`), 거기에
  `VoiceSessionConfig`를 더 받습니다.
- **L257**의 `_supervise()`는 **하나**의 `asyncio.TaskGroup`을 열고, session 전체에 대해 정확히 **두
  개**의 long-lived task를 만듭니다: `"voice-input"` (`_input_loop`)과 `"voice-asr-events"`
  (`_asr_loop`).
- turn별 작업은 nested task를 spawn합니다: `f"asr-finalize:{gen}"`, `f"voice-generation:{gen}"`,
  그리고 `_run_generation` 안쪽에서 `f"agent-text:{gen}"` + `f"tts:{gen}"`를 가진 inner `TaskGroup`.

session 전체에 대해 long-lived task 두 개, 거기에 turn마다 짧게 사는 nest 하나. Pipecat의 processor별
runtime — chain의 각 원소가 몇 개의 task와 queue를 소유하는가 — 은 [[ch-04/read]]입니다. figure의 두 번째
panel이 realtime_voice 쪽을 그것에 대한 명시적 placeholder에 대비해 그려 주므로, mechanism을 알기 전에도
모양 차이를 볼 수 있습니다.

또한 [[rtv-pipeline-session]] 기준으로 **없는 것들** — 결함이 아니라 부재로 진술합니다: `Pipeline`에
해당하는 것이 없고, `PipelineTask`도, `PipelineRunner`도, `ParallelPipeline`도, observer plane도,
frame 수준 metric도 없습니다. `errors.py` (30 L)가 core package의 error surface 전부입니다:
`RealtimeVoiceError(Exception)`와 `InvalidAudioFrameError(RealtimeVoiceError, ValueError)`,
`QueueOverflowError`, `QueueClosedError`, `ProviderError`,
`ProviderTimeoutError(ProviderError, TimeoutError)`, `SessionClosedError`.

### 3.3 슬롯은 교체 가능, topology는 고정

`protocols.py`는 82 lines이고 다섯 개의 `@runtime_checkable Protocol`을 정의하며 그 외에는 아무것도
없습니다 ([[rtv-pipeline-session]]):

```python
# packages/realtime_voice/realtime_voice/protocols.py — shape as recorded in [[rtv-pipeline-session]]
class VAD(Protocol):    async def process(self, frame: AudioFrame, correlation: CorrelationIds) -> Sequence[VADEvent]
class StreamingASR(Protocol):  async def start / push_audio / finalize / close;  def events() -> AsyncIterator[ASREvent]
class StreamingTTS(Protocol):  async def start / cancel(generation_id) / close;  def synthesize(request) -> AsyncIterator[SynthesizedAudio]
class StreamingConversationAgent(Protocol):  def stream(request) -> AsyncIterator[AgentTextDelta];  async def cancel / close
class VoiceTransport(Protocol):  async def start / send_audio / send_event / close;  def incoming_audio() -> AsyncIterator[AudioFrame]
```

붙잡고 있어야 할 문장은 이것입니다: **슬롯은 교체 가능하고, topology는 아니다.** `VoiceSession`에 다른
`StreamingTTS`를 건네줄 수 있습니다. 하지만 ASR과 agent *사이에* processor를 넣을 수는 없습니다.
"사이"라는 것이 없기 때문입니다 — VAD → ASR → agent → TTS라는 순서가 `_supervise()`와 그 nested task
구조 안에 쓰여 있습니다. `link()`도 없고, `FrameDirection`도 없고, upstream push도 전혀 없습니다.
data는 한 방향으로만 흐릅니다.

마지막 항목은 겉치레가 아닙니다. Pipecat에서 tool-call loop를 닫는다는 것은 `LLMContextFrame`을
**upstream**으로 push한다는 뜻입니다 ([[function-calling]], 그리고 [[ch-09/read]]). 단방향 supervisor
에는 그 모양을 위한 mechanism이 없습니다. realtime_voice는 대신 tool loop 전체를 package 바깥으로
위임하는데, 그것이 §7.6입니다.

interposition(중간 삽입)이 불가능하다는 것은 정확히 [[ch-01/read]]이 Pipes-and-Filters가 사 준다고 한
성질입니다. realtime_voice는 그걸 포기합니다. 포기해서 얻는 것은, 연산 순서가 사는 파일이 정확히 하나
있고 그 한 파일을 읽으면 data path 전체를 알 수 있다는 것 — 이건 실제 성질이고, 561줄짜리 class가 그나마
이해 가능한 이유입니다.

scheduling policy는 datum의 type에 있지 않으니, bounded queue들이 그것이 간 자리입니다. L41의
`VoiceSessionConfig`, [[rtv-pipeline-session]]에서 그대로 옮긴 기본값:

```python
sample_rate=16_000; channels=1; language="ko"
ingress_queue_size=64; event_queue_size=256; phrase_queue_size=8; audio_queue_size=32
vad_prefix_frames=5; phrase_min_chars=12; phrase_max_chars=60
phrase_hard_max_chars=None; phrase_batch_max_chars=320; adaptive_sentence_batching=True
```

네 개의 queue에 걸쳐 세 가지 overflow policy:

| queue | policy | mechanism |
|---|---|---|
| ingress (64) | **overflow 시 reject** | `push_audio` (L164)가 `QueueOverflowError("ingress queue full; frame rejected instead of adding latency")`를 raise |
| phrase (8) | backpressure | producer가 await |
| audio (32) | backpressure | producer가 await |
| event (256) | 사실상 unbounded | `_emit`이 await |

class docstring L92–94: *"Ingress uses reject-on-overflow so a transport cannot silently extend user
turn latency."* 이건 queue policy로 encode된 latency 보존 결정입니다 — [[ch-11/read]]까지 들고 가세요.
거기서 Pipecat의 latency accounting이 측정됩니다.

> **💡 쉬운 설명 — reject-on-overflow vs back-pressure를 콜센터로**
> ingress queue가 꽉 찼을 때 선택지는 두 가지입니다. (a) **back-pressure**: "잠깐만요" 하고 producer를
> 기다리게 함 → 오디오가 버려지지는 않지만, 큐에 쌓인 만큼 **사용자 turn의 latency가 조용히 늘어납니다**.
> 뒤늦게 처리된 음성은 이미 늦은 음성이라 어차피 쓸모가 적습니다. (b) **reject**: 그 frame을 버리고
> `QueueOverflowError`를 던짐 → 오디오는 잃지만 시간축은 지켜집니다. realtime_voice는 마이크 쪽(ingress)
> 에서만 (b)를 고르고, 나머지 세 queue에서는 (a)를 고릅니다. 이유는 docstring에 그대로 있습니다:
> *transport가 사용자 turn latency를 몰래 늘리게 두지 않는다.*

### 3.4 shutdown: 전파되는 frame 대 손으로 순서 매긴 24줄

Pipecat은 linked list를 타고 가는 frame을 push해서 pipeline을 종료합니다:

**`src/pipecat/frames/frames.py` L1899–1910**

```python
class EndFrame(ControlFrame, UninterruptibleFrame):
    """Frame indicating pipeline has ended and should shut down.

    Indicates that a pipeline has ended and frame processors and pipelines
    should be shut down. If the transport receives this frame, it will stop
    sending frames to its output channel(s) and close all its threads. Note,
    that this is a control frame, which means it will be received in the order it
    was sent.

    This frame is marked as UninterruptibleFrame to ensure it is not lost when
    an InterruptionFrame is processed. Terminal frames must survive interruption
    to guarantee proper pipeline shutdown.
```

순서(ordering), flush 의미론, 그리고 interruption 생존이 모두 *frame의 type 안에서* 표현됩니다:
`ControlFrame`이 data 대비 in-order 전달을 주고, `UninterruptibleFrame` mixin이 barge-in에서 살아남게
만듭니다. frame이 지나갈 때 각 processor 자신의 `__cancel` / shutdown path가 실행됩니다.

realtime_voice에는 `EndFrame`이 없으므로 L231의 `close()`는 명시적인 sequence입니다
([[rtv-pipeline-session]], 24 lines):

```
emit SESSION_CLOSING
  → cancel active generation with semantic_interrupt=False
      (comment L239-241: "Closing a transport is lifecycle cleanup, not a customer interruption")
  → _ingress.put(_STOP)
  → asr.close()
  → await supervisor
  → vad.close()
  → agent.close()
  → tts.close()
  → _audio.close()
  → emit SESSION_CLOSED
  → _events.put(_STOP)
```

그리고 terminal frame이 없기 때문에, loop 종료에는 sentinel object가 대신 필요합니다:

```python
# packages/realtime_voice/realtime_voice/pipeline/session.py L37 — via [[rtv-pipeline-session]]
_STOP = object()
```

`_STOP`은 ingress, event, phrase queue에 push되어 각 loop를 끝냅니다. 그래서 **모든 queue의 type이
`asyncio.Queue[X | object]`가 되고 모든 consumer가 `cast(...)`를 합니다**. type에 붙은 저 `| object`가
`EndFrame`이 없는 것의 직접적이고 눈에 보이는 비용입니다 — queue의 element type이 `VoiceRuntimeEvent`가
아닌 것을 받아들이려고 넓어져야 했고, 그 결과 closed union의 exhaustiveness 보장이 queue boundary에서
끊깁니다.

이건 진짜로 흥미로운 디테일이고 곱씹을 가치가 있습니다: closed union은 *event*에 대해 exhaustive하지만,
lifecycle signal은 event가 아니므로 union을 빠져나갔고 type safety를 데리고 나갔습니다. Pipecat은
lifecycle signal을 sum type *안에* 넣었고 (`EndFrame`은 `ControlFrame`입니다) 따라서 아무것도 넓힐
필요가 없었습니다 — 대신 `EndFrame`이 577개 site 모두가 따져야 할 수도 있는 123개 descendant 중 하나가
더 늘어나는 대가를 치렀습니다.

> **💡 쉬운 설명 — `| object`가 왜 "type safety를 데리고 나갔다"인가**
> `asyncio.Queue[VoiceRuntimeEvent]`였다면 큐에서 꺼낸 값에 대해 `match`를 쓸 때 type checker가
> "네 개 case 다 처리했나?"를 검사해 줍니다. 그런데 종료 신호를 넣으려고 `Queue[VoiceRuntimeEvent | object]`
> 로 넓히는 순간, 꺼낸 값의 static type은 사실상 `object`가 되고 checker는 아무것도 모릅니다. 그래서 소비
> 지점마다 `cast(...)`가 붙고, 그 cast는 사람이 옳다고 **주장**하는 것이지 checker가 **증명**하는 게
> 아닙니다. 즉 exhaustiveness는 union 선언에서는 살아 있지만 queue를 통과하는 순간 죽습니다.

### 3.5 부재 하나 더, 부재도 evidence이므로 보고함

[[rtv-pipeline-session]]은 dead-code 발견을 기록합니다: `clock.py` (16 L)가
`MonotonicClock(Protocol)`과 `SystemMonotonicClock`을 선언하는데, `grep -rn MonotonicClock`은 package
안에서도 dental gateway 안에서도 **어디에서도 import되지 않음**을 찾아냅니다. `VoiceSession`은 L108에서
`now_ns: Callable[[], int] = time.monotonic_ns`를 직접 받습니다. test는 대신 `testing/fakes.py`의
`FakeMonotonicClock`을 주입합니다. `clock.py`는 `__init__.py`에서 re-export되지도 않습니다.

Pipecat에는 살아 있는 대응물이 있습니다 — `FrameProcessor.get_clock()`이
`self.processor_setup.clock`을 반환하고 (`src/pipecat/processors/frame_processor.py` L681–690) clock은
`FrameProcessorSetup`을 통해 모든 processor로 전달됩니다. mechanism으로만 진술하면: 한쪽은 모든
processor가 받는 setup object를 통해 clock을 배선하고, 다른 쪽은 protocol을 선언해 놓고 실제로는 함수를
직접 넘겼습니다. [[ch-13/read]] 전에 이 dead code가 여전히 dead인지 확인하세요 — 6주 된 관측입니다.

### 3.6 framework-extension probe #2

Lina TMR에는 **barge-in confirmation delay**가 필요합니다: VAD가 발화하면 assistant를 즉시 취소하지
말고, 250 ms 기다렸다가 speech가 여전히 있을 때만 취소해서, 기침 한 번이 문장을 죽이지 않게 하는 것.

1. Pipecat에서 그 component는 어디로 갑니까? 그건 `FrameProcessor`입니다 — 그런데 *어느 두 원소
   사이*이고, 그것이 신경 쓰는 frame들은 어느 방향으로 이동합니까? (위에 `link()`, `FrameDirection`,
   `_link_processors`가 있습니다. [[ch-08/read]]의 cascade는 아직 없으니 topology만으로 답하세요.)
2. realtime_voice에서는 어디로 갑니까? `session.py`에서 바뀌는 구체적 method 이름을 대고, 그 변경이
   삽입(insertion)인지 수정(edit)인지 말하세요.
3. 세 번째 질문, 그리고 이게 중요한 질문입니다: 둘 중 어느 쪽에서 그 delay를 **A/B test** 할 수
   있습니까 — 두 path를 다 출시하고 session별로 전환하는 것 — turn logic의 나머지가 들어 있는 파일을
   고치지 않고서?

---

## 4. VAD: 서로 다른 것을 세는 두 개의 hysteresis machine

### 4.1 Pipecat이 구현하는 것

**`src/pipecat/audio/vad/vad_analyzer.py` L25–44**

```python
VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


class VADState(Enum):
    """Voice Activity Detection states.

    Parameters:
        QUIET: No voice activity detected.
        STARTING: Voice activity beginning, transitioning from quiet.
        SPEAKING: Active voice detected and confirmed.
        STOPPING: Voice activity ending, transitioning to quiet.
    """

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4
```

**`src/pipecat/audio/vad/vad_analyzer.py` L47–60**

```python
class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    Parameters:
        confidence: Minimum confidence threshold for voice detection.
        start_secs: Duration to wait before confirming voice start.
        stop_secs: Duration to wait before confirming voice stop.
        min_volume: Minimum audio volume threshold for voice detection.
    """

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME
```

gate는 **conjunction(AND)**이고, state machine은 네 개의 state입니다:

**`src/pipecat/audio/vad/vad_analyzer.py` L202–246**

```python
        while len(self._vad_buffer) >= num_required_bytes:
            audio_frames = self._vad_buffer[:num_required_bytes]
            self._vad_buffer = self._vad_buffer[num_required_bytes:]

            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0
```

`else` branch를 주의 깊게 읽으세요. `STARTING → QUIET`은 **아무것도 emit하지 않는 transition**입니다.
`STARTING`에 들어갔다가 `_vad_start_frames`가 지나기 전에 조용해진 blip은 아무 흔적을 남기지 않습니다:
`UserStartedSpeakingFrame`도 없고, downstream cancellation도 없고, turn도 없습니다. state machine은
false start를 publish하지 않고 폐기하는 mechanism 그 자체입니다.

chunk 크기는 transport가 아니라 analyzer가 고정합니다:

**`src/pipecat/audio/vad/silero.py` L191–197**

```python
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            Number of frames required (512 for 16kHz, 256 for 8kHz).
        """
        return 512 if self.sample_rate == 16000 else 256
```

두 개의 sample rate가 모두 지원됩니다: **16 kHz에서 512 frame, 8 kHz에서 256 frame.** 8 kHz branch가
telephony audio를 분석 가능하게 만드는 바로 그것입니다 (§6.3).

analyzer 위층에 사는 knob 두 개 더:

```bash
$ grep -n "_MODEL_RESET_STATES_TIME" src/pipecat/audio/vad/silero.py
23:_MODEL_RESET_STATES_TIME = 5.0
218:            if diff_time >= _MODEL_RESET_STATES_TIME:

$ grep -n "audio_idle_timeout" src/pipecat/audio/vad/vad_controller.py | head -3
75:        audio_idle_timeout: float = 1.0,
83:            audio_idle_timeout: Timeout in seconds to force speech stop
103:        self._audio_idle_timeout = audio_idle_timeout
```

`VADController(audio_idle_timeout=1.0)` — *"Timeout in seconds to force speech stop"* — 는 발화 도중에
audio 전달을 멈춘 microphone, 예를 들어 끊긴 WebRTC track에 대한 watchdog입니다. 이게 없으면 `SPEAKING`
에 앉아 있는 VAD는 영원히 거기서 나오지 못합니다. 그것을 움직여 줄 것이 아무것도 도착하지 않으니까요.

### 4.2 realtime_voice가 구현하는 것

[[rtv-vad-chunking]]에서, 하나의 state machine을 공유하는 두 개의 구현:

- `EnergyVADConfig` (`vad/energy.py` L15): `speech_rms: float = 500.0`, `min_speech_frames: int = 2`,
  `min_silence_frames: int = 4`. `EnergyVAD.rms()` (L104)는 `array("h")` 위에서 도는 순수 Python
  `math.sqrt(sum(s*s)/n)` — numpy도 없고 model도 없습니다. docstring: *"RMS hysteresis VAD
  intended for fallback and deterministic tests."*
- `SileroVADConfig` (`vad/silero.py` L21): `threshold: float = 0.5`, `min_speech_frames: int = 2`,
  `min_silence_frames: int = 6`. `SileroVAD.process`는 16 kHz mono가 아닌 모든 것에 대해 L58에서
  `ValueError("SileroVAD requires 16 kHz mono PCM")`를 raise합니다. `from_pretrained()` (L44)는
  `silero_vad.load_silero_vad`를 lazy import하고, model은 L123에서 `await asyncio.to_thread(call_model)`
  를 통해 실행됩니다.

parameter 대 parameter ([[rtv-vad-chunking]], [[vad-silero]]) — 이 표는 기술(describe)하는 것이지
순위를 매기지 않습니다:

| | realtime_voice | Pipecat |
|---|---|---|
| confidence gate | `SileroVADConfig.threshold = 0.5` | `VADParams.confidence = 0.7` |
| volume gate | 없음 | `VADParams.min_volume = 0.6`, confidence와 **AND** |
| speech onset | `min_speech_frames = 2` (**frame 수**) | `start_secs = 0.2` (**초**) |
| speech offset | `min_silence_frames = 6` (**frame 수**) | `stop_secs = 0.2` (**초**) |
| state 수 | 2 (`self._speaking: bool`) | 4 (`QUIET / STARTING / SPEAKING / STOPPING`) |
| analysis chunk | transport가 주는 대로 | `num_frames_required()` — 16 kHz에서 512, 8 kHz에서 256 |
| model state reset | `reset()`이 `model.reset_states()`가 있으면 호출 | `_MODEL_RESET_STATES_TIME = 5.0` 초마다 강제 |
| idle-mic watchdog | 없음 | `VADController(audio_idle_timeout=1.0)` |
| 8 kHz path | 없음 — `vad/silero.py` L58에서 `ValueError` | `256` frame branch |
| self-measured latency | `endpoint_latency_ms = self._silence_frames * frame.duration_seconds * 1000` (`energy.py` L79, `silero.py` L89) | observer plane을 통해 보고 (§6.5, [[ch-11/read]]) |

두 system 모두 model 호출을 event loop 밖으로 내보냅니다. mechanism이 다릅니다 — Pipecat은
`ThreadPoolExecutor(max_workers=1)`(`vad_analyzer.py` L16에서 import, L92에서 생성, L191에서 사용)에
대고 `loop.run_in_executor(self._executor, self._run_analyzer, buffer)`를 돌리고, realtime_voice는
`await asyncio.to_thread(call_model)`을 하는데 이건 호출마다 default executor에서 thread를 하나
가져옵니다.

그리고 Pipecat이 수행하는 seconds→frames 변환. §4.3과 §4.4가 둘 다 여기에 의존합니다:

**`src/pipecat/audio/vad/vad_analyzer.py` L159–165**

```python
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
```

(`vad_frames_per_sec`는 오해를 부르는 이름입니다 — 실제로는 chunk당 *초*이고, `512 / 16000 = 0.032`
입니다.) 16 kHz에서는 onset과 offset 모두 `round(0.2 / 0.032) = round(6.25) = 6` chunk가 나옵니다.
[[vad-silero]]와 [[rtv-vad-chunking]]은 둘 다 **7**로 보고합니다. 위 code의 산술은 **6**을 주고, 저는
code를 보고합니다.

### 4.3 단위 불일치(unit mismatch), 공식보다 먼저 계산해 보기

frame 대 seconds 행은 구체적으로 해 볼 가치가 있습니다. 그렇게 안 생겼지만 실은 운영상의 성질이기
때문입니다.

`min_silence_frames = 6`이 뜻하는 것: VAD가 speech를 주장하기를 멈춘 뒤, 연속 여섯 frame이 조용할 때까지
기다렸다가 turn이 끝났다고 선언한다. 그러면 여섯 frame은 얼마나 긴가요?

- transport가 audio를 **20 ms** frame으로 전달 → 6 × 20 ms = end-of-turn 대기 **120 ms**.
- transport가 audio를 **100 ms** frame으로 전달 → 6 × 100 ms = end-of-turn 대기 **600 ms**.

같은 상수, 같은 code, 그런데 **time-to-first-response에서 480 ms의 차이**가, VAD가 한 번도 들여다보지
않는 transport의 성질에 의해 결정됩니다. [[rtv-vad-chunking]]은 code 어디에도 frame duration을 assert
하는 것이 없다고 기록합니다. 그리고 이 stack에서 frame duration은 한 번 고르는 상수가 아닙니다:
[[rtv-webrtc-transport]]는 `InboundAudioPump`가 **PyAV resampler output마다 `AudioFrame` 하나**를
emit한다고 기록하므로, 크기는 브라우저가 보낸 것을 resampler가 만들어 낸 그 크기입니다.

Pipecat은 threshold를 초 단위(`start_secs` / `stop_secs`)로 표현하고 analysis chunk 자체를
`num_frames_required()`로 고정함으로써 이 coupling을 피해 갑니다. 그래서 transport의 framing이 state
machine에 아예 도달하지 않습니다. 이건 두 mechanism에 대한 기술(description)이지 순위가 아닙니다:
frame-count 정식화는 frame 크기가 고정돼 있을 때 더 단순하고 정확합니다. seconds 정식화는 변환 하나와
고정 크기 re-chunking buffer를 비용으로 치르고 transport에 대해 불변(invariant)입니다.

이제 공식. 먼저 필요하지는 않았던 것:

```
endpoint_wait_seconds  =  min_silence_frames × frame_duration_seconds      # realtime_voice
endpoint_wait_seconds  =  stop_secs                                        # Pipecat
```

첫 줄의 우변에는 VAD가 통제하지 않는 항이 들어 있습니다. 그게 관측의 전부입니다.

> **💡 쉬운 설명 — 왜 "단위"가 운영 사고를 만드나**
> "6 frame 기다려"는 "여섯 걸음 걸어"와 같습니다. 걸음 폭을 누가 정하는지 안 적혀 있죠. 브라우저와
> resampler가 걸음 폭을 정하고, 그건 배포 환경마다 다릅니다. "0.2초 기다려"는 걸음 폭과 무관합니다.
> 그래서 같은 코드가 개발 노트북에서는 120 ms, 어떤 고객의 브라우저에서는 600 ms의 end-of-turn 지연을
> 내고, 이건 코드 리뷰로는 절대 안 잡힙니다 — 상수는 6으로 똑같으니까요.

### 4.4 두 frame짜리 blip, 두 machine에서 추적하기

speech처럼 보이는 에너지 두 frame — 기침, 의자 소리, 한국어 backchannel "네" — 을 assistant turn 도중에
각 machine에 주입합니다.

**Pipecat.** `speaking`은 `confidence >= 0.7 and volume >= 0.6`입니다. 두 chunk 동안 둘 다 충족됐다고
합시다. `QUIET → STARTING`, `_vad_starting_count`가 2에 도달합니다. `_vad_start_frames`는 위 L159–165
변환을 통해 `start_secs = 0.2`에서 유도됩니다 — 16 kHz에서 512-frame chunk는 각각 32 ms이므로
`_vad_start_frames = round(0.2 / 0.032) = 6` chunk가 필요합니다.
frame 3이 조용함 → `else` branch가 `case VADState.STARTING: self._vad_state = VADState.QUIET`을
발화시킵니다. 아무것도 emit되지 않습니다. assistant는 계속 말합니다.

**realtime_voice.** `min_speech_frames = 2`이고 state는 bool입니다. `threshold = 0.5`를 넘는 두 frame이
— `min_volume` conjunct 없이 — `self._speaking`을 `True`로 뒤집고 진짜 `SPEECH_STARTED`를 emit합니다.
[[rtv-vad-chunking]]에 따르면 `VoiceSession._on_speech_started` (L284)가 그다음 **generation을 전진시키고
assistant를 취소**하며, [[rtv-pipeline-session]]에 따르면 그 취소를 await하기 *전에* VAD event를
emit합니다 — comment L290–292: *"Publish media invalidation before awaiting provider/Gateway
cancellation."*

두 동작 모두 각 machine이 쓰인 대로 하는 것입니다. two-state machine에는 "어쩌면 speech"에 대한 표현이
없으므로 가설을 붙들 수 없습니다. four-state machine의 `STARTING` state가 바로 그 표현입니다. 가설을
붙드는 것이 늘어난 onset latency만큼의 값어치가 있는지는 실제 hold music과 실제 backchannel이 있는 한국어
tele-sales call에 관한 제품 질문이고, 어느 code listing으로도 결판나지 않습니다.

→ 여기서 figure의 **세 번째 panel**을 쓰세요: 같은 주입 blip을 두 machine에 통과시키고 각각이 frame별로
무엇을 하는지만 보고합니다. blip을 2 frame으로 한 번, 8 frame으로 한 번 돌리고, 두 동작이 어디에서
수렴하는지 기록하세요.

### 4.5 이 layer에서 Pipecat 대응물이 없는 realtime_voice mechanism 하나

`VoiceSessionConfig.vad_prefix_frames = 5`는 pre-speech frame들의 `deque(maxlen=5)`(`session.py`
L131)를 유지하고, 이는 `SPEECH_STARTED`에서 **ASR로 replay**됩니다 (L296–299). 그래서 발화의 첫
~100 ms — 사용자가 실제로 말을 시작했을 때 아직 `QUIET`이었던 부분 — 이 손실되지 않습니다
([[rtv-vad-chunking]]).

Pipecat의 대응물은 존재하지만 이 layer에 노출되어 있지 않습니다: `SegmentedSTTService`의
docstring(`src/pipecat/services/stt_service.py` L797–813)은 *"Maintains a small audio buffer to
account for the delay between actual speech start and VAD detection"* 이라고 말합니다 — VAD의 tunable이
아니라 STT service 내부의 buffer입니다. 같은 문제, 다른 소유자, 그리고 둘 중 하나만 그 깊이를 config
field로 만듭니다.

---

## 5. Speech-to-text: streaming interface 대 unary call

### 5.1 Pipecat이 선언하는 것

Pipecat의 STT layer에는 base class가 **두 개** 있고, 둘의 차이가 정확히 쟁점이 되는 mechanism입니다.

**`src/pipecat/services/stt_service.py` L51–70**

```python
class STTService(AIService):
    """Base class for speech-to-text services.

    Provides common functionality for STT services including audio passthrough,
    muting, settings management, and audio processing. Subclasses must implement
    the run_stt method to provide actual speech recognition.

    Includes an optional keepalive mechanism that sends silent audio when no real
    audio has been sent for a configurable timeout, preventing servers from closing
    idle connections (e.g. when behind a ServiceSwitcher). Subclasses that enable
    keepalive must override ``_send_keepalive()`` to deliver the silence in the
    appropriate service-specific protocol.

    A streaming STT reports latency through TTFB — speech end to final transcript —
    and not through processing metrics. Audio arrives continuously, so there is no
    discrete request whose duration a
    :meth:`~pipecat.processors.frame_processor.FrameProcessor.start_processing_metrics`
    window could measure; anchoring one to a speech or turn boundary measures how
    long the user talked. :class:`SegmentedSTTService` does issue a discrete
    request per utterance, so its subclasses time that call and report both.
```

**`src/pipecat/services/stt_service.py` L797–813**

```python
class SegmentedSTTService(STTService):
    """STT service that processes speech in segments using VAD events.

    Uses Voice Activity Detection (VAD) events to detect speech segments and runs
    speech-to-text only on those segments, rather than continuously.

    Requires VAD to be enabled in the pipeline to function properly. Maintains a
    small audio buffer to account for the delay between actual speech start and
    VAD detection.

    The buffered segment is passed to :meth:`run_stt` as a WAV container by
    default, which is what cloud providers want for their upload APIs. Local
    models that consume raw 16-bit PCM directly override
    :attr:`wants_wav_segments` to return ``False`` so they receive the
    unwrapped buffer instead. This is a subclass-level contract, not a
    user-configurable option: the format is dictated by what the model expects.
    """
```

두 docstring을 서로 맞대고 읽으세요. Pipecat은 segmented 대신 streaming을 *고른* 것이 아닙니다 — 둘 다
출시하고, 첫 번째 docstring이 그 선택이 latency를 *측정 가능한 방식 자체*를 어떻게 바꾸는지 설명합니다:
streaming STT에는 discrete request가 없으므로 speech-end에서 final transcript까지의 TTFB를 보고하고,
segmented STT에는 request가 있으므로 그 호출 시간을 잽니다. 두 output을 두 frame type이 나릅니다 —
`frames.py` L450의 `TranscriptionFrame`과 L476의 `InterimTranscriptionFrame`.

### 5.2 realtime_voice가 구현하는 것

[[rtv-vs-pipecat-gap]] 기준: `OpenAICompatibleUnaryASR`은 **발화 전체를 WAV로 buffer한 다음
`finalize()`에서 `audio.transcriptions.create` 호출을 한 번** 합니다 — `openai_compat.py` L194–242,
`timeout_seconds=1.5`. Pipecat의 어휘로 말하면 이건 `SegmentedSTTService` 모양입니다. package 안에
streaming 모양은 없습니다.

type system은 streaming 모양이 **의도되었다**고 말합니다. `ASREventKind.INTERIM`과
`ASREventKind.END_OF_TURN`은 **`types.py`에 선언되어 있고**, [[rtv-vs-pipecat-gap]]에 따르면 **어떤 실제
provider도 emit하지 않습니다** — test fake(`testing/fakes.py`의 `FakeStreamingASR`)만 emit합니다.
§3.3의 `StreamingASR` Protocol은 streaming signature를 가지고 있고 —
`def events() -> AsyncIterator[ASREvent]` — 출시된 유일한 구현은 그것을 통해 정확히 하나의 final event를
yield합니다.

이건 closed union의 또 다른 얼굴이고, 정확히 이름 붙일 가치가 있습니다: union이 interim case를 *선언*
하고, `mypy`가 모든 consumer에게 그것을 handle하라고 요구하고, wire mapper `_control_event`가 그것을
`"transcript.interim"`으로 mapping하고, consumer 동작을 assert하는
`test_interim_asr_is_observable_but_never_calls_agent`라는 이름의 test까지 있습니다. event의 downstream에
있는 모든 것이 만들어졌고 검증되었습니다. 없는 것은 producer뿐입니다. exhaustive type system은 그 case가
존재한다고 말해 줍니다. 하지만 그 무엇도 그것을 구성(construct)하지 않는다는 것은 말해 줄 수 없습니다.

> **💡 쉬운 설명 — type system이 못 보는 구멍**
> type checker는 "이 값이 오면 어떻게 처리할래?"는 강제하지만 "이 값이 실제로 만들어지긴 하냐?"는 묻지
> 않습니다. `INTERIM`은 완벽하게 handle되고, wire로 mapping되고, test까지 있는데, production에서는 한
> 번도 발생하지 않습니다. 즉 **exhaustiveness는 소비 측 보장이지 생산 측 보장이 아닙니다.** 그래서
> §11의 질문 2 — "지난 incident 중 몇 개를 mypy가 잡았을까?" — 가 뾰족한 질문인 겁니다.

### 5.3 그 호출이 timeline의 어디에 떨어지는가

judgement 아닌 mechanism:

```
realtime_voice, per turn:
  [user speaks] ... [VAD: min_silence_frames × frame_dur] → finalize()
      → one HTTPS round trip, whole-utterance WAV upload, timeout_seconds=1.5
      → first AgentTextDelta possible
      → KoreanPhraseChunker holds until first sentence boundary
      → first TTS request → first audible sample

Pipecat with a streaming STTService, per turn:
  [user speaks] → partial transcripts arriving during speech
      → [VAD stop_secs] → final transcript (already mostly computed)
      → first LLM token
      → TTS
```

첫 번째 모양에서 transcription round trip은 endpoint 결정 **뒤에 직렬(serial)**입니다. 두 번째에서는
사용자 자신의 발화와 겹칩니다(overlap). boson에 대한 `CLAUDE.md`의 목표 — [[rtv-vs-pipecat-gap]]에
인용됨 — 는 *"P50 at or below 1.0 seconds and P95 at or below 1.5 seconds,"* 이며, 측정은
*"from the last voiced user sample to the first audible assistant sample, including end-of-turn/VAD
time."* 입니다. 위의 두 항 모두 그 측정 window 안에 있습니다. [[ch-06/read]]가 turn-boundary chain을
제대로 분해하고 [[ch-11/read]]가 예산 산술을 합니다. 이 chapter의 일은 두 모양을 같은 timeline에 올려
각각에서 어떤 항이 직렬인지 볼 수 있게 하는 것뿐입니다.

---

## 6. breadth, 특성화가 아니라 카운트

이 section의 모든 것은 `ls` 아니면 `grep`입니다. 다시 돌려볼 수 있도록 command를 인쇄해 둡니다.

### 6.1 service provider

```bash
$ ls -d src/pipecat/services/*/ | wc -l
62
```

Pipecat에 **62개의 service directory**. realtime_voice는 **2개**의 provider를 출시합니다 — Boson ASR과
Boson TTS — 둘 다 OpenAI-compatible이고, [[rtv-vs-pipecat-gap]] 기준 둘 다 하나의 478-line 파일 안에
있습니다. `BosonHiggsASR(OpenAICompatibleUnaryASR)`은 구성상 in-house-model 전용입니다: 이 package는
당신 자신의 model과 대화하도록 쓰였고, 실제로 그렇게 합니다.

### 6.2 transport

```bash
$ ls -d src/pipecat/transports/*/
src/pipecat/transports/daily/
src/pipecat/transports/heygen/
src/pipecat/transports/lemonslice/
src/pipecat/transports/livekit/
src/pipecat/transports/local/
src/pipecat/transports/moq/
src/pipecat/transports/smallwebrtc/
src/pipecat/transports/tavus/
src/pipecat/transports/vonage/
src/pipecat/transports/websocket/
src/pipecat/transports/whatsapp/
```

**11개 transport package.** realtime_voice는 **1개** — aiortc WebRTC — 에 test용 `FakeVoiceTransport`를
더해 출시합니다.

(정확성을 위한 note: [[rtv-vs-pipecat-gap]]은 "12 transports"라고 말하면서 열한 개의 directory 이름을
나열합니다. 이 commit에서의 directory 수는 11입니다. 제가 직접 돌린 count를 보고합니다.)

두 WebRTC 구현은 크기와 library 선택에서 비슷합니다 — 둘 다 aiortc, 둘 다 PyAV resampling, 둘 다 data
channel을 운용합니다:

```bash
$ wc -l src/pipecat/transports/smallwebrtc/*.py
     825 connection.py
     266 request_handler.py
    1085 transport.py
    2176 total
```

이에 대비해 realtime_voice는 `manager.py` 248, `control.py` 226, `peer.py` 231, `tracks.py` 216,
`buffer.py` 123, `config.py` 64, `transport.py` 168로 총 ~960 L입니다 ([[rtv-webrtc-transport]]).

### 6.3 telephony serializer

```bash
$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

**6개의 telephony serializer** — `exotel`, `genesys`, `plivo`, `telnyx`, `twilio`, `vonage` — 에
`protobuf`(telephony 아님)와 base class가 더 있습니다. realtime_voice는 **0개**를 출시하고, 지금 쓰인
대로는 하나도 host할 수 없습니다: `SileroVAD.process`가
`ValueError("SileroVAD requires 16 kHz mono PCM")`를 raise하고 μ-law path가 없습니다
([[rtv-vad-chunking]], [[rtv-vs-pipecat-gap]]). [[rtv-vs-pipecat-gap]]은 boson의 `CLAUDE.md`가
*"future SIP/RTP or telephony adapters"* 를 의도로 명시한다고 기록합니다. serializer를
transport-adapter로 보는 아이디어가 본격적으로 전개되는 곳은 [[ch-05/read]]입니다.

한국어 보험 tele-sales agent에게 이 행은 두 번 읽을 가치가 있고, 그런 다음 [[ch-13/read]] 전까지는 그에
따라 행동하지 않을 가치가 있습니다.

### 6.4 connection recovery

Pipecat의 WebRTC connection은 renegotiation과 ICE restart를 일급 method로 노출합니다:

```bash
$ grep -n "def renegotiate\|def ask_to_renegotiate\|def pc_id" src/pipecat/transports/smallwebrtc/connection.py
302:    def pc_id(self) -> str:
443:    async def renegotiate(self, sdp: str, type: str, restart_pc: bool = False):
799:    def ask_to_renegotiate(self):
```

그리고 request handler의 DTO가 restart flag를 끝에서 끝까지 나릅니다:

**`src/pipecat/transports/smallwebrtc/request_handler.py` L25–41**

```python
@dataclass
class SmallWebRTCRequest:
    """Small WebRTC transport session arguments for the runner.

    Parameters:
        sdp: The SDP string (Session Description Protocol).
        type: The type of the SDP, either "offer" or "answer".
        pc_id: Optional identifier for the peer connection.
        restart_pc: Optional whether to restart the peer connection.
        request_data: Optional custom data sent by the customer.
    """

    sdp: str
    type: str
    pc_id: str | None = None
    restart_pc: bool | None = None
    request_data: Any | None = None
```

realtime_voice에는 둘 다 없습니다: [[rtv-webrtc-transport]] 기준 유일한 recovery path는 새로운
`accept_offer(reconnect=True)`입니다. 거기 없는 것 또 있습니다: video (`RawVideoTrack`), screen share,
그리고 WebRTC 외의 어떤 transport도 — 마지막 것은 명시적 구성에 의한 것입니다.
`WebRTCTransportConfig.__post_init__`이 *"the first WebRTC transport supports mono assistant audio."*
라며 `output_channels != 1`을 거부하기 때문입니다.

### 6.5 observability

Pipecat에는 chain 안에 들어가지 않고도 모든 hop의 모든 frame을 보는 read-only plane이 있습니다:

**`src/pipecat/observers/base_observer.py` L90–97**

```python
class BaseObserver(BaseObject):
    """Base class for pipeline frame observers.

    Observers can view all frames that flow through the pipeline without
    needing to inject processors into the pipeline structure. This enables
    non-intrusive monitoring capabilities such as frame logging, debugging,
    performance analysis, and analytics collection.
    """
```

```bash
$ ls src/pipecat/observers/
__init__.py  base_observer.py  loggers  startup_timing_observer.py
turn_tracking_observer.py  user_bot_latency_observer.py
```

processing hook(`observer.on_process_frame`, §2.3에 인용, `frame_processor.py` L835)과 push
hook(`observer.on_push_frame`, §3.1에 인용, L1181) 둘 다 이미 base class에 배선되어 있으므로, observer는
`link()` 변경 한 줄 없이 pipeline을 봅니다.

realtime_voice의 instrumentation은 [[rtv-vs-pipecat-gap]]과 [[rtv-webrtc-transport]] 기준으로:
event에 붙는 `provider_latency_ms` / `endpoint_latency_ms` field, 14개의 `VoiceEventKind` 값을 data
channel로 fan-out하는 `VoiceEvent` stream, 그리고 `BoundedAudioOutput.discarded_frames` — 노출되어
있지만 excerpt 기준으로 **아무것도 읽지 않는** counter. OTel도 없고, span도 없고, aggregation도
없습니다. 그 모양에 주목하세요: *event stream* 자체는 풍부하고 typed이며 client로 갑니다. 없는 것은 그
위의 aggregation layer입니다. observer plane과 latency budget이 함께 다뤄지는 곳은 [[ch-11/read]]입니다.

---

## 7. realtime_voice가 구현했지만 Pipecat에 대응물이 없는 것

이 section의 모든 것은 반대 방향으로도 확인했습니다: 각 항목에 대해 Pipecat tree를 grep해서 대응물을
찾아보고 무엇을 찾았는지 보고합니다. 여기서 "대응물 없음"은 "commit `0cbf9c5b`에서 찾아봤지만 못
찾았다"는 뜻이지 "만들 수 없다"는 뜻이 아닙니다.

이 각각은 figure의 다섯 번째 panel에서 체크표시가 아니라 **algorithm**으로 펼쳐집니다. 거기서 열어
보세요. 체크표시였다면 정작 중요한 부분을 건너뛰게 됐을 겁니다.

### 7.1 `KoreanPhraseChunker` — 283 lines

[[rtv-vad-chunking]]에서:

```
__init__(*, min_chars=12, max_chars=60, hard_max_chars=None,
         batch_max_chars=320, adaptive_batching=True)
# when hard_max_chars is None it resolves to min(batch_max_chars, max_chars * 2)   (L56-60)
```

docstring L28–34: *"Adaptive mode emits the first complete sentence immediately, batches the next two
complete sentences, then holds the remaining response as one final group until `flush`. `max_chars`
is a soft latency target rather than an immediate cut point."*

그건 **1 → 2 → bounded-tail schedule**이고, `_accept_adaptive`(L115–149)에서 `_batch_phase` 0/1/2로
구현되어 있습니다. phase 0은 문장이 완성되는 순간 하나를 보내서 time-to-first-audio를 최적화하고,
phase 1은 TTS request overhead를 쌍 단위로 amortize하고, phase 2는 고객이 이미 듣고 있는 상태이므로
꼬리를 더 잘게 쪼개기를 멈춥니다.

`_BoundaryKind`는 `SENTENCE / SOFT / OVERLONG / FINAL_TAIL`의 `StrEnum`이고,
`_STRONG_END = frozenset(".!?。！？\n")`, `_SOFT_END = frozenset(",，;；:")`,
`_CLOSING_PUNCTUATION = frozenset("\"'”’)]}」』】")` 위에서 동작합니다 — strong-end 집합이 CJK를 인지하고
closing 집합에 `」』】`가 포함되어 있다는 점에 주목하세요.

한국어 특유의 지식이 들어 있는 곳은 guard들입니다:

- `_is_safe_period` (L255)는 `1.5` 안의 점, `...` 안의 점, 그리고 ASCII token 문자들 사이의 점에서
  자르기를 거부합니다. comment L266–269: *"A dot between ASCII token characters belongs to a model name,
  hostname, abbreviation, or identifier rather than ending a Korean sentence."* 그것이 `gpt-4.1`과
  domain name이 두 개의 TTS request가 되지 않게 막아 주는 rule입니다.
- `_is_numeric_separator` (L277)는 `_SOFT_END`의 comma로부터 `1,000`을 보호합니다.
- `_INTERNAL_TAG = re.compile(r"\[(?:interruption|system|tool|objection|customer|assistant)[^\]]*\]")`
  는 gateway control tag를 말해질 text에서 벗겨내면서 `start_char` / `end_char`로 **원본(source)** span을
  그대로 유지합니다 — 그래야 §7.2의 ledger가 말해진 문자들을 원래 문자열로 다시 mapping할 수 있습니다.

마지막 성질이 눈여겨볼 것입니다: 벗겨내기와 span 보존이 같이 이루어지는 이유는, downstream consumer가
"*원본* text의 어느 문자들이 실제로 들렸는가"에 답해야 하기 때문입니다.

**Pipecat 쪽, 확인함.** Pipecat의 TTS service들은 generic heuristic으로 자체 문장 분할을 합니다.
1→2→tail schedule도 없고, 한국어 numeric/identifier guard도 없고, span 보존을 동반한 tag 제거도
없습니다. [[tts-service-interface]]와 [[ch-07/read]] 참조.

### 7.2 `AudioTextPlayoutLedger` — 110 lines

[[rtv-vad-chunking]]에서. `GenerationId`를 key로 하는 네 개의 dict: `_phrases`(
`PhrasePlayout(request, sample_start, sample_end, complete)`의 list), `_by_phrase`, `_next_sample`,
`_played_sample`. `begin(request)` / `append(request, sample_count) -> (start, end)` /
`finish(request)`가 TTS가 stream되는 동안 text→sample map을 만들고,
`acknowledge(generation_id, played_sample)`가 `max(current, played_sample)`로 client cursor를
움직입니다 — monotonic하므로 늦게 도착한 acknowledgement가 cursor를 되돌릴 수 없습니다.

L74의 `audible_text()`가 성과물입니다. cursor까지 phrase들을 걸어가고, 부분적으로만 재생된 phrase에
대해서는 다음을 계산합니다:

```
ratio = (cursor - sample_start) / (sample_end - sample_start)
heard = text[: int(len(text) * ratio)]
```

그 식에서 곧바로 두 가지 성질이 떨어져 나오고 둘 다 load-bearing입니다:

1. TTS로부터 **sample count 말고는 아무것도 필요하지 않습니다**. word timestamp도, alignment metadata도
   필요 없습니다. PCM을 내보내는 어떤 TTS든 동작합니다.
2. 그것은 **character-per-sample의 선형 근사**입니다. 단어 내부에서, 그리고 말하기 속도가 변하는 phrase
   전체에 걸쳐, 반환되는 character index는 alignment가 아니라 추정치입니다.

L98의 `playout_complete()` — 모든 phrase가 `complete`이고 **동시에** `played_sample >= queued_samples` —
는 `_cancel_generation`(`session.py` L502–507)이 `semantic_interrupt` flag를 올바로 설정하게 해 주는
술어(predicate)입니다. 즉 *"고객이 내 말을 끊었다"* 와 *"내 turn을 다 마쳤고 고객이 답했다"* 를 구별해
줍니다. 이 둘은 서로 다른 conversation history와 gateway 안의 서로 다른 stage transition을 만들어 내므로,
그 구별은 겉치레가 아닙니다.

**Pipecat 쪽, 확인함.** Pipecat은 관련된 보장을 산술이 아니라 구조적으로 달성합니다: assistant context
aggregator가 `transport.output()` **뒤에** 앉아 있으므로 실제로 방출된 text만 볼 수밖에 없고, 그것은
word-timestamp가 붙은 `TTSTextFrame`들에 의해 페이싱됩니다 ([[interruption-cascade]],
[[ch-08/read]]). context에 `[interrupted]` marker를 쓰지 않습니다. input 요구사항이 다른 두 개의 다른
mechanism입니다 — 하나는 timestamp를 내보내는 TTS가 필요하고, 다른 하나는 sample counter가 필요합니다.
Pipecat tree에는 ledger object가 없습니다.

> **💡 쉬운 설명 — sample-ratio 근사, 숫자로 한 번**
> phrase text가 `"안녕하세요, 김민수 고객님"` (13자)이고 TTS가 그 phrase에 대해 sample 24,000개를
> 큐잉했다고 합시다 (24 kHz면 1초). 고객이 중간에 끊었고 client가 `played_sample = 30,000`을 ack했는데
> 이 phrase의 구간이 `sample_start=18,000`, `sample_end=42,000`이라면:
> `ratio = (30000-18000) / (42000-18000) = 0.5` → `heard = text[:int(13*0.5)] = text[:6]` =
> `"안녕하세요,"`. 여기서 가정은 **문자마다 재생 시간이 같다**는 것입니다. 실제로는 "김민수" 같은 고유
> 명사가 더 느리게 발음될 수 있어 실제 들린 지점과 몇 글자 어긋납니다. 그 대신 얻는 것은 TTS가 word
> timestamp를 안 줘도 된다는 것 — vendor 요구사항이 sample 개수 하나뿐이라는 점이 이 설계의 전부입니다.

→ figure의 **네 번째 panel**이 이것을 위해 만들어져 있습니다: 단어 중간 slider를 끌어서 선형 근사가 진짜
alignment에서 어디서부터 벌어지는지 보고, 그다음 "timestamp-less TTS" 토글을 뒤집어 두 mechanism 중
어느 쪽이 여전히 input을 갖는지 보세요.

### 7.3 `WebRTCSessionManager` — 248 lines

[[rtv-webrtc-transport]]에서. `manager.py` L51, docstring: *"Create short-lived authorized sessions
and enforce one live peer each."*

- `create_session(customer_id, *, session_id=None, metadata=None) -> VoiceSessionTicket(session_id, token, expires_at, customer_id)`
- `secrets.token_urlsafe(32)`를 발급하고 **오직** `hashlib.sha256(token).digest()`만 저장
- `_authorize` (L227)가 만료를 확인한 뒤 `hmac.compare_digest`
- `session_token_ttl_seconds = 15 * 60`
- `accept_offer(..., reconnect: bool = False)`는 `reconnect=True`가 아니면
  `SessionConflictError("this voice session already has a live peer")`를 raise — **명시적 reconnect,
  조용한 탈취(silent takeover) 없음**
- `accept_offer(sdp, type="offer")`는 `SignalingError("only an SDP offer may be accepted")`를 raise —
  answer 전용이고, server가 먼저 시작하는 일은 없음
- session별 동시성은 `_ManagedSession` 위의 `asyncio.Lock`인데 **`accept_offer`만 그 lock을 잡습니다** —
  `send_audio`, `activate_generation`, `send_control`은 lock 없이 `session.peer`를 읽고 그것이 `None`
  이면 `False`/`0`을 반환합니다. 이건 "media path를 signaling 때문에 절대 막지 않는다"는 의도적 선택
  입니다

**Pipecat 쪽, command와 함께 확인함:**

```bash
$ grep -rn "token_urlsafe\|compare_digest" src
src/pipecat/runner/run.py:324:    if not hmac.compare_digest(expected, sig):
src/pipecat/transports/whatsapp/client.py:181:        if not hmac.compare_digest(expected_signature, received_signature):
```

두 hit 모두 webhook signature 확인이지 voice-session authorization이 아닙니다.
`SmallWebRTCConnection.__init__`은 `ice_servers`와 `connection_timeout_secs`를 받고 그 외에는 아무것도
받지 않습니다:

**`src/pipecat/transports/smallwebrtc/connection.py` L245–248**

```python
    def __init__(
        self,
        ice_servers: list[str] | list[IceServer] | None = None,
        connection_timeout_secs: int = 60,
    ):
```

token도, TTL도, customer binding도 없고 `request_handler.py`는 맨 offer/answer endpoint입니다. Pipecat의
설계에서 이건 application-layer 영역이고, realtime_voice는 그것을 transport package 안에 넣었습니다.

### 7.4 `ControlEvent` v1 — `control.py`의 226 lines

[[rtv-webrtc-transport]]에서. L25의 `@dataclass(frozen=True, slots=True)`이고 field는
`session_id, type, sequence, payload, turn_id, generation_id, version=CONTROL_PROTOCOL_VERSION`,
여기서 `CONTROL_PROTOCOL_VERSION = 1`입니다.

docstring L28–30: *"Audio bytes are intentionally prohibited. Microphone and assistant audio belong
on RTP tracks, never in JSON or base64."*

그리고 그 금지는 문서화된 것이 아니라 강제됩니다. `_reject_audio_payload` (L117)는 payload를
**재귀적으로** 걸어가며 다음에 대해 raise합니다:

- `{"audio","audio_base64","audio_data","base64_audio","pcm","pcm16","wav"}` 중 하나로 정규화되는 모든 key
- `data:audio/`로 시작하는 모든 string
- 모든 `bytes` / `bytearray` / `memoryview`

`from_json` (L64)은 알 수 없는 top-level field를
`SignalingError(f"unexpected control fields: ...")`로 거부하고, object가 아닌 payload를 거부하고,
version 불일치를 거부합니다.

`OrderedControlChannel` (L136)은 부분 신뢰성(partially-reliable) channel을 **생성 시점에** 거부합니다:
`ordered=False`이거나, `maxRetransmits`가 `None`이 아니거나, `maxPacketLifeTime`이 `None`이 아니면 각각
`SignalingError`를 raise하며(L147–154), comment는 *"Control events must not silently disappear."*
입니다. `receive()`는 엄격한 in-order 전달을 강제합니다 —
`SignalingError(f"out-of-order control event: expected {self._next_inbound}, received {event.sequence}")`
— 그리고 `send()`는 `max_control_message_bytes = 64 * 1024`를 넘는 것을 거부합니다. outbound sequence는
private counter이므로 **ordering의 소유자는 server**이지 client가 아닙니다.

**Pipecat 쪽, 확인함.** data channel은 application event handler까지 쭉 type 없는 passthrough입니다:

```bash
$ grep -n "_on_app_message\|on_app_message" src/pipecat/transports/smallwebrtc/transport.py
66:        on_app_message: Called when an application message is received.
71:    on_app_message: Callable[[Any, str], Awaitable[None]]
261:        async def on_app_message(connection: SmallWebRTCConnection, message: Any):
578:        await self._callbacks.on_app_message(message, sender)
989:            on_app_message=self._on_app_message,
1001:        self._register_event_handler("on_app_message")
1047:    async def _on_app_message(self, message: Any, sender: str):
1051:        await self._call_event_handler("on_app_message", message, sender)
```

`message: Any`. schema도, sequence check도, size cap도, audio 금지도 없습니다. Pipecat에 typed client
protocol이 있긴 합니다 — RTVI ([[rtvi-observability]]) — 하지만 그건 다른 layer에 얹혀 가고 data channel
자체에서 강제되지 않습니다.

§7.3과 §7.4가 같은 모양의 발견이라는 점에 주목하고, 이름을 붙일 가치가 있습니다: 둘 다 plumbing(배관)이
아니라 **policy(정책)**입니다. plumbing(aiortc, SDP, data channel)은 두 system에서 같습니다. 다른 것은
그 policy가 transport package 안에 강제되는 code로 쓰였는가, 아니면 application에 맡겨졌는가입니다.

### 7.5 `GenerationAudioQueue.discard_generation()`

[[rtv-pipeline-session]]에서: `queues.py` L12, 66 lines, 손으로 만든 `deque`에 `asyncio.Condition`을
더한 것 — 정확히 하나의 연산을 위해 그렇게 쓰였습니다.

L42의 `discard_generation(generation_id) -> int`는 **condition lock 아래에서 deque를 atomic filter로
재구축**합니다. `asyncio.Queue`에는 그런 연산이 없습니다: `asyncio.Queue`에서 선택된 항목만 제거하려면
전부 빼내고 다시 넣어야 하는데, 그건 동시 consumer에 대해 atomic하지 않습니다. `close()`는 비우고 모든
waiter를 깨우고, 닫힌 빈 queue에 대한 `get()`은 `QueueClosedError`를 raise합니다.

그것의 transport 쪽 쌍둥이는 `BoundedAudioOutput` (`buffer.py` L21)입니다: `asyncio.Queue(maxsize=64)`
이고 그 `activate_generation(generation_id)` (L53)은 **더 오래된 것을 동기적으로 전부 빼내고 drop 수를
반환**합니다. `put()`은 오래된 generation에 대해 `False`를 반환하고, live peer가 `backpressure_ms = 250`
안에 비워 가지 않으면 `AudioBufferFull`을 raise합니다 ([[rtv-webrtc-transport]]).

**Pipecat 쪽.** Pipecat은 같은 문제를 filtered rebuild가 아니라 priority queue + task cancellation으로
풉니다 — §8 참조.

**그리고 [[ch-13/read]]로 손대지 않고 그대로 들고 갈 발견 하나**, [[rtv-webrtc-transport]]에서:
`WebRTCVoiceTransport` — protocol을 준수하는 adapter — 는 **`activate_generation()`을 절대 호출하지
않습니다**. dental `voice_server.py`가 대신 manager callback을 직접 배선합니다. 그래서 `VoiceTransport`
Protocol을 만족하는 adapter가 두 live path 중 능력이 떨어지는 쪽이고, 능력 있는 path는 abstraction을
우회합니다. 그건 abstraction boundary에 관한 사실이고, 누군가 "그래서 `VoiceTransport` interface 뒤에는
정확히 무엇이 있는가?"라고 물을 때 문제가 될 것입니다.

### 7.6 `StreamingConversationAgent` — 슬롯 그 자체

이 Protocol은 의도를 담은 한 줄입니다 (§3.3):

```python
class StreamingConversationAgent(Protocol):
    def stream(request) -> AsyncIterator[AgentTextDelta]
    async def cancel / close
```

이것은 **오직** `AgentTextDelta`만 yield합니다. tool call도 아니고, context object도 아니고, message도
아니고, text delta입니다. [[rtv-vs-pipecat-gap]]에 따르면 tool은 `packages/basement` +
`packages/gateway`에 살고, `GatewayConversationAgent.stream()` → `bridge.dispatch_transcript()`
(`agents/dental-w-tool-gateway/voice_server.py` L163–205)를 통해 도달합니다. boson 자신의 `CLAUDE.md`가
이 슬롯이 존재하는 이유인 제약을 명시합니다: *"Keep Basement and the dental business logic text-native"*,
그리고 *"Basement and Gateway must not import provider-specific audio code."*

그래서 voice package가 brain과 맺은 계약은 이렇습니다: **text를 달라, 내 text를 가져가라, 그리고 내가
tool이 무엇인지 알게 만들지 마라.**

**Pipecat 쪽.** Pipecat의 LLM service는 context를 소유하고 tool loop를 pipeline 안에서 닫습니다:
`LLMContext(tools=[...])`, `register_function`, `params.result_callback(...)`, 그리고 loop는
`LLMContextFrame`을 **upstream**으로 push함으로써 닫힙니다 ([[function-calling]]). `LLMContextFrame`은
`frames.py` L551에 있는, 어떤 branch에도 속하지 않고 `Frame`을 직접 subclass하는 그 class입니다.

"tool에 대해 아는 것이 금지된 text-in / text-out agent 슬롯"에 대한 Pipecat의 대응물은 없습니다. 두 계약이
공존할 수 있는지, 그리고 그러려면 무엇이 필요한지는 [[ch-09/read]]의 주제 전체입니다. 여기서 해결하지
마세요.

---

## 8. parity를 parity라고 이름 붙이기: barge-in과 cancellation

이 section이 존재하는 이유는 정직한 발견이 *같음(sameness)*이기 때문이고, 차이를 목록화하고 있을 때
같음은 과소 보고되기 쉽기 때문입니다.

두 system 모두 같은 보장에 도달합니다 — *고객이 assistant 위로 말할 때, 이미 큐잉된 assistant audio는
재생되면 안 되고, assistant의 진행 중인 작업은 멈춰야 한다* — 다만 mechanism이 다릅니다.

**Pipecat: 하나의 signal, cascade.** `InterruptionFrame`은 `SystemFrame`입니다:

**`src/pipecat/frames/frames.py` L1142–1150**

```python
class InterruptionFrame(SystemFrame):
    """Frame pushed to interrupt the pipeline.

    This frame is used to interrupt the pipeline. For example, when a user
    starts speaking to cancel any in-progress bot output. It can also be pushed
    by any processor.
    """

    pass
```

`SystemFrame`이라는 것이 그것으로 하여금 모든 queue를 건너뛰게 만드는 요인입니다. priority queue는
`isinstance`로 tier를 배정합니다:

**`src/pipecat/processors/frame_processor.py` L132–143**

```python
class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for the frames arriving at a frame processor.

    Frames are dequeued in three tiers: the `StartFrame` first, then
    `SystemFrame`, then data and control frames. Frames of the same tier keep
    their arrival order.

    """

    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20
```

그리고 `put()`이 `isinstance`로 그것들을 배정합니다. 세 줄짜리입니다:

**`src/pipecat/processors/frame_processor.py` L162–168**

```python
        frame, _, _ = item
        if isinstance(frame, StartFrame):
            priority = self.START_PRIORITY
        elif isinstance(frame, SystemFrame):
            priority = self.SYSTEM_PRIORITY
        else:
            priority = self.DEFAULT_PRIORITY
```

그리고 input task는 system frame을 process queue에 올리지 않고 아예 즉시 실행으로 라우팅합니다:

**`src/pipecat/processors/frame_processor.py` L1304–1307**

```python
            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
```

그다음 각 processor는 그 frame이 도착하면 **자기 자신의** task를 취소합니다:

**`src/pipecat/processors/frame_processor.py` L1130–1150**

```python
    async def _start_interruption(self):
        """Start handling an interruption by cancelling current tasks."""
        try:
            current_is_uninterruptible = isinstance(
                self.__process_current_frame, UninterruptibleFrame
            )
            if current_is_uninterruptible:
                # The frame currently being processed is uninterruptible, so we
                # must not cancel it. Just flush non-uninterruptible frames from
                # the queue; any uninterruptible ones will be kept and processed
                # after the current frame finishes.
                self.__reset_process_queue()
            else:
                # Cancel and re-create the process task. Previously this branch
                # was skipped when the queue contained an uninterruptible frame,
                # which caused slow non-uninterruptible frames to block
                # interruptions. Uninterruptible queued frames are safe here
                # because __create_process_task calls __reset_process_queue
                # internally, which always preserves them.
                await self.__cancel_process_task()
                self.__create_process_task()
```

frame 하나, processor N개, 각자가 자기 cancellation을 책임집니다. 그것이 composable한 정식화입니다:
당신이 내년에 쓸 processor는 이 method를 상속하고 아무것도 하지 않음으로써 interruption에 올바르게
참여합니다.

**realtime_voice: 정수 하나, 모든 곳에서 비교.** [[rtv-pipeline-session]] 기준
`self._active_generation`은 **여섯** 지점에서 비교되고, 각각은 `VoiceEventKind.GENERATION_DROPPED`로
stale한 작업을 버립니다:

| site | file:line |
|---|---|
| `next_audio` | `session.py` L180 |
| `_asr_loop` | L328 |
| `_produce_phrases` | L410 |
| `_consume_phrases` | L447 |
| `_consume_phrases` | L456 |
| `acknowledge_playout` | L199 |

여기에 `GenerationAudioQueue.discard_generation()` (§7.5)이 더해지고, transport 쪽에서는
jitter buffer를 비우는 `BoundedAudioOutput.activate_generation()` (§7.5), 그리고 generation이 바뀔 때
자기 `av.AudioFifo`를 버리는 `OutboundAudioTrack` (`tracks.py` L142–146, L152–154 — cancellation으로서의
jitter-buffer flush)이 더해집니다.

`_on_speech_started` (L284)는 generation을 전진시키고 provider cancellation을 await하기 **전에**
invalidation을 publish합니다 — comment L290–292: *"Publish media invalidation before awaiting
provider/Gateway cancellation"* — 그래서 `test_barge_in_event_precedes_slow_provider_cancellation`
이라는 이름의 test가 있는 것입니다.

같은 보장, 두 가지 모양:

| | Pipecat | realtime_voice |
|---|---|---|
| signal | chain을 순회하는 `SystemFrame` | session 위의 정수 field |
| staleness를 어떻게 결정하는가 | stale한 작업을 처리 중인 task가 cancel됨 | 각 site가 `generation_id`를 비교하고 버림 |
| 어디에서 강제되는가 | base `FrameProcessor`에서 한 번, 모두가 상속 | 이름 붙은 6개 site + buffer 2개, 손으로 작성 |
| 새 component가 참여하는 방식 | 자동으로, `_start_interruption`을 상속함으로써 | 그 author가 비교문을 넣었을 때만 |
| 모든 강제 지점을 열거할 수 있는가 | 아니오 — processor가 있는 곳이면 어디든 | 예 — 6개 site가 한 파일 안에 있음 |

두 column 모두 mechanism에 관한 참인 진술입니다. 이 trade는 §2.6의 그 trade가 옷만 갈아입은 것입니다:
분산되고 자동적인 것 대 중앙집중되고 열거 가능한 것. [[ch-08/read]]가 Pipecat의 cascade를 끝에서 끝까지
분해합니다.

> **💡 쉬운 설명 — 왜 이게 §2.6과 "같은 trade"인가**
> §2.6은 *데이터 종류*에 관한 것이었고 §8은 *취소 정책*에 관한 것이지만, 구조는 동일합니다. Pipecat은
> 규칙을 **base class에 한 번** 넣고 상속으로 퍼뜨립니다 → 새 참여자는 공짜로 옳게 동작하지만, 규칙이
> 실제로 적용되는 지점 전체를 아무도 목록화할 수 없습니다. realtime_voice는 규칙을 **6개 지점에 손으로**
> 씁니다 → 목록이 존재하고 감사(audit) 가능하지만, 7번째 지점을 추가하는 사람이 잊으면 아무도 못
> 잡습니다. "자동이지만 열거 불가" vs "수동이지만 열거 가능"이 이 course 전체를 관통하는 축입니다.

---

## 9. evidence로서의 test 이름

test *개수*는 약한 signal입니다. test *이름*은 강한 signal입니다. 이름은 author가 못 박아 둘 가치가
있다고 믿은 invariant를 진술하기 때문입니다. [[rtv-vs-pipecat-gap]]에서: 6개 파일, 1,504 lines, 60개
test function, `asyncio_mode = "auto"`.

**`test_session_pipeline.py` (483 L, 15 tests)**

- `test_confirmed_barge_in_purges_server_audio_and_cancels_old_work`
- `test_barge_in_event_precedes_slow_provider_cancellation`
- `test_late_stale_asr_final_is_dropped`
- `test_playout_acknowledgement_returns_only_heard_prefix`
- `test_late_playout_ack_cannot_extend_interrupted_audible_prefix`
- `test_next_turn_after_completed_playout_is_not_an_interruption`
- `test_each_generation_has_exactly_one_terminal_event`
- `test_ingress_overflow_rejects_frame_instead_of_growing_latency`
- `test_two_sessions_do_not_share_generation_or_provider_state`
- `test_interim_asr_is_observable_but_never_calls_agent`

**`test_chunking_and_ledger.py` (270 L, 22 tests)**

- `test_safety_split_never_breaks_a_long_latin_token`
- `test_dots_inside_ascii_identifier_are_not_sentence_boundaries`
- `test_overlong_token_waits_across_streamed_deltas_for_whitespace`
- `test_playout_ledgers_are_generation_isolated`

**`test_webrtc_transport.py` (257 L, 6 tests)** 에는 진짜
`test_aiortc_loopback_negotiates_ordered_control_channel` — mock이 아니라 실제 negotiation — 이 있고,
`test_outbound_track_underflow_is_explicit_zero_pcm_silence`가 있는데 이건 `tracks.py` L201에 기록된
PyAV 버그를 못 박습니다: *"PyAV does not guarantee zero-initialized AudioFrame storage. Sending a fresh
allocation as 'silence' can therefore produce full-scale random PCM."* 그건 **silence가 고객 귀에
full-scale noise로 나가는** 버그에 대한 regression test이고, 이름으로 못 박혀 있습니다.

저 목록을 하나의 specification으로 읽고, 그것이 *무엇에 대한* specification인지 보세요. 15개 session
test 중 여섯 개가 interruption, staleness, playout accounting 사이의 상호작용에 관한 것입니다. 목록에
smoke test는 없습니다. "pipeline이 돌아간다"를 assert하는 것도 없습니다. 모든 이름이, 위반되면 전화
통화에서 구체적으로 나쁜 경험을 만들어 내는 invariant를 assert합니다: assistant가 고객 위로 계속 말하거나,
stale한 transcript가 이미 대체된 것에 대한 응답을 촉발하거나, conversation history가 고객이 듣지도 않은
문장을 assistant가 말했다고 주장하거나.

`realtime_voice/testing/fakes.py` (271 L)는 여섯 개의 deterministic double을 **public API**로 출시하며,
`realtime_voice.testing`에서 export합니다: `ScriptedVAD`, `FakeStreamingASR`,
`FakeStreamingConversationAgent`, `FakeStreamingTTS`, `FakeVoiceTransport`, `FakeMonotonicClock`.
determinism은 나중에 덧댄 것이 아니라 처음부터 설계된 것이었고 — 이것이 또한, §5.2에 따르면,
`ASREventKind.INTERIM`에 producer가 있기라도 한 이유입니다.

**플래그하겠다고 말했던 불일치.** [[rtv-vs-pipecat-gap]]은 Pipecat을 *"226 test files / 92,538 L /
3,959 test functions (0.55 ratio)"* 로 보고합니다. commit `0cbf9c5b`에서의 제 재측정은 **236개 test
파일**과 **4,236개 `def test_` match**(그중 4,231개는 공백 뒤 줄머리)를 줍니다. line count는 92,538로
정확히 일치합니다. 236 / 4,236을 쓰세요 — command는 §1에 있고 직접 다시 돌릴 수 있습니다 — 그리고
excerpt의 파일/함수 count는 약간 다른 tree 상태나 다른 matcher에서 나온 것으로 취급하세요.

---

## 10. fact sheet

이것이 이 chapter가 앞으로 넘기는 것입니다. measurement와 absence의 표입니다. recommendation은 담고 있지
않으며, 어떤 행이 한쪽에서 더 길다는 사실은 점수가 아닙니다.

| Layer | realtime_voice (excerpt-attested) | Pipecat (`0cbf9c5b`에서 code-verified) |
|---|---|---|
| unit of data | closed union, `VoiceRuntimeEvent`, `types.py:201`, 9개 frozen dataclass 위의 4개 member; 모든 payload에 correlation ID | open sum type, `Frame` (`frames.py` L65), 2,415-line 파일 안 123개 descendant, 3개 scheduling branch + `UninterruptibleFrame` mixin; base에 correlation field 없음 |
| 새 datum의 extension 비용 | 1줄 + 모든 `isinstance` chain 수정; type checker가 site를 찾아줌 | 아무 데서나 subclass 선언; 136개 파일의 577개 `isinstance(frame, ...)` site가 갱신 필요할 수 있음; 아무도 찾아주지 않음 |
| handle되지 않은 datum | `transport.py:_control_event` (L118–156)에서 `TypeError` | 조용히 drop — `process_frame` (L820–847)에 fallthrough 없음 |
| unit of work | `VoiceSession` 하나, 561 L, `_supervise()` L257 → `TaskGroup` 1개, session당 long-lived task 2개 | `FrameProcessor` + `link()` (L671–679); processor별 runtime은 [[ch-04/read]] |
| topology | 고정된 VAD→ASR→agent→TTS; 5개 `Protocol` 슬롯 교체 가능; interposition 불가 | list 위의 `Pipeline._link_processors()` (L197–202); interposition은 list 수정 |
| direction | 단방향; upstream push 없음 | `FrameDirection.{DOWNSTREAM,UPSTREAM}` (L60–69), `_next` / `_prev` |
| shutdown | `close()` L231, 손으로 순서 매긴 24줄, 모든 queue type을 넓히는 `_STOP = object()` sentinel | `EndFrame(ControlFrame, UninterruptibleFrame)` (L1899)이 전파되고 interruption에서 살아남음 |
| VAD | 2-state bool; `threshold=0.5`; volume gate 없음; frame-count threshold; idle watchdog 없음; 16 kHz mono 전용 | 4-state `VADState`; `confidence=0.7` **그리고** `min_volume=0.6`; 초 기반; `audio_idle_timeout=1.0`; 512@16k / 256@8k |
| ASR로의 pre-roll | `vad_prefix_frames=5`, `SPEECH_STARTED`에서 replay (L296–299) | VAD tunable이 아니라 `SegmentedSTTService` 내부의 buffer |
| STT | unary 전용 — WAV + `finalize()`에서 `audio.transcriptions.create` 한 번; `INTERIM`/`END_OF_TURN`은 선언되었으나 fake만 생산 | `STTService`(streaming) **와** `SegmentedSTTService`(발화 단위)가 별개의 base class |
| TTS | streaming provider 1개 (`OpenAICompatibleStreamingTTS`, 24 kHz PCM) | 62개 service directory 중 하나; [[ch-07/read]] 참조 |
| provider | 2 | 62개 service directory |
| transport | 1 (aiortc WebRTC ~960 L) + fake | 11개 transport package; smallwebrtc는 2,176 L |
| telephony | 0; `SileroVAD`가 8 kHz 거부 | 6개 serializer (`exotel/genesys/plivo/telnyx/twilio/vonage`) |
| recovery | 새 `accept_offer(reconnect=True)`뿐 | `renegotiate()` L443, `ask_to_renegotiate()` L799, request DTO 위의 `restart_pc` |
| session auth | `WebRTCSessionManager`: `token_urlsafe(32)`, SHA-256 digest, `compare_digest`, 15분 TTL, one-live-peer + 명시적 reconnect | transport에는 없음 — `SmallWebRTCConnection.__init__`은 `ice_servers`, `connection_timeout_secs`만 받음 |
| control channel | `ControlEvent` v1: dot 형식 type 검증, 엄격한 in-order `sequence`, 64 KiB cap, 재귀적 `_reject_audio_payload`, 생성 시점 ordered-channel 강제 | `on_app_message(message: Any, sender: str)` — type 없는 passthrough |
| text chunking | `KoreanPhraseChunker` 283 L: 1→2→tail schedule, `_is_safe_period`, `_is_numeric_separator`, source span을 유지하는 `_INTERNAL_TAG` 제거 | 대응물 없음; TTS service들이 generic heuristic으로 분할 |
| playout accounting | `AudioTextPlayoutLedger` 110 L: sample-ratio `audible_text()`, monotonic ack, `playout_complete()` → `semantic_interrupt` | ledger 없음; `transport.output()` 뒤의 aggregator 배치 + word-timestamp가 붙은 `TTSTextFrame` |
| agent slot | `StreamingConversationAgent`가 `AgentTextDelta`만 yield; tool은 계약상 package 바깥에 있음 | LLM service가 context를 소유하고 upstream `LLMContextFrame`으로 pipeline 안에서 tool loop를 닫음 |
| barge-in | 6개 site의 generation-ID 동등성 + `discard_generation()` + `activate_generation()` + FIFO flush | `InterruptionFrame(SystemFrame)` + priority queue (L132–170, L1304) + processor별 `_start_interruption` (L1130–1150) |
| observability | `provider_latency_ms` / `endpoint_latency_ms`, data channel로 가는 14개 `VoiceEventKind` 값, 아무도 읽지 않는 `discarded_frames` counter | `process_frame`과 `push_frame`에 hook된 `BaseObserver` plane; 출시된 observer 3개 + logger |
| 발견된 dead code | `clock.py`의 `MonotonicClock` / `SystemMonotonicClock`이 어디에서도 import되지 않음 | — (이 chapter에서는 감사하지 않음) |
| scale | src 3,886 L / test 1,504 L / test 60개 / 파일 6개 | src 168,847 L / test 92,538 L / `def test_` 4,236개 / 파일 236개 |

**realtime_voice 쪽의 부재, 완곡함 없이 그대로 나열:** streaming STT 없음, telephony path 없음, 8 kHz
없음, renegotiation이나 ICE restart 없음, video 없음, observer plane 없음, metric aggregation 없음,
processor interposition 없음, upstream direction 없음, 두 번째 transport 없음, 두 번째 provider 쌍
없음, 그리고 dead module 하나.

**Pipecat 쪽의 부재, 같은 정도의 담백함으로:** session token / TTL / one-live-peer 강제 없음, data
channel에서의 검증된 control-plane schema 없음, 한국어 phrase chunker 없음, playout ledger 없음, tool을
소유하는 것이 금지된 text 전용 agent 슬롯 없음, 그리고 base frame의 correlation ID 없음.

어느 목록도 아직 의미를 갖는 방식으로 더 길지 않습니다.

---

## 11. 이 chapter가 의도적으로 열어 두는 질문들

[[ch-13/read]] 전에 글로 답하세요. 지금 답하지는 마세요.

1. **다음 component는 누가 쓰는가?** 앞으로 네 분기 동안, `types.py`를 고칠 수 없는 사람이 쓰게 될
   voice-stack component가 몇 개입니까? 답이 0이라면, open sum type은 당신에게 무엇을 사 줍니까? 0이
   아니라면, closed union은 당신에게 무엇을 비용으로 물립니까?
2. **exhaustiveness는 incident 기준으로 실제 얼마짜리인가?** boson의 최근 N개 production voice incident
   중, union에서 `mypy`가 잡았을 것은 몇 개이고, type system이 볼 수 없는 것 — 예를 들어
   `ASREventKind.INTERIM`이 선언되고, handle되고, wire-mapping되고, test까지 있는데 한 번도 생산되지
   않는 것(§5.2) — 때문에 발생한 것은 몇 개입니까?
3. **§7.5의 abstraction boundary는 load-bearing인가?** `WebRTCVoiceTransport`는 `VoiceTransport`
   Protocol을 만족하면서 `activate_generation()`을 절대 호출하지 않고, production path는 그것을
   우회합니다. 그 Protocol은 당신의 system을 기술하고 있습니까, 아니면 당신이 더 이상 돌리지 않는
   system을 기술하고 있습니까?
4. **당신의 production traffic에서 `min_silence_frames = 6`은 지금 이 순간 몇 밀리초입니까?** 원리상이
   아니라 — 측정된 값으로, 실제 한국어 통화에서, 브라우저가 실제로 전달하는 frame 크기들에 걸쳐서.
   §4.3이 산술을 줬고, 녹음은 당신이 가지고 있습니다.
5. **당신의 TTS는 word timestamp를 내보냅니까?** 그 답이 §7.2의 선형 근사가 당신에게 주어진 유일한
   mechanism인지 아니면 둘 중 하나인지를 결정합니다. provider에 대고 한 줄이면 확인되는 것이고, 그것이
   [[ch-08/read]] 논의의 모양을 완전히 바꿉니다.
6. **text 전용 agent 슬롯과 pipeline 내부 tool loop가 공존하기라도 할 수 있는가?** 답하지 마세요.
   [[ch-09/read]]가 그것에 답하려고 만들어졌고, 직관으로 먼저 답해 버리는 것은 evidence 대신 자기 직관을
   방어하게 되는 길입니다.

---

## 다음 챕터로

이 chapter는 세 가지를 앞으로 넘기고, 네 번째는 넘기지 않습니다.

**fact sheet** (§10). 이후 모든 chapter가 여기서 끌어다 씁니다. [[ch-05/read]]가 telephony serializer
여섯 개를 보여줄 때, baseline은 "0개, 그리고 `SileroVAD`는 8 kHz에서 raise한다"입니다.
[[ch-08/read]]가 interruption cascade를 분해할 때, baseline은 "이름 붙은 여섯 site의 generation-ID
동등성에 buffer flush 두 개"입니다. [[ch-11/read]]가 latency budget을 세울 때, baseline은
`endpoint_latency_ms`, `provider_latency_ms`, 그리고 아무도 읽지 않는 `discarded_frames` counter입니다.
이 sheet가 그 chapter들을 관광이 아니라 비교로 만들어 주는 것입니다.

**아직 답하지 않은 framing question.** open sum type 대 closed union은 두 codebase에 관한 질문이
아닙니다. **다음 component를 누가 쓰는가**에 관한 질문이고, 이제 당신은 두 구현을 파일-줄 해상도로 눈앞에
두고 있습니다. mechanics chapter들을 통과하는 동안 이 질문을 열어 두세요. 더 많은 evidence와 함께
[[ch-13/read]]에서 다시 물어볼 것입니다.

**의도적으로 미뤄 둔, 이름 붙은 두 개의 충돌.** agent 슬롯 (§7.6) — realtime_voice의
`StreamingConversationAgent`는 text를 yield하고 tool에 대해 아는 것이 계약상 금지되어 있는 반면, Pipecat의
LLM service는 context를 소유하고 `LLMContextFrame`을 upstream으로 push해서 tool loop를 닫습니다. 그리고
topology 질문 (§3.3) — 슬롯은 교체 가능하고 순서는 아니다. 어느 것도 여기서 해결되지 않습니다.
[[ch-09/read]]가 첫 번째를 소유하고, [[ch-12/read]]가 rule layer를 middleware로 배치할 때 두 번째를
소유합니다.

다음은 [[ch-04/read]]이고, 그것은 이 chapter가 계속 미뤄야만 했던 질문에 답하며 mechanics phase를
엽니다: **`worker.run()`을 호출하면 실제로 무엇이 도는가?** *processor마다* task 두 개와 queue 두 개,
priority tier가 세 개인 `FrameProcessorQueue`, 그리고 `InterruptionFrame`이 audio로 가득 찬 queue를
추월하게 해 주는 out-of-band path. realtime_voice의 답은 이미 봤습니다 — `TaskGroup` 하나, long-lived
task 두 개, overflow policy 세 가지를 가진 bounded queue 네 개. 이제 다른 쪽을 보러 가세요.
