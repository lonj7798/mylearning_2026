---
title: "Pipes and Filters: splicing을 가능하게 만드는 uniform interface"
chapter: ch-01
phase: read
course: pipecat
lang: ko
companion_of: read.md
sources:
  - theory-pipes-and-filters
  - frame-processor
  - pipeline-composition
  - parallel-pipeline
  - processor-vocabulary
  - pipecat-design-philosophy
  - boson-agent-loop
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# ch-01 — Pipes and Filters: splicing을 가능하게 만드는 uniform interface

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 (frame, processor, pipeline, queue, back-pressure, uniform interface,
> sum type, narrow waist 등).

## 왜 이 챕터인가

당신이 Pipecat에서 얻고 싶다고 말한 것은 lego-block 느낌이었습니다. *"중간에 process를 하나 추가하거나
빼는 걸 아주 쉽게 할 수 있다"* — 이 문장이 이 chapter의 주제 전체이고, 가장 먼저 알아둘 가치가 있는
사실은 그것이 **느낌**도 아니고 **Pipecat이 발명한 것**도 아니라는 점입니다. 그것은 1964년 origin memo와
1993/94년 formal write-up, 그리고 정확하게 명시된 비용(cost)을 가진, 이름이 붙은 architectural style입니다.
Pipecat은 그 style의 한 implementation이고, 그 implementation은 이 페이지를 다 읽을 때쯤이면 머릿속에
통째로 담을 수 있을 만큼 작습니다.

이 chapter는 세 가지 질문에 순서대로 답합니다.

1. **정확히 무엇이 splicing을 합법(legal)으로 만드는가?** — 하나의 method signature, 하나의 write verb,
   두 개의 값만 가진 enum 하나, 그리고 body가 세 statement인 `link()`.
2. **공짜로 얻는 algebra는 무엇이고, 그게 실제로 ship되는가?** — identity, zero, associativity, parallel,
   conditional, higher-order — 각각이 diagram이 아니라 `src/`에 있는 실제 class로 witness됩니다.
3. **그 유연함의 대가는 무엇인가?** — 서로 구별되는 두 개의 tax, 그리고 둘 다 *silent* failure를
   만들어냅니다. 그래서 이 chapter는 summary가 아니라 checklist로 끝납니다.

그러고 나서 이 모든 것을 `boson-agent`와 대조합니다. 거기서는 한 turn이 561줄짜리 함수 하나이고,
splice해 넣을 position N 자체가 존재하지 않습니다. 이 대조는 점수를 매기는 작업이 아닙니다. 두 design
모두 그것이 *무엇을 하는지*로 기술될 뿐이고, ch-01부터 ch-12까지 어디에서도 둘의 순위를 매기지 않습니다.
[[ch-13/read]]가 유일하게 무언가를 채점하는 곳이며, 그 채점은 이 chapter들이 공급하는 evidence에
근거해서 이루어집니다.

한 가지 process note — 나머지를 어떻게 읽을지에 영향을 주기 때문에 미리 말합니다.
**이 chapter의 모든 수치는 commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` 시점의 tree에 대해
집필 중에 다시 측정되었습니다.** source excerpt에 있던 두 개의 count는 재측정을 통과하지 못했습니다.
저는 그것들을 조용히 수정된 값으로 바꿔치기하는 대신 inline으로 표시합니다. codebase에 대한 *어떤*
주장이 취약(fragile)한지 아는 것 자체가 이 chapter가 가르치려는 skill의 일부이기 때문입니다.

> 💡 **쉬운 설명 — 왜 "재측정"을 강조하나요?**
> source excerpt는 사람이 쓴 요약이고, codebase는 계속 움직입니다. "131개 override가 모두 같은 shape"
> 같은 문장은 grep 한 번이면 검증되지만, 대부분의 사람은 검증하지 않고 인용합니다. 이 chapter는
> "인용된 숫자는 commit hash와 함께여야 의미가 있다"는 습관을 가르치려고 일부러 틀린 두 숫자를
> 지우지 않고 남겨둡니다.

---

## 1. 이 style에는 이름이 있고, 그 이름에는 청구서가 딸려 있다

### 1.1 McIlroy의 문장이 곧 당신의 문장이다

Doug McIlroy, Bell Labs internal memo, 1964년 10월 11일:

> We should have some ways of coupling programs like garden hose — screw in another segment when
> it becomes necessary to massage data in another way.
>
> (프로그램들을 garden hose처럼 연결하는 방법이 있어야 한다 — 데이터를 다른 방식으로 주무를 필요가
> 생기면 segment를 하나 더 돌려 끼우는 식으로.)

Ken Thompson이 1973년 1월 Unix에서 이것을 구현했습니다. Garlan & Shaw가 이것을 architectural style로
이름 붙이고 분석했습니다 — *An Introduction to Software Architecture* (CMU-CS-94-166, 1994년 1월;
또한 *Advances in Software Engineering and Knowledge Engineering* Vol. I, World Scientific, 1993).
POSA Vol. 1 (Buschmann et al., Wiley, 1996)은 이것을 pattern으로 정리했습니다:
*"provides a structure for systems that process a stream of data. Each processing step is
encapsulated in a filter component. Data are passed through pipes between adjacent filters."*
(데이터 stream을 처리하는 시스템의 구조를 제공한다. 각 processing step은 filter component로
encapsulate된다. 데이터는 인접한 filter들 사이의 pipe를 통해 전달된다.)

"Screw in another segment when it becomes necessary to massage data in another way"는 **곧**
"중간에 process를 하나 추가한다"입니다. 62년 간격을 두고 같은 문장입니다. 당신은 Pipecat에게 없는
feature를 요구하고 있는 게 아니라, Pipecat이 이미 구현하고 있는 style의 **정의적 속성(defining
property)**을 요구하고 있는 겁니다. 여기 나오는 citation은 전부 [[theory-pipes-and-filters]]에서
나옵니다.

### 1.2 reuse claim을 정확하게 진술하면

Garlan & Shaw의 reuse claim은 조심해서 읽을 가치가 있습니다. 왜냐하면 engineering의 전부가 그
단서 조항(proviso)에 들어 있기 때문입니다.

> [Pipe-and-filter systems] support reuse: **any two filters can be hooked together, provided
> they agree on the data that is being transmitted between them.**
>
> ([Pipe-and-filter system은] reuse를 지원한다: **어떤 두 filter든 서로 연결될 수 있다 — 단, 둘
> 사이에 transmit되는 data에 대해 서로 합의한다면.**)

두 filter는 transmitted data에 대해 합의할 때에만(iff) compose됩니다. 이건 *conditional*(조건문)입니다.
대부분의 시스템은 이 조건을 만족하지 못합니다 — `dict[str, float]`를 반환하는 함수는 `AudioChunk`를
받는 함수와 compose되지 않습니다. Pipecat의 수는 그 antecedent(전건)를 **자명하게, 보편적으로 참**으로
만들어버리는 것입니다: 모든 processor에게 단 하나의 universal type 위에서 동일한 signature를 부여함으로써.
합의는 splice site에서 협상되는 것이 아니라, shape가 하나뿐이기 때문에 **구성상(by construction)**
보장됩니다.

> 💡 **쉬운 설명 — "antecedent를 참으로 만든다"가 무슨 뜻인가요?**
> "A이면 B이다"라는 정리에서, A를 매번 증명하는 대신 A가 항상 성립하도록 시스템을 설계해버리는 겁니다.
> Garlan & Shaw는 "data에 합의하면 연결 가능"이라고 했는데, Pipecat은 data type을 `Frame` 하나로
> 만들어서 "합의"를 검사할 필요조차 없게 했습니다. USB-C가 모든 케이블을 같은 모양으로 만들어
> "이 커넥터가 맞나?"라는 질문 자체를 없앤 것과 같습니다 — 단, 꽂힌다고 해서 그 기기가 원하는
> 신호를 준다는 뜻은 아니라는 점까지 똑같습니다(§8에서 다시 나옵니다).

### 1.3 두 개의 style invariant, 그리고 Pipecat의 준수 여부

Garlan & Shaw는 어떤 시스템이 pipe-and-filter *이기 위해* 만족해야 하는 두 개의 invariant도
진술합니다.

> Filters must be independent entities: in particular, they should not share state with other
> filters. Another important invariant is that filters do not know the identity of their
> upstream and downstream filters.
>
> (Filter는 독립적인 entity여야 한다: 특히, 다른 filter와 state를 share해서는 안 된다. 또 하나의
> 중요한 invariant는 filter가 자신의 upstream/downstream filter의 identity를 알지 못한다는 것이다.)

특히 invariant 2를 잘 붙들어 두십시오. Pipecat은 이것을 *구조적으로(structurally)* 강제합니다 —
processor는 원한다고 해도 이웃을 type으로 이름 부를 수 없습니다. 이웃은 오직 slot, 즉 `self._next` /
`self._prev`를 통해서만 도달 가능하고, 그 slot은 외부의 linker가 채워주기 때문입니다. 이것을 해내는
세 줄을 §3에서 보게 됩니다. invariant 1은 convention으로만 강제되며, §11이 그것이 물기 시작하는
지점입니다.

---

## 2. uniform interface는 정확히 두 개의 method다

### 2.1 하나 읽고, 하나 쓴다

`src/pipecat/processors/frame_processor.py:820`

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
```

`src/pipecat/processors/frame_processor.py:1004`

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

이것이 contract 전부입니다. read verb 하나, write verb 하나. 1,333줄짜리 `frame_processor.py`의
나머지 전부는 이 둘을 떠받치는 machinery입니다.

### 2.2 direction axis는 정확히 두 개의 값을 가진다

`src/pipecat/processors/frame_processor.py:60-69`

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

member 두 개. 세 개도 아니고, 열린 registry도 아닙니다. 이것이 classic style로부터의 Pipecat의 첫 번째
구조적 이탈(structural deviation)입니다: canonical pipe는 unidirectional인데, Pipecat은 return path를
추가해서 downstream stage가 자기보다 먼저 실행된 stage에게 신호를 보낼 수 있게 합니다. 그 두 번째
direction이 `push_error(...)`가 application까지 도달하게 만드는 것이고, assistant aggregator가 자기보다
*upstream*에 있는 LLM에게 다시 prompt를 던질 수 있게 만드는 것입니다. `UPSTREAM`을 장식이 아니라 style에
대한 **수리(repair)**로 읽으십시오 — Garlan & Shaw는 interactive application을 이 style의 약한 사례로
명시적으로 지목했고, voice agent가 바로 그 사례입니다.

> 💡 **쉬운 설명 — 왜 return path가 "수리"인가요?**
> 고전적인 Unix pipe는 `a | b | c`처럼 한 방향입니다. `c`가 `a`에게 "그거 다시 보내줘"라고 말할 방법이
> 없습니다. 그런데 voice agent에서는 assistant aggregator(맨 끝)가 "사용자가 말을 끊었으니 LLM을 다시
> 돌려야 한다"고 말해야 합니다. LLM은 pipeline상 앞에 있습니다. `UPSTREAM`은 그 역류를 위한
> 최소한의 추가이고, 값이 정확히 2개인 이유는 그 이상이 필요 없기 때문입니다 — 방향은 축(axis) 하나에
> 대한 부호(sign)일 뿐입니다.

### 2.3 transmitted data는 하나의 union type이다

`src/pipecat/frames/frames.py:64-65` 그리고 `:104-138`

```python
@dataclass
class Frame:
    """Base frame class for all frames in the Pipecat pipeline.
    ...
    """

    id: int = field(init=False)
    name: str = field(init=False)
    pts: int | None = field(init=False)
    broadcast_sibling_id: int | None = field(init=False)
    metadata: dict[str, Any] = field(init=False)
    transport_source: str | None = field(init=False)
    transport_destination: str | None = field(init=False)
```

```python
@dataclass
class SystemFrame(Frame):
    """System frame class for immediate processing.

    A frame that takes higher priority than other frames. System frames are
    handled in order and are not affected by user interruptions.
    """


@dataclass
class DataFrame(Frame):
    """Data frame class for processing data in order. ..."""


@dataclass
class ControlFrame(Frame):
    """Control frame class for processing control information in order. ..."""
```

`Frame`은 `:65`에, 그리고 세 개의 branch: `SystemFrame`은 `:105`, `DataFrame`은 `:116`,
`ControlFrame`은 `:128`. 그 3-way split이 [[ch-02/read]]의 주제 전체 — the narrow waist — 이고,
동시에 barge-in을 feature가 아니라 산술(arithmetic)로 만들어 주는 것이기도 합니다([[ch-04/read]]의
priority tier). ch-01에서는 결론 하나만 필요합니다: **transmitted data가 단일한 open union이므로,
"transmitted data에 대한 합의"는 보편적으로 성립한다.**

### 2.4 실제로 이 shape를 presenting하는 것은 몇 개인가 — 그리고 excerpt가 틀린 지점

[[theory-pipes-and-filters]]는 이렇게 말합니다: *"131 `async def process_frame` overrides exist across
117 files in `src/` — and every one of them presents this same shape."* 앞 절반은 맞고, 뒤 절반은
틀렸습니다. 그리고 그 오류는 중요합니다 — 이 style의 주장 *전체*가 shape uniformity에 얹혀 있기
때문입니다.

이 commit에서 측정한 결과:

| Measurement | Command | Value |
|---|---|---|
| `async def process_frame` definitions in `src/` | `grep -rn "async def process_frame" src/ \| wc -l` | **131** |
| files containing one | `grep -rln ... \| wc -l` | **117** |
| definitions with the pipeline signature `(self, frame: Frame, direction: FrameDirection)` | `grep -rn "...` | **101** |
| files containing one of those | | **87** |
| definitions with a *different* signature | | **30** |
| files containing one of those | | **30** |

87 + 30 = 117이고, 두 종류를 동시에 담은 파일은 하나도 없습니다. 따라서 131개의 definition 중
**101개가 pipeline interface**(override 100개 + `frame_processor.py:820`의 base implementation)이고,
**30개는 단지 같은 method 이름을 우연히 쓰고 있을 뿐인, 서로 무관한 7개의 class hierarchy에
속합니다**:

| Hierarchy | Base class | Signature | Anchor |
|---|---|---|---|
| audio input filters | `BaseAudioFilter(ABC)` | `process_frame(self, frame: FilterControlFrame)` — no `direction` | `audio/filters/base_audio_filter.py:18`, method `:50` |
| audio mixers | `BaseAudioMixer(ABC)` | `process_frame(self, frame: MixerControlFrame)` | `audio/mixers/base_audio_mixer.py:18`, method `:51` |
| VAD control | `VADController(BaseObject)` | `process_frame(self, frame: Frame)` | `audio/vad/vad_controller.py:31`, method `:122` |
| turn-start strategies | `BaseUserTurnStartStrategy(BaseObject)` | `-> ProcessFrameResult \| None` — **returns a value** | `turns/user_start/base_user_turn_start_strategy.py:39`, method `:164` |
| turn-stop strategies | `BaseUserTurnStopStrategy(BaseObject)` | `-> ProcessFrameResult \| None` | `turns/user_stop/base_user_turn_stop_strategy.py:38`, method `:171` |
| user-mute strategies | `BaseUserMuteStrategy(BaseObject)` | `-> bool` | `turns/user_mute/base_user_mute_strategy.py:14`, method `:46` |
| context summarizer | `LLMContextSummarizer(BaseObject)` | `process_frame(self, frame: Frame)` | `processors/aggregators/llm_context_summarizer.py:57`, method `:144` |

이들 중 어느 것도 `Pipeline`에 link되지 않습니다. `BaseUserTurnStartStrategy.process_frame`은 심지어
값을 *return*하는데, 이건 pipe로서는 구조적으로 불가능합니다 — filter는 return이 아니라 push로
씁니다. excerpt 자신의 guideline은 "three unrelated classes"라고 말하지만, 실제로는 7개 hierarchy,
30개 definition입니다.

**왜 이걸 각주가 아니라 두 문단으로 다루는가.** `grep -rn "process_frame"`은 Lina를 포팅하기 시작할 때
당신이 가장 먼저 할 일이고, hit의 약 1/4은 당신이 찾는 그 interface가 아닙니다. 더 나쁜 건, 그 7개
hierarchy 중 셋(`turns/*`)이 [[ch-06/read]]에서 turn-boundary strategy chain을 다룰 때 가장 열심히 읽게
될 바로 그 코드라는 점입니다 — 즉 collision이 하필 당신이 가장 감당하기 어려운 지점에서 일어납니다.
uniform-interface 주장 자체는 살아남습니다(한 shape의 definition 101개면 여전히 uniform interface입니다).
하지만 "uniform in name"과 "uniform in type"은 다른 property이고, 이 repo는 grep을 필터링해야만
두 번째 property를 갖습니다.

> 💡 **쉬운 설명 — "uniform in name" vs "uniform in type"**
> 같은 이름의 method가 131개 있다고 해서 그것들이 같은 interface인 건 아닙니다. Python은 duck typing
> 언어라 `process_frame`이라는 이름은 아무 class나 쓸 수 있습니다. `link()`에 꽂을 수 있는 건
> `(self, frame: Frame, direction: FrameDirection)` signature를 가진 101개뿐이고, 나머지 30개는
> 그냥 이름이 겹치는 남남입니다. 실무 규칙: grep할 때 이름이 아니라 **signature 전체**로 검색하세요.

---

## 3. `link()`가 splicing 스토리의 전부이고, 그것은 아무것도 validate하지 않는다

lego-block composition을 참으로 만드는 함수가 여기 있습니다.

`src/pipecat/processors/frame_processor.py:671-679`

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

세 statement: `:677`의 pointer 대입, `:678`의 back-pointer 대입, `:679`의 debug log. type check가
없습니다. capability negotiation도 없습니다. "내가 emit하는 걸 네가 받을 수 있냐"는 handshake도
없습니다. ordering assertion도 없습니다. Pipecat에서 "pipeline"은 doubly-linked list이고, `link`는
`insert`입니다.

`link`가 **저장하지 않는 것**에 주목하십시오: 두 processor 중 어느 쪽이 무엇을 consume하고 무엇을
produce하는지에 대한 정보가 전혀 없습니다. 그것이 invariant 2가 코드로 강제된 모습입니다 — link 이후,
두 processor는 서로를 class로 이름 부를 수 없고 오직 slot을 통해서만 도달할 수 있습니다.

write path가 이를 확인해 줍니다. `push_frame`은 `__internal_push_frame`에 위임합니다:

`src/pipecat/processors/frame_processor.py:1160-1194` (일부 생략)

```python
    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        ...
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")
                ...
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                ...
                await self._prev.queue_frame(frame, direction)
```

`:1182`의 `await self._next.queue_frame(frame, direction)`, `:1194`의
`await self._prev.queue_frame(...)`. **push한다는 것은 이웃에게 enqueue하는 것이고, 결코 직접
호출(direct call)이 아닙니다.** processor는 slot을 이름 부르고, 그 slot은 마지막으로 `link`를 실행한
쪽이 채웁니다. 이것이 또한 "왜 아무것도 안 나왔지"가 여기서는 call stack에서와는 다른 종류의 debugging
문제인 이유입니다 — §11에서 다시 다룹니다.

> **지금, §4로 넘어가기 전에 [`figures/splice-algebra.html`](figures/splice-algebra.html)을 여십시오.**
> processor를 position N으로 drag해 넣고, fold가 실행되는 동안 `link()`가 정확히 이 세 statement로
> 전개되는 것을 보십시오. 이 interaction의 요점은 — 당신이 그것을 깨뜨리려고 시도해 봄으로써 —
> 당신의 배치를 reject할 수 있었을 네 번째 statement 같은 것은 어디에도 없다는 사실을 스스로
> 납득하는 것입니다.

---

## 4. `Pipeline`은 `link`를 operator로 쓰는 fold다

`src/pipecat/pipeline/pipeline.py:99-121`

```python
    def __init__(
        self,
        processors: Sequence[FrameProcessor],
        *,
        source: FrameProcessor | None = None,
        sink: FrameProcessor | None = None,
    ):
        """Initialize the pipeline with a list of processors.

        Args:
            processors: Sequence of frame processors to connect in sequence.
            source: An optional pipeline source processor.
            sink: An optional pipeline sink processor.
        """
        super().__init__(enable_direct_mode=True)

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = source or PipelineSource(self.push_frame, name=f"{self}::Source")
        self._sink = sink or PipelineSink(self.push_frame, name=f"{self}::Sink")
        self._processors: list[FrameProcessor] = [self._source, *processors, self._sink]

        self._link_processors()
```

`src/pipecat/pipeline/pipeline.py:197-202`

```python
    def _link_processors(self):
        """Link all processors in sequence and set their parent."""
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.link(curr)
            prev = curr
```

`:197`의 def 아래 body 네 줄(`:199-202`). 이것은 `link`를 operator로 쓰고 직전 원소 외에는 accumulator가
없는, list에 대한 left fold입니다. **`src/pipecat/pipeline/` 어디에도 type check, capability
negotiation, ordering assertion이 없습니다.** 저는 찾아봤습니다.
`grep -n "raise\|assert " pipeline.py base_pipeline.py`는 정확히 한 개의 hit을 반환하고, 그것은 teardown이
best-effort라는 주석입니다. 이 package의 core composition path 전체에서 유일한 `raise`들은
`parallel_pipeline.py`에 있고, 그것들은 semantics가 아니라 *arity와 container type*을 검사합니다:

`src/pipecat/pipeline/parallel_pipeline.py:47-48, 60-61`

```python
        if len(args) == 0:
            raise Exception("ParallelPipeline needs at least one argument")
```

```python
                raise TypeError(f"ParallelPipeline argument {processors} is not a list")
```

"list인가?" — "branch 2가 merge가 기대하는 것을 emit하는가?"가 아닙니다.

### 4.1 믿고 쌓아 올리면 안 되는, 낡은 docstring 하나

`:198`을 다시 읽으십시오: *"Link all processors in sequence **and set their parent**."* body는 parent를
전혀 설정하지 않습니다. `FrameProcessor`에는 `parent`나 `_parent` attribute 자체가 없습니다 —
`grep -c "_parent\b" src/pipecat/processors/frame_processor.py`는 **0**을 반환합니다. codebase에 있는
유일한 `parent`는 `BaseWorker.parent`이고, 이것은 `src/pipecat/workers/base_worker.py:271`의 property로
[[ch-04/read]]의 worker topology에 관한 것이지 processor에 관한 것이 아닙니다. 이 docstring은
잔재(leftover)입니다. 만약 processor에서 그것을 담고 있는 pipeline으로 거슬러 올라갈 계획이었다면 —
그럴 수 없습니다. 그리고 이것은 이 course에서 repo의 산문(prose)이 코드가 하지 않는 일을 기술하는
여러 지점 중 첫 번째입니다.

### 4.2 그래서 "position N에 splice"는 문자 그대로 `list.insert`다

§3과 §4를 합치면 lego claim은 기계적인 것이 됩니다:

```python
# before
Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])

# after — a rule processor spliced between STT and the user aggregator
Pipeline([transport.input(), stt, korean_honorific_rule, user_aggregator, llm, tts,
          transport.output(), assistant_aggregator])
```

argument는 `Sequence[FrameProcessor]`입니다. splice는 `list.insert(N, p)`입니다. 제거는 `list.pop(N)`
입니다. **모든 `p`에 대해, 모든 `N`에서, 언제나 type-check를 통과합니다.** 그것이 당신이 원했던
property를 정확하게 진술한 것입니다 — 그리고 정확하게 진술하고 나면, 이미 약간 불안해져야 정상입니다.
"항상 type-check를 통과한다"와 "항상 옳다"는 아주 멀리 떨어져 있으니까요. §8이 그 간극에 이름을
붙이는 곳입니다.

---

## 5. algebra, 그리고 실제로 ship되는 witness들

style은 집합(set)과 연산자(operator)를 줍니다. Pipecat의 집합은 모든 `FrameProcessor`이고, 연산자는
`link`입니다. 이것을 은유(metaphor) 이상으로 만드는 것은, 이 algebra의 distinguished element들이
diagram이 아니라 **`src/`에서 import할 수 있는 class**라는 사실입니다.

| Law | Witness in `src/` |
|---|---|
| `p ∘ Identity = Identity ∘ p = p` | `IdentityFilter` — `processors/filters/identity_filter.py:17` |
| `p ∘ Null ≈ Null` (data frames only) | `NullFilter` — `processors/filters/null_filter.py:18` |
| `(a ∘ b) ∘ c = a ∘ (b ∘ c)` | `Pipeline(BasePipeline(FrameProcessor))` — `pipeline/pipeline.py:91`, `pipeline/base_pipeline.py:19` |
| `a ∥ b` (fan-out, weak merge) | `ParallelPipeline` — `pipeline/parallel_pipeline.py:24` |
| `if cond then p else pass` | `FunctionFilter` — `processors/filters/function_filter.py:21` |
| higher-order: a pipeline parameterized by a strategy | `ServiceSwitcher` — `pipeline/service_switcher.py:247` |

이 operator는 **commutative가 아닙니다**. `stt ∘ llm ≠ llm ∘ stt`. type system에는 그 사실이 전혀
기록되지 않습니다. 이 생각을 §8까지 붙들고 계십시오. 그것이 "the price"의 내용 전부입니다.

### 5.1 Identity — 그리고 repo가 광고하지 않는 사실 하나

`src/pipecat/processors/filters/identity_filter.py:17-23, 37-45`

```python
class IdentityFilter(FrameProcessor):
    """A pass-through filter that forwards all frames without modification.

    This filter acts as a transparent passthrough, allowing all frames to flow
    through unchanged. It can be useful when testing `ParallelPipeline` to
    create pipelines that pass through frames (no frames should be repeated).
    """
```

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame by passing it through unchanged.

        Args:
            frame: The frame to process and forward.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
```

두 statement. 이것을 아무 position에나 끼워 넣는 것은 *증명 가능한* no-op이며, 동시에 "관심 없는
것은 그냥 통과시켜라(tolerate what you don't care about)"의 reference implementation입니다:
hardcode된 방향이 아니라 **frame이 도착한 그 방향으로** forward합니다.

이제 그 사실. `grep -rn "IdentityFilter" src tests`는 **76** hit을 반환합니다. `grep -rn
"IdentityFilter" src/`는 **1**을 반환합니다 — 자기 자신의 class definition. 나머지 75개 reference는
전부 `tests/`에 있습니다. Pipecat composition algebra의 identity element는 오직 test suite만
사용하고, 실제 ship되는 code path 중 그것과 compose하는 것은 없습니다. 이것은 비판이 아니라
"the algebra"가 무엇을 위해 존재하는지에 대한 데이터입니다: 그것은 production pipeline이 no-op stage를
필요로 해서가 아니라, composition을 *compositional하게 추론하고 test하기 위해* 존재합니다.

### 5.2 Zero — 그리고 그것이 왜 의도적으로 *near*-zero인가

`src/pipecat/processors/filters/null_filter.py:18-24, 38-48`

```python
class NullFilter(FrameProcessor):
    """A filter that blocks all frames except system and end frames.

    This processor acts as a null filter, preventing frames from passing
    through the pipeline while still allowing essential system and end
    frames to maintain proper pipeline operation.
    """
```

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, only allowing system and end frames through.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (SystemFrame, EndFrame)):
            await self.push_frame(frame, direction)
```

absorbing zero라면 *모든 것*을 삼켜야 합니다. 이것은 그럴 수 없습니다. `:47-48`이 `SystemFrame`과
`EndFrame`을 새게(leak) 합니다. 그리고 그래야만 합니다: `SystemFrame`이 없으면 `StartFrame`이
pipeline의 downstream 절반에 도달하지 못하고, 그러면 downstream의 아무것도 자신의 process task를
만들지 못하며(§7.2), `InterruptionFrame`이 전파되지 않아 barge-in이 그 filter에서 죽습니다.
`EndFrame`이 없으면 transport가 shutdown해야 한다는 것을 배우지 못해, session이 crash하는 대신
hang합니다.

**여기가 algebra가 현실과 만나는 지점이고, 이 chapter에서 가장 중요한 구조적 사실입니다.** Pipecat의
composition operator는 *data plane*에서는 annihilating(소멸적)이고 *control plane*에서는
transparent(투명)합니다. 당신은 Pipecat pipeline을 대수적으로 절단할 수 없습니다. `StartFrame`,
`EndFrame`, `CancelFrame`, `InterruptionFrame`은 구성상 모든 stock filter를 살아남습니다 — 그리고
같은 escape hatch가 바로 옆 파일에 그대로 나타납니다:

`src/pipecat/processors/filters/frame_filter.py:36-41`

```python
    def _should_passthrough_frame(self, frame):
        """Determine if a frame should pass through the filter."""
        if isinstance(frame, self._types):
            return True

        return isinstance(frame, (EndFrame, SystemFrame))
```

그래서 `FrameFilter(types=())`는 곧 `NullFilter()`이고, test들은 class 대신 정확히 그 표기를 씁니다:
`tests/test_filters.py:61`, `tests/test_pipeline.py:751`, `:1042`, `:1065`. 한편
`grep -rn "NullFilter" src tests examples`는 **1** hit을 반환합니다 — `null_filter.py:18`의 자기 자신의
definition. 아무도 필요해서 `NullFilter`를 만들지 않습니다. 그것은 **algebra를 닫기(close) 위해**
존재합니다. 그것은 의도적인 저작 행위(authorial act)이고, 저자들이 이런 용어로 사고하고 있었다는 것을
말해 줍니다.

> 💡 **쉬운 설명 — "algebra를 닫는다"는 게 뭔가요?**
> 덧셈에 0이 있고 곱셈에 1이 있듯, composition operator에도 identity element와 zero(absorbing) element가
> 있으면 "이 구조는 monoid다" 같은 말을 할 수 있고, 그 위에서 법칙을 이용해 추론할 수 있습니다.
> Pipecat 저자들은 production에서 아무도 쓰지 않는 `NullFilter`와 `IdentityFilter`를 굳이 만들어
> 뒀습니다. 쓰려고 만든 게 아니라 **구조가 완결되게 하려고** 만든 겁니다. 그리고 zero가 진짜 zero가
> 되지 못하고 `SystemFrame`/`EndFrame`을 흘리는 것 — 그게 "이 시스템은 순수한 수학이 아니라
> lifecycle을 가진 런타임"이라는 사실이 새어 나오는 지점입니다.

### 5.3 Associativity는 type 상의 사실이고, Pipecat은 자기 코드에서 그것에 의존한다

`src/pipecat/pipeline/base_pipeline.py:19-24`

```python
class BasePipeline(FrameProcessor):
    """Base class for all pipeline implementations."""

    def __init__(self, **kwargs):
        """Initialize the base pipeline."""
        super().__init__(**kwargs)
```

`src/pipecat/pipeline/pipeline.py:91-97`

```python
class Pipeline(BasePipeline):
    """Main pipeline implementation that connects frame processors in sequence.

    Creates a linear chain of frame processors with automatic source and sink
    processors for external frame handling. Manages processor lifecycle and
    provides metrics collection from contained processors.
    """
```

`Pipeline`은 `BasePipeline`이고 `BasePipeline`은 `FrameProcessor`입니다. **pipeline은 processor다.**
따라서 `Pipeline([a, Pipeline([b, c])])`와 `Pipeline([Pipeline([a, b]), c])`는 둘 다 legal이고, 둘 다
동일하게 동작합니다 — nesting은 `PipelineSource`(`pipeline.py:21`) / `PipelineSink`(`:55`) 쌍을 하나
끼워 넣을 뿐이고, 그 `process_frame`은 forward하는 two-arm direction switch입니다. grouping이
공짜입니다. 그것이 바로 associativity가 사주는 것입니다.

이것은 이론적 우아함이 아닙니다. Pipecat 자신의 runtime이 *당신의* composition 안으로, 당신이 쓰는
것과 같은 연산을 써서 splice해 들어옵니다:

`src/pipecat/pipeline/worker.py:537`

```python
            pipeline = Pipeline([edge_source, pipeline, edge_sink])
```

지역 변수 `pipeline`이, 자기 자신을 가운데 원소로 담은 새 `Pipeline`으로 rebind됩니다. 만약 `Pipeline`이
`FrameProcessor`가 아니었다면 이 줄은 valid typing으로 파싱조차 되지 않습니다. framework가 당신의
pipeline 중간에 process를 추가하는데, 당신이 호출한 것과 같은 constructor를 호출해서 그렇게 합니다.
"position N에 splice"가 편의 기능이 아니라 primitive라는 것에 대해 이보다 강한 증거는 없습니다.

그리고 이 envelope는 붕 떠 있는 overhead가 아닙니다 — `Pipeline`, `PipelineSource`, `PipelineSink`는
모두 `enable_direct_mode=True`로 생성되며(`pipeline.py:113`, `:36`, `:72`), `frame_processor.py:717-719`에
따라 그것은 internal input queue를 건너뛰고 caller의 task에서 inline으로 처리한다는 뜻입니다. nesting은
queue hop 비용이 0입니다. [[parallel-pipeline]]과 대조하십시오: `ParallelPipeline`은 의도적으로 direct
mode를 쓰지 *않고*, `parallel_pipeline.py:43-44`의 주석에서 그렇게 말합니다 —
*"We don't set it to direct mode because we use frame pausing and that requires queues."*

> 💡 **쉬운 설명 — direct mode가 왜 associativity와 같이 나오나요?**
> 수학적으로 `(a∘b)∘c = a∘(b∘c)`여도, 구현이 그룹핑마다 queue를 하나씩 더 만들면 nesting은
> latency 비용을 발생시킵니다. voice agent에서 queue hop 하나는 진짜 milliseconds입니다.
> Pipecat은 `enable_direct_mode=True`로 nesting의 런타임 비용까지 0으로 만들어서, associativity가
> "타입 상으로만 참"이 아니라 "실무에서도 공짜"가 되게 했습니다. `ParallelPipeline`만 예외인데,
> frame pausing 때문에 queue가 반드시 필요해서입니다.

### 5.4 Parallel은 진짜 combinator이고, 진짜 주의사항이 있다

`src/pipecat/pipeline/parallel_pipeline.py:24, 33-50`

```python
class ParallelPipeline(BasePipeline):
    """Pipeline that processes frames through multiple sub-pipelines concurrently.
```

```python
    def __init__(self, *args):
        """Initialize the parallel pipeline with processor lists.

        Args:
            *args: Variable number of processor lists, each becoming a parallel branch.

        Raises:
            Exception: If no processor lists are provided.
            TypeError: If any argument is not a list of processors.
        """
        # We don't set it to direct mode because we use frame pausing and that
        # requires queues.
        super().__init__()

        if len(args) == 0:
            raise Exception("ParallelPipeline needs at least one argument")
```

각 positional arg는 `list`이고, 그것이 escape hatch가 다시 배선된 하나의 진짜 `Pipeline`이 됩니다.
ch-01에서 기록해 둘 것은, `∥`가 깔끔한 product type이 *아니라*는 점입니다: fan-out은 **동일한 frame
object**를 모든 branch에 전달하고(복사하지 않음), merge는 `frame.id` 기준 first-arrival dedup이며
ordering은 정확히 세 개의 frame type에 대해서만 synchronize됩니다. 자세한 내용은
[[parallel-pipeline]]에 있고 [[ch-04/read]]에서 필요합니다. 이것을 기본 composition tool로 손이 먼저
가게 두지 마십시오.

### 5.5 conditional

`src/pipecat/processors/filters/function_filter.py:57-71, 73-85`

```python
    def _should_passthrough_frame(self, frame, direction):
        """Check if a frame should pass through without filtering."""
        # Always passthrough frames in the wrong direction
        if self._direction and direction != self._direction:
            return True

        # Always passthrough lifecycle frames
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            return True

        # If not filtering system frames, passthrough all other system frames
        if not self._filter_system_frames and isinstance(frame, SystemFrame):
            return True

        return False
```

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame through the filter.

        Args:
            frame: The frame to process.
            direction: The direction the frame is moving in the pipeline.
        """
        await super().process_frame(frame, direction)

        passthrough = self._should_passthrough_frame(frame, direction)
        allowed = await self._filter(frame)
        if passthrough or allowed:
            await self.push_frame(frame, direction)
```

주목할 것이 세 가지이고, 전부 `:82-84`에 있습니다.

1. control-plane escape hatch가 *세 번째로* 등장하는데, 이번에는 `CancelFrame`과 `StartFrame`이 명시적으로
   이름 붙어 있습니다(`:64`). §5.2와 같은 design commitment입니다.
2. `:83`의 `await self._filter(frame)`은 **무조건 실행됩니다** — `passthrough`가 이미 `True`여서 그 답이
   버려질 상황에서도 실행됩니다. 따라서 side-effect가 있는 predicate는 자기가 절대 gate하지 않을
   모든 `StartFrame`, `EndFrame`, `CancelFrame`, `SystemFrame`을 보게 됩니다. Lina의 rule을
   `FunctionFilter` predicate로 포팅했는데 그 rule이 state를 mutate한다면, 당신이 의도하지 않은
   frame에서도 발화(fire)합니다. predicate는 pure하게 만드십시오.
3. `direction`의 default는 `DOWNSTREAM`이므로, `direction=None`으로 설정하지 않는 한 upstream frame은
   필터링 없이 통과합니다.

> 💡 **쉬운 설명 — 2번 함정을 코드로**
> ```python
> count = 0
> async def rule(frame):            # 나쁜 predicate: side effect가 있음
>     global count
>     count += 1                    # StartFrame/EndFrame에서도 증가한다
>     return isinstance(frame, TextFrame)
> ```
> `passthrough`가 True인 lifecycle frame에서도 `rule`이 호출되므로 `count`는 당신이 세려던 것보다
> 큽니다. "이 predicate의 반환값이 쓰이지 않는다"와 "이 predicate가 호출되지 않는다"는 다릅니다.

### 5.6 higher-order composition: strategy로 parameterize된 pipeline

이것이 "list of steps"에서 가장 멀리 나아간 witness이고, Lina를 위해 무엇을 설계하기 전에 공부해 둘
가치가 가장 큰 것입니다.

`src/pipecat/pipeline/service_switcher.py:247`

```python
class ServiceSwitcher(ParallelPipeline, Generic[StrategyType]):
    """Parallel pipeline that routes frames to one active service at a time.
```

`src/pipecat/pipeline/service_switcher.py:267-279`

```python
    def __init__(
        self,
        services: list[FrameProcessor],
        strategy_type: type[StrategyType] = ServiceSwitcherStrategyManual,
    ):
        """Initialize the service switcher with a list of services and a switching strategy.

        Args:
            services: List of frame processors to switch between.
            strategy_type: The strategy class to use for switching between services.
                Defaults to ``ServiceSwitcherStrategyManual``.
        """
        _strategy = strategy_type(services)
        super().__init__(*self._make_pipeline_definitions(services, _strategy))
```

`src/pipecat/pipeline/service_switcher.py:369-397`

```python
    @staticmethod
    def _make_pipeline_definition(
        service: FrameProcessor, strategy: ServiceSwitcherStrategy
    ) -> Any:
        async def filter(_: Frame) -> bool:
            return service == strategy.active_service

        # Layout: Filter → Service → Filter
        #
        # filter_system_frames: we want to run filter functions also on system
        # frames.
        #
        # enable_direct_mode: filter functions are quick so we don't need
        # additional tasks.
        return [
            FunctionFilter(
                filter=filter,
                direction=FrameDirection.DOWNSTREAM,
                filter_system_frames=True,
                enable_direct_mode=True,
            ),
            service,
            FunctionFilter(
                filter=filter,
                direction=FrameDirection.UPSTREAM,
                filter_system_frames=True,
                enable_direct_mode=True,
            ),
        ]
```

이게 무엇을 하고 있는지 읽으십시오. `ServiceSwitcher`는 *processor의 list*와 *strategy class*를 받아서,
각 branch가 `[FunctionFilter, service, FunctionFilter]`인 `ParallelPipeline`을 **생성(generate)**합니다 —
그 filter들은 frame 시점에 strategy object를 조회하는 predicate를 close over합니다. composition이
작성되는(written) 것이 아니라 계산되고(computed) 있습니다. ship되는 세 개의 strategy는
`ServiceSwitcherStrategy`(`:31`), `ServiceSwitcherStrategyManual`(`:148`),
`ServiceSwitcherStrategyFailover`(`:180`)입니다.

**Framework-extension move — 그리고 이건 당신이 실제로 챙겨 갔으면 하는 것입니다.** Lina 통화에는 당신이
이미 지고 있는 provider risk가 있습니다: 한국어 STT나 TTS 벤더가 통화 중간에 degrade되는데 in-band
failover가 없다는 것. `ServiceSwitcherStrategyFailover`가 정확히 그 shape이고, 이미 만들어져 있으며,
`FrameProcessor`에 대해 generic합니다 — 즉 member가 STT service인지 아닌지 신경 쓰지 않습니다.
pipeline에서 같은 position을 차지할 수 있는 어떤 두 processor든 switcher member가 될 수 있습니다.
그래서: 한국어 TTS 벤더 둘을 switcher 하나 뒤에 두거나; deal stage를 읽는 strategy로 전환되는
"cheap model / careful model" LLM service 쌍; 또는 — 가장 흥미로운 것 — 고객이 의무 고지(mandatory
disclosure)를 이미 들었는지 여부에 따라 permissive rule processor와 strict rule processor 중 하나를
`active_service`로 고르는 `InsuranceComplianceStrategy`. routing code를 한 줄도 쓰지 않고 per-frame
routing을 얻습니다. routing이 filter 쌍으로 compile된 predicate이기 때문입니다. 천장(ceiling)도
기억하십시오: switcher는 한 번에 **하나**의 member만 활성화합니다. 그것은 selector이지 arbiter가
아니고, [[ch-12/read]]가 그 둘의 차이가 아프기 시작하는 곳입니다.

> 💡 **쉬운 설명 — selector와 arbiter의 차이**
> selector: N개 후보 중 **하나를 고른다**. (지금 활성 TTS는 A인가 B인가?)
> arbiter: N개 후보가 **동시에 각자의 주장을 내놓고**, rank/priority로 충돌을 해소한다.
> (rule layer 3개가 각각 "인사해라", "가격 언급해라", "고지 먼저 해라"를 주장하면 어느 게 이기나?)
> Lina의 `ACTION_PRIORITY` 테이블은 arbiter입니다. `ServiceSwitcher`로는 그걸 만들 수 없습니다.

---

## 6. substitutability가 모든 processor를 unit-testable하게 만드는 이유다

`src/pipecat/tests/utils.py:169-190`

```python
    received_up = asyncio.Queue()
    received_down = asyncio.Queue()
    source = QueuedFrameProcessor(
        queue=received_up,
        queue_direction=FrameDirection.UPSTREAM,
        ignore_start=ignore_start,
    )
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=ignore_start,
    )

    pipeline = Pipeline([source, processor, sink])

    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout=False,
        enable_rtvi=enable_rtvi,
        observers=observers,
        params=pipeline_params,
    )
```

`:182`의 `Pipeline([source, processor, sink])`. 그것이 framework 전체의 test harness 전부입니다.
`source`와 `sink`는 `asyncio.Queue`로 흘려보내는 `QueuedFrameProcessor`이고, *어떤* processor든 가운데
slot에 떨어집니다: LLM service, filter, transport output, nested pipeline, 오늘 아침에 당신이 쓴 무언가.

60개 provider를 가진 framework의 모든 component에 대해 generic한 test rig가 가능한 이유는 오직
**position N에 type이 없기** 때문입니다. 여기서 testability는 별도의 design 노력이 아니라 uniform
interface의 downstream 결과입니다. harness가 무엇을 증명하고 무엇을 증명하지 않는지도 함께
기억하십시오: `run_test()`는 *frame plumbing*을 assert합니다 — 어떤 frame이, 어떤 방향으로 나왔는가.
그 배치가 의미적으로(semantically) 옳았는지는 assert할 수 없고, 그것이 §8의 문제입니다.
`AGENTS.md:214-223`도 같은 말을 하며 behaviour는 `pipecat.evals`를 보라고 가리킵니다.

---

## 7. 대가, 1부: transparency tax와 반드시 있어야 하는 한 줄

### 7.1 모든 processor는 자기가 관심 없는 모든 frame을 견뎌야 한다

자기 frame만 이해하는 filter는 substitutable하지 않습니다 — 그것은 정확히 한 position에서만 작동하는
filter이고, 그러면 이 style 전체가 무너집니다. 그래서 uniformity는 *모든* 저자에게 세금을 매기고,
실제로 작고 현실적인 processor에서 그 세금을 내는 모습은 이렇습니다:

`src/pipecat/processors/aggregators/sentence.py:40-63`

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and aggregate text into complete sentences.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # We ignore interim description at this point.
        if isinstance(frame, InterimTranscriptionFrame):
            return

        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
            if match_endofsentence(self._aggregation):
                await self.push_frame(TextFrame(self._aggregation))
                self._aggregation = ""
        elif isinstance(frame, EndFrame):
            if self._aggregation:
                await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)
```

`:62-63`의 `else`가 그 세금입니다: 이 processor가 아무 의견도 없는 모든 frame class에 대해, 도착한
방향 그대로 앞으로 밀어 보냅니다. 그 한 arm이 `SentenceAggregator`를 어떤 position에도 insert 가능하게
유지해 줍니다.

나머지를, 24줄 안에 얼마나 많은 것이 비자명할 수 있는지 보여주는 worked example로 읽으십시오:

- **`:50-51`은 passthrough가 아니라 swallow입니다.** `InterimTranscriptionFrame`은 맨 `return`을
  맞습니다. 이것을 interim transcript를 렌더링하는 무언가의 위쪽에 splice하면 interim transcript가
  사라지는데, error는 없습니다. 그것은 *의도적인* swallow이지만, splice site에서는 그것이 존재한다는
  사실을 아무것도 알려주지 않습니다.
- **`:56`과 `:60`은 `direction`을 버립니다.** `push_frame(TextFrame(...))`은 default인 `DOWNSTREAM`을
  씁니다. downstream text만 보는 aggregator에게는 괜찮지만, upstream frame도 다뤄야 하는 Lina rule
  processor에 복사해 넣을 pattern으로는 버그 생산기입니다.
- **`:56`과 `:60`은 또한 새로운 bare `TextFrame`을 만듭니다.** 입력이 `TranscriptionFrame`(`user_id`와
  `timestamp`를 실은 `TextFrame` subclass)이었다면, 출력은 평범한 `TextFrame`이고 그 metadata는
  사라집니다. subclass identity는 이 processor를 통과하며 살아남지 못합니다.
  [[processor-vocabulary]]는 `StatelessTextTransformer`(`text_transformer.py:48`)에서 동일한 위험을
  지적하며 그것을 재사용하지 말고 custom processor를 쓰라고 말하는데, 그 조언은 여기에도
  적용됩니다.

### 7.2 반드시 있어야 하는 첫 줄, 그리고 그것이 정확히 무엇을 사주는가

그 override들은 전부 똑같이 시작합니다:

```python
await super().process_frame(frame, direction)     # not optional
```

이유는 이렇습니다. base implementation은 bookkeeping이 아니라 — **그 processor의 lifecycle입니다.**

`src/pipecat/processors/frame_processor.py:820-847`

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

observer notification, 그리고 다섯 개의 lifecycle case, `:847`의 `await self.__resume(frame)`에서 끝.
여기 *없는* 것에 주목하십시오: frame을 push하지 않습니다. `super()` 호출은 결코 아무것도 forward하지
않습니다. forwarding은 언제나 당신의 몫입니다.

이제 `StartFrame`을 따라가 봅시다:

`src/pipecat/processors/frame_processor.py:1091-1097`

```python
    async def __start(self, frame: StartFrame):
        """Handle the start frame to initialize processor state.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self.__create_process_task()
```

`__start`의 body 전체는 `:1097`의 호출 한 줄입니다.

`src/pipecat/processors/frame_processor.py:1222-1229`

```python
    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if self._enable_direct_mode:
            return

        if not self.__process_frame_task:
            self.__reset_process_task()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())
```

### 7.3 저자가 그 줄을 잊으면 실제로 무슨 일이 일어나는가 — 단정이 아니라 추적으로

저는 excerpt를 반복하는 대신 이것을 직접 추적했습니다. 이 failure mode가 새 Pipecat 저자가 저지를 수
있는 가장 비싼 실수 하나이고, 그 메커니즘은 정확하게 알아둘 가치가 있기 때문입니다.

각 processor는 **두 개**의 queue 위에서 **두 개**의 task를 돌립니다 — [[frame-processor]]가
"the physics of barge-in"이라고 부르는 구조이고, [[ch-04/read]]와 [[ch-08/read]]가 제대로 분해합니다.
여기서는 버그 하나를 따라갈 만큼만 필요합니다. *input* task는 `process_frame`이 아니라 `queue_frame`이
만듭니다:

`src/pipecat/processors/frame_processor.py:713-728`

```python
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        if self._enable_direct_mode:
            await self.__process_frame(frame, direction, callback)
            return

        await self.__input_queue.put((frame, direction, callback))

        # Nothing drains the queue until the StartFrame arrives, so a processor
        # never acts on a frame before it has been started. Frames pushed
        # between setup and the StartFrame simply wait, and the StartFrame is
        # dequeued ahead of them.
        if isinstance(frame, StartFrame):
            self.__create_input_task()
```

그러니까 *input* task는 당신이 `super()`를 호출했는지와 무관하게 시작됩니다. 그 다음 input task가
system frame을 나머지와 분리합니다:

`src/pipecat/processors/frame_processor.py:1287-1313`

```python
    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queue for another task
        to execute.

        """
        while True:
            (frame, direction, callback) = await self.__input_queue.get()

            if self.__should_block_system_frames and self.__input_event:
                logger.trace(f"{self}: system frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_system_frames = False
                logger.trace(f"{self}: system frame processing resumed")

            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
            else:
                raise RuntimeError(
                    f"{self}: __process_queue is None when processing frame {frame.name}"
                )

            self.__input_queue.task_done()
```

이제 연쇄가 정확해집니다:

1. 당신의 override가 `await super().process_frame(...)`을 빠뜨립니다.
2. `StartFrame`이 도착 → `queue_frame`이 **input** task를 만듭니다(`:728`) → input task가 그것을
   dequeue → 그것은 `SystemFrame` → `__process_frame` → 당신의 override가 실행 → 그리고
   `__start()`에는 **결코 도달하지 않습니다**.
3. 따라서 `__create_process_task()`가 호출되지 않고, `self.__process_frame_task`는 `None`으로 남아
   `__process_frame_task_handler`(`:1315`)는 결코 실행되지 않습니다.
4. 이후의 모든 `DataFrame`과 `ControlFrame`은 여전히 `:1307`에서 `self.__process_queue`에 들어갑니다.
   그 queue는 `__init__`의 `:284`에서 생성되고, 절대 `None`으로 대입되지 않으며,
   `FrameQueue(asyncio.Queue)`는 `__len__`도 `__bool__`도 정의하지 않기 때문입니다 — 그래서
   `elif self.__process_queue:`는 항상 truthy이고 `:1309-1311`의 `RuntimeError`는 도달 불가능합니다.
5. 결과: **processor가 frame을 영원히 받아들이면서 아무것도 emit하지 않습니다.** pipeline 한가운데의
   silent black hole. exception도, warning도, log 한 줄도 없습니다.

특히 4단계를 눈여겨보십시오. 방어적인 `raise`가 바로 거기 앉아 있는데, 그것은 발화할 수 없습니다.
만약 발화할 수 있었다면, `super()`를 잊었을 때 processor 이름과 frame 이름을 담은 요란한 crash가
났을 겁니다. 대신 그 guard는 결코 발생하지 않는 조건을 막고 있고, 실제로 발생하는 조건은 조용합니다.
이것은 Pipecat의 bug는 아니지만, framework의 진단(diagnostics)이 어디를 겨누고 어디를 겨누지 않는지에
대한 좋은 예시입니다.

> 💡 **쉬운 설명 — `elif self.__process_queue:`가 왜 항상 True인가요?**
> Python에서 `if obj:`는 `obj.__bool__()`을 부르고, 없으면 `len(obj) != 0`을 시도하고, 그것도 없으면
> "객체가 존재하므로 True"로 판정합니다. `asyncio.Queue`는 `__bool__`도 `__len__`도 정의하지 않으므로,
> **빈 queue든 아니든 항상 truthy**입니다. 저자는 아마 "queue에 뭔가 있으면"이 아니라 "queue가 None이
> 아니면"을 의도했을 텐데, 그 의도대로 작동하긴 합니다 — 다만 `__process_queue`는 절대 None이 되지
> 않으므로 `else` 가지는 죽은 코드입니다. 교훈: `if not None`을 의도했다면 `is not None`이라고 쓰세요.

같은 뿌리에서 나오는 두 개의 형제 failure, 둘 다 silent:

- super 호출 *위쪽*의 이른 `return`, 또는 re-push 위쪽의 이른 `return`은 downstream 전체를 굶깁니다.
  `SentenceAggregator:50-51`은 그것의 통제된(controlled) 사례이고, 사고로 생긴 것은 겉보기에
  똑같습니다.
- `EndFrame`은 `class EndFrame(ControlFrame, UninterruptibleFrame)`(`frames/frames.py:1899`)이므로
  system queue가 아니라 *data* queue를 탑니다. 이것을 삼키면 transport에 도달하지 못하고, shutdown이
  crash 대신 hang합니다. 이것이 바로 `NullFilter`가 그것을 re-push하는 이유입니다(§5.2) — zero
  element조차 진짜 zero가 될 여유가 없습니다.

### 7.4 이 contract는 문서화되어 있지 않고, 100개의 call site가 강제한다

`grep -rn "super().process_frame" --include="*.md" .`를 repository 전체에 돌리면 **0** hit입니다.
`AGENTS.md`에도, `CONTRIBUTING.md`에도, `README.md`에도 없습니다. `AGENTS.md`는 Observers bullet에서만
`process_frame`을 언급합니다. ([[pipecat-design-philosophy]]는 처음에 이 rule이 `AGENTS.md`에 명시되어
있다고 기록했지만, [[theory-pipes-and-filters]]가 그것을 정정했고, grep이 그 정정을 확인해 줍니다.)

그러면 무엇이 강제하는가? 오직 pattern뿐입니다. 저는 grep이 아니라 AST pass로 확인했습니다. 여기서는
grep이 over-count하기 때문입니다:

```
overrides of the pipeline signature (excluding the base): 100
  reaching the base via `await super().process_frame(...)`:  99
  reaching the base via an explicit unbound call:             1
  not reaching the base at all:                               0
```

86개 파일에 걸쳐서입니다. 준수율은 100%이고 — 그 유일한 예외는 실수가 아니라 오히려 교훈적입니다:

`src/pipecat/extensions/voicemail/voicemail_detector.py:163-185`

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state based on notifier signals.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await FrameProcessor.process_frame(self, frame, direction)

        # Gate logic: open gate allows all frames, closed gate filters frames
        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            # Only allow speaking frames if conversation was NOT detected (i.e., voicemail case)
            # This prevents the UserContextAggregator from issuing a warning about no aggregation
            # to push.
            if not self._conversation_detected:
                await self.push_frame(frame, direction)
        elif isinstance(frame, (SystemFrame, EndFrame, StopFrame)):
            # Always allow system frames through
            # This includes the UserStartedSpeakingFrame and UserStoppedSpeakingFrame
            # which are used to detect voicemail timing.
            await self.push_frame(frame, direction)
```

`:121`의 `ClassifierGate(NotifierGate)`. 그 부모인 `NotifierGate.process_frame`(`:91`)은 자체 push
logic을 갖고 있고 `ClassifierGate`는 그것을 실행하면 *안 됩니다* — 그래서 `:170`은 부모를 건너뛰고
조부모에게 explicit unbound call로 도달합니다:
`await FrameProcessor.process_frame(self, frame, direction)`. tree 전체에서 유일한 그런 호출입니다.

**rule을 올바르게 다시 진술하십시오. 이 예시가 통상적인 표현이 틀렸음을 증명하기 때문입니다:**
contract는 "`super()`를 호출하라"가 아닙니다. **"`FrameProcessor.process_frame`이, 정확히 한 번,
다른 어떤 것보다 먼저 실행되어야 한다"**입니다. 이미 `process_frame`을 override한 processor를
subclass할 때, `super()`는 부모의 behaviour까지 가져오는데 — 그것이 당신이 원하지 *않는* 바로 그것일
수 있습니다.

또한 `grep`이 무엇이라고 말했을지도 기억하십시오: `grep -rn "await super().process_frame" src/ | wc -l`은
**107**을 반환하지만, 그 107개 중 7개는 §2.4의 `turns/user_start`, `turns/user_stop`,
`turns/user_mute` strategy hierarchy에 속합니다 — 다른 base class, 다른 method, 우연히 동일한 문자열.
pipeline-interface count는 100입니다. [[theory-pipes-and-filters]]는 enforcement count를 107로
보고하지만, 실제 enforcement count는 100입니다.

---

## 8. 대가, 2부: "어떤 processor든 아무 데나"는 type-level 진실이자 semantic 거짓말이다

repo에 ship되는 canonical하고 실제로 동작하는 voice bot이 여기 있습니다:

`examples/getting-started/06-voice-agent.py:81-91`

```python
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

일곱 개의 원소이고, 그 순서는 최소 네 가지 독립적인 방식으로 load-bearing입니다: STT는 transcription을
consume하는 aggregator보다 앞서야 하고; aggregator는 그 context를 consume하는 LLM보다 앞서야 하며;
LLM은 그 text를 consume하는 TTS보다 앞서야 하고; `assistant_aggregator`는 `transport.output()`
**뒤에** 앉아서, assistant의 turn으로 기록되는 것이 LLM이 생성한 것이 아니라 실제로 wire로 나간
것이 되도록 합니다.

이제 대신 이렇게 써 보십시오:

```python
Pipeline([transport.input(), tts, llm, transport.output()])   # links, starts, runs, wrong
```

`_link_processors`는 이것을 기꺼이 받아들입니다. `PipelineWorker`가 그것을 시작합니다. 모든 processor가
setup됩니다. construction 시점에도, link 시점에도, start 시점에도, 통화 중 어느 시점에도 exception이
발생하지 않습니다. 그리고 이것은 그냥 틀렸습니다: `tts`는 `TextFrame`을 하나도 받지 못합니다. 그것을
produce하는 LLM이 그보다 *downstream*에 앉아 있기 때문입니다. **bot은 침묵하고, 그 침묵에는 아무런
error도 붙어 있지 않습니다.**

ordering constraint는 실재하고 전면적입니다 — STT before LLM before TTS — 그리고 그것은 코드 어디에도
살고 있지 않습니다. 그것은 오직 각 processor가 우연히 어떤 `isinstance` branch를 test하는지에, 주석
(`# LLM`, `# TTS`)에, 그리고 examples 디렉터리에 encode되어 있습니다. `AGENTS.md:60`은 `Pipeline`을
다섯 단어로 기술합니다 — *"Chains processors together"* — 그리고 순서에 대해서는 아무 말도 하지
않습니다. "processor는 block하면 안 된다"도, "turn을 넘어 state를 들고 있으면 안 된다"도, 문서화된
contract 어디에도 ordering rule이 없습니다.

**이것은 Pipecat의 결함이 아니라 style이 설계대로 작동하는 모습입니다.** interface에서 position을
지워버리는 것이 *바로* 어떤 두 filter든 연결 가능하게 만드는 그 수입니다. Garlan & Shaw의 세 번째
liability가 그 청구서를 직접 명시합니다:

> [Pipe-and-filter systems] may force a lowest common denominator on data transmission,
> resulting in added work for each filter to parse and unparse its data.
>
> ([Pipe-and-filter system은] data transmission에 최소공통분모를 강요할 수 있고, 그 결과 각 filter가
> 자기 data를 parse하고 unparse하는 추가 작업이 생긴다.)

Pipecat의 `Frame` union이 **바로** 그 lowest common denominator입니다. 100개의 `process_frame` override에
흩뿌려진 `isinstance` check가 **바로** 그 parse 작업입니다. 당신은 이 style을 채택함으로써 type
discipline을 제거한 것이 아닙니다. 그것을 compile time에서 runtime으로, 그리고 한 곳에서 백 곳으로
**옮겼을** 뿐입니다. [[ch-02/read]]는 그 거래가 값어치를 하는지, 그리고 그 union이 안쪽에서 보면 어떻게
생겼는지에 관한 chapter입니다.

> 💡 **쉬운 설명 — "compile time에서 runtime으로 옮겼다"**
> 정적 타입 시스템이라면 `tts(llm(stt(x)))`처럼 함수 합성으로 쓸 때 순서가 틀리면 컴파일이 실패합니다.
> Pipecat은 전부 `Frame → Frame`이라 컴파일러가 할 말이 없습니다. 대신 각 processor가 런타임에
> `isinstance(frame, TextFrame)`으로 "내가 아는 거냐"를 확인합니다. 검사는 사라진 게 아니라
> 100군데로 흩어졌고, 실패는 "타입 에러"가 아니라 "아무 일도 안 일어남"으로 나타납니다.

모든 splice 결정에 지고 가야 할 두 개의 corollary:

1. **splice *out*은 splice *in*과 정확히 똑같이 위험하고, 더 안전하게 느껴집니다.** processor를
   제거하는 것은 "downstream에서 이것이 produce했거나 보존했던 것을 여전히 필요로 하는 게 있는가"이고,
   컴파일러가 답할 수 없는 똑같은 질문을 거꾸로 묻는 것입니다. lego 직관은 제거가 아무것도 깨뜨릴 수
   없을 것처럼 느끼게 만듭니다. 명백히 깨뜨릴 수 있습니다.
2. **failure signature는 crash가 아니라 침묵입니다.** text 시스템에서는 잘못된 배치가 보통 throw합니다.
   voice 시스템에서는 잘못된 배치가 아무도 말하지 않는 통화를 만들어내고, 당신이 알아챌 때쯤이면
   고객은 이미 끊었습니다.

---

## 9. position-N checklist

splice site에서 type system은 아무것도 검사하지 않으므로, 이 검사들은 당신 몫입니다. 순서대로
실행하십시오. 처음 셋은 **processor**에 대한 것이고, 마지막 둘은 **position**에 대한 것입니다.

1. **Lifecycle.** `FrameProcessor.process_frame`이, 정확히 한 번, 첫 statement로, 이른 return을 포함한
   모든 경로에서 실행되는가? 보통은 `await super().process_frame(frame, direction)`을 뜻하고, 부모가
   그 자체로 processor이고 당신이 그 behaviour를 대체하는 중이라면 `voicemail_detector.py:170`의
   unbound form을 뜻할 수도 있습니다. 이것을 실패하면 §7.3의 silent black hole입니다.
2. **Transparency.** 이 processor가 consume하지 *않는* 모든 frame class에 대해, **frame이 도착한
   방향으로** `push_frame` 하는가? reference implementation은 `identity_filter.py:44-45`입니다.
   reference *anti*-pattern은 맨 `return`(`sentence.py:50-51`)이거나 `direction`을 버리는
   `push_frame`(`sentence.py:56`)입니다.
3. **base class의 identity.** 이것이 실제로 `FrameProcessor`인가? 이 repo에는 다른 arity와 return type을
   가진 `process_frame`이라는 이름의 method를 정의하는 hierarchy가 7개 더 있습니다(§2.4). 그중 어느
   것도 link되지 않습니다. method 이름이 일치하기 때문에 정확히 그 이유로 틀리기 쉽습니다.
4. **Upstream supply.** 이것이 consume하는 frame이 position N보다 앞의 무언가에 의해 실제로
   *produce*되는가? LLM 위쪽의 `TextFrame` consumer는 legal하고, 조용하고, 쓸모없습니다.
5. **Downstream demand.** position N 뒤의 무언가가 이 processor가 삼키거나 다시 쓰는 것을 여전히
   필요로 하는가? 양방향으로 물으십시오: splice-in에 대해서, 그리고 splice-out에 대해서.

**진단 규율:** 후보가 1-3을 실패하면 그 processor가 망가진 것입니다. 4-5를 실패하면 processor는
멀쩡하고 *position*이 틀린 것입니다. 그 두 진단을 갈라놓는 것이 이 style이 요구하는 debugging skill의
대부분입니다. interface는 둘 다 보고하지 않고, 관찰 가능한 증상 — 아무것도 안 나옴 — 은 둘이
동일하기 때문입니다.

figure의 두 failure preset은 정확히 이 구별을 훈련하기 위해 존재합니다: **'Wrong order'**는 정상
processor를 가진 position failure이고, **'Forgot super()'**는 올바른 position에 놓인 processor
failure입니다. 둘은 똑같은 침묵을 만들어냅니다.
[`figures/splice-algebra.html`](figures/splice-algebra.html)에서 둘 다 실행하고, side panel을 열기
*전에* 각각이 어느 범주인지 스스로 말하게 만드십시오.

---

## 10. 대조: `boson-agent`에는 splice해 넣을 seam이 없다

위의 모든 것은 position N이 *존재한다*고 전제합니다. 당신의 agent에는 그것이 없고, 그래서 비교가 더
날카롭습니다. 이 section의 모든 사실은 [[boson-agent-loop]]에서 나오며, private repo의 commit
`0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb`에서 읽은 것입니다. Pipecat clone에 대해서는 검증할 수
없습니다.

### 10.1 한 turn이 함수 하나다

`packages/basement/basement/loop/agent_loop.py:176` — via [[boson-agent-loop]]

```python
async def run_agent_loop(runtime: AgentRuntime, user_input: str) -> AsyncIterator[StreamEvent]:
```

561줄이고, 단일 loop를 중심으로 구축되어 있습니다:

`packages/basement/basement/loop/agent_loop.py:207-209` — via [[boson-agent-loop]]

```python
turn_count = 0
while turn_count < runtime.config.max_turns:
```

끝나는 곳은:

`packages/basement/basement/loop/agent_loop.py:363` — via [[boson-agent-loop]]

```python
        break  # Done — text response means end of turn
```

한 함수가 동시에 turn bounding, message-list mutation, provider streaming, tool dispatch, hook firing,
그리고 cancellation 이후의 history repair를 소유합니다. step을 추가한다는 것은 그 `while` body를
편집한다는 뜻입니다. **position N이 없습니다.** turn은 atomic call입니다.

### 10.2 function call은 pipe와 정반대의 connector다

`run_agent_loop`를 §1.3의 두 style invariant에 비추어 검사해 봅시다:

| Garlan & Shaw invariant | Pipecat | boson-agent |
|---|---|---|
| Filters do not know the identity of their neighbours | `link()` fills `_next`/`_prev` slots; `push_frame` names a slot, never a class (`frame_processor.py:1182`, `:1194`) | The caller names the callee directly: `async for event in run_agent_loop(runtime, content)` at `gateway/core.py:323` |
| Filters do not share state | Each processor holds its own state; frames carry data | `ctx = runtime.context_manager`, `api = runtime.conversation_api`, `hooks = runtime.hook_registry` at `:184-186` — the whole world aliased into three locals, shared outright |

두 invariant가 모두 위반되고, 그것이 곧 pipe-and-filter system이 아니라는 것의 정의입니다. 그것은
**call-and-return** system이고, 이는 다른 property 집합을 가진 별개의 이름 붙은 style입니다. 두 진술
모두 비판이 아닙니다. 그것들은 구조적 사실이며, 서로 다른 것을 예측합니다.

### 10.3 각 connector가 실제로 무엇을 주는가

이 부분은 정확하게 짚을 가치가 있습니다. 정직한 비교는 "seam이 있냐 없냐"가 아니기 때문입니다.

**call-and-return 구조가 boson-agent에서 만들어내는 것:**

- 전순서(totally-ordered)이고 grep 가능한 control flow 하나. turn 안의 모든 state transition이
  위에서 아래로 한 화면에 읽힙니다.
- 정확히 두 지점에서의 명시적 cancellation semantics: `cancellation_flag`가 `:344`(tool batch 이후)와
  `:513`(tool 하나가 완료된 이후)에서 읽힙니다. 둘 다 손가락으로 가리킬 수 있습니다. excerpt는 그
  지점들이 *어디에* 있는지의 결과도 기록합니다: 그 flag는 `TextDelta` 사이에서는 결코 검사되지 않아서,
  cooperative cancel은 token 생성을 멈출 수 없고 다음 re-prompt만 멈출 수 있습니다 —
  `gateway/interrupt/cancellation.py:171`이 주석으로 그렇게 말합니다.
- failure mode가 stack trace입니다. "왜 아무것도 안 나왔지"는 line을 이름 부르는 traceback이
  답해 줍니다.
- turn *policy*가 있어야 할 자명한 자리가 있습니다. `max_turns`는 loop condition입니다. 소진(exhaustion)
  에는 `:365`의 `while/else`가 있어서 사용자에게 보이는 메시지를 내보냅니다.

**pipe-and-filter 구조가 Pipecat에서 만들어내는 것:**

- 모든 position에 seam이 있고, splice 비용은 §3-§4에서 측정한 대로 pointer 대입 두 개입니다.
- 분산된 control flow. "왜 아무것도 안 나왔지"는 N개의 processor를 가로지르는 frame-flow trace가
  답합니다. 그것이 `FramePushed` / `FrameProcessed` observer hook이 애초에 존재하는 이유이고
  (`frame_processor.py:829`, `:1174`, `:1186`), [[ch-11/read]]가 observer plane에 대한 chapter 하나
  통째인 이유입니다.
- turn policy에는 **자명한 자리가 없습니다**. Pipecat은 어디에도 turn limit을 명시하지 않습니다.
  `max_turns`는 세는 `FrameProcessor`로 다시 만들거나 의식적으로 포기해야 합니다.
- cross-cutting state가 구조적으로 억제됩니다. invariant 1이 filter는 state를 share하면 안 된다고
  하므로, turn 전체에 진짜로 global한 것은 어떤 processor의 private state가 되거나, frame payload가
  되거나, pipeline 바깥의 무언가가 되어야 합니다 — 그리고 그것이 정확히 [[ch-10/read]]가
  `FlowManager`에서 발견하는 design입니다.

그것이 ch-01이 하는 비교의 전부입니다. 두 개의 connector, 두 벌의 결과, 판정(verdict) 없음.
두 번째 counter-design — 당신이 이미 ship한 `realtime_voice` — 는 [[ch-03/read]]의 주제이고 여기서는
의도적으로 미리 보여주지 않습니다. 비교는 그 baseline이 충분히 상세하게 확립된 다음에야 할
가치가 있기 때문입니다.

### 10.4 이것이 함의하는 migration의 모양

`run_agent_loop`에는 포팅해 *들어갈* Pipecat counterpart가 없습니다. 그것은 해체되고(dissolve), 아주
다르게 행동하는 두 더미로 해체됩니다:

- **Mechanics** — stream → `tool_uses` 수집 → 실행 → 결과 append → re-prompt. 이것이 Pipecat의 기본
  pipeline이고 공짜로 딸려 옵니다. [[ch-09/read]]가 정확한 frame topology를 보여줍니다.
- **Policy** — `max_turns`, cancellation reconciliation, `<system-reminder>` injection, 한국어 sales
  stage machine. **이 중 어느 것도 이 style 안에 자리가 없습니다.** 그것은 cross-cutting state이고,
  이 style의 첫 번째 invariant가 정확히 그것을 금지합니다.

processor를 한 줄도 쓰기 *전에* 이 분류를 하십시오. 첫 splice는 쌉니다. 하지만 네 개 processor의
`isinstance` branch에 발라놓은 policy는 이 style에서 다시 빼내기 가장 어려운 것입니다 — 빼내는 것이
splice-*out*이고, §8의 corollary에 따르면 그쪽이 위험한 방향이기 때문입니다.

**구체적인 첫 수, 그리고 진짜 test.** Lina의 sales stage를 정확히 하나만 단일 `FrameProcessor`로
포팅하십시오. 그런 다음 — `pipecat.tests.utils.run_test()`와 §6의 generic harness로 — 두 이웃 중
어느 쪽도 건드리지 않고 그것을 position N에 `list.insert`하고 `list.pop`할 수 있다는 것, 그리고 그
stage가 inert일 때 frame stream이 splice하지 않은 실행과 byte-identical하다는 것을 증명하십시오.
그 test를 통과시킬 수 없다면 당신이 가진 것은 filter의 signature를 입은 shared state이고, 당신은
integration 시점이 아니라 processor 하나 값으로 그것을 배운 것입니다.

---

## 11. Lina를 위한 세 개의 framework-extension move

기술할 수 있는 mechanics는 겨냥할 수 있는 mechanism보다 값이 쌉니다. 여기 셋이 있고, 이 chapter를
얼마나 많이 사용하는지 오름차순입니다.

**(1) `IdentityFilter` regression harness.** §5.1은 identity element가 ship되고 `tests/`에서 75번
행사(exercise)된다는 것을 확립했습니다. 그것을 당신 자신의 splice에 대한 *diff oracle*로 쓰십시오:
스크립트화된 한국어 통화를 `Pipeline([...])`에 한 번 흘리고, 다음에는 실제 rule processor를 넣으려는
그 position에 identity를 끼운 `Pipeline([..., IdentityFilter(), ...])`에 흘린 뒤, 두 frame stream이
동일함을 assert하십시오. 동일하지 않다면 당신의 harness는 nondeterministic하고, 그 이후에 하는 어떤
splice도 진단 가능하지 않습니다. 이것은 값이 싸고, 그렇지 않으면 [[ch-08/read]]에서 일주일을 태울
버그 부류 — timing에 의존하는 frame ordering — 를 잡아냅니다.

**(2) compliance kill-switch로서의 near-zero.** §5.2는 stock filter들이 control plane을 절단할 수
없음을 보여주었습니다: `SystemFrame`과 `EndFrame`은 항상 샙니다. 그것은 보통 한계로 기술됩니다.
당신의 도메인에 겨누면 그것은 당신이 위에 쌓아 올릴 수 있는 *보장(guarantee)*입니다. 한국 보험
tele-sales 통화에는 agent가 절대 produce하면 안 되는 발화가 있습니다 — 승인되지 않은 상품 주장,
수익 보장. `TextFrame`/`TTSSpeakFrame`을 predicate로 차단하는 `FunctionFilter`는 하드 mute를 주고,
algebra는 mute된 상태에서도 통화가 여전히 시작되고, 여전히 interrupt되고, 여전히 깔끔하게
종료된다는 것을 보장합니다. `StartFrame`, `InterruptionFrame`, `EndFrame`이 구성상 당신의 filter를
우회하기 때문입니다(`function_filter.py:60-71`). "session을 죽이지 않고 bot을 침묵시키기"를 공짜로
얻는 셈이고, 그것은 원래라면 직접 설계해야 했을 property입니다. 만들면서 §5.5의 함정을 유념하십시오:
당신의 predicate는 자기가 절대 gate하지 않을 frame에서도 실행되므로, pure하게 유지하십시오.

**(3) rule-strength selector로서의 `ServiceSwitcher`.** §5.6은 이 chapter에서 당신에게 leverage가
가장 큰 class입니다. Pipecat composition이 literal list로 *작성*되는 대신 *policy object로부터 계산될*
수 있음을 보여주기 때문입니다. 오늘 당신의 rule layer는 turn 단위로 arbitrate합니다.
`ServiceSwitcher`는 frame 단위로 arbitrate하지만, 같은 position을 차지하는 대안 집합에 대해서만
그렇습니다. 작동하는 매핑은 "layers → switcher"가 아니라 "하나의 decision point → 하나의 switcher"
입니다: 고지 완료 vs 미완료, 첫 통화 vs 후속 통화, 고가치 lead vs 표준 lead. 각각이 `active_service`가
두 processor 중 하나를 반환하는 strategy가 됩니다. 그것이 명시적으로 할 수 *없는* 것은 당신의
`ACTION_PRIORITY` 테이블이 하는 일입니다 — N개 layer로부터 경쟁하는 action을 모아 rank로 해소하는 것
(`layers/pipeline.py:42`, via [[pipeline-composition]] 및 [[processor-vocabulary]]). switcher는
선택(select)하지, 중재(arbitrate)하지 않습니다. 그 한계를 지금 적어 두십시오. [[ch-12/read]]가 당신에게
arbitration이 살아야 할 seam을 직접 도출하게 만들 것이고, 어떤 stock class가 그것을 *거의* 해내는지
이미 알고 도착하면 그 도출이 더 나아질 것입니다.

---

## 12. 머릿속에 담아 둘 것

- lego-block property에는 원인이 하나 있습니다: **하나의 signature, 하나의 write verb, 두 값짜리
  direction enum 하나, 하나의 union type 위에서.** `frame_processor.py:820`의 `process_frame`,
  `:1004`의 `push_frame`, `:60-69`의 `FrameDirection`.
- **splicing은 `list.insert`입니다.** `:671-679`의 `link()`는 pointer 대입 두 개와 log 하나입니다.
  `pipeline.py:197-202`의 `_link_processors`는 `link`를 operator로 하는 fold입니다.
  `src/pipecat/pipeline/`의 무엇도 composition을 validate하지 않습니다.
- **algebra는 실제로 ship됩니다**: identity(`identity_filter.py:17`), near-zero(`null_filter.py:18`),
  type에 의한 associativity(`base_pipeline.py:19`, `pipeline.py:91`, `worker.py:537`에서 의존),
  parallel(`parallel_pipeline.py:24`), conditional(`function_filter.py:21`),
  higher-order(`service_switcher.py:247`).
- **control plane은 compose로 없앨 수 없습니다.** `SystemFrame`과 `EndFrame`은 설계상 모든 stock
  filter를 통과해 샙니다. 그것은 실수가 아니라 보장입니다.
- **두 개의 silent tax.** `FrameProcessor.process_frame`에 도달시키는 것을 잊으면 도달 불가능한 guard가
  바로 옆에 앉아 있는 black hole이 생기고, 잘못된 배치는 동작은 하지만 침묵하는 bot을 만듭니다.
  둘 다 position-N checklist로 진단되지, 컴파일러로는 결코 진단되지 않습니다.
- **이 commit에서 재측정한 수치:** `src/`의 117개 파일에 `process_frame` definition 131개, 그중
  **101개가 pipeline interface**로 87개 파일에 있고(override 100개 + base 1개), **30개는 서로 무관한
  7개 hierarchy**에 속하며 30개 파일에 있습니다. override는 **100/100**이 base에 도달합니다 — 99개는
  `super()`로, 1개는 `voicemail_detector.py:170`의 unbound grandparent call로, **0**개가 건너뜁니다.
  repo에서 이 rule을 언급하는 markdown 파일은 **0**개입니다.
- **boson-agent는 pipe-and-filter가 아니라 call-and-return이고**, 구성상 두 style invariant를 모두
  위반합니다. 그 구조가 주는 것은 grep 가능한 control flow 하나, 이름 부를 수 있는 cancellation site
  둘, 그리고 stack trace입니다. pipe 구조가 주는 것은 모든 position의 seam, 분산된 control flow,
  그리고 turn policy를 위한 자리 없음입니다. 여기에 판정은 없습니다. [[ch-13/read]]가 유일하게
  무언가를 채점하는 곳입니다.

---

## 다음 챕터로

이 chapter는 네 가지를 앞으로 넘깁니다.

**[[ch-02/read]]로 (the narrow waist).** §2.3과 §8은 하나의 질문을 열어 둔 채 남기고, 그것이 핵심
질문입니다: uniform interface는 오직 단일한 universal type이 있기 때문에 작동하고, §8은 그 type의
대가가 백 개의 `isinstance` branch에서 지불된다는 것을 보여주었습니다. ch-02는 `Frame`을 sum type으로
열어 실제로 그 안에 무엇이 있는지 세고, 그 waist가 무엇에 대해 충분히 *narrow*해야 하는지를
묻습니다 — `SystemFrame` / `DataFrame` / `ControlFrame`이 서로 다른 두 가지 일(priority와
interruptibility)을 동시에 하는 3-way split이라는 구체적 긴장을 포함해서.

**[[ch-04/read]]로 (the runtime).** §7.3은 two-queues-two-tasks 구조를 버그 하나를 설명할 만큼만
추적했습니다. 전체 그림 — `frame_processor.py:132-183`의 `START_PRIORITY = 1` / `SYSTEM_PRIORITY = 10` /
`DEFAULT_PRIORITY = 20` tier를 가진 `FrameProcessorQueue(asyncio.PriorityQueue)`, `WorkerRunner`, 그리고
비대칭적인 `CancelFrame` / `EndFrame` shutdown — 이 ch-04의 주제이고, 각 mechanism이 당신이 이미
가지고 있는 질문에 답하도록 하나의 Lina sales call 위에 다시 세워집니다.

**[[ch-03/read]]로 (the baseline).** §10은 counter-design 하나를 의도적으로 주었습니다:
call-and-return agent loop. 두 번째 counter-design은 당신이 이미 ship한 것이고, ch-03은
`realtime_voice`를 Pipecat이 open sum type을 가진 자리에서 *closed* union으로 특징짓습니다. 이
course의 이후 모든 비교가 그 baseline에 기대므로, 끝에서 만들어내는 대신 시작에서 확립합니다.

**[[ch-12/read]]로 (rule layers).** §5.6과 §11은 구체적인 한계를 심어 두었습니다: `ServiceSwitcher`는
선택하지 중재하지 않고, `ACTION_PRIORITY` 스타일의 해소에는 stock으로 존재하는 자리가 없습니다.
ch-12는 세 개의 constraint를 주고 rule-processor seam을 당신이 직접 도출하게 만듭니다.
position-N checklist를 외운 채로 도착하십시오 — 그것이 그 도출이 사용하는 도구입니다.

**실무 carry-over.** ch-02 전에 §10.4 연습을 하십시오: `run_agent_loop`의 모든 조각을 *mechanics*
(Pipecat에서 공짜)이거나 *policy*(자리 없음)로 이름 붙이십시오. 그것을 하는 데 Pipecat이 설치되어 있을
필요는 없고, 그 분류가 이 course의 남은 부분이 당신에게 요구하는 거의 모든 design decision의
입력값입니다.
