---
title: "The Narrow Waist: Frame as a Sum Type and What It Costs (한국어 companion)"
chapter: ch-02
phase: composition
course: pipecat
companion_of: "[[read]]"
sources:
  - theory-narrow-waist
  - frame-taxonomy
  - flows-actions
  - pipecat-design-philosophy
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# The Narrow Waist — Frame은 sum type이고, 그 대가는 무엇인가

> 이 문서는 [[read]] (`ch-02/read.md`)의 한국어 companion입니다. **section 번호가 원문과 1:1로
> 일치**하므로 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, line number,
> 측정 수치는 원문 그대로이며 번역하지 않았습니다. CS/ML 용어는 영어를 유지합니다
> (frame, processor, pipeline, queue, back-pressure, uniform interface, sum type,
> narrow waist, dispatch, mixin — 번역하지 않음).

## 왜 이 챕터인가

[[ch-01/read]]는 splicing의 **mechanism**을 보여줬습니다. 하나의 method signature, 하나의
write verb, body 전체가 pointer assignment 두 줄과 log 한 줄뿐인 `link()`, 그리고 아무것도
validate하지 않는 `_link_processors` fold. 그 chapter가 답한 질문은 *어떻게* processor를
position N에 꽂아 넣을 수 있는가였습니다.

그 chapter가 답하지 **않은** 질문은 *왜 그게 애초에 가능한가*입니다. uniform method signature는
그 안을 흐르는 것이 uniform하지 않다면 아무 가치가 없습니다. `async def process_frame(self,
frame: Frame, direction: FrameDirection)`가 모든 position에서 type-check되는 이유는 오직
**정확히 하나의 datatype만이 모든 boundary를 건너기** 때문입니다 — `Frame`. Lego block은
증상이고, hourglass가 원인입니다.

이 chapter는 그 하나의 datatype에 대한 것입니다. 그것이 무엇을 실어 나르는가(아무것도 아님),
어떻게 쪼개지는가(content taxonomy가 아니라 scheduling contract), 몇 개나 되는가(한 파일 안에
119개의 concrete class, repo 전체로 150개 — AST walk로 세었고 그 과정을 보여줄 것입니다),
그 대가가 무엇인가(136개 파일에 걸친 577개의 `isinstance` site, 그리고 어디에도 automatic
pass-through가 없음), 그리고 — 이 부분이 실제로 쓰게 될 부분인데 — **어떤 boson 개념이 frame이
되는 것을 허용받는지 결정하는 규칙**입니다.

그 규칙은 조언이 아닙니다. Pipecat 자신의 `flows/` package는 node, transition, action을 갖는
conversation state machine이고, 구조적으로 boson의 stage machine ([[boson-stage-machine]])
더하기 layered rule engine ([[boson-layers-rules]])과 **같은 문제**입니다. `flows/`가 추가한
frame class는 **정확히 2개**입니다. 20개가 아닙니다. 그 선례가 여러분이 물려받는 budget이고,
section 12는 그것을 모든 boson 개념에 migration code 한 줄 쓰기 전에 돌려볼 수 있는 3-way
test로 바꿉니다.

이 chapter가 **하지 않는** 것 하나: migration이 할 만한 가치가 있는지는 말하지 않습니다.
migration을 한다면 waist가 무엇을 요구하는지를 말할 뿐입니다. keep/replace 논쟁은
[[ch-13/read]]의 것이고 다른 어디의 것도 아닙니다.

---

## 1. 위를 세고, 아래를 세고, 가운데를 세라

이론에 들어가기 전에 측정부터 합니다. hourglass 주장은 falsifiable합니다. plurality는 위와
아래에 살고 singularity는 가운데에 산다고 말하니까요. 그러니 세어봅시다.

**glass의 위쪽 — services.**

```
$ ls src/pipecat/services | wc -l
73
$ ls -d src/pipecat/services/*/ | wc -l
62
$ ls src/pipecat/services/*.py | wc -l
11
```

62개의 provider **directory**(vendor당 하나) 더하기 11개의 느슨한 `.py` module — 이 11개는
vendor가 아니라 abstract service base입니다.

```
src/pipecat/services/__init__.py        src/pipecat/services/stt_latency.py
src/pipecat/services/ai_service.py      src/pipecat/services/stt_service.py
src/pipecat/services/image_service.py   src/pipecat/services/tts_service.py
src/pipecat/services/llm_service.py     src/pipecat/services/vision_service.py
src/pipecat/services/mcp_service.py     src/pipecat/services/websocket_service.py
src/pipecat/services/settings.py
```

정확성을 위한 note: [[pipecat-design-philosophy]]는 "`src/pipecat/services/` 아래 73 dirs"라고
보고합니다. 73은 directory 수가 아니라 **entry** 수입니다. 이 commit에서 directory count는
62입니다. excerpt와 tree가 어긋나면 **tree가 이깁니다** — 이건 이 course 전체의 standing rule입니다.

**glass의 아래쪽 — transports와 serializers.**

```
$ ls src/pipecat/transports/
__init__.py  base_input.py  base_output.py  base_transport.py
daily  heygen  lemonslice  livekit  local  moq  smallwebrtc  tavus  vonage  websocket  whatsapp
```

11개의 provider package, 더하기 3개의 abstract base. 그리고 `serializers/` 아래에는:

```
$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

그 package 안에 7개의 concrete `FrameSerializer` 구현 —
`ExotelFrameSerializer`, `GenesysAudioHookSerializer`, `PlivoFrameSerializer`,
`ProtobufFrameSerializer`, `TelnyxFrameSerializer`, `TwilioFrameSerializer`,
`VonageFrameSerializer`. (정밀하게 말하면: tree에는 여덟 번째 `FrameSerializer` subclass인
`RTVIEvalSerializer`가 `src/pipecat/evals/serializer.py:87`에 있지만, 이건 test-harness
serializer이고 transport bottom의 일부가 아닙니다. glass 아래쪽의 숫자는 7입니다.)

**glass의 가운데.**

class 하나. `Frame`, `src/pipecat/frames/frames.py:65`.

위에 62개 vendor, 아래에 11개 transport와 7개 wire format, 그 사이에 하나의 datatype. 그게
이 commit에서 측정된 hourglass입니다. 이 chapter의 나머지 전부는 그 결과입니다.

> 💡 **쉬운 설명 — hourglass를 숫자로 다시 보기.**
> 만약 waist가 없다면, 62개 vendor × (11 transport + 7 serializer) = 1,116개의 짝을 각각
> 이어붙이는 adapter code가 필요합니다. waist가 있으면 vendor는 "Frame을 emit/consume한다"
> 하나만 구현하고, transport도 "Frame을 실어 나른다" 하나만 구현합니다. 62 + 18 = 80.
> 1,116 → 80. 이 비율이 hourglass가 사는 이유 전부입니다.

---

## 2. 문헌 — 이름과 연도까지

code를 쓰는 데 이 section이 필요하진 않습니다. 필요한 이유는, 앞으로 나올 failure mode들이
*이미 이름이 붙은* failure mode이고, 이름을 아는 것이 boson에서 그것들이 일주일을 잡아먹기
전에 알아보는 방법이기 때문입니다.

**먼저 worked example 하나. "narrow waist"는 낯선 operational concept이고, 보통 mechanism보다
diagram이 먼저 가르쳐지기 때문입니다.**

Wi-Fi 위의 laptop에서 fibre 위의 server로, 들어본 적도 없는 4개의 중간 carrier를 거쳐 message를
보낸다고 합시다. 세상에 physical link technology가 40개, application이 400개 있다고 칩시다.
모든 application이 모든 link technology를 말해야 한다면, adapter code 조각이 16,000개
필요하고, link technology 하나를 새로 추가하면 400개의 adapter를 새로 써야 합니다. 대신 모두가
가운데의 **하나의** format에 합의합니다 — IP packet — 그리고 그건 거의 아무것도 말하지 않습니다.
source address, destination address, payload, flag 몇 개. reliability 없음, ordering 없음,
security 없음, session 없음. 새 link technology는 adapter **하나**를 쓰고(IP packet을 실어
나른다), 새 application도 adapter **하나**를 씁니다(IP packet을 emit한다). 16,000이 440이 됩니다.

직관에 반하는 부분, 그리고 이 문헌 전체의 thesis: **가운데의 format이 채택 가능한 이유는 바로
그것이 약속하는 게 너무 적기 때문입니다.** 만약 IP가 in-order delivery를 보장했다면, 모든 link
technology가 자격을 얻기 위해 reordering을 구현해야 했을 것이고, 대부분은 그냥 안 했을 겁니다.
waist에서의 generality는 capability를 *더해서*가 아니라 *빼서* 사는 것입니다.

이제 citation.

- **figure의 기원.** CSTB/NRC, *Realizing the Information Future: The Internet and
  Beyond*, National Academy Press, **1994** — 아래에 많은 transmission technology, 위에 많은
  application, waist에 하나의 packet-transport "bearer service".
- **어휘.** Steve Deering, **"Watching the Waist of the Protocol Hourglass,"**
  keynote, **ICNP '98**, Austin TX, 1998년 10월. "narrow waist"와
  *IP over everything / everything over IP*를 표준 용어로 만든 바로 그 talk입니다.
- **이론.** Micah Beck, **"On the Hourglass Model," *CACM* 62(7):48–57, 2019년 7월**.
  thesis는 **Deployment Scalability Trade-off**입니다: spanning layer가 *약하고* 더 minimal할수록,
  그것을 지원하는 구현이 많아지고 그 위에 세워지는 application도 많아진다. Beck은 "spanning
  layer"라는 용어를 David D. Clark에게 credit합니다.
- **evolvability, 경험적으로.** Akhshabi & Dovrolis, *The Evolution of Layered Protocol
  Stacks Leads to an Hourglass-Shaped Architecture*, **SIGCOMM '11**. 이들의 EvoArch model은
  hourglass가 competition에서 emerge한다는 것, 그리고 waist가 **ossify(경화)**한다는 것을
  보여줍니다 — 위아래 layer가 자유롭게 innovate하는 것은 *바로 waist가 움직일 수 없기 때문*입니다.
- **청구서.** Philip Wadler, **"The Expression Problem,"** java-genericity mailing list,
  **1998년 11월 12일**, 원문 그대로: *"The goal is to define a datatype by cases, where one can add
  new cases to the datatype and new functions over the datatype, without recompiling
  existing code, and while retaining static type safety (e.g., no casts)."*

Wadler는 section 8까지 붙잡아 두세요. Akhshabi & Dovrolis는 section 10까지. Beck은 지금 바로
test 가능하니 지금 test합니다.

> 💡 **쉬운 설명 — Deployment Scalability Trade-off를 한 문장으로.**
> "spanning layer가 더 많은 것을 보장할수록, 그 보장을 못 지키는 구현들이 탈락한다."
> USB-C를 생각해 보세요. 만약 USB-C 규격이 "100W 전력 공급 필수"를 요구했다면 이어폰 어댑터는
> USB-C가 될 수 없었을 겁니다. 요구가 적어서 모두가 낄 수 있는 것입니다.

---

## 3. `Frame`은 아무것도 실어 나르지 않는다 — Beck의 minimality, 문자 그대로

Beck이 옳다면, waist class는 여전히 유용하면서도 가능한 한 아무것도 assert하지 않아야 합니다.
열어봅시다.

**`src/pipecat/frames/frames.py:64-101`**

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

    def __post_init__(self):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"
        self.pts: int | None = None
        self.broadcast_sibling_id: int | None = None
        self.metadata: dict[str, Any] = {}
        self.transport_source: str | None = None
        self.transport_destination: str | None = None

    def __str__(self):
        return self.name
```

**거기에 없는 것**을 읽으세요. payload field가 없습니다. `data`도, `bytes`도, `content`도,
`kind` discriminator string도, 믿을 수 있는 timestamp도 없습니다(`pts`는 `None`으로 set되고
나중에 신경 쓰는 쪽이 채웁니다). direction도 없고 schema version도 없습니다.

일곱 field 전부가 `field(init=False)`입니다. 이건 style 선택이 아니라 의도적인 constructional
fact입니다: **이것들 중 어느 것도 constructor에 넘길 수 없습니다.** `Frame` subclass의
`__init__` signature는 전적으로 자기 자신의 dataclass field로만 구성되고, base의 일곱 field는
그 뒤에 `__post_init__`에서 assign됩니다. base class는 그 아래 119개 class에 constructor
argument를 **0개** 기여합니다.

일곱 field 중 둘은 순수 identity (`id`, `name`), 둘은 transport가 채우는 routing hint
(`transport_source`, `transport_destination`), 하나는 both-directions broadcast가 쓰는
correlation pointer (`broadcast_sibling_id`), 하나는 timestamp slot (`pts`), 그리고 하나는
untyped escape hatch (`metadata`)입니다. 그게 spanning layer의 전부입니다.

이건 정확히 Beck의 예측입니다. waist가 거의 아무것도 assert하지 않으므로, 거의 모든 것이 그것을
구현할 수 있습니다. 구체적으로: Pipecat frame이 되기 위해 어떤 type이 공급해야 하는 것은
*전혀 없습니다* — `@dataclass class MyFrame(ControlFrame): pass`는 완결되고 legal하며 routable한
frame입니다. 입장료가 0입니다. 그래서 119개가 있는 것입니다.

> 💡 **쉬운 설명 — `field(init=False)`가 왜 중요한가.**
> 보통 dataclass에서 base class가 field를 선언하면 subclass의 `__init__`에 그 field가 앞자리로
> 끼어들어 옵니다. 그러면 `TextFrame("hello")` 같은 자연스러운 생성이 깨지고, base field 순서가
> 119개 class의 constructor를 인질로 잡습니다. `init=False`는 그 결합을 끊습니다. base가
> subclass의 API 표면에 **한 글자도** 나타나지 않게 만드는 장치입니다. 이것이 "waist는
> 아무것도 요구하지 않는다"의 코드 수준 구현입니다.

---

## 4. 3-way split은 taxonomy가 아니라 scheduling contract다

`Frame` 바로 아래에 세 개의 class가 있습니다. 뻔한 추측은 이들이 *content*를 분류한다는 것입니다 —
system 것, data 것, control 것. 그 추측은 틀렸고, docstring이 자기 입으로 그렇게 말합니다.
셋을 함께 읽으세요.

**`src/pipecat/frames/frames.py:104-140`**

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

저 세 docstring에서 구분을 만드는 모든 절은 content가 아니라 **time**에 대한 것입니다:
"higher priority," "handled in order," "not affected by user interruptions," "cancelled by
user interruptions," "after everything is flushed." content를 언급하는 유일한 절 —
"usually contains data such as LLM context, text, audio or images" — 은 "usually"로 hedge되어
있고 normative가 아니라 descriptive합니다.

그러니 frame 저자가 base class를 고르면서 답하는 질문은 *이게 무엇인가?*가 아니라
**언제 실행되어야 하고, barge-in에서 살아남는가?**입니다.

code가 그것을 두 군데에서 증명합니다.

**첫 번째 장소 — priority queue.** `src/pipecat/processors/frame_processor.py:132-171`

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

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()
        # Counts every frame enqueued, which keeps frames of the same tier in
        # arrival order and stops the queue from ever having to compare frames.
        self.__counter = 0

    async def put(self, item: tuple[Frame, FrameDirection, FrameCallback | None]):
        """Put an item into the priority queue.

        The `StartFrame` outranks every other frame and `SystemFrame` frames
        outrank data and control frames.

        Args:
            item: The frame to enqueue, with its direction and callback.

        """
        frame, _, _ = item
        if isinstance(frame, StartFrame):
            priority = self.START_PRIORITY
        elif isinstance(frame, SystemFrame):
            priority = self.SYSTEM_PRIORITY
        else:
            priority = self.DEFAULT_PRIORITY

        self.__counter += 1
        await super().put((priority, self.__counter, item))
```

tier가 셋이고, 세 번째 tier는 `else`입니다. branch 구조를 세게 들여다보세요. section 5가 그
안에 살고 있습니다: queue는 `StartFrame`을 test하고, 그다음 `SystemFrame`을 test하고, 그다음
포기하고 20을 assign합니다. `DataFrame`이나 `ControlFrame`은 **이름조차 언급되지 않습니다**.

> 💡 **쉬운 설명 — `self.__counter`가 왜 필요한가.**
> `asyncio.PriorityQueue`는 tuple을 비교하는데, priority가 같으면 그다음 원소를 비교하려고
> 합니다. counter가 없다면 `(20, frame_a)` vs `(20, frame_b)`에서 `Frame`끼리 `<` 비교를
> 시도하다 `TypeError`가 납니다. counter는 (a) tie-break을 arrival order로 고정하고 (b)
> frame을 절대 비교하지 않게 만드는 두 가지 일을 동시에 합니다. comment가 정확히 그 말입니다.

**두 번째 장소 — input loop.** `src/pipecat/processors/frame_processor.py:1295-1312`

```python
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
```

`SystemFrame`은 **input task 위에서 inline으로** 실행됩니다. 나머지 전부는 별도로 cancel 가능한
두 번째 queue로 밀려 들어갑니다. 그게 out-of-band mechanism의 전부이고, 여섯 줄입니다.
`InterruptionFrame`이 말하느라 바쁜 processor에 도달할 수 있는 이유는 scheduler도 supervisor도
아니라 line 1304의 `isinstance(frame, SystemFrame)`입니다.

boson으로 가져가야 할 design 결론: **Pipecat에서 priority는 structural하고 class-definition
time에 한 번 선언된다.** per-send priority argument가 없고 `push_frame(frame, urgent=True)`도
없습니다. barge-in 동안 boson signal이 queue를 뛰어넘게 하고 싶다면, 그건 `class X(SystemFrame)`을
쓸 때 내리는 결정이고, call site마다 바꿀 수 없습니다.

---

## 5. 정직한 발견 #1 — runtime에서 split은 사실 2-way다

이제 section 4의 관찰을 현금화합니다. 각 branch에 대한 dispatch를 tree에서 grep합니다.

```
$ grep -rn --include='*.py' "isinstance(frame, SystemFrame)" src/pipecat/ | wc -l
10
$ grep -rn --include='*.py' "isinstance(frame, DataFrame)" src/pipecat/ | wc -l
0
$ grep -rn --include='*.py' "isinstance(frame, ControlFrame)" src/pipecat/ | wc -l
0
```

`SystemFrame`은 10번 dispatch됩니다. `DataFrame`과 `ControlFrame`은 `src/pipecat` 어디에서도
**0번** dispatch됩니다. grep을 넓혀도 결과는 유지됩니다: `frames.py` 밖에서 `DataFrame`이
등장하는 모든 non-import occurrence는 `class X(DataFrame):` 선언입니다 —
`DailySIPTransferFrame`, `OpenClawSendFrame`, `LLMSearchResponseFrame` 등등. `DataFrame`은
오직 상속용 base로만 쓰이고, test 대상으로는 절대 쓰이지 않습니다.

그래서 runtime taxonomy는: **`StartFrame`, 그다음 `SystemFrame`, 그다음 나머지 전부.**
Data-versus-Control은 사람이 `frames.py`를 읽는 데 도움을 주는 documentation convention입니다.
core pipeline에서 runtime 의미가 없습니다.

그러면 뻔한 후속 질문이 생깁니다. `DataFrame` docstring은 "Data frames are cancelled by user
interruptions"라고 말합니다. 아무도 `DataFrame`을 test하지 않는다면, 실제로 무엇이 그것들을
cancel합니까?

**`Frame`이 아닌 mixin이 합니다.** `frames.py:147`의 `UninterruptibleFrame`:

**`src/pipecat/frames/frames.py:146-158`**

```python
@dataclass
class UninterruptibleFrame:
    """A marker for data or control frames that must not be interrupted.

    Frames with this mixin are still ordered normally, but unlike other frames,
    they are preserved during interruptions: they remain in internal queues and
    any task processing them will not be cancelled. This ensures the frame is
    always delivered and processed to completion.

    """

    pass
```

이것은 `Frame`을 상속하지 **않습니다**. 맨 marker dataclass이고,
`class EndFrame(ControlFrame, UninterruptibleFrame)`처럼 씁니다. 그리고 이건 dispatch됩니다 —
세 번, 전부 한 파일 안에서.

**`src/pipecat/utils/frame_queue.py:73-95`**

```python
    def _put(self, item: Any) -> None:
        if isinstance(self._frame_getter(item), UninterruptibleFrame):
            self._uninterruptible_count += 1
        super()._put(item)

    def _get(self) -> Any:
        item = super()._get()
        if isinstance(self._frame_getter(item), UninterruptibleFrame):
            self._uninterruptible_count -= 1
        return item

    def reset(self) -> None:
        """Remove all non-UninterruptibleFrame items, keeping uninterruptible ones."""
        kept: asyncio.Queue = asyncio.Queue()
        while not self.empty():
            item = self.get_nowait()
            if isinstance(self._frame_getter(item), UninterruptibleFrame):
                kept.put_nowait(item)
            self.task_done()
        while not kept.empty():
            item = kept.get_nowait()
            self.put_nowait(item)
            kept.task_done()
```

`reset()`은 interruption purge입니다. `UninterruptibleFrame`을 지키고 나머지를 전부 버립니다 —
버려지는 frame이 `DataFrame`인지 `ControlFrame`인지, 아니면 (section 6 참조) 어느 branch에도
없는지와 **무관하게**. 그리고 caller는 interruption path 그 자체입니다.

**`src/pipecat/processors/frame_processor.py:1130-1151`**

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

docstring이 흐려놓은 그 문장을 그대로 말합시다.

> **frame은 interruption에서 기본적으로 cancel된다. `UninterruptibleFrame` mixin을 달고 있을
> 때만 살아남는다. `DataFrame`이나 `ControlFrame` 소속 여부는 그것과 아무 상관이 없다.**

`EndFrame` / `CancelFrame` 쌍이 반대편에서 같은 얘기를 합니다. `EndFrame` (`frames.py:1899`)은
`ControlFrame, UninterruptibleFrame`입니다 — graceful drain: 앞에 queue된 모든 것 뒤에 도착하고,
purge될 수 없습니다. `CancelFrame` (`frames.py:999`)은 `SystemFrame`입니다 — hard stop:
그것들보다 *앞서* 도착합니다. 두 개의 서로 다른 shutdown semantics가 전적으로 base-class 선택만으로
표현되고, 그 둘을 비교하는 shutdown code는 없습니다.

**왜 신경 써야 하는가.** section 12에서 `class RuleViolationFrame(ControlFrame)`을 쓸 때,
여러분은 barge-in 결정을 내리는 것이고, base class만으로는 그 결정이 완전하지 않습니다. 고객이
interrupt했어도 verdict가 반드시 전달되어야 한다면 — 예를 들어 법적 결과가 있는 DNC registration
([[boson-stage-machine]]은 DNC를 `register_dnc`로 route합니다) — `ControlFrame`만으로는
부족합니다. `class X(ControlFrame, UninterruptibleFrame)`이 필요하고, 그걸 알아야 하는 이유는
어떤 docstring이 말해줘서가 아니라 여러분이 `frame_queue.py`를 읽었기 때문입니다.

> 💡 **쉬운 설명 — 왜 mixin이지 base class가 아닌가.**
> "interrupt에서 살아남는다"는 성질은 "언제 실행되는가"(3-way split)와 **직교**하는 축입니다.
> 만약 이것도 base class 계층으로 표현했다면 `UninterruptibleControlFrame`,
> `UninterruptibleDataFrame`... 조합 폭발이 일어납니다. 직교하는 성질은 mixin으로 빼는 것이
> 정석이고, Pipecat은 그렇게 했습니다. 다만 대가로 "base class만 보면 안 되고 MRO 전체를
> 봐야 한다"는 인지 부담이 생깁니다.

---

## 6. 정직한 발견 #2 — taxonomy가 샌다

`LLMContextFrame`은 voice pipeline에서 아마 가장 중요한 frame입니다. LLM service에게 "여기
conversation이 있으니 generate해라"라고 말하는 것이니까요. 무엇을 상속하는지 보세요.

**`src/pipecat/frames/frames.py:550-561`**

```python
@dataclass
class LLMContextFrame(Frame):
    """Frame containing a universal LLM context.

    Used as a signal to LLM services to ingest the provided context and
    generate a response based on it.

    Parameters:
        context: The LLM context containing messages, tools, and configuration.
    """

    context: LLMContext
```

`class LLMContextFrame(Frame)`. `SystemFrame`도 `DataFrame`도 `ControlFrame`도 아니고 —
`Frame` 직접입니다. **어느 branch에도 없습니다.**

`frames.py`에 대한 제 AST walk는 파일 전체에서 정확히 하나의 그런 class를 찾습니다.

```
in no branch: ['LLMContextFrame']
in two branches: ['InputTextRawFrame']
```

(`frames.py:1481`의 `InputTextRawFrame`은 거울상 anomaly입니다:
`class InputTextRawFrame(SystemFrame, TextFrame)`이므로 `SystemFrame`과 `DataFrame` 양쪽에
모두 counting됩니다. 두 anomaly는 산술에서 상쇄됩니다 — section 7 참조.)

어느 branch에도 없는 frame에게는 무슨 일이 일어날까요? `FrameProcessorQueue.put()`으로 돌아가
보세요. `StartFrame`을 test하고, `SystemFrame`을 test하고, `else: priority =
self.DEFAULT_PRIORITY`. `LLMContextFrame`은 20으로 떨어집니다. 정확히 `DataFrame`처럼
행동합니다 — ordered, interruption에서 purgeable — 그러나 **선언에 의해서가 아니라 fallthrough에
의해서입니다.**

오늘 실질적으로 깨지는 건 없습니다. default-priority가 이 frame에 원하는 behaviour니까요.
문제는 epistemic하고, migration 도중에 무는 종류의 문제입니다: `LLMContextFrame`의 scheduling
behaviour는 class 정의 어디에도 명시되어 있지 않습니다. 그것은 *다른 파일*에 있는 `else`
branch의 emergent property입니다. 나중에 누군가 그 `else`를 바꾸면 — 예를 들어 unclassified
frame에 대해 raise하거나, unclassified frame을 `SYSTEM_PRIORITY`로 route하도록 — `LLMContextFrame`은
조용히 scheduling class가 바뀌고 `frames.py`에서는 아무것도 달라 보이지 않습니다.

이것이 leaky sum type이 실제로 어떻게 생겼는지입니다. type system은 `class X(Frame)`을
허용하므로, docstring이 기술하는 3-way contract는 compiler가 강제하지 않는 convention입니다.
그리고 이건 cost 구조 전체의 예고편입니다. 이제 개수를 셉니다.

---

## 7. 대가, 측정 — 그리고 counting trap

> **Figure.** 지금 [`figures/frame-waist.html`](figures/frame-waist.html)을 열고 section 7과
> 8 내내 열어두세요. 이 commit의 hourglass를 여러분이 곧 검증할 count로부터 그려주고, 그다음
> cost calculator로 변합니다: "add a column"을 클릭해 새 processor가 즉시 compose되는 것을
> 보고, 그다음 "add a row"를 클릭해 같은 동작이 577개 `isinstance` site를 전부 밝히고 여러분의
> frame을 조용히 drop할 processor들을 flag하는 것을 보세요. 두 클릭을 연달아 해보세요 — 둘
> 사이의 asymmetry가 section 8의 논지 전부이고, 산문보다 animation으로 훨씬 빨리 꽂힙니다.

`frames.py`는 2,415줄입니다. 질문은 그것이 몇 개의 frame type을 정의하느냐이고, 여기가 Pipecat에
대한 모든 순진한 측정이 틀어지는 지점입니다 — [[theory-narrow-waist]]의 측정도 포함해서요.
그 excerpt는 in-file concrete 120개, repo-wide 151개라고 보고합니다. 이 commit에서 올바른
숫자는 **119**와 **150**이고, 그 차이는 짚고 넘어갈 만합니다. 같은 trap이 여러분 자신의
migration을 audit할 때 그대로 물 테니까요.

**grep하지 마세요.** `grep -c "^class .*Frame"`은 overcount하고(mixin, non-frame helper)
undercount합니다(base 이름이 문자 그대로 `*Frame`이 아닌 subclass). AST를 walk하고 transitive
closure를 계산하세요.

```python
import ast
def basename(b):
    # normalize subscripted bases: Generic[X] -> Generic
    while isinstance(b, ast.Subscript): b = b.value
    if isinstance(b, ast.Name): return b.id
    if isinstance(b, ast.Attribute): return b.attr
    return None

t = ast.parse(open('src/pipecat/frames/frames.py').read())
tops = [n for n in t.body if isinstance(n, ast.ClassDef)]
bases = {c.name: [basename(b) for b in c.bases] for c in tops}

desc, changed = {'Frame'}, True
while changed:                       # fixed point
    changed = False
    for n, bs in bases.items():
        if n not in desc and any(b in desc for b in bs):
            desc.add(n); changed = True
```

결과:

```
top-level ClassDefs: 129
Frame descendants (incl Frame): 123
NON-descendants: 6 ['UninterruptibleFrame', 'AudioRawFrame', 'ImageRawFrame',
                    'FunctionCallResultProperties', 'DTMFFrame', 'FunctionCallFromLLM']
concrete (desc minus Frame + 3 branch): 119
SystemFrame 46
DataFrame 33
ControlFrame 40
```

**129개 top-level class → 122개의 진짜 `Frame` subclass → 119개 concrete frame type**,
3개의 branch class(`SystemFrame`, `DataFrame`, `ControlFrame`)를 뺀 뒤의 숫자입니다. 이 셋은
아무도 instantiate하지 않는 abstract marker입니다.

**`Frame` descendant가 아닌 7개의 top-level class** — 부주의한 count가 정확히 어떤 것들을
쓸어 담는지 볼 수 있도록 이름을 밝힙니다:

| Class | Line | 실제 정체 |
|---|---|---|
| `Frame` | 65 | waist 그 자체 — *descendant*가 아님 |
| `UninterruptibleFrame` | 147 | mixin (section 5) |
| `AudioRawFrame` | 161 | payload mixin: `audio`, `sample_rate`, `num_channels`, `num_frames` |
| `ImageRawFrame` | 181 | payload mixin: `image`, `size`, `format` |
| `DTMFFrame` | 843 | keypad-digit payload mixin |
| `FunctionCallResultProperties` | 748 | 평범한 result-metadata dataclass |
| `FunctionCallFromLLM` | 1330 | 평범한 tool-call descriptor dataclass |

그 일곱 중 셋은 이름이 `Frame`으로 끝나는데 frame이 아닙니다. 그래서 grep이 실패합니다.

**이제 trap.** *같은* walk를 돌리되 `ast.ClassDef.bases`를 문자 그대로 읽으세요 — `ast.Name`
base만 취하고, subscript를 normalize하지 않고:

```python
bases = {c.name: [b.id for b in c.bases if isinstance(b, ast.Name)] for c in tops}
```

```
naive descendants: 120
missing: ['AudioRawFrame', 'DTMFFrame', 'FunctionCallFromLLM',
          'FunctionCallResultProperties', 'ImageRawFrame', 'UninterruptibleFrame',
          'LLMUpdateSettingsFrame', 'STTUpdateSettingsFrame', 'TTSUpdateSettingsFrame']
```

앞의 여섯은 올바르게 제외된 것들입니다. 마지막 셋은 **사라져 버린 진짜 frame**이고,
120 − 1 (`Frame`) − 3 (branch) = 116, 셋이 모자랍니다. 이유는 이렇습니다:

**`src/pipecat/frames/frames.py:2250-2251, 2282-2299`**

```python
@dataclass
class ServiceUpdateSettingsFrame(ControlFrame, UninterruptibleFrame, Generic[TSettings]):
```

```python
@dataclass
class LLMUpdateSettingsFrame(ServiceUpdateSettingsFrame[LLMSettings]):
    """Frame for updating LLM service settings."""

    pass


@dataclass
class TTSUpdateSettingsFrame(ServiceUpdateSettingsFrame[TTSSettings]):
    """Frame for updating TTS service settings."""

    pass


@dataclass
class STTUpdateSettingsFrame(ServiceUpdateSettingsFrame[STTSettings]):
    """Frame for updating STT service settings."""

    pass
```

`ServiceUpdateSettingsFrame` 자신은 naive walk에서 살아남습니다 — 첫 base인 `ControlFrame`이
평범한 `ast.Name`이니까요. 깨지는 건 그것의 **세 subclass**입니다. 이들의 유일한 base는
`ServiceUpdateSettingsFrame[LLMSettings]`, 즉 `ast.Subscript`입니다. `ast.Name` base만 모으는
walk는 이들을 base가 아예 없는 것으로 보고 hierarchy 밖으로 떨궈버립니다.

이건 반올림 오차보다 훨씬 중요합니다. 저 셋은 tree 전체에서 migration 관련성이 가장 높은 축에
드는 frame들이기 때문입니다. `LLMUpdateSettingsFrame`은 section 12에서 boson의 stage prompt를
바꾸는 데 쓸 frame이고, `TTSUpdateSettingsFrame`은 통화 중간에 한국어 TTS voice를 바꾸는
방법입니다. 이들을 조용히 떨구는 frame audit은 그 capability가 존재하지 않는다고 말해줄 것입니다.

> 💡 **쉬운 설명 — `ast.Subscript`가 뭔지 한 줄로.**
> `class A(B)`에서 `B`는 `ast.Name`입니다. `class A(B[C])`에서 `B[C]`는 `ast.Subscript`이고,
> 진짜 이름은 `.value` 안에 한 겹 들어가 있습니다. Generic을 쓰는 순간 base가 한 겹 감싸지므로,
> AST 기반 audit tool은 반드시 `while isinstance(b, ast.Subscript): b = b.value`로 껍질을
> 벗겨야 합니다. 이 한 줄이 3개의 frame을 살립니다.

**branch split.**

```
SystemFrame  : 46
DataFrame    : 33
ControlFrame : 40
             -----
               119
```

46 + 33 + 40 = 119, 정확히 맞습니다. 그런데 이건 특정한 종류의 산술적 우연입니다:
`LLMContextFrame`은 어느 branch에도 없고(합의 coverage에서 하나를 뺌), `InputTextRawFrame`은
두 branch에 있습니다(하나를 도로 더함). 우연히 상쇄되는 두 개의 독립적인 anomaly. 합이 맞는
것을 branch count의 validation으로 취급하지 마세요.

숫자가 추상적으로 남지 않도록, 대표 member들:

- **SystemFrame (46):** `StartFrame` (924), `CancelFrame` (999), `ErrorFrame` (1016),
  `InterruptionFrame` (1142), `UserStartedSpeakingFrame` (1154),
  `UserStoppedSpeakingFrame` (1165), `VADUserStartedSpeakingFrame` (1226),
  `ProposedUserStartedSpeakingFrame` (1256), `BotStartedSpeakingFrame` (1282),
  `InputAudioRawFrame` (1449), `InputDTMFFrame` (1552).
- **DataFrame (33):** `OutputAudioRawFrame` (201), `TTSAudioRawFrame` (241),
  `TextFrame` (303), `LLMTextFrame` (343), `TTSTextFrame` (417),
  `TranscriptionFrame` (450), `InterimTranscriptionFrame` (476), `LLMRunFrame` (634),
  `LLMMessagesAppendFrame` (645), `LLMSetToolsFrame` (694), `TTSSpeakFrame` (795).
- **ControlFrame (40):** `EndFrame` (1899), `StopFrame` (1923), `HeartbeatFrame` (2007),
  `LLMFullResponseStartFrame` (2059), `TTSStartedFrame` (2210), `TTSStoppedFrame` (2231),
  `ServiceUpdateSettingsFrame` (2251), `LLMUpdateSettingsFrame` (2283),
  `VADParamsUpdateFrame` (2320).

**Repo 전체.** 같은 closure를 `src/pipecat/` 아래 모든 `.py`에 돌립니다:

```
repo-wide concrete frame classes (top-level only): 150
outside frames.py: 31
  9  src/pipecat/processors/frameworks/rtvi/frames.py
  8  src/pipecat/transports/daily/transport.py
  6  src/pipecat/services/openclaw/frames.py
  3  src/pipecat/transports/livekit/transport.py
  2  src/pipecat/flows/actions.py
  1  src/pipecat/tests/utils.py
  1  src/pipecat/services/google/frames.py
  1  src/pipecat/pipeline/sync_parallel_pipeline.py
```

**150 = 119 + 31.** 저 패턴을 눈여겨보세요 — section 11의 논지 전체가 미리 나와 있습니다.
31개의 out-of-tree frame은 무작위로 흩어져 있지 않습니다. 세 개의 subsystem-local `frames.py`
파일에 뭉쳐 있습니다(RTVI 9, Daily 8, OpenClaw 6). 아무도 자기 subsystem의 어휘를 공용 waist에
추가하지 않았습니다.

**150에 대한 caveat 하나.** 이 walk는 top-level `ClassDef`만 셉니다. 그런데 `frames.py`는
indent된 block 안에 네 개를 더 숨기고 있습니다:

**`src/pipecat/frames/frames.py:1830-1848`**

```python
# The leaf aliases below intentionally subclass the deprecated TaskFrame /
# TaskSystemFrame bases; silence the subclassing DeprecationWarning that
# @deprecated would otherwise emit while this module is imported.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    @deprecated(
        "`EndTaskFrame` is deprecated since 1.4.0 and will be removed in 2.0.0. "
        "Use `EndWorkerFrame` instead."
    )
    @dataclass
    class EndTaskFrame(EndWorkerFrame, TaskFrame):
        """Deprecated alias for :class:`EndWorkerFrame`.

        .. deprecated:: 1.4.0
            Use :class:`EndWorkerFrame` instead. Will be removed in 2.0.0.
        """

        pass
```

그런 alias가 넷 있습니다 — `EndTaskFrame` (1841), `StopTaskFrame` (1855),
`CancelTaskFrame` (1869), `InterruptionTaskFrame` (1883) — `with warnings.catch_warnings():`
block 안에 nested되어 있고, function 안이 아닙니다. (확인했습니다: function body에 대한 AST
walk는 이 파일에서 nested class를 0개 찾습니다. container는 `with`입니다.) 이들을 세면 repo
총계는 154입니다. 이들은 `PipelineTask → PipelineWorker` rename에서 나온 1.4.0 deprecated
alias이므로, 저는 150을 쓰되 caveat을 숨기지 않고 명시합니다.

**dispatch tax.**

```
$ grep -rn --include='*.py' "isinstance(frame, " src/pipecat/ | wc -l
577
$ grep -rln --include='*.py' "isinstance(frame, " src/pipecat/ | wc -l
136
```

**577 site, 136 파일.** 다음 section 내내 머릿속에 붙잡고 있어야 할 숫자입니다.

---

## 8. Wadler의 청구서: column은 싸고 row는 비싸다

Wadler의 framing을 여기에 적용하면:

- **Row = datatype의 case = frame type.** `frames.py`에 119개.
- **Column = datatype 위의 function = processor.** 모든 `process_frame` override.

sum type(`isinstance` dispatch를 쓰는 class hierarchy)은 한 축을 싸게, 다른 축을 비싸게 만들고,
Pipecat은 자기가 싸게 할 축을 의도적으로 골랐습니다.

**column 추가는 거의 공짜입니다.** signature 하나짜리 method 하나를 구현하면 기존의 모든 frame과
compose됩니다:

```python
class MyProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
```

이것은 완결되고 correct하며 universally-composable한 processor입니다. 어떤 `Pipeline([...])`의
어떤 position N에도 들어가고 type-check됩니다 ([[ch-01/read]]). 기존 파일 변경 없음. 아무것도
recompile할 필요 없음. **matrix에서 새 column 말고는 단 하나의 cell도 건드리지 않습니다.**

**row 추가는 pipeline에게는 공짜이고 모든 processor에게는 부채입니다.** 사람들이 이 부분을
틀리므로, mechanism을 못박아 둡시다. base 구현으로 돌아갑니다.

**`src/pipecat/processors/frame_processor.py:820-847`**

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

마지막 줄을 읽고, 그다음 method 전체를 다시 읽으세요. **`else`가 없습니다. 이 method 어디에도
`await self.push_frame(frame, direction)`이 없습니다.**

이것이 이 chapter에서 가장 결과가 큰 하나의 사실입니다. base class는 정확히 여섯 종류의 frame을
처리하고 — `StartFrame`, `InterruptionFrame`, `CancelFrame`, 그리고 pause/resume 쌍 — 존재하는
그 밖의 모든 frame에 대해 **아무것도 하지 않고 return합니다.** forwarding은 상속되지 않습니다.
그건 subclass 저자의 일이고, 매번, `frame_processor.py:1004`의 `push_frame`을 통해서 합니다.

**Worked example — 다섯 줄로 frame 죽이기.** row를 하나 추가한다고 합시다:

```python
@dataclass
class RuleVerdictFrame(ControlFrame):
    verdict: str
    rule_name: str
```

rule processor에서 이걸 push합니다. downstream으로
`STTMuteFilter → LLMUserContextAggregator → BosonRuleProcessor → OpenAILLMService → …`
를 타고 갑니다. 그런데 그 chain의 processor 하나가 이렇게 쓰여 있다고 합시다 — 이건 filtering
processor의 지극히 정상적인 모양입니다:

```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    await super().process_frame(frame, direction)
    if isinstance(frame, TextFrame):
        await self.push_frame(self._transform(frame), direction)
    elif isinstance(frame, (StartFrame, EndFrame, InterruptionFrame)):
        await self.push_frame(frame, direction)
```

여러분의 `RuleVerdictFrame`은 어느 branch에도 match하지 않습니다. `super().process_frame()`도
자기 여섯 개 중 아무것도 match하지 않았습니다. 그래서 method는 아무것도 하지 않고 return합니다.
frame은 **사라집니다** — queue되지도, error도, log도 없이. exception 없음, warning 없음, metric
없음. 그 frame을 위해 쓴 downstream guard는 그것을 영영 못 보고, 실패는 "rule이 가끔 안 걸려요"
로 나타납니다. 재현 불가능한 tele-sales 통화에서 최악의 버그 모양입니다.

그게 row cost이고, 577이라는 숫자가 중요한 이유입니다: 136개 파일에 걸친 577개의
`isinstance(frame, ...)` site는, 어떤 저자가 한 번, 어떤 frame을 forward할지 결정한 577개의
장소입니다. 그 전부가 여러분의 frame이 존재하기 **전에** 쓰였습니다. 그 전부가 잠재적인
silent drop입니다.

두 operation을 정직하게 비교하면:

| Operation | 변경 파일 | 기존 code 건드림 | 틀렸을 때 failure mode |
|---|---|---|---|
| processor 추가 (column) | 새 파일 1개 | 없음 | 시끄러움 — pipeline이 조용해지고, test call 한 번에 알아챔 |
| frame 추가 (row) | 새 파일 1개 | *필수로는* 없음 | 조용함 — 알 수 없는 hop에서 frame이 사라짐 |

이 뒤틀린 symmetry를 보세요: row를 추가하는 데 아무것도 건드릴 필요가 없다는 것이 정확히 그것이
위험한 이유입니다. compiler는 136개 파일 중 어느 것을 update했어야 하는지 알려주지 않습니다.
sum type에는 실패할 exhaustiveness check가 없기 때문입니다. (그 생각을 붙잡아 두세요 —
[[ch-03/read]]는 정반대로 베팅해서 exhaustiveness를 얻고 대신 다른 것을 내준 design에 대한
chapter입니다.)

> 💡 **쉬운 설명 — Expression Problem을 2×2 표로.**
> 세로축을 frame type, 가로축을 processor라고 두고 각 cell이 "이 processor는 이 frame을
> 어떻게 다루는가"라고 합시다.
> - **새 column(processor)** = 표에 세로줄 하나 추가. 새 파일 하나에 다 채워 넣을 수 있음.
> - **새 row(frame)** = 표에 가로줄 하나 추가. 그 줄의 cell들은 **577개의 서로 다른 파일**에
>   흩어져 있음. 그런데 Pipecat은 빈 cell을 "아무것도 안 함"으로 기본 채우기 때문에, 컴파일도
>   되고 실행도 되고, 그냥 frame이 없어질 뿐입니다.
>
> OOP(class hierarchy)는 column이 싸고 row가 비쌉니다. functional style(closed union +
> pattern match)은 정반대입니다. 어느 쪽도 공짜가 아니라는 것이 Wadler의 요점입니다.

---

## 9. Pipecat은 column을 골랐고, streaming framework에는 그게 옳은 축이다

이 선택은 측정 가능한 근거 하나로 방어됩니다: **실제로 어느 축이 자라는가.**

Frame 어휘는 *media* 어휘입니다 — audio, text, image, LLM context, transcription, DTMF.
그 집합은 10년간 안정적이었고 앞으로 10년도 안정적일 것입니다. Service 어휘는 *vendor*
어휘이고, vendor는 매주 도착합니다. 이 commit에서 62개의 service directory가 있고, README는
그 의도를 한 단어로 말합니다:

**`README.md:23-29`**

```
## 🧠 Why Pipecat?

- **Voice-first**: Integrates speech recognition, text-to-speech, and conversation handling
- **Pluggable**: Supports many AI services and tools
- **Composable Pipelines**: Build complex behavior from modular components
- **Multi-Agent Ready**: Each pipeline is an agent. Compose them with handoff, parallel fan-out, sidecar workers, or distributed deployments
- **Real-Time**: Ultra-low latency interaction with different transports (e.g. WebSockets or WebRTC)
```

"Pluggable"과 "Composable Pipelines"는 둘 다 column-axis 주장입니다. frame 어휘를 확장하는
것에 대한 bullet은 없습니다.

governance position이 이걸 더 날카롭게 만듭니다. [[pipecat-design-philosophy]]에 따르면 repo
전체에서 유일한 명시적 거절은 `COMMUNITY_INTEGRATIONS.md:9-11`입니다: *"**What we don't
do:** The Pipecat team does not code review, test, or maintain community integrations."*
core team은 column 축을 너무 세게 최적화한 나머지 column 자체를 다른 사람들에게 넘겨버립니다.
검토되지 않은 third-party column을 받아들이려는 framework는 *반드시* column을 싸게 만들어야
하고, column 저자가 row 집합 전체를 이해하도록 요구해서는 안 됩니다.

그리고 `AGENTS.md`가 실제로 명시하는 유일한 direction rule (`:207`)은 column 쪽 contract입니다:
*"By default, all frames should be pushed in the direction they came."* "모든 frame을 handle해라"가
아니라, 그냥 "forward하는 것의 방향을 뒤집지 마라"입니다. framework가 column 저자에게 요구하는
것이 적은 이유는 바로 column 저자가 많을 것을 기대하기 때문입니다.

**Extension move, boson을 위한 것.** 여러분의 growth vector는 vendor가 아닙니다. 앞으로 1년간
boson은 rule check, stage, script step, compliance verdict를 추가할 것입니다 — 오늘 13개의
live `@check` ([[boson-layers-rules]]), 9개의 registered stage ([[boson-stage-machine]]).
그 각각이 Pipecat의 어느 축에 mapping되는지 자문하세요. rule check는 *완결된 utterance 위의
logic*입니다 — 그건 column입니다. stage는 *prompt + tool + legal successor의 묶음*입니다 —
그건 frame이 아예 아니고, state입니다 (section 12). 일주일에 네 번째로 새 frame class에 손이
가고 있다면, 여러분은 row를 일부러 비싸게 매긴 framework의 row 축을 키우려 하고 있는 것입니다.
그게 check입니다.

---

## 10. Ossification, 그리고 Pipecat 자신도 거의 안 쓰는 escape hatch

Akhshabi & Dovrolis의 발견은 waist가 ossify한다는 것, 그리고 그것이 *feature*라는 것입니다:
위아래 layer가 자유롭게 innovate할 수 있는 것은 바로 waist가 움직일 수 없기 때문입니다.
Pipecat은 그 양쪽 절반을 다 보여줍니다.

**waist는 얼어 있습니다.** `Frame`의 일곱 field (section 3)는 사실상 이제 변경 불가능합니다.
119개의 concrete class 전부가 그것들을 상속하고, tree의 모든 `__post_init__`이 그것들에
의존하며, 여덟 번째를 추가하면 모든 것의 constructor semantics를 건드리게 됩니다. 그래서 per-frame
side-channel data가 필요해졌을 때, 답이 새 base field일 수는 없습니다.

**선언된 escape hatch는 `metadata: dict[str, Any]`입니다** — untyped, 모든 processor에게 열려
있고, type system에 보이지 않습니다. 그것이 문헌이 예측하는 ossification 압력 밸브입니다.

**정직한 발견 #3: Pipecat은 그것을 한 번 쓰고 한 번도 읽지 않습니다.** grep했습니다:

```
$ grep -rn --include='*.py' "\.metadata\[" src/pipecat/ | wc -l
0
```

`src/pipecat/` 전체에서 유일한 frame-metadata assignment:

**`src/pipecat/pipeline/worker.py:1226` 및 `:1457-1464`**

```python
        start_frame.metadata = self._create_start_metadata()
```

```python
    def _create_start_metadata(self) -> dict[str, Any]:
        """Build and return start metadata including user-provided values."""
        start_metadata = {}

        # Update with user provided metadata.
        start_metadata.update(self._params.start_metadata)

        return start_metadata
```

write 한 번, `StartFrame` 위에, `WorkerParams.start_metadata`로부터 채워짐. core 어디에도 read가
없습니다. 그러니 `frame.metadata`는 **provisioned되었지만 exercise되지 않은** convention입니다:
모든 frame에 실재하는 dict이고 넣은 것을 실어 나르긴 하겠지만, Pipecat 안의 어떤 것도 그것을
propagate하지 않고, frame 경계를 넘어 merge하지 않으며, log하지도 않습니다. boson에서 correlation
ID용으로 쓴다면 양쪽 끝을 다 소유해야 합니다 — write, read, 그리고 옛 frame으로부터 새 frame을
만드는 모든 transformation을 가로지르는 propagation까지. section 12에서 그것을 쓰라는 조언은
여전히 유효합니다. caveat은 여러분이 이 pipeline의 첫 번째 진지한 사용자라는 것입니다.

**waist는 넓혀지는 게 아니라 지금 적극적으로 좁혀지고 있습니다.** 이것이 Beck 해석이 제가
덧씌운 것이 아니라는 가장 강한 증거입니다 — maintainer들이 지금 이 순간 frame에서 capability를
제거하고 있습니다. `StartFrame`은 7개의 configuration field를 실어 나르는데, 1.8.0 기준으로
일곱 전부가 deprecated입니다:

**`src/pipecat/frames/frames.py:923-995` (축약)**

```python
@dataclass
class StartFrame(SystemFrame):
    """Initial frame to start pipeline processing.

    This is the first frame that should be pushed down a pipeline to
    initialize all processors with their configuration parameters.

    Parameters:
        audio_in_sample_rate: Input audio sample rate in Hz.

            .. deprecated:: 1.8.0
                Read ``audio_in_sample_rate`` in ``FrameProcessorSetup.setup()`` instead.
                Will be removed in 2.0.0.
    ...
    """

    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_metrics: bool = False
    enable_tracing: bool = False
    enable_usage_metrics: bool = False
    report_only_initial_ttfb: bool = False
    tracing_context: TracingContext | None = None

    def __getattribute__(self, name: str) -> Any:
        # Reads warn, writes don't: assignment goes through ``__setattr__``. The None
        # guard is for ``tracing_context``, the only field whose default is None and
        # so can't be told apart from unset.
        if name in _START_FRAME_DEPRECATED_FIELDS:
            value = object.__getattribute__(self, name)
            if value is not None:
                warn_deprecated_read(
                    f"`StartFrame.{name}` is deprecated since 1.8.0, "
                    f"read `{name}` in `FrameProcessorSetup.setup()` instead. "
                    "Will be removed in 2.0.0."
                )
            return value
        return object.__getattribute__(self, name)
```

changelog fragment는 그 의도를 hedge 없이 말합니다:

**`changelog/5316.deprecated.md`**

```
- Deprecated `StartFrame.audio_in_sample_rate`, `StartFrame.audio_out_sample_rate`,
  `StartFrame.enable_metrics`, `StartFrame.enable_tracing`, `StartFrame.enable_usage_metrics`,
  `StartFrame.report_only_initial_ttfb` and `StartFrame.tracing_context`, which will be
  removed in 2.0.0. Read the same values from `FrameProcessorSetup` in `setup()` instead.
  The fields still carry the pipeline's configuration, so a processor that reads one keeps
  working and emits a `DeprecationWarning`, once per call site.
```

configuration은 대신 어디로 갈까요? **frame이 아닌** 평범한 dataclass로 갑니다:

**`src/pipecat/processors/frame_processor.py:76-115` (축약)**

```python
class FrameProcessorSetup:
    """Configuration parameters for frame processor initialization.
    ...
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    pipeline_worker: PipelineWorker
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_metrics: bool = False
    enable_tracing: bool = False
    enable_usage_metrics: bool = False
    observer: BaseObserver | None = None
    report_only_initial_ttfb: bool = False
    tracing_context: TracingContext | None = None
```

강제되고 있는 규칙: **frame은 data를 나르고, setup은 configuration을 나른다.** 이것은
`AGENTS.md:143`과 운을 맞춥니다. 거기서는 `@dataclass`(frame — "high-frequency, no validation
needed")와 Pydantic `BaseModel`(service param, transport/VAD param, serializer param)을
나눕니다.

section 12를 위해 이걸 적어두세요. 가장 흔한 migration 실수이고 boson은 그 실수를 저지르기 좋은
위치에 있습니다: **port가 per-turn configuration을 frame payload에 쑤셔 넣는다면, 여러분은
maintainer들이 지금 적극적으로 당기고 있는 방향의 반대로 밀고 있는 것입니다.** stage prompt,
tool list, rule threshold는 configuration입니다. 그것들은 manager object나
`FrameProcessorSetup`에 속하고, service에는 settings frame을 통해 도달합니다 — bespoke frame
위의 bespoke payload로가 아니라.

*(완결성을 위한 한 줄짜리 side note: Pipecat은 cross-worker communication을 위해 두 번째,
평행한 hourglass를 돌립니다 — `bus/messages.py`가 `BusMessage`를 정의하고 그것을
`BusDataMessage` / `BusSystemMessage`로 나눕니다. 다른 waist에서의 같은 priority split이고,
`BusFrameMessage(BusDataMessage)`는 문자 그대로 Frame-over-Bus, 하나의 codebase 안에서의
IP-over-everything 수법입니다. [[bus-and-extensions]] 참조. 이 chapter의 주제는 아닙니다.)*

---

## 11. 부하를 지탱하는 선례: `flows/`는 정확히 2개의 frame을 추가했다

위의 모든 것은 진단입니다. 여기가 처방이고, 이건 제 의견이 아닙니다 — Pipecat 자신의 팀이
boson과 정확히 같은 문제를 마주했을 때 실제로 한 일입니다.

`src/pipecat/flows/`는 conversation state machine입니다: node, transition, action, 그리고
pipeline을 구동하는 manager. 구조적으로 그건 boson의 stage machine 더하기 action 어휘입니다.
이 repo에서 row 축을 키울 면허를 가진 subsystem이 있다면 바로 이것입니다.

**그것은 2개의 frame을 정의했습니다. 둘 다 `ControlFrame`. 둘 다 `frames.py` 밖에.**

**`src/pipecat/flows/actions.py:49-66`**

```python
@dataclass
class FunctionActionFrame(ControlFrame):
    """Frame containing a function action to be executed.

    Parameters:
        action: Action configuration dictionary.
        function: Function handler to execute.
    """

    action: dict
    function: FlowActionHandler


@dataclass
class ActionFinishedFrame(ControlFrame):
    """Frame indicating that an action has completed execution."""

    pass
```

class 2개, 정의 12줄. 그리고 이들이 존재하는 이유는 하나의 구체적인 것 때문입니다: action
실행에는 **speech에 대한 in-band ordering**이 필요합니다. `function` action은 그 주변 TTS
output에 대해 알려진 지점에서 실행되어야 하고, Pipecat에서 "stream 안의 알려진 지점에서"를
표현하는 유일한 방법은 stream 안의 frame이 되는 것입니다. `ActionFinishedFrame`이 존재하는
이유는, downstream observer가 handler가 return했을 때가 아니라 effect가 실제로 착륙했을 때
ongoing-action counter를 감소시킬 수 있도록 하기 위해서입니다 ([[flows-actions]]).

**나머지 전부는 기존 frame을 타고 갑니다.** `FlowManager`는 이미 존재하는 frame을 queue해서
pipeline을 구동합니다:

**`src/pipecat/flows/manager.py:764-770`**

```python
            frames = []

            # New path: role_message as LLM system instruction (persists until changed)
            if role_message:
                frames.append(
                    LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=role_message))
                )
```

**`src/pipecat/flows/manager.py:827-841`**

```python
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
```

system prompt 변경 → `LLMUpdateSettingsFrame`. tool 변경 → `LLMSetToolsFrame`. message 추가
또는 교체 → `LLMMessagesAppendFrame` / `LLMMessagesUpdateFrame`. inference trigger →
`LLMRunFrame()` (`manager.py:709`). 고정된 대사 말하기 → `TTSSpeakFrame` (`actions.py`의
`tts_say` handler). 이 전부가 기존의 모든 LLM/TTS service가 이미 올바르게 처리하는 기존
frame입니다.

**그리고 state는 pipeline 밖에 완전히 남아 있습니다.**

**`src/pipecat/flows/manager.py:147-149`**

```python
        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None
```

current node는 `str`입니다. 공유 state는 평범한 `dict`입니다. `NodeConfig`, `FlowResult`,
`ActionConfig`는 `flows/types.py`의 `TypedDict`입니다(line 182, 41, 112). **그중 어느 것도
frame이 아닙니다.** 한 component만 읽는 state는 universal waist에 있을 이유가 없습니다.

이 패턴은 section 7의 31개 out-of-tree frame 전부에 걸쳐 성립합니다 — RTVI는 9개를
`processors/frameworks/rtvi/frames.py`에, Daily는 8개를 자기 `transport.py`에, OpenClaw는
6개를 `services/openclaw/frames.py`에 넣었습니다. 아무도 `frames.py`를 키우지 않았습니다.
`frames.py`는 **공유 media 어휘**를 위해 예약되어 있고, 모든 subsystem은 자기 사적 어휘를
사적으로 유지합니다.

*(`flows/`가 실제로 어떻게 동작하는지 — `set_node`, transition mechanics, pre/post-action
ordering, 그리고 transition이 LLM function call일 필요가 없다는 발견 — 은 [[ch-10/read]]의
주제입니다. 이 section은 거기서 정확히 하나만 가져옵니다: frame budget.)*

---

## 12. boson을 위한 규칙, 그리고 3-way test

규칙을 한 번에 말하면:

> **`ControlFrame`을 subclass해라. `DataFrame`은 건드리지 마라. class는 `boson/frames.py`에
> 넣고 `frames.py`에는 절대 넣지 마라. 총 2~4개로 budget을 잡아라.**

이제 추론입니다. 예외를 발견했을 때 방어할 수 있도록.

**왜 `DataFrame`이 아닌가?** `DataFrame`은 conversation의 *content*입니다 — 고객이 듣고 말하는
audio와 text. stage transition은 content가 아닙니다. boson signal을 `DataFrame` branch에 넣는
것은 runtime에서 아무것도 사주지 않고(아무것도 그걸 dispatch하지 않습니다, section 5), 그 뒤로
그 frame을 읽는 모든 사람에게 잘못된 label을 붙입니다.

**왜 `SystemFrame`이 아닌가?** `SystemFrame`은 "queue를 뛰어넘고 barge-in에서 살아남는다"를
의미하고, 그건 boson signal이 원하는 것이 거의 아니기 때문입니다. turn N에 대한 rule verdict가
turn N의 audio보다 앞서 뛰면, 그것이 대상으로 하는 것보다 먼저 도착합니다. 더 나쁜 것은,
`SystemFrame`은 **input task 위에서 inline으로** 실행되므로(`frame_processor.py:1304`), 그것이
trigger하는 모든 작업이 processor의 frame intake를 막는다는 점입니다. `SystemFrame`은 그 signal이
진짜로 barge-in급 event일 때만 고르세요.

**왜 `ControlFrame`인가?** rule verdict나 stage signal이 필요로 하는 것을 정확히 줍니다:
주변 speech에 대해 ordered이고, 고객이 interrupt하면 purge됨 — 그리고 그건 correct합니다.
무효화된 utterance에 대한 verdict는 그것과 함께 죽어야 하니까요. barge-in을 뚫고서라도 반드시
전달되어야 하는 특정 signal에 대해서만 `UninterruptibleFrame` (section 5)을 추가하세요. boson에서
그건 짧은 목록입니다: DNC registration과 consent recording이 그럴듯한 후보입니다. 법적 결과가
있고, 하나를 조용히 떨구는 것은 UX glitch가 아니라 compliance failure이기 때문입니다.

### 3-way test

모든 boson 개념을 `Pipeline([...])` 근처에 가져가기 전에 이것을 통과시키세요.

**(a) STATE인가? → frame이 아니다.**

현재 stage, 누적된 rule 결과, script cursor, sentiment window, round counter. 이것들은
평범한 dataclass를 담은 평범한 `BosonStageManager` object에 살고, `FlowManager._state`와
`FlowManager._current_node` (`manager.py:147-149`)를 그대로 반영합니다. test는:
*둘 이상의 component가 이것을 stream의 일부로서, 순서대로 읽어야 하는가?* 아니라면 — 그리고
state에 대해서는 답이 거의 항상 아니오입니다 — 그건 manager의 field이지 frame이 아닙니다.

구체적으로 [[boson-stage-machine]]에서: `session.active_stage`는 string입니다. `StageMachine`은
명시적으로 stateless이고 공유됩니다. `MAX_ROUNDS`와 `FALLBACK_TRANSITIONS`는 dict입니다.
그 전부가 manager object로 그대로 port되고 waist를 0번 건드립니다.

**(b) EFFECT인가? → 기존 frame.**

boson 어휘의 대부분이 여기에 착륙하고, 깔끔하게 착륙합니다. mapping:

| boson 개념 | 이미 존재하는 Pipecat frame | Line |
|---|---|---|
| stage transition이 system prompt를 바꿈 | `LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=...))` | `frames.py:2283` |
| stage transition이 보이는 tool을 바꿈 | `LLMSetToolsFrame(tools=[...])` | `frames.py:694` |
| `Respond(text)` — 대본 그대로의 대사 | `TTSSpeakFrame(text=..., append_to_context=...)` | `frames.py:795` |
| `Inject(content)` — rule verdict를 context에 | `LLMMessagesAppendFrame(messages=[...])` | `frames.py:645` |
| stage 진입 시 context reset | `LLMMessagesUpdateFrame(messages=[...])` | `frames.py:661` |
| transition 후 inference trigger | `LLMRunFrame()` | `frames.py:634` |
| 통화 중 한국어 TTS voice 교체 | `TTSUpdateSettingsFrame(delta=TTSSettings(...))` | `frames.py:2290` |

7개의 boson behaviour, 0개의 새 frame class, 그리고 tree의 모든 기존 LLM/TTS service가 이미
일곱 전부를 올바르게 처리합니다. 그게 "full composability"가 현금화되는 방식입니다.

이것이 무엇을 대가로 하는지 정직하게 짚습니다 — 부드럽게 말하지 않겠다고 약속했으니까요:
boson script engine ([[boson-script-engine]])은 `Respond(step.text)`를 **LLM replacement**로
반환합니다 — model은 그 turn을 아예 보지 않습니다. 그걸 `TTSSpeakFrame`으로 표현하면 발화는
되지만, "이 turn에 대해 LLM을 억제한다"는 semantics는 frame이 실어 나르지 않습니다. 그건
`LLMRunFrame`을 emit하지 *않음*으로써 processor가 내려야 하는 control 결정입니다. frame은
mapping되지만 control inversion은 mapping되지 않습니다. 그 gap은 실재하고, 이 chapter가 아니라
[[ch-12/read]]의 문제입니다.

**(c) 진짜로 NEW한 in-band signal인가? → `boson/frames.py`, `ControlFrame` subclass.**

기준을 정확히 말하면: **signal이 speech에 대해 ordered되어야 하고, 그 consumer가 그것을 생산한
component가 아닌 processor여야 한다.** 두 절 모두. 생산자와 소비자가 같은 object라면 method를
호출하세요 — 나갔다가 같은 processor로 돌아오는 frame은 매우 비싼 function call입니다.

후보들을 통과시켜 봅시다:

| 후보 | speech에 대해 ordered? | 다른 consumer? | Verdict |
|---|---|---|---|
| `RuleViolationFrame` — downstream guard가 TTS emit 전에 verdict를 봐야 함 | yes | yes (guard ≠ rule processor) | **class를 얻는다.** `ControlFrame`. |
| `StageEnteredFrame` — observability/analytics가 stream 안의 stage 경계를 원함 | yes | yes (observer ≠ manager) | **class를 얻는다**, 단 실제로 그 observer를 만들 때만. 아니면 state. |
| `ScriptStepFrame` — script engine이 자기 자신에게 지금 몇 번째 step인지 알려줌 | no | no (같은 object) | **기각.** manager field. |
| `StageTransitionFrame` — "`purchase`로 옮겨 주세요" | — | no (manager가 결정하고 적용까지 함) | **기각.** manager에 대한 method call이고, 그 *effect*가 (b)의 frame들. |
| `SentimentScoreFrame` — 5-turn negative window | no | no | **기각.** manager field. |
| `ConsentCheckpointFrame` — 법적으로 요구되는 consent 확인 | yes | yes | **class를 얻는다**, 그리고 이것이 `ControlFrame, UninterruptibleFrame`이 필요한 바로 그 하나. |

6개 후보 중 2~3개 생존이고, budget 안에 들어옵니다. budget 자체는 **2–4**이고, flows의 2와
OpenClaw의 6에 맞춘 것입니다. Correlation ID와 stage tag는 `frame.metadata`를 타고 갑니다
(section 10의 caveat 적용 — Pipecat 안의 어떤 것도 그걸 읽지 않으므로, propagation은 여러분
소유입니다).

**그리고 failure condition을, 경고가 아니라 tripwire로 말하면:** 열 번째 frame class를 쓰고
있는 자신을 발견하면 멈추고 여러분이 무엇을 지었는지 읽으세요. 10개의 class는 여러분이 boson의
내부 protocol을 — `gateway/schemas/actions.py`의 8-verb `ActionType` union 더하기 stage와
script signal을 — Pipecat의 waist *안에* 다시 구현했다는 뜻입니다. 그 시점에서 여러분의 10개
type 각각은 section 8의 row cost에 노출됩니다: 그들을 모르는 577개의 `isinstance` site,
automatic pass-through 없음, 알 수 없는 hop에서의 silent drop. sum-type tax를 전액 내면서
migration의 이유였던 interchangeability는 하나도 못 받게 됩니다. 열 번째 frame은 code smell이
아닙니다. **design이 세 frame 전에 이미 잘못됐다는 signal**입니다.

**아무것도 쓰기 전에 해볼 만한 sanity check.** 오늘 boson의 protocol은 10개의 string입니다:
`VALID_CLIENT_TYPES = {"user_message", "partial_transcript", "interrupt", "get_history"}`와
`VALID_SERVER_TYPES = {"text_delta", "turn_end", "error", "interrupted", "stage_changed",
"history"}` ([[frame-taxonomy]]). 무엇을 발명하기 전에 먼저 mapping하세요:
`"interrupt"` → `InterruptionFrame` (`frames.py:1142`); `"partial_transcript"`는 진짜 구분인
`InterimTranscriptionFrame` (476) vs `TranscriptionFrame(finalized=...)` (450)로 갈라집니다;
`"text_delta"` → `LLMTextFrame` (343); `"turn_end"` → 누구의 turn이 끝났느냐에 따라
`UserStoppedSpeakingFrame` (1165) 또는 `TTSStoppedFrame` (2231); `"error"` → `ErrorFrame`
(1016). 10개 string 중 8개가 이미 집이 있습니다. 새로운 무언가의 후보는 `"stage_changed"`와
`"history"` 둘뿐이고 — 그중 `"history"`는 state입니다.

---

## 13. 이 chapter가 정리하지 않은 것

세 가지가 의도적으로 열려 있습니다. 닫힌 것으로 착각하지 마세요.

1. **boson의 rule layer가 Pipecat pipeline 안에서 veto를 표현할 수 있는가 자체.**
   `FrameProcessor`는 `push_frame()`을 호출하는 순간 행동합니다. boson의 `LayerPipeline`은
   모든 action을 stage해 두고 나중 layer의 `Filter`가 그 전부를 — 추가된 user message까지
   포함해 — 폐기할 수 있게 합니다 ([[boson-layers-rules]]). sum type은 two-phase commit에
   대해 할 말이 없습니다. 그건 [[ch-12/read]]의 문제이고, 알고 보면 latency 문제입니다.
2. **section 12에서 고른 frame들이 의미 있을 만큼 충분히 일찍 생산될 수 있는가.**
   rule verdict는 inference가 시작되기 *전에* 착륙해야만 쓸모가 있고, 그것을 보장하기 위해
   pipeline의 어디에 설 수 있는가는 type 질문이 아니라 runtime 질문입니다 — runtime은
   [[ch-04/read]], millisecond budget은 [[ch-11/read]].
3. **open sum type이 애초에 옳은 베팅인가.** 그건 법칙이 아니라 베팅입니다. 반대편 베팅은
   존재하고, 출시되어 있고, 여러분이 직접 썼습니다.

---

## 다음 챕터로

이 chapter는 세 가지를 앞으로 넘깁니다.

**[[ch-03/read]]로, 즉시:** **open sum type**이라는 구절 — 가격표를 붙인 채로. 누구나 core를
건드리지 않고 `Frame` subclass를 추가할 수 있고, 그 대가를 exhaustiveness check도 automatic
pass-through도 없는 577개 `isinstance` site에 걸쳐 지불합니다. ch-03은 정확히 반대로 베팅한
design을 특징짓습니다: `voice-chat-dev` branch의 `packages/realtime_voice/`, data 단위가
**closed union**인 설계 —
`VoiceRuntimeEvent = VADEvent | ASREvent | AgentTextDelta | VoiceEvent` ([[rtv-pipeline-session]]).
두 개의 design, Wadler에 대한 두 개의 답, 서로 다른 저자 인구를 위한 최적화. ch-03은 각각이
무엇을 하는지 진술합니다. 표를 던지지 않고, 이 chapter도 마찬가지입니다.

**[[ch-10/read]]로:** `flows/`가 정확히 2개의 `ControlFrame`을 추가했고 `_current_node`를
`str`로 유지했다는 사실. 이 chapter는 그것을 budget으로 썼습니다. ch-10은 그 package를 실제
정체대로 읽습니다 — 의도적으로 pipeline *밖에* 살면서 `queue_frames`를 통해 그것을 구동하는
state machine.

**[[ch-12/read]]로:** 3-way test 그 자체, 더하기 2–4 budget과 10에서의 tripwire. boson의 rule
seam을 설계하러 앉을 때 frame 질문은 이미 답이 나와 있을 것이고, 남는 것은 processor가 어디에
서느냐와 그 position이 millisecond로 얼마인가입니다.

가져갈 한 문장: **waist가 narrow한 이유는 그것이 아무것도 약속하지 않기 때문이고, 거기에
더하는 모든 약속은 136개 파일이 읽지 않은 약속이다.**
