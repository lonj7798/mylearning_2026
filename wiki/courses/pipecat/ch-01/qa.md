# ch-01 Q&A — Pipes and Filters

Reading-phase questions for [[read]] (pipecat course). Kernel answers only;
full traces live in `read.md`. Append-only across cycles.

---

## Q1. `FrameProcessor`가 뭐고, `process_frame`이랑 `push_frame`은 뭐지?

**Kernel.** `FrameProcessor`는 파이프의 한 칸이다. `process_frame`은 그 칸의 **입구**
(들어온 frame을 받는다), `push_frame`은 **출구**(다음 칸으로 내보낸다). 읽는 함수 하나 +
쓰는 함수 하나 — 이 두 개짜리 signature가 모든 processor에 동일하다는 사실이 ch-01이
말하는 uniform interface의 전부이고, 아무 자리에나 splice할 수 있는 이유다.

| | 시그니처 |
|---|---|
| 입구 | `async def process_frame(self, frame: Frame, direction: FrameDirection)` — `frame_processor.py:820` |
| 출구 | `async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM)` — `:1004` |
| 방향 | `DOWNSTREAM = 1` / `UPSTREAM = 2` — `:60-69` |
| 배선 | `link()`이 `self._next = processor; processor._prev = self` — `:677-678` |

**비자명했던 부분 — push는 호출이 아니라 큐잉이다.**
`push_frame`(:1013) → `__internal_push_frame`(:1160) → `self._next.queue_frame(...)`(:1182)
→ `self.__input_queue.put(...)`(:721). 다음 processor의 `process_frame`을 직접 부르지
않고 **그 processor의 큐에 넣는다.** processor마다 자기 큐와 자기 task를 갖는다.

→ 그래서 지연이 코드 속도가 아니라 **queue depth**의 함수가 된다. 이것이 `InterruptionFrame`이
`SystemFrame`이어서 큐를 건너뛰어야만 하는 이유의 뿌리다. 큐/task 해부는 [[ch-04/read]],
out-of-band 우선순위는 [[ch-03/read]].

**함정.** base `process_frame`은 비어 있지 않다(`:837-847`): `StartFrame` → `__start`,
`InterruptionFrame` → `_start_interruption`, `CancelFrame` → `__cancel`, pause/resume.
따라서 custom processor의 **첫 줄은 반드시** `await super().process_frame(frame, direction)`.
빼먹으면 에러 없이 조용히 죽는다 — `StartFrame`이 `__start`에 도달 못 해 process task가
아예 안 생기고, barge-in도 그 지점에서 멈춘다. 전체 추적은 `read.md` §7.2–7.3.

**열어둔 probe.** `push_frame`의 기본 방향이 `DOWNSTREAM`인데, **UPSTREAM으로 밀어야만
하는 frame은 무엇인가** — 파이프 뒤쪽에서 앞쪽으로 되돌아가야 하는 정보의 정체.
[[ch-03/read]]에서 out-of-band의 두 번째 축으로 다룬다.

---

## Q2. filter는 `return`이 아니라 `push`로 쓴다는 게 무슨 말이지?

**Kernel.** `return`은 **1:1 계약**이고 `push`는 **0:N 계약**이다. `process_frame`은 값을
반환하지 않는다 — 정의 131개 중 반환값을 쓰는 곳이 없다. 출력은 오직 `push_frame` 호출로만
생긴다. 따라서 **버린다 = `push_frame`을 호출하지 않는다.**

`aggregators/sentence.py:40-63` 한 메서드에 네 경우가 모두 있다:

| | 상황 | 입력 → 출력 | 코드 |
|---|---|---|---|
| ① | `InterimTranscriptionFrame` | 1 → **0** (버림) | `:50-51` `return`, push 없음 |
| ② | 문장 중간 `TextFrame` | **N → 1** (합침) | `:53-57` |
| ③ | `EndFrame` | **1 → 2** | `:58-61` push 두 번 |
| ④ | 나머지 | 1 → 1 | `:63` |

`return f` 시그니처로는 ①②③ 중 어느 것도 표현할 수 없다. ①의 `return`은 *값 반환*이 아니라
파이썬 함수 탈출이며, push를 안 했으므로 frame이 소멸한 것이다.

push여야 하는 이유 둘 더: **방향** — `push_frame(f, UPSTREAM)`은 뒤로 보내는데 return에는
방향을 담을 자리가 없다. **시간** — TTS는 텍스트 한 덩이를 받아 오디오를 수 초에 걸쳐 나눠
push한다. return이면 완료까지 기다려야 하므로 streaming이 아니라 batch가 되고 TTFA가 죽는다.

대조: `IdentityFilter`는 무조건 한 번 push(`identity_filter.py:44-45`) → 항등원.
`FrameFilter`는 조건부 push(`frame_filter.py:50-53`) → 조건 불일치 시 아무 일도 안 일어남.

---

## Q3. `Frame`을 주고받으면서 signature를 동일하게 끌고 가는 건가?

**Kernel.** 그렇다. 그것이 narrow waist다. `Frame`이 base class이고 모든 구체 frame이 그
서브클래스이므로 `process_frame(self, frame: Frame, direction: FrameDirection)`의 **정적
타입은 절대 변하지 않는다** — 오디오든 transcript든 LLM 토큰이든 `Frame` 하나. 측정:
`process_frame` 정의 **131개** 중 **101개**가 글자 그대로 동일한 시그니처.

**대가 = dispatch tax.** 정적 타입이 안 변하니 실제로 뭐가 왔는지는 런타임에 알아내야 한다:
`isinstance(frame, ...)`가 **577곳 / 136파일**. 타입 시스템이 할 수 있는 일을 매 processor가
`if isinstance` 사다리로 직접 한다. 그리고 frame 타입을 하나 추가하면 그것을 무시해야 하는
모든 processor가 조용히 무시해줘야 하며, 그것이 `super().process_frame()` 계약이 강제되는
이유다 (→ [[Q1]]의 함정).

[[ch-02/read]]가 이를 **expression problem**으로 정식화한다: processor 추가는 싸고 frame
타입 추가는 비싸다. sum type은 한 축을 최적화하고 다른 축에 세금을 매긴다.

**열어둔 probe.** Q2의 ②에서 `self._aggregation`에 반쯤 모인 문장이 남아 있는 동안
`InterruptionFrame`이 도착하면 그 상태는 어떻게 되는가. barge-in이 어려운 진짜 이유이며
[[ch-08/read]]의 주제.

---

## Q4. UPSTREAM / DOWNSTREAM은 왜 나뉘어 있고, UPSTREAM은 언제 일어나나?

**Kernel.** 파이프라인은 **선(line)이지 고리(loop)가 아니다.** wraparound가 없으므로 뒤쪽
processor가 앞쪽에 무언가 알리려면 길이 UPSTREAM 하나뿐이다. `link()`가 `_next`와 `_prev`를
**둘 다** 꽂는 이유(`frame_processor.py:677-678`)이고, 방향은 `__internal_push_frame`에서
어느 포인터를 따를지 고르는 것이다(`:1170` DOWNSTREAM→`_next` / `:1183` UPSTREAM→`_prev`).

**의미가 다르다.** DOWNSTREAM = **데이터**(오디오→transcript→토큰→오디오, 일이 진행되는
방향). UPSTREAM = **제어와 보고**(에러·종료·준비완료·갱신된 context·설정변경).

repo 전체 UPSTREAM 사용처 49곳의 분류:

| | 용도 | 대표 위치 |
|---|---|---|
| ① | **에러 보고** — 하류에 알려봐야 소용없고 파이프라인 소유자가 알아야 함 | `frame_processor.py:1002` (`push_error_frame`) |
| ② | **종료·취소** — 파이프 끝에 닿아야 "정말 끝"임을 알 수 있고, 그걸 worker에게 되알림 | `worker.py:1360-1369` (End/Stop/Cancel/Interruption WorkerFrame) |
| ③ | **준비 완료** | `base_output.py:214` `OutputTransportReadyFrame` |
| ④ | **context 되돌리기** ★ | `llm_response_universal.py:1706, 1711, 1716, 1721` |
| ⑤ | **설정 업데이트** | `ivr_navigator.py:152, 276, 280` (LLM/VAD 설정), `xai/realtime/llm.py:882` (DTMF) |
| ⑥ | **양방향 broadcast** | `frame_processor.py:1054, 1089`; `Frame.broadcast_sibling_id`(`frames.py:75-77`)가 짝 id를 보관 |

**④가 이 코스에서 가장 중요하다.** `LLMAssistantAggregator`(클래스 `:1384`)는 파이프라인
맨 끝, `transport.output()` **뒤**에 앉는데 LLM은 **중간**에 있다. 그래서 메시지가 추가되고
`run_llm`이면 갱신된 context를 UPSTREAM으로 거슬러 올려 LLM에 닿게 한다.

→ 여기서 한 바퀴가 완성된다: [[ch-10/read]]의 `FlowManager._set_node`는 pipeline **머리**에
`LLMMessagesAppendFrame`을 밀어넣고(`flows/manager.py:841`), 그것이 하류로 흘러 **꼬리**의
assistant aggregator에 닿고, aggregator가 context를 갱신한 뒤 **UPSTREAM으로 되밀어** 중간의
LLM에 도달한다. **노드 전환 한 번이 파이프를 한 바퀴 돈다.**

이것이 [[ch-03/read]] out-of-band의 **두 번째 축**이다(첫 번째는 `SystemFrame`의 큐 건너뛰기
= 우선순위, 두 번째가 이 방향). 둘 다 순수 Pipes-and-Filters로는 불가능한 일이고, 둘 다
그래서 존재한다.

**열어둔 probe.** UPSTREAM은 `_prev`를 따라가므로 **파이프 내 위치에 종속**된다. 그러면
[[ch-12/read]]에서 rule processor를 어디에 놓느냐가 "그 rule이 무엇을 되돌려 보낼 수 있느냐"
까지 결정한다. rule이 LLM 생성을 **막아야** 한다면 LLM보다 앞이어야 하나 뒤여야 하나 —
뒤에 놓고 UPSTREAM으로 막으면 안 되는 이유는 무엇인가.

---

## Q5. UPSTREAM 대신 그냥 session state에 전부 저장하면 안 되나?

**Kernel.** 학습자의 직관이 맞다 — **pipecat은 이미 그렇게 하고 있다.**
`LLMContextAggregatorPair.__init__`(`llm_response_universal.py:2254`)은 `context: LLMContext`
**하나**를 받아 `LLMUserAggregator(context, ...)`(:2296)와 `LLMAssistantAggregator(context, ...)`
(:2309)에 **같은 객체**를 넘긴다. docstring이 명시: *"The shared context guarantees the
announcement is added exactly once."*

그리고 UPSTREAM으로 미는 frame은 데이터를 나르지 않는다. `_get_context_frame()`(:481-487)은
`LLMContextFrame(context=self._context)` — 같은 객체를 감쌀 뿐이고, `LLMContextFrame`의
docstring은 스스로를 **"a signal to LLM services to ingest the provided context"**라 부른다
(`frames.py:551-561`).

> **frame이 나르는 것은 데이터가 아니라 "지금"이다.**

**공유 변수가 못 하는 네 가지:**

| | 공유 변수 | frame |
|---|---|---|
| 언제 | 변경을 알리지 않음 (polling/콜백 필요) | frame 자체가 이벤트 |
| 순서 | 쓰기 순서 미정의 | 큐에서 다른 frame과 한 줄로 정렬 |
| 가시성 | 아무도 못 봄 | 경로 위 모든 processor를 지나감 |
| 되돌리기 | 이미 쓴 것은 못 되돌림 | 큐에 있는 동안 `InterruptionFrame`에 폐기 가능 |

*순서*는 barge-in에서 결정적이다("context 갱신" vs "취소"의 순서가 결과를 바꾼다).
*가시성*은 [[ch-12/read]]의 seam 질문 그 자체다 — 중간의 rule processor는 지나가는 context
frame을 **가로채 막을 수 있지만**, 공유 변수 대입은 아무도 보지 못한다.
*관측*도 여기 딸려온다: 모든 push가 observer를 통과한다(`frame_processor.py:1173-1181`,
`FramePushed`). 공유 변수 대입은 흔적이 없고, [[ch-11/read]]의 observer plane이 전부 이 위에
서 있다.

**pipecat도 "그냥 공유"를 제공한다:** `src/pipecat/bus/`(1,051 L, `bus.py:96 subscribe`,
`:142 publish`)는 파이프에서 인접하지 않은 것들을 잇는 사이드채널이다. 저자들도 "전부
frame으로"라고 하지 않았다.

**설계 규칙:** 상태는 공유하되, **"지금"은 frame으로 말한다.**

**마이그레이션 급소.** 공유 객체 설계에는 대가가 있고 그것이 [[ch-09/read]]의 가장 날카로운
단절점이다: pipecat의 context는 **live list**(양쪽이 같은 객체를 직접 변형)인 반면 boson의
`ContextManager.get_messages()`는 **deepcopy**를 반환한다. 두 프레임워크가 정반대 결정을 했다.
학습자의 질문에 대한 답은 곧 "pipecat이 이미 내린 결정이며, 그 결정 때문에 boson을 얹기가
어려워졌다"이다.

**열어둔 probe.** boson의 `SharedLayerContext`도 공유 상태 설계다. rule layer를 옮길 때
**무엇이 frame이 되고 무엇이 공유로 남아야 하는가** — 위 4칸(언제/순서/가시성/되돌리기)을
기준으로 나누면 [[ch-12/read]]를 반쯤 미리 푸는 셈이다.

---

## Q6. interruption은 어떻게 대응하나? 현재 status가 어떤 Frame이냐에 따라 다른가?

**Kernel.** 아니다 — 상태는 frame에 있지 않다. 각 processor가 **자기 상태를 자기 안에** 들고
있고, `InterruptionFrame`은 "네 상태를 버려"라고 전원에게 방송하는 frame이며, 버리는 일은
각자가 한다. frame에 달려 있는 것은 **딱 하나뿐**이다(아래 ③).

**3층 메커니즘**

① **큐를 건너뛴다.** `InterruptionFrame(SystemFrame)`(`frames.py:1142`). processor마다 큐가
   둘이고, `__input_frame_task_handler`(`frame_processor.py:1304-1307`)가 갈라놓는다:
   `isinstance(frame, SystemFrame)`이면 그 자리에서 `__process_frame`, 아니면
   `__process_queue.put(...)`. 그래서 두 번째 큐에 오디오가 쌓여 있어도 기다리지 않는다.
② **양방향으로 퍼진다.** `broadcast_interruption()`(`:1017-1022`) =
   `__reset_process_task()` → `stop_all_metrics()` → `broadcast_frame(InterruptionFrame)`.
③ **각자 버린다.** base `process_frame`(`:839-841`)이 잡아 `_start_interruption()` 호출.

**질문에 대한 정확한 답 — frame에 달린 것은 이 분기 하나다** (`:1130-1150`):

```python
current_is_uninterruptible = isinstance(self.__process_current_frame, UninterruptibleFrame)
if current_is_uninterruptible:
    self.__reset_process_queue()        # 처리 중인 것은 살리고 큐만 비움
else:
    await self.__cancel_process_task()  # 통째로 취소
    self.__create_process_task()
```

`UninterruptibleFrame`을 다는 것들: `FunctionCallResultFrame`(`frames.py:770`),
`FunctionCallInProgressFrame`(:2164), `EndFrame`(:1899), `StopFrame`(:1923) 등.
**tool 결과가 interruption에 죽으면 안 되는 이유**는 명확하다 — tool은 이미 실행됐고(DB 쓰기,
API 호출), 결과 frame을 버리면 context에 "호출함"만 남고 "결과"가 없어 상태가 어긋난다.
취소되는 것은 **큐에 줄 서 있던 frame**이며, 이미 지나간 것은 되돌릴 수 없다([[Q5]]의
"push는 되돌릴 수 없다"가 여기서 물리적으로 드러난다).

**진짜 배울 지점 — 리셋은 자동이 아니다.** `_start_interruption`은 **task와 큐**만 정리하고
**파이썬 객체 필드는 건드리지 않는다.** 실측:

| processor | interruption 처리 | |
|---|---|---|
| `services/tts_service.py` | ✅ | `_streamed_text=""`, sequencer clear, word timestamp reset (`:1038-1043`) |
| `aggregators/llm_response_universal.py` | ✅ | `_handle_interruptions` → `reset()` (`:1723-1725`) |
| `aggregators/sentence.py` | ❌ 0회 언급 | `_aggregation`은 문장 완성(`:57`)과 생성자(`:38`)에서만 비워짐 |
| `aggregators/gated.py` / `dtmf_aggregator.py` / `audio/audio_buffer_processor.py` | ❌ 0회 | |

→ [[Q3]]에서 열어둔 probe의 답: **`SentenceAggregator`의 반쯤 모인 문장은 barge-in을
살아남는다.** 다만 실제로 물리지는 않는다 — 표준 voice bot 파이프라인에 없고
`examples/getting-started/04-sync-speech-and-image.py` 한 곳에서만 쓰인다. 패턴은 명확하다:
**barge-in 임계 경로의 컴포넌트는 모두 처리하고, 주변부는 하지 않는다.**

> **규칙: interruption은 "취소해도 되는 것"을 자동으로 취소한다. "내가 들고 있는 것"은
> 내가 버려야 한다.**

**열어둔 probe.** [[ch-12/read]]의 `BosonRuleProcessor`도 상태를 든다(발화 조각 누적,
rollback용 context 스냅샷). 그 리셋은 아무도 해주지 않으므로 `_handle_interruptions`를 직접
구현해야 한다. 더 미묘한 결정 하나: **rule 판정 결과는 `UninterruptibleFrame`이어야 하는가?**
tool 결과처럼 "이미 실행됐으니 버리면 안 되는" 성질인가, "취소돼도 되는" 성질인가.

---

## Q7. "`Pipeline`에 link되지 않습니다"가 무슨 말이지? (read_kor.md §2.4, L284)

**Kernel.** `link()`는 **`FrameProcessor`만 가진 메서드**다(`frame_processor.py:671`,
`_next`/`_prev`를 꽂는 두 줄). 그러므로 `FrameProcessor`를 상속하지 않는 클래스는 `link()`도
`_next`/`_prev`도 `push_frame`도 없고, **파이프에 꿰어질 수가 없다.** 이름만 `process_frame`일
뿐 파이프의 칸이 아니다.

해당 7개 hierarchy는 전부 `ABC` 또는 `BaseObject` 상속이다: `BaseAudioFilter`(ABC,
`audio/filters/base_audio_filter.py:18`), `BaseAudioMixer`(ABC, `:18`),
`VADController`(BaseObject, `audio/vad/vad_controller.py:31`),
`BaseUserTurnStartStrategy`/`BaseUserTurnStopStrategy`/`BaseUserMuteStrategy`(BaseObject,
`turns/user_start|user_stop|user_mute/...`), `LLMContextSummarizer`(BaseObject,
`processors/aggregators/llm_context_summarizer.py:57`).

**대신 소유자가 직접 호출한다:**
```python
await self._params.audio_in_filter.process_frame(frame)   # transports/base_input.py:253
result = await strategy.process_frame(frame)              # turns/user_turn_controller.py:204
```
`result = await ...` — **값을 받는다.** 실제 시그니처가
`async def process_frame(self, frame: Frame) -> ProcessFrameResult | None`
(`turns/user_start/base_user_turn_start_strategy.py:164`)이다.
**반환값의 존재 자체가 "파이프의 칸이 아니다"의 증거다** — 파이프의 칸은 [[Q2]]대로 return이
아니라 push로 쓴다.

**합성 층위가 둘이다:**

| 층위 | 꽂히는 것 | 통신 | 상속 |
|---|---|---|---|
| 파이프 | processor가 `link()`로 이어짐 | `push_frame` (0:N, 방향 있음) | `FrameProcessor` |
| 부품 | 전략·필터가 processor **안에** | 직접 호출 + **반환값** | `ABC` / `BaseObject` |

왜 나눴나 — VAD 전략 교체가 **파이프라인 구조 변경이면 안 되기** 때문이다. transport의 자리는
그대로 두고 그 안의 부품만 갈아끼운다. 즉 이들은 **stage가 아니라 plugin**이다.

**실무 함의.** `grep -rn "async def process_frame" src/` → 131개 중 101개만 파이프
인터페이스이고 **30개(약 1/4)는 남남**이다. 하필 그 7개 중 셋(`turns/*`)이 [[ch-06/read]]에서
가장 열심히 읽을 코드다. 챕터가 excerpt를 정정한 지점이기도 하다 — excerpt는 "three unrelated
classes"라 했으나 실제로는 7 hierarchy / 30 definition.

**열어둔 probe.** [[ch-12/read]]의 결론 "모든 rule layer를 하나의 processor로 합쳐라"는 곧
**layer들이 파이프의 칸이 아니라 한 processor 안의 부품이 된다**(위 표의 아래 행)는 뜻이다.
boson의 `@check` 함수들은 이미 반환값으로 판정을 돌려주므로 그 모양이다. 그렇다면 **왜 layer
하나당 processor 하나로 하지 않는가?** 힌트는 [[Q5]]의 "push는 되돌릴 수 없다".

---

## Q8. `FrameProcessor`가 정확히 뭐고, processor 말고 다른 것들은 무엇이 있나?

**Kernel.** `class FrameProcessor(BaseObject)`(`processors/frame_processor.py:195`)는 인터페이스가
아니라 **파이프의 칸이 되기 위한 기반 클래스**이며 public 메서드가 **54개**다. 상속하면 배선
(`link`/`next`/`previous`), 큐 2개와 task 2개(`queue_frame`/`push_frame`/`setup`/`cleanup`),
생명주기 처리(Start/Interruption/Cancel/pause/resume), 실행 환경(`get_clock`/`get_event_loop`/
`pipeline_worker`), 에러 라우팅(`push_error*` → UPSTREAM)이 딸려온다. **54개 중 19개가 메트릭
관련** — 관측이 사후 추가가 아니라 기반 클래스 수준의 일급 관심사라는 뜻이며 [[ch-11/read]]의
observer plane이 그 위에 선다.

**파이프에 들어가는 것 = `FrameProcessor` 자손 239개** (AST 전수 조사):

| 디렉터리 | 수 | |
|---|---:|---|
| `services/` | **157** | 66%. 벤더 API 래퍼(STT/TTS/LLM) |
| `transports/` | 30 | |
| `processors/` | 27 | 순수 처리 로직은 이것뿐 |
| `pipeline/` | 10 | `Pipeline` 자신도 processor(→ 중첩 가능, read.md §5.3) |
| `extensions/` | 8 | ivr, voicemail |
| bus/turns/evals/tests | 7 | |

**파이프에 안 들어가는 것.** `BaseObject` 자손은 총 325개이고 그중 239개만 `FrameProcessor`다 —
**86개는 processor가 아니다.**

| 계열 | 자손 | 정체 |
|---|---:|---|
| `Frame` | ~150 | 흐르는 데이터 ([[ch-02/read]]) |
| `BaseTransport` | 15 | ⚠ 아래 |
| `BaseObserver` | 13 | 파이프를 **지켜보되 끼어들지 않음** |
| `FrameSerializer` | 8 | 텔레포니 + protobuf |
| `BaseUserTurnStartStrategy` / `StopStrategy` | 6 / 6 | turn 판정 |
| `VADAnalyzer` | 4 | Silero 등 |
| `BaseAudioFilter` | 4 | 노이즈 제거 |

클래스 계보가 아닌 서브시스템: `bus/`(사이드채널), `flows/`(`FlowManager`는 파이프 **밖**에서
구동, [[ch-10/read]]), `runner/`·`workers/`(실행), `adapters/`(provider별 tool schema).

**⚠ `BaseTransport`는 processor가 아니다.** 표준 파이프라인이 `Pipeline([transport.input(), ...,
transport.output()])`인 이유 — `BaseTransport`(`transports/base_transport.py:96`, 42줄 클래스)의
abstract 메서드는 `input()`(`:122`)과 `output()`(`:131`) 둘뿐이고, transport는 **processor 두 개를
들고 있는 상자**다. 하나의 물리적 연결(WebRTC peer, WebSocket)의 입력과 출력이 파이프의 서로
다른 끝에 앉아야 하므로, 객체 하나로는 두 자리를 못 차지한다.

**세 층으로 정리:**

| 층 | 예 | 연결 방식 |
|---|---|---|
| 흐르는 것 | `Frame` 계열 | — |
| 파이프의 칸 | services, aggregators, filters, `transport.input()` | `link()` → `push_frame` |
| 부품·인프라 | VAD, turn strategy, serializer, audio filter, observer, bus, `FlowManager` | 직접 호출 / 소유 / 구독 |

[[Q7]]이 2층과 3층의 구분이었고 [[Q2]]가 2층의 통신 규약이었다.

**열어둔 probe.** `BaseObserver` 13개는 **지켜보되 끼어들지 않는** 계열이다. 그러면
[[ch-12/read]]에 선택지가 하나 더 있다: rule을 processor로 넣으면 막을 수 있지만 지연을 먹고,
observer로 넣으면 지연이 0이지만 **막을 수 없다.** boson의 rule을 "관찰만" 하는 것과 "막아야"
하는 것으로 나눌 수 있다면 둘을 다른 층에 놓을 수 있다.

---

## Q9. Pipeline은 processor가 Frame을 순서대로 처리하도록 감싸는 라인인가?

**Kernel.** 절반만 맞다. 순서를 정하는 것은 맞지만, (a) Pipeline은 감싸개가 아니라 **그 자체가
processor**이고, (b) 실행은 **순차가 아니라 겹친다.**

**(a) Pipeline은 processor다.** `class Pipeline(BasePipeline)`(`pipeline/pipeline.py:91`),
`class BasePipeline(FrameProcessor)`(`pipeline/base_pipeline.py:19`). 따라서 `[A, B, C]`와
`[A, Pipeline([B, C])]`는 바깥에서 구분되지 않으며 파이프라인을 중첩할 수 있다(read.md §5.3의
결합법칙). 구성 작업의 전부는 4줄 fold다(`:197-202`):
```python
prev = self._processors[0]
for curr in self._processors[1:]:
    prev.link(curr)
    prev = curr
```
그 대가로 앞뒤에 어댑터가 자동으로 붙는다(`:117-119`): `Pipeline([stt, llm, tts])`의 실제
사슬은 `[PipelineSource, stt, llm, tts, PipelineSink]`이며, Source는 바깥에서 온 frame을 안쪽
첫 칸에 넘기고 안쪽에서 UPSTREAM으로 올라온 것은 `self.push_frame`으로 바깥에 내보낸다.
중첩을 성립시키는 접착제다.

**(b) 순차가 아니다.** processor마다 자기 큐와 자기 task가 있으므로([[Q1]], [[Q6]]) 어느
시점에 A는 frame 3, B는 frame 2, C는 frame 1을 **동시에** 처리한다. 단계는 순서대로지만 실행은
겹친다 — CPU pipeline과 같은 의미이고 그래서 이름이 pipeline이다. 컨베이어 벨트이지 순차 함수
호출이 아니다.

> **`link()`가 순서를 정하고, 각자의 task가 동시성을 만든다.**

이 겹침이 없으면 voice agent가 성립하지 않는다. LLM이 토큰을 뱉는 **동안** TTS가 앞부분 오디오를
만들고 있어야 time-to-first-audio가 산다([[ch-07/read]]).

**곁다리.** Pipeline은 `super().__init__(enable_direct_mode=True)`(`:113`)로 자기 큐를 쓰지
않는다 — 아무 일도 하지 않고 넘기기만 하므로 큐를 둘 이유가 없고, `queue_frame`의 direct-mode
분기(`frame_processor.py:717-719`)로 곧장 `__process_frame`을 부른다. **구조를 만드는 비용이
0에 가깝다**는 뜻이며, 그래서 중첩을 마음껏 해도 된다.
