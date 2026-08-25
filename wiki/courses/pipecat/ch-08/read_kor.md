---
title: "Barge-In과 Interruption Cascade"
chapter: ch-08
phase: read
course: pipecat
kind: korean-companion
source: [[read]]
sources:
  - interruption-cascade
  - frame-taxonomy
  - frame-processor
  - theory-out-of-band-priority
  - boson-interrupt-subsystem
  - rtv-vad-chunking
  - rtv-vs-pipecat-gap
figure: figures/bargein-timeline.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# Barge-In과 Interruption Cascade

> 이 문서는 [[read]] (`ch-08/read.md`)의 한국어 companion입니다. 섹션 번호가 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 — frame, processor, pipeline, queue, aggregator, endpointing,
> back-pressure, interruption, timestamp, cascade, barge-in, serializer, playout buffer.

## 왜 이 챕터인가

[[ch-06/read]]은 코드 한 줄로 끝났습니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1269-1270`**
```python
        if params.enable_interruptions:
            await self.broadcast_interruption()
```

ch-06이 가르친 모든 것 — VAD hysteresis machine, `UserTurnStartStrategy` chain, turn boundary가 제안되고
승인(ratify)될 수 있는 세 개의 layer — 은 저 `if`에 도달하기 위해 존재합니다. 이 chapter는 그 다음 줄에서
시작합니다. **interruption이 발생했다는 판단은 이미 우리 뒤에 있습니다.** provider 선택도 뒤에 있습니다.
여기서 던지는 질문은 더 좁고, 실제 전화선 너머에 고객이 있는 순간부터는 더 비쌉니다:

> `broadcast_interruption()`이 호출된 순간과, 이미 발화된 마지막 byte가 고객의 수화기에서 나오기를 멈추는
> 순간 사이에 — 무엇이, 어떤 순서로, 어느 task 위에서 일어나며, 그게 다 끝났을 때 conversation history에는
> 무엇이 남아 있는가?

이 chapter가 5번이 아니라 8번 자리에 놓인 이유는, cascade가 그보다 앞에서는 진짜로 가르칠 수 없기
때문입니다. 이걸 따라가려면 지금 여러분이 갖고 있는 두 가지가 필요합니다:

- **[[ch-04/read]] §4** — processor당 two-queue / two-task runtime. priority tier 1/10/20, `SystemFrame`은
  input task 위에서 inline 실행, 나머지는 두 번째 *cancel 가능한* process task로 relay,
  `FrameQueue.reset()`은 `UninterruptibleFrame`을 유지. 이게 없으면 "interruption이 processor를
  cancel한다"는 그냥 구호입니다.
- **[[ch-07/read]]** — TTS output queue, word-timestamp path, 그리고 output transport의 clock task.
  이게 없으면 "truncation은 pipeline position에서 emergent하게 발생한다"는 문장을 검증할 수 없습니다.

이 chapter에는 **아이디어가 정확히 하나** 있고, 그것은 section 7입니다. section 1–6은 section 7을
손짓(hand-waving) 없이 읽기 위해 필요한 machinery이고, section 8–10은 그걸로 무엇을 하느냐입니다.

### 시작하기 전 두 개의 정정

둘 다 이 course가 조립된 자료 안에 있었고, 둘 다 commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`
시점의 tree에 대해 틀렸습니다. excerpt와 source가 어긋나면 **source가 이깁니다**. 저는 조용히 한쪽을
고르는 대신 본문에서 그렇다고 말합니다.

**정정 1 — 여러분이 배운 그 frame들은 존재하지 않습니다.** `StartInterruptionFrame`도 없고
`StopInterruptionFrame`도 없습니다. field가 하나도 없는 `InterruptionFrame` 하나뿐입니다. §2에서 grep을
돌립니다.

**정정 2, 더 큰 쪽 — truncation routine이라는 것은 없습니다.** 느린 것도, 숨겨진 것도, 다른 이름을 가진
것도 없습니다. `LLMContext`에는 어떤 종류의 interruption handling도 없습니다. 밖에서 보기에
"Pipecat이 assistant message를 발화된 만큼으로 truncate한다"처럼 보이는 것은, assistant aggregator가
*list의 어디에 앉아 있느냐*의 **부수효과(side effect)** 더하기 그 입력이 audio clock에 의해 pacing된다는
사실입니다. §7이 실제 코드로 추적하고, 그 다음 그 emergent behaviour가 무너지는 두 지점을 보여줍니다 —
그중 하나는 **여러분에게 구체적으로** 닥칩니다. 여러분의 한국어 TTS 벤더가 word timestamp를 내보내느냐에
달려 있기 때문입니다.

### 증거를 읽는 방법

아래의 모든 Pipecat 줄 번호는 commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`에서 열어본
것입니다(2026-08-25; `CHANGELOG` head `[1.7.0] - 2026-08-01`). `boson-agent`의 `gateway/interrupt/`와
`realtime_voice`에 대한 모든 주장은 excerpt library — [[boson-interrupt-subsystem]], [[rtv-vad-chunking]],
[[rtv-vs-pipecat-gap]] — 에서 나오며, 이것들은 2026-08-25에 여러분의 private repo에서 읽은 것입니다.
그 repo들은 여기서 다시 열지 않습니다.

그리고 이 course의 상시 invariant를 다시 말합니다. 이 chapter가 그걸 깨기 가장 쉬운 자리이기 때문입니다:
**§9는 mechanism differential이지 채점(scoring)이 아닙니다.** 세 시스템 각각이 *무엇을 하는지*만
진술합니다. "더 낫다"도, "이긴다"도, "채택해야 한다"도 없습니다. [[ch-13/read]]가 무언가를 채점하는
유일한 곳입니다.

---

## 1. signal은 어디서 오는가 — 여섯 개의 발화점, 하나의 호출

`broadcast_interruption()`은 `FrameProcessor`의 method입니다. 즉 *어떤* processor든 이걸 쏠 수 있습니다.
tree에는 열 개의 call site가 있고, 그중 하나는 `frame_processor.py` 자체 안의 deprecated shim이므로
실제 originator는 아홉 개입니다:

```
$ grep -rn "await self.broadcast_interruption()" src/
src/pipecat/extensions/voicemail/voicemail_detector.py:370
src/pipecat/processors/frame_processor.py:1036        ← deprecated shim
src/pipecat/processors/aggregators/llm_response_universal.py:1270
src/pipecat/processors/aggregators/dtmf_aggregator.py:106
src/pipecat/processors/frameworks/rtvi/processor.py:146
src/pipecat/turns/user_turn_processor.py:210
src/pipecat/services/ultravox/llm.py:650
src/pipecat/services/google/gemini_live/llm.py:1333
src/pipecat/services/aws/nova_sonic/llm.py:1528
src/pipecat/services/aws/nova_sonic/llm.py:1554
```

| Originator | Line | 무엇이 결정했는가 |
|---|---|---|
| `LLMUserAggregator._on_user_turn_started` | `llm_response_universal.py:1270` | ch-06의 turn-start strategy chain이 user turn을 ratify했다 |
| `UserTurnProcessor` | `turns/user_turn_processor.py:210` | 동일한 세 줄이, 독립 processor 형태로 |
| `DTMFAggregator` | `aggregators/dtmf_aggregator.py:106` | 고객이 키를 눌렀다 |
| `RTVIProcessor` | `frameworks/rtvi/processor.py:146` | *client*가 RTVI protocol로 명시적 interrupt를 보냈다 |
| `VoicemailDetector` | `extensions/voicemail/voicemail_detector.py:370` | 지금 자동응답기와 통화 중이다 |
| Gemini Live / Nova Sonic / Ultravox | `:1333` / `:1528`,`:1554` / `:650` | **provider**가 server-side에서 barge-in을 감지하고 우리에게 알려줬다 |

여러분이 쓰게 될 것은 첫 번째입니다. 주변 네 줄이 ch-06과 이 chapter를 잇는 이음매(seam)이므로 전체를
그대로 봅니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1253-1272`**
```python
    async def _on_user_turn_started(
        self,
        controller: UserTurnController,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        logger.debug(f"{self}: User started speaking (strategy: {strategy})")

        self._user_turn_start_timestamp = time_now_iso8601()
        self._full_user_turn_aggregation = None

        if params.enable_user_speaking_frames:
            await self.broadcast_frame(UserStartedSpeakingFrame)

        await self._user_idle_controller.process_frame(UserStartedSpeakingFrame())

        if params.enable_interruptions:
            await self.broadcast_interruption()

        await self._call_event_handler("on_user_turn_started", strategy)
```

앞으로 들고 갈 세 가지 사실.

**(a) `enable_interruptions`의 default는 `True`입니다.** `base_user_turn_start_strategy.py:56` —
`enable_interruptions: bool = True`. barge-in은 끄지 않는 한 켜져 있고, strategy 단위로 설정되며,
`BaseUserTurnStartStrategy`는 개별 strategy가 trigger 시점에 override하는 것도 허용합니다(`:200-220`).
"compliance disclosure를 읽는 동안에는 고객이 끼어들 수 없어야 한다" — 한국 보험 tele-sales에서 실제로
있는 요구사항이고, 이 knob을 모르면 downstream에서 hack으로 때우고 싶어질 바로 그 요구사항입니다.

**(b) `UserStartedSpeakingFrame`과 `InterruptionFrame`은 서로 다른 일을 하는 서로 다른 frame입니다.**
둘 다 field 없는 `SystemFrame`입니다(`frames.py:1154`와 `:1142`). `UserStartedSpeakingFrame`은 *turn*을
표시합니다 — 자기 docstring이 "usually means that some transcriptions are already available"라고 말합니다.
`InterruptionFrame`은 *하던 걸 멈춰라*라는 뜻입니다. 여기서는 한 줄 간격으로 broadcast되지만, 구조적으로
둘을 묶는 것은 아무것도 없습니다: `enable_user_speaking_frames`와 `enable_interruptions`는 독립적인
boolean이고, 위 여섯 originator 중 다섯은 turn frame 없이 interruption만 push합니다. 하나에서 다른 하나를
추론하는 processor를 만들지 마십시오.

**(c) aggregator를 통째로 건너뛰는 일곱 번째 경로가 있습니다.** interruption이 pipeline 바깥에서 —
worker bus를 통해, 예컨대 supervisor process나 "이 고객한테 지금 당장 말 그만해"라고 말하는 REST
endpoint에서 — 도착하면, worker가 frame을 직접 주입합니다:

**`src/pipecat/pipeline/worker.py:1280-1286`**
```python
        elif isinstance(frame, InterruptionWorkerFrame):
            # Tell the worker we should interrupt the pipeline. Note that we are
            # bypassing the push queue and directly queue into the
            # pipeline. This is in case the push worker is blocked waiting for a
            # pipeline-ending frame to finish traversing the pipeline.
            logger.debug(f"{self}: received interruption worker frame upstream {frame}")
            await self._pipeline.queue_frame(InterruptionFrame())
```

주석을 읽으십시오. [[ch-04/read]] §6.2에서 추적한 engine loop인 `_push_queue`를 **우회합니다** — 정확히
그 loop이 `_wait_for_pipeline_end(frame)` 안에 앉아 있을 수 있고, 그러면 interruption을 처리할 차례가
영영 오지 않기 때문입니다. 이것은 out-of-band 원칙을 한 층 위에 적용한 것입니다: *worker의* 단일 ordered
queue조차 queue이고, control signal은 그 안에 앉아 있게 두면 안 됩니다. Lina 입장에서 이건 "CRM 콘솔에서
봇의 현재 발화를 죽인다"의 mechanism입니다 — [[ch-11/read]]과 [[ch-12/read]] 둘 다 이걸 원하게 됩니다.

> 💡 **쉬운 설명 — "push queue를 우회한다"가 왜 필요한가요?**
> worker가 frame을 pipeline에 넣는 정상 경로는 `_push_queue`라는 하나의 줄서기입니다. 그런데 그
> 줄서기를 돌리는 loop이 "지금 보낸 `EndFrame`이 pipeline 끝까지 도달할 때까지 기다리는 중"일 수
> 있습니다. 그 상태에서 supervisor가 "당장 멈춰"를 같은 줄에 세우면, 앞사람이 끝나야 처리됩니다 —
> 그런데 앞사람이 끝나려면 봇이 말을 다 마쳐야 합니다. 순환입니다. 그래서 이 한 종류의 frame만
> 줄을 건너뛰고 `pipeline.queue_frame()`으로 직접 들어갑니다. §4의 "signal은 queue에 앉으면 안 된다"가
> pipeline 안이 아니라 worker 층에서 한 번 더 반복되는 것입니다.

여기서부터 signal은 존재합니다. 이 chapter의 나머지 전부는 그 결과(consequence)입니다.

---

## 2. 존재하지 않는 frame, 그리고 존재하는 frame

### 2.1 grep

이 course가 조립된 자료 — 그리고 대부분의 Pipecat 블로그 글, 그리고 2024년 버전 라이브러리를 외운 모든
LLM — 은 `StartInterruptionFrame` / `StopInterruptionFrame`을 언급합니다. 사라졌습니다:

```
$ grep -rn "StartInterruptionFrame\|StopInterruptionFrame\|BotInterruptionFrame" src/
$ echo $?
1
```

package 전체에서 hit 0. start/stop 쌍도 없고, bot-side variant도 없습니다. frame은 하나입니다:

**`src/pipecat/frames/frames.py:1141-1151`**
```python
@dataclass
class InterruptionFrame(SystemFrame):
    """Frame pushed to interrupt the pipeline.

    This frame is used to interrupt the pipeline. For example, when a user
    starts speaking to cancel any in-progress bot output. It can also be pushed
    by any processor.
    """

    pass
```

`pass`. **field가 없습니다.** generation id도, timestamp도, reason도, "얼마나 재생됐는지" 힌트도 없습니다.
instance의 모든 field는 `Frame` root에서 옵니다 — `id`, `name`, `pts`, `broadcast_sibling_id`, `metadata`,
`transport_source`, `transport_destination` (`frames.py:83-98`), 전부 `field(init=False)`.

그 비어 있음이 design 전체입니다. generation id를 가진 frame이라면 모든 processor가 그것을 무언가와
*비교*해야 하고, 그러면 모든 processor가 그 무언가를 들고 있어야 하고, 그러면 processor를 하나 추가할
때마다 비교를 추가하는 걸 기억해야 합니다. field 없는 frame은 수신자가 단 하나만 알면 됩니다: *멈춰라*.
대가는 어떤 processor도 "**어느** turn을 멈추라는 거지?"를 물을 수 없다는 것이고 — §7.7이 그 청구서가
날아오는 자리입니다.

[[ch-03/read]] §8은 이미 이것을 `realtime_voice`의 선택 옆에 놓았습니다. 거기서는 signal이 정수이고
(`self._active_generation`), 이름 붙은 여섯 지점에서 비교됩니다. 둘 다 mechanism에 대한 참인 진술이고,
여기서 어느 쪽도 채점되지 않습니다.

> 💡 **쉬운 설명 — field 없는 frame의 진짜 trade-off**
> generation id가 있으면 "지금 도착한 이 audio chunk가 죽은 turn 것인가?"를 각 지점에서 물을 수 있습니다.
> 정확하지만, 물어보는 코드를 모든 지점에 사람이 손으로 심어야 합니다(= `realtime_voice`의 여섯 site).
> field가 없으면 물어볼 수 없지만, 대신 base class의 `_start_interruption()` 하나만 상속하면 자동으로
> 참여합니다(= §4). 정확성을 확장성과 맞바꾼 것이고, §7.7의 one-hop race가 정확히 그 맞바꾼 값입니다.

### 2.2 instance는 하나가 아니라 둘

`broadcast_interruption`은 네 줄입니다:

**`src/pipecat/processors/frame_processor.py:1017-1022`**
```python
    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        self.__reset_process_task()
        await self.stop_all_metrics()
        await self.broadcast_frame(InterruptionFrame)
```

아무것도 push되기 전인 1020번 줄을 보십시오: `self.__reset_process_task()`. 발화한 processor가 frame이
어디로든 가기 전에, 동기적으로, 자기 *자신의* process task부터 뜯어냅니다. 자기 broadcast를 수신하기를
기다리지 않습니다 — 기다릴 수도 없습니다. `broadcast_frame`은 이웃에게 push하지 자기 자신에게는
push하지 않기 때문입니다.

그 다음:

**`src/pipecat/processors/frame_processor.py:1038-1054`**
```python
    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.

        This method creates two instances of the given frame class using the
        provided keyword arguments (without deep-copying them) and pushes them
        upstream and downstream.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)
```

**서로 다른 두 객체입니다.** `id`가 다르고, `name`이 다르고, `broadcast_sibling_id`로 상호 연결되어 있고,
interruption이 발화한 지점에서 서로 반대 방향으로 달려 나갑니다.

[[ch-04/read]] §11의 canonical 7-processor chain 위에 얹어 보되, interruption이 3번 위치에서 발생했다고
합시다:

```
  transport.input()    stt    user_agg    llm    tts    transport.output()    assistant_agg
        1               2        3         4      5            6                   7
        ◄───────────────◄────  [FIRE]  ────►─────►──────────────►───────────────────►
             upstream instance            downstream instance
```

- **Upstream**은 `stt`와 `transport.input()`에 도달합니다. 거기서는 아무것도 bot speech를 생산하고 있지
  않으므로, 이쪽 절반은 대체로 input 쪽 자신의 metrics와 process task를 reset하는 일입니다.
  `STTService.process_frame`에는 TTFB state를 reset하고 forward하는 명시적 branch가 있고
  (`services/stt_service.py:510-512`), `BaseInputTransport`에는 `InterruptionFrame` branch가 아예 없습니다 —
  `grep -n InterruptionFrame src/pipecat/transports/base_input.py`는 아무것도 반환하지 않습니다 — 그래서
  input transport는 §4의 base-class 동작만 받습니다.
- **Downstream**은 bot speech를 생산하거나, buffering하거나, 기록하고 있는 모든 것에 도달합니다: LLM,
  TTS service, output transport, 그리고 assistant aggregator. 이 chapter의 나머지 전부가 그곳에 삽니다.

### 2.3 `broadcast_sibling_id`를 읽는 것 — 정확히 하나

이 상호 연결은 coordination처럼 보입니다. 아닙니다. tree 전체를 grep하십시오:

```
$ grep -rn "broadcast_sibling_id" src/
src/pipecat/transports/base_output.py:716,717,768,769     ← writes (BotStarted/StoppedSpeaking)
src/pipecat/frames/frames.py:75,86,95                      ← declaration
src/pipecat/processors/frame_processor.py:1051,1052,1086,1087  ← writes
src/pipecat/processors/frameworks/rtvi/observer.py:429      ← the ONLY read
```

reader는 하나, 그리고 그것은 observer입니다:

**`src/pipecat/processors/frameworks/rtvi/observer.py:427-430`**
```python
        # For broadcast frames (pushed in both directions), only process
        # the downstream copy to avoid sending duplicate RTVI messages.
        if frame.broadcast_sibling_id is not None and direction != FrameDirection.DOWNSTREAM:
            return
```

즉 sibling id는 cascade를 위한 coordination handle이 아니라 **observability plane을 위한
de-duplication handle**입니다([[ch-11/read]]의 주제). interruption path의 어떤 processor도 이걸 읽지
않습니다. "반대편에서 온 이 interruption은 이미 봤으니 건너뛴다"고 말하는 processor를 쓸 생각이었다면 —
framework 안에서 그렇게 하는 것은 아무것도 없고, 여러분이 첫 번째가 됩니다.

---

## 3. `EndFrame`은 `SystemFrame`이 아니고, 그것이 증거다

[[ch-04/read]] §4의 data/signal 분리가 구현상의 사고가 아니라 의도된 design이라는 가장 깔끔한 증거가
여기 있습니다. Pipecat pipeline이 멈추는 두 가지 방식을 비교하십시오.

**`src/pipecat/frames/frames.py:998-1009`** — 난폭한 쪽:
```python
@dataclass
class CancelFrame(SystemFrame):
    """Frame indicating pipeline should stop immediately.

    Indicates that a pipeline needs to stop right away without
    processing remaining queued frames.

    Parameters:
        reason: Optional reason for pushing a cancel frame.
    """

    reason: Any | None = None
```

**`src/pipecat/frames/frames.py:1899-1912`** — 우아한 쪽:
```python
@dataclass
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

`EndFrame`은 의도적으로 **in-band**입니다. `ControlFrame`이므로 `FrameProcessorQueue.put`이
`DEFAULT_PRIORITY = 20`을 부여하고, input task는 그것을 다른 data frame과 똑같이 느린 process queue로
relay합니다. 그것이 뒤따르던 audio *다음에* 도착합니다 — 정확히 여러분이 원하는 바입니다. "통화를
끝내라"가 봇의 마지막 문장을 잘라서는 안 되니까요. 만약 `EndFrame`이 priority channel을 탔다면
`worker.stop_when_done()`은 모든 작별 인사를 truncate했을 것입니다.

그리고 in-band이기 때문에 interruption flush에 파괴될 것이므로 — `UninterruptibleFrame` mixin을
달고 있고, docstring이 그 이유를 그대로 말합니다.

**`src/pipecat/frames/frames.py:146-157`**
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

이것이 **`Frame` subclass가 아니라는 점**에 주목하십시오 — [[frame-taxonomy]]가 기록한 대로 순수한
`@dataclass` mixin이고, 그래서 priority tiering을 흔들지 않고 세 개의 진짜 base class 아무거나와
결합됩니다. 열 개의 class가 이것을 직접 선언합니다:

```
$ grep -n "UninterruptibleFrame)" src/pipecat/frames/frames.py
770:  class FunctionCallResultFrame(DataFrame, UninterruptibleFrame)
1735: class EndWorkerFrame(WorkerFrame, UninterruptibleFrame)
1754: class StopWorkerFrame(WorkerFrame, UninterruptibleFrame)
1899: class EndFrame(ControlFrame, UninterruptibleFrame)
1923: class StopFrame(ControlFrame, UninterruptibleFrame)
1939: class PipelineFlushFrame(ControlFrame, UninterruptibleFrame)
2142: class LLMContextSummaryResultFrame(ControlFrame, UninterruptibleFrame)
2164: class FunctionCallInProgressFrame(ControlFrame, UninterruptibleFrame)
2363: class AudioBufferStartRecordingFrame(ControlFrame, UninterruptibleFrame)
2368: class AudioBufferStopRecordingFrame(ControlFrame, UninterruptibleFrame)
```

([[ch-03/read]]의 AST walk는 *transitive*하게, 즉 이 열 개의 subclass까지 포함해서 13개를 셌습니다.)

저 목록을 **정책 선언문**으로 읽으십시오. barge-in에서 살아남아야 하는 것은: pipeline 종료
(`EndFrame`, `StopFrame`, 두 개의 worker variant), 진행 중이던 tool call의 *정산(settlement)*
(`FunctionCallInProgressFrame`, `FunctionCallResultFrame`, `LLMContextSummaryResultFrame`), 그리고
recording 경계. 나머지 전부 — bot audio의 모든 byte, bot text의 모든 단어, 모든 context frame — 는
기본적으로 버려도 되는 것(expendable)입니다.

mixin을 존중하는 selective flush는 아홉 줄입니다:

**`src/pipecat/utils/frame_queue.py:84-95`**
```python
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

전부 빼내고, 생존자만 순서대로 다시 넣습니다. `_uninterruptible_count`는 `_put`/`_get`을 override해서
O(1)로 유지되므로(`frame_queue.py:73-82`), §5.5가 의존하는 `has_uninterruptible` property는 scan이 아니라
비교 한 번입니다.

### 3.1 GStreamer가 2001년에 그은 선, 그리고 Pipecat이 그것을 그대로 들여온 것

[[theory-out-of-band-priority]]에 따르면: GStreamer 자신의 plugin-writer 문서는 downstream event를 두
종류로 나눕니다 — *in-band*, "serialised with the buffer flow", 그리고 *out-of-band*, "travelling through
the pipeline instantly, possibly not in the same thread as the streaming thread that is processing the
buffers, **skipping ahead of buffers being processed or queued in the pipeline**". `SEGMENT`, `CAPS`,
`TAG`, `EOS`는 serialised입니다. `FLUSH_START`는 out-of-band이고, "unblocks the streaming thread by making
all pads reject data."

대응은 정확합니다:

| GStreamer | Pipecat |
|---|---|
| buffers | `DataFrame` |
| out-of-band events | `SystemFrame` |
| `FLUSH_START` (out-of-band) | `InterruptionFrame`, `CancelFrame` |
| `EOS` (serialised) | `EndFrame` is a `ControlFrame` |
| pads reject data after flush | `_cancelling` → `queue_frame` returns early (`frame_processor.py:714-715`) |

그리고 이것은 나중에 누군가 발견한 우연한 닮은꼴이 아닙니다 — Pipecat은 진짜 `Gst` graph를 ship합니다:

**`src/pipecat/processors/gstreamer/pipeline_source.py:39`**
```python
class GStreamerPipelineSource(FrameProcessor):
```

Pipecat은 2001년 media-framework design을 다시 유도해낸 것입니다. 만약 대칭성을 위해 `EndFrame`이
`SystemFrame`"이어야 한다"고 주장하고 싶어진다면, 여러분은 `EOS`가 out-of-band여야 한다고 주장하는
것이고, GStreamer의 25년이 아니라고 답합니다.

> 💡 **쉬운 설명 — in-band와 out-of-band를 한 문장으로**
> in-band = "줄을 서서 간다". 앞의 audio가 다 나간 다음에 도착한다. 그래서 *뒤따르던 내용을 존중*해야
> 하는 신호에 맞습니다 (통화 종료 = 마지막 문장까지 다 말한 뒤).
> out-of-band = "줄을 건너뛴다". 앞의 audio를 무시하고 즉시 도착한다. 그래서 *뒤따르던 내용을 파괴*하는
> 것이 목적인 신호에 맞습니다 (barge-in = 지금 당장 입 다물어).
> 즉 어느 channel을 타느냐는 성능 최적화가 아니라 **의미론(semantics)** 선택입니다. `EndFrame`을
> priority channel에 올리면 빨라지는 게 아니라 **틀려집니다**.

---
## 4. cascade에는 coordinator가 없다

이것이 구조적 주장이고, 가능한 한 무뚝뚝하게 말할 가치가 있습니다: **Pipecat에서 interruption을
orchestrate하는 것은 아무것도 없습니다.** `InterruptionManager`도 없고, 정해진 teardown 순서도 없고,
barrier도 없고, acknowledgement도 없습니다. N개의 processor에 도달하는 frame 하나가 있고, N개 각각이
독립적으로 자기 자신에게 똑같은 작은 일을 합니다.

그 작은 일은 base class에 있고, 모든 processor가 그것을 상속하며 — [[ch-01/read]] §7.2에 따라 — 모든
processor가 그것을 호출할 의무가 있습니다:

**`src/pipecat/processors/frame_processor.py:837-847`**
```python
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

그리고:

**`src/pipecat/processors/frame_processor.py:1130-1150`**
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

branch는 둘입니다. 지금 처리 중인 frame이 uninterruptible이거나 — 이 경우 *queue*만 flush하고 실행 중인
coroutine은 건드리지 않습니다 — 아니면 process task를 통째로 죽이고 다시 만듭니다.

`InterruptionFrame`이 `SystemFrame`이기 때문에, 이 모든 것은 **input task 위에서** 실행되고, 그 task는
cancel되는 대상이 아닙니다. 그것이 [[ch-04/read]] §4.3이 세워둔 비대칭입니다: process task는 소모품이고,
input task는 아닙니다. processor는 자기 data half를 cancel하면서도 살아서 *다음* system frame을 계속
듣습니다. 두 half가 한 task였다면, processor는 cancellation을 실행하고 있는 바로 그 task를 cancel해야
했을 것입니다.

### 4.1 청구서가 날아온다: adjacency는 거짓말이다

`:1149`의 `await self.__cancel_process_task()`는 여러분의 coroutine이 **어디에 있든**, 여러분의
`process_frame()` 안에서 suspend되어 있던 그 `await`에서, `CancelledError`를 던져 넣습니다.

누구나 쓰는 가장 평범한 processor를 생각해 보십시오 — 한 frame에서 누적하고, 다른 frame에서 flush하는 것:

```python
class SentenceLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)          # mandatory, ch-01 §7.2
        if isinstance(frame, LLMTextFrame):
            self._buffer += frame.text                          # frame A
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._sink.write(self._buffer)                # frame B
            self._buffer = ""
        await self.push_frame(frame, direction)
```

frame A와 B는 인접해 보입니다. 아닙니다: 그것들은 `__process_frame_task_handler`를 통과하는 서로 다른 두
번의 여정이고, interruption이 그 사이에 착지할 수 있습니다. 그러면 `_buffer`는 절반만 찬 채로 남고,
process task는 재생성되며, *다음* turn의 첫 `LLMTextFrame`이 지난 turn의 조각 위에 붙습니다. 여러분의
CRM log는 이제 `"고객님 이 상품은 65세까지 갱네, 안녕하세요"`라고 적혀 있습니다.

이건 가정이 아닙니다. **Pipecat은 자기 코드에서 이걸 겪었고, 명시적인 한 줄로 값을 치릅니다**:

**`src/pipecat/services/tts_service.py:1041`**
```python
        self._aggregated_frame_sequencer.clear()  # discard all pending slots on interruption
```

이 주석은 누군가 그 버그를 출시했기 때문에 존재합니다. framework 자신의 TTS service에 있는 모든
accumulator는 `InterruptionFrame`에서 손으로 clear됩니다 — §5.3에 목록이 있습니다 — 정확히 이 이유로.

여기서 떨어져 나오는 규칙, 그리고 [[ch-12/read]]에서 rule-layer processor를 쓸 때 기억해야 할 가장 중요한
것:

> **모든 `process_frame()`을, 자기가 인접하다고 믿는 두 frame 사이에서 cancel될 수 있다고 가정하고
> 쓰십시오. 실제로 그렇기 때문입니다.**

§8이 그것을 checklist로 바꿉니다.

> 💡 **쉬운 설명 — 왜 "인접해 보이는데 인접하지 않은가"**
> `process_frame`은 frame 하나마다 한 번씩 호출되는 **독립적인 함수 호출**입니다. `_buffer`만 instance에
> 남아 있을 뿐, 호출과 호출 사이는 event loop에게 완전히 열려 있습니다. 즉 A와 B 사이의 "틈"은
> 코드에서는 `elif` 한 줄이지만 런타임에서는 수백 밀리초짜리 구멍이고, 그 구멍에 cancellation이
> 들어옵니다. 일반 Python에서 `try/finally`로 지킬 수 있는 것과 다릅니다 — 여기서는 A의 호출이 이미
> **정상 종료**했고, B의 호출은 **시작조차 안 했기** 때문입니다. 지킬 `finally`가 없습니다.
> 유일한 방어는 "`InterruptionFrame`을 보면 내 state를 내가 지운다"는 세 번째 branch입니다.

---

## 5. canonical pipeline에서 hop 단위로

이제 downstream instance를 canonical chain의 4, 5, 6, 7번 위치로 걸어 보냅니다. 각 hop은 독립적이고,
각 hop은 서로 다른 일을 하며, 아무것도 다른 것을 기다리지 않습니다.

### 5.1 LLM — abort API는 없다

부정적 결과부터 시작합니다. 사람들이 당연히 있다고 가정해 버리는 부분이기 때문입니다. `LLMService`는
frame을 처리하긴 합니다:

**`src/pipecat/services/llm_service.py:688-689`**
```python
        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
```

그리고 handler는 세 줄입니다:

**`src/pipecat/services/llm_service.py:758-761`**
```python
    async def _handle_interruptions(self, _: InterruptionFrame):
        for function_name, entry in self._functions.items():
            if entry.cancel_on_interruption:
                await self._cancel_function_call(function_name)
```

이것이 LLM 쪽 interruption handling의 **전부**입니다: tool call을 cancel한다. `client.abort()`도 없고,
`response.close()`도 없고, OpenAI나 Anthropic에 보내는 cancellation token도 없습니다. 그리고 이 branch가
frame을 **forward하지 않는다**는 점도 보십시오 — `LLMService.process_frame`은 push 없이 떨어집니다.
forwarding은 구체 provider subclass의 `else`에서 일어납니다:

**`src/pipecat/services/openai/base_llm.py:590-615`**
```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Handles LLMContextFrame to trigger LLM completions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

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
        else:
            await self.push_frame(frame, direction)
```

그러면 실제로 무엇이 generation을 죽입니까? **§4입니다.** `_process_context`는
`__process_frame_task` 안에서 실행 중이고, provider의 streaming `async for` 안 어느 `await`에 suspend되어
있습니다. `__cancel_process_task()`가 거기에 `CancelledError`를 던지고, async generator가 닫히고,
provider로 가는 HTTP/websocket connection이 자기 context manager에 의해 뜯깁니다.

> **Source correction.** [[interruption-cascade]]는 "`LLMContextFrame`은 `DataFrame`이다"라고 말합니다.
> 아닙니다. **`src/pipecat/frames/frames.py:551`**은 `class LLMContextFrame(Frame)`입니다 — 세 branch 중
> 어디에도 속하지 않고 root를 직접 상속합니다. ([[ch-03/read]] §7.6이 이미 이 class를 taxonomy leak으로
> 표시했습니다.) 그럼에도 excerpt가 끌어낸 *결론*은 맞고, 이제 여러분은 그걸 받아들이는 대신 직접 유도할
> 수 있습니다: `FrameProcessorQueue.put`(`frame_processor.py:162-168`)은 `isinstance(frame, SystemFrame)`
> 일 때만 `SYSTEM_PRIORITY`를 부여하므로, 맨 `Frame`은 `else` branch로 떨어져 `DEFAULT_PRIORITY = 20`을
> 받습니다. `__input_frame_task_handler`(`:1304-1307`)는 `SystemFrame`이 아닌 모든 것을
> `__process_queue`로 relay합니다. 그리고 `FrameQueue.reset()`은 `UninterruptibleFrame`이 아닌 모든 것을
> 버립니다. 같은 운명, 다른 이유. 이것은 taxonomy가 샌다는 ch-02의 경고의 살아 있는 사례입니다 — 세
> branch 밖의 frame은 조용히 *default* 취급을 받고, 여기서는 마침 그 default가 맞았습니다.

**반드시 내면화해야 할 결과: interrupt된 turn에서는 `finally` block이 끝까지 실행되지 않습니다.**
`CancelledError`가 `_process_context` 밖으로 전파되고, Python은 cancellation 도중에도 `finally` block을
실행하기는 하지만, `await self.push_frame(LLMFullResponseEndFrame())` 자체가 cancel된 task 안의
`await`이고 그 task는 해체되는 중입니다. 실무적으로 **interrupt된 turn에서 `LLMFullResponseEndFrame`은
downstream으로 전달되지 않습니다**.

여기서 이 chapter의 나머지가 답하는 질문이 나옵니다: 평소에 assistant turn을 닫는 frame이 결코 도착하지
않는다면, **누가 그 turn을 닫습니까?** assistant aggregator의 dispatch table을 보십시오:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1552-1566`** (일부)
```python
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._handle_end_or_cancel(frame)
            await self.push_frame(frame, direction)
        ...
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
```

같은 방으로 들어가는 서로 다른 두 개의 문. 깨끗한 turn에서는 `_handle_llm_end` →
`_trigger_assistant_turn_stopped()`. interrupt된 turn에서는 `_handle_interruptions` →
`_trigger_assistant_turn_stopped(interrupted=True)`. §7.4가 그 방을 엽니다.

### 5.2 Tool call — 그리고 두 번째 source correction

[[boson-interrupt-subsystem]]과 [[theory-out-of-band-priority]] 둘 다 "`InterruptionFrame`은 bot turn을
truncate하지만 `ToolResultBlock`을 결코 합성하지 않는다"고 주장하고, [[ch-04/read]]는 그것을
[[ch-09/read]]를 위한 열린 질문으로 park해 뒀습니다. **그 주장은 이 commit에서 틀렸고, 여러분이 무엇을
포팅해야 하는지를 바꾸기 때문에 지금 정정할 가치가 있습니다.**

추적합시다. `_cancel_function_call`:

**`src/pipecat/services/llm_service.py:2016-2020`**
```python
    async def _cancel_function_call(self, function_name: str | None):
        await self._cancel_function_call_tasks(
            lambda item: item.registry_item.function_name == function_name,
            reason="interruption",
        )
```

그 docstring이 하는 일 세 가지를 이름으로 말합니다:

**`src/pipecat/services/llm_service.py:1896-1901`**
```
        Cancelling a call delivers ``asyncio.CancelledError`` to its handler so
        cancel-aware handlers run their cleanup, broadcasts a
        ``FunctionCallCancelFrame`` so the rest of the pipeline can settle the
        call, and notifies application code via ``on_function_calls_cancelled``.
```

`FunctionCallCancelFrame`은 그 자체로 `SystemFrame`이고(`frames.py:1363`), `function_name`,
`tool_call_id`, `run_llm: bool = False`를 실어 나릅니다 — 그리고 마지막 것의 field docstring은 정확합니다:
*"an interruption must not trigger inference"*. assistant aggregator가 그것을 받아 message를 정산합니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1931-1950`**
```python
    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        logger.debug(
            f"{self} FunctionCallCancelFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        function_call = self._function_calls_in_progress.get(frame.tool_call_id)
        if not function_call:
            return

        # Update context with the function call cancellation. Async calls are
        # settled with a developer message, the same channel their results
        # arrive on.
        if function_call.cancel_on_interruption:
            self._update_function_call_result(frame.function_name, frame.tool_call_id, "CANCELLED")
        else:
            self._context.add_message(
                async_tool_messages.build_cancelled_message(frame.tool_call_id)
            )

        group_id = function_call.group_id
        del self._function_calls_in_progress[frame.tool_call_id]
```

그리고 `_update_function_call_result`가 placeholder를 제자리에서 덮어씁니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:2157-2165`**
```python
    def _update_function_call_result(self, function_name: str, tool_call_id: str, result: Any):
        for message in self._context.get_messages():
            if (
                not isinstance(message, LLMSpecificMessage)
                and message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result
```

덮어쓰이는 그 placeholder는 call이 시작되는 순간에 쓰였습니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1769-1781`**
```python
        is_async = not frame.cancel_on_interruption
        if is_async:
            self._context.add_message(async_tool_messages.build_started_message(frame.tool_call_id))
        else:
            self._context.add_message(
                {
                    "role": "tool",
                    "content": "IN_PROGRESS",
                    "tool_call_id": frame.tool_call_id,
                }
            )

        self._function_calls_in_progress[frame.tool_call_id] = frame
```

그래서 정정된 진술은 정확히 이렇습니다: **Pipecat은 앞서 `IN_PROGRESS` placeholder를 써 두고 그것을
`"CANCELLED"`로 덮어씀으로써, interruption을 통과하는 엄격한 tool-message alternation을 유지합니다.**
이것은 *placeholder-then-overwrite*이고, boson의 `handler.py`는 *synthesize-on-demand*입니다:
[[boson-interrupt-subsystem]]에 따르면 `InterruptHandler.handle_barge_in`은 다음 user turn에서
`_collect_unanswered_tool_uses`를 훑고, 답이 없는 모든 `tool_use`에 대해
`ToolResultBlock(tool_use_id, content=f"canceled: {tname}")`을 내보내서 Anthropic 식의 엄격한 role
alternation이 살아남게 합니다.

두 개의 mechanism, 그리고 그 사이의 gap은 실재하지만 excerpt가 암시한 것보다 훨씬 작습니다:

| | Pipecat | boson `gateway/interrupt/handler.py` ([[boson-interrupt-subsystem]] 기준) |
|---|---|---|
| tool message가 언제 존재하는가 | call 시작 시점에 `"IN_PROGRESS"`로 | result나 repair가 생성될 때만 |
| barge-in에서 무엇이 그것을 정산하는가 | 제자리에 `"CANCELLED"` 기록 | 다음 user turn에 `ToolResultBlock(content="canceled: {tname}")` 합성 |
| 어떤 call이 정산되는가 | `_cancel_function_call`이 도달하는 것들, 즉 `cancel_on_interruption=True`로 등록된 것만 | scan으로 찾은 답 없는 모든 `tool_use` |
| async / 오래 걸리는 tool | `cancel_on_interruption=False` → 아예 cancel되지 않음; `async_tool_messages`를 통해 정산 | tool별 `_TOOL_CANCEL_HANDLERS`, 없으면 기본 user+assistant 쌍 |
| 사용자에게 보이는 text | 없음 | `"[tool call canceled, user interrupted: {tool_name}]"` |

실제로 포팅해야 할 잔여물(residue)은 *서사(narrative)* 쪽 절반 — boson의 tag 문자열 — 이지 alternation
repair가 아닙니다. [[ch-09/read]]가 tool-loop 충돌의 나머지를 소유하고, 이 chapter가 고치는 것은 그것이
물려받는 전제입니다.

> 💡 **쉬운 설명 — 왜 `IN_PROGRESS` placeholder를 미리 쓰는가**
> Anthropic/OpenAI 계열 API는 `tool_use`가 하나 있으면 그에 대응하는 `tool_result`가 반드시 하나
> 있어야 합니다(strict alternation). barge-in이 tool 실행 도중에 들어오면 결과가 영영 안 옵니다.
> 두 가지 대응이 가능합니다.
> (1) **미리 자리를 잡아둔다** — call을 시작할 때 `content="IN_PROGRESS"` message를 넣어두고, 끝나면
> 진짜 결과로, cancel되면 `"CANCELLED"`로 덮어쓴다. 자리는 항상 있으므로 alternation은 절대 안 깨진다.
> (2) **나중에 메꾼다** — 아무것도 안 써두고, 다음 turn 직전에 "답 없는 tool_use"를 훑어서 그 자리에
> 가짜 결과를 만들어 넣는다.
> Pipecat이 (1), boson이 (2)입니다. (1)은 context에 항상 한 줄이 더 있는 대신 scan이 필요 없고,
> (2)는 깨끗한 대신 scan 로직이 정확해야 합니다.

### 5.3 TTS layer A — service는 자기가 소유한 모든 accumulator를 clear한다

`TTSService.process_frame`의 dispatch:

**`src/pipecat/services/tts_service.py:773-775`**
```python
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)
```

그리고 그 handler는 framework 전체에서 가장 긴 interruption routine입니다. 이것을 *streaming TTS
service가 누적하는 모든 것*의 카탈로그로 읽으십시오. 실제로 그것이기 때문입니다:

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

**14개의 서로 다른 state 조각**을 손으로 clear합니다. 자동으로 해준 것은 아무것도 없습니다. 그중 여섯
개는 무엇을 막아주는지 이름을 붙일 가치가 있습니다:

| Cleared | 안 했다면 |
|---|---|
| `self._text_aggregator.handle_interruption()` | sentence splitter의 미완성 문장이 다음 turn의 첫 문장 앞에 붙는다 |
| `self._aggregated_frame_sequencer.clear()` | 죽은 turn의 pending word slot들이 sequencer에서 다음 turn의 frame을 붙잡는다 |
| `self._pending_llm_response_end_frames.clear()` | 죽은 turn의 붙잡혀 있던 `LLMFullResponseEndFrame`이 나중에 다시 push된다 (§7.6의 경로) |
| `reset_word_timestamps()` | 다음 turn의 word PTS가 옛 baseline 기준으로 계산된다 — 과거로 예약된 단어, 또는 몇 초 뒤 미래의 단어 |
| `_serialization_queue.reset()` | ordering queue가 여전히 죽은 turn의 frame들을 들고 있다 |
| `_word_last_pts = 0` | §7.3의 clock-queue 트릭이 다음 turn의 aggregation frame에 낡은 timestamp를 찍는다 |

`_serialization_queue`는 service 내부에서 `UninterruptibleFrame` mixin이 밥값을 하는 유일한 자리이고,
생성자 주석이 그렇게 말합니다:

**`src/pipecat/services/tts_service.py:402-407`**
```python
        # Created once here so it survives interruptions: on interruption we call reset()
        # which drops non-UninterruptibleFrame items while keeping uninterruptible ones
        # (e.g. FunctionCallResultFrame) that must not be lost mid-flight.
        self._serialization_queue: FrameQueue = FrameQueue(
            frame_getter=lambda item: item if isinstance(item, Frame) else None
        )
```

`FunctionCallResultFrame`은 `DataFrame, UninterruptibleFrame`입니다(`frames.py:770`) — 버려지면 안 되는
*data* frame. 이 조합은 잠시 멈춰 볼 만큼 특이합니다: "줄에서 네 자리는 지키되, 증발하지는 마라"는
뜻입니다. 고객이 끼어드는 동안 도착한 tool result도 여전히 context에 기록되어야 합니다. 안 그러면
§5.2의 `IN_PROGRESS` placeholder가 영영 정산되지 않습니다.

이 routine은 deadlock에 관한 아홉 줄짜리 주석으로 끝나는데, 이 경로에 얼마나 날카로운 모서리가 사는지에
대한 좋은 척도입니다:

**`src/pipecat/services/tts_service.py:1057-1066`**
```python
        # When pause_frame_processing=True, the process task may be blocked at
        # __process_event.wait() because pause_processing_frames() was called
        # after LLMFullResponseEndFrame and an UninterruptibleFrame was dequeued
        # before the interrupt arrived. _start_interruption() in the base class
        # handles the common case (non-uninterruptible frames) by cancelling and
        # recreating the process task. But when _start_interruption() detects an
        # UninterruptibleFrame it only resets the queue, leaving the process task
        # blocked. BotStoppedSpeakingFrame never arrives (no audio played), so we
        # must resume here to prevent a permanent deadlock.
        await self._maybe_resume_frame_processing()
```

`_start_interruption`(§4)의 uninterruptible branch가 [[ch-04/read]] §4.2의 pause machinery와 상호작용해서
hang을 만들어내는 것입니다. 고치는 데 patch 하나와 한 문단짜리 주석이 들었습니다. pause하는 processor를
쓸 때 그만큼의 예산을 잡으십시오.

> 💡 **쉬운 설명 — 저 deadlock을 세 줄로**
> ① uninterruptible frame이 dequeue된 상태에서 `pause_processing_frames()`가 걸려 process task가
> `__process_event.wait()`에 잠들어 있다. ② interruption이 도착하지만, base class는 "지금 처리 중인 게
> uninterruptible이니 task는 건드리지 않고 queue만 비운다"를 택한다 → task는 여전히 잠들어 있다.
> ③ 잠을 깨워줄 `BotStoppedSpeakingFrame`은 재생된 audio가 없어서 오지 않는다 → 영원히 잠든다.
> 그래서 handler 마지막 줄에서 강제로 resume합니다. **"안전한 branch"가 만들어낸 hang**이라는 점이
> 요점입니다 — cancel하지 않는 선택도 공짜가 아닙니다.

### 5.4 TTS layer B — socket bounce, 그리고 실제로 그걸 하는 주체

일부 TTS service는 한 걸음 더 나가서 **provider connection을 버립니다**. 이유는 class docstring에 그대로
적혀 있고, 그것은 Pipecat이 아니라 벤더 서버에 관한 사실입니다:

**`src/pipecat/services/tts_service.py:1969-1974`**
```python
class InterruptibleTTSService(WebsocketTTSService):
    """Websocket-based TTS service that handles interruptions without word timestamps.

    Designed for TTS services that don't support word timestamps. Handles interruptions
    by reconnecting the websocket when the bot is speaking and gets interrupted.
    """
```

**`src/pipecat/services/tts_service.py:1992-2002`**
```python
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

통화 중에 disconnect하고 reconnect합니다. 절반쯤 발화된 server-side synthesis context는 wire 너머로
되감을 수 없기 때문입니다. 벤더에게 "마지막 1.4초는 잊어라"라고 말할 수 없습니다. 사용 가능한 동사는
*끊는다* 하나뿐입니다.

> **Source correction.** [[interruption-cascade]]는 "Websocket TTS subclass들이 한 걸음 더 나간다
> (L1992-2003)"고 말합니다. class hierarchy는 그보다 더 구체적이고, 그 구체성이 흥미로운 부분입니다:
> ```
> $ grep -n "^class " src/pipecat/services/tts_service.py
> 109:  class TTSService(AIService)
> 1882: class WordTTSService(TTSService)
> 1899: class WebsocketTTSService(TTSService, WebsocketService)
> 1969: class InterruptibleTTSService(WebsocketTTSService)          ← the bounce lives here
> 2040: class WebsocketWordTTSService(WebsocketTTSService)
> 2062: class InterruptibleWordTTSService(InterruptibleTTSService)
> 2083: class AudioContextTTSService(WebsocketTTSService)
> 2121: class AudioContextWordTTSService(AudioContextTTSService)
> ```
> `WebsocketTTSService`는 bounce하지 **않습니다**. `InterruptibleTTSService`와 그 subclass만 하고,
> docstring은 이 family가 **word timestamp가 없는** provider를 위해 존재한다고 말합니다. 이 tree에서
> 그것을 상속하는 provider들: `lmnt`, `rime`, `deepgram`, `smallest`, `neuphonic`, `nvidia/sagemaker`,
> `sarvam`, `fish`.

이 생각을 붙들고 계십시오 — §7.6이 회전하는 축과 같은 축입니다. **socket bounce가 필요한 TTS provider는
text가 단어 단위로 도착하지 않는 provider이고, 그 provider들이 바로 Pipecat의 emergent truncation이
all-or-nothing으로 퇴화하는 provider입니다.** 이 둘은 하나의 벤더 capability에서 나오는 두 개의
결과이고, [[ch-07/read]]의 용어로 한국어 TTS 벤더를 고를 때 여러분은 둘을 동시에 고르는 것입니다.

실무 비용: 한국어 TTS 벤더로의 websocket disconnect/reconnect는 TLS handshake 더하기 auth round trip이고,
그것이 하필 **고객이 막 말을 시작했고 봇은 답할 준비가 되어 있어야 하는** 그 창(window)에 정확히
떨어집니다. [[ch-11/read]]의 latency budget에 이 항목을 한 줄 넣어야 합니다.

### 5.5 output transport — clock을 cancel하고, 새 queue를 할당한다

output transport는 audio가 물리적으로 사는 곳이고, 그 interruption handling에는 주의 깊게 읽을 만한
ordering 디테일이 있습니다:

**`src/pipecat/transports/base_output.py:359-361`**
```python
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            await self._handle_frame(frame)
```

**먼저 push, 그 다음 handle.** frame은 이 transport가 자기 queue를 뜯어내기 *전에* assistant aggregator로
forward됩니다. 이건 장식이 아닙니다: aggregator의 commit(§7.4)이 discard보다 *뒤*가 아니라 *앞*에
enqueue된다는 뜻입니다. 다만 이것이 동기적 ordering guarantee를 사주지는 않습니다 — `push_frame`은
이웃의 input queue에 enqueue할 뿐이고, 그것을 실행하는 것은 이웃 자신의 input task입니다. §7.7이 정확히
그 틈으로 무엇이 새는지에 관한 이야기입니다.

routing:

**`src/pipecat/transports/base_output.py:373-384`**
```python
    async def _handle_frame(self, frame: Frame):
        """Handle frames by routing them to appropriate media senders."""
        if frame.transport_destination not in self._media_senders:
            logger.warning(
                f"{self} destination [{frame.transport_destination}] not registered for frame {frame}"
            )
            return

        sender = self._media_senders[frame.transport_destination]

        if isinstance(frame, InterruptionFrame):
            await sender.handle_interruptions(frame)
```

그리고 sender의 teardown:

**`src/pipecat/transports/base_output.py:566-593`**
```python
        async def handle_interruptions(self, _: InterruptionFrame):
            """Handle interruption events by restarting tasks and clearing buffers.

            Args:
                _: The start interruption frame (unused).
            """
            # Cancel tasks.
            await self._cancel_clock_task()
            await self._cancel_video_task()

            if self._audio_queue.has_uninterruptible or self._mixer:
                # Keep the audio task running but drain all interruptible frames
                # so the pending UninterruptibleFrames are still delivered. With
                # a mixer, cancelling the task would also stop mixer-only output
                # during the restart, causing an audible gap in the background
                # audio (made worse by telephony serializers that clear the
                # playout buffer on interruptions).
                self._audio_queue.reset()
            else:
                await self._cancel_audio_task()
                self._create_audio_task()

            # Create tasks.
            self._create_video_task()
            self._create_clock_task()

            # Let's send a bot stopped speaking if we have to.
            await self._bot_stopped_speaking()
```

parameter 이름을 보십시오 — `_` — 그리고 여전히 "start interruption frame"이라고 말하는 docstring도.
더 이상 존재하지 않는 frame(§2.1)의 흔적입니다. frame이 아무것도 실어 나르지 않으므로 handler도
아무것도 필요하지 않습니다.

이 chapter의 나머지에 중요한 teardown은 두 개입니다.

**(a) clock queue는 flush되지 않습니다 — 교체됩니다.**

**`src/pipecat/transports/base_output.py:1067-1077`**
```python
        def _create_clock_task(self):
            """Create the clock/timing processing task."""
            if not self._clock_task:
                self._clock_queue = asyncio.PriorityQueue()
                self._clock_task = self._transport.create_task(self._clock_task_handler())

        async def _cancel_clock_task(self):
            """Cancel and cleanup the clock processing task."""
            if self._clock_task:
                await self._transport.cancel_task(self._clock_task)
                self._clock_task = None
```

`_cancel_clock_task`가 `self._clock_task = None`을 설정하고, 그러면 `_create_clock_task`가 falsy 값을 보고
**완전히 새로운 `asyncio.PriorityQueue()`를 bind합니다.** 옛 queue — presentation time에 아직 도달하지
못한 모든 timestamped frame을 들고 있던 — 는 그냥 참조가 끊기고 GC됩니다. 여기에는 selective flush도
없고 `UninterruptibleFrame` 예외도 없습니다: timed frame이란 정의상 *이제 오지 않을 순간에* 들리도록
예약된 frame이기 때문입니다.

§7.3에 도착했을 때 이걸 기억하십시오. **assistant 문장의 아직 발화되지 않은 모든 단어가 저 queue 안에
있습니다.**

**(b) sub-chunk 나머지는 flush가 아니라 폐기됩니다.**

**`src/pipecat/transports/base_output.py:746-756`**
```python
        async def _bot_stopped_speaking(self):
            """Handle bot stopped speaking event."""
            if not self._bot_speaking:
                return

            self._bot_speaking = False
            self._tts_audio_received = False

            # Any remaining leftover here (e.g. from an interruption) is
            # discarded rather than flushed, since it's no longer wanted.
            self._audio_buffer = bytearray()
```

`handle_audio_frame`은 *완전한* chunk만 enqueue하므로, 최대 chunk 하나 분량의 PCM이 `_audio_buffer`에
남아 있을 수 있습니다. 깨끗한 정지에서는 silence로 padding해서 flush하고(`handle_tts_stopped`,
`:659-672`), interruption에서는 버립니다. 이것은 의도적이고, 주석까지 달린, 40밀리초 미만짜리
결정입니다 — 이 경로가 얼마나 세밀하게 튜닝됐는지에 대한 좋은 지표입니다.

### 5.6 여러분의 process 너머 — 통신사의 playout buffer

이제 내부에서 아무리 flush해도 닿을 수 없는 부분입니다. 이미 wire에 써 보낸 byte는 *다른 누군가의*
buffer 안에 있습니다. Twilio가 갖고 있습니다. 고객의 단말기가 갖고 있을 수도 있습니다. 여러분의 queue를
비우는 것은 이미 떠난 audio에 대해 아무것도 하지 못합니다.

그래서 모든 telephony serializer는 carrier-side flush command를 내보내고, carrier마다 철자가 다릅니다:

| Serializer | Line | `InterruptionFrame`에서 내보내는 것 |
|---|---|---|
| `serializers/twilio.py` | `:186-188` | `{"event": "clear", "streamSid": self._stream_sid}` |
| `serializers/plivo.py` | `:138-140` | `{"event": "clearAudio", "streamId": self._stream_id}` |
| `serializers/telnyx.py` | `:150-151` | `{"event": "clear"}` |
| `serializers/exotel.py` | `:98-99` | `{"event": "clear", "stream_sid": self._stream_sid}` |
| `serializers/vonage.py` | `:90-92` | `{"action": "clear"}` |
| `serializers/genesys.py` | `:602-603` | `json.dumps(self.create_barge_in_event())` |

> **Source correction, 사소한 것.** [[interruption-cascade]]와 [[theory-out-of-band-priority]]는
> `telnyx.py:150`, `vonage.py:90`, `exotel.py:98`을 *대입(assignment)* 줄로 인용합니다. 이 commit에서
> 그 줄들은 `elif isinstance(frame, InterruptionFrame):` 줄이고, 대입은 한 줄 뒤입니다(`:151`, `:92`,
> `:99`). 두 excerpt 모두 `plivo.py`와 `genesys.py`는 아예 언급하지 않는데, 둘 다 이것을 처리합니다.

Twilio 것을 그대로 봅니다. Lina가 US 프론트 carrier로 발신한다면 실제로 그 위에 ship하게 될 shape이기
때문입니다:

**`src/pipecat/serializers/twilio.py:186-188`**
```python
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
```

Genesys가 예외적이고 볼 가치가 있습니다. protocol 자체가 이 개념에 이름을 붙였기 때문입니다:

**`src/pipecat/serializers/genesys.py:470-486`**
```python
    def create_barge_in_event(self) -> dict[str, Any]:
        """Create a barge-in event message.

        This notifies Genesys Cloud that the user has interrupted the bot's
        audio output. Genesys will stop any queued audio playback.

        Returns:
            Dictionary of the barge-in event message.
        """
        msg = self._create_message(
            AudioHookMessageType.EVENT,
            parameters={"entities": [{"type": "barge_in", "data": {}}]},
        )

        logger.debug("🔇 Barge-in event sent to Genesys")

        return msg
```

여섯 개의 serializer, 여섯 개의 wire format, 하나의 개념. 그리고 그 개념은 **여러분 권한의 경계**입니다:
**cascade는 여러분의 process 가장자리에서 멈추고, 그 너머로는 부탁할 수 있을 뿐입니다.** carrier가 그
clear를 존중하는지, 얼마나 빨리 하는지는 이 repo 안에 없습니다. Pipecat에 serializer가 없는 한국
carrier나 SBC 위의 Lina에게는, 이 표의 그 행은 *여러분이 직접 써야 할* 것입니다 —
`InterruptionFrame`에서 여러분 carrier의 flush command를 `serialize()`가 반환하는 `FrameSerializer`
subclass. [[ch-05/read]]이 interface를 줬고, 이것이 barge-in이 의존하는 그 interface의 한 method입니다.

---

## 6. chunk size는 interrupt-granularity 결정이다

[[ch-04/read]] §5.1이 산수를 줬습니다. 여기가 그것을 쓰는 자리이고, 그것이 진술된 방식의 한 디테일을
정정하는 자리입니다.

### 6.1 상수들

**`src/pipecat/transports/base_output.py:132-136`**
```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

**`src/pipecat/transports/base_transport.py:72`**
```python
    audio_out_10ms_chunks: int = 4
```

10밀리초짜리 chunk 네 개: **쓰기 한 번당 40 ms의 PCM.** 주석이 조용한 부분을 소리 내어 말합니다 —
*"This will help with interruption handling."* 이것은 interruption knob으로 분장한 buffering knob이
아닙니다. source가 그것을 그것이라고 부릅니다.

### 6.2 실제로 audio queue를 비우는 것 — 정정

[[theory-out-of-band-priority]]와 이 chapter가 명세된 outline은 둘 다 `_audio_queue`가
"`_clock_task_handler`(`base_output.py:1079`)에 의해 realtime으로 drain된다"고 말합니다. **아닙니다.**
`MediaSender` 안에는 독립적인 두 개의 queue가 있고, 서로 다른 것을 실어 나릅니다:

| Queue | Created | Drained by | Carries |
|---|---|---|---|
| `_audio_queue` | `base_output.py:690` (`FrameQueue()`) | `_audio_task_handler` (`:896`) via `_next_frame` (`:829`) | `OutputAudioRawFrame` chunks + sync frames |
| `_clock_queue` | `base_output.py:1070` (`asyncio.PriorityQueue()`) | `_clock_task_handler` (`:1079`) | frames with a `pts` — the word-level `TTSTextFrame`s of §7.3 |

audio task는 queue가 공급하는 만큼 빠르게 당겨 갑니다:

**`src/pipecat/transports/base_output.py:836-843`**
```python
            async def without_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
                while True:
                    try:
                        frame = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=vad_stop_secs
                        )
                        yield frame
                        self._audio_queue.task_done()
                    except TimeoutError:
                        # Fallback: notify the bot stopped speaking upstream if necessary based on timeout.
                        await self._bot_stopped_speaking()
```

realtime pacing은 한 층 아래, transport의 `write_audio_frame`에 있습니다. websocket server transport를
보면:

**`src/pipecat/transports/websocket/server.py:474-480`**
```python
        if not await self._write_frame(frame):
            return False

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

        return True
```

**`src/pipecat/transports/websocket/server.py:506-515`**
```python
    async def _write_audio_sleep(self):
        """Simulate audio device timing by sleeping between audio chunks."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval
```

interval은 setup에서 설정됩니다:

**`src/pipecat/transports/websocket/server.py:379`**
```python
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
```

저 식은 처음 읽으면 틀려 보이므로 단위를 따져 보십시오. `audio_chunk_size`는 **byte** 단위입니다.
16-bit mono에서는 sample당 2 byte이므로 `audio_chunk_size / sample_rate`는 chunk 길이(초)의 *두 배*이고,
`/ 2`가 그것을 바로잡습니다. default에서: `audio_chunk_size = (24000/100) × 1 × 2 × 4 = 1920` byte;
`1920 / 24000 = 0.08`; `/2 = 0.04` s. **40 ms — 정확히 chunk 하나의 길이.** sleep이 write를 realtime으로
pacing하고, *거기에* `r`이 삽니다.

이 정정은 결론을 바꾸지 않습니다. residency를 어디서 찾아야 하는지를 바꿉니다. **N은 한 곳이 아니라 세
곳에 흩어져 있습니다**: `_audio_queue`, transport의 write path가 받아들였지만 아직 보내지 않은 것,
그리고 carrier의 playout buffer. 이 중 앞의 둘만 여러분 것입니다.

### 6.3 Lina 문장 하나에 대해 산수하기

Lina가 *"고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금 가입하시면…"*을 말하는 중입니다.
3초짜리 발화라고 합시다. 여러분의 한국어 TTS 벤더는 그 전체를 websocket으로 대략 400 ms 만에
스트리밍해 줬습니다 — 벤더들은 그렇게 합니다. audio가 네트워크 대역보다 작기 때문입니다.

- 3 s ÷ 40 ms = **약 75개 chunk**가 residency에 있고, 거기에 carrier가 들고 있는 것이 더해집니다.
- drain rate `r`은 물리학이 정합니다. 스피커에서 소리가 나가는 속도입니다. 여러분이 튜닝할 수 없습니다.
- 그 chunk들 **뒤에** 붙은 in-band stop signal은 `N/r` ≈ **3초 늦게** 전달됩니다 — handler가 아무리
  빨라도. `return` 한 줄짜리 handler도 여전히 3초 늦습니다.

그것이 priority channel에 대한 논증 전부이고, 그것은 산수이지 허술함이 아닙니다. 그리고 그것이
`audio_out_10ms_chunks`가 진짜 제품 결정인 이유이기도 합니다: `2`(20 ms chunk)로 반토막 내면 write-loop
반복 횟수가 두 배가 되고 — syscall이 더 많아지고, `asyncio.sleep` wakeup이 더 많아지고 — 그 대가로
비행 중인 마지막 chunk의 cancellation granularity가 더 미세해집니다. [[ch-11/read]]이 그 숫자를 지출하고,
여기서는 그것이 buffering 취향이 아니라는 것만 알면 됩니다.

> 💡 **쉬운 설명 — `N/r`을 한 번 더**
> `N` = 이미 줄 서 있는 audio의 양(초), `r` = 그 audio가 소비되는 속도(= 실시간, 1). in-band signal은
> 줄 맨 뒤에 서므로 `N/r`초 뒤에 처리됩니다. 3초치 audio가 buffer에 있으면 "멈춰"는 3초 뒤에야
> 읽힙니다 — 그때는 이미 봇이 다 말해버린 뒤입니다. 코드를 최적화해서 줄일 수 있는 값이 아니라
> **queue에 얼마나 쌓여 있느냐**에만 달린 값이고, 그래서 해법은 "빠른 handler"가 아니라
> "줄을 안 서는 channel"뿐입니다.

→ **[bargein-timeline.html](figures/bargein-timeline.html)** — 지금 열어서 여기서 §9까지 옆에 두십시오.
CASCADE 패널에서 시작하십시오: Lina의 3초짜리 한국어 문장을 따라 interruption 지점을 드래그하면서
독립적인 hop들이 발화하는 것을 보십시오 — `broadcast_frame`이 sibling으로 연결된 두 instance를 만들고,
각 processor가 자기 task를 cancel하고, `FrameQueue.reset()`이 `UninterruptibleFrame`인 `EndFrame`을
남기고, `tts_service.py:1041`에서 TTS accumulator가 clear되고, output transport가 새 `PriorityQueue`를
bind하고, Twilio의 `{"event": "clear"}`가 여러분의 process 너머로 나가는 것. 그 다음 §4.1의
accumulate-then-flush 장난감을, cancellation 지점을 frame A와 B 사이로 드래그해서, 먼저 reset handler
없이, 그 다음 있는 채로 돌려 보십시오. 범위에 대한 두 가지 note: 이 figure는 **이미 존재하는
interruption signal에서 시작합니다** — 그 안에 VAD control은 전혀 없습니다. 그 layer는
[[ch-06/read]]의 `turn-boundary.html`에서 만들어져 interactive해졌고 caption이 그쪽을 가리킵니다 —
그리고 in-band/out-of-band 오버레이는 [[ch-04/read]]의 `N/r` 계산기를 재유도하는 대신 재사용합니다.

---

## 7. 핵심: truncation routine은 없다

지금까지는 전부 소리를 멈추는 일에 관한 것이었습니다. 이 section은 conversation history에 관한 것이고,
이 chapter가 존재하는 이유인 그 하나의 아이디어입니다.

사람들이 기술하는 동작은 이것입니다: *"Pipecat은 assistant의 message를 사용자가 실제로 들은 만큼으로
truncate한다."* **동작은 실재합니다. routine은 존재하지 않습니다.** 다른 이름으로도, helper module에도,
context class 깊숙이에도 없습니다. 존재하는 것은 *list 안의 위치* 하나와 *audio clock* 하나이고,
truncation은 그 둘을 interrupt했을 때 나오는 결과물입니다.

### 7.1 부재의 증거, 실행

부재를 증명하는 것부터 시작합니다. 부재 주장은 검증 가능해야 하기 때문입니다.

```
$ grep -n "interrupt" src/pipecat/processors/aggregators/llm_context.py
$ echo $?
1
$ wc -l src/pipecat/processors/aggregators/llm_context.py
510 src/pipecat/processors/aggregators/llm_context.py
```

**510줄에서 hit 0.** `LLMContext` — message와 tool과 tool choice를 들고 있는 class이자, 모든 LLM service가
모든 inference마다 읽는 객체 — 는 interruption이 존재한다는 사실을 모릅니다. 그것은 **설계상**
interruption-unaware이고, 실수가 아니라 의도된 layering 결정입니다: context는 자료구조이고, turn-boundary
policy는 aggregator에 삽니다.

이제 아깝게 빗나가는 것 하나. `grep truncat`은 실제로 이 파일에 걸리고, 왜 그것이 해당되지 않는지
알아야 합니다:

**`src/pipecat/processors/aggregators/llm_context.py:221-260`**
```python
    def get_messages(
        self,
        llm_specific_filter: str | None = None,
        *,
        truncate_large_values: bool = False,
    ) -> list[LLMContextMessage]:
        """Get the current messages list.

        Args:
            llm_specific_filter: Optional filter to return LLM-specific
                messages for the given LLM, in addition to the standard
                messages. If messages end up being filtered, an error will be
                logged; this is intended to catch accidental use of
                incompatible LLM-specific messages.
            truncate_large_values: If True, return deep copies of messages with
                large values shortened. For standard messages, known binary
                data (base64-encoded images, audio) is replaced with short
                placeholders. For LLM-specific messages, long string values
                are truncated.
        ...
        if truncate_large_values:
            messages = LLMContext._truncate_large_values_from_messages(messages)

        return messages
```

`:315`의 `_truncate_long_strings(value, *, max_length: int = 100)`. tree 안의 모든 caller가
`truncate_large_values=True`를 넘기는 곳은 정확히 한 종류입니다:

**`src/pipecat/adapters/services/open_ai_adapter.py:230-236`**
```python
        Returns:
            List of messages in a format ready for logging about OpenAI.
        """
        return cast(
            list[dict[str, Any]],
            self.get_messages(context, truncate_large_values=True),
        )
```

*"in a format ready for logging."* `LLMContext`에 있는 유일한 "truncate"는 log 줄이 읽히도록 base64
덩어리를 줄이는 것입니다. barge-in과는 아무 관계가 없습니다.

그래서: truncation routine 없음, context에 interruption 인식 없음. 그러면 그 동작은 어디서 나옵니까?

### 7.2 사실 하나 — aggregator는 output transport 뒤에 앉는다

**`examples/getting-started/06-voice-agent.py:81-91`**
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

[[ch-04/read]] §11은 이 list가 house pattern이고, 7번 위치가 *그 지점에만 증거가 존재한다*는 사실로
고정되어 있다고 이미 말했습니다. 이제 그것을 현금화합니다: **assistant aggregator가 볼 수 있는 유일한
text는 output transport가 이미 내보낸 text입니다.** LLM이 생성한 text가 아닙니다. TTS에게 말하라고
요청한 text도 아닙니다. 스피커를 소유한 그것의 반대편으로 나온 text입니다.

`assistant_aggregator`를 5번 위치로 — `transport.output()` 앞으로 — 옮기면, 그것은 LLM이 생산한 모든
단어를 충실히 기록합니다. 고객이 잘라버린 열두 단어까지 포함해서. 아무것도 error를 내지 않습니다.
아무것도 warning하지 않습니다. pipeline은 여전히 type-correct합니다. [[ch-01/read]] §8의 요점이
여전히 유효하기 때문입니다: *"어떤 processor든 어디에나"는 type 수준의 진실이자 semantic 수준의
거짓말이다.* 이것이 framework 전체에서 그 거짓말의 가장 비싼 사례이고, call site에서는 보이지 않습니다.

**Truncation은 list 안 위치의 property입니다.**

### 7.3 사실 둘 — assistant text는 audio clock에 의해 pacing된다

위치만으로는 부족합니다. 만약 단어가 TTS가 생산하는 속도로 7번 위치에 도착한다면, aggregator는 문장
전체를 그것이 들리기 훨씬 전에 들고 있게 되고, interruption은 그 전부를 commit해버릴 것입니다. 그래서
mechanism의 나머지 절반은 word 수준 text가 **audio를 내보내는 바로 그 clock에 의해 timestamp되고
release된다**는 것입니다.

한국어 단어 하나를 따라가 봅시다.

**Step 1 — TTS service가 각 단어에 presentation timestamp를 찍습니다.**

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

`_initial_word_timestamp`는 **첫 audio chunk**에서 확립되는 baseline입니다 — `start_word_timestamps()`가
`:1737-1740`의 `_handle_audio_context`에서 `if not timestamps_started` 가드와 함께 호출됩니다. 즉 word
clock은 벽시계가 아니라 **audio에 고정(anchor)되어** 있습니다.

`TTSTextFrame`은 `frames.py:417`의 `AggregatedTextFrame(TextFrame(DataFrame))`입니다 — 평범한 data frame,
priority 20, interruptible, `pts`를 실어 나름.

**Step 2 — output transport는 `pts`가 있는 것은 무엇이든 clock queue로 보냅니다.**

**`src/pipecat/transports/base_output.py:395-400`**
```python
        elif isinstance(frame, TTSStoppedFrame):
            await sender.handle_tts_stopped(frame)
        elif frame.pts:
            await sender.handle_timed_frame(frame)
        else:
            await sender.handle_sync_frame(frame)
```

**`src/pipecat/transports/base_output.py:643-649`**
```python
        async def handle_timed_frame(self, frame: Frame):
            """Handle frames with presentation timestamps.

            Args:
                frame: The frame with timing information to handle.
            """
            await self._clock_queue.put((frame.pts, next(self._clock_queue_counter), frame))
```

**Step 3 — clock task가 그 단어가 들릴 때까지 자고, 그 다음 push합니다.**

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

`await asyncio.sleep(wait_time)` — 단어는 그것이 들리는 순간까지 붙잡혀 있다가, 그때에야 다음
processor로 push되고, 그 다음 processor가 assistant aggregator입니다.

**이것이 트릭의 전부입니다.** 단어는 그것이 들리게 되는 바로 그 순간에 history를 쓰는 그것에 도달합니다.
그리고 §5.5(a)가 interruption이 저 queue에 무슨 짓을 하는지 말했습니다: flush하지 않고, *교체합니다*.
영영 오지 않을 순간을 위해 예약된 모든 단어는 queue 객체와 함께 참조가 끊깁니다.

**Step 4 — 그리고 framework는 flush frame을 단어들 *뒤에* 두려고 일부러 애를 씁니다.**

이것이 이 design이 운이 좋은 것이 아니라 의도적이라는 것을 증명하는 조각입니다. TTS context가 끝나면
service는 aggregator에게 commit하라고 말하는 frame을 push합니다. 그 frame에는 자연스러운 `pts`가 없으므로
*audio* queue 경로를 타게 되고 마지막 word frame들을 추월할 수 있습니다 — 마지막 단어들이 도착하기 전에
commit해버리는 것입니다. 그래서 service가 거기에 stamp를 찍습니다:

**`src/pipecat/services/tts_service.py:920-933`**
```python
        if isinstance(frame, TTSStoppedFrame) and frame.context_id:
            if frame.context_id in self._tts_contexts:
                if self._tts_contexts[frame.context_id].push_assistant_aggregation:
                    aggregation_frame = LLMAssistantPushAggregationFrame()
                    # When word-level TTSTextFrames are routed through the
                    # transport's clock queue (PTS-based), the aggregation frame
                    # would otherwise take the audio (sync) queue path and
                    # could overtake the final word frames. Stamping it with a
                    # PTS just past the last word forces it through the clock
                    # queue too, so the assistant aggregator sees every word
                    # before flushing.
                    if self._word_last_pts:
                        aggregation_frame.pts = self._word_last_pts + 1
                    await self.push_frame(aggregation_frame)
```

`self._word_last_pts + 1`. 마지막 단어보다 1 나노초 뒤. 누군가 이 버그를 발견하고 `+ 1`로 고쳤으며,
queue race를 설명하는 일곱 줄짜리 주석을 남겼습니다. 같은 트릭이 열두 줄 아래의 segment-announcement
frame에도 적용되어 있습니다(`:950-964`).

저 주석을 우연히 쓰는 사람은 없습니다. audio로 pacing되는 aggregation은 하중을 받는 구조재이고,
알려져 있으며, 방어되고 있습니다.

> 💡 **쉬운 설명 — "+1 나노초"가 왜 필요한가**
> queue가 두 개입니다. 단어들은 `pts`가 있어서 **clock queue**를 타고, "이제 commit해"라는 aggregation
> frame은 `pts`가 없어서 **audio(sync) queue**를 탑니다. 두 queue는 서로 순서를 보장하지 않으므로,
> 아무것도 안 하면 "commit해"가 마지막 단어보다 먼저 도착할 수 있습니다 → 마지막 몇 단어가 history에서
> 사라집니다. 해법은 aggregation frame에도 `pts`를 억지로 붙여서 **같은 queue에 태우는 것**이고,
> 값은 "마지막 단어 + 1 ns"면 충분합니다. 두 줄이 하나의 줄이 되는 순간 순서 문제는 사라집니다.

### 7.4 interruption에서 aggregator가 실제로 하는 일

이제 네 개의 method를, 실행되는 순서대로.

**Dispatch** — 그리고 이것이 파일 어디에 착지하는지 보십시오: `InterruptionFrame`은 `SystemFrame`이므로
[[ch-04/read]] §4.2에 따라 이것은 **input task 위에서**, inline으로, 이 processor에 줄 서 있는 모든
priority-20 frame보다 앞서 실행됩니다.

**`src/pipecat/processors/aggregators/llm_response_universal.py:1545-1554`**
```python
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
```

1545의 `super().process_frame`이 먼저 `_start_interruption()`을 실행합니다 — 이 aggregator 자신의
process task, 즉 `_handle_text` 도중이었을 수도 있는 그 task를 cancel합니다. 그 다음 1553번 줄.

**handler** — 두 줄:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1723-1725`**
```python
    async def _handle_interruptions(self, frame: InterruptionFrame):
        await self._trigger_assistant_turn_stopped(interrupted=True)
        await self.reset()
```

**turn-stop** — 그리고 순서를 주의 깊게 읽으십시오. §7.8에서 중요해집니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:2175-2197`**
```python
    async def _trigger_assistant_turn_stopped(self, *, interrupted: bool = False):
        if not self._assistant_turn_start_timestamp:
            return

        aggregation = await self.push_aggregation()
        if aggregation:
            # Strip turn completion markers from the transcript
            aggregation = self._maybe_strip_turn_completion_markers(aggregation)

        message = AssistantTurnStoppedMessage(
            content=aggregation,
            interrupted=interrupted,
            timestamp=self._assistant_turn_start_timestamp,
        )
        await self._call_event_handler("on_assistant_turn_stopped", message)
        if aggregation:
            await self.broadcast_frame(
                LLMContextAssistantTurnFrame,
                text=aggregation,
                timestamp=self._assistant_turn_start_timestamp,
            )

        self._assistant_turn_start_timestamp = ""
```

**commit** — history를 쓰는 그 줄:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1677-1694`**
```python
    async def push_aggregation(self) -> str:
        """Push the current assistant aggregation with timestamp."""
        if not self._aggregation:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()

        self._context.add_message({"role": "assistant", "content": aggregation})

        # Push context frame
        await self.push_context_frame()

        # Push timestamp frame with current time
        timestamp_frame = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(timestamp_frame)

        return aggregation
```

그리고 애초에 `self._aggregation`을 채운 것 — release된 모든 `TextFrame`이 하나씩 append된 것:

**`src/pipecat/processors/aggregators/llm_response_universal.py:2051-2072`**
```python
    async def _handle_text(self, frame: TextFrame):
        # Skip TextFrame types not intended to build the assistant context
        if isinstance(frame, (TranscriptionFrame, TranslationFrame, InterimTranscriptionFrame)):
            return

        if not frame.append_to_context:
            return

        # Make sure we really have text (spaces count, too!)
        if len(frame.text) == 0:
            return

        text = (
            frame.raw_text
            if isinstance(frame, AggregatedTextFrame) and frame.raw_text
            else frame.text
        )
        self._aggregation.append(
            TextPartForConcatenation(
                text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
            )
        )
```

`push_aggregation`의 1685번 줄을 한 번 더 읽으십시오:

```python
        self._context.add_message({"role": "assistant", "content": aggregation})
```

**이게 전부입니다.** 평범한 assistant message. `[interrupted]` tag 없음. 말줄임표 없음. `interrupted`
field 없음. metadata 없음. *interrupt된* turn과 *완결된* turn은 byte 단위로 동일한 message shape을
만들어냅니다. 유일한 차이는 한쪽이 더 짧다는 것뿐입니다.

그리고 아무것도 aggregate되지 않았다면 — 첫 단어가 들리기 전에 고객이 끼어들었다면 — 1679번 줄이 맨
위에서 `""`를 반환하고 **`add_message`는 아예 호출되지 않습니다**. assistant turn은 history에 어떤
형태로도 나타나지 않습니다. 다음 inference에서 모델의 관점에서, **Lina는 아무 말도 하지 않았습니다.**

### 7.5 test를 돌려보기 — 문자 그대로의 message list

이 중 아무것도 제 말만 믿을 필요가 없습니다. Pipecat이 두 번 assert합니다.

**`tests/test_context_aggregators_universal.py:1319-1348`**
```python
        frames_to_send = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            SleepFrame(),
            InterruptionFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            LLMTextFrame("there!"),
            LLMFullResponseEndFrame(),
        ]
        expected_down_frames = [
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
            InterruptionFrame,
            LLMContextFrame,
            LLMContextAssistantTimestampFrame,
            LLMContextAssistantTurnFrame,
        ]
        await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertEqual(should_start, 2)
        self.assertEqual(should_stop, 2)
        self.assertTrue(stop_messages[0].interrupted)
        self.assertEqual(stop_messages[0].content, "Hello")
        self.assertFalse(stop_messages[1].interrupted)
        self.assertEqual(stop_messages[1].content, "Hello there!")
```

두 개의 turn, 두 번의 `on_assistant_turn_stopped` event, 그리고 첫 번째는 `interrupted=True`와
`content="Hello"`를 실어 나릅니다(뒤의 공백은 concatenation에서 먹힙니다). `expected_down_frames`를
보십시오: interrupt된 turn에 대해서도 완전한 `LLMContextFrame`이 push됩니다 — interrupt된 부분 문장이
context로 들어가고 곧장 LLM으로 되돌아갑니다. 완결된 turn과 똑같이.

두 번째 test는 더 직접적입니다. **문자 그대로의 message list**를 assert하기 때문입니다:

**`tests/test_context_aggregators_universal.py:2434-2451`**
```python
        frames_to_send = [
            TranscriptionFrame(text="Hi!", user_id="", timestamp="now"),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            SleepFrame(),
            InterruptionFrame(),
        ]
        await run_test(
            Pipeline([user, assistant]),
            frames_to_send=frames_to_send,
        )

        roles_contents = [(m["role"], m["content"]) for m in context.get_messages()]
        # User message written when assistant started; assistant message
        # written immediately on interruption with interrupted=True.
        self.assertEqual(roles_contents, [("user", "Hi!"), ("assistant", "Hello")])
        self.assertEqual(len(assistant_messages), 1)
        self.assertTrue(assistant_messages[0].interrupted)
```

```
[("user", "Hi!"), ("assistant", "Hello")]
```

저것이 다음 turn에 모델이 보는 context입니다. `"Hello…"`가 아닙니다. `"Hello [interrupted]"`가
아닙니다. `"Hello"`입니다.

### 7.5.1 `interrupted=True`가 실제로 사는 곳, 그리고 그걸 볼 수 있는 사람

flag는 실재합니다. 다만 LLM이 읽는 그 어디에도 없을 뿐입니다.

**`src/pipecat/processors/aggregators/llm_response_universal.py:327-345`**
```python
@dataclass
class AssistantTurnStoppedMessage:
    """An assistant turn stopped message containing an assistant transcript update.

    A message in a conversation transcript containing the assistant
    content. This is the aggregated transcript that is then used in the context.

    Parameters:
        content: The message content/text. May be empty if the LLM
            returned zero tokens (e.g. turn was interrupted before any tokens
            were received or pushed)
        interrupted: Whether the assistant turn was interrupted.
        timestamp: When the assistant turn started.

    """

    content: str
    interrupted: bool
    timestamp: str
```

일회성 event payload이고, `on_assistant_turn_stopped` handler에게 건네집니다. transcript observer,
RTVI client, 여러분의 CRM logger — 전부 이걸 받습니다. `LLMContext`는 받지 못합니다.

turn text를 다른 processor로 실어 나르는 frame도 마찬가지입니다:

**`src/pipecat/frames/frames.py:531-546`**
```python
@dataclass
class LLMContextAssistantTurnFrame(DataFrame):
    """The aggregated text of a completed assistant turn.

    Broadcast by the LLM assistant aggregator when a turn ends, carrying the
    same text that is stored in the LLM context. Processors upstream and
    downstream (e.g. STT services) can handle this frame to react to each
    completed bot reply without needing a separate observer.

    Parameters:
        text: The assistant's aggregated spoken text for this turn.
        timestamp: ISO-8601 timestamp of when the assistant turn started.
    """

    text: str
    timestamp: str
```

`text`와 `timestamp`. **`interrupted` field 없음.** tree 안의 유일한 소비자는 STT service입니다:

**`src/pipecat/services/stt_service.py:513-515`**
```python
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            await self._process_assistant_turn(frame.text)
            await self.push_frame(frame, direction)
```

즉 봇의 마지막 발화로 자기 자신을 conditioning하는 STT service도, 그 발화가 완결됐는지 잘렸는지 알
방법이 없습니다.

"이 turn은 interrupt됐다"는 사실이 표현 가능한 곳의 요약:

| Carrier | `interrupted`가 있는가? | 누가 읽는가 |
|---|---|---|
| `AssistantTurnStoppedMessage` (`:328`) | **있음** | `on_assistant_turn_stopped` handler — transcript observer |
| context message (`:1685`) | 없음 | **LLM이, 이후의 모든 turn에서** |
| `LLMContextAssistantTurnFrame` (`frames.py:533`) | 없음 | `STTService._process_assistant_turn` |
| `InterruptionFrame` (`frames.py:1142`) | field 자체가 없음 | 모든 processor가, 일시적으로 |

**그 사실을 가장 필요로 하는 소비자가, 그것을 얻을 수 없는 유일한 소비자입니다.**

### 7.6 timestamp 없는 branch — 문장 단위 all-or-nothing

§7.3의 모든 것은 word timestamp를 가정했습니다. TTS 생태계의 절반은 그것을 내보내지 않고, Pipecat은 그런
provider에 대해 완전히 다른 경로를 탑니다. 이것이 여러분의 한국어 벤더 선택을 결정할 branch이므로 꼼꼼히
읽으십시오.

**`src/pipecat/services/tts_service.py:1322-1341`**
```python
        if self._push_text_frames and not self._is_streaming_tokens:
            # In TTS services that support word timestamps, the TTSTextFrames
            # are pushed as words are spoken. However, in the case where the TTS service
            # does not support word timestamps (i.e. _push_text_frames is True), we send
            # the original (non-transformed) text after the TTS generation has completed.
            # This way, if we are interrupted, the text is not added to the assistant
            # context and the context that IS added does not include TTS-specific tags
            # or transformations.
            #
            # In streaming (TOKEN) mode this is handled instead by the sequencer's
            # per-sentence promotion (see AggregatedFrameSequencer._promote): a call
            # here represents a single token, not the sentence-level unit this frame
            # should carry.
            frame = TTSTextFrame(text, aggregated_by=type)
            frame.will_be_spoken = True
            frame.includes_inter_frame_spaces = includes_inter_frame_spaces
            frame.context_id = context_id
            frame.append_to_context = append_tts_text_to_context
            # Appending to the context, so it preserves the ordering.
            await self.append_to_audio_context(context_id, frame)
```

1326-1329번 줄의 주석을 두 번 읽으십시오: *"we send the original (non-transformed) text **after** the TTS
generation has completed. This way, **if we are interrupted, the text is not added to the assistant
context**."*

framework가 자기 입으로, 여기서는 mechanism이 다르다고 말하고 있습니다. word timestamp가 없으면 pacing할
대상이 없으므로, 문장 전체의 `TTSTextFrame`이 audio-context serialization queue에서 그 문장의 audio 뒤에
줄 서 있다가 synthesis가 끝나면 release됩니다. 그 전 어느 지점에서든 interrupt하면
`_handle_interruption`의 `_stop_audio_context_task()` + `_serialization_queue.reset()`(§5.3)이 그 frame을
버립니다.

**따라서 truncation의 granularity는 문장이고, 규칙은 all-or-nothing입니다.**

두 경로를 나란히 놓으십시오. 같은 barge-in, 같은 순간, 다른 벤더:

| | Word-timestamp provider | Timestamp 없는 provider (`InterruptibleTTSService` family) |
|---|---|---|
| aggregator가 받는 것 | 단어당 `TTSTextFrame` 하나, clock task가 그 `pts`에 release | **문장**당 `TTSTextFrame` 하나, synthesis 완료 후 release |
| 문장 3의 60 % 지점에서 interrupt | 문장 1–2 전체 + 재생된 3의 단어들 | 문장 1–2 전체; **문장 3은 아무것도 기여하지 않음** |
| 문장 3의 95 % 지점에서 interrupt | 문장 1–2 + 3의 거의 전부 | 문장 1–2; **문장 3은 여전히 아무것도 기여하지 않음** |
| 문장 1 도중에 interrupt | 재생된 1의 단어들 | **assistant message 자체가 없음** (`push_aggregation`이 `""` 반환) |
| barge-in 시 socket 처리 | 필요 없음 | 완전한 `_disconnect()` / `_connect()` bounce (§5.4) |

굵게 표시된 두 행이 벤더 미팅에 들고 갈 것입니다. timestamp 없는 TTS에서, Lina가 세 문장짜리 65세 갱신
설명을 읽다가 세 번째 문장 중간에 잘리면, 기록에는 *"…65세까지 갱신 없이 보장이 됩니다."*로 남습니다.
마치 세 번째 문장을 시작조차 하지 않은 것처럼 — compliance transcript로서는 고객이 실제로 들은 것과
실질적으로 다른 기록입니다.

이것이 또한 [[ch-07/read]]이 남긴 질문의 구체적 형태입니다. "당신의 한국어 TTS 벤더가 word timestamp를
내보내는가?"는 있으면 좋은 것이 아닙니다. 그것이 **두 개의 truncation mechanism 중 어느 것을 갖게 되는지**,
그리고 모든 barge-in마다 websocket reconnect 비용을 치를 것인지를 결정합니다.

> 💡 **쉬운 설명 — 왜 timestamp가 없으면 문장 전체가 사라지는가**
> word timestamp가 있으면 "이 단어는 재생 시작 후 1.24초에 들린다"를 알 수 있으므로, 그 시각에 맞춰
> 단어를 하나씩 aggregator에 흘려보낼 수 있습니다. interrupt 시점까지 흘러간 단어들 = 들린 말 = 기록.
> timestamp가 없으면 그 시각을 알 수 없으니 단어를 언제 흘려보낼지 정할 수 없습니다. 그래서 Pipecat은
> 안전한 쪽을 택합니다: **synthesis가 끝난 뒤에 문장 전체를 한 번에 보낸다.** 그 결과 문장 도중에
> 끊기면 아직 보내지 않았으므로 통째로 버려집니다. 과소 기록(under-record)이지 과대 기록이 아니라는
> 점에서 방향은 안전하지만, "95 %를 들려주고 0 %를 기록한다"는 결과는 compliance 관점에서는 안전하지
> 않습니다.

### 7.7 mechanism이 닫을 수 없는 one-hop race

주장하는 것이 아니라 machinery에서 유도되는 property 하나가 더 있고, 이것이 그 보장에 대한 가장 날카로운
한계입니다.

`_start_interruption`(§4)은 `__process_queue`를 reset합니다. `__input_queue`는 **건드리지 않습니다.**
막 들리게 된 단어 하나에 대해 그게 무슨 뜻인지 추적하십시오:

1. `_clock_task_handler`가 그 단어의 presentation time에 도달해서
   `self._transport.push_frame(frame)`을 호출합니다(`base_output.py:1098`). **고객이 그 단어를 듣습니다.**
2. `push_frame` → `__internal_push_frame` → `self._next.queue_frame(frame, DOWNSTREAM)` — aggregator의
   **input queue**에, `DEFAULT_PRIORITY = 20`으로.
3. aggregator의 input task가 그것을 pop해서 `__process_queue`로 relay하고, process task가 `_handle_text`를
   실행해서 `self._aggregation`에 append합니다.

이제 step 2와 3 사이에서 interrupt하십시오. `InterruptionFrame`이 같은 input queue에
`SYSTEM_PRIORITY = 10`으로 들어와서 **먼저** pop됩니다. `_start_interruption`이 process queue를 다시
만들고, `_handle_interruptions`가 `self._aggregation`을 있는 그대로 commit합니다 — *그 단어 없이*.
그 다음 input task가 그 단어를 pop해서 **새** process queue로 relay하고, `_handle_text`가 그것을 방금
`reset()`된, 어느 commit된 turn에도 속하지 않는 aggregation에 append합니다.

두 개의 결과, 둘 다 실재합니다:

- **commit된 prefix는 들린 것의 정확한 일치가 아니라 하한(lower bound)입니다.** 창은 한 hop 폭이고 —
  일반적인 경우 밀리초 미만, aggregator의 input task가 suspend되어 있었다면 더 넓어집니다 — 따라서
  전형적인 손실은 0 또는 1 단어입니다. 하지만 그것은 **단방향**입니다: mechanism은 들린 것보다 *더 많이*
  commit할 수는 결코 없고, 오직 더 적게만 할 수 있습니다.
- **길 잃은 단어가 다음 turn의 aggregation으로 살아남습니다.** `_assistant_turn_start_timestamp`가
  `:2197`에서 비워지므로 다음 `_trigger_assistant_turn_stopped`는 곧장 return합니다 — 다음
  `LLMFullResponseStartFrame`에서 `_trigger_assistant_turn_started`가 발화하기 전까지는. 그 시점에 남은
  조각은 이미 `_aggregation`에 앉아 있고, 다음 assistant message의 *prefix*가 됩니다. 이것이 §4.1의
  절반만 찬 accumulator가, framework 자신의 aggregator 안에서 일어나는 것입니다.

저는 이것을 **측정된 것이 아니라 유도된 것(derived, not measured)**으로 표시합니다. 이걸 결정지을 관측:
`on_assistant_turn_started` handler를 등록해서 `len(aggregator._aggregation)`을 로깅하고, 실제 한국어
audio로 barge-in을 백 번 돌린 뒤 0이 아닌 판독값의 개수를 세십시오. 한 번도 0이 아니지 않다면, 실무에서
input-task hop이 항상 interruption 도착보다 빠른 것이고 이 우려는 이론적입니다. 가끔 0이 아니라면,
여러분은 그렇지 않았다면 원인을 밝힐 수 없었을 "봇 답변이 이전 문장의 단어로 시작한다"는 간헐적 버그의
mechanism을 찾아낸 것입니다.

### 7.8 왜 marker의 부재가 실제 behavioural gap인가

문제를 한 문장으로: **맨(bare) 부분 문장은, 모델 입장에서 완결된 짧은 답변과 구별되지 않습니다.**

구체적으로, Lina 통화에서. Lina가 시작합니다:

> "고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금 가입하시면 첫 달 보험료가…"

고객이 "지금"에서 끼어듭니다. context에는 이제 이것이 들어 있습니다:

```json
{"role": "assistant", "content": "고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금"}
```

다음 inference에서 모델은 부사 하나에 매달려 흐지부지 끝나는 **완결된 assistant turn**을 읽습니다.
잘렸다는 것을 알 방법이 없습니다. 흔한 downstream 실패를, 보게 될 빈도 순으로 대략 나열하면: 모델이
이미 그 얘기를 했다고 믿기 때문에 끊긴 지점을 이어가지 않는다; 조각이 잘못된 시작(false start)처럼
읽혀서 pitch 전체를 처음부터 반복한다; 매달린 절을 문체적 선택으로 여기고 흉내 낸다.

스크립트가 있는 sales flow에서는 이것이 복리로 불어납니다. [[ch-10/read]]의 stage machine이
"갱신 조건은 설명됐는가?"를 물을 텐데, transcript는 그렇다고 답하기 때문입니다.

고치는 비용은 구체적으로 얼마인가. `push_aggregation`은 `LLMAssistantAggregator` 위의 평범한
`async def`이므로, 가장 작은 변경은 subclass입니다:

```python
class TaggedAssistantAggregator(LLMAssistantAggregator):
    """Append an explicit interruption marker to a truncated assistant turn."""

    def __init__(self, *args, tag: str = "[고객 끼어듦]", **kwargs):
        super().__init__(*args, **kwargs)
        self._tag = tag
        self._interrupting = False

    async def _handle_interruptions(self, frame: InterruptionFrame):
        self._interrupting = True
        try:
            await super()._handle_interruptions(frame)
        finally:
            self._interrupting = False

    async def push_aggregation(self) -> str:
        if self._interrupting and self._aggregation:
            self._aggregation.append(
                TextPartForConcatenation(self._tag, includes_inter_part_spaces=True)
            )
        return await super().push_aggregation()
```

제대로 해야 할 두 가지, 둘 다 위에 인용한 코드에서 떨어져 나옵니다:

- **`on_assistant_turn_stopped` handler에서 message를 append하는 식으로 하려고 하지 마십시오.**
  `_trigger_assistant_turn_stopped`는 `:2179`에서 `push_aggregation()`을 호출하고 `:2189`에서 event
  handler를 발화합니다 — 여러분의 handler가 돌 때는 message가 *이미 context 안에* 있습니다.
  `on_assistant_turn_stopped` 방식이라면 하나를 append하는 게 아니라 **마지막 message를 다시 써야**
  합니다.
- **빈 aggregation 케이스는 여전히 사라집니다.** `self._aggregation`이 비어 있으면 `push_aggregation`은
  여러분의 hook이 뭔가 하기 전에 `:1679`에서 return합니다. "Lina가 말을 시작했는데 단어 하나 착지하기
  전에 잘렸다"를 표현 가능하게 만들고 싶다면 message를 직접 써야 하고 — tag만 있는 message든,
  marker-only message든 — 그 자리는 `push_aggregation`이 아니라 `_handle_interruptions`입니다.

[[interruption-cascade]]에 따르면, boson은 여러분이 여기서 쓸 tag 어휘를 이미 갖고 있습니다:
`_TAGS = {"interrupted": "[interrupted-by-user]", "tool_canceled": "[tool call canceled, user interrupted: {tool_name}]", "barge_in_prefix": "[barge-in] "}`,
`set_interrupt_tags`로 override 가능하고, docstring 자신의 예시가 한국어입니다(`"[고객 끼어듦]"`).
그 어휘는 business logic이고 포팅됩니다. 그 주변의 machinery는 포팅할 필요가 없습니다.

---

## 8. 작성자의 의무

위의 모든 것은 이 시스템에서 `FrameProcessor`를 쓰는 사람 — [[ch-12/read]]부터는 여러분 — 을 위한 여섯
개의 규칙으로 바뀝니다.

**1. `InterruptionFrame`에서 여러분이 소유한 모든 accumulator를 reset하십시오. 아무도 대신 해주지
않습니다.**
이것이 §4.1이고 `tts_service.py:1030-1056`의 손으로 쓴 14줄입니다. 여러분의 processor가 frame 사이에
부분 state를 들고 있다면 — buffer, turn을 key로 하는 dict, counter, 절반쯤 만들어진 rule-evaluation
record — process task가 임의의 `await`에서 cancel될 때 그것은 절반만 만들어진 채 남고, 다음 turn을
오염시킵니다.

**2. `SystemFrame` 처리를 빠르게 유지하십시오.**
[[frame-processor]]와 [[ch-04/read]] §4.2에 따라: system frame은 **input task 위에서** 실행됩니다.
거기의 느린 branch는 *그 processor에서* 이후의 모든 system frame을 정체시킵니다 — 다음 interruption,
`CancelFrame`, error path까지 포함해서. `InterruptionFrame` branch 안에서 network call을 `await`하지
마십시오. (`TTSService`는 그렇게 합니다 — §5.4의 `_disconnect()`/`_connect()` — 그리고 그것은 의도적이고,
비싸며, 문서화된 예외입니다.)

**3. 항상 `await super().process_frame(frame, direction)`을, 첫 줄에서.**
[[ch-01/read]] §7.2가 이것을 가르쳤고, 여기 interruption에 특화된 결과가 있습니다. `_start_interruption`은
오직 base implementation을 통해서만 도달됩니다(`frame_processor.py:839-841`). `super()` 호출을 건너뛰면
여러분의 processor는 **barge-in에서 자기 process task를 결코 cancel하지 않습니다** — 다른 모든 processor가
넘어간 뒤에도 죽은 turn의 frame들을 계속 씹고 있습니다. error도 없고 warning도 없습니다.

**4. 절대 버려지면 안 되는 frame에는 `UninterruptibleFrame` mixin을 붙이십시오.**
`frames.py:147`. 이 mixin은 `Frame` subclass가 아니므로 결합해서 씁니다:
`class LinaRuleCommitFrame(ControlFrame, UninterruptibleFrame)`. 여러분 것이 여기 속하는지에 대한
검사로 §3의 열 개 class 목록을 쓰십시오: 무언가를 **종료**시키는가, 아니면 이미 절반쯤 진행된 것을
**정산**하는가? 그렇다면 예. bot 출력을 실어 나르는가? 그렇다면 아니오.

**5. 무엇이 pending인지 물으려면 `has_queued_frame(frame_type)`을 쓰십시오.**
`frame_processor.py:1244`. [[ch-04/read]] §4.3에서 물려받은 주의사항 하나: docstring의 O(1) 주장은
`has_uninterruptible`(counter)에 대해서는 참이고 `has_frame`에 대해서는 거짓입니다. 후자의 body는
deque의 선형 scan입니다(`frame_queue.py:63-66`). voice pipeline 깊이에서는 괜찮습니다. hot loop에는
넣지 마십시오.

**6. `_cancelling`이 여러분의 frame을 조용히 사라지게 만든다는 것을 기억하십시오.**
`frame_processor.py:253`이 그것을 `False`로 설정하고, `__cancel()`이 `:1105`에서 `True`로 설정하며,
그러면:

**`src/pipecat/processors/frame_processor.py:713-715`**
```python
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return
```

`CancelFrame` 이후, `queue_frame`은 enqueue하지도 않고 logging하지도 않은 채 return합니다. shutdown 중에
여러분이 push하는 것 — 마지막 CRM 쓰기, "통화 종료" event, metrics flush — 은 바닥에 떨어집니다. 반드시
일어나야 한다면, 그것은 `CancelFrame` 경로가 아니라 `EndFrame` 경로에서 일어납니다(§3에서 본 대로 그것은
in-band이고 uninterruptible입니다).

### 8.1 규칙 하나를 예제로

§4.1의 `SentenceLogger`를 가져와 올바르게 만드십시오. diff는 네 줄입니다:

```python
class SentenceLogger(FrameProcessor):
    def __init__(self, sink, **kwargs):
        super().__init__(**kwargs)
        self._sink = sink
        self._buffer = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)     # obligation 3 — also runs _start_interruption

        if isinstance(frame, InterruptionFrame):
            self._buffer = ""                             # obligation 1 — fast, no await (obligation 2)
        elif isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._sink.write(self._buffer)
            self._buffer = ""

        await self.push_frame(frame, direction)
```

이 버전이 첫 번째 버전과 달리 제대로 하는 세 가지: reset이 존재하고, input task에서 I/O를 하지 않으며,
`super()` 호출이 무조건적이고 첫 줄에 있습니다. figure의 accumulate-then-flush 패널은 두 버전을 같은
드래그 지점에 대해 돌립니다 — 첫 processor를 쓰기 전에 한 번 돌려 보십시오.

*필요하지 않은* 것에 주목하십시오: 어딘가에 등록하는 것, interface를 구현하는 것, interruption을
처리한다고 누군가에게 알리는 것. cascade는 emergent이고(§4), 그것은 **올바른 processor는 평범한 네 줄을
씀으로써 참여하고, 올바르지 않은 processor는 조용히 참여하지 않음으로써 참여한다**는 뜻입니다.

---

## 9. 세 개의 mechanism, 나란히

이 section은 세 시스템 각각이 barge-in에서 **무엇을 하는지** 진술합니다. 비교 형용사가 하나도 없고
결론에 도달하지 않습니다. [[ch-13/read]]가 이 중 무엇이든 채점하는 유일한 곳이고, 지금 이걸 적어두는
이유는 **먼저 정확히 기술하지 않은 것을 채점할 수는 없기** 때문입니다.

출처: Pipecat은 이 chapter 전체에서 인용한 코드; `boson-agent`의 `gateway/interrupt/`는
[[boson-interrupt-subsystem]]; `realtime_voice`는 [[rtv-vad-chunking]]과 [[rtv-vs-pipecat-gap]], 더 충실한
논의는 [[ch-03/read]] §7.2와 §8.

### 9.1 differential

| Axis | **Pipecat** | **boson `gateway/interrupt/`** | **`realtime_voice`** |
|---|---|---|---|
| **무엇이 signal을 만들 수 있는가** | `broadcast_interruption()`을 호출하는 모든 processor; ship되는 경로는 VAD → turn-start strategy (`llm_response_universal.py:1270`) | text만. 모든 결정 지점이 `text: str`을 받습니다 — `PartialDetector.is_partial`, `WordFilterPolicy.evaluate`, `fillers.is_filler`, `InterruptionGate.allows`. audio 경로 없음, energy threshold 없음, VAD 없음 | VAD frame-count hysteresis (`SileroVADConfig(threshold=0.5, min_speech_frames=2, min_silence_frames=6)`), 2-state, 16 kHz mono 전용 |
| **signal이 존재할 수 있는 가장 이른 시점** | VAD start-of-speech 이후 | STT partial이 존재한 이후, 더하기 `DurationPolicy(min_ms=500)`; production gate는 `server/websocket.py`의 2000 ms silence timer | `min_speech_frames`개의 audio frame 이후 |
| **signal이 실어 나르는 것** | 아무것도 — `InterruptionFrame`에는 field가 없음 (`frames.py:1142`) | 끼어든 `text`, session id, 그리고 `elapsed_ms` | 정수 하나: 새 `_active_generation` |
| **작업이 어떻게 중단되는가** | preemptive: 각 processor가 `await` 도중에 자기 process task를 cancel (`_start_interruption`, `:1130-1150`); coordinator 없음 | cooperative: `CancellationFlag.set()` / `.check()`가 `CancellationError`를 raise; docstring이 명시적 — *"Cooperative — tool runs to completion, then flag is checked"* | 이름 붙은 여섯 지점에서 generation-ID 동등성 비교, 각각 낡은 작업을 `GENERATION_DROPPED`로 버림 |
| **열거 가능한 enforcement point** | 없음 — processor가 있는 곳 어디든 | 있음 — flag의 check site들 | 있음 — 한 파일 안의 여섯 지점, 더하기 두 개의 buffer |
| **새 component가 참여하려면** | 자동으로, `_start_interruption`을 상속함으로써 (그리고 `super().process_frame`을 호출하는 경우에만) | 작성자가 flag를 check하는 경우에만 | 작성자가 generation 비교를 추가하는 경우에만 |
| **줄 서 있는 output audio** | `MediaSender.handle_interruptions` (`base_output.py:566-593`): clock + video task cancel, `_audio_queue.reset()` 또는 cancel-and-recreate, clock에는 **새** `asyncio.PriorityQueue` (`:1067-1071`) | 범위 밖 — boson의 gateway는 계약상 text-native | `GenerationAudioQueue.discard_generation()`이 condition lock 아래에서 deque를 atomic filter로 재구축; `BoundedAudioOutput.activate_generation()`이 더 오래된 것을 전부 drain하고 drop count를 반환 |
| **process 너머로 손 뻗기** | 여섯 개의 serializer가 carrier flush를 내보냄 (§5.6) | n/a | `OutboundAudioTrack`이 generation 변경 시 자기 `av.AudioFifo`를 버림 |
| **TTS provider 처리** | `InterruptibleTTSService` (`tts_service.py:1969`)가 timestamp 없는 provider에 대해 websocket을 bounce | n/a | n/a |
| **발화된 prefix를 어떻게 유도하는가** | **pipeline position** — aggregator가 `transport.output()` 뒤에 앉음 — 더하기 presentation time에 `_clock_task_handler`가 release하는 `TTSTextFrame.pts` | 유도하지 않음. partial은 agent가 stream out한 것 그대로, `partial_text` | **sample-ratio 재구성**: `AudioTextPlayoutLedger.audible_text()`가 acknowledged cursor까지 phrase를 훑고, 부분 phrase에 대해 `ratio = (cursor - sample_start) / (sample_end - sample_start)` 그 다음 `text[:int(len(text) * ratio)]` |
| **TTS가 제공해야 하는 것** | word 수준 경로에는 word timestamp; 없으면 단위는 문장 전체 (§7.6) | 아무것도 | sample count. timestamp를 전혀 내보내지 않는 TTS와도 동작 |
| **정확도 특성** | timestamp가 존재하는 곳에서는 word 경계에서 정확; 없는 곳에서는 문장 단위; one-hop 하한 (§7.7) | 발화된 prefix 측정이 아님 — 생성된 prefix 측정임 | sample당 문자 수의 선형 근사; 단어 내부에서, 그리고 발화 속도가 변하는 phrase에 걸쳐 drift |
| **history에 무엇이 쓰이는가** | `{"role": "assistant", "content": "<spoken-so-far>"}`, tag 없음. 재생된 단어가 0이면 아무것도 없음 (`:1679-1680`) | `Message(role="assistant", content=f"{partial_text}[interrupted-by-user]")` — `cancellation.py:128-132` | ledger의 audible prefix를 명시적으로 기록, "잘렸다"와 "다 말했고 상대가 답했다"를 구별하는 `semantic_interrupt` flag 포함 (`playout_complete()`, `session.py:502-507`) |
| **"이건 interrupt됐다"가 LLM에게 표현 가능한가** | 아니오 — `interrupted=True`는 일회성 `AssistantTurnStoppedMessage`에만 존재 | 예 — 문자 그대로의 tag, override 가능, 예시가 한국어 | 예 — `semantic_interrupt` flag |
| **Tool-call repair** | `IN_PROGRESS` placeholder를 `"CANCELLED"`로 덮어씀 (§5.2), `cancel_on_interruption=True`로 등록된 tool에 대해 | `_collect_unanswered_tool_uses`가 다음 user turn에 답 없는 `tool_use`마다 `ToolResultBlock(content=f"canceled: {tname}")`을 합성, 엄격한 role alternation 보존 | 범위 밖 — tool은 gateway에 삶 |

### 9.2 세 개의 실패 방식, 이름 붙이기

각 mechanism은 다르게 실패하고, 그 실패들은 서로 겹치지 않습니다.

**Pipecat은 표현(representation)에서 실패합니다.** commit하는 prefix는 들린 것의 좋은 측정치이고,
그 다음 turn이 잘렸다고 말하는 그 한 비트를 버립니다. 모델은 truncate된 문장을 완결된 문장으로
읽습니다(§7.8). 부차적으로, prefix의 정확도는 *벤더*의 함수입니다: timestamp 없는 TTS에서는 truncation이
문장 단위로 무너집니다(§7.6).

**boson은 발원(origination)에서 실패합니다.** 그 subsystem은 transcript가 존재하기 전에는 존재할 수 없는
signal의 downstream에 앉아 있는 581줄의 세심한 policy입니다. [[boson-interrupt-subsystem]]과
[[theory-out-of-band-priority]]에 따르면 그것은 queue-depth latency가 아니라 *signal-origination*
latency이고, 이 chapter가 기술한 모든 mechanism의 **upstream**에 앉아 있습니다. 발원을 먼저 audio로
옮기지 않은 채 signal만 priority channel에 올리는 것은 아무것도 사주지 않습니다: `InterruptionFrame`은
buffering된 TTS를 추월하지만, ASR partial이 도착하기 전에는 interrupt를 구성할 수 없습니다. 같은
excerpt에 따르면 같은 행에 속하는 사실이 두 개 더 있습니다: `PartialDetector`는 `bootstrap.py:316`에서 한
번 생성되고 **그 field는 어디에서도 읽히지 않습니다** — dead code이고, 실제 경로는 `_partial_transcripts`
dict 더하기 `server/websocket.py:288-317, 616-735`의 2000 ms silence timer입니다. 그리고
`WordFilterPolicy(ignore_words=["hmm","uh","um","ah"], max_chars=3)`은 **문자**를 세므로, 한국어 통화선에서
"네"와 "아니요"는 조용히 무시됩니다.

**realtime_voice는 근사(approximation)에서 실패합니다.** `audible_text()`는 sample당 문자 수의 선형
추정입니다. phrase 경계에서는 정확하고 단어 내부에서는 틀리며, 오차는 phrase에 걸친 발화 속도 변화와 함께
커집니다. 그 대가로 TTS에게서 sample count 외에는 아무것도 필요로 하지 않고, 그것이 벤더 교체에서
살아남는 입력 요구사항입니다.

### 9.3 실제 한국어 트래픽에서 이 셋을 구별할 관측은 무엇인가

세 개의 서로 다른 dependency를 가진 세 개의 서로 다른 mechanism이라는 결론은 측정으로만 현금화할 수
있습니다. 감탄하기보다 실제로 돌릴 수 있게 쓴 protocol이 여기 있습니다.

**Setup.** 실제 Lina 통화에서 녹음된 한국어 barge-in 50건, 각각에 대해: 렌더링된 그대로의 assistant
audio, 고객의 audio, 그리고 사람이 주석한 ground-truth 문자 index — *고객 자신의 발화가 그것을 가리기
전에 고객이 들을 수 있었던 마지막 문자*. 그 index를 귀로 주석하는 것이 비싼 부분이고 우회로는 없습니다.
그것이 세 mechanism 모두가 추정하고 있는 ground truth입니다.

**측정 항목, barge-in 하나당:**

1. **Prefix error, 문자 단위.** `len(committed_prefix) - ground_truth_index`, 부호 포함. Pipecat의 것은
   구성상 ≤ 0이어야 합니다(§7.7); realtime_voice의 것은 0을 사이에 두고 분포하되 분산이 단어 내부에
   집중되어야 합니다; boson의 것은 애초에 같은 양을 측정하고 있지 않으므로 *생성된* 것 빼기 *들린* 것으로
   따로 보고해야 합니다.
2. **문장 경계 붕괴율(Sentence-boundary collapse rate).** commit된 prefix가 정확히 문장 경계에서 끝나는
   barge-in의 비율. 1.0에 가까우면 §7.6의 timestamp 없는 경로가 여러분 벤더에서 활성이라는 확인이고,
   base rate에 가까우면 word 경로라는 확인입니다.
3. **길 잃은 조각 비율(Stray-fragment rate).** §7.7의 test: `on_assistant_turn_started` handler에서
   `len(_aggregation)`을 로깅. 0이 아닌 판독값은 전부 이전 turn에서 새어 나온 조각입니다.
4. **Signal-origination latency.** 고객의 첫 유성음 sample부터 interrupt signal이 존재하는 순간까지의
   밀리초. 이것이 boson을 나머지 둘과 갈라놓는 축이고, 어떤 truncation 수치보다도 **먼저** 보고되어야
   합니다. 700 ms의 발원 지연은 prefix 정확도를 학술적인 이야기로 만들어 버리기 때문입니다.
5. **Downstream 행동 결과.** 녹음된 barge-in마다, 그 결과 context를 모델에 재생하고 다음 turn을
   분류하십시오: *올바르게 이어간다*, *처음부터 반복한다*, *조각을 완결된 것으로 취급한다*. 이것이 §7.8을
   논증에서 숫자로 바꾸는 측정입니다.

측정 1과 2가 Pipecat의 word 경로와 sentence 경로를 구별합니다. 측정 1 대 5가 "prefix가 정확하다"와
"모델이 올바르게 행동한다"를 구별합니다 — 이 둘은 같은 property가 아니고, 둘을 뭉뚱그리는 것이 잘못된
쪽을 최적화하게 되는 경로입니다. 측정 4는 가장 먼저 취해야 하는 것입니다.

아직 아무것도 던지지 마십시오. 숫자를 적어 두십시오.

---

## 10. Lina를 위한 세 개의 framework-extension move

요약이 아니라 만들 것들입니다. 각각은 위의 mechanism을, 이 chapter가 제기하지 않은 문제에 적용한
것입니다.

**Move 1 — interrupt 불가능한 compliance segment.**
한국 보험 tele-sales에는 전문을 읽어야 하는 법정 고지가 있습니다. §1(a)가 스위치를 줬습니다:
`enable_interruptions`는 strategy 단위이고 default가 `True`이며
(`base_user_turn_start_strategy.py:56`), `BaseUserTurnStartStrategy`는 trigger별 override를
지원합니다(`:200-220`). 하지만 그것을 끄는 것만으로는 충분하지 않습니다. 고객의 audio는 여전히 도착하고
있고 turn boundary는 여전히 발화하기 때문입니다 — interrupt는 억제하면서, STT가 잘못된 turn에 귀속될
발화를 계속 누적하게 놔두는 셈입니다. 완전한 설계는 세 조각입니다: (a) [[ch-10/read]]의 flow node가
설정하는 flag에 따라 `enable_interruptions`를 gate하는 `FrameProcessor`; (b) 빠져나온 interruption이
있더라도 살아남도록 disclosure segment를 표시하는 custom `ControlFrame, UninterruptibleFrame` (§3);
(c) 그 segment 동안 포착된 고객 발화를 어떻게 할지에 대한 결정 — buffer해 두었다가 끝난 뒤 재생할지,
버릴지. (c)는 framework가 답해주지 않을 *제품* 질문이고, muting에 대한 Pipecat 자신의 답
(`UserMuteStartedFrame`, `frames.py:1176`)은 "억제하고 버린다"라는 점에 유의하십시오.

**Move 2 — interruption marker, 더하기 marker가 고칠 수 없는 것.**
§7.8이 subclass를 줬습니다. extension은 그것만으로는 충분하지 않다는 것을 깨닫는 데 있습니다:
timestamp 없는 TTS에서는 marker가 *문장 단위* prefix(§7.6)에 붙으므로,
`"...보장이 됩니다.[고객 끼어듦]"`은 고객이 5 %만 들었을 수도 있는 문장을 들었다고 주장합니다.
정직한 버전은 marker를 쓰**면서 동시에** 그것이 계산된 granularity를 기록합니다 — 예컨대 message의
`metadata` 항목이나, 여러분의 `AssistantTurnStoppedMessage` handler의 CRM 쓰기에 두 번째 field로 —
그래야 나중 분석이 word 정확도 truncation과 sentence 정확도 truncation을 구별할 수 있습니다. 그 구별이
정확히 §9.3의 측정 2이고, 처음부터 그걸 넣어 두는 것이 측정을 나중에 개조하는 대신 싸게 만드는
방법입니다.

**Move 3 — `AudioTextPlayoutLedger`를 `FrameProcessor`로 포팅하되, 조건부로만.**
[[rtv-vad-chunking]]의 migration note는 ledger가 "선택한 TTS가 word timestamp를 내보내는 *경우에만*
중복이 될 것"이라고 말합니다. §7.6은 그 조건을 검사 가능한 것으로 날카롭게 만듭니다: 후보 벤더에 대해
§9.3의 측정 2를 하루 돌리십시오. 문장 경계 붕괴율이 1.0에 가깝다면 word 경로가 활성이 아닌 것이고,
sample count만 필요로 하는 ledger는 Pipecat의 mechanism이 계산하지 *않는* 무언가를 계산하고 있는
것입니다. 포팅 shape은 canonical list에서 `transport.output()` **옆**(6.5번 위치)에 놓이는
`FrameProcessor`로, 지나가는 `TTSAudioRawFrame`의 sample을 세고 phrase→sample map을 들고 있으며,
의무 1에 따라 `InterruptionFrame`에서 `reset()`합니다. 이것을 복사가 아니라 진짜 extension으로 만드는
것은, ledger의 `acknowledge(generation_id, played_sample)` cursor에 Pipecat 쪽 입력이 없다는
점입니다 — Pipecat에는 client playout acknowledgement가 없습니다 — 그래서 cursor를 transport 자신의 write
position에서 가져와야 하고, 그것은 그 숫자의 의미를 바꿉니다: *고객 단말기가 재생한 byte*가 아니라
*우리가 carrier에게 넘긴 byte*. 그 치환이 받아들일 만한지 판단하는 것이 진짜 작업이고, 그것은
[[ch-13/read]] 질문입니다.

---

## 다음 챕터로

이 chapter가 앞으로 넘기는 것들, 이후 chapter가 재유도하지 않고 인용할 수 있도록 이름을 붙입니다:

- **cascade는 emergent이고 coordinator가 없다.** field 없는 `InterruptionFrame` 하나(`frames.py:1142`)가
  `broadcast_frame`(`frame_processor.py:1038-1054`)에 의해 sibling으로 연결된 두 instance로 broadcast되어
  N개의 processor에 도달하고, 각각이 독립적으로 자기 자신에게 `_start_interruption()`(`:1130-1150`)을
  실행합니다. 아무것도 등록하지 않고, 아무것도 acknowledge하지 않고, 아무것도 기다리지 않습니다.
  [[ch-12/read]]가 이것을 직접 물려받습니다: 여러분의 rule-layer processor는 평범한 네 줄(§8.1)을
  씀으로써 올바르게 참여하고, 조용히 안 씀으로써 잘못 참여합니다.
- **in-band / out-of-band 분리는 design이고, `EndFrame`이 그 증거다.** `frames.py:1899`의
  `class EndFrame(ControlFrame, UninterruptibleFrame)` 대 `:999`의 `class CancelFrame(SystemFrame)` —
  우아한 종료는 봇이 문장을 끝낼 수 있도록 의도적으로 순서를 지키고, 난폭한 경로만 priority channel을
  탑니다. 열 개의 class가 이 mixin을 달고 있고, 그 목록은 정책처럼 읽힙니다: 종료, 정산, 기록.
- **truncation은 pipeline position 더하기 audio clock의 property이고, Pipecat은 marker를 쓰지 않는다.**
  7번 위치의 assistant aggregator가, `_clock_task_handler`가 각자의 `pts`에 release하는 word 수준
  `TTSTextFrame`을 먹고, tag도 말줄임표도 `interrupted` field도 없이
  `{"role": "assistant", "content": <spoken-so-far>}`를 commit합니다 — 그리고 재생된 단어가 없으면 아예
  아무것도 쓰지 않습니다. `grep -n interrupt src/pipecat/processors/aggregators/llm_context.py`는 510줄에서
  hit 0을 반환합니다. `tests/test_context_aggregators_universal.py:2449`가 문자 그대로 assert합니다:
  `[("user", "Hi!"), ("assistant", "Hello")]`.
- **truncation의 단위는 여러분 TTS 벤더의 함수다.** word timestamp가 있으면 → 단어 단위 granularity.
  없으면 → 문장 전체, all-or-nothing(`tts_service.py:1322-1341`), *더하기* 모든 barge-in마다 완전한
  websocket bounce(`InterruptibleTTSService`, `:1969`). [[ch-07/read]]의 벤더 질문이 이것을 결과로
  갖습니다.
- **chunk size는 interrupt-granularity 결정이고, drain path는 하나가 아니라 두 개의 queue다.**
  `audio_out_10ms_chunks: int = 4` → 쓰기당 40 ms, `_audio_queue`는 `_audio_task_handler`가 drain하고
  realtime pacing은 transport의 `_write_audio_sleep`(`websocket/server.py:379,506-515`)이 하며,
  `_clock_queue`는 `_clock_task_handler`가 별도로 drain합니다. [[ch-11/read]]이 latency budget에서 두
  숫자를 모두 지출합니다.
- **cascade는 여러분 process의 가장자리에서 멈춘다.** 여섯 개의 telephony serializer가 carrier에게
  flush를 요청하고(§5.6), carrier가 무엇을 하는지는 이 repo 안에 없습니다. ship된 serializer가 없는 한국
  carrier는 여러분이 그 method를 직접 쓴다는 뜻입니다.
- **들고 갈 두 개의 정정.** `frames.py:551`의 `LLMContextFrame(Frame)`은 `DataFrame`이 아니라 root를 직접
  상속합니다 — 런타임에서는 같은 운명, 다른 이유이고, [[ch-02/read]]의 taxonomy leak의 살아 있는
  사례입니다([[frame-taxonomy]]가 이 class가 벗어나 있는 branch들을 정리합니다). 그리고 Pipecat은
  interruption에서 tool-message alternation을 **실제로 repair합니다**. `IN_PROGRESS` → `"CANCELLED"`
  경로로(§5.2) — [[ch-04/read]]은 그 반대를 [[ch-09/read]]를 위한 열린 질문으로 park해 뒀습니다. 전제가
  틀렸고 gap은 보이는 것보다 좁습니다.

[[ch-09/read]]가 다음 충돌을 다룹니다. 이 chapter는 계속 "LLM의 process task가 cancel되고 generation이
죽는다"를 마치 LLM이 list 안의 processor 하나인 것처럼 말했습니다 — Pipecat의 design에서는 실제로
그렇습니다. boson의 design은 정반대를 말합니다: `StreamingConversationAgent`는 `AgentTextDelta`만
yield하고, tool은 `basement`와 `gateway`에 살며, `CLAUDE.md`는 "Basement and Gateway must not import
provider-specific audio code"를 요구합니다. 두 개의 agent loop, 두 개의 context 소유자, 두 개의 tool loop
소유자, turn이 언제 끝나는지에 대한 두 개의 정의. ch-09는 그것이 기술되는 데 그치지 않고 해소되어야 하는
곳입니다.

잃어버리지 않도록 여기 park해 두는 열린 질문들:

- **§7.7의 stray-fragment race.** machinery에서 유도된 것이지 측정된 것이 아닙니다. 한 줄짜리 실험이
  §7.7에 있습니다. [[ch-11/read]]이 observer plane을 만들기 **전에** 돌리십시오. observer plane이 바로
  그것을 지켜볼 자리이기 때문입니다.
- **marker가 무엇을, 어떤 언어로 말해야 하는가.** [[interruption-cascade]]는 boson의 override 가능한
  tag를 한국어 예시(`"[고객 끼어듦]"`)와 함께 기록합니다. 그 외에는 한국어인 context 안의 한국어 marker가
  multilingual model에게 도움이 되는지 혼란을 주는지는 architecture 질문이 아니라 evaluation 질문이고,
  [[ch-12/read]]의 rule layer에 속합니다.
- **ledger 포팅이 값어치를 하는가.** Move 3의 조건은 측정이고, 그 측정에는 선택된 TTS 벤더가 필요합니다.
  [[ch-13/read]]가 그것을 닫습니다.

