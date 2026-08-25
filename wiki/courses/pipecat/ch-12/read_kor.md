---
title: "Rule Layers as Middleware: boson의 rule을 Pipecat 위에 설계하기"
chapter: ch-12
phase: collision
course: pipecat
lang: ko
companion_of: read.md
sources:
  - design-boson-rules-on-pipecat
  - boson-layers-rules
  - boson-script-engine
  - custom-processor-guide
  - processor-vocabulary
  - bus-and-extensions
  - flows-state-machine
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# ch-12 — Rule Layers as Middleware: boson의 rule을 Pipecat 위에 설계하기

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로
> 대응하므로 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문
> 그대로입니다. CS/ML 용어는 영어를 유지합니다 (frame, processor, node, transition, action,
> aggregator, latency, observer, falsifier, seam, veto, rollback, snapshot 등).
>
> **읽는 순서에 대한 경고 — 이것이 이 chapter의 전부입니다.** 원문은 constraint → 열린 질문 →
> figure 작업 → 그제서야 결론의 순서로 **의도적으로** 배열되어 있고, 이 companion도 같은 순서를
> 글자 그대로 지킵니다. 이 페이지 맨 위에는 요약이 없고, mapping table도, 결론도 앞으로 당겨오지
> 않았습니다. 그것은 누락이 아니라 설계입니다. STEP 4를 먼저 읽으면 답은 얻고 도출(derivation)은
> 잃습니다.

## 왜 이 챕터인가

이 chapter는 당신이 요청한 것입니다. 그리고 일부러 [[ch-11/read]] 뒤에 놓았습니다. 이 chapter의
중심 trade-off가 **latency** trade-off이고, 값을 매길 수 없는 latency trade-off는 그냥 의견에
불과하기 때문입니다. ch-11이 budget을 만들었습니다. 이 chapter가 그것을 씁니다.

또한 이 chapter는 이 course가 Pipecat을 *설명하는* 일을 멈추고 당신에게 Pipecat을 상대로
*설계하게* 만드는 지점입니다. 그래서 이전의 모든 chapter와 다르게 쓰였고, 그 차이는 문체가 아니라
**요점 그 자체**입니다.

**이 chapter는 자신의 답을 보류합니다.** 네 단계로 구성되어 있고, 이 순서로만 진행됩니다.

| | 무엇이 일어나는가 | 무엇이 의도적으로 빠져 있는가 |
|---|---|---|
| **STEP 1** | 검증된 사실 네 개. permission 하나, constraint 셋. 각각 당신이 직접 `awk` 해 볼 수 있는 파일과 줄 번호가 붙어 있음. | 어떤 design도, 어떤 입장도, 어떤 processor 이름도, 어떤 code도 |
| **STEP 2** | 질문 두 개, 제기되고 **열린 채로 남음**. | 둘 중 어느 하나라도 답하는 문장 |
| **STEP 3** | `figures/rule-processor-placement.html`을 열고 두 개의 block을 직접 배치함. | figure의 reveal button은 당신이 배치하기 전까지 disabled 상태 |
| **STEP 4** | 두 개의 결론, 열한 행짜리 mapping table, pipeline listing, 그리고 실제 `process_frame` 약 40줄 — **lesson이 아니라 당신 도출에 대한 check로 제시됨** | 없음 |

STEP 4를 먼저 읽으면 당신은 답을 배우고 도출은 배우지 못합니다. 그리고 답은 도출보다 훨씬 값이
쌉니다. 답은 commit `0cbf9c5b` 시점의 Pipecat에만 특수한 것이고, 도출은 *당신*에게 특수한
것이니까요. constraint가 셋이고 결론이 둘인데, 결론은 constraint로부터 당신이 직접 도달할 수 있을
만큼 빡빡하게 따라 나옵니다. 그래서 constraint에는 줄 번호가 붙고, 결론에는 "당신의 답이 이 행과
어긋난다면 우리 둘 중 하나가 틀린 것이고, 여기 그것을 판정하는 줄이 있다"가 붙습니다.

이 chapter가 **아닌** 것 두 가지. 이것은 keep-or-replace 논증이 아닙니다 — 그건 [[ch-13/read]]의
일이고 다른 어디의 일도 아닙니다. 그리고 이것은 Pipecat에 뭔가가 빠져 있다는 주장이 아닙니다.
Pipecat은 아무것도 빠뜨리지 않았습니다. 다만 transaction이 어디에 사는지에 대해 다른 베팅을 했을
뿐이고, 이 chapter는 그 베팅이 당신에게 비용을 청구하는 정확한 지점을 찾아냅니다.

사전 조건 하나: 이 chapter는 당신이 이미 [[ch-02/read]]의 three-way frame test, [[ch-09/read]]의
`LLMContext` ownership model(특히 §2.2와 §2.3 — slice assignment와 live list), [[ch-10/read]]의
`flows/`를 pipeline *바깥*에 사는 machine으로 읽어낸 독해, 그리고 [[ch-11/read]]의 rule evaluation
자리가 비어 있는 latency budget을 가지고 있다고 가정합니다. 그중 어느 것도 다시 도출하지 않습니다.

---
---

# STEP 1 — THE FACTS

사실 네 개. 하나는 **permission** — 당신이 있다고 가정했을 blocker를 제거합니다. 셋은
**constraint** — 그것들은 벽이고, design은 그 안쪽에서 지어져야 합니다.

STEP 1의 어떤 것도 design이 아닙니다. STEP 1의 어떤 것도 processor, position, pipeline을 이름
부르지 않습니다. 여기서 결론을 읽고 있다는 느낌이 든다면 그건 제가 실패한 것이고, discuss phase에서
그렇게 말해 주십시오.

---

## 1. FACT ZERO (the permission): Flows는 transition이 LLM function call일 것을 요구하지 않는다

이것은 constraint가 아니라, 알고 보니 열려 있는 자물쇠입니다. 이것이 맨 앞에 오는 이유는, 만약
당신이 그 반대를 믿는다면 — 그리고 이 course 자신의 이전 working assumption도 그 반대를 믿었습니다,
[[boson-stage-machine]]의 migration note를 보십시오 — boson의 deterministic stage machine 전체가
이식 불가능해 보이고, 당신은 존재하지도 않는 벽을 피해서 설계하게 되기 때문입니다.

[[boson-stage-machine]]에 진술된 boson의 core invariant: *"the transition is performed by
deterministic rule code, not by the model … The LLM is told which stage it is in and never gets
to choose the next one."* 등록된 Lina stage 아홉 개, stage마다 `transitions` whitelist, 그리고
list에 없는 것은 무엇이든 거부하는 `StageMachine.transition()`.

Pipecat Flows에는 `NodeConfig`의 producer가 **두 개** 독립적으로 존재하고, 그중 하나만이 model을
관여시킵니다.

**Path A — LLM function call.** tool handler가 `(result, next_node)`를 반환하고, 그 tuple의 두 번째
원소가 *곧* transition입니다.

**`src/pipecat/flows/manager.py:443-447`**

```python
    async def _create_transition_func(
        self,
        name: str,
        handler: Callable | FlowsDirectFunctionWrapper,
    ) -> Callable:
```

그 path는 concurrency guard로 gate되어 있고, 이건 볼 가치가 있습니다. Flows가 in-flight tool batch를
진지하게 다룬다는 것을 알려주기 때문입니다:

**`src/pipecat/flows/manager.py:525-537`**

```python
    async def _check_and_execute_transition(self) -> None:
        """Check if all functions are complete and execute transition if so."""
        if not self._pending_transition:
            return

        # Check if all function calls are complete using Pipecat's state
        assistant_aggregator = self._context_aggregator.assistant()
        if not assistant_aggregator.has_function_calls_in_progress:
            # All functions complete, execute transition
            transition_info = self._pending_transition
            self._pending_transition = None

            await self._execute_transition(transition_info)
```

`has_function_calls_in_progress`는 `llm_response_universal.py:1503`이고, 그 body는 `:1509`의
`return bool(self._function_calls_in_progress)`입니다. parallel tool batch는 batch 도중에 transition
할 수 없습니다. 좋은 engineering이고, 당신에게는 무관합니다. 당신은 Path A를 쓰지 않을 것이기
때문입니다.

**Path B — 순수한 code, model 관여 없음.** method가 public입니다:

**`src/pipecat/flows/manager.py:588-591`**

```python
    async def set_node_from_config(self, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Used to manually transition between nodes in a flow.
```

`async def`, 앞에 underscore 없음, docstring은 *"Used to manually transition."*이라고 말합니다.
그리고 이건 이론적인 API가 아닙니다 — in-tree에서 서로 다른 두 종류의 callback으로부터 평범한
Python으로 두 번 호출됩니다:

**`examples/flows/warm_transfer.py:259-261`**

```python
async def start_human_agent_interaction(flow_manager: FlowManager):
    """Transition to the "human_agent_interaction" node."""
    await flow_manager.set_node_from_config(create_human_agent_interaction_node())
```

flow state로 guard된 Daily transport event handler로부터 구동됩니다:

**`examples/flows/warm_transfer.py:656-658`**

```python
            user_id = participant.get("info", {}).get("userId")
            if user_id == "agent" and flow_manager.current_node == "transferring_to_human_agent":
                await start_human_agent_interaction(flow_manager=flow_manager)
```

그리고 다시 한 번, worker lifecycle event로부터:

**`examples/flows/multi_worker_handoff.py:345-352`**

```python
    @worker.event_handler("on_activated")
    async def on_activated(worker, args):
        if not initialized["done"]:
            initialized["done"] = True
            await flow_manager.initialize(party_size_node())
        else:
            # Control was handed back to us; restart the reservation flow.
            await flow_manager.set_node_from_config(party_size_node())
```

두 call site 모두 tool handler가 아닙니다. 두 곳 모두 어느 시점에도 model을 관여시키지 않습니다.

> 💡 **쉬운 설명 — Path A와 Path B의 차이가 왜 생사를 가르나요?**
> boson의 핵심 안전 장치는 "다음 stage를 model이 고르지 못한다"입니다. 만약 Pipecat Flows에서
> transition을 일으키는 방법이 Path A(= LLM이 function을 호출해야 함)뿐이라면, 그 안전 장치는
> Pipecat 위에서 **원리적으로** 재현 불가능해집니다. model이 함수를 부르지 않기로 하면 stage가
> 안 넘어가니까요. Path B가 존재한다는 사실은 "boson의 deterministic stage machine을 그대로
> 들고 올 수 있다"는 허가증입니다. 그래서 constraint가 아니라 permission으로 분류됩니다.

> **Source correction.** [[design-boson-rules-on-pipecat]]는 Path B의 증거로
> `warm_transfer.py:658`을 인용하고, [[flows-state-machine]]은 `:658`과 `:259-261`을 둘 다
> 인용합니다. 파일을 열면 판정이 납니다: **`:261`이 `set_node_from_config` 호출**이고,
> **`:658`은 그것을 담은 함수를 호출하는 transport callback**입니다. 두 줄 다 실재하고 서로 다른
> 것입니다. "code가 public API를 호출한다"는 뜻이면 `:261`을, "그리고 그 caller가 transport
> event다"라는 뜻이면 `:658`을 인용하십시오. chapter spec의 `:261`이 더 타이트한 citation이고
> 저는 그것을 씁니다.

**Fact Zero가 주는 것, 받아 마땅한 만큼만 좁게 진술하면:** boson의 chain —
`RuleEngine.evaluate` → `Action` → `ExecutionResult.pending_transition` →
`core._apply_stage_transition` → `StageMachine.transition` ([[boson-stage-machine]]) — 은 Pipecat
안에 **model이 도달할 수 없는** sink를 가집니다. `flow_manager` reference를 들고 있게 될 어떤
object든 `set_node_from_config(node_for(target))`를 호출할 수 있고, LLM은 결코 stage machine에 대한
control을 얻지 못합니다.

**Fact Zero가 주지 *않는* 것:** 그 object가 *어디에* 사는지에 대해 아무 말도 하지 않고, 한 turn 중
*언제* 행동해도 되는지에 대해 아무 말도 하지 않으며, 그것이 하나 이상일 수 있는지에 대해서도 아무
말도 하지 않습니다. 그 둘이 STEP 2의 두 질문이고, Fact Zero는 어느 쪽도 답하지 않습니다. 거기
도달하면 이 주장을 직접 검증하십시오.

---

## 2. CONSTRAINT ONE — boson의 rule은 **완결된** user utterance를 증명 가능하게 요구한다

"선호한다"가 아닙니다. "그게 더 잘 동작한다"도 아닙니다. **요구합니다**, 구조적으로. 그리고 그 증명은
boson 자신의 code 안에 서로 독립적인 세 군데에 있습니다. 셋 다 [[boson-layers-rules]]와
[[design-boson-rules-on-pipecat]]에서 나옵니다 (boson-agent는 private입니다; 이것들은 pre-read
excerpt이고 저는 그 repo를 열지 않습니다).

**(a) type signature.** `LayerPipeline.process` (`gateway/layers/pipeline.py:87-94`):

```python
async def process(
    self,
    session_id: str,
    content: str,
    session: SessionState,
    *,
    user_message_appended: bool = False,
)
```

`content: str`. `AsyncIterator[str]`도 아니고, stream handle도 아니고, final flag가 달린 partial도
아닙니다. 완결된 utterance 하나, 호출 하나. rule이 prefix를 받을 수 있는 API 자체가 없습니다.

**(b) gateway가 partial 전달을 거부한다.** `server/websocket.py:293-317`의 `partial_transcript`
handler는 모든 branch가 `continue`로 끝납니다. partial은 `self._partial_transcripts[session_id]`에
쌓이고 `:374`의 `_replace_active_task(...)`에 결코 도달하지 못하는데, 그것이 `_message_handler`로
가는 *유일한* 경로입니다. `:288-292`의 주석은 명시적이고, [[boson-layers-rules]]에 verbatim으로
인용되어 있습니다: *"Incremental ASR: keep only the latest hypothesis. A partial may stop an
in-flight response promptly, but only after the same filler/policy gate used by explicit final
frames has authorized the interruption. **Rules/LLMs/tools still do not see incomplete text.**"*

**(c) rule들이 그 사실을 이용하고 있어서, 이걸 약화시키면 rule이 조용히 깨진다.** 이 부분이 중요한
부분입니다. (a)와 (b)는 policy이고 원리적으로는 완화될 수 있지만, (c)는 그럴 수 없습니다.

`end_signal.py`는 `user_message.lower().strip()`에 대해 whole-string `kw in lower` matching을
합니다. 5개 category × 약 10개 한국어/영어 keyword에 걸쳐, `:54-62`의 `STAGE_SIGNALS` table로
gate되어서. `_detect_natural_close`는 `messages[-4:]`까지 거슬러 올라갑니다. 둘 다 whole-string
연산입니다. 어느 쪽에든 prefix를 먹이면 error가 나지 않습니다 — **엉뚱한 것에 대해 일찍
발화(fire)합니다**. 그게 더 나쁩니다.

filler filter도 같은 모양입니다:

```python
@check("korean_filler_filter", mode="sequential", priority=10)
def filter_fillers(messages, user_message, session):
    """Filter filler words ONLY during agent streaming."""
    status = getattr(session, "pre_turn_status", None) or session.get_agent_status()
    if status not in ("generating", "tool_processing"):
        return Pass()
    if _is_filler_text(user_message):
        return Filter(
            reason=f"filler_word:{_normalize(user_message) or user_message.strip()}"
            f" | agent_status:{status}"
        )
    return Pass()
```

(`agents/test-lina-gateway/layers/01-filler-filter/rules/korean_fillers.py:53-75`,
[[boson-layers-rules]] 경유.) `_is_filler_text`는 비-한글 문자를 정규화해 없앤 뒤
`KOREAN_FILLERS = ["네","예","아","음","어","그","응","아아","음음","네네","예예","아하","흠","그래요"]`
에 대한 membership을 테스트합니다. `"네"`는 그 list에 있습니다. 그리고
`"네 그런데 제가 지난번에 여쭤본 게…"`의 첫 글자도 그 list에 있습니다. 그 문장의 이른 hypothesis를
실은 `TranscriptionFrame`은 backchannel로 filter되고, 고객의 실제 질문은 버려집니다.

그리고 당신이 이미 만난 적 있는, Pipecat 자신의 vocabulary 안에도 이 문제의 층이 하나 더 있습니다.
`InterimTranscriptionFrame` (`frames.py:476`)은 `TextFrame`의 subclass입니다. 따라서
`isinstance(frame, TextFrame)`을 테스트하고 `.text`를 읽는 *모든* code는, 명시적으로 opt out 하지
않는 한 interim hypothesis를 봅니다. Pipecat 자신의 `SentenceAggregator`는 맨 `return` 하나로 opt
out 합니다 (`aggregators/sentence.py:50-51`, [[custom-processor-guide]]에 인용됨).

> 💡 **쉬운 설명 — "error가 아니라 wrong answer"가 왜 더 나쁜가요?**
> prefix를 받은 rule이 exception을 던져 준다면 오히려 다행입니다. 로그에 뜨고, 고치면 됩니다.
> 그런데 `"네 그런데…"`의 앞 글자 `"네"`를 받은 filler filter는 **정상적으로 동작**해서 "이건
> 맞장구다"라고 결론 내리고 고객의 문장을 버립니다. 예외도, 경고도, 흔적도 없습니다. 이 차이가
> Constraint One을 "policy 선호"가 아니라 "structural requirement"로 만듭니다.

**Constraint One을 한 줄로:** 모든 boson rule은 input contract가 *`str`로 된 완결된 utterance
하나*인 순수 Python 함수이고, 살아 있는 13개 check 중 셋은 prefix에 대해 error가 아니라 **틀린 답**을
내놓습니다.

---

## 3. CONSTRAINT TWO — `push_frame`은 되돌릴 수 없다

un-push는 없습니다. retraction frame도 없습니다. `pop_frame`도, `recall_frame`도, `cancel_pushed`도
없습니다. grep 해 보십시오. 존재하지 않습니다.

존재하는 것은 이것입니다:

**`src/pipecat/processors/frame_processor.py:1004-1015`**

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

statement 세 개. 앞의 hook 하나, 이웃에 enqueue, 뒤의 hook 하나. [[ch-01/read]]가
`__internal_push_frame`이 무엇을 하는지 확립했습니다: frame을 *다음 processor의* input queue에
넣습니다. 그것이 return하는 순간 frame은 다른 processor의 queue에 있고, 그 processor의 task는 이미
그것을 실행하고 있을 수도 있습니다. 당신은 더 이상 그것을 소유하지 않습니다. 애초에 소유한 적도
없습니다.

framework 전체에서 retraction에 가장 가까운 것은 broadcast interruption입니다:

**`src/pipecat/processors/frame_processor.py:1017-1022`**

```python
    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        self.__reset_process_task()
        await self.stop_all_metrics()
        await self.broadcast_frame(InterruptionFrame)
```

그것이 무엇이고 무엇이 *아닌지*를 읽으십시오. *이* processor의 process task를 reset하고, metric을
멈추고, 양방향으로 `InterruptionFrame`을 보냅니다. downstream processor들은 그에 반응해 queue를
비웁니다 ([[ch-08/read]]가 전체 cascade를 추적했습니다). 이것은 **best-effort global abort**이지
undo가 아닙니다. 누구에게도 무엇을 un-do 하라고 말해 주지 않고, 누군가 mutate한 state를 복원하지
않으며, 그것이 도착할 때쯤이면 일어날 예정이었던 side effect는 이미 일어났습니다 — audio는 이미
회선 위에 있을 수 있습니다.

boson이 하는 것과 대조하십시오. [[boson-layers-rules]]는 `LayerPipeline._process_active`
(`:128-335`)를 진짜 two-phase commit으로 기술합니다. **Phase 1** (`:178-248`)은 모든 layer를
평가하고 action을 *staging만* 합니다 — 아무것도 밖으로 나가지 않습니다. `decision == "filter"`
(`:205-239`)이면 `session.messages`를 **object identity 기준으로 뒤에서부터** 훑어서 자기가 append
했던 바로 그 `pipeline_user_message` object를 삭제하고, `_pending_stage_injection`을 clear하고,
return합니다. **Phase 2** (`:254-316`)는 Phase 1이 commit한 경우에만 staged action을 replay합니다.
그리고 arbitration을 deterministic하게 만드는 flow-control precedence:

```python
ACTION_PRIORITY = {
    "filter": 0, "respond": 1, "inject": 2,
    "stage_transition": 3, "compact": 3, "pre_tool": 3,
    "pass": 4, "continue": 4,
}
```

(`gateway/layers/pipeline.py:42-51`, [[boson-layers-rules]] 경유.) `filter`는 0입니다. commit 이전
어느 시점이든, 어느 layer에서 왔든, 모든 것을 이깁니다.

> 💡 **쉬운 설명 — "staging"과 "push"의 차이를 은행으로**
> boson의 Phase 1은 은행 창구에서 이체 전표를 **작성만** 해 두는 것과 같습니다. 네 명이 각자
> 전표를 쓰고, 마지막 사람이 "이 거래 취소"라고 하면 전표 뭉치를 통째로 찢으면 됩니다. 돈은 아직
> 움직이지 않았습니다.
> Pipecat의 `push_frame`은 **송금 버튼**입니다. 누르는 순간 돈은 상대 계좌에 있습니다.
> `broadcast_interruption()`은 "모두 하던 일 멈춰!"라는 방송이지, 송금 취소가 아닙니다.

**Constraint Two를 한 줄로:** boson의 veto는 staged effect에 대한 transaction rollback이고,
Pipecat에는 write verb가 정확히 하나뿐이며 그것은 호출 시점에 효력이 발생하고, 유일하게 제공되는
global abort는 아무것도 복원하지 않습니다.

---

## 4. CONSTRAINT THREE — data dependency, 그리고 그 안의 문제가 되는 두 줄

이 chapter 전체가 이 constraint 위에 얹혀 있으므로, 하위 절 네 개를 주고 `awk`를 직접 돌려 보라고
말하겠습니다. chapter spec은 이전 draft가 여기서 잘못된 증거 줄을 인용하고 그 위에 도출을 쌓았다고
표시하고 있습니다. [[ch-09/read]] §0(d)가 같은 부류의 오류를 표시했습니다. 이것만큼은 제 말로 받지
마십시오.

### 4.1 aggregator 쪽: 쓰고, 그다음 push하고, 그다음 알린다

**`src/pipecat/processors/aggregators/llm_response_universal.py:856-873`**

```python
    async def push_aggregation(self) -> str:
        """Push the current aggregation."""
        if len(self._aggregation) == 0:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()
        self._context.add_message(
            cast(LLMContextMessage, {"role": self.role, "content": aggregation})
        )
        await self.push_context_frame()

        message = UserTurnMessageAddedMessage(
            content=aggregation, timestamp=self._user_turn_start_timestamp
        )
        await self._call_event_handler("on_user_turn_message_added", message)

        return aggregation
```

검증:

```console
$ awk 'NR>=856 && NR<=873 {printf "%d\t%s\n", NR, $0}' \
    src/pipecat/processors/aggregators/llm_response_universal.py
```

줄 번호 셋, 서로 다른 event 셋, 이 순서로:

| line | statement | 그 의미 |
|---|---|---|
| **`:863`** | `self._context.add_message({"role": self.role, "content": aggregation})` | 사용자의 완결된 turn이 이제 **공유 `LLMContext` 안에** 있다 |
| **`:866`** | `await self.push_context_frame()` | `LLMContextFrame`이 aggregator를 떠나 downstream으로 이동한다 |
| **`:871`** | `await self._call_event_handler("on_user_turn_message_added", message)` | event가 발화한다 |

`:863` 다음 `:866` 다음 `:871`. convention이 아니라 — 직선으로 뻗은 함수 body입니다.

`push_context_frame()`은 같은 context object로부터 frame을 만듭니다
(`_get_context_frame` → `LLMContextFrame(context=self._context)`,
`llm_response_universal.py:564-568`, [[ch-09/read]] §3에서 읽음). 그 frame은 소포가 아니라
초인종입니다: `LLMContextFrame`은 reference를 실어 나르고, [[ch-02/read]] §3이 이미 그것이 세 branch
중 어디에도 속하지 않고 `Frame`을 직접 subclass한다고 언급했습니다.

### 4.2 service 쪽: 서로 다른 네 가지 일을 하는 연속된 네 줄

**`src/pipecat/services/openai/base_llm.py:599-605`**

```python
        await super().process_frame(frame, direction)          # :599

        if isinstance(frame, LLMContextFrame):                 # :601
            try:
                await self.push_frame(LLMFullResponseStartFrame())   # :603
                await self.start_processing_metrics()                # :604
                await self._process_context(frame.context)           # :605
```

검증:

```console
$ awk 'NR>=599 && NR<=605 {printf "%d\t%s\n", NR, $0}' \
    src/pipecat/services/openai/base_llm.py
```

| line | statement | 그 의미 |
|---|---|---|
| **`:601`** | `isinstance(frame, LLMContextFrame)` | **유일한** trigger. 다른 어떤 frame도 completion을 시작시키지 않는다. |
| **`:603`** | `await self.push_frame(LLMFullResponseStartFrame())` | **frame 하나가 이 processor를 떠나 downstream으로 나간다** |
| **`:604`** | `await self.start_processing_metrics()` | metric clock이 시작된다 |
| **`:605`** | `await self._process_context(frame.context)` | **completion이 여기서 시작된다** |

### 4.3 `:601 → :605` window는 비어 있지 않고, 그게 요점 전부다

`:603`을 다시 읽으십시오. 어떤 token이 생성되기도 전에, `_process_context`에 진입하기도 전에,
`LLMFullResponseStartFrame`은 이미 downstream으로 push되었습니다. 그건 `push_frame` 호출이고,
Constraint Two는 `push_frame` 호출이 되돌릴 수 없다고 말합니다.

그러므로 "service가 frame을 인식했다"와 "service가 generation을 시작했다" 사이의 window는
**네 줄 폭이고 이미 오염되어 있습니다**. 당신은 그 안으로 손을 뻗을 수 없습니다. `:601`과 `:605`
사이에는 hook이 없고, 설령 하나를 발명한다 해도 `:604` 시점이면 되돌릴 수 없는 push가 이미
일어났고 downstream processor들은 존재하지 않을 수도 있는 response의 여는 괄호에 이미 반응하고
있습니다.

이것이 "generation을 시작시키고 나중에 취소한다"는 아이디어가 *공짜*가 아니라 *두 번 지불*로
평가되어야 하는 이유입니다. 하지만 그 평가는 STEP 2에서 당신의 몫입니다.

> 💡 **쉬운 설명 — 왜 "window가 오염됐다"고 표현하나요?**
> 직관적으로는 "LLM이 실제로 토큰을 만들기 전이면 아직 아무 일도 안 일어난 것 아닌가?"라고
> 생각하게 됩니다. 아닙니다. `:603`에서 `LLMFullResponseStartFrame`이 이미 나갔고, downstream의
> TTS나 RTVI observer 같은 것들은 그걸 보고 "봇이 응답을 시작했다"는 상태로 이미 전이했습니다.
> 그 frame을 회수할 방법은 없습니다. 그래서 `:601`~`:605`는 "아직 아무것도 안 쓴 깨끗한 틈"이
> 아니라 "이미 한 번 결제가 일어난 구간"입니다.

### 4.4 한 줄 더, 당신의 rollback surface가 그것에 달려 있으므로

`LLMContext`가 제공하는 replace verb는 정확히 하나입니다:

**`src/pipecat/processors/aggregators/llm_context.py:361-383`**

```python
    def add_message(self, message: LLMContextMessage):
        """Add a single message to the context.

        Args:
            message: The message to add to the conversation history.
        """
        self._messages.append(message)

    def add_messages(self, messages: list[LLMContextMessage]):
        """Add multiple messages to the context.

        Args:
            messages: List of messages to add to the conversation history.
        """
        self._messages.extend(messages)

    def set_messages(self, messages: list[LLMContextMessage]):
        """Replace all messages in the context.

        Args:
            messages: New list of messages to replace the current history.
        """
        self._messages[:] = messages
```

`set_messages(list)`. 그것은 **list**를 받습니다. message 하나도, index도, object handle도 받지
않습니다. `remove_message`도, `delete_message_by_identity`도, `pop_message`도 없습니다.
[[ch-09/read]] §2.2가 `self._messages[:] = messages`가 rebind가 아니라 slice assignment인 이유를
이미 설명했고, §2.3이 `get_messages()`가 **live list**를 반환한다는 것을 이미 확립했습니다. 두 사실
모두 여기서 하중을 받고(load-bearing) 있으며 STEP 4에서 현금화하겠습니다.

**Constraint Three를 한 줄로:** `:863`의 write가 `:866`의 push보다 먼저 일어나고, `:866`의 push는
*유일한* trigger가 `:601`이고 completion이 `:605`에서 시작하는 service에 도달하는데 — `:603`에서
되돌릴 수 없는 downstream push가 이미 지출된 상태로 — 그리고 그 경로 어디에서든 유일한 rollback
verb는 `set_messages(list)`입니다.

---

## STOP.

여기까지가 STEP 1입니다. permission 하나, constraint 셋, 검증된 줄 번호 열한 개.

아직 STEP 4로 스크롤하지 마십시오. 이미 답이 형성되고 있다면, STEP 2를 읽기 전에 어딘가에 적어
두십시오 — 질문을 보기 *전에* 쓴 답이 본 *후에* 쓴 답보다 훨씬 좋은 진단 자료입니다.

---
---

# STEP 2 — THE TWO QUESTIONS

질문 둘. 제기하고 열어 둡니다. 여기서부터 STEP 3의 끝까지 어느 것도 둘 중 하나를 답하지 않으며,
제가 미리 답하는 것을 잡아낸다면 그건 이 chapter의 결함이고 discuss에서 듣고 싶습니다.

둘 다 STEP 1만으로 답할 수 있습니다. 제가 아직 주지 않은 사실은 하나도 필요하지 않습니다.

---

## Q1 — SEAM은 어디인가?

> **Pipecat pipeline의 어느 단일 position에서, 완결된 user turn이 존재하면서 동시에 inference가
> 아직 시작되지 않았는가?**

어떻게 답하느냐에서 정밀도가 중요합니다. "가운데 어딘가"는 답이 아닙니다. 답은 *"X의 바로
downstream이고 Y의 바로 upstream"*의 형태를 가지며, X와 Y 둘 다 **data dependency**로 정당화되어야
합니다 — 당신이 필요로 하는 무언가를 쓰는 줄 번호, 또는 당신이 앞질러야 하는 무언가를 소비하는 줄
번호. 취향으로도, 관례로도, "보통 middleware는 저기 놓으니까"로도 안 됩니다.

당신의 답에 추가로 요구되는 두 가지:

**(i) 단일 position이어야 하거나, 하나 이상임을 증명해야 합니다.** Constraint Three가 줄 번호의
chain을 줍니다. 그것을 걸어가면서 질문의 두 절반 — *완결됨*과 *아직 inference 안 함* — 을 동시에
살아남는 후보 position이 몇 개인지 보십시오.

**(ii) 두 개의 특정 rival을 이름으로 knock out 해야 합니다.** 둘 다 그럴듯하고, 둘 다 유능한
engineer가 첫날에 제안하는 것이기 때문입니다:

- **Rival B — event handler.** `on_user_turn_message_added`는 `:871`에서 발화하고 완결된 aggregate
  turn을 `UserTurnMessageAddedMessage`로 건네줍니다. 그건 정확히 rule이 원하는 data입니다. 왜 그게
  seam이 아닌가? (Constraint Three, §4.1에 답이 있습니다. 줄 번호 하나가 판정합니다.)
- **Rival C — 시작하고 취소하기.** LLM을 시작시키고, rule을 concurrent하게 돌리고, rule이 veto하면
  `broadcast_interruption()`을 호출한다. pass path — 대부분의 turn — 에서 추가 latency 0. 왜 그게
  공짜가 아닌가? (Constraint Two와 Three, §4.3. 비용이 둘이고, 그중 하나는 latency 비용이
  전혀 아닙니다.)

---

## Q2 — LAYER는 반드시 붕괴해야 하는가?

> **boson의 cross-layer veto가 N개의 인접한 `FrameProcessor`에 흩어진 채로 살아남을 수 있는가,
> 없는가?**

boson에는 살아 있는 layer가 넷 있습니다 — `01-filler-filter`, `02-analyzer`, `03-orchestrator`,
`04-committer` ([[boson-layers-rules]]). [[ch-01/read]]에서 확립된 Pipecat의 미학 전체는 pipeline이
list이고 거기에 물건을 splice해 넣는다는 것입니다. layer 넷, processor 넷, 순서대로. 그게 뻔한
모양이고 [[custom-processor-guide]]가 문자 그대로 권장하는 모양입니다:

> *"porting it as a single monolithic processor keeps it working but forfeits the per-processor
> queueing, observability, and interruption semantics Pipecat gives each stage — the honest port
> splits its four layers … into four `FrameProcessor`s."*

그것은 `frame_processor.py`를 읽은 사람이 쓴, excerpt library의 진짜 권장 사항입니다. 그게 옳은지
판단하십시오.

답하려면 시나리오 하나에 대해 구체적이어야 합니다:

> layer `03-orchestrator`가 `Inject` 하나와 `PreTool` 하나를 stage합니다. 그다음 layer
> `04-committer`가 `Filter(reason=...)`를 반환합니다. `ACTION_PRIORITY`에 따라 `filter`는 0이고
> 이깁니다: **모든** layer의 **모든** staged effect가 폐기되며, 여기에는 layer 01이 돌기 전에
> append되었던 user message도 포함되고, 그것은 object identity로 삭제됩니다.

이제 그 시나리오를 네 개의 분리된 processor에 걸쳐 돌리십시오. 각 processor가 다음 processor와
소통하는 유일한 방법이 `push_frame`인 상황에서, 답하십시오:

- **(a)** processor 04가 veto를 결정하는 시점에, processor 03에서 이미 나가버린 것은 무엇인가?
- **(b)** Constraint Three §4.4가 rollback verb를 정확히 하나 줍니다. 그것이 boson의 rollback이
  필요로 하는 shape를 가지고 있는가? boson이 무엇을 *기준으로* 삭제하는지(object identity)와
  `set_messages`가 무엇을 받는지(list)를 보십시오. 일반적으로, 하나가 다른 하나의 표현으로
  환원되는가?
- **(c)** (a)에 대한 당신의 답이 "아직 아무것도 나가지 않았다, processor 03이 그냥 *push 안 하면*
  되니까"라면, 그 순간 processor 04는 무엇을 하고 있는지 말하십시오. 04는 자기가 평가하고 있는
  frame을 어디서 얻는가?

(c)에는 옳은 답이 있고, 그것이 이 질문 전체의 경첩(hinge)입니다.

---

**STEP 3 전에 두 답을 다 적으십시오.** 머릿속 말고. STEP 3은 어떤 position에서 실제로 무슨 일이
일어나는지 보고하는 bench이고, bench는 예측을 들고 갔을 때만 쓸모가 있습니다.

---
---

# STEP 3 — GO WORK THE FIGURE

지금 여십시오:

**→ [`figures/rule-processor-placement.html`](figures/rule-processor-placement.html)**

이것은 **derivation bench이지 answer key가 아닙니다**. 아무것도 건드리지 않고 위에서 아래로 읽으면
아무것도 주지 않습니다 — 그건 의도된 것이고 figure 자신의 code에서 강제됩니다.

이걸로 무엇을 할지, 순서대로:

**PANEL ONE — 두 block을 배치하십시오.** `transport.input(), stt, user_aggregator, llm, tts,
transport.output(), assistant_aggregator`만 들어 있는 빈 pipeline과 배치되지 않은 block 두 개가
주어집니다. 아무 데나 떨어뜨리십시오. 이 도구는 결코 "정답"이라고 말해 주지 않습니다. 대신 하는 일은
한국어 utterance 하나를 replay하고 **당신이 고른 position에서 각 processor가 실제로 무엇을
받았는지** — 어떤 frame type이 도착했고 rule code가 무엇을 건네받았을지 — 보고하는 것입니다.
`stt` 앞에 block을 두고 text-matching rule이 `InputAudioRawFrame`을 받는 것을 보십시오. `llm` 뒤에
하나 두고 당신의 `Inject`가 투표권을 얻기 전에 model이 generation을 끝내는 것을 보십시오. 여기서는
틀린 position이 옳은 position보다 더 교육적이므로, 최소 네 번은 시도하십시오.

**두 block이 모두 배치되기 전까지 reveal button은 disabled입니다.** 그건 bug가 아니고 우회하는
keyboard shortcut도 없습니다. mapping table이 그 gate 뒤에 앉아 있고, 그것이 열릴 때 붙는 label은
**CHECK YOUR DERIVATION**입니다 — 그때쯤이면 당신이 도출을 하나 만들어 놓았을 것이기 때문입니다.

**PANEL TWO — interception marker를 드래그하십시오.** 이건 Q1을 timeline으로 렌더링한 것입니다:
`:863`의 `add_message`, `:866`의 `push_context_frame`, `:871`의 `on_user_turn_message_added`, 그리고
service 쪽의 `:601`, `:603`, `:604`, `:605` — `:601 → :605` window가 `:603`에서 이미 탈출한 frame을
품은 채로 그려집니다. marker를 직접 위치시키십시오. panel은 당신이 떨어뜨린 위치로부터
**CAN VETO / CANNOT VETO**를 계산합니다. 일부러 `:871` 위에 떨어뜨리고 Rival B가 죽는 것을 보십시오.
**여기서 최소 한 번은 veto verdict를 계산해야 합니다.** PANEL FOUR의 code view가 그때까지 잠겨
있기 때문입니다.

**PANEL THREE — two-phase commit을 두 번 돌리십시오.** 네 layer가 action을 stage하고, 마지막 하나가
`Filter`를 발화하고, boson의 identity 기반 rollback이 append된 user message를 삭제하는 것을 보십시오.
그다음 동일한 round를 네 개의 분리된 Pipecat processor로 다시 돌리십시오. `push_frame`으로 이미
탈출한 모든 frame이 빨간색으로 표시됩니다. 빨간 것을 세십시오. 그 개수가 Q2에 대한 당신의 답이고,
그것은 의견이 아니라 숫자입니다.

**PANEL FOUR, FIVE, SIX**는 check입니다 — "`super()` 호출을 삭제" 스위치가 달려 [[ch-01/read]]의
black hole을 재현하는 live `process_frame` body, ch-11의 빈 슬롯에 떨어뜨린 latency bill, 그리고
`llm` input에서 센 transition race. 이들은 STEP 4의 interactive 쌍둥이입니다. STEP 4를 읽은 뒤에,
또는 함께 하십시오.

**이 줄을 지나쳐 스크롤하기 전에, 종이 위에 이것들이 있어야 합니다:** 이름 붙은 정당화 둘을 가진
seam position 하나, 각각을 죽이는 줄 번호와 함께 knock out 된 rival 둘, 그리고 collapse에 대한
yes/no와 그 이유.

---
---

# STEP 4 — CHECK YOUR DERIVATION

여기서부터는 전부 **check**이지 lesson이 아닙니다.

이 chapter의 나머지에 적용되는 상시 규칙: **당신의 답이 아래의 무언가와 어긋난다면, 둘 중 하나가
틀린 것이고, 본문이 그것을 판정하는 source 줄을 이름 부릅니다.** 그 줄로 가십시오. 제게 오지 말고,
excerpt에도 가지 마십시오 — excerpt는 pre-read 요약이고 이 chapter는 그중 네 개를 교정합니다.

---

## 5. CHECK — 결론 (a): the seam

**seam은 user aggregator의 downstream이고 LLM service의 upstream이다.**

그런 position은 정확히 하나 존재하고, 양쪽 벽이 둘 다 data dependency입니다.

**downstream 쪽 벽 (왜 더 앞이면 안 되는가).** aggregator 앞에서 날아다니는 frame은
`TranscriptionFrame`과 `InterimTranscriptionFrame` — 조각들입니다. Constraint One은 조각을 받은
rule이 실패하는 게 아니라 틀린 답을 낸다고 말합니다: `end_signal.py`의
`kw in user_message.lower()`는 prefix에 발화하고, LLM intent-matcher의 prompt는 —
*"Most recent turn (PRIMARY SIGNAL — evaluate against THESE)"*에 anchor되어 있는데
(`intent_matcher.py:205-271`, [[design-boson-rules-on-pipecat]] 경유) — 반쪽짜리 문장을 건네받고
점수를 매깁니다. 완결된 turn은 `:863`이 그것을 context에 쓰기 전까지 존재하지 않습니다.

**upstream 쪽 벽 (왜 더 뒤면 안 되는가).** LLM service 이후에는 `:605`가 이미 돌았습니다. `Inject`는
자기 뒤에 오는 generation을 조종하려고 존재합니다. generation 이후에는 조종할 것이 남아 있지
않습니다. `Respond`와 `Filter`는 cancel-and-redo로 격하됩니다. 그리고 §4.3에 따라 `:603`은 이미
push했습니다.

**그러므로: `:866`이 push한 뒤, `:601`이 테스트하기 전.** pipeline 용어로는 user aggregator와 LLM
service 사이의 processor slot 하나입니다.

**그 position에서, 그리고 오직 그 position에서만 rollback이 실재합니다.** 당신은 aggregator가
`:863`에서 쓴 것과 **같은** `LLMContext`에 대한 reference를 들고 있습니다 —
`LLMContextFrame.context`가 바로 그 object입니다 ([[ch-09/read]] §3: frame은 소포가 아니라
초인종). message list를 snapshot하고, rule을 돌리고, veto 시 `set_messages(snapshot)`을 호출합니다.
당신이 push하지 않았기 때문에 downstream의 무엇도 아직 아무것도 보지 못했습니다.

### 5.1 Rival B는 죽었고, 줄 하나가 그것을 죽인다

`on_user_turn_message_added`는 **`:871`**에서 발화합니다. `push_context_frame()`은 **`:866`**에서
돕니다.

다섯 줄 앞에서 frame이 떠났습니다. 당신의 event handler가 호출될 시점이면 `LLMContextFrame`은 LLM
service의 input queue에 있고 어쩌면 이미 `:603`을 지났습니다. `:871`의 event handler는
**notification**이지 gate가 아닙니다. 관찰할 수는 있고, 결코 veto할 수는 없습니다.

그건 문체상의 반대가 아닙니다. 함수 body 하나 안의 줄 번호 두 개에 대한 산술이고, 그래서
[[design-boson-rules-on-pipecat]]가 그 기각을 *"decisive rather than stylistic"*이라고 부르는
것입니다.

**Rival B가 여전히 유용한 지점**, 그리고 이건 챙겨 둘 가치가 있습니다: observe-only rule들.
`end_signal`, `turn_counter`, `stage_round_tracker`는 veto하지 않습니다 — 분류하고 session에 finding을
씁니다. 그것들은 정당하게 `on_user_turn_message_added` handler 안에 살 수 있고 critical path에
비용을 0으로 지불합니다. §11의 bill을 깎아야 할 일이 생기면, 칼이 먼저 들어갈 곳이 거기입니다.

### 5.2 Rival C는 죽었고, 그것은 서로 다른 두 가지를 비용으로 치른다

generation을 시작시키고 취소하는 것은 공짜처럼 보입니다. pass path — 대부분의 turn — 에서는 아무것도
추가하지 않으니까요. 공짜가 아니고, 두 비용은 종류가 다릅니다:

**비용 1, latency.** `broadcast_interruption()` (`frame_processor.py:1017-1022`)이 중단시키고 당신은
처음부터 다시 시작합니다. 첫 generation의 TTFB를 지불하고 그다음 두 번째 것을 또 지불합니다. ch-11의
budget에 대고 보면 그건 작은 세금이 아닙니다. veto되는 모든 turn에서 LLM leg가 대략 두 배가 됩니다.

**비용 2, 그리고 이것은 latency 비용이 전혀 아닙니다.** `:605`와 당신의 veto 사이에 token이
생성되었고, TTS로 stream되었고, 어쩌면 회선 위에서 audio로 렌더링되었습니다. 규제된 동의(consent)
script text를 읽는 한국 보험 tele-sales agent에게, "봇이 하면 안 되는 말을 반 문장 하다가 멈췄다"는
latency regression이 아닙니다. 그것은 compliance event입니다. [[boson-script-engine]]은 script가 애초에
왜 model을 우회하는지에 대해 명시적입니다: *"Korean insurance-consent script text is legally fixed."*
audio가 재생된 뒤에 도착하는 veto는 veto가 아닙니다.

> 💡 **쉬운 설명 — 두 비용이 왜 "종류가 다른가"**
> 비용 1은 숫자로 흥정할 수 있습니다. "TTFB 두 배? 하루 100턴이면 감당 가능한데?" 같은 대화가
> 가능합니다. 비용 2는 흥정 대상이 아닙니다. 법적으로 고정된 고지 문구를 봇이 반쯤 잘못 말한 뒤
> 멈춘 것은 "조금 느렸다"가 아니라 "규정 위반이 발생했다"입니다. 이 두 번째 비용 때문에
> Rival C는 latency 예산 논쟁에서 이기더라도 여전히 탈락합니다.

> **당신의 도출이 seam을 다른 곳에 두었다면**, 그것을 판정하는 두 줄은
> `llm_response_universal.py:866` (push — 이것이 그 이후의 모든 것을 notification으로 만드는 것)과
> `openai/base_llm.py:601` (test — 이것이 그 이후의 모든 것을 cancellation으로 만드는 것)입니다.
> 그 둘 사이의 모든 것이 seam입니다. 그 바깥의 어떤 것도 아닙니다.

---

## 6. CHECK — 결론 (b): 모든 layer는 하나의 processor로 붕괴해야 한다

**그렇습니다, layer는 붕괴해야 합니다. cross-layer veto는 N개의 processor에 흩어진 채로 살아남을 수
없습니다.** [[custom-processor-guide]]의 *"the honest port splits its four layers into four
`FrameProcessor`s"*는 틀렸고, Constraint Two와 Three가 그 이유입니다.

Q2의 시나리오 모양으로 된 증명이 여기 있습니다.

**(a) processor 03에서 이미 나간 것은 무엇인가?** processor 04가 무언가를 평가하고 있으려면,
processor 03은 반드시 push했어야 합니다. 그것이 03에서 04로 frame이 가는 유일한 방법이니까요 —
`push_frame`, 즉 이웃에 enqueue이고 되돌릴 수 없습니다 (Constraint Two). 따라서 Q2(c)에 대한 답 —
경첩 — 은 이것입니다: **processor 03이 이미 commit하지 않았다면 processor 04는 돌고 있을 수
없습니다.** 두 event는 단지 순서가 있는 게 아니라 인과적으로 사슬로 묶여 있습니다. "layer 3이
staging하고 layer 4의 판정을 기다린다"는 processor의 linked list 안에서는 표현될 방법이 없습니다.
layer 4에게 data를 가져다주는 그 mechanism이 *곧* commit이기 때문입니다.

더 나쁜 것은, layer 03의 action이 frame만은 아니라는 점입니다. `Inject`는 `add_message`를 호출해
공유 `LLMContext`를 mutate합니다 — [[ch-09/read]]의 핵심 발견은 Pipecat이 **하나의 object에 대한
다수의 holder**를 최적화한다는 것이고, 따라서 그 mutation은 일어나는 즉시 모두에게 보이며, frame도
관여하지 않고 보류할 push도 없습니다. `PreTool`은 합성된 tool-call history를 append합니다. 04가
`Filter`라고 말할 시점이면 그 write들은 공유 context 안에 있고 un-push할 것이 없습니다.

**(b) rollback verb가 올바른 shape를 가지고 있는가?** 아니오, 그리고 이쪽이 더 날카로운 절반입니다.

| | boson | Pipecat |
|---|---|---|
| 무엇이 삭제되는가 | *바로 그 object*, `session.messages`를 뒤로 훑으며 identity를 비교해 찾음 | — |
| verb | `pipeline_user_message`의 identity delete (`gateway/layers/pipeline.py:205-239`) | `set_messages(list)` (`llm_context.py:377`) |
| 사용 가능한 handle | `:154-156`부터 들고 있던 Python object reference | 없음 — API는 list를 받고, 아무것도 반환하지 않으며, handle을 내주지 않음 |

`set_messages`는 *list 전체를 교체*입니다. 그것은 "이 snapshot을 복원하라"로 표현 가능하고, 그것은
"이 object를 삭제하라"보다 거친(coarse) 연산입니다. 거친 버전은 snapshot과 restore 사이에 mutate하는
당사자가 **하나**일 때에만 동등합니다. 네 processor에 흩어져 있으면, processor 01의 snapshot과
processor 04의 restore 사이에 다른 processor 셋, 같은 context object를 이미 들고 있을 수도 있는 LLM
service, 그리고 assistant aggregator가 있습니다. 거기서의 whole-list restore는 layer 04의 실수를
되돌리지 않습니다 — 모두의 write를 더 이상 의미 없는 시점으로 뭉개 버립니다.

> 💡 **쉬운 설명 — "identity delete"와 "list replace"의 표현력 차이**
> identity delete는 "이 사진 한 장만 빼줘"입니다. list replace는 "앨범을 아까 찍어둔 사본으로
> 통째로 교체해줘"입니다. 나 혼자 앨범을 만지고 있다면 둘은 결과가 같습니다 — 사본 이후에 내가
> 넣은 것만 사라지니까. 그런데 그 사이에 다른 세 사람이 각자 사진을 넣었다면, 통째 교체는 내
> 실수뿐 아니라 **남들의 정당한 작업까지** 지워 버립니다. 이것이 §9.3에서 `snapshot[:-1]`이
> "특수한 경우에만 옳다"고 말하게 되는 이유의 뿌리입니다.

**(c) PANEL THREE의 개수.** 네 processor 버전을 돌리고 빨간 frame을 세십시오. 빨간 frame 하나하나가
`set_messages`가 회수할 수 없는 것 하나입니다.

### 6.1 붕괴가 실제로 당신에게 청구하는 비용

비용이 없는 척하지 말고 값을 솔직하게 말합시다. [[custom-processor-guide]]는 결론에 대해서는
틀렸지만 비용에 대해서는 옳습니다. 하나의 processor로 붕괴시키면 다음을 포기합니다:

- **per-layer queueing.** processor 넷은 `FrameProcessorQueue` 넷과 독립적인 process task 넷을
  줍니다. processor 하나는 하나를 줍니다. layer 03의 LLM check가 같은 coroutine 안에서 layer 02와
  04를 block합니다.
- **per-layer observability.** [[ch-11/read]]의 observer plane은 *processor*마다
  `on_process_frame`을 발화시킵니다. processor 넷은 trace에 이름 붙은 행 넷을 주고, 하나는
  `BosonRuleProcessor`라는 행 하나를 주며 나머지는 당신 자신의 logging입니다.
- **per-layer interruption semantics.** `super().process_frame`은 processor마다
  `InterruptionFrame`에서 state를 clear합니다 (`frame_processor.py:839-841`). processor 하나는
  reset 하나를 뜻하고, 독립적으로 clear되기를 원했던 per-layer state는 당신 문제가 됩니다.

그것들은 진짜 비용입니다. 그것들은 *correctness* 비용이 아닙니다. cross-layer veto는 correctness
property이고, 그것을 지키는 방법은 정확히 하나뿐입니다.

> **당신의 도출이 "processor 넷, 그리고 `EventNotifier`로 조율하겠다"였다면** — 그건 옳은 본능이고
> 틀린 문제입니다. §13이 notifier pattern이 진짜로 Pipecat의 답인 지점을 보여줍니다. 그것은
> *notification*을 조율합니다. processor 03에게 un-push할 방법을 주지는 않습니다.

---

## 7. CHECK — mapping table

**열한 행.** 각 행은 당신이 반대할 수 있는 주장이고, 오른쪽 column이 그 이견을 판정하는 줄을 이름
부릅니다.

| # | boson mechanism | Pipecat에서의 집 | 무엇을 잃는가 | 판정하는 줄 |
|---|---|---|---|---|
| 1 | `@check(mode="sequential")` + first-non-continue short-circuit | processor 내부, **verbatim 이식** | 없음 | Pipecat은 rule scheduler를 아예 ship하지 않습니다 — 싸울 상대가 없습니다. `rules/engine.py:68-69`가 `process_frame` 안에서 변경 없이 돕니다. |
| 2 | `@check(mode="parallel")` under one `asyncio.gather` | 같은 processor, **같은 `asyncio.gather`** | 없음 | `rules/engine.py:74-80`. 당신은 coroutine 안에 있고, `gather`는 예전과 정확히 똑같이 동작합니다. |
| 3 | `Respond(text)` | `TTSSpeakFrame(text)`를 push하고 context frame을 **삼킨다** | 없음 | `frames.py:795-809` — `TTSSpeakFrame(text: str, append_to_context: bool = True)`. Flows의 `tts_say` action은 틀린 도구입니다: node-set 시점에만 발화하므로(`actions.py:104`) turn 단위 scripted line에는 지나치게 거칩니다. |
| 4 | `Inject(content)` | push하기 **전에** `context.add_message({...})` | 마지막 user message에 merge하는 option | `llm_context.py:361` `add_message`는 append합니다. boson의 `_merge_system_reminder`(`gateway/layers/pipeline.py:341-372`)는 `<system-reminder>…</system-reminder>`를 가장 최근 user message *안으로* 접어 넣는데 — 그렇게 하는 frame이나 context verb는 없습니다. append하거나 list 전체를 다시 쓰거나, 세 번째 수는 없습니다. |
| 5 | `PreTool(name, args, preamble)` | Flows의 `pre_actions` 안 `function` action — **정확히 맞음** | preamble을 첫 stream chunk로 쓰는 것 | `actions.py:279-285`: `elif previous_action_type == "function": … needs_wait = True`, 무조건. `"function"` action은 **항상** 기다리며, 그것이 정확히 boson의 synchronous-before-generation semantics입니다. preamble(`["감사합니다!"]`)은 그 앞에 순서 지어진 별도의 `tts_say`로 격하됩니다. |
| 6 | `Compact()` | **boson의** compactor를 호출하는 `function` pre-action | background-async compaction; pre-action은 block함 | Flows의 summary path는 쓰지 **마십시오**. `manager.py:801-822`: `asyncio.wait_for(..., timeout=5.0)`, 그리고 `TimeoutError` 시 warning을 로그하고 `update_config.strategy = ContextStrategy.APPEND`로 설정합니다 — **조용히 append로 격하**됩니다. 조용히 compact하지 않는 compaction은 compaction이 없는 것보다 나쁩니다. |
| 7 | `_GLOBAL_TOOLS` | `FlowManager(global_functions=[...])` | 없음 — 정확히 일치 | `manager.py:100`이 kwarg를 선언하고, `manager.py:654`가 그것을 섞습니다: `functions_list = self._global_functions + node_config.get("functions", [])`. 모든 node에서 global-then-node. |
| 8 | `ScriptEngine.process_turn(state, msg, registry)` | processor 안에서 **변경 없이** 돎 | 없음 — 이 시스템에서 가장 깔끔한 이식 | 전부 `@staticmethod`이고, dict를 받아 `(new_state, Action)`을 반환하며, input을 mutate하지 않습니다 ([[boson-script-engine]]). gateway coupling 0. `script_state`를 processor에 두거나(또는 `flow_manager.state`, `manager.py:157`) 호출하십시오. |
| 9 | `Continue()` | **action으로서의 집이 없음** | — | action은 round가 해소된 *뒤에* 실행되고, 그때는 user message가 이미 context에 있습니다 (`:863`). "계속 진행"은 effect가 아니라 effect의 부재이고, action list가 아니라 processor의 control flow에 속합니다. |
| 10 | `Pass()` | **action으로서의 집이 없음** | — | 9행과 같은 이유. `Pass()`는 이 layer가 투표를 사양하는 것입니다. "나는 사양한다"를 뜻하는 frame은 없습니다. branch하지 않음으로써 표현합니다. |
| 11 | `Filter(reason)` | **action으로서의 집이 없음** — 그것은 processor의 verdict | 이미 보낸 VAD interruption은 un-broadcast할 수 없음 | 이게 중요한 것입니다. `Filter`는 *pre-LLM routing verdict*입니다: push하지 말고, snapshot을 복원하고, return. 그것을 action으로 모델링하면 round 이후에 실행되고, `:863`에 따라 그때는 message가 이미 commit되어 있습니다. 또한: `FunctionFilter`(`filters/function_filter.py:21`)에 손을 뻗지 **마십시오** — 그것은 predicate로 pass/block을 정할 뿐 context rollback을 할 방법이 없습니다. |

**9행, 10행, 11행을 함께 읽으십시오.** 집이 없는 세 action은 정확히 effect가 아니라 *routing
verdict*인 셋입니다. 그건 우연이 아니고 챙겨 갈 가치가 있는 일반화입니다: **boson의 `Action` type은
"무엇을 할지"와 "진행할지 말지"를 뒤섞고 있고, Pipecat port는 그 둘을 분리합니다.** effect는 frame
이나 context write가 되고, verdict는 하나의 `process_frame` 안의 control flow가 됩니다.

**의도적으로 행이 아닌 것.** `StageTransition(target)`은 없습니다. §1이 이미 그것을 판정했기
때문입니다 — Fact Zero, `manager.py:588`, 평범한 code에서 호출됨. 그것은 더 이상 mapping 문제가
아니라 *sequencing* 문제이고, §9.4가 다룹니다.

역시 없는 것들, 그리고 3개월 차에 발견하는 대신 지금 없다는 것을 알아야 하는 것들: `skills`
(`StageDefinition.skills`)는 대응하는 Pipecat 개념이 전혀 없고, per-session attribute namespace
(`SharedLayerContext.__getattr__`/`__setattr__`)도 없으며 — `flow_manager.state`는 맨
`dict[str, Any]`이기 때문입니다 (`manager.py:157`) — `TOOL_PROCESSING`을 뜻하는 frame도 없어서
boson의 `AgentStatusTracker` 3-state model은 `BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame`으로
구동되는 2-state로 격하됩니다.

---

## 8. CHECK — the pipeline

```python
pipeline = Pipeline([
    transport.input(),
    stt,                       # Korean 8 kHz telephony STT
    BosonFillerGate(),         # boson layer 01
    user_aggregator,           # LLMContextAggregatorPair(context).user()
    BosonRuleProcessor(...),   # boson layers 02 / 03 / 04, Tier 1 + Tier 2
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])

# FlowManager(llm=llm, context_aggregator=pair, worker=worker,
#             global_functions=[...])  is NOT in this list.
```

entry 아홉 개. 자명하지 않은 모든 항목은 취향이 아니라 data dependency로 정당화됩니다. 당신의
list가 다르다면, 논증은 아래에 있고 그것은 falsifiable합니다.

### 8.1 `stt`와 `user_aggregator` 사이의 `BosonFillerGate`

또 두 개의 벽.

**더 앞은 안 됩니다.** `stt` 앞에서 유일한 frame은 `InputAudioRawFrame`입니다. `_is_filler_text()`는
문자열을 받습니다. gate를 position 1에 두면 rule에게 input이 없습니다 — 성능이 저하되는 게 아니라
**죽습니다**. PANEL ONE은 거기 떨어뜨리면 정확히 이것을 보여줍니다.

**더 뒤도 안 됩니다.** aggregator 뒤에서는 `:863`이 이미 돌았습니다: `"네"`가 공유 context에
`add_message`되었고 `:866`이 이미 push했습니다. 한 줄짜리 문자열 membership test가 context rollback
더하기 삼켜진 frame이 되어 버렸습니다. 시스템에서 가장 값싼 rule을 아무 이유 없이 가장 비싼 rule로
바꿔 놓은 것입니다.

**그리고 그것은 `InterimTranscriptionFrame`을 무조건 버려야 합니다.** 그렇지 않으면 downstream의
모든 keyword rule이 prefix에 발화합니다 (Constraint One(c)). Pipecat 자신의 `SentenceAggregator`가
`aggregators/sentence.py:50-51`의 맨 `return`으로 이것을 합니다. 그것을 복사하십시오.

### 8.2 `user_aggregator`와 `llm` 사이의 `BosonRuleProcessor`

이것은 결론 (a)를 list index로 렌더링한 것입니다. `push_aggregation`이 `:863`에서 쓰고 `:866`에서
push하며, `base_llm.py:601`이 소비합니다. 당신의 processor는 그 틈에 들어갑니다. §5가 논증이고,
이것은 그것이 착지하는 자리일 뿐입니다.

### 8.3 `transport.output()` 뒤의 `assistant_aggregator`

이것은 Pipecat의 하우스 패턴이고 [[ch-09/read]]가 Pipecat 쪽 이유를 이미 설명했습니다. 여기서 이름
붙일 가치가 있는 것은 **boson이 다른 이유로 같은 invariant를 가진다**는 점입니다: TTS 도중에
interrupt된 `Respond()`는 말해진 것으로 기록되면 안 됩니다. assistant aggregator를
`transport.output()` 앞으로 옮기면 history는 고객이 결코 듣지 못한 text를 기록합니다 — 그것이 바로
boson의 identity 기반 rollback이 막으려고 존재하는 drift가 다른 문으로 들어온 것입니다. 두 시스템,
두 근거, 하나의 ordering. 그 일치는 좋은 신호입니다.

### 8.4 transition turn에서 rule processor는 context frame을 SWALLOW한다

이것이 반직관적인 항목이고 first principle에서 나올 조언을 뒤집습니다.

"두 가지가 inference를 trigger할 수 있다"에 대한 순진한 해법은 *ordering*입니다: 먼저
`set_node_from_config()`를 호출하고, 그다음 당신의 frame을 push하고, node의 `LLMRunFrame`이 경주에서
이기게 한다. **그것은 동작하지 않습니다**, 그리고 이유는 구조적입니다:

**`src/pipecat/flows/manager.py:838-841`**

```python
            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))

            await self._worker.queue_frames(frames)
```

**worker**에 대한 `queue_frames`. [[ch-10/read]]의 핵심 발견은 `FlowManager`가 `FrameProcessor`가
아니라는 것입니다 — 그것은 **head**에 주입해서 바깥에서 pipeline을 구동합니다. 따라서 node의
`LLMMessagesAppendFrame`, `LLMSetToolsFrame`, `LLMRunFrame`은 `transport.input()`에서 진입해
`llm`에 도달하기 전에 `stt` → `BosonFillerGate` → `user_aggregator`를 통과해야 합니다. 당신이 push한
`LLMContextFrame`은 이미 `llm`에 인접해 있고 즉시 도달합니다.

두 호출의 순서를 정한다고 해서 그것들의 도착 순서가 정해지지 않습니다. 그들은 서로 다른 거리를
이동합니다.

> 💡 **쉬운 설명 — 거리 vs 순서**
> 두 사람에게 동시에 "출발!"이라고 외쳤다고 해서 동시에 도착하지 않습니다. 한 명은 옆방에서
> 출발하고 한 명은 건물 반대편에서 출발한다면요. `FlowManager`가 queue하는 frame은 pipeline 맨
> 앞에서 출발해 processor 셋을 거쳐야 하고, 당신의 processor가 push하는 frame은 `llm` 바로
> 옆에서 출발합니다. "먼저 호출했다"는 "먼저 도착한다"와 아무 관계가 없습니다. 이것이 §12.4의
> 일반형이고, 이 design 전체에서 "그냥 호출 순서를 맞추자"가 결코 해법이 아닌 이유입니다.

**Swallow는 race를 구성적으로 제거합니다.** transition turn에서는: context를 mutate하고,
`set_node_from_config`를 호출하고, push **없이** `return`합니다. 이제 시스템 안에 inference를
trigger하는 frame이 정확히 하나 존재하고, 그것은 `manager.py:707-709`의 node의 `LLMRunFrame`입니다:

```python
            respond_immediately = node_config.get("respond_immediately", True)
            if respond_immediately:
                await self._worker.queue_frames([LLMRunFrame()])
```

transition이 아닌 turn에서는 정상적으로 push합니다.

테스트해야 할 failure mode는 **양쪽 방향 모두**이고 §14의 두 번째 prototype이 그것을 셉니다:
양쪽 path를 다 swallow하면 generation이 **0**개가 되고(통화 중간에 봇이 침묵), push하면서 transition
까지 하면 **2**개가 됩니다(봇이 자기 자신에게 답함).

---

## 9. CHECK — the code

class 이름을 짓는 것은 class를 설계하는 것이 아닙니다. 여기 body가 있고, 그것은
[[custom-processor-guide]]가 열한 chapter 동안 향해 온 그 모양입니다 — *"four lines of ceremony
around one `if isinstance(...)` chain."*

### 9.1 companion gate

**`BosonFillerGate` — boson layer 01, 약 16줄**

```python
class BosonFillerGate(FrameProcessor):
    """boson layer 01: drop Korean backchannels before they reach the aggregator."""

    def __init__(self, *, session: SessionState, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)           # 1  never optional

        if isinstance(frame, InterimTranscriptionFrame):        # 2  Constraint One(c)
            return                                              #    unconditional drop

        if isinstance(frame, TranscriptionFrame):
            status = self._session.pre_turn_status               # 3  korean_fillers.py:66
            if status in ("generating", "tool_processing") and _is_filler_text(frame.text):
                self._signals.add(Signal(reason=f"filler_word:{frame.text}"))
                return                                          # 4  drop == not pushing

        await self.push_frame(frame, direction)                 # 5  everything else
```

줄 2는 optional이 아니고 성능 튜닝도 아닙니다. `InterimTranscriptionFrame`은 `TextFrame`을
subclass하므로(`frames.py:476`), 그 `return`이 없으면 문장 `"네 그런데…"`의 interim hypothesis
`"네"`가 backchannel로 filter되고 고객의 질문이 사라집니다. Constraint One(c)를 한 줄로 강제한
것입니다.

줄 3은 `pre_turn_status` 읽기이고, 이 gate가 `FunctionFilter`가 아니라 stateful인 이유입니다.
`korean_fillers.py:66`이 그것으로 gate하고, 그 gate를 빼면 filter가 자기 자신을 filter합니다 —
동의 질문에 대한 진짜 `"네"` 대답을 먹어 버리는데, 한국 보험 판매에서 이 파일이 할 수 있는 가장
나쁜 일이 바로 그것입니다.

### 9.2 rule processor

**`BosonRuleProcessor.process_frame` — 약 40줄**

```python
class BosonRuleProcessor(FrameProcessor):

    def can_generate_metrics(self) -> bool:                      # see §11.4 — not optional
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)            #  1  literally first

        if not isinstance(frame, LLMContextFrame):               #  2  the only frame we own
            await self.push_frame(frame, direction)              #     everything else passes
            return

        ctx = frame.context                                      #  3  the SAME object :863 wrote
        snapshot = list(ctx.get_messages())                      #  4  COPY — see §9.3
        user_message = _text_of(snapshot[-1])                    #  5  the turn :863 appended

        await self.start_ttfb_metrics()                          #  6  the bill — see §11.4

        staged: list[Action] = []                                #  7  boson Phase 1
        for engine in self._engines:                             #     02-analyzer, 03, 04
            decision = await engine.evaluate_sequential(         #  8  TIER 1: 11 checks
                snapshot, user_message, self._ctx_for(engine),   #     pure Python, sub-ms
            )
            staged.extend(decision.actions)
            if decision.short_circuit:                           #  9  engine.py:68-69, verbatim
                break

        if not _has_veto(staged):                                # 10  TIER 2: the 2 llm checks
            results = await asyncio.gather(*[                    # 11  engine.py:74-80, verbatim
                check(snapshot, user_message, self._ctx_for(check))
                for check in self._llm_checks                    #     intent_rules, sentiment
            ])                                                   #     wall clock = max, not sum
            staged.extend(_flatten(results))

        await self.stop_ttfb_metrics()
        verdict = resolve_actions(staged)                        # 12  ACTION_PRIORITY, unchanged

        if verdict.filter is not None:                           # 13  THE VETO
            ctx.set_messages(snapshot[:-1])                      # 14  rollback — see §9.3
            self._signals.add(Signal(reason=verdict.filter.reason))
            return                                               # 15  no push == no inference

        if verdict.respond is not None:                          # 16  script text, verbatim
            ctx.set_messages(snapshot[:-1])                      #     the LLM never sees this turn
            ctx.add_message({"role": "assistant", "content": verdict.respond.text})
            await self.push_frame(TTSSpeakFrame(verdict.respond.text,
                                                append_to_context=False))
            return                                               # 17  swallow the context frame

        for inject in verdict.injects:                           # 18  mutate in place, then fall
            ctx.add_message({"role": "developer", "content": inject.content})

        if verdict.transition is not None:                       # 19  Fact Zero, cashed
            result = self._stage_machine.transition(             # 20  boson legality PRE-CHECK
                self._session.active_stage, verdict.transition.target,
            )
            if result.success:
                self._session.active_stage = result.new_stage.name
                await self._flow_manager.set_node_from_config(   # 21  manager.py:588
                    self._node_for(result.new_stage.name),
                )
                return                                           # 22  SWALLOW — see §8.4
            logger.warning(f"illegal transition rejected: {result.error}")

        await self.push_frame(frame, direction)                  # 23  the ordinary turn
```

### 9.3 그 body에서 excerpt가 말하는 것과 다른 두 줄

> **Source correction, 그리고 이것은 rollback과 no-op의 차이입니다.**
> [[design-boson-rules-on-pipecat]] §4는 snapshot을 *"snapshot then
> `context.set_messages(snapshot)`"*로 쓰고, figure spec은
> `snapshot = frame.context.get_messages()`로 씁니다. **`llm_context.py`를 열면 그것은 bug입니다.**
>
> **`src/pipecat/processors/aggregators/llm_context.py:244-260`**
>
> ```python
>         if llm_specific_filter is None:
>             messages = self._messages
>         else:
>             messages = [ ... ]
>         if truncate_large_values:
>             messages = LLMContext._truncate_large_values_from_messages(messages)
>
>         return messages
> ```
>
> filter도 없고 truncation도 없으면 `get_messages()`는 `self._messages` 자기 자신을 반환합니다 —
> **live list**이고, [[ch-09/read]] §2.3이 확립한 그대로이며, 의도적으로 그렇습니다. 그리고
> `set_messages`는 `self._messages[:] = messages`입니다 (`:383`).
>
> 그 둘을 합치십시오. `snapshot = ctx.get_messages()`는 `snapshot`을 `ctx._messages`와 같은 list
> object에 bind합니다. round 중의 어떤 `add_message`든 *snapshot에도* append합니다. 그다음
> `ctx.set_messages(snapshot)`은 list를 자기 자신에게 slice-assign합니다.
> **rollback이 조용히 아무것도 복원하지 않습니다** — exception도, warning도 없이, veto하지 않는
> veto가 됩니다.
>
> 줄 4의 `list(...)`는 방어적 style이 아닙니다. 그것은 동작하는 code와 조용히 commit해 버리는
> two-phase commit 사이의 차이입니다.

> 💡 **쉬운 설명 — aliasing 버그를 한 줄로**
> ```python
> snapshot = ctx.get_messages()   # 별칭(alias)이다. 사본이 아니다.
> ctx.add_message(x)              # snapshot 에도 x 가 들어간다 (같은 객체니까)
> ctx.set_messages(snapshot)      # self._messages[:] = self._messages  →  아무 일도 안 일어남
> ```
> 이 코드는 예외도 안 나고 로그도 안 남깁니다. 테스트에서 "veto 했더니 메시지가 그대로 있네?"를
> 눈으로 보기 전까지 아무도 모릅니다. `list(...)` 한 번이 이 전체를 고칩니다.

두 번째: **줄 14는 `snapshot`이 아니라 `snapshot[:-1]`입니다.** boson의 `Filter`는 append된 user
message를 object identity로 삭제합니다 (`gateway/layers/pipeline.py:205-239`). 당신의 processor는 `:863`이
그것을 append한 *뒤에* 처음으로 context를 보므로, `snapshot[-1]`이 **바로 그** message입니다 —
position이 boson에게 identity가 준 것을 회복합니다. `set_messages(snapshot)`은 당신의 round는
되돌리지만 사용자의 발화를 history에 남기고, `set_messages(snapshot[:-1])`이 boson과 일치합니다.

그 비용에 대해 솔직해야 합니다: 이 positional recovery가 옳은 것은 오직 `:863`과 여기 사이에서
당신이 유일한 writer이기 때문이고, 그것이 참인 것은 *layer가 붕괴했기 때문*입니다. §6의 (b)행은
identity가 **일반적으로는** list replace로 표현되지 않는다고 말했습니다. 이것은 표현되는 특수한
경우이고 — 그 특수한 경우를 만들어내는 것이 바로 collapse입니다. processor 넷을 유지했다면
`snapshot[-1]`은 추측이었을 것입니다.

### 9.4 two-tier 분할, 그리고 줄 10이 거기 있는 이유

줄 8은 **Tier 1**입니다: deterministic check 열한 개 — `korean_filler_filter`, `end_signal`,
`response_classifier`, `hesitation_hook`, `turn_counter`, `stage_round_tracker`,
`preload_on_question`, `script_flow`, `tool_gate`, `help_responder`, `auto_compact`. 문자열과 dict
위의 순수 Python. sub-millisecond. 공짜.

줄 11은 **Tier 2**입니다: `check_type="llm"`인 check 두 개 — `intent_rules`(priority 30)와
`sentiment_tracker`(priority 10), boson에서는 오늘 둘 다 `mode="parallel"`입니다
([[design-boson-rules-on-pipecat]] §1). 하나의 `gather` 아래 있으므로 wall clock은 `sum`이 아니라
`max`입니다. 그것은 당신이 공짜로 물려받는 boson의 design decision이고, bill을 절반으로 줄입니다.

줄 10 — `if not _has_veto(staged)` — 은 boson에 없습니다. 그것은 이 port가 가능하게 만드는
optimisation입니다: Tier 1이 이미 `Filter`나 `Respond`(`ACTION_PRIORITY` 0과 1)를 만들어냈다면
Tier 2의 결과는 결과를 바꿀 수 없으므로, 건너뛰고 250-400 ms를 아낍니다. filler turn에서는 —
Lina에게는 고객이 `"네"`를 끊임없이 말하므로 turn 중 큰 비중입니다 — 0을 지불합니다. ship하기 전에
boson의 arbitration에 대고 검증하십시오: `RuleEngine.evaluate`는 non-continue인 parallel 결과를 모두
보관하고 phase에 걸쳐 transition을 arbitrate하므로(`engine.py:100-127`), Tier 2를 건너뛰는 것은
*어떤 signal이 쓰이는지*를 바꾸는 것이지 어떤 action이 이기는지를 바꾸지 않습니다. 나중 layer가
`session.sentiment`를 읽는다면 당신은 behaviour를 바꾼 것입니다. 의도적으로 결정하십시오.

### 9.5 줄 1, 다시 한 번

`await super().process_frame(frame, direction)`는 문자 그대로 첫 statement이고, 그것을 건너뛰는 것은
style 위반이 아니라 **black hole**입니다. base implementation은 no-op이 아닙니다:

**`src/pipecat/processors/frame_processor.py:827-847`**

```python
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

건너뛰면: 당신의 processor는 internal task를 결코 시작하지 않고(`StartFrame` → `__start`), barge-in
에서 state를 결코 clear하지 않고(`InterruptionFrame` → `_start_interruption`), metric을 결코 멈추지
않으며, **모든 observer에게 보이지 않습니다** — 즉 ch-11의 observability plane 전체에 보이지
않습니다. 그것이 하지 *않는* 일도 함께 보십시오: **frame을 push하지 않습니다.** forwarding은 전적으로
당신의 일이고, 그래서 줄 2의 `else` 모양 branch가 존재합니다.

figure의 PANEL FOUR에는 "`super()` 호출을 삭제" 스위치가 있습니다. 그것을 켜고 [[ch-01/read]]의
black hole이 당신 자신의 processor에서 재현되는 것을 보십시오.

---

## 10. code가 무엇이 빠졌는지에 대해 정직한 지점

§9.2의 세 가지는 Pipecat 대응물이 없는 boson code에 대한 placeholder이고, 제가 쓰기 귀찮아서 안 쓴
helper 함수가 아니라 그런 것으로 인식해야 합니다.

**`self._ctx_for(engine)` — `SharedLayerContext`에는 집이 없습니다.** boson의 per-turn proxy
(`layers/context.py:17`)는 알 수 없는 attribute read를 `:40`에서, write를 `:54`에서 살아 있는
`SessionState`로 그대로 전달합니다. 그래서 `session.checklist_state`, `session.fired_rules`,
`session.script_state`가 rule 안에서 read/write되고 persist됩니다. Pipecat은 맨 `dict[str, Any]`인
`flow_manager.state`를 제공합니다 (`manager.py:157`). **zero-edit 선택은 진짜 `SessionState` object를
processor에 두고 그것을 `session` argument로 전달하는 것입니다.** 열세 개 rule의 `getattr(session, …)`
를 dict lookup으로 다시 쓰는 것은 아무것도 사주지 않으면서 열세 개 파일의 diff를 비용으로
치릅니다.

**`self._signals` — `SignalQueue`는 평범한 object로 남습니다.** `get_recent(seconds, source_layer)`
(`layers/signals.py:63`)는 append-only list에 대한 **query**입니다. §13이 그 verb가 왜 bus에 대응물이
없는지를 길게 설명합니다.

**`self._stage_machine` — 순수 validator로 유지합니다.** 줄 19-22가 `set_node_from_config` *전에*
`StageMachine.transition()`을 호출합니다. Flows에는 from→to check가 어디에도 없습니다:
`_validate_node_config` (`manager.py:867-898`)는 정확히 두 가지만 검사합니다 — `task_messages`가
존재하는지, 그리고 각 `functions` entry가 callable이거나 `FlowsFunctionSchema`인지. node registry는
없고, `get_or_generate_node_name`은 이름 없는 node에 대해 `str(uuid.uuid4())`로 fallback합니다.
boson의 `StageMachine`을 버리면 **모든** transition legality를 조용히 버리는 것입니다. §12.1에
자세한 내용이 있습니다.

---

## 11. THE LATENCY BILL — 이제 값을 매길 수 있다, ch-11이 budget을 만들었으므로

[[ch-11/read]]는 budget에 *rule evaluation*이라는 label이 붙은 슬롯을 남겨 두었고, 아직 아무것도
그것을 채우지 않았기 때문에 비워 두었습니다. 채우십시오.

### 11.1 Tier 1은 공짜다

문자열 하나와 dict의 list 하나에 대한 순수 Python check 열한 개. substring test, dict lookup,
`re.split` 하나. sub-millisecond이고, 수백 밀리초 단위로 표시된 budget에는 나타나지 않습니다.
무시하십시오.

### 11.2 Tier 2가 bill이다: 매 turn 약 250-400 ms

check 둘, 둘 다 `check_type="llm"`, 둘 다 `mode="parallel"`, 둘 다 하나의 `gather` 아래.
`evaluate_intent_rules` (`transition_detector.py:82`)가 `_llm_match_descs`
(`intent_matcher.py:205-271`)를 호출합니다 — **하나의 batched call**, model은
`("boson", "Qwen3.6-27B-FP8")`, `temperature=0.1` (`llm_config.py:20,34`), output은 comma로 구분된
index list 또는 `"none"` (~5 token). `sentiment_tracker`가 동시에 발화합니다. 따라서 비용은
**Qwen3.6-27B TTFB 하나 더하기 output token 몇 개 ≈ 250-400 ms**이고, gather되어 있으므로 `sum`이
아니라 `max`입니다.

### 11.3 실재하는 것에 대고 크기를 재라

Pipecat은 측정된 STT reference를 ship합니다:

**`src/pipecat/services/stt_latency.py:37-46`**

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
```

> **Source correction.** [[design-boson-rules-on-pipecat]] §3은 *"0.45 s for Deepgram"*을
> 인용합니다. **ship되는 상수는 `0.35`**입니다 (`stt_latency.py:45`). `0.45`는 그 파일에 정확히 한 번,
> `:27`의 docstring 안에 나타나며 *측정된* 값을 전달하는 방법을 보여주는 예시입니다:
> `stt = DeepgramSTTService(api_key="...", ttfs_p99_latency=0.45)` — 예시이지 reference가 아닙니다.
> **0.35**를 쓰십시오. chapter spec의 "0.35-0.45 s" 범위는 둘 다 포괄하지만, 당신이 아무것도 하지
> 않으면 code가 실제로 쓸 숫자는 0.35입니다.

이제 산술. budget의 pre-LLM 절반에 대해:

| leg | 비용 | source |
|---|---|---|
| STT finalisation (Deepgram reference) | 0.35 s | `stt_latency.py:45` |
| 한국어 8 kHz telephony STT | **table에 항목 없음** | benchmark해야 함 — [[ch-06/read]] §19 |
| Tier 1 rules | < 0.001 s | 순수 Python check 11개 |
| **Tier 2 rules** | **0.25 – 0.40 s** | Qwen3.6-27B TTFB 하나, gather됨 |

**Tier 2는 pre-LLM 절반을 대략 두 배로 만듭니다.** 반올림 오차가 아니라 — STT 하나 값어치의
지연이, 모든 단일 turn에, sales LLM이 token 하나 보기 전에 추가되는 것입니다.

### 11.4 `LatencyBreakdown`에 나타나게 만들라, 그리고 excerpt가 여기서 틀린 것에 주의하라

chapter spec은 processor를 `start_processing_metrics()`/`stop_processing_metrics()`로 감싸라고, 안
그러면 "bill이 `LatencyBreakdown`에 결코 나타나지 않는다"고 말합니다. **그것은 절반만 맞고, 틀린
절반이 당신의 오후 하나를 비용으로 가져갑니다.** 독립적인 gotcha 둘, 둘 다 검증되었습니다:

**Gotcha 1 — `LatencyBreakdown`은 processing metric을 결코 읽지 않습니다.**

**`src/pipecat/observers/user_bot_latency_observer.py:322-341`**

```python
        now = time.time()
        for metrics_data in frame.data:
            if isinstance(metrics_data, TTFBMetricsData) and metrics_data.value > 0:
                self._ttfb.append(
                    TTFBBreakdownMetrics(
                        processor=metrics_data.processor,
                        model=metrics_data.model,
                        start_time=now - metrics_data.value,
                        duration_secs=metrics_data.value,
                    )
                )
            elif isinstance(metrics_data, TextAggregationMetricsData):
```

`TTFBMetricsData`와 `TextAggregationMetricsData`. 그게 목록 전부입니다.
`ProcessingMetricsData`는 존재하지만(`metrics/metrics.py:99`) 이 observer가 결코 읽지 않습니다.
`LatencyBreakdown` 자신의 field가 그것을 확인해 줍니다 (`:107-111`): `ttfb`, `text_aggregation`,
`user_turn_start_time`, `user_turn_secs`, `function_calls`. **processing 슬롯은 없습니다.**

그러니 `start_ttfb_metrics()` / `stop_ttfb_metrics()`를 쓰십시오 — §9.2의 줄 6이 하는 일입니다.
이름을 약간 오용하는 것이지만(byte-stream되는 것이 없으므로) breakdown observer가 듣고 있는 유일한
channel이고, 그 결과 당신의 rule processor가 `chronological_events()` (`:113-140`) 안에 `stt`, `llm`,
`tts`와 나란히, start time으로 정렬되어, 당신이 고른 `processor` label과 함께 나타납니다.

**Gotcha 2 — 평범한 `FrameProcessor`는 metric을 전혀 방출하지 않습니다.**

**`src/pipecat/processors/frame_processor.py:488-494`**

```python
    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False
```

`False`. 그리고 모든 metric method가 그것으로 guard되어 있습니다 — `start_ttfb_metrics`는 `:511`,
`stop_ttfb_metrics`는 `:528`, `start_processing_metrics`는 `:570`, 전부
`if self.can_generate_metrics() and self.metrics_enabled:`로. §9.2의 override가 없으면 당신의
`start_ttfb_metrics()` 호출은 **조용한 no-op**이고 bill은 어디에도 나타나지 않습니다.

세 가지, 전부 필수이며, 아니면 당신은 눈을 감고 비행하는 것입니다:

1. processor에 `def can_generate_metrics(self) -> bool: return True`.
2. `PipelineParams`에 `enable_metrics=True` (`metrics_enabled`가
   `self._setup.enable_metrics`를 읽습니다, `frame_processor.py:419`).
3. processing 쌍이 아니라 `start_ttfb_metrics` / `stop_ttfb_metrics`.

> 💡 **쉬운 설명 — 왜 "TTFB metric의 오용"이 정답인가요?**
> 이름만 보면 TTFB(time to first byte)는 stream을 내는 service용이고 rule processor에는 안
> 맞습니다. 하지만 observability는 "의미상 맞는 이름"이 아니라 "관측자가 실제로 구독하는
> channel"로 결정됩니다. `UserBotLatencyObserver`는 TTFB와 text-aggregation 두 종류만 읽으므로,
> 당신의 250-400 ms를 그 breakdown에 올리는 방법은 TTFB channel뿐입니다. 이름의 정확성보다
> 숫자가 보이는 것이 중요합니다.

### 11.5 product owner에게 빚진 한 문장

지불하지 않는 대안은 Tier 2를 *LLM의 첫 token과 concurrent하게* 돌리고 완료되면
`set_node_from_config()`를 호출하는 것인데 — 그러면 모든 stage 변경이 **한 turn 늦게** 착지합니다.
이미 답해 버린 turn을 veto할 수 없으므로, 그 변형은 in-turn `Filter`와 `Respond`도 포기합니다.

trade를 한 문장으로 진술하고 장식하지 마십시오:

> **In-turn veto와 in-turn steering은 매 turn 250-400 ms를 비용으로 한다. Next-turn transition은
> 0 ms이고 veto를 포기한다. boson은 오늘 250-400 ms를 지불하고 있다.**

figure의 PANEL FIVE에 스위치가 있습니다. 켜고 막대가 0으로 가고 stage 변경이 한 turn 오른쪽으로
미끄러지는 것을 보십시오.

선택을 요구하지 않으면서 깎는 두 가지 방법:

- **§5.1의 잔여물.** observe-only rule 셋(`end_signal`, `turn_counter`, `stage_round_tracker`)을
  `on_user_turn_message_added` handler로 옮기십시오. 어차피 veto할 수 없고, critical path 밖에서는
  0을 비용으로 합니다.
- **§9.4의 줄 10.** Tier 1이 이미 veto했으면 Tier 2를 건너뛰십시오. filler가 많은 한국어 tele-sales
  트래픽에서는 turn 중 큰 비중이고 behaviour 비용은 0입니다 — *단*, Tier 2가 썼을 signal을
  downstream의 무엇도 읽지 않는다는 것을 검증했다면.

---

## 12. FOUR REAL FRICTIONS, 어느 것도 치명적이지 않음

0개월 차에 적어 두지 않으면 2개월 차에 물어뜯을 것들입니다.

### 12.1 transition legality를 잃는다

boson: `StageDefinition.transitions`가 합법적인 successor의 whitelist이고,
`StageMachine.transition()`은
`TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")`를 반환합니다
(`stage/machine.py:57-60`, [[boson-stage-machine]] 경유).

Flows: 없음. `_validate_node_config` (`manager.py:867-898`)는 두 가지를 검사하고 어느 것도 from→to
test가 아닙니다. node registry가 없습니다. `self._current_functions: set[str]`
(`manager.py:148`)은 `:704`에서 할당되고 — [[flows-state-machine]]에 따르면 — **codebase 어디에서도
읽히지 않습니다**. 그것은 gate가 아니라 죽은 state입니다.

**완화책, 그리고 값이 쌉니다:** boson의 `StageMachine`을 `set_node_from_config` 앞에 앉은 순수
pre-check validator로 유지하십시오 (§9.2의 줄 19-22). prompt와 tool 배관은 버리고 edge whitelist를
유지하십시오. [[boson-stage-machine]]의 guideline은 여기서 verbatim으로 반복할 가치가 있습니다:
*"port the edge whitelist first, not the prompts."* 그리고 그것이 이름 붙인 failure mode에
주목하십시오 — `stage_config.py`의 모든 `v0.7.5 (#12)` regression 주석은 whitelist가 조용히 거부한
transition을 rule이 방출한 것이고, *"invisible in logs unless you check `TransitionResult.error`."*
줄 22의 `logger.warning`은 장식이 아닙니다.

### 12.2 모든 Flows transition은 기본값으로 inference trigger다

`respond_immediately`의 기본값은 `True`입니다 (`flows/types.py:203-204, 237`). 그래서 사용자의 turn
에서 발화된 rule 구동 transition은 `manager.py:707-709`에서 `LLMRunFrame()`을 queue하고 봇이
**말합니다**. boson의 transition은 agent loop가 돌기 *전에* 일어나는 조용한 장부 기록입니다.

**완화책:** 이식된 node에는 `respond_immediately=False`가 필요합니다 — 단, node가 유일한 inference
trigger가 되기를 의도적으로 원하는 turn은 제외합니다(§8.4). 그것은 global setting이 아니라 node별
결정이고, 어느 방향으로든 틀리면 §14의 두 번째 prototype입니다.

### 12.3 silent transition이라는 것 자체가 없다

`respond_immediately=False`여도, node set은 결코 공짜가 아닙니다:

**`src/pipecat/flows/manager.py:838-839`**

```python
            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))
```

`LLMSetToolsFrame`이 **무조건** append됩니다. 그리고 node가 function을 선언하지 않으면
`formatted_tools`는 `NOT_GIVEN`이고 (`manager.py:670-672`) node는 **tool set을 비웁니다**. 한편
`task_messages`는 `Required[list[dict]]` key입니다 (`types.py:224`).

no-op node에 가장 가까운 것은 `task_messages=[]`이고, `_validate_node_config`가 key *존재*만
검사하므로 validation을 통과합니다. 그래도 `LLMMessagesAppendFrame(messages=[...])` 하나와
`LLMSetToolsFrame` 하나를 방출합니다.

**Lina에 대한 귀결:** 이것은 `metatool/router.py`에 있는 boson의 `_allowed_tools_var` ContextVar
tool gate와 정면으로 충돌합니다 ([[boson-stage-machine]]이 그 충돌을 표시합니다). 광고되는 tool
array에 대해 둘 다 authoritative한 시스템 둘, 그중 하나는 모든 node set마다 재방출.
**하나를 고르십시오.** Flows가 tool을 소유하면 ContextVar gate를 삭제하고, router가 tool을
소유하면 node에 `functions`를 절대 두지 마십시오.

### 12.4 ordering이 queue ordering이 된다

boson은 `core._apply_stage_transition`을 `run_agent_loop` 전에 **inline**으로 호출합니다. 호출
순서가 실행 순서입니다. Flows는 pipeline head에서 frame을 queue하므로 (`manager.py:841`),
in-pipeline processor와 flow 자신의 frame이 호출 순서로 실행되는 대신 같은 queue를 통해 경주합니다.

이것이 §8.4의 구체적 race의 일반형이고, 이 design 어디에서도 "그냥 두 호출을 올바른 순서로 하면
된다"가 fix가 아닌 이유입니다. 당신의 processor가 push하는 것과 `FlowManager`가 queue하는 것 사이의
ordering을 추론하고 있는 자신을 발견할 때마다, 멈추고 각각이 이동하는 *거리*를 확인하십시오.

---

## 13. BUS를 side channel로 — 그리고 그것이 틀린 도구인 때

당신은 `src/pipecat/bus/`를 보고 그것이 `SharedLayerContext`와 `SignalQueue`의 자연스러운 집이라고
생각할 것입니다. 아닙니다, 그리고 이유는 없는 verb 하나입니다.

### 13.1 bus가 무엇인가

**worker 사이의** 이름 주소 방식 pub/sub. 단위는 `BusMessage` dataclass이고, 전달은 단방향
callback입니다 — `async def on_bus_message(self, message) -> None` (`bus/subscriber.py:25`);
`BusSubscriber` (`bus/subscriber.py:12`)는 `name` property 하나 더하기 그 method 하나이고, 그것이
receive-side contract 전부입니다. `bus/bus.py:153-160`의 fan-out은 모든 subscriber에게 모든 message를
주고, `target`은 *받는 쪽*이 filter합니다.

**read API가 없습니다.** `bus.get()`도, snapshot도, query도 없습니다. bus에게 현재 state가 무엇인지
물어볼 수 없습니다. 무언가 일어났을 때 통보받을 수만 있습니다. 게다가 system이 아닌 모든 message에
대해 queue hop 두 번어치의 staleness가 붙습니다 — `_router_task` (`bus/bus.py:169`)가 data를
`data_queue`로 보내고 `_data_dispatch_task` (`:187`)가 그것을 비웁니다.

### 13.2 boson의 layer들이 실제로 필요로 하는 것

verb 셋, 그리고 bus는 그중 하나도 가지고 있지 않습니다:

| boson | verb | 위치 |
|---|---|---|
| `SharedLayerContext.__getattr__` / `__setattr__` | **live state에 대한 동기적 read/write** | `layers/context.py:40,54` — 진짜 `SessionState`로 그대로 proxy |
| `AgentStatusTracker.get_status()` | **poll** | `layers/status.py:59` |
| `SignalQueue.get_recent(seconds, source_layer)` | append-only list에 대한 **query** | `layers/signals.py:63` |

read, poll, query. bus가 제공하는 것은 push입니다. `SharedLayerContext`를 그 위에 이식하면 모든
layer가 자기 replica를 유지해야 하고, 오늘은 살아 있는 `SessionState`에 write-through하는
`session.counter += 1`이 **조용히** 아무도 다시는 보지 못하는 local copy에 대한 write로 바뀝니다.
조용히. error도, warning도 없이, 틀린 답으로.

> 💡 **쉬운 설명 — push와 read/poll/query가 왜 대체 불가능한가**
> push(bus)는 "무슨 일이 생기면 알려줄게"입니다. read는 "지금 값이 뭐야?"이고, poll은 "지금
> 상태 어때?"이며, query는 "최근 5초 안에 layer 02에서 나온 signal 다 줘"입니다. push만 있는
> 시스템에서 read/poll/query를 흉내 내려면 각 구독자가 지금까지 받은 것을 자기 안에 쌓아
> **자기만의 사본**을 유지해야 합니다. 사본이 여러 개면 `+= 1` 같은 read-modify-write가 즉시
> 깨지고, 그 깨짐은 예외가 아니라 오답으로 나타납니다.

### 13.3 Pipecat 자신의 답은 무엇인가

bus가 아닙니다. `ParallelPipeline`의 서로 다른 두 branch에 있는 processor 네 개가 조율해야 할 때,
ship되는 voicemail extension이 무엇을 하는지 보십시오:

**`src/pipecat/extensions/voicemail/voicemail_detector.py:634-648`**

```python
        # Create notification system for coordinating between components
        self._gate_notifier = EventNotifier()  # Signals classification completion
        self._conversation_notifier = EventNotifier()  # Signals conversation detected
        self._voicemail_notifier = EventNotifier()  # Signals voicemail detected

        # Create the processor components
        self._classifier_gate = ClassifierGate(self._gate_notifier, self._conversation_notifier)
        self._conversation_gate = ConversationGate(self._voicemail_notifier)
        self._classification_processor = ClassificationProcessor(
            gate_notifier=self._gate_notifier,
            conversation_notifier=self._conversation_notifier,
            voicemail_notifier=self._voicemail_notifier,
            voicemail_response_delay=voicemail_response_delay,
        )
        self._voicemail_gate = TTSGate(self._conversation_notifier, self._voicemail_notifier)
```

평범한 `EventNotifier()` object 셋 (`utils/sync/event_notifier.py:14`), `__init__`에서 생성되어
processor 넷에게 건네집니다 — 그중 하나인 `TTSGate`는 `.gate()`를 통해 *바깥* pipeline의 완전히
다른 지점에 splice됩니다. 그것이 비인접성(non-adjacency)에 대한 Pipecat 자신의 답이고, message가
아니라 공유 object입니다.

그러면 배치는 셋:

| 필요 | 도구 | 이유 |
|---|---|---|
| layer가 live session state를 읽음 | **공유 평범한 object** — [[ch-09/read]]의 `LLMContext` 패턴 | 동기적 read; 하나의 object에 다수의 holder |
| "layer A가 인접하지 않은 layer B에게 무언가 일어났다고 알림" | **공유 `EventNotifier`** | `voicemail_detector.py:635-637`이 선례 |
| 모든 processor의 모든 frame에 대한 cross-cutting 관찰 | **Observer** ([[ch-11/read]]) | observer는 *pipe에 splice되지 않고서* 모든 것을 봅니다 — 정확히 `SignalQueue`가 필요로 하는 비인접성 property |
| classifier나 analytics worker를 **out of process**로 | **the bus** (`RedisBus` / `PgmqBus` + `BusJobRequestMessage`) | 진짜 process boundary; status를 가진 request/response |
| "component X가 준비될 때까지 기다림" | **`WorkerRegistry.watch(name, handler)`** | `registry/registry.py:80` — idempotent하고, 이미 등록되어 있으면 즉시 발화하므로(`:100-102`) startup race를 공짜로 닫아 줍니다; boson은 오늘 이것을 손으로 만들고 있습니다 |

마지막 행은 작은 공짜 이득입니다. 가져가십시오.

**bus에 손을 뻗는다면 경고 하나.** `BusBridgeProcessor` (`bus/bridge_processor.py:41`)는
**local frame을 소비합니다**: `:114-121`의 정상 경로는 `BusFrameMessage`를 만들고 `self._bus.send(msg)`
를 호출하며, `push_frame`을 결코 호출하지 않습니다. 그것은 pipeline 중간의 *terminator*입니다 —
그 downstream의 모든 것은 bus에서 돌아오는 것만 받습니다. `_LIFECYCLE_FRAMES` (`:37`),
`_PASSTHROUGH_FRAMES` (`:38`), 그리고 당신의 `exclude_frames`만 local로 통과합니다. 그것을 모르고
하나 splice해 넣으면 pipeline이 조용해집니다.

---

## 14. THREE OPEN RISKS, 각각 이름 붙은 prototype과 함께

이 중 어느 것도 읽어서 해결되지 않습니다. 각각이 실험이고, 각각이 실행 전에 쓸 수 있는 pass/fail
조건을 가지고 있습니다.

### Risk 1 — filler filter 대 energy 기반 barge-in

**gap.** boson은 `"네"`를 *내용*으로, 그리고 `pre_turn_status`로 filter합니다. Pipecat은 **VAD
energy**로 interrupt하는데, 그것은 STT의 upstream이고 내용을 보지 않습니다. 따라서 `BosonFillerGate`가
어떤 text든 보게 될 시점이면 봇은 이미 interrupt된 상태입니다. gate는 backchannel이 turn이 되는 것을
막을 수 있습니다. transcript가 존재하기 전에 broadcast된 interruption을 un-interrupt할 수는 없습니다.

**이것을 측정하십시오, 추정하지 말고.** interruption broadcast와 `TranscriptionFrame` 도착 사이를
timestamp-diff하십시오. **8 kHz telephony audio 위의 단독 한국어 backchannel** 코퍼스에 대해서.
gap은 양수일 것입니다 — 질문은 얼마나 크냐, 그리고 그 안에서 봇의 TTS가 들리는 출력을 냈느냐입니다.

**[[ch-06/read]]가 이미 판정한 것, 그래서 다시 하지 않아도 되는 것.** §19.3이 두 후보 strategy를
검토했습니다:

- `TranscriptionUserTurnStartStrategy`
  (`turns/user_start/transcription_user_turn_start_strategy.py:14`)는 존재하고 *지원되는* Pipecat
  config이지만, 방향이 틀렸습니다: turn-start를 **더** 성급하게 만듭니다 (`:38-40`에서
  `InterimTranscriptionFrame` → `trigger_user_turn_started()`). text trigger를 추가하는 것이지,
  VAD trigger를 보류하는 것이 아닙니다.
- `MinWordsUserTurnStartStrategy`는 보류하지만, 틀린 단위를 셉니다:

  **`src/pipecat/turns/user_start/min_words_user_turn_start_strategy.py:108-111`**

  ```python
          min_words = self._min_words if self._bot_speaking else 1

          word_count = len(frame.text.split())
          should_trigger = word_count >= min_words
  ```

  `"네"`는 **한 단어**이고 `min_words=1`을 통과합니다. boson의 `WordFilterPolicy(max_chars=3)`은
  **문자** 기준으로 그것을 잡습니다. `min_words=2`로 설정하면 `"잠깐만요"` — 고객의 진짜 발언권
  주장 — 도 함께 억제됩니다. `:108`도 주목하십시오: threshold는 **봇이 말하고 있는 동안에만**
  적용되고, 침묵에서는 1로 떨어집니다. 그 비대칭은 옳고, 당신이라면 직접 만들 것을 기억해 내야
  했을 것입니다.

**그래서 prototype은 [[ch-06/read]] §20 Probe 1이고, 이 chapter의 기여는 그것이 이제 흥미로운
것이 아니라 필수라는 점입니다.** 측정된 gap이 양수라면 transcript가 존재할 때까지 turn-start를
보류하고 한국어 backchannel test를 통과하는 custom `BaseUserTurnStartStrategy` subclass가 필요합니다.
그리고 그것은 측정되지 않은 한국어 STT TTFS를 비용으로 합니다 (§11.3: `stt_latency.py`에 항목 없음).
그것은 Risk 1의 측정이 산출하는 것과 같은 숫자입니다. 측정을 한 번 하면 두 질문에 답합니다.

### Risk 2 — transition frame race

**Prototype.** rule 구동 transition을 발화시키십시오. `llm` **input**에 `FrameLogger`
(`processors/logger.py:23`)를 두고 같은 trace에 대해 두 가지를 assert하십시오:

1. turn당 inference를 trigger하는 frame이 **정확히 하나** 도착한다.
2. `LLMSetToolsFrame`이 그것보다 **앞선다**.

**양쪽 failure 방향 모두**를 지켜보십시오. 그것들은 대칭이고 그중 하나는 놓치기 쉽습니다:

| 당신이 한 것 | `llm`의 frame | 증상 |
|---|---|---|
| context frame을 push **하고** `set_node_from_config`도 호출 | **둘** | 봇이 자기 자신에게 답함; TTFB 두 번; 비용 두 배 |
| 양쪽 path를 다 swallow | **0** | 봇이 통화 중간에 침묵하고 아무 error도 나지 않음 |
| context frame만 swallow (§8.4) | **하나** | 정답 |

0인 경우가 위험한 경우입니다 — 라이브 sales call에서 침묵하는 봇은 exception도, log line도 만들지
않고, 고객이 전화를 끊게 만듭니다.

**계측 도구 자체 안의 함정**, [[processor-vocabulary]]에서: `FrameLogger.__init__`은
`ignored_frame_types`를 받고 기본값은 네 개짜리 tuple이며, `:64`의 guard는
`if self._ignored_frame_types and not isinstance(...)`입니다. `ignored_frame_types=()`를 넘기면 전부
logging하는 대신 **모든** logging이 비활성화됩니다 — 빈 tuple은 falsy이기 때문입니다. 당신이 직접
침묵시킨 logger로 침묵하는 봇을 디버깅하지 마십시오.

figure의 PANEL SIX가 세 행 모두를 돌립니다.

### Risk 3 — two-phase-commit blast radius

**gap.** boson은 `session.messages`에 대해 object identity로 rollback합니다. `LLMContext`는
`set_messages(list)`만 제공하고 aggregator는 `:863`에서 이미 썼습니다. §9.3이 positional 대역
`snapshot[:-1]`과 그것이 layer가 붕괴했기 *때문에만* 건전한 이유를 보여주었습니다. "논증 안에서
건전하다"와 "실제 코퍼스 위에서 건전하다"는 다른 주장입니다.

**Prototype.** rule round 전체를 snapshot/restore로 감싸고 Lina e2e suite
(`agents/test-lina-gateway/tests/`, `e2e_runner.py`)를 replay하면서 boson의 현재 behaviour에 대한
divergence를 turn 단위로 세십시오.

**divergence가 있을 곳, 구체적으로:** `PreTool`이 합성된 tool-call history를 append한 **뒤에**
나중 layer가 filter한 turn들. 그것이 당신이 복원하는 list와 boson의 identity walk가 만들어냈을
list가 같은 list가 아닌 경우입니다. `PreTool`은 message를 하나 이상 쓰는데 `snapshot[:-1]`은 정확히
하나만 제거하기 때문입니다. 그 test를 먼저 쓰십시오.

**Pass 조건:** `PreTool`-then-`Filter` 부분집합에서 divergence 0, 또는 각각에 대해 문서화되고
의도된 차이. 규제된 동의 text를 말하는 시스템에게 "충분히 가깝다"는 pass 조건이 아닙니다.

---

## 15. Source corrections, 모아서

넷, 전부 `0cbf9c5b`에서 파일을 열어 검증했습니다. excerpt는 pre-read 요약이고, 틀린 것보다 맞은
것이 훨씬 많았지만, source와 어긋나는 곳에서는 **source가 이깁니다**.

| # | excerpt / spec의 주장 | 파일이 말하는 것 |
|---|---|---|
| 1 | Path B는 `warm_transfer.py:658`에서 증명됨 ([[design-boson-rules-on-pipecat]] §2) | `:261`이 `set_node_from_config` 호출이고, `:658`은 거기 도달하는 transport callback입니다. 서로 다른 두 줄이고 둘 다 실재합니다. §1 |
| 2 | rollback은 `snapshot = ctx.get_messages()` 후 `set_messages(snapshot)` ([[design-boson-rules-on-pipecat]] §3, 그리고 figure spec) | `get_messages()`는 **live list**를 반환하고 (`llm_context.py:244-245,260`) `set_messages`는 `self._messages[:] = messages`입니다 (`:383`). `list(...)` 없이는 rollback이 자기 자신에 대한 대입이고 조용히 아무것도 복원하지 않습니다. §9.3 |
| 3 | "Pipecat 자신의 STT TTFS P99 reference는 Deepgram에 대해 0.45 s" ([[design-boson-rules-on-pipecat]] §3) | `DEEPGRAM_TTFS_P99 = 0.35` (`stt_latency.py:45`). `0.45`는 `:27`의 docstring 예시에만 나타납니다. §11.3 |
| 4 | `start_processing_metrics()`/`stop_processing_metrics()`로 감싸라, 안 그러면 "bill이 `LatencyBreakdown`에 결코 나타나지 않는다" (chapter spec) | `UserBotLatencyObserver._handle_metrics_frame`은 `TTFBMetricsData`와 `TextAggregationMetricsData`만 읽고 (`:322-341`), `LatencyBreakdown`에는 processing field가 없습니다 (`:107-111`). 그리고 `FrameProcessor.can_generate_metrics()`는 `False`를 반환하므로 (`:494`), 평범한 custom processor에서는 *모든* metric 호출이 조용한 no-op입니다. §11.4 |

그리고 사실이 아니라 *권장 사항*에 대한 교정 하나: [[custom-processor-guide]]의
*"the honest port splits its four layers into four `FrameProcessor`s"*는 §6에 의해 knock out
됩니다. processor 넷이 무엇을 줄 것인지에 대한 그 사실들은 전부 옳습니다. 결론이 Constraint Two와
Three를 살아남지 못할 뿐입니다.

---

## 16. 이 chapter가 결정하지 않은 것

셋, 침묵을 동의로 오해하지 않도록 이름 붙입니다.

**migration을 아예 할 것인가.** [[ch-13/read]]가 그것을 소유합니다. 이 chapter는 "rule layer를
이식한다면 어떤 모양이어야 하고 비용이 얼마인가"에 답했습니다. "해야 하는가"에는 답하지 않았습니다.

**Tier 2가 영원히 critical path에 있어야 하는가.** §11.5는 trade와 boson의 현재 입장을 진술합니다.
250-400 ms가 Lina에게 그만한 가치가 있다고 말해 주지는 않습니다. 그것은 이제 당신이 가진 숫자
더하기 아직 가지고 있지 않은 한국어 STT TTFS 숫자로 정보를 얻는 product 결정입니다.

**collapse가 운영상 수용 가능한가.** §6.1이 잃어버린 per-layer queueing, observability, interruption
semantics로 값을 매겼습니다. 그것들은 실재하고, 당신 팀이 per-processor trace를 읽으며 디버깅한다면
논증이 시사하는 것보다 더 아플 수 있습니다. correctness 결론은 그와 무관하게 성립합니다. 운영상의
판정은 당신의 몫입니다.

---

## 다음 챕터로

이 chapter는 [[ch-13/read]]에게 세 가지를 넘깁니다. ch-13은 capstone이고 subsystem 단위로
keep-or-replace를 결정해야 합니다.

**rule subsystem에 대한 값이 매겨진 행 하나.** "이식 가능" 또는 "어렵다"가 아니라 —
**`FrameProcessor` 하나, 열세 개 rule 파일에 대한 편집 0, 매 turn +250-400 ms, 그리고 layer가
붕괴했기 때문에만 보존되는 cross-layer veto.** 그것은 transport 행과 STT 행 옆에 놓고 비교할 수
있는 행입니다.

**Pipecat에 정말로 집이 없는 것들의 짧은 목록**, ch-13이 다시 발견하지 않아도 되도록: transition
legality (`StageDefinition.transitions`), `skills`, per-session attribute namespace
(`SharedLayerContext`), `TOOL_PROCESSING` agent status, 그리고 `Inject`의 마지막 user message에
merge하는 option. 그중 모든 것은 boson code로 남거나 사라집니다.

**그리고 Pipecat이 돌려주는 것들의 짧은 목록**, ch-13이 장부의 반대편에 credit해야 할 것들:
stage를 위한 Flows node, `_GLOBAL_TOOLS`를 위한 `global_functions`, `PreTool`을 위한 always-wait
semantics를 가진 `function` pre-action, boson이 손으로 만들고 있는 startup race를 위한
`WorkerRegistry.watch`, 그리고 — §13에서 — message bus로 해결하려던 모든 "인접하지 않은 component가
조율해야 한다" 문제를 위한 `EventNotifier` 패턴.

이 chapter에서 들고 나갈 두 문장.

**seam은 취향의 문제가 아닙니다.** `:863`이 쓰고, `:866`이 push하고, `:601`이 테스트하고, `:603`이
되돌릴 수 없게 push하고, `:605`가 generate합니다. 그 sequence 안에서 완결된 turn이 존재하면서
아무것도 지출되지 않은 틈은 정확히 하나이고, 이 chapter의 모든 design 선택이 그 틈이 어디인가로부터
떨어져 나옵니다.

**그리고 collapse는 veto의 값입니다.** Pipecat의 유일한 write verb는 호출 시점에 효력이 발생합니다.
따라서 processor를 가로지르는 transaction은 rollback할 수 없고, 그러므로 cross-layer veto가 필요한
시스템은 하나의 processor여야 합니다 — 그것은 Pipecat이 결함이 있다는 뜻이 아니라, transaction이
어디에 사는지에 대한 Pipecat의 베팅이 당신에게 청구서를 내미는 것입니다.
