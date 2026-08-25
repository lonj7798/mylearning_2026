---
title: "Pipecat Flows: pipeline 바깥에 사는 state machine"
chapter: ch-10
phase: collision
course: pipecat
lang: ko
companion_of: read.md
sources:
  - flows-state-machine
  - flows-node-types
  - flows-actions
  - flows-insurance-example
  - boson-stage-machine
  - theory-narrow-waist
deps:
  - ch-02
  - ch-09
figure: figures/flow-node-transition.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# ch-10 — Pipecat Flows: pipeline 바깥에 사는 state machine

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 (frame, processor, node, transition, action, aggregator, latency,
> observer, falsifier, pipeline, worker, batch, handler 등).

> **Scope — 미리 명시하고 chapter 전체에 걸쳐 강제합니다.**
>
> **하나의 아이디어: the inversion(뒤집힘).** `FlowManager`는 `FrameProcessor`가 **아닙니다**.
> `Pipeline` list 안에 있지도 않습니다. 그것은 pipeline의 head에 frame을 밀어 넣음으로써 pipeline을
> **바깥에서** 구동합니다. 이 chapter에 나오는 다른 모든 사실 — frame batch, string node ID, 없는
> validation, 세 개의 action verb, insurance graph — 은 전부 그 하나의 구조적 선택의 **귀결**이고,
> 각 section은 자기가 어떤 귀결인지를 명시합니다.
>
> **rule-layer design은 이 chapter에 없습니다.** 여기서 배우게 되는 것은 Flows가 node transition을
> tool call뿐 아니라 평범한 코드로부터도 받아들인다는 사실입니다. 그것은 *mechanism*입니다. 그것을
> boson의 `RuleEngine → StageTransition → set_node_from_config` seam으로 — processor가 어디에 서는지,
> milliseconds로 얼마를 지불하는지, layer가 붕괴하는지 — 바꾸는 일은 [[ch-12/read]]의 주제 전부이고
> [[ch-11/read]]가 분모를 공급합니다. 스스로 `BosonRuleProcessor`를 스케치하고 있다면, 당신은 이
> chapter를 벗어난 것입니다.
>
> **비교 판정 없음.** 이 chapter는 Flows가 **무엇을 하는지**, **무엇을 비용으로 지불하는지**,
> **무엇을 가지고 있지 않은지**를 말합니다. 채택할지 말지, 이미 ship한 것을 이기는지, boson의 stage
> machine을 교체해야 하는지는 말하지 않습니다. [[ch-13/read]]가 이 course에서 무언가를 채점하는
> 유일한 곳이고, 그것은 열두 개 subsystem을 전부 본 뒤에야 그 자격을 얻습니다.

---

## 왜 이 챕터인가

[[ch-02/read]] §11은 `flows/`를 frame budget으로 사용하고 나서 의도적으로 열기를 거부했습니다:

> *(How `flows/` actually works — `set_node`, the transition mechanics, the pre/post-action
> ordering, and the finding that transitions do not have to be LLM function calls — is
> [[ch-10/read]]'s subject. This section takes exactly one thing from it: the frame budget.)*
>
> (`flows/`가 실제로 어떻게 동작하는지 — `set_node`, transition mechanics, pre/post-action ordering,
> 그리고 transition이 LLM function call일 필요가 없다는 발견 — 은 [[ch-10/read]]의 주제다. 이
> section은 거기서 정확히 하나만 가져온다: frame budget.)

[[ch-09/read]]는 이 chapter를 올바르게 읽기 위해 무엇을 믿어야 하는지를 이름 붙이며 끝났습니다:

> a state machine that mutates a context it does not own is either elegant or terrifying depending
> entirely on whether you know who else is holding a reference.
>
> (자기가 소유하지 않은 context를 mutate하는 state machine은, 다른 누가 그 reference를 쥐고 있는지
> 당신이 아느냐에 전적으로 달려서 우아하거나 끔찍하거나 둘 중 하나다.)

이 chapter가 바로 그 chapter이고, "다른 누가 reference를 쥐고 있는가"라는 질문에 대한 답은 각주가
아니라 **design 전체**로 밝혀집니다.

주목할 가치가 있는 것은 이것입니다. Pipecat은 조직 원리가 **모든 것은 list 안의 processor다**인
framework입니다 — [[ch-01/read]]가 splice algebra를 증명했고, [[ch-04/read]]가 runtime을 증명했고,
[[ch-02/read]]가 waist를 증명했습니다. 만약 당신이 "Pipecat이 conversation state machine을 ship한다"는
말을 듣고 그 모양을 추측하라고 요구받는다면, 당신은 `Pipeline` list 안 user aggregator와 LLM 사이
어딘가에 넣는, frame을 intercept하고, current-state field를 들고, 무엇을 forward할지 결정하는
`FrameProcessor` subclass를 추측할 것입니다.

그 추측은 틀렸고, 그 틀린 방식이 사실 자체보다 값어치가 큽니다. `FlowManager`는 **평범한 Python
class**입니다. list 안에 없습니다. `process_frame`이 없습니다. frame이 자기 옆을 지나가는 것을 결코
보지 않습니다. 그것은 pipeline 옆에 앉아 `PipelineWorker`에 대한 reference를 쥐고, **pipe의 head에
frame을 queue함으로써** 대화를 구동합니다 — transport event handler가 Lina를 먼저 말하게 만들 때
사용하는 것과 같은 public API입니다([[ch-04/read]] §11).

당신에게 특히 이것이 두 번 중요합니다.

**첫째, boson의 stage machine이 같은 모양이고, 당신은 그 모양을 의도적으로 고른 것이 아닙니다.**
`StageMachine`(`gateway/stage/machine.py`)은 stateless하고 공유되며, `session.active_stage`는 session
위의 string이고, transition은 agent loop가 돌기 전에 rule 코드가 수행하는 bookkeeping입니다
([[boson-stage-machine]]). 그중 무엇도 pipeline node가 아닙니다. Pipecat 자신의 팀이, frame 기반
framework 안에 conversation state machine을 만들라는 과제를 받고서 같은 architecture에 도달했습니다 —
state는 평범한 manager object에, 효과는 기존 frame 위에. 그 수렴은 evidence이고, 이 chapter는 그것이
정확히 *무엇에 대한* evidence이며 무엇을 비용으로 치르는지 말해 줍니다.

**둘째, in-tree 예제가 보험 견적 봇입니다.** `examples/flows/insurance_quote.py`는 나이, 결혼 여부,
보험료, 보장 조정을 음성으로 수집하는 다섯 개의 node factory, 380줄입니다. 당신은 한국어로 전화로
보험을 팝니다. §11이 그 파일을 한 줄씩 읽고 이전 가능한 decomposition pattern을 추출하며, §13이
그것을 당신이 이미 가지고 있는 아홉 개의 Lina stage 위에 매핑합니다.

**이 chapter의 코드를 읽는 법.** 아래의 모든 경로, 줄 번호, class 이름, 개수는 집필 중에 commit
`0cbf9c5b031eef06e53f0a193b9a67d60230e6be` 시점의 `wiki/raw-data/pipecat/pipecat-src`에 대해 다시
읽어졌습니다. curated excerpt가 source와 어긋나는 곳에서는 source가 이기고, 그 불일치는 조용히
고쳐지는 대신 inline으로 명시됩니다. §0이 그것들을 미리 나열하는데, 그중 넷이 이 package를 처음 읽을 때
자연스럽게 틀리게 되는 것들에 대한 정정이기 때문입니다.

---

## 0. 무언가를 그 위에 쌓기 전에 먼저 진술하는 여섯 개의 정정

conversation state machine은 다른 framework에서 가져온 기대를 안고 도착해서 그 기대를 통해 코드를 읽게
되는, 딱 그런 종류의 subsystem입니다. `pipecat.flows`에서 그런 일이 벌어지는 여섯 지점이 여기 있고,
각각은 파일을 열어서 확인한 것입니다.

### 0.1 이 commit에 `FlowConfig`는 존재하지 않는다

flow framework에 대한 모든 튜토리얼-모양 mental model에는 document가 있습니다: node, edge,
`initial_node` key를 가진 JSON 또는 YAML object. Pipecat Flows에는 그런 것이 없습니다.

```
$ cd wiki/raw-data/pipecat/pipecat-src
$ grep -rn "FlowConfig" . | wc -l
       0
$ grep -rni "static flow\|static_flow" . | wc -l
       0
$ grep -rni "dynamic flow\|dynamic_flow" . | wc -l
       0
$ grep -rn '"initial_node"' . | wc -l
       0
```

repository 전체 — source, tests, examples, docs — 에 대한 네 개의 grep이고, 네 개 모두 0을
반환합니다. `FlowConfig`도, 선언적 flow document도, edge table도, `initial_node` key도 없고,
**static 대 dynamic 구분도 없습니다**. Pipecat Flows에 관해 "static flows"(`FlowConfig` dict)와
"dynamic flows"(Python 함수)를 대비시키는 오래된 자료를 읽은 적이 있다면, 그 어휘는 이 tree에서
사라졌습니다. runtime-determined 스타일만 살아남았고, package docstring은 그것을 mode가 아니라
design으로 진술합니다:

**`src/pipecat/flows/__init__.py:6-14`**
```python
"""Pipecat Flows - Structured conversation framework for Pipecat.

This package provides a framework for building structured conversations in Pipecat.
The FlowManager handles conversation flows with support for state management,
function calling, and cross-provider compatibility.

Pipecat Flows determines conversation structure at runtime, supporting function
calling, action execution, and seamless transitions between conversation states.
"""
```

*"determines conversation structure at runtime."* (대화 구조를 runtime에 결정한다.) node는
`NodeConfig`를 반환하는 Python 함수입니다. edge는 handler가 반환한 tuple의 두 번째 원소입니다. 그것이
graph representation 전부이고, §6.4가 그것에 type을 부여합니다.

### 0.2 built-in action은 정확히 **세 개**다

verb library가 아닙니다. 연속된 세 줄에 등록된 셋이고, §9가 그것을 인용합니다.

### 0.3 `_validate_node_config`는 transition legality 검사를 전혀 하지 않는다

두 가지를 검사하고 그중 어느 것도 from→to edge가 아닙니다. §10이 method 전체를 인용하므로 검사 개수를
직접 세어 볼 수 있습니다.

### 0.4 `self._current_functions`는 `src/`에서 죽어 있다 — 하지만 "어디에도 없다"는 아니다

솔깃한 문장은 "그것은 대입되고 codebase 어디에서도 읽히지 않는다"입니다. 그 문장은 틀렸고, 정정된
버전이 더 흥미롭습니다:

```
$ grep -rn "_current_functions" src/ tests/ examples/
src/pipecat/flows/manager.py:148:        self._current_functions: set[str] = set()  # Track registered functions
src/pipecat/flows/manager.py:704:            self._current_functions = new_functions
tests/test_flows_manager.py:191:        self.assertEqual(flow_manager._current_functions, set())
tests/test_flows_manager.py:319:        self.assertEqual(len(flow_manager._current_functions), 3)
tests/test_flows_manager.py:758:        self.assertIn("test1", flow_manager._current_functions)
tests/test_flows_manager.py:759:        self.assertIn("test2", flow_manager._current_functions)
tests/test_flows_manager.py:1174:        self.assertEqual(flow_manager._current_functions, set())
tests/test_flows_manager.py:1203:        self.assertEqual(flow_manager._current_functions, set())
```

`src/`에 두 site — 선언 하나, 대입 하나. **`src/`에서 read는 0회.** `tests/test_flows_manager.py`에서
read 6회. 그래서 정확한 주장은 이것입니다: *`_current_functions`는 runtime gate로서는 죽어 있고,
test-assertion surface로서는 살아 있다.* 실행 중인 시스템의 무엇도 function을 dispatch하기 전에 그것을
참조하지 않습니다. test suite가 `_set_node`가 추적하기로 되어 있던 것을 추적했는지 확인하기 위해
참조합니다. §5.4가 이 구분이 그 위에 무엇을 지을 수 있는지를 어떻게 바꾸는지 설명합니다.

### 0.5 `FlowResult`는 **"No replacement."**와 함께 deprecate되었다

**`src/pipecat/flows/types.py:40-53`**
```python
@deprecated("`FlowResult` is deprecated since 1.5.0 and will be removed in 2.0.0. No replacement.")
class FlowResult(TypedDict, total=False):
    """Optional convention TypedDict for ``status``/``error`` results.

    .. deprecated:: 1.5.0
        No replacement. ``FlowResult`` is no longer required or referenced by
        any handler type, and Pipecat's upstream function-call-result contract
        is ``Any`` — define your own ``TypedDict`` or return any
        JSON-serializable value. Will be removed in 2.0.0.

    Parameters:
        status: Status of the function execution.
        error: Optional error message if execution failed.
    """
```

deprecation body를 조심해서 읽으십시오. 이례적이기 때문입니다. 대부분의 Pipecat deprecation은 첫
문장에서 대체재를 이름 붙입니다 — repo 자신의 `AGENTS.md`가 그것을 요구합니다. 이것은 **"No
replacement."**라고 말하고 나서 이유를 설명합니다: function-call result에 대한 upstream contract는
평범한 `Any`다. `FlowResult`는 convention이었고, 그 convention은 폐기되었으며, 아무것도 그 자리를
차지하지 않았습니다. 그에 따라 `insurance_quote.py`는 아무것도 상속하지 않는 자신만의 맨
`TypedDict`들(`AgeCollectionResult`, `QuoteCalculationResult`)을 선언합니다.

### 0.6 `NodeConfig`의 key 집합에는 `role_messages`가 있고, `task_messages`가 유일한 `Required`다

복수형과 단수형이 둘 다 존재합니다. `role_message`(단수, `str`)가 현행이고, `role_messages`(복수,
`list[dict]`)는 1.5.0부터 deprecated이며 pipeline을 통과하는 완전히 다른 경로를 탑니다 — §4.3이
둘이 단지 shape만 다른 게 아니라 **서로 다른 frame을 emit한다**는 것을 보여 줍니다. 그리고 아홉 개의
key 중 정확히 하나가 `Required[...]`입니다. §6.1에 표가 있습니다.

---

## 1. 하나의 구조적 사실

### 1.1 class 선언

**`src/pipecat/flows/manager.py:80-89`**
```python
class FlowManager:
    """Manages conversation flows.

    The FlowManager orchestrates conversation flows by managing state transitions,
    function registration, and message handling across different LLM providers,
    with comprehensive action handling and error management.

    The manager coordinates all aspects of a conversation including LLM context
    management, function registration, state transitions, and action execution.
    """
```

`class FlowManager:` — 맨몸입니다. base class 없음. `FrameProcessor`도, `BasePipeline`도, `BaseObject`도
아닙니다. event system도, task manager도, [[ch-04/read]] §4의 two-queue runtime도, `push_frame`도
상속하지 않습니다. `process_frame` method가 없고, 이는 [[ch-01/read]]이 splice algebra 전체를 그 위에
지은 uniform interface — `processors/frame_processor.py:820`의 `async def process_frame(self, frame:
Frame, direction: FrameDirection)` — 가 여기에는 그냥 부재한다는 뜻입니다.

**`FlowManager`는 무엇에도 `link()`할 수 없습니다.** 그것은 algebra의 원소가 아닙니다.

### 1.2 그것은 `Pipeline` list 안에 없다 — `hello_world.py`를 읽으십시오

192줄짜리 `hello_world.py`는 tree에서 가장 작은 완전한 Flows 프로그램이고, 그것에 대해 중요한 것은
pipeline list가 *담고 있지 않은* 것입니다.

**`examples/flows/hello_world.py:135-167`**
```python
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )
```

pipeline 항목을 세어 보십시오: **일곱 개**, 그리고 그 전부가 [[ch-04/read]] §11의 canonical voice
bot에서 온 것입니다. `transport.input()`, `stt`, `context_aggregator.user()`, `llm`, `tts`,
`transport.output()`, `context_aggregator.assistant()`. 이것은 state machine이 전혀 없는 봇을 위해
당신이 쓸 바로 그 list입니다. Pipecat 봇에 conversation state machine을 추가하는 것은 `Pipeline`
constructor의 항목을 **0개** 바꿉니다.

이제 construction 순서를 읽으십시오. 그것이 load-bearing이기 때문입니다:

1. `Pipeline([...])`가 지어집니다 — flow는 아직 존재하지 않습니다.
2. `PipelineWorker(pipeline, ...)`가 그것을 감쌉니다.
3. `WorkerRunner(...)`가 구성됩니다.
4. `await runner.add_workers(worker)`가 그것을 등록합니다.
5. **그러고 나서야** `FlowManager(worker=worker, llm=llm, context_aggregator=context_aggregator,
   transport=transport)`.

`FlowManager`는 pipeline이 완전히 조립되고 등록된 *다음에* 구성됩니다. 그것이 조립된 pipeline의
**부품**이 아니라 **소비자**이기 때문입니다. 그것은 이미 지어진 `worker`를 constructor 인자로 받습니다.
그것이 splice해 들어갈 수 있는 어떤 순서도 존재하지 않습니다. splice할 대상 자체가 없기 때문입니다.

그리고 kickoff:

**`examples/flows/hello_world.py:169-180`**
```python
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()
```

대화는 **transport event handler**에서 시작되지, 어딘가에 frame이 도착해서 시작되지 않습니다. 이것은
[[ch-04/read]] §11이 voice agent가 먼저 말하는 이유로 지목한 바로 그 모양입니다: pipeline은
기계장치이고, 바깥의 무언가가 대화를 시작해야 합니다. `flow_manager.initialize(create_initial_node())`가
그 무언가입니다.

`insurance_quote.py:350-361`은 initial node만 다른, byte 단위로 동일한 pattern입니다.
`restaurant_reservation.py`도 그렇습니다. `patient_intake.py`도 그렇습니다. 이 모양은 hello world의
우연이 아닙니다.

### 1.3 list 안의 position 대신 그것이 쥐고 있는 것

position이 없으므로 reference를 쥐어야 합니다. state 전체가 여기 있습니다, verbatim:

**`src/pipecat/flows/manager.py:134-154`**
```python
        self._worker = worker
        self._llm = llm
        self._action_manager = ActionManager(worker, flow_manager=self)
        self._adapter = LLMAdapter()
        self._initialized = False
        self._context_aggregator = context_aggregator
        self._pending_transition: dict[str, Any] | None = None
        self._context_strategy = context_strategy or ContextStrategyConfig(
            strategy=ContextStrategy.APPEND
        )
        self._transport = transport
        self._global_functions = global_functions or []

        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None

        self._showed_deprecation_warning_for_role_messages = False
        self._showed_deprecation_warning_for_reset_with_summary = False
        self._showed_deprecation_warning_for_zero_arg_handler = False
        self._showed_deprecation_warning_for_legacy_handler = False
```

바깥으로 향하는 reference 셋(`_worker`, `_llm`, `_context_aggregator`, 그리고 optional인 `_transport`),
소유하는 sub-object 하나(`_action_manager`), 그리고 실제 conversation state field 셋: `_state`는 맨
`dict`, `_current_functions`는 `src/`의 무엇도 읽지 않는 `set[str]`, `_current_node`는 `str | None`.

네 개의 `_showed_deprecation_warning_for_*` boolean도 주목하십시오. deprecated code path 하나당 flag
하나이고, 각각은 warning이 node transition마다가 아니라 `FlowManager`마다 한 번 발생하도록 존재합니다.
898줄짜리 module에 그것이 넷이라는 것은 이 API 중 얼마만큼이 나가는 중인지에 대한 측정치이고, §6.7이
제거 예정 목록을 나열합니다.

constructor는 **keyword-only**이고(`def __init__(self, *, llm, context_aggregator, worker=None,
task=None, ...)`, `manager.py:91-101`), `task`는 `worker`의 deprecated된 1.5.0 표기입니다. 둘 다
넘기면 raise합니다:

**`src/pipecat/flows/manager.py:121-132`**
```python
        if worker is not None and task is not None:
            raise ValueError("Pass either 'worker' or 'task' (deprecated), not both.")
        if task is not None:
            warnings.warn(
                "The 'task' parameter is deprecated since 1.5.0 and will be removed "
                "in 2.0.0. Use 'worker' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            worker = task
        if worker is None:
            raise ValueError("FlowManager requires a 'worker' (PipelineWorker).")
```

마지막 세 줄은 이 section의 요점을 runtime check로 다시 진술한 것입니다: **`PipelineWorker` 없는
`FlowManager`는 존재할 수 없다.** `FrameProcessor`에는 그런 의존성이 없습니다 — 하나 만들어서, 들고
있다가, 나중에 link할 수 있습니다. `FlowManager`는 자기가 구동할 이미 조립된 것을 construction 시점에
요구하거나, 아니면 지어지기를 거부합니다.

> 💡 **쉬운 설명 — "inversion"이 정확히 무엇을 뒤집나요?**
> 보통 framework의 확장 지점은 "당신의 코드를 framework 안에 넣는다"입니다. `FrameProcessor`를 만들어
> list에 넣으면 framework가 그것을 호출합니다 (inversion of control). 그런데 `FlowManager`는 반대입니다 —
> **framework의 조립된 결과물(worker)을 인자로 받아서, 자기가 그것을 호출합니다.** 그래서 생성 순서가
> 강제됩니다(pipeline이 먼저 완성되어야 함). "state machine을 pipeline에 꽂는다"가 아니라
> "state machine이 pipeline을 리모컨으로 조종한다"에 가깝습니다.

### 1.4 이것은 [[ch-02/read]]의 규칙이 최대 규모로 적용된 것이다

[[ch-02/read]] §12는 boson 자신의 state에 대한 3-way test를 진술했습니다: **(a)** state → frame이
아님, **(b)** effect → 기존 frame, **(c)** 진짜로 새로운 in-band signal → subsystem-local `frames.py`,
budget 2~4개.

`flows/`는 Pipecat 자신의 팀이 가장 어려운 경우에 자기 규칙을 지키는 모습이고, 이제 frame 개수만이
아니라 준수의 전체 모양을 볼 수 있습니다:

| Test | `flows/`가 한 것 | Evidence |
|---|---|---|
| (a) state → not a frame | `_current_node: str`, `_state: dict`, `NodeConfig` as `TypedDict` | `manager.py:147-149`; `types.py:182` |
| (b) effects → existing frames | `LLMUpdateSettingsFrame`, `LLMMessagesAppendFrame`/`UpdateFrame`, `LLMSetToolsFrame`, `LLMRunFrame`, `TTSSpeakFrame`, `EndFrame` | `manager.py:768, 838, 839, 709`; `actions.py:323, 359` |
| (c) new in-band signals | 정확히 둘, 둘 다 `ControlFrame`, 둘 다 `flows/actions.py` 안 | `actions.py:49-66` |
| **(d) — ch-02가 볼 수 없었던 부분** | **manager 자체가 pipeline 안에 아예 없다** | `manager.py:80`; `hello_world.py:135-145` |

(d) 행이 이 chapter가 추가하는 것입니다. [[theory-narrow-waist]] §4가 두 개의 frame을 세고 3-way
test를 도출했고, [[ch-02/read]]가 frame budget을 읽고 `flows/`가 자기 private vocabulary를 private하게
유지했다고 올바르게 결론지었습니다. 그것이 보여 줄 수 없었던 것은 — `frames.py`를 읽고 있었기
때문에 — *component* 역시 private하게 남았다는 사실입니다. frame budget이 2인 것은 **component
budget이 0**이기 때문입니다.

---

## 2. 두 개의 touch point, 그리고 오직 둘뿐

pipeline 바깥의 무언가가 pipeline을 구동하려면 질문은 정확히 둘입니다: effect는 어떻게 들어가고,
observation은 어떻게 나오는가. Flows는 각각에 한 곳에서 답합니다.

### 2.1 Out: `queue_frames`, 두 번

```
$ grep -n "queue_frames\|queue_frame(" src/pipecat/flows/manager.py src/pipecat/flows/actions.py
src/pipecat/flows/actions.py:322:            await self._worker.queue_frame(
src/pipecat/flows/actions.py:329:            await self._worker.queue_frame(ActionFinishedFrame())
src/pipecat/flows/actions.py:353:            await self._worker.queue_frame(
src/pipecat/flows/actions.py:359:        await self._worker.queue_frame(EndFrame())
src/pipecat/flows/actions.py:385:            await self._worker.queue_frame(FunctionActionFrame(action=action, function=handler))
src/pipecat/flows/manager.py:269:                    await flow_manager.worker.queue_frame(
src/pipecat/flows/manager.py:709:                await self._worker.queue_frames([LLMRunFrame()])
src/pipecat/flows/manager.py:841:            await self._worker.queue_frames(frames)
```

`manager.py:269`는 docstring 예제 안이지 코드가 아닙니다. 그래서 manager의 진짜 write surface는
**두 줄**입니다: `:709`가 inference trigger를 queue하고, `:841`이 settings-and-context batch를
queue합니다. 나머지 전부는 `ActionManager`이고, 그것의 write 다섯 개는 전부 세 개의 built-in action
handler 안에 있습니다.

그게 전부입니다. 898 + 400 + 518 + 68 + 62 = 1,946줄짜리 subsystem이 실시간 voice pipeline을
**일곱 개의 `queue_frame*` call site**로 구동합니다.

### 2.2 In: 한 개의 event, 세 개의 frame type으로 필터링

되읽는 쪽이 더 어렵습니다. pipeline 바깥의 무언가는 무엇에 인접해 있음으로써 frame을 관찰할 수 없기
때문입니다. Flows는 worker의 downstream-arrival event를 사용합니다:

**`src/pipecat/flows/actions.py:103-127`**
```python
        # Register built-in actions
        self._register_action("tts_say", self._handle_tts_action)
        self._register_action("end_conversation", self._handle_end_action)
        self._register_action("function", self._handle_function_action)

        # Add pipeline observation
        worker.set_reached_downstream_filter(
            (ActionFinishedFrame, FunctionActionFrame, BotStoppedSpeakingFrame)
        )

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame_reached_downstream(worker, frame):
            if isinstance(frame, FunctionActionFrame):
                # Run function action
                await frame.function(frame.action, flow_manager)
                self._decrement_ongoing_actions_count()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                # Execute deferred post-actions if the bot's turn is over.
                # A BotStoppedSpeakingFrame only indicates that the bot's turn is over if there are
                # no ongoing actions (otherwise one of those actions may have been responsible for it).
                if self._ongoing_actions_count == 0:
                    await self._execute_deferred_post_actions()
            elif isinstance(frame, ActionFinishedFrame):
                # Handle action finished
                self._decrement_ongoing_actions_count()
```

세 개의 frame type, 세 개의 behaviour:

- `FunctionActionFrame` — **`function` action의 handler가 실제로 실행되는 곳이 여기입니다.** action이
  스케줄된 시점이 아닙니다. handler는 frame 안에 field로 실려 다니고
  (`function: FlowActionHandler`, `actions.py:59`), frame이 pipeline의 tail에 도달할 때 호출됩니다.
  §9.2가 왜 그 배치가 `function` verb의 요점 전부인지 설명합니다.
- `BotStoppedSpeakingFrame` — deferred post-action의 trigger이고, 주석이 말하는 이유로 ongoing-action
  count에 대해 guard됩니다.
- `ActionFinishedFrame` — `tts_say`의 완료 signal이고, §9.6의 wait table이 그 위에서 block하는 counter를
  감소시킵니다.

worker 쪽의 mechanism:

**`src/pipecat/pipeline/worker.py:1320-1329`**
```python
    async def _sink_push_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames coming downstream from the pipeline.

        This tasks process frames coming downstream from the pipeline. For
        example, heartbeat frames or an EndFrame which would indicate all
        processors have handled the EndFrame and therefore we can exit the worker
        cleanly.
        """
        if isinstance(frame, tuple(self._reached_downstream_types)):
            await self._call_event_handler("on_frame_reached_downstream", frame)
```

worker의 내부 sink — [[ch-04/read]] §6.3이 당신의 pipeline을 감싼다고 보여 준 그 processor — 가
등록된 집합에 type이 들어 있는 모든 frame에 대해 event를 발화합니다. Flows는 무엇에도 인접하지
않습니다. 그것은 **먼 쪽 끝에 구독**되어 있습니다.

### 2.3 정직한 발견: `set_reached_downstream_filter`는 REPLACE하고, Flows가 유일한 caller다

worker가 노출하는 두 method를 보십시오:

**`src/pipecat/pipeline/worker.py:695-717`**
```python
    def set_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
        """Set which frame types trigger the on_frame_reached_downstream event.

        Args:
            types: Tuple of frame types to monitor for downstream events.
        """
        self._reached_downstream_types = set(types)

    def add_reached_upstream_filter(self, types: tuple[type[Frame], ...]):
        """Add frame types to trigger the on_frame_reached_upstream event.

        Args:
            types: Tuple of frame types to add to upstream monitoring.
        """
        self._reached_upstream_types.update(types)

    def add_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
        """Add frame types to trigger the on_frame_reached_downstream event.

        Args:
            types: Tuple of frame types to add to downstream monitoring.
        """
        self._reached_downstream_types.update(types)
```

`set_...`은 `= set(types)`를 합니다. `add_...`는 `.update(types)`를 합니다. 기본값은 비어
있으므로(`worker.py:563`: `self._reached_downstream_types: set[type[Frame]] = set()`), 새 worker에서는
둘이 동등합니다 — 아마도 그것이 `ActionManager`가 `set_`을 쓰는 이유일 것입니다.

그러나:

```
$ grep -rn "set_reached_downstream_filter\|add_reached_downstream_filter" src/ examples/
src/pipecat/pipeline/worker.py:695:    def set_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
src/pipecat/pipeline/worker.py:711:    def add_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
src/pipecat/flows/actions.py:109:        worker.set_reached_downstream_filter(
```

`ActionManager.__init__`이 tree 전체에서 두 method 중 어느 것이든 호출하는 **유일한** caller이고,
`add_reached_downstream_filter`는 caller가 0입니다. 그래서 이것은 repo의 무엇도 현재 행사하지 않는,
잠복해 있는 순서 의존적 conflict입니다:

- 당신의 application이 `worker.set_reached_downstream_filter((MyFrame,))`을 호출하고 **그 다음**
  `FlowManager`를 구성하면, 당신의 filter는 조용히 지워지고 handler는 발화를 멈춥니다.
- `FlowManager`를 구성하고 **그 다음** `set_reached_downstream_filter(...)`를 호출하면, Flows의
  filter를 지우게 됩니다 — 그리고 그 failure mode는 exception이 아닙니다. `function` action이 실행을
  멈춥니다. handler는 오직 `on_frame_reached_downstream`에서만 실행되기 때문입니다. deferred
  post-action은 결코 발화하지 않습니다. `tts_say`는 자기 counter를 감소시키지 않으므로, §9.6의 wait
  table은 대기가 필요한 다음 batch에서 영원히 block합니다.

해결책은 한 단어입니다: 당신 쪽에서는 항상 `add_reached_downstream_filter`를 쓰고, 다른 무엇이 filter를
등록하기 전에 `FlowManager`를 구성하십시오. 지금 적어 두십시오. 증상("post-action이 가끔 안 돌아요")은
원인 근처를 전혀 가리키지 않기 때문입니다.

이것은 어떤 excerpt에도 없습니다. §2.2를 확인하면서 worker를 grep하다가 나왔습니다. 이것은 어떤
component가 자기가 속하지 않은 pipeline을 구동할 때 나타나는 딱 그 부류의 coupling입니다:
component는 worker 위의 공유된 단일 슬롯 configuration에 손을 뻗어야 하고, 단일 슬롯 configuration은
compose되지 않습니다.

> 💡 **쉬운 설명 — 왜 이게 "바깥에 있음"의 필연적 결과인가요?**
> pipeline 안의 processor라면 자기 자리로 흘러오는 frame을 그냥 보면 됩니다 — 아무 전역 설정도 건드릴
> 필요가 없습니다. 바깥에 있으면 볼 자리가 없으니 "worker야, 이 세 type이 끝에 도달하면 나한테
> 알려줘"라고 등록해야 합니다. 그런데 그 등록 슬롯이 하나뿐이라 두 번째 사용자가 첫 번째를 덮어씁니다.
> 즉 **관찰 능력을 얻기 위해 전역 상태를 점유해야 하는 것**이 non-adjacency의 청구서입니다.

---

## 3. "바깥"의 비용: frame은 HEAD로 들어간다

이 section이 inversion을 architectural한 것이 아니라 구체적인 것으로 만드는 곳이고,
[[ch-12/read]]가 자기 핵심 hazard를 그 위에 세울 곳입니다.

### 3.1 direction split, 세 줄로 회상

[[ch-04/read]] §10이 mechanism을 확립했습니다. 여기서 필요한 한 문장은 이것입니다. 기본값
`FrameDirection.DOWNSTREAM`으로 호출된 `queue_frame`은 frame을 `self._push_queue`에 올리고, 그것이
`self._pipeline.queue_frame(...)`을 먹입니다 — 그래서 **frame은 head로 들어가 모든 processor를 순서대로
통과합니다.** upstream frame은 대신 tail로 들어갑니다. 그리고 `queue_frames`는 평범한 loop입니다:

**`src/pipecat/pipeline/worker.py:810-829`**
```python
    async def queue_frames(
        self,
        frames: Iterable[Frame] | AsyncIterable[Frame],
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        """Queue multiple frames to be pushed through the pipeline.

        Downstream frames are pushed from the beginning of the pipeline.
        Upstream frames are pushed from the end of the pipeline.

        Args:
            frames: An iterable or async iterable of frames to be processed.
            direction: The direction to push the frames. Defaults to downstream.
        """
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame, direction)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame, direction)
```

batching 없음. atomicity 없음. **`FlowManager`가 내는 "frame batch"는 `for` loop가 그것을 썼다는
의미에서만 batch입니다.** 다른 곳에서 온 frame이 사이에 끼어들 수 있습니다.

Flows는 direction을 결코 넘기지 않습니다. 그것의 일곱 개 write 전부가 downstream, head입니다.

### 3.2 `LLMRunFrame` 하나를 추적해서 inference가 실제로 어디서 시작되는지 알아내기

정확하게 답할 가치가 있는 질문이 여기 있습니다. 뻔한 답이 틀리기 때문입니다. `_set_node`는
`manager.py:709`에서 `await self._worker.queue_frames([LLMRunFrame()])`을 합니다. §1.2의 일곱-processor
pipeline에서 **어느 processor가 그것을 inference call로 바꿉니까?**

뻔한 답은 `llm`입니다. 아닙니다.

**`src/pipecat/processors/aggregators/llm_response_universal.py:814-821`** — list의 세 번째 processor인
`LLMUserAggregator.process_frame` 안:
```python
        elif isinstance(frame, LLMRunFrame):
            await self._handle_llm_run(frame)
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_llm_messages_append(frame)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            await self._handle_llm_messages_update(frame)
        elif isinstance(frame, LLMMessagesTransformFrame):
            await self._handle_llm_messages_transform(frame)
```

**`src/pipecat/processors/aggregators/llm_response_universal.py:1173-1184`**
```python
    async def _handle_llm_run(self, frame: LLMRunFrame):
        await self.push_context_frame()

    async def _handle_llm_messages_append(self, frame: LLMMessagesAppendFrame):
        self.add_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame()

    async def _handle_llm_messages_update(self, frame: LLMMessagesUpdateFrame):
        self.set_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame()
```

그리고 같은 `elif` chain의 꼬리:

**`src/pipecat/processors/aggregators/llm_response_universal.py:849-850`**
```python
        else:
            await self.push_frame(frame, direction)
```

세 조각을 함께 읽으십시오. `LLMRunFrame`, `LLMMessagesAppendFrame`, `LLMMessagesUpdateFrame`은 각각
handler를 호출하고 **`push_frame`을 호출하지 않는** `elif` branch에 match합니다. chain의 종단 `else`가
유일한 pass-through입니다. 따라서:

> **`LLMRunFrame`, `LLMMessagesAppendFrame`, `LLMMessagesUpdateFrame`은 user aggregator가 CONSUME합니다.
> 그것들은 LLM service에 결코 도달하지 않습니다. assistant aggregator에도 결코 도달하지 않습니다.**

LLM에 도달하는 것은 `push_context_frame()`이 downstream으로 push하는 `LLMContextFrame`이고 —
[[ch-09/read]] §2가 복사본이 아니라 살아 있는 reference임을 증명하는 데 한 section을 쓴 그 공유된
`LLMContext` object를 실어 나릅니다. node의 message는 몇 microsecond 전에, 같은 processor가,
`add_messages` / `set_messages`로 그 object 안에 써넣었습니다.

그래서 "node transition이 어디서 inference가 되는가"에 대한 진짜 답은: **list의 세 번째 processor,
LLM보다 두 hop 앞에서**이고, mechanism은 공유 object의 mutation 뒤에 그것에 대한 pointer가 따라가는
것입니다.

이것은 또 당신이 물어볼 생각조차 못했을 질문 하나를 정리합니다. `LLMAssistantAggregator`는 context
frame을 **UPSTREAM**으로 push하는 자기만의 `_handle_llm_run`(`llm_response_universal.py:1705-1706`)을
가지고 있습니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:1705-1716`**
```python
    async def _handle_llm_run(self, frame: LLMRunFrame):
        await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_llm_messages_append(self, frame: LLMMessagesAppendFrame):
        self.add_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame(FrameDirection.UPSTREAM)

    async def _handle_llm_messages_update(self, frame: LLMMessagesUpdateFrame):
        self.set_messages(frame.messages)
        if frame.run_llm:
            await self.push_context_frame(FrameDirection.UPSTREAM)
```

canonical pipeline에서 그 코드는 downstream `LLMRunFrame`에 대해서는 도달 불가능합니다. user
aggregator가 네 processor 전에 그것을 먹었기 때문입니다. 그것은 frame이 **upstream**으로 queue되는
경우 — tail로 들어오는 경우 — 와, 앞에 user aggregator 없이 assistant aggregator를 가진 pipeline을
위해 존재합니다. 대칭인 handler 둘, direction당 하나가 살아 있음. 그것이 [[ch-04/read]] §10의
direction split이 실제 processor에서 중복된 코드로 나타난 모습입니다.

### 3.3 `LLMSetToolsFrame`은 끝까지 가는 유일한 것이고 — 두 번 처리된다

tool frame은 다르게 동작하고, 그 차이가 시사적입니다.

**`src/pipecat/processors/aggregators/llm_response_universal.py:822-834`** — user aggregator:
```python
        elif isinstance(frame, LLMSetToolsFrame):
            # Normalize and validate (a plain list of direct functions / FunctionSchema
            # objects becomes a ToolsSchema) so the tool-change diff and
            # set_tools see a consistent type.
            normalized_tools = LLMContext._normalize_and_validate_tools(frame.tools)
            self._maybe_add_tool_change_messages(normalized_tools)
            self.set_tools(normalized_tools)
            # Push the LLMSetToolsFrame as well, since speech-to-speech LLM
            # services (like OpenAI Realtime) may need to know about tool
            # changes; unlike text-based LLM services they won't just "pick up
            # the change" on the next LLM run, as the LLM is continuously
            # running.
            await self.push_frame(frame, direction)
```

**`src/pipecat/processors/aggregators/llm_response_universal.py:1585-1591`** — assistant aggregator:
```python
        elif isinstance(frame, LLMSetToolsFrame):
            # Normalize and validate (a plain list of direct functions / FunctionSchema
            # objects becomes a ToolsSchema) so the tool-change diff and
            # set_tools see a consistent type.
            normalized_tools = LLMContext._normalize_and_validate_tools(frame.tools)
            self._maybe_add_tool_change_messages(normalized_tools)
            self.set_tools(normalized_tools)
```

같은 세 줄에서 push만 빠졌습니다. 그래서 frame은 processor 3에서 처리되고, forward되고, processor
4에서 LLM service가 보고, processor 7에서 다시 처리되고 거기서 멈춥니다.

**같은 공유 `LLMContext`**에 대해 두 aggregator가 `set_tools`를 부르는 것은 idempotent합니다. 두
aggregator가 `_maybe_add_tool_change_messages`를 부르는 것은 idempotent하지 않을 텐데 — 그것은 delta를
알리는 developer message를 append합니다 — 그 method의 docstring이 이 점을 이례적으로 명시적으로
생각해 두었기 때문에 그렇지 않습니다:

**`src/pipecat/processors/aggregators/llm_response_universal.py:413-428`**
```python
    def _maybe_add_tool_change_messages(self, new_tools: ToolsSchema | NotGiven) -> None:
        """Append a developer message describing tool add/remove deltas.

        No-op unless ``add_tool_change_messages`` was enabled on the aggregator,
        and no-op when the diff against the currently advertised tools is empty.
        Custom (LLM-specific) tools are ignored — only standard tools are diffed.

        Both aggregators call this on every ``LLMSetToolsFrame`` they handle.
        Whichever aggregator handles the frame first computes a real diff
        against the shared context and adds the announcement; by the time
        the other aggregator sees it (if at all), the context already
        reflects the new tools, so its diff is empty and no duplicate
        message is added. This is order-independent: it works whether the
        frame flows downstream (user aggregator first) or upstream
        (assistant aggregator first, and consumed without being forwarded).
        """
```

*"the context already reflects the new tools, so its diff is empty."* (context가 이미 새 tool을
반영하므로 그 diff는 비어 있다.) 이 중복 제거는 guard가 아닙니다. 두 aggregator가 **같은 살아 있는
object**에 대해 diff한다는 사실의 귀결입니다. 이것은 [[ch-09/read]] §2.3의 "살아 있는 list가
load-bearing이다"라는 주장에 대한 가장 날카로운 단일 예시입니다 — 중복 억제 behaviour의 정확성이
두 processor가 복사본 둘이 아니라 하나의 object를 보고 있다는 사실에 전적으로 얹혀 있습니다.

그리고 text LLM에 대해 tool 변경은 실제로 어디서 발효됩니까? `LLMSetToolsFrame`에서가 아닙니다:

**`src/pipecat/services/llm_service.py:712-723`**
```python
        if isinstance(frame, LLMContextFrame):
            # Sync the registered handlers with the tools advertised in the
            # context: register any newly advertised handler, drop the ones we
            # auto-registered that are no longer advertised. The context carries
            # the current tool set on every inference, so this is the single place
            # tool changes take effect for text LLMs.
            #
            # Realtime (speech-to-speech) services run continuously and don't get a
            # fresh context frame per turn, so they additionally call
            # _sync_registered_tool_handlers on their own LLMSetToolsFrame
            # handling.
            self._sync_registered_tool_handlers(frame.context.tools)
```

*"the single place tool changes take effect for text LLMs."* (text LLM에 대해 tool 변경이 발효되는
유일한 곳.) 그래서 node의 tool 교체가 안착하는 이유는 user aggregator가 run 전에 그것을 공유 context에
써넣었고 LLM이 context를 읽었기 때문입니다. LLM을 지나쳐 여행하는 `LLMSetToolsFrame`은 오직
speech-to-speech service를 위한 것입니다. 이것은 [[ch-09/read]] §3의 "LLM service는 아무것도 쥐지
않는다"를 Flows 쪽에서 다시 진술한 것이고, `FlowManager`가 `register_function`을 결코 호출하지 않는다는
뜻입니다 — 그것은 schema 위에 handler를 광고하고 service가 그것들을 auto-register하게 둡니다(§8.4).

### 3.4 trace 조립하기

§3.1–§3.3을 합치십시오. node transition 하나, canonical pipeline 하나, 그리고 각 frame이 실제로
가는 경로:

| Frame (queue 순서) | 들어가는 곳 | Consume/정지 지점 | 한 일 |
|---|---|---|---|
| `LLMUpdateSettingsFrame` *(conditional)* | head | `llm` (`llm_service.py:692`) | service에 system instruction 설정 |
| `LLMMessagesAppendFrame` **또는** `LLMMessagesUpdateFrame` | head | `context_aggregator.user()` (proc 3) | 공유 context에 `add_messages` / `set_messages` |
| `LLMSetToolsFrame` | head | `context_aggregator.assistant()` (proc 7) | 공유 context에 `set_tools`, 두 번, idempotent하게 |
| — *별개의 `queue_frames` 호출* — | | | |
| `LLMRunFrame` *(conditional)* | head | `context_aggregator.user()` (proc 3) | `LLMContextFrame`을 downstream push → inference |

그 표에서 [[ch-12/read]]로 가져갈 세 가지:

1. **모든 flow frame은 무엇이든 하기 전에 `transport.input()`과 `stt`를 통과해야 한다.** 그 두
   processor는 이 type들 중 무엇도 handle하지 않으므로 종단 `push_frame`으로 떨어집니다 — 하지만
   그것들은 실제 event loop 위의 실제 queue hop이고, rule 평가를 위해 head에 끼워 넣는 어떤
   processor든 *state machine 자신의 트래픽 앞에* 앉게 됩니다.
2. **batch는 transaction이 아니다.** frame 넷, `queue_frames` 호출 둘, 각각 안에 평범한 `for` loop,
   그리고 queue 위의 다른 무엇에 대해서도 ordering 보장 없음.
3. **tool 교체는 processor 7에서 완료되는데 inference trigger는 processor 3에서 완료된다.** 그것들은
   tools-then-run 순서로 queue되고 각각은 *각 processor에서* 순서대로 처리되므로, processor 3에서
   tools frame이 run frame보다 먼저 보이고 context는 올바릅니다. 하지만 두 효과는 서로 다른 깊이에서
   끝나고, §4.6이 barge-in이 그 중간에 떨어질 때 무슨 일이 벌어지는지 보여 줍니다.

> 💡 **쉬운 설명 — "서로 다른 깊이에서 끝난다"가 왜 문제인가요?**
> frame이 파이프를 따라 흘러가면서, 각 frame은 자기를 소비하는 processor에서 멈춥니다. run frame은
> 3번에서, tools frame은 7번에서 멈춥니다. 즉 "transition이 완료됐다"는 시점이 frame마다 다릅니다.
> 만약 중간에 barge-in이 나서 queue가 비워지면, 어떤 frame은 이미 자기 일을 끝냈고 어떤 frame은
> 아직 3번 앞에서 줄 서 있다가 사라집니다. **부분적으로만 적용된 transition**이 남고, `FlowManager`는
> 자기가 전부 적용했다고 믿습니다. §4.6이 이것을 정확히 다룹니다.

---

## 4. `_set_node`가 machine 전부다

transition에서 Flows가 하는 모든 일은 122줄짜리 method 하나 안에서 일어납니다. 조각으로 읽으십시오.

### 4.1 docstring의 순서, 그리고 진짜 순서

**`src/pipecat/flows/manager.py:602-623`**
```python
    async def _set_node(self, node_id: str, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Handles the complete node transition process in the following order:
        1. Execute pre-actions (if any)
        2. Set up messages (role and task)
        3. Register node functions
        4. Update LLM context with messages and tools
        5. Update state (current node and functions)
        6. Trigger LLM completion with new context
        7. Execute post-actions (if any)

        Args:
            node_id: Identifier for the new node.
            node_config: Complete configuration for the node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        if not self._initialized:
            raise FlowTransitionError(f"{self.__class__.__name__} must be initialized first")
```

일곱 단계이고, docstring은 정확합니다. 하지만 step 1이 *맨 앞*이라는 것이 무슨 뜻인지 주목하십시오.
그것이 §9.5가 던질 질문에 대한 답이기 때문입니다: **pre-action은 새 node의 message와 tool이 설치되기
전에 실행된다.** "Pre"는 "transition 이전"을 뜻하지 않습니다. transition은 이미 확정되었습니다.
그것은 "`_update_llm_context` 이전"을 뜻합니다.

### 4.2 setup과 function-schema 구성

**`src/pipecat/flows/manager.py:625-672`**
```python
        try:
            # Clear any pending transition state when starting a new node
            # This ensures clean state regardless of how we arrived here:
            # - Normal transition flow (already cleared in _check_and_execute_transition)
            # - Direct calls to set_node/set_node_from_config
            self._pending_transition = None

            self._validate_node_config(node_id, node_config)
            logger.debug(f"Setting node: {node_id}")

            # Clear any deferred post-actions from previous node
            self._action_manager.clear_deferred_post_actions()

            # Register action handlers from config
            for action_list in [
                node_config.get("pre_actions", []),
                node_config.get("post_actions", []),
            ]:
                for action in action_list:
                    self._register_action_from_config(action)

            # Execute pre-actions if any
            if pre_actions := node_config.get("pre_actions"):
                await self._execute_actions(pre_actions=pre_actions)

            # Build the node's function schemas (carrying handlers)
            new_functions: set[str] = set()

            # Mix in global functions that should be available at every node
            functions_list = self._global_functions + node_config.get("functions", [])

            standard_functions: list[FunctionSchema] = []
            for func_config in functions_list:
                if callable(func_config):
                    tool = FlowsDirectFunctionWrapper(function=func_config)
                elif isinstance(func_config, FlowsFunctionSchema):
                    tool = func_config
                else:
                    raise InvalidFunctionError(
                        f"Invalid function format in node '{node_id}'. "
                        "Use FlowsFunctionSchema or direct functions."
                    )
                standard_functions.append(await self._create_function_schema(tool))
                new_functions.add(tool.name)

            formatted_tools = (
                ToolsSchema(standard_tools=standard_functions) if standard_functions else NOT_GIVEN
            )
```

주목할 것이 넷입니다.

**`:654`의 `self._global_functions + node_config.get("functions", [])`** — global function은 모든
node의 list에 **앞에 붙습니다**. 이것이 constructor에서 한 번 설정되는(`global_functions=[...]`)
mixin channel이고, boson의 `_GLOBAL_TOOLS`에 직접 매핑됩니다(§13.2).

**`:670-672`의 `ToolsSchema(...) if standard_functions else NOT_GIVEN`** — function이 없는 node는
빈 schema가 아니라 `NOT_GIVEN`을 만들어 냅니다. §4.3이 그것이 무엇을 하는지 보여 줍니다.

**`callable`인 무엇에든 `FlowsDirectFunctionWrapper(function=func_config)`** — direct-function
path입니다. node의 `functions` list에 있는 맨 `async def`는 wrap되고 그 schema는 signature와
docstring에서 유도됩니다(§8.4).

**어디에도 `register_function` 호출이 없다.** handler는 `FunctionSchema` 위에 찍히고 LLM service가
§3.3에 따라 context에서 그것을 집어 갑니다.

### 4.3 role-message 분기, 그리고 context 갱신

**`src/pipecat/flows/manager.py:674-700`**
```python
            role_message = node_config.get("role_message")
            role_messages = node_config.get("role_messages")

            if role_message and role_messages:
                logger.warning(
                    "Both 'role_message' and 'role_messages' specified; using 'role_message'"
                )

            if role_messages and not role_message:
                if not self._showed_deprecation_warning_for_role_messages:
                    self._showed_deprecation_warning_for_role_messages = True
                    warnings.warn(
                        "'role_messages' is deprecated and will be removed in 2.0.0. "
                        "Use 'role_message' (singular, str) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            # Update LLM context
            await self._update_llm_context(
                role_message=role_message,
                role_messages=role_messages if not role_message else None,
                task_messages=node_config["task_messages"],
                functions=formatted_tools,
                strategy=node_config.get("context_strategy"),
            )
            logger.debug("Updated LLM context")
```

`node_config["task_messages"]` — 직접 subscript, `.get`이 아닙니다. 여기가 `Required` key가
load-bearing이 되는 곳이고, `_validate_node_config`가 같은 이유로 이미 `FlowError`를 raise하지
않았다면 `KeyError`가 났을 것입니다(§10.1).

그리고 이제 batch 자체입니다.

### 4.4 emit되는 frame batch — 이것을 외우십시오

**`src/pipecat/flows/manager.py:762-771`**
```python
        try:
            frames = []

            # New path: role_message as LLM system instruction (persists until changed)
            if role_message:
                frames.append(
                    LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=role_message))
                )

            messages = []
```

**`src/pipecat/flows/manager.py:773-777`**
```python
            # Legacy path: role_messages prepended to context messages
            if role_messages:
                messages.extend(role_messages)

            update_config = strategy or self._context_strategy
```

**`src/pipecat/flows/manager.py:824-845`**
```python
            # Add task messages
            messages.extend(task_messages)

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

            logger.debug(
                f"Updated LLM context using {frame_type.__name__} with strategy {update_config.strategy}"
            )
```

**규칙 넷. 규칙으로 익히십시오. Flows에서 벌어지는 모든 운영상의 놀라움은 이 중 하나가 발화한
것이기 때문입니다.**

**Rule 1 — `LLMUpdateSettingsFrame`은 conditional하고 PERSISTENT하다.** 오직 `if role_message`일 때만
나타납니다. "system instruction을 지우는" 경로는 없습니다. node A가 한 번 `role_message`를 설정하면,
그 system instruction은 그것을 생략한 node B, C, D 동안 LLM service 위에 남아 있다가 node E가 다른
것을 설정할 때까지 유지됩니다. `insurance_quote.py`에서는 정확히 하나의 node — 최초의 것 — 만 그것을
설정하고, persona는 네 번의 transition을 전부 살아남습니다. 그것이 의도된 사용법이고,
`patient_intake.py`는 그것을 증명하는 예외를 보여 줍니다: `create_prescriptions_node`는 동일한
`role_message` string을 다시 선언하는데, 그것이 `ContextStrategy.RESET`도 설정하기 때문에 저자가
지워진 message list와 나란히 persona를 재천명하고 싶었기 때문입니다.

**Rule 2 — Append 대 Update 선택이 곧 `ContextStrategy`다.** 그것은 "frame 선택에 영향을 주는
strategy"가 아닙니다. package 전체에서 이 enum의 유일한 runtime 효과가 이 ternary입니다. `RESET`과
`RESET_WITH_SUMMARY`는 `LLMMessagesUpdateFrame`(즉 `set_messages` — 대체)으로 매핑되고, 그 외는
`LLMMessagesAppendFrame`(`add_messages` — 추가)으로 매핑됩니다. §7이 통째로 이 줄에 관한 것입니다.

그 위의 주석도 읽으십시오. 그것이 아니었다면 버그라고 불렀을 design decision을 설명하기 때문입니다:
*"even the first node follows the same rule: appending ensures any prior context contributions, such
as by tts_say pre-actions, is preserved rather than replaced."* (첫 node조차 같은 규칙을 따른다:
append는 tts_say pre-action 같은 이전의 context 기여가 대체되지 않고 보존되도록 보장한다.) 대화의
가장 첫 node는 빈 context에 append하는데, 그것은 대체하는 것과 동일합니다 — pre-action이 이미 한 줄을
말하고 그것을 append하지 않았다면. append는 첫 node를 포함한 모든 node에서 안전한 기본값입니다.

**Rule 3 — `LLMSetToolsFrame`은 UNCONDITIONAL하다.** `if`가 없습니다. `:839`, 모든 단일 transition,
예외 없이. 그리고 node가 아무 function도 선언하지 않으면 `functions`는 `NOT_GIVEN`(§4.2)이고,
`frames.py:700-704`에 있는 그 docstring이 그것을 그대로 말합니다: *"or ``NOT_GIVEN`` to clear tools."*

> **`functions` key가 없는 node는 "tool을 그대로 둔다"가 아닙니다. tool 집합을 CLEAR합니다.**

이것은 design 전체에서 가장 결과가 큰 비대칭이고 §13.5가 그것이 boson에서 무엇과 충돌하는지에 관한
것입니다. tool-neutral한 node 같은 것은 존재하지 않습니다.

**Rule 4 — `LLMRunFrame`은 conditional하고 별도로 QUEUE된다.** 그것은 `frames` 안에 없습니다. 그것은
`_set_node`로 돌아와 `:709`에서, `_update_llm_context`가 이미 반환하고 이미 자기 batch를 flush한
뒤에 queue됩니다:

**`src/pipecat/flows/manager.py:702-719`**
```python
            # Update state
            self._current_node = node_id
            self._current_functions = new_functions

            # Trigger completion with new context
            respond_immediately = node_config.get("respond_immediately", True)
            if respond_immediately:
                await self._worker.queue_frames([LLMRunFrame()])

            # Execute post-actions if any
            if post_actions := node_config.get("post_actions"):
                if respond_immediately:
                    await self._execute_actions(post_actions=post_actions)
                else:
                    # Schedule post-actions for execution after first LLM response in this node
                    self._schedule_deferred_post_actions(post_actions=post_actions)

            logger.debug(f"Successfully set node: {node_id}")
```

왜 `frames`의 다섯 번째 원소가 아니라 별도일까요? 서로 독립적인 이유 둘이고, 두 번째가 뻔하지 않은
쪽입니다.

첫째는 구조적입니다: `respond_immediately`는 `_set_node`에서 읽히고, `_update_llm_context`는 그것을
받지 않습니다.

둘째는 **context frame이 진짜로 스스로 inference를 trigger할 수 없다**는 것입니다:

**`src/pipecat/frames/frames.py:644-657`**
```python
@dataclass
class LLMMessagesAppendFrame(DataFrame):
    """Frame containing LLM messages to append to current context.

    A frame containing a list of LLM messages that need to be added to the
    current context.

    Parameters:
        messages: List of context messages to append.
        run_llm: Whether the context update should be sent to the LLM.
    """

    messages: list[LLMContextMessage]
    run_llm: bool | None = None
```

`run_llm: bool | None = None`. Flows는 `frame_type(messages=messages)`를 구성합니다 — messages는
positional이고 `run_llm`은 기본값 `None`으로 남습니다. 그리고 aggregator의 handler는
`if frame.run_llm:`(§3.2)인데 `None`은 그것을 실패합니다. 그래서 message는 context에 안착하고
**아무것도 돌지 않습니다**. 별도의 `LLMRunFrame`은 스타일 문제가 아닙니다. 그것은 batch에서
inference를 시작하는 유일한 것이고, `respond_immediately=False`는 prompt와 tool 집합을 설치한 다음
사용자가 말하기를 조용히 기다리는 node를 진짜로 만들어 냅니다.

그 flag가 outbound 대 inbound 스위치이고, §12.2가 `restaurant_reservation.py`가 그것을 정확히 한
줄로 쓰는 것을 보여 줍니다.

→ **[ch-10 `_set_node` emitter를 여십시오](./figures/flow-node-transition.html)** — 그리고 더 읽기 전에
네 규칙을 직접 구동해 보십시오. toggle로 `NodeConfig`를 조립하고, 다음 세 가지를 순서대로 하십시오:
(1) node 1에 `role_message`를 설정하고, 그것을 생략하는 node를 둘 진행시키고, `LLMUpdateSettingsFrame`이
*다시 나타나지 않는데도* system instruction이 설정된 채로 남아 있는 것을 보십시오 — 그것이 Rule 1입니다;
(2) `context_strategy`를 `APPEND`에서 `RESET`으로 뒤집고, frame class가 바뀌는 동안 실행 중인 message
list가 자라기를 멈추고 대체되기 시작하는 것을 보십시오 — Rule 2; (3) `functions`에서 모든 항목을 지우고
`LLMSetToolsFrame`이 빈 집합과 함께 그래도 발화하는 것을 보십시오 — Rule 3. layout이 곧 논증입니다:
`FlowManager`는 `Pipeline` box 바깥에 그려져 있고, 각 frame은 head에서 stt와 user aggregator를 거쳐
무엇이든 llm에 도달하기 전에 animate합니다.

### 4.5 batch는 transaction이 아니다

한 번, 정확하게 말하겠습니다. [[ch-12/read]]가 그것을 필요로 하기 때문입니다. node transition은
**두 번의** `queue_frames` 호출을 통해 최대 네 개의 frame을 emit하고, 각 호출은 `queue_frame`에 대한
`for` loop(§3.1)이며, 각각은 `await self._push_queue.put(frame)`을 합니다. 그 put 중 임의의 둘 사이에
event loop는 다른 것을 실행할 수 있습니다 — 도착하는 transcription, push하는 rule processor, VAD
event, `set_node_from_config`를 호출하는 또 다른 coroutine.

이 경로 어디에도 lock도, sequence number도, atomic batch primitive도 없습니다.

실무에서 당신을 구해 주는 것은, 단일 processor 안에서는 frame이 순서대로 dequeue된다는
것([[ch-04/read]] §4.2)입니다. 그래서 `LLMRunFrame`보다 먼저 queue된 `LLMSetToolsFrame`은 processor
3에서 그보다 먼저 *보입니다*. 당신을 구해 주지 *않는* 것은 두 producer 사이의 global ordering에 관한
어떤 것이고, [[ch-04/read]] §10이 이미 그것에 대해 아무 보장도 없다고 진술했습니다.

### 4.6 정직한 발견: batch는 오직 **부분적으로만** interruption-proof다

어떤 excerpt도 언급하지 않는 것이 여기 있고, 그것은 [[ch-08/read]] §3에서 그대로 떨어져 나옵니다.

[[ch-08/read]]는 barge-in 시 `FrameQueue.reset()`(`utils/frame_queue.py:84-95`)이 모든 processor의
queue를 비우고 `UninterruptibleFrame` mixin을 지닌 frame **만** 다시 넣는다는 것을 확립했습니다. 이제
node transition이 emit하는 네 frame을 확인하십시오:

| Frame | 선언 | `UninterruptibleFrame`? |
|---|---|---|
| `LLMUpdateSettingsFrame` | `frames.py:2283`, extends `ServiceUpdateSettingsFrame` | **YES** — `class ServiceUpdateSettingsFrame(ControlFrame, UninterruptibleFrame, Generic[TSettings])`, `frames.py:2251` |
| `LLMMessagesAppendFrame` | `frames.py:645`, `class LLMMessagesAppendFrame(DataFrame)` | no |
| `LLMMessagesUpdateFrame` | `frames.py:661`, `class LLMMessagesUpdateFrame(DataFrame)` | no |
| `LLMSetToolsFrame` | `frames.py:694`, `class LLMSetToolsFrame(DataFrame)` | no |
| `LLMRunFrame` | `frames.py:634`, `class LLMRunFrame(DataFrame)` | no |

persona 변경은 barge-in을 살아남습니다. message, tool 집합, inference trigger는 살아남지 못합니다.

구체적으로: 고객이 말하기 시작할 때 node transition의 frame들이 아직, 가령 STT processor 앞에 queue되어
있다면, `reset()`은 `LLMUpdateSettingsFrame`을 유지하고 나머지 셋을 버립니다. `_set_node`는 이미
성공적으로 반환했습니다. `self._current_node`는 이미 대입되었습니다(`:703`). `FlowManager`는 자기가
새 node에 있다고 믿습니다. LLM은 새 persona, **옛** message, **옛** tool 집합을 가지고 있습니다.

reconciliation 경로는 없습니다. `FlowManager`는 무엇이 안착했는지 결코 되읽지 않습니다. §2.2가
그 유일한 inbound channel이 세 개의 action 관련 frame type이고 그중 무엇도 context batch에 대해
보고하지 않는다는 것을 보여 주었습니다.

실무에서 도달 가능한가? queue한 시점과 frame들이 pipeline 앞부분을 벗어나는 시점 사이의 window 안에
barge-in이 나야 하고, 그 창은 짧습니다. `respond_immediately=False` node가 그것을 넓힙니다. 그런 node는
사용자의 turn에 진입하기 때문입니다 — 정확히 사용자가 말하고 있을 가능성이 높은 때입니다.
transcription에서 발화된 rule 기반 transition([[ch-12/read]]이 설계할 모양)은 구성상 사용자의 turn
동안 진입합니다.

[[ch-12/read]]를 위한 constraint 목록에 적으십시오: **사용자의 turn 중에 발화된 transition은 부분적으로
유실될 수 있고, framework는 그것을 당신에게 말해 주지 않는다.** 완화책은 이국적이지 않습니다 —
interruption이 가라앉은 뒤 §12.3이 `warm_transfer.py`가 transport callback에서 사용하는 것으로 보여 주는
같은 public `set_node_from_config` 경로를 써서 node를 다시 천명하십시오 — 하지만 그것을 만들어야 한다는
것을 알아야 합니다.

> 💡 **쉬운 설명 — 왜 persona만 살아남나요?**
> barge-in은 "지금 대기 중인 것들을 다 버려라"입니다. 그런데 전부 버리면 시스템이 망가지는 것들이
> 있어서, Pipecat은 `UninterruptibleFrame`이라는 표식을 단 frame만 살려 둡니다. `LLMUpdateSettingsFrame`은
> service 설정 변경이라 살려 두는 부류에 속하고, message/tools/run은 "이번 turn의 데이터"로 분류되어
> 버려집니다. 문제는 **node transition이 이 두 부류를 동시에 쓴다**는 것입니다. 그래서 절반만 살아남고,
> `FlowManager`는 자기가 이미 `_current_node`를 바꿔 놨기 때문에 불일치를 알아챌 방법이 없습니다.

---

## 5. Node state는 string이다

### 5.1 선언과 대입

**`src/pipecat/flows/manager.py:147-149`**
```python
        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None
```

**`src/pipecat/flows/manager.py:702-704`**
```python
            # Update state
            self._current_node = node_id
            self._current_functions = new_functions
```

```
$ grep -n "_current_node" src/pipecat/flows/manager.py
149:        self._current_node: str | None = None
248:        return self._current_node
703:            self._current_node = node_id
```

module 전체에 세 site: 선언, `current_node` property의 return, 그리고 단 하나의 대입.
`node_id`는 `str`입니다. **`self._current_node`는 결코 `NodeConfig`를 담지 않습니다.**

이것은 들리는 것보다 더 중요합니다. 그것은 `FlowManager`가 자기가 있는 node에 대한 기억이 없고, 오직
그 node가 무엇으로 *불렸는지*만 기억한다는 뜻입니다. 그것은 현재 node의 `functions`도, `task_messages`도,
`context_strategy`도 말해 줄 수 없습니다. "지금 어떤 tool이 광고되어 있나"라고 물으면 Flows에는 답이
없습니다 — 답은 aggregator들이 써넣은 공유 `LLMContext`(§3.3) 안에 살고, 그것은 완전히 다른 코드가
소유하는 완전히 다른 object입니다.

node object 자체는 `_set_node`가 반환한 뒤 garbage입니다. 그것은 `NodeConfig` dict였고, 그 내용은
frame으로 분해되어 push되었으며, dict는 버려집니다.

[[ch-02/read]] §11은 frame budget의 evidence로 이 같은 세 줄을 인용하며 "현재 node는 `str`이다"라고
말했습니다. 이 chapter는 그것이 왜 임시방편이 아니라 *design*인지를 덧붙입니다: Flows에서 node는
**entity**가 아니라 **event**입니다. 그것은 일어나지, 지속되지 않습니다.

### 5.2 node identity는 한 줄에서 결정되고, 기본값은 UUID다

**`src/pipecat/flows/types.py:509-518`**
```python
def get_or_generate_node_name(node_config: NodeConfig) -> str:
    """Get the node name from configuration or generate a UUID if not set.

    Args:
        node_config: Node configuration dictionary.

    Returns:
        Node name from config or generated UUID string.
    """
    return node_config.get("name", str(uuid.uuid4()))
```

body 한 줄. `name`은 optional이고(§6.1), 그것이 없으면 그 node에 들어갈 때마다 **새 일회용 UUID**가
생깁니다.

그 결과는 미용상의 문제가 아닙니다:

```python
# A node without "name"
if flow_manager.current_node == "collecting_consent":   # never True
    ...
```

이름 없는 같은 node에 두 번 들어가면 `current_node`는 서로 다른 두 string을 담습니다. 어떤 guard든,
어떤 log correlation이든, node 기준 dashboard grouping이든 — 전부 망가지고, 조용히 망가지며, 어디에도
warning이 emit되지 않습니다.

**규칙: 모든 node에 `name`을 설정하십시오.** `insurance_quote.py`는 다섯 개 전부에 설정합니다.
`hello_world.py`는 둘 다에 설정합니다 — 다만 그 end node의 이름이 node의 이름이 아니라 factory의
이름인 `"create_end_node"`라는 점에 유의하십시오. 예제의 작은 비일관성이고, 아무것도 그것을
validate하지 않는다는 것을 상기시킵니다.

### 5.3 guard pattern, in-tree

`warm_transfer.py`는 `current_node`를 실제로 사용하는 파일이고, 그것이 그 예제의 유일한 concurrency
control입니다:

**`examples/flows/warm_transfer.py:650-658`**
```python
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport: DailyTransport, participant: dict[str, Any]):
            """Handle the human agent maybe having joined the call:
            - If the participant who joined is the human agent and we're currently in the "transferring_to_human_agent" node, go to the "human_agent_interaction" node.
            - Otherwise...nothing, for the purposes of this demo. We're assuming the human agent won't join while the conversation flow is any other node.
            """
            user_id = participant.get("info", {}).get("userId")
            if user_id == "agent" and flow_manager.current_node == "transferring_to_human_agent":
                await start_human_agent_interaction(flow_manager=flow_manager)
```

*([[flows-state-machine]] excerpt는 이 guard를 `warm_transfer.py:658`로 인용하지만, source에서는
**657**입니다. 사소하지만, 규칙은 source가 이긴다는 것입니다.)*

`flow_manager.current_node == "transferring_to_human_agent"`는 application 코드 안에서, 상수도 enum도
없이, 손으로 쓴 literal에 대한 string 비교입니다. 그것이 guard 전부입니다. docstring은 나머지에 대해
솔직합니다 — *"Otherwise...nothing, for the purposes of this demo. We're assuming the human agent won't
join while the conversation flow is any other node."*

production tele-sales system에서는 여기가 당신이 자기 규율을 공급해야 하는 지점입니다: node 이름을
위한 enum 또는 상수 module, 그리고 모든 out-of-band transition에 대한 guard. Flows는 도와주지
않습니다.

### 5.4 `_current_functions`: `src/`에서 죽고 tests에서 살아 있음 — 그리고 왜 그 구분이 중요한가

§0.4가 grep을 주었습니다. 정확한 범위가 무엇을 결론지을 수 있는지를 어떻게 바꾸는지가 여기 있습니다.

만약 그것이 어디에서도 전혀 읽히지 않는다면, 당신은 이렇게 말할 것입니다: dead code, 무시하라.
그것이 `tests/test_flows_manager.py`의 여섯 site에서 읽히기 때문에, 당신은 더 조심스러운 무언가를
말해야 하고, 그 조심스러운 버전이 더 유용합니다.

test가 무엇을 assert하는지 보십시오:

**`tests/test_flows_manager.py:186-191`**
```python
        # Test valid config
        valid_config = {"name": "test", "task_messages": []}
        await flow_manager.set_node_from_config(valid_config)

        self.assertEqual(flow_manager._current_node, "test")
        self.assertEqual(flow_manager._current_functions, set())
```

**`tests/test_flows_manager.py:316-319`**
```python
        # Verify all functions were advertised (each carrying a handler) and tracked
        handlers = get_advertised_tool_handlers(self.mock_worker)
        self.assertEqual(set(handlers), {"func_0", "func_1", "func_2"})
        self.assertEqual(len(flow_manager._current_functions), 3)
```

이 field는 **`_set_node`가 올바른 function list를 처리했는지 확인하기 위해 test suite가 사용하는
bookkeeping mirror**입니다. 그것은 실제로 wire에 나간 것을 읽는
`get_advertised_tool_handlers(self.mock_worker)`와 나란히 assert됩니다. `_current_functions`는 값싼
내부 검사이고, mock-worker read가 진짜 검사입니다.

그래서 당신이 할 수 있는 두 주장:

- ✅ *"어떤 runtime code path도 function을 dispatch하기 전에 `_current_functions`를 참조하지 않는다.
  그것은 gate가 아니고, allowlist가 아니며, LLM이 무엇을 호출할 수 있는지에 대한 source of truth가
  아니다."* — 증명 가능, `src/`에 read 0회.
- ❌ *"그것은 codebase 어디에서도 읽히지 않는다."* — 거짓이고, design review에서 잡히는 종류의
  거짓입니다.

그리고 실무적 귀결: **그 위에 permission check를 짓지 마십시오.** boson의 stage별 tool allowlist
([[boson-tool-router]]의 `_allowed_tools_var` ContextVar gate)를 포팅한다면, 그것을 걸어 둘 Flows
field가 없습니다. 모델이 무엇을 호출해도 되는지는 inference 시점의 공유 `LLMContext.tools`에 있는 것에
의해 전적으로 결정되고, 노출과 구별되는 enforcement를 원한다면 그것은 당신의 handler 안이나 감싸는
catch-all 안에 살아야 합니다. §13.5가 이것으로 돌아옵니다.

---

## 6. type vocabulary

`flows/types.py`는 518줄이고 class hierarchy가 없습니다. 그것은 `TypedDict`들, `Enum` 하나,
`@dataclass` 셋, `Callable` alias 더미, 그리고 wrapper class 하나입니다. vocabulary가 곧 design이므로
그렇게 읽으십시오 — [[flows-node-types]]가 사전 요약이고, 아래 전부는 파일에 대해 다시 확인되었습니다.

### 6.1 `NodeConfig`는 `TypedDict(total=False)`다

**`src/pipecat/flows/types.py:224-237`**
```python
    task_messages: Required[list[dict]]
    name: str
    role_message: str
    role_messages: list[dict[str, Any]]
    # ``FlowsFunctionSchema`` and ``FlowsDirectFunction`` are defined below
    # (see the note above ``ConsolidatedFunctionResult``); string forward
    # references keep ``NodeConfig`` definable here without re-introducing the
    # cross-module forward reference that ``ConsolidatedFunctionResult`` used
    # to require.
    functions: "list[FlowsFunctionSchema | FlowsDirectFunction]"
    pre_actions: list[ActionConfig]
    post_actions: list[ActionConfig]
    context_strategy: ContextStrategyConfig
    respond_immediately: bool
```

아홉 개 key, `total=False`, 그래서 `Required`로 표시된 하나를 **제외한** 전부가 optional입니다.

| key | type | 무엇을 제어하는가 | 비고 |
|---|---|---|---|
| `task_messages` | `Required[list[dict]]` | node의 목표 | **유일한 required key**; `manager.py:696`에서 직접 subscript |
| `name` | `str` | node label | 없으면 → 진입마다 새 UUID (§5.2) |
| `role_message` | `str` | persona, LLM **system instruction**으로 | `LLMUpdateSettingsFrame` emit; **node를 넘어 persist** |
| `role_messages` | `list[dict[str, Any]]` | persona, **context message**로 | 1.5.0 deprecated; `messages`에 prepend, settings frame 아님 |
| `functions` | `list[FlowsFunctionSchema \| FlowsDirectFunction]` | 광고되는 tool 집합 | 없으면 → `NOT_GIVEN` → **tool을 clear** |
| `pre_actions` | `list[ActionConfig]` | `_update_llm_context` 이전의 side effect | 여기서 raise하면 node가 중단됨 (§9.5) |
| `post_actions` | `list[ActionConfig]` | run trigger 이후의 side effect | `respond_immediately=False`면 deferred |
| `context_strategy` | `ContextStrategyConfig` | manager 기본값에 대한 node별 override | Append/Update 분기 (§7) |
| `respond_immediately` | `bool` | 진입 시 `LLMRunFrame()` queue | 기본 `True` |

`role_message` / `role_messages` 비대칭을 제대로 주목하십시오. 표가 그 둘이 얼마나 다른지를 감추기
때문입니다. `role_message`(단수)는 `LLMUpdateSettingsFrame`으로 설정되는 **LLM service 위의 system
instruction**이 되어 message list 바깥에 살고, 덮어쓰일 때까지 persist합니다. `role_messages`(복수,
deprecated)는 **`messages`에 prepend되는 평범한 message**가 되고, 이는 그것들이 `ContextStrategy`의
지배를 받는다는 뜻이며 — `RESET`이 그것들을 지웁니다 — 그것들을 선언하는 모든 node에서 다시
추가됩니다. 서로 다른 두 저장 위치, 서로 다른 두 수명. 복수형에서 단수형으로의 migration은 철자
변경이 아니었습니다.

### 6.2 `ContextStrategyConfig`는 `TypedDict`가 아니라 `@dataclass`다

**`src/pipecat/flows/types.py:155-179`**
```python
@dataclass
class ContextStrategyConfig:
    """Configuration for context management.

    Parameters:
        strategy: Strategy to use for context management.
        summary_prompt: Required prompt text when using RESET_WITH_SUMMARY.

            .. deprecated:: 1.5.0
                Use ``LLMContextSummaryConfig.summarization_prompt`` instead.
                Deprecated together with ``RESET_WITH_SUMMARY``. Will be removed
                in 2.0.0.
    """

    strategy: ContextStrategy
    summary_prompt: str | None = None

    def __post_init__(self):
        """Validate configuration.

        Raises:
            ValueError: If summary_prompt is missing when using RESET_WITH_SUMMARY.
        """
        if self.strategy == ContextStrategy.RESET_WITH_SUMMARY and not self.summary_prompt:
            raise ValueError("summary_prompt is required when using RESET_WITH_SUMMARY strategy")
```

node를 저작할 때 손으로 쓰는 두 shape 중 **`NodeConfig`만 `TypedDict`입니다.**
`ContextStrategyConfig`는 진짜 constructor와 진짜 `__post_init__`을 가진 진짜 class입니다.

이 비대칭은 장식이 아닙니다 — 그것은 *언제 당신이 틀렸다는 것을 알게 되는가*의 차이입니다:

- prompt 없는 `ContextStrategyConfig(strategy=RESET_WITH_SUMMARY)` → **construction 시점의
  `ValueError`**, 당신이 쓴 그 줄에서, field 이름이 담긴 메시지와 함께.
- 오타가 있는 `NodeConfig(task_mesages=[...])` → **construction 시점에는 아무것도 없고**, 나중에
  `_validate_node_config`로부터 나오는 `FlowError`, ID string으로부터 알아내야 하는 node에서
  missing-field 오류로 표현됨.
- 오타가 있는 `NodeConfig(respond_immediatly=False)` → **아무것도, 영원히.** key는 dict 안에 앉아
  있고 결코 조회되지 않습니다. `node_config.get("respond_immediately", True)`가 `True`를 반환하고,
  기다리도록 설계한 node가 먼저 말합니다.

마지막 것은 voice agent에서 그것이 어떻게 나타나는지 때문에 운영 노트 하나의 값어치가 있습니다.
당신은 조용히 진입해서 고객을 기다리도록 만든 node를 썼습니다. 그것이 대신 고객 위에 말합니다.
error도, warning도, log 줄도 없습니다. 그것을 찾는 유일한 방법은 key 이름을 한 글자씩 읽거나
behaviour를 테스트하는 것입니다.

`ActionConfig`(`types.py:112-131`) 역시 `Required` key 하나(`type`)를 가진
`TypedDict(total=False)`이고, 그 docstring이 개방성을 명시합니다: *"Additional fields are allowed and
passed to the handler."* 그래서 임의의 payload key가 같은 dict에 실려 다닙니다 — 이는 action key의
오타가 단지 검사되지 않는 것이 아니라, *design상* payload field와 구별 불가능하다는 뜻입니다.

> 💡 **쉬운 설명 — `TypedDict` vs `@dataclass`가 만드는 실무 차이**
> `TypedDict`는 런타임에는 그냥 `dict`입니다. type checker(mypy/pyright)를 돌리지 않으면 오타가
> 전혀 잡히지 않습니다. `@dataclass`는 진짜 class라서 알 수 없는 키워드를 주면 즉시 `TypeError`가
> 납니다. 같은 파일 안에서 두 shape가 섞여 있으므로, **`NodeConfig`를 쓰는 코드에는 type checker가
> 사실상 필수**라고 생각하십시오. voice agent에서는 "조용히 잘못 동작"이 가장 비싼 실패입니다.

### 6.3 edge type

**`src/pipecat/flows/types.py:260-280`**
```python
# ``ConsolidatedFunctionResult`` is the public return-type alias for "direct"
# functions. It must be defined **after** ``NodeConfig`` and without a string
# forward reference: ``get_type_hints()`` on a user-defined direct function
# resolves names against the user's module globals, not this module's, so a
# ``"NodeConfig"`` forward reference here would fail unless the user happened
# to import ``NodeConfig`` themselves.
ConsolidatedFunctionResult = tuple[Any, NodeConfig | None | _NoResponse]
"""Return type for "consolidated" functions.

Return type for "consolidated" functions that do either or both of:
- doing some work
- specifying the next node to transition to after the work is done

The first tuple element is the function-call result delivered to the LLM.
Any JSON-serializable value is accepted (matching Pipecat's upstream
``FunctionCallResultCallback`` contract). The second element is the next node
to transition to, ``None`` to stay on the current node and respond, or
:data:`NO_RESPONSE` to finish without transitioning or responding. Pass a
``None`` *result* to signal a transition-only handler; FlowManager substitutes
an acknowledgement result.
"""

```

**이 한 줄짜리 type alias가 Pipecat Flows의 graph representation 전부입니다.**

`tuple[Any, NodeConfig | None | _NoResponse]` — 원소 0은 tool result로 LLM에 가고, 원소 1이
edge입니다. `Edge` class도, adjacency list도, registry도 없습니다. edge는 handler가 반환하는 순간에
존재하고, 그 전에는 존재하지 않습니다.

세 결과와, source가 그것들에 쓰는 어휘:

**`src/pipecat/flows/manager.py:496-514`**
```python
                is_no_response = next_node is NO_RESPONSE
                if is_no_response or not next_node:
                    # Node function: stay on the current node.
                    properties = FunctionCallResultProperties(
                        run_llm=not is_no_response,
                        on_context_updated=None,
                    )
                else:
                    # Edge function: transition to the returned node.
                    self._pending_transition = {
                        "next_node": next_node,
                        "function_name": name,
                        "arguments": params.arguments,
                        "result": result,
                    }
                    properties = FunctionCallResultProperties(
                        run_llm=False,
                        on_context_updated=self._check_and_execute_transition,
                    )
```

| 반환값 | 호칭 | behaviour |
|---|---|---|
| `(result, some_node_config)` | **edge function** | transition; *새 node*가 run을 trigger하므로 `run_llm=False` |
| `(result, None)` | **node function** | 머무름, `run_llm=True` — 일을 하고, 모델이 그것에 대해 말하게 함 |
| `(result, NO_RESPONSE)` | — | 머무름, `run_llm=False` — 결과는 context에 안착, 모델은 침묵 |

"node function"과 "edge function"이라는 용어는 source 자신의 것이고(`manager.py:498, 504`),
`examples/flows/README.md:34`에서 반복됩니다. 그것을 쓰십시오. docs와 코드가 합의한 어휘입니다.

### 6.4 `NO_RESPONSE`는 identity로 비교된다

**`src/pipecat/flows/types.py:240-257`**
```python
class _NoResponse:
    """Type of the :data:`NO_RESPONSE` sentinel."""

    def __repr__(self) -> str:
        return "NO_RESPONSE"


NO_RESPONSE = _NoResponse()
"""Function return value (in the "next node" slot) indicating "don't immediately respond".

Return ``(result, NO_RESPONSE)`` from a "consolidated" function when the bot
should remain silent after the function finishes, rather than immediately
responding. The function result will make it into the context, but the next
response will be triggered by something else, like a user utterance.

For functions that transition to another node,
``NodeConfig.respond_immediately`` provides the equivalent control.
"""
```

method 하나짜리 private class, module 수준에서 한 번 instantiate되고, `manager.py:496`의 검사는
`next_node is NO_RESPONSE` — **identity이지, equality도 truthiness도 아닙니다.** 정확히 import된
singleton을 반환해야 합니다. 다른 `_NoResponse()` instance는 `not next_node`로 떨어지고, `__bool__`이
없는 object에 대해 그것은 `False`이므로 edge function으로 취급되어 `_set_node`가 `_NoResponse`
instance를 `NodeConfig`로 건네받게 됩니다. 그것은 `_validate_node_config` 안에서 `task_messages` field
누락에 관한 `FlowError`로 실패합니다 — 원인 근처를 전혀 가리키지 않는 오류 메시지입니다.

docstring의 마지막 문장이 대칭을 정확히 이름 붙입니다: `NO_RESPONSE`는 "여기 머물고 침묵하라";
`respond_immediately=False`는 "거기로 가고 침묵하라". 같은 의도, 두 개의 mechanism. 하나는 호출별
결정이고 다른 하나는 node별 property이기 때문입니다.

in-tree 사용례 하나는 진짜 handoff입니다:

**`examples/flows/multi_worker_handoff.py:330-334`**
```python
        # The router is now responsible for the next turn, so hand off with
        # NO_RESPONSE to avoid running the LLM. Note that we don't need to
        # transition to any next node: on_activated re-seeds party_size_node
        # when control returns to the reservation worker.
        return {"status": "transferred"}, NO_RESPONSE
```

다른 worker가 다음 turn을 소유합니다. 결과를 기록하되 말하지 않는 것이 정확히 옳습니다. Lina의 경우
이 모양은 바깥 시스템으로 제어를 넘기는 tool — 결제 단계, SMS 발송 — 에 매핑됩니다. 고객이 이
pipeline이 아닌 다른 무언가에 의해 응대될 곳 말입니다.

### 6.5 forward-reference 함정

§6.3에서 인용한 `types.py:260-265`의 주석을 다시 읽으십시오. 그것은 내부 메모가 아니라 당신 코드에
대한 실제 제약입니다:

> `get_type_hints()` on a user-defined direct function resolves names against the **user's** module
> globals, not this module's.
>
> (사용자 정의 direct function에 대한 `get_type_hints()`는 이 module이 아니라 **사용자의** module
> globals에 대해 이름을 resolve한다.)

실무 규칙: direct function에 `-> tuple[AgeCollectionResult, NodeConfig]`라고 annotate한다면
`NodeConfig`는 **당신의 module에서** import 가능해야 합니다. 그래서 모든 예제 파일이 `FlowManager`를
손으로 구성하지 않을 때조차 맨 위에 `from pipecat.flows import FlowManager, NodeConfig`를 가지고
있는 것입니다. schema 추출이 당신이 쓰지 않은 type을 언급하는 `NameError`로 깨진다면, 이것이 이유입니다.

### 6.6 나머지 type shape들, 간단히

`FlowsFunctionSchema`(`types.py:354-379`)는 `name`, `description`, `properties`, `required`,
`handler`, 그리고 `cancel_on_interruption: bool = False`와 `timeout_secs: float | None = None`을 가진
`@dataclass`입니다. JSON-Schema `properties`가 Python signature로 표현할 수 없는 shape을 필요로 할 때
쓰십시오 — enum, 중첩 object, provider 고유 제약. 그 외에는 examples README가 기본값을 진술합니다:
*"The examples define their functions as 'direct functions' — async functions whose schema is derived
from the signature and docstring — which is the recommended pattern"*(`examples/flows/README.md:47`).

`FlowArgs = dict[str, Any]`(`types.py:59`), 그리고 2.0.0에서 `Mapping[str, Any]`로 넓힐 계획이라는
주석과 함께.

exception hierarchy는 62줄 안의 다섯 class이고, 전부 하나의 base에서 내려옵니다:
`FlowError` ← `FlowInitializationError`, `FlowTransitionError`, `InvalidFunctionError`,
`ActionError`(`exceptions.py:15-61`). `_set_node` 안의 모든 실패는 다시 감싸집니다:
`raise FlowError(f"Failed to set node {node_id}: {str(e)}") from e`(`manager.py:723`). 그래서 단일
`except FlowError`가 이 subsystem이 던질 수 있는 모든 것을 잡습니다 — 원래 type이 `__cause__` 한 hop
떨어져 있게 되는 비용으로.

### 6.7 export된 이름 여섯 개는 이미 2.0.0에서 제거 예정이다

`__init__.py:48-73`은 스물한 개 이름을 export합니다. 다음은 나가는 중입니다:

| deprecated (1.5.0부터) | 대체재 |
|---|---|
| `FlowResult` | **없음** — "No replacement."; JSON-serializable 값 아무거나 반환 |
| `role_messages` (NodeConfig key) | `role_message` (단수 `str`) |
| `ContextStrategy.RESET_WITH_SUMMARY` + `ContextStrategyConfig.summary_prompt` | pre-action에서 `LLMSummarizeContextFrame` |
| `ZeroArgFunctionHandler`, `LegacyFunctionHandler` | `FlowFunctionHandler` — 2인자 `(args, flow_manager)` |
| `LegacyActionHandler` | `FlowActionHandler` — 2인자 `(action, flow_manager)` |
| `flows_direct_function` | `flows_tool_options` |
| `FlowManager(task=...)` 및 `FlowManager.task` | `worker=` / `.worker` |

**당신이 짓는 무엇이든의 target surface:** string `role_message`를 가진 `NodeConfig`;
`FlowsDirectFunction` 또는 `FlowsFunctionSchema`; 2인자 handler; `flows_tool_options`;
`ContextStrategy.APPEND`와 `ContextStrategy.RESET`. 그 외에는 없습니다.

그 목록은 고무적일 만큼 짧고 경고가 될 만큼 깁니다: 이 subsystem의 public API 중 대략 3분의 1이 삭제
예정입니다. 그것은 성숙한 framework 안의 어린 package이고, [[ch-13/read]]는 그것을 없는 척하는 대신
채택 비용으로 가격을 매겨야 합니다.

---

## 7. `ContextStrategy`는 Flows에서 잊어버리는 유일한 knob이다

### 7.1 enum 전체, 그리고 그 단 하나의 효과

**`src/pipecat/flows/types.py:134-152`**
```python
class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Parameters:
        APPEND: Append new messages to existing context (default).
        RESET: Reset context with new messages only.
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary.

            .. deprecated:: 1.5.0
                Use :class:`LLMSummarizeContextFrame` instead — push it in a
                pre-action to trigger on-demand summarization during a node
                transition. See
                https://docs.pipecat.ai/guides/fundamentals/context-summarization.
                Will be removed in 2.0.0.
    """

    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"
```

멤버 셋. 이 enum의 유일한 runtime 소비자는 §4.4에서 인용한 `manager.py:831-836`의 ternary입니다.
`RESET` → `LLMMessagesUpdateFrame` → `set_messages`(대체). `APPEND` → `LLMMessagesAppendFrame` →
`add_messages`(추가).

그것이 mechanism 전부이고, 따라서 그것이 *무엇인지*를 진술할 가치가 있습니다: **`ContextStrategy`는
transition별 context truncation입니다.** `RESET`인 node는 LLM의 message list를 자기 자신의
`task_messages`에서부터 다시 시작하고 그 외에는 아무것도 없습니다. 고객이 말한 모든 것, 봇이 말한
모든 것, 모든 tool result — 다음 inference에서 모델이 보는 것에서 사라집니다.

선택은 manager 전역 기본값과 함께 node별입니다:
`update_config = strategy or self._context_strategy`(`manager.py:777`), 여기서
`self._context_strategy`는 `ContextStrategyConfig(strategy=ContextStrategy.APPEND)`로 기본값을
가집니다(`manager.py:141-143`).

`patient_intake.py`가 의도된 사용법을 보여 줍니다 — 각 수집 단계가 독립적인 긴 구조화된 intake:

**`examples/flows/patient_intake.py:233-246`**
```python
def create_prescriptions_node() -> NodeConfig:
    """Create the prescriptions collection node."""
    return NodeConfig(
        name="get_prescriptions",
        role_message="You are Jessica, an agent for Tri-County Health Services. You must ALWAYS use one of the available functions to progress the conversation. Be professional but friendly.",
        task_messages=[
            {
                "role": "developer",
                "content": "This step is for collecting prescriptions. Ask them what prescriptions they're taking, including the dosage. Get to the point by saying 'Thanks for confirming that. First up, what prescriptions are you currently taking, including the dosage for each medication?'. After recording prescriptions (or confirming none), proceed to allergies.",
            }
        ],
        context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        functions=[record_prescriptions],
    )
```

`RESET`이 무엇을 강제했는지 보십시오: `role_message`가 최초 node의 것과 글자 그대로 동일하게 다시
선언되었습니다. 그것은 중복이 아닙니다 — §6.1이 `role_message`가 system instruction으로서 message
list *바깥에* 산다는 것을 확립했으므로, `RESET`은 그것을 지우지 않았을 것입니다. 다시 선언하는 것은
저자가 방어적으로 구는 것이고, 진입마다 `LLMUpdateSettingsFrame` 하나를 더 비용으로 냅니다.
persona/message 분리가 reference 예제조차 몸을 사릴 만큼 미묘하다는 힌트로 읽으십시오.

**`RESET`을 살아남는 것:** system instruction(`role_message`), 광고된 tool 집합(별개의 field를 쓰는
별개의 frame), 그리고 `flow_manager.state`(manager 위의 평범한 dict, 결코 context 안에 없음).
**살아남지 못하는 것:** 모든 message.

세 번째 항목이 §11.3을 위해 중요한 것입니다: `flow_manager.state`는 context truncation을 살아남는
channel이고, 그것이 정확히 예제들이 수집된 데이터를 transcript에 의존하는 대신 거기에 넣는 이유입니다.

### 7.2 `RESET_WITH_SUMMARY`: 그 위에 짓지 마십시오

deprecated된 세 번째 멤버는 볼 만한 mechanism을 가지고 있습니다. failure mode가 조용하기 때문입니다.

**`src/pipecat/flows/manager.py:792-822`**
```python
            if (
                update_config.strategy == ContextStrategy.RESET_WITH_SUMMARY
                and self._context_aggregator
                and self._context_aggregator.user()._context
            ):
                # We know summary_prompt exists because of __post_init__ validation in ContextStrategyConfig
                summary_prompt = cast(str, update_config.summary_prompt)
                try:
                    # Try to get summary with 5 second timeout
                    summary = await asyncio.wait_for(
                        self._create_conversation_summary(
                            summary_prompt,
                            self._context_aggregator.user()._context,
                        ),
                        timeout=5.0,
                    )

                    if summary:
                        summary_message = self._adapter.format_summary_message(summary)
                        messages.append(summary_message)
                        logger.debug(f"Added conversation summary to context: {summary_message}")
                    else:
                        # Fall back to APPEND strategy if summary fails
                        logger.warning(
                            "Failed to generate summary, falling back to APPEND strategy"
                        )
                        update_config.strategy = ContextStrategy.APPEND

                except TimeoutError:
                    logger.warning("Summary generation timed out, falling back to APPEND strategy")
                    update_config.strategy = ContextStrategy.APPEND
```

세 가지를, 당신을 얼마나 괴롭혀야 하는지 오름차순으로.

**transition 경로 안에 inline으로 박힌 하드코딩된 5.0초 `asyncio.wait_for`.** 설정 불가능합니다.
이 strategy를 쓰는 node transition은 단 하나의 frame이 queue되기 전에 5초 동안 block할 수 있습니다.
음성 대화에서 그것은 latency 퇴행이 아니라 죽은 회선입니다. [[ch-11/read]]가 전체 budget에 숫자를
붙일 것이고, 5초는 그 전부보다 큽니다.

**timeout이나 빈 summary에 대한 조용한 `APPEND` fallback.** `logger.warning` 하나, 그러고 나서 대화는
당신이 설정한 것과 *정반대의* context semantics로 진행됩니다. 당신은 "요약과 함께 깨끗하게 시작"을
요청했는데 "전부 유지"를 받았습니다. 긴 intake의 경우 이것은 2,000 token prompt와 30,000 token
prompt의 차이이고, runtime에, 곁가지 LLM 호출이 제때 돌아왔는지에 의해 결정됩니다.

**`update_config.strategy = ContextStrategy.APPEND`는 config object를 mutate합니다.**
`update_config`는 `strategy or self._context_strategy`인데 — node가 `context_strategy`를 공급했다면
그것은 *node factory 자신의 object*입니다. 당신의 factory가 호출마다 새로 구성하는 대신 module 수준
`ContextStrategyConfig` instance를 반환한다면, 한 번의 timeout이 그 node를 프로세스가 끝날 때까지
영구적으로 격하시킵니다. 아무것도 그것을 복구하지 않습니다.

그리고 `flows/adapters.py`는 이름이 시사하는 것이 아닙니다. 그것은 이 strategy만을 위해 존재하는
2-method class 하나를 담은 68줄입니다:

**`src/pipecat/flows/adapters.py:22-37`**
```python
class LLMAdapter:
    """Helpers for generating and formatting conversation summaries."""

    def format_summary_message(self, summary: str) -> dict:
        """Format a summary as a developer message.

        Summary messages use the LLMContextMessage format (OpenAI-style),
        as summarization triggers an LLMMessagesUpdateFrame.

        Args:
            summary: The generated summary text.

        Returns:
            A developer message containing the summary.
        """
        return {"role": "developer", "content": f"Here's a summary of the conversation:\n{summary}"}
```

per-provider function-schema adapter — [[ch-09/read]] §4가 `src/pipecat/adapters/`에서 찾은 그것 —
를 기대했다면, 이것은 그것이 아닙니다. Flows는 provider-neutral한
`ToolsSchema(standard_tools=[...])`(`manager.py:670-672`)를 emit하고 LLM service 자신의 adapter가
변환하게 함으로써 provider shaping을 통째로 우회합니다. `flows/adapters.py`는 summary를 하고 그 외에는
아무것도 하지 않습니다.

**대신 무엇을 할 것인가**, deprecation body 자신에 따르면: node에 `ContextStrategy.RESET`, 그리고
정말로 summary를 원한다면 `LLMSummarizeContextFrame`을 push하는 pre-action. 그것은 summarization을
transition을 block하는 대신 Pipecat의 native한 out-of-band 경로로 옮깁니다.

### 7.3 왜 이것이 boson이 공짜로 얻는 유일한 것인가

§13.6을 위해 이것을 붙들되, mechanism이 신선한 지금 적어 두십시오. boson의 `_inject_stage`는 오직
`<system-reminder>`로 감싼 stage prompt를 **append**할 뿐입니다([[boson-stage-machine]]). stage layer
어디에도 reset 경로가 없습니다. 그래서 introduction → product_focused → consultation →
informed_consent → purchase를 걷는 긴 Lina 통화는 이전 모든 stage의 prompt를 이후의 모든 inference에
영원히, 자라나면서 끌고 다닙니다.

node 위의 `ContextStrategyConfig(strategy=ContextStrategy.RESET)`가 그 문제를 key 하나로 해결한
것입니다. 그것은 Flows가 제공하는 것 중 boson이 현재 가지고 있지 않은 가장 명확한 단일 항목입니다.

---

## 8. handler dispatch는 typing이 아니라 arity introspection이다

### 8.1 `len(sig.parameters)`에 대한 세 branch

**`src/pipecat/flows/manager.py:409-441`**
```python
        # Get the function signature
        sig = inspect.signature(handler)

        # Calculate effective parameter count
        effective_param_count = len(sig.parameters)

        # Handle different function signatures. inspect.signature has already
        # proven the shape, so each cast narrows the union to the branch we know
        # we're in.
        if effective_param_count == 0:
            if not self._showed_deprecation_warning_for_zero_arg_handler:
                self._showed_deprecation_warning_for_zero_arg_handler = True
                warnings.warn(
                    "Zero-argument function handlers are deprecated and will be "
                    "removed in 2.0.0. Update handlers to accept "
                    "(args: FlowArgs, flow_manager: FlowManager) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return await cast(ZeroArgFunctionHandler, handler)()
        elif effective_param_count == 1:
            if not self._showed_deprecation_warning_for_legacy_handler:
                self._showed_deprecation_warning_for_legacy_handler = True
                warnings.warn(
                    "Single-argument (legacy) function handlers are deprecated "
                    "and will be removed in 2.0.0. Update handlers to accept "
                    "(args: FlowArgs, flow_manager: FlowManager) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return await cast(LegacyFunctionHandler, handler)(args)
        else:
            return await cast(FlowFunctionHandler, handler)(args, self)
```

`0` / `1` / `else`. `Protocol`도, `isinstance`도, 등록 flag도 아닌 — parameter 개수입니다. `cast()`
호출들은 순수하게 type checker를 달래기 위한 것이고, 개수 외에는 runtime에 아무것도 검증되지
않습니다. §1.3의 one-shot deprecation boolean들이 여기 있습니다.

붙들고 있어야 할 귀결: **union `FunctionHandler = ZeroArgFunctionHandler | LegacyFunctionHandler |
FlowFunctionHandler`는 세는 것으로 판별되므로, parameter가 셋인 handler는 `else`에 떨어져 둘로
호출됩니다.** `TypeError`. Flows의 무엇도 상한을 검사하지 않습니다.

### 8.2 `actions.py`는 같은 일을 다른 임계값으로 한다

**`src/pipecat/flows/actions.py:178-209`**
```python
                # Determine if handler can accept flow_manager argument by inspecting its signature
                # Handlers can either take (action) or (action, flow_manager)
                try:
                    sig = inspect.signature(handler)
                    can_handle_flow_manager_arg = len(sig.parameters) > 1
                except (ValueError, TypeError):
                    logger.warning(
                        f"Unable to determine handler signature for action type '{action_type}', "
                        "falling back to legacy single-parameter call"
                    )
                    can_handle_flow_manager_arg = False

                # Invoke handler appropriately, with async and flow_manager arg as needed
                if can_handle_flow_manager_arg:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(action, self._flow_manager)
                    else:
                        handler(action, self._flow_manager)
                else:
                    if not self._showed_deprecation_warning_for_legacy_action_handler:
                        self._showed_deprecation_warning_for_legacy_action_handler = True
                        warnings.warn(
                            "Single-argument (legacy) action handlers are deprecated "
                            "and will be removed in 2.0.0. Update handlers to accept "
                            "(action: dict, flow_manager: FlowManager) instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                    if asyncio.iscoroutinefunction(handler):
                        await handler(action)
                    else:
                        handler(action)
```

3-way 대신 2-way(`> 1`), 그리고 sync/async branch가 추가되었습니다 — action handler는 평범한 `def`일
수 있습니다. `except (ValueError, TypeError)` fallback에 주목하십시오: `inspect.signature`는 일부 C
builtin과 특정 `functools.partial` shape에서 실패하고, 복구는 legacy 1인자 형태를 가정하는 것입니다.

### 8.3 Pipecat 자신의 built-in이 Pipecat 자신의 deprecation warning을 발동시킨다

이제 §8.2를 `actions.py:104-106` 옆에 놓으십시오:

```python
        self._register_action("tts_say", self._handle_tts_action)
```

`self._handle_tts_action`은 **bound method**입니다. 그 선언은
`async def _handle_tts_action(self, action: dict) -> None:`(`actions.py:302`)이고, 일단 bind되면
`self`는 signature에서 사라집니다. 따라서 `len(inspect.signature(handler).parameters)`는 **1**입니다.

그래서 `can_handle_flow_manager_arg`는 `False`이고, legacy branch가 돌며, 어떤 Flows 프로그램에서든
첫 `tts_say` action은 이것을 emit합니다:

```
DeprecationWarning: Single-argument (legacy) action handlers are deprecated and will be
removed in 2.0.0. Update handlers to accept (action: dict, flow_manager: FlowManager) instead.
```

...Pipecat 자신의 built-in에 대해서 말입니다. built-in 셋 모두 이 shape을 가지고 있습니다
(`_handle_tts_action(self, action)`, `_handle_end_action(self, action)`,
`_handle_function_action(self, action)`).

무해합니다 — one-shot flag 덕분에 `ActionManager`당 한 번 발화합니다 — 하지만 두 가지 이유로 알아 둘
가치가 있습니다. 첫째, test에서 `DeprecationWarning`을 error로 바꾼다면(합리적인 일입니다) Flows는
자기 기본값에서 실패합니다. 둘째, 그것은 arity sniffing이 "legacy handler"와 "보이는 parameter가
우연히 하나인 bound method"를 구별하지 못한다는 작은 증거입니다. 당신 자신의 bound-method handler도
같은 branch를 조용히 맞게 되고, 당신이 기대한 `flow_manager` 인자 없이 호출됩니다 — 그리고 만약
`(self, action, flow_manager)`로 선언했다면 그것은 보이는 parameter 둘로 bind되어 *현대* branch를
타므로 `TypeError`로 드러납니다. 두 경우는 parameter 하나 차이인데 완전히 다르게 동작합니다.

**규칙: bound method가 아니라 평범한 module 수준 `async def` handler를 등록하십시오.** method를 꼭
써야 한다면 `@staticmethod`로 만들어서 보이는 parameter 개수가 선언된 shape과 맞게 하십시오.

> 💡 **쉬운 설명 — bound method가 왜 개수를 속이나요?**
> `def f(self, action)`은 parameter 2개짜리 함수처럼 보이지만, `obj.f`로 꺼내는 순간 `self`가 이미
> 채워진 bound method가 되어 **밖에서 보이는 parameter는 `action` 하나**입니다. Flows는 그 "보이는
> 개수"로 handler 세대를 판별하므로, 당신 의도와 무관하게 legacy 취급을 받습니다. 개수로 type을
> 대신하면 이런 종류의 오판이 반드시 생깁니다.

### 8.4 direct function: `flow_manager`는 keyword로, 그리고 그 이름은 문자 그대로다

**`src/pipecat/flows/types.py:461-468`**
```python
    @classmethod
    def special_first_param_name(cls) -> str:
        """Get the special first parameter name for Flows direct functions.

        Returns:
            The string "flow_manager" which is expected as the first parameter.
        """
        return "flow_manager"
```

**`src/pipecat/flows/types.py:496-506`**
```python
    async def invoke(self, args: Mapping[str, Any], flow_manager: "FlowManager"):
        """Invoke the wrapped function with the provided arguments.

        Args:
            args: Arguments to pass to the function.
            flow_manager: FlowManager instance for function execution context.

        Returns:
            The result of the function call.
        """
        return await self.function(flow_manager=flow_manager, **args)
```

`self.function(flow_manager=flow_manager, **args)` — **keyword로.** direct function의 첫 parameter는
반드시 `flow_manager`라고 *이름 지어져야* 합니다. `fm`도, `manager`도, `flow`도 아닙니다. 이름을
바꾸면 등록 시점이 아니라 호출 시점에 예상치 못한 keyword argument에 관한 `TypeError`를 받습니다.

그리고 direct function은 tuple을 반환해야 합니다:

**`src/pipecat/flows/manager.py:483-490`**
```python
                else:
                    result = handler_response
                    next_node = None
                    # FlowsDirectFunctions should always be "consolidated" functions that return a tuple
                    if isinstance(handler, FlowsDirectFunctionWrapper):
                        raise InvalidFunctionError(
                            f"Direct function {name} expected to return a tuple (result, next_node) but got {type(result)}"
                        )
```

`FlowsFunctionSchema` handler는 맨 result를 반환해도 되고, direct function은 안 됩니다. 두 shape,
두 contract.

마지막으로 `transition_func` 안의 error containment에 주목하십시오:

**`src/pipecat/flows/manager.py:518-521`**
```python
            except Exception as e:
                logger.error(f"Error in transition function {name}: {str(e)}")
                error_result = {"status": "error", "error": str(e)}
                await params.result_callback(error_result)
```

**raise하는 handler는 결코 turn을 crash시키지 않습니다.** exception은
`{"status": "error", "error": "..."}`가 되어 평범한 result callback을 통해 전달되고, 그래서 LLM은
실패한 tool call을 보고 그것에 대해 말할 수 있습니다. transition은 일어나지 않습니다 —
`_pending_transition`이 애초에 설정되지 않았습니다. 영업 통화에서 이것은 옳은 기본값입니다:
`verify_personal_info` 중의 database timeout이 끊긴 통화가 아니라 "죄송합니다, 잠시만요"가 됩니다.
하지만 그것은 또 **당신의 handler의 exception이 log를 보지 않는 한 보이지 않는다**는 뜻이기도 합니다.
upstream의 무엇에게도 알려지지 않기 때문입니다.

### 8.5 transition은 in-flight tool call에 대해 gate된다

Flows가 pipeline 바깥에 있기 때문에만 존재하는 mechanism이 하나 더 있습니다. edge function은 즉시
transition하지 않습니다. 의도를 저장하고 callback을 등록합니다:

**`src/pipecat/flows/manager.py:525-550`**
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

    async def _execute_transition(self, transition_info: dict[str, Any]) -> None:
        """Execute the stored transition."""
        next_node = transition_info.get("next_node")

        try:
            if next_node:
                node_name = get_or_generate_node_name(next_node)
                logger.debug(f"Transition to function-returned node: {node_name}")
                await self._set_node(node_name, next_node)
        except Exception as e:
            logger.error(f"Error executing transition: {str(e)}")
            raise
```

`if not assistant_aggregator.has_function_calls_in_progress`가 concurrency guard이고, 그것은
**pipeline 안의 processor** 위의 field를 읽습니다 — [[ch-09/read]] §5가 function-call lifecycle을
소유한다고 보여 준 assistant aggregator입니다. `FlowManager`는 frame을 관찰함으로써 in-flight tool
call에 대해 알 수 없습니다. 관찰할 position이 없기 때문입니다. 그래서 그것은 aggregator의 public
state에 직접 손을 뻗습니다.

이것이 사 주는 것: 모델이 세 개의 tool call을 병렬 batch로 emit하고 그중 하나가 edge를 반환할 때,
transition은 셋 모두를 기다립니다. 그것이 없다면 `_set_node`는 두 호출이 아직 resolve 중인 동안
`LLMSetToolsFrame`을 발화시켜 tool 집합을 지웠을 것입니다.

그 비용: `_worker` reference 위에 더해, pipeline 바깥에서 특정 processor의 내부 state로 향하는 두
번째 hard reference. `FlowManager`는 `get_current_context()`로도 aggregator 쌍에 coupling되어
있습니다(`manager.py:349`: `context = self._context_aggregator.user()._context` — underscore로 시작하는
attribute라는 점에 주목). pipeline 바깥에 있다는 것이 pipeline과 decoupling되어 있다는 뜻은 아닙니다.
그것은 coupling이 frame flow가 아니라 attribute access로 이루어진다는 뜻입니다.

---

## 9. Actions: 정확히 세 개의 verb, 그리고 maintainer는 당신이 하나를 쓰기를 원한다

### 9.1 완전한 built-in vocabulary

**`src/pipecat/flows/actions.py:103-106`**
```python
        # Register built-in actions
        self._register_action("tts_say", self._handle_tts_action)
        self._register_action("end_conversation", self._handle_end_action)
        self._register_action("function", self._handle_function_action)
```

세 줄. 그것이 Pipecat Flows의 built-in action vocabulary 전부입니다([[flows-actions]]).

`transition` action도, `set_node`도, `inject_message`도, `log`도, `webhook`도, `http`도, `wait`도,
`branch`도 **없습니다**. 각각을 찾아봤고, 하나도 존재하지 않습니다. 문자 그대로 한 줄의 텍스트를
말하거나 전화를 끊는 것이 아닌 모든 것은 `{"type": "function", "handler": fn}`으로 표현됩니다.

### 9.2 각각이 하는 일

**`tts_say`** — `actions.py:302-332`. `text`를 요구하고, `append_text_to_context`는 optional(기본
`True`). `TTSSpeakFrame(text=..., append_to_context=...)`를 queue한 다음 `ActionFinishedFrame()`을
queue합니다. `text`가 없으면 **error를 log하고 return합니다** — raise하지 않으므로, 잘못 만들어진
`tts_say`는 §9.6의 wait table에 대해 여전히 성공적으로 실행된 action으로 계산되는 조용한 no-op입니다.

**`end_conversation`** — `actions.py:334-362`. optional한 `text` 작별 인사, 그다음 `EndFrame()`.
주석에 주목하십시오:

**`src/pipecat/flows/actions.py:359-362`**
```python
        await self._worker.queue_frame(EndFrame())

        # NOTE: there's no point queueing an ActionFinishedFrame here, since the previously-queued
        # EndFrame ensures that it'll never get delivered to our observer
```

그리고 `execute_actions`는 그것 다음에 loop를 빠져나옵니다:

**`src/pipecat/flows/actions.py:215-219`**
```python
                # If action was end_conversation, break
                # (If we didn't, we could end up waiting for the next actions to finish, and...they
                # never would)
                if action_type == "end_conversation":
                    break
```

`post_actions` 배열에서 `end_conversation` 뒤에 나열한 무엇이든 **결코 실행되지 않습니다**. 버그가
아닙니다 — 주석이 계속 진행하면 `EndFrame`이 이미 전달 불가능하게 만든 frame에 대해 wait table이
deadlock에 빠질 것이라고 설명합니다. 하지만 그것은
`post_actions=[{"type": "end_conversation"}, {"type": "function", "handler": log_call_outcome}]`가
당신의 logging을 조용히 버린다는 뜻이기도 합니다.

terminal-node pattern이 작동하는 이유는 [[ch-04/read]] §8이 이미 확립한 mechanism 때문입니다:
`EndFrame`은 순서대로 처리되는 `ControlFrame`이라서 `LLMRunFrame`과 그것이 만들어 내는 발화 *뒤에*
줄을 서고, pipeline은 작별 인사가 다 나온 뒤에 teardown되지 그 도중이 아닙니다.
`hello_world.py:112`와 `insurance_quote.py:297` 둘 다 그 ordering에 의존합니다.

**`function`** — `actions.py:364-388`. `handler`를 요구하고, `FunctionActionFrame`을 queue합니다.

**`src/pipecat/flows/actions.py:380-388`**
```python
        # Mark that we're starting the action
        self._increment_ongoing_actions_count()

        # Queue the action frame (we're queueing rather than running it here to ensure it happens
        # at the appropriate time in the pipeline, like when the bot's turn is over, for example).
        await self._worker.queue_frame(FunctionActionFrame(action=action, function=handler))

        # NOTE: we do NOT queue an ActionFinishedFrame here; instead, we will decrement the ongoing
        # actions count when the function has finished executing (the function may take some time)
```

괄호 안을 읽으십시오: *"we're queueing rather than running it here to ensure it happens at the
appropriate time in the pipeline."* (pipeline에서 적절한 시점에 일어나도록 여기서 실행하는 대신
queue한다.) **이것이 `function`이 "그냥 함수를 호출한다"가 아니라 action type으로 존재하는 이유
전부입니다.** handler는 action이 스케줄될 때 실행되지 않습니다. 그것은 frame이 pipeline의 downstream
끝에 도달할 때(§2.2) 실행되고, 그것은 그 앞에 queue된 모든 것이 모든 processor에 의해 처리된
*뒤*입니다 — TTS 오디오를 포함해서. 그것이 `warm_transfer.py`가 봇이 아직 말하는 중이 아니라 "잠시만
기다려 주세요"를 다 말한 순간에 hold music을 시작하는 방법입니다.

이것은 또한 [[ch-02/read]]의 두 frame 중 셋째가 자기 자리를 벌어들이는 것이기도 합니다:
`FunctionActionFrame`이 존재하는 이유는 side effect를 speech에 대해 순서 짓는 일이 stream *안에*
있기를 요구하는데, 평범한 method 호출은 stream 안에 있지 않기 때문입니다.

### 9.3 maintainer는 주석에서 custom action을 de-emphasize한다

**`src/pipecat/flows/actions.py:286-297`**
```python
        else:
            # Either previous action was:
            # - None (the upcoming action is the first one), so there's nothing to wait for.
            # - A fully custom action, where we don't wait, like we've always done. Note that we
            #   could, in the future, add new API affordances for users to tell us to wait for the
            #   the action to finish before moving on to the next one along with a way for them to
            #   tell us when the action is done. But let's hold off on doing that since we're
            #   de-emphasizing custom actions in favor of "function" actions, which should meet most
            #   needs.
            # Note that it should not be possible for the previous action to be "end_conversation",
            # since we stop processing actions after that one.
            pass
```

*"we're de-emphasizing custom actions in favor of `function` actions, which should meet most needs."*
(대부분의 필요를 충족할 `function` action을 선호하여 custom action을 de-emphasize하고 있다.) docs가
아니라 source 주석에 진술된 design 방향입니다. 받아들이십시오: 여러 node에 걸쳐 같은 verb가 필요하고
config가 선언적으로 읽히기를 원할 때만 custom action type을 등록하십시오. 그 외에는
`{"type": "function", "handler": fn}`으로 끝내십시오.

trade-off는 구체적이고 §9.6의 표에 살아 있습니다: custom action type은 **ordering support를 전혀 받지
못합니다** — wait table은 어느 방향으로도 그것을 기다리지 않습니다. `function` action은 항상
block합니다. 그래서 escape hatch가 올바른 ordering을 가진 쪽이고, "더 깔끔한" API가 조용히 race하는
쪽입니다.

### 9.4 action이 할 수 없는 것

세 개의 부재이고, 그것들은 올바른 종류의 부재입니다:

1. **action은 다음 node를 고를 수 없다.** transition하는 action type이 없습니다. node 선택은 전적으로
   `ConsolidatedFunctionResult`의 두 번째 원소(§6.3), 또는 out-of-band `set_node_from_config` 호출에
   삽니다.
2. **action은 transition을 거부(veto)할 수 없다.** framework가 검사하는 반환값이 없습니다. 현대
   signature는 `Callable[[dict, FlowManager], Awaitable[None]]` — `None`을 반환합니다.
3. **action은 어떤 것에도 값을 반환할 수 없다.** LLM에도, node에도, `FlowManager`에도. 그것은 오직
   side effect로만 소통합니다: `flow_manager.state` mutate, frame queue, 당신 자신의 시스템 호출.

boson에게 이것은 가장 큰 단일 vocabulary mismatch입니다.
`ActionType = Literal["continue", "respond", "inject", "compact", "pre_tool", "stage_transition",
"filter", "pass"]`가 `stage_transition`, `filter`, `pass`를 포함하기 때문입니다 — transition verb 하나와
routing verdict 둘. 셋 중 무엇도 Flows에 action-모양의 거처가 없습니다. §13.3과 [[ch-12/read]]가
그것들이 대신 어디로 가는지를 다룹니다.

### 9.5 raise하는 PRE-action은 node를 중단시키고, post-action 실패는 그럴 수 없다

veto API는 없지만, veto처럼 동작하는 ordering 사실은 **있습니다**.

**`src/pipecat/flows/actions.py:220-224`**
```python
            except Exception as e:
                # Undo any increment of ongoing actions count that happened during this action
                if self._ongoing_actions_count > ongoing_actions_count:
                    self._decrement_ongoing_actions_count()  # Assumption: on increment per action
                raise ActionError(f"Failed to execute action {action_type}: {str(e)}") from e
```

handler 안의 exception은 `ActionError`가 되고, 그것은 `_execute_actions` 밖으로, `_set_node`의 `try`
밖으로 전파되어 `manager.py:721-723`에서
`FlowError(f"Failed to set node {node_id}: ...")`로 다시 감싸집니다.

이제 §4.1의 ordering을 상기하십시오:

```
pre-actions (:647-648)  →  _update_llm_context (:693)  →  _current_node = node_id (:703)
                        →  LLMRunFrame (:709)          →  post-actions (:712-714)
```

- **raise하는 pre-action**은 `_update_llm_context` 이전에 일어납니다. 어떤 frame도 queue되지
  않았습니다. `self._current_node`는 대입되지 않았습니다. **node는 사실상 중단**되고 caller는
  `FlowError`를 받습니다. 이것은 진짜로 쓸 수 있는 guard입니다: pre-action에 validation check를 넣으면
  실패가 대화를 있던 자리에 유지합니다.
- **raise하는 post-action**은 그 전부 이후에 일어납니다. frame은 queue되어 여행 중이고, node ID는
  설정되었고, inference는 돌고 있습니다. **node는 un-set될 수 없습니다.** 당신은 이미, 돌이킬 수 없이
  일어난 transition을 기술하는 `FlowError`를 받습니다.

또한 pre-action guard는 "context가 바뀌기 전"이라는 `respond_immediately=True` semantics에 대해서*만*
신뢰할 수 있다는 점에 유의하십시오. 그것은 아무것도 roll back하지 않습니다. 그저 roll back할 것이
아무것도 없을 만큼 일찍 돌 뿐입니다.

### 9.6 wait table은 하드코딩되어 있고 사용자 설정 불가능하다

**`src/pipecat/flows/actions.py:264-285`**
```python
        needs_wait = False
        if previous_action_type == "tts_say":
            # "tts_say" enqueues a TTSSpeakFrame, which has an effect when it hits the TTS node in
            # the pipeline.
            # As long as the upcoming action enqueues a frame with an effect at the same point or
            # later in the pipeline, we don't need to wait.
            # If the upcoming action is:
            # - "tts_say": no need to wait (effect happens at the same point)
            # - "end_conversation": no need to wait (effect happens at the end of the pipeline)
            # - "function": no need to wait (effect happens at the end of the pipeline)
            #  - None: wait (we're done with this set of actions; the next thing to occur may be a
            #    node change/LLM context update, which has an effect earlier in the pipeline)
            # - custom action: wait (we don't know what it will do)
            if upcoming_action_type not in ["tts_say", "end_conversation", "function"]:
                needs_wait = True  # None or custom action
        elif previous_action_type == "function":
            # "function" enqueues a FunctionActionFrame, which has an effect at the end of the
            # pipeline.
            # Functions can take some time to execute (and don't hold up the pipeline as they're
            # doing so), so we need to wait for them to finish before proceeding with the next
            # action or moving on from the current set of actions.
            needs_wait = True
```

| previous | upcoming | 기다리나? | 왜 |
|---|---|---|---|
| `None` (첫 action) | 무엇이든 | no | in-flight인 것이 없음 |
| `tts_say` | `tts_say` / `end_conversation` / `function` | no | 효과가 같은 깊이 또는 그보다 뒤에 안착 |
| `tts_say` | custom type, 또는 `None` (batch의 끝) | **yes** | 알 수 없는 깊이, 또는 context update가 다음이고 그것이 *더 앞에* 안착 |
| `function` | `None`을 포함해 무엇이든 | **yes, 항상** | 시간이 걸릴 수 있고 pipeline은 그것을 붙들지 않음 |
| custom type | 무엇이든 | no | §9.3의 의도적인 공백 |

그 논리는 시간이 아니라 진정으로 **pipeline 깊이**에 관한 것입니다. frame의 효과는 pipe 안의 고정된
position에 안착하므로, 다음 것이 in-flight인 것보다 *더 앞에서* 작용할 때만 기다리면 됩니다. 그것은
pipes-and-filters 시스템에 대한 compositional 논증입니다 — [[ch-01/read]]의 topology가 scheduling
rule로 나타난 것입니다.

bookkeeping은 counter 하나에 event 하나입니다:

**`src/pipecat/flows/actions.py:390-400`**
```python
    def _increment_ongoing_actions_count(self) -> None:
        """Increment the count of ongoing actions and reset the finished event if this is the first action."""
        self._ongoing_actions_count += 1
        if self._ongoing_actions_count == 1:
            self._ongoing_actions_finished_event.clear()

    def _decrement_ongoing_actions_count(self) -> None:
        """Decrement the count of ongoing actions and set the finished event if this was the last action."""
        self._ongoing_actions_count = max(0, self._ongoing_actions_count - 1)
        if self._ongoing_actions_count == 0:
            self._ongoing_actions_finished_event.set()
```

**감소는 handler가 반환하는 것이 아니라 frame이 downstream에 도착하는 것에 의해 구동됩니다** — 그것이
§2.2의 `on_frame_reached_downstream`입니다. 그리고 그것은 §2.3으로 되돌아갑니다: downstream filter를
지우면 이 counter는 결코 감소하지 않고, `actions.py:300`의
`await self._ongoing_actions_finished_event.wait()`는 영원히 hang합니다.

### 9.7 built-in은 이름으로 override 가능하다

`_register_action`은 평범한 dict write입니다:

**`src/pipecat/flows/actions.py:129-142`**
```python
    def _register_action(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say").
            handler: Async or sync function that handles the action.

        Raises:
            ValueError: If handler is not callable.
        """
        if not callable(handler):
            raise ValueError("Action handler must be callable")
        self._action_handlers[action_type] = handler
        logger.debug(f"Registered handler for action type: {action_type}")
```

collision check가 없습니다. 그래서 `flow_manager.register_action("tts_say", my_handler)`는 built-in을
조용히 대체하고, in-tree 예제가 `end_conversation`으로 정확히 그렇게 합니다:

**`examples/flows/multi_worker_handoff.py:338-341`**
```python
    async def end_conversation_action(action: dict) -> None:
        await worker.end(reason=action.get("reason"))

    flow_manager.register_action("end_conversation", end_conversation_action)
```

multi-worker 구성에서 "대화를 끝낸다"는 공유 pipeline에 `EndFrame`을 queue하는 것이 아니라 *이
worker*를 끝내야 합니다. 이름으로 override하는 것이 그렇게 말하는 지원되는 방법입니다.

하지만 override가 무엇을 잃는지 주목하십시오: `_maybe_wait_for_ongoing_actions_to_finish`는 여전히
*string* `"end_conversation"`을 특수 처리하므로(`actions.py:218`, 그리고 `:295-296`의 note), `break`
behaviour와 ordering 가정은 당신의 대체물이 완전히 다른 일을 함에도 여전히 적용됩니다. wait table은
behaviour가 아니라 이름을 key로 삼습니다.

암묵적 등록 경로가 나머지 하나입니다:

**`src/pipecat/flows/manager.py:379-392`**
```python
        action_type = action.get("type")
        handler = action.get("handler")

        # Register action if not already registered
        if action_type and action_type not in self._action_manager._action_handlers:
            # Register handler if provided
            if handler and callable(handler):
                self.register_action(action_type, handler)
                logger.debug(f"Registered action handler from config: {action_type}")
            else:
                raise ActionError(
                    f"Action '{action_type}' not registered. "
                    "Provide handler in action config or register manually."
                )
```

`ActionConfig` 안의 inline `handler`는 첫 사용 시 self-register합니다.
`if action_type not in self._action_manager._action_handlers`에 주목하십시오 — **첫 등록이 이깁니다.**
그래서 같은 type을 다른 handler로 선언하는 두 번째 node는 조용히 무시됩니다. node마다 다른 것을
의미하는 verb에 inline pattern을 쓴다면, 오직 첫 node의 handler만 실행됩니다.

---

## 10. Flows가 주지 않는 것

port를 고려하고 있다면 두 번 읽을 section입니다. 그 부재들이 빠진-feature 모양이 아니라 구조적이기
때문입니다.

### 10.1 `_validate_node_config`는 정확히 두 가지를 검사한다

**`src/pipecat/flows/manager.py:867-898`**
```python
    def _validate_node_config(self, node_id: str, config: NodeConfig) -> None:
        """Validate the configuration of a conversation node.

        This method ensures that:
        1. Required fields (task_messages) are present.
        2. Each function is either a ``FlowsFunctionSchema`` or a valid direct
           function.

        Args:
            node_id: Identifier for the node being validated.
            config: Complete node configuration to validate.

        Raises:
            FlowError: If required fields are missing.
            InvalidFunctionError: If function format is invalid.
        """
        # Check required fields
        if "task_messages" not in config:
            raise FlowError(f"Node '{node_id}' missing required 'task_messages' field")

        # Get functions list with default empty list if not provided
        functions_list = config.get("functions", [])

        # Validate each function configuration if there are any
        for func in functions_list:
            if callable(func):
                FlowsDirectFunctionWrapper.validate_function(func)
            elif not isinstance(func, FlowsFunctionSchema):
                raise InvalidFunctionError(
                    f"Invalid function format in node '{node_id}'. "
                    "Use FlowsFunctionSchema or direct functions."
                )
```

그것이 method 전부이고, Pipecat Flows의 validation surface 전부입니다.

1. `"task_messages"`가 key인가? (비어 있지 않은 list인지가 아님. 그 항목들이 well-formed message인지도
   아님. `task_messages=[]`는 통과합니다.)
2. `functions`의 각 항목이 callable이거나 `FlowsFunctionSchema`인가?

**현재 node를 target node에 대해 검사하는 것은 없습니다. 어디에도.** 여기에도, `_set_node`에도,
`_execute_transition`에도, `set_node_from_config`에도.

```
$ grep -rn "not allowed\|illegal transition\|invalid transition" src/pipecat/flows/
(no output)
```

이는 곧: `insurance_quote.py`의 종단 `end` node에서, `create_initial_node()`를 반환하는 handler는
introduction으로 깔끔하게 transition합니다. Flows는 initial node에 `task_messages`가 있는지 validate하고,
있다는 것을 발견하고, 갑니다. error도, warning도, `debug` 위의 log 줄도 없습니다.

### 10.2 node registry가 없다

`_set_node`는 `NodeConfig` **object**를 받습니다. 아무것도 조회하지 않습니다. 알려진 node의 dict도,
`register_node`도, `has_node(name)`도, `get_node(name)`도, 열거도 없습니다.

그래서 다음은 shipped API로는 전부 불가능합니다:

- **graph를 열거하기.** "어떤 node가 존재하는가"를 물을 수 없습니다 — 답은 당신이 우연히 쓴 Python
  함수들 안에 삽니다.
- **initial node를 validate하기.** 설정된 시작 node가 존재하는지 아무것도 확인할 수 없습니다. 존재는
  Flows가 평가할 수 있는 property가 아니기 때문입니다.
- **graph를 그리기.** edge는 handler body 안의 `return` statement입니다. static 추출은 당신의 source를
  파싱해야 할 것입니다.
- **reachability나 termination을 assert하기.** 분석할 구조가 없습니다.

boson과 대조하십시오, [[boson-stage-machine]]에서: `load_stages(config, prompts)`가
`agents/*/stage_config.py`와 `stages/*.md`로부터 이름을 key로 하는 `dict[str, StageDefinition]`를
만들고, `StageMachine.transition(from_stage, to_stage)`는 target이 현재 stage의 `transitions` list에
없을 때 `TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")`를 반환합니다.
graph는 읽고, 출력하고, pull request에서 diff하고, test할 수 있는 자료구조입니다.

그 두 property — registry와 edge check — 중 어느 것도 `flows/types.py`나 `flows/manager.py`에 대응물이
없습니다.

### 10.3 이것이 구체적으로 의미하는 것

trade를 양방향으로 진술하십시오. 진짜로 양방향이기 때문입니다.

**Flows가 주는 어려운 것:** system instruction, message list, tool array의 거의-atomic한 교체를
올바르게 순서 지어, in-flight function call에 대해 올바르게 sequence하고(§8.5), inference trigger와
pre/post-action ordering까지 처리해서. 그것은 진짜 일이고 제대로 하기 성가신 부분입니다.

**Flows가 전혀 주지 않는 것:** transition legality, node 등록, graph introspection, 그리고 node가
자기 후계자를 안다는 개념 일체.

그래서 진지한 port의 모양은 "stage machine을 `FlowManager`로 교체"가 아닙니다. 그것은 **당신의
validator를 유지하고, prompt-and-tool plumbing을 버리는 것**입니다. `StageMachine.transition()`은
`set_node_from_config` 앞의 순수한 pre-check로 남고, `build_stage_injection`과
`StageContext.filter_tools`가 Flows가 대체하는 것입니다. §13이 그 매핑을 field 단위로 훑습니다.

---

## 11. `examples/flows/insurance_quote.py` — 진짜 graph를 읽으십시오

380줄, 다섯 개의 node factory, 그리고 봇에게 자기 나이와 결혼 여부를 소리 내어 말해야 하는 고객.
이것은 Pipecat tree에서 당신이 만들고 있는 것에 가장 가까운 것이므로, 데모가 아니라 원재료로
읽으십시오.

### 11.1 graph

다섯 factory, 네 개의 tool edge와 하나의 self-loop:

```
create_initial_node()
   └─ collect_age(age) ──────────────────────────────▶ marital_status

create_marital_status_node()
   └─ collect_marital_status(marital_status) ────────▶ quote_calculation

create_quote_calculation_node(age, marital_status)
   └─ calculate_quote(age, marital_status) ──────────▶ quote_results

create_quote_results_node(quote)
   ├─ update_coverage(coverage_amount, deductible) ──▶ quote_results   (self-loop)
   └─ end_quote() ───────────────────────────────────▶ end

create_end_node()
   post_actions=[{"type": "end_conversation"}], no functions
```

factory들의 arity에 주목하십시오. 그것이 design이기 때문입니다. `create_initial_node()`와
`create_marital_status_node()`는 아무것도 받지 않습니다.
`create_quote_calculation_node(age, marital_status)`는 둘을 받습니다. `create_quote_results_node(quote)`는
dict을 받습니다. **node factory의 parameter는 그 node가 자기 prompt에 구워 넣어야 하는 데이터입니다.**

### 11.2 initial node가 persona를 한 번 설정한다

**`examples/flows/insurance_quote.py:211-237`**
```python
def create_initial_node() -> NodeConfig:
    """Create the initial node asking for age."""
    return NodeConfig(
        name="initial",
        role_message="You are a friendly insurance agent. Your responses will be converted to audio, so avoid special characters. Always use the available functions to progress the conversation naturally.",
        task_messages=[
            {
                "role": "developer",
                "content": "Start by asking for the customer's age.",
            }
        ],
        functions=[collect_age],
    )


def create_marital_status_node() -> NodeConfig:
    """Create node for collecting marital status."""
    return NodeConfig(
        name="marital_status",
        task_messages=[
            {
                "role": "developer",
                "content": "Ask about the customer's marital status for premium calculation.",
            }
        ],
        functions=[collect_marital_status],
    )
```

persona는 node 1 위의 string 하나이고 **파일의 다른 어디에도 나타나지 않습니다**. §4.4의 Rule 1이
그것이 작동하는 이유입니다: `role_message`는 이후 모든 transition을 가로질러 persist하는 system
instruction이 됩니다.

persona string에 대한 세 가지는 prompt 스타일이 아니라 voice-agent 규율이고, 셋 다 한국어에
이전됩니다:

- *"Your responses will be converted to audio, so avoid special characters."* — 모델은 그 외에는
  자기가 말하고 있다는 것을 알지 못합니다.
- *"Always use the available functions to progress the conversation naturally."* — 광고된 function이
  하나뿐일 때, 이것은 모델에게 그 function이 *곧* 앞으로 가는 길이라고 말하는 방법입니다.
- 무엇을 해야 하는지에 대한 나머지 전부는 여기가 아니라 `task_messages`에 삽니다.

그리고 `create_marital_status_node`는 모든 비-initial node의 pattern입니다: 이름 하나, 명령형 developer
message 하나, function 하나. 그게 다입니다. 열한 줄.

### 11.3 results node — 파일에서 가장 시사적인 블록

**`examples/flows/insurance_quote.py:258-281`**
```python
def create_quote_results_node(
    quote: QuoteCalculationResult | CoverageUpdateResult,
) -> NodeConfig:
    """Create node for showing quote and adjustment options."""
    return NodeConfig(
        name="quote_results",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Quote details:\n"
                    f"Monthly Premium: ${quote['monthly_premium']:.2f}\n"
                    f"Coverage Amount: ${quote['coverage_amount']:,}\n"
                    f"Deductible: ${quote['deductible']:,}\n\n"
                    "Explain these quote details to the customer. When they request changes, "
                    "use update_coverage to recalculate their quote. Explain how their "
                    "changes affected the premium and compare it to their previous quote. "
                    "Ask if they'd like to make any other adjustments or if they're ready "
                    "to end the quote process."
                ),
            }
        ],
        functions=[update_coverage, end_quote],
    )
```

*([[flows-insurance-example]] excerpt는 이 블록을 `L262-281`로 인용하지만, `def`는 실제로 **258**에서
시작합니다. source가 이깁니다.)*

네 가지, 그리고 첫 번째가 훔쳐 갈 것입니다.

**`${quote['monthly_premium']:.2f}` — 숫자는 모델이 아니라 Python에서 형식화됩니다.** 보험료는 float로
도착합니다: `rates["base_rate"] * rates["risk_multiplier"]` = `150 * 1.5` = `225.0`. 그대로
interpolate하면 모델은 `225.0`을 받고 "이백이십오 점 영"이라고 말할 수 있습니다. `:.2f`가 그것을
`225.00`으로 만들고 `:,`가 coverage를 `250,000`으로 만듭니다. **voice agent가 말하는 어떤 숫자든
prompt 안에서 미리 형식화되어야 합니다.** 한국어에서는 이것이 덜이 아니라 더 시급합니다: 월 보험료
`45,000원` 대 `45000.0`, 그리고 모델이 당신이 측정할 수도 없고 고객이 들은 뒤에는 고칠 수도 없는
어떤 비율로 틀리게 될 만/억 단위 묶음.

**self-loop는 그저 자기가 호출된 node를 반환하는 함수입니다.** `update_coverage`는
`create_quote_results_node(result)`를 반환합니다 — 같은 `name`과 다른 prompt를 가진 *새*
`NodeConfig`. loop 구문 같은 것은 없습니다. node에 다시 들어간다는 것은 새 데이터로 factory를 다시
호출하는 것이고, 매 재진입은 `_set_node` batch 전체를 다시 돌립니다: 같은 tool이 다시 광고되고, 새
`task_messages`가 append됩니다(기본 `APPEND`이므로 *이전* 견적의 숫자들이 여전히 context 안에
있습니다 — 그것이 애초에 *"compare it to their previous quote"*를 가능하게 만드는 것입니다).

**`functions=[update_coverage, end_quote]`가 rail guard입니다.** 이 node에는 정확히 두 개의 합법적인
수가 있습니다. 고객은 다른 어디로도 route될 수 없습니다. 다른 무엇도 광고되지 않았기 때문입니다 —
§10.1이 legality check가 없음을 보여 주었지만, *model-driven* 경로에는 그런 것이 있을 필요가
없습니다. node의 function list가 모델의 메뉴 전부이기 때문입니다. legality 공백은 오직 out-of-band
경로에서만 열립니다(§12.3, [[ch-12/read]]).

**terminal node는 `post_actions`를 지니고 function은 없습니다.**

**`examples/flows/insurance_quote.py:284-298`**
```python
def create_end_node() -> NodeConfig:
    """Create the final node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Thank the customer for their time and end the conversation. "
                    "Mention that a representative will contact them about the quote."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )
```

`functions` key가 아예 없습니다. Rule 3에 따라 이것은 **tool 집합을 clear합니다** — terminal node에는
정확히 옳은 일이고, 무조건적인 `LLMSetToolsFrame`이 hazard가 아니라 feature인 유일한 경우입니다.
모델은 작별 인사를 하는 동안 물리적으로 아무것도 호출할 수 없습니다.

### 11.4 데이터를 앞으로 나르는 세 channel

handler들:

**`examples/flows/insurance_quote.py:114-146`**
```python
async def collect_age(
    flow_manager: FlowManager, age: int
) -> tuple[AgeCollectionResult, NodeConfig]:
    """Record customer's age.

    Args:
        age (int): The customer's age.
    """
    logger.debug(f"collect_age handler executing with age: {age}")

    flow_manager.state["age"] = age
    result = AgeCollectionResult(age=age)

    next_node = create_marital_status_node()

    return result, next_node


async def collect_marital_status(
    flow_manager: FlowManager, marital_status: str
) -> tuple[MaritalStatusResult, NodeConfig]:
    """Record marital status after customer provides it.

    Args:
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    logger.debug(f"collect_marital_status handler executing with status: {marital_status}")

    result = MaritalStatusResult(marital_status=marital_status)

    next_node = create_quote_calculation_node(flow_manager.state["age"], marital_status)

    return result, next_node
```

세 channel, 세 가지 다른 일에 쓰임:

| channel | LLM에 보이나? | `ContextStrategy.RESET`을 살아남나? | 하는 일 |
|---|---|---|---|
| 함수의 **반환값** (tuple 원소 0) | **예** — tool result로 전달 | message이므로 **아니오** | 호출이 성공했고 무엇을 만들었는지 모델에게 알림 |
| **`flow_manager.state`** | **아니오** — manager 위의 평범한 dict | **예** — 결코 context 안에 없음 | application의 데이터를 보관 |
| `task_messages` 안의 **f-string** | **예** — 그것이 *곧* 다음 prompt | 새 node의 message이므로, 남는 것이 그것 | 다음 발화를 조종 |

세 번째가 사람들이 덜 쓰는 것입니다. **봇이 다음에 무엇을 말하는지**를 신뢰성 있게 제어하는 유일한
channel은 다음 node의 `task_messages`에 interpolate하는 텍스트입니다. tool result는 모델이 기댈 수도
있고 아닐 수도 있는 message이고, `task_messages` directive는 node의 목표입니다.

docstring convention에도 주목하십시오. `"""Record customer's age.\n\n    Args:\n        age (int): The
customer's age.\n    """` — 그 docstring이 **곧** JSON Schema입니다. 요약 줄이 tool description이 되고,
`Args:` 블록이 `properties`가 됩니다. 이는 docstring이 모델이 읽는 routing hint이기도 하다는 뜻이고,
`patient_intake.py`는 그것을 의도적으로 활용합니다: *"Record the user's prescriptions. **Once
confirmed, the next step is to collect allergy information.**"*(`patient_intake.py:142`). 다음 단계가
현재 단계의 tool description 안에서 모델에게 문서화되어 있습니다.

### 11.5 왕복: `state → prompt → model → args`

값 하나를 끝까지 따라가십시오. 그것이 이 pattern의 유일한 진짜 약점이고 pattern을 복사하기 전에 봐야
하기 때문입니다.

1. `collect_age`가 `flow_manager.state["age"] = age`(`:124`)를 씁니다. 이것이 파일의 **유일한**
   `state` write입니다.
2. `collect_marital_status`가 다음 node를 구성하기 위해 그것을 되읽습니다:
   `create_quote_calculation_node(flow_manager.state["age"], marital_status)`(`:144`).
3. 그 factory가 그것을 prompt에 interpolate합니다:

**`examples/flows/insurance_quote.py:240-255`**
```python
def create_quote_calculation_node(age: int, marital_status: str) -> NodeConfig:
    """Create node for calculating initial quote."""
    return NodeConfig(
        name="quote_calculation",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Calculate a quote for {age} year old {marital_status} customer. "
                    "First, call calculate_quote with their information. "
                    "Then explain the quote details and ask if they'd like to adjust coverage."
                ),
            }
        ],
        functions=[calculate_quote],
    )
```

4. 그리고 `calculate_quote(flow_manager, age, marital_status)`가 그것들을 **LLM이 공급한 argument로**
   받습니다 — 모델이 prompt에서 `age`를 읽어서 tool call에 도로 넣습니다.

**`examples/flows/insurance_quote.py:149-158`**
```python
async def calculate_quote(
    flow_manager: FlowManager, age: int, marital_status: str
) -> tuple[QuoteCalculationResult, NodeConfig]:
    """Calculate initial insurance quote.

    Args:
        age (int): The customer's age.
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    logger.debug(f"calculate_quote handler executing with age: {age}, status: {marital_status}")
```

그래서 application이 이미 Python dict 안에 가지고 있던 값이 여행합니다: **`state["age"] → 다음 node의
prompt → 모델 → 다시 function argument로.**

작동합니다. 그리고 그것은 값을 language model을 통과시키는데, 그것은 아무도 요청하지 않은 세탁
단계입니다. `36`은 `36`으로 돌아올 수도, `35`로 돌아올 수도, prompt 표현이 흔들리면 `"mid-thirties"`로
돌아올 수도 있습니다. prompt와 argument 사이에는 JSON type을 넘는 schema validation이 없습니다.

**Lina를 위한 규칙, 평이하게 진술하면:** 모델이 바꿔 말하면 안 되는 무엇이든 — 주민등록번호, 증권
코드, 계좌번호, 동의 timestamp, `saedam_355_m35_plan_default` 같은 상품 ID — 는 **handler 안에서
`flow_manager.state`로부터 읽혀야** 하고, 결코 function parameter로 선언되어서는 안 됩니다. 고객이
이번 turn에 실제로 말하고 있는 것만 선언하십시오.

```python
# WRONG for identifiers — the model relays the value
async def save_payment_info(
    flow_manager: FlowManager, resident_number: str, account_number: str
) -> tuple[dict, NodeConfig]: ...

# RIGHT — the model triggers the action, the handler reads the data
async def save_payment_info(
    flow_manager: FlowManager, confirmed: bool
) -> tuple[dict, NodeConfig]:
    """Save the customer's payment information once they have confirmed.

    Args:
        confirmed (bool): Whether the customer confirmed the details read back to them.
    """
    rrn = flow_manager.state["resident_number"]        # never left Python
    account = flow_manager.state["account_number"]
    ...
```

이것은 정확성만이 아니라 compliance 모양이기도 합니다. `task_messages`에 결코 들어가지 않고 tool
argument가 결코 되지 않는 주민등록번호는 LLM provider의 request body에 결코 들어가지 않습니다.

> 💡 **쉬운 설명 — "왜 굳이 모델을 거치게 만들었나요?"**
> 예제가 그렇게 한 이유는 편의입니다: `calculate_quote`의 schema에 `age`가 있으면 모델이 알아서 채워
> 주니까 handler 쪽에서 state key 이름을 신경 쓸 필요가 없습니다. 하지만 그 편의의 대가는 **값이 한 번
> 자연어를 통과한다**는 것입니다. 데모에서는 무해하고, 금융/보험에서는 무해하지 않습니다. 판단 기준은
> 간단합니다: 그 값이 틀리면 고객에게 실질적 피해가 가는가? 그렇다면 parameter가 아니라 state입니다.

### 11.6 이전 가능한 decomposition pattern

파일에서 추출해 recipe로 진술하면:

1. **node 하나 = 추출하거나 달성하는 것 하나.** `insurance_quote.py`는 두 parameter짜리 function 하나를
   가진 node 하나로도 작동함에도 나이와 결혼 여부를 두 node로 나눕니다. 그 분할이 각 `task_messages`를
   하나의 모호하지 않은 지시로 만드는 것입니다.
2. **`task_messages`는 명령형이고 모델을 향한다**, `role: "developer"`, message 하나.
   *"Ask about the customer's marital status for premium calculation."* node에 대한 기술이 아니라
   명령입니다.
3. **`functions`가 rail guard다.** node의 function list가 모델의 완전한 메뉴입니다. 전화를 끊는 것이
   잘못될 node에서는 `end_conversation` 모양의 function을 빼십시오.
4. **`role_message`는 한 번, initial node에.** 그것은 persist합니다.
5. ***다음* prompt가 필요로 하는 데이터 → factory의 parameter로 f-string.** *application*이 필요로
   하는 데이터 → `flow_manager.state`. *모델*이 인지해야 하는 데이터 → 반환값.
6. **모든 숫자를 Python에서 미리 형식화하십시오.**
7. **Terminal node: `post_actions=[{"type": "end_conversation"}]`, `functions` 없음.**
8. **항상 `name`을 설정하십시오.** (§5.2 — 이것은 예제 자신의 규율에는 없습니다. `hello_world.py`가
   node를 자기 factory 이름으로 부르기 때문입니다. 그래도 하십시오.)

Lina의 stage 하나 위에 shape check로 매핑하면 — 네 개의 한국어 동의 질문(개인정보 수집·이용,
신용정보집중기관 조회, 주민번호 처리, 건강정보 처리)을 가진 `informed_consent`:

- 질문당 하나씩 네 개의 node, 각각 다음으로 edge하는 `record_consent` function 하나를 가지거나;
- 또는 `(result, create_consent_node())`를 반환하는 `record_consent(consent_item: str, agreed: bool)`
  function을 가진 node 하나 — `update_coverage` 같은 self-loop — 로, `flow_manager.state`가 네 답을
  누적하고 factory가 매번 *"이미 동의하신 항목: …. 다음 항목: …"*을 `task_messages`에 다시
  렌더링합니다.

두 번째가 `patient_intake.py`의 모양이고 첫 번째가 `insurance_quote.py`의 모양입니다. 어느 쪽이
옳은지는 고객이 넷을 한 번에 답하는 것("네 네 다 동의합니다")이 tool call 하나여야 하는지 넷이어야
하는지에 달렸고, 그것은 이 chapter가 당신을 대신해 답할 수 없는 product 질문입니다.

---

## 12. 예제 셋, pattern 셋

### 12.1 `patient_intake.py` — LLM은 확인을 받고, app은 payload를 받는다

**`examples/flows/patient_intake.py:142-155`**
```python
async def record_prescriptions(
    flow_manager: FlowManager, prescriptions: list[dict]
) -> tuple[PrescriptionRecordResult, NodeConfig]:
    """Record the user's prescriptions. Once confirmed, the next step is to collect allergy information.

    Args:
        prescriptions (list[dict]): List of prescription objects, each with "medication" (str, the medication's name) and "dosage" (str, the prescription's dosage).
    """
    # Store prescriptions in flow state
    flow_manager.state["prescriptions"] = prescriptions

    # In a real app, this would store in patient records
    return PrescriptionRecordResult(count=len(prescriptions)), create_allergies_node()
```

복사할 가치가 있는 두 가지.

**docstring만으로 전부 기술된 `list[dict]` parameter.** `FlowsFunctionSchema`도, 명시적 JSON Schema도
없습니다 — `Args:` 줄이 object shape을 산문으로 풀어 쓰고 schema 추출이 그것을 `properties`로
바꿉니다. 이것이 direct-function 경로를 떠나지 않고 "구조화된 항목 여럿을 한 turn에 어떻게
수집하나"에 대한 답입니다.

**반환값은 payload가 아니라 `count`입니다.** `PrescriptionRecordResult(count=len(prescriptions))`가
모델에게 가고, 실제 list는 `flow_manager.state`로 갑니다. 모델은 *"3"*을 받고 "세 가지 약을
기록했습니다"라고 말할 수 있습니다. 그것은 약 이름을 결코 되읽지 않으므로, 하나라도 바꿔 말할 기회를
결코 얻지 못합니다. 그것이 §11.5의 규칙을 *argument* channel이 아니라 *return* channel에 적용한
것이고, 의료나 금융 도메인에서 그것은 올바른 기본값입니다.

`patient_intake.py`에는 Flows가 사소하게 값싸게 만드는 두 개의 backward edge도 있습니다:
`revise_information`은 `create_prescriptions_node()`(더 이른 node로의 loop back)를 반환하고,
`confirm_information`은 `create_confirmation_node()`를 반환합니다. node가 그저 function이 반환한
`NodeConfig`이므로, 뒤로 가는 것은 앞으로 가는 것과 정확히 같은 비용입니다 — 모든 backward edge가
추가하는 것을 기억해야 하는 행인 edge-whitelist design에 비하면 진짜 이점입니다.

### 12.2 `restaurant_reservation.py` — 결과 분기, 그리고 outbound/inbound flag

**`examples/flows/restaurant_reservation.py:150-166`**
```python
        party_size (int): Number of people in the party.
    """
    # Check availability with mock API
    is_available, alternative_times = await reservation_system.check_availability(party_size, time)

    # Result: availability status and alternative times, if any
    result = TimeResult(
        status="success", time=time, available=is_available, alternative_times=alternative_times
    )

    # Next node: confirmation or no availability
    if is_available:
        next_node = create_confirmation_node()
    else:
        next_node = create_no_availability_node(alternative_times)

    return result, next_node
```

**분기가 Python의 `if` statement입니다.** edge 위의 condition도, config 파일 안의 guard 표현식도 아닌 —
`if`. 이것이 edges-as-return-values의 가장 큰 표현력 이점입니다: routing 논리가 API 응답, database,
시각, 그리고 `flow_manager.state`에 접근할 수 있는 평범한 코드입니다. 어떤 DSL도 그것과 경쟁할 수
없습니다.

그리고 실패 분기는 자기 데이터를 close over합니다:

**`examples/flows/restaurant_reservation.py:227-243`**
```python
    times_list = ", ".join(alternative_times)
    return NodeConfig(
        name="no_availability",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Apologize that the requested time is not available. "
                    f"Suggest these alternative times: {times_list}. "
                    "Ask if they'd like to try one of these times. If they pick a time, check "
                    "its availability. If they'd rather not book after all, call the "
                    "end_conversation function to wrap up."
                ),
            }
        ],
        functions=[check_availability, end_conversation],
    )
```

f-string 이전의 `", ".join(alternative_times)` — §11.3의 미리-형식화 규칙을 list에 적용한 것입니다.

그리고 당신에게 가장 중요한 flag:

**`examples/flows/restaurant_reservation.py:174-188`**
```python
# Node configurations
def create_initial_node(wait_for_user: bool) -> NodeConfig:
    """Create initial node for party size collection."""
    return NodeConfig(
        name="initial",
        role_message="You are a restaurant reservation assistant for La Maison, an upscale French restaurant. Be casual and friendly. This is a voice conversation, so avoid special characters and emojis.",
        task_messages=[
            {
                "role": "developer",
                "content": "Warmly greet the customer and ask how many people are in their party. This is your only job for now; if the customer asks for something else, politely remind them you can't do it.",
            }
        ],
        functions=[collect_party_size],
        respond_immediately=not wait_for_user,
    )
```

`respond_immediately=not wait_for_user`. **flag 하나가 outbound call(봇이 먼저 말함)과 inbound
call(봇이 기다림) 사이를 전환합니다.** node에 대한 다른 모든 것은 동일합니다.

Lina는 outbound tele-sales agent이므로 `respond_immediately=True`가 그녀의 introduction node에 옳습니다 —
전화를 걸고, 고객이 받고, 말합니다. 하지만 §13.7이 rule-driven design에서 *기본값*이 `True`인 것이
다른 *모든* node에 hazard임을 보여 줄 것입니다. 고객의 turn에 발화된 transition이 그녀가 고객 위에
말하게 만들기 때문입니다.

산문으로 된 rail-guard 문장에도 주목하십시오: *"This is your only job for now; if the customer asks for
something else, politely remind them you can't do it."* `collect_party_size`만 광고된 상태에서 모델은
다른 수가 없습니다 — 하지만 여전히 헛소리를 늘어놓을 수는 있습니다. prompt가 tool list가 남긴 틈을
메웁니다. 이탈 방지가 중요한 한국어 tele-sales 스크립트에서는, 이 문장을 모든 수집 node에 쓰십시오.

### 12.3 `warm_transfer.py` — escalation recipe, 그리고 tool call이 아닌 transition

715줄, 그리고 repository에서 tele-sales product를 위한 ship 가능한 feature에 가장 가까운 것. 그
다섯-node graph는 header 주석에 문서화되어 있습니다:

**`examples/flows/warm_transfer.py:82-92`**
```python
# Flow nodes:
#
# 1. initial_customer_interaction
#    The initial node, where the bot interacts with the customer and tries to help with their requests.
#    Functions:
#    - check_store_location_and_hours_of_operation (always succeeds)
#    - start_order (always fails)
#    - end_customer_conversation
#    Transitions to either:
#    - continued_customer_interaction
#    - transferring_to_human_agent
```

가져갈 mechanism 셋.

**(a) escalation은 모든 task function의 실패 분기다.**

**`examples/flows/warm_transfer.py:249-255`**
```python
# Helpers
def next_node_after_customer_task(result: Mapping[str, Any]) -> NodeConfig:
    """Transition to either the "continued_customer_interaction" node or "transferring_to_human_agent" node, depending on the outcome of the previous customer task"""
    if result.get("status") == "success":
        return create_continued_customer_interaction_node()
    else:
        return create_transferring_to_human_agent_node()
```

평범한 helper이고, 모든 task handler가 호출해서 각 handler가 분기를 반복하지 않고도 모든 실패가
escalation으로 route되게 합니다. Lina에게 이것은 명시적 intent rule에서만이 아니라 tool이 실패하는
어디에서든 도달되는 `escalate_to_human`입니다.

**(b) node는 function이 전혀 없는 순수 side effect일 수 있다.**

**`examples/flows/warm_transfer.py:321-342`**
```python
def create_transferring_to_human_agent_node() -> NodeConfig:
    """Create the "transferring_to_human_agent" node.
    This is the node where the customer is asked to please hold while the bot transfers them to a human agent. Hold music plays while the customer waits.
    """
    return NodeConfig(
        name="transferring_to_human_agent",
        task_messages=[
            {
                "role": "developer",
                "content": "Start by apologizing to the customer that there was an issue fulfilling their last request, then inform them that they are being transferred to a human agent. Tell them to please hold while you connect them, and thank them for their patience.",
            }
        ],
        pre_actions=[
            ActionConfig(type="function", handler=mute_customer),
        ],
        post_actions=[
            ActionConfig(type="function", handler=start_hold_music),
            ActionConfig(type="function", handler=make_customer_hear_only_hold_music),
            ActionConfig(type="function", handler=print_human_agent_join_url),
        ],
    )
```

네 개의 action, 전부 `type="function"` — framework 자신의 가장 정교한 예제가 §9.3의 지침을 따르고
있습니다. 그리고 ordering이 정확히 `function` action이 호출하는 대신 queue하는 이유입니다:
`mute_customer`는 node의 message가 설치되기 전에 실행되고(§4.1), 세 개의 post-action은 "잠시만
기다려 주세요" 줄이 말해진 *뒤* pipeline tail에서 실행됩니다(§9.2). hold music을 한 frame 일찍
시작하면 그것이 사과 위로 재생됩니다.

**(c) tool call이 아닌 transition.** human agent가 Daily room에 합류하는 것은 모델이 function을 호출할
수 있는 종류의 일이 아니므로, transition은 transport event handler에서 push됩니다 —
`warm_transfer.py:657`, §5.3에서 인용했고, 이것을 호출합니다:

**`examples/flows/warm_transfer.py:258-261`**
```python
# Transitions
async def start_human_agent_interaction(flow_manager: FlowManager):
    """Transition to the "human_agent_interaction" node."""
    await flow_manager.set_node_from_config(create_human_agent_interaction_node())
```

**`set_node_from_config`는 PUBLIC**이고(`manager.py:588`) 어떤 coroutine에서도 호출 가능합니다:

**`src/pipecat/flows/manager.py:588-600`**
```python
    async def set_node_from_config(self, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Used to manually transition between nodes in a flow.

        Args:
            node_config: Configuration for the new node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        await self._set_node(get_or_generate_node_name(node_config), node_config)
```

body 한 줄, 곧장 `_set_node`로. **Flows는 transition을 모델을 통하도록 강제하지 않습니다.**
`NodeConfig`의 producer는 정확히 둘 — handler가 그것을 반환하는 LLM function call, 그리고 이 method를
호출하는 임의의 coroutine — 이고, 두 번째는 in-tree에서 두 번 증명되었습니다
(transport callback에서의 `warm_transfer.py:261`, `@worker.event_handler("on_activated")`에서의
`multi_worker_handoff.py:352`).

그 사실이 [[ch-12/read]]의 경첩이고, 여기서는 **mechanism으로만** 진술됩니다. 그것이 boson의 rule
layer에게 무엇을 가능하게 하는지 — processor가 어디에 서는지, queue race가 무엇을 비용으로 하는지,
layer가 붕괴하는지 — 는 이 chapter의 소관이 아닙니다.

briefing node는 handoff에서의 context reset의 의도된 사용법도 보여 줍니다:

**`examples/flows/warm_transfer.py:360-365`**
```python
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY,
            summary_prompt=(
                "Summarize the conversation with the customer, including what they were trying to accomplish and what, if anything, went wrong while trying to fulfill their requests. Include specific error details."
            ),
        ),
```

human agent는 transcript가 아니라 summary를 받습니다. 그 *의도*는 Lina의 `escalate_to_human`에 정확히
옳습니다. 그 *mechanism*은 §7.2의 deprecated된 것이므로, `ContextStrategy.RESET` 더하기
`LLMSummarizeContextFrame`을 push하는 pre-action으로 지으십시오.

---

## 13. boson의 stage machine 위에 매핑하기

boson은 이미 이 graph를 가지고 있습니다. 그것은 선언적이고, 한 파일 안에 있고, node가 아홉 개입니다.

### 13.1 이미 있는 것

[[boson-stage-machine]]에서: `agents/test-lina-gateway/stage_config.py`가
`initial_stage = "introduction"`과 아홉 개의 등록된 stage — `introduction`, `product_focused`,
`escalate_to_human`, `consultation`, `purchase`, `reschedule`, `dnc_processing`, `informed_consent`,
`end` — 를 선언하고, 각각은 세 field를 가진 dict입니다:

```python
@dataclass
class StageDefinition:
    name: str; prompt: str = ""; tools: list[str] = []; skills: list[str] = []; transitions: list[str] = []
```

그리고 whitelist, [[boson-stage-machine]]에 따라 verbatim:

| stage | 허용된 후계자 | stage tools |
|---|---|---|
| `introduction` (initial) | product_focused, purchase, dnc_processing, reschedule, escalate_to_human, end | — |
| `product_focused` | consultation, purchase, reschedule, dnc_processing, escalate_to_human, end | check_product_detail, check_product_summary, lookup_faq |
| `consultation` | purchase, informed_consent, reschedule, dnc_processing, escalate_to_human, end | + check_available_products; skill `product_manager` |
| `informed_consent` | consultation, end, reschedule, dnc_processing, escalate_to_human | record_consent, get_consent_status |
| `purchase` | end, escalate_to_human | agreement_record, agreement_status, check_product_detail, verify_personal_info, save_payment_info, save_address; skill `payment_manager` |
| `reschedule` | consultation, end, escalate_to_human | reschedule |
| `dnc_processing` | end, escalate_to_human | register_dnc, check_dnc_status |
| `escalate_to_human` | end | escalate_to_human |
| `end` | *(없음 — terminal)* | — |

이제 세 field를 매핑하십시오.

### 13.2 `tools` → `NodeConfig["functions"]` — 직접

`stages[X]["tools"]`는 tool 이름의 list이고, `NodeConfig["functions"]`는 direct function이나
`FlowsFunctionSchema`의 list입니다. port는 기계적입니다: 이름 list가 import list가 됩니다.

`_GLOBAL_TOOLS`(현재 `[]`)는 `FlowManager(global_functions=[...])`에 매핑되고, Flows는 그것을 모든
node에서 섞어 넣습니다 — §4.2에서 인용한 `manager.py:654`:

```python
            functions_list = self._global_functions + node_config.get("functions", [])
```

앞에 붙이고, 모든 node에서, 무조건적으로. 정확히 `_GLOBAL_TOOLS`가 가진 semantics입니다.

boson의 `@tool` handler는 `FlowsDirectFunction`에 매핑됩니다 — docstring이 이미 description을 지니고
있고, 그것은 두 시스템이 쓰는 같은 convention입니다. [[boson-tool-router]]가 지적하는 두 contract
mismatch는 진짜이지만 그것들은 이 chapter가 아니라 [[ch-09/read]] §9.4의 주제입니다: Pipecat handler는
`FunctionCallParams` 하나를 받고 `await params.result_callback(result)`로 정산하는 반면 boson의 것은
값을 반환하는 `handler(**arguments)`입니다. Flows는 그 위에 앉아 있으므로 mismatch를 더하지 않고
물려받습니다 — Flows direct function은
`async def f(flow_manager, **params) -> tuple[Any, NodeConfig | None]`이고, 그것은 *세 번째* shape입니다.

### 13.3 `transitions` → **데이터이기를 그만둔다**

이것이 구조적 재작성이고, 무엇을 얻고 무엇을 잃는지 정확할 가치가 있습니다.

`stages[X]["transitions"]`는 stage별 allowlist입니다. `StageMachine.transition()`이 그것을 확인하고
`TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")`를 반환합니다. Flows에는
**어떤 수준에도 대응물이 없습니다** — `types.py`에도(edge type도 node type도 없음, §10.2),
`_validate_node_config`에도(§10.1), `set_node_from_config`에도(§12.3).

Flows에서 합법적인 후계자는 각 function이 어떤 `NodeConfig`를 반환할 수 있는지에 암묵적으로 있습니다.
그것들은 handler body에 흩어져 있습니다. 그것들을 읽을 단 하나의 장소는 없습니다.

**무엇을 얻는가.** 버그 한 부류 전체가 구조적으로 불가능해집니다. `stage_config.py`가 자기 주석에서
그것을 두 번 문서화합니다 — `transition_detector.py:157`이 `StageTransition("purchase")`를 emit했는데
whitelist가 `"purchase"`를 빠뜨려서 stage machine이 그것을 거부한 `v0.7.5 (#12)` note. Flows에서
`create_purchase_node()`를 반환하는 handler는 purchase node로 transition합니다. 끝. 동기화 상태로
유지할 두 번째 list가 없으므로 어긋날 것이 없습니다.

그리고 [[boson-stage-machine]]의 `core._apply_stage_transition` 읽기에 따라, 그 버그가 오늘 어떻게
실패하는지 보십시오:

```python
result = self._stage_machine.transition(from_stage=session.active_stage, to_stage=target)
if not result.success:
    return                      # ← silent no-op on a rejected edge
session.active_stage = target
self._inject_stage(session, result.new_stage)
```

실패 시 `return`. `error` string은 구성되고 버려집니다. rule이 발화하고, transition이 거부되고,
그것이 일어났다는 것을 어디에서도 아무것도 기록하지 않습니다. 그 failure mode는 누군가
`TransitionResult.error`를 확인하지 않는 한 보이지 않고, 그것을 확인할 code path는 존재하지 않습니다.

**무엇을 잃는가.** 단일 파일의, 읽을 수 있고, diff할 수 있는 transition table. 오늘은 reviewer가
`stage_config.py`를 열어서 `purchase`가 `end`나 `escalate_to_human`으로만 갈 수 있다는 것을 한 화면에
볼 수 있습니다 — 그것은 보험 영업 통화의 compliance 관련 property이지 개발자 편의가 아닙니다. 순진한
port 이후에는 그 property가 어디에도 진술되지 않습니다. 그것은 purchase node의 handler가 우연히
호출하는 factory들의 창발적 귀결입니다.

**둘 다 지키는 모양.** `StageMachine`을 `set_node_from_config` 앞의 **순수 validator**로 유지하고,
그것의 prompt와 tool plumbing만 버리십시오:

```python
NODES: dict[str, Callable[[], NodeConfig]] = {
    "introduction": create_introduction_node,
    "product_focused": create_product_focused_node,
    # ... nine entries
}

async def go(flow_manager: FlowManager, target: str) -> None:
    result = stage_machine.transition(from_stage=flow_manager.current_node, to_stage=target)
    if not result.success:
        logger.warning(result.error)          # ← the line that is missing today
        return
    await flow_manager.set_node_from_config(NODES[target]())
```

그 dict이 check를 넘어 사 주는 두 가지. 그것은 §10.2가 Flows에 없다고 말하는 **node registry**이므로
`has_stage(name)`과 initial-stage validation이 돌아옵니다. 그리고 `NodeConfig["name"]`은 boson stage
이름으로 설정되어야 합니다. 그렇지 않으면 — 이제 legality check의 좌변이 된 —
`flow_manager.current_node`가 UUID를 담고 check는 쓰레기를 비교합니다(§5.2).

이 chapter는 거기까지입니다. `go()`에 대한 호출이 어디서 오는지는 [[ch-12/read]]입니다.

### 13.4 `skills` → 없음

`stages[X]["skills"]`(`consultation`의 `product_manager`, `purchase`의 `payment_manager`)는
**Flows 개념이 전혀 없습니다.** `NodeConfig`에 `skills` key가 없고, `types.py`에 아무것도 없고,
package 어디에도 second-tier tool 개념이 없습니다.

Flows가 도와주지 않는 두 선택지: 각 skill을 그것이 노출하는 function들로 평탄화하거나, skill을
`use_skill` direct function 뒤의 boson meta-tool로 유지하는 것([[boson-tool-router]]). 두 번째는
2-tier 구조를 보존하지만 node의 `functions` list가 모든 node에서 `[use_tool, use_skill]`이라는 뜻이고 —
그것은 §13.5와 나쁘게 상호작용합니다.

역시 대응물이 없는 것: boson의 `<system-reminder>` protocol,
`ContextManager.pop_pending_reminders()`, 그리고 `session/history.py`로부터의 turn별
`<system-reminder>Active stage: {session.active_stage}</system-reminder>` 재천명. Flows는 현재 node를
모델에게 매 turn 다시 진술하지 않습니다. node의 `task_messages`는 transition 시점에 한 번 append되고
대화가 자라면서 history 속으로 물러납니다. 긴 `consultation` stage에서는 그것이 의미 있는 behaviour
차이이고, `APPEND`는 그것을 낫게 하는 게 아니라 악화시킵니다: stage prompt는 매 turn 더 멀어집니다.

### 13.5 충돌: 무조건적 `LLMSetToolsFrame` 대 byte-stable한 tool array

이것이 두 design 사이의 가장 날카로운 충돌이고, 두 사실을 나란히 놓기 전에는 나타나지 않습니다.

**사실 하나, §4.4 Rule 3에서.** `LLMSetToolsFrame(tools=functions)`는 `manager.py:839`에서 **모든**
node transition마다, 무조건적으로, 광고 집합 전체와 함께 emit됩니다.

**사실 둘, [[boson-tool-router]]에서.** boson은 prompt caching을 보존하기 위해 광고되는 tool array를
**stage를 가로질러 byte-stable하게** 의도적으로 유지합니다. 모델에게는 `use_tool`과 `use_skill`만
보이고, stage별 접근 제어는 그 아래에서 dispatch 시점에 ContextVar allowlist를 통해 일어납니다:

> *"Do not collapse 'the model can see it' into 'the model may call it' — boson deliberately keeps
> the advertised array byte-stable across stages to preserve prompt caching while the allowlist
> changes underneath."*
>
> ('모델이 그것을 볼 수 있다'를 '모델이 그것을 호출해도 된다'로 뭉개지 마라 — boson은 allowlist가
> 그 아래에서 바뀌는 동안 prompt caching을 보존하기 위해 광고되는 array를 stage를 가로질러
> byte-stable하게 의도적으로 유지한다.)

세 개의 gate가 구별되어 유지됩니다: **exposure**(모델이 보는 것), **availability**(stage가 허용하는
것), **permission**(애초에 실행되어도 되는 것).

Flows에는 gate가 **하나**뿐이고, 그것은 exposure입니다. `NodeConfig["functions"]`는 광고되는 array
*이면서* availability list이고 *또* — permission hook이 없으므로 — permission model입니다. §5.4가
별도의 check를 걸어 둘 field가 없음을 보여 주었습니다.

그래서 순진한 port는 골라야 합니다:

**Option A — node의 `functions` list를 문자 그대로 포팅.** 각 node가 자기 stage tool을 광고합니다.
이제 tool array가 매 transition마다 바뀌고, 그것이 prompt prefix를 바꾸고, 그것이 매 stage 경계에서
provider 쪽 prompt cache를 무효화합니다. 네다섯 stage를 걷는 통화라면 그것은 system prompt 전체를
포함하는 prefix에 대한 네다섯 번의 cache miss입니다. [[ch-11/read]]가 가격을 매기게 해 줄 것입니다.
여기서의 요점은 그것이 *boson이 현재 지불하지 않는 비용*이고, 끄는 스위치가 없는 mechanism에 의해
도입된다는 것입니다.

**Option B — 두 항목짜리 meta-tool array를 유지.** 모든 node에서 `functions=[use_tool, use_skill]`,
어디서나 동일하게, 그리고 ContextVar allowlist를 availability gate로 유지. array는 byte-stable하므로
cache는 살아남습니다 — 하지만 이제 `NodeConfig["functions"]`는 아홉 node 전부에서 같고 아무 정보도
싣지 않으며, §11.3의 node별 rail guard가 사라집니다. `insurance_quote.py`의 design을 안전하게 만든
것 — 모델의 메뉴가 *곧* node의 합법적인 수 — 은 존재하지 않습니다. 메뉴가 항상 같은 두 항목이기
때문입니다.

**Option C — 두 gate를 다 유지.** node별 function을 광고하고(Option A의 array) *또* dispatch 시점의
allowlist를 두 번째 check로 유지. cache 비용을 지불하고 defence in depth를 유지합니다.

boson의 caching behaviour와 Flows의 node별 rail guard를 둘 다 얻는 Option D는 없습니다. 그 둘은 같은
knob을 반대 방향으로 돌린 것이기 때문입니다. **하나를, 의도적으로 고르고, 어느 것인지 적어
두십시오.** [[ch-13/read]]가 그것을 채점합니다. 이 chapter의 일은 그 선택이 우발적이 아니라 보이게
만드는 것입니다.

> 💡 **쉬운 설명 — prompt cache가 왜 tool array에 걸리나요?**
> provider의 prompt cache는 request의 **앞부분이 byte 단위로 같을 때만** 적중합니다. tool 정의는 보통
> system prompt 바로 다음(또는 그 안)에 직렬화되므로, tool 목록이 한 글자라도 바뀌면 그 뒤 전부가
> cache miss입니다. 즉 "node마다 다른 tool을 광고한다"는 선택은 **stage 경계마다 전체 prefix를 다시
> 계산한다**는 뜻입니다. 이것이 §13.5가 취향 문제가 아니라 비용 문제인 이유입니다.

### 13.6 `ContextStrategy.RESET`이 유일하게 명확한 이득이다

§7.3이 그것을 진술했습니다. 여기 boson 쪽을 붙여서 다시:

boson의 `_inject_stage`는 `f"[Stage: {stage.name}]\n\n{stage.prompt}"`를
`<system-reminder>`로 감싼 message로 append합니다([[boson-stage-machine]], `stage/context.py:73`).
append만 합니다. stage layer에 reset 경로가 없습니다.

귀결: `purchase`에 도달한 Lina 통화는 자기 context에 introduction prompt, product_focused prompt,
consultation prompt, informed_consent prompt, purchase prompt를 — 그 사이의 모든 대화 turn과 함께 —
그 시점 이후 매 단일 inference마다 담고 있습니다. prompt는 통화 길이만큼 단조 증가하고, *가장 이른*
지시가 모델이 가장 오래 읽어 온 것입니다.

`purchase` node 위의 `ContextStrategyConfig(strategy=ContextStrategy.RESET)`는 message list를 그 node의
`task_messages`로 대체합니다. persona는 살아남습니다(system instruction이므로, §6.1). tool 집합은
살아남습니다(별개의 frame, 별개의 field). `flow_manager.state`는 살아남습니다(결코 context 안에 없음).
사라지는 것은 대화입니다.

그것은 공짜 승리가 아니라 진짜 trade입니다: `purchase`는 `consultation`에서 어떤 상품이 논의되었는지
알아야 합니다. 그것이 `flow_manager.state`와 f-string channel이 있는 이유입니다 — §11.4의 표가
`RESET`을 쓸 수 있게 만드는 design pattern이고, `patient_intake.py:244`가 작동하는 예제입니다.

`RESET_WITH_SUMMARY` 위에 짓지 **마십시오**(§7.2 — deprecated, 5초 blocking cap, 조용한 fallback,
config mutation). summary가 필요하면 `RESET` 더하기 `LLMSummarizeContextFrame`을 push하는 pre-action을
쓰십시오.

### 13.7 rule 기반 transition에는 `respond_immediately=False`가 필수다

기본값은 `True`입니다(`manager.py:707`). 따라서 모든 `_set_node` 호출이 `LLMRunFrame`을 queue하고
봇이 말합니다.

boson의 stage transition은 **agent loop가 돌기 전에 수행되는 조용한 bookkeeping**입니다
([[boson-stage-machine]]): rule이 `StageTransition(target)`을 emit하고, `_apply_stage_transition`이
`session.active_stage`를 옮기고 prompt를 주입하고, *그러고 나서* loop가 한 번 돌아 응답 하나를
만듭니다. transition은 발화를 만들어 내지 않습니다. 그것은 다음 발화가 만들어지는 조건을 바꿉니다.

그것을 순진하게 포팅하면 모든 rule 기반 transition은 — 고객이 아직 말하고 있는 중에 발화하는
것들까지 포함해서 — `LLMRunFrame`을 queue하고 Lina는 고객이 일어난 줄도 모르는 stage 변화에 응답하려고
고객을 끊습니다.

그러므로: **rule 기반 port의 모든 node는 `respond_immediately=False`를 지닌다**, 다만 outbound call의
introduction node는 예외이고 그것이 §12.2의 `wait_for_user=False` 경우입니다.

그리고 그 flag가 §4.4에 따라 무엇을 더 바꾸는지 주목하십시오: post-action이 inline 실행을 멈추고
`_ongoing_actions_count == 0`인 다음 `BotStoppedSpeakingFrame`으로 deferred됩니다(§2.2, §9.6).
`respond_immediately=False`를 설정하면서 post-action도 지닌 node는 자기 side effect가 *봇의* speech에
의해 결정되는 시점에 발화하고, 그것은 당신이 기대하는 것보다 여러 turn 뒤일 수 있습니다. 포팅한 모든
node에서 그 조합을 확인하십시오.

### 13.8 무엇이 포팅되지 않는가, 그리고 이 chapter가 설계하기를 거부하는 것

**포팅되지 않음:** `stages[X]["skills"]`(§13.4). `<system-reminder>` protocol. turn별 active-stage
재천명. `schemas/actions.py`의 8-verb vocabulary에서 온 `Continue()`, `Pass()`, `Filter(reason)` —
그것들은 pre-LLM routing verdict이고, Flows action이 실행될 때쯤이면 message는 이미 context 안에
있습니다(§9.4).

**깔끔하게 포팅됨:** `Respond(text)` → `{"type": "tts_say", ...}`. `PreTool(...)` →
`pre_actions=[{"type": "function", "handler": fn}]`, 앞선 `tts_say`가 뒤따르는 `function`에 대해
직렬화되지 **않는다**는 §9.6의 단서와 함께. Terminal stage(`end`, `dnc_processing`) →
`post_actions=[{"type": "end_conversation"}]`와 function 없음. `escalate_to_human` →
`warm_transfer.py` recipe 전체(§12.3).

**그리고 여기가 이 chapter가 멈추는 곳입니다.** 당신은 이제 `set_node_from_config`가 public이고, 임의의
coroutine으로부터 `NodeConfig`를 받아들이고, Flows가 legality check를 부과하지 않으며, `StageMachine`을
그 앞에 순수 validator로 유지할 수 있다는 것을 압니다. 그것은 mechanism 하나와 constraint 집합입니다.

그것을 design으로 바꾸는 것 — rule processor가 pipeline 어디에 서는지, inference가 시작되기 전에 그
position에 완전한 user turn이 존재하는지, boson의 cross-layer veto가 인접한 processor들에 흩뿌려지고도
살아남는지, 그리고 in-pipeline processor와 `FlowManager`의 head 주입 frame 사이의 queue race가 무엇을
비용으로 하는지 — 는 [[ch-12/read]]의 일이고, 그것은 무엇이든 가격을 매기기 전에 [[ch-11/read]]의
millisecond budget을 필요로 합니다. 여기서 스케치하지 마십시오.

---

## 14. 머릿속에 담아 둘 것

| # | Fact | Where |
|---|---|---|
| 1 | `FlowManager`는 평범한 class이고, `FrameProcessor`가 아니며, `Pipeline` list 안에 없다 | `manager.py:80`; `hello_world.py:135-145` |
| 2 | 그것은 worker *이후에* 구성되고 worker를 요구한다; pipe를 바깥에서 구동한다 | `manager.py:121-134`; `hello_world.py:147-167` |
| 3 | touch point 둘: `queue_frames` out(`:709`, `:841`), 필터링된 downstream event 하나 in | `manager.py:709, 841`; `actions.py:109-127` |
| 4 | `set_reached_downstream_filter`는 **대체**한다; Flows가 유일한 caller; 지우면 action이 조용히 죽는다 | `worker.py:695-701`; `actions.py:109` |
| 5 | flow frame은 **head**로 들어가 전부를 통과한다; `LLMRunFrame`은 user aggregator에서 소비된다 | `worker.py:793-808`; `llm_response_universal.py:814, 1173` |
| 6 | batch: `LLMUpdateSettingsFrame`? → `Append`/`Update` → `SetTools`(항상) → 그다음 별도로 `LLMRun`? | `manager.py:762-841, 709` |
| 7 | `functions`가 없는 node는 tool 집합을 **clear**한다 (`NOT_GIVEN`) | `manager.py:670-672, 839` |
| 8 | batch는 atomic이 아니다 — `queue_frames` 호출 둘, 평범한 loop, ordering 보장 없음 | `worker.py:810-829` |
| 9 | `LLMUpdateSettingsFrame`만 `UninterruptibleFrame`이다; barge-in이 나머지를 버릴 수 있다 | `frames.py:2251` vs `634, 645, 661, 694` |
| 10 | `_current_node`는 `str`이고 결코 `NodeConfig`가 아니다; 이름 없는 node는 새 UUID를 받는다 | `manager.py:149, 703`; `types.py:518` |
| 11 | `_current_functions`: `src/`에 read 0회, `tests/test_flows_manager.py`에 6회 | grep, §0.4 |
| 12 | `FlowConfig`는 존재하지 않는다; static/dynamic 구분 없음; 구조는 runtime에 결정 | grep = 0; `__init__.py:12-13` |
| 13 | `NodeConfig`는 `TypedDict(total=False)`; `task_messages`가 유일한 `Required` key; 오타는 조용하다 | `types.py:224-237` |
| 14 | `ContextStrategyConfig`는 `@dataclass`이고 `__post_init__`에서 validate한다 — `NodeConfig`와의 비대칭 | `types.py:155-179` |
| 15 | edge는 `ConsolidatedFunctionResult`의 두 번째 원소다: node / `None` / `NO_RESPONSE`(identity로) | `types.py:266, 247`; `manager.py:496` |
| 16 | `ContextStrategy`는 `Update` 대 `Append`로 매핑되며 잊어버리는 유일한 knob이다 | `manager.py:831-836` |
| 17 | `RESET_WITH_SUMMARY`: deprecated, 5.0초 blocking cap, 조용한 fallback, config를 mutate | `manager.py:799-822` |
| 18 | `FlowResult`는 **"No replacement."**로 deprecated; export된 이름 여섯이 2.0.0에서 사라진다 | `types.py:40-53`; §6.7 |
| 19 | handler dispatch는 `len(sig.parameters)`; built-in은 bound method라 legacy branch를 발동시킨다 | `manager.py:409-441`; `actions.py:180-188` |
| 20 | built-in action은 정확히 **셋**: `tts_say`, `end_conversation`, `function` | `actions.py:104-106` |
| 21 | maintainer는 custom action보다 `function`을 선호하여 de-emphasize한다 | `actions.py:292-294` |
| 22 | action은 node를 고르거나, transition을 veto하거나, 값을 반환할 수 없다; raise하는 **pre**-action은 중단시킨다 | `actions.py:220-224`; `manager.py:647, 693` |
| 23 | `_validate_node_config`는 두 가지를 검사한다; from→to check는 **어디에도** 없다 | `manager.py:867-898` |
| 24 | node registry가 없다 — node는 구성되지 등록되지 않는다; graph는 열거 불가능하다 | §10.2 |
| 25 | `set_node_from_config`는 public이다: transition은 모델에서 올 필요가 **없다** | `manager.py:588`; `warm_transfer.py:261` |
| 26 | 데이터 전달: 반환값(모델이 봄), `state`(`RESET`을 살아남음), `task_messages` 안의 f-string(조종) | `insurance_quote.py:124, 144, 248` |
| 27 | `state → prompt → model → args` 왕복은 실재한다; 식별자는 handler 안에서 `state`로부터 읽으라 | `insurance_quote.py:144-158`; §11.5 |
| 28 | 모든 숫자를 Python에서 미리 형식화하라; 모델이 보험료를 렌더링해서는 절대 안 된다 | `insurance_quote.py:269-271` |
| 29 | `respond_immediately=not wait_for_user`가 outbound/inbound 스위치다 | `restaurant_reservation.py:187` |
| 30 | boson 매핑: `tools`→`functions` 직접; `transitions`→**데이터이기를 그만둠**; `skills`→없음 | §13.2–13.4 |

---

## 다음 챕터로

이 chapter는 네 가지를 앞으로 넘깁니다.

**해소된 architecture.** `FlowManager`는 pipeline 옆의 평범한 object입니다. 그것은 두 줄에서
`queue_frames`로 쓰고 필터링된 event 하나로 읽습니다. 그 node state는 string입니다. 그 graph는 Python
함수들의 집합이고 그 edge는 tuple 원소입니다. 그것은 성격 규정이 아닙니다. 그것은 `manager.py:80`,
`hello_world.py:135-167`, `manager.py:149`, `types.py:266`입니다.

**외울 frame batch.** frame 넷, 규칙 넷: `LLMUpdateSettingsFrame`은 conditional하고 persistent,
`Append` 대 `Update`가 **곧** `ContextStrategy`, `LLMSetToolsFrame`은 **무조건적**이라 빈 node가 tool을
clear함, `LLMRunFrame`은 conditional하고 별도로 queue됨. Flows에서 벌어지는 모든 운영상의 놀라움은
그 넷 중 하나가 발화한 것입니다.

**정확히 이름 붙은 세 개의 부재.** transition legality 없음. node registry 없음. exposure와 구별되는
permission gate 없음 — 그리고 그것일 수도 있어 보이는 field인 `_current_functions`는 `src/`에 read가
0회이고 test suite에 6회입니다. boson이 유지하는 어떤 guard layer든, Flows가 그것을 가지고 있지 않기
때문에 유지하는 것입니다.

**이후 chapter를 위한 두 개의 열린 constraint.** batch는 pipeline의 **head**로 들어가 LLM에 도달하기
전에 모든 upstream processor를 통과해야 합니다 — rule 평가를 위해 당신이 끼워 넣는 어떤 processor든
포함해서. 그리고 그 네 frame 중 오직 하나만 barge-in을 살아남으므로, 고객의 turn 중에 발화된
transition은 error 없이, 그리고 `FlowManager`가 알아챌 방법 없이 부분적으로 유실될 수 있습니다.

다음은 [[ch-11/read]] — **latency budget과 observer plane**입니다. 그것이 rule design보다 먼저 오는
것은 의도적입니다. 이 chapter의 두 숫자가 이미 분모를 기다리고 있습니다: `RESET_WITH_SUMMARY`의
하드코딩된 5.0초 cap, 그리고 §13.5의 무조건적 tool 재광고의 prompt-cache 비용. 어느 쪽도 측정할 budget이
있기 전에는 싸다고도 비싸다고도 논증될 수 없고, observer plane이 budget을 알 수 있게 만드는
계측기입니다 — frame graph 위의 read-only한 두 번째 plane이며, 그것은 정확히 pipeline 바깥의
component가 필요로 하는 그 non-adjacency property입니다.

그다음 [[ch-12/read]]가 §12.3의 public `set_node_from_config`를 가져다 design으로 바꿉니다. §4.5의
비원자성, §4.6의 interruptibility 분할, §3.4의 head-injection trace를 함께 가져가십시오 — 그 셋이
rule seam이 그것에 맞서 지어져야 하는 constraint이고, [[ch-12/read]]는 어떤 답을 보여 주기 전에
그것들을 derivation으로 당신에게 되돌려 줄 것입니다.
