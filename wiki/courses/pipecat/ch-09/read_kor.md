---
title: "두 개의 Agent Loop: context, tool, turn의 끝을 누가 소유하는가"
chapter: ch-09
phase: collision
course: pipecat
lang: ko
companion_of: read.md
sources:
  - llm-service-context
  - function-calling
  - boson-agent-loop
  - boson-tool-router
  - rtv-pipeline-session
deps:
  - ch-03
  - ch-08
figure: figures/two-loops.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# ch-09 — 두 개의 Agent Loop: context, tool, turn의 끝을 누가 소유하는가

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 (frame, processor, pipeline, queue, aggregator, context, registry,
> dispatch, turn, interruption, back-pressure, timestamp, catch-all, gate 등).

> **범위(scope). 미리 명시하고, 이 chapter 전체에 걸쳐 강제합니다.**
>
> **하나의 아이디어: loop ownership.** 세 개의 질문, 오직 셋뿐입니다 — 누가 message list를 들고 있는가,
> 누가 tool을 dispatch하는가, 누가 turn이 끝났다고 결정하는가. 아래의 모든 것은 이 셋 중 하나에
> 복무합니다.
>
> **Context compaction은 이 chapter에 없습니다.** `LLMContextSummarizer`
> (`src/pipecat/processors/aggregators/llm_context_summarizer.py:57`) 대 boson의 `gateway/compact/`는
> 서로 다른 failure mode와 서로 다른 trigger를 가진 별개의 subsystem이고, [[ch-13/read]]의 give-back
> list에 올라가 있습니다. 만약 스스로 *언제 history가 짧아지는가*를 추론하고 있다면, 당신은 이
> chapter를 벗어난 것입니다. *애초에 누가 history를 건드릴 자격이 있는가*로 돌아오십시오.
>
> **비교 판정(comparative verdict)은 없습니다.** 이 chapter는 각 design이 **무엇을 하는지**와 각 선택이
> **무엇을 대가로 치르는지**만 말합니다. 어느 쪽이 더 낫다, 어느 쪽이 이긴다, 무엇을 채택해야 한다,
> 무엇이 "옳은 선택"이다 — 이런 말은 하지 않습니다. 세 개의 resolution이 §11에서 전개되고 가격이
> 매겨지지만 **그중 어느 것도 recommended로 표시되지 않습니다**. [[ch-13/read]]가 이 course에서 무언가에
> 점수를 매기는 유일한 곳이고, 그것은 열두 개 subsystem을 모두 본 뒤에야 가능하기 때문에 허용됩니다.
> 이 invariant는 이 outline이 review되는 동안 두 번 깨졌습니다. 여기서는 깨지지 않습니다.

---

## 왜 이 챕터인가

[[ch-03/read]]는 §7.6을 의도적으로 끝맺지 않은 문장으로 마쳤습니다:

> There is no Pipecat analogue of a "text-in / text-out agent slot that is forbidden to know about
> tools." Whether the two contracts can coexist, and what it would take, is [[ch-09/read]]'s entire
> subject. Do not resolve it here.
>
> ("tool에 대해 아는 것이 금지된 text-in / text-out agent slot"에 해당하는 Pipecat의 대응물은 없다.
> 두 contract가 공존할 수 있는지, 그리고 그러려면 무엇이 필요한지가 [[ch-09/read]]의 주제 전부다.
> 여기서 해결하지 말 것.)

이것이 그 chapter이고, collision은 저 문장이 들리게 한 것보다 더 깊습니다. Pipecat에 text-only agent
slot이 없다는 얘기가 아닙니다. **Pipecat과 boson-agent는 "the agent runs a turn"이라는 문장의 주어가
누구인지에 대해 서로 의견이 다르다**는 것입니다.

boson에서 주어는 함수입니다. `run_agent_loop(runtime, user_input)`은 561줄이고, message list, provider
call, tool dispatch, turn counter, hook, 그리고 cancellation 이후의 history 복구 — 여섯 가지 전부를
하나의 lexical scope 안에서 소유합니다. turn 전체를 위에서 아래로 읽을 수 있습니다.

Pipecat에는 주어가 없습니다. turn은 **순환(circulation)**입니다: 평범한 object 하나가 message를 들고,
LLM 양쪽의 processor 두 개가 그것을 mutate하고, LLM service는 그것을 읽고 그것 외에는 아무것도 읽지
않으며, loop가 닫히는 이유는 pipeline의 *끝*에 있는 processor가 frame을 *가운데*에 있는 processor로
**거꾸로** push하기 때문입니다. 아무도 loop를 돌리고 있지 않습니다. loop는 graph의 모양(shape)입니다.

그 차이는 이 chapter 내내 측정하게 될 세 개의 결과를 낳고, 그중 어느 것도 Lina TMR에서 공짜가 아닙니다:

1. Pipecat의 `get_messages()`는 **live list**를 그대로 건네줍니다. 의도적입니다 — assistant aggregator가
   tool-result dict를 **in place**로 다시 쓰기 때문입니다. boson의 `ContextManager.get_messages()`는
   `deepcopy(self._messages)`를 반환합니다. 이것도 의도적이고, 정확히 그런 일을 막기 위해서입니다.
   한 줄짜리 "방어적" 포팅이 Pipecat의 tool path를 조용히 망가뜨립니다 — exception도, log도 없이,
   그냥 tool 결과가 model에 영영 도달하지 않습니다.
2. Pipecat에는 **turn cap이 없습니다.** 큰 값이 있는 게 아니라 — 아예 없습니다.
   `grep -rn "max_turns\|max_iterations\|max_tool_calls" src/pipecat/`는 hit이 0입니다. boson의
   `max_turns=50`과 그 `while/else` 소진(exhaustion) 경로는 얹힐 자리 자체가 없습니다.
3. boson은 tool에 대해 **세 개의 분리된 gate**를 유지합니다 — exposure, availability, permission. 그리고
   Pipecat의 registry는 `dict[str | None, FunctionCallRegistryItem]`이며 그 셋 중 어느 것을 담을 slot도
   없습니다.

셋 중 둘은 Pipecat이 *가지고 있지 않은* 것입니다. 이것은 비판이 아닙니다 — framework는 policy를 갖기를
거절할 수 있습니다. 이것은 inventory 항목이고, 그 inventory에 가격을 매기는 것이 [[ch-13/read]]입니다.

---

## 0. 이 chapter에서 evidence를 읽는 법

[[ch-03/read]] §0와 같은 two-class rule입니다. 이 course의 어느 곳보다 여기서 더 중요합니다. 왜냐하면
이 chapter가 바로, 이전 draft의 course outline이 **서로 다른 네 번**에 걸쳐 틀렸던 곳이고, 그 오류
하나하나가 전부 파일을 여는 대신 summary를 믿은 데서 나왔기 때문입니다.

| Class | Source of truth | 검증 방법 |
|---|---|---|
| **Pipecat claim** — path, 줄 번호, class 이름, grep, LOC | commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`의 `wiki/raw-data/pipecat/pipecat-src` | 파일을 여십시오. 아래의 모든 줄 번호는 2026-08-25에 그 tree에 대해 다시 읽었습니다. excerpt와 source가 어긋난 곳에서는 source가 이겼고, 그렇다고 밝혔습니다. |
| **boson-agent / realtime_voice claim** | `wiki/raw-data/pipecat/excerpts/` 아래의 `boson-*` 및 `rtv-*` excerpt, private repo에서 읽은 것 | 당신의 repo에 대고 확인하십시오. 이 wiki에서는 검증 불가능하고, 여기서도 그런 척하지 않습니다. 모든 boson code block은 repo path **와** excerpt wikilink를 함께 달고 있습니다. |

boson-agent excerpt는 commit `0a5e8ced2fd9e631dcbbe8c5f4adb68b89a4fafb` (2026-08-20)에서 나왔습니다.
realtime_voice는 branch `voice-chat-dev`, commit `034ce4ca09a2f109e6c248a43bc989f8d26a6abf`
(2026-07-29)입니다. 둘 중 어느 쪽이든 움직였다면, 아래의 shape는 그때의 모습입니다.

### 0.1 이 chapter의 plan이 틀렸던 네 가지 사실

이걸 먼저 읽으십시오. 사소한 잡지식이 아닙니다 — 둘은 전체 논증을 떠받치고, 하나는 *오직 줄 번호
하나 덕분에만* 성립하는 논증입니다.

**(a) `create_context_aggregator()`는 존재하지 않습니다.** 약 1년보다 오래된 Pipecat tutorial은 전부
`context_aggregator = llm.create_context_aggregator(context)`로 시작합니다. 이 commit에서 그 method는
사라졌습니다:

```console
$ cd wiki/raw-data/pipecat/pipecat-src
$ grep -rn "create_context_aggregator" src/ examples/ | wc -l
0
```

tree 전체에서 유일한 hit은 `CHANGELOG.md`에 있으며, 그 제거를 문서화합니다:

```markdown
# CHANGELOG.md:4278-4286
- ⚠️ Removed deprecated service-specific context and aggregator machinery,
  which was superseded by the universal `LLMContext` system.

  Service-specific classes removed: `AnthropicLLMContext`,
  `AnthropicContextAggregatorPair`, `AWSBedrockLLMContext`,
  `AWSBedrockContextAggregatorPair`, `OpenAIContextAggregatorPair`, and their
  user/assistant aggregators. Also removed `create_context_aggregator()` from
  `LLMService`, `OpenAILLMService`, `AnthropicLLMService`, and
  `AWSBedrockLLMService`.
```

대체물은 `LLMContext(messages, tools)` + `LLMContextAggregatorPair(context)`입니다. 옛 형태를 보여주는
tutorial은 어떤 것도 실행되지 않습니다. 이건 타이핑 문제를 넘어섭니다: 이 제거는 context가 *service가
제조하는 것*이기를 그만두고 **application이 소유해서 건네주는 것**이 되었다는 *바로 그* 신호입니다.
그것이 §2의 전부입니다.

**(b) `register_direct_function`은 deprecated입니다.** 여전히 존재하지만:

```python
# src/pipecat/services/llm_service.py:982-998
    @deprecated(
        "`LLMService.register_direct_function` is deprecated since 1.4.0 and will be removed in "
        "2.0.0. Use `LLMContext(tools=[...])` instead."
    )
    def register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool | None = None,
        timeout_secs: float | None = None,
    ):
        """Register a direct function handler for LLM function calls.

        .. deprecated:: 1.4.0
            Direct functions are now registered automatically. List them in
            ``LLMContext(tools=[...])`` for tools available at session start, or
            push an :class:`LLMSetToolsFrame` to change tools mid-session.
            Will be removed in 2.0.0.
```

(a)와 같은 신호를, 이번엔 tool 차원에서: registration이 service에서 떨어져 나와 context로 옮겨갔습니다.

**(c) Pipecat에는 tool loop도 없고 turn cap도 없습니다.** 둘 다 부재(absence)이고, 둘 다 grep으로
검증되며, 각각 §5와 §6에 있습니다.

**(d) completion은 `base_llm.py:605`에서 시작합니다. `:604`가 아닙니다.** 이전 draft의 outline은
`:604`를 인용하고 그 위에 핵심 주장을 세웠습니다. 그건 틀렸고, 이 수정은 미용상의 문제가 아닙니다 —
연속된 네 줄이 네 가지 서로 다른 일을 하고, "service는 frame당 정확히 한 번의 inference를 수행한 뒤
반환한다"는 논증은 *inference가 어느 줄에 있는가*에 관한 논증이기 때문입니다:

```python
# src/pipecat/services/openai/base_llm.py:599-605
        await super().process_frame(frame, direction)          # :599  mandatory, non-negotiable

        if isinstance(frame, LLMContextFrame):                 # :601  the only trigger
            try:
                await self.push_frame(LLMFullResponseStartFrame())   # :603  bracket opens
                await self.start_processing_metrics()                # :604  metrics clock starts
                await self._process_context(frame.context)           # :605  ← the completion
```

`:601`은 `isinstance` 검사입니다. `:603`은 여는 bracket frame을 push합니다. `:604`는 metrics clock을
시작합니다. `:605`가 token이 생성되는 곳입니다. 다음 문단을 읽기 전에 직접 확인하십시오:

```console
$ awk 'NR>=599 && NR<=605 {printf "%d\t%s\n", NR, $0}' src/pipecat/services/openai/base_llm.py
```

이걸 강하게 짚는 이유는, 이것이 그 outline이 저지른 모든 오류의 모양이기 때문입니다: 어떤 summary가
"service는 start frame을 push하고 context를 처리한다"고 말했고, 읽는 사람이 그것을 줄 번호 하나로
압축했고, 그 압축된 버전 위에 논증이 세워졌습니다.

> 💡 **쉬운 설명 — 왜 `:604`와 `:605`의 차이가 논증을 바꾸나요?**
> "service는 frame 하나당 inference 하나만 한다"를 증명하려면 *inference가 일어나는 지점*을 정확히
> 가리키고, 그 지점 뒤에 `while`이나 재귀 호출이 없다는 것을 보여야 합니다. `:604`(metrics clock)를
> 가리키면 "clock을 시작한 뒤 반환한다"가 되어 논증이 성립하지 않습니다. 줄 번호가 논증의 *전제*
> 자체입니다. 인용된 줄 번호는 장식이 아니라 증명의 일부라는 습관을 여기서 들이십시오.

---

## 1. 세 개의 ownership 질문, 그리고 테이블 위에 놓인 세 개의 답

코드에 들어가기 전에 질문을 고정하십시오. 이 chapter의 모든 것은 이 셋 중 정확히 하나에 대한 답입니다.

| | **Q1. 누가 message list를 들고 있는가?** | **Q2. 누가 tool을 dispatch하는가?** | **Q3. 누가 turn이 끝났다고 결정하는가?** |
|---|---|---|---|
| **Pipecat** | `LLMContext`. *application*이 생성하는 평범한 object (`llm_context.py:83`). aggregator 두 개가 그것을 in place로 mutate한다. LLM service는 아무것도 들고 있지 않다. | `LLMService.run_function_calls()` → `_run_function_call()` (`llm_service.py:1437`, `:1547`). registry는 context가 채운다. | 아무도 아니다. `LLMContextFrame`이 upstream으로 push되지 않으면 turn이 끝난다 — 결정이 아니라 부재(absence). |
| **boson-agent** | `ContextManager`. `runtime.context_manager`로 도달하고, 오직 `run_agent_loop` 내부에서만 mutate된다 (`agent_loop.py:184`). | `ToolRouter.dispatch()` (`metatool/router.py:103`), `_execute_tool_uses`를 거쳐서 (`agent_loop.py:393`). | `run_agent_loop` 자신이, `break  # Done — text response means end of turn` (`agent_loop.py:363`)에서. |
| **realtime_voice** | voice package 안에는 아무것도 없다. `StreamingConversationAgent`는 message list를 본 적이 없다. | voice package 안에는 아무것도 없다. tool은 `GatewayConversationAgent` 뒤에 있다. | agent slot이, 자신의 `AsyncIterator[AgentTextDelta]`를 끝냄으로써. |

세 번째 행을 주의 깊게 읽으십시오. realtime_voice의 세 질문 모두에 대한 답은 **"내가 아니다"**입니다.
그건 누락이 아니라 contract입니다. [[rtv-vs-pipecat-gap]]에 인용된 boson 자신의 `CLAUDE.md`에서 나왔고
[[ch-03/read]] §7.6에 재수록되어 있습니다: *"Keep Basement and the dental business logic text-native."*
voice package는 tool이 무엇인지 아는 것이 금지되어 있습니다.

그 세 번째 행이 이 chapter가 가설이 아닌 이유입니다. §9가 거기로 돌아옵니다.

---

## 2. Pipecat, Q1: message list는 application이 건네주는 평범한 object다

### 2.1 `LLMContext`는 field 세 개다

열어 보십시오. 510줄이고, state는 attribute 세 개입니다:

```python
# src/pipecat/processors/aggregators/llm_context.py:83, 91-112
class LLMContext:
    """Manages conversation context for LLM interactions.
    ...
    """

    def __init__(
        self,
        messages: list[LLMContextMessage] | None = None,
        tools: ToolsSchema | list[FunctionSchema | DirectFunction] | NotGiven = NOT_GIVEN,
        tool_choice: LLMContextToolChoice | NotGiven = NOT_GIVEN,
    ):
        ...
        self._messages: list[LLMContextMessage] = messages if messages else []
        self._tools: ToolsSchema | NotGiven = LLMContext._normalize_and_validate_tools(tools)
        self._tool_choice: LLMContextToolChoice | NotGiven = tool_choice
```

거기 **없는 것**에 주목하십시오. provider 없음. client 없음. model 이름 없음. system prompt field 없음.
turn counter 없음. lock 없음. `asyncio` 관련 아무것도 없음. `LLMContext`는 method가 달린 값 세 개짜리
자루이고, `FrameProcessor`가 **아닙니다** — `Pipeline` list에 등장하는 일이 결코 없고, list에 등장하는
것들에게 *전달되는* 대상입니다.

§8에서 message schema를 비교할 때 중요해질 디테일 하나: `LLMContextMessage`는 union이고, 그 "standard"
가지는 OpenAI의 type입니다 — 단, type checker가 볼 때만 그렇습니다:

```python
# src/pipecat/processors/aggregators/llm_context.py:50-65
# The aliases resolve under type checking only. Every LLM service reaches this
# module, so importing the OpenAI SDK here would put its load on the startup
# path of pipelines that never talk to OpenAI. At runtime both are structurally
# dicts or strings, which is all the annotations need them to be.
if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolChoiceOptionParam,
    )

    LLMStandardMessage: TypeAlias = ChatCompletionMessageParam
    LLMContextToolChoice: TypeAlias = ChatCompletionToolChoiceOptionParam
else:
    LLMStandardMessage: TypeAlias = Any
    LLMContextToolChoice: TypeAlias = Any
```

그러므로 runtime에서 Pipecat message는 OpenAI shape의 **평범한 dict**이고, validation은 전혀 없습니다.
`Message` class도, pydantic model도, `ContentBlock`도 없습니다. 이 생각을 §8.4까지 붙들고 계십시오 —
boson의 `basement.schemas.message_schema` layer 전체가 여기에는 착륙 지점이 없습니다.

### 2.2 mutator들, 그리고 그중 in place로 mutate하는 하나

```python
# src/pipecat/processors/aggregators/llm_context.py:361-394
    def add_message(self, message: LLMContextMessage):
        """Add a single message to the context.
        ...
        """
        self._messages.append(message)

    def add_messages(self, messages: list[LLMContextMessage]):
        """Add multiple messages to the context.
        ...
        """
        self._messages.extend(messages)

    def set_messages(self, messages: list[LLMContextMessage]):
        """Replace all messages in the context.
        ...
        """
        self._messages[:] = messages

    def transform_messages(
        self, transform: Callable[[list[LLMContextMessage]], list[LLMContextMessage]]
    ):
        """Transform the current messages using the provided function.
        ...
        """
        self.set_messages(transform(self._messages))
```

`set_messages`는 `self._messages = messages`가 아니라 `self._messages[:] = messages`입니다. 이것은
slice assignment입니다: attribute를 rebind하는 대신 기존 list object를 mutate합니다. 같은 list에 대한
reference를 들고 있는 다른 누구든 — 그리고 실행 중인 pipeline에는 그런 것이 여럿 있습니다 — 그 교체를
봅니다. 만약 rebind였다면, `ctx.get_messages()`를 cache해 둔 processor는 영원히 옛 list만 쳐다보게
됩니다.

이것이 이 chapter 전체를 관통하는 design 원칙의 첫 목격입니다: **Pipecat은 하나의 object에 여러 holder가
있는 상황에 최적화합니다.** boson은 holder 하나에 최적화하고 방어적으로 복사해서 내보냅니다.

> 💡 **쉬운 설명 — slice assignment가 왜 다른가**
> ```python
> a = [1, 2, 3]
> b = a          # b와 a는 같은 list object
> a = [9]        # rebind: b는 여전히 [1, 2, 3]
>
> a = [1, 2, 3]
> b = a
> a[:] = [9]     # slice assignment: b도 [9]가 된다
> ```
> Pipecat은 두 번째 쪽입니다. 그래서 aggregator, LLM service, 그리고 당신의 processor가 각자 들고 있는
> reference가 전부 같은 최신 상태를 봅니다. "누가 이 list를 들고 있는가"를 추적할 필요가 없게 만든
> 대가로, "누가 이 list를 쓰고 있는가"를 추적해야 하는 세계입니다.

### 2.3 `get_messages()`는 live list를 반환하고, 그것은 의도적이다

주의 깊은 engineer라면 "고치고" 싶어질 method가 여기 있습니다:

```python
# src/pipecat/processors/aggregators/llm_context.py:221-258
    def get_messages(
        self,
        llm_specific_filter: str | None = None,
        *,
        truncate_large_values: bool = False,
    ) -> list[LLMContextMessage]:
        """Get the current messages list.
        ...
        """
        if llm_specific_filter is None:
            messages = self._messages
        else:
            messages = [
                msg
                for msg in self._messages
                if not isinstance(msg, LLMSpecificMessage) or msg.llm == llm_specific_filter
            ]
            if len(messages) < len(self._messages):
                logger.error(
                    f"Attempted to use incompatible LLMSpecificMessages with LLM '{llm_specific_filter}'."
                )

        if truncate_large_values:
            messages = LLMContext._truncate_large_values_from_messages(messages)

        return messages
```

세 개의 경로가 있고, 그중 복사하는 것은 하나뿐입니다:

| 호출 | 무엇이 돌아오는가 |
|---|---|
| `get_messages()` | `self._messages` — **live list object 그 자체**. 그 안의 dict를 mutate하면 context가 mutate된다. |
| `get_messages("openai")` | 새 list지만, **같은 dict object들** — shallow filter. 그 안의 dict를 mutate하면 여전히 context가 mutate된다. |
| `get_messages(truncate_large_values=True)` | base64 blob이 placeholder로 대체된 deep copy. 안전하고, logging에 쓰인다. |

deep copy를 하는 경로는 정확히 하나이고, 그것은 log 출력을 위해 존재하지 safety를 위해 존재하지
않습니다. 내부 state를 반환하는 getter가 code smell인 codebase에서 왔다면, 이건 버그처럼 보입니다.
이건 load-bearing이고, §2.4가 그 증명입니다.

### 2.4 이유: `_update_function_call_result`는 dict를 in place로 다시 쓴다

assistant aggregator는 history의 tool 절반을 **두 단계**로 씁니다. 두 단계 사이는 tool이 실행되는
시간만큼 벌어져 있습니다.

**1단계**, call이 시작되는 순간. message 두 개가 append됩니다: `tool_calls` array를 실은 assistant
message, 그리고 content가 문자열 `"IN_PROGRESS"` 그 자체인 placeholder `tool` message.

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1747-1781
    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        logger.debug(
            f"{self} FunctionCallInProgressFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )

        # Update context with the in-progress function call
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments, ensure_ascii=False),
                        },
                        "type": "function",
                    }
                ],
            }
        )

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

1761번 줄의 `ensure_ascii=False`는, 특히 당신에게는, 장식이 아닙니다: 한국어 tool argument가 model이
다시 읽는 context에서 `\uXXXX` escape가 되는 대신 한국어로 살아남습니다.

**2단계**, 결과가 도착할 때마다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1908-1929
    async def _handle_function_call_finished(
        self, frame: FunctionCallResultFrame, in_progress_frame: FunctionCallInProgressFrame
    ):
        """Handle the final result of a function call.

        Removes the call from the in-progress map, updates the context, and
        triggers LLM inference when appropriate.
        """
        is_async = not in_progress_frame.cancel_on_interruption
        del self._function_calls_in_progress[frame.tool_call_id]

        result = json.dumps(frame.result, ensure_ascii=False) if frame.result else "COMPLETED"

        if is_async:
            # For async function calls inject a developer message so the LLM is
            # notified of the completed result instead of updating the IN_PROGRESS
            # tool message.
            self._context.add_message(
                async_tool_messages.build_final_result_message(frame.tool_call_id, result)
            )
        else:
            self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
```

그리고 전체 design이 얹혀 있는 여섯 줄짜리 method가 여기 있습니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:2157-2165
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

`get_messages()`를 호출하고, dict들을 훑고, `tool_call_id`가 일치하는 것을 찾아서
`message["content"] = result`를 합니다. write-back이 없습니다. `set_messages`도 없습니다. **그 대입이
곧 write-back입니다** — 그리고 그것은 오직 `get_messages()`가 진짜 list 안의 진짜 dict를 넘겨줬기
때문에만 동작합니다.

sync/async 비대칭도 함께 보십시오. Lina에서 실제로 마주칠 진짜 behavioural fork이기 때문입니다:
**sync** tool(`cancel_on_interruption=True`, 기본값)은 placeholder가 다시 쓰이고, **async**
tool(`cancel_on_interruption=False`)은 아예 다시 쓰이지 않습니다 — 그 결과는 나중에 append되는 *새*
developer message로 도착합니다. flag 하나에서 두 개의 서로 다른 history shape가 나옵니다.

### 2.5 collision: `deepcopy`는 이것을 조용히 망가뜨린다

boson의 `ContextManager`는 의도적으로 정반대를 합니다:

```python
# packages/basement/basement/context/manager.py:47-52
# (boson-agent, private; excerpt-attested via [[llm-service-context]])
def get_messages(self):
    """Return a copy of the messages to prevent external mutation."""
    return deepcopy(self._messages)
```

이것은 교과서적으로 옳은 방어적 프로그래밍이고, boson에서는 실제로 옳습니다: `run_agent_loop`가 유일한
writer이고, 그것은 `ctx.add_message(...)`를 통해 쓰며, provider adapter에게 건네진 복사본은 그것에
의해 오염될 수 없습니다.

이제 그것을 포팅해 보십시오. 당신은 `BosonLLMContext(LLMContext)`를 쓰고 있고 docstring 습관도 그대로
가져옵니다:

```python
class BosonLLMContext(LLMContext):
    def get_messages(self, llm_specific_filter=None, *, truncate_large_values=False):
        """Return a copy of the messages to prevent external mutation."""
        return deepcopy(super().get_messages(llm_specific_filter,
                                             truncate_large_values=truncate_large_values))
```

tool call 하나에서 무슨 일이 일어나는지 단계별로 추적하십시오:

1. Model이 `tool_calls: [{id: "call_a1", function: {name: "lookup_policy", ...}}]`를 emit합니다.
2. `_handle_function_call_in_progress`가 message 두 개를 append합니다. context는 이제
   `{"role": "tool", "content": "IN_PROGRESS", "tool_call_id": "call_a1"}`로 끝납니다.
3. `lookup_policy`가 실행되고, `params.result_callback({"premium": 43000})`을 await합니다.
4. `_handle_function_call_finished`가 `result = '{"premium": 43000}'`을 계산합니다.
5. `_update_function_call_result("lookup_policy", "call_a1", '{"premium": 43000}')`이 **deep copy**를
   순회하며, **복사본 안에서** 일치하는 dict를 찾아, **복사본에** `content`를 설정합니다.
6. loop가 끝나면 복사본은 버려집니다. 아무것도 raise되지 않습니다. 아무것도 log되지 않습니다.
7. `_maybe_push_context_after_function_result`는 그래도 `LLMContextFrame`을 upstream으로 push합니다.
8. LLM은 tool message가 여전히 `"IN_PROGRESS"`로 읽히는 context로 다시 실행됩니다.

관측되는 증상은 *"잠시만요, 확인 중입니다"*를 영원히 말하는 bot이거나, 보험료를 hallucinate하거나,
`lookup_policy`를 다시 부르는 bot입니다 — 그리고 log 어디에도 error가 없습니다. 아무것도 error를 내지
않았기 때문입니다. 이것이 두 codebase 사이의 가장 날카로운 단일 incompatibility이고, 그 정체는 선의로
쓰인 한 줄의 코드입니다.

> **가져갈 규칙:** Pipecat에서 `LLMContext`는 **설계상 여러 concurrent writer를 가진 공유 mutable
> object**입니다. 여기에 어떤 hardening을 추가하든 `_messages` 안의 dict들의 identity를 보존해야
> 합니다. snapshot이 필요하다면 call site에서 명시적으로 뜨십시오
> (`copy.deepcopy(ctx.get_messages())`) — getter를 안전하게 만들지 마십시오.

> 💡 **쉬운 설명 — 왜 이게 "가장 날카로운" incompatibility인가**
> 대부분의 porting 실수는 시끄럽습니다: import error, type error, 500. 이 실수는 조용합니다.
> test suite가 초록불이고, log가 깨끗하고, bot이 말을 하며, 오직 *내용*만 틀립니다. 게다가 그
> 한 줄은 code review에서 "좋은 습관"으로 칭찬받을 종류의 코드입니다. 실패의 원인이 실패처럼
> 생기지 않았을 때, 그것을 잡아내는 유일한 방법은 §9.1이 말하는 test를 미리 써 두는 것뿐입니다.

### 2.6 하나의 object, 두 개의 processor: `LLMContextAggregatorPair`

pair는 processor가 아닙니다. 유일한 진짜 임무가 *같은* context object를 양쪽 절반에게 건네주는 것인,
두 줄짜리 factory입니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:2290-2314
        user_params = user_params or LLMUserAggregatorParams()
        assistant_params = assistant_params or LLMAssistantAggregatorParams()
        if add_tool_change_messages is not None:
            user_params.add_tool_change_messages = add_tool_change_messages
            assistant_params.add_tool_change_messages = add_tool_change_messages

        self._user = LLMUserAggregator(
            context,
            params=user_params,
            _realtime_service_mode=realtime_service_mode,
        )
        # Wire the assistant→user back-reference unconditionally: realtime mode
        # may be auto-configured later (realtime_service_mode=None), so the
        # reference must already exist when it flips on. ...
        self._assistant = LLMAssistantAggregator(
            context,
            params=assistant_params,
            _realtime_service_mode=realtime_service_mode,
            _paired_user_aggregator=self._user,
        )
```

`context`가 사이에 복사 없이 두 번 등장하는 것에 주목하십시오. `__iter__` (`:2332`)가
`user_agg, assistant_agg = LLMContextAggregatorPair(context)`를 쓸 수 있게 해 주는 것입니다.

**user 절반**은 완료된 transcript를 message 하나와 frame 하나로 바꿉니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:856-873
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

실질적인 것은 두 줄입니다: 공유 context에 append하고, 그다음 text가 아니라 **context 자체**를 실은
frame을 push합니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:481-496
    def _get_context_frame(self) -> LLMContextFrame:
        """Create a context frame with the current context.
        ...
        """
        return LLMContextFrame(context=self._context)

    async def push_context_frame(self, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a context frame in the specified direction.
        ...
        """
        frame = self._get_context_frame()
        await self.push_frame(frame, direction)
```

그리고 그 frame은 field 하나입니다:

```python
# src/pipecat/frames/frames.py:550-561
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

[[ch-02/read]] §3을 떠올리십시오: `LLMContextFrame`은 세 branch 중 어디에도 속하지 않고 `Frame`을
**직접** subclass합니다. data도, control도, system도 아닙니다. 이건 부주의가 아닙니다 — 값이 아니라
*공유 mutable state에 대한 reference*를 실어 나르는 frame에 대한 정직한 type입니다. 동시에 날아다니는
두 개의 `LLMContextFrame`은 같은 object를 가리킵니다. frame은 소포가 아니라 초인종입니다.

**assistant 절반**은 실제로 발화된 text에 대해 거울상을 수행합니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1677-1694
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

### 2.7 두 절반이 물리적으로 어디에 앉는가

```python
# examples/function-calling/function-calling-openai.py:101-117
    context = LLMContext(tools=[get_current_weather, get_restaurant_recommendation])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
```

list의 position만 읽어도 이미 세 가지를 알 수 있습니다:

- user aggregator는 LLM **앞**에 있으므로, 그것의 `push_context_frame()` 기본값(DOWNSTREAM)은 *다음*
  processor로서 LLM에 도달합니다.
- assistant aggregator는 **`transport.output()` 뒤**에 있습니다 — 오디오를 재생하는 것보다도 뒤,
  맨 마지막입니다. 그 배치가 [[ch-08/read]]의 주제이고(assistant history에 실제로 들렸던 text만 담기는
  이유입니다), 동시에 assistant의 re-prompt가 LLM에게 돌아가기 위해 **processor 네 개를 거슬러 upstream
  으로** 여행해야 하는 이유이기도 합니다.
- `Pipeline.__init__`은 list를 감쌉니다: `self._processors = [self._source, *processors, self._sink]`
  (`src/pipecat/pipeline/pipeline.py:119`). 따라서 assistant aggregator의 downstream 이웃은 아무것도
  아닌 게 아니라 `PipelineSink`입니다. [[ch-04/read]] §6.3이 이것을 가르쳤습니다. 즉 assistant 절반이
  downstream으로 `push_context_frame()`을 하면 그것은 sink로 가서 조용히 죽습니다.

---

## 3. Pipecat, Q1 계속: LLM service는 아무것도 들고 있지 않다

### 3.1 base service의 `process_frame`은 context를 정확히 한 번 건드린다

```python
# src/pipecat/services/llm_service.py:679-723
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.
        ...
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
        elif isinstance(frame, LLMConfigureOutputFrame):
            self._skip_tts = frame.skip_tts
        elif isinstance(frame, LLMUpdateSettingsFrame):
            ...
        elif isinstance(frame, LLMContextSummaryRequestFrame):
            await self._handle_summary_request(frame)

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

*설정*에 관한 `elif` 네 개, 그리고 context에 관한 statement는 정확히 하나 — tool handler를 sync하라.
여기서 message list를 읽지 않습니다. message list를 저장하지 않습니다. `LLMService`는 2,220줄이고
그중 어느 줄도 history를 보관하지 않습니다.

(`:709`의 `LLMContextSummaryRequestFrame`이 compaction의 진입점입니다. scope 밖 — scope box를 보십시오.
[[ch-13/read]]의 give-back list에 있습니다.)

### 3.2 구체(concrete) service: frame 하나 들어오고, completion 하나, 끝

```python
# src/pipecat/services/openai/base_llm.py:590-615
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

스물여섯 줄. 그것이 service 쪽 "agent loop"의 전부입니다. 관측 네 가지, 각각 위 block에 대고 확인할 수
있습니다:

1. **`:599`는 필수입니다.** [[ch-01/read]] §7.2가 이유를 가르쳤습니다: `super().process_frame`을
   건너뛰면 base class가 `StartFrame`을 결코 보지 못하고, process task를 시작하지 못하며, processor는
   무력해집니다. 이 chapter에 인용된 모든 `process_frame`은 이 줄로 시작합니다.
2. **`:601`은 direction을 검사하지 않습니다.** 이 파일에서 가장 결과가 큰 단일 누락입니다.
   `LLMContextFrame`은 user aggregator에서 왔든(downstream) assistant aggregator에서 왔든(**upstream**)
   completion을 trigger합니다. §5는 이 줄 위에 세워집니다.
3. **context frame은 소비됩니다.** `if` branch 안에서 `frame`은 다시 push되지 않습니다. `:614`의
   `else`만 forward합니다. 따라서 `LLMContextFrame`은 양방향 모두에서 **LLM service에서 멈춥니다**.
   LLM 뒤의 어떤 것도 downstream으로 여행하는 그것을 보지 못하고, LLM 앞의 어떤 것도 upstream으로
   여행하는 그것을 보지 못합니다.
4. **`finally`는 항상 `LLMFullResponseEndFrame`을 push합니다** — task가 cancel되는 경우만 예외이고,
   그건 [[ch-08/read]]의 영역이라 여기서 다시 가르치지 않습니다.

그리고 `_process_context`는 어디서 끝날까요? loop에서가 아닙니다:

```python
# src/pipecat/services/openai/base_llm.py:563-588
        if function_name:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments or "{}")
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: Failed to parse function call arguments: {arguments}")
                    continue
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)
```

`await self.run_function_calls(function_calls)`, 그리고 함수는 반환합니다. `while`이 없습니다.
`continue`가 없습니다. `_process_context`에 대한 두 번째 호출이 없습니다. service는 tool을 쏘고
집에 갑니다.

> **계속 읽기 전에 스스로에게 한 문장으로 진술하십시오:** *Pipecat LLM service는 `LLMContextFrame`당
> 정확히 한 번의 inference를 수행하고, tool call을 쏘는 것이 그것이 하는 마지막 일이다.*

---

## 4. Pipecat, Q2: 누가 tool을 dispatch하는가

[[function-calling]]은 이 section 전체를 한 문장으로 요약합니다 — *"Pipecat has no multi-turn tool
loop inside the LLM service"* — 그리고 아래 절들은 그 문장을 기계적 부품으로 분해한 것이며, 각각
source에 대고 다시 읽었습니다. 아래의 모든 줄 번호를 다시 측정했습니다. excerpt와 tree는 일치하고,
*outline*이 둘 모두로부터 벗어나 있던 곳 — adapter LOC 수치, §4.7 — 에서는 tree가 이기고 명령을
보여드립니다.

### 4.1 세 개의 registration 경로, 하나의 dictionary

모든 경로는 `self._functions: dict[str | None, FunctionCallRegistryItem]`
(`llm_service.py:380`)에서 끝납니다. item은 field 여섯 개짜리 dataclass입니다:

```python
# src/pipecat/services/llm_service.py:200-205
    function_name: str | None
    handler: FunctionCallHandler | DirectFunctionWrapper
    cancel_on_interruption: bool
    timeout_secs: float | None = None
    cancellable_by_llm: bool = False
    auto_registered: bool = False
```

**경로 1 — direct function.** 첫 parameter가 `params`이고 **docstring이 곧 schema**인 평범한 async
callable:

```python
# examples/function-calling/function-calling-openai.py:36-52
async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    await params.result_callback({"conditions": "nice", "temperature": "75"})


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"name": "The Golden Dragon"})
```

wrapping은 `ToolsSchema.__init__` 안에서 일어나고, 그곳이 "list 안의 callable"이 "schema + wrapper"가
되는 지점입니다:

```python
# src/pipecat/adapters/schemas/tools_schema.py:54-69
        def _map_standard_tools(tools):
            schemas = []
            direct_functions = []
            for tool in tools:
                if isinstance(tool, FunctionSchema):
                    schemas.append(tool)
                elif callable(tool):
                    wrapper = DirectFunctionWrapper(tool)
                    schemas.append(wrapper.to_function_schema())
                    direct_functions.append(wrapper)
                else:
                    raise TypeError(f"Unsupported tool type: {type(tool)}")
            return schemas, direct_functions

        self._standard_tools, self._direct_functions = _map_standard_tools(standard_tools)
        self._custom_tools = custom_tools
```

그리고 invocation은 한 줄입니다:

```python
# src/pipecat/adapters/schemas/direct_function.py:279-289
    async def invoke(self, args: Mapping[str, Any], params: "FunctionCallParams"):
        """Invoke the wrapped function with the provided arguments.
        ...
        """
        return await self.function(params=params, **args)
```

boson의 `@tool` decorator에서 왔다면, 이것은 같은 트릭입니다 — docstring을 description으로, signature를
JSON Schema로 — decorator만 제거된 형태입니다. `@tool`은 docstring이 없으면 raise하고
(`tools/decorator.py:42-45`, [[boson-tool-router]] 기준), `DirectFunctionWrapper`는 같은 두 source에서
schema를 유도합니다.

**경로 2 — handler를 실은 `FunctionSchema`.**

```python
# src/pipecat/adapters/schemas/function_schema.py:32-58
    def __init__(
        self,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
        handler: "FunctionCallHandler | None" = None,
    ) -> None:
        """Initialize the function schema.

        Args:
            name: Name of the function to be called.
            description: Description of what the function does.
            properties: Dictionary defining parameter types, descriptions, and constraints.
            required: List of property names that are required parameters.
            handler: Optional handler for this function. When provided, the LLM
                service registers it automatically wherever the schema is
                advertised in the `LLMContext`, making a separate
                ``register_function`` call unnecessary. ...
        """
        self._name = name
        self._description = description
        self._properties = properties
        self._required = required
        self._handler = handler
```

`handler=None`인 `FunctionSchema`는 **advertise 전용**입니다: model은 그 tool을 보고, 그것을 호출하면
catch-all이나 missing-handler 경로로 떨어집니다. 그 비대칭은 §10에서 원하게 될 hook입니다.

**경로 3 — 명시적 `register_function`, catch-all 포함.**

```python
# src/pipecat/services/llm_service.py:879-897
    def register_function(
        self,
        function_name: str | None,
        handler: Any,
        *,
        cancel_on_interruption: bool | None = None,
        timeout_secs: float | None = None,
        cancellable_by_llm: bool | None = None,
    ):
        """Register a function handler for LLM function calls.

        Call options resolve with the precedence **explicit argument >
        ``@tool_options`` decorator > default**. ``None`` (the default) means
        "not provided" — the option falls back to the ``@tool_options`` value on
        the handler, then to the documented default.

        Args:
            function_name: The name of the function to handle. Use None to handle
                all function calls with a catch-all handler.
```

`function_name=None`은 **catch-all**을 설치합니다: registry가 이름으로 매칭하지 못한 모든 call을 받는
handler 하나. 이걸 기억하십시오. §10.2의 "wrap" resolution의 기계적 기반 전부입니다.

### 4.2 registry는 context frame마다 다시 sync된다

이것이 service가 아니라 context를 tool에 대한 source of truth로 만드는 mechanism입니다:

```python
# src/pipecat/services/llm_service.py:1196-1212
        # Register direct functions.
        for wrapper in normalized.direct_functions:
            if wrapper.name in self._functions:
                continue
            if wrapper.name in self._explicitly_unregistered_function_names:
                # Explicitly unregistered while still advertised — leave it gone so
                # calls hit the missing-handler recovery path.
                continue
            self._register_direct_function(wrapper.function)
            # Mark the entry as advertised-tool-set-managed so it can be pruned on a
            # later sync that stops advertising it. Names already in _functions are
            # skipped above, so explicit registrations keep their default
            # auto_registered=False and are never pruned.
            self._functions[wrapper.name].auto_registered = True
            logger.debug(
                f"{self}: auto-registered handler for advertised direct function '{wrapper.name}'"
            )
```

그리고 pruning 절반:

```python
# src/pipecat/services/llm_service.py:1339-1354
    def _unregister_unadvertised_tool_handlers(self, advertised: set[str | None]) -> None:
        """Drop auto-registered handlers for tools no longer advertised.

        Only entries with ``auto_registered=True`` are eligible; explicit
        registrations, the catch-all handler, and built-in tools are untouched.
        ...
        """
        stale = [
            name
            for name, item in self._functions.items()
            if item.auto_registered and name not in advertised
        ]
        for name in stale:
            del self._functions[name]
```

two-tier ownership이고, 이름 붙여 둘 가치가 있습니다. stage machine이 원하는 바로 그 seam이기
때문입니다:

| Registration | `auto_registered` | 광고가 끊기면 pruning되는가? |
|---|---|---|
| 광고된 direct function / `FunctionSchema(handler=...)` | `True` | 예, 다음 `LLMContextFrame`에서 |
| `register_function(name, handler)` | `False` | 결코 아니오 |
| catch-all `register_function(None, handler)` | `False` | 결코 아니오 |

그러므로 "model이 볼 수 있는 것"(`context.tools`)이 "service가 실행할 수 있는 것"(`_functions`)을
구동합니다 — 자동으로, inference마다, 한 방향으로. 그것이 boson이 셋을 유지하는 자리에서 gate 하나입니다.
§8.5.

### 4.3 Dispatch: `run_function_calls` → `_run_function_call` → `result_callback`

```python
# src/pipecat/services/llm_service.py:1446-1487
        if len(function_calls) == 0:
            return

        # Exclude the built-in cancel tool — it's an internal mechanism and
        # should not be surfaced to user-facing event handlers or frames.
        user_visible_calls = [
            fc for fc in function_calls if fc.function_name not in self._cancel_tool_names
        ]
        if user_visible_calls:
            await self._call_event_handler("on_function_calls_started", user_visible_calls)
            await self.broadcast_frame(FunctionCallsStartedFrame, function_calls=user_visible_calls)

        # When group_parallel_tools is True all calls share a group_id so the
        # aggregator triggers the LLM exactly once after the last one completes.
        # When False, group_id is None and each result triggers inference independently.
        group_id = str(uuid.uuid4()) if self._group_parallel_tools else None

        runner_items = []
        for function_call in function_calls:
            if function_call.function_name in self._functions.keys():
                item = self._functions[function_call.function_name]
            elif None in self._functions.keys():
                item = self._functions[None]
            else:
                self._log_missing_function_call(function_call.function_name, function_call.context)
                item = self._build_missing_function_call_registry_item(function_call.function_name)

            runner_items.append(
                FunctionCallRunnerItem(
                    registry_item=item,
                    function_name=function_call.function_name,
                    tool_call_id=function_call.tool_call_id,
                    arguments=function_call.arguments,
                    context=function_call.context,
                    group_id=group_id,
                )
            )

        if self._run_in_parallel:
            await self._run_parallel_function_calls(runner_items)
        else:
            await self._run_sequential_function_calls(runner_items)
```

3단계 이름 해소 — 정확한 이름, 그다음 `None` catch-all, 그다음 error message로 call을 정산해서 turn이
그래도 종료되게 만드는 합성된 missing-handler item.

`:1456`의 `broadcast_frame`은 펼쳐 볼 가치가 있습니다. §9.3의 turn-cap processor가 그 frame이 정확히
어디로 가는지에 의존하기 때문입니다:

```python
# src/pipecat/processors/frame_processor.py:1038-1054
    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.

        This method creates two instances of the given frame class using the
        provided keyword arguments (without deep-copying them) and pushes them
        upstream and downstream.
        ...
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)
```

**instance 두 개**, `broadcast_sibling_id`로 서로 연결되고, 방향마다 하나씩. [[ch-08/read]]가
`InterruptionFrame`에 대해 추적한 것과 같은 mechanism입니다. 그러므로 `FunctionCallsStartedFrame`은
pipeline의 *모든* processor에게 보이지만, 개별 processor는 자기 옆을 지나가는 쪽 **하나**만 봅니다.

그다음 call마다:

```python
# src/pipecat/services/llm_service.py:1547-1583
    async def _run_function_call(self, runner_item: FunctionCallRunnerItem):
        # Re-resolve the registry item at execution time. The function may have
        # been unregistered between queuing and execution, in which case we
        # fall back to the missing-function handler so the call still terminates
        # with a normal tool result.
        if runner_item.function_name in self._functions.keys():
            item = self._functions[runner_item.function_name]
        elif None in self._functions.keys():
            item = self._functions[None]
        ...
        # Broadcast function call in-progress. This frame will let our assistant
        # context aggregator know that we are in the middle of a function
        # call. Some contexts/aggregators may not need this. But some definitely
        # do (Anthropic, for example).
        await self.broadcast_frame(
            FunctionCallInProgressFrame,
            function_name=runner_item.function_name,
            tool_call_id=runner_item.tool_call_id,
            arguments=runner_item.arguments,
            cancel_on_interruption=item.cancel_on_interruption,
            group_id=runner_item.group_id,
        )
```

그리고 실제 invocation, 두 가지 handler shape와 함께:

```python
# src/pipecat/services/llm_service.py:1650-1683
        if item.timeout_secs or self._function_call_timeout_secs:
            timeout_task = self.create_task(timeout_handler())

        try:
            if isinstance(item.handler, DirectFunctionWrapper):
                # Handler is a DirectFunctionWrapper
                await item.handler.invoke(
                    args=runner_item.arguments,
                    params=FunctionCallParams(
                        function_name=runner_item.function_name,
                        tool_call_id=runner_item.tool_call_id,
                        arguments=runner_item.arguments,
                        llm=self,
                        pipeline_worker=self.pipeline_worker,
                        context=runner_item.context,
                        result_callback=function_call_result_callback,
                        app_resources=self.pipeline_worker.app_resources,
                        worker_runner=self.pipeline_worker.worker_runner,
                    ),
                )
            else:
                # Handler is a FunctionCallHandler
                params = FunctionCallParams(
                    ...
                )
                await item.handler(params)
```

### 4.4 handler contract: 절대 값을 return하지 않는다

```python
# src/pipecat/services/llm_service.py:142-155
    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    # `LLMService[Any]` so any concrete subclass (regardless of how — or
    # whether — it parameterizes the adapter type) can be assigned here.
    # Plain `LLMService` would invoke the TypeVar default and pyright would
    # treat it invariantly, rejecting `LLMService[XAdapter]` at the call
    # sites that build FunctionCallParams.
    llm: LLMService[Any]
    pipeline_worker: PipelineWorker
    context: LLMContext
    result_callback: FunctionCallResultCallback
    app_resources: Any = None
    worker_runner: WorkerRunner | None = None
```

Pipecat tool handler는 argument **하나**를 받고 **아무것도** return하지 않습니다. 보고는
`params.result_callback(result)`를 await함으로써 합니다. 대신 값을 return하면 그 값은 버려지고 call은
결코 정산되지 않습니다 — `function_call_timeout_secs`를 설정했다면 그것이 발화할 때까지, 설정하지
않았다면 영원히 hang합니다.

[[boson-tool-router]] 기준 boson의 contract와 비교하십시오:

```python
# packages/basement/basement/tools/executor.py:67 — _invoke_handler
# (boson-agent, private; excerpt-attested via [[boson-tool-router]])
#   async handlers:  await spec.handler(**arguments)
#   return contract: a returned ToolResultBlock passes through unchanged;
#                    anything else is str()-ified into ToolResultBlock(...)
#                    exceptions become "Tool error: {type}: {msg}" with is_error=True
```

차이 둘, 둘 다 기계적이고, 둘 다 당신이 소유한 모든 tool에 영향을 미칩니다:

| | boson | Pipecat |
|---|---|---|
| Signature | `handler(**arguments)` — kwargs를 뿌린다 | `handler(params: FunctionCallParams)` — object 하나, argument는 `params.arguments` 안에 |
| Result | `return value` | `await params.result_callback(value)` |

[[boson-tool-router]]는 **`agents/*/tools/` 아래 22개 tool**을 셉니다. 각각이 signature 변경 또는 shim을
필요로 합니다. 그것이 §10.1을 위한 숫자입니다.

`FunctionCallParams`가 handler에게 주는데 boson의 `handler(**arguments)`는 주지 않는 것도 눈여겨보십시오:
`params.context` — *live* `LLMContext`입니다. Pipecat tool handler는 tool 안에서 conversation history를
읽고 다시 쓸 수 있습니다. 당신의 규율에 따라 이것은 매우 유용한 hook이거나 footgun입니다. boson에는
대응물이 없습니다. boson tool은 자기 argument만 받기 때문입니다.

### 4.5 function-call frame 둘 다 uninterruptible이고, 그것은 의도적이다

```python
# src/pipecat/frames/frames.py:769-791
@dataclass
class FunctionCallResultFrame(DataFrame, UninterruptibleFrame):
    """Frame containing the result of an LLM function call.

    This is an uninterruptible frame because once a result is generated we
    always want to update the context.

    Parameters:
        function_name: Name of the function that was executed.
        tool_call_id: Unique identifier for the function call.
        arguments: Arguments that were passed to the function.
        result: The result returned by the function.
        run_llm: Whether to run the LLM after this result.
        properties: Additional properties for result handling.

    """

    function_name: str
    tool_call_id: str
    arguments: Any
    result: Any
    run_llm: bool | None = None
    properties: FunctionCallResultProperties | None = None
```

```python
# src/pipecat/frames/frames.py:2163-2187
@dataclass
class FunctionCallInProgressFrame(ControlFrame, UninterruptibleFrame):
    """Frame signaling that a function call is currently executing.

    This is an uninterruptible frame because we always want to update the
    context.
    ...
        group_id: Identifier shared by all function calls originating from the
            same LLM response batch. Used to determine when the last call in a
            group completes so the LLM can be triggered exactly once.
    """
```

mixin 자신의 docstring이 그것이 무엇을 사주는지 말합니다:

```python
# src/pipecat/frames/frames.py:146-157
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

둘을 함께 읽으십시오: 시작된 tool call은 **반드시** history에 기록되고, 생성된 tool result는 **반드시**
history에 착륙합니다. 고객이 call 도중에 끼어들어도 그렇습니다. 그것이 정확히 boson이 손으로 약 80줄을
써서 유지하는 invariant입니다 — `_reconcile_cancelled_tool_uses` (`agent_loop.py:56`)와
`_await_tool_boundary` (`:113`)이고, [[boson-agent-loop]]에 따르면 답 없는 모든 `tool_use`를 합성된
`ToolResultBlock(content=f"canceled: {tu['name']}", is_error=True)`와 짝지은 뒤 re-raise합니다. Pipecat은
같은 invariant를 dataclass 위의 mixin 하나 + scheduling rule 하나에서 얻습니다. cascade는
[[ch-08/read]]의 소유입니다. 여기서 요점은 오직 **누가 그 코드를 썼는가**입니다: frame flag냐,
reconciliation routine이냐.

> 💡 **쉬운 설명 — "mixin 하나 = 80줄"이 왜 성립하나**
> boson은 turn을 함수 안에서 돌리므로, cancellation이 오면 그 함수가 스스로 "지금까지 뭘 하다 말았지?"를
> 계산해서 history를 봉합해야 합니다. Pipecat은 turn을 frame으로 돌리므로, 같은 보장을 "이 frame은
> interruption 때 queue에서 버리지 말 것"이라는 표식 하나로 얻습니다. runtime이 이미 queue와
> cancellation을 소유하고 있기 때문에, 정책을 데이터에 붙일 수 있는 겁니다. 대신 그 보장이 어디서
> 오는지는 코드에서 눈에 덜 띕니다 — dataclass 상속 목록에 적힌 단어 하나가 전부니까요.

### 4.6 유일한 bound는 call 단위 timeout이다

```python
# src/pipecat/services/llm_service.py:1632-1648
        # Start a timeout task for deferred function calls
        async def timeout_handler():
            effective_timeout = item.timeout_secs or self._function_call_timeout_secs
            # This task is only started when one of the two is set.
            assert effective_timeout is not None
            await asyncio.sleep(effective_timeout)
            logger.warning(
                f"{self} Function call [{runner_item.function_name}:{runner_item.tool_call_id}] timed out after {effective_timeout} seconds and is being cancelled."
                f" You can increase this timeout by passing `timeout_secs` to `register_function()`,"
                f" or set a global default via `function_call_timeout_secs` on the LLM constructor."
            )
            # Settle the call before cancelling, so a result racing in while the
            # handler unwinds is rejected instead of broadcast.
            runner_item.settled = True
            # Cancelling goes through a detached task because it awaits the
            # function call task this handler runs inside of.
            self.create_task(self._timeout_function_call(runner_item))
```

두 bound 모두 **call 단위**입니다: tool 하나에 대한 `timeout_secs`, service에 대한
`function_call_timeout_secs`. 어느 쪽도 *call이 몇 번 일어나는지*를 bound하지 않습니다. §6까지 붙들고
계십시오.

### 4.7 provider별 tool shaping은 abstract method 하나 뒤에 산다

```python
# src/pipecat/adapters/base_llm_adapter.py:113-123
    @abstractmethod
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[Any]:
        """Convert tools schema to the provider's specific format.

        Args:
            tools_schema: The standardized tools schema to convert.

        Returns:
            List of tools in the provider's expected format.
        """
        pass
```

boson이 손으로 여섯 번 쓰는 shape 차이를 볼 수 있도록, 구현 두 개:

```python
# src/pipecat/adapters/services/open_ai_adapter.py:206-220
        functions_schema = tools_schema.standard_tools
        # `function=...` expects a `FunctionDefinition` TypedDict; the dict
        # produced by `to_default_dict()` is structurally compatible. Cast at
        # the boundary.
        formatted_standard_tools: list[ChatCompletionToolParam] = [
            ChatCompletionToolParam(type="function", function=cast(Any, func.to_default_dict()))
            for func in functions_schema
        ]
        custom_openai_tools: list[ChatCompletionToolParam] = []
        if tools_schema.custom_tools:
            custom_openai_tools = cast(
                list[ChatCompletionToolParam],
                tools_schema.custom_tools.get(AdapterType.OPENAI, []),
            )
        return formatted_standard_tools + custom_openai_tools
```

```python
# src/pipecat/adapters/services/anthropic_adapter.py:499-500
        functions_schema = tools_schema.standard_tools
        return [self._to_anthropic_function_format(func) for func in functions_schema]
```

**숫자를 올바른 경로에 붙이십시오.** outline이 이걸 한 번 틀렸고, 이건 review를 살아남는 종류의
오류이기 때문입니다:

```console
$ find src/pipecat/adapters/services -name "*.py" | xargs wc -l | tail -1
    3647 total
$ find src/pipecat/adapters/services -name "*.py" | wc -l
13          # 12 concrete adapter modules + an empty __init__.py

$ find src/pipecat/adapters -name "*.py" | xargs wc -l | tail -1
    4549 total
$ find src/pipecat/adapters -name "*.py" | wc -l
19
```

**`adapters/services/`의 provider adapter module 12개에 걸쳐 3,647 L. `adapters/` 전체는 19개 파일에
걸쳐 4,549 L.** 여분의 약 900줄은 base adapter, schema layer(`function_schema.py`, `tools_schema.py`,
`direct_function.py`), 그리고 registry입니다. 어느 쪽을 뜻하는지 인용하고, 어느 쪽을 뜻했는지 밝히십시오.

12개 module, 큰 것부터: `gemini_adapter.py` 802, `anthropic_adapter.py` 500,
`bedrock_adapter.py` 375, `open_ai_responses_adapter.py` 294, `open_ai_adapter.py` 269,
`grok_realtime_adapter.py` 265, `inworld_realtime_adapter.py` 261,
`open_ai_realtime_adapter.py` 244, `aws_nova_sonic_adapter.py` 240, `perplexity_adapter.py` 173,
`mistral_adapter.py` 135, `gemini_live_adapter.py` 89.

그에 맞서, [[boson-tool-router]] 기준 boson은 `PROVIDER_REGISTRY`(`llm/registry.py:61-68`)에
**provider factory 6개**를 가집니다: `anthropic`, `openai`, `google`, `boson`, `xai`, `openrouter` —
마지막 셋은 OpenAI-compatible subclass입니다. 각각이 자기 tool shaping을 손으로 씁니다: Anthropic의
`{"name","description","input_schema"}` 대 OpenAI의
`{"type":"function","function":{"name","description","parameters"}}` — `input_schema` 대
`parameters`이고, 그것이 정확히 `to_provider_tools_format`이 흡수하려고 존재하는 차이입니다.

---

## 5. Pipecat, Q3: loop는 frame이 거꾸로 가기 때문에 닫힌다

이제 그것을 loop로 만드는 조각 하나만 빼고 전부 갖췄습니다. 여기 있습니다.

### 5.1 upstream push

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1859-1889
    async def _maybe_push_context_after_function_result(self) -> None:
        """Decide whether to push a context frame after a function call settles.

        Push an ``LLMContextFrame`` upstream (with care to avoid duplicate
        pushes while results are queued or the bot is still speaking).
        Cascade LLMs use the context frame to re-run inference with the
        new tool result in scope. Realtime LLMs read the new tool result
        out of the context the same way — they don't get function results
        from ``FunctionCallResultFrame`` directly — so the same push is
        load-bearing for both modes.
        """
        if self.has_queued_frame(FunctionCallResultFrame):
            # Another FunctionCallResultFrame is already queued. Defer the context push
            # to bundle all results into a single LLM call instead of triggering one
            # inference pass per result. The context will be pushed once the last
            # function call in the queue is processed.
            logger.debug(
                f"{self}: More FunctionCallResultFrames queued — deferring context frame push."
            )
        elif self._bot_speaking:
            # Defer the context frame push until the bot finishes speaking. If multiple
            # function call results arrive while the bot is speaking, they all accumulate
            # in the context and a single push is performed once speaking stops, preventing
            # the LLM from running multiple times and producing duplicated responses.
            # This should be an edge case, since it would require a FunctionCallResultFrame
            # being queued between an LLM response start and end frame.
            logger.debug(f"{self}: Bot is speaking — deferring context frame push.")
            self._push_context_on_bot_stopped_speaking = True
        else:
            logger.debug(f"{self}: Pushing context frame!")
            await self.push_context_frame(FrameDirection.UPSTREAM)
```

`FrameDirection.UPSTREAM`, **pipeline의 마지막 processor**에서, LLM이 읽을 context에 대한 reference를
싣고서. `:1889`의 그 한 줄이 tool loop 전부입니다.

### 5.2 그 앞의 gate

그 method가 호출되기도 전에, `_handle_function_call_result`가 애초에 re-prompt가 정당한지를 결정합니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:1811-1848
        run_llm = False

        # Append any images that were generated by function calls.
        if frame.tool_call_id in self._function_calls_image_results:
            image_frame = self._function_calls_image_results[frame.tool_call_id]

            del self._function_calls_image_results[frame.tool_call_id]

            # If an image frame has been added to the context, let's run inference.
            run_llm = await self._maybe_append_image_to_context(image_frame)

        # Run inference if the function call result requires it.
        if frame.result:
            if properties and properties.run_llm is not None:
                # If the tool call result has a run_llm property, use it.
                run_llm = properties.run_llm
            elif frame.run_llm is not None:
                # If the frame is indicating we should run the LLM, do it.
                run_llm = frame.run_llm
            else:
                # Run the LLM when this is the last function call in the group
                # to complete. If group_id is set, only consider sibling calls;
                # otherwise always execute as soon as we receive the result.
                if group_id:
                    run_llm = not any(
                        f is not None
                        and f.group_id == group_id
                        # We are now able to receive "updates", so the current
                        # frame can still be in the in progress list, and we need to
                        # ignore it.
                        and f.tool_call_id != frame.tool_call_id
                        for f in self._function_calls_in_progress.values()
                    )
                else:
                    run_llm = True

        if run_llm and not self._user_speaking:
            await self._maybe_push_context_after_function_result()
```

그러므로 re-prompt 결정은 4단계 precedence입니다:

1. handler가 설정한 `properties.run_llm` → 그것에 따른다.
2. result frame에 설정된 `frame.run_llm` → 그것에 따른다.
3. `group_id`가 설정됨 → batch에서 아직 남아 있는 마지막 sibling일 때만 re-prompt한다.
4. 그 외 → 즉시 re-prompt한다.

여기에 suppressor 둘이 붙습니다: `self._user_speaking`(말하고 있는 고객 위에 re-prompt하지 말 것)과
`self._bot_speaking`(bot의 현재 발화가 끝날 때까지 미룰 것).

따라서 handler는 `FunctionCallResultProperties(run_llm=False)`로 정산함으로써 자기 안에서 turn을 끝낼 수
있습니다 — re-prompt 없음, 답변 없음, 침묵. 그것이 `end_call`이나 `transfer_to_human` 같은 tool을 위한
당신의 hook이고, Pipecat이 가진 것 중 의도적인 "turn이 끝났다" 선언에 가장 가까운 것입니다. 그래도 여전히
turn *cap*은 아닙니다. 계속하기를 거절하는 tool 하나일 뿐입니다.

### 5.3 cycle을 linked list 위의 walk로 추적하기

§2.7의 canonical pipeline을 가져와 tool을 쓰는 turn 하나를 따라가십시오. 모든 hop이 위에서 찾을 수 있는
실제 `push_frame`입니다.

```
transport.input  →  stt  →  user_agg  →  llm  →  tts  →  transport.output  →  assistant_agg
                                          ↑                                        │
                                          └────────────── UPSTREAM ────────────────┘
```

| # | 어디서 | 무슨 일이 일어나는가 | 줄 |
|---|---|---|---|
| 1 | `user_agg` | turn이 끝남; `add_message({"role":"user", ...})`; `push_context_frame()` DOWNSTREAM | `:863-866` |
| 2 | `llm` | `isinstance(frame, LLMContextFrame)` → `LLMFullResponseStartFrame` push, metrics 시작, `_process_context()` | `:601-605` |
| 3 | `llm` | model이 `tool_calls`를 emit; `FunctionCallFromLLM`들을 만듦; `run_function_calls()` | `base_llm.py:579-588` |
| 4 | `llm` | `FunctionCallsStartedFrame`을 **양방향**으로 broadcast | `llm_service.py:1456` |
| 5 | `llm` | call마다: `FunctionCallInProgressFrame`을 양방향으로 broadcast | `llm_service.py:1576-1583` |
| 6 | `assistant_agg` | in-progress frame을 봄 → `tool_calls` message + `"IN_PROGRESS"` placeholder를 append | `:1753-1779` |
| 7 | `llm` | `finally:` `LLMFullResponseEndFrame` push. **`process_frame`이 반환. service는 이제 idle.** | `base_llm.py:611-613` |
| 8 | handler | `await params.result_callback({...})` → `FunctionCallResultFrame` broadcast | `llm_service.py:1589+` |
| 9 | `assistant_agg` | placeholder를 **in place**로 다시 씀 | `:2157-2165` |
| 10 | `assistant_agg` | `run_llm`이 True로 해소됨 → `push_context_frame(UPSTREAM)` | `:1847-1889` |
| 11 | `transport.output` → `tts` → `llm` | frame이 upstream으로 세 hop을 걸어감 | `frame_processor.py:1183-1194` |
| 12 | `llm` | `isinstance(frame, LLMContextFrame)` — **direction을 검사하지 않음** → 또 한 번의 completion | `base_llm.py:601` |
| 13 | `llm` | model이 text로 답함; `LLMTextFrame`들이 downstream으로 `tts`에 감 | — |
| 14 | `assistant_agg` | `push_aggregation()`이 `{"role":"assistant","content": ...}`를 append | `:1685` |
| 15 | — | `FunctionCallResultFrame`이 없으므로 upstream으로 아무것도 push되지 않음. **turn은 부재로 끝난다.** | — |

step 12가 경첩(hinge)입니다. step 15가 Q3에 대한 답입니다.

> **Q3, Pipecat의 답, 한 문장으로:** *아무것도 turn이 끝났다고 결정하지 않는다. turn은 어떤 processor도
> LLM에게 `LLMContextFrame`을 push하지 않을 때 끝나고, 그것을 upstream으로 push하는 유일한 것은 정산된
> tool result다.*

`break`가 없습니다. state machine이 없습니다. terminal event가 없습니다 — realtime_voice와
비교하십시오. 그쪽은 `self._terminal: dict[GenerationId, VoiceEventKind]`와 `pipeline/session.py:516`의
`_terminal_event()`로 **generation당 정확히 하나의 terminal event**를 강제합니다
([[rtv-pipeline-session]] 기준). Pipecat의 turn boundary는 system 안에 아예 표현되어 있지 않습니다.

> 💡 **쉬운 설명 — "부재로 끝난다"가 왜 불편한가**
> 프로그램에서 "일어난 일"은 log할 수 있지만 "일어나지 않은 일"은 log할 수 없습니다. boson에서는
> `break`에 breakpoint를 걸 수 있습니다. Pipecat에서는 turn이 끝나는 순간에 걸 지점이 없습니다 —
> 그냥 아무도 아무것도 push하지 않았을 뿐입니다. 그래서 §9.3의 counting processor가 필요할 때,
> 당신이 세는 것은 "turn의 끝"이 아니라 "upstream으로 지나가는 re-prompt"입니다. 관측 가능한 사건을
> 세는 것 외에 다른 방법이 없습니다.

### 5.4 여기서 figure를 쓰십시오

→ **[ch-09 two-loops viewer 열기](./figures/two-loops.html)** 그리고 tool을 쓰는 turn 하나를 두 column
모두에서 단계별로 진행시키십시오.

이 도구로 다음 세 가지를, 순서대로 하십시오. 세 번째를 대충 넘기지 마십시오:

1. **Pipecat column을 in-place rewrite까지 진행시키십시오.** `_update_function_call_result`가
   `message["content"] = result`를 대입할 때 공유 message list가 번쩍이는 것을 보고, 같은 순간에 boson
   column의 `deepcopy`가 list를 갈라내는 것을 보십시오. 그 번쩍임이 §2.5 — error를 내지 않는 실패입니다.
2. **runaway-tool counter를 toggle하십시오.** model이 계속 tool을 부르게 두십시오. Pipecat의 counter는
   천장이 없어서 끝없이 올라가고, boson의 counter는 50에서 멈추고
   `MessageEnd(stop_reason="max_turns")`를 emit합니다. 이것이 §6입니다.
3. **resolution toggle을 adopt / wrap / bypass로 전환하며** 각각이 무엇을 깨뜨리는지 읽으십시오.
   figure는 그중 어느 것도 recommended로 표시하지 않습니다 — 세 개의 cost column을 보여 줄 뿐입니다.
   그것은 §10과 같은 규율이고, 의도적입니다.

---

## 6. Pipecat, 그 부재: tool loop 없음, turn cap 없음

grep 두 개, 둘 다 2026-08-25에 commit `0cbf9c5b` tree에 대고 다시 실행했습니다.

```console
$ cd wiki/raw-data/pipecat/pipecat-src
$ grep -rn "max_turns\|max_iterations\|max_tool_calls" src/pipecat/ | wc -l
0
```

hit이 0입니다. 이름이 다른 게 아니라 — 개념이 부재합니다. `src/pipecat/`에서 turn 안의 inference를 세는
무언가를 찾아보면 같은 무(無)를 발견하게 됩니다. [[function-calling]]도 같은 말로 진술합니다:
*"Think-act-observe is a topology, not a `while` block — and consequently it has **no turn cap**."*

이제 그것을 §5.3의 step 12 옆에 놓고, Lina의 domain에서 failure mode를 구체적으로 상상하십시오:

> 고객이 *"제 보험료가 얼마죠?"*라고 묻습니다. model이 `lookup_policy`를 호출합니다. tool이
> `{"error": "policy_id required"}`를 반환합니다. error를 본 model이 추측한 id로 `lookup_policy`를 다시
> 호출합니다. 같은 error. 또. 또.

각 cycle은: result frame → placeholder rewrite → `run_llm=True` → upstream `LLMContextFrame` →
completion → `run_function_calls` → result frame. 그 cycle 안의 어떤 것도 세지 않습니다. 그 안의 어떤
것도 멈추지 않습니다. 당신이 가진 bound는:

| Bound | 어디에 | 실제로 무엇을 제한하는가 |
|---|---|---|
| `timeout_secs` | tool 단위, `register_function(..., timeout_secs=...)` | call **하나**가 얼마나 오래 실행될 수 있는가 |
| `function_call_timeout_secs` | `LLMService.__init__` `:307` | **어떤** call이든 얼마나 오래 실행될 수 있는가 |
| `FunctionCallResultProperties(run_llm=False)` | result 단위, handler가 설정 | **이 하나의** result가 re-prompt를 하는가 |
| — | — | **cycle이 몇 번 일어나는지는 아무것도 제한하지 않는다** |

각 cycle은 inference 하나의 비용이 들고, voice call에서는 고객이 침묵이나 filler를 계속 듣게 만듭니다.
cycle당 700 ms면 백 번은 실제 전화 통화에서 1분이 넘는 dead air이고, 그것을 끝내는 유일한 것은 고객이
끊는 것 또는 `PipelineWorker`의 idle timeout([[ch-04/read]] §9)이 발화하는 것뿐입니다.

그에 맞서, boson은 두 줄로 bound합니다:

```python
# packages/basement/basement/loop/agent_loop.py:207-209
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
turn_count = 0
while turn_count < runtime.config.max_turns:
    ...
```

`max_turns: int = Field(default=50, ge=1, le=1000)` (`schemas/config_schema.py:62`)와 함께, 그리고
`:365`의 `while/else`에 명시적 소진 경로가 있어서 `TextDelta(text="\n[Max turns exceeded — stopping]")`와
`MessageEnd(stop_reason="max_turns")`를 yield합니다.

> **부드럽게 말하는 것이 오히려 해가 되므로, 그대로 말합니다:** boson의 `max_turns` guard와 그
> `while/else` 소진 경로는 **Pipecat에 대응물이 없습니다**. 포팅할 수 없습니다. counting
> `FrameProcessor`로 재구축하거나, 위험을 기록해 둔 채 의식적으로 버려야 합니다. §9.3이 그 processor를
> 만듭니다.

---

## 7. 막간: Pipecat의 ownership model이 사주는 것

두 문단, 코드 없음, 그리고 곧장 collision으로 복귀. 부재만 나열하는 chapter는 편향된 chapter이기 때문에
여기 있습니다.

ownership을 세 갈래로 쪼개는 것 — context는 평범한 object에, dispatch는 service에, loop closure는
topology에 — 은 각 조각을 독립적으로 교체 가능하게 만들고, 그것이 [[ch-01/read]]의 substitutability
주장을 LLM layer에서 현금화한 것입니다. context를 건드리지 않고 `OpenAILLMService`를
`AnthropicLLMService`로 바꿀 수 있습니다. context가 provider에 대해 아무것도 모르기 때문입니다.
그 둘 중 어느 것도 건드리지 않고 processor를 splice해서 loop를 gate할 수 있습니다:
`GatedLLMContextAggregator` (`gated_llm_context.py:14`)는 **가장 최근의** `LLMContextFrame`을 붙들고
있다가 notifier가 발화하면 놓아주는 82줄이고, 오래된 것들은 queue에 쌓지 않고 버립니다 — loop가 call
stack이 아니라 splice point를 통과하기 때문에만 존재할 수 있는 hold-the-turn primitive입니다.

그리고 loop 전체를 아무것도 계측하지 않고 관측할 수 있습니다. §5.3 표의 모든 step은 processor 경계를
넘는 frame이고, 그것은 [[ch-11/read]]의 observer plane이 공짜로 본다는 뜻입니다. boson에서 이에 상응하는
가시성은 `run_agent_loop` 안에서 손으로 발화시키는 여섯 개의 hook event입니다(`ON_TURN_START`,
`PRE_LLM_CALL`, `PRE_TOOL_CALL`, `POST_TOOL_CALL`/`ON_ERROR`, `POST_LLM_CALL`, `ON_TURN_END`,
[[boson-agent-loop]] 기준) — 집필 시점에 고정된 집합이고, loop를 편집해야만 확장됩니다.

그것이 topology가 사주는 것입니다. §8은 그것이 당신의 기존 코드에 대해 치르는 대가입니다.

---

## 8. boson의 답: 세 가지를 모두 소유하는 함수 하나

### 8.1 entry point와 세 개의 alias

```python
# packages/basement/basement/loop/agent_loop.py:176, 184-186
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
async def run_agent_loop(runtime: AgentRuntime, user_input: str) -> AsyncIterator[StreamEvent]:
    ...
    ctx = runtime.context_manager
    api = runtime.conversation_api
    hooks = runtime.hook_registry
```

세 개의 local이 세계 전체를 alias합니다. `AgentRuntime` (`schemas/runtime.py:18`)은 `provider`,
`tool_registry`, `tool_router`, `permissions`, `cancellation_flag`, `skip_user_append`,
`exposed_meta_tools`, `on_tool_start`, `on_tool_end`를 포함하는 15개 field의 dataclass입니다.

Pipecat 쪽과 비교하십시오: `LLMContext.__init__`은 argument 세 개를 받고 field 세 개를 저장합니다.
그 비대칭이 chapter를 축소해 놓은 것입니다 — boson의 turn은 모든 것에 닿는 object 하나를 갖고,
Pipecat의 turn은 아무것에도 닿지 않는 object 하나와 모든 것에 닿는 graph를 갖습니다.

### 8.2 단 한 번의 inference, 그리고 그 뒤의 두 갈래

```python
# packages/basement/basement/loop/agent_loop.py:251-255
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
async for event in runtime.provider.stream(
    messages=ctx.get_messages(),
    system=ctx.get_system_prompt(),
    tools=tools,
):
```

`ctx.get_messages()`는 §2.5의 `deepcopy`이고, provider에게 건네집니다. `ctx.get_system_prompt()`는
**별도의 accessor**이고 — [[llm-service-context]]에 따르면 `LLMContext`에는 대응물이 아예 없습니다.
Pipecat의 system prompt는 `Service.Settings(system_instruction=...)`이거나, adapter가
`_extract_initial_system`으로 뽑아내는 선두의 `{"role": "system"}` message입니다
(`src/pipecat/adapters/base_llm_adapter.py:208`, anthropic·bedrock·gemini adapter에서 호출). 따라서
boson의 깔끔한 *system* 대 *messages* 분리는 Pipecat 쪽에서는 service setting이거나 adapter가 다시
추출해야 하는 message가 됩니다.

그다음, [[boson-agent-loop]] 기준, 두 branch가 사이에 arbiter 없이 turn을 결정합니다:

```python
# packages/basement/basement/loop/agent_loop.py:293, 347, 355-363
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
if tool_uses:
    ...                      # build the assistant tool_use message, execute the batch
    continue                 # :347 — back to the LLM
else:
    ctx.add_message("assistant", text)   # :355
    ...                                  # fire POST_LLM_CALL
    break  # Done — text response means end of turn      # :363
```

여기에 세 번째 exit이 더해집니다. tool batch 이후의 협조적(cooperative) cancellation:
`if cancellation_flag is not None and cancellation_flag.is_set: break` (`:343-345`).

> **Q3, boson의 답, 한 문장으로:** *`run_agent_loop`가 `agent_loop.py:363`에서 결정하고, 규칙은
> "text-only response는 turn이 끝났다는 뜻이다"이다.*

두 framework는 같은 behaviour에 도달합니다 — text-only 응답은 turn을 끝냅니다. Pipecat이 그렇게 되는
이유는 text-only 응답이 `FunctionCallResultFrame`을 만들지 않고 따라서 아무도 upstream으로 아무것도
push하지 않기 때문입니다. boson이 그렇게 되는 이유는 `break` statement가 그렇게 말하기 때문입니다.
behaviour는 일치하고, *결정의 위치(locus)*는 일치하지 않습니다. 그리고 당신이 포팅하는 것은 바로
그것입니다.

### 8.3 tool array는 iteration마다 다시 계산된다

```python
# packages/basement/basement/loop/agent_loop.py:224-237
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
if runtime.tool_router:
    from basement.metatool.registry import meta_tool_names
    exposed = (runtime.exposed_meta_tools
               if runtime.exposed_meta_tools is not None else meta_tool_names())
    tools = [s for s in runtime.tool_registry.get_all_specs() if s.name in exposed] or None
else:
    tools = runtime.tool_registry.get_all_specs() or None
```

`while` 안에 있으므로 **매** iteration마다 실행됩니다. Pipecat의 구조적 대응물은
`llm_service.py:723`의 `_sync_registered_tool_handlers(frame.context.tools)`이고 — 이것도 inference마다
실행됩니다. 모든 inference가 context frame 위에 실려 도착하기 때문입니다. 같은 주기, 반대 방향:
boson은 model이 **보는** 것을 registry로부터 다시 계산하고, Pipecat은 service가 **실행할 수 있는** 것을
model이 보는 것으로부터 다시 계산합니다.

Pipecat에서 session 중간에 광고 집합을 바꾸려면 `LLMSetToolsFrame`을 push하고, user aggregator가
그것을 처리합니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py:822-834
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

`_maybe_add_tool_change_messages`는 `LLMContextAggregatorPair`의 `add_tool_change_messages=True`
기능입니다 (`:2260`, `:2292-2294`): tool 집합이 대화 중간에 바뀌면 announcement message가 추가되어
model이 사라진 tool을 계속 부르지 않게 합니다. 그것이 `use_tool` indirection이 회피하려고 발명된 바로
그 hallucination 문제에 대한 Pipecat의 내장 답입니다 — pair 수준의 flag가 존재하는 이유가 정확히 양쪽
절반이 모두 참여하게 하기 위해서이고, 공유 context가 announcement가 정확히 한 번 착륙함을 보장합니다.

### 8.4 message schema collision: boson에는 `tool` role이 없다

[[boson-agent-loop]] 기준:

```python
# packages/basement/basement/schemas/message_schema.py:46
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
role: Literal["user", "assistant"]
```

role 두 개. 그것이 enum 전부입니다. tool result는 content block으로서 **user message 안에** 실려
다닙니다:

```python
# packages/basement/basement/loop/agent_loop.py:540
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
ctx.add_message("user", [result])     # result is a ToolResultBlock(tool_use_id, content, is_error)
```

그리고 assistant tool-use message는 `:316-328`에서 손으로 조립됩니다: `re.sub`로 반향된
`<system-reminder>` block을 제거하고, text가 있었다면 `TextBlock(text=text)`를 추가하고, call마다
`ToolUseBlock(id, name, input)` 하나씩.

그것은 Anthropic의 message shape입니다. Pipecat의 것은 OpenAI의 것입니다. tool call 하나를 나란히
놓고 무엇이 바뀌어야 하는지 세어 보십시오:

```python
# boson — Anthropic-shaped, two roles, blocks inside content
[
  {"role": "assistant", "content": [
      {"type": "text",     "text": "확인해 보겠습니다"},
      {"type": "tool_use", "id": "toolu_01", "name": "lookup_policy",
       "input": {"policy_id": "P-9931"}},
  ]},
  {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "toolu_01",
       "content": "{\"premium\": 43000}", "is_error": False},
  ]},
]
```

```python
# Pipecat — OpenAI-shaped, three roles, tool_calls array + a separate tool message
[
  {"role": "assistant", "content": "확인해 보겠습니다"},        # push_aggregation, :1685
  {"role": "assistant", "tool_calls": [                          # _handle_..._in_progress, :1753
      {"id": "call_01", "type": "function",
       "function": {"name": "lookup_policy",
                    "arguments": "{\"policy_id\": \"P-9931\"}"}},
  ]},
  {"role": "tool", "content": "{\"premium\": 43000}",            # placeholder, then rewritten
   "tool_call_id": "call_01"},                                   # in place at :2165
]
```

이름 바꾸기(rename)가 아닌 차이들:

| | boson | Pipecat |
|---|---|---|
| Role 수 | 2 (`user`, `assistant`) | 3+ (`user`, `assistant`, `tool`, `system`, developer message) |
| Tool result 운반체 | `ToolResultBlock`을 담은 `user` message | `tool_call_id`가 있는 `tool` message |
| Tool call 운반체 | assistant `content` 안의 `ToolUseBlock` | `content` **옆에** 있는 `tool_calls` array |
| Argument | dict (`input`) | **JSON string** (`arguments`) |
| Text + tool call | block 둘을 가진 assistant message 하나 | assistant message **두 개** |
| Error 신호 | block 위의 `is_error=True` | field 없음 — 문자열 content가 전부 |
| Validation | pydantic `Message` / `ContentBlock` | 없음; 평범한 dict |

표의 마지막 행이 포팅 중에 무는 것입니다: `is_error`를 놓을 자리가 없습니다. Pipecat의 error 관례는
문자열 자체입니다 —
`FUNCTION_CALL_ERROR_MESSAGE_TEMPLATE = "The function \`{function_name}\` failed and returned no result."` (`llm_service.py:299-301`). `is_error`에 기대어 downstream 분기를 하는 boson tool은,
당신이 직접 result payload에 그것을 인코딩하지 않는 한 그 신호를 잃습니다.

그리고 boson이 의도적으로 유지하는 strict-alternation 제약에 주목하십시오.
`_check_cancellation_and_emit` (`agent_loop.py:139-156`)에서 나온 것입니다: cancel entry는 tool result와
**같은 user message로 병합**됩니다. *"mirroring the merged-blocks shape of
`InterruptHandler.handle_barge_in`, which preserves strict assistant→user role alternation (some
models reject consecutive same-role messages)"* — 즉 일부 model은 연속된 같은 role의 message를
거부합니다. Pipecat의 shape는 관례적으로 연속된 `assistant` message 두 개를 emit합니다(text, 그다음
`tool_calls`). OpenAI 계열 model 아래에서는 괜찮습니다. strict-alternation model 아래에서는 아닙니다 —
그리고 boson의 provider list는 `anthropic`이 첫 번째입니다.

> **그대로 말합니다:** boson의 history를 `LLMContext`로 포팅하는 것은 rename이 아니라 **schema
> rewrite**입니다. `Message` / `ContentBlock` pydantic layer는 착륙 지점이 없습니다 — `LLMContext`는
> 평범한 dict를 저장하고 provider shaping을 adapter로 내려보냅니다.

> 💡 **쉬운 설명 — "text + tool call이 message 하나 vs 두 개"가 왜 중요한가**
> boson에서 "확인해 보겠습니다"와 tool 호출은 *같은 assistant turn*의 두 block입니다. Pipecat에서는
> 서로 다른 두 message입니다 — 하나는 assistant aggregator가 발화 후에 쓰고(§2.6), 다른 하나는
> tool이 시작될 때 씁니다(§2.4). 이건 단순한 표기 차이가 아닙니다: **누가 언제 쓰는가**가 다르고,
> 따라서 barge-in으로 발화가 잘리면 Pipecat에서는 text message가 아예 안 쓰일 수도 있는데
> `tool_calls` message는 쓰입니다(§4.5의 uninterruptible). 같은 대화가 두 schema에서 서로 다른
> 흔적을 남길 수 있다는 뜻이고, transcript 마이그레이션이 왜 함수 하나로 안 끝나는지의 이유입니다.

### 8.5 세 개의 gate, 그리고 Pipecat에는 하나의 slot만 있다

turn cap 다음으로, 이 chapter에서 두 번째로 큰 구조적 부재입니다. [[boson-tool-router]] 기준, boson은
세 개의 독립적 결정을 따로 유지합니다:

| Gate | 어디에 사는가 | 무엇을 결정하는가 | 왜 분리되어 있는가 |
|---|---|---|---|
| **Exposure** | `agent_loop.py:224-237`, `gateway/core.py:280-303`에서 pin | model이 tools array에서 **보는** 것 | `maximize_caching` 아래 `{"use_tool","use_skill"}`로 pin됨. *"the prompt-cache prefix is ordered tools → system → messages, so changing exposure on a stage transition invalidates the entire cached conversation"* — prompt-cache prefix가 tools → system → messages 순서라서, stage 전환에서 exposure를 바꾸면 cache된 대화 전체가 무효화되기 때문 |
| **Availability** | `_allowed_tools_var: ContextVar[set[str] | None]` (`metatool/router.py:32`), `dispatch` 안에서 검사 | **현재 stage에서 실행될 수 있는** 것 | `ContextVar`는 asyncio task 단위이므로, router instance 하나가 동시 session들을 서빙 |
| **Permission** | `PermissionChecker.check_tool(tool_name)`, `dispatch`의 첫 번째 (`router.py:103`) | **이 caller**가 애초에 실행할 자격이 있는 것 | `PermissionDeniedError`를 raise하고, stage와 독립적 |

dispatch 순서는, [[boson-tool-router]]에서 그대로 옮기면: (1) permission check → raise;
(2) stage gate — `_allowed_tools_var.get()`이 `None`이 아니고 이름이 그 안에도 없고 meta-tool도 아니면,
content가 `f"Tool '{tool_name}' is not available in the current stage."`인 `is_error=True`
`ToolResultBlock`을 반환; (3) 없으면 `ToolNotFoundError`; (4) `return await _invoke_handler(spec, arguments)`.

exposure gate가 `use_tool`이 애초에 존재하는 이유입니다. 그 광고된 schema는 의도적으로 generic합니다:

```json
{"type":"object",
 "properties":{"tool_name":{"type":"string","description":"Name of the tool to call"},
               "arguments":{"type":"object","description":"Arguments to pass to the tool"}},
 "required":["tool_name","arguments"]}
```

tools array에 항목 두 개, 모든 stage 전환에 걸쳐 byte 단위로 안정적이어서 prompt-cache prefix가 결코
무효화되지 않습니다 — 그 아래의 allowlist는 자유롭게 바뀌는 동안에도.

이제 Pipecat의 registry item을 다시 보십시오 (`llm_service.py:200-205`): `function_name`, `handler`,
`cancel_on_interruption`, `timeout_secs`, `cancellable_by_llm`, `auto_registered`. **permission field
없음. allowlist field 없음. caller identity 없음.** 그리고 `_run_function_call` (`:1552-1566`)은
이름을 handler로 해소하고 그것을 invoke합니다 — 해소와 invocation 사이에 가로챌 지점이 없습니다.

Pipecat의 gate는 하나, exposure이고, 그것은 registration과 융합되어 있습니다: context가 광고하는 것이
registry가 담는 것입니다(§4.2). availability와 permission은 다시 만들어야 하고, 놓을 자리는 정확히 두
곳입니다:

1. **각 handler 안에** — 같은 guard 22벌, 또는 22번 적용된 decorator 하나.
2. **catch-all 안에** — `register_function(None, guard_handler)`. 모든 call을 받아 permission과
   allowlist를 검사한 뒤 당신 자신의 table로 dispatch합니다. 이것이 §10.2이고, 구조적으로 `use_tool`과
   같은 수(move)입니다.

`use_tool` indirection 자체는 포팅해도 그대로 살아남는다는 점에 주목하십시오: 그것은
`LLMContext(tools=[use_tool, use_skill])`가 됩니다 — 진짜 dispatch table을 뒤에 둔 두 항목짜리 광고
집합. meta-tool 트릭은 portable합니다. 자리가 없는 것은 3-gate 분리입니다.

발견 하나 더, 부재도 증거이고 [[boson-tool-router]]가 표시해 두었기에 보고합니다:
`register_meta_tool(name)`은 `metatool/registry.py`에 존재하지만 `grep -rn "register_meta_tool"`은
**boson repo 어디에서도 call site를 찾지 못합니다**. 확장 지점이 만들어졌고 한 번도 쓰이지 않았습니다.
죽은 확장 지점을 새 framework로 포팅하지 마십시오. 살아 있는 둘만 포팅하십시오.

### 8.6 sequential 대 parallel, 이것은 당신에게 correctness 문제다

```python
# packages/basement/basement/loop/agent_loop.py:393
# (boson-agent, private; excerpt-attested via [[boson-agent-loop]])
async def _execute_tool_uses(runtime, tool_uses, hooks, api, ctx):
    for idx, tu in enumerate(tool_uses):
        ...
```

평범한 `for`입니다. [[boson-agent-loop]] 기준 엄격하게 sequential이고 parallelism이 없습니다. 그리고
[[boson-tool-router]] 기준, `tools/executor.py:30`은 module 수준의
`_SYNC_HANDLER_LOCK = threading.Lock()`을 들고 있습니다. 이유는 *"production tools do read-modify-write
on shared YAML/JSON files."*

Pipecat의 기본값:

```python
# src/pipecat/services/llm_service.py:303-308
    def __init__(
        self,
        run_in_parallel: bool = True,
        group_parallel_tools: bool = True,
        function_call_timeout_secs: float | None = None,
        enable_async_tool_cancellation: bool = False,
        settings: LLMSettings | None = None,
        **kwargs,
    ):
```

`run_in_parallel=True`. model이 한 응답에서 tool call 두 개를 emit하면
`_run_parallel_function_calls` (`:1534`)가 call마다 task를 만들고, 그것들은 모든 `await`에서
interleave합니다. read-modify-write하는 22개 tool을 그 기본값 위로 포팅하면, model이 그중 둘을 동시에
부르기로 결정하는 날 data race가 생깁니다.

고치는 방법은 keyword 하나입니다 — `OpenAILLMService(..., run_in_parallel=False)` — 이는
`_run_sequential_function_calls` (`:1542`)로 라우팅되고, 그것은 단일 runner task가 비우는 queue입니다.
단어 하나이고, **다시 명시해야만 하는** 단어 하나입니다. 기본값이 당신의 tool이 가정하는 것의 정반대이기
때문입니다. 주석이 아니라 migration checklist에 쓰십시오.

---

## 9. 네 개의 collision, 각각에 가격을 매김

위의 모든 것은 넷으로 압축됩니다. 각각에 실제로 중요한 단위로 비용이 매겨집니다: **새 코드가 얼마나
필요한가, 그리고 건너뛰면 무엇이 깨지는가.**

### 9.1 live list 대 `deepcopy`

- **collision:** `LLMContext.get_messages()`는 `self._messages`를 반환합니다 (`:245`);
  `ContextManager.get_messages()`는 `deepcopy(self._messages)`를 반환합니다 (`manager.py:47-51`).
- **각각이 자기 집에서 옳은 이유:** Pipecat은 identity가 필요합니다.
  `_update_function_call_result`가 반환된 reference를 *통해* 쓰기 때문입니다 (`:2158-2165`). boson은
  isolation이 필요합니다. provider adapter에게 건네진 복사본은 오염될 수 없기 때문입니다.
- **순진하게 포팅했을 때의 failure mode:** tool 결과가 조용히 model에 영영 도달하지 않습니다.
  exception 없음, log 줄 없음. §2.5가 여덟 단계를 추적합니다.
- **해소 비용:** *알고 있다면* 0줄입니다. 비용 전부가 지식입니다. migration 문서에 hard rule로
  쓰십시오: **절대 `get_messages`를 override하지 말 것, 절대 copy-on-read 하지 말 것, snapshot은 call
  site에서 뜰 것.**
- **이것을 잡아낼 test:** tool을 호출하고, result가 정산된 뒤
  `any(m.get("content") != "IN_PROGRESS" for m in ctx.get_messages() if m.get("role") == "tool")`을
  assert하십시오.

### 9.2 message schema rewrite

- **collision:** role 두 개 + content 안의 block 대 role 셋 이상 + `tool_calls` array; dict argument 대
  JSON-string argument; `is_error` 대 field 없음. §8.4에 표가 있습니다.
- **해소 비용:** history가 migration을 살아남아야 한다면(기존 session, transcript archive, evaluation
  set) 양방향 converter 하나씩, 그리고 `is_error`에 대한 결정. 옛 history를 load할 필요가 없다면 비용은
  22개 tool의 result shape뿐입니다.
- **잃는 것:** message에 대한 pydantic validation. `LLMContext`는 schema 없는 평범한 dict를 저장합니다.
  손으로 만든 message dict의 오타는 당신의 type checker가 아니라 provider의 400이 발견합니다.
- **얻는 것:** adapter layer(§4.7)가 지금 당신이 손으로 쓰는 provider shaping 여섯 개를 해 줍니다.
- **rename이 아닙니다.** 팀의 누군가가 이것을 "s/user/tool/"로 scoping한다면, 바로잡으십시오.

### 9.3 turn cap 없음 → counting processor를 직접 만든다

이건 실제로 줄 수로 가격을 매길 수 있으므로, 줄 수로 제시합니다. 직관이 아니라 위에서 확립된 frame
사실로부터 만드십시오:

- turn의 **첫** inference는 user aggregator에서 온 **downstream** `LLMContextFrame`으로 LLM에
  도착하고, LLM이 그것을 **소비**합니다(§3.2, 관측 3). 따라서 LLM *뒤에* 놓인 processor는 그것을 결코
  보지 못합니다.
- 모든 **re-prompt**는 assistant aggregator에서 온 **upstream** `LLMContextFrame`으로 도착하고,
  `assistant_agg → transport.output → tts → llm`을 여행합니다. **`llm`과 `tts` 사이에** 놓인 processor는
  그 경로 위에 앉아 있고, LLM보다 먼저 그것들을 전부 봅니다.
- `UserStartedSpeakingFrame`은 `transport.input`에서 downstream으로 여행하는 `SystemFrame`
  (`frames.py:1154`)입니다. LLM이 그것을 `else` branch로 forward하므로 (`base_llm.py:614-615`), 같은
  processor가 그것을 보고 reset에 사용할 수 있습니다.

그로부터, 세어야 할 것과 reset할 계기 둘 다를 보는 배치가 하나 나옵니다:

```python
# your code — e.g. lina/processors/tool_turn_cap.py
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ToolTurnCap(FrameProcessor):
    """Bound the number of tool→inference cycles inside a single user turn.

    Place BETWEEN the LLM service and the TTS service:

        [... user_agg, llm, ToolTurnCap(max_turns=8), tts, transport.output(), assistant_agg]

    Every re-prompt after a tool result travels UPSTREAM through this position on
    its way back to the LLM (llm_response_universal.py:1889), so counting them here
    counts exactly the cycles that Pipecat itself does not bound.
    """

    def __init__(self, *, max_turns: int = 8, **kwargs):
        super().__init__(**kwargs)
        self._max_turns = max_turns
        self._turns = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)   # non-negotiable; ch-01 §7.2

        if isinstance(frame, UserStartedSpeakingFrame):
            self._turns = 0

        elif isinstance(frame, LLMContextFrame) and direction == FrameDirection.UPSTREAM:
            self._turns += 1
            if self._turns > self._max_turns:
                logger.warning(
                    f"{self}: tool cycle cap hit ({self._turns} > {self._max_turns}); "
                    "dropping the re-prompt and closing the turn."
                )
                # Drop the frame: the LLM never runs again this turn.
                await self.push_frame(
                    TTSSpeakFrame("죄송합니다, 확인이 오래 걸리네요. 다시 말씀해 주시겠어요?"),
                    FrameDirection.DOWNSTREAM,
                )
                return

        await self.push_frame(frame, direction)
```

**코드만 읽지 말고 이것의 비용을 읽으십시오.** 실재하는 비용이고, 이것들 없이 채택해서는 안 됩니다:

1. **inference가 아니라 re-prompt를 셉니다.** turn의 첫 inference는 이 위치에서 보이지 않습니다.
   여기서 `max_turns=8`은 boson의 "총 8회 iteration"이 아니라 "tool cycle 8회"를 뜻합니다. 구성상
   하나 어긋납니다(off by one).
2. **frame을 버리는 것은 굶겨서(starvation) turn을 끝냅니다.** 다른 어떤 것도 LLM에게
   `LLMContextFrame`을 push하지 않으므로, `TTSSpeakFrame`이 없으면 bot은 그냥 침묵합니다. 그 발화
   대사는 장식이 아닙니다. 통화를 살려 두는 유일한 것입니다.
3. **context가 tool 도중 상태로 남습니다.** 마지막 tool result가 그 뒤에 assistant 응답 없이 history에
   앉아 있습니다. 다음 user turn은 `tool` message 바로 뒤에 `user` message를 append합니다. OpenAI는
   그것을 받아들입니다. ship 전에 당신의 provider가 그러는지 확인하십시오.
4. **reset 계기로 `UserStartedSpeakingFrame`을 쓰는 것은 선택이지 유일한 답이 아닙니다.**
   [[ch-06/read]]가 turn-strategy chain을 가르쳤습니다. 다른 turn-start 신호를 쓴다면 그것으로 reset
   하십시오. 잘못된 frame에서 reset하면 counter가 영영 reset되지 않거나(첫 turn 이후 모든 turn이 즉시
   cap됨) turn 중간에 reset됩니다(cap이 결코 발화하지 않음).
5. **더 온건한 변형이 있습니다.** 버리는 대신, nudge message를 append하고 tool choice를 지운 뒤 frame을
   forward하십시오:
   `frame.context.add_message({"role": "system", "content": "[Tool budget exhausted — answer the customer directly, without calling tools.]"})`.
   그것은 inference를 하나 더 쓰지만 통조림 문장 대신 진짜 문장을 얻습니다. OpenAI shape이므로, 의존하기
   전에 당신의 adapter가 history 중간의 `system` message를 어떻게 다루는지 확인하십시오.

대략 30줄 + test 하나. boson의 두 줄짜리 `while` 조건에 맞서서. 그것이 topology에 의한 loop closure의
정직한 가격입니다: bound를 넣을 loop header가 없으므로 bound를 processor로 다시 표현해야 합니다.

> 💡 **쉬운 설명 — 왜 하필 `llm`과 `tts` 사이인가**
> 배치가 논증 전체를 결정합니다. LLM 앞(user_agg와 llm 사이)에 두면 첫 inference는 보이지만 re-prompt는
> 보이지 않습니다 — re-prompt는 반대편에서 upstream으로 오기 때문입니다. LLM 뒤 아주 멀리
> (assistant_agg 뒤)에 두면 sink만 이웃이라 아무 소용이 없습니다. `llm`과 `tts` 사이는 upstream
> re-prompt가 LLM에 닿기 *직전*에 지나가는 유일한 지점이면서, 동시에 downstream `SystemFrame`
> (`UserStartedSpeakingFrame`)도 지나가는 지점입니다. 세어야 할 것과 reset 계기를 둘 다 보는
> position은 이것 하나뿐입니다.

### 9.4 세 개의 gate → 하나의 slot

- **collision:** §8.5. Pipecat의 `FunctionCallRegistryItem`에는 permission field도 allowlist field도
  없고, `_run_function_call`에는 이름 해소와 invocation 사이에 가로챌 지점이 없습니다.
- **해소 비용:** decorate된 handler 22개, 또는 catch-all 하나
  (`register_function(None, ...)`) + 당신 자신의 dispatch table — 그것은 `ToolRouter`를 다시 host한
  것입니다. `use_tool` / `use_skill` meta-tool indirection은 두 항목짜리 `LLMContext(tools=[...])`로
  그대로 포팅됩니다.
- **포팅되지 않는 것:** `ToolRegistry.discover_tools()` — `@tool` decorate된 함수의 filesystem
  discovery — 는 **Pipecat 대응물이 없습니다**. 그것은 `LLMContext(tools=[...])`에 건네줄 list를
  만들어내는 boson glue로 남습니다.
- **지금 잡아 둘 만한 미묘한 점:** `_allowed_tools_var` `ContextVar`는 asyncio task 단위이고, 그것이
  router 하나가 동시 session들을 서빙하는 방식입니다. Pipecat의 session 단위 isolation은
  `PipelineWorker` 단위이며([[ch-04/read]] §3), tool handler는 `ContextVar`가 아니라
  `params.pipeline_worker` / `params.app_resources`를 통해 자기 session에 도달합니다. Pipecat task
  바깥에서 설정된 `ContextVar`가 `_run_function_call`의 task 안에서도 보이는지는 그 task가 어디서
  생성되었는지에 달려 있습니다 — `create_task`는 생성 시점의 현재 context를 복사합니다. 가정하지
  마십시오. 의존하기 전에 test하십시오.

---

## 10. 세 번째 답, 이미 ship된 것: realtime_voice

resolution들에 앞서, 사실 하나를 사실로서 진술합니다.

[[ch-03/read]] §7.6은 realtime_voice의 agent slot이 정확히 한 가지 type을 yield하는 Protocol 하나임을
기록했습니다:

```python
# packages/realtime_voice/realtime_voice/protocols.py — shape as recorded in [[rtv-pipeline-session]]
class StreamingConversationAgent(Protocol):
    def stream(request) -> AsyncIterator[AgentTextDelta]
    async def cancel / close
```

`AgentTextDelta` 그리고 그 외에는 아무것도. tool call 아님. context object 아님. message 아님.
[[rtv-pipeline-session]]과 [[rtv-vs-pipecat-gap]] 기준, tool은 `packages/basement`와
`packages/gateway`에 살고, `GatewayConversationAgent.stream()` → `bridge.dispatch_transcript()`
(`agents/dental-w-tool-gateway/voice_server.py:163`)를 통해 도달됩니다. 그 slot에 대한 excerpt 자신의
요약: *"Pipecat assumes it owns the LLM call, whereas boson deliberately delegates to
`GatewayConversationAgent` so stages/rules/tools stay text-native."*

그것을 §1의 세 질문 위에 매핑하면 세 질문 모두에 대한 답이 **"voice package가 아니다"**입니다. voice
layer는 context를 들고 있지 않고, tool을 dispatch하지 않으며, iterator가 멈출 때 turn이 끝났음을
알게 됩니다.

그것이 구조적으로 아래의 세 번째 resolution — **bypass** — 이고, 이미 구현되었고, 이미 돌아가고 있고,
이미 한국어 전화 통화를 실어 나르고 있습니다. 그러니 §11.3을 읽을 때 당신이 읽는 것은 가설이 아닙니다:
branch `voice-chat-dev`에 존재하는 코드에 대한 기술입니다.

**그것이 좋은 결정이었는지에 대해서는 여기서 어떤 주장도 하지 않습니다.** 그것은 당신 repo의 상태에
대한 사실이고, "adopt"와 "wrap"이 각각 얼마의 비용인지를 바꿉니다 — 둘 다 오늘 작동하는 무언가를
되돌린다는 뜻이고, 그 되돌리기는 line item이기 때문입니다. [[ch-13/read]]가 가격을 매깁니다.

---

## 11. 세 개의 resolution, 가격을 매김. 어느 것도 recommended가 아님.

이 section이 무엇을 하는지에 대해 애매함이 없도록, 기본 규칙:

- 각 resolution은 **무엇을 쓰는가**, **무엇을 지우는가**, **무엇을 잃는가**, **무엇을 재구축해야
  하는가**, 그리고 **그것을 결정할 열린 질문들**을 받습니다.
- 어떤 option도 recommended로 표시되지 않습니다. 어떤 option도 더 낫다, 더 깔끔하다, 더 안전하다,
  옳다고 불리지 않습니다.
- 비용은 같은 단위로 진술되어 [[ch-13/read]]가 비교할 수 있게 합니다.

### 11.1 Adopt — tool들을 Pipecat의 loop 위로 포팅한다

**shape.** `LLMContext`가 message list가 됩니다. `LLMContextAggregatorPair`가 LLM service를 걸터앉습니다.
`run_agent_loop`는 삭제됩니다. 각 boson tool은 direct function이나 `FunctionSchema(handler=...)`가
됩니다. tool loop는 pipeline을 통해 닫힙니다.

**무엇을 쓰는가**

| 항목 | 크기 | Anchor |
|---|---|---|
| 22개 tool signature 변경: `handler(**args) → return v`가 `handler(params) → await params.result_callback(v)`가 됨 | 함수 22개, 또는 22번 적용되는 shim decorator 1개 | §4.4; count from [[boson-tool-router]] |
| LLM service constructor에 `run_in_parallel=False` | keyword 1개 | `llm_service.py:305` |
| `max_turns`를 복원할 `ToolTurnCap` processor | 약 30줄 + test | §9.3 |
| permission + allowlist 재host (decorator 또는 catch-all) | catch-all 1개 + dispatch table 1개, 또는 22번의 decoration | §9.4 |
| 옛 session을 load해야 한다면 message-history converter | 함수 2개 | §8.4 |
| System-reminder 주입 processor (`pop_pending_reminders`) | `FrameProcessor` 1개 | [[llm-service-context]] migration note; see [[custom-processor-guide]] |
| Hook 재host: event 6개 → `llm.event_handler(...)` + processor 배치 | 6곳 | [[boson-agent-loop]] |

**무엇을 지우는가.** `run_agent_loop` (561 L). `_reconcile_cancelled_tool_uses` (`:56`)와
`_await_tool_boundary` (`:113`)는 대체로 증발합니다 — [[boson-agent-loop]] 기준,
`FunctionCallCancelFrame` + `runner_item.settled` + 두 개의 `UninterruptibleFrame`이 그 약 80줄이 손으로
유지하던 것을 커버합니다. `basement/llm/*`의 손으로 쓴 provider tool-shaper 여섯 개는 12개 adapter로
대체됩니다(§4.7).

**무엇을 잃는가**

- turn을 한 곳에서 읽을 수 없게 됩니다. "왜 저 tool을 두 번 불렀지"를 debug하려면 함수 하나 대신
  pipeline과 frame log를 읽어야 합니다.
- §9.3을 만들기 전까지 `max_turns`.
- §9.4를 만들기 전까지 3-gate 분리.
- 일급 field로서의 `is_error` (§8.4).
- message에 대한 pydantic validation.
- public contract로서의 `AsyncIterator[StreamEvent]` — `TextDelta` / `ToolUseStart` / `MessageEnd`가
  `LLMTextFrame` / `FunctionCallInProgressFrame` / `LLMFullResponseEndFrame`이 되므로,
  `gateway/core.py:323`의 `async for event in run_agent_loop(...)` — `:305-389`의 system-reminder
  tail-buffer scrubber를 포함해 — 는 LLM과 TTS 사이의 `FrameProcessor`가 됩니다.

**무엇을 얻는가**

- 손으로 쓴 shaper 6개 대신 provider adapter 12개.
- `_reconcile_cancelled_tool_uses` 없이 §4.5의 interruption behaviour.
- §5.3의 모든 step이 observer plane([[ch-11/read]])에 공짜로 보임.
- `params.context` — tool handler가 history를 읽고 다시 쓸 수 있음.

**그것을 결정할 열린 질문들**

1. 살아 있는 Lina tool 중 control flow를 위해 `is_error`에 의존하는 것이 있는가, 아니면 그것은 오직
   model에게 text로 노출될 뿐인가?
2. `ctx.get_system_prompt()`가 `system_instruction` setting이나 선두 `system` message로는 감당할 수
   없는 곳에서 쓰이는가?
3. 옛 session이 load되어야 하는가, 아니면 깔끔한 절단(clean cut)이 허용되는가?

### 11.2 Wrap — boson의 loop를 담은 custom `FrameProcessor` 하나

**shape.** Pipecat이 transport, VAD, STT, TTS, 그리고 interruption cascade를 소유합니다.
`FrameProcessor` 하나가 LLM service가 있을 자리에 앉아, 누적된 user text를 받고, 내부적으로
`run_agent_loop`를 호출하고, `LLMTextFrame`들을 downstream으로 emit합니다. boson은 context, tool, gate,
turn counter를 계속 갖습니다.

**무엇을 쓰는가**

| 항목 | 크기 | Anchor |
|---|---|---|
| `BosonAgentProcessor(FrameProcessor)` — turn-end 신호를 소비하고, `run_agent_loop`를 호출하고, `StreamEvent` → frame으로 번역 | 약 150–250줄 | [[custom-processor-guide]]; contract in [[ch-01/read]] §7 |
| 그 안의 interruption 처리: `InterruptionFrame`에서 accumulator reset, 진행 중인 loop cancel | 약 30줄 | [[ch-08/read]] |
| `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` bracketing에서 boson의 `MessageEnd`로 가는 bridge | 약 20줄 | `base_llm.py:603, 613` |

**무엇을 지우는가.** boson에서는 아무것도. Pipecat 쪽에서는 `LLMContext`,
`LLMContextAggregatorPair`, `LLMService`, adapter layer, `run_function_calls`를 사용하지 않습니다.

**무엇을 잃는가**

- §11.1이 얻는 모든 것. adapter layer, `FunctionCallParams`, uninterruptible result frame, tool call의
  frame 수준 가시성. 당신이 직접 frame을 emit하지 않는 한 tool call은 [[ch-11/read]]의 observer plane에
  보이지 않습니다.
- aggregator pair의 turn-boundary machinery — [[ch-06/read]]의 turn-strategy chain은
  `LLMUserAggregator`에 배선되어 있으므로, 그것을 우회한다는 것은 "user의 turn이 끝났다"가 어디서
  오는지를 다시 결정해야 한다는 뜻입니다.
- Pipecat Flows([[ch-10/read]])와의 호환성. Flows는 당신이 쓰지 않고 있는 `LLMContext`를 조작해서
  conversation state를 구동합니다.

**무엇을 유지하는가.** `max_turns`. 세 개의 gate. `Message` schema. hook들. 22개 tool signature, 손대지
않음. `get_messages()`의 `deepcopy` — 이건 계속 옳습니다. live list에 의존하는 aggregator를 쓰지 않기
때문입니다.

**이 option의 구체적 위험, 이름을 붙여서**

- **cancellation seam이 위험 지점입니다.** [[boson-agent-loop]]는 `cancellation_flag`가 정확히 두
  곳에서만 읽힌다고 기록합니다 — `:344`의 tool batch 이후와 `:513`의 tool 하나 완료 이후 — 그리고
  **`TextDelta` 사이에서는 결코 읽히지 않습니다**. `gateway/interrupt/cancellation.py:171`이 그대로
  말합니다: `NOTE: Cooperative — tool runs to completion, then flag is checked.` Pipecat의 interruption은
  당신의 `process_frame` task를 그것이 주차되어 있는 `await` 지점에서 cancel합니다([[ch-08/read]]).
  그 지점이 `run_agent_loop`의 provider stream 안이라면, 정확히 그런 경우를 위해 설계된 복구 장치를
  가진 561줄 함수 한가운데에서 hard `CancelledError`를 받게 됩니다. 동작해야 *합니다*. 가장 먼저,
  가장 세게, 그리고 한국어 오디오로 test할 seam입니다.
- **turn 전체 동안 `process_frame` 하나 안에 있게 됩니다.** 모든 tool call, 모든 re-prompt가
  그렇습니다. [[ch-04/read]] §4가 processor의 two-task model에서 그것이 무엇을 뜻하는지 가르쳤습니다:
  오래 도는 `process_frame`은 그 지속 시간 내내 cancellable task를 점유합니다.

**그것을 결정할 열린 질문들**

1. `run_agent_loop` 도중의 hard `CancelledError`가 실제로 history를 복구 가능한 상태로 남기는가 —
   unit test가 아니라 진짜 barge-in 트래픽에서?
2. Flows([[ch-10/read]])를 원하는가? 그렇다면 이 option은 그것을 비싸게 만듭니다.
3. `LLMUserAggregator`가 pipeline에 없다면 turn-end 신호는 누가 소유하는가?

### 11.3 Bypass — Pipecat을 voice I/O 전용으로 쓴다

**shape.** Pipecat이 `transport.input()`, VAD, STT, TTS, `transport.output()`과 interruption cascade를
돌립니다. `LLMContext` 없음, aggregator pair 없음, LLM service 없음. processor 하나가 최종 transcript를
기존의 text contract를 통해 gateway로 내보내고, 돌아온 text를 TTS로 넣습니다. boson이 agent에 관한
모든 것을 계속 갖습니다.

**이것이 realtime_voice가 이미 구현한 shape입니다**(§10) — 다만 voice layer가 Pipecat이 아니라
realtime_voice입니다.

**어떤 Pipecat processor가 pipeline에 남는가** — figure가 열거하라고 하므로, 구체적으로:

```python
pipeline = Pipeline([
    transport.input(),          # WebRTC / WebSocket / telephony      — ch-05
    stt,                        # STTService                          — ch-06
    turn_detection_or_vad,      # if not folded into the aggregator   — ch-06
    boson_bridge,               # YOUR processor: transcript out, text in
    tts,                        # TTSService                          — ch-07
    transport.output(),         # MediaSender + clock task            — ch-07, ch-08
])
```

canonical §2.7 pipeline에서 사라진 것: `user_aggregator`, `llm`, `assistant_aggregator`. 일곱 자리 중
세 자리.

**무엇을 쓰는가**

| 항목 | 크기 | Anchor |
|---|---|---|
| `boson_bridge` processor: `TranscriptionFrame` → gateway; 돌아온 text → `LLMTextFrame`/`TTSSpeakFrame` | 약 100–150줄 | [[custom-processor-guide]] |
| turn-end 결정. `LLMUserAggregator`가 없으므로 | 선택한 strategy에 따라 다름 | [[ch-06/read]] |
| gateway의 `CancellationFlag`로 가는 당신 자신의 interruption forwarding | 약 20줄 | [[ch-08/read]] |

**무엇을 잃는가**

- §11.1의 gain column 전부, 여기에 Flows([[ch-10/read]]) 전체, 여기에 agent의 behaviour에 대한 어떤
  frame 수준 가시성도.
- pipeline middleware로서의 rule layer([[ch-12/read]])는 voice 절반에만 적용됩니다. agent 쪽 rule은
  boson에 남습니다.

**무엇을 유지하는가.** boson 전부, 손대지 않고. text-native contract. gate들. `max_turns`. schema.
그리고 [[ch-05/read]]부터 [[ch-08/read]]까지 목록화한 transport 폭, serializer, interruption cascade를
가진 voice layer.

**구체적 비용, 이름을 붙여서.** 당신은 framework 두 개를 돌리고 있습니다. Pipecat의 frame model은
boson 안으로 뻗지 않고, boson의 `StreamEvent` model은 Pipecat 안으로 뻗지 않으므로, bridge processor는
당신이 영원히 소유하는 번역 layer입니다. agent에게 중요한 모든 새 frame type — rule 위반, stage 전환,
barge-in 분류 — 은 손으로 그것을 통해 배관되어야 합니다.

**그것을 결정할 열린 질문들**

1. gateway로 나갔다 돌아오는 추가 hop의 실제 latency 비용은, 진짜 한국어 통화에서 측정하면 얼마인가?
   [[ch-11/read]]가 그것에 답하는 budget을 만듭니다.
2. gateway의 `CancellationFlag`가 Pipecat `InterruptionFrame`에 [[ch-08/read]]의 barge-in 목표를
   만족시킬 만큼 빠르게 반응하는가?
3. realtime_voice가 이미 이 shape를 구현했다면, Pipecat의 voice layer로 갈아 끼우는 것이 구체적으로
   무엇을 사주는가 — 그리고 그것은 이 chapter가 아니라 [[ch-13/read]]의 질문입니다.

### 11.4 세 개의 비용, 나란히

| | **Adopt** | **Wrap** | **Bypass** |
|---|---|---|---|
| 새로운 Pipecat 쪽 코드 | 약 30 L cap + gate + reminder processor | processor 하나에 약 200–300 L | 약 120–170 L bridge |
| 삭제되는 boson 코드 | `run_agent_loop` (561 L) + 약 80 L cancellation 복구 + provider shaper 6개 | 없음 | 없음 |
| 변경되는 tool signature | 22 | 0 | 0 |
| `max_turns` | processor로 재구축 | 유지 | 유지 |
| 세 개의 gate | catch-all 또는 decorator로 재구축 | 유지 | 유지 |
| Message schema | 다시 씀 | 유지 | 유지 |
| adapter layer 사용 | 예 (adapter 12개) | 아니오 | 아니오 |
| Flows([[ch-10/read]]) 사용 가능 | 예 | 비쌈 | 아니오 |
| Observer plane이 tool call을 보는가 | 예 | 당신이 emit한 것만 | 아니오 |
| agent의 interruption | framework 제공 | 당신이 증명해야 할 seam | 당신이 증명해야 할 seam |
| 당신 repo에 이미 구현되어 있는가 | 아니오 | 아니오 | **예, voice layer가 realtime_voice인 형태로** |

**이 표의 어떤 행도 판정이 아니며, 마지막 행은 당신 repository에 대한 사실이지 논증이 아닙니다.**
[[ch-13/read]]가 이것을 채점합니다. 이 chapter는 미해결 상태로 넘깁니다. 그것이 임무 전부입니다.

---

## 12. Lina를 위한 세 개의 framework-extension move

당신의 가장 강한 mode를, 위의 mechanism들에 적용한 것입니다. 각각은 이번 주에 스케치할 수 있는
design이고, 어느 것도 resolution을 먼저 고를 필요가 없습니다.

**Move 1 — permission kernel로서의 catch-all.** `register_function(None, handler)`는 registry가
이름으로 매칭하지 못한 **모든** call을 받는 handler를 설치합니다 (`llm_service.py:1467-1468`, 그리고
실행 시점에 다시 `:1554-1555`). 그것을 `FunctionSchema(handler=None)` — advertise 전용 schema(§4.1,
경로 2) — 와 결합하면 Pipecat-native `ToolRouter`를 얻습니다: handler 없는 schema N개를 광고해서
model이 진짜 parameter schema를 가진 진짜 tool 이름을 보게 하고, catch-all은 정확히 하나만 등록하고,
그 안에서 permission → allowlist → dispatch를 boson의 순서로(§8.5) 수행합니다. 세 개의 gate를 되찾고
**동시에** `use_tool` indirection 대신 prompt에 진짜 tool 이름을 유지합니다. 비용: tool 단위
`timeout_secs`와 `cancel_on_interruption`을 잃습니다. 그것들이 registry item 위에 살고 registry item이
하나뿐이기 때문입니다. 만들기 전에 답할 design 질문: Lina에게 tool 단위 timeout이 gate 분리보다 더
가치 있는가?

**Move 2 — pin된 prefix를 가진, `LLMSetToolsFrame` 구동 stage machine.** boson은 `maximize_caching`
아래 광고 집합을 두 항목으로 pin합니다. *"the prompt-cache prefix is ordered tools → system →
messages"*이기 때문입니다(§8.5). Pipecat의 `LLMSetToolsFrame` 경로(§8.3)는 session 중간에 광고 집합을
바꾸고, aggregator pair의 `add_tool_change_messages=True` (`:2260`)는 model이 사라진 것을 그만 부르도록
announcement를 append합니다. 이 두 사실은 서로 긴장 관계에 있습니다: tool 변경을 알리는 것은 **message를
append**하는 것이고, message는 cache prefix에서 tools block *뒤에* 있습니다 — 따라서 announcement는
cache 관점에서 비용이 0인 반면, `tools`를 바꾸는 것은 prefix 전체를 날립니다. 그로부터 정확하게 진술할
수 있는 design이 나옵니다: **byte 단위로 안정적인 superset을 광고하고, availability 변경은 message로
알리고, allowlist로 강제한다.** prompt cache 보존, model은 정직하게 유지, gate는 온전. 한 페이지짜리
design으로 써 보고 token을 확인하십시오.

**Move 3 — end-of-call primitive로서의 `run_llm=False`.** §5.2는 handler가
`FunctionCallResultProperties(run_llm=False)`로 정산하면 re-prompt가 일어나지 않음을 보여주었습니다.
그것이 Pipecat에서 *tool*이 "turn이 끝났다"고 말할 수 있는 유일한 곳입니다. tele-sales agent에게
`end_call`, `transfer_to_human`, `schedule_callback`은 정확히 답변 없이 turn을 끝내야 할 tool들이고 —
오늘 boson에서는 각각이 model을 설득해 tool 호출을 그만두게 함으로써만 turn을 끝냅니다.
`run_llm=False`를 쓰는 세 handler를 [[ch-04/read]] §8의 `EndFrame` / `CancelFrame` shutdown 경로와 함께
스케치하면, model behaviour에 의존하지 않는 결정론적 end-of-call을 얻습니다. 기계적으로 boson에는 오늘
대응 hook이 없습니다 — 그 turn은 `agent_loop.py:363`에서 text-only 응답에 의해 끝나고, model이 그것을
생산하도록 설득되어야 합니다. 그 차이는 두 mechanism에 대한 사실이지 점수가 아닙니다. 세 resolution
모두에서의 후보로서 [[ch-13/read]]의 give-back list에 올라갑니다.

---

## 13. 머릿속에 담아 둘 것

열두 개의 사실. 이 chapter에서 다른 것을 전혀 기억하지 못하더라도 이것들은 기억하십시오. 그리고 그중
하나하나가 이미 본 명령으로 확인 가능하다는 것도 기억하십시오.

1. `LLMContext`는 field 세 개이고 (`llm_context.py:110-112`) application이 그것을 소유합니다.
2. `create_context_aggregator()`는 이 commit에 **존재하지 않습니다**. `src/`와 `examples/`에서 hit 0.
3. `register_direct_function`은 1.4.0부터 `@deprecated`입니다 (`llm_service.py:982-984`).
4. `get_messages()`는 **live list**를 반환합니다 (`:245`); `truncate_large_values=True`만 복사합니다.
5. live list를 반환하는 **이유는** `_update_function_call_result`가 그것을 통해
   `message["content"] = result`를 대입하기 때문입니다 (`:2158-2165`).
6. boson의 `get_messages()`는 `deepcopy(...)`를 반환합니다. 그 방어를 포팅하면 tool path가 조용히
   망가집니다.
7. `LLMContextFrame`당 completion 하나. `:599` super, `:601` isinstance, `:603` start frame,
   `:604` metrics, **`:605` completion**.
8. `:601`은 **direction을 검사하지 않고**, 그래서 upstream frame이 같은 service를 다시 prompt합니다.
9. loop는 `llm_response_universal.py:1889`에서 닫힙니다 —
   pipeline의 마지막 processor에서 `await self.push_context_frame(FrameDirection.UPSTREAM)`.
10. `grep -rn "max_turns\|max_iterations\|max_tool_calls" src/pipecat/` → **0**. 유일한 bound는 call
    단위 timeout입니다.
11. boson은 gate 셋을 유지합니다 — exposure, availability (`_allowed_tools_var`), permission
    (`PermissionChecker`). `FunctionCallRegistryItem`에는 그중 어느 것의 slot도 없습니다.
12. `run_in_parallel`의 기본값은 `True`입니다. boson의 tool들은 sequential을 가정하고
    `threading.Lock`을 쥡니다. 다시 명시하십시오.

그리고 Q3에 대해 system당 한 문장:

- **Pipecat:** 아무도 `LLMContextFrame`을 upstream으로 push하지 않을 때 turn이 끝납니다. 부재(absence).
- **boson:** `agent_loop.py:363`, `break  # Done — text response means end of turn`에서 turn이
  끝납니다. 진술(statement).
- **realtime_voice:** agent의 `AsyncIterator[AgentTextDelta]`가 멈출 때 turn이 끝납니다. 위임
  (delegation).

---

## 다음 챕터로

이 chapter는 세 가지를 앞으로 넘깁니다.

**해소된 mechanism 하나.** 이제 파일-및-줄 해상도로 압니다: 누가 context를 소유하는지(application이
생성하는 평범한 object), 누가 tool을 dispatch하는지(service가, context가 채우고 pruning하는 registry
로부터), 그리고 누가 turn이 끝났다고 결정하는지(아무도 — upstream frame의 부재로 끝납니다). 그것은
Pipecat에 대한 의견이 아니라 `llm_context.py`, `llm_response_universal.py`, `llm_service.py`,
`base_llm.py`를 읽은 결과입니다.

**두 개의 부재와 하나의 역전, 가격이 매겨진 채로.** turn cap 없음(재구축에 약 30줄, §9.3에 네 개의
비용이 명명됨). registry에 permission이나 allowlist slot 없음(catch-all 하나 또는 22번의 decoration,
§9.4). 그리고 `deepcopy` 대 live list 역전 — 비용이 0줄이면서 동시에 주의력 전부인 것(§9.1).

**미해결의 선택 하나.** Adopt, wrap, bypass — §11에서 가격이 매겨졌고, **어느 것도 recommended가
아니며**, realtime_voice가 이미 bypass shape를 구현하고 있다는 사실이 기록되었습니다. [[ch-13/read]]가
그것을 채점하고, 그것은 [[ch-10/read]]부터 [[ch-12/read]]까지가 나머지를 보여준 뒤에야 가능합니다.

다음은 [[ch-10/read]] — **Pipecat Flows: pipeline 바깥에 사는 state machine.** 이것이 이 chapter에
의존하는 이유는 §2에서 이미 예측할 수 있습니다: Flows는 방금 aggregator 둘이 서로 다투는 것을 지켜본
바로 그 `LLMContext`를 조작해서 대화를 구동하고, §8.3의 `LLMSetToolsFrame` 경로로 광고된 tool 집합을
교체합니다. 모든 node 전환이 공유 object에 대한 `set_messages` / `set_tools` 쌍입니다. live list가
load-bearing이라는 §2.3의 주장을 완전히 믿지 않았다면, [[ch-10/read]]가 믿게 만들 것입니다 —
자기가 소유하지 않은 context를 mutate하는 state machine은, 다른 누가 reference를 들고 있는지 아는지
여부에 따라 우아하거나 끔찍하기 때문입니다.

§8.5의 세 gate를 가지고 오십시오. Flows에도 stage 개념이 있고, 그 비교가 요점입니다.
