<!-- chapter: ch-02
     track: internals
     kind: content
     title: Architecture Overview: WorldState, the Episode Loop, and Toolset Modes
     deps: [ch-01]
     sources: [[automationbench-harness]], [[automationbench-overview]]
     figures: figures/episode-flow.html
-->

# 02장 — Architecture Overview: WorldState, the Episode Loop, and Toolset Modes

> **핵심 통찰.** AutomationBench는 모든 SaaS backend를 단일 in-process Pydantic object로 시뮬레이션함으로써 결정론적이고 sub-1% run variance를 달성한다 — HTTP 서버도, 데이터베이스도, subprocess도 없다. 전체 "world"는 typed Python value로, 모든 tool call에 참조(reference)로 전달된다. network fault도, race condition도, 외부 state도 없으므로, 동일한 initial state는 동일한 model output 아래에서 항상 동일한 trajectory를 만들어 낸다.

> **가이드라인.** 재현 가능한 agent benchmark를 원한다면, typed state로 world를 in-process에서 시뮬레이션하고, model-facing schema *뒤에서* tool에 inject하라. model은 `world` parameter를 절대 보지 못한다; harness가 dispatch 시점에 inject한다. 이 decoupling이 바로 어떤 tool 구현도 변경하지 않고 discovery, execution, API-style access를 세 개의 clean mode로 ablate할 수 있게 해 주는 것이다.

---

## 1. What `verifiers` Provides and What AutomationBench Adds

AutomationBench는 `verifiers` library 위에 만들어진다. entry point는 다음과 같다:

```python
# automationbench/runner.py
class AutomationBenchEnv(vf.StatefulToolEnv):
```

`vf.StatefulToolEnv`는 대부분의 agent benchmark에 필요한 scaffolding을 제공한다: dataset iteration loop, `add_tool`이 있는 tool registry, agent loop의 `max_turns` guard, 그리고 tool call을 dispatch하고 `ToolMessage`를 반환하는 `env_response` method. AutomationBenchEnv가 그 위에 추가하는 것은:

- **WorldState** — 44개 SaaS backend의 in-process Pydantic simulation(§2 참조).
- **세 가지 toolset mode** — `zapier`, `limited_zapier`, `api`(§4 참조).
- **`world` injection trick** — `args_to_skip`을 통해 내부 `world` argument를 model-facing JSON schema에서 숨김(§5 참조).
- **Per-task tool filtering** — `setup_state`가 현재 task의 `info.zapier_tools`에서 허용된 tool만으로 전체 registry에서 tool list를 좁힘.
- **Meta-message compression** — `_compress_meta_messages`가 `execute_tool`이 호출되면 오래된 search 결과를 짧은 summary로 재작성하여 긴 episode에서 context bloat을 방지함.

constructor signature는 핵심 configuration knob를 담고 있다:

```python
# automationbench/runner.py  L49-59
def __init__(
    self,
    dataset: Dataset,
    rubric: vf.Rubric,
    tools: list[Callable] | None = None,
    max_turns: int = 25,
    allow_all_tools: bool = False,  # Enforce per-task tool restrictions
    toolset: str = "zapier",
    use_meta_tools: bool | None = None,  # None = infer from toolset
    search_top_k: int | None = None,  # Hard cap on search_tools top_k (None = no cap)
    **kwargs,
):
```

`max_turns`는 **25**가 기본값이며, 50이 아니다. system prompt는 model에게 50 turn 안에 task를 완료하도록 지시하지만, 이는 soft prompt-level budget이다; episode를 종료하는 hard guard는 `max_turns=25`다. 문서와의 전체 reconciliation은 §6을 참조하라.

---

## 2. WorldState: One In-Process Pydantic Root Model

시뮬레이션된 world는 `schema/world.py`에 정의된 단일 Pydantic `BaseModel`이다. 그 root:

```python
# automationbench/schema/world.py  L70-73
class WorldState(BaseModel):
    """Root world state containing all app states."""

    model_config = ConfigDict(extra="forbid")
```

`extra="forbid"`는 load-bearing이다. 이는 dataset row의 `initial_state` dict로 `WorldState`를 construct할 때, dict에 선언된 field와 일치하지 않는 key가 하나라도 있으면 즉시 `ValidationError`를 raise한다는 의미다. 이것이 benchmark task를 정직하게 유지하는 첫 번째 safety check다 — 오타가 난 app 이름은 episode start에 잡히며, 조용히 무시되지 않는다.

모든 field는 typed sub-state class와 `default_factory`로 선언된다:

```python
# automationbench/schema/world.py  L76-128 (selected lines)
    meta: WorldMeta = Field(default_factory=WorldMeta)
    airtable: AirtableState = Field(default_factory=AirtableState)
    asana: AsanaState = Field(default_factory=AsanaState)
    ...
    gmail: GmailState = Field(default_factory=GmailState)
    google_calendar: GoogleCalendarState = Field(default_factory=GoogleCalendarState)
    google_sheets: GoogleSheetsState = Field(default_factory=GoogleSheetsState)
    hubspot: HubSpotState = Field(default_factory=HubSpotState)
    salesforce: SalesforceState = Field(default_factory=SalesforceState)
    slack: SlackState = Field(default_factory=SlackState)
    ...
    zendesk: ZendeskState = Field(default_factory=ZendeskState)
```

44개의 app sub-state가 SaaS category를 커버한다: CRM(Salesforce, HubSpot), productivity(Notion, Trello, Asana, Monday, Jira, Confluence, Airtable), communication(Slack, Gmail, Twilio, Zoom), marketing(Mailchimp, Buffer, Facebook, LinkedIn, Google Ads), HR(BambooHR, Recruitee), finance(QuickBooks, Xero, Wave), support(Zendesk, Freshdesk, Intercom, Gorgias, HelpScout, Reamaze, Zoho Desk, Hiver, HelpCrunch).

각 sub-state는 typed record list를 보유한다(예: `SalesforceState.contacts: list[Contact]`). Tool은 메모리 안에서 `world.<app>.<collection>`을 mutate한다 — persistence layer가 없다. object가 모든 tool call에 참조로 전달되기 때문에, `salesforce_contact_update` 내부의 mutation은 `world.salesforce.contacts`를 읽는 다음 tool call에 즉시 반영된다. 이것이 simulation을 결정론적으로 만들고, rubric evaluation을 trivially cheap하게 만드는 이유다: episode가 끝나면 grader가 최종 `world` object를 직접 읽는다.

`WorldMeta` sub-model은 `current_time` timestamp(`datetime.now(utc)` 기본값)와 noise-injection layer에서 사용되는 `no_same_sender_noise` flag를 갖는다. 둘 다 동일한 `extra="forbid"` configuration을 공유한다.

---

## 3. Episode Lifecycle

전체 pipeline의 시각적 walkthrough는 [episode-flow diagram](figures/episode-flow.html)을 참조하라.

### 3.1 Dataset Row: prompt + info

HuggingFace dataset의 각 task row는 다음 형태를 갖는다:

```
{
  "prompt":  "<trigger message, e.g. an email notification>",
  "info": {
    "initial_state":  { ... WorldState dict ... },
    "zapier_tools":   [ "gmail_send_email", "slack_send_message", ... ],
    "assertions":     [ { "type": "...", ... }, ... ],
    "invariants":     [ ... ]
  }
}
```

HuggingFace는 모든 row에 걸쳐 dataset schema를 정규화한다: 어느 row에라도 존재하는 모든 key가 `None` default와 함께 모든 row에 추가된다. Pydantic의 `default_factory` pattern은 `None`이 명시적으로 전달될 때 깨진다 — factory가 bypass되고 `None`이 typed field로 전파된다. AutomationBench는 Pydantic을 건드리기 전에 이를 수정한다:

```python
# automationbench/runner.py  L22-35
def strip_none_values(obj):
    """
    Recursively strip None values from nested dicts and lists.

    HuggingFace Dataset normalizes schemas across rows, adding all possible keys
    and setting missing values to None. This breaks Pydantic's default_factory
    since None is passed instead of the field being omitted.
    """
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    else:
        return obj
```

이 함수는 Pydantic construction 이전에 `initial_state`와 `assertions` 양쪽에 적용된다(runner.py L149-153).

### 3.2 setup_state

`setup_state`는 agent loop가 시작되기 전에 episode당 한 번 호출된다. 세 가지 작업을 수행한다:

```python
# automationbench/runner.py  L136-183
async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
    """Initialize per-task world state and filter tools."""
    state = await super().setup_state(state, **kwargs)

    # Get task info (deserialize from JSON if it's a string)
    info = state.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
        state["info"] = info

    # Initialize world state
    initial_state_dict = strip_none_values(info.get("initial_state", {}))

    # Also strip None values from assertions (same HuggingFace normalization issue)
    if "assertions" in info:
        info["assertions"] = [strip_none_values(a) for a in info["assertions"]]
    world = WorldState(**initial_state_dict)
    state["world"] = world
    state["initial_state"] = copy.deepcopy(initial_state_dict)
    ...
```

L156의 `copy.deepcopy(initial_state_dict)`는 rubric grader를 위한 guard다: agent가 무언가를 하기 전에 이미 참이었던 assertion("free assertion" — 잠재적인 reward-hacking vector)을 감지하기 위해 원본 state가 필요하다. episode 동안 `state["world"]`를 mutate해도 이 baseline copy를 손상시켜선 안 된다.

`WorldState`를 build한 뒤, `setup_state`는 tool list를 필터링한다:

```python
# automationbench/runner.py  L159-181
    if self.use_meta_tools:
        # Meta-tools mode: model always gets the discovery tools
        filtered_tools = self._all_tool_defs
    elif self.allow_all_tools or self.toolset == "api":
        filtered_tools = self._all_tool_defs
    else:
        # If tools not specified, model gets NO tools (empty array)
        allowed_tools = info.get("zapier_tools", [])

        # Validate tool names - fail loudly if unknown tool specified
        all_tool_names = {t.name for t in self._all_tool_defs}
        unknown_tools = set(allowed_tools) - all_tool_names
        if unknown_tools:
            raise ValueError(
                f"Unknown tools specified in task: {unknown_tools}. Available: {all_tool_names}"
            )

        filtered_tools = [
            tool for tool in self._all_tool_defs if tool.name in allowed_tools
        ]

    state["tool_defs"] = filtered_tools
```

test suite가 filtering contract를 확인한다: 빈 `zapier_tools` list는 빈 tool array를 만들고(test_runner.py L65-82), 알 수 없는 tool 이름은 메시지에 `"Unknown tools"`가 포함된 `ValueError`를 raise한다(L85-102).

### 3.3 Agent Loop

loop는 `vf.StatefulToolEnv`가 구동한다. 각 turn에서:

1. model이 전체 message history와 `state["tool_defs"]`와 함께 호출된다.
2. model이 tool call을 emit하면 `env_response`가 호출된다.
3. `env_response`는 `_extract_usage_and_debug`를 호출하여(`prompt_tokens`와 `completion_tokens`를 `state["_usage"]`에 누적) `super().env_response`에 위임한다.
4. dispatch 전에 `update_tool_args`가 숨겨진 `world` argument를 재inject한다(§5 참조).
5. `zapier` mode에서 `_compress_meta_messages`는 결과로 나온 `ToolMessage`를 후처리한다: 이미 dead weight가 된 이전 turn의 `search_tools` 결과를 `[Previously found: name1, name2]`로 재작성하여 긴 episode에서 token을 절약한다.
6. model이 tool call 없는 response를 생성하거나 `max_turns`(기본값 25)에 도달하면 loop가 종료된다.

`env_response` override 전문:

```python
# automationbench/runner.py  L296-308
async def env_response(
    self,
    messages: vf.Messages,
    state: vf.State,
    **kwargs: Any,
) -> vf.Messages:
    """Process tool calls. In meta-tools mode, compress old search results."""
    # Extract usage/debug from the latest model response before processing tool calls
    self._extract_usage_and_debug(state)

    tool_messages = await super().env_response(messages, state, **kwargs)

    return tool_messages
```

### 3.4 Rubric Grading

loop가 종료되면 `verifiers` framework가 rubric을 호출한다. AutomationBench rubric은 `state["world"]`(최종 mutated state)와 `state["initial_state"]`(deep-copied baseline)를 읽고 각 assertion을 평가한다. Partial credit이 지원된다: 5개 assertion 중 3개를 완료한 task는 0.6 점수를 받는다.

### 3.5 export_results

run의 모든 episode가 완료된 후 `export_results`가 호출된다:

```python
# automationbench/export.py  L28-37
def export_results(
    outputs: list[dict[str, Any]],
    usage: RunUsage,
    model: str,
    domains: list[str],
    output_path: Path | str | None = None,
    duration_seconds: float | None = None,
    toolset: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
```

per-task score, per-assertion 결과, 전체 message history, token count, 그리고 cost data를 단일 JSON structure로 수집한다. 이 output이 benchmark visualizer를 구동한다. Token count는 character-count fallback estimate보다 `state["_usage"]`(`_extract_usage_and_debug`가 turn별로 누적)를 우선한다. `benchmark_version` field는 `pyproject.toml`에서 읽힌다 — run 간 version drift를 방지하는 single source of truth다.

---

## 4. The Three Toolset Modes

toolset은 model이 어떤 tool을 보는지와 무엇을 테스트하는지를 결정한다. 이것이 AutomationBench의 primary ablation axis다.

### 4.1 `zapier` — Two Meta-Tools (Default)

```python
# automationbench/runner.py  L79-85
    if self.use_meta_tools:
        from automationbench.tools.zapier.meta import execute_tool, make_search_tools, search_tools

        # Register only the 2 meta-tools for tool discovery
        actual_search = make_search_tools(max_top_k=search_top_k) if search_top_k is not None else search_tools
        self.add_tool(actual_search)
        self.add_tool(execute_tool, args_to_skip=["world"])
```

model은 정확히 **두 개**의 tool을 본다:

- `search_tools(query, top_k=5)` — tool corpus에 대한 BM25 search로, `{name, description, parameters}` entry의 JSON list를 반환한다. model은 어떤 tool이 존재하는지 discover하기 위해 먼저 이것을 호출해야 한다.
- `execute_tool(tool_name, arguments)` — 전체 registry를 통해 dispatch하여 이름으로 지정된 tool을 실행한다.

이것이 headline mode다: *discovery*(model이 올바른 tool을 찾을 수 있는가?)와 *execution*(model이 올바르게 호출할 수 있는가?) 양쪽을 테스트한다. 논문에서 "two tools"라고 할 때는 이 mode를 의미한다.

`use_meta_tools`는 `toolset == "zapier"`일 때 자동으로 `True`가 되고, `limited_zapier`와 `api`에서는 `False`가 된다(runner.py L73-77).

### 4.2 `limited_zapier` — Named Tools Filtered Per Task

```python
# automationbench/runner.py  L86-93
    else:
        # Register tools based on toolset selection
        tool_list = API_TOOLS if toolset == "api" else ALL_TOOLS  # limited_zapier also uses ALL_TOOLS (filtered per-task in setup_state)
        for tool in tool_list:
            # Auto-detect args_to_skip: skip 'world' only if the function accepts it
            sig = inspect.signature(tool)
            args_to_skip = ["world"] if "world" in sig.parameters else []
            self.add_tool(tool, args_to_skip=args_to_skip)
```

`limited_zapier` mode에서는 약 400개의 named tool이 env construction 시점에 모두 등록되지만, `setup_state`가 per-task view를 `info.zapier_tools`에 나열된 것들로만 필터링한다. model은 full signature를 가진 named function을 본다 — discovery 단계가 필요 없다.

이 mode는 **discovery skill**에서 **execution skill**을 분리한다: model이 `zapier`보다 `limited_zapier`에서 훨씬 낮은 점수를 받는다면, 병목은 tool discovery이지 argument construction이 아니다. `test_setup_state_filters_tools` test가 이를 직접 검증한다:

```python
# tests/test_runner.py  L41-62
async def test_setup_state_filters_tools(self):
    env = AutomationBenchEnv(dataset=dataset, rubric=rubric, allow_all_tools=False, toolset="limited_zapier")
    state = cast(vf.State, {
        "info": {
            "initial_state": {},
            "zapier_tools": ["salesforce_query"],  # Only one tool
        }
    })
    state = await env.setup_state(state)
    tool_names = [t.name for t in state["tool_defs"]]
    assert tool_names == ["salesforce_query"]
```

### 4.3 `api` — Three REST-Shaped Tools

```python
# automationbench/tools/api/__init__.py
API_TOOLS = [api_search, api_fetch, base64_encode]
```

model은 REST API interface를 미러링하는 세 개의 tool을 본다:

- `api_search` — tab-separated endpoint index(`schemas/index.txt`)에 대한 BM25 search로, `.jsonc` schema file에서 lazily rebuild된다. model이 keyword로 endpoint를 검색한다.
- `api_fetch(method, url, params, body)` — REST-style call을 실행한다. harness가 `_url_to_internal_path()`를 통해 URL을 올바른 in-process world mutation으로 라우팅한다. 중요하게, routing layer는 약 15개의 알려진 model URL hallucination을 정규화한다(예: `slack.googleapis.com` → 올바른 Slack router). 따라서 URL 오타가 task failure로 채점되지 않는다.
- `base64_encode` — base64 encoding이 필요한 payload 구성을 위한 utility.

`api` mode는 다른 skill profile을 테스트한다: model이 function name과 typed parameter schema가 아닌 HTTP method, URL 구조, request/response shape에 대해 추론해야 한다. Zapier-style function calling에는 강하지만 REST semantics에 약한 model(또는 그 반대)은 mode 간에 체계적인 gap을 보일 것이다.

### 4.4 What Each Mode Ablates

| Mode | Tools visible | Tests | Isolates |
|------|---------------|-------|---------|
| `zapier` | `search_tools` + `execute_tool` | Discovery + Execution | Neither: joint score |
| `limited_zapier` | ~N named functions (per-task) | Execution only | Discovery (difference vs zapier) |
| `api` | `api_search` + `api_fetch` + `base64_encode` | REST semantics | Abstraction style |

---

## 5. The `world` Injection Trick

모든 tool 구현은 `world: WorldState`를 parameter로 받는다 — tool이 시뮬레이션된 backend를 읽고 mutate하는 방법이다. 하지만 model이 보는 JSON schema에 복잡한 Pydantic model을 노출하는 것은 쓸모없을 뿐 아니라(model이 Python object를 전달할 수 없다) 잠재적으로 혼란스럽다. AutomationBench는 `world`를 model-facing schema에서 완전히 제거한다.

메커니즘은 `tool_wrapper.py`에 있다:

```python
# automationbench/tool_wrapper.py  L13-38
def _create_tool_wrapper(func: Callable, args_to_skip: list[str]) -> Callable:
    """Create a wrapper function with skipped args removed from signature.

    This is needed because convert_func_to_oai_tool uses the function signature
    to generate JSON schema, and the strict schema validation fails on complex
    types like WorldState before we can strip them.
    """
    original_sig = inspect.signature(func)
    original_hints = get_type_hints(func)

    # Build new parameters without skipped args
    new_params = [p for name, p in original_sig.parameters.items() if name not in args_to_skip]
    new_sig = original_sig.replace(parameters=new_params)

    # Build new type hints without skipped args
    new_hints = {k: v for k, v in original_hints.items() if k not in args_to_skip}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Replace signature and annotations
    wrapper.__signature__ = new_sig  # type: ignore
    wrapper.__annotations__ = new_hints

    return wrapper
```

핵심 동작은:
1. `inspect.signature`에서 `world`를 제거한다 — `convert_func_to_oai_tool`(verifiers의 JSON schema 생성 utility)이 절대 보지 못하도록.
2. `__annotations__`에서 `world`를 제거한다 — schema 생성에 사용되는 `get_type_hints`도 clean한 결과를 만들도록.
3. 실제 함수는 `functools.wraps`를 통해 모든 argument와 함께 여전히 호출된다.

재injection은 `update_tool_args`의 dispatch 시점에 일어난다:

```python
# automationbench/runner.py  L118-134
def update_tool_args(
    self,
    tool_name: str,
    tool_args: dict,
    messages: vf.Messages,
    state: vf.State,
    **kwargs,
) -> dict:
    """Auto-inject skipped args into tool calls."""
    updated_args = dict(tool_args)

    # Auto-inject any skipped args that exist in state
    skipped = self.skipped_args.get(tool_name, [])
    if "world" in skipped:
        updated_args["world"] = state["world"]

    return updated_args
```

이 contract를 직접 검증하는 test:

```python
# tests/test_runner.py  L138-154
def test_update_tool_args_injects_world(self):
    world = WorldState()
    state = cast(vf.State, {"world": world})

    updated = env.update_tool_args(
        "salesforce_query",
        {"object_type": "Contact", "where_clause": "Email = 'test@example.com'"},
        [], state,
    )
    assert updated["world"] is world
```

`is` check에 주목하라: inject된 `world`는 copy가 아닌 정확히 동일한 object다. 이것은 의도적이다 — tool 내부의 mutation이 동일한 episode의 후속 tool call에 반영되어야 한다.

어떤 tool에 `args_to_skip`이 필요한지 자동 감지는 construction 시점에 일어난다:

```python
# automationbench/runner.py  L90-93
            sig = inspect.signature(tool)
            args_to_skip = ["world"] if "world" in sig.parameters else []
            self.add_tool(tool, args_to_skip=args_to_skip)
```

Python signature에 실제로 `world` parameter를 포함하는 tool만 skip 처리를 받는다. world state를 건드리지 않는 utility tool(예: `base64_encode`)은 수정 없이 통과된다.

---

## 6. Reconciling Documentation vs. Code

논문 수준의 설명과 실제 구현 사이에 두 가지 구체적인 tension이 존재한다.

### 6.1 "Two tools"는 `zapier` mode만을 가리킨다

논문과 README는 AutomationBench가 agent에게 "two tools"를 준다고 설명한다. 이는 기본 `zapier` mode(`search_tools` + `execute_tool`)에서만 정확하다. benchmark는 각기 다른 tool interface를 가진 세 가지 distinct mode를 함께 제공한다:

| Claim | Accurate scope |
|-------|---------------|
| "Two tools" | `toolset="zapier"` only |
| "~400 named tools" | `toolset="limited_zapier"`, per-task filtered |
| "REST-style tools" | `toolset="api"` |

실험 결과를 읽을 때는 항상 어떤 `toolset`이 사용되었는지 확인하라. `zapier`와 `limited_zapier` 간의 점수 차이가 tool discovery 비용의 추정치다.

### 6.2 "Max 50 steps"는 hard guard가 아닌 prompt hint다

system prompt는 model에게 50 turn 안에 task를 완료하도록 지시한다. 코드의 hard guard는 `max_turns=25`다(runner.py L54). 이는 두 개의 다른 knob이다:

- `max_turns=25` — `vf.StatefulToolEnv`가 강제 적용; 25 turn이 경과하면 episode가 무조건 종료된다. 이를 늘리려면 `AutomationBenchEnv`에 `max_turns=N`을 전달해야 한다.
- system prompt budget — model 행동에 영향을 주는 soft instruction(많은 model이 제한된 step을 부여받으면 효율적이려 한다)이지만 enforcement 메커니즘이 없다.

subtask당 `search_tools` 한 번, `execute_tool` 한 번을 호출하는 model에게 25 turn은 약 12개의 subtask를 지원한다. 복잡한 multi-step workflow(search → validate → execute → verify → chain)에서 25-turn ceiling은 50-turn prompt가 시사하는 것보다 더 빡빡하다.

---

## 7. Meta-Message Compression in `zapier` Mode

`zapier` mode에서 `search_tools`는 tool name, docstring, 전체 parameter schema를 담은 verbose JSON을 반환한다. model이 `execute_tool`을 호출하여 그 결과를 활용한 뒤에는 verbose payload가 context window에서 dead weight가 된다. `_compress_meta_messages`가 이를 처리한다:

```python
# automationbench/runner.py  L234-293
def _compress_meta_messages(
    self,
    messages: vf.Messages,
    tool_messages: vf.Messages,
    state: vf.State,
) -> vf.Messages:
    """Compress old search_tools results after execute_tool is called.

    Once the model acts on search results by calling execute_tool, the verbose
    search results (full descriptions + parameter schemas) are dead weight.
    Replace them with a brief tool name list to save tokens on future turns.

    Only compresses search results from PREVIOUS turns, never the current turn.
    This ensures schemas remain available for tools searched in the same turn
    as an execute_tool call, preventing argument-name hallucination when the
    model searches and executes in parallel.
    """
```

compression은 turn-aware다: *이전* turn의 search 결과만 재작성하며, 현재 turn은 절대 건드리지 않는다. 주석이 이유를 설명한다 — model이 동일한 turn에서 `search_tools`와 `execute_tool`을 함께 발행하는 경우(유효한 parallel call pattern), 검색된 tool의 schema가 argument name이 검증될 때 여전히 context에 있어야 한다. 이를 조기에 compress하면 `execute_tool` call에서 hallucinated argument name이 발생할 것이다.

이전 turn의 search 결과는 다음으로부터:

```json
[{"name": "gmail_send_email", "description": "...", "parameters": {...}}, ...]
```

다음으로 재작성된다:

```
[Previously found: gmail_send_email, slack_send_message]
```

이는 content가 ≥200 characters일 때만 적용되며(짧은 결과는 재작성할 가치가 없다), JSON object list로 파싱 가능한 경우에만 적용된다.

---

## Summary

AutomationBench는 하나의 구조적 선택으로 결정론과 재현성 보장을 구축한다: 전체 world가 `extra="forbid"`를 가진 단일 in-process Pydantic object다. 이 선택은 다른 모든 설계 결정에 전파된다:

- `setup_state`가 episode당 한 번 `WorldState(**initial_state_dict)`를 construct하고 harness state dict에 전달한다.
- `update_tool_args`가 `tool_wrapper.py`를 통해 model-facing schema에서 strip한 뒤 dispatch 시점에 `world`를 재inject한다.
- rubric이 episode end에 `state["world"]`를 읽는다 — round-trip도, serialization도 없다.
- `export_results`가 end state, assertion 결과, token cost를 JSON visualizer feed로 묶는다.

세 가지 toolset mode(`zapier`, `limited_zapier`, `api`)는 동일한 infrastructure 위의 ablation으로, model이 보는 tool interface만 다를 뿐이다 — 기저의 world simulation도, grading logic도, episode lifecycle도 변경되지 않는다. 그 clean separation이 ablation 결과를 해석 가능하게 만드는 것이다.

See also: [[automationbench-harness]], [[automationbench-overview]]
