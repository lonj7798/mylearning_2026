<!-- chapter: ch-04
track: internals
kind: content
title: Execute and the Simulated World
deps: [ch-03]
sources: [[automationbench-harness]], [[automationbench-tasks-grading]]
-->

# 04장 — Execute and the Simulated World

> **핵심 통찰.** AutomationBench의 모든 tool call은 단일 in-process Pydantic 객체를 읽고 쓰는 Python function으로 resolve된다. HTTP server도, database connection도, subprocess도, episode 시작 시 seeded된 것 이외의 randomness도 없다. 이 설계 선택이 반복 실행 간 variance를 1% 미만으로 유지하고, grading을 완전히 deterministic하게 만드는 것이다.

> **가이드라인.** agentic benchmark를 구축할 때는 world state를 in-process에 두고 쓰기 시점에 검증하라(`extra="forbid"`). API surface fidelity — pagination, 4xx codes, required-field checks — 는 모두 routing layer에 밀어 넣고, grading layer에는 넣지 말라. Grading은 느슨한 world model을 절대 보완해서는 안 된다.

---

## 1. The execution path from tool call to world mutation

Ch-02에서 `world` injection trick을 소개하고, Ch-03에서 agent가 tool을 찾는 방법을 보여 줬다. 이번 장은 agent가 실제로 tool을 호출했을 때 무슨 일이 일어나는지를 다룬다.

toolset mode에 따라 두 가지 execution path가 있다:

| Toolset mode | Entry point | Route |
|---|---|---|
| `zapier` / `limited_zapier` | `execute_tool(world, tool_name, arguments)` | `ToolRegistry.execute` → named Python function |
| `api` | `api_fetch(world, method, url, params, body)` | `_url_to_internal_path` → `route_<app>` → named Python function |

두 경로 모두 `automationbench/tools/api/impl/<app>.py`의 동일한 Python function으로 수렴한다. routing layer는 순전히 dispatch concern이다.

---

## 2. `execute_tool` → `ToolRegistry.execute`

`execute_tool`은 Zapier mode의 agent-facing entry point다
(`automationbench/tools/zapier/meta.py`, L189–204):

```python
# automationbench/tools/zapier/meta.py  L189–204
def execute_tool(world: WorldState, tool_name: str, arguments: str) -> str:
    """Execute a discovered tool by name with the given arguments.

    Use search_tools first to find the right tool and its parameter schema,
    then call this with the tool name and a JSON string of arguments.

    Args:
        world: The current world state (injected automatically).
        tool_name: The exact tool name from search results.
        arguments: JSON string of arguments matching the tool's parameter schema.

    Returns:
        The tool's return value (JSON string).
    """
    registry = _get_registry()
    return registry.execute(tool_name, arguments, world=world)
```

`world`는 signature에 타입이 명시돼 있지만, agent가 보는 JSON schema에는 나타나지 않는다 — Ch-02의 `args_to_skip=["world"]`가 schema 생성 전에 제거한다. agent는 `tool_name`과 `arguments`(JSON string)를 공급하고, `world`는 harness를 통해 도착한다.

L98–110의 `ToolRegistry.execute`는 세 가지 일을 한다:

```python
# automationbench/tools/zapier/meta.py  L98–110
def execute(self, tool_name: str, arguments: str, **injected: Any) -> str:
    """Execute a tool by name with JSON arguments string."""
    func = self._tool_map.get(tool_name)
    if func is None:
        raise ValueError(
            f"Unknown tool: {tool_name}. Use search_tools to discover available tools."
        )
    parsed_args = json.loads(arguments)
    merged = {**parsed_args, **injected}
    result = func(**merged)
    if isinstance(result, str):
        return result
    return json.dumps(result)
```

1. **Lookup** — `tool_name`을 `_tool_map`(registry 초기화 시 `ALL_TOOLS`로부터 구축됨)에서 확인한다. 알 수 없는 이름은 `search_tools`를 먼저 호출하라는 명시적 hint와 함께 `ValueError`를 발생시킨다. 이는 의도적인 것이다: agent는 추측해서 execute할 수 없다.

2. **JSON-parse + inject** — `arguments`를 `json.loads`로 디코드한다. 그런 다음 `world`(및 기타 injected kwargs)를 `{**parsed_args, **injected}`로 병합한다. agent가 공급한 args는 `world`를 shadow할 수 없는데, `args_to_skip`이 이미 published schema에서 제거했기 때문이다.

3. **Dispatch + coerce** — merged kwargs로 함수를 호출한다. 반환값이 이미 `str`이면(흔한 경우: tool이 `json.dumps(...)`를 반환) 그대로 통과시킨다. non-string은 `json.dumps`로 직렬화한다. agent는 항상 JSON string을 받는다.

`automationbench/tool_wrapper.py`의 `_create_tool_wrapper`(L13–38)는 schema-facing signature를 담당한다:

```python
# automationbench/tool_wrapper.py  L13–38
def _create_tool_wrapper(func: Callable, args_to_skip: list[str]) -> Callable:
    """Create a wrapper function with skipped args removed from signature."""
    original_sig = inspect.signature(func)
    original_hints = get_type_hints(func)

    new_params = [p for name, p in original_sig.parameters.items() if name not in args_to_skip]
    new_sig = original_sig.replace(parameters=new_params)
    new_hints = {k: v for k, v in original_hints.items() if k not in args_to_skip}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__signature__ = new_sig
    wrapper.__annotations__ = new_hints
    return wrapper
```

wrapper는 schema 생성 시점(`_get_parameter_schema`, L84–92)에만 사용된다. 실행 시점에는 `func(**merged)`가 `world`를 이미 re-inject한 상태로 실제 함수를 직접 호출한다. wrapper는 실제 tool call 중에는 절대 invoke되지 않는다.

---

## 3. API mode: `api_fetch` → `_url_to_internal_path` → `route_<app>`

agent가 `api` mode에 있을 때는 완전한 REST URL로 `api_fetch`를 호출한다. routing은 2단계 lookup이다(`automationbench/tools/api/fetch.py`):

```python
# automationbench/tools/api/fetch.py  L252–283
def api_fetch(
    world: WorldState,
    method: str,
    url: str,
    params: Optional[str] = None,
    body: Optional[str] = None,
) -> str:
    try:
        parsed_params = _coerce_to_dict(params)
        body_dict = _coerce_to_dict(body)
    except json.JSONDecodeError as e:
        return json.dumps({"error": {"code": 400, "message": f"Invalid JSON: {e}"}})

    internal_path, router = _url_to_internal_path(url)
    if router is not None:
        return router(world, method, internal_path, parsed_params, body_dict)

    return json.dumps({"error": {"code": 404, "message": f"Unknown API URL: {url}"}})
```

**Stage 1 — `_url_to_internal_path`**는 전체 URL을 `(internal_path, router_fn)` 쌍으로 변환한다. 이 함수는 세 개의 lookup table을 유지한다(L74–248):

- `_STATIC_URL_ROUTERS` — `(url_prefix, internal_prefix, router)` triple의 목록. 각 항목은 실제 URL prefix를 제거하고 canonical internal prefix를 앞에 붙인다. 예:
  `("https://api.hubapi.com/", "hubspot/", route_hubspot)`은
  `https://api.hubapi.com/crm/v3/objects/contacts` →
  `hubspot/crm/v3/objects/contacts`로 변환한다.

- `_DYNAMIC_HOST_ROUTERS` — subdomain이 고객의 account를 인코딩하는 서비스(Salesforce, Zendesk, Freshdesk, Mailchimp)를 위한 hostname-suffix matching. `https://mycompany.my.salesforce.com/services/data/v60/query`는 `.salesforce.com`에 매칭되어 `salesforce/services/data/v60/query`가 된다.

- 동일한 host에서 여러 router로 fan out하는 서비스를 위한 special-case blocks. `graph.facebook.com`(L143–157)은 path 구조에 따라 네 개의 별도 router로 분기된다 — Instagram, Facebook Pages, Facebook Conversions, Facebook Lead Ads — (path가 `/events`, `/leads`, `/leadgen_forms`로 끝나는지, `act_`가 포함됐는지, 단일 segment인지 등). 마찬가지로 `api.linkedin.com/rest/`(L159–165)는 path가 `conversionEvents`로 시작하는지에 따라 분기된다. 그리고 `*.atlassian.net`(L167–172)은 `rest/`(→ Jira) 대 `wiki/`(→ Confluence)로 분기된다.

**Stage 2 — `route_<app>`**은 `(world, method, internal_path, params, body_dict)`를 받아 올바른 impl function으로 dispatch한다. 각 router는 `automationbench/utils/routing.py`의 `make_router`로 구축된다:

```python
# automationbench/utils/routing.py  L20–53
def make_router(
    routes: list[tuple[str, str, str]],
    handlers: dict[str, Callable],
) -> Callable:
    def _route(world, method, path, params, body_dict) -> str:
        method_upper = method.upper()
        for route_method, pattern, handler_key in routes:
            if route_method != method_upper:
                continue
            match = re.match(pattern, path)
            if not match:
                continue
            ids = list(match.groups())
            return handlers[handler_key](world, ids, params, body_dict)
        return json.dumps({"error": {"code": 404, "message": f"No handler for {method} {path}"}})
    return _route
```

Route는 `(method, pattern, handler_key)` regex triple이다. pattern의 capture group이 handler에 전달되는 `ids` list가 된다. `r"slack/conversations\.history$"`(group 없음)는 `ids=[]`를 yield하고, `r"items/([^/]+)$"`는 `ids=["abc123"]`를 yield한다. First-match-wins; method 비교는 case-insensitive다(L42). 매칭되지 않은 요청은 404 JSON을 반환한다 — exception을 raise하지 않는다.

Slack router가 이 패턴을 잘 보여 준다
(`automationbench/tools/api/routes/slack.py`, L35–92):

```python
# automationbench/tools/api/routes/slack.py  L35–92
_ROUTES: list[tuple[str, str, str]] = [
    ("GET",  r"slack/conversations\.list$",    "conversations_list"),
    ("POST", r"slack/chat\.postMessage$",      "chat_post_message"),
    ("GET",  r"slack/conversations\.history$", "conversations_history"),
    # Model hallucinations: channels/C_XXX/messages → conversations.history
    ("GET",  r"slack/channels/([^/]+)/messages$", "channels_messages_alias"),
    # Model hallucinations: channels.C_XXX/messages (dot) → conversations.history
    ("GET",  r"slack/channels\.([^/]+)/messages$", "channels_dot_messages_alias"),
    # Model hallucinations: channels.history → conversations.history
    ("GET",  r"slack/channels\.history$",      "conversations_history"),
    # ... (19 total routes)
]

_HANDLERS = {
    "conversations_list":    lambda w, ids, p, b: slack_conversations_list(w, **p),
    "chat_post_message":     lambda w, ids, p, b: slack_chat_post_message(w, **b),
    "conversations_history": lambda w, ids, p, b: slack_conversations_history(w, **p),
    "channels_messages_alias":
        lambda w, ids, p, b: slack_conversations_history(w, channel=ids[0], **p),
    "channels_dot_messages_alias":
        lambda w, ids, p, b: slack_conversations_history(w, channel=ids[0], **p),
    # ...
}

route_slack = make_router(_ROUTES, _HANDLERS)
```

Handler는 `(world, ids, params, body_dict)` calling convention을 underlying impl function의 kwargs로 adapt하는 lambda다. GET endpoint는 `params`에서, POST/PATCH endpoint는 `body_dict`에서 읽는다. `channels_messages_alias`에서 capture-group trick이 사용된다: `ids[0]`가 URL segment에서 파싱된 channel ID를 들고 있으며, 이것이 `slack_conversations_history`에 `channel=`로 forwarding된다.

---

## 4. URL-hallucination tolerance: the deliberate over-forgiveness tradeoff

routing table에는 ~15개의 알려진 model 실수가 포함되어 있으며, 올바른 router로 자동으로 normalize된다. `fetch.py`에서 엄선한 샘플:

| Hallucinated URL prefix | Canonical intent | Line |
|---|---|---|
| `https://slack.googleapis.com/` | Slack Web API | L89 |
| `https://www.slack.com/api/` | Slack Web API (extra `www.`) | L86 |
| `https://slack.com/` | Slack Web API (missing `/api/`) | L88 |
| `https://www.googleapis.com/gmail/` | Gmail API | L76 |
| `https://www.googleapis.com/sheets/` | Google Sheets | L80 |
| `https://www.sheets.googleapis.com/` | Google Sheets (hallucinated domain) | L81 |
| `https://calendar.googleapis.com/` | Google Calendar | L79 |
| `https://sandbox-quickbooks.api.intuit.com/` | QuickBooks (sandbox alias) | L115 |
| `https://3.basecamp.com/` | Basecamp 3 (alias) | L113 |

의도는 inline comment에 명시적으로 드러난다: `# models sometimes use www.googleapis.com for Gmail`(L76), `# models hallucinate this domain`(L81, L89).

**설계 tradeoff.** hallucinated URL을 normalize하는 것은 의도적인 eval-robustness 선택이다: measurement에서 noise를 제거한다. agent가 올바른 operation을 수행했지만 `slack.com/api/` 대신 `slack.googleapis.com`을 사용했다면, benchmark는 reasoning이나 workflow capability와 무관한 실수에 대해 404로 페널티를 줄 것이다. benchmark의 논지는 흥미로운 난점이 cross-app coordination과 policy adherence에 있다는 것이지, documentation lookup으로 즉시 해결할 수 있는 URL trivia에 있지 않다는 것이다.

반론은 over-forgiveness가 실질적인 capability gap을 가릴 수 있다는 것이다. 잘못된 base URL로 통신하는 실제 Zapier automation은 실패한다; benchmark는 시뮬레이션에서는 동작하지만 배포 시 URL을 hallucinate하는 agent를 잡아내지 못한다. benchmark는 실제로 측정하고자 하는 capability에 대한 measurement variance 감소를 위해 URL correctness에 대한 sim-to-real fidelity를 거래한다. URL accuracy를 측정하고자 하는 연구자는 hallucination tolerance를 비활성화한 api mode에서 실행해야 하지만, 그 옵션은 현재 CLI flag로 노출되어 있지 않다.

---

## 5. App state: Pydantic with `extra="forbid"` and `to_display_dict()`

모든 app schema는 `model_config = ConfigDict(extra="forbid")`를 사용한다. 이는 강력한 보증이다: model에 선언되지 않은 키는 쓰기 시점에 `ValidationError`를 trigger한다. silent field drop도 없고, task seed나 tool implementation이 선언되지 않은 데이터를 몰래 넣을 수 있게 해 주는 `extra="ignore"` permissiveness도 없다.

`automationbench/schema/hubspot.py`의 `HubSpotContact`가 이 패턴을 잘 보여 준다:

```python
# automationbench/schema/hubspot.py  L29–72
class HubSpotContact(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    id: str = Field(default_factory=generate_hubspot_id)
    email: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    jobtitle: Optional[str] = None
    lifecyclestage: str = Field(default="lead", validation_alias="lifecycle_stage")
    lead_score: Optional[int] = None
    properties: Dict[str, str] = Field(default_factory=dict)
    # ... (UTM attribution, open_ticket, industry, NPS, billing, notes ...)
    utm_source: Optional[str] = None
    utm_campaign: Optional[str] = None
    nps_score: Optional[int] = None
    payment_status: Optional[str] = None
    lifetime_value: Optional[float] = None
    notes: Optional[str] = None
```

거의 모든 field가 `Optional[T] = None`이다. 새로 seed된 contact는 해당 task와 관련된 field만 담는다. 이것은 게으름이 아니라 의도적인 realism 선택이다: 실제 CRM record는 sparse하다. 대부분의 contact는 NPS score, UTM attribution chain, payment status를 동시에 갖지 않는다.

`to_display_dict()`(L74–143)는 sparse presentation을 강제한다:

```python
# automationbench/schema/hubspot.py  L74–143
def to_display_dict(self) -> dict:
    result = {
        "id": self.id,
        "email": self.email,
        "firstname": self.firstname,
        "lastname": self.lastname,
        "phone": self.phone,
        "company": self.company,
        "jobtitle": self.jobtitle,
        "lifecyclestage": self.lifecyclestage,
        "hs_object_id": self.id,
        "createdAt": self.created_at.isoformat(),
        "updatedAt": self.updated_at.isoformat(),
    }
    if self.lead_score is not None:
        result["lead_score"] = self.lead_score
    if self.utm_source:
        result["utm_source"] = self.utm_source
    if self.nps_score is not None:
        result["nps_score"] = self.nps_score
    if self.payment_status is not None:
        result["payment_status"] = self.payment_status
    if self.lifetime_value is not None:
        result["lifetime_value"] = self.lifetime_value
    if self.notes is not None:
        result["notes"] = self.notes
    return result
```

핵심 identity field(`id`, `email`, 이름들, `lifecyclestage`, timestamps)는 항상 존재한다. 선택적 enrichment는 non-None일 때만 추가된다. agent는 실제 HubSpot API response가 가볍게 채워진 contact에 대해 반환하는 것과 동일한 JSON shape를 보게 된다: 30개의 null-padded key가 아니라 소수의 field.

이는 agent 동작에 중요하다. field 존재 여부로 pattern-match하는 agent(예: `payment_status`가 있는지에 따라 분기)는 이 model 하에서 올바르게 동작한다. `record["payment_status"]`를 확인 없이 항상 읽는 agent는 `KeyError`를 발생시킬 것이다 — live API와 동일한 failure mode다.

동일한 `extra="forbid"` + sparse `to_display_dict()` 패턴이 44개 모든 app schema에 반복된다. `WorldState` 자체도 root level에서 이를 적용한다
(`automationbench/schema/world.py`, L70–73):

```python
# automationbench/schema/world.py  L70–73
class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: WorldMeta = Field(default_factory=WorldMeta)
    airtable: AirtableState = Field(default_factory=AirtableState)
    # ... (44 app-state fields)
```

계층의 모든 layer가 이 제약을 강제한다. `WorldState`와 leaf record model 사이에 검증되지 않은 쓰기가 통과할 수 있는 gap은 존재하지 않는다.

---

## 6. Real-API fidelity: pagination, required fields, and 4xx codes

시뮬레이션은 HTTP mechanics를 모른 척하지 않는다. route handler는 live API가 거부할 것과 동일한 조건에 대해 structured error response를 emit하고, list endpoint는 output을 paginate한다.

**4xx codes.** `test_api_fetch.py`(L224–251)는 계약을 명문화한다:

```python
# tests/test_api_fetch.py  L224–251
class TestApiFetch:
    def test_unknown_url_returns_404(self):
        world = WorldState()
        result = json.loads(api_fetch(world, "GET", "https://unknown.example.com/data"))
        assert result["error"]["code"] == 404

    def test_invalid_json_params(self):
        world = WorldState()
        result = json.loads(
            api_fetch(world, "GET",
                      "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                      params="{bad")
        )
        assert result["error"]["code"] == 400
        assert "Invalid JSON" in result["error"]["message"]

    def test_valid_gmail_route(self):
        world = WorldState()
        result = json.loads(
            api_fetch(world, "GET",
                      "https://gmail.googleapis.com/gmail/v1/users/me/messages")
        )
        assert result.get("error", {}).get("code") != 404
```

동일한 패턴이 router 내부에도 적용된다. `test_api_routes.py`(L88–93, L147–151, L226–231)는 인식된 host 내의 알 수 없는 path도 여전히 404를 반환함을 확인한다:

```python
# tests/test_api_routes.py  L88–93
def test_unknown_route_404(self):
    world = WorldState()
    result = json.loads(route_gmail(
        world, "GET", "gmail/v1/nonexistent", {}, {}
    ))
    assert result["error"]["code"] == 404
```

**Update operations and not-found 4xx.** Gorgias impl test
(`tests/test_api_impl_gorgias.py`, L77–80)는 존재하지 않는 ticket에 대한 update가 silent create나 discard 대신 error를 반환함을 검증한다:

```python
# tests/test_api_impl_gorgias.py  L77–80
def test_update_not_found(self):
    world = WorldState()
    result = json.loads(gorgias_tickets_update(world, ticket_id="fake"))
    assert "error" in result
```

**End-to-end state mutations.** 동일한 test file(L31–34)은 성공적인 create가 생성된 record를 반환하는 동시에 `world.gorgias.tickets`에 append함을 확인한다:

```python
# tests/test_api_impl_gorgias.py  L31–34
def test_create_basic(self):
    world = WorldState()
    result = json.loads(gorgias_tickets_create(world, subject="Help"))
    assert result["subject"] == "Help"
    assert len(world.gorgias.tickets) == 1
```

이것이 핵심 loop다: agent가 tool을 호출하고, tool이 `world`를 mutate하고, grader가 episode 종료 후 `world`를 읽는다. 위의 test는 세 단계 모두가 올바르게 함께 작동함을 보여 주는 최소 증명이다.

**Route dispatch fidelity.** `test_routing.py`(L54–64)는 `make_router` failure mode를 명시적으로 규정한다: 매칭되는 route 없음 → 404, method mismatch → 404, case-insensitive method → 성공:

```python
# tests/test_routing.py  L54–64
def test_no_matching_route_returns_404(self):
    routes = [("GET", r"items$", "list_items")]
    router = make_router(routes, {"list_items": lambda w, i, p, b: "[]"})
    result = json.loads(router(WorldState(), "GET", "nonexistent", {}, {}))
    assert result["error"]["code"] == 404

def test_method_mismatch(self):
    routes = [("GET", r"items$", "list_items")]
    router = make_router(routes, {"list_items": lambda w, i, p, b: "[]"})
    result = json.loads(router(WorldState(), "POST", "items", {}, {}))
    assert result["error"]["code"] == 404
```

fidelity 수준은 의도적이다. benchmark는 실제 API의 마찰(올바른 method, 올바른 path, required field 존재, valid JSON body)을 타겟으로 하면서, 그 불안정성(rate limit, auth token 만료, transient 5xx, network latency)은 복제하지 않는다. 이것이 유용한 simulation과 production client 사이의 경계선이다.

---

## 7. Why in-process simulation buys determinism and <1% variance

핵심 설계 결정 — 모든 것이 단일 Python process 내에서 실행된다 — 은 네 가지 구체적인 결과를 낳는다.

**No I/O non-determinism.** live sandbox는 HTTP round-trip, authentication state, rate-limit header, server-side side effect를 도입한다. 이 중 어느 것이든 동일한 episode의 두 실행 간에 달라질 수 있다. in-process에서는 non-determinism의 유일한 원인이 Python의 random number generator인데, AutomationBench는 이를 `example_id`당 deterministic하게 seed한다(noise seeding은 Ch-07 참조). 동일한 model에서 episode 501을 두 번 실행하면 동일한 `WorldState` snapshot을 얻는다.

**Instant state inspection.** grader는 agent loop가 끝난 후 `world`를 직접 읽는다. DB를 query하거나, API를 poll하거나, log를 deserialize할 필요가 없다. `world.hubspot.contacts[0].lifecyclestage`는 Python attribute access다. assertion logic(`rubric/assertions/<app>.py`)은 순수한 field comparison이다.

**Atomic rollback.** 각 episode는 새로운 `WorldState()`를 얻는다. episode 간에 공유된 mutable state가 없다. task 간 evaluation을 병렬화하는 것이 구조적으로 안전하다 — locking도, teardown도, fixture cleanup도 없다.

**Reproducibility for RL.** `partial_credit` score(Ch-06)는 RL 훈련에 직접 사용 가능한 dense reward signal이다. environment가 deterministic하기 때문에, training loop가 동일한 seed로 동일한 episode를 재실행하면 동일한 reward signal을 얻는다. 이것이 AutomationBench가 training environment로도 기능하게 만드는 속성이다(Ch-10).

비용은 sim-to-real transfer다. in-process에서 60%를 기록하는 agent가 live Salesforce instance에서는 reasoning과 전혀 무관한 이유로 더 낮은 점수를 받을 수 있다: stale OAuth token, minor version bump로 이동한 API endpoint, spring product update에서 바뀐 field 이름. benchmark는 tool-using workflow에 대한 reasoning을 측정한다; 실제 API credential을 규모 있게 유지하는 DevOps overhead는 명시적으로 측정하지 않는다. AutomationBench가 타겟으로 하는 목적들 — release 간 model capability 비교, toolset mode의 효과 측정, RL training episode 생성 — 에 대해 이 tradeoff는 올바르다.

---

## Key takeaways

- `execute_tool` → `ToolRegistry.execute`: 세 단계 — 이름으로 lookup(알 수 없으면 raise), argument string을 `json.loads`, `**injected`로 `world`를 merge, 실제 함수 호출. String 결과는 그대로 통과; 나머지는 `json.dumps`.
- `api_fetch` → `_url_to_internal_path` → `make_router`: 두 dispatch stage. 첫 번째 stage는 static prefix table, dynamic hostname-suffix table, special-case fan-out block(Facebook, LinkedIn, Atlassian)을 사용해 전체 URL을 `(canonical_path, router_fn)` 쌍으로 매핑한다. 두 번째 stage는 `(method, path)`를 app별 route table에 regex-match해서 impl function을 호출한다.
- URL-hallucination tolerance는 ~15개의 알려진 model 실수를 올바른 router로 normalize한다. 이는 benchmark가 실제로 타겟으로 하는 capability의 measurement noise 감소를 위해 sim-to-real URL fidelity를 의도적으로 거래하는 것이다.
- 모든 Pydantic model class의 `extra="forbid"`는 모든 layer에서 선언되지 않은 쓰기를 방지한다. `to_display_dict()`는 `None` field를 생략하여 실제 API response를 모방하는 sparse record를 생성한다.
- In-process execution은 1% 미만의 run-to-run variance, instant state inspection, trivial parallelism, RL-compatible determinism을 제공한다. 비용은 명시적이다: URL accuracy와 DevOps friction은 측정되지 않는다.
