<!-- chapter: ch-10
     track: extension
     kind: lab
     title: Lab: Run It, Extend It, Reuse It as an RL Environment
     deps: [ch-09]
     sources: [[automationbench-harness]], [[automationbench-tasks-grading]], [[benchmark-comparison]]
     capstone_for: automation-bench
-->

# 10장 — Lab: Run It, Extend It, Reuse It as an RL Environment

> **핵심 통찰.** AutomationBench는 단순한 evaluator가 아니다 — dense-reward RL environment다. `partial_credit = passed / total`(`create_rubric`에서 weight 1.0)은 `verifiers` training loop이 직접 사용할 수 있는 continuous signal이다. harness의 모든 설계 결정 — typed in-process world, deterministic seeded noise, must-not-occur guard, free-assertion exclusion — 은 reward가 eval 시점뿐 아니라 training 시점에도 정직하고 hackable-resistant하도록 만들기 위해 이루어졌다.
>
> **가이드라인.** 확장하기 전에 실행하라; 설계하기 전에 확장하라. Stage 1은 25-turn episode가 실제로 어떻게 생겼는지를 가르친다. Stage 2는 extension surface의 비용이 얼마인지를 가르친다. Stage 3은 `partial_credit`이 sparse 0/1보다 더 나은 training signal인 이유를 가르친다. Stage 4는 그 교훈들을 나만의 agent benchmark 설계로 변환한다. Stage 5는 루프를 닫는다: Lina TMR sales-call simulator는 AutomationBench의 assertion engineering을 상속한다 — τ-style 대화 + AB-style end-state grading + pass^k 신뢰도.

---

## Goal

repo에서 재현 가능한 세 artifact:

1. **A run artifact.** 실제 evaluation이 생성한 `visualizer/runs/local/<model>-<timestamp>.json` — 최소 `simple` domain(200 tasks, haiku-class pricing 기준 약 \$0.50)이지만 이상적으로는 하나의 scored domain(`sales` 또는 `hr`). visualizer 스크린샷 또는 터미널 출력을 memo에 포함하라.
2. **An extension.** 다음 중 하나: (a) 기존 domain에 새 task 추가, 또는 (b) rubric에 새로운 assertion type 등록. `uv run pytest tests/` 아래에서 깨끗하게 실행되어야 한다.
3. **A deliverable memo.** `ab-task-spec.md`(새 task + rubric) 또는 `lina-bench-spec.md`(Lina TMR benchmark spec). Stage 5의 template을 참고하라.

---

## Full-budget path

Target: standard cloud API(한 domain에 약 \$5–15), local Python env.

- **Run.** 여섯 개 scored domain 전체(600 tasks). `--max-concurrent 100`(기본값) 및 thinking model에는 `--reasoning-effort medium` 사용. `claude-haiku-4-5` 기준 약 \$10 budget.
- **Visualize.** `compare.html`을 두 run으로 열어 — baseline vs `--toolset limited_zapier` — tool-discovery vs tool-execution 격차를 확인하라.
- **Extend.** 새 `sales` task 추가(cross-app, 최소 4개 assertion, 그중 하나는 negative guard) + 새 `@AssertionRegistry.register` type 추가.
- **Memo.** 전체 Lina TMR benchmark spec(Stage 5 참고).

## Resource-constrained path

Target: 최소 API 지출(약 \$0.50–2), GPU 불필요.

- **Run.** `simple` domain만(`--domains simple`). 약 200 tasks, haiku-class model. 먼저 `--num-examples 20`으로 smoke test 추가.
- **Extend.** 새 assertion type 하나만 — 시간이 부족하면 전체 task constructor는 건너뛰어도 된다.
- **Memo.** 두 deliverable 옵션 중 하나.

---

## Stage 1 — Run a real evaluation

### Install and first smoke test

```bash
# automationbench/README.md — Quick start
git clone https://github.com/zapier/AutomationBench.git
cd AutomationBench

# Install dependencies
uv sync

# Set your API key (or create a .env file)
export OPENAI_API_KEY=sk-...
# Anthropic: auto-detected via `claude-*` prefix
export ANTHROPIC_API_KEY=sk-ant-...

# Smoke test: 5 examples, simple domain
uv run auto-bench --model claude-haiku-4-5-20251001 \
  --domains simple \
  --num-examples 5

# Full simple domain
uv run auto-bench --model claude-haiku-4-5-20251001 --domains simple

# Single scored domain
uv run auto-bench --model claude-haiku-4-5-20251001 --domains sales
```

기본 `--toolset`은 `api`다(REST-shaped tools: `api_search`, `api_fetch`, `base64_encode`). discovery-tested headline mode를 위해서는 `zapier`로 전환 — agent가 약 400개의 tool 중에서 필요한 것을 찾기 위해 `search_tools(query)`를 호출해야 한다([[automationbench-harness]] §The three toolset modes):

```bash
# zapier toolset: discovery + execution tested together
uv run auto-bench --model claude-haiku-4-5-20251001 \
  --domains sales \
  --toolset zapier \
  --export-json visualizer/runs/local/haiku-sales-zapier.json
```

### The CLI option surface

`automationbench/scripts/eval.py`가 전체 option set을 노출한다. 주요 flag를 그대로 인용:

```
--model          Model name (default: gpt-5-mini)
--domains        Comma-separated domains or "all"
--toolset        api | zapier | limited_zapier
--num-examples   Number of examples (-1 for all)
--max-steps      Max model response steps per task (default: 50)
--max-concurrent Max concurrent tasks (default: 100)
--reasoning-effort  low / medium / high / xhigh / max
--input-cost     Per-token input cost in USD (overrides lookup)
--output-cost    Per-token output cost in USD (overrides lookup)
--export-json    Path to export results JSON
--save-every     Save incremental results every N tasks (default: 1)
--skip           Skip first N tasks
--tasks          Comma-separated task names to run
```

cost override는 local model이나 proxy 뒤의 fine-tuned checkpoint를 실행할 때 유용하다 — `--input-cost 0.000001 --output-cost 0.000002`를 전달하면 pricing-DB entry 없이도 정확한 \$/task 보고가 가능하다(`pricing.py`는 exact → normalized → alias lookup 순서로, 24h llm-prices.com cache와 hardcoded fallback을 통해 resolve한다).

### What to watch during an episode

`eval.py:248`은 `env.evaluate`에 `state_columns=["_usage", "_debug", "_assertion_results", "_end_state"]`를 전달한다. run 이후 JSON export는 task별로 다음을 포함한다:

```json
{
  "task": "sales.multi_hop_lookup",
  "reward": 0.6,
  "_assertion_results": [
    {"type": "salesforce_field_equals", "passed": true,  "excluded": false, "params": {...}},
    {"type": "gmail_message_sent_to_with_body_contains", "passed": false, "excluded": false, "params": {...}},
    {"type": "gmail_message_not_sent_to", "passed": true, "excluded": false, "params": {...}}
  ],
  "_end_state": { ... }
}
```

여기서 `reward`는 `partial_credit` — excluded되지 않은 assertion 중 통과한 비율이다. `task_completed_correctly`는 `partial_credit == 1.0`일 때만 1.0이다(`rubric/__init__.py:150`).

### Open the visualizer

```bash
# visualizer/README.md — Quick Start §2
python3 visualizer/serve.py
# → http://localhost:8000 (redirects to /compare.html)
```

`serve.py:99`는 `/`를 `compare.html`로 자동 redirect한다. `visualizer/runs/local/`에서 두 JSON 파일을 불러와 같은 domain에서 `api` vs `zapier` toolset을 비교하라. comparison view(`compare.html`)는 run을 **Average Score** 기준으로 순위를 매기고(pass rate는 secondary), cost vs score scatter plot을 제공한다. `index.html` single-run view는 score distribution histogram, task별 token usage, assertion별 pass count를 보여 준다.

---

## Stage 2 — The extension surface

harness에는 세 가지 자연스러운 extension point가 있다: task, app, assertion type.

### 2a — Add a task

각 domain의 `tasks.py`는 `example_id`, `task`, `prompt`, `answer`, `info` 키를 가진 plain `dict`를 반환하는 constructor 함수들을 export한다. `info` dict는 `zapier_tools`, `initial_state`, `assertions`를 담는다.

`automationbench/domains/simple/tasks.py`의 실제 예시(task 3001, sanity baseline):

```python
# automationbench/domains/simple/tasks.py
def get_simple_email_sf_contact_phone_update() -> dict:
    return {
        "example_id": 3001,
        "task": "simple.email_sf_contact_phone_update",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Jordan Lee just emailed us with a new phone number. "
                    "Can you find that email and update her phone number in Salesforce?"
                ),
            },
        ],
        "answer": "",          # always empty — grading is assertion-based
        "info": {
            "zapier_tools": [  # per-task allowlist for limited_zapier mode
                "gmail_find_email",
                "gmail_get_email_by_id",
                "salesforce_find_records",
                "salesforce_contact_update",
            ],
            "initial_state": {
                "gmail": { "messages": [ { ... } ], "labels": [], "drafts": [] },
                "salesforce": { "contacts": [ { ... } ] },
            },
            "assertions": [
                {"type": "salesforce_contact_phone_equals",
                 "contact_id": "003xx000003JORDAN",
                 "phone": "+15550101"},
            ],
        },
    }
```

non-trivial sales task에는 최소한 다음이 필요하다: cross-app traversal을 요구하는 positive assertion 하나, duplicate-create를 막는 count-lock(`salesforce_collection_count_equals`) 하나, shotgun routing을 막는 negative guard(`gmail_message_not_sent_to`) 하나. `initial_state`는 task가 실제로 건드리는 app만 채우면 된다 — 나머지 모든 app state는 `WorldState`의 `Field(default_factory=...)`를 통해 기본값으로 empty가 된다.

새 task의 `initial_state`에 noise를 직접 seed하거나, domain의 `apply_noise(world_dict, seed=example_id)`를 `_noise.py`에서 호출하라. sales noise pool은 `099` ID range(`001xx000099NA001`…)를 사용하므로 task-critical record와 절대 충돌하지 않는다(`_noise.py` 주석: "Noise IDs use the 099 range … to avoid conflicts").

### 2b — Add an app (Pydantic *State + tools + route)

새 SaaS app을 추가하는 데는 세 파일이 필요하다:

1. **`automationbench/schema/<app>.py`** — `extra="forbid"`를 가진 Pydantic `*State`:

```python
# automationbench/schema/hubspot.py (excerpt — HubSpotContact as the record shape)
class HubSpotContact(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    id: str = Field(default_factory=generate_hubspot_id)
    email: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    lifecyclestage: str = Field(default="lead", validation_alias="lifecycle_stage")
    lead_score: Optional[int] = None
    # ... typed fields; None = absent, not zero
```

2. **`automationbench/tools/zapier/<app>/`** — tool 하나당 Python module 하나. 각 함수는 `world: WorldState`를 첫 번째 positional argument로 받는다(`tool_wrapper.py`가 inject하며 `args_to_skip`을 통해 model의 JSON schema에서 숨겨진다).

3. **`automationbench/tools/api/routes/<app>.py`** — REST route handler. routes 패키지에 `route_<app>` 함수를 추가하여 등록; `api_fetch`가 `_url_to_internal_path`를 통해 dispatch한다(`fetch.py`).

4. `WorldState`에 새 state field를 등록한다(`schema/world.py:70`):

```python
# automationbench/schema/world.py (pattern — all 44 apps follow this)
class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # ... existing apps ...
    hubspot: HubSpotState = Field(default_factory=HubSpotState)
    # add your app here:
    # myapp: MyAppState = Field(default_factory=MyAppState)
```

### 2c — Add an assertion type

새 assertion type은 `AssertionRegistry.register`를 통해 등록된 decorated function 하나다. `automationbench/rubric/assertions/salesforce.py`의 패턴:

```python
# automationbench/rubric/assertions/salesforce.py
@AssertionRegistry.register("salesforce_contact_phone_equals")
def salesforce_contact_phone_equals(world: WorldState, assertion: dict) -> bool:
    """Check if a Salesforce contact's phone matches the expected value."""
    contact = world.salesforce.get_by_id("contacts", assertion["contact_id"])
    if contact is None:
        return False
    actual = _normalize_phone(getattr(contact, "phone", "") or "")
    expected = _normalize_phone(assertion["phone"])
    return actual == expected
```

negative(must-not-occur) assertion에는 추가로 `@negative_assertion()`이 붙는다:

```python
# automationbench/rubric/registry.py — the decorator
@AssertionRegistry.register("gmail_message_not_sent_to")
@negative_assertion("gmail")
def gmail_message_not_sent_to(world: WorldState, assertion: dict) -> bool:
    # returns True if the forbidden message was NOT sent
    ...
```

`negative_assertion`은 `fn._negative_assertion = True`로 표시한다(`registry.py:103`). `rubric/__init__.py`의 `partial_credit` scorer는 이 flag를 사용하여 다음을 강제한다: negative assertion은 해당 task의 모든 positive assertion이 통과할 때만 credit을 받는다. decorator는 선택적인 app-name argument를 받지만 이는 문서 목적뿐이며 runtime 효과는 없다.

새 assertion 파일을 추가한 후에는 `automationbench/rubric/assertions/__init__.py`에 import해야 side-effect 등록이 실행된다. 검증: `uv run pytest tests/test_assertions.py`.

---

## Stage 3 — AutomationBench as a verifiers / RL environment

### The dense reward

`automationbench/rubric/__init__.py:153`은 설계 의도를 직접 인용한다:

```python
def create_rubric():
    """
    ``partial_credit`` is the primary reward (weight 1.0) — denser signal for
    training and iterative development. ``task_completed_correctly`` is the
    strict 0/1 benchmark metric, weighted 0.0 so it doesn't affect ``reward``
    but is still surfaced in the eval output.
    """
    import verifiers as vf
    return vf.Rubric(
        funcs=[partial_credit, task_completed_correctly],
        weights=[1.0, 0.0],
    )
```

`partial_credit`는 `passed / total`(0.0–1.0)이다. 이것이 `verifiers` training loop이 누적하는 `reward` field다. `task_completed_correctly`는 metric으로 logging되지만 weight가 0.0이므로 gradient를 형성하지 않는다. 이 설계는 "모델이 개선되고 있는가?"(partial credit, dense)와 "모델이 성공했는가?"(binary pass rate, honest)를 분리한다.

free-assertion exclusion rule(`rubric/__init__.py:52–120`)은 anti-reward-hacking 메커니즘이다: `initial_state`에서 이미 만족된 assertion은 agent가 그것을 깨뜨리지 않는 한 분모에서 제외된다(깨뜨리면 실패로 count됨). 이는 아무것도 하지 않는 policy가 미리 seed된 "free" assertion으로 partial credit을 얻는 것을 방지한다.

### The verifiers base class

`runner.py:38`에서 상속 구조를 보여 준다:

```python
# automationbench/runner.py
class AutomationBenchEnv(vf.StatefulToolEnv):
    def __init__(self, dataset, rubric, tools=None, max_turns=25,
                 toolset="zapier", search_top_k=None, **kwargs):
        ...
```

`vf.StatefulToolEnv`는 `verifiers` library(Prime Intellect / Will Brown)에서 온다. `AutomationBenchEnv`는 `setup_state`(`WorldState` deserialize), tool dispatch(`update_tool_args`가 `world`를 re-inject), grading path를 override한다. `eval.py:243`의 `evaluate` 호출은 training loop가 만들 것과 동일한 호출이다 — 차이점은 training에서는 `rollouts_per_example > 1`로 설정하고 `reward`를 print하는 대신 PPO/GRPO optimizer에 feed한다는 것이다.

### Prime Intellect Environments Hub

benchmark가 Prime Intellect의 hosted runner에 등록되어 있다:

```bash
# README.md — Prime Intellect Environments Hub
prime env install zapier/AutomationBench
prime eval run zapier/AutomationBench

# Smoke test with 5 examples
prime eval run zapier/AutomationBench --num-examples 5

# Run a single domain
prime eval run zapier/AutomationBench --env-args '{"domains": "sales"}'
```

이는 environment를 직접 호스팅하지 않고도 AutomationBench를 RL training target으로 사용할 수 있다는 뜻이다 — Prime Intellect client를 통해 policy model을 연결하면 `partial_credit`이 reward로 흘러들어온다.

### Why dense reward matters for multi-step tasks

8개의 assertion과 25-turn budget을 가진 task는 episode 경계에서 sparse 0/1 signal을 가진다. dense reward는 optimizer에게 *policy가 이미 어떤 sub-goal에 도달했는지*를 알려 준다(예: 올바른 account를 찾아 stage를 업데이트했지만 escalation email은 보내지 못함). partial credit 없이는 모든 미완성 episode가 동등하게 잘못된 것이다. partial credit이 있으면 policy gradient는 "8개 중 5개 맞음"과 "8개 중 0개 맞음"을 구분할 수 있다 — 수학 RL에서 process reward model vs outcome reward model을 동기부여하는 것과 같은 통찰이다.

---

## Stage 4 — Designing your own agent benchmark

AutomationBench의 설계 결정은 하나의 template이다. 새 domain에 적용할 때 체크리스트는 다음과 같다:

**1. Type the world.** agent가 건드릴 수 있는 모든 mutable object는 `extra="forbid"`를 가진 Pydantic model이다. schemaless dict는 없다. 이는 assertion을 trivially correct하게 만들고 state inspection을 무료로 만든다. [[automationbench-harness]]는 이것을 "determinism을 보장하는 핵심"이라고 부른다.

**2. End-state assertions, not output parsing.** agent가 말한 것이 아니라 episode *이후* world가 어떻게 생겼는지를 grade하라. `answer`는 모든 AutomationBench task에서 항상 `""`다. programmatic check가 존재하는 곳에서는 LLM-judge를 쓰지 않는다([[automationbench-tasks-grading]]).

**3. Negative guard는 선택이 아니다.** positive assertion만 있는 benchmark는 shotgun behavior(모든 것을 만들고, 모두에게 보내고)에 의해 reward-hackable하다. routing이나 selective action을 포함하는 모든 task에는 최소 하나의 `*_not_sent_to` 또는 `*_not_exists` guard가 필요하며, duplicate-create 전략을 막는 count-lock assertion(`salesforce_collection_count_equals`)도 필요하다.

**4. Seed noise by `example_id`.** deterministic noise injection(`_noise.py:apply_noise`)은 같은 `example_id`가 항상 같은 world를 생성함을 보장한다 — run variance < 1%. 이는 pass^k를 저렴하게 만든다: 같은 task를 10번 반복하는 데 드는 것은 API budget의 10배이지, 10개의 다른 task를 만드는 engineering effort의 10배가 아니다.

**5. A sanity domain is a harness-validity control.** AutomationBench는 `simple`(200 tasks, frontier model 약 97% 점수)을 제공한다. 낮은 main-domain score는 harness 오류가 아닌 실제 agent 실패다. [[insights]]는 이것을 "sanity domain은 harness-validity control"이라 부른다. 모든 benchmark는 동등한 것을 제공해야 한다.

**6. Cost as a first-class metric.** `_extract_usage_and_debug`는 turn별로 token을 누적하고, `pricing.py`가 model → price를 resolve한다. task당 cost는 모든 exported JSON에 있다. visualizer의 기본 scatter plot은 **cost vs score**다. \$0.05/task에 65%를 얻는 모델이 deployment에 따라 \$0.50/task에 70%를 얻는 모델보다 더 나을 수 있다.

**7. Two metrics from one rubric.** training signal과 개발자 iteration을 위한 `partial_credit`; 정직한 reporting을 위한 `task_completed_correctly`(binary). 절대로 혼동하지 마라.

---

## Stage 5 — Transfer to Lina TMR

### The structural mismatch

[[benchmark-comparison]]에서: AutomationBench는 back-office automator(user 없음, 많은 app, buried policy)다. Lina TMR은 conversational sales agent(multi-turn prospect, one domain, explicit goals)다. 빌려올 evaluation engineering은 toolset이 아니라 assertion framework이다.

Lina의 eval 문제는 llm-training ch-29에서 파킹된 sim-to-real / end-model-eval 질문과 정확히 같다: τ-style user simulator가 대화 측을 생성하지만 simulator 자체가 실제 prospect에서 drift할 수 있고, grader는 그 위에 또 다른 LLM-judge layer를 추가해서는 안 된다. AutomationBench의 programmatic end-state rubric이 이 gap을 닫는다: 대화가 끝나고, world가 검사되고, assertion이 실행된다.

### The recipe

세 가지 component를 결합하라:

```
τ-style user simulator   ←→   Lina agent   →   typed CRM world
                                                      ↓
                                          AssertionRegistry.check(world, assertions)
                                                      ↓
                                          partial_credit (RL reward)
                                          task_completed_correctly (eval metric)
                                          pass^k (reliability, k=10 cheap due to determinism)
```

**τ-style user simulator**: hidden `goal_state`를 가진 prospect로 prompted된 LLM(예: `{"budget": 50000, "timeline": "Q3", "pain_point": "manual_reporting"}`). simulator는 요청을 받았을 때만 정보를 공개하고, objection을 시뮬레이션하고, 통화를 종료한다. 이는 static trigger가 제공할 수 없는 대화 variance를 제공한다.

**Typed CRM world**: `contacts`, `opportunities`, `call_logs`, `scheduled_followups`, `disqualified_leads`를 가진 Pydantic `LinaCRMState`. Lina가 취할 수 있는 모든 action(note 생성, follow-up 예약, deal stage 업데이트, sequence에 추가)이 in-process로 이 world를 mutate한다.

**Assertion rubric**: 통화가 끝난 후 end-state assertion이 world를 grade한다. 예시 task:

```
Task: Qualify a cold prospect (VP of Engineering, 50-person SaaS startup).
Goal: Identify MEDDIC fields, advance to demo stage if qualified, log a follow-up.

Assertions (must-pass):
  {"type": "crm_opportunity_stage_equals", "opp_id": "opp_lina_001",
   "stage": "Demo Scheduled"}
  {"type": "crm_call_log_exists", "contact_id": "contact_001",
   "contains": ["budget", "timeline"]}
  {"type": "crm_followup_scheduled", "contact_id": "contact_001"}

Assertions (must-not-occur / negative guards):
  {"type": "crm_lead_not_disqualified_without_reason", "contact_id": "contact_001"}
  {"type": "crm_no_duplicate_opportunity", "contact_id": "contact_001"}

Noise:
  Three other contacts in the CRM with similar company profiles (seeded by example_id).
  One with a compliance hold note: "Do not contact — pending legal review."
```

**Sanity domain**: 첫 번째 turn에서 prospect가 즉시 budget과 timeline을 공개하는 20개 task(elicitation 불필요). 제대로 작동하는 Lina agent는 >90% 점수를 내야 한다. 낮은 sanity score = 깨진 harness, 깨진 agent가 아님.

**Metrics**:
- `partial_credit`(dense, 0–1) — GRPO/PPO training의 RL reward.
- `task_completed_correctly`(binary) — eval headline.
- pass^k (k=10): 각 `example_id`를 10번 실행(저렴 — deterministic world, simulator만 variance를 도입). 70% pass rate에 pass^10=0.02는 production에서 실격; pass^10=0.45는 acceptable. [[benchmark-comparison]] §Two metric philosophies.
- 통화당 cost(tokens × price), AutomationBench와 동일한 JSON export format으로 surfaced.

---

## Deliverable memo

아래 두 가지 옵션 중 **하나**를 선택해 `ch-10-memo.md`로 repo에 포함하라.

### Option A — New AutomationBench task + rubric

완전히 실행 가능한 task constructor. 필수 섹션:

1. **Task name and domain.** `sales.my_new_task` 또는 `hr.my_new_task`.
2. **Trigger prompt.** 자연어, prompt에 answer 없음.
3. **Initial world.** `initial_state` dict — 채워진 app만, 현실적인 seed data, domain noise pool에서 최소 하나의 noise contact/record.
4. **Assertion set.** 최소: positive assertion 2개(하나는 cross-app), count-lock 1개, negative guard 1개. 각각 justify하라.
5. **Run evidence.** `uv run pytest tests/test_assertions.py` 아래에서 assertion error 없이 task가 실행됐음을 보여 주는 터미널 출력 또는 JSON snippet.

### Option B — Lina TMR benchmark spec

1–2 페이지 spec. 필수 섹션:

1. **Typed world.** `LinaCRMState` schema: 어떤 collection, 어떤 field, 어떤 `extra="forbid"` Pydantic model. agent가 건드릴 수 있는 모든 app state의 이름.
2. **Assertion set.** 최소 positive 4개 + negative guard 2개. 각각에 대해 dict shape를 제시하고 왜 gamed될 수 없는지 justify하라.
3. **Seeded noise.** noise pool이 어떻게 생겼는지 — 최소 compliance-hold decoy 하나와 near-match entity trap 하나(유사한 이름, 다른 contact_id).
4. **Sanity domain.** 제대로 작동하는 agent가 trivially 통과해야 할 기준을 가진 예시 task 3개.
5. **Metrics.** `partial_credit`, `task_completed_correctly`, pass^k (k=10), cost/call. sales-call 시나리오에서 pass^k가 왜 올바른 headline인지 설명하라.

### Acceptance criteria (both options)

Hard gate:

1. assertion set의 모든 `"type"`이 registered handler에 mapping된다(또는 registration을 추가했다). `AssertionRegistry.check(world, assertion)` 실행 시 `ValueError: Unknown assertion type`이 raise되지 않는다.
2. 최소 하나의 negative guard가 존재하며 agent가 모든 contact에 대해 행동할 경우(shotgun behavior) fire될 것이다.
3. count-lock 또는 동등한 것이 duplicate-create 전략이 partial credit을 얻는 것을 방지한다.
4. sanity domain / sanity task가 올바른 agent가 ≥90% 점수를 낼 만큼 충분히 쉽다.
5. Noise가 `example_id`로 seed된다 — 같은 task를 두 번 실행하면 같은 world를 본다.
6. memo가 rubric이 포착하지 못하는 하나의 specific한 failure mode를 명시한다(즉, benchmark의 blind spot에 대한 솔직한 인정).

---

## Connections

- **ch-02** — `AutomationBenchEnv(vf.StatefulToolEnv)`는 그곳에서 처음 만난 `verifiers` base class의 concrete instantiation이다.
- **ch-03** — typed in-process world(`WorldState`, 44개 Pydantic app state)는 ch-03의 environment architecture 논의에서 온 design pattern이다.
- **ch-05 / ch-06** — task anatomy(trigger + initial state + assertion rubric)와 hardening(decoy, negative guard, free-assertion exclusion)은 ch-05/06 개념이 여기에 instantiated된 것이다.
- **ch-09** — [[benchmark-comparison]]의 구조적 대비(AB vs τ-bench)가 Stage 5를 동기부여한다.
- **llm-training ch-29** — 그곳에 파킹된 Lina TMR sim-to-real / end-model-eval 문제가 이 lab의 Stage 5에서 닫힌다.

## Further reading

- [[automationbench-harness]] — episode lifecycle, world injection, BM25 tool discovery, cost metric.
- [[automationbench-tasks-grading]] — task anatomy, noise mechanism, scoring logic, free-assertion exclusion.
- [[benchmark-comparison]] — AB vs τ-bench 구조적 비교; pass-rate + cost vs pass^k metric 철학.
- [[insights]] — cross-source insight index; "sanity domain is a harness-validity control."
- `verifiers` library (Prime Intellect / Will Brown) — `StatefulToolEnv`, `Rubric`, `evaluate`; `AutomationBenchEnv`가 plug-in하는 training-loop interface.
