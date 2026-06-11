<!-- chapter: ch-08
     track: internals
     kind: content
     title: Metrics, Cost, and Reproducibility
     deps: [ch-07]
     sources: [[automationbench-harness]], [[benchmark-comparison]], [[taubench]]
-->

# 08장 — Metrics, Cost, and Reproducibility

> **핵심 통찰.** AutomationBench는 두 개의 조율된 축 — binary pass-rate와
> cost-per-task — 을 report하며, 전체 "세계"를 하나의 결정론적 in-process Pydantic
> 객체로 시뮬레이션함으로써 거의 0에 가까운 run 분산을 확보한다. 이 조합 덕분에
> 각 task를 수십 번 반복하지 않고도 "이 모델이 일을 해낼 수 있는가, 그리고
> 비용은 얼마인가?"에 답할 수 있다.

> **가이드라인.** pass-rate를 capability gate로, cost-per-task를 deployment
> filter로 사용하라. 두 축을 함께 봐야 값싸고 불안정한 모델과 비싸고 안정적인
> 모델을 구별할 수 있다. 어느 한 축만으로는 구매 결정이 절반밖에 되지 않는다.

---

## 1. 공식 지표: binary pass-rate

리더보드에 공개되는 점수는 **binary pass-rate**다: task는 통과하거나 통과하지
못하거나 둘 중 하나다. 부분 진행은 headline 숫자에 아무런 기여도 하지 않는다.

이 방식은 rubric 레이어에서 강제된다. `rubric/__init__.py`는 두 함수를 정의하고
가중치를 명시적으로 할당한다:

```python
# automationbench/rubric/__init__.py  L143–L166

def task_completed_correctly(state: Any, **kwargs) -> float:
    """Binary pass/fail metric: 1.0 iff every scored assertion passed, else 0.0.

    This is the official benchmark pass-rate signal. It reads the cached
    `partial_credit` value stored by that function, so it avoids re-running
    every assertion.
    """
    return float(state.get("partial_credit", 0.0) == 1.0)


def create_rubric():
    """Create the rubric for AutomationBench task evaluation.

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

두 rubric 함수가 실행되지만, `reward` 필드를 실제로 구동하는 것은 하나뿐이다.
`partial_credit`(weight 1.0)은 분수 점수, 즉 "assertion 중 몇 분의 몇이 통과했는가?"
를 제공하며, 이는 training pipeline과 RL reward signal로 피드백된다(학습 중에는
희소한 0/1보다 촘촘한 gradient가 더 유용하다). `task_completed_correctly`(weight 0.0)
는 엄격한 gate로, `partial_credit`이 정확히 1.0일 때만 1.0을 반환한다. Weight 0.0은
이 함수가 `reward`에 전혀 영향을 주지 않는다는 뜻이다. 출력에는 표시되지만 보상은
전혀 받지 않는다.

CLI가 출력하고 `export_results`가 저장하는 `pass_rate`는 `reward`가 아닌
`task_completed_correctly`로부터 계산된다:

```python
# automationbench/scripts/eval.py  L288–L290

binary_scores = [float(ro.get("metrics", {}).get("task_completed_correctly", 0.0)) for ro in raw_outputs]
pass_rate = sum(binary_scores) / len(binary_scores) if binary_scores else None
print_avg_reward(avg_reward, pass_rate)
```

그리고 `export.py`(L168–L172)에서:

```python
# automationbench/export.py  L167–L171

"pass_rate": (
    sum(1 for t in task_results if t["passed"]) / len(task_results)
    if task_results else 0.0
),
```

여기서 `t["passed"]`는 `reward == 1.0`이다(export.py L57). 공개 데이터셋은
여섯 개 domain에 걸쳐 606개 task(sales 106개, 나머지 다섯 domain 각 100개)로
구성되며, 공식 리더보드에는 비공개 private set 600개 이상이 추가된다. 2026-06
기준 SOTA pass-rate는 12–17%다([[automationbench-overview]]).

### partial_credit는 버려지지 않는다 — 재경로를 탄다

`partial_credit` 값은 `state["partial_credit"]`에 저장되며, export의 task별
`score` 필드(`export.py L55`)에도 등장한다. 목적은 두 가지다: 하네스를 reward
환경으로 직접 사용하는 모든 RL fine-tuning pipeline에 더 촘촘한 signal을 제공하고,
사후 분석("모델이 assertion 4개 중 3개를 완료했다; 빠뜨린 건 어느 것인가?")을
가능하게 한다. binary gate와 fractional score는 공존하며 서로 다른 질문에 답한다.

---

## 2. 1등 지표로서의 cost-per-task

### 턴에 걸친 token accumulation

각 task는 multi-turn이다: 에피소드가 끝나기 전에 모델은 최대 25회 tool call을
발행할 수 있다(기본값 `max_turns`). Token count는 마지막 턴만 표본 추출하는 게
아니라 모든 턴에 걸쳐 합산해야 한다.

`runner.py`의 `_extract_usage_and_debug`는 매 턴이 끝날 때 `env_response` 내부에서
호출된다 — 모델이 응답하고 tool 결과가 반환되기 직전:

```python
# automationbench/runner.py  L185–L213

def _extract_usage_and_debug(self, state: vf.State) -> None:
    """Extract token usage and debug info from the latest trajectory step.

    Called at env_response time to process the most recent model response.
    Reads from state["trajectory"][-1]["response"] (a vf.Response object).
    """
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return

    step = trajectory[-1]
    response = step.get("response")
    if response is None:
        return

    # Extract usage from vf.Response
    usage = getattr(response, "usage", None)
    if usage is not None:
        if "_usage" not in state:
            state["_usage"] = {"input_tokens": 0, "output_tokens": 0}
        state["_usage"]["input_tokens"] += getattr(usage, "prompt_tokens", 0)
        state["_usage"]["output_tokens"] += getattr(usage, "completion_tokens", 0)
```

결과는 턴마다 누적되는 accumulator dict `state["_usage"]`다. `usage.py`의
companion 함수(`extract_usage_from_state`, L43–L74)는 xAI Grok과 OpenAI o-series
모델의 reasoning token을 처리한다 — 이 token은 output token으로 청구되지만
별도의 `completion_tokens_details.reasoning_tokens` 필드에 보고된다. accumulator
패턴 덕분에 초반 턴에서 가장 많은 token이 생성되더라도 25턴 에피소드에서 과소
계산이 발생하지 않는다.

`calculate_run_usage`(`usage.py` L77)는 하나의 run에 속한 모든 task에 걸쳐
집계한다:

```python
# automationbench/usage.py  L102–L126

for task_name, state in rollout_items:
    # Prefer the _usage field we accumulated in add_model_response (via state_columns).
    # Fall back to the framework's token_usage if available, then to trajectory scanning.
    custom_usage = state.get("_usage") or {}
    if custom_usage.get("input_tokens", 0) or custom_usage.get("output_tokens", 0):
        input_tokens = int(custom_usage.get("input_tokens", 0) or 0)
        output_tokens = int(custom_usage.get("output_tokens", 0) or 0)
    else:
        token_usage = state.get("token_usage") or {}
        input_tokens = int(token_usage.get("input_tokens", 0) or 0)
        output_tokens = int(token_usage.get("output_tokens", 0) or 0)

    total_input += input_tokens
    total_output += output_tokens

    # Calculate costs if pricing available
    if pricing is not None:
        input_cost = input_tokens * pricing.input_cost_per_token
        output_cost = output_tokens * pricing.output_cost_per_token
        total_cost = input_cost + output_cost
```

공식은 `in_tok * in_price + out_tok * out_price`이며, task별로 적용한 후 합산한다.
`RunUsage.total_cost`(`usage.py` L140–L142)는 run 전체 합계에 `pricing.calculate_cost`
를 사용한다:

```python
# automationbench/pricing.py  L199–L201

def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
    """Calculate total cost for given token counts."""
    return input_tokens * self.input_cost_per_token + output_tokens * self.output_cost_per_token
```

### Pricing DB: 네 가지 전략의 lookup과 24시간 cache

모델 이름은 다양한 형식으로 들어온다: `openai/gpt-4o`, `claude-sonnet-4-5-20251001`,
`vertex_ai/gemini-3-flash-preview`. `PricingDatabase.get_pricing`은 실시간 가격
피드를 조회하기 전에 이를 정규화한다.

```python
# automationbench/pricing.py  L293–L344

def get_pricing(self, model: str) -> ModelPricing | None:
    """
    Get pricing for a model.

    Uses multiple lookup strategies:
    1. Exact match
    2. Normalized query against exact keys (strips provider prefixes, date suffixes)
    3. Alias lookup for known naming variations
    4. Normalized query against normalized keys

    Returns None if model pricing is unknown and no CLI override provided.
    """
    # CLI override takes precedence
    if self._input_cost_override is not None and self._output_cost_override is not None:
        return ModelPricing(
            input_cost_per_token=self._input_cost_override,
            output_cost_per_token=self._output_cost_override,
            source="cli-override",
        )

    pricing_data = self._get_pricing_data()

    # Strategy 1: Exact match
    if model in pricing_data:
        result = self._make_pricing(pricing_data[model], pricing_data)
        if result:
            return result

    # Strategy 2: Normalized query against exact keys
    normalized = normalize_model_name(model)
    if normalized in pricing_data:
        result = self._make_pricing(pricing_data[normalized], pricing_data)
        if result:
            return result

    # Strategy 3: Check aliases for known naming variations
    aliased = MODEL_ALIASES.get(normalized)
    if aliased and aliased in pricing_data:
        result = self._make_pricing(pricing_data[aliased], pricing_data)
        if result:
            return result

    # Strategy 4: Normalized query against normalized keys
    for key, entry in pricing_data.items():
        if normalize_model_name(key) == normalized:
            result = self._make_pricing(entry, pricing_data)
            if result:
                return result

    return None
```

정규화(`pricing.py` L50–L88)는 provider prefix(`openrouter/anthropic/`,
`vertex_ai/`, `gemini-dev/` 등), region prefix(`us-east-1/`), date suffix
(`-20251101`, `@20251001`), version tag(`-v1:0`), `-preview` suffix를 제거한다.
docstring 예시: `"openai/gpt-4o"→"gpt-4o"`, `"claude-opus-4-5-20251101"→"claude-opus-4-5"`,
`"vertex_ai/gemini-3-flash-preview"→"gemini-3-flash"`.

데이터 소스는 `llm-prices.com`(L14의 상수 `LLM_PRICES_URL =
"https://www.llm-prices.com/current-v1.json"`)으로, 24시간에 최대 한 번 조회하고
`~/.cache/automationbench/model_prices.json`에 cache한다(`CACHE_TTL_SECONDS = 24 * 60 * 60`,
L18). cache miss나 네트워크 실패 시 코드는 하드코딩된 `FALLBACK_PRICING` dict
(`pricing.py` L93–L172)로 폴백한다. 이 dict에는 현 세대 모델 패밀리 전체의 항목이
들어 있으며 — 주석에 따르면 2026년 3월 기준으로 업데이트됐다. `ModelPricing`의
`source` 필드는 어떤 경로가 사용됐는지 기록한다: `"llm-prices"`, `"fallback"`,
또는 `"cli-override"`.

### CLI override는 무조건 우선한다

두 flag로 pricing DB 전체를 우회할 수 있다:

```python
# automationbench/scripts/eval.py  L369–L377  (argparse section)

parser.add_argument(
    "--input-cost",
    type=float,
    default=None,
    help="Per-token input cost in USD (overrides pricing lookup)",
)
parser.add_argument(
    "--output-cost",
    type=float,
    default=None,
    help="Per-token output cost in USD (overrides pricing lookup)",
)
```

`--input-cost`와 `--output-cost`가 모두 제공되면, `get_pricing`은 DB 조회를 전혀
하지 않고 즉시 `source="cli-override"`로 반환한다(pricing.py L306–L311). 이는
로컬 vLLM 모델, 협상된 엔터프라이즈 가격, 또는 아직 `llm-prices.com`에 등록되지
않은 inference provider에 적합한 override 방법이다.

---

## 3. Reproducibility: 결정론적 세계, 시드된 noise, <1% 분산

τ-bench(115개 retail task)는 10회 이상 반복이 필요하다 — 시뮬레이션된 LLM 사용자가
매 run마다 확률적 분산을 주입하기 때문이다(턴 순서, 자발적으로 제공하는 정보,
종료 신호)([[taubench]]). AutomationBench는 근본적으로 다른 보장을 달성한다.

전체 "세계"는 **하나의 in-process Pydantic 객체**다(`WorldState`, `schema/world.py`).
HTTP 서버도, 데이터베이스 프로세스도, 서브프로세스도 없다 — 모든 tool call은
메모리 내에서 직접 `world.<app>.<collection>`을 변경한다. 무작위성(distractor
레코드, noise 전화번호)은 런타임이 아닌 authoring 시점에 task 정의에 시드된다.
같은 task dict는 매 run마다 같은 world를 생성한다.

`setup_state`(`runner.py` L136`)는 free-assertion 감지를 위해 초기 world dict를
deep-copy한다(`state["initial_state"] = copy.deepcopy(initial_state_dict)`). 이
복사본도 동일한 결정론적 seed 데이터를 담고 있다. 확률적 사용자 시뮬레이터의
sampling이 전혀 없다. 결과: run-to-run variance는 일반적으로 **<1%**다
([[automationbench-harness]], [[automationbench-overview]]).

실용적 의미: 공개된 606개 task를 한 번 통과하는 것만으로 통계적으로 안정적인
추정치를 얻을 수 있다. 숫자를 신뢰하기 위해 suite를 10회 반복할 필요가 없다.
606개 task에서 1 percentage point의 움직임은 실제 신호이지 noise가 아니다.

---

## 4. harness 유효성 대조군으로서의 `simple` domain

낮은 pass-rate는 두 가지를 의미할 수 있다: task가 어렵다, 또는 harness가 망가졌다
(잘못된 tool schema, 나쁜 assertion, 망가진 world mutation). AutomationBench는
**200개 task로 구성된 `simple` domain**으로 이 둘을 분리한다.

`domains/simple/tasks.py`(L23–80)의 sample task가 구조를 보여 준다:

```python
# automationbench/domains/simple/tasks.py  L23–80  (excerpt)

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
        "info": {
            "zapier_tools": [
                "gmail_find_email",
                "gmail_get_email_by_id",
                "salesforce_find_records",
                "salesforce_contact_update",
            ],
            "initial_state": { ... }
        }
    }
```

simple task는 메인 벤치마크 task와 동일한 tool 인프라를 요구한다 — 동일한
`AutomationBenchEnv`, 동일한 `WorldState`, 동일한 rubric — 하지만 도구를 하나
또는 두 개만 사용하고, policy 제약도, distractor도, noise 주입도 없다. 올바르게
작동하는 모델이라면 어느 것이든 이것들을 사소하게 완료해야 한다.

관찰된 결과: 소형 모델도 **`simple`에서 ~97% pass-rate**를 달성한다
([[automationbench-overview]]). 이 숫자는 이중 보장을 제공한다. 첫째, run에서
`simple`이 97%이고 메인 domain이 8%라면 harness는 정상 작동 중이다 — 낮은 점수는
실제 orchestration 난이도이지 버그가 아니다. 둘째, run에서 `simple`이 50%라면
메인 점수를 읽는 것을 멈춰라: API 키, tool dispatch, 또는 환경 설정에 문제가 있다.

`simple` domain은 `DEFAULT_DOMAINS`에 포함되지 않는다(`domains/__init__.py` L34):

```python
# automationbench/domains/__init__.py  L33–L34

PUBLIC_DOMAINS = ["sales", "marketing", "operations", "support", "finance", "hr"]
DEFAULT_DOMAINS = list(PUBLIC_DOMAINS)
```

`--domains simple`을 명시적으로 전달해야 한다. 설계상 분리돼 있다 — 쉬운 task를
기본 run에 섞으면 headline pass-rate가 부풀려질 것이다.

---

## 5. run 읽기: export JSON, assertion별 결과, end state

`export_results`(`export.py` L28)는 완료된 run에서 visualizer로 나가는 단일 출구다.
모든 task는 구조화된 레코드를 생성한다:

```python
# automationbench/export.py  L128–L152

task_result = {
    "id": i + 1,
    "name": task_name,
    "score": float(reward),          # partial_credit (0.0–1.0)
    "passed": reward == 1.0,         # binary gate
    "assertions_total": assertions_total,
    "assertions_passed": assertions_passed,
    "assertion_results": assertion_results,  # per-assertion pass/fail + excluded flag
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "cost": task_cost,
    "steps": steps,
    "messages": messages,            # full chat completion
    "end_state": output.get("_end_state"),
}
```

`assertion_results` 리스트가 핵심 진단 필드다. 각 항목은 `type`, `passed`,
`excluded`, `params`를 담는다. 에이전트가 행동하기 전에 이미 충족되지 않았던
assertion에서 `passed=False`(`excluded=False`)가 나오면 구체적인 실패를 포착한
것이다: assertion type 이름, 기대했던 정확한 파라미터, 그것을 충족하지 못한
end-state world. 무엇이 잘못됐는지에 대한 모호함이 없다.

`_end_state`는 에피소드 종료 시점의 `WorldState`를 완전히 `model_dump()`한 것이다
(`rubric/__init__.py` L126). 같은 export 레코드 내의 `initial_state`와 비교하면
에피소드를 머릿속으로 재생할 수 있다: 에이전트가 무엇을 바꿨는지, 무엇을 그대로
뒀는지, 무엇을 손상시켰는지.

export JSON의 최상위 `summary` 블록(`export.py` L154–L196)은 다음을 수집한다:
`avg_score`(평균 partial credit), `pass_rate`(binary), `passed_count`, `failed_count`,
`total_input_tokens`, `total_output_tokens`, `total_cost`, `cost_formatted`, 그리고
API나 streaming 문제를 나타내는 empty response와 zero-output task에 대한 debug 카운터.

### "잘못된 tool call에 대한 거짓 확신"은 어떻게 보이는가

지배적인 실패 모드 — **실패의 72–91%**([[automationbench-overview]]) — 는 모델이
자신 있게 명시했지만 잘못된 인수로 tool을 호출하고, tool이 오류나 조용히 잘못된
결과를 반환하며, 모델이 어쨌든 성공을 선언하는 것이다. export는 두 경로를 통해
이를 표면화한다:

1. `assertion_results`는 실패한 특정 assertion을 보여 준다(예: task
   `simple.email_sf_contact_phone_update`에 대한 `contact_phone_equals`). 에이전트가
   `salesforce_contact_update`를 호출했음에도 end-state의 전화번호가 기대값과 일치하지
   않는다.

2. `_debug`의 `finish_reasons`와 `empty_responses`(`runner.py` L215–L231)는 모델이
   task 도중 생성을 멈추거나 빈 completion을 내보내는 경우를 포착한다 — tool 결과를
   잘못 읽고 완료로 해석했다는 신호다.

이 두 필드를 함께 보면 세 가지 실패 클래스를 구별할 수 있다: 잘못된 인수(올바른
tool, 잘못된 데이터), 잘못된 tool(올바른 의도, 잘못된 tool 이름), 조기 종료(여기까지는
올바르지만 모든 단계 전에 중단).

---

## 6. pass-rate + cost vs τ-bench pass^k: 두 가지 deployment 질문

AutomationBench와 τ-bench가 서로 다른 headline 지표를 보고하는 것은 deployment
준비도에 관해 서로 다른 질문에 답하기 때문이다([[benchmark-comparison]]).

### pass^k 정의

τ-bench의 pass^k는 task set 전체에 걸쳐 평균한, 같은 task에 대한 **k번의 독립 시행에
걸친 reliability**를 측정한다([[taubench]]):

```
pass^k = (1/|T|) Σ_i p̂_i^k
```

불편 추정량은 `ρ(n, c, k) = 1 − C(n−c, k) / C(n, k)` (n은 시행 횟수, c는 통과
횟수). 90% 확률로 성공하는 task는 pass^8에 `0.9^8 ≈ 0.43`만 기여한다. Reliability는
빠르게 붕괴한다. 원래 τ-bench 수치: GPT-4o는 pass^1 retail ~61%에서 pass^8 25% 미만으로
떨어진다 — 같은 모델, 같은 task, k=1 대신 k=8로 측정했을 뿐이다.

τ-bench가 pass^k를 **필요로 하는** 이유는 확률적 사용자 시뮬레이터가 상당한
run-to-run variance를 유발하기 때문이다. 시뮬레이션된 사용자가 우연히 선택하는
표현에 따라 같은 task가 성공하거나 실패할 수 있다. k=10 이상의 반복 실행만이
에이전트가 반복 배포에서 실제로 어떻게 동작하는지에 대한 안정적인 추정치를
얻는 유일한 방법이다.

### AutomationBench가 pass-rate + cost를 사용하는 이유

AutomationBench의 결정론은 인센티브 계산을 다르게 만든다. <1% 분산으로, 606개
task에 걸친 단일 run은 안정적인 추정치를 제공한다 — 각 task에서 모델이 어떻게
수행하는지 이미 알 수 있다. k번 반복하면 k배의 컴퓨팅 비용이 들고 거의 동일한
숫자가 나온다. 결정론이 주어지면 pass^k는 통계적으로 중복이다.

두 번째 축 — **cost-per-task** — 가 그 공백을 채운다. 비슷한 pass-rate를 가진
두 모델이 있을 때, cost-per-task는 어느 것이 배포 가능한지 답한다. pass-rate
15%에서 task당 \$0.08인 모델은 pass-rate 17%에서 task당 \$0.40인 모델과 다른
범주에 있다. 이 축은 τ-bench에 유사한 것이 없다 — τ-bench는 token 사용량이나
비용을 first-class 지표로 export하지 않기 때문이다.

### AutomationBench가 *할 수 있는* 것

harness는 pass^k를 저렴하게 채택할 수 있다: 결정론은 task를 10번 반복하는 것이
k개의 동일한 seed로 하나의 task를 실행하는 것과 동등하다는 의미다. 하지만 거의
아무런 정보도 추가하지 않을 것이다 — 그 지표가 잡으려는 분산이 존재하지 않는다.
설계 선택은 의도적이다: capability-at-a-price(back-office deployment 질문)를
보고하고, reliability-over-trials(customer-service deployment 질문)는 보고하지
않는다([[benchmark-comparison]]).

두 질문은 deployment context에 매핑된다:

| 질문 | 지표 | 중요한 경우 |
|------|------|------------|
| "이 모델이 일을 해낼 수 있고 비용은 얼마인가?" | pass-rate + cost/task | Back-office automation; 각 task가 고유함; 처리량과 경제성이 결정을 좌우함 |
| "같은 고객이 매번 물어볼 때마다 성공할 것인가?" | pass^k | Customer-service 에이전트; 같은 task가 반복됨; 20% 실패율은 실격 사유 |

잘못된 deployment context에 잘못된 지표를 선택하면 양방향으로 거짓 확신을 갖게
된다: pass^k에 최적화된 모델은 비싸고 느릴 수 있다; pass-rate + cost에 최적화된
모델은 같은 task가 반복될 때 개별 task 수준에서 불안정할 수 있다.

---

## Summary

AutomationBench의 측정 시스템은 내적으로 일관성이 있다: binary pass-rate는 각
task가 고유한 비즈니스 이벤트인 벤치마크에 적합한 headline이고; cost-per-task는
deployment 경제성을 평가하는 올바른 두 번째 축이며; 결정론(<1% 분산)은 반복 없이도
두 지표를 안정적으로 만들고; `simple` domain은 메인 domain 점수를 신뢰하기 전에
harness를 검증하며; assertion별 export는 단순한 0이 아닌 구체적인 실패를 제공한다.
τ-bench의 pass^k와의 대비는 결함이 아니다 — 각 벤치마크가 답하도록 설계된 deployment
질문에 관한 선택이다.

**Next:** [[ch-09]] — τ-bench 및 τ²-bench와의 구조적 일대일 비교 전체.
