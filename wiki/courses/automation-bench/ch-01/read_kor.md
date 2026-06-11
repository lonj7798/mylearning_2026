<!-- chapter: ch-01
     track: landscape
     kind: content
     title: The Agentic Tool-Call Benchmark Landscape
     deps: []
     sources: [[automationbench-overview]], [[taubench]], [[benchmark-comparison]]
-->

# 01장 — The Agentic Tool-Call Benchmark Landscape

> **핵심 통찰.** 이 분야는 2022–2024년 동안 모델이 소수의 candidate set에서 올바른 function을 *선택*할 수 있는지를 평가하는 데 집중했다. 그 bar는 이제 fine-tuned 7B 모델도 넘는다. 여전히 어려운 것 — 그리고 AutomationBench가 측정하도록 설계된 것 — 은 agent가 400개의 tool 중 어느 것을 사용할지 *discover*하고, 여러 application에 걸쳐 조율하며, seeded artifact 속에 묻힌 policy를 따르고, 각 시스템에 올바른 final state를 남길 수 있는지다. 이 세 가지 요구사항을 모두 결합한 prior benchmark가 없었기 때문에 frontier model들은 여기서 20% 미만의 점수를 기록한다.

> **가이드라인.** agentic system에 맞는 benchmark를 선택할 때, benchmark의 *structure*를 agent의 deployment shape에 맞춰라: 잘못된 shape의 benchmark에서 높은 점수를 받는 것은 거의 예측력이 없다. AutomationBench는 back-office automator(단일 trigger, 사용자 없음, 다수 app, find-your-own-tools)를 측정한다. τ-bench는 customer-service conversationalist(interrogate해야 하는 multi-turn 사용자, 하나의 app, 주어진 tools)를 측정한다. AutomationBench 점수로 τ-bench 성능을 예측하거나 — 또는 그 반대 — 는 category error다.

---

## 1. The Bar Moved: From Function Selection to Agentic Orchestration

"tool-call benchmark"라는 표현은 2022년과 2026년 사이에 의미가 바뀌었다. 초기 세대 — Berkeley Function-Calling Leaderboard(BFCL), API-Bank, ToolBench — 는 좁은 질문을 던졌다: 자연어 instruction과 *사전 제공된 candidate tool set*이 주어졌을 때, 모델이 올바른 것을 선택하고 구문적으로 유효한 call을 emit할 수 있는가? task는 single-turn이었고, tool set은 건네졌으며, ground truth는 function 이름과 argument schema였다.

그 질문은 raw function-calling reliability를 측정하는 진단 가치가 여전히 있지만, deployment-grade agentic behavior에 대해서는 거의 아무것도 알려주지 않는다. 프로덕션에서 agent는:

1. 사전 필터링된 candidate set을 받지 않는다 — catalog를 검색해야 한다.
2. 한 step에서 끝나지 않는다 — call을 chain하고, 중간 결과를 읽고, 분기한다.
3. 단일 application 안에서만 작동하지 않는다 — enterprise workflow는 하나의 task에서 CRM, email, calendar, spreadsheet, ticketing, messaging에 걸쳐 있다.
4. prompt에 없는 policy를 적용해야 한다 — 그것은 spreadsheet 행, inbox 메시지, agent가 찾아야 하는 document에 있다.

이 각각의 요구사항은 원래의 BFCL-style benchmark를 더 어려운 문제의 proxy로서 불충분하게 만든다. 이 분야의 대응은 일련의 escalating benchmark들이었는데, 각각이 하나 또는 두 개의 axis를 더 열면서 다른 것들은 고정했다. AutomationBench의 기여는 네 가지 요구사항을 모두 단일 eval에, 스케일로, programmatic grading과 함께 결합하는 것이다.

### The escalation timeline in brief

| Year | Benchmark | What it added |
|------|-----------|---------------|
| 2022–23 | BFCL, API-Bank, ToolBench | candidate set에서의 function selection; single-turn 또는 얕은 multi-step |
| 2023 | AppWorld | end-state grading, 단일 simulated environment 내의 현실적인 multi-step |
| 2023–24 | WebArena, Mind2Web | tool substrate로서의 web browsing; UI grounding; real web pages |
| 2024 | τ-bench | Multi-turn user simulator(triad: tool + agent + user); pass^k reliability metric; domain policy compliance |
| 2024–25 | τ²/τ³ | dual-control tasks; task-quality fixes; RAG/doc discovery(τ³ banking domain) |
| 2026 | **AutomationBench** | cross-application coordination + ~400-tool catalog에서의 autonomous API discovery + policy-in-artifacts, 모두 하나의 task에서 |

"What it added" 열은 "독점적으로 측정하는 것"이 아니다. BFCL은 여전히 raw function-calling precision 진단에 중요하다. τ-bench는 여전히 conversational reliability에 중요하다. AutomationBench는 enterprise automation at scale을 지배하는 특정 shape의 작업에 대한 gap을 채운다.

---

## 2. The Benchmark Family in Detail

### 2.1 BFCL / API-Bank / ToolBench

이 benchmark들은 구조적 가정을 공유한다: candidate tool set이 주어진다. BFCL은 모델이 올바른 argument로 올바른 function을 호출하는지 평가한다; *결과*(world state)가 아니라 *output*(call text)을 grading한다. API-Bank와 ToolBench는 retrieved tool documentation이 있는 multi-step chain으로 이것을 확장했지만, retrieval은 제공되거나 search-assisted다 — agent가 대규모의 미분류 catalog에서 올바른 tool을 discover할 수 있는지는 평가하지 않는다.

이것들은 function-calling mechanics를 위한 빠르고 저렴한 regression test로 올바르게 사용된다. cross-application orchestration의 유효한 proxy가 아니다.

### 2.2 AppWorld

AppWorld는 단일 simulated multi-app environment 내에서 end-state grading을 도입했다: agent는 task를 완료해야 하고 evaluator는 결과 world state가 예상 state와 일치하는지 확인한다. 이것은 AutomationBench가 만드는 것과 동일한 철학적 움직임(output text가 아닌 결과를 grading)이며, AppWorld는 그 설계에 대한 credit을 받을 만하다. 제약은 task가 single-environment라는 것이다: agent가 독립적인 schema를 가진 application 간에 조율할 필요가 없으며, tool은 사전 제공된다. [[automationbench-overview]]는 Table 1 항목에서 대비를 요약한다(AppWorld 행: cross-app ✗, API discovery ✗, end-state ✓, business rules ✗).

### 2.3 WebArena / Mind2Web

WebArena와 Mind2Web은 실제 또는 replayed web browser를 action substrate로 사용한다 — 클릭, form-filling, 내비게이션. grading은 부분적으로 결과 기반이다(agent가 올바른 페이지에 도달했는가, 올바른 필드를 채웠는가?). 이 benchmark들은 *UI-grounded* tool use를 측정하며, programmatic API use가 아니다. failure mode가 다르고(HTML parsing, element localization, session management), skill이 AutomationBench가 차지하는 API-invocation 세계로 불완전하게 transfer된다.

### 2.4 τ-bench / τ² / τ³

τ-bench([[taubench]])는 현재 landscape에서 AutomationBench와 구조적으로 가장 다르며, 대비로서 이해하는 것이 가장 중요하다.

τ-bench task는 네 부분으로 구성된다: (1) agent에게 절대 보여지지 않는 LLM user simulator가 보유한 사용자 *goal*; (2) 가변 JSON database state; (3) agent가 읽을 수 있는 posted policy wiki; (4) 고정된 Python-backed tool set. agent는 multi-turn 대화를 통해 goal을 *elicit*하고, domain policy를 따르며, database를 올바른 final state로 남겨야 한다. grading은 `final_DB_state == goal_state`, 정확히, 이진 — 역시 AutomationBench와 공유되는 output이 아닌 결과 철학이다.

정의적 특징은 **user simulator**다: 별도의 LLM이 고객을 연기하고, 묻지 않은 정보를 보류하며, `###STOP###`으로 종료한다. 이것은 설계에 내재된 확률적 분산을 주입하는데, 버그가 아니다 — simulated user가 다른 정보를 자발적으로 제공하기 때문에 동일한 task가 run마다 변한다. 그 분산이 τ-bench가 **pass^k**를 도입한 이유다:

```
pass^k = (1/|T|) Σ_i p̂_i^k
```

unbiased estimator `ρ(n,c,k) = 1 − C(n−c,k)/C(n,k)`와 함께. `p=0.9` per task에서, `pass^8 ≈ 0.43` — reliability가 빠르게 붕괴한다. 동일한 task에서 90%의 시간에 성공하는 모델은 pass^8에서 43%만 기록할 것이다: 단일 shot 평가에서는 유능해 보이지만 그 metric으로는 거의 배포 불가능하다. `pass^1 = pass@1 = mean success`.

τ²-bench(arXiv 2506.07982)는 **dual-control**을 추가했다: *사용자*도 tool을 가지고(예: 라우터 재부팅, config 읽기) agent는 정보를 단순히 elicit하는 것 이상으로 사용자 action을 가이드해야 한다. GPT-4는 56–74%(single-control)에서 ~34%(dual-control)로 떨어진다. τ³-bench는 audit으로 식별된 ~75개의 결함 있는 task fix를 통합하고(항공사 pass^1은 수정 후 14–20 포인트 상승) RAG/doc-tool discovery가 있는 `banking` domain을 추가했다.

---

## 3. AutomationBench's Three-Part Gap Thesis

AutomationBench의 동기는 [[automationbench-overview]]의 한 문장에 담겨 있다:

> *Real enterprise agentic work is cross-application coordination + autonomous API discovery + policy adherence, all at once — and no prior tool-call benchmark combined the three, so frontier models score <20%.*

논문의 Table 1([[automationbench-overview]]에서 재현)은 gap을 명시적으로 보여준다:

| Benchmark | Cross-app | API discovery | End-state grading | Business rules |
|-----------|-----------|---------------|-------------------|----------------|
| WebArena / Mind2Web | ✗ | ✗ | ✗ | ✗ |
| ToolBench / API-Bank | ✗ | retrieval-assisted | varies | ✗ |
| AppWorld | ✗ (single env) | ✗ | ✓ | ✗ |
| τ³-bench | ✗ (single app) | partial (banking) | ✓ | ✓ |
| **AutomationBench** | **✓** | **✓ (BM25)** | **✓** | **✓** |

각 열은 우연이 아닌 설계 선택이다. 각각이 왜 존재하는지 추적해보자.

### 3.1 Cross-application coordination

Zapier의 프로덕션 catalog는 월 ~20억 개의 task를 처리한다. 고가치 workflow의 압도적 다수는 본질적으로 multi-app이다: 새로운 Salesforce opportunity가 Gmail routing notification을 trigger하고 그 내용은 Google Sheets policy 행과 Google Drive account-hierarchy document에 의존한다. 단일 application benchmark는 agent가 공통 schema를 공유하지 않는 독립적으로 typed된 state object들 간에 조율하는지 테스트할 수 없다.

[[automationbench-tasks-grading]]의 multi-hop sales task(example_id 501)가 이것을 구체적으로 보여준다. trigger가 말한다:

```
"We just closed the Meridian Corp Platform Deal! Mark it won and route the win notice per
our routing policy. Confirm the account tier from the 'Account Hierarchy' sheet, convert
currency (see 'FX Rates'), and check for open support escalations."
```

이것을 해결하려면: Salesforce(Closed Won으로 표시) → Google Drive(두 개의 sheet 찾기) → Google Sheets(tier와 FX rate에 대한 날짜가 충돌하는 행 해결, 가장 최근 것 취하기) → Salesforce 다시(계정 *및* 상위 계정의 escalation 확인) → Gmail(policy에 따라 올바른 팀에 routing). routing policy는 prompt에 없다 — agent가 discover해야 하는 seeded inbox 메시지에 있다.

### 3.2 Autonomous API discovery

기본 `zapier` toolset 모드에서 agent는 정확히 두 개의 tool을 받는다: `search_tools(query, top_k)`와 `execute_tool(tool_name, arguments)`. 기반 catalog에는 ~400개의 named endpoint가 있다. agent는 call하기 전에 어떤 tool이 관련 있는지 파악하기 위해 BM25 query를 발행해야 한다. 이것은 모든 prior benchmark가 회피했거나(tool set을 건네줌으로써) 축소했던(agent에게 사전 필터링된 candidate list를 제공하는 retrieval-assistance로) skill을 테스트한다.

`limited_zapier` 모드는 agent에게 per-task allowlist로 필터링된 전체 named tool set을 제공한다 — 이것은 discovery skill에서 execution skill을 분리하는 ablation이다. `api` 모드는 agent에게 세 개의 REST-shaped tool(`api_search`, `api_fetch`, `base64_encode`)을 제공하고 agent는 유효한 URL을 구성해야 한다. 이 세 모드는 discovery를 일급 측정 가능한 capability로 만들면서도 그것의 통제된 ablation을 허용한다.

### 3.3 Policy adherence

system prompt는 agent에게 명확한 질문을 하지 말라고 지시한다. Policy는 seeded artifact 속에 내장된다 — spreadsheet 행, inbox 메시지, document. agent는 policy를 찾아 파싱하고 올바르게 적용해야 한다. 이것은 τ-bench의 posted policy wiki(agent에게 존재한다고 알려지고 읽을 수 있는)보다 구조적으로 더 어려우며, 어떤 prior benchmark에서도 테스트되지 않았다.

[[automationbench-tasks-grading]]의 예시들: 사용자가 "also process severance"라고 말하지만 policy artifact는 HR이 해서는 안 된다고(Payroll만 가능) 말하는 scope-creep trap; noise contact의 description 필드에 "do not enroll — pending compliance review"가 있는 compliance-hold trap; agent가 같은 sheet의 날짜가 충돌하는 행 중에서 가장 최근의 FX rate를 식별해야 하는 recency-conflict 행.

---

## 4. "Outcomes, Not Output": End-State Grading as a Philosophy

README는 이 원칙을 명시적으로 기술한다:

> **Verifiability** — All tasks must be programmatically verifiable. If we can't automatically check whether a task was completed correctly, it doesn't belong in the benchmark.

```
Every run reports two per-task metrics:
- partial_credit (0.0 - 1.0) - fraction of assertions satisfied.
- task_completed_correctly (0.0 or 1.0) - strict pass/fail; 1.0 only if every assertion passes.
```

이것은 두 가지 더 싼 대안의 원칙적 거부다:

**Output-text grading**(예: "모델이 올바른 회사 이름을 말했는가?")은 실제 효과를 놓친다. 잘못된 Salesforce API를 호출하고, deal stage를 오염시키고, *그러면서도* 올바른 email 제목 줄을 보내는 모델은 output check를 통과하고 outcome check를 실패할 것이다. output grading은 체계적으로 partial failure를 감지하지 못하고 text-fluent hallucination에 과도한 보상을 준다.

**LLM-as-judge**는 programmatic check를 두 번째 모델 호출로 대체한다. AutomationBench는 strict type을 가진 Pydantic model에 대한 순수 Python field comparison을 사용한다. 이것은 자원 제약이 아니다 — 재현성, 비용, 그리고 judge-model bias에 대한 저항을 위한 의도적인 설계 선택이다. assertion registry는 `world.salesforce.opportunities[id].stage_name == "Closed Won"`을 직접 확인한다. 모호함이 없고 judge variance도 없다.

이진 `task_completed_correctly`(공식 pass-rate용)와 부동소수점 `partial_credit`(RL reward signal과 per-stage 디버깅용)의 2단계 scoring 구조는 두 번째 원칙을 반영한다: *metric은 use case와 일치해야 한다*. 비즈니스 사용자에게 agent를 배포하려면 이진 reliability가 필요하다 — 부분적으로 완료된 task는 실패한 task다. agent를 훈련하려면 dense reward signal이 필요하다 — 이진 0/1은 gradient flow에 너무 sparse하다. AutomationBench는 동일한 assertion rubric에서 두 가지를 모두 제공한다.

### Why Zapier built it this way

Zapier는 9,000개 이상의 app integration과 월 ~20억 개의 task를 운영한다. 그들의 내부 의사결정 문제는 프로덕션에 배포할 모델을 선택하는 것이었다. 그들은 적절한 공개 benchmark를 찾지 못했다:

> *Zapier built it internally to decide which models to deploy in production, found no public benchmark adequate, and open-sourced it. Its substrate is Zapier's real catalog (9,000+ app integrations, 66,000+ triggers/actions, ~2B monthly tasks) abstracted into 47 simulated apps and ~500 endpoints across six high-frequency business domains.* — [[automationbench-overview]]

"outcomes, not output" 철학은 프로덕션 맥락에서 직접 따라온다. 프로덕션에서는 결과가 중요하다: deal이 Closed Won으로 표시되었는가, 올바른 팀에 알림이 갔는가, compliance-hold contact가 제외되었는가? benchmark는 이 기준을 deployment 맥락에서 그대로 가져온다.

---

## 5. The Dataset: Scope, Shape, and Synthetic Construction

공개 task set은 여섯 domain에 걸친 606개 task와 200-task `simple` sanity set이다:

| Domain | Tasks | Coverage |
|--------|-------|----------|
| Sales | 106 | CRM, lead management, cross-app workflows |
| Marketing | 100 | Campaigns, ad performance, content ops, brand monitoring |
| Operations | 100 | Facility management, project tracking, vendor workflows, compliance |
| Support | 100 | Ticket routing, SLA monitoring, knowledge base, multi-platform helpdesk |
| Finance | 100 | AP/AR, expenses, reporting, bookkeeping |
| HR | 100 | Recruitment, employee onboarding, time off, payroll |
| **Simple** | **200** | Single- and two-step harness validation(not scored) |

공식 leaderboard를 위해 보류된 600개 이상의 private task는 동일한 분포를 따른다. 공개 set의 점수는 방향적으로 예측 가능하지만 leaderboard 점수와 동일하지 않다.

Task는 실제 고객 workflow의 *shape*에서 합성적으로 생성되고 Zapier의 Agents service의 negative feedback으로 강화되었다 — PII 없음, raw 고객 데이터 없음. generation code는 공개되지 않았다; task는 `domains/*/tasks.py`의 수제 Python constructor dict로 나타난다. 각 task는 결정론적 noise injection을 seed하는 고유한 `example_id`를 가진다 — 동일한 task는 항상 동일한 distractor를 가지며, 이것이 run-to-run variance가 <1%인 이유다.

---

## 6. Why the Field Scores Below 20%: Difficulty by Construction

2026년 중반 현재 공식 leaderboard에서의 SOTA pass-rate는 약 12–17%다. 논문 제출 시 헤드라인은 "모든 SOTA 모델이 10% 미만으로 점수를 낸다"였다. 이 수치들은 더 강한 모델이 평가됨에 따라 적당히 상향되었지만, ceiling은 여전히 매우 낮다.

이것은 broken benchmark의 신호가 아니다 — 잘 calibrated된 benchmark의 신호다. 증거는 **`simple` domain**에서 나온다: 소형 모델조차도 ~97%를 기록한다. [[automationbench-overview]]가 말한다:

> *Even small models hit ~97% on the `simple` domain, confirming low main-benchmark scores reflect genuine orchestration difficulty, not a broken harness.*

simple task는 개별 app에 대한 single- 및 two-step 작업이다. 모델은 무엇을 해야 할지 알고 world는 adversarially seeded되지 않는다. 97% pass rate는 harness가 작동하고 모델이 기본 tool을 사용할 수 있음을 확인한다. main task에서의 12–17%는 다음의 marginal cost를 측정한다:

1. **Discovery overhead**: 무언가를 하기 전에 agent는 BM25 query를 통해 올바른 tool을 찾아야 한다. 검색 실패는 agent가 tool 이름을 hallucinate하거나 잘못된 것으로 진행한다는 것을 의미한다. [[automationbench-overview]]에 보고된 dominant failure mode는 "incorrect tool call에 대한 false confidence" — 실패의 72–91%가 여기에 해당한다.

2. **Cross-application coordination**: agent는 여러 app에 걸쳐 context를 유지하고, 한 tool call의 중간 결과를 다른 것의 argument로 전달하며, 여러 app state에 나타나는 entity를 혼동하지 않아야 한다.

3. **Policy discovery and correct application**: agent는 policy가 존재함을 알아차리고(prompt에 표시되지 않음), 올바른 artifact를 retrieve하고, policy를 파싱하며, 자연어 instruction이 다른 action을 제안하는 것처럼 보이는 때에도 올바르게 적용해야 한다.

4. **Adversarial seed data**: near-match entity trap(`acme-corp.com` vs `acmecorp.com`), noise contact의 compliance-hold description, recency conflict, 그리고 shotgun 행동을 처벌하는 negative assertion guard가 모두 "운이 좋은" 모델이 달성할 수 있는 floor를 높인다.

### What a low ceiling tells you about headroom

SOTA가 97%를 기록하는 benchmark는 측정 범위를 소진했다 — 모든 것이 상단에 밀집되기 때문에 미래의 개선을 구별할 수 없다. SOTA가 15%를 기록하는 benchmark는 많은 headroom이 있다: 15%와 30%의 차이는 benchmark가 깨끗하게 측정할 수 있는 의미 있는 capability 도약이다. main AutomationBench task의 낮은 ceiling은 문제가 아니라 기능이다.

sanity control(simple domain)은 중요한 보완이다. 그것 없이는 15%의 점수가 진정한 task 난이도를 반영하는지 harness 버그를 반영하는지 구별할 수 없다. 그것과 함께 — 그리고 동일한 모델이 simple에서 97%를 기록한다는 지식과 함께 — 낮은 main-benchmark 점수를 benchmark가 테스트하도록 설계된 어려운 요구사항에 자신 있게 귀속시킬 수 있다.

---

## 7. The Comparison Axis: AutomationBench vs. τ-bench

이 landscape에서 가장 중요한 두 benchmark 간의 구조적 대비는 내부 챕터에 들어가기 전에 여기서 명시적으로 다룰 가치가 있다. [[benchmark-comparison]]은 이것을 자세히 발전시킨다; 요약 버전:

| Axis | AutomationBench | τ-bench (τ²/τ³) |
|------|-----------------|-----------------|
| Interaction | 사용자 없음 — 단일 NL trigger; agent가 완료까지 실행 | Multi-turn LLM user simulator; agent가 hidden goal을 elicit해야 함 |
| Apps per task | 다수(cross-application이 핵심) | task당 하나의 app/domain |
| Tools given? | 아니오 — ~400개 tool에 대한 generic Search+Execute | 예 — per-domain 고정 tool set |
| Policy location | seeded world artifact에 묻혀 있음 | agent가 읽을 수 있는 posted policy wiki |
| Grading | End-state assertion rubric(must-pass + must-not-occur) | Final DB-state == goal-state, 정확히 |
| Partial signal | `partial_credit`(RL 사용); 공식 점수는 이진 | 이진 r∈{0,1}, partial 없음 |
| Headline metric | pass-rate + cost/task | pass^k(k번 시도에 걸친 reliability) |
| Run-to-run noise | <1%(결정론적 seeded world) | 높음(LLM user stochasticity; 10+ 반복 권장) |
| Hardest sub-skill | Tool discovery + cross-app coordination + policy-under-noise | Information elicitation + user coordination + consistency |

두 metric 철학은 두 가지 다른 deployment 질문을 표현한다. **pass-rate + cost**는 "지금 당장 어떤 가격에 capability를 가지는가?"에 답한다 — 각 workflow trigger가 독립적이고 throughput economics가 중요할 때 적절하다. **pass^k**는 "고객이 같은 것을 요청할 때마다 성공할 것인가?"에 답한다 — 동일한 task가 반복되고 30% failure rate가 peak performance에 상관없이 자격을 박탈할 때 적절하다.

[[benchmark-comparison]]이 인용할 가치 있는 포인트를 만든다:

> *They measure different agents. AutomationBench measures a back-office automator (one trigger, no human, many apps, find-your-own-tools, follow buried policy); τ-bench measures a customer-service conversationalist (multi-turn user you must interrogate, one app, given tools, follow a posted policy).*

실질적인 결과: workflow automation agent를 구축하는 경우, AutomationBench 점수는 τ-bench 점수가 그렇지 않은 방식으로 프로덕션 동작을 예측하며, customer-service agent의 경우 그 반대다. 이 챕터의 목표는 내부 챕터들이 AutomationBench가 네 가지 요구사항 각각을 code level에서 어떻게 enforce하는지 설명할 수 있도록 landscape를 확립하는 것이다.

---

## 8. Running AutomationBench: The Minimal Footprint

README가 entry point를 그대로 기술한다:

```bash
# Clone the repo
git clone https://github.com/zapier/AutomationBench.git
cd AutomationBench

# Install dependencies
uv sync

# Set your API key (or create a .env file)
export OPENAI_API_KEY=sk-...

# Run evaluation
uv run auto-bench --model gpt-5-mini

# Run specific domains
uv run auto-bench --model gpt-5-mini --domains sales

# Anthropic models — auto-detected via `claude-*` prefix
export ANTHROPIC_API_KEY=sk-ant-...
uv run auto-bench --model claude-haiku-4-5-20251001
```

benchmark 해석에 중요한 핵심 CLI parameter들:

- `--toolset`: `api` | `zapier` | `limited_zapier`. 이것을 변경하면 측정하는 것이 달라진다(§3.2 참조). 기본 `api` 모드는 REST-shaped tool을 사용하고; `zapier`는 discovery-required headline 모드다.
- `--max-steps 50`: README와 CLI 기본값은 50이라고 하지만; [[automationbench-harness]]는 code 기본값을 `max_turns=25`로 보고한다. 이 불일치(docs vs code)는 ch-02에서 해결된다.
- `--domains`: sales만 또는 simple만 등을 실행한다. `--domains simple`을 먼저 실행하는 것은 빠른 harness-validity 확인이다.
- cost metric은 이진 pass/fail과 함께 task당 기본으로 emit된다 — opt-in이 아니다.

benchmark는 또한 Prime Intellect Environments Hub에서 hosted environment로 실행되는데, 이는 RL training environment로 직접 사용될 수 있음을 의미한다:

```bash
prime env install zapier/AutomationBench
prime eval run zapier/AutomationBench
```

harness가 `task_completed_correctly`와 함께 compute하는 `partial_credit` 점수는 RL을 위한 dense reward signal이다 — ch-10이 중심으로 만드는 세부사항이다.

---

## Where This Course Goes

이 챕터는 landscape를 확립했다: benchmark bar가 왜 이동했는지, AutomationBench가 이 family에서 어디에 위치하는지, 그리고 세 부분의 gap thesis(cross-app coordination + autonomous discovery + policy adherence)가 왜 진정으로 어려운 eval을 만드는지. 여기서부터 이후의 모든 것은 기계적이다.

Ch-02는 execution engine을 연다: `AutomationBenchEnv`, `WorldState` Pydantic root model, episode lifecycle, 그리고 세 가지 toolset 모드 — tool count와 step limit에 대한 doc-vs-code 불일치 포함. Ch-03과 Ch-04는 two-phase execution model에 깊이 들어간다: BM25 tool discovery 후 in-process simulated execution. Ch-05는 전체 task dict anatomy와 여섯 business domain을 살펴본다. Ch-06은 assertion grading engine을 다룬다 — must-pass와 must-not-occur assertion이 어떻게 평가되는지, 그리고 아무것도 하지 않음으로써 reward-hacking을 방지하는 free-assertion exclusion logic. Ch-07은 hardening을 다룬다: seeded noise, near-match trap, compliance-hold pattern, anti-shotgun negative assertion. Ch-08은 internals 단계를 metric, cost accounting, 그리고 재현성으로 마무리한다.

Ch-09는 internals 챕터에 걸쳐 seed된 [[benchmark-comparison]] thread를 통합한다 — τ-bench family와의 전체 구조적 head-to-head와 주어진 agent shape에 맞는 올바른 benchmark를 선택하기 위한 decision framework. Ch-10은 lab이다: 실제 evaluation 실행, 새로운 task와 assertion으로 benchmark 확장, 그리고 코스에 걸쳐 추출된 원칙을 사용하여 자신만의 end-state-graded agent benchmark 설계.
