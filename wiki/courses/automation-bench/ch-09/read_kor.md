<!-- chapter: ch-09
     track: comparison
     kind: content
     title: AutomationBench vs the τ-bench Family: A Structural Comparison
     deps: [ch-08, ch-05, ch-06]
     sources: [[benchmark-comparison]], [[taubench]], [[automationbench-overview]]
     figures: figures/ab-vs-tau.html
-->

# 09장 — AutomationBench vs the τ-bench Family: A Structural Comparison

> **핵심 통찰.** AutomationBench와 τ-bench는 동일한 agent를 측정하지 않는다. AutomationBench는 *back-office automator*를 측정한다: trigger 하나, user 없음, 여러 app, tool discovery 필수, policy는 artifact 안에 묻혀 있다. τ-bench는 *customer-service conversationalist*를 측정한다: multi-turn user simulator, app 하나, 고정된 tools, 공개된 policy wiki. 잘못된 benchmark에서 높은 점수를 받아도 실제 배포 환경에서의 실패 패턴을 거의 예측하지 못한다.

> **가이드라인.** evaluation을 한 번이라도 실행하기 전에, 자신의 agent의 배포 형태에 맞는 benchmark 구조를 먼저 맞춰라. Agent가 user와 전혀 대화하지 않는다면, τ-bench의 reliability 수치는 별 의미가 없다. Agent가 순수하게 conversational하다면, AutomationBench의 cross-app discovery 점수는 별 의미가 없다. 비교가 실행 가능해지는 것은 상대방 benchmark에서 빌려야 할 구조적 특성이 무엇인지 정확히 파악했을 때뿐이다.

---

## 1. The Core Axis: Interaction Model

두 benchmark 사이의 가장 근본적인 차이는 tooling도, grading도, metric도 아니다. 그것은 루프 안에 human user가 존재하는가의 여부다.

**AutomationBench: trigger-and-run.** task는 단 하나의 natural-language event — 이메일 알림, Slack 메시지, 요청 — 로 시작되며, agent는 자율적으로 완료까지 실행한다. 물어볼 user도 없고, 명확화를 위한 turn도 허용되지 않는다. system prompt가 이를 직접 강제한다:

```python
# automationbench/domains/sales/tasks.py  L31-37
SYSTEM_PROMPT = (
    "You are a workflow automation agent. Execute the requested tasks using the available tools. "
    "Do not ask clarifying questions - use the information provided and make reasonable assumptions when needed. "
    ...
)
```

policy의 모호함, 충돌하는 데이터, 누락된 정보는 초기 world state에 인코딩되어 있으며, user에게 묻는 것이 아니라 *artifact를 읽는 것*으로 해결해야 한다. win notification의 routing policy는 agent가 직접 찾아야 하는 Gmail inbox 메시지 안에 있다. FX 환율은 두 개의 충돌하는 행이 있는 sheet 안에 있는데, recency가 암묵적인 규칙이다. 아무도 알려주지 않는다. 직접 찾거나, 아니면 assertion에서 실패한다.

**τ-bench: tool-agent-USER triad.** 별도의 LLM이 고객 역할을 한다. 그것은 목표와 persona를 부여받고, 아직 질문받지 않은 정보는 보류하며, 만족했을 때 `###STOP###`을 출력한다. Agent는 여러 turn에 걸쳐 *전체 task를 이끌어내야* 하며, ground truth 목표를 처음부터 보는 법이 없다. [[taubench]]에서 인용하면:

> *"A task = 4 parts: (1) a user goal held by the simulator, not given to the agent; (2) a mutable JSON database state; (3) a policy document (wiki) of domain rules; (4) a fixed set of Python-backed tools."*

user simulator는 정보 elicitation을 일급 테스트로 만드는 메커니즘이다. 동시에 확률적 분산을 주입하는 메커니즘이기도 하다: 시뮬레이터된 user가 정보를 자발적으로 제공하는 방식이나 종료 시점이 달라지므로, 동일한 task도 실행할 때마다 달라진다. 이것은 결함이 아니라, 실제 고객의 가변성을 모델링한 것이다. 그러나 이로 인해 평가 철학이 달라지는데, 그것이 바로 pass^k가 등장하는 이유다.

---

## 2. Tooling: Discovery vs. Provision

두 번째 구조적 분기점은 tools가 agent에게 전달되는 방식이다.

**AutomationBench: ~400개 tools에 대한 generic Search+Execute.** 기본 `zapier` 모드(ch-02 참조)에서 agent는 정확히 두 개의 meta-tool만 받는다: `search_tools(query, top_k=5)`와 `execute_tool(tool_name, arguments)`. BM25 index는 시뮬레이션된 모든 app의 모든 action에 걸쳐 있으며, 약 400개의 endpoint에 달한다. Agent는 쿼리를 구성하고, 후보를 검색하고, schema를 읽고, 호출을 구성해야 한다. Tool discovery는 명시적으로 테스트되는 역량이다.

`limited_zapier` ablation 모드는 agent에게 task별로 이름이 지정된 subset을 제공한다(예: `sales.multi_hop_lookup`의 `info.zapier_tools`에 있는 여덟 개 tools). `zapier` vs. `limited_zapier` 점수를 비교하면 모델의 실패 중 얼마만큼이 execution이 아닌 discovery에 기인하는지를 정확히 분리할 수 있다. 두 수치 간의 격차가 discovery tax다.

**τ-bench: 처음부터 제공되는 domain별 고정 tool set.** Agent는 task 시작 시 전체 domain tool set을 받는다 — discovery 단계가 없다. Tools는 두 클래스로 나뉜다: read tools(자유롭고 non-mutating)와 write tools(DB를 변경하며, task 내에서 되돌릴 수 없다). Policy는 destructive write 전에 주요 파라미터를 user에게 확인할 것을 요구한다. Tool 선택과 argument 구성이 테스트된다; *어떤 tools가 존재하는지 찾는 것*은 테스트되지 않는다.

실용적 함의는, τ-bench의 실패 모드는 argument 오류(불충분한 elicitation 이후의 잘못된 필드 값)와 policy 위반(confirm 전 write)에 집중된다는 것이다. AutomationBench의 실패는 *tool selection*(잘못된 tool에 대한 오신뢰)에 집중되며, [[automationbench-overview]]에 따르면 실패의 72–91%가 incorrect tool call에 기인한다.

---

## 3. Policy and State: Buried vs. Posted

두 benchmark 모두 policy 준수를 요구한다. 차이는 policy가 agent에게 어떻게 드러나는가에 있다.

**AutomationBench: policy-in-artifacts.** routing rule, compliance hold, FX convention, account tier — 이 중 어느 것도 task prompt에 나타나지 않는다. 이것들은 seeded world state 안에 있다: inbox 이메일, 스프레드시트 메모, Notion 페이지. Agent는 *policy를 찾아야 한다는 결정을 스스로 내리고*, *어디를 봐야 하는지 알고*, *비정형 텍스트에서 그것을 추출해야* 한다. [[automationbench-tasks-grading]]의 가이드라인을 인용하면: *"bury the policy in the world so the agent has to find it."*

ch-05의 `multi_hop_lookup` walkthrough가 이것의 정석적인 예시다: routing policy는 `msg_routing_policy`에 있으며, `initial_state.gmail.messages`에 seeded된 Gmail 메시지다. Agent는 policy가 Gmail에 있는지 Salesforce에 있는지 스프레드시트에 있는지 아무런 단서가 없다. 행동하기 전에 이것을 검색하지 못하면 잘못된 팀에 이메일이 전송되고, 그것은 `gmail_message_not_sent_to` negative assertion을 유발한다.

**τ-bench: posted policy wiki.** Agent는 task 시작 시 구조화된 policy 문서를 받는다. wiki가 존재한다는 것을 알고, 어디에 있는지 알고, 쿼리할 수 있다. 도전은 policy *발견*이 아니라 policy *적용*이다: 올바른 규칙 분기를 읽고, confirm 전에 쓰지 않으며, edge case를 처리하는 것. user가 inconsistent한 정보를 제공할 때 더 어려워진다 — user가 반품 기간이 지났다고 주장하지만 database는 아니라고 할 때, 어느 분기가 적용되는가?

**State scope: multi-app vs. single-app.** AutomationBench task들은 여러 독립적인 app state를 동시에 넘나든다 — ch-05의 Salesforce+Google Drive+Google Sheets+Gmail 체인이 예외가 아닌 표준이다. τ-bench task들은 task당 하나의 application 안에 존재한다(retail CRM, airline 예약 시스템, banking). τ³-bench는 제한적인 RAG/doc discovery를 가진 `banking` domain을 추가하지만, 핵심 루프는 single-app로 유지된다. 이것이 [[automationbench-overview]]의 landscape table에서 가장 날카로운 격차다:

| | Cross-app | API discovery | End-state grading | Business rules |
|---|---|---|---|---|
| τ³-bench | ✗ (single app) | partial (banking) | ✓ | ✓ |
| **AutomationBench** | **✓** | **✓ (BM25)** | **✓** | **✓** |

---

## 4. Grading: Assertion Rubric vs. DB-State Equality

두 benchmark 모두 agent의 텍스트가 아닌 outcome을 기준으로 채점한다. 둘 다 LLM judge 없이 순수 Python을 사용한다. 메커니즘은 다르다.

**AutomationBench: must-pass AND must-not-occur assertion rubric.** Assertion은 `AssertionRegistry`를 통해 dispatch되는 typed dict다(ch-06). task는 다음을 가질 수 있다:

```python
# must-pass
{"type": "salesforce_field_equals", "collection": "opportunities",
 "record_id": "006xx000004MER1", "field": "stage_name", "value": "Closed Won"}

# must-not-occur (anti-shotgun)
{"type": "gmail_message_not_sent_to", "to": "vp-sales@example.com",
 "subject": "Deal Closed Notification"}
```

negative assertion은 anti-reward-hacking 메커니즘이다. "혹시 모르니" 다섯 개 mailbox 전부에 deal notification을 보내는 agent는 must-not-occur check 세 개에서 실패한다. 여러 opportunity를 won으로 표시하는 agent는 대상이 아닌 record의 `salesforce_field_equals` guard에서 실패한다. free-assertion exclusion은 아무것도 하지 않아서 reward를 얻는 것을 방지한다: assertion이 초기 world state에서 이미 충족되어 있다면 분모에서 제외되지만, pre-passing guard를 *깨뜨리는* 것은 여전히 실패로 집계된다.

공식 점수는 binary다(`task_completed_correctly = 1 iff all assertions pass`). `partial_credit = passed/total`도 존재하지만, RL reward signal 및 diagnostic용으로만 쓰이며 headline leaderboard 수치에는 나타나지 않는다.

**τ-bench: final DB-state equality check.** 채점은 terminal database state를 annotated goal state와 비교하는 것이다: exact match, binary reward `r∈{0,1}`, partial credit 없음. task가 단일 application에 scoped되어 있기 때문에 메커니즘이 더 단순하다: "conversation이 끝난 뒤 DB가 정확히 이런 모습인가?" Negative assertion은 없지만, write tools의 비가역성이 동일한 anti-gaming 기능을 수행한다 — task 내에서 잘못된 write는 되돌릴 수 없으므로, shotgun 행동은 explicit guard가 아니라 state 오염으로 패널티를 받는다.

---

## 5. The Two Metric Philosophies

이것이 가장 깊은 대비이며, 정확하게 이해할 가치가 있다.

### 5a. pass-rate + cost (AutomationBench)

AutomationBench는 `pass-rate`(올바르게 완료된 task의 비율)를 `cost/task`(완료되었거나 시도된 task당 LLM inference 비용, 달러)와 함께 보고한다. 이 쌍이 답하는 질문은: *"지금 당장 이 agent가 무엇을 할 수 있으며, 그 가격은 얼마인가?"*

각 task는 고유하고 독립적인 episode다. Benchmark는 여섯 개의 서로 다른 business domain에 걸쳐 606개의 public task를 가진다 — 반복되는 task는 없다. 평가 질문은 예산 내에서의 capability breadth다. AutomationBench가 Zapier에 의해 만들어진 이유는 production 배포 질문에 답하기 위해서였다: 단순히 "작동하는가?"가 아니라 "규모에서 운영할 수 있는가?" SOTA pass-rate는 2026-06 기준 약 12–17%; frontier 모델들은 논문 제출 당시 10% 미만이었다([[automationbench-overview]]).

Pass-rate가 여기서 적합한 이유는, AutomationBench의 in-process Pydantic world가 본질적으로 deterministic하기 때문이다: run-to-run 분산이 1% 미만이다. 안정적인 추정치를 얻기 위해 trial을 반복할 필요가 없다. 각 task를 한 번 실행하면 신뢰할 수 있는 signal을 얻는다.

### 5b. pass^k (τ-bench)

τ-bench는 `pass^k` — k번의 i.i.d. trial에 걸친 reliability, task 평균 — 를 보고한다. [[taubench]]의 공식과 unbiased estimator:

```
pass^k = (1/|T|) Σ_i p̂_i^k

unbiased estimator:  ρ(n, c, k) = 1 − C(n−c, k) / C(n, k)
```

여기서 task i에 대해 n번의 trial이 실행되고 c번 성공했다.

**왜 pass@k가 아닌 pass^k인가?** pass@k(k번 시도 중 *적어도 한 번* 성공)는 breadth에 보상한다. 어떤 agent가 80%의 확률로 task에 실패하지만 20%는 성공한다면, 큰 k에서 높은 pass@k를 가진다. pass^k는 *모든* trial이 성공해야 한다 — consistency를 측정한다. 동일한 transaction이 매일 수천 번 반복되는 customer-service agent에서, 어떤 단일 task 유형에서의 30% 실패율은, agent가 다른 task에서 아무리 뛰어나더라도 자격 박탈이다. pass^1 = pass@1 = mean success rate.

**수치 예시.** task의 trial당 성공 확률이 p = 0.9라고 가정하자 — agent가 90% 확률로 성공한다. pass^8에 대한 그 task의 기여는:

```
p^8 = 0.9^8 = 0.43
```

10번 중 9번 성공함에도 불구하고, 그 task는 pass^8 평균에 0.43만 기여한다. p = 0.7이라면: `0.7^8 ≈ 0.06`. Reliability는 k가 증가함에 따라 빠르게 붕괴한다. 원래 2024 τ-bench 결과가 이를 구체적으로 보여준다: GPT-4o는 retail에서 pass^1 약 61%를 기록했지만 pass^8은 25% 미만이었다 — 단일 실행에서는 사용 가능해 보이지만, 실제 배포가 요구하는 기준에서는 무너진다([[taubench]]).

**AutomationBench가 pass^k를 채택할 수 있는가?** 그렇다, 그것도 저렴하게: near-zero 분산이므로 trial을 반복하는 비용은 inference compute뿐이고, 새로운 환경 노이즈가 추가되지 않는다. Benchmark는 *pass^k를 secondary metric으로 보고할 수 있다*. 오늘날 그렇게 하지 않는 이유는, 1차 질문이 reliability-across-deployments가 아니라 capability-at-a-price이기 때문이다. τ-bench는 user simulator가 내재적 stochasticity를 도입하기 때문에 *pass^k가 반드시 필요하다* — trial을 반복하지 않으면 agent 분산과 simulator 노이즈를 구별할 수 없다.

---

## 6. Side-by-Side Comparison Table

| Axis | AutomationBench | τ-bench (τ²/τ³) |
|------|-----------------|-----------------|
| **Interaction model** | User 없음. Single NL trigger; agent가 완료까지 실행 | Multi-turn LLM user simulator; agent가 숨겨진 목표를 *이끌어내야* 함 |
| **Apps per task** | 다수 (cross-app이 핵심; 정석 task에서 5개 이상 app) | task당 app/domain 하나 (τ³ banking이 제한적 doc discovery 추가) |
| **Tools given to agent?** | 아니오 — ~400개에 대한 generic Search+Execute; discovery가 테스트됨 | 예 — domain별 고정 tool set, 처음부터 제공 |
| **Where policy lives** | Seeded world에 묻혀 있음 (inbox 메시지, sheet 행, Notion 페이지) | Posted policy wiki; agent가 존재와 위치를 알고 있음 |
| **Grading mechanism** | Assertion rubric: must-pass + must-not-occur, 순수 Python | Final DB-state == annotated goal-state, exact match, 순수 Python |
| **Partial signal** | RL reward용 `partial_credit`; 공식 점수는 binary | Binary `r∈{0,1}`, partial credit 전혀 없음 |
| **Anti-reward-hacking** | Negative assertions + count-locks + free-assertion exclusion | 비가역적 writes; confirm-before-write policy |
| **Headline metric** | pass-rate + cost/task | pass^k (k번의 trial에 걸친 reliability) |
| **Run-to-run variance** | < 1% (in-process Pydantic, seeded determinism) | 높음 (LLM user stochasticity); 10회 이상 반복 필요 |
| **Hardest sub-skill tested** | Tool discovery + cross-app coordination + policy-under-noise | Information elicitation + user coordination + consistency |
| **Benchmark size (public)** | 606 tasks (6 domains × ~100 tasks) | 165 tasks (retail 115 + airline 50); τ³가 banking 추가 |
| **SOTA ceiling (2026-06)** | ~12–17% pass-rate | ~0.86 pass^1 retail, ~0.70 airline (Claude Sonnet 4.5) |

---

## 7. What Each Tests That the Other Cannot

비교 표는 두 benchmark를 겹치는 feature set처럼 보이게 만들 위험이 있다. 그렇지 않다. 각각은 상대방의 아키텍처가 구조적으로 도달할 수 없는 역량을 테스트한다.

**AutomationBench만 테스트할 수 있는 것:**

- *Autonomous tool discovery.* τ-bench는 tools를 agent에게 건네준다. AutomationBench는 agent가 `google_drive_find_multiple_files`를 사용하기 전에 그것이 존재한다는 사실을 스스로 파악하도록 요구한다. 모델이 tool *인수*가 아니라 tool *가용성*에 대해 추론할 수 있는지 여부는, tools를 미리 제공하는 어떤 benchmark에도 보이지 않는다.
- *Cross-application coordination.* `multi_hop_lookup`(ch-05)의 multi-hop Salesforce → Google Drive → Google Sheets → Gmail 체인은 공유 세션도 없고 이를 연결하는 API도 없이, 독립적인 app 경계를 넘어 일관된 state를 유지해야 한다. τ-bench의 single-app scope는 이것을 모델링할 수 없다.
- *Policy discovery under noise.* 근사 일치 entity 이름, 충돌하는 sheet 행, prompt의 scope-creep 함정이 있는 가운데 decoy 메시지들 사이의 inbox에서 routing policy를 찾는 것은, single-app benchmark가 탐색할 수 없는 역량이다. AutomationBench의 hardening 메커니즘(ch-07)은 이것을 명시적으로 인코딩한다: ID-namespaced decoy pool, compliance-hold 함정, recency 충돌.
- *Cost as a first-class axis.* AutomationBench는 pass-rate와 함께 task당 달러를 보고한다. τ-bench에는 이에 해당하는 것이 없다.

**τ-bench(τ²/τ³)만 테스트할 수 있는 것:**

- *Conversational information elicitation.* 여러 turn에 걸쳐 올바른 질문을 할 수 없는 agent는, tool call이 아무리 훌륭해도 τ-bench에서 실패할 것이다. User simulator는 명시적으로 요청되지 않은 정보를 보류한다 — under-elicitation은 잘못된 argument의 write를 낳는다.
- *User coordination (τ²-bench).* dual-control Dec-POMDP 공식화에서, user도 tools를 갖는다 — 라우터를 troubleshooting하는 고객은 agent가 지시할 때 직접 재부팅해야 한다. Agent는 정보를 이끌어내는 것뿐 아니라 *user의 행동을 안내해야* 한다. GPT-4는 약 56–74%(single-control)에서 약 34%(dual-control)로 떨어진다. 그 격차는 agent가 일방적으로 행동하는 어떤 benchmark에도 보이지 않는다.
- *Consistency across a conversation.* τ-bench agent가 대화 중간에 자기모순을 한다면 — turn 3에서 반품을 확인했다가 turn 7에서 자격이 없다고 부정하는 경우 — 마지막 write가 올바르더라도 final DB-state check에서 실패한다. Conversational coherence는 trigger-and-run benchmark가 측정할 수 없는 속성이다.
- *Reliability as a deployment property.* pass^k는 동일한 task가 규모에서 반복될 때 가장 의미 있다. Customer service transaction은 본질적으로 반복적이고, enterprise automation workflow는 종종 일회성이다. τ-bench의 metric은 그것이 측정하는 agent의 배포 현실에 더 잘 맞는다.

---

## 8. The τ² and τ³ Extensions

τ-bench는 진화했다. Extensions는 AutomationBench와의 격차 일부를(전부는 아니지만) 좁히기 때문에 관련이 있다.

**τ²-bench (arXiv 2506.07982):** dual-control Dec-POMDP를 도입한다 — user도 정보뿐 아니라 agency를 가진다. 통신 troubleshooting domain은 agent가 고객에게 hardware를 재부팅하고 config 값을 읽도록 안내하도록 요구한다. 이것은 benchmark를 multi-party orchestration 방향으로 확장하지만, 여전히 단일 application context 내에 머문다. Compositional task generator가 제어된 난이도 스케일링을 가능하게 한다.

**τ³-bench (taubench.com):** 원래 τ-bench에서 발견된 약 75개의 결함 있는 task를 수정한다(수정 후 airline pass^1이 14–20포인트 상승했다 — 냉정한 benchmark 위생 알림). RAG와 document-tool discovery를 갖춘 `banking` domain을 추가하여 policy-discovery 격차를 부분적으로 좁힌다. Voice/full-duplex metric을 추가한다. 2026년 기준, τ³가 새로운 결과 보고를 위한 권장 baseline이다.

τ³가 *좁히지 못하는 것*: cross-application coordination은 여전히 부재한다. Banking domain의 doc discovery는 로컬 문서 코퍼스에 대한 쿼리이지, 다섯 개의 독립적인 SaaS backend에 걸친 ~400개 tool 카탈로그에 대한 BM25가 아니다. 근본적인 single-app 제약은 architectural한 것이지, 간과가 아니다.

---

## 9. Shared Blind Spot: The Sim2Real Gap

두 benchmark 모두 세계를 시뮬레이션한다. 어느 쪽도 시뮬레이션과 production 사이의 격차에서 자유롭지 않다.

AutomationBench의 in-process Pydantic world는 deterministic하고 빠르지만, 모든 record를 직접 seed한다. 실제 enterprise workflow에는 어떤 benchmark task도 인코딩하지 않는 문서화되지 않은 edge case, legacy 데이터 불일치, API rate limit가 있다. AutomationBench에서 17%를 기록한 모델은 production에서 더 낮을 수도(실제 task가 더 노이지하다), 더 높을 수도 있다(benchmark의 engineered 함정이 실제 workflow보다 더 촘촘하다).

τ-bench의 LLM user simulator는 문서화된 방식으로 실제 user와 다르다. arXiv 2603.11245([[taubench]]에서 인용)에 따르면: 시뮬레이션된 user는 turn-taking 패턴, error recovery, persona consistency에서 실제 user와 다르다. τ-bench simulator를 처리하도록 훈련된 agent는 simulator의 특정 실패 모드에 과도하게 특화될 수 있다.

sim2real gap은 두 benchmark를 무시할 이유가 아니다 — 그것은 높은 점수를 *필요조건이지 충분조건이 아닌* 것으로 취급해야 하는 이유다. AutomationBench에서 온라인 측정 없이 높은 점수를 받는 것은 실제 trial을 실행하기 위한 청신호이지, 그것의 대체물이 아니다. τ-bench pass^k도 마찬가지다.

---

## 10. Choosing a Benchmark: A Worked Example

이 비교는 구체적인 agent에 적용될 때 가장 실행 가능해진다. **Lina TMR sales agent**를 고려하자 — 영업 담당자를 대신해 lead를 qualify하고, deal을 관리하며, 내부 팀과 조율하는 conversational agent다.

**Lina의 배포 형태는 무엇인가?**

- 그녀는 multi-turn dialogue로 고객과 대화한다. → 대화 루프에 대한 *τ-shape*.
- 그녀는 통화 후 자율적으로 Salesforce를 업데이트하고, Slack 알림을 보내고, Google Sheets 항목을 기록한다. → back-office 실행에 대한 *AB-shape*.
- 그녀의 실패 모드는: 잘못된 tier routing, 중복 알림, hallucinated deal 금액, 조기 escalation. → *두* benchmark 모두 관련 실패 signal을 포착한다.

**어떤 benchmark를 먼저 실행할 것인가?**

τ-bench가 Lina의 더 나은 *주요* benchmark다. 그녀의 핵심 루프가 conversational하기 때문이다: 고객의 실제 상황을 이끌어내고, policy playbook을 따르고, confirm 전에 destructive write를 피하는 것. pass^k는 "Lina가 다음 8번의 유사한 통화에서 성공할 것인가?"로 직접 해석된다.

**AutomationBench에서 빌릴 것:**

eval *engineering*이 AutomationBench가 가장 많이 기여하는 부분이다. 구체적으로:

1. **typed world에 대한 End-state assertions.** 각 시뮬레이션된 Lina 대화 후, `WorldState`에 해당하는 것에 대해 assertion을 실행하라: Salesforce에 올바른 stage가 기록되었는가, 올바른 팀이 Slack 알림을 받았는가, 금액이 일치하는가? 이것은 LLM-judge rubric보다 깔끔하며, AutomationBench가 텍스트 출력으로 채점하는 benchmark에서 발견한 scoring gap을 방지한다.

2. **Must-not-occur guards.** Lina가 SMB deal에 대해 VP-Sales 팀에 알리지 *않았는가*? 중복 opportunity를 만들지 *않았는가*? Negative assertion은 작성하기 저렴하며, positive assertion이 놓치는 reward-hacking 행동을 잡는다.

3. **Deterministic seeded noise.** Lina의 테스트 대화에 충돌하는 pricing 행, 근사 일치 회사 이름, CRM의 compliance-hold 연락처를 seed하라. AutomationBench의 hardening 철학(ch-07)이 직접 적용된다: 난이도는 prompt가 아니라 데이터에 engineered되어야 한다.

4. **Pass^k를 reliability bar로.** τ-bench에서 빌려라. deal-close 대화의 30%에서 실패하는 Lina는 pass@1 평균이 어떻든 배포 불가능하다.

결과적인 eval stack: 대화를 위한 τ-style LLM user simulator + back-office write를 위한 AutomationBench-style end-state assertions + reliability verdict를 위한 pass^k. 이것이 ch-10의 deliverable이다 — conversational-automator hybrid가 실제로 어디서 실패하는지를 triangulate하는 custom eval harness.

---

## 11. Why This Chapter Exists

이 장에서 전개된 구조적 비교는 ch-02부터 ch-08까지 쌓인 모든 것의 payoff다. Toolset 모드(ch-02)는 discovery/execution 분리를 조명한다. Task anatomy(ch-05)는 policy-in-artifacts 설계를 드러낸다. Assertion rubric(ch-06)은 must-not-occur guard가 DB-state equality가 잡을 수 없는 것을 왜 잡는지 보여준다. Metrics 장(ch-08)은 pass-rate-vs-pass^k 대비를 준비한다.

τ-bench는 AutomationBench의 경쟁자가 아니다. 그것은 다른 클래스의 agent를 위한 benchmark이며 — autonomous가 아닌 conversational이 핵심인 agent. 두 benchmark는 보완적이다: user-facing chat 레이어와 back-office execution 레이어를 모두 결합하는 production agent 배포는, 올바르게 평가하기 위해 두 가지 benchmark shape를 모두 필요로 한다.

이 분야의 현재 격차는 둘 다를 가진 agent를 커버하는 단일 benchmark가 없다는 것이다. 하나를 만들거나 — 양쪽의 기존 인프라를 구성하는 것이 이 장이 가리키는 현재 진행 중인 연구 문제다. 구체적인 시도는 ch-10을 보라.

구조적 차이의 부차적 함의: 두 benchmark는 단순한 평가 도구가 아니라 RL을 위한 보완적 *training signal*이기도 하다. AutomationBench의 `partial_credit` 점수(ch-06)는 전체 assertion rubric에 걸친 dense reward signal을 제공한다 — tool-selection과 multi-hop coordination을 형성하는 데 유용하다. τ-bench의 stochastic MDP에서의 binary `r∈{0,1}` reward는 더 노이지하지만, elicitation과 conversational coherence를 위한 더 풍부한 curriculum이다. 하나의 signal에만 독점적으로 훈련된 agent는 그것의 interaction shape에 overfit될 것이다. 생산적인 연구 방향은 execution backbone을 위해 AutomationBench partial credit을 사용하고, conversational 레이어의 reliability gate로 τ-bench pass^k를 사용하는 joint training curriculum이다.

참고: [[benchmark-comparison]], [[taubench]], [[automationbench-overview]], ch-05 (task anatomy), ch-06 (grading), ch-08 (metrics), [interactive comparison](figures/ab-vs-tau.html).
