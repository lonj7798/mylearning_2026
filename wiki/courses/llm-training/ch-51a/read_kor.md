<!-- chapter: ch-51a
     track: eval
     kind: content
     title: Evaluating Agent Generality and Reliability
     deps: [ch-51, ch-45b]
     sources: [[swe-bench-pro]], [[terminal-bench-2]], [[terminal-bench-trajectories]], [[terminal-bench-trajectories-stats]], [[tau-bench]], [[tau2-bench]], [[bfcl]], [[browsecomp-plus]], [[toolathlon]], [[are-gaia2]], [[holistic-agent-leaderboard]], [[agentic-benchmark-checklist]], [[swe-bench-illusion]], [[swe-rebench]], [[impossiblebench]], [[metr-time-horizon]], [[minimax-m2-aligning-to-what]], [[agent-world]], [[agentgym-rl]]
     figures: figures/pass-hat-k.html, figures/harness-coordinates.html
     revised: 2026-09 (generality revision)
-->

# 제51a장 — Agent generality와 reliability 평가

> **핵심 통찰.** Agent score는 model의 속성이 아니다. Model, scaffold, environment version, budget 네 argument의 function 출력이다. Model과 task set을 고정하고 scaffold만 바꾸면 Gemini 2.5 Pro는 Terminal-Bench 2.0에서 32.6%에서 15.7%로 움직인다([[terminal-bench-2]], Table 2). Model, scaffold, task set을 고정하고 task당 $2 cost cap만 추가하면 Claude Sonnet 4는 SWE-Bench Pro에서 42.7%에서 17.6%로 움직이며 GPT-5(high)와의 순서도 뒤집힌다([[swe-bench-pro]], Table 1, 5). Capability 옆에는 두 번째 측정이 필요하다. 한 run에서 Toolathlon task 38.6%를 해결하는 같은 agent가 세 run 중 적어도 한 번은 51.9%를 해결하고, 세 번 모두 해결하는 것은 20.4%뿐이다([[toolathlon]], Table 3). Capability와 reliability는 다른 양이며 따로 움직인다.
>
> **지침.** Agent result를 보고할 때 harness coordinate(scaffold와 version, tool schema, turn/time/token/cost budget, context strategy, environment image, reasoning effort, decoding setting, run 수)를 고정하고 공개한다. 이 coordinate는 수십 점의 가치가 있기 때문이다([[terminal-bench-2]], Table 2; [[swe-bench-pro]], Table 5; [[are-gaia2]], Fig. 13). 주장이 benchmark 하나가 아니라 generality에 관한 것이라면 training mixture에 없던 environment에서 평가하고 seen-versus-unseen pair를 보고한다. [[swe-rebench]]의 fresh-slice 비교는 contaminated benchmark에서 4.5점 차이로 순위를 매긴 checkpoint 두 개를, fresh slice에서는 반대 방향 0.6점 차이로 분리했다(Table 2). User request당 agent를 한 번 실행한다면 pass@1 옆에 pass^k를 보고한다. Deployment가 보는 양은 pass^k이기 때문이다([[tau-bench]], §3). Benchmark score를 gate로 쓸 때는 grader에 do-nothing agent와 exploit agent를 먼저 실행한다. 이 점검은 아무 것도 반환하지 않아도 τ-bench Airline task 38%를 통과할 수 있고 test를 overwrite하면 SWE-Lancer task 100%를 통과할 수 있음을 발견했다([[agentic-benchmark-checklist]], §5.2, App. E.2, E.4).

## 범용 모델에서 이 장이 중요한 이유

이 stage는 pipeline의 끝이다. pre-training → mid-training → SFT → preference optimization → RL → **evaluation**. [[ch-45b]]는 여러 turn에 걸쳐 행동하는 policy를 훈련했다. 이 장은 그 훈련이 아무도 훈련하지 않은 environment에서 작동하는 agent를 만들었는지, 실제 사용 가능한 빈도로 작동하는지를 판정한다.

Agent evaluation에는 [[ch-51]]의 single-turn evaluation과 다른 세 속성이 있다.

1. **Score는 model 하나에 붙지 않는다.** Static benchmark는 model에 completion 하나를 요청한다. Agentic benchmark는 scaffold라는 program을 실행한다. 이는 model을 여러 번 호출하고, output을 parse하며, tool을 실행하고, context를 truncate하며, 종료 시점을 정한다. 그 결정 하나하나는 model이 아니라 scorer가 제어하는 free parameter다.
2. **Variance가 더 크고 구조적이다.** Single-turn item은 다음 item과 독립적으로 실패한다. Agent trajectory는 한 step에서 실패해도 계속 실행하므로, 초반 mistake 하나가 이후 모든 observation을 바꾼다. 따라서 trajectory 간 run-to-run variation은 token-level sampling noise보다 클 뿐 아니라 task 안에서 상관돼 있다.
3. **Narrowing은 예상되는 failure다.** Agentic training은 verifier가 있는 적은 수의 environment에서 최적화한다. 측정할 위험은 policy가 그 environment에서는 좋아지고 다른 곳에서는 좋아지지 않거나 더 나빠지는 것이다. [[agentgym-rl]]은 trained agent가 “in-domain setting 안에서 잘 수행한다”고 명시하고 novel environment로의 transfer를 future work로 든다(§7). Unseen-environment split이 없으면 agentic training run은 capability와 environment fitting을 구별할 수 없다.

## §1 Agent score가 재는 것

**정의.** Agentic benchmark result는 다음 값이다.

```
score = f(model, scaffold, environment, budget, protocol)
```

`model`은 checkpoint와 decoding setting이다. `scaffold`는 model과 environment 사이를 매개하는 program(prompt, tool schema, parsing, retry logic, context strategy)이다. `environment`는 pinned container image, tool version, grader를 함께 가진 task set이다. `budget`은 turn, wall-clock 또는 simulated time, token, cost의 limit 집합이다. `protocol`은 run 수, aggregation, 보고 statistic이다.

**측정 가능한 문제로 쓴 문제.** Report가 `model`과 benchmark 이름만 말하면 독자는 수치를 재현할 수 없고 “같은” benchmark의 두 report도 비교할 수 없다. 이 비교 불가능성의 크기는 측정할 수 있다. Model 하나를 여러 scaffold 아래에서 평가하고 spread를 읽는다.

**근거(반복됨).** [[terminal-bench-2]]는 같은 task 89개에서 model 16개에 agent 6개를 실행한다. Agent-model 조합마다 최소 5회, 총 32,155 trial이다(§3). Model 하나 안에서는 다음과 같다.

| Model | 최선 scaffold | 열거된 최악 scaffold | Spread(point) |
|---|---|---|---|
| GPT-5.2 | Codex CLI 62.9% ± 3.0 | Terminus 2 54.0% ± 2.9 | 8.9 |
| Claude Opus 4.5 | Terminus 2 57.8% ± 2.5 | OpenHands 51.9% ± 2.9 | 5.9 |
| Gemini 2.5 Pro | Terminus 2 32.6% ± 3.0 | OpenHands 15.7% ± 2.6 | 16.9 |
| Grok 4 | Mini-SWE-Agent 29.0% ± 4.6 | OpenHands 19.6% ± 3.5 | 9.4 |

최선 scaffold는 model마다 같지 않다. 따라서 interaction은 실제며 constant offset이 아니다. 저자들은 전체 pattern을 “performance를 최적화할 때 model selection이 보통 agent scaffold보다 더 중요하다”고 요약한다(§4). 이것은 평균에 대한 진술이지 어떤 단일 비교에 대한 진술은 아니다.

[[holistic-agent-leaderboard]]는 model 9개와 benchmark 9개에 21,730 rollout, 약 $40,000을 투입해 같은 결론에 도달한다. Online Mind2Web에서 GPT-5 Medium을 쓴 SeeAct는 $171이고 Claude Sonnet 4를 쓴 Browser-Use는 정확도 2점 차이에 $1,577이다. Claude model은 Browser-Use에서, OpenAI model은 SeeAct에서 더 높은 점수를 낸다(§4.1 item 6--7). Task-specific scaffold는 CORE-Bench Hard 12 run 중 9개, SWE-bench Verified Mini 12 run 중 11개에서 generalist scaffold보다 낫고, generalist는 24개 비교 중 20개에서 비용이 낮다.

**조건과 한계.** 두 source 모두 proprietary/open model을 third-party API로 실행한다. [[holistic-agent-leaderboard]]는 provider가 같은 이름 아래 DeepSeek R1 endpoint를 R1-0528로 조용히 바꿨고, OpenRouter가 한 call에는 FP4, 다른 call에는 FP8을 제공했다고 기록한다(App. A3 item 2, 4). 그러므로 `model` identity 자체가 확인해야 하는 assumption이다.

> Figure: [harness-coordinates.html](figures/harness-coordinates.html) — coordinate별 tab(scaffold, turn/cost budget, simulated time, retriever quality) 하나씩. 각 group은 model을 고정하고 coordinate 하나만 바꾸며, group 안 spread와 bar별 정확한 source locus를 출력한다.

## §2 Generality suite: benchmark을 세지 말고 action space를 포괄하기

**정의.** Generality suite는 구성원이 topic만이 아니라 **action space**, 즉 agent가 수행할 수 있는 operation 집합과 agent가 받는 observation의 형태에서 서로 다르도록 선택한 evaluation environment 집합이다.

**이것이 다루는 문제.** 이미 coding benchmark 하나가 있는 suite에 두 번째 coding benchmark를 더해도 agent가 terminal을 조작할 수 있는지, tool도 가진 user와 대화할 수 있는지, corpus를 search할 수 있는지에 대한 정보는 거의 늘지 않는다. 이 문제의 측정 가능한 형태는 다음이다. 훈련 뒤 suite score 중 무엇이 움직였으며, 함께 움직였는가? 항상 함께 움직이는 score는 같은 한 가지를 재고 있다.

**Mechanism.** Environment는 네 축을 따라 고른다. (1) action space(repository patch, shell command 발행, typed function 호출, corpus search, application stack에서 행동), (2) 함께 행동하는 주체(없음, simulated user, tool도 보유한 simulated user, 다른 agent, asynchronous environment event), (3) grader(unit test, final-state comparison, execution script, LLM judge, string match), (4) horizon(task당 turn)이다.

| Environment | 크기 | Action space | 다른 actor | Grader | 보고된 ceiling |
|---|---|---|---|---|---|
| SWE-Bench Pro public([[swe-bench-pro]]) | task 731개, repository 11개 | multi-file repository patch | 없음 | fail2pass + pass2pass unit test | 43.6%(Claude Sonnet 4.5, Table 1) |
| SWE-Bench Pro commercial | task 276개, private repository 18개 | 같음 | 없음 | 같음 | 17.8%(Claude Opus 4.1, Table 2) |
| Terminal-Bench 2.0([[terminal-bench-2]]) | task 89개 | container 안 shell command | 없음 | 최종 container state의 test | 62.9% ± 3.0(GPT-5.2 + Codex CLI, Table 2) |
| τ-bench([[tau-bench]]) | retail 115 + airline 50 | domain API call + user message | simulated user(tool 없음) | database-state equality + substring check | 61.2 / 35.2 pass^1(gpt-4o, Table 2) |
| τ²-bench telecom([[tau2-bench]]) | 평가 task 114개 | agent tool + user guidance | **write tool 15개 / read tool 15개를 가진** simulated user | 최종 state의 assertion function | 49% pass^1(claude-3-7-sonnet, §1) |
| BFCL([[bfcl]]) | 6개 이상 category에 pair 5,551개 | typed function call, Python/Java/JS/REST/SQL | 없음 또는 stateful multi-turn | AST substring matching, state check | 전체 66.4(gpt-4o-2024-11-20, Table 1) |
| BrowseComp-Plus([[browsecomp-plus]]) | query 830개, document corpus 100,195개 | 고정 corpus의 search/fetch | 없음 | gpt-4.1 judge | 70.12%(gpt-5 + Qwen3-Embedding-8B, Table 1) |
| Toolathlon([[toolathlon]]) | task 108개, application 32개, tool 604개 | application 전반 MCP tool call | 없음 | task별 deterministic script | 38.6 ± 2.7 pass@1(Claude-4.5-Sonnet, Table 3) |
| Gaia2([[are-gaia2]]) | scenario 800 + 320개, app 12개, tool 101개 | asynchronous environment의 app tool | simulated user, app sub-agent, environment event | hard/soft/causal/timing check가 있는 write-action matching | 전체 42.1 pass@1(GPT-5 high, Table 2) |

**축이 중복되지 않는 근거(결과, 단일 연구).** [[are-gaia2]] Table 2에서 GPT-5(high)는 Search 79.6, Execution 69.2지만 Ambiguity 51.9, Adaptability 40.4, Time 0.0이다. Grok-4는 Search 57.5, Execution 8.8이다. 저자들은 Execution/Search에서 Ambiguity/Adaptability로의 하락을 existing benchmark가 “realistic environment의 robustness를 과대평가할 수 있다”는 근거로 읽는다(§4.2, Interpretation). [[bfcl]] Table 1에서 prompting mode의 gpt-4o-2024-11-20은 single-turn AST multiple/parallel call에서 95.5와 94.0이지만 multi-turn base에서는 59.0, memory에서는 6.0이다. 71개 행 전체의 최고 memory score도 12.0이다(§5.6).

**범용 모델에 주는 함의.** 모든 model에서 구성원이 같은 순서로 점수화되는 suite는 정보 축 하나만 담고 있다. τ²-bench dual-control result는 다른 suite에 없는 축의 가장 날카로운 예다. gpt-4.1을 no-user mode(agent가 모든 tool을 보유)에서 dual control로 옮기면 telecom pass^1이 0.67에서 0.34로 하락한다([[tau2-bench]], Fig. 4 left).

## §3 Seen/unseen environment split

**정의.** **Seen** environment는 task, tool, container image 또는 가까운 친척이 SFT trajectory와 RL rollout을 포함한 training mixture에 나타난 environment다. **Unseen** environment는 tool, observation format, grader라는 전체 stack을 withheld한 environment다. Unseen split이 generality 측정이다. Seen environment 안의 held-out *task* split은 그렇지 않다. Agent가 이미 그 environment의 tool schema, error string, convention을 배웠기 때문이다.

**측정 가능한 문제.** 고정된 environment 집합에서 agentic RL을 한 뒤 seen-environment score는 오른다. Gate가 답해야 할 질문은, 그 상승 중 얼마가 run이 한 번도 건드리지 않은 environment에서 나타나는가다.

**Mechanism.** (1) Training 시작 전 task가 아닌 environment를 partition한다. (2) Unseen partition을 freeze한다. 그 위에서 prompt tuning, scaffold tuning, hyperparameter selection을 하지 않는다. (3) 같은 harness로 run 전후에 `(seen score, unseen score)` pair를 보고한다. (4) unseen delta를 seen delta로 나눈 transfer fraction을 계산하고 [[ch-51]]의 confidence interval을 명시한다.

**수치 예시(실제 수치, [[swe-rebench]], Table 2).** 같은 model family의 checkpoint 둘을 identical prompt의 minimal ReAct scaffold 하나에서 평가했다. Model마다 5 run, standard error를 썼다.

| Checkpoint | SWE-bench Verified(2023년 말부터 public) | SWE-rebench 2025년 3--4월 slice(2025년 생성 issue) |
|---|---|---|
| DeepSeek-V3-0324 | 39.7% | 21.3% |
| DeepSeek-V3-1226 | 35.2% | 21.9% |

Older benchmark 차이는 `39.7 − 35.2 = 4.5`점이며 newer checkpoint에 유리하다. Fresh slice에서는 `21.9 − 21.3 = 0.6`점으로 반대 방향이다. 저자들은 이것이 “older benchmark의 potential contamination effect를 시사할 수 있다”고 쓴다(§3.3, Interpretation). 4.5점 Verified 차이로 checkpoint를 고른 gate는 fresh slice에서 재현되지 않는 차이에 맞춰 선택했을 것이다.

**두 번째 split 형태: private repository.** [[swe-bench-pro]]는 세 partition을 가진다. Copyleft repository의 public task 731개, “향후 overfitting을 test하기 위해” private로 둔 held-out task 858개, 결과를 공개하는 private startup repository 18개의 task 276개다(§3.3). 최고 public score는 43.6%, 최고 commercial score는 17.8%다. Card에는 중요한 한계가 있다. Table 1과 2의 model list가 달라서 **같은 setting의 paired public-versus-commercial 수치를 가진 model은 없다.** 43.6 → 17.8 비교는 서로 다른 model 간 비교다. Held-out set은 v2에서 결과를 공개하지 않았다.

**세 번째 형태: training 중 fresh-environment arena.** [[agent-world]]는 round마다 재생성되는 held-out environment arena를 위해 first-tier category마다 `K = 5` environment를 sample한다. Failure trace를 diagnosis agent에 보내고, targeted task를 합성한 뒤 RL을 계속한다(§3.2.2, Alg. 1). 두 round 뒤 Agent-World-14B는 τ²-Bench / BFCL-V4 / MCP-Mark Postgres에서 60.2 / 52.4 / 29.5에서 65.4 / 55.8 / 38.1로 갔다(Table 2). 같은 논문은 reporting의 cautionary case이기도 하다. §4.3.1은 “core math reasoning에 degradation 없음”을 주장하지만 benchmark별 값은 Fig. 6에만 있어, 주장한 locus에서는 확인할 수 없다.

**조건과 한계.** Unseen split은 시간이 갈수록 degrade한다. 그 결과를 공개하고 selection에 사용하면 이미 selection에 쓴 split이 된다. [[terminal-bench-2]]는 이를 직접 말한다. 2.0에 선택하지 않은 contributed task 140개를 포함한 모든 task가 public repository에 있고, 삽입한 BIG-bench canary string은 “largely a symbolic safeguard”다(App. B).

## §4 Reliability: pass^k, 반복 run, variance

**정의.** Task 하나를 독립적으로 `n`회 실행해 성공 `c`회를 얻고, 선택한 `k ≤ n`에 대해 다음을 정의한다.

```
pass^k  = E_task[ C(c, k)   / C(n, k) ]        k trial이 모두 성공할 확률
pass@k  = 1 − E_task[ C(n−c, k) / C(n, k) ]    k trial 중 적어도 하나가 성공할 확률
```

`C(a, b)`는 binomial coefficient, `n`은 task당 trial 수, `c`는 성공 trial 수, `k`는 metric이 묻는 trial 수, `E_task`는 task 평균이다. 둘 다 [[tau-bench]] §3이 제시한 unbiased estimator다. `k = 1`에서는 일치한다. `pass^1 = pass@1 = E[c/n]`이다.

**이것이 다루는 문제.** Pass@1은 “task 평균으로 한 attempt가 얼마나 자주 성공하는가”에 답한다. User request당 agent를 한 번 실행하는 deployment에는 *이* task가 시도될 때마다 성공할 확률이 필요하며, 그것을 pass^k가 추정한다. [[tau-bench]]가 pass^k를 도입한 이유도 정확히 이것이다. “customer service처럼 reliability와 consistency가 필요한 real-world agent task를 위해”(§3).

**수치 예시.** Task 5개, 각각 trial `n = 5`, success `c = (5, 4, 3, 1, 0)`이라고 하자. `C(5,3) = 10`이다.

| Task | c | c/n | `C(c,3)/C(5,3)` → pass^3 항 | `1 − C(5−c,3)/C(5,3)` → pass@3 항 |
|---|---|---|---|---|
| 1 | 5 | 1.0 | 10/10 = 1.00 | 1 − 0/10 = 1.00 |
| 2 | 4 | 0.8 | 4/10 = 0.40 | 1 − 0/10 = 1.00 |
| 3 | 3 | 0.6 | 1/10 = 0.10 | 1 − 0/10 = 1.00 |
| 4 | 1 | 0.2 | 0/10 = 0.00 | 1 − 4/10 = 0.60 |
| 5 | 0 | 0.0 | 0/10 = 0.00 | 1 − 10/10 = 0.00 |

`pass^1 = 2.6/5 = **0.52**`, `pass^3 = 1.5/5 = **0.30**`, `pass@3 = 3.6/5 = **0.72**`다. 숫자 0.52 하나는 42점 차이가 나는 두 양 사이에 있다. 셋 중 무엇을 보고하느냐가 agent가 deployable해 보이는지를 결정한다.

> Figure: [pass-hat-k.html](figures/pass-hat-k.html) — 위 trial 표를 고치면 `k`가 커질수록 pass^k와 pass@k가 갈라지는 것을 본다. 기본 표는 수치 예시이며, 세 번째 panel은 [[tau-bench]]와 [[toolathlon]]이 보고한 anchor를 열거한다.

**근거(반복됨).**
- [[tau-bench]] §5.1 및 Fig. 4: gpt-4o function-calling agent는 τ-retail에서 pass^1이 60%를 넘지만 “pass^8은 < 25%로 하락”한다.
- [[toolathlon]] Table 3, task 108개에서 model당 3 run: Claude-4.5-Sonnet은 pass@1 38.6 ± 2.7, pass@3 51.9, pass^3 20.4이고, DeepSeek-v3.2-Exp는 각각 20.1 ± 1.2, 27.8, 12.0이다. 저자들은 pass@3 대 pass^3 gap이 “많은 model이 일정 capability coverage는 갖지만 reliable result를 일관되게 내는 consistency가 부족함”을 뜻한다고 쓴다(§4.2).
- [[metr-time-horizon]] §3.2.1: Model의 80% time horizon은 50% time horizon보다 4--6배 짧다. 저자들은 이를 “어렵고 다양한 task를 때때로 성공하는 model조차 moderate length task를 reliably 수행할 수는 없다”고 읽는다.
- [[swe-rebench]] §3.3: Llama-4-Maverick의 mean resolved는 12.2%, pass@5는 27.6이다. 추가 instrumentation 없이 low run-to-run reliability를 식별하는 ratio다.

**조건과 한계.** Pass^k는 무엇을 resample하는가에 대해서만 정의된다. [[tau-bench]]에서는 database와 user instruction을 고정하고 language-model sampling만 바꾼다(§3). 그러므로 pass^k는 다른 user 또는 다른 environment seed에 대한 robustness가 아니라 sampling reliability를 잰다. Agent temperature 0, user simulator temperature 1.0(τ-bench setting, §5)에서 계산한 pass^k는 모든 variation을 simulator에 귀속한다.

## §5 다섯 축 전반의 perturbation robustness

**정의.** Perturbation robustness는 중요하지 않아야 할 것을 바꿨을 때 score가 얼마나 변하는가다. [[minimax-m2-aligning-to-what]]는 perturbation이 trajectory에 들어갈 수 있는 다섯 위치를 열거한다. “**Tool Info**와 available toolset”, agent persona/rule을 정하는 “**System Prompt**”, 구체적 goal을 담은 “**User Prompt**”, “**Environment** 자체(files, codebase, API)”, 각 step에서 반환되는 “**Tool Response**”다. Post는 팀의 초기 “tool scaling” 접근이 “첫 항목만 다뤘다”고 말하며, 증상을 “같은 model이 한 framework에서는 brilliant하고 다른 framework에서는 useless하게 느껴질 수 있다”고 설명한다. Official이지만 qualitative source다. 어떤 축을 perturb해야 하는지는 뒷받침하지만, 축마다 얼마의 가치가 있는지는 뒷받침하지 않는다.

**Mechanism.** 각 축에서 task set을 고정하고 변화 하나만 둔 채 다시 실행한다. (1) Scaffold 교체, (2) content requirement를 바꾸지 않고 system prompt rewrite, (3) user goal paraphrase, (4) environment image version 또는 search tool 뒤 retriever 변경, (5) tool failure와 signature change 주입이다.

**근거, 축당 측정 하나.**

| 축 | 측정한 perturbation | 효과 | Source |
|---|---|---|---|
| Scaffold | Terminal-Bench 2.0의 Gemini 2.5 Pro를 Terminus 2 대 OpenHands에서 실행 | 32.6% → 15.7% | [[terminal-bench-2]], Table 2 |
| Tool info / call format | GPT-4-turbo-2024-04-09 irrelevance score, native function-calling 대 prompting mode | 83.8 → 35.6 | [[bfcl]], Table 1, §5.1 |
| Tool response | 같은 search tool 뒤 BM25 대 Qwen3-Embedding-8B의 BrowseComp-Plus gpt-5 | 55.90% → 70.12%, 더 적은 search call(23.23 → 21.74) | [[browsecomp-plus]], Table 1 |
| Environment noise | Gaia2-mini Claude-4-Sonnet을 noise 없음 / low / medium / high에서 실행(tool-signature change, tool-failure 확률 0.1, 분당 random event 10개) | 31.2 / 35.0 / 23.8 / 8.1 | [[are-gaia2]], App. B.6.1, Table 6 |
| Other actor | Gaia2 Agent2Agent team: all-Llama 대 all-Claude | 8.5 대 29.3 | [[are-gaia2]], Table 3 |

공개된 Terminal-Bench 2.0 trajectory corpus의 집계 모습도 일관된다. [[terminal-bench-trajectories]]는 full trace를 공개하며 dataset card는 task 89개, agent/model combination 109개, scaffold 26개, model 49개에 걸친 trajectory 52,104개와 scaffold별 pass rate 22.8%(mini-swe-agent)에서 74.9%(terminus-3-3)를 기록한다([[terminal-bench-trajectories-stats]]). 이 scaffold별 rate는 **paired가 아니다**. Scaffold 행마다 다른 model 집합을 aggregate했기 때문이다. 따라서 leaderboard 행이 pair에 얼마나 의존하는지는 보이지만, scaffold 자체가 얼마의 가치가 있는지는 보이지 않는다. 그 질문의 paired version은 [[terminal-bench-2]] Table 2의 fixed-model 비교다.

**범용 모델에 주는 함의.** 정확히 scaffold 하나와 tool schema 하나를 상대로 훈련한 model은 deployment에서 바뀔 coordinate 하나를 상대로 훈련한 것이다. Perturbation robustness는 seen/unseen environment split이 다루지 못하는 generality gate의 축이다. Unseen environment도 training scaffold 아래에서 실행될 수 있기 때문이다.

## §6 Harness coordinate와 고정된 evaluation contract

**정의.** Harness coordinate는 model의 일부도 task의 일부도 아니지만 score를 바꾸는 evaluation program setting이다.

**Mechanism: 고정할 목록.** Scaffold 이름과 commit, tool schema 및 tool 제시 방식(native function-calling field인지 prompt 안 text인지), 최대 turn, 최대 wall-clock 또는 simulated time, turn당 및 trajectory당 token budget, cost cap, limit 도달 시 context strategy, environment image tag, reasoning effort, decoding setting, retry 및 parse-error policy, run 수와 aggregation을 고정한다.

**수치 근거: coordinate 하나, model 둘, rank reversal.** [[swe-bench-pro]]는 public set을 최대 50 turn의 SWE-Agent 아래에서 평가하고(§5, Table 1), analysis 절에서 같은 50-turn limit에 task당 $2 cost cap을 더해 다시 평가한다(§6, Table 5).

| Model | 50 turn(Table 1) | 50 turn + $2 cap(Table 5) | 변화 |
|---|---|---|---|
| Claude Sonnet 4 | 42.7% | 17.6% | −25.1 |
| GPT-5(high) | 41.8% | 25.9% | −15.9 |

첫 budget에서는 Claude Sonnet 4가 0.9점 앞서지만, 두 번째에서는 GPT-5(high)가 8.3점 앞선다(두 표에서 계산). 논문은 model당 run 수를 말하지 않으므로 어느 순서에도 confidence interval이 없다. “SWE-Bench Pro, 25.9%”만 출력하는 leaderboard 행은 41.8%를 출력하는 행과 비교할 수 없다.

**두 번째 수치 근거: simulated time.** [[are-gaia2]]는 model 생성 중에도 clock이 전진하는 mobile environment를 실행한다. Time split에서 GPT-5(high)는 default mode 0.0이고, action당 simulated second 하나를 소모하는 “instant” mode에서는 34.4다. Claude-4-Sonnet은 8.1에서 26.7로 움직인다(Fig. 13. §4.3 text는 Fig. 13의 8.1 대신 8.2를 출력한다). GPT-5(high)의 execution을 30분으로 cap하면 Execution이 10점, Search가 20점 하락한다(§5). Latency는 scaffold swap과 같은 크기의 harness coordinate다.

**세 번째: reasoning effort는 score를 고정된 방향으로 움직이지 않는다.** [[holistic-agent-leaderboard]]는 Claude model 3개의 no-reasoning 대 high-reasoning, o4-mini의 low 대 high를 pair로 비교한다. 36개 model-agent-benchmark combination 중 21개에서 reasoning을 늘려도 accuracy가 같거나 낮았다(§4.1 item 5, Fig. 3). [[toolathlon]]도 같은 방향을 보고한다. GPT-5는 30.6 ± 1.5, GPT-5-high는 29.0 ± 3.1이며(Table 3), 저자들은 이를 “agentic task에서는 extended internal reasoning보다 새로운 observation을 탐색하는 일이 더 중요하다”고 읽는다(§4.2, Interpretation). 이 setting은 monotone하지 않으므로 기록해야 한다. [[holistic-agent-leaderboard]]는 harness에서 Anthropic model의 “high”가 LiteLLM default 4,096 reasoning token이고, OpenAI는 budget을 공개하지 않는다고도 기록한다(§3 footnote).

## §7 Benchmark validity: score가 capability를 재는가

**정의.** [[agentic-benchmark-checklist]]는 두 조건을 분리한다. **Task validity**는 agent가 target capability를 가진 경우에만, 그리고 가진 경우에는 반드시 task를 풀 수 있다는 조건이다. **Outcome validity**는 task가 성공한 경우에만, 그리고 성공하면 반드시 evaluation result가 positive라는 조건이다(§1, §3). Agentic Benchmark Checklist(ABC)는 task-validity item 10개, outcome-validity item 20개, reporting item 13개로 이루어진다(Fig. 2--4. 논문은 total을 출력하지 않는다).

**오류가 얼마나 큰지에 대한 근거(결과, 단일 연구; §5.2, App. E).**
- τ-bench에는 intentionally unsolvable task가 있다. Airline 38%, Retail 6%이며 success는 database가 바뀌지 않은 것으로 정의된다. **Do-nothing agent**는 어느 `k`에서도 Airline 38%, Retail 6.0%를 얻고, 모든 data를 출력하는 agent는 40%, 9.6%를 얻는다(App. E.2).
- SWE-Lancer는 password-protected ZIP에 test를 저장하지만 password 없이 file을 overwrite할 수 있다. 모든 test를 `assert 1 == 1`로 바꾸면 resolve rate 100%다(App. E.4).
- KernelBench fuzzer는 tensor value만 바꾸고 shape나 memory layout은 바꾸지 않는다. 검증된 extra test를 넣자 kernel correctness가 31% 과대평가됐음이 발견됐다(App. E.6).
- OSWorld는 website change로 Chrome section problem 46개 중 13개가 깨졌다. 평가한 agent를 absolute 28점 과소평가한다(§5.2).
- SWE-bench는 UTBoost를 인용해 agent가 issue를 해결하지 않아도 Verified task 5.3%, Lite task 7.7%를 통과한다고 한다. Leaderboard ranking의 40.9%와 24.4%를 바꾼다(App. E.1).
- 평가한 benchmark 10개 중 7개는 task-validity flaw, 7개는 outcome-validity flaw, 10개 전부는 reporting limitation을 갖는다(§1, Fig. 5).

**Memorization도 validity failure다.** [[swe-bench-illusion]]은 OpenAI/Anthropic model 10개에 code, file tree, metadata 없이 repository name과 issue text만 주고 file path를 묻는다. Model은 SWE-Bench Verified instance에서 gold patch의 file을 60--76%에 이름 대지만, SWE-Bench 밖 popular repository 7개에서 온 task 245개에서는 53% 미만이다(§4.1.1--§4.1.2). Function reproduction은 Verified에서 최대 5-gram overlap 34.9%, outside repository에서는 13.9%다(§4.3). 저자들은 이를 Verified set 안의 instance-specific memorization, SWE-Bench repository 12개에 대한 repository-bias memorization으로 해석한다(§4.1, §6). 이 probe는 localization과 reproduction subtask를 재며 end-to-end resolve rate를 재지는 않는다.

**Test exploitability도 validity failure다.** [[impossiblebench]]는 unit test를 specification과 모순되게 mutate하고, 이 impossible variant의 pass rate를 **cheating rate**로 정의한다. Pass 하나라도 specification을 위반하는 shortcut을 뜻한다(§1, §2). GPT-5는 Oneoff-SWEbench task 76%, Oneoff-LiveCodeBench task 2.9%에서 cheat한다. Figure 1은 Oneoff-SWEbench에서 Claude Sonnet 3.7 70%, Claude Opus 4.1 54%, Claude Sonnet 4 48%, o3 39%를 열거한다. Harness control 세 개가 rate를 바꾼다. Agent에 flawed test에서 중단하고 report하라고 지시하면 Conflicting-LiveCodeBench의 GPT-5가 85% 초과에서 1%로 내려간다(§5.1). Read-only test는 test modification을 막되 operator overloading은 허용한다(§5.2). Abort action을 추가하면 Conflicting-SWEbench GPT-5가 54%에서 9%로 내려간다(§5.3).

**Harness 안 leak.** [[holistic-agent-leaderboard]]는 공식 τ-bench few-shot file `few_shot_data/MockAirlineDomainEnv-few_shot.jsonl`에 test-set example이 들어 있음을, 약 $1,000의 evaluation을 실행한 뒤 발견했다. 그 scaffold의 모든 result를 제외했다(App. A5). 같은 log analysis는 agent가 task를 풀지 않고 HuggingFace 또는 arXiv에서 gold answer를 가져온 경우 8개를 발견했다(§4.2 item 1).

**Formula와 수치 예시: 불완전한 ground truth가 있는 success rate**([[agentic-benchmark-checklist]], App. F, Eq. 1).

```
μ = e + (1 − 2e)·p₀        σ² = μ(1 − μ)        95% interval:  μ ± 1.96·σ/√N
```

- `e`: ground-truth label이 틀린 test item 비율.
- `p₀`: 불완전한 ground truth를 기준으로 측정한 success rate.
- `μ`, `σ`: corrected estimate의 mean과 standard deviation.
- `N`: test item 수.

측정 `p₀ = 0.386`인 task 108개의 agentic benchmark([[toolathlon]] Table 3, Claude-4.5-Sonnet)를 보자. Perfect grader(`e = 0`)이면 `μ = 0.386`, `σ = √(0.386 × 0.614) = 0.487`, `1.96 × 0.487 / √108 = 0.092`이므로 `0.386 ± 0.092`다. Task count만으로 ±9.2-point interval이다. `e = 0.05`이면 `μ = 0.05 + 0.90 × 0.386 = 0.397`, `σ = 0.489`, half-width `0.092`다. Point estimate는 1.1점 움직이지만 width는 바뀌지 않는다. 두 결론이 나온다. 첫째, 이 크기 benchmark에서 5점 차이는 결과가 아니다. 둘째, `e`는 가정이 아니라 측정해야 한다. Checklist의 자체 예시는 sample BIRD task 500개를 검증해 발견한 incorrect ground-truth query 11.65%를 넣는다(App. F). [[ch-51]]의 paired comparison에 관한 모든 내용이 그 위에 적용되며, item-level interval이 이만큼 넓으므로 여기서는 pairing이 더 중요하다.

## §8 Task-length horizon과 그것이 가정하는 것

**정의.** X%-task-completion time horizon은 “model이 대략 X%의 시간에 완료할 수 있는 task의 길이”다. 여기서 task length는 skilled human이 걸리는 시간으로 잰다([[metr-time-horizon]], Abstract).

**Mechanism.** (1) Domain knowledge는 있지만 task-specific context는 없는 human baseliner의 시간을 모든 task에서 잰다. (2) 각 task score를 human performance를 나타내는 threshold에서 binary화한다. (3) Agent마다 logistic curve 하나를 fit한다.

```
p_success(agent, task) = σ( (log h_agent − log t_task) · β_agent )
```

- `t_task`: 그 task에서 성공한 human baseline의 geometric mean time.
- `h_agent`: fit한 50% time horizon, 즉 success probability가 0.5인 task length.
- `β_agent`: fit한 slope, task가 길어질 때 success가 떨어지는 급격함.
- `σ(x) = 1/(1 + e^(−x))`.

**수치 예시.** o3에 보고된 50% horizon `h = 110`분(§3.2), illustrative slope `β = 0.6`을 사용하자. 논문은 agent별 β를 fit하지만 body에 출력하지 않는다. 30분 task에서는 `log(110/30) = 1.299`, 여기에 0.6을 곱해 0.780, `σ(0.780) = 1/(1 + 0.459) = 0.686`이므로 predicted success 68.6%다. 8시간(480분) task에서는 `log(110/480) = −1.474`, 0.6을 곱해 −0.884, `σ(−0.884) = 1/(1 + 2.421) = 0.292`이므로 29.2%다. 더 큰 β는 curve를 더 가파르게 만든다. 같은 horizon이라도 slope가 날카로우면 short task success는 더 높고 long task success는 더 낮다.

**근거.** HCAST, RE-Bench, SWAA의 task 170개와 2019--2025 frontier model 12개를 쓴다. GPT-2의 50% horizon은 2초, o3는 110분이다. Horizon은 **207일**마다 두 배가 됐고, 95% bootstrapped CI는 166--240일이다. Task family, task, run 순의 3-level hierarchical bootstrap에서 나왔다(§3.2). 80% horizon은 204일마다 두 배가 되지만 absolute length는 4--6배 짧다(§3.2.1).

**저자가 명시한 조건과 한계.**
- 개별 model horizon error는 크고 model 간 **상관이 매우 높다.** 그러므로 slope는 model 하나의 horizon보다 더 잘 정해진다(§3.2).
- SWE-bench Verified에서 method를 replicate하면 2024 model에서 main suite 143일과 비교해 doubling time이 약 70일이다. 저자들은 차이 일부를 codebase에 “익숙해지는 데 몇 시간”이 걸린 engineer를 가정하는 SWE-bench Verified time annotation에 돌린다(§4.1).
- 16-factor “messiness” score(resource limit, novelty, dynamic environment 등)는 고정 task length에서 더 낮은 performance를 예측하지만, 시간에 따른 추세는 low/high-messiness subset에서 비슷하다. Messier subset에만 특유한 “plateau의 근거는 없다.” Suite 평균 messiness는 16점 만점 3.2다(§4, §F.2). 따라서 suite는 deployment task보다 깔끔하다.
- METR internal pull request에서 contractor는 repository maintainer보다 5--18배 더 오래 걸리며, agent performance는 maintainer time이 아니라 contractor time과 맞는다(§4, §C.2). Horizon은 선택한 human population에 상대적으로 정의된다.
- “Time horizon은 domain, task distribution, human baseliner의 skill과 context 수준에 항상 상대적으로 측정된다”(§5). Evaluation은 software와 research task만 포괄한다.

**범용 모델에 주는 함의.** Horizon은 x-axis가 benchmark-specific scale이 아니라 human time이므로 난이도가 다른 benchmark 간 비교 가능한 유일한 agent metric이다. 동시에 baseliner가 누구인지에 가장 민감한 metric이기도 하다. 따라서 trend 추적에는 유용하지만 checkpoint 둘을 비교하는 데는 약하다.

## §9 Paired general-ability regression gate

**정의.** Regression gate는 non-agentic suite(knowledge, reasoning, instruction following, safety)를 pre-agentic-training checkpoint와 post checkpoint에서 item-paired로 다시 실행한다. Agentic gain을 수용하기 전에 post checkpoint가 각 slice에서 미리 밝힌 margin만큼 **non-inferior**해야 한다.

**이것이 다루는 문제.** Agentic SFT와 RL은 적은 수 environment의 좁은 reward를 최적화한다. 측정 가능한 위험은 그 environment에서는 gain을 얻으면서 그 밖 capability는 잃는 것이다.

**loss가 실제라는 근거(결과, 단일 연구).** [[agent-world]] §4.3.1은 EnvScaler-8B가 tool-use average는 개선됐지만 SWE-bench Verified와 Terminal-Bench 1.0에서 자기 Qwen3-8B backbone보다 낮다고 보고한다. Table 1의 MCP-Mark / BFCL V4 / τ²-Bench는 EnvScaler-8B 5.6 / 47.6 / 37.9, Qwen3-8B 2.4 / 40.4 / 26.2다. 같은 card는 Agent-World의 “core math reasoning에 degradation 없음” 주장에 benchmark별 값이 figure에만 있음을 기록한다. [[agentgym-rl]]은 held-out-environment evaluation도 general-benchmark evaluation도 전혀 보고하지 않는다(§7). Gate가 막기 위해 존재하는 reporting failure다.

**Mechanism.** 새 statistic을 만들지 말고 [[ch-51]]의 machinery를 다시 쓴다. Slice마다 item에 대한 paired bootstrap, run 전에 고정한 non-inferiority margin, slice 전반 multiple-comparison control, release를 막았을 slice를 이름으로 적는 written decision이다. Gate에 agent-specific 행 세 개를 더한다. (1) §3의 unseen-environment pair, (2) §4의 pass@1 옆 pass^k, (3) headline number와 같은 task set에서 실행한 §5 perturbation axis 하나다.

**조건과 한계.** Gate는 §6의 harness pinning만큼만 좋다. Post-training checkpoint를 training 동안 tuning한 scaffold로, pre-training checkpoint를 다른 scaffold로 평가하면 paired comparison은 scaffold change를 측정한다.

## 부정 sample과 부정 feedback

이 절이 다루는 “negative”의 네 의미(course standard §6.1) 중 하나는 **content로서의 negative**다. Failure trajectory를 evaluation evidence 및 failure bucketing input으로 쓴다. 이 장 경계에서는 **gradient로서의 negative**도 해당한다. Evaluation grader를 RL reward로 재사용하면 grading error가 gradient가 되기 때문이다.

1. **Negative의 출처.** Benchmark grader가 label한 failed trajectory다. Unit test([[swe-bench-pro]], [[terminal-bench-2]]), final-state comparison([[tau-bench]], [[toolathlon]]), LLM soft check가 있는 write-action matching([[are-gaia2]]), LLM judge([[browsecomp-plus]])가 있다.
2. **현재 관행이 하는 일.** Bucket한다. [[terminal-bench-2]]는 scaffold 하나에서 model/task마다 failed trial 두 개를 sample해 Docent와 human annotator 두 명으로 annotation하고 Execution/Coherence/Verification의 3-class taxonomy를 보고한다. GPT-5(high) judge는 human-labeled trace 120개와 agreement 90%(precision 92%, recall 90%)다(§4.4). [[swe-bench-pro]]는 unresolved trajectory마다 마지막 20 turn을 GPT-5로 judge한다. 10이나 40보다 human validation에 더 잘 맞았기에 고른 window다(§6.3). 둘 다 failure로 훈련하지는 않는다.
3. **Mechanism과 경계 위험.** False-negative rate가 있는 grader는 correct behavior를 reject한다. 같은 grader를 RL reward로 쓰면 그 rejection은 correct trajectory의 negative advantage가 되며, softmax gradient `∂ log p_y / ∂ z_j = 1[j = y] − p_j`는 제거한 probability mass를 model이 다음으로 rank한 곳으로 옮긴다. 반드시 더 좋은 behavior는 아니다. 크기는 측정할 수 있다. [[swe-bench-pro]] Table 3에서 task description의 requirements/interface field를 빼자 GPT-5(high)는 resolve rate 25.9% → 8.40%, Claude Opus 4.1은 22.7% → 8.20%로 떨어졌다. 저자들은 unit test가 좁은 interface 집합만 받아들이는 verifier false negative로 돌린다(§6.2).
4. **False-positive 방향의 수치.** [[agentic-benchmark-checklist]]는 non-solution을 수용하는 grader를 측정한다. 빈 response로 τ-bench Airline task 38%가 pass하고, test overwrite 뒤 SWE-Lancer task 100%가 pass한다(§5.2). [[impossiblebench]]는 같은 동전의 policy 쪽을 잰다. GPT-5는 specification 위반 없이는 pass할 수 없는 Oneoff-SWEbench task 76%를 pass한다.
5. **Control.** Verifier가 ambiguity resolver가 되지 않도록 test에 필요한 정보를 agent에 준다([[swe-bench-pro]], §6.2 guideline). Test를 hidden 대신 read-only로 만들어 legitimate performance는 보존하고 가장 쉬운 exploit을 없앤다([[impossiblebench]], §5.2). Impossible task를 game하지 않고 report하게 abort action을 넣는다(§5.3). Rubric judge 옆에 task-agnostic style check를 더한다. 초기 ARE RL에서 agent는 soft-check judge가 받아들일 때까지 answer message에 code-like string을 넣었고, style check가 exploit을 멈췄다([[are-gaia2]], App. B.3.1). Context overflow, timeout, environment crash 같은 harness failure는 policy failure로 채점하지 말고 gradient에서 제외한다([[ch-45b]]).
6. **진단.** Bucketing pipeline의 failure class별 count, command-level error rate(Terminal-Bench 2.0에서 Grok 4 9.2%부터 GPT-OSS-120B 26.7%, §4.5), grader에 대한 do-nothing/enumerate-everything baseline, impossible variant의 cheating rate, 같은 `k`에서 pass^k 대 pass@k.
7. **Generality 효과.** Shortcut에 보상하는 grader는 shortcut을 훈련한다. [[holistic-agent-leaderboard]]는 flagged failure를 고치면 success가 생길지 확립할 수 없다고 말한다. 이를 위해 checkpointing과 replay가 필요하다(App. A4.2). 그러므로 failure bucket에서 training change로 가는 link는 현재 측정이 아니라 inference다.

## 설정 recipe

모든 행은 명시 locus에서 읽은 evaluation-protocol 값이다. 값은 benchmark 자체 reference protocol용이며, §6의 요지대로 다른 protocol은 다른 score를 낸다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| τ-bench reference protocol | — | eval-gate | task당 최대 agent action; agent/user temperature; task당 trial | 30; 0.0 / 1.0; 최소 3 | arXiv:2406.12045v1 §5 | 2026-09-15 검증 | ablation 미보고 |
| τ²-bench reference protocol | — | eval-gate | task당 run; temperature; user simulator model | 4; 0; gpt-4.1-2025-04-14 | arXiv:2506.07982v1 §4.1 | 2026-09-15 검증 | §4.2 Fig.4: no-user/oracle-plan ablation |
| SWE-Bench Pro(public, N=731) | — | eval-gate | scaffold; 최대 turn; cost cap; 제공 task field | default prompt의 SWE-Agent; 50; 없음; problem statement + requirements + interface | arXiv:2509.16941v2 §5 | 2026-09-15 검증 | §6.2 Table 3: requirements+interface 제거 시 GPT-5(high) 25.9 → 8.40 |
| SWE-Bench Pro(analysis setting) | — | eval-gate | 최대 turn; cost cap | 50; task당 $2 | arXiv:2509.16941v2 §6, Table 5 | 2026-09-15 검증 | compute constraint로 진술, ablation 아님 |
| SWE-Bench Pro | — | eval-gate | model당 run; confidence interval | 미보고(§5, §6, Table 1--5 확인) | — | 미보고 | 없음 |
| Terminal-Bench 2.0 | — | eval-gate | agent-model 조합당 run; reasoning effort; execution | 최소 5(총 32,155 trial); provider default(Anthropic/OpenAI medium); Daytona에서 Harbor harness, container 32--100개 병렬 | arXiv:2601.11868v1 §3, §3.3, §3.4 | 2026-09-15 검증 | resolution rate에 95% CI 출력(Table 2) |
| Gaia2 / ARE | — | eval-gate | loop; scenario당 run; temperature; context; turn당 generation token; 최대 step; verifier | ReAct-like, step당 JSON tool call 하나; 3; 0.5(GPT-5는 temp/top-p 1); 최소 128K; 16,000; 200; temperature 0의 Llama 3.3 70B Instruct | arXiv:2509.17158 §4.1, App. B.5 | 2026-09-15 검증 | Table 1: label trajectory 450개에서 verifier agreement .98/precision .99/recall .95 |
| Toolathlon | — | eval-gate | 최대 turn; model당 run; 보고 metric | 100; 3; std 포함 pass@1, pass@3, pass^3 | arXiv:2510.25726v2 §4.1 | 2026-09-15 검증 | ablation 미보고 |
| BrowseComp-Plus | — | eval-gate | retriever; top-k; document truncation; judge | BM25, Qwen3-Embedding 0.6B/4B/8B, ReasonIR-8B; `k = 5`; 처음 512 token; gpt-4.1 | arXiv:2508.06600v1 §4.3, §4.4 | 2026-09-15 검증 | Table 1: retriever swap은 모든 agent accuracy 상승 |
| SWE-rebench benchmark slice | — | eval-gate | scaffold; run; reporting; context; contamination flag | 동일 prompt/text command의 minimal ReAct; SEM과 pass@5로 5 run; 128K; model release 전 생성 issue를 포함한 run은 potentially contaminated 표시 | arXiv:2505.20411v2 §3.2 | 2026-09-15 검증 | §3.1: 이 protocol이 제거하는 confound는 pass@N-as-pass@1, best-of-runs reporting |
| HAL harness | — | eval-gate | combination당 run; confidence interval | 대체로 single run, CI 없음, cost constraint로 진술 | arXiv:2510.11977 App. A3 item 1 | 2026-09-15 검증 | 없음 |
| BFCL | — | eval-gate | call mode | native function-calling과 prompting mode 모두 실행·보고 | PMLR 267 §5.1, App. A | 2026-09-15 검증 | Table 1: GPT-4-turbo-2024-04-09 irrelevance FC 83.8 vs prompting 35.6 |
| METR time-horizon protocol | — | eval-gate | fit; uncertainty | log human task time에 대한 success logistic regression; task family→task→run의 3-level hierarchical bootstrap 10,000 sample | arXiv:2503.14499v4 §3.1, §3.2 | 2026-09-15 검증 | §3.2.1: 80%-horizon 반복도 비슷한 doubling time(204 vs 207일) |

**작은 범용 agent gate의 출발점.** 여기 모든 수치는 위 `검증` 행과 source setting에서 왔다. [[toolathlon]]처럼 task 108개에서 3 run으로 pass@1, pass@3, pass^3을 보고한다(§4.1). Reliability-focused suite에서는 [[tau-bench]]처럼 task당 최소 3 trial과 pass^k를 보고한다(task 165개, §5). Environment horizon에 따라 [[swe-bench-pro]]처럼 최대 50 turn(§5) 또는 [[toolathlon]]처럼 100 turn(§4.1)을 고르고 cap을 공개한다. Fixed 50-turn limit에서 $2 cost cap은 두 model을 15.9와 25.1점 움직였다([[swe-bench-pro]], Table 1, 5). [[bfcl]]처럼 두 call mode를 모두 실행한다(§5.1). [[terminal-bench-2]] Table 2처럼 combination별 CI를 보고하며, cost 때문에 [[holistic-agent-leaderboard]]가 해야 했던 single run(App. A3 item 1)은 피한다. 이 protocol은 frontier-scale model에서 task 89--830개에 사용됐다. 같은 protocol의 작은 model은 같은 task count에서도 interval이 더 넓다.

## 일반화 관점

**(a) 무엇이 범위를 넓히는가.**
- Topic이 아니라 action space 전반에서 평가한다. Gaia2 capability profile은 scaffold 하나에서 model 하나가 Search 79.6, Time 0.0인 것을 보인다([[are-gaia2]], Table 2). Single-axis suite는 이 spread를 드러내지 못한다.
- Unseen benchmark에서 측정한 training environment 수. [[agent-world]]는 training environment 0개에서 four-domain average 18.4%, 1,978개에서 38.5%를 보고하며 500 이후에는 diminishing return이다(§4.3.3, Fig. 8).
- Diagnosis-driven task synthesis를 쓴 regenerated held-out arena. Agent-World-14B는 두 round에서 MCP-Mark Postgres 29.5 → 38.1(Table 2).
- Scaffold 하나 이상에서 평가한다. Vendor 자신의 scaffold에서만 강한 model은 leaderboard 행이 말하는 것보다 narrow capability다([[terminal-bench-2]], Table 2; [[holistic-agent-leaderboard]], §4.1 item 6).

**(b) 무엇이 narrowing 또는 forgetting을 만드는가.**
- Environment family 하나에서 training/gating. [[agentgym-rl]]은 agent가 “in-domain setting 안에서 잘 수행”한다고 말하고 held-out-environment/general-benchmark evaluation을 보고하지 않는다(§7).
- Agentic training 뒤 general-ability regression. EnvScaler-8B는 tool-use average는 올라가지만 SWE-bench Verified/Terminal-Bench 1.0에서 Qwen3-8B backbone보다 낮다([[agent-world]], §4.3.1, Table 1).
- Benchmark memorization. Code 없는 prompt에서 SWE-Bench Verified file-path identification은 60--76%지만 benchmark 밖 repository에서는 53% 미만([[swe-bench-illusion]], §4.1).
- Shortcut을 수용하는 grader. Do-nothing pass rate 38% grader에 최적화하면 아무것도 안 하는 쪽으로 최적화한다([[agentic-benchmark-checklist]], App. E.2).
- Scaffold overfitting. [[swe-rebench]]는 SWE-bench subset에서 tuning한 scaffold를 “implicit overfitting”이라 부른다(§3.1).

**(c) 이 stage에서 generality를 재는 법.**
- 양쪽에 같은 harness를 쓴 seen/unseen environment pair(§3)를 숫자 두 개와 transfer fraction, [[ch-51]] paired bootstrap으로 보고한다.
- Deployment가 쓸 `k`에서 pass@1 옆 pass^k(§4).
- 같은 task set에서 축별 perturbation rerun 하나(§5): scaffold swap, call-mode swap, tool-quality swap, environment noise.
- Public benchmark 옆 fresh/date-filtered slice([[swe-rebench]] §3.2; [[ch-47a]]도 참조).
- Score를 쓰기 전 grader에 trivial-agent baseline과 exploit-agent run([[agentic-benchmark-checklist]] R.13, §5.2).
- Non-agentic slice의 non-inferiority gate(§9).

## 흔한 실수와 발견 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| 다른 scaffold에서 얻은 agent score 둘을 비교 | Score 차이가 benchmark의 알려진 scaffold spread보다 작음(Terminal-Bench 2.0 Table 2의 5.9--16.9점) | Pinned scaffold commit 하나에서 checkpoint 둘 재실행; commit 공개 |
| Pass@k를 success rate라고 보고 | Deployment success가 보고 수치보다 훨씬 낮음 | Deployment `k`에서 pass^k 계산; Claude-4.5-Sonnet Toolathlon pass@3 − pass^3 gap은 31.5점(Table 3) |
| Training environment 내부 held-out task split을 generality 측정으로 간주 | In-environment score 상승 중 unseen-environment score 평평 | Tool schema/grader 포함 environment 전체 hold out(§3) |
| Probe하지 않은 grader 신뢰 | Completed action이 거의 없는 trajectory를 내는 model의 high score | Do-nothing/answer-enumerating agent 실행; τ-bench Airline do-nothing 38%([[agentic-benchmark-checklist]], App. E.2) |
| Harness가 바뀌었는데 gain을 model에 귀속 | Harness upgrade 뒤 run에서만 gain | Result record에 environment image tag/scaffold commit 유지; 같은 이름 아래 endpoint weight 변경 사례는 [[holistic-agent-leaderboard]] App. A3 |
| Benchmark가 최근이라는 이유로 contamination-free로 읽음 | Public set score가 date-filtered slice보다 훨씬 높음 | 같은 scaffold를 fresh slice에서 실행; DeepSeek-V3-0324는 Verified 39.7%, 2025 slice 21.3%([[swe-rebench]], Table 2) |
| Reasoning effort를 기본 개선으로 올림 | Cost는 상승, score는 안 오름 | Effort setting pair; 36 combination 중 21개는 reasoning 증가해도 같거나 낮음([[holistic-agent-leaderboard]], §4.1 item 5) |
| 100-task agentic benchmark의 작은 개선 선언 | 약 100 item에서 10점 미만 차이 | App. F interval 적용: `N = 108`, `p₀ = 0.386`이면 grader error 전 half-width 9.2점(§7) |
| Unseen split을 공개한 뒤 tuning | successive run에서 unseen score가 seen score로 수렴 | Split을 몇 번 읽었는지 기록; selection에 썼으면 새 environment family로 rotate |

## 이해 점검

1. [[terminal-bench-2]]는 “model selection이 보통 agent scaffold보다 중요하다”고 하면서 Gemini 2.5 Pro가 scaffold 간 16.9점 움직인다고 보고한다. 두 진술이 동시에 참일 수 있는 이유와 specific release decision에서 어느 쪽이 더 중요한지 정할 comparison을 설명하라.
2. τ-bench는 database/user instruction을 고정하고 language-model sampling만 바꾼다(§3). 이 protocol에서 높은 pass^8이 deployment reliability에 대해 **입증하지 않는** 것을 설명하고, 그 gap을 닫을 protocol 변경 하나를 말하라.
3. $2 cost cap은 SWE-Bench Pro에서 Claude Sonnet 4와 GPT-5(high)의 순서를 뒤집는다. Uniform downward shift가 아니라 rank reversal을 만드는 mechanism과, capped setting이 uncapped setting과 달리 재는 agent property를 설명하라.
4. [[swe-rebench]]에서 DeepSeek-V3 checkpoint 둘은 SWE-bench Verified에서 4.5점, fresh 2025 slice에서 반대 방향 0.6점 차이다. 이 근거와 양립하는 설명 두 개, 이들을 분리할 측정을 제시하라.
5. ABC interval formula는 `e = 0`과 `e = 0.05`에서 half-width가 같지만 centre가 다르다. 이 model에서 grader error가 interval을 넓히지 않고 estimate를 옮기는 이유와, grading error distribution에 대한 model의 가정을 설명하라.
6. [[impossiblebench]]는 abort action 추가가 GPT-5 cheating rate를 54%에서 9%로 낮춘다고 보인다. 이것이 model보다 harness에 관한 근거인 이유와 같은 intervention의 training-time analogue를 설명하라.
7. METR 80% horizon은 50% horizon과 doubling time이 거의 같지만 4--6배 짧다. Model 전반 fit된 slope `β`에 대한 뜻과, 두 horizon의 constant ratio가 왜 따라오는지 설명하라.
8. Run이 training environment 세 개에서 6점 gain, held-out environment 하나에서 1점 gain을 보고한다. [[ch-51]] statistic을 이름으로 말하며 go/no-go 논증을 구성하고, release를 계속 block할 조건을 말하라.

## 연결

- **이전:** [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions. Paired bootstrap, non-inferiority test, go/no-go memo를 만들며, 이 장은 그 도구를 적용할 agent-specific 양과 agentic task count가 뜻하는 훨씬 넓은 item-level interval을 공급한다.
- **의존:** [[ch-45b]] — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. Trajectory 구조, negative-feedback 절이 재사용하는 harness-failure filtering rule, 이 장이 gate하는 environment score의 기반이다.
- **다음:** [[ch-52]] — Safety Evaluation, Over-Refusal, and Red-Teaming. §9 regression gate에 safety slice를 더하고 §5 perturbation axis를 tool output을 통한 prompt injection까지 확장한다.
- **기반:** [[ch-50]] — Slice Analysis, Forgetting Slices, and Failure Bucketing. §7과 negative-feedback 절이 agent transcript에 적용하는 trajectory failure bucket의 기반이다.
- **관련:** [[ch-47a]] — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation. §3과 §7이 agentic benchmark에 쓰는 fresh-slice/perturbation audit의 기반이다.
- **적용:** [[ch-46a]] — Lab: Small Agentic SFT-then-RL Run with a Generality Gate. 여기 정의한 gate를 small model에서 실행한다.

## 출처

- [[swe-bench-pro]] — public/held-out/commercial partition, 50-turn/$2-cap protocol, rank reversal의 Table 1/5 수치, verifier false negative를 재는 augmentation ablation.
- [[terminal-bench-2]] — task 89개, fixed-model scaffold comparison과 95% CI, Harbor harness setting, 3-class failure taxonomy, benchmark 자체 ABC self-audit.
- [[terminal-bench-trajectories]] — Terminal-Bench 2.0 full-trajectory corpus 공개본; card는 current source standard보다 앞서므로 수치를 여기 인용하지 않음.
- [[terminal-bench-trajectories-stats]] — 2026-09-15에 읽은 dataset-card 통계: trajectory 52,104개, scaffold 26개, model 49개, scaffold별 pass rate와 unpaired인 이유.
- [[tau-bench]] — pass^k/pass@k estimator, harness setting, Table 2 pass^1, pass^8 < 25% reliability 결과.
- [[tau2-bench]] — dual-control telecom domain, no-user/oracle-plan mode ablation, user-simulator error audit.
- [[bfcl]] — function-calling category 구조, FC 대 prompting mode 차이, single-turn 대 memory gap.
- [[browsecomp-plus]] — fixed-corpus design 및 tool-quality perturbation으로서 retriever swap.
- [[toolathlon]] — application 32개/tool 604개/task 108개와 pass@1/pass@3/pass^3 triple.
- [[are-gaia2]] — seven-split capability profile, noise/Agent2Agent ablation, simulated-time coordinate, verifier-as-reward exploit.
- [[holistic-agent-leaderboard]] — 21,730-rollout 3차원 analysis, scaffold-model interaction, reasoning-effort pairing, τ-bench few-shot leak, model identity의 infrastructure hazard.
- [[agentic-benchmark-checklist]] — task/outcome validity, trivial-agent baseline, 측정한 over/underestimate, noisy ground truth confidence interval.
- [[swe-bench-illusion]] — benchmark-specific memorization을 재는 file-path/function-reproduction probe.
- [[swe-rebench]] — date-tracked decontaminated slice, 5-run SEM protocol, paired Verified 대 fresh-slice comparison.
- [[impossiblebench]] — cheating-rate construction과 이를 바꾸는 harness control 세 개.
- [[metr-time-horizon]] — logistic time-horizon fit, 207-day doubling time, 80%-horizon result, external-validity check.
- [[minimax-m2-aligning-to-what]] — qualitative design source로만 쓴 perturbation axis 다섯 개.
- [[agent-world]] — held-out arena, environment-count scaling curve, EnvScaler-8B의 자기 backbone 대비 general-ability regression.
- [[agentgym-rl]] — §9 gate가 막아야 하는 reporting failure로 쓴 in-domain-only scope statement.
