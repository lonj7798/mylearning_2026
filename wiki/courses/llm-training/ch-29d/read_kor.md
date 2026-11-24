<!-- chapter: ch-29d
     track: synthetic
     kind: content
     title: 사용자 시뮬레이터, 궤적 검증, 실패한 궤적
     deps: [ch-29c, ch-22]
     sources: [[tau-bench]], [[tau2-bench]], [[apigen-mt]], [[kimi-k2]], [[userrl]], [[simulator-collapse]], [[simulator-collapse-recipe]], [[lost-in-multi-turn]], [[explorer]], [[agent-early-experience]], [[agent-early-experience-recipe]], [[os-genesis]], [[learning-from-failure-nat]], [[eto-trial-and-error]], [[ipr-step-level-refinement]], [[glm-5]], [[deepswe]], [[deepswe-recipe]], [[openthoughts-agent-glm46-teacher]], [[together-coderforge-agent-trajectories]], [[likelihood-displacement]]
     figures: figures/failed-rollout-signal.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29d — 사용자 시뮬레이터, 궤적 검증, 실패한 궤적

> **핵심 통찰.** 사용자 시뮬레이터는 훈련 데이터 분포의 일부이며, 그 오류는 측정할 수 있다. τ²-bench에서 수동 주석은 항공 대화의 47%, 소매 대화의 40%, 통신 대화의 16%에서 사용자 시뮬레이터 오류를 찾았고, 이 중 과업 완료를 막는 오류를 포함한 대화 비율은 각각 13%, 12%, 6%였다([[tau2-bench]] Table 2). 하나의 고정된 LLM 시뮬레이터를 상대로 에이전트를 RL로 훈련하면 held-out 성공률은 초기에 최고점에 도달한 뒤 미훈련 수준 쪽으로 되돌아가고 정책 엔트로피는 하락한다. Verbalized Sampling, 세 계열 시뮬레이터 앙상블, 공동 훈련은 Qwen3-4B-Instruct의 τ²-bench Retail held-out 성공률을 46.1%(단일 시뮬레이터 최선 checkpoint)에서 55.5–62.2%로 올렸다([[simulator-collapse]] Table 1). 검증은 어느 궤적이 목표가 되는지를 결정하며, 검증기도 측정 가능한 오류를 낸다. Explorer 궤적 100개 중 GPT-4o 검증기가 통과시킨 것 가운데 14%는 사람이 실패로 라벨링했다([[explorer]] Table 10). 실패 궤적은 가장 흔하게 버려지지만, 공개된 근거가 있는 대안은 네 가지다. 오류 구간을 loss에서 mask하는 방법([[glm-5]] §3.1), 실패를 `incorrectly` 접두사 아래에서 훈련하는 방법([[learning-from-failure-nat]] Table 2), 전문가 성공 궤적에 맞선 궤적 수준 DPO([[eto-trial-and-error]] Table 2), Monte Carlo rollout에서 만드는 단계 수준 대조 쌍([[ipr-step-level-refinement]] Table 2)이다.
>
> **지침.** 사용자 시뮬레이터가 훈련 궤적이나 RL reward를 생성한다면, 그 대화의 표본에 critical 오류와 benign 오류를 주석으로 달고, 훈련에 쓰지 않은 시뮬레이터 계열 및 실제 사용자에서 평가하며, 정책 엔트로피와 zero-variance group 비중을 추적하라. [[simulator-collapse]]의 세 단일 시뮬레이터 실행에서는 모두 훈련 reward가 상승하는 동안 held-out reward가 하락했기 때문이다(Fig. 3). 신뢰할 수 있는 outcome check가 실패를 라벨링하고 같은 과업에 대한 전문가 또는 성공 궤적이 있다면, 실패를 버리는 대신 쌍으로 만들거나(ETO, IPR) 조건으로 사용하고(NAT), 양의 likelihood 항을 유지하라. IPR에서 SFT 항을 제거하면 InterCodeSQL reward가 61.3에서 31.7로 떨어졌기 때문이다([[ipr-step-level-refinement]] Table 4). rollout이 환경 crash로 끝나면 실패 점수를 주지 말고 group에서 제거하라. GLM-5는 이런 실패가 모델 능력이 아니라 환경 불안정을 반영한다고 명시한다([[glm-5]] §4.1.2). timeout, context limit, step limit으로 끝난 rollout은 종료 사유를 기록하고 DeepSWE처럼 mask하라. 이런 실행에 보상 또는 벌점을 주는 것은 DeepSWE의 Qwen3-14B ablation에서 reward collapse보다 앞서 나타났으며, 그 근거는 곡선으로만 제시됐다([[deepswe]] §2.3, Figure 6). 단, loop나 중복 탐색이 이런 종료의 대부분을 유발한다면 masking은 그 행동을 억제하는 압력을 없앤다(열린 질문, §6). 실패가 LLM judge만으로 라벨링됐다면 어떤 negative-gradient 항에도 넣지 마라. Explorer 검증기가 거부한 궤적 중 0.05 / 0.47 = 10.6%는 사람이 성공으로 라벨링했으며([[explorer]] Table 10에서 도출), 이 장의 어떤 출처도 judge-라벨 실패에 negative-gradient 훈련을 시험하지 않았다.

## 범용 모델에서 이 장이 중요한 이유

에이전트 post-training 데이터는 궤적으로 구성된다. 궤적은 모델 메시지, tool call, tool observation, 그리고 대화형 과업에서는 사용자 메시지의 연속이다. 이 장은 환경을 SFT 데이터와 RL signal로 바꾸는 파이프라인의 세 구성요소를 분리한다. **사용자 시뮬레이터(user simulator)** 는 에이전트에게 숨긴 과업 지시를 받아 사용자를 연기하는 언어 모델이다. **궤적 생성기(trajectory generator)** 는 환경에서 행동하는 정책 또는 teacher다. **검증기(verifier)** 는 완성된 궤적을 성공 또는 실패로 라벨링하는 규칙, test, 또는 judge다. ch-29c는 환경과 과업을 만들었고, 이 장은 환경을 SFT 데이터와 RL signal로 전환하는 세 구성요소를 다룬다.

각 구성요소는 모델을 좁힐 수 있다. 하나의 지배적인 행동을 가진 시뮬레이터는 정책에게 그 행동만 처리하도록 가르친다. 실패를 통과시키는 검증기는 오류를 양의 목표로 바꾼다. 모든 실패를 버리는 파이프라인은 어떤 행동이 실패하는지에 관한 정보를 제거하고, 모든 timeout에 벌점을 주는 파이프라인은 실제로는 올바른 궤적의 확률을 낮춘다. 측정할 질문은 (1) 각 구성요소가 얼마나 자주 틀리는가, (2) 그 오류가 held-out 성능을 어떻게 바꾸는가, (3) 실패 궤적의 어떤 사용법이 어떤 라벨 아래 held-out 성능을 개선하는가이다. 강의 파이프라인에서 이 장의 위치는 SFT 데이터 구축과 SFT에서 RL로 넘어가는 hand-off다(pre-training → mid-training → **SFT** → preference optimization → **RL** → evaluation). ch-22의 품질·다양성 선택법도 궤적에 적용되며, 이 장은 에이전트에 특유한 검증과 실패 처리까지 더한다.

이 장에서 negative는 언제나 Negative samples 절이 정의한 네 의미 가운데 하나다. negative marginal value(버림), content로서의 negative, conditioning으로서의 negative, gradient로서의 negative이다.

## §1 데이터 구성요소로서의 사용자 시뮬레이터

**정의.** 사용자 시뮬레이터는 에이전트에게는 숨겨진 과업 지시를 받고 다중 turn 상호작용의 사용자 측 발화를 만드는 LLM이다. 그 지시는 보통 단 하나의 결과가 정답이 되도록 사용자의 신원, 목표, 선호를 정한다([[tau-bench]] §3).

**해결하려는 문제.** 에이전트가 여러 turn에 걸쳐 정보를 수집해야 하는 과업은 사람 사용자를 써서 대규모로 실행할 수 없다. τ-bench는 gpt-4를 시뮬레이터, gpt-4o를 에이전트로 쓸 때 τ-retail 과업 하나당 시뮬레이터 비용을 $0.23, 에이전트 비용을 $0.38로 추정한다([[tau-bench]] §5.1). UserRL은 RL 실행 한 번에 시뮬레이터 요청이 약 4M회라고 추정한다(2k 과업 × 15 epoch × 8 rollout × 16 turn)([[userrl]] App. C). 측정 가능한 문제는 시뮬레이터와의 대화가 사람과의 대화와 같은 에이전트 행동 및 같은 성공 추정치로 이어지는지다.

**출처에 있는 다섯 설계.**

1. **숨은 지시를 받는 LLM 사용자(τ-bench, 2024-06).** gpt-4-0613이 system prompt의 지시를 바탕으로 사용자를 연기하며, 대화는 보지만 에이전트의 tool call은 보지 못한다. 이 시뮬레이터가 `###STOP###`을 출력하면 episode가 끝난다([[tau-bench]] §3). 최종 database가 주석된 목표와 같고 에이전트 메시지가 필수 출력을 포함할 때만 reward는 1이다.
2. **이중 제어 아래 tool 제약을 받는 사용자(τ²-bench, 2025-06).** 통신 도메인에서 사용자는 shared state 위의 tool도 쓴다(예: airplane mode 전환). 사용자 tool은 사람이 읽을 수 있는 출력을 돌려주며, prompt는 사용자의 tool call을 반응형으로 제한한다. 즉, “에이전트가 요청했을 때만 tool을 호출하라” 및 “정보를 점진적으로 공개하라”는 제약이다([[tau2-bench]] §1, App. C.2).
3. **Best-of-N을 쓰는 persona 안내 사용자(APIGen-MT, 2025-04).** 사람 역할의 LM이 검증된 과업 blueprint를 turn에 걸쳐 공개한다. 무작위성을 줄이기 위해 시뮬레이션 사용자는 응답 전에 N = 4개의 후보 답변을 sample하고 자기 비평한다. τ-bench retail에서 이 방법은 gpt-4o 성공률을 62.8에서 67.0으로 올리고, 5회 trial의 분산을 11.1에서 2.6으로 낮췄다([[apigen-mt]] §4.2, Table 3).
4. **tool 시뮬레이터와 LLM 생성 persona(Kimi K2, 2025-07).** “서로 다른 의사소통 스타일과 선호를 가진 LLM 생성 사용자 persona가 다중 turn 대화에 참여”하고, stateful tool simulator는 “성공, 부분 실패, edge case를 포함한 다양한 결과를 내도록 제어된 stochasticity를 도입”한다([[kimi-k2]] §3.1.1). 코딩과 software engineering에서는 시뮬레이션 대신 실제 sandbox를 쓴다.
5. **규칙 기반 완료 판정을 쓰는 gym 사용자(UserRL, 2025-09).** 8개의 gym은 과업 완료 판정을 규칙 기반으로 유지하고 LLM이 사용자 답변을 만들게 한다. 훈련에는 Qwen3-32B를 시뮬레이터로 쓰고 평가에는 GPT-4o를 쓴다([[userrl]] §3, App. C).

**시뮬레이터 선택이 결과를 바꾼다는 근거.** τ²-bench 통신에서 에이전트는 Hard persona 과업보다 Easy persona 과업에서 더 높은 점수를 받는다([[tau2-bench]] Figure 7). UserRL에서 Qwen3-32B 대신 GPT-4o를 상대로 훈련하면 “일반적으로 더 높은 성능”이 나오며, 저자들은 원인의 일부가 훈련과 평가에서 같은 시뮬레이터를 사용했기 때문이라고 본다(해석)([[userrl]] §5.3). UserRL의 소규모 사람 연구에서는 박사과정 학생 다섯 명이 TurtleGym 및 TelepathyGym에서 GPT-4o를 대체했고, Qwen3-8B 점수는 각각 0.1854에서 0.3127, 0.5610에서 0.7805로 올랐다. 사람은 “때때로 미묘한 힌트”를 줬지만 GPT-4o는 “Yes”, “No”, “Maybe”라고 답했기 때문이다([[userrl]] Table 5, App. C). 따라서 시뮬레이터-사람 간 격차의 방향은 과업에 따라 달라진다. 이 연구는 시뮬레이터가 사람보다 덜 유익하다고 찾았지만, [[lost-in-multi-turn]] §9는 자신의 시뮬레이터가 사람보다 더 질서정연하다고 주장한다.

**ch-25 고객 시뮬레이터 질문으로의 연결.** ch-25 `qa.md` Q12는 텔레마케팅 고객을 연기한 LLM이 거의 매번 거절했다고 기록한다(횟수는 기록하지 않았다). §2의 출처들은 이 관찰에 측정 가능한 형태를 부여한다. 한 답변을 높은 확률로 출력하는 시뮬레이터는 [[simulator-collapse]] Def. 3.1의 의미에서 “mode-collapsed”이며, adversarial reward를 받은 공동 훈련 시뮬레이터는 거절률 98%와 평가 reward 0.07에 도달했다([[simulator-collapse]] App. F.8). ch-25의 관찰을 이 정의에 연결하는 것은 이 강의의 해석이며, 어느 논문도 판매 전화를 연구하지 않았다.

**범용 모델에 대한 함의.** 시뮬레이터는 훈련에 어떤 사용자 행동이 나타나는지를 정의한다. 하나의 persona 스타일을 상대로 훈련한 모델은 그 스타일을 서비스하도록 학습할 것으로 예상된다(해석. [[simulator-collapse]] Fig. 3은 하나의 고정된 시뮬레이터를 상대로 한 RL에서 이를 뒷받침한다). 시뮬레이터가 훈련 데이터를 만들면, 시뮬레이터 집단을 과업 혼합처럼 설계하고 측정해야 하는 mixture로 다뤄라.

## §2 시뮬레이터의 실패 양상과 시뮬레이터 검증 방법

**근거가 있는 네 가지 실패 양상.**

| 실패 양상 | 정의 | 근거 |
|---|---|---|
| 지시 이탈 | 시뮬레이션 사용자가 자신의 지시에 있는 제약을 빼먹거나 모순되게 행동한다 | τ²-bench retail에서 주석으로 확인한 오류 20개 중 4개는 제약 누락이고 3개는 조기 종료였다. critical 오류의 대부분은 이 두 유형에서 나왔다([[tau2-bench]] App. E.1). |
| 과도한 순응과 제한된 추론 | 주의 깊은 사용자가 할 점검 없이 사용자가 에이전트 제안을 받아들인다 | τ-bench에서는 “사용자가 기능을 다시 확인하지 않고 에이전트가 추천한 lamp를 승인한다”고 보고한다([[tau-bench]] §6). |
| 질서정연한 과소명세 | 실제 사용자는 따르지 않는 고정되고 협조적인 순서로 시뮬레이터가 정보를 공개한다 | Lost in Multi-Turn은 매 turn 정확히 shard 하나를 공개하고 마지막 turn이 과업을 완료하도록 보장한다. 저자들은 실제 세계의 성능 저하가 더 클 것으로 예상한다(해석)([[lost-in-multi-turn]] §9). |
| 시뮬레이터 붕괴 | 시뮬레이터의 답변 분포가 하나의 mode에 집중되고 정책이 그 mode를 이용하도록 학습한다 | 단일 시뮬레이터 RL에서는 held-out reward가 일찍 정점에 도달한 뒤 하락하고 entropy는 0에 가까워진다([[simulator-collapse]] §3.3, Fig. 3). |

**측정한 오류율.** τ²-bench는 gpt-4.1이 사용자와 에이전트를 모두 연기한 대화를 주석 처리했다. 두 명의 주석자가 각 대화를 네 기준, 즉 시뮬레이터 guideline 준수, 과업 지시 준수, 올바른 user-tool 사용, 자연스러운 대화 지속에 따라 검토했다([[tau2-bench]] §4.3).

| 도메인 | 대화 수 | Critical | Benign | 합계 |
|---|---:|---:|---:|---:|
| airline | 100 | 13% | 34% | 47% |
| retail | 50 | 12% | 28% | 40% |
| telecom | 50 | 6% | 10% | 16% |

Table 2 caption은 telecom에 critical 오류가 없었다고 말하지만, 표는 50개 중 3개를 보고한다. 여기서는 표의 값을 사용한다. 저자들은 prose 지시 대신 tool과 관찰 가능한 state로 사용자를 제약했기 때문에 telecom 오류율이 더 낮다고 본다(해석. retail이나 airline에서 검증하지 않았다, §5). Lost in Multi-Turn은 GPT-4o-mini 시뮬레이터를 검사하여, 검사한 대화의 97.8%가 valid이고 사용자 turn의 96.0%에서 shard가 완전히 공개됐다고 보고한다([[lost-in-multi-turn]] App. D, Table 5). 단, 검사한 대화 수는 문서의 서로 다른 위치에서 “several hundred”, 200, 100으로 인쇄돼 있다.

**계산 예시: critical 오류율이 성공 추정치에 미치는 영향(도출).** 오류 없는 사용자라면 에이전트가 retail 과업의 60%에서 성공하고, 과업 난이도와 독립적으로 대화의 12%에서 critical 시뮬레이터 오류가 난다고 하자. 정의상 critical 오류는 완료를 막으므로 측정 성공률은 최대 $0.60 × (1 − 0.12) = 0.528$이다. SFT 데이터 생성에서도 같은 오류는 원래 성공했을 궤적의 12%를 제거하고, RL에서는 에이전트 행동이 실제로 올바를 수 있었던 rollout에 reward 0을 부여한다. 독립성 가정은 이 강의가 추가한 것이며, τ²-bench는 오류와 난이도의 상관을 보고하지 않는다.

**시뮬레이터 붕괴: 메커니즘.** [[simulator-collapse]](arXiv 2026-08)는 시뮬레이터 $φ$, 에이전트 메시지 $a^π$ 뒤 상태 $s$, 그리고 그 상태에서 가장 가능성 높은 답변 $a⋆$에 대해 다음을 정의한다.

```
ε_φ(s, a^π) = 1 − φ(a⋆ | s, a^π)
‖∇_θ J_φ(θ) − ∇_θ J_mode(θ)‖ ≤ 2 · B · R_max · ε̄_H(θ)
```

- $ε_φ$는 시뮬레이터가 mode 밖에 두는 확률 질량이다. 방문한 turn들에 걸친 기대 $ε_φ$가 최대 $ε⋆$이면 시뮬레이터는 $ε⋆$-collapsed다(Def. 3.1).
- $ε̄_H(θ)$는 rollout의 $H$개 turn에 걸친 $ε_φ$의 기대 합이다. $J_φ$는 실제 시뮬레이터를 쓰는 RL objective이고, $J_mode$는 매 시뮬레이터 turn이 $a⋆$를 내는 경우의 objective다.
- $B$는 궤적 score $Σ_t ∇_θ \log π_θ$의 norm을 상계한다. reward는 $[0, R_max]$에 있다(Thm. 3.2).

$ε̄_H$가 작으면 정책 gradient는 언제나 modal 답변을 하는 사용자에 대한 gradient에 가깝다. 따라서 훈련은 사실상 한 명의 사용자에 맞춰 최적화한다. 저자들은 이 bound를 “tight bound라기보다 bias 방향에 대한 analytic guide”라고 부른다(§3.2).

**계산 예시: 붕괴가 RL signal도 없애는 이유(도출).** group-normalized advantage에서는 모든 reward가 같을 때 $G$개 rollout의 group이 모든 구성원에게 zero advantage를 준다. 각 rollout이 확률 $p$로 독립 성공한다면, $G = 8$ group의 reward가 전부 같은 확률은 $p⁸ + (1 − p)⁸$이다. $p = 0.5$에서는 $2/256 = 0.8%$, $p = 0.1$ 또는 $p = 0.9$에서는 43.0%, $p = 0.95$에서는 66.3%다. 거의 언제나 거절하는 시뮬레이터는 대부분 과업에서 $p$를 0 쪽으로 민다. 측정한 진단값도 이 방향과 맞는다. τ²-bench Retail의 단일 시뮬레이터 RL에서 zero-variance batch는 60%에서 85% 초과로, all-failure batch는 70%로 상승했으며, entropy는 1.9에서 0.4 nat로 하락했다([[simulator-collapse]] App. F.3, Fig. 19).

**수정법의 근거.** τ²-bench의 Qwen3-4B-Instruct에 대한 Retail / Airline held-out 성공률은 다음과 같다. Base는 40.4 / 24.0, 하나의 고정 GPT-5-mini 시뮬레이터를 상대로 한 RL은 46.1 / 29.8(일시적 정점인 최선 checkpoint), Persona-Guided는 49.2 / 31.6, 세 시뮬레이터 계열 앙상블은 57.1 / 40.1, Verbalized Sampling은 55.5 / 36.9, Co-Training은 60.5 / 44.4, Population Co-Training은 62.2 / 45.7이었다([[simulator-collapse]] Table 1). 사전 등록한 사람 연구(각 cell $N = 40$)의 과업 outcome은 Base 0.41, 단일 시뮬레이터 RL 0.43, Verbalized Sampling 0.63, Co-Training 0.70이었다([[simulator-collapse]] Table 3). **결과(단일 연구)** 이며, text-only 영어 two-agent 과업에서의 결과다.

**조건과 한계.** Table 1은 세 훈련 시뮬레이터 계열을 포함한 panel로 선택한 최선 checkpoint를 보고한다. 저자들도 이것이 “partial training-simulator signal을 selection에 leak”한다고 말한다(§4.1). 공동 훈련 시뮬레이터에는 정책 reward 분산을 목표값 근처에 유지하는 reward가 필요하다. adversarial 시뮬레이터 reward는 거절 98%, 평가 reward 0.07을 냈고, cooperative reward는 pushback을 2%로 낮추고 0.17을 냈으며, variance-targeting reward는 0.40을 냈다([[simulator-collapse]] App. F.8).

**이 출처들에서 뽑은 검증 protocol.**

1. 도메인별로 시뮬레이션 대화의 최소한 표본에 critical 및 benign 시뮬레이터 오류를 주석으로 단다([[tau2-bench]] §4.3).
2. CONCAT 방식 control을 실행한다. 전체 지시를 한 turn에 주고, 다중 turn 손실이 재작성 과정에서 사라진 정보 때문이 아닌지 확인한다([[lost-in-multi-turn]] §3.3; CONCAT은 FULL의 평균 95.1%였다, §6.1).
3. 훈련에 쓰지 않은 시뮬레이터 계열과 실제 사용자 모두를 상대로 훈련된 에이전트를 평가한다([[simulator-collapse]] §4.1, Table 3; [[userrl]] Table 5).
4. 평균 성공률만 보고하지 말고 $pass^k$와 unreliability도 보고한다(아래 §4).

**범용 모델에 대한 함의.** 유창성만으로 검증한 시뮬레이터도 하나의 상호작용 pattern을 가르칠 수 있다. 측정해야 할 양은 시뮬레이터 오류율, held-out-simulator gap, human-user gap, 정책 entropy다.

## §3 궤적 생성 전략

**정의.** 궤적 생성 전략은 어떤 모델이, 어떤 환경에서, 어떤 과업으로 행동하는지와 과업 텍스트를 행동 전에 쓰는지 후에 쓰는지를 정한다.

| 전략 | 작동 방식 | 근거(설정, 측정값, 수치) |
|---|---|---|
| 실제 환경에서 강한 teacher rollout | teacher 모델이 실행 가능한 환경에서 행동하고, 성공한 실행만 남긴다 | CoderForge: 51K개 과업에서 Qwen3-Coder-480B가 만든 궤적 258,134개 중 155,144개가 모든 test를 통과했다. Qwen3-32B SFT는 SWE-Bench Verified에서 59.4% pass@1에 도달했다([[together-coderforge-agent-trajectories]]) |
| blueprint 우선, 대화 후생성 | 실행, policy unit test, LLM committee로 ground-truth action을 검증한 뒤 대화를 시뮬레이션하고 state가 일치하는 실행을 남긴다 | APIGen-MT: agentic feedback을 넣은 과업 검증 성공률은 70%, 넣지 않으면 28%였다. xLAM-2-70b-fc-r의 τ-bench pass@1은 56.2%로, Llama 3.1 70B Instruct의 38.2%보다 높았다([[apigen-mt]] Fig. 4, Table 2) |
| 먼저 탐색하고 과업을 정제 | proposer가 실제 site에서 행동하고, refiner가 매 action 뒤에 과업을 다시 쓰며, summarizer가 실행된 action들로 최종 과업을 쓴다 | Explorer: 175K개 궤적 중 94K개를 남겼고 성공 하나당 비용은 $0.28이었다. 83개 과업에서 Qwen2-VL-7B의 Mind2Web-Live full-task 성공률은 14.5에서 19.3으로 올랐다([[explorer]] Table 3, Table 5, App. B) |
| 역방향 과업 합성 | rule로 GUI element를 탐색한 뒤, 관찰한 ⟨pre-state, action, post-state⟩ triplet에서 low-level 및 high-level 지시를 쓴다 | OS-Genesis: 1K개 궤적으로 Qwen2-VL-7B의 AndroidWorld 성공률은 17.41%였고, task-driven data의 6.25%보다 높았다([[os-genesis]] Table 1) |
| reward 없는 early experience | expert state에서 agent 자신의 대안 action을 실행하고, 그 결과 next state나 그 결과에 근거한 rationale로 훈련한다 | WebShop의 Llama-3.2-3B: imitation 41.8, implicit world modeling 60.2, self-reflection 52.7([[agent-early-experience]] Table 1) |
| 중간 step에서 Monte Carlo continuation | expert 궤적의 step $t$에서 branch하고, $N$개 rollout으로 step value를 추정한다 | IPR: step당 continuation $N = 5$개를 사용했다. step reward 순서는 WebShop page-scoring rule과 최대 82%의 pair에서 일치했다([[ipr-step-level-refinement]] §4.1, §5.3) |

**Backtracking을 동반한 search.** 이 장에서 검증한 출처 중 explicit backtracking을 하는 tree search가 생성한 궤적으로 훈련하는 것은 없다. expert prefix에서 branch하는 IPR(§5)이 여기서 다루는 가장 가까운 mechanism이다. 이는 이 장의 근거 기반에서의 **open question**이다.

**조건과 한계.** 각 전략에는 명시된 narrowing risk가 있다. CoderForge는 하나의 scaffold와 tool set으로 모든 데이터를 만들며, 모델이 “서로 다른 scaffolds, tools, prompt templates로 쓸 때 더 나쁘게 수행할 수 있다”고 경고한다([[together-coderforge-agent-trajectories]] Limitations). APIGen-MT는 생성에 사용한 동일한 두 τ-bench domain, API, policy에서 평가하며 overlap check를 보고하지 않는다([[apigen-mt]] §4.3, §5.1). Explorer의 synthetic-only 모델은 Multimodal-Mind2Web에서 평균 step 성공 43.0으로, Mind2Web human data만으로 훈련한 49.5와 둘 다 사용한 54.3보다 낮다([[explorer]] Table 6). OS-Genesis의 향상은 궤적 수가 늘면서 포화한다. 저자들은 이를 model capacity와 GPT-4o의 task completion 능력 탓으로 돌린다([[os-genesis]] §5.3, Fig. 6).

**긴 궤적.** CoderForge 궤적의 median 길이는 36K--42K token이고 P99는 약 100K다. 32B 모델은 sequence parallelism 및 boundary-aware masking을 쓴 packing으로 128K context에서 훈련한다([[together-coderforge-agent-trajectories]]). APIGen-MT 궤적은 29 turn에 이른다([[apigen-mt]] Fig. 4). 이런 궤적의 inference 및 RL context 관리는 ch-45c에서 다룬다.

**범용 모델에 대한 함의.** 전략들은 과업 다양성이 나오는 위치가 다르다. CoderForge와 APIGen-MT는 고정된 과업 pool에서, Explorer와 OS-Genesis는 환경 자체의 state에서, early experience는 agent 자신의 대안 행동에서 다양성을 얻는다. 다양성 mechanism이 다른 source를 섞으면 한 generator의 과업 분포에 대한 의존을 줄일 것으로 예상된다(해석). 다만 이 장의 어떤 출처도 그 mixture를 직접 측정하지는 않는다.

## §4 궤적 검증과 필터링

**정의.** 궤적 검증은 완료된 궤적 또는 개별 step에 성공, 실패, 혹은 등급화된 점수라는 label을 부여하는 일이다. 필터링은 이 label을 사용해 어떤 궤적을 훈련에 넣을지 결정한다.

**네 가지 검증 계층.**

1. **Outcome 검사.** 최종 환경 상태 또는 test 결과를 reference와 비교한다. τ-bench는 최종 database를 유일한 목표 database와 비교하고 필요한 output substring도 검사한다([[tau-bench]] §3). CoderForge는 최종 patch가 repository의 모든 test를 통과한 궤적을 남긴다([[together-coderforge-agent-trajectories]]). 한계가 있다. τ-bench는, 예를 들어 agent가 policy가 요구하는 confirmation 없이 행동한 경우처럼 $r = 1$이 “필요조건일 수는 있어도 충분조건은 아닐 수 있다”고 명시한다([[tau-bench]] §3).
2. **Step 유효성과 policy 검사.** APIGen-MT는 dialogue를 simulation하기 전에 format을 검사하고, candidate action을 실행해 state diff를 기록하며, domain policy를 Python unit test로 변환한다([[apigen-mt]] §4.1.2). GLM-5는 search question을 세 단계로 필터링한다. 그중 verification agent는 “유일하지 않은 답, 일관되지 않은 evidence, 잘못된 label”을 거부한다([[glm-5]] §4.2.3).
3. **Tool-call protocol 검사.** 잘못된 identifier, malformed argument, 반복적으로 실패하는 edit를 포함한 call 수를 센다. τ-retail에서 gpt-4o function calling은 과업당 존재하지 않는 ID를 쓰는 call이 0.46회, gpt-3.5-turbo function calling은 2.08회, gpt-3.5-turbo Act는 6.34회였다([[tau-bench]] §5.2). CoderForge는 초기 student training에서 `str_replace_editor`가 실패한 사례가 “매우 많았다”고 보고한다. 모델이 교체하려는 string을 반복해서 써서 생긴 실패였으며, 대응으로 repetition penalty를 시험했다([[together-coderforge-agent-trajectories]]).
4. **Judge 기반 필터링.** LLM이 과업 또는 rubric에 비추어 궤적의 점수를 매긴다. Kimi K2는 LLM judge가 rubric을 충족한다고 평가한 궤적만 남긴다. rejection rate는 보고하지 않았다([[kimi-k2]] §3.1.1). Explorer의 GPT-4o verifier는 과업, action, screenshot, 최종 page를 보고 성공 label을 매긴다([[explorer]] §3.2).

**계산 예시: judge filter의 precision(Explorer, 도출).** 무작위 궤적 100개에서 share matrix는 다음과 같다. 사람이 성공, judge도 성공으로 판정한 비율은 0.39, 사람이 성공·judge 실패는 0.05, 사람이 실패·judge 성공은 0.14, 사람이 실패·judge 실패는 0.42다([[explorer]] Table 10). Judge는 $0.39 + 0.14 = 0.53$의 궤적을 accept한다. Precision은 $0.39 / 0.53 = 73.6\%$이므로, accept된 궤적 중 26.4%는 사람이 실패로 label한 사례인데도 양성 SFT target이 된다. Recall은 $0.39 / (0.39 + 0.05) = 88.6\%$다. Raw agreement는 $0.39 + 0.42 = 81\%$이며, 논문이 보고한 수치다. 그러나 agreement만 보면 accept 쪽에 error가 집중되어 있다는 점이 가려진다. Reject 쪽에서 judge는 0.47의 궤적을 거절하고, 그중 $0.05 / 0.47 = 10.6\%$는 사람이 성공으로 label한 사례다. 거절 궤적을 DPO loser로 쓴다면 이 false negative들이 음의 gradient를 받는다.

**Judge 편향.** 출처들에는 세 종류의 편향이 나타난다. (a) **같은 judge로 평가하기**: Explorer의 in-domain test는 훈련 data를 필터링한 것과 같은 verifier를 쓴다([[explorer]] §5.1). 따라서 filter와 evaluator가 공유하는 error는 탐지되지 않는다. (b) **Curation 편향**: τ-bench는 gpt-4-turbo agent를 실행해 user instruction을 tuning했고, 저자들은 이를 “implicit bias”라고 부른다([[tau-bench]] §6). (c) **이진 judge 대신 등급 judge 사용하기**: OS-Genesis는 1--5점 GPT-4o score에서 $P(g_i) = R_i / \Sigma_k R_k$의 확률로 궤적을 sample한다. 점수가 5, 3, 1이면 확률은 각각 $5/9$, $3/9$, $1/9$이므로 1점을 받은 궤적도 11% 확률로 뽑힌다([[os-genesis]] Algorithm 1). GPT-4o 점수는 100개 궤적에서 human annotator와 Spearman 0.813(mobile), 0.798(web)의 상관을 보인다([[os-genesis]] Table 6). 일반적인 judge calibration은 ch-49에서 다룬다.

**검증 target으로서의 일관성: $pass^k$.** $n$번 trial 중 $c$번 성공한 과업에서([[tau-bench]] §3):

```
pass^k = E_task[ C(c, k) / C(n, k) ]          (k번 trial이 모두 성공)
pass@k = 1 − E_task[ C(n − c, k) / C(n, k) ]  (k번 중 적어도 하나가 성공)
```

- $C(a, b)$는 이항계수다. Expectation은 과업에 대해 취한다. Trial은 같은 instruction과 database로 독립동일분포(i.i.d.)로 다시 실행한 결과다.

계산 예시로 $n = 8$, $c = 6$을 두자. $pass^2 = C(6,2)/C(8,2) = 15/28 = 0.536$, $pass@2 = 1 − C(2,2)/C(8,2) = 27/28 = 0.964$, $pass^4 = 15/70 = 0.214$다. τ-retail의 gpt-4o function calling에서는 $pass^8$이 25% 미만이다([[tau-bench]] §5.1). Lost in Multi-Turn은 instruction마다 10회 실행한 score의 90th percentile에서 10th percentile을 뺀 값으로 같은 속성을 unreliability $U$로 측정한다. Sharded multi-turn conversation에서는 single-turn 대비 aptitude가 16% 하락하는 동안 $U$가 평균 112% 상승했다([[lost-in-multi-turn]] §6.2). 과업마다 한 번의 성공만 남기는 filter는 8번 중 1번 푼 과업과 8번 모두 푼 과업을 구분하지 못한다. 과업별 $c/n$을 기록하면 이 정보를 보존한다.

**범용 모델에 대한 함의.** Verifier가 모델이 성공으로 배우는 것이 무엇인지 결정한다. Outcome 검사는 이 출처들에서 false-accept rate를 측정하지 않았지만 policy를 위반한 실행도 통과시킬 수 있다([[tau-bench]] §3). Judge는 open task까지 검증 범위를 넓히지만 Explorer에서는 accept된 궤적의 26.4%가 측정된 false accept였다([[explorer]]). RL에서 false accept는 양의 advantage가 된다. 따라서 verifier에는 raw agreement보다 precision이 더 중요하다.

## §5 실패 궤적: 다섯 가지 사용법

실패 궤적은 verifier가 과업을 완료하지 못했다고 label한 궤적이다. 출처들은 실패를 전혀 사용하지 않는 방식부터 gradient로 가장 직접 사용하는 방식까지, 아래의 다섯 방식으로 사용한다. 대화형 그림 [figures/failed-rollout-signal.html](figures/failed-rollout-signal.html)에서는 한 group 안의 성공, 실패, timeout, 환경 crash 수를 설정하고, 각 처리 규칙에서 어느 rollout이 어떤 advantage로 gradient를 받는지 볼 수 있다. 두 번째 panel은 하나의 과업에 대해 $n$과 $c$로부터 $pass^k$와 $pass@k$를 계산한다.

### 5.1 버리기(discard): 음의 한계가치

APIGen-MT는 reward가 1인 궤적만 보관하고, 실패에서 얻는 “additional contrastive signal”은 미래 과제로 적었다([[apigen-mt]] §4.2, §6). CoderForge는 통과한 궤적 155,144개만으로 훈련하고 103K개의 실패를 공개한다. 실패 궤적은 성공 궤적보다 step이 18--28% 더 많으며, 저자들은 모델을 “효율적인 과업 해결 쪽으로 밀기 위해” 버린다고 설명한다(해석)([[together-coderforge-agent-trajectories]]). Kimi K2도 rubric judge에 실패한 궤적을 버린다([[kimi-k2]] §3.1.1). 반면 OS-Genesis는 미완료 궤적을 더 낮은 sampling probability를 갖는 양성 target으로 유지한다(§4).

### 5.2 오류 구간의 loss를 mask하기: 내용으로서의 음성 사례

**정의.** 궤적은 오류 step을 context 안에 남기되, 해당 token에는 loss를 0으로 준다. 이후의 수정은 통상대로 훈련한다. GLM-5는 “궤적 안의 오류 구간은 보존하지만 loss function에서 mask하여, 잘못된 행동을 강화하지 않으면서 모델이 오류 수정 행동을 배우게 한다”고 쓴다([[glm-5]] §3.1).

```
L(θ) = − Σ_t m_t · log π_θ(y_t | y_<t, x),   m_t ∈ {0, 1}
```

- $y_t$는 궤적의 token $t$이고, $x$는 prompt다. 환경 observation, user turn, 오류 구간 내부 token에는 $m_t = 0$, 나머지 assistant token에는 $m_t = 1$이다.

계산 예시를 보자. assistant turn이 각각 400 token인 10 turn 궤적이 있다. Turn 4가 잘못된 tool call을 내고, turn 5가 error message를 읽고, turn 6이 수정된 call을 낸다. Turn 4를 mask하면 4,000개 assistant token 중 400개(10%)가 loss에서 빠진다. Turn 6은 turn 4와 그 error observation을 context로 둔 채 훈련되므로, 모델은 오류를 본 조건에서의 수정을 배운다. 궤적 전체를 버리면 4,000 token 전체가 사라진다. GLM-5는 그 외에는 좋은 궤적 안의 결함 있는 slide page에도 같은 생각을 적용하며, 버리면 “effective data utilization을 줄이고 generation cost를 높인다”고 말한다([[glm-5]] §4.2.5).

**근거와 한계.** GLM-5는 masking 대 discarding ablation, mask된 token 비율, 오류 구간을 탐지하는 방법을 보고하지 않았다([[glm-5]]). **단일 연구 결과이며, 효과가 분리되어 측정되지 않았다.**

### 5.3 실패 조건부 훈련: 조건으로서의 음성 사례

**정의.** 실패 궤적은 틀렸음을 표시하는 instruction 아래에서 보통의 cross-entropy로 훈련한다. Inference에서는 “correct” instruction만 사용한다. NAT는 음성 사례에 “Please generate a solution that **incorrectly** answers the question.”을, 양성 사례에 그 “correctly” 변형을 덧붙인다([[learning-from-failure-nat]] §3.3).

**근거.** LLaMA-2-7B-Chat, GPT-3.5 ReAct 궤적, calculator를 쓰는 수학 과업, 양성 2k개와 음성 10k개에서 평균 accuracy는 55.90(양성만), 63.39(marker 없이 음성을 추가한 NUT), 64.64(NAT)였다([[learning-from-failure-nat]] Table 2). 양성 5k개에서는 각각 64.17, 64.95, 67.42였다. 13B에서 양성 5k개를 썼을 때 marker 없는 음성은 accuracy를 70.76에서 69.10으로 낮췄지만, NAT는 71.28이었다. Marker text를 “A”/“B”나 무작위 string으로 바꾸면 63.15--64.04였고 Correct/Incorrect prefix의 63.55와 한 점 이내였다. 따라서 저자들은 단어 자체가 아니라 두 group을 분리하는 일이 중요하다고 결론 내린다(Table 7). 더 약한 fine-tuned LLaMA-2-7B가 만든 음성은 accuracy를 올리는 대신 3.16점과 6.20점 낮췄다(Table 5). **단일 연구 결과**이며 ground-truth answer가 있는 과업만 다뤘다.

**Mechanism(해석).** Marker 없는 음성은 잘못된 action의 likelihood를 올린다. NUT의 GSM8K action error는 10.47%였고, 양성만 쓴 경우는 3.58%, NAT는 7.90%였다(Table 6). Marker는 그 likelihood 대부분을 inference에서 쓰지 않는 condition으로 옮긴다.

### 5.4 성공 대 실패 preference pair: gradient로서의 음성 사례

**정의.** ETO(arXiv 2024-03)는 같은 과업에 대해 SFT agent의 더 낮은 reward 궤적과 expert 궤적을 pair로 만들고 DPO로 훈련한 뒤, 업데이트된 agent로 exploration을 반복한다([[eto-trial-and-error]] §3).

```
L_DPO = − E log σ( β log πθ(e_w | u)/πref(e_w | u) − β log πθ(e_l | u)/πref(e_l | u) )
```

- $u$는 과업 instruction이고, $e_w$, $e_l$은 각각 더 높은 reward와 더 낮은 reward의 궤적이다. $πref$는 iteration 시작 시점의 policy, $β$는 constraint weight, $σ$는 sigmoid다.

**근거.** Llama-2-7B에서 평균 reward(WebShop / ScienceWorld seen / unseen / ALFWorld seen / unseen)는 SFT가 63.1 / 67.4 / 53.0 / 60.0 / 67.2, RFT(성공 사례를 SFT에 추가)가 63.6 / 71.6 / 54.3 / 62.9 / 66.4, ETO가 67.4 / 73.8 / 65.0 / 68.6 / 72.4였다([[eto-trial-and-error]] Table 2). 그러나 ETO 대 RFT 비교는 loss와 선택된 sample을 모두 바꾸므로, 거절 항이 이득에 기여한 비율은 측정하지 않았다.

**실패 양상.** Final reward로 만든 step-level pair는 WebShop reward를 LR $1e-6$, $β = 0.1$에서 8.3까지 낮췄다(Table 4). Behavioral-cloning stage 없이 ETO를 쓰면 12.5였고, untuned model의 17.9보다도 낮았다(Table 5). WebShop과 ScienceWorld에서는 세 번째 iteration 이후, ALFWorld에서는 첫 번째 iteration 이후 성능이 하락했다(§4.3). 다른 연구에서는 expert action을 chosen, agent 자신의 대안 action을 rejected로 한 DPO가 WebShop과 ALFWorld에서 “수십 optimization step 안에 collapse”했다. Collapse 전 최고 checkpoint(53.1, 82.8)도, 같은 대안 action을 content로 쓰는 implicit world modeling(58.6, 85.9)보다 낮았다([[agent-early-experience]] §6.3, Table 3).

### 5.5 Step-level refinement: 국소화된 gradient로서의 음성 사례

**정의.** IPR(arXiv 2024-06)은 실패를 하나의 step으로 국소화한다. Agent는 expert 궤적의 처음 $t − 1$ step에서 계속 진행한다. Frozen SFT scorer의 Monte Carlo rollout은 step $t$에서 expert action과 agent action의 value를 추정한다([[ipr-step-level-refinement]] §3.2-3.3).

```
r_s(s_t, a_t) ≈ (1/N) Σ_{i=1..N} r_o(u, e^(i))
L = L_o-DPO + L_s-DPO + L_SFT
```

- $r_o(u, e^{(i)})$는 step $t$에서 시작한 rollout $i$의 outcome reward이고, $N$은 rollout 수다.
- $L_{o-DPO}$는 whole trajectory의 DPO이고, $L_{s-DPO}$는 공유 prefix $e_{t−1}$을 조건으로 한 두 continuation의 DPO이며, $L_{SFT}$는 winning trajectory의 negative log-likelihood다.

계산 예시에서 $N = 5$라고 하자. Expert action 뒤 rollout의 reward가 1, 0, 1, 1, 0이면 $r_s = 0.6$이다. Agent action 뒤 rollout이 0, 0, 1, 0, 0이면 $r_s = 0.2$다. 차이 0.4는 WebShop($τ = 0.01$)과 InterCodeSQL($τ = 0.1$)에서 쓴 threshold는 넘지만 ALFWorld($τ = 0.5$)에서는 넘지 않는다. Agent 자신의 continuation도 더 낮은 outcome reward를 가지면, expert continuation을 chosen으로 하는 step pair를 만든다.

**근거.** Llama-2-7B 평균 reward(WebShop / InterCodeSQL / ALFWorld seen / unseen)는 ETO가 67.4 / 57.2 / 68.6 / 72.4, IPR가 71.3 / 61.3 / 70.3 / 74.7(best iteration)였다([[ipr-step-level-refinement]] Table 2). WebShop / InterCodeSQL / ALFWorld unseen에서의 ablation은 outcome DPO를 빼면 70.2 / 59.3 / 72.4, step DPO를 빼면 66.4 / 58.0 / 70.2, SFT term을 빼면 61.8 / 31.7 / 64.9였다(Table 4). 양성 likelihood term을 제거했을 때 손실이 가장 컸다. 이는 rejected-sample gradient 옆에 anchor를 남겨야 한다는 근거가 된다. Training time은 ETO의 “세 배 미만”이었다(App. C). **단일 연구 결과**이며, main table은 하나의 base model family만 쓴다.

**범용 모델에 대한 함의.** 다섯 사용법은 실패 action의 likelihood에 어떤 일이 생기는지에 따라 다르다. Discard에서는 변하지 않고, mask에서는 변하지 않지만 주변 context로 훈련되며, NAT에서는 inference에 쓰지 않는 condition으로 옮겨진다. ETO와 IPR에서만 다른 output 쪽으로 probability mass를 밀어낼 수 있다. 이들을 쓰는 출처는 anchor term(IPR의 $L_{SFT}$)을 추가하고, iteration 수를 제한하며(ETO, IPR), judge가 아니라 environment reward로 실패를 label한다.

## §6 잘린·시간 초과·환경 오류 rollout: mask할 것인가 penalize할 것인가

**정의.** Context limit, step limit, wall-clock timeout, 또는 environment crash 때문에 끝난 rollout에는 검증된 outcome이 없다. **Penalize**는 이 rollout에 실패 reward를 부여한다. **Mask**는 이를 loss에서 제외하고(group method에서는 group 통계에서도 제외한다).

**문제.** Penalize된 truncated rollout은 그 action이 맞았을 수도 있는데 negative advantage를 받는다. Mask된 rollout은 signal을 전혀 주지 않으며, masking이 잦으면 group의 크기가 줄어든다.

**수치를 사용한 mechanism(도출).** Rollout 8개로 된 group을 보자. 성공 2개, 실제 실패 4개, timeout 2개다. Group z-score advantage는 $A_i = (r_i − mean)/std$다.

1. Timeout을 0으로 penalize하면 mean = 0.25, std = 0.433이다. 성공은 +1.732를 받고, timeout을 포함한 나머지 6개 rollout은 모두 −0.577을 받는다.
2. Timeout을 mask하면 6개 rollout이 남고, mean = 0.333, std = 0.471이다. 성공은 +1.414, 실패는 −0.707을 받으며, timeout에는 gradient가 없다.
3. Std normalization이 없는 leave-one-out baseline(DeepSWE 형식, $A_i = r_i −$ 다른 reward들의 mean)에서는 penalize가 +0.857과 −0.286을, mask가 +0.800과 −0.400을 준다.

그림은 이 값을 재현하고 count를 바꿀 수 있게 한다.

**출처가 실제로 하는 일.** DeepSWE의 compact filtering은 maximum context, maximum environment step, 또는 20분 generation timeout에 도달한 궤적을 mask한다. 저자들은 두 이유를 든다. Agent가 처음 10 step 안에 올바른 fix를 만들고 그 뒤 관련 없는 file을 수정하는 식으로 우연히 모든 test를 pass할 수 있으며, 이런 궤적에 reward를 주었더니 curve로만 제시된 Qwen3-14B ablation에서 collapse가 일어났다. 또 environment step 수는 늘어나는데 step당 response length는 줄어든다([[deepswe]] §2.3, Figures 6-7). Released training script는 `agent.overlong_filter=True`와 `algorithm.mask_truncated_samples=False`를 설정하며, timeout도 blog와 다르다([[deepswe-recipe]]). GLM-5는 각 agentic RL sample의 failure reason을 기록하고 environment collapse 때문에 실패한 sample을 제외한다. Group에서 valid sample이 절반을 넘게 남으면 valid sample을 반복해 group을 padding하고, 그렇지 않으면 group을 버린다([[glm-5]] §4.1.2). 어느 valid sample을 반복하는지는 보고하지 않았고, 그 선택은 group mean을 바꾼다. 위 예에서 성공 2개를 반복하면 mean은 0.5가 되고 advantage는 ±1.0이 된다(도출). Simulated user와 하는 multi-turn RL에서는 Eq. 2의 [[simulator-collapse]]에 따라 all-failure group이 zero advantage를 내므로, environment나 simulator가 rollout 대부분을 실패시키면 명시적 mask 없이도 effective batch가 줄어든다.

**조건과 한계.** 이 장의 어느 출처도 timeout을 mask하는 것과 penalize하는 것을 통제 비교한 수치를 보고하지 않았다. DeepSWE의 근거는 curve이고 GLM-5의 근거는 명시된 설계 선택이다. Overlong rollout을 전부 mask하면 overlong behavior에 대한 어떤 압력도 없어진다. Kimi K2는 response length에 대해 반대 선택을 한다. Task type별로 정한 maximum token budget을 넘는 response를 truncate하고 penalty를 준다([[kimi-k2]] §3.2.3). **열린 질문**은 어떤 timeout 원인이 policy-caused(루프, 중복 탐색)여서 penalize해야 하고, 어떤 원인이 environment-caused여서 mask해야 하는가다.

**범용 모델에 대한 함의.** RL의 failure label은 policy failure와 infrastructure failure를 분리해야 한다. 모든 rollout마다 finish reason을 기록하는 것(CoderForge의 released record에는 `finish_reason`과 `reward` field가 있다([[together-coderforge-agent-trajectories]]))이 어느 선택을 하든 전제 조건이다. Multi-turn RL의 처리는 ch-45b에서 계속된다.

## §7 공개 agent SFT 세트와 teacher ablation을 distillation 근거로 읽기

**정의.** Agent distillation은 더 강한 teacher가 환경에서 행동해 만든 trajectory로 student를 훈련하는 방법이다. 여기서 가장 큰 두 변수는 **teacher를 무엇으로 선택하는가**와 **task를 어느 source에서 가져오는가**다. Teacher의 일반 benchmark 순위가 높다는 사실만으로 student에게 좋은 training trace를 만든다고 결론낼 수는 없다.

**근거: teacher의 benchmark 강도는 student의 성능 향상을 예측하지 못했다.** OpenThoughts-Agent는 하나의 task source(Freelancer)에서 서로 다른 teacher가 생성한 trace로 Qwen3-8B student를 훈련하고, 70개 task development set에서 평가했다. GLM-4.6 trace를 쓴 student는 17.2, GPT5-Nano는 8.2, GPT5는 8.0, GPT5-Mini는 7.6이었다([[openthoughts-agent-glm46-teacher]] teacher chart). 저자들은 GPT5를 “TerminalBench 자체에서 가장 좋은 model”이라고 설명한다. 즉, 이 비교에서는 TerminalBench에서의 teacher 순위가 8B student가 배우는 정도를 예측하지 못했다. **단일 연구 결과**이며, 하나의 task source와 하나의 student size만 사용했고 run 수는 밝히지 않았다.

**근거: 어떤 task source는 student를 시작점보다 낮춘다.** GPT-5-Nano source ablation에서 시작 Qwen3-8B의 development-set 점수는 5.7이었다. 그러나 Mind2Web, Synatra, CodeActInstruct trajectory로 훈련한 student는 각각 5.3, 4.9, 3.5였다([[openthoughts-agent-glm46-teacher]] source chart). 즉 이 세 source는 source 전체 수준에서 **음의 한계가치(negative marginal value)**를 가진 trajectory 묶음이다. Positive target으로 넣었지만, 그 source를 추가한 결과 student가 base checkpoint보다 낮아졌다.

**근거: test verification을 거친 규모 확장 사례.** CoderForge는 Qwen3-Coder-480B를 teacher로 하여, test를 통과한 trajectory 155,144개로 Qwen3-32B를 distill했다. SWE-Bench Verified에서 pass@1 59.4%를 보고했다. Dataset card는 변화를 `23.0% → 59.4%`로 표기하지만, blog 본문은 “base model보다 23.0% 높다”고 쓴다([[together-coderforge-agent-trajectories]]). SWE-Bench Verified와 repository 및 base commit 또는 problem statement를 공유하는 task는 제외했다. 같은 data로 훈련한 Qwen3-4B는 epoch 5에서 43.0%를 기록했다. APIGen-MT는 gpt-4o와 DeepSeek V3 trajectory를 xLAM-2 model로 distill했고, xLAM-2-8b-fc-r는 τ-bench에서 46.7로 Llama 3.1 70B Instruct의 38.2를 넘었다([[apigen-mt]] §5.2, Table 2). UserRL은 GPT-4o trajectory 1k개를 SFT cold start로 distill한 뒤 RL을 수행하며, 이 cold start가 없으면 RL이 이른 시점에 plateau에 도달한다고 보고한다([[userrl]] §5.3).

**조건과 한계.** OpenThoughts-Agent는 공개한 SFT trace를 verifier success로 filter했는지 밝히지 않았다([[openthoughts-agent-glm46-teacher]]). CoderForge의 평가는 주로 SWE-Bench Verified에 집중되어 있으며, scaffold 의존성, bug-fix 범위, user interaction 부재를 한계로 든다. 실험실마다 distillation 단계를 training pipeline의 어느 위치에 두는지는 ch-35에서 다룬다.

**범용 모델에 대한 함의.** Agent SFT용 teacher를 고를 때는 task source를 고정한 상태에서 held-out task의 **student 점수**로 teacher를 비교해야 한다. [[openthoughts-agent-glm46-teacher]]에서는 teacher 자신의 benchmark 순위가 student 점수를 예측하지 못했기 때문이다. 새 task source를 추가할 때도 그 source로 훈련한 student가 starting checkpoint 아래로 떨어지는지 확인해야 한다. 같은 연구에서 세 source가 실제로 그렇게 만들었다.

## 음성 샘플과 음성 피드백

이 절에서 “음성(negative)”은 네 뜻으로 쓴다. (1) **음의 한계가치**: positive target으로 썼을 때 성능을 낮추므로 버리는 sample. (2) **content로서의 음성**: failure를 input 또는 수정된 target 안에 넣고 보통의 cross-entropy로 훈련하는 것. (3) **conditioning으로서의 음성**: inference에서는 쓰지 않을 `incorrectly` 같은 marker 아래 failure를 훈련하는 것. (4) **gradient로서의 음성**: DPO의 rejected term이나 음의 advantage처럼 sample likelihood를 명시적으로 낮추는 것. 이 중 (4)만 그 sample에서 probability mass를 제거한다.

**1. 음성 사례의 출처와 label 방식.** Outcome check에는 database state match([[tau-bench]]), repository test([[together-coderforge-agent-trajectories]]), [0, 1] 범위의 final reward([[eto-trial-and-error]])가 있다. Step value는 Monte Carlo continuation에서 온다([[ipr-step-level-refinement]]). Judge에는 rubric judge([[kimi-k2]]), accepted trajectory 가운데 false accept 비율이 측정된 trajectory verifier(26.4%, [[explorer]] Table 10 및 §4의 도출), 1--5점 reward model([[os-genesis]])이 있다. Simulated user는 label noise를 더한다. τ²-bench 대화의 6--13%에는 critical simulator error가 있다([[tau2-bench]] Table 2). Infrastructure는 timeout, context/step limit, sandbox crash처럼 label 없는 종료를 만든다([[deepswe]]; [[glm-5]]).

**2. 현재 관행이 이를 쓰는 방식.**

| 출처 | 사용법 | 위 정의의 유형 |
|---|---|---|
| APIGen-MT, CoderForge, Kimi K2 | 실패 trajectory를 버림 | 1. 음의 한계가치 |
| OpenThoughts-Agent | 세 task source가 student를 base 아래로 낮춤 | 1. 음의 한계가치(source 수준) |
| OS-Genesis | incomplete trajectory를 더 낮은 sampling probability로 target에 보존 | 음성 signal이 아니라 down-weight된 positive |
| GLM-5 SFT | error segment는 context에 보존하고 loss는 mask | 2. content로서의 음성 |
| Early experience(IWM, SR) | non-expert action 결과를 target 또는 rationale input으로 훈련 | 2. content로서의 음성 |
| NAT | failure를 `incorrectly` marker 아래 훈련 | 3. conditioning으로서의 음성 |
| ETO, IPR, early-experience DPO baseline | DPO의 rejected 측 | 4. gradient로서의 음성 |
| GRPO 형식 agent RL([[simulator-collapse]], [[deepswe]]) | 평균 아래 rollout에는 음의 advantage, truncated/crashed rollout은 mask | 4. gradient로서의 음성; mask된 rollout은 signal 없음 |

**3. Mechanism.** Logit $z$에 대한 softmax $p = \operatorname{softmax}(z)$에서 $∂\log p_y / ∂z_j = 1[j=y] - p_j$다. Rejected token $c$의 $\log p_c$를 낮추는 step은 각 logit을 $-η(1[j=c]-p_j)$만큼 옮긴다. 따라서 rejected logit은 $η(1-p_c)$만큼 내려가고 나머지는 각각 $ηp_j$만큼 오른다. 없어진 mass는 현재 probability에 비례해 대안으로 간다. 예를 들어 $p=(A\ 0.6,B\ 0.3,C\ 0.1)$이고 $η=1$로 C를 reject하면 logit 변화는 $(+0.6,+0.3,-0.9)$, 새 확률은 $(0.710,0.263,0.026)$이다. A는 0.110을 얻지만 B는 0.037을 잃는다. 이미 희귀한 sample을 누르면 가장 가능성 큰 대안에 mass가 집중되며, 그 대안도 틀릴 수 있다. Multi-token response에서는 chosen/rejected response의 hidden representation이 유사할 때 preferred response도 likelihood를 잃을 수 있음을 [[likelihood-displacement]]가 보인다(Theorem 6, CHES score). 같은 task의 agent trajectory는 긴 prefix와 tool-call format을 공유하므로 이 의미에서 유사하다(해석). WebShop에서 LR $1e-6$일 때 step-level ETO pair가 8.3을 낸 일과 early-experience DPO collapse는 이 우려와 부합하지만, 두 출처 모두 chosen-trajectory likelihood를 측정하지 않았다.

**4. 수치를 둔 근거.** 이득은 ScienceWorld unseen에서 ETO가 SFT의 53.0을 65.0으로 올린 것([[eto-trial-and-error]] Table 2), InterCodeSQL에서 IPR가 ETO의 57.2를 61.3으로 올린 것([[ipr-step-level-refinement]] Table 2), NAT가 positives-only의 55.90을 64.64로 올린 것([[learning-from-failure-nat]] Table 2)이다. 실패 양상은 WebShop의 step-level DPO pair 8.3([[eto-trial-and-error]] Table 4), non-expert action DPO의 수십 step 내 collapse([[agent-early-experience]] §6.3), weak-model negative의 −3.16 및 −6.20점([[learning-from-failure-nat]] Table 5), ETO에서 1--4 round를 넘기고 IPR에서 4 round를 넘길 때 reward 하락이다(ALFWorld는 1회 뒤, WebShop/ScienceWorld는 3회 뒤).

**효과 크기.** 이 장의 어느 출처도 gain 중 rejected term 하나가 차지하는 몫을 재지 않았다. IPR ablation은 positive SFT term을 빼는 손실이 어느 DPO term 하나를 뺄 때보다 크다고 보인다(Table 4). 음성이 이득의 주된 원인이라는 주장은 여기의 어떤 setting에서도 뒷받침되지 않는다.

**5. Control.** Failure를 국소화한다(IPR의 threshold $τ$ step pair). Positive NLL term을 둔다(IPR의 $L_{SFT}$). Iteration을 제한한다(ETO 1--2회, IPR 최대 4회). Positive와 같은 generator에서 negative를 취한다(NAT: GPT-3.5 negative는 accuracy를 올렸지만 fine-tuned LLaMA-2-7B negative는 3.16, 6.20점 낮췄다). 이 장의 출처는 SFT를 위해 current policy에서 sample한 negative를 시험하지 않았다. Unmarked negative 대신 marker를 쓴다(NAT 대 NUT). Policy 탓이 아닌 segment와 rollout은 mask한다(GLM-5, DeepSWE). Verifier precision이 측정되지 않았다면 judge-labeled failure를 gradient term에 넣지 않는다.

**6. 진단.** Trajectory DPO에서는 chosen/rejected log-probability를 따로 기록한다(ETO는 보고하지 않았다). Agent RL에서는 zero-variance 및 all-failure group의 비율, entropy, 부호별 advantage 통계를 기록한다([[simulator-collapse]] App. F.3). Rollout별 finish reason을 기록한다. Task별 pass@k와 pass^k, action error rate([[learning-from-failure-nat]] Table 6), invalid tool-call count([[tau-bench]] §5.2)를 보고한다.

**7. 일반성에 미치는 효과.** ETO의 negatives-as-gradient 이득은 ScienceWorld unseen variation과 ALFWorld unseen split에서도 유지됐지만([[eto-trial-and-error]] Table 2), 각 ETO agent는 한 task용으로 훈련됐고 multi-task transfer는 시험하지 않았다(한계). 긴 failed trajectory를 버리면 data가 짧은 solution 쪽으로 치우며, CoderForge는 이를 의도한다. 이 출처들 중 failure training 뒤 calibration, over-refusal, 큰 $k$에서의 pass@k를 재는 것은 없다.

## 레시피

모든 행은 2026-09-14 또는 2026-09-15에 링크된 card 또는 chapter excerpt로 확인했다.

| 모델(정확한 release) | 크기 | 단계 | Setting | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---:|---|---|---|---|---|---|
| τ-bench evaluation(gpt-4o agent) | — | eval-gate | user simulator; temperature; action cap; trial | gpt-4-0613; agent 0.0, user 1.0; agent action 30회; task당 ≥3 trial | arXiv:2406.12045v1 §3, §5([[tau-bench]]) | 2026-09-15 verified | ablation 미보고 |
| τ²-bench evaluation | — | eval-gate | user simulator; run; temperature | gpt-4.1-2025-04-14; task당 4 run; 0 | arXiv:2506.07982v1 §4.1([[tau2-bench]]) | 2026-09-15 verified | ablation 미보고 |
| xLAM-2-fc-r series | 1B--70B | SFT | trajectory filter; attempt; user; epoch; masking | $r=1$만 유지; task당 최대 3 attempt; BoN user $N=4$; 최대 3 epoch; assistant token에 loss | arXiv:2504.03601v4 §4.2, §5.1([[apigen-mt]]) | 2026-09-14 verified | Table 3: BoN user가 gpt-4o variance 11.1→2.6 |
| Qwen3-8B / Qwen3-4B(UserRL Equalized/R2G) | 4B, 8B | distill-SFT | data; LR; epoch; schedule; cutoff | GPT-4o agent-and-user trajectory 1k개(gym별 top-K); $1.0e-5$; 3; cosine, warmup 0.1; 16,384 | arXiv:2509.19736v1 App. B Table 6([[userrl]]) | 2026-09-15 verified | Figure 2 left: SFT cold start 대 direct RL |
| Qwen3-8B / Qwen3-4B(UserRL) | 4B, 8B | RL | simulator; LR; batch; rollout; turn; KL; $γ$; epoch | Qwen3-32B(train), GPT-4o(eval); $1e-6$; 128; 8; 16; off; 0.8; 15 | arXiv:2509.19736v1 §5.1, Table 6 | 2026-09-15 verified | Table 3: Equalized/R2G best average; Figure 2 right: GPT-4o train simulator |
| Qwen3-4B-Instruct-2507, Qwen3-8B(τ²-bench) | 4B, 8B | RL | step; prompt × G; LR; KL coef; clip; temperature | 250; $16×8$; $1×10^{-6}$ constant; 0.005; 0.2 / 0.28; 0.7 | arXiv:2608.12253v2 App. C.4 Table 6([[simulator-collapse-recipe]]) | 2026-09-14 verified | Ensemble training 50 step에서 LR tune |
| Population Co-Training | 4B, 8B | RL | simulator pool; simulator reward | $K=5$ checkpoint, 4 step마다 하나; $\exp(-(σ_π^2-0.25)^2/0.02)$ | Table 5; App. F.1, F.8([[simulator-collapse-recipe]]) | 2026-09-14 verified | Fig. 17: K=5 > K=10 > K=3 > K=1; App. F.8 eval 0.40 대 0.07, 0.17 |
| Llama-3.2-3B / Qwen-2.5-7B / Llama-3.1-8B, Tau-Bench retail | 3B--8B | SFT | method별 epoch/LR; batch | IL 6, $1e-5$; IWM 1, $5e-6$; SR 6, $1e-5$; 16 | arXiv:2510.08558v3 App. C.4([[agent-early-experience-recipe]]) | 2026-09-14 verified | ablation 미보고 |
| ETO on Llama-2-7B-Chat | 7B | SFT | batch; LR; warmup; schedule; epoch | 64; $1e-5$; 3%; cosine; 3 | arXiv:2403.02502v2 §4.1([[eto-trial-and-error]]) | 2026-09-14 verified | ablation 미보고 |
| ETO on Llama-2-7B-Chat | 7B | preference | DPO LR; batch; epoch; $β$; iteration | $1e-6$; 32; 3; 0.1(WebShop, ScienceWorld), 0.5(ALFWorld); 2, 2, 1 | arXiv:2403.02502v2 §4.1 | 2026-09-14 verified | §4.3 Fig. 4: WebShop/ScienceWorld 3회 후, ALFWorld 1회 후 하락 |
| IPR on Llama-2-7B | 7B | preference | epoch; batch; MC sample; $τ$; iteration cap | 3; 48; temperature 1에서 $N=5$; 0.01(WebShop), 0.1(InterCodeSQL), 0.5(ALFWorld); 4 | arXiv:2406.11176v2 §4.1([[ipr-step-level-refinement]]) | 2026-09-15 verified | Table 4: WebShop iteration 4 최고 |
| IPR on Llama-2-7B | 7B | preference | LR; $β$ | $1e-5$--$5e-5$ search; 0.1--0.5 search(selected 값 미보고) | arXiv:2406.11176v2 §4.1 | selected value 미보고(§4.1, App. A--E 확인) | — |
| NAT on LLaMA-2-Chat | 7B, 13B | SFT | epoch; batch; max LR; schedule; negative | 2; 64; $5×10^{-5}$; cosine, warm-up 3%; 10k(math main run) | arXiv:2402.11651v2 §4.1, App. B([[learning-from-failure-nat]]) | 2026-09-15 verified | Fig. 2: negative 약 11k에서 gain plateau |
| GLM-5 | 총 744B / active 40B | SFT | agent trajectory의 erroneous segment | context에는 보존, loss에서 mask; 비율 미보고 | arXiv:2602.15763v2 §3.1([[glm-5]]) | 2026-09-15 verified | ablation 미보고 |
| GLM-5 | 744B / 40B | RL | environment-collapse sample; incomplete group | 제외; valid가 절반 초과면 valid sample 반복으로 pad, 아니면 group drop | arXiv:2602.15763v2 §4.1.2 | 2026-09-15 verified | ablation 미보고 |
| DeepSWE-Preview | 32B | RL | masked trajectory(blog) | max context, max environment step, 20분 generation timeout | together.ai/blog/deepswe §2.3([[deepswe]]) | conflict(blog가 final run 설명) | Blog Figure 6(Qwen3-14B): curve만 있음 |
| DeepSWE-Preview | 32B | RL | masked trajectory(released script) | `agent.overlong_filter=True`; `algorithm.mask_truncated_samples=False`; `agent.trajectory_timeout=5400` | rllm@709dec43b740 scripts/agent/swe/deepswe_32b.sh L55, L70, L71([[deepswe-recipe]]) | conflict(script가 final run 재현임을 repository가 밝히지 않음) | ablation 미보고 |
| OpenThinker-Agent-v1-SFT | 8B | distill-SFT | teacher; trace; max turn; LR; epoch | GLM-4.6(AWQ); 약 15,200; 32; $4e-05$; 7.0 | HF dataset/model card([[openthoughts-agent-glm46-teacher]]) | 2026-09-14 verified | teacher chart: GLM-4.6 17.2 대 GPT5 8.0 |
| CoderForge-Preview-32B | 32B | distill-SFT | teacher; sampling; attempt; filter; data; context; LR | Qwen3-Coder-480B; T 0.7, top_p 0.8; task당 8 / 8 / 4; all test pass; 258,134개 중 155,144개; 128K; $1e-5$ cosine | Together AI blog 2026-02-25([[together-coderforge-agent-trajectories]]) | 2026-09-15 verified | R2E-Gym solve rate 62.9%→80.3%(8 attempt)가 multiple attempt 동기; LR ablation 없음 |

**작은 범용 run의 시작점.** Multi-turn user-facing agent로 훈련할 4B--8B instruct model이라면, 검증된 행에서 다음 구성을 얻는다. UserRL setting(5 training gym, 4 H200 GPU)의 teacher trajectory 1k개를 LR $1.0e-5$, cosine schedule, warmup ratio 0.1, 3 epoch으로 SFT cold start한다. 이어 constant LR $1×10^{-6}$, 16 prompt × 8 rollout, clip 0.2 / 0.28의 RL을 하고, frozen simulator 하나 대신 5개 최근 simulator checkpoint pool 또는 simulator family ensemble을 쓴다([[simulator-collapse-recipe]]; τ²-bench, P4G, CooperBench, 8×H100). Outcome label이 달린 failed trajectory와 같은 task의 expert trajectory가 있으면 preference stage의 검증된 참조는 ETO다. SFT 뒤 LR $1e-6$, $β=0.1$로 DPO 1--2회 한다(Llama-2-7B-Chat, single-task agent, 8×A100). ETO는 이 단계 뒤 RL을 하지 않았으므로 RL과의 상호작용은 검증되지 않았다. GLM-5처럼 environment-crash rollout은 group에서 빼고, timeout을 mask할지 정하기 전 finish reason을 기록한다.

## 일반화 관점

**(a) breadth를 넓히는 것.**

- Simulator diversity: ensemble, verbalized sampling, co-trained simulator population은 frozen simulator 하나보다 held-out-simulator와 human-user success를 높였다([[simulator-collapse]] Tables 1, 3).
- Agent 자신의 alternative action을 content로 쓰기: implicit world modeling은 Llama-3.1-8B의 ALFWorld out-of-domain success를 63.3에서 78.1로 높였다([[agent-early-experience]] Table 2).
- Success와 failure를 pair로 만들기: ETO는 ScienceWorld unseen reward를 53.0에서 65.0으로, ALFWorld unseen을 67.2에서 72.4로 높였고 RFT와 PPO는 ALFWorld unseen에서 SFT 아래로 떨어졌다([[eto-trial-and-error]] Table 2).
- 실행된 행동에서 task description을 만들기: OS-Genesis instruction/trajectory는 비교한 synthetic set 중 Sentence-BERT cosine distance 기준 diversity가 가장 높다([[os-genesis]] §5.1).

**(b) narrowing 또는 forgetting을 만드는 것.**

- Frozen simulator 하나: 초기 peak 뒤 held-out reward가 내려가고 entropy가 떨어진다([[simulator-collapse]] Fig. 3).
- Scaffold 하나, task type 하나: CoderForge는 scaffold dependence와 bug-fix scope를 한계로 든다([[together-coderforge-agent-trajectories]] Limitations).
- Overlap check 없이 generation/evaluation에 같은 domain 사용: τ-bench 위 APIGen-MT([[apigen-mt]] §4.3).
- 고정 task set에서 preference training 반복: ETO와 IPR은 1--4 iteration 뒤 하락한다([[eto-trial-and-error]] §4.3; [[ipr-step-level-refinement]] Table 4).
- Weak/mismatched source: 세 task source는 OpenThoughts-Agent student를 base 아래로 낮췄고, weak-model negative는 NAT accuracy를 낮췄다([[openthoughts-agent-glm46-teacher]]; [[learning-from-failure-nat]] Table 5).
- 이 장의 출처는 agent trajectory training 전후 general-capability benchmark(knowledge, math, instruction following)를 보고하지 않는다. 여기서는 forgetting이 측정되지 않았다.

**(c) 이 단계에서 측정하는 법.**

- Mean pass@1뿐 아니라 pass^k와 instruction별 unreliability([[tau-bench]] §3; [[lost-in-multi-turn]] §4.2).
- Training에 쓰지 않은 held-out simulator family와 real user. Checkpoint selection도 disjoint simulator에서 한다([[simulator-collapse]] §4.1은 이를 지키지 않을 때 leakage를 명시한다).
- Held-out environment: UserRL은 IntentionGym, TelepathyGym, SearchGym을 reserve하고([[userrl]] §5.1), ETO/IPR은 unseen split을 보고한다.
- Domain별 simulator error annotation([[tau2-bench]] Table 2)과 verifier confusion matrix([[explorer]] Table 10).
- Repository와 commit([[together-coderforge-agent-trajectories]]) 또는 repository([[deepswe]] §2.1) 기준 decontamination.
- 알려진 측정 오류: Mind2Web-Live best-of-three run([[explorer]] App. A.1), filtering과 in-domain evaluation에 같은 verifier 사용([[explorer]] §5.1), end-of-training collapse를 가리는 best-checkpoint table([[simulator-collapse]] §4.2), policy-violating run을 통과시키는 benchmark reward([[tau-bench]] §3).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| 같은 simulator로 training/evaluation | Training reward와 in-distribution eval은 오르지만 human/other-simulator success는 오르지 않음 | Training에 쓰지 않은 simulator family와 human sample에서 평가 |
| Training simulator가 들어간 panel로 checkpoint 선택 | Reported best checkpoint가 held-out user에서 end-of-training checkpoint보다 좋음 | Disjoint simulator로 선택하고 final checkpoint도 보고 |
| Judge agreement를 precision으로 취급 | Agreement는 높지만 accepted trajectory에 failure 존재 | Labeled sample에서 accepted trajectory precision 계산 |
| Task당 success 하나만 보존 | Train task pass@1은 높지만 pass^k 낮음 | Task별 c/n을 저장하고 pass^k 보고 |
| Unmarked failed trajectory SFT | 훈련 뒤 action/tool-call error rate 상승 | Negative 유무의 action error rate 비교(NAT Table 6) |
| Anchor 없이 또는 너무 여러 번 trajectory DPO | Rejected와 chosen log-prob가 함께 떨어지고 몇 iteration 뒤 reward 하락 | 두 log-prob 기록, NLL term 추가, iteration cap |
| Timeout/crash를 failure로 penalize | Verified outcome 없는 rollout에 negative advantage, reward collapse | Finish reason 기록, masked/penalized run 비교 |
| Rule 기록 없이 group padding | Padding 순서에 따라 group mean 변함 | Repeated sample 및 pre-padding statistic 기록 |
| 거의 항상 refuse/comply하는 simulator | Zero-variance/all-failure group 비율이 절반 초과 | Group variance share, entropy, task별 simulator reply distribution 기록 |
| 가장 강한 benchmark model이 최선 teacher라고 가정 | Held-out task student score가 teacher score를 따르지 않음 | Fixed task source의 held-out dev set에서 teacher ablation |

## 이해 점검

1. τ²-bench는 telecom의 user-simulator error rate가 retail보다 낮다고 보고한다. Tool로 user를 제한하면 왜 error가 줄 수 있는지, 같은 결론을 retail에 적용하기 전에 어떤 추가 실험이 필요한지 설명하라.
2. [[simulator-collapse]] Thm. 3.2 bound를 써서 low-entropy simulator에서 training reward는 오르고 held-out reward는 떨어질 수 있는 이유를 설명하라. 이어서 같은 collapse가 non-zero advantage group 수를 줄이는 이유도 설명하라.
3. Explorer는 judge agreement 81%를 보고한다. Table 10에서 accepted trajectory의 precision을 계산하고, agreement가 아니라 이 수가 failure가 positive SFT target이 되는 양을 좌우하는 이유를 설명하라.
4. NAT의 Random-string prefix는 Correct/Incorrect와 비슷하게 작동했다. 이것이 marked negative의 mechanism에 무엇을 뜻하는지, unmarked negative가 action error를 왜 높이는지 설명하라.
5. IPR에서 SFT term 제거의 비용은 어느 DPO term 제거보다 크다. Softmax gradient와 likelihood displacement를 써서, 유사 trajectory만 둔 DPO objective가 correct action probability를 줄일 수 있는 이유를 설명하라.
6. Success 2개, failure 4개, timeout 2개의 GRPO group에서 timeout을 penalize할 때와 mask할 때 success advantage를 계산하라. Masking이 틀린 선택인 경우도 설명하라.
7. OpenThoughts-Agent는 8B student에 GPT5 trace보다 GLM-4.6 trace가 좋았다고 보였다. 이 결과를 낼 수 있는 가설 두 개와 그들을 구분할 measurement 하나를 제시하라.
8. UserRL model은 GPT-4o user보다 human user에서 점수가 높았고, Lost in Multi-Turn은 real user가 score를 낮출 것이라 주장한다. 둘이 동시에 성립하는 이유와, new domain simulator validation에 주는 뜻을 설명하라.

## 연결

- 이전: ch-29c — *Agentic Environment and Task Synthesis at Scale* (이 장 simulator, verifier, failure handling이 작동하는 environment/task).
- 다음: ch-29e — *Instruction Tuning and Generalization to Unseen Tasks* (agent SFT set에 적용할 held-out task protocol).
- 의존성: ch-22 — *Quality, Diversity, and Gradient-Based Data Selection* (trajectory filtering이 확장하는 selection 방법).
- 앞 장: ch-25 — *Multi-Turn Conversation Synthesis* (qa.md Q12 customer-simulator observation, §1 callback); ch-27 — *Agentic Trajectory Data* (trajectory format/source).
- 뒤 장: ch-31a — *Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood* (type 2/3 전체 처리); ch-35 — *Distillation in Practice A: Where Labs Insert Teacher Data*; ch-43a — *Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages* (type 4 derivation); ch-45b — *Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability* (§6 계속); ch-45c — *Context Management for Long-Horizon Agents*; ch-49 — *Judge Models: Bias, Calibration, and Judge-Specific Overfitting*; ch-51a — *Evaluating Agent Generality and Reliability*.

## 출처

- [[tau-bench]] — LLM user simulator 설계, state-based reward 및 그 불충분성, pass^k 정의, simulator 한계, invalid tool-call count(chapter excerpt).
- [[tau2-bench]] — dual control, tool-constrained user prompt, persona, simulator error rate/error type(chapter excerpt).
- [[apigen-mt]] — blueprint-then-dialogue 생성, Best-of-N user, failure discard, xLAM-2 distillation.
- [[kimi-k2]] — LLM-generated persona, stateful tool simulator, rubric judge filtering, length-budget penalty.
- [[userrl]] — gym simulator, training/evaluation simulator, SFT cold start, real-user comparison, RL/SFT configuration(chapter excerpt).
- [[simulator-collapse]] — collapse definition/bound, diagnostic, fix, human study, measurement caveat.
- [[simulator-collapse-recipe]] — Recipe table의 RL configuration/co-training setting.
- [[lost-in-multi-turn]] — sharded simulation, aptitude/unreliability metric, simulator inspection/limit(chapter excerpt).
- [[explorer]] — explore-then-refine 생성, verifier confusion matrix, synthetic-only limit.
- [[agent-early-experience]] — non-expert action의 reward-free 사용, DPO collapse 비교, out-of-domain result.
- [[agent-early-experience-recipe]] — Tau-Bench retail SFT setting.
- [[os-genesis]] — reverse task synthesis, graded trajectory reward model, reward-proportional sampling.
- [[learning-from-failure-nat]] — failure-conditioned training, negative quality/prefix ablation(chapter excerpt).
- [[eto-trial-and-error]] — success-failure pair의 trajectory-level DPO, iteration/granularity failure mode.
- [[ipr-step-level-refinement]] — Monte Carlo step reward, step-level DPO, SFT-term ablation(chapter excerpt).
- [[glm-5]] — SFT의 masked erroneous segment, agentic RL에서 environment-collapse sample 제외/group padding(chapter excerpt).
- [[deepswe]] — truncated/timed-out trajectory compact filtering.
- [[deepswe-recipe]] — blog와 script의 masking flag 충돌.
- [[openthoughts-agent-glm46-teacher]] — agent SFT의 teacher/task-source ablation.
- [[together-coderforge-agent-trajectories]] — 대형 test-verified trajectory set, failed-trajectory statistic, decontamination, limit(chapter excerpt).
- [[likelihood-displacement]] — rejected-sample gradient가 preferred response likelihood를 낮출 수 있는 mechanism.
