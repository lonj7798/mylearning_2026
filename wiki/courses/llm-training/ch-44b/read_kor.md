<!-- chapter: ch-44b
     track: rl
     kind: content
     title: 일반 능력을 위한 다중 도메인 RL: 검증 불가능한 보상과 도메인 혼합
     deps: [ch-44a, ch-38a]
     sources: [[deepseek-r1]], [[deepseek-r1-recipe]], [[qwen-3]], [[qwen-3-general-rl]], [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[kimi-k2]], [[kimi-k2-recipe]], [[nemotron-cascade]], [[rubrics-as-rewards]], [[rlcf-checklists]], [[deepseek-grm]], [[guru-cross-domain-rl]], [[general-reasoner]], [[nemotron-crossthink]], [[multi-domain-rlvr-data-centric]], [[scaling-reasoning-losing-control-mathif]], [[qwen3-2507-instruct-thinking-split]], [[deliberative-alignment]], [[openai-safe-completions]], [[ifbench]], [[grpo]]
     figures: figures/cascade-stage-tracker.html
     revised: 2026-09 (일반성 개정)
-->

# 제44b장 — 일반 능력을 위한 다중 도메인 RL: 검증 불가능한 보상과 도메인 혼합

> **핵심 통찰.** 여러 도메인에서 수행하는 RL은 그 도메인들에 걸친 평균 점수를 올리지만, 어느 한 도메인에서의 최고 점수는 보통 그 도메인만으로 학습한 실행이 얻는다. [[nemotron-crossthink]]에서 가장 좋은 혼합 블렌드는 7개 벤치마크 평균 58.12로 수학 전용 블렌드의 57.82보다 높지만, AMC 23에서는 65.0으로 수학 전용의 70.0보다 낮다(표 3). [[multi-domain-rlvr-data-centric]]에서도 3도메인 실행은 전체 평균이 가장 높지만(56.57), 퍼즐 평균은 퍼즐 전용 학습의 61.98보다 낮은 49.73이다(표 9). 프런티어 레시피는 이 교환을 받아들인다. 일반 RL 뒤 Qwen3-32B는 추론 RL 체크포인트와 비교해 IFEval에서 12.0점을 얻고 AIME'24에서 2.4점을 잃는다([[qwen-3-general-rl]], 표 22). 일반 RL을 가능하게 하는 레시피의 구성 요소는 보상이다. 검증기가 없는 곳에서는 루브릭, 체크리스트, 생성형 보상 모델, 안전 명세 판정기가 검증기를 대신한다.
>
> **지침.** 한 단계가 프로그램으로 검사할 수 없는 과제를 포함해야 한다면, 같은 판정기가 낸 단일 Likert 점수보다 인스턴스별 루브릭이나 체크리스트를 판정기가 채점하게 하라. 7B에서 RaR-Implicit는 HealthBench 31.2를 기록했지만, 루브릭 없는 Likert 보상은 25.5였다([[rubrics-as-rewards]], 그림 2). 규칙 기반 검증기가 있으나 응답 품질도 중요하면, 검증기만 쓰지 말고 하나의 보상 안에서 검증기와 보상 모델 점수를 결합하라. 검증기 전용 instruction-following RL은 AlpacaEval 2를 33.5에서 21.3으로 낮췄고([[ifbench]], 표 3), [[nemotron-cascade]]에서는 사람 정렬도 악화시켰다(§4.4.2). 도메인마다 검증 지연 시간과 응답 길이가 다르면, 순차 도메인별 RL이 보고된 선택지다. 이 방식에서 이전 단계 벤치마크에 대한 가장 큰 후속 단계 손실은 ArenaHard 3.8점으로, 14B-Thinking 모델에서 RLHF 뒤 93.1이었던 점수가 Math RL 뒤 89.3이 됐다([[nemotron-cascade]], 표 4-8). 목표가 하나의 균형 잡힌 모델이고 도메인들의 비용이 이미 비슷하다면, 혼합 도메인 RL은 가장 높은 교차 도메인 평균을 보인 보고된 선택지다([[guru-cross-domain-rl]], 표 3). 두 설계를 같은 데이터와 계산량으로 비교한 출처는 없다.

## 이 장이 범용 모델에 중요한 이유

이 트랙의 앞 장들은 한 번에 하나의 보상 채널을 훈련한다. 수학과 코드에는 검증기를 사용하고([[ch-44]]), 길이는 제어하며([[ch-44a]]), 채팅에는 Bradley-Terry 보상 모델을 사용한다([[ch-41]]). 배포된 모델은 같은 세션에서 이 모든 요청을 받으므로, post-training의 마지막 부분은 여러 목적을 동시에 유지해야 한다. 이 장은 추론 RL과 [[ch-45]]의 자기개선 루프 사이에 놓인 단계를 다룬다. 즉, 사전학습 → 중간학습 → SFT → 선호 최적화 → **추론 RL → 일반 다중 도메인 RL** → 평가다.

이 단계가 모델의 범위를 넓히는지 좁히는지는 두 질문으로 결정된다. 첫째는 검사기가 없는 과제의 보상을 어디에서 얻는가이다. 이 질문에 잘못 답하면 모델은 과제 자체가 아니라 판정기를 최적화하게 된다([[ch-42]]). 둘째는 서로 다른 도메인의 프롬프트를 어떤 순서로 배치하거나 어떻게 섞는가이다. 한 도메인에서의 RL 업데이트는 다른 도메인에서의 행동을 양방향으로 바꾸며, 측정된 변화량은 최대 22.56점이다([[multi-domain-rlvr-data-centric]], 표 9).

## §1 프런티어 레시피가 최종 RL 단계에서 하는 일

**정의.** *일반 RL*([[qwen-3]], [[glm-5]]이 사용하는 용어)은 프롬프트 집합이 프로그램으로 검증할 수 없는 과제를 포함한 여러 과제 유형에 걸쳐 있고, 보상이 규칙 검사·보상 모델·판정기 모델의 혼합에서 나오는 RL 단계다. *다중 도메인 RLVR*은 모든 도메인에 여전히 검증기가 있는 더 좁은 경우다.

보고서에는 두 구조가 등장한다. **혼합(mixed)** 구조에서는 하나의 단계가 모든 도메인에서 프롬프트를 뽑고, 각 프롬프트를 그 프롬프트 전용 보상 함수로 라우팅한다. **순차(sequential, cascaded)** 구조에서는 각 도메인에 자체 단계를 주고, 같은 체크포인트에서 그 단계들을 하나씩 차례로 실행한다.

| 레시피 | 최종 RL 단계의 구조 | 보상 출처 | 간섭에 관해 보고서가 보여주는 것 |
|---|---|---|---|
| DeepSeek-R1 ([[deepseek-r1]]) | 혼합: 추론 프롬프트에는 규칙 보상, 일반 프롬프트에는 선호 보상을 하나의 1,700-step 단계에서 사용. 일반 데이터와 선호 보상은 마지막 400 step에서만 사용 | 규칙 + helpful RM(66k 쌍) + safety RM(106k 프롬프트) + 형식 + 언어 일관성(식 8-10) | helpful RM을 넣으면 보상은 상승하지만 Codeforces 성능은 하락한다(B.5, 그림 6). 400-step 창이 명시된 완화책이다(§3.2.2). |
| Qwen3 ([[qwen-3]], [[qwen-3-general-rl]]) | 혼합: 추론 RL과 thinking-mode 융합 뒤, “20개가 넘는 서로 다른 과제”에 걸친 하나의 General RL 단계 | 규칙; 참조 답변을 쓰는 모델 기반 보상(Qwen2.5-72B-Instruct); 참조 없는 보상 모델(§4.4) | 표 22에서 일반 및 instruction 지표는 상승하고 thinking mode의 AIME'24와 LiveCodeBench는 하락한다. 저자들은 다재다능성을 위해 이 교환을 받아들인다고 밝힌다. |
| GLM-4.5 ([[glm-4-5]]) | 도메인별 전문가 모델(추론, 에이전트, 일반 채팅)을 각각 cold start하고 RL 훈련한 다음, 하나의 모델로 증류. 일반 RL은 네 트랙에서 마지막에 실행 | 사람 선호로 훈련한 RM + AI 루브릭 채점; instruction following에는 규칙 + RM + critique 모델; function calling에는 step-wise 규칙(§3.4) | 병리 현상은 “출력의 1% 미만인 경우가 많다”. 보고서는 이것을 일반 RL 내부에서 고치기에는 sample-inefficient하다고 말한다(§3.4). |
| GLM-5 ([[glm-5]]) | 순차: 추론 RL → 에이전트 RL → 일반 RL, 그 뒤 on-policy cross-stage 증류 | 규칙 기반 함수 + outcome reward model + 생성형 reward model, 그리고 스타일 앵커로서 사람이 작성한 응답(§3.4) | 증류 단계가 있는 이유는 “서로 다른 목적을 순차적으로 최적화하면 이전에 획득한 능력이 누적적으로 저하될 수 있기” 때문이다(§3.5). |
| Kimi K2 ([[kimi-k2]]) | 공동: 검증 가능한 보상 gym과 자기비평 루브릭 보상을 함께 훈련 | 검증기(수학, STEM, 논리, 코드, SWE sandbox, hack-check 층이 있는 IF) + core·prescriptive·사람 루브릭에 대한 K2 critic 순위화(§3.2) | 망각과 RL 과제 집합 과적합을 막기 위해 PTX 손실(사람이 고른 고품질 샘플의 보조 next-token-prediction 손실을 RL 목적에 추가)을 쓴다(§3.2.3). ablation은 없다. |
| Nemotron-Cascade ([[nemotron-cascade]]) | 순차: RLHF → IF-RL → Math RL → Code RL → SWE RL. 14B 실행에서 총 2,232 RL step | 72B Bradley-Terry RM; 같은 RM과 결합한 규칙 기반 IF 검증기; 기호 수학 검증기; unit test; SWE test | 모든 벤치마크에 대해 단계별 표를 제공한다. 뒤 단계들은 “이전 도메인에서 달성한 벤치마크 성능을 거의 저하시키지 않는다”(초록). |

이 표는 순위가 아니라 두 설계 계열을 설명한다. 혼합 단계에는 응답 길이가 한 자릿수 규모만큼 다른 프롬프트들을 위한 하나의 보상 라우터와 하나의 길이 예산이 필요하다. 순차 단계에서는 각 도메인이 자체 최대 응답 길이, temperature, 검증기 지연 시간을 유지할 수 있다. 이것이 [[nemotron-cascade]]가 §1에서 밝힌 동기인데, 수학 검증은 코드 검증보다 “orders of magnitude” 더 빠르다.

## §2 검증기가 없을 때의 보상

### 2.1 네 가지 보상 출처

두 보고서는 겹치는 보상 집합과 각 집합의 명시된 절충점을 제시한다. [[qwen-3-general-rl]](§4.4)은 아래 1-3 유형을, [[glm-5]](§3.4)은 1·3·4 유형을 열거한다.

1. **규칙 기반 보상** — 프로그램이 판정한다. 정확하고 저렴하며 해킹하기 어렵지만, 규칙으로 표현할 수 있는 것에 한정된다.
2. **참조 답변이 있는 모델 기반 보상** — 판정기가 응답을 정답(gold answer)과 비교한다. Qwen3는 여기서 Qwen2.5-72B-Instruct를 사용하며, 목적은 “순수 규칙 기반 보상에서 생길 수 있는 false negative를 피하는 것”이라고 밝힌다.
3. **참조 답변이 없는 모델 기반 보상** — 사람 선호로 훈련한 스칼라 보상 모델이다. GLM-5는 이를 outcome reward model이라고 부른다. 분산이 낮고 효율적이지만 “reward hacking에 더 취약하다”.
4. **생성형 보상 모델** — 언어 모델이 기준이나 비평을 작성한 뒤 점수를 매긴다. GLM-5의 표현으로는 “그런 악용에는 더 견고하지만, 분산은 더 높은 경향이 있다”.

§2의 나머지는 (2)와 (4)를 제대로 구성했을 때 실제로 어떤 형태가 되는지 다룬다.

### 2.2 보상으로서의 루브릭

**정의.** 루브릭 보상은 품질 점수 하나를, 해당 단일 프롬프트를 위해 작성한 가중치 있는 k개 기준으로 바꾼다. 각 기준은 판정기가 검사한다([[rubrics-as-rewards]], §2.2).

**해결하려는 문제.** 판정기의 단일 Likert 점수는 응답마다 숫자 하나만 주며 무엇이 잘못됐는지는 말해 주지 않는다. 크기가 다른 판정기들은 그 점수에서도 서로 불일치한다(§5, 그림 3).

**작동 방식.** LLM은 데이터셋의 참조 답변을 조건으로 하여 프롬프트마다 자체 완결적인 항목 7-20개를 작성한다. 각 항목에는 범주(Essential, Important, Optional, Pitfall)와 가중치가 붙는다(§3.2). 훈련 시 판정기는 모든 기준을 보고 하나의 점수를 반환하거나(implicit), 각 기준을 검사한 뒤 점수들을 결합한다(explicit, 식 1).

```
r(x, ŷ) = Σ_j w_j · c_j(x, ŷ) / Σ_j w_j
```

x는 프롬프트, ŷ는 샘플링한 응답, k는 기준 수, w_j는 j번째 기준의 가중치다. c_j(x, ŷ) ∈ {0, 1}은 판정기가 ŷ가 j번째 기준을 만족한다고 내린 판정이다. 분모는 기준 수가 서로 다른 프롬프트 사이에서도 보상을 비교할 수 있게 한다.

**계산 예시.** RaR-Explicit는 Essential에 1.0, Important에 0.7, Optional에 0.3, Pitfall에 0.9를 매긴다(§4.4). Essential 하나, Important 하나, Pitfall 하나가 있는 루브릭을 생각하면 분모는 1.0 + 0.7 + 0.9 = 2.6이다. Essential과 Pitfall은 만족하고 Important를 놓친 응답의 점수는 (1.0 + 0.9) / 2.6 = 1.9 / 2.6 = 0.73이다. 반대로 Essential을 놓치고 나머지 둘을 만족하면 (0.7 + 0.9) / 2.6 = 1.6 / 2.6 = 0.62이다. 집계가 선형이므로 어떤 기준 하나를 놓치는 비용은 그 기준의 가중치를 전체로 나눈 값이다. Essential은 1.0 / 2.6 = 0.38, Important는 0.7 / 2.6 = 0.27이고, 두 비용의 비는 1.43 = 1.0 / 0.7이다.

**근거.** gpt-4o-mini 판정기를 쓴 Qwen2.5-7B의 GRPO에서 HealthBench는 RaR-Implicit 31.2, RaR-Explicit 29.7, reference-Likert 28.9, direct Likert 25.5, 모든 프롬프트가 공유하는 일반 루브릭 네 개 12.5, 별도 학습 없이 쓰는 Qwen2.5-7B-Instruct 22.7이었다. GPQA-Diamond는 같은 순서로 37.6 / 36.9 / 36.5 / 34.8 / 31.7 / 35.0이었다(그림 2 막대 레이블). 참조 답변을 보고 작성한 루브릭은 참조 답변 없이 작성한 루브릭보다 HealthBench-1k에서 높았다. 35.9 대 32.0이다(표 1). **결과(단일 연구).**

**조건과 한계.** 정책 크기는 하나(7B)이고, GRPO는 300 step이며, 의료와 과학만 다뤘다. 대화, 도구 사용, 에이전트 과제는 시험하지 않았다(§9). 훈련 보상과 평가를 같은 판정기 모델이 채점하므로(그림 2 캡션), 측정된 향상의 일부는 과제 품질이 아니라 그 판정기와의 일치일 수 있다.

### 2.3 체크리스트 피드백

[[rlcf-checklists]]는 instruction following에 같은 발상을 적용하고 프로그램을 추가한다. Qwen2.5-72B-Instruct는 작은 모델 네 개가 낸 후보 응답을 조건으로, 지시문마다 예/아니오 체크리스트를 작성한다. 각 항목에는 100점 만점의 중요도 가중치를 준다. 각 항목은 0-100에서 판정기 샘플 25개의 평균으로 채점한다. 판정기가 요구사항을 정확히 검사할 수 있다고 확신하는 경우에는 생성한 검증 프로그램도 사용하며, 두 점수의 평균을 낸다. 가중치 있는 항목 점수들을 평균해 응답마다 점수 하나를 만들고, 기준별 차이가 가장 큰 쌍의 40%를 DPO 쌍으로 삼아 DPO로 정책을 훈련한다.

일반성에 중요한 세부 사항은 두 가지다. 첫째, 저자들은 긴 서문으로 시작하는 응답을 관찰한 뒤 이를 reward hacking으로 해석하고, 모든 체크리스트에 “보편 요구사항” 두 개를 추가했다. 직접성(directness)과 문맥에 맞는 어조(context-appropriate tone)이며, 각각 가중치가 100/100이다(§2). 둘째, 이 방법은 선호 기반이다. 체크리스트 피드백을 policy gradient에 사용하는 일은 향후 연구로 적혀 있다(§7).

**근거.** Qwen2.5-7B-Instruct에서 FollowBench 평균 hard satisfaction rate는 71.4에서 RLCF로 75.3이 됐다. Skywork-Reward-Gemma-2-27B를 통한 DPO는 69.5, ArmoRM을 통한 DPO는 70.4였다. InFoBench 전체 점수는 78.1에서 84.1 / 82.0 / 83.5로, Arena-Hard vanilla은 51.3에서 54.6 / 55.1 / 50.8로 변했다. 보상 모델들은 일부 벤치마크에서는 도움이 되고 다른 곳에서는 하락했지만, RLCF는 다섯 벤치마크 모두에서 양의 변화였다(표 2-4). **결과(단일 연구).**

**논문이 직접 측정한 범위 축소.** RLCF 프롬프트는 일상 지원과 글쓰기에 치우친 WildChat에서 왔다. 그 분포 밖 과제에서는 점수가 하락한다. XSTest unsafe는 83.0 → 81.0, GSM8K는 83.2 → 82.2, TruthfulQA MC1은 43.5 → 42.0이다(표 9). 이것이 이 장에서 반복되는 패턴이다. 프롬프트 분포에 맞춰 잘 설계한 보상도 그 분포 밖에서는 모델의 범위를 좁힌다.

### 2.4 범용 생성형 보상 모델

[[deepseek-grm]]은 판정기 자체를 RL로 훈련한다. 모델은 원칙(principles)을 만들고, 이어 비평을 만들고, 응답마다 pointwise 점수를 낸다. Self-Principled Critique Tuning은 이 행동을 훈련한다. rejective fine-tuning은 점수가 정답인 최고 응답을 엄격하게 가장 높게 순위 매긴 궤적만 남긴다(N_RFT = 3). 그 뒤 규칙 기반 RL은 점수 순서가 맞으면 +1, 그렇지 않으면 −1을 부여하고, 출력 형식을 유지하려고 더 큰 KL 계수를 사용한다(§3.2). 추론 때는 응답 순서를 섞은 k개 샘플 궤적의 점수를 합산하고, 선택적으로 meta reward model로 필터링한다.

**근거.** RewardBench, PPE Preference, PPE Correctness, RMB에 걸친 전체 정확도는 greedy에서 69.9, Voting@32에서 71.0, meta-RM Voting@32에서 72.8이었다. Nemotron-4-340B-Reward는 70.5, Skywork-Reward-Gemma-2-27B는 66.9였다(표 2). 원칙 생성을 제거하면 2.4점이 든다(69.9 → 67.5, 표 4). 스칼라 보상 모델은 PPE Correctness에서 가장 높은 점수를 내고 다른 곳에서는 뒤처지며, 저자들은 이를 domain bias라고 설명한다(§5.2). **결과(단일 연구).**

**여기서 중요한 한계.** 이 논문은 GRM을 판정기로 평가한다. 이를 온라인 RL 내부의 보상으로 쓰는 것은 향후 연구로 적혀 있으므로(§7), 판정기 정확도가 정책 품질로 옮겨가는지는 측정하지 않았다. 그 이전의 실용적 적용은 [[glm-5]] §3.4에 있다. 생성형 보상 모델은 General RL의 세 보상 유형 중 하나지만, 그 기여만 분리한 ablation은 없다.

### 2.5 자기비평 루브릭 보상

[[kimi-k2]]는 정책 안에서 판정기를 실행한다. K2 critic은 SFT 중 선호 데이터로 초기화한 뒤, core 루브릭, reward hacking을 없애기 위한 prescriptive 루브릭, 사람이 주석을 단 루브릭을 기준으로 actor 응답을 쌍대 비교해 순위화한다(§3.2.2). RL 동안 critic은 검증 가능한 보상 프롬프트에서 나온 on-policy rollout으로 업데이트된다. 보고서는 이를 RLVR 신호를 평가 모델로 증류하는 것이라고 설명한다.

prescriptive 루브릭은 구체적인 행동 금지다. 시작 부분의 칭찬을 금지하고, 응답이 왜 좋은지를 설명하는 문장을 금지한다(App. F.2). 보고서는 App. F.3에서 부작용도 명시한다. 이 규칙들은 완곡한 표현(hedging)과 단서를 붙이는 표현(disclaimers)을 벌하므로, 뉘앙스가 적절한 곳에서 “모델이 때때로 확신을 과장할 수 있다”. 표면 패턴을 금지하는 행동 보상은 그 패턴의 정당한 사용도 함께 없앤다.

### 2.6 검증기와 보상 모델의 결합

제약 만족에 대해 0 또는 1을 반환하는 검증기는 응답 품질에 대해서는 아무 말도 하지 않는다. 두 출처가 그 결과를 측정한다. [[ifbench]]는 Tülu-3-8B-DPO 정책에서 검증기만 사용한 instruction-following RLVR이 AlpacaEval 2를 33.5에서 21.3으로, 제약을 제거한 같은 응답의 GPT-4.1 판정기 품질을 10점 만점 7에서 6.4로 옮겼다고 보고한다(표 3, §5). 보상 모델 항을 추가하면 IF 점수를 대가로 AlpacaEval 2가 31.6까지 회복된다(App. E). [[nemotron-cascade]]는 규칙 기반 IF 검증기만으로는 “사람 정렬 결과가 악화됐다”고 보고하며, thinking 모델에 사용한 결합 보상을 제시한다(§4.4.2).

```
r_i = R_IF(o_i) + sigmoid(R̂_RM(o_i))   if R_IF(o_i) = 1
r_i = 0                                 otherwise
R̂_RM(o_i) = (R_RM(o_i) − mean({R_RM(o_j)}_{j=1..G})) / std({R_RM(o_j)}_{j=1..G})
```

o_i는 크기 G인 그룹에서 i번째로 샘플링한 응답, R_IF ∈ {0, 1}은 검증기 출력, R_RM은 스칼라 보상 모델 점수, R̂_RM은 그 점수를 그룹 안에서 표준화한 값이다. sigmoid는 이를 (0, 1)로 보내므로, 보상 모델 항은 검증기 결정을 절대로 뒤집을 수 없다.

**계산 예시.** 제약이 있는 프롬프트 하나에 G = 4개 응답이 있다고 하자. 둘은 모든 제약을 만족하고 표준화 RM 점수가 각각 0.8, −1.2다. 나머지 둘은 제약을 위반한다. 보상은 각각 1 + sigmoid(0.8) = 1.69, 1 + sigmoid(−1.2) = 1.23, 0, 0이다. 그룹 평균은 0.73, 표준편차는 0.748이므로 GRPO advantage는 +1.28, +0.67, −0.98, −0.98이다. 제약을 지킨 모든 응답은 양의 advantage를 유지하고, 더 잘 쓴 응답은 더 약한 응답보다 약 두 배의 업데이트 압력을 받는다. 의도한 순서는 제약을 먼저 만족하고, 그다음 잘 쓰는 것이다.

같은 보고서의 8B 통합 모델은 같은 문제에 다른 경로를 쓴다. RLHF는 thinking mode와 non-thinking mode 모두에서 수행하고, IF-RL은 non-thinking mode에서만 수행한다. 명시된 가설은 이미 정렬된 모델의 저렴한 모드에 IF-RL을 적용하면 규칙 기반 검증기를 reward-hacking할 가능성이 더 낮다는 것이다(§4.4.2). RLHF와 IF-RL의 순서를 뒤집으면 “훨씬 더 나쁜 결과”가 나왔지만, 수치는 출력하지 않았다. **결과(단일 연구), 정성적.**

## §3 도메인 간 전이와 간섭

**정의.** *전이(transfer)*란 도메인 A에서 한 RL이 도메인 B의 성능에 일으킨 변화다. 그 변화가 음수이면 *간섭(interference)*이다. 아래의 모든 연구는 같은 방식으로 이를 측정한다. 하나의 프롬프트 집합에서 학습하고, 모든 도메인을 덮는 고정된 평가 스위트에서 평가한다.

### 3.1 전이는 도메인과 과제 난이도에 따라 비대칭이다

[[guru-cross-domain-rl]]은 Qwen2.5-7B-Base와 Qwen2.5-32B-Base를 Math, Code, Science, Logic, Simulation, Tabular의 각 3K-샘플 부분집합 및 이들을 합친 18K 혼합 데이터에서 학습시키고, 여섯 도메인 모두에서 평가한다. 보고된 결과는 다음과 같다. Math, Code, Science는 다른 도메인에서 한 RL로도 향상되지만, Logic, Simulation, Tabular는 해당 도메인의 데이터가 필요하다. 향상되는 도메인 안에서도 쉬운 과제(MATH500, AMC, HumanEval, MBPP)는 어려운 과제(AIME24, LiveCodeBench)보다 도메인 밖 RL에서 더 많이 향상된다. 가장 점수가 낮은 과제(ARC-AGI, CodeI/O)는 도메인 간 향상이 “미미하거나 거의 없다” (§3.1). 저자들은 그 이유를 사전학습 동안의 노출로 설명한다(**해석**).

학습된 모델도 같은 점을 도메인 내부 성능으로 보여 준다. GURU-7B는 17개 벤치마크 평균 43.29로 Open-Reasoner-Zero-7B의 35.42와 General-Reasoner-7B의 33.76보다 높다. GURU-32B의 평균은 54.24이고 비교 모델은 47.53이다(표 3). 예외는 코퍼스 안의 어느 도메인도 직접 겨냥하지 않은 instruction following이다. GURU-7B는 IFEval에서 35.81로, General-Reasoner-7B의 39.56 및 SimpleRL-7B의 36.69보다 낮다. 이 비교에서 추론 도메인을 넓게 덮었다는 사실은 instruction following까지 넓게 덮는 결과로 이어지지 않았다.

### 3.2 난이도 필터링은 도메인 내부 최적화이며 전이를 희생할 수 있다

추론 RL에서는 한 도메인의 프롬프트를 어려운 항목으로만 걸러 쓰는 일이 흔하다. [[guru-cross-domain-rl]]은 그 선택의 도메인 간 비용을 측정한다(표 2, Qwen2.5-7B-Base, 200단계, 최고 검증 정확도).

| 학습 데이터 | MATH500 | AMC | AIME24 | HumanEval | LiveCodeBench | HiTab | Multihiertt |
|---|---|---|---|---|---|---|---|
| 필터링하지 않은 수학 | 75.8 | 52.1 | 15.8 | 82.3 | 11.1 | 56.5 | 32.0 |
| 난이도로 필터링한 수학 | 78.6 | 58.4 | 21.7 | 73.1 | 10.7 | 53.5 | 35.5 |

도메인 내부에서는 각각 +2.8, +6.3, +5.9점 향상됐지만, 행에서 가장 쉬운 두 도메인 밖 과제인 HumanEval에서는 −9.2, HiTab에서는 −3.0점이 됐다. 저자들은 필터링된 데이터로 학습하는 동안 두 과제 모두에서 정확도가 붕괴했다고 보고한다. 가장 어려운 두 도메인 밖 과제는 반대 방향으로 움직이거나 거의 움직이지 않는다. LiveCodeBench는 0.4점 하락하고 Multihiertt는 3.5점 상승한다. **결과(단일 연구).**

### 3.3 혼합 데이터와 혼합이 주는 이득

[[nemotron-crossthink]]은 GRPO로 Qwen2.5-7B에서 데이터 혼합을 비교한다. 가장 좋은 혼합은 범용 추론 데이터 대 수학 데이터 비율이 2:1인 경우다. 이 혼합은 7개 벤치마크 평균 58.12로, 학습하지 않은 모델의 44.75와 수학 전용 혼합의 57.82보다 높다. 범용 데이터만 사용하면 평균은 53.30이다. 벤치마크별 결과는 평균 차이가 생기는 위치를 보여 준다. 혼합 모델은 수학 전용 대비 MMLU-Pro에서 +3.56, AGIEval에서 +2.32이지만 AMC 23에서는 −5.0이다(표 3). 같은 보고서는 응답 길이도 측정한다. 혼합 모델의 정답은 수학 전용 모델의 정답보다 평균 토큰 수가 28% 적다. 또한 혼합 모델은 과제 종류에 따라 길이를 바꾼다. 수학 프롬프트에서는 622토큰, 일반 프롬프트에서는 385토큰으로 62% 차이지만, 수학 전용 모델의 차이는 14%다(§6, 표 12). **결과(단일 연구).**

[[general-reasoner]]는 다른 코퍼스로 같은 비교를 한다. 규칙 기반 매칭으로 답을 확인할 수 없는 경우에는 1.5B 생성형 검증기를 사용하며, 웹에서 수집한 질문 230K개를 쓴다. 7B에서는 수학 전용 학습이 수학 관련 평균에서 0.6점 더 높다(49.1 대 48.5). 그러나 세 일반 벤치마크 모두에서는 더 낮다. MMLU-Pro는 2.0점(56.9 대 58.9), GPQA는 1.5점(32.8 대 34.3), SuperGPQA는 4.4점(29.8 대 34.2) 낮다. 14B에서는 전체 데이터를 쓴 모델이 모든 열에서 더 높다(표 4). 검증기 ablation은 별개의 결과다. Qwen3-4B-Base에서 120단계 동안 조건을 동일하게 맞추면, 모델 기반 검증기가 네 평가 세트 전부에서 규칙 기반 검증기보다 높다(MMLU-Pro 60.1 대 58.1). 규칙 기반 실행은 약 60단계 부근에서 정체한다(표 5, 그림 4). 저자들은 규칙 기반 매칭이 유효한 답을 오답으로 표시하여 쓸 수 있는 보상 신호를 제거하기 때문이라고 해석한다(**해석**).

§3.1의 결과까지 함께 보면 다음 두 결론은 **반복 관찰됨**으로 분류된다. 같은 규모에서 도메인을 섞으면 단일 도메인 실행보다 다중 도메인 평균이 높아진다([[nemotron-crossthink]] 표 3, [[general-reasoner]] 표 4, [[multi-domain-rlvr-data-centric]] 표 9). 그리고 한 단일 도메인의 최고 성능은 대체로 그 도메인에 특화했거나 그 도메인과 더 가까운 데이터로 학습한 실행이 갖는다([[nemotron-crossthink]]의 AMC 23, [[multi-domain-rlvr-data-centric]]의 puzzles, [[guru-cross-domain-rl]] 표 2의 도메인 내부 난이도 필터링 및 표 3의 IFEval). [[guru-cross-domain-rl]]은 전이 연구(그림 3)의 단일 도메인 대 혼합 데이터별 셀 값을 출력하지 않는다. 따라서 여기서 그 연구가 제공하는 것은 전이의 방향이지, 혼합 대 단일의 평균 비교가 아니다.

### 3.4 간섭은 측정되며, 평균을 내는 관례가 이를 가릴 수 있다

[[multi-domain-rlvr-data-centric]]은 Qwen2.5-7B-Base에서 Math, Code, Puzzle의 일곱 가지 모든 조합을 실행한다(표 9).

| 학습 데이터 | 수학 평균 | 코드 평균 | 퍼즐 평균 | 전체 평균 |
|---|---|---|---|---|
| Base | 22.48 | 67.46 | 9.07 | 31.50 |
| Math | 47.48 | 64.23 | 22.42 | 45.11 |
| Puzzle | 29.47 | 71.35 | 61.98 | 50.72 |
| Code | 19.17 | 73.95 | 22.55 | 35.78 |
| Math + Puzzle | 49.72 | 44.90 | 49.78 | 48.36 |
| Puzzle + Code | 32.06 | 74.88 | 55.15 | 50.89 |
| Math + Code | 47.22 | 75.06 | 25.34 | 48.92 |
| Math + Puzzle + Code | 49.75 | 73.63 | 49.73 | 56.57 |

Math + Puzzle은 코드에서 base 모델보다 22.56점 낮으며, 두 단일 도메인 실행 어느 쪽보다도 나쁘다. 그 쌍에 코드 데이터를 더하면 코드 평균은 73.63으로 회복되고 퍼즐 평균은 거의 그대로다(49.78 → 49.73). 반대로 가장 좋은 쌍과 비교하면 세 도메인 조합은 퍼즐 정확도 5.42점을 잃는다(55.15 → 49.73). 대신 수학에서는 17.69점을 얻는다. 세 도메인 조합은 전체 평균이 가장 높고 어떤 도메인에서도 붕괴하지 않는다. 이것이 혼합을 지지하는 근거다.

**평균을 내는 관례의 계산 예시.** 그 표의 “전체 평균”은 세 도메인 평균의 평균이 아니라, 개별 벤치마크 7개 점수의 평균이다. Base 모델에서 7개 점수(MATH500 56.40, CountDown 1.05, AIME24 10.00, HumanEval 70.12, MBPP 64.80, KK 17.86, Zebra 0.27)는 220.50 / 7 = 31.50이다. 반면 세 도메인 평균으로 계산하면 (22.48 + 67.46 + 9.07) / 3 = 33.00이다. 두 관례는 실행의 순위를 바꿀 수 있다. 과제 평균 아래에서는 Puzzle + Code(50.89)가 Puzzle 전용(50.72)보다 높지만, 도메인 평균 아래에서는 더 낮다. 즉 (32.06 + 74.88 + 55.15) / 3 = 54.03은 (29.47 + 71.35 + 61.98) / 3 = 54.27보다 낮다. 첫 번째 관례에서는 스위트에 벤치마크가 더 많은 도메인이 더 큰 가중치를 받는다. 혼합을 평균으로 비교할 때는 어느 관례를 썼는지 명시하고 그 옆에 도메인별 수치를 출력해야 한다. 이 표의 8개 실행 중 두 개의 순위가 두 관례에 따라 바뀌기 때문이다.

같은 논문의 두 번째 경고는 학습이 아니라 측정에 관한 것이다. 동일한 체크포인트가 평가에 쓰는 chat template에 따라 CountDown에서 0.00 또는 20.79을 받는다. instruct 모델은 맞지 않는 템플릿 아래에서 MATH500 점수가 1.80까지 떨어진다(표 11). base 모델과 instruct 모델에 하나의 템플릿을 공통으로 적용해 만든 도메인 간 비교는 서로 비교할 수 없다.

## §4 추론 능력과 일반 능력의 관계

서로 독립적인 세 측정은 추론 중심 학습이 지시 제약 준수를 낮추고, 일반 RL은 최고 추론 성능을 낮춘다고 말한다.

**추론 학습은 제약 따르기를 낮춘다.** [[scaling-reasoning-losing-control-mathif]]은 Python으로 검증 가능한 출력 제약을 1~3개씩 붙인 수학 질의 420개로 MathIF를 구성한다. 23개 추론 모델 전체에서 가장 높은 hard 정확도는 Qwen3-14B의 50.71%다. 논문의 통제 실험은 Qwen2.5 base 네 개에서 long-CoT SFT와 GRPO를 실행했다. 수학 평균 정확도는 16개 학습 변형 중 15개에서 올랐지만, hard 및 soft 제약 정확도는 16개 중 15개에서 낮아졌다(표 4). RL 최대 rollout 길이를 1k 토큰에서 8k 토큰으로 늘리자 수학 정확도는 28.73에서 39.82로 올랐고 hard 정확도는 19.05에서 14.29로 낮아졌다(표 5). 이는 **결과(단일 연구)**이며, 저자들이 검증한 기제는 제약 반복 개입을 통해서만 시험했다.

**일반 RL은 최고 추론 성능을 낮춘다.** [[qwen-3-general-rl]] 표 22는 Qwen3-32B가 추론 RL, thinking-mode 융합, 일반 RL을 거칠 때의 변화를 추적한다. IFEval strict prompt는 73.0 → 78.4 → 85.0, Multi-IF는 61.4 → 64.6 → 73.0, ToolUse는 63.3 → 70.4 → 85.5가 된다. 반면 AIME'24는 83.8 → 81.9 → 81.4, LiveCodeBench v5는 68.4 → 67.2 → 65.7이다. 보고서는 그 교환을 직접 명시한다. 저자들은 더 넓은 범위의 일반 과제로 학습했기 때문에 저하가 생겼다고 **추측**하며, 전체 다재다능성을 위해 이를 받아들였다. 단계별 점수는 하나씩만 제공되며 분산은 보고하지 않는다.

**Cascade는 같은 교환을 단계 수준의 수치로 제시한다.** [[nemotron-cascade]]에서 14B-Thinking 모델의 RLHF는 ArenaHard를 76.9에서 93.1로 올리고 IFEval을 69.8에서 56.2로 낮춘다. 이어지는 IF-RL은 IFEval을 81.3, IFBench를 25.6에서 40.4로 올리는 대신 ArenaHard를 90.2로 낮춘다. ArenaHard는 이후 단계에서도 RLHF 직후의 93.1로 돌아가지 않는다. 이 보고서는 IF-RL이 entropy를 낮추고 추론 trace를 짧게 만든다고도 기록한다(§4.4.3). 이는 이 단계를 [[ch-43]]의 다양성 제어와 연결한다.

**하이브리드 모델과 분리 모델.** Qwen은 thinking / non-thinking 하이브리드 모델을 출시한 뒤 이를 분리했다. "최고 품질을 얻기 위해 하이브리드 thinking mode 사용을 중단하고 Instruct와 Thinking 모델을 별도로 학습하기로 했다"고 설명한다([[qwen3-2507-instruct-thinking-split]]). 카드의 벤치마크 표에서 Thinking-2507은 하이브리드 체크포인트의 thinking mode보다 높다(AIME25 81.5 → 92.3, Arena-Hard v2 61.5 → 79.7). 다만 2507 모델은 모드 분리 외에도 변경된 것이 있으므로 통제된 비교는 아니다. 반대 주장은 [[nemotron-cascade]]가 제시한다. 이 보고서의 8B 통합 모델은 같은 프롬프트에 대해 병렬 thinking 및 non-thinking 응답으로 학습하고, RLHF 배치의 절반을 각 모드에 배정한다. 최종 결과는 전용 8B-Thinking 모델의 AIME 2024 88.8, SWE-bench Verified 38.5에 대해 각각 89.5, 37.2이며, IFEval은 통합 모델이 더 높다(90.2 대 83.7)(표 8). 표 8의 캡션에 따르면 통합 모델의 IFEval은 non-thinking mode에서, 다른 벤치마크는 thinking mode에서 측정했다. 모드 분할 ablation(§6.1, 그림 9)은 half-and-half 배치가 단일 모드 RLHF 어느 쪽보다 ArenaHard·수학·코드에서 낫다고 보인다. **열린 질문**은 프런티어 규모에서 한 모델이 추론 비용 없이 두 모드를 함께 유지할 수 있는가이다. 두 보고서는 공유 벤치마크 프로토콜 없이 상반된 주장을 한다.

## §5 단계 순서, 망각, 그리고 단계 간 증류

[대화형 그림: 도메인별 순차 RL의 각 단계 뒤 점수](figures/cascade-stage-tracker.html)
— 모델과 비교할 단계를 선택하면 Nemotron-Cascade 파이프라인에서 각 단계마다 어느 벤치마크가 얼마나 움직였는지 볼 수 있다.

### 5.1 순차 RL이 이전 도메인에 하는 일

[[nemotron-cascade]]는 매 단계 뒤 전체 벤치마크 스윕을 공개한 여기서 유일한 출처다. 14B-Thinking 모델에서 이전 단계의 수학·코드 점수는 후속 단계에서도 대략 1점 이내로 유지된다(AIME 2025: RLHF 뒤 81.8, 이후 82.3, 83.3, 83.5, 83.3; LiveCodeBench v6: 72.3, 72.7, 72.4, 74.8, 74.6). 반면 한 단계에서 가장 크게 움직이는 값은 모두 정렬 지표다(IFEval: RLHF에서 −13.6, IF-RL에서 +25.1; ArenaHard: RLHF에서 +16.2; IFBench: IF-RL에서 +14.8).

이 모델의 표 4 변화 화살표 중 두 개는 인쇄된 점수와 맞지 않는다. IFEval 하락은 ↓12.4로 표시되어 있지만 표 2의 69.8과 표 4의 56.2 차이는 13.6이고, ArenaHard 상승은 ↑19.2로 표시되어 있지만 76.9와 93.1의 차이는 16.2다. 8B 모델의 화살표는 인쇄 점수와 일치하므로, 이 장은 전 구간에서 인쇄된 점수를 사용한다. SWE-bench Verified는 RLHF에서 4.3점 오른다(34.5 → 38.8). IF-RL, Math RL, Code RL을 거치는 동안의 변화는 최대 1.3점이며, 자신의 단계에서는 3.5점 오른다(39.6 → 43.1). **결과(단일 연구)**, seed는 하나다.

이 보고서는 순차 RL이 순차 SFT보다 덜 망각하는 이유를 네 부분으로 설명한다(§4.1.1, **해석**). 첫째, RL 데이터는 현재 정책이 생성하므로 계속 보상을 받는 이전 행동도 계속 표본으로 뽑힌다. 둘째, RL은 토큰 수준 표적을 맞추는 대신 기대 보상을 최적화한다. 셋째, 새 도메인의 보상이 이전 도메인의 보상과 충돌할 때, 특히 의미적으로 비슷한 프롬프트에서 망각이 예상된다. 넷째, 파이프라인은 프롬프트 중복을 최소화하고 단계를 일반적인 것(RLHF, IF-RL)에서 전문적인 것(수학, 코드, SWE) 순으로 배치한다. 이는 [[ch-38a]]의 on-policy 논증과 연결된다.

같은 보고서의 두 결정은 논증이 아니라 측정으로 뒷받침된다. RLHF 프롬프트 집합에서 수학 및 경쟁 프로그래밍 프롬프트를 제거한다. 수학 프롬프트를 남겨 두면 AIME25에서 약 2%의 비용이 들었다(§4.3.1). 또한 보상 모델 크기는 정책이 학습하는 내용을 바꾼다. Cascade 체크포인트가 아니라 AceReason-Nemotron-1.0-7B를 정책으로 쓴 별도 연구에서, 7B RM은 더 긴 출력을 생성해 보상을 올렸고 스타일 제어를 적용하면 ArenaHard 이득이 줄어들었다. 반면 72B RM은 응답 길이를 안정적으로 유지했고 AIME25는 약 3% 더 높였다(§6.2, 그림 10). 정책 출력에 대해 out-of-distribution인 보상 모델은 불안정성의 원천이다. 이것은 [[reward-model-overoptimization]]의 보상-모델-크기 사례다.

### 5.2 혼합 단계에는 그 단계 안의 격리가 필요하다

혼합 구조에는 이전 도메인을 보호할 단계 경계가 없으므로, 격리는 단계 내부에서 해야 한다. [[deepseek-r1]]은 일반 지시 데이터와 선호 보상을 1,700단계 중 마지막 400단계로 제한한다. 모델 기반 선호 보상에 더 오래 노출하면 보상은 오르지만 Codeforces 성능은 떨어지는 reward hacking이 발생했기 때문이다(§3.2.2, B.5, 그림 6). [[kimi-k2]]는 망각을 막기 위해 손으로 고른 고품질 표본의 PTX loss를 RL 목적함수에 더한다(§3.2.3). [[glm-4-5]]는 네 트랙에서 일반 RL을 마지막 단계로 실행하며, 일반 RL 안에서 드문 병리(언어 혼합, 반복, 서식)를 고치려면 표본 효율이 낮다고 보고한다. 이 병리는 “출력의 흔히 1% 미만”에서만 나타나기 때문이다(§3.4).

### 5.3 회복으로서의 단계 간 증류

**정의.** *On-policy distillation*은 학생 모델 자신의 표본으로 학생을 학습하되, 환경 보상 대신 교사의 로그 확률로 각 토큰을 채점한다.

[[glm-5]]는 자기 파이프라인의 단계들 사이에 이를 적용한다. 보고서는 이 단계를 두 방식으로 배치한다. §1에서는 증류를 순차 파이프라인 전체에 걸쳐 사용한다고 말하는 반면, §3.5에서는 이를 “최종 단계로” 실행한다. 본문은 두 설명을 조정하지 않는다. §3.5 설명에서 선행 단계의 최종 체크포인트들이 교사가 되고, 프롬프트는 그 교사들의 RL 학습 집합에서 뽑는다. 그리고 GRPO advantage를 다음으로 바꾼다.

```
Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ]
```

y_{i,t}는 표본 i의 토큰 t, x는 프롬프트, π_teacher^infer는 추론 엔진에서 계산한 교사의 확률, π_θ^train은 학습 엔진 아래 현재 정책의 확률, sg는 stop-gradient 연산자다. advantage가 더 이상 baseline을 추정하려고 표본 그룹을 필요로 하지 않으므로, 그룹 크기는 1, 배치 크기는 1,024로 둔다(§3.5).

**계산 예시.** 한 토큰에 교사가 0.60, 학생이 0.20의 확률을 부여하면 advantage는 log(0.60 / 0.20) = log 3 = +1.10이고, 업데이트는 그 토큰에 대한 학생의 확률을 높인다. 교사가 0.10, 학생이 0.50을 부여하면 advantage는 log(0.2) = −1.61이고, 업데이트는 학생 확률을 낮춘다. 둘이 일치하는 토큰의 기여는 거의 0이다. 신호는 조밀하다(토큰마다 값 하나). 또한 양·음 양쪽 부호를 가지므로 프롬프트당 표본 하나도 충분한 정보를 담는다.

**기제에 대한 증거이며 GLM-5 사용에 대한 증거는 아니다.** [[glm-5]]는 증류 단계만 분리한 수치를 보고하지 않는다. 가장 가까운 측정 비교는 [[qwen-3-general-rl]] 표 21의 Qwen3-8B다. 같은 off-policy 증류 체크포인트에서 출발했을 때 RL은 17,920 GPU 시간으로 AIME'24 67.6에 도달하고, on-policy 증류는 1,800 GPU 시간으로 74.4에 도달한다. MMLU-Redux는 각각 86.9와 88.3, GPQA-Diamond는 61.3과 63.3이다. 이는 수학·코드 질의와 한 학생 크기라는 한 설정에서의 **결과(단일 연구)**다. [[glm-4-5]]는 같은 계열의 이동을 다르게 쓴다. 세 도메인 전문가를 따로 학습한 뒤 수백만 개 전문가 표본으로 만든 SFT 집합을 통해 한 모델에 증류하고(§3.1), agentic RL에서는 RL과 자기 증류를 교대한다(§3.3.2).

범용 모델을 위한 설계 지점은 다음과 같다. 순차 파이프라인은 이전 각 단계의 체크포인트를 교사로 삼아, 그 단계의 RL을 다시 실행하는 대신 저렴한 회복 단계를 둘 수 있다. 공개 출처가 주지 않는 값은 그 회복량의 크기다.

## §6 같은 RL 단계 안의 안전성과 도움성

안전 데이터는 자체 보상 모델과 자체 간섭 양상을 가진, 같은 혼합 안의 또 다른 도메인이므로 이 장에 속한다. DeepSeek-R1은 안전/위험 라벨이 붙은 106,000개 프롬프트로 별도의 안전 보상 모델을 학습한다. 이는 66,000쌍 도움성 RM 옆의 point-wise loss다(§3.1). 그리고 일반 RL 프롬프트 집합에 무해성 질문 12,000개를 더한다(B.3.1).

OpenAI 보고서 두 편은 보상 설계를 더 자세히 설명한다.

**숙고형 정렬**([[deliberative-alignment]])은 모델에게 안전 명세의 텍스트를 가르치고, 그 명세를 근거로 추론한 결과에 보상한다. SFT 데이터는 문맥 증류(context distillation)로 만든다. 명세를 system prompt에 넣고, 명세도 가진 judge 모델이 생성된 completion을 필터링한다. 그 다음 명세를 프롬프트에서 제거하므로 학습된 모델은 명세를 기억해 내야 한다. RL 단계에서는 같은 judge가 안전 프롬프트의 보상을 제공한다. chain of thought는 judge에 숨긴다. 저자들은 이것이 CoT에 최적화 압력을 가하지 않으므로 기만적인 CoT를 학습하는 일을 피한다고 말한다. 보고된 효과는 다음과 같다. StrongREJECT goodness@0.1은 o1에서 0.88이고 GPT-4o에서는 0.37이며, XSTest non-overrefusal은 각각 0.93과 0.88이다(표 1). SFT 또는 RL 중 어느 쪽에서든 안전 데이터를 빼면 중간 수준의 결과가 나온다(§4.1). CoT 안의 정책-검색 정확도는 hard refusal / safe completion / compliance에서 0.75 / 0.91 / 0.54이고, 학습하지 않은 baseline에서는 0.27 / 0.21 / 0.09다(표 2). 일반화는 가장 강한 주장이다. 안전 데이터에 비영어 예시와 인코딩된 예시가 전혀 없는 모델은 인코딩 jailbreak에서 0.97 ± 0.02, 다국어 jailbreak에서 0.69 ± 0.01을 기록한다. 이는 전체 데이터 모델의 0.95 ± 0.03 및 0.68 ± 0.01과 비교되며, 안전 학습이 전혀 없을 때의 0.65 ± 0.06 및 0.44 ± 0.01보다 높다(표 3).

**Safe-completions**([[openai-safe-completions]])은 보상 형태를 거절 여부 판정에서 두 보상 모델의 곱으로 바꾼다.

```
r_i = h_i · s_i
```

s_i ∈ [0, 1]는 출력의 정책 준수도다(1은 준수, 0은 심각한 위반, 그 사이는 경계선상 저심각도 사례). h_i ∈ [0, 1]는 도움성이다. 하나의 보상 모델이 직접 도움성(명시된 과업을 수행함)과 간접 도움성(대안, 위험 맥락화, 투명한 거절)을 채점하며, 둘 중 하나라도 높으면 h_i는 높다.

**계산 예시.** 정책을 위반하는 상세 답변은 h = 0.9, s = 0이어서 r = 0이다. 정책 안에 머무는 고수준 답변은 h = 0.6, s = 1이어서 r = 0.6이다. 경계선상 답변은 h = 0.9, s = 0.3이어서 r = 0.27이다. 이 순서는 두 번째 응답에 가장 큰 보상을 준다. 보고서가 hard refusal을 대체하는 기제로 제시한 것은 이것이다. s_i가 낮으면 보상을 높이는 유일한 방법은 간접 도움성이다.

보고된 효과는 다음과 같다. 통제된 ablation(아키텍처와 post-training 레시피는 같고 안전 패러다임만 다름)에서 dual-use 프롬프트의 안전성은 개선된다. 안전한 출력을 조건으로 한 도움성은 악의적 프롬프트에서 1–4 척도로 1.0점 넘게 상승한다. 실제 배포 쌍에서는 GPT-5 Thinking이 dual-use 및 악의적 프롬프트에서 o3보다 안전성이 각각 9 및 10 percentage point 높다. biorisk 프롬프트 620개에서 위험한 응답만 놓고 보았을 때, high 또는 moderate harm으로 평가된 비율은 14.7%이고 비교값은 42.7%다(§3.3). 이는 **결과(단일 연구, 공식 보고)**다. 기반 데이터, 보상 모델 크기, RL 설정은 공개되지 않았다.

이 장의 보상 설계를 위한 일반 교훈은 다음과 같다. 곱셈형 게이트(안전성 × 도움성)는 별도 penalty term을 더하지 않고 한 목적을 다른 목적의 hard constraint로 만든다. 그리고 이진 거절 표적과 달리, 제약을 받는 쪽에도 쓸 수 있는 gradient를 남긴다.

## 음성 표본과 음성 피드백

다중 도메인 RL은 이 용어의 네 가지 의미 모두에서 음성 표본을 만든다(유도식은 [[ch-43a]]에 있다).

**음성 표본은 어디서 오는가.** 검증기는 0을 반환한다. 보상 모델은 낮은 스칼라를 반환한다. judge는 루브릭 항목이 충족되지 않았다고 판정한다. 안전 judge는 s = 0을 반환한다. 서식 또는 도구 호출 검사는 보상 0으로 trace를 중단한다([[glm-4-5]] §3.3.2). 이 출처들 중 어느 것도 false-negative 비율을 보고하지 않는다. 가장 가까운 측정은 [[general-reasoner]] 표 5다. 여기서는 모델 기반 검증기가 모든 평가 집합에서 규칙 기반 검증기를 이긴다. 이는 수학 밖에서 규칙 기반 검증기가 정답을 오답으로 표시한다는 간접 증거다.

**실무에서는 이를 어떻게 처리하는가.**

1. *음의 한계 가치(negative marginal value)* — 버린다. [[deepseek-r1]]의 rejection sampling은 800K SFT 집합에 정답 응답만 남긴다(B.3.3). [[glm-4-5]]는 judge agent가 완료로 표시하지 않은 trajectory를 버린다(§3.1).
2. *내용으로서의 음성 표본* — 보존하되 mask한다. [[glm-5]]는 그 밖의 부분은 좋은 agent trajectory 안의 오류 구간을 보존하고 loss에서 mask한다. 이는 “잘못된 행동을 강화하지 않으면서 모델이 오류 수정 행동을 학습하도록” 하기 위함이다(§3.1).
3. *조건부 입력으로서의 음성 표본* — 이 장의 출처들은 사용하지 않는다.
4. *gradient로서의 음성 표본* — group-normalized advantage다. GRPO([[grpo]])에서는 그룹 평균보다 낮은 모든 응답이 음의 advantage를 받는다. 따라서 혼합 단계에서는 해당 프롬프트를 채점한 보상 채널이 음의 gradient를 적용한다.

**기제.** 표본 토큰 y가 음의 advantage를 받을 때 업데이트는 softmax logit gradient `∂ log p_y / ∂ z_j = 1[j = y] − p_j`를 따른다. y에서 빠진 확률은 현재 확률에 비례하여 다른 토큰들에 재분배되므로, 가장 확률이 높은 대안이 그 대부분을 흡수한다. 레시피에는 두 결과가 보인다. 첫째, 이미 드문 행동을 벌점 주는 일은 비효율적이다. [[glm-4-5]]는 병리가 출력의 1% 미만에서 나타나므로 일반 RL이 이를 거의 표본으로 뽑지 못한다고 말한다(§3.4). 둘째, 아래로 누르는 힘의 크기가 중요하다. [[nemotron-cascade]]는 RLHF에서 code-switched 응답의 보상을 배치 최솟값에서 10을 뺀 값으로 두며, 이는 그룹에서 가장 음의 advantage를 보장한다(§4.3.2). 그러나 Code RL에서 같은 팀은 code switching에 −1 벌점을 주면 코딩 성능이 하락함을 발견했고, 대신 0을 썼다. 그룹 내 모든 rollout이 틀리거나 섞여 있을 때 이 벌점 아래 모델이 “code-switching 없이 틀린 답을 생성”했기 때문이다(§4.6.2). 같은 음성 신호가 한 단계에서는 도움이 되고 다른 단계에서는 해가 됐다.

**이 레시피들이 쓰는 제어 장치.** 벌점의 상한을 둔다(code switching에서 −1 대신 0). 빼는 대신 게이트를 쓴다([[openai-safe-completions]]의 r = h · s에서는 s = 0이 특정 토큰 패턴을 밀어 내리지 않고 보상을 없앤다). 양의 항으로 anchor를 둔다([[kimi-k2]]의 PTX loss). 음성 표본을 on-policy로 유지한다([[nemotron-cascade]] §4.1.2의 엄격한 on-policy GRPO). 음성 신호가 양성 신호를 지배하지 못하게 한다(§4.4.2에서 검증기 아래 sigmoid로 bounded된 RM 항). 점수 threshold가 아니라 margin으로 쌍을 고른다([[rlcf-checklists]]는 criterion별 차이가 가장 큰 쌍의 40%를 남긴다).

**진단.** 집계값만이 아니라 도메인별 보상과 pass rate를 기록한다. 분산이 0인 그룹의 비율(gradient 없음)을 기록한다. 단계가 preference 기반이면 chosen과 rejected 점수를 따로 기록한다. 도메인별 entropy와 pass@k를 추적한다([[ch-43]]). 안전 보상은 위험한 출력만 벌점 주면 모델을 거절로 움직이므로, 안전 스위트([[xstest]])에서 과잉 거절과 abstention을 추적한다.

**효과 크기에 대한 정직성.** 이 장의 다중 도메인 출처 중 어느 것도 이 설정에서 개선의 얼마가 업데이트의 음의-gradient 절반에서 왔는지 측정하지 않는다. 따라서 음성 표본에 대한 이 장의 주장은 attribution이 아니라 실패 양상과 제어 장치에 관한 것이다. **열린 질문.**

## 레시피

값은 지정한 위치에서 인용했다. 출처가 출력하지 않은 값은 다른 출처에서 채우지 않았다.

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1 | 671B MoE | RL | 혼합 단계 step; 선호 보상 window | 1,700 step; 일반 데이터와 선호 보상은 마지막 400 step에서만 사용 | arXiv:2501.12948v2 §3.2.2 | 2026-09-14 검증됨([[deepseek-r1-recipe]]) | B.5 그림 6: helpful RM을 쓰면 보상은 오르지만 Codeforces pass@1은 하락 |
| DeepSeek-R1 | 671B MoE | RL | 보상 구성 | Reward = Reward_reasoning (규칙) + Reward_general (RM + 서식) + Reward_language | v2 §3.2.2 식 8-10 | 2026-09-15 검증됨 | ablation 보고 없음 |
| DeepSeek-R1 | 671B MoE | RL | 일반 프롬프트 | 도움성 질문 66k개; 무해성 질문 12,000개 | v2 B.3.1, 표 4 | 2026-09-14 검증됨 | ablation 보고 없음 |
| DeepSeek-R1 safety RM | R1 + reward head | 보상 모델 | 데이터; loss | 안전/위험 라벨 프롬프트 106,000개; point-wise | v2 §3.1 | 2026-09-14 검증됨 | ablation 보고 없음 |
| Qwen3(32B 및 235B-A22B) | 32B, 235B MoE | RL | General RL 과제 범위 | “20개가 넘는 구별된 과제”; 규칙, 참조 기반 judge(Qwen2.5-72B-Instruct), preference RM | arXiv:2505.09388v1 §4.4 | 2026-09-15 검증됨([[qwen-3-general-rl]]) | §4.7 표 22: Qwen3-32B의 단계별 점수 |
| Qwen3-8B | 8B | distill-SFT | 같은 checkpoint에서 시작한 on-policy distillation 대 RL | AIME'24 74.4 대 67.6; GPU 시간 1,800 대 17,920 | §4.7 표 21 | 2026-09-15 검증됨 | 표 21이 ablation |
| GLM-5 | 총 744B / 활성 40B | RL | reasoning-RL 알고리즘 설정 | GRPO + IcePop, KL 제거; β = 2; ε_low 0.2; ε_high 0.28; 그룹 32; 배치 32; 완전 on-policy | arXiv:2602.15763v2 §3.2 | 2026-09-15 검증됨([[glm-5]]) | ablation 보고 없음 |
| GLM-5 | 744B / 활성 40B | RL | general-RL 보상 종류 | 규칙 기반 함수 + outcome RM + 생성형 RM; 사람이 쓴 응답을 style anchor로 사용; 가중치는 미출력 | §3.4 | 2026-09-15 검증됨 | ablation 보고 없음 |
| GLM-5 | 744B / 활성 40B | distill-SFT | 단계 간 on-policy distillation | 교사 = 선행 단계의 최종 checkpoint; advantage = sg[log(π_teacher/π_θ)]; 그룹 크기 1; 배치 1,024 | §3.5 식 2 | 2026-09-15 검증됨 | ablation 보고 없음 |
| GLM-4.5 | 355B / 활성 32B | RL | holistic general-RL 프롬프트 | 7개 주 분류, 33개 보조 분류, 139개 3차 분류에 걸친 약 5,000개 프롬프트; 인간 선호와 AI 루브릭 점수로 학습한 RM | arXiv:2508.06471v1 §3.4 | 2026-09-14 검증됨([[glm-4-5-recipe]]) | ablation 보고 없음 |
| GLM-4.5 | 355B / 활성 32B | RL | instruction-following RL | 7개 대분류 및 151개 소분류 제약 유형; 규칙 + RM + critique 모델 | §3.4, 그림 9 | 2026-09-14 검증됨 | 약 1,000 step까지 뚜렷한 reward hacking 없음 |
| Kimi-K2-Instruct | 1.04T / 활성 32B | RL | 검증 불가 보상 | 자기비평 루브릭 보상: K2 critic이 core·prescriptive·human rubric에 대해 쌍의 순위를 매김; 검증 가능 보상 rollout으로 critic 갱신 | arXiv:2507.20534v2 §3.2.2 | 2026-09-14 검증됨([[kimi-k2-recipe]]) | App. F.3은 과신을 부작용으로 열거; ablation 없음 |
| Nemotron-Cascade 8B / 14B | 8B, 14B | 보상 모델 | RM base; 데이터; 학습 | Qwen2.5-72B-Instruct + 선형 head, Bradley-Terry; 82K 쌍(HelpSteer2 10K, filtering 뒤 HelpSteer3 36K, 생성 쌍 추가); 배치 256; LR 2e-6; 1 epoch | arXiv:2512.13607v2 §4.2 | 2026-09-15 검증됨([[nemotron-cascade]]) | §6.2: 72B RM은 7B RM보다 AIME25 약 3% 높고 응답 길이가 더 안정적 |
| Nemotron-Cascade 8B / 14B | 8B, 14B | RL | RLHF 설정 | 최대 응답 12K, 과도한 길이 filtering 없음; rollout 8개; temperature 0.6; top-p 0.95; LR 2e-6; entropy 및 KL 계수 0; 약 800 step | §4.3.2 | 2026-09-15 검증됨 | §4.3.3, 표 4: IFEval 외 모든 benchmark에서 이득 |
| Nemotron-Cascade 8B / 14B | 8B, 14B | RL | RLHF 배치 크기; step | 본문: 배치 128, 약 800 step; App. 표 15: 배치 256, 800(8B) 및 900(14B) step | §4.3.2 대 표 15 | 충돌 | 어느 값이 릴리스 checkpoint를 만들었는지 보고서가 말하지 않음 |
| Nemotron-Cascade 14B-Thinking | 14B | RL | IF-RL 보상 | R_IF = 1이면 r_i = R_IF + sigmoid(R̂_RM), 아니면 0; R̂_RM은 group-normalized | §4.4.2 | 2026-09-15 검증됨 | §4.4.2: 규칙 기반 검증기만으로는 human-alignment 결과가 저하(수치는 없음) |
| Nemotron-Cascade 14B-Thinking | 14B | RL | Math RL 단계 | 최대 응답 28K → 40K; LR 2.5e-6; temperature 1.2 → 1.1; step 0-120 및 120-220; rollout 8개; 배치 128 | App. 표 18 | 2026-09-15 검증됨 | 표 6: AIME 2024 89.2 → 90.4 |
| Nemotron-Cascade 14B-Thinking | 14B | RL | Code RL | 최대 응답 56K; LR 4e-6; 64 step; temperature 1.0; code-switch 보상 0(−1 아님) | App. 표 19; §4.6.2 | 2026-09-15 검증됨 | §4.6.2: −1 code-switch 벌점은 코딩 성능을 낮춤 |
| GURU-7B / GURU-32B | 7B, 32B | RL | 혼합 도메인 GRPO | LR 1e-6, 10-step warm-up; step당 프롬프트 512개; temperature 1.0에서 프롬프트당 표본 16개; mini-batch 64; 프롬프트 4K / 생성 8K; KL 없음, entropy loss 없음; ε = 0.2 | arXiv:2506.14965v1 §4.1 | 2026-09-15 검증됨([[guru-cross-domain-rl]]) | 표 3: 17개 benchmark 평균 43.29 및 54.24 |
| Nemotron-CrossThink-7B | 7B | RL | 도메인 blend | 범용 추론 대 수학 2:1 | arXiv:2504.13941v3 §5, 표 3 | 2026-09-15 검증됨([[nemotron-crossthink]]) | 표 3: 평균 58.12, 수학 전용 57.82 및 범용 전용 53.30 대비 |
| Qwen2.5-7B에서 출발한 RaR policy | 7B | RL | 루브릭 보상 | GRPO; 프롬프트당 표본 16개; LR 5e-6; 300 step; judge gpt-4o-mini; 가중치 Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9 | arXiv:2507.17746v2 §4.2, §4.4, 표 10 | 2026-09-14 검증됨([[rubrics-as-rewards]]) | 그림 2: HealthBench 31.2, Likert 보상 25.5 대비 |
| Qwen2.5-7B-Instruct에서 출발한 RLCF | 7B | preference | 체크리스트 피드백 | Qwen2.5-72B-Instruct 체크리스트; 항목당 judge 표본 25개의 평균; criterion별 차이로 쌍 상위 40% 유지; DPO 2 epoch, 배치 1024, cosine LR 3e-6 → 2e-6 | arXiv:2507.18624v2 §3, §4.1 | 2026-09-15 검증됨([[rlcf-checklists]]) | 표 2-4: 다섯 benchmark 모두 양의 결과; RM baseline은 혼재 |
| DeepSeek-GRM-27B | 27B | 보상 모델 | SPCT | RFT LR 5e-6, 배치 1024; RL LR 4e-7, 배치 512; 각각 900 step; KL β = 0.08; N_RFT = 3 | arXiv:2504.02495v3 App. C.1 | 2026-09-15 검증됨([[deepseek-grm]]) | App. D: β grid {0, 0.01, 0.02, 0.08}; 더 작은 값은 일부 subset에서 붕괴 |

**소규모 범용 실행의 시작점.** 여기의 모든 수치는 위 `검증됨` 행과 그 수치가 쓰인 조건에서 가져왔다. 혼합 추론 도메인으로 학습하는 7B급 policy에는, 공개된 다중 도메인 결과를 가진 [[guru-cross-domain-rl]] 설정이 있다. GRPO, LR 1e-6, step당 512개 프롬프트, temperature 1.0에서 프롬프트당 16개 표본, 생성 한도 8K, KL과 entropy 항 없음, ε = 0.2이며, 7B에서 8개 Hopper GPU의 20개 node로 약 3일을 쓴다. 같은 크기에서 그 위에 얹는 일반(검증 불가) 단계에는 [[rubrics-as-rewards]]가 참조 답에 맞춰 생성한 루브릭과 gpt-4o-mini judge로, GRPO·프롬프트당 표본 16개·LR 5e-6·300 step을 H100 8장 노드 하나에서 썼다. 두 설정 중 어느 것도 상대의 데이터에 맞춰 조정하지 않았으며, 이 규모에서의 공동 설정을 보고한 출처는 없다.

## 일반화 관점

**(a) 폭을 넓히는 것.**
- 모델에 요청할 도메인에서 온 프롬프트. 과소 대표된 도메인은 도메인 내부 데이터로 이득을 보며 도메인 간 데이터로는 그렇지 않다([[guru-cross-domain-rl]] §3.1). 수학+퍼즐 혼합에 코드 데이터를 넣으면 코드 22.56점 붕괴가 사라진다([[multi-domain-rlvr-data-centric]] 표 9).
- 이례적인 형식의 정답도 받아들이는 검증기. 모델 기반 검증기는 네 평가 집합 모두에서 규칙 matching을 이기며 step 60에서 plateau하지 않는다([[general-reasoner]] 표 5, 그림 4).
- 하나의 전역 품질 점수 대신 instance별 루브릭 또는 체크리스트([[rubrics-as-rewards]] 그림 2; [[rlcf-checklists]] 표 2-4).
- 제약 만족만이 유일한 목표가 되지 않게 검증기 항 옆에 품질 항을 둠([[ifbench]] App. E; [[nemotron-cascade]] §4.4.2).
- 작성된 명세를 근거로 추론하는 것으로 안전을 학습함. 이는 안전 학습 데이터에 없던 jailbreak 형식으로 전이됐다([[deliberative-alignment]] 표 3).
- 한 모델 안의 mode coverage: thinking과 non-thinking 프롬프트로 RLHF batch를 반씩 나눈 것이 보고서 자체 비교에서 ArenaHard·수학·코드에 가장 높은 점수를 냈다([[nemotron-cascade]] §6.1).

**(b) 좁아짐 또는 망각을 일으키는 것.**
- 선호 보상 모델을 여러 step 동안 최적화함: 보상은 오르지만 held-out 능력은 내려간다([[deepseek-r1]] B.5, 그림 6). 그래서 선호 window는 1,700 step 중 400 step이다.
- 좁은 제약 분류체계에서 규칙 전용 검증기를 씀: AlpacaEval 2 33.5 → 21.3([[ifbench]] 표 3).
- 한 도메인 안에서 공격적인 난이도 filtering: 학습 중 정확도 붕괴와 함께 HumanEval −9.2, HiTab −3.0([[guru-cross-domain-rl]] 표 2).
- 단일 도메인 RL: 수학 전용 학습은 코드를 67.46에서 64.23으로, CountDown 전용 학습은 29.59로 낮춘다([[multi-domain-rlvr-data-centric]] §3.1).
- 추론 지향 SFT와 RL은 16개 통제 변형 중 15개에서 제약 따르기를 낮추며([[scaling-reasoning-losing-control-mathif]] 표 4), general RL은 최고 추론을 낮춘다([[qwen-3-general-rl]] 표 22).
- 한 능력을 무시하는 프롬프트 분포: assistant와 writing 프롬프트의 RLCF는 GSM8K를 83.2 → 82.2, TruthfulQA MC1을 43.5 → 42.0으로 움직인다([[rlcf-checklists]] 표 9). GURU-7B는 IFEval에서 두 개 더 좁은 baseline보다 낮다([[guru-cross-domain-rl]] 표 3).
- 표면 패턴을 금지하는 루브릭은 그 패턴의 정당한 사용도 금지한다. 예컨대 hedging 벌점은 과도하게 단정적인 확신으로 이어진다([[kimi-k2]] App. F.3).

**(c) 이 단계를 위해 측정하는 방법.**
- 집계 보상만이 아니라 학습 중 도메인별 보상과 도메인별 정확도.
- 프롬프트 혼합에 *없는* 도메인을 포함하는 held-out suite. [[guru-cross-domain-rl]]은 IFEval과 LiveBench를 unseen 도메인으로 평가하며, [[rlcf-checklists]]는 XSTest·GSM8K·TruthfulQA를 평가한다.
- 고정 평가 cadence와 명시적 checkpoint 규칙: 10 step마다 online 과제 13개를 돌리고 최고 평균 checkpoint를 고른다([[guru-cross-domain-rl]] §3.1). 매 단계 뒤 모든 benchmark를 sweep하는 것은 순차 파이프라인 버전이다([[nemotron-cascade]], 표 2-8).
- judge 독립성: [[rubrics-as-rewards]]는 학습 보상과 평가 judge에 모두 gpt-4o-mini를 써서 judge 일치와 과제 품질을 confound한다. judge가 학습 보상을 공급하면 다른 judge 또는 프로그램 지표로 해당 단계의 평가를 보고한다.
- 고정 평균 관례와 그 옆의 도메인별 수치(§3.4), 그리고 비교 checkpoint 전반의 고정 chat template([[multi-domain-rlvr-data-centric]] 표 11).
- 혼합을 결정하는 benchmark의 contamination 제어: 수학 RL 데이터의 9-gram filtering 및 학습 cutoff 뒤에 공개된 문제만으로 평가([[nemotron-cascade]] §4.5.1, §5).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| 혼합 도메인 단계에 하나의 집계 보상 곡선만 사용 | 전체 보상은 오르지만 한 도메인의 정확도는 떨어지며, 곡선에서는 원인이 보이지 않음 | 데이터 출처별 보상과 pass rate를 기록하고 단계 전후 도메인별 평가를 비교 |
| instruction following에 검증기 전용 보상 | 제약 지표는 오르지만 개방형 품질 지표는 하락(IFBench의 AlpacaEval 2 33.5 → 21.3) | in-loop suite에 일반 품질 평가를 넣고 제약을 뺀 응답을 점검 |
| policy 출력에서 멀리 벗어난 보상 모델 | 보상과 함께 응답 길이가 늘며 style-controlled 점수가 비통제 점수보다 뒤처짐([[nemotron-cascade]] §6.2) | 보상 대비 평균 응답 길이를 추적하고 style-controlled ArenaHard를 보고하며 더 큰 RM을 사용 |
| 선호 보상을 단계 전체에 적용 | 보상은 꾸준히 오르지만 검증 가능한 benchmark는 하락([[deepseek-r1]] 그림 6) | 선호 보상 window를 제한하고 검증 가능 프롬프트와 교대하며 코드 또는 수학 평가를 감시 |
| 하나의 평균으로 혼합을 비교 | 다른 평균 관례 아래 두 혼합의 순위가 뒤바뀜(§3.4) | 관례를 고정하고 도메인별 수치와 도메인당 benchmark 수를 공개 |
| 서로 다른 chat template으로 checkpoint 평가 | 일부 과제에서 설명되지 않는 큰 하락([[multi-domain-rlvr-data-centric]] 표 11) | 모델 family당 template 하나를 고정하고 결론 전 template-matched control 실행 |
| 모든 도메인에 기본으로 난이도 filtering | 도메인 내부 이득과 쉬운 도메인 외 과제의 붕괴([[guru-cross-domain-rl]] §3.3) | 난이도 분포를 남기거나 도메인 간 프롬프트를 더하고 학습 중 쉬운 held-out 과제를 감시 |
| 일반 RL 안에서 드문 병리에 벌점 | 측정 가능한 변화가 없고 행동은 rollout의 1% 미만에서 나타남([[glm-4-5]] §3.4) | 먼저 병리 비율을 측정하고, 드물면 선별 프롬프트 또는 SFT 데이터로 겨냥 |
| 표면 속성에 큰 음성 보상 | 속성은 사라지지만 해당 도메인 정확도가 하락([[nemotron-cascade]] §4.6.2) | 하나를 채택하기 전 held-out split에서 벌점 크기(0 대 −1)를 비교 |
| 학습 보상 judge = 평가 judge | judge의 지표에서는 이득이 보이나 독립 지표에서는 보이지 않음([[rubrics-as-rewards]] 그림 2 캡션) | 다른 judge 또는 프로그램 지표로 평가 |

## 이해 점검

1. [[nemotron-crossthink]]에서 최선의 혼합 blend는 평균에서 수학 전용 blend를 0.30점 이기지만 AMC 23에서는 5.0점을 잃는다. 범용 모델을 만드는 연구소가 그래도 혼합 blend를 택할 이유를 설명하고, 그 결정을 바꾸게 할 평가 결과를 하나 제시하라.
2. [[nemotron-cascade]]는 RLHF 단계에서 수학 및 경쟁 프로그래밍 프롬프트를 제거한다. 보상 모델의 성질로부터 막으려는 실패를 유도하고, 그 프롬프트를 남겼다면 AIME25 점수에 무엇이 일어날지 예측하라.
3. IF-RL 보상은 검증기가 통과할 때 `R_IF + sigmoid(R̂_RM)`이고, 그렇지 않으면 0이다. sigmoid와 group normalization이 둘 다 필요한 이유와 RM 점수를 bound 없이 더하면 무엇이 망가지는지 설명하라.
4. [[guru-cross-domain-rl]]은 난이도 filtering이 도메인 내부에는 도움되고 쉬운 도메인 외 과제에는 해가 된다고 찾는다. 학습 중 policy가 무엇을 표본으로 뽑는가로 그 비대칭의 기제를 제시하고, 이를 검증하려면 무엇을 기록할지 말하라.
5. GLM-5는 최종 단계에서 group-normalized advantage를 `sg[log(π_teacher/π_θ)]`로 바꾼다. 이 변경이 왜 그룹 크기 1을 허용하는지, 그리고 같은 대체가 reasoning RL 단계 자체에는 왜 나쁜 선택인지 설명하라.
6. Safe-completions는 위험한 내용에 벌점을 빼는 대신 도움성에 안전성을 곱한다. 응답이 매우 도움이 되지만 중간 정도로 위험할 때, 두 설계 각각에서 policy가 무엇을 향해 밀리는지 단계별로 설명하라.
7. [[rubrics-as-rewards]]는 학습 보상과 평가에 같은 judge를 쓴다. “policy가 과제에서 실제로 나아졌다”와 “policy가 이 judge에 동의하는 능력이 나아졌다”를 분리하는 실험을 설명하라.
8. Qwen3는 general-RL 이득과 맞바꾸어 AIME'24 2.4점 손실을 받아들였고, Qwen은 뒤에 하이브리드 모델을 별도 Instruct와 Thinking checkpoint로 분리했다. 그 분리가 그 trade-off에 대한 올바른 대응이었는지 결정하려면 어떤 증거가 필요한지 말하라.

## 연결

- 이전 장: [[ch-44a]] — RL의 길이: 과도하게 긴 응답, 길이 제어, Long-Context RL. 그 장의 길이 예산과 과도한 길이 filtering 선택은 다중 도메인 단계가 도메인마다 한 번씩 정해야 하는 값이다.
- 의존 장: [[ch-38a]] — SFT 대 RL 일반화: On-Policy 데이터, Base 모델로의 KL, 출력 다양성. §5.1의 망각 논증은 그 장 결과의 다중 도메인 사례다.
- 다음 장: [[ch-45]] — 자기 개선 루프와 다단계 추론 파이프라인. §5.3의 expert-model 및 self-distillation 이동은 그 루프의 진입점이다.
- [[ch-41]] — Reward Modeling: Bradley–Terry, 과최적화, 보상 모델 일반화: 여기에서 하나의 채널로 쓰이는 보상 모델.
- [[ch-42]] — Reward Hacking과 Judge 설계: 루브릭·체크리스트·critic 보상이 겨냥하는 실패 양상.
- [[ch-43]] — Entropy, 출력 다양성, RL의 KL 제어: 이 장의 음성 피드백 절에서 이름을 든 다양성 진단인 entropy와 pass@k.
- [[ch-43a]] — 음성 표본과 음성 gradient: Likelihood Displacement, Squeezing, 음의 Advantage: 이 장의 음성 피드백 절 뒤에 있는 유도식.
- [[ch-46]] — Lab: 음성 신호 ablation과 held-out 능력 보존을 갖춘 DPO 또는 RLVR 실험: 여기서 설계한 held-out 보존 측정을 그곳에서 실행한다.

## 출처

- [[deepseek-r1]], [[deepseek-r1-recipe]] — 혼합 최종 RL 단계: 보상 구성(식 8-10), 400-step 선호 window, helpful 및 safety 보상 모델, reward-hacking 근거.
- [[qwen-3]], [[qwen-3-general-rl]] — 20개 초과 과제의 General RL, 세 보상 유형, Qwen3-32B 단계별 표, on-policy distillation 비교.
- [[glm-4-5]], [[glm-4-5-recipe]] — expert-model 반복과 증류, holistic 및 instruction-following general RL, 병리의 희소성.
- [[glm-5]] — 순차 reasoning / agentic / general RL, 세 부분 hybrid 보상 시스템, 단계 간 on-policy distillation 목적함수.
- [[kimi-k2]], [[kimi-k2-recipe]] — 자기비평 루브릭 보상, closed-loop critic 갱신, PTX loss, 명시된 과신 부작용.
- [[nemotron-cascade]] — cascade 파이프라인, 단계별 benchmark 표, 결합된 IF 및 RM 보상, RM 크기 연구, mode-split ablation.
- [[rubrics-as-rewards]] — 루브릭 보상: 집계 공식, 가중치, HealthBench 및 GPQA 결과, judge-alignment 연구.
- [[rlcf-checklists]] — 체크리스트 피드백: 생성, judge+프로그램 점수화, DPO 쌍, benchmark 결과, target 밖 과제의 회귀.
- [[deepseek-grm]] — SPCT, 범용 생성형 보상 모델의 추론 시간 scaling, 온라인 RL 보상으로서의 한계.
- [[guru-cross-domain-rl]] — 도메인 간 전이 비대칭, 난이도 filtering ablation, 17-benchmark 결과, pass@k 분석.
- [[general-reasoner]] — 다양한 학습 데이터 대 수학 전용 학습 데이터, 모델 기반 대 규칙 기반 검증기 ablation.
- [[nemotron-crossthink]] — blend 비율, 수학 전용 학습과의 benchmark별 분할, token 효율 측정.
- [[multi-domain-rlvr-data-centric]] — 하나의 base 모델에서의 일곱 도메인 조합 전체, template 민감도 표.
- [[scaling-reasoning-losing-control-mathif]] — MathIF 및 추론 학습이 제약 따르기를 낮춤을 보인 통제 실행.
- [[qwen3-2507-instruct-thinking-split]] — 하이브리드 thinking mode를 끝낸 공식 설명과 model-card 표.
- [[deliberative-alignment]] — 명세 기반 안전 보상, 단계 ablation, policy retrieval, out-of-distribution 일반화.
- [[openai-safe-completions]] — r = h · s 보상, 통제 및 production 비교, harm 심각도 분석.
- [[ifbench]] — 검증 가능한 instruction following, 제약 일반화 ablation, 보상 모델 혼합 결과.
- [[grpo]] — 이 장의 거의 모든 레시피가 쓰는 group-normalized advantage.
- [[reward-model-overoptimization]], [[xstest]] — RM 크기 및 과잉 거절 진단의 배경.
