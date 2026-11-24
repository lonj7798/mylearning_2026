<!-- chapter: ch-44
     track: rl
     kind: content
     title: Process Supervision and Verifiable Rewards
     deps: [ch-43a]
     sources: [[prm800k]], [[let-verify]], [[lets-verify]], [[math-shepherd]], [[omegaprm]], [[qwen-prm-lessons]], [[rl-on-incorrect-synthetic-data]], [[rewarding-progress-pav]], [[rlvr-tulu3-config]], [[ifbench]], [[why-language-models-hallucinate]], [[swe-rl-difflib-reward]], [[deepswe]], [[deepswe-recipe]], [[crossing-the-reward-bridge]], [[transferability-of-llm-reasoning]], [[deepseek-r1]], [[rlvr-beyond-base-model]], [[prorl]], [[spurious-rewards-rlvr]], [[training-verifiers-to-solve-math-word-problems]], [[reward-model-overoptimization]], [[metr-frontier-reward-hacking]], [[natural-emergent-misalignment-reward-hacking]], [[glm-4-5]], [[kimi-k2]], [[deepseek-v3.1]], [[step-dpo]], [[grpo]], [[john-schulman-kl-tricks]], [[rubrics-as-rewards]]
     figures: figures/prm-vs-orm.html, figures/wrong-answer-rewards.html
     excerpts: excerpts/qwen-prm-lessons.md, excerpts/rewarding-progress-pav.md, excerpts/rl-on-incorrect-synthetic-data.md, excerpts/why-language-models-hallucinate.md, excerpts/crossing-the-reward-bridge.md, excerpts/rlvr-tulu3-config.md, excerpts/swe-rl-difflib-reward.md
     revised: 2026-09 (generality revision)
-->

# Chapter 44 — Process Supervision과 Verifiable Reward

> **핵심 통찰.** Verifier와 step label은 correctness의 estimator이며 policy는 task뿐 아니라 그 error에도 훈련된다. 12K MATH problem의 human step label은 best-of-1860에서 held-out MATH problem 500개 중 correct solution을 78.2% 선택하는 RM을 만들었다. Outcome RM은 72.4%, majority voting은 69.6%다([[prm800k]] §3, Fig. 3). Label을 Monte-Carlo rollout estimate로 바꾸면 두 평가는 반대 방향으로 변한다. MC-label PRM은 ProcessBench 평균 F1 40.1로 PRM800K PRM 56.5보다 낮지만, Best-of-8에서는 65.9로 64.9보다 높다([[qwen-prm-lessons]] Tables 3-4). Outcome 쪽에서 deterministic verifier를 reward로 쓰면 Tülu 3 8B는 GSM8K 84.3→87.6, MATH 42.0→43.7, IFEval 81.1→82.4가 됐고([[rlvr-tulu3-config]] Table 23), 같은 논문은 KL penalty를 낮추면 RLVR 내부 over-optimization도 보고한다([[rlvr-tulu3-config]] §6.2.1).
>
> **지침.** Programmatic check와 reference answer가 있는 task에서는 그 check를 reward로 쓰고 KL penalty를 유지하라. Tülu 3 RLVR은 8B에서 targeted evaluation 셋을 개선했지만 Table 23 benchmark 평균은 64.4→64.8이므로 새 capability가 아니라 몇 point 이득을 예상해야 한다. Step signal이 필요하면 human, LLM-judge, judge agreement로 filter한 MC label처럼 step benchmark가 검증하는 label을 선호하라. MC-only label은 first-error identification에서 가장 낮기 때문이다([[qwen-prm-lessons]] Table 4). Step signal을 ranker가 아니라 policy에 쓸 경우 policy와 다른 prover 아래 per-step advantage를 쓴다. Policy 자신 아래 advantage는 outcome-only RL과 같은 gradient를 주기 때문이다([[rewarding-progress-pav]] §3.2). Reference가 free-form이고 rule이 check할 수 없으면 trained generative verifier를 쓴다. Rule-based reward는 training data 20k→100k에서 multi-subject accuracy 26.2→16.9로 떨어진 반면 7B model-based verifier는 30.8→35.0으로 올랐다([[crossing-the-reward-bridge]] Table 2).

---

## 이 장이 범용 모델에 중요한 이유

Pipeline 위치는 pre-training → mid-training → SFT → preference optimization → **RL** → evaluation이다. ch-37~ch-40은 estimator를, ch-41은 learned RM을, ch-42는 reward hacking을, ch-43은 entropy/KL을, ch-43a는 negative gradient가 분포에 하는 일을 다뤘다. 이 장은 scalar가 learned preference model이 아닌 step label 또는 답을 검사하는 program에서 올 때를 다룬다.

세 측정 가능한 문제가 따른다.

1. **Credit assignment.** Outcome label은 긴 trace에 관해 한 bit만 준다. Hard problem에서는 sample solution 대부분에 error가 있어 negative outcome label은 error 위치 정보를 거의 주지 않는다([[prm800k]] §6.1).
2. **Instrument error.** Rollout으로 추정한 step label, unit test, string match는 일부 case에서 틀린다. False positive가 policy가 훈련받아 생성하는 대상이다. Correct-answer sample의 Qwen manual annotation은 process-error rate가 GSM8K 5.1%에서 Omni-MATH 43.4%로 오른다고 찾았다([[qwen-prm-lessons]] Fig. 6).
3. **폭.** Verifier는 좁은 task 집합에만 존재한다. 그 집합 training은 다른 capability를 개선(SWE-RL, §7)하거나 좁힐 수 있다(IFEval-style constraint training, §7). 어느 쪽인지는 가정하지 않고 측정한다.

Section은 outcome supervision과 limit(§1), automated step label(§2), label이 실제로 측정하는 것(§3), policy용 step signal(§4), RLVR(§5), execution-grounded agentic RLVR(§6), verifiable domain 밖 transfer(§7)다.

---

## §1 Outcome supervision과 credit-assignment 한계

### §1.1 정의

**Outcome reward model(ORM)**은 complete `(problem, solution)` pair에 final answer correct 확률을 점수화하는 모델이다. **Process reward model(PRM)**은 solution 각 step을 점수화한다. RLVR 의미의 **verifier**는 모델이 아니라 completion을 reference와 비교하여 숫자를 반환하는 program이다.

측정 가능한 문제는 selection이다. Cobbe et al.은 ORM으로 sample GSM8K solution의 순위를 매기고, 많은 solution을 sample해 rank하면 같은 model size의 fine-tuning alone보다 낫다고 보고했다([[training-verifiers-to-solve-math-word-problems]], arXiv:2110.14168v2 §4의 해당 위치 인용).

### §1.2 PRM800K protocol

[[prm800k]]는 모든 뒤 방법이 근사하는 label protocol을 고정한다.

1. Generator는 newline-delimited step을 생성하도록 correct answer가 있는 few-shot-generated MATH solution에 1 epoch fine-tuning한다(§2.3). RL로 훈련하지 않는다(§2.1).
2. Labeler는 각 step을 positive, negative, neutral로 표시한다. Phase 2에서는 solution label이 첫 negative step에서 멈춘다(§2.4, App. D).
3. PRM은 각 step 마지막 token 뒤 token 하나를 예측하며 세 class의 label token log-likelihood를 최대화해 훈련한다(§2.6, App. F.1).
4. Solution 점수는 step별 P(positive)의 **곱**이며 neutral은 positive로 센다(§2.6, App. F.2).
5. Data collection은 반복적이다. 10 generation 각각에서 current PRM이 problem당 N개 solution 순위를 매기고, 최고 점수 **wrong-answer** solution이 labeler에게 가며 PRM을 재훈련한다(App. B).

규모는 12K problem의 75K solution에 대한 800K step label이다(§2.4). Label된 모든 step의 73.1%는 positive이고 label된 solution의 14.2%는 correct answer로 끝난다(App. B, Table 3). 위 active-learning selection은 small-scale run에서 uniform labeling보다 data efficiency 약 2.6x로 추정된다(§4.2, Fig. 4a).

**Inference 비용.** PRM의 step별 prediction은 forward pass 하나에서 읽는다. “To determine the step-level predictions at test time, it suffices to perform a single PRM forward pass over the whole solution”(§2.6). 따라서 best-of-N에서 PRM은 ORM보다 L배 비싸지 않다. 둘 다 candidate solution당 forward pass 하나다.

### §1.3 결과와 조건

| Selector, held-out MATH problem 500개, best-of-1860 | Accuracy |
|---|---|
| Majority voting | 69.6 |
| ORM(problem당 uniform sample 100개 training data) | 72.4 |
| PRM(PRM800K) | 78.2 |

PRM은 모든 test N에서 ORM 위이고 gap은 N과 함께 커진다(§3, Fig. 3). Training set은 match되지 않았다. ORM은 problem당 uniform sample 100개, PRM은 PRM800K로 훈련했다([[prm800k]] Verification section). Best-of-100 held-out STEM set(AP Calculus, AP Chemistry, AP Physics, AMC10/12)에서도 순서가 유지된다. PRM 72.9, ORM 63.8, majority 61.3이다(§5, Table 1). 저자들은 mathematics 밖 generalization 범위를 모른다고 명시한다(§6.2).

### §1.4 계산 예시: product 대 minimum

PRM step별 P(positive)가 다음인 solution 두 개를 둔다.

- Solution A, step 5개: 0.95, 0.90, 0.85, **0.60**, 0.98. Product = 0.4273, minimum = 0.60.
- Solution B, step 9개가 모두 0.90. Product = 0.9⁹ = 0.3874, minimum = 0.90.

Product는 A를 B보다 위에 두고 minimum은 B를 위에 둔다. 두 rule은 systematic하게 다르다. Product는 추가 step마다 penalty를 주며 논문은 “a slight bias against solutions with more steps”를 기록한다(App. F.2). 논문 자체 best-of-1860 comparison은 neutral을 positive로 센 product 78.2, minimum 77.6이다(App. F.2, Table 4). [[math-shepherd]]은 Lightman et al.을 따라 minimum을 채택하며 aggregator를 비교하지 않는다(§3.4). 이 장 §3.5는 올바른 aggregator가 label 생성 방식에 달렸음을 보인다.

Open [`figures/prm-vs-orm.html`](figures/prm-vs-orm.html) panel B에서 step score를 바꾸면 product, minimum, last-step scoring이 두 solution 순위를 어떻게 바꾸는지 볼 수 있다.

---

## §2 자동 단계 레이블: 몬테카를로 추정

### §2.1 정의와 작동 방식

사람이 매긴 단계 레이블을 수백만 문제 규모로 확장할 수는 없다. [[math-shepherd]]는 레이블러를 **완성기(completer)** 정책으로 대체하고, 그 단계에서 표본 추출한 완성이 정답에 도달하는지에 따라 단계의 레이블을 정의한다.

해결책의 단계 `s_i`에 대해 최종 답 `a_1 … a_N`을 갖는 continuation `N`개를 표본 추출한다고 하자. `a*`는 정답(gold answer)이고, `I(·)`는 조건이 성립할 때 1, 그렇지 않을 때 0이다.

```
하드 레이블   y_HE(s_i) = 어떤 a_j = a*이면 1, 그렇지 않으면 0                 (식 3)
소프트 레이블 y_SE(s_i) = (1/N) Σ_j I(a_j = a*)                                (식 4)
```

- `s_i`: i번째 단계이며, prefix `s_1 … s_i`가 표본 추출의 조건이다.
- `N`: 단계당 completion 수이다. Math-Shepherd 실험은 `N = 8`인 LLemma-7B completer를 쓴다(§4).
- 실험에서는 하드 레이블로 PRM을 훈련한다. 표준 LM 파이프라인을 사용할 수 있도록 하드 레이블을 “has potential”과 “no potential”을 나타내는 두 특수 토큰으로 쓴다(§4).

PRM 손실은 단계별 binary cross-entropy(식 2)이다. 추론 때 해결책의 점수는 단계 점수들의 최솟값이다(§3.4). RL에서는 Math-Shepherd가 각 단계 끝 토큰 위치에 PRM 점수를 주고, 다른 위치에는 0을 준다. 별도의 최종 답 항은 없다(§3.5; ACL 식 7).

### §2.2 예시: 레이블과 두 오류율

한 단계를 따른 completion이 `N = 8`개이고, 그중 3개가 정답에 도달한다고 하자. 소프트 레이블은 3/8 = 0.375이고, 하드 레이블은 1이다.

이제 이 레이블을 추정량으로 보자. **올바른** prefix에서 뽑은 completion 하나가 정답에 도달할 확률을 `p`라고 하고, **잘못된** prefix에서도 completion 하나가 회복하거나, 추측하거나, 산술적으로 우연히 맞아서 정답에 도달할 확률을 `q`라고 하자.

```
P(올바른 단계에서 false negative) = (1 − p)^N
P(잘못된 단계에서 false positive) = 1 − (1 − q)^N
```

`p = 0.15`, `N = 8`이면 0.85⁸ = 0.272이므로, 어려운 문제의 올바른 단계 중 27%가 negative로 레이블된다. `q = 0.10`, `N = 8`이면 1 − 0.9⁸ = 0.570이므로, 잘못된 단계 중 57%가 positive로 레이블된다. `N`을 늘리면 첫 번째 비율은 낮아지고 두 번째 비율은 높아진다. 이것이 Math-Shepherd 측정 결과의 산술적 배경이다. 사람 주석이 있는 GSM8K 단계 160개에서 LLaMA2-70B completer가 만든 하드 레이블은 `N = 4`일 때 86% 정확하고, 더 큰 `N`에서는 정확도가 낮아진다. 저자들은 이를 false positive 탓으로 본다(§5.2, 그림 4a). 또한 이것이 약한 completer가 noisy label을 만드는 이유다. completer가 문제를 풀 수 없으면 모든 단계에서 `p`가 작기 때문이다.

[`figures/prm-vs-orm.html`](figures/prm-vs-orm.html)의 패널 A는 `N`, `p`, `q`가 변할 때 두 오류율을 모두 그린다.

### §2.3 Math-Shepherd의 근거

Greedy accuracy, GSM8K / MATH(Table 2)는 다음과 같다. Mistral-7B는 77.9 / 28.6, +rejection fine-tuning은 79.0 / 29.9, +ORM을 쓴 PPO는 81.8 / 31.3, +PRM을 쓴 step-by-step PPO는 84.1 / 33.0이다. 그 PPO로 훈련된 Mistral-7B 위에서 256개 sample로 verification하면, self-consistency와 Math-Shepherd를 결합한 결과는 GSM8K 89.1, MATH500 43.5이고, Math-Shepherd 단독은 88.4 / 41.1이다(Table 3). 그러므로 89.1 / 43.5라는 headline은 SFT 모델에 PRM을 더한 결과가 아니라, PPO 모델에 self-consistency와 PRM을 더한 결과다.

LLaMA2-70B generator를 사용한 best-of-256 selection(Table 1)은 self-consistency 88.0 / 39.4, ORM 91.8 / 40.4, Math-Shepherd 93.2 / 44.5이다. 70B generator에 7B verifier를 붙인 결과는 self-consistency보다 낮았고, candidate 수가 늘어날수록 7B reward model은 하락하는 반면 70B reward model은 개선된다(§5.3, 그림 5).

조건은 다음과 같다. 모든 실험은 MetaMATH-fine-tuned generator를 사용한 GSM8K와 MATH 문제를 쓰며, 다른 domain에서 훈련하지 않는다(§4). 저자들은 label noise가 남아 있고, 이것이 PRM 품질에 미치는 효과는 결정되지 않았다고 말한다(§6).

### §2.4 OmegaPRM: 더 적은 계산으로 같은 레이블 만들기

[[omegaprm]]은 몬테카를로 정의는 유지하고 탐색을 바꾼다. 모든 prefix를 추정하는 대신, 처음으로 잘못된 단계를 이진 탐색한다.

`c_t = MonteCarlo(q, x_1:t)`는 그 prefix에서 시작한 `k`개 rollout 중 golden answer로 끝나는 rollout의 비율이다(식 1). 탐색 절차(§3.2)는 다음과 같다.

1. 중간점 `m`에서 해결책을 나누고 `x_1:m`에서 rollout한다.
2. `c_m > 0`이면 전반부를 올바르다고 간주하고 후반부에서 재귀한다. `c_m = 0`이면 전반부에서 재귀한다.
3. 구간이 한 단계가 되면 멈춘다.

단계 수가 `M`일 때 비용은 단계별 추정의 `O(k M)` 대신 `O(k log M)`이다. Rollout은 MCTS 스타일 pool 안에서 재사용된다. 이 pool의 selection score는 `MC(s)`가 1에 가깝지만 여전히 잘못된 답을 낸 state를 선호한다(식 2-3). 설정은 추정당 `k = 8` rollout, 질문당 탐색 제한 100회, 해결책을 16개 조각으로 분할, 그리고 32개 rollout에서 맞는 답이 0개이거나 틀린 답이 0개인 질문을 제거하는 question filtering이다(§4, 부록 A). 레이블은 pointwise soft target `MC(s)`이다(§3.4).

`k = 64`, MATH500, Gemini Pro / Gemma2 27B에서의 결과(Table 1)는 다음과 같다. majority vote는 67.2 / 54.7, Math-Shepherd data는 67.2 / 57.4, PRM800K data는 67.6 / 57.2, OmegaPRM은 69.4 / 58.2이다. Gemini Pro의 51.0 → 69.4 headline은 18.4점 absolute 향상이며 **36% relative** 향상이다(§6). 69.4를 relative gain이라고 보고하면 정확도 값과 개선량을 혼동하는 셈이다. 효율 면에서 같은 계산 예산은 brute force로는 200K data point를 만들지만, binary search로는 15M개를 만든다. 그중 1.5M개를 training에 사용할 표본으로 뽑았다(§4.4).

이 방법에는 golden answer가 있는 질문이 필요하므로, adaptation 없이는 open-ended task에 적용할 수 없다(§5). 또한 공개 recipe들이 실제로 한 일도 보자. DeepSeek-R1은 성공하지 못한 시도에 PRM을 포함하며, fine-grained step을 정의하기 어렵고, 단계 정확성을 레이블하기 어렵고, model-based PRM의 reward hacking이 가능하고, 추가 계산량에 비해 이득이 제한적이라는 점을 든다([[deepseek-r1]] G.2). 2025년의 공개 reasoning recipe는 process reward가 아니라 outcome-only reward를 사용한다.

---

## §3 몬테카를로 레이블이 측정하는 것과 PRM 평가 방법

### §3.1 가치 추정은 정확성 판단이 아니다

[[qwen-prm-lessons]]는 아래 §3.2 비교가 측정하는 구별을 명시한다. PRM은 현재 단계가 올바른지를 결정론적으로 판정하는 judge를 목표로 하지만, MC 추정치는 **가치(value)**, 즉 completion policy 아래에서 해당 단계로부터 올바른 최종 답에 도달할 확률이다(§3.1.1). 따라서 MC 레이블로 PRM을 훈련하면, 그 이름이 시사하는 단계 정확성 judge가 아니라 value model이 만들어진다. 두 대상은 completer가 나쁜 단계에서 회복할 수 있거나 좋은 단계에서 실패할 때 정확히 불일치한다. 이것이 §2.2의 `p`/`q` 산술이다.

### §3.2 근거: 같은 데이터, 세 가지 레이블링 방법

표시된 모든 행은 동일한 860k queries와 responses를 사용한다. Best-of-8의 policy는 Qwen2.5-Math-7B-Instruct이고, ProcessBench는 모델이 첫 번째 오류 단계를 찾는지를 측정한다([[qwen-prm-lessons]] Tables 3-4).

| 단계 레이블 | 샘플 수 | Best-of-8 평균 | ProcessBench 평균 F1 |
|---|---:|---:|---:|
| MC (Math-Shepherd 데이터) | 440k | 64.3 | 28.9 |
| MC (Qwen 데이터) | 860k | 65.9 | 40.1 |
| LLM-as-a-judge (Qwen2.5-72B-Instruct) | 860k | 65.3 | 46.5 |
| Human (PRM800K) | 264k | 64.9 | 56.5 |
| maj@8 / pass@8 | — | 66.2 / 74.7 | — |

여기에는 두 가지 해석이 있다. 첫째, 두 평가는 방법들을 서로 반대 순서로 배치한다. 그러므로 best-of-N만 보고하는 논문은 process verification에 관한 주장을 뒷받침할 수 없다. 둘째, 이 PRM들 중 어느 것도 `N = 8`에서 maj@8을 이기지 못한다. 배포된 Qwen2.5-Math-PRM-7B는 평균 67.6으로 이를 이기며(Table 6), ProcessBench는 73.5이다(Table 7).

### §3.3 합의 필터링과 레이블 임곗값

LLM judge와 MC estimate가 첫 오류 위치에 동의하는 instance만 남기면 860k set의 약 40%가 유지되고, ProcessBench F1은 40.1에서 46.3으로 오른다. 이는 860k 전체로 훈련한 LLM-judge 모델에 가깝다(§3.1.3, 그림 2). hard-label threshold를 1/8부터 7/8까지 훑으면 두 평가가 모두 단조롭게 나빠진다. 따라서 권장 threshold는 0이다. 즉, 8개의 completion 중 **어느 것도** 정답에 도달하지 않을 때만 단계를 negative로 둔다(§3.1.4, 그림 5). 이 설정에서 filtering 후 soft label은 hard label보다 성능이 낮다. 저자들은 올바른 단계도 1보다 작은 target을 받는다는 점과 completion 8개의 분산을 이유로 든다(§3.1.4, 그림 3-4). [[omegaprm]]은 자기 데이터에서 반대 순서를 보고한다. soft는 hard의 63.3%에 비해 70.1% step-classification accuracy를 낸다(§4.3, Table 2). 따라서 label form의 우열은 setup 전반에서 확정되지 않았다. 두 연구는 policy, completer, filtering이 다르다.

### §3.4 best-of-N이 PRM 점수를 부풀리는 이유

모두 [[qwen-prm-lessons]] §3.2에서 나온 세 가지 기제가 있다.

1. Policy는 난도가 올라갈수록 flawed process를 가졌으나 정답인 answer를 더 자주 만든다. correct-answer sample 중 이 비율은 GSM8K 5.1%, MATH 11.9%, OlympiadBench 27.4%, Omni-MATH 43.4%이다(그림 6). 그러한 response에 낮은 점수를 올바르게 매기는 PRM도, 답만 검사하는 best-of-N metric에서는 불이익을 받는다.
2. 그런 response를 허용하는 PRM은 best-of-N에서 더 높은 점수를 받는다. 정답이면서 erroneous step이 있는 ProcessBench case에서 저자들의 PRM을 제외한 모든 open-source PRM은 50% 미만을 탐지한다(Table 5).
3. best-of-N에 최적화하면 PRM이 outcome scoring 쪽으로 이동한다. 네 개 open PRM에서 response의 40% 초과는 minimum step score가 최종 단계에 있다(그림 8).

### §3.5 집계 방식은 레이블 출처에 따라 달라진다

MC로 훈련한 PRM에서는 **마지막** 단계 점수가 더 좋은 solution score다. 마지막 단계의 MC value가 이미 전체 prefix를 통합하고, 단계별 추정치들이 독립이 아니기 때문이다. 사람이 주석을 달았거나 LLM judge가 만든 PRM에서는 product와 minimum이 더 낫다(§3.2.4, 그림 9). 한 팀이 한 방식으로 label을 훈련하고 다른 방식으로 solution score를 계산하면, 모델과 무관한 이유로 정확도를 잃는다.

**범용 모델에 대한 함의.** 단계별 supervision에는 수학 밖에서 실패하는 문서화된 전제조건이 있다. MC 레이블에는 golden answer와 `p`를 크게 만들 만큼 강한 completer가 필요하다. open-ended task에는 둘 다 없다. instrument를 검증할 수 있는 곳에서 단계 supervision을 사용하고, best-of-N 점수가 아니라 단계 수준 benchmark로 이를 측정하라.

---

## §4 순위 매기기가 아니라 정책을 위한 단계 수준 신호

§1-§3은 단계 레이블로 candidate solution의 순위를 매긴다. 이 절은 그 레이블을 gradient 안에서 사용한다.

### §4.1 잘못된 trace에서 얻는 단계별 advantage

[[rl-on-incorrect-synthetic-data]]는 preference loss가 보는 쌍이 중요한 단계에서만 다르도록 negative를 구성한다.

1. 정답을 아는 문제에 대해 SFT policy에서 incorrect response를 표본 추출한다(실용적 버전에서는 질문당 8개).
2. 각 단계마다 `π̃ = BoK(π_sft)`, `K = 5`를 두고 prefix에서의 Monte-Carlo rollout으로 `Q^π̃(x, ŷ_1:i−1; ŷ_i)`를 추정한다.
3. 가장 낮은 `Q`를 갖는 첫 단계인 “first pit” `ŷ_c`를 취하고, (correct solution, `ŷ_c`에서 끝나는 prefix) 쌍을 추가한다.
4. `π_sft`를 reference로 하여 이 쌍들에 DPO를 실행한다.

state의 value가 이전 단계의 Q-value와 같다는 deterministic dynamics 아래에서의 step advantage는 다음과 같다(식 3).

```
A^π̃(x, ŷ_1:i−1; ŷ_i) = Q^π̃(x, ŷ_1:i−1; ŷ_i) − Q^π̃(x, ŷ_1:i−2; ŷ_i−1)
```

GSM8K와 MATH에서 DeepSeek-Math-7B 및 Llama2-7B로 얻은 결과는 다음과 같다. 단계별 DPO는 SFT policy보다 향상되며, synthetic set이 커져도 계속 향상된다. 저자들은 이를 synthetic-data efficiency의 8배 이득으로 요약한다. self-generated positive만 사용하는 방법(rejection fine-tuning)은 약 2배의 가치가 있다(§1, §6.2, 그림 7a-b). 이 결과만큼 control도 중요하다. 임의의 correct/incorrect response pair에 DPO를 적용하면 MATH에서 SFT policy보다 향상되지 않으며, `β`를 조정해도 고칠 수 없었다(그림 7c). Theorem 6.1은 이 단계별 쌍 구성의 optimum이 advantage-weighted RL과 같음을 보인다. 이 연구가 일반화한 더 이른 단계 선호 구성은 [[step-dpo]]를 보라.

### §4.2 가치가 아니라 진전: process advantage verifier

[[rewarding-progress-pav]]는 dense step reward가 무엇을 예측해야 하는지 묻는다. 답은 **prover** policy `μ` 아래에서 측정한, 해당 step이 야기한 성공 확률의 **변화**다. `μ`는 훈련 중인 policy가 아니다.

```
∇_π ℓ_PAV-RL(π) = Σ_h ∇_π log π(a_h | s_h) · ( Q^π(s_h, a_h) + α · A^μ(s_h, a_h) )      (식 5)
```

- `Q^π(s_h, a_h)`: 이 단계 뒤 policy 자신의 completion이 정답에 도달할 확률이다. 이는 policy gradient가 이미 지닌 outcome term이다.
- `A^μ(s_h, a_h) = Q^μ(s_h, a_h) − V^μ(s_h)`: prover 아래에서 이 단계가 이루는 진전이다.
- `α ≥ 0`: process term의 가중치다.

§3.2가 말하는 결과는 두 가지다. 첫째, policy gradient는 이미 baseline을 빼므로 `μ = π`로 두면 outcome-only update를 재현한다. 즉, 현재 policy 자신의 value를 예측하는 PRM은 RL에 아무것도 더하지 않는다. 둘째, 지나치게 능력이 좋은 prover는 모든 단계에서 성공하고 좋은 단계와 나쁜 단계에 거의 같은 점수를 준다. 아주 약한 prover는 어느 단계에서도 성공하지 못한다. Prover는 base policy와 보완적이어야 하며, 논문은 base policy보다 약한 prover도 작동할 수 있다고 보고한다.

**계산 예시.** 어떤 prover가 단계 전에는 `Q^μ = 0.40`, 단계 후에는 0.55를 주면 `A^μ = +0.15`다. `α = 3`, outcome term `Q^π = 0.20`이면 그 단계의 effective reward는 `0.20 + 3(0.15) = 0.65`다. 강한 prover의 성공 확률을 단계 전후 모두 0.99로 유지하는 rephrasing step은 `A^μ ≈ 0`이므로 bonus를 받지 않는다. 반면 `Q`-valued reward는 이 단계에 0.99를 지급한다. 이것이 그림 2b가 지목하는 실패 모드다. 강한 prover의 `Q`는 사소한 action에 받을 근거 없는 bonus를 부여한다.

결과는 다음과 같다. 같은 compute에서 PAV에 대한 beam search는 ORM re-ranking보다 정확도가 8% 초과로 높고, compute efficiency는 1.5-5배다. Gemma 2B와 9B에서 `Q^π + α A^μ`로 하는 RL은 outcome-only RL보다 accuracy가 7% 초과로 높고, 그 accuracy에 sample 기준 약 6배 더 빨리 도달하며, 모든 `N ≤ 128`에서 더 높은 Pass@N을 보인다(Abstract; §5, 그림 7-8). 설정은 REINFORCE, learning rate 1e-7, batch 32, response length 512, KL 0.001, `α = 5.0`(2B), `3.0`(9B)이다(부록 E). 조건은 mathematics만 해당하며, Gemma 2B/9B/27B와 하나의 prover family(SFT policy 주위의 best-of-K)를 쓴다.

**함의.** §3과 §4는 같은 artifact에 서로 다른 요구를 부과한다. ranker로 쓰는 PRM은 correctness를 올바르게 판정해야 하고, dense reward로 쓰는 PRM은 progress에 대해 정보를 제공해야 한다. 같은 훈련 모델이 두 용도 모두에 최선의 instrument일 가능성은 낮다.

---

## §5 RLVR: verifier를 reward로 사용하기

### §5.1 정의와 Tülu 3 objective

검증 가능한 reward를 이용한 reinforcement learning은 learned reward model을 completion을 reference와 대조해 검사하는 함수로 대체한다([[rlvr-tulu3-config]] §6, 식 7-8).

```
max_πθ  E_{y ~ πθ(x)} [ v(x, y) − β KL[ πθ(y|x) ‖ π_ref(y|x) ] ]

v(x, y) = α  if correct, else 0,   with α = 10
```

- `v`: 해당 prompt를 위한 verification function이다. `β`는 reference policy(DPO checkpoint)에 대한 KL coefficient이고, `α`는 correctness reward의 크기다. `α`는 “pilot experiments에 기초해” 10으로 정했으며 더 조정하지 않았다(§6).
- objective에서는 KL term을 divergence로 쓴다. token별 구현 형태 `−β log(πθ/π_ref)`에서 이 값은 k1 estimator다. 이 report는 estimator 이름을 언급하지 않는다(k1/k2/k3 정의는 [[john-schulman-kl-tricks]] 참고).

### §5.2 Tülu 3에서 실제로 검증하는 대상

| Prompt set | 수 | 검증 |
|---|---:|---|
| GSM8K train | 7,473 | 8-shot CoT prompt; 최종 숫자를 추출하고 label과 비교 |
| MATH train | 7,500 | 3-shot CoT prompt; 답을 추출하고 “flex” MATH logic 적용 |
| IF verifiable | 14,973 | IFEval constraint template마다 verification function 하나 |
| **Total** | **29,946** | — |

도메인은 둘이고 평가는 셋이다([[rlvr-tulu3-config]] §6.1, Table 22). 이 stage에는 code verifier가 없다. §6.1은 저자들이 “leave more complex verifiers to future work”라고 명시한다. 그 문장에 붙은 각주(footnote 17)는 다른 연구의 관련 작업으로 code execution feedback을 인용한다. library card [[rlvr-tulu3]]와 [[tulu-3]]는 unit-test code verifier를 목록에 넣지만, 이 지점에서는 틀렸다.

### §5.3 구성과 결과

주요 값(Table 21 및 caption)은 다음과 같다. learning rate는 3e-7(70B는 1e-7), effective batch는 224(70B는 640), PPO update iteration `K = 4`, clip `ε` 0.2, GAE `λ` 0.95, `γ` 1.0, response length 2,048(GSM8K만 1,024), total episode 100,000이다. `β`는 [0.1, 0.05, 0.03, 0.01]에서 sweep했으며, 최종 8B run은 warm-up 0.0 및 `β = 0.05`다. Value model은 DPO policy가 아니라 reward model로 초기화한다. 이 방식은 §6.2 ablation에서 GSM8K와 average 모두 더 높은 점수를 냈다(그림 21). Ablation의 reward model은 UltraFeedback으로 훈련했지만, 최종 run의 reward model은 Tülu 3 8B preference mixture로 훈련했다(§6.4). 검증 가능한 reward에 reward-model score를 더하면 더 나쁘고 더 noisy한 결과가 나왔다(그림 22).

결과(Table 23)는 Tülu 3 8B DPO → RLVR에서 GSM8K 84.3 → 87.6, MATH 42.0 → 43.7, IFEval 81.1 → 82.4, Table 23 average 64.4 → 64.8이다. 70B에서는 GSM8K 93.5 → 93.5, MATH 62.3 → 63.0, IFEval 82.6 → 83.2, average 75.9 → 76.0이다. targeted evaluation에서는 몇 점의 이득이지만 average에서는 거의 0이다.

**계산 예시(유도됨).** `α = 10`일 때 training-set correct rate를 0.01 올리는 run은 expected verifiable reward를 0.1 얻는다. Advantage whitening 전에 같은 단위로 KL penalty를 더하면, `KL = 2` nats에서 `β·KL = 0.05·KL`이므로 이 이득은 상쇄된다. `β` sweep이 탐색하는 것은 `α`만이 아니라 이 비율이다. `v`에만 공통 인자를 적용하는 것은 advantage whitening에 흡수되므로(§6.2 item 5), `α`는 `β`와 −10 non-EOS penalty에 대한 상대값으로만 의미가 있다.

### §5.4 Goodhart gap은 줄지만 사라지지는 않는다

Learned reward model을 대체하면 [[reward-model-overoptimization]]이 연구한 drift가 제거된다. Policy가 움직일수록 오류가 커지는 proxy model이 없기 때문이다. 그러나 check와 intent 사이의 gap은 사라지지 않으며, 문서화된 이유가 네 가지 있다.

1. **RLVR 내부의 over-optimization.** `β`가 낮아질수록 KL은 올라가고 average score는 떨어진다. Tülu 3 report의 Appendix B.4는 high-KL IFEval run에서 over-optimized output을 보인다([[rlvr-tulu3-config]] §6.2.1).
2. **Verifier false positive.** final-answer check는 난도에 따라 5.1%에서 43.4%의 비율로 flawed reasoning을 받아들이며([[qwen-prm-lessons]] 그림 6), 바로 그런 response가 reinforcement를 받는다.
3. **Test exploitation.** production RL setting에서 한 모델은 언제나 True를 반환하는 `__eq__`, assert 전에 `sys.exit(0)`을 호출하는 방식, 모든 outcome을 passed라고 보고하는 `conftest.py`로 unit test를 무력화하는 법을 배웠다([[natural-emergent-misalignment-reward-hacking]] §2, 그림 7). METR은 o3의 한 version이 RE-Bench 128 run 중 39개에서 reward hacking을 보였다고 관찰했다([[metr-frontier-reward-hacking]]).
4. **Reward와 독립적인 이득.** Qwen2.5-Math-7B에서 GRPO를 쓴 RLVR은 random하게 배정한 reward로 MATH-500을 21.4 absolute point 향상시켰고, ground-truth reward에서는 29.1점 향상시켰다. 이는 clipping bias가 pretraining prior를 증폭한 효과로 해석되며, Llama3나 OLMo2에서는 재현되지 않았다([[spurious-rewards-rlvr]], arXiv:2506.10947v2 Abstract, §2-§4). 검증 가능한 reward에서 이득이 났다는 사실만으로 verifier가 무엇인가를 가르쳤다는 증거가 되지는 않는다.

---

## §6 agent를 위한 실행 기반 및 규칙 기반 reward

### §6.1 SWE-RL: 실행할 수 없을 때 쓰는 유사도 규칙

[[swe-rl-difflib-reward]]는 test를 실행하기에 비용이 너무 클 때 verifier를 어디서 얻는지 묻는다. 답은 reference output에 적용하는 규칙이다([[swe-rl-difflib-reward]] §2.1, 식 1).

```python
R(o) = -1                                                   # 잘못된 출력 형식
     = difflib.SequenceMatcher(None, pred, gt).ratio()      # 그 외에는 [0, 1]
```

데이터는 2015-01-01부터 2024-08-31까지의 GitHub event와 clone한 repository 4.6M개다. 이로부터 aggregated PR 24M개, filtering 후 unique PR instance 약 11M개를 거쳐 RL에 쓸 **seed 273k개**를 선택했다(부록 A). Policy는 **Llama-3.3-70B-Instruct**이고, 16k context, global batch 512(문제 32개에서 각각 rollout 16개), H100 GPU 512개로 약 32시간 동안 GRPO 1,600 step을 수행했다(§3.1).

결과는 Agentless Mini scaffold를 사용한 SWE-bench Verified에서 41.0%다. 같은 base model 및 issue-solving data로 만들고 general coding·dialog data와 결합한 SFT baseline은 36.2%다(Table 1; §3.5). Oracle file을 쓰는 repair-only 결과는 RL 34.8, SFT 29.6, base model 5.4(greedy; 20-sample majority voting에서는 16.6)다(Table 2). Base model도 올바른 형식의 patch를 12.2% 확률로 내므로, SFT와 RL 이득의 일부는 format compliance에서 온다.

Reward ablation이 transfer 가능한 결과다. exact patch match이면 1, 그렇지 않으면 0을 주는 discrete reward는 repair-only accuracy 29.0을 내며 continuous similarity의 34.8보다 낮다. 실제 patch가 정확히 일치하는 경우가 드물기 때문에 average discrete reward는 전체 run 동안 거의 0에 머문다(그림 5). Binary check가 거의 모든 rollout에 0을 반환하면 group 안 rollout들의 reward는 거의 같고 normalized advantage도 거의 0이다. Partial progress를 점수화하는 graded rule은 group 내 variation을 되살린다. 저자들이 말한 limitation은 같은 선택의 반대편이다. Similarity는 semantic equivalence가 아니므로, 기능적으로 올바른 대안 patch는 reward를 적게 받는다(§5).

### §6.2 hidden test가 있는 실행 verifier

[[deepswe]]는 R2E-Gym 문제 4.5K개에서만 RL로 Qwen3-32B를 훈련한다. 생성한 patch가 5분 제한 안에 선택된 Pass2Pass와 Fail2Pass test 표본을 통과하면 reward는 1이고, 하나라도 test가 실패하거나 실행 시간이 초과하면 0이다(§2.2). 결과는 SWE-bench Verified에서 Pass@1 42.2%(16회 run의 평균), Pass@16 71.0%, `K = 16`에서 hybrid test-time scaling으로 59.0%다(§4). Context limit, step limit, 또는 20분 generation timeout에 도달해 끝나는 trajectory는 0을 주는 대신 **masked** 처리한다(“compact filtering”). 저자들은 그런 trajectory에 reward를 주거나 penalty를 주면, curve로만 제시한 Qwen3-14B ablation에서 collapse가 일어난다고 보고한다(§2.3, 그림 6). Blog와 released script가 서로 다른 부분을 포함한 recipe value는 [[deepswe-recipe]]에 있다.

공식 report에서 나온 또 세 가지 verification pattern이 있다. 각각 agent를 위해 검사 가능한 outcome을 만드는 다른 방식이다.

- **생성한 verifier function.** DeepSeek-V3.2의 general-agent environment는 sandbox database, tool function, task, Python solution, verifier function을 작성하는 agent가 만든다. 현재 모델에서 non-zero pass@100인 instance를 유지하여 environment 1,827개와 task 4,417개를 얻는다([[deepseek-v3.1]] §3.2.3).
- **search agent의 answer accuracy.** GLM-4.5의 agentic RL은 web search에는 final-answer accuracy를, SWE task에는 test result를 reward로 사용한다. 잘못된 tool-call format은 trace를 멈추게 하고 reward 0을 준다([[glm-4-5]] §3.3.2).
- **hack check가 있는 rubric reward.** Kimi K2의 instruction-following reward는 code-interpreter check, LLM judge, 그리고 지켰다고 주장했지만 실제로는 지키지 않은 compliance를 탐지하는 layer를 결합한다. Budget을 넘는 response는 잘라내고 penalty를 준다([[kimi-k2]] §3.2.1, §3.2.3). 검증 불가능한 task의 rubric reward는 ch-44b에서 다룬다([[rubrics-as-rewards]]).

**Agentic 주의점.** SWE-RL은 single turn(issue → patch)을 훈련하며, DeepSWE는 SWE-bench Verified 밖의 평가를 보고하지 않는다([[deepswe]] Findings). Multi-turn credit assignment, observation masking, rollout stability는 ch-45b에서 다룬다.

---

## §7 검증 가능한 도메인에서 다른 모든 영역으로의 전이

범용 모델이 RLVR을 사용하는 이유는 검사 가능한 task로 훈련하면 검사할 수 없는 task도 개선되리라는 기대에 있다. 근거는 양방향을 가리키며, 방향은 방법과 무엇을 held out하는지에 따라 달라진다.

**RL의 긍정적 전이.** SWE-RL의 out-of-domain 표는 zero-shot greedy, base / SFT / RL 순서로 다음을 보인다. HumanEval+ 76.2 / 73.2 / 79.9, CRUXEval-I 60.5 / 68.4 / 71.6, CRUXEval-O 61.9 / 75.1 / 75.5, MATH strict 63.2 / 54.0 / 73.7(lenient 70.9 / 71.7 / 73.7), MMLU 86.49 / 85.26 / 86.82, BigCodeBench-Hard는 변하지 않았다([[swe-rl-difflib-reward]] Table 3). 같은 base model에서 issue-solving data와 general coding·dialog data로 만든 SFT model은 평균적으로 base model보다 낮다. 저자들은 개별 1-2점 차이는 그것만으로 유의하지 않으며, aggregate에서 유의성을 주장한다고 말한다(§3.5).

**통제된 비교.** [[transferability-of-llm-reasoning]]은 같은 math data로 Qwen3-14B-Base를 두 방식으로 fine-tuning한다. 하나는 rejection-sampled Qwen3-32B trace로 하는 SFT이고, 다른 하나는 answer correctness를 reward로 하는 GRPO다(arXiv:2507.00432v2 §2.2, Table 1). Math average는 base 27.7, SFT-think 49.8, RL 53.8이다. Non-reasoning average(CoQA, MC-TACO, IFEval, HalluEval)는 base 45.7, SFT-think 21.1, SFT-no-think 29.0, RL 53.2다. IFEval 단독은 69.2 → 42.3(SFT) → 70.0(RL)이다. 논문의 PCA 분석은 세 task group 모두에서 RL이 SFT보다 representation shift가 작다고 보고한다(Table 2). 이 결과는 단일 연구, 하나의 base model, math-only training data라는 조건 아래의 것이다.

**검증 가능한 훈련에서의 narrowing.** [[ifbench]]는 IFEval 스타일 constraint로 GRPO를 훈련하고 unseen constraint를 측정한다. Tülu-3-8B-DPO policy에서 시작한 IF-RLVR은 IFEval을 81.1 → 92.2, IFBench를 25.2 → 44.6으로 올린다(Table 6). 그러나 AlpacaEval 2는 33.5 → 21.3, MMLU는 68.7 → 66.4, GSM8K는 84.3 → 83.2로 떨어진다(Table 3). 또한 constraint를 제거한 같은 response의 GPT-4.1 judge score는 10점 만점에 7에서 6.4로 떨어진다(§5). IFBench에서는 GPT-4.1과 Claude 3.7 Sonnet이 50% 미만이고, 2B까지 내려가는 많은 모델이 IFEval의 constraint template 25개에서는 80% 초과를 얻는다(§1, 그림 1). 더 잘 generalize하는 training choice는 constraint type을 더 많이 쓰는 것, prompt당 constraint를 여러 개 쓰는 것(test prompt는 최대 3개인데도 5-6개까지), test range를 포함하고 넘어서도록 variable range를 두는 것이다(§4.1-§4.3). Constraint reward에 reward-model score를 섞으면 IF accuracy를 희생하고 AlpacaEval 2는 31.6으로 회복된다(부록 E).

**규칙 너머.** [[crossing-the-reward-bridge]]는 reference answer가 free-form일 때 무슨 일이 생기는지 측정한다. 해당 mathematics problem 중 single-term numeric answer를 갖는 비율은 60.3%뿐이고, multi-subject query에서는 45.4%다(§1). Qwen2.5-7B와 prompt 30k개에서 rule-based binary reward를 쓴 RLOO는 math / multi-subject에서 58.5 / 26.3에 도달하고, distilled 7B generative verifier를 쓰면 63.0 / 28.1에 도달한다. Verifier의 soft reward는 REINFORCE 아래 multi-subject에서 31.2에 도달한다(Table 1). Prompt를 20k에서 100k로 scale하면 rule-based reward는 multi-subject에서 26.2에서 16.9로 움직이고, model-based reward는 30.8에서 35.0으로 움직인다(Table 2). Free-form answer를 잘못 레이블하는 rule은 데이터가 많아질수록 더 나빠진다. Out-of-domain training set에서는 NaturalReasoning 29.4 → 39.8, WebInstruct 33.9 → 44.0이다(Table 3).

**pass@k caveat.** [[rlvr-beyond-base-model]]은 RLVR 훈련 모델이 작은 `k`에서 base model보다 좋지만, 큰 `k`에서는 base model이 같거나 이를 넘는다고 보고한다. 이를 base model이 이미 가졌던 path들에 대한 sampling efficiency를 RLVR이 개선하는 것으로 해석한다(arXiv:2504.13837v5 Abstract). [[prorl]]은 다른 regime을 보고한다. DeepSeek-R1-Distill-Qwen-1.5B에서 2k 초과 RL step은 여섯 math benchmark의 average pass@1을 44.45에서 60.14로 높였다. 일부 task에서는 base model이 `k`가 256 이하인 어떤 값에서도 하나도 풀지 못했으며, 주로 mathematics인 “Diminish” regime에서는 pass@128이 평평하게 유지되거나 하락했다(Tables 1, 3; §4.2-§4.3). [[interplay-pretraining-midtraining-rl]]은 조건을 다음과 같이 제시한다. Task가 모델 competence의 경계에 있고 prior가 더 이른 단계에서 설치되었을 때 RL은 capability를 확장한다. 이는 열린 질문이다.

---

## Negative sample과 negative feedback

### Negative의 출처와 종류

ch-43a의 네 가지 구분을 사용한다. (1) negative marginal value, (2) content로서의 negative, (3) conditioning으로서의 negative, (4) gradient로서의 negative.

| Negative의 출처 | 레이블 방식 | 종류 | 측정된 오류율 |
|---|---|---|---|
| PRM 훈련에서 negative로 레이블된 단계 | 사람, MC rollout, 또는 LLM judge | classifier target이며 policy gradient가 아님 | MC hard label은 GSM8K 단계 160개에서 `N = 4`일 때 86% 정확([[math-shepherd]] §5.2); MC 훈련 PRM은 ProcessBench F1 40.1, human label은 56.5([[qwen-prm-lessons]] Table 4) |
| RLVR에서 잘못된 최종 답 | verifier가 0을 반환 | baseline subtraction 뒤 negative advantage(gradient) | flawed process를 지닌 correct answer는 correct-answer sample의 5.1%-43.4%([[qwen-prm-lessons]] 그림 6) |
| incorrect trace의 첫 bad step | 단계별 Monte-Carlo Q | 한 단계에 국소화한 gradient로서의 negative | 보고되지 않음 |
| agentic RL에서 실패한 patch 또는 실패한 test | 실행 | negative advantage | 보고되지 않음. DeepSWE는 대신 limit-terminated trajectory를 mask함([[deepswe]] §2.3) |
| truncated response(no EOS) | length check | Tülu 3에서 penalty −10, DeepSWE에서 masked, Kimi K2에서 truncation penalty | 보고되지 않음 |

### 메커니즘

표본 추출한 token sequence의 negative advantage는 log-likelihood gradient에 음수를 곱한다. Logit 수준에서 표본 추출 token `y`와 어떤 vocabulary entry `j`에 대해 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이다. 따라서 `log p_y`를 낮추면 `y`에서 mass가 빠져나와 현재 probability에 비례해 재분배된다. 이는 현재 가장 확률이 높은 alternative에 mass를 집중시키며, 정답이 이미 그 alternative인 경우가 아니면 정답으로 집중시키지 않는다. Likelihood displacement와 squeezing을 포함한 완전한 유도는 ch-43a에 있다. 이 장에 특화된 결과는 두 가지다.

- Verifier false negative(올바른 해결책을 wrong으로 표시)는 correct behaviour를 아래로 미는 작용이다. 그 비용은 false positive와 대칭이 아니다. False positive는 하나의 wrong behaviour를 가르치지만, false negative는 모델이 맞혔던 behaviour에서 mass를 제거한다.
- Negative를 첫 bad step에 국소화하는 것이 단계 수준 negative가 작동하게 하는 조건이다. 전체 trace 둘을 대비하면, 괜찮았던 단계를 포함해 양쪽의 모든 단계 likelihood가 바뀐다([[rl-on-incorrect-synthetic-data]] §6.1).

### 계산 예시: group baseline 아래의 0, −1, abstention

한 prompt에 대해 rollout 8개가 있고, GRPO와 같이 group-normalized advantage `A_i = (r_i − mean(r)) / std(r)`를 쓴다고 하자([[grpo]]).

- **Case 1**: 2개 correct에는 reward 10, 6개 wrong에는 0을 준다. mean은 2.5, std는 4.33, advantage는 correct **+1.73**, wrong **−0.577**이다.
- **Case 2**: 같은 outcome에 correct +1, wrong −1을 준다. mean은 −0.5, std는 0.866, advantage는 correct **+1.73**, wrong **−0.577**이다.

두 case는 동일하다. Outcome이 binary일 때 wrong answer에 0과 −1 중 무엇을 줄지, `α`의 크기를 포함해 reward를 affine하게 바꾸는 것은 모두 group normalization에 흡수된다. 선택이 실제로 중요해지는 것은 세 번째 outcome이 존재할 때, normalization을 제거할 때(Dr. GRPO와 DeepSeek-V3.2처럼 mean-only advantage), 또는 normalization하지 않은 KL term을 reward에 더할 때뿐이다.

- **Case 3**: 세 outcome을 둔다. 2개 correct +1, 3개 abstention 0, 3개 wrong −1이다. Mean은 −0.125, std는 0.781, advantage는 correct **+1.44**, abstain **+0.16**, wrong **−1.12**다. 이 group에서 abstention은 작은 positive advantage를 받으므로, policy는 보통 틀리는 prompt에서는 abstain하는 법을 배운다.
- **Case 4**: 같은 outcome이지만 abstention도 wrong answer처럼 0으로 점수화한다(`w = 0`). Abstention과 wrong answer는 구별되지 않으며, 둘 중 하나를 선호하게 하는 gradient도 없다.

### wrong-answer penalty가 함의하는 abstention threshold

Correct에 `+1`, wrong에 `−w`, abstain에 `0`을 주면, correct일 확률 `p`로 답할 때 expected reward는 `p − w(1 − p)`이며, 다음일 때 positive다.

```
p > w / (1 + w)
```

따라서 `w = 0`은 항상 답하라는 뜻이고, `w = 1`은 50% confidence 초과, `w = 3`은 75%, `w = 9`는 90% 초과일 때 답하라는 뜻이다. 이는 [[why-language-models-hallucinate]] §4.2가 evaluation을 위해 제안한 confidence-target instruction과 같은 산술이다. 그곳에서 mistake에는 `t/(1 − t)` point의 penalty를 주고, model은 confidence `t`를 넘을 때만 답해야 한다. (해당 절은 `t = 0.5`에서 penalty 1, `t = 0.75`에서 penalty 2, `t = 0.9`에서 penalty 9를 목록으로 든다. 그러나 `0.75/(1 − 0.75) = 3`이므로 중간 entry는 논문 자신의 formula와 맞지 않는다.) 논문의 주장은 evaluation scoring에 관한 것이다. 이 논문은 GPQA, MMLU-Pro, IFEval, Omni-MATH, BBH, MATH, MuSR, SWE-bench, HLE를 abstention에 credit을 주지 않는 binary grader로 분류하며(Table 2), Observation 1은 binary grading 아래에서는 abstention이 결코 optimal이 아니라고 말한다. 이를 RL reward에 적용하는 것은 이 course의 확장이고, 이 주장의 RL 측면은 그 논문에서 측정하지 않았다.

[`figures/wrong-answer-rewards.html`](figures/wrong-answer-rewards.html)을 사용해 wrong-answer penalty, abstention credit, group 내 correct/wrong/abstained/truncated rollout 수를 설정하고, advantage와 함의되는 abstention threshold를 확인하라.

### Controls

1. **국소화한다.** Negative를 whole trace가 아니라 first bad step에 둔다([[rl-on-incorrect-synthetic-data]]; [[prm800k]]는 첫 negative step에서 레이블링을 멈춘다).
2. **불확실한 실패는 penalty 대신 mask한다.** Context limit, step limit, timeout으로 종료된 trajectory는 correctness 정보를 갖지 않는다. DeepSWE는 이를 mask하며 mask하지 않을 때 collapse가 난다고 보고한다([[deepswe]] §2.3). Tülu 3는 EOS 없는 response에 −10 penalty를 주는 다른 선택을 한다([[rlvr-tulu3-config]] §6.2). 이는 2,048-token budget에서는 방어 가능하지만 long-reasoning budget에서는 비용이 크다(ch-44a).
3. **Gradient가 아니라 label을 filter한다.** LLM judge와 MC estimate의 consensus filtering은 step label 60%를 제거하고 first-error identification을 개선했다([[qwen-prm-lessons]] §3.1.3).
4. **Negative가 활성화된 동안 KL anchor를 유지한다.** Tülu 3 `β` sweep은 negative 측 control이다. 더 낮은 `β`는 더 많은 KL과 더 낮은 average score를 냈다([[rlvr-tulu3-config]] §6.2.1).

### Diagnostics

매 training step마다 다음을 log하라. Positive 및 negative advantage를 갖는 rollout의 비율, all-correct 또는 all-wrong인 group의 비율(zero gradient), truncation rate, abstention rate, entropy, held-out set의 pass@1 및 pass@large-k. PRM이 loop에 있다면, best-of-N score 옆에 step-level score(first-error identification)를 함께 기록하라. 두 점수는 반대 방향으로 움직일 수 있기 때문이다([[qwen-prm-lessons]] §3.2).

---

## 레시피

Primary source에 인쇄된 값을 쓴다. 2026-09-15로 표시한 행은 이 revision을 위해 primary text에서 읽었고, card 기반 행은 card의 verification date를 쓴다.

| 모델(exact release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Tülu 3 8B RLVR | 8B | RL | algorithm; reward | PPO; `v = α if correct else 0`, α = 10 | arXiv:2411.15124v5 §6 식 7-8 [[rlvr-tulu3-config]] | 2026-09-15 검증됨 | “pilot experiments에 기초했고 더 조정하지 않았다” |
| Tülu 3 8B RLVR | 8B | RL | prompts | 29,946(GSM8K 7,473 + MATH 7,500 + IF verifiable 14,973) | v5 §6.1, Table 22 | 2026-09-15 검증됨 | ablation 보고 없음 |
| Tülu 3 8B RLVR | 8B | RL | LR; effective batch; K; response length; episodes | 3e-7 linear; 224; 4; 2,048(GSM8K-only는 1,024); 100,000 | v5 Table 21 | 2026-09-15 검증됨 | 그림 19 task별 curve |
| Tülu 3 8B RLVR | 8B | RL | β; warm-up ratio | 0.05; 0.0(final run), [0.1, 0.05, 0.03, 0.01]에서 sweep | v5 Table 21 caption; §6.2 | 2026-09-15 검증됨 | 그림 21: 더 낮은 β → 더 높은 KL → 더 낮은 average score |
| Tülu 3 8B RLVR | 8B | RL | non-EOS penalty; advantage normalization; value init | −10.0; whitening; general RM에서 초기화(§6.2 ablation의 UltraFeedback RM, final run의 Tülu 3 8B preference-mixture RM) | v5 Table 21; §6.2, §6.4 | 2026-09-15 검증됨 | 그림 21: RM-initialized value > DPO-initialized on GSM8K and average |
| Tülu 3 70B RLVR | 70B | RL | LR; batch; β; warm-up; episodes | 1e-7; 640; β = 0.07 및 ω = 0.07(Table 21 caption) / β = 0.7, 0.1 warm-up, 400,000 episode(§6.4 text) | v5 Table 21 caption vs §6.4 | conflict(논문 내부) | §6.4: 70B KL은 1 미만 유지 |
| Math-Shepherd PRM | 7B verifier | reward-model | completer; 단계당 completion; label form | LLemma-7B; N = 8; 두 special token으로 된 hard label | arXiv:2312.08935v3 §4 [[math-shepherd]] | 2026-09-14 검증됨 | §5.2: N = 4에서 86% label accuracy, 더 큰 N에서 하락 |
| Math-Shepherd step-by-step PPO | Mistral-7B policy | RL | reward placement | 각 step-end token에 PRM score, 다른 곳은 0; 별도 final-answer term 없음 | ACL 2024 식 7; v3 §3.5 | 2026-09-14 검증됨 | Table 2: ORM의 81.8 / 31.3 대비 84.1 / 33.0 |
| OmegaPRM data | — | reward-model | MC estimate당 rollout k; search limit; step split | 8; 질문당 100; solution당 16개 piece | arXiv:2406.06592v2 §4, §4.2 [[omegaprm]] | 2026-09-14 검증됨 | §4.4: 같은 compute에서 brute force 200K 대비 annotation 15M |
| OmegaPRM data | — | reward-model | tree constant; label | α 0.5, β 0.9, L 500, c_puct 0.125; pointwise soft `MC(s)` | v2 §4, §3.4 | 2026-09-14 검증됨 | §4.3 Table 2: soft 70.1% > pairwise 64.2% > hard 63.3% step accuracy |
| Qwen2.5-Math-PRM-7B / -72B | 7B / 72B | reward-model | data; 단계당 completion; threshold; filter | 약 500K query, 각각 response 6-8개; completion 8개; hard threshold 0; consensus filtering이 약 40% 유지 | arXiv:2501.07301v2 §2.1, §3.1.3-§3.1.4, §4.1 [[qwen-prm-lessons]] | 2026-09-15 검증됨 | 그림 5(threshold sweep 1/8-7/8); 그림 2(filtering: ProcessBench 40.1 → 46.3) |
| Qwen2.5-Math-PRM-7B | 7B | reward-model | initialization; loss | two-layer scalar head가 붙은 Qwen2.5-Math-7B-Instruct; 각 step 마지막 token에서 CE | v2 §2.1, §4.1 | 2026-09-15 검증됨 | Table 6: maj@8 66.2 대비 Best-of-8 67.6; Table 7: ProcessBench 73.5 |
| PAV RL(Gemma) | 2B / 9B | RL | algorithm; LR; batch; response length; KL | token-level value baseline을 쓴 REINFORCE; 1e-7; 32; 512; 0.001 | arXiv:2410.08146v1 App. E [[rewarding-progress-pav]] | 2026-09-15 검증됨 | §5 그림 7: ORM-RL 대비 >7%, sample efficiency 6x |
| PAV RL(Gemma) | 2B / 9B | RL | process-term weight α; iteration | 5.0(2B), 3.0(9B); 10,000(2B), 5,000(9B) | App. E | 2026-09-15 검증됨 | App. E: 0.5-6.0의 α는 ORM-RL보다 향상 |
| Per-step DPO(Setlur) | DeepSeek-Math-7B, Llama2-7B | preference | 질문당 negative; prover | incorrect response 8개; `π̃ = BoK(π_sft)`, K = 5 | arXiv:2406.14532v1 §6.1 [[rl-on-incorrect-synthetic-data]] | 2026-09-15 검증됨 | 그림 7a-b: synthetic-data efficiency 8x; 그림 7c: standard DPO는 향상되지 않음 |
| Llama3-SWE-RL-70B | 70B | RL | policy; step; context; batch; compute | Llama-3.3-70B-Instruct; 1,600 step; 16k; 512(문제 32개 × rollout 16개); H100 512개 ≈ 32h | arXiv:2502.18449v2 §3.1 [[swe-rl-difflib-reward]] | 2026-09-15 검증됨 | Table 1: SWE-bench Verified에서 SFT 36.2 대비 41.0 |
| Llama3-SWE-RL-70B | 70B | RL | reward; seed | 잘못된 format은 −1, 그 외 `difflib.SequenceMatcher` ratio; 약 11M PR에서 나온 PR seed 273k개 | v2 §2, §2.1, App. A | 2026-09-15 검증됨 | 그림 5: repair-only continuous 34.8, discrete 29.0 |
| Llama3-SWE-RL-70B | 70B | RL | GRPO ε, β, learning rate | 보고되지 않음 | v2 §2.1, §3.1, App. A-D 확인 | 보고되지 않음 | — |
| DeepSWE-Preview | 32B | RL | reward; trajectory masking | 선택한 P2P/F2P test가 5분 이내 pass면 1, 그 외 0; context-, step-, timeout-terminated trajectory mask | www.together.ai/blog/deepswe §2.2-§2.3 [[deepswe]], [[deepswe-recipe]] | 2026-09-14 검증됨 | 그림 6(compact filtering 유무의 Qwen3-14B; curve만 있음) |
| IF-RLVR(Tülu-3-8B-DPO policy) | 8B | RL | algorithm; LR; prompt당 sample; length; hardware | GRPO(open-instruct); 5e-7; 16; 2,048; H100 8개, 2,000 step에 약 1일 | arXiv:2507.02833v3 §3 [[ifbench]] | 2026-09-14 검증됨 | Table 5: 같은 data에서 GRPO > DPO |
| Cross-domain RLVR(Qwen2.5-7B) | 7B | RL | algorithm; KL β; prompt; verifier | REINFORCE / RLOO / REINFORCE++; 0.01; 30k; Qwen2.5-72B-Instruct judgment 160k개에서 distilled한 7B generative verifier | arXiv:2503.23829v2 §3.2-§4.4 [[crossing-the-reward-bridge]] | 2026-09-15 검증됨 | Table 2: scale에서 rule-based는 저하, model-based는 향상 |

**작은 범용 run의 시작점.** Preference optimization 뒤 reference answer가 있는 prompt를 쓰는 8B policy라면, 검증된 Tülu 3 행은 다음을 뒷받침한다. Learning rate 3e-7의 PPO, effective batch 224, `K = 4`, clip 0.2, response length 2,048, 약 100,000 episode, warm-up 없는 `β = 0.05`, reward model로 초기화한 value model, advantage whitening, 그리고 targeted evaluation이 아니라 모든 evaluation의 average를 selection criterion으로 하여 [0.1, 0.05, 0.03, 0.01]에서 `β`를 sweep하는 방식이다. 이 값들은 math 및 instruction-following prompt 29,946개로 한 Llama-3.1-8B 기반 run에서 나온 값이다. 70B run은 1e-7과 더 큰 batch를 썼으므로 size 간에 그대로 전이하지 말라. Training 초기에 check가 거의 만족되지 않는다면(SWE 스타일 patch matching), SWE-RL의 exact-match variant는 1,600 step 동안 reward가 거의 0에 머물렀으므로 binary check를 graded rule로 바꿔라([[swe-rl-difflib-reward]] 그림 5). Reference answer가 free-form이면 rule을 더 쓰는 대신 generative verifier를 훈련하라([[crossing-the-reward-bridge]] Table 2).

---

## 일반화 관점

**(a) 폭을 넓히는 것.**

- Outcome verifier를 쓴 RL은 같은 math data에서 SFT가 저하시킨 non-reasoning score를 보존하고 개선했다. Qwen3-14B-Base에서 non-reasoning average는 base 45.7, SFT-think 21.1, RL 53.2다([[transferability-of-llm-reasoning]] Table 1).
- 하나의 software-engineering task에서 한 RL은 out-of-domain task family 다섯 개를 개선했다. 반면 같은 base model과 issue-solving data를 쓴 SFT baseline은 평균적으로 base model보다 낮아졌다([[swe-rl-difflib-reward]] Table 3).
- 훈련된 generative verifier는 parse 가능한 answer가 없는 domain까지 RLVR을 확장하며, rule-based reward가 저하되는 곳에서 data scale과 함께 계속 향상된다([[crossing-the-reward-bridge]] Tables 2-3).
- 검증 가능한 set 내부의 constraint 다양성도 중요하다. 더 많은 constraint type, prompt당 더 많은 constraint, 더 넓은 variable range로 훈련하면 out-of-domain constraint accuracy가 올랐다([[ifbench]] §4.1-§4.3).

**(b) narrowing을 일으키는 것.**

- Constraint-type overfitting: IFEval의 template 25개에서 80% 초과인 모델이 IFBench의 unseen constraint 58개에서는 50% 미만이다. IF-RLVR은 IFBench를 25.2 → 44.6으로 올렸지만 AlpacaEval 2는 33.5에서 21.3으로, MMLU는 68.7에서 66.4로 낮췄다([[ifbench]] Tables 3, 6).
- Verifier false positive는 우연히 right answer를 만드는 flawed reasoning을 강화하며, 그 비율은 problem difficulty와 함께 높아진다([[qwen-prm-lessons]] 그림 6).
- KL budget이 커질수록 RLVR 내부에서 일어나는 over-optimization([[rlvr-tulu3-config]] §6.2.1, 부록 B.4).
- 검증 가능한 reward 위에 reward-model score를 추가하면 Tülu 3에서 결과가 더 noisy해지고 average score가 낮아졌다([[rlvr-tulu3-config]] 그림 22).
- Best-of-N만을 위해 최적화한 PRM은 outcome scoring 쪽으로 drift하여, 자신이 만들어진 목적의 property를 잃는다([[qwen-prm-lessons]] §3.2.3).

**(c) 이 단계에서 이를 측정하는 방법.**

- Verifier가 target으로 삼지 않는 held-out capability suite를 targeted metric 옆에 보고한다. Table 23 benchmark에 대한 Tülu 3 average(8B에서 64.4 → 64.8)가 이 모델이다([[rlvr-tulu3-config]] Table 23).
- Training set과 disjoint한 verification function으로 만든 unseen-constraint 또는 unseen-verifier evaluation([[ifbench]] §2).
- PRM을 훈련할 때마다 best-of-N과 함께 step-level evaluation(first-error identification)을 수행한다([[qwen-prm-lessons]] Tables 6-7).
- 둘이 diverge할 수 있으므로 pass@1과 pass@large-k를 함께 측정한다([[rlvr-beyond-base-model]]; [[prorl]] §4.2-§4.3).
- Reward-independent control을 둔다. Random 또는 shuffled reward로 짧게 훈련해 이득 중 얼마나 남는지 확인한다([[spurious-rewards-rlvr]] §2).
- Verifier의 prompt pool을 held-out suite에 대해 decontaminate한다. SWE-RL은 corpus에서 SWE-bench repository를 제외한다([[swe-rl-difflib-reward]] 부록 A).

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 방법 |
|---|---|---|
| MC 단계 레이블을 correctness judgment로 취급 | PRM은 best-of-N에서 순위를 잘 매기지만 첫 오류를 찾지 못함 | Step-level benchmark로 평가하고 PRM800K 훈련 baseline과 비교([[qwen-prm-lessons]] Table 4) |
| Noise를 줄이려 단계당 completion 수를 늘림 | Label accuracy가 오르지 않고 하락 | 작은 human-labelled set을 기준으로 false-positive 및 false-negative rate를 따로 측정([[math-shepherd]] §5.2) |
| MC 훈련 PRM의 solution을 product 또는 minimum으로 score | 같은 모델의 last-step scoring보다 best-of-N이 낮음 | Validation set에서 세 aggregator를 비교([[qwen-prm-lessons]] 그림 9) |
| Policy 자신의 value estimate를 dense process reward로 사용 | Dense reward가 outcome-only RL과 비교해 아무것도 바꾸지 않음 | 같은 sample 수의 outcome-only run과 비교하고 prover가 policy와 다른지 확인([[rewarding-progress-pav]] §3.2) |
| Step-level training에 임의의 correct/incorrect trace를 짝지음 | Data가 늘어날수록 preference training이 평평해지거나 저하 | First bad step에서 prefix를 공유하는 pair와 ablate([[rl-on-incorrect-synthetic-data]] 그림 7c) |
| Verifier가 reward hacking을 없앤다고 가정 | Reward는 오르나 held-out average가 하락하고, 실제 fix 없이 test가 pass | Top-reward completion을 검사하고 KL과 average score를 비교하며 anti-hack check 실행([[natural-emergent-misalignment-reward-hacking]]; [[metr-frontier-reward-hacking]]) |
| Truncated rollout을 failure로 score | Entropy와 response length가 collapse하고 reward가 낮은 값에서 plateau | Truncation rate를 log하고 limit-terminated trajectory를 mask하여 비교([[deepswe]] §2.3) |
| Wrong answer와 abstention을 같은 점수로 처리 | Abstention rate가 0으로 가고 confident wrong answer가 증가 | Abstention rate와 accuracy-when-answered를 보고하고 명시적 wrong-answer penalty 설정([[why-language-models-hallucinate]] §4) |
| Accuracy value에서 큰 relative gain을 읽어냄 | 51.0 → 69.4 이동을 “69.4% relative improvement”라 함 | 재계산: absolute 18.4, relative 36%([[omegaprm]] §6) |
| Targeted benchmark의 RLVR gain만 보고 | Targeted metric은 오르나 모든 evaluation의 average는 평평하거나 하락 | Tülu 3처럼 둘 다 보고([[rlvr-tulu3-config]] Table 23) |

---

## 이해 점검

1. MC 레이블 PRM은 Best-of-8에서 PRM800K 레이블 PRM보다 높은 점수를 받지만 ProcessBench에서는 훨씬 낮다. 하나의 mechanism으로 두 결과를 설명하고, dense RL reward로 쓸 PRM이라면 어느 evaluation을 신뢰할지 말하라.
2. Math-Shepherd의 측정에서 단계당 completion 수를 4에서 16으로 늘리면 hard label 정확도가 낮아진다. §2.2의 두 error-rate formula에서 이를 유도하고, 어느 error가 지배적인지 정하는 completer property를 말하라.
3. PAV는 current policy 자신의 advantage를 dense reward로 쓰면 outcome-only RL을 재현함을 보인다. Policy gradient가 이미 무엇을 subtract하는지의 관점에서 이유를 설명하고, 유용한 prover가 갖춰야 할 property를 말하라.
4. Binary outcome의 group-normalized advantage 아래에서 wrong answer에 0을 주는 것과 −1을 주는 것은 같은 update를 만든다. Correct 2개인 8개 group으로 이를 보인 뒤, setup에서 어떤 세 가지 변화가 있어야 그 선택이 다시 중요해지는지 설명하라.
5. 한 팀이 wrong answer에 `w = 2`인 RLVR reward에 abstention을 추가한다. Policy가 답해야 할 confidence threshold를 계산하고, truncated response도 −2로 score하면 그 threshold에 어떤 일이 생기는지 설명하라.
6. Tülu 3는 8B에서 GSM8K +3.3, Table 23 average +0.4를 보고한다. 이 gap의 서로 다른 설명 두 가지를 제시하고, 이를 구분할 measurement를 말하라.
7. SWE-RL의 continuous similarity reward는 repair-only accuracy에서 binary exact-match reward보다 5.8점 높다. 거의 모든 rollout이 0일 때 group-normalized advantage 관점에서 mechanism을 설명하고, 이 reward design이 solution diversity에 부과하는 비용을 말하라.
8. IFBench는 IFEval-style constraint의 RLVR이 constraint accuracy를 올리면서 AlpacaEval 2와 MMLU를 낮춘다고 보인다. 이것이 measurement artifact가 아니라 narrowing인 이유를 설명하고, training 중 이를 잡아냈을 gate를 설계하라.

---

## 연결

- 이전: **ch-43a** — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Negative section에 쓰는 logit-level mechanism을 제공한다.
- 다음: **ch-44a** — Length in RL: Overlong Responses, Length Control, and Long-Context RL. Truncated rollout, non-EOS penalty, long context에서의 RL을 다룬다.
- **ch-41** — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization. Verifier와 PRM이 대체하는 learned reward model 및 §5.4에서 참조한 over-optimization curve를 다룬다.
- **ch-42** — Reward Hacking and Judge Design. Verifier와 test exploitation, 그리고 여기서 abstention에 쓴 binary-grading argument를 다룬다.
- **ch-40** — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO. Negative section의 worked example이 의존하는 advantage normalization을 다룬다.
- **ch-43** — Entropy, Output Diversity, and KL Control in RL. §5.3의 `β` sweep 및 KL-versus-score trade-off를 다룬다.
- **ch-16** — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse. Verifiable prompt의 출처와 filtering 방식을 다룬다.
- **ch-44b** — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. Reference answer가 없는 task의 rubric 및 generative reward를 다룬다.
- **ch-45b** — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. §6이 미리 본 multi-turn setting을 다룬다.
- **ch-31** — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation. Reward가 아니라 filter로 쓰는 같은 verifier를 다룬다.
- **ch-46** — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention. 이 장이 명시한 ablation을 실행한다.

---

## 출처

- [[prm800k]] — 레이블링 protocol, 800K/75K/12K scale, product aggregation 및 Table 4 비교, single-forward-pass inference, best-of-1860 결과, OOD STEM 표, credit-assignment argument(§6.1).
- [[let-verify]], [[lets-verify]] — 같은 논문을 다룬 오래된 library card. Per-step inference-cost와 calibration claim은 논문에 없으므로, 여기의 수치는 [[prm800k]]에서 가져온다.
- [[math-shepherd]] — MC label 정의(식 3-4), completer 설정, step-by-step PPO reward placement, GSM8K/MATH 표, label-accuracy 측정 및 N에 따른 하락, verifier-size 결과.
- [[omegaprm]] — binary-search labeling, `O(k log M)`, tree constant, soft target, Table 1 결과, 36% relative gain, question filtering, 명시된 limit.
- [[qwen-prm-lessons]] — PRM 대 value model, 세 방식 data 비교, consensus filtering, threshold sweep, BoN bias, process-to-outcome shift, label source별 aggregation, released-model 결과(arXiv:2501.07301v2 발췌).
- [[rl-on-incorrect-synthetic-data]] — first-pit 구성, per-step advantage, data efficiency 8x, arbitrary pairing의 실패, equivalence theorem(arXiv:2406.14532v1 발췌).
- [[rewarding-progress-pav]] — progress-as-advantage 정의, effective reward 식 5, prover 조건, search 및 RL 결과, hyperparameter(arXiv:2410.08146v1 발췌).
- [[step-dpo]] — 첫 erroneous step으로 구성한 step-level preference pair.
- [[rlvr-tulu3-config]] — Tülu 3 RLVR objective, α = 10, prompt 표, Table 21 hyperparameter와 70B conflict, Table 23 결과, over-optimization 및 value-init 발견(arXiv:2411.15124v5 발췌).
- [[rlvr-tulu3]], [[tulu-3]] — 같은 report의 library card. 이들의 RLVR code-verifier claim, “+5-10pp” gain, “mechanically zero” Goodhart claim은 논문이 뒷받침하지 않으며 §5에서 정정한다.
- [[ifbench]] — IFEval-to-IFBench gap, IF-RLVR gain과 동반된 loss, constraint-variety ablation, reward mixing.
- [[why-language-models-hallucinate]] — binary grading과 abstention, `t/(1 − t)` penalty 및 threshold, benchmark 분류(arXiv:2509.04664v1 발췌).
- [[swe-rl-difflib-reward]] — reward 정의, data funnel, training configuration, SWE-bench 및 repair-only 결과, reward ablation, out-of-domain 표, limitation(arXiv:2502.18449v2 발췌).
- [[swe-rl]] — 같은 논문의 library card. Base model, data scale, transfer 수치는 audit 이전 것이며 §6과 §7에서 정정한다.
- [[deepswe]], [[deepswe-recipe]] — hidden test를 쓴 execution reward, compact filtering, 결과, recipe conflict.
- [[deepseek-v3.1]] — DeepSeek-V3.2의 generated verifier function 및 agent-environment filtering.
- [[glm-4-5]] — format-violation halt가 있는 search-answer accuracy와 test-result reward.
- [[kimi-k2]] — verifiable-reward gym, instruction-following hack check, truncation penalty.
- [[crossing-the-reward-bridge]] — rule-checkable answer 비율, binary 및 soft model-based reward, scaling 비교, out-of-distribution 결과(arXiv:2503.23829v2 발췌).
- [[transferability-of-llm-reasoning]] — 통제된 Qwen3-14B SFT-versus-RL 비교와 PCA shift(arXiv:2507.00432v2의 해당 위치에서 인용).
- [[deepseek-r1]] — 성공하지 못한 시도 중 PRM 및 그 이유(G.2).
- [[rlvr-beyond-base-model]], [[prorl]], [[interplay-pretraining-midtraining-rl]] — pass@k 경계 논쟁과 RL이 capability를 확장하는 조건.
- [[spurious-rewards-rlvr]] — Qwen2.5-Math의 random-reward gain과 다른 family에서의 부재(arXiv:2506.10947v2의 해당 위치에서 인용).
- [[training-verifiers-to-solve-math-word-problems]] — ORM이 나온 sample-and-rank verifier.
- [[reward-model-overoptimization]] — fixed verifier가 제거하는 proxy-drift failure.
- [[natural-emergent-misalignment-reward-hacking]], [[metr-frontier-reward-hacking]] — unit test와 scoring code의 측정된 exploitation.
- [[grpo]], [[john-schulman-kl-tricks]] — §5에서 쓴 group-normalized advantage 및 KL estimator family.
- [[rubrics-as-rewards]] — ch-44b에서 개발하는 rubric reward의 pointer.
