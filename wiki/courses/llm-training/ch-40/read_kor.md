<!-- chapter: ch-40
     track: rl
     kind: content
     title: 그룹-베이스라인 RL: RLOO, GRPO, Dr. GRPO, DAPO, GSPO
     deps: [ch-42a]
     sources: [[rloo]], [[rloo-vs-grpo]], [[reinforce-plus-plus]], [[grpo]], [[deepseekmath]], [[dr-grpo]], [[dapo]], [[gspo]], [[vapo]], [[spurious-rewards-rlvr]], [[reinforcement-learning-with-one-training-example]], [[octothinker]], [[nathan-lambert-grpo]], [[deepswe]], [[deepswe-recipe]], [[gigpo-verl-agent]], [[negative-sample-reinforcement]], [[john-schulman-kl-tricks]], [[trl-grpo]], [[verl-grpo]], [[deepseek-r1-recipe]], [[magistral-recipe]], [[asymmetric-reinforce]], [[likelihood-displacement]]
     figures: figures/group-baseline.html
     revised: 2026-09 (generality revision)
-->

# Chapter 40 — 그룹-베이스라인 RL: RLOO, GRPO, Dr. GRPO, DAPO, GSPO

> **핵심 통찰.** 그룹-베이스라인 알고리즘은 PPO의 학습된 가치 모델을, 같은 프롬프트에 대해 샘플링한 여러 응답의 통계량으로 대체한다. 변형 알고리즘들은 네 지점에서 다르다. 어떤 베이스라인을 빼는가(leave-one-out 평균, 그룹 평균, 배치 평균), 어드밴티지를 분산 추정치로 나누는가, 토큰 손실을 하나의 스칼라로 어떻게 집계하는가, KL 항을 어디에 적용하는가이다. 이 선택 각각은 특정 샘플이 받는 그래디언트를 바꾼다. Dr. GRPO가 제거한 두 나눗셈, DAPO의 네 변경, GSPO의 시퀀스 수준 비율이라는 공개된 수정은 각각 측정된 실패, 즉 틀린 응답 길이의 증가, 엔트로피 붕괴, 그래디언트가 0인 그룹, Mixture-of-Experts 학습 붕괴에 연결되어 있다.
>
> **지침.** 보상이 검증 가능한 0/1 또는 ±1 신호이고 응답이 길다면, 표준편차 나눗셈 없이 길이에 독립적인 손실 정규화기를 갖는 그룹 베이스라인을 사용하라. [[dr-grpo]]는 Qwen2.5-1.5B에서 더 짧은 틀린 응답과 같거나 더 높은 정확도를 보고했고(§3.2, Fig. 5; App. C Fig. 8 및 3회 실행의 Fig. 9), [[dapo]]는 네 변경을 추가하면서 Qwen2.5-32B의 AIME 2024 avg@32가 30에서 50으로 상승했다고 보고한다(Table 1). 보상이 연속적인 reward-model 점수이고 그룹에 근소한 동점이 자주 있다면, 그룹 수준이 아니라 배치 수준에서 분산 정규화기를 유지하라. 그룹 수준 나눗셈은 0.02의 보상 차이를 최대 크기의 어드밴티지로 바꾸며(§5의 계산 예), [[reinforce-plus-plus]]는 그룹 수준 통계량이 모든 유한 그룹에서 편향되었다고 주장한다(App. A.1, Thm. 1). 큰 Mixture-of-Experts 모델을 롤아웃 배치당 여러 번 그래디언트 업데이트하며 학습한다면, 시퀀스 수준 importance ratio를 사용하라. [[gspo]]는 Qwen3-30B-A3B에서 GRPO가 수렴하려면 토큰 수준 비율에 routing-replay 우회법이 필요했다고 보고한다(§5.3). 가치 모델을 학습하고 워밍업할 수 있다면 critic이 쓸모없어졌다고 단정하지 말라. [[vapo]]는 같은 베이스 모델에서 DAPO의 50에 비해 가치 모델 사용 시스템이 AIME 2024 avg@32 60.4를 기록했다고 보고한다(§5.2, Table 1).

---

## 범용 모델에서 이 장이 중요한 이유

강화학습은 SFT 이후(또는 베이스 모델에 직접) 출시 모델이 처리해야 하는 과제 범위보다 좁은 프롬프트 집합에 적용된다. 이 장에서 인용하는 모든 공개 레시피는 수학 문제([[grpo]] §4.2, [[dr-grpo]] §3, [[dapo]] §3.5), SWE-Bench 형식의 저장소 과제([[deepswe]] §2.1), 또는 하나의 에이전트 환경군([[gigpo-verl-agent]] App.)처럼 하나의 좁은 조각에서 학습한다. 따라서 옵티마이저는 모든 파라미터를 바꾸는 동안 입력 분포의 좁은 조각만 본다. 알고리즘의 세 속성이 모델의 능력 폭이 얼마나 유지되는지를 결정한다.

1. **어떤 샘플이 그래디언트를 받는가.** 응답들이 모두 같은 보상을 받는 그룹은 어드밴티지가 0이므로 아무 기여도 하지 않는다(§5). 프롬프트 집합 대부분이 너무 쉽거나 너무 어렵다면 유효 배치가 줄어들고, 업데이트는 작고 대표성이 없을 수 있는 부분집합에 의해 좌우된다.
2. **음수 어드밴티지를 어떻게 적용하는가.** 틀린 응답의 확률을 낮추면, 모델이 이미 그럴듯하다고 여기는 다른 응답들로 확률 질량이 이동한다(§ Negative samples). 제어 없이 수행하면 출력 다양성이 줄어들며, 이는 큰 k에서 pass@k가 떨어지는 것으로 측정된다([[negative-sample-reinforcement]]).
3. **손실을 어떻게 정규화하는가.** 길이 정규화는 긴 틀린 응답이 길이에 비례하여 패널티를 받는지를 결정한다. 그렇지 않으면 긴 틀린 응답의 토큰당 패널티가 짧은 틀린 응답보다 작아지고, 학습 보상이 더 이상 좋아지지 않은 뒤에도 틀린 응답의 길이는 계속 증가한다([[dr-grpo]] §3.1, Fig. 5).

이 장은 이후 장에서도 반복되는 측정 질문에도 답한다. 수학 프롬프트에 대한 RL은 DeepSeekMath 7B의 Maj@K는 높였지만 Pass@K는 높이지 못했고([[grpo]] §5.2.2, Fig. 7), Qwen2.5-Math에서 공개된 여러 이득은 Llama나 OLMo에서 재현되지 않는다([[spurious-rewards-rlvr]] §3). 한 모델 계열에서 측정한 알고리즘 순위는 다른 모델 계열에 대한 증거가 아니다.

---

## §1 베이스라인 문제: 학습된 critic에서 샘플링된 동료 응답으로

**정의.** 시퀀스 수준 보상에 대한 정책 그래디언트 추정량은 `∇J = E[(R(x, y) − b) ∇ log π_θ(y | x)]`이다. 여기서 `x`는 프롬프트, `y`는 샘플링된 응답, `R`은 보상, `b`는 `y`에 의존하지 않는 베이스라인이다. 그러한 `b`는 어느 것이든 추정량의 편향을 바꾸지 않고 분산만 바꾼다([[rloo]] §2.2–2.3, ch-37에서 재정리).

**측정 가능한 문제.** 베이스라인이 없으면 프롬프트의 모든 샘플은 자기 보상의 부호 방향으로 밀린다. 한 프롬프트에서 reward model 점수가 `{5.2, 5.0, 4.9, 4.9}`라면 네 응답 모두 강화되고, 실제로 정보가 있는 것은 그들 사이의 차이뿐이다. 그러면 추정량의 분산은 업데이트가 따라야 할 프롬프트 내 차이보다 프롬프트 간 `R`의 수준에 의해 지배된다.

**PPO의 답과 언어 모델에서 달라지는 점.** PPO는 학습된 가치 모델 `V_φ(s_t)`로 `b`를 추정하고, 단계별 오류를 GAE로 결합한다. `Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}`, `δ_l = R_l + γV(s_{l+1}) − V(s_l)`([[dapo]] Eqs. 2–3, Schulman et al. 인용). LLM 환경의 두 속성이 이 계산을 바꾼다.

- **종결 보상.** 0이 아닌 유일한 보상이 마지막 토큰에서 오고 `γ = λ = 1`이면, 합은 `Â_t = R − V(s_t)`로 망원경처럼 소거된다. 각 토큰의 어드밴티지는 종결 보상에서 그 토큰 위치 상태의 가치를 뺀 값이다. `R − V(s_0)`가 아니다. 가치 모델은 여전히 모든 위치에서 조회된다. `λ < 1`이면 추정량은 부트스트랩된 가치를 섞으며, 토큰별 KL 패널티를 보상에 더하면(InstructGPT 및 RLOO 정식화, ch-38) 보상은 더는 종결 보상이 아니다.
- **가치 모델 오차.** 정책과 같은 크기의 가치 모델은 비용이 크고 긴 추론 사슬에서 맞추기 어렵다. [[vapo]]는 가치 모델 붕괴 때문에 Qwen2.5-32B에서 일반 PPO가 AIME 2024 avg@32 5에 도달했다고 보고한다(§5.2, Table 1).

**무엇이 이를 대체하는가.** 필요한 양이 `V(prompt) = E_{y∼π}[R(x, y)]`뿐이라면 프롬프트당 `G`개 응답을 샘플링해 이를 직접 Monte-Carlo 추정할 수 있다. 이것이 그룹 베이스라인이다. 문헌의 순서는 다음과 같다. leave-one-out 추정량은 Kool et al. 2019에서 왔으며 [[rloo]] §2.3이 출처로 인용한다. DeepSeekMath의 GRPO는 2024년 2월 arXiv:2402.03300으로 공개되었고([[grpo]]), Ahmadian et al.의 RLOO 연구는 같은 달 뒤에 arXiv:2402.14740으로 나왔다([[rloo]]).

**조건과 한계.** critic 제거가 확정된 결론은 아니다. Qwen2.5-7B-Instruct로 ALFWorld를 수행할 때 critic이 있는 PPO는 종합 80.4를 기록한 반면 GRPO는 77.6, RLOO는 75.5였지만, 1.5B에서는 순서가 뒤집힌다(PPO 54.4, GRPO 72.8)([[gigpo-verl-agent]] Table 1, 3개 seed). [[vapo]]는 32B에서 가치 모델 시스템이 DAPO보다 높다고 보고한다(§5.2). [[deepseek-r1-recipe]]는 16B MoE에서 반대 방향을 보고한다. GAE `λ = 0.95`의 PPO는 MATH에서 GRPO보다 나쁘고, `λ = 1.0`은 GRPO에 가깝다(v2 App. A.3, Fig. 4). 각 결과는 서로 다른 과제와 규모에서 나온 단일 연구 결과다.

---

## §2 RLOO: leave-one-out 베이스라인

**정의.** `k`개 응답 `y_1 … y_k ∼ π_θ(· | x)`를 샘플링하고 점수를 매긴 뒤, 각 샘플에 대해 *나머지* `k − 1`개 샘플의 평균 보상을 사용한다.

```
b_i = (1/(k−1)) Σ_{j≠i} R_j
∇J_RLOO ≈ (1/k) Σ_i [ R_i − b_i ] ∇ log π_θ(y_i | x)
```

`R_i = R(x, y_i)`이고 `k`는 프롬프트당 온라인 샘플 수다. 이 추정량은 [[rloo]] §2.3에 제시되어 있다.

**제외가 중요한 이유.** `b_i`는 `{R_j : j ≠ i}`에만 의존하므로, `x`가 주어졌을 때 `y_i`와 독립이고 추정량은 편향되지 않는다. 베이스라인에 `R_i`를 넣으면 베이스라인이 자신이 보정할 샘플에 의존한다. 그 결과 추정량은 같은 방향을 가리키지만 크기가 조정된다(§6).

**계산 예 (`k = 4`, 연속 보상).** 보상은 `{2.0, 1.0, 0.5, 0.5}`이고 평균은 1.0이다.

| 샘플 | `R_i` | `b_i`(나머지의 평균) | RLOO 어드밴티지 | `R_i − mean` |
|---|---|---|---|---|
| 1 | 2.0 | (1.0+0.5+0.5)/3 = 0.667 | +1.333 | +1.0 |
| 2 | 1.0 | (2.0+0.5+0.5)/3 = 1.0 | 0.0 | 0.0 |
| 3 | 0.5 | (2.0+1.0+0.5)/3 = 1.167 | −0.667 | −0.5 |
| 4 | 0.5 | 1.167 | −0.667 | −0.5 |

모든 RLOO 어드밴티지는 평균을 뺀 어드밴티지의 `4/3`배다. 이 항등식은 모든 `k`에서 정확하다.

```
R_i − (1/(k−1)) Σ_{j≠i} R_j = (k R_i − k R̄)/(k−1) = k/(k−1) · (R_i − R̄)
```

[[dr-grpo]]도 `G/(G−1)`로 같은 관계를 제시한다(§3.2, App. A). 두 추정량의 차이는 `k`가 커질수록 사라지는 항이 아니라 상수 배율이다.

**`k = 2`인 경우.** `b_1 = R_2`, `b_2 = R_1`이므로 어드밴티지는 `R_1 − R_2`, `R_2 − R_1`이다. 이는 선호도 손실 항(ch-39)이 아니라 정책 그래디언트 가중치로 쓰이는 쌍별 비교다.

**증거.** TL;DR 요약과 Anthropic-HH에서 Pythia-6.9B 및 Llama-7B를 사용하고, 참조 completion에 대한 모의 승률을 측정했다(arXiv:2402.14740v2의 Table 1; 각 행의 세 수는 TL;DR, Pythia-6.9B의 Anthropic-HH, Llama-7B의 Anthropic-HH 순서다). RLOO k=4는 77.9 / 43.7 / 64.1, RLOO k=2는 74.2 / 47.6 / 62.2, RAFT k=4는 73.2 / 42.1 / 63.3, 이동평균 베이스라인 REINFORCE는 70.7 / 37.9 / 55.3, PPO는 67.6 / 29.2 / 32.0, DPO는 66.6 / 39.0 / 61.9다. PPO 행은 그 연구에서 조정한 한 구성이다. 같은 논문은 PPO clipping을 제거하면 “약간의 향상”이 있고, 자신들의 설정에서는 배치당 평균적으로 손실이 clip되는 비율이 5% 미만이라고 보고한다(§3.2). 단일 연구 결과다.

**조건과 한계.** RLOO 실행은 512토큰 컨텍스트, 롤아웃 배치당 두 그래디언트 단계, 짧은 생성(TL;DR)을 사용했다. “clipping이 거의 필요 없다”는 결과는 이 조건에 속한다. 하나의 롤아웃 배치를 16개 미니배치로 나누는 레시피(DeepSeek-R1-Zero, [[deepseek-r1-recipe]] v2 §2.1)나 4개로 나누는 레시피([[gspo]] §5.1)는 마지막 미니배치 전까지 정책을 크게 이동시킨다. 그 경우 clip은 활성화된다(§7, §8).

---

## §3 REINFORCE++: 전체 배치에 걸친 정규화

**정의와 메커니즘.** [[reinforce-plus-plus]]는 PPO-clip surrogate와 critic 없는 어드밴티지를 유지하지만, 정규화 통계량을 한 프롬프트의 그룹이 아니라 전역 학습 배치 전체에서 계산한다. 두 변형은 다음과 같다(v9 §3.1–3.2).

```
REINFORCE++ (k ≥ 1):      A_{q,o_t} = r(o_1:T, q) − β Σ_{i=t..T} KL(i)          (Eq. 4)
                          A^norm   = (A − mean_batch(A)) / (std_batch(A) + ε)   (Eq. 5)
w/ Baseline (k > 1):      A′       = R − mean_group(R)                          (Eq. 6)
                          A^norm   = (A′ − mean_batch(A′)) / (std_batch(A′) + ε)(Eq. 7)
                          L        = L_PPO(A^norm) − λ · E[½ (log π_θ/π_ref)²]  (Eq. 8)
```

`KL(i) = log[π_θold(o_i | q, o_<i) / π_ref(o_i | q, o_<i)]`이다. `β`는 보상 내부에 적용하는 KL 계수다. `λ`는 별도의 k2 KL 손실에 가중치를 주지만, 그 값은 보고되지 않았다.

**주장.** 저자들은 분모가 `r_i`에 의존하므로, 그룹 수준의 `(r_i − mean)/std`가 모든 그룹 크기 `N ≥ 2`에서 편향된 추정량임을 증명한다(App. A.1, Thm. 1). 또한 `k = 4` 또는 `8`일 때 거의 같은 보상은 그룹 표준편차를 0에 가깝게 만들어 어드밴티지가 “폭발”한다고 주장한다(§2.2). 아래 §5는 이 효과의 크기를 구체적으로 보인다. 폭발은 유한하게 제한되며, 정확히 동일한 경우에는 큰 값이 아니라 0이 된다.

**증거.** 약 700K쌍으로 학습한 Bradley–Terry reward model과 20,000개 프롬프트를 사용한 Llama-3-8B-SFT에서, k = 1인 REINFORCE++의 Chat-Arena-Hard는 46.7이었고 k = 4인 GRPO는 46.8이었다(v9 Table 1). AIME-24 문제 30개로 학습했을 때, GRPO는 학습 pass@1 95.0, AIME-25 pass@1 0.0, pass@16 0.4에 도달했다. 반면 REINFORCE++는 각각 71.0 / 2.5 / 40.0에 도달했다(v9 Table 2; 모델은 명시되지 않음). 도구 사용 average@32는 w/ Baseline 24.10, GRPO 22.58, PPO 21.85였다(v9 Table 4). 각 비교는 보고된 seed가 없는 단일 실행이다.

**범용 모델에 대한 함의.** 30문제 실험은 그룹 정규화 목적함수가 작은 프롬프트 집합에는 맞지만 전혀 전이하지 않을 수 있다는 이 계열의 가장 분명한 공개 경고다. held-out 연도의 pass@16 0.4(표에 인쇄된 단위)는 문제마다 16번 시도해도 정답 풀이가 거의 나오지 않았다는 뜻이다. 그와 동시에 학습 정확도는 95.0이었다. 프롬프트 집합 구성은 ch-16에서 다룬다.

---

## §4 GRPO: 그룹 z-score와 KL 항의 위치

**정의.** 질문 `q`에 대해 `π_θold`에서 `G`개 출력을 샘플링하고 점수를 매긴 다음, 출력 `i`의 모든 토큰에 같은 어드밴티지를 준다(DeepSeekMath §4.1.2, [[grpo]]에 인용).

```
Â_{i,t} = (r_i − mean(r_1 … r_G)) / std(r_1 … r_G)
```

**목적함수(논문의 Eq. 3).**

```
J_GRPO(θ) = E[q, {o_i}]  (1/G) Σ_i (1/|o_i|) Σ_t { min[ ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t} ] − β D_KL[π_θ || π_ref] }
ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θold(o_{i,t} | q, o_{i,<t})
```

`|o_i|`는 출력 `i`의 토큰 길이, `ε`는 clipping 파라미터, `β`는 KL 계수, `π_ref`는 참조 정책이다. 논문은 `ε`나 롤아웃 temperature를 보고하지 않는다([[grpo]] Verification).

**KL 항: 추정량과 배치 위치.** DeepSeekMath의 Eq. 4는 `π_ref/π_θ − log(π_ref/π_θ) − 1`이다. Schulman의 노트를 인용하며 “항상 양수임이 보장된다”고 설명한다([[grpo]] §4.1.1). 그 노트([[john-schulman-kl-tricks]], 2020년 3월)는 `q`에서 샘플링한 값에 대해 `KL[q, p]`를 구하는 세 Monte-Carlo 추정량을 제시한다. 여기서 `r = p(x)/q(x)`다.

| 추정량 | 수식 | 편향 | 분산 | 부호 |
|---|---|---|---|---|
| k1 | `−log r` | 비편향 | 높음(“샘플의 절반에서 음수”) | 어느 쪽이든 가능 |
| k2 | `½ (log r)²` | 편향됨(KL과 2차까지 일치하는 f-divergence) | 낮음 | ≥ 0 |
| k3 | `(r − 1) − log r` | 비편향(k1에 평균 0인 control variate `r − 1`을 더함) | 낮음 | ≥ 0 |

계산 예. 한 토큰에서 `π_θ = 0.5`, `π_ref = 0.4`이면 `r = 0.8`이고, k1 = 0.2231, k2 = 0.0249, k3 = 0.0231이다. `π_θ = 0.4`, `π_ref = 0.5`이면 `r = 1.25`이고, k1 = −0.2231이다. 음수가 될 수 없는 양을 추정하는데도 음수가 된다. k2 = 0.0249, k3 = 0.0269다.

이 계열에서는 KL의 위치가 서로 다르며, 이는 로깅만이 아니라 그래디언트를 바꾼다.

| 알고리즘 | KL 적용 위치 | 추정량 |
|---|---|---|
| PPO / InstructGPT, [[rloo]] | 어드밴티지 계산 전 토큰별 보상에 더함 | log ratio(k1 형식) |
| GRPO, DeepSeek-R1-Zero | 손실에 더함 | k3([[grpo]] Eq. 4; [[deepseek-r1-recipe]] v2 §2.1, 계수 0.001) |
| [[reinforce-plus-plus]] | 보상 안에 k1, w/ Baseline은 별도 k2 손실을 추가 | k1 / k2 |
| [[dr-grpo]], [[dapo]], Magistral, DeepSWE | KL 항 없음(`β = 0`) | — |

보상 안의 KL 항은 어드밴티지 정규화를 통과하지만, 손실 안의 KL 항은 통과하지 않는다. [[reinforce-plus-plus]]는 App. B.1에서 k2가 reverse KL에 대한 적절한 별도 손실 추정량이고, 손실로 쓰인 k3는 forward KL을 추정한다고 주장한다(저자들의 해석. verl은 두 종류를 모두 노출한다: `kl_loss_type: kl(k1), abs, mse(k2), low_var_kl(k3)`, [[verl-grpo]]).

**증거.** 약 144K개의 GSM8K/MATH 형식 질문에서 GRPO로 DeepSeekMath-Instruct 7B를 DeepSeekMath-RL 7B로 학습했을 때, GSM8K는 82.9 → 88.2, MATH는 46.8 → 51.7, MGSM-zh는 73.2 → 79.6, CMATH는 84.6 → 88.8이었다(Table 5). 논문은 7B에서 GRPO와 PPO의 직접 정확도 비교를 보고하지 않는다([[grpo]] Guideline). 단일 연구 결과다.

---

## §5 그룹 어드밴티지의 부호 구조와, 그룹이 신호를 전혀 전달하지 않는 경우

이진 보상에서는 누가 강화되는지에 관한 모든 것이 하나의 수, 즉 `G`개 샘플 중 정답인 샘플 수에서 나온다. 아래 표는 `G = 8`, 정답 보상 1, 오답 보상 0일 때 세 규칙에서 정답 샘플과 오답 샘플이 받는 어드밴티지를 보인다. “GRPO (pop)”는 모집단 표준편차 `√(p(1−p))`로 나눈다. “GRPO (sample)”은 verl과 TRL이 쓰는 표본 표준편차인 `torch.std`로 나눈다([[verl-grpo]] L319–321).

| 8개 중 정답 | 평균 | 모집단 std | GRPO(pop) ✓ / ✗ | GRPO(sample) ✓ / ✗ | Dr. GRPO ✓ / ✗ | RLOO ✓ / ✗ |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | — / 0 | — / 0 | — / 0 | — / 0 |
| 1 | 0.125 | 0.331 | +2.646 / −0.378 | +2.475 / −0.354 | +0.875 / −0.125 | +1.000 / −0.143 |
| 2 | 0.250 | 0.433 | +1.732 / −0.577 | +1.620 / −0.540 | +0.750 / −0.250 | +0.857 / −0.286 |
| 4 | 0.500 | 0.500 | +1.000 / −1.000 | +0.935 / −0.935 | +0.500 / −0.500 | +0.571 / −0.571 |
| 6 | 0.750 | 0.433 | +0.577 / −1.732 | +0.540 / −1.620 | +0.250 / −0.750 | +0.286 / −0.857 |
| 7 | 0.875 | 0.331 | +0.378 / −2.646 | +0.354 / −2.475 | +0.125 / −0.875 | +0.143 / −1.000 |
| 8 | 1.000 | 0.000 | 0 / — | 0 / — | 0 / — | 0 / — |

이 표의 상호작용 버전은 [figures/group-baseline.html](figures/group-baseline.html)이다. 선택한 `G`, 성공 수, 연속 보상 모드에 대해 각 열을 다시 계산한다. 표에서 다음 네 사실이 따른다.

1. **모두 같은 그룹은 신호를 전달하지 않는다.** 모든 `r_i`가 같으면 분자 `r_i − mean`은 정확히 0이다. 따라서 분모와 관계없이 어드밴티지는 `0/(0 + ε) = 0`이다. verl 구현은 이를 명시한다. `ε = 1e-6`일 때 `scores[i] = (scores[i] − id2mean) / (id2std + 1e-6)`이다([[verl-grpo]] L272, L319–329). 어드밴티지 0은 발산이 아니라 코드 수준의 사실이다. [[dapo]] §3.2도 같은 해석을 제시하며 그 결과를 유효 배치의 축소라고 부른다.
2. **z-score는 상한이 있다.** 모집단 표준편차를 쓰면 그룹 어드밴티지의 절댓값은 `√(G−1)`을 넘을 수 없다. `G = 8`에서는 2.646이고, 정확히 하나만 다른 행에서 도달한다. 큰 어드밴티지는 균일한 그룹에서가 아니라 거의 균일한 그룹에서 나온다.
3. **쉬운 프롬프트에서는 실패가 신호를 전달한다.** 8개 중 7개가 정답이면 정답 응답들은 각각 +0.378을 받고, 하나의 실패는 −2.646을 받는다. 모델이 대부분 푸는 프롬프트 집합에서 그룹-베이스라인 RL은 그래디언트 관점에서 모델 자신의 드문 실패를 대상으로 학습한다.
4. **표준편차로 나누면 프롬프트의 가중치가 다시 매겨진다.** 한 그룹에서 `|A|`를 합하면, 8개 중 1개 정답일 때 GRPO(pop)는 2.646 + 7 × 0.378 = 5.29, Dr. GRPO는 0.875 + 7 × 0.125 = 1.75다. 4개 정답일 때 GRPO는 8.0, Dr. GRPO는 4.0이다. 어려운 그룹과 균형 잡힌 그룹 사이의 비율은 나눗셈이 있으면 0.66, 없으면 0.44다. 이것이 [[dr-grpo]] §3.1이 부르는 “difficulty bias”다.

**그룹이 무효가 되는 빈도.** 한 프롬프트의 샘플당 성공 확률이 `p`이고 샘플들이 독립이라면, 크기 `G`인 그룹이 전부 정답이거나 전부 오답일 확률은 `p^G + (1−p)^G`다.

| `p` | G = 4 | G = 8 | G = 16 |
|---|---|---|---|
| 0.1 | 0.656 | 0.431 | 0.185 |
| 0.2 | 0.411 | 0.168 | 0.028 |
| 0.5 | 0.125 | 0.008 | ~0.000 |
| 0.9 | 0.656 | 0.431 | 0.185 |

`p = 0.9`, `G = 8`이면 그룹의 43%는 아무 기여도 하지 않는다. [[dapo]] Fig. 3b는 정확도 1인 프롬프트의 비율이 학습 중 상승함을 보인다. 프롬프트 간 효과가 아니라 시간에 따른 같은 효과다. 두 대응은 프롬프트 필터링(ch-16)과 동적 샘플링(§7)이다.

**연속 보상: 근소한 동점의 증폭.** reward model의 그룹 보상 `{0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.50}`은 평균 0.500, 모집단 표준편차 0.0112다. z-score는 `{0, +0.89, −0.89, 0, +1.79, −1.79, 0, 0}`이다. reward-model 잡음일 수 있는(ch-41) 0.02 보상점 차이가 정답과 오답의 구분과 같은 크기의 어드밴티지가 된다. 나누지 않으면 어드밴티지는 `±0.01`–`0.02`에 머물고 해당 프롬프트의 기여는 작다. 연속 보상에서 그룹 수준 표준편차 나눗셈이 만드는 실패 모드는 모두 같은 경우가 아니라 바로 이것이다.

---

## §6 Dr. GRPO: 길이 항과 표준편차 항 제거

[[dr-grpo]](arXiv:2503.20783, 2025년 3월; COLM 2025)는 Eq. 3에서 그래디언트를 편향시키는 두 항을 찾아 둘 다 제거한다. 모든 실행에서 `β = 0`이므로 KL 항은 없다(App. G Table 6).

### 6.1 `1/|o_i|` 응답 길이 항

**메커니즘(§3.1).** 각 응답의 토큰 손실은 그 응답이 실제로 생성한 길이로 평균 낸다. 정답 응답(`Â > 0`)에서는 짧은 응답이 더 큰 토큰당 업데이트를 받는다. 오답 응답(`Â < 0`)에서는 같은 총량이 더 많은 토큰에 퍼지므로, 긴 응답의 토큰당 패널티가 더 작다.

**계산 예.** 같은 그룹의 두 오답 응답, `Â = −0.5`, `G = 8`, 토큰별 비율 ≈ 1이라고 하자.

| 집계 방식 | 100토큰 응답, 토큰당 가중치 | 1000토큰 응답, 토큰당 가중치 | 총 패널티 비율(1000 : 100) |
|---|---|---|---|
| GRPO `(1/G)(1/\|o_i\|)` | −0.5/(8·100) = −6.25e-4 | −0.5/(8·1000) = −6.25e-5 | 1 : 1 |
| L = 3000인 Dr. GRPO `(1/G)(1/L)` | −0.5/(8·3000) = −2.08e-5 | −2.08e-5 | 10 : 1 |
| DAPO 토큰 수준 `1/Σ_i \|o_i\|` | 배치의 모든 토큰에서 동일 | 동일 | 10 : 1 |

GRPO에서는 오답 응답에 잘못된 토큰 900개를 추가해도 총 기여량은 바뀌지 않는다. 길이에 독립적인 정규화기에서는 더 긴 오답 응답이 10배의 패널티를 누적한다. [figures/group-baseline.html](figures/group-baseline.html)의 패널 3은 임의의 두 길이, 어드밴티지, 상수 정규화기에 대해 이 표를 다시 계산한다.

**코드에서.** TRL은 `loss_type`으로 정규화기를 선택한다([[trl-grpo]], commit a08e713, `grpo_trainer.py` L2548–2562, 원문 그대로).

```python
if self.loss_type in ["grpo", "sapo"]:
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "bnpo":
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
elif self.loss_type in ["cispo", "dapo", "vespo"]:
    normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
    loss = (per_token_loss * mask).sum() / normalizer
```

TRL의 기본값은 `loss_type="dapo"`, `beta=0.0`, 그룹 std 어드밴티지 스케일링이다. 이는 DeepSeekMath 정식화가 아니므로, 재현에서는 셋 모두를 명시적으로 설정해야 한다([[trl-grpo]] Guideline). verl은 같은 선택을 `actor_rollout_ref.actor.loss_agg_mode`으로 노출한다. 기본값은 `token-mean`이며, Dr. GRPO에는 `use_kl_loss: False`, `algorithm.norm_adv_by_std_in_grpo: False`와 함께 `seq-mean-token-sum-norm`이 문서화되어 있다([[verl-grpo]], docs/algo/grpo.md L57–62).

### 6.2 `/std(r)` 난이도 항

Dr. GRPO의 어드밴티지는 §5에 이미 표로 보인 열인 `Ã_{i,t} = R(q, o_i) − mean(R)`이다. 평균은 샘플 자신을 포함하므로 이 추정량은 leave-one-out 추정량의 `(G−1)/G`배다. App. A는 RLOO와의 `G/(G−1)` 관계를 제시한다.

**증거.** MATH 문제에서 R1 템플릿을 쓴 Qwen2.5-1.5B는 Dr. GRPO 사용 시 출력 길이가 더 이상 증가하지 않았고, 벤치마크의 오답 응답 길이가 GRPO보다 짧았다(§3.2, Fig. 5; 논문은 표가 아니라 곡선을 보고한다). 1.5B ablation에서 두 항 중 어느 하나만 따로 제거해도 일반 GRPO보다 보상과 정확도가 상승했다(App. C, Fig. 8; Fig. 9는 3회 실행에 걸쳐 GRPO와 Dr. GRPO를 반복 비교한다). 공개된 Dr. GRPO 모델은 Qwen2.5-Math-7B 기반 Oat-Zero-7B다. 3k토큰 예산에서 AIME24 43.3, AMC 62.7, MATH500 80.0, Minerva 30.1, OlympiadBench 41.0을 기록했고(Table 4), A100 8장으로 27시간 학습했다.

**조건과 한계.** 이것은 Qwen2.5 및 Llama-3.2 베이스, 수학 전용 프롬프트, `β = 0`, 3000토큰 예산에서 수행한 한 논문이다. “더 짧은 오답 응답으로 같거나 더 좋은 정확도”의 증거이지, 엄격한 개선의 증거는 아니다. 이후의 두 자료는 특히 표준편차 항에 관해 다르다. [[reinforce-plus-plus]]는 분산 정규화기를 유지하되 배치 전체에서 계산해야 한다고 주장하고(§3.1, App. A), [[gigpo-verl-agent]] §5.2는 선택이 과제 의존적이라고 찾았다. `F_norm = 1`은 더 어려운 ALFWorld 하위 과제와 WebShop에서 더 높은 성공률을 보였고, 다른 곳에서는 두 설정이 비슷했다(Table 1, 3개 seed). [[nathan-lambert-grpo]]는 같은 길이 정규화 주장을 실무자 관점에서 요약한 글(2025년 3월)이며, 독립 실험은 보고하지 않는다.

---

## §7 DAPO: 한 번에 하나씩 측정한 네 변경

[[dapo]](arXiv:2503.14476, 2025년 3월)는 Qwen2.5-32B base를 정답 최종 답에는 `+1`, 그 외에는 `−1`을 주는 규칙 기반 보상(Eq. 7), KL 항 없음(§2.3)으로 학습하고, 각 변경이 AIME 2024 avg@32에 미친 효과를 보고한다.

| 구성 | AIME24 avg@32 |
|---|---|
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (DAPO) | 50 |
| DeepSeek-R1-Zero-Qwen-32B(참조) | 47 |

(Table 1. 행마다 한 번 실행했으며, 행은 누적되므로 어떤 행도 최종 시스템 대비 단일 변경을 분리하지 않는다.)

**Clip-Higher(§3.1).** clip 범위를 `ε_low`, `ε_high`로 분리한다(Eq. 10). 실행에서는 각각 0.2, 0.28이다. 주장은 산술적이다. `ε = 0.2`와 양의 어드밴티지에서 `π_θold = 0.9`인 토큰은 1.08까지 올라갈 수 있어 실질적 상한이 없다. `π_θold = 0.01` 토큰은 0.012로 제한된다. 위쪽 경계만 올리면 저확률 토큰이 자랄 수 있다. 아래쪽 경계를 올리면 토큰들을 0으로 밀어 샘플링 공간을 붕괴시킨다. 측정한 효과는 entropy(Fig. 2b)와 정확도(Fig. 2a)다.

**Dynamic Sampling(§3.2).** `G`개 샘플이 모두 정답이거나 모두 오답인 프롬프트를 거르고, 0이 아닌 어드밴티지 그룹으로 배치가 찰 때까지 다시 샘플링한다. Eq. 11의 제약은 `0 < |{o_i : is_equivalent(a, o_i)}| < G`다. Qwen2.5-32B에서 DAPO를 재현한 verl은 dynamic sampling 사용 시 AIME 2024 정확도 52%, 미사용 시 50%, token-level loss와 dynamic sampling을 모두 빼면 44%를 보고한다(각각 한 번 실행. 44% 실행은 다른 하드웨어)([[verl-grpo]], docs/algo/dapo.md L36–38).

**토큰 수준 손실(§3.3).** `(1/G) Σ_i (1/|o_i|) Σ_t`가 `1/(Σ_i |o_i|) Σ_i Σ_t`(Eq. 12)가 된다. 따라서 배치의 모든 토큰은 같은 가중치를 받고, 긴 저품질 샘플은 길이에 비례하여 억제된다. Figure 4는 이것이 없을 때 entropy와 평균 응답 길이가 상승함을 보인다.

**과도하게 긴 응답 처리(§3.4) — long-context 내용.** 두 메커니즘은 대안 관계다.

- *Overlong Filtering*은 잘린 샘플의 손실을 mask한다. 길이 제한에서 잘린 응답은 패널티 대신 그래디언트가 없다.
- *Soft Overlong Punishment*(Eq. 13)는 버퍼 안에 길이 의존 항을 더한다. `|y| ≤ L_max − L_cache`이면 `R_length = 0`; `L_max − L_cache < |y| ≤ L_max`이면 `((L_max − L_cache) − |y|)/L_cache`; `L_max`를 넘으면 `−1`이다. `L_max = 20,480`, `L_cache = 4,096`일 때 18,432토큰 응답은 `(16,384 − 18,432)/4,096 = −0.5`를 받아 정답성 보상에 더한다.

둘 다 존재하는 이유는 보상 잡음이다. “건전한 추론 과정이 과도한 길이만으로 패널티를 받을 수 있다”(§3.4). 기록된 충돌도 있다. Table 1은 Overlong Filtering을 누적 단계로 넣지만, verl 유지보수자는 그것이 Overlong Reward Shaping과 겹치므로 최고 실행을 포함한 대부분의 논문 실험은 이를 쓰지 않았다고 말한다([[verl-grpo]], docs/algo/dapo.md FAQ). 길이 제어는 ch-44a다.

**여기서는 clip이 활성화된다.** DAPO 롤아웃 배치는 512 프롬프트 × 16 응답이고 미니배치는 512이므로, 롤아웃 배치당 16번 그래디언트 업데이트한다(§4.1). 첫 업데이트 뒤 샘플링 정책과 최적화 정책은 달라지고, `ρ_{i,t}`는 롤아웃 배치 안에서 1에서 멀어진다. 이 장의 해석은 이것이 RLOO에서는 clipping이 거의 없었던 반면 DAPO의 분리된 clip 경계가 정확도를 바꾸는 이유다(해석. 두 논문 모두 배치당 업데이트 수에 대한 clipped-token 비율을 보고하지 않음). 따라서 RLOO 시대의 “clipping은 거의 필요 없다”는 관찰은 여기로 전이하지 않는다.

**DAPO의 결과가 아니다.** 최고 entropy 토큰만으로 학습하는 것은 별도 결과다(arXiv:2506.01939, “Beyond the 80/20 Rule”). TRL은 논문 값을 ρ = 0.2로 두는 entropy-quantile mask로 이를 구현하며 config docstring에서 그 논문에 귀속한다([[trl-grpo]], grpo_config.py L276–281). DAPO의 네 기법 중 하나가 아니다.

---

## §8 GSPO와 VAPO: 비율과 critic의 재검토

**GSPO(arXiv:2507.18071, 2025년 7월, Qwen 팀).** 토큰 수준 importance ratio `w_{i,t} = π_θ(y_{i,t} | x, y_{i,<t}) / π_θold(y_{i,t} | x, y_{i,<t})`는 각 다음-토큰 분포에 대한 한 샘플 보정이다. 저자들은 이것이 분포 보정 역할을 할 수 없고, 긴 응답에서 누적되는 분산을 주입한다고 주장한다(§3). GSPO는 이를 길이 정규화된 시퀀스 비율로 바꾸고 응답 전체를 clip한다(Eqs. 5–7).

```
s_i(θ) = ( π_θ(y_i | x) / π_θold(y_i | x) )^(1/|y_i|)
       = exp( (1/|y_i|) Σ_t log [ π_θ(y_{i,t} | x, y_{i,<t}) / π_θold(y_{i,t} | x, y_{i,<t}) ] )
J_GSPO(θ) = E[ (1/G) Σ_i min( s_i(θ) Â_i, clip(s_i(θ), 1−ε, 1+ε) Â_i ) ],   Â_i = (r_i − mean) / std
```

**계산 예.** 네 토큰 응답의 토큰별 비율이 `{1.10, 0.95, 1.30, 0.90}`이면 평균 log 비율은 0.0503이고 `s_i = 1.0515`다. `ε = 0.2`인 토큰 수준 clipping에서는 1.30 토큰만 clip된다. 보고된 GSPO 범위(3e-4, 4e-4)로 시퀀스 clipping하면 응답 전체가 범위 밖이어서 그래디언트에서 제외된다. 두 목적함수의 clip 범위는 비교할 수 없으며, 측정된 clipped-token 비율은 두 자릿수 차이다. GSPO 약 0.15, GRPO 0.0013(Fig. 2)인데도 GSPO는 컴퓨트 단위당 더 높은 학습 보상에 도달한다(Fig. 1).

**증거와 한계.** Qwen3-30B-A3B-Base에서 cold-start fine-tuning하고 각 롤아웃 배치를 네 미니배치로 나눈 경우, GSPO는 안정적으로 학습하며 학습 보상, AIME'24, LiveCodeBench, CodeForces Elo에서 조정된 GRPO baseline(clip 0.2 / 0.27)보다 빠르게 개선했다(Fig. 1 — 최종 표 없는 곡선). MoE에서는 48층 Qwen3-30B-A3B-Base에서 하나의 그래디언트 업데이트 뒤 같은 응답에 활성화되는 expert의 약 10%가 바뀌어 토큰 수준 비율을 무효화한다. GRPO는 수렴하려면 Routing Replay 우회법이 필요했으나 GSPO는 필요 없었다(§5.3, Fig. 3). seed나 표가 없는 단일 연구 결과다.

**VAPO(arXiv:2504.05118, 2025년 4월).** DAPO와 같은 Qwen2.5-32B base와 같은 벤치마크에서, 가치 모델 시스템은 5,000 step 안에 AIME 2024 avg@32 60.4를 기록했다. DAPO 50, DeepSeek-R1-Zero-Qwen-32B 47과 비교되며, 3회 반복은 60–61이었다(§5.2). 구성요소와 각각을 뺀 비용(Table 1)은 다음과 같다. value-pretraining(보상 모델에서 50 step 가치 warm-up; 제거 시 11), `λ_critic = 1.0`인 decoupled GAE(제거 시 33), `α = 0.05`에서 `λ_policy = 1 − 1/(αl)`인 길이 적응형 정책 λ(제거 시 45), Clip-Higher 0.2/0.28(46), token-level loss(53), 가중치 0.1의 positive-example NLL loss(54), 그룹 샘플링(8,192 프롬프트 × 1 대신 512 프롬프트 × 16; 55)이다. 길이 적응 규칙의 계산 예: `l = 100`은 `λ_policy = 0.8`, `l = 1000`은 0.98이므로 긴 응답은 장거리 credit을 유지하고 짧은 응답은 낮은 분산을 유지한다.

이 장의 두 결론. 첫째, critic이 있어도 프롬프트당 반복 샘플링은 도움이 된다. VAPO 자체 ablation에서 그룹 샘플링 행은 5점을 낸다. 둘째, “critic-free”는 한 계열 시스템의 속성일 뿐 측정된 법칙이 아니다. 비교는 규모, 과제, 가치 모델에 투입한 엔지니어링에 의존한다.

---

## §9 그룹-베이스라인 RL이 바꾸는 것과, 어떤 모델에서 바뀌는가

**Maj@K는 오르고 Pass@K는 오르지 않는다(DeepSeekMath 7B).** SFT와 RL 체크포인트를 temperature 0.7, `K` 최대 64로 평가하면 RL은 Maj@K를 개선하지만 Pass@K는 개선하지 않는다. 저자들은 이를 “근본 능력의 향상”이 아니라 “TopK에서 정답 응답을 boost”한 것으로 설명한다([[grpo]] §5.2.2, Fig. 7; 저자들의 해석).

**좁은 프롬프트, 더 넓은 개선.** DeepSeekMath의 RL은 GSM8K/MATH 형식 chain-of-thought 질문만 사용했으며, 저자들은 다른 벤치마크를 domain 밖으로 본다. MGSM-zh 73.2 → 79.6, CMATH 84.6 → 88.8도 개선됐다(Table 5). RL 뒤 비수학 벤치마크는 보고되지 않았으므로, 이것은 언어를 가로지르는 수학 내부 전이의 증거이지 일반 능력이 보존됐다는 증거는 아니다.

**양수와 음수 그래디언트는 coverage에 다른 효과를 낸다.** MATH에서 프롬프트당 8 rollout으로 Qwen2.5-Math-7B를 학습한 [[negative-sample-reinforcement]]는 RLVR 그래디언트를 양수 부분(PSR)과 음수 부분(NSR)으로 분리한다. MATH pass@1 / pass@256은 base 63.2 / 96.9, GRPO 76.3 / 95.5, PSR-only 74.1 / 91.2, NSR-only 75.7 / 96.9, W-REINFORCE(양수를 `λ = 0.1`로 down-weight) 76.6 / 96.7이다. AIME 2025 pass@256은 base 46.7, GRPO 50.0, PSR 43.3, NSR 53.3, W-REINFORCE 56.7이다(Table 1). 정답 응답만 강화하면 pass@1을 얻는 대신 coverage를 잃는다.

**모델 계열 의존성.** [[spurious-rewards-rlvr]]는 정보량이 줄어드는 보상 함수로 300 step GRPO(프롬프트당 16 rollout, 64프롬프트 rollout 배치, LR 5e-7)를 수행한다. Qwen2.5-Math-7B에서 무작위 보상은 MATH-500에 +21.4점, ground-truth 보상은 +29.1점이다(Abstract). Llama3.1-8B-Instruct 및 OLMo2-7B에서 같은 허위 보상은 작은 이득 또는 손실을 낸다(§3, Fig. 1). 제안한 메커니즘은 높은 사전 확률 행동을 증폭하는 clipping bias다. 세 가지 방식으로 clipping을 끄면 무작위 보상은 일관된 이득을 내지 않는다(§4, Fig. 4). 논문은 one-shot RLVR와 test-time-training 신호도 Qwen에서는 재현되지만 다른 계열에서는 종종 재현되지 않는다고 보고한다(§3, App. E Fig. 15). [[reinforcement-learning-with-one-training-example]]의 one-shot 결과는 Qwen2.5-Math-1.5B가 하나의 학습 예로 MATH500 36.0% → 73.6%, 6개 벤치마크 평균 17.6% → 35.7%가 된다는 것이다. 그러나 같은 절차의 Llama-3.2-3B-Instruct는 6개 벤치마크 평균이 한 예에서 17.5 → 19.0, 두 예에서 21.0이고, 전체 1,209개 예에서는 19.8이다(Table 4).

**베이스 모델의 기여.** [[octothinker]]는 동일한 GRPO 설정(MATH8K 프롬프트, batch 128, 쿼리당 16 rollout)을 Llama-3.2-3B-Base와 Qwen2.5-3B-Base에 적용해 RL 후 MATH500 10.0, 66.4를 보고한다. Llama를 정제 수학 코퍼스로 200B token mid-training하고 20B decay token을 추가한 뒤에는 RL 후 65.2가 된다(Fig. 1, 그림에 표기된 값). 저자들의 결론은 RL 확장성은 RL 시작 전에 결정된다는 것이다.

**함의.** 알고리즘 비교를 공개할 때 베이스 모델 이름을 명시하고, 적어도 한 비교를 두 번째 계열에서 반복하라. 무작위 보상 실행은 값싼 control이다. 이득 대부분이 재현되면 실험은 보상의 정보가 아니라 pretrained 행동의 elicitation을 측정한 것이다.

---

## §10 다중 턴 에이전트를 위한 그룹 베이스라인(미리 보기)

하나의 종결 보상이 있는 많은 단계의 trajectory에 이 장의 대수를 적용하는 두 방법이며, 둘 다 ch-45b에서 완전히 다룬다.

**Trajectory 수준 그룹, leave-one-out, KL 없음.** DeepSWE-Preview는 Qwen3-32B를 4.5K R2E-Gym 문제에서만 RL로 학습하고 SWE-Bench-Verified Pass@1 42.2%(16회 실행 평균)를 보고한다([[deepswe]]). GRPO++는 DAPO의 clip-high와 KL loss 제거, Dr. GRPO의 표준편차 항 제거 및 최대 컨텍스트 길이 정규화, RLOO의 leave-one-out 베이스라인을 결합하고, 저자 변경 두 개를 추가한다. entropy loss를 쓰지 않으며 compact filtering으로 context limit, step limit, 20분 generation timeout에서 끝난 trajectory의 손실을 mask한다(§2.3). rollout 8개 중 성공 2개면 실패 어드밴티지는 `0 − 2/7 ≈ −0.29`, 성공은 `1 − 1/7 ≈ 0.86`이다. 실패 8개면 모든 어드밴티지는 0이다([[deepswe]] §2.3에서 도출).

**반복 상태에서의 단계 수준 그룹.** [[gigpo-verl-agent]]는 두 번째 그룹 수준을 추가한다. 한 과제의 trajectory들은 같은 초기 상태에서 시작하므로 같은 환경 상태가 trajectory 사이에 다시 나타나며, 그 “anchor state”에서 취한 행동들이 단계 수준 그룹을 이룬다. 각 행동은 할인 return `R_t = Σ_{k≥t} γ^{k−t} r_k`으로 점수화되며 총 어드밴티지는 `A(a_t) = A_E(τ_i) + ω · A_S(a_t)`다(Eqs. 5–8). 조정 없이 `ω = 1`, `γ = 0.95`를 쓴다. 그룹화에 추가 rollout은 필요 없다. Qwen2.5-7B-Instruct의 ALFWorld에서 GiGPO 종합 성공률은 90.8, GRPO 77.6, RLOO 75.5, PPO 80.4였다(Table 1, 3개 seed).

---

## 음수 샘플과 음수 피드백

**어떤 종류의 음수인가.** 이 장의 음수는 **그래디언트로서의 음수**다. 어드밴티지가 그룹 평균보다 낮은 응답은 토큰 log-probability가 감소한다. 그룹-베이스라인 RL은 이를 기계적으로 만든다. 그룹 평균을 베이스라인으로 쓰면 일부 샘플은 반드시 평균 아래에 있어야 한다.

**어디에서 오고 오류율은 얼마인가.** verifier(답 일치, unit test)나 reward model이 각 샘플 응답에 라벨을 붙인다. verifier 오류는 보통 보고되지 않는다. DeepSeekMath는 인간 process supervision이 라벨 출처일 때 PRM800K에 약 20%의 잘못된 annotation이 있다고 인용한다([[grpo]] §5.2.3, footnote 7). reward model을 쓰면 §5에서 보았듯 그룹 내 0.02 잡음은 그룹 표준편차로 나눈 뒤 최대 크기의 음수가 된다.

**메커니즘.** 어휘에 대한 softmax에서 `∂ log p_y / ∂z_j = 1[j = y] − p_j`다. 샘플링된 토큰 `y`의 음수 어드밴티지는 `z_y`를 `η|A|(1 − p_y)`만큼 낮추고, 다른 모든 logit `z_j`를 `η|A| p_j`만큼 높인다. 제거된 확률 질량은 현재 확률에 비례하여 재분배된다. 토큰 `(a, b, y)`에 대해 `p = (0.6, 0.3, 0.1)`, `η|A| = 1`인 계산 예에서 `y`를 낮추면 `(0.710, 0.263, 0.026)`이 된다. 이미 확률이 높은 `a`가 제거된 질량 대부분을 흡수하며, `b`는 logit이 높아졌는데도 확률은 떨어진다. 같은 업데이트 크기 `η|A| = 1`로 고확률 토큰 `a`를 낮추면 `(0.438, 0.441, 0.120)`이 된다. 첫 경우 가장 큰 단일 토큰 확률 변화는 0.110, 둘째는 0.162다. 두 결과가 따른다. 드문 실패를 패널티하면 현재 mode에 분포가 집중되고, 그럴듯한 토큰을 패널티하면 한 step에서 정책이 크게 이동한다. ch-43a는 이의 likelihood-displacement와 squeezing 버전을 도출하며, [[likelihood-displacement]]는 preference optimization 사례를 제시한다.

**현재 관행은 무엇을 하는가.** 제한을 두고 음수를 유지한다. clip은 토큰별 비율을 제한하며, 아래 경계는 음수 어드밴티지 토큰에 작용한다. DAPO는 `ε_low`를 일부러 올리지 않는다. 그렇게 하면 확률을 0으로 밀어 sampling space가 붕괴하기 때문이다(§3.1). TRL의 `delta` 옵션은 `ρ`의 위쪽을 cap하며, 정의상 음수 어드밴티지에만 영향을 준다([[trl-grpo]], L2508–2515에서 도출). DeepSWE와 DAPO는 잘리거나 timeout된 trajectory를 mask하여, 불확실한 실패가 패널티 대신 그래디언트를 받지 않게 한다.

**효과 크기에 대한 증거.** Qwen2.5-Math-7B의 PSR/NSR 분해(§9)에서 음수 전용 학습은 MATH pass@256을 base 수준인 96.9에 유지하면서 pass@1을 75.7로 개선했다. 양수 전용은 비슷한 pass@1 74.1이지만 pass@256은 91.2로 낮아졌다. 양수를 `λ = 0.1`로 down-weight한 것이 그 연구에서 최선의 조합이었다. `λ ≤ 0.2`는 비슷했고, `λ = 1`(일반 REINFORCE)은 MATH pass@256 92.0으로 `λ = 0.1`의 96.7보다 낮았다(Table 1; App. E에서 `λ ∈ {0, 0.05, 0.1, 0.2, 1}` ablation). 한 모델 계열, 한 데이터셋의 결과이므로 §9의 Qwen2.5-Math 주의가 여기에도 적용된다.

**off-policy RL의 관련 control.** [[asymmetric-reinforce]]는 Llama-3.1-8B-Instruct와 MATH에서 rollout이 stale이고 베이스라인이 behavior policy의 평균 보상에 놓일 때(`δV = 0`), 학습 및 테스트 정확도가 7회 실행 모두에서 붕괴했음을 보인다. `δV = −0.1`의 7회 실행, 즉 그룹 평균보다 약간 낮은 베이스라인으로 양수에 더 큰 가중치를 주는 경우는 더 안정적이라고 보고한다. 더 큰 `δV > 0`에서는 붕괴가 더 이르게 온다(§5.2, Figs. 3–5). 베이스라인이 그룹 평균 대비 어디에 있는지는 적용하는 음수 신호의 양을 조절하는 control 자체다.

**진단.** 각 step에서 다음을 기록하라. 어드밴티지가 0인 그룹 비율(TRL의 `frac_reward_zero_std`), 부호별 평균 어드밴티지 크기, 어드밴티지 부호별 clipped-token 비율(TRL은 `A < 0`일 때만 low-clip, `A > 0`일 때만 high-clip을 센다), 정답성별 평균 응답 길이, entropy, 그리고 held-out set에서 pass@1과 큰 k의 pass@k.

**일반성에 미치는 효과.** 음수 그래디언트는 양수 전용 학습보다 coverage(pass@k)를 더 잘 유지한다. 또한 그룹 메커니즘상 쉬운 프롬프트에서는 신호 거의 전부가 음수다(§5). 위험은 부호가 아니라 라벨이다. 정확하지만 드문 응답을 거짓 음수로 라벨링하면, 바로 분포를 넓힌 행동을 제거하게 된다.

---

## 레시피

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DeepSeekMath-RL 7B | 7B | RL | 알고리즘; 질문당 출력 G; KL 계수(손실 내); 정책 LR; 최대 길이; 학습 배치 | GRPO; 64; 0.04; 1e-6; 1024; 1024(단위 미기재) | arXiv:2402.03300v3 §4.2 ([[grpo]]) | verified 2026-09-14 | Table 5: MATH 46.8 → 51.7; 이 값들의 ablation 없음 |
| DeepSeekMath-RL 7B | 7B | RL | clip ε; rollout temperature | 보고되지 않음(§4.1–4.2 확인) | — | not reported | — |
| DeepSeek-R1-Zero | 671B MoE | RL | 질문당 출력; step당 질문; LR; KL 계수(손실 내); rollout 분할 | 16; 32; 3e-6; 0.001; 8,192개 출력을 16개 mini-batch, 1 inner epoch로 분할 | arXiv:2501.12948v2 §2.1 ([[deepseek-r1-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| DeepSeek-R1(stage-1 RL) | 671B MoE | RL | clip ε | 10 | v2 §3.2.1 ([[deepseek-r1-recipe]]) | verified 2026-09-14 | §3.2.1: 낮은 ε는 그래디언트를 자르고, 높은 ε는 불안정화할 수 있음(표 없음) |
| Oat-Zero-7B(Dr. GRPO, Qwen2.5-Math-7B 기반) | 7B | RL | 질문당 응답; temperature; 최대 응답; KL loss 및 penalty; clip ε; inner epochs; LR | 8; 1.0; 3000; 0.0 및 0.0; 0.2; 1; 1e-6 constant, AdamW(0.9, 0.95) | arXiv:2503.20783v2 App. G Table 6 ([[dr-grpo]]) | verified 2026-09-14 | Fig. 8 / Fig. 9는 1.5B에서 두 bias term을 ablate; 이 값들의 ablation 없음 |
| Oat-Zero-7B | 7B | RL | loss normalizer; 컴퓨트 | 상수 MAX_TOKENS = 3000; A100 8장 27 h | Listing 1 + App. G; §1 ([[dr-grpo]]) | derived; verified 2026-09-14 | — |
| DAPO(Qwen2.5-32B base) | 32B | RL | prompt batch; 프롬프트당 응답; mini-batch; LR; ε_low/ε_high; 최대 생성; KL | 512; 16; 512(rollout step당 16 update); 20-step warm-up의 1e-6 constant; 0.2 / 0.28; 20,480 = 16,384 + 4,096 cache; 없음 | arXiv:2503.14476v2 §4.1, §2.3 ([[dapo]]) | verified 2026-09-15 | Table 1 누적 ablation(30 → 50 AIME24 avg@32) |
| DAPO | 32B | RL | 보고된 최고 실행의 overlong filtering | Table 1은 누적 단계로 열거; verl docs는 논문의 최고 실행이 이를 쓰지 않았다고 말함 | arXiv:2503.14476v2 Table 1; verl docs/algo/dapo.md FAQ | conflict | — |
| VAPO(Qwen2.5-32B base) | 32B | RL | actor LR; critic LR; λ_critic; λ_policy; ε_low/ε_high; prompts × samples; positive LM loss weight; value warm-up | 1e-6; 2e-6; 1.0; 1 − 1/(0.05·l); 0.2 / 0.28; 512 × 16; 0.1; 50 step | arXiv:2504.05118v3 §5.1 ([[vapo]]) | verified 2026-09-15 | Table 1: 구성요소별 제거 비용 49, 27, 15, 14, 7, 6, 5점 |
| GSPO(cold start from Qwen3-30B-A3B-Base) | 30B MoE | RL | sequence-clip 범위(left/right); rollout batch당 mini-batch; GRPO baseline clip | 3e-4 / 4e-4; 4; 0.2 / 0.27 | arXiv:2507.18071v2 §5.1 ([[gspo]]) | verified 2026-09-15 | Fig. 1 곡선; Fig. 2 clipping 비율 0.15 vs 0.0013 |
| RLOO, TL;DR의 Pythia-6.9B | 6.9B | RL | k; rollout batch; step batch; β(보상 내 KL); LR; steps; 배치당 gradient steps | 2 또는 4; 512; 256; 0.03; 1e-6 constant, 3% warm-up; 600; 2 | arXiv:2402.14740v2 App. “Preference Training” ([[rloo]]) | verified 2026-09-15 | LR은 {1e-6, 1e-5, 2e-5} sweep에서 선택 |
| Anthropic-HH의 RLOO, Pythia-6.9B / Llama-7B | 6.9B / 7B | RL | β; steps; batches | 0.10; 393(Pythia); 2 epoch에 걸쳐 2048 rollout 및 step batch(Llama) | 같은 위치([[rloo]]) | verified 2026-09-15 | Table 1 win-rate |
| REINFORCE++ v1 runs(Llama3.1-8B-SFT, Qwen2.5-7B-Instruct) | 8B / 7B | RL | 프롬프트당 샘플; β; clip ε; actor LR; γ | 4; 일반 0.01, 수학 0.001; 0.2; 5e-7; 1.0 | arXiv:2501.03262 v1 §4.2 Table 1 ([[reinforce-plus-plus]]) | verified 2026-09-14 | ablation 보고 없음 |
| DeepSWE-Preview | 32B | RL | advantage; std 정규화; KL loss; ε_high; 문제당 trajectory; LR; loss normalizer | leave-one-out; off; off; 0.28; 8; 1e-6; 최대 context 길이 | blog §2.3; released script L26–L56 ([[deepswe-recipe]]) | verified 2026-09-14 | Fig. 6: 14B에서 compact filtering ablation(곡선만) |
| GiGPO, ALFWorld(Qwen2.5-1.5B / 7B-Instruct) | 1.5B / 7B | RL | group size; rollout당 groups; γ; ω; KL loss coef; actor LR; max steps | 8; 16(128 environments); 0.95; 1(조정 없음); 0.01; 1e-6; 50 | arXiv:2505.10978v3 App. “Hyperparameters for ALFWorld” ([[gigpo-verl-agent]]) | verified 2026-09-15 | Table 1(3개 seed) |
| Magistral Medium / Small | not reported / 24B | RL | KL; advantage; zero-advantage groups; ε_high | 없음; `r_i − μ` 후 minibatch 정규화; 제거됨; 0.26–0.28(Small: 0.3) | arXiv:2506.10910v1 §2.1, §5.3 ([[magistral-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| Spurious-reward runs(Qwen2.5-Math-7B 및 다른 7 모델) | 1.5B–8B | RL | 프롬프트당 rollout; rollout batch; mini-batch; LR; temperature; steps | 16; 64 prompts; 128 rollouts; 5e-7 constant; 1.0; 300 | arXiv:2506.10947v2 App. D ([[spurious-rewards-rlvr]]) | verified 2026-09-15 | Fig. 1(모델별 결과) |

**작은 범용 실행의 시작점.** 검증 가능한 0/1 보상과 약 3,000토큰 미만 응답을 쓰는 1.5B–7B 모델에는 Dr. GRPO 행이 가장 직접적인 근거가 있는 구성이다. `G = 8`, temperature 1.0, KL 항 없음, `ε = 0.2`, inner epoch 하나, AdamW(0.9, 0.95)와 1e-6 constant LR, `r_i − mean` 어드밴티지, 생성 예산과 같은 상수로 정규화한 손실이다(arXiv:2503.20783v2 App. G Table 6. 이 설정은 MATH level 3–5 프롬프트를 사용해 Qwen2.5-Math-7B에서 A100 8장으로 27시간 만에 Oat-Zero-7B를 만들었다). 채택 전 더 큰 규모의 증거가 필요한 변경 두 개는 `ε_high = 0.28`과 dynamic sampling이다. 이는 프롬프트당 16 응답, 20,480토큰 예산의 32B 실행에서 나왔다([[dapo]] §4.1). GSPO clip 범위는 30B MoE 실행에서 나왔다(§5.1). 보상이 verifier가 아니라 reward-model 점수라면 “정규화 없음” 선택을 검증 없이 옮기지 말라. §5와 [[reinforce-plus-plus]] §3.1을 보라.

---

## 일반화 관점

**(a) 무엇이 폭을 넓히는가.**

- 그래디언트의 음수 부분을 유지하는 것: NSR-only와 W-REINFORCE(`λ = 0.1`)는 MATH pass@256을 base 모델의 96.9에 유지하고, base 46.7 대비 AIME 2025 pass@256을 53.3, 56.7로 올린다. 양수 전용은 91.2, 43.3으로 내린다([[negative-sample-reinforcement]] Table 1; Qwen2.5-Math-7B에만 해당).
- entropy 붕괴를 막는 것: Clip-Higher는 DAPO Fig. 2에서 entropy와 AIME 정확도를 함께 높인다. [[reinforcement-learning-with-one-training-example]]는 1-shot 설정에서 saturation 이후 이득을 위해 entropy 항이 필요하다고 보고한다(Abstract, §4). 다양성과 entropy는 ch-43이다.
- 풀 수는 있지만 아직 풀지 못한 그룹이 0이 아닌 비율로 남는 프롬프트 집합: §5는 pass rate가 0 또는 1에 가까워질 때 그룹이 얼마나 빨리 무효가 되는지 정량화한다(ch-16).
- RL 전 mid-training: 같은 RL 레시피는 Llama-3.2-3B-Base에서 MATH500 10.0, 220B토큰 mid-training 뒤 65.2를 기록한다([[octothinker]] Fig. 1).

**(b) 무엇이 좁아짐 또는 망각을 일으키는가.**

- 모델이 이미 선호하는 경로에 질량을 집중하고 큰 k의 pass@k를 낮추는 양수 전용 강화([[negative-sample-reinforcement]] Table 1).
- noisy continuous reward에서 reward-model 잡음을 최대 세기 그래디언트로 올리는 그룹 수준 표준편차 나눗셈(§5).
- 틀린 응답이 길어지도록 허용하는 길이 정규화 손실([[dr-grpo]] §3.1, Fig. 5): 과제 coverage가 아니라 사용할 수 있는 출력의 좁아짐이다.
- 작은 프롬프트 집합: AIME-24 30개에 대한 GRPO는 학습 pass@1 95.0, AIME-25 pass@16 0.4였다([[reinforce-plus-plus]] v9 Table 2).
- 학습 보상 추적: DAPO는 최종 학습 보상이 “종종 validation set 정확도와 상관이 거의 없으며, 이는 training set overfitting을 나타낸다”고 보고한다(§4.3).

**(c) 이 단계에서 어떻게 측정하는가.** 평균 정확도만이 아니라 held-out set에서 pass@1 및 큰 k의 pass@k를 보고하라. RL domain 밖 벤치마크를 적어도 하나 전후 평가하라. 두 번째 모델 계열에서 대표 비교를 반복하라. 무작위 보상 control([[spurious-rewards-rlvr]])을 실행하라. dead group 비율, truncation 비율, entropy, 정답성별 응답 길이를 기록하라. 알려진 측정 오류는 seed 없는 단일 실행 비교(§7, §8의 대부분), 하나의 변경을 분리할 수 없는 누적 ablation 표(DAPO Table 1), RL 전 base 모델을 실제보다 약하게 보이게 하는 template 효과([[dr-grpo]] §3.3)다.

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| 모두 같은 그룹이 엄청난 어드밴티지를 만든다고 믿음 | 없음 — 그룹은 조용히 기여하지 않음 | zero-advantage 그룹 비율(TRL의 `frac_reward_zero_std`)을 기록하고, 측정된 pass rate의 `p^G + (1−p)^G`와 비교 |
| reward-model 점수에 그룹-std 나눗셈을 유지 | 보상 범위가 수백분의 일인 그룹에서 ±1~±2 어드밴티지 | 그룹별 보상 범위와 최대 어드밴티지 크기를 함께 기록; 배치 수준 정규화 또는 없음으로 변경 |
| clip이 결코 걸리지 않는다고 가정 | 양쪽 부호 평균이라 clip 비율이 거의 0으로 보고됨 | `A > 0`, `A < 0`별 clipped-token 비율 기록; rollout 배치당 gradient update 수 계산 |
| TRL/verl 기본값을 “GRPO 논문”으로 사용 | 재현이 공개 수치와 다름 | TRL 기본은 `loss_type="dapo"`, `beta=0.0`, group-std scaling; verl 기본은 `adv_estimator=gae`, `rollout.n=1`, `token-mean` — 각각 명시 설정 |
| 잘린 응답을 실패로 점수화 | truncation 비율과 평균 길이가 함께 상승; entropy 불안정 | truncation 비율 기록; overlong filtering 또는 soft punishment 적용([[dapo]] §3.4) |
| 긴 chain에 길이 정규화 손실 사용 | 오답 평균 길이가 정답보다 더 빨리 증가 | 정답성별 길이 기록([[dr-grpo]] Fig. 5); normalizer 변경 |
| scale을 고려하지 않고 G값 또는 RLOO와 Dr. GRPO 비교 | 한 실행이 다른 learning rate를 필요로 하는 듯 보임 | 두 어드밴티지는 `G/(G−1)`배 다름; z-score는 대략 `1/std`로 크기를 바꿈 |
| 한 모델 계열의 알고리즘 순위를 결론으로 삼음 | Qwen에서는 이득이 재현되고 다른 곳에서는 사라짐 | 두 번째 계열에서 같은 비교와 무작위 보상 control 실행([[spurious-rewards-rlvr]] §3) |
| 학습 보상을 진전으로 읽음 | 학습 보상은 오르지만 validation 정확도는 평평함 | validation set을 hold out하고 그 값으로 gate([[dapo]] §4.3) |
| 큰 MoE 정책에 토큰 수준 비율 사용 | 정상처럼 시작한 뒤 학습 보상이 되돌릴 수 없이 붕괴 | `π_θold`, `π_θ`의 expert-activation overlap 기록; sequence-level ratio 고려([[gspo]] §5.3) |

---

## 이해 점검

1. 검증 가능한 과제에서 8 rollout 그룹이 모두 0을 반환한다. GRPO와 Dr. GRPO 모두에서 이 그룹이 어드밴티지 수식상 왜 그래디언트를 만들지 않는지, 분모의 `+1e-6`이 왜 이를 바꾸지 않는지 설명하라.
2. 8개 중 7개 정답이면 한 실패의 어드밴티지는 −2.65, 각 성공은 +0.38이다. 대부분 푸는 프롬프트 집합에서 어떤 샘플이 업데이트를 이끄는지와, 표준편차 나눗셈을 없애면 이 구조가 어떻게 되는지 설명하라.
3. DeepSeekMath Eq. 3은 각 응답 토큰 손실을 `|o_i|`로 나눈다. 이것이 1000토큰 오답의 토큰당 패널티를 100토큰 오답보다 10배 작게 하는 이유와, 총 시퀀스 손실이 그래도 같은 이유를 도출하라.
4. RLOO는 clip이 토큰 5% 미만에서 활성이라고 찾았고 DAPO는 clip 경계 분리가 정확도를 바꾼다고 측정했다. 설정별 rollout 배치당 gradient update 수로 차이를 설명하라.
5. GSPO는 GRPO 0.0013 대비 약 0.15를 clip하지만 컴퓨트당 더 높은 학습 보상에 도달한다. 저자들의 설명과 그 주장이 의존하는 토큰 수준 비율의 속성을 말하라.
6. 동료가 새 RL 변형이 Qwen2.5-Math-7B에서 MATH-500을 20점 개선했다고 보고한다. “보상이 모델에 무언가를 가르쳤다”와 “목적함수가 pretrained 행동을 증폭했다”를 구분할 control 두 개와 각각을 지지하는 결과를 설명하라.
7. 양수 전용 학습은 pass@1을 올리고 pass@256을 낮추며, 음수 전용은 pass@256을 유지한다. softmax logit gradient로 두 방향이 출력 분포 폭에 다르게 작용하는 이유를 설명하라.
8. DeepSWE는 표준편차 항을 없애고 leave-one-out 베이스라인을 쓰며 context/time limit trajectory를 mask한다. 세 선택이 각각 어떤 측정된 실패를 다루는지, 보상이 0/1이 아닌 continuous score면 어느 선택이 달라지는지 말하라.

---

## 연결

- **이전 장 — ch-42a, Narrow Training, Broad Behaviour Change: Emergent Misalignment, Sycophancy, and Trait Transmission.** 좁은 행동 조각에 적용한 목적함수가 모델 전체를 바꾸는 이유. 이 장은 RL 사례의 그래디언트 메커니즘을 제공한다.
- **다음 장 — ch-16, RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse.** §5의 dead-group 산술이 프롬프트 난이도 필터링이 존재하는 이유이며, dynamic sampling은 그 in-loop 대응물이다.
- **ch-37, Policy-Gradient Foundations for Language Models** — 이 장이 적용하는 추정량과 베이스라인 이론.
- **ch-38, KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax** — §1이 비교하는 critic, GAE, KL-in-reward 정식화.
- **ch-38a, SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity** — SFT 대비 RL이 바꾸는 것에 대한 더 넓은 증거.
- **ch-43, Entropy, Output Diversity, and KL Control in RL** — Clip-Higher와 entropy 항이 겨냥하는 entropy 붕괴.
- **ch-43a, Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages** — 이 장 음수 절의 메커니즘 완전 도출.
- **ch-44a, Length in RL: Overlong Responses, Length Control, and Long-Context RL** — overlong filtering과 soft punishment 자체를 다룬다.
- **ch-45b, Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability** — §10의 완전한 설명.
- **ch-55, verl Internals**, **ch-57, TRL Internals** — 여기 인용한 구현을 trainer 전체 수준에서 다룬다.

---

## 출처

- [[rloo]] — Ahmadian et al., arXiv:2402.14740(2024년 2월): leave-one-out 추정량, Table 1 win-rate, clipping과 hyperparameter 세부. Card에는 Verification 절이 없으며, 여기 수치는 v2에서 읽어 `excerpts/rloo.md`에 모았다.
- [[grpo]] — Shao et al., arXiv:2402.03300(2024년 2월): GRPO Eq. 3, KL 추정량 Eq. 4, outcome-supervision advantage, DeepSeekMath-RL 결과, Maj@K 대 Pass@K 분석.
- [[deepseekmath]] — [[grpo]]로 redirect하는 duplicate card; 증거로 별도 인용하지 않음.
- [[dr-grpo]] — Liu et al., arXiv:2503.20783(2025년 3월): 길이/난이도 bias, Dr. GRPO 목적함수, RLOO와 `G/(G−1)` 관계, App. G 레시피.
- [[reinforce-plus-plus]] — Hu et al., arXiv:2501.03262: 전역 advantage normalization, 그룹 수준 normalization 편향 증명, 작은 프롬프트 집합 overfitting 결과.
- [[rloo-vs-grpo]] — primary artifact 없는 비교 참고; axis 목록에만 사용. “limit에서 동치” 주장은 [[dr-grpo]] App. A의 정확한 `G/(G−1)` 항등식으로, “clip이 드물다”는 주장은 §7의 실행별 update 수로 대체했다.
- [[dapo]] — arXiv:2503.14476(2025년 3월), chapter excerpt: clip-higher, dynamic sampling, token-level loss, overlong shaping, Table 1, 학습 세부.
- [[gspo]] — arXiv:2507.18071(2025년 7월), chapter excerpt: sequence-level ratio, 그래디언트 비교, clipping 비율, MoE 안정성.
- [[vapo]] — arXiv:2504.05118(2025년 4월), chapter excerpt: DAPO와 같은 base의 가치 모델 시스템, 구성요소 ablation, 길이 적응형 GAE.
- [[spurious-rewards-rlvr]] — arXiv:2506.10947, chapter excerpt: Qwen2.5-Math에서 무작위 보상 이득, Llama와 OLMo에서 그 이득의 부재, clipping-bias 메커니즘과 그 ablation.
- [[reinforcement-learning-with-one-training-example]] — arXiv:2504.20571, chapter excerpt: 1-shot RLVR 수치와 Llama-3.2-3B-Instruct에서 그 효과의 크기.
- [[octothinker]] — arXiv:2506.20512, chapter excerpt: Llama와 Qwen base에 대한 동일 RL 레시피, 그리고 그 격차 대부분을 메우는 mid-training 개입.
- [[negative-sample-reinforcement]] — arXiv:2506.01347, chapter excerpt: PSR/NSR 분해, pass@k 표, `λ = 0.1`인 W-REINFORCE.
- [[deepswe]], [[deepswe-recipe]] — Agentica와 Together AI(2025년 7월): GRPO++ 구성요소, compact filtering, 8 rollout의 leave-one-out 어드밴티지.
- [[gigpo-verl-agent]] — arXiv:2505.10978, chapter excerpt: anchor-state 단계 그룹, `F_norm` 비교, ALFWorld와 WebShop 표.
- [[john-schulman-kl-tricks]] — k1/k2/k3 추정량, 그리고 k3가 비편향이고 음이 아닌 이유.
- [[trl-grpo]], [[verl-grpo]] — 인용한 두 구현: loss normalizer, advantage scaling, KL 추정량 옵션, zero-advantage 처리, dynamic-sampling 구성.
- [[deepseek-r1-recipe]] — R1-Zero와 R1 stage-1 RL 설정. App. A.3의 GAE-λ 비교를 포함한다.
- [[magistral-recipe]] — KL 항 없음, minibatch advantage normalization, zero-advantage 그룹 제거를 갖는 production GRPO 변형.
- [[asymmetric-reinforce]] — behaviour policy 평균 보상 대비 베이스라인의 위치와 `δV ≥ 0`에서 관찰된 붕괴.
- [[likelihood-displacement]] — 음수 절에서 쓰는 재분배 메커니즘의 preference-optimization 사례.
- [[nathan-lambert-grpo]] — 길이 정규화 수정의 실무자 추적(2025년 3월); 독립 실험은 없음.
