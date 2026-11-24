# 제43a장 — 부정 샘플과 부정 그래디언트: 우도 변위, squeezing, 부정 advantage

> **핵심 통찰.** 명시적인 부정 그래디언트는 모델에서 확률을 없애지 않는다. 그 위치에서 모델이 이미 가장 높게 순위를 매긴 대상 쪽으로 확률을 옮긴다. 따라서 그 효과는 벌점을 받은 샘플이 원래 얼마나 높은 확률이었는지에 달려 있다. 패널티를 받는 샘플이 정의상 현재 모델이 실제로 생성한 샘플인 on-policy 부정 샘플은, 이를 분리해 측정한 한 연구에서 출력 다양성을 보존했다. MATH로 학습한 Qwen2.5-Math-7B에서 negative-only 학습의 MATH pass@256은 96.9로, base model의 96.9와 같았고 positive-only 학습의 91.2보다 높았다. 반면 positive-only 학습은 올바른 학습 샘플의 비율을 가장 빨리 올렸으며 entropy를 가장 많이 잃었다([[negative-sample-reinforcement]] Table 1, Figures 5b-d). 패널티 대상 샘플이 현재 정책 아래에서 낮은 확률인 off-policy 부정 샘플은 확률 질량을 이미 가장 가능성 높은 continuation에 집중시키고([[learning-dynamics-llm-finetuning]] §3.3), 학습이 올리려는 응답의 우도까지 낮출 수 있다([[likelihood-displacement]] §6.2; [[gradient-entanglement-margin-alignment]] §3.1.1). 생성 모델에서 양·음의 기여 비율을 측정한 경우에도 개선의 대부분은 positive가 담당했다. NFT 32B 실행에서는 80% 대 20%였다([[nft-negative-aware-finetuning]] §5.3).
>
> **지침.** verifier가 라벨링한 부정 샘플이 on-policy 샘플이라면, 어느 한쪽을 버리지 말고 둘 다 loss에 남긴 채 positive 항의 가중치를 낮춰라. positive 항에 λ = 0.1을 준 경우, 동등 가중치의 pass@256 92.0보다 높은 96.7을 유지하면서 최고 pass@1과 같은 값(MATH에서 76.6)을 냈다([[negative-sample-reinforcement]] Tables 1 and 4). rollout이 오래된 정책에서 나왔다면 importance ratio로 부정 항을 제한하라(하한 없이 1에서 truncate, [[topr-tapered-off-policy-reinforce]] §3). 또는 baseline을 평균 보상보다 낮춰라([[asymmetric-reinforce]] §5.3). truncate하지 않은 부정 항에는 하한이 없기 때문이다. 이는 [[topr-tapered-off-policy-reinforce]]의 off-policy GSM8K 실행에서 퇴화한 생성을 만들었고(§2.1, Figure 1), [[asymmetric-reinforce]]에서는 δV = 0일 때 MATH에서 Llama-3.1-8B-Instruct의 7개 실행 모두가 train·test collapse를 보였다(§5.2, Fig. 5 left). 부정과 positive가 같은 종류의 응답일 때, 예를 들어 둘 다 refusal이거나 대부분의 token을 공유하는 두 trace일 때에는, 학습 전 pair를 걸러내거나 패널티를 서로 다른 token에만 국소화하라. 공유 token이 두 그래디언트를 결합시키기 때문이다([[likelihood-displacement]] §6.3; [[gradient-entanglement-margin-alignment]] §4.3; [[lazy-likelihood-displacement-grpo]] §5). parser가 놓친 경우, truncation, 도달할 수 없는 test처럼 실패 라벨이 틀렸을 수 있다면 패널티를 주지 말고 그 샘플을 mask하라. DAPO ablation에서 truncation된 샘플만 mask해도 AIME 2024 avg@32가 30에서 36으로 올랐다([[dapo]] Table 1).

## 이 장이 범용 모델에 중요한 이유

모든 post-training 단계는 실패를 만들며, 각 단계는 그 실패를 어떻게 처리할지 결정한다. [[ch-31a]]는 supervised 선택지, 즉 실패를 버리기, 실패를 입력에 넣기, label을 조건으로 주기, 또는 제한된 부정 항을 더하기를 다룬다. [[ch-39]]는 DPO와 그 변형의 rejected 항을 다룬다. [[ch-40]]은 집단 평균보다 낮은 샘플이 누군가 부정 항을 직접 작성하지 않아도 negative advantage를 받는 group-baseline RL을 다룬다. 이 장은 앞 장들이 가리키는 유도식을 모으고, 일반성에 대해 무엇이 측정되었는지를 명시한다.

범용 모델의 다음 세 성질이 걸려 있다.

1. **Coverage.** 큰 k에서의 pass@k는 모델이 여전히 생성할 수 있는 서로 다른 정답 풀이의 수를 측정한다. [[negative-sample-reinforcement]]의 Qwen2.5-Math-7B 실행에서 positive-only reinforcement는 pass@1을 올리는 동시에 pass@256을 base model보다 낮췄다(Table 1). 이는 [[rlvr-beyond-base-model]]이 RLVR 전반에서 보고한 방향과 같다.
2. **Retention.** 낮은 확률의 샘플에 적용한 부정 그래디언트는 패널티를 받은 token만이 아니라 그 위치의 전체 분포를 바꾼다([[learning-dynamics-llm-finetuning]] §3.3). 따라서 refusal을 포함해 한 번도 라벨링되지 않은 행동도 제거할 수 있다([[likelihood-displacement]] §6.2).
3. **Calibration and abstention.** 이진 verifier는 “모르겠습니다”를 오답과 정확히 같은 점수로 매긴다. 따라서 그것으로 학습한 부정 그래디언트는 abstention과 error를 함께 낮춘다([[why-language-models-hallucinate]] §4.1).

파이프라인에서의 위치는 preference optimization 및 RL이며, SFT로의 역참조가 있다. 측정 도구는 큰 k에서의 pass@k, held-out set에서의 entropy, chosen 및 rejected log-probability의 분리 기록, 그리고 advantage 부호로 나눈 통계다.

## §1 “negative gradient”의 의미와 부호의 발생원

[[ch-31a]] §1은 “negative”의 네 가지 의미를 정의한다. (1) negative marginal value: filter가 제거한 샘플, (2) content로서의 negative: 입력 또는 수정된 target에 넣은 실패, (3) conditioning으로서의 negative: inference에는 쓰지 않는 label 아래에서 학습한 실패, (4) gradient로서의 negative: 샘플의 likelihood를 명시적으로 낮추는 것. **이 장은 (4)의 의미만 다룬다.** (1)~(3)은 실패에서 확률을 멀어지게 옮기지 않으므로 displacement 또는 squeezing 위험이 없다.

RL에서 부호는 사람이 직접 선택하지 않는다. advantage에서 온다. 보상 r_i를 받은 하나의 prompt에 대한 G개 응답의 그룹에서, GRPO advantage는 Â_i = (r_i − mean(r))/std(r)이고 response i의 모든 token에 할당된다([[grpo]]; [[dr-grpo]] §3.1). 여기에는 세 가지 결과가 따른다.

- 이진 verifiable reward에서는 그룹 평균이 두 보상값 사이에 놓이므로, 정답 응답은 positive advantage를 받고 오답 응답은 negative advantage를 받는다. 샘플 기여의 부호는 그 샘플의 정오답이며 batch composition 때문에 뒤집힐 수 없다([[negative-sample-reinforcement]] §C).
- 학습된 reward model에서는 부호가 상대적이다. 같은 응답도 약한 batch에서는 positive advantage를, 강한 batch에서는 negative advantage를 받을 수 있다([[negative-sample-reinforcement]] §C). 따라서 advantage 부호로 나눈 diagnostic은 RLVR과 RLHF에서 서로 다른 의미를 갖는다.
- 한 그룹의 모든 응답이 같은 reward를 받으면, 모든 응답의 advantage는 0이고 그 그룹은 그래디언트에 기여하지 않는다([[dapo]] §3.2). 그러한 그룹을 버릴지, 유지할지, 재가중할지는 §5에서 다루는 설계 결정이다.

아래에서 **negative-only training**은 모델이 negative reward만 받는다는 뜻이 아니라 positive 항을 제거한다는 뜻이다. **push-down**은 sampling된 token 또는 response의 log π를 낮추는 그래디언트 단계다.

## §2 Softmax 업데이트: 제거된 확률은 어디로 가는가

**정의.** 한 위치에서 모델은 vocabulary 전체에 대한 logit $z$와 확률 $p_j = \exp(z_j)/\Sigma_i \exp(z_i)$를 낸다. sampling된 token $y$, advantage $A$, step size $\eta$에 대한 policy-gradient step은 logit을 다음과 같이 바꾼다.

```
Δz_j = η · A · (1[j = y] − p_j)
```

- $z_j$: token $j$의 logit, $p_j$: 현재 확률, $1[\cdot]$: 조건이 참이면 1, 거짓이면 0, $y$: 이 token을 포함하는 response에서 sampling된 token, $A$: 그 response의 advantage, $\eta$: step size.
- $A < 0$이면 sampling된 token의 logit은 $\eta|A|(1-p_y)$만큼 내려가며, **나머지 모든 logit은 $\eta|A|p_j$만큼 올라간다.** 즉 모델이 이미 부여한 확률에 비례해서 올라간다. 이는 [[ch-31a]] §2의 cross-entropy gradient에서 부호를 뒤집은 것과 같은 식이다.

**측정 가능하게 쓴 문제.** push-down 뒤 패널티를 받은 token은 확률을 잃는다. 범용성에 관심 있는 독자가 답해야 할 질문은 어느 token이 그 확률을 얻는지, 그리고 결과 분포가 더 평평해져 탐색을 보존하는지 또는 더 뾰족해져 탐색을 줄이는지다.

**계산 예시.** logit이 $(2, 1, 0, -1, -2)$인 다섯 token은 $p = (0.6364, 0.2341, 0.0861, 0.0317, 0.0117)$이고 entropy $H = 1.0000$ nat이다. $|A| = 1$, $\eta = 0.5$인 한 step을 적용한다.

| 업데이트 | sampling된 token | 새 $p$ | $ΔH$ (nats) |
|---|---|---|---|
| 최상위 token에 negative | $p_y = 0.6364$ | $(0.5720, 0.2837, 0.0969, 0.0347, 0.0126)$ | $+0.0751$ |
| 희귀 token에 negative | $p_y = 0.0317$ | $(0.6948, 0.2090, 0.0714, 0.0155, 0.0093)$ | $-0.1231$ |
| 최상위 token에 positive | $p_y = 0.6364$ | $(0.6959, 0.1899, 0.0752, 0.0284, 0.0106)$ | $-0.0883$ |
| 희귀 token에 positive | $p_y = 0.0317$ | $(0.5668, 0.2550, 0.1010, 0.0630, 0.0142)$ | $+0.1363$ |

두 번째 행은 첫 번째 행과 sampling된 token만 다르다. 희귀 token을 패널티하면 그 token에서 0.0162가 빠지고, 이미 가장 확률이 높았던 token에는 0.0583이 들어간다. token 2와 3도 확률을 잃는다. 즉 바로잡으려던 update가 분포를 더 날카롭게 만들었다. 반대로 최상위 token을 패널티하면 그 질량이 나머지 token으로 퍼지고 entropy가 올라간다.

**근거.** [[learning-dynamics-llm-finetuning]] §3.3은 모든 softmax head에 대해 같은 네 사실을 제시하고, multi-class logistic regression에서 이를 증명한다(App. E, Claims 1–5). 패널티 label의 확률은 내려간다. 줄어든 질량은 update 전 가장 확신하던 label로 대체로 압착된다. 고확률 dimension은 오르는 경향이 있고 저확률 dimension은 내려가는 경향이 있다. 분포가 더 뾰족할수록 더 많이 압착되며, 패널티 label의 확률이 작을수록 효과가 더 강하다. 저자들은 이를 **squeezing effect**라고 부른다. 증명에 대한 결과는 단일 연구이며, [[negative-sample-reinforcement]] §4.2의 token-level gradient 식에서 같은 결과가 재현된다.

**advantage 부호별 entropy.** [[entropy-mechanism-llm-rl]]은 population 수준의 버전을 제시한다. Lemma 1은 한 update가 policy entropy를 action의 log-probability와 그 logit 변화 사이 covariance의 음수만큼 근사적으로 바꾼다고 말한다. Theorem 1은 vanilla policy gradient에서 이를 $-\eta\cdot\operatorname{Cov}(\log \pi(a), \pi(a)\cdot A(a))$로, Theorem 2는 natural policy gradient에서 $-\eta\cdot\operatorname{Cov}(\log \pi(a), A(a))$로 평가한다(§3.1–3.2). 이 논문이 측정한 centered token covariance는 $\operatorname{Cov}(y_i) = (\log \pi(y_i) - \operatorname{mean}_j \log \pi(y_j))\cdot(A(y_i) - \operatorname{mean}_j A(y_j))$다(Eq. 10). 이를 앞의 계산 예시와 합치면 다음 네 사분면으로 읽을 수 있다.

|  | 평균보다 높은 확률 | 평균보다 낮은 확률 |
|---|---|---|
| **positive advantage** | centered product $> 0$ → entropy 하락(exploitation이 강화됨) | centered product $< 0$ → entropy 상승 |
| **negative advantage** | centered product $< 0$ → entropy 상승 | centered product $> 0$ → entropy 하락(squeezing) |

[[entropy-mechanism-llm-rl]]의 card는 논문이 부호 규칙은 말하지만 advantage 부호별로 나눈 통계나 결과는 보고하지 않는다고 기록한다. 따라서 이 사분면 표는 Eq. 10과 위 계산 예시를 이 과정에서 해석한 것이다(Interpretation). 또한 Clip-Cov와 KL-Cov는 $p\cdot A$가 아니라 이 centered product로 token을 고른다. 그래서 negative-advantage token도 선택 후보가 되지만, 가장 큰 $p\cdot A$ 순위로 고르면 그렇지 않다.

**범위를 제한한 변형.** 네 loss는 update에서 서로 다른 weight로 같은 token을 push-down한다.

| Loss | sampling된 token logit 하락에 곱해지는 factor | $p_y \to 0$일 때의 동작 | 출처 |
|---|---|---|---|
| Policy gradient push-down, $-A \log \pi$ | $(1-p_y)$ | 1에 접근: token이 이미 희귀할 때 가장 강함 | 위 §2 |
| NSR, 확률 $-R\cdot\pi$ 위에 쓴 loss | $\pi_y(1-\pi_y)$ | 0에 접근 | [[negative-sample-reinforcement]] Eq. 8 |
| Unlikelihood, $-\log(1-p_y)$ | $p_y$. scalar weight $p_y/(1-p_y)$가 동일한 $(1-p_y)$에 곱해지기 때문 | 패널티 token에서 0에 접근하며, token을 확신할수록 가장 큼 | [[negative-sample-reinforcement]] Eq. 11; [[ch-31a]] §5.1의 유도 |
| NPO, $-(2/\beta)\cdot\log \sigma(-\beta \log(\pi_\theta/\pi_\mathrm{ref}))$ | $W(1-p_y)$, $W = 2\pi_\theta^\beta/(\pi_\theta^\beta+\pi_\mathrm{ref}^\beta)$ | $\pi_\theta \ll \pi_\mathrm{ref}$가 되면 0에 접근 | [[negative-preference-optimization]] Eq. 5 |

$\beta = 1$인 NPO weight의 계산 예시: $\pi_\theta = \pi_\mathrm{ref}$일 때 weight는 1이다. $\pi_\theta/\pi_\mathrm{ref}=0.1$이면 0.182이고, 0.01이면 0.0198이다. $\beta=0.1$이면 같은 ratio에서 0.885와 0.774가 나온다. 이 때문에 논문은 $\beta\to0$일 때 NPO가 gradient ascent에 수렴함을 보인다(Proposition 1).

NSR 형태에는 실무적 주의점이 하나 있다. [[negative-sample-reinforcement]] Eq. 6은 loss를 $\log\pi$가 아니라 확률 $\pi$ 위에 쓰므로 추가 $\pi_y$ factor가 생긴다. 반면 PPO importance ratio $\pi_\theta/\pi_\mathrm{old}$를 쓰는 구현은 on-policy point에서 gradient가 $\nabla\log\pi_\theta$다. 따라서 논문이 설명한 $(1-\pi_y)$ damping은 유지되지만, 추가 $\pi_y$ factor는 유지되지 않는다(derived). 이 논문 주장에서 다른 설정으로 옮길 수 있는 부분은 확신 있는 token과 확신 없는 token에서 update 크기가 상대적으로 다르다는 점이며, 정확한 exponent가 아니다.

대화형 보조 자료 [figures/negative-gradient-lab.html](figures/negative-gradient-lab.html)는 위 표를 재현한다. logit을 바꾸고, update rule 및 advantage 부호를 고르고, step을 적용한 다음 entropy와 네 사분면 해석이 어떻게 변하는지 볼 수 있다. 기본값은 이 절의 수치를 재현한다.

## §3 Multi-token 응답과 공유 파라미터

응답 전체 수준에서는 update가 더 이상 국소적이지 않다. 패널티를 받는 응답은 학습이 올리려는 응답과 token, prefix, parameter를 공유한다.

**우도 변위(likelihood displacement).** [[likelihood-displacement]]는 loss가 내려가는 동안 $\log \pi(y^+\mid x)$의 평균이 감소하는 현상을 displacement로 정의한다. 거절(refusal) 실험에서 SORRY-Bench prompt에서 얻은 on-policy pair에 DPO를 적용하자 Llama-3-8B-Instruct의 training-set 거절률은 74.4%에서 33.4%로, Gemma-2B-IT의 거절률은 80.5%에서 54.8%로 낮아졌다(§6.2). Llama-3-8B-Instruct에서는 370개 pair 중 262개(71%)가 두 응답 모두 거절이었다. 두-거절 pair만으로, 또는 두-비거절 pair만으로 학습해도 거절률 하락이 발생했다(§6.2–6.3, Figure 4). 논문의 CHES(centered hidden embedding similarity) 점수는 두 응답의 hidden embedding이 얼마나 비슷한지 측정하며, length-normalized CHES가 가장 낮은 pair 5%만 남기도록 filter하자 거절률이 회복됐다(§6.3). 이 결과는 두 모델에서 수행한 단일 연구의 결과다.

**그래디언트 얽힘(gradient entanglement).** [[gradient-entanglement-margin-alignment]]는 모든 margin loss에 대한 1차 근사를 제시한다(Eqs. 4–5).

```
Δ log π_w ≈ C (‖∇ log π_w‖² − ⟨∇ log π_w, ∇ log π_l⟩)
Δ log π_l ≈ C (⟨∇ log π_w, ∇ log π_l⟩ − ‖∇ log π_l‖²)
```

- $\pi_w$, $\pi_l$: 선택된 응답과 거절된 응답의 확률이다. $\nabla\log\pi$: 그 확률의 $\theta$에 대한 gradient다. DPO에서는 $C=\eta\beta c(\theta)>0$이다.
- 조건 1: 선택 응답의 log-probability가 오르려면 inner product가 $\lVert\nabla\log\pi_w\rVert^2$ 이하이어야 한다. 거절 응답의 log-probability가 내려가려면 inner product가 $\lVert\nabla\log\pi_l\rVert^2$ 이하이어야 한다.

**계산 예시.** $\lVert\nabla\log\pi_w\rVert^2=1$, $\lVert\nabla\log\pi_l\rVert^2=4$, $\langle\nabla\log\pi_w,\nabla\log\pi_l\rangle=2$, $C=1$이라고 하자. 그러면 $\Delta\log\pi_w=1-2=-1$, $\Delta\log\pi_l=2-4=-2$다. margin은 여전히 1만큼 개선되지만 두 log-probability는 모두 떨어진다. 이는 Table 1의 Case 2이며, 논문은 거절 응답 gradient의 norm이 더 큰 경우가 많고, 그때 학습이 이 경우에 놓인다고 보고한다(§3.1.1). 관측 가능한 흔적은 Figure 1의 두 곡선이다. TL;DR에서 Mistral-7B-Instruct-v0.3는 끝에 두 log-probability가 모두 상승한 반면, Meta-Llama-3-8B-Instruct에서는 둘 다 하락했다.

**어느 token이 원인인가.** token 수준에서 두 응답이 실제로 다른 token(감성 설정에서 “positive” 대 “negative”)의 gradient는 inner product가 0보다 작다. 반면 두 응답이 공유하는 동일 token은 일부 token에서 cosine similarity가 1에 가깝다고 저자들이 설명한다(§4.3, Figure 4b). 얽힘은 대조되는 token이 아니라 공유 token이 만든다. 이것이 겉으로는 관계없어 보이는 두 권고의 기제다. 즉 응답이 거의 중복된 pair를 filter하고([[likelihood-displacement]]), 패널티를 서로 다른 token 또는 step에 국한해야 한다(아래 §5 및 [[ch-39]]의 step-level preference loss).

**응답 수준에서 질량은 어디로 가는가.** [[on-policy-suboptimal-preference-data]]의 bandit 설정(Figure 16)에서는 prompt 수가 적을 때 contrastive training이 $y_w$의 implicit reward를 올리고 $y_l$의 implicit reward를 내린다. 그러나 prompt 수가 늘면 $y_w$의 implicit reward도 내려간다. §5.2의 요지는 일반적인 조건이다. $y_l$이 $y_w$와 충분히 다르지 않고 capacity가 제한되어 있으면, “회복된 확률 질량은 $y_w$가 아니라 다른 응답들의 likelihood를 올리는 데로 간다.” AlpacaFarm의 Pythia-1.4B에서는 둘 다 하락했다. 반면 두 응답이 능력이 다른 모델에서 나와 더 뚜렷이 구분되는 UltraFeedback의 Mistral-7B에서는 선택 응답이 오르고 거절 응답이 내렸다(Figure 17). 어떤 경우가 되는지는 model capacity, dataset 크기, 두 응답이 얼마나 다른지가 결정한다.

**Off-policy가 악화 요인이며, 먼저 negative를 위로 끌어올리면 도움이 된다.** [[learning-dynamics-llm-finetuning]] §4.2는 off-policy DPO 경우를 직접 측정했다. probe한 거의 모든 응답의 log-probability가 떨어졌지만, greedy decoding한 응답의 log-probability는 8 epoch 안에 약 −113에서 −63으로 올랐다. 이들의 완화책은 DPO 시작 전 SFT 단계에 $(x,y^-)$를 넣는 것이다. 그러면 $y^-$는 더 이상 low-probability region에 있지 않다. Qwen1.5-1.8B와 5,000-example Anthropic-HH subset에서, 표준 pipeline에 맞선 이 pipeline의 GPT-3.5-Turbo 및 Claude3-Haiku 판단 win rate는 SFT 뒤 0.4729 / 0.4679였고, DPO 2, 4, 6 epoch 뒤에는 각각 0.6518 / 0.5151, 0.6928 / 0.6045, 0.6667 / 0.5432였다(App. F.3, Table 1). SFT 뒤 0.5 미만이고 preference training 뒤 0.5 초과라는 것은 preference training 전에는 뒤처진 pipeline이 후에는 앞선다는 뜻이다. 따라서 이 개입은 SFT checkpoint의 품질 자체보다 negative gradient가 도달하는 probability region을 통해 작동한다.

**On-policy sampling도 이 효과를 없애지는 못한다.** [[lazy-likelihood-displacement-grpo]]는 단일 GRPO update 아래 question별 likelihood 변화를 측정한다(Definition 4.1, Eq. 2). 많은 question에서 정답 응답 log-probability 변화가 작거나 음수이며, 이를 **lazy likelihood displacement**라고 부른다. negative advantage를 mask하는 “Pos Only”는 그러한 변화를 높이고, 일부 question에서는 낮춘다. 따라서 negative gradient가 일률적으로 해로운 것은 아니다. Pos Only의 평균 점수는 GRPO보다 1.3점 낮았다(Table 2: Qwen2.5-Math-1.5B에서 39.84 대 41.14). 변화가 가장 작은 question은 오답 응답이 거의 정답이거나, 정답이지만 output format이 틀린 경우다(§3, Figure 2). Theorem 4.4는 이 효과를 negative token의 hidden embedding과 정답 응답 사이 inner product에 돌린다. 논문의 GWHES(Group Weighted Hidden Embedding Score) 순위는 top-10 question에서 실제 likelihood-change 순위와 50% 및 60% 겹쳤고, random ranking은 17.5% 및 21.3% 겹쳤다(Table 1).

## §4 양성 강화와 음성 강화를 분리하여 측정하기

[[negative-sample-reinforcement]]는 RLVR 목적함수를 positive sample reinforcement(PSR)와 negative sample reinforcement(NSR)로 나누고, 두 항을 각각 단독으로 학습한다(Eqs. 2–4). 둘 다 현재 모델이 생성한 sample을 사용하므로, 이 실험이 분리하는 것은 data source가 아니라 gradient의 **부호**다. 아래 표의 **W-REINFORCE**는 논문에서 두 항을 결합한 방법이다. PSR 항에는 가중치 $\lambda$를 곱하고 NSR 항의 가중치는 1로 둔다(Eq. 9). 따라서 $\lambda=1$이면 일반 REINFORCE가 복원되고, $\lambda=0$이면 NSR만 남는다.

**결과(단일 연구, 세 model family).** MATH로 학습한 Qwen2.5-Math-7B를 256개 sample의 unbiased pass@k estimator로 평가한 결과다(Table 1).

| 방법 | MATH pass@1 | MATH pass@256 | AIME 2025 pass@256 | AMC23 pass@256 |
|---|---:|---:|---:|---:|
| Base model | 63.2 | 96.9 | 46.7 | 100.0 |
| PPO | 76.6 | 96.3 | 43.3 | 97.5 |
| GRPO | 76.3 | 95.5 | 50.0 | 97.5 |
| REINFORCE | 74.8 | 92.0 | 50.0 | 92.5 |
| PSR(positive만 사용) | 74.1 | 91.2 | 43.3 | 92.5 |
| NSR(negative만 사용) | 75.7 | 96.9 | 53.3 | 100.0 |
| W-REINFORCE($\lambda=0.1$) | 76.6 | 96.7 | 56.7 | 97.5 |

held-out test set에서 NSR의 entropy는 base model의 entropy 부근에 유지된다. PSR에서는 가장 빠르게 떨어지고, PPO와 GRPO에서는 그 중간 속도로 떨어진다(Figure 5b). NSR은 correct-sample ratio도 PSR보다 천천히 올리며, 완전히 푼 prompt의 비율도 더 낮은 지점에 도달한다(Figures 5c–d). 다시 말해 NSR은 모든 training prompt에 대해 모델이 같은 식으로 답하도록 밀어붙이지 않는다. $\lambda$ sweep(Table 4)은 시험한 범위에서 trade-off가 급격히 변하지 않음을 보인다. $\lambda=0, 0.05, 0.1, 0.2, 1$일 때 MATH pass@256은 각각 96.9, 97.1, 96.7, 95.9, 92.0이다.

**조건과 한계.** 범용 모델의 관점에서 세 가지가 중요하다. 첫째, 출발 base model이 결과를 좌우한다. Llama-3.1-8B-Instruct에서는 NSR을 포함한 모든 방법의 최종 pass@256이 base model보다 낮았다(Figure 4). 둘째, negative-only training은 수백 gradient step이 지나면 성능이 저하되며, 저자들은 이를 limitation으로 보고했다. W-REINFORCE에서는 이 저하가 나타나지 않았다(App. F). 셋째, 이 연구가 다루는 reward는 sparse한 binary verifiable reward뿐이다.

**반대 방향에서 얻은 두 번째 측정.** [[raft-reinforce-rej-minimalist]]는 Numina-Math에서 positive-only RAFT와 RAFT++를 GRPO 및 REINFORCE와 비교한다(Table 1). Qwen2.5-Math-7B-base에서 세 benchmark 평균은 RAFT 52.3, RAFT++ 56.1, GRPO 56.3, Reinforce-Rej 56.4, PPO 52.5, iterative DPO 48.8이다. LLaMA-3.2-3B-instruct에서는 REINFORCE 24.2가 positive-only RAFT++ 27.6보다 낮고, GRPO는 28.4에 도달한다. 이 논문의 entropy curve는 NSR 연구를 반대편에서 확인한다. RAFT++는 두 model 모두에서 GRPO와 비교해 policy entropy가 훨씬 더 빠르게 감소하며, entropy가 낮은 값에서 안정화된 뒤에는 개선이 느려지고 GRPO가 이를 추월한다(Figure 3). Ablation은 어떤 negative가 해로운지도 분리한다. 모든 response가 틀린 prompt를 제거하면 일반 REINFORCE보다 reward가 가장 크게 상승한다. all-correct prompt를 제거해도 변화는 작고, reward normalization은 GRPO 우위의 원인이 아니다(Figure 4). 사용할 수 있는 baseline이 없는 $\pm1$ reward의 all-incorrect group은 batch 안에 positive anchor 없이 순수하게 확률을 내리는 update다.

**두 부호를 함께 쓸 때 효과의 크기.** 아래에서 RFT(rejection-sampling fine-tuning)와 RAFT는 positive-only arm이다. response를 sample하고 정답만 남긴 뒤, 오답에 대한 항 없이 그 정답들로 fine-tuning한다.

| 설정 | negative 없음 | negative 사용 | 귀속(attribution) |
|---|---|---|---|
| NFT, Qwen2.5-32B, 6-benchmark 평균(base 29.6) | RFT 52.8 | NFT 59.2 | “positive data (RFT) contributes to 80% … while negative data only accounts for the remaining 20%” ([[nft-negative-aware-finetuning]] §5.3) |
| NFT, Qwen2.5-Math-7B(base 31.6) | RFT 48.3 | NFT 51.7 | 20.1점 중 3.4점, 16.9%(Table 1에서 계산) |
| NSR/W-REINFORCE, Qwen2.5-Math-7B MATH | PSR pass@256 91.2 | NSR 96.9 | 이 pair에서는 negative가 pass@k 차이 전체를 설명한다. pass@1 차이는 1.6점이다([[negative-sample-reinforcement]] Table 1). |
| BCPG-NSA, DeepSeek-R1-Distill-Qwen-14B, 4-benchmark 평균(start 66.09) | RFT 64.64 | BCPG 67.44, BCPG-NSA 68.45 | positive-only training은 시작 model보다 1.45점 낮아졌다([[bcpg-nsa]] Table 3). |
| RAFT++ 대 GRPO, Qwen2.5-Math-7B-base | RAFT++ 56.1 | GRPO 56.3 | 0.2점([[raft-reinforce-rej-minimalist]] Table 1) |

이 숫자에는 두 가지 주의점이 따른다. NFT 비교에서는 prompt weighting도 바뀐다. NFT는 $\omega(q)=1-\hat r_q$를 쓰지만 RFT는 constant weight를 쓴다(App. C). 그러므로 20%를 negative term 하나에만 귀속할 수는 없다. RAFT++ 행과 NSR 행은 pass@1에서 negative의 가치가 어느 정도인지에는 서로 다른 결론을 내리지만, entropy에 대해서는 일치한다. 둘 다 Qwen math model에서 수행한 단일 연구다.

**negative에도 사용할 수 있는 내용이 들어 있다.** [[bcpg-nsa]] §3은 같은 prompt에서 final answer가 틀린 response 19,000개와 맞는 response 19,000개를 각각 fine-tuning에 사용했다. AIME24는 incorrect set에서 41.67, correct set에서 52.75였고, starting model은 10.00이었다(Table 1). 틀린 long-CoT response 안에도 맞는 step, self-reflection, 대안적 접근이 들어 있다. 이는 “negative”의 의미 (2)를 지지하는 근거이지, 더 강한 push-down의 근거는 아니다. 그리고 §5의 step-level 처리로 이어지는 동기다.

## §5 온라인 group RL 안의 negative

**advantage가 0인 group.** group 안의 response가 모두 같은 결과이면, 그 group의 모든 member의 advantage는 0이다. [[dapo]] §3.2는 모든 batch가 mixed group을 갖도록 그런 prompt를 oversample한 뒤 filter한다. Ablation은 DAPO가 AIME 2024에서 얻은 20점 중 8점을 dynamic sampling에 귀속한다(42 → 50, Table 1). [[ngrpo]]는 all-incorrect group에 반대의 방법을 쓴다. group statistic에 maximum reward를 가진 virtual sample 하나를 더해, all-incorrect group이 균일한 negative advantage를 받도록 한다(Eqs. 5–6).

**계산 예.** $G=8$이고 reward가 binary라고 하자. correct response가 하나인 group을 잡으면, sample standard deviation으로 standardize했을 때 correct response의 advantage는 $+2.47$, incorrect response 각각의 advantage는 $-0.35$이다. 여기에 $r_{\max}=1$인 virtual sample을 더하면 각각 $+1.76$, $-0.50$으로 바뀌며, 이는 [[ngrpo]] Figure 3에 인쇄된 값이다. all-incorrect group에서는 같은 convention이 GRPO의 $0.00$ 대신 균일한 $-0.33$을 만든다(계산값이다. 인쇄된 Eq. 6은 square root 안에 $1/(G+1)$을 사용하지만, 그림의 값은 sample standard deviation과 일치한다). 따라서 calibration은 동시에 두 일을 한다. homogeneous-incorrect group에 0이 아닌 gradient를 부여하고, 모든 mixed group에서 negative 쪽을 더 무겁게 만든다.

calibrated advantage의 합은 negative number이므로, update에는 지속적인 아래 방향 압력이 생긴다. 저자들은 이것이 entropy를 올리고 training을 불안정하게 만들 수 있다고 말한다. 이들이 둔 control은 asymmetric clipping이다. $\epsilon_{\mathrm{neg}}=0.16$, $\epsilon_{\mathrm{pos}}=0.24$이다(§4.2). DAPO는 반대쪽에서 같은 비대칭성에 도달한다. unlikely token이 promote될 수 있게 $\epsilon_{\mathrm{high}}$를 0.28로 올리고, $\epsilon_{\mathrm{low}}$는 0.2로 유지한다. 그 이유는 이를 올리면 “이 token들의 확률이 0으로 억제되어 sampling space가 collapse”하기 때문이다(§3.1). 두 설정은 같은 양에 작용한다. $\epsilon_{\mathrm{low}}$는 한 negative step이 token을 얼마나 아래로 밀 수 있는지에 대한 per-update bound이며, sampling space를 보호하는 parameter다.

**결과.** Qwen2.5-Math-7B에서 NGRPO는 AIME2025의 pass@k AUC를 31.28로 보고했다. DAPO는 30.27, PSR-NSR은 28.85, GRPO는 28.33이다. AIME2025 pass@256은 60.00이고, GRPO와 DAPO는 둘 다 53.33이다(Table 1). Ablation은 구성 요소를 분리한다. asymmetric clipping만 쓰면 28.48, advantage calibration만 쓰면 29.56, 둘을 함께 쓰면 30.54, homogeneous-incorrect group도 유지하는 full method는 31.28이다(Table 2). 이 결과는 단일 연구, 단일 model의 결과다. DAPO와 NGRPO는 all-wrong group을 어떻게 다룰지에 대해 서로 다르다는 점에 유의해야 한다. [[raft-reinforce-rej-minimalist]] ablation은 plain REINFORCE에서는 all-wrong prompt가 해로운 것이라고 찾았다. 차이는 NGRPO의 virtual sample이 그런 group의 advantage를 작은 magnitude로 bound하는 반면, plain REINFORCE는 그 token 각각에 완전한 $-1$을 부여한다는 데 있다.

**penalty를 국소화하기.** 다음 두 방법은 wrong response 안에 있지만 그 자체는 wrong이 아닌 token의 penalty를 줄인다.

- [[lazy-likelihood-displacement-grpo]]는 각 negative token이 group의 correct response likelihood에 미치는 influence로 score를 매긴다(NTHR, Eq. 7). 그리고 선택된 token의 advantage를 0으로 설정한다($\beta=1$, $\eta=0$). 다섯 math benchmark 평균은 다음과 같다(Table 2): Qwen2.5-Math-1.5B는 41.14 → 41.94, Qwen2.5-0.5B-Instruct는 11.72 → 12.66, Qwen2.5-1.5B-Instruct는 26.96 → 28.48, Qwen2.5-3B는 33.88 → 36.30, DeepScaleR에서 Qwen2.5-Math-1.5B는 37.80 → 39.60이다. random-token control은 likelihood measure에서 작은 개선만 보였다(Figure 4).
- [[bcpg-nsa]]는 LLM judge와 PRM으로 negative response의 각 step에 label을 붙이고, 둘 다 동의할 때만 그 step을 correct로 유지한다. 그런 token의 negative weight에는 $\beta \in [-1,1]$을 곱한다(Eqs. 6–7). $\beta=0.5$일 때 four-benchmark average는 68.45이다. 같은 objective에서 $\beta=1$이면 67.44, starting model이면 66.09이다(Table 3). consensus labeler는 correct token을 가장 적게 mining했다(PRM-only의 38M, LLM-only의 65M에 비해 26M). 그럼에도 가장 좋은 결과를 냈다(Tables 4–5). 따라서 step label에서는 recall보다 precision이 더 중요했다.

두 방법 모두 response-level negative signal은 유지하고, 그 response 안에서 어느 token이 그 signal을 받을지만 바꾼다. 이는 §3의 pair-filtering 권고에 해당하는 RL counterpart다.

## §6 Off-policy negative

**off-policy negative가 불안정한 경우인 이유.** 고정된 data distribution $\mu$ 아래에서 naive off-policy REINFORCE는 $\sum_{T^+}\mu\log\pi-\sum_{T^-}\mu\log\pi$를 최대화한다. [[topr-tapered-off-policy-reinforce]] §2.1은 두 번째 항이 “$\pi$의 관점에서 위로 unbounded이며, $\mu$가 support하는 단일 trajectory 하나의 확률을 0으로 몰아가면 임의로 크게 만들 수 있다”고 말한다. on-policy에서는 model이 더는 만들지 않는 trajectory는 더는 sample되지 않으므로 이 문제가 생기지 않는다. GSM8K run에서 naive REINFORCE는 처음에는 개선됐지만 이후 degenerate generation으로 collapse했다. PPO는 $\epsilon=0.2$에서 대부분의 sample이 $[1-\epsilon,1+\epsilon]$ 밖으로 나가므로 거의 진전하지 못했다(Figures 1, 3).

**bounded하게 처리하는 세 가지 방법.**

1. **ratio를 비대칭적으로 truncate한다.** TOPR의 canonical form은 negative 쪽 importance ratio를 $[\pi/\mu]_0^1$로 설정하고, positive 쪽은 평범한 SFT update로 둔다(Eq. 8). negative에서 lower limit가 0인 것이 핵심이다. model이 wrong trajectory에서 멀어질수록 그 trajectory의 contribution은 감소한다. “어떤 $a^- > 0$도 naive REINFORCE와 마찬가지로 결국 model degeneracy를 일으켜야 한다”(§3). negative 60% data set과 100.0으로 느슨하게 한 gradient clipping에서, truncate하지 않은 importance sampling은 training 마지막에 bad reasoning이 31%가 되었지만 base model은 12%였다. TOPR은 여전히 base model보다 개선됐다(§4.3, Figure 7).
2. **baseline을 이동한다.** [[asymmetric-reinforce]](AsymRE)는 off-policy에서 $A=r-V$를 연구한다. $V<V^\mu$이면 limit policy가 넓은 support를 유지하고, $V\ge V^\mu$가 되면 그 support가 일반적으로 한 원소로 “갑자기 축소된다”는 것을 증명한다(Thm. 4.2). LLM run에서는 $δV=0$인 MATH의 Llama-3.1-8B-Instruct run 7개가 모두 collapse했고, 저자들은 $δV=-0.1$에서 더 큰 안정성을 보고했다(§5.2, Fig. 5 left). 평균 reward보다 낮은 baseline은 positive에는 더 큰 weight를 주고 negative에는 더 작은 weight를 준다.
3. **KL anchor를 추가하고, 언제 필요한지 안다.** [[pmpo-positive-negative-feedback]]는 $\alpha\cdot\mathbb{E}_{D_a}[\log\pi]-(1-\alpha)\cdot\mathbb{E}_{D_r}[\log\pi]-\beta\cdot\mathrm{KL}(\pi_{\mathrm{ref}},\pi_\theta)$ objective를 유도한다(Eq. 10). negative-only 경우에는 KL term이 필요하다고 보고한다. §5.2의 Control Suite task에서 $\alpha=0$이면 performance가 $\beta$에 매우 민감하고 $\beta>1.0$이 필요했다. $\alpha=1$이면 민감하지 않았고, $\alpha=0.5$이면 $\beta>0.5$가 필요했다(Figure 3). language-alignment run도 같은 양상을 반복한다. negative-only training에는 $\beta>(1-\alpha)$가 필요하다고 적었다(§5.4, Figure 5). offline RGB-stacking benchmark에서 evaluation episode 100개의 reward는 behaviour cloning 24, accept+BC 26, accept only 27, reject+BC 77, 세 항 모두 사용했을 때 93이었다(Table 1).

**계산 예(baseline을 data composition으로 보기).** TOPR의 effective positive proportion은 $\tilde p=p(1-c)/(1+c(1-2p))$다. 여기서 $p$는 실제 positive share이고 $c$는 reward baseline이다(Figure 5 caption). $p=0.1$, $c=0$이면 effective share는 0.10이다. $c=-0.8$이면 0.50이 된다. $p=0.5$, $c=0.6$이면 0.20으로 내려간다. GSM8K sweep은 positive 10% data set과 50% data set 모두에서 effective positive proportion이 10–20%일 때 performance가 peak를 이루고, 50%를 넘으면 현저히 나빠진다고 찾았다(§4.3). negative를 버리는 것은 $c=-1$인 특수한 경우다.

**unlearning은 극단적인 경우다.** 개선이 아니라 제거가 목적이면 negative term만 남는다. [[negative-preference-optimization]]은 forget set에 gradient ascent를 적용하면 catastrophic collapse가 생긴다고 보인다. utility가 0으로 떨어지고 model이 gibberish를 낸다(§2.1). 반면 preferred term을 삭제한 DPO인 NPO loss는 각 update에 $W=2\pi_\theta^\beta/(\pi_\theta^\beta+\pi_{\mathrm{ref}}^\beta)$를 곱한다. 이 손실은 선형이 아니라 logarithmic하게 발산한다(Theorem 2). 여기서 다른 task로 옮길 수 있는 부분은 unlearning task 자체가 아니라 functional form이다. unbounded push-down은 선형으로 발산하지만, sample이 unlikely해질수록 weight가 감소하는 push-down은 logarithmic하게 발산한다.

**on-policy sampling이 다른 이유.** [[on-policy-suboptimal-preference-data]] §5.3은 on-policy sampling과 negative gradient가 상호 보완적인 benefit을 낸다고 보고한다. bandit 및 synthetic LLM setting에서 on-policy DPO와 IPO는 offline contrastive training, on-policy RL, supervised variant보다 더 빠르게 수렴하고 더 나은 solution에 도달했다. on-policy sampling은 response space의 coverage를 제공하고, negative gradient는 sample 하나당 더 강한 signal을 제공한다.

## §7 Negative의 품질: near miss, 쉬운 negative, false negative

negative sample의 두 성질이 gradient가 무엇을 할지 결정한다. 그 sample을 incorrect라고 표시한 label이 맞는지, 그리고 현재 policy 아래에서 그 sample이 얼마나 확률이 높은지다.

**near-miss negative는 정보량이 크면서 위험하다.** GRPO가 correct answer의 likelihood를 올리지 못하는 질문은, wrong response가 거의 맞았거나 reasoning은 맞지만 final format이 잘못된 질문과 정확히 일치한다([[lazy-likelihood-displacement-grpo]] §3). 또한 group의 correct response에 대한 hidden-embedding influence가 가장 큰 token은 wrong response 안에 있는, 논리적으로 또는 step 수준에서는 올바른 term인 경우가 많다(§5.1, Figure 3). 바로 이 성질 때문에 step level에서 mining할 가치도 생긴다([[bcpg-nsa]]).

**쉬운 negative와 off-policy negative는 기여가 작고 해로울 수 있다.** 현재 model이 거의 생성하지 않을 negative는 low-probability region에 있다. 이는 squeezing 분석이 최악의 regime이라고 지목한 구간이다(§2). off-policy에서는 §6의 ratio-based bound가 설계상 update를 거의 0까지 줄이는 구간이기도 하다.

**verifier의 false negative는 올바른 행동의 억제가 된다.** 측정된 네 사례는 다음과 같다.

| False negative의 원인 | 측정값 | 출처 |
|---|---|---|
| Answer parser가 correct solution을 거부함 | retained correct math solution의 비율: 25% (regex + sympy) → 73% (model judge) | [[bespoke-stratos]] blog, “Data Curation” |
| Unit test가 명시되지 않은 interface를 encode함 | requirements와 interface field를 제거하면 GPT-5 (high)는 25.9% → 8.40%, Claude Opus 4.1은 22.7% → 8.20% | [[swe-bench-pro]] §6.2, Table 3 |
| Truncation을 failure로 계산함 | truncated sample을 mask하면 AIME 2024 avg@32가 30 → 36, soft length shaping은 추가로 3점 | [[dapo]] Table 1, §3.4 |
| Truncation을 failure로 계산함, 반대 방향의 사례 | NFT는 truncated answer를 negative로 취급하며 그 선택의 ablation을 보고하지 않음 | [[nft-negative-aware-finetuning]] App. C |

DAPO가 적용한 control은 penalize가 아니라 mask다. “sound reasoning process can be penalized solely due to its excessive length” (§3.4)라는 문제를 피한다. 여기서 일반화되는 규칙은 이 절의 규칙이다. label이 틀렸을 수 있으면 negative advantage를 주지 말고 sample을 loss에서 빼야 한다. 이를 검증하는 비교는 truncated sample을 loss에서 mask한 단일 ablation run이다.

**abstention은 구조적으로 false negative다.** [[why-language-models-hallucinate]] §4.1은 abstention이 0점이고 우연히 맞힌 guess가 1점인 binary grader 아래에서는 abstention이 optimal response가 될 수 없음을 보인다(Observation 1). Table 2는 조사한 popular benchmark 10개 중 9개가 이 방식으로 grading한다고 찾는다. verifiable-reward RL run은 이 구조를 물려받는다. “I don't know”와 confident fabrication은 동일한 negative advantage를 받으므로, negative gradient는 abstention과 error를 함께 아래로 민다. 따라서 pre-training 뒤에 calibration되어 있던 base model도 calibrated 상태로 남을 필요가 없다(§3.1, Figure 2).

**계산 예(confidence target이 바꾸는 것).** 제안된 instruction에서는 correct answer가 1점, abstention이 0점, error가 $-t/(1-t)$점을 받는다. correctness probability가 $p$인 answer의 expected score는 $p-(1-p)\cdot t/(1-t)$이며, 이는 정확히 $p>t$일 때 양수다(§4.2). $t=0.5$이면 penalty는 1이다. confidence가 0.6인 answer는 $0.6-0.4=+0.2$를 받아 answer가 이긴다. $t=0.9$이면 penalty는 9이고, 같은 answer는 $0.6-3.6=-3.0$을 받아 abstention이 이긴다. 이를 RL run의 reward에 옮기면 abstention에서 advantage의 부호가 바뀐다. negative gradient가 abstention을 억제하는 대신 가르칠 수 있는 유일한 방법이다. 이 장의 어떤 source도 그 reward를 train하고 general capability에 미치는 효과를 측정하지 않았다. 이는 Open question이다.

## §8 파이프라인 전반에서 실패를 다시 사용하는 방식의 지도

| 실패의 사용 방식 | 의미 | 단계 | 근거 기준점 |
|---|---|---|
| 버리기 | (1) | SFT, distillation, RFT | positive-only RFT는 시작 model보다 1.45점 낮았다([[bcpg-nsa]] Table 3). RAFT는 52.3, GRPO는 56.3이었다([[raft-reinforce-rej-minimalist]] Table 1). |
| 합리화하기(answer를 제공한 채 다시 추론하기) | (2) | SFT | STaR + rationalization은 CQA에서 72.5%, direct fine-tuning은 60.0%였다([[star]] Table 1). |
| 교정 target(failure를 input에 넣음) | (2) | SFT | [[ch-31a]] §3 |
| failure-conditioned(label은 inference에서 사용하지 않음) | (3) | SFT | [[ch-31a]] §4 |
| preference pair의 rejected 쪽 | (4) | preference optimization | [[likelihood-displacement]], [[gradient-entanglement-margin-alignment]], [[ch-39]] |
| 전체 response에 주는 negative advantage | (4) | RL | §4, §5 |
| step 또는 token 수준 negative advantage | (4) | RL | [[bcpg-nsa]] Eq. 7; [[lazy-likelihood-displacement-grpo]] §5 |
| failure로 verifier 또는 PRM을 학습하기 | policy에는 (2), verifier 내부에는 (4) | reward modeling | TOPR의 verifier accuracy는 32.6% → 70.9%, invalid rate는 34.2% → 0.90%였다([[topr-tapered-off-policy-reinforce]] Table 2). step label에는 MC rollout을 쓴다([[math-shepherd]]). |
| 암묵적 negative policy(명시적 negative term 없음) | 효과상 (4) | online SFT | [[nft-negative-aware-finetuning]] Eq. 8 |

이 지도는 이 장의 실무적 요약이다. 같은 failed rollout도 버릴 수 있고, 다시 쓸 수 있고, 조건으로 제공할 수 있고, 확률을 낮추도록 학습할 수도 있다. 그 가운데 마지막 선택만이 §2와 §3의 위험을 수반한다.

## Negative sample과 negative feedback

이 절은 RL 및 preference 단계에 course 표준을 적용한다. 의미 (1)–(4)는 [[ch-31a]] §1의 분류를 따른다.

1. **negative가 생기는 곳.** final answer의 rule-based verifier([[negative-sample-reinforcement]], [[nft-negative-aware-finetuning]], [[dapo]]), 실행 및 unit test([[swe-bench-pro]]), reward model과 judge([[pmpo-positive-negative-feedback]]), step judge와 PRM([[bcpg-nsa]]), human preference label([[likelihood-displacement]]는 tie-breaking에 PairRM을 사용한다)이다. 측정된 false-negative 원인은 §7에 나열했다. 이 장의 어떤 RL 논문도 자기 verifier의 false-negative rate를 보고하지 않는다.
2. **현재 관행이 negative를 쓰는 방식.** RFT, STaR-style loop, RAFT는 이를 버린다(의미 1). GRPO, DAPO, NGRPO, TOPR, AsymRE, DPO, NPO는 gradient로 쓴다(의미 4). NFT는 암묵적 negative policy를 통해 gradient로 쓴다. BCPG-NSA와 NTHR은 step 또는 token 수준 gradient로 쓰며, BCPG-NSA의 preliminary study는 content(의미 2)로도 사용한다.
3. **메커니즘.** $∂\log p_y/∂z_j = 1[j=y]-p_j$다. push-down은 sampled token의 logit을 $η|A|(1-p_y)$만큼 내리고, 다른 모든 logit을 $η|A|p_j$만큼 올린다. 따라서 제거된 mass는 이미 가장 확률이 높은 token으로 간다. unlikely token에서는 distribution을 sharper하게 만든다(§2; 계산 예에서 0.0317 token을 push-down하면 0.6364 token에 0.0583이 가고 entropy가 0.12 nats 내려갔다). response 수준에서는 같은 step이 penalized response와 token 또는 embedding을 공유하는 모든 response를 움직인다(§3).
4. **숫자로 측정된 근거.** 이점: §4 표, NGRPO의 pass@256 60.00 대 53.33, NTHR의 +0.8~+2.4점, BCPG-NSA의 평균 +1.0이다. 실패: refusal 74.4% → 33.4%([[likelihood-displacement]]), $δV=0$에서 7/7 collapse run([[asymmetric-reinforce]]), truncate하지 않은 ratio에서 bad reasoning 31%([[topr-tapered-off-policy-reinforce]]), $α=0$에서 KL term 없이 collapse([[pmpo-positive-negative-feedback]]), gradient ascent에서 gibberish([[negative-preference-optimization]])다.
5. **control.** negative를 on-policy로 유지하거나 importance ratio로 bound한다($a^-=0$, $b^-=1$). batch 또는 loss에 positive anchor를 유지한다(positive term의 $λ=0.1$, positive NLL term, reference에 대한 KL). 한 update의 push-down을 bound한다($ε_{low}$, $ε_{neg}$, NFT의 $ε$). 다른 token 또는 첫 wrong step으로 국소화한다. near-duplicate pair를 CHES로 filter한다. 불확실한 failure는 penalize하지 말고 mask한다.
6. **진단.** chosen/rejected log-probability를 따로 log한다([[gradient-entanglement-margin-alignment]] Figure 1). advantage 부호별로 gradient norm, clip fraction, entropy를 log한다(§2의 four-quadrant 표가 예상값을 제시한다). $k\ge64$에서 pass@1과 pass@k를 추적한다. group이 homogeneous인 prompt 비율을 추적한다([[dapo]] Figure 3b). abstention rate와 calibration curve를 추적한다. 작은 probe set에서는 single update 뒤 known-good response의 $Δ\log π$를 잰다([[lazy-likelihood-displacement-grpo]] Eq. 2).
7. **generality에 미치는 효과.** Coverage: negative-only training은 MATH에서 positive-only가 5.7점 잃은 곳에서 pass@256을 보존했다([[negative-sample-reinforcement]] Table 1). Calibration과 hallucination: binary grader는 abstention을 negative로 만들며, 여기의 어떤 source도 RLVR 뒤 calibration을 측정하지 않았다([[why-language-models-hallucinate]] §4.1; Open question). 과도한 억제: DPO refusal training에서 refusal rate가 41점 떨어졌다([[likelihood-displacement]] §6.2). Forgetting: 이 장의 RLVR 연구에서는 측정하지 않았으며, KL 기반 처리는 [[ch-43]]에 있다.

## 레시피

| Model(정확한 release) | Size | 단계 | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B(NSR / W-REINFORCE) | 7B | RL | data; prompt batch; prompt당 rollout; mini-batch; LR; temperature; max context | MATH 7,500문제; 1,024; 8; 256; 1e-6; 1.0; 4,096 | arXiv:2506.01347v2 §3.1, App. D.1 | verified 2026-09-15 | ablation 미보고 |
| Qwen2.5-Math-7B(W-REINFORCE) | 7B | RL | positive term의 weight $λ$ | 0.1 | arXiv:2506.01347v2 §5, Eq. 9 | verified 2026-09-15 | Table 4: $λ=0/0.05/0.1/0.2/1$에서 MATH pass@256은 96.9 / 97.1 / 96.7 / 95.9 / 92.0 |
| Qwen2.5-Math-7B, Qwen2.5-32B(NFT) | 7B, 32B | RL | negative-ratio clip $ε$; prompt weight $ω(q)$; prompt당 sample; rollout step; LR | 1.0; $1-\hat r_q$; 16 × 512문제; 320; linear warm-up 포함 1e-6 | arXiv:2505.18116v3 §5.1, App. C | verified 2026-09-15 | §5.4 Fig. 10: $ε→0$이면 성능 하락; Fig. 9: $ω(q)=1-\hat r_q$가 $ω(q)=1$보다 좋음 |
| Qwen2.5-32B(DAPO) | 32B | RL | $ε_{low}$; $ε_{high}$; dynamic sampling; soft overlong window; max generation; prompt batch × response; LR | 0.2; 0.28; on; 16,384 + 4,096 cache; 20,480 token; 512 × 16; constant 1e-6, rollout 20 step warm-up | arXiv:2503.14476v2 §3.1–3.4, §4.1 | verified 2026-09-15 | Table 1: 30 → 36(overlong filtering) → 38(clip-higher) → 41(soft punishment) → 42(token-level loss) → 50(dynamic sampling) |
| Qwen2.5-Math-7B(NGRPO) | 7B | RL | virtual sample; $ε_{neg}$; $ε_{pos}$; epoch; batch; LR; max response; compute | group당 하나의 $r_{max}$; 0.16; 0.24; MATH에서 20; 1,024; 1e-6; 3,072 token; 8 × H100 | arXiv:2509.18851v1 §4.1–4.2, §5.1.1 | verified 2026-09-15 | Table 2(pass@k AUC, AIME2025/AMC): 28.33/81.04 → 28.48/81.22(clipping) → 29.56/83.85(calibration) → 30.54/84.37 → 31.28/86.09 |
| Llama 3 8B Instruct(TOPR) | 8B | RL | truncation limit $(a^+,b^+,a^-,b^-)$; LR; optimizer; KL; grad clip; reward; baseline | 1, 1, 0, 1; constant 5e-7; Adafactor; 없음; 1.0; ±1; 없음 | arXiv:2503.14286v2 §3, §4.2 | verified 2026-09-15 | §4.3 Fig. 7: 60%-negative dataset에서 truncate하지 않은 ratio는 base model의 12% 대비 bad reasoning 31% |
| Llama 3 8B Instruct(TOPR) | 8B | RL | effective positive proportion $\tilde p$ | 10–20% | arXiv:2503.14286v2 §4.3, Figs. 5–6 | verified 2026-09-15 | GSM8K 및 MATH sweep; 50%를 넘으면 성능 하락 |
| Llama-3.1-8B-Instruct(AsymRE) | 8B | RL | baseline offset $δV$; prompt당 sample; behaviour-policy refresh; LR | 약 −0.1; 8; gradient step 250회마다; 6e-8 | arXiv:2506.20520v2 §5.2–5.3, App. B | verified 2026-09-14(card [[asymmetric-reinforce]]) | Fig. 5 left: $δV=0$에서 7/7 run collapse; −0.1에서 더 안정적 |
| Qwen2.5 0.5B–3B(GRPO + NTHR) | 0.5B–3B | RL | threshold scale $β$; retained-token advantage scale $η$; data | 1; 0(selected token에 penalty 없음); MATH level 3–5 | arXiv:2505.18830v1 §5.3 | verified 2026-09-15 | Table 2: 네 model에서 GRPO 대비 평균 +0.8~+2.4점 |
| DeepSeek-R1-Distill-Qwen-14B(BCPG-NSA) | 14B | RL | fixed rollout set의 offline training; mining coefficient $β$; behaviour constraint $τ$; LR; epoch; batch; sequence length | 2,069문제 / 66,208 sample; 0.5; 1e-3; 5e-7(min 2.5e-7); 8; 512; 32k | arXiv:2505.14403v4 §5.1, App. C, Table 3 | verified 2026-09-15 | §6.2 Fig. 2: $β$가 감소할수록 AIME24/25 평균은 오르다 내리며, $β=-0.5$를 포함한 넓은 범위에서 $β=1$을 이김 |
| DeepMind Control Suite task의 PMPO | 미보고 | RL | $α=0$일 때 KL weight $β$; $α=1$; $α=0.5$ | > 1.0 필요; $β$에 둔감; 0.5 초과 | arXiv:2410.04166v3 §5.2, Fig. 3 | verified 2026-09-15 | Fig. 3: $β=0.0,0.5,1.0,1.5,2.0$ sweep; negative-only learning은 $β>1.0$ 필요 |
| Gemma 2B(PMPO, language alignment) | 2B | RL | $α$; prompt; prompt당 generation; batch; $β$ 규칙 | 두 signal이 동등하게 신뢰될 때 시작점 0.5; 약 4,000 learner step에 500k prompt; 4; 128 prompt; $β>(1-α)$ | arXiv:2410.04166v3 §5.4, App. A | verified 2026-09-15 | §5.4 Fig. 5: negative-only training에도 충분히 높은 $β$가 필요함; language run에는 numeric sweep 미인쇄 |
| Gemma-2B-IT, Llama-3-8B-Instruct(CHES filtering) | 2B, 8B | preference | filtered subset | length-normalized CHES가 가장 낮은 pair 5% 유지 | arXiv:2410.08847v4 §6.3 | verified 2026-09-14(card [[likelihood-displacement]]) | Figure 3: refusal rate 회복; 15%까지 유사 |
| Llama2-7b-chat(NPO) | 7B | preference | loss(TOFU forget set의 unlearning); $β$ | retain term이 있는 NPO; 0.1 | arXiv:2404.05868v2 §3, App. D.1 | verified 2026-09-15 | §5: NPO-based method만 Forget05/Forget10에서 forget quality 0.05 초과 |

**작은 범용 run의 시작점.** verifier가 있는 7B math 또는 code model의 RLVR run에서는 두 부호를 모두 유지한다. 가장 빠른 pass@1보다 큰 $k$의 pass@k가 중요하면 positive term에 $λ=0.1$을 둔다(Qwen2.5-Math-7B, MATH, 1,024 prompt × 8 rollout, LR 1e-6에서 검증). $ε_{low}=0.2$, $ε_{high}=0.28$을 쓴다(512 × 16 rollout과 20,480-token generation cap의 Qwen2.5-32B에서 검증). truncated sample은 penalize하지 말고 mask한다(같은 run). homogeneous group은 버리거나, 대부분의 group이 all-incorrect인 hard prompt를 학습한다면 $ε_{neg}=0.16$과 함께 virtual maximum-reward sample 하나를 쓴다(Qwen2.5-Math-7B, MATH, 20 epoch, 8 × H100에서 검증). 이 문단의 모든 값은 위 `verified` 행에서 왔으며, binary verifier를 쓰는 7B–32B Qwen math model에서만 측정됐다. mixed-domain instruction model에서는 측정되지 않았다.

## 일반화 관점

**(a) breadth를 넓히는 것.**

- negative term을 유지하면서 positive term을 down-weight하면 세 math benchmark에서 base model의 pass@256을 보존했고, AIME 2025 pass@256을 base 46.7에서 56.7로 올렸다([[negative-sample-reinforcement]] Table 1).
- independent study 두 편에서 negative는 positive-only training보다 entropy를 높게 유지한다. NFT 대 RFT(7B 및 32B, [[nft-negative-aware-finetuning]] Figure 8)와 Qwen 및 LLaMA에서 GRPO 대 RAFT++([[raft-reinforce-rej-minimalist]] Figure 3)다. Replicated.
- failure를 gradient만이 아니라 content로 쓰면 push-down으로는 추가할 수 없는 capability를 넣는다. incorrect long-CoT response 19,000개로 SFT하면 AIME24가 10.00에서 41.67로 올랐다([[bcpg-nsa]] Table 1).
- bounded off-policy negative는 correct sample이 전혀 없거나 거의 없는 prompt 수를 줄인다. 이는 sharpening이 아니라 coverage gain이다. 저자들은 negative의 이득이 “reducing the number of questions for which no or few solutions are found”에서 온다고 썼다([[topr-tapered-off-policy-reinforce]] Figure 4).

**(b) narrowing 또는 forgetting을 일으키는 것.**

- 이미 unlikely한 sample의 push-down은 mass를 most-likely continuation에 집중시키고 entropy를 낮춘다(§2; [[learning-dynamics-llm-finetuning]] §3.3). 같은 source §4.2에서는 model의 greedy output이 어떤 labeled response보다 더 빨리 confident해진다.
- 두 쪽이 같은 종류의 response인 preference pair는 가르치려던 behaviour를 제거한다. refusal rate가 74.4%에서 33.4%로 내려갔다([[likelihood-displacement]] §6.2).
- off-policy의 unbounded negative는 support를 collapse한다. $V\ge V^\mu$일 때의 theoretical support shrinkage 및 $δV=0$에서 7/7 collapse run([[asymmetric-reinforce]] Thm. 4.2, §5.2), naive REINFORCE의 degenerate generation, truncate하지 않은 importance sampling의 bad reasoning 31%([[topr-tapered-off-policy-reinforce]] §4.3)이 근거다.
- negative-only training은 시작이 좋아도 수백 step 뒤 성능이 저하된다([[negative-sample-reinforcement]] App. F).
- binary verifier는 abstention을 negative로 만들므로, hallucination을 줄일 behaviour를 상대로 negative gradient를 학습한다([[why-language-models-hallucinate]] §4.1).

**(c) 이 단계를 측정하는 방법.**

- pass@1 대신이 아니라 함께, unbiased estimator로 넓은 $k$ 범위의 pass@k를 측정한다([[negative-sample-reinforcement]]의 문제당 256 sample; [[ngrpo]]의 $k\in\{1,…,256\}$ pass@k AUC).
- training batch의 구성은 filtering rule로 변하므로, training batch가 아니라 held-out set에서 entropy를 측정한다([[negative-sample-reinforcement]] Figure 5b).
- chosen/rejected log-probability를 따로 기록하고 Condition 1이 비교하는 norm도 함께 기록한다([[gradient-entanglement-margin-alignment]] §3.1.1).
- held-out correct response에서 single-update probe를 한다. 이것이 lazy displacement를 보이게 하는 유일한 방법이다([[lazy-likelihood-displacement-grpo]] Eq. 2).
- training domain 밖의 held-out capability suite와 abstention rate를 둔다. 이 장의 어떤 RL 연구도 둘을 보고하지 않았고, 이것이 여기의 주된 measurement gap이다. [[ch-46]]은 한 run에서 이를 닫는 lab이다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| “negative signal이 더 많으면 항상 더 좋다”고 취급 | pass@1은 오르지만 $k\ge64$의 pass@k가 base model보다 하락 | 모든 checkpoint에서 여러 $k$의 unbiased pass@k 평가 |
| 현재 policy가 거의 만들지 않는 response를 penalize(stale rollout, import된 negative) | entropy 하락, greedy-decoding log-probability가 labeled response보다 더 빨리 상승 | negative sample의 importance-ratio distribution을 log하고, lower bound 없이 1에서 truncate |
| chosen/rejected가 같은 종류의 response인 pair | chosen/rejected log-probability가 둘 다 하락; refusal 같은 target behaviour 저하 | length-normalized CHES를 계산하고 상위 decile을 검사; probe set에서 behaviour rate 추적 |
| 한 step만 틀린 전체 wrong response를 penalize | near-miss failure가 있는 prompt에서 correct answer likelihood가 천천히 오르거나 하락 | prompt별 single-update $Δ\log π$ probe; 가장 낮은 $Δ$ prompt에서 near-miss negative 검사 |
| truncation, parser miss, missing test를 failure로 취급 | length cap에 도달하면 reward curve와 entropy가 불안정; strict output format task에서 accuracy 하락 | truncated sample을 mask한 run 하나를 ablate; “failure” 50개를 sample해 사람이 label |
| zero-advantage group이 해롭지 않다고 판단 | logged batch size는 그대로인데 effective batch size가 training 동안 감소 | batch별 homogeneous group 비율 log |
| positive anchor 하나면 collapse를 막는다고 예상 | loss는 감소하지만 chosen/rejected log-probability가 둘 다 하락 | probe batch에서 $‖∇\log π_w‖²$, $‖∇\log π_l‖²$, inner product를 log해 Condition 1 확인 |
| negative가 gain을 일으킨다고 가정 | positive-only baseline을 수행하지 않음 | matched step에서 positive-only arm(RFT, RAFT, PSR)을 실행하고 [[nft-negative-aware-finetuning]] §5.3처럼 split 보고 |

## 이해 점검

1. §2의 계산 예에서 0.0317 token의 push-down은 entropy를 0.12 nats 낮췄고 0.6364 token의 push-down은 0.075 nats 올렸다. $Δz_j=ηA(1[j=y]-p_j)$에서 두 부호를 유도하고, 실험을 인용하지 않고 이것이 import된 negative보다 on-policy negative를 더 안전하게 만드는 이유를 설명하라.
2. NSR-only training은 pass@256을 base model 수준으로 보존했지만 PSR-only training은 MATH에서 5.7점을 잃었다. token-level gradient 식에서 그 차이까지의 causal chain을 제시하고, Llama-3.1-8B-Instruct 결과(pass@256에서 모든 method가 base 아래)가 그 chain의 어느 단계를 깨는지 말하라.
3. DAPO와 NGRPO는 all-incorrect group을 반대 방향에서 다루지만 둘 다 gain을 보고한다. 그런 group의 token에 각 방법이 주는 advantage magnitude를 사용해 둘 다 참일 수 있는 이유를 설명하라.
4. [[raft-reinforce-rej-minimalist]]는 all-incorrect prompt를 버리는 것이 plain REINFORCE에선 어떤 normalization 변화보다 더 도움이 된다고 찾았다. ±1 reward에서 그 prompt가 harmful case인 이유는 어떤 성질 때문이며, 같은 논증이 all-correct prompt에 적용되지 않는 이유는 무엇인가?
5. Gradient entanglement는 margin은 항상 좋아지지만 개별 log-probability는 둘 다 하락할 수 있다고 말한다. $‖∇\log π_w‖²=1$, $‖∇\log π_l‖²=4$, inner product 2일 때 어느 case인지 보이고, pair의 어떤 token-level 성질이 그 inner product를 만들었는지 말하라.
6. TOPR은 negative에 $a^-=0$, positive에 $a^+=1$을 둔다. policy가 data-generating policy에서 멀어질 때 일어나는 일로 각 선택을 설명하고, $a^-=0$을 NPO weight $W$와 NFT의 $ε$에 연결하라.
7. 한 팀이 negative를 더하자 agent benchmark가 4점 올랐다고 보고한다. negative가 gain을 일으켰다고 받아들이기 전에 어떤 세 측정을 요구하겠는가? 각각 무엇을 배제하는가?
8. customer-support agent verifier가 gold answer가 policy statement일 때마다 “사람에게 확인해야 합니다”를 incorrect로 표시한다. 그런 response의 negative advantage가 model distribution에 하는 일을 추적하고, 한 run 안에서 문제를 잡아낼 reward 변경과 diagnostic을 설계하라.

## 연결

- 이전: [[ch-43]] — RL의 Entropy, Output Diversity, KL Control. 이 장이 advantage 부호별로 읽는 covariance law와 KL estimator를 제공한다.
- 다음: [[ch-44]] — Process Supervision과 Verifiable Reward. §5의 localized negative가 의존하는 step-level label을 제공한다.
- 의존: [[ch-39]] — Offline Preference Optimization: DPO와 그 variant(rejected term과 variant); [[ch-31a]] — Supervised Training의 Negative Sample: Correction, Failure Conditioning, Critique, Unlikelihood(네 의미와 supervised bounded loss).
- 여기에서 적용: [[ch-40]] — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, GSPO; [[ch-41]] — Reward Modeling: Bradley–Terry, Over-Optimization, Reward-Model Generalization; [[ch-42]] — Reward Hacking과 Judge Design; [[ch-29d]] — User Simulator, Trajectory Verification, Failed Trajectory; [[ch-35]] — Distillation in Practice A: Lab이 Teacher Data를 넣는 위치; [[ch-44a]] — RL의 Length: Overlong Response, Length Control, Long-Context RL; [[ch-54]] — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, Agentic Environment.
- 측정: [[ch-46]] — Lab: Negative-Signal Ablation과 Held-Out Capability Retention을 갖춘 DPO 또는 RLVR Experiment; [[ch-38a]] — SFT 대 RL Generalization: On-Policy Data, Base Model KL, Output Diversity.

## 출처

- [[learning-dynamics-llm-finetuning]] — squeezing effect, 다섯 claim과 proof, off-policy DPO dynamics, win rate를 포함한 “extend” mitigation.
- [[likelihood-displacement]] — displacement 정의, refusal-rate collapse, CHES score와 filtering.
- [[gradient-entanglement-margin-alignment]] — chosen/rejected log-probability의 first-order coupling, Condition 1과 세 case, token-level gradient similarity.
- [[negative-sample-reinforcement]] — PSR/NSR 분해, token-level gradient, pass@k와 entropy 결과, W-REINFORCE와 $λ$ sweep, 명시된 limitation.
- [[raft-reinforce-rej-minimalist]] — GRPO 대비 positive-only RAFT/RAFT++, positive-only training의 entropy decline, all-incorrect prompt를 분리한 ablation.
- [[nft-negative-aware-finetuning]] — implicit negative policy, $ε$ clip, entropy curve, 32B에서 80/20 attribution.
- [[lazy-likelihood-displacement-grpo]] — GRPO의 lazy likelihood displacement, GWHES, NTHR token selection과 결과.
- [[dapo]] — clip-higher와 $ε_{low}$를 유지하는 이유, dynamic sampling, overlong filtering과 soft punishment, progressive ablation.
- [[ngrpo]] — virtual maximum-reward sample, asymmetric clipping, pass@k AUC 결과와 component ablation.
- [[bcpg-nsa]] — content로서 wrong long-CoT response의 가치, consensus step labeling, mining coefficient $β$와 ablation.
- [[topr-tapered-off-policy-reinforce]] — off-policy unbounded negative objective, asymmetric truncation, data composition으로서 baseline, verifier training.
- [[asymmetric-reinforce]] — positive/negative balance로서 baseline offset, support-shrinkage theorem, $δV=0$ collapse.
- [[negative-preference-optimization]] — gradient-ascent collapse, NPO weight, linear 대 logarithmic divergence.
- [[pmpo-positive-negative-feedback]] — 분리된 positive/negative term, negative-only learning에 필요한 KL term, offline-RL loss-mixture 표.
- [[on-policy-suboptimal-preference-data]] — mode-seeking update로서 negative gradient, recovered mass의 행방, on-policy sampling과의 상보성.
- [[why-language-models-hallucinate]] — binary grader가 abstention을 penalize함, explicit confidence target과 behavioral calibration.
- [[entropy-mechanism-llm-rl]] — entropy change의 covariance law와 centered-covariance token selection; four-quadrant 표에서 사용.
- [[swe-bench-pro]], [[bespoke-stratos]] — unit test와 answer parser에서 측정한 verifier false negative.
- [[star]], [[math-shepherd]], [[grpo]], [[dr-grpo]], [[rlvr-beyond-base-model]] — failure-reuse map과 §1에서 참조한 rationalization, rollout의 step label, group-baseline advantage, pass@k critique.
