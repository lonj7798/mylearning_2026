# 챕터 39 — Offline Preference Optimization: DPO와 그 변형

> **핵심 통찰.** 어떤 reward function이든 KL-regularized RLHF objective의 maximizer는 `π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x)`다. 이 identity를 뒤집으면 reward는 `β log π*/π_ref`와 prompt-only term으로 표현된다. 그 term은 Bradley-Terry difference 안에서 상쇄된다. 따라서 preference learning은 reward model도 policy sampling도 없이 pair의 binary classification loss가 된다([[dpo]] §4, Eq. 4–7). resulting loss는 pair의 margin `h = log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)`만으로 결정된다. 두 likelihood를 각각 제약하지 않고 차이만 제약한다. 이 장의 모든 variant는 objective가 추가로 무엇을 제약하는지 바꾸며, family에서 측정된 failure(chosen likelihood 하락, likelihood displacement, 60% 미만 ranking accuracy, 한 epoch 안 over-optimization)는 이 속성의 결과다.
>
> **지침.** fixed preference set이 있고 목표가 chat quality·instruction following·truthfulness라면, data를 만들거나 score한 SFT checkpoint를 reference model로 하는 DPO, 한 epoch, held-out win rate의 early stopping을 쓴다. [[unpacking-dpo-ppo]] 13B 연구에서 preference training은 factuality를 약 1점 안에 두면서 instruction following·truthfulness를 8점 이상 움직였고(§3.1, Table 1), [[daa-overoptimization-scaling]] TL;DR에서는 wide KL budget 아래 best checkpoint가 대략 한 epoch의 25% 뒤에 나타났기 때문이다(§3.1, Fig. 2). chosen response가 sampling으로 계속 도달 가능해야 하는 정답, 즉 math·code·tool call이면 chosen response에 positive term을 더한다([[rpo]]의 `α · NLL`, ORPO SFT term, [[dpo-positive]] DPOP penalty, [[noise-contrastive-alignment]] NCA form). 이 setting에서 plain DPO는 같은 data에 anchor를 둔 경우보다 sampled accuracy를 낮췄기 때문이다(GSM8K, 한 iteration 뒤 61.8% 대 73.1%, [[rpo]] Table 1). feedback이 pair가 아니라 single thumb면 KTO를 쓴다([[kto]]). current policy에서 pair를 다시 만들 수 있다면 static set보다 그것을 선호하고([[on-policy-suboptimal-preference-data]] §5.3; [[on-off-policy-rlhf]] §2.1), 이 장 offline objective는 RL 대체물이 아니라 그 sample에 적용할 loss로 취급한다.

---

## 이 장이 범용 모델에 중요한 이유

pipeline 위치는 pre-training → mid-training → SFT → **preference optimization** → RL → evaluation이다. 이는 training signal이 target sequence가 아니라 comparative한 첫 stage이며, loss term이 model이 만들었거나 만들 수 있는 text probability를 명시적으로 *낮추는* 첫 stage다. capability breadth에 두 결과가 중요하다.

첫째, 측정된 gain은 집중돼 있다. [[unpacking-dpo-ppo]] controlled 13B study에서 preference set 14개에 대한 DPO는 SFT model보다 instruction following을 최대 +8.6점, truthfulness를 최대 +12.7점 옮겼지만, 모든 dataset에서 MMLU는 1점 band 안이었다(SFT 55.4, training 뒤 54.7–55.7, §3.1 Table 1). 이는 fact나 reasoning을 가르치는 stage가 아니다. 기존 ability가 표현되는 방식을 reshape하며, 너무 오래 실행하면 ability를 제거할 수 있다.

둘째, negative term은 공짜가 아니다. rejected response는 explicit push-down으로 gradient에 들어간다(course negative-feedback taxonomy category 4, negative as gradient). model이 이미 unlikely하다고 보는 response를 아래로 밀면 되찾은 probability mass 대부분은 이미 가장 likely하던 token으로 간다([[learning-dynamics-llm-finetuning]] §3.3). 이것이 run이 training margin을 올리고 loss를 낮추면서도 preferred response probability([[likelihood-displacement]] §2.2) 또는 safety prompt refusal rate(Llama-3-8B-Instruct 74.4% → 33.4%, §6.2)를 낮출 수 있는 방식이다.

---

## §1 Offline setting이란 무엇인가

**정의.** Offline preference optimization은 fixed triple `{(x, y_w, y_l)}` — prompt, preferred response, dispreferred response — set에서 policy `π_θ`를 학습한다. [[kto]]의 경우 unary example `{(x, y, b)}`를 쓰며 `b ∈ {desirable, undesirable}`다. training 중 `π_θ`에서 response를 sample하지 않는다.

**해결하는 문제.** [[rlhf-instructgpt]] RLHF pipeline에는 reward model과 online sampling이 필요하다. update마다 separate network가 score한 fresh generation이 필요하다. Offline preference optimization은 둘 다 없애는 대신 다른 distribution에서 뽑은 response로 학습한다.

**pair의 출처.** human comparison([[hh-rlhf]]), model-scored multi-model completion([[ultrafeedback]]), 또는 starting checkpoint에서 response를 regenerate한 뒤 reward model로 rank한 것이다. [[simpo]]의 “Instruct” setup은 SFT model에서 temperature 0.8로 UltraFeedback prompt마다 response 다섯 개를 sample하고 PairRM([[pairrm]]) 아래 best와 worst를 취한다. 저자들은 이를 on-policy setting에 더 가깝다고 설명한다(§3). **해석.** §9에 모은 비교에서는 pair 출처가 아래 loss 중 무엇을 적용하는가보다 outcome을 더 바꾼다.

## §2 DPO

### 2.1 Objective와 closed-form optimum

KL-regularized RLHF objective([[dpo]] §3, Eq. 3)는 다음이다.

```
max_π  E_{x ~ D, y ~ π(·|x)} [ r(x, y) ]  −  β · D_KL( π(·|x) ‖ π_ref(·|x) )            (1)
```

`x`는 prompt, `y`는 response, `r`은 reward, `π_ref`는 reference policy(실제로는 SFT checkpoint), `β > 0`은 KL constraint 강도다. fixed `x`에서 이것은 response distribution에 대한 optimization이다. maximizer는 다음과 같다([[dpo]] Eq. 4, App. A.1).

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x, y) / β ),   Z(x) = Σ_{y'} π_ref(y'|x) exp( r(x, y')/β )   (2)
```

`Z(x)`는 partition function이다. 모든 response에 대한 합이므로 실제로 계산할 수 없다.

### 2.2 Reward로 역산하고 `Z(x)`를 상쇄하기

(2)에 log를 취해 `r`에 대해 풀면([[dpo]] Eq. 5) 다음이다.

```
r(x, y) = β · log[ π*(y|x) / π_ref(y|x) ]  +  β · log Z(x)                              (3)
```

Bradley-Terry model `P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l))` 아래에서는(ch-41 reward model이 fit되는 likelihood와 같다), 두 response가 prompt를 공유하므로 `β log Z(x)`가 양쪽 sign으로 나타나 상쇄된다. 이 difference를 Bradley-Terry negative log-likelihood에 넣고 `π*`를 trainable `π_θ`로 다루면 DPO loss가 된다([[dpo]] Eq. 7).

```
L_DPO(π_θ; π_ref) = − E_{(x,y_w,y_l) ~ D} [ log σ( β·log[π_θ(y_w|x)/π_ref(y_w|x)]
                                                 − β·log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]   (4)
```

`σ`는 logistic function이다. `π_θ(y|x)`는 response token probability의 곱이므로 `log π_θ(y|x)`는 response 전체 token log-probability의 **합**이다(reference implementation이 실제로 계산하는 값, §10 참조). **implicit reward**를 다음과 같이 정의한다([[dpo]] §4).

```
r̂_θ(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ]                                            (5)
```

**margin**은 `h = log[π_θ(y_w|x)/π_ref(y_w|x)] − log[π_θ(y_l|x)/π_ref(y_l|x)]`다. 따라서 pair당 `L = −log σ(β h)`다.

### 2.3 Gradient

[[dpo]] §4(App. A.4)는 다음을 준다.

```
∇_θ L_DPO = − β · E[ σ( r̂_θ(x,y_l) − r̂_θ(x,y_w) ) · ( ∇_θ log π_θ(y_w|x) − ∇_θ log π_θ(y_l|x) ) ]   (6)
```

scalar weight `σ(r̂_l − r̂_w)`는 (0, 1)에 있고 implicit reward가 현재 dispreferred response를 preferred response보다 높게 rank할수록 커진다. paper는 vector term 둘을 “`y_w` likelihood 증가”, “`y_l` likelihood 감소”라고 부른다.

**계산 예시.** `β = 0.1`로 둔다. pair 하나에서 policy/reference summed log-probability가 `log π_θ(y_w) = −40`, `log π_ref(y_w) = −42`, `log π_θ(y_l) = −55`, `log π_ref(y_l) = −50`이라고 하자. 그러면 `r̂_w = 0.1 × 2 = 0.2`, `r̂_l = 0.1 × (−5) = −0.5`, `h = 7`, `βh = 0.7`, loss `−log σ(0.7) = 0.403`, gradient weight `σ(−0.7) = 0.332`다. initialization(`π_θ = π_ref`)에서는 margin 0, loss `log 2 = 0.693`, weight 0.5다. model이 현재 wrong way로 rank한 pair는 weight가 0.5보다 크고 1에 접근한다.

**중요한 invariance.** reference보다 세 nat *낮은* `log π_θ(y_w) = −45`, `log π_θ(y_l) = −60`인 두 번째 pair를 잡는다. 다시 `h = 7`, loss 0.403, gradient weight도 동일하다. objective는 “chosen response가 reference 대비 7 nat 더 likely해졌다”와 “chosen response는 3 nat 덜 likely해지고 rejected response는 10 nat 덜 likely해졌다”를 구별하지 못한다. §8과 negative-feedback section은 optimizer가 이 freedom으로 무엇을 하는지 보인다.

[figures/dpo-landscape.html](figures/dpo-landscape.html)은 adjustable `β`, `τ`, `γ` 아래 DPO·IPO·SimPO의 pair별 loss·gradient weight를 margin에 대해 그리고, [[preference-ranking-accuracy]] Theorem 4.1에서 ranking이 flip되기 전 pair가 도달해야 하는 loss value를 표시한다(§8.2).

### 2.4 `β`가 하는 일

`β`는 sigmoid 안 margin을 곱한다. 더 큰 `β`는 같은 log-ratio difference로 더 큰 preference probability를 만들어 loss를 낮추는 데 `π_θ` 움직임이 덜 필요하다. 더 작은 `β`는 더 많은 움직임을 허용한다. [[dpo]] IMDb sweep은 `β ∈ {0.05, 0.1, 1, 5}`였고(§6.1, Fig. 2 left), 보고한 default는 0.1이며 “DPO의 β를 의미 있게 tune하지 않았다”고 한다(§6.2). `β`는 training time에 계산하는 KL penalty가 아니다. run이 실제 달성한 KL은 outcome이다. DPO·IPO·SLiC 전체에서 [[daa-overoptimization-scaling]]은 fixed budget이 아니라 “β parameter와 training 끝 corresponding KL 사이에 clear dependency”를 보고한다(§3.1).

---

## §3 IPO: logistic link 대신 bounded target

**측정된 문제.** [[ipo]]는 pair가 data에서 deterministic(`p*(y_w ≻ y_l) = 1`)이면 Bradley-Terry fit이 reward difference의 발산을 요구하고, optimum policy는 “KL-regularisation에 어떤 constant τ를 쓰든” `π*(y_l) = 0`으로 둔다고 관찰한다(§4. bounded `Ψ`를 동기화한 DPO·RLHF 분석). finite data에서는 대부분 pair의 empirical preference probability가 0 또는 1이므로 edge case가 아니라 normal case다.

**메커니즘.** `ΨPO`는 non-decreasing `Ψ`에 대해 objective를 `max_π E[Ψ(p*(y_w ≻ y_l|x))] − τ D_KL(π ‖ π_ref)`로 일반화한다. `Ψ(p) = log(p/(1−p))`는 DPO를 복구하고 `Ψ(p) = p`는 IPO를 준다. IPO sampled loss는 다음이다([[ipo]] Eq. 17).

```
L_IPO = E_{(x,y_w,y_l) ~ D} [ ( h_π(y_w, y_l, x) − 1/(2τ) )^2 ]                          (7)
h_π(y, y', x) = log[ π_θ(y|x) π_ref(y'|x) / ( π_θ(y'|x) π_ref(y|x) ) ]
```

`τ > 0`은 regularization strength이고 `h_π`는 §2.2와 같은 margin이다. optimum은 finite value `h* = 1/(2τ)`다. target을 overshoot한 것도 undershoot한 것처럼 벌을 받는다.

**계산 예시.** `τ = 0.1`이면 target은 `1/(2τ) = 5`다. `h = 7` pair의 loss는 `(7−5)² = 4`, gradient는 `2(h − 5) = +4`이며 margin을 *줄인다*. DPO에서 같은 pair는 loss 0.403이고 margin을 계속 키우는 gradient를 가진다.

**근거와 한계.** IPO paper experiment는 uniform reference policy로 Adam 18,000 step을 학습한 3-action bandit 문제이며 language model이 아니다(§5.3–5.4). language-model evidence는 후속 비교에서 온다. [[simpo]] Table 4에서 IPO는 AlpacaEval 2 length-controlled가 Mistral-Base 11.8, Llama-3-Instruct 35.6으로 DPO 15.1, 40.3보다 낮다. 반면 [[daa-overoptimization-scaling]] §3.1에서 IPO는 1B–6.9B TL;DR에서 같은 constraint 아래 더 낮은 KL과 DPO·SLiC보다 적은 over-optimization을 보인다. 각각 **결과(단일 연구)**이며 두 결과는 tuned setting win rate와 KL budget 전반 degradation이라는 다른 것을 측정하므로 IPO가 더 좋은지에 관해 상충한다.

---

## §4 KTO: unary label과 batch reference point

**문제.** production feedback은 보통 response당 signal 하나(accept, thumbs-down, escalation)이지 pair가 아니다. 여기서 pair를 만들면 data를 버린다.

**메커니즘**([[kto]] §4.1, Eq. 8). `r_θ(x, y) = log[π_θ(y|x)/π_ref(y|x)]`로 둔다(안에 `β`는 없음).

```
z_0 = KL( π_θ(y'|x) ‖ π_ref(y'|x) )                                                      (8)
v(x, y) = λ_D · σ( β ( r_θ(x,y) − z_0 ) )   if y is desirable
        = λ_U · σ( β ( z_0 − r_θ(x,y) ) )   if y is undesirable
L_KTO = E_{(x,y) ~ D} [ λ_y − v(x, y) ]
```

`λ_D`, `λ_U`는 desirable/undesirable side weight, `β`는 argument scale, `z_0`는 reference point다. batch에서 각 prompt를 다른 example response와 pair하여 estimate한다. `j = (i mod m) + 1`일 때 `ẑ_0 = max(0, (1/m) Σ_i log[π_θ(y_j|x_i)/π_ref(y_j|x_i)])`이며 이 항을 통한 gradient는 흐르지 않는다(§4.1, “KL Estimate”). mismatched pairing은 의도적이다. matched response는 흔히 의도적으로 good 또는 bad output이라 representative하지 않은 값을 주기 때문이다.

**계산 예시.** `β = 0.1`, `z_0 = 0.5`, `λ_D = λ_U = 1`로 둔다. `r_θ = 2`인 desirable response는 `v = σ(0.1 × 1.5) = 0.537`, loss `1 − 0.537 = 0.463`이다. `r_θ = −3`인 undesirable response는 `v = σ(0.1 × 3.5) = 0.587`, loss 0.413이다. desirable side `r_θ`와 `z_0`가 같은 양만큼 오르면 loss는 변하지 않는다. 이것이 paper가 명시한 의도다. KL term이 평평할 때만 reward가 증가할 수 있다(§4.1).

**Class imbalance.** 권고는 desirable/undesirable example count `n_D`, `n_U`에 대해 `λ_D n_D / (λ_U n_U) ∈ [1, 3/2]`다(§4.2, Eq. 9). paper worked case는 1:10 ratio, `λ_U = 1`이며 `λ_D = 13.33`으로 둔다. Llama-7B에서 desirable data 90%를 버리고 이렇게 rebalance해도 evaluation에서 DPO보다 좋았다(§4.3, Fig. 5).

**근거.** UltraFeedback으로 Zephyr-β-SFT를 한 epoch align하면 GSM8K는 39.0(SFT) → 40.0(DPO) → 53.5(KTO), BBH는 46.3 → 44.1 → 52.6, MMLU는 57.2 → 58.2 → 58.6이다(Table 2). prompt당 response 하나(`one-y-per-x`)로 학습하면 data 절반으로 GSM8K 50.0이다. 권고 learning rate는 AdamW `5e-6`이며 “DPO optimum learning rate의 2x~10x”다. paper는 model이 다른 어떤 hyperparameter보다 learning rate에 민감하다고 한다(§4.2). **결과(단일 연구)**다.

---

## §5 SimPO와 length-normalized DPO: reward 단위를 바꾸기

**측정된 문제.** DPO implicit reward는 token 합이므로 token별 log-probability를 균일하게 올리면 response length에 비례해 바뀐다. [[simpo]]는 DPO training 뒤 average log-likelihood에서 `p_θ(y_w|x) > p_θ(y_l|x)`를 만족하는 training triple이 약 50%뿐이라고 보고한다. 여기서 `p_θ`는 ranking에 쓰는 length-normalized likelihood다(§2.2, Fig. 4b). training 중 최적화한 quantity와 generation을 결정하는 quantity가 data 절반에서 불일치한다.

**메커니즘.** log-ratio를 average token log-probability로 바꾸고 reference를 버린다([[simpo]] Eq. 4, Eq. 6).

```
r_SimPO(x, y) = (β / |y|) · Σ_{t=1..|y|} log π_θ(y_t | x, y_<t)                          (9)
L_SimPO = − E[ log σ( (β/|y_w|) log π_θ(y_w|x) − (β/|y_l|) log π_θ(y_l|x) − γ ) ]        (10)
```

`|y|`는 token count이고 `γ ≥ 0`은 loss가 saturate하기 전 difference가 넘어야 할 target margin이다. reward가 이제 token당이므로 `β`는 DPO보다 큰 값을 쓴다. paper setting은 `β = 2.0, γ = 1.6`(Mistral-Base), `2.5 / 0.3`(Mistral-Instruct), `2.0 / 1.0`(Llama-3-Base), `2.5 / 1.4`(Llama-3-Instruct)이며 batch 128, 한 epoch다(App. B, Table 8). 같은 paper의 일반 지침은 `β ∈ [2.0, 2.5]`, `γ ∈ [0.5, 1.5]`(§3)다. PairRM 대신 ArmoRM reward model이 pair를 label한 뒤 Llama-3-Instruct v0.2 setting은 `β = 10`, `γ = 3`으로 이를 벗어난다(App. H).

**계산 예시.** `y_w`는 100 token, average log-probability −1.0(sum −100), `y_l`은 50 token, average −1.2(sum −60)로 둔다. summed log-probability는 `y_l`을 위에 rank하지만 token당 값은 `y_w`를 위에 rank한다. `β = 2.5, γ = 1.4`이면 reward는 −2.5와 −3.0, argument는 `0.5 − 1.4 = −0.9`, loss는 `−log σ(−0.9) = 1.241`이므로 pair는 여전히 큰 gradient를 낸다. DPO에서는 같은 pair를 summed quantity로 score하므로 length difference가 지배한다.

**근거.** AlpacaEval 2 length-controlled win rate, Llama-3-Instruct v0.1 setup은 SFT 26.0, DPO 40.3, SimPO 44.7이다. Mistral-Instruct는 17.1 / 26.8 / 32.1이다(Table 4). v0.2 label에서는 순서가 DPO 48.2, SimPO v0.1 44.7, SimPO v0.2 53.7로 바뀐다(Table 12). 따라서 DPO 대비 margin은 loss뿐 아니라 annotator에 달렸다. ablated component 중 length normalization을 빼는 비용이 가장 크다(Table 5 “w/o LN”: Mistral-Base 21.5 대비 11.9).

**같은 paper가 말한 비용.** Open LLM Leaderboard task에서 GSM8K는 거의 모든 preference method에서 떨어진다. Llama-3-Base SFT 46.32 → DPO 38.67 → SimPO 31.54(Table 9)다. Llama-3-8B-Instruct에서 learning rate 세 개로 이어간 Table 16의 AlpacaEval 2 LC / ZeroEval GSM / ZeroEval MMLU는 시작 model 26.0 / 78.5 / 61.7, LR `4e-7`에서 38.8 / 77.9 / 62.6, `5e-7`에서 44.6 / 77.0 / 62.3, released v0.2 checkpoint인 `1e-6`에서 53.7 / 57.4 / 54.9다. 저자들은 마지막 행을 GSM8K·MMLU의 catastrophic forgetting이라고 부른다. hyperparameter 하나가 chat metric과 capability metric을 교환한다.

**Length-normalized DPO.** Tülu 3는 reference model을 유지하고 각 log-ratio를 자기 length로 normalize한다([[tulu-3]] §5.2, Eq. 6). `L = −log σ( (β/|y_c|)·log[π_θ(y_c)/π_ref(y_c)] − (β/|y_r|)·log[π_θ(y_r)/π_ref(y_r)] )`다. early SFT checkpoint와 UltraFeedback에서 한 algorithm ablation(§5.4.1, Table 18)의 average score는 SFT base 55.7, SimPO 51.8·52.9, DPO(β = 0.1, 3 epoch) 55.2, PPO 54.5·55.5, length-normalized DPO 46.8–57.3(β·epoch에 따라 다름)다. 최고는 `β = 5`, 한 epoch, LR `5e-7`에서 57.3이다. starting checkpoint를 이긴 것은 length-normalized variant뿐이므로 released Tülu 3 model이 이를 쓴다([[open-instruct-allenai-recipes-recipe]]).

---

## §6 ORPO: 한 stage, odds ratio, reference model 없음

**측정된 문제.** chosen response의 cross-entropy에는 rejected response를 낮추는 항이 없다. [[hh-rlhf]] chosen side로 OPT-350M을 학습하면 [[orpo]] Fig. 3은 chosen과 함께 rejected response log-probability도 상승하며, 때로 rejected response가 chosen보다 더 probable해진다고 보인다(§3).

**메커니즘.** [[orpo]]는 sequence likelihood를 **length-normalized** quantity로 정의한다(Eq. 3). `log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t | x, y_<t)`이며 `m`은 token count다. 그 위에서 다음을 둔다.

```
odds_θ(y|x) = P_θ(y|x) / (1 − P_θ(y|x))                                                  (11)
OR_θ(y_w, y_l) = odds_θ(y_w|x) / odds_θ(y_l|x)                                           (12)
L_ORPO = E[ L_SFT(y_w|x) + λ · L_OR ],   L_OR = −log σ( log OR_θ(y_w, y_l) )              (13)
```

`L_SFT`는 chosen response의 causal-LM negative log-likelihood, `λ`는 odds-ratio term weight다. Eq. 3 length normalization이 odds를 의미 있게 한다. `P_θ`가 raw sequence probability라면 수백 token probability의 곱이라 `1 − P_θ ≈ 1`이 되고 `log OR`는 plain log probability ratio로 붕괴한다.

**계산 예시.** mean token log-probability가 −0.5인 chosen response는 `P_θ = 0.6065`, `odds = 1.541`이다. mean −1.0인 rejected는 `P_θ = 0.3679`, `odds = 0.582`다. 따라서 `log OR = 0.974`, `L_OR = −log σ(0.974) = 0.320`이다. 대응하는 log probability ratio는 `−0.5 − (−1.0) = 0.5`, loss 0.474다. 이 pair에서 odds-ratio margin이 둘 중 더 크므로 loss와 gradient는 더 작다. 방향은 paper Fig. 6 caption(“같은 input probability pair에서 log OR가 더 넓은 range”)과 맞지만, §7.1 body의 “probability ratio가 odds ratio보다 disfavored response를 더 극단적으로 discriminate한다”는 서술 및 SFT와 preference term을 합칠 때 odds ratio를 정당화하는 이유와는 맞지 않는다. source의 두 statement는 반대 방향이다. 위 계산은 caption과 일치한다. 저자가 어느 비교를 의도했는지는 **열린 질문**이다(§7.1 plot은 probability ratio에 `β`를 곱하고 caption은 그렇지 않다).

**근거.** separate SFT stage 없이 UltraFeedback만으로 ORPO를 학습한 Mistral-7B는 AlpacaEval 2.0 12.20%(Mistral-ORPO-β), IFEval instruction-level loose 66.19%, MT-Bench 7.32였다(Table 1, Table 6). OPT-1.3B에서 DPO 대비 ORPO reward-model win rate는 HH-RLHF 70.9%, UltraFeedback 57.8%였고(Tables 2–3), OPT-125M에서는 ORPO가 DPO에 진다(41.7% / 48.8%). 이 장에서 유용한 것은 Mistral-7B `λ` ablation(App. E.1, Fig. 9)이다. `λ = 0.1`에서는 rejected log-probability가 떨어지지 않고, `λ = 0.5`에서는 chosen은 오르고 rejected는 내리며, `λ = 1.0`에서는 margin은 커져도 둘 다 내려간다. training hyperparameter는 DPO-scale이 아니라 SFT-scale(peak LR `8e-6`)이다(App. C).

---

## §7 Iterative RPO: DPO + NLL anchor, round마다 refresh

**측정된 문제.** verifier-labeled pair가 있는 reasoning data에서 plain DPO는 positive term을 유지한 method보다 GSM8K accuracy를 낮췄다. Llama-2-70B-Chat에서 DPO 61.8%, SFT model에서 DPO 60.3%인 반면 gold chain of thought SFT는 63.5%, 아래 method 한 iteration은 73.1%였다([[rpo]] Table 1).

**메커니즘**([[rpo]] Eq. 1). pair마다 `c`를 chain of thought, `y`를 answer로 둔다.

```
L_DPO+NLL = −log σ( β·log[M_θ(c_w,y_w|x)/M_t(c_w,y_w|x)] − β·log[M_θ(c_l,y_l|x)/M_t(c_l,y_l|x)] )
            − α · log M_θ(c_w, y_w | x) / ( |c_w| + |y_w| )                              (14)
```

`M_t`는 previous iteration model이며 initialization과 reference 양쪽으로 사용한다. NLL term은 chosen sequence total length로 normalize하고 `α`는 두 term 균형을 잡는다. 각 iteration은 problem당 solution `N = 30`개를 sample하고 final answer를 gold answer와 비교해 label하며, problem당 `K = 10` pair(약 55–60K pair)를 만들어 next model을 학습한다.

**근거.** GSM8K training prompt만 쓴 Llama-2-70B-Chat은 zero-shot CoT 55.6% → 73.1% → 78.0% → 81.1% → 81.6%로 4 iteration에 걸쳐 상승했다(32 sample majority voting이면 88.7%). MATH는 12.5 → 20.8, ARC-Challenge는 77.8 → 86.7이다(Abstract, Table 1). iteration 1 data를 두 배로 하면 74.8%로, second iteration 78.0%보다 낮다. 그러므로 gain은 data quantity 효과만은 아니다(§4). setting은 `α = 1`({0.25, 0.5, 1, 2}에서 tuning), `β = 0.1`({0.05, 0.1, 0.5, 1.0}에서 tuning), LR `7e-7`, batch 16이다. paper Figure 3은 NLL term 없는 DPO에서 chosen-sequence log-probability가 training 중 떨어지고, NLL term이 있으면 떨어지지 않음을 보인다.

**산업 버전.** Llama 3.1 405B는 chosen sequence NLL coefficient 0.2, `β = 0.1`, LR `1e-5`, 양 response의 header·termination token masking을 둔 DPO를 쓴다. preference batch는 주로 가장 최근 round에서 뽑는다(arXiv:2407.21783v3 §4.1.4). post-training은 annotate-then-train 6 round를 실행한다(§4.1.6; [[llama-3-recipe]]). report는 [[rpo]], [[dpo-positive]]를 인용하며 NLL term이 training을 stabilize하고 chosen log-probability 감소를 막는다고 하지만 ablation은 인쇄하지 않는다.

---

## §8 이 family가 개선하는 것과 하지 않는 것

### 8.1 Capability profile

[[unpacking-dpo-ppo]]는 base policy(Tülu 2 13B SFT), evaluation suite, data size를 고정하고 factor 하나씩 바꾼다. SFT model(factuality 55.4, reasoning 47.8, coding 45.1, truthfulness 56.6, safety 91.8, instruction following 44.2) 대비 best dataset(aspect별 score를 붙인 UltraFeedback, 60,908 pair)의 DPO는 55.3 / 50.9 / 45.9 / 69.3 / 91.9 / 52.8이다. dataset 14개 전체에서 factuality는 54.7–55.7, reasoning은 46.0–50.9 범위에 있다. 같은 paper가 정한 factor 순서는 preference data가 첫째, learning algorithm 둘째, reward-model quality 셋째, 추가 unlabeled prompt 마지막이다(Abstract). dataset 다섯 개 평균 PPO−DPO는 reasoning +1.3, coding +2.9, safety +2.3, truthfulness −2.5, overall +0.7이다(Table 2). **결과(단일 연구)**, base model 하나다.

이를 [[simpo]] Table 9·Table 16, [[noise-contrastive-alignment]] Table 3과 함께 읽는다. 후자에서 Mixtral-8×7B-SFT를 UltraInteract에서 학습한 eight reasoning benchmark 평균은 DPO에서 56.3 → 50.1, NCA에서 57.9다. 독립 연구 셋에서 반복되는 pattern은 chat·instruction-following metric은 개선하고, math·code가 먼저 degrade하며, factuality가 양방향으로 가장 적게 움직인다는 것이다. **반복 확인됨(Replicated).**

### 8.2 Ranking accuracy는 학습하는 대상이 아니다

ranking accuracy를 `R = E 1[π_θ(y_w|x) ≥ π_θ(y_l|x)]`로 정의한다([[preference-ranking-accuracy]] Def. 2.3). 이는 trainer가 log하는 implicit-reward accuracy가 아니다. 후자는 reference model이 들어가는 `r̂_w`와 `r̂_l`을 비교한다. AlpacaFarm validation split에서 length-normalized / raw ranking accuracy는 Zephyr-7B-DPO 54% / 42%, Tülu-2-DPO-7B 53% / 42%, Llama-2-7B-Chat-HF 53% / 40%다. 같은 reference model에서 계산한 idealized ranking accuracy(`β` 범위 median)는 세 model에서 length-normalized 97–98%, raw 99%다(Table 1; 표의 네 번째 Gemma-7B-IT는 length-normalized median 73%).

메커니즘은 Theorem 4.1이다. reference log-ratio가 `log[π_ref(y_l|x)/π_ref(y_w|x)] = c`인 pair에서 trained model이 pair를 correctly rank할 필요충분조건은 pair DPO loss가 `L ≤ −log σ(βc)`인 것이다.

**계산 예시.** `β = 0.1`로 둔다. reference model이 `c = 5` nat만큼 틀린 pair는 pair loss가 `log 2 = 0.693`에서 `−log σ(0.5) = 0.474`까지 떨어져야 한다. reference가 `c = 30` nat만큼 틀린 pair는(token 수십 개 길이가 다른 response에서는 쉬움) `−log σ(3) = 0.049`가 필요하다. average loss를 0.3으로 낮춘 training은 첫 종류 pair는 flip하지만 둘째는 flip하지 않는다. Pythia-2.8B의 Anthropic HH validation loss 최저점에서 empirical하게 initially mis-ranked training pair 중 flip된 것은 10% 미만이었다(§4.1, Fig. 2).

**함의.** trainer log의 `rewards/accuracies` 상승은 policy가 대부분 pair에서 rejected response에 더 높은 likelihood를 여전히 배정하는 것과 양립한다. win rate와 상관하는 quantity는 reference에서의 distance다. ranking accuracy와 win rate는 초기에 같이 움직이지만 policy가 `π_ref`에서 멀어지면 anti-correlated가 된다(§5).

---

## §9 Offline pair, on-policy pair, over-optimization

**reward peak가 있는 위치.** [[on-policy-suboptimal-preference-data]]는 on-policy sampling, sample reuse, negative gradient로 method를 분류하고, 마지막 둘의 benefit을 geometry에 연결한다. high-reward region이 `π_ref`가 작은 mass를 두는 곳에 있으면 둘 다 도움이 되며, high-reward region이 이미 `π_ref` 아래 likely하면 어느 것도 필요 없다(§5, Fig. 1 right). 근거는 bandit problem, synthetic length-control task, AlpacaFarm/UltraFeedback run이다.

**같은 loss, 다른 sampling.** [[on-off-policy-rlhf]]는 IPO loss·hyperparameter를 고정하고 pair를 current policy에서 sample하는지만 바꾼다. on-policy sampling은 시험한 dataset 네 개 모두에서 더 높은 peak win rate에 도달한다. pairwise classifier로 쓰면 offline policy는 training 중 약 70% accuracy에서 peak하지만 같은 size proxy preference model은 70–90%에 도달하고 offline policy는 더 나쁜 response를 생성한다(§2.1 Fig. 1; §5.3.1 Fig. 8).

**external reward model 없이도 생기는 over-optimization.** [[daa-overoptimization-scaling]]은 Pythia 1B·2.8B·6.9B에서 TL;DR(92K pair)의 DPO·IPO·SLiC을 regularization setting 일곱 개로 학습한다. reference summary 대비 GPT-4 win rate는 모든 objective에서 achieved KL에 대해 hump-shaped이고, wide KL budget에서는 best checkpoint가 한 epoch의 약 25% 뒤에 나타난다(§3.1, Figs. 1–2). model size 안에서는 implicit-reward classification accuracy도 training loss도 win rate를 예측하지 못한다(§3.2, Fig. 4). 가장 작은 model이 가장 빨리 degrade하고 IPO가 가장 적게 degrade한다.

**chosen likelihood 하락은 일부 예상된다.** `π_ref`가 preferred response로 학습한 SFT model이면 `E_{p_D(y_w|x)}[log π_θ(y_w|x)/π_ref(y_w|x)] ≈ −D_KL(π_ref ‖ π_θ)`이며 이는 최대 0이고 policy가 움직일수록 내려간다([[daa-overoptimization-scaling]] §3.5, Eq. 7). 따라서 chosen log-probability 하락만으로 broken run 증거가 되지는 않는다. 중요한 것은 sample quality가 함께 하락하는가다.

**더 높은 likelihood도 더 좋지 않다.** [[likelihood-overoptimisation-daa]]는 Command R 7B·35B를 binarised UltraFeedback과 internal set에서 Hinge·DPO·IPO로 학습한다. mean chosen log-likelihood에 따른 win probability는 line보다 quadratic fit이 더 좋다(RMSE가 7B에서 24.4%, 35B에서 25.8% 낮음). best checkpoint는 chosen/rejected likelihood heatmap Pareto corner가 아니라 중간에 있고, 약 1,000 step 뒤 더 큰 margin은 falling win probability와 함께 온다(§4.2, Figs. 1–3). 제안한 stopping signal은 top-k entropy reversal과 shrinking top-k probability mass다(§4.3).

---

## §10 implementation이 계산하는 것

commit 64c1cc4의 [[openrlhf-dpo]], `openrlhf/models/loss.py` L246–281은 다음과 같다.

```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios

if self.ipo:
    losses = (logits - 1 / (2 * self.beta)) ** 2   # Eq. 17 of arXiv:2310.12036v2
else:
    losses = (
        -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        - F.logsigmoid(-self.beta * logits) * self.label_smoothing
    )
loss = losses.mean()
chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

`logits`는 §2.2 margin `h`다. DPO, cDPO(label smoothing), IPO는 마지막 expression만 다르다. 주변 code에서 모든 trainer에 확인할 만한 implementation fact 세 가지가 나온다.

1. `_get_batch_logps`(dpo_trainer.py L362–388)는 `average_log_prob == False`를 assert하고 prompt token을 mask한 token log-probability의 **합**을 반환한다. mean을 반환하는 trainer는 length-normalized DPO를 계산하는 것이며 다른 `β`가 필요하다(Tülu 3는 5, [[tulu-3]] Table 20).
2. optional NLL term은 chosen response mean token log-probability이며 `nll_loss * args.model.nll_loss_coef`로 더한다(dpo_trainer.py L157, L172–180). CLI default coefficient는 0이다([[openrlhf-dpo-recipe]]). 따라서 설정하지 않으면 §7 anchor는 off다.
3. log quantity는 `acc = (chosen_reward > reject_reward).mean()`, `chosen_reward`, `reject_reward`다(dpo_trainer.py L184–197). `chosen_reward`는 `β(log π_θ(y_w) − log π_ref(y_w))`다. frozen reference에서는 trend가 chosen log-probability trend다. `acc`는 §8.2의 implicit-reward accuracy이지 ranking accuracy가 아니다.

같은 loss의 online variant, 즉 매 step 두 response를 regenerate해 judge로 label하는 방식은 [[trl-online-dpo]]를 보라. 그 label을 내는 judge는 ch-42(Reward Hacking and Judge Design) 주제다.

---

## §11 Single-turn chat 너머: long context와 agent trajectory

이 setting에서 바뀌는 것은 loss가 아니라 pair construction이다.

- **Long context.** Llama 3.1은 short-context preference data만으로 DPO를 실행했고 SFT model이 이미 long context에 강하다는 조건 아래 long-context performance가 regression하지 않았다고 보고한다([[llama-3-recipe]] §4.3.4). [[longpo]]는 대신 long-context setting 안에서 pair를 만든다. chosen은 relevant chunk만 주었을 때 model 자신의 answer, rejected는 full document를 주었을 때 answer다. reference distribution은 chunk에 condition하고 chosen long sequence에는 NLL term을 유지한다. Mistral-7B-Instruct-v0.2는 InfiniteBench 13.82 → 39.27, MMLU 59.15 → 59.99로 움직인다. 같은 pair의 DPO는 25.56에 도달하고 short-context score를 degrade한다.
- **Agent trajectory.** [[eto-trial-and-error]]는 trajectory 전체를 pair 양쪽으로 쓴다. expert trajectory가 chosen, 같은 task에서 SFT agent 자신의 failed trajectory가 rejected다. DPO는 `β = 0.1`(ALFWorld에서는 0.5), LR `1e-6`, 3 epoch다. Llama-2-7B-Chat은 WebShop average reward가 SFT agent 63.1 대비 67.4에 도달하지만 세 번째 explore-train iteration 뒤 decline한다. rejected side는 failure가 몇 step에 localize된 long sequence지만 gradient는 모든 token을 동일하게 취급한다. 대안인 step-level credit assignment는 ch-44(Process Supervision and Verifiable Rewards), negative gradient 자체의 행동은 ch-43a가 다룬다.

---

## 음성 샘플과 음성 피드백

**어떤 음성인가.** 이 장 모든 objective의 rejected response는 **gradient로 쓰는 음성**(category 4)이다. loss에는 `log π_θ(y_l|x)`를 낮추는 효과의 term이 있다. ORPO·RPO는 chosen response에 positive cross-entropy term도 유지하며 [[d2o-negating-negatives]]는 positive side를 self-sample distribution으로 바꾼다.

**음성의 출처와 신뢰도.** human comparison([[hh-rlhf]]), per-aspect score LLM judge([[ultrafeedback]]), verifier([[rpo]] final-answer check), policy self-sample 위 reward model([[simpo]] Instruct, [[pairrm]])이다. label reliability도 input 일부다. [[d2o-negating-negatives]] Fig. 1에서 PKU-SafeRLHF *preferred* response 33.88%, HH 8.03%를 off-the-shelf classifier가 toxic으로 표시한다. [[likelihood-displacement]] §6.2에서는 SORRY-Bench prompt 70% 이상에서 sample 둘 다 refusal이라 “rejected” label이 같은 종류 response 둘을 구분한다.

**softmax 하나 수준의 메커니즘.** vocabulary 위 next-token distribution `p`에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. `log p_y`를 낮추는 step은 `Δz = −η(e_y − p)`다. target logit은 `η(1 − p_y)`만큼 떨어지고 모든 다른 logit은 이미 likely한 정도 `η p_j`만큼 오른다. renormalization 뒤 mass는 argmax에 집중한다.

**계산 예시.** `p = (0.70, 0.20, 0.08, 0.02)`에서 `η = 1`로 마지막 token을 push-down하면 새 distribution은 `(0.806, 0.140, 0.050, 0.004)`다. target은 0.02 → 0.004, 가장 likely token은 +0.106, 중간 둘도 mass를 잃는다(−0.060, −0.030). `η = 3`이면 `(0.924, 0.059, 0.016, 0.0002)`다. 이미 unlikely한 token을 내리면 recovered mass 거의 전부가 current argmax로 간다. 이는 [[learning-dynamics-llm-finetuning]] App. E가 logistic regression에서 증명한 squeezing effect이며 `y* = argmax_{i ≠ y⁻} π(i)`로 squeeze되고 distribution이 peakier할수록, `π(y⁻)`가 작을수록 강하다. [figures/negative-gradient-squeeze.html](figures/negative-gradient-squeeze.html)에서는 distribution·step size·다른 token의 positive term 동시 적용 여부를 바꿀 수 있다.

**positive term이 자동으로 고치지 않는 이유.** 같은 예에서 token 2에 `η = 1` positive step만 적용하면 `(0.392, 0.502, 0.083, 0.022)`다. token 2 positive와 token 4 negative를 같이 적용하면 `(0.526, 0.409, 0.060, 0.006)`다. positive만 쓴 경우와 비교하면 push-down은 mass를 dominant token 1(0.392 → 0.526)으로 보내고 intended token 2(0.502 → 0.409)에서 뺀다. negative redistribution 몫 대부분은 positive가 올리는 response가 아니라 argmax로 간다. 이는 [[dpo-positive]] §3 sequence-level 결과의 token counterpart다. 한 position에서 다른 pair의 뒤 position logit DPO gradient는 `s_j^{(y_l^{<k})} − s_j^{(y_w^{<k})}`이며 difference 뒤 모든 position의 correct-token logit을 낮춘다. MetaMath sample 900개에서 preferred completion의 differing index 뒤 token average log-probability는 reference −0.37, DPOP −0.26, DPO −1.82였다(§5, Fig. 4).

**이익 근거.** [[on-policy-suboptimal-preference-data]] §5.2는 reward peak가 `π_ref` 아래 unlikely할 때 negative gradient를 포함한 offline objective가 같은 data supervised objective보다 더 빨리 더 좋은 solution에 수렴한다고 한다. [[orpo]] Fig. 3 대 Fig. 7은 negative term이 없으면 chosen-response SFT 중 rejected log-probability가 오르고 `λ = 1.0`이면 떨어짐을 보인다.

**해악 근거.** [[likelihood-displacement]]의 SORRY-Bench on-policy refusal pair DPO는 training-set refusal rate를 Llama-3-8B-Instruct 74.4% → 33.4%, Gemma-2B-IT 80.5% → 54.8%로 내렸고 preferred log-probability mean change는 −48.1 ± 22.1, −59.2 ± 5.3이었다(§6.2, Table 16). [[rpo]] Table 1은 DPO GSM8K 61.8%, 같은 data NLL term 73.1%를, [[noise-contrastive-alignment]] Table 3은 Mixtral-8×7B UltraInteract DPO에서 eight-benchmark average 56.3 → 50.1을 보고한다.

| 제어 | 형태 | 근거 |
|---|---|---|
| chosen positive NLL anchor | `+ α·NLL(y_w)`, token당 `α = 1`([[rpo]] Eq. 1); 405B coefficient 0.2([[llama-3-recipe]] §4.1.4) | 한 iteration GSM8K 61.8 → 73.1([[rpo]] Table 1); chosen log-probability 하락 중단(Fig. 3) |
| preference loss 내부 SFT term | ORPO `L_SFT + λ·L_OR`([[orpo]] Eq. 6) | λ = 0.5는 chosen 상승/rejected 하락, λ = 1.0은 둘 다 하락(App. E.1, Fig. 9) |
| chosen이 reference 아래로 갈 때 penalty | DPOP sigmoid 안 `−λ·max(0, log[π_ref(y_w)/π_θ(y_w)])`, `λ = 50`, `β = 0.3`([[dpo-positive]] Eq. 3) | low-edit-distance MetaMath에서 base model을 개선한 유일 method(Fig. 2) |
| relative 아닌 absolute target | NCA `−log σ(r_w) − ½log σ(−r_w) − ½log σ(−r_l)`([[noise-contrastive-alignment]] Table 1) | 8×7B reasoning average 50.1(DPO) → 57.9(NCA)(Table 3) |
| positive side를 self-sample로 대체 | D2O `(β/K)Σ_i log[π_θ(y_i)/π_{r-}(y_i)] − α log[π_θ(y_l)/π_{r+}(y_l)]`([[d2o-negating-negatives]] Eq. 3) | harmfulness −4.27/win rate 61.82, DPO −1.02/32.43, negative plain gradient ascent 1.21/20.13(Table 1) |
| pair curate | high length-normalized CHES pair drop([[likelihood-displacement]] §6.3) | lowest-CHES 5% subset이 SFT term 추가보다 refusal rate를 더 복구 |
| rejected side를 먼저 덜 unlikely하게 함 | SFT stage에 `y⁻` 포함([[learning-dynamics-llm-finetuning]] §4.3) | baseline pipeline 대비 win rate: DPO epoch 2/4/6에서 0.65 / 0.69 / 0.67(ChatGPT judge, Table 1) |

**control 하나의 계산 검증.** NCA form에는 relative loss에 없는 fixed point가 있다. chosen term에서 `d/dr[−log σ(r) − ½ log σ(−r)] = −(1 − σ(r)) + ½σ(r) = 0`은 `σ(r) = 2/3`, 즉 `r_w = ln 2`에서 성립한다. chosen implicit reward는 `+∞`가 아니라 positive constant로 향한다. 이것이 Fig. 5 likelihood behaviour의 paper 설명이다.

**진단.** `chosen_reward`, `reject_reward`를 difference만이 아니라 따로 log한다(§10). failure mode는 둘이 같이 떨어지는 것이다. greedy-decode log-probability 또는 top-k entropy도 log한다([[likelihood-overoptimisation-daa]] §4.3; [[learning-dynamics-llm-finetuning]] §4.2에서 greedy-decoded sequence log-probability는 8 epoch에 −113 → −63이지만 관측한 나머지는 모두 하락). reasoning data는 loss 대신 sampled pass@1·pass@k를 log한다. model size 안에서 loss는 quality를 예측하지 않는다([[daa-overoptimization-scaling]] §3.2). safety data는 held-out prompt refusal rate를 log한다([[likelihood-displacement]] §6).

**일반성에 미치는 효과.** negative term은 SFT model이 점유한 region 밖으로 probability mass를 옮기며 label되지 않은 response에서도 mass를 제거한다. 측정 결과는 correct response가 rare한 task의 sampled accuracy 감소([[rpo]], [[noise-contrastive-alignment]]), pair 양쪽이 같은 종류일 때 safety prompt refusal 감소([[likelihood-displacement]]), output diversity 감소이며 reversal point는 win-rate turning point와 일치한다([[likelihood-overoptimisation-daa]]).

## 레시피

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DPO paper runs (GPT-2-large, Pythia-2.8B, GPT-J) | ≤ 6B | preference | β | 0.1 (TL;DR run: 0.5) | arXiv:2305.18290v3 App. B | verified 2026-09-14 ([[dpo]]) | "we did not meaningfully tune DPO's β" (§6.2); IMDb sweep {0.05, 0.1, 1, 5} |
| DPO paper runs | ≤ 6B | preference | optimizer; LR; warmup; batch | RMSprop; 1e-6; linear over 150 steps; 64 | App. B | verified 2026-09-14 ([[dpo]]) | no ablation reported |
| Tülu 2 13B + DPO | 13B | preference | β; LR; epochs | 0.01; 5e-7; 3 | arXiv:2406.09279v2 App. F.1 | verified 2026-09-15 ([[unpacking-dpo-ppo]]) | pilot sweep β {0.1, 0.01, 0.001}, LR {5e-6, 5e-7, 5e-8} on HH-RLHF and UltraFeedback |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | loss; β; LR; batch (pairs); epochs; max length | length-normalized DPO; 5; 5e-7; 128; 1; 2,048 | open-instruct `docs/tulu3.md` L144–159; report §5.4.1 | verified 2026-09-14 ([[open-instruct-allenai-recipes-recipe]]) | Table 18: SFT base 55.7, DPO (β 0.1) 55.2, SimPO 51.8/52.9, DPO-norm best 57.3 |
| Llama 3.1 405B | 405B | preference | β; LR; NLL coefficient on chosen; token masking | 0.1; 1e-5; 0.2; header and termination tokens masked | arXiv:2407.21783v3 §4.1.4 (rounds: §4.1.6, 6 rounds) | verified 2026-09-14 ([[llama-3-recipe]]) | no ablation printed; NLL term cited to [[rpo]] and [[dpo-positive]] |
| Llama-3-Instruct SimPO v0.1 (8B) | 8B | preference | β; γ; LR; batch; epochs; max length | 2.5; 1.4; 1e-6; 128; 1; 2,048 | arXiv:2405.14734v3 App. B, Table 8 | verified 2026-09-15 ([[simpo]]) | Table 12: AlpacaEval 2 LC 44.7 against DPO 48.2 and SFT 26.0 in the same setting |
| Llama-3-Instruct SimPO v0.2 (8B, preferences labeled by ArmoRM) | 8B | preference | β; γ; other settings | 10; 3; as v0.1 | arXiv:2405.14734v3 App. H | verified 2026-09-15 ([[simpo]]) | Table 12: LC 53.7, the highest in the table; Table 16 LR comparison 4e-7 / 5e-7 / 1e-6 → LC 38.8 / 44.6 / 53.7 with ZeroEval GSM 77.9 / 77.0 / 57.4 |
| ORPO runs (OPT-125M/350M/1.3B, Phi-2 2.7B, Llama-2 7B, Mistral 7B) | 0.125–7B | preference | peak LR; epochs; max length | 8e-6; 10 (best checkpoint by evaluation loss); 2,048 for UltraFeedback, 1,024 for HH-RLHF | arXiv:2403.07691v2 App. C | verified 2026-09-15 ([[orpo]]) | no ablation reported for LR |
| Mistral-ORPO-α, -β | 7B | preference | λ | 0.1 | arXiv:2403.07691v2 §6.1 | verified 2026-09-15 ([[orpo]]) | App. E.1 Fig. 9: λ ∈ {0.1, 0.5, 1.0} changes whether rejected log-probabilities fall; λ = 0.25 used for Phi-2 and 0.2 for Llama-2 (§6.1) |
| Zephyr-β-SFT + KTO | 7B | preference | β; λ_D, λ_U; effective batch; epochs | 0.1; 1.0, 1.0; 32; 1 | arXiv:2402.01306 §4.2 (batch), Table 1 caption (λ), Table 2 (run) | verified 2026-09-15 ([[kto]]) | Table 2: GSM8K 39.0 (SFT) / 40.0 (DPO) / 53.5 (KTO) |
| Llama-3 8B, SFT+KTO | 8B | preference | LR; β | 5e-6; 0.05 | Table 1 | verified 2026-09-15 ([[kto]]) | Table 1: AlpacaEval LC 10.59, BBH 65.15, GSM8K 60.20; the paper states the model is more sensitive to the learning rate than any other hyperparameter (§4.2) |
| Iterative RPO from Llama-2-70B-Chat | 70B | preference | β; α; LR; batch; samples per problem; pairs per problem; iterations | 0.1; 1.0; 7e-7; 16; N = 30; K = 10; 4 | arXiv:2404.19733v3 "Experimental setup" | verified 2026-09-15 ([[rpo]]) | α tuned over {0.25, 0.5, 1, 2}, β over {0.05, 0.1, 0.5, 1.0}; Table 1: 73.1 → 78.0 → 81.1 → 81.6 |
| Mistral-7B-based model + DPOP | 7B | preference | β; λ | 0.3; 50 | arXiv:2402.13228v2 §5 | verified 2026-09-15 ([[dpo-positive]]) | λ ∈ {5, 50, 500} changed results little; β ∈ {0.1, 0.3, 1.0} did not prevent the DPO failure (Fig. 3) |
| Mixtral-8×7B-SFT + NCA | 8×7B (MoE) | preference | β; LR; epochs; schedule | 0.1; 5e-7; 1; cosine, warmup ratio 0.1 | arXiv:2402.05369v3 App. C | verified 2026-09-15 ([[noise-contrastive-alignment]]) | Table 3: average 56.3 (SFT) / 50.1 (DPO) / 57.9 (NCA) |
| Pythia 1B / 2.8B / 6.9B on TL;DR | 1–6.9B | preference | data; batch; optimizer; LR; epochs | 92K pairs; 128; RMSProp; 5e-7 with 150 warmup steps; 1 | arXiv:2406.02900v2 App. "Experimental details" | verified 2026-09-15 ([[daa-overoptimization-scaling]]) | Figs. 1–2: hump-shaped win rate versus KL; best point near 25% of an epoch |
| Command R 7B / 35B | 7B, 35B | preference | batch; LR; epochs; monitoring | 32; 5e-6 or 1e-5; 1; evaluate every 50 steps with early stopping | arXiv:2410.11677v2 §4.1 | verified 2026-09-15 ([[likelihood-overoptimisation-daa]]) | Figs. 1–4: quadratic fit of win probability against chosen log-likelihood |
| OpenRLHF `train_dpo.py` CLI defaults | any | preference | β; label smoothing; NLL coefficient; epochs; LR | 0.1; 0; 0; 1; 1e-5 | train_dpo.py L236–246, L255 | verified 2026-09-14 ([[openrlhf-dpo-recipe]]) | no ablation reported; class defaults differ (β 0.01, 2 epochs) |
| OpenRLHF `train_dpo_llama.sh` example | 8B | preference | β; LR; epochs; max length | 0.1; 5e-7; 1; 8,192 | train_dpo_llama.sh L13–17 | verified 2026-09-14 ([[openrlhf-dpo-recipe]]) | no ablation reported |


**소규모 범용 run의 시작점.** 약 61K pair binarised UltraFeedback 규모 preference set과 8B SFT checkpoint라면 이 표에서 가장 근거가 많은 구성은 summed-log-probability DPO `β = 0.1`, LR `5e-7`, batch 128 pair, 한 epoch, max length 2,048, SFT checkpoint frozen reference다. LR `5e-7`은 OpenRLHF 8B example script와 Tülu 3 8B 모두의 값이고, batch 128/max length 2,048은 Tülu 3 8B 값이다(OpenRLHF example은 global batch 256, max length 8,192; [[openrlhf-dpo-recipe]], [[open-instruct-allenai-recipes-recipe]]). trainer가 summed가 아니라 mean log-probability를 반환하면 해당 setting은 length-normalized DPO `β = 5`다(Tülu 3 8B, Table 18: starting checkpoint 55.7 대비 57.3). data가 verifiable answer를 담으면 chosen NLL `α = 0.2`–`1.0`을 더한다([[llama-3-recipe]], [[rpo]]). epoch 끝 대신 50–100 step마다 평가한다. [[daa-overoptimization-scaling]] TL;DR peak가 epoch 약 25%에 있었고 [[likelihood-overoptimisation-daa]] Command R run이 50 step마다 early stopping과 함께 monitor됐기 때문이다.

## 일반화 관점

**(a) 범위를 넓히는 것.** 더 나은 preference data가 먼저다. [[unpacking-dpo-ppo]]에서 fixed algorithm의 dataset 14개 spread(average 51.5~61.0)는 DPO 대 PPO gap 0.7보다 크고 per-aspect-scored synthetic data가 14개 중 최고다. pair 양 side가 semantically distinct한 것: [[on-policy-suboptimal-preference-data]] Fig. 17은 UltraFeedback에서 Mistral-7B chosen likelihood는 상승(다른 generator model response)하지만 AlpacaFarm Pythia-1.4B에서는 하락함을 보인다. current checkpoint로 regenerate한 pair: [[simpo]] on-policy Instruct setup, Tülu 3 on-policy preference construction([[tulu-3]] §5.2.1), [[on-off-policy-rlhf]] controlled comparison. correctness가 중요하면 chosen response positive term([[rpo]], [[noise-contrastive-alignment]], [[dpo-positive]]).

**(b) 좁아짐을 일으키는 것.** win-rate peak를 넘겨 실행(모든 size·objective에서 KL 대비 hump-shaped win rate, [[daa-overoptimization-scaling]] §3.1). 이미 instruction-tuned checkpoint에서 큰 LR: released Llama-3-8B-Instruct SimPO checkpoint LR `1e-6`은 ZeroEval GSM 78.5 → 57.4, MMLU 61.7 → 54.9([[simpo]] Table 16). near-identical reasoning pair: low-edit-distance MetaMath failure([[dpo-positive]] Fig. 2), UltraInteract regression([[noise-contrastive-alignment]] Table 3). 양 side가 같은 kind response인 pair: refusal 74.4% → 33.4% collapse([[likelihood-displacement]] §6.2). diversity: top-k entropy 방향이 reverse하며 win-rate turning point와 일치([[likelihood-overoptimisation-daa]] §4.3).

**(c) 이 stage를 측정하는 법.** length control held-out win rate([[length-controlled-alpacaeval]]: Chatbot Arena Spearman 0.94 → 0.98, verbosity prompt raw score 22.9% → 64.3%, control 뒤 41.9% → 51.6%). preference data가 target하지 않는 fixed capability suite(MMLU, GSM8K, HumanEval, IFEval)를 같은 checkpoint에서 평가한다. 이것이 SimPO regression을 보이게 했다. loss 대신 sampled metric을 쓴다. training loss·implicit-reward accuracy 모두 model size 안 win rate를 예측하지 않는다([[daa-overoptimization-scaling]] §3.2). safety data가 mixture에 있으면 refusal rate safety slice도 둔다. 알려진 error: implicit-reward accuracy는 ranking accuracy가 아니고([[preference-ranking-accuracy]] Remark 2.5), ranking accuracy는 policy가 reference에서 멀어지면 win rate와 anti-correlated가 되며(§5), judge가 long answer를 선호하는 것을 length control이 없앤다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| `rewards/accuracies`를 pair correct ranking 근거로 읽음 | accuracy 0.7 이상인데 sampled output 개선 없음 | held-out pair에서 직접 `1[π_θ(y_w) ≥ π_θ(y_l)]` 계산; 60% 미만 예상([[preference-ranking-accuracy]] Table 1) |
| margin만 log | loss 하락·margin 상승·quality 하락 | §10처럼 `chosen_reward`, `reject_reward`를 따로 log; 둘 다 하락해 margin이 크면 displacement regime |
| chosen log-probability 하락 자체를 bug로 취급 | healthy run을 멈추거나 필요 없이 anchor 추가 | forward-KL identity([[daa-overoptimization-scaling]] §3.5)와 비교하고 sampled accuracy/win rate로 결정 |
| 다른 loss family β 사용 | loss가 거의 안 움직이거나 수백 step 안 diverge | reward unit에 β를 맞춘다: summed DPO 0.1, length-normalized DPO 5([[tulu-3]] Table 20), SimPO β 2.0–2.5·γ 0.5–1.5([[simpo]] §3, Table 8; v0.2는 β=10, γ=3, App. H) |
| data 출처와 다른 reference model | ranking accuracy 정체, Theorem 4.1의 큰 `c` | response를 만들거나 score한 SFT checkpoint로 `π_ref` 설정([[dpo]] §4; [[preference-ranking-accuracy]] Theorem 4.1) |
| intermediate eval 없이 full epoch 이상 학습 | final checkpoint가 epoch 25% checkpoint보다 나쁨 | 50–100 step마다 평가([[daa-overoptimization-scaling]] Fig. 2; [[likelihood-overoptimisation-daa]] §4.1) |
| pair 양쪽이 같은 종류 | loss는 healthy하나 refusal/task accuracy 하락 | pair similarity(length-normalized CHES, [[likelihood-displacement]] §5) score 후 highest drop |
| 학습 뒤 judge win rate만 보고 | chat score 상승, capability score 하락·미보고 | fixed capability suite([[simpo]] Tables 9·16)와 함께 length-controlled win rate([[length-controlled-alpacaeval]]) 보고 |
| near-identical math pair에서 DPO 실행 | training margin 상승·sampled accuracy 하락 | pair edit distance 측정, DPOP penalty/NLL anchor 추가([[dpo-positive]] §5; [[rpo]] Table 1) |

## 이해도 점검

1. margin `h = 7`이 같은 두 pair가 하나는 chosen log-probability를 2 nat 올리고 다른 하나는 3 nat 내렸다. DPO loss·gradient weight가 같은 이유와 책임지는 gradient term을 설명하라.
2. 동료가 `rewards/accuracies` 0.85를 보고 model이 preferred를 dispreferred보다 rank한다고 결론낸다. [[preference-ranking-accuracy]] Theorem 4.1로 이 결론이 실패하는 조건을 말하고 `β = 0.1`, reference log-ratio 20 nat pair를 flip하는 per-pair loss를 계산하라.
3. ORPO odds ratio가 probability ratio와 다르게 행동하는 이유, Eq. 11 `P_θ(y|x)`가 [[orpo]] Eq. 3 length-normalized quantity가 아닌 raw sequence probability면 생길 문제를 설명하라.
4. squeezing 예시는 argmax token으로 mass를 보낸다. *current* policy가 이미 unlikely하다고 보는 response를 내리는 일이 likely response를 내리는 일보다 이 점에서 왜 더 나쁜지 설명하고 [[on-policy-suboptimal-preference-data]] on-policy 결과에 연결하라.
5. Llama-3-8B-Instruct에서 SimPO는 AlpacaEval 2 LC 26.0 → 53.7, ZeroEval GSM 78.5 → 57.4다. objective·learning rate 관점의 causal account와 release LR를 정하기 위해 측정할 것을 말하라.
6. Llama 3의 NLL coefficient 0.2와 header/termination token mask가 각각 막는 것, 둘째가 loss가 token sum이라는 사실에 특유한 이유를 설명하라.
7. [[daa-overoptimization-scaling]]은 model size 안 training loss와 win rate 관계가 없고 [[likelihood-overoptimisation-daa]]는 higher chosen likelihood가 monotonic하게 좋지 않다고 한다. 같은 finding인가? quantity별 측정 대상에 근거해 답하라.
8. judge pair 60K, 8B SFT model이 있고 GSM8K·MMLU가 1점 이상 떨어지면 안 된다. objective·stopping rule을 고르고 각각의 근거를 말하라.

## 연결

- **이전: ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** Eq. (1)이 formalize하는 on-policy/offline framing·KL-to-reference 논거.
- **다음: ch-42 — Reward Hacking and Judge Design.** 이 장 pair label judge도 optimization target이며 여기 length control은 그 장 general problem의 사례.
- **ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization.** §2.2 Bradley-Terry likelihood와 [[daa-overoptimization-scaling]]이 비교하는 over-optimization baseline.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.** 여기 rejected term 역할을 하는 negative advantage의 on-policy family.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.** 이 장 negative-feedback section에서 요약한 완전한 derivation.
- **ch-45a — Preference-Optimization and RL Stage Recipes Side by Side.** 이 recipe 표를 RL-stage 표 옆에 둔다.

## 출처

- [[dpo]] — derivation (Eq. 3–7), implicit reward, gradient, App. B hyperparameters, TL;DR and CNN/DailyMail results.
- [[ipo]] — ΨPO family, the deterministic-preference argument, the squared loss with target `1/(2τ)`.
- [[kto]] — unary-label objective, the batch reference point `z_0`, the `λ_D n_D / λ_U n_U` rule, Table 1 and Table 2 results.
- [[simpo]] — length-normalized reference-free reward, `γ`, Table 4 win rates, Table 9 and Table 16 capability regressions.
- [[orpo]] — length-normalized odds, the joint SFT plus odds-ratio loss, the `λ` ablation on rejected log-probabilities.
- [[rpo]] — DPO+NLL loss (Eq. 1), iteration protocol, GSM8K/MATH/ARC numbers, chosen-log-probability curves.
- [[likelihood-displacement]] — definition of displacement, CHES score, refusal-rate collapse and its mitigation.
- [[learning-dynamics-llm-finetuning]] — the squeezing effect and its proof setting, greedy-response log-probability, the SFT "extend" mitigation.
- [[dpo-positive]] — the low-edit-distance failure mode, the DPOP penalty, the token-level measurement.
- [[noise-contrastive-alignment]] — DPO as a special case of InfoNCA, the absolute-likelihood NCA form, UltraInteract results.
- [[d2o-negating-negatives]] — alignment from negatives only against a distribution of self-samples, and the failure of plain gradient ascent on negatives.
- [[unpacking-dpo-ppo]] — the controlled 13B comparison of data, algorithm, reward model and prompts.
- [[preference-ranking-accuracy]] — ranking accuracy definition and measurements, Theorem 4.1, relation to win rate.
- [[daa-overoptimization-scaling]] — KL budgets, intra-epoch degradation, loss and implicit-reward accuracy as non-predictors.
- [[likelihood-overoptimisation-daa]] — quadratic relation between chosen likelihood and win probability; entropy and top-k stopping signals.
- [[on-policy-suboptimal-preference-data]] — the on-policy / sample-reuse / negative-gradient taxonomy and the geometric condition.
- [[on-off-policy-rlhf]] — same-loss online versus offline comparison and the classification-versus-generation gap.
- [[length-controlled-alpacaeval]] — the GLM, the length-controlled win rate, and the verbosity-prompt test.
- [[openrlhf-dpo]], [[openrlhf-dpo-recipe]] — the reference implementation of the loss, the NLL option, the logged quantities, and the defaults.
- [[open-instruct-allenai-recipes-recipe]], [[tulu-3]] — length-normalized DPO, Table 18 algorithm ablation, and the released 8B/70B/405B settings.
- [[llama-3-recipe]] — the 405B DPO settings, NLL coefficient, token masking, and round structure.
- [[trl-online-dpo]] — the online variant of the same loss, used in ch-42.
- [[ultrafeedback]], [[hh-rlhf]], [[pairrm]] — the preference datasets and the ranking model used to build on-policy pairs.
- [[longpo]], [[eto-trial-and-error]] — pair construction for long context and for agent trajectories.
- [[rlhf-instructgpt]] — the reward-model-plus-PPO pipeline whose objective this chapter optimizes without RL.
