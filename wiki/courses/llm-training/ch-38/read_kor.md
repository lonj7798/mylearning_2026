<!-- chapter: ch-38
     track: rl
     kind: content
     title: KL-Controlled RLHF — TRPO, PPO, InstructGPT
     deps: [ch-37]
     sources: [[trpo]], [[ppo]], [[rlhf-instructgpt]], [[kl-control-rlhf]], [[llama-2]], [[costa-huang-ppo-details]], [[hf-rlhf-illustrated]], [[lilianweng-rlhf]], [[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]]
     figures: figures/ppo-clip.html
-->

# 제38장 — KL-Controlled RLHF: TRPO, PPO, InstructGPT

> **핵심 통찰.** 현대 RLHF는 "reward model에 RL을 돌리는 것"이 아니다. 이는 `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`에서 나오는 *KL-regularized* amortized inference다(Korbak 관점, [[kl-control-rlhf]]). PPO clipped surrogate는 TRPO trust region([[trpo]])의 1차 근사이며, InstructGPT의 Equation 2([[rlhf-instructgpt]])는 그 surrogate에 token별 KL penalty와 pretraining-mix coefficient를 붙인다. 나머지 — Llama-2의 dual RM, Costa-Huang의 37가지 trick, TRL/OpenRLHF/verl의 loss kernel — 는 이 두 사실 주변의 배관이다.
>
> **지침.** PPO-clip을 ε=0.2, GAE λ≈0.95, γ=1.0(reward가 EOS에 집중됨)으로 사용하라. `−β·KL(π‖π_ref)`는 loss가 아니라 token별 *reward*에 더하라(k3 estimator). value head는 RM에서 초기화하고, advantage를 minibatch별로 whiten하며, value prediction을 clip하라. β≈0.02(InstructGPT) 또는 β≈0.01(Llama-2)에서 시작하라. reward hack이 보이면 ε나 LR을 건드리기 전에 β를 올려라.

---

## §1 TRPO — 구현하지 않는 조상

[[trpo]]는 *monotonic-improvement* bound를 증명한다.

```
η(π_new) ≥ L_{π_old}(π_new)  −  C · D_KL^{max}(π_old, π_new)
```

여기서 `L_{π_old}(π) = η(π_old) + E_{s∼ρ_{π_old}, a∼π}[ (π(a|s)/π_old(a|s)) · A_{π_old}(s,a) ]`는 `π_old` 주변에서 true return을 선형화한 것이고, `C`는 worst-case advantage bound를 접어 넣은 상수다. 두 항은 각각 "expected improvement"와 "distribution-shift penalty"다. TRPO는 penalty를 constraint로 바꾼다.

```
maximize_θ  L_{π_old}(π_θ)    subject to    E_s[ D_KL(π_old(·|s) ‖ π_θ(·|s)) ] ≤ δ
```

`δ ≈ 0.01`이다. Lagrangian은 Fisher-vector product에 대한 conjugate-gradient로 계산한 **natural-gradient step**으로 풀린다.

```
θ_new = θ_old + sqrt( 2δ / (g^T F^{-1} g) ) · F^{-1} g,      g = ∇L_{π_old}
```

그 뒤 KL constraint와 실제 surrogate improvement를 모두 강제하기 위해 geometric line-search를 수행한다. 이것이 증명 가능한 monotonic improvement를 준다.

**왜 아무도 LLM-RL에 TRPO를 쓰지 않는가.** CG iteration당 Fisher-vector product 두 번 × CG iteration 10번 = update당 약 20번의 추가 backward pass다. 70B 규모에서는 금지적으로 비싸다. 10^10개 parameter 위의 Fisher matrix는 ill-conditioned이고, natural-gradient 방향은 ratio가 폭발하는 rare tail token에 좌우된다. PPO는 surrogate `L_{π_old}(π_θ)`는 유지하되 KL constraint를 **clipping trick**으로 바꿔, 어떤 2차 구조 없이도 trust region을 *token별로* 강제한다.

---

## §2 PPO — clipped surrogate를 항별로 유도하기

확률 ratio를 정의하자.

```
r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t).
```

각 epoch 시작에서 `θ = θ_old`이므로 모든 곳에서 `r_t = 1`이다. unclipped surrogate는:

```
L^{CPI}(θ) = E_t[ r_t(θ) · Â_t ]
```

("CPI" = conservative policy iteration, [[ppo]]). `L^{CPI}`를 직접 maximize하는 것은 TRPO가 KL constraint로 bound하는 대상이다. PPO는 constraint를 두 항에 대한 *pessimistic min*으로 바꾼다.

```
L^{CLIP}(θ) = E_t[ min( r_t(θ) · Â_t ,   clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]
```

**항별 해석.** `clip(r, 1−ε, 1+ε)` 항은 trust region `[1−ε, 1+ε]` 밖에서 ratio를 포화시킨다. 그 창 밖에서 θ에 대한 gradient는 0이다. `min`은 clipped와 unclipped를 비관적으로 결합한다. 둘 중 낮은 값을 취하므로 objective가 true improvement의 *lower-bound*가 된다. 경우를 나누면:

- `Â_t > 0`이고 `r_t < 1+ε`: `r_t · Â_t`가 더 작은 항이다. gradient는 `r_t`를 올린다. 좋은 action의 likelihood를 높이므로 좋다.
- `Â_t > 0`이고 `r_t ≥ 1+ε`: `clip(r_t) · Â_t = (1+ε)·Â_t`가 더 작은(같은) 항이며, gradient는 0이다. 더 밀어붙이기를 거부하고 trust region에 머문다.
- `Â_t < 0`이고 `r_t > 1−ε`: `r_t · Â_t`가 더 작은(더 negative) 항이다. gradient는 `r_t`를 낮춘다. 나쁜 action의 likelihood를 낮추므로 좋다.
- `Â_t < 0`이고 `r_t ≤ 1−ε`: `clip(r_t)·Â_t = (1−ε)·Â_t`가 더 작은 항이며, gradient는 0이다. 더 밀어붙이기를 거부한다.

*Pessimism asymmetry*는 의도적이다. unclipped 항이 clip이 허용하는 것보다 surrogate를 더 개선하려 할 때만 update를 거부한다. ratio가 이미 우리에게 *불리하게* 폭발한 경우에는 unclipped 항이 여전히 기여하고, 그 gradient를 유지한다. 이것이 [[verl-ppo-loss]](Ye 2020)의 dual-clip trick이 나중에 patch하는 지점이다. PPO-clip은 ratio가 큰 negative-advantage token의 loss에 floor를 두지 않으며, 긴 LLM rollout에서는 이 때문에 폭발할 수 있다.

결합 objective는 value loss와 entropy bonus를 더한다.

```
L^{CLIP+VF+S}(θ) = E_t[ L^{CLIP}(θ)  −  c_1 · L^{VF}(θ)  +  c_2 · S[π_θ](s_t) ]
```

원 논문([[ppo]])에서는 `c_1 = 1.0`, `c_2 = 0.01`이다. `L^{VF} = (V_θ(s_t) − V_target)^2`는 return에 대한 MSE이고, `S = −Σ_a π log π`는 entropy bonus다. RLHF에서는 보통 entropy term을 *버린다*([[lilianweng-rlhf]], [[rlhf-instructgpt]]). reward에 추가된 `−β·KL(π‖π_ref)`가 이미 policy를 SFT distribution 쪽으로 regularize하기 때문이다.

---

## §3 GAE — PPO를 작동하게 하는 advantage

`L^{CLIP}`의 advantage `Â_t`는 *generalized advantage estimation*(Schulman 2016)이다. TD residual을 정의하자.

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t).
```

GAE는 미래 TD residual의 exponentially weighted sum이다.

```
Â_t^{GAE(γ, λ)} = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}
               = δ_t + (γλ)·δ_{t+1} + (γλ)^2·δ_{t+2} + …
```

`λ` knob은 bias-variance를 보간한다. `λ=0`은 `Â_t = δ_t`(pure TD, low variance, value-function error로 biased)를 주고, `λ=1`은 `Â_t = Σ_k γ^k r_{t+k} − V(s_t)`(pure Monte-Carlo return minus baseline, unbiased but high variance)를 준다. LLM-RL([[lilianweng-rlhf]])에서는 `γ = 1.0`(reward가 EOS에 집중된다. discounting은 긴 completion에서 signal을 줄인다)과 `λ = 0.95`(Schulman default, 분산을 tractable하게 유지)를 쓴다.

**Value-head ancestry.** `V(s_t)`는 policy trunk 위의 scalar head이며, `L^{VF} = (V_θ(s_t) − R_t)^2`로 학습된다. 여기서 `R_t = Â_t + V_{old}(s_t)`는 GAE return target이다. value head가 PPO를 actor-critic으로 만든다. 이것이 없으면 advantage는 Monte-Carlo return에서 constant baseline을 뺀 것(REINFORCE)으로 줄고, 긴 시퀀스에서는 분산이 training을 죽인다. [[trl-ppo]]와 [[verl-ppo-loss]]는 모두 `V_old` 주변의 value-loss clip을 구현해 policy-ratio clip을 mirror한다. 이는 Costa-Huang trick([[costa-huang-ppo-details]], item 4)이다.

**[figures/ppo-clip.html](figures/ppo-clip.html)**를 보라. `Â`의 부호 양쪽에 대해 `r ∈ [0, 2]` 전반의 `L^{CLIP}`와 gradient를 보여 주는 interactive ratio scrubber와, β-vs-reward KL-budget sweep이 있다.

---

## §4 InstructGPT — 하나의 방정식으로 보는 PPO-ptx

[[rlhf-instructgpt]] Equation 2, 원문 그대로:

```
objective(φ) = E_{(x,y)~D_RL}[ r_φ(x,y)  −  β · log( π_φ^{RL}(y|x) / π^{SFT}(y|x) ) ]
             + γ · E_{x~D_pretrain}[ log π_φ^{RL}(x) ]
```

세 항, 세 역할:

1. **`r_φ(x,y)`** — Bradley-Terry RM score. Scalar이며 end-of-sequence에 적용된다.
2. **`−β · log(π^{RL}/π^{SFT})`** — token별 KL penalty. [[kl-control-rlhf]]의 *KL-control* 항이다. 구현상 loss가 아니라 token별 reward에 **더한다**. `t < |y|`에 대해 `r̂_t = −β·(log π^{RL}(y_t|…) − log π^{SFT}(y_t|…))`이고, EOS에서 `r̂_{|y|} = r̂_{|y|} + r_φ(x,y)`다. KL을 reward에 더하면 token별 GAE advantage가 잘 정의된다([[trl-ppo]] §What to notice, [[verl-ppo-loss]] §Context). loss에 더하면 advantage-based policy gradient가 깨지고 경험적으로 더 나쁘게 학습된다.
3. **`γ · E_{D_pretrain}[log π_φ^{RL}(x)]`** — "ptx" mix. 매 update에 섞는 pretraining cross-entropy 항으로, alignment tax를 막는다. `γ = 0`이면 KL-penalized reward에 대한 plain PPO("InstructGPT")이고, `γ > 0`이면 PPO-ptx다.

**표준 hyperparameters** ([[rlhf-instructgpt]]):

| Knob | InstructGPT value |
|------|-------------------|
| PPO LR | 1.41e-5 (fixed) |
| PPO batch size | 512 prompts |
| PPO rollout length | ≤ 2048 tokens |
| KL coef β | 0.02 (adaptive controller optional) |
| Pretraining coef γ | 27.8 (PPO-ptx) or 0 (InstructGPT) |
| Clip ε | 0.2 |
| Epochs per rollout K | 4 |

명시적 entropy bonus는 없다. Entropy collapse는 adaptive-KL controller([[kl-control-rlhf]])가 추적하는 *failure signal*이다. 관측 KL이 target보다 올라가면 β를 multiplicatively 올리고, target보다 낮으면 내린다. Korbak의 Bayesian-inference 관점([[kl-control-rlhf]])은 β가 왜 자연스러운 "temperature" 해석을 갖는지 설명한다. 고정 β에서 objective의 closed-form optimum은 `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`이며, 이는 정확한 tilted-posterior sampling이다.

---

## §5 Llama-2 PPO — dual RM과 보수적 LR

[[llama-2]]는 다섯 번의 RLHF iteration(V1–V5)을 실행한다. V1–V3는 Rejection-Sampling Fine-Tuning뿐이고, PPO는 RSFT checkpoint 위에 V4/V5에서 추가된다. appendix에 인용된 hyperparameters(70B policy):

| Knob | Llama-2 value |
|------|---------------|
| PPO LR | 1e-6 (policy) |
| KL coef β | 0.01 |
| Batch size | 512 |
| Sequence length | 4K |
| Value function | standard PPO with clipped ratio + GAE |

**두 reward model.** InstructGPT의 단일 Bradley-Terry RM 대신, Llama-2는 **Helpfulness RM**과 **Safety RM**을 따로 학습한다. PPO scoring 때는 rule이 각 prompt를 어떤 RM(또는 weighted combination)으로 score할지 고른다. safety 관련 prompt는 Safety RM(또는 max-safety piecewise)을, helpfulness 관련 prompt는 Helpfulness RM을 쓴다. 이는 단일 RM이 scalar output에 강제로 밀어 넣는 helpfulness-vs-safety tradeoff를 해소한다.

**왜 보수적 LR인가.** 1e-6은 InstructGPT의 1.41e-5보다 약 14배 작다. 70B 규모에서 iterative RLHF(매주 새로운 preference batch)를 수행할 때 더 큰 LR은 iteration마다 policy를 너무 멀리 밀고, dual-RM rule을 불안정하게 만든다. 하나의 minibatch 안에서 helpfulness gain이 safety RM에 의해 되돌려질 수 있다. Llama-2 recipe는 보수적 LR, 낮은 β, 많은 iteration이다. β=0.01(InstructGPT의 0.02 절반)은 iterative schedule에서 핵심이다. iteration 사이에 policy가 움직일 자유를 주고, 더 많은 iteration으로 보상한다.

**Margin-weighted RM loss.** Annotator는 *margin*("significantly better / better / slightly better / negligibly better")을 label하고, RM loss는 large-margin pair에 더 큰 가중치를 준다.

```
L_RM = −E[ log σ( r(x, y_w) − r(x, y_l) − m(label) ) ]
```

여기서 `m`은 margin별 scalar다. 이는 PPO 자체의 변화는 아니지만, PPO가 최적화하는 *reward surface*를 형성한다. 그리고 Llama-2의 dual-RM system이 β=0.01에서도 안정적으로 학습되는 이유의 일부다.

---

## §6 Costa-Huang — LLM-RL에서 중요한 세부사항

[[costa-huang-ppo-details]]는 OpenAI Baselines → Stable-Baselines의 37가지 trick을 목록화한다. 대부분은 MuJoCo/Atari에 중요하지 LLM에는 덜 중요하다. RLHF에서 실제 training outcome을 바꾸는 subset은 다음과 같다(2024 follow-up과 [[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]]에서 cross-validated).

1. **Advantage normalization (whitening).** minibatch별로 mean을 빼고 std로 나눈다. reward magnitude가 prompt마다 drift할 때 `L^{CLIP}`의 scale을 안정적으로 유지한다. 세 framework가 모두 수행한다.
2. **Value-loss clipping.** `V_clipped = clamp(V_new, V_old − c, V_old + c)`; `L^{VF} = max((V_new − R)^2, (V_clipped − R)^2)`. policy ratio clip을 mirror한다. value head가 K epoch 동안 단일 rollout에 overfit하는 것을 막는다. [[trl-ppo]] lines 820–840은 이를 그대로 구현한다.
3. **Ratio clip bounds.** ε = 0.2 symmetric이 표준 기본값이다. 현대 DAPO/OpenReasonerZero는 **asymmetric** `ε_low = 0.2`, `ε_high = 0.28`을 쓴다([[verl-ppo-loss]], [[openrlhf-ppo]]). 위쪽을 덜 공격적으로 clip해 rare positive-advantage token이 더 빨리 upweight되게 한다. reward gain 없이 entropy가 collapse될 때 가장 먼저 시도할 것은 asymmetric clipping이다.
4. **Length normalization of the policy loss.** `loss_agg_mode = "token-mean"`은 모든 response token을 평균한다. `"seq-mean-token-sum"`(Dr.GRPO)은 시퀀스별로 합한 뒤 시퀀스에 대해 평균한다. 선택은 long-tailed completion-length distribution에서 gradient를 실제로 바꾼다. Length normalization은 짧은 completion이 덜 지배하게 한다.
5. **Global gradient clipping.** policy에서 gradient norm을 1.0으로 clip한다(MuJoCo PPO에서 쓰는 0.5보다 높다). 협상 불가다. `Â_t = 10`, `r_t = 5`인 rare token 하나가 아니면 run을 NaN으로 만들 수 있다.
6. **KL-to-reference in the reward, not the loss.** 표준 구현이다. [[trl-ppo]]의 `non_score_reward`, [[verl-ppo-loss]]의 k3 estimator를 쓰는 external `kl_penalty`. KL을 loss에 더하면 GAE가 깨진다.
7. **Value head initialization from the RM's value head.** RM은 이미 scalar preference를 예측하도록 학습되어 있다. 그 value-head weight는 random init보다 따뜻한 시작점을 주며, 없으면 rollout budget의 10–30%를 태우는 초기 value-loss spike를 피한다.

LLM-RL critical path가 아닌 것: observation normalization(observation이 없음), orthogonal weight init(pretraining init이 지배), LR annealing(RLHF LR은 이미 작다. cosine decay가 보통이지만 부차적이다).

---

## §7 Framework-level 그림

[[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]]는 모두 같은 loss를 구현한다.

```
pg_losses1 = −advantages · ratio
pg_losses2 = −advantages · clip(ratio, 1−ε_low, 1+ε_high)
pg_loss    = mean( max(pg_losses1, pg_losses2) )     # max because losses are negated
```

그리고 세 framework 모두 reward-shaping step을 통해 loss 밖에서 KL을 더한다.

```
r̂_t = r_t · 1[t = |y|]   −   β · (log π(y_t|…) − log π_ref(y_t|…))
```

차이점은 다음과 같다. TRL은 symmetric `cliprange`와 adaptive KL controller(InstructGPT style)를 쓴다. OpenRLHF는 `clip_eps_low/high`, dual-clip, GSPO sequence-level ratios, vLLM importance-sampling correction을 노출한다. verl은 policy-loss variant(`vanilla`, `gspo`, etc.)를 등록하고 loss-aggregation mode를 꽂는다. clipped-surrogate algebra는 셋 모두 동일하다. ecosystem은 ablation surface다.

---

## §8 Acceptance — 이 장 이후 할 수 있어야 하는 것

1. `Â_t`의 부호와 clip window 대비 `r_t`의 위치에 대한 casework로 `L^{CPI}`에서 `L^{CLIP}`를 유도하라. `min(·, clip)`이 왜 pessimistic인지 설명하라.
2. GAE expansion `Â_t = Σ (γλ)^k δ_{t+k}`를 쓰고 `λ=0`(TD)과 `λ=1`(Monte-Carlo) 한계를 말하라.
3. InstructGPT Equation 2를 인용하고 각 항(RM score, token별 KL penalty, ptx mix)을 label하라. β가 loss가 아니라 reward에 적용되는 이유를 말하라.
4. Llama-2의 PPO appendix hyperparameters(LR 1e-6, β=0.01, batch 512, seq 4K)를 인용하라. LR과 β가 InstructGPT보다 작은 이유를 설명하라.
5. LLM-RL transfer에서도 살아남는 Costa-Huang trick 5–7개를 이름 붙이고 각각을 정당화하라.

---

## Connections

- **ch-37 (Policy-Gradient Foundations)** — REINFORCE, score-function estimator, baseline subtraction. ch-38은 value-function baseline + trust region이 들어오는 지점에서 이어진다.
- **ch-39 (Offline Preference Optimization)** — DPO는 online KL penalty를 closed-form implicit reward `r_θ = β·log(π_θ/π_ref)`로 대체한다. Korbak의 tilted-posterior 관점([[kl-control-rlhf]])은 DPO와 PPO-with-KL이 같은 `π*`를 target한다는 것을 보여 준다.
- **ch-40+ (Critic-Free RL)** — GRPO, RLOO, REINFORCE++는 value head를 버리고 group-mean baseline을 쓴다. `L^{CLIP}` surrogate는 살아남지만 `Â_t`가 다르게 계산된다.

## Further reading

- [[trpo]] — trust-region derivation, natural-gradient step, line search.
- [[ppo]] — clipped surrogate, GAE, canonical hparams.
- [[rlhf-instructgpt]] — PPO-ptx, β=0.02, γ=27.8, labeler protocol.
- [[kl-control-rlhf]] — KL-as-reward, k3 estimator, Korbak Bayesian reformulation.
- [[llama-2]] — dual RM, RSFT then PPO, LR 1e-6, β=0.01.
- [[costa-huang-ppo-details]] — 37 tricks + RLHF-specific follow-up.
- [[hf-rlhf-illustrated]], [[lilianweng-rlhf]] — tutorial framings.
- [[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]] — framework source as the ablation surface.

## Companion visualization

**[figures/ppo-clip.html](figures/ppo-clip.html)** — 두 패널짜리 interactive. Panel 1: `r ∈ [0, 2]`를 scrub하고 `Â` sign을 toggle한다. `L^{CLIP}`와 gradient를 plot한다. Panel 2: reward-hacking과 mode-collapse regime이 주석 처리된 KL-vs-reward Pareto front의 β-sweep.
