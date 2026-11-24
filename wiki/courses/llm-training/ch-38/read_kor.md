<!-- chapter: ch-38
     track: preference
     kind: content
     title: KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax
     deps: [ch-37]
     sources: [[ppo]], [[trpo]], [[rlhf-instructgpt]], [[hh-rlhf]], [[mitigating-alignment-tax-rlhf]], [[llama-2]], [[llama-2-recipe]], [[llama-2-ppo]], [[costa-huang-ppo-details]], [[n-implementation-details-rlhf-ppo]], [[kl-control-rlhf]], [[john-schulman-kl-tricks]], [[trl-ppo]], [[trl-ppo-recipe]], [[openrlhf-ppo]], [[reward-model-overoptimization]], [[gpt-4-technical-report]], [[lilianweng-rlhf]]
     figures: figures/ppo-clip.html
     revised: 2026-09 (generality revision)
-->

# 38장 — KL-Controlled RLHF: PPO, InstructGPT, Alignment Tax

> **핵심 통찰.** KL-controlled RLHF는 reward-model score와 SFT model에 대한 log-ratio의 token별 penalty로 이루어진 reward에서 PPO로 policy를 학습한다. Optimum은 `exp(r/β)`로 reweight한 SFT distribution이다([[kl-control-rlhf]], Korbak et al. Eq. 5–6). InstructGPT에서 pretraining term 없는 PPO는 175B few-shot SQuADv2 F1을 69.75(GPT-3)에서 51.95, FR→EN BLEU를 39.93에서 26.58로 낮췄다. `γ = 27.8`의 pretraining gradient(PPO-ptx)를 추가하면 69.93과 36.76이었고 저자는 이것이 “does not lead to large changes in labeler preference”라고 보고한다([[rlhf-instructgpt]] Table 14, §4.1). KL coefficient를 100배 올려도 이 regression은 없어지지 않았다(App. E.6). Anthropic의 13M–52B 연구에서 tax는 small model에 나타났지만 13B/52B RLHF model은 zero-shot NLP evaluation에서 높았다([[hh-rlhf]] §1.1, Figure 3). RLHF는 GPT-4 MMLU calibration도 낮췄다. Expected calibration error는 0.007에서 0.074가 됐다([[gpt-4-technical-report]] Figure 8).
>
> **지침.** PPO-based RLHF가 pretrained base에서 시작하고 goal이 broad capability라면 PPO update에 pretraining(또는 replay) log-likelihood term을 더하라. InstructGPT의 `γ ≥ 20`은 1.3B에서 SQuADv2와 DROP을 회복했지만 larger β는 그렇지 않았다([[rlhf-instructgpt]] App. E.6). Pretraining data가 없으면 pre-RLHF 및 post-RLHF checkpoint 사이 weight interpolation을 평가한다. OpenLLaMA-3B에서 alignment-forgetting Pareto front는 early stopping, L1/L2 regularization, LoRA, distillation, 0.05–0.2 KL penalty보다 위였다([[mitigating-alignment-tax-rlhf]] §4.1, App. C.2). β는 proxy reward가 아니라 human 또는 held-out judge에 맞춰 tune한다. InstructGPT Likert score는 β=0.01–0.02 부근에서 peak였고 0과 2에서 낮았다(App. E.7). Llama 2는 7B/13B에 0.01, 34B/70B에 0.005를 썼다([[llama-2]] §3.2.3). 모든 RLHF run에서 reward 옆에 held-out capability suite, borderline-prompt false-refusal rate, calibration metric을 보고한다. Published RLHF run에서 이들 각각은 reward가 오르는 동안 움직였다.

## 이 장이 general-purpose model에 중요한 이유

Pipeline 위치: pre-training → mid-training → SFT → **preference optimization(이 장)** → RL → evaluation. [[ch-41]]은 pairwise preference에서 reward model(RM)을 학습했고, [[ch-37]]은 policy-gradient estimator와 baseline을 도출했다. 이 장은 이를 결합한다. PPO는 RM에 대해 policy를 optimize하고 KL penalty는 policy를 SFT model 가까이에 둔다.

이 stage가 generality에 중요한 이유는 objective에 RM이 score하지 않는 capability term이 없기 때문이다. API-style instruction에서 학습한 RM은 reading comprehension이나 translation에 reward를 주지 않으므로 objective의 어떤 term도 policy가 그것을 잃는 일을 벌하지 않는다. InstructGPT는 이 loss를 **alignment tax**라 불렀다. Alignment procedure가 target하지 않은 task의 performance cost다([[rlhf-instructgpt]] §1). 이 장은 stage에 대한 세 course 질문에 답한다.

1. Breadth: RLHF는 held-out labeler에서 instruction following을 개선했고 저자는 code 및 non-English instruction으로 qualitative transfer를 보고한다(§7). [[hh-rlhf]]의 large model은 zero-shot NLP evaluation에서 gain을 냈다(§6).
2. Narrowing: public NLP task regression(§6), calibration 감소(§6.5), borderline prompt false refusal(§8), proxy reward over-optimization(§4, §9).
3. Measurement: base model 및 SFT model 모두에 score한 fixed public NLP suite, held-out labeler, cross-validated RM accuracy, false-refusal rate, calibration error. 알려진 measurement error는 Generalization lens에 있다.

## §1 PPO가 근사하는 trust-region 문제, TRPO

**정의.** Trust Region Policy Optimization(TRPO)은 old/new policy 간 KL divergence constraint 아래 expected return의 local surrogate를 최대화한다([[trpo]] §4).

**문제.** `π_old` sample에서 계산한 policy-gradient step은 `π_old` 근처에서만 정확하다. 너무 큰 step은 sampled surrogate가 증가해도 true return을 낮출 수 있다.

**메커니즘과 수식.** Schulman 등은 new policy return의 lower bound를 증명한다(arXiv:1502.05477v5, Theorem 1, Eq. 9).

```
η(π̃) ≥ L_π(π̃) − C · D_KL^max(π, π̃),     C = 4εγ / (1 − γ)²,     ε = max_{s,a} |A_π(s,a)|
```

- `η(π̃)`: new policy π̃의 expected discounted return.
- `L_π(π̃)`: old policy visitation `ρ_π`로 state를 가중한 surrogate `η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a)`.
- `D_KL^max`: state 전반 `KL(π(·|s) ‖ π̃(·|s))`의 maximum.
- γ: discount factor. `A_π`: old policy 아래 advantage.

Right-hand side 최대화는 non-decreasing η를 보장한다(Eq. 10). 저자는 theoretical C를 쓰면 “the step sizes would be very small”하다고 말하므로 TRPO는 대신 constraint를 쓴다(Eq. 11–12). Sampled state 평균 KL ≤ δ라는 조건에서 `L_θold(θ)`를 maximize한다. Conjugate gradient로 Fisher-vector product를 계산하고 line search를 이어 한다(§6, App. C).

**계산 예.** γ=0.99, ε=1이면 `C = 4 × 1 × 0.99 / 0.01² = 39,600`이다. KL 0.001은 bound에서 return 39.6 unit을 비용으로 내므로 surrogate gain이 39.6을 넘어야 그런 step이 η를 개선한다고 인증된다. LLM response에 쓰는 undiscounted setting(§3)의 γ=1에서는 C가 정의되지 않는다. 이것이 bound가 language model에는 step-size rule이 아니라 motivation으로 쓰이는 이유 중 하나다(**Interpretation**).

**근거와 비용.** TRPO는 모든 experiment에서 δ=0.01을 썼다(§8, Table 2). Conjugate-gradient iteration `k = 10`을 썼다. Naive implementation은 compute의 90% 초과를 Fisher-vector product에 쓰며, 저자는 data의 10%에서 이를 계산하여 줄인다(App. C).

**함의.** PPO는 surrogate L을 유지하고 second-order constrained step을 first-order clipped objective로 대체한다. [[costa-huang-ppo-details]](core detail 8)가 요약한 Engstrom 등은 other implementation detail을 고정하면 PPO clipped objective가 TRPO objective와 비슷하게 수행했다고 발견했다(control task이며 language model은 아님).

## §2 PPO clipped objective, term별 분석

**정의.** Proximal Policy Optimization(PPO)은 sample batch마다 여러 minibatch epoch 동안 clipped surrogate objective를 최대화한다([[ppo]] §3).

**문제.** `L^CPI(θ) = Ê_t[r_t(θ) Â_t]`를 직접 최대화하면 update가 과도하게 커진다(§3). PPO에는 probability ratio를 1에서 멀리 움직일 incentive를 제거하는 first-order objective가 필요하다.

**수식**(arXiv:1707.06347v2 Eq. 6–7):

```
r_t(θ) = π_θ(a_t | s_t) / π_θold(a_t | s_t)
L^CLIP(θ) = Ê_t[ min( r_t(θ) Â_t ,  clip(r_t(θ), 1 − ε, 1 + ε) Â_t ) ]
```

- `r_t(θ)`: current policy와 rollout policy 아래 action a_t(LLM에서는 token t)의 probability ratio; `r_t(θ_old) = 1`.
- `Â_t`: timestep t의 advantage estimate. ε: clip range이며 논문은 0.2를 제안.
- `Ê_t`: batch timestep 전반 empirical mean.

**메커니즘, case별.**
1. `Â_t > 0`, `r_t < 1 + ε`: unclipped term이 더 작으므로 gradient가 `π_θ(a_t)`를 높인다.
2. `Â_t > 0`, `r_t ≥ 1 + ε`: clipped term `(1 + ε)Â_t`가 더 작고 θ에서 constant이므로 gradient는 0이다.
3. `Â_t < 0`, `r_t > 1 − ε`: unclipped term이 더 작고(더 negative), gradient가 `π_θ(a_t)`를 낮춘다.
4. `Â_t < 0`, `r_t ≤ 1 − ε`: clipped term `(1 − ε)Â_t`가 더 작고 constant이므로 gradient는 0이다.

Min은 objective를 개선할 ratio change만 무시하며 objective를 악화하는 change는 유지한다(§3). 따라서 `L^CLIP`은 unclipped surrogate `L^CPI`의 pessimistic(lower) bound이고 `θ_old` 주변에서 first order까지 `L^CPI`와 같다(§3). True return 변화의 bound는 아니다.

**계산 예**(ε=0.2). `Â=+1, r=1.3`이면 `min(1.3, 1.2)=1.2`, gradient는 0이다. `Â=−1, r=0.7`이면 `min(−0.7, −0.8)=−0.8`, gradient는 0이다. `Â=−1, r=1.5`이면 `min(−1.5, −1.2)=−1.5`, gradient는 nonzero다. 세 번째 case에서 objective는 r가 커질수록 bound 없이 linear하게 계속 감소한다. OpenRLHF의 optional `dual_clip`은 `Â < 0`일 때 이 loss를 `c·|Â|`로 cap한다([[openrlhf-ppo]], loss.py L143–148).

**근거.** MuJoCo task 7개 × seed 3개에서 average normalized score는 ε=0.2에서 0.82, ε=0.1에서 0.76, ε=0.3에서 0.70이다. Clipping/penalty 없음은 −0.39, best adaptive-KL variant(`d_targ=0.01`)는 0.74, best fixed KL penalty(β=3)는 0.72였다([[ppo]] Table 1). **Result(단일 연구)**이며 control task만 해당한다.

**Adaptive KL-penalty variant**(Eq. 8): `Ê_t[r_t Â_t − β KL[π_θold, π_θ]]`를 최대화한다. Update마다 `d = Ê_t[KL] < d_targ/1.5`면 `β ← β/2`, `d > 1.5 d_targ`면 `β ← 2β`다([[ppo]] §4). 이 controller는 previous policy의 KL에 작동한다. §4의 RLHF penalty는 SFT model의 KL에 작동한다.

**Combined objective**(Eq. 9): `L^{CLIP+VF+S} = Ê_t[L^CLIP − c_1 (V_θ(s_t) − V_t^targ)² + c_2 S[π_θ](s_t)]`이며 S는 entropy, c_1/c_2는 coefficient다. Atari는 `c_1 = 1`, `c_2 = 0.01`을 썼다. MuJoCo run은 policy/value function 간 parameter를 공유하지 않았고 entropy bonus도 쓰지 않았다([[ppo]] Table 5, §6.1).

**조건.** 논문에는 language-model experiment가 없다. Batch당 epoch(MuJoCo 10, Atari 3)는 LLM에 tune하지 않았고 InstructGPT는 inner epoch 하나를 썼다(§5).

**Interactive figure.** [figures/ppo-clip.html](figures/ppo-clip.html)의 panel 1은 Â sign별 `L^CLIP` term 하나와 r에 대한 gradient를 plot하므로, 모든 ε에서 위 네 case를 확인할 수 있다.

## §3 Language model의 advantage: token별 reward, GAE, value model

**정의.** RLHF에서 state `s_t`는 prompt와 지금까지 생성한 token이고 action `a_t`는 next token이다. Response가 끝나면 episode가 끝난다. InstructGPT는 이를 bandit environment라 부른다. RM이 prompt-response pair당 reward 하나를 내고 episode가 끝난다([[rlhf-instructgpt]] §3.5).

**문제.** RM은 response 전체에 scalar 하나를 주지만 PPO는 모든 token의 advantage가 필요하다.

**메커니즘.**
1. Token별 reward vector를 만든다. 모든 response token은 token별 KL estimate의 −β배를 받고 RM score는 final position에 더한다.
2. 각 position부터 return을 예측하도록 value model V를 실행한다.
3. End부터 거꾸로 generalized advantage estimation(GAE)을 계산한다.
4. Value target을 `R_t = Â_t + V(s_t)`로 둔다.

**수식**([[ppo]] Eq. 11–12):

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
Â_t = δ_t + (γλ) δ_{t+1} + … + (γλ)^{T−t−1} δ_{T−1}
```

- `r_t`: token별 reward. V: value model; last token 뒤의 V는 0.
- γ: discount. InstructGPT는 discount를 적용하지 않는다(γ=1)(App. C.4). λ: GAE parameter.
- T: response length. λ=1, γ=1이면 `Â_t`는 remaining reward 합에서 `V(s_t)`를 뺀 값.

Commit a08e713의 TRL implementation([[trl-ppo]], ppo_trainer.py L775–799):

```python
logr = ref_logprobs - logprobs
kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
non_score_reward = -args.kl_coef * kl
rewards = non_score_reward.clone()
...
rewards[actual_start, actual_end] += scores
...
delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
lastgaelam = delta + args.gamma * args.lam * lastgaelam
...
returns = advantages + values
advantages = masked_whiten(advantages, ~padding_mask)
```

**계산 예.** Token 3개 response, β=0.05, RM score=1.0, γ=1, λ=0.95.

| t | log π_θ − log π_ref(k1) | KL reward −β·k1 | RM score | r_t | V(s_t) | δ_t | Â_t |
|---|---|---|---|---|---|---|---|
| 1 | +0.4 | −0.02 | 0 | −0.02 | 0.5 | −0.02 + 0.6 − 0.5 = 0.08 | 0.08 + 0.95 × 0.3905 = 0.4510 |
| 2 | −0.2 | +0.01 | 0 | +0.01 | 0.6 | 0.01 + 0.8 − 0.6 = 0.21 | 0.21 + 0.95 × 0.19 = 0.3905 |
| 3 | +0.2 | −0.01 | 1.0 | 0.99 | 0.8 | 0.99 + 0 − 0.8 = 0.19 | 0.19 |

λ=1이면 `Â_1 = 0.08 + 0.21 + 0.19 = 0.48`이며 reward sum(0.98) − `V(s_1)`(0.5)와 같다. Token 2는 negative k1 값이라 positive KL reward를 받는다. Token별 k1 term은 KL value가 아니며 expectation만 그렇다(§4). [figures/ppo-clip.html](figures/ppo-clip.html)의 panel 2는 모든 input에 대해 이 table을 재계산한다.

**실제 value model.**
- InstructGPT는 모든 size policy에 6B RM에서 초기화한 별도 6B value function을 썼다. Value learning rate는 1.3B/6B policy에 9e-6, 175B policy에 5e-6이다([[rlhf-instructgpt]] App. C.4). 175B RM은 training이 덜 stable하여 “which made them less suitable for use as initializations for the PPO value functions”였기 때문에 일부로 사용하지 않았다(App. C.2).
- TRL PPO trainer는 scalar head가 있는 별도 value model을 쓰며 example script는 RM에서 이를 초기화한다([[trl-ppo]]; [[trl-ppo-recipe]]).
- Huang 등은 value network를 RM에서 초기화한다. 이 warm start가 “can greatly improve initial gradients to the policy and reduce drift / alignment tax over training”이라고 쓰고 Noukhovitch et al. 2023을 cite하지만 자체 ablation은 보고하지 않는다([[n-implementation-details-rlhf-ppo]] Detail 22).

**Value-loss clipping.** OpenAI ppo2는 previous value 주변의 value prediction을 clip한다: `L^V = max[(V_θ − V_targ)², (clip(V_θ, V_old − ε, V_old + ε) − V_targ)²]`. 이는 [[costa-huang-ppo-details]] core detail 9다. 같은 post는 Engstrom 등이 도움이 된다는 evidence를 찾지 못했고 Andrychowicz 등이 해로울 수 있다고 제시했다(control task)고 보고한다. TRL은 0.5 factor 및 `cliprange_value`=0.2로 구현한다([[trl-ppo]] L831–839).

**함의.** λ<1인 GAE는 V를 신뢰하여 variance를 낮춘다. Initialization처럼 V가 틀리면 bias가 모든 token advantage에 들어간다. Value initialization과 warm-up이 반복되는 detail인 이유다(§9).

## §4 KL penalty: estimator, placement, target distribution

**정의.** KL-regularized RL objective는 `J(θ) = E_{x∼π_θ}[r(x)] − β D_KL(π_θ, π_0)`다. π_0는 initial(SFT) model이고 β≥0은 trade-off를 정한다([[kl-control-rlhf]], Korbak et al. arXiv:2205.11275v2 Eq. 3).

**문제.** RM은 학습한 proxy다. Constraint가 없으면 PPO는 RM에서 high score, human에서 low score인 output을 찾을 수 있다. Llama 2는 penalty가 “is useful for training stability, and to reduce reward hacking whereby we would achieve high scores from the reward model but low scores from human evaluation”라고 보고한다([[llama-2]] §3.2.3).

**Target distribution.** Korbak 등은 objective가 closed-form optimum을 가지며 그 optimum에 대한 divergence로 쓸 수 있음을 보인다(Eq. 5–7).

```
π*(x) = (1/Z) π_0(x) exp(r(x)/β),      J(θ) ∝ −D_KL(π_θ, π*)
```

- x: full response; Z: normalizing constant; β는 reward의 temperature로 작동한다.

**계산 예.** `π_0 = (0.5, 0.5)`, `r = (1, 0)`인 response 두 개를 생각하자. β=0.5이면 `π*(response 1) = e² / (e² + 1) = 0.881`이다. β=2이면 `π* = e^0.5 / (e^0.5 + 1) = 0.622`다. 작은 β는 higher-reward response로 mass를 더 옮긴다. β→0이면 모든 mass가 reward maximum으로 가고 β→∞이면 policy는 π_0에 남는다.

**Placement와 estimator.** InstructGPT Eq. 2는 sample response마다 reward에서 `β log(π^RL(y|x)/π^SFT(y|x))`를 뺀다([[rlhf-instructgpt]] §3.5). Token 합에서는 sampled log-ratio이며 Schulman notation에서 k1이다. Sample `x ∼ q`, ratio `r = p(x)/q(x)`일 때 Schulman은 다음을 정의한다([[john-schulman-kl-tricks]]).

| KL[q,p] estimator | Formula | Bias | Sign | Std/true KL, q=N(0,1), p=N(0.1,1), true KL 0.005 | Std/true KL, p=N(1,1), true KL 0.5 |
|---|---|---|---|---|---|
| k1 | −log r | unbiased | negative 가능 | 20 | 2 |
| k2 | ½(log r)² | biased(true KL의 0.2%, 25%) | ≥0 | 1.42 | 1.73 |
| k3 | (r−1)−log r | unbiased | ≥0 | 1.42 | 1.7 |

RLHF에서 q는 policy π_θ, p는 π_ref이므로 k1은 token별 `log π_θ − log π_ref`다. TRL default `kl_estimator`는 “k1”이고 “k3”가 option이다([[trl-ppo-recipe]]). OpenRLHF는 default로 reward에 KL을 넣으며 reward-side KL에는 k1, loss-side KL에는 k2/k3을 권장한다고 출력한다([[openrlhf-ppo]], train_ppo_ray.py L675–680). Loss 안 k3 estimator는 [[ch-40]], [[ch-43]]이 다루는 GRPO convention이다.

**Fixed 또는 adaptive β.** InstructGPT는 fixed β=0.02를 썼다(App. C.4). Ziegler et al. 2019를 cite하는 OpenRLHF `AdaptiveKLController`는 `β ← β(1 + clip(KL/target − 1, −0.2, 0.2) · n_steps/horizon)`로 β를 update한다. KL target을 설정했을 때만 사용하고 default는 fixed coefficient 0.01이다([[openrlhf-ppo]], kl_controller.py L4–19). Controller는 entropy가 아니라 measured KL에 작동한다.

**β 선택 근거.**
- InstructGPT의 pretraining mix 조건 human Likert score는 “Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02”라고 한다([[rlhf-instructgpt]] App. E.7, Figure 36; E.7의 model size 미기재).
- Anthropic은 `λ_KL=0.001`을 썼고 “likely has a very minor impact during most of RL training (as D_KL < 100 typically), and might actually be wholly unnecessary”라고 썼다([[hh-rlhf]] §4.1). RLHF training 중 preference-model score와 `√D_KL(π‖π_0)` 사이에 대략 linear relation도 관측했다(§4.3).
- 1.2B policy의 synthetic gold-RM 연구에서는 nonzero KL penalty가 early stopping처럼 행동하고 gold-score-versus-KL frontier를 높이지 못했다. 저자는 result가 hyperparameter-sensitive라고 설명한다([[reward-model-overoptimization]] §3.6).
- Huang 등의 1B TL;DR PPO run은 KL 약 50 및 85에 도달했다. RLHF reward는 높았지만 GPT-3.5는 reference summary보다 이를 20% 미만으로 선호했고 sample에는 space 없는 concatenated string이 있었다([[n-implementation-details-rlhf-ppo]] §7.1, β=0.05).

이 result는 방향에서 **Replicated**다([[reward-model-overoptimization]]; [[n-implementation-details-rlhf-ppo]]). KL penalty 하나만으로 모든 setting의 proxy over-optimization을 막지는 못한다. Best β가 RM scale, reward normalization, model size에 의존한다는 것은 **Interpretation**이다. Llama 2는 ablation 없이 size별 다른 β를 썼으며 이 장의 어느 source도 이 factor 전반 β를 sweep하지 않는다.

**함의.** KL penalty는 RL prompt의 distribution에서 policy를 π_SFT 가까이 둔다. 다른 input에서 π_SFT가 가진 ability는 target하지 않으므로 §6에는 별도 mechanism이 필요하다.

## §5 InstructGPT: recipe와 model name

**Naming.** InstructGPT는 trained model type 세 가지를 정의한다([[rlhf-instructgpt]] §3.5).
- **SFT**: labeler demonstration에서 fine-tune한 GPT-3.
- **PPO**: RM reward와 token별 KL penalty에서 PPO로 fine-tune한 SFT model이며 γ=0.
- **PPO-ptx**: pretraining gradient를 mix한 PPO(γ>0).

“Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models”(§3.5).

**Objective**(Eq. 2):

```
objective(φ) = E_{(x,y)∼D_{π_φ^RL}} [ r_θ(x,y) − β log( π_φ^RL(y|x) / π^SFT(y|x) ) ] + γ E_{x∼D_pretrain} [ log π_φ^RL(x) ]
```

- `π_φ^RL`: parameter φ의 RL policy. `π^SFT`: SFT model(KL reference이기도 함). `r_θ`: RM.
- `D_{π_φ^RL}`: RL prompt set의 prompt x 및 policy에서 sample한 response y. `D_pretrain`: GPT-3 pretraining distribution.
- β: KL reward coefficient. γ: pretraining loss coefficient이며 “PPO” model은 γ=0.

**논문이 말한 setting**(App. C.3, C.4, E.9, E.11):
1. RL policy는 demonstration에서 pretraining data 10%를 mix하여 2 epoch 학습한 SFT model에서 initialize한다. §3.5의 16-epoch SFT baseline과 다르다. Init model은 KL reference도 된다.
2. β=0.02. RL run은 unique prompt 약 31k를 포괄하는 episode 256k를 쓴다. Iteration당 batch 512를 minibatch 64개씩 8개로 나누며 inner epoch 하나다.
3. Constant learning rate이며 첫 iteration 10개에는 peak의 10분의 1에서 시작하는 warm-up을 둔다. Policy peak learning rate는 값 하나로 출력하지 않는다. 1.3B/6B에는 2.55e-6–2.55e-5 log-linear sweep을 실행했고 pretraining mix 없이 8.05e-6 초과 run은 모두 diverged했다. 175B에는 2.55e-6 및 3.74e-6을 시험했다. Final checkpoint는 highest Likert score checkpoint다(E.9).
4. Decay 0.992의 weight EMA; GAE에 discount 없음; clip ratio 0.2; rollout sampling temperature 1.
5. 6B RM과 6B value function; value LR은 1.3B/6B에서 9e-6, 175B에서 5e-6.
6. PPO-ptx: RL episode보다 8배 많은 pretraining example, PPO/pretraining gradient는 minibatch당 consecutive step에서 accumulate하고 pretraining gradient는 γ=27.8을 곱한다.
7. RL 전에 labeler demonstration score 평균이 0이 되게 bias로 RM을 shift한다(§3.5).

**도출 count.** `256,000 episodes / iteration당 512 = PPO iteration 500개`. `500 × minibatch 8개 = optimizer step 4,000개`. `8 × 256,000 = pretraining example 2,048,000개`다. “256k”가 256,000을 뜻한다고 가정한다(**도출**).

**Headline result.** 1.3B InstructGPT output은 175B GPT-3보다 선호됐다. 175B InstructGPT는 175B GPT-3보다 85 ± 3%, few-shot 175B GPT-3보다 71 ± 4%의 시간 동안 선호됐다(§4.1). “Adding updates on the pretraining mix during PPO does not lead to large changes in labeler preference”(§4.1). GPT-3 model에 API로 제출한 prompt에서는 PPO-ptx model이 larger model size에서 약간 더 나쁘게 수행했다(§4.1, Figure 3).

## §6 Alignment tax와 mitigation

### §6.1 InstructGPT가 측정한 것

**정의.** Alignment tax는 alignment procedure가 target하지 않은 task의 performance drop이다. InstructGPT는 GPT-3 대비 public NLP dataset regression으로 측정한다([[rlhf-instructgpt]] §1, §4.2).

**근거**(Table 14의 175B column; SQuADv2/DROP는 F1, HellaSwag은 accuracy, WMT15 FR→EN은 BLEU):

| Task, setting | GPT-3 | SFT | PPO(γ=0) | PPO-ptx(γ=27.8) | PPO−GPT | PPO-ptx−GPT |
|---|---|---|---|---|---|---|
| SQuADv2, few-shot | 69.75 | 65.90 | 51.95 | 69.93 | −17.80 | +0.18 |
| SQuADv2, zero-shot | 64.30 | 57.67 | 43.68 | 59.85 | −20.62 | −4.45 |
| DROP, few-shot | 35.27 | 35.85 | 27.78 | 33.34 | −7.49 | −1.93 |
| DROP, zero-shot | 27.53 | 15.79 | 13.08 | 15.23 | −14.45 | −12.30 |
| HellaSwag, few-shot | 0.791 | 0.741 | 0.759 | 0.820 | −0.032 | +0.029 |
| FR→EN, few-shot | 39.93 | 35.07 | 26.58 | 36.76 | −13.35 | −3.17 |
| FR→EN, zero-shot | 38.92 | 36.90 | 24.16 | 34.28 | −14.76 | −4.64 |

Table은 세 방식으로 읽는다.
1. PPO-ptx는 PPO regression 대부분을 회복하지만 전부는 아니다. 저자는 PPO-ptx가 “still lags behind GPT-3 on DROP, SQuADv2, and translation”이고 HellaSwag에서는 GPT-3를 넘는다고 한다(§4.2).
2. Drop 일부는 SFT부터 나타난다. Zero-shot DROP에서 SFT는 GPT-3보다 11.74점 낮고 PPO가 2.71점을 더한다. “RLHF tax” 크기는 baseline이 base model인지 SFT model인지에 의존한다.
3. 저자는 PPO regression이 많은 dataset에서 “particularly in the few-shot setting” 발생한다고 설명한다(App. E.1). 하지만 175B에서 SQuADv2/DROP/FR→EN zero-shot absolute drop은 few-shot보다 커 table의 모든 task에는 성립하지 않는다.

[figures/ppo-clip.html](figures/ppo-clip.html)의 panel 3은 setting 간 비교를 위해 이 값을 plot한다.

### §6.2 PPO-ptx: mechanism과 ablation

**메커니즘.**
1. PPO minibatch마다 sampled response의 PPO gradient를 계산한다.
2. Pretraining text를 sample하고 log-likelihood gradient를 계산한다.
3. Pretraining gradient에 γ를 곱하고 optimizer step 전 두 gradient를 accumulate한다(App. C.4).

Ptx term은 lost ability가 학습된 distribution의 positive log-likelihood anchor다. KL penalty는 RL prompt에서만 policy를 anchor한다.

**Ablation**(별도 언급 없으면 1.3B, App. E.6–E.11):
- γ≥20은 regression을 회복했다. γ 증가에 따라 validation reward는 떨어졌고 27.8 하나가 1.3B–175B에서 작동했다. Human Likert score는 정확한 γ에 insensitive였다(E.6).
- γ=0에서 β를 default의 100배인 2.0까지 올려도 DROP/SQuADv2 regression은 고치지 못하고 validation reward를 유의하게 낮췄다(E.6, Figure 34). 이 sweep의 KL reference는 SFT가 아니라 pretrained GPT model이다. **Result(단일 연구)**.
- Pretraining mix와 seed 3개로 episode 256k 대신 512k를 학습하면 DROP/SQuADv2는 GPT-3보다 약간 낮아졌다(E.6, Figure 35).
- Pretraining data ratio: 4×에서 pretraining log-probability loss는 흔히 training 중 증가했다. 32× preliminary run은 몇 배 training time에서 Likert score가 더 좋았다. 8×는 no-mix 대비 training time을 두 배로 했다(E.11).
- Pretraining mix PPO는 learning rate에 덜 sensitive했다(E.9).

**한계.** 저자는 ptx term이 “does not completely mitigate performance regressions, and may make certain undesirable behaviors more likely for some tasks (if these behaviors are present in the pretraining data)”라고 한다(§5.4).

### §6.3 Anthropic HH-RLHF 연구의 scale dependence

[[hh-rlhf]]는 preference model에 대해 PPO로 13M–52B의 model size 7개를 학습했다(§3.1, §4.1). Finding(§1.1, §4.6.1, Figure 3):
- “Smaller models experience severe 'alignment taxes'.” MMLU, Lambada, HellaSwag, OpenBookQA, ARC-Easy, ARC-Challenge, TriviaQA mean accuracy는 RLHF 뒤 하락했다.
- 13B/52B RLHF model은 zero-shot NLP evaluation에서 더 좋고 few-shot에서는 같았다. TriviaQA를 제외한 모든 경우에 12B/52B RLHF model은 base model보다 좋았다(§1.2).
- 저자의 measurement caveat: explicit choice를 둔 multiple-choice format은 “tends to improve performance for large models, while decreasing the performance of small models, leading to the arguably misleading appearance of a 'grok' curve”(§4.6.1).

InstructGPT(1.3B–175B, API prompt, KL β=.02)는 175B regression을, Anthropic(13M–52B, HH dialogue, λ_KL=.001)은 13B/52B gain을 발견했다. Data, evaluation format, baseline이 달라 scale effect는 replicated law가 아니라 **Open question**이다.

### §6.4 Mitigation으로서 model averaging

Lin 등은 SFT model θ_0 대비 forgetting으로 tax를 연구한다([[mitigating-alignment-tax-rlhf]], arXiv:2309.06256v4 §3–4).

```
θ_α = (1 − α) θ_0 + α θ,     α ∈ [0, 1]
```

- θ_0: RLHF 전 instruction-tuned model(ShareGPT의 OpenLLaMA-3B). θ: RLHF 뒤 model.
- α: interpolation weight. α=1은 RLHF model, α=0은 SFT model.

**Setting.** Main experiment는 HH-RLHF dataset의 rejection-sampling fine-tuning이고 PPO/DPO로도 확인하며 Mistral-7B까지 확장한다. ARC Easy/Challenge, RACE, PIQA(accuracy), SQuAD/DROP(F1), WMT14 FR→EN(BLEU)에서 tax를 측정한다(§3).

**Finding.**
- RLHF reward가 오를수록 translation/reading comprehension은 낮아지고 commonsense QA는 먼저 올랐다가 낮아졌다(§4, App. E.1).
- Early stopping, θ_0에 대한 L1/L2 regularization, LoRA, knowledge distillation, stochastic moving averaging, model averaging은 reward의 일부 비용으로 tax를 줄인다. Model averaging Pareto front는 “supersedes nearly all other methods across various hyper-parameters”(§4.1, Figure 3).
- PPO KL penalty 0.05/0.1/0.2는 forgetting을 “partially mitigate”했지만 model averaging이 alignment-forgetting trade-off에서 “much more effective”했다(App. C.2, Figure 8).
- RLHF data의 4배까지 pretraining replay(비교는 token 400M)는 reading comprehension에서만 model averaging을 이기고 commonsense QA/translation에서는 낮았다. 저자는 replay가 token 1.2T pretraining set의 약 0.03%만 cover하기 때문이라 귀속한다(App. C.1; 같은 논문 §2는 약 0.01%를 제시).
- Transformer lower third만 평균하면 alignment reward와 NLP score 모두 개선했다. K=3 layer group별 ratio를 쓰는 Heterogeneous Model Averaging(HMA)은 front를 더 밀었다(§5–6).

**조건.** Main comparison은 numeric table 없이 figure에서 읽었다. Model은 3B/7B이며 KL 비교 reward scale은 InstructGPT와 다르다. Replay result는 1.3B–175B InstructGPT PPO-ptx와 방향이 충돌한다. Data access(GPT-3 own pretraining distribution 대 0.03% subset)와 replay ratio(8× 대 최대4×)가 다르다. Given scale에서 어느 mitigation이 나은지는 **Open question**이다. [[ch-30c]]가 일반 weight averaging을 다룬다.

### §6.5 RLHF 뒤 calibration: GPT-4

GPT-4 technical report는 “the pre-trained model is highly calibrated”, “after the post-training process, the calibration is reduced”라고 말한다([[gpt-4-technical-report]] §5). MMLU subset에서 expected calibration error(ECE)는 pre-trained model 0.007, plot label이 “model=ppo”인 post-trained model 0.074였다(Figure 8). Exam benchmark multiple-choice 부분에서 base model average는 73.7%, RLHF model은 74.0%였다(App. B, Table 8). **Result(단일 연구)**. Report는 calibration에 대한 SFT/RL effect를 분리하지 않는다.

**General-purpose model에의 함의.** 두 측정은 benchmark가 다르다. Exam multiple-choice accuracy는 거의 불변(73.7%→74.0%)인데 MMLU subset ECE는 0.007→0.074다. Accuracy만 나열한 capability report는 calibration 변화를 보이지 않는다. Calibration 및 abstention metric은 evaluation gate에 속한다([[ch-49]], [[ch-52]]).

## §7 InstructGPT가 보고한 generalization

**Held-out labeler.** Training data를 하나도 만들지 않은 별도 labeler set은 InstructGPT output을 training labeler와 비슷하게 rank했으며, 모든 InstructGPT model은 그 judgment에서 “still greatly outperform the GPT-3 baselines”했다([[rlhf-instructgpt]] §4.1, Figure 3). Inter-annotator agreement는 training labeler 사이 72.6 ± 1.5%, held-out labeler 사이 77.3 ± 1.3%다(§3.4). Held-out labeler는 같은 vendor 출신이지만 screening test는 치르지 않았다(§3.4).

**RM cross-validation.** Labeler를 group 5개로 나누어 4개에서 학습하면 RM은 held-out group preference를 69.6 ± 0.9% accuracy로 예측했으며 training group 안에서는 72.4 ± 0.4%다(§4.1). 저자는 model이 training labeler preference만 fit하는 것은 아니라고 결론낸다. 동시에 broader user group 및 human disagreement input에 더 많은 연구가 필요하다고 말한다(§1).

**Code와 non-English instruction.** Dataset은 English가 96% 초과다(§3.3). 175B PPO-ptx model은 GPT-3보다 다른 언어 instruction을 따르고 code question에 더 reliably 답하지만, “often produces an output in English even when the instruction is in another language”(§4.3). 이 result는 qualitative이며 “We do not track these behaviors quantitatively”(§4.3, Figure 8, non-cherry-picked output을 둔 cherry-picked prompt)다.

**Public NLP task collection은 다른 distribution이다.** FLAN 또는 T0++로 fine-tune한 GPT-3 175B는 API prompt에서 SFT baseline보다 약간 낮았다. SFT baseline 대비 InstructGPT win rate는 73.4 ± 2%, T0은 26.8 ± 2%, FLAN은 29.8 ± 2%다(§1, §4.1). Generalist에 두 결과가 따른다. Academic task collection 학습은 real user prompt에서 더 좋은 result를 만들지 못했다. 반대 방향으로 RLHF model은 academic task에서 regression했다(§6.1).

**Failure mode.** 175B PPO-ptx model은 false premise를 수용하고 simple question을 과도하게 hedge했다(§4.3, Figure 9).

**Output diversity.** Anthropic은 RL-finetuned model이 “typically have much narrower, lower-entropy output distributions”라고 보고한다([[hh-rlhf]] §4.6). Llama 2는 RLHF 뒤 factual prompt에서 Self-BLEU diversity 감소와 creative prompt에서 diversity 유지를 보고한다([[llama-2]] §5.1, Figure 21). RL generalization과 diversity의 trade-off는 [[ch-38a]]가 다룬다.

## §8 Llama 2 PPO: reward model 두 개, size별 KL, false refusal

**Schedule.** “Until RLHF (V4), we used only Rejection Sampling fine-tuning, and after that, we combined the two sequentially, applying PPO on top of the resulted Rejection Sampling checkpoint before sampling again”([[llama-2]] §3.2.3). Published PPO comparison은 PPO 유무의 RLHF-V5이며 PPO version은 Meta RM/GPT-4 judge 모두에서 ChatGPT 대비 더 높은 win rate를 보인다(Figure 11, plot에서 읽음). 이 문구는 V4 자체가 PPO를 썼는지 확립하지 않는다.

**Reward**(Eq. 4 및 piecewise rule, §3.2.3):

```
R_c(g|p) = R_s(g|p)  if is_safety(p) or R_s(g|p) < 0.15,   else R_h(g|p)
R̃_c(g|p) = whiten( logit( R_c(g|p) ) )
R(g|p) = R̃_c(g|p) − β D_KL( π_θ(g|p) ‖ π_0(g|p) )
```

- p: prompt, g: generation, `R_s`/`R_h`: (0,1)의 safety/helpfulness RM score, π_0: original policy.
- 0.15: unsafe response filtering threshold이며 Meta Safety test set precision 0.89, recall 0.55로 선택.
- whiten: zero mean/unit variance normalization. 저자는 final linear score를 whiten하는 것이 “important ... in order to increase stability and balance properly with the KL penalty term (β)”라고 발견.

**계산 예.** Safety-related가 아닌 prompt가 `R_s=0.10`, `R_h=0.90`을 받는다면 `R_s < 0.15`라 `R_c=0.10`, whitening 전 `logit(0.10)=ln(0.10/0.90)=−2.197`이다. 같은 response가 `R_s=0.20`이면 R_h를 쓰고 `logit(0.90)=+2.197`이다. Rule은 weighted sum이 아니라 hard switch다.

**Setting**([[llama-2-recipe]]; [[llama-2-ppo]]). AdamW: β1=.9, β2=.95, eps=1e-5, weight decay .1, gradient clipping 1.0, 모든 model에서 constant LR 1e-6. Batch 512, clip .2, minibatch 64, minibatch당 gradient step 하나, prompt당 generation 하나. β는 7B/13B .01, 34B/70B .005다. Held-out prompt에서 early stopping으로 200–400 iteration 학습했다. Report는 β/LR ablation, PPO maximum response length, sampling temperature를 주지 않는다. 70B PPO iteration 하나는 약 330초다.

**Helpfulness-safety trade-off로서 false refusal.** Llama 2는 false refusal을 irrelevant safety concern 때문에 legitimate prompt를 거절하는 일로 정의하고 refusal classifier로 측정한다([[llama-2]] §4.2.3). 이 measurement는 safety-data scaling ablation에서 왔다. Helpfulness data는 약 .9M sample로 고정하고 safety data는 약 .1M sample의 0%–100%다. PPO-only experiment가 아니다. Safety data가 늘어도 mean helpfulness RM score는 거의 일정했다(Figure 15). Helpfulness test set false refusal은 .006%(occurrence 1)에서 .05%(8)로 늘었다. Sensitive word를 포함하지만 safe한 prompt 210개 borderline set에서는 15%에서 27%로 늘었다(Figure 33). Model은 “‘bomb’처럼 unsafe generation에 자주 나타나는 word”가 있으면 prompt safety 구분에 때때로 어려움을 겪는다(§4.2.3).

**함의.** Aggregate helpfulness score는 변화를 보이지 않았지만 targeted borderline set은 12 percentage point 증가를 보였다. Over-refusal evaluation은 [[ch-52]]가 다룬다.

## §9 결과를 바꾸는 implementation detail

**PPO의 37개 detail에서**([[costa-huang-ppo-details]], ICLR Blog Track 2022). Post가 번호 붙인 core detail 13개는 (1) vectorized architecture, (2) orthogonal initialization, (3) Adam epsilon 1e-5, (4) Adam learning-rate annealing, (5) GAE, (6) shuffling을 둔 minibatch update, (7) minibatch level advantage normalization, (8) clipped surrogate objective, (9) value-function loss clipping, (10) overall loss 및 entropy bonus, (11) norm .5의 global gradient clipping, (12) debug variable, (13) shared/separate policy/value network다. Post의 evidence는 control/Atari task에서 왔다. Per-minibatch advantage normalization은 “not affect performance much”, value-loss clipping은 도움이 된다는 evidence가 없었고, global gradient clipping은 “a small performance boost”를 냈다. Post는 이를 language model에서 측정하지 않았다.

**PPO RLHF의 N+ implementation detail에서**([[n-implementation-details-rlhf-ppo]], arXiv:2403.17031v1; TL;DR summarization의 Pythia 1B, 2.8B, 6.9B, seed 4개):
- Detail 7: dropout을 disable한다. Dropout이 active면 token log-probability가 reproducible하지 않아 KL penalty가 unreliable해지고 first epoch PPO ratio가 1이 아니다.
- Detail 22: RM에서 value model을 initialize한다(§3).
- Detail 23: “EOS trick”. EOS token 없는 completion에는 constant −1 score를 준다. RM score가 EOS에서만 정의되기 때문이다. Length limit에 닿는 long completion도 벌한다. TRL은 이를 `missing_eos_penalty`로 expose한다([[trl-ppo]] L759–761).
- Detail 24: optional reward whitening. Ablation에서 더 짧은 completion과 더 낮은 preference rate를 냈고 length-controlled comparison은 유무가 비슷했다(§7.1, Figure 11–12).
- Detail 25: advantage whitening.
- Result: GPT-3.5는 best 6.9B PPO model summary를 reference summary보다 거의 80% 시간 선호했다. PPO response는 SFT response보다 길고 length control 뒤에도 모든 length ratio에서 PPO가 SFT보다 높았다(§7.1, Figure 12).

**Model report에서.** InstructGPT는 decay .992 weight EMA와 RM bias shift를 썼고 Llama 2는 reward logit을 whiten했다. 양쪽 모두 batch 512, minibatch 64다. InstructGPT는 1.3B에서 minibatch 32가 “slightly better than 64”였지만 GPU utilization 때문에 64를 유지했고, human evaluation에서 batch 512가 64–1024 중 최고였다([[rlhf-instructgpt]] App. E.11).

**Framework default는 논문 값이 아니다.** Commit a08e713 TRL `PPOConfig` default는 LR 3e-6, batch당 PPO epoch 4, k1의 `kl_coef` .05, cliprange .2, `vf_coef` .1, γ=1.0, λ=.95, temperature .7, response length 53 token이다([[trl-ppo-recipe]]). 이는 N+ TL;DR setup(Table 7)을 따르며 InstructGPT의 inner epoch 하나 및 β=.02와 다르다. “PPO learning rate 1.41e-5, epoch 4” 같은 값은 InstructGPT 논문에 없다.

## 음성 샘플과 음성 피드백

**어떤 의미인가.** PPO 기반 RLHF는 이 과정의 분류체계에서 의미 4, 즉 gradient로서 negative를 사용한다. `Â_t < 0`인 token은 확률이 낮아지는 방향으로 밀린다. RM 자체도 거절된 response를 negative gradient로 사용해 학습되었다([[ch-41]]). PPO-ptx의 pretraining term은 positive log-likelihood anchor이며 negative가 아니다.

**이 단계에서 negative는 어디서 오는가.**
1. Value baseline에 비해 RM score가 낮으면 그 response token에 negative advantage가 생긴다.
2. Token별 KL reward `−β·k1`은 `log π_θ > log π_ref`인 모든 곳에서 negative다.
3. Constant penalty: truncated completion에 주는 EOS trick의 `−1`([[n-implementation-details-rlhf-ppo]] Detail 23).
4. Llama 2에서는 safety-tagged prompt 또는 `R_s < 0.15`일 때 safety RM이 helpfulness RM을 대체한다. 따라서 safe한 prompt의 response도 safety score 때문에 낮아지는 방향으로 밀릴 수 있다. 이 mechanism은 §8의 false refusal과 일관되지만, false-refusal ablation은 PPO를 분리하지 않았다(**해석**).

이 장의 source는 이 label 가운데 어느 것에도 false-negative rate를 보고하지 않는다.

**메커니즘.** Logit `z`를 갖는 softmax policy에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이다. Token별 policy-gradient contribution은 `Â_t (1[j = y] − p_j)`이다. `Â_t < 0`이면 sampled token의 logit은 내려가고, 다른 모든 logit은 현재 확률 `p_j`에 비례해 올라간다. 계산 예: `p = (0.7, 0.2, 0.1)`, sampled token 3, `Â = −1`, logit에 한 unit step을 취한다. Logit change는 `(+0.7, +0.2, −0.9)`이고, 새 확률은 `(0.832, 0.144, 0.024)`이다. 제거된 mass(token 3에서 0.076, token 2에서 0.056)는 이미 가장 likely한 token 1로 갔다. 그러므로 unlikely token을 낮추는 것은 mass를 mode에 집중시킨다. PPO에서 token 3의 ratio는 `0.024/0.1 = 0.24`로 `1 − ε = 0.8`보다 작다. 따라서 같은 batch의 뒤 minibatch는 이 token에 zero gradient를 받는다(§2, case 4).

**KL-controlled PPO에 있는 control.**
- Clipping은 batch별 push-down을 ratio `1 − ε`에서 bound한다. Negative-advantage token의 ratio가 오를 때 loss를 bound하지는 않는다(§2, case 3). Dual-clip은 그 case를 bound한다([[openrlhf-ppo]]).
- KL reward는 RL prompt에서 mass를 `π_ref` 쪽으로 다시 끌어당긴다(§4).
- Pretraining term(PPO-ptx) 또는 replay는 다른 data에서 positive NLL anchor를 더한다(§6.2).
- Reward whitening과 bias shift는 reward scale을 β와 비교 가능한 상태로 유지한다(§8, §9).
- Advantage whitening은 advantage를 center하므로 absolute quality와 관계없이 batch token의 약 절반이 negative advantage를 받는다(**해석**; mean subtraction에서 따른다).

**숫자로 제시된 failure mode evidence.** KL `≈ 50–85`에서 win rate가 20% 미만인 over-optimization([[n-implementation-details-rlhf-ppo]] §7.1). Safety data가 늘면서 borderline false refusal이 15%에서 27%로 증가([[llama-2]] Figure 33). Post-training 뒤 ECE가 0.007에서 0.074로 증가([[gpt-4-technical-report]] Figure 8). 이 source 중 어느 것도 negative-advantage token 효과를 positive token 효과와 분리하지 않는다. PPO 기반 RLHF에서 negative가 gain 또는 damage에 기여한 share는 **보고되지 않았다**. [[ch-43a]]는 다른 algorithm에서 이를 측정한 연구를 다룬다.

**진단.** `Â > 0` token과 `Â < 0` token을 분리해 mean advantage, clip fraction, log-probability change를 log한다. Policy entropy, `π_ref`에 대한 KL(token에 걸친 `k1` 합), consecutive policy 사이 `approxkl`([[trl-ppo]]), EOS 없이 끝난 completion rate도 log한다. Generality를 위해 borderline refusal rate, held-out multiple-choice set의 ECE, reasoning set의 large `k`에서 pass@k도 log한다.

**Generality에 미치는 효과.** Negative gradient는 이미 likely한 output에 mass를 집중시킨다. 이는 보고된 output diversity(§7)와 calibration(§6.5) loss와 일관되지만, 이 장의 source는 negative를 원인으로 분리하지 않는다(**열린 질문**).

## 레시피

| Model (정확한 release) | Size | Stage | Setting | Value | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| InstructGPT PPO-ptx ([[rlhf-instructgpt]]) | 1.3B, 6B, 175B | RL | KL coefficient β, placement | 0.02; SFT init model에 대한 token별 log-ratio, reward 안 | arXiv:2203.02155v1 App. C.4, Eq. 2 | 2026-09-15 검증됨 | App. E.7 Figure 36: Likert optimum은 “around 0.01 and 0.02”; 0과 2는 낮음 |
| same | 1.3B, 6B, 175B | RL | pretraining coefficient γ; pretraining example | 27.8; RL episode의 8× | App. C.4 | 2026-09-15 검증됨 | E.6: `γ ≥ 20`은 1.3B regression을 회복; E.11: ratio 4 vs 8 vs 32 |
| InstructGPT PPO (`γ = 0`) | 1.3B, 6B, 175B | RL | pretraining coefficient γ | 0 | §3.5 | 2026-09-15 검증됨 | Table 14 regression(§6.1) |
| InstructGPT PPO 및 PPO-ptx | 1.3B, 6B, 175B | RL | episode; unique prompt | 256k; 약 31k | App. C.4 | 2026-09-15 검증됨 | E.11: ptx가 있는 1.3B에서 256k 이후 benefit 없음; E.6: 512k는 regression |
| same | 1.3B, 6B, 175B | RL | batch; minibatch; inner epoch | 512; 64(8 minibatch); 1 | App. C.4 | 2026-09-15 검증됨 | E.11: batch 512가 64–1024 중 최고; minibatch 32가 64보다 약간 낫지만 GPU utilization 때문에 64 유지(1.3B) |
| same | 1.3B, 6B | RL | policy learning rate | 하나의 값으로 print되지 않음; 2.55e-6에서 2.55e-5 sweep, 가장 높은 Likert checkpoint 선택 | App. C.4, E.9 | 보고되지 않음(C.4, E.9, Figure 38 확인) | E.9: ptx 없는 8.05e-6 초과 run은 diverged |
| same | 175B | RL | policy learning rate | print되지 않음; 2.55e-6 및 3.74e-6 시도 | App. E.9 | 보고되지 않음 | E.9 Figure 38 |
| same | 1.3B, 6B, 175B | RL | LR schedule; EMA; GAE discount; clip; temperature | peak의 1/10에서 10 iteration warm-up을 둔 constant; EMA decay 0.992; no discount; 0.2; 1 | App. C.4 | 2026-09-15 검증됨 | ablation 보고 없음 |
| same | 1.3B, 6B / 175B | RL | value model; value learning rate | 6B, 6B RM에서 initialize; 9e-6 / 5e-6 | App. C.2, C.4 | 2026-09-15 검증됨 | C.2: 175B RM은 value initialization으로 덜 stable |
| InstructGPT PPO init model | 1.3B, 6B, 175B | SFT | epoch; pretraining mix; batch; selected LR | 2; 10%; 32(1.3B, 6B), 8(175B); 5e-6, 1.04e-5, 2.45e-6 | App. C.3 | 2026-09-15 검증됨 | E.8, Figure 37: 0/10/50% pretraining data와 1–2 epoch variant |
| Llama 2-Chat ([[llama-2-recipe]]) | 7B, 13B | RL | KL coefficient β | 0.01, reward 안(Eq. 4) | arXiv:2307.09288v2 §3.2.3 | 2026-09-14 검증됨 | ablation 보고 없음 |
| Llama 2-Chat | 34B, 70B | RL | KL coefficient β | 0.005, reward 안 | §3.2.3 | 2026-09-14 검증됨 | ablation 보고 없음 |
| Llama 2-Chat | all | RL | optimizer; LR; batch; minibatch; clip; iteration | AdamW (0.9, 0.95, eps 1e-5), wd 0.1, grad clip 1.0; constant 1e-6; 512; 64, 각각 one step; 0.2; early stopping을 둔 200–400 | §3.2.3 | 2026-09-14 검증됨 | ablation 보고 없음 |
| Llama 2-Chat | all | RL | reward | safety-tagged이거나 `R_s < 0.15`이면 `R_s`, 아니면 `R_h`; whiten(logit) | §3.2.3 | 2026-09-14 검증됨 | threshold: Meta Safety test set에서 precision 0.89, recall 0.55 |
| Llama 2-Chat | all | RL | max response length; sampling temperature | print되지 않음 | §3.2.3 | 보고되지 않음(body와 A.3 확인) | n/a |
| Anthropic HH RLHF policy ([[hh-rlhf]]) | 13M–52B | RL | KL coefficient `λ_KL` | 0.001 | arXiv:2204.05862v1 §4.1, Eq. 4.1 | 2026-09-15 검증됨 | 저자: “might actually be wholly unnecessary”; ablation 없음 |
| Pythia TL;DR PPO ([[n-implementation-details-rlhf-ppo]]) | 1B, 2.8B, 6.9B | RL | episode; optimizer; schedule; batch | 1,000,000(약 8.56 epoch); AdamW eps 1e-5, LR 3e-6; linear; 512 | arXiv:2403.17031v1 Table 7 | 2026-09-15 검증됨 | “closely follows Stiennon et al. (2020), except for a modified learning rate”(Detail 20) |
| same | 1B, 2.8B, 6.9B | RL | β; γ; λ; minibatch; PPO epoch; ε; value clip; `c_1`; temperature | 0.05; 1.0; 0.95; 1; 4; 0.2; 0.2; 0.1; 0.7 | Table 7 | 2026-09-15 검증됨 | §7.1: 이 setting에서 1B run은 KL 약 50 및 85에서 over-optimized |
| TRL PPOConfig default ([[trl-ppo-recipe]]) | any | RL | LR; PPO epoch; `kl_coef`; estimator; response length | 3e-6; 4; 0.05; k1; 53 | trl@a08e713 ppo_config.py | 2026-09-14 검증됨 | framework default, ablation 없음 |

**Small general-purpose run의 starting point.** Separate RM을 쓰는 1B–7B policy에서 검증된 row는 다음 configuration을 뒷받침한다. Reward 안 KL coefficient는 first sweep으로 0.01–0.02. 이 range는 InstructGPT의 Likert optimum(model은 최대 175B, API prompt)과 Llama 2의 7B/13B value다. Batch 512, minibatch 64 및 minibatch당 one step(InstructGPT와 Llama 2), clip 0.2, `γ = 1`, `λ = 0.95`(N+ TL;DR, 1B–6.9B). RM에서 initialize한 value model(InstructGPT; N+). InstructGPT의 ptx 없는 1.3B/6B run은 8.05e-6을 넘으면 diverged했고 Llama 2는 7B–70B에서 1e-6을 썼으므로, sweep으로 고른 constant learning rate. Pretraining 또는 mid-training data가 있으면 episode보다 8배 많은 pretraining example과 `γ = 27.8`로 log-likelihood term을 더한다. 이 setting은 GPT-3 자체 pretraining distribution에서 1.3B–175B로 test됐다. 그렇지 않으면 SFT checkpoint를 보존하고 interpolation `θ_α`를 평가한다(OpenLLaMA-3B, Mistral-7B). 이 combination 전체를 test한 source는 없다.

## Generalization 관점

**(a) Breadth를 늘리는 것.**
- RLHF는 training data를 하나도 만들지 않은 labeler의 preference를 개선했고, RM은 held-out labeler group에서 `69.6 ± 0.9%` accuracy를 유지했다([[rlhf-instructgpt]] §4.1). **결과(단일 연구)**.
- 96%를 초과해 English인 dataset에서 instruction following은 code question과 non-English instruction으로 qualitative하게 transfer했다([[rlhf-instructgpt]] §4.3). 정량화되지 않았다.
- 13B와 52B에서 RLHF는 zero-shot NLP accuracy를 높였다([[hh-rlhf]] §1.1). Python-finetuned model에 natural-language HH RLHF를 적용하면 smaller model의 HumanEval pass@1은 낮아지고 larger model은 높아졌으며, 52B에서는 gain이 pass@k의 large k에서도 유지됐다([[hh-rlhf]] Figure 21). **결과(단일 연구)**이며 format-sensitive다.
- Pretraining-gradient mixing(`γ = 27.8`)은 175B에서 few-shot SQuADv2와 HellaSwag를 GPT-3 수준으로 회복하거나 넘겼다([[rlhf-instructgpt]] Table 14).
- 특히 lower layer에서 SFT weight 쪽으로 interpolate하면 reward 대부분을 유지하며 NLP score를 높였다([[mitigating-alignment-tax-rlhf]] §5–6).

**(b) Narrowing 또는 forgetting을 일으키는 것.**
- Pretraining gradient 없는 PPO: 175B에서 few-shot SQuADv2 `−17.80 F1`, few-shot FR→EN `−13.35 BLEU`([[rlhf-instructgpt]] Table 14). Tax는 Anthropic 연구에서 small scale에 나타났고([[hh-rlhf]] Figure 3), OpenLLaMA-3B에서는 reward와 함께 커졌다([[mitigating-alignment-tax-rlhf]] §4). 서로 다른 scale과 data의 세 연구에서 방향이 **재현됨**.
- Ptx가 있어도 longer training(512k episode)은 DROP과 SQuADv2를 GPT-3 아래로 가져왔다([[rlhf-instructgpt]] App. E.6).
- High KL에서 proxy over-optimization: KL 약 50–85의 1B TL;DR run은 win rate 20% 미만이었다([[n-implementation-details-rlhf-ppo]] §7.1).
- Calibration: MMLU에서 ECE `0.007 → 0.074`([[gpt-4-technical-report]] Figure 8).
- Over-refusal: safety data가 늘며 borderline false refusal `15% → 27%`([[llama-2]] Figure 33).
- Output diversity: RL 뒤 더 좁고 lower-entropy distribution([[hh-rlhf]] §4.6), factual prompt에서 lower Self-BLEU diversity([[llama-2]] Figure 21).
- Larger β만으로는 tax를 없애지 못했고 reward를 대가로 치렀다([[rlhf-instructgpt]] App. E.6; [[mitigating-alignment-tax-rlhf]] App. C.2). **재현됨**.

**(c) 이 stage에서 이를 측정하는 법.**
- Fixed public suite를 base model과 SFT init model 모두에 score하고, PPO − base와 PPO − SFT를 따로 보고한다. Table 14에서 175B zero-shot DROP의 14.45-point drop 중 11.74는 SFT에서 이미 나타난다.
- Few-shot과 zero-shot을 따로 보고한다. 저자는 PPO regression을 “particularly in the few-shot setting”(App. E.1)이라고 설명하지만, 175B에서 SQuADv2, DROP, FR→EN의 zero-shot drop이 더 크다([[rlhf-instructgpt]] Table 14).
- Held-out labeler 또는 judge와 annotator group 간 cross-validated RM accuracy를 사용한다.
- Proxy reward가 아니라 gold 또는 human score를 `√KL`에 대해 추적한다([[reward-model-overoptimization]]; [[hh-rlhf]] §4.3).
- Borderline over-refusal set([[llama-2]] §4.2.3; [[ch-52]]) 및 calibration metric(ECE)([[gpt-4-technical-report]] Figure 8)를 추가한다.
- Preference win rate를 비교할 때 length를 control한다([[n-implementation-details-rlhf-ppo]] Figure 12).
- 알려진 measurement error: multiple-choice formatting은 small/large model을 반대 방향으로 바꾼다([[hh-rlhf]] §4.6.1). Qualitative transfer claim은 cherry-picked prompt다([[rlhf-instructgpt]] Figure 8). Aggregate refusal rate는 borderline behavior를 숨긴다([[llama-2]] §4.2.3). Figure-only result는 paper 사이에서 수치로 비교할 수 없다([[mitigating-alignment-tax-rlhf]]).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 방법 |
|---|---|---|
| `γ = 0` model을 “InstructGPT”라고 부름 | InstructGPT에 pretraining mix가 없었다고 주장; released model에 tax를 귀속 | §3.5: “InstructGPT refers to the PPO-ptx models”; `γ = 0` model은 “PPO” |
| Framework default를 논문 setting으로 읽음 | “InstructGPT used 4 PPO epochs”; learning rate를 세 자리로 인용 | InstructGPT App. C.4: inner epoch 하나; policy LR은 single value로 print되지 않음(E.9) |
| 모든 size에 KL coefficient 하나를 사용 | Llama 2 70B를 `β = 0.01`로 인용 | Llama 2 §3.2.3: 7B/13B는 0.01, 34B/70B는 0.005 |
| Reward-side KL을 “k3”라고 부름 | Reward penalty가 항상 positive; log에 negative token별 KL reward가 없음 | Eq. 2와 TRL default는 `k1 = log π_θ − log π_ref`; k3는 loss-side option |
| Capability regression을 고치려고 β를 올림 | Validation reward는 떨어지는데 DROP/SQuADv2는 낮게 유지 | App. E.6 Figure 34: β를 2.0까지 올려도 회복하지 못함; ptx 또는 averaging 추가 |
| Tax를 base model에 대해서만 측정 | SFT 뒤 이미 생긴 drop을 RLHF 탓으로 돌림 | 같은 suite에서 SFT init checkpoint를 score(Table 14 SFT column) |
| Value model을 random initialize하거나 warm-up 없이 shared | Early iteration에서 큰 value loss; noisy advantage | RM에서 initialize([[rlhf-instructgpt]] C.4; N+ Detail 22); explained variance log |
| PPO 중 dropout을 켜 둠 | 첫 minibatch에서 ratio ≠ 1; update 전 KL estimate가 nonzero | N+ Detail 7: dropout disable; step 0에서 mean ratio = 1 assert |
| EOS handling 없음 | Truncated text의 RM score; response length가 limit까지 증가 | EOS 없는 completion rate를 log; `missing_eos_penalty` 적용(N+ Detail 23) |
| RM reward로 progress 판단 | Reward와 KL은 오르는데 held-out judge win rate가 하락 | Judge score를 `√KL`에 대해 plot; peak 부근에서 stop([[reward-model-overoptimization]]) |
| Aggregate refusal rate만 safety-helpfulness metric으로 사용 | false refusal 0.05%를 보고하지만 borderline refusal은 27% | Borderline set 분리([[llama-2]] Figure 33) |
| Accuracy만 보고 calibration은 보고하지 않음 | MCQ accuracy는 평평하지만 confidence가 uninformative | RL 전후 held-out MCQ set에서 ECE([[gpt-4-technical-report]] Figure 8) |
| Clipping이 negative-advantage update를 bound한다고 믿음 | `Â < 0`이고 ratio가 큰 token에서 loss spike | Advantage sign별 clip fraction과 ratio log; dual-clip 고려([[openrlhf-ppo]]) |

## 이해 확인

1. InstructGPT는 β를 2.0까지 올려도 DROP과 SQuADv2를 회복하지 못했지만 `γ = 27.8`은 회복시켰다. KL penalty의 정의(KL은 어느 input에서 측정되는가?)와 ptx term의 정의를 사용해, 두 regularizer가 서로 다른 capability를 보호하는 이유를 설명하라.
2. Table 14에서 zero-shot DROP은 GPT-3에서 SFT로 `11.74 F1` 하락하고 SFT에서 PPO로 추가 `2.71` 하락한다. 이것이 RLHF에 tax를 귀속하는 데 무엇을 뜻하는지, 그리고 자신의 pipeline에서 forgetting report의 baseline으로 어느 checkpoint를 쓸지를 설명하라.
3. `Â = −1`, `ε = 0.2`인 token에서 PPO gradient는 `r = 0.7`에서 zero이고 `r = 1.5`에서 nonzero다. `L^CLIP`의 min이 이 asymmetry를 만드는 이유를 설명하고, LLM rollout에서 negative-advantage token에 `r > 1`이 되는 training situation을 서술하라.
4. Token별 term `−β(log π_θ − log π_ref)`은 어떤 token에는 positive일 수 있다. 이것이 reward에 허용되는 이유와, 이를 zero에서 clip하면 objective optimum `π* ∝ π_ref exp(r/β)`가 왜 바뀌는지 설명하라.
5. Anthropic은 13B와 52B에서 alignment bonus를, InstructGPT는 175B에서 regression을 발견했다. 반대 방향을 설명할 수 있는 setup 차이 세 가지를 열거하고, model scale을 evaluation format에서 분리할 experiment 하나를 설계하라.
6. Llama 2에서는 safety data가 늘어도 helpfulness RM score는 일정했지만 borderline false refusal은 15%에서 27%로 늘었다. 0.15 threshold가 있는 piecewise reward rule이 PPO 동안 safe prompt에서 refusal을 만들 수 있는 방법을 설명하고, 그 설명을 test하려고 RM 또는 PPO prompt set에 어떤 data를 추가할지 서술하라.
7. GPT-4의 exam multiple-choice accuracy는 RLHF 전 73.7%, 후 74.0%였지만 MMLU subset ECE는 0.007에서 0.074로 올랐다. Preference reward에 대한 policy-gradient objective가 accuracy를 바꾸지 않고 calibration을 바꿀 수 있는 방법을 설명하라. 이어서 두 개의 서로 다른 benchmark를 쓴 것이 이 비교를 제한하는 이유를 설명하라.
8. OpenLLaMA-3B에서는 4× RLHF data replay가 model averaging보다 낮았지만, InstructGPT의 8× pretraining mix는 175B에서 작동했다. Data coverage와 anchor strength를 사용해 두 결과가 함께 성립할 수 있는 이유를 설명하라.

## 연결

- **이전:** ch-37 — Language Model을 위한 Policy-Gradient Foundations(§1–§3가 세우는 score-function estimator, baseline, actor-critic view).
- **다음:** ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity(§7에서 제기한 diversity 및 out-of-distribution trade-off).
- ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization(`r`을 공급하는 RM; §4에서 쓴 over-optimization curve).
- ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control(§6의 SFT-stage counterpart).
- ch-30c — Weight Averaging and Model Merging for Generalist Models(model averaging, §6.4).
- ch-39 — Offline Preference Optimization: DPO and Its Variants(online sampling 없이 같은 KL-regularized optimum `π*`).
- ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO(critic-free advantage; loss-side KL).
- ch-42 — Reward Hacking and Judge Design(RM이 exploit될 때의 failure mode).
- ch-43 — Entropy, Output Diversity, and KL Control in RL(entropy와 KL 진단).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages(gradient로서 negative의 전체 설명).
- ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting(calibration metric).
- ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming(borderline false-refusal set).
- ch-56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls; ch-57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers(§3–§4의 framework code).

## 출처

- [[ppo]] — `L^CPI`, `L^CLIP`, adaptive KL rule, combined objective, truncated GAE, MuJoCo Table 1(arXiv:1707.06347v2); `excerpts/`에 발췌.
- [[trpo]] — Theorem 1 bound, Eq. 9–12 constraint, `δ = 0.01`, conjugate-gradient cost(arXiv:1502.05477v5 읽음; library card의 hyperparameter table은 사용하지 않음); `excerpts/`에 발췌.
- [[rlhf-instructgpt]] — Eq. 2, model naming, App. C.2–C.4 setting, E.6–E.11 ablation, Table 14, held-out labeler, qualitative transfer(arXiv:2203.02155v1 읽음; library card의 hyperparameter table은 사용하지 않음); `excerpts/`에 발췌.
- [[hh-rlhf]] — `λ_KL = 0.001`, `√KL` relation, scale-dependent tax와 bonus, format caveat, lower-entropy output(arXiv:2204.05862v1 읽음); `excerpts/`에 발췌.
- [[mitigating-alignment-tax-rlhf]] — Model averaging, regularizer와의 comparison, KL penalty와 replay, HMA(arXiv:2309.06256v4); `excerpts/`에 발췌.
- [[llama-2]], [[llama-2-recipe]], [[llama-2-ppo]] — PPO schedule, piecewise reward, size별 β, setting, false-refusal study, diversity.
- [[costa-huang-ppo-details]] — 번호가 있는 13개 core PPO detail의 numbering과 evidence(ICLR Blog Track post 읽음); `excerpts/`에 발췌.
- [[n-implementation-details-rlhf-ppo]] — Detail 7과 22–25, Table 7, over-optimization 및 length result(arXiv:2403.17031v1); `excerpts/`에 발췌.
- [[kl-control-rlhf]] — Korbak et al. Eq. 3–7, KL-regularized optimum(arXiv:2205.11275v2 읽음); `excerpts/`에 발췌.
- [[john-schulman-kl-tricks]] — k1/k2/k3 definition과 Gaussian bias/variance number(2020-03-07 post 읽음); `excerpts/`에 발췌.
- [[trl-ppo]], [[trl-ppo-recipe]] — Token별 reward와 GAE code, value clipping, commit a08e713의 default; `excerpts/`에 발췌.
- [[openrlhf-ppo]] — KL placement option, estimator recommendation, adaptive controller, dual-clip.
- [[reward-model-overoptimization]] — Synthetic gold-RM setting에서 early stopping처럼 작동하는 KL penalty.
- [[gpt-4-technical-report]] — Post-training 전후 calibration, base versus RLHF exam accuracy(arXiv:2303.08774v6); `excerpts/`에 발췌.
- [[lilianweng-rlhf]] — RLHF pipeline의 tutorial overview; 이 장에서는 어떤 number의 evidence로도 사용하지 않음.
