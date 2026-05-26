<!-- chapter: ch-37
     track: rl
     kind: content
     title: Policy-Gradient Foundations
     deps: [ch-36]
     sources: [[vanilla-pg]], [[trpo]], [[ppo]], [[rloo]], [[reinforce-plus-plus]],
              [[lilianweng-rlhf]], [[nathan-lambert-rl-overview]], [[costa-huang-ppo-details]],
              [[maximum-entropy-rl]]
     figures: figures/pg-variance.html
     opens: rl-track (ch-37..ch-46)
-->

# 제37장 — 정책 그래디언트 기초

> **핵심 통찰.** LLM을 위한 RL은 지도학습과 전혀 다른 괴물이 아니다. 각 샘플마다 *선택된* 가중치를 붙인 지도학습이다. 정책 그래디언트 정리는 하나의 항등식으로 줄어든다. `∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A(s,a)]`. 현대의 모든 LLM-RL 알고리즘 — PPO, TRPO, RLOO, GRPO, REINFORCE++ — 은 (a) `A(s,a)`를 무엇으로 둘지, (b) 그것을 샘플에서 어떻게 추정할지, (c) `π_θ`를 기준 정책 근처에 붙들어 두기 위해 어떤 정규화 항을 덧붙일지에 대한 서로 다른 선택이다. 알고리즘 계열은 2차 관심사다. 1차 관심사는 `A` 추정량의 분산이다. 분산이 바로 그래디언트 배치가 모델을 움직일지, 아니면 서로 상쇄될지를 결정하기 때문이다.
>
> **지침.** RL 알고리즘을 고르기 전에 (1) 어떤 형태의 score-function 추정량을 쓰는지(타임스텝별 vs 시퀀스별, causal vs full-trajectory), (2) 어떤 baseline이 action-independent floor를 빼는지(none / moving average / learned V / leave-one-out / group-mean / global-batch), (3) 정규화 항이 *reward 안의* KL-to-reference penalty(RLHF 표준)인지, 아니면 *loss 안의* 별도 KL constraint / entropy bonus(고전 RL, GRPO)인지 알아야 한다. "PPO는 잘 되니까"라는 이유로 PPO를 기본값으로 삼으면, 분산이 시퀀스 길이에 비례해 커지는 advantage 추정량을 디버깅하는 데 한 분기를 쓰게 된다.

---

## 이 장이 필요한 이유

이 장은 RL 트랙의 시작이다. SFT 트랙(ch-30 … ch-36)은 이제 `π_ref`가 될 고정된 `checkpoint-final`을 만들었다. 이것은 다음 열 개 장의 모든 RL 방법이 정규화 기준으로 삼는 reference policy다. TRPO, PPO, DPO, GRPO, RLVR을 소개하기 전에, 그들이 모두 공유하는 하나의 대상이 필요하다. 바로 정책 그래디언트의 score-function 추정량이다. Nathan Lambert의 프레이밍([[nathan-lambert-rl-overview]])은 이를 깔끔하게 말한다. "이 분야는 소수의 알고리즘 템플릿(PPO, DPO, GRPO)으로 수렴했다." 그래서 이 장이 그 템플릿이다. 뒤따르는 모든 내용은 특수화다.

이 장이 끝날 때 얻어야 할 산출물은 네 가지다. (1) 종이에 다시 쓸 수 있을 만큼 `∇J(θ) = E[∇log π · A]`를 유도할 수 있어야 한다. (2) 임의의 baseline `b(s)`를 빼도 unbiased이며 분산을 줄인다는 증명을 알아야 한다. (3) LM-RL이 로보틱스 RL보다 구조적으로 왜 더 단순한지, 그리고 그 단순성이 [[rloo]]가 말하듯 PPO 장치의 절반을 왜 죽이는지 이해해야 한다. (4) entropy regularisation에 대한 작동하는 멘탈 모델, 즉 무엇을 고치고 무엇을 가리는지 이해해야 한다.

---

## §1 정책 그래디언트 정리 — 유도

`π_θ(a|s)`를 확률적 정책이라고 하고, `J(θ) = E_{τ ∼ π_θ}[R(τ)]`를 trajectory `τ = (s_0, a_0, r_0, s_1, a_1, r_1, …)`에 대한 기대 return이라고 하자. 우리는 `∇_θ J(θ)`를 원한다. `π_θ` 아래에서 trajectory의 확률은 다음처럼 factorize된다.

```
p_θ(τ) = ρ_0(s_0) · Π_t π_θ(a_t | s_t) · P(s_{t+1} | s_t, a_t)
```

가운데 항만 `θ`에 의존한다. **log-prob trick**(score-function identity)은 다음과 같다.

```
∇_θ p_θ(τ) = p_θ(τ) · ∇_θ log p_θ(τ) = p_θ(τ) · Σ_t ∇_θ log π_θ(a_t | s_t)
```

초기 상태와 transition 항에는 `θ`가 없으므로 gradient에 0을 기여한다. `∇_θ`를 expectation 안으로 밀어 넣으면:

```
∇_θ J(θ) = ∇_θ ∫ p_θ(τ) R(τ) dτ
         = ∫ ∇_θ p_θ(τ) · R(τ) dτ
         = ∫ p_θ(τ) · [Σ_t ∇_θ log π_θ(a_t | s_t)] · R(τ) dτ
         = E_{τ∼π_θ}[ Σ_t ∇_θ log π_θ(a_t | s_t) · R(τ) ]
```

이것이 **score-function (REINFORCE) estimator**이며, Williams 1992([[vanilla-pg]])가 connectionist 형태로 처음 적었다. 이 추정량은 *unbiased*다. trajectory 하나만 샘플해도 `E[∇̂J] = ∇J`다. 동시에 *high-variance*다. log-prob-gradient의 합에 하나의 scalar return을 곱한 값의 분산은 trajectory 길이와 `R`의 scale에 따라 커진다.

### Causal form (return-to-go)

미래는 과거에 영향을 줄 수 없다. `R(τ) = Σ_{t'} r_{t'}`는 나눌 수 있다. `t' < t`인 reward는 `a_t`와 독립이므로 expectation에서 이 항들은 0을 기여한다. 따라서 모든 현대 구현에서 쓰는 **causal** 형태가 나온다([[vanilla-pg]] §Technical Details).

```
∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]   where   G_t = Σ_{t'≥t} γ^{t'−t} r_{t'}
```

`G_t`는 step `t`부터의 **return-to-go**다. 과거 reward 항을 버리면 bias를 만들지 않고 분산을 엄밀히 줄인다.

---

## §2 Baseline과 분산 감소 증명

action에 의존하지 않는 임의의 함수 `b(s)`에 대해:

```
E_{a∼π_θ(·|s)}[ ∇_θ log π_θ(a|s) · b(s) ]
    = b(s) · Σ_a π_θ(a|s) · ∇_θ log π_θ(a|s)
    = b(s) · Σ_a ∇_θ π_θ(a|s)
    = b(s) · ∇_θ Σ_a π_θ(a|s)
    = b(s) · ∇_θ 1
    = 0
```

따라서 **baseline-augmented** 추정량도 여전히 unbiased다([[vanilla-pg]] Theorem 1).

```
∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)) ]
```

**왜 분산을 줄이는가.** `X = ∇log π · G`, `X' = ∇log π · (G − b)`라고 하자. 둘의 평균은 같다. `X'`의 한 좌표에 대한 분산은:

```
Var(X'_i) = Var(X_i) − 2 · Cov(X_i, ∇log π_i · b) + Var(∇log π_i · b)
```

좌표별 최소 분산 constant baseline은 `b* = E[G · (∇log π_i)^2] / E[(∇log π_i)^2]`, 즉 `∇log π`로 가중한 평균 return이다. 실제로는 `E[G | s]`에 가까운 어떤 `b(s)`라도 분산을 크게 줄인다. 그래서 *value function* `V^π(s) = E_π[G | s]`가 표준 baseline이며, raw return 대신 "advantage" `A(s,a) = Q^π(s,a) − V^π(s)`를 추정량에 넣는 것이다.

### Baseline의 동물원

| Method | Baseline | Where it lives | Source |
|---|---|---|---|
| Raw REINFORCE | 0 | — | [[vanilla-pg]] |
| REINFORCE w/ moving avg | `b̄ = (1/S) Σ_s R_s` | scalar EMA | [[vanilla-pg]] |
| Actor-critic / A2C | `V_φ(s)` | learned head | 아래 §3 |
| PPO | `V_φ(s)` + GAE | learned head + λ knob | [[ppo]] |
| RLOO | `(1/(k−1)) Σ_{j≠i} R(y_j, x)` | rollout 간 leave-one-out | [[rloo]] |
| GRPO | group G 위의 `(r_i − mean(r)) / std(r)` | group z-score | [[reinforce-plus-plus]] |
| REINFORCE++ | full batch 위의 `(G − mean_B(G)) / std_B(G)` | global z-score | [[reinforce-plus-plus]] |

모두 conditioning state가 주어졌을 때 action-independent이므로 unbiased다(LLM에서는 prompt `x`). 차이는 오직 *분산*과, *얼마나 많은 sampling / memory / compute*를 요구하는지뿐이다. [[rloo]]의 기여는 정확히 다음 관찰이었다. prompt당 `k ≥ 2` rollout이 있으면 leave-one-out baseline은 학습된 `V_φ(s)`보다 통계적으로 더 좋고, value network를 시스템에서 완전히 제거한다. KL을 맞춘 TL;DR과 HH-RLHF에서 PPO 대비 약 50% 메모리 footprint로 더 높은 win-rate를 얻는다.

### LLM-specific form — 시퀀스당 하나의 advantage

terminal reward `R(x, y)`를 갖는 시퀀스 `y = (y_1, …, y_T)`가 `π_θ(·|x)`에서 autoregressive하게 샘플되었다고 하자([[vanilla-pg]] §LLM-specific form).

```
∇_θ J = E_{y∼π_θ(·|x)}[ R(x, y) · Σ_t ∇_θ log π_θ(y_t | x, y_<t) ]
```

합 `Σ_t ∇_θ log π_θ(y_t | x, y_<t) = ∇_θ log π_θ(y | x)`는 부호가 뒤집힌 SFT cross-entropy gradient에 정확히 해당하며, `R(x, y)`로 가중된다. **그것이 전체 알고리즘이다**. generation을 실행하고, 시퀀스별 가중치를 계산한 뒤, weighted SFT step을 수행한다. [[rloo]]는 이 식에 leave-one-out baseline을 더한 것일 뿐이다.

---

## §3 Actor-critic: 분산을 낮추기 위해 bias를 받기

Monte-Carlo `G_t`는 unbiased지만 high-variance다(`T − t`개의 noisy reward를 더하기 때문이다). **Bootstrapped** 추정량은 학습된 value function으로 대체한다.

```
Â_t^{(1)} = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)     (TD residual; biased if V_φ ≠ V^π, low variance)
Â_t^{(∞)} = G_t − V_φ(s_t)                       (MC advantage; unbiased, high variance)
```

[[ppo]] §Technical Details는 전체 스펙트럼을 잇는 **GAE** interpolation을 제시한다.

```
δ_t = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)
Â_t^{GAE(λ)} = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + …
```

`λ = 1`이면 Monte-Carlo(unbiased, high variance)를 회복한다. `λ = 0`이면 1-step TD(`V_φ`의 오차로 biased, low variance)를 회복한다. RLHF 기본값은 `λ = 0.95, γ = 1.0`이다([[ppo]] canonical hparams, [[lilianweng-rlhf]] RLHF defaults). `γ = 1.0`, 즉 undiscounted를 쓰는 이유는 [[lilianweng-rlhf]]가 말하듯 LLM RL에서 "rewards concentrate at EOS"이기 때문이다. step별 non-zero reward는 정확히 하나(EOS token에서 RM score)뿐이므로, `γ < 1`은 signal을 버릴 뿐이다.

**Bias-variance tradeoff는 정량적이다.** `V_φ`가 `V^π`에 대해 mean-squared error `ε^2`를 가진다고 하자. 그러면 `Â_t^{(1)}`의 bias는 step당 `O(ε)`이고, T-step 합에서는 최악의 경우 `O(Tε)`로 누적된다. MC 추정량의 분산은 `Σ_{t'≥t} Var(r_{t'})`에 비례한다. terminal reward가 하나뿐인 LLM에서는 T와 무관하게 `Var(R)`이므로 MC 추정량의 분산은 *bounded*다. 이것이 중요한 LLM-specific 관찰이다. 긴 horizon 때문에 bootstrap이 강제되는 로보틱스와 달리, LLM의 terminal-reward 구조는 unbiased MC 추정량을 경쟁력 있게 만든다. 바로 이 이유로 [[rloo]]는 `V_φ`를 완전히 버릴 수 있다고 주장한다.

---

## §4 LM-RL이 특별한 이유

고전 RL 논문(TRPO, PPO, SAC)은 로보틱스와 게임을 위해 쓰였다. LLM post-training의 네 가지 구조적 속성은 그 논문들이 최적화했던 가정을 깨뜨린다. [[rloo]] §Key Contributions가 가장 명확한 열거이며, 아래 목록은 그것을 [[reinforce-plus-plus]]와 [[lilianweng-rlhf]]의 refinement로 요약하고 확장한다.

**(a) Deterministic dynamics.** prefix `(x, y_<t)`와 KV cache가 주어지면 stochastic environment transition은 없다. 유일한 randomness는 `π_θ(·|x, y_<t)`에서 `y_t`를 샘플하는 것이다. Bellman equation의 stochasticity가 사라진다. environment stochasticity를 다루기 위해 만든 variance-reduction 장치(target networks, double Q, clipped double-Q)는 LM에 *무관*하다.

**(b) Full-trajectory rewards.** 거의 모든 LLM RL reward는 terminal이다. `y_T = EOS` 이후 scalar `R(x, y)` 하나가 나온다. step별 reward는 shaped KL term이나 process-reward model([[prm800k]], [[math-shepherd]])을 추가할 때만 존재한다. terminal-only reward에서는 모든 `t`에 대해 `G_t = R(x, y)`이므로, causal per-step gradient는 `R(x, y) · ∇_θ log π_θ(y | x)`로 붕괴한다. 이는 §2의 LLM-specific form이다.

**(c) 매우 긴 episode, 매우 짧은 batch.** rollout 하나가 2K–32K token일 수 있다([[ppo]] MuJoCo: actor당 T=2048 vs 현대 RLHF run의 32K token). token별 advantage는 한 시퀀스 안에서 강하게 상관되어 있다. 같은 terminal reward를 공유하기 때문이다. [[reinforce-plus-plus]]는 이 상관 때문에 global batch normalization이 group-local normalization을 이긴다고 주장한다. k가 작을 때 per-prompt group mean의 분산 자체가 크기 때문이다.

**(d) 100K-way vocabulary 위의 discrete action.** policy entropy, clip threshold, KL divergence는 모두 `|V| ≈ 128K`인 categorical 위에서 계산된다. exact KL `Σ_y π(y|x) log(π(y|x)/π_ref(y|x))`는 token별로 tractable하며 token-level KL의 기본값이다([[lilianweng-rlhf]] KL penalty implementation, [[reinforce-plus-plus]] k1 estimator).

구체적으로, 이 네 속성이 PPO recipe에서 죽이는 것은 다음과 같다. value network는 선택 사항이 된다([[rloo]]가 제거하고, [[reinforce-plus-plus]]가 제거하고, GRPO가 제거한다). reward가 terminal일 때 GAE는 중복된다(`λ=1`은 bias가 없다). rollout당 K>1 epoch는 종종 역효과다. 각 rollout이 4K-token 시퀀스일 때 PPO-clip의 trust region은 빠르게 위반된다. [[rloo]]는 K=1을 쓴다. 살아남는 것은 **score-function estimator**, **baseline**, **KL-to-reference regulariser**다. 이 세 대상이 최소 실행 가능한 LLM-RL 방법이며, 다음 열 개 장은 각각을 어떻게 만들지에 대한 논쟁이다.

---

## §5 Entropy term — 정규화 항인가, 임시방편인가

고전 RL은 loss에 **entropy bonus**를 더한다. `+ c2 · H(π_θ(·|s))`. 명시적 목적은 정책이 충분히 탐색하기 전에 단일 action으로 붕괴하지 않도록 하는 것이다. [[ppo]]는 MuJoCo에서 `c2 = 0.01`을 쓴다. CleanRL의 Atari config도 같다. [[maximum-entropy-rl]]은 더 깊은 이유를 유도한다. 최적 max-ent policy는 `π*(a|s) ∝ exp(Q(s,a)/α)`인 soft Boltzmann이며, 그 temperature `α`가 entropy coefficient다. soft-Q value function은 `V(s) = α · log Σ_a exp(Q(s,a)/α)`이고, auto-α tuning(SAC-v2)은 `α`를 조정해 *target entropy* `H̄`를 맞춘다.

**LLM-RL에서는 entropy-bonus 이야기가 다르다.** [[lilianweng-rlhf]]는 직설적으로 말한다. "entropy bonus is often dropped in RLHF because KL regularization to reference policy already regularizes the policy toward a stochastic distribution." 표준 InstructGPT-style objective는:

```
L_RLHF(θ) = −E_{y∼π_θ(·|x)}[ R(x, y) ] + β · KL( π_θ(·|x) || π_ref(·|x) )
```

KL term은 trust-region constraint이자 entropy lower-bound다. `KL(π_θ || π_ref) = E_π_θ[log π_θ − log π_ref] = −H(π_θ) − E_π_θ[log π_ref]`. 따라서 KL-to-reference를 penalize하면(`π_ref`의 support가 주어졌을 때) `−H(π_θ)`를 아래에서 직접 bound한다. 정책은 KL 비용을 치르지 않고 entropy를 collapse시킬 수 없다. 그 위에 명시적인 `+ c2 · H(π)`를 더하는 것은 대체로 중복이다.

**언제 entropy bonus가 임시방편인가?** 문헌에 기록된 두 증상이 있다. (1) *Entropy collapse* — token별 entropy가 training 중간에 0으로 향하고, policy가 greedy로 퇴화하며, RL이 멈춘다. 원인은 너무 작은 β(KL이 실제로 constrain하지 않음), 잘못 scale된 reward(하나의 high-reward mode가 mass를 삼킴), on-policy staleness 등이 있다. entropy bonus를 추가하는 것은 reward / KL config를 고치는 대신 이를 *가리는* 것이다. (2) *Reasoning RL (R1-style) distribution collapse* — [[nathan-lambert-rl-overview]]는 GRPO의 length-normalisation artifact가 짧은 response를 단조롭게 선호하게 만들어 policy를 수축시킬 수 있다고 지적한다. 여기서도 fix는 loss다. [[dr-grpo]]가 normalization을 고친다. entropy patch가 아니다.

**언제 진짜 regulariser인가?** exploration이 실제 bottleneck이고 KL-to-reference가 너무 약할 때(`π_ref` 자체가 관련 prompt에서 low-entropy인 경우)다. [[maximum-entropy-rl]]의 target entropy에 대한 auto-α가 원칙적인 해결책이다. `H̄`를 SFT model의 token별 entropy의 일부로 설정하고, `α`가 조정되게 한다. 대부분의 production RLHF stack은 둘 다 하지 않는다. `c2 = 0`으로 두고 KL-to-ref를 믿는다. 그리고 그것이 ch-38 이후의 올바른 기본값이다. entropy term은 entropy collapse를 *관찰했을 때* 꺼낼 도구이지, 기본으로 실어 보낼 것이 아니다.

---

## §6 이 템플릿으로 나머지 트랙을 어떻게 예측할 수 있는가

템플릿 `∇J = E[∇log π · A] + regulariser`는 다음 열 개 장을 knob들의 표로 붕괴시킨다.

| Chapter | Algorithm | `A` estimator | Regulariser | Distinctive component |
|---|---|---|---|---|
| ch-38 | TRPO / PPO / InstructGPT | `V_φ`가 있는 GAE-λ | token별 KL-to-ref `β` | clipped ratio / trust region |
| ch-39 | DPO / IPO / KTO / SimPO / ORPO | closed-form; 명시적 `A` 없음 | `β`를 통한 implicit | offline, rollout 없음 |
| ch-40 | GRPO / Dr. GRPO | group-mean `(r_i − μ_G)/σ_G` | per-token KL (k3) | value net 없음; group rollouts |
| ch-41 | RLOO / REINFORCE++ | leave-one-out 또는 global-batch z-score | shaped reward로서의 KL | critic-free |
| ch-42 | RLVR (verifier rewards) | GRPO/PPO와 동일 | KL-to-ref | deterministic `r(x,y)=v(x,y)` |
| ch-43 | Process-reward RL | PRM의 per-step `r_t` | KL-to-ref | per-step credit assignment |
| ch-44 | Iterative / online RLHF | round를 도는 flywheel | KL-to-ref (fresh) | loop structure |
| ch-45 | Self-play / RLAIF | RM 또는 model-judge reward | KL-to-ref | signal source |
| ch-46 | Track capstone | — | — | — |

모든 행은 같은 방정식이다. 알고리즘 문헌이 실제보다 더 커 보이는 이유는 아무도 같은 방정식을 두 번 적지 않기 때문이다. 대신 바꾼 knob 하나를 중심으로 새 논문을 쓴다.

---

## Companion visualization

**[figures/pg-variance.html](figures/pg-variance.html)** — interactive gradient-variance simulator. baseline type(none / constant / value-function / leave-one-out)을 고르고 작은 RLHF-like 문제(prompt당 k rollouts, terminal reward)에서 추정량의 iteration별 variance curve를 보라. 이 curve는 [[rloo]]의 leave-one-out이 작은 k에서 moving-average baseline을 왜 이기는지, 그리고 k가 1–2일 때 global-batch normalisation([[reinforce-plus-plus]])이 group-local normalisation을 왜 이기는지를 구체화한다. ch-38을 읽기 *전에* 사용하라. 분산 논증이 이후 모든 알고리즘 선택을 이끈다.

---

## Connections

- **ch-36 (SFT capstone)** — packed SFT run의 `checkpoint-final`은 이 트랙의 모든 RL 장에서 `π_ref`다. 이 장의 템플릿에 있는 모든 regularisation term은 그 checkpoint까지의 거리 penalty다.
- **ch-38 (KL-Controlled RLHF)** — TRPO의 monotonic-improvement bound([[trpo]])와 PPO의 clipped surrogate([[ppo]])는 템플릿의 처음 두 특수화다. InstructGPT는 PPO + token별 KL-to-ref다.
- **ch-39 (Offline preference)** — DPO는 sample-based `∇J`를 Bradley-Terry preference 아래의 closed form으로 대체한다. train time에 `π_θ` rollout이 없다.
- **ch-40 / ch-41 (GRPO / RLOO / REINFORCE++)** — 세 critic-free 특수화이며, 각각 다른 baseline을 고른다.
- **ch-42 (RLVR)** — 학습된 reward model을 deterministic verifier로 대체한다. policy-gradient 구조는 바뀌지 않는다.
- **ch-47..ch-53 (Eval, Reward, Judge)** — 이 장의 reward signal `R(x, y)`는 바로 그 장들이 구축하는 대상이다.

## Further reading

- [[vanilla-pg]] — Williams 1992. policy-gradient theorem, baseline-invariance theorem, eligibility/score function. 이 장의 §1과 §2가 유도하는 source.
- [[trpo]] — Schulman 2015. Monotonic-improvement bound + KL trust region. "score function"에서 "trust-region policy optimisation"으로 가는 첫 단계.
- [[ppo]] — Schulman 2017. Clipped surrogate, combined actor-critic loss, GAE. 표준 hparams.
- [[rloo]] — Ahmadian 2024. LLM-RL에는 value network, GAE, clip, K>1 epoch가 필요 없다. leave-one-out baseline은 약 50% 메모리로 PPO를 이긴다.
- [[reinforce-plus-plus]] — Hu 2025. Global batch advantage normalisation; `k=1` critic-free recipe.
- [[lilianweng-rlhf]] — RLHF tutorial; `r_total = r(x,y)·1[y=EOS] − β·log(π/π_ref)`, reward whitening, 그리고 entropy bonus가 보통 제거되는 이유.
- [[nathan-lambert-rl-overview]] — algorithm-to-reward-signal framing; reward-signal source가 1차 선택이다.
- [[costa-huang-ppo-details]] — 37-trick implementation reference; 논문을 재현 가능하게 만드는 요소.
- [[maximum-entropy-rl]] — SAC의 `π*(a|s) ∝ exp(Q/α)`, auto-α tuning, target entropy `H̄`.
