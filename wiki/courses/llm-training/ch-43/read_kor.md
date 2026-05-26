<!-- chapter: ch-43
     track: rl
     kind: content
     title: Entropy Dynamics and KL Control
     deps: [ch-42]
     sources: [[entropy-mechanism-llm-rl]], [[entropy-collapse-ppo]], [[entropy-regularization-ppo]],
              [[maximum-entropy-rl]], [[john-schulman-kl-tricks]], [[kl-control-rlhf]],
              [[nathan-lambert-entropy-rl]], [[openrlhf-entropy-debugging]],
              [[entropy-logging-patterns]], [[sampling-temperature-schedule]]
     figures: figures/entropy-dynamics.html
-->

# 43장 — 엔트로피 동역학과 KL 제어

> **핵심 통찰.** RLHF와 RLVR은 "보상 최대화"가 아니다. 둘은 *같은 PPO/GRPO 골격을 공유하는 두 가지 정규화된 최적화*다. 하나의 정규화 항은 **엔트로피** H(π)로, 조건부 토큰 분포를 넓게 유지해 탐색이 죽지 않게 한다. 다른 하나는 **참조 정책 대비 KL** KL(π‖π_ref)로, 정책을 SFT 사전분포(prior) 근처에 붙들어 보상 해킹이 유창성을 잡아먹지 않게 한다. 둘은 서로 다른 질문에 답한다. 한 상태에서의 분포 폭과, 여러 상태에 *걸친* 사전분포로부터의 거리다. 그리고 서로 다른 방식으로 실패한다. 2024–2025년 RL에서 가장 중요한 경험적 사실 두 가지는 (a) Cui 2025의 법칙 `R(step) = −a·exp(H(step)) + b` — 보상 상한은 엔트로피 동역학에 따라 오르내린다는 것 — 그리고 (b) KL에 대한 Schulman의 k3 추정량 `(p/q − 1) − log(p/q)`다. k3는 편향이 없고 *비음수*이기 때문에 모든 프로덕션 스택에서 순진한 k1을 대체했다. 이 장의 나머지는 이 두 사실에 대한 해설이다.
>
> **가이드라인.** 매 스텝마다 토큰별 엔트로피와 배치별 KL을 로그로 남기고, 이를 1급 진단 지표로 다뤄라. 마지막 토큰 분포에서 엔트로피가 0.1 nats 아래로 떨어지면 붕괴한 것이다. 공분산을 겨냥한 개입([[entropy-mechanism-llm-rl]]의 Clip-Cov / KL-Cov)이 평평한 엔트로피 보너스식 구조보다 낫다. KL 페널티가 필요하다면 k3를 *손실 항*으로 쓰거나(GRPO 관례) *보상 shaping*으로 쓰되(PPO 관례), k1으로 쓰지 말고 둘을 동시에 쓰지도 말라. 롤아웃 temperature는 둘과 독립적인 탐색 노브다. β를 다시 튜닝하기 전에 `~1.1`로 올려 보라. 그리고 DPO는 KL-free가 아니다. 단지 KL이 암묵적 보상 `r_θ = β·log(π/π_ref)` 안으로 흡수되어 있을 뿐이므로, 같은 과최적화 곡선이 적용된다.

---

## 1. 엔트로피 메커니즘 — Cui 2025의 법칙

[[entropy-mechanism-llm-rl]] (Cui, Zhang, Chen, Yuan et al., 2025, arXiv:2505.22617)은 "엔트로피 붕괴"를 구전 지식에서 예측 가능한 곡선으로 바꾼 논문이다. 20개가 넘는 모델 × 알고리즘 × 레시피 조합(PPO, GRPO, RLOO, Reinforce++)에서 같은 형태가 관찰된다. 토큰별 정책 엔트로피 `H(π)`는 SFT 초기값(2–3 nats)에서 대략 지수적으로 감소해 몇백 번의 업데이트 안에 0에 가까워지고, 달성 가능한 보상 `R`은 다음 식에 맞는다.

```
R(step) = −a · exp(H(step)) + b
```

여기서 `a, b`는 실행별로 양수다. 학습 초반 ~10% 구간으로 `a, b`를 맞추면 보상 상한을 *예측*할 수 있다. `H → 0`이면 `R → b − a`다. 그 이후에는 거의 결정론적인 정책을 조금 더 결정론적으로 만드는 데 그래디언트를 쓰는 셈이고, 엔트로피 nat 하나당 한계 보상은 사라진다.

논문의 메커니즘 부분은 경험적 부분보다 더 깔끔하다. advantage `A(s,a)`를 쓰는 policy-gradient 목적함수로 학습되는 softmax 정책 `π(a|s) ∝ exp(z_a(s))`에 대해, 토큰 엔트로피의 기대 1스텝 변화는 다음을 만족한다.

```
E[ΔH(s)] ∝ − Cov_{a~π(·|s)}( log π(a|s),  A(s,a) )
```

이 식이 이 장의 하중을 지탱한다. 한 상태에서 엔트로피가 log π(a|s)와 A(s,a) 사이의 공분산에 비례해 죽는다는 뜻이다. 이미 확률이 높은 토큰에 큰 advantage가 붙으면(양의 공분산) 엔트로피가 빠르게 탄다. 낮은 확률 토큰에 큰 advantage가 붙으면(음의 공분산) 엔트로피는 *올라간다*. A2C 스타일의 평평한 엔트로피 보너스 `+ c_H · H(π)`는 모든 토큰을 대칭적으로 다루므로, 얇은 꼬리에 있는 공분산 이상치들이 붕괴를 주도할 때 과소 보정한다. 바로 LLM 영역이 그렇다. LLM 규모의 vocabulary가 blanket-bonus 가정을 왜 깨는지는 [[entropy-regularization-ppo]]를 보라.

공분산 관점에서 두 가지 표적 개입이 나온다.

- **Clip-Cov.** 각 배치에서 토큰을 `p_t · A_t` 기준으로 순위화한다. 상위 일부 비율(논문의 기본값은 ~2%)의 그래디언트를 0으로 만든다. 나머지 분포는 계속 업데이트되고, 엔트로피를 붕괴시켰을 날카로운 spike만 침묵시킨다. 의사코드:

  ```python
  # per batch, per token, inside the policy-gradient backward pass
  cov_score = (log_prob.exp() * advantage).detach()     # p_t * A_t
  k = max(1, int(0.02 * cov_score.numel()))             # top 2%
  topk_mask = torch.zeros_like(cov_score, dtype=torch.bool)
  topk_mask.view(-1)[cov_score.view(-1).topk(k).indices] = True
  pg_loss = -(advantage * log_prob).masked_fill(topk_mask, 0.0).mean()
  ```

- **KL-Cov.** 같은 이상치 집합을 쓰되, 그래디언트를 0으로 만드는 대신 해당 토큰에만 토큰 수준 KL 페널티 `β_KL · k3(π_new, π_old)`를 적용한다. 그래디언트는 여전히 흐르지만 크기가 약해진다.

입증된 이득([[entropy-mechanism-llm-rl]]의 intervention 표): Qwen2.5-7B와 Qwen2.5-Math-7B에 GRPO를 적용했을 때, 전체 실행 동안 엔트로피가 `0.1`보다 의미 있게 높게 유지되고, AIME / MATH 정확도 상한이 vanilla GRPO보다 몇 포인트 오른다. 또한 vanilla에 평평한 엔트로피 보너스를 더한 방식보다도 높다. 논문은 그 방식이 과보정된다는 것을 보인다.

## 2. 붕괴 임계값과 triage tree

"엔트로피가 감소한다"는 언제 "엔트로피가 붕괴했다"가 되는가? 실무 임계값은 [[entropy-mechanism-llm-rl]], [[nathan-lambert-entropy-rl]], [[openrlhf-entropy-debugging]] 전반에서 수렴한다.

- **토큰별 엔트로피 `H < 0.1` nats가 여러 업데이트 동안 지속** → 붕괴.
- **마지막 토큰 분포에서 `H < 0.2` nats** → Lambert의 "멈추고 점검하라" 규칙. 보상이 아직 움직이더라도 위험 구역에 들어온 것이다.
- **`H`가 < 100 steps 안에 ≥ 30% 하락** → [[entropy-logging-patterns]]의 프레임워크 비의존 붕괴 시그니처. 보통 `ppo_kl > 0.1` 및 `clipfrac → 1`과 함께 나타난다.

두 regime은 곡선 중간 구간에 대해서는 의견이 다르다. [[entropy-collapse-ppo]](Andrychowicz 2020, MuJoCo)는 엔트로피 계수 튜닝을 2순위 노브로 둔다. advantage normalization과 PPO-clip 파라미터가 지배적이다. LLM 규모에서는 순서가 다르다. 소수의 높은 `p·A` 토큰이 붕괴를 주도하므로 평평한 보너스는 구조하지 못하고, 엔트로피 관련 레버가 우선순위 위로 올라온다.

엔트로피가 급락할 때의 커뮤니티 표준 triage([[openrlhf-entropy-debugging]]):

1. **참조 대비 KL이 켜져 있고 유한한지** 확인하라. `ppo_kl`의 `nan` 또는 실수로 0으로 설정된 β가 원인 1순위다.
2. **롤아웃 temperature**를 0.1–0.2 올려라(§5 참조). 이는 어떤 목적함수 계수도 바꾸지 않고 탐색을 다시 주입한다.
3. **엔트로피 계수** `c_H`를 한 자릿수 올려라(예: 0 → 1e-3, 1e-3 → 1e-2). 붕괴가 분포 bulk에서 온다면 작동한다. 얇은 꼬리가 주도한다면 대신 Clip-Cov가 필요하다.
4. **Advantage normalization**이 배치별 zero-mean unit-variance인지 확인하라. OpenRLHF / verl에서는 기본 ON, TRL에서는 OFF다([[entropy-logging-patterns]]).
5. 위 단계를 모두 거친 뒤에만 보상 신호를 의심하고, 그다음 재학습하라.

## 3. KL 추정량 — k1, k2, k3 유도

모든 LLM-RL 프레임워크의 KL 페널티는 Monte-Carlo 추정값이다. 샘플 `a ~ q`가 있고, 여기서 `q = π_new`는 현재 정책, `p = π_ref`(또는 `π_old`)는 비교 대상이다. 비율 `r(a) = p(a) / q(a)`이면 `KL(q‖p) = E_q[−log r]`다. 문제는 그 기댓값에 대해 *어떤 one-sample 추정량을 쓸 것인가*다. [[john-schulman-kl-tricks]]는 표준 삼총사를 제시한다.

**k1 — 직접 추정량.** `k1 = −log r`. 정의상 편향이 없다. `E_q[−log r] = KL(q‖p)`. 문제:
  - 특정 샘플 `a`에서 `p(a) > q(a)`이면 단일 샘플 값이 음수가 될 수 있다. 실제 KL은 ≥ 0인데도 그렇다. 대시보드에서 Monte-Carlo KL이 가끔 음수가 되는 것은 버그가 아니라 k1이다.
  - 분산이 높다. 평균은 같고 분산만 다른 Gaussian `q, p`를 생각하라. 대부분의 샘플은 중간 정도의 `|log r|`을 주지만, 드문 tail 샘플은 거대한 `|log r|`을 낸다.

**k2 — 제곱 로그 추정량.** `k2 = ½·(log r)²`. 동기: `r = 1` 주변에서 `−log r`를 Taylor 전개한다. `r = 1 + ε`라고 쓰면,

```
−log(1 + ε) = −ε + ε²/2 − ε³/3 + ...
E_q[−log r]   = 0 + ½·E_q[ε²] − (1/3)·E_q[ε³] + ...
```

왜냐하면 `E_q[ε] = E_q[r − 1] = E_q[p/q] − 1 = ∫ p dx − 1 = 0`이기 때문이다. 따라서 leading order에서 `KL ≈ ½·E_q[ε²] = ½·E_q[(log r)²]`다(두 번째 moment에서 squared-log로 바꾸는 것도 또 하나의 Taylor 단계다). 그래서 k2는 **편향은 있지만 저분산**인 추정량이다. 평균이 정확히 `KL(q‖p)`는 아니지만 O(ε²)까지 추적하고, `ε²`에는 k1의 무거운 음의 tail이 없으므로 분산은 k1보다 훨씬 작다.

**k3 — 볼록한 비편향 추정량.** `k3 = r − 1 − log r`. 종이에 확인할 수 있는 성질 두 가지가 있다.

- *비편향.* `E_q[r − 1 − log r] = E_q[r] − 1 + KL(q‖p) = (∫ p dx) − 1 + KL = 0 + KL = KL`.
- *비음수.* `f(r) = r − 1 − log r`라고 두자. 그러면 `f(1) = 0`, `f'(r) = 1 − 1/r`, `f''(r) = 1/r² > 0` for `r > 0`. 따라서 `f`는 `r = 1`에서 최솟값 `0`을 갖는 엄격히 볼록한 함수이고, 모든 양의 `r`에 대해 `f(r) ≥ 0`이다. 단일 k3 샘플은 절대 음수가 되지 않는다.

`r = 1` 근방의 분산 비교: `f(r) = ½·(r−1)² − (1/3)·(r−1)³ + ...`로 전개된다. `ε = r − 1`라고 쓰면 `f ≈ ½·ε²`다. 한편 k2 ≈ ½·(log(1+ε))² ≈ ½·(ε − ε²/2)² = ½·ε² − ½·ε³ + O(ε⁴). 따라서 k2와 k3는 `ε`의 leading order에서 일치한다. k3가 k2보다 나은 점은 zero bias이고, k3가 k1보다 나은 점은 부호가 제한되어 있으며 정책과 참조가 가까울 때(첫 몇 PPO 업데이트 이후의 운용 regime) 분산도 더 작다는 것이다.

세 추정량을 표로 정리하면:

| Estimator | Formula | Unbiased | Sign | Notes |
|-----------|---------|----------|------|-------|
| `k1` | `−log r` | Yes | any | 분산이 높음; 샘플별로 음수가 될 수 있음 |
| `k2` | `½·(log r)²` | No | ≥ 0 | `r ≈ 1` 근처에서만 낮은 편향; 안정적인 모니터 |
| `k3` | `r − 1 − log r` | Yes | ≥ 0 | 볼록, 비음수, GRPO의 선호 기본값 |

Costa Huang의 주의점([[john-schulman-kl-tricks]]): 초기 TRL 실험에서 `k3`가 "어떤 이유로 폭발했다". 가능한 원인은 tail에서 매우 큰 `r`이다. `r − 1`은 선형으로 증가하지만 `−log r`은 로그로 증가하므로, tail 이벤트 하나가 큰 양의 k3 값을 더한다. 실무자들이 수렴한 해결책은 verl처럼 *지수화 전에 log-ratio를 `[−20, 20]`으로 clamp*하는 것이다([[entropy-logging-patterns]] verl excerpt).

## 4. KL-to-reward vs KL-as-loss

KL 페널티를 주입하는 위치는 두 곳이며, 둘은 서로 바꿔 쓸 수 없다.

**KL-to-reward (PPO / InstructGPT 관례).** [[kl-control-rlhf]]는 Jaques 2019 → Stiennon 2020 → Ouyang 2022(InstructGPT)로 이어지는 계보를 형식화한다. PPO advantage 추정기가 보는 토큰별 보상은 다음처럼 shaping된다.

```
r̂_t = r_t − β · ( log π_φ(y_t | y_<t, x) − log π_ref(y_t | y_<t, x) )
```

여기서 `r_t`는 RM 보상(보통 terminal token을 제외하면 0)이고 KL 항은 모든 토큰에 적용된다. 그러면 advantage 추정량 `A_t = Σ_t' γ^{t'−t} · (r̂_{t'} + ...)`가 KL 비용을 토큰별, 시간적으로 자연스러운 방식으로 policy gradient에 실어 나른다. InstructGPT는 `β ≈ 0.02`를 보고한다. [[openrlhf-entropy-debugging]]에 따르면 프로덕션 스택은 보상 스케일의 `0.01–0.1` 범위에서 β를 운용한다. [[kl-control-rlhf]]는 이렇게 강조한다. *KL을 손실이 아니라 보상에 더하라.* 그렇지 않으면 advantage 기반 policy gradient가 깨지고 경험적으로 학습이 나빠진다.

**KL-as-loss (GRPO / DPO 관례).** 최신 GRPO(DeepSeekMath 2024, [[entropy-logging-patterns]] TRL GRPO excerpt)는 KL 항을 토큰별 손실에 직접 둔다.

```python
# trl/trainer/grpo_trainer.py
per_token_kl  = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1   # k3
per_token_loss = per_token_loss + self.beta * per_token_kl
```

보상 shaping도, AdaptiveKLController도 없다. 정당화는 (a) GRPO의 group-relative advantage가 baseline 없이도 이미 잘 정의되어 있어 KL을 보상 채널로 우회시켜 "보호"할 대상이 없고, (b) k3-in-loss가 정확하고 비음수인 페널티를 안정적인 그래디언트로 제공한다는 것이다.

**Korbak의 Bayesian 관점**([[kl-control-rlhf]], Korbak 2022)은 둘을 통합한다. `argmax_π E_π[r] − β·KL(π‖π_ref)`의 closed-form optimum은

```
π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )
```

즉 RL-with-KL 목적함수는 tilted posterior `π*`에 대한 *정확한 variational inference*다. `β`는 tilt 강도, `π_ref`는 prior, `r/β`는 log-likelihood다. DPO도 같은 tilt를 직접 물려받는다. DPO의 암묵적 보상은 `r_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))`이므로, DPO 모델을 학습하는 것은 RLHF-PPO에 등장하는 같은 β로 *암묵적으로* KL-정규화된 보상을 최적화하는 것과 같다. 그래서 DPO의 과최적화 곡선이 PPO와 닮는다. 같은 목적함수, 다른 추정량이다.

**방향이 중요하다.** 위 페널티는 `KL(π_new ‖ π_ref)`로 쓰인다. 참조의 관점에서는 *reverse* KL이다. Reverse KL은 mode-seeking이다. `π_new`가 `π_ref`의 support 전체를 덮게 하기보다, `π_ref`가 질량을 두는 곳에만 질량을 두도록 민다. 이것이 RLHF 튜닝 모델이 출력 분포를 "넓히기"보다 "날카롭게" 만드는 형식적 이유다. 또한 엔트로피 붕괴 이야기와 KL 페널티 이야기가 분리되는 이유이기도 하다. `KL(π‖π_ref)`를 조여도 반드시 `H(π)`가 줄어드는 것을 막지는 못한다. 참조 자체가 많은 상태에서 꽤 뾰족할 수 있기 때문이다.

## 5. Max-ent RL의 계보 — LM-RL이 물려받은 것과 버린 것

현대 LLM-RL의 모든 엔트로피 노브는 Soft Actor-Critic([[maximum-entropy-rl]], Haarnoja 2018)에서 내려온다. SAC의 목적함수는 다음과 같다.

```
J(π) = Σ_t E_{(s_t,a_t)~ρ_π} [ r(s_t, a_t) + α · H(π(·|s_t)) ]
```

보상 단독이 아니라, 보상 *더하기* 스텝별 엔트로피 보너스다. soft-Bellman 방정식은 `max_a Q`를 log-sum-exp로 바꾼다.

```
V(s) = α · log ∫ exp( Q(s, a) / α ) da
Q(s, a) ← r(s, a) + γ · E_{s' ~ p}[ V(s') ]
```

그리고 최적 정책은 Boltzmann 형태 `π*(a|s) ∝ exp( Q(s,a) / α )`다. 이는 §4의 Korbak RLHF posterior와 같은 모양이며, `Q/α`가 `r/β`의 역할을 한다. SAC-v2는 자동 α 튜닝을 추가한다. target entropy `H̄`(continuous control에서는 `H̄ = −dim(A)`)를 정하고, `L(α) = E[−α·(log π(a|s) + H̄)]`에 대해 gradient descent를 수행해 학습 과정에서 `H(π) ≈ H̄`를 유지한다.

LLM-RL이 물려받은 것:

- **손실 항 자체.** A3C/PPO([[entropy-regularization-ppo]])는 SAC의 on-policy, 작은-α 극한으로서 보너스 `+ c_H · H(π)`를 그대로 싣는다. TRL의 `entropy_coef`, OpenRLHF의 `c_H`, verl의 entropy registry가 모두 이 계보에서 내려온다.
- **Target-entropy 아이디어.** Cui 2025의 공분산 개입은 SAC-v2가 대칭적으로 했던 일을 비대칭적으로 재발견한 것이다. 정규화 강도를 온라인으로 조정해 H를 floor 위에 유지한다.
- **Boltzmann-tilt 관점.** Korbak의 `π* ∝ π_ref · exp(r/β)`는 SAC의 `π* ∝ exp(Q/α)`와 구조적으로 동일하다. RLHF의 참조 정책은 SAC의 uniform prior 역할을 한다.

LM-RL이 버린 것:

- **Off-policy replay.** SAC는 off-policy다(replay buffer + twin critic을 쓰는 soft Q-learning). LLM-RL은 주로 on-policy다(PPO/GRPO with fresh rollouts). 엄청난 context / action space에서 안정성을 얻기 위해 sample efficiency를 포기한 거래다. async actor-learner가 왜 진짜 off-policyness를 복원하지 못하는지는 [[openrlhf-entropy-debugging]]을 보라.
- **Soft-Q critics.** 최신 GRPO 계열 방법([[entropy-logging-patterns]])에는 value head가 전혀 없다. advantage는 group-relative z-score다. 엔트로피 정규화는 이제 actor loss 또는 reward stream에만 존재한다.
- **자동 α 튜닝.** 프로덕션 LLM-RL은 온라인 temperature controller인 SAC-v2 대신 고정 β와 고정 c_H(거친 schedule 포함)를 사용한다. Cui 스타일 Clip-Cov / KL-Cov가 가장 가까운 적극적 재발명이다.
- **진짜 max-ent 의미론.** 손실 항으로서 `+ c_H · H(π)`는 max-ent RL의 작은-α 극한일 뿐이다. 전체 soft-Bellman recursion(Q target 안의 엔트로피, soft-max로서 log-sum-exp)은 어떤 프로덕션 LLM 스택에서도 쓰이지 않는다.

롤아웃 temperature([[sampling-temperature-schedule]])는 세 번째 독립 레버다. `P_T(a|s) = softmax(z(s)/T)`는 *샘플 시점*에 logits를 재스케일링하며 파라미터는 바꾸지 않는다. 따라서 목적함수 편향을 도입하지 않고 롤아웃을 넓힌다. 실용 레시피: R1은 `T = 1.0, top_p = 0.95`를 쓴다. Tülu 3는 RLVR 동안 `T = 1.0`을 쓰다가 마지막 DPO polish에서 `T = 0.7`로 anneal한다. 붕괴 시 OpenRLHF의 re-warm 규칙은 다른 것을 다시 튜닝하기 전에 `N`개의 롤아웃 동안 `T`를 0.2 올리는 것이다. 학습 중에는 `top_p < 1.0`과 `top_k`를 피하라. 이들은 PPO clip이 흡수할 수 없는 비미분 support truncation을 도입한다.

## 6. 네 레버의 mental model과 상호작용

앞의 다섯 절은 각각 하나의 레버를 소개했다. 실무적 결정은 어느 레버를 먼저 잡을 것인가이며, 답은 보이는 failure mode에 달려 있다. 한 표로 요약하면:

| Lever | Where it lives | Primary failure it addresses | Collapse effect | Side effect |
|-------|----------------|------------------------------|-----------------|-------------|
| Entropy bonus `c_H · H(π)` | actor loss | bulk-distribution premature sharpening | LLM 규모에서는 약함 | 질량을 낮은 `p·A` 토큰 쪽으로 밈 |
| Clip-Cov / KL-Cov | actor loss on top-`k%` tokens | 공분산 주도 tail 붕괴 | 강함; 외과적 | 보상 상한 도달이 약간 지연 |
| KL-to-reference `β · KL(π‖π_ref)` | reward stream (PPO) or loss (GRPO) | SFT prior에서의 drift; 보상 해킹 | 간접적(참조 자체가 뾰족할 수 있음) | β가 너무 크면 전체 보상 상승을 늦춤 |
| Rollout temperature `T` | sampler | 학습 후반의 정체된 탐색 | 직접적; 즉각적 | IS correction 없이 T ≠ 1이면 off-policy bias |

레버들은 대체로 독립적이지만 직교하지는 않는다. `T`를 올리면 고정된 파라미터에서 표본 분포가 넓어지므로 effective `KL(π_sampled ‖ π_ref)`가 커진다. 이 때문에 adaptive-KL controller가 β를 낮추도록 강제될 수 있고, 참조 제약이 간접적으로 느슨해진다([[openrlhf-entropy-debugging]] triage step 2가 이를 활용한다). Clip-Cov와 엔트로피 보너스는 대략 합성된다(서로 다른 토큰 집합에 작동한다). 그러나 둘 다 엔트로피를 건드리므로, 재튜닝 없이 함께 돌리는 것은 흔한 과보정이다. KL-to-reference와 KL-Cov는 같은 정책에 걸리는 KL 항이다. 기본 β 값으로 둘 다 적용하면 어느 한쪽의 이득도 상쇄될 수 있다.

**이 표에서 DPO의 위치.** DPO는 *KL-free*가 아니다. 손실 `L_DPO = −log σ(β · (log π(y_w|x) − log π(y_l|x) − log π_ref(y_w|x) + log π_ref(y_l|x)))`는 `r_θ = β · log(π/π_ref)`를 대입한 RLHF 목적함수와 정확히 같다. 따라서 β 계수는 PPO에서와 같은 KL-budget 역할을 한다. 실용적 결과: DPO는 같은 `β · KL(π‖π_ref)` 과최적화 곡선을 물려받는다(ch-39의 [[dpo]] / ch-41의 [[reward-model-overoptimization]] 논의가 적용된다). 대신 토큰별 진단 대시보드를 잃는다. 롤아웃이 없기 때문에 step별 `ppo_kl`을 그릴 수 없다. Lambert([[nathan-lambert-entropy-rl]])는 DPO의 암묵적 KL이 reference를 loss shape에 인코딩하므로 엔트로피 붕괴를 *부분적으로* 막지만, 작은 pair set에서는 preference over-fitting이라는 고유 failure mode를 도입한다고 지적한다.

**엔트로피와 보상 해킹은 다른 축이다.** [[openrlhf-entropy-debugging]]에서 주의하라는 failure pattern 중 하나는 "entropy healthy, rollouts exploding in length"다. 이는 ch-42의 reward-hacking 시그니처다. 정책이 reward의 허점(장황함, refusal, format padding)을 찾았지만 분포 폭은 건드리지 않은 것이다. §4의 KL-to-reference 항은 *참조가 그 자체로 잘 행동한다는 한도에서* 이것을 방어하지만, 엔트로피 붕괴를 해결하지는 않는다. 반대로 Clip-Cov는 엔트로피 붕괴를 고치지만 length hacking을 방어하지 않는다. 이들은 직교하는 failure axis이므로 직교하는 진단이 필요하다. 대시보드는 토큰별 엔트로피와 response-length histogram을 모두 포함해야 하며, 길이를 보지 않은 붕괴 진단은 불완전하다.

---

## Connections

- **ch-38** (KL-Controlled RLHF) — PPO clip + InstructGPT의 KL-to-reward 관례를 유도한다. ch-43은 k1/k2/k3와 collapse-plus-triage 관점으로 그 흐름을 확장한다.
- **ch-40** (Online / Group-Baseline RL) — GRPO는 여기 문서화된 k3-in-loss를 사용한다. Dr.GRPO bias correction은 엔트로피 동역학과 직교한다.
- **ch-41 / ch-42** (Reward Modeling / Reward Hacking) — §4의 KL budget은 ch-41에서 도입한 과최적화 곡선을 제한한다. 엔트로피 붕괴는 보상 해킹과 별개의 failure axis이지만 종종 함께 발생한다.
- **ch-44** (Process Supervision / RLVR) — 검증 가능한 보상 실행에도 엔트로피 제어가 필요하다. DeepSeek-R1의 긴 롤아웃은 [[sampling-temperature-schedule]]에 따른 탐색 budget 역할도 한다.
- **ch-55** (verl internals) — `kl_penalty` switch와 `actor/entropy` logging은 ch-55의 tour가 `core_algos.py`에 도달했을 때 읽게 될 파일이다.

## Further reading

- [[entropy-mechanism-llm-rl]] — `R = −a·exp(H) + b` 법칙, Clip-Cov / KL-Cov.
- [[entropy-collapse-ppo]] — Andrychowicz 2020 대규모 PPO sweep; failure mode로서의 붕괴.
- [[entropy-regularization-ppo]] — A3C/PPO 엔트로피 보너스 손실 형태, LLM 규모로의 carry-through.
- [[maximum-entropy-rl]] — SAC와 auto-α; 조상 목적함수.
- [[john-schulman-kl-tricks]] — k1/k2/k3 유도와 실무 주의점.
- [[kl-control-rlhf]] — Jaques / Stiennon / Ouyang / Korbak 계보, KL-as-reward 관례.
- [[nathan-lambert-entropy-rl]] — 2025년에 엔트로피가 병목인 이유에 대한 실무자 종합.
- [[openrlhf-entropy-debugging]] — 크로스 프레임워크 triage protocol.
- [[entropy-logging-patterns]] — verl / OpenRLHF / TRL 엔트로피와 KL logging 비교.
- [[sampling-temperature-schedule]] — 독립적 탐색 레버로서 rollout-T.

## Companion visualization

**[figures/entropy-dynamics.html](figures/entropy-dynamics.html)** — 대화형 two-panel figure. Panel 1: intervention(none / Clip-Cov / KL-Cov / flat entropy bonus)을 선택하고 `H < 0.1` 붕괴 임계값이 겹쳐진 entropy-vs-step 곡선을 본다. Panel 2: toggle 가능한 `q` vs `p` regime에서 1000개의 `r` 값을 샘플링하고 k1 / k2 / k3 추정량 분산을 나란히 관찰한다. 곡선은 설명용이며 [[entropy-mechanism-llm-rl]]와 [[john-schulman-kl-tricks]]에서 입증된 정성적 형태와 맞춘 것이다. 절대 수치는 벤치마킹이 아니라 교육용이다.
