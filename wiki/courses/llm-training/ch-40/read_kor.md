<!-- chapter: ch-40
     track: rl
     kind: content
     title: Online / Group-Baseline RL Family
     deps: [ch-39]
     sources: [[rloo]], [[reinforce-plus-plus]], [[grpo]], [[dr-grpo]], [[deepseekmath]], [[rloo-vs-grpo]], [[trl-grpo]], [[verl-grpo]], [[nathan-lambert-grpo]], [[on-off-policy-rlhf]]
     figures: figures/group-baseline.html
-->

# 제40장 — Online / Group-Baseline RL Family

> **핵심 통찰.** 2024년에 RLHF 분야는 조용히 PPO critic을 죽였다. Ahmadian의 RLOO는 LLM에서는 — deterministic token transition, full-sequence reward, one-epoch update — 학습된 value head가 사중량임을 보였다. 그것을 *다른 k-1개 peer sample의 평균*으로 바꾸면 REINFORCE가 절반의 메모리로 PPO를 이긴다. 네 달 뒤 DeepSeekMath의 GRPO는 같은 아이디어를 가져와 group을 G=64로 키우고, group std로 normalize하며, KL을 unbiased k3 estimator로 loss 안에 넣었다. R1은 이것으로 출시됐다. 그리고 2025년 3월 Dr.GRPO는 GRPO loss에 wrong-and-long output을 reward하는 두 개의 length-biased division이 있음을 증명했고, fix는 그것을 *삭제하는 것*이었다. 이 장의 arc는 하나의 subtraction이다. PPO minus critic minus std denominator minus 1/|o_i| equals "peer baseline을 가진 REINFORCE", 이것이 오늘날 전체 field가 실제로 돌리는 것이다.
>
> **지침.** verifiable 0/1 reward가 있는 reasoning RL(DeepSeek-R1 setting)에는 기본적으로 **Dr.GRPO**를 써라. 메모리가 빡빡하고 reward가 continuous RM score이면 k=2–4의 **RLOO**를 써라. prompt당 k=1만 감당할 수 있지만 큰 global batch가 있으면 **REINFORCE++**를 써라. heterogeneous reward spread를 가진 prompt들 사이에서 std-normalization의 variance-magnitude control이 꼭 필요할 때만 vanilla **GRPO**를 고려하라. 그때도 매 epoch `mean(|o_wrong|) − mean(|o_right|)`를 log하라. 커지면 length bias가 활성화된 것이고 switch해야 한다.

---

## §1 LLM-land에서 PPO critic의 문제

잠시 [[ppo]]로 돌아가자. PPO의 advantage는 `Â_t = δ_t + (γλ) δ_{t+1} + …`이며 `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`다. value head `V_φ`는 mean-squared target `(V_φ(s_t) − R_t)²`로 학습되는 두 번째 full-size network다. Atari나 MuJoCo에서는 괜찮다. episode는 수천 개의 stochastic transition이고, V의 bootstrap bias는 그것이 제거하는 분산에 묻힌다.

LLM RLHF에서는 세 가정이 동시에 깨진다(Ahmadian §3, [[rloo]]).

1. **Deterministic transitions** — `y_{<t}`가 고정되면 `y_t`는 다음 state를 유일하게 결정한다. 추정해야 할 "stochastic dynamics 위의 expected return"이 없다. 유일한 randomness는 자기 자신의 sampling이다.
2. **Full-trajectory reward** — RM 또는 verifier는 end-of-sequence에서 scalar 하나를 준다. intermediate `r_t`가 없다. GAE의 `(γλ)`-weighted delta는 `R − V(s_0)`, 즉 하나의 learned baseline으로 붕괴한다.
3. **One epoch per rollout** — TRL/verl/OpenRLHF는 모두 기본값이 `μ = 1`이다. PPO clip은 multiple epoch 동안의 drift를 막기 위해 존재한다. one epoch에서는 clip이 거의 bind되지 않는다.

따라서 PPO critic은 *하나의* 일을 한다. `V(s_0) = V(prompt)`를 학습한다. 그런데 `V(prompt) ≈ E_{y ∼ π}[R(prompt, y)]`이고, 이 expectation의 unbiased minimum-variance estimator는 prompt당 하나 이상의 rollout을 sample하면 이미 공짜로 갖고 있는 **같은 prompt의 다른 sample들의 평균 reward**다. critic은 이미 있는 quantity의 fragile learned approximation이다. 삭제하라.

---

## §2 RLOO — leave-one-out identity

`k`개의 response `y_1, …, y_k ∼ π_θ(·|x)`를 sample한다. reward `R_i = R(x, y_i)`를 계산한다. constant baseline `b`를 가진 REINFORCE gradient는:

```
∇J ≈ (1/k) Σ_i (R_i − b) · ∇ log π_θ(y_i | x)
```

`y_i`와 독립인 baseline은 gradient를 unbiased로 남긴다(고전 policy-gradient 결과, [[vanilla-pg]]). sample마다 `b_i`를 따로 고르자. 유일한 요구사항은 `x`가 주어졌을 때 `b_i ⊥ y_i`다. 그 독립성을 지키면서 분산을 가장 많이 줄이는 선택이 **leave-one-out mean**이다.

```
b_i = (1/(k−1)) Σ_{j ≠ i} R_j
```

각 `b_i`는 `{R_j : j ≠ i}`의 함수이고, 이는 `{y_j : j ≠ i}`에 의존하지만 `y_i`에는 의존하지 않는다. 따라서 구성상 unbiased다. 대입하면:

```
∇J_RLOO ≈ (1/k) Σ_i [ R_i − (1/(k−1)) Σ_{j ≠ i} R_j ] · ∇ log π_θ(y_i | x)
```

두 가지 sanity check. (a) **k=2 case**: `b_1 = R_2`, `b_2 = R_1`이므로 sample 1의 advantage는 `R_1 − R_2`, sample 2의 advantage는 `R_2 − R_1`이다. 순수 pairwise comparison이다. 그래서 k=2 RLOO는 종종 "log-sigmoid 없는 online DPO"라고 설명된다. (b) **Large-k limit**: `b_i → mean(R)`이므로 RLOO의 advantage는 `→ R_i − mean(R)`이 된다. 이는 정확히 Dr.GRPO의 advantage(§5)다. RLOO와 Dr.GRPO는 O(1/k) correction을 제외하면 같은 estimator다.

PPO 대비 삭제되는 것([[rloo]] Table):

| Component | PPO | RLOO |
|-----------|-----|------|
| Value network | required | **removed** → ~50% memory |
| GAE | `γλ` weighted δ sum | 필요 없음(full-seq reward) |
| Clip ε | yes | **no** (1 epoch) |
| Epochs per rollout | 4 (classic), 1 (LLM) | 1 |
| Baseline | learned V_φ | k peer 위의 leave-one-out |
| KL location | token별 shaped reward | token별 shaped reward (same) |

경험적으로 TL;DR summarization과 HH-RLHF helpfulness에서 RLOO k=4는 모든 KL budget에서 PPO Pareto frontier를 지배한다(Ahmadian Fig. 3). 메시지는 이것이다. LLM RLHF에서 PPO의 overhead는 feature가 아니라 tax다.

---

## §3 REINFORCE++ — k=1 regime을 위한 global normalization

Jian Hu 2025([[reinforce-plus-plus]])는 같은 논리를 한 단계 더 밀었다. prompt당 k ≥ 2 rollout을 감당할 수 없다면? prompt당 sample이 하나면 RLOO의 peer baseline은 정의되지 않는다. Fix는 prompt 안이 아니라 **prompt들의 전체 mini-batch**에서 normalize하는 것이다.

**Per-token shaped reward**(InstructGPT-style PPO와 동일):

```
r̃_t = R(x, y) · 𝟙{t = T} − β · KL_t,    KL_t = log π_θ_old(y_t|·) − log π_ref(y_t|·)
```

step t부터의 **Cumulative return**: `G_t = Σ_{t'≥t} r̃_{t'}` with γ = 1.

full batch B 위의 **Global advantage normalization**:

```
Â_t = (G_t − mean_{B}(G)) / std_{B}(G)
```

mean과 std는 batch 안의 모든 `(prompt, response, token)` triple에 걸쳐 계산한다. prompt별이 아니다. batch size 512–2048 sequences면 std estimate는 단단하다. k=1의 prompt 내부에서는 wild할 것이다.

**Loss**는 표준 PPO-clip surrogate를 그대로 재사용한다.

```
L(θ) = − E_t[ min(ρ_t(θ) Â_t, clip(ρ_t(θ), 1-ε, 1+ε) Â_t) ]
```

여기서는 RLOO와 달리 clip을 유지한다. REINFORCE++는 token별 drift의 위험이 있기 때문이다. advantage magnitude가 크면 단일 token의 ratio가 한 step 안에 1.0에서 멀리 뛸 수 있고, ε=0.2가 그것을 bound한다. value net 없음, k=1, global normalization, loss가 아니라 reward에 KL. 메모리 효율적이고 안정적이며 2025년 중반 OpenRLHF에서 mainstream이 되었다.

---

## §4 GRPO — DeepSeekMath의 group z-score

Shao et al. 2024([[grpo]], [[deepseekmath]])는 group-baseline 아이디어를 큰 G로 가져가고 두 가지를 더했다. advantage를 **std-normalize**하고, KL을 k3 estimator로 **loss 안에** 옮긴다.

**Rollout.** batch의 prompt q에 대해 `o_1, …, o_G ∼ π_θ_old`를 sample한다. RM이 `r_i`를 준다.

**Advantage (outcome supervision).** `o_i`의 모든 token은 같은 advantage를 갖는다.

```
Â_{i,t} = (r_i − mean(r_1, …, r_G)) / std(r_1, …, r_G)
```

왜 std-normalize하는가? Prompt A의 reward는 `{0.1, 0.11, 0.12}`(모든 rollout이 거의 equivalent)이고, prompt B의 reward는 `{0.0, 0.5, 1.0}`(high variance)라고 하자. std 없이 B의 gradient magnitude는 A의 약 10배지만, B의 *relative* signal이 더 informative한 것은 아니다. std로 나누면 prompt별 gradient magnitude가 equalize되어 optimizer가 high-variance prompt에 모든 update를 집중하지 않는다. continuous RM score이고 prompt마다 "difficulty spread"가 다를 때 도움이 된다.

**Objective (Eq. 3 of [[grpo]]).**

```
J_GRPO(θ) = E[q, {o_i}] (1/G) Σ_i (1/|o_i|) Σ_t {
    min[ ρ_{i,t} Â_{i,t},  clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t} ]
    − β · D_KL^k3(π_θ || π_ref)
}
```

여기서 `ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})`다.

**k3 KL estimator (Eq. 4).** [[john-schulman-kl-tricks]] 문헌에서:

- **k1:** `log(π_θ/π_ref)` — low variance, 하지만 divergence가 ≥0이어야 하는데 sign이 biased되어 negative가 될 수 있다.
- **k2:** `½·(log ratio)²` — unbiased지만 제곱 때문에 항상 positive라 sign information이 사라진다.
- **k3:** `π_ref/π_θ − log(π_ref/π_θ) − 1` — unbiased, **항상 ≥0**(convex Bregman distance). GRPO는 k3를 쓴다.

유도 sketch: `x = log(π_ref/π_θ)`라고 하자. 그러면 `k3 = e^x − x − 1`이다. x=0 주변에서 Taylor-expand하면 `e^x − x − 1 = x²/2 + x³/6 + …`. 작은 KL(RL regime)에서는 k3 ≈ k2 = x²/2다. drift가 커지면 k3는 선형으로 자라고 k1 estimator를 아래에서 bound한다. reference forward pass 하나가 추가되고, logprob과 같은 shape의 tensor이며, positivity가 보장된다.

**KL-in-loss vs KL-on-reward.** RLOO와 REINFORCE++는 advantage 계산 전에 β·KL을 token별 reward에 넣는다. GRPO는 reward를 건드리지 않고 token별 loss에 `−β · KL_t`를 더한다. 수치적으로 미묘하게 다르다. on-reward KL은 advantage normalization을 통과한다(std가 그것을 나눈다). in-loss KL은 그렇지 않다. in-loss 형태는 *advantage*를 순수하게 outcome reward의 함수로 유지하므로, verifiable 0/1 task에 더 깔끔하다.

**Paper recipe** (MATH / GSM8K, [[deepseekmath]]): G=64, ε=0.2, β=0.04, LR 1e-6, batch 1024 prompts, T=1.0 sampling, max 1024 tokens, π_ref frozen SFT. 결과: MATH 51.7%(PPO 51.0, RFT 49.0, SFT 46.8에서). GRPO는 DeepSeek가 R1-Zero와 R1에 사용한 loss다.

---

## §5 Dr.GRPO — 어떤 division이 bias였고 왜인가

Liu et al. 2025([[dr-grpo]])는 reasoning RL 동안 compound되는 GRPO의 두 bias를 확인했다. response별 token 평균 `1/|o_i|`와 advantage의 `/std(r)`다. 둘 다 *겉으로는 무해해 보이는* division이지만 실제로는 unbiased가 아니다.

### Bias 1: `1/|o_i|` per-response mean

GRPO의 loss는 각 rollout의 token별 loss를 realized length `|o_i|`로 평균한다. 같은 advantage `Â = −1.0`, 같은 token별 ratio `ρ_t ≈ 1`을 갖는 두 incorrect response를 생각하자.

- **Short wrong:** `|o_1| = 50`. token별 loss ≈ `+1`. `(1/50) · Σ_t ≈ +1`로 aggregate된다. 전체 gradient contribution은 full magnitude에 가깝다.
- **Long wrong:** `|o_2| = 500`. token별 loss는 여전히 ≈ `+1`. `(1/500) · Σ_t ≈ +1`로 aggregate된다. aggregate loss는 같지만 10배 많은 token에 분산된다.

aggregate loss는 동일해 보이므로 *sequence별* gradient magnitude도 동일하다. 하지만 long wrong rollout에서 token당 gradient는 10배 작다. 따라서 optimizer는 `o_2`의 각 token logprob을 `o_1`보다 10배 덜 움직인다. 잘못된 response 안에서 repetition은 사실상 공짜가 된다. wrong token 200개를 더해도 aggregate sequence loss는 바꾸지 않으면서 token당 penalty를 희석하기 때문이다.

**Training에서의 net effect:** wrong-and-long response는 underpenalized되고, `|o_wrong|`가 단조롭게 커지며, chain-of-thought가 rambling filler가 된다. Dr.GRPO Figure 1은 vanilla GRPO에서는 이 length curve가 위로 치솟지만 corrected version에서는 flat하게 유지됨을 보여 준다.

Fix는 `(1/|o_i|)`를 **fixed constant** `(1/L_max)`로 바꾸는 것이다. 예를 들어 generation budget인 4096이다. 이제 response가 실제로 얼마나 길었는지와 무관하게 모든 token의 loss contribution은 같은 absolute magnitude를 갖는다. 긴 wrong rollout은 길이에 비례해 더 많은 penalty를 누적한다. 이것이 [[trl-grpo]]의 `loss_type="dr_grpo"`다.

```python
# dr_grpo aggregation (TRL L2418+)
loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
```

`loss_type="grpo"`와 비교하면:

```python
# grpo aggregation (TRL L2418+)
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
```

유일한 변화는 denominator다. 숫자 하나다. 그것이 bias correction이다.

### Bias 2: `/std(r)` difficulty weighting

같은 batch의 두 prompt를 보자. prompt A는 `r = {1, 1, 1, 0, 0, 0, 0, 0}`(std ≈ 0.52, mean = 0.375)이고, prompt B는 `r = {1, 0, 0, 0, 0, 0, 0, 0}`(std ≈ 0.35, mean = 0.125)이다. 둘 다 informative하다. A에서는 rollout의 절반이 맞았고, B에서는 하나만 맞았다. raw advantage `(r_i − mean)`는 prompt 사이에서 다르지만, std-normalization은 advantage를 prompt 사이에서 **비슷한 magnitude**로 만든다.

도움이 되는 것처럼 들리지만 pathological edge를 생각하라. 모든 G rollout이 r=0(너무 어려운 prompt) 또는 r=1(너무 쉬운 prompt)이면 std → 0이고 advantage는 0/0 = undefined가 된다. 구현은 `+ ε = 1e-6`으로 clip하는데, 그러면 advantage가 *enormous*해지고 불안정해진다. 더 나쁜 점은 중간 regime에서 우연히 std가 작은 prompt를 over-weight한다는 것이다. easy-or-hard prompt가 update를 지배하고, rollout이 갈라지는 가장 informative한 prompt에 집중하는 대신 바로 그 prompt들의 current reward distribution에서 policy를 밀어낸다.

Fix: `/std`를 완전히 버린다.

```
Ã_{i,t} = r_i − mean(r_1, …, r_G)          # Dr.GRPO, unbiased
```

이는 대수적으로 RLOO의 `k`-large-limit이다. across-prompt gradient-magnitude equalization은 잃지만, single-prompt reward collapse에 대한 invariance를 얻고 difficulty-weighting bias를 제거한다. `r ∈ {0, 1}`인 verifiable-reward RL에서는 `/std`를 버리는 것이 엄밀히 개선이다.

### Dr.GRPO loss (putting it together)

```
J_Dr.GRPO(θ) = E[ (1/G) Σ_i (1/L_max) Σ_t min(ρ_{i,t} Ã_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε) Ã_{i,t})
                   − β · D_KL^k3(π_θ || π_ref) ]
```

다른 모든 곳은 GRPO와 같다. 같은 ρ, 같은 clip, 같은 k3 KL. 다만 `Â_{i,t} = (r_i − mean)/std → Ã_{i,t} = r_i − mean`이고 `(1/|o_i|) → (1/L_max)`다. 두 개를 삭제한 것이다.

Empirical result(Dr.GRPO Table 2 / Figure 1): Qwen2.5-Math-7B on MATH/AIME/AMC에서 vanilla GRPO와 맞먹거나 더 나은 accuracy를 내고, **completion은 약 30% 짧아지며**, `|o_wrong|`는 training 내내 flat하게 유지된다. "long wrong" pathology가 제거된다.

---

## §6 네 variant — 하나의 comparison table

| Variant | Year | Advantage form | KL location | Clip | Group size | 2025 adoption |
|---------|------|----------------|-------------|------|------------|---------------|
| **RLOO** | 2024 (Ahmadian) | `r_i − (1/(k−1))Σ_{j≠i} r_j` | token별 reward | no | k ∈ {2, 4} | niche — TRL, OpenRLHF; small-k RLHF에 사용 |
| **REINFORCE++** | 2025 (Hu) | `(G_t − mean_B) / std_B` (global) | token별 reward (k1) | yes (ε=0.2) | k = 1 OK | big-batch setting에서 OpenRLHF mainstream |
| **GRPO** | 2024 (DeepSeekMath) | `(r_i − mean_g) / std_g`, agg `(1/|o_i|)` | in-loss, k3 estimator | yes (ε=0.2) | G ∈ {8, 64} | post-R1 dominant; verl, TRL, OpenRLHF |
| **Dr.GRPO** | 2025 (Liu et al.) | `r_i − mean_g`, agg `(1/L_max)` | in-loss, k3 estimator | yes (ε=0.2) | G ∈ {8, 64} | 2025 reasoning RL default |

표를 읽는 방법: 아래로 갈수록 무언가를 *뺀다*. RLOO에는 clip이 없다. REINFORCE++는 clip을 더하지만 per-group이 아니라 globally normalize한다. GRPO는 per-group std-norm과 in-loss k3 KL을 더한다. Dr.GRPO는 GRPO를 bias시킨 두 division을 삭제한다. 네 variant 모두 critic을 버린다. 전체 family는 baseline이 있는 vanilla REINFORCE에서 한 번 뺀 것이다.

**Limit에서의 equivalence** ([[rloo-vs-grpo]]):

- RLOO의 `b_i = (1/(k−1)) Σ_{j≠i} R_j → mean(R)` as k grows.
- 따라서 RLOO의 advantage → `R_i − mean(R)` = Dr.GRPO의 advantage exactly.
- GRPO의 PPO-clip은 rollout당 1 epoch에서 거의 bind되지 않는다.
- 결론: **RLOO (large k) ≈ /std와 clip 없는 GRPO ≈ Dr.GRPO**. 같은 estimator다.

---

## §7 Framework implementations — equation이 실제로 사는 곳

**verl** ([[verl-grpo]]). `verl/trainer/ppo/core_algos.py` ~L290–335는 `compute_grpo_outcome_advantage`를 등록한다. core loop는 reward를 prompt별로 groupby하고, per-group mean과 std를 계산한 뒤 `norm_adv_by_std_in_grpo=True`(GRPO)이면 `(score − mean)/std`, `False`(Dr.GRPO)이면 `score − mean`만 쓴다. Dr.GRPO toggle은 boolean 하나다. critic이 없으므로 `(advantages, returns) = (scores, scores)`다. `scores.unsqueeze(-1) * response_mask`로 token별 broadcast한다. 모든 response token이 같은 scalar advantage를 공유하며, length bias는 downstream policy-loss aggregator에서 들어온다.

**TRL** ([[trl-grpo]]). `trl/trainer/grpo_trainer.py`는 모든 것을 `_compute_loss`에 fuse한다. `loss_type` switch가 aggregator를 고른다.

```python
if self.loss_type == "grpo":
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
```

세 줄이다. 유일한 차이는 denominator다. k3 KL은 β≠0일 때 `exp(Δ) − Δ − 1`로 inline 계산된다. TRL은 추가 `loss_type` branch로 DAPO, CISPO, BNPO도 지원한다. **OpenRLHF**는 PPO `PolicyLoss` module을 재사용하고 experience-buffer pre-processing에서 group baseline을 계산한다. algebra는 같고 code home만 다르다.

---

## §8 2024–2025 shift — 왜 group baseline이 이겼는가

맥락으로 [[on-off-policy-rlhf]]를 읽어라. 2023–2024년 초에 field는 "RL methods"(critic이 있는 PPO)와 "RL-free methods"(DPO, offline preference)로 갈라졌다. Offline DPO는 PPO보다 못했고, DeepMind 팀은 그 원인을 algorithm family가 아니라 distribution shift(격차의 약 80%)로 분리했다. 이는 community를 다시 **online** methods로 밀어 넣었다. iterative DPO, online DPO, 그리고 결정적으로 critic-free online RL이다.

동시에 DeepSeek가 GRPO와 R1을 출시했다. R1 recipe는 reasoning을 위한 RL을 **verifiable rewards**(0/1 math correctness, code pass rate) 중심으로 다시 프레이밍했다. 이 regime에서는 verifier function이 RM training을 대체한다. PPO critic은 deterministic binary variable을 예측하도록 학습하는 셈이고, 이는 낭비다. Group-baseline methods(RLOO, GRPO)가 엄밀히 지배한다. 더 단순하고, 메모리가 덜 들며, accuracy가 같거나 더 좋다.

2025년까지 field는 수렴했다. **critic-free, online, group-baseline**이 기본값이다. PPO는 이미 좋은 value network가 학습되어 있거나(드묾), value-loss side를 통해 explicit entropy regularization을 원할 때만 남는다. 나머지는 모두 RLOO–Dr.GRPO line에 있다.

2026년 4월 현재 열린 논쟁은 family *내부*에 있다. global(REINFORCE++)로 normalize할 것인가, per-group(GRPO)으로 normalize할 것인가? clip을 유지할 것인가 버릴 것인가? low-entropy token만 mask할 것인가(DAPO), 모든 token을 쓸 것인가(classic)? 이것들이 새로운 축이다. critic은 결론났다.

---

## Companion visualization

**[figures/group-baseline.html](figures/group-baseline.html)** — interactive 2-panel explorer. **Panel 1:** group size K와 reward distribution(binary 0/1, discrete 3-level, continuous Gaussian)을 설정한다. RLOO, GRPO, Dr.GRPO의 advantage distribution이 overlay된 것을 본다. std가 작을 때 `/std`가 GRPO magnitude를 어떻게 왜곡하는지 관찰하라. **Panel 2:** length-bias illustration — response length `|o_i|`와 rollout correctness를 설정하고, GRPO(`(1/|o_i|)`) vs Dr.GRPO(`(1/L_max)`)에서 token별 gradient magnitude를 본다. 짧은 correct response와 10배 긴 wrong response는 GRPO에서 같은 sequence-loss를 기여한다. Dr.GRPO는 긴 wrong one을 길이에 비례해 penalize한다.

---

## Further reading

- [[rloo]] — Ahmadian 2024 derivation; k=2 vs k=4 vs k=8 Pareto frontier.
- [[grpo]], [[deepseekmath]] — Shao 2024 paper; Algorithm 1; k3 estimator Eq. 4.
- [[dr-grpo]] — Liu 2025 bias correction; Figure 1이 proof다.
- [[reinforce-plus-plus]] — Hu 2025 OpenRLHF-native variant.
- [[rloo-vs-grpo]] — equivalence-in-the-limit argument가 있는 comparative reference.
- [[trl-grpo]], [[verl-grpo]] — 두 dominant open-source implementation.
- [[nathan-lambert-grpo]] — 2025 fixes에 대한 practitioner tracking; worked length-bias example.
- [[on-off-policy-rlhf]] — 왜 2024년에 다시 online으로 돌아갔는지; §8의 narrative를 근거 짓는다.
- [[john-schulman-kl-tricks]] — k1/k2/k3 KL estimator derivation.

## Connections

- **ch-38** — PPO와 이 장이 빼는 critic.
- **ch-39** — offline preference(DPO family); field가 여기로 다시 수렴하기 전에 택했던 다른 branch.
- **ch-41** — reward modeling; 이 장은 `R(x, y)`가 존재한다고 가정했고, ch-41은 그것을 어떻게 만드는지다.
- **ch-42** — reward hacking; §5의 length bias가 archetype이다.
- **ch-44+** — DeepSeek-R1 recipe chapters; 그곳의 모든 loss는 GRPO / Dr.GRPO variant다.
