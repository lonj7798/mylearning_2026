<!-- chapter: ch-40
     track: rl
     kind: content
     title: Online / Group-Baseline RL Family
     deps: [ch-39]
     sources: [[rloo]], [[reinforce-plus-plus]], [[grpo]], [[dr-grpo]], [[deepseekmath]], [[rloo-vs-grpo]], [[trl-grpo]], [[verl-grpo]], [[nathan-lambert-grpo]], [[on-off-policy-rlhf]]
     figures: figures/group-baseline.html
-->

# Chapter 40 — Online / Group-Baseline RL Family

> **Core insight.** In 2024 the RLHF field quietly killed the PPO critic. Ahmadian's RLOO showed that for LLMs — deterministic token transitions, full-sequence rewards, one-epoch updates — a learned value head was dead weight; replace it with the mean of the *other k-1 peer samples* and REINFORCE beats PPO at half the memory. Four months later DeepSeekMath's GRPO took the same idea, scaled the group to G=64, normalized by group std, and put the KL inside the loss as an unbiased k3 estimator. R1 shipped with it. Then in March 2025 Dr.GRPO proved the GRPO loss has two length-biased divisions that reward wrong-and-long outputs — and the fix is to *delete them*. The arc of this chapter is one subtraction: PPO minus the critic minus the std denominator minus 1/|o_i| equals "REINFORCE with a peer baseline," which is what the entire field actually runs today.
>
> **Guideline.** Default to **Dr.GRPO** for reasoning RL with verifiable 0/1 rewards (the DeepSeek-R1 setting). Use **RLOO** with k=2–4 when memory is tight and the reward is a continuous RM score. Use **REINFORCE++** when you can only afford k=1 per prompt but have a large global batch. Only reach for vanilla **GRPO** when you need the std-normalization's variance-magnitude control across prompts of heterogeneous reward spread — and even then log `mean(|o_wrong|) − mean(|o_right|)` every epoch: if it grows, the length bias is active and you should switch.

---

## §1 The problem with PPO's critic in LLM-land

Return to [[ppo]] for a moment. PPO's advantage is `Â_t = δ_t + (γλ) δ_{t+1} + …` where `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`. The value head `V_φ` is a second full-size network trained with a mean-squared target `(V_φ(s_t) − R_t)²`. In Atari or MuJoCo this is fine: episodes are thousands of stochastic transitions and V's bootstrap bias is swamped by the variance it removes.

For LLM RLHF three assumptions break at once (Ahmadian §3, [[rloo]]):

1. **Deterministic transitions** — once `y_{<t}` is fixed, `y_t` uniquely determines the next state. There is no "expected return over stochastic dynamics" to estimate; the only randomness is your own sampling.
2. **Full-trajectory reward** — the RM or verifier gives one scalar at end-of-sequence. No intermediate `r_t`. GAE's `(γλ)`-weighted deltas collapse to `R − V(s_0)`, i.e. a single learned baseline.
3. **One epoch per rollout** — TRL/verl/OpenRLHF all default to `μ = 1`. PPO's clip exists to protect against drift across multiple epochs; with one epoch the clip rarely binds.

So PPO's critic is doing *one* job: learning `V(s_0) = V(prompt)`. But `V(prompt) ≈ E_{y ∼ π}[R(prompt, y)]`, and the unbiased minimum-variance estimator of that expectation is… **the mean reward of other samples from the same prompt**. The critic is a fragile learned approximation of a quantity you already have free-of-charge once you sample more than one rollout per prompt. Delete it.

---

## §2 RLOO — the leave-one-out identity

Sample `k` responses `y_1, …, y_k ∼ π_θ(·|x)`. Compute rewards `R_i = R(x, y_i)`. The REINFORCE gradient with a constant baseline `b` is:

```
∇J ≈ (1/k) Σ_i (R_i − b) · ∇ log π_θ(y_i | x)
```

Any baseline independent of `y_i` leaves the gradient unbiased (classic policy-gradient result, [[vanilla-pg]]). Pick `b_i` separately for each sample — the only requirement is `b_i ⊥ y_i` given `x`. The tightest variance-reducing choice that still respects that independence is the **leave-one-out mean**:

```
b_i = (1/(k−1)) Σ_{j ≠ i} R_j
```

Each `b_i` is a function of `{R_j : j ≠ i}`, which depends on `{y_j : j ≠ i}` but not on `y_i` — unbiased by construction. Substituting:

```
∇J_RLOO ≈ (1/k) Σ_i [ R_i − (1/(k−1)) Σ_{j ≠ i} R_j ] · ∇ log π_θ(y_i | x)
```

Two sanity checks. (a) **k=2 case**: `b_1 = R_2`, `b_2 = R_1`, so the advantage for sample 1 is `R_1 − R_2` and for sample 2 is `R_2 − R_1` — a pure pairwise comparison. This is why RLOO with k=2 is often described as "online DPO without the log-sigmoid." (b) **Large-k limit**: `b_i → mean(R)`, so RLOO's advantage `→ R_i − mean(R)`, which is exactly Dr.GRPO's advantage (§5). RLOO and Dr.GRPO are the same estimator modulo a O(1/k) correction.

What gets deleted vs PPO ([[rloo]] Table):

| Component | PPO | RLOO |
|-----------|-----|------|
| Value network | required | **removed** → ~50% memory |
| GAE | `γλ` weighted δ sum | not needed (full-seq reward) |
| Clip ε | yes | **no** (1 epoch) |
| Epochs per rollout | 4 (classic), 1 (LLM) | 1 |
| Baseline | learned V_φ | leave-one-out over k peers |
| KL location | per-token shaped reward | per-token shaped reward (same) |

Empirically: TL;DR summarization, HH-RLHF helpfulness — RLOO k=4 dominates the PPO Pareto frontier at every KL budget (Ahmadian Fig. 3). The message: PPO's overhead is a tax, not a feature, for LLM RLHF.

---

## §3 REINFORCE++ — global normalization for k=1 regimes

Jian Hu 2025 ([[reinforce-plus-plus]]) pushed the same logic one step further: what if you can't afford k ≥ 2 rollouts per prompt? A single sample per prompt means RLOO's peer baseline is undefined. The fix: normalize across the **whole mini-batch** of prompts instead of within-prompt.

**Per-token shaped reward** (identical to InstructGPT-style PPO):

```
r̃_t = R(x, y) · 𝟙{t = T} − β · KL_t,    KL_t = log π_θ_old(y_t|·) − log π_ref(y_t|·)
```

**Cumulative return** from step t: `G_t = Σ_{t'≥t} r̃_{t'}` with γ = 1.

**Global advantage normalization** over the full batch B:

```
Â_t = (G_t − mean_{B}(G)) / std_{B}(G)
```

Mean and std are computed across every `(prompt, response, token)` triple in the batch — not per prompt. With batch size 512–2048 sequences, the std estimate is tight; within-prompt it would be wild for k=1.

**Loss** is the standard PPO-clip surrogate reused unchanged:

```
L(θ) = − E_t[ min(ρ_t(θ) Â_t, clip(ρ_t(θ), 1-ε, 1+ε) Â_t) ]
```

The clip is kept here (unlike RLOO) because REINFORCE++ runs the risk of per-token drift — with a huge advantage magnitude, a single token's ratio can jump far from 1.0 in one step; ε=0.2 bounds that. No value net, k=1, global normalization, KL on the reward not on the loss — memory-efficient, stable, mainstream in OpenRLHF by mid-2025.

---

## §4 GRPO — DeepSeekMath's group z-score

Shao et al. 2024 ([[grpo]], [[deepseekmath]]) took the group-baseline idea to large G and added two wrinkles: **std-normalize** the advantage, and **move KL inside the loss** as the k3 estimator.

**Rollout.** For prompt q in batch, sample `o_1, …, o_G ∼ π_θ_old`. RM gives `r_i`.

**Advantage (outcome supervision).** Every token in `o_i` gets the same advantage:

```
Â_{i,t} = (r_i − mean(r_1, …, r_G)) / std(r_1, …, r_G)
```

Why std-normalize? Prompt A has rewards `{0.1, 0.11, 0.12}` (all rollouts near-equivalent); prompt B has `{0.0, 0.5, 1.0}` (high variance). Without std, B's gradient magnitude is ~10× A's even though B's *relative* signal is equally informative. Dividing by std equalizes per-prompt gradient magnitudes so the optimizer doesn't focus all updates on the high-variance prompts. This helps when rewards are continuous RM scores and prompts differ in "difficulty spread."

**Objective (Eq. 3 of [[grpo]]).**

```
J_GRPO(θ) = E[q, {o_i}] (1/G) Σ_i (1/|o_i|) Σ_t {
    min[ ρ_{i,t} Â_{i,t},  clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t} ]
    − β · D_KL^k3(π_θ || π_ref)
}
```

where `ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})`.

**k3 KL estimator (Eq. 4).** From the [[john-schulman-kl-tricks]] literature:

- **k1:** `log(π_θ/π_ref)` — low variance, biased in sign (can be negative for a divergence that must be ≥0).
- **k2:** `½·(log ratio)²` — unbiased, but always positive by squaring so the sign information is lost.
- **k3:** `π_ref/π_θ − log(π_ref/π_θ) − 1` — unbiased, **always ≥0** (convex Bregman distance). GRPO uses k3.

Derivation sketch: let `x = log(π_ref/π_θ)`. Then `k3 = e^x − x − 1`. Taylor-expand around x=0: `e^x − x − 1 = x²/2 + x³/6 + …`. For small KL (the RL regime), k3 ≈ k2 = x²/2; for large drift, k3 grows linearly and bounds the k1 estimator from below. One extra reference forward pass, tensor identical shape to logprobs, positivity guaranteed.

**KL-in-loss vs KL-on-reward.** RLOO and REINFORCE++ put β·KL into the per-token reward before computing advantages. GRPO leaves the reward untouched and adds `−β · KL_t` to the per-token loss. Numerically subtly different: on-reward KL propagates through the advantage normalization (std divides it); in-loss KL does not. The in-loss form keeps the *advantage* purely a function of the outcome reward, which is cleaner for verifiable 0/1 tasks.

**Paper recipe** (MATH / GSM8K, [[deepseekmath]]): G=64, ε=0.2, β=0.04, LR 1e-6, batch 1024 prompts, T=1.0 sampling, max 1024 tokens, π_ref frozen SFT. Result: MATH 51.7% (from PPO's 51.0, RFT's 49.0, SFT's 46.8). GRPO is the loss DeepSeek used for R1-Zero and R1.

---

## §5 Dr.GRPO — which divisions were biased, and why

Liu et al. 2025 ([[dr-grpo]]) identified two biases in GRPO that compound during reasoning RL: the `1/|o_i|` per-response token average, and the `/std(r)` in the advantage. Both come from divisions that *look innocent* and are in fact not unbiased.

### Bias 1: the `1/|o_i|` per-response mean

GRPO's loss averages each rollout's per-token losses by the realized length `|o_i|`. Consider two incorrect responses with the same advantage `Â = −1.0`, same per-token ratio `ρ_t ≈ 1`:

- **Short wrong:** `|o_1| = 50`. Per-token loss ≈ `+1`. Aggregated by `(1/50) · Σ_t ≈ +1`. Full gradient contribution ≈ full magnitude.
- **Long wrong:** `|o_2| = 500`. Per-token loss ≈ `+1` still. Aggregated by `(1/500) · Σ_t ≈ +1`. Same aggregated loss — but distributed over 10× more tokens.

The aggregated loss looks identical, so the *per-sequence* gradient magnitude is identical. But the gradient *per token* is 10× smaller on the long wrong rollout. The optimizer therefore moves each token's logprob in `o_2` 10× less than in `o_1`. Repetition within a wrong response is effectively free: adding 200 more wrong tokens dilutes the penalty per token without changing the aggregated sequence loss.

**Net effect over training:** wrong-and-long responses are underpenalized; `|o_wrong|` grows monotonically; chain-of-thought becomes rambling filler. Dr.GRPO's Figure 1 shows this length curve shooting upward for vanilla GRPO but staying flat for the corrected version.

The fix is to replace `(1/|o_i|)` with a **fixed constant** `(1/L_max)` — e.g. 4096, the generation budget. Now every token's contribution to the loss is the same absolute magnitude regardless of how long the response happened to be; longer wrong rollouts accumulate proportionally more penalty. This is `loss_type="dr_grpo"` in [[trl-grpo]]:

```python
# dr_grpo aggregation (TRL L2418+)
loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
```

compared to `loss_type="grpo"`:

```python
# grpo aggregation (TRL L2418+)
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
```

The only change is the denominator — one number. That is the bias correction.

### Bias 2: the `/std(r)` difficulty weighting

Two prompts in the same batch: prompt A has `r = {1, 1, 1, 0, 0, 0, 0, 0}` (std ≈ 0.52, mean = 0.375); prompt B has `r = {1, 0, 0, 0, 0, 0, 0, 0}` (std ≈ 0.35, mean = 0.125). Both are informative — in A you know half the rollouts got it right; in B only one did. The raw advantage `(r_i − mean)` differs between prompts, but std-normalization makes the advantages **comparable in magnitude** across prompts.

That sounds helpful, but consider the pathological edge: if all G rollouts get r=0 (the prompt is too hard) or all get r=1 (too easy), std → 0 and the advantage → 0/0 = undefined. Implementations clip with `+ ε = 1e-6`, which makes the advantage *enormous* and unstable. Worse, the in-between regime over-weights prompts where std happens to be small — easy-or-hard prompts dominate the update, pushing the policy away from its current reward distribution on those very prompts rather than concentrating on the ones where the rollouts are split (which are the most informative ones).

The fix: drop the `/std` entirely.

```
Ã_{i,t} = r_i − mean(r_1, …, r_G)          # Dr.GRPO, unbiased
```

This is algebraically RLOO's `k`-large-limit. You lose the across-prompt gradient-magnitude equalization, but you gain invariance to single-prompt reward collapse and remove the difficulty-weighting bias. For verifiable-reward RL where `r ∈ {0, 1}`, dropping `/std` is strictly an improvement.

### Dr.GRPO loss (putting it together)

```
J_Dr.GRPO(θ) = E[ (1/G) Σ_i (1/L_max) Σ_t min(ρ_{i,t} Ã_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε) Ã_{i,t})
                   − β · D_KL^k3(π_θ || π_ref) ]
```

Identical to GRPO in every other place: same ρ, same clip, same k3 KL. Just `Â_{i,t} = (r_i − mean)/std → Ã_{i,t} = r_i − mean` and `(1/|o_i|) → (1/L_max)`. Two deletions.

Empirical result (Dr.GRPO Table 2 / Figure 1): Qwen2.5-Math-7B on MATH/AIME/AMC: matches or beats vanilla GRPO on accuracy, **produces ~30% shorter completions**, and `|o_wrong|` stays flat through training. The "long wrong" pathology is eliminated.

---

## §6 The four variants — one comparison table

| Variant | Year | Advantage form | KL location | Clip | Group size | 2025 adoption |
|---------|------|----------------|-------------|------|------------|---------------|
| **RLOO** | 2024 (Ahmadian) | `r_i − (1/(k−1))Σ_{j≠i} r_j` | per-token reward | no | k ∈ {2, 4} | niche — TRL, OpenRLHF; used for small-k RLHF |
| **REINFORCE++** | 2025 (Hu) | `(G_t − mean_B) / std_B` (global) | per-token reward (k1) | yes (ε=0.2) | k = 1 OK | mainstream in OpenRLHF for big-batch settings |
| **GRPO** | 2024 (DeepSeekMath) | `(r_i − mean_g) / std_g`, agg `(1/|o_i|)` | in-loss, k3 estimator | yes (ε=0.2) | G ∈ {8, 64} | dominant post-R1; verl, TRL, OpenRLHF |
| **Dr.GRPO** | 2025 (Liu et al.) | `r_i − mean_g`, agg `(1/L_max)` | in-loss, k3 estimator | yes (ε=0.2) | G ∈ {8, 64} | default for reasoning RL in 2025 |

Reading the table: moving down you *subtract* things. RLOO has no clip. REINFORCE++ adds clip but normalizes globally not per-group. GRPO adds per-group std-norm and in-loss k3 KL. Dr.GRPO deletes the two divisions that biased GRPO. All four drop the critic; the entire family is one subtraction away from vanilla REINFORCE with a baseline.

**Equivalence in the limit** ([[rloo-vs-grpo]]):

- RLOO's `b_i = (1/(k−1)) Σ_{j≠i} R_j → mean(R)` as k grows.
- So RLOO's advantage → `R_i − mean(R)` = Dr.GRPO's advantage exactly.
- PPO-clip in GRPO rarely binds at 1 epoch per rollout.
- Conclusion: **RLOO (large k) ≈ GRPO without /std and without clip ≈ Dr.GRPO**. They are the same estimator.

---

## §7 Framework implementations — where the equations actually live

**verl** ([[verl-grpo]]). `verl/trainer/ppo/core_algos.py` ~L290–335 registers `compute_grpo_outcome_advantage`. The core loop groupbys rewards by prompt, computes per-group mean and std, then writes `(score − mean)/std` when `norm_adv_by_std_in_grpo=True` (GRPO) or just `score − mean` when `False` (Dr.GRPO). The Dr.GRPO toggle is one boolean. `(advantages, returns) = (scores, scores)` because there is no critic. Per-token broadcast via `scores.unsqueeze(-1) * response_mask` — every response token shares the same scalar advantage; length bias enters downstream in the policy-loss aggregator.

**TRL** ([[trl-grpo]]). `trl/trainer/grpo_trainer.py` fuses everything into `_compute_loss`. The `loss_type` switch selects the aggregator:

```python
if self.loss_type == "grpo":
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
```

Three lines; the only difference is the denominator. k3 KL is computed inline via `exp(Δ) − Δ − 1` when β≠0. TRL also supports DAPO, CISPO, BNPO as additional `loss_type` branches. **OpenRLHF** reuses its PPO `PolicyLoss` module and computes group baselines in the experience-buffer pre-processing — same algebra, different code home.

---

## §8 The 2024–2025 shift — why group baselines won

Read [[on-off-policy-rlhf]] as context. 2023–early 2024: the field had split into "RL methods" (PPO with critic) and "RL-free methods" (DPO, offline preference). Offline DPO underperformed PPO; the DeepMind team isolated the cause as distribution shift (~80% of the gap), not the algorithm family. This pushed the community back to **online** methods: iterative DPO, online DPO, and — crucially — critic-free online RL.

Simultaneously, DeepSeek shipped GRPO and then R1. The R1 recipe reframed RL-for-reasoning around **verifiable rewards** (0/1 math correctness, code pass rate) where RM training is replaced by a verifier function. In that regime, PPO's critic is learning to predict a deterministic binary variable, which is wasteful. Group-baseline methods (RLOO, GRPO) strictly dominate: simpler, less memory, equal or better accuracy.

By 2025 the field had converged: **critic-free, online, group-baseline** is the default. PPO persists only where someone has a good value network already trained (rare) or wants explicit entropy regularization through the value-loss side. Everything else is in the RLOO–Dr.GRPO line.

The open debate as of April 2026 is *within* the family: should you normalize globally (REINFORCE++) or per-group (GRPO)? Should you keep the clip or drop it? Should you mask out low-entropy tokens (DAPO) or all tokens (classic)? These are the new axes. The critic is settled.

---

## Companion visualization

**[figures/group-baseline.html](figures/group-baseline.html)** — interactive 2-panel explorer. **Panel 1:** set group size K and reward distribution (binary 0/1, discrete 3-level, continuous Gaussian); see advantage distribution under RLOO, GRPO, and Dr.GRPO overlaid. Watch how `/std` distorts GRPO's magnitude when std is small. **Panel 2:** the length-bias illustration — set response length `|o_i|` and rollout correctness; see per-token gradient magnitude under GRPO (`(1/|o_i|)`) vs Dr.GRPO (`(1/L_max)`). A short correct response and a 10×-longer wrong response both contribute the same sequence-loss in GRPO; Dr.GRPO penalizes the long wrong one proportionally to its length.

---

## Further reading

- [[rloo]] — Ahmadian 2024 derivation; k=2 vs k=4 vs k=8 Pareto frontier.
- [[grpo]], [[deepseekmath]] — Shao 2024 paper; Algorithm 1; k3 estimator Eq. 4.
- [[dr-grpo]] — Liu 2025 bias correction; Figure 1 is the proof.
- [[reinforce-plus-plus]] — Hu 2025 OpenRLHF-native variant.
- [[rloo-vs-grpo]] — comparative reference with the equivalence-in-the-limit argument.
- [[trl-grpo]], [[verl-grpo]] — the two dominant open-source implementations.
- [[nathan-lambert-grpo]] — practitioner tracking of the 2025 fixes; worked length-bias example.
- [[on-off-policy-rlhf]] — why 2024 pushed back to online; grounds §8's narrative.
- [[john-schulman-kl-tricks]] — k1/k2/k3 KL estimator derivation.

## Connections

- **ch-38** — PPO and the critic this chapter subtracts.
- **ch-39** — offline preference (DPO family); the other branch the field took before converging back here.
- **ch-41** — reward modeling; this chapter assumed `R(x, y)` exists, ch-41 is how you build it.
- **ch-42** — reward hacking; §5's length bias is the archetype.
- **ch-44+** — DeepSeek-R1 recipe chapters; every loss there is a GRPO / Dr.GRPO variant.
