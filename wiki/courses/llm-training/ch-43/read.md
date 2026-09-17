<!-- chapter: ch-43
     track: rl
     kind: content
     title: Entropy, Output Diversity, and KL Control in RL
     deps: [ch-16]
     sources: [[entropy-mechanism-llm-rl]], [[high-entropy-minority-tokens]], [[clip-low-clip-high-entropy]],
              [[dapo]], [[pass-at-k-training]], [[unlikeliness-reward]], [[noveltybench]],
              [[john-schulman-kl-tricks]], [[kl-control-rlhf]], [[entropy-logging-patterns]],
              [[verl-ppo-loss]], [[grpo]], [[dr-grpo]], [[prorl]], [[rls-razor]], [[simpletir]],
              [[deepswe]], [[deepswe-recipe]], [[kimi-k2-recipe]],
              [[allenai-olmo3-open-instruct-scripts-recipe]], [[rlhf-instructgpt]], [[tulu-3]],
              [[entropy-regularization-ppo]], [[entropy-collapse-ppo]], [[maximum-entropy-rl]],
              [[nathan-lambert-entropy-rl]], [[openrlhf-entropy-debugging]], [[rlvr-beyond-base-model]]
     figures: figures/entropy-dynamics.html
     revised: 2026-09 (generality revision)
-->

# Chapter 43 — Entropy, Output Diversity, and KL Control in RL

> **Core insight.** Policy entropy falls during RL because the update is correlated with token
> log-probability: the first-order change in entropy is the negative covariance between a token's
> log-probability and the change in its logit, which under a policy-gradient update is proportional to its
> advantage ([[entropy-mechanism-llm-rl]] §3, Theorems 1-2). The same correlation shows up at the sequence
> level as falling diversity — fewer distinct generations after each post-training stage on OLMo 2
> ([[noveltybench]] Fig. 6) and falling pass@k at large k in parts of long RL runs ([[prorl]] §4.2). The
> interventions that work in published runs act on a selected part of the distribution (a covariance-outlier
> set, the high-entropy 20% of tokens, one side of the clip range, or the advantage of a whole group), not
> on all tokens at once ([[entropy-mechanism-llm-rl]] Table 2, [[high-entropy-minority-tokens]] Table 2,
> [[clip-low-clip-high-entropy]] §3.2, [[pass-at-k-training]] Table 1). The KL penalty to a reference policy
> is a separate control with a separate purpose: it bounds how far the policy moves, which is the quantity
> that predicts forgetting on tasks outside the RL data ([[rls-razor]] §3).
>
> **Guideline.** When a run's validation accuracy plateaus while entropy is still falling, change which
> tokens receive the update — raise `ε_high` (clip-higher), restrict the policy gradient to the high-entropy
> tokens of the batch, or penalize covariance outliers — because on Qwen2.5-32B KL-Cov raised the 7-benchmark
> math average from 45.8 to 52.2 and on Qwen3-32B forking-token DAPO raised the 6-benchmark average from
> 66.59 to 70.69, while in the same paper's entropy-loss sweep the coefficients either had a minor influence
> on entropy (0.0001, 0.001), caused entropy explosion (0.01), or did not beat the baseline (0.005)
> ([[entropy-mechanism-llm-rl]] §4.1, Table 2; [[high-entropy-minority-tokens]] Table 2). When the goal is
> coverage rather than single-sample accuracy, change the objective instead of the regularizer: pass@k
> training and unlikeliness reward raise pass@k at large k in their settings
> ([[pass-at-k-training]] Table 1, [[unlikeliness-reward]] Table 3). When capability outside the RL data
> matters, keep a measured KL budget to the starting policy and evaluate retention directly, because β = 0
> recipes are reported only with in-domain metrics ([[dapo]] §2.3, [[dr-grpo]] §3,
> [[allenai-olmo3-open-instruct-scripts-recipe]]), and a larger KL coefficient alone did not remove
> capability regressions in the one run that swept it 100× ([[rlhf-instructgpt]] App. E.6).

---

## Why this chapter matters for a general-purpose model

RL is the last stage in the pipeline (pretraining → mid-training → SFT → preference optimization → RL →
evaluation), and it is the stage that makes the output distribution narrower on purpose: the objective is to
raise the probability of responses that score well on the training prompts. Three costs of that narrowing are
measurable.

1. **Coverage.** pass@k at large k is a measurement of how many distinct correct solutions the policy can
   still reach. In [[prorl]]'s Diminish regime — "particularly in the math domain" — pass@1 improves while
   pass@128 often declines (§4.2, Fig. 4). Under symmetric clipping, pass@8 declines over GSM8K training
   while mean@8 rises ([[clip-low-clip-high-entropy]] Fig. 6).
2. **Output diversity on open-ended prompts.** On the OLMo 2 checkpoints, `distinct_10` (distinct equivalence
   classes among 10 samples at temperature 1.0) falls at every post-training stage: 7B 7.46 → 5.96 → 5.72 and
   32B 7.25 → 5.22 → 5.08 for SFT → DPO → RLVR ([[noveltybench]] §4.4, Fig. 6).
3. **Capability outside the RL data.** Forgetting is predicted by the KL divergence between the fine-tuned
   and the base policy measured on the *new* task, with `R² = 0.96` in the controlled
   ParityMNIST/FashionMNIST setting and `R² = 0.71` on the LLM sweep ([[rls-razor]] §3). That makes the KL budget a control variable for retention, not only
   a stability trick. ch-38a covers the SFT-versus-RL comparison this law comes from.

Entropy sits underneath all three: it is the per-step quantity a training loop can log, and its dynamics are
now derived rather than asserted. This chapter states the derivation, the interventions with their reported
settings, the KL estimators and where the penalty is applied, and what each choice does to breadth.

---

## §1 Three quantities that are often conflated

**Definitions.**

- **Policy entropy** `H(π | s) = −Σ_a π(a|s) log π(a|s)` at one position, averaged over the tokens of the
  sampled responses to a prompt batch ([[entropy-mechanism-llm-rl]] Eq. 5;
  [[clip-low-clip-high-entropy]] Eq. 4). It is a property of one conditional distribution, measured in nats.
- **Output diversity** is a property of a set of full samples: `distinct_k`, the number of equivalence
  classes among k generations ([[noveltybench]] Eq. 1), or pass@k, the probability that at least one of k
  samples is correct.
- **KL to a reference** `KL(π_θ ‖ π_ref)` (or its reverse) is a distance between the trained policy and a
  fixed policy, aggregated over the states visited.

**Why they must be measured separately.** Take a 5-token vocabulary at one position with
`π = (0.70, 0.20, 0.06, 0.03, 0.01)`. Then `H = 0.892` nats. A policy with the same entropy at every position
can still produce ten near-identical responses if the high-entropy choices are all local wording choices;
and a policy whose average entropy is lower can produce more distinct responses if the remaining entropy sits
at the points that choose a solution path. That is the empirical picture in
[[high-entropy-minority-tokens]]: in over 10⁶ Qwen3-8B chain-of-thought tokens, 50.64% of tokens have entropy
below 10⁻², and only 20% exceed 0.672 nats; the high-entropy minority are the positions where the reasoning
branches (§3).

**Consequence for logging.** A single scalar "entropy" per step is a compressed view of a distribution over
positions. Log the mean, but also log a quantile (for example the 80th percentile used as the forking-token
threshold in [[high-entropy-minority-tokens]] §3) and a sequence-level diversity metric on a fixed prompt set.

---

## §2 Why entropy falls: the covariance result

**The problem, stated as a measurement.** In runs from 11 base models of four families with no entropy term
and no reference KL, entropy decreases monotonically toward zero while validation accuracy rises and then
saturates; 73% of the entropy consumption and 76% of the performance gain occur in the first 200 of 2,400
gradient steps ([[entropy-mechanism-llm-rl]] §2.3, Fig. 2). The paper fits `R = −a·exp(H) + b` between
validation accuracy `R` and entropy `H` (Eq. 6) and reports that fitting on the first 36 steps predicts the
next 200 with RMSE 0.9% on math and 1.2% on code (§2.4). The authors state this predictability "is not
arguably universal", since other policy models and off-policy data showed different patterns (§2.6). Treat
the fit as a **Result (single study)** with its own scope statement attached.

**Mechanism, step by step** ([[entropy-mechanism-llm-rl]] §3):

1. Write the policy at a state as a softmax over logits `z`.
2. Lemma 1 (first order in the logit change): `ΔH(s) ≈ −Cov_{a~π(·|s)}(log π(a|s), Δz_{s,a})`. Symbols: `Δz`
   is the change of the logit of action `a` at state `s`; the covariance is taken over actions drawn from the
   current policy at that state.
3. Theorem 1 (vanilla policy gradient): the logit change is `Δz_{s,a} = η·π(a|s)·A(s,a)`, so
   `ΔH ≈ −η·Cov(log π(a|s), π(a|s)·A(s,a))`. Symbols: `η` learning rate, `A` advantage.
4. Theorem 2 (natural policy gradient): `ΔH ≈ −η·Cov(log π(a|s), A(s,a))`.
5. Empirically, on Qwen2.5-7B, `−dH` tracks the measured covariance and the covariance stays positive during
   training (§3.3, Fig. 8).

The sign rule follows from the covariance: entropy falls when the update is *positively* correlated with
log-probability — when likely tokens get positive advantage or unlikely tokens get negative advantage.

**Worked example (checkable by hand).** Take the five-token distribution above,
`π = (0.70, 0.20, 0.06, 0.03, 0.01)`, `H = 0.8916` nats, and one sampled token with advantage `A = ±1`,
learning rate `η = 0.5`, updating the logits by the exact policy-gradient direction
`Δz_j = η·A·(1[j = a] − π_j)`:

| Sampled token | Advantage | New `π_a` | Exact `ΔH` | First-order `−Cov` prediction |
|---|---|---|---|---|
| `π_a = 0.70` (above average) | +1 | 0.745 | **−0.0775** | −0.0753 |
| `π_a = 0.70` | −1 | 0.651 | **+0.0721** | +0.0753 |
| `π_a = 0.03` (below average) | +1 | 0.062 | **+0.1591** | +0.1511 |
| `π_a = 0.03` | −1 | 0.014 | **−0.1410** | −0.1511 |

The dividing line is the policy's own mean log-probability, `Σ_j π_j log π_j = −H = −0.892`: the sampled
token at `p = 0.70` has `log π_a = −0.357 > −H`, and the token at `p = 0.03` has `log π_a = −3.507 < −H`.
The four rows are the four quadrants of the next section. The figure
[figures/entropy-dynamics.html](figures/entropy-dynamics.html) lets you change the distribution, the sampled
token, the advantage and `η` and read the exact entropy change and the mass that moves to each alternative;
it also reproduces the estimator comparison of §5.

---

## §3 Entropy by advantage sign: the four quadrants

Every token in a batch falls into one of four cells. Using the centered form of the covariance that
[[entropy-mechanism-llm-rl]] uses to select tokens (Eq. 10), the contribution of token `y_i` is
`(log π(y_i) − mean_j log π(y_j)) · (A(y_i) − mean_j A(y_j))`; a positive product pushes entropy down.

| | Above-average probability | Below-average probability |
|---|---|---|
| **Positive advantage** | product > 0 → entropy falls (the policy commits to what it already prefers) | product < 0 → entropy rises (a rare option is lifted) |
| **Negative advantage** | product < 0 → entropy rises (mass leaves the mode) | product > 0 → entropy falls (a rare option is removed and its mass returns mostly to the mode) |

Two consequences matter for practice.

**(a) Ranking tokens by `p·A` cannot find half of the mass.** `p > 0`, so the largest values of `p·A` are all
positive-advantage tokens. With the four tokens
`(π, A) = (0.90, +1.2), (0.50, −0.8), (0.05, +1.0), (0.01, −1.5)` in a batch, the batch means are
`mean log π = −2.100` and `mean A = −0.025`, and the centered products are `+2.443`, `−1.090`, `−0.918`,
`+3.695`. The largest entropy-reducing contribution is the *negative-advantage* token at `π = 0.01`, which a
`p·A` ranking scores at `−0.015` and never selects. This is why [[entropy-mechanism-llm-rl]] selects on the
centered product (Eq. 10) rather than on `p·A`: the selection rule is sign-agnostic by construction.

**(b) The two clip bounds act on different quadrants.** In the clipped surrogate, the lower bound `1 − ε_low`
binds on negative-advantage tokens and the upper bound `1 + ε_high` binds on positive-advantage tokens.
[[clip-low-clip-high-entropy]] proves (Theorem 1, first order, tabular softmax, random symmetric advantages)
that the entropy change splits into a clip-low term that raises entropy and a clip-high term that lowers it,
with weights `p_k = P(clip-low)` and `q_k = P(clip-high)`; the sign claim needs two inequalities that the
authors state are not universal but hold in their measurements (Ineq. 7, Figs. 1-2). Under the symmetric
default `ε_low = ε_high = 0.2`, the clip-high term dominates and entropy falls even when the reward is random
noise (§2.3, Fig. 3, Qwen2.5-1.5B-Instruct; the same downward drift under random rewards appears for Qwen,
Llama and Olmo base models in Fig. 4, and for Llama3.2-1B-Instruct in Fig. 8a).

**Negative advantages are the mechanism behind squeezing.** Take the same five-token distribution and
decrease only the logit of the token at `π = 0.03`, by the 0.485 that the update of §2 applies to it. After
renormalization the distribution is `(0.7082, 0.2023, 0.0607, 0.0187, 0.0101)` and entropy falls from 0.8916
to 0.8586 nats. The 0.0113 of probability removed from the pushed-down token is redistributed in proportion
to the remaining probabilities, so 0.0082 of it — 72% — goes to the token already at 0.70 and 0.0023 to the
next-largest. The softmax has no other option. ch-43a derives this from
`∂ log p_y / ∂ z_j = 1[j = y] − p_j` and covers its offline counterparts.

---

## §4 Interventions that change *which* tokens are updated

### 4.1 A flat entropy bonus

Form: add `+ c_H·H(π)` to the objective. The term comes from A3C, which credits it to Williams & Peng (1991)
and uses `β = 0.01` for Atari and TORCS and `10⁻⁴` for its MuJoCo runs, with no ablation
([[entropy-regularization-ppo]] §4, Supp. §8-9). The bonus is therefore older than the maximum-entropy
objective of Soft Actor-Critic, which adds `α·H(π(·|s_t))` to the reward at every step and is off-policy with
a soft-Q critic ([[maximum-entropy-rl]] Eq. 1, §4.2); that setting shares the term but not the training loop,
and SAC's own sensitivity result is to the reward scale rather than to a token-level entropy target
(§5.2, Fig. 3b). In a large continuous-control sweep, no tested regularizer, entropy penalty or entropy
constraint included, helped significantly except on HalfCheetah ([[entropy-collapse-ppo]] §3.8, App. K.1).

At LLM scale the reported outcomes are: coefficients 0.0001 and 0.001 had minor influence, 0.01 caused
entropy explosion, and 0.005 stabilized entropy without beating the other baselines
([[entropy-mechanism-llm-rl]] §4.1, Fig. 9); coefficient 0.005 "might cause the collapse of the model" in the
Pass@k-training comparison, and the smaller values tested did not outperform the alternative
([[pass-at-k-training]] §3.1); DeepSWE removed the entropy loss because it "led to exponentially increasing
entropy and collapse", stating it is unnecessary when base-model token entropy is within 0.3-1
([[deepswe]] §2.3). The interpretation offered in [[high-entropy-minority-tokens]] (§6, Discussion 3,
**Interpretation**, no controlled comparison) is that a uniform bonus raises the entropy of the low-entropy
majority, which are the positions that complete an ongoing linguistic or mathematical structure, rather than
of the high-entropy minority at which the reasoning branches.

**Recommendation.** When entropy is falling and the accuracy curve has flattened, do not begin with `c_H`;
the four options below have reported gains in the same setting. If a bonus is used, treat 0.005-0.01 as the
region in which the §4.1 sweep observed entropy explosion or an unimproved baseline
([[entropy-mechanism-llm-rl]] §4.1, Fig. 9; the model used for that figure is not printed).

### 4.2 Covariance-outlier interventions: Clip-Cov and KL-Cov

Both act on a small selected fraction of batch tokens chosen by the centered covariance of Eq. 10.

- **Clip-Cov** (Eqs. 11-12): sample `⌊r·N⌋` token indices uniformly among tokens whose covariance lies in
  `[ω_low, ω_high]` and detach their policy gradient. Reported values: `r = 2×10⁻⁴`, `ω_low = 1`,
  `ω_high = 5`, both bounds more than 500× the average covariance (§4.3). The scale of the selection: at step
  1 on Qwen2.5-7B, the mean covariance of the top 0.02% of tokens is 5.654 against 0.003 over all tokens
  (Table 1).
- **KL-Cov** (Eqs. 13-14): the tokens with `rank(Cov) ≤ k·N` receive the extra loss term
  `−β·D_KL(π_θ_old ‖ π_θ)`, implemented in Listing 1 as `|log π − log π_old|`. Reported values:
  `k = 2×10⁻³` (7B), `2×10⁻⁴` (32B), `β = 1`. Note that this KL is to the *old* policy of the same step, not
  to a frozen reference.

Pseudocode for the selection rule, written from Eqs. 10-12 (the released implementation lives in verl's
`clip_cov` and `kl_cov` loss modes, adapted from the authors' repository, per [[entropy-logging-patterns]]):

```python
# per batch of N rollout tokens
cov = (logp - logp.mean()) * (adv - adv.mean())          # Eq. 10, both factors centered
band = (cov > omega_low) & (cov < omega_high)            # omega_low = 1, omega_high = 5
idx  = band.nonzero().flatten()
sel  = idx[torch.randperm(idx.numel())[: int(r * cov.numel())]]   # r = 2e-4, uniform in the band
mask = torch.zeros_like(cov, dtype=torch.bool); mask[sel] = True
pg   = -(adv * logp)
pg   = pg.masked_fill(mask, 0.0)                         # Clip-Cov: detach the selected tokens
```

Results, 7 math benchmarks, average ([[entropy-mechanism-llm-rl]] Table 2; DAPO-MATH prompts, 256 prompts ×
8 responses, temperature 1, 8 updates per rollout step, generation limit 8,192):
Qwen2.5-7B — GRPO 38.6, clip-higher 38.8, Clip-Cov 40.4, KL-Cov 40.6; Qwen2.5-32B — GRPO 45.8, clip-higher
47.2, Clip-Cov 50.3, KL-Cov 52.2. At the baseline's entropy plateau, the KL-Cov run keeps entropy more than
10× higher (§4.3). **Conditions and limits:** math only, one run per cell, no pass@k, and the authors report
no relationship between the controlled entropy level and final accuracy, calling the optimal entropy an open
question (§4.5).

### 4.3 Clip-higher and clip-lower

[[dapo]] raises the upper bound to `ε_high = 0.28` with `ε_low = 0.2` because at `ε = 0.2` a token at
`π_old = 0.01` can rise at most to 0.012 while a token at `π_old = 0.9` has a non-binding bound of 1.08; the
measured mean probability of up-clipped tokens is below 0.2 (§3.1, Fig. 3a). Adding clip-higher to DAPO's
progressive stack moved AIME24 avg@32 from 36 to 38 on Qwen2.5-32B (Table 1, one run per row). DAPO leaves
`ε_low` at 0.2 "because increasing it will suppress the probability of these tokens to 0, resulting in the
collapse of the sampling space" (§3.1).

[[clip-low-clip-high-entropy]] takes the other side of the same lever: *lowering* `ε_low` makes the clip-low
event more frequent, which raises entropy. With clip-high disabled (`ε_high = ∞`), `ε_low = 0.15` was the
value that avoided both collapse and entropy explosion for Qwen2.5-3B-Instruct on GSM8K (§3.2, Fig. 5), and
the entropy-controlled runs preserved pass@8 at comparable mean@8, while the symmetric `0.2 / 0.2` run lost
pass@8 over training (Figs. 6-7). The paper reports curves, not tables.

### 4.4 Restricting the update to high-entropy tokens

[[high-entropy-minority-tokens]] multiplies DAPO's token-level loss by `1[H_t^i ≥ τ_ρ^B]`, where `τ_ρ^B` is
the entropy threshold that selects the top-ρ fraction of tokens in the batch, and normalizes by the count of
selected tokens (Eq. 6). With `ρ = 20%` and no KL loss and no entropy loss, the 6-benchmark average moves
from 66.59 to 70.69 on Qwen3-32B base (AIME'24 55.83 → 63.54, AIME'25 45.63 → 56.67), from 61.40 to 64.39 on
Qwen3-14B, and from 53.71 to 54.23 on Qwen3-8B (Table 2, Acc@16 at temperature 1.0). Training on the
bottom-80% low-entropy tokens instead gives a substantial decline (§5.3). The gain grows with model size,
which the authors attribute to capacity (**Interpretation**, §5.3). On out-of-distribution LiveCodeBench the
top-10% and top-20% runs stay above the all-token run (§5.4, Fig. 9, no table values).

TRL exposes this as `top_entropy_quantile`, default 1.0 (no masking), with help text citing this paper and
recommending 0.2 ([[entropy-logging-patterns]]).

### 4.5 Changing the objective instead of the regularizer

**Pass@k training** ([[pass-at-k-training]]) rewards a *group* of k samples with 1 if any member is correct
and derives the advantage analytically: with `N` rollouts of which `N_neg` are wrong,
`R̄ = 1 − C(N_neg, k)/C(N, k)`, `σ = sqrt(R̄(1 − R̄))`, `Â_pos = (1 − R̄)/σ`, and
`Â_neg = (1 − R̄ − C(N_neg−1, k−1)/C(N−1, k−1))/σ` (Eqs. 11-15).

Worked example, `N = 8`, `k = 4`, binary reward:

| Group composition | Pass@1 (GRPO) `A_pos / A_neg` | Pass@k training `Â_pos / Â_neg` |
|---|---|---|
| 7 correct, 1 wrong (easy prompt) | +0.378 / **−2.646** | 0 / 0 (every group of 4 passes: `R̄ = 1`) |
| 4 correct, 4 wrong | +1.000 / −1.000 | +0.120 / −0.120 |
| 1 correct, 7 wrong (hard prompt) | +2.646 / −0.378 | +1.000 / −0.143 |

The row that matters for diversity is the first: under pass@1 the single deviating sample on an
already-solved prompt receives the largest negative advantage in the batch, which is exactly the
"below-average probability × negative advantage" quadrant that removes entropy. Pass@k training gives that
prompt no gradient at all. Reported effects: entropy stays at a higher level and rises from about step 200
(§3.2, Fig. 7b, no numbers); Qwen2.5-7B-Instruct Pass@1/Pass@k scores rise on in-domain Enigmata
(12.9/21.3 → 17.9/29.8) and on out-of-domain KORBench (37.7/45.6 → 47.7/63.5) and AIME 2025
(5.4/19.1 → 7.1/22.4) relative to pass@1 training (Table 1). Larger `k` shrinks every advantage, so the
authors raise the learning rate to compensate (§3.4).

**Unlikeliness reward** ([[unlikeliness-reward]]) attacks the same quadrant from the positive side in formal
theorem proving. The uplift rate — the fraction of correct samples at a given probability rank whose
probability rises during training — increases with the sample's initial probability, and the authors report
that the low-probability correct samples "are almost never uplifted" ("rank bias", §3.5, Fig. 4). The reward
of a correct sample is therefore scaled by
`1 − β_rank·(G − rank)/G` with rank 0 the most likely sample and `β_rank = 0.25`. With `G = 8` and correct
samples at ranks 0, 3 and 6, the group advantages move from `+1.291` for all three correct samples to
`+1.055`, `+1.283`, `+1.511` — the rarest correct proof now gets the largest push. Reported effects: pass@N
rises at large N with a small loss at pass@1 and pass@2 (Fig. 5); unique proofs per step drop and then
recover, unlike the monotone decline of the other variants (§5.3); on MiniF2F-test the final model reaches
pass@32 48.8 ± 0.7 and pass@128 50.6 ± 0.5 against DeepSeek-Prover-V1.5-RL's 49.2 ± 0.6 and 51.2 ± 0.3
(Table 3). The KL coefficient was also raised from 0.02 to 0.10 in these runs, which the authors say helped
prevent pass@N from deteriorating but was not sufficient by itself (§5, App. D) — so the rank term and the
KL change are not fully separated.

---

## §5 KL estimators: k1, k2, k3

**Problem.** The penalty needs `KL(q ‖ p)` where `q` is the policy being trained and `p` the reference, but
only sampled tokens and their log-probabilities are available. With `r = p(x)/q(x)` and samples `x ~ q`
([[john-schulman-kl-tricks]]):

| Estimator | Formula | Bias | Sign | Note |
|---|---|---|---|---|
| `k1` | `−log r` | none | any | negative for about half the samples |
| `k2` | `½(log r)²` | biased | ≥ 0 | its expectation is an f-divergence with `f''(1) = 1`, so it matches KL to second order |
| `k3` | `(r − 1) − log r` | none | ≥ 0 | `−log r + λ(r−1)` with `λ = 1`; the control variate `r − 1` has zero mean under `q` |

Two checks to do on paper. Unbiasedness of k3: `E_q[r − 1 − log r] = E_q[r] − 1 + KL(q‖p) = 0 + KL(q‖p)`,
because `E_q[r] = ∫ q·(p/q) = 1`. Non-negativity: `f(r) = r − 1 − log r` has `f(1) = 0`, `f'(r) = 1 − 1/r`,
`f''(r) = 1/r² > 0`, so `f` is convex with its minimum 0 at `r = 1`.

**Numbers from the blog's own experiment** (samples from `q`; the code sets `p = N(0,1)` and `q = N(0.1,1)`,
then `q = N(1,1)`; the surrounding prose names the two distributions in the opposite order, and the numbers
below are the ones the code produces):

| True KL | `k1` bias / stdev (relative) | `k2` | `k3` |
|---|---|---|---|
| 0.005 | 0 / 20 | 0.002 / 1.42 | 0 / 1.42 |
| 0.5 | 0 / 2 | 0.25 / 1.73 | 0 / 1.70 |

Panel 2 of [figures/entropy-dynamics.html](figures/entropy-dynamics.html) re-runs this experiment in the
browser so the bias-variance trade-off can be checked against the table above.

**What the blog does not claim.** It says nothing about which estimator to use in RLHF, about framework
defaults, or about the gradient of these expressions when they are used as a loss. The last point is the
subject of the next section, and it changes the answer.

---

## §6 Where the KL goes: reward stream, loss term, and what each gradient computes

**Two placements.**

- *KL in the reward* ([[rlhf-instructgpt]] Eq. 2; the lineage and its closed-form optimum
  `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)` are in [[kl-control-rlhf]], Korbak et al. Eq. 5): the per-token reward
  becomes
  `r̂_t = r_t − β·(log π_φ(y_t|y_<t,x) − log π_SFT(y_t|y_<t,x))`, and PPO's advantage estimator sees the
  shaped reward. InstructGPT used `β = 0.02` for all its RL models (App. C.4) and reports the optimal value
  as "around 0.01 and 0.02" on the human Likert score (App. E.7).
- *KL in the loss* ([[grpo]] §4.1.1, Eq. 4): the per-token loss gets `+ β·D̂_KL`, with `D̂` the k3 expression;
  DeepSeekMath-RL 7B used `β = 0.04` (§4.2). The paper's stated reason for the placement is "avoiding
  complicating the calculation of Â".

**The gradients are not the same, and not all of them estimate the reverse KL.** Let `q_θ` be the policy,
`p` the reference, sample `a ~ q_θ` with the sample treated as fixed (the usual implementation). Then:

- `k1` as a loss: `∇_θ (log q_θ(a) − log p(a)) = ∇_θ log q_θ(a)`, and `E_q[∇ log q_θ] = 0`. **A k1 loss term
  has zero expected gradient.** It logs the right value and trains nothing.
- `k2` as a loss: `∇_θ ½(log q_θ(a) − log p(a))² = (log(q_θ(a)/p(a)))·∇_θ log q_θ(a)`, whose expectation is
  `∇_θ KL(q_θ ‖ p)`. **k2 gives the reverse-KL gradient.**
- `k3` as a loss: `∇_θ (p/q_θ − 1 + log q_θ − log p) = (1 − p(a)/q_θ(a))·∇_θ log q_θ(a)`, whose expectation is
  `−E_p[∇_θ log q_θ] = ∇_θ KL(p ‖ q_θ)`. **k3 gives the forward-KL gradient**, which is mass-covering rather
  than mode-seeking.
- `k1` in the *reward*: the score-function gradient of `E_q[−β·log(q_θ/p)]` is
  `−β·E_q[log(q_θ/p)·∇ log q_θ]`, which is `−β·∇ KL(q_θ ‖ p)`. **k1 in the reward does give the reverse-KL
  gradient**, because the policy-gradient estimator supplies the missing factor.

verl's code states the same conclusion — k1 and k3 have the expected value of the KL but not its expected
gradient, k2 gives the right gradient, and a `+` suffix such as `k3+` uses k2 in the backward pass — and
OpenRLHF prints a recommendation of k2 or k3 as a loss and k1 in the reward ([[entropy-logging-patterns]]).
The practical reading: the number in the dashboard and the quantity being optimized can differ, so state
which estimator is in the loss and which is in the log.

**Defaults differ across frameworks** ([[entropy-logging-patterns]], code at pinned April 2026 commits): verl
has reference KL off by default in both placements (`kl_coef` 0.001 when enabled, reward-side estimator k1,
loss-side k3); OpenRLHF puts k1 in the reward with `init_coef` 0.01; TRL's PPO puts k1 or k3 in the reward
with `kl_coef` 0.05; TRL's GRPO puts k3 in the loss with β = 0.0, and with β = 0 it does not even load the
reference model. Comparing "the KL curve" across two runs from different stacks without checking these four
facts compares different quantities.

**Adaptive controllers raise β when measured KL exceeds the target.** verl's rule is
`β ← β·(1 + clip(KL/target − 1, −0.2, 0.2)·n_steps/horizon)`. With `β = 0.001`, `target_kl = 0.1`,
`horizon = 10000`, one step, and a measured KL of 0.3, the ratio is `0.3/0.1 − 1 = 2`, clipped to `0.2`, so
`β ← 0.001·(1 + 0.2·1/10000) = 1.00002×10⁻³`. The direction is upward and the per-step change is small by
construction; a run that drifts away from the reference is corrected over thousands of steps, not within one.

---

## §7 KL budget in practice: β = 0 recipes, reference resets, and retention

**The β = 0 position.** [[dapo]] §2.3 removes the KL term with the argument that a long-chain-of-thought
model is expected to move far from its initial distribution, so the restriction "is not necessary"; no
ablation with and without the term is reported, and the evaluation is AIME only. [[dr-grpo]] sets β = 0
throughout (§3, App. G Table 6). Olmo 3's released RL scripts set β = 0.0 at 7B and 32B
([[allenai-olmo3-open-instruct-scripts-recipe]]). DeepSWE sets `use_kl_loss=False` and leaves the in-reward
KL call commented out ([[deepswe-recipe]]). [[simpletir]] uses β = 0 and entropy coefficient 0. This is the
common setting in 2025-2026 reasoning recipes, and every one of those reports its results on in-domain
benchmarks.

**The β > 0 position.** [[prorl]] keeps a KL term in the loss (Eq. 4) and hard-resets the reference policy and
the optimizer when validation stagnates, because the KL term "may increasingly dominate the loss" (§2.3.1,
§3.3). The authors state the KL penalty gave "a stronger and more stable solution" than higher rollout
temperature or clip-higher for entropy (§2.2.1, §2.3.1), with no numeric comparison and no ablation of the
reset. Tülu 3's
final RLVR runs used β = 0.05 (8B) and β = 0.07 (70B) with a KL penalty in the PPO objective
([[tulu-3]], arXiv:2411.15124v5 Table 21 caption), and §6.2.1 reports that "incurring more KL budget does not
necessarily lead to improvements in verifiable rewards".

**Evidence that a KL penalty can cost accuracy.** In [[entropy-mechanism-llm-rl]] §4.1 (Fig. 10),
reference-KL coefficients between 0.001 and 0.1 stabilized entropy but lowered accuracy in that setting.

**Evidence that a KL penalty alone does not buy retention.** InstructGPT swept the KL reward coefficient up
to 2.0 — 100× its default — with the pretraining term switched off, and the regressions on public NLP
datasets remained, while a large coefficient lowered the validation reward; the pretraining data mix, not the
penalty, removed them ([[rlhf-instructgpt]] App. E.6). [[rls-razor]] gives the complementary positive result:
across SFT and RL runs, forgetting on held-out prior tasks is predicted by `E_{x~τ}[KL(π_0 ‖ π)]` measured on
the *new* task, with `R² = 0.96` (controlled ParityMNIST/FashionMNIST setting) and `R² = 0.71` (LLM sweep with
Qwen2.5-3B-Instruct on math, chemistry and tool use). The two results are consistent: what predicts forgetting is the distance
actually travelled, and a penalty coefficient is only an indirect and weakly calibrated way to control it.

**What to do with this.** Log the realized KL to the starting policy on a fixed held-out prompt set, and plot
retention on a multi-domain held-out suite against that measured KL, rather than assuming a β value implies a
budget. Whether β = 0 recipes lose retention at the same measured KL as β > 0 recipes is an **open question**:
no source in this library runs that comparison.

---

## §8 Entropy, low-probability tokens, and tool-call boundaries

Agentic RL breaks an assumption of §2: part of the context is written by the environment, not by the policy.
[[simpletir]] measures what follows in multi-turn tool-integrated reasoning from a Qwen2.5-7B base model.

1. The interpreter's output is concatenated into the next turn's prompt and can be far from the pretraining
   distribution; the model's own continuations then contain tokens to which it assigns low probability. In the
   traced rollout, probabilities are high in turn 1, low-probability segments appear in the model's own text
   in turns 2 and 3, and turn 4 collapses (§3.1, Fig. 3). Masking the loss on the tool tokens does not stop
   this, because the drift is in the model's own continuation.
2. Proposition 3.1 gives the per-token gradient norm with respect to the logits:
   `(m_{i,t}/Σ_j m_{i,j})·ρ_{i,t}(θ)·g_{i,t}·|Â_i|·sqrt(1 − 2P(c) + Σ_j P(j)²)`, with `m` the feedback mask,
   `ρ` the importance ratio, `g` the unclipped gate, `Â` the trajectory advantage, and `P` the policy
   distribution at that position. Worked example with the five-token distribution of §2
   (`Σ_j P(j)² = 0.5346`): for the token at `P(c) = 0.01` the square-root factor is
   `sqrt(1 − 0.02 + 0.5346) = 1.231`; for the token at `P(c) = 0.70` it is `sqrt(1 − 1.40 + 0.5346) = 0.367`.
   A rare token contributes about 3.4× the per-token gradient norm of the mode, before the importance ratio
   is applied — and for a negative-advantage trajectory that ratio is unbounded above and largest exactly
   where `π_old` was small.
3. The fix used is trajectory filtering: a turn that yields neither a complete code block nor a final answer
   is a "void turn", and any trajectory containing one is removed from the batch. Filtering by lowest token
   probability or by highest importance ratio did not stabilize training. Highest score within 1,000 steps:
   SimpleTIR AIME24 50.5 / Math500 88.4; naive multi-turn 20.8 / 73.1; low-probability filtering 23.3 / 72.8;
   high-ratio filtering 26.3 / 75.0; stopping generation on a void turn without filtering 26.1 / 77.3
   (Table 2). Settings: rollout temperature 1.0, clip 0.2 / 0.28, entropy coefficient 0, β = 0, 16 rollouts
   per prompt, PPO epochs 4 (App. Table 6).

Two related choices in agentic runs: DeepSWE trains with no KL loss and no entropy loss and relies on
temperature 1.0 rollouts ([[deepswe-recipe]]); Kimi K2 uses a "temperature decay schedule, to shift from exploration
to exploitation throughout the training", with the values not printed in the report ([[kimi-k2-recipe]], §3.2.3). The claim that
policy entropy *spikes* immediately after tool feedback is not measured by any verified source in this
library — SimpleTIR measures token probabilities and gradient norms, not entropy — so treat it as an **open
question** and measure it in your own run. When you do, compute entropy only over positions the policy
generated: including masked observation tokens in the average mixes environment text into a policy statistic.
ch-45b covers observation masking and turn-level credit assignment.

---

## Negative samples and negative feedback

**Which kind of negative.** This chapter deals with type (4), *negative as gradient*: tokens and sequences
whose likelihood the update decreases, through a negative advantage. It also touches type (1), *negative
marginal value*: samples removed from the batch entirely (void-turn filtering in [[simpletir]], zero-variance
group filtering in [[prorl]] §2.3 and ch-16). Types (2) and (3) — negatives as content or as conditioning —
belong to ch-31a. ch-43a holds the full derivations.

**Where negatives come from at this stage.** A verifier or judge labels a rollout; the group baseline turns
below-average rewards into negative advantages. With a binary verifier and a group of `G`, every group that
is neither all-correct nor all-wrong produces both signs.

**Mechanism.** For the softmax, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, so decreasing the log-probability of a
sampled token adds mass to every other token in proportion to its current probability. The consequence for
entropy is the second row of the §3 table: a push-down on an above-average-probability token raises entropy;
a push-down on a below-average-probability token lowers it, because most of the freed mass returns to the
mode. The numeric example in §3 shows 0.0082 of the 0.0113 freed units returning to the token at
`π = 0.70`.

**Evidence with numbers.**

- Removing the lower clip entirely (`ε_low = 1.0`) lowered entropy, and strengthening it (`ε_low = 0.15`)
  raised entropy to the point where pass@8 was preserved at equal mean@8
  ([[clip-low-clip-high-entropy]] §3.2, Figs. 5-6). The lower clip is the bound that limits per-update
  suppression of negative-advantage tokens.
- DAPO keeps `ε_low` at 0.2 with the stated reason that raising it suppresses low-probability tokens to zero
  and collapses the sampling space ([[dapo]] §3.1).
- The dual clip caps the loss of a negative-advantage token at `c·|A|` with verl's default `c = 3.0`, taken
  from the Honor of Kings dual-clip PPO paper, which introduced it because the ratio is unbounded above for
  negative advantages with off-policy data ([[verl-ppo-loss]]).
- Pass@k training removes the large negative advantage on the lone deviating sample of an already-solved
  prompt (worked example in §4.5: `−2.646 → 0`).
- In multi-turn tool use, negative-advantage trajectories are where the unbounded importance ratio and the
  low-probability factor multiply ([[simpletir]] Prop. 3.1).

**Controls that make negatives safer at this stage:** keep the data on-policy; bound the update with the
lower clip and the dual clip; select which tokens are updated (high-entropy masking, covariance outliers);
change the group objective so that easy prompts stop producing large penalties (pass@k training); drop
trajectories whose failure is structural rather than substantive (void-turn filtering).

**Diagnostics.** Log entropy, gradient norm and clip fraction split by advantage sign. verl logs
`actor/pg_clipfrac_lower` for negative-advantage tokens under the dual clip; TRL GRPO logs
`clip_ratio/low_mean` separately from `clip_ratio/high_mean` ([[entropy-logging-patterns]]). Add pass@k at a
large k and a `distinct_k` measurement on a fixed open-ended prompt set.

**Honesty about size of effect.** No source cited in this chapter measures what share of the accuracy gain
comes from negative-advantage tokens; the measured decompositions are in ch-43a. What the sources here
support is narrower: the *entropy* and *coverage* effects of negatives depend on which quadrant the
pushed-down token sits in, and bounding or re-weighting the negative side changed pass@k in
[[clip-low-clip-high-entropy]] and [[pass-at-k-training]].

---

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| InstructGPT (PPO-ptx) | 1.3B-175B | RL | KL placement; β | per-token KL in the reward (Eq. 2); 0.02 | arXiv:2203.02155v1 App. C.4 | verified 2026-09-15 | App. E.7: Likert score poor at 0 and at 2, optimal "around 0.01 and 0.02" |
| InstructGPT (PPO, γ = 0) | 1.3B | RL | KL sweep for retention | up to 2.0 | arXiv:2203.02155v1 App. E.6 | verified 2026-09-15 | regressions on public NLP datasets persisted at 100× β; fixed by the pretraining mix instead |
| DeepSeekMath-RL | 7B | RL | KL placement; β; estimator | KL in the loss; 0.04; Eq. 4 (k3 form) | [[grpo]] §4.1.1, §4.2 | verified 2026-09-14 | no ablation reported |
| Tülu 3 RLVR (final) | 8B; 70B | RL | β; warmup ratio; generation temperature | 0.05 (8B), 0.07 (70B); 0.0 / 0.07; 1.0 | [[tulu-3]], arXiv:2411.15124v5 Table 21 | verified 2026-09-15 | β swept over [0.1, 0.05, 0.03, 0.01]; §6.2.1: more KL budget did not always raise verifiable reward |
| DAPO (Qwen2.5-32B base) | 32B | RL | KL; ε_low / ε_high; rollout | none; 0.2 / 0.28; 512 prompts × 16 | [[dapo]] §2.3, §4.1 | verified 2026-09-15 | Table 1: clip-higher 36 → 38 AIME24 avg@32, one run per row; no ablation of KL removal |
| Dr. GRPO runs | 1.5B-7B | RL | KL loss coef; KL penalty coef; temperature; top-p / top-k | 0.0; 0.0; 1.0; 1.0 / −1 | [[dr-grpo]] §3, App. G Table 6 | verified 2026-09-14 | no ablation in this paper |
| Cui et al. runs (Qwen2.5) | 7B, 32B | RL | Clip-Cov r; ω_low; ω_high | 2×10⁻⁴; 1; 5 | [[entropy-mechanism-llm-rl]] §4.3 | verified 2026-09-14 | Fig. 12 (7B): entropy rises with more clipped tokens; accuracy versus r not reported |
| Cui et al. runs (Qwen2.5) | 7B; 32B | RL | KL-Cov k; β | 2×10⁻³ (7B), 2×10⁻⁴ (32B); 1 | [[entropy-mechanism-llm-rl]] §4.3 | verified 2026-09-14 | Table 2: 7-benchmark average 38.6 → 40.6 (7B), 45.8 → 52.2 (32B) |
| Cui et al. §4.1 sweep | not stated (the §4 experiments use Qwen2.5-7B and 32B) | RL | entropy-loss coefficient sweep | {0.0001, 0.001, 0.005, 0.01} | [[entropy-mechanism-llm-rl]] §4.1, Fig. 9 | verified 2026-09-15 (model not printed for Fig. 9) | Fig. 9: 0.01 entropy explosion; 0.005 stable but no gain |
| Forking-token DAPO (Qwen3 base) | 8B, 14B, 32B | RL | ρ; ε_low / ε_high; KL; entropy loss; LR | 20%; 0.2 / 0.28; none; none; 1e-6 constant | [[high-entropy-minority-tokens]] §5.2 | verified 2026-09-15 | Table 2: 32B average 66.59 → 70.69; §5.3: ρ = 10/50/100% all lower entropy than 20% |
| Park et al. GSM8K run (Qwen2.5-3B-Instruct) | 3B | RL | ε_high; ε_low; KL; entropy loss; LR; temperature | ∞; 0.15; none; none; 5e-7; 1.0 | [[clip-low-clip-high-entropy]] §3.2, App. C.1 Table 1 | verified 2026-09-15 | Fig. 5: balances collapse against explosion; Fig. 6: pass@8 preserved at equal mean@8 |
| Nemotron-Research-Reasoning-Qwen-1.5B (ProRL) | 1.5B | RL | KL placement; β; reference reset; temperature | KL in the loss (Eq. 4); not reported; hard reset of π_ref and optimizer on stagnation; 1.2 | [[prorl]] §2.3.1, §3.2, §3.3 | β not reported (body, App. D-F, model card checked) | no ablation of KL, reset, or temperature |
| Olmo 3 RL (released scripts) | 7B, 32B | RL | KL coefficient; temperature; clip | β 0.0 (default 0.05 unused); 1.0; 0.2 / 0.272 | [[allenai-olmo3-open-instruct-scripts-recipe]] | verified 2026-09-14 | no ablation reported |
| DeepSWE-Preview | 32B | RL | KL; entropy coefficient; temperature | `use_kl_loss=False`, in-reward KL commented out; 0.0; 1.0 | [[deepswe-recipe]] | verified 2026-09-14 | Blog §2.3: entropy loss led to rising entropy and collapse; curves only |
| SimpleTIR (Qwen2.5-7B base) | 7B | RL | β; entropy coefficient; clip; rollouts; PPO epochs | 0; 0; 0.2 / 0.28; 16 per prompt; 4 | [[simpletir]] App. Table 6 | verified 2026-09-15 | Table 2: void-turn filtering 50.5 AIME24 versus 20.8-26.3 for the alternatives |
| Kimi-K2-Instruct | 1.04T / 32B active | RL | Rollout temperature | decay schedule from high to lower (values not printed) | [[kimi-k2-recipe]] §3.2.3 | verified 2026-09-14 (mechanism) | no ablation reported |
| GRPO-Unlikeliness-2 (DeepSeek-Prover-V1.5-SFT) | 7B | RL | β_KL; β_rank; PPO epochs | 0.10; 0.25; 2 | [[unlikeliness-reward]] Table 1 | verified 2026-09-15 | Table 2: 8,065 of 9,600 problems solved versus 7,860 for GRPO-Default; Table 3 pass@128 |
| verl default config | any | RL | reference KL; estimator; entropy coefficient | off in both placements (`kl_coef` 0.001, reward k1, loss k3); 0 | [[entropy-logging-patterns]] | verified 2026-09-14 | no ablation reported |
| OpenRLHF CLI default | any | RL | reference KL; estimator; entropy coefficient | in the reward, `init_coef` 0.01; k1; None | [[entropy-logging-patterns]] | verified 2026-09-14 | no ablation reported |
| TRL `GRPOConfig` default | any | RL | reference KL β; estimator; `top_entropy_quantile` | 0.0 (reference model not loaded); k3; 1.0 | [[entropy-logging-patterns]] | verified 2026-09-14 | help text recommends 0.2 for entropy masking, citing [[high-entropy-minority-tokens]] |

**Starting point for a small general-purpose run.** Every number here comes from a `verified` row above, with
the conditions of that row. For a 7B-class model on verifiable prompts with group-baseline RL: rollout
temperature 1.0 (Dr. GRPO 1.5B-7B; Olmo 3 7B; SimpleTIR 7B) with top-p 1.0 (Dr. GRPO 1.5B-7B), `ε_low = 0.2` and
`ε_high = 0.28` (DAPO at 32B; reused at 7B-32B by the forking-token and SimpleTIR runs), entropy coefficient
0 (verl default; DeepSWE 32B; SimpleTIR 7B), and either β = 0 with a measured KL logged against the starting
policy (Olmo 3 7B/32B, Dr. GRPO) or a small in-reward β near 0.02-0.05 if retention on tasks outside the RL
data is part of the acceptance gate (InstructGPT 1.3B-175B; Tülu 3 8B). If entropy falls while accuracy
flattens, the first change with reported gains at this scale is restricting the policy gradient to the
highest-entropy 20% of batch tokens (Qwen3-8B/14B/32B base, math prompts) or penalizing covariance outliers
with KL-Cov `k = 2×10⁻³`, `β = 1` (Qwen2.5-7B, math prompts).

---

## Generalization lens

**(a) What increases breadth.**
- Objectives that reward coverage: pass@k training raised out-of-domain KORBench Pass@1/Pass@k from
  37.7/45.6 to 47.7/63.5 and AIME 2025 from 5.4/19.1 to 7.1/22.4 relative to pass@1 training on the same
  prompts ([[pass-at-k-training]] Table 1, one run per cell).
- Entropy control through clipping preserved pass@8 at equal mean@8 where the symmetric default lost it
  ([[clip-low-clip-high-entropy]] Figs. 5-6).
- Restricting the update to high-entropy tokens improved an out-of-domain code benchmark for a math-trained
  policy ([[high-entropy-minority-tokens]] §5.4, Fig. 9, curves only).
- Long RL with a KL term and reference resets produced gains on held-out Reasoning Gym tasks, including one
  where the starting checkpoint scored 0.00 at every k up to 256 ([[prorl]] §4.3, Table 3).

**(b) What causes narrowing or forgetting.**
- Sustained positive advantage on already-likely tokens, and negative advantage on already-unlikely tokens
  (§2-§3); under symmetric clipping the clip-high bias adds to this even when the reward carries no
  information ([[clip-low-clip-high-entropy]] §2.3).
- Rank bias: under GRPO the uplift rate of a correct sample rises with its initial probability, and the
  lowest-probability correct samples "are almost never uplifted" ([[unlikeliness-reward]] §3.5, Fig. 4).
- Post-training stages in sequence reduced `distinct_10` at every OLMo 2 size ([[noveltybench]] Fig. 6).
- Distance travelled from the base policy predicts forgetting on prior tasks ([[rls-razor]] §3), and a larger
  KL coefficient by itself did not repair the regressions ([[rlhf-instructgpt]] App. E.6).
- In parts of long runs, pass@1 rises while pass@128 falls on tasks the starting model already covered
  ([[prorl]] §4.2); [[rlvr-beyond-base-model]] is the pass@k critique that both [[prorl]] and
  [[entropy-mechanism-llm-rl]] §2.6 respond to.

**(c) How to measure it at this stage.** Report, on a fixed schedule: per-token entropy (mean and an upper
quantile, computed on policy-generated positions only); measured KL to the starting policy on a held-out
prompt set, not the β value; pass@1 and pass@k at a large k on held-out prompts; `distinct_k` or an
equivalent diversity metric on open-ended prompts at temperature 1.0 ([[noveltybench]] §3); and a
multi-domain retention suite that includes domains absent from the RL prompt mix. Known measurement errors:
entropy averaged over masked observation tokens is not a policy statistic (§8); pass@k estimated from few
samples is high-variance; `distinct_k` depends on the partitioner and on the sampling temperature
([[noveltybench]] §3, §4.1).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Ranking tokens by `p·A` when selecting "high-covariance" tokens | Selected set contains only positive-advantage tokens; entropy keeps falling | Print the advantage sign histogram of the selected set; compare with the centered product of Eq. 10 ([[entropy-mechanism-llm-rl]]) |
| Reading a KL number from the dashboard as the quantity being optimized | Two runs with the same "kl" value behave differently | Check estimator, the two policies compared, and placement (reward or loss) for each stack ([[entropy-logging-patterns]]) |
| Using k1 as a loss term | The KL curve looks controlled but the policy moves as if unregularized | The expected gradient of a k1 loss is zero (§6); switch to k2 in the backward pass or move k1 into the reward |
| Assuming β = 0 costs nothing because in-domain scores rose | Held-out or out-of-domain scores fall while AIME-style scores rise | Measure realized KL to the starting policy and run a retention suite ([[rls-razor]], [[rlhf-instructgpt]] App. E.6) |
| Raising the entropy coefficient as the first response to collapse | Entropy rises then explodes; response length and reward degrade | Sweep with the reported outcomes in mind (0.01 caused entropy explosion in [[entropy-mechanism-llm-rl]] §4.1; 0.005 destabilized in [[pass-at-k-training]] §3.1); prefer selective interventions |
| Treating entropy as a proxy for output diversity | Entropy is stable while the model returns near-duplicate answers | Measure `distinct_k` or pass@k directly ([[noveltybench]], [[clip-low-clip-high-entropy]] Fig. 6) |
| Averaging entropy over tool-output positions in agentic runs | Entropy curve moves with tool verbosity, not with the policy | Apply the feedback mask before averaging (§8, [[simpletir]]) |
| Comparing pass@k across runs at different sampling temperatures | Coverage differences that disappear when temperature is fixed | Fix temperature and top-p for every evaluation; report them |

---

## Check your understanding

1. The covariance result says entropy falls when log-probability and advantage covary positively. Explain why
   a *negative* advantage on a low-probability token still lowers entropy, and where the removed probability
   mass goes.
2. A colleague proposes to fight entropy collapse by zeroing the gradient of the top 2% of tokens ranked by
   `p_t·A_t`. Using the four-token example of §3, explain what this rule can and cannot select, and what it
   would miss in a run whose collapse is driven by suppression of rare tokens.
3. Why does a k1 term added to the loss produce a controlled-looking KL curve and no regularization, while
   the same expression added to the reward does regularize? Answer with the gradient of each.
4. DAPO raises `ε_high` and Park et al. lower `ε_low`; both report higher entropy. Explain how two changes to
   opposite ends of the clip range can have the same sign of effect on entropy.
5. Pass@k training gives zero advantage to a group of 8 samples in which 7 are correct when `k = 4`. Explain
   why that is the case arithmetically, and why it matters for pass@k at large k.
6. InstructGPT raised the KL coefficient 100× and the regressions on public NLP datasets remained, while
   RL's Razor reports that KL on the new task predicts forgetting with `R² = 0.71` on LLMs. Reconcile the two
   results.
7. In multi-turn tool use, why does masking the loss on tool-output tokens fail to prevent the gradient-norm
   explosion that SimpleTIR describes? Use Proposition 3.1 in your answer.
8. You have two runs with identical mean entropy curves; one produces near-duplicate answers on open-ended
   prompts and the other does not. Give a mechanism consistent with the token-entropy statistics of
   [[high-entropy-minority-tokens]], and the measurement that would distinguish them.

---

## Connections

- Previous: **ch-16 — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse.**
  Difficulty filtering decides which prompts produce groups with both advantage signs; zero-variance groups
  carry no gradient and therefore no entropy change.
- Next: **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative
  Advantages.** The softmax redistribution sketched in §3 and in the negative-feedback section is derived
  there, together with the offline analogues.
- **ch-37 — Policy-Gradient Foundations for Language Models.** The estimator whose covariance with
  log-probability this chapter analyses.
- **ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax.** The KL-in-reward convention and
  the alignment-tax measurements that §7 draws on.
- **ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** The
  retention evidence (RL's Razor, diversity loss under alignment) that makes the KL budget a generality
  control.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.** Where the group baseline, clip-higher,
  and the β = 0 recipes are introduced as algorithms.
- **ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing.** Breadth of
  the reward signal, which interacts with everything in §7.
- **ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.** The agentic
  setting of §8 in full.
- **ch-55 — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation.** The
  `kl_penalty`, `clip_cov` and `kl_cov` code paths referenced here.

---

## Sources

- [[entropy-mechanism-llm-rl]] — the entropy-performance fit, the covariance theorems, Clip-Cov and KL-Cov
  with their selection rules and results; also the reported outcomes of entropy-loss and reference-KL sweeps.
- [[high-entropy-minority-tokens]] — token-entropy statistics of a chain of thought, the forking-token
  masking rule (Eq. 6), its results across Qwen3 sizes, and the entropy-bonus-versus-clip-higher argument.
- [[clip-low-clip-high-entropy]] — theorems and experiments showing that clip-low raises entropy and
  clip-high lowers it, and that entropy control through clipping preserves pass@k.
- [[dapo]] — the stated reason for removing the KL term and the clip-higher mechanism with its values.
- [[pass-at-k-training]] — the group-level objective, the analytical advantage, entropy behaviour and
  in-domain / out-of-domain results.
- [[unlikeliness-reward]] — rank bias in GRPO, the rank-scaled reward, and pass@N results in theorem proving.
- [[noveltybench]] — `distinct_k` and `utility_k`, and the per-stage diversity drop on OLMo 2.
- [[rls-razor]] — forgetting predicted by KL between the fine-tuned and base policy measured on the new task.
- [[john-schulman-kl-tricks]] — the k1/k2/k3 estimators and the bias/variance numbers used in §5.
- [[kl-control-rlhf]] — the KL-regularized objective and its closed-form optimum (Korbak); used with the
  InstructGPT loci given in §6.
- [[rlhf-instructgpt]] — β = 0.02 in the reward, the optimal range, and the KL sweep that did not restore
  capability.
- [[grpo]] — KL in the loss with the Eq. 4 estimator and DeepSeekMath-RL's β = 0.04.
- [[dr-grpo]] — a recipe with β = 0 throughout.
- [[prorl]] — KL in the loss with reference-policy resets, and the pass@k regimes of a long RL run.
- [[tulu-3]] — RLVR β values and the observation that more KL budget did not always raise verifiable reward.
- [[allenai-olmo3-open-instruct-scripts-recipe]] — released RL scripts with β = 0.0 and temperature 1.0.
- [[deepswe]], [[deepswe-recipe]] — an agentic run with no KL loss and no entropy loss, and the stated reason.
- [[kimi-k2-recipe]] — rollout-temperature decay as a disclosed RL mechanism.
- [[simpletir]] — low-probability tokens after tool feedback, the gradient-norm proposition, and void-turn
  filtering.
- [[entropy-logging-patterns]] — what verl, OpenRLHF and TRL compute, log and default to, including the
  adaptive-KL formula and the advantage-sign diagnostics.
- [[verl-ppo-loss]] — the clipped surrogate with asymmetric clipping and the dual clip on negative advantages.
- [[entropy-regularization-ppo]] — the origin and coefficients of the entropy bonus (A3C, crediting
  Williams & Peng 1991).
- [[entropy-collapse-ppo]] — the continuous-control sweep in which no regularizer helped significantly except
  on one task.
- [[maximum-entropy-rl]] — the maximum-entropy objective and temperature sensitivity in SAC, and why the
  analogy to LLM-RL is limited.
- [[rlvr-beyond-base-model]] — the pass@k critique that the coverage discussion responds to; cited for its
  claim, not for numbers (the card has not been re-verified).
- [[nathan-lambert-entropy-rl]], [[openrlhf-entropy-debugging]] — practitioner syntheses with no single
  checkable artifact and no verification record. Cited as framing only; their entropy thresholds (0.1 and 0.2
  nats) have no located primary source and are not used in this chapter.
