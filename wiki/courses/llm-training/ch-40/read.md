<!-- chapter: ch-40
     track: rl
     kind: content
     title: Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO
     deps: [ch-42a]
     sources: [[rloo]], [[rloo-vs-grpo]], [[reinforce-plus-plus]], [[grpo]], [[deepseekmath]], [[dr-grpo]], [[dapo]], [[gspo]], [[vapo]], [[spurious-rewards-rlvr]], [[reinforcement-learning-with-one-training-example]], [[octothinker]], [[nathan-lambert-grpo]], [[deepswe]], [[deepswe-recipe]], [[gigpo-verl-agent]], [[negative-sample-reinforcement]], [[john-schulman-kl-tricks]], [[trl-grpo]], [[verl-grpo]], [[deepseek-r1-recipe]], [[magistral-recipe]], [[asymmetric-reinforce]], [[likelihood-displacement]]
     figures: figures/group-baseline.html
     revised: 2026-09 (generality revision)
-->

# Chapter 40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO

> **Core insight.** A group-baseline algorithm replaces the learned value model of PPO with a statistic of several responses sampled for the same prompt. The variants differ in four places: which baseline is subtracted (leave-one-out mean, group mean, batch mean), whether the advantage is divided by a spread estimate, how token losses are aggregated into a scalar, and where the KL term is applied. Each of those choices changes the gradient a specific sample receives, and the published corrections — Dr. GRPO's two removed divisions, DAPO's four changes, GSPO's sequence-level ratio — are each tied to a measured failure: rising incorrect-response length, entropy collapse, zero-gradient groups, and collapse of Mixture-of-Experts training.
>
> **Guideline.** When the reward is a verifiable 0/1 or ±1 signal and responses are long, use a group baseline without standard-deviation division and with a length-independent loss normalizer, because [[dr-grpo]] reports equal or higher accuracy with shorter incorrect responses on Qwen2.5-1.5B (§3.2, Fig. 5; App. C Fig. 8, and Fig. 9 over 3 runs) and [[dapo]] reports AIME 2024 avg@32 rising from 30 to 50 on Qwen2.5-32B as its four changes are added (Table 1). When the reward is a continuous reward-model score and groups often contain near-ties, keep a spread normalizer at the batch level rather than the group level, because group-level division turns a 0.02 reward difference into a full-magnitude advantage (§5 worked example) and [[reinforce-plus-plus]] argues the group-level statistic is biased for any finite group (App. A.1, Thm. 1). When training a large Mixture-of-Experts model with several gradient updates per rollout batch, use a sequence-level importance ratio, because [[gspo]] reports that token-level ratios required a routing-replay workaround for GRPO to converge on Qwen3-30B-A3B (§5.3). When a value model can be trained and warmed up, do not assume the critic is obsolete: [[vapo]] reports AIME 2024 avg@32 60.4 with a value model against 50 for DAPO on the same base model (§5.2, Table 1).

---

## Why this chapter matters for a general-purpose model

Reinforcement learning is applied after SFT (or directly to a base model) on a prompt set that is narrow compared with the range of tasks the released model must handle. Every open recipe quoted in this chapter trains on one such slice: mathematics questions ([[grpo]] §4.2, [[dr-grpo]] §3, [[dapo]] §3.5), SWE-Bench-style repository tasks ([[deepswe]] §2.1), or one agent environment family ([[gigpo-verl-agent]] App.). The optimizer therefore sees a narrow slice of the input distribution while it changes every parameter. Three properties of the algorithm decide how much of the model's breadth survives:

1. **Which samples receive a gradient.** A group whose responses all get the same reward produces a zero advantage and contributes nothing (§5). If most of the prompt set is too easy or too hard, the effective batch shrinks and the update is driven by a small, possibly unrepresentative subset.
2. **How negative advantages are applied.** Pushing down an incorrect response moves probability mass to whatever the model already finds likely (§ Negative samples). Done without controls this reduces output diversity, which is measured as a drop in pass@k at large k ([[negative-sample-reinforcement]]).
3. **How the loss is normalized.** Length normalization decides whether long incorrect responses are penalized in proportion to their length. When they are not, the per-token penalty on a long incorrect response is smaller than on a short one, and the length of incorrect responses keeps rising after the training reward stops improving ([[dr-grpo]] §3.1, Fig. 5).

The chapter also answers a measurement question that recurs in later chapters: RL on math prompts raised DeepSeekMath 7B's Maj@K but not its Pass@K ([[grpo]] §5.2.2, Fig. 7), and several published gains on Qwen2.5-Math do not reproduce on Llama or OLMo ([[spurious-rewards-rlvr]] §3). An algorithm ranking measured on one model family is not evidence about another.

---

## §1 The baseline problem: from a learned critic to sampled peers

**Definition.** The policy-gradient estimator for a sequence-level reward is `∇J = E[(R(x, y) − b) ∇ log π_θ(y | x)]`, where `x` is a prompt, `y` a sampled response, `R` the reward, and `b` a baseline that does not depend on `y`. Any such `b` leaves the estimator unbiased and changes its variance ([[rloo]] §2.2–2.3, recapitulated in ch-37).

**The measurable problem.** Without a baseline, every sample of a prompt is pushed in the direction of its own reward sign. With a reward model whose scores on one prompt are `{5.2, 5.0, 4.9, 4.9}`, all four responses are reinforced, and only the differences between them carry information. The estimator's variance is then dominated by the prompt-to-prompt level of `R`, not by the within-prompt differences the update should follow.

**PPO's answer and what changes for language models.** PPO estimates `b` with a learned value model `V_φ(s_t)` and combines per-step errors with GAE: `Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}`, `δ_l = R_l + γV(s_{l+1}) − V(s_l)` ([[dapo]] Eqs. 2–3, quoting Schulman et al.). Two properties of the LLM setting change the calculation:

- **Terminal reward.** If the only non-zero reward arrives at the final token and `γ = λ = 1`, the sum telescopes to `Â_t = R − V(s_t)`: each token's advantage is the terminal reward minus the value of the state at that token. It is not `R − V(s_0)`; the value model is still queried at every position. With `λ < 1` the estimate mixes in bootstrapped values, and with a per-token KL penalty added to the reward (the InstructGPT and RLOO formulation, ch-38) the reward is no longer terminal at all.
- **Value-model error.** A value model of policy size is expensive and hard to fit on long chains. [[vapo]] reports vanilla PPO on Qwen2.5-32B reaching AIME 2024 avg@32 of 5 because of value-model collapse (§5.2, Table 1).

**What replaces it.** If the only quantity needed is `V(prompt) = E_{y∼π}[R(x, y)]`, sampling `G` responses per prompt gives a direct Monte-Carlo estimate of it. That is the group baseline. The order of the literature is: the leave-one-out estimator is from Kool et al. 2019, cited as its source by [[rloo]] §2.3; DeepSeekMath's GRPO appeared as arXiv:2402.03300 (February 2024, [[grpo]]); Ahmadian et al.'s RLOO study appeared as arXiv:2402.14740 later the same month ([[rloo]]).

**Conditions and limits.** Removing the critic is not a settled result. On ALFWorld with Qwen2.5-7B-Instruct, PPO with a critic scores 80.4 overall against GRPO's 77.6 and RLOO's 75.5, while at 1.5B the ordering reverses (PPO 54.4, GRPO 72.8) ([[gigpo-verl-agent]] Table 1, 3 seeds). [[vapo]] reports a value-model system above DAPO at 32B (§5.2). [[deepseek-r1-recipe]] reports the opposite direction on a 16B MoE: PPO with GAE λ = 0.95 is worse than GRPO on MATH, and λ = 1.0 is close to GRPO (v2 App. A.3, Fig. 4). Result (single study) in each case, on different tasks and scales.

---

## §2 RLOO: the leave-one-out baseline

**Definition.** Sample `k` responses `y_1 … y_k ∼ π_θ(· | x)`, score them, and use for each sample the mean reward of the *other* `k − 1` samples:

```
b_i = (1/(k−1)) Σ_{j≠i} R_j
∇J_RLOO ≈ (1/k) Σ_i [ R_i − b_i ] ∇ log π_θ(y_i | x)
```

`R_i = R(x, y_i)`; `k` is the number of online samples per prompt. The estimator is quoted at [[rloo]] §2.3.

**Why the exclusion matters.** `b_i` is a function of `{R_j : j ≠ i}` only, so it is independent of `y_i` given `x` and the estimator stays unbiased. Including `R_i` in the baseline makes the baseline depend on the sample it corrects; the resulting estimator points in the same direction but is scaled (§6).

**Worked example (k = 4, continuous reward).** Rewards `{2.0, 1.0, 0.5, 0.5}`. Mean is 1.0.

| Sample | `R_i` | `b_i` (mean of others) | RLOO advantage | `R_i − mean` |
|---|---|---|---|---|
| 1 | 2.0 | (1.0+0.5+0.5)/3 = 0.667 | +1.333 | +1.0 |
| 2 | 1.0 | (2.0+0.5+0.5)/3 = 1.0 | 0.0 | 0.0 |
| 3 | 0.5 | (2.0+1.0+0.5)/3 = 1.167 | −0.667 | −0.5 |
| 4 | 0.5 | 1.167 | −0.667 | −0.5 |

Every RLOO advantage is `4/3` times the mean-subtracted one. The identity is exact at every `k`:

```
R_i − (1/(k−1)) Σ_{j≠i} R_j = (k R_i − k R̄)/(k−1) = k/(k−1) · (R_i − R̄)
```

[[dr-grpo]] states the same relation with `G/(G−1)` (§3.2, App. A). The two estimators differ by a constant factor, not by a term that vanishes as `k` grows.

**The k = 2 case.** `b_1 = R_2` and `b_2 = R_1`, so the advantages are `R_1 − R_2` and `R_2 − R_1`: a pairwise comparison used as a policy-gradient weight rather than as a preference-loss term (ch-39).

**Evidence.** Pythia-6.9B and Llama-7B on TL;DR summarization and Anthropic-HH, simulated win-rate against reference completions (Table 1 of arXiv:2402.14740v2; the three numbers per row are TL;DR, Anthropic-HH with Pythia-6.9B, and Anthropic-HH with Llama-7B): RLOO k=4 77.9 / 43.7 / 64.1; RLOO k=2 74.2 / 47.6 / 62.2; RAFT k=4 73.2 / 42.1 / 63.3; REINFORCE with a moving-average baseline 70.7 / 37.9 / 55.3; PPO 67.6 / 29.2 / 32.0; DPO 66.6 / 39.0 / 61.9. The PPO row is one tuned configuration in that study; the same paper reports that removing PPO's clipping "gives a slight boost" and that the loss is clipped on average less than 5% of the time per batch in their setting (§3.2). Result (single study).

**Conditions and limits.** The RLOO runs used a 512-token context, two gradient steps per rollout batch, and short generations (TL;DR). The "clipping is rarely necessary" finding belongs to that regime; recipes that split one rollout batch into 16 mini-batches (DeepSeek-R1-Zero, [[deepseek-r1-recipe]] v2 §2.1) or four ([[gspo]] §5.1) move the policy substantially before the last mini-batch, and there the clip is active (§7, §8).

---

## §3 REINFORCE++: normalization over the whole batch

**Definition and mechanism.** [[reinforce-plus-plus]] keeps the PPO-clip surrogate and the critic-free advantage but computes the normalization statistics over the global training batch rather than over one prompt's group. Two variants (v9 §3.1–3.2):

```
REINFORCE++ (k ≥ 1):      A_{q,o_t} = r(o_1:T, q) − β Σ_{i=t..T} KL(i)          (Eq. 4)
                          A^norm   = (A − mean_batch(A)) / (std_batch(A) + ε)   (Eq. 5)
w/ Baseline (k > 1):      A′       = R − mean_group(R)                          (Eq. 6)
                          A^norm   = (A′ − mean_batch(A′)) / (std_batch(A′) + ε)(Eq. 7)
                          L        = L_PPO(A^norm) − λ · E[½ (log π_θ/π_ref)²]  (Eq. 8)
```

`KL(i) = log[π_θold(o_i | q, o_<i) / π_ref(o_i | q, o_<i)]`; `β` is the KL coefficient applied inside the reward; `λ` weights a separate k2 KL loss and is not reported.

**Argument.** The authors prove that the group-level `(r_i − mean)/std` is a biased estimator for any group size `N ≥ 2`, because the denominator depends on `r_i` (App. A.1, Thm. 1), and argue that with `k = 4` or `8` near-equal rewards drive the group standard deviation toward zero so the advantage "explodes" (§2.2). §5 below makes the size of that effect concrete: the explosion is bounded, and the exactly-equal case gives zero, not a large value.

**Evidence.** Llama-3-8B-SFT with a Bradley–Terry reward model trained on about 700K pairs, 20,000 prompts: Chat-Arena-Hard 46.7 at k = 1 against GRPO's 46.8 at k = 4 (v9 Table 1). Trained on 30 AIME-24 questions: GRPO reaches 95.0 train pass@1 with AIME-25 pass@1 0.0 and pass@16 0.4, while REINFORCE++ reaches 71.0 / 2.5 / 40.0 (v9 Table 2; the model is not stated). Tool-use average@32: w/ Baseline 24.10, GRPO 22.58, PPO 21.85 (v9 Table 4). Each comparison is a single run without reported seeds.

**Implication for a general-purpose model.** The 30-question experiment is the clearest published warning in this family that a group-normalized objective can fit a small prompt set and transfer nothing: a pass@16 of 0.4 on the held-out year (units as printed in the table) means that sixteen attempts per problem almost never produced a correct solution, while training accuracy was 95.0. Prompt-set construction is ch-16.

---

## §4 GRPO: group z-scores and where the KL term goes

**Definition.** For a question `q`, sample `G` outputs from `π_θold`, score them, and give every token of output `i` the same advantage (DeepSeekMath §4.1.2, quoted in [[grpo]]):

```
Â_{i,t} = (r_i − mean(r_1 … r_G)) / std(r_1 … r_G)
```

**Objective (Eq. 3 of the paper).**

```
J_GRPO(θ) = E[q, {o_i}]  (1/G) Σ_i (1/|o_i|) Σ_t { min[ ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t} ] − β D_KL[π_θ || π_ref] }
ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θold(o_{i,t} | q, o_{i,<t})
```

`|o_i|` is the token length of output `i`; `ε` is the clipping parameter; `β` the KL coefficient; `π_ref` the reference policy. The paper does not report `ε` or the rollout temperature ([[grpo]] Verification).

**The KL term: estimator and placement.** DeepSeekMath's Eq. 4 is `π_ref/π_θ − log(π_ref/π_θ) − 1`, cited to Schulman's note and described as "guaranteed to be positive" ([[grpo]] §4.1.1). That note ([[john-schulman-kl-tricks]], March 2020) names three Monte-Carlo estimators of `KL[q, p]` for samples from `q`, with `r = p(x)/q(x)`:

| Estimator | Formula | Bias | Variance | Sign |
|---|---|---|---|---|
| k1 | `−log r` | unbiased | high ("negative for half of the samples") | any |
| k2 | `½ (log r)²` | biased (an f-divergence agreeing with KL to second order) | low | ≥ 0 |
| k3 | `(r − 1) − log r` | unbiased (k1 plus the zero-mean control variate `r − 1`) | low | ≥ 0 |

Worked example. For one token with `π_θ = 0.5` and `π_ref = 0.4`, `r = 0.8`: k1 = 0.2231, k2 = 0.0249, k3 = 0.0231. For `π_θ = 0.4`, `π_ref = 0.5`, `r = 1.25`: k1 = −0.2231 (negative for a quantity that must be non-negative), k2 = 0.0249, k3 = 0.0269.

Placement differs across the family and changes the gradient, not only the logging:

| Algorithm | KL placement | Estimator |
|---|---|---|
| PPO / InstructGPT, [[rloo]] | added to the per-token reward before advantages | log ratio (k1 form) |
| GRPO, DeepSeek-R1-Zero | added to the loss | k3 ([[grpo]] Eq. 4; [[deepseek-r1-recipe]] v2 §2.1, coefficient 0.001) |
| [[reinforce-plus-plus]] | k1 in the reward; w/ Baseline adds a separate k2 loss | k1 / k2 |
| [[dr-grpo]], [[dapo]], Magistral, DeepSWE | no KL term (β = 0) | — |

A KL term inside the reward passes through the advantage normalization; a KL term in the loss does not. [[reinforce-plus-plus]] argues in App. B.1 that k2 is the appropriate separate-loss estimator for reverse KL and that k3 used as a loss estimates forward KL (Interpretation by those authors; verl exposes both, `kl_loss_type: kl(k1), abs, mse(k2), low_var_kl(k3)`, [[verl-grpo]]).

**Evidence.** DeepSeekMath-Instruct 7B → DeepSeekMath-RL 7B with GRPO on about 144K GSM8K/MATH-style questions: GSM8K 82.9 → 88.2, MATH 46.8 → 51.7, MGSM-zh 73.2 → 79.6, CMATH 84.6 → 88.8 (Table 5). The paper reports no direct GRPO-versus-PPO accuracy comparison at 7B ([[grpo]] Guideline). Result (single study).

---

## §5 Sign structure of group advantages, and when a group carries no signal

With a binary reward, everything about who is reinforced follows from one number: how many of the `G` samples were correct. The table below is `G = 8`, reward 1 for correct and 0 for incorrect, and shows the advantage a correct and an incorrect sample receives under three rules. "GRPO (pop)" divides by the population standard deviation `√(p(1−p))`; "GRPO (sample)" divides by `torch.std`, the sample standard deviation used by verl and TRL ([[verl-grpo]] L319–321).

| Correct of 8 | mean | pop std | GRPO(pop) ✓ / ✗ | GRPO(sample) ✓ / ✗ | Dr. GRPO ✓ / ✗ | RLOO ✓ / ✗ |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | — / 0 | — / 0 | — / 0 | — / 0 |
| 1 | 0.125 | 0.331 | +2.646 / −0.378 | +2.475 / −0.354 | +0.875 / −0.125 | +1.000 / −0.143 |
| 2 | 0.250 | 0.433 | +1.732 / −0.577 | +1.620 / −0.540 | +0.750 / −0.250 | +0.857 / −0.286 |
| 4 | 0.500 | 0.500 | +1.000 / −1.000 | +0.935 / −0.935 | +0.500 / −0.500 | +0.571 / −0.571 |
| 6 | 0.750 | 0.433 | +0.577 / −1.732 | +0.540 / −1.620 | +0.250 / −0.750 | +0.286 / −0.857 |
| 7 | 0.875 | 0.331 | +0.378 / −2.646 | +0.354 / −2.475 | +0.125 / −0.875 | +0.143 / −1.000 |
| 8 | 1.000 | 0.000 | 0 / — | 0 / — | 0 / — | 0 / — |

Four facts follow, and the interactive version of this table is [figures/group-baseline.html](figures/group-baseline.html), which recomputes every column for a chosen `G`, a chosen number of successes, and a continuous-reward mode.

1. **All-equal groups carry no signal.** When every `r_i` is equal, the numerator `r_i − mean` is exactly 0, so the advantage is `0/(0 + ε) = 0` regardless of the denominator. verl's implementation makes this explicit, and the zero-advantage case is a code-level fact, not a divergence: `scores[i] = (scores[i] − id2mean) / (id2std + 1e-6)` with `ε = 1e-6` ([[verl-grpo]] L272, L319–329). [[dapo]] §3.2 gives the same reading and calls the consequence a shrinking effective batch.
2. **The z-score is bounded.** With the population standard deviation, no group advantage can exceed `√(G−1)` in magnitude; at `G = 8` that is 2.646, reached exactly by the lone outlier rows. Large advantages come from nearly uniform groups, not from uniform ones.
3. **On easy prompts the signal is carried by the failures.** At 7 correct of 8, the correct responses receive +0.378 each and the single failure receives −2.646. Group-baseline RL on a prompt set the model mostly solves is, in gradient terms, training against its own rare failures.
4. **Dividing by the standard deviation re-weights prompts.** Summing `|A|` over a group: at 1 correct of 8, GRPO(pop) gives 2.646 + 7 × 0.378 = 5.29 and Dr. GRPO gives 0.875 + 7 × 0.125 = 1.75; at 4 correct, GRPO gives 8.0 and Dr. GRPO 4.0. The ratio between a hard group and a balanced group is 0.66 with the division and 0.44 without it. This is the "difficulty bias" named by [[dr-grpo]] §3.1.

**How often a group is dead.** If a prompt's per-sample success probability is `p` and samples are independent, the probability that a group of size `G` is all-correct or all-wrong is `p^G + (1−p)^G`:

| `p` | G = 4 | G = 8 | G = 16 |
|---|---|---|---|
| 0.1 | 0.656 | 0.431 | 0.185 |
| 0.2 | 0.411 | 0.168 | 0.028 |
| 0.5 | 0.125 | 0.008 | ~0.000 |
| 0.9 | 0.656 | 0.431 | 0.185 |

At `p = 0.9` and `G = 8`, 43% of groups contribute nothing. [[dapo]] Fig. 3b shows the share of prompts with accuracy 1 rising through training, which is the same effect appearing over time rather than across prompts. The two responses are prompt filtering (ch-16) and dynamic sampling (§7).

**Continuous rewards: the near-tie amplification.** Group rewards from a reward model `{0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.50}` have mean 0.500 and population standard deviation 0.0112. The z-scores are `{0, +0.89, −0.89, 0, +1.79, −1.79, 0, 0}`: differences of 0.02 reward points, which may be reward-model noise (ch-41), become advantages of the same magnitude as a correct-versus-incorrect distinction. Without the division the advantages stay at `±0.01`–`0.02` and the prompt contributes little. This, not the all-equal case, is the failure mode group-level standard-deviation division creates with continuous rewards.

---

## §6 Dr. GRPO: removing the length term and the standard-deviation term

[[dr-grpo]] (arXiv:2503.20783, March 2025; COLM 2025) identifies two terms in Eq. 3 that bias the gradient and removes both. All its runs set `β = 0`, so no KL term is involved (App. G Table 6).

### 6.1 The `1/|o_i|` response-length term

**Mechanism (§3.1).** Each response's token losses are averaged by that response's realized length. For a correct response (`Â > 0`) a shorter response gets a larger per-token update; for an incorrect response (`Â < 0`) a longer response is penalized less per token, because the same total is spread over more tokens.

**Worked example.** Two incorrect responses in the same group, `Â = −0.5`, `G = 8`, per-token ratio ≈ 1:

| Aggregation | 100-token response, weight per token | 1000-token response, weight per token | Total penalty ratio (1000 : 100) |
|---|---|---|---|
| GRPO `(1/G)(1/\|o_i\|)` | −0.5/(8·100) = −6.25e-4 | −0.5/(8·1000) = −6.25e-5 | 1 : 1 |
| Dr. GRPO `(1/G)(1/L)` with L = 3000 | −0.5/(8·3000) = −2.08e-5 | −2.08e-5 | 10 : 1 |
| DAPO token-level `1/Σ_i \|o_i\|` | equal for every token in the batch | equal | 10 : 1 |

Under GRPO, adding 900 wrong tokens to a wrong response leaves its total contribution unchanged. Under a length-independent normalizer, the longer wrong response accumulates ten times the penalty. Panel 3 of [figures/group-baseline.html](figures/group-baseline.html) recomputes this table for any two lengths, advantage, and constant normalizer.

**In code.** TRL selects the normalizer by `loss_type` ([[trl-grpo]], commit a08e713, `grpo_trainer.py` L2548–2562, verbatim):

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

TRL's default is `loss_type="dapo"` with `beta=0.0` and group-std advantage scaling, which is not the DeepSeekMath formulation; reproductions must set all three explicitly ([[trl-grpo]] Guideline). verl exposes the same choice as `actor_rollout_ref.actor.loss_agg_mode`, with `token-mean` as the default and `seq-mean-token-sum-norm` documented for Dr. GRPO together with `use_kl_loss: False` and `algorithm.norm_adv_by_std_in_grpo: False` ([[verl-grpo]], docs/algo/grpo.md L57–62).

### 6.2 The `/std(r)` difficulty term

Dr. GRPO's advantage is `Ã_{i,t} = R(q, o_i) − mean(R)`, the column already tabulated in §5. The mean includes the sample itself, so the estimator equals `(G−1)/G` times the leave-one-out one; App. A states the `G/(G−1)` relation to RLOO.

**Evidence.** Qwen2.5-1.5B with the R1 template on MATH questions: with Dr. GRPO, output length stops rising and the length of incorrect responses on benchmarks is lower than under GRPO (§3.2, Fig. 5; the paper reports curves, not tables). Removing either term separately raised reward and accuracy over vanilla GRPO in a 1.5B ablation (App. C, Fig. 8; Fig. 9 repeats GRPO against Dr. GRPO over 3 runs). The published Dr. GRPO model, Oat-Zero-7B from Qwen2.5-Math-7B, scores AIME24 43.3, AMC 62.7, MATH500 80.0, Minerva 30.1, OlympiadBench 41.0 at a 3k-token budget (Table 4), trained in 27 hours on 8× A100.

**Conditions and limits.** This is one paper, on Qwen2.5 and Llama-3.2 bases, math-only prompts, `β = 0`, and a 3000-token budget. It is evidence for "equal or better accuracy with shorter incorrect responses", not for a strict improvement. Two later sources disagree about the standard-deviation term specifically: [[reinforce-plus-plus]] argues for keeping a spread normalizer but computing it over the batch (§3.1, App. A), and [[gigpo-verl-agent]] §5.2 finds the choice task-dependent — `F_norm = 1` gives higher success on their harder ALFWorld subtasks and on WebShop, while the two settings are similar elsewhere (Table 1, 3 seeds). [[nathan-lambert-grpo]] is a practitioner summary of the same length-normalization argument (March 2025); it reports no independent experiment.

---

## §7 DAPO: four changes measured one at a time

[[dapo]] (arXiv:2503.14476, March 2025) trains Qwen2.5-32B base with a rule-based reward of `+1` for a correct final answer and `−1` otherwise (Eq. 7), no KL term (§2.3), and reports each change's effect on AIME 2024 avg@32:

| Configuration | AIME24 avg@32 |
|---|---|
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (DAPO) | 50 |
| DeepSeek-R1-Zero-Qwen-32B (reference) | 47 |

(Table 1. One run per row; the rows are cumulative, so no row isolates a single change against the final system.)

**Clip-Higher (§3.1).** The clip range is decoupled into `ε_low` and `ε_high` (Eq. 10), with 0.2 and 0.28 in the runs. The argument is arithmetic: with `ε = 0.2` and a positive advantage, a token at `π_θold = 0.9` may rise to 1.08 (no effective bound), while a token at `π_θold = 0.01` is bounded at 0.012. Raising only the upper bound lets low-probability tokens grow; raising the lower bound would push tokens to 0 and collapse the sampling space. The measured effect is on entropy (Fig. 2b) and accuracy (Fig. 2a).

**Dynamic Sampling (§3.2).** Prompts whose `G` samples are all correct or all wrong are filtered out and resampled until the batch is full of groups with a non-zero advantage — the constraint `0 < |{o_i : is_equivalent(a, o_i)}| < G` in Eq. 11. The verl reproduction of DAPO on Qwen2.5-32B reports AIME 2024 accuracy 52% with dynamic sampling, 50% without, and 44% without token-level loss and dynamic sampling (one run each; the 44% run used different hardware) ([[verl-grpo]], docs/algo/dapo.md L36–38).

**Token-level loss (§3.3).** `(1/G) Σ_i (1/|o_i|) Σ_t` becomes `1/(Σ_i |o_i|) Σ_i Σ_t` (Eq. 12), so every token in the batch carries the same weight; long low-quality samples are suppressed in proportion to their length. Figure 4 shows entropy and mean response length rising without it.

**Overlong handling (§3.4) — long-context content.** Two mechanisms, and they are alternatives:

- *Overlong Filtering* masks the loss of truncated samples, so a response cut off by the length limit contributes no gradient rather than a penalty.
- *Soft Overlong Punishment* (Eq. 13) adds a length-dependent term inside a buffer: `R_length = 0` for `|y| ≤ L_max − L_cache`; `((L_max − L_cache) − |y|)/L_cache` for `L_max − L_cache < |y| ≤ L_max`; `−1` beyond `L_max`. With `L_max = 20,480` and `L_cache = 4,096`, a response of 18,432 tokens receives `(16,384 − 18,432)/4,096 = −0.5`, added to the correctness reward.

The reason both exist is reward noise: "a sound reasoning process can be penalized solely due to its excessive length" (§3.4). Recorded conflict: Table 1 lists Overlong Filtering as a cumulative step, while the verl maintainers state that most experiments in the paper, including the best run, were run without it because it overlaps with Overlong Reward Shaping ([[verl-grpo]], docs/algo/dapo.md FAQ). Length control is ch-44a.

**The clip is active here.** DAPO's rollout batch is 512 prompts × 16 responses with a mini-batch of 512, which is 16 gradient updates per rollout batch (§4.1). After the first update the sampling policy and the optimized policy differ, so the ratio `ρ_{i,t}` moves away from 1 within the rollout batch. The chapter's reading is that this is why DAPO's decoupled clip bound changes accuracy while RLOO's setting showed little clipping (Interpretation; neither paper reports a clipped-token fraction against the number of updates per batch), and why the RLOO-era observation that clipping is "rarely necessary" does not transfer to this setting.

**Not from DAPO.** Training on only the highest-entropy tokens is a separate result (arXiv:2506.01939, "Beyond the 80/20 Rule"); TRL implements it as an entropy-quantile mask with paper value ρ = 0.2, attributed to that paper in the config docstring ([[trl-grpo]], grpo_config.py L276–281). It is not one of DAPO's four techniques.

---

## §8 GSPO and VAPO: the ratio and the critic, revisited

**GSPO (arXiv:2507.18071, July 2025, Qwen team).** The token-level importance ratio `w_{i,t} = π_θ(y_{i,t} | x, y_{i,<t}) / π_θold(y_{i,t} | x, y_{i,<t})` is a one-sample correction for each next-token distribution, which the authors argue cannot perform its distribution-correction role and instead injects variance that accumulates over long responses (§3). GSPO replaces it with a length-normalized sequence ratio and clips whole responses (Eqs. 5–7):

```
s_i(θ) = ( π_θ(y_i | x) / π_θold(y_i | x) )^(1/|y_i|)
       = exp( (1/|y_i|) Σ_t log [ π_θ(y_{i,t} | x, y_{i,<t}) / π_θold(y_{i,t} | x, y_{i,<t}) ] )
J_GSPO(θ) = E[ (1/G) Σ_i min( s_i(θ) Â_i, clip(s_i(θ), 1−ε, 1+ε) Â_i ) ],   Â_i = (r_i − mean) / std
```

**Worked example.** A four-token response with per-token ratios `{1.10, 0.95, 1.30, 0.90}`: the mean log ratio is 0.0503, so `s_i = 1.0515`. Under token-level clipping with `ε = 0.2` only the 1.30 token is clipped; under sequence clipping with GSPO's ranges (3e-4 and 4e-4 in the reported run) the whole response falls outside the range and is excluded from the gradient. The clipping ranges are not comparable between the two objectives, and the measured clipped-token fractions differ by two orders of magnitude: about 0.15 for GSPO against 0.0013 for GRPO (Fig. 2), while GSPO still reaches higher training reward per unit of compute (Fig. 1).

**Evidence and limits.** A cold-start model fine-tuned from Qwen3-30B-A3B-Base, with each rollout batch split into four mini-batches: GSPO trains stably and improves faster than a tuned GRPO baseline (clip 0.2 / 0.27) on training reward, AIME'24, LiveCodeBench, and CodeForces Elo (Fig. 1 — curves without a final table). For Mixture-of-Experts models, about 10% of the experts activated for the same response change after a single gradient update on the 48-layer Qwen3-30B-A3B-Base, which invalidates token-level ratios; GRPO required a Routing Replay workaround to converge, GSPO does not (§5.3, Fig. 3). Result (single study, no seeds or tables).

**VAPO (arXiv:2504.05118, April 2025).** A value-model system on the same Qwen2.5-32B base and the same benchmark as DAPO: AIME 2024 avg@32 60.4 within 5,000 steps, against DAPO 50 and DeepSeek-R1-Zero-Qwen-32B 47, with 60–61 across three repeated runs (§5.2). Its components and the cost of removing each (Table 1): value-pretraining (50 steps of value warm-up from a reward model; without it, 11), decoupled GAE with `λ_critic = 1.0` (without it, 33), length-adaptive `λ_policy = 1 − 1/(αl)` with `α = 0.05` (without it, 45), Clip-Higher 0.2/0.28 (46), token-level loss (53), a positive-example NLL loss with weight 0.1 (54), and group sampling — 512 prompts × 16 samples instead of 8,192 prompts × 1 (55). Worked example of the length-adaptive rule: `l = 100` gives `λ_policy = 0.8`, `l = 1000` gives 0.98, so long responses keep long-range credit while short ones stay low-variance.

Two conclusions for this chapter. First, repeated sampling per prompt helps even when a critic is present: the group-sampling row is worth 5 points in VAPO's own ablation. Second, "critic-free" is a property of one line of systems, not a measured law; the comparison depends on scale, task, and how much engineering the value model receives.

---

## §9 What group-baseline RL changes, and on which models

**Maj@K rises, Pass@K does not (DeepSeekMath 7B).** Evaluating the SFT and RL checkpoints at temperature 0.7 for `K` up to 64, RL improves Maj@K but not Pass@K; the authors attribute the gain to "boosting the correct response from TopK rather than the enhancement of fundamental capabilities" ([[grpo]] §5.2.2, Fig. 7; Interpretation by those authors).

**Narrow prompts, broader improvement.** DeepSeekMath's RL used GSM8K/MATH-style chain-of-thought questions only, and the authors treat the other benchmarks as out of domain: MGSM-zh 73.2 → 79.6 and CMATH 84.6 → 88.8 improved as well (Table 5). Non-math benchmarks after RL are not reported, so this is evidence of transfer within mathematics across languages, not of preserved general ability.

**Positive and negative gradients have different effects on coverage.** Training Qwen2.5-Math-7B on MATH with 8 rollouts per prompt, [[negative-sample-reinforcement]] separates the RLVR gradient into a positive part (PSR) and a negative part (NSR). MATH pass@1 / pass@256: base 63.2 / 96.9; GRPO 76.3 / 95.5; PSR-only 74.1 / 91.2; NSR-only 75.7 / 96.9; W-REINFORCE (positives down-weighted by λ = 0.1) 76.6 / 96.7. On AIME 2025, pass@256: base 46.7, GRPO 50.0, PSR 43.3, NSR 53.3, W-REINFORCE 56.7 (Table 1). Reinforcing correct responses alone buys pass@1 and costs coverage.

**Model-family dependence.** [[spurious-rewards-rlvr]] runs GRPO for 300 steps (16 rollouts per prompt, 64-prompt rollout batch, LR 5e-7) with reward functions of decreasing information content. On Qwen2.5-Math-7B, random rewards give +21.4 points on MATH-500 against +29.1 for ground-truth rewards (Abstract). On Llama3.1-8B-Instruct and OLMo2-7B the same spurious rewards give small gains or losses (§3, Fig. 1). The proposed mechanism is a clipping bias that amplifies behaviours with high prior probability; when clipping is disabled in three different ways, random rewards stop producing consistent gains (§4, Fig. 4). The paper also reports that one-shot RLVR and test-time-training signals reproduce on Qwen but often not on other families (§3, App. E Fig. 15). [[reinforcement-learning-with-one-training-example]] is the one-shot result itself: Qwen2.5-Math-1.5B rises from 36.0% to 73.6% on MATH500 and 17.6% → 35.7% on a six-benchmark average from a single training example, but the same procedure on Llama-3.2-3B-Instruct moves the six-benchmark average from 17.5 to 19.0 with one example and 21.0 with two, against 19.8 for the full 1,209-example set (Table 4).

**What the base model contributes.** [[octothinker]] runs the same GRPO setup (MATH8K prompts, batch 128, 16 rollouts per query) on Llama-3.2-3B-Base and Qwen2.5-3B-Base and reports MATH500 after RL of 10.0 and 66.4 respectively; mid-training Llama on 200B tokens of a curated math corpus and then 20B decay tokens raises the post-RL score to 65.2 (Fig. 1, values as labelled in the figure). The conclusion the authors draw is that RL scalability is set before RL starts.

**Implication.** Publish algorithm comparisons with the base model named, and repeat at least one comparison on a second family. A random-reward run is a cheap control: if it reproduces most of the gain, the experiment has measured elicitation of pretrained behaviour rather than the reward's information.

---

## §10 Group baselines for multi-turn agents (preview)

Two ways of applying this chapter's algebra when a trajectory has many steps and one terminal reward, both covered in full in ch-45b.

**Trajectory-level group, leave-one-out, no KL.** DeepSWE-Preview trains Qwen3-32B with RL only on 4.5K R2E-Gym problems and reports 42.2% Pass@1 on SWE-Bench-Verified (average of 16 runs) ([[deepswe]]). Its GRPO++ combines clip-high and no KL loss from DAPO, removal of the standard-deviation term and normalization by the maximum context length from Dr. GRPO, a leave-one-out baseline from RLOO, and two changes of the authors: no entropy loss, and compact filtering — masking the loss of trajectories that end by context limit, step limit, or a 20-minute generation timeout (§2.3). With 8 rollouts and 2 successes, a failure's advantage is `0 − 2/7 ≈ −0.29` and a success gets `1 − 1/7 ≈ 0.86`; with 8 failures every advantage is 0 (derived in [[deepswe]] from §2.3).

**Step-level groups from repeated states.** [[gigpo-verl-agent]] adds a second group level: trajectories for one task start from identical initial states, so the same environment state recurs across trajectories, and the actions taken from that "anchor state" form a step-level group. Each action is scored by its discounted return `R_t = Σ_{k≥t} γ^{k−t} r_k`, and the total advantage is `A(a_t) = A_E(τ_i) + ω · A_S(a_t)` (Eqs. 5–8), with `ω = 1` and `γ = 0.95` used without tuning. The grouping requires no extra rollouts. On ALFWorld with Qwen2.5-7B-Instruct: GiGPO 90.8 overall success against GRPO 77.6, RLOO 75.5, and PPO 80.4 (Table 1, 3 seeds).

---

## Negative samples and negative feedback

**Which kind of negative.** This chapter's negatives are **negative as gradient**: a response whose advantage is below the group mean has its token log-probabilities decreased. Group-baseline RL creates them mechanically — with a group mean as baseline, some samples must be below it.

**Where they come from and their error rate.** A verifier (answer match, unit tests) or a reward model labels each sampled response. Verifier errors are not usually reported; DeepSeekMath cites PRM800K as containing about 20% incorrect annotations when the label source is human process supervision ([[grpo]] §5.2.3, footnote 7). With a reward model, §5 showed that within-group noise of 0.02 becomes a full-magnitude negative once divided by the group standard deviation.

**Mechanism.** For the softmax over the vocabulary, `∂ log p_y / ∂z_j = 1[j = y] − p_j`. A negative advantage on the sampled token `y` decreases `z_y` by `η|A|(1 − p_y)` and increases every other logit `z_j` by `η|A| p_j`. The mass removed is redistributed in proportion to the current probabilities. Worked example with `p = (0.6, 0.3, 0.1)` for tokens `(a, b, y)` and `η|A| = 1`: pushing down `y` gives `(0.710, 0.263, 0.026)`. The already-likely token `a` absorbs almost all the removed mass and `b` loses probability even though its logit increased. Pushing down a high-probability token instead — `a` with `η|A| = 1` — gives `(0.438, 0.441, 0.120)`: the largest single-token probability change is 0.110 in the first case and 0.162 in the second, for the same update size. Two consequences: penalizing rare failures concentrates the distribution on the current mode, and penalizing likely tokens moves the policy a long way in one step. ch-43a derives the likelihood-displacement and squeezing versions of this, and [[likelihood-displacement]] gives the preference-optimization case.

**What current practice does.** Keeps them, with limits: the clip bounds the per-token ratio (the lower bound acts on negative-advantage tokens); DAPO deliberately does not raise `ε_low` because that would push probabilities to 0 and collapse the sampling space (§3.1); TRL's `delta` option caps `ρ` from above, which by construction affects only negative advantages ([[trl-grpo]], derived from L2508–2515); DeepSWE and DAPO mask truncated or timed-out trajectories so an uncertain failure carries no gradient rather than a penalty.

**Evidence on size of effect.** From the PSR/NSR decomposition on Qwen2.5-Math-7B (§9): negative-only training preserves pass@256 at the base level (96.9 on MATH) and improves pass@1 to 75.7, while positive-only training reaches a comparable pass@1 (74.1) and lowers pass@256 to 91.2. Down-weighting positives by λ = 0.1 gave the best combination in that study; λ ≤ 0.2 behaved similarly, and λ = 1 (plain REINFORCE) gave MATH pass@256 92.0 against 96.7 at λ = 0.1 (Table 1; ablation over λ ∈ {0, 0.05, 0.1, 0.2, 1} in App. E). One model family, one dataset — the §9 caution about Qwen2.5-Math applies to this result too.

**A related control from off-policy RL.** [[asymmetric-reinforce]] shows with Llama-3.1-8B-Instruct on MATH that when rollouts are stale and the baseline sits at the behaviour policy's mean reward (`δV = 0`), train and test accuracy collapsed in all 7 runs, while the 7 runs at `δV = −0.1` (a baseline slightly below the group mean, which weights positives more) are reported as more stable; the collapse comes earlier for larger `δV > 0` (§5.2, Figs. 3–5). Where the baseline sits relative to the group mean is itself a control on how much negative signal is applied.

**Diagnostics.** Log, per step: the fraction of groups with zero advantage (`frac_reward_zero_std` in TRL); mean advantage magnitude split by sign; clipped-token fraction split by advantage sign (TRL counts low-clip only where `A < 0` and high-clip only where `A > 0`); mean response length split by correctness; entropy; and pass@1 together with pass@k at a large k on a held-out set.

**Effect on generality.** Negative gradients maintain coverage (pass@k) better than positive-only training, and the group mechanism means that on easy prompts nearly all the signal is negative (§5). The risk is not the sign but the label: a false negative on a correct-but-unusual response removes exactly the behaviour that widened the distribution.

---

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeekMath-RL 7B | 7B | RL | algorithm; outputs per question G; KL coef (in loss); policy LR; max length; training batch | GRPO; 64; 0.04; 1e-6; 1024; 1024 (unit not stated) | arXiv:2402.03300v3 §4.2 ([[grpo]]) | verified 2026-09-14 | Table 5: MATH 46.8 → 51.7; no ablation of these values |
| DeepSeekMath-RL 7B | 7B | RL | clip ε; rollout temperature | not reported (checked §4.1–4.2) | — | not reported | — |
| DeepSeek-R1-Zero | 671B MoE | RL | outputs per question; questions per step; LR; KL coef (in loss); rollout split | 16; 32; 3e-6; 0.001; 8,192 outputs into 16 mini-batches, 1 inner epoch | arXiv:2501.12948v2 §2.1 ([[deepseek-r1-recipe]]) | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (stage-1 RL) | 671B MoE | RL | clip ε | 10 | v2 §3.2.1 ([[deepseek-r1-recipe]]) | verified 2026-09-14 | §3.2.1: lower ε truncates gradients, higher ε may destabilize (no table) |
| Oat-Zero-7B (Dr. GRPO, from Qwen2.5-Math-7B) | 7B | RL | responses per question; temperature; max response; KL loss and penalty; clip ε; inner epochs; LR | 8; 1.0; 3000; 0.0 and 0.0; 0.2; 1; 1e-6 constant, AdamW(0.9, 0.95) | arXiv:2503.20783v2 App. G Table 6 ([[dr-grpo]]) | verified 2026-09-14 | Fig. 8 / Fig. 9 ablate the two bias terms at 1.5B; no ablation of these values |
| Oat-Zero-7B | 7B | RL | loss normalizer; compute | constant MAX_TOKENS = 3000; 27 h on 8× A100 | Listing 1 + App. G; §1 ([[dr-grpo]]) | derived; verified 2026-09-14 | — |
| DAPO (Qwen2.5-32B base) | 32B | RL | prompt batch; responses per prompt; mini-batch; LR; ε_low/ε_high; max generation; KL | 512; 16; 512 (16 updates per rollout step); 1e-6 constant with 20-step warm-up; 0.2 / 0.28; 20,480 = 16,384 + 4,096 cache; none | arXiv:2503.14476v2 §4.1, §2.3 ([[dapo]]) | verified 2026-09-15 | Table 1 cumulative ablation (30 → 50 AIME24 avg@32) |
| DAPO | 32B | RL | overlong filtering in the reported best run | Table 1 lists it as a cumulative step; verl docs state the paper's best run did not use it | arXiv:2503.14476v2 Table 1; verl docs/algo/dapo.md FAQ | conflict | — |
| VAPO (Qwen2.5-32B base) | 32B | RL | actor LR; critic LR; λ_critic; λ_policy; ε_low/ε_high; prompts × samples; positive LM loss weight; value warm-up | 1e-6; 2e-6; 1.0; 1 − 1/(0.05·l); 0.2 / 0.28; 512 × 16; 0.1; 50 steps | arXiv:2504.05118v3 §5.1 ([[vapo]]) | verified 2026-09-15 | Table 1: removing each component costs 49, 27, 15, 14, 7, 6, 5 points |
| GSPO (cold start from Qwen3-30B-A3B-Base) | 30B MoE | RL | sequence-clip range (left/right); mini-batches per rollout batch; GRPO baseline clip | 3e-4 / 4e-4; 4; 0.2 / 0.27 | arXiv:2507.18071v2 §5.1 ([[gspo]]) | verified 2026-09-15 | Fig. 1 curves; Fig. 2 clipping fractions 0.15 vs 0.0013 |
| RLOO, Pythia-6.9B on TL;DR | 6.9B | RL | k; rollout batch; step batch; β (KL in reward); LR; steps; gradient steps per batch | 2 or 4; 512; 256; 0.03; 1e-6 constant, 3% warm-up; 600; 2 | arXiv:2402.14740v2 App. "Preference Training" ([[rloo]]) | verified 2026-09-15 | LR chosen from a sweep of {1e-6, 1e-5, 2e-5} |
| RLOO, Pythia-6.9B / Llama-7B on Anthropic-HH | 6.9B / 7B | RL | β; steps; batches | 0.10; 393 (Pythia); 2048 rollout and step batch over 2 epochs (Llama) | same locus ([[rloo]]) | verified 2026-09-15 | Table 1 win-rates |
| REINFORCE++ v1 runs (Llama3.1-8B-SFT, Qwen2.5-7B-Instruct) | 8B / 7B | RL | samples per prompt; β; clip ε; actor LR; γ | 4; 0.01 general, 0.001 math; 0.2; 5e-7; 1.0 | arXiv:2501.03262 v1 §4.2 Table 1 ([[reinforce-plus-plus]]) | verified 2026-09-14 | no ablation reported |
| DeepSWE-Preview | 32B | RL | advantage; std normalization; KL loss; ε_high; trajectories per problem; LR; loss normalizer | leave-one-out; off; off; 0.28; 8; 1e-6; max context length | blog §2.3; released script L26–L56 ([[deepswe-recipe]]) | verified 2026-09-14 | Fig. 6: compact filtering ablation at 14B (curve only) |
| GiGPO, ALFWorld (Qwen2.5-1.5B / 7B-Instruct) | 1.5B / 7B | RL | group size; groups per rollout; γ; ω; KL loss coef; actor LR; max steps | 8; 16 (128 environments); 0.95; 1 (no tuning); 0.01; 1e-6; 50 | arXiv:2505.10978v3 App. "Hyperparameters for ALFWorld" ([[gigpo-verl-agent]]) | verified 2026-09-15 | Table 1 (3 seeds) |
| Magistral Medium / Small | not reported / 24B | RL | KL; advantage; zero-advantage groups; ε_high | none; `r_i − μ` then minibatch normalization; removed; 0.26–0.28 (Small: 0.3) | arXiv:2506.10910v1 §2.1, §5.3 ([[magistral-recipe]]) | verified 2026-09-14 | no ablation reported |
| Spurious-reward runs (Qwen2.5-Math-7B and 7 other models) | 1.5B–8B | RL | rollouts per prompt; rollout batch; mini-batch; LR; temperature; steps | 16; 64 prompts; 128 rollouts; 5e-7 constant; 1.0; 300 | arXiv:2506.10947v2 App. D ([[spurious-rewards-rlvr]]) | verified 2026-09-15 | Fig. 1 (per-model results) |

**Starting point for a small general-purpose run.** For a 1.5B–7B model with a verifiable 0/1 reward and responses under about 3,000 tokens, the configuration with the most direct support is the Dr. GRPO row: `G = 8`, temperature 1.0, no KL term, `ε = 0.2`, one inner epoch, LR 1e-6 constant with AdamW(0.9, 0.95), advantage `r_i − mean`, loss normalized by a constant equal to the generation budget (arXiv:2503.20783v2 App. G Table 6; that setting produced Oat-Zero-7B on 8× A100 in 27 hours from Qwen2.5-Math-7B with MATH level 3–5 prompts). Two changes require larger-scale evidence before adopting: `ε_high = 0.28` and dynamic sampling come from a 32B run with 16 responses per prompt and a 20,480-token budget ([[dapo]] §4.1), and GSPO's clip ranges come from a 30B MoE run (§5.1). If the reward is a reward-model score rather than a verifier, do not carry over the "no normalization" choice untested: see §5 and [[reinforce-plus-plus]] §3.1.

---

## Generalization lens

**(a) What increases breadth.**
- Keeping the negative part of the gradient: NSR-only and W-REINFORCE (λ = 0.1) preserve MATH pass@256 at the base model's 96.9 and raise AIME 2025 pass@256 to 53.3 and 56.7 against the base's 46.7, while positive-only training drops to 91.2 and 43.3 ([[negative-sample-reinforcement]] Table 1; Qwen2.5-Math-7B only).
- Keeping entropy from collapsing: Clip-Higher raises entropy and AIME accuracy together in DAPO's Fig. 2; [[reinforcement-learning-with-one-training-example]] reports that an entropy term is needed for the post-saturation gains in its 1-shot setting (Abstract, §4). Diversity and entropy are ch-43.
- Prompt sets that leave a non-zero fraction of solvable-but-not-solved groups: §5 quantifies how quickly groups go dead as the pass rate approaches 0 or 1 (ch-16).
- Mid-training before RL: the same RL recipe gives MATH500 10.0 on Llama-3.2-3B-Base and 65.2 after 220B tokens of mid-training ([[octothinker]] Fig. 1).

**(b) What causes narrowing or forgetting.**
- Positive-only reinforcement, which concentrates mass on paths the model already prefers and lowers pass@k at large k ([[negative-sample-reinforcement]] Table 1).
- Group-level standard-deviation division with a noisy continuous reward, which promotes reward-model noise to full-strength gradient (§5).
- Length-normalized loss, which lets incorrect responses grow ([[dr-grpo]] §3.1, Fig. 5) — a narrowing of usable output rather than of task coverage.
- Small prompt sets: GRPO on 30 AIME-24 questions reached 95.0 train pass@1 with AIME-25 pass@16 of 0.4 ([[reinforce-plus-plus]] v9 Table 2).
- Training-reward tracking: DAPO reports that the final training reward "often exhibits little correlation with the accuracy on the validation set, which indicates overfitting to the training set" (§4.3).

**(c) How to measure it for this stage.** Report pass@1 and pass@k at large k on held-out sets, not only average accuracy; evaluate at least one benchmark outside the RL domain before and after; repeat the headline comparison on a second model family; run a random-reward control ([[spurious-rewards-rlvr]]); log the fraction of dead groups, the truncation rate, entropy, and response length split by correctness. Known measurement errors: single-run comparisons without seeds (most rows in §7 and §8), cumulative ablation tables that cannot isolate one change (DAPO Table 1), and template effects that can make a base model look weaker than it is before RL ([[dr-grpo]] §3.3).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Believing an all-equal group produces an enormous advantage | none — the group silently contributes nothing | Log the share of groups with zero advantage (`frac_reward_zero_std` in TRL); compare with `p^G + (1−p)^G` for the measured pass rate |
| Keeping group-std division with a reward-model score | advantages of ±1 to ±2 on groups whose reward range is a few hundredths | Log per-group reward range next to max advantage magnitude; switch to batch-level normalization or none |
| Assuming the clip never binds | clip fraction reported as near zero because it is averaged over both signs | Log clipped-token fraction separately for `A > 0` and `A < 0`; count gradient updates per rollout batch |
| Using TRL or verl defaults as "the GRPO paper" | reproduction differs from published numbers | TRL default is `loss_type="dapo"`, `beta=0.0`, group-std scaling; verl default is `adv_estimator=gae`, `rollout.n=1`, `token-mean` — set each explicitly |
| Scoring truncated responses as failures | truncation rate and mean length rise together; entropy unstable | Log truncation rate; apply overlong filtering or soft punishment ([[dapo]] §3.4) |
| Length-normalized loss on long chains | mean length of incorrect responses grows faster than that of correct ones | Log length split by correctness ([[dr-grpo]] Fig. 5); switch normalizer |
| Comparing G values or RLOO against Dr. GRPO without accounting for scale | one run appears to need a different learning rate | The two advantages differ by `G/(G−1)`; z-scoring changes magnitude by roughly `1/std` |
| Concluding an algorithm ranking from one model family | gains reproduce on Qwen and vanish elsewhere | Run the same comparison on a second family and add a random-reward control ([[spurious-rewards-rlvr]] §3) |
| Reading training reward as progress | training reward rises while validation accuracy is flat | Hold out a validation set and gate on it ([[dapo]] §4.3) |
| Token-level ratios on a large MoE policy | training reward collapses irreversibly after a normal-looking start | Log expert-activation overlap between `π_θold` and `π_θ`; consider sequence-level ratios ([[gspo]] §5.3) |

---

## Check your understanding

1. A group of 8 rollouts on a verifiable task returns all zeros. Explain, from the advantage formula, why this group produces no gradient under both GRPO and Dr. GRPO, and why the `+1e-6` in the denominator does not change that.
2. At 7 correct out of 8, the single failure receives an advantage of −2.65 while each success receives +0.38. Explain what this implies about which samples drive the update on a prompt set the model mostly solves, and what happens to that structure if the standard-deviation division is removed.
3. DeepSeekMath's Eq. 3 divides each response's token losses by `|o_i|`. Derive why this makes the per-token penalty on a 1000-token incorrect response ten times smaller than on a 100-token one, and explain why the total sequence loss is nevertheless identical.
4. RLOO's paper found the clip active on less than 5% of tokens, while DAPO measured a change in accuracy from decoupling the clip bounds. Explain the difference in terms of how many gradient updates each setting takes per rollout batch.
5. GSPO clips about 0.15 of tokens against GRPO's 0.0013, yet reaches higher training reward per unit of compute. Give the authors' explanation, and state what property of the token-level ratio their argument relies on.
6. A colleague reports that a new RL variant improves MATH-500 by 20 points on Qwen2.5-Math-7B. Describe two control experiments that would distinguish "the reward taught the model something" from "the objective amplified a pretrained behaviour", and say what result would support each.
7. Positive-only training raises pass@1 and lowers pass@256; negative-only training preserves pass@256. Using the softmax logit gradient, explain why the two directions act differently on the breadth of the output distribution.
8. DeepSWE removes the standard-deviation term, uses a leave-one-out baseline, and masks trajectories that hit the context or time limit. State which measured failure each of the three choices addresses, and which of them would change if the reward were a continuous score instead of 0/1.

---

## Connections

- **Previous chapter — ch-42a, Narrow Training, Broad Behaviour Change: Emergent Misalignment, Sycophancy, and Trait Transmission.** Why an objective applied to a narrow slice of behaviour changes the model everywhere; this chapter supplies the gradient mechanism for the RL case.
- **Next chapter — ch-16, RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse.** §5's dead-group arithmetic is the reason prompt difficulty filtering exists; dynamic sampling is its in-loop counterpart.
- **ch-37, Policy-Gradient Foundations for Language Models** — the estimator and baseline theory this chapter applies.
- **ch-38, KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax** — the critic, GAE, and KL-in-reward formulation that §1 compares against.
- **ch-38a, SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity** — the broader evidence on what RL changes relative to SFT.
- **ch-43, Entropy, Output Diversity, and KL Control in RL** — entropy collapse, which Clip-Higher and entropy terms target.
- **ch-43a, Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages** — the full derivation of the mechanism sketched in this chapter's negatives section.
- **ch-44a, Length in RL: Overlong Responses, Length Control, and Long-Context RL** — overlong filtering and soft punishment in their own right.
- **ch-45b, Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability** — the full treatment of §10.
- **ch-55, verl Internals** and **ch-57, TRL Internals** — the implementations quoted here, at the level of the whole trainer.

---

## Sources

- [[rloo]] — Ahmadian et al., arXiv:2402.14740 (Feb 2024): the leave-one-out estimator, the Table 1 win-rates, the clipping and hyperparameter details. Card has no Verification section; numbers here were read from v2 and are collected in `excerpts/rloo.md`.
- [[grpo]] — Shao et al., arXiv:2402.03300 (Feb 2024): GRPO Eq. 3, the KL estimator Eq. 4, the outcome-supervision advantage, the DeepSeekMath-RL results, and the Maj@K-versus-Pass@K analysis.
- [[deepseekmath]] — duplicate card that redirects to [[grpo]]; not cited separately as evidence.
- [[dr-grpo]] — Liu et al., arXiv:2503.20783 (Mar 2025): the length and difficulty biases, the Dr. GRPO objective, the `G/(G−1)` relation to RLOO, the App. G recipe.
- [[reinforce-plus-plus]] — Hu et al., arXiv:2501.03262: global advantage normalization, the bias proof for group-level normalization, the small-prompt-set overfitting result.
- [[rloo-vs-grpo]] — comparative reference with no primary artifact; used only for the axis list (baseline, normalization, clip, KL placement). Its "equivalence in the limit" claim is replaced here by the exact `G/(G−1)` identity from [[dr-grpo]] App. A, and its "the clip rarely binds" claim by the per-run update counts in §7.
- [[dapo]] — arXiv:2503.14476 (Mar 2025), chapter excerpt: clip-higher, dynamic sampling, token-level loss, overlong shaping, Table 1, training details.
- [[gspo]] — arXiv:2507.18071 (Jul 2025), chapter excerpt: sequence-level ratio, gradient comparison, clipping fractions, MoE stability.
- [[vapo]] — arXiv:2504.05118 (Apr 2025), chapter excerpt: value-model system on the same base as DAPO, component ablation, length-adaptive GAE.
- [[spurious-rewards-rlvr]] — arXiv:2506.10947, chapter excerpt: random-reward gains on Qwen2.5-Math, their absence on Llama and OLMo, the clipping-bias mechanism and its ablation.
- [[reinforcement-learning-with-one-training-example]] — arXiv:2504.20571, chapter excerpt: 1-shot RLVR numbers and their size on Llama-3.2-3B-Instruct.
- [[octothinker]] — arXiv:2506.20512, chapter excerpt: identical RL recipe on Llama and Qwen bases, and the mid-training intervention that closes most of the gap.
- [[negative-sample-reinforcement]] — arXiv:2506.01347, chapter excerpt: PSR/NSR decomposition, pass@k table, W-REINFORCE with λ = 0.1.
- [[deepswe]], [[deepswe-recipe]] — Agentica and Together AI (Jul 2025): GRPO++ components, compact filtering, the leave-one-out advantage worked for 8 rollouts.
- [[gigpo-verl-agent]] — arXiv:2505.10978, chapter excerpt: anchor-state step groups, the `F_norm` comparison, the ALFWorld and WebShop tables.
- [[john-schulman-kl-tricks]] — the k1/k2/k3 estimators and why k3 is unbiased and non-negative.
- [[trl-grpo]], [[verl-grpo]] — the two implementations quoted: loss normalizers, advantage scaling, KL estimator options, zero-advantage handling, dynamic-sampling configuration.
- [[deepseek-r1-recipe]] — R1-Zero and R1 stage-1 RL settings, including the GAE-λ comparison in App. A.3.
- [[magistral-recipe]] — a production GRPO variant with no KL term, minibatch advantage normalization, and zero-advantage group removal.
- [[asymmetric-reinforce]] — where the baseline sits relative to the behaviour policy's mean reward, and the collapse observed at `δV ≥ 0`.
- [[likelihood-displacement]] — the preference-optimization case of the redistribution mechanism used in the negatives section.
- [[nathan-lambert-grpo]] — practitioner tracking of the length-normalization corrections (Mar 2025); no independent experiment.
