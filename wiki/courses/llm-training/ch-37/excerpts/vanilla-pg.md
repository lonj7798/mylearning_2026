---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/vanilla-pg.md
source_url: https://link.springer.com/article/10.1007/BF00992696
created_at: "2026-04-23"
---

# Excerpt: Williams 1992 — the policy-gradient theorem and the baseline identity

**Source library:** `wiki/raw-data/llm-training/papers/vanilla-pg.md`
**Artifact:** *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* — Ronald J. Williams, 1992. The paper that coined REINFORCE and wrote down the score-function estimator in its modern connectionist form.

---

## Why this source anchors ch-37

Every RL algorithm in chapters 38–46 is a specialisation of one equation: `∇J(θ) = E[∇log π(a|s) · A(s,a)]`. Williams 1992 is the first place that equation appears applied to neural-network policies, and it is the first place where the **baseline-invariance identity** — subtracting any `b(s)` does not change the gradient's expectation — is proved. Ch-37 §1 is the derivation, ch-37 §2 is the baseline proof; both live here.

---

## The attested derivation (score function)

From the source (lines 18–19, §Key Contributions):

> Policy gradient theorem for stochastic policies — derives `∇_θ J(θ) = E_{a~π_θ}[ ∇_θ log π_θ(a) · R ]`.
> Eligibility trace / score function: `∇_θ log π_θ(a|s)` depends only on the chosen action's log-prob.

This is the **log-prob trick** (score-function identity). The argument pushes `∇_θ` through an expectation via `∇p = p · ∇log p`. One line in the paper, three decades of algorithms downstream.

> Notice: the identity is agnostic to `R`. Williams wrote "reinforcement" because the 1992 frame is scalar-reward reinforcement learning, but the same identity holds for any reward-like scalar — including a learned RM score (RLHF), a deterministic verifier `v(x, y)` (RLVR), or a closed-form Bradley-Terry advantage (DPO's implicit reward). The algorithm doesn't know or care where `R` came from.

---

## The causal / return-to-go form

From the source (lines 34–36, §Technical Details):

> Per-step variant (causal form):
> `∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]`
> where `G_t = Σ_{t'≥t} γ^{t'−t} r_{t'}` is the return-to-go.

The causal form uses the fact that past rewards are independent of future actions in expectation, so terms with `t' < t` zero out. Practically: it halves the per-token variance for any algorithm that does per-token gradient accumulation on trajectories.

> Notice: for LLMs with a terminal-only reward (one RM score at EOS), `G_t = R(x, y)` for every `t`, because there are no per-step rewards to sum past. The causal form collapses to the per-sequence form `R(x, y) · ∇log π(y | x)`. This is why [[rloo]] can ignore GAE entirely — causality buys nothing when rewards concentrate at the end.

---

## The baseline identity (Theorem 1)

From the source (lines 20, 38–41):

> Proves that adding a state-dependent baseline b(s) is unbiased and reduces variance.
> For any function b(s) independent of the action:
> `∇_θ J = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · ( G_t − b(s_t) ) ]`
> Still unbiased (because `E_a[∇ log π · b(s)] = 0`), lower variance.

The unbiasedness comes from the score identity itself:

```
E_a[∇log π(a|s) · b(s)] = b(s) · E_a[∇log π(a|s)] = b(s) · ∇_θ Σ_a π(a|s) = b(s) · ∇_θ 1 = 0
```

Ch-37 §2 reproduces this derivation line-by-line. The variance claim is separate — the estimator `∇log π · (G − b)` has strictly lower variance iff `Cov(∇log π · G, ∇log π · b) > (1/2)·Var(∇log π · b)`, which is nearly always satisfied when `b(s) ≈ E[G|s]`. This is why `V^π(s)` is the "natural" baseline.

> Notice: the paper is careful that `b` is a function of `s`, not of `a`. If the baseline depends on the action, unbiasedness breaks — you're subtracting off part of the signal you meant to estimate. This is a recurring debugging failure: when engineers implement "advantage normalisation" at the token level, they often accidentally normalise using statistics that leak action information (e.g. per-token z-score using only the sampled action's reward). The safe invariant is: the baseline must be computable *before you sample a*.

---

## The menagerie of modern baselines

From the source (lines 46–53, §Common baseline choices):

| Method | Baseline |
|--------|----------|
| Original REINFORCE | Running average of returns |
| Actor-Critic | Learned V_φ(s) |
| RLOO (LLM RL) | Mean of other k−1 rollouts for the same prompt |
| GRPO (LLM RL) | Mean over G group samples, std-normalized |
| PPO | V_φ(s) with GAE advantage |

Ch-37 §2's "menagerie of baselines" table is a direct expansion of this, with REINFORCE++ (global-batch normalisation) added from [[reinforce-plus-plus]]. Every modern LLM-RL paper is a choice in this column.

> Notice: the paper does *not* say any baseline is uniformly better. The right baseline is an empirical question determined by: (a) how many rollouts per prompt you can afford, (b) whether you're willing to pay memory for a value network, (c) how correlated your rewards are across prompts in a batch. Williams 1992 sets up the search; the 2024–2025 papers run the comparison.

---

## The LLM-specific form

From the source (lines 55–58, §LLM-specific form):

> For a sequence y = y_1,…,y_T given prompt x and terminal reward R(x,y):
> `∇_θ J = E[ R(x,y) · Σ_t ∇_θ log π_θ(y_t | x, y_<t) ]`
> One advantage per sequence; KL penalty typically added as a token-level shaped reward.

Ch-37 §2's punch-line — "weighted SFT step" — is this equation. `Σ_t ∇_θ log π_θ(y_t | x, y_<t) = ∇_θ log π_θ(y | x)` is the SFT cross-entropy gradient; multiplying by `R(x, y)` is the per-sequence weight. No RL-specific code path is needed — run your SFT loop with a weight column.

---

## Variance pathology that motivates every successor

From the source (lines 60–62, §Variance issues):

> Raw REINFORCE variance scales with |y| (sum of T log-prob gradients).
> Mitigations: baseline, importance clipping (PPO), multiple samples per prompt (RLOO/GRPO), advantage normalization.

This is the clean statement of why the field did not stop at REINFORCE in 1992. For `T = 4096` tokens, the raw estimator's variance is so large that batches of reasonable size do not move the model. Every 2015–2025 algorithm is a variance-reduction scheme layered on the Williams 1992 identity. Ch-37's companion figure (`figures/pg-variance.html`) shows this concretely: pick `baseline = none`, set k=1, `σ_R = 1`, and watch var(ĝ) stay flat and high; flip baseline to `leave-one-out` at k=4 and watch it collapse.

---

## What ch-37 keeps from this source

- The full derivation of `∇J = E[∇log π · G]` via the log-prob trick (§1).
- The causal return-to-go form (§1, end).
- The baseline-invariance proof `E_a[∇log π · b(s)] = 0` (§2).
- The baseline menagerie table (§2).
- The per-sequence LLM form (§2, end).
- The variance-scales-with-|y| statement, which motivates §3 (bias-variance) and §4 (why LM-RL is special).

---

## Connections

- **ch-37 §1 / §2** — this chapter's derivations are drawn from here.
- [[trpo]] — first successor; adds a KL trust region to the surrogate objective.
- [[ppo]] — clipped-ratio first-order successor to TRPO.
- [[rloo]] — the leave-one-out baseline row of the menagerie, argued to be optimal for LLM-RL.
- [[reinforce-plus-plus]] — the global-batch-normalisation row, for small-k regimes.
- [[grpo]] / [[dr-grpo]] — group-mean / group-z-score baselines; ch-40.
- [[excerpts/rloo]] — companion excerpt that operationalises the baseline for LLM RLHF.
