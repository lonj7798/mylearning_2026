---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/john-schulman-kl-tricks.md
source_url: http://joschu.net/blog/kl-approx.html
created_at: "2026-04-23"
---

# Excerpt: Schulman — the k1 / k2 / k3 KL estimator trio

**Source library:** `wiki/raw-data/llm-training/blogs/john-schulman-kl-tricks.md`
**Author:** John Schulman
**Year:** 2020 (original post); widely cited 2023–2025 in RLHF / GRPO implementations
**URL:** http://joschu.net/blog/kl-approx.html

---

## Why this source anchors ch-43

The k1/k2/k3 estimator trio is the bedrock of modern KL control. Every RLHF/RLVR stack in 2025 defaults to **k3**, and the derivation *why* is short enough to do at the whiteboard. Section 3 of the read chapter derives all three; this excerpt pins down the original blog's framing and surfaces the one practitioner caveat (Costa Huang's "k3 exploded") that matters in implementation.

---

## The problem statement

Source lines 14–15:

> Schulman's post addresses a deceptively simple problem: you have samples from a policy q and want to estimate `KL(q ‖ p)` where p is a reference distribution (e.g., the pre-RL SFT model).

In PPO/GRPO the "samples from q" come free — they are the rollout tokens — and `q = π_new`, `p = π_ref` (or `π_old`). You want a Monte-Carlo estimator of `KL(q ‖ p) = E_q[−log(p/q)] = E_q[−log r]`, where `r = p/q`. The three canonical one-sample estimators for this expectation are k1, k2, k3.

---

## Formulas and properties, in one table

Source lines 33–38:

| Estimator | Formula | Unbiased for KL(q,p)? | Sign | Notes |
|-----------|---------|-----------------------|------|-------|
| k1 | `−log r` | Yes | any (can be < 0) | High variance near p ≠ q |
| k2 | `0.5 * (log r)^2` | No | >= 0 | Lowest bias only near p ~ q |
| k3 | `r − 1 − log r` | Yes | >= 0 always | Preferred for RLHF |

**k1 derivation.** By definition: `E_q[−log r] = E_q[−log(p/q)] = KL(q ‖ p)`. So k1 is unbiased as a single-sample estimator. A specific draw can be negative (whenever `p(a) > q(a)` for that `a`), and the estimator's variance is controlled by the tails of `log(p/q)` under q — which can be heavy.

**k2 derivation.** Taylor-expand `−log r` around `r = 1`. Writing `r = 1 + ε` with `ε = r − 1`,

```
−log(1 + ε) = −ε + ε²/2 − ε³/3 + O(ε⁴)
```

Take the expectation under q. The linear term vanishes: `E_q[ε] = E_q[p/q] − 1 = ∫ p dx − 1 = 0`. So `KL ≈ ½·E_q[ε²] + O(ε³)`. Swap `ε ≈ log r` (valid near r = 1) to get `k2 = ½·(log r)²`. This is *biased* (the higher-order terms do not vanish in general) but has *lower variance* than k1 because `(log r)²` is a bounded-below squared quantity, not a signed log.

**k3 derivation.** Let `f(r) = r − 1 − log r`. Two checks:

- Unbiasedness: `E_q[r − 1 − log r] = E_q[r] − 1 + KL(q ‖ p)`. The tricky piece is `E_q[r] = E_q[p/q] = ∫ (p/q) · q dx = ∫ p dx = 1`. So `E_q[k3] = 1 − 1 + KL = KL`.
- Non-negativity: `f(1) = 0`, `f'(r) = 1 − 1/r`, `f''(r) = 1/r² > 0` for r > 0. So f is strictly convex with minimum 0 at r = 1, giving `f(r) ≥ 0` on `r > 0`.

Both properties together make k3 the preferred default.

---

## Why k3 is non-negative, more carefully

Source lines 39–40:

> Why k3 is non-negative: `f(r) = r − 1 − log(r)` is a convex function with minimum 0 at r=1; since r > 0, f(r) >= 0.

A different way to see it: Jensen's inequality applied to `−log` (concave) gives `−log E_q[r] ≤ E_q[−log r]`, i.e. `0 ≤ KL(q ‖ p)`. k3 is the single-sample analogue — it replaces the global Jensen gap with a pointwise convex surrogate that averages back to KL. The convexity is what makes each single draw non-negative, not just the average.

---

## Variance comparison in the regime that matters

The operating regime of PPO/GRPO after a few warmup steps is `π_new ≈ π_old` (or `≈ π_ref`), i.e. `r ≈ 1` and `log r ≈ ε`. In that regime:

- `k1 ≈ −ε` — zero-mean Gaussian-ish noise with variance `Var_q(log r)`.
- `k2 = ½·(log r)² ≈ ½·ε²` — non-negative, roughly `Var_q(log r) / 2` by the moments of a squared zero-mean variable.
- `k3 = r − 1 − log r ≈ (1 + ε + ε²/2) − 1 − ε = ε²/2` — the same leading order as k2.

So near r = 1 all three have similar *magnitude* — but k1 alone can go negative per sample, and its variance depends on `Var_q(log r)` *directly*, not on its squared counterpart. Once `log r` picks up any tails (rollouts drifting from reference, early-training), k1's variance dominates.

---

## The Costa Huang caveat

Source line 44:

> Caveat from practitioners: Costa Huang noted on X that the k3 estimator "exploded for some reason" in early TRL experiments, likely due to large r in the tails. GRPO in DeepSeekMath adopts k3 successfully — the regime where policy and reference stay close is what matters.

The likely mechanism: for very large positive `log r`, `r = exp(log r)` grows exponentially while `−log r` grows linearly, so `k3 = r − 1 − log r` is dominated by the `r` term and can become huge. One tail sample can overwhelm the batch.

The fix is textbook: clamp the log-ratio before exponentiation. verl does this explicitly ([[entropy-logging-patterns]] verl excerpt):

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
```

With clamping, the r-term in k3 is bounded by `exp(20) ≈ 4.85e8` in the worst case (still awful, but finite and catchable); without clamping, an outlier `log r ≈ 40` produces `r ≈ 2.35e17` and silently destabilizes a training step. Practitioner consensus: clamp, use k3, monitor clipfrac.

---

## Gradient subtlety — loss vs reward

Source line 47:

> DeepSeek's GRPO formulation treats the KL penalty as a loss term (not a reward), so the gradient flows through `log(pi_theta/pi_ref)` correctly. Using k3 as an *objective* rather than as a reward augmentation is the modern convention.

The read chapter's §4 expands this: PPO/InstructGPT places KL into the *reward* (so the advantage estimator carries it), GRPO places KL into the *loss* (so it is an explicit additive penalty). Both are valid; they differ in how the KL term is weighted by GAE discounting and how it interacts with advantage normalization.

---

## Connections

- Read-chapter §3 derives k1/k2/k3 from first principles; this excerpt is the source.
- [[entropy-logging-patterns]] frames how each framework actually implements the three estimators.
- [[kl-control-rlhf]] places the k3 estimator inside the InstructGPT/GRPO objective.
- Schulman's blog also underlies [[costa-huang-ppo-details]] for the wider PPO-for-RLHF implementation landscape (covered in ch-38).
- DPO (ch-39) sidesteps this entire estimator question by deriving an exact KL-constrained solution.
