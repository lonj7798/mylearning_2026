---
chapter: ch-14
course: llm-inference
phase: read
excerpt_of: "Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman, Matias 2023)"
source_url: https://proceedings.mlr.press/v202/leviathan23a.html
created_at: "2026-05-21"
---

# Excerpt: Leviathan-Kalman-Matias — Lossless Sampling Speculative Decoding

**Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
**Year:** 2023
**Venue:** ICML 2023
**URLs:** https://proceedings.mlr.press/v202/leviathan23a.html ; arXiv 2211.17192
**Raw-data source:** [[raw-data/fast-inference-from-transformers-via-speculative-decoding]]

---

## The acceptance rule (Algorithm 1, page 4)

For each drafted token `x' ~ q` (drafter distribution `q`) at a given position, with target distribution `p`:

```math
\text{sample } u \sim \text{Uniform}(0, 1)
```

```math
\text{accept } x'
\quad \text{iff} \quad
u \le \min\!\left(1, \, \frac{p(x')}{q(x')}\right)
```

If accepted: commit `x'`, advance position. If rejected: stop drafting, **resample** the position from the positive residual distribution:

```math
p_{\text{resid}}(x) =
\frac{\max(0, \, p(x) - q(x))}{\sum_y \max(0, \, p(y) - q(y))}
```

Commit the resampled token; discard remaining draft.

---

## The proof of distribution preservation

The claim: `P(committed token = x) = p(x)` for all `x`, regardless of `q`.

**Total probability decomposition.** A token is committed via *either* acceptance after being drafted *or* residual resampling after rejection.

```math
P(\text{commit } x)
= q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right)
+ (1 - \alpha) \cdot p_{\text{resid}}(x)
```

where `α = E_{x' ~ q}[min(1, p(x')/q(x'))]` is the marginal acceptance rate.

**Case 1: `p(x) ≥ q(x)`**. `min(1, p/q) = 1`, so the first term equals `q(x)`. The residual probability `max(0, p(x)−q(x)) = p(x) − q(x)`. Normalizing constant `Z = Σ_y max(0, p(y) − q(y)) = 1 − α` (standard total-variation identity). So:

```math
P(\text{commit } x) = q(x) + (1 - \alpha) \cdot \frac{p(x) - q(x)}{1 - \alpha} = q(x) + p(x) - q(x) = p(x). \quad\checkmark
```

**Case 2: `p(x) < q(x)`**. `min(1, p/q) = p(x)/q(x)`, so first term equals `q(x) · p(x)/q(x) = p(x)`. Residual at `x` is `max(0, p(x) − q(x)) = 0` (since `p < q`). So:

```math
P(\text{commit } x) = p(x) + 0 = p(x). \quad\checkmark
```

Either way `P(commit x) = p(x)`. **The committed-token distribution is exactly the target's distribution `p`.** Speculative decoding is lossless.

---

## The bonus-token rule

When all K draft tokens are accepted (no rejection within the round), the target's `K+1`-th forward emission is a *free* sample from `p` at position `K+1`. Commit it as a bonus. Without this rule, you give up ~10-15% of the speedup.

```python
if all_accepted:
    bonus = sample_from(target_logits[K])     # one more committed token
    commit(bonus)
```

---

## The speedup expression (Theorem 3.8)

For independent acceptance with constant rate `α`:

```math
\text{expected accepted per round} = \frac{1 - \alpha^{K+1}}{1 - \alpha}
```

(That's `K+1` because of the bonus.) Per-round cost: 1 target forward + `K` drafter forwards. With drafter cost ratio `c = T_{\text{draft}} / T_{\text{target}}`:

```math
\text{wall-clock speedup} = \frac{1 - \alpha^{K+1}}{(1 - \alpha)(1 + Kc)}
```

For `α=0.7, K=4, c=0.05`: `(1 − 0.7⁵)/((0.3)(1.2)) = 0.832 / 0.36 ≈ 2.31×`.

The paper reports speedups of **2-3×** on T5-XXL using T5-small/base as drafters.

---

## What's specifically Google's contribution

Speculative draft-and-verify existed before (SpecDec, [[excerpts/speculative-decoding]]). Leviathan-Kalman-Matias adds:

1. **The acceptance rule for sampling**, with proof of distribution preservation.
2. **The residual resampling step**, which is the non-trivial part.
3. **The bonus-token rule** for maximum K.
4. **The closed-form speedup analysis** — separating drafter quality, K, and target-verify cost.

Together, these turn speculative decoding from a heuristic into a *lossless* acceleration with provable correctness. This is the version every modern serving framework implements.

---

## Choice of drafter (Section 4 of the paper)

Empirically:
- **T5-small** (~60M) drafting T5-XXL (~11B): α ≈ 0.6-0.7, speedup ~2-2.5×.
- **T5-base** (~220M) drafting T5-XXL: α ≈ 0.7-0.8, speedup ~2.7-3×.
- Larger drafters → higher α but higher `c` → diminishing returns.

The paper recommends drafter ≈ target / 100 in parameter count as a heuristic. This is why Llama-3-8B → Llama-3-70B (ratio ~9) tends to underperform the heuristic, while TinyLlama-1B → Llama-2-70B (ratio ~70) sits near the sweet spot.

---

## Connections

- [[excerpts/speculative-decoding]] — Xia 2022 predecessor, greedy version.
- [[excerpts/hf-assisted-generation]] — reference library implementation.
- [[excerpts/speculative-speedup-math]] — companion derivation of optimal K and concrete numbers.
- [[ch-14]] — parent chapter.
