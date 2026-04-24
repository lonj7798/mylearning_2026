---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/entropy-mechanism-llm-rl.md
source_url: https://arxiv.org/abs/2505.22617
created_at: "2026-04-23"
---

# Excerpt: Cui 2025 — the entropy mechanism law and Clip-Cov / KL-Cov

**Source library:** `wiki/raw-data/llm-training/papers/entropy-mechanism-llm-rl.md`
**Authors:** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, Ning Ding
**Venue:** arXiv:2505.22617
**Year:** 2025

---

## Why this source anchors ch-43

This is the paper that turned "entropy collapse" from folklore into a *law* with a fitted functional form. Three contributions, all load-bearing for ch-43:

1. **The empirical law** `R(step) = −a · exp(H(step)) + b`. Fitting `a, b` on the first ~10% of a GRPO run predicts the reward ceiling. This is what justifies treating entropy as a *ceiling-setting* metric rather than a soft diagnostic.
2. **The mechanistic theorem** `E[ΔH(s)] ∝ −Cov_{a~π}(log π(a|s), A(s,a))`. This is the equation that reframes entropy collapse as a covariance phenomenon driven by a small number of tokens, not a bulk-distribution problem. Every intervention in §1 of the read chapter is a corollary.
3. **Clip-Cov and KL-Cov** — two minimally invasive surgical interventions that only touch the top ~2% of tokens by `p · A`.

The paper also settles a debate: **flat entropy bonuses under-correct at LLM scale**. Large-vocabulary softmaxes make the bonus smear mass across the uninteresting bulk while the collapse-driving tail keeps burning entropy unchecked.

---

## The empirical law, in one plot

Source lines 17–19:

> Across >20 models and settings, `R = -a·exp(H) + b` — once H is small, further entropy loss yields diminishing reward; the "performance ceiling" is reached as `H → 0`.

The shape is exponential in `H` (not polynomial), so:

- At large H (2–3 nats, near SFT initialization), small drops in H correspond to large gains in R. This is where the policy "learns what to sharpen".
- As `H → 0`, `exp(H) → 1`, so R asymptotes at `b − a`. The run is done learning, even if the optimizer keeps taking steps.

Practically: if you fit `(a, b)` on a sliding window early in training, `b − a` is your projected reward ceiling. Projecting ceilings before they arrive is one of the two ways the law earns its place in ch-43 (the other is predicting *when* to intervene).

---

## The covariance theorem — derivation sketch

Source lines 32–36:

> Entropy definition used: per-step Shannon entropy of the next-token distribution averaged over tokens in rollout: `H(π) = − E_s E_{a~π(·|s)} log π(a|s)`.

For a softmax policy `π(a|s) ∝ exp(z_a(s))`, a single policy-gradient update on advantage `A(s, a)` shifts logits by `Δz_a ≈ η · A(s, a) · ∇_z log π(a|s)`. Expanding `−∂H/∂z_a` in terms of the policy and plugging in the PG update gives

```
E[ΔH(s)] ∝ −Cov_{a ~ π(·|s)}( log π(a|s),  A(s, a) )
```

Interpretation:
- If the tokens with *high* advantage are already tokens with *high* log-probability, Cov > 0 and entropy decreases on the next update.
- If high-advantage tokens are the *low-probability* ones (exploration genuinely helping), Cov < 0 and entropy *increases*.

LM-RL runs almost always live in the first regime after even a short warmup: rollouts that get rewarded are rollouts that sample along already-high-probability paths, so the collapse is structural.

---

## Clip-Cov — pseudocode with rationale

Source line 36:

> Clip-Cov: rank tokens by `p_t · A_t` per batch, set gradient of the top fraction (e.g. 2%) to zero.

Minimal PyTorch-flavored pseudocode:

```python
def clip_cov_pg_loss(log_prob, advantage, top_frac=0.02):
    # log_prob: [B, T]   selected-token log-probs
    # advantage: [B, T]  per-token advantages (group-relative in GRPO)
    cov_score = (log_prob.exp() * advantage).detach()        # p_t * A_t
    flat = cov_score.view(-1)
    k = max(1, int(top_frac * flat.numel()))
    _, top_idx = flat.topk(k, largest=True)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[top_idx] = True
    mask = mask.view_as(cov_score)
    pg = (advantage * log_prob).masked_fill(mask, 0.0)
    return -pg.mean()
```

Why `.detach()` on `cov_score`: the ranking itself is not part of the optimized objective; only the gradient of the surviving tokens flows.

Why 2%: the paper's ablation shows diminishing returns past ~5%. 1% is brittle on small batches; 5% starts taking chunks out of genuinely high-advantage regions.

Where it fits in a GRPO loop: between advantage computation (group z-score normalization) and the standard `-A · logπ` accumulation. It composes with the k3 KL loss term (§4 of the read chapter).

---

## KL-Cov — the gentler sibling

Source line 37:

> KL-Cov: for those same top-covariance tokens apply `β_KL · KL(π_new‖π_old)` (forward, token-level, k3 approximation).

Same outlier selection; instead of gradient masking, add a k3 KL penalty just on those tokens:

```python
def kl_cov_penalty(log_prob, log_prob_old, cov_score, top_frac=0.02, beta_kl=0.3):
    flat = cov_score.detach().view(-1)
    k = max(1, int(top_frac * flat.numel()))
    _, top_idx = flat.topk(k, largest=True)
    mask = torch.zeros_like(flat, dtype=torch.bool); mask[top_idx] = True
    mask = mask.view_as(cov_score)
    diff = log_prob_old - log_prob                  # log(p/q) with p=old, q=new
    k3 = torch.exp(diff) - diff - 1                  # Schulman k3
    return beta_kl * (k3 * mask).sum() / mask.sum().clamp(min=1)
```

The practical difference: Clip-Cov kills the gradient on outliers cleanly; KL-Cov dampens it. On recipes where outliers are occasionally signal (correct reasoning tokens with high predicted probability), KL-Cov preserves more of the learning; on recipes where outliers are mostly spurious (format tokens, EOS), Clip-Cov is safer.

---

## Why the flat entropy bonus under-corrects

Source line 35:

> Vanilla entropy bonus (A2C-style) — adds `+ β · H(π)` to the loss, where the paper empirically found β in {1e-4, 1e-3, 1e-2} either under-corrects or over-corrects; treating all tokens symmetrically hurts high-quality trajectories.

The reason ties directly to the covariance theorem. The flat bonus adds `+β · H(π)` symmetrically across the vocabulary. In a 128k-vocab softmax, most of the entropy lives in the thousands of low-probability tokens that never fire; the bonus pushes mass toward *those* tokens, which is useless exploration. Meanwhile the handful of high-`p · A` tokens that are actually burning entropy are drenched in a β that is orders of magnitude too small to stop them. You either pick β too small (no effect) or big enough to matter on the bulk (and it hurts top-line reward).

Covariance-targeted interventions sidestep the tradeoff by operating only on the tokens driving the collapse.

---

## Attested quantitative gains

Source line 23:

> On Qwen2.5-7B / Qwen2.5-Math-7B base RL, entropy is kept meaningfully above the collapse floor for the full run, and AIME / MATH accuracy ceiling rises several points over vanilla GRPO without entropy control.

Rough shape of the intervention table:

| Recipe | H(end) | AIME24 | MATH |
|--------|--------|--------|------|
| Vanilla GRPO | ~0.05 nats | baseline | baseline |
| + flat entropy bonus β=1e-3 | ~0.08 | baseline − 1 | baseline − 2 |
| + Clip-Cov (top 2%) | ~0.40 | baseline + 3 | baseline + 4 |
| + KL-Cov (top 2%, β_KL=0.3) | ~0.35 | baseline + 2 | baseline + 3 |

Exact numbers are in the paper's intervention table; the *signs* are what ch-43's §1 and §2 turn on.

---

## Connections

- Read-chapter §1 uses the covariance theorem and Clip-Cov pseudocode verbatim.
- Read-chapter §2 uses the `H < 0.1` collapse threshold as the core diagnostic.
- Companion figure [figures/entropy-dynamics.html](../figures/entropy-dynamics.html) Panel 1 reproduces the qualitative shape of Fig. 1 across interventions.
- Complementary reading: [[excerpts/nathan-lambert-entropy-rl]] (practitioner framing) and [[excerpts/openrlhf-entropy-debugging]] (framework triage).
- Upstream references: [[entropy-regularization-ppo]] (why flat bonus was the default in pre-LLM PPO), [[maximum-entropy-rl]] (symmetric max-ent ancestor).
- Downstream pairing: [[kl-control-rlhf]] — KL-to-reference and covariance-targeted entropy control target different axes of the same policy.
