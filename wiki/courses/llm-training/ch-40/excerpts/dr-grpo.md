---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dr-grpo.md
source_url: https://arxiv.org/abs/2503.20783
created_at: "2026-04-23"
---

# Excerpt: Dr.GRPO — two deletions that fix the length blow-up

**Source library:** `wiki/raw-data/llm-training/papers/dr-grpo.md`
**Artifact:** Liu et al. 2025, "Understanding R1-Zero-Like Training: A Critical Perspective." Figure 1 (response length over training) is the empirical proof; Section 3 gives the cleanest bias derivation in the literature.

---

## Why this source anchors ch-40 §5

Dr.GRPO is the whole reason ch-40 §5 exists. Once you have GRPO, the natural question is "are the divisions I reach for — length-normalize per response, std-normalize per group — actually unbiased?" Liu et al. answer "no" with a concrete mechanism and an even more concrete fix: delete both divisions. Ch-40 §5 is a walkthrough of that argument with numerical worked examples so a learner can *see* why the biases matter.

---

## The length bias in one worked example

Source lines 42–45. Ch-40 §5 reproduces and expands this:

Two incorrect responses with the same advantage `Â = −1.0`, same per-token ratio `ρ_t ≈ 1`:

- **Short wrong** (|o_1| = 50): per-token loss ≈ +1. Aggregated `(1/50) · Σ_t ≈ +1`.
- **Long wrong** (|o_2| = 500): per-token loss ≈ +1. Aggregated `(1/500) · Σ_t ≈ +1`.

Per-*sequence* aggregated loss is identical. But per-*token* gradient contribution is 10× smaller on `o_2`. So each logprob in the long wrong rollout is updated 10× less. Adding 200 more wrong tokens to a response dilutes the penalty per token without changing the sequence-level loss — the optimizer is blind to repetition within a wrong rollout.

**Fix**: replace `(1/|o_i|)` with a fixed `(1/L_max)`. Every token contributes equally regardless of realized length; wrong-and-long is penalized proportionally to its length.

---

## The std bias worked example

Source lines 18–20. Ch-40 §5 adds a numerical scenario:

Prompt A: `r = {1, 1, 1, 0, 0, 0, 0, 0}` — std ≈ 0.52, mean = 0.375.
Prompt B: `r = {1, 0, 0, 0, 0, 0, 0, 0}` — std ≈ 0.35, mean = 0.125.

Both prompts are informative. After std-normalization, B's single "right" rollout has `Â = (1 − 0.125) / 0.35 ≈ 2.5`; A's "right" rollouts have `Â = (1 − 0.375) / 0.52 ≈ 1.2`. B's right rollout gets twice the gradient magnitude of A's right rollout — the optimizer *over-weights* the harder prompt just because its std is smaller.

Worse edge case: if all G rollouts get r=0 (prompt too hard) or all get r=1 (too easy), std → 0. Implementations add ε=1e-6, making the advantage enormous and unstable. Dr.GRPO deletes `/std` and advantages stay bounded by the reward range.

---

## The Dr.GRPO loss (ch-40 §5 reproduces this)

Source lines 36–42:

```
Ã_{i,t} = r_i − mean({r_1, …, r_G})          # no /std

J_Dr.GRPO = E[ (1/G) Σ_i (1/L_max) Σ_t
                 min(ρ_{i,t} Ã_{i,t}, clip(ρ_{i,t}, 1±ε) Ã_{i,t})
                 − β D_KL(π_θ || π_ref) ]
```

Every other element — ρ, clip, k3 KL — is identical to GRPO. Two changes: `/std` deleted, `1/|o_i|` replaced by `1/L_max`. Minimal diff.

---

## Why the fix is unbiased (ch-40 §5 closing argument)

- Removing `/std`: advantage is now `r_i − mean(r)`, a linear function of the reward. Any policy gradient computed from it is an unbiased estimator of the expected-return gradient (standard REINFORCE-with-baseline). Division by a data-dependent std introduced bias because std is itself a function of the `r_i` being weighted; the unbiased-baseline property required the baseline to be independent of `y_i`.
- Replacing `(1/|o_i|)` with `(1/L_max)`: the aggregated per-sequence loss now scales *linearly* with the number of tokens in the response. Long wrong responses contribute more total loss, not the same total loss. Gradient is unbiased with respect to token-level credit assignment.

Both changes return the estimator to the textbook "REINFORCE with group-mean baseline" form — which is exactly RLOO's limit (large k).

---

## Empirical validation ch-40 §5 cites

Source lines 21, 24–26, Figure 1 and Table 2:

- Qwen2.5-Math-7B on MATH / AIME / AMC.
- Dr.GRPO matches or exceeds vanilla GRPO accuracy.
- **Response lengths are ~30% shorter.**
- Figure 1 length curves: GRPO shoots up monotonically; Dr.GRPO stays flat. `|o_wrong|` specifically stays flat — the pathology is eliminated.

This empirical result is what turned Dr.GRPO from "theoretical nitpick" into "default for reasoning RL" within three months of publication.

---

## Framework adoption (ch-40 §7 references this excerpt)

- `verl`: `norm_adv_by_std_in_grpo=False` toggle enables Dr.GRPO mode (one boolean flip).
- `trl`: `loss_type="dr_grpo"` switches the aggregator to `sum / (B · max_completion_length)`.
- Both are drop-in replacements; no new data structures needed.

---

## Connections to the rest of the track

- [[grpo]] — the parent algorithm with both biased divisions.
- [[rloo]] — the philosophical cousin; same unbiased group baseline, derived differently.
- [[trl-grpo]], [[verl-grpo]] — where the toggle lives in code.
- [[nathan-lambert-grpo]] — practitioner context; Lambert tracked these fixes in real time.
