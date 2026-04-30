---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reinforce-plus-plus.md
source_url: https://arxiv.org/abs/2501.03262
created_at: "2026-04-23"
---

# Excerpt: REINFORCE++ — global normalization for k=1 regimes

**Source library:** `wiki/raw-data/llm-training/papers/reinforce-plus-plus.md`
**Artifact:** Jian Hu 2025, "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models." OpenRLHF-native; targets the regime where you can only afford k=1 rollout per prompt.

---

## Why this source anchors ch-40 §3

REINFORCE++ fills the gap that RLOO and GRPO leave: what if k=1 per prompt (because you have a huge, diverse prompt pool and can't afford multiple rollouts each)? Neither a leave-one-out baseline nor a group-relative advantage is defined for a singleton. REINFORCE++'s answer: normalize advantages across the *entire mini-batch*. Ch-40 §3 is a walkthrough of this idea plus its three retained PPO-borrowings (clip, token-level KL, batch-wide norm).

---

## The three design choices ch-40 §3 attributes to this paper

From source lines 17–22 and 48–55:

1. **Global advantage normalization** — mean/std computed over the full mini-batch of sequences (not per prompt, not per group). Works at k=1 because a single-prompt std is undefined.
2. **PPO-clip retained** — REINFORCE++ alone in the family pairs k=1 with the clip. Rationale: with k=1 the per-token ratio can drift far from 1.0 under a single large advantage; ε=0.2 bounds the worst-case step.
3. **KL as per-token shaped reward** — identical to RLOO, not identical to GRPO. `r̃_t = r(x,y)·𝟙{t=T} − β·KL_t` with KL_t a k1 estimator.

---

## The algorithm ch-40 §3 reproduces

Source lines 31–42:

```
r̃_t = r(x,y) · 𝟙{t = T} − β · KL_t
G_t = Σ_{t'≥t} γ^{t'-t} r̃_{t'}                    # γ = 1 typical
Â_t = (G_t − mean_B(G)) / std_B(G)                 # global batch norm
L = −E_t[ min(ρ_t Â_t, clip(ρ_t, 1±ε) Â_t) ]
```

Ch-40 §3 reproduces this sequence because it cleanly names the four design choices: token-level KL shaping, cumulative return, global normalization, PPO-clip surrogate.

---

## Why global is better than per-group at small k (source lines 7, 17)

> "Prompt-local advantage normalization (GRPO's per-group, RLOO's leave-one-out) is high-variance when groups are small."

For small groups, the per-prompt mean and std are noisy estimates of the true prompt-conditional reward distribution. With B=2048 sequences in a global batch, the mean and std are near-exact estimates of the batch-wide reward distribution. The tradeoff: you lose the "this response is good *for this prompt*" signal and replace it with "this response has above-average reward *across all prompts in the batch*." For LLM RLHF where prompts are heterogeneous but batches are huge, the latter is often a better estimator.

---

## Attested hyperparameters

Source lines 57–65:

| Knob | Value |
|------|-------|
| Clip ε | 0.2 |
| KL coef β | 0.01–0.05 |
| Learning rate | 5e-7 – 1e-6 |
| Global batch size | 512–2048 sequences |
| k (samples per prompt) | 1–4 |
| Epochs per rollout | 1 |
| Sampling T | 1.0 |

Ch-40 §3 calls out the k=1 column specifically: this is the only variant in the family where k=1 is a supported default. RLOO requires k ≥ 2; GRPO requires G ≥ 2.

---

## The table that ch-40 §3 quotes (source lines 48–55)

| Component | PPO | RLOO | GRPO | REINFORCE++ |
|-----------|-----|------|------|-------------|
| Value network | yes | no | no | **no** |
| Clip ε | yes | no | yes | **yes** |
| KL location | per-token reward | per-token reward | in-loss (k3) | **per-token reward** |
| Advantage baseline | learned V | leave-one-out | group mean/std | **global batch mean/std** |
| Group size requirement | — | k ≥ 2 | G ≥ 2 | **k = 1 OK** |

Ch-40 §6 folds REINFORCE++ into its 4-row comparison table. The key cells to notice: it pairs k=1 with clip, unlike RLOO which has neither.

---

## Connections to the rest of the track

- [[rloo]], [[grpo]], [[dr-grpo]] — the group-baseline relatives that require k ≥ 2.
- [[ppo]] — the source of the retained clip.
- [[openrlhf-ppo]] — the reference implementation home.
- [[entropy-mechanism-llm-rl]] — relevant for understanding why small-k / large-B works empirically.
