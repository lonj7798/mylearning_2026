---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/costa-huang-ppo-details.md
source_url: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
created_at: "2026-04-23"
---

# Excerpt: Costa-Huang's 37 tricks — the LLM-RL subset

**Source library:** `wiki/raw-data/llm-training/blogs/costa-huang-ppo-details.md`
**Artifact:** 37 implementation tricks + 2024 RLHF-specific follow-up.

---

## Why this source anchors ch-38

Ch-38 §6 filters the 37-item catalog down to the 5–7 tricks that actually change outcomes in LLM-RL. The filtering is what makes the source useful — the original list is MuJoCo-first; many items are no-ops in a language-model setting (no observations to normalize, orthogonal init dominated by pretrained weights).

---

## The source's own LLM-specific filter

From the source:

> **RLHF-specific follow-up highlights:**
> - Whitening the scalar reward before injecting into the PPO reward stream.
> - Handling per-token vs per-sequence log-prob correctly when computing the ratio.
> - KL-to-reference is added to per-token reward, not to loss, in the canonical impl (equivalent but commonly mis-coded).
> - Value head initialization: start from the RM's value head to avoid warmup regression.

Ch-38 §6 uses these four as the core of its 7-item list, augmented with advantage whitening, value-loss clipping, ratio-clip bounds, and global gradient clipping.

---

## The tricks ch-38 promotes as load-bearing

From the source's item list, ch-38 §6 carries:

> 1. **Orthogonal initialization + layer scaling** — policy and value heads initialized with orthogonal matrix and specific gain.
> 4. **Value function loss clipping** — clip value predictions around the old value (mirrors policy ratio clipping).
> 5. **Advantage normalization** — per-minibatch whitening of advantages.
> 6. **Generalized Advantage Estimation (GAE)** with lambda ~0.95.
> 7. **Global gradient clipping** at max-norm 0.5 (RL) or 1.0 (RLHF typical).
> 10. **PPO clip epsilon** 0.2 is the most common default.

Ch-38 §6 frames these as: advantage whitening, value-loss clipping, ratio clip bounds, global grad clip. Plus from the follow-up: KL-in-reward, value-head-from-RM, length normalization.

---

## Tricks ch-38 explicitly demotes for LLM-RL

Not in the critical-path 7:

- **Observation normalization** — "In LLM RLHF the analog is reward whitening." (Directly quoted reasoning in the source.) Observations don't exist as such in LLM-RL.
- **Orthogonal init** — pretraining init dominates; orthogonal init of the policy head is already a byproduct of the LM head init.
- **LR annealing** — RLHF LRs are tiny (1e-6 to 1e-5); cosine decay is typical but the delta to no-anneal is much smaller than in 3e-4 MuJoCo PPO.
- **Minibatch shuffle** — standard and uncontested; not where bugs live.

---

## Why this filter matters

The 37-item list has a reputation as a "PPO bible," but applied naively to LLM-RL it wastes effort on knobs that don't move numbers. Ch-38's curation is the answer to "which of these should I debug first when my RLHF run is misbehaving?"

---

## What ch-38 keeps, changes, drops

Keeps: advantage whitening, value-loss clipping, ratio clip bounds, global grad clip, KL-in-reward, value-head-from-RM, length normalization. Changes: elevates length normalization (not explicit in the source; comes from [[verl-ppo-loss]]'s `loss_agg_mode`). Drops: 30 of the 37 items as MuJoCo-specific or dominated by pretraining init.
