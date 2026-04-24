---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/nemotron-ultra.md
source_url: https://arxiv.org/abs/2512.20856
created_at: "2026-04-23"
---

# Excerpt: Nemotron 3 (Nano/Super/Ultra) — multi-environment RL succession

**Source library:** `wiki/raw-data/llm-training/model-reports/nemotron-ultra.md`
**Release:** NVIDIA, *Nemotron 3*, 2025.

---

## Why this source anchors ch-35

Nemotron 3 is the 2025 successor to Nemotron-4 340B. Only Nano (3.2B active / 31.6B total MoE) has shipped at the time of the raw-data entry; Super and Ultra tech reports are expected later. The source header:

> **Core Insight:** "Multi-environment reinforcement learning" — one RL run across reasoning, tool-use, and agentic environments with a GenRM reward model — beats single-environment RL stages for agentic generalization.
>
> **Guideline:** Ship your reward model alongside the policy; Nvidia's open GenRM release lets downstream users resume RLHF without retraining the RM.

Ch-35 §2 uses this as the successor framing: the Nemotron-4 synthetic-SFT apparatus is now the *substrate* rather than the headline. The 2025 delta is how RL is organized on top of it.

---

## The two deltas vs Nemotron-4

From the "Innovations vs predecessors" section:

> Changes from **Nemotron-4 340B -> Nemotron 3**:
> - Shifted to MoE for Nano (Nemotron-4 was dense).
> - Multi-environment RL supersedes the sequential RLHF stages used in Nemotron-4.
> - GenRM released publicly — Nemotron-4's RM was not open.
> - Granular reasoning-budget control introduced; absent in Nemotron-4.
> - Smaller active-parameter footprint (3.2B vs 340B dense) reflects the 2025 MoE efficiency shift.

The two deltas that matter for ch-35:

**Multi-environment RL.** Nemotron-4 ran *sequential* RL stages: reasoning-RL, tool-use-RL, alignment-RL. Nemotron 3 collapses these into a *single* RL run covering all environments with the same GenRM scoring across them. The claim is better generalization to agentic tasks. This is directly relevant to ch-46 / ch-53 (agentic evals) but also to ch-35's "when do you need RL" question — multi-environment RL is presented as the answer for agentic tasks, where distillation-SFT has its lowest ceiling.

**GenRM as open artifact.** Nemotron-4's 340B-Reward was open-weight but the recipe for training it was not fully reproducible from the paper. Nemotron 3's GenRM ships as an open artifact with the policy; downstream users can resume RLHF on their own policy with this RM as-is. This is an explicit disintermediation of the RM-training step.

## What the source does NOT disclose

The "Gaps" section is explicit:

> Nano white-paper summary is thin on hyperparameters. Not disclosed: exact RL algorithm, KL β, LR, batch size, clip ε, group size G, rollouts per prompt, RL step counts, GenRM loss form, preference-data sizes, multi-environment reward mixing weights.

For ch-35 this is a reminder that **"open artifact" is not the same as "open recipe"**. You can download the Nemotron 3 policy + GenRM and resume training, but you cannot reproduce the original run from the paper alone. Contrast this with Tülu 3 (ch-33), which discloses LR / KL / batch and ships with the full training configs.

## SFT inheritance

From the "Post-training pipeline" section:

> **SFT data:** NVIDIA-curated mix covering reasoning, agentic, multi-step tool use; size not publicly itemized in the Nano release summary. Uses NVIDIA's prior Nemotron-4 data pipeline as a base.

The Nemotron-4 synthetic pipeline (excerpted in `[[nemotron-4-synthetic]]`) is carried forward. The 98%-synthetic framing is not re-stated for Nemotron 3, but the source notes the same pipeline base, so the Nemotron-4 synthetic ratio is approximately preserved for the SFT corpus.

## Reasoning budget control

> Granular "reasoning budget control" lever letting users trade tokens for accuracy at inference.

This is the Nemotron 3 analog of Qwen 3's thinking-budget mode (ch-34) and s1's budget-forcing (ch-35 §4 and `[[s1]]`). Different mechanism, same user-facing affordance: at inference the operator chooses how many reasoning tokens to spend. For ch-35 the significance is that *even after distillation SFT*, the resulting policy has a non-trivial inference-time knob that distills-SFT-only models (Stratos, Sky-T1) do not ship.

---

## Why ch-35 includes this case study

Nemotron 3 is the bridge from ch-35 (SFT case studies) to ch-40+ (RL family). The synthetic pipeline is no longer a headline because it is now *assumed*; the question has shifted to **how RL environments are structured on top of the synthetic substrate**. Readers should carry forward three things:

1. The Nemotron-4 synthetic apparatus is *still there*, just not foregrounded.
2. The RL-organization shift (single multi-env run vs sequential single-env runs) is the 2025 frontier direction.
3. Open-artifact release (GenRM + policy + datasets) is the new norm; open-recipe release is lagging.
