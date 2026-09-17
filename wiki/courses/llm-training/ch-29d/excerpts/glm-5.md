---
chapter: ch-29d
course: llm-training
phase: read
excerpt_of: GLM-5 Team (Zhipu AI and Tsinghua University), "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763v2 (2026-02-24); no library card at wiki/raw-data/llm-training/**/glm-5.md on 2026-09-15
source_url: https://arxiv.org/abs/2602.15763
source_type: official technical report
created_at: "2026-09-15"
---

# Excerpt: GLM-5 — masked erroneous segments in SFT and excluded environment failures in agentic RL

Only passages used by [[read]] §5, §6, the Negative samples section, and the Recipe table are quoted. Checked on 2026-09-15 against the v2 PDF text.

## §3.1 Supervised Fine-Tuning (Coding and Agent data)
> "For Coding and Agent tasks, compared to GLM-4.5, GLM-5 constructs a large number of execution environments to obtain high-quality trajectories, with particular emphasis on real-world scenarios and long-horizon tasks. We further improve the SFT data using expert reinforcement learning and rejection sampling. Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions."

## §4.1.2 Optimizing Asynchronous Training Stability ("Dropping off-policy and noisy samples")
> "We discard a sample if its oldest rollout version is too stale, i.e., if w′ − w_0 > τ, where τ is a predefined threshold."

> "Additionally, coding-agent sandboxes can be inherently unstable and may fail for reasons unrelated to the model (e.g., environment crashes). Such failures introduce noisy training signals because they reflect environment instability rather than the model's capability. To mitigate this, we record the failure reason for each sample and exclude samples that fail due to environment collapse. For group-based sampling methods such as GRPO, removing failed samples can leave an incomplete group. In that case, we pad the group by repeating valid samples if the number of valid samples exceeds half of the group size; otherwise, we drop the entire group."

- The same subsection masks tokens whose importance ratio falls outside [1 − ε_ℓ, 1 + ε_h] from the gradient (Eqs. 3-5).

## §4.2.3 Search tasks (difficulty filter and verification)
> "(1) Remove questions that a tool-free reasoning model correctly answers in at least one of eight independent attempts. (2) Filter out questions solvable by an early-stage agent with basic search, browsing, and computation within a few steps. (3) Apply a verification agent for bidirectional validation [...] rejecting samples with non-unique answers, inconsistent evidence, or incorrect labels."

## §4.2.5 Slide Generation (rejection sampling and masking)
- "the reward functions used in RL are transferred into a data filtering pipeline"; Best-of-N keeps the highest-quality candidate.
- "some trajectories contain defects confined to only a small number of pages. Discarding such samples would reduce effective data utilization and increase generation cost. To address this, we introduce a masking-based correction mechanism that automatically identifies defective pages and applies masking, while retaining the high-quality content within the same trajectory."
- Reported outcome of the whole pipeline (RL, rejection sampling, masking together): pages with a strict 16:9 aspect ratio rose from 40% to 92%; human evaluation against GLM-4.5 gave an overall win rate of 67.5%.

## Not reported
The share of SFT tokens masked as erroneous segments; how erroneous segments are detected; any ablation of masking versus discarding; the staleness threshold τ; which valid samples are repeated when a group is padded; the fraction of rollouts excluded for environment collapse.
