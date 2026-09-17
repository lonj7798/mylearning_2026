---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: GLM-5 Team (Zhipu AI and Tsinghua University), "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763v2
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
source_type: official technical report
---

# Excerpt: GLM-5 SFT thinking modes, masked erroneous segments, and token-in-token-out

No library card for this source exists yet (planned slug `glm-5`). Only passages used by [[read]] §1 and §7 are quoted.

## §3.1 Supervised Fine-Tuning

- "GLM-5 extends the maximum context length to 202,752 tokens during SFT. Along with an updated chat template, the model supports three distinct thinking characteristics".
- "Interleaved Thinking: the model thinks before every response and tool call, improving instruction following and the quality of generation."
- "Preserved Thinking: in coding agent scenarios, the model automatically retains all thinking blocks across multi-turn conversations, reusing existing reasoning instead of re-deriving it from scratch. This reduces information loss and inconsistencies, and is well-suited for long-horizon, complex tasks."
- "Turn-level Thinking: the model supports per-turn control over reasoning within a session—disable thinking for lightweight requests to reduce latency/cost, enable it for complex tasks to improve accuracy and stability."
- Coding and agent data: "We further improve the SFT data using expert reinforcement learning and rejection sampling. Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions."

## §4.1.2 Optimizing Asynchronous Training Stability

"In an RL rollout setting, token-in-token-out (TITO) means the training pipeline consumes the exact tokenization and decoded-token stream produced by the inference engine, and uses it directly to build trajectories for learning. In contrast, text-in-text-out treats the rollout engine as a black box that returns finalized text; the trainer then reconstructs the trajectory by re-tokenizing that text (and often re-deriving boundaries and truncation) before computing losses. This seemingly small choice is consequential: re-tokenization can introduce subtle mismatches in token boundaries, whitespace/normalization handling, truncation, or special-token placement, which in turn can corrupt step alignment between actions and rewards/advantages—especially when rollouts are streamed, truncated, or interleaved across many actors."

## Not reported in these sections

The share of SFT tokens that are masked erroneous segments, any ablation of masking against discarding those trajectories, and SFT optimizer settings.

## Connections

- [[read]] §1.1 (mask table), §7, Negative samples section; [[minimax-m2-interleaved-thinking]].
