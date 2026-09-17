---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2509.02479
primary_version: arXiv:2509.02479v2 (3 Sep 2025)
created_at: "2026-09-15"
---

# Excerpt: SimpleTIR — End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning

Xue, Zheng, Liu, Li, Zheng, Ma, An (Nanyang Technological University; TikTok), 2025.

## Diagnosis (§3.1-§3.2)
- In multi-turn tool-integrated reasoning the interpreter output for turn `k` is concatenated into the prompt
  for turn `k + 1`. That text can be outside the model's pretraining distribution, and the model's own
  continuations then contain tokens to which it assigns very low probability. Fig. 3 traces one rollout:
  token probabilities are high in turn 1, low-probability segments appear in the model's own text in turns 2
  and 3, and turn 4 collapses. Masking the loss on the tool tokens does not prevent this (§3.1).
- Proposition 3.1: the L2 norm of the policy gradient with respect to the logits of token `c` at step `t` is
  `(m_{i,t} / Σ_j m_{i,j}) · ρ_{i,t}(θ) · g_{i,t} · |Â_i| · sqrt(1 − 2P(c) + Σ_{j∈A} P(j)²)`, where `m` is the
  feedback mask, `ρ` the importance ratio, `g` a gate that is active when the PPO update is not clipped, and
  `P` the policy distribution at that position.
  - For a negative-advantage trajectory the ratio `ρ` is unbounded above, and it is largest where
    `π_old(c|·)` is small.
  - The probability term approaches its maximum when `P(c)` is small while the rest of the distribution is
    sharp (large `Σ_j P(j)²`).
- Credit assignment: low-probability tokens are more frequent in later turns, and a terminal-only reward
  assigns the same negative advantage to the earlier, well-formed turns (§3.2).

## Method (§3.3)
A turn that produces neither a complete code block nor a final answer is a "void turn". Any trajectory that
contains a void turn is removed from the batch before the GRPO update (the policy loss for the whole
trajectory is masked). The authors report that filtering by lowest token probability or by highest importance
ratio does not stabilize training in this setting (§3.3, Fig. 5 bottom).

## Results
- Table 1, from the Qwen2.5-7B base model: SimpleTIR-7B reaches AIME24 50.5, AIME25 30.9, MATH500 88.4,
  OlympiadBench 54.8, AMC23 79.1, HMMT25 29.7. The abstract compares AIME24 50.5 with a text-only baseline
  of 22.1.
- Table 2 (highest score within 1,000 gradient steps): SimpleTIR-7B AIME24 50.5 / Math500 88.4; naive
  multi-turn 20.8 / 73.1; low-probability-token filtering 23.3 / 72.8; high-importance-ratio filtering
  26.3 / 75.0; stopping generation on a void turn without filtering the trajectory 26.1 / 77.3.
- Implementation (§3.4): no chat template for base models; tool output prefixed with "Code Execution Result:";
  generation is stopped after a complete code block and the real interpreter output is appended.
- Hyperparameters (App. Table 6): rollout temperature 1.0; initial maximum response length 16,384; initial
  maximum 5 interaction turns; train batch 512; sampling batch 1,280; 16 rollouts per prompt; PPO clip ratio
  0.2 / 0.28; entropy coefficient 0; KL coefficient β = 0; PPO epochs 4; actor learning rate 1e-6; gradient
  clipping 1.0; discount 1.0; GAE λ 1.0. The appendix text states that exploration is induced by temperature
  and multi-rollout sampling rather than by an entropy bonus.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2509.02479v2 (scratchpad `sources/simpletir.txt`).
- Not reported: policy entropy curves (the paper measures token probabilities and gradient norms, not
  entropy); pass@k; retention on non-math benchmarks.
