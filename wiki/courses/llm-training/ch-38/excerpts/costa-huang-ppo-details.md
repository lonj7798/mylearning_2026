---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/costa-huang-ppo-details.md
source_url: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
revised_at: "2026-09-15"
---

# Excerpt: The 37 Implementation Details of PPO: core details and their evidence

Checked against the ICLR Blog Track post (2022-03-25) on 2026-09-15. Used in read.md §1, §3, §9. The library card numbers its items differently from the post (for example it lists value-loss clipping as item 4); the numbering below follows the post. The RLHF-specific follow-up is [[n-implementation-details-rlhf-ppo]].

## The 13 core details, in the post's order
1. Vectorized architecture. 2. Orthogonal initialization of weights and constant initialization of biases. 3. Adam epsilon 1e-5. 4. Adam learning-rate annealing. 5. Generalized Advantage Estimation. 6. Mini-batch updates (shuffle and split). 7. Normalization of advantages, "at the minibatch level instead of the whole batch level". 8. Clipped surrogate objective. 9. Value function loss clipping. 10. Overall loss and entropy bonus. 11. Global gradient clipping (norm 0.5). 12. Debug variables. 13. Shared and separate MLP networks for policy and value.
The post also lists 9 details for continuous-action robotics tasks, 5 LSTM details, and others.

## Evidence quoted in the post (control and Atari tasks, not language models)
- Detail 7: Andrychowicz et al. "find per-minibatch advantage normalization to not affect performance much".
- Detail 8: Engstrom et al. "find the PPO's clipped objective to have similar performance to TRPO's objective when they controlled other implementation details to be the same".
- Detail 9: L^V = max[(V_θt − V_targ)², (clip(V_θt, V_θt−1 − ε, V_θt−1 + ε) − V_targ)²]; Engstrom et al. "find no evidence that the value function loss clipping helps with the performance"; Andrychowicz et al. "suggest value function loss clipping even hurts performance".
- Detail 11: global gradient clipping offers "a small performance boost".
