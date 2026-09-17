---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: arXiv:2410.08146v1 (Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning)
source_url: https://arxiv.org/abs/2410.08146
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: Process advantage verifiers (PAV)

Used by [[read]] §4 and the negatives section. Authors: Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, Aviral Kumar (Google Research, Google DeepMind, CMU). arXiv v1 2024-10-10; checked on 2026-09-15.

## Definitions (§2, §3)
- A response is a sequence of steps separated by a newline; the prefix is the state `s_h`, the next step is the action `a_h`, and the dynamics are deterministic.
- `Q^π(s_h, a_h)` is the probability that completions sampled from policy `π` after the prefix plus step reach the correct final answer, measured with a regular-expression answer check `Rex` (Eq. 1). This is the quantity Math-Shepherd-style and OmegaPRM-style labels estimate.
- `A^μ(s_h, a_h) = Q^μ(s_h, a_h) − V^μ(s_h)` is the progress made by the step under a **prover policy** `μ`, which is not the base policy.

## Objective (§3.2)
Effective per-step reward, from the policy-gradient form of the PAV objective (Eq. 4, Eq. 5):

```
∇_π ℓ_PAV-RL(π) = Σ_h ∇_π log π(a_h | s_h) · ( Q^π(s_h, a_h) + α · A^μ(s_h, a_h) )
```

- `Q^π` is the outcome term the standard policy gradient already carries; `A^μ` is the dense process term; `α ≥ 0` weights it.
- Setting `μ = π` makes the update identical to outcome-only RL, because a policy gradient already subtracts a baseline (§3.2). This is the paper's argument that a PRM predicting the base policy's own values adds nothing to RL.
- Prover choice: an over-capable prover succeeds from every step and so scores good and bad steps alike; a very weak prover succeeds from none. The paper defines good provers as complementary to the base policy, and reports that provers weaker than the base policy can still help (§3.3, §3.4; Fig. 1a).

## Results
- Test-time beam search against PAVs: more than 8% higher accuracy than best-of-n re-ranking with an ORM at equal compute, and 1.5-5x more compute-efficient (Abstract; Fig. 1b).
- Online RL with `Q^π + α A^μ` as dense reward on Gemma 2B and 9B, initialized from SFT then rejection fine-tuning: PAV-RL is more than 7% better in test accuracy than ORM-RL and reaches that accuracy about 6x faster in samples; the RFT policy improves by 11% (2B) and 15% (9B) (§5, Fig. 7).
- Pass@N after RL is higher for PAV-RL than ORM-RL for all `N ≤ 128` (2B, §5 Fig. 8).
- The 2B SFT policy was the better prover for both the 2B and 9B runs, and both provers became weaker than the base policy within a few RL steps while the fixed PAV kept helping (§5).

## Settings (App. E)
- Algorithm REINFORCE with a token-level value baseline; 10,000 iterations (2B) and 5,000 (9B); Adam; learning rate 1e-7; batch size 32; maximum response length 512; linear warmup over 10% of iterations then cosine decay.
- KL regularization against the initial RL iterate with coefficient 0.001.
- `α` in RL: 5.0 for Gemma 2B and 3.0 for 9B; values in the range 0.5 to 6.0 improved over ORM-RL. `α` for test-time search: 0.5 for 2B and 9B, 0.2 for 27B, selected on a validation set over a grid of [0.0, 1.0] at 0.1 intervals (App. D, App. E).
