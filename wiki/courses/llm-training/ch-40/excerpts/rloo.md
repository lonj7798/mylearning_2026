---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2402.14740v2 (Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs); library card [[rloo]]
source_url: https://arxiv.org/abs/2402.14740
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the primary source because the library card has no Verification section)"
---

# Excerpt: RLOO — the leave-one-out baseline and what the paper actually measured

Used by [[read]] §1, §2, and the Recipe. Authors: Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, Sara Hooker (Cohere For AI). Text read from arXiv v2 (26 Feb 2024) on 2026-09-15.

## Estimator (§2.3)
```
(1/k) Σ_i [ R(y^(i), x) − (1/(k−1)) Σ_{j≠i} R(y^(j), x) ] ∇ log π(y^(i) | x),   y^(1..k) i.i.d. ~ π_θ(·|x)
```
The paper attributes the estimator to Kool et al. 2019 and describes the baseline as "akin to a parameter-free value-function, but estimated at each training step". KL control is applied inside the reward: `R(x, y) = r_φ(x, y) − β log(π_θ/π_ref)` (Eq. 3), following InstructGPT.

## Results (Table 1: simulated win-rate against reference completions)
| Method | TL;DR | HH (Pythia) | HH (Llama) |
|---|---|---|---|
| RLOO (k=4) | 77.9 | 43.7 | 64.1 |
| RAFT (k=4) | 73.2 | 42.1 | 63.3 |
| RLOO (k=2) | 74.2 | 47.6 | 62.2 |
| RAFT (k=2) | 72.1 | 37.7 | 58.4 |
| REINFORCE with baseline | 70.7 | 37.9 | 55.3 |
| Vanilla PG | 70.4 | 36.4 | 52.3 |
| PPO | 67.6 | 29.2 | 32.0 |
| DPO | 66.6 | 39.0 | 61.9 |

RLOO k=4 exceeds PPO by 10.3, 14.5, and 32.1 points on the three settings (§5.2). Averaged over the three pairings, RLOO wins 61.3 (k=2) and 61.9 (k=4) against RAFT's 56.1 and 59.5.

## Settings (App. "Preference Training")
Pythia-6.9B on TL;DR and Anthropic-HH; Llama-7B on Anthropic-HH; context length 512 tokens for SFT and RM training. TL;DR: 600 steps, rollout batch 512, step batch 256, β = 0.03. HH (Pythia): 393 steps, same batches, β = 0.10. Llama: rollout and step batch 2048 over 2 epochs. Constant LR 1e-6 with 3% linear warm-up, chosen from a sweep of {1e-6, 1e-5, 2e-5}; two gradient steps per batch.

## The clipping observation (§3.2) and its scope
"We empirically found in our RLHF setting that the loss is actually clipped on average < 5% of the time per batch, throughout training across all dataset and base-model pairings", and removing clipping "gives a slight boost in performance". This holds for two gradient steps per rollout batch on short generations; recipes that take 16 updates per rollout batch ([[dapo]], [[deepseek-r1-recipe]]) or four ([[gspo]]) are further off-policy within a batch, and there the clip bound is active.

## Corrections to the library card made here
- The card's hyperparameter table (β 0.05, LR 1e-6–3e-6, batch 32–64 prompts, temperature 1.0, max new tokens 53/256) is not what the appendix prints; the values above are.
- "Removes the value network → ~50% memory footprint" and "beats PPO by 5–20% win rate" are not stated in this form; the measured win-rate gaps are 10.3–32.1 points and no memory measurement is reported.
- The card's "equivalent (up to scaling) when G is large" phrasing understates an exact relation: the leave-one-out and mean-subtracted advantages differ by the factor `k/(k−1)` at every `k` ([[dr-grpo]] App. A).
