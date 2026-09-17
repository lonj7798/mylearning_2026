---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sft-memorizes-rl-generalizes.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2501.17161
created_at: "2026-09-15"
---

# Excerpt: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

**Authors:** Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma.
**Version read:** arXiv:2501.17161v2 (26 May 2025), ICML 2025 (PMLR 267).
**Status:** no library card existed for this slug on 2026-09-15; every quote and number below was read at the stated locus in the v2 PDF.

## Setup (§3, §4, §5, App. C)
Backbone Llama-3.2-Vision-11B. RL is multi-turn PPO over a sequential-revision formulation with a verifier; SFT runs first in every main experiment and the RL and SFT compute budgets are then scaled separately from that shared checkpoint. Tasks: GeneralPoints (arithmetic card game, target 24) and V-IRL (navigation), each in a pure-language (-L) and a vision-language (-VL) variant. Rule variant for GeneralPoints: train with J/Q/K = 10, evaluate with J/Q/K = 11/12/13. Rule variant for V-IRL: train with an absolute orientation action space, evaluate with a relative one. Visual variant: train on black suits, evaluate on red suits; train on New York City routes, evaluate on the V-IRL VLN mini benchmark. SFT data are "optimal single-turn prompt-response pairs, without any verification or revision steps" (App. C.1). All training on 8 H800 GPUs (App. C.2).

## Out-of-distribution results under rule variants (§5.1, Fig. 6)
Equal compute for RL and SFT; the shared starting checkpoint is the baseline.

| Task | Init | RL | SFT |
|---|---|---|---|
| GP-L (episode success) | 11.5% | 15.0% (+3.5) | 3.4% (−8.1) |
| V-IRL-L (per-step accuracy) | 80.8% | 91.8% (+11.0) | 1.3% (−79.5) |
| GP-VL | 11.2% | 14.2% (+3.0) | 5.6% (−5.6) |
| V-IRL-VL | 35.7% | 45.0% (+9.3) | 2.5% (−33.2) |

## Visual variants (§5.2, Fig. 7)
GP-VL: 23.6% → 41.2% (RL, +17.6) and → 13.7% (SFT, −9.9). V-IRL-VL: 16.7% → 77.8% (RL, +61.1) and → 11.1% (SFT, −5.6). The RL result is +33.8 points over the previous published V-IRL VLN mini result (44.0% → 77.8%).

## Conditions the paper states (§5.4, §6)
- "without SFT, all end-to-end RL runs fail to improve"; the base model "tends to generate long, tangential, and unstructured responses", so task rewards cannot be extracted (§5.4, Fig. 9, Fig. 20). The authors add that this does not contradict DeepSeek-R1 because the backbone differs.
- RL started from "an overly-tuned SFT checkpoint" does not recover OOD performance; the model "collapses to the training rule" (§6, Figs. 19, 21).
- Verification iterations matter: at equal compute, OOD gains were +0.48% (1 step), +2.15% (3), +2.99% (5), +5.99% (10) (§5.5, Fig. 10).
- An ablation with sub-optimal SFT trajectories (errors plus verifier messages, matching the RL data format) still shows memorization with degraded OOD performance (App. C.1, Fig. 15).
- Scaling SFT compute lowered card-recognition accuracy; scaling RL compute raised it (§5.3, Fig. 8). The authors' hypothesis for the drop is local overfitting to reasoning tokens (§6).

## Reward design (App. A.3)
r = 5 for a legal equation equal to the target; r = −1 for a legal equation using each card once that does not equal the target; r = −1 for exceeding the maximum verification step (5); r = −2 for a legal equation containing numbers not among the cards; r = −3 for all other illegal equations; an extra r = −1.5 in GP-VL when the cards are not recognized correctly.

## Hyperparameters (App. C.2, D.1)
One shared learning rate per run, all components tunable. SFT learning-rate search {1e-4, 1e-5, 1e-6, 5e-7, 1e-7} with all parameters trained, {1e-6, 1e-7} with the vision encoder frozen, {1e-6, 5e-7, 1e-7} with encoder and adapter frozen. RL learning-rate search {2e-6, 1e-6}. Compute accounting: X_SFT = 6N(D_init + D_SFT), X_RL = 6N(D_init + D_RL) + 2N·D_buffer (App. C.3).

## How ch-38a uses it
§2 (the headline rule-variant and visual-variant numbers), §3 (the conditions above, read against [[debunk-sft-generalization]]), Negative-feedback section (the negative reward values), Recipe.
