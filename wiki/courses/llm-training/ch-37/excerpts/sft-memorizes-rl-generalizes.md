---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: arXiv:2501.17161v2 (no library card for slug sft-memorizes-rl-generalizes on 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2501.17161
created_at: "2026-09-15"
---

# Excerpt: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

- **Authors:** Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, et al. (HKU, UC Berkeley, Google DeepMind, NYU, University of Alberta)
- **Year:** 2025 (arXiv v1 2025-01-28; v2 2025-05-26; ICML 2025)
- **Source type:** paper
- **Used in:** ch-37 §5 (SFT versus on-policy RL), agentic section (sequential revision with verifier text), Generalization lens.

## Setup (§3–§5, App. A)
- Backbone: Llama-3.2-Vision-11B. Following the RLHF pipeline, the model is initialized with SFT before RL; RL uses PPO (§3, §5).
- Multi-turn formulation ("sequential revision", §3): at step t the input is the system prompt concatenated with all prior model outputs and verifier outputs [v_k^out, v_k^ver]_{k=0}^{t−1}. The verifier returns a reward and text.
- GeneralPoints (§4.1): four cards, produce an equation equal to 24 using each card once. GP-L gives cards as text, GP-VL as an image.
- GeneralPoints reward (App. A.3): r = 5 for a legal equation equal to the target; r = −1 for a legal equation using each card once but not equal to the target; r = −1 for exceeding the maximum of 5 verification steps; r = −2 for numbers not among the cards; r = −3 for other illegal equations; an additional −1.5 in GP-VL for misrecognized cards.
- Rule variant (§5.1, App. A): training and in-distribution evaluation count J, Q, K as 10; the out-of-distribution rule counts them as 11, 12, 13. V-IRL: absolute orientation action space in distribution, relative orientation out of distribution.
- Visual variant (§5.2): GP-VL trained on black suits and tested on red suits; V-IRL trained on New York City routes and tested on routes from other cities.

## Results (§5.1, §5.2, Fig. 6–7)
| Task (out-of-distribution) | Initialization | After RL | After SFT |
|---|---|---|---|
| GP-L, rule variant | 11.5% | 15.0% | 3.4% |
| V-IRL-L, rule variant | 80.8% | 91.8% | 1.3% |
| GP-VL, rule variant | 11.2% | 14.2% | 5.6% |
| V-IRL-VL, rule variant | 35.7% | 45.0% | 2.5% |
| GP-VL, visual variant | 23.6% | 41.2% | 13.7% |
| V-IRL-VL, visual variant | 16.7% | 77.8% | 11.1% |

- Verification steps (§5.5, Fig. 10, GP-L, same compute): OOD improvement +0.48% with 1 verification step, +2.15% with 3, +2.99% with 5, +5.99% with 10.
- SFT before RL (§5.4, Fig. 9): end-to-end RL from the base Llama-3.2-Vision-11B failed to improve, because the base model produced long, unstructured responses from which rewards could not be extracted. The authors note this does not contradict DeepSeek-R1 because the backbone differs.

## Limits
- Two synthetic or narrow tasks, one 11B backbone; SFT data are task demonstrations, not a general instruction mixture.
- Seeds or confidence intervals for the OOD numbers are not reported in §5.1–5.2.

## Verification
- Read on 2026-09-15 against the arXiv:2501.17161v2 PDF text (Abstract, §1, §3–§5, App. A).
