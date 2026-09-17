---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2309.00267v3 (RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback); library card [[rlaif-scaling]]
source_url: https://arxiv.org/abs/2309.00267
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and primary source)"
---

# Excerpt: canonical RLAIF and direct RLAIF (Lee et al.)

Used by [[read]] §3.5. Checked against arXiv v3 (2024-09-03) on 2026-09-15.

The library card for this slug has no Verification section. The earlier version of this excerpt stated that d-RLAIF uses the labeler's log-probability of "Response 1 is better" as the reward and that AI labels are about 100× cheaper. Corrections:
- d-RLAIF prompts the LLM to rate a single response from 1 to 10 and uses the probability-weighted expected score, rescaled to [−1, 1], as the reward (§2.2.2).
- The authors estimate AI labeling at over 10× cheaper than human annotation (§4.1, App. L).

## Direct RLAIF (§2.2.2)
- "In d-RLAIF, the LLM is prompted to rate the quality of a generation between 1 and 10."
- The likelihood of each score token 1..10 is normalized to a distribution; `s(y|x) = Σ_{i=1}^{10} i · P(i | y, x)`; the score is normalized to [−1, 1] and used as the RL reward in place of an RM score.
- Motivation: an RM trained on earlier policy samples becomes "stale" as the policy is trained; d-RLAIF scores current generations directly and removes AI preference labeling and RM training (§2.2.2).

## Results (Table 1, human evaluation)
| Comparison | Summarization | Helpful dialogue |
|---|---|---|
| RLAIF vs SFT | 71% | 63% |
| RLHF vs SFT | 73% | 64% |
| RLAIF vs RLHF | 50% | 52% |
| Same-size RLAIF vs SFT | 68% | – |
| d-RLAIF vs SFT | 74% | 66% |
| d-RLAIF vs same-size RLAIF | 60% | – |

- Harmless rate: SFT 64%, RLHF 76%, RLAIF 88% (Table 1).
- d-RLAIF experiments use the instruction-tuned PaLM 2 XS as the AI labeler (§4.3).
- RLAIF and RLHF policies produce longer responses than SFT; after controlling for length both still outperform SFT (§4.1, App. J).
