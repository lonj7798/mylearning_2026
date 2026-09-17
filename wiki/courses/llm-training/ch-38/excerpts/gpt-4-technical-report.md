---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: (no library card at time of writing; primary source)
source_url: https://arxiv.org/abs/2303.08774
revised_at: "2026-09-15"
---

# Excerpt: GPT-4 Technical Report: calibration and capability after RLHF

Organization: OpenAI. arXiv v1 2023-03; checked against arXiv:2303.08774v6 (2024-03-04) on 2026-09-15. Source type: official technical report. Used in read.md §6.5 and the Generalization lens.

## Calibration (§5, Figure 8)
- "Interestingly, the pre-trained model is highly calibrated (its predicted confidence in an answer generally matches the probability of being correct). However, after the post-training process, the calibration is reduced (Figure 8)."
- Figure 8: calibration on a subset of MMLU, bins by the model's log-probability of each A/B/C/D choice. Pre-trained model ECE 0.007; post-trained model (panel labeled "model=ppo") ECE 0.074. Caption: "The post-training hurts calibration significantly."

## Capability (App. B, Table 8)
- Multiple-choice portions of the exam benchmark: "the base model achieves an average score of 73.7% while the RLHF model achieves a score of 74.0%, suggesting that post-training does not substantially alter base model capability."

## Not reported
- Separate effects of SFT and RL on calibration; RLHF hyperparameters.
