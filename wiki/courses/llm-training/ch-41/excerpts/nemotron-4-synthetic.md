---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md and nemotron-4-synthetic-recipe.md (arXiv:2406.11704v2)
source_url: https://arxiv.org/abs/2406.11704
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and primary source)"
---

# Excerpt: Nemotron-4-340B-Reward as a multi-attribute regression RM

Used by [[read]] Core insight, §3.6, §4.3, and the Recipe. Source cards: [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]]. The duplicate card [[nemotron]] is not cited.

The earlier version of this excerpt and the old figure used attribute weights 0.35 / 0.35 / 0.15 / 0.10 / 0.05, described separate per-attribute heads with 0–4 Likert labels as reported in this report, and described the RM as the scorer inside RPO at different weightings. The report gives no attribute weights, no RM learning rate, epochs, or loss (it refers to Wang et al., 2024), and describes one linear projection to a 5-dimensional vector (§3.1).

## Reward model (§3.1, Table 4)
- Nemotron-4-340B-Base with the final softmax layer replaced by a linear projection from last-layer hidden states to five HelpSteer attributes: Helpfulness, Correctness, Coherence, Complexity, Verbosity; aggregated by a weighted sum at inference.
- Trained on 10K HelpSteer2 human preference examples.
- The report states that multi-attribute regression separates helpfulness from artifacts such as length better than pairwise ranking models; no ablation is given (Interpretation).
- RewardBench: overall 92.0, Chat 95.8, Chat-Hard 87.1, Safety 91.5, Reasoning 93.7, Prior Sets 67.4 (Table 4).

## Use in the alignment pipeline (§3.2)
- Over 98% of SFT and preference data is synthetic; about 20K human-annotated examples (10K SFT, 10K HelpSteer2).
- Preference pairs: ground truth or a verifier where available; otherwise LLM-as-judge early and Reward-Model-as-judge later; Chat-Hard accuracy 0.87 (RM-as-judge) vs 0.54 (LLM-as-judge) (§3.2.3).
- Synthetic dialogues below a reward-model score threshold are removed; threshold not given (§3.2.2).

## Later out-of-distribution evaluations
- RM-Bench: average 69.5, hard 56.1; correctness and verbosity scores separate chosen from rejected only in safety ([[rm-bench]] Table 3, §4.3).
- PPE: DPO pairs labelled by this RM produced a Llama-3.1-8B policy with Arena score 1172 (CI 1163–1180), against 1178 for the base model ([[ppe-reward-model-eval]] Table 3).
