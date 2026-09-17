---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/pairrm.md (arXiv:2306.02561v3; llm-blender/PairRM model card @5b880cc)
source_url: https://arxiv.org/abs/2306.02561
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and primary source)"
---

# Excerpt: PairRanker and the PairRM checkpoint

Used by [[read]] §3.4. Source card: [[pairrm]].

The earlier version of this excerpt described `[CLS] x [SEP] y_A [SEP] y_B` inputs, swap-and-average inference worth 2–3 pp, an O(N log N) tournament, and a ~2 pp DPO lift from pair filtering. None of these is in the paper or model card. Corrections are below.

## Architecture and loss (§3.2-3.3, App. A)
- Input: `<s><source> x </s> <candidate1> y_i </s> <candidate2> y_j </s>` on DeBERTa-v3-large (400M).
- Five-layer tanh MLPs on the source+candidate embeddings give `s^i` and `s^j`; pair score `s_ij = s^i − s^j`.
- Per-metric BCE: `L_Q = −z_i log σ(s^i) − z_j log σ(s^j)` with `(z_i, z_j)` set by which candidate scores higher on metric Q (BARTScore on MixInstruct).
- 5 sampled pairs per input; candidate order shuffled during training; no separate swap-and-average step or order ablation is described.
- Aggregation: MaxLogits and MaxWins need O(N²) comparisons; one bubble-sort pass needs N − 1.

## Results
- MixInstruct (N = 11): average GPT-Rank 3.20 for PairRanker vs 3.90 best single LLM, 3.50 SimCLS, 3.66 SummaReranker (Table 2). Pearson with GPT-Rank 46.98 vs 41.13 for SummaReranker (Table 3).
- PairRM checkpoint (0.4B), model card: Auto-J 59.05 (UltraRM-13B 59.85, GPT-4 61.9); HHH-Alignment 84.62 (UltraRM-13B 83.71, GPT-4-0613 88.69); MT-Bench human judgments 59 (UltraRM-13B 56, GPT-4-0613 63.87).
- PairRM training data: summarize_from_feedback, webgpt_comparisons, synthetic-instruct-gptj-pairwise, hh-rlhf, chatbot_arena_conversations, UltraFeedback; hyperparameters not reported.

## Generality notes
- No single open LLM is best: Vicuna is top-ranked on 21.22% of 5,000 instructions (Fig. 1).
- With shuffled candidate order the ranker agrees with itself more than 90% of the time (App. C.5).
