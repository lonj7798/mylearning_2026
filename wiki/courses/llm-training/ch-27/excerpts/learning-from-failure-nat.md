---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: "Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents (arXiv:2402.11651v2)"
source_url: https://arxiv.org/abs/2402.11651
created_at: "2026-09-15"
note: "No library card exists for this source as of 2026-09-15; this excerpt is taken from the primary text."
---

# Excerpt: NAT — failed agent trajectories trained under a failure label

**Authors:** Renxi Wang, Haonan Li, Xudong Han, Yixuan Zhang, Timothy Baldwin (LibrAI; MBZUAI; The University of Melbourne). arXiv v1 2024-02, v2 2024-04-16.

## Method (§3.3)
1. Seed questions with gold answers; GPT-3.5 (1106) generates ReAct trajectories three times at temperatures 0.2, 0.5, 0.7; each trajectory is labeled positive or negative by comparing its answer with the gold answer.
2. Negative-aware reformatting: positives get the suffix "Please generate a solution that **correctly** answers the question."; negatives get "Please generate a solution that **incorrectly** answers the question." (the actual prompts are slightly longer).
3. Loss is computed only on model-generated text. At inference the positive prompt is used.
- Baselines: Vanilla (positives only) and NUT (negatives added without a label).
- Type in the course standard: negative as conditioning (§6.1 type 3). No likelihood is pushed down.

## Settings (§4.1)
- Llama-2-Chat 7B and 13B; 2 epochs; batch 64; cosine schedule with 3% warmup; max LR 5e-5; 4×A100 with DeepSpeed ZeRO-3.
- Math tools: SymPy calculator; QA tools: Serper Google search with MPNet and DPR re-ranking.

## Results
- Math average over GSM8K, ASDiv, SVAMP, MultiArith (Table 2):
  | Model, positives | Vanilla | NUT | NAT |
  |---|---|---|---|
  | 7B, 2k | 55.90 | 63.39 | 64.64 |
  | 7B, 5k | 64.17 | 64.95 | 67.42 |
  | 13B, 2k | 65.18 | 66.53 | 67.74 |
  | 13B, 5k | 70.76 | 69.10 | 71.28 |
- HotpotQA 7B (1,500 negatives) EM / F1: Vanilla 27.44 / 36.41; NUT 28.04 / 40.96; NAT 28.80 / 41.37 (Table 3). StrategyQA with 1,000 positives and 500 negatives: 7B 55.40 / 62.40 / 65.80 (Table 4).
- Negative quantity (§5.1, Fig. 2): with 2k or 5k positives fixed, performance rises with negatives and plateaus near 11k. With 10k negatives fixed, the added value of negatives decreases as positives increase (Fig. 3).
- Negative quality (§5.2, Table 5): 10k negatives from a fine-tuned Llama-2-7B instead of GPT-3.5 change the average by −3.16 (2k positives) and −6.20 (5k positives), vs +8.74 and +3.25 with GPT-3.5 negatives.
- Behavior (§5.3, Table 6, GSM8K test, 7B): accuracy / action error — Vanilla 35.63 / 3.58%; NUT 44.43 / 10.47%; NAT 46.93 / 7.90%. Training data action error is 4.01% for positives only and 15.33% with negatives.
- Prompt wording (§5.4, Table 7): Correct/Incorrect 63.55; Good/Bad 63.91; A/B 63.15; random strings 64.04; the authors conclude that the gain comes from separating positive and negative data, not from the wording.
- Perplexity on held-out successful trajectories decreases as negatives are added, but a gap to models with more positives remains (§5.3, Fig. 4).

## Limits
- Tasks are math and QA with search or calculator tools; no multi-turn environment with state is tested. Results are single runs except HotpotQA (mean of 5 runs, Table 3).
