---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ultrafeedback-construction.md
source_url: https://arxiv.org/abs/2310.01377
primary_version: arXiv:2310.01377v2 (2024-07-16); v1 2023-10; ICML 2024
created_at: "2026-04-23"
revised_at: "2026-09-15 (rewritten from the primary text; the earlier version of this excerpt stated 0-10 aspect scales and other unsupported values)"
---

# Excerpt: UltraFeedback: Boosting Language Models with Scaled AI Feedback, construction and agreement sections

Authors: Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, et al. (Tsinghua University and collaborators). Source type: paper. Read in the v2 PDF text on 2026-09-15 for ch-15 §2 and §7. UltraFeedback labels are produced by GPT-4, not by humans; ch-15 uses it as the AI-labeled contrast case and for its human-agreement measurement.

## Construction (§2.2-§2.4)
- "we obtain 63,967 instructions of various types from the six publicly available high-quality datasets" (§2.2).
- "we set up a pool of 17 models" (commercial, LLaMA-series, non-LLaMA-series) and "randomly sample four different models from the pool to complete each instruction" (§2.3). Different sizes and families were chosen "to alleviate the potential spurious correlation between text styles and response quality" (§2.3).
- A principle (for example honesty or helpfulness) is sampled and added to the system prompt for each completion (§2.3, App. G.1).
- "After generating 255,864 model completions based on the 63,967 instructions, we employ GPT-4 to provide two types of feedback": scalar scores per aspect and a textual critique; "over 1 million feedback data in total" (§2.4).
- Four annotation techniques (§2.4): (1) decomposition into instruction-following, truthfulness, honesty, and helpfulness; (2) "detailed documentation of scores from 1 to 5"; (3) all four completions of one instruction are scored in one prompt "to reduce randomness"; (4) a rationale is generated before the score.

## Reward-model data (App. D.1, E.1)
- Loss: L_ranking = −log(σ(r_θ(x, y_c) − r_θ(x, y_r) − m(r))), where m(r) is the absolute difference between the annotated rewards of the two texts, set to 0 for datasets with only rankings and normalized to (0, 1] (App. D.1).
- UltraRM training set: 749,702 comparison pairs, "with 340,025 from UltraFeedback, 198,556 from Stanford SHP, 92,858 from OpenAI Summarize, and 118,263 from Anthropic Helpful" (App. E.1).
- The RM trained with overall (critique) scores "discernably lags behind" the fine-grained variants on WebGPT (§3.1, Table 2).

## Human agreement check (§4.1, Table 4)
400 comparison pairs (100 each from UltraFeedback, AlpacaEval, Evol-Instruct, UltraChat test sets); 3 annotators (undergraduate and graduate students); win/tie/lose labels; "We include tie votes and the random agreement is 33%."

| Judge | A-1 | A-2 | A-3 | Average | Majority |
|---|---|---|---|---|---|
| GPT-4 | 59.2% | 60.8% | 59.1% | 59.7% | 68.6% |
| A-1 | – | 58.1% | 54.7% | 57.3% | 60.3% |
| A-2 | 58.1% | – | 55.4% | 58.1% | 63.3% |
| A-3 | 54.7% | 55.4% | – | 56.4% | 62.0% |

"Majority" is agreement with the majority vote of the other three judges.
