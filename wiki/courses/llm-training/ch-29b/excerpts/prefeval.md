---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prefeval.md (library card not present on 2026-09-15; content taken from the primary source)
source_url: https://arxiv.org/abs/2502.09597
primary_version: arXiv:2502.09597v1 (2025-02; ICLR 2025)
created_at: "2026-09-15"
---

# Excerpt: Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs (PrefEval)

Authors: Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, Kaixiang Lin (Amazon AGI; UCLA; University of Minnesota). Checked against the v1 PDF on 2026-09-15.

## Construction (§2)
- 1,000 preference-query pairs × 3 preference forms = 3,000 pairs over 20 topics, "manually curated with the assistance of GPT-4, Claude 3 Sonnet, and Claude 3.5 Sonnet" (§2.2).
- Forms (§2.3): explicit (one user sentence); implicit choice-based (two-turn dialogue in which the user accepts or rejects offered options); implicit persona-driven (4-8 turns in which the preference appears once and is not the main topic).
- The query "is constructed such that a generic, non-personalized response would likely violate the previously stated preference" (§2.1).
- Distractor turns: multi-session turns sampled from LMSYS-Chat-1M, inserted between preference and query, up to 100k tokens (§2.4).
- Generation task judged by Claude 3 Sonnet with four binary checks (violate, acknowledge, hallucinate, helpful) aggregated into four error types; 200 manually checked evaluations had a 5% error rate. A four-option classification task is also provided (§2.5, Table 1).

## Results
- Zero-shot, explicit preferences: accuracy "drops steeply from approximately 80% to below 30% as the number of conversation turns increases to merely 5", and falls close to zero from 30 to 300 turns (§3.2, Fig. 3).
- Table 2 (travel restaurant topic; zero-shot / reminder): at 10 turns (~3k tokens) Claude-3.5-Sonnet 0.07 / 0.45, Gemini-1.5-Pro 0.07 / 0.91, GPT-o1-preview 0.50 / 0.98; at 300 turns (~103k tokens) 0.02 / 0.02, 0.09 / 0.05, 0.14 / 0.98.
- SFT (§3.7, Fig. 10): Mistral-7B fine-tuned on 80% of topics and evaluated on the remaining 20% unseen topics. Targets: Mistral-7B's own responses generated with the Reminder prompt, without contextual turns; during training 0, 5, or 10 contextual turns are inserted between preference and query (2, 7, or 12 turns). The fine-tuned model in the zero-shot setting surpasses "the previous best-performing method (RAG)"; training with 10 inserted turns "generalizes to 70-turn contexts much more effectively than when trained with fewer contexts". Attention to preference regions increases by up to 4.97%. Exact accuracy values are shown only in the figure.

## Limits
- Topics are daily-life recommendation domains; no broad capability benchmarks are reported for the SFT model.
