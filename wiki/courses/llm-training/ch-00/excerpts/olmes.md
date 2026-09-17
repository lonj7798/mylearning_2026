---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source arXiv:2406.08446v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2406.08446
created_at: "2026-09-15"
---

# Excerpt: OLMES — A Standard for Language Model Evaluations

**Paper:** Gu, Tafjord, Kuehl, Haddad, Dodge, Hajishirzi (Allen Institute for AI; University of Washington). arXiv v1 2024-06, read at v2 (2025-02-11). Source type: paper. Code and prompts: https://github.com/allenai/olmes.

## Problem (§1, Table 1)
- Published ARC-Challenge scores for one model differ across references because shots, curated examples, formulation, and normalization differ. Llama3-8B: 60.2 on the Hugging Face Open LLM Leaderboard (25-shot, cloze) and 78.6 in the Llama 3 model card (25-shot, multiple choice); OLMES reports 79.3 (§3.4, Table 1).

## Definitions (§2.1, §3.3)
- MCF (multiple-choice formulation): options with letter labels in the prompt; the model is scored on the label.
- CF (completion/cloze formulation): each answer string is scored separately by its probability.
- CF normalizations: none ln P(a_i|q); token ln P(a_i|q)/num_tokens(a_i); character ln P(a_i|q)/num_characters(a_i); pmi ln[P(a_i|q)/P(a_i|u)] with u = "Answer:".

## Findings (§3.4, Fig. 1-2, Table 3-4)
- OLMo-7B-0424 on MMLU: CF gives signal early; MCF is near random until about 400B training tokens and then gives the stronger signal (Fig. 1).
- ARC-Challenge across 15 base models: the weakest 8 score near random with MCF but above random with CF; Llama3-70B scores 93.7 (MCF) vs 69.0 (CF) (§3.4, Fig. 2, Table 4).
- Normalization per task is within 0.0-1.1 points of the per-model best ("diff oracle", Table 3).

## The standard (§3.1-3.5, §4, Table 2)
- Test split if labels are public, else validation; sample 1,000 instances when a dataset has more than 1,500 (Random(1234).sample).
- "Question: <question>" prefix and "Answer:" suffix, with task-specific exceptions (PIQA "Goal:", HellaSwag and WinoGrande changes); " A." labels with a leading space.
- Fixed, manually curated 5-shot examples from the training set.
- Normalization: pmi for ARC-Challenge, CommonsenseQA, OpenBookQA; character for ARC-Easy, HellaSwag, PIQA, Social IQa, MMLU; none for BoolQ, WinoGrande.
- Evaluate with both MCF and CF and report the better score.
- MMLU macro average over 57 subjects; leading space counted in character normalization; inputs restricted to 2,048 tokens; default model precision.

## Limits (Limitations)
- Designed for multiple-choice tasks on base models during training and for final base-model comparison; generative tasks, chain-of-thought, and chat-model message splitting are future work.

## Verification
- Read on 2026-09-15 against arXiv:2406.08446v2 PDF text (§1-6, Tables 1-4, Limitations).
