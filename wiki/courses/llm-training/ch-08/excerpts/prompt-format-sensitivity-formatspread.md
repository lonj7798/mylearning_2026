---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2310.11324
source_url: https://arxiv.org/abs/2310.11324
created_at: "2026-09-17"
revised: 2026-09 (generality revision)
---

# Excerpt: Sclar, Choi, Tsvetkov, Suhr 2023, "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design"

**Artifact:** arXiv:2310.11324 (v1 2023-10; v2 2024-07-01), ICLR 2024. Method name: FORMATSPREAD.
**Read on:** 2026-09-17 from the PDF text cached at
`scratchpad/sources/prompt-format-sensitivity-formatspread.txt`.
**Source type:** paper. **No library card exists for this artifact yet**; ch-08 cites this excerpt.

---

## Definition used by ch-08

*Spread* for a task and model: `max_i m(p_i, D) − min_i m(p_i, D)`, where `p_i` ranges over prompt formats
that preserve the meaning of the task instruction, `D` is the evaluation set, and `m` the metric (§3.2, L233-235).
Formats differ only in separators, casing, spacing and item wording, generated from a grammar (§3.1).

## Numbers

- Accuracy varies by up to **76 points** between equivalent formats for LLaMA-2-13B, and by about
  **10 accuracy points on average across 50+ tasks** and several models (abstract; §1, L52-56).
- With **10 randomly sampled formats per task** across 53 tasks at 1- and 5-shot, the **median spread is
  7.5 accuracy points**; **20% of tasks have a spread of at least 15 points in every LLaMA-2 setting** and at
  least 9 points in every Falcon setting; several tasks exceed 70 points (§4.2, L318-326).
- Because only 10 formats are sampled, these are **lower bounds**: about 17% of tasks are expected to gain at
  least 5 points of spread when moving from 10 to 20 sampled formats (§4.5, L591-593, Fig. 8).
- Sensitivity is **not removed by model size, by more few-shot examples, or by instruction tuning**
  (§4.2, L326-329, Figs. 2a-2c and Fig. 11 for Llama-2-70B).
- Model comparisons reverse under format change: assuming model M beats M' by at least d = 0.02 under format
  p, LLaMA-2-13B and -70B reverse with probability 0.141 under a different format p' (§4.2, L332-336, Fig. 4).
- On an API-gated model the authors measure a spread up to 56 points with a **median of 6.4 points for GPT-3.5
  across 320 formats** (§1, L122-124).

## Conditions and limits

The evaluation tasks are mostly classification tasks scored by accuracy; §B.2 extends the analysis to a
selection of non-classification tasks. The formats are restricted to those the grammar judges equivalent to the
task's original format (§3.1, L203). FORMATSPREAD searches formats as a bandit problem under an evaluation
budget; with 51,200 evaluations Thompson sampling lands within 1 accuracy point of the true sample spread,
naive sampling within 4, UCB within 11 (§4.5, L596-599).

## Why ch-08 uses it

A single-format evaluation of an SFT checkpoint measures the checkpoint and the format together. The lab's
template-robustness requirement (score under at least two formats and report both) follows the paper's own
recommendation to report a range of performance across plausible formats rather than one number (abstract).

## Connections

- [[trl-sft-trainer]] — `chat_template_path` and `assistant_only_loss` decide the single format the model is
  trained under.
- [[system-prompt-diversity]] — library card on varying system prompts during training.
