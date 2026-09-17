---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2404.13076
created_at: "2026-09-15"
---

# Excerpt: LLM Evaluators Recognize and Favor Their Own Generations

**Authors:** Arjun Panickssery, Samuel R. Bowman, Shi Feng (MATS; New York University; Anthropic, PBC)
**Year:** 2024 (arXiv v1 2024-04-15)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v1 PDF text.

## Why ch-49 uses it

[[judge-llm-bias]] observed a self-preference effect but states its own study cannot determine whether it exists, because a controlled design is hard. This paper supplies the controlled design, so it is the source ch-49 cites for the effect rather than the observation.

## Setup (§2.1)

Summarization on XSUM and CNN/DailyMail, each example a news article plus a human-written summary. Evaluators and generators: Llama-2-7b-chat, GPT-3.5, GPT-4. Summaries are sampled at temperature zero; punctuation and initial capitalization are standardized to match the human summaries so formatting is not a cue.

Two properties are measured separately:
- **Self-preference** — the evaluator scores its own output higher than others rate it.
- **Self-recognition** — the evaluator can identify which output it produced.

Measurement is pairwise (choose which text is higher quality, or which one it wrote) with confidence read from normalized option-token probabilities, and individually (§2.2).

## Findings (§1)

1. All three evaluators show self-preference on both datasets.
2. All three exceed 50% self-recognition accuracy without fine-tuning; GPT-4 reaches 73.5% at distinguishing itself from two other LLMs and humans.
3. Fine-tuning on 500 examples raises GPT-3.5 and Llama 2 above 90% self-recognition accuracy.
4. Fine-tuning models to raise or lower self-recognition moves self-preference along a linear trend (Fig. 1).

## Controls (§3.3–§3.4)

- **Inverse direction.** They test whether the evaluator recognizes its own outputs because those outputs are objectively better, and report that only 22% of evaluator-domain pairs show a preference above 0.51 for the fine-tuned model's generations. Before fine-tuning, the Kendall τ between correctly recognizing its own summary in a pair and preferring that summary is 0.41, and this correlation does not change meaningfully as self-recognition ability rises.
- **Confounders.** GPT-3.5 and Llama are additionally fine-tuned on three properties unrelated to self-recognition — length, vowel count, Flesch–Kincaid readability — and on always answering "1" and answering randomly. All eight fine-tuned models per dataset fall on the same linear trend as the self-recognition ones (Fig. 1).

The authors state that correlation plus these controls is evidence for, not proof of, the causal hypothesis; validating it would require mechanistic tools that do not exist for LLMs (§3).

## Order bias (§2.5)

Evaluators reverse their pairwise preference when the option order is reversed at rates of 25% (GPT-4), 58% (GPT-3.5), and 89% (Llama 2), averaged across tasks and datasets (Fig. 5). Their adjustment is to average the logit-based confidence scores across the two orderings; an alternative reported is to discard pairs whose preference reverses.

## Connections

[[judge-llm-bias]] (the original observation and its stated caveat), [[arena-hard-benchbuilder]] (measures self-bias at the benchmark level and mitigates with a judge ensemble), [[meta-rewarding-lm]] (a pipeline in which the evaluator and the evaluatee are the same model).
