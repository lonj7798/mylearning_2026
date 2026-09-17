<!-- scope: MMLU-Pro (Wang et al., NeurIPS 2024 Datasets and Benchmarks): a ten-option, reasoning-weighted replacement for MMLU, with prompt-robustness and CoT-versus-direct measurements
     deps: []
     see-also: [[ifeval]], [[xstest]]
-->

# MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
- **Core Insight:** MMLU-Pro contains 12,032 questions across 14 disciplines with ten answer options instead of four; GPT-4o scores 72.6% with 5-shot CoT, the GPT-4o−GPT-4-Turbo gap widens from about 1% on MMLU to 9% on MMLU-Pro, and the spread of a model's score across 24 prompt styles falls from 4–5% (maximum 10.98%) on MMLU to about 2% (maximum 3.74%) on MMLU-Pro (§1, §6.1, §6.3, Table 2).
- **Guideline:** When a held-out knowledge-and-reasoning benchmark must separate two checkpoints that both score near the MMLU ceiling, use MMLU-Pro with chain-of-thought prompting, because CoT raised GPT-4o by 19.1 points on MMLU-Pro and by only 1.5 points on MMLU, and lowered GPT-4-Turbo by 0.2 points on MMLU (§6.2, Table 3).
- **Authors:** Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, et al. (University of Waterloo, University of Toronto, Carnegie Mellon University)
- **Year:** 2024 (arXiv v1 2024-06; v6 2024-11-06; NeurIPS 2024 Datasets and Benchmarks)
- **URL:** https://arxiv.org/abs/2406.01574
- **Source type:** paper
- **Relevant topics:** held-out evaluation, multiple-choice benchmarks, prompt robustness, chain-of-thought, benchmark saturation

## Abstract
MMLU has begun to saturate, which makes differences between models hard to read. MMLU-Pro extends the knowledge-driven MMLU with reasoning-focused questions and expands the option set from four to ten, removing trivial and noisy items. Evaluating more than 50 models, the paper reports a large accuracy drop relative to MMLU, greater stability under prompt variation, and a larger benefit from chain-of-thought prompting.

## Key Contributions
- A 12,032-question, 14-discipline set with ten options per question (83% have ten, 17% fewer; mean 9.47 options), reducing the chance-correct rate (§1, §3).
- Two rounds of review: expert verification, then LLM-flagged error identification with targeted human re-verification (§1).
- Evaluation of more than 50 open and closed models with 5-shot CoT (§5, Table 2).
- Prompt-robustness measurement over 24 prompt styles (§6.3, Figure 5).
- Error analysis of 120 GPT-4o errors: 39% reasoning flaws, 35% missing domain knowledge, 12% calculation errors (§1, §6.4).

## Key Figures/Tables to Study
- **Table 2:** 5-shot CoT accuracy per model and discipline.
- **Table 3:** CoT minus direct-answer accuracy on MMLU and on MMLU-Pro.
- **Figure 5:** per-model score distribution over 24 prompts on MMLU and MMLU-Pro.

## Technical Details
- **Size and shape (§3).** 12,032 questions, 14 discipline subsets; ten options per question for 83% of items, average 9.47.
- **Default protocol (§5).** 5-shot chain-of-thought prompting, adapted from the original CoT prompts. Gemini-1.5-Pro and Gemini-1.5-Flash in Table 2 are evaluated 0-shot.
- **Scores (Table 2, overall, 5-shot CoT).** GPT-4o 72.6; Gemini-1.5-Pro 69.0; Claude-3-Opus 68.5; GPT-4-Turbo 63.7; Llama-3-70B-Instruct 56.2; Phi-3-medium-4k-instruct 55.7; Llama-3-70B 52.8; Llama-3-8B-Instruct 41.0; Mixtral-8x7B-Instruct-v0.1 43.3; Llama-2-70B 37.5.
- **Discrimination (§6.1).** On MMLU, Gemini-1.5-Flash, Llama-3-70B-Instruct, Phi-3-medium-4k-instruct and Qwen1.5-110B all fall in a 4% band (78–82%); MMLU-Pro spreads them over about 10%. The GPT-4o / Claude-3-Opus / GPT-4-Turbo spread grows from about 2% to about 9%.
- **CoT versus direct (§6.2, Table 3).** GPT-4o: +1.5 points from CoT on MMLU, +19.1 on MMLU-Pro. GPT-4-Turbo: −0.2 on MMLU, +15.3 on MMLU-Pro.
- **Prompt robustness (§6.3).** 24 different reasonable prompts per model. MMLU: score influence generally 4–5%, peak 10.98%. MMLU-Pro: generally about 2%, maximum 3.74%.
- **Limits (§7).** The multiple-choice format does not capture open-ended generation quality; the benchmark is text-only.

## Findings relevant to generality
- **Use as a retention gate.** The prompt-robustness numbers bound the measurement error of a before-and-after comparison: a change of about 2 points on MMLU-Pro is within the reported prompt-induced spread for a fixed model, so a retention check should use a fixed prompt and treat differences below that scale as unresolved.
- **CoT dependence.** Because CoT changes MMLU-Pro scores by up to 19 points for a single model, the generation configuration is part of the measurement and must be held identical before and after a training stage.

## Connections
- [[ifeval]] — the instruction-following half of a held-out retention suite.
- [[xstest]] — the over-refusal half.
- [[rls-razor]] — uses MMLU (not MMLU-Pro) among its prior-capability benchmarks.

## Verification
- Checked on 2026-09-15 against the cached primary text of https://arxiv.org/abs/2406.01574 (arXiv v6, 2024-11-06): Abstract, §1, §3, §5 Table 2, §6.1–§6.4, §7.
- Corrections to the previous card version: no previous card existed in the library for this slug.
- Not reported by the source: per-item variance or confidence intervals for the Table 2 scores; results for models released after mid-2024.
