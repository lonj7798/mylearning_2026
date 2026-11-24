<!-- scope: General-Reasoner (arXiv:2505.14652) — WebInstruct-verified 230K multi-domain RL set with Gemini-extracted short answers; 1.5B generative General-Verifier replacing rule matching; zero-RL across domains; verifier and data-diversity ablations
     deps: [[verifier-pitfalls]], [[grpo]]
     see-also: [[math-verify]], [[master-rm-one-token]], [[transferability-of-llm-reasoning]], [[rubrics-as-rewards]], [[rl-design-guide]], [[diversity-axes]]
-->

# General-Reasoner: Advancing LLM Reasoning Across All Domains
- **Core Insight:** Outside math, rule matching rejects most correct answers: on 50K answers that Gemini-2.0-Flash judged correct, a rule-based verifier agreed 22.2% of the time and the 1.5B generative General-Verifier 78.7% (§5.3); with this verifier and 230K multi-domain questions, zero-RL on Qwen3-14B reached 56.1 on GPQA-D and 70.3 on MMLU-Pro (Table 2).
- **Guideline:** When RL prompts include non-math domains with short but non-canonical answers (units, matrices, expressions, multiple choice), grade with a small generative verifier that sees the question, reference and response, and filter prompts to those where 8 sampled solutions are neither all right nor all wrong, because the model verifier beat the rule verifier on every reported metric in a matched run (MMLU-Pro 60.1 vs 58.1 at 120 steps, rule verifier plateauing near 58% from step 60; Table 5, Fig. 4), and mixed-domain data beat math-only on SuperGPQA by 4.4 (7B) and 3.9 (14B) points (Table 4).
- **Authors:** Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, Wenhu Chen (University of Waterloo, Vector Institute, TikTok Singapore, M-A-P)
- **Year:** 2025 (arXiv:2505.14652; v5 read, 9 Jun 2025)
- **URL:** https://arxiv.org/abs/2505.14652
- **Source type:** paper
- **Relevant topics:** multi-domain RL data, reference-answer extraction, model-based verifiers, length-difference penalty, domain diversity

## Summary
The authors re-crawl WebInstruct source pages, extract question / short-answer pairs with Gemini, label type, subject and difficulty, filter by sampled-solution pass rate, and train with GRPO from base models (Zero RL) using a learned verifier as the reward.

## Key Contributions
- WebInstruct-verified: about 230K questions across physics, chemistry, social science, finance and others, with multiple-choice, numerical and matrix answers (§3.1, Figs. 2–3).
- General-Verifier: 1.5B, initialized from Qwen2.5-Math-1.5B, writes a short chain of thought then outputs true/false (§3.2).
- Ablations of verifier type and data diversity (Tables 4–5).

## Key Figures/Tables to Study
- Fig. 2 (data pipeline), Table 2 (general benchmarks), Table 3 (math), Table 4 (data diversity), Table 5 / Fig. 4 (verifier ablation).

## Technical Details
**Data (§3.1).** ~5M WebInstruct instructions → re-crawl and drop items without clearly human-written answers → Gemini-1.5-Pro extracts ~1M questions with verifiable short answers → Gemini-2.0-Flash labels answer type, subject, difficulty; math easier than university level removed → Gemini-2.0-Flash generates 8 candidate solutions; drop questions where all 8 fail or all 8 pass → ~230K.

**Reward (§4.1).** Extraction failure: −0.5. Verified correct: 1 minus a length-difference penalty 0.05 × min(10, |len(ground truth) − len(answer)|).

**Training (§4.1, App. A.5).** GRPO Zero-RL in verl on 4 × 8 H100; up to 700 steps (Qwen2.5) and 400 (Qwen3-Base).

**Results.** MMLU-Pro 7B: 58.9 vs Qwen2.5-7B-Instruct 57.0; 14B: 66.6 vs 62.7. Qwen3-14B: GPQA-D 56.1, TheoremQA 54.4 (GPT-4o 50.0, 43.6); MMLU-Pro 70.3 vs Qwen3-14B-Instruct non-think 70.9 (Tables 2–3).

**Data-diversity ablation (Table 4, full vs math-only).** 7B: MMLU-Pro 58.9 vs 56.9; SuperGPQA 34.2 vs 29.8; math-related 48.5 vs 49.1. 14B: MMLU-Pro 66.6 vs 64.8; SuperGPQA 39.5 vs 35.6; math-related 53.9 vs 48.6.

## Verification
- Read arXiv HTML v5 on 2026-11-24. Precision/recall or human agreement of General-Verifier are not reported in the portion read; agreement is with Gemini-2.0-Flash only.
- Table 4 lists GPQA 34.3 for the 7B model while Table 2 lists 38.8 GPQA-D; this card avoids that number.
- [[master-rm-one-token]] measured General-Verifier at 9.7% average and 87.0% worst-case FPR against master-key responses (its Table 1); [[verifier-pitfalls]] measured general-verifier at 22.1 attack success under adversarial prefixes.
