---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2412.16339v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2412.16339
created_at: "2026-09-15"
---

# Excerpt: Deliberative Alignment: Reasoning Enables Safer Language Models

- **Authors:** Melody Y. Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Helyar, et al. (OpenAI)
- **Year:** 2024 (arXiv v1 2024-12; v2 2025-01-08)
- **Source type:** official technical report
- **Used in:** [[read]] §6, Generalization lens

## Method (§2)
- SFT stage: a helpfulness-only reasoning model is shown the category-specific safety specification in the system prompt and generates (prompt, CoT, output) samples; a judge reasoning model G_RM, also given the spec, filters completions; the spec is then stripped from the prompt so the trained model must recall it. No human-written chains of thought or answers are used.
- RL stage: for safety-relevant prompts, the same judge G_RM with access to the policies supplies additional reward. "While G_RM receives CoT during SFT data filtering, the CoT is hidden from G_RM during RL. We avoid applying direct optimization pressure on the CoT during RL to enable the underlying model to reduce the chance of encouraging deceptive CoTs." This RL reward was added for o1 and o3-mini.

## Results
- Table 1 (o1 vs GPT-4o): StrongREJECT goodness@0.1 0.88 vs 0.37; XSTest not-overrefuse 0.93 vs 0.88; safe completion (self-harm) style adherence 0.92 vs 0.04; hard-refusal style adherence 0.79 vs 0.72; Challenging Refusal Evaluation not-unsafe 0.9 vs 0.8.
- Stage ablation (§4.1, Fig. 14): "Safety in SFT only" and "Safety in RL only" give intermediate results between no safety training and safety in both; in this ablation setup safety training also increases overrefusals. Providing the summarized spec at inference time without safety training learns less safety behavior than training with it, and adheres worst to response-style guidelines. Figure values are in the figure image.
- Policy retrieval (§4.2, Table 2): the fraction of prompts where the CoT cites the correct policy category is 0.75 (hard refusal), 0.91 (safe completion), 0.54 (comply) for the trained model, against 0.27 / 0.21 / 0.09 for the untrained baseline.
- Out-of-distribution generalization (§4.3, Table 3; o1-mini variants): encoding-jailbreak accuracy 0.65 ± 0.06 (no safety training), 0.97 ± 0.02 (safety data with all non-English and encoded examples removed), 0.95 ± 0.03 (full data); multilingual jailbreak accuracy 0.44 ± 0.01, 0.69 ± 0.01, 0.68 ± 0.01.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2412.16339 (v2, 2025-01-08): Abstract, §1, §2.3.3, §2.4, §3.1, §4.1–4.3, Tables 1–3.
- Not reported: model sizes, amount of safety data, RL algorithm and hyperparameters, numeric values behind Figures 2 and 14.
