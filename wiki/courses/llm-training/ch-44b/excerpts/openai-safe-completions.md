---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2508.09224v1 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2508.09224
created_at: "2026-09-15"
---

# Excerpt: From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training

- **Authors:** Yuan Yuan, Tina Sriskandarajah, Anna-Luisa Brakman, Alec Helyar, Alex Beutel, Andrea Vallone, Saachi Jain (OpenAI)
- **Year:** 2025 (arXiv v1 2025-08-12)
- **Source type:** official technical report
- **Used in:** [[read]] §6, Negative samples and negative feedback

## Method (§2)
Built on deliberative alignment: an SFT stage over ideal (CoT, answer) pairs, then an RL stage whose reward is the product of two reward-model scores for the prompt and the final response:

  r_i = h_i · s_i

- s_i ∈ [0, 1]: adherence to the category's content-policy specification. "Perfectly compliant responses receive s_i = 1, severe or definitive violations receive s_i = 0, and intermediate values reflect borderline, low-severity violations."
- h_i ∈ [0, 1]: helpfulness, from one RM that scores direct helpfulness (fulfilling the stated task) and indirect helpfulness (constructive alternatives, transparent refusals) and "is high if the model scores highly on either type".
Under a low s_i the model can still raise reward through indirect helpfulness, which is the stated mechanism for replacing hard refusals. The illicit-wrongdoing policy was rewritten around "meaningful facilitation" rather than the wording of the request (§2.3).

## Evaluation (§3)
- Two comparisons: a controlled ablation (CE-Refusal vs CE-SafeComplete, same architecture, pretraining corpus, and remaining post-training recipe, o4-mini-like) and a production pair (o3 vs GPT-5 Thinking). About 9,000 anonymized safety-related production prompts, four completions each, graded by a reasoning model for Safety (binary) and Helpfulness (1–4, scored only on safe responses), stratified by intent (benign, dual-use, malicious).
- Controlled experiment: safety improves on dual-use prompts and stays similar on benign and malicious prompts; helpfulness given a safe output rises "by more than 1.0 point on the 1–4 scale" on malicious prompts. Production pair: gpt5-r improves safety over o3 on dual-use and malicious prompts by 9 and 10 percentage points. Bar values are in the figure images (Fig. 4).
- Harm severity among unsafe responses (§3.2.2, Fig. 5): mass shifts from Moderate/High to Low/Negligible. On 620 biorisk prompts, the share of unsafe responses graded high or moderate harm is 3.7% + 11.0% = 14.7% for gpt5-r against 42.7% for o3 (§3.3).

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2508.09224 (v1, 2025-08-12): Abstract, §1, §2.1–2.3, §3.1–3.3, Table 1.
- Not reported: RL algorithm and hyperparameters, RM sizes and training data, numeric values behind Figures 4–7, effects on non-safety capabilities.
