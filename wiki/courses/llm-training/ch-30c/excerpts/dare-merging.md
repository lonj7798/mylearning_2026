---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (Yu et al.)"
source_url: https://arxiv.org/abs/2311.03099
created_at: "2026-09-15"
---

# Excerpt: DARE (Drop And REscale)

This excerpt stands in for the library card `dare-merging`, which did not exist when ch-30c was written.
Every number below was read in arXiv:2311.03099v3 at the stated locus.

- **Authors:** Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li (Alibaba Group)
- **Year:** arXiv v1 2023-11; v3 2024-06-13; ICML 2024 (PMLR 235)
- **Source type:** paper

## Method (§3.1-3.2, Eqs. 1-3)
- Delta parameters: δ^t = θ_SFT^t − θ_PRE.
- DARE: m^t ∼ Bernoulli(p); δ̃^t = (1 − m^t) ⊙ δ^t; δ̂^t = δ̃^t / (1 − p) (Eq. 1). p is the drop rate.
- For a linear layer, the expected output with rescaling factor γ is h^PRE + (1 − p)·γ·Δh; γ = 1/(1 − p) restores
  the expectation (§3.1).
- Merge with Task Arithmetic: θ_M = θ_PRE + λ·Σ_k (θ_DARE^tk − θ_PRE) (Eq. 3). DARE can precede Average Merging,
  Fisher Merging, RegMean, or TIES-Merging (§3.2).

## Results
- SFT delta parameters of the tested models are typically within 0.002; DARE removes 90% (sometimes 99%) without a
  large drop, and tolerance grows with model size: WizardMath-70B works at p = 0.99 while 7B and 13B fail (§4.2, Fig. 3).
- Dropping without rescaling (DropOnly) gives worse results as p grows; at p = 0.5 / 0.9 the layer-embedding cosine
  similarity of WizardMath-7B falls to about 0.85 / 0.68, while DARE stays above 0.95 at p = 0.9 (§4.4, Fig. 6).
- **13B decoder merges (Table 1; WizardLM-13B = LM, WizardMath-13B = Math, llama-2-13b-code-alpaca = Code; all from Llama-2-13b):**

| Merge | DARE | AlpacaEval | GSM8K | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|---|
| LM alone | – | 67.20 | 2.20 | 0.04 | 36.59 | 34.00 |
| Math alone | – | – | 64.22 | 14.02 | – | – |
| Code alone | – | – | – | – | 23.78 | 27.60 |
| Task Arithmetic LM & Math | No / Yes | 67.04 / 67.45 | 66.34 / 66.26 | 13.40 / 12.86 | 28.66 / 26.83 | 30.60 / 32.40 |
| Task Arithmetic LM & Math & Code | No / Yes | 69.03 / 69.28 | 58.45 / 56.48 | 9.88 / 10.16 | 18.29 / 23.17 | 29.80 / 31.60 |
| TIES LM & Math | No / Yes | 68.63 / 68.70 | 15.77 / 36.16 | 2.04 / 4.56 | 37.80 / 36.59 | 35.60 / 37.00 |
| TIES LM & Code | No / Yes | 63.63 / 67.15 | – | – | 0.0 / 18.29 | 0.0 / 26.40 |

- Scaling term searched in [0.5, 1.0]; TIES retain ratio in [0.5, 0.7, 0.9] (§4.3). The drop rate used for Table 1 is
  not stated in §4.3.
- Encoder merges (BERT-base, RoBERTa-base, GLUE): average change from adding DARE is 0.58, 0.36, 0.37, −0.03 and
  0.84 (%) for Average Merging, Task Arithmetic, Fisher Merging, RegMean and TIES (§4.3, Fig. 5).
- "An essential prerequisite for effective model merging is that each source model to be merged should be well
  fine-tuned" (§4.3), stated after the Code model underperformed WizardLM-13B on code.
- Two 7B merges (supermario v1: p = 0.3, λ = 0.8; v2: p = 0.5, λ = 0.5) reached Open LLM Leaderboard averages of
  74.85 and 75.49; v2 ranked first among 7B models as of 2024-01-28 (Table 2; App. A.5).

## When DARE fails (§4.6, Fig. 10)
- WizardCoder-Python-13B was fine-tuned from CodeLlama-13b-Python, which was trained on 500B additional code tokens.
  Using Llama-2-13b as the reference, delta magnitudes are often above 0.01 (vs within 0.0002 against
  CodeLlama-13b-Python), and dropping 10% of deltas takes HumanEval / MBPP pass@1 from 63.41 / 55.4 to 0.0 / 0.0.
- After continued pre-training, delta parameters "can rapidly reach around 0.03, making DARE infeasible" (§1).

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2311.03099v3 (PDF), §1-§5 and App. A.4-A.5.
