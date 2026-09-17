---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2505.14652v5 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2505.14652
created_at: "2026-09-15"
---

# Excerpt: General-Reasoner: Advancing LLM Reasoning Across All Domains

- **Authors:** Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, Wenhu Chen (University of Waterloo, Vector Institute, TikTok, M-A-P)
- **Year:** 2025 (arXiv v1 2025-05; v5 2025-06-09)
- **Source type:** paper
- **Used in:** [[read]] §2.1, §3, Recipe

## Data and verifier
- WebInstruct-verified: about 230K reasoning questions re-crawled from web sources and filtered for verifiable answers, covering physics, chemistry, finance, social science and other disciplines (§1, §2).
- General-Verifier: a 1.5B generative verifier initialized from Qwen2.5-Math-1.5B and trained on Gemini-2.0-generated candidate answers and verification annotations. It reads (question, reference answer, extracted student answer) and emits a chain of thought plus a binary equivalence label (Eq. 3, §3). The stated motivation is that rule-based matching fails on short strings, LaTeX expressions, and other structured answers outside mathematics (§3).

## Diverse data versus math-only data (§5.1, Table 4; Zero-RL from base models)
| Backbone | Data | MMLU-Pro | GPQA | SuperGPQA | Math-Related |
|---|---|---|---|---|---|
| Qwen2.5-7B-Base | Full | 58.9 | 34.3 | 34.2 | 48.5 |
| Qwen2.5-7B-Base | Math only | 56.9 | 32.8 | 29.8 | 49.1 |
| Qwen2.5-14B-Base | Full | 66.6 | 43.4 | 39.5 | 53.9 |
| Qwen2.5-14B-Base | Math only | 64.8 | 38.9 | 35.6 | 48.6 |
At 7B, math-only training is 0.6 points higher on Math-Related and about 2 points lower on each general benchmark; at 14B the full-data model is higher on every column.

## Model-based versus rule-based verifier (§5.2, Table 5; Qwen3-4B-Base, 120 steps, identical conditions)
MMLU-Pro 60.1 vs 58.1; GPQA 39.4 vs 37.9; SuperGPQA 30.5 vs 30.1; Math-Related 50.4 vs 50.0. Figure 4 shows the rule-based run plateauing near 58% on MMLU-Pro at about step 60 while the model-based run keeps rising.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2505.14652 (v5, 2025-06-09): Abstract, §1, §3, §4.1, §5.1–5.2, Tables 2, 4, 5.
- Inconsistency inside the source: Table 2 reports GPQA-Diamond 38.8 for General-Reasoner-7B while Table 4 reports GPQA 34.3 for the same full-data model; the two tables label the column differently and the paper does not reconcile them. This excerpt uses Table 4 for the data-diversity comparison.
- Not reported: RL hyperparameters per run in the tables cited here; verifier false-positive rate on non-STEM answers.
