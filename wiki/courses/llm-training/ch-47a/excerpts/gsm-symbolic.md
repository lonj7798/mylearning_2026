---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models (arXiv:2410.05229v2, v1 2024-10-07; ICLR 2025)"
source_url: https://arxiv.org/abs/2410.05229
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug gsm-symbolic). Values read from the v2 PDF on 2026-09-15. Templates and generated data: github.com/apple/ml-gsm-symbolic."
---

# Excerpt: GSM-Symbolic, GSM-P1/P2, and GSM-NoOp

**Authors:** Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar (Apple; Washington State University).

## Construction and setup (§3, §3.2)

- 100 GSM8K test questions are rewritten as symbolic templates with named variables, sampling domains, and consistency conditions (Fig. 1). Numeric ranges are kept close to GSM8K's so the test is of reasoning, not arithmetic width (§3.1, App. A.6).
- 50 samples per template → 5,000 items per benchmark, that is 50 datasets of 100 items each (§3.2).
- Variants: **Symbolic-M1** (one clause removed), **Symbolic** (names and values changed), **Symbolic-P1** and **-P2** (one and two extra clauses added), **Symbolic-NoOp** (a seemingly relevant but inconsequential clause added) (§3.2, §4.3, §4.4).
- More than 20 open models from 2B to 27B plus GPT-4o-mini, GPT-4o, o1-mini, o1-preview; about 500 evaluations in total. Unless stated otherwise: 8-shot chain-of-thought prompting with greedy decoding (§3.2).

## Distributional results (§4.1)

- All models show non-negligible variance across the 50 sets. Gemma2-9B's best-to-worst gap exceeds 12%; Phi-3.5-mini's is about 15%.
- The score on the 100 original GSM8K questions is more than one standard deviation from the centre of the model's GSM-Symbolic distribution for **21 of 25 models**, usually on the high side. The authors name contamination as one explanation (§4.1).
- Models are more robust to changed proper names than to changed numeric values (§4.2). Average accuracy falls and variance grows as clauses are added (§4.3).

## Table 1 (8-shot accuracy, %, ± is the standard deviation across the 50 sets)

| Model | GSM8K (full) | GSM8K (100) | Symbolic | Symbolic-P1 | Symbolic-P2 | Symbolic-NoOp |
|---|---|---|---|---|---|---|
| Phi-3-mini-128k-instruct | 83.7 | 85.0 | 80.7 (± 2.94) | 63.4 | 37.5 | 18.0 (± 3.83) |
| Phi-3-medium-128k-instruct | 87.3 | 89.0 | 82.5 (± 2.86) | 75.8 | 53.1 | 29.4 (± 4.18) |
| Phi-3.5-mini-instruct | 84.9 | 88.0 | 82.1 (± 3.38) | 64.8 | 44.8 | 22.4 (± 4.03) |
| Mathstral-7b-v0.1 | 80.1 | 80.0 | 74.0 (± 3.49) | 57.4 | 35.5 | 20.4 (± 3.58) |
| Gemma2-27b-it | 89.7 | 92.0 | 88.3 (± 2.56) | 80.7 | 63.4 | 30.0 (± 3.39) |
| GPT-4o-mini | 94.2 | 95.0 | 91.7 (± 2.02) | 81.1 | 72.4 | 54.1 (± 3.85) |
| GPT-4o | 95.2 | 95.0 | 94.9 (± 1.87) | 93.9 | 88.0 | 63.1 (± 4.53) |
| o1-mini | 95.1 | 93.0 | 94.5 (± 1.58) | 94.3 | 89.1 | 66.0 (± 4.60) |
| o1-preview | 94.9 | 96.0 | 92.7 (± 1.82) | 95.4 | 94.0 | 77.4 (± 3.84) |

Caution when quoting other rows: in the extracted table several base/instruct pairs (Gemma-7b and -7b-it, Gemma2-2b and -2b-it, Llama3-8b and -8b-instruct) carry identical Symbolic and NoOp values, which the surrounding text does not explain. The rows above are unique and were cross-checked against the per-model histograms in §4.1–4.2.

## GSM-NoOp (§4.4)

- NoOp templates add a clause that is topically related but carries no operation, for example "five of the kiwis were smaller than average" in a counting problem (Fig. 7). Models convert such clauses into arithmetic operations.
- §4.4 states a drop of "over 65%" for Phi-3-mini and "significant declines" for o1-preview; the abstract states drops "up to 65%" across state-of-the-art models. Fig. 8a prints per-model relative drops (o1-preview −17.5, GPT-4o −32.0, Phi-3-medium-128k −57.8, Llama3-8b-instruct −57.4, Phi-2 −44.9). For GPT-4o and o1-preview these are within about 1.5 points of the ratio implied by Table 1; for Phi-3-medium-128k the ratio implied by Table 1 (29.4 from 82.5, −64.4%) is 6.6 points larger than the printed bar, and the paper does not state which baseline each bar uses.
- Shot-source ablation (Fig. 8b, Phi-3-medium-128k): 87.3 with GSM8K questions and GSM8K shots; 82.5 Symbolic; 29.4 NoOp with GSM8K shots; 30.2 NoOp with 8 Symbolic shots of the same question; 22.6 NoOp with 8 NoOp shots of other questions. Providing the reasoning chain for the same question in the shots does not recover the loss.

## Used in

ch-47a §3 (perturbation and no-op variants), §7, Recipe.
