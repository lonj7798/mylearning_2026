---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Longpre et al. — "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning" (ICML 2023)
source_url: https://arxiv.org/abs/2301.13688
created_at: "2026-09-15"
note: "Library card for this slug was planned but not present on 2026-09-15; ch-04 uses only §3.1-3.2. Written from the primary PDF (arXiv v2, 2023-02-14)."
---

# Excerpt: The Flan Collection — mixed prompt settings

**Artifact:** arXiv:2301.13688 (v1 2023-01; read as v2). First author: Shayne Longpre (Google Research), et al.

## Method ablations (§3.1, Table 1)

Flan-T5 XL (3B) trained on Flan 2022 with four methods: mixture balancing, chain-of-thought tasks, mixed zero-shot and few-shot templates, and input inversion. Each row removes one method. Scores are zero-shot / few-shot.

| Model | Held-In | CoT | MMLU | BBH | BBH-CoT |
|---|---|---|---|---|---|
| T5-XL Flan 2022 | 73.8 / 74.8 | 35.8 / 34.1 | 50.3 / 52.4 | 26.2 / 39.3 | 33.9 / 35.2 |
| − CoT | 73.3 / 73.2 | 28.8 / 24.6 | 47.5 / 46.9 | 18.2 / 30.0 | 18.2 / 12.0 |
| − Input Inversion | 73.8 / 74.1 | 32.2 / 23.5 | 41.7 / 41.2 | 18.4 / 24.2 | 15.7 / 13.0 |
| − Mixture Balancing | 71.2 / 73.1 | 32.3 / 30.5 | 45.4 / 45.8 | 15.1 / 24.3 | 13.8 / 15.4 |
| − Few Shot Templates | 72.5 / 62.2 | 38.9 / 28.6 | 47.3 / 38.7 | 27.6 / 30.8 | 18.6 / 23.3 |

The authors summarize: chain-of-thought training helps chain-of-thought evaluation, input inversion helps held-out MMLU and BBH, few-shot template training helps few-shot evaluation, and mixture balancing helps all metrics (§3.1).

## Mixed prompt settings (§3.2, Fig. 3)

- Prior instruction-tuned models mostly used template sets for one prompt setting (zero-shot or few-shot). InstructGPT mixed settings without examining the choice.
- Training jointly with zero-shot and few-shot templates improved both held-in and held-out (MMLU) performance, including for 3B models. Adding as little as 5% few-shot templates improved zero-shot performance, and adding 10% or more zero-shot data improved few-shot performance; performance peaked between 10% and 90% few-shot data, above training with one setting only.

## Limits

T5 encoder-decoder models; NLP task collections rather than chat conversations; the templates varied are instruction and exemplar formats, not chat control tokens.
