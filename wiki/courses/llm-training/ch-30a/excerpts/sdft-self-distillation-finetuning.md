---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sdft-self-distillation-finetuning.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2402.13669
created_at: "2026-09-15"
---

# Excerpt: Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning

**Authors:** Zhaorui Yang, Tianyu Pang, Haozhe Feng, Han Wang, Wei Chen, Minfeng Zhu, Qian Liu (Zhejiang University; Sea AI Lab; Tencent)
**Version read:** arXiv:2402.13669v2 (28 May 2024); v1 February 2024.
**Status:** no library card existed for this slug on 2026-09-15; equations and tables checked against the v2 PDF text.

## Method (§3.2-3.3, Eq. 2-5)
- Vanilla fine-tuning: L_FT(θ) = −log f_θ(y^t | c^t, x^t).
- Rewrite with the seed model: ỹ ~ f_θ(y | c^t, x^t, y^t), using the template "Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your own response." (Fig. 3).
- Selection: ỹ′ = ỹ if Extract(ỹ) = y^t, otherwise y^t (for math, the final answer is compared).
- Loss: L_SDFT(θ) = −log f_θ(ỹ′ | c^t, x^t).

## Settings (§4.1, App. A)
Seed model Llama-2-7b-chat; LoRA on query and value, r = 8; LR 1×10⁻⁴ with cosine decay to zero; batch size 8. 2,000 sampled examples for 2 epochs (Alpaca, Dolly, MagiCoder); 20,000 OpenHermes examples for 2 epochs; full train sets for GSM8K and OpenFunctions (5 epochs) and LIMA (2 epochs). Safety: AdvBench keyword matching (raw) and with adversarial suffixes (jailbreak); helpfulness: AlpacaEval with GPT-4.

## Table 1 (downstream; OpenFunctions / GSM8K / HumanEval)
| Method | Fine-tuned on | OpenFunctions | GSM8K | HumanEval |
|---|---|---|---|---|
| Seed LM | — | 19.6 | 29.4 | 13.4 |
| Vanilla FT | OpenFunctions | 34.8 | 21.5 | 9.8 |
| SDFT | OpenFunctions | 36.6 | 29.1 | 15.2 |
| Vanilla FT | GSM8K | 17.9 | 31.9 | 12.2 |
| SDFT | GSM8K | 17.9 | 34.4 | 14.6 |

## Table 2 (vanilla FT → SDFT; seed LM 99.81 raw safe, 88.85 jailbreak safe, 66.04 win rate)
| Fine-tuned on | Raw safe rate | Jailbreak safe rate | AlpacaEval win rate |
|---|---|---|---|
| OpenFunctions | 98.27 → 99.23 | 87.31 → 94.42 | 35.49 → 67.66 |
| GSM8K | 82.12 → 87.12 | 54.81 → 65.58 | 23.38 → 66.73 |
| MagiCoder | 96.73 → 97.88 | 83.65 → 88.65 | 76.52 → 76.09 |

## Other statements
- §5.3, Table 5 (one fine-tuning dataset per setting; vanilla FT → SDFT): full fine-tuning of Llama-2-7b-chat on GSM8K, OpenFunctions 5.36 → 16.07, win rate 23.04 → 61.19; LoRA on Llama-2-13b-chat on GSM8K, OpenFunctions 19.64 → 24.11, win rate 40.27 → 75.93; LoRA on Llama-3-8B-Instruct on OpenFunctions, GSM8K 77.79 → 79.45 (seed 81.58), jailbreak safe rate 79.81 → 96.15. SDFT is higher than vanilla FT in every column of Table 5.
- §4.5: on MMLU, TruthfulQA, ARC, HellaSwag, Winogrande, "models' capabilities in general knowledge are relatively unaffected" by either method.
- §5.1: a larger mix ratio of distilled samples raises BLEU-4, ROUGE-L, and embedding similarity to seed outputs and lowers parameter shift, with better benchmark results.

## How ch-30a uses it
§5.4 (self-distillation as a forgetting control), Recipe row, Negative-feedback section (fallback to the original response when the rewrite's answer does not match).
