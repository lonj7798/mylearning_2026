<!-- scope: chapter-local excerpt for ch-22; no library card existed at the 2026-09 revision
     source: Ivison, Zhang, Brahman, Koh, Dasigi. "Large-Scale Data Selection for Instruction Tuning". arXiv:2503.01807v2 (v1 2025-03)
     checked: 2026-09-15 against https://arxiv.org/abs/2503.01807 (v2, 19 Jun 2025)
-->

# Excerpt: Large-Scale Data Selection for Instruction Tuning

- **Core result.** Selecting from pools of up to 5.8M samples and evaluating on 7 tasks, many proposed selectors fall below random selection, and 4 of 7 methods score lower with a larger pool; RDS+ (weighted mean pool of the base model's last hidden states, cosine similarity to query samples) is best in every setting tested (Abstract, §4).
- **Source type:** paper. **Reliability:** single study, one run per non-random setting; random baselines are means of 3 runs (§3.3).

## Setup (§3)
- Pools: Tülu 2 unfiltered (5,817,792 samples after exact-match deduplication, dominated by FLAN and Open Orca) and Tülu 3 unfiltered (4,783,043) (§3.2, Fig. 2).
- Methods: random (unbalanced and source-balanced), Length (longest samples), Top-PPL and Mid-PPL (base-model loss), IFD (Cherry LLM code), LESS (Princeton code), Embed (NV-Embed-v2, GTR-base), RDS+ (§3).
- Aggregation: round-robin over query points (single task) and over tasks (multi-task), which worked better than averaging scores (§3.1, App. D).
- Training: full fine-tuning of Llama 2 7B, 2 epochs, batch 1 with 128 gradient-accumulation steps, LR 2e-5 (1e-5 for 70B), linear warmup 3%, linear cooldown (§3.3).
- FLOPs: 6N per training token and 2N per inference token (Kaplan et al.) (§3.3).
- Evaluations and selection targets: MMLU, GSM8K, BBH, TydiQA, HumanEval-Codex, SQuAD, AlpacaEval (§3.3).

## Table 1 — single-task selection of 10K samples (average over 7 tasks)
| Method | 200K pool | 5.8M pool |
|---|---|---|
| Random (unbal.) | 40.9 | 40.7 |
| Random (bal.) | 42.2 | 41.1 |
| LESS | 44.2 | not run (compute) |
| Embed (NV) | 42.7 | 41.6 |
| Embed (GTR) | 42.1 | 46.4 |
| Top-PPL | 34.2 | 30.4 |
| Mid-PPL | 39.4 | 38.9 |
| IFD | 38.7 | 35.8 |
| Length | 43.5 | 17.8 |
| RDS+ | 46.4 | 50.5 |

Selected per-task values (5.8M pool): IFD SQuAD 46.6, AlpacaEval 44.6; Length GSM8K 2.4, TydiQA 17.8, SQuAD 0.2, AlpacaEval 3.1; RDS+ GSM8K 33.9, AlpacaEval 53.5. At 200K: LESS HumanEval-Codex 19.6 vs random 27.0; IFD SQuAD 57.9 vs random 83.5.

## Table 2 — multi-task selection of 326K samples from the 5.8M Tülu 2 pool (average)
Rand. (unbal.) 44.5; Rand. (bal.) 47.5; Top-PPL 36.6; Mid-PPL 41.3; Embed (GTR) 48.0; Embed (NV) 45.3; IFD 35.7; Length 47.6; Tülu 2 49.5; RDS+ 50.9; RDS+ with WildChat queries 47.8; RDS+ with Arena Hard queries 50.4.

## Table 3 — Tülu 3 pool, Llama 3.1 8B, 939K selected (average)
Random (unbal.) 74.6; Random (bal.) 74.7; Tülu 3 SFT 73.3; RDS+ 76.1; RDS+ with Arena Hard queries 67.3.

## Scaling and compute (§4.3, Figs. 3–4)
- RDS+ beats balanced random at every selection size from 10K to 2.5M; at 326K (about 6% of the pool) it matches training on all data (50.8 vs 50.9).
- Counting selection FLOPs, RDS+ is more compute-efficient than random only at about 326K selected samples and above.

## Analysis (§5)
- PPL selects more ShareGPT data and IFD more FLAN data than RDS+ or random; the authors connect this to IFD's relatively high AlpacaEval and low scores elsewhere.
- Smaller selection models (including other families) still work for RDS+; larger models gave no gain over Llama 2 7B (App. F).

## Limits (Limitations)
- Two data pools only. Any method needing forward passes over the pool scales in cost with pool size.
