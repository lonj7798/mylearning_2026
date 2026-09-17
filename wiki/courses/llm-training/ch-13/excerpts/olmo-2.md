---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md (library card not verified on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2501.00656
primary_version: arXiv:2501.00656v3 (2025-10-08); v1 2025-01
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: 2 OLMo 2 Furious — mid-training mixture (Dolmino Mix 1124)

Values used by ch-13 `read.md` §4, §5, §7, and Recipe, read in the v3 PDF text on 2026-09-15. Organization: Allen Institute for AI.

## Pretraining mix (Table 4)
OLMo 2 Mix 1124: DCLM-Baseline 3.71T tokens; StarCoder (filtered) 83.0B; peS2o 58.6B; arXiv 20.8B; OpenWebMath 12.2B; Algebraic Stack 11.8B; Wikipedia & Wikibooks 3.7B; total 3.90T. The text says "over 95% derived from web data" (§2.4.1).

## Dolmino source pool (Table 5)
- High-quality subset: DCLM-Baseline filtered (FastText top 7%, FineWeb ≥ 2) 752B; FLAN (decontaminated) 17.0B; peS2o 58.6B; Wikipedia & Wikibooks 3.7B; Stack Exchange curated Q&A 1.26B; total 832.6B.
- Math mix: TuluMath 230M; Dolmino SynthMath 28.7M; TinyGSM-MIND 6.48B; MathCoder2 Synth Books 3.87B; Metamath (OWM-filtered) 84.2M; CodeSearchNet (OWM-filtered) 1.78M; GSM8K train split 2.74M; total 10.7B.

## Final mixes (Table 13, "Mix %" column sums to 100)
| Source | 50B mix | 100B mix | 300B mix |
|---|---|---|---|
| Filtered DCLM | 47.2 | 50.2 | 51.9 |
| Decontam. FLAN | 16.6 | 16.7 | 11.3 |
| StackExchange Q&A | 2.45 | 2.47 | 1.68 |
| peS2o | 5.85 | 9.52 | 19.4 |
| Wikipedia/Wikibooks | 7.11 | 3.57 | 4.86 |
| Dolmino Math | 20.8 | 17.5 | 10.8 |

> "We train OLMo 2 7B on the 50B mix. To account for the larger batch size (Section §2.3), we use the 100B mix for OLMo 2 13B, ensuring the same number of steps during learning rate anneal." (§4.5)

The 100B mix repeats StackExchange Q&A and math twice; the 300B mix repeats them four times, FLAN twice, and Wiki four times (§4.5).

## Candidate mid-training mixes, 7B from a 4T-token checkpoint, 50B tokens (Table 11)
| Mix | OLMES (MCF) | OLMES-Gen | MMLU (MCF) | GSM* |
|---|---|---|---|---|
| pretrain checkpoint | 69.6 | 63.2 | 59.8 | 28.5 |
| PT Mix | 74.0 | 64.5 | 61.8 | 27.0 |
| Web FW72 | 75.2 | 63.8 | 63.1 | 28.5 |
| Web FW72 + Ins | 74.2 | 64.1 | 63.0 | 46.0 |
| Web FW72 + Math | 75.7 | 69.7 | 62.3 | 52.0 |
| Web FW72 + Math + Ins | 75.7 | 70.2 | 63.1 | 46.5 |

GSM* is a random sample of 200 GSM8K questions used for development.

## Microanneals (§4.4.2)
Experiment 1: "the 35/65 mixture yields a GSM* of 63.5, and the 10/90 mixture yields a GSM* of 61. This suggests that it is not strictly necessary to have a large proportion of domain-specific data in the annealing mixture, just that domain-specific data is present." In total, 19 microanneals used 130B tokens. (The Experiment 2 text gives 61 for one copy of math data while its table gives 63.5; ch-13 does not use Experiment 2.)

## Held-out split and results (footnote 6, Table 9)
> "GSM8k (Cobbe et al., 2021) was only partially held-out, as we subsampled 200 of 1319 GSM8k examples for mid-training data development when we noticed poor math capabilities after pretraining; we call this dev set GSM*. The remaining 1119 GSM8k examples we reserve as held-out and report final performance on them only."

Table 9, OLMo 2 7B, pretraining → pretraining & mid-training: average 53.0 → 62.9; GSM8K 24.1 → 67.5; MMLU Pro (held-out) 27.4 → 31.0; TriviaQA (held-out) 74.6 → 78.0. The 7B mid-trained checkpoint averages three runs on 50B Dolmino tokens (Table 9 caption).
