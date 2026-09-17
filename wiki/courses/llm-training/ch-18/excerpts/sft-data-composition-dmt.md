---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition (Dong, Yuan, Lu, Li, Xue, Liu, et al., Alibaba Group)"
source_url: https://arxiv.org/abs/2310.05492
created_at: "2026-09-15"
---

# Excerpt: SFT data composition across math, code, and general chat (DMT)

Chapter-local excerpt written because no library card for this paper existed when ch-18 was revised.
Every number below was read in arXiv:2310.05492v4 (2024-06-07) on 2026-09-15.

- **Authors:** Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, et al. (Alibaba Group)
- **Year:** 2023 (arXiv v1 2023-10; v4 2024-06)
- **Source type:** paper

## Setup
- Base models: LLaMA 7B, 13B, 33B (§3.1).
- Data (App. A, Table 2): GSM8K RFT (7.5K questions, 110,142 responses), Code Alpaca (20,022), ShareGPT (86,060).
  Subsets at k = 1/4, 1/16, 1/64, 1/256 of each; for example k = 1/256 gives 430 / 78 / 336 samples.
- Evaluation: GSM8K (greedy, maj@1), HumanEval (Pass@1), MT-Bench (§3.1, App. B).
- Training: 3 epochs, batch size 128, peak LR 2e-5 with 3% warmup, final epoch evaluated (App. C).

## Findings ch-18 uses
1. **Different scaling per ability (§3.2, Fig. 2).** Math and code keep improving with more data; general
   (MT-Bench) ability appears with about 1K samples (1/256 to 1/64 of ShareGPT) and then improves slowly.
2. **Mixing helps at low data and conflicts at high data (§3.3, Fig. 3).** Mixed-source training is better than
   single-source at 1/256 and worse at the full amount for LLaMA-7B, with a crossover between 1/64 and 1/16.
3. **Amount matters more than ratio (Abstract; §3.4, Fig. 4)** for math and general ability; code is sensitive to
   the ratio, which the authors attribute to code content inside ShareGPT (Interpretation).
4. **Strategies (§3.5, Table 1)** — GSM8K / HumanEval / MT-Bench:

| Strategy | LLaMA-7B | LLaMA-13B | LLaMA-33B |
|---|---|---|---|
| General only | 11.10 / 10.42 / 5.88 | 14.02 / 16.40 / 6.13 | 26.06 / 24.30 / 6.63 |
| Math only | 49.10 / 6.71 / 2.53 | 51.40 / 12.8 / 2.54 | 57.91 / 15.5 / 3.18 |
| Code only | 4.51 / 18.40 / 4.30 | 5.15 / 17.1 / 3.53 | 6.06 / 26.82 / 4.18 |
| Multi-task (all mixed) | 47.53 / 14.63 / 5.76 | 50.94 / 19.50 / 5.73 | 56.69 / 18.9 / 6.07 |
| Sequential (code → math → general) | 31.39 / 15.85 / 5.72 | 39.12 / 20.12 / 5.93 | 47.27 / 24.80 / 6.73 |
| Mixed sequential (code+math → general) | 32.60 / 15.24 / 6.02 | 40.48 / 18.30 / 5.93 | 44.24 / 24.4 / 6.43 |
| DMT (k = 1/256) | 41.92 / 17.68 / 6.08 | 46.47 / 19.50 / 6.03 | 56.36 / 25.00 / 6.73 |

5. **DMT (§3.5).** Stage 1: SFT on the full specialized (code + math) data. Stage 2: SFT on general data plus a
   fraction k of the specialized data, which reduces forgetting of math and code relative to mixed sequential
   training while keeping MT-Bench.

## Limits relevant to ch-18
- One general benchmark (MT-Bench) and one benchmark per specialized skill; out-of-distribution checks on MBPP and
  MATH are in App. F.
- Single runs per cell; no variance reported in Table 1.
