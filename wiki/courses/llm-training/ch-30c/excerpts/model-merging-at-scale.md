---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "What Matters for Model Merging at Scale? (Yadav et al.)"
source_url: https://arxiv.org/abs/2410.03617
created_at: "2026-09-15"
---

# Excerpt: Model merging at scale (PaLM-2, 1B-64B)

This excerpt stands in for the library card `model-merging-at-scale`, which did not exist when ch-30c was written.
Every number below was read in arXiv:2410.03617v1 at the stated locus.

- **Authors:** Prateek Yadav, Tu Vu, Jonathan Lai, Alexandra Chronopoulou, Manaal Faruqui, Mohit Bansal, et al. (UNC; Google; Virginia Tech)
- **Year:** arXiv v1 2024-10-04 (no venue on the PDF)
- **Source type:** paper

## Setup (§3, App. A-B)
- Base models: PaLM-2 at 1B, 8B, 24B, 64B, and PaLM-2-IT, the same models after 60,000 further steps on Flan-v2 with
  the T0 tasks removed (App. B).
- Experts: full fine-tuning on each of 8 held-in T0 task categories (16 datasets), default 2,000 steps, LR 3e-5,
  dropout 0.05 (App. B). 64 experts in total (§3).
- Merge methods: Averaging (mean of θ_i, no base), Task Arithmetic, TIES, Dare-TIES (§2.1).
- Grid: 2 base types × 4 sizes × 4 methods × {2, 4, 6, 8} experts × 3 seeds for expert selection = 384 merges (§3).
- Held-out: 4 task categories, 7 datasets (sentence completion, NLI, coreference, word sense) (§3).
- Metric: held-in scores normalized by the corresponding expert; held-out scores normalized by the base model;
  averaged within category, across categories, then over seeds (§3).
- Multitask baseline: one model trained on the mixture of all eight held-in categories (§4.3).
- Merge hyperparameters (λ, trim ratio, drop rate) are not reported.

## Selected values (App. C, Tables 1-4; Task Arithmetic unless noted)
| Base, size | Experts | Held-in merge | Held-in multitask | Held-out merge | Held-out multitask |
|---|---|---|---|---|---|
| PaLM-2-IT 1B | 2 → 8 | 0.91 → 0.86 | 0.97 → 0.96 | 1.03 → 1.05 | 1.11 |
| PaLM-2-IT 8B | 2 → 8 | 0.95 → 0.88 | 0.96 | 1.06 → 1.03 | 1.12 |
| PaLM-2-IT 24B | 2 → 8 | 0.96 → 0.92 | 0.99 → 0.98 | 1.05 → 1.18 | 1.18 |
| PaLM-2-IT 64B | 2 → 8 | 1.00 → 0.93 | 0.99 | 1.00 → 1.09 | 1.05 |
| PaLM-2 1B | 2 → 8 | 0.66 → 0.39 | 0.88 → 0.87 | 1.01 → 1.07 | 1.10 |
| PaLM-2 8B | 2 → 8 | 0.68 → 0.42 | 1.06 | 1.06 → 1.00 | 1.62 |
| PaLM-2 64B | 2 → 8 | 0.80 → 0.67 | 0.97 → 0.96 | 1.29 → 1.35 | 1.73 → 1.72 |

- PaLM-2-IT 64B, 8 experts, held-out: Average 1.09, Dare-TIES 1.09, TIES 1.10 (Table 2); held-in: all four methods 0.93 (Table 1).
- PaLM-2-IT 64B, 6 experts, held-out: Task Arithmetic 1.06 vs multitask 1.05 (Table 2).

## Authors' findings (Abstract; §4.1-4.6)
1. Experts from the instruction-tuned base merge better than experts from the pre-trained base (§4.1, Fig. 3).
2. Larger models merge better on held-in tasks (§4.2, Fig. 4).
3. Merged models exceed their base model on held-out tasks in most settings; with PaLM-2-IT the text states the merge
   beats multitask training "when combining more than 6 large instruction-tuned expert models (over 24B)" (§4.3).
4. Larger models can merge more experts (§4.4). Methods perform similarly for 64B PaLM-2-IT (§4.5, Fig. 7).
5. Recommendation: build experts from the best available base model; merged models usually trail the experts (§4.6).

## Discrepancies found while reading
- §4.4 text: "merging eight 8B PaLM-2 models decreases performance from 0.66 to 0.39" and "PaLM-2-IT ... from 0.91 to
  0.86". These values match the 1B Task Arithmetic column of Tables 3 and 1; the 8B column gives 0.68 → 0.42 and
  0.95 → 0.88 (derived by comparing text and tables).
- In Table 2 only the 64B column has merges above multitask on held-out (1.06-1.10 vs 1.05); at 24B the best merge
  equals multitask (1.18). For the pre-trained PaLM-2 base, multitask held-out scores (1.10-1.73) exceed every merge (Table 4).

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2410.03617v1 (PDF), §1-§6 and App. A-C.
