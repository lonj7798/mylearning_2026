---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: arXiv:2308.01825v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2308.01825
created_at: "2026-09-15"
---

# Excerpt: Scaling Relationship on Learning Mathematical Reasoning with Large Language Models

- **Authors:** Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, et al. (Alibaba DAMO Academy)
- **Year:** 2023 (arXiv v1 2023-08; v2 2023-09-13; preprint)
- **Source type:** paper
- **Used in:** ch-31 §1.4, §4, Recipe, Generalization lens

## What the paper does

The paper studies how pre-training loss, supervised data amount, and self-generated data amount affect GSM8K accuracy of LLaMA and LLaMA 2 models after supervised fine-tuning (Abstract). It proposes Rejection sampling Fine-Tuning (RFT): an SFT model samples reasoning paths for the training questions, correct paths are kept, and the pretrained model is fine-tuned on the human data plus the kept paths (§3.3).

## Technical details with loci

- **Sampling.** k = 100 candidate paths per training question at temperature 0.7 (§3.3; App. A.3).
- **Filter.** Paths with a wrong answer or a wrong calculation (checked by Python evaluation) are removed (§3.3).
- **Deduplication.** Equations are extracted from `<<equation>>` markers; one path is kept per distinct equation list, and among paths with the same list the one with the largest summed Levenshtein distance to the other kept paths is chosen (App. A.3, Algorithm 1). Different element order or equation order counts as a different path (§3.3).
- **Fine-tuning target.** The augmented set D′ = D ∪ {q_i, r_ij, a_i} is used to fine-tune the pretrained model, not the SFT model (§3.3).
- **SFT settings (A.1).** 3 epochs, batch size 128, peak learning rate 2e-5 with 3% warmup, evaluation at the final epoch, greedy decoding for maj1@1 and temperature 0.7 for maj1@100. RFT rows in Table 5 list 3 epochs; an RFT-specific learning rate is not printed.
- **Metric.** maj1@1 is greedy-decoding accuracy on the GSM8K test split; maj1@100 is majority-vote accuracy over 100 samples (§3.3).

## Results with loci

| Model | SFT | RFT k = 100 | RFT-U13B | Correct paths / question (k = 100) | Distinct paths / question (k = 100) |
|---|---|---|---|---|---|
| LLaMA-7B | 35.9 | 41.7 | 49.3 | 53.3 | 5.25 |
| LLaMA 2-7B | 41.6 | 47.5 | 50.3 | 60.8 | 5.19 |
| LLaMA-13B | 43.0 | 49.1 | 52.1 | 62.5 | 5.26 |
| LLaMA 2-13B | 50.0 | 54.8 | 55.4 | 71.6 | 5.29 |
| LLaMA-33B | 54.6 | 54.5 | 56.5 | 88.7 | 2.78 |

Sources: Table 1 (SFT, RFT k = 100, paths), Table 3 and Table 5 (U13B). RFT-U33B for 33B is 57.9 (Table 5).

- **Distinct paths versus k (Table 2, LLaMA-7B).** k = 1: 1.17; k = 3: 1.44; k = 12: 2.20; k = 50: 3.94; k = 100: 5.25. Pooled U13B: 12.84; U33B: 13.65.
- **k sweep (§3.3, Fig. 4, Table 5).** At k = 3, RFT exceeds SFT by about 2 points; larger k mostly helps, with diminishing gain per doubling of k.
- **Deduplication (§3.3).** k = 100 with and without deduplication give similar accuracy; deduplication is better for 3 of 4 models and needs less training time. LLaMA-7B: 41.7 deduplicated (~47K examples) vs 43.6 without deduplication (400K examples) (Table 5).
- **33B case (§3.3).** The 33B SFT model generates the most correct but the fewest distinct paths; the authors state that it overfits the training set. At temperature 1.0 it gives 82.4 correct and 4.77 distinct paths per question.
- **Pooling (§3.3, Fig. 6).** Models up to 13B each contribute about 15% unique paths to D′_U33B; only 6.5% come exclusively from the 33B SFT model. For 65B, training on D′_U13B did not improve over SFT (§3.3; Table 5: SFT 59.3, RFT-U13B 59.0).
- **Diversity at inference (§4.1, Fig. 7).** Models trained on U13B produce more unique calculation processes per test question than SFT models.
- **Compute (§4.2).** The authors estimate SFT and RFT cost about 1 × 10⁻⁵ and 1 × 10⁻⁴ of pre-training compute; RFT inference GPU hours in Table 4 are computed for 7,473 training questions × 100 samples on A100 80GB GPUs.

## Limits stated by the authors

No RFT for 65B/70B in the main analysis, no math-corpus pre-training, and no fitted scaling law because several numbers are estimated (§7). All accuracy numbers are on GSM8K; no out-of-domain task is reported.

## Verification

- Checked on 2026-09-15 against: https://arxiv.org/abs/2308.01825 (v2, 2023-09-13), full text including App. A, B, and Tables 1–6.
- Not reported by the source: RFT-specific learning rate and batch size; out-of-domain evaluations; seeds.
