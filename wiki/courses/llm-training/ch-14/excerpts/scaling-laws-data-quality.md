---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/scaling-laws-data-quality.md (card has no verification section and no numbers; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2510.03313
primary_version: arXiv:2510.03313v2 (2026-02-23; v1 2025-10)
created_at: "2026-09-15"
revised: 2026-09 (generality revision; replaces the 2026-04 excerpt, which gave a formula with a quality-dependent irreducible loss E(q) that the paper does not use)
---

# Excerpt: Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining

Verbatim quotations and table values used by ch-14 `read.md`, with loci. Authors: Anirudh Subramanyam, Yuxin Chen, Robert L. Grossman (University of Chicago). Checked against the v2 PDF on 2026-09-15. Source type: paper.

## The law (§1, §6)
> "The law predicts loss as L(N, D, Q) = A/N^α + B/(D^β Q^γ) + E, capturing the interplay between model size, data volume, and data quality."
> "we introduce a single dimensionless parameter Q ∈ (0, 1] characterizing the usable information in a corpus. A value of Q = 1 represents fully clean and representative data, while smaller Q values reflect increasing corruption or redundancy."
- In this form E does not depend on Q. §5.5: the plots "suggest that the additive terms in our proposed scaling law, A/N^α + E, indeed do not vary with data quality Q."

## Corruption-rate estimator (§3.1)
> "if we have a dataset consisting of D tokens and 10% are corrupted, then we would say that the CR = 10% and the data quality Q is 90%."

## Relation between γ and effective data (§4.1, App. A.3)
- Definition 3: "D_eff := D g(Q)".
- App. A.3: "ρ(Q) ≈ c Q^γ0 as Q → 1, so ρ(Q)^−β ≈ c^−β Q^−βγ0. Absorbing c^−β into B and setting γ = βγ0, we obtain L_D ≈ B / (D^β Q^γ)."

## Experimental setting (§5.1-§5.3)
- Causal language modeling: "a 8L Llama 3 ... model with a hidden size of 512 and a context length of 2048"; C4 (en) subsets of "100M, 1B and 10B tokens" trained "for a single epoch"; 63 runs.
- Machine translation: "a 8L GPT Neo model with a hidden size of 1024 with approximately ∼ 133M params"; Paracrawl v8 English-German, 500K, 1M, and 2M sentence pairs; 63 runs.
- Quality is set by synthetic noise: "if 25% of all samples are perturbed, then the quality of that dataset is considered to be 0.75." For CLM, "we randomly swap 50% of all non-special tokens with valid non-special tokens from the tokenizer vocabulary." Quality levels: "Q = {1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5}".
- Model size is fixed within each task; App. E.3 fits "L ≈ B/(D^β Q^γ) + E (assuming N is fixed or large enough)".

## Table 2 (estimated parameters)
| Task | Method | B | β | γ | E |
|---|---|---|---|---|---|
| NMT | Least Squares | 166.568727 | 0.262933 | 0.185135 | 0.146998 |
| NMT | Huber | 139.602744 | 0.250067 | 0.173161 | 0.066539 |
| CLM | Least Squares | 1428.225931 | 0.395142 | 0.388678 | 3.439888 |
| CLM | Huber | 1441.505289 | 0.395859 | 0.400657 | 3.439047 |

## Authors' interpretation (§5.5)
> "the estimated exponents for data quality, γ̂, are significantly less than one: γ̂ ≈ 0.173 for NMT and γ̂ ≈ 0.401 for CLM (with Huber estimation). This indicates that the effective dataset size decays sublinearly with quality, i.e., models are more robust to moderate corruption than predicted by simple effective sample-size theories"
> "Table 3 shows us that using our pre-trained models to evaluate loss on unseen data also follows a scaling law similar to our fit on in distribution data"

Note for readers (derived, not stated by the paper): with γ = βγ0, the clean-token equivalent of D tokens at quality Q is D · Q^(γ/β). With the CLM Huber row, γ/β = 0.400657 / 0.395859 = 1.012; with the NMT Huber row, 0.173161 / 0.250067 = 0.692.
