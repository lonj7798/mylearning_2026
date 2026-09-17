---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: primary source arXiv:2410.16713v4 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2410.16713
created_at: "2026-09-15"
---

# Excerpt: Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World

**Paper:** Joshua Kazdan, Rylan Schaeffer, Apratim Dey, Matthias Gerstgrasser, Rafael Rafailov, David Donoho, Sanmi Koyejo (Stanford). arXiv v1 2024-10; v4 2025-03-17 read. Source type: paper.

## Workflows (Abstract, §2–3)
- **Replace:** real data used only at the first iteration. Collapses in all three settings studied: multivariate Gaussian estimation, kernel density estimation, and supervised fine-tuning of language models (§2).
- **Accumulate:** real plus all synthetic data kept and trained on. Test loss on real data stays bounded in all three settings, although the real-data fraction goes to zero (Abstract).
- **Accumulate-subsample (fixed compute):** the pool accumulates but each iteration trains on a fixed-size subsample. Across five settings its test loss lies between replace and accumulate, and it typically plateaus while replace typically diverges (§3, Fig. 4).

## Language-model SFT setting (§2.3, Fig. 3)
- Gemma 2 (2B, 9B, 27B) fine-tuned on NVIDIA HelpSteer2; replace trains on ~12.5K examples each iteration, accumulate on ~12.5K × t.
- Replace degrades cross-entropy on real test data over iterations; accumulate does not (Fig. 3).

## Real-data cardinality versus proportion (§4, Fig. 5)
- Gemma 2 2B fine-tuned on HelpSteer2; 100k completions generated, those over 512 tokens removed, leaving 55,000 synthetic samples; datasets with varied real and synthetic mixes.
- R² 0.59 for the transformed number of real data and 0.34 for the real-data proportion; both terms significant (F-test p-values 6.9×10⁻²⁵ and 4.6×10⁻²⁵).
- With 1024 or fewer real examples, a small amount of synthetic data (optimum near 1024 synthetic examples) lowered test loss.
- With more than 1024 real examples, adding synthetic data almost always raised test loss at a fixed real count; real-only datasets were more valuable than datasets with ten times more real data mixed with synthetic data.

## Stated limits (§5)
- The experiments do not filter data for quality, which the authors call pessimistic relative to practice.

## Verification
- Read on 2026-09-15 against arXiv:2410.16713v4 PDF text (Abstract, §2.3, §3, §4, §5).
