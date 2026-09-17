---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source arXiv:2405.10938v3 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2405.10938
created_at: "2026-09-15"
---

# Excerpt: Observational Scaling Laws and the Predictability of Language Model Performance

**Paper:** Ruan, Maddison, Hashimoto (Stanford University; University of Toronto; Vector Institute). arXiv v1 2024-05, read at v3 (2024-10-01); NeurIPS 2024. Source type: paper.

## Model (§3.1, Eq. 3-5)
- For model m with benchmark errors B_{i,m}, a capability vector S_m ∈ R^K satisfies σ⁻¹(E_m) ≈ βᵀS_m + α (downstream error), S_m ≈ θ_f log(C_m) + ν_f (compute within family f), and B_{i,m} ≈ γ_iᵀS_m (benchmarks). Model families differ only in how efficiently they convert compute C into capabilities.

## Low-dimensional capability space (§3.2, Fig. 2)
- 77 pretrained base models from 21 families; benchmarks MMLU, ARC-C, HellaSwag, Winogrande, GSM8K, HumanEval, TruthfulQA, XWinograd, taken from standardized leaderboards or the LM Eval Harness; missing values under 1% imputed by PCA.
- The top 3 principal components explain about 97% of the variance; the first alone nearly 80%.
- PC-1 is a weighted average of all metrics ("general capability"); PC-2 emphasizes math and coding ("reasoning"); PC-3 mainly programming.
- Related work cited: a single factor explains 85% of Open LLM Leaderboard and GLUE performance (Ilić); three factors explain 82% of HELM variation (Burnell et al.) (§2).

## Compute relation (§3.3, Fig. 3)
- Within model families with controlled recipes, PC-1 correlates linearly with log training FLOPs (C ≈ 6ND) with R² > 0.9.

## Uses and limits (§1, §7)
- Predicts several emergent phenomena, agent performance of GPT-4 on AgentBench and AgentBoard from weaker models, and gains from chain-of-thought and self-consistency (§1).
- Limits: applies to post-training scaling analyses, not pretraining in the way compute scaling laws do; focuses on few-shot and basic prompting; does not account for benchmark contamination or heterogeneity within model families (§7).

## Verification
- Read on 2026-09-15 against arXiv:2405.10938v3 PDF text (Abstract, §1-3.3, §7).
