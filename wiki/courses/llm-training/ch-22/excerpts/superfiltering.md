---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/superfiltering.md
source_url: https://arxiv.org/abs/2402.00530
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the primary source; the library card was unverified and stated a warmup step and 'same or better' quality that the paper does not report)"
---

# Excerpt: Superfiltering — Weak-to-Strong Data Filtering for Fast Instruction-Tuning

**Paper:** Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, Ning Cheng, Tianyi Zhou (University of Maryland; Ping An Technology). arXiv:2402.00530 (v1 2024-02; ACL 2024). Checked 2026-09-15 against arXiv v2 (7 Jun 2024).

## Definitions (§2.1)
- PPL(y_i | x_i) = exp(−(1/N) Σ_j log p(y_{i,j} | x_i, y_{i,1..j−1})) (Eq. 1).
- IFD(y_i | x_i) = PPL(y_i | x_i) / PPL(y_i) (Eq. 2). A higher IFD indicates less instructional help and greater difficulty.

## Method (§3.3, §6.1)
- GPT-2 (124M) computes IFD for each sample directly, with no further training and no hold-out set.
- The top k% of samples with the highest IFD below 1 are selected.

## Table 1 — agreement with LLaMA2-7B
| Dataset | Filter | ρ (perplexity) | ρ (IFD) | overlap 5% / 10% / 15% |
|---|---|---|---|---|
| Alpaca | GPT-2 | 0.726 | 0.679 | 0.28 / 0.41 / 0.49 |
| Alpaca | GPT-NEO (1.3B) | 0.846 | 0.802 | 0.38 / 0.51 / 0.59 |
| Alpaca-GPT4 | GPT-2 | 0.730 | 0.788 | 0.24 / 0.40 / 0.51 |
| WizardLM 70k | GPT-2 | 0.763 | 0.802 | 0.42 / 0.54 / 0.61 |

## Table 2 — LLaMA2 fine-tuned on GPT-2-selected Alpaca data
| Base | Ratio (size) | Pairwise winning score | Open LLM avg | MMLU | AlpacaEval |
|---|---|---|---|---|---|
| LLaMA2-7B | 100% | 1.000 | 55.25 | 47.02 | 27.75 |
| LLaMA2-7B | 5% (2,600) | 1.133 | 55.67 | 45.21 | 33.04 |
| LLaMA2-7B | 15% (7,800) | 1.193 | 56.61 | 46.73 | – |
| LLaMA2-13B | 100% | 1.000 | 58.78 | 54.05 | 35.00 |
| LLaMA2-13B | 5% (2,600) | 1.174 | 60.96 | 55.79 | 45.71 |

## Table 3 — ablation (LLaMA2-7B, Alpaca; winning score vs full data at 5 / 10 / 15%)
Random 0.936 / 0.968 / 0.977; Diversity 0.927 / 0.977 / 0.982; Perplexity 0.261 / 0.569 / 0.610; IFD with GPT-2-large 1.165 / 1.046 / 1.193; IFD with LLaMA2-7B 1.303 / 1.330 / 1.294; Superfilter (GPT-2) 1.133 / 1.101 / 1.193.

## Table 4 — comparison with other selectors (Superfiltering's winning score vs each; filtering time)
Superfiltering 8 min. vs ChatGPT score 1.028 / 1.174 / 1.170 (120 min); vs reward-model score 1.280 / 1.096 / 1.147 (1,400 min); vs IFD with LLaMA2-7B 0.853 / 0.761 / 0.927 (161 min). Values below 1 mean Superfiltering data lost.

## Training settings (§4.2)
Vicuna prompt and code base; Adam; LR 2e-5 (7B), 1e-5 (13B); batch 128; 3 epochs; max length 2048; warmup rate 0.03.

## Human evaluation (§5.1)
100 WizardLM test instructions, 3 participants, LLaMA2-7B: Alpaca 5% vs 100% 50 wins / 18 ties / 32 losses; Alpaca-GPT4 5% vs 100% 49 / 5 / 46.
