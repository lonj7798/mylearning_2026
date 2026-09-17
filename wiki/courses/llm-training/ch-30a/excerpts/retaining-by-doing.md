---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/retaining-by-doing.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2510.18874
created_at: "2026-09-15"
---

# Excerpt: Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting

**Authors:** Howard Chen, Noam Razin, Karthik Narasimhan, Danqi Chen (Princeton Language and Intelligence)
**Version read:** arXiv:2510.18874v3 (26 Jun 2026), which carries "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"; v1 October 2025.
**Status:** no library card existed for this slug on 2026-09-15; Table 1 and App. A.3 checked against the v3 PDF text. The full treatment is in ch-38a.

## Metrics (§2.1)
Target gain Δg := A(π_θT, T) − A(π_θ0, T). Non-target drop Δd := (1/M) Σ_j [A(π_θ0, T′_j) − A(π_θT, T′_j)]. Targets: IFEval, MMLU, Countdown; non-target tasks also include MATH, WildJailbreak, and WildGuardTest (§2.2).

## Methods compared (§2.2)
SFT on Llama-3.3-70B-Instruct responses; Self-SFT on responses from the initial model (correct ones only); RL with GRPO (reward 1 for correct, 0 otherwise).

## Table 1 (gain / drop, %)
| Model | Method | IFEval | MMLU | Countdown |
|---|---|---|---|---|
| Llama 3.1 8B Inst. | SFT | 25.2 / 27.8 | 11.1 / 38.5 | 25.5 / 36.4 |
| | REINFORCE | 17.8 / 7.7 | 8.6 / −0.1 | 7.5 / −0.8 |
| | GRPO | 18.4 / 3.4 | 14.6 / −0.2 | 60.4 / −0.5 |
| Qwen 2.5 7B Inst. | SFT | 24.2 / 5.6 | 9.4 / 14.6 | 10.4 / 29.2 |
| | REINFORCE | 5.7 / 2.9 | 6.4 / −0.6 | 11.9 / −0.1 |
| | GRPO | 17.0 / 0.2 | 8.4 / 0.2 | 29.2 / −0.3 |

## Statements used in ch-30a
- §2.3: "a high learning rate is typically required to reach high target performance for SFT, often at the cost of severe forgetting; a smaller learning rate reduces forgetting but fails to reach the same target performance even with more epochs" (Fig. 3: Self-SFT at LR 1e−5 and 1e−4, 2 and 10 epochs).
- §4.1: "non-regularized GRPO achieves a similar target task gain and non-target tasks drop tradeoff as KL-regularized GRPO across all considered models and datasets, except for Llama models trained on IFEval" (β = 0.05 vs β = 0.0).
- §4.2, Fig. 7: Iterative-SFT, which "uses data generated at the start of each round (i.e., epoch)", reaches the target accuracy of SFT "while only exhibiting mild to no forgetting."
- Related work: the "connection between KL divergence from the initial policy and forgetting does not always hold in our setting" (contrast with [[rls-razor]]).

## Settings (App. A.3)
AdamW; LR 1e−4 (Llama-3.2-1B-Instruct, Qwen-2.5-1.5B-Instruct) and 5e−6 (Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct); cosine schedule, warmup ratio 0.03; batch 128 (IFEval, MMLU) and 64 (Countdown); 2 epochs; SFT max sequence length 4096; Self-SFT keeps correct responses among 5 samples; GRPO KL-penalty coefficient 0.05, group size 5.

## How ch-30a uses it
§5.2 (learning rate trade-off), §5.7 (on-policy data), Recipe row, Negative-feedback section.
