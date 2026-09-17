---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/retaining-by-doing.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2510.18874
created_at: "2026-09-15"
---

# Excerpt: Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting

**Authors:** Howard Chen, Noam Razin, Karthik Narasimhan, Danqi Chen (Princeton Language and Intelligence).
**Version read:** arXiv:2510.18874v3 (26 Jun 2026); the PDF carries "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026". v1 is October 2025.
**Status:** no library card existed for this slug on 2026-09-15. A shorter excerpt for ch-30a exists at `ch-30a/excerpts/retaining-by-doing.md`; this one adds §3, §4.1 and App. A.5–A.6.

## Metrics (§2.1)
Target gain Δg := A(π_θT, T) − A(π_θ0, T); non-target drop Δd := (1/M) Σ_j [A(π_θ0, T′_j) − A(π_θT, T′_j)], where A(π, T) is accuracy of policy π on task T, π_θ0 the initial policy and π_θT the policy after T optimization steps.

## Methods compared (§2.2)
SFT on responses from Llama-3.3-70B-Instruct; Self-SFT on responses from the initial model (5 samples per prompt, incorrect ones filtered by the reward function); RL with GRPO (reward 1 for correct, 0 otherwise). Targets: IFEval, MMLU, Countdown. Non-target evaluations also include MATH, WildJailbreak and WildGuardTest.

## Table 1 — gain / drop (%)
| Model | Method | IFEval | MMLU | Countdown |
|---|---|---|---|---|
| Llama 3.1 8B Inst. | SFT | 25.2 / 27.8 | 11.1 / 38.5 | 25.5 / 36.4 |
| | REINFORCE | 17.8 / 7.7 | 8.6 / −0.1 | 7.5 / −0.8 |
| | GRPO | 18.4 / 3.4 | 14.6 / −0.2 | 60.4 / −0.5 |
| Qwen 2.5 7B Inst. | SFT | 24.2 / 5.6 | 9.4 / 14.6 | 10.4 / 29.2 |
| | REINFORCE | 5.7 / 2.9 | 6.4 / −0.6 | 11.9 / −0.1 |
| | GRPO | 17.0 / 0.2 | 8.4 / 0.2 | 29.2 / −0.3 |

Caption: "The advantage estimate of GRPO is not responsible for its robustness to forgetting." REINFORCE "lags behind GRPO in optimizing the target task accuracy, yet maintains a similar low level of forgetting" (§4.1).

## KL ablation (§4.1, Fig. 6)
"non-regularized GRPO achieves a similar target task gain and non-target tasks drop tradeoff as KL-regularized GRPO across all considered models and datasets, except for Llama models trained on IFEval" (β = 0.05 against β = 0.0).

## Degree of on-policyness (§4.2, Fig. 7)
Self-SFT, which generates data only from the initial policy, "suffers from severe forgetting". Iterative-SFT, which "uses data generated at the start of each round (i.e., epoch)", reaches target accuracy higher than or comparable to SFT "while only exhibiting mild to no forgetting". SFT on traces produced during an RL run also reduces forgetting (App. A.4.1, Fig. 10).

## Mechanism (§1, §3)
LM post-training is modelled as a mixture of an "old" mode (prior knowledge) and a "new" mode (the target task). Minimizing forward KL (SFT) "first stretches the new mode … and then moves probability mass from the old mode to cover the target, leading to forgetting"; minimizing reverse KL (RL) "maintains the shape of the old mode and covers the target distribution by shifting the new mode". If the initial policy is uni-modal, SFT can forget less than RL (§3.2); with a multi-modal initial policy, which the authors take to be the practical case for LMs, mode-seeking RL forgets less (§3.3).

## KL and forgetting (App. A.5, Table 2)
KL[π_θ0 ‖ π_θ] estimated on 100 examples from the evaluation set (units not stated).

| Model | Method | IFEval drop / KL | MMLU drop / KL | Countdown drop / KL |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | Self-SFT | 6.9 / 52.4 | 34.6 / 39.3 | 25.3 / 796.7 |
| | SFT | 26.2 / 61.1 | 28.9 / 48.4 | 24.2 / 1254.7 |
| | GRPO | 1.6 / 2.6 | 0.3 / 4.1 | −0.6 / 70.6 |
| Qwen-2.5-1.5B-Instruct | Self-SFT | 3.0 / 25.0 | 14.0 / 3.0 | 19.5 / 896.1 |
| | SFT | 6.2 / 47.8 | 11.9 / 9.2 | 29.5 / 846.6 |
| | GRPO | 0.6 / 1.5 | 0.5 / 0.4 | 0.9 / 34.9 |

"the Pearson correlation between KL divergence and non-target tasks drop across all models, methods, and datasets is 0.52. Yet, when comparing Self-SFT and SFT, the relation between KL divergence and forgetting is less monotonic — a larger KL does not necessarily imply a higher degree of forgetting."

## AlpacaEval after training on each target (App. A.6, Table 3)
| Model | Initial | SFT (IFEval / MMLU / Countdown) | RL (IFEval / MMLU / Countdown) |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 41.6 | 23.5 / 11.6 / 0.0 | 43.1 / 41.2 / 42.0 |
| Qwen2.5-7B-Instruct | 36.5 | 22.5 / 19.9 / 0.4 | 35.2 / 36.0 / 34.6 |

## Settings (App. A.3)
AdamW; LR 1e−4 (Llama-3.2-1B-Instruct, Qwen-2.5-1.5B-Instruct) and 5e−6 (Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct); cosine schedule with warmup ratio 0.03; batch 128 (IFEval, MMLU) and 64 (Countdown); 2 epochs; SFT max sequence length 4096; Self-SFT keeps correct responses among 5 samples; GRPO KL-penalty coefficient 0.05, group size 5, update applied right after each group is generated (no advantage clipping); bfloat16 on at most 8 H100 GPUs. Data splits: IFEval 13,000 train / 1,972 eval; MMLU 12,000 / 2,042; Countdown 10,000 / 1,000.

## How ch-38a uses it
§2 (Table 1, AlpacaEval table), §3 (Iterative-SFT and Self-SFT), §4 (the KL counter-evidence), §5 (the mixture account), Negative-feedback section (REINFORCE vs GRPO), Recipe.
