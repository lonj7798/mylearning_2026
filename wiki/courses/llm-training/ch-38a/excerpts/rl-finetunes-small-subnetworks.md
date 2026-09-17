---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rl-finetunes-small-subnetworks.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2505.11711
created_at: "2026-09-15"
---

# Excerpt: Reinforcement Learning Finetunes Small Subnetworks in Large Language Models

**Authors:** Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tür, Hao Peng (University of Illinois Urbana-Champaign).
**Version read:** arXiv:2505.11711v2 (18 Dec 2025), NeurIPS 2025.
**Status:** no library card existed for this slug on 2026-09-15; numbers read at the stated loci in the v2 PDF.

## Definition (§2.2)
Update sparsity := 1 − ‖θ₁ − θ₀‖₀ / n, where θ₀ and θ₁ are the parameter vectors before and after fine-tuning, ‖·‖₀ counts non-zero entries, and n is the parameter count. Two bfloat16 values count as equal when their absolute difference is at most 10⁻⁵; all analysed models are bfloat16. Sparsity under other thresholds is in Table 6.

## Table 1 — update sparsity of released checkpoints
| Algorithm | Init → RL model | Update sparsity (%) |
|---|---|---|
| DPO | Llama-3.1-Tulu-3-8B-SFT → -8B-DPO | 81.4 |
| DPO | Llama-3.1-Tulu-3-70B-SFT → -70B-DPO | 95.2 |
| GRPO | deepseek-math-7b-instruct → -7b-rl | 68.5 |
| GRPO | DeepSeek v3 base → DeepSeek-R1-Zero | 86.0 |
| ORPO | mistral-7B-v0.1 → mistral-orpo-beta | 76.9 |
| KTO | Eurus-7b-sft → Eurus-7b-kto | 96.0 |
| PPO | mistral-7b-sft → math-shepherd-mistral-7b-rl | 80.8 |
| SimPO | Meta-Llama-3-8B-Instruct → Llama-3-Instruct-8B-SimPO | 86.5 |
| PRIME | Eurus-2-7b-sft → Eurus-2-7B-PRIME | 77.0 |

SFT stages of the same released checkpoints show 6%–15% sparsity (Fig. 1). PRIME decomposition (Fig. 2): about 20% of parameters form the consistently updated subnetwork, about 8% receive gradients that cancel out, and about 72% are never updated.

## Subnetwork-only training (§4)
Masking gradients to the final subnetwork produces a model that matches or outperforms full fine-tuning on the reported tasks, with 94.0% of weights equal to the full-fine-tuning model for DPO and 90.5% for PRIME; both are 100% identical at a 10⁻⁴ tolerance. Updates are near full rank: mean update rank as a percentage of the maximum is 99.8 (Tülu 8B DPO), 99.5 (Eurus 7B PRIME), 99.2 (Llama-3 8B KTO), 99.4 (DeepSeek Math 7B GRPO) (Table 2). Layer-normalization parameters are the exception with little to no update (§3, Fig. 3).

## Cause the authors propose (§6, Table 5)
| Model | Method | Sparsity (%) | In-distribution |
|---|---|---|---|
| Qwen2.5-Math-7B | rejection-sampling fine-tuning | 91.2 | yes |
| Qwen2.5-Math-7B | RAFT++ (iterative RFT) | 69.4 | yes |
| Llama-3.1-8B-SFT | SFT | 6.8 | no |
| Llama-3.1-8B-SFT | DPO | 6.8 | no |
| Zephyr-7b-Beta | DPO | 7.7 | no |
| Llama-3.1-8B-DPO | DPO | 81.4 | yes |

Takeaway 4: "We conjecture that training on in-distribution data could be a reason of update sparsity; KL-divergence regularization and gradient clipping have limited impact." The body text gives the Zephyr out-of-distribution DPO number as 6.8% while Table 5 lists 7.7%; the table value is used here. Training duration lowers sparsity early and then converges to about 80% for PRIME; DeepSeek-R1-Zero keeps 86.0% after roughly 8K steps (§6).

## Conflict to record
[[rls-razor]] §6 attributes the observed sparsity to bfloat16 rounding: repeating the same training in float32 gave "models with identical performance but without any sparsity in their weight updates". Mukherjee et al. state the same mechanism as their own explanation of why near-zero updates vanish ("updates with very small magnitudes … cannot be represented and are effectively discarded", §6), but keep the sparsity measurement as evidence about where the effective update mass sits. The conclusion that RL changes fewer parameters than SFT is therefore precision-dependent.

## How ch-38a uses it
§5 (the subnetwork claim, its in-distribution explanation, and the precision caveat), Common mistakes.
