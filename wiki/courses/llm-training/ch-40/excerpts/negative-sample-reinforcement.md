---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2506.01347v2 (The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning)
source_url: https://arxiv.org/abs/2506.01347
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source — the library has no card for this paper, only references to it)"
---

# Excerpt: PSR and NSR — what each half of the gradient does to pass@k

Used by [[read]] §9, the negatives section, and the Generalization lens. Authors: Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, Yu Meng (University of Virginia; Princeton PLI). arXiv v2, 25 Oct 2025; read on 2026-09-15.

## Decomposition (§2, Eq. 3)
The RLVR objective with a binary reward splits into Positive Sample Reinforcement (increase the likelihood of correct responses) and Negative Sample Reinforcement (decrease the likelihood of incorrect ones): `L_RLVR = L_PSR + L_NSR`. Both are on-policy. PSR-only and NSR-only training update the policy with only the correct or only the incorrect responses of each batch.

## Setup (§3)
verl; MATH training set (7,500 problems); prompt batch 1,024 with 8 rollouts per prompt; mini-batch 256; LR 1e-6; training temperature 1.0; max context 4,096 (Qwen2.5-Math-7B, Llama-3.1-8B-Instruct) or 32,768 (Qwen3-4B). Evaluation samples 256 responses per prompt (64 for Qwen3-4B) and reports pass@k for k up to 256.

## Results (Table 1, Qwen2.5-Math-7B)
| Method | MATH pass@1 | MATH pass@256 | AIME 2025 pass@1 | AIME 2025 pass@256 | AMC23 pass@256 |
|---|---|---|---|---|---|
| Base model | 63.2 | 96.9 | 6.1 | 46.7 | 100.0 |
| PPO | 76.6 | 96.3 | 8.5 | 43.3 | 97.5 |
| GRPO | 76.3 | 95.5 | 10.3 | 50.0 | 97.5 |
| REINFORCE | 74.8 | 92.0 | 9.2 | 50.0 | 92.5 |
| PSR only | 74.1 | 91.2 | 11.6 | 43.3 | 92.5 |
| NSR only | 75.7 | 96.9 | 10.0 | 53.3 | 100.0 |
| W-REINFORCE (λ = 0.1) | 76.6 | 96.7 | 10.6 | 56.7 | 97.5 |

## W-REINFORCE (§4)
`L_W-REINFORCE = λ · L_PSR + L_NSR`, with `λ = 1` recovering REINFORCE. The paper uses `λ = 0.1`. The ablation over `λ ∈ {0, 0.05, 0.1, 0.2, 1}` finds similar behaviour for `λ ≤ 0.2` and a substantial drop in pass@256 at `λ = 1` (App. E).

## Mechanism stated by the authors (§4)
NSR suppresses the sampled incorrect generation and redistributes its probability mass over other candidates in proportion to the model's current beliefs, which refines existing knowledge rather than adding new behaviour; PSR sharpens the distribution onto already-preferred paths and lowers diversity.

## Limits
Mathematics only; the main table is Qwen2.5-Math-7B, which [[spurious-rewards-rlvr]] shows is unusually responsive to RLVR of any kind; Qwen3-4B and Llama-3.1-8B-Instruct results are reported separately in the paper. No seeds stated for Table 1.
