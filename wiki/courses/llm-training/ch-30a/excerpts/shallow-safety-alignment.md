---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/shallow-safety-alignment.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2406.05946
created_at: "2026-09-15"
---

# Excerpt: Safety Alignment Should Be Made More Than Just a Few Tokens Deep

**Authors:** Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, et al. (Princeton University; Google DeepMind)
**Version read:** arXiv:2406.05946v1 (10 Jun 2024), marked "Preprint. Under review."
**Status:** no library card existed for this slug on 2026-09-15; quotes and tables checked against the v1 PDF text.

## Definition (§2)
"we say that a model undergoes shallow safety alignment if it primarily adapts the base model's generative distribution only over the very first few output tokens to induce a basic refusal response." Evaluation: HEx-PHI (330 harmful instructions, 11 categories), GPT-4 judge; Harmfulness Rate without attack, ASR with attack (§2.1).

## Evidence of shallowness
- Table 1: prefilling a refusal prefix lowers the harmfulness rate of unaligned base models. Llama-2-7B base: 68.6 ± 0.8 with no prefix, 5.4 ± 1.3 with "I cannot fulfill", 2.1 ± 0.2 with "I apologize, but I cannot". Gemma-7B base: 85.4 ± 0.6 → 2.7 ± 0.5 ("I cannot fulfill").
- Fig. 1: per-token KL between aligned and base models on Harmful HEx-PHI "is significantly higher in the first few tokens than for later tokens."
- Fig. 3 caption (fine-tuning Llama-2-7B-Chat on 100 harmful examples, LR 2×10⁻⁵, batch 64): "ASR of initially aligned model = 1.5%; 2) After 2 gradient steps = 22.4%; 3) After 4 gradient steps = 76.4%; 4) After 6 gradient steps = 87.9%."

## Data augmentation with safety recovery examples (§3.1, Eq. 2, App. A.3)
min_θ α · E[−log π_θ(r | x, h_≤k)] + (1 − α) · E[−log π_θ(y′ | x′)], with 256 triplets (x, h, r), Alpaca prompts whose responses are distilled from the initial Llama-2-7B-Chat, α = 0.2 implemented as "16 examples from D_H and 64 examples from D_B in each batch", k = 0 with probability 50% and k ~ Uniform[1, 100] otherwise, 10 epochs on D_H, LR 2×10⁻⁵, AdamW.
- AlpacaEval win rate 49.5% vs 51.8% for the initial model (§3.1).
- Table 2 ASR, Initial → Augmented: prefill 5 tokens 42.1 → 2.8; prefill 40 tokens 57.0 → 4.5; GCG (AdvBench) 65.6 → 19.0; decoding exploit (MaliciousInstruct) 84.3 → 1.0.

## Token-wise constrained SFT (§4.1, Eq. 3-5)
min_θ E[ −Σ_t (2/β_t) log σ( β_t log( π_θ(y_t | x, y_<t) / π_aligned(y_t | x, y_<t) ) ) ].
Gradient (Eq. 5): the cross-entropy gradient multiplied by w_t := 2σ(β_t Δ_t), with Δ_t = log π_aligned(y_t | ·) − log π_θ(y_t | ·). "at the beginning of the fine-tuning when π_θ is initialized as π_aligned, the weight w_t = 1."
Settings (§4.2, App. A.5): β_1 = 0.5, β_t = 2 for 2 ≤ t ≤ 5, β_t = 0.1 for t > 5; AdamW (β1 = 0.5, β2 = 0.999); LR 2×10⁻⁵ (Llama-2-7B-Chat); batch 64; 10-step linear warmup; benign datasets 3 epochs; attacks 25 epochs.

Table 3 (Llama-2-7B-Chat; Initial / Standard SFT / Constrained SFT):
| Dataset | ASR | Utility |
|---|---|---|
| Harmful Examples | 1.5 / 88.9 / 4.6 | — |
| Samsum | 1.5 / 23.4 / 3.2 | 25.5 / 51.7 / 50.1 (ROUGE-1) |
| SQL Create Context | 1.5 / 15.4 / 3.2 | 14.9 / 99.1 / 98.5 |
| GSM8k | 1.5 / 3.3 / 2.0 | 25.5 / 41.7 / 37.4 (accuracy) |

Table 4 (ablation, uniform β): uniform β = 0.1 leaves Harmful-Examples ASR at 86.2%; uniform β = 2.0 gives 0.5% ASR but GSM8k utility 2.1%.

## How ch-30a uses it
§4 (shallow alignment and per-token dynamics), §5 mitigation table, Recipe rows, Negative-feedback section (recovery examples are negatives used as content).
