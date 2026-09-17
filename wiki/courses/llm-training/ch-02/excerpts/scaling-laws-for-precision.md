---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, et al. — Scaling Laws for Precision"
source_url: https://arxiv.org/abs/2411.04330
created_at: "2026-09-15"
revised: "2026-09-15 (created from arXiv:2411.04330v2; generality revision)"
---

# Excerpt: Scaling Laws for Precision (arXiv 2024-11)

Created on 2026-09-15 from arXiv:2411.04330v2 (2024-11-30). No library card exists for this paper.

## Setup (§2.3, §3, §4)
- OLMo-style Transformer++ models on Dolma V1.7; N ∈ {30, 60, 110, 220}M non-embedding parameters;
  D ∈ {1.5, 3, 6, 13, 26}B tokens; more than 20 runs per (N, D), sweeping 8 precisions for weights,
  activations, and attention (KV cache) (§2.3).
- 465 pretraining runs in 3-16 bit precision, each post-train quantized to several precisions; predictions
  validated up to 1.7B parameters and 26B tokens (Abstract, §1).
- Post-training quantization (PTQ) of BF16-trained models uses GPTQ, replicated with two other methods (§3).
- Quantized training quantizes the forward pass in integer type with fake (simulated) quantization on H100
  GPUs (§4, App. D.3).
- Notation: D tokens, N parameters, P_w, P_a, P_kv training precisions of weights, activations, KV cache;
  P_post the post-training weight precision; δ_PTQ the loss increase caused by PTQ (§2).

## Finding 1: over-trained models degrade more under PTQ (§3.1, Eq. 2)
`δ_PTQ(N, D, P_post) = C_T · (D^γ_D / N^γ_N) · e^(−P_post / γ_post)`
- C_T, γ_D, γ_N, γ_post are positive fitted constants. δ_PTQ increases with D for every N; for fixed D larger
  models degrade less; δ_PTQ grows exponentially as P_post decreases (§3.1, Fig. 2).
- Consequence stated by the authors: for a model that will be post-train quantized, there is a data budget
  beyond which more pretraining data increases loss after quantization (§3.1; critical D in App. E.1, Eq. 13).
- App. E.1: at common precisions such as 8-bit the critical data size is very large; for D/N ≫ 10^3 the
  degradation effects "become nontrivial around 5-bits, and dominant below that".
- Context given in §2.2: Llama-3-8B is trained to D/N ≈ 2000; Gemma-2 up to D/N > 1000. Experiments reach
  D/N ≈ 10^3; the law is analysed up to D/N ≈ 10^5.

## Finding 2: effective parameter count (§4.1-4.2, Eq. 3-4)
`L = A·N_eff^(−α) + B·D^(−β) + E`,  `N_eff = N (1 − e^(−P_w/γ_w)) (1 − e^(−P_a/γ_a)) (1 − e^(−P_kv/γ_kv))`
- Gains from more weight bits saturate around 6-7 bits (§4.1). Effects of weight, activation, and KV precision
  are modeled as independent and multiplicative (Finding 2).

## Finding 3: compute-optimal training precision (§4.3)
- With cost C = (6/16)·N·D·P, jointly optimizing N, D, P gives an optimal precision independent of compute;
  fits on integer-type quantization give 7-8 bits (§4.3.2). A compute-matched floating-point check from FP4
  to FP32 (220M to 1.6B parameters) agrees qualitatively (§4.3.2, Fig. 6).
- When N is fixed in advance, compute-optimal precision grows approximately with log C (§4.3.3).

## Finding 4: training precision and PTQ robustness (§5, Eq. 8-11)
- Models trained in lower precision degrade less under PTQ; the fitted unified form has R² = 0.90 on more
  than 1000 points (§5).

## Fitted constants (App. K, Table 2) and their limits
- γ_D = 0.5068, γ_N = 0.3439, C_T = 0.0598, α = 0.4965 (tied to β), E = 2.7648.
- The authors: "our numerical constants are unlikely to be useful" across architectures and datasets; "the
  trends we identify are the key findings" (App. K). Fits include bias offsets omitted from the main-text
  notation (App. K).

## Limitations stated by the authors (§6)
- One fixed architecture; low-precision training often uses architectural changes that close part of the gap.
- Halving precision gives less than 2x speedup in practice because of systems overhead.
- "We only consider loss scaling without downstream model evaluations."

## Used in ch-02
- §5 (precision and tokens per parameter, worked example with γ_D), Generalization lens (b), Common mistakes.
