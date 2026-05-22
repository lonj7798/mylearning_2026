<!-- scope: consolidating note for the BitNet b1.58 'era of 1-bit LLMs' thesis and its follow-ups
     deps: [[bitnet]], [[bitnet-b158]]
     see-also: [[bitnet-a48]], [[onebit]]
-->

# The Era of 1-bit LLMs — Survey-Style Consolidation
- **Core Insight:** The BitNet b1.58 results define a *new scaling law*: a 1.58-bit LLM matches a full-precision LLM of the same size and tokens, so the FP16 baseline that the rest of the field uses is actually a strict over-spend on bits — the cost-optimal frontier sits at sub-2-bit weights with INT8 (or INT4) activations.
- **Guideline:** When sizing a pretraining run for cost, treat the bit-width as a third scaling-law axis (alongside N parameters and D tokens); the BitNet recipe is the practical instantiation that achieves the sub-2-bit frontier without specialty data.
- **Authors:** consolidated entry — covers [[bitnet]] (Wang 2023), [[bitnet-b158]] (Ma 2024) and follow-ups
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.17764 (BitNet b1.58 paper articulating the "era" thesis); HuggingFace community follow-ups: https://huggingface.co/blog/1_58_llm_extreme_quantization
- **Relevant topics:** 1.58-bit scaling law, ternary weights, energy-per-token, hardware-software co-design

## Abstract
This entry captures the broader thesis that the BitNet b1.58 paper made famous: at a fixed parameter count and training-token budget, a 1.58-bit LLM (ternary weights, INT8 activations) reaches the same perplexity and end-task performance as a full-precision LLM of the same size. The implication is that the entire field's FP16 default is sub-optimal — the cost-optimal frontier on the (parameters × tokens × bits) volume sits at sub-2-bit weights. Follow-ups (BitNet a4.8, OneBit, HuggingFace's fine-tune-to-1.58 recipe) consolidate this into a practical recipe.

## Key Contributions (consolidated)
- Articulates the 3-axis scaling-law framing: cost is a function of (N, D, bits/weight), not just (N, D).
- Demonstrates parity with FP16 at 1.58 bits on 3B-scale models trained from scratch (BitNet b1.58 2B paper, follow-up).
- Shows that the 1.58-bit recipe can be reached *not only* by scratch training, but by adapter-style fine-tuning of an existing FP16 LLM with gradual quantization scheduling (HuggingFace recipe).
- Motivates dedicated 1-bit hardware: ternary MACs reduce to additions + sign flips, eliminating multipliers; preliminary energy estimates show 71× reduction in arithmetic energy vs LLaMA at the same parameter count.

## Key Figures/Tables to Study
- BitNet b1.58 Figure 1: scaling-law plot showing 1.58-bit and FP16 converging at 3B parameters.
- HuggingFace blog: lambda-scheduled gradual quantization curves during fine-tuning.
- Energy bar chart (BitNet b1.58, Section 4): arithmetic energy per token, 1.58-bit vs FP16.

## Technical Details

### The 1.58-bit weight rule
Each weight is one of `{−1, 0, +1}`, encoded in log₂(3) ≈ 1.58 bits. Forward pass:
`scale_w = 1 / mean(|W|)`,  `W_q = clip(round(W·scale_w), −1, 1) / scale_w`
Activations stay at INT8 (BitNet b1.58) or INT4 (BitNet a4.8).

### Why 1.58 not 1
A pure 1-bit `{−1, +1}` weight loses the "zero" — the model cannot turn off a connection. Empirically the zero is essential to recover FP-quality at scale (Ma 2024 ablation). The cost of supporting zero is 0.58 bits/weight, well worth it.

### Fine-tune-to-1.58 (lambda schedule)
For each training step t out of T:
`λ_t = min(2t/T, 1)`
`w_quant = w + λ_t · (weight_quant(w) − w).detach()`  (lambda-mixed STE)
This warms the model up to ternary precision over the first half of training, then keeps it locked for the second half. Avoids the abrupt-information-loss problem of switching to ternary at step 0.

### Scaling law (informal)
Empirical fit (BitNet b1.58 paper, extrapolated):
`Loss(N, D, b) ≈ A · N^{−α} + B · D^{−β} + C(b)`
with C(b) plateauing for b ≥ 1.58 — every bit beyond ternary is wasted at fixed N, D. Compute cost scales linearly in b for HBM-bound regimes (decode), making sub-2-bit strictly cheaper at iso-quality.

### Hardware implication
A ternary MAC `y += w · x` with w ∈ {−1, 0, +1} reduces to:
- w = +1: y += x.
- w = −1: y −= x.
- w = 0: skip.
No multiplier needed. Lookup-table-based MACs in custom silicon (BitNet white paper) project 10–70× energy reduction vs FP16.

## Connections
- Original 1-bit weight LLM: [[bitnet]].
- The 1.58-bit recipe: [[bitnet-b158]].
- Pushing activations to 4-bit: [[bitnet-a48]].
- Alternative 1-bit decomposition: [[onebit]].
- Classical 1-bit roots: [[bnn]], [[xnor-net]].
- Production-scale follow-up tracked under: bitnet-b158-2b, bitnet-scaling-laws.
