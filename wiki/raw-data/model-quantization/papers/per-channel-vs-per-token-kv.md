<!-- scope: analytical study of K-vs-V quant asymmetry — why K demands per-channel and V demands per-token
     deps: [[kivi]], [[llm-int8]]
     see-also: [[kvquant]], [[qaq]], [[wkvquant]]
-->

# Per-Channel vs Per-Token KV Quantization — Analytical Study
- **Core Insight:** The K cache inherits the *residual-stream channel outliers* of the input (via W_k) and these outliers are *persistent across tokens*, so K must be quantized per-channel; the V cache, by contrast, has no consistent channel-outlier pattern but has token-wise variation driven by attention mass, so V should be quantized per-token. Quantizing both along the wrong axis introduces order-of-magnitude precision loss in the outlier channels.
- **Guideline:** Treat the K/V quant-axis choice as fixed (K per-channel, V per-token) for any LLM with residual-stream channel outliers — measured to hold across all GPT/LLaMA/Mistral family models; per-layer axis re-selection (WKVQuant) yields marginal additional gains.
- **Authors:** consolidated entry — the formal study is the KIVI distribution analysis (Section 3.2) and the WKVQuant 2D KV strategy; this entry consolidates them
- **Year:** 2024
- **URL:** primary: [[kivi]] https://arxiv.org/abs/2402.02750 ; complementary 2D-axis study: [[wkvquant]] https://arxiv.org/abs/2402.12065
- **Relevant topics:** K vs V asymmetry, per-channel quant, per-token quant, residual-stream outliers

## Abstract
This entry distills the asymmetry-of-axes result that KIVI surfaced and that WKVQuant generalized to per-layer adaptive selection. The empirical finding: K and V have *different* outlier geometries — K is channel-wise outlier-concentrated (inherited from W_k projecting the residual stream), V is token-wise variable (driven by which tokens get attention mass at use time). Choosing the right axis is the single largest determinant of low-bit KV-cache accuracy; all subsequent KV-quant work assumes this asymmetry.

## Key Contributions (consolidated)
- Empirical demonstration that K outliers are channel-wise and persistent across tokens; V outliers are token-wise and inconsistent across channels.
- Causal explanation: K = W_k · x where W_k inherits the residual-stream outlier channels (the same outliers that motivate SmoothQuant and LLM.int8()); V = W_v · x is structurally similar but attention reweighting averages out channel patterns over many tokens.
- Quantitative gap: per-channel K at INT2 beats per-token K at INT2 by 5+ PPL points on LLaMA-2.
- Generalisation: WKVQuant shows that per-layer auto-selection between the two axes gives marginal further gains (~0.1 PPL).

## Key Figures/Tables to Study
- KIVI Figure 2: K and V distribution heatmaps on LLaMA-2 — the load-bearing visual evidence.
- KIVI Table 4: ablation of K-axis × V-axis at INT2 — only K-per-channel + V-per-token works at full INT2.
- WKVQuant Figure 2: per-layer optimal axis distribution — mostly K-per-channel and V-per-token, with a few exceptions.

## Technical Details

### Why K is channel-wise
K_t = W_k · x_t. The input x_t is the residual stream, which carries a small number of consistent "outlier channels" (the LLM.int8() finding) of magnitude 10–100× the bulk. W_k mostly preserves this channel structure (its columns project the residual stream into per-head subspaces). Across all tokens t, the *same* channels of K are large — by construction.

If you quantize K per-token (one scale per token spanning all channels), the scale is dominated by the outlier channels at every token, crushing precision on the bulk. If you quantize per-channel (one scale per channel spanning all tokens), the outlier channels get their own big scale and bulk channels get tight scales — recovering precision.

### Why V is token-wise
V_t = W_v · x_t structurally inherits the same outliers, but V is *used* at attention time as a weighted average: `out_q = Σ_t a_{q,t} V_t`. The attention weights a_{q,t} re-mix the channel pattern of V; over the softmax average, channel-wise outliers wash out and what dominates is which *token* contributes (token-wise variation).

If you quantize V per-channel, the per-channel scale wastes precision on channels that the attention reweighting will downplay. Per-token quantization fits the actual use pattern.

### The numbers
On LLaMA-2-7B at INT2 KV (KIVI ablation):
- K per-channel, V per-token: PPL 7.0.
- K per-token, V per-token: PPL 12.4 (5+ point hit from wrong K axis).
- K per-channel, V per-channel: PPL 8.3 (1+ point hit from wrong V axis, smaller than wrong-K).
- K per-token, V per-channel: PPL 14+ (both wrong).

### Per-layer axis adaptivity (WKVQuant)
A small number of layers (typically the first 1–2 and last 1–2) have slightly different geometry. WKVQuant runs a calibration that picks per-channel or per-token per layer for K and V independently. Yields ~0.1–0.2 PPL improvement over KIVI's globally-fixed axes.

### Practical recipe
Default: K per-channel with group 32 along token axis, V per-token with group 32 along channel axis. Reach for per-layer adaptivity only if pushing below INT2.

## Connections
- Primary empirical source: [[kivi]].
- Generalisation paper: [[wkvquant]].
- Outlier-channel ancestor finding: [[llm-int8]].
- KV-quant siblings building on this asymmetry: [[kvquant]] (adds non-uniform + dense-and-sparse), [[gear]] (adds low-rank residual), [[qaq]] (adds adaptive bit allocation), [[skvq]] (adds sliding window).
- KV-cache compression survey: [[kv-cache-survey]].
