<!-- scope: LLM.int8() — first INT8 PTQ that survives 6.7B+ via mixed-precision outlier decomposition
     deps: [[straight-through-estimator]], [[int8]]
     see-also: [[smoothquant]], [[gptq]], [[spqr]], [[dettmers-llm-int8-blog]]
-->

# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **Core Insight:** Once transformers cross ~6.7B parameters, a tiny handful of feature dimensions develop coordinated, large-magnitude "outlier" activations that destroy naive INT8 quantization; decomposing the GEMM into INT8 for the 99.9% of normal columns and FP16 for the ~6 outlier columns recovers full precision at half the memory.
- **Guideline:** Use vector-wise (per-row activation × per-column weight) INT8 quantization as the default; pre-scan each activation row for any |x| > 6.0, route those columns through an FP16 path, accumulate, and add — this is what `bitsandbytes`'s `Linear8bitLt` does out of the box.
- **Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2208.07339
- **Relevant topics:** outlier features, mixed-precision PTQ, INT8 GEMM, emergence, bitsandbytes

## Abstract
Quantizing transformer feed-forward and attention projections to INT8 halves inference memory but had always degraded models above ~2.7B parameters. The paper traces the failure to systematic outlier features that emerge at scale: at a critical threshold around 6.7B parameters, a small set of hidden dimensions (≈6 features in a 6.7B model) suddenly coordinate across all layers and produce activations 20–100× larger than the rest. The authors introduce a two-part procedure — vector-wise quantization with separate normalization constants per inner product, plus a mixed-precision decomposition that isolates outlier columns into FP16 — that preserves full FP16 perplexity all the way up to OPT-175B. The method is shipped in `bitsandbytes` and lets a 175B model run on a single 8×A100 node at INT8.

## Key Contributions
- First INT8 PTQ that holds zero-shot accuracy at OPT-175B / BLOOM-176B scale.
- Identifies the **emergence threshold ≈ 6.7B parameters** (correlates with perplexity, not raw parameter count) at which outliers become layer-coordinated and the standard absmax INT8 scheme breaks.
- **Vector-wise quantization**: per-token scale `c_x` for activations, per-channel scale `c_w` for weights — separate normalization constant for every inner product.
- **Mixed-precision decomposition**: split columns by activation magnitude into a "regular" set R and an "outlier" set O (|x_i| > α=6.0); do INT8 GEMM on R and FP16 GEMM on O, then add.
- Ships as `Linear8bitLt` in `bitsandbytes`, enabling consumer-GPU inference of 175B-class models.

## Key Figures/Tables to Study
- **Figure 1 (memory + accuracy curves):** shows the cliff at 6.7B where vanilla absmax INT8 collapses while vector-wise + decomposition stays flat.
- **Figure 2 (outlier magnitude vs scale):** outlier peaks grow from ~15 (6B) → ~60 (13B) → ~95 (66B); critical evidence for the emergence framing.
- **Table 1:** OPT/BLOOM perplexity and 0-shot accuracy at 125M…175B — INT8 with decomposition is within noise of FP16 everywhere.

## Technical Details

### Vector-wise quantization
For input `X ∈ R^{s×h}` and weight `W ∈ R^{h×o}`, instead of a single scalar scale:
- per-token activation scale: `c_x ∈ R^s`, `c_{x,i} = 127 / max_j |X_{ij}|`
- per-output-channel weight scale: `c_w ∈ R^o`, `c_{w,j} = 127 / max_i |W_{ij}|`
- Quantize: `X̂ = round(c_x ⊙ X)`, `Ŵ = round(c_w ⊙ W)` (both INT8 in [-127, 127])
- Compute INT32 GEMM `X̂ · Ŵ`, dequantize by `(c_x c_w^⊤)^{-1}` element-wise.

This is equivalent to per-row × per-column rescaling — each output entry gets its own normalization constant, so a single outlier doesn't poison an entire row.

### Mixed-precision decomposition for outliers
Define the outlier set `O = { i : ∃ j, |X_{ji}| ≥ α }` with **α = 6.0** (chosen empirically; covers the emergent dims with margin). Let `R = {0,…,h-1} \ O`. Then
```
X · W = Σ_{i∈O} X_{·i} W_{i·}   +   Σ_{i∈R} X_{·i} W_{i·}
       └── FP16 GEMM ──────────┘   └── INT8 GEMM ───────────┘
```
- Typically |O| = 6–20 columns out of 12k–14k hidden dims (<0.1% of dims, ~99.9% of values stay INT8).
- The FP16 path is cheap because |O| is tiny; the INT8 path carries the work.
- No calibration, no retraining — outlier set is recomputed per forward pass from the activation absmax.

### Why α = 6.0
Empirically the outlier features at the emergent scale have magnitudes 15–100; non-outlier dims stay below 6. Setting α=6 captures all emergent dims at every scale tested while never picking up >0.1% of values.

### Hyperparameters / knobs
| Knob | Value |
|------|-------|
| α (outlier threshold) | 6.0 |
| Activation scale | per-token (per-row), absmax |
| Weight scale | per-output-channel, absmax |
| Bit-width | INT8 + FP16 outlier path |
| Calibration data | none — purely runtime |
| Frameworks | `bitsandbytes.nn.Linear8bitLt`, HF `load_in_8bit=True` |

## Connections
- Dettmers' companion blog with the emergence intuition: [[dettmers-llm-int8-blog]].
- Outlier migration (move difficulty into weights instead of isolating columns): [[smoothquant]], [[awq]].
- Outlier weights (not activations) kept in fp16: [[spqr]], [[squeezellm]], [[owq]].
- 4-bit successor from the same group: [[qlora]], [[spqr]].
- The bridge to rotation methods that eliminate the outlier problem entirely: [[quip]] → [[quarot]].
- Framework reference: [[bitsandbytes-int8]].
