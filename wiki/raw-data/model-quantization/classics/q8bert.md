<!-- scope: Q8BERT — 8-bit BERT via simulated-quantization QAT
     deps: integer-only-inference, quantization-mapping
     see-also: i-bert, q-bert, bibert
-->

# Q8BERT: Quantized 8Bit BERT
- **Core Insight:** Vanilla 8-bit PTQ destroys BERT accuracy because of attention-output and FFN activation outliers; running standard QAT with simulated quantization on all GEMMs (Q/K/V projections, attention output, FFN) recovers the loss to <1% across GLUE — proving int8 BERT is a calibration problem, not an architectural one.
- **Guideline:** Insert fake-quant ops on all GEMM inputs (per-channel symmetric weights, per-tensor asymmetric activations); train for 1 additional epoch on the downstream task with the original SFT recipe and learning rate; calibrate activation scales on the first batch.
- **Authors:** Ofir Zafrir, Guy Boudoukh, Peter Izsak, Moshe Wasserblat
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1910.06188
- **Relevant topics:** BERT, INT8 QAT, fake-quant, GLUE, simulated quantization

## Abstract
Q8BERT applies the simulated-quantization QAT recipe of Jacob 2018 to BERT-Base, demonstrating that 8-bit weights and 8-bit activations on every linear layer preserve >99% of the fp32 GLUE score after one epoch of additional fine-tuning. The paper documents the practical pitfalls (calibration on the wrong batch, per-tensor weight quant being too coarse, residual-stream outliers) and prescribes the now-standard fix of per-channel weight scales. It is the canonical reference establishing that int8 transformer inference is achievable on commodity hardware — but only with QAT.

## Key Contributions
- First end-to-end int8 BERT-Base / BERT-Large via QAT with documented recipe.
- Demonstrates per-channel weight scales are mandatory for transformer LayerNorm + Softmax stacks.
- Documents the "first batch is unlucky" calibration pitfall; recommends multi-batch min/max + EMA.
- Records 4× model size reduction with <1% GLUE drop, motivating the entire int8 BERT line.
- Shows non-linearities (Softmax, GELU, LayerNorm) still need fp — sets up the i-BERT successor.

## Key Figures/Tables to Study
- **Table 2** — per-task GLUE: Q8BERT vs fp32 BERT-Base across MNLI/SST-2/QNLI/STS-B.
- **Figure 1** — fake-quant insertion diagram across the BERT block (where Q goes, where it does not).

## Technical Details

### Quantizer (per-tensor asymmetric, derived from Jacob 2018)
`q = clamp(round(x/S) + Z, 0, 255)`
`S = (max − min) / (Q_max − Q_min),  Z = round(Q_min − min/S)`

### Per-channel weight quantization
For W ∈ ℝ^{out×in}: maintain per-row S_c, Z_c=0 (symmetric for weights).
Per-tensor scale gives 1.5–2 point GLUE drop on MNLI — empirically observed.

### Simulated quantization in forward
Wrap every linear layer with fake-quant:
```
x_fq = FakeQuant_A(x)
w_fq = FakeQuant_W(W)
y    = x_fq @ w_fq + b
```
FakeQuant forward = quantize + dequantize (rounds to grid in fp). Backward = STE (identity within clip range).

### Modules quantized vs left in fp
- Quantized: Q/K/V projections, attention-output projection, both FFN linears, classifier head.
- Left fp: Softmax, GELU, LayerNorm, residual add, embedding lookup.
This last list is precisely what [[i-bert]] fixes by adding INT approximations.

### Calibration
- Activation S, Z: EMA over the first ~5 batches, momentum 0.99.
- Weight S, Z: recomputed every step (no EMA — weights move).

### QAT recipe
- Init from fp BERT fine-tuned on the task.
- Train 1 additional epoch.
- LR: 2e-5 (same as SFT); no LR scaling.
- Loss: original task loss, no auxiliary quantization loss.

### Empirical effect
- BERT-Base GLUE-avg: fp32 82.5 → Q8BERT 82.3 (Δ = −0.2).
- 4.0× model size reduction.
- 2-4× CPU inference latency on Intel Cascade Lake.

## Connections
- [[integer-only-inference]] — Jacob 2018 QAT recipe Q8BERT directly applies.
- [[quantization-mapping]] — sits inside the symmetric-weight + asymmetric-activation cell.
- [[i-bert]] — direct successor; closes the non-linearity gap by replacing Softmax/GELU/LayerNorm.
- [[q-bert]] — pushes below 8-bit by adding Hessian-aware mixed precision.
- [[bibert]] — extreme of the same direction: binary BERT.
- [[llm-int8]] — modern LLM successor that handles outliers without QAT (mixed-precision INT8 + fp16 path).
