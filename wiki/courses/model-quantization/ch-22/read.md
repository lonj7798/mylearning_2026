<!-- chapter: ch-22
     track: eval-lab-capstone
     kind: capstone
     title: Capstone — Reproduce a Frontier Method on a Small LLM
     deps: [ch-15, ch-17, ch-18, ch-20]
     sources: [[turboquant]], [[kivi]], [[kvquant]], [[gear]], [[deepseek-v3-fp8]], [[nvfp4-training]]
     capstone_for: model-quantization
-->

# Chapter 22 — Capstone: Reproduce a Frontier Method on a Small LLM

> **Capstone objective.** Pick one frontier quantization method (KIVI / KVQuant / GEAR / TurboQuant / NVFP4 inference) and **reproduce it from the paper alone — not from the author's repo.** Run it on Llama-3-8B (full-budget) or TinyLlama-1.1B-chat (resource-constrained), evaluate under the [[ch-20]] long-context harness, and produce a **reproduction memo** that names the gap (or lack of gap) between your numbers and the paper's reported numbers within ±2%. If your numbers don't match, the memo's job is to explain *where the gap is and why*. That explanation is the actual learning outcome.
>
> **Guideline.** Implement the method from the paper text and equations. Treat the author's repo as a *verification* resource only — use it to check your implementation works, not to copy from. The capstone tests your ability to extract a runnable system from a paper, which is the core researcher skill the course was built for.

---

## Why this capstone exists

The lab (ch-21) gave you four known-good methods with `pip install` recipes. The capstone tests something different: can you read a paper, implement the method, hit the paper's numbers, and write a reproduction report that is as good as the paper itself?

This is the hardest skill in research engineering. Every quantization paper looks reproducible until you try. The gap between "the paper says 3-bit KV is lossless" and "I got 3-bit KV with PPL gap 0.3" is where you learn what the paper *didn't* tell you — and where the field's actual progress lives.

The five method options span the 2024–2026 frontier across three different surfaces (weight quant, KV-cache quant, low-precision training). Pick the one whose surface matters most for your career direction.

---

## Method options

Each option below has: the core mechanism, the key reproduction risk, and the resource floor.

### Option 1 — KIVI (asymmetric 2-bit KV cache)

**Mechanism** (per [[kivi]]):
- KV cache split: K per-channel (channel-axis scale, token-axis grouping at g=32), V per-token (token-axis scale, channel-axis grouping at g=32), both INT2.
- Tuning-free: no calibration step, no fine-tuning.
- Streaming buffer: maintain a small FP16 residual of the last <32 tokens; quantize in chunks of 32 to avoid recomputing per-channel statistics.

**Why this is the right starting point.** KIVI is the simplest of the five — no calibration, no learned rotations, no native-hardware dependencies. The mechanism is "split K and V along different axes." The reproduction risk is the streaming buffer's chunk-quantization correctness and the per-channel-K integration with PagedAttention.

**Key reproduction risk.** The K per-channel scale must be computed *along the token axis grouped by channels*. Getting the axis right is the single most common bug — naive implementations end up per-channel-per-token (which is just FP16) or per-tensor (which is just naive scalar). Test: on a small synthetic K tensor, verify your per-channel quant matches the paper's stated effective bit budget (2 + 16/32 = 2.5 bits/element).

**Target numbers** (from [[kivi]]):
- Llama-2-7B at INT2 KV: Wikitext-2 PPL gap <0.2 vs FP16.
- LongBench average within 1 point of FP16.
- 2.6× peak memory reduction, 2.35–3.47× throughput.

**Reproduction risks specific to KIVI:**
- The streaming chunk size (g=32) interacts with the model's actual KV cache layout. PagedAttention's block size may not align with 32 — accept misalignment and pay the per-block bookkeeping, or align the block size.
- The CUDA kernel for INT2 packed K dequant + matmul has to fuse into attention; a slow Python reference implementation is fine for correctness but will not reproduce throughput numbers.

### Option 2 — KVQuant (sub-4-bit KV via pre-RoPE quant + dense-and-sparse)

**Mechanism** (per [[kvquant]]):
- Quantize K *before* RoPE rotation, store K pre-RoPE, apply RoPE on dequant at attention time.
- Per-channel non-uniform code: per K-channel quantile-fit k-means codebook of 2^B bins (B=2 or 3).
- Dense-and-sparse decomposition: top 1% absolute-value elements stored as FP16 sparse vector; rest as the non-uniform code.
- Q-norm quantization for the Query side so QK^T runs entirely low-bit.

**Why this is the next step up.** Same problem as KIVI (KV-cache compression) but four jointly-applied techniques instead of one. Tests whether you can reproduce a multi-piece method where ablating any single piece degrades the result.

**Key reproduction risks:**
- The pre-RoPE quantization changes the attention math: standard implementations apply RoPE to K right after the projection. Hooking this requires modifying the attention forward path, not just the KV-cache storage path.
- The per-channel non-uniform code requires fitting a per-channel k-means at calibration time. The codebook learning has a hyperparameter (`n_iterations`); under-trained codebooks give catastrophic results that look like the implementation is broken.
- The dense-and-sparse decomposition has a top-K selector that must run *after* quantization, not before. Order matters; reversing it gives a different (worse) sparse pattern.

**Target numbers** (from [[kvquant]]):
- 3-bit KV on Llama-7B: <0.1 PPL degradation on Wikitext-2 and C4.
- 1M-token context on a single A100-80GB.
- ~1.7× speedup over FP16 matvec via custom CUDA kernels.

### Option 3 — GEAR (KV cache via low-rank residual + sparse outlier)

**Mechanism** (per [[gear]]):
- `KV ≈ Q + L + S` where `Q` is uniform b-bit quant, `L = AB^T` is rank-r SVD of the residual, `S` is a sparse outlier matrix holding top-1% residual entries.
- Streaming SVD that updates A, B incrementally as new tokens arrive.
- Periodic full SVD refresh every ~256 tokens to prevent drift.

**Why this is the algorithmic curveball.** GEAR is the one with non-trivial *online* computation — the streaming SVD update is the hard part. Tests whether you can implement an incremental-numerical-linear-algebra routine correctly under autoregressive decoding.

**Key reproduction risk.** Streaming SVD updates are notoriously easy to get *almost* right — the periodic full refresh hides bugs. Test: turn off the periodic refresh, run for 1000 tokens, verify the incremental A, B match a from-scratch SVD on the same accumulated data. If they drift, your incremental update has an orthogonality bug. The most common bug: forgetting to re-orthogonalise B after projecting a new column.

**Target numbers** (from [[gear]]):
- 4-bit KV near-lossless on WikiText-2; long-context accuracy matches FP16.
- 2.38× throughput, 2.29× peak memory reduction.

### Option 4 — TurboQuant (data-oblivious KV via random rotation + QJL)

**Mechanism** (per [[turboquant]]):
- Random Hadamard rotation `v' = R · v` applied to each KV vector.
- Per-coordinate scalar quant using the analytically optimal quantizer for the Beta distribution that the rotation induces.
- 1-bit Quantized Johnson-Lindenstrauss (QJL) sketch on the residual `e = v' − Q(v')`: store only `sign(Π · e)` for a JL matrix `Π`.
- Inner-product estimator combines the quantized leg and the QJL residual leg.

**Why this is the most theoretically interesting.** Data-oblivious — no calibration, online-feasible, hits the rate-distortion bound up to a small constant. The mechanism is unusual: a *fixed* (not learned, not calibrated) quantizer that works because the rotation makes the input distribution analytically known.

**Key reproduction risk.** The QJL asymmetric inner-product estimator from [[turboquant]] (`α · ⟨sign(Π·e_u), Π·e_v⟩`) has a specific α calibration that depends on JL width `m`. Getting α wrong gives biased inner-product estimates that look like the JL sketch is broken when it's actually the estimator. Reference the QJL paper directly for the α formula.

**Target numbers** (from [[turboquant]]):
- 3.5-bit KV: absolute quality neutrality on Gemma and Mistral on LongBench, NIAH, RULER, ZeroSCROLLS, L-Eval.
- 2.5-bit KV: marginal degradation only.
- 8× speedup over FP32 keys at 4-bit on H100.

### Option 5 — NVFP4 inference (or DeepSeek-V3 FP8-style)

**Mechanism** (per [[nvfp4-training]] or [[deepseek-v3-fp8]]):
- Two-level scaling: FP4/FP8 elements × FP8 block scale × FP32 per-tensor scale.
- 16-element blocks for NVFP4, 1×128 activation tiles + 128×128 weight blocks for DSV3-FP8.
- For NVFP4 inference (the more tractable reproduction): no training, just convert an FP16 model to NVFP4 format with the two-level scaling, plus selective high-precision exceptions (embedding, head, early-block sensitivity per [[fp4-inference-diagnosis]]).

**Why this is the production-frontier option.** Targets actual deployment-format quantization on Blackwell-class hardware. Will not run natively without H100/H200/B100/B200; on Ampere you must emulate (per-element FP4 stored as FP8 with scale arithmetic).

**Key reproduction risk.** Native NVFP4 hardware support is required for the published throughput numbers. On non-Blackwell hardware, you can only verify *quality* (the two-level scaling math is portable), not throughput. State this explicitly in the memo — emulated-FP4 numbers are *not* comparable to the paper's native-hardware throughput.

**Target numbers** (post-training NVFP4 inference on a converted Llama-3-8B):
- ~2× memory reduction vs FP16.
- PPL gap that depends on the exception policy; with MLP-up/down kept at FP8 per [[fp4-inference-diagnosis]], target <0.3 PPL gap.
- Native throughput requires Blackwell; emulate-only path doesn't show throughput gains.

---

## Recommendation matrix

If you have **3 days** and want clean reproduction: pick **KIVI**.
If you want to understand multi-component PTQ: pick **KVQuant**.
If you want a streaming-numerics challenge: pick **GEAR**.
If you want the theoretically deepest: pick **TurboQuant**.
If you have **Blackwell hardware** and care about training-format work: pick **NVFP4 inference**.

The capstone is graded on the reproduction memo, not on the specific method.

---

## Full-budget path

**Target.** 1 × H100 (80 GB), Llama-3-8B base, ~3–5 days end-to-end.

**Method-specific notes:**
- KIVI / KVQuant / GEAR / TurboQuant: KV-quant methods; the FP16 weights are unchanged. Convert KV-cache storage and update the attention forward path.
- NVFP4 inference: replace linear-layer weights and activations with NVFP4-emulated equivalents; native NVFP4 GEMM only runs on Blackwell.

**Long-context eval (mandatory).** Per [[ch-20]] §4:
- **NIAH** heatmap at 4K / 8K / 16K / 32K / 64K context × 11 depth points.
- **RULER** 4-task subset (single-needle, multi-needle, multi-hop, aggregation) at 32K context.
- **LongBench** per-category averages at 16K context.
- Compare each to FP16-KV baseline; report deltas, not just absolutes.

**Target bit budget.**
- KIVI / TurboQuant: 2.5–3 bits/element.
- KVQuant: 2–3 bits/element.
- GEAR: 4 bits/element (effective ~4.6 with low-rank + sparse overhead).
- NVFP4: 4 bits/element on linear weights (plus exceptions).

---

## Resource-constrained path

**Target.** 1 × consumer GPU (≥ 24 GB; RTX 3090 / 4090 / A5000), TinyLlama-1.1B-chat or Qwen2.5-1.5B-Instruct, ~2 days wall-clock.

**Reduced context.** NIAH/RULER/LongBench at 4K and 8K context only — 1.5B-class models often don't have stable >16K context to begin with, so longer evals would conflate model capability with quantization quality.

**Method scope unchanged.** Implement the *full* method, not a simplified version. The paper's algorithm is the deliverable; reducing scope makes the reproduction inconclusive.

**Caveat.** TinyLlama's KV cache outlier patterns may differ from Llama-2 / Llama-3's — the per-channel-K assumption from [[kivi]] / [[kvquant]] should hold qualitatively but the per-channel scale magnitudes will differ. Document any such differences in the memo.

---

## The reproduction workflow

A six-step skeleton that works for any of the five methods:

### Step 1 — Read the paper twice

First pass: high-level mechanism. Build a one-paragraph explanation of "what does this method *do* to the KV cache / weights." If you can't write the paragraph from the paper alone, you don't understand the method yet — read it again before writing code.

Second pass: equation-level. For every equation in the methods section, write the corresponding line of pseudocode. The paper-to-pseudocode mapping is what you'll lean on during debugging.

### Step 2 — Implement the math, no kernel optimization

Write a pure-PyTorch reference implementation. Slow is fine. Use `torch.einsum` and Python loops where convenient. The goal is *correctness*: a quantized model that produces sensible outputs for short sequences.

For KIVI specifically, the reference is:
```python
def quant_K_per_channel(K, group_size=32, bits=2):
    # K shape: (n_tokens, n_heads, head_dim)
    # Quantize per channel (head_dim axis), groups along token axis
    T, H, D = K.shape
    n_groups = (T + group_size - 1) // group_size
    K_q = torch.zeros_like(K, dtype=torch.int8)
    scales = torch.zeros(n_groups, H, D, dtype=K.dtype, device=K.device)
    for g in range(n_groups):
        s, e = g * group_size, min((g+1) * group_size, T)
        block = K[s:e, :, :]                            # (g_sz, H, D)
        # Per-channel (D axis) scale within this token-group
        scale = block.abs().amax(dim=0)                 # (H, D)
        scales[g] = scale / (2**(bits-1) - 1)
        K_q[s:e] = torch.round(block / scales[g]).clamp(-2**(bits-1), 2**(bits-1) - 1).to(torch.int8)
    return K_q, scales
```

Run this on a synthetic K tensor, dequantize, verify error is bounded by the expected `Δ/2` per-channel.

### Step 3 — Hook it into the model

Replace the model's KV-cache storage path with your quantized version. For HuggingFace transformers, monkey-patch the `update` method of the model's KV cache class:

```python
def _quantized_cache_update(self, key_states, value_states, layer_idx, **kwargs):
    # ... your KIVI/KVQuant/GEAR/TurboQuant quantization here ...
    return dequantized_keys, dequantized_values
```

For HF's newer `Cache` API (transformers ≥ 4.36), subclass `DynamicCache` and override `update`. For older models, monkey-patch the attention forward directly.

Verify: generate 100 tokens with FP16 KV and quantized KV; the generated token sequences should differ but be sensible (no NaN, no repetition collapse, no all-zeros).

### Step 4 — Run the eval harness

Apply the [[ch-20]] §6 harness:
- PPL on Wikitext-2 + C4.
- MMLU + GSM8K (short-context tasks; KV-quant should barely move these).
- **Long-context: NIAH heatmap + RULER 4-task subset + LongBench average** (the metrics that matter for KV-quant).

Compare to the paper's reported numbers. The gap-vs-paper is the headline number for the memo.

### Step 5 — Iterate until the gap is ≤2% or you understand why it's bigger

If your numbers match the paper within 2% on PPL and within 1 pp on benchmark accuracies, congratulations — you've reproduced the method.

If not, the gap is the actual lesson. Walk down the debugging tree:
1. Implementation bug? Compare your reference math to the paper's equations line-by-line.
2. Calibration set difference? Re-run with the paper's exact calibration corpus + size + seed (the paper usually states this).
3. Model architecture mismatch? Llama-3 has different attention head count + KV-head count (GQA) than Llama-2; per-channel-K behavior differs.
4. Hardware-dependent? Kernel-fused operations may give different numerics than pure-PyTorch reference.

State your hypothesis. Test it. Document the result.

### Step 6 — Write the memo

The reproduction memo (template in [[excerpts/reproduction-memo]]) is the deliverable. It is the most important artifact of the capstone — more important than the code, because the *understanding* the memo demonstrates is what transfers to future work.

---

## What "reproduction" means

Reproduction does *not* mean: "I cloned the author's repo and ran their script."

Reproduction means: "I read the paper, implemented the method, ran it on comparable infrastructure, and got numbers within the paper's claimed bounds — or I can name precisely where my numbers diverge and offer a hypothesis for why."

Both outcomes are valid. The memo for a successful reproduction documents the recipe that worked. The memo for a failed reproduction documents the gap and the diagnosis — which is often *more* valuable to the field, because it surfaces what the paper omitted.

A bad memo: "I couldn't reproduce. Maybe the paper is wrong."
A good memo: "I got 0.45 PPL gap on Llama-3-8B at 3-bit KV with KIVI; the paper reported 0.18 on Llama-2-7B. The gap is reproduced when I run my implementation on Llama-2-7B (0.20 PPL gap, within 2% of paper). I attribute the Llama-3-8B gap to GQA's smaller KV head count, which makes per-channel K scales more sensitive to per-channel outliers. Recommendation: KIVI at 3-bit should add per-channel outlier isolation for GQA architectures."

The good memo is publishable as a workshop note. The bad memo is gossip.

---

## Optional stretch goals

Only attempt these if the core capstone is complete and the memo is finalized.

### Stretch A — Compound with W4A4 weight quant

Stack the KV-quant method with a W4A4 weight quantization scheme (QuaRot is the simplest open recipe). The compounding question: does the long-context degradation under KV-quant alone *add* to the W4A4 degradation, or does it *multiply*?

Hypothesis from [[turboquant]] and [[kvquant]]: under healthy rotation + non-uniform-code, the two are roughly additive (degradations sum, not multiply). Under naive uniform per-token KV + W4A4 RTN, they multiply catastrophically.

Run KIVI-INT2-KV + QuaRot-W4A4 on Llama-3-8B; compare to KIVI alone and QuaRot alone; report the compounding factor on Wikitext PPL and LongBench average.

### Stretch B — Deploy on vLLM and benchmark serving throughput

Wrap your KV-quant implementation as a vLLM custom backend (subclass `AttentionBackend` and register a custom attention kernel). Benchmark serving throughput at:
- batch sizes 1, 16, 64.
- context lengths 4K, 16K, 32K.
- concurrent requests = batch_size.

Compare to FP16 vLLM serving and to the paper's reported throughput (where the paper has matching numbers).

This stretch goal turns the academic reproduction into a production-relevant artifact. It is the natural next step but is substantial additional work — budget another 2-3 days.

---

## Connections

- **Back to [[kivi]] (ch-15), [[kvquant]] (ch-15), [[gear]] (ch-15)** — three of the five options were introduced in the KV-cache chapter.
- **Back to [[deepseek-v3-fp8]] (ch-17), [[nvfp4-training]] (ch-17)** — low-precision training chapter; NVFP4 inference is the deployment-side reproduction of this lineage.
- **Back to [[turboquant]] (ch-18)** — data-oblivious KV; the most theoretically novel option.
- **Back to [[ch-20]]** — the evaluation methodology this capstone applies, especially §4 (long-context evaluation) and §5 (calibration design).
- **Forward** (post-course) — submit the reproduction memo as a workshop note (e.g., MLSys Workshop on Quantization; Efficient LLM Inference Workshop at NeurIPS). The bar for these venues is exactly "novel evidence about an existing method," which a clean reproduction or a documented gap satisfies.

## Excerpts

- [[excerpts/method-comparison]] — side-by-side breakdown of the five method options with reproduction-risk scoring.
- [[excerpts/reproduction-memo]] — the memo template + worked example.
- [[excerpts/long-context-harness]] — the NIAH / RULER / LongBench scripts pinned for the capstone.
- [[excerpts/debugging-tree]] — the diagnostic tree for "my numbers don't match the paper."
