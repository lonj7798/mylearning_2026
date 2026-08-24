# Transformer Math 101
<!-- slug: transformer-math-101 · type: doc · source: https://blog.eleuther.ai/transformer-math/ -->

**Core Insight.** Training memory has four additive buckets — parameters, optimizer states, gradients, activations — each with exact byte-per-parameter formulas, and the dominant bucket rotates between activations (long sequences, large batches) and optimizer states (small batches).

**Guideline.** Budget training memory as: `(model bytes) + (optimizer bytes) + (gradient bytes) + (activation bytes)`. For AdamW mixed-precision, the static floor is 16 bytes/param; activations blow past that with any real sequence length.

## Technical Details

- **Inference bytes/param by dtype:** int8 → 1 B, fp16/bf16 → 2 B, fp32 → 4 B.
- **Training static memory (mixed precision, AdamW):**
  - Parameters (bf16 working copy): 2 B/param
  - Parameters (fp32 master copy): 4 B/param
  - Gradients (fp32): 4 B/param
  - AdamW optimizer (fp32 momentum + variance): 8 B/param
  - **Total static: 18 B/param** (Bekman formulation; EleutherAI quotes same floor)
- **Activation memory formulas** (per-GPU, with tensor-parallel degree *t*):
  - No recomputation: `sbhL(10 + 24/t + 5·a·s/h·t)` bytes
  - Selective recomputation: `sbhL(10 + 24/t)` bytes
  - Full recomputation: `2·s·b·h·L` bytes
  (s = seq len, b = batch per GPU, h = hidden dim, L = layers, a = attention heads)
- ZeRO-1 divides optimizer memory by GPU count; ZeRO-2 also divides gradients; ZeRO-3 divides all three static components.
- **Compute-optimal recipe cited:** 20 tokens per parameter (Chinchilla); minimum 200B tokens for any LLM run.
- **Training-memory angle:** Provides the canonical per-component accounting table that lets a practitioner build a pre-flight memory estimate. The activation formula makes explicit why sequence length blows memory quadratically (attention term `5·a·s/h·t` scales as s²) while batch size scales linearly.

## Citation
Quentin Anthony et al., EleutherAI Blog, 2023. https://blog.eleuther.ai/transformer-math/
