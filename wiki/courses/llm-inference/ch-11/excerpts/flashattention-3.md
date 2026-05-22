---
chapter: ch-11
course: llm-inference
phase: read
excerpt_of: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
source_url: https://arxiv.org/abs/2407.08608
created_at: "2026-05-21"
---

# Excerpt: FlashAttention-3 — Hopper async + WGMMA + FP8

**Authors:** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
**Year:** 2024
**Venue:** arXiv 2407.08608
**URL:** https://arxiv.org/abs/2407.08608
**Raw-data source:** [[raw-data/flashattention-3]]

---

## Why FA2 underuses Hopper

H100 has four features FA2 doesn't exploit:

| Feature | Speedup unlocked |
|---|---|
| TMA (async HBM↔SRAM) | hides memory latency under compute |
| WGMMA (async warp-group matmul) | overlaps successive matmuls |
| FP8 tensor cores | 2× FP16 throughput |
| Distributed shared memory | direct SM↔SM SRAM reads |

On H100, FA2 achieves ~340 TFLOPS in FP16 — only ~35% of the chip's ~1000 TFLOPS FP16 peak. The other ~65% is sitting idle waiting for kernel-internal serialization.

---

## Warp specialization (the load-bearing scheduling change)

FA3 splits warps into producer and consumer roles:

```text
Producer warps:
    TMA(K_j into SRAM)        # async memory load
    TMA(V_j into SRAM)        # async memory load
    signal mbarrier when done

Consumer warps:
    wait mbarrier
    WGMMA(S = Q @ K_j^T)      # async matmul
    softmax + scale
    WGMMA(O += P @ V_j)       # async matmul
```

Producers and consumers run concurrently. While the consumer is computing on block `j`, the producer is already fetching block `j+1`. **Memory latency is fully hidden under math.**

---

## Matmul-softmax interleaving

FA2's softmax (non-matmul) blocks the next matmul. FA3 issues the matmul async (WGMMA) and runs softmax while the matmul is in flight. Two pipelined streams: math + softmax.

This is the "ping-pong" pattern referenced in the chapter — borrowed from cuBLAS GEMM but adapted for the irregular softmax shape.

---

## FP8 attention with block scaling + incoherent processing

Naive FP8 cast loses ~1 bit because per-tile dynamic range is large and FP8's mantissa is small. FA3 applies two techniques:

**Block quantization.** Each tile of Q and K gets its own FP8 scale instead of a per-tensor scale. Reduces the effective dynamic range per tile by ~10×.

**Incoherent processing.** Pre-multiply Q and K by a random Hadamard matrix `H`:

```math
Q' = Q H, \quad K' = K H, \quad Q' (K')^\top = Q H H^\top K^\top = Q K^\top
```

Hadamard preserves dot products but smears outliers across all coordinates of the tile. After Hadamard, no single coordinate has a huge value; per-tile scale is much closer to fitting in FP8.

Net result: FP8 attention with error ~2.6× lower than naive FP8 attention, ~1.5× higher than FP16 attention. Most fine-tuned models tolerate this.

---

## Headline numbers

| Setting | H100 TFLOPS | Speedup vs FA2 |
|---|---|---|
| FA2 (FP16, H100) | ~340 | 1.0× |
| **FA3 (FP16, H100)** | **~740** | **2.2×** |
| **FA3 (FP8, H100)** | **~1200** | **3.5×** |

End-to-end Llama-3-70B prefill on H100: **~1.7× faster TTFT** vs FA2.

---

## Where FA3 doesn't apply

- **Ampere (A100/A10/RTX)**: no TMA, no WGMMA, no FP8 tensor cores. Use FA2.
- **Decode**: query length 1; producer-consumer overlap is less valuable. Use FlashDecoding.
- **Very small sequences (L < 256)**: setup overhead dominates. Use FA2 or even cuBLAS GEMM.

---

## Connections

- [[ch-11]] — parent chapter.
- [[excerpts/flashattention-2]] — the work-partitioning baseline FA3 extends.
- [[excerpts/flashinfer]] — packages FA3 as a backend dispatch target.
- [[ch-12]] / [[cuda-graphs-inference]] — FA3 wraps cleanly inside CUDA graphs because shapes are static; the two stack.
- [[ch-20]] — FP8 deployment in DeepSeek V3, GPT-OSS, Qwen3 uses FA3's FP8 techniques.
