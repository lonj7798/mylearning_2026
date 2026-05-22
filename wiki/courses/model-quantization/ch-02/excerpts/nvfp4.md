---
chapter: ch-02
course: model-quantization
phase: read
excerpt_of: "NVFP4 — NVIDIA Blackwell-Era 4-bit Float (2024)"
source_url: https://developer.nvidia.com/blog/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing/
created_at: "2026-05-21"
raw_data_source: [[raw-data/nvfp4]]
---

# Excerpt: NVFP4 — two-level block-scaled FP4 on Blackwell

**Authors:** NVIDIA (introduced with Blackwell architecture, 2024).
**Year:** 2024 (Blackwell announcement); 2025 (production in TensorRT-LLM, vLLM).
**URLs:** Blackwell blog — see source_url; Transformer Engine API https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html

---

## The one-box structure

```
NVFP4 tensor = (s_tensor, [(S_b, [P_{b,0}, P_{b,1}, ..., P_{b,15}]) for b in blocks])
```

- `s_tensor`: **FP32** per-tensor scale; one value for the whole tensor.
- `S_b`: **FP8 E4M3** block scale; one per 16-element block.
- `P_{b,i}`: **E2M1** element value; 16 per block.

**Decoded value:**

```math
\text{value}(P_{b,i}) \,=\, s_{\text{tensor}} \cdot \text{decode\_e4m3}(S_b) \cdot \text{decode\_e2m1}(P_{b,i})
```

---

## Two key differences from MXFP4

| | MXFP4 | NVFP4 |
|---|---|---|
| Element format | E2M1 | E2M1 |
| Block size | 32 | **16** |
| Block scale | E8M0 (8b, exponent-only) | **FP8 E4M3 (8b, exp + mantissa)** |
| Per-tensor scale | none | **FP32** |
| Effective bits/elem | 4.25 | **4.5** |
| Quality (Llama-7B PPL gap from BF16) | ~0.3 | ~0.15 |
| Hardware | Blackwell, MI355X, Gaudi 3 | **Blackwell only** |
| Standardization | OCP | NVIDIA-proprietary |

---

## Why finer block + mantissa-bearing scale wins

- **Block 16 vs 32:** outliers in a 16-element neighborhood are less likely to dominate a smaller block's scale; halves the "outlier blast radius."
- **E4M3 vs E8M0 scale:** E8M0 can only express scales of the form `2^k` → a block whose true max-magnitude is 1.4 must round its scale up to 2 (wasting half the FP4 range) or down to 1 (clipping 40% of the block). E4M3's mantissa lets you store scale = 1.5 → **no rounding loss on the scale itself**.

---

## Bit budget

```math
\text{effective bits/element} \,=\, 4 \,+\, \frac{8}{16} \,+\, \frac{32}{N_{\text{tensor}}} \,\approx\, 4.5
```

vs MXFP4: `4 + 8/32 = 4.25`. NVFP4 trades 0.25 extra bits for ~0.15 PPL of quality.

---

## Use in training (FP4 native pretraining)

- **Forward**: weights + activations in NVFP4; tensor-core matmul in FP4 with FP32 accumulator.
- **Backward**: gradients in NVFP4 (or E5M2-element NVFP4 variant for wider gradient range).
- **Master weights**: FP32.
- **Stochastic rounding** ([[excerpts/stochastic-rounding]]) on the weight update — preserves expectation.

See [[nvfp4-training]] in ch-17 for the full recipe.

---

## Use in inference

- **W4A4 (weights and activations both NVFP4):** production target for serving on Blackwell; ~2× throughput over FP8.
- **W4A16:** weights NVFP4, activations BF16; backward-compatibility with non-Blackwell deployments.
- **KV-cache in NVFP4:** 4× memory savings over FP16 KV cache; viable with per-token / per-channel partitioning.

---

## Hardware support

- **NVIDIA B100 / B200 (Blackwell)**: native NVFP4 tensor cores at ~9 PFLOPS dense, ~18 PFLOPS sparse.
- **TensorRT-LLM**: NVFP4 inference path with Transformer Engine.
- **vLLM**: NVFP4 weight loading + Marlin-style kernels announced.
- **Pre-Blackwell**: NVFP4 storage works but requires dequant-to-BF16 emulation.

---

## Limitations

- **Proprietary** → not portable to AMD / Intel hardware (those use MXFP4).
- 0.25 extra bits/element vs MXFP4 → marginal memory cost.
- Two-level scale dequant logic is slightly more complex than MX's single-level.

---

## Connections

- [[excerpts/fp4-e2m1]] — the underlying element format.
- [[mx-formats]] — the cross-vendor competing standard.
- [[excerpts/fp8-e4m3]] — used as the NVFP4 block scale.
- [[nvfp4-training]] — production FP4 pretraining recipes (ch-17).
- [[transformer-engine]] — NVIDIA's FP4 / FP8 software stack.
- [[ch-02]] — parent synthesis.
