---
chapter: ch-17
course: model-quantization
phase: read
excerpt_of: "NVIDIA Blackwell quantization architecture (B200 / B300 / GB200 / GB300, 2024-2026 product announcements) + NVFP4-QAD (NVIDIA 2026)"
source_url: https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ + https://arxiv.org/abs/2601.20088
created_at: "2026-05-21"
---

# Excerpt: Blackwell + NVFP4 in production

**Authors:** NVIDIA architecture team (Dally, Alben et al.); QAD paper authors (Xin, Priyadarshi, Han, Ping et al.)
**Year:** 2024 (Blackwell announce); 2025-2026 (production); 2026 (QAD)
**URL:** Blackwell architecture brief + https://arxiv.org/abs/2601.20088 (QAD)
**Raw-data source:** [[raw-data/blackwell-quantization]] + [[raw-data/nvfp4-qad]]

---

## The hardware that made FP4 production

Blackwell's 5th-generation Tensor Core is the first GPU tensor core to consume **NVFP4 (16-element FP4 + FP8 block scale + FP32 tensor scale)** natively. The MMA instruction takes all three operands at once; per-block scale dispatch is done in silicon, not software.

### Per-chip specs (B200, as of NVIDIA's product announcement)

| Quantity | Value |
|----------|-------|
| FP4 dense | ~10 PFLOPS |
| FP4 sparse | ~20 PFLOPS |
| FP8 dense | ~5 PFLOPS |
| FP8 sparse | ~10 PFLOPS |
| HBM3e | ~192 GB |
| Memory bandwidth | ~8 TB/s |
| NVLink-5 per GPU | 1.8 TB/s bidirectional |

### Rack-level (GB200 NVL72)

72 × B200 + 36 × Grace CPUs in one rack; ~13.4 EFLOPS FP4 sparse aggregate; ~30 TB HBM3e. Designed as one virtual GPU via NVLink-5 / NVSwitch.

### B300 / GB300 (2025-2026 product cycle)

~2× FP4 throughput vs B200; ~288 GB HBM3e per GPU; "2× attention" acceleration — attention-specific datapath improvements aimed at long-context inference where KV-cache bandwidth dominates.

---

## NVFP4 vs MXFP4 vs FP8 — production framing

Per NVIDIA's NVFP4 inference blog (as of the 2025 product announcement):

- **NVFP4 vs FP16**: 3.5× less memory.
- **NVFP4 vs FP8**: 1.8× less memory; ~2× throughput at the bandwidth limit.
- **NVFP4 quality**: < 1 % quality drop with proper calibration on representative LLMs.

Hardware-managed scaling means no software dequant overhead inside the GEMM loop — the per-block scale is consumed as a tensor-core operand.

---

## Coexisting precision tiers

Every Blackwell SM has all of these tensor-core paths:

| Format | Bits | Use case |
|--------|------|----------|
| FP32 | 32 | scale factors, accumulators |
| BF16 / FP16 | 16 | master weights, sensitive layers |
| TF32 | 19 | scientific compute |
| FP8 E4M3 / E5M2 | 8 | activations + weights (forward / backward) |
| FP6 E3M2 / E2M3 | 6 | intermediate, less common |
| **NVFP4** | **4+scale** | **production inference + pretraining target** |
| FP4 E2M1 (plain) | 4 | element format inside NVFP4 |
| MXFP4 | 4+scale | OCP-spec inference path |
| INT8 / INT4 | 8 / 4 | classic int inference, KV cache |

The second-generation Transformer Engine (TE 2.x) picks the precision per layer per call automatically.

---

## NVFP4-QAD — the inference-recovery path

For models that were *post-trained* (SFT, RL, distillation, merging) and need NVFP4 inference without re-running the entire pipeline in quant-aware mode, the production answer is **quantization-aware distillation**.

### Objective

```
L_QAD = KL(softmax(z_T / T) || softmax(z_S / T))
```

`z_T` = frozen BF16 teacher logits, `z_S` = NVFP4 student logits, `T` = temperature.

Key difference from ordinary QAT: the target is the *teacher distribution*, not the original SFT/RL task label.

### Why QAD is the production choice after RL/SFT/merge

Modern post-training pipelines combine SFT + DPO/PPO + distillation + safety data + model merging. Replaying that stack in quantization-aware mode is expensive and unstable. QAD reduces the recovery problem to *matching a frozen reference model after quantization is inserted*.

### Practical recipe

1. Start from a BF16 / high-precision post-trained teacher.
2. Insert NVFP4 quantization into the student.
3. Train on recovery data with KL-to-teacher logits.
4. Validate against BF16 on the same downstream suite, not just calibration loss.
5. Export to the serving stack with NVFP4 kernels (TRT-LLM, vLLM).

### Reported coverage (NVIDIA, 2026)

Stable recovery across:
- AceReason Nemotron
- Nemotron 3 Nano
- Nemotron Nano V2
- Nemotron Nano V2 VL (vision-language)
- Llama Nemotron Super v1

Important framing: Nemotron 3 Nano NVFP4 quantizes **both weights and activations** (W4A4), unlike weight-only MXFP4 releases like GPT-OSS. QAD is especially relevant to W4A4 deployment because activation quantization is what makes post-trained models brittle without recovery.

### Robustness

The paper reports robustness to *limited or imperfect recovery data*, because the teacher distribution carries information beyond hard labels. This is the practical reason QAD scaled to the Nemotron family without re-running each model's full training pipeline.

---

## Where NVFP4-trained vs QAD-recovered models land

| Scenario | Path |
|----------|------|
| Pretraining from scratch on Blackwell | [[nvfp4-training]] recipe (4 ingredients) |
| Pretraining checkpoint already exists, was BF16/FP8 | run QAD to produce NVFP4 deployment checkpoint |
| Already post-trained (SFT/RL/merge), need NVFP4 inference | run QAD |
| Need inference accuracy match with BF16 | QAD is the production standard |

---

## Connections

- [[nvfp4]] / ch-02 — the format spec.
- [[microscaling-formats]] / ch-16 — the OCP MX cousin Blackwell also supports.
- [[nvfp4-training]] / [[excerpts/nvfp4-training]] — the 12B / 10T pretraining recipe that needed this hardware.
- [[deepseek-v3-fp8]] — DSV3's FP8 weights can be served as-is on Blackwell or auto-converted to NVFP4 by TRT-LLM.
- [[transformer-engine]] / [[excerpts/transformer-engine]] — the software layer exposing Blackwell precision recipes.
- [[tinychat-and-tensorrt-llm-quant]] / ch-19 — TRT-LLM's NVFP4 path.
- [[llm-qat]] / ch-12 — older data-free QAT baseline; QAD is the 2026 production-oriented distillation variant.
- [[ch-17]] — parent synthesis.
