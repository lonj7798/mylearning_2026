# Per-GPU Memory Formula Assembly: Full-FT ZeRO-3 vs LoRA
<!-- slug: memory-calculator-notes · type: report · source: https://nanotron-ultrascale-playbook.static.hf.space + ultrascale-playbook.md + zero-memory-optimization + megatron-tp-sp -->

**Core Insight.** Peak GPU memory during training has five additive components — weights (W), optimizer states (O), gradients (G), activations (A), and logit-buffer spike (L) — and they do not all peak simultaneously. ZeRO-3 divides W+G+O by N; activations and the logit spike are independent of N and must be budgeted separately.

**Guideline.** Before any run, compute the estimate: `M = (16Ψ/N) + A_per_layer×L_layers + L_spike + overhead`. If M > GPU_capacity, the ordered levers are: (1) increase N (more GPUs, ZeRO-3), (2) enable activation checkpointing (halves A), (3) reduce batch/sequence, (4) switch LoRA (removes G+O for frozen params). Never reduce ZeRO stage without first verifying that the freed communication overhead is actually the bottleneck.

## Technical Details

**Full fine-tune with ZeRO-3 (per GPU):**

| Component | Formula | Notes |
|-----------|---------|-------|
| Weights (bf16) | 2Ψ/N | sharded across N DP ranks |
| Gradients (bf16) | 2Ψ/N | reduce-scattered |
| Optimizer states (fp32 Adam) | 12Ψ/N | master param 4Ψ + m 4Ψ + v 4Ψ, all sharded |
| **Model-state subtotal** | **16Ψ/N** | |
| Activations (per layer, no checkpointing) | `(sbh/t)(34 + 5as/h)` bytes | s=seq, b=batch, h=hidden, a=heads, t=TP degree |
| Activations (with checkpointing) | ~1 tensor per layer boundary | recompute on backward; exact cost depends on checkpoint granularity |
| AllGather buffer (FSDP peak) | ≈ 2 × P_unit | transient peak during forward/backward of largest unit |
| **Logit-buffer spike** | `vocab_size × seq_len × batch × 2` bytes | fp16 logits computed once per forward-backward boundary; must reserve this even if model fits comfortably otherwise |
| Miscellaneous overhead | ~1–2 GB | CUDA context, framework allocator fragmentation |

**Full-FT ZeRO-3 example: 7B model, N=8, s=2048, b=1, h=4096, a=32, t=1, vocab=32000:**
- Model states: 16 × 7×10⁹ / 8 ≈ **14 GB**
- Activations (no ckpt, 32 layers): (2048×1×4096/1) × (34 + 5×32×2048/4096) × 32 bytes × 32 ≈ **16 GB**
- Logit spike: 32000 × 2048 × 1 × 2 ≈ **0.13 GB**
- AllGather buffer peak: ≈ 2 × (7B/32 layers × 2) ≈ **0.44 GB**
- **Estimated total: ~31 GB** (fits on 2× A100 40 GB with headroom; tight on 1× without ckpt)

**LoRA memory (frozen base model):**

| Component | Formula | Notes |
|-----------|---------|-------|
| Frozen weights (bf16) | 2Ψ | not sharded by default (loaded fully); ZeRO-3 can still shard |
| LoRA adapter weights | ≈ 2rΨ_target / d | tiny; r=rank, Ψ_target = targeted param count |
| Adapter gradients + optimizer states | ≈ 14r×Ψ_target / d | only adapters have optimizer states |
| **Activations** | same as full-FT | frozen base forward still caches activations for adapter backward |
| Logit spike | same as full-FT | world-invariant |

- Key LoRA insight: **activations dominate** LoRA memory. "The main memory consumption during LoRA fine-tuning comes from activation gradients in the frozen weights rather than the LoRA parameters." LoRA optimizer states are negligible; activation budget is identical to full fine-tune at the same s and b.
- **World-invariant activation/logit:** Unlike model states (divided by N), activations per GPU do not decrease as you add more DP replicas — each replica processes the same tokens. Only TP (t) reduces activations.

- **Phase-boundary peak:** The maximum memory occurs at the seam of the forward and backward passes: all layer activations are cached (forward peak), the logit buffer is allocated, and the first backward AllGather is triggered before any activation is freed. Budget accordingly.

**Training-memory angle:** This formula is the practitioner's single worksheet — all five terms must fit, and they peak at different times. ZeRO-3 attacks the 16Ψ term; activation checkpointing attacks A; TP attacks both weight compute and A; but nothing attacks the logit spike except reducing vocab size or sequence length.

## Citation
Synthesized from: Rajbhandari et al. ZeRO 2020 (arxiv 1910.02054); Korthikanti et al. 2022 (arxiv 2205.05198); HuggingFace Ultra-Scale Playbook (https://nanotron-ultrascale-playbook.static.hf.space); HuggingFace Transformers Memory Anatomy (https://huggingface.co/docs/transformers/model_memory_anatomy); LoRA-FA (arxiv 2308.03303).
