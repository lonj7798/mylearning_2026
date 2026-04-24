---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/adam.md
source_url: https://arxiv.org/abs/1412.6980
created_at: "2026-04-23"
---

# Excerpt: AdamW under ZeRO-1/2/3 — Optimizer-State Sharding

**Papers:**
- Adam: A Method for Stochastic Optimization — Kingma & Ba, 2014 (arXiv: 1412.6980)
- Decoupled Weight Decay Regularization — Loshchilov & Hutter, 2017 (arXiv: 1711.05101)

**This excerpt focuses on how AdamW's optimizer state is sharded across ranks under ZeRO-1/2/3 and FSDP, and why the `12P` optimizer-state line in the FSDP memory formula dominates the arithmetic.** The single-GPU Adam / AdamW semantics are covered in ch-01.

---

## The Adam state — per-parameter cost

Quoted update rule:

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

```math
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

AdamW's only change:

```math
\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
```

**Per-parameter state.** For each scalar parameter θ, AdamW stores:

- `m` (first moment) — same dtype as master weight, typically fp32.
- `v` (second moment) — same dtype as master weight, typically fp32.
- θ* (master weight, if mixed precision) — fp32.

Total: **3 × fp32 = 12 bytes per parameter** when the forward path is bf16.

Quoted:

> "Memory cost: Adam stores `m_t` and `v_t` per parameter — 2x model parameters in optimizer state. With fp32 state and bf16 weights, optimizer state is 4x the model in bytes. This is why ZeRO-1/2/3 and 8-bit Adam exist."

This is the `12P` line in the FSDP memory table ([[excerpts/fsdp-sft]]):

| Component                        | Bytes per param |
|----------------------------------|-----------------|
| m (fp32)                         | 4               |
| v (fp32)                         | 4               |
| θ* master weight (fp32)          | 4               |
| **Total optimizer state**        | **12**          |

At 70B params: optimizer state = 70B × 12 = **840 GB**. Divided across N = 8 GPUs under ZeRO-1/2/3: 105 GB/GPU. Still too much for 80 GB; this is why ZeRO-3 is required at 70B and why activation checkpointing is not optional.

---

## ZeRO-1 / 2 / 3 — three stages of sharding

The ZeRO paper (Rajbhandari et al., 2019) introduced three increasingly aggressive sharding stages. FSDP's `ShardingStrategy` maps 1:1:

| Stage   | Shards θ? | Shards grad? | Shards opt state? | FSDP name          |
|---------|-----------|--------------|-------------------|--------------------|
| ZeRO-1  | no        | no           | yes               | (no direct FSDP eq) |
| ZeRO-2  | no        | yes          | yes               | `SHARD_GRAD_OP`    |
| ZeRO-3  | yes       | yes          | yes               | `FULL_SHARD`       |

Per-GPU memory (parameters only):

```math
\text{Mem}_{\text{ZeRO-1}} = 2P + 2P + \frac{12P}{N}
```

```math
\text{Mem}_{\text{ZeRO-2}} = 2P + \frac{2P}{N} + \frac{12P}{N}
```

```math
\text{Mem}_{\text{ZeRO-3}} = \frac{2P}{N} + \frac{2P}{N} + \frac{12P}{N}
```

For 70B on 8 GPUs (bytes, no activation):

- ZeRO-1: `140 + 140 + 105 = 385 GB/GPU` (infeasible).
- ZeRO-2: `140 + 17.5 + 105 = 262.5 GB/GPU` (infeasible).
- ZeRO-3: `17.5 + 17.5 + 105 = 140 GB/GPU` + 140 GB AllGather buffer = `280 GB/GPU` (still infeasible, needs activation ckpt).

**Notice:** the `12P / N` optimizer state dominates at small N. At N = 8, fp32 optimizer state is 105 GB/GPU — larger than the parameter shards by 6×. This is why 8-bit optimizer (bitsandbytes `AdamW8bit`) exists: it compresses m, v to 8-bit blockwise quantization, dropping the optimizer-state line by 4×. Used in QLoRA and some 70B SFT recipes.

---

## Why β₂ = 0.95 matters under sharding

Quoted:

> "Modern LLM defaults: β_2 = 0.95 (instead of 0.999). At LR-warmup completion, v_t should reflect *recent* gradient variance to track non-stationary loss landscapes. 0.999 has effective memory of ~1000 steps; 0.95 has ~20 steps and reacts faster to phase changes."

The sharding angle: `v_t` is the element-wise second moment, stored per-parameter. Under ZeRO-3, each rank owns a shard of `v_t` corresponding to its parameter shard. The update

```math
v_t^{(\text{shard } i)} = \beta_2 v_{t-1}^{(\text{shard } i)} + (1 - \beta_2) (g_t^{(\text{shard } i)})^2
```

happens entirely locally — the rank reads its own gradient shard (freshly ReduceScatter'd from the backward pass) and updates its own `v` shard. No cross-rank communication needed for the optimizer step.

**This is what makes ZeRO-3 / FSDP FULL_SHARD cheap.** The optimizer step is embarrassingly parallel across ranks; the only communication is the next forward pass's AllGather. Contrast with model-parallel optimizer (Megatron TP): the optimizer state for a TP-split linear is split across TP ranks, but the `v_t` update is still local per shard.

**Notice:** β₂ = 0.95 has effective memory of ~20 steps. If your gradient-accumulation steps × warmup ratio is less than 20, `v̂_t` is still heavily biased by the initial zero state. The bias correction `v̂_t = v_t / (1 − β_2^t)` mitigates this but amplifies noise in early steps. This is why Llama and Qwen use 3% warmup — long enough for `v_t` to stabilize before full LR.

---

## The fp32 state requirement — from [[excerpts/mixed-precision]]

Quoted:

> "Adam's `v_hat` underflows to zero in fp16 for any parameter with small gradients."

Concretely: fp16's smallest normal positive is ~6e-8. If `g_t ≈ 1e-4` (typical for warmed-up Llama), then `g_t² ≈ 1e-8` → underflows to zero. After a few thousand steps, `v_t ≈ 0` for a sizeable fraction of parameters, and the update `α · m̂ / (√v̂ + ε)` blows up by `1/ε` → NaN.

bf16 has fp32's range (~1e-38) so `v_t` underflow is not an issue. But bf16 has only 7 mantissa bits; the `v_t` update `v_t = 0.95 · v_{t-1} + 0.05 · g_t²` with `|g_t²| ≈ 1e-8` is added to `|v_{t-1}| ≈ 1e-7` — the update is 3× the ambient value, representable in bf16 with ~1% error. Accumulated over 1e6 steps, the ~1% error compounds into a measurable drift.

**Conclusion.** fp32 master `v_t` is the universal default. The FSDP memory table's `12P` line is not negotiable at frontier scale; the alternative (mixed-precision v_t) silently degrades 70B+ runs.

---

## AdamW weight decay — the distributed subtlety

Quoted:

> "AdamW (2017): decoupled weight decay; demonstrated that Adam's 'poor generalization' was an artifact of L2-as-regularizer coupling. AdamW's weight-decay coefficient lambda is independent of learning rate."

The update line:

```math
\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
```

**Under ZeRO-3:** the weight decay `λθ` term reads from the rank-local shard of θ (bf16 compute copy? no — the fp32 master). The update is computed entirely locally. No cross-rank sync needed.

**Subtle point:** `λ` is applied to the fp32 master, not the bf16 compute copy. If your framework applies weight decay to `param_dtype` tensors, you are decaying a rounded copy — the effective `λ_eff` drifts from the configured value. PyTorch's `torch.optim.AdamW` applies decay to whatever dtype the optimizer was passed; with FSDP `MixedPrecision(param_dtype=bf16)`, the optimizer sees fp32 master weights (FSDP internally converts), so this is handled correctly. But a hand-rolled "decay in forward" (as in some LoRA implementations) can be wrong.

**No-decay group — distributed impact.** Standard practice excludes LayerNorm weights, LayerNorm biases, and embedding biases from weight decay. These parameters are FSDP-sharded identically to the rest, but the optimizer maintains two parameter groups: `decay` (λ = 0.1) and `no_decay` (λ = 0). Under FSDP this means two separate iterations over the sharded parameter list at step time; correctness requires that the same partitioning was used during `FSDP(model)` construction as during optimizer construction.

---

## Pretrain vs SFT vs RL — the LR table revisited under sharding

Quoted:

| Hyperparameter | Pretrain | SFT | RL (PPO/GRPO) |
|---|---|---|---|
| β_1 | 0.9 | 0.9 | 0.9 |
| β_2 | **0.95** | 0.95–0.999 | 0.95 |
| ε | 1e-8 | 1e-8 | 1e-8 |
| λ (wd) | **0.1** | 0.0–0.01 | 0.0 |
| Peak LR | 1e-4 to 6e-4 | 1e-5 to 5e-5 | 1e-6 to 1e-5 |

Under FSDP, the peak LR is unchanged from single-GPU — the optimizer state is sharded but the update magnitude per parameter is the same. What does change is the **effective global batch size**: FSDP with N = 8 ranks, per-rank micro-batch 1, grad-accum 16 gives global batch 128.

**Notice:** the SFT peak LR of 1e-5 is calibrated for global batch ~128. Increasing N (say to 64 ranks) without reducing grad-accum gives global batch 1024, which shifts the effective LR. Standard scaling rule: LR scales with `√batch` (not linearly, which is the DDP-era rule). So for 8× the batch, multiply LR by `√8 ≈ 2.8`.

---

## The memory savings breakdown — what FSDP actually buys you

Full decomposition per component, with and without FSDP, for 70B (N = 8):

| Tensor class               | DDP (GB) | FSDP FULL_SHARD (GB)        |
|----------------------------|----------|------------------------------|
| Parameters (bf16)          | 140      | 17.5                         |
| Gradients (bf16)           | 140      | 17.5                         |
| AdamW m (fp32)             | 280      | 35                           |
| AdamW v (fp32)             | 280      | 35                           |
| Master θ* (fp32)           | 280      | 35                           |
| AllGather buffer           | 0        | 17.5 (one block × 2 bytes)   |
| **Total (per GPU)**        | **1120** | **157.5**                    |

The `12P` optimizer state is the single biggest line item — bigger than parameters and gradients combined. FSDP's win comes primarily from sharding this line; parameter and gradient sharding are secondary.

**The "ZeRO-1 if you only shard the optimizer" observation.** If your model fits replicated but optimizer state does not (a common regime for 13–30B on 8×40GB), ZeRO-1 alone gives 85% of the memory savings at 1/3 the communication cost. DeepSpeed exposes this; FSDP does not have a direct ZeRO-1 equivalent (no stage shards only optimizer in standard FSDP).

---

## Connections

- [[excerpts/fsdp-sft]] — the `12P` optimizer-state line in the memory table.
- [[excerpts/mixed-precision]] — master-weight fp32 requirement; why bf16 v_t drifts.
- [[excerpts/gradient-clipping]] — clip fires before the AdamW step on the ReduceScatter'd gradient shard.
- [[excerpts/loss-masking-prompt]] — the optimizer sees a gradient already correctly normalized by the loss contract.
- [[excerpts/sequence-packing]] — packing doesn't affect AdamW directly but changes per-step gradient statistics.
- [[ch-05]] — synthesis and the 70B SFT recipe (AdamW β=(0.9, 0.95), wd=0.0).
