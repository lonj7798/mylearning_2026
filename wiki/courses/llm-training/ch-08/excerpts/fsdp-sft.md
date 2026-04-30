---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fsdp-sft.md
source_url: https://arxiv.org/abs/2304.11277
created_at: "2026-04-23"
---

# Excerpt: FSDP — the sharding contract the lab config inherits

**Source library:** `wiki/raw-data/llm-training/papers/fsdp-sft.md`
**Paper:** Zhao et al. 2023, "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"

---

## Why this source anchors ch-08 §Full-budget path and §5

Ch-08's full-budget lab config sets `fsdp="full_shard auto_wrap"` with `transformer_layer_cls_to_wrap=["LlamaDecoderLayer"]`. Those two strings encode a specific numerical recipe whose guarantees — sharded optimizer state in fp32, sharded-aware grad clipping, per-block AllGather — come from this paper. The lab's gradient-clipping silent-failure discussion (ordering line, green band) reduces to "does Accelerate's dispatch actually hit FSDP's clip_grad_norm_"; this excerpt is where that contract is defined.

---

## The recipe the lab copies — verbatim

From the source (lines 70-81), the "Typical SFT recipe (70B)" block:

| Knob | Value |
|------|-------|
| Strategy | FULL_SHARD |
| Precision | BF16 params + BF16 reduce + FP32 optim master |
| Activation checkpointing | per transformer block |
| Micro-batch per GPU | 1 |
| Gradient accumulation | 16 |
| Packing | yes |
| Max seq length | 4096 |
| Learning rate | 1e-5, cosine, warmup 3% |
| Optimizer | AdamW β = (0.9, 0.95) |

Ch-08's §Full-budget path is a direct instantiation of this, with one SFT-vs-pretraining delta: LR = 2e-5 (from the Zephyr-7B recipe in [[hf-alignment-handbook]]) rather than 1e-5, and warmup = 10% rather than 3% (because the lab run is very short and a too-fast warmup spikes the first grad-norms). Every other row survives unchanged.

---

## The sharded `clip_grad_norm_` contract — why TRL dispatches correctly

From the source (line 44 context — the distributed-training pitfall discussion):

Under `FULL_SHARD`, each rank owns a disjoint slice of the parameter tensor. A naive `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` computes the L2 norm of *local* shards only, then rescales locally — different ranks see different norms, the rescale is inconsistent, and the effective step across ranks is no longer a uniform rescale. Direction is no longer preserved. Silent.

FSDP ships `FullyShardedDataParallel.clip_grad_norm_` which performs:

```
local_norm_sq = sum ||g_i||^2 over this rank's shards
global_norm_sq = all_reduce(local_norm_sq, op=SUM)
global_norm = sqrt(global_norm_sq)
if global_norm > max_norm:
    scale = max_norm / global_norm
    for g_i on this rank: g_i *= scale      # same scale on every rank
return global_norm                           # pre-clip, for logging
```

Accelerate's `accelerator.clip_grad_norm_` introspects the model: if it is `FSDP`-wrapped it dispatches to the method above; otherwise it falls through to the stock `nn.utils` version. HF `Trainer` calls `accelerator.clip_grad_norm_`. TRL's `SFTTrainer` inherits this path unchanged.

Ch-08's memo picks (§Deliverables failure-mode §1) flag the inverse case: if a learner monkey-patches clipping in a custom callback, they typically reach for `nn.utils.clip_grad_norm_` and bypass the FSDP dispatch. The silent-failure signature is: grad-norm log looks suspiciously local (varies per rank even though loss is global), training diverges ~1000 steps in.

---

## The `MixedPrecision` clause — why the lab keeps optimizer state fp32

From the source (lines 63-67):

```
MixedPrecision(param_dtype=torch.bfloat16,
               reduce_dtype=torch.bfloat16,
               buffer_dtype=torch.bfloat16)
```

> Keeps FP32 master weights in the optimizer, BF16 everywhere else — combined with ZeRO sharding this is the standard modern SFT setup.

Ch-08 inherits this default through `bf16=True` in the HF `TrainingArguments`. The attested consequence: optimizer state (AdamW `m`, `v`, master copy) stays fp32 — three 4-byte tensors per parameter — which is why the "12P" row in the memory table exists. The lab doesn't *do* anything with this directly but it matters for the memo: "what you would instrument next" should include per-layer weight-norm tracking precisely because bf16 params + fp32 master means small drift across a resume is possible and is invisible without a norm log.

---

## The wrap policy — the one line that determines save granularity

From the source (line 59):

> `auto_wrap_policy = transformer_auto_wrap_policy({LlamaDecoderLayer})`
> Each transformer block becomes an FSDP unit → AllGather only per-block params, not the whole model.

Ch-08's config sets this via `fsdp_config={"transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]}`. Replace `LlamaDecoderLayer` with the correct class for your model: `Qwen2DecoderLayer`, `MistralDecoderLayer`, etc. HF `Trainer` does not auto-detect — passing the wrong class name silently wraps nothing (the entire model becomes one unit) and AllGather fires once per step for the whole parameter set. Throughput collapses by ~5× at 7B, which the learner will notice; but on a 135M model the throughput delta is small enough to miss.

Memo line: "class-name wrap-policy mismatch" is a top-three silent-failure candidate if the learner swapped base models mid-lab.

---

## The memory formula's implication for the resource-constrained path

From the source (lines 29-41):

For `P = 125M` (SmolLM-135M): DDP steady ≈ 16P = 2 GB, FSDP FULL_SHARD on 1 GPU ≈ (16P / 1) + 2P = 18P = 2.25 GB. On a 135M model, FSDP's per-GPU benefit is *negative* — you pay the 2P AllGather buffer and get zero shard reduction. Hence ch-08's resource-constrained path explicitly says "Drop FSDP to DDP (`--num_processes=1`) or FSDP-1-node if you want to exercise the sharding path." DDP is the correct default at 135M; FSDP is exercised at that scale only as a code-path dry-run, not for memory reasons.

---

## What this paper does not cover — the ch-08 gap

The paper covers compute sharding: parameters, gradients, optimizer state. It explicitly does *not* cover:

- Per-rank RNG state (ch-06 §1 item 6).
- Dataloader iterator position (ch-06 §5.1).
- LR scheduler counter (ch-06 §5.3).
- Loss-scaler state under fp16 (ch-06 §5.2).

Ch-08's concept-mapping §4–§7 leaves these to the HF `Trainer` parent class and the DCP save path; ch-06 already covered them. The point is that the FSDP paper defines *one* layer of the lab's contract; the full contract requires HF + Accelerate + DCP + the lab's unit tests, all composed.

---

## Connections

- [[excerpts/gradient-clipping]] — the global-norm clipping contract FSDP's method implements correctly.
- [[excerpts/hf-alignment-handbook]] — where `fsdp="full_shard auto_wrap"` lands in a real SFT config.
- [[excerpts/sequence-packing]] — packing + FSDP is the 2024–25 SFT backbone; both papers must be held jointly.
- [[ch-05]] — full FSDP mechanics chapter.
- [[ch-06]] — checkpointing; DCP is the only correct save path under FULL_SHARD.
- [[ch-08]] — §Full-budget path (the recipe), §5 (grad clipping silent failure), figures/trainer-map.html ("FSDP wrap" node).
