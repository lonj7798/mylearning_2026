<!-- scope: Megatron-LM FP8 training integration
     deps: [[fp8-e4m3]], [[fp8-e5m2]]
     see-also: [[transformer-engine-fp8]], [[transformer-engine-blog]]
-->

# Megatron-LM — FP8 Training Integration
- **Core Insight:** Megatron-LM integrates FP8 training by replacing its tensor-parallel `ColumnParallelLinear` / `RowParallelLinear` modules with TransformerEngine FP8 equivalents and orchestrating amax reduction across the TP+DP+PP mesh.
- **Guideline:** Enable FP8 with `--fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max`; TP-aware amax all-reduce is mandatory or you'll desync FP8 scales across ranks.
- **Authors:** NVIDIA Megatron-LM team
- **Year:** 2023 (initial FP8 support); ongoing
- **URL:** https://github.com/NVIDIA/Megatron-LM
- **Relevant topics:** FP8, tensor parallelism, pipeline parallelism, amax reduction

## Summary
Megatron-LM is NVIDIA's reference distributed training framework for large transformer models (GPT-3/Bloom/Llama/Nemotron). Its FP8 integration is essentially a thin layer over Transformer Engine: when `--transformer-impl transformer_engine` and `--fp8-format hybrid` flags are set, the standard `ParallelTransformerLayer` is replaced by `te.TransformerLayer`, and the tensor-parallel `Linear` modules use TE's `Float8Tensor` machinery. The framework's contribution is correctly handling the distributed scale-state synchronization: amax buffers must be all-reduced across the DP+TP+PP mesh before the next scale update, because tensor-parallel matrix slices see different per-rank amax values. Megatron-Core (the modular refactor) further cleans up the FP8 integration via the `ModelParallelConfig` and `TransformerConfig` interfaces.

## Key Points
- FP8 layers come from TransformerEngine; Megatron orchestrates them.
- Tensor-parallel amax must be all-reduced inside each TP group.
- Pipeline-parallel doesn't need amax communication (each PP stage has independent scales).
- `--fp8-format hybrid`: E4M3 fwd, E5M2 bwd (matches TE default).
- FP8 weight checkpoints saved as FP32 master + amax history (so resumes pick up the same scale).
- Compatible with sequence parallelism and selective activation recomputation.

## Technical Details

### Repository layout
- repo: `https://github.com/NVIDIA/Megatron-LM`
- FP8 integration: `megatron/training/arguments.py` (`--fp8-format`, etc.) + `megatron/training/training.py` (`setup_model_and_optimizer`)
- TE wrapping: `megatron/core/transformer/transformer_layer.py` + `megatron/core/transformer/transformer_block.py` (when `transformer_impl="transformer_engine"`)
- TE Linear wrappers: `megatron/core/extensions/transformer_engine.py` (`TEColumnParallelLinear`, `TERowParallelLinear`, `TENorm`, `TEDotProductAttention`)
- FP8 utility: `megatron/training/utils.py` and `megatron/core/fp8_utils.py` (amax sync helpers)

### Tensor-parallel FP8 linear wrappers
```python
# megatron/core/extensions/transformer_engine.py
class TEColumnParallelLinear(te.Linear):
    def __init__(self, in_features, out_features, *,
                 tp_group, parallel_mode="column", **kwargs):
        super().__init__(
            in_features, out_features,
            tp_group=tp_group, parallel_mode=parallel_mode,
            sequence_parallel=cfg.sequence_parallel, ...
        )
```

### Amax synchronization rule
```python
# Pseudocode (combined from megatron core + TE)
# After each forward/backward:
amax_local = each rank's current FP8 amax buffer
amax_global = all_reduce(amax_local, op=MAX, group=tp_group)
# DP all-reduce of amax happens inside TE's reduce_amax helper
amax_dp = all_reduce(amax_global, op=MAX, group=dp_group)
# Update scale from amax_dp
```

### Key APIs (user-facing CLI)
- `--transformer-impl transformer_engine` — use TE layers (required for FP8).
- `--fp8-format hybrid` — E4M3 forward, E5M2 backward.
- `--fp8-amax-history-len 1024` — amax history depth.
- `--fp8-amax-compute-algo max` — `max` or `most_recent`.
- `--fp8-wgrad` — use FP8 for weight gradients (requires hopper).
- `--fp8-margin 0` — extra safety margin in scale.
- `--bf16` — master dtype (FP8 only quantizes the GEMM operands; master weights stay BF16).

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `fp8` | None | "e4m3", "e5m2", "hybrid" |
| `fp8_format` | "hybrid" | shorthand for fwd-E4M3 + bwd-E5M2 |
| `fp8_amax_history_len` | 1024 | TE amax history depth |
| `fp8_amax_compute_algo` | "max" | also "most_recent" |
| `first_last_layers_bf16` | False | keep first/last layers in BF16 (stability) |
| `fp8_param` | False | also store params in FP8 (saves memory but tricky) |

### Production runs (publicly disclosed)
- **Nemotron-4 340B** (NVIDIA, 2024): trained with Megatron + TE FP8 hybrid, ~9 trillion tokens.
- **Llama-3 405B**: BF16 (Meta did not use FP8 for the base run).
- **DeepSeek-V3**: uses a DeepSeek-internal fork with custom block-wise FP8 (see `deepseek-v3-fp8.md`).

### Common pitfalls
- Tensor-parallel without amax all-reduce: ranks compute different scales → numerical divergence.
- `--num-layers-per-virtual-pipeline-stage` with FP8: requires extra synchronization at PP boundaries.
- Activation recomputation interaction: TE handles this if `--recompute-granularity selective`.

## Connections
- [[transformer-engine-fp8]] — supplies the FP8 modules Megatron wraps.
- [[transformer-engine-blog]] — practitioner overview.
- [[fp8-e4m3]] / [[fp8-e5m2]] — operand formats.
- [[deepseek-quant]] — DeepSeek uses a forked Megatron-style stack with custom FP8.
