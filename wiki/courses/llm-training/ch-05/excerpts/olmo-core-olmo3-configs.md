---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo3
source_type: released config/code
verified_on: "2026-09-17"
verified_against: "cached copies of src/scripts/official/OLMo3/*.py and src/scripts/official/OLMo3/README.md"
---

# Excerpt: OLMo 3 official training scripts (OLMo-core)

**Artifact:** `allenai/OLMo-core`, `src/scripts/official/OLMo3/` — the released training scripts for the
OLMo 3 7B and 32B stages, plus the directory `README.md`.
**Used by:** [[read]] §2, §3, §4, Recipe.

These are released configuration files, so every value below is a config literal, not a paper claim.
Where the OLMo 3 report ([[olmo-3]]) and these scripts could disagree, the chapter uses the scripts.

## Stage table (README.md)

| Model | Stage | Tokens | GPUs | Script |
|---|---|---|---|---|
| OLMo 3 7B | stage 1 (pretraining) | 5.93 Trillion | 512 H100s | `OLMo-3-1025-7B-pretrain-1.py`, `-2.py` |
| OLMo 3 7B | stage 2 (midtraining) | 100 Billion | 128 H100s | `OLMo-3-1025-7B-midtrain.py` |
| OLMo 3 7B | stage 3 (long-context) | 50 Billion | 256 H100s | `OLMo-3-1025-7B-long-context.py` |
| OLMo 3 32B | stage 1 (pretraining) | 5.50 Trillion | 1024 H100s | `OLMo-3-1025-32B-pretrain.py` |
| OLMo 3 32B | stage 2 (midtraining) | 100 Billion ×2 | 512 H100s | `OLMo-3-1025-32B-midtrain-ingredient-1.py`, `-2.py` |
| OLMo 3 32B | stage 3 (long-context) | 100 Billion | 1024 H100s | `OLMo-3-1025-32B-long-context.py` |

Architecture table from the same README: OLMo 3 7B — 32 layers, hidden 4096, 32 Q heads, 32 KV heads,
context length 65,536. OLMo 3 32B — 64 layers, hidden 5120, 40 Q heads, 8 KV heads, context length 65,536.

## Batch and sequence literals

| Script | `DEFAULT_SEQUENCE_LENGTH` | `GLOBAL_BATCH_SIZE` | `rank_microbatch_size` |
|---|---|---|---|
| `OLMo-3-1025-7B-pretrain-1.py` | 8192 | `8192 * 512` (comment: `~4M tokens`) | `2 * 8192` |
| `OLMo-3-1025-7B-midtrain.py` | 8192 | `2**21` (comment: `~2M tokens`) | `2 * 8192` |
| `OLMo-3-1025-7B-long-context.py` | 65536 | `65536 * 64` (comment: `~4M tokens`) | `sequence_length` |
| `OLMo-3-1025-32B-pretrain.py` | 8192 | `8 * 1024 * 1024` (comment: `~8M tokens`) | `sequence_length` |
| `OLMo-3-1025-32B-long-context.py` | 65536 | `8 * 1024 * 1024` (comment: `~8M tokens`) | `sequence_length` |

In the 7B long-context script the `rank_microbatch_size` line carries the comment
`# for CP we want only 1 instance per rank`.

## Parallelism literals

`OLMo-3-1025-7B-pretrain-1.py`:

```python
dp_config=TransformerDataParallelConfig(
    name=DataParallelType.hsdp,
    param_dtype=DType.bfloat16,
    reduce_dtype=DType.float32,
    wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
),
```

`OLMo-3-1025-32B-pretrain.py`: same `hsdp`, `wrapping_strategy=...full`, plus `shard_degree=64`, and

```python
ac_config=TransformerActivationCheckpointingConfig(
    mode=TransformerActivationCheckpointingMode.budget,
    activation_memory_budget=0.5,
),
```

`OLMo-3-1025-7B-long-context.py`:

```python
dp_config=TransformerDataParallelConfig(
    name=DataParallelType.hsdp,
    param_dtype=DType.bfloat16,
    reduce_dtype=DType.float32,
    shard_degree=1,
),
cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4),
ac_config=None,
float8_config=Float8Config(enabled=True, ao=AOFloat8LinearConfig.recommended()),
z_loss_multiplier=1e-5,
max_grad_norm=1.0,
```

`OLMo-3-1025-32B-long-context.py`:

```python
dp_config=TransformerDataParallelConfig(
    name=DataParallelType.hsdp, ..., wrapping_strategy=...full, shard_degree=8,
),
cp_config=TransformerContextParallelConfig.llama3(
    degree=8,  # 64k tokens per instance -> 8k tokens per device
    head_stride=4,
),
ac_config=TransformerActivationCheckpointingConfig(
    mode=TransformerActivationCheckpointingMode.budget,
    activation_memory_budget=0.3,
),
```

The constructor name `TransformerContextParallelConfig.llama3(...)` names the all-gather-K/V context
parallelism described in [[llama-3]] §3.3.2, not ring attention.

## Data-loader and resume literals

`OLMo-3-1025-7B-long-context.py`:

```python
dataset_config = NumpyPackedFSLDatasetConfig.from_data_mix(
    DataMix.OLMo_longmino_mix_0625,
    ...
    sequence_length=sequence_length,
    generate_doc_lengths=True,  # enables intra-document masking
    source_group_size=8,
    source_permutation_seed=123,
)

data_loader_config = NumpyDataLoaderConfig(
    global_batch_size=GLOBAL_BATCH_SIZE,
    seed=34521,
    num_workers=4,
)
```

Trainer load flags, `OLMo-3-1025-7B-long-context.py`:

```python
load_path="https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684/",
load_strategy=LoadStrategy.always,
load_trainer_state=False,
load_optim_state=True,
```

`OLMo-3-1025-7B-midtrain.py` uses the same three flags (`load_trainer_state=False`,
`load_optim_state=True`) when loading `.../OLMo25/step1413814/`.

The 32B long-context dataset config adds
`instance_filter_config=InstanceFilterConfig(repetition_max_period=13, repetition_min_period=1, repetition_max_count=32)`.

## Further literals from the same scripts (not used by this chapter's numbers)

`OLMo-3-1025-7B-long-context.py`: `LR = 0.00020712352850360292`,
`scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0)`,
`optim=SkipStepAdamWConfig(lr=LR, weight_decay=0.1, betas=(0.9, 0.95), ...)`,
`YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)`,
`max_duration=Duration.tokens(int(5e12))` with
`hard_stop=Duration.steps(int(597046))` and the comment
`# But at this step we decided to extend schedule to 7T`.

## Not present in these scripts (checked)

- No printed world-size product; the GPU counts come from the README table, and the DP degree is the
  quotient the chapter derives.
- No MFU or throughput numbers.
- No per-domain loss instrumentation settings beyond the callback list.
