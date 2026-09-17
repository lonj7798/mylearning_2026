<!-- excerpt for: ch-32e
     source: allenai/OLMo-core, src/scripts/official/OLMo3/ (commit 66f768b, 2026-08-11) and its README
     scope: released midtraining and long-context training scripts for Olmo 3 Base 7B and 32B
     library card: none as of 2026-09-15 (proposed slug olmo-core-olmo3-configs)
-->

# OLMo-core official Olmo 3 scripts — verified extract

Read on 2026-09-15 from the script files at commit 66f768b (the commit changed only checkpoint URLs). Source type: released config/code.

## OLMo-3-1025-7B-long-context.py (lines 37-39, 46-51, 53-62, 73-82, 89, 100-109)
```python
DEFAULT_SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
LR = 0.00020712352850360292
...
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )
    dataset_config = NumpyPackedFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_longmino_mix_0625,
        ...
        generate_doc_lengths=True,  # enables intra-document masking
...
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
...
        cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4),
...
            load_path="https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684/",
            ...
            max_duration=Duration.tokens(int(5e12)),  # Originally scheduled for 5T
            hard_stop=Duration.steps(  # But at this step we decided to extend schedule to 7T. See OLMo3-7B-second-half.py
                int(597046)
```

## OLMo-3-1025-7B-midtrain.py (lines 40-43, 57, 80, 105)
```python
DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.00020712352850360292
        mix=DataMix.OLMo_midtraining_mix_0625_100B,
        scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
            max_duration=Duration.tokens(MAX_TOKENS),
```

## 32B scripts
- `OLMo-3-1025-32B-midtrain-ingredient-1.py` L41-44, L58, L84: sequence 8192; `GLOBAL_BATCH_SIZE = 4 * 1024 * 1024`; `MAX_TOKENS = 100_000_000_000`; `LR = 0.0002071235285`; `DataMix.OLMo_midtraining_mix_0925_ingredient1_100B`; `LinearWithWarmup(warmup=0, alpha_f=0.0)`. Ingredient 2 uses the ingredient-2 mix.
- `OLMo-3-1025-32B-long-context.py` L41-44, L55, L59, L87, L120: sequence 65536; `GLOBAL_BATCH_SIZE = 8 * 1024 * 1024`; `MAX_TOKENS = 100_000_000_000`; same LR; YaRN factor 8 from 8192; `DataMix.OLMo_longmino_mix_0925`; `LinearWithWarmup(warmup=200, alpha_f=0.0)`.

## README stage table
7B: stage 1 5.93T on 512 H100s; stage 2 100B on 128; stage 3 50B on 256. 32B: stage 1 5.50T on 1024; stage 2 "100 Billion x2" on 512; stage 3 100B on 1024. "we soup (with simple averaging of parameters) the outputs of two separate midtraining runs and we soup the final three checkpoints produced by the long-context stage."

Derived (this course): the 7B long-context `load_path` step 47684 equals 100B tokens / 2,097,152 tokens per midtraining step (47,683.7), which matches starting from the end of the 7B midtraining run.

## Differences from the paper (arXiv:2512.13961v2 Table 35)
- 7B peak LR: script 2.0712e-4 in both 7B scripts; Table 35 prints 2.074e-4 for 7B and 2.071e-4 for 32B.
- 7B long-context script: `max_duration` 5T tokens and `hard_stop` step 597046 do not correspond to the 50B-token stage in Table 35 (50B / 4,194,304 tokens per step is about 11,921 steps); the comments refer to the pretraining schedule. The script does not state how the 50B stop was applied.
- 32B long-context script: `load_path` points to the pretraining checkpoint `stego32-highlr-filter3/step656000`, and the merged midtraining checkpoint path is commented out (L113-114). The paper states the extension starts from the merged midtrained 32B model (§3.5.4, §3.6).
