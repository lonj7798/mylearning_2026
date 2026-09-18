---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "allenai/OLMo-core, src/scripts/official/OLMo3/ at commit 66f768b (2026-08-11) and its README"
source_url: https://github.com/allenai/OLMo-core/tree/66f768b/src/scripts/official/OLMo3
created_at: "2026-09-17"
note: "No library card exists for this slug as of 2026-09-17 (proposed slug olmo-core-olmo3-configs). The script values below were read at commit 66f768b on 2026-09-15 for the ch-32e excerpt of the same files; ch-58a uses the stage table and the paper-versus-config differences."
---

# Excerpt: OLMo-core official Olmo 3 base-stage scripts — stage table and config-versus-paper differences

Source type: released config/code.

## README stage table (GPU counts per stage)
- Olmo 3 Base 7B: stage 1, 5.93T tokens on 512 H100s; stage 2 (mid-training), 100B tokens on 128; stage 3 (long context),
  50B tokens on 256.
- Olmo 3 Base 32B: stage 1, 5.50T tokens on 1024; stage 2, "100 Billion x2" on 512; stage 3, 100B tokens on 1024.
- README: "we soup (with simple averaging of parameters) the outputs of two separate midtraining runs and we soup the
  final three checkpoints produced by the long-context stage."

## Script values
`OLMo-3-1025-7B-midtrain.py` L40-43, L57, L80, L105:

```python
DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.00020712352850360292
        mix=DataMix.OLMo_midtraining_mix_0625_100B,
        scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
```

`OLMo-3-1025-7B-long-context.py` L37-39, L46-51, L73-82, L100-109:

```python
DEFAULT_SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
LR = 0.00020712352850360292
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )
        generate_doc_lengths=True,  # enables intra-document masking
        scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
        cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4),
            load_path="https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684/",
            max_duration=Duration.tokens(int(5e12)),  # Originally scheduled for 5T
            hard_stop=Duration.steps(int(597046))
```

32B scripts: `OLMo-3-1025-32B-midtrain-ingredient-1.py` sets sequence 8192, `GLOBAL_BATCH_SIZE = 4 * 1024 * 1024`,
`MAX_TOKENS = 100_000_000_000`, the same LR, `DataMix.OLMo_midtraining_mix_0925_ingredient1_100B`;
`OLMo-3-1025-32B-long-context.py` sets sequence 65536, `GLOBAL_BATCH_SIZE = 8 * 1024 * 1024`, `MAX_TOKENS` 100B, YaRN
factor 8 from 8192, and `LinearWithWarmup(warmup=200, alpha_f=0.0)`.

Derived (this course): the 7B long-context `load_path` step 47684 equals 100B tokens / 2,097,152 tokens per mid-training
step (47,683.7), so the long-context stage starts at the end of the 7B mid-training run.

## Differences from the paper (arXiv:2512.13961v2 Table 35)
- 7B peak LR: 2.0712e-4 in both 7B scripts; Table 35 prints 2.074e-4 for the 7B and 2.071e-4 for the 32B.
- The 7B long-context script's `max_duration` of 5T tokens and `hard_stop` at step 597046 do not correspond to the
  50B-token stage of Table 35 (50B / 4,194,304 tokens per step is about 11,921 steps); the comments refer to the
  pre-training schedule, and the script does not state how the 50B stop was applied.
- The 32B long-context script's `load_path` points to the pre-training checkpoint `stego32-highlr-filter3/step656000`,
  with the merged mid-training checkpoint path commented out (L113-114), while the paper states the extension starts from
  the merged mid-trained 32B model (§3.5.4, §3.6).

## Verification
- Read at commit 66f768b on 2026-09-15 for ch-32e; reassembled for ch-58a on 2026-09-17 without changing a value or locus.
- Not reported in the scripts: wall-clock or GPU-hour totals; the token budget actually consumed by the long-context run.
