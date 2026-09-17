<!-- excerpt for: ch-32e
     source: huggingface/smollm, text/pretraining/smollm3/*.yaml (main branch, fetched 2026-09-14; identical copies in HF dataset HuggingFaceTB/smollm3-configs)
     scope: nanotron configs for SmolLM3 pretraining stages and the two long-context stages
     library card: none as of 2026-09-15 (proposed slug smollm3-training-configs)
-->

# SmolLM3 nanotron configs — verified extract

Read on 2026-09-15. Source type: released config. The repository commit hash could not be recorded (GitHub API rate limit at fetch time); the GitHub copy and the HF-dataset copy of `long_context_4k_to_32k.yaml` were compared and are identical for the fields below.

## stage1_8T.yaml (L184, L194, L201, L206-212, L224-225, L250-255)
```yaml
    max_position_embeddings: 4096
    rope_theta: 50000.0
    no_rope_layer: 4
    learning_rate: 0.0002
    lr_decay_starting_step: 4198001
    lr_decay_steps: 522000
    lr_decay_style: linear
    lr_warmup_steps: 2000
    min_decay_lr: 0
  dp: 192
  micro_batch_size: 3
  sequence_length: 4096
  train_steps: 4720000
```
`stage3_9T_11T.yaml` repeats the same schedule fields (L518-589).

Derived (this course): tokens per step = 3 × 192 × 4096 = 2,359,296 (the blog's "2.36M"). Schedule length 4,720,000 steps = 11.14T tokens; decay starts at step 4,198,001 ≈ 9.90T tokens and lasts 522,000 steps ≈ 1.23T tokens (11.1% of steps). The blog describes the decay stage as 10T → 11.1T.

## long_context_4k_to_32k.yaml (L5-7, L228, L238, L245, L250-256, L268-269, L293-298)
```yaml
  load_lr_scheduler: true
  load_optimizer: true
  resume_checkpoint_path: s3://smollm3/tp-fix-final-pre-training/elie-lc-prolong-think-chatml-mix01
    max_position_embeddings: 32768
    rope_theta: 2000000.0
    no_rope_layer: 4
    learning_rate: 0.00002
    lr_decay_starting_step: 1000
    lr_decay_steps: 19000
    lr_decay_style: cosine
    lr_warmup_steps: 1000
    min_decay_lr: 0
  context_parallel_size: 4
  dp: 12
  batch_accumulation_per_replica: 6
  micro_batch_size: 1
  sequence_length: 32768
  train_steps: 20000
```

## long_context_32k_to_64.yaml (L5-7, L231, L241, L253-259, L271-272, L296-301)
```yaml
  load_lr_scheduler: false
  load_optimizer: false
  resume_checkpoint_path: s3://smollm3/tp-fix-final-pre-training/elie-lc-prolong-think-chatml-mix01/20000
    max_position_embeddings: 65536
    rope_theta: 5000000.0
    learning_rate: 0.00002
    lr_decay_starting_step: 1000
    lr_decay_steps: 21000
    lr_decay_style: cosine
    lr_warmup_steps: 1000
  context_parallel_size: 4
  dp: 12
  batch_accumulation_per_replica: 3
  micro_batch_size: 1
  sequence_length: 65536
  train_steps: 22000
```

Derived (this course): 4k→32k stage 32,768 × 1 × 6 × 12 = 2,359,296 tokens per step × 20,000 steps = 47.2B tokens; 32k→64k stage 65,536 × 1 × 3 × 12 = 2,359,296 × 22,000 = 51.9B tokens. The blog states 50B per stage.

## Differences from the blog
- RoPE θ in the 4k→32k stage: config 2,000,000; blog 1.5M. The blog does not say which value produced the released checkpoint.
- Both long-context configs list data weights with comments; the first stage includes rows commented as DCLM 16k and reasoning data, but no long:short token ratio is stated.
