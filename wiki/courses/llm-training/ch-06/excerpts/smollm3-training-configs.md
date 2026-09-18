---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: "huggingface/smollm, SmolLM3 nanotron pre-training configs (stage1_8T.yaml, stage2_8T_9T.yaml, stage3_9T_11T.yaml, long_context_4k_to_32k.yaml)"
source_url: https://github.com/huggingface/smollm
created_at: "2026-09-17"
note: "No library card exists for the slug `smollm3-training-configs` as of 2026-09-17. All values below were read in the cached copies of the YAML files."
---

# Excerpt: SmolLM3 nanotron configs — checkpoint cadence and asynchronous in-loop evaluation

Source type: released config/code (official; Hugging Face). Model: SmolLM3-3B, trained on 11T tokens in three
pre-training stages plus a long-context stage.

## Checkpoint block (identical across the four configs)

```yaml
checkpoints:
  checkpoint_interval: 2000
  checkpoints_path_is_shared_file_system: false
  load_lr_scheduler: true
  load_optimizer: true
  resume_checkpoint_path: s3://smollm3/tp-fix-final-pre-training/1704-48n-part1
  save_final_state: true
  save_initial_state: false
```

Points to note: the checkpoint cadence is 2,000 optimizer steps; optimizer state and learning-rate-scheduler
state are separately toggled load options, so a resume can be configured to drop either one; checkpoints are
written to object storage rather than to a shared file system.

## Token and batch settings (stage1_8T.yaml, `tokens:`)

```yaml
tokens:
  batch_accumulation_per_replica: 1
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: 3
  sequence_length: 4096
  train_steps: 4720000
  val_check_interval: 100
```

`limit_val_batches: 0` means no in-loop validation-loss pass is run in this configuration, even though
`val_check_interval` is set.

## In-loop downstream evaluation is asynchronous and checkpoint-driven (`lighteval:` block)
- Stage 1 and stage 2: `eval_interval: 4000` steps. Stage 3 (decay): `eval_interval: 6000`.
  Long-context stage (`train_steps: 20000`): `eval_interval: 400000`, which disables in-loop evaluation for
  that stage.
- The evaluation runs as a separate Slurm job with its own parallelism (`dp: 4, pp: 1, tp: 2`) and
  `batch_size: 8`, reading from `local_checkpoint_dir` "Will store under {local_checkpoint_dir}/{run_name}/{step}",
  with results uploaded to Weights & Biases project `smollm3-3B-evals`.
- The task list is held in a separate file named by `eval_config_override: ".../smollm3_eval.yaml"`, which is
  not part of the cached configs.

## Verification
- Read on 2026-09-17 in the cached copies of `stage1_8T.yaml` (L1–9, L249–275), `stage2_8T_9T.yaml`,
  `stage3_9T_11T.yaml`, and `long_context_4k_to_32k.yaml`. Absolute paths inside the configs are the authors'
  cluster paths and are reproduced only where they carry information.
