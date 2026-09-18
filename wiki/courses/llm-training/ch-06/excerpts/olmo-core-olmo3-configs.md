---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: "allenai/OLMo-core, src/scripts/official/OLMo3/ at commit 66f768b (2026-08-11), plus src/olmo_core/train/trainer.py and src/olmo_core/data/data_loader.py at the same commit"
source_url: https://github.com/allenai/OLMo-core/tree/66f768b/src/scripts/official/OLMo3
created_at: "2026-09-17"
note: "No library card exists for the slug `olmo-core-olmo3-configs` as of 2026-09-17. All code below was read in the cached copies of those files."
---

# Excerpt: OLMo-core — checkpoint state, resume behaviour, and in-loop evaluation callbacks

Source type: released config/code (official; Allen Institute for AI).

## What the trainer saves that is not model or optimizer state
`src/olmo_core/train/trainer.py` L87–96 and L787–801:

```python
class TrainerStateDict(TypedDict):
    global_step: int
    global_train_tokens_seen: int
    global_train_petaflops: float
    max_steps: Optional[int]
    data_loader: Dict[str, Any]
    epoch: int
    world_size: int
    rng: Dict[str, Any]
    callbacks: Dict[str, Dict[str, Any]]
```

Model and optimizer state are saved separately, by handing the train module to the checkpointer
(`trainer.py` L977: `self.checkpointer.save(path, self.train_module, cast(Dict[str, Any], self.state_dict()), ...)`).

## Resume behaviour that is conditional, not automatic
`trainer.py` L837–849 — RNG is restored only when the world size matches:

```python
if state_dict["world_size"] == get_world_size():  # global world size here on purpose
    rng_state = EnvRngStates.from_dict(state_dict["rng"])
    if not rng_state.restore():
        log.warning("Some RNG states were not restored due to differences in library versions")
else:
    log.warning(
        "Trainer will not restore rank RNG states since the RNG states in the checkpoint "
        "were saved with a different world size."
    )
```

## Data-loader state and the fingerprint guard
`src/olmo_core/data/data_loader.py` L441–467:

```python
def state_dict(self) -> Dict[str, Any]:
    return {
        "dataset_fingerprint_version": self.dataset.fingerprint_version,
        "dataset_fingerprint": self.dataset.fingerprint,
        "batches_processed": self.batches_processed,
        "tokens_processed": self.tokens_processed,
        "seed": self.seed,
        "epoch": self._epoch,
    }
```

`load_state_dict` raises `RuntimeError` when `dataset_fingerprint` differs, with the message "Dataset
fingerprint does not match the fingerprint in the checkpoint, set ignore_fingerprint_mismatch=True to ignore
this error. This will probably result in a different data order!" A differing `seed` produces a warning only.
The FSL subclass recomputes position from tokens rather than batches (L757–764):
`self.batches_processed = self.tokens_processed // self.global_batch_size`.

## Checkpoint and evaluation cadence in the official Olmo 3 7B scripts
`OLMo-3-1025-7B-pretrain-1.py` and `-pretrain-2.py`:
- `CheckpointerCallback(save_interval=1000, ephemeral_save_interval=None, save_async=False)`
- `LMEvaluatorCallbackConfig(eval_dataset=... DataMix.v3_small_ppl_validation ..., eval_interval=10_000)`
- `DownstreamEvaluatorCallbackConfig(tasks=[...], eval_interval=10_000)` with a 26-task "fast" set:
  OLMES subsets (`arc_challenge_test_bpb_5shot`, `arc_challenge_test_mc_5shot_fast`, `arc_easy_*`,
  `hellaswag_bpb_5shot`, the four `mmlu_*_test_bpb_5shot` and `mmlu_*_test_mc_5shot_fast` splits),
  six `basic_skills_*_rc_5shot` tasks, generative tasks scored in bits-per-byte
  (`codex_humaneval_gold_bpb_3shot`, `codex_mbpp_gold_bpb_3shot`, `minerva_math_500_gold_bpb_0shot`,
  `mt_mbpp_{cpp,java,rust}_gold_bpb_3shot`), and `copycolors_10way_fast`, commented in the file as
  "Sanity check for MCQA ability".
- `TrainerConfig(..., metrics_collect_interval=10, cancel_check_interval=10)`.

## A resume that deliberately changes the schedule
`OLMo-3-1025-7B-pretrain-2.py` L70–87 and L104–111:

```python
    # Scheduler settings from the first half
    original_warmup_steps = 2000
    original_max_steps = int(5e12) // GLOBAL_BATCH_SIZE
    ...
        scheduler=HalfCosWithWarmup(  # Scheduler updated to extend lr from where we left off.
            warmup_steps=original_max_steps // 2 + original_warmup_steps // 2
        ),
    ...
            load_path="https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step596047/",
            load_strategy=LoadStrategy.always,
            load_trainer_state=True,
            load_optim_state=True,
            max_duration=Duration.tokens(int(7e12)),  # Changed from 5T -> 7T
```

`pretrain-1.py` used `CosWithWarmup(warmup_steps=2000)` and `max_duration=Duration.tokens(int(5e12))`.
Both scripts use `SkipStepAdamWConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))` with weight decay 0.0 on
`embeddings.weight`, and `GLOBAL_BATCH_SIZE = 8192 * 512` (about 4M tokens).

## README (src/scripts/official/OLMo3/README.md)
- Olmo 3 7B stages: stage 1, 5.93T tokens on 512 H100s; stage 2 (midtraining), 100B tokens on 128 H100s;
  stage 3 (long context), 50B tokens on 256 H100s. Olmo 3 32B: 5.50T on 1024; "100 Billion x2" on 512;
  100B on 1024.
- On the 32B: "Unlike for Olmo 3 7B, we use model merging ("souping") at multiple points during pretraining. In
  particular, we soup (with simple averaging of parameters) the outputs of two separate midtraining runs and we
  soup the final three checkpoints produced by the long-context stage."

## Verification
- Read on 2026-09-17 in the cached copies of the files at commit 66f768b. Line numbers are those of the cached
  files and may shift in later commits.
