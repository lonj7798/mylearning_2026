---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "allenai/OLMo-core src/scripts/official/OLMo3 (7B pretrain-1, pretrain-2, midtrain; 32B pretrain, midtrain ingredient 1) and src/olmo_core/optim/scheduler.py, src/olmo_core/train/trainer.py at commit 66f768b223171b5fde71cf67ccefa843ab9c51cb (no library card as of 2026-09-15; chapter-local verified extract)"
source_url: https://github.com/allenai/OLMo-core/tree/66f768b223171b5fde71cf67ccefa843ab9c51cb/src/scripts/official/OLMo3
created_at: "2026-09-15"
---

# Excerpt: OLMo-core official Olmo 3 scripts — schedule as configured

Source type: released training scripts and framework code. Commit 66f768b (2026-08-11) is the latest commit touching the OLMo3 script directory; the `main` copies fetched on 2026-09-14 and the pinned copies of pretrain-2 and midtrain are byte-identical.

## OLMo-3-1025-7B-pretrain-1.py
```python
# L40-42
DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8192 * 512  # ~4M tokens
LR = 3e-4
# L72-80
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),
# L89-90
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
# L99-102
            max_duration=Duration.tokens(int(5e12)),  # Originally scheduled for 5T
            hard_stop=Duration.steps(  # But at this step we decided to extend schedule to 7T. See OLMo-3-1025-7B-pretrain-2.py
                int(597046)
            ),
# L108
                save_interval=1000,
```

## OLMo-3-1025-7B-pretrain-2.py
```python
# L71-72
    original_warmup_steps = 2000
    original_max_steps = int(5e12) // GLOBAL_BATCH_SIZE
# L85-87
        scheduler=HalfCosWithWarmup(  # Scheduler updated to extend lr from where we left off.
            warmup_steps=original_max_steps // 2 + original_warmup_steps // 2
        ),
# L104-111
            load_path="https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step596047/",
            load_strategy=LoadStrategy.always,
            load_trainer_state=True,
            load_optim_state=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(int(7e12)),  # Changed from 5T -> 7T
            hard_stop=Duration.epochs(1),
```

## Midtrain and 32B scripts
- 7B midtrain: `GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens`, `MAX_TOKENS = 100_000_000_000  # 100B`, `LR = 0.00020712352850360292` (L41-43); `LinearWithWarmup(warmup=0, alpha_f=0.0)` (L80); loads `.../OLMo25/step1413814/` with `load_trainer_state=False`, `load_optim_state=True` (L101-104).
- 32B pretrain: `GLOBAL_BATCH_SIZE = 8 * 1024 * 1024  # ~8M tokens`, `LR = 6e-4` (L43-44); `CosWithWarmup(warmup_steps=2000)` (L83); `max_duration=Duration.epochs(1)` (L107). The script has no stop at 5.5T tokens.
- 32B midtrain ingredient 1: `GLOBAL_BATCH_SIZE = 4 * 1024 * 1024`, `LR = 0.0002071235285` (L42-44).

## Scheduler and trainer code
```python
# scheduler.py L536-558, HalfCosWithWarmup.get_lr
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        ...
        if current < warmup:
            max_lr = eta_min + (initial_lr - eta_min) / 2
            return _linear_warmup(max_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            current += t_max
            t_max *= 2
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * current / t_max)) / 2
```
- `units` defaults to steps (scheduler.py L34); `alpha_f` defaults to 0.1 (L514). `Scheduler.set_lr` passes `trainer.global_step` and `trainer.max_steps` (L67-77).
- `Trainer.max_steps` is computed from `max_duration`, not from `hard_stop`; for a token duration it is `global_step + ceil(tokens_remaining / tokens_per_batch)` (trainer.py L472-476, L521-527).

## Derived schedule (used in ch-14a)
- Tokens per step 4,194,304. Part 1: `max_steps` = ceil(5e12 / 4,194,304) = 1,192,093; the cosine midpoint is at step (2,000 + 1,192,093) / 2 = 597,046.5, which is the part-1 hard stop; LR there 1.650 × 10^−4.
- Part 2: `warmup` = 1,192,092 // 2 + 1,000 = 597,046 and `max_steps` = ceil(7e12 / 4,194,304) = 1,668,931. It loads step 596,047, 999 steps before the part-1 hard stop; the files do not explain the gap. Steps 596,047–597,045 fall in the warmup branch, which returns 1.647 × 10^−4 at the load step; the half-cosine starts at step 597,046 from 1.65 × 10^−4 and would reach 3.0 × 10^−5 at 7T tokens.
- The one-epoch stop is step 1,413,814 (the midtrain load step), 5.930T tokens. There the half-cosine argument is 0.8810 of its span and the LR is 3.93 × 10^−5, assuming the restored trainer state keeps the step and token counters.
- Table 35 of the report prints a 7B final pre-training LR of 3.0 × 10^−5 and midtraining and long-context peak LRs of 2.074 × 10^−4; the scripts set 2.0712 × 10^−4 for 7B midtraining. Table 35's temperature 2.051 × 10^−14 equals (2.074 × 10^−4)² / 2,097,152; 2.0712 × 10^−4 would give 2.046 × 10^−14.

## Verification
- Read on 2026-09-15: the scripts, scheduler.py, and trainer.py at 66f768b; GitHub commits API for the OLMo3 directory.
- Not in the scripts: the reason the midtraining LR is 2.0712 × 10^−4; why part 2 loads step 596,047; whether the loaded optimizer state changes the LR used at restart.
