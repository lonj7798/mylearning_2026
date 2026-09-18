---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py
source_url: https://github.com/huggingface/transformers
created_at: "2026-09-17"
revised: 2026-09 (generality revision)
---

# Excerpt: where the Hugging Face `Trainer` clips, steps, and computes its horizon

**Artifact:** `huggingface/transformers` tag `v4.57.1`, file `src/transformers/trainer.py`.
**Read on:** 2026-09-17 from the file text cached at `scratchpad/sources/c08-hf-trainer.txt`
(fetched from `raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/trainer.py`).

`SFTTrainer` ([[trl-sft-trainer]]) inherits this loop, so the three lines the lab instruments are in this file.

---

## The update block is in `_inner_training_loop`, not in `training_step`

`_inner_training_loop` is defined at L2353. Inside it, guarded by `if do_sync_step:` (L2692):

```python
# L2696-2718 (abridged: the SageMaker-MP and Apex branches are omitted)
# Gradient clipping
if args.max_grad_norm is not None and args.max_grad_norm > 0:
    ...
    _grad_norm = self.accelerator.clip_grad_norm_(
        model.parameters(),
        args.max_grad_norm,
    )
...
# L2739-2751
with context():
    self.optimizer.step()
...
learning_rate = self._get_learning_rate()
if not self.accelerator.optimizer_step_was_skipped:
    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.lr_scheduler.step()
model.zero_grad()
self.state.global_step += 1
```

Order at these loci: clip (L2697-2718) → `optimizer.step()` (L2740) → `lr_scheduler.step()` (L2750) →
`model.zero_grad()` (L2752). Under DeepSpeed the logged `grad_norm` is read back from
`model.get_global_grad_norm()` instead of the clip return value (L2721-2728).

- `training_step` (L3981) runs the forward pass, divides the loss by
  `self.current_gradient_accumulation_steps` when the model does not accept loss kwargs (L4059-4063), calls
  `self.accelerator.backward(loss, **kwargs)` (L4070), and returns `loss.detach()`. It does not clip and it does
  not step.
- `_maybe_log_save_evaluate` (L3190) assembles the log dictionary (`loss`, `grad_norm`, `learning_rate`), then
  handles saving and evaluation. It is called after the step (L2757).

## The scheduler horizon is computed from the length of the prepared dataloader

`set_initial_training_values` (L5660-5720):

```python
# L5675-5689
max_steps = args.max_steps
epoch_based = max_steps < 0
len_dataloader = len(dataloader) if has_length(dataloader) else None
if len_dataloader is not None:
    num_update_steps_per_epoch = max(
        len_dataloader // args.gradient_accumulation_steps
        + int(len_dataloader % args.gradient_accumulation_steps > 0),
        1,
    )
    if epoch_based:
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
```

`len_dataloader` is the length of the dataloader built over the dataset the trainer was given. In
`SFTTrainer` the dataset is packed inside `_prepare_dataset` during `__init__` ([[trl-sft-trainer]] L1662-1678),
before `super().__init__` builds the dataloader, so `num_update_steps_per_epoch` already counts packed rows.
A cosine schedule created with `num_training_steps=max_steps` therefore reaches its minimum at the last step
of the run, whether or not packing changed the row count.

## Correction recorded here

The version of ch-08 written in 2026-04 stated that clipping is "inherited from
`Trainer._maybe_log_save_evaluate` and `accelerator.clip_grad_norm_`" and that
"`Trainer.training_step` calls `self.accelerator.clip_grad_norm_`". Both statements name the wrong method:
at v4.57.1 the call is in `_inner_training_loop` (L2715). The same version stated that packing makes a cosine
schedule finish at `T/2.5`; the horizon is computed from the packed dataloader (L5682-5689), so that failure
mode requires a schedule built by hand with a pre-packing step count.
