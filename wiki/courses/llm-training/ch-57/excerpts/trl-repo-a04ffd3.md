---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: github.com/huggingface/trl at commit a04ffd337108285002df29d9ae73ef7e855e03f1
source_url: https://github.com/huggingface/trl/tree/a04ffd337108285002df29d9ae73ef7e855e03f1
created_at: "2026-09-17"
---

# Excerpt: the TRL repository at commit `a04ffd3` (2026-09-14)

**Artifact.** `huggingface/trl`, commit `a04ffd337108285002df29d9ae73ef7e855e03f1`, authored 2026-09-14, at
the time the head of `main`. This excerpt records the repository layout and the configuration-field values that
[[read]] §1-§8 cites, so each claim can be checked without re-fetching the tree.

Read on 2026-09-17 from: the GitHub tree API for the commit, plus the raw files
`trl/trainer/sft_config.py`, `trl/trainer/sft_trainer.py`, `trl/trainer/dpo_config.py`,
`trl/trainer/dpo_trainer.py`, `trl/trainer/grpo_config.py`, `trl/trainer/grpo_trainer.py`,
`trl/trainer/distillation_config.py`, `trl/trainer/distillation_trainer.py`, `trl/trainer/callbacks.py`,
`trl/trainer/utils.py`, `trl/rewards/__init__.py`, `trl/experimental/gkd/gkd_config.py`,
`trl/experimental/async_grpo/async_grpo_config.py`, `trl/experimental/merge_model_callback.py`.

---

## 1. Package layout (blobs under `trl/`, abridged to the trainers and losses)

```
trl/trainer/       base_config.py  base_trainer.py  callbacks.py
                   distillation_config.py  distillation_trainer.py
                   dpo_config.py  dpo_trainer.py
                   grpo_config.py  grpo_trainer.py
                   kto_config.py  kto_trainer.py
                   model_config.py  reward_config.py  reward_trainer.py
                   rloo_config.py  rloo_trainer.py
                   sft_config.py  sft_trainer.py  utils.py
trl/experimental/  a2po/  async_distillation/  async_grpo/  bco/
                   bema_for_ref_model/  cpo/  distillation/  gkd/  gmpo/
                   gold/  gspo_token/  harbor/  iw_opd/  kto/  minillm/
                   nash_md/  online_dpo/  openreward/  orpo/  prm/  sdft/
                   sdpo/  server_distillation/  ssd/  tpo/  xpo/
                   merge_model_callback.py  utils.py
trl/losses/        dpo_loss.py  fused_linear_distillation.py
                   fused_linear_ppo.py  fused_linear_preference.py
                   grpo_loss.py  jsd_loss.py
trl/models/        activation_offloading.py  utils.py
trl/rewards/       accuracy_rewards.py  format_rewards.py  other_rewards.py
trl/generation/    vllm_client.py  vllm_generation.py
trl/scripts/       distillation.py  dpo.py  grpo.py  kto.py  reward.py
                   rloo.py  sft.py  vllm_serve.py
```

**Absent from the tree at this commit:** `trl/trainer/ppo_trainer.py`, `trl/trainer/ppo_config.py`,
`trl/experimental/ppo/`, `trl/models/modeling_value_head.py`, `trl/core/`, and any replay-buffer GRPO trainer.
See [[trl-ppo]] for the removal record (PR #7020, commit `700b845`, 2026-09-04, released in v1.13.0).

`trl/rewards/__init__.py` exports: `accuracy_reward`, `get_cosine_scaled_reward`, `reasoning_accuracy_reward`
(`accuracy_rewards`); `think_format_reward` (`format_rewards`); `get_repetition_penalty_reward`,
`get_soft_overlong_punishment` (`other_rewards`).

---

## 2. `SFTConfig` fields cited by the chapter (`trl/trainer/sft_config.py`)

| Field | Line (field) | Default |
|---|---|---|
| `learning_rate` | L141 | `2e-5` |
| `max_length` | L203 | `1024` |
| `packing` | L223 | `False` |
| `packing_strategy` | L230 | `"bfd"` |
| `completion_only_loss` | L259 | `None` |
| `assistant_only_loss` | L271 | `False` |
| `loss_type` | L281 | `None` → `"chunked_nll"` in `__post_init__` (L332-L334) |

Docstring quotes (L68-L115):

> `max_length` (`int` or `None`, *optional*, defaults to `1024`): Maximum length of the tokenized sequence.
> … When packing is enabled, this value sets the sequence length.
>
> `packing_strategy` (`str`, *optional*, defaults to `"bfd"`): … Can be `"bfd"` (best-fit decreasing, truncates
> overflow), `"bfd_split"` (best-fit decreasing, splits overflow sequences), or `"wrapped"` (aggressive, cuts
> mid-sequence).
>
> `padding_free` … When packing is enabled with strategy `"bfd"`, padding-free is enabled, regardless of the value
> of this parameter.
>
> `completion_only_loss` (`bool`, *optional*): … If `None` (default), the behavior depends on the dataset: loss is
> computed on the completion for prompt-completion datasets, and on the full sequence for language modeling datasets.
>
> `assistant_only_loss` (`bool`, *optional*, defaults to `False`): … loss is computed only on the assistant
> responses, which is supported only for conversational datasets.
>
> `loss_type` … `"nll"`; `"dft"` (Dynamic Fine-Tuning, huggingface.co/papers/2508.05629); `"chunked_nll"`: same
> math as `"nll"`, but the `lm_head` projection is computed on non-ignored tokens only.

`sft_trainer.py` L1256-L1277 is quoted verbatim in [[read]] §2.2: the `{% generation %}` template swap and the
end-of-turn-token warning. `sft_trainer.py` L1596-L1601 raises a related warning when an example produces no
assistant mask at all.

There is **no** `train_on_response_only` field and **no** `max_seq_length` field at this commit, and no use of
`ConstantLengthDataset` in the packing path.

---

## 3. `DPOConfig` fields cited by the chapter (`trl/trainer/dpo_config.py`)

| Field | Line (field) | Default |
|---|---|---|
| `learning_rate` | L155 | `1e-6` |
| `loss_type` | L241 | `["sigmoid"]` |
| `loss_weights` | L250 | `None` → `[1.0] * len(loss_types)` (`dpo_trainer.py` L769) |
| `f_divergence_type` | L267 | `"reverse_kl"` |
| `beta` | L293 | `0.1` |

Docstring (L81-L88), verbatim:

> `loss_type` (`list[str]`, *optional*, defaults to `["sigmoid"]`): Type of loss to use. Possible values are:
> `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`, `'nca_pair'`, `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`,
> `'aot_unpaired'`, `'apo_zero'`, `'apo_down'`, `'discopop'`, `'sft'`, `'sigmoid_norm'`. If multiple loss types
> are provided, they will be combined using the weights specified in `loss_weights`.
>
> `loss_weights` … Example: `[0.8, 0.2, 1.0]` for MPO.

`dpo_trainer.py` loci: the weighted combination loop L1470-L1472; `delta_score` L1468; the `"sigmoid"` branch
L1472-L1473; the `"sft"` branch (cross-entropy on the chosen completion) L1581-L1587; the `ValueError` listing the
15 valid values L1597-L1600; PEFT reference handling L919-L924 and L1412-L1418; logged metrics L1346-L1363
(`logits/chosen`, `logits/rejected`, `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`,
`rewards/margins`, `logps/chosen`, `logps/rejected`).

---

## 4. `GRPOConfig` fields cited by the chapter (`trl/trainer/grpo_config.py`)

| Field | Line (field) | Default |
|---|---|---|
| `num_generations` | L479 | `8` |
| `max_completion_length` | L493 | `512` |
| `use_vllm` | L582 | `False` |
| `vllm_mode` | L589 | `"colocate"` |
| `beta` | L676 | `0.0` |
| `epsilon` | L688 | `0.2` |
| `multi_objective_aggregation` | L772 | `"sum_then_normalize"` |
| `scale_rewards` | L784 | `"group"` |
| `loss_type` | L796 | `"dapo"` |
| `top_entropy_quantile` | L861 | `1.0` |

Docstring quotes: `reward_weights` (L213-L215) "Weights for each reward function. Must match the number of reward
functions. If `None`, all rewards are weighted equally with weight `1.0`."; `multi_objective_aggregation`
(L216-L224) defining `"sum_then_normalize"` and `"normalize_then_sum"`; `top_entropy_quantile` (L861 docstring)
attributing ρ to "Beyond the 80/20 Rule" (huggingface.co/papers/2506.01939) with paper value 0.2;
`num_generations_eval` (L483-L486) allowing fewer generations at eval than at train.

Trainer loci: advantage and multi-reward aggregation `grpo_trainer.py` L2787-L2826; `nanstd` with the unbiased
correction `count / (count - 1)` at `trl/trainer/utils.py` L881-L885. The loss-normalizer block, KL term, and
sign-split clip metrics are quoted from the pinned card [[trl-grpo]] at commit `a08e713`; that card records the
`a04ffd3` line shifts (`_compute_loss` at L3113).

---

## 5. `DistillationConfig` and `GKDConfig`

`trl/trainer/distillation_config.py`: `learning_rate` L171 `1e-6`; `max_completion_length` L222 `512`;
`temperature` L245 `1.0`; `beta` L354 `1.0`. Docstring L133-L137, verbatim:

> `beta` (`float`, *optional*, defaults to `1.0`): Interpolation coefficient for the Generalized Jensen-Shannon
> Divergence loss. When `0.0`, the loss is the forward KL divergence. When `1.0`, the loss is the reverse KL
> divergence. When `0.5`, it is the standard JSD. Unlike GRPO's `beta` (a KL-penalty coefficient against a
> reference model), here it selects the divergence itself; there is no reference-model KL penalty.

`DistillationTrainer` class docstring (L295-L300): "The student is trained on-policy — it generates the
completions itself — to match the teacher's next-token distribution under a generalized Jensen-Shannon divergence
(interpolating forward KL, reverse KL, and JSD via `beta`), as introduced in On-Policy Distillation of Language
Models (https://huggingface.co/papers/2306.13649)" — that is [[gkd-on-policy-distillation]].

Divergence implementation, `distillation_trainer.py` L140-L156: `beta == 0.0` →
`F.kl_div(student_log_probs, teacher_log_probs, log_target=True)`; `beta == 1.0` → the arguments swapped;
otherwise the mixture `logsumexp([student + log1p(-β), teacher + log β])` and
`jsd = β·kl_teacher + (1 − β)·kl_student`. The `lm_head` projection is chunked so the full `(B, C, V)` logits are
never materialized (L1828-L1832).

`trl/experimental/gkd/gkd_config.py` (subclasses `SFTConfig`): `temperature` L55 `0.9`; `lmbda` L59 `0.5`;
`beta` L66 `0.5`; `max_new_tokens` L74 `128`; `seq_kd` L96 `False`. Docstring for `lmbda`: "Lambda parameter that
controls the student data fraction (i.e., the proportion of on-policy student-generated outputs)."

---

## 6. `AsyncGRPOConfig` (`trl/experimental/async_grpo/async_grpo_config.py`)

`dtype` L177 `"float32"`; `num_generations` L232 `8`; `max_completion_length` L236 `2048`;
`max_staleness` L358 `4`; `queue_maxsize` L365 `1024`; `weight_sync_steps` L369 `1`. Docstring L120-L133:

> `max_inflight_tasks` (defaults to `-1`): … auto, which sets it to
> `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`.
>
> `max_staleness` (defaults to `4`): Maximum number of weight update steps a rollout sample can lag behind the
> current model version before being discarded.
>
> `weight_sync_steps` (defaults to `1`): Number of training steps between weight synchronizations to the vLLM server.

The `dtype` docstring (L36-L46) gives the reason for the float32 default: the training-inference mismatch measured
in "Defeating the Training-Inference Mismatch via FP16" (huggingface.co/papers/2510.26788); the vLLM server must be
served in the same dtype, and a mismatch is logged as a warning at train start. Note block (L143-L149):
`logging_steps` 1, `gradient_checkpointing` True, `bf16` True when `fp16` is unset, `learning_rate` 1e-6,
`lr_scheduler_type` `constant`, `ignore_data_skip` True.

---

## 7. Callbacks

`trl/trainer/callbacks.py` defines `SyncRefModelCallback` (L111), `RichProgressCallback` (L152),
`LogCompletionsCallback` (L263), `WeaveCallback` (L355), `BEMACallback` (L584).

`trl/experimental/merge_model_callback.py`: `MergeConfig` (L48) with `method` in
`{"linear", "ties", "dare_ties", "slerp"}` — `linear` weights 0.5/0.5, `ties`/`dare_ties` policy density
`[1.0, 0.7, 0.1]`, `slerp` `t_values` 0.5, all `dtype "float16"` (L82-L112); `MergeModelCallback` (L294) with
`merge_at_every_checkpoint` and `push_to_hub`, merging the checkpoint with a target model via `mergekit`
(L333-L339). The file reports no evaluation of the merge.

---

## Connections
- [[read]] — the chapter that cites every locus above.
- [[trl-grpo]], [[trl-ppo]], [[trl-online-dpo]] — verified library cards pinned at commit `a08e713` (2026-04-21),
  with their own records of what changed by `a04ffd3`.
- [[gkd-on-policy-distillation]] — the paper the stable distillation trainer names.
