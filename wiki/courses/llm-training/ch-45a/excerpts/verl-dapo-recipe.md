---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the released documentation and code"
source_url: https://github.com/volcengine/verl/blob/main/docs/algo/dapo.md
primary_version: "verl docs/algo/dapo.md and the referenced trainer code, fetched 2026-09-14"
created_at: "2026-09-15"
---

# Excerpt: verl's DAPO recipe configuration

ch-45a uses this file as the released-config row that sits next to the DAPO paper row ([[dapo]]).

## Clip-Higher
```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```
```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

## Dynamic sampling with group filtering
```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc
    max_num_gen_batches: 10
```
> "Setting `filter_groups.enable` to `True` will filter out groups whose outputs' `metric` are all the same, e.g.,
> for `acc`, groups whose outputs' accuracies are all 1 or 0. The trainer will repeat sampling with
> `gen_batch_size` until there are enough qualified groups for `train_batch_size` or reaching the upper limit
> specified by `max_num_gen_batches`."

`gen_batch_size: 1536` is three times `train_batch_size: 512`. The DAPO paper prints the prompt batch size 512
but not this over-sampling factor, so the factor is a property of this configuration, not of the paper's run.

## Overlong reward shaping
```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```
```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```
The penalty is linear in the overshoot and reaches −1.0 at the hard cap. The paper states the 16,384 + 4,096
split but does not print `penalty_factor`.

## Verification
- Read on 2026-09-15 from the cached copy of verl `docs/algo/dapo.md` (scratchpad
  `sources/rc03-docs_algo_dapo_md.txt`), fetched 2026-09-14.
- The file is documentation with example configurations; it does not claim to be the exact configuration of the
  paper's released run, and no Beaker- or W&B-style run link is given.
