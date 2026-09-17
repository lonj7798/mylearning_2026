---
chapter: ch-16
course: llm-training
phase: read
artifact: "verl documentation — Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)"
source_url: https://verl.readthedocs.io/en/latest/algo/dapo.html
version: docs page, "Last updated: 06/19/2025" (cached 2026-09-14)
verified_on: "2026-09-15"
note: no library card exists for this slug yet; this file is the chapter's verified extract
---

# Excerpt: verl DAPO recipe — the configuration that implements dynamic sampling

Used by [[read]] §3.2 and the Recipe table. Source type: released documentation and configuration
of the framework used by the DAPO authors ([[dapo]]); reliability: official for the framework,
practitioner-evidence for the reproduction numbers.

## Dynamic sampling configuration (section "Dynamic Sampling (with Group Filtering)")

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10 # Non-positive values mean no upper limit
```

Documentation text: "Setting `filter_groups.enable` to `True` will filter out groups whose outputs'
`metric` are all the same, e.g., for `acc`, groups whose outputs' accuracies are all 1 or 0. The
trainer will repeat sampling with `gen_batch_size` until there are enough qualified groups for
`train_batch_size` or reaching the upper limit specified by `max_num_gen_batches`."

Control flow quoted on the same page:

```python
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    num_gen_batches += 1
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        continue          # keep generating
    else:
        raise ValueError(...)
else:
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```

Two consequences the chapter uses: generation is provisioned at three times the training batch
size (1536 against 512), and a pool that cannot fill a batch within 10 generation rounds raises
an error rather than training on degenerate groups.

## Reproduction runs (section "Reproduction Runs")

| Setup | AIME 2024 Acc. | Hardware | Commit |
|---|---|---|---|
| DAPO | 52% | 16×8×H800 | verl `4f80e46` |
| DAPO w/o Dynamic Sampling | 50% | 16×8×H800 | verl `4f80e46` |
| DAPO w/o Token-level Loss & Dynamic Sampling | 44% | 16×8×H20 | verl `4f80e46` |

One run per row; the third row uses different hardware and a different container image, so it is
not a clean comparison with the first two.

## Other DAPO components on the same page

Clip-Higher: `clip_ratio_low: 0.2`, `clip_ratio_high: 0.28`. Loss aggregation:
`loss_agg_mode: "token-mean"` is the default and normalizes over all tokens of the mini-batch.
