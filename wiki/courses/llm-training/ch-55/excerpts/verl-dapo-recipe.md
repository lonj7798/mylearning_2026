---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: verl docs/algo/dapo.md at commit 753aed3 (no library card yet)
source_url: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/docs/algo/dapo.md
created_at: "2026-09-17"
---

# Excerpt: verl's DAPO documentation — configuration blocks and reproduction table

**Artifact:** `docs/algo/dapo.md` in verl-project/verl at commit `753aed3e1c286ba6825a74342b28669e72c083ea` (2026-09-14), plus the DAPO paper values it points to (arXiv:2503.14476v2).
**Status:** the llm-training library has no `verl-dapo-recipe` card yet; this excerpt holds the verified extract the chapter cites. Create the card, then replace this file with a pointer.

---

## Reproduction table (L34–38, verbatim values)

| Setup | AIME 2024 Acc. | Hardware | Image | Commit |
|---|---|---|---|---|
| DAPO | 52% | 16×8×H800 | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | `4f80e4` |
| DAPO w/o Dynamic Sampling | 50% | 16×8×H800 | same | `4f80e4` |
| DAPO w/o Token-level Loss & Dynamic Sampling | 44% | 16×8×H20 | `hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix` | `4f80e4` |

One training run per row, AIME 2024 only, with a W&B record linked. The 44% row also differs in hardware and image, so it is not a clean single-variable comparison.

## Separated clip epsilons — Clip-Higher (L52–57)

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

`clip_ratio_low` and `clip_ratio_high` are ε_low and ε_high of the DAPO objective (L59). The paper keeps ε_low at 0.2 "because increasing it will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space" (arXiv:2503.14476v2 §3.1). Table 1 of that paper reports AIME24 avg@32 moving from 36 to 38 when Clip-Higher is added to the progressive stack on Qwen2.5-32B base.

## Dynamic sampling with group filtering (L73–86)

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

Enabling `filter_groups` removes groups whose outputs' `metric` values are all the same, for example accuracies all 1 or all 0 (L84). The trainer repeats sampling with `gen_batch_size` until enough qualified groups exist for `train_batch_size`, or raises once `max_num_gen_batches` is reached (L86, L90–107).

## Loss aggregation (L113–134)

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean" # / "seq-mean-token-sum" / "seq-mean-token-mean"
    # NOTE: "token-mean" is the default behavior
```

`token-mean` averages the policy-gradient loss over all tokens of all sequences in a mini-batch (L120). The quoted implementation shows `seq-mean-token-sum` summing per sequence then averaging over sequences, and `seq-mean-token-mean` averaging per sequence first (L127–132).

## Overlong reward shaping (L141–165)

```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

The penalty rises linearly from 0 to `penalty_factor` as the output exceeds `max_response_length - overlong_buffer.len` by 0 to `overlong_buffer.len` tokens (L153). Implementation (L157–164):

```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```

The FAQ states that the paper's Overlong Filtering is not implemented in verl because it overlaps with this shaping (L169–171).

## Connections

- [[verl-ppo-loss]] — the loss function the clip values and aggregation modes feed.
- [[verl-grpo]] — the advantage estimator DAPO runs on top of.
- [[dr-grpo]] — the length-bias argument behind the aggregation-mode choice.

## Verification

- Checked on 2026-09-17 against https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/docs/algo/dapo.md and arXiv:2503.14476v2 §3.1, Table 1.
- Not reported by the source: any ablation of `max_num_gen_batches`, `overlong_buffer.len`, or `penalty_factor`; results on any benchmark other than AIME 2024.
