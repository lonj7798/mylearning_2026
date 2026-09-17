---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: verl-project/verl@753aed3 `docs/algo/dapo.md` ("Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)", page dated 06/19/2025) — Overlong Reward Shaping section and FAQ (chapter-local verified extract)
source_url: https://github.com/verl-project/verl/blob/753aed3/docs/algo/dapo.md
created_at: "2026-09-15"
---

# Excerpt: verl DAPO recipe — overlong buffer configuration and code

- **Source type:** released config/code documentation (official verl repository)
- **Used in:** ch-44a §2, Recipe.

## Configuration block (docs/algo/dapo.md, "Overlong Reward Shaping")
```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

"Setting `overlong_buffer.enable` to `True` will penalize the outputs whose lengths are overlong but still within the hard
context limit. Specifically, the penalty increases linearly from `0` to `overlong_buffer.penalty_factor` when the length of
the output exceeds the `max_response_length - overlong_buffer.len` by `0` to `overlong_buffer.len` tokens."

## Core relevant code (same section)
```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```

The penalty is computed from `valid_response_length`, so a response shorter than `expected_len` receives 0 and the
penalty is clamped at 0 from above. The `-1` case of the paper's Eq. 13 cannot occur here, because generation stops at
`max_response_length`.

## FAQ: "Where is the 'Overlong Filtering' in the paper?"
"Most experiments in the paper, including the best-performant one, are run without Overlong Filtering because it's somehow
overlapping with Overlong Reward Shaping in terms of properly learning from the longest outputs. So we don't implement it
here."

## Loss aggregation (same document, "Flexible Loss Aggregation Mode")
```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean" # / "seq-mean-token-sum" / "seq-mean-token-mean"
```
"Setting `loss_agg_mode` to `token-mean` will mean the (policy gradient) loss across all the tokens in all the sequences in
a mini-batch."

## Reproduction table (same document)
Qwen2.5-32B on 16×8×H800: DAPO 52% AIME 2024; "DAPO w/o Dynamic Sampling" 50%; "DAPO w/o Token-level Loss & Dynamic
Sampling" 44% (the last row also differs in hardware and image; one run each).
