---
chapter: ch-16
course: llm-training
phase: read
artifact: "Hugging Face TRL — trl/experimental/grpo_with_replay_buffer/ (trainer, config, docs)"
source_url: https://github.com/huggingface/trl/blob/main/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py
version: cached copy fetched 2026-09-14 (commit hash not recorded by the fetch); header year line "2020-2026"
verified_on: "2026-09-15"
supersedes: the 2026-04 version of this file, which repeated the library card's claim that frameworks replay prompts and never trajectories
---

# Excerpt: what TRL's replay buffer actually stores

Used by [[read]] §6.1, §6.2 and the Common mistakes table. This file exists because the library
card `wiki/raw-data/llm-training/papers/replay-buffer-rlhf.md` states that the operative pattern is
"replay the prompt, never replay the completion" and attributes a prompt-only buffer to TRL. The
code and the documentation of the trainer it names say otherwise; the card needs regeneration by
its owner.

## Documentation (`docs/source/grpo_with_replay_buffer.md`, cached)

"This experimental trainer, trains a model with GRPO but replaces groups (and corresponding
completions) that have 0 standard deviation with groups with high rewards and standard deviation
that've been used to train a model in prior batches."

Config field (`GRPOWithReplayBufferConfig`, `replay_buffer_size`, default 64): "A cache that stores
the rollouts with the highest advantage scores and variance per group. If a new group has 0
variance, it is replaced with a group sampled from the replay buffer."

## The buffer (class `ReplayBuffer`)

A bounded min-heap of `(score, data)` pairs. `add` pushes while the heap is below `max_size`, and
otherwise replaces the minimum when the new score is higher. `sample` draws without replacement
(with replacement when more samples are requested than the heap holds) with probabilities
proportional to the stored scores.

## What one entry holds (`update_replay_buffer`)

```python
buffered_output = {
    "prompt_ids": group_prompt_ids,
    "completion_ids": group_completion_ids,
    "advantages": group_advantages[group_idx].tolist(),
    "prompt_mask": group_prompt_mask,
    "completion_mask": group_completion_mask,
}
# plus "old_per_token_logps" / "ref_per_token_logps" when those tensors exist,
# and "importance_sampling_ratio" when vLLM importance-sampling correction is on
replay_buffer_scores = (group_advantages.abs() * group_std_rewards).sum(dim=-1)[groups_with_variance]
self.replay_buffer.add(replay_buffer_scores.tolist(), buffered_outputs)
```

Completions and their advantages are stored. Groups with non-zero reward standard deviation are the
ones added.

## How a stale group re-enters training (`update_with_replay_buffer`)

Groups in the new batch whose reward standard deviation is zero are replaced slot for slot:

```python
prompt_ids[idx_range] = sampled_data["prompt_ids"][i]
completion_ids[idx_range] = sampled_data["completion_ids"][i]
group_advantages[group_idx] = sampled_data["advantages"][i]
if "old_per_token_logps" in sampled_data:
    old_per_token_logps[idx_range] = sampled_data["old_per_token_logps"][i]
```

The stored advantages are reused as computed at storage time; they are not recomputed against the
current reward statistics.

## Whether the replayed group carries an importance correction

`old_per_token_logps` is computed only when generation and optimization steps are misaligned, or
when vLLM importance-sampling correction is enabled; otherwise the trainer sets it to `None`
(`_generate_and_score_completions`, comment: "If the steps are aligned, importance sampling isn't
necessary and we set old_per_token_logps to None"). In the parent `GRPOTrainer._compute_loss`
([[trl-grpo]], commit a08e713):

```python
old_per_token_logps = inputs.get("old_per_token_logps")
old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
log_ratio = per_token_logps - old_per_token_logps
```

so a replayed group with no stored log-probabilities enters the loss with a ratio of exactly 1 and
never triggers clipping, however stale it is.

## Scoring function

With the default `scale_rewards="group"`, advantages are divided by the group standard deviation
before storage, so `Σ_i |A_i| · σ_group ≈ Σ_i |r_i − r̄|`, which for a binary verifier with k
successes in a group of G equals `2k(G − k)/G`. The buffer therefore prefers groups whose pass rate
is near 0.5 — the same criterion as the pass-rate filters in [[read]] §2.

## Logged metrics seen in the same file

`frac_reward_zero_std` (fraction of groups with zero reward standard deviation) is logged in every
run; `sampling/importance_sampling_ratio/{min,mean,max}` is logged when the vLLM correction is on.

## Status

Released code and documentation of an experimental trainer; no benchmark result is published for
it in the repository, so it is evidence about what a framework does, not about whether replay
helps.
