<!-- scope: TRL's experimental GRPO replay buffer — the one released implementation of rollout replay in an LLM RL framework
     deps: [[grpo]]
     see-also: [[trl-grpo]], [[openrlhf-ppo]], [[verl-ppo-loss]], [[on-off-policy-rlhf]], [[minibatch-sharing-rl]], [[kimi-k1-5]]
-->

# TRL `trl.experimental.grpo_with_replay_buffer` — `GRPOWithReplayBufferTrainer`
- **Core Insight:** TRL's experimental GRPO replay buffer exists to fix one specific waste in GRPO: a prompt group whose rollouts all get the same reward has zero reward standard deviation and therefore contributes no gradient. The trainer stores past groups that *did* have reward variance and uses them to replace the zero-variance groups in the current batch, keeping the effective batch size full.
- **Guideline:** When a GRPO run wastes a large share of its batch on zero-variance groups, set `replay_buffer_size` above 0 (the dataclass default is 64) to substitute stored high-scoring groups for them. The module is under `trl.experimental`, so treat its API as unstable and pin a TRL version.
- **Authors:** Hugging Face TRL contributors
- **Year:** code read at TRL v0.24.0 (2025); the module is not present on `main` as of 2026-09-18
- **URL:** https://github.com/huggingface/trl/blob/v0.24.0/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py
- **Source type:** released config/code
- **Relevant topics:** replay buffer, GRPO, zero-variance groups, experience reuse, off-policy correction

## Summary
`GRPOWithReplayBufferTrainer` subclasses `GRPOTrainer` and adds a bounded `ReplayBuffer` held as a
min-heap keyed by a score. After computing advantages for a batch, the trainer splits prompt groups by
whether their per-group reward standard deviation is greater than zero. Groups with variance are pushed
into the buffer together with their completions, masks, advantages, and cached log-probabilities. Groups
without variance are then *replaced* by groups drawn from the buffer, sampled with probability
proportional to their stored score. What is replayed is a full stored rollout group, not a prompt to be
re-sampled.

## Key Contributions (code-level)
- A concrete, released definition of "replay" for GRPO: replace gradient-free groups with stored
  gradient-carrying groups inside the same training step.
- A priority score that combines advantage magnitude and reward spread:
  `score_group = Σ_j |advantage_{group,j}| · std_rewards_{group,j}`, summed over the last axis and taken
  only for groups with variance (`update_replay_buffer`, the `replay_buffer_scores` line).
- A fixed-capacity min-heap retention rule: below `max_size` every group is pushed; at capacity a new group
  replaces the current minimum only if its score is higher (`ReplayBuffer.add`).
- Sequence-length reconciliation: sampled groups and the live batch are padded to a common prompt and
  completion length before substitution, left-padding prompts and right-padding completions.

## Key Code References
- `ReplayBuffer.__init__ / add / sample` — capacity, heap retention, and `torch.multinomial` sampling over
  score-normalized probabilities (`probabilities = scores / scores.sum()`).
- `GRPOWithReplayBufferTrainer.__init__` — `self.replay_buffer = ReplayBuffer(args.replay_buffer_size) if
  args.replay_buffer_size > 0 else None`.
- `update_replay_buffer(...)` — builds one dict per variance-carrying group containing `prompt_ids`,
  `completion_ids`, `prompt_mask`, `completion_mask`, `advantages`, and, when present,
  `old_per_token_logps`, `ref_per_token_logps`, `importance_sampling_ratio`, and vision fields.
- `update_with_replay_buffer(...)` — `groups_with_variance = group_std_rewards.max(dim=0).values > 0`;
  returns early if `self.replay_buffer.max_size <= 0` or if no group needs replacing.
- `GRPOWithReplayBufferConfig.replay_buffer_size` — `int`, field default `64`; the docstring states the
  default is `0`, so the two disagree within the same file.

## Technical Details
- **Unit of replay:** a prompt group, meaning one prompt together with its `num_generations` completions
  and their advantages. The prompt is not re-sampled; the stored completions are reused.
- **Eligibility:** only groups with non-zero reward standard deviation enter the buffer; only zero-variance
  groups are replaced. The trainer logs `frac_reward_zero_std`, the fraction of zero-variance groups.
- **Importance sampling:** when vLLM generation is used with `vllm_importance_sampling_correction`, the
  stored group carries its `importance_sampling_ratio` forward, so the correction is the one computed when
  the group was generated, not one recomputed against the current policy.
- **Stored log-probabilities:** `old_per_token_logps` and `ref_per_token_logps` are stored and reused, so a
  replayed group's PPO-style ratio is measured against the policy that produced it.
- **Status:** the module lives under `trl/experimental/`. Fetching the same path on `main` on 2026-09-18
  returns 404, and the current `trl/experimental/` listing does not contain `grpo_with_replay_buffer`.

## Findings relevant to negative feedback
- The zero-variance case includes both all-correct and all-incorrect groups. Because GRPO advantages are
  computed within a group, an all-incorrect group produces no negative gradient; it is replaced rather than
  used as a negative signal. Replacement raises the share of the batch carrying signed advantage, not the
  share that is negative.
- The buffer's retention score weights by `|advantage| · std`, so groups with the largest spread between
  successful and unsuccessful rollouts are the ones kept.

## Connections
- [[grpo]], [[trl-grpo]] — the base algorithm and base trainer this module subclasses.
- [[openrlhf-ppo]] — OpenRLHF defines `NaiveReplayBuffer` in `openrlhf/trainer/ppo_utils/replay_buffer.py`,
  but `PPOActor.fit` calls `replay_buffer.clear()` after each `ppo_train` (`trainer/ray/ppo_actor.py`), so it
  is per-step experience storage for multiple PPO epochs, not cross-step replay.
- [[on-off-policy-rlhf]] — measures what happens when policy-gradient training uses stale or reordered data,
  which is the regime a replayed group enters.
- [[minibatch-sharing-rl]] — sharing signal across prompts within a step, the same-step counterpart.
- [[kimi-k1-5]] — partial rollouts store unfinished trajectories and resume them (§2.6.2 of that report),
  a different mechanism from replaying completed groups.

## Verification
- Checked on 2026-09-18 against:
  https://raw.githubusercontent.com/huggingface/trl/v0.24.0/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py
  and `.../grpo_with_replay_buffer_config.py`;
  https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/main/openrlhf/trainer/ppo_utils/replay_buffer.py and
  `.../openrlhf/trainer/ray/ppo_actor.py`; the DeepSeek-R1 report (arXiv:2501.12948).
- Corrections to the previous card version:
  - The card was a multi-source synthesis with no primary artifact. It now describes the one released
    artifact its claims were drawn from, the TRL experimental module.
  - "What does appear in practice is prompt-level replay: keeping hard or high-variance prompts in a buffer
    and re-sampling them, not replaying old completions" and "IS correction: not needed because the prompt
    is re-sampled freshly — the old completions are discarded; only the prompt is replayed" → the buffer
    stores and reuses `completion_ids`, `completion_mask`, `advantages`, `old_per_token_logps`,
    `ref_per_token_logps`, and `importance_sampling_ratio` (`update_replay_buffer`). Completions are
    replayed.
  - "Zero-variance downweight: prompts with `variance == 0` ... get probability 0 in replay sampling" →
    zero-variance groups are never inserted; they are the groups that get replaced.
  - "prompts are sampled with probability `p_i ∝ var_i + ε`" → probability is proportional to the stored
    score `Σ |advantage| · std`, normalized by its sum (`ReplayBuffer.sample`).
  - "with probability `p_replay` (default 0.25), draw from the buffer" → there is no `p_replay`; the number
    of groups drawn equals the number of zero-variance groups in the batch.
  - "Each inserted entry records `(prompt_ids, rewards, advantages)`" and the `BufferEntry` dataclass with
    `rewards`, `variance`, `seen_steps` fields → no such dataclass exists; entries are plain dicts and the
    heap holds `(score, datum)` tuples.
  - "oldest evicted when the buffer hits capacity" → eviction is by lowest score, not age
    (`heapq.heapreplace` guarded by `score > self.heap[0][0]`).
  - "OpenRLHF has no replay buffer" → it defines `NaiveReplayBuffer`; the accurate statement is that the
    buffer is cleared after every `ppo_train` call.
- Removed as unsupported by the source: "DeepSeek-R1 §3.2 'RL at scale' notes attempting trajectory replay
  destabilized training; they abandon it in favor of fresh rollouts + larger batch size" (§3.2 of the R1
  report is "Distilled Model Evaluation", and neither "replay" nor "buffer" appears anywhere in the
  report); "verl `experience_buffer.py` in `verl/experience_makers/`" (path returns 404) and the
  "rollout-level replay with bounded IS correction" line attributed to verl; the attribution to Costa
  Huang's "37 Implementation Details of PPO"; OpenRLHF's "`vllm_kl`"; the four numbered "why trajectory
  replay fails" claims, including "with 100-token responses and 0.01 drift per token, the ratio explodes
  past any clip bound within ~20 steps" (no source, and no such measurement exists in the cited code);
  "every trajectory is worth ~10k–100k tokens of gradient signal"; "rollouts are cheap relative to training
  updates in LLM RL"; "TRL's `grpo_with_replay_buffer` is the reference implementation" (it is an
  experimental module that is no longer on `main`).
- Not reported by the source: any benchmark or ablation showing that the replay buffer improves final
  reward, pass@k, or sample efficiency. The module ships no evaluation.
