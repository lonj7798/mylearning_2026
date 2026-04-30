<!-- scope: replay buffers in LLM RL — synthesized from framework implementations
     deps: [[ppo]], [[grpo]]
     see-also: [[trl-grpo]], [[openrlhf-ppo]], [[verl-ppo-loss]], [[on-off-policy-rlhf]]
-->

# Replay Buffers in LLM RL — Framework Synthesis
- **Core Insight:** Classical RL replay buffers (as in DQN, Ape-X) do not directly apply to LLM RLHF because (a) every trajectory is worth ~10k–100k tokens of gradient signal, so fresh rollouts dominate over reuse, and (b) PPO/GRPO's importance-sampling clip already constrains policy drift. What *does* appear in practice is "prompt-level replay": keeping hard or high-variance prompts in a buffer and re-sampling them, not replaying old completions.
- **Guideline:** Don't replay completed trajectories in RLHF PPO — importance-sampling weights blow up. Instead, replay *prompts*: keep a buffer of high-variance or hard prompts and re-feed them to the rollout engine. TRL's `grpo_with_replay_buffer` is the reference implementation.
- **Authors / Sources:** Synthesized from (i) HuggingFace TRL `trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py`; (ii) DeepSeek R1 technical report discussion (§3.2 "RL at scale"); (iii) Costa Huang's "37 Implementation Details of PPO" blog notes on rollout mixing.
- **Year:** framework code 2024–2026
- **URLs:**
  - TRL replay-buffer GRPO: https://github.com/huggingface/trl/blob/main/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py
  - DeepSeek-R1 report: https://arxiv.org/abs/2501.12948
- **Relevant topics:** replay buffer, prompt-level replay, experience reuse, off-policy correction

## Abstract (synthesized)
There is no single canonical paper on LLM-RL replay buffers — instead, the technique has evolved in framework code. This page synthesizes the dominant pattern across TRL, verl, and OpenRLHF. Classical experience replay (trajectory-level) breaks PPO/GRPO because the importance-sampling ratio between current and stored policies explodes as the policy updates. The pattern that survives is prompt-level replay: store `(prompt, group_rewards, group_variance)` tuples, prioritize high-variance prompts, and feed them back to the rollout engine for fresh sampling.

## Key Contributions (framework-level)
- **TRL `GRPOWithReplayBufferTrainer`:** maintains a FIFO buffer of prompts tagged with their reward variance; prompts with all-same rewards (zero group variance) contribute no GRPO gradient and are downweighted in replay.
- **Prioritized prompt sampling:** probabilities weighted by group-reward variance (harder, information-richer prompts get more replay).
- **Rollout-level replay (experimental):** a smaller line of work in verl stores `(prompt, responses, old_logprobs)` from the last ~K steps and re-uses them with *bounded IS correction* — only viable with ratio clipping tight enough to prevent blow-ups.
- **Negative result, formalized:** DeepSeek-R1 §3.2 notes attempting trajectory replay destabilized training; they abandon it in favor of fresh rollouts + larger batch size.

## Key Code References
- **TRL `GRPOWithReplayBufferTrainer`:** file-level — it subclasses `GRPOTrainer` and overrides the prompt sampler to draw from a replay buffer `self.replay_buffer` keyed by group variance. Each inserted entry records `(prompt_ids, rewards, advantages)` and prompts are sampled with probability `p_i ∝ var_i + ε`.
- **verl `experience_buffer.py`** (in `verl/experience_makers/`) keeps per-step experience but is *not* used for cross-step replay; it's an intra-step scratch buffer for multiple PPO epochs.
- **OpenRLHF** has no replay buffer — the default flow is `rollout → train → discard`; off-policyness is handled via `vllm_kl` and IS correction (see [[openrlhf-ppo]]).

## Technical Details — Prompt-Level Replay (TRL pattern)
- **Buffer schema:**
  ```python
  @dataclass
  class BufferEntry:
      prompt_ids: torch.Tensor
      rewards:    torch.Tensor          # (n,) per-rollout outcome rewards
      variance:   float                  # rewards.var().item()
      seen_steps: int
  ```
- **Insertion:** at the end of each step, every prompt's entry is pushed to the buffer; oldest evicted when the buffer hits capacity.
- **Sampling:** for the next step's batch, with probability `p_replay` (default 0.25), draw from the buffer weighted by `variance`; otherwise draw fresh prompts.
- **Zero-variance downweight:** prompts with `variance == 0` (all rollouts correct or all incorrect) contribute no gradient under GRPO; they get probability 0 in replay sampling.
- **IS correction:** *not* needed because the prompt is re-sampled freshly — the old completions are discarded; only the *prompt* is replayed.

## Why trajectory replay fails for LLM RL (synthesis)
1. **Ratio explosion:** `exp(logπ_new(y) − logπ_old(y))` accumulates across stored steps; with 100-token responses and 0.01 drift per token, the ratio explodes past any clip bound within ~20 steps.
2. **Token-level credit assignment:** stored trajectories' per-token advantages become stale because the critic (or the group baseline) has shifted.
3. **Compute asymmetry:** rollouts are cheap relative to training updates in LLM RL; reuse saves little.
4. **KL regularization interference:** the KL-to-reference term in the reward is computed against the *current* π_ref, not the one active when the trajectory was collected.

## Connections
- Prompt replay is the practical form of [[minibatch-sharing-rl]]'s "batch across prompts" idea extended over time.
- The trajectory-replay failure modes are what [[on-off-policy-rlhf]] characterizes theoretically.
- Contrast with classical off-policy RL (DQN, Ape-X, R2D2) — those rely on *bootstrap learning*, which LLM-RL doesn't use.
- Related to [[rest-em]] / [[rejection-sampling-finetuning]]: the "keep good completions" pattern is replay-adjacent but operates in the SFT phase, not the RL gradient.
