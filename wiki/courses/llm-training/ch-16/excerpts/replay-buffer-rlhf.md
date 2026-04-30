---
chapter: ch-16
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/replay-buffer-rlhf.md
source_url: https://github.com/huggingface/trl/blob/main/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py
created_at: "2026-04-23"
---

# Excerpt: Replay buffers in LLM RL — prompt-level replay, not trajectory replay

**Source library:** `wiki/raw-data/llm-training/papers/replay-buffer-rlhf.md`
**Primary reference:** TRL `GRPOWithReplayBufferTrainer` + DeepSeek-R1 §3.2 (negative result) + verl `experience_buffer.py`.

---

## Why this source anchors ch-16

This is the page ch-16's §3 derives from. The source page is not a paper — it's a synthesis of operator lore encoded in framework code (TRL / verl / OpenRLHF) plus the DeepSeek-R1 report's §3.2 negative result ("we tried trajectory replay; it destabilized training; we abandoned it"). Three claims from the source drive ch-16:

1. Trajectory-level replay fails for principled reasons (IS-ratio explosion).
2. Prompt-level replay survives — but only with variance weighting.
3. No framework shipping in 2024–2026 stores completions across gradient steps; all that survives is prompt bookkeeping.

Ch-16 cites this source at §3.1 (classical vs policy-gradient replay), §3.2 (the IS-ratio derivation), §3.3 (the three compounding pathologies beyond IS), and §3.4 (the TRL `BufferEntry` dataclass as the reference implementation).

---

## The BufferEntry contract

From the source (lines 32–39):

> ```python
> @dataclass
> class BufferEntry:
>     prompt_ids: torch.Tensor
>     rewards:    torch.Tensor          # (n,) per-rollout outcome rewards
>     variance:   float                  # rewards.var().item()
>     seen_steps: int
> ```

What the dataclass does *not* contain is more informative than what it does: no `responses`, no `logprobs`, no `advantages`, no `kl_per_token`. All the items classical replay stores are *absent*. This is the operational version of the IS-ratio argument — if the gradient never touches stored completions, the `π_θ / π_old` correction is never needed.

Ch-16's §6 `RLPrompt` dataclass is a direct adaptation, with the addition of `verifier_id` and `reference` fields (needed for verifier-driven reward computation) and a `pass_rate` EMA (for the curriculum filter §2).

---

## The sampling rule and the zero-variance downweight

From the source (lines 41–42):

> Sampling: for the next step's batch, with probability `p_replay` (default 0.25), draw from the buffer weighted by `variance`; otherwise draw fresh prompts.
>
> Zero-variance downweight: prompts with `variance == 0` (all rollouts correct or all incorrect) contribute no gradient under GRPO; they get probability 0 in replay sampling.

This is the single most important operational detail in the chapter. The zero-variance downweight is how the pass-rate filter (§2) shows up in the replay buffer — it isn't a separate filter stage, it's a sampling weight that automatically ejects prompts as they become either fully solved or fully failed. Ch-16's `RLPromptPool.sample_batch` implements the same logic: `replay = [pr for pr in band if pr.var > 1e-6]`.

The `p_replay = 0.25` default is not derived from theory. It's a TRL empirical choice — too low, and the buffer never re-uses high-variance prompts; too high, and the batch loses the fresh-prompt diversity that keeps the policy exposed to the broader distribution. No published ablation; 0.25 is a conventional starting point.

---

## The IS-ratio blow-up — why trajectory replay cannot survive

From the source (§ "Why trajectory replay fails for LLM RL", lines 45–49):

> 1. **Ratio explosion:** `exp(logπ_new(y) − logπ_old(y))` accumulates across stored steps; with 100-token responses and 0.01 drift per token, the ratio explodes past any clip bound within ~20 steps.
> 2. **Token-level credit assignment:** stored trajectories' per-token advantages become stale because the critic (or the group baseline) has shifted.
> 3. **Compute asymmetry:** rollouts are cheap relative to training updates in LLM RL; reuse saves little.
> 4. **KL regularization interference:** the KL-to-reference term in the reward is computed against the *current* `π_ref`, not the one active when the trajectory was collected.

Ch-16's §3.2 spells out the log-normal moment argument behind (1). The three other pathologies are ch-16's §3.3. The *compute asymmetry* point (3) is worth internalizing: classical RL invested heavily in replay because each environment step was expensive relative to a gradient update. In LLM RL the ratio inverts — a PPO update on a 128-GPU trainer can burn 10–100× the compute of the rollout that produced the data. Reusing data to save rollout cost optimizes the wrong variable.

---

## DeepSeek-R1's negative result as the canonical counter-evidence

From the source (line 23):

> **Negative result, formalized:** DeepSeek-R1 §3.2 notes attempting trajectory replay destabilized training; they abandon it in favor of fresh rollouts + larger batch size.

This is the empirical companion to the IS-ratio argument. DeepSeek-R1's team tried exactly what the theory says fails, confirmed the instability, and documented the pivot to "fresh rollouts + larger batch size." Ch-16 treats this as canonical evidence: the question "why not trajectory replay?" has both a theoretical answer (§3.2) and an industrial-scale empirical answer (DeepSeek-R1 §3.2, same section number, coincidentally).

---

## What this excerpt unlocks for the next chapters

- **ch-16 §3** uses every piece of this source — the dataclass, the sampling rule, the IS explosion, the DeepSeek negative result.
- **ch-16 §6** reference implementation is a direct adaptation of the TRL `BufferEntry` / `GRPOWithReplayBufferTrainer` pattern.
- **Track 3 (synthetic data)** — the synthetic-prompt generator produces `RLPrompt` objects; this excerpt is the spec for what fields they must carry.

## Connections

- [[excerpts/kimi-k1-5]] — partial rollouts are the legitimate exception to "never replay completions"; this excerpt explains why they don't violate the rule.
- [[excerpts/tulu-3]] — Tülu 3's RLVR uses fresh rollouts every step; no buffer at all. Works because the prompt pool (§1) is large enough that the zero-variance downweight operates implicitly via fresh sampling.
- [[excerpts/on-off-policy-rlhf]] — the theoretical ground for the IS-ratio and coverage arguments.
- [[ch-16]] — §3 (the derivation), §6 (the drop-in manager).
