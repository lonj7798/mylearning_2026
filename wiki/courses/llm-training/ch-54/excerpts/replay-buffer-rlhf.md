---
chapter: ch-54
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/replay-buffer-rlhf.md
source_url: https://github.com/huggingface/trl/blob/main/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py
created_at: "2026-04-23"
---

# Excerpt: Replay buffers in RLHF — why ch-54 §4 rejects trajectory replay

**Source library:** `wiki/raw-data/llm-training/papers/replay-buffer-rlhf.md`
**Artifact:** the four-mechanism failure story for trajectory replay, and the TRL prompt-replay schema that survives.

---

## Why this source anchors ch-54

§4 of ch-54 is where a newcomer from classical RL would expect "and here is the replay buffer chapter." The source is the synthesis page that explains why classical experience replay breaks in LLM-RL and what replaces it. The structural fact is that LLM-RL does not bootstrap from value estimates — it does policy-gradient with an IS clip — so the classical replay motivation (off-policy Q-learning) does not transfer.

---

## The four mechanisms §4 quotes

From the source (line 45):

> 1. **Ratio explosion:** `exp(logπ_new(y) − logπ_old(y))` accumulates across stored steps; with 100-token responses and 0.01 drift per token, the ratio explodes past any clip bound within ~20 steps.
> 2. **Token-level credit assignment:** stored trajectories' per-token advantages become stale because the critic (or the group baseline) has shifted.
> 3. **Compute asymmetry:** rollouts are cheap relative to training updates in LLM RL; reuse saves little.
> 4. **KL regularization interference:** the KL-to-reference term in the reward is computed against the *current* π_ref, not the one active when the trajectory was collected.

ch-54 §4 reproduces this list verbatim because each mechanism compounds: even if you fix the IS clip (1), you still have stale advantages (2), and the KL term in the reward is now wrong (4). Four independent failures is not "a bug" — it is structural.

---

## DeepSeek-R1 §3.2 — the attested negative result

From the source (line 23):

> **Negative result, formalized:** DeepSeek-R1 §3.2 notes attempting trajectory replay destabilized training; they abandon it in favor of fresh rollouts + larger batch size.

ch-54 treats this as the strongest empirical signal against trajectory replay at scale: when a team with R1-Zero's budget tries it and backs off, the pattern is real.

---

## What survives: prompt-level replay

From the source (line 32), the TRL schema:

```python
@dataclass
class BufferEntry:
    prompt_ids: torch.Tensor
    rewards:    torch.Tensor          # (n,) per-rollout outcome rewards
    variance:   float                  # rewards.var().item()
    seen_steps: int
```

And (line 41):

> **Sampling:** for the next step's batch, with probability `p_replay` (default 0.25), draw from the buffer weighted by `variance`; otherwise draw fresh prompts.
> **Zero-variance downweight:** prompts with `variance == 0` (all rollouts correct or all incorrect) contribute no gradient under GRPO; they get probability 0 in replay sampling.

The key correctness point (line 43):

> **IS correction:** *not* needed because the prompt is re-sampled freshly — the old completions are discarded; only the *prompt* is replayed.

This is why prompt replay sneaks past the four mechanisms: there is no old-policy content in the gradient. It is curriculum, not off-policy correction.

---

## What §4 forbids you to confuse

- **verl `experience_makers/experience_buffer.py`** is an *intra-step* scratch pad for multiple PPO epochs over the *same* rollout. It is not a cross-step replay buffer. Different object; same name.
- **OpenRLHF** has no buffer. `rollout → train → discard`. Any apparent "off-policyness" in OpenRLHF comes from async staleness (§5), not replay.
- **`rest-em` / [[rejection-sampling-finetuning]]** keep completions but use them for SFT, not for the RL gradient. Replay-adjacent, not replay.

---

## Connections

- **ch-54 §4** — the four-mechanism list and the prompt-replay schema.
- **ch-54 §5** — async staleness is the only form of "off-policyness" surviving in production stacks.
- **ch-40 ([[grpo]])** — zero-group-variance prompts are where `μ_i = mean_k R_{i,k}` makes no signal.
- **[[on-off-policy-rlhf]]** — characterizes the bias theoretically; replay is the operational wrinkle.
- **[[deepseek-r1]] §3.2** — the attested scale-level negative result.
