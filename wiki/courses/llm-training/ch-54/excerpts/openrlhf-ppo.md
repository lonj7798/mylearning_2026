---
chapter: ch-54
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md
source_url: https://github.com/OpenRLHF/OpenRLHF
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF's PolicyLoss — where ch-54 §5's IS correction actually lives

**Source library:** `wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md`
**Artifact:** `openrlhf/models/loss.py` `PolicyLoss.forward` body (≈ lines 68–168) — specifically the three `vllm_is_correction_type` branches and the `vllm_kl` return.

---

## Why this source anchors ch-54

§5 of ch-54 describes async RL architecturally (queue, pause, staleness `k`). The correctness guarantee comes from the loss. This page shows where exactly the IS correction lives in OpenRLHF's code — it is a flat branch inside `PolicyLoss.forward`, not a separate framework. The rollout-vs-train logprob gap is first-class: the loss returns **two** KL numbers, `ppo_kl` and `vllm_kl`, and both are plumbed to the logger.

---

## The return signature §5 treats as contract

From the source (line 13):

> **Core pattern:** Module-style loss object that returns `(loss, clip_ratio, ppo_kl, vllm_kl)`.

ch-54 §5 elevates `vllm_kl` to "the canary" — its name in any RL log should be treated as the first metric you look at. It is computed (line 76 of the excerpt):

```python
vllm_kl = masked_mean(rollout_log_probs − old_log_probs, action_mask, dim=None)
```

This is `KL(π_train || π_rollout)` — the gap between the logprobs your trainer recomputes on the same sequence and the logprobs vLLM produced during generation. If `vllm_kl` > 0.1 and clipfrac is pegged, the async pipeline is broken in one of two ways ([[async-rollout]] §Failure signature).

---

## The three correction modes §5 enumerates

From the source (line 61):

> - **vLLM IS correction** is essential when rollouts are produced by a different inference engine than the trainer's forward pass — `tis` is the simple Truncated IS, `icepop` masks tokens whose IS weight falls outside `[low, high]`, `seq-mask-tis` masks at the sequence level.

And from the code (lines 62–75 of the excerpt):

```python
log_ratio_v = old_log_probs − rollout_log_probs
if self.vllm_is_correction_type == "icepop":
    vllm_is = torch.exp(log_ratio_v).detach()
    vllm_is = vllm_is * ((vllm_is >= low) & (vllm_is <= high))   # zero OOR tokens
    loss = vllm_is * loss
elif self.vllm_is_correction_type == "seq-mask-tis":
    seq_log_ratio = masked_mean(log_ratio_v, action_mask, dim=-1)
    seq_is = torch.exp(seq_log_ratio)
    seq_mask = (seq_is >= low) & (seq_is <= high)               # drop whole seqs
    vllm_is = torch.exp(log_ratio_v).detach()
    loss = seq_mask.unsqueeze(-1) * vllm_is * loss
else:  # "tis"
    vllm_is = torch.exp(log_ratio_v).clamp(min=low, max=high).detach()  # clip
    loss = vllm_is * loss
```

ch-54 §5 orders these by aggressiveness:

- `tis` — clip, keep all tokens.
- `icepop` — zero tokens outside range.
- `seq-mask-tis` — drop entire sequences.

Pick `tis` first; escalate if `vllm_kl` still spikes.

---

## KL-to-ref lives outside the loss

From the source (line 17):

> OpenRLHF separates concerns sharply: `PolicyLoss` is just the clipped surrogate; the KL-to-reference penalty is folded into per-token rewards by the experience builder (`AdaptiveKLController` updates β each iteration based on observed KL, mirroring InstructGPT).

This is why ch-54 §5 can discuss the IS branches without simultaneously tangling with KL-to-ref — they are *different KL terms*, with different purposes:

- `ppo_kl` — train-vs-old monitoring (Schulman K1 in expectation).
- `vllm_kl` — rollout-vs-train sampler-drift diagnostic.
- KL-to-reference — reward-shaping term applied outside the loss, controlled by `AdaptiveKLController`.

Confusing the three is the single most common bug in async RL debugging. ch-54 §5 makes them three separate entries in the canary list.

---

## Why OpenRLHF's factorization matters

From the source (line 96):

> **KL handling:** OpenRLHF and verl both apply token-level KL via reward shaping (so KL becomes part of the GAE/GRPO advantage); TRL adds it as `non_score_reward` for logging only and clips to a controller target.

This factorization — KL *in the reward*, IS correction *in the loss* — is what ch-54 §5 calls "OpenRLHF's lasting contribution." It cleanly separates the "what we want the policy to do" (reward shaping including KL) from the "how we correctly score off-policy data" (IS correction).

---

## Connections

- **ch-54 §5** — the three correction modes and the `vllm_kl` canary.
- **ch-54 §7** — OpenRLHF at the async + no-replay + sync-broadcast corner of the cube.
- **ch-56** — deep dive into OpenRLHF's Ray actor topology and AdaptiveKLController.
- **[[async-rollout]]** — the architecture this loss operates inside of.
- **[[verl-rollout]]** — the sibling implementation with the engine-pause alternative to `vllm_lock`.
