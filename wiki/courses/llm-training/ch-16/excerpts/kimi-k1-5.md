---
chapter: ch-16
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/kimi-k1-5.md
source_url: https://arxiv.org/abs/2501.12599
created_at: "2026-04-23"
---

# Excerpt: Kimi K1.5 — partial rollouts, difficulty mining, and the prioritized-sampling curriculum

**Source library:** `wiki/raw-data/llm-training/model-reports/kimi-k1-5.md`
**Paper:** Kimi Team (Moonshot AI), "Kimi k1.5: Scaling Reinforcement Learning with LLMs" (Jan 2025).

---

## Why this source anchors ch-16

Kimi K1.5 is the single clearest 2025 demonstration of the chapter's thesis — that curriculum, difficulty mining, and replay-buffer-backed infrastructure are load-bearing, not cosmetic. Three distinct pieces of K1.5's recipe map one-to-one onto ch-16 sections:

1. **Pass-rate measurement at the SFT policy** (source line 38) → ch-16 §2's difficulty-mining protocol.
2. **Partial rollouts from a replay buffer** (source line 20) → ch-16 §3.5's "legitimate exception" to the no-trajectory-replay rule.
3. **Prioritized sampling weighted by `1 − success_rate`** (source line 38) → ch-16 §4's prioritized curriculum.

Ch-16 cites K1.5 at §2 (the protocol), §3.5 (partial rollouts), and §4 (the `1 − p` prioritization). The learning-economic pressure K1.5 operates under — 128K-token RL context, expensive long-CoT rollouts — is what forced these innovations to be public in the first place.

---

## The pass-rate measurement protocol

From the source (lines 24, 38):

> **Curriculum + prioritized sampling** for prompt selection — difficulty from pass-rate of 10 SFT samples at high temperature.
>
> **Self-improvement / iterative:** curriculum sampling and prioritized sampling (problems sampled ∝ 1 − success-rate) drive implicit self-curriculum.

Three operational details are packed in:

- **K = 10 rollouts for the measurement**, not the larger `K = 64` used for the GRPO gradient step. Measurement is a one-shot cost; making it cheap matters.
- **Measured against the SFT-stage policy**, not the live RL policy. The frozen measurement is how K1.5 keeps the protocol tractable; the prioritized-sampling loop compensates by re-ranking prompts as success_rate shifts during training.
- **"High temperature"** — the measurement is conservative-pessimistic about a prompt's difficulty. A high-temperature sample is noisier than a low-temperature sample, so a prompt whose `p̂ = 0.5` at high temperature may have `p̂ = 0.9` at low temperature. The filter is sizing for the temperature the actual rollouts will use.

Ch-16's §2 tabulates the K1.5 band as `[0.05, 0.5]` (hard-skew). The source doesn't quote exact thresholds; ch-16 infers them from the "prioritized sampling ∝ 1 − success_rate" weighting, which effectively concentrates probability mass below 0.5.

---

## Partial rollouts — the legitimate exception

From the source (line 20):

> **Partial rollouts:** long responses are broken into segments across iterations; previous trajectory segments are reused from a replay buffer — fixed output token budget per iteration.

This is ch-16 §3.5's "looks like trajectory replay, but isn't." The reason is in the geometry: the reused segment is a **prefix** (prompt + early reasoning). Subsequent token positions are rolled out fresh under the current policy. The importance-ratio argument from [[excerpts/replay-buffer-rlhf]] applies to the *gradient target* (the tokens on which `∇ log π_θ` is taken), not to the context. Context is passive — it conditions the distribution but does not require an IS correction.

This is why partial rollouts scale to 128K context without quadratic cost: the replay buffer amortizes the prefix across iterations (you don't re-generate the 60K-token chain-of-thought prefix to sample the next 8K continuation), but the gradient only flows through freshly generated tokens.

**Implementation implication for the §6 manager:** the current chapter's `RLPromptPool` is a *pure* prompt pool — it does not implement partial rollouts. A K1.5-style system needs a separate `PrefixBuffer` alongside it, storing `(prompt_ids, prefix_ids, prefix_length, last_iteration)`. Ch-16's §6 references this structure without implementing it; a follow-up lab chapter could build it out.

---

## Why the prioritized sampling loop re-ranks against the live policy

From the source, the method section describes curriculum sampling and prioritized sampling as distinct mechanisms that compose. The operational interpretation:

- **Curriculum sampling** = stage-wise restriction of the prompt pool. Start with a broad set; narrow to harder-only prompts as the run progresses.
- **Prioritized sampling** = within-stage weighting. Inside the current stage, sample prompts with weight `∝ 1 − success_rate_i` where `success_rate_i` is updated *per prompt per epoch* against the live RL policy.

The re-measurement step is what keeps the curriculum responsive to policy improvement. A prompt that started the run at `p̂ = 0.3` and now sits at `p̂ = 0.8` has its sampling weight drop from `0.7` to `0.2` — it's still in the pool, but it's seen ~3.5× less often. Without this, the "hard" prompts of step 0 would dominate sampling for the whole run and saturate into zero-variance territory.

Ch-16's §2 ("Annealing requires re-measurement") and §6's `remeasure_if_due` method both derive from this design.

---

## The length-penalty tie-in — why pass-rate alone isn't enough

From the source (line 34):

> **Length penalty:** `len_reward(i) = {λ if correct; min(0, λ) if incorrect}` with `λ = 0.5 − (len(i) − min_len)/(max_len − min_len)`. Penalizes long-correct answers and long-incorrect answers; warmed up gradually, not applied from step 0.

This is off-axis for ch-16 (it's a *reward-shaping* trick, not a *prompt-curation* trick), but it matters for the chapter's §2 band-tuning discussion. A prompt with `p̂ = 0.9` at short-response length may have `p̂ = 0.4` at long-response length — K1.5's length penalty effectively discovers this by penalizing long-correct answers. The pass-rate measurement depends on what response-length regime the model is operating in, and K1.5's warmup schedule is how they keep that regime stable during the measurement.

---

## What this excerpt unlocks for the next chapters

- **ch-16 §2** uses the K=10 + high-temperature + frozen-SFT measurement protocol as the reference.
- **ch-16 §3.5** distinguishes partial rollouts from trajectory replay.
- **ch-16 §4** tabulates K1.5's schedule as the "prioritized `1 − p`" row.
- **Track 4 (RL)** — K1.5's online policy mirror descent is referenced in the RL-algorithm chapter, but the *prompt side* is already fully covered in this chapter.

## Connections

- [[excerpts/replay-buffer-rlhf]] — the general no-trajectory-replay rule; K1.5 is the exception that proves the rule.
- [[excerpts/kimi-k2]] — the K2 follow-up extends this infrastructure to agentic long-horizon tasks.
- [[excerpts/tulu-3]] — contrast: Tülu 3 uses a fixed band, K1.5 uses re-measured prioritization.
- [[ch-16]] — §2 (protocol), §3.5 (partial rollouts), §4 (curriculum).
