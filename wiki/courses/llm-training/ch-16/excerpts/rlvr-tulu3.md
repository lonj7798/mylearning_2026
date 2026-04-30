---
chapter: ch-16
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlvr-tulu3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
---

# Excerpt: RLVR (Tülu 3 methodology) — the verifier-as-reward invariant

**Source library:** `wiki/raw-data/llm-training/papers/rlvr-tulu3.md`
**Paper:** Lambert, Morrison, Pyatkin, Huang, Ivison et al. (Allen AI), "Tülu 3" — RLVR methodology subpage.

---

## Why this source anchors ch-16

Where [[excerpts/tulu-3]] gives the *recipe* (hyperparameters, prompt counts, verifier list), this subpage gives the *framework*: the formal definition of RLVR, the prompt-curation invariant, and the Goodhart-gap argument for why verifier-grounded reward collapses to zero proxy/gold drift. Ch-16 §1 leans on this source for its definition of "what an RL prompt is."

---

## The formal RLVR setup and the prompt-curation invariant

From the source (lines 18, 25):

> **Formal RLVR setup:** for a prompt `x` paired with a verifier `v: (x, y) → {0, 1}`, the reward is simply `r(x, y) = v(x, y)` — no RM.
>
> **Prompt curation:** only prompts with a verifier + a known reference answer enter the RLVR set; RLHF/DPO handles the rest.

This is the invariant that drives ch-16 §1's three rules for what an RL prompt must be. Specifically:

1. **Every prompt must be paired with a grader.** The source makes this an absolute — "only prompts with a verifier + a known reference answer enter the RLVR set." Ch-16 takes this as the first pool-eligibility condition.
2. **The grader returns {0, 1}.** Not a continuous score. This shapes the group-baseline variance: under binary rewards, variance is `p(1 − p)`, which the chapter's §2 and the rollout-passrate HTML figure both use as the primary quality signal.
3. **The verifier is fixed.** Not learned, not drifting. This collapses the Goodhart gap that plagues classical RLHF — a learned RM can be hacked; a SymPy equivalence check cannot.

---

## The three verifier domains

From the source (lines 19–23):

> - *Math:* extract the final numeric/symbolic answer and compare to the reference using a tolerant grader (SymPy / normalized string match on MATH, exact integer match on GSM8K).
> - *Constrained instruction following:* IFEval-style constraints ("respond in JSON", "use exactly 3 bullet points") checked with regex / parsers.
> - *Code:* run model-generated code against unit tests in a sandbox; reward = 1 iff all tests pass.

These map onto the verifier-type table in ch-16 §1. The source's critical methodological note (line 39):

> **Failure mode to watch:** if the verifier has loopholes (string-match math graders that accept "42" inside prose), RLVR can hack those loopholes. Treat verifier engineering like unit-test engineering.

This is the "verifier hacking" category that ch-16 references implicitly when it says "the verifier is a fixed, interpretable function" (§1). The chapter doesn't belabor the point — it belongs to the reward-engineering chapter in the RL track — but it underpins why the filter in §2 operates on rollout pass-rate and *not* on rollout reward distribution alone. If the verifier is loophole-prone, reward alone is misleading; pass-rate relative to a held-out reference is more robust.

---

## Why KL control is still needed

From the source (line 24):

> **KL control still used:** standard KL-to-SFT penalty (per-token, added to reward) with a small β, otherwise the policy collapses to a degenerate high-reward mode like constant-answer-guessing.

This is the RLVR-specific version of a broader point: the chapter's §3.3 "KL drift" pathology (when replaying trajectories against a moved `π_ref`) is exactly this KL penalty becoming stale. In live Tülu 3 runs, `π_ref` is frozen at the SFT checkpoint, which is why the KL penalty stays consistent. If a replay buffer stored rollouts from step 1 and `π_ref` remained constant, the KL-reward piece would still be valid — but the policy-gradient piece would still suffer the IS-ratio explosion. That's why [[excerpts/replay-buffer-rlhf]] concludes "replay prompts, not trajectories" and not "replay trajectories against frozen references."

---

## Why this collapses the proxy/gold gap

From the source (line 23):

> **Why it sidesteps reward hacking:** the verifier is a fixed, interpretable function. There is no proxy RM to drift; there is no OOD region where the reward spuriously rises. Goodhart's gap (see [[reward-model-overoptimization]]) is mechanically zero on verifiable prompts.

This is load-bearing for ch-16's *bridge to synthetic* (§5). The synthetic-prompt generator (Track 3) produces prompts that must carry verifiers; the Goodhart-zero guarantee is what makes synthetic-prompt RL scalable. If we were using learned reward models, each synthetic prompt would inject its own RM-drift risk. Verifier-grounded prompts don't. That structural property is why the 2025 pattern is "synthetic prompts + mechanical verifiers" rather than "synthetic prompts + synthetic reward models."

---

## What this excerpt unlocks

- **ch-16 §1** — verifier taxonomy and the "grader is mandatory" rule.
- **ch-16 §3.3** — KL-drift pathology connects to the frozen-`π_ref` setup described here.
- **ch-16 §5** — the Goodhart-zero property is why synthetic prompts are viable.

## Connections

- [[excerpts/tulu-3]] — the same paper from the recipe angle.
- [[excerpts/replay-buffer-rlhf]] — KL control interacts with trajectory-replay bias.
- [[ch-16]] — §1, §3.3, §5.
