---
chapter: ch-16
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
---

# Excerpt: Tülu 3 — the open-source RLVR prompt-curation recipe

**Source library:** `wiki/raw-data/llm-training/model-reports/tulu-3.md`
**Paper:** Lambert, Morrison, Pyatkin, Huang, Ivison et al. (Allen AI), "Tülu 3: Pushing Frontiers in Open Language Model Post-Training" (Nov 2024).

---

## Why this source anchors ch-16

Tülu 3 is the open-source reference for everything ch-16 claims about RLVR prompt pools. Unlike Kimi K1.5 (partial disclosure, no code) or Kimi K2 (proprietary tool library), Tülu 3 releases the full pipeline: prompts, verifiers, hyperparameters, training code (`open-instruct`). Ch-16 treats it as the "open-source instantiation" of the recipe described abstractly in [[excerpts/rlvr-tulu3]].

Three specific numbers from this source drive the chapter:

1. 939,344 SFT prompts (57% public, 43% synthetic) — the upper bound on what RLVR can filter from.
2. RLVR PPO runs for **10,000,000 episodes** — the episode budget that drives ch-16 §5's "prompt pool exhaustion" argument.
3. KL beta = 0.05, clip ε = 0.2, LR = 3e-7 — the reference config for verifier-grounded PPO.

---

## The prompt-pool construction

From the source (lines 33, 38, 55–58):

> - **Total prompts:** 939,344 (57% public sources incl. WildChat/OpenAssistant, 43% synthetic/in-house).
> - **Preference data size:** hundreds of thousands of pairs, curated from on-policy sampling of the SFT model + reward model ranking.
> - **Verifiers used:**
>   - GSM8K / MATH: exact-match / sympy equivalence.
>   - IFEval: constraint-satisfaction checker.
>   - Code tasks: unit-test execution.

Ch-16's §1 claim — "the RL pool is typically a subset of the SFT pool, not a superset" — comes from exactly this structure. The 939K SFT prompts feed the full pipeline; DPO draws its preferences from a subset, and RLVR draws only from the subset that has a verifier. The paper does not state the exact RLVR-eligible prompt count, but a plausible estimate is 10^4–10^5 based on which SFT sources pair naturally with verifiers (math, code, IFEval constraints).

---

## The 10M-episode budget and why exhaustion matters

From the source (line 54):

> **Total episodes:** 10,000,000.

With `K = 8` rollouts per prompt and a ~10^5-prompt RLVR-eligible pool, 10M episodes means each prompt is visited ~12 times over the run. At 12× visitation rate, a prompt's reward variance has plenty of opportunity to collapse — once the policy solves it, the zero-variance downweight from [[excerpts/replay-buffer-rlhf]] kicks in and the prompt is effectively ejected from sampling.

Ch-16's §5 "prompt-pool exhaustion" section uses exactly this arithmetic. The lesson is not "10M is too much" but "10M against 10^5 prompts is the regime in which you need synthetic-data regeneration to keep the pool fresh." Tülu 3 doesn't have a synthetic-regeneration loop (it runs once and stops); Kimi K2 and the Track 3 synthetic-prompt generators do.

---

## The RLVR config block — what the chapter's reference numbers come from

From the source (lines 47–54):

> - **Algorithm:** PPO (not GRPO).
> - **Learning rate:** 3e-7
> - **Beta (KL coeff):** 0.05
> - **Clip epsilon:** 0.2
> - **PPO update epochs (K):** 4
> - **Mini-batches per update (N_mb):** 1
> - **GAE lambda:** 0.95; **gamma:** 1.0
> - **Local mini batch size:** 32; **local rollout batch size:** 32.

Ch-16 doesn't reproduce this block (that's the next course's job in the RL-algorithms chapter), but it references two numbers from it:

- **Clip ε = 0.2** — the IS-ratio analysis in ch-16 §3.2 uses this as the implicit clip bound. `E[ρ] > 1.2` means most samples will be hard-clipped.
- **Beta KL = 0.05** — small enough that the policy can move away from the SFT reference, large enough to prevent degenerate collapse. Ch-16 §1 cites this as the "small KL penalty" that RLVR runs with.

---

## What Tülu 3 does *not* do — the absence of a replay buffer

The source does not mention prompt-level replay. Tülu 3's flow is the classical `rollout → train → discard`. This is not an oversight; with a ~10^5 prompt pool and `K = 8` fresh rollouts, the zero-variance downweight happens *automatically* — prompts that become solved simply produce zero-gradient batches and get re-sampled fresh next step, while the ones with `p̂ ∈ (0, 1)` continue to produce useful signal. Explicit replay buffers become load-bearing only in smaller-pool regimes (where you want to concentrate on high-variance prompts) or in long-rollout regimes (where partial rollouts save compute).

Ch-16's §6 manager presents replay as optional (`p_replay = 0.25` default); setting `p_replay = 0` recovers the Tülu-3-style no-buffer flow.

---

## What this excerpt unlocks

- **ch-16 §1** — verifier domain taxonomy (math, IFEval, code) is the Tülu 3 list verbatim.
- **ch-16 §2** — the `[0.1, 0.9]` "Tülu-3 fixed band" is inferred from the paper's not-explicitly-filtered-by-pass-rate approach (the band is wide because the filter is mostly passive).
- **ch-16 §5** — 10M-episode budget vs ~10^5 pool is the quantitative backbone of the exhaustion argument.

## Connections

- [[excerpts/rlvr-tulu3]] — the methodology paper; this excerpt is the *recipe*, that one is the *framework*.
- [[excerpts/replay-buffer-rlhf]] — contrast: Tülu 3 does not use replay, which works because of pool size.
- [[excerpts/on-off-policy-rlhf]] — Tülu 3's PPO-with-KL is the on-policy flavor that avoids the DPO distribution-shift problem.
- [[ch-16]] — §1, §2 (fixed band), §5 (exhaustion).
