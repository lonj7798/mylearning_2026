---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlvr-tulu3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
---

# Excerpt: RLVR — skip the reward model

**Source library:** `wiki/raw-data/llm-training/papers/rlvr-tulu3.md`, `wiki/raw-data/llm-training/model-reports/tulu-3.md`
**Anchor paper:** Lambert et al. 2024 — "Tülu 3: Pushing Frontiers in Open Language Model Post-Training"

---

## Why this source anchors ch-44

Tülu-3 is the first fully open recipe that ships a concrete RLVR implementation with every hyperparameter disclosed. If process supervision is the step-level escape hatch from RM-based training, RLVR is the outcome-level one — and Tülu-3 is where the numbers live.

---

## The formal reward — verbatim

From `rlvr-tulu3.md` §Key Contributions:

> **Formal RLVR setup:** for a prompt `x` paired with a verifier `v: (x, y) → {0, 1}`, the reward is simply `r(x, y) = v(x, y)` — no RM.

That is the entire contribution, at the formula level. Every implementation detail (verifier domains, PPO config, KL coefficient) follows from picking this functional form.

---

## The three verifier domains — verbatim

From `rlvr-tulu3.md` §Key Contributions:

> **Three verifier domains used in Tülu 3:**
>   - *Math:* extract the final numeric/symbolic answer and compare to the reference using a tolerant grader (SymPy / normalized string match on MATH, exact integer match on GSM8K).
>   - *Constrained instruction following:* IFEval-style constraints ("respond in JSON", "use exactly 3 bullet points") checked with regex / parsers.
>   - *Code:* run model-generated code against unit tests in a sandbox; reward = 1 iff all tests pass.

Three patterns worth separating:

| Verifier | Cost per call | Failure mode |
|----------|---------------|---------------|
| Math (SymPy) | ~1 ms | grader accepts "42" inside prose |
| IFEval regex | ~ms | regex under- or over-matches |
| Code unit tests | seconds + sandbox | timeouts, flaky tests, undefined behaviour |

The first two are effectively free; the third is the budget constraint for RLVR-code runs. Tülu-3 runs code verifiers with isolate-style sandboxes and a 5 s per-rollout timeout.

---

## The PPO config — verbatim from tulu-3.md

From `tulu-3.md` §Technical Details — RLVR:

> **Algorithm:** PPO (not GRPO).
> **Learning rate:** 3e-7
> **Beta (KL coeff):** 0.05
> **Clip epsilon:** 0.2
> **PPO update epochs (K):** 4
> **Mini-batches per update (N_mb):** 1
> **GAE lambda:** 0.95; **gamma:** 1.0 (episodic)
> **Local mini batch size:** 32; **local rollout batch size:** 32.
> **Total episodes:** 10,000,000.

And the verifier list:

> **Verifiers used:**
>   - GSM8K / MATH: exact-match / sympy equivalence.
>   - IFEval: constraint-satisfaction checker.
>   - Code tasks: unit-test execution.

This is the `open-instruct` RLVR config block, reproducible from the released repo. The `LR = 3e-7` is an order of magnitude below a typical SFT learning rate precisely because a 0/1 reward with small KL has high variance — see ch-43 for why large steps under small KL blow up entropy.

---

## Why Goodhart's gap is zero — verbatim

From `rlvr-tulu3.md` §Key Contributions:

> **Why it sidesteps reward hacking:** the verifier is a fixed, interpretable function. There is no proxy RM to drift; there is no OOD region where the reward spuriously rises. Goodhart's gap (see **[[reward-model-overoptimization]]**) is mechanically zero on verifiable prompts.

The "mechanically zero" phrase is strong and precise. It is zero because the proxy is the target — there is no gap between what you measure and what you want, only a gap between what you measure and what you *intended* to measure (the verifier-bug class). From `rlvr-tulu3.md` §Technical Details:

> **Failure mode to watch:** if the verifier has loopholes (string-match math graders that accept "42" inside prose), RLVR can hack those loopholes. Treat verifier engineering like unit-test engineering.

That is the one non-zero risk: the verifier itself is a program, and programs have bugs. The hacking is deterministic and auditable once found, which is not true of RM drift.

---

## What RLVR actually buys — verbatim

From `tulu-3.md`:

> Measured gains relative to DPO-only checkpoint: +5–10pp on GSM8K, +~4pp on IFEval, neutral-to-positive on other evals. No reward hacking observed because the verifier is ground-truth.

The gains are modest — RLVR is not a revolution on top of DPO; it is a clean, hack-proof marginal improvement on tasks where a verifier exists. The revolutionary claim is in the *risk* profile, not the absolute delta: gains without a Goodhart surface.

---

## Prompt curation — verbatim

From `rlvr-tulu3.md` §Key Contributions:

> **Prompt curation:** only prompts with a verifier + a known reference answer enter the RLVR set; RLHF/DPO handles the rest.

Carry this into ch-46 lab framing. The RLVR prompt set is curated *before* training — there is no "this prompt is ambiguous, try the RM instead" fallback at training time. Every prompt in the RLVR set has a verifier implementation and a reference answer. Everything else is handled in earlier stages (DPO).

---

## Carry into ch-44

- §6 of read.md uses the exact formula `r(x, y) = v(x, y) in {0, 1}` and the full Tülu-3 config block.
- The `+5..+10 pp` GSM8K delta is the benchmark the ch-46 lab's RLVR option has to reach.
- "Verifier engineering is unit-test engineering" is the risk framing used in §6.
- Connects back to ch-42 (reward hacking) by explaining why RLVR reduces Goodhart's gap to mechanically zero — the taxonomy of ch-42 is about learned RMs, and RLVR removes the learned RM.
