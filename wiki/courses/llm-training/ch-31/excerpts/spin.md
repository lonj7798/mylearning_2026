---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/spin.md
source_url: https://arxiv.org/abs/2401.01335
created_at: "2026-04-23"
---

# Excerpt: SPIN — iterative DPO without preference data, via human-text-as-chosen

**Source library:** `wiki/raw-data/llm-training/papers/spin.md`
**Artifact:** the SPIN self-play loop that converts SFT data into a DPO signal

---

## Why this source anchors ch-31

SPIN (Chen et al. 2024) is the one algorithm in the iterative-SFT family that works with *zero* preference labels and *zero* reward model — just an SFT dataset. It gives ch-31 the "no RM, no verifier" branch of the decision tree a concrete algorithm to recommend. Without SPIN, the decision tree's node 4 ("NO verifier -> ...") has no productive path; with SPIN, the productive path is "treat the human response as chosen and the previous-iter sample as rejected, run DPO, repeat."

---

## The attested recipe — quoted

From the source (lines 30–36):

> **Per iteration:**
> 1. Sample 50K (prompt, response) pairs from `\pi_{t-1}` at T=1.0.
> 2. Build DPO pairs (`chosen=y_human`, `rejected=y_gen`) 1:1 with the SFT data.
> 3. DPO-train `\pi_t` from `\pi_{t-1}`: beta=0.1, lr=5e-7, 3 epochs, batch 64.
> 4. Reset reference to `\pi_{t-1}` for the next iteration.

And from the source (lines 7–8):

> SFT data alone yields a DPO signal if you treat the human-written response as "chosen" and the *previous iteration's* model sample as "rejected" — no reward model, no human preference labels, just iterated self-play until the policy distribution matches the data.

> When you have SFT data but no preference data, run SPIN: each iteration, sample a response from `\pi_{t-1}` and DPO-train `\pi_t` with (human_response, `\pi_{t-1}_sample`) as the chosen/rejected pair.

---

## Why SPIN is in the iterative-SFT family, not the preference-DPO family

SPIN's loss is DPO's loss algebraically. But the *source of the preference signal* is not a reward model, not a human annotator, and not an LLM-judge — it is the human text itself, treated as always preferred over any model sample. This shifts SPIN out of the [[dpo]] / [[ipo]] family and into the iterative-SFT family.

Mechanism: the DPO objective pushes `\pi_t` to assign higher log-prob to `y_human` than to `y_{previous model}`. As iterations proceed, `y_{previous model}` improves, so the "negative" side of the contrast gets harder and harder — the policy has to keep closing the gap between its own samples and the human text. At equilibrium, `\pi_t` generates samples that are indistinguishable from human text under the DPO margin, which by SPIN's Theorem 4.1 is the fixed point where `\pi_t = \pi_{data}`.

Ch-31's decision tree node 4 ("no verifier, no RM, but have SFT data -> SPIN") points at this. It is the SFT-data-only escape hatch.

---

## What SPIN does NOT do

From the source (lines 7, 8):

SPIN does not require preference labels. It does not require a verifier. It does not require a reward model. It *does* require:

- An SFT dataset with high-quality human (or teacher) responses.
- Willingness to re-train DPO 3 times (3 iterations × 3 epochs per iteration = 9 SFT-equivalents of compute).
- A policy capable enough that its samples are not trivially bad — otherwise the DPO contrast is too easy and the gradient dies.

The last point matters: SPIN does not work on a barely-SFT'd model, because the early-iter "negative" samples are so bad that the margin between chosen and rejected is saturated and the DPO loss goes to zero without updating the policy. The paper starts from a Mistral-7B SFT'd on UltraChat-200K — already a competent policy.

---

## The SPIN - Self-Rewarding - Meta-Rewarding ladder

Ch-31 §7 treats SPIN, Self-Rewarding LM, and Meta-Rewarding LM as points on the same ladder, with different judges:

| Algorithm | Judge / preference source | Ceiling |
|-----------|--------------------------|---------|
| SPIN | Human text (fixed) | Distribution match (Theorem 4.1) |
| Self-Rewarding LM | Policy-as-judge | 3 iterations before reward drift |
| Meta-Rewarding LM | Policy-as-judge + meta-judge | 4+ iterations, at 2x compute per round |

The observable pattern: the richer the judge, the higher the ceiling, but the more fragile the iteration. SPIN's judge is the human text — it cannot drift, but its ceiling is exactly the data distribution. Self-Rewarding's judge can improve but can also reward-hack. Meta-Rewarding stabilizes the judge at extra cost.

Ch-31's decision tree "stop at 3 iterations" default is calibrated for Self-Rewarding-style setups. SPIN's ceiling is data-bounded, so it saturates naturally at distribution match (paper reports monotone gain for 3 iters, minor gain after). Meta-Rewarding's ceiling is higher but so is the per-round cost.

---

## What SPIN is related to but not identical to

From the source (line 41):

> Related to [[rejection-sampling-finetuning]] (Llama-2 RSFT): both refine via self-samples, but RSFT trains on `chosen_only`, SPIN uses both chosen and rejected.

This is the precise distinction. RSFT throws away the non-top-1 samples. SPIN uses all non-human samples as rejected side of a contrast. The DPO loss then pushes toward human-likelihood and away from self-sample-likelihood simultaneously.

For ch-31, this means SPIN is a *DPO-shaped* iterative algorithm, not an SFT-shaped one, despite living in the "no RM" branch. Node 4 of the decision tree routes to SPIN specifically when you have SFT data and want a preference-optimization signal without paying for preference labels.

---

## Ch-31's take on SPIN

- Use when: you have clean SFT data, no RM, no verifier, but the policy is already competent (post-SFT).
- Do not use when: the policy is barely SFT'd (gradient dies), or the SFT data is teacher-generated (you are distilling from yourself in a circle).
- Stop at: 3 iterations, matching the paper's reported plateau.
- Failure mode: if the 1:1 human:generated pair ratio is off, the margin saturates and the loss stops updating.

---

## Connections

- [[dpo]] — the loss SPIN uses algebraically.
- [[self-rewarding-lm]] — the LLM-judge sibling.
- [[meta-rewarding-lm]] — the meta-judge sibling that extends the iteration ceiling.
- [[self-play-preference]] — Nash-MD's game-theoretic cousin; SPIN's equilibrium is distribution-matching, Nash-MD's is preference-matching.
- [[west-of-n]] — synthetic-preference construction generalized.
- **ch-31 decision tree node 4** — routes to SPIN when no RM/verifier is available.
- **ch-31 §7** — SPIN vs Self-Rewarding vs Meta-Rewarding comparison.
