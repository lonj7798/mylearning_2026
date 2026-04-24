---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-rewarding-lm.md
source_url: https://arxiv.org/abs/2401.10020
created_at: "2026-04-23"
---

# Excerpt: Self-Rewarding LM — the canonical "policy as judge" loop

**Source library:** `wiki/raw-data/llm-training/papers/self-rewarding-lm.md`
**Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta AI / NYU)
**Year:** 2024 (arXiv 2401.10020)

---

## Why this source anchors ch-45

Self-Rewarding is the reference row in the ch-45 filter table because it is the
**shortest distance from "I have an SFT checkpoint" to "I have a preference loop."**
Every subsequent method in the chapter is a modification of one arrow in
Self-Rewarding's diagram:

- Meta-Rewarding adds a third role on the judge arrow.
- SPIN replaces the judge with the SFT data distribution.
- Nash-LM replaces the DPO retrain with mirror-descent.
- ReST-EM replaces the DPO retrain with SFT on verifier-filtered samples.
- R1-Zero replaces the whole filter with a rule-based reward inside GRPO.

So read the original diagram (Figure 1 of the paper, summarized in source §Key Figures)
as the common backbone, and understand ch-45's other methods as edits.

---

## The loop, attested verbatim from source lines 33-39

> Preference-pair construction per iteration:
>   1. Sample 4 responses per prompt from the current policy at T=0.7, top-p=0.9.
>   2. Score all 4 with the policy-as-judge (pairwise or 5-point, averaged over 3 judge samples).
>   3. Take the highest-scored response as `chosen`, lowest as `rejected`.
>   4. Run DPO with beta = 0.1 for 1 epoch from the previous iteration's checkpoint.

Four numbers are load-bearing here and every one has an ablation in the paper:

| Knob | Value | What moves if you change it |
|---|---|---|
| K samples/prompt | 4 | K=2 halves the signal range and halves the DPO margin; K=8 doubles judge compute for minor win-rate gain |
| Judge replicas | 3 | single-sample judge is noisy enough that `chosen`/`rejected` gets flipped ~20 % of the time |
| DPO beta | 0.1 | higher beta (0.5) collapses to SFT; lower beta (0.01) causes divergence from the previous iterate |
| Epochs | 1 | 2 epochs overfits and loses the next iteration's judge signal |

Notice the two independent uses of "averaging" — one inside the judge call (3 judge
samples per scoring) and one outside across K=4 generations. The paper does not
ablate collapsing them, so the architecture of the averaging is itself an untested
prior worth knowing you carried forward if you reimplement.

---

## The AlpacaEval curve — the paper's one-line claim

Source lines 19-22 give the numbers ch-45 quotes:

> Demonstrated 3 iterations of Iterative DPO on Llama-2-70B lifts AlpacaEval 2.0 win-rate
> from 9.94 % (SFT) -> 15.38 % -> 20.44 % -> 20.8 %, passing GPT-4 (June 2023) at iter 2.
> Showed the judge's Spearman correlation with held-out human preference improves from
> 0.62 (iter 0) to 0.71 (iter 3) on the same Open-Assistant rubric.

Two curves, not one. The actor improves (9.94 -> 20.8) and the judge improves (0.62 -> 0.71).
This is the emergent property the paper sells: a *frozen* reward model cannot get better
during training; a *policy-as-judge* can, because every DPO step also updates the parameters
the judge forward pass depends on. Ch-45's core insight — that the filter is the whole story —
follows directly.

---

## Saturation at 3 — the ceiling that matters

Source line 39:

> Stopping: 3 iterations — the paper notes iter 4 regresses on reward bench (likely reward hacking).

This is not a soft recommendation. Iter 4 regresses, full stop. The mechanism
(ch-45 §2): both actor and judge distill into a narrower high-score basin;
the judge's systematic errors become the actor's training signal; DPO amplifies them.

Meta-Rewarding ([[excerpts/meta-rewarding-lm]]) is the direct answer to this ceiling —
add a meta-judge and the plateau pushes to iter 5. The structural lesson is that
**self-rewarding without external signal is self-limiting**, and the ceiling is
set by how fast the judge's calibration drifts relative to the actor's gain.

---

## The cost asymmetry — why iteration matters in practice

Source line 40:

> Cost asymmetry: each iteration's judge pass dominates total compute
> (4 generations x 3 judge calls per prompt x ~20K prompts).

4 x 3 x 20K = 240K judge forward passes per iteration, at 70B. That is the
hidden cost of Self-Rewarding. A single RLVR pass ([[ch-44]]) runs the verifier
in microseconds per sample; Self-Rewarding pays a full LLM forward per scoring.
If you have a verifier available, use it — that is why ReST-EM + R1-Zero came
to dominate reasoning pipelines even though Self-Rewarding came first chronologically.

---

## Connections

- Extended by [[excerpts/meta-rewarding-lm]] to push past iter 3.
- Contrasted with [[excerpts/spin]] (uses data as judge, not policy).
- Superseded on verifiable tasks by [[excerpts/rest-em]] and [[excerpts/r1-zero-analysis]].
- Host chapter: [[ch-45]] §2.
- Forward to [[ch-46]] lab: the β=0.1 and K=4 knobs here become the DPO sweep baseline.
