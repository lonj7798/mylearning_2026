---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prm800k.md
source_url: https://arxiv.org/abs/2305.20050
created_at: "2026-04-23"
---

# Excerpt: PRM800K — the step-level label protocol

**Source library:** `wiki/raw-data/llm-training/papers/prm800k.md`, `wiki/raw-data/llm-training/papers/lets-verify.md`, `wiki/raw-data/llm-training/papers/let-verify.md`
**Anchor paper:** Lightman et al. 2023 — "Let's Verify Step by Step"

---

## Why this source anchors ch-44

PRM800K is the dataset every process-reward paper since cites as its gold. It defines what a "step label" *is*, what "process supervision" operationally means, and it is the only large-scale publicly released human-labeled PRM dataset. Every automated alternative ([[math-shepherd]], [[omegaprm]], [[rstar-math]]) is explicitly engineered to approximate PRM800K cheaply.

---

## The label unit — verbatim from the raw-data page

From `prm800k.md` §Technical Details:

> **Step separator:** a newline or a literal "Step k:" token; the PRM emits a score at that position using its hidden state.
> **Labels per step:** `+1` correct, `-1` incorrect, `0` neutral (ambiguous or filler). Training loss is on non-neutral steps only.

And from `let-verify.md` §Technical Details:

> **Labeling protocol:** labelers see one step at a time, mark in {positive, negative, neutral}; first negative step is the failure point.

The labeling UI intentionally shows the labeler *one step at a time*, not the whole solution. This is not cosmetic — it is what makes the labels cheap enough to collect at 800K scale. The labeler makes a three-way choice, not an open-ended critique.

---

## The size — verbatim

From `prm800k.md` §Key Contributions:

> **PRM800K dataset:** ~800K step-level labels (`correct / incorrect / neutral`) on ~75K GPT-4-generated solutions to the MATH competition dataset.
> **Cost:** step-level labeling is ~10x more expensive per example than outcome labeling — the paper explicitly notes active learning is needed to make PRMs practical.

The ~10x multiplier is the single number you should carry from this source. It is the reason Math-Shepherd and OmegaPRM exist.

---

## The aggregation functions

The paper reports three scoring functions tested for reducing per-step scores to a solution-level score:

```
prod:         S = prod_t p_correct(step_t)                    # Lightman 2023 default
min:          S = min_t  p_correct(step_t)                    # Math-Shepherd default
softmax-avg:  S = softmax-weighted average of p_correct       # smoother, rarely used
```

From `let-verify.md` §Technical Details the choice is clear:

> **Scoring a full solution:** multiply per-step "good" probabilities -> solution score.

But Math-Shepherd's Table 4 (cited in ch-44 §2) ablates all three on GSM8K and MATH and finds `min` Pareto-dominates `prod` and `mean`. Both are valid; the chapter's guideline is `min` for long chains where a single bad step should kill the whole score.

---

## Active learning — the 2.6x multiplier

From `prm800k.md`:

> **Active learning:** prioritizing labeling on solutions where the current PRM is uncertain or disagrees with an ORM gives a ~2.6x data efficiency multiplier.

And from `let-verify.md`:

> **Active learning:** rank unlabeled steps by model uncertainty (entropy on the good/bad head); label top quantile.

Operationally: train a PRM on a small uniform-sampled slice, score the unlabeled pool, send the highest-uncertainty steps (or the steps where PRM says "good" but the final outcome was wrong — the "convincing-wrong" slice) back to the labelers. Repeat. The 2.6x number means a 38% label budget reaches the same PRM quality as 100% uniform labeling.

---

## Why this carries forward

Three downstream chapters inherit this protocol:

1. **Ch-44 §3** uses the PRM-vs-ORM table from the same paper (MATH-500 78.2 PRM vs 72.4 ORM vs 69.6 majority).
2. **Ch-44 §4** shows that Math-Shepherd's `MC(s_t) = (1/K) sum I[rollout reaches gold]` is *exactly* the label schema PRM800K defines, with the human labeler swapped for a rollout policy and the binary label softened to a fraction.
3. **Ch-46 lab** references PRM800K as an open training set for the PRM option if the learner chooses the step-supervised path.

The invariant across all of them is: step separator + per-step label + aggregator. Every variation — MC rollouts, divide-and-conquer, pairwise preferences — mutates one of those three pieces while keeping the other two.

---

## What PRM800K cannot tell you

Two things this source is silent on, which the chapter fills from adjacent papers:

1. **Process labels + RL.** PRM800K trains a PRM as a Best-of-N reranker only. Math-Shepherd is the paper that slots the PRM as a dense reward in PPO via `R_total = r_final + lambda * sum PRM(step_t)`.
2. **Scaling without humans.** PRM800K stops at 800K because the labelers stop. Scaling past that requires Math-Shepherd / OmegaPRM — same label schema, different labeler.

These gaps are why the chapter covers four papers and not one.
