---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/generative-reward-models.md
source_url: https://arxiv.org/abs/2410.12832
created_at: "2026-04-23"
---

# Excerpt: Generative Reward Models — judge = reward = log-prob

**Source library:** `wiki/raw-data/llm-training/papers/generative-reward-models.md`
**Papers:** Mahan et al. 2024 "Generative Reward Models" (2410.12832); Ankner et al. 2024 "Critique-out-Loud Reward Models" (2408.15240)

---

## Why this source is the bridge from "judge" to "reward"

Ch-49 §6 argues that "judge" and "reward" are the same math used at different phases. This paper is the source for that claim. The identity

> `r(x, y) = log P_RM("A is better" | x, y_A, y_B, rubric)`

collapses the distinction entirely. An LLM judge that emits a verdict token has an implicit scalar reward in its log-probs, and that reward can be trained against, optimized against, or simply read at eval time.

---

## The construction

Source §Key Contributions:

> "GenRM scoring: `r(x, y) = log P_RM('A is better' | x, y_A, y_B, rubric)` -- or a soft margin between 'A' and 'B' tokens."
> "Critique-then-verdict (CoT-RM): sample a critique `c ~ P_RM(.|prompt)` first, then score the verdict given the critique -- accuracy improves 3-10 pp on RewardBench over no-CoT."
> "Training: fine-tune the LM with next-token supervision on (prompt, critique, verdict) triples; no dedicated scalar head -- keeps the RM in the same model family as the policy."

Three things follow that ch-49 uses:

1. A judge is not architecturally distinct from an RM; it is a prompt-and-tokenization contract over the base LM.
2. CoT-before-verdict is not optional styling — it is worth 3–10 pp RewardBench accuracy. Ch-49 §4 template #2 takes this seriously.
3. Training a judge on (prompt, critique, verdict) triples is exactly what Con-J and J1 do downstream ([[direct-judgement-preference]]).

---

## Calibration — the §7 claim

Source §Key Contributions:

> "Calibration: the LM's verdict probability is reliably tied to ground-truth agreement -- useful as an uncertainty signal (feeds back into reward-ensembling-style LCB combinations)."

And §Key Figures:

> "Fig. 4 (calibration plot) -- generative RMs are well-calibrated where BT RMs are overconfident."

Panel 2 of `figures/judge-bias.html` seeds the BT vs GenRM curves from this figure. Note the precise wording: "calibrated *where* BT RMs are overconfident" — not uniformly. Ch-49 §7 elaborates: calibration is rubric-conditional.

---

## Rubric as policy knob

Source §Key Contributions:

> "Robustness: when the rubric is extended to say 'longer is not better, be concerned if the response is sycophantic', the RM generalizes those constraints to unseen prompts -- the RM is steerable via its own context, which scalar RMs cannot be."

This is the structural reason ch-49 §3 can recommend "rubric extensions" as corrections for verbosity and formatting biases. The judge's context is the steering. Scalar BT RMs have no equivalent lever.

---

## Compute trade-off

Source §Key Contributions:

> "Compute trade-off: GenRMs are slower (need to generate critique tokens) but reuse the base-LM inference stack and scale with model capability."

Ch-49 §6 encodes this as "RL-time RM vs eval-time judge" split. RL-time wants per-token speed (short-rubric GenRM or PairRM); eval-time wants critique depth (long-rubric CoT GenRM). Same math, different compute budgets.

---

## The failure modes it still has

Source §Technical Details:

> "Failure modes: verbosity bias and self-enhancement (see judge-llm-bias) still apply; mitigated by rubric wording and by using a judge from a different model family than the policy."

GenRM is not a *fix* for judge bias — it is a *platform* on which bias corrections can be applied. Ch-49 §5(d) (cross-family judging) is the structural mitigation that survives even the generative construction.

---

## Connections

- `read.md` §6 — "judge = reward = log-prob" identity from this paper.
- `read.md` §7 — BT-vs-GenRM calibration curve comes from Fig. 4 of this paper.
- `figures/judge-bias.html` Panel 2 — BT and GenRM curves seeded from source Fig. 4 shape.
- [[direct-judgement-preference]] (Con-J / STE / J1) — extends this construction to the training-free synthetic-judge line.
- [[pairrm]] — the joint-encoder pairwise RM that motivates GenRM's "put both responses in the context" structure.
