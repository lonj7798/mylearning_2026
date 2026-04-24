---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/judge-llm-bias.md
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
---

# Excerpt: Judging LLM-as-a-Judge — the judge-variance floor that bootstrap must include

**Source library:** `wiki/raw-data/llm-training/papers/judge-llm-bias.md`
**Artifact:** position bias (20–30% flip), verbosity bias, self-enhancement bias; swap-and-average + reference-guided mitigations.

---

## Why this source is ch-51's "LLM-as-judge" variance anchor

Every go/no-go memo that uses an LLM judge (pairwise win-rate, MT-Bench-style numeric score, RLAIF label) inherits a variance floor that is *not* reducible by adding more items. It is structural — the judge itself is noisy. Ch-51's §1 variance table lists "judge variance" as a standalone row for this reason.

Source §Key Contributions:

> Position bias: A vs B ordering changes the winner in ~20–30% of cases; mitigated by swap-and-average or "two-game" scoring.
> Verbosity bias: longer responses win more often than a length-controlled baseline.
> Self-enhancement bias: GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer.

The 20–30% flip rate is the headline number. Without swap-and-average, a 3 pp "win" could be entirely a position artifact — and a bootstrap CI *computed on one ordering* will not capture this bias, only the ordering-conditional variance.

---

## The mitigations ch-51 §1 borrows wholesale

Source §Technical Details:

> Position-bias mitigation: evaluate both orders, take a win only if the judge is consistent; otherwise declare tie.
> Verbosity-bias mitigation: length-controlled evaluation pairs where responses differ only in length; compute length-residualized win rate.
> Self-enhancement mitigation: never use the candidate as its own judge; for preference-label generation, use a stronger independent model; for RM training data, pool multiple judges.

Ch-51 §6 memo's "judge-bias check: position-swap parity on IFEval-judge subset = 94% (≥90% threshold)" comes directly from this. The threshold is a concrete acceptance criterion; if position-swap parity < 90%, the go/no-go is halted pending a judge audit or a stronger judge.

---

## The judge-agreement number as a ceiling on claimable effect

Source §Abstract:

> GPT-4 reaches ~80% agreement with human experts — the same rate as humans agree among themselves.

This is an upper bound on judge reliability, not a lower bound on variance. If the judge disagrees with humans 20% of the time on any single item, a win-rate shift smaller than ~20 pp has a non-trivial probability of being a judge-opinion shift rather than a model-capability shift. Ch-51 §7(c) lists this as a named failure mode; the fix is (a) swap-and-average, (b) pool ≥2 judges, (c) report judge-agreement rate alongside win-rate.

---

## Reference-guided grading — the single cheapest fix

Source §Technical Details:

> Reference-guided grading: attach a gold reference solution to the prompt; raises agreement on objective tasks (math, coding), less effect on writing tasks.

For any eval where a gold answer exists (MATH, GSM8K, code with unit tests), reference-guided grading moves σ_judge toward 0 by making the judge a near-verifier. Ch-51's §1 variance table notes "σ_judge ≈ 0 for rule-based verifiers; 1–4 pp for LLM-as-judge pairwise." The guideline: prefer verifiers over LLM judges whenever the task admits one.

---

## Why this matters for paired bootstrap

Paired bootstrap §4 assumes the per-item scores are valid comparable estimates. When the judge has position bias, the *score itself* is miscalibrated per item — pairing does not fix it. The fix is upstream: run both orderings, average; then paired-bootstrap on the averaged scores. Ch-51 §4 sign test is the robust alternative: it uses only the *direction* of the preference, which is ~80% aligned with human judgment even under position bias.

---

## Connections

- **[[bradley-terry-rm]]** — Chatbot Arena Elo is the Bradley-Terry-at-scale version of these pairwise judgments.
- **[[constitutional-ai]], [[rlaif-scaling]]** — AI-labeling pipelines inherit these biases directly.
- **[[reward-hacking-taxonomy]]** — verbosity / self-enhancement are attested proxy failures.
- **ch-51 §1** — judge-variance row.
- **ch-51 §4 sign test** — robust to judge miscalibration when directions still align with humans.
- **ch-52** — safety eval frequently uses LLM judges; inherits the same variance floor.
