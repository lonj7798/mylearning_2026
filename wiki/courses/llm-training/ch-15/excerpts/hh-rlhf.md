---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/hh-rlhf.md
source_url: https://arxiv.org/abs/2204.05862
created_at: "2026-04-23"
---

# Excerpt: HH-RLHF — two-axis annotation and the tension-curve lesson

**Source library:** `wiki/raw-data/llm-training/papers/hh-rlhf.md`
**Year / authors:** 2022 / Bai, Jones, Askell, Kaplan et al. (Anthropic).

---

## Why this source anchors ch-15

HH-RLHF is the first public preference dataset at scale (161K dialogues, MIT license) and the first to demonstrate — *empirically, with a learned Pareto curve* — that a single-axis rubric is inadequate for dialogue. Anthropic collected two campaigns: a helpfulness campaign (crowdworkers chat, pick the more helpful of two responses) and a harmlessness red-team campaign (crowdworkers try to elicit harmful output, pick the *less* harmful response). A model trained on one degrades on the other. The implication for ch-15 §1 (rubric design) is that **axes of a rubric can be partially anti-correlated, and a single scalar preference erases the tradeoff**.

---

## The two-campaign protocol

```
# hh-rlhf.md, reconstructed from §2 and §3
Campaign A — Helpfulness
  Worker initiates a dialogue with the 52B Anthropic base assistant.
  At each assistant turn, TWO candidate responses are sampled.
  Worker picks the more helpful; dialogue continues with the chosen one.
  Output: a single dialogue trajectory + per-turn chosen/rejected pair.

Campaign B — Harmlessness (red team)
  Worker is instructed to elicit harmful, unethical, or biased output.
  At each assistant turn, TWO candidates are sampled.
  Worker picks the LESS harmful (dispreferred behavior-wise, but safer).
  Output: trajectory + per-turn safer-vs-less-safe pair.

Both → binarized as (chosen, rejected) under HH's release format.
```

Two non-obvious annotation-design choices.

**First**, the preference labeling is *inline with the trajectory*. The worker's own next turn is conditioned on their previous preference, so the collected distribution is on-policy to the sequence of their choices. This is the 2022 ancestor of ch-15 §4's on-policy preference discussion: [[tulu-3]] sampling from the *current* DPO policy for the next round's preference pairs is an industrial-scale version of this inline pattern.

**Second**, the harmlessness campaign inverts the helpfulness instruction: the crowdworker is told to play an adversary, not a user. This is a rubric-design choice that cannot be captured by a single-axis preference. A generic "pick the better response" rubric would confuse a helpful response to a harmful query (helpful axis wins, harmless axis loses) with a helpful response to a benign query. The two-campaign split is what makes the tradeoff legible.

---

## The tension curve — what the rubric taught the model

The paper's most-reproduced figure is the helpful-vs-harmless Pareto plot. Models trained on helpfulness-only score high on helpfulness evals but produce harmful completions at a dangerous rate; models trained on harmlessness-only refuse benign requests ("tell me a recipe for chocolate cake" → "I cannot help with that") and score low on helpfulness. The *joint* training produces a Pareto-improved front: a model that refuses harmful requests (high harmlessness) while being helpful on benign ones (high helpfulness).

```
# The conceptual structure (text description of Fig. 2)
  helpfulness
      |
      |   ● helpfulness-only  (bad harmlessness)
      |
      |       ● JOINT (HH)
      |
      |           ● harmlessness-only (bad helpfulness)
      |_________________________________________ harmlessness
```

The ch-15 lesson: **if your rubric's axes are partially anti-correlated, single-axis training produces a corner solution; you need the product distribution and joint optimization to recover the Pareto front**. The 2024 generalization is [[ultrafeedback-construction]]'s 4-aspect rubric, where each aspect is labeled separately and the downstream training can choose per-aspect or aggregated preferences depending on target.

---

## The agreement numbers — the noise floor

From the paper's Appendix D (reconstructed):

> Human-human agreement on held-out helpful comparisons: 70–75%.
> Human-human agreement on held-out harmless comparisons: lower, ~65–70%, because harm judgment is more contentious.

That's κ ≈ 0.40–0.50 on helpful, 0.30–0.40 on harmless. The latter figure is close to the "fair" threshold of Landis-Koch — right at the edge of what should ship. The paper's defense is scale: 161K dialogues at 0.40 κ train a better RM than 10K at 0.80 κ, because the noise averages out. The 2024 reply is that it depends on the downstream use: for DPO, where every pair enters the loss, label noise bounds the achievable margin; for BoN re-ranking with a strong judge, you can tolerate more noise because the re-ranker votes multiple times.

The 70–75% agreement floor is what [[judge-llm-bias]]'s ~80% judge-human agreement must be compared against. GPT-4 agrees with humans ~80% of the time; humans agree with each other ~72% of the time; the judge is therefore operating *above* the noise floor, which is why judge-driven preference data has not collapsed in the way early critics feared.

---

## Length bias — baked into HH

A known confound documented in the paper:

> Length bias — longer responses often preferred; a known confound in downstream RMs trained on HH.

The HH preference data has response-length-correlated labels, so any RM trained on it inherits the bias. Every 2024 successor project has a length-control step: [[tulu-3]]'s **length-normalized DPO** (β=5.0) is the explicit mitigation. [[ultrafeedback-construction]]'s 4-aspect rating attempts to separate length from quality by asking about truthfulness and helpfulness as distinct aspects, but GPT-4's own length bias leaks in.

For ch-15 §1 (rubric design), the moral is: **the rubric must explicitly call out length — "prefer the more concise of two equally-correct responses" or "ignore length when rating helpfulness" — otherwise length leaks in as an uncontrolled confound**.

---

## Connections

- [[excerpts/rlhf-instructgpt]] — the one-axis ancestor; HH-RLHF is the two-axis expansion.
- [[excerpts/prosocial-dialog]] — the RoT-anchored alternative approach to safety rubrics; complements HH's red-team approach.
- [[excerpts/ultrafeedback-construction]] — multi-aspect rubric generalization; 4 axes instead of 2.
- [[excerpts/judge-llm-bias]] — the 80% judge-human number that uses HH's 72% human-human as its comparison baseline.
- [[ch-15]] — this excerpt supports §1 (tension-curve rubrics), §2 (the 70–75% noise floor as calibration target), and §5 (when humans override for safety-critical).
