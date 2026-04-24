---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/judge-llm-bias.md
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
---

# Excerpt: Zheng 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Source library:** `wiki/raw-data/llm-training/papers/judge-llm-bias.md`
**Paper:** Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 Datasets and Benchmarks.

---

## Why this source anchors ch-42

LLM-as-a-judge is load-bearing for every modern alignment stack — RLAIF labels, reward-model training data, benchmark Elos. This paper is the definitive measurement of what those judges get wrong. Ch-42 §3 is almost entirely a restatement and operationalization of this paper's bias catalogue.

Raw-data header:

> **Core Insight:** Strong LLM judges (GPT-4) agree with humans ~80% of the time — matching inter-human agreement — but they come with specific, measurable biases: position (order), verbosity (length), self-enhancement (prefer their own outputs), and limited reasoning on math/coding pairs.

## The agreement result

GPT-4 vs human-expert agreement is 85%+ on MT-Bench and ~80% on Chatbot Arena — the same rate two humans agree with each other. This is the "parity" headline. It is *necessary* for LLM judges to be useful but *not sufficient* for them to be safe: a judge that agrees with humans 80% of the time can still systematically bias the remaining 20% in a reward-hackable direction.

## The three named biases

### Position bias

A vs B ordering changes the winner in 20–30% of cases. Specifically:

- GPT-4 flips ~22%.
- GPT-3.5 flips ~40%.

The mitigation is the "two-game" score: evaluate both orders, count a win only if the judge is consistent, otherwise tie. This halves throughput but is the only correction without a theoretical hole. Ch-42 §3's diagnostic table reports the same thresholds and lists position-swap consistency as a pre-deployment check (§7).

### Verbosity bias

Longer responses win more often than a length-controlled baseline. The paper constructs length-controlled pairs where responses differ only in length and shows the judge still prefers longer. The correction is length-residualized win rate.

This is the structural reason length bias appears in ch-42 §2 as hack #1 — the verbosity bias in the judge propagates into the RM trained on judge labels, which propagates into the policy trained under that RM. Every layer inherits it.

### Self-enhancement

GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer; Claude shows the same toward Claude. Measured via the judge-vs-human win-rate delta on the same pairs. The mitigation is explicit: never use the candidate model as its own judge, and for RM training data, pool multiple judges from distinct model families.

Ch-42 §7's pre-deployment checklist formalizes this as "judge-rotation audit": rotate the judge across model families; a win-rate swing > 8 pp is the self-enhancement flag.

## Limited reasoning on objective tasks

On math and coding pairs, LLM judges can confirm a wrong answer if it is presented confidently. This links directly to U-sophistry — a rhetoric-over-truth failure mode. The paper's fix: reference-guided grading.

## Reference-guided grading

Attach a gold reference solution to the judge prompt. Raises agreement by ~10 pp on MT-Bench objective categories (math, coding). Less effect on writing tasks, where no single gold reference exists.

This is the structural argument for RLVR on verifiable prompts: when a reference is available and checkable, the judge is just a thin wrapper around a verifier, and you might as well cut out the judge.

## CoT judging

Asking the judge to reason step-by-step before verdict improves agreement but does **not** eliminate the biases — it only makes them more legible. A CoT judge is still subject to position, verbosity, and self-enhancement; you can just read the rationale and see where it is going wrong.

## Dataset release

- 3K expert MT-Bench votes.
- ~30K Chatbot Arena conversations.

Widely reused by later preference-dataset work (UltraFeedback, Nectar, Skywork-Reward-Preference). Also a known leakage risk: benchmarks trained on this data measure "how much the model agrees with the MT-Bench rater distribution" rather than a ground-truth preference.

## Tie handling

When judges declare ties, the Elo update is a small-delta adjustment. This matters for RM calibration: a dataset with many ties (indistinguishable pairs) under-trains the BT model on the informative region. The paper recommends explicit tie handling rather than forced A/B choice.

## Takeaways for the chapter

1. LLM judges reach human-level agreement *on average* but carry specific systematic biases. Every bias propagates down the RLHF stack.
2. Position bias has the cleanest fix (two-game scoring); verbosity and self-enhancement require structural changes (length-controlled eval, judge rotation).
3. Reference-guided grading is the biggest single-intervention gain (+10 pp); this is why RLVR is preferred over LLM-judged RL when a verifier exists.
4. CoT rationales are legible but do not remove biases. Do not mistake legibility for correctness.
5. Every number in ch-42's judge-bias table (§3) and pre-deployment checklist (§7) comes from this paper's figures.
