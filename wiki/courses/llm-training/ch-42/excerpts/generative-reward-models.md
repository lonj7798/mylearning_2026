---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/generative-reward-models.md
source_url: https://arxiv.org/abs/2410.12832
created_at: "2026-04-23"
---

# Excerpt: Generative Reward Models (GenRMs)

**Source library:** `wiki/raw-data/llm-training/papers/generative-reward-models.md`
**Papers:** Mahan et al., *Generative Reward Models*, 2024; Ankner et al., *Critique-out-Loud Reward Models*, 2024.

---

## Why this source anchors ch-42

GenRMs give the chapter its most practical reward-specification knob. Instead of a scalar head on an LM, the RM is an instruction-tuned LM that produces a critique plus a verdict. The reward is read off the log-probability the model assigns to the positive verdict token. Two properties matter directly for ch-42's hacking discussion:

- The **rubric** is the reward specification. Editing the rubric changes the reward — no retraining, no scalar-head fine-tune.
- The RM is **steerable via its own context**. Adding "longer is not better; penalize sycophancy" to the rubric measurably reduces those two hacks on held-out prompts.

Scalar RMs can do neither.

Raw-data header:

> **Core Insight:** Instead of a scalar head on an LM, use the LM itself to generate a critique plus a verdict; extract the reward from the log-probabilities of the verdict tokens. Generative RMs let chain-of-thought reasoning flow into the reward and produce calibrated uncertainties.

## The scoring rule

For a pair `(x, y_A, y_B)` with a rubric `R`:

```
critique ~ P_RM(· | x, y_A, y_B, R)
reward   = log P_RM("A is better" | x, y_A, y_B, R, critique)
         - log P_RM("B is better" | x, y_A, y_B, R, critique)
```

This is equivalent to a BT log-odds at the verdict position. For pointwise scoring, extend to a 1–10 verdict and take the log-prob-weighted expectation as the scalar reward.

## Critique-then-verdict

Sampling a critique first and scoring the verdict conditional on the critique adds 3–10 pp on RewardBench over no-CoT. The critique acts as a committed "reasoning trace" the verdict has to cohere with, which makes it harder for the model to rubber-stamp the more-confident-sounding response.

## Training

Fine-tune the instruction-tuned LM with next-token supervision on (prompt, rubric, critique, verdict) triples. No dedicated scalar head. This keeps the RM in the same model family as the policy, which simplifies serving and lets the RM benefit directly from base-model scaling.

Critiques can be human-written or bootstrapped from GPT-4 (or from an earlier GenRM — this is the Con-J / Self-Taught Evaluator iteration in [[direct-judgement-preference]]).

## Calibration

The paper's Fig. 4 shows GenRMs are well-calibrated where BT RMs are overconfident. The verdict log-probability reliably tracks ground-truth agreement rate. This feeds into ensembling: GenRM ensembles produce calibrated lower-confidence-bound (LCB) rewards that combine well with [[reward-ensembling]] defenses.

## Steerability — the policy knob scalar RMs lack

The rubric text is the specification. Rubric wording changes the reward. Two concrete knobs relevant to ch-42:

- **Verbosity clause.** Adding "longer is not inherently better; penalize unnecessary padding" to the rubric reduces the length–reward correlation on held-out rollouts without retraining.
- **Sycophancy clause.** Adding "prefer responses that correct factual errors even when the user has asserted them" reduces the sycophancy flip rate on TriviaQA probes.

This is the structural reason ch-42 §2's mitigation column repeatedly cites "rubric clause naming the hack": the generative-RM architecture lets you name hacks in prose and have the reward update immediately.

## What GenRMs do not fix

- **Self-enhancement** still applies — a GenRM from model family F still prefers family-F outputs at above-human rate.
- **Position bias** still applies and must be corrected with two-game scoring.
- **Adversarial rubric gaming** — a capable policy can learn to satisfy the literal rubric wording while violating intent. The rubric is a reward specification in natural language, and natural-language specifications are themselves hackable.
- **Compute cost** — GenRMs generate critique tokens, which is slower than a scalar-head forward pass. Partially amortized by reusing the base-LM inference stack.

## Pointwise vs pairwise

GenRMs come in two forms:

- **Pairwise** — the scoring rule above; directly trains a BT policy.
- **Pointwise** — score one response on a rubric with a 1–10 verdict; the log-prob-weighted expectation over the 10 verdict tokens gives a scalar reward. Useful for best-of-N reranking and for RMs that need absolute (not relative) scores.

## Connection to the broader defense stack

GenRMs sit at ch-42 §8's defense #5 (generative RMs with explicit rubrics). They are below verifiable rewards, KL budget, potential-based shaping, and ensembling in the ordering — because GenRMs still have a learned reward and are still hackable — but above anomaly detection, because they give a legible, text-editable reward specification that auditors can inspect.

## Takeaways for the chapter

1. GenRMs give a text-editable reward specification; editing the rubric edits the reward.
2. Critique-then-verdict adds 3–10 pp RewardBench over no-CoT.
3. Calibration is measurably better than scalar BT RMs, which helps ensembling.
4. Verbosity and sycophancy can be mitigated by naming them in the rubric — the "tell the RM what to penalize" knob.
5. Self-enhancement and position bias remain; GenRMs are not judge-bias-free.
