---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prm800k.md
source_url: https://arxiv.org/abs/2305.20050
created_at: "2026-04-23"
---

# Excerpt: PRM800K — step-level annotation, active learning, and the 10× cost rule

**Source library:** `wiki/raw-data/llm-training/papers/prm800k.md`
**Year / authors:** 2023 / Lightman, Kosaraju, Sutskever, Cobbe et al. (OpenAI).

---

## Why this source anchors ch-15

PRM800K pushes annotation granularity to its operational extreme. Instead of labeling one pair per prompt, it labels *every intermediate reasoning step* in ~75,000 math solutions. The result is ~800K step-level labels, ~10× more labeling work per problem than outcome supervision. For ch-15 §3 (adjudication) and §4 (preference sampling), PRM800K is the data point that says **annotation cost scales with granularity, not just with count — and active learning is how you survive the cost**.

---

## The labeling schema — three labels per step

```
# prm800k.md, §3
For each step in a GPT-4-generated MATH solution:
  +1  correct
  -1  incorrect
   0  neutral / ambiguous / filler

Training loss: cross-entropy on NON-NEUTRAL steps only.
Step separator: newline or literal "Step k:" token.
Model predicts label at separator position using hidden state.
```

Three design choices worth unpacking against ch-15 §1 (rubric design).

**First**, the three-way schema with an explicit "neutral" bucket. Most binary-rubric projects would force labelers to pick +1 or -1; PRM800K explicitly lets them opt out. The neutral rate in the dataset is nontrivial (~15-20% of steps). This is the annotation-side version of abstention: rather than forcing a noisy label, the rubric encodes "don't know / not a substantive step" as its own class. For ch-15 §2 (κ), the neutral bucket complicates the agreement statistic: you need to measure κ only on non-neutral items, or use Krippendorff α with a distance function that treats neutral as distant from both ±1.

**Second**, the step separator is a tokenizer artifact. The rubric is physically realized as "a score at this token position." If the solution uses an idiosyncratic step delimiter, the annotation pipeline breaks. The paper's requirement that solutions use `"Step k:"` or newlines is the tokenizer-extension constraint of ch-11 reappearing in the annotation layer.

**Third**, the loss is masked to non-neutral steps. This is the training-side equivalent of ch-15 §6's audit trail: the rubric version determines which labels count, and the training loss respects that.

---

## The cost — 10× outcome labeling

> Cost: step-level labeling is ~10× more expensive per example than outcome labeling — the paper explicitly notes active learning is needed to make PRMs practical.

The math: a MATH solution averages ~10 reasoning steps; labeling each step at the same per-label cost as a single outcome label produces 10× the labeling budget. Because each step requires reading the step in context (not just reading the problem + final answer), the per-step cognitive cost is comparable to the per-outcome cost. There is no amortization.

For ch-15 §3's cost model, this multiplies the crowdworker line by 10 for any step-level rubric. At $0.50/outcome, PRM labeling is $5/solution — putting process supervision in the same cost band as expert-tier outcome labeling. The economic case for process supervision collapses unless active learning delivers compensating efficiency.

---

## Active learning — the 2.6× multiplier

> Active learning: prioritizing labeling on solutions where the current PRM is uncertain or disagrees with an ORM gives a ~2.6× data efficiency multiplier.

This is the single largest operational lever in the chapter. The procedure:

1. Train an initial PRM on a small uniform sample.
2. Score a large candidate pool (solutions not yet labeled).
3. Prioritize items where: (a) PRM uncertainty is high — predicted probability of correctness near 0.5, (b) the PRM and an ORM disagree about the solution's overall correctness.
4. Label those items, retrain, repeat.

The 2.6× means **800K active-learned labels match 2.1M uniformly-sampled labels on downstream Best-of-N performance**. At the 10× granularity tax, active learning claws back 2.6× — the net is 10× / 2.6× ≈ 3.8× more expensive than outcome labeling, a tolerable multiplier for the quality gain.

For ch-15 §4 (preference sampling), this is the formal case study. The criterion for "which item to label next" has two terms: **model uncertainty** (label this because the label will carry information) and **inter-model disagreement** (label this because two models disagree and the ground truth will break the tie). Both generalize directly to preference data: UF's aspect-gap sampling is the first term; [[tulu-3]]'s on-policy-RM-ranked sampling is the first term on an on-policy distribution.

---

## The aggregation choice — min, prod, or softmax-avg

Given per-step probabilities of correctness, how do you score a full solution?

```
# prm800k.md, §5
prod        — ∏_t p_correct(step_t)         [used in the paper]
min         — min_t p_correct(step_t)       [used by Math-Shepherd later]
softmax-avg — softmax-weighted average      [smoother; worse Best-of-N]
```

`prod` is equivalent to `exp(sum log p_correct)`; a single wrong step dominates (log-prob goes to -∞). `min` is strictly dominated by the worst step — slightly more robust to noise, slightly less signal. Later work (Math-Shepherd) shifted to `min`; OmegaPRM uses tree-MCTS aggregation.

The ch-15 lesson: **the rubric is not just the per-step schema; it includes the aggregation rule**. A rubric that says "label each step ±1/0" is incomplete without specifying how those step labels combine into a solution-level judgement. Two annotation programs could label the same solutions identically and produce different downstream RMs because they aggregated differently.

---

## The 78.2% headline — what step-level buys you

With PRM Best-of-1860 on MATH: 78.2% vs 72.4% ORM, 69.6% majority. **+5.8pp from switching outcome → process supervision with matched compute.** That is the data point that justifies the 10× annotation cost multiplier for reasoning-heavy domains.

For ch-15 §5 (when humans override), the frame is: **reasoning tasks concentrate error at the step level; outcome labels are a compressed view that discards the error localization**. For code (unit-testable) or IF (constraint-checkable), the verifier provides outcome labels cheaply enough that step-level is unnecessary. For math beyond grade-school and for open reasoning, step-level annotation is the only human-tier signal with enough localization to train a useful verifier.

---

## Connections

- [[excerpts/rlhf-instructgpt]] — outcome-level K-way ranking; the baseline PRM800K displaces for reasoning.
- [[excerpts/ultrafeedback-construction]] — per-aspect outcome rating; PRM800K is the same idea at step-level granularity.
- [[excerpts/tulu-3]] — RLVR uses deterministic verifiers when available; PRM800K is the fallback when verifiers don't exist (open reasoning).
- [[ch-15]] — this excerpt supports §2 (agreement with abstention bucket), §3 (10× cost multiplier), §4 (active learning as the 2.6× lever), §5 (process vs outcome decision boundary).
