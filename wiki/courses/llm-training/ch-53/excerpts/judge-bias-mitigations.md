---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/judge-llm-bias.md
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
---

# Excerpt: Judge-LLM bias — three biases, three mitigations

**Source library:** `wiki/raw-data/llm-training/papers/judge-llm-bias.md`
**Anchor paper:** Zheng et al. 2023 — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

---

## Why this source anchors §5 of ch-53

Every pairwise number the harness emits is produced by an LLM judge. If the judge is biased and the harness does not probe for it, every downstream comparison is suspect. Zheng 2023 is the reference paper because it quantified three specific biases with numbers the harness can gate on.

From `judge-llm-bias.md` §Key Contributions:

> **Agreement numbers:** GPT-4 vs human-expert agreement is 85%+ on MT-Bench and ~80% on Chatbot Arena; the same rate two humans agree with each other.
> **Position bias:** A vs B ordering changes the winner in ~20-30% of cases; mitigated by swap-and-average or "two-game" scoring.
> **Verbosity bias:** longer responses win more often than a length-controlled baseline.
> **Self-enhancement bias:** GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer; Claude shows the same toward Claude.

The three mitigations in the §5 probe come directly from this list.

---

## The position-swap probe — the 20-30% gate

From `judge-llm-bias.md` §Fig. 2:

> **Fig. 2** (position bias sweep) — swap A/B, see how often the judge flips; GPT-4 flips ~22%, GPT-3.5 ~40%.

The ch-53 gate uses 20% as the hard ceiling: if flip-rate exceeds this, the judge is effectively a coin-flip on close pairs and the harness refuses to emit pairwise win-rates. GPT-4 sits at the ceiling; weaker judges fail this gate routinely. The mitigation is the "two-game" scoring recommended in the paper: a win counts only if the judge picks the same side under both orderings; otherwise it's a tie.

---

## The verbosity probe — why length controls matter

From `judge-llm-bias.md` §Fig. 4:

> **Fig. 4** (verbosity vs win rate) — clear upward slope.

And §Technical Details:

> **Verbosity-bias mitigation:** length-controlled evaluation pairs where responses differ only in length; compute length-residualized win-rate.

The harness constructs the length-controlled pairs by truncating the longer candidate to the shorter candidate's token count (or padding with whitespace — both tested in the paper, truncation is stricter). If raw win-rate and length-controlled win-rate differ by more than 5 pp, the verbosity flag trips and the memo marks the pairwise number as `verbosity-biased`. RLHF-tuned models frequently emit longer responses; this probe is non-optional on any MT-Bench-style comparison.

---

## The self-enhancement guard — the single hardest rule to violate by accident

From `judge-llm-bias.md` §Fig. 5:

> **Fig. 5** (self-enhancement heatmap) — judge x candidate self-preference matrix.

And §Technical Details:

> **Self-enhancement mitigation:** never use the candidate as its own judge; for preference-label generation, use a stronger independent model; for RM training data, pool multiple judges.

The harness hard-asserts that `judge_family` is not in `candidate_families`. The common accidental failure is subtle: ch-44-RL is trained from a base that shares a tokenizer and mid-training with the judge you pulled off the shelf. Two concrete guards:

1. `judge_family` must resolve to a different model family (not just a different checkpoint).
2. If the candidate pool spans families, rotate the judge across at least two independent families and ensemble — the paper's last-resort recommendation.

---

## Reference-guided grading — the free +10 pp

From `judge-llm-bias.md` §Technical Details:

> **Reference-guided grading:** attach a gold reference solution to the prompt; raises agreement on objective tasks (math, coding), less effect on writing tasks.

And §Key Contributions:

> Providing the judge with a reference solution before evaluation raises agreement by ~10 pp on MT-Bench.

The harness attaches the gold answer to the judge prompt on the `math` and `coding` MT-Bench categories only — for writing/roleplay the benefit vanishes and the reference can anchor the judge to a specific style (measurable via position-swap flip-rate, which tends to drop when a reference is attached even if verdicts are unchanged).

---

## The baseline number — why we trust the gate at all

From `judge-llm-bias.md` §Table 3 summary:

> The same rate two humans agree with each other.

This is the ceiling. An unbiased judge cannot exceed ~80% agreement with a single human because humans disagree ~20% of the time among themselves. The ch-53 memo carries this ceiling forward: any `judge_agreement` number above 85% in a run is flagged as suspicious (usually because the judge memorized the reference) rather than celebrated.

---

## What this source does not tell you

Zheng 2023 does not cover refusal-specific judging. For safety pairwise evals the harness uses a safety-aware judge family (WildGuard-class, or an independent frontier model with explicit refusal-aware instructions) and runs the same three probes on safety-labeled pairs. The bias structure is assumed to transfer; calibration evidence is slim outside MT-Bench.
