---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2403.13787
created_at: "2026-09-15"
---

# Excerpt: RewardBench — Evaluating Reward Models for Language Modeling

**Authors:** Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi (Allen Institute for AI; University of Washington; Berkman Klein Center)
**Year:** 2024 (arXiv v1 2024-03; v2 2024-06-08)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v2 PDF text.

## Why ch-49 uses it

It is the benchmark that most judge-training papers report, so its structure and its scoring rule are needed to read those numbers; and its known limits are the reason [[rm-bench]] and [[ppe-reward-model-eval]] exist.

## Dataset structure (§4.1)

Prompt–chosen–rejected trios in five sections:

1. **Chat** — pairs from AlpacaEval and MT Bench.
2. **Chat Hard** — MT Bench examples with similar ratings plus adversarial examples from LLMBar, reformatted for reward models.
3. **Safety** — custom versions of XSTest, Do-Not-Answer, and an in-development AI2 refusals set; the chosen response is a refusal and the rejected is harmful text, plus the reverse case of incorrect refusals.
4. **Reasoning** — HumanEvalPack pairs with correct code as chosen and buggy code as rejected; PRM800k reference answers paired with incorrect model generations.
5. **Prior Sets** — test splits of existing preference datasets: Anthropic Helpful (the only multi-turn data), the Anthropic HHH subset of BIG-Bench, a curated subset of Stanford Human Preferences, and OpenAI's Learning to Summarize.

Example subset size for scale: MT Bench Hard has N = 37; Anthropic Helpful has N = 6192 (Table 1).

## Scoring (§4.2)

- A trio is scored correct when the model assigns the chosen response a higher score than the rejected one. Random performance is 50%.
- Inside each section except Reasoning, subsets are combined by per-prompt weighted averaging. Prior Sets uses an unweighted average over subsets because of the large size disparity.
- The final RewardBench score is the weighted average across section scores, with Prior Sets weighted at 0.5 of the other sections, which the authors justify by noise and lack of clearly defined tasks (footnote 4).

## Findings relevant to generality (§5)

- Some subsets are solved near 100% by small reward models; Chat Hard and Reasoning subsets have low ceilings or high variance.
- The Reasoning section spans the widest range, from 35% (below random) to 97% across the models evaluated.
- The adversarial LLMBar subsets in Chat Hard are the hardest for most reward models.
- Performance on Prior Sets is described as noisy and not clearly task-defined, which is why it is down-weighted.

## Limits that ch-49 carries forward

The authors state that more work is needed to relate RewardBench performance to RLHF success. [[ppe-reward-model-eval]] later reports a negative correlation between RewardBench scores of top reward models and post-RLHF human preference outcomes, and [[rm-bench]] reports Pearson r = 0.21 (p = 0.51) for RewardBench against policy performance in its own comparison.

## Connections

[[self-taught-evaluators]], [[direct-judgement-preference]], [[generative-reward-models]] (all report RewardBench as their main metric), [[rm-bench]] and [[ppe-reward-model-eval]] (successor benchmarks addressing style robustness and downstream correlation), [[judge-llm-bias]] (source of MT Bench pairs used in Chat and Chat Hard).
