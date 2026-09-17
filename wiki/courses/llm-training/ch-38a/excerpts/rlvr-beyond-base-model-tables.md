---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlvr-beyond-base-model.md (card exists but carries no numbers and no Verification section as of 2026-09-15)
source_url: https://arxiv.org/abs/2504.13837
created_at: "2026-09-15"
---

# Excerpt: metric definitions and numbers from "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"

**Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang (Tsinghua, SJTU).
**Version read:** arXiv:2504.13837v5 (24 Nov 2025).
**Status:** the library card [[rlvr-beyond-base-model]] describes the argument but states no numbers and has not been verified to the 2026-09 card standard. Everything below was read at the stated locus in the v5 PDF.

## pass@k estimator (App. A.2, Eq. 2)
With n samples drawn per problem and c_i of them correct for problem x_i,

```
pass@k := E_{x_i ∼ D} [ 1 − C(n − c_i, k) / C(n, k) ]
```

where C(a, b) is the binomial coefficient and k ≤ n. The paper sets n to the largest k shown in each curve: n = 128 for MATH500, Minerva and GSM8K; n = 1024 for AMC23 and AIME24; Olympiad uses 128 for Qwen models and 1024 for LLaMA-3.1-8B.

## Evaluation protocol (§3.1)
Temperature 0.6, top-p 0.95, up to 16,384 generated tokens for both base and RLVR models. Chains of thought are manually checked on a subset because a wrong chain can still produce the right final answer as k grows. App. C.8 raises the RLVR model's temperature until its output entropy matches the base model's; the RLVR model "still underperform[s] the base model" under that match.

## Main results
- "When k is small (e.g., k = 1 …), RL-trained models outperform their base counterparts … as k increases, with steeper curves, base models consistently catch up to and eventually surpass RL-trained models across all benchmarks" (§3.2). Example given: "on the Minerva benchmark with a 32B-sized model, the base model outperforms the RL-trained model by approximately 9% at k = 128".
- Table 2, solvable-problem coverage, base against SimpleRLZoo, AIME24 at k = 1024 and MATH500 at k = 128:

| Base solves | RLVR solves | AIME24 | MATH500 |
|---|---|---|---|
| ✓ | ✓ | 63.3% | 92.4% |
| ✓ | ✗ | 13.3% | 3.6% |
| ✗ | ✓ | 0.0% | 1.0% |
| ✗ | ✗ | 23.3% | 3.0% |

- Sampling-efficiency gap Δ_SE: the difference between an RL model's pass@1 and the base model's pass@k at k = 256. "Across all algorithms (e.g., PPO, GRPO, Reinforce++), Δ_SE shows only minor variation yet remains consistently large" (§1, Fig. 8).
- Perplexity analysis: responses produced by the RLVR model already lie in the base model's output distribution (§4.1, Fig. 6).
- Distillation is treated as the contrasting case: it "can transfer new reasoning patterns from a stronger teacher to the student", and distilled models "often demonstrate an expanded reasoning scope beyond that of the base model" (§1, §5).

## Stated caveats (§3.1)
The authors note that with an astronomically large k even uniform sampling would eventually produce a correct path, and answer that their crossovers occur at k = 128 or 1024, "well within practical resource limits". Random guessing is a known confound for final-answer-only math scoring, which motivates the manual chain-of-thought checks.

## How ch-38a uses it
§8 (pass@k definition, the worked crossover example, the coverage table, Δ_SE), §9 (evaluation protocol and the temperature-matching control), Generalization lens.
