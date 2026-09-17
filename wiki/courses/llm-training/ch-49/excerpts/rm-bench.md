---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2410.16184
created_at: "2026-09-15"
---

# Excerpt: RM-Bench — Benchmarking Reward Models of Language Models with Subtlety and Style

**Authors:** Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, Juanzi Li (Fudan University; Tsinghua University; HKUST)
**Year:** 2024 (arXiv v1 2024-10-21)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v1 PDF text.

## Why ch-49 uses it

It measures a reward model where substance and style disagree, which is the case a policy will find during optimization, and it reports how that measurement correlates with policy quality.

## Construction (§3)

- **Subtle content differences:** the rejected response differs from the chosen one by a small factual change rather than by being written by a weaker model.
- **Style-controlled variants:** following the Chatbot Arena style-control design, two features (length and markdown) produce three response types for both the chosen and the rejected response — `y∅` short and plain, `y^L` detailed and plain, `y^{L,M}` detailed with markdown. gpt-4o is used to strip markdown and to summarize into the concise form without altering content (§3.3).
- Four domains: Chat, Safety, Code, Math.

## Style–Substance Evaluation Matrix (§3.3)

Comparing each chosen style against each rejected style gives a 3×3 matrix. Three metrics:

- **Easy accuracy** — mean of the lower triangle; the chosen response also has the more favourable style.
- **Normal accuracy** — mean of the diagonal; both responses share a style.
- **Hard accuracy** — mean of the upper triangle; the rejected response has the more favourable style.

Published matrix for `sfairXC/FsfairX-LLaMA3-RM-v0.1` in the chat domain (Fig. 2), rows = chosen style, columns = rejected style:

```
              y_r       y_r^L     y_r^{L,M}
y_c         83.61%     3.83%      2.19%
y_c^L       99.45%    66.12%     49.73%
y_c^{L,M}  100.00%    80.33%     66.67%
```

Derived from these entries: easy 93.26%, normal 72.13%, hard 18.58%.

## Results (§4.1, Table 3)

Nearly 40 reward models evaluated, from 2B to 340B, trained as classifiers or with DPO. Top rows:

| Model | Chat | Math | Code | Safety | Easy | Normal | Hard | Avg |
|---|---|---|---|---|---|---|---|---|
| Skywork-Reward-Llama-3.1-8B | 69.5 | 60.6 | 54.5 | 95.7 | 89.0 | 74.7 | 46.6 | 70.1 |
| URM-LLaMa-3.1-8B | 71.2 | 61.8 | 54.1 | 93.1 | 84.0 | 73.2 | 53.0 | 70.0 |
| Nemotron-340B-Reward | 71.2 | 59.8 | 59.4 | 87.5 | 81.0 | 71.4 | 56.1 | 69.5 |

The abstract's headline is that state-of-the-art models average 46.6% under style-bias interference, below the 50% random baseline.

## Correlation with policy performance (§5)

Four Tülu-v2.5 reward models trained on different preference datasets (HH-RLHF, StackExchange, Chatbot Arena 2023, Nectar), each sampled to 60k examples, with policies trained by PPO using the same data and hyperparameters.

- **Style-controlled correlation (§5.1, Fig. 4):** higher chat-domain hard accuracy is associated with a higher style-control score for the policy, where the style-control score is the relative drop from uncontrolled to style-controlled Arena-Hard-Auto performance.
- **Downstream correlation (§5.2, Fig. 5):** reward-model score against normalized policy performance on GSM8k, Big Bench Hard, HumanEval+, MBPP+, ToxiGen, and XSTest gives Pearson r = 0.55 (p = 0.07); the same comparison for RewardBench gives r = 0.21 (p = 0.51). Hard accuracy is used for math and safety, normal accuracy for code, because code responses are style-consistent in markdown.

## Limits stated by the paper

The authors describe the downstream correlation as limited (Appendix: "Limited Correlation with Policy Models") and the correlation is computed over four reward models.

## Connections

[[rewardbench]] (the benchmark it is positioned against), [[lmsys-style-control]] (source of the length-and-markdown style definition), [[ppe-reward-model-eval]] (the other successor benchmark, which measures downstream effect end to end), [[arena-hard-benchbuilder]] (the style-controlled policy evaluation used in §5.1).
