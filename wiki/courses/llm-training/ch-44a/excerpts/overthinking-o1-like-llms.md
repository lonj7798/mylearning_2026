---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2412.21187v2 ("Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs"), §1, §2.2, §2.3, §3, Tables 3-4 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2412.21187
created_at: "2026-09-15"
---

# Excerpt: On the Overthinking of o1-Like LLMs

- **Authors:** Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, et al. (Tencent AI Lab; Shanghai Jiao Tong University)
- **Year:** 2024 (arXiv v1 2024-12; this extract read from v2, 2025-02-01)
- **Source type:** paper
- **Used in:** ch-44a §4, Negative samples, Common mistakes.

## Observation (§1, §2.2)
- On the question "what is the answer of 2 plus 3?", "o1-like models consumed 1,953% more tokens than conventional models to
  reach the same answer" (Figure 1a); Figure 2 shows QwQ-32B-Preview producing 13 solutions for it.
- "In more than 92% of cases, the initial round of solutions produces the correct answer", and the first round "generally
  comprises less than 60% of the total tokens generated" (Figure 5).

## Outcome efficiency metric (§2.2, Eq. 1)
`ξ_O = (1/N) Σ_i (T̂_i / T_i) · σ_i`, where `N` is the number of instances, `T_i` the total tokens of instance i, `σ_i = 1`
if at least one solution in the response is correct (else 0), and `T̂_i` is the number of tokens up to the first correct
answer when `σ_i = 1` (and `T_i` otherwise). Worked example given in the paper: the first solution is correct at `T̂ = 39`
tokens out of 901 generated, so `ξ_O = 39/901 = 4.3%`.

## Self-training data (§3.1, §3.2)
- Prompts: PRM12K. Ten samples per question at temperature 1.0; samples without a correct answer are discarded.
- Positive-example variants and their statistics (Table 3, mean solutions #S / mean tokens / outcome efficiency / process
  efficiency): Shortest Response 2.5 / 1051.3 / 69.8% / 80.3%; First-Correct Solutions (FCS) 1.1 / 681.0 / 99.5% / 99.1%;
  FCS+Reflection 1.9 / 878.7 / 78.4% / 82.4%; Greedily Diverse Solutions 1.6 / 856.8 / 86.8% / 94.2%.
- Negative example: the longest sampled response. "in our preliminary experiments, we found it [the greedy-search response]
  less effective than using the longest sampled response as the negative example. One possible reason is that the longest
  sampled response provides a clearer contrastive signal."

## Results (§3.3, Table 4; policy QwQ-32B-Preview)
| Setting | Accuracy | #Solutions | #Tokens | Outcome eff. | Process eff. |
|---|---|---|---|---|---|
| MATH500 baseline | 93.0 | 3.2 | 2407.9 | 52.3% | 71.2% |
| + SFT Shortest Response | 93.2 | 3.0 | 2359.5 | 60.4% | 75.6% |
| + DPO Shortest Response | 94.0 | 2.7 | 1929.5 | 65.8% | 79.1% |
| + RPO Shortest Response | 91.6 | 2.7 | 2015.7 | 64.8% | 79.2% |
| + SimPO Shortest Response | 92.4 | 2.5 | 1871.8 | 67.6% | 80.9% |
| + SimPO First-Correct Solution | 91.0 | 1.4 | 1016.0 | 88.7% | 98.1% |
| + SimPO FCS+Reflection | 92.8 | 1.9 | 1330.7 | 80.0% | 89.5% |
| + SimPO Greedily Diverse Solutions | 91.8 | 1.7 | 1286.1 | 84.3% | 93.6% |

Other test sets with SimPO FCS+Reflection: ASDIV 96.9 → 96.8 accuracy, 741.8 → 381.6 tokens; GSM8K 94.8 → 96.0, 772.8 →
416.6; GPQA 59.6 → 59.1, 3228.4 → 2085.7; AIME24 46.7 → 43.3, 9480.9 → 5154.5.

"SFT only slightly reduces the number of solution rounds and tokens ... underperforming the preference optimization
methods. Among these methods, SimPO achieves the best results, reducing the number of generated tokens by 22.3% on
MATH500." FCS alone "decreases performance on the difficult MATH500 test set, which may require more rounds of reflection".

## Not reported
Preference-optimization hyperparameters in the main text, per-benchmark variance, and any evaluation of instruction
following or other general capabilities.
