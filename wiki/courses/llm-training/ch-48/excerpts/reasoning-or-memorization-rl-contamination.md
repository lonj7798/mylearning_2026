<!-- excerpt for ch-48; extract of one primary source. Loci are sections/tables of the arXiv PDF.
     Created 2026-09 (generality revision) from https://arxiv.org/abs/2507.10532 (arXiv v3, 2025-12-17).
-->

# Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination

- **Authors:** Mingqi Wu, Zhihao Zhang, Qiaole Dong, Zhiheng Xi, Jun Zhao, Senjie Jin, et al.
  (Fudan University; Shanghai AI Laboratory; UC Davis)
- **Year:** 2025 (arXiv v1 2025-07-14; v3 2025-12-17; AAAI 2026)
- **URL:** https://arxiv.org/abs/2507.10532 ; code github.com/wumingqi/LLM-Math-Evaluation
- **Source type:** paper

## What the chapter uses

**Question (§1).** Published RLVR results report gains on Qwen2.5 from random or incorrect rewards, and the
same procedures give little benefit on Llama3.1-8B. The paper tests whether benchmark contamination in
Qwen2.5's pre-training corpus explains the difference.

**Two black-box probes (§3).**
- *Partial-prompt completion rate*: give the model the first 40%, 60% or 80% of a problem statement and
  measure exact match (EM) and ROUGE-L of the generated remainder against the true remainder.
- *Partial-prompt answer accuracy*: check whether the continuation also contains the correct final answer.

**Results, greedy decoding without a chat template (§4.2, Table 2).** Completion EM at the 60% prefix:

| Model | MATH-500 | AMC | AIME2024 | AIME2025 | LiveMathBench |
|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 54.60 | 42.17 | 20.00 | 0.00 | 0.00 |
| Qwen2.5-7B | 21.20 | 33.73 | 13.33 | 0.00 | 0.00 |
| Llama3.1-8B | 3.80 | 0.00 | 0.00 | 0.00 | 0.00 |

Qwen2.5-Math-7B also answers 53.6% of the 60%-prefix MATH-500 items correctly, against 2.4% for
Llama3.1-8B (§1). With an 80% prefix it reaches 63.8% accuracy on MATH-500, and 41.2% at a 40% prefix
(§4.2). The paper notes that matching only the final numeric answer lets a model be right by accident,
which it uses to explain an anomalous Llama number on AIME2025.

**The date control (§1, §4.2).** On LiveMathBench version 202505, released after Qwen2.5, the completion
rate falls to 0.0% for both families, and AIME2025 behaves like a clean set while AIME2024 does not. The
separation between pre-release and post-release benchmarks is the paper's main evidence.

**RL result on a clean set (§4.3).** The authors generate *RandomCalculation*, arithmetic expressions of
controlled length with random operands, all post-dating the models. Training Qwen2.5-Math-7B with RLVR on
it, only correct rewards produce consistent gains; random and incorrect rewards do not. Zero-shot accuracy
on the clean set falls monotonically with the number of computation steps, which the authors read as
absence of memorisation.

**Authors' recommendation (§6).** Evaluate on uncontaminated benchmarks and repeat RL claims on more than
one model family before attributing a gain to the algorithm.

## Verification
- Read on 2026-09-15 against the arXiv v3 PDF, §§1–5, Tables 1–2, 5, 14 and Figs. 1, 5.
- Not reported: which documents in the Qwen2.5 corpus contain the benchmarks (the corpus is not public);
  EPG-style estimates of how many benchmark points the contamination is worth; results for models other
  than the Qwen2.5/Qwen3 and Llama3.1 families tested.
