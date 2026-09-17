---
chapter: ch-16
course: llm-training
phase: read
artifact: "Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination"
source_url: https://arxiv.org/abs/2507.10532
version: arXiv v3 (2025-12-17); AAAI 2026 copyright line
verified_on: "2026-09-15"
note: no library card exists for this slug yet; this file is the chapter's verified extract
---

# Excerpt: contamination as the explanation for spurious-reward RL gains

Used by [[read]] §7 and the "Why this chapter matters" section. Loci refer to the cached primary
text (`scratchpad/sources/reasoning-or-memorization-rl-contamination.txt`, arXiv:2507.10532v3).

## The question (§1)

Prior work reports that random or incorrect rewards improve Qwen2.5 on MATH-500, AMC and AIME while
giving little or no benefit to Llama3.1-8B. The paper tests two hypotheses for the asymmetry: data
contamination of Qwen's pretraining corpus, or stronger mathematical capacity.

## Memorization metrics (§4.2)

Questions are truncated at 40%, 60%, and 80% of their length and fed as prompts.
**Partial-prompt completion rate** is ROUGE-L / exact match between the generated continuation and
the ground-truth continuation; **partial-prompt answer accuracy** is whether the continuation
contains the correct answer. Decoding: greedy, no chat template (Table 1).

Table 2, exact-match completion at a 60% prefix:

| Model | MATH-500 | AMC | AIME2024 | AIME2025 | Minerva Math | LiveMathBench |
|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 54.60 | 42.17 | 20.00 | 0.00 | 0.37 | 0.00 |
| Qwen2.5-7B | 21.20 | 33.73 | 13.33 | 0.00 | 0.37 | 0.00 |
| Llama3.1-8B | 3.80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

At a 40% prefix Qwen2.5-Math-7B still reconstructs 39.20 EM of MATH-500. Answer accuracy with
partial prompts (§4.2, Figure 5): 63.8% on MATH-500 at an 80% prefix and 41.2% at a 40% prefix for
Qwen2.5-Math-7B. The report notes that matching only the final numeric answer allows accidental
hits, which explains a non-zero Llama score on AIME2025 from one problem solved with faulty
reasoning.

## The clean benchmark (§1, §4.3)

RandomCalculation is generated automatically: arithmetic expressions of arbitrary length with
random operands and operators, guaranteed to post-date the models' release. Zero-shot accuracy
declines monotonically with the number of computation steps, which the authors read as absence of
memorization. Under RLVR on Qwen2.5-Math-7B, correct rewards give consistent gains that pass the
base model's performance ceiling, random rewards make training unstable with no reliable
improvement, and inverted ("mv-incorrect") rewards degrade mathematical reasoning.

## Recommendation the paper states (Abstract, §1)

Evaluate on uncontaminated benchmarks and, where possible, test more than one model series before
drawing conclusions about RL methods.

## Limits

The contamination evidence is behavioural (reconstruction from a prefix), not an inspection of the
pretraining corpus, which is not public. The RL experiments are on 7B Qwen checkpoints and
Llama3.1-8B-Instruct.
