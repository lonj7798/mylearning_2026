---
chapter: ch-16
course: llm-training
phase: read
artifact: "DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning"
source_url: https://arxiv.org/abs/2504.11456
version: arXiv v2 (2025-05-22)
verified_on: "2026-09-15"
note: no library card exists for this slug yet; this file is the chapter's verified extract
---

# Excerpt: DeepMath-103K — decontamination, difficulty rating, answer verification

Used by [[read]] §1, §3.1, §7 and the Recipe table. Loci refer to the cached primary text
(`scratchpad/sources/deepmath-103k.txt`, arXiv:2504.11456v2).

## Contamination of the raw pool (§2, Figure 3)

Contamination rate is defined as the percentage of benchmark test samples found within the raw data
pool. Reported values: Omni-MATH 92.6%, AIME 83-24 91.1%, AIME24 90.0%, AMC23 90.0%,
Math Odyssey 88.4%, Gaokao (MC) 78.0%, MATH-500 76.6%, Gaokao (MQA) 70.1%, MATH 67.7%,
JEEBench 49.3%, Minerva Math 35.7%, MMLU-STEM 35.1%, CMATH 33.9%, OlympiadBench 33.6%,
OlympicArena 32.3%, GSM8K 9.8%, GPQA 5.0%. The pool was built from training splits only, which is
the point the authors make: using training splits is not sufficient to avoid overlap.

## Curation pipeline (§3, Figure 7)

Four stages, from a raw pool of 2,869K questions to 95K, plus 8K problems from SimpleRL for the
final 103K.

1. **Source analysis.** Sources whose difficulty distribution is skewed low (MetaMathQA,
   dart-math-hard, OpenMathInstruct-2, NuminaMath-CoT) are avoided as primary sources; Math
   StackExchange subsets of MMIQC and WebInstructSub are chosen for their heavier mid-to-high
   difficulty mass, with NuminaMath-CoT added for topical diversity.
2. **Decontamination.** Embedding similarity search (paraphrase-multilingual-MiniLM-L12-v2) returns
   the top k = 5 most similar test items across 14 benchmarks; Llama-3.3-70B-Instruct then judges
   whether the candidate is an identical question or a paraphrase, and any positive discards the
   candidate.
3. **Difficulty filtering.** GPT-4o rates each problem on the Omni-MATH/AoPS scale, queried six
   times per problem and averaged; only problems rated level 5 or higher are kept.
4. **Answer verification.** GPT-4o discards problem types unsuitable for verification and rewrites
   conversational questions into a standard form with a single numerical or symbolic answer; then
   three distinct solution paths are generated with DeepSeek-R1, a rule-based verifier extracts the
   final answer from each (and from the source solution when available), and the problem is kept
   only if all extracted answers are identical.

Each released problem carries a verifiable final answer and three DeepSeek-R1 solutions (§1).

## What the chapter uses

The pipeline shows a difficulty filter that does not depend on the trained policy: an external
rating model plus a consistency check, applied before any RL run. Compare with the model-aware
filters of [[llama-nemotron]], [[olmo-3]] and [[skywork-or1-rl-data]], which measure the checkpoint
that will be trained.

## Not reported in the part used here

The relative contribution of each filtering stage to downstream scores; contamination rates after
decontamination.
