<!-- excerpt for ch-48; extract of one primary source. Loci are sections/tables of the arXiv PDF.
     Created 2026-09 (generality revision) from https://arxiv.org/abs/2310.17623 (arXiv v2, 2023-11-24).
-->

# Proving Test Set Contamination in Black Box Language Models

- **Authors:** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori B. Hashimoto
  (Stanford University; Columbia University)
- **Year:** 2023 (arXiv v1 2023-10-26; v2 2023-11-24; ICLR 2024)
- **URL:** https://arxiv.org/abs/2310.17623
- **Source type:** paper

## What the chapter uses

**Idea (§1, §3).** Many benchmarks are *exchangeable*: the order of their examples carries no information,
so under the null hypothesis of no contamination every ordering is equally likely. A model trained on the
published file learns the canonical order, so it assigns the canonical ordering higher log-likelihood than a
shuffled one. The test compares those log-likelihoods and returns a p-value with false-positive-rate control.
It needs only log-probabilities: no training data, no weights.

**Sharded test (§3.2, Algorithm 1).** Partition the n examples into r contiguous shards. Within each shard,
compare the canonical-order log-likelihood with the mean over m within-shard permutations, producing a shard
statistic s_i. Take the mean s over shards and run a one-sided t-test for E[s_i] > 0. Unlike a plain
permutation test, whose p-value cannot fall below 1/(m+1), the sharded test reaches very small p-values.

**Power (§4.1, Table 1, Fig. 2).** A 1.4B GPT-2 model was trained on Wikitext with benchmarks injected at
controlled duplication counts. At duplication 1, no detection: BoolQ p = 0.156, HellaSwag p = 0.478,
OpenbookQA p = 0.462 (1,000, 1,000 and 500 examples). At duplication 10: MNLI p = 1.96e-11,
TruthfulQA p = 3.43e-13, Natural Questions numerically zero. At 50 and 100 the p-values are numerically zero.
The reliable detection threshold is around duplication 4.

**Audit of released models (§4.3, Table 2).** 50 shards, 250 permutations per shard, test sets truncated to
the first 5,000 examples. Across LLaMA2-7B, Mistral-7B, Pythia-1.4B, GPT-2 XL and a BioMedLM negative
control on Arc-Easy, BoolQ, GSM8K, LAMBADA, NaturalQA, OpenBookQA and PIQA, almost all p-values are
non-significant; the exceptions are Arc-Easy for Mistral (p = 0.001) and aggregated MMLU (LLaMA2 p = 0.014,
Mistral p = 0.011). The authors caution that with a multiple-comparison correction p = 0.001 sits at the
boundary of significance, and present the results as first steps toward third-party audits, not proof.
Cost: 65 files tested on a 7B model in under 24 hours.

**Limits stated by the authors (§4.1, §4.3).** No power at duplication 1; benchmarks with sequential indices
or duplicate examples are not exchangeable and break the guarantee (at least 14 MMLU subsets were judged
non-exchangeable and filtered); failing to reject is not evidence of cleanliness.

## Verification
- Read on 2026-09-15 against the arXiv v2 PDF, §§1–4 and Tables 1–2, Figs. 2–3.
- Not reported: effect of the detected contamination on benchmark scores (the test says "present", not
  "how much"); results for models above 7B; behaviour on paraphrased or reformatted copies of a benchmark.
