---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/evaluation-data-contamination-contam.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2411.03923
primary_version: arXiv:2411.03923v1 (2024-11-06)
created_at: "2026-09-15"
---

# Excerpt: Evaluation data contamination in LLMs: how do we measure it and (when) does it matter?

Verbatim quotations and table values used by ch-14 `read.md`, with loci. Authors: Aaditya K. Singh, Muhammed Yusuf Kocyigit, Andrew Poulton, David Esiobu, Maria Lomeli, Gergely Szilvasy, Dieuwke Hupkes (UCL, Boston University, Cohere, Meta). Checked against the v1 PDF on 2026-09-15. Source type: paper. Llama 3 §5.1.4 states that its contamination analysis follows this paper.

## Setting (§1, §3.2-§3.3)
> "We compare these metrics across 5 parameter settings, on 13 benchmarks, for 7 models of various sizes trained from 2 pre-training corpora."
- Models: Llama 1 7B, 13B, 33B, 65B (Llama 1 corpus) and Pythia 1.4B, 6.9B, 12B (the Pile).

## Metrics (§3.1)
- TOKEN-MATCH (Brown et al., 2020): "the contamination score as the fraction of tokens in the evaluation sample that occurs in a contaminated n-gram"; lowercasing and punctuation removal.
- NGRAM-MATCH (Chowdhery et al., 2023): "the score is calculated as a fraction of the contaminated n-grams in the evaluation sample, rather than contaminated tokens."
- TOKEN-EXTEND (Touvron et al., 2023b): "allows for a skip_budget, which allows for substitution mismatches".
- LONGEST-MATCH (new): "only considers the longest token span (again, with a possible skip_budget). The score is then calculated as the fraction of tokens that is part of this longest match." It "mitigates the issue where templated strings ... confound the contamination scores."
- "we consider n-grams in terms of tokens rather than words (i.e. an 8-gram is 8 tokens long, rather than 8 words long)."

- Table 1: TOKEN-MATCH and NGRAM-MATCH use normalisation (lowercasing, punctuation removal); TOKEN-EXTEND and LONGEST-MATCH do not. "Because the skip budget can be used to account for differces in formatting as well as other differences, TOKEN-EXTEND does not use normalisation." (§3.1)

## Estimated performance gain and threshold selection (§4.1-§4.2)
> "The EPG for a contamination metric and corresponding contamination threshold is model-dependent and is defined as the difference of the model’s performance on the entire benchmark and the subsample of that benchmark marked by the method as uncontaminated or ‘clean’."
> "z-score = EPG / err, where err is the standard error for the given clean subset size (computed as σ/√N, where σ is the standard deviation on the full benchmark)."
> "Empirically, we have found it important to select thresholds separately for each model-benchmark pair, which we found to be crucial for eliminating false positives."
> "we find this method especially necessary for math world problems, where small snippets of context are often meaninglessly contaminated (e.g. “a mosaic with chips of”)."

## Table 3 (optimal metric and threshold per benchmark; n = 8, mincount = 1, skip_budget = 0)
- Selected thresholds range from 0.04 (Big Bench Hard, Pythia 1.4B, NGRAM-MATCH) to 0.40 (Natural Questions, Llama 1 7B, TOKEN-MATCH). Examples: HellaSwag (Llama 1, TOKEN-EXTEND) 0.26/0.26/0.24/0.24 for 7B/13B/33B/65B; GSM8K (Llama 1, LONGEST-MATCH) 0.12/0.12/0.10/0.11; MMLU (Llama 1, LONGEST-MATCH) 0.08/0.09/0.08/0.08.

## Results (§5.2.1-§5.2.2)
> "For 8 of the 13 datasets that we considered, on average more than 50% of the samples are marked contaminated for the Llama 1 pre-training corpus."
> "For the largest Llama model, both HumanEval and Big Bench Hard have an estimated increase in performance of more than 15% (18% and 25%, respectively). Three additional datasets (HellaSwag, MMLU, and PiQA) have an EPG of 10 points or higher. For the smaller Pythia models, these numbers are substantially lower, ranging between 2 and 8 approximately."
> "For the benchmarks COPA, and SIQA, on the other hand ..., up to around 50% contamination was detected, but there was no meaningful impact on performance for any of the models."

## EPG across model scale (§5.2.3)
> "For around half of the model-benchmark pairs, there is a clear trend in which larger models appear to benefit more from contaminated examples."
> "for TriviaQA (Figure 5c), HellaSwag, COPA, and PiQA, larger Llama 1 models benefit less from contamination.We hypothesise that this may be because the larger Llama1 models perform very well on those benchmarks even without contamination"

## Hyperparameters (§6.1, §6.3)
> "We find that, almost across the board, values of n larger than 8 lead to false negatives"
> "only 33.8% of examples have nonzero contamination scores at n = 10 while 67.9% have nonzero contamination scores n = 8." (Figure 6, NGRAM-MATCH, Llama 1 65B, PiQA)
> "setting mincount to values higher than 1 results may exclude examples that do have a real increase in performance as a consequence"

## Recommendations (§7.3)
> "1. Running multiple contamination metrics is preferred, but if you only run one, use LONGEST-MATCH; 2. Use z-scores supplemented by ConTAM to do benchmark specific threshold selection. This is cheap, as it can be done post-hoc; 3. Report % contaminated, EPG and the selected thresholds."
> "the optimal threshold is substantially more stable within than across benchmarks (std 0.02 within benchmark vs 0.1 across benchmark means)"

## Limitations (§8.1-§8.3)
> "ConTAM does not allow to detect contamination if too little or too much of the benchmark is marked as contaminated."
> "metrics that are too lenient may detect pre-training data that is beneficial for a model because they facilicate generalisation."
> "they cannot detect types of contamination that consist of paraphrases of benchmark examples" (§8.2, text-based metrics)
> "All our analyses are thus fundamentally correlational." (§8.3)
