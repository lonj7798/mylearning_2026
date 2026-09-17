---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: Olmo Team — "Olmo 3" (§3.5.3 Decontamination, §3.5.4 Key findings, App. A.5 decon details); library card wiki/raw-data/llm-training/model-reports/olmo-3.md does not cover these sections
source_url: https://arxiv.org/abs/2512.13961
primary_version: arXiv:2512.13961v2 (2026-04-14; v1 2025-12)
created_at: "2026-09-15"
revised: 2026-09 (generality revision; replaces the 2026-04 excerpt, which described OlmoTrace as a decontamination filter and gave per-stage n-gram thresholds that the report does not contain)
---

# Excerpt: Decontamination of mid-training data in Olmo 3

Verbatim quotations used by ch-14 `read.md`, with loci. Checked against the v2 PDF on 2026-09-15. Source type: official technical report. Correction to the earlier excerpt: OlmoTrace traces model outputs back to training documents; decontamination in Olmo 3 uses a separate package, decon (github.com/allenai/decon).

## Stage placement (§3.5.3)
> "In Olmo 3 midtraining we use a decontamination tool to ensure minimal contamination with evaluation datasets. We focus our decontamination efforts on the midtraining stage (and the long-context extension, which drew from the same data pools) in light of results suggesting that memorization occurs most strongly near the end of training (Magar and Schwartz, 2022; Bordt et al., 2024)."

## Method (§3.5.3)
> "we search for and remove matches of any split of any benchmark dataset that are part of in our evaluation harness, as for some we increased sample size by evaluating on training splits."
> "1. Detection phase For each midtraining document, decon samples n-grams at a regular stride, checking whether the current n-gram matches known n-gram for any benchmark in the evaluation suite. 2. Cluster expansion phase If a match is found, the matching text is expanded on both sides, counting the number of adjacent ngrams that are also contaminated; if the value is above a specified threshold, the document is deemed contaminated removed."
> "We tune the contamination score to balance precision and recall based on numerous qualitative review."
> "the first version fails to decontaminate against SQuAD v2 due to a preprocessing issue; DROP is also incorrectly processed due to its short-question-about-a-passage format. We address these issues by evaluating question, answer, and passage components separately—matching primarily on questions, but using answer/passage matches as supporting information for shorter or edited questions. We also improve precision for multiple-choice evals by matching against full answers rather than just A/B/C/D labels. The decon repository includes configuration files that reproduce both the earlier and final approaches."
- Footnote 22: "We decontaminate against all benchmarks in the OLMES package".

## Scoring details (App. A.5)
- Cluster expansion: "Once a specific document reaches 11 misses, it is removed from the active set."
- IDF-weighted overlap: O = Σ_{x ∈ U_t ∩ U_e} idf(x) / Σ_{y ∈ U_e} idf(y), "where U_t is the set of unique n-grams in the training document segment and U_e is the set of unique n-grams in the evaluation document."
- Length decay: "By default L_start is set by the configuration perfect_match_decay_start: 20 and L_end is set by the configuration perfect_match_decay_end: 50."
- Component weights: "QAP (all components): 0.7 question, 0.2 answer, 0.1 passage"; "QA (no passage): 0.75 question, 0.25 answer"; "QP (no answer): 0.85 question, 0.15 passage"; "Q (question only): 1.0 question".
- Not printed in the report: the n-gram length, the sampling stride, and the final contamination threshold (the report refers to the decon repository configuration).

## Findings (§3.5.4)
> "Not all contamination was subtle—we found many templated contamination instances, in which fields from benchmarks were exactly matched, with templated content inserted between them. Furthermore, many of these were not isolated instances, but complete validation or test splits. For instance, Flan is constructed from templates on benchmark data, and can include validation data that is used for model development decisions since test sets are hidden (e.g., DROP)."
> "We investigate this by comparing our final decontaminated 100B anneal with a matched 100B anneal using the non-decontaminated data versions."
> "Note that we remove contamination of all splits for all benchmarks, such as for DROP removing over 60,000 training examples from sources such as Flan. So performance differences may indicate that decontamination is preventing memorization or also removing in-distribution training examples."
> "other benchmarks do not show inflated performance, despite contamination: we see that DeepSeek LeetCode performance is close to 0 with or without contamination, and SQuAD under the easier MC metric is saturated in either case."
> "despite the fact that our decontamination procedure detected complete leakage of GSM8K in our data, this does not result in better performance with the contaminated data. Instead we see that performance is in fact better with the decontaminated data, a phenomenon that the Marin authors explain occurs due to the contaminated formatting not matching the evaluated format."
> "between Round 3 and Round 5 we also introduce our decontamination process, which means that the gains of Round 5 relative to Round 1 and Round 3 are likely underestimated in this table, given that only Round 5 reflects decontaminated data."
