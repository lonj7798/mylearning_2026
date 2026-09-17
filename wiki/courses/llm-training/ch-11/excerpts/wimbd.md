---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (no library card planned on 2026-09-15)
source_url: https://arxiv.org/abs/2310.20707
created_at: "2026-09-15"
---

# Excerpt: What's In My Big Data? (WIMBD)

**Authors:** Yanai Elazar, Akshita Bhagia, Ian Magnusson, Abhilasha Ravichander, Dustin Schwenk, Alane Suhr, et al. (AI2, University of Washington, UC Berkeley, UC Irvine).
**Version read:** arXiv:2310.20707v2 (5 Mar 2024); v1 Oct 2023; ICLR 2024.
**Status:** no library card existed; values read in the v2 PDF text at the stated locus.

## Tool (§1, §3)
Two building blocks: Count (map-reduce over the full corpus) and Search (an Elasticsearch index returning matching documents and counts). Sixteen analyses on ten corpora, including C4, mC4-en, OSCAR, The Pile, RedPajama, S2ORC, peS2o, LAION-2B-en, The Stack, OpenWebText. The abstract states the tool analyzes more than 35 terabytes on a standard compute node.

## Duplicates (Abstract)
"About 50% of the documents in RedPajama and LAION-2B-en are duplicates."

## Benchmark contamination (§4.4.1, Fig. 5)
82 PromptSource datasets with a test split and at least two input fields; a test instance counts as contaminated when all input fields appear in one document ("an upper bound of exact-match dataset contamination"). 67 of 82 datasets were not found in any of the four corpora checked (The Pile, C4, RedPajama, OSCAR). RedPajama was the most contaminated, with COPA fully contained.

## PII (§4.4.2, Table 5, App. B.3.2)
Method: three regular expressions (email addresses, phone numbers, IP addresses) with post-processing rules that remove common false positives such as ISBN numbers (Table 17). Precision was estimated from 100 manually inspected matches per type and corpus; recall was not estimated ("The nature of this retrieval task makes it challenging to estimate the recall").

| Corpus | Emails (count, precision %) | Phone numbers | IP addresses |
|---|---|---|---|
| OpenWebText | 364K, 99 | 533K, 87 | 70K, 54 |
| C4 | 7.6M, 99 | 19.7M, 92 | 796K, 56 |
| mC4-en | 201M, 92 | 4B, 66 | 97.8M, 44 |
| The Pile | 19.8M, 43 | 38M, 65 | 4M, 48 |
| RedPajama | 35.2M, 100 | 70.2M, 94 | 1.1M, 30 |
| S2ORC | 630K, 100 | 1.4M, 100 | 0, 0 |
| peS2o | 418K, 97 | 227K, 31 | 0, 0 |
| LAION-2B-en | 636K, 94 | 1M, 7 | 0, 0 |
| The Stack | 4.3M, 53 | 45.4M, 9 | 4.4M, 55 |
Counts are extrapolated frequencies. Email precision exceeded 80% on 8 of 10 corpora and phone precision on 5 of 10. mC4-en contains the most PII also after normalizing by tokens (Table 19).

## How ch-11 uses it
§6 (PII prevalence and regex precision), §6 provenance tools, Common mistakes.
