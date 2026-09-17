---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (no library card planned on 2026-09-15)
source_url: https://arxiv.org/abs/2401.17377
created_at: "2026-09-15"
---

# Excerpt: Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

**Authors:** Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, Hannaneh Hajishirzi (University of Washington, AI2).
**Version read:** arXiv:2401.17377v4 (7 Apr 2025); v1 Jan 2024; COLM 2024.
**Status:** no library card existed; values read in the v4 PDF text at the stated locus.

## ∞-gram estimate (§2)
P_∞(w_i | w_1:i−1) = cnt(w_{i−(n−1):i−1} w_i | D) / cnt(w_{i−(n−1):i−1} | D), with n = max{n′ ∈ [1, i] | cnt(w_{i−(n′−1):i−1} | D) > 0}. The effective n is one plus the length of the longest suffix of the prompt that occurs in the corpus D. The estimate is sparse when one next token has probability 1.

## Index (§3)
- Suffix array over the tokenized corpus: counting a query of length L in a corpus of length N costs O(L + log N).
- Token array: 2 bytes per token, "assuming that |V| < 2^16 = 65536"; documents separated by `\xff\xff`.
- Pointers: 5 bytes each for shards of 2B-500B tokens; index size 7N bytes (3.5× the raw token array, §1).
- Building the RedPajama index (1.4T tokens) took about 48 hours on one node with 128 CPUs and 1 TiB RAM, using 10 TB of disk (§1, §3).
- Indexes built: Dolma (3T tokens), RedPajama (1.4T), Pile (380B), C4 (200B); together 5T tokens (§3).
- Latency on RedPajama: n-gram count under 20 ms; n-gram probability 40 ms; ∞-gram probability 200 ms (§3, §A.5).

## Analyses (§1, §4)
- ∞-gram next-token accuracy on human-written text: 47%; conventional n-gram with small n: 29% (§1).
- Interpolating ∞-gram with neural LMs reduces perplexity by up to 73%, including a 70B neural LM (§1, §5).
- Reference data were decontaminated against the evaluation sets before these analyses (§4, App. B).
- Indexes are tokenizer-specific: "we built three versions of the infini-gram index on Pile-train and RedPajama, one for each tokenizer" (GPT-2/GPT-Neo/GPT-J, Llama-2, SILO), and perplexities across tokenizers are not comparable (§5.1).
- App. E (extended discussion, proposed uses without experiments): data curation, where the SEARCHDOC query retrieves all documents containing an n-gram term or a CNF expression of terms (for example PII) so they can be removed, with additive and subtractive indexes for iterative removal; and detection of test-set contamination, memorization, and plagiarism by n-gram lookup.

## How ch-11 uses it
§6 (provenance tools and memorization checks), worked example (index size), Generalization lens.
