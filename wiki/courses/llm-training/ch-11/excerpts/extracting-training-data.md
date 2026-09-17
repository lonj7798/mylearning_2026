---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (no library card planned on 2026-09-15)
source_url: https://arxiv.org/abs/2012.07805
created_at: "2026-09-15"
---

# Excerpt: Extracting Training Data from Large Language Models

**Authors:** Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, et al. (Google, Stanford, UC Berkeley, Northeastern, OpenAI, Harvard, Apple).
**Version read:** arXiv:2012.07805v2 (15 Jun 2021); v1 Dec 2020; USENIX Security 2021.
**Status:** no library card existed; values read in the v2 PDF text at the stated locus.

## Attack (§4-§6.1)
Black-box query access to GPT-2 XL (1.5B, trained on WebText). Generate samples with three strategies (top-n sampling, sampling with a decaying temperature, and conditioning on Internet text), rank them with six metrics (GPT-2 perplexity; ratios of GPT-2 XL perplexity to GPT-2 Small, to GPT-2 Medium, to zlib entropy, and to the perplexity of the lowercased sample; minimum perplexity over a 50-token sliding window), and inspect the top candidates (§6.1).

## Results (§1, §6.2, Table 1)
- 1,800 candidates (100 for each of 3 × 6 configurations); 604 unique memorized training examples confirmed with the GPT-2 authors. In the best configuration, 67% of candidates were verbatim training examples.
- Categories among the 604 (Table 1) include named individuals (non-news samples) 46 and contact information (address, email, phone, twitter) 32.
- One extracted sample contains an individual's full name, physical address, email address, phone number, and fax number (Fig. 1).
- The abstract states that extraction "is possible even though each of the above sequences are included in just one document in the training data".
- "Larger models are more vulnerable than smaller models" (Abstract; §7 "Model Size & Insertion Frequency").

## Mitigations discussed (§1, §8)
Differentially private training (utility and training-time cost), deduplication of documents ("empirically will help to mitigate memorization but cannot prevent all attacks"), and auditing.

## How ch-11 uses it
§6 (extraction evidence for PII), Negative samples, Generalization lens.
