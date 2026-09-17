---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2508.14444v4 (NVIDIA Nemotron Nano 2), §2.6 Long-Context Extension, Table 4, §3.2 SFT stages (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2508.14444
created_at: "2026-09-15"
---

# Excerpt: NVIDIA Nemotron Nano 2 — Phase LC synthetic document-QA data

- **Authors:** NVIDIA
- **Year:** 2025 (arXiv v1 2025-08; v4 2025-09-02 used here)
- **Source type:** official technical report
- **Used in:** ch-29a §2.3, §7, Recipe

## Phase LC (§2.6)
- "In Phase LC, we did continuous pretraining (CPT) with a context length of 524,288 (512k) tokens using a constant learning
  rate of 4.5 · 10−6. Although the target context length of Nemotron Nano 2 is 128k, in preliminary studies on the Nemotron-H
  8B model, we found it better to do CPT with 512k sequence length, instead of 256k or 128k."
- Stated intuition: "longer training sequence can effectively lower the chance of long coherent documents being cut and separated
  by the Concat & Chunk algorithm for pretraining data loading." (Interpretation by the authors.)
- "We used a global batch size of 12 to ensure the total number of tokens per global batch during long-context CPT is the same as
  during pretraining: around 6M tokens. Phase LC consisted of 18.9 billion tokens."

## Synthetic long-document QA (§2.6)
- Seeds: academic pretraining documents "longer than 32k tokens".
- "We split each document into chunks of 1,024 tokens and then randomly selected 10% of the chunks to be fed into
  Qwen-2.5-72B-Instruct for data synthesis. We asked the generator to generate a QA pair based on the information in the text
  chunk. We concatenated the QA pairs and appended them to the end of the original document as a sample of the long-context
  document QA data."
- Blend: "We proportionally downscaled the weights of all Phase 3 data to 80% of their original values, allocating the remaining
  20% to the newly added long-context document-QA data." The authors report this extended context "without degrading regular
  benchmark scores"; no before/after table for the 12B model is printed in §2.6.

## Table 4 (ablation on Nemotron-H 8B)
| Train length | 128k | 256k | 256k | 512k |
|---|---|---|---|---|
| Synthetic data | yes | no | yes | yes |
| RULER-128k | 73.68 | 70.19 | 79.04 | 81.04 |

## SFT stages (§3.2)
- Stage 1 concatenates samples "into sequences of approximately 128k tokens, reducing padding overhead and encouraging
  long-range learning."
- Stage 2: "tool-calling accuracy degraded. We attribute this to sample concatenation at 128k, which likely disrupted learning of
  tool-calling patterns. Thus, Stage 2 was trained without concatenation."
- Stage 3 "reinforces long-context capability. It incorporates long-context data following the recipe used in Nemotron-H
  preparation", plus reasoning traces truncated to 1–2k tokens.

## Not reported
Number of QA pairs, QA verification or filtering, loss masking on appended QA, the 12B short-benchmark before/after Phase LC.
