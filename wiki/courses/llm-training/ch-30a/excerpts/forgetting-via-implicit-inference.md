---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/forgetting-via-implicit-inference.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2309.10105
created_at: "2026-09-15"
---

# Excerpt: Understanding Catastrophic Forgetting in Language Models via Implicit Inference

**Authors:** Suhas Kotha, Jacob Mitchell Springer, Aditi Raghunathan (Carnegie Mellon University)
**Version read:** arXiv:2309.10105v2 (14 Apr 2024), "Published as a conference paper at ICLR 2024"; v1 September 2023.
**Status:** no library card existed for this slug on 2026-09-15; quotes checked against the v2 PDF text.

## Hypothesis (§2.7, Eq. 7)
w_θ(X, y) = g_θ(X, y) w_disc(X, y) + (1 − g_θ(X, y)) w_cont(X, y), where g_θ is "some weighting function on the discrete solution estimating the posterior probability that the prompt is drawn from D_disc." "A capability refers to whether the transformer can internally perform an algorithm of interest ... and task inference refers to whether the model can correctly disambiguate which algorithm to use." The authors add: "Due to limited mechanistic understanding, we can not test whether this is how language models compute solutions."

## Synthetic evidence (§2.5-2.8)
A 22.4M-parameter GPT-2-style transformer is pretrained on a mixture of 64 fixed regression tasks and a Gaussian task prior, then fine-tuned for 400 steps on the fixed tasks. Fine-tuning "rapidly improves performance on D_disc" and causes "large performance drops on D_cont" (§2.5). "the largest increase is for D_disc samples closest to D_cont" (Fig. 5 caption). Scaling the labels (conjugate prompting with γ ∈ {1.5, 2.0}) "recovers the pretrained solution of ridge regression, especially on lower sample counts" (Fig. 6 caption).

## Language-model evidence
- Table 1 (400 samples, 4 ICL-vs-instruction tasks): LLaMA → Alpaca, English: pretrained ICL accuracy 92.00%, fine-tuned 35.25%, drop 56.75%; French drop 29.00%; Leetspeak drop 1.50%.
- Table 2 (XNLI, 2490 test samples): LLaMA-2 → Code LLaMA, English 44.26% → 35.90% (drop 8.36%); Spanish 38.11% → 38.88% (drop −0.77%).
- Table 3 (100 AdvBench instructions): answer rate for "GPT-3.5 without safety fine-tuning" vs ChatGPT: English 92% → 3%; Malayalam 71% → 65%.

## Quote used in ch-30a
"we claim the ridge regression solution has not been 'forgotten' but 'suppressed' since it can be partially recovered through manipulating task inference" (§2.8).

## How ch-30a uses it
§3.1 (task inference as a mechanism) and the measurement consequence: a score drop on an English benchmark can coexist with a retained capability that a transformed prompt recovers.
