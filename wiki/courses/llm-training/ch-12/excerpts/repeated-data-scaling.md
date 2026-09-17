---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: primary source arXiv:2205.10487v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2205.10487
primary_version: arXiv:2205.10487v1 (2022-05-21), full text read 2026-09-15
created_at: "2026-09-15"
---

# Excerpt: Scaling Laws and Interpretability of Learning from Repeated Data

**Paper:** Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, et al. (Anthropic). arXiv v1 2022-05. Source type: paper.

## Setup (§1, Figure 1, §3)
- Training data per run: a fraction drawn once from a 400B-token text dataset and a fraction made of repeats of a small subset of the same dataset. The main text runs hold 10% of training tokens as repeats (other scans use 3%, 20%, 50%, 90%, Figure 4 caption) and vary the size of the repeated subset (for example 0.01% of tokens repeated 1,000 times, or 1% repeated 10 times) (Figure 1 caption).
- Text dataset: 55% heavily filtered Common Crawl (220B tokens), 32% internet books (128B), plus OpenWebText, Wikipedia, Stack Exchange, mostly from The Pile; GPT-2 encoding with 50,304 vocabulary (§3).
- Decoder-only Transformers, 8,192-token context, trained for 100B tokens; test loss on a held-back set that contains no repeated data (§3, Figure 1).
- Model size, repeated-subset size, and repeated-token fraction varied by 3, 2.5, and 2 orders of magnitude (§3). Model sizes in the figures range from 1.57M to 805M parameters (Figures 5–6).

## Results
- **Double descent.** Test loss can rise midway through training and fall again; with 90% repeated data the curve is a literal double descent, with 50% a long plateau (§2, Figure 3).
- **Size of the effect.** Abstract: an 800M model is degraded "to that of a 2x smaller model (400M params) by repeating 0.1% of the data 100 times", while 90% of training tokens stay unique. §2: for 10% repeated data the worst range for an 800M model is "roughly 100x repeats of 0.1% of the data", with performance "nearly to that of a 340M parameter model".
- **Diagnostic.** The peak degradation coincides with training loss on the repeated subset approaching zero (§1.1, Figure 2 right).
- **Poor-performance region.** Both boundaries of the region with at least 50% of the maximum degradation fit E = k·N^α, where E is repeated epochs and N is parameters: right boundary k = 5.1e7, α = −0.50; left boundary k = 4.2e6, α = −0.56 (§2). Extrapolation predicts degradation from repeating data about 2 times for models with hundreds of billions of parameters, for a fixed 100B training tokens; the authors state training past the peak helps, so the effect would likely be smaller in practice (§2).
- **Copying.** Loss on the first paragraph of Harry Potter copied 11 times: 3% repeated data at the worst repeat count gave up to a 3× reduction in effective model size, against at most 1.15× on test loss (§1.1, §2, Figure 5).
- **Induction heads.** Prefix-matching score at 3% repeated tokens: §1.1 gives an average 32% reduction in effective model size, the Figure 6 caption an average 1.47 model-size multiplier, and the §2 text a 2× effective parameter decrease; test loss shows at most 1.15× (§1.1, §2, Figure 6). In a 2-layer attention-only model with 50% repeated data, paragraph-level copying broke down (Figure 7).
- **Out of distribution.** Python loss for text models: model-size multipliers of 0.84 and 0.75 at 50% and 90% repeated data; the disproportionate drop appears mainly at high repeated fractions (Figure 10).
- **Fine-tuning.** A model pre-trained on repeated data at the double-descent peak, then fine-tuned on Python, had a 1.6× reduction in effective parameters relative to training from scratch (§2, §5.4). §5.4 and the Figure 12 caption state 90% repeated tokens (73× effective-size reduction before fine-tuning); the §2 text describes the 800M experiment as 50% repeated data with a 10× reduction (Figure 15).

## Author interpretation (§5.1)
Informal arithmetic: an 800M model has loss about 2.0 nats/token, a 400M model about 2.2, and fully memorized data 0. Memorizing a 10% repeated share at the cost of a 2× effective size reduction gives 0.9 × 2.2 + 0.1 × 0 = 1.98 < 2.0, so the training objective can favor memorization. The degradation peaks where the repeated data can be memorized and doing so uses a large fraction of capacity (Interpretation).

## Limitations stated by the authors (§5.5)
1. Fixed 100B training tokens for all models (pre-Chinchilla sweep).
2. Poor-performance-region fits are noisy and needed aggregation.
3. The repeated data was a random subset, not intentionally upsampled high-quality data such as Wikipedia.
4. Loss was measured, not downstream NLP evaluations.
5. No early stopping, dropout, weight decay, or other regularization explored.
