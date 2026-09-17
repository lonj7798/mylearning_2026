---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/repeated-data-scaling.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2205.10487
created_at: "2026-09-15"
---

# Excerpt: Scaling Laws and Interpretability of Learning from Repeated Data

**Authors:** Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, et al. (Anthropic)
**Version read:** arXiv:2205.10487v1 (21 May 2022).
**Status:** no library card existed for this slug on 2026-09-15; values read in the v1 PDF text.

## Setup (Fig. 1, §1.1, §3)
- Decoder-only transformers, 8,192-token context, trained for 100B tokens with settings of Askell et al. (2021).
- Source dataset: 400B tokens; 55% filtered Common Crawl (220B), 32% internet books (128B), plus OpenWebText, Wikipedia, Stack Exchange; GPT-2 vocabulary of 50,304 (§3).
- A fixed fraction of training tokens (3%, 10%, 20%, 50%, 90%) comes from repeats of a small random subset; the rest is seen once. Model size, repeated dataset size, and repeated fraction varied over 3, 2.5, and 2 orders of magnitude (§3). Test loss on a held-back split without repeated data.
- Python: 45B tokens for 2.2 epochs; fine-tuning uses half the learning rate and reduced warmup (§3).

## Results
- Abstract: an 800M-parameter model can be degraded to the performance of a 2× smaller model (400M) by repeating 0.1% of the data 100 times while 90% of training tokens remain unique. §2 states the degradation is "nearly to that of a 340M parameter model".
- Double descent (§2, Fig. 2-3): damage peaks at an intermediate number of repeats; the peak coincides with train loss on the repeated subset approaching zero. With 90% repeated data (100×-10,000× repeats) a literal double descent appears; with 50% it becomes a long plateau.
- Band of poor performance (§2, Fig. 4): both boundaries fit E = k·N^α (E repeated epochs, N parameters): right boundary k = 5.1e7, α = −0.50; left boundary k = 4.2e6, α = −0.56. Extrapolation predicts significant degradation from repeating data as little as 2× for models with hundreds of billions of parameters at 100B total tokens; the authors expect less degradation for longer training.
- 10% repeated, 1,220 repeats: dip to 0.55× effective model size at 10M-100M parameters, recovering to 0.8× at 1B (§2).
- Copying (§1.1, §2, Fig. 5): loss on the first Harry Potter paragraph copied 11 times; 3% repeated data at the worst repeat count gives a 3× reduction in effective model size on copying against at most 1.15× on test loss.
- Prefix matching (induction heads) is preferentially degraded at low repeated fractions (§2, Fig. 6).
- Out-of-distribution Python loss drops disproportionately mostly at 50-90% repetition (§1.1).
- Fine-tuning (§2, §5.4): §2 states that an 800M model pre-trained on 50% repeated data at the peak had effective parameters reduced 10×, and that fine-tuning on Python from it gives a 1.6× reduction against training from scratch. §5.4 attributes the 1.6× ossification effect to 90% repeated tokens (73× reduction before fine-tuning), and Fig. 12 describes 90% repeated data; the two sections disagree on the repeated share.
- Worked arithmetic (§5.1): losses of roughly 2.0 nats/token (800M) and 2.2 (400M); memorizing 10% repeated tokens to 0 loss while degrading the rest to 2.2 gives 0.9 × 2.2 + 0.1 × 0 = 1.98 < 2.0.

## Limitations (§5.5)
Fixed 100B tokens (pre-Chinchilla); noisy boundary fits; repeated data is a random subset, not intentionally upweighted high-quality data; loss rather than downstream evaluations; no regularization studied.

## How ch-12a uses it
§9 (non-uniform repetition vs rephrasing), figure panel C, Generalization lens (b), Common mistakes.
