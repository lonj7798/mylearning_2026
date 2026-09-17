---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/repeated-data-scaling.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2205.10487
primary_version: arXiv:2205.10487v1 (2022-05-21)
created_at: "2026-09-15"
---

# Excerpt: Scaling Laws and Interpretability of Learning from Repeated Data

Verbatim quotations used by ch-14 `read.md`, with loci. Authors: Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, et al. (Anthropic). Checked against the v1 PDF on 2026-09-15. Source type: paper.

## Setup (Figure 1 caption, §1.1)
> "we draw 90% of our desired training dataset in a non-repeated fashion, and 10% as repeats of a tiny portion of the original dataset ... the sample to be repeated might be very small, like 0.01% of the total training tokens repeated 1000x, or relatively large, like 1% of the total training tokens repeated 10x. A small, held-back portion of the original dataset ..., not including any repeated data, is used as a test set"
> "varying the repeated dataset size, model size, and fraction of tokens trained on repeated data over 2-3 orders of magnitude. All models were trained for 100B tokens."

## Main result (Abstract)
> "performance of an 800M parameter model can be degraded to that of a 2x smaller model (400M params) by repeating 0.1% of the data 100 times, despite the other 90% of the training tokens remaining unique."

## Double descent and a diagnostic (§1.1, Figure 2)
> "data repeated a few times does not cause much damage to language model performance, data repeated very many times also does not cause much damage, but there is a peak in the middle where damage is surprisingly large."
> "the peak performance hit coincides with where the train loss on the repeated data approaches zero, similar to previously observed double-descent phenomena. This also provides a practical diagnostic for when repeated data is likely to be harming the model."
> "Extrapolating the region of large degradation in Figure 4 predicts meaningful degradation of repeating data only 2 times for large (GPT-3 size) models, though the region would be shifted if the models were trained to the compute optimal frontier"

## Effects on copying and induction heads (§1.1)
> "using 3% repeated data at the worst number of repeated epochs caused up to a 3x reduction in effective model size (performance equal to model with 3x fewer parameters) on this task whereas it only caused at most a 15% reduction in effective model size on test loss." (copying eval: the first paragraph of Harry Potter copied 11 times)
> "using 3% repeated data at the worst number of repeated epochs caused on average a 32% reduction in effective model size on this task" (prefix matching score, an induction-head measure)
> "Repeated text data causes a small but still disproportionate performance drop out of distribution, as measured by cross entropy loss on Python code. Unlike our the Harry Potter copying and prefix matching evals we mostly see the performance drop with higher levels of repetition, 50-90%."

## Data and training (§3)
> "The decoder-only transformer models were trained on an 8192 token context with the same settings as described in [Askell et al., 2021] for 100B tokens. Our language experiments utilized a 400B token dataset with 55% heavily filtered common crawl data (220B tokens), 32% internet books (128B tokens), and some smaller distributions including OpenWebText, Wikipedia, and Stack Exchange"

## Limitations (§5.5)
> "We used a fixed number of tokens for all models (similar to the GPT-3 model sweep), because these models were trained prior to the release of Chinchilla"
> "The data we repeated was a random subset of the original dataset, and is thus not directly applicable to the situation where higher quality data (such as Wikipedia) is intentionally repeated to improve quality."
> "We measured loss, rather than downstream NLP evaluations. Overfitting does not always entail worse performance on downstream tasks"
> "We did not explore the effects of early stopping, dropout, weight decay, or other regularization."

## Interpretation (Abstract)
> "We suspect there is a range in the middle where the data can be memorized and doing so consumes a large fraction of the model’s capacity, and this may be where the peak of degradation occurs."
> "data repetition disproportionately damages copying and internal structures associated with generalization, such as induction heads, providing a possible mechanism for the shift from generalization to memorization."
