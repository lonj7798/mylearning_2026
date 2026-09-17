---
chapter: ch-08b
course: llm-training
phase: read
excerpt_of: primary source arXiv:2205.05055v6 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2205.05055
created_at: "2026-09-15"
---

# Excerpt: Data Distributional Properties Drive Emergent In-Context Learning in Transformers

**Paper:** Stephanie C.Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya K. Singh, Pierre H. Richemond, et al. (DeepMind; University College London; Stanford University). arXiv v1 2022-05, read at v6 (2022-11-17). Source type: paper.

## Question and hypotheses (Abstract, §1)
- Asks which aspects of the training regime lead to in-context learning (ICL) without explicit meta-training.
- Candidate data properties of natural data: burstiness (items cluster in time), a skewed Zipfian marginal with a long tail of rare items, and dynamic meaning (polysemy, synonymy).
- In-weights learning (IWL): slow, gradient-based storage in weights, as in standard supervised learning.

## Design (§2, App. A)
- Omniglot: 1,623 character classes, 20 handwritten examples each (§2.1).
- Training sequence: 8 image-label pairs (16 context elements) then a query image; loss on the query label; labels fixed across training (§2.1).
- Bursty sequences: the query class appears 3 times in the context, and a second image-label pair also appears 3 times; non-bursty sequences are drawn uniformly (§2.1).
- Model: embedder (ResNet for images, embedding layer for labels) + causal transformer, 12 layers, embedding size 64, 8 heads; sinusoidal positions (§2.2, App. A).
- Training: 500k steps on 16 TPU v2/v3 cores; Adam; linear warmup to 3e-4 at 4000 steps, then inverse square root decay; 3 seeds for Figs. 5-6, 5 runs otherwise (App. A).
- ICL evaluation: 4-shot 2-way on holdout classes, labels re-assigned to 0/1 per sequence, accuracy over the two labels, chance 1/2 (§2.3).
- IWL evaluation: trained classes with training labels, query class absent from context, chance usually 1/1600 (§2.3).
- ICL is evaluated on novel classes but not novel labels (§2.3, App. B).

## Results (§3)
- Burstiness: more bursty training gives better ICL and lower IWL; "the models can in some cases lose an initial bias towards in-context learning, moving towards in-weights learning over the course of training" (§3.1, Fig. 2).
- Number of classes at p(bursty) = 0.9: 100 → 1600 improves ICL and lowers IWL; 12,800 classes (rotations and flips) improve ICL further; "we need both burstiness and a large number of classes for in-context learning to emerge" (§3.1, Fig. 3).
- Label multiplicity (several labels per class, consistent within a sequence) increases ICL (§3.1, Fig. 4).
- Within-class variation (single exemplar, pixel noise 0.1 or 0.5, full Omniglot) increases ICL and decreases IWL (§3.1, Fig. 5).
- Zipfian marginal, Eq. 1: p(X = x) ∝ x^(−α), X the class rank, α ∈ [0, ∞) the skew (the extracted text drops the minus sign; rank 1 is the most common class). With 12,800 classes and p(bursty) = 0.9: no skew gives ICL but no IWL; more skew loses ICL and gains IWL on the 10 most common classes; α = 1 keeps both; rare classes stay at chance for all exponents (§3.2, Fig. 6). Footnote 3: at α = 3 the three most common classes form 97% of the data.
- The α = 1 sweet spot is for "this particular training regime" (Fig. 6 caption).
- Architecture: vanilla RNN and LSTM matched on layers, hidden size, and parameter count never exceed chance on ICL (1600 classes, p(bursty) = 0.9); transformers also have similar or slightly higher IWL (§3.3, Fig. 7, App. C.2).
- Sweep: max LR 15 log-uniform samples in [1e-5, 0.1], warmup 15 log-uniform samples in [1, 10000]; 15 runs per architecture (2 or 12 layers), 90 runs total. Parameters: Transformer 12L 831,479; LSTM 12L 627,959; Transformer 2L 331,639; LSTM 2L 297,719 (App. C.1).
- Multi-class evaluation shows the same patterns; α = 1 models output labels from context less often (App. C.4).

## Discussion (§4)
- Transformer architecture alone was insufficient; data needed at least burstiness and many classes.
- "Broader implications": supervised datasets are frequently rebalanced toward uniform distributions; the authors suggest this may miss an opportunity for in-context learning in non-language domains.
- "Future directions": experiments with symbolic inputs and next-token or masked-token prediction are listed as open.

## Verification
- Read on 2026-09-15 against arXiv PDF v6 (Abstract, §1-§4, App. A-C). The paper reports results as figures; no per-condition accuracy numbers are printed in the text, so none are quoted.
