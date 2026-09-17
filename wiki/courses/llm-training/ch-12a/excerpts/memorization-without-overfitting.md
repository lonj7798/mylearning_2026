---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/memorization-without-overfitting.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2205.10770
created_at: "2026-09-15"
---

# Excerpt: Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models

**Authors:** Kushal Tirumala, Aram H. Markosyan, Luke Zettlemoyer, Armen Aghajanyan (Meta AI Research)
**Version read:** arXiv:2205.10770v2 (2 Nov 2022); v1 May 2022; NeurIPS 2022.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v2 PDF text.

## Definition (§3, Definition 1)
A context c = (s, y) is memorized if argmax f(s) = y. Exact memorization M(f) is the fraction of training contexts memorized (token-level argmax accuracy on training data). T(N, τ) is the minimum number of times a model with N parameters must see each training example to reach M(f) ≥ τ; T_update(N, τ) counts gradient updates for single-pass data.

## Setup (§3, App. A.4)
- Models 125M, 355M, 1.3B, 2.7B, 6.7B, 13B (FairSeq); causal and masked LM.
- Data: WikiText-103 (about 103M tokens) and the RoBERTa corpus (about 39B tokens).
- Adam β1 0.9, β2 0.98, ε 1e-8; weight decay 0, dropout 0; polynomial schedule, warmup over 375M tokens, decay to 0 over T − 375M tokens (T = 100B for causal WikiText-103; T = 300B for masked and RoBERTa runs); sequence length 512 (App. A.4).
- Peak LR and global batch (tokens) (App. A.4 Table 1): 125M 6.0e-4, 0.5M; 355M 3.0e-4, 0.5M; 1.3B 2.0e-4, 1M; 2.7B 1.6e-4, 1M; 6.7B 1.2e-4, 2M; 13B 1.0e-4, 2M.

## Results
- Larger models memorize faster: T(N, 0.9) decreases monotonically with N for causal LM on WikiText-103 (§4, Fig. 1); T_update(N, τ) decreases with N on the RoBERTa corpus for causal and masked LM (§4.1, Fig. 3). Masked LM on WikiText-103 is not monotonic for τ below a transition between 0.6 and 0.7 (§4.1, Fig. 2).
- Memorization before overfitting (first epoch where validation perplexity rises) increases with model size (§4.2, Fig. 4). The effect remains at a fixed learning rate (§4.2, Fig. 5).
- Prepending a unique document ID speeds memorization for the 125M model (§4.3, Fig. 6).
- Nouns, proper nouns, and numerals are memorized faster than verbs and adjectives (§4.4, Fig. 7, 355M).
- Forgetting (§5): a held-out "special batch" seen once is forgotten quickly at first, then memorization approaches a floor, the "forgetting baseline" (Fig. 8, 2.7B). The baseline increases monotonically with model size (Fig. 8 right). It is not sensitive to batch order (Fig. 9). Repeated injection raises the baseline (differences of order 10^-2); spaced repetition has minimal effect (order 10^-3) (Fig. 10, 125M).
- In forgetting experiments perplexity on the special batch can still rise while memorization approaches the baseline (§6, App. A.2 Fig. 14).

## Limits stated by the authors
Definition 1 targets information that is sensitive if output verbatim, not all of privacy (§6). A complete explanation of why larger models memorize faster is out of scope (§4.2).

## Not reported
Downstream benchmark effects; numeric values of M(f) before overfitting in the text (only plotted in Fig. 4).

## How ch-12a uses it
§1 (exact memorization), §4 (memorization before overfitting, forgetting baseline), Recipe.
