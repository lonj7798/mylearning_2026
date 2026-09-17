<!-- excerpt for ch-48; extract of one primary source. Loci are sections/tables of the arXiv PDF.
     Created 2026-09 (generality revision) from https://arxiv.org/abs/2310.16789 (ICLR 2024).
-->

# Detecting Pretraining Data from Large Language Models (Min-K% Prob)

- **Authors:** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, et al.
  (University of Washington; Princeton University)
- **Year:** 2023 (arXiv v1 2023-10-25; ICLR 2024)
- **URL:** https://arxiv.org/abs/2310.16789
- **Source type:** paper

## What the chapter uses

**Method (§3).** Score a candidate text x = x_1..x_N under the model, keep the k% of tokens with the lowest
token probability as the set Min-K%(x), and average their log-likelihoods:

    MIN-K%PROB(x) = (1/E) · Σ_{x_i ∈ Min-K%(x)} log p(x_i | x_1..x_{i-1})

E is the size of Min-K%(x). A high value means few high-surprise tokens, which the authors take as evidence
that the text was seen in pre-training. Detection is a threshold on this score; no reference model and no
training data are needed. k = 20 was chosen by a sweep over {10, 20, 30, 40, 50} on a held-out set with
LLaMA-65B and used everywhere afterwards (§4.3).

**Benchmark results (§4.3, Table 1).** On WIKIMIA (Wikipedia text split by model cutoff date), average AUC
over Pythia-2.8B, NeoX-20B, LLaMA-30B, LLaMA-65B and OPT-66B in original and paraphrase settings:
Min-K% Prob 0.72; perplexity 0.67; zlib 0.65; neighbour attack 0.65; smaller-reference 0.66; lowercase 0.61.
The reported improvement over the best baseline is 7.4% AUC.

**Benchmark-leak simulation (§6, Table 3).** 200 examples each from BoolQ, IMDB, TruthfulQA and Commonsense
QA were inserted into a 27M-token RedPajama corpus (0.1% of tokens) and LLaMA-7B was fine-tuned for one
epoch at constant LR 1e-4. AUC for separating the inserted 200 from 200 held-out examples: Min-K% Prob 0.86
average (BoolQ 0.91, Commonsense QA 0.80, IMDB 0.98, TruthfulQA 0.74); perplexity 0.84; zlib and lowercase
0.68; neighbour 0.66.

**Conditions (§6.2).** Detection degrades as the number of occurrences falls and as the learning rate falls
(raising LR from 1e-5 to 1e-4 raised AUC on all four tasks). AUC rose with corpus size when the inserted
examples were distributional outliers, and did not when a control corpus of post-cutoff news was used
instead.

**Copyright case study (§5).** AUC 0.88 on a validation set of 50 memorized and 50 post-cutoff books for GPT-3
(text-davinci-003), above perplexity 0.84 and zlib 0.81.

## Verification
- Read on 2026-09-15 against the ICLR 2024 PDF of arXiv:2310.16789, §§3–6 and Tables 1–3.
- Not reported: per-instance precision at a fixed operating point for production-scale corpora; results for
  models above 66B; performance when the benchmark was seen once.
