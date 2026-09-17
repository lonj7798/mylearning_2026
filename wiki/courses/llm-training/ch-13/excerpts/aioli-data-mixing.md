---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/aioli-data-mixing.md (planned card; not present on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2411.05735
primary_version: arXiv:2411.05735v2 (2025-04-21); v1 2024-11
created_at: "2026-09-15"
---

# Excerpt: Aioli: A Unified Optimization Framework for Language Model Data Mixing

Authors: Mayee F. Chen, Michael Y. Hu, Nicholas Lourie, Kyunghyun Cho, Christopher Ré (Stanford University; New York University; Genentech). Source type: paper. Read in the v2 PDF text on 2026-09-15 for ch-13 `read.md` §1, §2, §3, and the generalization lens.

## Main claims (Abstract)
> "Surprisingly, we find that no existing method consistently outperforms a simple stratified sampling baseline in terms of average test perplexity."

> "Aioli outperforms stratified sampling on 6 out of 6 datasets by an average of 0.27 test perplexity points, whereas existing methods fail to consistently beat stratified sampling, doing up to 6.9 points worse."

## Framework (§1, §3)
Linear Mixing Optimization (LMO): methods minimize average loss per group subject to a mixing law L_{t+1}(p_t) = σ(A_t p_t) (equal up to linear transformation), where p_t is the mixture over m groups at round t, A_t ∈ R^{m×m}, and σ = Id or exp. "Existing offline methods assume a static (T = 1) log-linear parameterization of the mixing law, while online methods assume a linear dynamic mixing law." Both parameterizations fit the true loss-proportion relationship with "an average of 0.0005 MSE and 0.969 R²"; the methods' parameter values deviate from fitted optima, and the deviation correlates with performance relative to stratified sampling (Figure 3). DoReMi in this framework trains a reference model with stratified sampling (§3.3).

## Setup (§4.1, §6)
SlimPajama groups: ArXiv/StackExchange, GitHub/C4, Books/StackExchange (m = 2); ArXiv/Books/StackExchange, CommonCrawl/GitHub/Wikipedia (m = 3); all 7 groups. "We train 160M parameter GPT-style decoder-only LLMs with batch size 8 and context length 2048. For m = 2, 3, we train for 5K steps, and for m = 7, we train for 40K steps." Stratified sampling sets p_i = 1/m. Unrestricted setting: up to 10S extra training steps per method.

## Table 2 (difference in average test perplexity vs stratified; negative = better)
| Method | A/SE | GH/C4 | B/SE | A/B/SE | CC/GH/W | SlimPajama | # < stratified | # extra runs |
|---|---|---|---|---|---|---|---|---|
| Stratified | 16.532 | 35.991 | 47.192 | 35.114 | 41.583 | 26.426 | - | 0 |
| GS | −0.399 | −0.407 | −0.645 | −0.247 | 0.298 | 0.490 | 4 | 10 |
| DML | −0.241 | −0.110 | −0.644 | −0.599 | 0.242 | 1.641 | 4 | 10 |
| Skill-It | −0.326 | 0.551 | −0.728 | −0.568 | −0.195 | −0.184 | 5 | m |
| DoReMi | −0.307 | 5.303 | −0.217 | −0.393 | 6.898 | 0.703 | 3 | 2 |
| DoGE | 0.419 | 0.184 | −0.678 | 1.843 | 0.604 | 0.949 | 1 | 1 |
| Aioli | −0.205 | −0.340 | −0.439 | −0.226 | −0.196 | −0.240 | 6 | 0 |
(Stratified row gives absolute perplexity; A = ArXiv, B = Books, GH = GitHub, SE = StackExchange, W = Wikipedia; DML = Data Mixing Laws.)

## Restricted setting (§6.2, Table 3)
When methods learn proportions on shortened runs, adding Aioli's online adjustment improves average test perplexity "in 28 out of 30 settings, by an average of 1.202 and a maximum of 12.012 points".

## Perplexity versus downstream tasks (App. F.1)
> "We find that lower perplexity is positively correlated with worse performance on downstream tasks. ... The correlation between perplexity and the macroaverage of our downstream tasks is 0.529, indicating that lower perplexity is predictive of worse downstream performance. In fact, DML obtains the best overall performance, even though it omits three out of seven datasets in SlimPajama"

Tasks: ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, LAMBADA, OpenBookQA, PiQA, WinoGrande (lm-eval-harness). App. F.4 repeats experiments with 1.4B models (not used in ch-13).

## Limits named by the source (§8)
Aioli adds evaluation cost; the role of group partitions is open ("C4 is a subset of CommonCrawl, and it is unclear if disjoint groups could improve performance").
