---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/classics/unlikelihood-training.md on 2026-09-15)
source_url: https://arxiv.org/abs/1908.04319
source_version: arXiv v2 (2019-09-26); v1 2019-08
created_at: "2026-09-15"
---

# Excerpt: Neural Text Generation with Unlikelihood Training (Welleck, Kulikov, Roller, Dinan, Cho, Weston; FAIR/NYU)

This file holds the facts from the primary source that [[read]] uses. Every number below was read in the arXiv v2 PDF on 2026-09-15.
When the library card `unlikelihood-training` is written, it is the canonical record; this excerpt must then be checked against it.

## Objective (§5.1, Eq. 3-5; §5.2, Eq. 7-8)
- Unlikelihood loss for step t with negative candidates C^t: L_UL^t = −Σ_{c∈C^t} log(1 − p_θ(c | x_<t)) (Eq. 3).
- Token-level objective: L_UL-token^t = −α · Σ_{c∈C^t} log(1 − p_θ(c | x_<t)) − log p_θ(x_t | x_<t) (Eq. 4).
- Token-level candidates: previous context tokens, C_prev-context^t = {x_1, …, x_{t−1}} \ {x_t} (Eq. 5).
- Sequence-level: decode a continuation from the model, and make x_t the single candidate if it is part of a repeated n-gram (Eq. 7-8). Fine-tuning mixes sequence-level updates and the original token-level loss with probability 0.5 each (§6 Training).

## Gradient (§5.1 Eq. 6; App. A Eq. 11-20)
- With p = softmax(a) and one negative candidate, the (negative) gradient is ∇L_a = x* − m ⊙ p, with m_i = 1 − α·p_neg/(1 − p_neg) for i ≠ i_neg and m_i = 1 + α for i = i_neg (Eq. 6, Eq. 19-20).
- App. A Eq. 12: ∂L/∂a_i = (1[i = i*] − p_i) − α·(p_neg/(1 − p_neg))·(1[i = i_neg] − p_i), for the objective L = log p(x*_t) + α·log(1 − p(x_neg)) (Eq. 11).
- The paper states that the ground-truth token's gradient grows with p_neg, the negative candidate's gradient is negative, and other tokens move from negative to positive as p_neg increases; with α = 1.0 every token's probability is increased when p_neg > 0.5 (§5.1).

## Setting and results (§6; Table 2 test split; Table 3)
- Model: 16-layer Transformer, 8 heads, embedding 1024, FFN 4096; Wikitext-103 (word level); sequence length 1,536; token-level training up to 150k updates on 8 GPUs, checkpoint with best validation perplexity; sequence-level fine-tuning 1,500 updates; prefix k = 50, continuation N = 100 (§6).
- The α value used in the Wikitext-103 runs: not reported in §5-6 or App. A-D.
- MLE baseline, beam search: seq-rep-4 .523, uniq-seq 9.5k; perplexity 25.64 (Table 2). L_UL-token+seq, beam: seq-rep-4 .013, uniq-seq 19.1k; perplexity 26.72. Human text: .006, 19.8k (Table 2).
- Greedy MLE continuations repeat 43% of n-grams vs 0.5% for human text; next-token predictions fall in the previous 128 words 62% of the time vs 49% for ground truth (§4).
- Human evaluation (Table 3, beam size 10): L_UL-token+seq beats the MLE baseline in 82% of crowdworker comparisons (significant). Against nucleus sampling (p = 0.9): crowdworkers 59% (not marked significant), experts 83% (significant). Against 4-gram beam blocking: crowdworkers 60% (not marked significant), experts 74% (significant).
- GPT-2 (medium) fine-tuned with sequence-level unlikelihood: seq-rep-4 .042 vs .506 (§6, App. C, Table 7); 10,000 single-GPU updates.

## Limits stated or visible in the source
- Negatives are defined by repetition (context tokens, repeated n-grams) or random tokens (App. D); the paper does not test semantic negatives such as wrong answers.
- Evaluation is language modeling and completion on Wikitext-103; no instruction-following or reasoning tasks.
