---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: primary source arXiv:2406.07515v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2406.07515
created_at: "2026-09-15"
---

# Excerpt: Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification

**Paper:** Yunzhen Feng, Elvis Dohmatob, Pu Yang, Francois Charton, Julia Kempe (Meta FAIR, NYU, Peking University). arXiv v1 2024-06; v2 2024-10-25 read. Source type: paper.

## Setting
- Synthesized data are labels (answers) for given inputs; a verifier selects which synthesized examples are kept (§1). Training is a single round on selected synthesized data; the theory uses only synthesized data "since the inclusion of additional real data is always advantageous" (§4).
- "Model collapse" in the experiments: a model trained on synthesized data scores below the generator (§6).

## Warm-up: the generator can produce better answers than it selects (§3, Table 1)
- Transformer predicting eigenvalues of 5×5 symmetric matrices, generator trained on 200,000 examples.
- Tolerance τ = 1%: greedy 20.2%; best of 50 beams (oracle picks) 60.4%; lowest-perplexity beam of 50 (model picks) 19.2%.

## Theory (§4, Theorem 4.2)
- p = P(synthesized label ≠ true label). Pruner parameters: ϕ = P(keep | label correct), ψ = P(keep | label wrong).
- Breakdown point p* = 1/(1 + ψ/ϕ). With infinite data: accuracy 100% if p < p*, 0% if p > p*.
- No pruning: ψ/ϕ = 1, p* = 1/2. Oracle pruning: ψ/ϕ = 0, p* = 1 (§4.4).
- Finite-data simulation: an oracle verifier matches training on clean labels; a weak verifier (θ_prune = π/12) has a best data size near 10,000 examples, and more data is worse (§5.2, Fig. 3).

## Experiments (§6)
- Eigenvalues, noisy verifier with p* = 1/(1 + p_noise): without verification (p* = 0.5), 10M synthesized examples (50× the generator's training set) score below the generator; the oracle verifier nearly doubles accuracy; the curve crosses the generator near p* ≈ 0.65 (§6.1, Fig. 4).
- News summarization: Llama-2-7B generator fine-tuned on 12.5% of XLSUM English (307,000 training examples), greedy decoding; selection rates 12.5%, 25%, 50%; ROUGE-1 metric (§6.2).
  - Random selection with the same data size scores below the generator (Fig. 5 left).
  - Oracle selection beats the generator in all settings; oracle selection of 12.5% beats a model trained on 100% of the original labels.
  - Self-selection by perplexity beats the generator; selection by a fine-tuned Llama-3 (higher ROUGE-1 than the generator) is close to random selection; p* tracks this ordering (§6.2).
- Authors' explanation for self-selection gains: selection over a larger pool of articles changes the input distribution (§7; Interpretation).
- Verifiers used in prior work are named for code and mathematics (compilers, solutions, heuristic verifiers) (§2).

## Training details (App. D)
- Generator: pre-trained Llama-2, LR 5e-5 cosine, 1 epoch, total batch 32, block size 1024. Model trained on selected data: LR 2e-5 constant, other settings the same. Greedy decoding throughout.

## Verification
- Read on 2026-09-15 against arXiv:2406.07515v2 PDF text (§1–7, App. D).
