---
chapter: ch-08a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2203.15556v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2203.15556
created_at: "2026-09-15"
---

# Excerpt: Training Compute-Optimal Large Language Models

**Paper:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, et al. (DeepMind). arXiv v1 2022-03 (v1 read). Source type: paper.

## Question and data (Abstract, §1, Eq. 1)
- N_opt(C), D_opt(C) = argmin over (N, D) with FLOPs(N, D) = C of the final pre-training loss L(N, D) (Eq. 1).
- Over 400 models from 70M to over 16B parameters on 5B to 500B tokens (Abstract). Loss is the smoothed training loss, which the authors treat as an unbiased estimate of test loss because training uses less than one epoch (§1, footnote 2).
- FLOPs include embedding matrices, and parameter counts include embeddings (App. F).

## Three approaches (§3.1–3.3, Table 2)
- Approach 1: fixed model sizes (70M to over 10B), each trained with 4 cosine horizons spanning 16×; the minimum-loss envelope over 1500 log-spaced FLOP values gives N_opt ∝ C^a, D_opt ∝ C^b (§3.1).
- Approach 2 (IsoFLOP): 9 budgets from 6 × 10^18 to 3 × 10^21 FLOPs, model sizes up to 16B, a parabola fitted to each loss-versus-size curve, cosine length matched to the token count (§3.2).
- Approach 3: parametric fit to all final losses (§3.3):
  L̂(N, D) = E + A/N^α + B/D^β (Eq. 2),
  fitted by minimizing Σ_i Huber_δ(log L̂(N_i, D_i) − log L_i) with L-BFGS from a grid of initializations, δ = 10^−3 (Eq. 3, App. D.2).
- Fitted values: E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28 (App. D.2, Eq. 10).
- Closed-form frontier under FLOPs ≈ 6ND: N_opt(C) = G·(C/6)^a, D_opt(C) = G^−1·(C/6)^b, G = (αA/(βB))^(1/(α+β)), a = β/(α+β), b = α/(α+β) (Eq. 4).
- Table 2 exponents a (b), with 10th–90th percentiles from bootstrapping: Approach 1 0.50 (0.50); Approach 2 0.49 (0.51); Approach 3 0.46 (0.54), percentile interval 0.454–0.455 for a; Kaplan et al. 0.73 (0.27).
- Approach 3 predicts smaller optimal models at large budgets; low-FLOP points (C ≤ 10^21) have larger residuals and are down-weighted by the Huber loss (§3.4).

## Allocation table (Table 3, Approach 1)
| Parameters | FLOPs | Tokens |
|---|---|---|
| 400M | 1.92e19 | 8.0B |
| 1B | 1.21e20 | 20.2B |
| 10B | 1.23e22 | 205.1B |
| 67B | 5.76e23 | 1.5T |
| 175B | 3.85e24 | 3.7T |
| 280B | 9.90e24 | 5.9T |
| 1T | 1.27e26 | 21.2T |

## Explanation of the difference from Kaplan et al. offered by the authors (§2, App. B)
- Kaplan et al. used one fixed cosine schedule to 130B tokens; intermediate losses for D′ ≪ 130B overestimate the loss of a run whose schedule matches D′, which "eventually contributes" to the conclusion that N should grow faster than D (§2).
- Setting the cosine cycle more than 25% longer than the target number of steps "leads to clear drops in performance" (App. B, Fig. A1). Schedules decay the learning rate 10×; max LR 2 × 10^−4 for the smallest and 1.25 × 10^−4 for the largest Approach-1 models (App. D.1).
- Most models in the analysis have more than 500M parameters, whereas most Kaplan et al. runs are smaller, many below 100M (§2).

## Chinchilla versus Gopher (§4, Table 1, Table 4, Table 6)
- For the Gopher budget (5.76 × 10^23 FLOPs, Fig. 2 caption) the optimal size is "somewhere between 40 and 70 billion parameters"; Chinchilla uses 70B parameters and 1.4T tokens (§4). Approach 3 alone projects 40B (Fig. 4 caption).
- Gopher 280B: 300B tokens, max LR 4 × 10^−5, batch 3M → 6M tokens; Chinchilla 70B: max LR 1 × 10^−4, batch 1.5M → 3M tokens, batch doubled midway; AdamW instead of Adam; MassiveText with a changed subset distribution (Table 1, Table 4, §4.1).
- MMLU 5-shot: Chinchilla 67.6%, Gopher 60.0% (Table 6); the abstract states 67.5%. Chinchilla underperforms Gopher on 4 MMLU subjects (§4.2.2).

## Limits stated by the authors (§5, App. C, App. E)
- Only two comparable large runs (Chinchilla and Gopher), no intermediate-scale tests (§5).
- Concavity in log N_opt at high compute suggests the optimal size of large models may still be overestimated (§5, App. E).
- All runs use less than one epoch; the multi-epoch regime is future work (§5). IsoFLOP results on C4 and GitHub code agree "as long as one does not train for more than one epoch" (App. C).

## Verification
- Read on 2026-09-15 against arXiv:2203.15556v1 PDF text (Abstract, §1–§4.2.2, §5, App. B–F).
- Source-internal difference: MMLU 67.5% (Abstract) versus 67.6% (§4.2.2, Table 6).
- Not reported by the source: a stated "20 tokens per parameter" rule; the ratio is derived from Table 3 (20.2B/1B = 20.2; 205.1B/10B = 20.5).
