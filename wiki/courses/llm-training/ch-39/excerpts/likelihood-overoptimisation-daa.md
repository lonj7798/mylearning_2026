---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2410.11677
created_at: "2026-09-15"
verified_against: "arXiv:2410.11677v2 (18 Oct 2024), cached plain text"
---

# Understanding Likelihood Over-optimisation in Direct Alignment Algorithms

Zhengyan Shi, Sander Land, Acyr Locatelli, Matthieu Geist, Max Bartolo (UCL, Cohere).

## Setting (§4.1)
Cohere Command R 7B and 35B; binarised UltraFeedback (62,600 train / 647 eval) and an internal
BINARIZED PREF set (over 100,000 examples); Hinge, DPO and IPO, six values of each method's
hyper-parameter; batch 32, max sequence length 8192, peak LR 5e-6 or 1e-5, one epoch, warmup 128 steps,
monitoring every 50 steps with early stopping; completion log-likelihood is the sum of token
log-likelihoods.

## Claims the chapter uses
- **Higher chosen likelihood is not monotonically better (§4.2, Fig. 1).** Win probability as a function
  of the mean log-likelihood of better completions is fitted better by a second-degree polynomial than by
  a line (RMSE lower by 24.42% at 7B and 25.78% at 35B).
- **Larger margins are not better (§4.2, Fig. 2, Fig. 3).** Larger log-likelihood differences between
  better and worse completions do not correspond to higher win probability; performance degrades for DPO
  and IPO after about 1,000 steps, and the best checkpoints lie in the middle of the heatmap rather than
  at the Pareto frontier of "highest chosen, lowest rejected".
- **Length is not the explanation (§4.2).** Pearson correlation between mean chosen log-likelihood and
  average output length: r = −0.114 (p = 0.266) at 7B and r = 0.198 (p = 0.173) at 35B.
- **NLL auxiliary term (§4.2, Fig. 4).** Adding an NLL loss on better completions with four values of λ
  and three values of β: when the NLL term has limited effect on the likelihood it has minimal effect on
  performance; the authors describe NLL as a tool to regulate completion likelihood that remains
  susceptible to likelihood over-optimisation.
- **Two stopping indicators (§1, §4.3).** Decreasing entropy over the top-k tokens, and a diminishing
  top-k probability mass, mark the point where output diversity has been over-optimised; the turning
  points coincide with the turning points of win probability.

## Conditions and limits
- Two models and two datasets, both instruction-tuned starting points; the authors name this as a
  limitation (§5, "Limitations").
