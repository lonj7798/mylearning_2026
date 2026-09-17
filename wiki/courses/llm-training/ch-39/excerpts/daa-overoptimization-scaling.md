---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2406.02900
created_at: "2026-09-15"
verified_against: "arXiv:2406.02900v2 (5 Nov 2024), cached plain text"
---

# Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms

Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, W. Bradley Knox, Chelsea Finn,
Scott Niekum.

## Setting (App., "Experimental details")
Reddit TL;DR with 92K preference pairs; Pythia 1B, 2.8B and 6.9B, SFT on TL;DR first; DPO, IPO and SLiC
objectives; seven values of the regularisation coefficient; batch size 128, RMSProp, LR 5e-7 with 150
warmup steps, 1 epoch; evaluation on 256 held-out prompts at temperature 1.0 with
gpt-4-turbo-2024-04-09 as judge.

## Claims the chapter uses
- **Over-optimisation exists without an external reward model (§3.1, Fig. 1).** Win rate against the
  dataset reference summary is hump-shaped in the achieved KL budget for all three objectives: past a
  point, more KL gives lower win rate.
- **Within one epoch (§3.1, Fig. 2; App. Fig. 8).** Under the wider KL budgets, runs reach their best win
  rate after about 25% of a single epoch and then decline while KL keeps increasing; the same pattern
  holds for the 1B and 2.8B models.
- **Objective and size (§3.1).** IPO reaches lower KL under the same constraint and shows less
  over-optimisation than DPO and SLiC; the 1B model reaches much higher KL and degrades almost
  immediately, the 6.9B model degrades least.
- **Training statistics do not predict quality (§3.2, Fig. 4).** Within a model size there is no
  discernible relationship between the DAA implicit-reward classification accuracy, or the training loss,
  and the win rate.
- **Falling chosen likelihood is expected (§3.5, Eq. 7).** When `π_ref` is the SFT model trained on the
  preferred responses, `E_{p_D(y_w|x)}[log π_θ(y_w|x)/π_ref(y_w|x)] ≈ −D_KL(π_ref ‖ π_θ)`, which is
  non-positive and decreases as the policy moves away from the reference.
- **Length is not the only over-optimised feature (§3.3, Fig. 3).** With the length-regularised objective
  of Park et al., over-optimisation still occurs; smaller models and smaller KL budgets extrapolate more
  strongly on the length feature.
