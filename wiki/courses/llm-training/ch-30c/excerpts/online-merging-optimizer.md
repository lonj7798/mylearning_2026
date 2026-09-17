---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "Online Merging Optimizers for Boosting Rewards and Mitigating Tax in Alignment (Lu et al., Qwen Team)"
source_url: https://arxiv.org/abs/2405.17931
created_at: "2026-09-15"
---

# Excerpt: Online Merging Optimizer (OnDARE, OnTIES)

This excerpt stands in for the library card `online-merging-optimizer`, which did not exist when ch-30c was written.
Every number below was read in arXiv:2405.17931v1 at the stated locus.

- **Authors:** Keming Lu, Bowen Yu, Fei Huang, Yang Fan, Runji Lin, Chang Zhou (Qwen Team, Alibaba)
- **Year:** arXiv v1 2024-05-28 ("Preprint. Under review.")
- **Source type:** paper. Code: github.com/QwenLM/online_merging_optimizers

## Mechanism (§3-4)
- Alignment tax: loss of abilities from pre-training and SFT during RLHF (§1).
- Offline merge of an RLHF model with its SFT reference restores benchmark scores but lowers preference scores (§1, §3).
- Online merge at step t (Eq. 1): θ^(t+1) = θ_b + F(τ^(t) + Δθ^(t)) ⊕ F(τ_r), with θ_b the pre-trained base,
  τ_r = θ_r − θ_b the SFT reference's delta, F a sparsification operator, ⊕ a consensus rule. Eq. 1 was unstable,
  so the paper relaxes it to θ^(t+1) ≈ θ^(t) + F(Δθ^(t)) ⊕ F(τ_r) (Eq. 2).
- OnDARE (Eq. 3): θ_m^(t) = θ_m^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r), with F_R a Bernoulli keep mask with reserve rate p
  (Eq. 4). OnTIES (Eqs. 5-7) uses top-p magnitude sparsification and a sign-based consensus.
- Δθ is the Adam update −η·m^(t)/(√v^(t) + ε) (Alg. 1). Rescaling of sparsified deltas is omitted because it "harms
  the numeric stabilities in multi-step optimization" (§4.2).

## Setup (§5.1, App. C)
- Off-policy DPO on UltraFeedback binarized (about 61K train pairs); DPO β = 0.1 for AdamW and the online optimizers.
- Reference SFT models: Qwen1.5-1.8B-SFT, Qwen1.5-7B-SFT, LLaMa-3-8B-it; batch size 128; LR grids in App. C.
- Offline merge baselines (mergekit): densities {0.3, 0.5, 0.7}, weights {0.1, 0.3, 0.5, 0.7, 0.9}.
- Evaluation: 12 benchmarks in 7 categories (math, code, instruction following, reading comprehension, knowledge,
  agent function calling, code-switching), plus MT-Bench and AlpacaEval 2.0 LC. MT-Bench standard deviation about 0.05;
  AlpacaEval LC standard deviation about 0.5 (App. B).
- Compute: 64 H800 GPUs, about 3 hours per training run (App. C).

## Results (Table 1; change vs AdamW DPO: benchmark average / MT-Bench / AlpacaEval 2.0 LC)
| Backbone | Linear merge | DARE merge | TIES merge | OnTIES | OnDARE |
|---|---|---|---|---|---|
| Qwen1.5-1.8B-Chat | +0.2 / −0.03 / −0.96 | +0.2 / +0.09 / −0.72 | −0.1 / −0.10 / −0.41 | +0.3 / +0.15 / +0.12 | +0.5 / +0.24 / +0.05 |
| Qwen1.5-7B-Chat | −0.2 / −0.28 / −1.65 | +0.2 / −0.13 / −1.59 | +0.6 / −0.21 / −1.35 | +0.5 / −0.08 / +0.91 | +1.1 / +0.12 / +0.28 |
| LLaMa-3-8B-Instruct | −1.2 / +0.11 / +0.02 | +0.5 / +0.14 / +0.18 | 0.0 / +0.01 / +0.21 | 0.0 / +0.16 / +0.84 | +1.3 / +0.19 / +1.57 |

- AdamW DPO vs SFT reference, benchmark average: 41.8 vs 41.8 (1.8B), 58.4 vs 57.3 (7B), 58.0 vs 57.7 (8B) (Table 1).
- Reserve rate p: stable down to 5e−4; OnTIES is more sensitive at very low p (§5.3, Fig. 2).
- Merging weight α (Table 2, OnDARE; the table's values match the 1.8B scale): benchmark average 7.4 / 37.2 / 40.1 /
  41.8 / 41.6 / 40.1 / 39.2 and MT-Bench 1.00 / 3.13 / 4.37 / 4.66 / 4.76 / 4.79 / 4.85 for α = 1e−4 / 5e−5 / 1e−5 /
  5e−6 / 1e−6 / 5e−7 / 1e−7. The authors suggest starting the α search near 1e−7 (§5.3).
- Step-K online merging (merge every K steps): as K grows over {5, 10, 50, 100, 200}, MT-Bench falls toward AdamW; the
  benchmark average falls less (§6.1, Fig. 3).
- KL interaction (Qwen1.5-7B-Chat): at DPO β = 1e−3, OnTIES keeps MT-Bench above 6.9 (the SFT reference level), while
  AdamW falls to 6.25 (§6.2, Fig. 4).
- IPO and KTO (Qwen1.5-7B-Chat, Table 3): MT-Bench AdamW / OnDARE / OnTIES = 7.06 / 7.09 / 7.34 (IPO) and 7.18 / 7.46 / 7.42 (KTO).

## Discrepancies found while reading
- §5.3 text says benchmark averages peak at α = 5e−7 and that MT-Bench rises as α increases; Table 2 shows the peak
  at 5e−6 (41.8) and MT-Bench rising as α decreases.
- §3 and §5.2 refer to "Tab. 4" for the main results; the main results are in Table 1.

## Limits (Limitations)
- Extra memory for the cached reference delta; not applicable to LoRA training unless the reference is also a LoRA model.

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2405.17931v1 (PDF), §1-§7 and App. A-C.
