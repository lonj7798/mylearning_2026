---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/olmix.md (planned card; not present on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2602.12237
primary_version: arXiv:2602.12237v1 (2026-02-12); related official blog https://allenai.org/blog/olmix (not read for ch-13)
created_at: "2026-09-15"
---

# Excerpt: Olmix: A Framework for Data Mixing Throughout LM Development

Authors: Mayee F. Chen, Tyler Murray, David Heineman, Matt Jordan, Hannaneh Hajishirzi, Christopher Ré, et al. (Allen Institute for AI; Stanford University; University of Washington). Source type: paper. Read in the v1 PDF text on 2026-09-15 for ch-13 `read.md` §1, §3, §4, and Recipe.

## Main claim (Abstract)
> "Over a sequence of five domain-set updates mirroring real-world LM development, mixture reuse matches the performance of fully recomputing the mix after each update with 74% less compute and improves over training without mixing by 11.6% on downstream tasks."

## Problem setup (§3.1)
A mixture p ∈ Δ^{m−1} over domains D_1..D_m of N_i tokens; "training on R total tokens uses a dataset with p_i · R tokens from D_i". Performance is bits per byte (BPB) of the gold answer; the goal is to minimize average BPB over n tasks. Offline schema: (1) swarm of K proxy models, (2) regression from mixture to performance, (3) optimization of the predicted performance.

## Experimental setup (§3.3.1)
DCLM partitioned into 24 WebOrganizer topics; 1B target models with the OLMo 2 architecture trained to 100B tokens (5x Chinchilla); batch 512, sequence length 4096, max LR 0.0018; 52 downstream tasks (math, code, commonsense QA) scored in BPB.

## Findings used in ch-13
- RQ1 proxy size (Figure 3): "proxy models with ≥ 15M parameters achieve strong rank correlation (ρ > 89). However, 1M models show significantly degraded correlation (r = 73)"; "we use 30M parameter proxy models with 3B tokens (5x Chinchilla), which achieves a Spearman correlation of 89.6 with 1B target models." Footnote 1: RegMix's public "1M implementation is closer to 15M".
- RQ2 swarm size (Figure 4): sample complexity is linear in m; "We recommend using at least K ≥ 3(m + 1) proxy runs with the log-linear regression model".
- RQ3 swarm distribution (Table 2): Dirichlet priors natural / strong / weak gave average BPB 0.765 / 0.763 / 0.797.
- RQ4 regression family (Figure 6): log-linear model f̂_i(p) = c_i + exp(A_iᵀ p) (adapted from Data Mixing Laws) had the best fit overall (ρ = 80 at K = 118) and the best downstream BPB at K = 128; LightGBM "requires more than 118 proxy runs for sufficient fit".
- RQ5 granularity (Table 3): per-task 0.765 BPB (fit 0.983); per-family 0.777 (0.958); aggregated 0.774 (0.866).
- RQ6 repetition constraints p_j ≤ k N_j / R (Table 4, k = 4): constrained swarm + unconstrained optimization: does not satisfy (repeats 5), 0.774694; unconstrained swarm + constrained optimization: satisfies, 0.764718; constrained swarm + constrained optimization: satisfies, 0.785517. Example in §3.3.4: "a mixture that proposes 40% code when code comprises only 5% of available data would oversample code 8 times."
- RQ7 solver (Figure 8): exact solver 0.7772; exact + KL(0.01) 0.7806; exact + KL(0.05) 0.7647; search 0.7938. "We recommend using an exact solver with a KL regularization of 0.05".

## OlmixBase (Algorithm 1)
Sample K = O(m) mixes; train proxies of at least 15M parameters; fit one log-linear model per task; solve min_p (1/n) Σ_i f̂_i(p) + λ KL(p ‖ p_0) subject to p_j ≤ k N_j / R, with p_0 the natural distribution. "the mixing method we used throughout Olmo 3 development" (§3).

## Evolving domains (§4-§5)
Mixture reuse freezes relative ratios of unaffected domains as one virtual domain and recomputes only affected ones. Worked example (§4.2): p̃ = [0.25, 0.25, 0.5], collapsed mix [0.4, 0.6] expands to [0.1, 0.1, 0.2, 0.6]. Five updates (add Stack-Edu, add six sources, revise PDFs, remove AlgebraicStack, partition PDFs) end at 64 domains; k = 4, R = 1T (§5.1.1). Results (§5.1.2): FullMixtureReuse +11.6% vs full recomputation +12.2% with 216 vs 832 proxy runs; PartialMixtureReuse +12.0% with 272 runs; swarm reuse +11.4% with 268 runs. PartialMixtureReuse "reaches the natural distribution's final BPB in approximately 20,000 steps versus 61,000 steps—a 3.05× speedup."

## Limits named by the source (§6)
Offline schema only; theory assumes log-linear regression; no automated rule for choosing which unaffected domains to recompute.
