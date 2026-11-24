<!-- scope: ScaleRL (arXiv:2510.13786, Meta) — sigmoidal compute→pass-rate fits for RL; ablations of async scheme, loss, precision, aggregation, normalization, filtering, length, batch size, generations per prompt; 100K GPU-hour run
     deps: [[pipelinerl]], [[dapo]], [[grpo]]
     see-also: [[brorl]], [[skywork-or1]], [[polaris]], [[rollout-training-mismatch-tis]], [[rl-design-guide]]
-->

# The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL)
- **Core Insight:** Fitting held-out pass rate against RL compute with R_C − R_0 = (A − R_0) / (1 + (C_mid/C)^B) separates design choices that raise the asymptote A from those that only change efficiency B: loss type (CISPO/GSPO over DAPO), FP32 LM-head logits (A 0.52 → 0.61), longer generation (14K → 32K), larger batches and dropping prompts with pass rate ≥ 0.9 raise A, while PipelineRL over PPO-off-policy mainly raises B (§3, Figs. 4–6, 9, 10).
- **Guideline:** When choosing RL rollout and loss settings, compare candidates by fitting the sigmoid after about 1.5K GPU-hours and comparing A, not by early reward, because smaller batches and shorter generations looked better early but reached lower asymptotes (Figs. 9, 10), and the run-to-run error on A is about ±0.02 (Fig. 8a); with total batch fixed, do not spend effort tuning generations per prompt between 8 and 32, because those fitted curves were essentially unchanged (App. A.13).
- **Authors:** Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S. Dhillon, David Brandfonbrener, Rishabh Agarwal (Meta; UT Austin; UCL; Periodic Labs)
- **Year:** 2025 (arXiv v1, 15 Oct 2025)
- **URL:** https://arxiv.org/abs/2510.13786
- **Source type:** paper (ablation study)
- **Relevant topics:** RL scaling laws, off-policy asynchrony, loss functions, numerical precision, prompt filtering, generation length, batch size, group size

## Summary
The paper runs over a hundred RL configurations on an 8B dense model and a Llama-4 Scout 17B×16 MoE, fits a saturating curve per run, and assembles the best choices into ScaleRL. The 8B ScaleRL run was fitted on data up to 50K GPU-hours and extended to 100K GPU-hours (about 7,400 steps), and the extension tracked the extrapolated curve (Fig. 1). A power-law fit to 1.5K–50K GPU-hours predicted A = 1.0, while the sigmoid gave A = 0.645 against an observed saturation near 0.65.

## Key Contributions
- A compute-scaling fit with interpretable A, B, C_mid (Eq. 1, §2.1, App. A.4).
- Design-choice ablations with each choice classified by its effect on A vs B (§3).
- Leave-one-out validation of the final recipe at 16K GPU-hours (§4, Fig. 7).
- Scaling axes: model size, generation length, batch size, generations per prompt, multi-domain (§5).

## Key Figures/Tables to Study
- Fig. 1 (100K GPU-hour extrapolation), Fig. 4 (async), Fig. 5 (loss, FP32 head), Fig. 6 (zero-variance filtering, no-positive-resampling), Fig. 7 (leave-one-out), Figs. 9–10 (length, batch).

## Technical Details
**ScaleRL recipe (§4).** PipelineRL with max off-policyness k = 8; interruption-based length control (a forced end-of-thinking phrase); FP32 at the LM head; CISPO (truncated-IS REINFORCE) loss; prompt-level loss aggregation; batch-level advantage normalization; zero-variance filtering; No-Positive-Resampling (prompts with pass rate ≥ 0.9 are excluded from later epochs).

**Ablation effects.**

| Choice | Result | Location |
|---|---|---|
| PipelineRL vs PPO-off-policy-k | similar A, higher B | §3.1, Fig. 4a |
| Max off-policyness | k = 8 best | Fig. 4b |
| DAPO vs GSPO vs CISPO loss | GSPO, CISPO raise A above DAPO | §3.2, Fig. 5a |
| FP32 LM head | A 0.52 → 0.61 | Fig. 5b |
| Loss aggregation | prompt-average highest A | App. A.9, Fig. 14a |
| Advantage normalization | prompt / batch / none similar | Fig. 14b |
| Zero-variance filtering | higher A | Fig. 6a |
| No-Positive-Resampling | higher A | Fig. 6b |
| Leave-one-out: DAPO instead of CISPO | B 1.77 vs 2.01 | Fig. 7 |

**Scaling axes (§5).** Scout MoE reaches a higher A than 8B dense using 1/6 of its RL compute (Fig. 1). 14K → 32K generation length lowers B, raises C_mid and raises A (Fig. 9). Batch of 2048 prompts reaches higher A than smaller batches (Fig. 10). Generations per prompt 8–32 at fixed total batch: curves essentially unchanged (App. A.13). Math + code joint training gives parallel per-domain trends (Fig. 11).

## Verification
- Read arXiv HTML v1, first 100K of 135K characters (through App. A.5), on 2026-11-24. The length-penalty comparison referenced in §2 was not in the portion read.
- The abstract says aggregation and normalization mainly change B; the body reports A differences for prompt-average aggregation and zero-variance filtering. The card reports the body figures.
- Most leave-one-out variants reach similar A (Fig. 7); the recipe's advantage at 16K GPU-hours is mostly in B.
