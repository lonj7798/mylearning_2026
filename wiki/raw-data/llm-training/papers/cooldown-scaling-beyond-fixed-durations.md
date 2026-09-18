<!-- scope: constant learning rate plus short cooldown (WSD) versus cosine for LLM pretraining; cooldown shape and length ablations; stochastic weight averaging; cheaper scaling-law sweeps from reusable runs
     deps: [[chinchilla-compute-optimal]], [[lr-schedules]]
     see-also: [[minicpm]], [[deepseek-llm]], [[continual-pretraining-rewarm-replay]], [[domain-upsampling-end-of-training]], [[model-soups]]
-->

# Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations
- **Core Insight:** A constant learning rate followed by a cooldown of about 20% of steps matches a length-matched cosine schedule in validation perplexity from 33M to 360M parameters and in downstream scores for a 1B model on 100B tokens (aggregate 46.20 linear-20% versus 46.26 cosine-to-10%), so one constant-LR run per model size can replace several cosine runs in a scaling sweep (§3.2 Fig. 3; §3.3 Table 4; §5 Fig. 12).
- **Guideline:** When the final token budget is not fixed in advance, or when a scaling-law sweep needs several token counts per model size, use a constant LR with a (1-sqrt) cooldown and branch cooldowns from saved checkpoints, because in these experiments it closely matched cosine at the tested training lengths and model sizes (§3.2 Fig. 3; §5 Fig. 12) and (1-sqrt) beat linear cooldown (§3.2 Fig. 4); when a separate decay phase cannot be run, stochastic weight averaging on the constant-LR run gives lower loss than the raw checkpoint but does not reach the cooldown loss (§4.1 Fig. 10).
- **Authors:** Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, Martin Jaggi (EPFL; Hugging Face)
- **Year:** 2024 (arXiv v1 2024-05; NeurIPS 2024 spotlight)
- **URL:** https://arxiv.org/abs/2405.18392 (code: https://github.com/epfml/schedules-and-scaling/)
- **Source type:** paper
- **Relevant topics:** learning-rate schedules, WSD, cooldown/annealing, stochastic weight averaging, schedule-free optimizer, scaling-law methodology, continual pretraining

## Abstract
The authors argue that scaling and training research has been more expensive than needed because the cosine schedule must match the training length, so each token count needs its own run. They study a constant learning rate followed by a cooldown and find that it scales as predictably as cosine. Stochastic weight averaging (SWA) improves models along the training trajectory without extra training cost. With these results, scaling experiments can use fewer training runs that are reused for several token counts, which reduces compute and GPU hours. Code is released.

## Key Contributions
- Shows that a length-matched cosine schedule gives the best loss only at its own endpoint, and that a constant LR plus cooldown matches it at 22k, 33k, and 44k steps for a 210M model (§2 Fig. 1; §3.2 Fig. 3).
- Introduces the (1-sqrt) cooldown shape, which outperforms linear, cosine, mirror-cosine, and (1-square) cooldowns in the ablations (§3.2 Fig. 4; App. B.1 Fig. 16–18).
- Measures cooldown length: gains plateau at about 20% of steps, and 10k steps (5%) nearly match cosine on a 200k-step run (§3.2 Fig. 5–6).
- Tests SWA and the schedule-free optimizer (SFO) as replacements for the cooldown; both fall short of an explicit cooldown (§4 Fig. 10–11).
- Reproduces a small scaling-law sweep (33M–360M, 0.3B–10B tokens) with one run per size, and estimates the saving for the Chinchilla suite (§5 Fig. 12–13).

## Key Figures/Tables to Study
- Fig. 1: cosine schedules of different lengths; each is best only at its own endpoint.
- Fig. 3: loss curves and LR sensitivity, cosine versus 20% cooldown (210M).
- Fig. 5–6 and App. Fig. 19–20: cooldown length as fraction and in absolute steps.
- Fig. 7: loss along linear interpolation between pre- and post-cooldown weights.
- Fig. 12: perplexity of cosine versus cooldown and versus SWA for every model in the sweep.
- Tables 4–5 (App. B.5): 1B downstream scores at 100B and 460B tokens.

## Technical Details
- **Schedule.** `η(n) = (n / N_warmup)·η_max` for `n < N_warmup`; `η_max` for `N_warmup < n ≤ N − N_decay`; `f(n, N, N_decay)·η_max` for `n > N − N_decay` (§3 Eq. 1). `η_max` is the peak LR, `N` total steps, `N_warmup` and `N_decay` warmup and cooldown steps, `f` a decreasing function. The authors call this "constant LR + cooldown"; it was earlier called trapezoidal (Zhai et al.) and WSD (Hu et al., MiniCPM) (§3).
- **(1-sqrt).** `f = 1 − sqrt((n − (N − N_decay)) / N_decay)` (§3.2). Written as `1 − x^a` with a = 0.5; a = 0.1 and 0.2 are worse, a = 0.3 and 0.4 differ marginally, and 0.5 is best (App. B.1 Fig. 18).
- **Learning rate.** LR sensitivity is similar for both schedules and slightly lower for cooldown; the cooldown optimum is at about half the optimal cosine peak LR (Fig. 3 caption). The sweep sets the constant LR to half the cosine maximum (§5).
- **Cooldown length.** For a 124M model, cooldown passes cosine between 10% and 20% of steps and largely stops improving beyond that (Fig. 5). On a 200k-step (about 20B tokens) run, 10k (1-sqrt) cooldown steps nearly match cosine (§3.2 Fig. 6).
- **Cosine final LR.** Decaying cosine to about zero instead of 10% lowers loss to the (1-sqrt) level (App. Fig. 22) but lowers the 1B aggregate downstream score from 46.26 to 45.88 at 100B tokens; the authors attribute this to early saturation and recommend setting peak and final LR independently (§3.2; Table 4).
- **Loss landscape.** Loss along a linear interpolation between checkpoints before and after cooldown drops smoothly, which the authors interpret as the model moving into a connected basin (§3.2 Fig. 7, Interpretation).
- **SWA.** Window h = 500 steps with window averages saved every 500 steps; windows ≤ 2,500 steps (256M tokens) are best; EMA was worse and not reported (§4.1).
- **SFO.** With (β1, β2) = (0.9, 0.95) SFO is worse and its loss rises near the end; with (0.95, 0.99) it performs well; cooldown matches or beats both (§4.2 Fig. 11).
- **Sweep fit.** Cooldown versus cosine perplexity fits `y = 0.99x + 0.23`; SWA-constant versus cosine fits `y = 0.99x − 0.30` (Fig. 12 panels).
- **Compute saving.** The §5 text reports that the 10/20/30 tokens-per-parameter sweep saves half the time and FLOPs; the Fig. 24 caption states "a factor of 12". For Chinchilla, with assumed sequence length 1024, 0.5M-token batches, and ratios {10, 15, 20, 25}, the estimate is 5.59e23 FLOPs with cosine versus 2.36e23 with 10% cooldowns (§5 Fig. 13b).
- **Total project compute.** About 2,500–3,000 GPU hours including prototyping (App. A.1).

## Recipe ledger
Full table (small-scale, 1B, and 8B settings): [[cooldown-scaling-beyond-fixed-durations-recipe]].

## Findings relevant to generality
- **Downstream agreement (Result, single study).** 1B model on FineWeb, 100B tokens: aggregate score 46.26 (cosine to 10%), 45.88 (cosine to 0), 46.23 ((1-sqrt) 20%), 46.20 (linear 20%) over MMLU, ARC, OpenBookQA, PIQA, HellaSwag, CommonSenseQA, SIQA, Winogrande (App. B.5 Table 4; A.2).
- **Longer run.** At 460B tokens: 48.03 (cosine to 0), 47.91 ((1-sqrt) 5%), 47.84 (linear 5%), 47.98 (linear 10%), 47.92 (linear 20%); longer cooldowns do not necessarily improve metrics (Table 5).
- **Per-benchmark response.** Some benchmarks rise at the start of cooldown (MMLU, HellaSwag); others show no comparable rise (OpenBookQA); the authors leave the cause open (App. B.5 Fig. 29–30).
- **Transfer across data.** Cooldown and SWA findings repeat on OpenWebText2 with 60M, 93M, and 166M models (App. B.4 Fig. 27).
- **Continual training.** The authors argue that pre-cooldown checkpoints can continue at high LR and avoid rewarming, which they show causes loss spikes for cosine (§2; §3.1 Fig. 1); no continual-training experiment with a new data distribution is reported.
- **Data mixture in cooldown.** Changing the mixture during cooldown is discussed as an option (§3.1) but all experiments keep the same mixture.
- **Limits.** Up to 8B parameters, and the 8B run is 12B tokens (20k steps); behavior at larger scale and much longer training may differ (§3.3; §6).

## Connections
- [[chinchilla-compute-optimal]] — the result that cosine length must match training length (§1–2).
- [[lr-schedules]] — library card on warmup, cosine, and WSD.
- [[minicpm]] — Hu et al., the WSD name and the cooldown analysis cited in §3.
- [[deepseek-llm]] — Bi et al., batch-size and LR scaling laws used for the 1B run (§3.3).
- [[continual-pretraining-rewarm-replay]] — Ibrahim et al., rewarming and forgetting cited in §2–3.1.
- [[domain-upsampling-end-of-training]] — Blakeney et al., concurrent work on data changes during the decay phase (§3.1).
- [[model-soups]] — weight averaging cited for noise reduction (§4.1).
- [[small-scale-proxies-instabilities]] — Wortsman et al., QK-norm cited for high-LR instabilities (§3.3, §6).
- [[resolving-scaling-discrepancies]] — Porian et al., concurrent result on constant-LR estimation of the optimal ratio (§7).
- [[beyond-chinchilla-inference-scaling]], [[overtraining-downstream-scaling]] — cited for training past compute-optimality (§7).
- [[kaplan-scaling-laws]] — the power-law form used in §5.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2405.18392 (arXiv v3, 2024-10-17; full PDF text including App. A–B). v1 date and NeurIPS 2024 spotlight comment checked on the arXiv abs page.
- Audit claims not found in the source: "Evidence base for the SmolLM2/3 and Apertus WSD choices" — this paper does not mention SmolLM2, SmolLM3, or Apertus; any citation of it must be checked in those reports.
- Not reported by the source: seeds or variance across repeated runs; model size of the 200k-step run in Fig. 4 and Fig. 6; batch-size unit for the 1B and 8B runs.
