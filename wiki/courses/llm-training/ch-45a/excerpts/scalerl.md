---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2510.13786
primary_version: arXiv:2510.13786v1 (15 Oct 2025)
created_at: "2026-09-15"
---

# Excerpt: ScaleRL — the sigmoid compute-performance fit and what moves A versus B

Khatri, Madaan, Tiwari, Bansal, Duvvuri, Zaheer et al. (Meta, UT Austin, UCL, UC Berkeley, Harvard, Periodic Labs), 2025. ch-45a uses this source to say which ledger
entries are worth copying and which are only worth copying together with their compute budget.

## The fit (Eq. 1)
> R_C − R_0 = (A − R_0) × 1 / (1 + (C_mid / C)^B)

- `R_C` — expected reward (pass rate, mean@16) on an i.i.d. validation set after RL compute `C`.
- `R_0` — the starting checkpoint's pass rate before RL.
- `A` — asymptotic pass rate, 0 ≤ A ≤ 1.
- `B` — scaling exponent; larger B means the curve rises through its mid-range faster, i.e. better compute
  efficiency.
- `C_mid` — the compute at the midpoint of the curve.
- `C` — RL training compute in GPU-hours, with model and training data held fixed.

Fits start after about 1.5k GPU-hours because the earliest regime is unstable (§2.1). Three independent ScaleRL
runs give an error margin of ±0.02 on A (Fig. 8a). On the 100k-GPU-hour 8B run, sigmoid fits over (1.5k, 50k),
(0, 100k), (0, 50k) and (5k, 50k) give A between 0.645 and 0.655, while a power-law fit over (5k, 50k) gives
A = 0.74 against 0.645 for the sigmoid over the same window (App. A.4, A.7).

## Scale of the study
"more than 400,000 GPU-hours" on Nvidia GB200 GPUs, at 8B dense scale, with individual ablation runs up to
16,000 GPU-hours; plus one 100,000-GPU-hour ScaleRL run on the 8B dense model and a 50,000-GPU-hour run on a
Llama-4 Scout 17B×16 MoE (Abstract, §1, §5).

## What changes the asymptote and what changes efficiency
> "(2) Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate
> compute efficiency without materially shifting the asymptote" (Abstract).

> "Common interventions thought to improve peak performance (e.g., loss aggregation, data curriculum, length
> penalty, advantage normalization) mainly adjust compute efficiency (B), while not changing the performance
> ceiling considerably" (§1).

Two axes did move A in this setup: the loss type (CISPO and GSPO above DAPO, Fig. 5a) and FP32 precision at the
LM head, which "improves the asymptotic performance A from 0.52 to 0.61" (§3.2, Fig. 5b). Batch size also moved
it: "larger batch size is slower in training but settles at a higher asymptote" (Fig. 10), with fitted values
A = 0.605 at batch 512 and A = 0.645 at batch 2048 (Table 1).

Table 1 (large-scale runs): ScaleRL-Scout C_mid 4242, B 1.65, A 0.710; ScaleRL-bs512 C_mid 2818, B 1.77,
A 0.605; ScaleRL-bs2048 C_mid 10909, B 1.70, A 0.645; ScaleRL math+code, math curve 2896 / 2.05 / 0.595;
code curve 1675 / 1.09 / 0.615.

Leave-one-out at 16k GPU-hours (§4, Fig. 7): reverting CISPO to DAPO inside ScaleRL leaves A about the same and
lowers efficiency, B = 2.01 (CISPO) versus B = 1.77 (DAPO).

## The comparison to published recipes is a re-implementation
§A.16 states that the "DeepSeek (GRPO)", "Qwen2.5 (DAPO)", "Magistral" and "MiniMax" curves in Fig. 2 are the
authors' reconstructions of those recipes inside their own codebase on an 8B dense model, with deliberate
deviations: the DAPO reconstruction uses ε_max = 0.26 and, instead of refilling the batch after dropping
zero-variance prompts, keeps a larger batch of 1280 (80 prompts × 16 generations) and drops them; the MiniMax
reconstruction is given the same larger batch. ScaleRL reaches an asymptotic reward A = 0.61 in that comparison
(Fig. 2 caption). These numbers therefore compare recipe *shapes* under one codebase and one model, not the
released checkpoints of those model families.

## Training regimen of the study
"training uses a sequence length of 16,384 tokens: 12,288 for thinking, 2,048 for the solution" (§2); generation
length is controlled by forced interruption (appending an end-of-thinking phrase) rather than a length penalty;
the ScaleRL recipe is PipelineRL-style asynchrony, forced length interruption, CISPO loss, prompt-level loss
averaging, batch-level advantage normalization, FP32 logits, zero-variance filtering, and no-positive-resampling
(§1, §4).

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2510.13786 (scratchpad `sources/scalerl.txt`),
  Abstract, §1, §2, §2.1, §3.1, §3.2, §4, §5, App. A.4, A.7, A.15, A.16, A.17, Table 1.
- Not reported here: the identity of the 8B dense base model; learning rate and KL settings of the ScaleRL runs.
