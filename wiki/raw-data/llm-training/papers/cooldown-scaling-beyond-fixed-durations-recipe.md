<!-- scope: recipe ledger for Hägele et al. 2024 (constant LR + cooldown vs cosine) — small-scale SlimPajama runs, the 33M–360M scaling sweep, the 1B FineWeb runs (100B/460B tokens), the 8B FineWeb-Edu run, SWA, and the Chinchilla cost estimate
     deps: [[cooldown-scaling-beyond-fixed-durations]]
     see-also: [[lr-schedules]], [[deepseek-llm]]
-->

# Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations — Recipe ledger
- **Parent card:** [[cooldown-scaling-beyond-fixed-durations]] (`papers/cooldown-scaling-beyond-fixed-durations.md`)
- **Source:** arXiv:2405.18392v3 (2024-10-17), App. A.1–A.2 Tables 1–3, §3–§5, App. B.
- **Units:** in App. A.1 and Table 2, batch size is in tokens ("batch size of 200, i.e., roughly 0.1M tokens for a sequence length of 512"). Table 3 (1B, 8B) prints batch_size without a unit. "Cooldown %" is the fraction of total steps.

## Small-scale runs (SlimPajama-6B, nanoGPT extension)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-style decoder (SwiGLU, RoPE, RMSNorm) | 33M–360M | pretrain-stable | optimizer | AdamW, (β1, β2) = (0.9, 0.95), decoupled weight decay 0.1, grad clip 1.0 | App. A.1 | verified 2026-09-14 | no ablation reported; "standard practices" |
| same | 33M–360M | pretrain-stable | warmup | 300 steps for most runs; 1000–3000 for runs above 100k total steps; 300 for all scaling-law runs | App. A.1 | verified 2026-09-14 | no ablation reported |
| same | 33M–360M | pretrain-stable | batch; sequence length | 200 sequences, about 0.1M tokens; 512 | App. A.1 | verified 2026-09-14 | no ablation reported |
| same | 33M–360M | pretrain-stable | tokenizer; vocab | GPT-2 tokenizer; 50,304 | App. A.1 | verified 2026-09-14 | — |
| same | 33M–360M | pretrain-decay/anneal | cosine final LR | 10% of maximum LR | App. A.1 | verified 2026-09-14 | App. B.1 Fig. 22: decaying lower improves loss; §3.2 and Table 4: cosine to 0 lowers 1B downstream score |
| same | 33M–360M | data | dataset; validation | SlimPajama-6B subset; about 3M-token random validation set; 32 fixed batches during training | §2; App. A.1 | verified 2026-09-14 | — |
| same | 210M | pretrain-decay/anneal | cooldown in main comparison | linear to zero over 20% of steps; cosine lengths 22k, 33k, 44k steps | §3.2 Fig. 3 | verified 2026-09-14 | Fig. 3: best cosine and best cooldown nearly equal; Fig. 5: gains plateau near 20% (124M) |
| same | 124M | pretrain-decay/anneal | cooldown length sweep | 15k, 25k, 35k, 50k steps; LRs 5e-4, 1e-3, 2e-3 at 25k | §3.2 Fig. 5 | verified 2026-09-14 | cooldown passes cosine between 10% and 20% |
| same | not stated | pretrain-decay/anneal | long run | 200k steps (about 20B tokens); cooldown every 20k steps for 20%; 10k-step (5%) cooldown test | §3.2 Fig. 4, Fig. 6 | verified 2026-09-14 | (1-sqrt) beats linear more as training gets longer |
| same | 210M | pretrain-stable | SWA | window h = 500 steps; best windows ≤ 2,500 steps (256M tokens) | §4.1 | verified 2026-09-14 | EMA performed worse (§4.1, not shown) |
| same | 210M | pretrain-stable | schedule-free optimizer | (β1, β2) = (0.9, 0.95) and (0.95, 0.99); no further tuning | §4.2 Fig. 11 | verified 2026-09-14 | (0.95, 0.99) better; cooldown matches or beats both |

## Scaling-law sweep (App. A.1 Table 2; cooldown = 20% linear, constant LR = half cosine peak, §5)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| sweep model | 33M, 53M, 60M, 93M | pretrain-stable | LR (cosine, constant); batch | (2e-3, 1e-3); 0.1M tokens | Table 2 | verified 2026-09-14 | "adjust the LR to be higher for smaller models" (§5); no ablation reported |
| sweep model | 124M, 151M, 210M | pretrain-stable | LR (cosine, constant); batch | (1e-3, 5e-4); 0.1M tokens | Table 2 | verified 2026-09-14 | constant LR = half cosine peak follows Fig. 3 optimum |
| sweep model | 360M | pretrain-stable | LR (cosine, constant); batch | (1e-3, 5e-4); 0.2M tokens | Table 2 | verified 2026-09-14 | no ablation reported |
| sweep model | 33M | pretrain | steps; tokens; tokens/param | [3k, 7k, 10k]; [0.3B, 0.7B, 1.0B]; [9.2, 21.4, 30.6] | Table 2 | verified 2026-09-14 | three token counts near D/N = 20 (§5) |
| sweep model | 124M | pretrain | steps; tokens; tokens/param | [15K, 25K, 35K]; [1.5B, 2.6B, 3.6B]; [12.4, 20.7, 29.0] | Table 2 | verified 2026-09-14 | same |
| sweep model | 360M | pretrain | steps; tokens; tokens/param | [25K, 37.5K, 50K]; [5.1B, 7.7B, 10.2B]; [14.2, 21.3, 28.5] | Table 2 | verified 2026-09-14 | same |

Rows for 53M, 60M, 93M, 151M, and 210M step and token lists are in Table 2 and are not repeated here.

## 1B and 8B runs (nanotron; App. A.2 Table 3)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| 1B (d_model 1792, 24 layers, 14 heads) | 1B | pretrain | data; tokens | FineWeb; 100B and 460B tokens | §3.3; App. A.2 | verified 2026-09-14 | — |
| 1B | 1B | pretrain | peak LR | 8e-4 | §3.3; Table 3 | verified 2026-09-14 | estimated from DeepSeek scaling laws for 100B tokens (Table 3 caption) |
| 1B | 1B | pretrain | batch size (unit not printed) | §3.3 text: 1.8M for the 100B and 460B runs; Table 3: (1.8M, 2M) for (100B, 460B) | §3.3 vs Table 3 | conflict | Table 3 caption: batch size and LR estimated with DeepSeek scaling laws for 100B tokens; no ablation reported |
| 1B | 1B | pretrain | steps; warmup; sequence length; vocab | (55k, 220k); 2000; 2048; 49,152 | Table 3 | verified 2026-09-14 | — |
| 1B, 100B tokens | 1B | pretrain-decay/anneal | schedules compared | cosine to 10%, cosine to 0, (1-sqrt) 20%, linear 20% | Table 4 | verified 2026-09-14 | aggregate 46.26 / 45.88 / 46.23 / 46.20 |
| 1B, 460B tokens | 1B | pretrain-decay/anneal | schedules compared | cosine to 0, (1-sqrt) 5%, linear 5%, 10%, 20% | Table 5 | verified 2026-09-14 | aggregate 48.03 / 47.91 / 47.84 / 47.98 / 47.92 |
| 8B (Llama 3 architecture) | 8B | pretrain | data; tokens; steps | FineWeb-Edu; 12B tokens; 20k steps | §3.3 Fig. 9; Table 3 | verified 2026-09-14 | — |
| 8B | 8B | pretrain | peak LR; batch; warmup; seq len | 3e-4; 0.6M (unit not printed); 1000; 4096 | Table 3 | verified 2026-09-14 | batch set by GPU limits (Table 3 caption) |
| 8B | 8B | pretrain-decay/anneal | schedules compared | cosine versus (1-sqrt) 20% | Fig. 9 | verified 2026-09-14 | matching loss, no instability observed |
| 1B | 1B | eval-gate | evaluation | lighteval; benchmarks truncated to 1000 samples; acc_norm | App. A.2 | verified 2026-09-14 | — |

## Chinchilla cost estimate (§5, Fig. 13b)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Chinchilla suite (Hoffmann et al. Table A9 configs) | suite | pretrain | assumed sequence length; batch; ratios; cooldown | 1024; 0.5M tokens; D/N ∈ {10, 15, 20, 25}; 10% cooldowns | §5 | verified 2026-09-14 | assumptions, since Chinchilla does not report exact configs |
| same | suite | pretrain | total FLOPs, cosine vs cooldown | 5.59e23 vs 2.36e23 | §5 Fig. 13b | verified 2026-09-14 | authors' estimate from the assumptions in the row above; "less than half the compute" (§5) |

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2405.18392 (arXiv v3).
- Not reported: seeds, run-to-run variance, compute per 1B or 8B run, released checkpoints.
