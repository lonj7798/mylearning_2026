<!-- scope: Recipe ledger for the Hunyuan-Large technical report (arXiv:2411.02265v3) — pre-training, annealing, long-context, SFT, and DPO settings as printed, with released context lengths from the official repository
     deps: [[hunyuan-large]]
     see-also: [[deepseek-v3-recipe]], [[nemotron-4-synthetic-recipe]]
-->

# Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent — Recipe ledger
- **Parent card:** [[hunyuan-large]] (full source card for the same report).
- **Source:** arXiv:2411.02265v3 (2024-11-06). Released context lengths from github.com/Tencent/Tencent-Hunyuan-Large `README.md` (main branch, read 2026-09-14).
- **Checked on:** 2026-09-14. The report has no appendix. Loci are sections, equations, and tables of arXiv v3.
- **Scope of checkpoints:** the report calls the post-trained model "Hunyuan-Large-Instruct" (§4.2); the repository releases it as Hunyuan-A52B-Instruct and the pre-trained model as Hunyuan-A52B-Pretrain (README model table). The README records an Instruct update on 2024-11-18, after arXiv v3; the report does not describe how the updated checkpoint was trained.
- **Units:** the report does not state the unit of "more than 1 million" SFT data, and does not state whether the 25% / 75% long-context mixture is a token share or a sampling weight. Values are recorded as printed.
- **Size column:** 389B total / 52B activated parameters for every row (Abstract, Table 1).

## Ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Hunyuan-Large pre-trained (Hunyuan-A52B-Pretrain) | 389B / 52B | pretrain-stable | tokens trained | 7T ("approximately 7T" in §2.3.1) | arXiv:2411.02265v3 Table 1, §1, §2.3.1 | verified 2026-09-14 | isoFLOP fit D_opt = D_c·C_min^β with D_c = 3.2, β = 0.50 gave ≈5.6T; 7T chosen citing "the same principle of smooth curves" and maximal use of training data "within the optimal cost-performance range" (§2.3.1, Fig. 4); no downstream ablation |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | synthetic data | nearly 1.5T of the 7T tokens; focus on mathematics, coding, low-resource, and high-educational-value fields | §1, §2.1.1 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | activated parameters | 52B | Abstract, Table 1, §2.3.1 | verified 2026-09-14 | fit N_opt = N_c·C_min^α with N_c = 5.9×10⁻³, α = 0.5305 gave ≈58.1B; 52B chosen "due to the smoothness of the quadratic curve around the optimal value" (§2.3.1, Fig. 3) |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | architecture | 64 layers; 80 attention heads; 8 KV heads; hidden size 6,400; SwiGLU; RoPE; 1 shared expert + 16 specialized experts, top-1 specialized expert per token; recycle routing for tokens of overloaded experts | Table 1, §2.2.1, §2.2.3 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | KV cache | GQA with 8 KV-head groups + Cross-Layer Attention sharing KV every 2 layers; "nearly 95%" KV-cache saving vs MHA | §2.2.2, Table 2 | verified 2026-09-14 | memory formulas only (Table 2); no quality ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | tokenizer | 128K vocabulary = 100K tiktoken tokens + 28K Chinese tokens | §2.1.2 | verified 2026-09-14 | compression 2.78 → 3.13 characters per token vs the LLama3.1 tokenizer (§2.1.2) |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | optimizer | AdamW | §2.2.4 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | expert-specific LR | shared expert uses ϵ_opt(B); specialized experts use ϵ_opt(B/n) with n = 16; ratio ϵ_opt(B)/ϵ_opt(B/n) ≈ 0.31 | §2.2.4, Eq. 1 | verified 2026-09-14 | derived from the Li et al. (2024a) batch-size/LR relation; no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | LR schedule shape | warmup → prolonged gradual decay → short anneal | §2.3.2 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-stable | peak LR; warmup length; decay shape; betas; weight decay; grad clip; global batch; sequence length of the main phase; compute | not given | §2 (body) and GitHub README checked | not reported | — |
| Hunyuan-Large pre-trained | 389B / 52B | pretrain-decay/anneal | anneal | final 5% of pre-training tokens; LR reduced to one-tenth of its peak; highest-quality data available | §2.3.2 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | long-context | stages | 32K → 256K, after the anneal | §2.3.3 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | long-context | tokens per stage | approximately 10 billion tokens in each of the 32K and 256K stages | §2.3.3 | verified 2026-09-14 | report states each stage reaches "satisfactory long-context abilities" while keeping normal-length ability; no numbers (§2.3.3) |
| Hunyuan-Large pre-trained | 389B / 52B | long-context | mixture (unit not stated) | natural long-context data from books and code "nearly 25%"; normal-length pre-training data "nearly 75%" | §2.3.3 | verified 2026-09-14 | report cites similar conclusions in Gao et al. (2024); no ablation in this report |
| Hunyuan-Large pre-trained | 389B / 52B | long-context | RoPE base | 1 billion during the 256K stage | §2.3.3 | verified 2026-09-14 | report states the choice was "inspired by Xiong et al. (2023)"; no ablation reported |
| Hunyuan-Large pre-trained | 389B / 52B | long-context | RoPE base in the 32K stage | not given | §2.3.3 checked | not reported | — |
| Hunyuan-A52B-Pretrain; Hunyuan-A52B-Instruct | 389B / 52B | eval-gate | supported context length of released checkpoints | Pretrain up to 256K; Instruct up to 128K | GitHub README, "Long-Context Processing Capability" bullet (main, read 2026-09-14) | verified 2026-09-14 | Table 5 evaluates Instruct on RULER and LV-Eval up to 128K |
| Hunyuan-Large-Instruct | 389B / 52B | SFT | data size (unit not stated) | "more than 1 million", selected from "more than 10 million" accumulated instructions | §3.1.1, §3.1.2, §3.1.3 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large-Instruct | 389B / 52B | SFT | data filters | rule-based filters; critique model built on a 70B dense Hunyuan model that assigns a four-tier quality score; human annotation | §3.1.2 | verified 2026-09-14 | no filter ablation or threshold reported |
| Hunyuan-Large-Instruct | 389B / 52B | SFT | epochs | 3 | §3.1.3 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large-Instruct | 389B / 52B | SFT | LR | decays from 2e-5 to 2e-6 | §3.1.3 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large-Instruct | 389B / 52B | SFT | dropout | attention dropout 0.1; hidden dropout 0.2 | §3.1.3 | verified 2026-09-14 | report states MoE Hunyuan models "could benefit more from incorporating suitable dropout rates" than dense models (§3.1.3); no table |
| Hunyuan-Large-Instruct | 389B / 52B | SFT | decay shape; warmup; batch size; max sequence length; packing; loss masking | not given | §3.1 checked; GitHub README fine-tuning manual lists script arguments, not the report's values | not reported | — |
| Hunyuan-Large-Instruct | 389B / 52B | preference | algorithm and loss | DPO in one stage that integrates offline and online training; SFT loss term on the chosen response; "exponential moving average strategy" | §3.2 | verified 2026-09-14 | report states the single-stage approach "demonstrates superior controllability and overall performance" (§3.2); no table or numbers |
| Hunyuan-Large-Instruct | 389B / 52B | preference | data | offline: pre-compiled preference dataset; online: current policy generates multiple responses per prompt and the reward model selects the most and least preferred | §3.2 | verified 2026-09-14 | no ablation reported |
| Hunyuan-Large-Instruct | 389B / 52B | preference | DPO β; SFT-loss weight; EMA decay and the averaged quantity; LR; batch; epochs; pair count; responses per prompt; offline/online ratio | not given | §3.2 checked | not reported | — |
| Hunyuan-Large-Instruct | 389B / 52B | reward-model | architecture; size; training data | not given (described only as "our reward model") | §3.2 checked | not reported | — |
| Hunyuan-Large-Instruct | 389B / 52B | eval-gate | checkpoint selection rule; post-training compute | not given | §3, §4.2 checked | not reported | — |

## Starting point for a small general-purpose run
The report gives no settings for models smaller than 389B total / 52B activated and no ablations for its SFT or DPO values, so this ledger supports no small-run default.
