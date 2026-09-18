<!-- scope: LongRoPE2 — needle-perplexity-guided search of per-dimension RoPE scale factors plus mixed context window mid-training; extends LLaMA3-8B and Phi3-mini-3.8B to 128k while keeping short-benchmark scores
     deps: [[longrope-data]], [[yarn]]
     see-also: [[position-interpolation]], [[localllama-ntk-aware-rope]], [[rope-extrapolation-scaling-laws]], [[ruler]], [[long-context-llama3]], [[skyladder]]
-->

# LongRoPE2: Near-Lossless LLM Context Window Scaling
- **Core Insight:** With 10B mid-training tokens, LongRoPE2 extends LLaMA3-8B from 8k to 128k and scores 82.03 on RULER at 128k, against 73.40 (LongRoPE), 73.19 (NTK), and 49.39 (YaRN) under the same mid-training, while keeping 98.6% of the original model's short-benchmark average (Table 2, Table 4b, §4.2).
- **Guideline:** When extending a RoPE model's context window by mid-training, train short packed documents with the original RoPE and a cross-document mask next to long segments with the rescaled RoPE, and select RoPE by length at inference, because removing this mixed training lowered Phi3-mini-3.8B MMLU from 70.07 to 66.56 and RULER-128k from 58.81 to 56.22 (Table 7); the method was tested only on a 3.8B and an 8B base model at a 128k target (§4.1).
- **Authors:** Ning Shang, Li Lyna Zhang, Siyuan Wang, Gaokai Zhang, Gilsinia Lopez, Fan Yang, et al. (Microsoft)
- **Year:** 2025 (arXiv v1 2025-02)
- **URL:** https://arxiv.org/abs/2502.20082
- **Source type:** paper
- **Relevant topics:** RoPE rescaling, context window extension, long-context mid-training, short-context retention, evolutionary search, needle retrieval

## Abstract
LongRoPE2 extends the effective context window of a pretrained LLM to a target length while keeping performance at the original shorter window. It has three parts: (1) a hypothesis that insufficient training of the higher RoPE dimensions causes the out-of-distribution (OOD) problems that remain in existing rescaling methods; (2) a RoPE rescaling algorithm that uses evolutionary search guided by "needle-driven" perplexity; (3) mixed context window training, which fine-tunes the weights to use rescaled RoPE for long sequences and the original RoPE for short ones. Experiments on LLaMA3-8B and Phi3-mini-3.8B support the hypothesis. LLaMA3-8B reaches a 128K effective context length and retains over 98.5% of short-context performance using 10B tokens, 80x fewer than Meta's approach, which the authors state does not reach its target effective length (abstract).

## Key Contributions
- A hypothesis: higher RoPE dimensions (long periods) see incomplete periods and few long-range dependencies in pretraining, so their empirical periods exceed 2π/θ_i, the real critical dimension d_rcd lies below the theoretical d_tcd, and scale factors above L/L_train are needed (§3.1).
- Needle-driven perplexity: perplexity computed only on the answer tokens of a "needle" placed at the start of a long document and asked for at the end (§3.2, App. C).
- A critical-dimension-aware evolutionary search that tunes factors only for dimensions i ≥ d_rcd and applies NTK scaling below d_rcd (§3.2, Alg. 1–2).
- Mixed context window training: original RoPE with document masks on short windows, rescaled RoPE on 128k long segments, and a length-based RoPE switch at inference (§3.3, Fig. 5, Fig. 10).
- A comparison with YaRN, NTK, and LongRoPE, all given the same mid-training, on RULER, Needle-in-a-Haystack, LOFT, InfiniteBench, LongBench, and short benchmarks (§4).

## Key Figures/Tables to Study
- Fig. 2 and Table 4a: Phi3-mini short scores after 128k extension with each method.
- Fig. 3: how many training tokens cover one period of the 8th versus the 48th RoPE dimension.
- Fig. 4: searched scale factors per dimension versus PI, YaRN, NTK, LongRoPE.
- Fig. 5 and Fig. 10: attention masks for short and long windows; pseudocode for the RoPE switch.
- Table 2 (RULER 4k–128k) and Table 3 (LOFT, InfiniteBench, LongBench).
- Tables 5, 6, 7, 10: ablations of d_rcd, needle perplexity, mixed window training, and searched dimensions.

## Technical Details
- **RoPE period.** Dimension i has rotation angle θ_i = θ_base^(−2i/d) and period T_i = 2π/θ_i, where d is the attention head dimension and θ_base the RoPE base (Eq. 5–6). Larger i means a longer period (§2.1).
- **Theoretical critical dimension.** d_tcd = 2⌈(d/2)·log_θbase(L_train/2π)⌉, where L_train is the pretrained window (Eq. 7). For Phi3-mini (d = 96, θ_base = 10000, L_train = 2048) this is 62, the 31st cosine dimension (§2.2). Check: log_10000(2048/2π) ≈ 0.628; 48 × 0.628 ≈ 30.2; ⌈30.2⌉ = 31 (derived). The 48th cosine dimension has a period of 51861 tokens, and a 2048-token window covers less than 4% of it (§2.1); the 8th dimension has a period of 24 tokens (§3.1).
- **OOD bound.** Rescaled angle θ̂_i = 1/(λ_i·θ_base^(2i/d)), with λ_i the factor for dimension i and L the target length (Eq. 8). Keeping rescaled periods in range requires λ_i ≥ L/L_train for i ≥ d_tcd (Eq. 10). For Phi3-mini 2k→128k, L/L_train = 64 (§2.4).
- **Prior methods (§2.3).** PI uses λ_i = L/L_train in every dimension. NTK (the Liu et al. variant used in the paper) raises θ_base using d_tcd. YaRN sets λ_i = 1 in low dimensions, L/L_train in high dimensions, and a linear ramp between. LongRoPE searches factors with perplexity and a monotone non-decreasing constraint.
- **Motivating observation (§2.4).** Above d_tcd = 31 for Phi3-mini, YaRN uses exactly 64, while NTK and LongRoPE use larger factors and score higher on RULER-128k (49.37, 53.71 vs 39.37) and MMLU (66.43, 67.26 vs 63.22) (Fig. 2c).
- **Search procedure (§3.2, Alg. 1–2, App. B).** Candidate d_rcd values run from d^10_tcd (the dimension with a theoretical 10 periods inside L_train) to d_tcd. Factors for i ≥ d_rcd are drawn from [L/L_train, 2·L/L_train] and mutated under λ_i ≤ λ_(i+1); dimensions below d_rcd get NTK scaling with the base implied by d_rcd. Fitness is needle perplexity on L-token documents built from 10 books of the PG19 validation set. Population 64, 40 iterations, mutation probability 0.3 (App. B).
- **Search result.** d_rcd = 25 for Phi3-mini (theoretical 31) and 30 for LLaMA3-8B (theoretical 35); factors above d_rcd are slightly larger than PI/YaRN/LongRoPE and smaller than NTK (§3.2, Fig. 4).
- **Mixed context window training (§3.3).** Short sequences (≤ L_train) and long sequences (8k–200k) are chunked into 128k segments with BOS and EOS tokens. Short-window segments use the original RoPE and an attention mask that blocks cross-document attention (via `flash_attn_varlen_func`, Fig. 10). Long-window segments use the rescaled RoPE with full attention inside the segment.
- **Inference switch.** Rescaled RoPE is used when input plus generated tokens exceed the pretrained window (App. B); the pseudocode tests `max(position_ids) + 1 > original_max_position_embeddings` (Fig. 10). Switching requires a one-time KV-cache recomputation, which the authors list as a limitation (App. B).
- **RULER (13-task average) at 128k.** LLaMA3-8B: YaRN 49.39, NTK 73.19, LongRoPE 73.40, LongRoPE2 82.03; Phi3-mini: 39.37, 49.37, 53.71, 58.81 (Table 2). LongRoPE on LLaMA3-8B falls from 81.23 at 64k to 73.40 at 128k (§4.2).
- **Real-world long benchmarks, LLaMA3-8B.** LOFT average: YaRN 26.14, NTK 67.14, LongRoPE 60.85, LongRoPE2 74.28; InfiniteBench KV retrieval: 2.2, 66.0, 74.0, 88.0 (Table 3). Models are evaluated without post-training (§4.2).
- **Ablations.** Applying d_rcd to YaRN raises LLaMA3-8B RULER-128k from 49.39 to 71.46 and to NTK from 73.19 to 77.25 (Table 5). A search guided by plain PG19 perplexity gives 78.68 (LLaMA3-8B) and 50.23 (Phi3-mini) at 128k, versus 82.03 and 58.81 with needle perplexity (Table 6). Searching all dimensions gives 78.07 and 57.34 (Table 10).

## Recipe ledger
Values apply to both extended models; the paper states one mid-training setup for "the two models" (§4.1).

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LongRoPE2-extended LLaMA3-8B-128k; Phi3-mini (3.8B)-128k | 8B; 3.8B | long-context | pretrained → target window | 8k → 128k; 2k → 128k | arXiv:2502.20082 v1 Fig. 4, Table 4 | verified 2026-09-14 | no ablation reported |
| same | 8B; 3.8B | long-context | tokens; epochs | 10B tokens; 1 epoch | §4.1 | verified 2026-09-14 | no ablation reported |
| same | 8B; 3.8B | long-context | mixture by source | RedPajama-v1 4.5B, RedPajama-v2 2.5B, StarCoder 2B (8k–200k); FineWeb-Edu 1B for short windows; per-source upsampling of Fu et al. | §4.1 | conflict (Table 1 short bin is 3B) | no ablation reported |
| same | 8B; 3.8B | long-context | mixture by length | ≤ L_train 3B; L_train–100k 3B; 100k–200k 4B | §3.3 Table 1 | conflict (§4.1 gives 1B short tokens) | no ablation reported |
| same | 8B; 3.8B | long-context | segment length; masking | 128k segments with BOS/EOS; short windows: original RoPE + cross-document mask; long: rescaled RoPE, full attention | §3.3, Fig. 5 | verified 2026-09-14 | Table 7: without mixed training Phi3-mini MMLU 66.56 vs 70.07, LLaMA3-8B RULER-128k 80.18 vs 82.03 |
| same | 8B; 3.8B | long-context | global batch | 64 (unit not stated) | §4.1 | verified 2026-09-14 | no ablation reported |
| same | 8B; 3.8B | long-context | learning rate | initial 2e-5, cosine schedule; warmup, final LR, optimizer, weight decay not reported (checked body, App. A–C) | §4.1 | verified 2026-09-14 / not reported | no ablation reported |
| same | 8B; 3.8B | long-context | factor search | population 64, 40 iterations, mutation probability 0.3 | App. B | verified 2026-09-14 | no ablation reported |
| same | 8B; 3.8B | long-context | search space and fitness | λ_i ∈ [L/L_train, 2·L/L_train] for i ≥ d_rcd, monotone; needle perplexity on 10 PG19 validation books | Alg. 1–2, §3.2 | verified 2026-09-14 | Table 6 (needle vs plain PPL); Table 10 (d_rcd+ vs all dims) |
| same | 8B; 3.8B | long-context | compute | 64 A100 GPUs; about 54 h (LLaMA3-8B), 39 h (Phi3-mini) for 10B tokens; nnScaler, FlashAttention-2 | §4.1, App. B | verified 2026-09-14 | — |
| same | 8B; 3.8B | long-context | checkpoint selection; seeds | not reported (checked body, App. A–C) | — | not reported | — |

## Findings relevant to generality and long context
- **Short-context retention.** Short benchmarks run with a 4096-token context (§4.1). Phi3-mini average: original 63.2, YaRN 53.6, NTK 57.3, LongRoPE 58.5, LongRoPE2 61.7; LLaMA3-8B: original 56.5, YaRN 52.1, NTK 54.0, LongRoPE 54.6, LongRoPE2 55.7, Meta LLaMA3.1-8B 57.2 (Table 4). YaRN and NTK lower Phi3-mini GSM8K by 21.15 and 14.55 points (§4.2). Retention is 97.6% (Phi3-mini) and 98.6% (LLaMA3-8B) (§4.2). Result (single study).
- **Mixed training also helps long context.** Disabling it lowers RULER at every length from 8k to 128k for both models (Table 7). The authors' explanation: keeping the original RoPE for short contexts lets long training focus on the new positions (§4.3). Interpretation.
- **Perplexity as a search target.** The paper states that average perplexity is dominated by tokens without long dependencies and does not capture long-context ability (§3.2); Table 6 shows the plain-perplexity search choosing factors with lower RULER scores.
- **Cost comparison.** Llama 3.1 used a six-stage extension with 800B tokens from 8k to 128k (§2.4), as reported by the LongRoPE2 authors.
- **Limits.** Two base models, one target length, no post-training (§4.1–4.2); no repeated runs reported. RULER CWE at 128k is 0.3 (Phi3-mini) and 9 (LLaMA3-8B) (Tables 8–9).

## Connections
- [[longrope-data]] — LongRoPE (v1), whose search and monotone constraint LongRoPE2 extends.
- [[yarn]], [[position-interpolation]], [[localllama-ntk-aware-rope]] — the baselines' rescaling rules.
- [[rope-extrapolation-scaling-laws]] — source of the critical-dimension formula (Eq. 7); [[rope-base-bounds-context-length]] — cited for larger-than-theory factors.
- [[ruler]], [[loft]], [[infinitebench]], [[longbench]] — long-context evaluations used.
- [[long-context-data-engineering]] — per-source upsampling followed for the 10B-token mix.
- [[llama-3]], [[long-context-llama3]] — the 800B-token Llama 3.1 extension used as the cost comparison; [[phi-3]] — Phi3-mini base model.
- [[longrecipe]], [[prolong]], [[controlled-long-context-extension]], [[longred]] — other long-context extension recipes and short-text degradation studies.
- [[hf-transformers-rope-utils]] — library implementation of `longrope` scaling with short and long factors.
- [[skyladder]] — context window scheduling in pretraining rather than mid-training.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.20082 (arXiv v1, 2025-02-27; the only listed version).
- Audit claims not found in the source: "retaining over 98.5% of short-context performance" for both LLaMA3-8B and Phi3-mini (the abstract states it for LLaMA3-8B; §4.2 gives 97.6% for Phi3-mini); "3B short, 3B 8K–100K, 4B 100K–200K" as the data split (Table 1 labels the middle bin L_train–100k, and §4.1 gives a different split with 1B short tokens; recorded as conflict).
- Internal inconsistencies in the source: Phi3-mini LongRoPE2 MMLU 70.04 (Table 4a) vs 70.07 (Tables 5, 7, 10); Phi3-mini RULER-8k 87.22 (Table 2), 86.87 (Table 7), 87.34 (Table 8); CWE "only 1%" (§4.2) vs 0.3 (Table 8).
- Not reported by the source: optimizer, warmup, final LR, weight decay, seeds, checkpoint selection; Llama 3.1 RULER values appear only in Fig. 1.
