<!-- scope: recipe ledger (continued long-context training, SFT, and ablation settings) for ProLong (Gao et al., "How to Train Long-Context Language Models (Effectively)")
     deps: [[prolong]]
     see-also: [[long-context-data-engineering]], [[long-context-llama3]], [[sequence-packing]]
-->

# Recipe ledger: How to Train Long-Context Language Models (Effectively)
- **Core Insight:** The paper prints its full recipe in Table 9 (40B tokens of continued training from Llama-3-8B-Instruct in two stages, then 1B tokens of UltraChat SFT); the released `train_512K.sh` sets a stage-2 peak learning rate of 5e-6, while Table 9 gives 1e-5 for each stage.
- **Guideline:** When reproducing ProLong, take values from arXiv v4 Table 9 and the released scripts, keep the stage-2 learning rate conflict explicit, and treat every "not reported" row as a choice the paper did not ablate.
- **Authors:** Tianyu Gao, Alexander Wettig, Howard Yen, Danqi Chen (Princeton Language and Intelligence)
- **Year:** 2024 (arXiv v1 2024-10; v4 2025-12; ACL 2025)
- **URL:** https://arxiv.org/abs/2410.02660 ; https://github.com/princeton-nlp/ProLong (commit 499fa29, 2025-09-12)
- **Source type:** paper + released config/code
- **Relevant topics:** long-context continued pretraining hyperparameters, data mixture, RoPE base, sequence parallelism, SFT settings

Companion to [[prolong]]. Paper values are from arXiv:2410.02660v4 (Table 9 is identical in v1). Script values are from `train_64K.sh`, `train_512K.sh`, and `train_sft.sh` at commit 499fa29. The paper uses binary prefixes: K = 2^10, M = 2^20, B = 2^30 (footnote 2), so "64K" is 65,536 tokens (`--per_device_max_tokens 65536`) and "512K" is 524,288 tokens.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ProLong-64k-Base (princeton-nlp/Llama-3-8B-ProLong-64k-Base) | 8B | long-context | initialization | Llama-3-8B-Instruct (original RoPE base 5×10^5) | v4 Table 9; train_64K.sh `model` | verified 2026-09-14 | App. B.3 Table 21 (5B-token ablation): Instruct init 55.0 long / 67.7 short vs base 54.6 / 65.5 |
| ProLong-64k-Base | 8B | long-context | tokens; max sequence length | 20B tokens; 64K | v4 Table 9, §4 | verified 2026-09-14 | §4 Fig. 4: recall, RAG, re-rank, summarization trend up over training |
| ProLong-64k-Base | 8B | long-context | data mixture (paper; token vs sampling share not stated) | 30% code repos, 30% books, 3% textbooks, 37% ShortMix; ShortMix = 27% FineWeb-Edu, 27% FineWeb, 11% Wikipedia, 11% StackExchange, 8% Tulu-v2, 8% OpenWebMath, 8% ArXiv | v4 Table 9 | verified 2026-09-14 | Table 4: books/repos 1:1 best long avg (54.6); Fig. 3: 60% long best avg; Table 6: ShortMix best; textbooks not ablated (App. A.2) |
| ProLong-64k-Base | 8B | long-context | data mixture (released script, domain mixing proportions) | code repos 0.3, books 0.3, fineweb-edu 0.1, fineweb-2023-50 0.1, stackexchange 0.04, dolmawiki 0.04, tuluv2 0.03, arxiv 0.03, openwebmath 0.03, textbooks 0.03 | train_64K.sh `domains` | verified 2026-09-14 | same as Table 9 row |
| ProLong-64k-Base | 8B | long-context | RoPE base | 8×10^6 | v4 Table 9; train_64K.sh `rope_theta=8000000` | verified 2026-09-14 | App. B.1 Table 18: avg 54.6 vs 48.7 at 4×10^6 (dynamic NTK suggestion) and 29.1 at 5×10^5 |
| ProLong-64k-Base | 8B | long-context | peak LR; warmup; decay | 1e-5; 10%; cosine to 1e-6 | v4 Table 9; train_64K.sh `lr=1e-5`, `warmup=0.1`, `--min_lr_ratio 0.1` | verified 2026-09-14 | no ablation reported (Limitations: optimization hyperparameters not exhausted) |
| ProLong-64k-Base | 8B | long-context | optimizer | AdamW, weight decay 0.1, β1 = 0.9, β2 = 0.95 | v4 Table 9; train_64K.sh | verified 2026-09-14 | no ablation reported |
| ProLong-64k-Base | 8B | long-context | gradient clipping | 1.0 (script only; not in paper) | train_64K.sh `--max_grad_norm 1.0` | verified 2026-09-14 | no ablation reported |
| ProLong-64k-Base | 8B | long-context | global batch | 4M tokens; script: 64 sequences × 64K, 5000 steps | v4 Table 9; train_64K.sh `bsz=64`, `steps=5000` | verified 2026-09-14 | no ablation reported |
| ProLong-64k-Base | 8B | long-context | attention; packing | cross-document attention masking; short data packed into 64K sequences; long data from documents ≥ 64K tokens | v4 §6.1, App. A.2, App. B.2 | verified 2026-09-14 | Table 20: masks 54.6 long / 65.5 short vs no masks 53.6 / 64.9 |
| ProLong-64k-Base | 8B | long-context | compute | 2.2K H100 hours | v4 Table 9 | verified 2026-09-14 | n/a |
| ProLong-512k-Base (princeton-nlp/Llama-3-8B-ProLong-512k-Base) | 8B | long-context | tokens; length; initialization | 20B tokens at 512K from the 64K model; LR schedule reset; script resumes optimizer state | v4 Table 9, §4, §6.1; README | verified 2026-09-14 | Table 7 (eval at 64K): +4B at 512K beats +4B at 64K on recall 98.5 vs 95.0 and re-rank 32.9 vs 28.0 |
| ProLong-512k-Base | 8B | long-context | length curriculum | code repos 50% at 512K / 50% at 64K; books 17% at 512K / 83% at 64K; textbooks at 512K; script: code 0.15 + 0.15, books 0.05 + 0.25, other domains as in stage 1 | v4 Table 9, App. A.2; train_512K.sh | verified 2026-09-14 | App. A.2: ratios chosen roughly by availability of 512K documents; no ablation reported |
| ProLong-512k-Base | 8B | long-context | RoPE base | 1.28×10^8 | v4 Table 9; train_512K.sh `rope_theta=128000000` | verified 2026-09-14 | Table 19 (5B tokens from ProLong-64K): 57.7 vs 57.6 (6.4×10^7) and 57.8 (2.56×10^8) |
| ProLong-512k-Base | 8B | long-context | peak LR | paper: 1e-5 with 10% warmup, cosine to 1e-6, "each stage" | v4 Table 9 | conflict | no ablation reported; the source that produced the released checkpoint is not stated |
| ProLong-512k-Base | 8B | long-context | peak LR | script default: 5e-6 (warmup 0.1, min_lr_ratio 0.1) | train_512K.sh `lr=${LR:-5e-6}` | conflict | no ablation reported |
| ProLong-512k-Base | 8B | long-context | global batch | 8M tokens; script: 128 × 512K / 8 (sequence-parallel size), 2500 steps | v4 Table 9; train_512K.sh | verified 2026-09-14 | no ablation reported |
| ProLong-512k-Base | 8B | long-context | parallelism; compute | DeepSpeed-Ulysses sequence parallelism over 8 GPUs, only for 512K sequences; 12.2K H100 hours | v4 App. A.3, Table 9; train_512K.sh `--seq_parallel_size 8` | verified 2026-09-14 | n/a |
| ProLong-512k-Instruct (princeton-nlp/Llama-3-8B-ProLong-512k-Instruct) | 8B | SFT | data; tokens | UltraChat only; 1B tokens; conversations concatenated into 64K chunks, truncated remainder discarded | v4 Table 9, §5, App. A.2; README | verified 2026-09-14 | Table 22: UltraChat 55.7 vs Tulu v2 49.1 vs ShareGPT 45.2; Table 8: 0% synthetic 55.7 vs 54.1 at 1% |
| ProLong-512k-Instruct | 8B | SFT | peak LR; warmup; decay; optimizer | 2e-5; 5%; cosine to 2e-6; AdamW (wd 0.1, β1 0.9, β2 0.95) | v4 Table 9; train_sft.sh `lr=2e-5`, `warmup=0.05` | verified 2026-09-14 | no ablation reported |
| ProLong-512k-Instruct | 8B | SFT | global batch | 4M tokens; script: 64 × 64K, 250 steps | v4 Table 9; train_sft.sh | verified 2026-09-14 | no ablation reported |
| ProLong-512k-Instruct | 8B | SFT | loss masking; loss averaging | instruction tokens masked; loss averaged over valid tokens instead of per sequence/device | v4 App. A.3; train_sft.sh `--apply_instruct_masks`, `--token_scaled_loss` | verified 2026-09-14 | no ablation reported |
| Ablation runs (Llama-3-8B base) | 8B | long-context | setting | 5B tokens at 64K; Table 9 hyperparameters; no textbooks; 60% long / 40% ShortMix; SFT as Table 9 | v4 App. A.4, §3.1 | verified 2026-09-14 | 64K chosen as the largest power of 2 trainable without sequence parallelism (App. A.4) |
| Ablation runs (Llama-3-8B base) | 8B | long-context | ShortMix proportions | FineWeb 25, FineWeb-Edu 25, Wikipedia 10, Tulu-v2 10, StackExchange 10, ArXiv 10, OpenWebMath 10 (%) | v4 Table 5, App. A.4 | verified 2026-09-14 | Table 6: long 54.6 / short 65.5 vs SlimPajama 52.9 / 64.2, FineWeb-Edu 53.0 / 63.0, DCLM-Baseline 52.0 / 64.8 |
| Synthetic long SFT data (ablation only) | 8B | SFT | mix; generator | 40% QA, 30% RAG, 30% summarization (token share); Llama-3-8B-Instruct, also Llama-3-70B-Instruct | v4 §5, App. A.5, App. B.5 | verified 2026-09-14 | Tables 8, 23: no ratio (1–50%) beats 0% (55.7) |
| Released data | n/a | long-context | released dataset sizes | prolong-data-64K 40B tokens; prolong-data-512K 40B tokens; prolong-ultrachat-64K 1B tokens (packed and sampled) | README@499fa29 | verified 2026-09-14 | n/a (training used 20B tokens per stage, Table 9) |
| All ProLong models | 8B | eval-gate | checkpoint selection rule | not reported | checked v4 §4, §6, App. A; README | not reported | none |
