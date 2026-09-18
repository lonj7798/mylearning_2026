<!-- scope: controlled comparison of context-extension methods (PI, NTK/Dynamic NTK, YaRN, CLEX, LongLoRA, Landmark, LM-Infinite, Self-Extend) on one LLaMA2-7B base with 1B tokens and a fixed recipe; perplexity vs NIAH, RULER, LongBench, many-shot ICL
     deps: [[position-interpolation]], [[yarn]]
     see-also: [[long-context-data-engineering]], [[ruler]], [[longbench]], [[helmet]], [[prolong]]
-->

# A Controlled Study on Long Context Extension and Generalization in LLMs
- **Core Insight:** With one LLaMA2-7B base, 1B tokens of length-upsampled SlimPajama and a 32k training length for every fine-tuned method, Dynamic NTK (NTK-32K) scores 59.42 on RULER at 32k and 46.26 at 64k, PI and YaRN score 57.66 and 36.95 at 32k but 0.00 at 64k, and fine-tuned LongLoRA scores 3.53 at 32k (Table 3).
- **Guideline:** When a RoPE model is extended by continued fine-tuning, use exact attention with a length-dependent RoPE scaling (Dynamic NTK in this study) and evaluate at and beyond the training length, because PI and YaRN score 0.00 on RULER at 64k while NTK-32K scores 46.26 (Table 3), and the fine-tuned approximate-attention methods score below the unextended base on LongBench (LongLoRA 23.30, Landmark 28.19 vs 32.92; Table 4); the evidence covers LLaMA2-7B (and Phi-2 in App. 9.1) at 32k only.
- **Authors:** Yi Lu, Jing Nathan Yan, Songlin Yang, Justin T. Chiu, Siyu Ren, Fei Yuan, et al. (Cornell University, Shanghai AI Lab, MIT)
- **Year:** 2024 (arXiv v1 2024-09; v2 2024-09-23; preprint, under review)
- **URL:** https://arxiv.org/abs/2409.12181 (code: https://github.com/Leooyii/LCEG)
- **Source type:** paper
- **Relevant topics:** context extension, RoPE scaling, perplexity as a long-context metric, length extrapolation, approximate attention, short-context degradation

## Abstract
Many methods extend pretrained LLMs to long contexts, but differences in base models and extension data make them hard to compare, and it is unclear how long-context performance should be evaluated. The authors implement a controlled protocol with a consistent base model, fixed extension data and standardized evaluation. They report three findings: perplexity remains a general-purpose performance indicator for long-context tasks; current approximate attention methods systematically underperform; exact fine-tuning methods are generally effective within their extension range, while extrapolation remains hard. Code, models and checkpoints are released (Abstract).

## Key Contributions
- One framework and one training recipe for 8 methods (10 variants) in 4 classes: exact frozen (NTK-Frozen), exact fine-tuned (PI, YaRN, CLEX, NTK-32K, NTK-64K), approximate frozen (LM-Infinite, Self-Extend), approximate fine-tuned (LongLoRA, Landmark) (§4, Table 1).
- Intrinsic metrics (PG19 and Proof-pile perplexity, NIAH, RULER) and extrinsic metrics (LongBench, many-shot TREC) on the same checkpoints (§4 "Metrics").
- Evidence that perplexity and downstream accuracy correlate for exact-attention methods under control (§6, Figure 4).
- A replication on Phi-2-base (App. 9.1) and released checkpoints.

## Key Figures/Tables to Study
- Table 1: overview (PPL, Needle, many-shot, LongBench, RULER) per method.
- Table 2: perplexity at 2k-64k on PG19 and Proof-pile. Table 3: RULER at 4k-64k.
- Table 4 / Table 11: LongBench per task. Figure 1: NIAH heatmaps with the training length marked.
- Figure 3: average NLL by token position. Figure 4: PPL@32K vs NIAH, LongBench, RULER.
- Tables 7-9: training scale factors, inference scale factors, NTK scale-factor grid search.

## Technical Details
- **RoPE scaling (§3.2):** each method rescales the frequencies θ_j = b^(−2j/d_k) (b = 10000) by a vector α. PI: α_j = C/C′ = 1/t (Eq. 8). NTK-RoPE: α_j = κ^(−2j/d_k) with κ = t^(d_k/(d_k−2)), so the lowest frequency matches PI and the highest is unchanged (Eq. 9). YaRN: α_j = ((1 − γ_j)/t + γ_j)/√T, with a ramp γ_j ∈ [0, 1] set from θ_j by thresholds p and q, and temperature T (Eq. 10-11). Symbols: C = pretrained length, C′ = target length, t = C′/C, d_k = head dimension, j = frequency index. Dynamic NTK recomputes the scaling from the current length, following Fu et al. (2024) (§3.2, App. 9.2).
- **Dynamic NTK factor (App. 9.2):** t is replaced by s · max(C′, C_test)/C − (s − 1), with s = C′/(2C). Worked check for NTK-32K (C = 4k, C′ = 32k): s = 4; up to 32k, t = 4 · 8 − 3 = 29; at 64k, t = 4 · 16 − 3 = 61. Both match the printed factors 29.0 and 61.0 (Tables 7-8). Table 7 prints 57.0 for NTK-64K, which is the grid-search choice in Table 9, not the value this formula gives with s = C′/(2C).
- **Approximate and frozen settings:** LongLoRA uses block-diagonal shifted local attention in training and full attention at inference (§3.3). LM-Infinite: global memory G = 10, local window M = 4096 (App. 9.2). Self-Extend: neighbor window M = 1024, group size N = 64 (App. 9.2). Landmark: training length 512, block size 64 (§4).
- **Evaluation:** perplexity with a sliding window of 256 (§4). RULER: 13 tasks, 500 examples per length at 4k, 8k, 16k, 32k, 64k (§5.2). LongBench: 32k window, prompt truncated from the middle; average dataset length 7,425, unit not stated; the text says approximately 7.5k (§4, §5.3, Table 11). Many-shot TREC News with 1 to 1000 examples (§5.3).
- **Perplexity (PG19, 32k):** NTK-32K 5.79, CLEX 5.82, YaRN 5.93, PI 5.95, LM-Infinite 6.71, Self-Extend 6.11, Landmark 8.13, LongLoRA 9.89, NTK-Frozen 14.52 (Table 2). At 64k: NTK-32K 5.76, CLEX 5.79, NTK-64K 5.85; PI and YaRN not reported on PG19; YaRN 106.38 on Proof-pile (Table 2). Table 1 prints PPL 5.85 for both PI and YaRN, which differs from Table 2.
- **RULER average (4k / 8k / 16k / 32k / 64k):** LLaMA2 80.94 at 4k; PI 84.56 / 76.04 / 69.64 / 57.66 / 0.00; NTK-32K 86.58 / 77.75 / 70.01 / 59.42 / 46.26; NTK-64K 86.60 / 76.34 / 69.56 / 60.03 / 49.31; YaRN 79.12 / 65.60 / 54.21 / 36.95 / 0.00; CLEX 50.18 / 63.93 / 64.35 / 52.17 / 30.61; LongLoRA 10.58 / 6.37 / 3.67 / 3.53 / 0.00; Landmark 22.37 / 17.52 / 16.31 / 13.56 / 14.15; LM-Infinite 81.05 / 30.01 / 18.02 / 12.34 / 10.56; NTK-Frozen 81.14 / 44.45 / 14.79 / 0.72 / 0.91; Self-Extend 65.03 / 50.73 / 44.02 / 29.50 / 9.34 (Table 3). Appendix Table 12 swaps the 4k values of YaRN (50.18) and CLEX (79.12).
- **NIAH:** NTK, PI and YaRN retrieve the needle "within the pretraining length" and only NTK and CLEX "beyond the pretraining length" (§5.2, Figure 1); Figure 1 marks the fine-tuning length, and the RULER 64k results (Table 3) are consistent with reading this as the 32k training length (Interpretation). With 1B tokens NTK-32K beats NTK-64K; NTK-64K trained on 2B tokens improves (§5.2, App. 9.4, Figure 5). Table 1 "Needle" scores: NTK-32K 83.7, CLEX 71.1, NTK-64K 69.1, Landmark 50.9, YaRN 46.7, PI 42.1, Self-Extend 25.8, LM-Infinite 23.9, LongLoRA 20.3, NTK-Frozen 18.8; the paper does not define the aggregation or length for this column.
- **LongBench average:** base 32.92; NTK-32K 35.32; NTK-64K 34.30; Self-Extend 33.62; PI 33.48; CLEX 33.48; YaRN 33.45; Landmark 28.19; LM-Infinite 25.84; NTK-Frozen 25.54; LongLoRA 23.30 (Tables 4, 11). The authors attribute the small gain over base to the average test length (approximately 7.5k) being shorter than the 32k window (§5.3) (Interpretation).
- **Many-shot TREC:** exact-attention methods gain +44.0% from 10 to 50 examples, +5.7% from 50 to 100, and +25.9% from 100 to 1000; approximate methods stay below them (§5.3, Figure 2).
- **Phi-2-base replication (RULER, 12 tasks, 32k):** NTK-32K 32.06, NTK-64K 25.66, CLEX 25.46, Self-Extend 7.83, PI 4.78, NTK-Frozen 0.06 (App. 9.1, Table 6). Only NTK generalizes in Proof-pile perplexity beyond the trained window (Table 5).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA2-7B extensions (PI, NTK-32K, YaRN, CLEX, LongLoRA, Landmark) | 7B | long-context | train length; tokens | 32k; 1B | arXiv:2409.12181v2 §4, App. 9.2 Table 7 (CLEX: §4, Table 2 "Len") | verified 2026-09-14 | 32k chosen because most downstream tasks need under 32k (§4) |
| LLaMA2-7B NTK-64K | 7B | long-context | train length; tokens | 64k; 1B (second run: 2B) | Table 7; App. 9.4 | verified 2026-09-14 | App. 9.4 Figure 5: 2B tokens improves NIAH over 1B |
| all fine-tuned variants | 7B | long-context | data | SlimPajama, per-source length upsampling, domain mixture kept; packed to training length across document boundaries | App. 9.3 | verified 2026-09-14 | follows Fu et al. (2024); no ablation reported |
| all fine-tuned variants | 7B | long-context | LR; warmup; weight decay | 2e-5; linear warm-up (length not reported); 0 | §4, App. 9.2, Table 7 | verified 2026-09-14 | no ablation reported |
| all fine-tuned variants | 7B | long-context | batch size | 32 (unit not stated) | Table 7 | verified 2026-09-14 | no ablation reported |
| all fine-tuned variants | 7B | long-context | weight EMA | constant decay; decay value not reported | §4, App. 9.2 | verified 2026-09-14 (value not reported) | no ablation reported |
| all fine-tuned variants | 7B | long-context | origin of hyperparameters | §4: "based on [Fu et al., 2024]"; App. 9.2: "derived from [Chen et al., 2023b]" | §4; App. 9.2 | conflict | no ablation reported |
| PI / YaRN / LongLoRA | 7B | long-context | RoPE scale factor (train and inference) | 8.0 | Tables 7-8 | verified 2026-09-14 | §4: original scale factor reused for consistency for PI, YaRN, NTK; no reason given for LongLoRA |
| NTK-32K | 7B | long-context | scale factor train; inference ≤32k / 64k | 29.0; 29.0 / 61.0 | Tables 7-8 | verified 2026-09-14 | Table 9 grid on first 2 PG19 documents: 29 best at 32k (6.82), 61 best at 64k (6.63) |
| NTK-64K | 7B | long-context | scale factor | 57.0 | Tables 7-8 | verified 2026-09-14 | Table 9: 57 gives 6.75 at 64k vs 6.77 for 121 |
| CLEX | 7B | long-context | max scale factor; activation | 32; SiLU | §4, App. 9.2 | verified 2026-09-14 | no ablation reported |
| LongLoRA | 7B | long-context | trainable parts | LoRA adapters + embeddings + normalization, merged for evaluation | §4 | verified 2026-09-14 | App. 9.6: reproduction of the released 32k model, PG19 32k PPL 7.32 vs 7.22 reported |
| all fine-tuned variants | 7B | long-context | hardware | 8 NVIDIA A100 | §4 | verified 2026-09-14 | not applicable |

Not reported: optimizer, betas, number of steps or epochs, warmup length, EMA decay value, gradient clipping, seeds.

## Findings relevant to generality and long context
- **Short-context cost:** at 2k on PG19 every fine-tuned exact method has higher perplexity than LLaMA2 (6.61): NTK-32K 6.63, YaRN 6.70, NTK-64K 6.83, CLEX 6.85, PI 6.88 (Table 2); on Proof-pile at 2k, NTK-32K (3.27) and YaRN (3.29) are below LLaMA2 (3.34) (Table 2). In per-position NLL, LLaMA2 is best up to 4k (§6, Figure 3). The authors summarize this as "context extension hurts in the short term and gains in the long term" (§6) (Interpretation). RULER at 4k does not show the drop: NTK-32K 86.58 and PI 84.56 vs 80.94 (Table 3).
- **Code completion:** LongBench LCC falls from 68.22 (base) to 56.78 (NTK-32K), 55.05 (PI), 54.06 (YaRN), 49.45 (CLEX); REPO (abbreviation as printed) falls from 61.73 to 49.09 (NTK-32K) and 39.68 (NTK-64K) (Table 11). The paper does not discuss these task-level drops.
- **Perplexity as a metric:** for exact fine-tuned methods PPL@32K correlates with NIAH, LongBench and RULER scores (linear fits in Figure 4; §6). LM-Infinite has good perplexity at 32k but fails NIAH beyond 4k, so approximate methods need downstream tests at several lengths (§6).
- **Extrapolation:** NTK-32K generalizes beyond 32k on NIAH and RULER, on par with NTK-64K, while NTK-Frozen generalizes only to 8k, which the authors read as extrapolation depending on continual fine-tuning (§6) (Interpretation).
- **Limits stated by the authors:** one base model (plus Phi-2), 32k training length only, and a fixed recipe that may favor some methods (§7).

## Connections
- [[position-interpolation]], [[yarn]], [[localllama-ntk-aware-rope]] — the RoPE scaling methods compared here.
- [[long-context-data-engineering]] — Fu et al. (2024): the 1B-token data mixture and the Dynamic NTK setup follow it.
- [[longalpaca]] — LongLoRA, the approximate fine-tuned method.
- [[streamingllm-attention-sinks]] — LM-Infinite and StreamingLLM keep initial tokens plus a local window (§2).
- [[ruler]], [[longbench]], [[needle-in-haystack-data]] — the evaluation suites.
- [[helmet]], [[prolong]] — later long-context evaluation and training work with its own conclusions on which metrics predict downstream performance; not cited in this paper.
- [[hf-transformers-rope-utils]] — implementations of the dynamic and YaRN RoPE scalings.
- [[longred]] — a method that targets the short-text degradation seen in Table 2.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2409.12181 (arXiv v2, 2024-09-23; v1 2024-09-18).
- Audit claims not found in the source: "a counterpoint to HELMET/ProLong" (the paper cites Sun et al. 2021 and An et al. 2023, not HELMET or ProLong); "NIAH 83.7 ... at 32K" (Table 1 gives no length for the Needle column); "The NTK variant led at 32K" on RULER holds among 32k-trained models only (NTK-64K 60.03 > NTK-32K 59.42, Table 3); "extended models show degradation at 2K-4K" holds for perplexity (Table 2, Figure 3) but not for RULER at 4k (Table 3).
