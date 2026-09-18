<!-- scope: LongReD (arXiv 2502.07365, ACL 2025 Main): causes of short-text degradation after RoPE scaling + continual pretraining (distribution drift, forgetting) and restoration distillation from the original model
     deps: [[long-context-data-engineering]], [[position-interpolation]]
     see-also: [[longpo]], [[llama-2-long]], [[pose-synthesis]], [[catastrophic-forgetting-continual-finetuning]]
-->

# LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation
- **Core Insight:** When Llama-3-8B is extended to 32K with ABF and about 1B tokens, continual pretraining on long text alone keeps 92.5% of the original short-text average (51.00 vs 55.16), while LongReD keeps 99.4% (54.85) and scores RULER 84.98 vs 82.80 (§5.2, Table 3).
- **Guideline:** When a context window is extended by RoPE scaling plus a short continual-pretraining run (about 1B tokens in this paper), add hidden-state distillation from the unextended model on 1K-token high-quality short text and a short-to-long distillation term, because this gave higher short-text averages than long-only and mixed-length CPT in all four settings of Table 3; at large extension ratios, check RULER as well, because for Llama-3-8B at 128K LongReD scored 64.93 (CREAM) and 68.41 (uniform) vs 69.70 for long-only CPT.
- **Authors:** Zican Dong, Junyi Li, Jinhao Jiang, Mingyu Xu, Wayne Xin Zhao, Bingning Wang, et al. (Renmin University of China; National University of Singapore; Baichuan Inc.)
- **Year:** 2025 (arXiv v1 2025-02; ACL 2025 Main)
- **URL:** https://arxiv.org/abs/2502.07365
- **Source type:** paper
- **Relevant topics:** context window extension, short-text degradation, catastrophic forgetting, hidden-state distillation, skipped positional indices, continual pretraining

## Abstract
Context windows are extended by scaling positional encodings and lightweight continual pretraining, which often lowers performance on short-text tasks. The authors identify two causes: distribution drift in hidden states and attention scores, and catastrophic forgetting during continual pretraining. LongReD (Long Context Pre-training with Restoration Distillation) reduces the discrepancy between the extended and original models. Besides training on long texts, it distills hidden states of selected layers from the original model on short texts, and adds a short-to-long distillation that aligns outputs on short texts with skipped positional indices to the original model's outputs. On common benchmarks LongReD preserves short-text performance while keeping long-text ability comparable to or better than baselines. Code: github.com/RUCAIBox/LongReD.

## Key Contributions
- Measures of drift (hidden-state cosine similarity, attention KL) and evidence that continual pretraining restores the original distribution only partly (§2.2, §3.1, Table 1).
- Correlation between hidden-state similarity and MMLU preservation across 15 extended models (§3.1, Fig. 2, App. D Table 12).
- Forgetting analysis over training steps and with short-text replay (§3.2, Fig. 3, Table 2).
- LongReD objective: long-text LM loss + short-text hidden-state distillation + short-to-long distillation (§4, Eq. 18).
- Ablations of loss weights, distilled layers, distillation length; comparison with model merging and attention-only tuning (§5.3, Tables 4-7).

## Key Figures/Tables to Study
- Table 1 and Fig. 2: drift before and after continual pretraining; similarity vs MMLU preservation.
- Table 3 (details in App. F Tables 14-16): short-text categories and RULER for all settings.
- Tables 4-7: ablations. Table 9: RoPE settings and dataset lengths. Table 10: training time.

## Technical Details
- **Drift measures.** Sim(H_l, Ĥ_l): mean cosine similarity between original and extended hidden states at layer l over positions (Eq. 5). KL(A_l^i, Â_l^i): mean KL divergence between attention distributions of head i (Eq. 6). Measured on 1,000 SlimPajama samples of 8,192 tokens (§3.1).
- **Partial restoration (Table 1).** Llama-3-8B-32K (base 2e7): similarity 0.92 → 0.95, KL (×1e-5) 3.51 → 1.51 after 1B tokens of CPT; Llama-3-8B-128K (base 1e8): 0.83 → 0.94, 6.74 → 1.68; Llama-3-8B-Instruct-262K (2.8e8): 0.76 → 0.91, 8.75 → 2.50.
- **Drift vs short-text loss.** Models with higher similarity keep more MMLU (Fig. 2). Examples: ABF 5e6 at 32K, MMLU ratio 0.960 with similarity 0.943; ABF 5e20, 0.872 with 0.856; PI×16, 0.866 with 0.862 (App. D Table 12, 256M tokens each). App. D.1 derives an upper bound on attention-score change that grows with the RoPE base (Table 13).
- **Forgetting.** With base 2e7, batch size 64, 32K length, short-text scores recover within the first steps (usually fewer than 32) and then decline as training continues (§3.2, Fig. 3). Replacing half the long data with 8K short text: MMLU 62.0 → 62.5, HumanEval 14.02 → 16.46, PIQA 74.10 → 78.24, TriviaQA 70.67 → 72.82 (Table 2).
- **Short-text distillation.** Select M layers with the largest attention KL between extended and original models, plus the last layer. L_short = −Σ_x Σ_i Sim(H_{l_i;p,Θe}, H_{l_i;p,Θo}) on short texts with normal positions p = [0, …, T_s−1] (§4.2, Eq. 8). Θe: extended model; Θo: original model.
- **Short-to-long distillation.** A text of original window length T is split into head, middle, and tail with threshold T_b. The head keeps its indices, the tail is shifted to end at the target length T_l, and the middle end index is sampled by CREAM (truncated Gaussian, LongReD-C) or uniformly (LongReD-U) (§4.3, App. A). L_s2l = −Σ_x Sim(H_{L;p̂,Θe}, H_{L;p,Θo}) on the last layer only (Eq. 17).
- **Joint loss.** L_final = L_long + α1·L_short + α2·L_s2l; each batch samples the three datasets at a fixed ratio (§4.4, Eq. 18).
- **Main results (Table 3; short avg / RULER).** Llama-3-8B original 8K: 55.16 / –. 32K ABF: Long CPT 51.00 / 82.80; Mix CPT 45.86 / 85.04; LongReD-C 54.85 / 84.98; LongReD-U 54.56 / 84.08. 32K PI: Long CPT 49.36 / 82.17; LongReD-C 53.94 / 81.19; LongReD-U 54.39 / 79.30. 128K ABF: Long CPT 50.14 / 69.70; Mix CPT 43.33 / 69.64; LongReD-C 54.03 / 64.93; LongReD-U 53.85 / 68.41. Mistral-7B-v0.3 original 32K: 51.09 / –; 128K ABF: Long CPT 40.68 / 44.63; Mix CPT 40.66 / 45.76; LongReD-C 46.30 / 58.37; LongReD-U 47.69 / 53.60.
- **MMLU alone (App. F Table 14).** Llama-3-8B 64.80 → Long CPT 62.00 → LongReD-C 64.40 (32K ABF). Mistral-7B-v0.3 62.30 → Long CPT 51.90 → LongReD-U 60.1 (128K ABF).
- **Ablations (Llama-3-8B, 32K, ABF, LongReD-C).** Without L_short (α2 = 15): Common 72.35 → 70.64, General 67.65 → 66.54, RULER 85.48. Without L_s2l: RULER 83.61. α2 = 100: RULER 80.40; α1 = 100: 80.19; α1 = 2: 86.03 (Table 4). Layers KL(6) RULER 84.98 vs Uniform(6) 82.53, All 81.47, Last 82.24 (Table 5). T_s 1024 / 2048 / 8192: RULER 84.98 / 83.20 / 75.54 (Table 6). Model merging: General 65.92, RULER 83.38; attention-only tuning: 66.82, 85.86; CPT: 65.39, 82.80 (Table 7).
- **Compute.** Llama-3-8B training time: 32K Long CPT 22.4 h, Mix CPT 19.9 h, LongReD 22.4 h; 128K 42.8 h, 30.4 h, 33.4 h (App. B.4, Table 10).
- **Table anomalies.** Mix CPT scores near zero on SQuADv2 (0.89) and DROP (0.17) at 32K ABF (Table 16), which lowers its RC average (34.65, Table 3); the paper does not discuss this. Table 15 prints identical rows for Mistral Long CPT and Mix CPT and labels both LongReD rows "LongReD-C". Table 7 lists LongReD-C General as 67.51, Table 3 as 67.65.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LongReD Llama-3-8B, 32K ABF | 8B | long-context | RoPE base; lengths of D1 / D2 / D3 (tokens) | 2e7; 32000 / 1024 / 8192 | arXiv:2502.07365v3 App. B.2 Table 9 | verified 2026-09-14 | no ablation reported |
| LongReD Llama-3-8B, 32K PI | 8B | long-context | PI ratio; lengths of D1 / D2 / D3 | 4; 32000 / 1024 / 8192 | App. B.2 Table 9 | verified 2026-09-14 | no ablation reported |
| LongReD Llama-3-8B, 128K ABF | 8B | long-context | RoPE base; lengths of D1 / D2 / D3 | 1e8; 128000 / 1024 / 8192 | App. B.2 Table 9 | verified 2026-09-14 | no ablation reported |
| LongReD Mistral-7B-v0.3, 128K ABF | 7B | long-context | RoPE base; lengths of D1 / D2 / D3 | 2e7; 128000 / 1024 / 32768 | App. B.2 Table 9 | verified 2026-09-14 | no ablation reported |
| all LongReD runs | 7B, 8B | long-context | Tokens; steps; tokens per batch | about 1B; 512; about 2M | §5.1; App. B.3 | verified 2026-09-14 | no ablation reported |
| all LongReD runs | 7B, 8B | long-context | LR; optimizer | 2e-5 constant, no warmup; AdamW, weight decay 0.1, β1 0.9, β2 0.95 | App. B.3 | verified 2026-09-14 | no ablation reported |
| all LongReD runs | 7B, 8B | long-context | Token ratio of the three datasets | 4:3:1 (mapping to D1/D2/D3 not stated) | §5.1 | verified 2026-09-14 | no ablation reported |
| 32K runs | 8B | long-context | α1; α2 | 5; 10 | App. B.3 | verified 2026-09-14 | Table 4 (32K ABF): α1 = 2 gives RULER 86.03 vs 84.98 at α1 = 5 |
| 128K runs | 7B, 8B | long-context | α1; α2 | 2; 15 | App. B.3 | verified 2026-09-14 | no 128K ablation reported |
| LongReD Llama-3-8B, 128K | 8B | long-context | Distilled layers M | 3 | §5.1 | verified 2026-09-14 | "better balancing" stated; no table |
| other LongReD runs | 7B, 8B | long-context | Distilled layers M | 6 (largest attention KL; last layer included) | §5.1; §4.2 | verified 2026-09-14 | Table 5: KL(6) RULER 84.98 vs Uniform(6) 82.53 (32K ABF) |
| all LongReD runs | 7B, 8B | long-context | Short distillation length T_s | 1024 | §5.3; Table 9 | verified 2026-09-14 | Table 6: RULER 84.98 (1024) vs 83.20 (2048) vs 75.54 (8192) |
| all LongReD runs | 7B, 8B | long-context | D1 and D3 mixture (SlimPajama, Llama-2 proportions) | CommonCrawl 0.534, C4 0.266, GitHub 0.050, ArXiv 0.043, Book 0.041, Wikipedia 0.034, StackExchange 0.032 | App. B.1 Table 8 | verified 2026-09-14 | follows Fu et al. (2024); no ablation reported |
| all LongReD runs | 7B, 8B | long-context | D2 mixture | Fineweb-Edu 0.476, Fineweb-Edu-Math 0.722, Proof-Pile-2 0.230, Stack-v2 0.095, PG19 0.095, Wikipedia 0.096, ArXiv 0.095, Instructions 0.047 | App. B.1 Table 8 | verified 2026-09-14 as printed; the printed values sum to 1.856, so at least one value is misprinted | no ablation reported |
| all LongReD runs | 7B, 8B | long-context | D2 instruction sets | Tulu-v2-sft-mixture, MathInstruct, WizardLM-evol-instruct-V2, Magicoder-Evol-Instruct-110K | App. B.1 | verified 2026-09-14 | no ablation reported |
| all LongReD runs | 7B, 8B | long-context | Framework; hardware | EasyContext with ring flash attention; 8× A800 | App. B.3-B.4 | verified 2026-09-14 | not applicable |
| forgetting analysis (not LongReD) | 8B | long-context | RoPE base; batch size (unit not stated); length | 2e7; 64; 32K | §3.2 | verified 2026-09-14 | Fig. 3 |

Not reported (checked body, App. A-F): LR decay, gradient clipping, precision, seeds, number of sequences per batch, CREAM/uniform choice for released checkpoints.

## Findings relevant to generality, long context, distillation
- **Generality.** Short-text loss is measured on 17 benchmarks in 5 categories (App. C). Long-only CPT lowers the Llama-3-8B short-text average by 4.16 points at 32K ABF and the Mistral-7B-v0.3 average by 10.41 points at 128K (derived from Table 3). LongReD narrows the Mistral gap to 3.40-4.79 points but does not close it (derived, Table 3).
- **Scope limit stated by the authors.** The degradation is observed after continual pretraining on "only several billion tokens"; the authors note that Xiong et al. (2024) and Dubey et al. (2024), training on more than 100B tokens, report some short-text abilities unaffected or improved (Limitations).
- **Long context.** RULER trade-off remains at 128K on Llama-3-8B (above). PI underperforms ABF with LongReD on both short and long tasks; the authors attribute this to larger drift under PI (Interpretation, §5.2). CREAM is better at 4× extension and worse than uniform sampling at 128K (§5.2). Longer distillation length breaks the continuity of positional vectors inside and outside the original window (App. E, Fig. 5).
- **Distillation.** Stage: continual pretraining for context extension. Teacher: the original unextended model (Θo). Signal: cosine similarity of hidden states, not token distributions. Data: FineWeb-Edu, Stack-v2, Proof-Pile-2, PG19, arXiv, Wikipedia, and instruction data (App. B.1). No teacher sampling is involved. Distilling all layers or only the last layer gives lower RULER than KL-selected layers (Table 5).

## Connections
- [[long-context-data-engineering]] — Fu et al. (2024); its SlimPajama sampling is used for the long and short-to-long datasets.
- [[prolong]] — Gao et al. (2024); motivates using higher-quality data for short-text distillation.
- [[llama-2-long]] — ABF (Xiong et al., 2024), one of the two RoPE scaling methods.
- [[position-interpolation]] — PI, the other scaling method.
- [[pose-synthesis]] — skipped positional indices used in the short-to-long term.
- [[ring-attention]] — ring flash attention in EasyContext used for training.
- [[ruler]] — long-text evaluation.
- [[catastrophic-forgetting-continual-finetuning]], [[spurious-forgetting]] — other forgetting studies (not cited by this paper).
- [[gkd-on-policy-distillation]] — cited as divergence-based distillation, in contrast to the hidden-state similarity used here.
- [[longpo]] — preserves short-context ability at the alignment stage with a KL term to short-chunk outputs.
- [[longer-context-deeper-thinking]] — RoPE θ scaling lowers pre-SFT reasoning accuracy there (its Table 3).
- [[tulu-2]], [[fineweb]] — sources of instruction and short-text data in D2.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.07365 (arXiv v3, 28 May 2025; full PDF including App. A-F; abs page for version dates and "ACL2025 Main" comment).
- Audit claims not found in the source: "Mistral-7B-v0.3 MMLU ~65.4 to ~55.3 at 128K" → 65.36 → 55.29 is the General category average (MMLU, BBH, LAMBADA); MMLU is 62.30 → 51.90 (Table 3 vs Table 14). "Llama-3-8B MMLU 68.18 to 65.39" → also the General average; MMLU is 64.80 → 62.00. "Mistral-7B-v0.3 to 128K with ABF or PI" → Mistral uses ABF only (App. B.2). "4:3:1 long : short-distill : short-to-long" → §5.1 gives 4:3:1 for "these three datasets" without naming the order. "95.9% retention" in the audit note → the paper states 99.4% (§5.2). "LongReD keeps short-text scores near the original" does not hold for Mistral-7B-v0.3 (47.69 vs 51.09).
