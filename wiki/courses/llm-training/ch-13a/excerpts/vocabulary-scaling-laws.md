---
chapter: ch-13a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/vocabulary-scaling-laws.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2407.13623
created_at: "2026-09-15"
---

# Excerpt: Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies

**Authors:** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong (HKU, Sea AI Lab, Contextual AI, Stanford, Ohio State)
**Version read:** arXiv:2407.13623v3 (1 Nov 2024), NeurIPS 2024; full PDF text including App. A-B.
**Status:** no library card existed for this slug on 2026-09-15; every value below was read at the stated locus.

## Definitions (§2.2)
- Parameters split as N = N_nv + N_v, with N_v = V·d (output layer only; the paper uses V·d rather than 2V·d because the output layer dominates FLOPs, footnote 1). V: vocabulary size; d: embedding dimension, set from N_nv (App. A.7.2 Table 5).
- Data measured in training characters H; tokens D = H·f(V), where f(V) is the tokens-per-character ratio, fitted as a quadratic in log V: `f(V) = a·log²(V) + b·log(V) + c`, a = 0.0064, b = −0.1581, c = 1.2047, fitted on tokenizers with V from 1K to 1024K (Eq. 3). V is capped at 200K before computing f(V) (App. A.9).
- Vocabulary-insensitive loss (Eq. 4): `L_u = −(1/T) Σ_i log[ p(w_i | w_<i, V) / p(w_i | V) ]`, where p(w_i | V) is the unigram frequency of token w_i in the tokenized corpus. The paper reports that L_u correlates with bits per character (BPC) (§2.2, App. A.5).
- App. A.10 (Fig. 13): across models with fixed N_nv and d but different V, ordinary language-modeling loss correlates positively with the average of 7 downstream tasks (larger vocabularies have higher per-token loss and better downstream scores), while L_u correlates negatively, as expected of a loss.
- FLOPs: C ≈ 6·N·D ≈ 6·(N_nv + V·d)·H·f(V) (Eq. 5).

## Why an optimum exists (§3)
At small V, a larger vocabulary lowers f(V), so a fixed token budget covers more characters. At large V, the gain in f(V) shrinks while rare-token embeddings are undertrained with limited data (§3 Analysis 1). For a fixed FLOPs budget, loss first falls then rises with V, and the minimizing V grows with budget (§3 Analysis 2, Fig. 3).

## Three estimation approaches (§4)
- Approach 1 (IsoFLOPs): 6 groups with N_nv from 33M to 1.13B; V from 4K to 96K (4096, 6144, 8192, 10240, 16384, 24576, 32768, 48128, 64512, 96256; App. A.7.1). Fits: N_nv = 0.08·C^0.50, N_v = 0.20·C^0.42, H = 6.42·C^0.50 (§4.1, Fig. 5); γ = 0.42/0.50 = 0.84.
- Approach 2 (derivative of FLOPs with respect to V at fixed loss): γ = 0.83 (§4.2). Abstract and §1 give γ ≈ 0.83.
- Approach 3 (parametric loss): `L_u = −E + A1/N_nv^α1 + A2/N_v^α2 + B/D^β` with A1 = 1.831, A2 = 0.196, B = 2.124, E = 5.533, α1 = β = 0.447, α2 = 0.671 (Eq. 8, §4.3).
- Data: SlimPajama sampled uniformly over domains (§4.1); related-work text says models are pretrained "on English corpora" (§6). Training: Llama architecture, sequence length 2,048, max LR 4e-4 decaying to 4e-5, AdamW, bfloat16, global batch 512 (App. A.7.1, A.7.3).

## Predictions and validation (§5)
- Table 1 (compute-optimal allocation), V_opt by Approach 1 / 2 / 3: N_nv 3B → 39K / 43K / 37K; 7B → 62K / 67K / 60K; 13B → 83K / 91K / 81K; 30B → 142K / 154K / 142K; 70B → 212K / 231K / 218K; 300B → 356K / 389K / 383K.
- Abstract: the optimal vocabulary of Llama2-70B "should have been at least 216K, 7 times larger than its vocabulary of 32K".
- Table 2 (N_nv = 2.87B, 1.2e21 FLOPs, compute-optimal data): V = 32K average 47.9 vs V_opt = 35K average 48.5.
- Table 3 (N_nv = 2.87B): undertraining 2.8e20 FLOPs, 32K 43.2 vs 24K 43.9; overtraining 2.3e21 FLOPs, 32K 50.3 vs 43K 51.6, ARC-Challenge 29.1 ± 1.3 → 32.0 ± 1.4. The ± values are standard deviations of downstream accuracy; pretraining runs have no repeated seeds or error bars "due to their computational cost" (NeurIPS checklist item 7).
- Fig. 7 (N_nv = 302M): the best V moves from 16K to 10K when data is the bottleneck and from 16K to 24K with excess data. The authors still recommend the V that is optimal for N_nv under compute-optimal data because larger V raises inference cost (§5).
- Fig. 6: at their reported training tokens, all listed public models except Gemma2-9B have fewer vocabulary parameters than predicted.

## Limits stated (App. B)
Validated up to 3B parameters, dense transformers only (B.2). Multilingual extension is future work: "Different languages compete with each other for the model's ability", so vocabulary sizes should account for relations between languages (B.4). Related work notes that expanding vocabularies during continual pretraining can degrade low-resource languages (Dou et al., cited in App. A.12).

## How ch-13a uses it
§4.3 (vocabulary parameters, optimum and γ, V_opt table, ARC-Challenge result, English-only scope), §5 (vocabulary-insensitive loss and BPC).
