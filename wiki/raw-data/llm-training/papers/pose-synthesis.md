<!-- scope: long-context extension — PoSE positional skip-wise fine-tuning: simulate target-length position indices inside the original context window
     deps: [[long-context-data-engineering]]
     see-also: [[prolong]], [[longrope-data]], [[longalpaca]]
-->

# PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training
- **Core Insight:** Fine-tuning LLaMA-7B on 2,048-token windows whose position indices are split into chunks with random skip offsets gives GovReport perplexity 4.60 at a 16k evaluation window, against 4.59 for full-length 16k fine-tuning (Table 1), and with YaRN interpolation the same recipe reaches 128k (PG-19 perplexity 11.33 at 128k, Table 2).
- **Guideline:** When the target context length is too long to fine-tune on directly, PoSE with N = 2 chunks and linear or YaRN interpolation is an alternative to full-length fine-tuning, because for LLaMA-7B extended to 16k its perplexity was never more than 0.07 above full-length fine-tuning on GovReport and Proof-pile at 2k-16k evaluation windows (Table 6). For 128k, use YaRN rather than linear interpolation, because PoSE-Linear-128k lost 11.71 HellaSwag points against the original LLaMA while PoSE-YaRN-128k lost 2.21 (Table 3).
- **Authors:** Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li (Peking University; Microsoft Corporation)
- **Year:** 2023 (arXiv v1 2023-09; ICLR 2024)
- **URL:** https://arxiv.org/abs/2309.10400 (code: https://github.com/dwzhu-pku/PoSE)
- **Source type:** paper
- **Relevant topics:** context window extension, RoPE, position interpolation (Linear, NTK, YaRN), compute-efficient long-context fine-tuning, passkey retrieval

## Abstract
LLMs are trained with a fixed context length, and adapting them to a longer length usually requires fine-tuning at the target length, which is costly. PoSE decouples training length from target length. It divides the original context window into chunks and adds a distinct skipping bias to the position indices of each chunk; the biases and chunk lengths are re-sampled for every training example so the model is exposed to all positions within the target length. PoSE reduces memory and time relative to full-length fine-tuning with minimal performance impact, extends LLaMA to 128k tokens with a 2k training window, and is reported to work with RoPE-based LLMs and several position interpolation strategies.

## Key Contributions
- Positional skip-wise training: chunked position indices with random skip biases inside a fixed training window (§3.2, Eq. 3-4).
- Memory and time for PoSE stay fixed as the target length grows from 4k to 16k, while full-length fine-tuning cost rises (§5.1, Fig. 3).
- Tested on LLaMA-7B, LLaMA2-7B, GPT-J-6B, Baichuan2-7B with Linear, NTK, and YaRN interpolation (§5.2, Fig. 4).
- Extension to 96k (Linear) and 128k (YaRN) (§5.3, Table 2).

## Key Figures/Tables to Study
- **Fig. 1:** position indices for full-length versus PoSE fine-tuning (2,048 → 8,192 example).
- **Table 1 / Table 6:** sliding-window perplexity for Full-length, RandPos, PI-only, and PoSE.
- **Fig. 2:** passkey retrieval accuracy by prompt length.
- **Table 3:** short-context benchmarks after extension.
- **Fig. 5 and Table 5 (App. B):** coverage of relative positions and perplexity versus chunk number.

## Technical Details
- **Problem:** extend a model pre-trained with context L_c to target length L_t (§3.1). α = L_t / L_c is the interpolation scale factor.
- **Chunking:** the window L_c is randomly divided into N chunks c_0 … c_{N−1} with lengths l_i, Σ l_i = L_c; the start index is st_i = Σ_{j<i} l_j (Eq. 3).
- **Skip bias:** u_i ∼ U({u_{i−1}, …, L_t − L_c}) and PoSE(c_i) = {u_i + st_i, …, u_i + st_i + l_i − 1} (Eq. 4). The constraint u_i ≥ u_{i−1} prevents overlapping indices; u_0 = 0 (§3.2).
- **Chunk content:** c_i = x[v_i + st_i : v_i + st_i + l_i] with v_i ∼ U({v_{i−1}, …, L_x − L_c}), v_0 = 0, where L_x is the document length (Eq. 5). Using v_i = 0 or v_i = u_i instead changed perplexity by at most 0.08 on GovReport and Proof-pile at 2k-16k windows (App. A, Table 4).
- **Interpolation:** position interpolation is applied after index manipulation; linear interpolation is the default (§3.2, §4.1). Attention is not modified (§2).
- **Chunk number:** N = 2 in all main experiments. More chunks raise coverage of relative positions in [2,048, 16,383] (Fig. 5); Proof-pile 16k perplexity is 2.60 (N = 2), 2.59 (N = 3), 7.73 (N = 2,048, equal to RandPos) (App. B, Table 5).
- **Language modeling (Table 1, linear, eval window 16k):** GovReport Full-length 4.59, PoSE 2k/16k 4.60, RandPos 2k/16k 15.16; Proof-pile 2.53, 2.60, 7.73. PoSE 2k/32k at 32k: 4.66 (GovReport), 2.59 (Proof-pile). The original LLaMA exceeds 10^3 beyond 2k.
- **Interpolation variants at 16k (Table 6):** with NTK, GovReport perplexity is 7.24 for both PoSE and Full-length; the authors attribute the rise to NTK not extending the window by the full factor α (App. C). With YaRN, 4.55 (PoSE) versus 4.53 (Full-length).
- **Passkey retrieval:** 50 trials per length with a random 5-digit passkey; PoSE-16k and PoSE-32k keep ≥ 90% accuracy within their target windows, while Original, PI-only, and RandPos drop to 0 beyond 2k (§4.3, Fig. 2).
- **Efficiency:** measured for 4k, 8k, 16k targets, 1,000 steps, global batch 16, 8 V100; full-length at 32k or above was not run because V100s could not fit it (§5.1, Fig. 3). The paper gives these as plots; no numeric speed-up ratio is stated.
- **96k and 128k (Table 2), perplexity at 128k:** PG-19 Linear 31.18, NTK 34.80, YaRN 11.33; Books3 Linear 70.87, NTK 37.00, YaRN 13.81. At 96k, PoSE-Linear-96k gives 13.57 (PG-19) and 13.42 (Books3). The authors conclude Linear works to 96k and YaRN to 128k (§5.3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| PoSE-extended LLaMA-7B (16k, 32k main runs) | 7B | long-context | training window; target lengths | 2,048 tokens (2k); 16k, 32k | arXiv:2309.10400v3 Abstract, §4.1, Table 1, Fig. 1 | verified 2026-09-14 | no ablation of training window length reported |
| same | 7B | long-context | steps; global batch (sequences) | 1,000; 64 | §4.1 | verified 2026-09-14 | no ablation reported |
| same | 7B | long-context | tokens seen | 131,072,000 = 1,000 × 64 × 2,048 | §4.1 inputs | derived | none |
| same | 7B | long-context | LR; schedule; warmup; optimizer | 2e-5; linear; 10 steps; AdamW with default hyperparameters | §4.1 | verified 2026-09-14 | no ablation reported |
| same | 7B | long-context | data | The Pile, documents ≥ 2,048 tokens | §4.1 | verified 2026-09-14 | no ablation reported |
| same | 7B | long-context | chunks N; skip bias; content bias | 2; u_1 ∼ U({0, …, L_t − L_c}); v_1 ∼ U | §3.2, Eq. 4-5 | verified 2026-09-14 | App. B Table 5: N = 3 gives 2.59 vs 2.60 (Proof-pile 16k), N = 2,048 gives 7.73; authors keep N = 2 to stay close to the pre-training position structure (§3.2); App. A Table 4 for v_i |
| same | 7B | long-context | interpolation; hardware | linear (default); 8 V100, DeepSpeed ZeRO stage 3 | §4.1 | verified 2026-09-14 | no ablation selected linear; §5.2 reports NTK and YaRN generally lower perplexity than linear, with NTK rising before the target length |
| PoSE efficiency runs, LLaMA-7B (4k/8k/16k) | 7B | long-context | steps; global batch | 1,000; 16 | §5.1 | verified 2026-09-14 | not applicable |
| PoSE LLaMA 96k and 128k (Linear/NTK/YaRN) | not stated | long-context | steps, batch, LR | not reported | checked §5.3, Table 2, §6 | not reported | §6 states the 128k extension ran on 8 V100 GPUs |
| all PoSE models | — | eval-gate | perplexity evaluation | sliding window, stride 1,024 (≤ 32k); step 16k (96k/128k); 50 docs per dataset ≥ 32,768 tokens; 20 books ≥ 128k | §4.2; §5.3 | verified 2026-09-14 | not applicable |

## Findings relevant to generality and long context
- **Short-context retention (Table 3):** PoSE-YaRN-16k: BoolQ 74.28, PIQA 78.02, WinoGrande 69.06, TruthfulQA 34.00, ARC-C 49.23, HellaSwag 77.04, against original LLaMA 75.11, 78.67, 69.85, 34.08, 51.19, 77.75. PoSE-Linear-128k drops to ARC-C 39.93 and HellaSwag 66.04; the authors call this the only exception to marginal degradation (§5.4).
- **Length trade-off:** all scaling methods show some degradation as the supported context length grows (for example PoSE GovReport perplexity at a 2k window is 4.84 for the 16k model and 4.91 for the 32k model, Table 1); the authors interpret this as a trade-off between the number of tokens and attention granularity per token (§4.2, Interpretation).
- **What is measured:** long-context ability is tested with perplexity and synthetic passkey retrieval only; no long-document QA, summarization quality, or reasoning benchmark is reported.

## Connections
- [[long-context-data-engineering]]: continued pre-training on long documents at the target length, which is the full-length setting PoSE is designed to avoid.
- [[prolong]]: long-context data mixture and token budgets for full-length long-context training.
- [[longrope-data]]: another RoPE-rescaling approach to context extension.
- [[longalpaca]]: LongLoRA, which PoSE cites as still requiring full-length fine-tuning (§2).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2309.10400 (v3, 2024-02-21; v1 2023-09-19)
- Corrections to the previous card version:
  - "each 4K-token sample" and "δ ∈ [0, target − 4K]" → training window 2,048 tokens; u_1 ∼ U({0, …, L_t − L_c}) with L_c = 2,048 (Abstract, §3.2, §4.1).
  - "split into two chunks at a random position" → N chunks with re-sampled lengths and biases; N = 2 in experiments (§3.2).
  - "Source: Pile, SlimPajama" → The Pile only, documents ≥ 2,048 tokens (§4.1).
  - "Target extensions 16K, 64K, 128K" → 16k and 32k (main), 4k/8k/16k (efficiency), 96k and 128k (§4-5).
  - "RoPE base rescaling (standard NTK-aware)" → linear interpolation by default; NTK and YaRN also tested; YaRN gave the lowest 128k perplexity (§4.1, §5.3, Table 2).
  - "competitive NIAH" → passkey retrieval ≥ 90% for the 16k and 32k models only (§4.3, Fig. 2).
  - "Short-context ppl preserved" → benchmark accuracy degrades marginally except PoSE-Linear-128k (Table 3).
  - "Attention mask allows chunk 1 tokens to attend to chunk 2 according to positions" → PoSE changes position indices only and does not modify attention (§2).
- Removed as unsupported by the source: "~4× less compute"; "~1/4 of full-length compute"; "~1K A100-hours for 128K"; "NIAH 128K ~90%"; "perplexity within 0.1 of full-length baseline" for the 128k model (no full-length 128k run exists); "PoSE-32K outperforms LongLoRA-32K"; "pretrained checkpoints"; "weaker on reasoning, BABILong lags"; "wrong base-θ rescaling breaks PoSE"; "does not learn semantic dependency across 100K tokens".
- Not reported by the source: training steps, batch, or LR for the 96k/128k runs; any long-context task benchmark beyond passkey retrieval.
