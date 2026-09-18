<!-- scope: Princeton study of long-context continued training and SFT for Llama-3-8B (evaluation protocol, long/short data mix, training length, RoPE base, document masking, SFT data) and the ProLong 512K model
     deps: [[long-context-data-engineering]]
     see-also: [[prolong-recipe]], [[long-context-llama3]], [[longalign]], [[ruler]], [[sequence-packing]], [[qwen-long-context-synth]]
-->

# How to Train Long-Context Language Models (Effectively)
- **Core Insight:** Continued training of Llama-3-8B-Instruct on 40B tokens (20B at 64K, 20B at 512K) of code repositories, books, textbooks, and short data, followed by 1B tokens of short UltraChat SFT, gives ProLong a HELMET average of 49.4 at 128K, above Llama-3.1-8B-Instruct at 46.5, which used about 800B long-context tokens (§6.1–6.2, Tables 9–10).
- **Guideline:** When extending a pre-trained LM's context by continued training, keep about 40% high-quality short data in the mix and choose design options by downstream long-context tasks evaluated after SFT, because in 5B-token Llama-3-8B ablations 100% long data kept improving PG19 perplexity while lowering downstream long-context scores, and short-task scores fell monotonically as the long share rose (§2.1 Fig. 1; §3.2 Fig. 3).
- **Authors:** Tianyu Gao, Alexander Wettig, Howard Yen, Danqi Chen (Princeton Language and Intelligence)
- **Year:** 2024 (arXiv v1 2024-10; v4 2025-12; ACL 2025)
- **URL:** https://arxiv.org/abs/2410.02660 (code, data, models: https://github.com/princeton-nlp/ProLong)
- **Source type:** paper
- **Relevant topics:** long-context continued pretraining, data mixture, short-context retention, RoPE frequency base, document masking, long-context SFT, long-context evaluation

## Abstract
The paper studies continued training and supervised fine-tuning (SFT) of a language model so that it uses long-context information. It first sets an evaluation protocol: a broad set of long-context downstream tasks instead of perplexity or needle-in-a-haystack (NIAH), measured after SFT. With this protocol it runs experiments on the continued-training data mix, the instruction-tuning data, and design choices such as position extrapolation. Findings: (1) code repositories and books are good long-data sources, but they must be combined with high-quality short data; (2) training on sequences longer than the evaluation length improves long-context performance; (3) SFT on short instruction data alone gives strong long-context results. The final model, ProLong-8B, is initialized from Llama-3, trained on 40B tokens, and is state of the art among similar-size models at 128K. It outperforms Llama-3.1-8B-Instruct on most long-context tasks with 5% of the long-context training tokens and processes up to 512K tokens.

## Key Contributions
- Evaluation protocol: a HELMET subset (JSON key-value recall, RAG, re-ranking, many-shot ICL, book QA, summarization) averaged over 32K and 64K, run after SFT, plus 5 short-context tasks (§2; App. A.1, Table 13).
- Data curation findings from 5B-token ablations: long data from books and code repositories at 60%, short data from a curated ShortMix at 40% (§3).
- Scaling findings: more tokens help, and training at 512K improves results evaluated at 64K (§4).
- SFT finding: short UltraChat data alone outperforms mixes with synthetic long instruction data (§5).
- ProLong model family, training data, and code released (§6; footnote 1; GitHub README).

## Key Figures/Tables to Study
- Fig. 1 (perplexity vs downstream score), Fig. 2 (before vs after SFT), Fig. 3 (long/short ratio).
- Tables 4 and 6 (long and short data sources), Table 7 (training length), Table 8 (synthetic SFT data).
- Table 9 (full recipe; ledger in [[prolong-recipe]]), Table 10 (HELMET at 128K), Table 25 (short-context after SFT).

## Technical Details
- **Units.** K = 2^10, M = 2^20, B = 2^30 throughout (footnote 2).
- **Ablation setting.** Llama-3-8B base, 5B tokens at 64K, Table 9 hyperparameters, evaluated after UltraChat SFT (§3; App. A.4). Long data is single-document 64K chunks; short data is packed to 64K (§3).
- **Why not perplexity.** As the long-data share rises to 100%, PG19 perplexity keeps improving but the downstream long-context average drops (§2.1, Fig. 1).
- **Why after SFT.** Before SFT, RAG and re-ranking gains over training tokens are unclear or absent; after SFT they appear, and QA and summarization become measurable (§2.2, Fig. 2).
- **Short-context loss from prior methods.** Relative to Llama-3-8B (MMLU 66.5, GSM8K 44.7), RoPE-base position extrapolation gives 64.7 / 40.1 and fine-tuning on the SlimPajama long mix of Fu et al. (2024) gives 63.1 / 40.6 (§2.3, Table 2).
- **Long data availability.** Documents ≥ 64K tokens: code repositories from the Stack 98.8B tokens; SlimPajama books 33.2B, CommonCrawl 15.3B, ArXiv 5.2B (§3.1, Table 3). A code-repository document concatenates all files of a repository without dependency ordering (§3.1).
- **Long data source (60% long + 40% ShortMix).** Long-context average: books/repos 1:1 54.6, books 53.8, books/repos/ArXiv 1:1:1 53.3, code repos 52.3, ArXiv 51.4, CommonCrawl 50.9; code repos alone give the highest recall, 99.2 (Table 4).
- **Short data source (40%).** Long / short average: ShortMix 54.6 / 65.5, SlimPajama 52.9 / 64.2, FineWeb-Edu 53.0 / 63.0, DCLM-Baseline 52.0 / 64.8; original Llama-3-8B short average 66.0 (Table 6).
- **Training length.** From the 20B-token 64K checkpoint, 4B more tokens at 512K vs at 64K (both evaluated at 64K): recall 98.5 vs 95.0, RAG 56.9 vs 56.4, re-rank 32.9 vs 28.0, ICL 79.2 vs 78.8 (§4, Table 7). Longer sequences cost more compute and may not be compute-optimal (footnote 4).
- **RoPE base.** At 64K: base 5×10^5 → 29.1, 4×10^6 → 48.7, 8×10^6 → 54.6. At 512K: 6.4×10^7 → 57.6, 1.28×10^8 → 57.7, 2.56×10^8 → 57.8 (App. B.1, Tables 18–19).
- **Document masking.** Masking attention across document boundaries: long 54.6 / short 65.5 vs 53.6 / 64.9 without masks (App. B.2, Table 20). Variable-length attention with minibatch reordering raises throughput from 2770 to 3095 tokens/s/GPU on 8 H100s (App. A.3, Table 15).
- **SFT data.** UltraChat conversations average 1.2K tokens (max 4.1K) (§5). UltraChat 55.7, Tulu v2 49.1, ShareGPT 45.2 (Table 22). Synthetic long data (40% QA, 30% RAG, 30% summarization, generated by Llama-3-8B-Instruct) mixed at 0/1/3/10/50% of tokens gives 55.7/54.1/53.5/53.9/43.3 (Table 8). A Llama-3-70B-Instruct generator gives 54.2/55.4/53.6/49.7 at 1/3/10/50% (App. B.5, Table 23).
- **Final results.** HELMET at 128K average: ProLong 49.4, Llama-3.1-8B 46.5, MegaBeam-Mistral-7B 45.4, Jamba-1.5-Mini 46.9, Llama-3.1-70B 49.7, GPT-4o 64.8 (Table 10). At 512K input, QA 49.7 and summarization 42.1 vs 31.7 and 40.4 at 32K (Table 11). NoCha <75K accuracy 28.4, the only 10B-scale model above the 25% random baseline (Table 12). RULER 71.9 vs Llama-3.1-8B 81.3; ∞Bench 48.0 vs 46.7 (App. B.8, Table 26).

## Recipe ledger
The full ledger (27 rows: stage 1, stage 2, SFT, ablation settings, released scripts, one learning-rate conflict) is in [[prolong-recipe]]. Headline values from arXiv v4 Table 9, verified 2026-09-14: initialization Llama-3-8B-Instruct; stage 1 20B tokens at 64K, RoPE base 8×10^6, batch 4M tokens, 2.2K H100 hours; stage 2 20B tokens at 512K, RoPE base 1.28×10^8, batch 8M tokens, 12.2K H100 hours; LR 1e-5 with 10% warmup and cosine decay to 1e-6 per stage (released `train_512K.sh` default: 5e-6); SFT 1B tokens UltraChat, LR 2e-5, batch 4M tokens.

## Findings relevant to generality, long context, distillation
- **Generality (short-context retention).** Short-task average decreases monotonically as the long-data share increases (§3.2, Fig. 3). After SFT, ProLong averages 69.4 on five short tasks vs 69.8 for Llama-3-8B-Instruct, but is lower on MMLU (64.6 vs 67.0) and GSM8K (58.9 vs 68.5); the authors attribute this to Llama-3-8B-Instruct's closed instruction data (App. B.7, Table 25).
- **Generality (measurement).** Development used a HELMET subset; the final model is also evaluated on held-out HELMET tasks, NoCha, RULER, and ∞Bench to address overfitting to development tasks (§2.1; Limitations). Results are limited to the 10B scale and Llama-3 models (Limitations).
- **Long context.** Only-long training hurts after SFT on recall and RAG even when it helps before SFT; the authors hypothesize that a long-only model is a poor initialization for generic SFT (§3.2).
- **Distillation / synthetic data.** Synthetic long instruction data did not help in this setting, including with a 70B generator (§5; App. B.5). The authors' hypotheses: prior work may have had too little long-context continued training, or much larger short instruction sets (§5).

## Connections
- [[long-context-data-engineering]]: Fu et al. (2024) is reproduced head-to-head with equal initialization, 5B tokens, and hyperparameters: long avg 51.8 vs 54.6, short avg after SFT 65.4 vs 67.5 (App. B.6, Table 24).
- [[long-context-llama3]]: Llama 3.1's 800B long-context tokens are the comparison for the 5% budget claim; Llama 3.1 also used synthetic long SFT data (§5, §6.2).
- [[longalign]]: cited as prior work that adds synthetic long instruction data at SFT; ProLong's SFT results differ (§5, §7).
- [[ruler]]: reported in App. B.8; the authors state RULER and ∞Bench have narrow coverage and noisy metrics.
- [[sequence-packing]]: related to the document-masking and variable-length attention choices (App. A.3, B.2).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2410.02660 (v4; v1 Table 9 and title identical), github.com/princeton-nlp/ProLong README and training scripts at commit 499fa29.
- Corrections to the previous card version:
  - "20B tokens of continued pretraining + 5B SFT" → 40B tokens (20B at 64K + 20B at 512K) + 1B tokens SFT (Table 9).
  - "Llama-3-8B base, extended from 8K" → final model initialized from Llama-3-8B-Instruct; ablations use the base model (§6.1; App. B.3).
  - "long-document quality/coherence dominates; concatenating short texts is strictly worse" → the paper's main data finding is that long data must be mixed with high-quality short data; 100% long data hurts (§3.2). No coherence filter is described.
  - "Final mix 30B: code 40%, books 25%, academic 15%, forums 10%, web 10%; web discarded; code ×4, books ×2" → 30% code repos, 30% books, 3% textbooks, 37% ShortMix (Table 9).
  - "RoPE base 500K → 128M; 64K first half, 512K second half" → 8×10^6 at 64K, then 1.28×10^8 at 512K (Table 9).
  - "LR 1e-4 → 1e-5 cosine" → 1e-5 with 10% warmup, cosine to 1e-6 (Table 9).
  - "no packing; one document per example" → short data packed to 64K with cross-document attention masking (§3; App. A.2, B.2).
  - "SFT: 70% long-instruction (Claude-3/GPT-4) + NIAH + UltraChat" → UltraChat only; synthetic data generated by Llama-3-8B/70B-Instruct was tested and not used (§5).
  - "~200K H100-hours" → 2.2K + 12.2K H100 hours for the two continued-training stages (Table 9).
  - "HELMET (512K): leading open 8B" → the main HELMET comparison is at 128K (Table 10); 512K results are QA and summarization only (Table 11).
- Removed as unsupported by the source: "ProLong-Data-30B, Apache-2.0"; "source pool: Llama-3 pretraining mix ~15T tokens"; "full-book PDFs parsed with structural fidelity"; "README → source → tests ordering" (the paper states no dependency ordering); "MMLU/GSM8K within 0.5 point of Llama-3.1-8B-Instruct"; "10+ HELMET points lost with concatenated short docs"; "multi-needle NIAH training samples"; "coherence filter is labor-intensive"; "code-heavy mix biases toward code tasks"; "20B budget is base-specific".
- Internal inconsistency in the source: §6.2 says ProLong beats Llama-3.1-8B-Instruct on all categories except summarization, but Table 10 shows ProLong higher on summarization (29.2 vs 27.0) and lower on Cite (1.4 vs 2.9). §6.1 text gives the original Llama RoPE base as 10^5, while Table 9 and App. B.1 give 5×10^5 for Llama-3.
- Not reported by the source: checkpoint selection rule; whether mixture percentages are token shares or sampling weights.
