<!-- scope: Qwen Team (arXiv:2501.15383, Jan 2025) — Qwen2.5-1M: progressive long-context pretraining to 262,144 tokens with synthetic long-range tasks, agent-generated long SFT data, two-stage SFT, short-sample offline RL, DCA + YaRN extrapolation to 1M, MInference sparse prefill
     deps: [[qwen-2.5]]
     see-also: [[long-context-llama3]], [[longalign]], [[ruler]], [[yarn]], [[llama-2-long]]
-->

# Qwen2.5-1M Technical Report
- **Core Insight:** Progressive pretraining of Qwen2.5-14B-1M from 32,768 to 262,144 tokens raised its RULER accuracy at 128K from 37.6 to 87.6 (Table 2), and training-free Dual Chunk Attention with YaRN attention scaling extends the 262K-trained models to 1M-token inference (§5.1).
- **Guideline:** When extending a dense Qwen2.5-class model toward 1M tokens, the report supports training to a longer length than the evaluation target (the 262K stage still raised RULER-128K from 83.8 to 87.6; Table 2) and extrapolating the rest at inference with DCA + YaRN; re-check short benchmarks afterwards, because the report calls them "similar" but some drop (14B GPQA 45.5 → 39.9; Table 6).
- **Authors:** An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, et al. (28 authors, alphabetical by last name; Qwen Team, Alibaba Group)
- **Year:** 2025 (arXiv v1 2025-01)
- **URL:** https://arxiv.org/abs/2501.15383
- **Source type:** official technical report
- **Relevant topics:** long-context pretraining, RoPE base frequency, synthetic long-range data, long-context SFT, offline RL, length extrapolation, DCA, sparse attention, short-context retention

## Abstract
Qwen2.5-1M extends Qwen2.5 to a 1M-token context through long-context pretraining and post-training. The training uses long-data synthesis, progressive pretraining, and multi-stage supervised fine-tuning to improve long-context performance at lower cost. The released inference framework includes a length-extrapolation method that expands context by at least four times without extra training, sparse attention with chunked prefill and a sparsity-refinement method, and engine optimizations (kernels, pipeline parallelism, scheduling), for a 3x to 7x prefill speedup at 1M tokens. The series includes the open Qwen2.5-7B-Instruct-1M and Qwen2.5-14B-Instruct-1M and the API model Qwen2.5-Turbo. Long-context results improve without loss on short-context tasks, and Qwen2.5-14B-Instruct-1M outperforms GPT-4o-mini on long-context tasks with an 8× longer context (abstract).

## Key Contributions
- Synthetic pretraining tasks for long-range dependency: fill-in-the-middle, keyword- and position-based paragraph retrieval, paragraph reordering (§3).
- A five-stage context schedule with a RoPE base increase at each extension (§3, Table 2).
- Long instruction data from Qwen2.5-written queries and Qwen-Agent-written answers; two-stage SFT; offline RL on short pairs only (§4).
- DCA + YaRN attention scaling for extrapolation; MInference sparse attention integrated with chunked prefill, DCA, and sparsity refinement on 1M-token sequences (§5.1-5.2).
- Open weights for the 7B and 14B Instruct-1M models under Apache 2.0 (Table 1); inference code integrated into vLLM (§5).

## Key Figures/Tables to Study
- Table 2: RULER of Qwen2.5-14B-1M after each pretraining stage.
- Table 3: LongBench-Chat before and after the short-sample RL stage.
- Figure 3: 1M-token retrieval tasks with and without DCA, 128K versus 1M models.
- Tables 4-6: RULER, LV-Eval, LongBench-Chat, and short-context benchmarks against the 128K Qwen2.5 models.

## Technical Details
- **Models.** Qwen2.5-7B-1M: 28 layers, 28 query / 4 KV heads; Qwen2.5-14B-1M: 48 layers, 40 / 8; both 1M context / 8K generation (Table 1). Architecture is unchanged from Qwen2.5 (GQA, SwiGLU, RoPE, QKV bias, RMSNorm) (§2). Qwen2.5-Turbo is an MoE API model (§1).
- **Pretraining data.** Natural long text from domains including Common Crawl, arXiv, books, and code repositories, augmented with the three synthetic tasks above, because natural text "often exhibits weak long-distance associations" (§3). Proportions are not given.
- **Context schedule.** Stages 1-2 reuse an intermediate Qwen2.5 base checkpoint: 4,096 then 32,768 tokens, with Adaptive Base Frequency raising the RoPE base from 10,000 to 1,000,000. Stages 3-5 use 65,536, 131,072, and 262,144 tokens with RoPE bases 1,000,000, 5,000,000, and 10,000,000. In stages 3-5, data is 75% sequences at the current maximum length and 25% shorter sequences (§3).
- **Stage results (Qwen2.5-14B-1M, RULER avg / 128K).** After 32,768: 82.3 / 37.6; 65,536: 86.8 / 56.0; 131,072: 92.5 / 83.8; 262,144: 92.7 / 87.6 (Table 2).
- **Long SFT data.** Long documents from the pretraining corpus; Qwen2.5 writes queries from a randomly extracted segment (summarization, information retrieval, multi-hop QA, reasoning, coding, others); the Qwen-Agent framework writes responses over the full document using retrieval-augmented generation, chunk-by-chunk reading, and step-by-step reasoning (§4). Example counts are not given.
- **Two-stage SFT.** Stage 1: short instructions up to 32,768 tokens, same number of training steps as Qwen2.5. Stage 2: mixed short and long data, from up to 32,768 to up to 262,144 tokens, with a balanced short:long ratio whose value is not given (§4).
- **RL.** Offline RL similar to DPO, reusing other Qwen2.5 models' offline-RL pairs, all at most 8,192 tokens (§4). LongBench-Chat before → after: 7B 7.32 → 8.08 (printed +0.75), 14B 8.56 → 8.76, Turbo 7.60 → 8.34 (Table 3).
- **DCA.** The sequence is split into chunks and relative positions are remapped so no query-key distance exceeds the pretraining length: intra-chunk attention keeps original positions, inter-chunk attention uses repeated position sequences, successive-chunk attention keeps original positions within a local window (§5.1, Fig. 2).
- **YaRN attention scaling.** Attention is softmax(qᵀk / (t√D)) with √(1/t) = 0.1 ln(s) + 1, where s is inference length divided by training length and D the head dimension (Eq. 1). All experiments use it together with DCA; neither changes behavior on sequences within the training length (§5.1).
- **Extrapolation evidence.** With DCA, Qwen2.5-7B-Instruct and 14B-Instruct (trained to 32K) exceed 80% on Passkey Retrieval up to 1M tokens; the 1M models extrapolate better than their 128K versions (§5.1, Fig. 3).
- **Sparse prefill.** At 1M tokens attention can exceed 90% of forward time (§5.2). Chunked prefill with 32,768-token chunks cuts activation memory by 96.7% (§5.2). Unrefined MInference drops 7B needle retrieval to 60% or lower beyond 400K tokens; continuous positions for critical-token selection plus sparsity refinement recover most of it at about 4× prefill speedup (§5.2, Fig. 6). The full system gives 3.2-6.7× TTFT speedup at 1M; 14B-Instruct-1M on H20 goes from 12.2 minutes to 109 seconds (§6.3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1M ("Qwen2.5-1M models", §3) | 7B, 14B | pretrain-stable | sequence length; RoPE base | 4,096 → 32,768 tokens; ABF base 10,000 → 1,000,000 | arXiv:2501.15383v1 §3 | verified 2026-09-14 | Table 2: 32,768 stage gives RULER-128K 37.6 (14B) |
| Qwen2.5-1M | 7B, 14B | long-context | stage lengths; RoPE bases | 65,536 / 131,072 / 262,144 tokens; 1,000,000 / 5,000,000 / 10,000,000 | §3 | verified 2026-09-14 | Table 2 (14B): RULER-128K 56.0 / 83.8 / 87.6 |
| Qwen2.5-1M | 7B, 14B | long-context | length mix | 75% at current max length, 25% shorter (share type not stated) | §3 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-1M | 7B, 14B | long-context | data mix; tokens per stage; LR; batch; steps | not reported (checked §2-4, Tables 1-3; report has no appendix) | — | not reported | — |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | SFT | stage 1 | short instructions ≤ 32,768 tokens; same step count as Qwen2.5 (number not given) | §4 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | SFT | stage 2 | short + long, up to 262,144 tokens; short:long ratio not reported | §4 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | SFT | long-data generators | queries: Qwen2.5; responses: Qwen-Agent (RAG, chunk-by-chunk reading, step-by-step reasoning); counts not reported | §4 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B/14B-Instruct-1M, Qwen2.5-Turbo | 7B, 14B, MoE | preference | algorithm; data length | offline RL similar to DPO; reused Qwen2.5 pairs ≤ 8,192 tokens; β and pair count not reported | §4 | verified 2026-09-14 | Table 3: LongBench-Chat +0.75 / +0.20 / +0.74 |

## Findings relevant to generality and long context
- **Longer training helps shorter lengths.** The 262,144-token stage still raised RULER at 128K (83.8 → 87.6) for the 14B model (Table 2).
- **Short-to-long transfer in RL.** Offline RL on pairs of at most 8,192 tokens improved LongBench-Chat (max 100K tokens) for all three models (Table 3; §6.1). The report reads this as RL generalizing from short to long context (§4, Interpretation).
- **Short-context retention (Table 6, 128K model → 1M model).** 14B: MMLU-Pro 63.7 → 63.3, GSM8K 94.8 → 94.8, MATH 80.0 → 79.5, GPQA 45.5 → 39.9, LiveCodeBench 42.6 → 38.6, IFEval 81.0 → 84.3. 7B: MMLU-Pro 56.3 → 54.3, MATH 75.5 → 72.9, MBPP 79.2 → 75.9, Arena-Hard 52.0 → 48.1, GPQA 36.4 → 41.4. The report summarizes these as similar to the 128K versions (§6.2).
- **Long-context results.** RULER at 128K: 14B-Instruct-1M 92.2, 7B-Instruct-1M 84.4, Turbo 84.5, GPT-4o-mini 65.8, Llama-3.1-70B-Instruct 66.6 (Table 4). LV-Eval at 256K: 14B-Instruct-1M 43.3, 7B-Instruct-1M 42.7 (Table 5). Qwen2.5-72B-Instruct (trained to 32K) with DCA + YaRN outperformed 14B-Instruct-1M on LV-Eval at all lengths (§6.1). Passkey Retrieval at 1M: 14B-Instruct-1M and Turbo perfect, 7B minor errors (§6.1, Fig. 1).
- **Measurement.** RULER covers at most 128K and LongBench-Chat at most 100K tokens; only Passkey Retrieval and the Figure 3/6 retrieval tests reach 1M (§5, §6.1). The report modified LV-Eval matching rules to avoid false negatives (§6.1).

## Connections
- [[qwen-2.5]] — base models and short-context post-training reused here (§2, §4).
- [[long-context-llama3]], [[longalign]] — cited as inspiration for synthetic long instruction data (§4); LongAlign is also the LongBench-Chat source (§6.1).
- [[llama-2-long]] — source of Adaptive Base Frequency (Xiong et al., 2023) used in §3.
- [[yarn]] — attention scaling used with DCA (§5.1); [[ruler]] — main long-context benchmark (§6.1).
- [[needle-in-haystack-data]] — Kamradt test used for the sparse-attention evaluation (§5.2).
- [[qwen-3]] — later Qwen family; its long-context stage is recorded in that card, not in this report.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2501.15383 (arXiv v1, 2025-01-26).
- Corrections: title "Qwen Long-Context Data Pipeline (Qwen 2.5-1M and Qwen 3)" → this report only (the Qwen3 report is a separate artifact); "3 stages 32K → 128K → 256K" → 5 stages 4,096 → 32,768 → 65,536 → 131,072 → 262,144 with RoPE bases 10K → 1M → 1M/5M/10M (§3); "SFT data generated by Qwen-Max" → queries by Qwen2.5, responses by Qwen-Agent (§4); "FILL-IN SFT task" → FIM is a pretraining synthetic task (§3); "DCA between-chunk uses a low-rank formulation" → DCA remaps relative positions (§5.1); "YaRN for base extension during training" → training uses ABF base changes; YaRN attention scaling is used at inference with DCA (§3, §5.1); "RULER 1M ~85%" → RULER stops at 128K (14B-1M 92.2, Table 4); "NIAH 1M ~100%" → Passkey Retrieval at 1M is perfect for 14B-Instruct-1M and Turbo (Fig. 1); "MMLU / GSM8K within 1 point" → see Table 6 values above (7B MMLU-Pro −2.0, 14B GPQA −5.6); "first open model family supporting 1M context" → the report lists earlier 1M models, GLM-9B-Chat-1M and Gradient Llama-3-1M (§1); "DCA implementation not fully open" → the inference framework is open-sourced and integrated into vLLM (§5); authors "Qwen Team (Alibaba DAMO)" → Qwen Team, Alibaba Group.
- Removed as unsupported by the source: per-stage token counts (~50B/50B/100B); document-type mix (code ~35%, books ~25%, ...); 1-8 multi-needle SFT, 50K-200K summarization, 5-20 passage RAG-QA tasks; multi-position answer filter; "hundreds of thousands of SFT pairs"; cross-document masks and multi-topic bundles; InfiniteBench result; "DCA degrades on true long-range reasoning"; Qwen 3 inheritance claims.
- Not reported by the source: token counts, learning rates, batch sizes, data proportions, SFT example counts, short:long SFT ratio, DPO β and pair count, compute.
