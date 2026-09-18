<!-- scope: SkyLadder — controlled pretraining study showing shorter context windows win at a fixed token budget, and a short-to-long attention-window schedule that improves standard and long-context scores while reducing training time (120M–3B, up to 32K, 100B tokens)
     deps: [[sequence-packing]]
     see-also: [[dataset-decomposition]], [[fewer-truncations-best-fit-packing]], [[in-context-pretraining]], [[longrope2]], [[ruler]]
-->

# SkyLadder: Better and Faster Pretraining via Context Window Scheduling
- **Core Insight:** At a fixed 100B-token budget, 1B models pretrained with shorter context windows have higher average scores on nine standard benchmarks, and scheduling the attention window from short to long (SkyLadder) raises the 8K Random baseline average from 46.3 to 50.0 while reducing training time by 13.1% at 8K and 22.2% at 32K (Fig. 1, Table 1, Table 8).
- **Guideline:** When pretraining from scratch toward a target window of 8K–32K tokens with a fixed token budget, grow the attention window linearly from a small start (8–32 tokens) over about 64% of the tokens and then hold it at the target, because in the paper's runs short-to-long schedules scored higher on both standard and long averages than a constant long window, 4K→32K continual pretraining, and a long-to-short schedule (Tables 7, 17), and 64% expansion gave the lowest 120M loss at three token budgets (Table 24); it was not tested above 3B parameters or 32K tokens (App. A.1).
- **Authors:** Tongyao Zhu, Qian Liu, Haonan Wang, Shiqi Chen, Xiangming Gu, Tianyu Pang, et al. (National University of Singapore, Sea AI Lab, City University of Hong Kong)
- **Year:** 2025 (arXiv v1 2025-03; NeurIPS 2025)
- **URL:** https://arxiv.org/abs/2503.15450
- **Source type:** paper
- **Relevant topics:** pretraining context window, context window scheduling, sequence length curriculum, packing and intra-document masking, long-context pretraining, training efficiency

## Abstract
Pretraining context windows have grown over time. A controlled study in the paper finds that, under a fixed token budget, models pretrained with shorter context windows consistently outperform long-context counterparts. SkyLadder is a short-to-long context window transition during pretraining. It keeps strong standard benchmark performance and matches or exceeds baselines on long-context tasks. The authors pretrain 1B models (up to 32K context) and 3B models (8K context) on 100B tokens and report gains of up to 3.7% on common benchmarks and up to 22% faster training compared with baselines (abstract).

## Key Contributions
- A controlled comparison of pretraining windows from 512 to 16,384 tokens at a fixed token budget, with all other settings held constant (§3, §3.2).
- Four findings: the short-window advantage holds across model sizes, across packing and masking methods, and without positional encoding; intra-document masking is the best baseline, which the authors attribute partly to its shorter effective windows (§3.3).
- SkyLadder: block-wise local causal masks whose window w(t) grows linearly with the training step until it equals the target window; data order and packing are unchanged (§4.1, App. A.5).
- Evaluation on CommonCrawl, FineWeb-Pro, and Python code; model sizes 120M, 360M, 1B, 3B; target windows 8K and 32K (§4.3–4.4).
- Ablations of expansion rate, initial window, schedule shape, cyclic schedules, token budget, sliding-window variant, and a comparison with Dataset Decomposition (§4.5–4.6, App. A.7.4).

## Key Figures/Tables to Study
- Fig. 1 (right) and Fig. 4: average accuracy and validation perplexity versus pretraining window.
- Fig. 5 and App. A.5 pseudocode: the SkyLadder mask with Random and IntraDoc packing.
- Tables 1–2: standard, reading comprehension, and long benchmarks for 1B models on CommonCrawl.
- Tables 5–8: model-size scaling, 32K window, schedule types, training time and FLOPs.
- Tables 9, 17, 21, 23, 24: stability metrics, long-to-short schedule, expansion rate, initial window, token budget.

## Technical Details
- **Terms.** Context window L: length of packed training sequences. Random: random packing with a causal mask. BM25: BM25-retrieval packing with a causal mask. IntraDoc: random packing with a mask that blocks attention across documents (§3.2).
- **Preliminary study setup.** TinyLlama codebase; 120M, 360M, 1B models; about 30B tokens from the CommonCrawl subset of SlimPajama, trained for up to 100B tokens (about 3.3 epochs); RoPE in all models (§3.2). Validation perplexity uses a sliding window of 512 tokens (Fig. 4). Benchmarks: HellaSwag, ARC-E, ARC-C, Winogrande, CSQA, OBQA, PIQA, SIQA, MMLU through OLMES with 5-shot demonstrations (§3.2).
- **Effective window under IntraDoc.** With an 8K sequence length, fewer than 1% of per-token context windows reach 8K (Fig. 4d, App. A.6). Raising the RoPE base to 100,000 improves long-context models on long evaluation but leaves the short-versus-long gap (App. A.7.2, Fig. 11).
- **Mask.** M_ij = 0 when ⌊i/w⌋·w ≤ j ≤ i, otherwise −∞, where i is the query position, j the key position, and w the local window (§4.1). The mask can be combined with the intra-document mask (§4.1).
- **Schedule.** w(t) = min(w_e, w_s + ⌊αt⌋), where t is the training step, w_s the starting window, w_e = L the final window, and α the expansion rate in tokens per step (§4.1). Example: with w_s = 32 and α = 1/8, step 8,000 gives w = 32 + 1,000 = 1,032 tokens (derived). Defaults w_s = 32 and α = 1/8 reach L = 8192 after about 64K steps, around 64B tokens (§4.2).
- **Implementation.** Mask boundaries every w tokens, merged with document boundaries under IntraDoc, passed as `cu_seqlens` to `flash_attn_varlen_func` (App. A.5).
- **1B, CommonCrawl, 8K (Table 1).** Standard average: Random 46.3 → 50.0 (+3.7); IntraDoc 47.4 → 49.3 (+1.9). Against Random: MMLU +2.5, ARC-E +7.4, HellaSwag +4.0 (§4.3). Significance uses a McNemar test on full benchmark sets (App. A.7.3).
- **Reading comprehension and long tasks (Table 2).** Reading comprehension average: Random 25.5 → 30.2; IntraDoc 28.7 → 29.1. Long average (MDQA 30-document, RULER synthetic tasks as defined by HELMET): Random 15.3 → 14.3 (RULER 12.8 → 10.3, MDQA 17.7 → 18.3); IntraDoc 13.0 → 13.2. The authors attribute the RULER difference to fluctuation from its synthetic nature and small size (§4.2–4.3).
- **FineWeb-Pro, 1B.** 8K: Random 52.5 / 11.1 → 55.2 / 12.3 (standard / long); IntraDoc 54.3 / 12.7 → 54.8 / 13.9 (Table 4). 32K with α = 1/2: Random 50.7 / 9.7 → 54.3 / 13.5; IntraDoc 54.0 / 13.0 → 54.9 / 14.4 (§4.4, Table 6).
- **Model size, FineWeb-Pro (Table 5).** 120M: 40.1 / 5.8 → 41.2 / 5.1; 360M: 47.2 / 8.9 → 49.6 / 8.9; 3B: 57.0 / 15.8 → 60.5 / 19.3.
- **Code, 1B, 100B Python tokens (Table 3).** HumanEval pass@1 (greedy): 32K 17.7 → 21.3, 8K 22.0 → 23.2; pass@100 (t = 0.8): 32K 51.8 → 59.8.
- **Schedule shape, 1B, 32K (Table 7, long / standard).** Constant 9.7 / 50.7; Linear 13.5 / 54.3; Stepwise 13.3 / 55.3; Sinusoidal 14.2 / 54.2; Exponential 11.5 / 54.7; continual pretraining (4K for about 97B tokens, then 32K for 3B) 10.0 / 52.9 (§4.5).
- **Schedule direction (Table 17, 1B, FineWeb-Pro, 8K).** No scheduling 52.5 / 11.1; short-to-long 55.2 / 12.3; long-to-short (64B tokens changing, then 36B at 8K) 52.6 / 10.7 (App. A.7.4).
- **Expansion rate and start (120M, 8K, validation loss at 512 / 4K / 8K).** Baseline 2.780 / 2.590 / 2.549; α = 1/8: 2.732 / 2.553 / 2.519; α = 1 (8B tokens to reach 8K): 2.751 / 2.563 / 2.522 (Table 21). With α = 1/8, w_s = 8 gives 2.725 / 2.545 / 2.510 and w_s = 256 gives 2.748 / 2.567 / 2.531 (Table 23). The paper treats a loss difference above 0.01 as significant (App. A.7.4).
- **Token budget (120M, Table 24).** At 12.5B, 25B, and 50B total tokens, spending 64% of tokens in expansion gave the lowest loss of the settings tested, e.g. 2.698 at L = 8192 for 12.5B versus 2.790 for the baseline.
- **Efficiency (Table 8).** 8K: time 86.9% of baseline, FLOPs 11.6 → 9.9 ×10^20; 32K: time 77.8%, FLOPs 25.5 → 18.8 ×10^20.
- **Stability (120M, first 30B tokens, Table 9).** From 1K to 16K windows, loss volatility rises from 0.023 to 0.041 and average clipped gradient norm from 0.335 to 0.416; the paper reports exploding max attention logits at 16K and lower logits for shorter windows (Fig. 8). SkyLadder lowers attention entropy and delays the attention sink (Fig. 7); the authors link this to the gains (§4.6). Interpretation.
- **Related methods.** Dataset Decomposition on IntraDoc, FineWeb-Pro 1B: 1 cycle 53.9 / 12.3, 8 cycles 54.5 / 13.5, versus SkyLadder 54.8 / 13.9 (Table 10). Cyclic window schedules change 8K loss by at most 0.013 relative to SkyLadder (Table 22). A sliding-window variant scores 54.4 / 12.8 versus 55.2 / 12.3 for local causal masks (Table 25).

## Recipe ledger
App. A.4 states that all models, of every size and context length, use the Table 12 hyperparameters, taken mostly from TinyLlama. Loci refer to arXiv:2503.15450 v2.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All SkyLadder and baseline runs | 120M–3B | pretrain (from scratch) | optimizer | AdamW, β1 0.9, β2 0.95, weight decay 0.1, gradient-norm clipping 1 | App. A.4 Table 12 | verified 2026-09-14 | no ablation reported |
| All runs | 120M–3B | pretrain (from scratch) | LR schedule | cosine; peak 4e-4; minimum 4e-5; warmup 2000 steps | Table 12 | verified 2026-09-14 | no ablation reported |
| All runs | 120M–3B | pretrain (from scratch) | global batch; total steps | 1,048,576 (2^20) tokens; 100,000 steps | Table 12 | verified 2026-09-14 | no ablation reported |
| All runs | 120M–3B | pretrain (from scratch) | tokens; corpus | 100B tokens seen; CommonCrawl subset of SlimPajama (about 30B tokens, about 3.3 epochs), FineWeb-Pro, or Python code with the StarCoder tokenizer | §3.2, §4.3 | verified 2026-09-14 | no ablation reported |
| TinyLlama 120M / 360M / 1B | 120M; 360M; 1B | pretrain (from scratch) | layers, heads, width, intermediate, query groups | 12, 12, 768, 2048, 1 / 18, 16, 1024, 4096, 16 / 22, 32, 2048, 5632, 4; vocab 32000 | App. A.3 Table 11 | verified 2026-09-14 | no ablation reported |
| Llama3.2-architecture 3B | 3B | pretrain (from scratch) | layers, heads, width, intermediate, query groups | 28, 24, 3072, 8192, 8; vocab 32000 | Table 11 | verified 2026-09-14 | no ablation reported |
| All runs | 120M–3B | pretrain (from scratch) | RoPE θ | 120M, 360M: 10000; 1B: 10000 at 8K, 1000000 at 32K; 3B: 100000 | Table 11 | verified 2026-09-14 | no ablation of these values; App. A.7.2 tests base 100,000 in the window study |
| SkyLadder, 8K target | 120M–3B | pretrain (from scratch) | window schedule | linear, w_s = 32, α = 1/8; about 64K steps (about 64B tokens) to reach 8192 | §4.2 | verified 2026-09-14 | Table 21 (α) and Table 23 (w_s), 120M |
| SkyLadder, 32K target | 1B | pretrain (from scratch) | window schedule | α = 1/2; w_s = 32; w_e = 32768 | §4.4; Table 20 caption | verified 2026-09-14 | Table 7 compares schedule shapes; α not ablated at 32K |
| Paper recommendation | — | pretrain (from scratch) | window schedule | start at 8 tokens; linear growth; about 60% of tokens in expansion (64% in Table 24) | §1, §4.5, Table 24 | verified 2026-09-14 | Tables 21, 23, 24 (120M) |
| All runs | 120M–3B | pretrain (from scratch) | attention implementation | FlashAttention 2; SkyLadder boundaries via `flash_attn_varlen_func` `cu_seqlens` | §4.2, App. A.5 | verified 2026-09-14 | Table 25: sliding-window variant 54.4 / 12.8 vs 55.2 / 12.3 |
| All runs | 120M–3B | pretrain (from scratch) | compute | ≤ 1B on A100 40G nodes, 3B on H100 nodes; 1B at 8K for 100B tokens about 200 h on one 8×A100 node | App. A.9 | verified 2026-09-14 | — |
| All runs | 120M–3B | pretrain (from scratch) | seeds; checkpoint selection | one run per setup; checkpoint selection not reported (checked §3–4, App. A) | checklist item 7 | verified / not reported | — |

## Findings relevant to generality and long context
- **Breadth at a fixed budget.** Shorter pretraining windows give higher standard-benchmark averages at 1B (Fig. 1, Fig. 4); SkyLadder gains appear on knowledge (MMLU, closed-book NQ and TriviaQA average 9.0 → 13.2, Table 15), reading comprehension (Table 2), and code (Table 3). Result (single study; one run per setup, NeurIPS checklist item 7).
- **Long context.** Long-task gains grow with size: −0.7 at 120M, 0 at 360M, +3.5 at 3B (Table 5). On 1B CommonCrawl Random, the long average fell 15.3 → 14.3 (Table 2). For the 3B 8K model, RAG average rises 30.3 → 35.5 and many-shot ICL average 73.9 → 76.5, with SST2 falling 94.5 → 92.2 (Tables 13–14).
- **Data-length curricula.** The authors argue that ordering data by document length introduces domain bias, because long documents cluster in domains such as books; SkyLadder changes only the mask (§2, §4.6). Interpretation, supported by Table 10.
- **Limits.** Up to 3B parameters and 32K tokens; no theoretical analysis (App. A.1). Hyperparameters of w(t), α, w_s were not searched extensively (§4.2).

## Connections
- [[dataset-decomposition]] — variable-length data curriculum compared in Table 10.
- [[sequence-packing]], [[fewer-truncations-best-fit-packing]], [[in-context-pretraining]] — packing methods; the BM25 packing is inspired by In-context Pretraining (§3.2).
- [[lost-in-the-middle]] (MDQA), [[ruler]], [[helmet]], [[olmes]] — evaluations used.
- [[nope-length-generalization]] — procedure followed for the no-positional-encoding ablation (§3.3); [[rope-base-bounds-context-length]] — motivates the RoPE-base ablation (App. A.7.2).
- [[streamingllm-attention-sinks]] — attention sink measured in Fig. 7; [[kimi-k2]] — max-attention-logit monitoring followed in Fig. 8.
- [[gemma-3]] — hybrid local/global attention variant tested in Table 18.
- [[longrope2]], [[prolong]], [[string-effective-context]] — context extension of an already pretrained model, the setting this paper contrasts with training long-window models from scratch (§2).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2503.15450 (arXiv v2, 2025-12-02, NeurIPS 2025 version; v1 2025-03-19).
- Audit claims not found in the source: none.
- Not reported by the source: repeated seeds; checkpoint selection; schedule values specific to the 3B run beyond the §4.2 defaults.
