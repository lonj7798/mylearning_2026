<!-- scope: reasoning-trace distillation — 800 curated long-CoT traces SFT'd into Qwen2.5-32B-Instruct
     deps: [[lima]], [[s1]]
     see-also: [[openmathinstruct]], [[star]], [[openmathinstruct-2]], [[numina-math]], [[open-thoughts]]
-->

# LIMO: Less is More for Reasoning
- **Core Insight:** Supervised fine-tuning of Qwen2.5-32B-Instruct on 800 curated long-CoT traces reaches 63.3% pass@1 on AIME24 and 95.6% on MATH500, against 16.5% and 79.4% for the same base model (§6.2, Table 1).
- **Guideline:** When the base model already scores well on short-form math and a strong reasoning teacher is available, select a few hundred problems that the teacher solves in only 1–3 of 32 attempts and keep the highest-scoring trace per problem, because at 32B the sweep from 400 to 2,000 samples moves AIME24 from 57.5 to 69.6 while MATH500 stays within 1 point (§6.3.5). Otherwise, when the base model is weak in the domain, expect the effect to be much smaller (§6.3.3).
- **Authors:** Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu (Shanghai Jiao Tong University, SII-GAIR, Fudan University, The Hong Kong Polytechnic University)
- **Year:** 2025 (arXiv v1 2025-02; v3 2025-07-29; COLM 2025)
- **URL:** https://arxiv.org/abs/2502.03387
- **Source type:** paper
- **Relevant topics:** reasoning-trace distillation, long-CoT, SFT data selection, sample efficiency, difficulty filtering

## Abstract
The paper argues that complex mathematical reasoning does not require large post-training datasets. Using supervised fine-tuning on 800 curated examples, the LIMO model reaches 63.3% on AIME24 and 95.6% on MATH500, compared to 6.5% and 59.2% for previous fine-tuned models, while using 1% of the training data of those approaches (Abstract). The paper reports a 45.8 percentage-point absolute improvement across a set of out-of-distribution benchmarks (Abstract). It states the Less-Is-More Reasoning Hypothesis: in foundation models where domain knowledge is already encoded during pre-training, reasoning can emerge from a small number of demonstrations that act as cognitive templates, so the threshold is set by (1) the completeness of pre-trained knowledge and (2) how effectively the post-training examples use inference-time computation (Abstract).

## Key Contributions
- A two-stage selection pipeline that produces 800 SFT examples from a pool of tens of millions of math problems (§3.1.1, §3.1.2).
- A rule-based scoring function over four surface properties of a reasoning chain, used to pick one trace per problem (§3.1.2).
- Controlled comparisons against OpenThoughts-114k and a 100k NuminaMath-CoT subset on the same base model (§6.2, Table 1).
- Ablations over chain quality (§6.3.1), question difficulty (§6.3.2), base model (§6.3.3), model size (§6.3.4), and dataset size (§6.3.5).
- Public release of data, model, and evaluation code at https://github.com/GAIR-NLP/LIMO (footnote 1).

## Key Figures/Tables to Study
- Figure 2 — the dataset construction pipeline from candidate pool to LIMO-Pool to 800 samples.
- Table 1 (§6.2) — pass@1 for LIMO against OpenAI-o1-preview, QwQ-32B-Preview, Qwen2.5-32B-Instruct, OpenThoughts-114k, and NuminaMath-100k on 10 benchmarks.
- Figure 6 (§6.3.4) — AIME24 accuracy against base-model size, 3B to 72B.
- Figures 7 and 8 (§6.3.5) — accuracy against dataset size, 400 to 2,000 samples.

## Technical Details
- **Candidate pool:** NuminaMath-CoT, DeepScaleR (about 40,000 unique problems), AIME problems from before 2024, MATH, and Chinese elementary through undergraduate exercises and examination papers, starting from tens of millions of problems (§3.1.1).
- **Difficulty filter, stage 1:** problems solved by Qwen2.5-Math-7B-Instruct within four attempts were excluded (§3.1.1).
- **Difficulty filter, stage 2:** DeepSeek-R1-Distill-Qwen-32B sampled 32 attempts per remaining problem; problems solved in 1–3 of 32 attempts were kept, yielding 2,125 problems called LIMO-Pool (§3.1.1).
- **Contamination check:** n-gram matching against all evaluation benchmarks, reported as no overlap (§3.1.1).
- **Trace generation:** multiple solutions sampled from DeepSeek-R1, DeepSeek-R1-Distill-Qwen-32B, and QwQ-32B (§3.1.2).
- **Trace scoring:** a rule-based weighted score over solution length (Elaborated Reasoning, 30%), frequency of validation words such as "check" and "verify" (Self-Verification, 20%), frequency of tentative expressions such as "perhaps" and "might" (Exploratory Approach, 25%), and connective phrases such as "therefore" and "since" (Adaptive Granularity, 25%); all keyword frequencies normalized by text length (§3.1.2).
- **Final selection:** highest-scoring solution per problem, then the top 800 problem–solution pairs by quality score (§3.1.2).
- **Evaluation protocol:** pass@1, zero-shot chain-of-thought; greedy decoding with one sample on benchmarks with 50 or more problems; 4 samples at temperature 0.6 with unbiased pass@1 on AIME24, AMC23, and CHMath; 32,768-token maximum output (§5).
- **Main results (Table 1, §6.2):** LIMO 63.3 AIME24 / 95.6 MATH500 / 96.3 AMC23; Qwen2.5-32B-Instruct 16.5 / 79.4 / 64.0; QwQ-32B-Preview 50.0 / 89.8 / 83.6; OpenAI-o1-preview 44.6 / 85.5 / 81.8. Ten-benchmark average: LIMO 78.1, QwQ-32B-Preview 66.9, OpenAI-o1-preview 61.1, OpenThoughts-114k 58.3, base 49.9, NuminaMath-100k 32.3.
- **Base-model ablation (§6.3.3):** the same 800 examples on Qwen1.5-32B-Chat give 9.2 AIME24 and a MATH500 score 30.4 points below LIMO; on Qwen2.5-32B-Instruct they give 63.3 and 95.6.
- **Size ablation (§6.3.4):** AIME24 rises from 2.5 at 3B to 68.3 at 72B; MATH500 is 95.6 at 32B and 94.8 at 72B.
- **Question-difficulty ablation (§6.3.2):** training on Advanced-500 (AIME problems) instead of easier sets gives a 16-point AIME24 gain, reaching 51.5, and 91.2 on MATH500.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LIMO | 32B | distill-SFT | base model | Qwen2.5-32B-Instruct | arXiv:2502.03387v3 §4 | verified (2026-09-18) | §6.3.3: Qwen1.5-32B-Chat gives 9.2 AIME24 vs 63.3 |
| LIMO | 32B | distill-SFT | training examples | 800 | §3.1.2 | verified (2026-09-18) | §6.3.5 Figs. 7–8: 400 → 57.5 AIME24 / 94.8 MATH500; 800 → 63.3 / 95.6; 1.2k → +0.9 / −0.2 vs 800; 2k → 69.6 / 95.8 |
| LIMO | 32B | distill-SFT | epochs | 15 | §4 | verified (2026-09-18) | no ablation reported |
| LIMO | 32B | distill-SFT | peak LR / schedule / warmup | 5.0e-6, cosine decay, no warmup | §4 | verified (2026-09-18) | no ablation reported |
| LIMO | 32B | distill-SFT | global batch | 64 examples | §4 | verified (2026-09-18) | no ablation reported |
| LIMO | 32B | distill-SFT | sequence length limit | 16,384 tokens | §4 | verified (2026-09-18) | chosen because all SFT responses are under 16,384 tokens (§4) |
| LIMO | 32B | distill-SFT | parameterization / stack | full-parameter FT, DeepSpeed ZeRO-3, FlashAttention-2 | §4 | verified (2026-09-18) | no ablation reported |
| LIMO | 32B | distill-SFT | loss masking, packing, optimizer betas, weight decay, compute | — | §4 | not reported (checked §4 and the paper body; the paper has no appendix) | — |

## Findings relevant to generality
- The paper separates in-domain (AIME24, MATH500, AMC23) from out-of-domain benchmarks (OlympiadBench, CHMath, Gaokao, Kaoyan, GradeSchool, Minerva, GPQA) and reports gains on both; the out-of-domain average is where the 45.8-point abstract claim comes from (§6.2, Table 1).
- Training on harder problems transfers: the model fine-tuned on Advanced-500 reaches 91.2 on MATH500 with no MATH training data (§6.3.2).
- Larger datasets are not harmful but saturate: 2,000 samples give the best overall result with little gain over 1,600 (§6.3.5).
- Uncurated scale can be worse than the base model: the 100k NuminaMath-CoT subset drops the ten-benchmark average to 32.3 from the base model's 49.9, which the authors attribute to uncurated reasoning chains (§6.2).

## Findings relevant to distillation
- All training traces come from other models: DeepSeek-R1, DeepSeek-R1-Distill-Qwen-32B, and QwQ-32B (§3.1.2). LIMO is a distillation-SFT recipe whose contribution is the selection function, not the trace source.
- Trace quality is selected by surface statistics, not by step-level verification. Chains are ranked into five levels L1–L5 by the same rule-based score on 500 questions with multiple correct solutions; models trained on L5 chains score highest on AIME24 and MATH500, with performance decreasing at each lower level (§6.3.1).

## Connections
- [[s1]] — same hypothesis on a comparable scale of curated traces, with a different selection procedure.
- [[lima]] — the alignment-stage precedent for small curated SFT sets.
- [[numina-math]] — supplies NuminaMath-CoT to the candidate pool (§3.1.1) and the 100k contrast run (§6.2).
- [[open-thoughts]], [[openmathinstruct]], [[openmathinstruct-2]] — scale-first reasoning-trace datasets; OpenThoughts-114k is the 114k baseline in Table 1.
- [[rlvr-beyond-base-model]], [[front-loading-reasoning]] — the latent-capability question that the LIMO Hypothesis states but does not test directly.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2502.03387 (arXiv v3, 29 Jul 2025, COLM 2025 camera-ready)
- Corrections to the previous card version:
  - "the original release uses 817 long-CoT samples" → the checked version reports 800 training samples throughout (§1, §3.1.2, Table 1). The 817 figure and its paired 57.1 / 94.8 scores belong to arXiv v1 and are not in v3.
  - "competition math, MATH, GSM8K-hard, and physics Olympiad problems" → NuminaMath-CoT, DeepScaleR, pre-2024 AIME, MATH, and Chinese school and undergraduate exercises (§3.1.1).
  - "collect multiple candidate traces from strong reasoning models and human editing" and "Manual curation: hand-filter down to the final small set, removing lucky guesses" → traces come from three models and are ranked by a rule-based weighted score; the authors inspected solutions to design the scoring dimensions, but the final 800 are the top-scoring pairs (§3.1.2).
  - "Correctness verifier: gold-answer exact match; no step-level verifier is used" → correctness is used at the question-filtering stage as the 1–3 of 32 success rate under DeepSeek-R1-Distill-Qwen-32B (§3.1.1); the chain selection itself is the rule-based score (§3.1.2).
  - "Year: 2025" without version → arXiv v1 2025-02, checked against v3; the 63.3 / 95.6 numbers are the v3 numbers.
- Removed as unsupported by the source: "817 samples"; "GSM8K-hard"; "physics Olympiad"; "human editing" of traces; "hand review removes traces with subtly wrong intermediate logic"; "trace diversity matters more than dataset size" as a separate claim; "labor-heavy curation" as a cost statement; "Risks + gotchas" items on curator subjectivity, base-model dependence as a general law, and benchmark overlap (the paper reports an n-gram decontamination check and no overlap, §3.1.1).
- Not reported by the source: token counts of the 800 examples; optimizer betas, weight decay, and gradient clipping; GPU type, GPU-hours, or wall-clock; per-benchmark variance or seeds; any RL stage.
