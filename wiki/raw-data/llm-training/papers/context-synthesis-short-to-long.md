<!-- scope: context synthesis for long-context instruction tuning — NIAH pilot on context composition and length, LLM-written background contexts for human-annotated QA pairs, LongBench comparison with instruction synthesis (LLaMA2-7B-64K, LLaMA3.1-8B)
     deps: [[longalign]], [[needle-in-haystack-data]]
     see-also: [[hierarchical-million-token-synth]], [[longbench]], [[ruler]], [[prolong]], [[long-context-llama3]]
-->

# Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning
- **Core Insight:** On LLaMA3.1-8B with UltraChat, adding 1.6k human-annotated instruction-answer pairs whose contexts were written by GPT-4o-mini (1 relevant + 9 irrelevant contexts) raises the 8-task LongBench average from 23.35 to 38.57, while the same 1.6k pairs with instruction synthesis from the original documents reach 25.39 (Table 3).
- **Guideline:** When long-context SFT data is built from an LLM, keep human-written instruction-answer pairs as targets and let the LLM write only the input context, and include irrelevant context in training samples, because in the NIAH pilot the model trained without distractors degraded when distractors appeared at test time, while the model trained with 1k-essay distractors stayed above 90% at 32k (§3.3, Figure 2).
- **Authors:** Wenhao Zhu, Pinzhen Chen, Hanxu Hu, Shujian Huang, Fei Yuan, Jiajun Chen, et al. (Nanjing University, University of Edinburgh, Shanghai AI Lab, University of Zurich)
- **Year:** 2025 (arXiv v1 2025-02)
- **URL:** https://arxiv.org/abs/2502.15592
- **Source type:** paper
- **Relevant topics:** long-context SFT, synthetic data, distractor context, length generalization, back-translation-style synthesis, data-quality diagnostics

## Abstract
Research on long-context LLMs has focused on position modelling; instruction tuning for long inputs has received less study, and long training examples are expensive to create. The authors run a controlled needle-in-a-haystack (NIAH) study and find that models instruction-tuned on short contexts can generalize to longer ones, and that instruction difficulty and context composition matter. They propose "context synthesis": an off-the-shelf LLM writes an extended background context for an existing high-quality instruction-answer pair. On LongBench document-level tasks the method outperforms instruction synthesis and comes close to human-annotated long-context instruction data (Abstract).

## Key Contributions
- A pilot on RULER-style NIAH tasks that varies context composition (needle only, needle + 1k essay, needle + 64k essay) under a fixed general SFT set (§3, Table 1).
- Context synthesis: the synthetic part is only the model input, so the instruction and answer keep their human-annotated quality; the authors compare it to back-translation (§1, §4.2).
- Context extension by a ~2,000-word generation target and by concatenating contexts written for other instructions as incoherent distractors (§4.2).
- A diagnostic: fine-tune with and without the context; a small gap indicates weak instruction-context dependence in the synthetic data (§4.3, Figure 4, App. G).

## Key Figures/Tables to Study
- Figure 2 + Table 1: base vs SFT1-SFT4 on four NIAH variants from ~0K to 64K.
- Figure 3: the context-synthesis prompt template.
- Table 3: LongBench (8 tasks) for LLaMA2-7B and LLaMA3.1-8B, including LongAlpaca, LongAlign, LongMIT data.
- Figure 4: context-free vs context-included tuning, with human-annotated data as a reference line.
- Tables 4, 5, 11: synthesis engines, unseen tasks, concatenation size n.

## Technical Details
- **Pilot setup:** base LLaMA2-7B-64k (LongAlign-7B-64k-base) (§3.2, footnote 3). SFT1 uses ShareGPT only; SFT2-SFT4 add 200 NIAH-style samples for each of four subtasks, 800 in total (§3.2). SFT2 context = needle sentence only; SFT3 = needle + 1k essay; SFT4 = needle + 64k essay; haystack = Paul Graham essays with newly generated needles (Table 1, footnote 4). Evaluation: single, multi-key, multi-query, multi-value NIAH, 500 samples each, exact match (Table 2).
- **Pilot results:** as test length grows, SFT1 diverges and is below the base model on single, multi-key and multi-query NIAH (§3.3). SFT2 may learn shortcuts and degrades with distracting information (§3.3). SFT3 stays above 90% on all NIAH tasks at 32k test length (§3.3). SFT4 is almost perfect; its gap to SFT3 is largest at the maximum length on multi-value NIAH (§3.3).
- **Instruction collection:** 200 instructions sampled from the training set of each of 8 LongBench-matched subtasks (NarrativeQA, Qasper, HotpotQA, 2WikiMultihopQA, MuSiQue, GovReport, QMSum, MultiNews), 1.6k total (§5.1, Table 2). Alpaca pairs were tried first and were ineffective because they test parametric knowledge (§4.2, footnote 5).
- **Context synthesis prompt:** "Please infer the missing context"; the context must lead to both question and answer, include numerical and factual details, and reach approximately 2,000 words (Figure 3). The 2,000-word target follows reports that current LLMs struggle to generate beyond this length (§4.2).
- **Concatenation:** n = 10 contexts (1 relevant, 9 irrelevant) in main results (§5.1 footnote 11, App. F).
- **Instruction-synthesis baseline:** same 1.6k samples, contexts and domain; GPT-4o-mini writes QA pairs from the original contexts (§5.1). The unconstrained template (Figure 6) scores 25.39 and the task-specific template (Figure 7) 23.12 on LLaMA3.1-8B (Table 9).
- **Evaluation:** LongBench codebase, zero-shot, greedy decoding (§5.1).
- **Main results, LLaMA2-7B (ShareGPT 89.3k as general data):** ShareGPT 21.33; + LongAlpaca (12.0k) 22.86; + LongAlign (9.9k) 24.57; + LongMIT (64.4k) 27.67; + instruction synthesis (1.6k) 24.66; + context synthesis (1.6k) 33.42 (Table 3, 8-task average).
- **Main results, LLaMA3.1-8B (UltraChat 515.3k):** UltraChat 23.35; + LongAlpaca 23.99; + LongAlign 19.60; + LongMIT 28.54; + instruction synthesis 25.39; + context synthesis 38.57 (Table 3). With ShareGPT instead: 24.28 / 26.73 / 35.81 (Table 10).
- **Versus human-annotated data:** context synthesis is comparable or better on single-document QA and summarization; a gap remains on multi-document QA (§5.2, Figure 4).
- **Concatenation size (LLaMA3.1-8B):** n = 1: 37.67; n = 5: 37.92; n = 10: 38.57 (Table 11). Without concatenation the model trained on shorter contexts still generalizes to the long-context test tasks (§6, Figure 5).
- **Synthesis engine (LLaMA3.1-8B, single-doc QA / multi-doc QA / summarization):** LongWriter-8B 38.85 / 44.68 / 30.79; Qwen2.5-72B 38.30 / 44.90 / 31.53; GPT-4o-mini 39.02 / 45.40 / 31.44 (Table 4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA2-7B-64k and LLaMA3.1-8B (128k) | 7B, 8B | SFT | optimizer | AdamW, β1 = 0.9, β2 = 0.95 | arXiv:2502.15592v1 App. D | verified 2026-09-14 | no ablation reported |
| both models above | 7B, 8B | SFT | peak LR, schedule, warmup | 2e-5, cosine decay, 3% warm-up ratio | App. D | verified 2026-09-14 | no ablation reported |
| both models above | 7B, 8B | SFT | epochs | 2 | App. D | verified 2026-09-14 | no ablation reported |
| both models above | 7B, 8B | SFT | packing and loss | packing with loss weighting (LongAlign); loss on output tokens only | App. D | verified 2026-09-14 | no ablation reported |
| LLaMA2-7B-64k | 7B | SFT | max tokens per packed sample | 65,536 | App. D | verified 2026-09-14 | LongAlign codebase default |
| LLaMA3.1-8B-128k | 8B | SFT | max tokens per packed sample | 32,768 | App. D | verified 2026-09-14 | LongAlign codebase default |
| both models above | 7B, 8B | SFT | per-device batch; hardware; sharding | 1; 8×H800 (one node); DeepSpeed ZeRO-3 | App. D, §5.1 | verified 2026-09-14 | not applicable |
| both models above | 7B, 8B | SFT | long-context examples | 1.6k (200 per subtask × 8) | §5.1 | verified 2026-09-14 | Table 3 vs instruction synthesis at equal size |
| LLaMA2-7B / LLaMA3.1-8B | 7B, 8B | SFT | general data | ShareGPT 89.3k / UltraChat 515.3k | Table 3, §5.1 | verified 2026-09-14 | no selection ablation; Table 10 repeats the comparison with ShareGPT on LLaMA3.1-8B |
| both models above | 7B, 8B | distill-SFT | generator and sampling | gpt-4o-mini-2024-07-18, default temperature and top-p | §5.1 footnote 12 | verified 2026-09-14 | Table 4: three engines give comparable scores |
| both models above | 7B, 8B | distill-SFT | context length target; concatenation | ~2,000 words; n = 10 (1 relevant + 9 irrelevant) | Figure 3; App. F | verified 2026-09-14 | Table 11: n = 10 best of n ∈ {1, 5, 10} |
| LLaMA2-7B-64k (pilot) | 7B | SFT | NIAH samples; general data | 800 (200 × 4 subtasks); ShareGPT | §3.2, Table 1 | verified 2026-09-14 | Figure 2 |

Not reported: pilot training hyperparameters (App. D does not say whether they apply to §3), tokens per epoch, warmup steps, weight decay, gradient clipping, seeds.

## Findings relevant to generality, negative feedback, long context, distillation
- **Distractors (negative as content):** irrelevant contexts are placed in the input and the answer is trained with ordinary cross-entropy; no sample receives a negative gradient (§4.2, App. D). Training without distractors in the pilot (SFT2) leads to degradation when test contexts contain distracting information (§3.3).
- **Short-to-long generalization:** 1k-essay training contexts give >90% NIAH accuracy at 32k (§3.3); without concatenation, the model trained on the shorter synthesized contexts still generalizes to the LongBench tasks; Figure 5 plots the synthetic and test context-length distributions (§6).
- **Held-out evaluation:** the 1.6k instructions come from the training splits of the same 8 datasets whose LongBench test sets are used for evaluation (§5.1, Table 2), so Table 3 measures in-distribution task families. The unseen-task check on LLaMA3.1-8B: RULER multi-value NIAH 91.57 → 96.54 (+4.97), ZeroSCROLLS QuALITY 71.43 → 76.19 (+4.76), SQuALITY 23.02 → 23.65 (+0.63) (Table 5). Short-context benchmarks (for example MMLU) are not reported.
- **Why instruction synthesis underperforms (Interpretation by the authors):** adding the long context to synthesized instructions yields no further improvement over context-free tuning, so the long input provides little learning signal (§6, Figure 4); for LongAlign data the context-free and context-included results differ little (App. G, Figure 8).
- **Distillation details:** stage = SFT data generation; prompt types = document-grounded QA and summarization instructions; teacher = GPT-4o-mini with default sampling; quality control = no per-sample filter, only the dataset-level context-free vs context-included diagnostic (§4.3). The Limitations section lists an automated verification framework to filter context-aware instructions as future work.
- **Contrast with [[prolong]]:** the authors cite Gao et al. (2024) as arguing that synthetic long-context data offers minimal benefit and that ShareGPT-style short data suffices, and they attribute their LongAlpaca/LongAlign null results on LLaMA3.1 to instruction quality instead (§2, §5.2) (Interpretation).

## Connections
- [[longalign]] — LongAlign data (baseline in Table 3) and the training codebase and packing-with-loss-weighting method used here.
- [[longmit]] — LongMIT multi-hop long instruction data (Chen et al., 2024b) is a baseline; check that the existing card describes that dataset.
- [[longalpaca]] — LongAlpaca-12k baseline.
- [[needle-in-haystack-data]], [[ruler]] — the pilot's NIAH tasks and the unseen multi-value NIAH test.
- [[longbench]] — main evaluation (8 tasks).
- [[prolong]] — Gao et al. (2024), the cited counter-position on synthetic long SFT data.
- [[long-context-llama3]] — Llama 3 concatenation of incoherent text, which the distractor construction follows (§4.2).
- [[hierarchical-million-token-synth]] — another short-context-generator pipeline; it scales by concatenating books with multi-turn QA instead of writing contexts for fixed QA pairs.
- [[longwriter]] — LongWriter-8B is one of the synthesis engines in Table 4.
- [[artificial-needles-real-haystacks]], [[wildlong]], [[longmagpie]] — other synthetic long-context SFT data methods.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.15592 (arXiv v1, 2025-02-21; only version).
- Audit claims not found in the source: "200 per task type ... covering single-doc QA, multi-doc QA, summarization" → 200 per subtask across 8 datasets (§5.1); "ZeroSCROLLS +4.76" → +4.76 is QuALITY only, SQuALITY is +0.63 (Table 5).
