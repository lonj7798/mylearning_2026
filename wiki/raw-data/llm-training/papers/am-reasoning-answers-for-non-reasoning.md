<!-- scope: a-m-team study of which part of a reasoning model's output (answer only, think summary + answer, original response, non-reasoning teacher answer) to use as SFT targets for a non-reasoning 32B model
     deps: [[am-deepseek-r1-distilled-1-4m]], [[deepseek-r1]]
     see-also: [[qwen3-2507-instruct-thinking-split]], [[nvidia-llama-nemotron-post-training-dataset]], [[tokenskip]], [[distilling-step-by-step]]
-->

# Leveraging Reasoning Model Answers to Enhance Non-Reasoning Model Capability
- **Core Insight:** SFT of Qwen2.5-32B on DeepSeek-R1's answer segment alone (thinking removed) raises GSM8K from 83.7 to 92.2, HumanEval from 80.5 to 90.9, and GPQA-Diamond from 34.3 to 40.3 over training on the original community responses; prepending a Qwen2.5-7B-Instruct summary of the thinking raises GPQA-Diamond to 47.7 but lowers IFEval prompt-strict to 61.2 against 73.4 for the baseline (Table 1).
- **Guideline:** When training a non-thinking model from a thinking teacher and instruction-format adherence matters, use the teacher's answer segment as the target rather than a summary-plus-answer, because in Table 1 the answer-only variant kept IFEval at 76.9 and MMLU at 82.5 while the summary variant fell to 61.2 on IFEval; the comparison is at 32B only, with no seeds, repeated runs, or variance reported.
- **Authors:** Haotian Wang, Han Zhao, Shuaiting Chen, Xiaoyu Tian, Sitong Zhao, Yunjie Ji, et al. (a-m-team)
- **Year:** 2025 (arXiv v1 2025-04; no venue)
- **URL:** https://arxiv.org/abs/2504.09639
- **Source type:** paper
- **Relevant topics:** distillation to non-reasoning models, SFT target construction, CoT summarization, instruction following, capability trade-offs

## Abstract
The paper asks whether high-quality outputs from reasoning models such as DeepSeek-R1 can improve "less
computationally demanding, non-reasoning models". The authors compare ways to turn reasoning-model outputs into SFT
targets for a model that answers directly. Through SFT experiments on established benchmarks they report
"consistent improvements across various benchmarks", and they state that the result depends on how the reasoning
output is structured into the target (Abstract; §4).

## Key Contributions
- A controlled comparison of SFT targets built from the same query set: original response, reasoning-model answer
  only, and summarized thinking plus answer, with a DeepSeek-V3-0324 answer set as a fourth dataset (§2.2).
- Evidence that answer-only targets improve math, code, and GPQA scores over the original responses (§3.3, Table 1).
- Evidence of a trade-off: thinking summaries raise GPQA-Diamond and MT-Bench but reduce IFEval (§3.3, Table 1).

## Key Figures/Tables to Study
- §2.2 equations (1)–(4): definitions of the three target types.
- Table 1: seven benchmarks for four SFT variants and an OLMo-2-32B-0325-SFT reference row.
- Fig. 2 (category distribution) and App. Fig. 3 (inference prompts) are images; their contents were not transcribed.

## Technical Details
- Prompts: approximately 1.3 million instances from open collections including Infinity Instruct, OpenCoder,
  PRIME, NuminaMath, CodeContests, FLAN, Orca, AM-DeepSeek-R1-Distilled-1.4M, and tuluv3, covering mathematics,
  code, science, general QA, instruction following, and tool use (§2.1).
- Responses: DeepSeek-R1 generations, supplemented by instances selected from AM-DeepSeek-R1-Distilled-1.4M;
  separate user prompts for math and code queries; outputs split into `<think></think>` and `<answer></answer>`
  (§2.1). Response filtering or verification for this study is not described.
- Target definitions (§2.2): for query Q, (T_reason, A_reason) = M_reason(Q) is the reasoning model's output,
  where T_reason is the thinking and A_reason the final answer; R_orig is the response in the source community
  dataset; M_sum is Qwen2.5-7B-Instruct; ⊕ is string concatenation.
  - R1 = R_orig (baseline, eq. 1).
  - R2 = A_reason (answer component only, eq. 2).
  - R3 = S_think ⊕ A_reason, where S_think = M_sum(T_reason) (eqs. 3–4).
- A fourth dataset uses DeepSeek-V3-0324 responses as a control group (§2.2 closing paragraph).
- Evaluation (§3.1.2): max generation 16,384 tokens. GPQA-Diamond: temperature 0.6, top-p 0.95, 8 samples per
  question for pass@1. GSM8K, HumanEval, IFEval, MMLU, AlignBench, MT-Bench: greedy, one sample. IFEval reports
  prompt-strict; MMLU is 5-shot; AlignBench and MT-Bench are judged by OpenAI GPT-4.

Results (Table 1; "—" = not reported):

| Model | GPQA-D | GSM8K | MMLU | HumanEval | IFEval | AlignBench | MT-Bench |
|---|---|---|---|---|---|---|---|
| OLMo-2-32B-0325-SFT | — | 78.4 | 76.1 | — | 72.4 | — | — |
| AM-32B-DeepSeek-V3-Answer-SFT | 47.9 | 90.2 | 70.1 | 90.9 | 76.6 | 7.6 | 8.2 |
| AM-32B-R1-Answer-SFT (R2) | 40.3 | 92.2 | 82.5 | 90.9 | 76.9 | 6.9 | 7.6 |
| AM-32B-Think-Summarization-SFT (R3) | 47.7 | 91.4 | 81.0 | 86.0 | 61.2 | 7.2 | 7.9 |
| AM-32B-Raw-Answer-SFT (R1, baseline) | 34.3 | 83.7 | 82.5 | 80.5 | 73.4 | 6.2 | 7.7 |

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All four AM-32B-*-SFT variants (one stated setup) | 32B | distill-SFT | initial checkpoint | Qwen2.5-32B (base/instruct not stated) | arXiv:2504.09639v1 §3.2 | verified 2026-09-14 | no ablation reported |
| same | 32B | distill-SFT | prompts | approximately 1.3 million instances | §2.1 | verified 2026-09-14 | no ablation reported |
| same | 32B | distill-SFT | peak LR; schedule; warmup ratio | 8e-6; cosine; 0.05 | §3.2 | verified 2026-09-14 | no ablation reported |
| same | 32B | distill-SFT | batch size | 64 (sequences or tokens not stated) | §3.2 | verified 2026-09-14 | no ablation reported |
| same | 32B | distill-SFT | max token length; epochs | 16,384; 3 | §3.2 | verified 2026-09-14 | no ablation reported |
| same | 32B | distill-SFT | optimizer, weight decay, packing, loss masking, seeds | not reported | checked §1–§5 and Appendix | not reported | — |
| AM-32B-R1-Answer-SFT, AM-32B-Think-Summarization-SFT | 32B | distill-SFT | teacher; summarizer | DeepSeek-R1; Qwen2.5-7B-Instruct | §2, §2.2 | verified 2026-09-14 | Table 1 comparison |
| AM-32B-DeepSeek-V3-Answer-SFT | 32B | distill-SFT | teacher | DeepSeek-V3-0324 | §2, §2.2 | verified 2026-09-14 | Table 1 comparison |
| teacher generation (all variants) | — | distill-SFT data | temperature, top-p, samples per prompt | not reported | checked §2.1, §2.2, Appendix text | not reported | — |

## Findings relevant to generality and distillation
- Stage and target: non-thinking SFT; what is kept from the teacher is the answer segment (R2) or a 7B summary of
  the thinking plus the answer (R3); no variant uses the unsummarized DeepSeek-R1 thinking T_reason (§2.2).
- Narrowing within the same setup: R3 has the second-highest GPQA-Diamond (47.7, after 47.9 for the
  DeepSeek-V3-0324 set) but the lowest IFEval (61.2); the authors state that "the substantial alteration of the
  original answer format may interfere" with strict instruction adherence (Interpretation, §3.3).
- Chat metrics: the authors attribute lower chat scores for R2 to answers that are "overly concise", because the
  procedural explanation sits in the thinking (Interpretation, §3.3).
- The DeepSeek-V3-0324 answer set has the highest AlignBench (7.6) and MT-Bench (8.2) but MMLU 70.1, 12.4 points
  below the baseline (Table 1; difference derived). The paper does not discuss this MMLU drop.
- The authors conclude that answer-only fine-tuning "does not automatically transfer the full spectrum of the source
  model's capabilities", especially conversational ability (§3.3). They name prompting the teacher to put concise
  reasoning inside the answer as untested future work (§5).
- Inconsistencies in the source: §3.3 says R2 is "marginally lower" than the baseline on AlignBench and MT-Bench, but
  Table 1 shows AlignBench 6.9 vs 6.2 (higher) and MT-Bench 7.6 vs 7.7; §4 says summary methods excel "in specific
  areas like instruction following", but R3 has the lowest IFEval (61.2); §2.2 refers to "Method 4" for M_sum
  although three methods are defined.

## Connections
- [[am-deepseek-r1-distilled-1-4m]] — same team; a prompt source and a source of selected response instances.
- [[deepseek-r1]], [[deepseek-v3]] — the reasoning teacher and the non-reasoning control teacher.
- [[qwen-2.5]] — base family of the 32B student and of the 7B summarizer.
- [[olmo-2]] — OLMo-2-32B-0325-SFT appears as a reference row in Table 1.
- [[ifeval]] — the benchmark on which the summary-plus-answer target loses 12.2 points against the baseline.
- [[qwen3-2507-instruct-thinking-split]], [[nvidia-llama-nemotron-post-training-dataset]] — other approaches to
  separate thinking and non-thinking behavior or data.
- [[tokenskip]], [[c3ot-compressed-cot]] — shortening CoT instead of removing or summarizing it.
- [[distilling-step-by-step]], [[orca]] — earlier work on training students with teacher rationales or explanations.
- [[flan]], [[tulu-3-sft-mix]], [[numina-math]] — prompt collections named in §2.1.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2504.09639 (v1, PDF).
- Audit claims not found in the source: "it shares the a-m-team data stack" for quality control (the paper
  describes no verification for this study); the audit's prompt-source list omitted AM-DeepSeek-R1-Distilled-1.4M,
  tuluv3, and the instruction-following and tool-use domains, and its results omitted the DeepSeek-V3-0324 control
  row, AlignBench, and MT-Bench (added from Table 1). HumanEval 90.9 for answer-only targets ties the
  DeepSeek-V3-0324 set rather than being the single best (Table 1).
