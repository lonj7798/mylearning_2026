<!-- scope: controlled teacher comparison for long-CoT SFT: the same 1.89M queries distilled from AM-Thinking-v1, Qwen3-235B-A22B, and DeepSeek-R1, verified to score >= 0.9, and used to train three Qwen2.5-32B base students
     deps: [[deepdistill]]
     see-also: [[am-thinking-v1]], [[capacity-gap-law-distillation]], [[naturalthoughts]], [[unveiling-cot-distillation-factors]]
-->

# Not All Correct Answers Are Equal: Why Your Distillation Source Matters
- **Core Insight:** With 1.89M queries held fixed and each teacher's response regenerated until its verification score is at least 0.9, Qwen2.5-32B base students trained on AM-Thinking-v1 / Qwen3-235B-A22B / DeepSeek-R1 outputs score 84.3 / 79.4 / 70.9 on AIME2024 and 65.9 / 59.6 / 57.0 on LiveCodeBench (Table 1).
- **Guideline:** When choosing a teacher for verified reasoning-trace SFT, compare teachers on the same query set by training a student on each set and inspecting trace length spread and perplexity, not only verification pass status, because in this study all three datasets passed the same ≥ 0.9 verification and still produced students 13.4 AIME2024 points apart (Table 1; derived). The evidence is one student model, one run per teacher, and a teacher built by the authors.
- **Authors:** Xiaoyu Tian, Yunjie Ji, Haotian Wang, Shuaiting Chen, Sitong Zhao, Yiping Peng, et al. (a-m-team, an internal team at Beike/Ke.com, footnote 1)
- **Year:** 2025 (arXiv v1 2025-05-20; v2 2025-05-22)
- **URL:** https://arxiv.org/abs/2505.14464 (dataset card: https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled)
- **Source type:** paper (with dataset card)
- **Relevant topics:** teacher selection, reasoning distillation, verified traces, response length distribution, perplexity of training data, adaptive generation length, long-CoT SFT

## Abstract
The authors distill verified outputs from three teacher models, AM-Thinking-v1, Qwen3-235B-A22B, and DeepSeek-R1, on a shared set of 1.89 million queries, producing three parallel datasets. The AM-Thinking-v1 data has greater token-length diversity and lower perplexity. Students trained on each dataset are evaluated on AIME2024, AIME2025, MATH500, and LiveCodeBench. The AM-Thinking-v1 student scores highest on all four (84.3, 72.2, 98.4, 65.9) and produces longer responses on harder tasks and shorter ones on easier tasks. The AM-Thinking-v1 and Qwen3-235B-A22B datasets are released.

## Key Contributions
- A parallel corpus: the same 1.89M queries answered by three teachers under one verification pipeline (§2.2).
- Distribution analysis of the three datasets: category shares by tokens, math token-span histogram, mean length, and perplexity (§2.3, Figs. 2-5).
- Three students trained with identical SFT settings, so the varied factor is the teacher data, including its per-teacher PPL filter (§2.2, §3.1, Table 1).
- An analysis of generation length by benchmark, showing length adaptation in the AM-Thinking-v1 student (Table 2).
- Release of the AM-Thinking-v1 and Qwen3-235B-A22B distilled sets; R1 data is not released because R1 distillation data is "easily accessible" (footnote 4).

## Key Figures/Tables to Study
- **Fig. 2:** instance-level category shares (shared) and token-level shares per teacher.
- **Fig. 3:** math token-span histogram per teacher. **Figs. 4-5:** box plots of token count and perplexity.
- **Table 1:** student accuracy per teacher. **Table 2:** student average generation length per benchmark.
- **Fig. 6:** training loss curves of the three students.

## Technical Details
- **Query pool and preprocessing:** open datasets in math, code, science, instruction following, multi-turn, and general reasoning; exact dedup; removal of high-Unicode-ratio, empty or incomplete, and URL- or table-containing queries; exact-match and bge-m3 semantic decontamination at similarity > 0.9 against the evaluation set, "e.g., AIME2024" (§2.1).
- **Distillation rule:** each teacher distills each query independently; "the distillation process was repeated on the same query until the generated response satisfied the verification criterion (i.e., the verification score ≥ 0.9)" (§2.2). Sampling temperature, top-p, maximum generation length, and number of attempts for distillation are not reported.
- **Verifiers (§2.2):** math, Math-Verify then Qwen2.5-7B-Instruct if necessary, binary; code, sandbox with up to 10 test cases (Python assert and input-output, C++ input-output), score = pass rate; science, Qwen2.5-7B-Instruct similarity to reference, normalized; IF, ifeval validator with missing constraints added by Qwen2.5-72B-Instruct, mean pass rate; multi-turn and general, Decision-Tree-Reward-Llama-3.1-8B coherence, correctness, helpfulness as a normalized composite. One threshold of 0.9 for all categories.
- **Quality filters (§2.2):** perplexity from "a strong 32B language model" [30] with a different threshold per teacher (values not reported); removal of 20-token n-grams occurring more than 20 times; even turn count for conversations; presence of both think and answer segments.
- **Instance shares (Fig. 2):** general chat 790.3K (41.8%), math 558.2K (29.5%), code 324.0K (17.1%), science 159.7K (8.4%), IF 58.5K (3.1%). "General chat" includes multi-turn and other data.
- **Token shares (Fig. 2):** AM-Thinking-v1 math 2,097.1M (33.4%), general chat 2,077.8M (33.1%), code 1,677.1M (26.7%); Qwen3-235B-A22B math 2,342.3M (38.0%), general chat 1,779.9M (28.9%), code 1,691.4M (27.5%); DeepSeek-R1 math 2,112.6M (41.0%), code 1,353.4M (26.3%), general chat 1,331.1M (25.8%).
- **Length:** mean tokens per instance Qwen3-235B-A22B 4,196.7, DeepSeek-R1 3,784.8, AM-Thinking-v1 3,757.3 (§2.3, Fig. 4). On math, AM-Thinking-v1 has many responses under 1,024 tokens and many over 10,240; Qwen3 is shifted to longer spans; R1 is concentrated between 1k and 8k (§2.3, Fig. 3).
- **Perplexity:** mean PPL AM-Thinking-v1 2.5, DeepSeek-R1 2.9, Qwen3-235B-A22B 3.0 (§2.3, Fig. 5).
- **Evaluation (§3.2):** maximum generation 49,152 tokens; temperature 0.6; top-p 0.95; 64 samples per question for AIME2024 and AIME2025, 16 for LiveCodeBench (queries from 2024-10 to 2025-02), 4 for MATH500; pass@1 from these samples. A unified system prompt requires `<think>` and `<answer>` tags; AIME and MATH500 prompts add "Let's think step by step and output the final answer within \box."
- **Results (Table 1, AM / Qwen3 / R1 teacher):** AIME2024 84.3 / 79.4 / 70.9; AIME2025 72.2 / 62.2 / 52.8; MATH500 98.4 / 93.9 / 95.8; LiveCodeBench 65.9 / 59.6 / 57.0.
- **Generation length (Table 2, tokens):** AIME2024 15,273.8 / 13,516.4 / 11,853.5; AIME2025 18,199.2 / 16,975.7 / 13,495.9; MATH500 3,495.7 / 6,429.4 / 3,613.0; LiveCodeBench 23,426.9 / 13,576.7 / 30,731.0.
- **Training loss:** the AM-Thinking-v1 student has the lowest loss throughout training (§4, Fig. 6).
- **Teacher reference scores (dataset card "Benchmark Performance"; evaluation settings not stated there):** AM-Thinking-v1 85.3 / 74.4 / 70.3, Qwen3-235B-A22B 85.7 / 81.5 / 70.7, DeepSeek-R1 79.8 / 70.0 / 64.3, Qwen3-32B 81.4 / 72.9 / 65.7 (AIME2024 / AIME2025 / LiveCodeBench). Student minus teacher on AIME2024: −1.0 (AM), −6.3 (Qwen3), −8.9 (R1) (derived).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Three Qwen2.5-32B base students (AM-Thinking-v1 / Qwen3-235B-A22B / DeepSeek-R1 data), identical settings | 32B | distill-SFT | queries; responses per teacher per query | 1.89M; one response per teacher, regenerated until verify_score ≥ 0.9 | arXiv:2505.14464v2 §2.2 | verified 2026-09-14 | no ablation of the 0.9 threshold reported |
| same | 32B | distill-SFT | output tokens per dataset | 6,286.3M / 6,158.3M / 5,153.1M (sum of Fig. 2 category token counts) | Fig. 2 | derived | not applicable |
| same | 32B | distill-SFT | peak LR; schedule | 8e-5; cosine, warmup 5% of total steps, decay to zero | §3.1 | verified 2026-09-14 | no ablation; "As suggested by [2, 4]" ([[deepdistill]], [[am-thinking-v1]]) |
| same | 32B | distill-SFT | epochs; global batch; max length; packing | 2; 64; 32k tokens; sequence packing; samples over 32k excluded | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 32B | distill-SFT | loss target | multi-turn dialogues: only the final response containing the reasoning process | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 32B | distill-SFT | optimizer, weight decay, seeds, compute; per-teacher PPL thresholds; teacher sampling settings | not reported | checked §2-§5, dataset card | not reported | none |
| same | 32B | eval-gate | max tokens; temperature; top-p; samples (AIME / LCB / MATH500) | 49,152; 0.6; 0.95; 64 / 16 / 4 | §3.2 | verified 2026-09-14 | not applicable |

## Findings relevant to generality, distillation
- **Distillation setup:** stage is SFT from a base model. Prompts are the shared 1.89M queries, 41.8% general chat by count. Teachers are sampled until one response per query passes verification (§2.2). Quality control is category verification, per-teacher PPL filtering, n-gram repetition removal, and structural checks (§2.2).
- **Teacher effect:** the teacher changed AIME2025 by 19.4 points and LiveCodeBench by 8.9 points between the best and worst student (Table 1; derived). Qwen3-235B-A22B data produced a weaker student than data from AM-Thinking-v1, a 32B-scale model (ref [4] title), on all four benchmarks. Result (single study, one run each).
- **Length adaptation:** the AM-Thinking-v1 student uses 3,495.7 tokens on MATH500 and 18,199.2 on AIME2025; the Qwen3 student uses 6,429.4 on MATH500. The authors attribute adaptation to the wider length distribution of the AM data (Interpretation, §4).
- **Measurement limits:** §3.2 says benchmarks cover "general chatbot performance", but Tables 1-2 contain only math and code benchmarks. No instruction-following or chat result is reported for any student. AM-Thinking-v1 is the authors' own model (ref [4]).
- **Negative samples:** responses below 0.9 are regenerated and discarded (negative marginal value). No negative is used as gradient.

## Connections
- [[deepdistill]]: same team's earlier pipeline; source of the verifiers, filters, and 8e-5 LR.
- [[am-thinking-v1]]: the AM teacher. [[qwen-3]]: the Qwen3-235B-A22B teacher. [[deepseek-r1]]: the R1 teacher.
- [[am-deepseek-r1-distilled-1-4m]]: reference [30], cited for the 32B PPL model.
- [[am-deepseek-r1-0528-distilled]]: later a-m-team distilled release.
- [[capacity-gap-law-distillation]]: teacher-student size relation; here the 32B-scale AM-Thinking-v1 teacher produced a stronger student than Qwen3-235B-A22B.
- [[naturalthoughts]], [[unveiling-cot-distillation-factors]], [[open-thoughts]]: other studies of which reasoning traces to distill.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2505.14464 (v2, 2025-05-22; v1 2025-05-20) and the AM-Thinking-v1-Distilled dataset card (fetched 2026-09-14).
- Audit claims not found in the source: teacher sampling at temperature 0.6, top-p 0.95, 32k (these are evaluation settings in §3.2, with a 49,152-token cap; distillation sampling is not reported); "deduplicated with bge-m3 at 0.9" (bge-m3 is used for decontamination against the evaluation set, §2.1); "AM-DeepSeek-R1-Distilled-1.4M release had 900k newly distilled R1 responses plus 500k" (a different artifact, not in this paper).
- Internal inconsistency recorded: §2.2 says each query yields "up to three" outputs, while the end of §2.2 says every query is paired with responses from all three teachers.
- Unexplained count difference (derived): Fig. 2 token totals order AM-Thinking-v1 (6,286.3M) above Qwen3-235B-A22B (6,158.3M), while Fig. 4 mean tokens per instance order Qwen3 (4,196.7) above AM (3,757.3); with 1.89M instances in each set these cannot both describe the same token count, and the paper does not state how each figure counts tokens.
