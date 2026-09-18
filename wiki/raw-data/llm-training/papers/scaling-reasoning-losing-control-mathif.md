<!-- scope: MathIF (Fu et al., May 2025) — benchmark of Python-verifiable format constraints on math problems for 23 reasoning models; controlled SFT / SFT+RL / cold-RL runs on Qwen2.5 bases; CoT-length and constraint-repetition interventions
     deps: [[ifeval]], [[grpo]]
     see-also: [[scaling-reasoning-losing-control-mathif-recipe]], [[ifbench]], [[qwen-3]], [[deepseek-r1]], [[s1]], [[deepscaler]], [[transferability-of-llm-reasoning]]
-->

# Scaling Reasoning, Losing Control: Evaluating Instruction Following in Large Reasoning Models
- **Core Insight:** On MathIF (420 math queries with 1–3 Python-verifiable constraints), the best of 23 reasoning models, Qwen3-14B, satisfies all constraints in 50.71% of queries; in the authors' controlled runs on four Qwen2.5 bases, long-CoT SFT and GRPO lowered both hard and soft constraint accuracy in 15 of 16 trained variants while raising average math accuracy in 15 of 16 (Table 3; Table 4, counts from the table).
- **Guideline:** When a reasoning-oriented SFT or RL stage is added to a model meant to stay general, track a verifiable instruction-following metric next to reasoning accuracy, because in these runs reasoning gains came with lower constraint compliance (Table 4) and raising the RL maximum rollout length from 1k to 8k tokens raised math accuracy from 28.73 to 39.82 while lowering hard accuracy from 19.05 to 14.29 (Table 5).
- **Authors:** Tingchen Fu, Jiawei Gu, Yafu Li, Xiaoye Qu, Yu Cheng (Renmin University of China; Shanghai AI Laboratory; The Chinese University of Hong Kong)
- **Year:** 2025 (arXiv v1 2025-05; v2 2025-05-25; preprint)
- **URL:** https://arxiv.org/abs/2505.14810 ; code and data https://github.com/TingchenFu/MathIF
- **Source type:** paper
- **Relevant topics:** instruction following, controllability, large reasoning models, long chain-of-thought, reasoning RL side effects, distillation side effects, evaluation

## Abstract
Instruction following is needed to align language models with user intent, but it has been measured little for reasoning-oriented models. The authors introduce MathIF, a benchmark for instruction following inside mathematical reasoning tasks. They report a consistent tension between reasoning capacity and controllability: models tuned on distilled long chains of thought or trained with reasoning-oriented RL often follow instructions worse, especially as generation length grows. Simple interventions partly restore compliance but reduce reasoning performance. The authors argue that current training paradigms need to become instruction-aware. Code and data are released.

## Key Contributions
- MathIF: 15 constraints in 4 categories, composed into 30 dual and 15 triple constraints, applied to 420 problems from five math sources (§1, §3, Table 2, Table 14).
- Hard accuracy (HAcc, all constraints met) and soft accuracy (SAcc, share of constraints met), reported next to answer correctness (§3, Eq. 1).
- An evaluation of 23 reasoning models from 0.6B to 70B (§4, Table 3).
- Controlled comparisons of SFT-only, SFT+RL, and cold-start RL on Qwen2.5-1.5B/7B and Qwen2.5-Math-1.5B/7B (§5.2, Table 4); settings in [[scaling-reasoning-losing-control-mathif-recipe]].
- Length interventions: budget forcing, RL maximum rollout length, and repeating the constraint after the CoT (§5.3, Fig. 7, Tables 5–6).

## Key Figures/Tables to Study
- Table 3 (23 models: HAcc, SAcc, correctness with and without constraints); Figure 4 (joint correct/followed breakdown); Figure 6 (compliance by CoT-length bin).
- Table 4 (training paradigms); Table 5 (RL rollout length); Table 6 (constraint repetition); Table 13 (IFEval and FollowBench, instruct model vs derived reasoning model); Table 7 (hyperparameters).

## Technical Details
**Benchmark (§3)**
- Categories: length, lexical (language, keywords), format (bullets, sections, case, punctuation), affix (prefix, suffix, quotation wrap); inspired by IFEval and ComplexBench; every constraint is checked by a Python program (§3, Tables 1 and 14).
- Compositions are sampled from the Cartesian products C² and C³ after manually removing incompatible and same-subtype combinations (§3).
- Problems: 90 each from GSM8K, MATH-500, Minerva, OlympiadBench, and all 60 AIME 2024–2025 problems; each source gets equal single, double, and triple subsets, giving 140 + 140 + 140 = 420 queries (§3, Table 2). Samples were manually checked for constraints that contradict the problem (§3).
- Metrics: for a query with n constraints and I(C_i) = 1 if constraint i is met, else 0, HAcc = ∏_{i=1..n} I(C_i) and SAcc = (1/n) Σ_{i=1..n} I(C_i); both are averaged over queries (Eq. 1). Correctness is an exact final-answer match regardless of constraints, measured with constraints in the prompt unless stated (§3).
- Worked example (derived from Eq. 1): a triple-constraint query with 2 of 3 constraints met has HAcc = 0 and SAcc = 2/3 ≈ 0.67.
- Decoding for all models: nucleus sampling T = 1.0, p = 0.95, maximum generation 16,384 tokens, vLLM (§4). Number of samples per query and seeds are not reported.

**Results on 23 models (§4.1, Table 3)**
- Highest: Qwen3-14B 50.71 HAcc / 67.06 SAcc. Lowest HAcc: Qwen2.5-Math-1.5B-Instruct 7.62. Qwen3 has the top HAcc in each size group (Qwen3-4B 44.05; Qwen3-14B; Qwen3-32B 43.81).
- DeepSeek-R1-Distill-Llama-70B (41.43 HAcc) scores below Qwen3-4B (44.05) at more than 15× the size (§4.1).
- Adding constraints lowers correctness for 21 of 23 models by 0.96 to 23.33 points; Qwen2.5-Math-1.5B-Instruct (+0.54%) and Open-Reasoner-Zero-32B (+3.27%) are the exceptions (§4.1, Table 3). The largest relative drops are DeepSeek-R1-Distill-Qwen-1.5B (−40.09%) and DeepSeek-R1-Distill-Llama-8B (−39.04%) (Table 3).
- Models without `<think>` separation (Qwen2.5-Math-Instruct, SimpleRL-Zoo) generally score below same-size peers; the authors suggest separation may help (§4.1).
- Harder sources get lower HAcc, e.g., Qwen3-14B 71.11 on GSM8K vs 20.00 on AIME (Table 9). More constraints lower HAcc for all 23 models while SAcc stays level or rises, e.g., Qwen3-0.6B HAcc 48.57 / 22.86 / 12.14 for single / double / triple (§4.2, Fig. 3, Table 8).
- Joint outcomes for Qwen3-0.6B: correct and followed 8.81%, correct and unfollowed 23.33%, incorrect and followed 19.05%, incorrect and unfollowed 48.81% (Fig. 4). The authors state that correct-and-followed is smaller than both correct-and-unfollowed and incorrect-and-followed (§5.1); in the extracted figure labels this holds for three of the four models, while Open-Reasoner-Zero-7B has 7.38% correct-and-followed and 6.19% incorrect-and-followed (Fig. 4).
- For three models (DeepSeek-R1-Distill-Llama-8B, Qwen3-0.6B, Qwen3-32B), HAcc and SAcc decline across six bins of thinking length measured between `<think>` and `</think>` (§5.1, Fig. 6).

**Controlled training (§5.2, Table 4; App. F Table 12)**
- Data: the DeepScaleR dataset, approximately 40k math reasoning samples. SFT data: QwQ-32B traces, dropping wrong answers and CoTs over 8,192 tokens, leaving 18k examples. RL: GRPO with verifiable outcome reward; a format-aware variant grants 0.1 when the output includes the special reasoning tokens and 1.0 for a correct solution; whether the two are summed is not stated (§5.2).
- Qwen2.5-7B, HAcc / SAcc / average math accuracy: base 15.95 / 33.13 / 13.59; +SFT 7.86 / 21.03 / 23.10; +SFT+RL 7.62 / 21.07 / 32.82; +cold-RL 10.48 / 27.26 / 28.39; +cold-RL with format reward 14.52 / 32.50 / 24.80 (Table 4).
- The only trained variant above its base on HAcc and SAcc is Qwen2.5-1.5B cold-RL with format reward (10.95 / 28.49 vs 10.00 / 27.26); the only variant below its base on math accuracy is Qwen2.5-Math-1.5B +SFT (14.39 vs 18.91) (Table 4). Math accuracy is the mean of AIME2024, AIME2025, AMC2023, Minerva, OlympiadBench (App. F).

**Length interventions (§5.3)**
- Budget forcing (appending "Wait" N = 2 to 8 times) on DeepSeek-R1-Distill-Qwen-1.5B lowers SAcc on the GSM8K subset as N grows (Fig. 7).
- Continued RL of DeepSeek-R1-Distill-Qwen-1.5B for three epochs, with overlong rollouts truncated and given no outcome reward: HAcc / SAcc / math accuracy for the original 17.14 / 36.62 / 36.13; 1k 19.05 / 39.88 / 28.73; 2k 16.43 / 36.75 / 36.32; 4k 16.91 / 35.87 / 40.03; 8k 14.29 / 34.13 / 39.82 (Table 5). The trend is not monotone at every step (2k→4k HAcc; 4k→8k math accuracy).
- Repeating the constraint after an appended "Wait": Qwen3-32B HAcc 43.81 → 59.29, correctness 70.00 → 63.81; Open-Reasoner-Zero-7B HAcc 13.57 → 14.53, correctness 51.90 → 30.00; DeepSeek-R1-Distill-Qwen-1.5B HAcc 17.14 → 21.66, correctness 31.67 → 22.38 (Table 6). The authors call the correctness cost modest (§5.3).
- General benchmarks, instruct model → derived reasoning model, IFEval prompt-level strict: Llama-3.3-70B-Instruct 90.38 → DeepSeek-R1-Distill-Llama-70B 78.74; Qwen2.5-32B-Instruct 80.96 → s1-32B 58.04; Qwen2.5-Coder-32B-Instruct 80.03 → OlympicCoder-32B 57.11. FollowBench (GPT-4o-mini judge): 61.82 → 49.26; 60.12 → 51.46; 59.14 → 46.70 (App. G, Table 13).

## Findings relevant to generality, long context, distillation
- Generality: in the controlled runs, reasoning-oriented SFT and RL on math data reduced a skill outside the training objective, constraint following (Result (single study), Table 4). Table 13 compares released models trained on different data, so it is not a controlled comparison.
- Long generation: compliance falls with thinking length in observational bins (Fig. 6) and under two interventions that lengthen it (Fig. 7, Table 5). The authors' explanation, that distance between instruction and answer dilutes attention to the constraint, is tested only by the constraint-repetition experiment (Interpretation, §5.1, §5.3).
- Distillation: SFT-only distilled R1 students show the largest relative correctness drops under constraints (Table 3), and the authors link this to limits of SFT (§5.1, Interpretation). s1-32B, also SFT-only, drops 3.04% (Table 3). In §5.2 the distillation stage is SFT, prompts are DeepScaleR math problems, the teacher is QwQ-32B, and quality control is an answer-correctness filter plus an 8,192-token length cap; teacher sampling settings are not reported (§5.2).
- Limits stated by the authors: text-only models, and GRPO as the only RL algorithm (App. B).

## Connections
- [[ifeval]] — constraint design source and one of the two general benchmarks in Figure 1 and Table 13.
- [[ifbench]] — later work on verifiable instruction following and generalization to unseen constraints.
- [[qwen-3]], [[qwen3-2507-instruct-thinking-split]] — Qwen3 models lead each MathIF size group; Qwen later released separate Instruct and Thinking models.
- [[deepseek-r1]], [[s1]], [[l1-lcpo]], [[open-reasoner-zero]], [[simplerl-zoo]] — sources of evaluated models; s1 also supplies budget forcing.
- [[deepscaler]], [[qwen-qwq-traces]], [[grpo]], [[verl-grpo]] — training data, QwQ-series teacher traces, RL algorithm, and the framework whose defaults Table 7 mostly follows (App. D).
- [[sft-memorizes-rl-generalizes]] — cited for the limits of SFT (§5.1).
- [[transferability-of-llm-reasoning]], [[overthinking-o1-like-llms]], [[inverse-scaling-test-time-compute]], [[reasoning-trap-tool-hallucination]] — other measurements of side effects of reasoning training or long reasoning.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2505.14810 (arXiv v2, 2025-05-25; v1 2025-05-20), full PDF text including Appendices A–H.
- Audit claims not found in the source: "pairs with Qwen3's observation that General RL trades peak reasoning for versatility" — MathIF does not discuss the Qwen3 General RL stage.
- Inconsistencies inside the source: OlympicCoder-32B is marked SFT-only (†) in Tables 8–10 but not in Table 3; Appendix E refers to a nonexistent "Section 4.3"; the Table 8 caption says results by math source while its columns are constraint counts.
- Not reported by the source: samples per query and seeds for the benchmark, RL step counts, teacher sampling settings, loss masking and packing for SFT.
