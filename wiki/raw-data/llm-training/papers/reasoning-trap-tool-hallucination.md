<!-- scope: The Reasoning Trap (arXiv:2510.22977) — SimpleToolHalluBench; reasoning RL (tool or math), R1 distillation, and thinking modes raise tool hallucination; ablations, CKA/probe analysis, prompt and DPO mitigations
     deps: [[grpo]], [[dpo]]
     see-also: [[bfcl]], [[toolrl]], [[scaling-reasoning-losing-control-mathif]], [[why-language-models-hallucinate]], [[transferability-of-llm-reasoning]]
-->

# The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination
- **Core Insight:** On Qwen2.5-7B-Instruct, think-then-act GRPO on SynTool (ReCall) raised the No-Tool-Available hallucination rate from 34.8% to 90.2% and the Distractor-Tool rate from 54.7% to 100.0%, while BFCL Multi-Turn rose from 13.6 to 23.5 and IFEval moved from 62.4 to 59.8 (Table 3); GRPO on GSM8K math alone also raised both rates (Fig. 3).
- **Guideline:** When reasoning RL or reasoning distillation is applied to a model that will also call tools, measure abstention on prompts whose required tool is missing or replaced by an irrelevant one, because in this study tool-calling (BFCL) and instruction-following (IFEval, ComplexBench) scores did not reveal the rise in tool hallucination (§4.4.2, Table 3).
- **Authors:** Chenlong Yin, Zeyang Sha, Shiwen Cui, Changhua Meng, Zechao Li (v2; v1 lists the first four)
- **Year:** 2025 (arXiv v1 2025-10; v2 2026-04; arXiv comment on v2: "Accepted to ACL 2026 Main")
- **URL:** https://arxiv.org/abs/2510.22977 (code: https://github.com/albert-y1n/Reasoning_Trap)
- **Source type:** paper
- **Relevant topics:** tool hallucination, abstention, reasoning RL side effects, GRPO, R1 distillation, thinking mode, DPO mitigation, representation drift

## Abstract
The paper asks whether strengthening reasoning increases tool hallucination, defined as fabricating non-existent tools or misusing available but irrelevant tools (§1). It introduces SimpleToolHalluBench with two settings: no tool available, and only a distractor tool available. Across RL, distillation, and toggleable reasoning modes, task-performance gains are accompanied by higher tool hallucination rates; the effect appears after RL on non-tool tasks (mathematics), and controlled ablations associate it with the reasoning step rather than with RL training in general. Prompt engineering gives marginal reduction, and DPO reduces hallucination with a drop in task utility. The authors conclude that training objectives must jointly optimize capability and reliability (Abstract, §7).

## Key Contributions
- SimpleToolHalluBench: No-Tool-Available (NTA) and Distractor-Tool (DT) tasks, 296 × 2 tool-query pairs (§3.1; App. A.1).
- Four experiments: tool-specific RL (§4.1), math-only RL (§4.2), distillation and thinking modes (§4.3, App. B), ablations that remove the reasoning step and test instruction-following degradation (§4.4).
- Representation analysis with CKA and linear probes (§5; App. E).
- Mitigation study: system-prompt instruction and DPO on ReCall-7B (§6).

## Key Figures/Tables to Study
- Fig. 2 and Fig. 3: task reward and hallucination rate over checkpoints (ReCall on SynTool; GRPO on GSM8K).
- Table 1 and Table 5: instruct vs reasoning variants. Tables 2-3: ablations. Table 4: mitigations.
- Fig. 4-5: CKA per layer and per-component discrimination scores.

## Technical Details
- Metric: R_NTA = H_NTA / N_NTA and R_DT = H_DT / N_DT, where H is the count of responses flagged as hallucinated by the judge and N the number of samples in the task set (Eq. 1). The judge is DeepSeek-R1; the authors inspected "a subset" manually and report that "the vast majority" agreed with humans; no agreement rate is given (App. A.4).
- Judge rules: answering the query directly without the required tool, claiming the tool exists, or inventing a tool counts as hallucination; explaining how the user could use the tool if they had it does not (App. A.4 prompts).
- Construction: 349 tools from Agent-SafetyBench; ChatGPT-4o writes an explicit-invocation query per tool (NTA, no tools in the system prompt) and an implicit-requirement query (DT, one irrelevant tool provided); two annotators plus a third reviewer removed queries answerable from internal knowledge, leaving 296 × 2 pairs (App. A.1). §3.1 states "selecting 296 tools".
- §4.1: ReCall reproduction on Qwen2.5-7B-Instruct, GRPO on the SynTool training split only, checkpoints every 100 steps; SynTool validation reward and both hallucination rates increase together (Fig. 2; App. D). Fig. 2 plots checkpoints to step 1,200 and Fig. 3 to step 400.
- §4.2: GRPO on GSM8K; GSM8K validation accuracy and both hallucination rates increase over training (Fig. 3).
- Table 1 (R_NTA / R_DT, %): Qwen2.5-7B Instruct 34.8 / 54.7 vs R1-Distill 74.3 / 78.7; Llama3.1-8B Instruct 62.5 / 99.7 vs R1-Distill 96.3 / 100; Qwen3-8B think off 4.1 / 36.2 vs on 5.4 / 56.8; Qwen3-32B off 5.1 / 46.6 vs on 8.8 / 50.7.
- Table 5 (App. B): Qwen3-4B-2507 Instruct 3.4 / 24.0 vs Thinking 29.4 / 32.1; Qwen3-235B-2507 3.7 / 23.3 vs 6.1 / 30.7; DeepSeek V3 10.8 / 33.8 vs R1 17.6 / 42.6; Kimi-K2 Instruct 1.0 / 15.5 vs Thinking 4.4 / 21.3.
- Table 2 (same data, reward, optimizer, hyperparameters): base 34.8 / 54.7, reward 0.22; direct tool-use RL without a `<think>` block 41.4 / 63.6, reward 0.28; think-then-act RL 90.2 / 100.0, reward 0.45 (§4.4.1).
- Table 3: IFEval 62.4 → 59.8, ComplexBench 60.8 → 59.4, BFCL Multi-Turn base subset 13.6 → 23.5 after ReCall GRPO (§4.4.2).
- CKA (GSM8K-GRPO model vs base): in-distribution inputs keep CKA > 0.9 in all layers; SimpleToolHalluBench inputs fall below 0.75 in early and middle layers (§5.1, Fig. 4). CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L)), with K = XXᵀ and L = YYᵀ the Gram matrices of the two models' representations and HSIC the Hilbert-Schmidt Independence Criterion.
- For the ReCall (SynTool) model, SynTool and GSM8K inputs both drop to about 0.86-0.88 in early layers and track each other; MLP-output CKA is lower than attention-output CKA (App. E.1-E.2, Figs. 6-7).
- Probes: discrimination score = linear-classifier accuracy − 0.5 on correct vs hallucinated activations (Eq. 2); residual stream from layer 20 onward exceeds 0.14, attention outputs average 0.02, MLP outputs 0.04 (§5.2, Fig. 5).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct + ReCall reproduction | 7B | RL | algorithm; data | GRPO with outcome reward; SynTool training split only | arXiv:2510.22977v2 §4.1, App. C.1, App. D | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B-Instruct + ReCall reproduction | 7B | RL | checkpoint cadence | every 100 steps | v2 §4.1 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B-Instruct GRPO-GSM8K | 7B | RL | data | GSM8K | v2 §4.2 | verified 2026-09-14 | no ablation reported |
| both RL runs | 7B | RL | LR, KL coefficient, clip ε, group size, max length, temperature, steps | not reported | checked v2 §4, App. C.1 (generic formula only), App. D | not reported | — |
| ReCall-7B + DPO | 7B | preference | pair types | tool missing: chosen honest abstention, rejected fabricated tool call and output; tool present: chosen correct call, rejected needless refusal | v2 §6.1, App. C.2 | verified 2026-09-14 | no ablation reported |
| ReCall-7B + DPO | 7B | preference | β, number of pairs, LR, epochs | not reported | checked v2 §6, App. C.2 | not reported | — |

## Findings relevant to generality, negative feedback, agentic training, distillation
- Generality: the target reward and a held-out tool-calling benchmark improved while abstention on missing tools degraded; IFEval and ComplexBench changed by −2.6 and −1.4 points (Table 3). The authors call tool hallucination "a distinct failure mode not captured by existing instruction-following or tool-calling benchmarks" (§4.4.2). Result (single study).
- Agentic training: in Table 2 the direct tool-use regime also reached a lower task reward (0.28 vs 0.45), so the two regimes differ in task gain as well as in the reasoning step.
- Distillation: R1-distilled 7B/8B models hallucinate more than their instruct counterparts (Table 1); the authors state that hallucination tendencies "transfer via distillation" (§4.3).
- Negative feedback (negative as gradient, DPO rejected term): R_NTA 90.2 → 55.8, R_DT 100.0 → 71.4, SynTool reward 0.45 → 0.34. The prompt "You must not use any tools that are not explicitly provided to you." gave 87.5 / 98.9 / 0.44 (Table 4; App. A.2.3).
- Limitations stated: single-step tool invocation only; no complete causal account; only prompt engineering and DPO tested (Limitations).

## Connections
- [[grpo]], [[dpo]] — the training and mitigation objectives (App. C).
- [[bfcl]], [[ifeval]] — the tool-calling and instruction-following checks in Table 3.
- [[toolrl]] — tool-calling RL work cited among RL frameworks for tool policies (§2).
- [[deepseek-r1]], [[qwen-3]], [[kimi-k2]], [[deepseek-v3]] — models compared in Tables 1 and 5.
- [[training-verifiers-to-solve-math-word-problems]] — GSM8K, the non-tool RL task (§4.2).
- [[scaling-reasoning-losing-control-mathif]] — another study of side effects of reasoning training (instruction following).
- [[why-language-models-hallucinate]], [[finetuning-new-knowledge-hallucination]] — hallucination mechanisms outside tool use.
- [[transferability-of-llm-reasoning]] — cross-domain effects of math RL vs SFT.
- [[fission-grpo]], [[openai-confessions]] — other approaches to tool robustness and honesty training.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.22977 (v2 PDF, 2026-04-17); v1 PDF (2025-10-27) read to compare wording.
- Version differences: v1 abstract claims "a causal relationship", hallucination rising "proportionally with task performance gains", an effect under supervised fine-tuning, and that utility is "consistently" degraded; v2 uses "consistently accompanied by", "systematically associated", and "unavoidably degrades utility", and adds §4.4 (Tables 2-3), Llama3.1-8B rows in Table 1, App. B (Table 5), and App. E.
- Audit claims not found in the source (v2): "raises tool hallucination in proportion to task gains" (v1 abstract only); "reasoning training disproportionately damages tool-reliability representations" (v1 wording; v2 reports larger representational shifts on tool inputs in §5.1, and App. E.2 finds comparable drift on tool and math inputs for the ReCall model); "this negative transfer must be gated when reasoning RL and agentic training share a model" (a course recommendation, not in the paper).
- Source inconsistencies: 296 tools selected (§3.1) vs 349 selected and 296 retained (App. A.1).
- Not reported by the source: RL and DPO hyperparameters, DPO pair count, total training steps in text, judge agreement rate.
