<!-- scope: APIGen (Salesforce, 2024) — single-turn function-calling data synthesis with format, execution, and semantic checks; xLAM-1B/7B (FC) SFT results on BFCL
     deps: [[toolllm]]
     see-also: [[apigen-mt]], [[xlam]], [[bfcl]], [[toolace]]
-->

# APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
- **Core Insight:** Generated function-calling samples pass three checks in sequence (format, execution, LLM semantic check); a 7B model fine-tuned on the verified data ranked 6th on BFCL (85.65 overall, leaderboard of 2024-06-15), and adding back samples that failed the semantic or execution check lowered BFCL accuracy (Table 2, Fig. 5).
- **Guideline:** When synthetic tool-call data comes from weaker generators, apply execution and semantic filtering before SFT, because pass rates fell to 34.42% for DeepSeek-Coder-33B-Inst versus 84.15% for DeepSeek-V2-Chat (Table 1) and training on the failed samples reduced BFCL accuracy more for the 1B model than the 7B model (Fig. 5).
- **Authors:** Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, et al. (Salesforce AI Research)
- **Year:** 2024 (arXiv v1 2024-06; the arXiv record lists no venue)
- **URL:** https://arxiv.org/abs/2406.18518
- **Source type:** paper
- **Relevant topics:** function calling, synthetic data, execution verification, LLM-as-judge filtering, SFT, BFCL

## Abstract
APIGen is a data generation pipeline for function-calling agents. The authors collect 3,673 executable APIs in 21 categories and generate query-answer pairs in which each answer is a JSON function call. Every sample is checked in three stages: format checking, execution of the call, and semantic verification by another LLM. Models trained on the data, including a 7B model, reach high positions on the Berkeley Function-Calling Benchmark (BFCL), and a 1B model scores above GPT-3.5-Turbo and Claude-3 Haiku. The authors release a 60,000-entry dataset (xlam-function-calling-60k on Hugging Face).

## Key Contributions
- A modular pipeline: API sampler, example (seed QA) sampler, prompt sampler, LLM generator, and a three-stage verifier; verified samples are added back to the seed set (§3.1, Fig. 2).
- Four query styles taken from BFCL categories: simple, multiple, parallel, parallel multiple (§3.3).
- An API library of 3,539 cleaned ToolBench REST APIs plus 134 Python functions = 3,673 APIs (§4.1).
- Filtering statistics for four open generator models (Table 1) and a released dataset of about 60,000 samples (§4.2).
- xLAM-1B (FC) and xLAM-7B (FC) trained on the data, with a filtering ablation (§5.2, Table 2, Fig. 5).

## Key Figures/Tables to Study
- Table 1 (§4.2): verified count and failures per stage for each generator model.
- Table 2 (§5.2): BFCL leaderboard as of 2024-06-15 with AST and executable sub-scores.
- Fig. 5 (§5.2): BFCL overall accuracy with verified data, +Fail Semantic data, +Fail Execution data.
- App. B.2: full semantic-checker prompt; App. B.3: relevance-detection data and training hyperparameters.

## Technical Details
- **API sources:** ToolBench's 16,464 REST APIs in 49 categories were filtered (remove bad docs and no-parameter APIs, test accessibility via the StableToolBench server, regenerate noisy docstrings) to 3,539 REST APIs; 134 Python functions were added; categories were merged into 21 (§4.1, Fig. 4).
- **Sampling:** the number of APIs and seed examples per generation is drawn at random from a predefined range; the range is not reported (§3.3). Prompt templates include ambiguous or misspelled user requests (§3.3). One LLM call can return several QA pairs ("batching", App. B.1).
- **Stage 1, format checker:** output must be JSON with "query" and "answer" fields (the authors usually add a "thought" field with the stated aim of raising pass rate); calls with functions or arguments not present in the given APIs are removed (§3.2).
- **Stage 2, execution checker:** Python functions run in a separate subprocess; REST APIs are called and status codes read; failures (type errors, invalid parameters, runtime errors, timeout, missing arguments) are removed (§3.2). No timeout value is reported.
- **Stage 3, semantic checker:** another LLM receives the functions, query, calls, and execution results and returns JSON {"thought", "pass": yes/no} under five fail rules (§3.2, App. B.2). The checker model is not named.
- **Generators:** DeepSeek-V2-Chat (236B), DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, Mixtral-8x7B-Inst; 40,000 target samples each; temperature 0.7 (§4.2).
- **Filtering statistics (Table 1):** verified / fail format / fail execution / fail semantic / pass rate:
  DeepSeek-Coder-33B-Inst 13,769 / 4,311 / 15,496 / 6,424 / 34.42%; Mixtral-8x7B-Inst 15,385 / 3,311 / 12,341 / 7,963 / 38.46%; Mixtral-8x22B-Inst 26,384 / 1,680 / 5,073 / 6,863 / 65.96%; DeepSeek-V2-Chat 33,659 / 817 / 3,359 / 2,165 / 84.15%. The four verified counts sum to 89,197.
- **Release:** about 60,000 samples from Mixtral-8x22B-Inst and DeepSeek-V2-Chat only (§4.2); license CC BY 4.0 (App. A.1).
- **Human check:** 3 evaluators inspected 600 released samples; 28 had minor issues (wrong parameter values or extra calls), about 95.3% without issues (App. A.3).
- **Models:** xLAM-1B (FC) from DeepSeek-Coder-1.3B-instruct and xLAM-7B (FC) from DeepSeek-Coder-7B-instruct-v1.5, trained with the AgentOhana/xLAM pipeline (§5.1, App. B.3). §1 describes the sizes as 1.3B and 6.7B.
- **BFCL results (Table 2):** xLAM-7B (FC) rank 6, overall 85.65; xLAM-1B (FC) rank 24, overall 74.41; GPT-3.5-Turbo-0125 (FC) rank 33, 63.88; base DeepSeek-v1.5 (Prompt) rank 45, 40.41.
- **Filtering ablation (Fig. 5):** printed deltas for xLAM-7B are −4.06 (+Fail Semantic) and −5.94 (+Fail Execution); for xLAM-1B −9.59 and −12.17. The text says the failed data from stage 3 and stage 2 was added to the training set; it does not state whether the +Fail Execution arm also contains the semantic failures, and the figure does not print bar values (§5.2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | base checkpoint | DeepSeek-Coder-1.3B-instruct; DeepSeek-Coder-7B-instruct-v1.5 | arXiv:2406.18518v1 §5.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V2-Chat, DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, Mixtral-8x7B-Inst (generators) | — | SFT (data) | sampling temperature; target samples per generator | 0.7; 40,000 | §4.2 | verified 2026-09-14 | no ablation reported |
| xlam-function-calling-60k | — | SFT (data) | released samples; source generators | approximately 60,000; Mixtral-8x22B-Inst + DeepSeek-V2-Chat | §4.2 | verified 2026-09-14 | Table 1 pass rates (stronger generators) |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | relevance-detection examples (empty call or refusal target) | 8,000 | App. B.3 | verified 2026-09-14 | no ablation reported |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | total training examples | not reported | checked §4.2, §5, App. A–B | not reported | — |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | peak LR; schedule; warmup | 5 × 10⁻⁶; cosine; 50 steps | App. B.3 | verified 2026-09-14 | no ablation reported |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | epochs; optimizer | 4; AdamW (betas, weight decay not reported) | App. B.3 | verified 2026-09-14 | no ablation reported |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | cutoff length; per-device batch; grad accumulation | 2048; 6; 2 | App. B.3 | verified 2026-09-14 | no ablation reported |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | global batch; checkpoint selection | not reported (data-parallel degree not stated) | App. B.3 | not reported | — |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | precision; hardware | bf16; 8 × A100 40GB | App. B.3 | verified 2026-09-14 | — |

## Findings relevant to generality, negative feedback, agentic training, distillation
- **Negative feedback (negative marginal value):** samples rejected by the execution or semantic checker lowered BFCL accuracy when used as positive SFT targets; the drop was larger for the 1B model (Fig. 5). Rejected samples are discarded, not used as gradient (§3.2).
- **Negative as content:** 8,000 relevance-detection targets teach an empty tool call or a short refusal when tools cannot answer or required arguments are missing (App. B.3).
- **Generality:** evaluation is BFCL only; no held-out-API split and no non-function-calling benchmarks are reported (§5). The authors report the main improvements in the parallel and multiple categories, where the base model scores low (§5.2).
- **Agentic scope:** generation is single-turn only; multi-turn is future work (§6).
- **Distillation:** data from larger open models trains 1.3B/7B students; stronger generators produce fewer format and execution failures (§4.2, Table 1).

## Connections
- [[toolllm]] — source of the ToolBench REST APIs that APIGen filters (§4.1).
- [[bfcl]] — the only evaluation benchmark used (§5.1).
- [[xlam]] — model family name and training pipeline used for xLAM-1B/7B (FC) (§5.1).
- [[apigen-mt]] — later Salesforce work on multi-turn data; APIGen covers single-turn only (§6).
- [[toolace]] — another synthetic function-calling data pipeline.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.18518 (v1, the only version; full PDF including App. A–B).
- Corrections to the previous card version:
  - "DeepSeek-Coder-V2-Instruct or GPT-4 generators" → DeepSeek-V2-Chat (236B), DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, Mixtral-8x7B-Inst (§4.2).
  - "GPT-4 semantic judge" → checker LLM not named (§3.2, App. B.2).
  - "xLAM-7B (Mistral-7B base) 88.24% BFCL-V1, #1 among <13B" → DeepSeek-Coder-7B-instruct-v1.5 base, rank 6, 85.65 (§5.1, Table 2).
  - "filter rejects ~40% of raw generations" → pass rates 34.42%–84.15% by generator (Table 1).
  - "ablation −6/−11/−18 for removing semantic/execution/format" → add-back ablation with printed deltas −4.06/−5.94 (7B) and −9.59/−12.17 (1B); no format arm (Fig. 5).
  - "start from ToolBench's 16K APIs, keep APIs with Python mock implementations" → 3,539 accessible REST APIs + 134 Python functions (§4.1).
  - "License CC-BY-NC-4.0" → CC BY 4.0 (App. A.1). "NeurIPS" → no venue on the arXiv record.
- Removed as unsupported by the source: xLAM-8x7B 88.9%; "$8K teacher API + 10K GPU-hours"; "5 sec timeout"; "<3% hallucination vs ~15% for ToolLLaMA"; "k=1–3 functions" and rare-category weighting; MinHash dedup; "each API appears 16×"; OpenAI `tool_calls` output format; "~60K sufficient to match larger unverified sets"; GPT-4 judge unit-error example; "3,673 APIs with ground-truth implementations".
- Not reported by the source: semantic-checker model, execution timeout, API/example sampling ranges, total SFT examples, global batch, held-out-API evaluation.
