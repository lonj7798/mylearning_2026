<!-- scope: tool-calling synthesis — evaluation-first tool-use benchmark + training set
     deps: [[toolllm]]
     see-also: [[gorilla]], [[bfcl]]
-->

# API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs
- **Core Insight:** Tool-use progress requires an evaluation-first benchmark separating the three skills — **call** (knows when/how to invoke), **retrieve** (picks the right API from many), **plan** (sequences multiple calls); API-Bank operationalizes each with its own test split and provides a matched training corpus (Lynx).
- **Guideline:** Score tool-use models on ability-aware axes, not aggregate accuracy: measure API-call correctness, API-retrieval recall, and multi-API planning success separately; a model strong on "call" may be weak on "retrieve" and vice versa.
- **Authors:** Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, Yongbin Li (Alibaba DAMO + Beihang)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2304.08244
- **Relevant topics:** tool-use evaluation, API benchmark, Lynx dataset, function calling

## Abstract
API-Bank introduces a benchmark for tool-augmented LLMs covering 73 commonly-used APIs and 314 dialog-format test cases, split into three ability-aware tracks — Call, Retrieve+Call, Plan+Retrieve+Call. Accompanying the benchmark is **Lynx**, an instruction-tuning training corpus of 1,888 dialog examples across 1,000+ APIs synthesized by a multi-agent "GPT-4 as user + GPT-4 as assistant + GPT-4 as API simulator" role-play pipeline. Lynx-trained Alpaca-7B lifts API-Bank Call accuracy from 18% → 66%.

## Key Contributions
- First tool-use benchmark with **three ability axes** (Call / Retrieve / Plan).
- **Lynx dataset** — 1,888 dialogs, 1,008 APIs, multi-agent role-play.
- Baseline evaluations of GPT-3.5, GPT-4, Alpaca, Claude at release.
- Documented multi-agent "U-A-T" (User-Assistant-Tool) synthesis protocol now standard in tool-data pipelines.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** 1,008 APIs curated from real-world docs + synthetic "plausible" APIs.
- **Multi-agent role-play pipeline:**
  1. **User-GPT:** receives an API description and generates a natural user query.
  2. **Assistant-GPT:** given the user query plus a list of candidate API docs, produces the reasoning + call.
  3. **Tool-GPT:** given the call and API description, generates a realistic response.
  4. Loop for multi-turn dialogs until task is complete.
- **Filtering:**
  - Role-played dialog must end with a terminating assistant turn (final answer).
  - GPT-4 judge rates dialog for (a) call correctness, (b) response realism, (c) instruction-following.
  - Accept if all three score ≥ 4/5.
- **Output shape:** 1,888 dialogs, avg 3–5 turns, avg 2 API calls per dialog.
- **Teacher:** GPT-4.
- **Cost:** ~$3K GPT-4 API.

## Evaluation methodology (REQUIRED — tool-calling)

### Three ability tracks
- **Call:** user gives an instruction; a specific API is already named; model must fill args correctly and emit a call.
- **Retrieve+Call:** user gives instruction; model must pick the correct API from a pool, then call it.
- **Plan+Retrieve+Call:** user gives multi-step instruction; model must pick and sequence multiple APIs.

### Test set
- 314 dialog test cases spanning 73 core APIs.
- Metrics: exact-match on API name, arg exact-match, final-answer ROUGE/accuracy.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** 73 (test) + 1,008 (train). Much smaller than ToolBench (16K) or ToolACE (26K).
- **Exact verification rules:** teacher-model-based GPT-4 judge only (Lynx training); for evaluation, symbolic match on name + args.
- **Hallucination-rate measurement:** not a distinct sub-score; bundled into overall Call accuracy.
- **Call format:** natural dialog with inline tool calls — pre-dates OpenAI `tool_calls` JSON standard.

## Quality / diversity evaluation
- Baseline Alpaca-7B on API-Bank Call: 18% — Lynx-trained Alpaca-7B: 66%.
- GPT-4 (with tools): ~88% Call.
- Plan-track exposes weakest current capability — even GPT-4 scores only ~60% on multi-API sequencing.

## Risks + gotchas
- **Small scale:** eclipsed by ToolBench, APIGen, ToolACE on training data scale.
- **Benchmark has been partially memorized by downstream models** that train on Lynx.
- **No executable APIs:** all tool responses are GPT-4-simulated, so execution bugs are missed.
- **Superseded by [[bfcl]]** for 2024+ evaluations.

## Connections
- Contemporary: [[toolllm]] (16K real APIs + DFS-DT).
- Evaluation ancestor of [[bfcl]] (same ability-axis idea, larger + live).
- Multi-agent synthesis protocol reused in [[toolace]] MAI module and [[apigen-mt]] role-play phase.
