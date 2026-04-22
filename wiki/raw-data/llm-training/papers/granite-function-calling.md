<!-- scope: tool-calling synthesis — IBM Granite function-calling data mix for enterprise LLMs
     deps: [[apigen]]
     see-also: [[xlam]], [[toolace]], [[hammer]]
-->

# Granite Function-Calling: Granite-20B-FunctionCalling
- **Core Insight:** Enterprise-grade function-calling models benefit from a multi-task, multi-dataset training mix covering seven function-calling sub-skills (nested, parallel, multi-turn, slot-filling, relevance, sequencing, tool-selection); IBM demonstrates that combining public corpora (APIGen, ToolLLM, Glaive-FC) with enterprise synthetic data yields consistent cross-benchmark gains.
- **Guideline:** Don't train on one tool-data source; blend 4–5 complementary corpora (APIGen for verified single-turn, ToolLLM for real-API trajectories, Glaive/Hermes for instruction diversity, private enterprise schemas for domain) and sample to balance the 7 sub-skills.
- **Authors:** Ibrahim Abdelaziz, Mayank Agarwal, Kinjal Basu, Maxwell Crouse, Pavan Kapanipathi, Soham Dan, Yara Rizk, et al. (IBM Research)
- **Year:** 2024 (arXiv 2407) — ongoing Granite updates through 2025
- **URL:** https://arxiv.org/abs/2407.00121 ; https://huggingface.co/ibm-granite
- **Relevant topics:** enterprise function calling, multi-task training, Granite, data-mix ablation

## Abstract
Granite-20B-FunctionCalling is IBM's 20B-parameter function-calling model (based on Granite-20B code base) trained on a multi-task, multi-source mix covering seven function-calling capabilities. The paper ablates each data source's contribution and shows the blend strictly dominates any single-source training. Granite-20B-FC achieves BFCL-V1 scores competitive with GPT-4 and xLAM-8x7B at release.

## Key Contributions
- **Seven-capability taxonomy** of function calling: Nested, Parallel, Multiple Functions, Multi-turn, Relevance Detection, Sequencing, Slot Filling.
- **Multi-source data mix** ablation — shows per-source contribution to each capability.
- Granite-20B-FC + Granite-3B-FC model releases (Apache-2.0).
- Downstream use in IBM watsonx AI agent platform.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Data sources in the mix (approximate proportions):**
  - **APIGen / xLAM-FC-60k** (~25%): 3-layer verified single-turn calls.
  - **ToolLLM / ToolBench** (~20%): real-API trajectories.
  - **Glaive Function Calling V2** (~15%): instruction-style synthetic function calls.
  - **Nexus / NexusRaven data** (~10%): nested-call examples.
  - **IBM in-house enterprise synthetic** (~20%): schemas from real customer API catalogs, GPT-4-generated instructions + calls.
  - **General instruction data** (~10%): OpenHermes / Dolphin for non-FC conversational quality preservation.
- **Balancing:** per-capability resampling ensures each of the 7 capabilities appears ≥ 10% of the mix.
- **Filtering:** source-specific — APIGen provides already-verified samples; Glaive/OpenHermes are filtered by length/format; enterprise samples pass an LLM-judge.
- **Output shape:** ~200K examples total after filtering and resampling.
- **Teacher(s):** upstream sources use GPT-4, DeepSeek-Coder-V2, etc.; IBM's in-house pipeline uses GPT-4.
- **Cost:** not disclosed.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** thousands (inherits APIGen 3.6K + ToolLLM 16K + Glaive's custom + IBM private).
- **Exact verification rules:** inherited per-source.
- **Hallucination-rate measurement:** Granite-20B-FC relevance-detection 85% on BFCL-V2 Live.
- **Call format:** OpenAI-compatible `tool_calls` JSON; specifically tuned for the watsonx chat template.
- **Capability ablation:** removing the ToolLLM slice → multi-turn drops 12 points; removing Nexus nested → nested track drops 18 points; removing relevance-data → relevance drops 10 points.

## Quality / diversity evaluation
- Granite-20B-FC: BFCL-V1 ~82%, overall competitive with xLAM-7B.
- Granite-3B-FC: ~75% BFCL-V1 — strong for its size.
- Enterprise-focus: evaluated internally on IBM customer API catalogs with > 90% intent-classification accuracy.
- Public eval table in paper shows per-capability breakdown.

## Risks + gotchas
- **Data-mix discovery effort:** getting ratios right required many training runs; a naive equal-mix gives 5–10 points less than the tuned mix.
- **Enterprise data is private:** the paper's full mix is not reproducible publicly.
- **Base model choice (Granite code-LLM) gives code bias** — model is strong on code-style tools, weaker on conversational tool use.

## Connections
- Components: [[apigen]], [[toolllm]], [[nexusraven]], [[glaive-function-calling]].
- Competing enterprise FC: xLAM ([[xlam]]), Hammer ([[hammer]]), ToolACE ([[toolace]]).
- Evaluation: [[bfcl]].
