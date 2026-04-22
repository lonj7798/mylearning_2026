<!-- scope: tool-calling synthesis — xLAM large-action-model family with APIGen + xLAM-2 MT pipeline
     deps: [[apigen]], [[apigen-mt]]
     see-also: [[toolace]], [[bfcl]]
-->

# xLAM: A Family of Large Action Models to Empower AI Agent Systems
- **Core Insight:** An "action model" optimized for function calling benefits from a unified data pipeline (APIGen → APIGen-MT) plus a curriculum covering single-turn, parallel, and multi-turn tool use; scaling from 7B to 70B shows consistent BFCL gains but the data pipeline is the lever, not the parameter count.
- **Guideline:** To build a function-calling specialist model, SFT on (a) APIGen 60K single-turn + (b) APIGen-MT multi-turn + (c) optional DPO on BFCL-style failure pairs; the resulting model matches GPT-4o on τ-bench at 70B scale.
- **Authors:** Jianguo Zhang, Tian Lan, Ming Zhu, Zuxin Liu, Thai Hoang, Shirley Kokane, Weiran Yao, Juntao Tan, Akshara Prabhakar, Haolin Chen, Zhiwei Liu, Yihao Feng, Tulika Awalgaonkar, Rithesh Murthy, Eric Hu, Zeyuan Chen, Ran Xu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong (Salesforce AI Research)
- **Year:** 2024 (xLAM-v1), 2025 (xLAM-2)
- **URL:** https://arxiv.org/abs/2409.03215 (xLAM-v1); https://arxiv.org/abs/2504.03601 (xLAM-2 via APIGen-MT)
- **Relevant topics:** function calling, large action models, multi-turn agents, BFCL, τ-bench

## Abstract
xLAM is Salesforce's family of function-calling-specialist "large action models". The v1 release (Sept 2024) introduced xLAM-1B-fc-r, xLAM-7B-fc-r, xLAM-8x7B, xLAM-8x22B, trained on APIGen-60k (single-turn). The 2025 xLAM-2 release (xLAM-2-1B-fc-r through xLAM-2-70B-fc-r) added multi-turn training data from APIGen-MT. xLAM-2-70B-fc-r leads open models on τ-bench and BFCL-V3.

## Key Contributions
- **xLAM family** of checkpoints at 1B / 7B / 8×7B / 8×22B / 70B scales.
- Unified chat template for function calling across scales (tool schemas in system prompt, JSON `tool_calls` in assistant turns).
- Staged training recipe: APIGen-60k SFT → APIGen-MT SFT → optional DPO.
- Open weights (CC-BY-NC-4.0) + open datasets.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Data mix for xLAM-v1
- **APIGen-60k** (single-turn function-calling, 3-layer verified). See [[apigen]].
- General-purpose instruction data (preserve chat quality): OpenOrca, WildChat subsets.
- Ratio roughly 40% function-calling / 60% general chat.

### Data mix for xLAM-2
- APIGen-60k single-turn.
- **APIGen-MT-5k** public + larger internal multi-turn split. See [[apigen-mt]].
- Optional DPO step on (preferred, rejected) tool-call pairs synthesized by sampling failure modes.

### Training recipe
- **SFT:** LR 2e-5 → 5e-6 cosine, 3 epochs, seq len 8K. Prompt masked (loss only on assistant tokens including tool calls).
- **Optional DPO:** β = 0.1; preference pairs = (correct tool call, common failure mode like hallucinated function name).
- **Bases:** Mistral-7B, Mixtral-8x7B, Llama-3.1-70B, DeepSeek-Coder-V2-8x22B.
- **Output shape:** SFT corpus ~100K (single-turn) + tens-of-thousands (multi-turn).
- **Teacher(s):** data comes from APIGen pipeline (DeepSeek-Coder-V2, GPT-4, Claude-3.5).
- **Cost / compute:** training not separately disclosed; dominated by the 70B / 8×22B SFT runs.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** training covers ~3,673 executable APIs (from APIGen) + tens of thousands more via APIGen-MT blueprints.
- **Exact verification rules (inherited):** 3-layer (APIGen) for single-turn + blueprint+state-match (APIGen-MT) for multi-turn.
- **Hallucination-rate measurement:** xLAM-7B-fc-r hallucination on BFCL-V1 irrelevance ~4%; xLAM-2-70B on τ-bench hallucinated-tool ~2%.
- **Call format:** OpenAI `tool_calls` JSON; `tool` role messages for observations.
- **Template detail:** tool schemas embedded in the system prompt; assistant alternates natural-language reasoning with `<tool_call>…</tool_call>` blocks.

## Quality / diversity evaluation
- **xLAM-7B-fc-r:** BFCL-V1 88.24% — #1 among <13B at release (Sept 2024).
- **xLAM-8x22B-fc-r:** BFCL-V1 ~89% — near GPT-4 overall.
- **xLAM-2-70B-fc-r:** τ-bench pass^1 56.2% / pass^4 39.4% — leading open model on multi-turn; BFCL-V3 multi-turn ~72%.
- Smaller variants surprisingly competitive: xLAM-2-8B beats GPT-4o on τ-bench retail.

## Risks + gotchas
- **License:** CC-BY-NC-4.0 (non-commercial). Not drop-in for product use.
- **Narrow skill:** function-calling specialist — general chat quality below general-purpose models of same size.
- **Chat-template sensitivity:** xLAM expects its exact tool-call format; prompt-engineering to other schemas degrades performance.
- **BFCL overfit risk:** community questions whether BFCL-V1 scores (91% ceiling) reflect real production reliability — pushed V2/V3 benchmarks.

## Connections
- Data pipelines: [[apigen]] + [[apigen-mt]].
- Competing family: [[toolace]] (same scale, different data pipeline).
- Evaluation: [[bfcl]].
- Hammer / NexusRaven / Granite-FC / Glaive-FC sit in same "function-calling specialist" niche.
