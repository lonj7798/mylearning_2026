<!-- scope: tool-calling synthesis — self-evolving API pool + multi-agent dialog + dual-layer verification
     deps: [[toolllm]], [[apigen]]
     see-also: [[apigen-mt]], [[xlam]], [[bfcl]]
-->

# ToolACE: Winning the Points of LLM Function Calling
- **Core Insight:** State-of-the-art function-calling data comes from three co-designed components — a **self-evolving API pool** (tools beget more specialized tools), a **multi-agent interactive dialog generator** with a complexity controller, and a **dual-layer (rule-based + model-based) verifier** — and the combination pushes an 8B model past many proprietary models on BFCL.
- **Guideline:** For broad + deep function-calling SFT, grow APIs via self-evolution (a generator LLM mutates existing APIs into specialized variants), generate dialogs with multiple LLM roles (user, assistant, tool) under an explicit complexity-evaluator, and verify each sample with both programmatic rules and an LLM judge.
- **Authors:** Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong Wang, Yuxian Wang, Wu Ning, Yutai Hou, Bin Wang, Chuhan Wu, Xinzhi Wang, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming Tang, Defu Lian, Qun Liu, Enhong Chen (Huawei Noah's Ark Lab + collaborators)
- **Year:** 2024 (arXiv Sept) — ICLR 2025
- **URL:** https://arxiv.org/abs/2409.00920
- **Relevant topics:** function calling, self-evolution, multi-agent dialog synthesis, dual verification, BFCL

## Abstract
ToolACE is an agentic pipeline for high-quality function-calling data, built around three modules: (1) **Tool Self-Evolution Synthesis (TSS)** which grows an API pool of 26,507 APIs starting from seed real-world APIs, (2) **Multi-Agent Interactive Dialog (MAI)** where multiple LLM roles (user, assistant, tool simulator) produce trajectories under a complexity evaluator that aims for a target distribution of difficulty, and (3) a **Dual-Layer Verification System** combining rule-based checks and an LLM judge. ToolACE-8B (Llama-3.1-8B-Instruct base) achieves **91.41% BFCL-V1 overall**, outperforming all open-source <13B models and matching GPT-4 / Claude-3.

## Key Contributions
- **TSS algorithm** — systematically evolves APIs for broader coverage (26,507 APIs from a smaller seed).
- **MAI dialog generator** — multi-agent role-play with a complexity-controller producing simple / nested / multi-API / parallel / missing-info dialogs.
- **Dual verification** — rule-based (schema + execution) + model-based (LLM judge).
- **ToolACE-8B** reaches 91.41% BFCL-V1, near-frontier at 8B scale.
- Open dataset and model releases.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Module 1 — Tool Self-Evolution Synthesis (TSS)
- **Seed APIs:** ~3K real-world APIs from public directories.
- **Evolution operators:** an LLM mutates seed APIs into new APIs by:
  - **Parameter extension:** add new input/output parameters.
  - **Domain transfer:** port an API's pattern to a new domain (e.g., weather→stock).
  - **Functionality refinement:** specialize a generic API into narrower sub-APIs.
- **Filtering:** generated API schemas must parse; names must be unique; LLM-judge scores novelty and utility.
- **Output:** 26,507 APIs covering 390 domains.

### Module 2 — Multi-Agent Interactive Dialog (MAI)
- **Roles:** user-LLM, assistant-LLM, tool-simulator-LLM; each role has a distinct prompt.
- **Complexity evaluator:** classifies target dialog into one of 5 difficulty levels (simple / multi-call / parallel / nested / info-incomplete) and conditions generation to hit a balanced distribution.
- **Dialog generation:** role-play proceeds turn-by-turn; tool-simulator generates realistic (but not always executed) responses.
- **Output:** ~11K dialogs spanning the 5 complexity classes.

### Module 3 — Dual-Layer Verification
- **Rule-based checks:** JSON schema validation; required parameters present; parameter types match; enum values within range; deterministic executable sanity check on a subset where simulators have Python implementations.
- **Model-based checks:** LLM judge (GPT-4) evaluates (a) whether the user query is ambiguous, (b) whether the assistant's call satisfies the query, (c) whether the simulated tool response is consistent with the schema.
- **Acceptance:** must pass both layers.
- **Output shape:** final 11,300 dialogs after filtering (~40% rejection rate). Single-turn or 2–4 turn dialogs (mostly single-turn function-calling tasks).
- **Teacher model(s):** GPT-4 variants for role-play and judging.
- **Cost / compute:** not precisely disclosed; ~$30K estimated in API.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** 26,507 APIs (390 domains) — largest public function-calling pool as of 2024.
- **Exact verification rules:**
  - **Rule-based:** schema + param presence + type check; execution check only where Python mock is defined.
  - **Model-based:** GPT-4 judge with 3-way verdict (query-clarity / call-correctness / response-consistency); requires all three "pass".
- **Hallucination-rate measurement:** on BFCL relevance-detection sub-task (irrelevant tool list) ToolACE-8B achieves 82.1% — strong evidence of low hallucination.
- **Complexity distribution of dialogs:**
  - Simple single-call: ~30%.
  - Multiple (choose correct from list): ~25%.
  - Parallel (multiple calls in same turn): ~20%.
  - Nested / multi-turn: ~15%.
  - Info-incomplete (requires user clarification): ~10%.
- **Call format:** OpenAI-compatible tool-call JSON.

## Quality / diversity evaluation
- **ToolACE-8B (Llama-3.1-8B base):** BFCL-V1 overall **91.41%**, Live 83%, Non-live 91%.
- Beats xLAM-7B (88.24%) at comparable size.
- Ablation: remove TSS → –4.3% BFCL; remove MAI complexity controller → –3.1%; remove model-judge → –5.2%; remove rule-checks → –2.8%.
- Generalization: holds up well on APIs unseen in training when retriever is used.

## Risks + gotchas
- **Tool-simulator hallucination:** without full executable endpoints, simulated tool responses can be unrealistic; the LLM judge partially catches this but not reliably.
- **API "evolution" can drift:** mutated APIs sometimes reference non-existent conventions.
- **Proprietary teacher dependency:** GPT-4 for generation AND judging — circular quality ceiling.
- **Mostly single-turn:** multi-turn is underweighted vs [[apigen-mt]].

## Connections
- Contemporary: [[apigen]] (3-layer verification; smaller but fully executable).
- Multi-turn sibling: [[apigen-mt]] (blueprint-then-rollout with real execution).
- Retrieval cousin: [[toolllm]] (real live APIs).
- Evaluation: [[bfcl]].
- Lineage of dual/triple verification: [[apigen]] (3-layer) → [[toolace]] (dual) → [[apigen-mt]] (blueprint + state-match).
