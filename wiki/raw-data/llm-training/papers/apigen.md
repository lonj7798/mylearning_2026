<!-- scope: tool-calling synthesis — 3-layer verified synthetic function-calling data
     deps: [[toolllm]]
     see-also: [[apigen-mt]], [[toolace]], [[xlam]], [[bfcl]]
-->

# APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
- **Core Insight:** Synthetic function-calling data can be trusted for SFT iff every sample passes a **3-layer verification**: (1) format check against the function schema, (2) executable check that the call runs, and (3) semantic check that an LLM-judge agrees the call satisfies the query — this eliminates the noisy-trajectory problem plaguing ToolBench.
- **Guideline:** For high-quality function-calling SFT, implement a 3-layer verifier gating every accepted sample: JSON-schema format → sandbox execution → LLM-as-judge semantic match; at this filter strictness, ~60K samples are sufficient to match larger un-verified sets.
- **Authors:** Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh Murthy, Liangwei Yang, Silvio Savarese, Juan Carlos Niebles, Huan Wang, Shelby Heinecke, Caiming Xiong (Salesforce AI Research)
- **Year:** 2024 (NeurIPS)
- **URL:** https://arxiv.org/abs/2406.18518
- **Relevant topics:** function calling, synthetic data, verification, tool use

## Abstract
APIGen is an automated pipeline that generates verifiable function-calling data by sampling (query, function-list) pairs and producing candidate tool-call sequences that must pass three independent checks before acceptance: format validation, execution validation, and semantic validation. The pipeline produced the **xLAM-function-calling-60k** dataset — 60,000 rigorously filtered examples across 3,673 executable APIs — on which Salesforce trained the **xLAM** family of "function-calling LLMs" that ranked first on BFCL among models < 13B at release.

## Key Contributions
- **3-layer verification pipeline** — format → execution → semantic; each sample traverses all three.
- **xLAM-function-calling-60k** dataset, public (CC-BY-NC-4.0).
- **3,673 executable APIs** curated with ground-truth implementations (so execution is possible).
- Strong downstream: xLAM-7B ranked #1 on BFCL among <13B models at release.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Step 1 — API curation:** start from ToolBench's 16K APIs but keep only the 3,673 APIs with executable reference implementations (Python mock or real endpoints under Salesforce control).
- **Step 2 — Seed sampling:** for each generation, sample (k=1–3) functions from the 3,673 pool. Diversity sampler weights rare API categories higher.
- **Step 3 — Query + solution generation:** prompt DeepSeek-Coder-V2-Instruct or GPT-4 (authors ablate both) with the sampled functions and ask for:
  - A natural-language user query.
  - The gold function-call sequence as structured JSON.
- **Step 4 — 3-layer verification:**
  - **Format check:** JSON must parse, fields must match function schema, types enforced.
  - **Execution check:** run the call(s) against the reference implementations; must not raise.
  - **Semantic check:** LLM-as-judge (GPT-4) is shown the query + the call + the execution result, and must answer "yes" to "does the call correctly fulfill the query?"
- **Step 5 — Dedup:** MinHash on (query, call) pairs.
- **Output shape:** 60,000 samples covering four data types:
  - Simple (1 call, 1 function).
  - Multiple (1 call, multiple candidate functions — correct one must be chosen).
  - Parallel (≥2 calls in same turn to same function).
  - Parallel-multiple (≥2 calls across multiple functions).
- **Teacher model:** DeepSeek-Coder-V2-Instruct (primary) and GPT-4 (comparison).
- **Cost / compute:** ~$8K in teacher API + ~10K GPU-hours for execution sandbox.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** 3,673 executable APIs (21 categories).
- **Exact verification rules:**
  - **Format:** valid JSON; required params present; types match schema (int / str / bool / enum / list).
  - **Execution:** Python sandbox with 5 sec timeout; call must return without exception.
  - **Semantic:** GPT-4 judge prompt requires "Yes" / "No" verdict with reasoning; only "Yes" accepted.
- **Hallucination-rate measurement:** the 3-layer filter rejects ~40% of raw generations; post-filter, hallucination rate on BFCL-V1 is <3% for xLAM-7B (vs ~15% for ToolLLaMA).
- **Call format:** OpenAI-compatible `tool_calls` JSON.

## Quality / diversity evaluation
- xLAM-7B (Mistral-7B base fine-tuned on APIGen-60k): **88.24% BFCL-V1 overall**, #1 among <13B.
- xLAM-8x7B (Mixtral base): **88.9% BFCL-V1**, close to GPT-4.
- Ablation: removing semantic check → –6% BFCL-V1; removing execution → –11%; removing format → –18%. All three layers are load-bearing.
- Dataset diversity: each of the 3,673 APIs appears on average 16× with different argument combinations.

## Risks + gotchas
- **Executable-API requirement** limits scale — the pipeline is bottlenecked on having reference implementations.
- **LLM-judge blind spots:** GPT-4 judge occasionally accepts semantically-close-but-wrong calls (e.g. wrong unit passed to a conversion function).
- **No multi-turn:** APIGen generates single-turn function calls only. Addressed in [[apigen-mt]].
- **License:** CC-BY-NC-4.0 — non-commercial.

## Connections
- Direct successor: [[apigen-mt]] (multi-turn extension, 2025).
- Contrasts [[toolllm]] (no execution verification; live APIs).
- Downstream consumer: [[xlam]] (Salesforce's model family).
- Similar 2-layer verification sibling: [[toolace]] (self-evolution + dual verification).
- Evaluation: [[bfcl]].
