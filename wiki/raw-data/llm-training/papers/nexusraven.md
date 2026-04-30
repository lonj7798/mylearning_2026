<!-- scope: tool-calling synthesis — instruction-style function-call fine-tune with curriculum
     deps: [[toolllm]]
     see-also: [[gorilla]], [[apigen]], [[bfcl]]
-->

# NexusRaven: Zero-shot Function Calling for Cybersecurity and Beyond
- **Core Insight:** Fine-tuning on a curriculum of progressively harder **instruction-style function calls** (simple → nested → composed) produces a sub-13B model that rivals GPT-4 on function-calling zero-shot, without any online API access at training time — all data is instruction-only synthetic.
- **Guideline:** For domain-specialist function-calling models (e.g., security tools), build a curriculum of single-call → parallel-call → nested-call examples, all with inline documentation, and fine-tune on instruction-style prompts; you do not need a live API substrate if schemas are well-specified.
- **Authors:** Nexusflow team (Jiawei Liu, Fanjia Yan, Banghua Zhu, et al.)
- **Year:** 2023 (v1) → NexusRaven-V2 2024
- **URL:** https://huggingface.co/Nexusflow/NexusRaven-V2-13B ; https://nexusflow.ai/blogs/ravenv2
- **Relevant topics:** function calling, cybersecurity, CodeLlama fine-tune, zero-shot tool use

## Abstract
NexusRaven is a CodeLlama-based fine-tune targeted at zero-shot function calling. V1 (Sept 2023) focused on cybersecurity APIs. V2 (Dec 2023) expanded to general software function calling with a larger curriculum. NexusRaven-V2-13B scored competitively with GPT-4 on NexusRaven's own Nexus benchmark and on related eval suites; it introduced the "nested function call" training category — calls whose arguments are themselves function calls — a pattern previously absent from tool-data corpora.

## Key Contributions
- **Nested-call training examples** — first public corpus with `f(g(x))` patterns.
- **Curriculum learning** for function calls: simple → parallel → nested.
- **NexusRaven-V2-13B** open-weights model; zero-shot composition on unseen APIs.
- **Nexus benchmark** for function-calling eval (later partially folded into BFCL).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** function signatures + docstrings curated from Python stdlib + popular libraries + cybersecurity APIs (VirusTotal, CVE, etc.).
- **Step 1 — Schema normalization:** each function is described in a unified format with name, description, parameters (name, type, description, required).
- **Step 2 — Simple-call generation:** for each function, GPT-4 generates 5–10 user instructions that would call it, along with the correct call.
- **Step 3 — Parallel-call generation:** combinations of 2–4 independent calls in the same turn.
- **Step 4 — Nested-call generation:** instructions requiring `func_A(func_B(x))` composition. Teacher model (GPT-4) is prompted specifically for such patterns with few-shot examples.
- **Step 5 — Filtering:** AST-style syntactic check on each generated call; manual review of a fraction.
- **Output shape:** ~100K training examples, single-turn. Mix ~60% simple, 20% parallel, 20% nested.
- **Teacher model:** GPT-4.
- **Cost:** ~$10K–15K GPT-4 API.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** ~1,200 unique functions covering software APIs + CVEs + general utilities.
- **Exact verification rules:** AST parse of generated call; parameter types must match schema; nested calls must reference declared functions.
- **Hallucination-rate measurement:** on Nexus-bench, NexusRaven-V2-13B hallucinated-arg rate ~5% vs GPT-4 ~3%.
- **Call format:** Python function-call syntax, e.g., `submit_report(ip=get_ip_address("attacker.com"))`.
- **Nested-call example:**
  ```python
  save_file(name="report.txt", content=summarize(translate("text", to="en")))
  ```
  — V1 and pre-NexusRaven tool-data corpora rarely contained such composed patterns; the V2 data addition is what unlocked zero-shot composition.

## Quality / diversity evaluation
- NexusRaven-V2-13B: Nexus-bench ~76% — exceeds GPT-3.5 by ~20 points.
- BFCL-V1 (community-reported): ~86%.
- Cybersecurity zero-shot: beats GPT-4 on 2023 VirusTotal/CVE task suite.
- Ablation: removing nested examples → −15 points on nested-eval track.

## Risks + gotchas
- **Python-syntax tooling assumption:** downstream integrations must support function-call parsing from Python source (not the standard OpenAI JSON tool-calls).
- **No multi-turn training:** V2 is single-turn only; multi-turn tool use requires additional data (xLAM-2 / APIGen-MT territory).
- **Superseded in general FC by xLAM / ToolACE / Hammer** at equivalent size; still strong on nested-call patterns.

## Connections
- Part of the "instruction-style FC" lineage alongside [[gorilla]].
- Nested-call idea recurs in [[toolace]] (complexity controller includes nested).
- Evaluation partly absorbed into [[bfcl]].
