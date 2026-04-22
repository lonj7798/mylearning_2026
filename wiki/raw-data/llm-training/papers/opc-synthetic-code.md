<!-- scope: OpenCoder OPC synthetic code SFT datasets (stage1/stage2) — 4.5M examples
     deps: [[oss-instruct]]
     see-also: [[wizardcoder]], [[code-evol-instruct]]
-->

# OpenCoder OPC-SFT: Open Cookbook Synthetic Code SFT Datasets
- **Core Insight:** A top-tier open code LLM needs a **two-stage SFT mix** — stage 1 broad (~4.21M filtered + extracted + large-scale diverse code instructions) to instill general coding competence, stage 2 narrow (~375K educational + evolved + test-verified) to polish — totaling ~4.5M high-quality code SFT examples.
- **Guideline:** For code-specialist fine-tuning, run a two-stage SFT: first a broad filtered mix (public code instructions + real-user code chats + synthetic diverse), then a smaller stage of verified-correct + evolved examples; include test-case validation in the stage-2 pipeline.
- **Author(s):** OpenCoder team (InfLM)
- **Year:** 2024
- **URL:** https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1 ; https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2 ; https://github.com/OpenCoder-llm/OpenCoder-llm
- **Relevant topics:** code SFT, two-stage post-training, OSS-Instruct, verified synthetic code

## Overview
OpenCoder releases its full post-training recipe including an **open 4.5M+ SFT corpus** split across two stages. Stage 1 is broad and large; stage 2 is small, verified, and polished. The mix is the clearest public template for how to structure modern code SFT.

## Stage 1 — opc-sft-stage1 (~4.21M samples)
Three primary components:
- **Filtered_infinity_instruct:** filtered from the large `infinity_instruct` corpus, keeping only code-related entries via LLM filtering.
- **Realuser_instruct:** bilingual code-related instructions extracted from real user GPT conversation histories (ShareGPT, WildChat) — grounds the distribution in actual user asks.
- **Largescale_diverse_instruct:** large-scale diverse synthetic code instructions (Self-Instruct / Evol-Instruct family).

## Stage 2 — opc-sft-stage2 (~375K samples)
Four polished components:
- **Educational_instruct:** uses an algorithmic corpus as seed; synthesizes `(instruction, code, test case)` triples; **validates each triple via a Python compiler + test execution**; only passing triples are kept.
- **Evol_instruct:** directly uses Magicoder-Evol-Instruct-110K open version (see [[evol-instruct]] / [[wizardcoder]]).
- **McEval_instruct:** multi-language eval-adjacent synthetic instructions.
- **Package_instruct:** library-usage tasks (Python packages like numpy, pandas, sklearn).

## Pipeline highlights
- **Test-case verification** in stage-2 educational_instruct is the distinguishing filter — ensures correctness of synthesized code.
- **Multi-language** coverage via McEval_instruct.
- **Real-user grounding** via ShareGPT/WildChat extraction in stage 1.
- **Companion datasets:**
  - `opc-annealing-corpus` — synthetic pretraining-adjacent data + algorithmic corpus.
  - Preference data for DPO / ORPO stage.

## Training outcome
- OpenCoder-8B-Instruct at release: strong HumanEval / MBPP / LiveCodeBench numbers; close to Qwen2.5-Coder-7B-Instruct.
- OpenCoder-1.5B-Instruct: best-in-class at its size point.

## Practitioner takeaways
- **Two-stage SFT** (broad → polished) is the current best open code recipe.
- **Real-user extraction** counters distribution bias from pure synthetic.
- **Test-case validation** is load-bearing — removes wrong-code noise that degrades smaller models.
- **Multi-language coverage** still challenging; McEval_instruct is a small but important share.

## Risks + gotchas
- **Language skew toward Python** — non-Python benchmarks lag.
- **Test harness scope** — only simple executable tests; algorithmic correctness beyond tests unguaranteed.
- **ShareGPT / WildChat extraction** inherits source-license nuances.

## Connections
- Heavy use of [[oss-instruct]] (MagicoderS) philosophy in stage 1.
- Evol subset overlaps with [[wizardcoder]] line.
- Practical reference companion to [[llama-3-synthetic-pipeline]]'s code subsection.
- Template for future open code-model releases (Qwen-Coder open siblings, etc.).
