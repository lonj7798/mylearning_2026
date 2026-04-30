<!-- scope: long-context synthesis — retrieval-augmented instruction generation bridging RAG + long-context
     deps: [[longalign]]
     see-also: [[long-context-data-engineering]], [[ruler]]
-->

# RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions
- **Core Insight:** Real RAG quality breaks down on multi-passage reasoning because the model never saw diverse RAG-style instructions during SFT; synthesizing 40K RAG instructions that span five task paradigms (simple QA, multi-hop, counterfactual, conflicting, irrelevant passages) and mixing them into SFT yields dramatic improvements on both short-RAG (single passage) and long-context RAG (20+ passages).
- **Guideline:** For long-context RAG models, synthesize RAG SFT data with explicit coverage of (a) relevant-only, (b) irrelevant-only, (c) conflicting, (d) counterfactual, (e) multi-hop passage sets; models trained without conflicting/irrelevant examples fabricate answers when retrieval is poor.
- **Authors:** Wanlong Liu, Junying Chen, Ke Ji, Li Zhou, Wenyu Chen, Benyou Wang (SRIBD + CUHK-SZ + UESTC)
- **Year:** 2024 (Dec arXiv) / 2025
- **URL:** https://arxiv.org/abs/2501.00353 (latest), https://github.com/FreedomIntelligence/RAG-Instruct
- **Relevant topics:** RAG, long-context, multi-passage QA, instruction synthesis

## Abstract
RAG-Instruct generates 40K diverse retrieval-augmented instructions covering five distinct task paradigms — standard relevance-based QA, unbiased QA with irrelevant passages, conflicting-information QA, counterfactual QA, and multi-hop synthesis. Fine-tuning Llama-3-8B-Instruct on RAG-Instruct yields substantial gains on RAG benchmarks (+13 on the authors' RAG-eval suite) and generalizes to unseen domains. The dataset is designed with long-context RAG in mind — passage sets scale from 1 to 20+.

## Key Contributions
- **RAG-Instruct-40K dataset** — 40K RAG-style instructions across five paradigms.
- **Five-paradigm taxonomy:**
  1. **Knowledge-based QA** (relevant passages only).
  2. **Irrelevant context QA** (passages don't contain answer).
  3. **Conflicting-info QA** (passages disagree).
  4. **Counterfactual QA** (passages contradict world knowledge).
  5. **Multi-hop QA** (multiple passages needed).
- Demonstration that models trained without irrelevant/conflicting examples fabricate at rates up to 30%.
- Open release of data and Llama-3-8B-RAG-Instruct model.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** existing QA corpora (NaturalQuestions, MS-MARCO, HotpotQA) + Wikipedia paragraph pool.
- **Step 1 — Passage sampling:**
  - Relevant: gold passage(s) from QA corpus.
  - Irrelevant: random Wikipedia passages not containing the answer.
  - Conflicting: GPT-4 rewrites a gold passage with wrong answer; mixed with original.
  - Counterfactual: GPT-4 generates a passage contradicting a real-world fact.
  - Multi-hop: two passages each containing a partial fact.
- **Step 2 — Instruction generation:** GPT-4 prompted with passage set and task paradigm, generates a natural user question.
- **Step 3 — Answer generation:** GPT-4 answers based on passage set. Crucially: for irrelevant paradigm, gold answer must say "no sufficient information"; for conflicting, gold must acknowledge the conflict.
- **Filtering:** LLM-judge scores answer quality; schema checks enforce paradigm-specific behavior (e.g., refusal for irrelevant).
- **Output shape:** 40K (passage-set, question, answer) tuples; passage count 1–20; total context 500–20K tokens.
- **Teacher model:** GPT-4.
- **Cost:** ~$15K GPT-4 API.

## Modality-specific technical details (REQUIRED — long-context / RAG)
- **Token-range:** 500 → 20K tokens (modest long-context; larger than standard RAG).
- **Needle-retrieval difficulty:** multi-hop paradigm demands cross-passage reasoning; conflicting paradigm demands controversy-detection.
- **Document-type mix:** Wikipedia + QA-corpus gold passages; homogeneous by design.
- **Packing strategy:** N/A for evaluation data; during SFT, standard packing.
- **Position-encoding adaptation:** inherits base model's long-context capability.
- **Per-paradigm proportion:** ~25/15/15/15/30 across the five paradigms.

## Quality / diversity evaluation
- Llama-3-8B-RAG-Instruct: +13 points average on authors' RAG-eval suite vs baseline Llama-3-8B-Instruct.
- Hallucination rate under irrelevant passages cut from 31% → 9%.
- Generalization to domain-unseen RAG tasks (medical, legal): +10 points.
- On synthetic NIAH: no harm (retrieval preserved).

## Risks + gotchas
- **GPT-4 teacher bias:** conflicting/counterfactual judgments inherit GPT-4's world model.
- **Five-paradigm taxonomy is not exhaustive** — real RAG scenarios include time-sensitive info, multilingual, multi-modal.
- **Passage-set size capped at 20** — very-long-context RAG (100+ passages) underrepresented.

## Connections
- Long-context cousin: [[longalign]] (long-instruction SFT), [[long-context-data-engineering]].
- Eval counterpart: [[ruler]] (synthetic retrieval), [[babilong]] (reasoning over long).
- Conflicting/counterfactual paradigm originates in hallucination-mitigation literature (Mallen 2023, etc.).
