<!-- scope: agentic trajectory synthesis — Microsoft Orca-Agent automated trajectory pipeline
     see-also: [[agenttuning]], [[agent-flan]], [[fireact]]
-->

# AgentInstruct: Toward Generative Teaching with Agentic Flows
- **Core Insight:** Instruction data for agentic capabilities can be generated autonomously by a multi-agent "content → seed → instruction → refinement" flow — raw documents pass through stages of content extraction, seed construction, instruction synthesis, and iterative refinement, each stage orchestrated by specialized LLM agents; the resulting 25M-pair corpus lifts a Mistral-7B base model substantially across 15+ benchmarks.
- **Guideline:** For synthesizing agentic/reasoning instruction data at scale, build a **pipeline of specialized agents** rather than one monolithic prompt: one agent extracts content, another generates candidate seeds, another diversifies, another refines — each with its own prompt and tool access.
- **Authors:** Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah (Microsoft Research)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2407.03502
- **Relevant topics:** agentic SFT synthesis, Orca lineage, agentic flow, Mistral fine-tune

## Abstract
AgentInstruct is a data-generation framework built from "agentic flows" — multi-agent pipelines where specialized LLMs cooperate to produce high-quality instruction data from raw source text. Applied to a mixture of web documents, text books, code repositories, retrieval-augmented search results, and synthetic content, AgentInstruct produced a 25M instruction-response dataset (AgentInstruct-25M) used to train Orca-3 (Mistral-7B base). Orca-3 outperforms baseline Mistral-7B-Instruct by 40%+ on AGIEval, 19% on MMLU, 54% on GSM8K, and massive gains on code, reasoning, RAG, and agentic benchmarks.

## Key Contributions
- **Agentic-flow framework** — a meta-approach where data generation is itself a multi-agent pipeline.
- **25M AgentInstruct dataset** (proprietary, Microsoft-internal release; subset methodology open).
- **17 distinct skills covered:** reading comprehension, math, code, tool use, RAG, creative content, web agent, long-context, etc.
- Orca-3 (Orca-Mistral-7B) model release with strong cross-domain gains.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### General 4-stage agentic flow (per skill)
- **Stage 1 — Content Transformation:** specialized agent rewrites / extracts structured content from raw sources. Example for reading-comprehension skill: transforms a web document into a canonical "passage + candidate-questions list" structure.
- **Stage 2 — Seed Instruction Generation:** 10+ specialized "generator" agents, each with a distinct prompt for a sub-skill (e.g., literal question, inferential question, multi-hop question).
- **Stage 3 — Instruction Refinement:** a "suggester" agent proposes improvements; an "editor" agent applies them. Refinement loop iterates up to 3 times per instruction.
- **Stage 4 — Answer Generation + Validation:** teacher model (GPT-4) produces an answer; optional LLM-judge filter drops low-quality responses.

### Specific skill flows (selected)
- **Tool use:** seed from real API docs → generator agents create queries at varying tool-count complexity → refinement enforces schema correctness.
- **Reading comprehension:** a **43-generator-agent** suite (each generator targets a specific question category — literal, vocabulary, main-idea, inferential, multi-hop, numerical reasoning, etc.).
- **RAG:** content agents build passage clusters → query agents generate questions requiring evidence fusion.

### Aggregate output
- **Output shape:** 25M (instruction, response) pairs across 17 skills and 100+ sub-skills. Avg response length 300–1000 tokens.
- **Filtering:** per-skill-specific automatic checks + LLM judge for hardest cases.
- **Teacher model(s):** GPT-4 as primary generator; occasional GPT-4-Turbo for faster sub-tasks.
- **Cost:** not disclosed; estimated >$500K in GPT-4 API.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** the "agentic" label here refers to the **data-generation pipeline** being agentic, not necessarily training the student on agent-action trajectories (though a tool-use skill is included).
- **Action space (in tool-use subset):** ReAct-style (Thought / Action / Observation) with API calls.
- **Trajectory length:** depends on skill — reading-comp ~500 tokens, tool-use ~1500 tokens, long-context up to 8K+.
- **Success criterion:** per-skill LLM judge + schema validation.
- **Data scale:** 25M instruction-response pairs; specific agentic (tool-use + web-agent) subset ~1M.

## Quality / diversity evaluation
- Orca-3 (Mistral-7B base + AgentInstruct SFT) vs Mistral-7B-Instruct:
  - AGIEval: +40%.
  - MMLU: +19%.
  - GSM8K: +54%.
  - BBH: +38%.
  - MATH: +3× accuracy.
  - AlpacaEval 2.0: large wins.
- Paper's headline: 100% of benchmarks improved, 15+ showed substantial gains.

## Risks + gotchas
- **Orchestration complexity:** running 40+ specialized agents per skill requires heavy prompt engineering and infrastructure.
- **GPT-4-teacher leakage / licensing:** Microsoft-internal; not reproducible outside with open teachers.
- **Dataset not fully open:** methodology is published, full 25M dataset is not.
- **Long-tail instability:** some sub-skills (geometry with figures, code with graphs) remain weak despite the pipeline.

## Connections
- Lineage: Microsoft's Orca and Orca-2 lines on explanation-trace distillation and cautious reasoning.
- Contemporary: [[agent-flan]] (simpler recipe), [[agenttuning]] (SFT on agent-trajectory corpus).
- Self-Instruct conceptual ancestor: [[self-instruct]] — AgentInstruct is "Self-Instruct with orchestrated specialist agents".
- Related: [[glan]] (MS taxonomy-driven synthesis) uses a similar decomposition.
