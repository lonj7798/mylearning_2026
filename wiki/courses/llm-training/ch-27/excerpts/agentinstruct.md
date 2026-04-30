---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/agentinstruct.md
source_url: https://arxiv.org/abs/2407.03502
created_at: "2026-04-23"
---

# Excerpt: AgentInstruct — 43 generators and the agentic-flow template

**Source library:** `wiki/raw-data/llm-training/papers/agentinstruct.md`
**Paper:** Mitra et al. 2024 (Microsoft Research), "AgentInstruct: Toward Generative Teaching with Agentic Flows."

---

## Why this source anchors ch-27 §1.1

Ch-27 §1.1's claim — "the generator count is a diversity knob, not a budget waste" — rests on one specific number from this paper: **43**. That's how many distinct generator agents the reading-comprehension skill pipeline uses. Not 5, not 10 — 43, each with its own prompt targeting a specific sub-skill (literal question, vocabulary question, main-idea question, inferential question, multi-hop question, numerical reasoning, …).

## The four generic flow stages

From the source (lines 25-28):

> - **Stage 1 — Content Transformation:** specialized agent rewrites / extracts structured content from raw sources.
> - **Stage 2 — Seed Instruction Generation:** 10+ specialized "generator" agents, each with a distinct prompt for a sub-skill.
> - **Stage 3 — Instruction Refinement:** a "suggester" agent proposes improvements; an "editor" agent applies them. Refinement loop iterates up to 3 times per instruction.
> - **Stage 4 — Answer Generation + Validation:** teacher model (GPT-4) produces an answer; optional LLM-judge filter drops low-quality responses.

Notice the sequencing. Content Transformation is **upstream**: one agent, canonical output schema. Seed Generation is **parallel fan-out**: many agents, divergent outputs. Refinement is a **serial loop**: suggester → editor → check → repeat. Answer + Validation is **gate**: binary keep/drop.

This is the classical pipeline pattern (Extract → Transform → Load) re-applied to synthesis. ETL people will recognize it immediately; ML people often won't, which is why naïve "prompt GPT-4 more, filter with a judge" pipelines plateau at ~3× worse quality than AgentInstruct's numbers.

## Why 43 generators beat 43 samples of one prompt

The temptation is to use one broad generator prompt ("generate a reading-comprehension question") and sample it 43 times with temperature 1.0. This fails for the same reason broad-taxonomy instruction-following datasets fail: **temperature diversity is not structural diversity**. Sampling "generate a question" 43× will over-represent literal questions (they're easiest to form) and under-represent multi-hop and numerical-reasoning questions (they require substrate the prompt doesn't constrain the model to seek out).

A 43-prompt pipeline with one sample per prompt produces a distribution that matches the sub-skill taxonomy by construction. This is the structural diversity the ch-27 §1.1 passage is arguing for.

## Skill-specific flow variants

From the source (lines 31-33):

> - **Tool use:** seed from real API docs → generator agents create queries at varying tool-count complexity → refinement enforces schema correctness.
> - **Reading comprehension:** a **43-generator-agent** suite.
> - **RAG:** content agents build passage clusters → query agents generate questions requiring evidence fusion.

The tool-use flow is the one most relevant to ch-27's agent-trajectory framing. "Varying tool-count complexity" means: some generators produce single-tool queries, others produce 2-tool chains, others produce composed 3+ tool sequences. Refinement validates schema correctness — the JSON-schema check is a hard filter, not a soft preference.

## Cost and scale

From the source (line 39):

> - **Cost:** not disclosed; estimated >$500K in GPT-4 API.

Half a million dollars of GPT-4 to produce 25M pairs. The per-pair cost is on the order of 2 cents — reasonable given GPT-4's per-token pricing and the multi-stage flow. The pipeline-orchestration cost (engineering time to maintain 17 skills × ~15 agents each = 250+ distinct prompts) is separately large and under-reported.

## What to take from AgentInstruct for ch-27

1. **Stages, not prompts.** Content → Seed → Refinement → Validation is a template. Don't skip any stage; each does something the others can't.
2. **Parallel generators are the diversity engine.** If you're tempted to use one prompt with high temperature, replace it with 10+ prompts at temperature 0.3.
3. **Refinement loops are worth the 1.5× cost.** Three suggester/editor passes adds ~3 points on hard sub-skills.
4. **Validation is binary.** LLM-judge gives a score; your pipeline must have a threshold that drops examples below it. No "soft" filtering.

## Connections

- [[ch-27]] §1.1, §1.2, §5 — the design-space tour and action-space table.
- [[excerpts/lumos]] — complementary structural-decomposition story (Plan/Ground/Execute).
- [[excerpts/agent-flan]] — the hallucination-negatives counterpart; AgentInstruct does not add negatives explicitly.
- [[excerpts/kimi-k2-agentic-data]] — the scale-to-pretraining cousin; AgentInstruct's flow template recurs in K2's sub-agent orchestration.
