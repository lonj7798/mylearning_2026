<!-- scope: agentic trajectory synthesis — multi-method ReAct fine-tuning (CoT, ReAct, Reflexion)
     deps: [[agenttuning]]
     see-also: [[lumos]], [[agent-flan]]
-->

# FireAct: Toward Language Agent Fine-tuning
- **Core Insight:** Fine-tuning on a **mix** of agent prompting-method traces (CoT + ReAct + Reflexion) yields a stronger general agent than fine-tuning on any single method alone; method diversity in training data covers a wider distribution of reasoning styles the agent encounters at test time.
- **Guideline:** For agent SFT, collect trajectories via multiple distinct prompting methods (not just ReAct); the mix regularizes and makes the agent robust to prompt-template variations at inference.
- **Authors:** Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, Shunyu Yao (Princeton + Cambridge)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.05915
- **Relevant topics:** agent fine-tuning, ReAct, Reflexion, multi-method training

## Abstract
FireAct is an agent-fine-tuning study showing that Llama-2 and CodeLlama fine-tuned on trajectories collected via **multiple** prompting methods (Chain-of-Thought, ReAct, Reflexion) on HotpotQA and Bamboogle outperforms single-method fine-tunes. FireAct-13B reaches 39.3% on HotpotQA EM, competitive with GPT-4 ReAct (42%) at a fraction of the cost.

## Key Contributions
- First systematic study of **method-diversity** in agent SFT.
- **FireAct dataset** — 2K+ trajectories mixing CoT / ReAct / Reflexion solutions.
- Evidence that agent fine-tuning is most effective when (a) method-diverse and (b) multi-task.
- Lightweight 7B/13B checkpoints released.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** questions from HotpotQA, Bamboogle, and related multi-hop QA sets.
- **Teacher prompting methods used in parallel:**
  - **Chain-of-Thought (CoT):** GPT-4 solves with reasoning only, no tool use.
  - **ReAct:** GPT-4 alternates `Thought / Action / Observation` with a Wikipedia search tool.
  - **Reflexion:** GPT-4 attempts a trajectory, observes failure, reflects, retries up to N=3.
- **Trajectory collection:** for each question, collect one trajectory per method; label with method name (so the model can learn method-specific formatting).
- **Filtering:**
  - Gold-answer exact-match for HotpotQA.
  - Drop trajectories that fail final answer OR exceed token budget.
- **Training mix:** at SFT, trajectories from all methods are combined with method-specific system prompts so the student can switch style at inference.
- **Output shape:** ~2,000 trajectories; avg ~800 tokens (CoT) to ~2K tokens (Reflexion with retries).
- **Teacher model:** GPT-4.
- **Cost:** ~$3K GPT-4 API.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** Wikipedia search API (ReAct sub-task) + internal reasoning (CoT).
- **Action space:** `Search[query]`, `Lookup[term]`, `Finish[answer]` — classic ReAct triplet.
- **Trajectory length:** 3–10 steps; total token length 500–3000.
- **Success criterion:** exact-match on question gold answer.
- **Data scale:** small (2K trajectories) — the paper argues method diversity compensates for volume.
- **Method-mixing ablation:**
  - CoT-only SFT: 38.9% HotpotQA.
  - ReAct-only: 37.3%.
  - Reflexion-only: 35.2%.
  - Mix (all three): **40.0%** — strict improvement, evidencing the core claim.

## Quality / diversity evaluation
- FireAct-13B on HotpotQA EM: 39.3 (vs Llama-2-13B-Chat ReAct baseline 19).
- Method-robust: can switch between CoT/ReAct/Reflexion at inference via system prompt.
- Generalization to Bamboogle (unseen): modest but positive.

## Risks + gotchas
- **Very small corpus:** ~2K trajectories is brittle; quality strongly depends on GPT-4 teacher.
- **Narrow task family:** multi-hop QA only; transfer to web / code agents is unclear.
- **Reflexion cost:** each Reflexion trajectory involves multiple retries → 3–5× the API cost of vanilla ReAct data.

## Connections
- Contemporary: [[agenttuning]], [[lumos]].
- Direct precursor to richer mixes: [[agent-flan]] (scales this idea to millions).
- ReAct / Reflexion ancestors: ReAct (Yao 2022), Reflexion (Shinn 2023).
