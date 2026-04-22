<!-- scope: agentic trajectory synthesis — modular Plan/Ground/Execute open-source agent training
     deps: [[agenttuning]]
     see-also: [[fireact]], [[agent-flan]]
-->

# Lumos: Learning Agents with Unified Data, Modular Design, and Open-Source LLMs
- **Core Insight:** Agent trajectories should be decomposed into three swappable modules — **Planning** (high-level subtasks), **Grounding** (subtask → concrete action), **Execution** (tool call / env interaction) — and each module trained separately on converted data; this yields better generalization than training a monolithic agent on raw ReAct trajectories.
- **Guideline:** When constructing agentic SFT data, convert every trajectory into the (Plan, Ground, Execute) tri-layer format; train three LoRAs / heads independently on their respective targets; compose at inference for modular tool/task swap.
- **Authors:** Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Raghavi Chandu, Kai-Wei Chang, Yejin Choi, Bill Yuchen Lin (Allen AI + UCLA + UW)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2311.05657
- **Relevant topics:** modular agents, planning, grounding, open-source agent SFT, trajectory decomposition

## Abstract
Lumos proposes a unified three-module agent architecture — Planning, Grounding, Execution — and converts existing agent trajectory datasets (from HotpotQA, ALFWorld, WebShop, Mind2Web, ScienceQA, StrategyQA, etc.) into this format. The Lumos suite includes an iterative variant (Lumos-I) and an onetime variant (Lumos-O), trained on Llama-2 base. Lumos outperforms much larger closed-source agents on complex QA, web, and math tasks, and generalizes to unseen tasks — the decomposition is doing work.

## Key Contributions
- **Unified Plan/Ground/Execute data format** for cross-task agent SFT.
- **Pipeline for converting existing agent data** into this format via GPT-4 annotator.
- Lumos-7B/13B open models; comparable to GPT-4-based ReAct agents on several benchmarks.
- Strong generalization to held-out environments.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Source data
- **HotpotQA** (multi-hop QA) → complex-reasoning trajectories.
- **StrategyQA** → open-domain QA.
- **ALFWorld, WebShop, Mind2Web** → interactive-env trajectories.
- **Musique** → multi-step QA.
- **GSM8K, MATH** → math-reasoning trajectories.

### Three-module data conversion
- **Planning:** GPT-4 annotator receives a raw task + gold answer and produces a list of subtasks with natural-language descriptions.
- **Grounding:** GPT-4 converts each subtask into a specific action expressed in a unified grammar — e.g., `Search[query]`, `Retrieve[doc_id]`, `Calculate[expr]`, `Click[element]`.
- **Execution:** actual tool / environment / function used for the grounded action. In training data, executions are the observed results.
- Each data instance becomes three aligned supervised targets: (task → plan), (subtask → grounded_action), (grounded_action → execution_result).

### Training
- Two variants:
  - **Lumos-I (iterative):** model alternates Plan → Ground → Execute at each step, replanning after each observation.
  - **Lumos-O (onetime):** plan the whole task up-front, then ground and execute sequentially.
- Each module can be a separate head or a separate LoRA.
- **Output shape:** ~40K task instances; each decomposed into 3–8 sub-task / action triples → ~200K training turns.
- **Teacher model:** GPT-4 for conversion / annotation.
- **Cost:** ~$15K GPT-4 API.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** heterogeneous — QA (retrieval), web (clicks), math (execution), household sim (ALFWorld).
- **Action space:** unified grammar over `Search`, `Retrieve`, `Calculate`, `Click`, `Type`, `Back`, `Finish`.
- **Trajectory length:** avg 6–10 steps (subtasks); total token length 1K–5K.
- **Success criterion:** per-task (gold answer match for QA, task-complete signal for sims).
- **Data scale:** ~40K tasks → 200K module-level training pairs.
- **Module decoupling benefit:** changing the execution backend (e.g., swapping a retriever) does not require retraining planning / grounding.

## Quality / diversity evaluation
- Lumos-13B (onetime): HotpotQA 39.3 EM — beats LLaMA-2-13B-chat ReAct (30.5) and approaches GPT-4 ReAct (44).
- Lumos-7B on Mind2Web: competitive with 30B-class alternatives.
- Generalization to unseen task: ~8-point drop only, vs ~20-point drop for monolithic ReAct fine-tunes.

## Risks + gotchas
- **Action-grammar rigidity:** if a new environment's action space doesn't fit the unified grammar, extensive conversion is needed.
- **GPT-4 annotator as bottleneck:** conversion quality limits downstream quality.
- **Three-module overhead:** inference involves three serial forward passes per step; slower than monolithic ReAct.
- **Narrow success signal per module:** hard to optimize modules jointly without full-trajectory RL.

## Connections
- Contemporary: [[agenttuning]], [[fireact]].
- Ancestor of modular agent lineage (continued in OpenHands, autonomous-agent stacks).
- Data-conversion idea re-used in [[agent-flan]] and [[agentinstruct]] (MS).
