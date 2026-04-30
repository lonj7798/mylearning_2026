<!-- scope: agentic trajectory synthesis — self-generated agent trajectories with zero human labels
     deps: [[self-instruct]]
     see-also: [[agent-flan]], [[fireact]]
-->

# AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning
- **Core Insight:** Agent fine-tuning data can be generated entirely by a base LLM from a small seed of tool descriptions plus raw questions — a planning-agent, a tool-using-agent, and a reflection-agent cooperate to produce successful trajectories, and only successful ones train the final agent; no human labels or frontier-teacher data required.
- **Guideline:** For low-cost agent SFT, run a self-differentiation loop where a single base model plays three roles (Plan, Tool, Reflect) on raw QA tasks; keep only self-verified successful trajectories; iterate.
- **Authors:** Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo, Wangchunshu Zhou, Yuchen Eleanor Jiang, Chengfei Lv, Huajun Chen (Zhejiang U + AntGroup + AIWaves)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2401.05268
- **Relevant topics:** self-generated agent data, no-human-label, plan/tool/reflect, QA agents

## Abstract
AutoAct trains agents for complex QA starting only from a small tool-set and a raw question pool, with no human-labeled trajectories and no distillation from closed-source models. A self-differentiation loop splits the base Llama-7B/13B into three specialized sub-agents (Plan, Tool, Reflect) via LoRA, each fine-tuned on its role-specific slice of the self-generated trajectories. The method achieves competitive performance on HotpotQA and ScienceQA without any external teacher.

## Key Contributions
- **Self-differentiation loop** — base model splits into three role-specialized sub-agents.
- **Zero human / GPT-4 labels** — trajectories are self-generated and self-filtered.
- Strong performance on multi-hop QA comparable to GPT-4-teacher baselines.
- Released code + datasets.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** small tool library (Search, Lookup, Calculator), raw QA dataset (HotpotQA).
- **Step 1 — Meta-agent prompts the base model** to classify its role (Plan, Tool, Reflect) for each turn.
- **Step 2 — Self-generation:** base model rolls out a trajectory for each question under each role.
- **Step 3 — Self-consistency filter:** only keep trajectories whose final answer matches gold answer OR matches self-consistency majority.
- **Step 4 — Role-specific SFT:** fine-tune three LoRAs (Plan, Tool, Reflect) on their respective role subsets.
- **Step 5 — Iterate:** use the trained sub-agents to generate new trajectories; repeat.
- **Output shape:** ~5K–10K self-generated trajectories across iterations; avg 5–10 turns.
- **Teacher:** none; base Llama-2-7B/13B self-instructs.
- **Cost:** compute only (no API fees).

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** QA with Wikipedia Search + calculator tools.
- **Action space:** `Search / Lookup / Calculate / Finish` (classic ReAct).
- **Trajectory length:** 5–10 turns.
- **Success criterion:** exact-match gold answer (HotpotQA) or self-consistency (no-gold settings).
- **Data scale:** ~10K self-generated trajectories — small but label-free.
- **Why sub-agent split helps:** each role has a narrower distribution than a unified ReAct agent, so a small LoRA can specialize effectively.

## Quality / diversity evaluation
- AutoAct-Llama-2-13B: HotpotQA EM ~36 — within ~4 points of GPT-4-teacher-distilled baselines.
- ScienceQA: strong transfer without explicit training.
- Iteration ablation: performance grows over 3 self-play iterations, saturates by iteration 4.

## Risks + gotchas
- **Self-consistency can anchor on wrong answers** — if the base model has a systematic bias, self-filter preserves it.
- **Narrow domain:** QA only; extension to web / code agents requires additional work.
- **Small data scale** means robustness on novel domains is limited.

## Connections
- Lineage: [[self-instruct]] (bootstrap from base LM), [[star]] (self-generated rationale + filter).
- Role-specialization cousin: [[lumos]] (Plan/Ground/Execute modular training).
- Related: [[spin]] (self-play for alignment).
