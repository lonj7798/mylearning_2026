---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lumos.md
source_url: https://arxiv.org/abs/2311.05657
created_at: "2026-04-23"
---

# Excerpt: Lumos — Plan / Ground / Execute as trajectory format

**Source library:** `wiki/raw-data/llm-training/papers/lumos.md`
**Paper:** Yin et al. 2023 (Allen AI + UCLA + UW), "Lumos: Learning Agents with Unified Data, Modular Design, and Open-Source LLMs."

---

## Why this source anchors ch-27 §1.2

Ch-27 §1.2's claim — "Lumos is the format you copy when designing a new agent-trajectory format" — rests on one specific design move: the three-module decomposition with a **unified action grammar**. This excerpt unpacks why the grammar matters more than the modules.

## The three modules

From the source (lines 32-36):

> ### Three-module data conversion
> - **Planning:** GPT-4 annotator receives a raw task + gold answer and produces a list of subtasks with natural-language descriptions.
> - **Grounding:** GPT-4 converts each subtask into a specific action expressed in a unified grammar — e.g., `Search[query]`, `Retrieve[doc_id]`, `Calculate[expr]`, `Click[element]`.
> - **Execution:** actual tool / environment / function used for the grounded action. In training data, executions are the observed results.

Three aligned supervised targets per data instance: `(task → plan)`, `(subtask → grounded_action)`, `(grounded_action → execution_result)`. Each module can be its own LoRA or its own head.

The structural trick is **alignment** — all three targets share the same underlying trajectory, so errors in Planning propagate to Grounding and Execute errors are observable. You can train on the joint distribution, but you get better generalization training the modules independently because each has a narrower output distribution.

## The unified grammar

From the source (line 49):

> - **Action space:** unified grammar over `Search`, `Retrieve`, `Calculate`, `Click`, `Type`, `Back`, `Finish`.

Seven actions. That's the entire API. This is the critical bet: a narrow action vocab that covers 80% of agent use cases (QA retrieval, calculation, web navigation, terminal output), with graceful failure for the rest.

Compare to Kimi-K2's 20,000+ tool schemas ([[kimi-k2-agentic-data]]). K2 trades safety and training stability for generalization breadth; Lumos trades breadth for training stability and generalization *across environments* (QA + web + math + household all fit the 7-action grammar).

## Lumos-I vs Lumos-O

From the source (lines 40-42):

> - **Lumos-I (iterative):** model alternates Plan → Ground → Execute at each step, replanning after each observation.
> - **Lumos-O (onetime):** plan the whole task up-front, then ground and execute sequentially.

Two inference modes, same trained modules. Lumos-I is more robust to environmental surprises (a click fails, a search returns nothing) because replanning corrects course. Lumos-O is faster at inference (plan once, skip the re-plan overhead) but brittle when environment state changes mid-trajectory.

## The generalization result

From the source (line 58):

> - Generalization to unseen task: ~8-point drop only, vs ~20-point drop for monolithic ReAct fine-tunes.

This is the one number to remember. On held-out tasks, Lumos loses ~8 points; monolithic ReAct fine-tunes lose ~20 points. Factor of 2.5 better held-out generalization. **The decomposition is doing work.**

The mechanism: when the action vocab is shared but concrete tools differ across environments, the Grounding module has seen `Click[element]` on both Mind2Web and ALFWorld; the monolithic agent has only seen the environment-specific action sequences. At test time, a new environment with the same action vocab is in-distribution for Grounding and out-of-distribution for the monolithic agent.

## Module swap at inference

From the source (line 53):

> - **Module decoupling benefit:** changing the execution backend (e.g., swapping a retriever) does not require retraining planning / grounding.

This is the API-stability argument. If Wikipedia's search API changes tomorrow, you rewrite the Execution module (or swap in a new retriever) without retraining Planning or Grounding. For the monolithic ReAct agent, every retrieval-API change is a full fine-tune.

## What to take from Lumos for ch-27

1. **Narrow action grammars generalize.** 7 actions covering 4 environments beats 50 actions for one environment.
2. **Modular decomposition halves the held-out generalization drop.** ~8 pts vs ~20 pts is the number.
3. **Plan / Ground / Execute is a reusable format.** Even if you deploy a monolithic model, annotate your data in this format — you can collapse the three modules at SFT time but still get the structural supervision benefit.
4. **Lumos-I vs Lumos-O is an inference-time tradeoff**, not a training-time tradeoff. Train once, choose at inference.

## Connections

- [[ch-27]] §1.2, §5 — decomposition pattern in the design-space tour, action-space grammar in the taxonomy table.
- [[excerpts/agentinstruct]] — AgentInstruct's 43-generator fan-out is a different diversity move; Lumos's module fan-out is a structural one.
- [[excerpts/autoact]] — AutoAct reuses the role-specialization pattern (Plan/Tool/Reflect) without external teacher.
- [[excerpts/openhands-data]] — OpenHands's unified action abstraction is Lumos-grammar at production scale.
