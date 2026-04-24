---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: Li et al. 2023 — "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Models"
source_url: https://arxiv.org/abs/2303.17760
created_at: "2026-04-23"
---

# Excerpt: CAMEL — Inception Prompting and Semantic Termination

**Source:** `wiki/raw-data/llm-training/papers/camel.md`
**Primary paper:** Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem (KAUST), 2023
**arXiv:** https://arxiv.org/abs/2303.17760

---

## Why this source anchors ch-25 §3

CAMEL does two things that no prior pipeline did: (1) it uses **role pairs** as the primary diversity axis (not topics), and (2) it introduces **semantic termination** via a task-completion marker instead of a fixed turn cap. Both moves propagate into every subsequent agentic / tool-calling synthesis pipeline — AgentInstruct, APIGen-MT, ToolACE all carry the Inception-prompting DNA.

From the source:

> *"Two LLM agents assigned distinct roles ('AI User' giving tasks, 'AI Assistant' solving them) and prompted under an Inception-prompting scheme produce genuinely collaborative, goal-directed multi-turn dialogues whose content is richer and more task-oriented than undirected self-chat — the role-specification is what unlocks diversity."*

---

## The Inception Prompting template — verbatim

From [[camel]] (and the open CAMEL implementation, `camel-ai/camel`):

```
Never forget you are a <user-role> and I am a <assistant-role>.
Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete the task: <task>.
You must instruct me based on my expertise and your needs to solve the task.

Give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
You should instruct me, not ask me questions.

Here is the format you must strictly follow:
Instruction: <YOUR INSTRUCTION>
Input:       <YOUR INPUT, or "None">

Do not add anything else other than your instruction and the optional input.

When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task.
```

The mirror prompt for the AI Assistant instructs:

> *"'I must provide a solution in the format `Solution: <YOUR SOLUTION>` and I must not flip to instructing you.'"*

Four load-bearing properties:

1. **Role lock.** `"Never flip roles! Never instruct me!"` is the first line because GPT-3.5's natural default, when role-playing a user, is to drift toward asking clarifying questions (assistant-like behavior). The negative constraint prevents the collapse.
2. **Protocol lock.** `"You should instruct me, not ask me questions."` The user must *instruct*, not *question*. This is a stricter protocol than either Baize's free-form [|Human|] or UltraChat's "behave as a curious user" — CAMEL narrows user behavior to a single speech act.
3. **Format lock.** `Instruction: <...> / Input: <...>` structured format. This is what keeps turn boundaries clean in the generated text. The assistant's mirror uses `Solution: <...>`.
4. **Semantic termination.** `<CAMEL_TASK_DONE>` is a single-token marker. The dialogue ends when the *user* (not the assistant) decides the task is solved. This is the key novelty — not "stop after K turns" but "stop when the work is done."

---

## The 50 × 50 × 20 combinatorial grid

From [[camel]]:

> *"50 assistant roles: Accountant, Architect, Astronaut, Biologist, … (domain experts).*
> *50 user roles: Entrepreneur, Graduate Student, Journalist, … (information-seeker personas).*
> *20 topic domains: per dataset variant (Society, Code, Math, Science)."*

How the role lists themselves were generated: prompt GPT-3.5 once to enumerate "50 common assistant roles for a knowledge-work society" and "50 common user roles for information-seeking." The bootstrap is meta — the teacher generates its own diversity axes. This is the first appearance of a pattern that becomes common by 2024 (Persona-Hub's 1B personas are a scaled-up version of the same trick, with real web bios as the source instead of meta-prompt enumeration).

Crossing 50 × 50 × 20 yields 50,000 conditioning triples. The AI Society dataset has ~1M dialogues because each triple is sampled 20× at different task instantiations. Cost ~\$5–10K API at 2023 Turbo rates.

---

## Semantic termination — why it changes turn-count statistics

From the raw-data notes:

> *"Task-completion protocol is the novel element — turns end when task is solved, not at a fixed turn budget."*

> *"Turn-count distribution: median 6-8, tail to 20."*

Compare with Baize's median-4 / max-10 and UltraChat's median-4-to-6 / max-14. CAMEL's distribution has both a higher median *and* a longer tail. The explanation:

- **Easy tasks terminate fast.** If the assistant's first Solution resolves the task, the user emits `<CAMEL_TASK_DONE>` at turn 3 (user-instruction + assistant-solution + user-done). This is ~15% of dialogues.
- **Hard tasks run long.** Multi-step code debugging, math proof construction, or multi-aspect writing tasks produce 15–20-turn dialogues. The cap at 20 is a safety net, not a typical terminator.
- **The median sits at 6–8** because typical knowledge-work tasks take a few back-and-forths to resolve (clarify, attempt, correct, attempt again).

This is the correct shape for task-oriented data. Baize/UltraChat's fixed caps squash the hard-task tail; CAMEL preserves it. Every later agentic-data pipeline inherits this — APIGen-MT explicitly cites CAMEL for the semantic-termination pattern.

---

## The formulaic-leak failure mode

From [[camel]]:

> *"Role-play artifacts: both agents introduce themselves in every turn ('As an accountant,…'); downstream model must learn to ignore these formulaic phrases."*

> *"Protocol rigidity: `<CAMEL_TASK_DONE>` formulation leaks into student models."*

Consequence: students fine-tuned on CAMEL data emit phrases like "As an accountant, I'd suggest…" and sometimes produce `<CAMEL_TASK_DONE>` as a completion marker even outside training distribution. Standard mitigations:

- **Regex strip in post-processing.** Strip `"As a <role>, "` openers before SFT.
- **Token-replacement.** Replace `<CAMEL_TASK_DONE>` with the student's actual EOS token during data-prep so the student learns the *semantic function* of termination (end-of-turn), not the specific string.
- **Ablation.** Some downstream mixes exclude CAMEL outright in favor of its cleaner successors (Airoboros, ToolACE) that have learned from this failure mode.

---

## Role-pair quality is uneven

From [[camel]]:

> *"50×50 coverage does not ensure quality — many role pairs yield low-quality dialogs (e.g., Astronaut × Bartender)."*

Why: the Inception prompt forces both roles to collaborate on a shared task. If no natural task connects the two roles (what does an Astronaut *need* from a Bartender?), the teacher produces contrived or low-content dialogues. This is a structural limitation of the combinatorial-grid approach — diversity of *conditioning* does not equal diversity of *content quality*.

Practitioner response in later pipelines: pre-filter role pairs by teacher-judged plausibility before generating. AgentInstruct explicitly introduces a "task suggester" stage that rejects implausible `(role_A, role_B, task)` triples before the expensive dialogue-generation step.

---

## What CAMEL contributes to the lineage

- **Inception Prompting** — the template pattern. Every explicit-role multi-agent pipeline after 2023 inherits it.
- **Semantic termination** — `<CAMEL_TASK_DONE>` becomes "executable verifier signal" in later agentic pipelines (APIGen-MT uses successful function-call execution; SWE-RL uses unit-test pass as the analog).
- **Role as diversity axis** — combines with UltraChat's taxonomy axis and later SystemChat's persona axis. Modern mixes condition on all three.

From the raw-data notes:

> *"Inception-prompting lineage adopted in: [[agentinstruct]], [[apigen-mt]], [[toolace]] (MAI multi-agent dialog)."*

---

## Connections

- Chapter synthesis: [[ch-25]] §3.
- Role-pair precedent: [[excerpts/ultrachat-two-model-protocol]] (task-less two-model role-play).
- Semantic-termination successor: ch-26's APIGen-MT uses execution-success as the termination signal analog.
- Persona axis extension: [[excerpts/system-prompt-diversity]].
