---
chapter: ch-20
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/orca.md
source_url: https://arxiv.org/abs/2306.02707
created_at: "2026-04-23"
---

# Excerpt: Orca — explanation traces and the 16-system-message scaffolding

**Source library:** `wiki/raw-data/llm-training/papers/orca.md`
**Paper:** Mukherjee et al., *Orca: Progressive Learning from Complex Explanation Traces of GPT-4*, Microsoft Research, 2023.

---

## Why this source anchors ch-20

Orca is the paper that made "distillation-as-data" a specific thing. Before Orca, "distill from GPT-4" meant Alpaca-style: query GPT-4 with an instruction, save `(instruction, response)`, SFT a smaller model. The student got labels; the student learned to mimic GPT-4's *surface*. Orca's thesis is that the learning signal is not in the label — it's in the **explanation trace**, the sequence of tokens between the prompt and the answer that shows how the teacher arrived there. Ch-20 uses Orca as the origin-point of every later reasoning-distillation recipe.

From the source (line 7):

> **Core Insight:** Smaller models improve more from rich explanation traces and teacher reasoning signals than from short answer-only imitation.

and (line 8):

> **Guideline:** For reasoning-heavy SFT, distill not only final answers but also the teacher's step-by-step traces, intermediate rationales, and richer task formats.

These two bullets are the seed of the entire R1-distill line.

---

## The 16 system messages — what they actually do

The raw-data entry is brief on this (line 25: "Training data mixes explanation traces, step-by-step reasoning, and more complex instructions"). The load-bearing mechanism, from the paper's Appendix A, is that **every GPT-4 query is prefixed with one of 16 hand-crafted system messages** — not because the teacher needs it, but because each message induces a *different trace style*.

The reason this matters for ch-20's thesis: a student trained on 5M outputs under a single CoT prompt learns the **surface** of that prompt's trace pattern — how CoT is formatted — not the reasoning behind it. A student trained on 5M outputs under 16 *different* prompts has to learn **what is invariant across styles**. That invariant is the reasoning.

The 16 messages (paraphrased):

- Empty (control) — direct short answer.
- Generic CoT — "think step-by-step and justify your steps".
- Long-form explanatory — "the user doesn't need to search outside".
- Literal instruction-follower — no elaboration.
- ELI5 — "explain like to a five-year-old".
- ELI5 + CoT — combine both.
- Teacher / guideline-extraction — restate task rules first, then apply.
- Literal-then-CoT — follow exactly, then justify.
- Post-hoc rationale — answer first, then justify.
- Plan-and-execute — explicit plan before execution.
- Short CoT with mandatory rationale — one-line explanation required.
- Translation-calibrated — multilingual reasoning framing.
- Part-wise definition unpacking — decompose task into parts with examples.
- Example-grounded reasoning — at least one example per step.
- MCQ-elimination — answer then rule out distractors.
- Expert-grounded — "world-class expert in this field".

Three of these (plan-and-execute, part-wise unpacking, MCQ-elimination) are direct ancestors of 2025 reasoning recipes: R1's `<think>` wrapper is a compressed form of plan-and-execute; OpenThoughts' difficulty-graded prompts descend from part-wise unpacking; Sky-T1's GPT-4o rewriter drops preamble filler that Orca's messages *deliberately elicited*. The 2025 moves are all differentials *against* Orca's choices.

---

## Progressive learning mix

From source line 23: "Student is a 13B model. Training data mixes explanation traces, step-by-step reasoning, and more complex instructions."

The concrete mix:

- **5M ChatGPT traces** (cheap teacher, broad coverage) — trained first.
- **1M GPT-4 traces** (expensive teacher, depth) — trained second on the same prompt pool.

This is a **curriculum** in the data-ordering sense — the student sees easier explanations before harder ones. Not loss reweighting, not separate epochs per source — just literal temporal ordering. This is worth noting because it's cheap and the later literature (Llama 3, R1 cold-start) uses the same pattern: warm-start on weaker-teacher data, then refine on stronger-teacher data. The "progressive" in the paper title is this ordering, not any gradient-scale trick.

---

## Why this matters for R1-distill

The R1-distill pipeline (ch-20 §3) is Orca at a higher level of automation:

| Orca (2023) | R1-distill (2025) |
|---|---|
| Teacher = GPT-4 (chat-tuned, not reasoning-trained) | Teacher = R1 (RL-trained to emit long CoT) |
| 16 hand-crafted system messages induce trace variety | `<think>…</think>` wrapper; variety emerges from RL training |
| Filter: format check + task-level correctness when possible | Filter: SymPy / unit tests / V3-judge |
| 6M total traces; progressive ChatGPT→GPT-4 | 800K traces; single teacher, RS-SFT stage |
| Student: Llama-13B, BBH 49.7 | Student: Qwen2.5-32B, AIME 70%+ |

The structural invariant: *teacher emits trace → filter trace → SFT student on filtered trace*. The 2025 improvement is in the teacher (RL-trained reasoner vs chat-tuned assistant) and the verifier (symbolic / executional vs format-check), not the student-side training algorithm.

---

## What's still load-bearing and what dated fast

**Load-bearing (still true in 2025):**
- Richer trace = richer supervision; the teacher's reasoning is the signal.
- Variety in prompting induces variety in trace, which the student must generalize across.
- Curriculum by teacher strength (weak first, strong second) is cheap and works.

**Dated:**
- GPT-4 as the teacher — output license is contested, and GPT-4's reasoning traces are dominated by reasoning-trained models in 2025.
- 6M samples — [[bespoke-stratos]] and [[s1]] show 17K / 1K curated pairs reach comparable quality.
- Hand-crafted system messages — 2025 teachers emit reasoning-shaped traces natively; system-prompt steering is redundant.

The connection to [[orca-2]] is the bridge from "execute the strategy the teacher picked" to "learn when each strategy applies" — the Prompt Erasing trick — which is the direct conceptual ancestor of R1's auto-regulated `<think>` budget.
