---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/openmathinstruct.md
source_url: https://arxiv.org/abs/2402.10176
created_at: "2026-04-23"
---

# Excerpt: OpenMathInstruct-1 — the wide-short-CoT template with a weak open teacher

**Source library:** `wiki/raw-data/llm-training/papers/openmathinstruct.md`
**Paper:** Toshniwal et al. 2024, "OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset" (Nvidia)

---

## Why this source anchors ch-24 §2

Ch-24 opens the wide-short-CoT lineage with OpenMathInstruct-1 because it is the first paper to demonstrate a **permissively-licensed** alternative to GPT-4 distillation for math SFT. Two claims live or die on this source:

1. A *weak* open teacher (Mixtral-8x7B-Instruct, Apache-2.0) can match GPT-4-derived datasets in downstream fine-tune quality, provided you spend your budget on **sampling many solutions per problem and filtering aggressively by final-answer correctness**.
2. **Tool-Integrated Reasoning (TIR)** — CoT interleaved with executable Python — covers the arithmetic-precision weakness of text-only CoT; OpenMathInstruct-1 ships TIR as default.

---

## Pipeline numbers that matter

Directly from the source (§Synthesis pipeline):

- Seed: GSM8K train 7.5K + MATH train 7.5K = ~15K problems.
- K = 32-64 solutions sampled per problem from Mixtral-8x7B-Instruct.
- Execution loop: `<llm-code>` Python → sandbox exec → splice `<llm-code-output>` back → continue.
- Accept iff final answer matches gold via SymPy-canonical equivalence (MATH) or numeric string match (GSM8K).
- Output: **1.8M solutions** — ~120 per GSM8K problem, ~100 per MATH problem.
- Teacher compute: ~500K GPU-hours.

The ratio is striking: **1.8M accepted / (15K × K=64) candidates ≈ 2%** overall acceptance, dominated by MATH-level-5 failures where Mixtral struggles even at K=64.

---

## The TIR template — why it matters operationally

From the source (§Modality-specific):

```
<llm-code>
from sympy import symbols, solve
x = symbols('x')
solve(3*x - 6)
</llm-code>
<llm-code-output>
[2]
</llm-code-output>
So x = 2.
```

Three properties the student inherits from this template:

1. The model learns to **write syntactically valid Python** inside its reasoning. For Track-4 RL consumers this matters: downstream code-executor reward models score actual executability.
2. The splice-back rule teaches the model that **execution results are authoritative** — later reasoning is conditioned on `<llm-code-output>`, not on the model's own arithmetic.
3. The structure is **self-segmenting**. `<llm-code>` is a natural step boundary — easier to post-hoc attach step-level preference signals ([[step-dpo]]) than to raw CoT.

---

## The verifier — and its silent leak

OpenMathInstruct-1 uses **terminal-only** correctness: only the final boxed answer is checked. Ch-24 §8's "false positives compound" warning draws on this paper's own risk note: ~5-10% of kept traces reach the gold answer via flawed intermediate steps. This is the cap that Step-DPO and OmegaPRM are built to raise.

The specific failure mode to remember: a Mixtral solution that computes `3 × 4 + 2 = 14` (wrong intermediate) then `14 - 2 = 12` (wrong but cancels). SymPy sees 12, the gold is 12, the trace is accepted. Students trained on such traces learn the *compensating-error shortcut*.

---

## Evaluation anchor numbers

From the source (§Quality evaluation):

- OpenMath-Mistral-7B: **80.2 GSM8K / 44.5 MATH**.
- OpenMath-Llama2-70B: **84.6 GSM8K / 50.7 MATH**.
- Ablation: CoT-only (drop code) → MATH -8 absolute. PoT-only (drop natural language) → GSM8K -5.

The 70B number is the one the ch-24 §2 comparison implicitly contrasts against OpenMathInstruct-2's OpenMath2-Llama3.1-8B at **91.7 / 67.8** — an 8B on a 405B teacher beats a 70B on a Mixtral teacher. The teacher-strength lesson is in that delta.

---

## Connections

- [[excerpts/openmathinstruct-2]] — the direct successor; 8× scale, 405B teacher, pure text-CoT.
- [[excerpts/metamath]] — the question-augmentation companion; OpenMathInstruct-1 is solution-side only, MetaMath is question-side only.
- [[excerpts/rstar-math]] — the step-level successor; replaces terminal-match with step-level code execution.
- [[ch-24]] §2 (wide-short-CoT lineage) and §8 (false-positive compounding).
