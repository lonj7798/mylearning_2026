---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/metamath.md
source_url: https://arxiv.org/abs/2309.12284
created_at: "2026-04-23"
---

# Excerpt: MetaMath — four question-rewrite operators, FOBAR is the non-obvious one

**Source library:** `wiki/raw-data/llm-training/papers/metamath.md`
**Paper:** Yu et al. 2023, "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models"

---

## Why this source anchors ch-24 §3

Ch-24 treats MetaMath as the canonical *question-side* augmentation paper. Where OpenMathInstruct scales *solutions per problem*, MetaMath scales *problems per seed*. The four operators are reusable templates that later papers (OMI-2's question-augmentation, MathScale's concept-graph, WizardMath's Evol-Instruct) inherit.

---

## The four operators — with FOBAR made precise

From the source (§Synthesis pipeline):

1. **AnsAug (Answer Augmentation)** — keep Q unchanged, sample K CoT solutions, keep the ones that match gold. Roughly equivalent to OpenMathInstruct's sampling loop at K ≈ 25.
2. **Rephrasing** — GPT-3.5 rewrites Q in different words preserving the math. Re-solve. Keep if final answer matches gold.
3. **Self-Verification (SV)** — rewrite as: *"Given Q and candidate answer A', determine correctness; if not, fix."* Forces the model to learn the verification operation explicitly.
4. **FOBAR (Forward-Backward Reasoning)** — mask a number in Q, treat the gold answer as known, ask "what is the masked number?"

The FOBAR template, lifted verbatim from the source:

```
Original: "Jane has 3 apples and buys 5 more. How many apples?"  →  8
FOBAR:    "Jane has 3 apples and buys x more. She now has 8. What is x?"  →  5
```

The correctness filter for FOBAR is not gold-match in the original sense — the gold "5" is the masked value. The trace is accepted iff the teacher's backward solution reconstructs the masked number. This is exactly the kind of verifier-depth move ch-24 §1 catalogues.

---

## The additivity of operators — why the stack matters

From the source (§Quality evaluation):

> AnsAug alone gives ~4-point gain over vanilla SFT; adding Rephrasing +3; SV +2; FOBAR +3. Stacking all four is **additive**.

Ch-24 §3 treats this as operationally significant: the operators are **complementary** because each inoculates against a different overfitting mode. Rephrasing breaks surface-form memorization; SV teaches verification-as-a-skill; FOBAR teaches reverse-direction reasoning. The additivity suggests the underlying error modes are close to orthogonal — a rare property in data-augmentation stacks.

---

## Why FOBAR is the non-obvious win

From the source (§Modality-specific technical details):

> **Why FOBAR helps**: teaches the model that a chain of reasoning can be run in reverse — reduces "direction overfitting" to forward word-problem templates.

Forward word problems have a rigid template: given X, compute Y. After training on millions of forward problems, a model memorizes the *direction* of inference. FOBAR inverts that: the output becomes the input, and the input becomes the unknown. This is algebraically trivial for a human but distributionally novel for a model that has only seen forward problems.

An under-appreciated consequence: FOBAR produces problems whose *answers are often simpler than the originals*. If forward Q has answer 87 computed via a three-step chain, the FOBAR equivalent might have answer "3" as the masked intermediate. Students trained on FOBAR learn to reason about simpler targets through more-complex prefixes — a useful stress test for the chain-of-thought machinery.

---

## Teacher ceiling — the FOBAR gotcha

From the source (§Risks + gotchas):

> FOBAR soundness: not every forward problem has a unique backward answer; authors filter but some ambiguous FOBARs leak in.

Example: "Jane has 3 apples and buys x. She now has between 5 and 10." Multiple x values work. The source does not quantify the leak rate; ch-24 §3 cites the operator as effective-in-practice while noting the theoretical soundness gap.

Rephrasing has a distinct failure mode: occasional *semantic drift* where the rephrased problem has a different numerical answer than the original. Authors filter by re-solving and comparing to gold; anything that fails the comparison is dropped.

---

## Numbers to anchor ch-24 §3

From the source (§Quality evaluation):

- MetaMath-7B: **66.5 GSM8K / 19.8 MATH**.
- MetaMath-70B: **82.3 GSM8K / 26.6 MATH**.
- Dataset size: **395K (question, CoT, answer) triples** = ~25× the seed count, entirely from 15K source problems.
- Teacher: GPT-3.5-turbo, ~$5-15K API cost.

Compare these to OpenMathInstruct-2's **91.7 / 67.8 on 14M samples** via Llama-3.1-405B. The gap is mostly teacher strength (GPT-3.5 vs 405B) and scale-of-solutions-per-problem (25× per seed for MetaMath vs thousands for OMI-2). The question-side operators themselves transfer — OMI-2 uses them at 405B scale.

---

## Connections

- [[excerpts/openmathinstruct]], [[excerpts/openmathinstruct-2]] — the solution-side companions; OMI-2 absorbs MetaMath's question-augmentation idea at stronger-teacher scale.
- [[excerpts/rstar-math]] — MCTS as a different route to the same goal of trace diversity.
- [[ch-24]] §3 (question-side diversity) and §8 (practical guidance — FOBAR as template).
