---
chapter: ch-20
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/distilling-step-by-step.md
source_url: https://arxiv.org/abs/2305.02301
created_at: "2026-04-23"
---

# Excerpt: Distilling Step-by-Step — joint (label + rationale) training

**Source library:** `wiki/raw-data/llm-training/papers/distilling-step-by-step.md`
**Paper:** Hsieh et al., *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*, ACL 2023 Findings.

---

## Why this source anchors ch-20

Orca is the popular distillation paper of 2023; *Distilling Step-by-Step* is the **sharper theoretical** one. Orca says "richer traces help"; Hsieh et al. give a concrete, testable claim: **a T5-770M student beats PaLM-540B few-shot on 4 benchmarks using only 80% of the labeled data**, if you train it with a joint (label, rationale) multi-task objective. This is the cleanest quantitative result in pre-R1 reasoning distillation, and the multi-task framing is the conceptual ancestor of every later reasoning-SFT recipe.

From source line 7:

> **Core Insight:** Extract natural-language rationales (not just labels) from a large teacher LLM via few-shot CoT prompting, then train a small student in a multi-task framework (predict label + generate rationale) — this outperforms label-only distillation and lets a 770M student beat a 540B few-shot teacher on multiple benchmarks using only 80% of the available data.

---

## The two-stage recipe — verbatim

From source lines 17–22:

> **Two-stage method:**
>   1. Extract rationales from teacher via few-shot CoT prompting on unlabeled data.
>   2. Train student with multi-task objective: label prediction + rationale generation.

Concretely (source lines 25–37):

```
Seed input  : task data (e.g., ANLI premises/hypotheses, CQA questions)
Generation  : few-shot CoT prompt PaLM-540B with 3–8 exemplars showing
              "question → rationale → answer"; run on every training example
Filtering   : drop teacher outputs with wrong format or wrong answer
Student     : T5 (220M / 770M / 11B)
  input     : x
  output    : (a) label y via classification head
              (b) rationale r via sequence head
  loss      : L = L_label + λ · L_rationale
Teacher     : PaLM-540B (original paper)
```

The `λ` balances label fidelity against rationale generation; the paper uses a single λ per benchmark, tuned via small validation sweep. Both heads share the T5 encoder — the rationale head exists to **regularize the encoder**, not to be used at inference.

---

## Why the result is important

A naive reading: "small student beats bigger teacher — scaling is dead." That's wrong. The small student beats the teacher's **few-shot** performance, not the teacher's finetuned performance. The teacher is still more capable; what the result shows is that **the teacher's reasoning, compressed into the student's encoder as joint rationale-prediction training, generalizes better than few-shot prompting on the teacher**.

Why that's important for ch-20:

- It establishes that **rationales carry more per-token signal than labels** — the thesis every downstream reasoning-distillation paper assumes.
- It shows **the rationale head can be discarded at inference** without losing the benefit, because the gradients have already shaped the shared encoder. The 2025 descendants (R1-Distill, Bespoke-Stratos) relax the multi-task objective — they emit rationale *and* answer in a single autoregressive stream — but the underlying gradient-shaping claim is the same.
- It measures **data efficiency**: 80% of labeled data suffices. This is the seed of the 2025 "less is more" literature ([[s1]], [[limo]]).

---

## The benchmarks beaten (source line 39-44)

T5-770M Distilled-Step-by-Step > PaLM-540B few-shot on:

- **ANLI** (Adversarial NLI) — hard natural language inference.
- **e-SNLI** (explained SNLI) — NLI with required explanations.
- **CQA** (CommonsenseQA) — multiple-choice commonsense.
- **SVAMP** (Math Word Problems) — arithmetic with adversarial phrasings.

The last one (SVAMP) is the one that matters most for ch-20's thesis: math word problems are the test bed where trace-rich supervision has the highest leverage, and it's exactly the domain R1-distill would saturate 2 years later.

---

## Failure modes that still bite in 2025

Source lines 51–55 flag three risks:

1. **"Teacher rationale quality bounds student ceiling; hallucinated rationales teach wrong reasoning patterns."** — this is literally the "wrong-question-correctly" problem Open-R1 re-discovers in 2025 (ch-20 §5.5). The teacher can produce a rationale that *looks* valid but solves the wrong problem; the student learns the invalid reasoning pattern.

2. **"Verifier matters when labels are available — use correctness filtering."** — Bespoke-Stratos's SymPy + unit-test + LLM-judge stack is a direct response to this. A filter-free distillation pipeline inherits every teacher hallucination.

3. **"Task scope in the original paper is moderate-reasoning; for deep reasoning (MATH, competition-level), later work (R1 distill) requires dramatically longer traces."** — Hsieh's average rationale is ~50–200 tokens; R1 distill's average is ~5K tokens. The multi-task framing scales with length but the data engineering to produce clean long rationales is a different problem.

---

## How ch-20 cites this

The ch-20 read uses this paper in §2 as the conceptual anchor for "rationale is the supervision, label is the byproduct." The connection to R1-distill is explicit: R1-distill drops the joint multi-task objective (single autoregressive head for both rationale and answer) but inherits the *data-generation* half of the recipe — sample teacher CoT, filter by answer correctness, train student on (prompt → rationale → answer). Hsieh's 2023 paper is where that data-generation recipe first appears in a rigorous evaluation.
