---
chapter: ch-20
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sky-t1.md
source_url: https://novasky-ai.github.io/posts/sky-t1/
created_at: "2026-04-23"
---

# Excerpt: Sky-T1 — $450 reasoning distillation with an open-weights teacher

**Source library:** `wiki/raw-data/llm-training/papers/sky-t1.md`
**Release:** NovaSky / Sky Computing Lab, UC Berkeley (Dacheng Li, Shiyi Cao, Tyler Griggs, Ion Stoica et al.), 2025.

---

## Why this source anchors ch-20

Sky-T1 is the third point on the Stratos/Open-R1/Sky-T1 triangle, and the one that deliberately **swaps the teacher** — from DeepSeek-R1 (closed-API, MoE, 671B) to QwQ-32B-preview (open-weights, dense, 32B). The recipe cost collapses from Stratos's ~$800 teacher bill to $0 (local inference). Training collapses to $450 total. The student is weaker — by ~20 points on AIME — because QwQ is a weaker teacher. The tradeoff is the lesson.

From source line 7:

> **Core Insight:** A 32B reasoning model matching o1-preview on math and code is trainable for ~$450 by distilling 17K traces from QwQ-32B-preview and fine-tuning Qwen2.5-32B-Instruct for 19 hours on 8×H100 — dramatically lowering the entry cost for o1-class open research.

---

## The pipeline (source lines 23–37)

```
Seed:
  APPs + TACO (code)                  ~5K
  NuminaMath-CoT + AIME/AMC (math)    ~10K
  STILL-2 + science                   ~2K

Teacher:
  QwQ-32B-preview (Alibaba, Apache-2.0)
  Served locally via vLLM on H100s
  T = 0.7; max 8K tokens per trace
  1 trace per problem

Filter:
  Math     : SymPy on \boxed{} answer
  Code     : execute against public unit tests
  Science  : GPT-4o-mini LLM-judge

Reformatting (the key Sky-T1 move):
  QwQ's raw output uses a non-standard thinking/answer format
  Reformat to <|im_start|>…<|im_end|> chat template
  Remove redundant "Alright, let me think" preambles via GPT-4o rewriter
  This cleanup lifted AIME by +4 points

Output: 17,000 pairs
Median trace length ~3K tokens

Training:
  Qwen2.5-32B-Instruct, 3 epochs, LR 1e-5, seq 32K, BF16, FSDP on 8×H100
  Framework: Llama-Factory

Cost: $450 total (RunPod H100 rates, 19 hours)
      Teacher inference free (local QwQ)
```

Three design points are worth pulling out because they are specific to the open-weights-teacher setting.

---

## 1. Open-weights teacher enables local inference

Stratos paid DeepSeek ~$800 for R1 API credits. Open-R1 paid ~$10K (HF cluster + API mix). Sky-T1 paid **$0 for teacher inference** because QwQ's weights are Apache-2.0 and the team served it locally on their own H100s. The 17K traces cost compute-time (negligible on already-provisioned hardware) but no marginal dollars.

This is the **operational consequence of the licensing discussion** in ch-20 §7. An Apache-2.0 open-weights teacher collapses the data-generation cost; a proprietary-API teacher spreads it over per-token billing. For a budget-constrained lab, the difference is the difference between feasibility and infeasibility.

---

## 2. The GPT-4o rewriter — teacher-output sanitization

Source line 33:

> **Reformatting trick:** QwQ's raw output uses a non-standard thinking/answer format; NovaSky reformats to `<|im_start|>…<|im_end|>`-style chat template and removes redundant "Alright, let me think" preambles via a GPT-4o rewriter (this cleanup lifted AIME by +4 points).

This is an underappreciated finding: **+4 AIME points from cosmetic cleanup alone**. QwQ's raw traces open with "Alright, let me think step by step..." preambles that are pure filler. The student trained on raw QwQ output learns to emit the preamble; the student trained on rewritten output does not. Saving the preamble tokens gives the student more effective context for actual reasoning.

The GPT-4o rewriter is a **secondary teacher in the pipeline**. Note what this means for licensing: Sky-T1's corpus is Apache-2.0 on the dataset, but the rewriting step used GPT-4o (OpenAI API) — the same ToS grey area that affects Orca/Dolphin. This is a subtle licensing cliff; Sky-T1's corpus inherits the rewriter's ToS risk.

---

## 3. The teacher ceiling shows up in evaluation

Source line 46–47:

> Sky-T1-32B-Preview: MATH500 ~82.4%, AIME24 ~43.3%, LiveCodeBench-Easy ~86.3%, GPQA-Diamond ~56.8%.
> Matches o1-preview on MATH500; beats it on LiveCodeBench Easy; within 2 points on AIME24.

And source line 51:

> **QwQ teacher quality is lower than R1** on hardest problems — Sky-T1 underperforms Bespoke-Stratos and R1-Distill on AIME25.

On AIME24 (2024 problems; QwQ likely trained on them), Sky-T1 is competitive with R1-Distill. On AIME25 (2025 problems; less likely to be in QwQ's training data), Sky-T1 is ~20 points behind. The gap is the teacher: QwQ is genuinely weaker than R1 on novel problems, and the student's ceiling is the teacher's ceiling.

**This is the cleanest demonstration in 2025 open research that teacher selection dominates every other recipe choice once the filter stack is adequate.** Stratos and Sky-T1 use nearly identical filter stacks (SymPy + unit tests + LLM-judge); they differ almost entirely in teacher. The AIME gap is the teacher-ceiling gap.

---

## 4. OpenThoughts qualifies this — "stronger ≠ better teacher"

Sky-T1's result (QwQ < R1 ⇒ student is weaker) looks clean in isolation. [[open-thoughts]] complicates it: when the student is a *small Qwen* (7B), QwQ-32B actually beats R1 as a teacher, despite R1 being stronger on benchmarks. The explanation is distributional: QwQ's output distribution is closer to the Qwen2.5 base distribution; R1's distribution is farther from any single dense base.

Sky-T1 is a 32B student on a 32B-preview teacher — the distributional match is good, but QwQ's *capability ceiling* is lower. OpenThinker3-7B uses QwQ on a 7B student — distributional match is very good, and QwQ's capability is not the binding constraint for 7B-level problems.

The takeaway for ch-20 §4.3: teacher selection is **jointly** about (a) capability ceiling and (b) distributional match to student. Ignoring either axis gives a weaker student than the budget would otherwise support.

---

## Risks flagged (source lines 50–54)

1. **QwQ teacher quality is the bottleneck on AIME25.** Not fixable with more data; the teacher has to improve.
2. **Narrow eval** — reported on math + code only; no safety or instruction-following eval.
3. **Reformatting brittleness** — GPT-4o-mini rewriter occasionally distorts reasoning steps.
4. **$450 is training-only** — excludes teacher inference (free only because they ran QwQ locally).

---

## How ch-20 cites this

Ch-20 §4 positions Sky-T1 as the **open-weights-teacher** point on the reproduction triangle — the case where the teacher license is Apache-2.0 rather than MIT, teacher inference is free, and the student ceiling is set by teacher capability not by data volume. The AIME24/AIME25 gap is the empirical anchor for "teacher ceiling is real" in §4.1. The GPT-4o rewriter appears in §7's licensing discussion as a subtle secondary-teacher risk that even "fully open" pipelines can incur.
