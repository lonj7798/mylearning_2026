---
chapter: ch-20
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/openr1.md
source_url: https://github.com/huggingface/open-r1
created_at: "2026-04-23"
---

# Excerpt: Open-R1 — HuggingFace's 220K-problem math corpus and Math-Verify

**Source library:** `wiki/raw-data/llm-training/papers/openr1.md`
**Release:** HuggingFace Open-R1 team (Tunstall, Beeching, Lambert, Ben Allal, Penedo et al.), 2025.

---

## Why this source anchors ch-20

Open-R1 is the **high-volume** point on the Stratos–Open-R1–Sky-T1 triangle. Where Stratos (17K) argues "curation beats scale," Open-R1 (440K) argues "scale + a reproducible verifier beats opacity." The two positions are not contradictory; they answer different questions. Stratos answers *what's the minimum to match R1-Distill?*. Open-R1 answers *what happens when you publish a reproducible pipeline at the volume where ablation studies actually work?*.

From source line 7:

> **Core Insight:** A fully-open community replication of R1 is feasible by combining (a) mass R1-distilled SFT on a 220K-problem math+code corpus (OpenR1-Math-220k) with (b) GRPO/RLVR on verifiable-reward problems — the HF team published both recipe and data so labs can train R1-quality reasoners without DeepSeek internals.

---

## The three-stage program (source lines 14–22)

```
Stage 1  : Replicate R1-Distill
           → OpenR1-Math-220k (the headline artifact ch-20 cites)
Stage 2  : Replicate R1-Zero (pure-RL from base via GRPO with verifiable rewards)
Stage 3  : Replicate full R1 multi-stage (SFT + RL + SFT + RL)
```

Ch-20 is about stage 1 only. Stages 2–3 belong to ch-24 (RLVR at scale).

---

## The corpus — 220K × 2 traces (source lines 23–32)

```
Seed        : 220K math problems from
                NuminaMath (cn_k12 / olympiads / aops_forum / amc_aime / orca_math)
                + supplementary AIME/AMC archive
Teacher     : DeepSeek-R1 (HuggingFace-hosted + API mix)
Sampling    : T = 0.6; 2 traces per problem (some problems sampled up to 8×)
Total       : 220K × 2 ≈ 440K samples

Filter      :
  - Extract \boxed{} answer with regex
  - Compare to gold using Math-Verify (open-source SymPy-based
    symbolic-equivalence checker)
  - Reject traces with format violations (missing </think>, no \boxed{})
Reject rate : ~20%

Trace length: median ~5K tokens, mean ~7K;
              ~10% of traces exceed 15K tokens
Cost        : ~$10K inference (HF H100 cluster + API mix)
License     : Apache-2.0
```

Two details are worth pulling out because they are the load-bearing engineering decisions.

---

## 1. Math-Verify is the recipe-reproducibility primitive

Source line 30:

> Compare to gold using **Math-Verify** (open-source SymPy-based symbolic equivalence checker) — the explicit tool that makes Open-R1's pipeline reproducible.

Stratos's SymPy-based filter is internal engineering. Open-R1 made the filter **a standalone open-source package**. This is non-trivial: it means any third party replicating the corpus on a new problem pool gets the *same* correctness decisions as the original Open-R1 team. Without a standardized verifier, "filtered by SymPy" means different things in different pipelines (different edge-case handling for `\frac` vs `/`, different tolerance for numeric answers, different handling of multi-valued answers).

Math-Verify's limits, from source line 37:

> Math-Verify (SymPy-backed) → reliable on algebraic/numeric equivalence but limited on geometry/proofs.

This is a structural weakness of every math-reasoning corpus built on symbolic verification. Geometry problems often have correct answers that SymPy cannot canonicalize (spatial descriptions, proof structures). Open-R1 simply drops these. A downstream student trained on Open-R1-Math-220K is silently weaker on geometry.

---

## 2. The "wrong-question-correctly" failure

Source line 50:

> **Verifier limits:** Math-Verify does not catch semantic drift (e.g., answering the wrong question correctly).

This is the critical failure mode ch-20 §5.5 elaborates on. Math-Verify compares the *final answer* to the *gold answer*. If R1 misreads the problem, solves a different but related problem, and the final numeric answer happens to coincide with the gold (common with small integer answers, common with problems that have multiple candidate interpretations), the trace passes the filter. The student learns a reasoning path that looks valid on the surface but is solving a different problem.

This is not fixable by outcome-based filtering. Process rewards (ch-25) — step-level verification — are the only known defense. Open-R1 flags it explicitly; no open corpus claims to address it.

---

## 3. Trace multiplicity — 2 traces per problem

Source line 44:

> Ablation: 2 traces vs 1 trace per problem → marginal gain; most signal comes from R1's quality not from trace diversity per problem.

Open-R1 samples 2 traces per problem (some up to 8×). The ablation: going from 1 to 2 traces adds measurable but marginal gain; the returns diminish quickly. OpenThoughts (ch-20 §6) later argues **multi-sampling is the easiest way to ≥16× expand a source** — but Open-R1's result qualifies that: multi-sampling *within* a single teacher has small returns; multi-sampling *across* teachers or *with temperature diversity* might have more.

The operational lesson: if the teacher is strong and deterministic-ish (T = 0.6), 1 trace per problem gets you most of the signal. Multi-sample only if your problem pool is small and you need to expand it.

---

## 4. The GRPO follow-up (source lines 42–43)

```
OpenR1-Qwen-7B SFT on OpenR1-Math-220k:
  MATH      ~80%
  AIME24    ~40%  (close to DeepSeek-R1-Distill-Qwen-7B baselines)

GRPO stage (separate from SFT) adds +3–5 AIME points.
  Uses binary 0/1 reward from Math-Verify
  Group-relative advantage with KL penalty
  Trained on a 40K-problem subset
```

Two things this shows that the Stratos-only view doesn't:

- **SFT-on-distill gets you 90% of the gain; downstream GRPO on verifiable rewards adds a few more points.** The distill corpus is the foundation; the RL stage is the polish. This is the ch-20/ch-24 handoff.
- **The verifier used for SFT filtering and the verifier used for GRPO reward are the same primitive (Math-Verify).** This is the cleanest demonstration that verifier engineering compounds — build the verifier once, use it in both the data pipeline and the RL pipeline.

---

## Risks flagged (source lines 46–50)

1. **R1 dependency** — gated on continued R1 weights / API availability; change in license retroactively affects corpus's legal status.
2. **Math-heavy** — 220K dataset is math-only; code / science / agentic tracks are in progress.
3. **License caveats** — R1 weights are MIT but the output distribution carries "trained by R1" attribution expectations.
4. **Semantic-drift verifier limit** — §5.5 failure, unaddressable with outcome filters.

---

## How ch-20 cites this

Ch-20 §4's comparison table positions Open-R1 as the **high-volume, open-verifier** reference. The Math-Verify package is the specific example of "recipe-as-code, not recipe-as-prose" — the thing that lets indie teams reproduce HF's results rather than just reading about them. The semantic-drift failure in §5.5 is cited directly from this source.
