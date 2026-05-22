---
chapter: ch-21
course: model-quantization
phase: read
excerpt_of: "Memo template for the ch-21 lab deliverable"
created_at: "2026-05-21"
---

# Excerpt: Pareto Memo Template

**Use as:** the literal scaffolding for `lab-memo.md` in your `quant-lab/` repo.

---

## Memo template (copy this into `lab-memo.md`)

```markdown
# Quant Lab Memo — <date>

**Model:** <e.g., meta-llama/Llama-3-8B>
**Hardware:** <e.g., 1 × H100 80 GB>
**Calibration:** 128 seq × 2048 tok from allenai/c4 (val split, seed=42)
**Eval seeds:** [42, 43, 44, 45, 46]
**Code:** https://github.com/<you>/quant-lab @ <commit-sha>

## Pareto table

| Method | Bits/wt | PPL Wikitext ± CI | MMLU ± CI | GSM8K | HumanEval | IFEval | TruthfulQA | Latency bs=1 (ms/tok) | Latency bs=16 | Peak VRAM (GB) | Quant time |
|--------|---------|--------------------|-----------|-------|-----------|--------|------------|----------------------|---------------|---------------|------------|
| FP16 baseline | 16.0 | x ± y | x ± y | x | x | x | x | x | x | x | — |
| AutoGPTQ-W4 g128 | 4.13 | x ± y | x ± y | x | x | x | x | x | x | x | ~30 min |
| AutoAWQ-W4 g128 | 4.13 | x ± y | x ± y | x | x | x | x | x | x | x | ~15 min |
| QLoRA-NF4 r=64 | 4.13 | x ± y | x ± y | x | x | x | x | x | x | x | ~30 min |
| bnb-INT8 | 8.5 | x ± y | x ± y | x | x | x | x | x | x | x | <2 min |

CIs are 5-seed std deviations on the same checkpoint.

## Pareto chart

(Embed: MMLU vs decode latency bs=1, one marker per method.)

## Required ablation: <group_size | alpha | lora_rank>

| <knob> | PPL Wikitext | MMLU | <one additional task> | <if relevant: storage / time> |
|--------|---------------|------|----------------------|-----------------------------|

**Bowl minimum / knee:** <value>
**Surprise (if any):** <one paragraph naming a specific phenomenon. If none, state "ablation reproduced expected curve.">

## One specific failure mode

<One paragraph. Name a *specific* pathology you observed — not "AWQ was faster than expected."
Examples of specific failure modes:
 - "GPTQ-W4 g=128 lost 4 pp on IFEval-strict where AWQ-W4 g=128 lost 1.5 pp; on inspection,
   GPTQ's per-column Hessian update systematically over-rounded the format-token paths."
 - "bnb-INT8 had identical PPL to FP16 (within 0.02) but failed HumanEval-Python at 35% vs FP16's 50%;
   trace: outlier-FP16 columns were not preserved through vLLM's quantization loader, so the LLM.int8
   recipe was effectively pure-INT8 at serve time."
 - "QLoRA-NF4 r=64 won MMLU by 1.5 pp over the calibration-only methods but lost 8 pp on TruthfulQA
   relative to FP16; the Alpaca SFT data nudged the model toward confidently-wrong answers."
>

## Pareto-frontier recommendations

| Workload class | Method | Rationale |
|----------------|--------|-----------|
| Latency-bound batch-1 chat | <method> | <one line> |
| Memory-bound long-context | <method> | <one line> |
| Fine-tune-required domain adapt | <method> | <one line> |
| OOD-robust serving | <method> | <one line> |

## Notes

- <One bullet for any non-default config choice and why.>
- <One bullet for any deviation from the recipe (e.g., "substituted Llama-3.1-8B for Llama-3-8B
  because of license restrictions in our test environment").>

```

---

## What makes a memo good

The memo passes the **peer-reproduces-from-repo** test:

1. Another engineer with the same hardware can clone the repo and run `make all` to recreate the table.
2. The peer arrives at numbers within `± CI` of the memo's table.
3. The peer agrees that the "specific failure mode" is real after running the eval themselves.
4. The Pareto recommendations are *actionable* — each row tells the peer which method to use next time.

A memo that fails this test is either (a) missing reproducibility details (calibration set, seeds, commit SHA), (b) reporting averages without CI (so the peer can't tell if a 0.1 PPL difference is signal), or (c) recommending methods without rationale (so the peer can't transfer the recommendation to a new workload).

---

## What to leave out

- **Library implementation comparisons.** "AutoGPTQ's CUDA kernel is 1.3× slower than Marlin" is a kernel-engineering finding, not a method finding. Note it briefly; don't dwell.
- **Generic statements.** "AWQ is faster than GPTQ" without numbers is folk-wisdom; if the numbers don't show it on your hardware, the statement is false on your hardware. Report what *you* measured.
- **Speculation about why a method behaves differently.** Either you ran the diagnostic that explains it (good — report the diagnostic) or you didn't (don't speculate — flag it as an open question for the next iteration).

The memo is a *primary source* for the peer reading it. Keep it tight.

---

## Connections

- [[ch-21]] §the-Pareto-deliverable — chapter section.
- [[ch-20]] §6 — the evaluation harness this memo summarises.
- [[ch-21]] §reflection-prompts — the questions the memo's "specific failure mode" + Pareto recommendations are answering.
