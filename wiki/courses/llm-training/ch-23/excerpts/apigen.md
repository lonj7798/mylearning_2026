---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/apigen.md
source_url: https://arxiv.org/abs/2406.18518
created_at: "2026-04-23"
---

# Excerpt: APIGen — the 3-layer verifier as an existence proof that rule-based gates work at scale

**Source library:** `wiki/raw-data/llm-training/papers/apigen.md`
**Authors:** Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh Murthy, Liangwei Yang, Silvio Savarese, Juan Carlos Niebles, Huan Wang, Shelby Heinecke, Caiming Xiong (Salesforce AI Research)
**Venue:** NeurIPS 2024; arXiv 2406.18518

---

## Why this source anchors ch-23

Ch-23 §6 Template B (3-layer rule-heavy verification) is APIGen's pipeline, abstracted. The paper is the load-bearing evidence that **rule-based verifiers scale**: 60K verified samples are sufficient to train xLAM-7B to #1 on BFCL among <13B models. The ablation table — removing any of the three layers costs 6–18% BFCL-V1 — is the cleanest published evidence that gate layers stack multiplicatively. Ch-23 §5 comparison table and §6 template both lift directly from this source.

---

## The 3-layer verification stack

```
# apigen.md, lines 29-33
Format check: JSON must parse, fields must match function schema,
              types enforced.
Execution check: run the call(s) against the reference implementations;
                 must not raise.
Semantic check: LLM-as-judge (GPT-4) is shown the query + the call +
                the execution result, and must answer "yes" to "does
                the call correctly fulfill the query?"
```

The three layers catch three distinct failure modes:

- **Format violations** — schema mismatches, missing required params, wrong types, invalid enum values. These are structural errors the generator cannot self-detect (the generator has no schema validator in its head; it imitates JSON).
- **Execution failures** — runtime errors, timeouts, resource violations. These are dynamic errors invisible to the generator (which lacks a Python interpreter). Roughly 10–15% of format-passing generations fail here.
- **Semantic errors** — the call runs and returns something, but the something is not what the query asked for. Wrong unit in a conversion function. Wrong sort order. Right function, wrong arguments. ~10% of execution-passing calls fail this.

Each layer requires a different kind of knowledge (structural, operational, semantic) and a different implementation cost (microseconds, seconds, teacher-model call). The cost order is roughly `format << execution << semantic`, which is also the order the layers should run — fastest filter first, so expensive semantic checks only see pre-filtered candidates.

---

## The ablation — all three layers are load-bearing

```
# apigen.md, lines 53-54
Removing semantic check → –6% BFCL-V1;
removing execution → –11%;
removing format → –18%. All three layers are load-bearing.
```

This is the most important number in the paper for ch-23's argument. The ablation is strong because:

1. **Each layer is removed independently** — not cumulatively. So the numbers measure each layer's unique contribution.
2. **The numbers are not small.** 18% is a transformative difference in BFCL performance; the layers are not ornamental.
3. **The ordering matches intuition.** Format errors are the most common failure type, so removing that layer admits the most noise. Execution catches a narrower slice. Semantic catches the narrowest but most subtle.

The multiplicative argument in ch-23 §6 derives from this table. If format rejects 20% of raw generations, execution rejects 15% of format-passing, semantic rejects 10% of execution-passing, the overall acceptance is `0.80 × 0.85 × 0.90 ≈ 0.61`. Remove semantic and acceptance jumps to `0.80 × 0.85 = 0.68`, with the 7% newly-admitted samples being semantically-wrong — the ones the ablation shows cost 6% BFCL. The math fits.

---

## The dataset — 60K samples, #1 BFCL <13B

```
# apigen.md, lines 49-51
xLAM-7B (Mistral-7B base fine-tuned on APIGen-60k): 88.24% BFCL-V1
overall, #1 among <13B.
xLAM-8x7B (Mixtral base): 88.9% BFCL-V1, close to GPT-4.
```

60K samples is remarkably small for a production SOTA. This is the sample-efficiency payoff of the gate: because every sample is verified (the noise is stripped out), the SFT signal is concentrated, and 60K clean samples beats 600K noisy samples. Contrast with ToolLLaMA (trained on ~120K unverified ToolBench trajectories): xLAM-7B on 60K verified outperforms ToolLLaMA on 2× the data.

The lesson for ch-23: **verification is a compute trade**, not just a safety measure. You spend verifier compute upfront to reduce training-data requirements downstream. At APIGen's ratios, the 3-layer filter rejects ~40% of raw generations; the 60K kept is equivalent to ~100K raw. If you trained on the 100K raw, you'd get worse results *and* spend more SFT compute. The gate pays twice.

---

## The API registry — why executability is possible at all

```
# apigen.md, lines 24-26
Start from ToolBench's 16K APIs but keep only the 3,673 APIs with
executable reference implementations.
```

The executability requirement is the bottleneck. APIGen can do Layer 2 (execution) only because someone (Salesforce) curated 3,673 APIs with working Python implementations or endpoints. Scaling this to 30K APIs would require 30K implementations, which is a major curation investment.

Two implications:

1. **Execution-based verification has a discovery cost.** Before you can gate, you need a verifier. For code: you need unit tests. For math: you need answer keys. For tool calls: you need reference implementations. The gate is cheap at runtime, expensive at setup.
2. **The scope of a verifier-gated modality is bounded by verifier availability.** APIGen covers 3,673 APIs, not 30K, because that's what could be made executable. Other modalities have similar implicit scopes.

Ch-23 §6 Template B's real pre-condition: you need a ground-truth signal at Layer 2. If you don't have one, you fall back to Template A (RM-as-judge), which is weaker but more broadly applicable.

---

## Hallucination rate — the downstream safety number

```
# apigen.md, line 48
The 3-layer filter rejects ~40% of raw generations; post-filter,
hallucination rate on BFCL-V1 is <3% for xLAM-7B (vs ~15%
for ToolLLaMA).
```

The <3% hallucination rate on BFCL-V1 is the end-to-end validation that the gate works. Hallucination in function-calling = the model invents a function that doesn't exist, or invents an argument the schema disallows. Both failure modes are ruled out by the format layer at training time; the model *learns* not to produce them because the training set never contains them.

This is how gate discipline transfers from training to inference. If your training set contains 15% hallucinated calls (ToolLLaMA), the model will hallucinate at inference. If your training set is 0% hallucinated (APIGen post-gate), the model's learned behavior is hallucination-free within distribution. The gate's value compounds through the training signal.

---

## The semantic-layer blind spot

```
# apigen.md, lines 57-59
LLM-judge blind spots: GPT-4 judge occasionally accepts
semantically-close-but-wrong calls (e.g. wrong unit passed to a
conversion function).
```

The semantic layer is the least reliable of the three. Rule-based Layers 1 and 2 are deterministic; Layer 3 is a model call with all the attendant variance. The paper flags this honestly: GPT-4 judge misses ~5% of true semantic errors. This is why the three layers stack — if semantic were perfect, you'd skip the others. It isn't, so you don't.

For ch-23 §6 Template B, the operational guideline is: **use rule-based verification for as much of the gate as possible; use LLM-as-judge only for the semantic layer; set the LLM-as-judge threshold conservatively** (require "Yes" with a reasoning step that a second LLM can optionally audit). The multi-step LLM-judge pattern (with critique) is explored further in [[excerpts/nemotron-4-synthetic]].

---

## Cost accounting — gate ≠ free

```
# apigen.md, line 40
~$8K in teacher API + ~10K GPU-hours for execution sandbox.
```

The gate cost: $8K for teacher API calls (generation + semantic judge) + 10K GPU-hours for the execution sandbox. For the resulting 60K verified samples, that's ~$0.13/sample + ~10 minutes GPU-time/sample. At modern cluster prices, roughly $0.50/sample all-in.

Compare to the cost of training xLAM-7B on the produced dataset (8 × A100 for ~24 hours, so ~$300): the gate cost (~$30K for 60K samples) is ~100× the training cost. This is the economic reality of verified synthetic data — the verifier dominates. It pays off because the alternative (training on unverified data and accepting worse downstream quality) has a much higher implicit cost (evaluation, iteration, reputation, deployment fixes).

---

## Why single-turn only — and what [[apigen-mt]] adds

```
# apigen.md, line 60
No multi-turn: APIGen generates single-turn function calls only.
Addressed in [[apigen-mt]].
```

Single-turn is the easier setting because the semantic judge only needs to evaluate one call in isolation. Multi-turn (user query → tool call → tool result → follow-up → …) requires the judge to evaluate a whole trajectory, which is a harder judgment task with more failure modes.

[[apigen-mt]] extends the 3-layer gate to trajectories: Layer 1 becomes per-step format; Layer 2 becomes per-step execution + trajectory-level state consistency; Layer 3 becomes trajectory-level semantic ("did this trajectory accomplish the user's goal?"). The same anti-collapse principle — gate everything, always — applies.

---

## Connections

- [[excerpts/model-collapse]] — APIGen is an existence proof that pure-generated synthetic (no human seeds) can scale if gated.
- [[excerpts/strong-model-collapse]] — 3-layer gate shrinks `σ_synth²` close to zero for function-calling.
- [[excerpts/faithful-synth-eval]] — Layers 1-3 instantiate Axis 2 (external verification) at three strengths.
- [[excerpts/nemotron-4-synthetic]] — strong-judge alternative when rule-based Layer 2 is infeasible.
- [[excerpts/synthetic-data-scaling-laws]] — APIGen's 60K verified ≈ rephrased synthetic behavior (scales cleanly).
- [[ch-23]] — §5 table and §6 Template B both derived from this paper.
