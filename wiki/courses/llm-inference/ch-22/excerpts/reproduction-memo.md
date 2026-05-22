---
chapter: ch-22
course: llm-inference
phase: read
excerpt_of: "Reproduction memo template + good vs bad memo example"
source_url: internal
created_at: "2026-05-21"
---

# Excerpt: Reproduction memo template + worked example

**Source:** distilled from reproduction workflows in the ML-systems community (MLSys, NeurIPS Efficient LLM Inference Workshop)
**Raw-data:** internal synthesis

---

## The template

```markdown
# Reproduction Memo — <method name> on <model name>

**Author:**           ____
**Capstone option:**  PagedAttention / RadixAttention / SpecDec / EAGLE / DistServe
**Paper:**            <citation, arxiv link, year>
**Hardware:**         <GPU model × count, CUDA version, interconnect if multi-GPU>
**Model:**            <HF id + commit hash>
**Workload:**         <dataset, seed, request count, length filter>
**Date:**             ____
**Repo:**             <link to your reproduction code>

## 1. Implementation summary

(One paragraph: what you built. Cite the paper's section/equation numbers
your code maps to. Include the line count and the file layout. If you
deliberately deviated from the paper's recipe — say so and why.)

## 2. Correctness verification

(How you confirmed the method actually does what it claims, BEFORE
benchmarking. For lossless methods: the lossless test. For paged caches:
identical generation under same seed. For RadixAttention: prefix match +
divergence test. For DistServe: equivalent-to-co-located.

Include the actual test code link in your repo. A reviewer should be
able to re-run your correctness verification.)

## 3. Headline number

| Metric                 | Paper number | My number | Δ %  | Within tolerance? |
|------------------------|-------------:|----------:|-----:|:-----------------:|
| <e.g. TPOT speedup>    |        2.50× |     2.10× | -16% |    Yes (±30%)     |
| <secondary, optional>  |       80% accept | 72%   |  -8% |    Yes            |

State your tolerance up front; don't pick it after seeing the number.

## 4. Where my number diverges (or matches) the paper

(REQUIRED — even on a successful reproduction.

If matched: which paper hyperparameter mattered most? Which one were you
tempted to deviate from but ended up keeping?

If not matched: which axis is the gap on?
  - Hardware (e.g., A100 in paper vs H100 in mine)
  - Batch size / context length
  - Hyperparameter (K, block size, chunk size, temperature)
  - Workload (ShareGPT filter, length distribution)
  - Quantization (FP16 in paper vs BF16)

Pick the most-likely axis. Run a SECOND experiment that varies along that
axis to test the hypothesis. Report the result of that experiment here.)

## 5. What the paper didn't tell me

(The most valuable section. Name at least one implementation decision
the paper glossed over. Examples:

  - "PagedAttention paper didn't specify whether to bump the filled-token
    counter at the first layer's KV write or the last layer's — getting
    this wrong causes off-by-one cache reads on every decode step."

  - "EAGLE paper says 'train the draft head' but doesn't specify the loss
    weights for token vs feature targets. The supplementary said 0.1/0.9
    but it's actually in a different paper's appendix.")

## 6. What I would do differently next time

(Concrete: which step took longest? Which step was riskiest? What would
you check earlier? This section is graded on specificity, not length.)

## 7. Optional stretch goal status

(If attempted: PR link, maintainer feedback, what landed and what didn't.
If not attempted: leave blank.)
```

---

## A good memo: an example fragment

The headline + Section 4 of a hypothetical EAGLE reproduction memo:

> **Headline:**
>
> | Metric | Paper number | My number | Δ % | Within tolerance? |
> |---|---:|---:|---:|:---:|
> | TPOT speedup vs Llama-3-8B target only | 3.0× (Llama-2-7B in paper) | **1.6×** | -47% | **No** (±30%) |
> | Per-token acceptance rate | ~80% | 76% | -5% | Yes |
>
> **Section 4 — Where my number diverges:**
>
> The acceptance rate matches the paper within 5 %, so the *draft-quality* aspect of EAGLE reproduces cleanly. The speedup, however, fell short by nearly half. The gap is on the **attention-architecture** axis.
>
> The paper benchmarks on Llama-2-7B (full MHA: 32 query heads, 32 KV heads). My target is Llama-3-8B (GQA-8: 32 query heads, 8 KV heads). The expected speedup formula from Leviathan §3.2 has cost factor `c = (draft_pass_cost / target_pass_cost)`. On Llama-2-7B, target-pass cost is dominated by KV-cache bandwidth — 32 KV heads to read per token. On Llama-3-8B-GQA, KV cost per token is already 4× lower, so the target pass is *already* fast, leaving less headroom for speculation to amortise.
>
> I ran a confirmatory experiment: same EAGLE implementation, target = Llama-2-7B (downloaded with `meta-llama/Llama-2-7b-chat-hf`). Speedup measured: 2.7×, within 10 % of paper's 3.0×. This confirms the architecture-axis hypothesis: EAGLE's headline number is conditioned on full-MHA targets and does NOT directly transfer to GQA targets.
>
> Recommendation: EAGLE-style methods should report speedup on multiple attention geometries. A GQA-specific draft architecture (e.g., a draft head that better predicts the K-replication pattern) might recover the gap, but that's outside the capstone scope.

This is a good memo because it:
- States the headline cleanly with explicit tolerance.
- Hypothesises a *specific* axis for the gap, not vague handwaving.
- Runs the *confirmatory* experiment that tests the hypothesis.
- Names a publishable follow-up direction.

---

## A bad memo: same situation, different write-up

> **Headline:** I got 1.6× speedup instead of 3×. EAGLE didn't work as well as expected.
>
> **Section 4 — Where my number diverges:** I'm not sure why it was slower. Maybe my hardware was different, or my draft head wasn't trained as well. The paper used Llama-2 and I used Llama-3, so maybe that matters.

This is a bad memo because it:
- Doesn't quantify the gap clearly.
- Lists possible causes without testing any of them.
- Doesn't run a confirmatory experiment.
- Has nothing actionable.

---

## Rubric for grading your own memo

Score each section out of 3:

| Section | 3 (full) | 2 (partial) | 1 (weak) | 0 (missing) |
|---------|---------|------------|---------|-------------|
| 1. Summary | Explicit paper→code mapping, file layout, line count | Code reference but no equation mapping | One sentence | — |
| 2. Correctness | Test code link + assertion that passes | Test described but not linked | "I checked it worked" | — |
| 3. Headline | Number + tolerance + secondary metric | Number only | Number with no tolerance | — |
| 4. Where diverges | Specific hypothesis + confirmatory experiment | Hypothesis without test | Vague mention | — |
| 5. Paper didn't tell | Specific implementation decision named | General observation | "It was hard" | — |
| 6. Different next time | Concrete change | Vague reflection | "Start earlier" | — |

A passing capstone memo scores ≥ 14/18. A publishable-quality memo scores ≥ 16/18 and has a clean Section 4.

---

## Connections

- [[ch-21]] — the lab memo template is a simpler precursor to this one.
- [[excerpts/debugging-tree]] — how to *generate* the content for Sections 4 and 5.
- [[excerpts/method-comparison]] — pick the method whose memo you most want to write.
