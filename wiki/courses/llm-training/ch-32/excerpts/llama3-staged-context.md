---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/long-context-llama3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 long-context - the six-stage schedule ch-32 uses

**Source library:** `wiki/raw-data/llm-training/papers/long-context-llama3.md`
**Artifact:** 8K -> 128K extension across ~800B tokens with RoPE base 10K -> 500K

---

## Why this source anchors ch-32

Llama 3's long-context subsection is the most cited "production 128K recipe" disclosure. It makes two things concrete that ch-32 otherwise would have to describe in abstract terms: (i) the staged RoPE-base rescale schedule, and (ii) the short-SFT-vs-long-SFT ratio that prevents short-context regression. Both are load-bearing for ch-32's claim that long-context is a decomposable three-job stage, not a single knob.

The schedule is also an existence proof for how much compute "long-context extension" actually costs on a frontier model: ~800B tokens of continued pretraining is a nontrivial fraction of the 15.6T pretrain itself. This is why ch-32's stage-allocation table lists long-context as proportionally large for Llama 3 (~5% of pretrain tokens), contrasted with OLMo 3's much smaller Longmino (0.8%).

---

## The six-stage schedule ch-32 transcribes

From the source (lines 25-32):

- **Stage A**: 8K -> 16K, ~100B tokens.
- **Stage B**: 16K -> 32K, ~100B tokens.
- **Stage C**: 32K -> 64K, ~150B tokens.
- **Stage D**: 64K -> 128K, ~200B tokens.
- (Additional intermediate stabilization stages.)
- Total: ~800B tokens across all stages.

Ch-32 uses this table verbatim. The design rule implicit in the schedule - **each stage doubles the context window and runs long enough for the new positions to stabilize before the next expansion** - is how the staged approach keeps RoPE aliasing bounded at every step.

---

## The RoPE rescale ch-32 cites

From the source (lines 38-41):

- Base rescaled from **10K to 500K** for the final 128K model.
- Scaling done progressively: at each stage, the RoPE base is adjusted to match the new context window.

Ch-32 quotes this as the canonical example of Job 1 (position-encoding extension) done in a staged rather than single-shot manner. The alternative - rescale once to 500K and train at 128K directly - is attested to be unstable; the staged schedule exists because intermediate stabilization matters empirically.

---

## The 0.1% long-SFT rule ch-32 cites

From the source (lines 43-46, 56-58):

- A small fraction (~0.1%) of SFT samples are long-context.
- Generation uses a larger Llama 3 model as teacher on full documents.
- Raising the long-SFT fraction above 1% costs ~1 MMLU point.

Ch-32 uses this to anchor its Job 3 claim: long-context SFT must exist, but kept tiny. The binding constraint is short-context regression, not long-context gain. This is the rule that separates "model with a long window" from "model that uses its long window without losing its short-window chat quality."

---

## The claimed-vs-effective gap ch-32 surfaces

From the source (lines 53, 64-65):

- Llama-3.1-70B: NIAH 128K ~99%; RULER 128K ~75% (effective context ~64K).
- Llama-3.1-405B: NIAH 128K ~99%; effective RULER context ~96K.

Ch-32 quotes these numbers to make concrete that "128K context" without specifying NIAH / RULER / BABILong is underspecified. The same staged schedule produces different effective contexts on different-sized models - a 64K effective context at 70B is not a training bug, it is expected. Meta's own paper acknowledges the gap.

---

## What ch-32 does not carry from the source

- Llama 3's specific per-stage data mixes (book / code / web / academic ratios) are not itemized in the raw-data summary; ch-32 references them qualitatively.
- RoPE base values for intermediate stages (8K -> 16K, 16K -> 32K, etc.) are not published; only the final 500K is attested.
- Llama-3.2 and 3.3 inherit this recipe with refinements; ch-32 does not cover those refinements.

---

## Connections

- **ch-28** - long-context modality chapter; this stage-level synthesis is the production counterpart to ch-28's research-paper-level survey.
- **[[prolong]]** - alternative smaller-budget recipe (20B tokens vs 800B) with much heavier data curation.
- **[[longalign]]** - the SFT side (Job 3) of long-context as a distinct stage.
- **[[ruler]]** - the evaluation suite that reveals the claimed-vs-effective gap.
