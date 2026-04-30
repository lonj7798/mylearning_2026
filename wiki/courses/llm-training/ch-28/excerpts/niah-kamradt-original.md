---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Greg Kamradt — "LLMTest_NeedleInAHaystack" (original NIAH blog + GitHub, Nov 2023)
source_url: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
created_at: "2026-04-23"
---

# Excerpt: The Kamradt NIAH — what went viral, and why it over-reports effective context

**Source:** `wiki/raw-data/llm-training/papers/needle-in-haystack-data.md`
**Author:** Greg Kamradt (independent researcher), November 2023
**URL:** https://github.com/gkamradt/LLMTest_NeedleInAHaystack

---

## Bibliographic header

> *"The simplest useful long-context stress test is to hide a single fact ('needle') inside many pages of filler essay text and ask the model to retrieve it; varying needle-depth and haystack-length gives a two-dimensional capability map — this one-liner eval, started by Greg Kamradt in Nov 2023, became the de-facto community standard for long-context claims and the ancestor of RULER / BABILong."*

Not a peer-reviewed paper. A blog post and a GitHub repo. That's part of the point — it went viral *because* it was accessible.

---

## The original setup

From the raw-data notes:

> *"Filler: Paul Graham essays (publicly available, mostly absent from post-2021 training cutoffs). Needle: a single sentence stating a specific fact (the 'best thing in SF' sentence). Injection: inserted at programmatic depths (e.g., 0%, 10%, 20%, …, 100% of filler). Context lengths tested: 1K, 4K, 8K, 16K, 32K, 64K, 128K, … up to the model's claimed max. Evaluation prompt: 'What is the best thing to do in San Francisco?' — scored against gold answer."*

The literal sentence Kamradt inserted was:

> *"The best thing to do in San Francisco is eat a sandwich at Dolores Park on a sunny day."*

Specific, unambiguous, contextually plausible, not present in Paul Graham essays. That combination is the key to the test working — if the sentence sounded like Paul Graham, the model couldn't distinguish needle from haystack; if it sounded *wildly* out-of-place, the retrieval would be too easy.

---

## Why the 2D heatmap went viral

Output shape: a grid where:

- x-axis = context length (1K → max)
- y-axis = needle depth (0% → 100%)
- cell colour = retrieval accuracy

The heatmap is *immediately legible*. "Red zone at 50% depth, 32K context" communicates a failure pattern in one glance. No statistics background, no p-values, no tables to decode. Every long-context release since has included one — Claude, GPT-4, Llama 3, Qwen 2.5, Gemini — because the heatmap is the fastest way to argue "our model does not have a positional blind spot."

The typical failure pattern that shows up: a **band of low accuracy around mid-depth at the longest context**, because models trained with attention-sink tokens at the start and recency-bias at the end fail hardest on middle-of-context retrieval ("lost in the middle").

---

## The multi-needle and RULER extensions

From the raw-data:

> *"Multi-key: N distinct needles, each with its own key (e.g., 'the best thing in A is X', 'the best thing in B is Y'); retrieve all. Multi-value: one key with multiple values across the haystack. Multi-query: multiple independent retrieval queries for one haystack. Anthropic multi-needle: 1–8 needles at varied depths, all retrieved in one completion."*

These post-2023 variants exist because single-needle NIAH *saturated* — in 2025, most frontier models score >95% on single-needle at claimed context. The discrimination moved to multi-needle, then to [[excerpts/ruler-task-family]]'s 13-task suite, then to [[excerpts/babilong-pg19-embed]]'s reasoning-in-a-haystack.

Multi-needle 8-needle discrimination in 2025 from the raw-data:

- Claude-3.5: ~90%
- Llama-3.1-70B: ~70%
- Weaker open models: <50%

This is the current "NIAH" discriminator — "NIAH" in the community has quietly become shorthand for "multi-needle NIAH" in practice.

---

## Why NIAH alone is misleading

> *"NIAH is not a proxy for real long-context reasoning — models can pass NIAH at 128K while failing multi-hop reasoning at 32K."*

Failure modes NIAH cannot detect:

1. **Distractor sensitivity.** NIAH's filler is neutral Paul Graham prose. Add distractor facts and accuracy drops. MK-NIAH measures this.
2. **High-recall output.** NIAH asks for one fact. Ask for all 8 and the model may return 5. MV-NIAH / MQ-NIAH measure this.
3. **Cross-span reasoning.** NIAH needs no reasoning. BABILong's "two supporting facts" / "counting" / "path finding" measure this.
4. **Aggregation.** NIAH is sparse lookup. CWE / FWE measure aggregation.

Any long-context claim that cites NIAH without citing at least one of the above is *underspecified*. This is ch-28's refrain.

---

## NIAH as a training signal

> *"Data recipe that uses NIAH as training signal: [[longalign]], [[prolong]], [[qwen-long-context-synth]]."*

The reverse move. ProLong and Qwen don't just *evaluate* on NIAH — they *train* on synthesised NIAH samples. The training rationale: NIAH-style retrieval is a *learnable skill*, and the base model after CPT usually has the capacity for it but not the behavioural habit. Injecting 1–8-needle samples into the SFT mix teaches the model to actually output the retrieved value rather than paraphrase or refuse.

This is why the NIAH heatmap is high for well-trained models *conditional on NIAH training*, and why NIAH scores should be interpreted with knowledge of whether NIAH was in the SFT mix. A 99% NIAH score with NIAH-in-SFT is less informative than a 90% score without it.

---

## Contamination gotcha

> *"Paul Graham contamination: many models have PG essays in training — canary filler choice matters."*

PG essays are in Common Crawl. Many pretraining datasets include them. A model that memorised PG essays might recognise the needle sentence as *not-PG* faster than a model that only does distribution-based detection — giving a misleading NIAH gain to the more memorising model. Best practice: substitute the filler with a corpus the model hasn't seen (recent news post training cutoff, a private corpus, or synthetic filler).

---

## Connections

- Chapter synthesis: [[ch-28]]
- Successor task families: [[excerpts/ruler-task-family]], [[excerpts/babilong-pg19-embed]]
- Used in every 2024+ training recipe's eval and sometimes its SFT: see ch-28 §1
- Natural-task complement: LongBench (see ch-28 §1)
