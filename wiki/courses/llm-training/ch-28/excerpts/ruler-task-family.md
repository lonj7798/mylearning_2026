---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Hsieh et al. — "RULER: What's the Real Context Size of Your Long-Context Language Models?"
source_url: https://arxiv.org/abs/2404.06654
created_at: "2026-04-23"
---

# Excerpt: RULER — the 13-task synthetic generator and effective-context definition

**Source:** `wiki/raw-data/llm-training/papers/ruler.md`
**Paper:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (NVIDIA), 2024
**arXiv:** https://arxiv.org/abs/2404.06654

---

## Bibliographic header

> *"RULER is valuable less as a leaderboard and more as a parameterized synthetic-task generator that separates context length from task complexity, exposing long-context failure modes that simple needle-in-a-haystack tests miss."*

The framing matters: RULER is a *generator*, not a fixed corpus. That is why it lives in a synthesis chapter — the durable contribution is the generation protocol, not the one-time leaderboard snapshot.

---

## The 13 task settings

> *"13 representative task settings selected from a larger configuration space after a task-correlation study, so the benchmark covers distinct failure modes instead of redundant variants."*

Reproduced from the raw-data notes — these are the exact configurations in the paper's large-scale evaluation:

1. `S-NIAH`: word → number, repeated-noise haystack (passkey retrieval).
2. `S-NIAH`: word → number, essay haystack (vanilla NIAH).
3. `S-NIAH`: word → UUID, essay haystack.
4. `MK-NIAH`: 4 distractor keys, word → number, essay haystack.
5. `MK-NIAH`: full-haystack distractor keys, word → number (line-retrieval-like).
6. `MK-NIAH`: full-haystack distractor keys, UUID → UUID (KV-retrieval-like).
7. `MV-NIAH`: 4 values for one key.
8. `MQ-NIAH`: 4 queried keys.
9. `VT`: 1 chain, 4 hops.
10. `CWE`: 10 common words × 30 repetitions, uncommon × 3.
11. `FWE`: Zeta α = 2.0, return top 3.
12. `QA`: SQuAD long-context adaptation.
13. `QA`: HotpotQA long-context adaptation.

The task-correlation study is the *choice* step: labs were already producing dozens of NIAH-style variants in 2023–early-2024, many of which measured the same skill with different surface forms. RULER's contribution is the de-duplication — pick 13 variants whose scores differ from each other on real models.

---

## Orthogonal knobs — why RULER is a generator

> *"The key design goal is to hold the evaluation domain narrow and controlled so that input length and task complexity can be varied independently."*

The generator factorises the test space:

- **Context length:** 4K, 8K, 16K, 32K, 64K, 128K main; 200K / 256K in analysis.
- **Needle type:** word / 7-digit number / 32-digit UUID — same retrieval shape, different difficulty on span-copy.
- **Haystack type:** repeated-noise sentences *or* natural prose (Paul Graham essays).
- **Distractor density:** MK-NIAH scales from 4 distractors to full-haystack distractors.
- **Output cardinality:** MV/MQ convert retrieval from single-span to multi-item.
- **Tracing difficulty:** VT varies chains × hops.
- **Aggregation difficulty:** CWE's common/uncommon ratio; FWE's Zeta α.

That factorisation is what lets you ask "did this model break because of length or because of distractor density?" — which is the diagnostic question every long-context release report should answer.

---

## The effective-context-size metric

> *"Effective context size is defined as the maximum length whose average score stays above the Llama2-7B @ 4K baseline of 85.6."*

This is the single most cited RULER number. The metric is:

```
effective_ctx(model) = max { L : score(model, L) ≥ 85.6 }
```

Llama2-7B at 4K scores 85.6 on the 13-task average. Any model that scores above 85.6 at length L is, by this definition, "as good as Llama2-7B-4K" at length L. The metric produces numbers like **Llama-3.1-70B: claimed 128K, effective ~64K** — the gap that launched a thousand recipe papers.

Two weighted averages are also reported — `wAvg. (inc)` and `wAvg. (dec)` — with linear weights that emphasise either long-context or short-context performance, to catch models that over-fit either end.

---

## Evaluation protocol

- **500 examples per task per length.** High enough to kill single-example noise.
- **Native chat template** — every model is wrapped in its own chat format.
- **Answer prefix appended** — models respond directly instead of refusing or adding preamble. Without this, a non-trivial fraction of weak models fail at the instruction-following layer rather than the retrieval layer, which contaminates the measurement.
- **Recall-based accuracy** — matches the target outputs rather than exact-string.

---

## What RULER adds over NIAH (the lineage point)

From the raw-data:

> *"Models may retrieve one item correctly but fail when needle format changes from numbers to UUIDs; models may find the right item once but fail to ignore hard distractors; models may retrieve one target but fail at high-recall multi-target output; models may copy local clues but fail at chain tracing across long-range dependencies; models may do sparse lookup but fail at aggregation."*

These are the five failure modes NIAH can't separately expose. Each of the 13 task variants isolates one. That is the analytical payoff.

---

## For long-context data designers

> *"Train and evaluate on families of synthetic generators, not one canned test. Separate length scaling from reasoning/load scaling. Include tasks where the answer is a set, not just a single span."*

The transfer to synthesis: if your SFT includes multi-needle, include *varied* multi-needle — 4 keys, 8 keys, UUID keys, word keys — exactly because RULER discovered that these configurations probe different failure modes. Qwen 2.5-1M's synthetic SFT mix (multi-needle / summarization / RAG-QA / FILL-IN) is a direct instance of this lesson ([[excerpts/qwen-1m-pipeline]]).

---

## Connections

- Chapter synthesis: [[ch-28]]
- Reasoning-in-a-haystack sibling: [[excerpts/babilong-pg19-embed]]
- NIAH ancestor: [[excerpts/niah-kamradt-original]]
- Qwen 1M uses RULER as both eval and training-signal target: [[excerpts/qwen-1m-pipeline]]
