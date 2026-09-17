---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ruler.md
source_url: https://arxiv.org/abs/2404.06654
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision — unverifiable per-task figures replaced with Table 3 rows)"
---

# Excerpt: RULER — length slices and the claimed-against-effective gap

**Source library:** `wiki/raw-data/llm-training/papers/ruler.md`
**Artifact:** RULER: What's the Real Context Size of Your Long-Context Language Models? Hsieh, Sun, Kriman,
Acharya, Rekesh, Jia, Zhang, Ginsburg (NVIDIA). arXiv:2404.06654, COLM 2024.
**Checked on 2026-09-15** against the arXiv PDF.

> **Correction carried by this excerpt.** The earlier version of this excerpt and of ch-50 attributed to
> Table 3 a per-task example ("~99% on S-NIAH, ~40% on MK-NIAH-full-haystack, ~20% on VT-4-hops, effective
> window 8K"). Table 3 reports the **average of the 13 tasks** per model and length, plus claimed and
> effective length; it contains no such per-task row. The per-task degradation the example was reaching for
> is in §5 and Fig. 2, quoted below with the numbers the paper actually gives.

---

## Why ch-50 uses this source

RULER supplies the length axis of the long-context slice grid in ch-50 §4, and the operational definition
of effective context length.

---

## Metric definitions (§4)

- 500 generated examples per task per length; lengths 4K, 8K, 16K, 32K, 64K, 128K; inputs wrapped in each
  model's chat template with an answer prefix appended; recall-based matching for accuracy.
- Each cell of Table 3 is the **average over the 13 tasks** at that length.
- **Effective context length** = the maximum length whose average exceeds the Llama2-7B score at 4K, which
  is **85.6**.
- Two weighted averages are reported, `wAvg. (inc)` and `wAvg. (dec)`, with weights increasing or
  decreasing linearly with length.

## Table 3 rows used in ch-50 §4

| Model | Claimed | Effective | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|---|---|
| Llama2 (7B) | 4K | — | 85.6 | | | | | |
| Gemini-1.5-Pro | 1M | >128K | 96.7 | 95.8 | 96.0 | 95.9 | 95.9 | 94.4 |
| GPT-4 | 128K | 64K | 96.6 | 96.3 | 95.2 | 93.2 | 87.0 | 81.2 |
| Llama3.1 (70B) | 128K | 64K | 96.5 | 95.8 | 95.4 | 94.8 | 88.4 | 66.6 |
| Qwen2 (72B) | 128K | 32K | 96.9 | 96.1 | 94.9 | 94.1 | 79.8 | 53.7 |
| Yi (34B) | 200K | 32K | 93.3 | 92.2 | 91.3 | 87.5 | 83.2 | 77.3 |
| Mixtral-8x22B | 64K | 32K | 95.6 | 94.9 | 93.4 | 90.9 | 84.7 | 31.7 |
| Mistral-v0.2 (7B) | 32K | 16K | 93.6 | 91.2 | 87.2 | 75.4 | 49.0 | 13.8 |
| LWM (7B) | 1M | <4K | 82.3 | 78.4 | 73.7 | 69.1 | 68.1 | 65.0 |

Of these, only Gemini-1.5-Pro's effective length reaches its claimed length. The two weighted averages
order models differently — LWM ranks 12th under `wAvg. (inc)` and 15th under `wAvg. (dec)` — so the
aggregation weighting is itself a reported choice rather than a property of the models.

## Per-generator degradation at fixed model — §5, Fig. 2 (Yi-34B-200K)

The paper's per-task analysis is run on one model with input lengths up to 256K:

> Figure 2 (left) shows that while Yi achieves almost perfect performance when using needle of word-number
> pair in the standard passkey retrieval and vanilla NIAH, performance degrades when the needle takes other
> forms. We observe the largest degradation in the task of retrieving UUIDs … (§5)

> Figure 2 (middle-left) shows that increasing the number of distracting needles steadily lowers
> performance, with Yi dropping by ∼40 points at 256K in the extreme version, where the context is full of
> irrelevant needles (#K=FULL). (§5)

The error analysis adds that in the `#K=FULL` setting Yi "often returns values from the vicinity of the
target, suggesting coarse match of the range but the lack of precision to locate the key" (§5).

## The 13 task configurations (Table 5, summarized)

Retrieval: S-NIAH (word→number with noise haystack; word→number with essay haystack; word→UUID),
MK-NIAH (4 distractor keys; full-haystack distractors word→number; full-haystack distractors UUID→UUID),
MV-NIAH (4 values), MQ-NIAH (4 queried keys). Multi-hop tracing: VT (1 chain, 4 hops). Aggregation:
CWE (10 common words × 30 occurrences), FWE (α = 2.0, top-3). QA: SQuAD, HotpotQA long-context adaptations.

## Used by

ch-50 §4 (length axis of the long-context grid), §8 (decision table), Recipe (examples-per-length and
effective-length-threshold rows), Common mistakes (effective length without its threshold).
