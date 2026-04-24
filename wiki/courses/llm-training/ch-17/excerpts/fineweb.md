---
chapter: ch-17
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md
source_url: https://arxiv.org/abs/2406.17557
created_at: "2026-04-23"
---

# Excerpt: FineWeb — ablation methodology and the per-snapshot MinHash surprise

**Source library:** `wiki/raw-data/llm-training/papers/fineweb.md`
**Paper:** Penedo, Kydlíček, Ben Allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf 2024, "FineWeb: Decanting the Web for the Finest Text Data at Scale."

---

## Why FineWeb is the ablation template ch-17 copies

Every ablation row in the lab is doing what [[fineweb]] §3 does: train a small model on each filter variant, evaluate on a fixed suite, and compare deltas. The numbers in ch-17's `ablation-grid.html` are scaled-down echoes of FineWeb's Figure 4 and Table 2.

From the source (lines 24–26):

> **Ablation curves** showing downstream accuracy vs filter aggressiveness — the classic quality-vs-quantity tradeoff.
> **Per-dump MinHash vs global dedup** comparison — HF found per-dump outperforms naive global dedup on downstream tasks (surprising; tied to removing near-identical re-crawls).
> **FineWeb-Edu threshold sweep** — accuracy on MMLU/ARC vs classifier-score threshold 1 through 5.

The lab reproduces the first in miniature, nominates the second as a stretch row, and defers the third to ch-20.

---

## The per-snapshot MinHash finding — the memo's "one thing I was wrong about" candidate

From the source (lines 33–35):

> 5. **MinHash deduplication per snapshot** (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.

This is the single most counter-intuitive published result in the 2024 web-data literature. The intuition says: duplicates are duplicates, remove every copy, done. The FineWeb finding says: a document that appears in snapshot N and then again in snapshot N+12 is *not* the same kind of repetition as a document that appears 30 times in snapshot N. The former is a website that has been stable for a year and shows up in every crawl; the latter is spam or boilerplate.

Global dedup treats both the same and removes most of the first class — which turns out to be high-quality stable web content: Wikipedia articles, educational sites, official documentation. FineWeb found that preserving cross-snapshot repeats and deduping only within snapshots produces better downstream benchmarks.

Ch-17's lab asks students to reproduce this on a small slice as a stretch row. At 1 GB the effect may be weak (the slice is usually one snapshot), but if the student's 10 GB path spans 2+ dumps, the sign of the `+MinHash-global` row minus the `+MinHash-per-snapshot` row should come out negative on MMLU/ARC — matching FineWeb. If it does not, the memo's "one thing I was wrong about" paragraph essentially writes itself.

---

## The FineWeb-Edu classifier — the ablation ch-17 does *not* run

From the source (lines 37–44):

> **FineWeb-Edu classifier:** Trained on 450K web samples annotated by Llama-3-70B-Instruct with integer scores 0–5, where 0 = not educational, 5 = highly educational. Small classifier head on top of a frozen embedding model. Filter keeps documents with score ≥ 3. This threshold removes ~92% of FineWeb, leaving 1.3T tokens. On a hold-out of 46,867 Llama-3-annotated samples, the score≥3 binary classifier reaches F1 = 82%.
>
> **Why classifier > heuristics at scale:** heuristics (C4, CCNet) plateau — adding more heuristic filters doesn't help MMLU. A single LLM-labeled educational-value classifier captures what no regex stack can: the vibe of a textbook vs the vibe of a forum post.

Ch-17 deliberately stays in the pre-classifier world. The reason is pedagogical: if your very first experience of data filtering is "train a classifier," you never feel the ceiling of heuristic stacks. FineWeb's own argument — that heuristics *plateau* — is a delta you can only feel once you have hit the plateau yourself. The lab's four-stage CCNet-style pipeline is how you hit it.

The ch-17 memo's expected observation: the gain from each filter is real but small (0.5–2 points per eval task, diminishing by stage 4). That diminishing return is the signal that would motivate a classifier in a real corpus; it is what FineWeb §4 formalises. The bridge to ch-20's FineWeb-Edu discussion is this paragraph in the memo: "I hit the heuristic plateau. A classifier is the next natural step."

---

## The F1-82% threshold lesson for any later classifier work

The 82% figure matters because FineWeb-Edu's threshold-3 cutoff removes 92% of tokens. A classifier with F1 = 0.82 at that aggressive a cut will make thousands of bad drop decisions on a 1 GB slice and tens of millions on a 15T-token corpus. The paper's stance is: this is fine, because the *correct* drops dominate the benchmark gain. Ch-17 lab students who stretch toward a classifier row in future coursework should know this tradeoff going in — a classifier replaces a deterministic error (heuristic over-inclusion) with a stochastic one (classifier mistakes), and stochastic errors are harder to diagnose from a memo. That is the ch-20 conversation, not ch-17's.
