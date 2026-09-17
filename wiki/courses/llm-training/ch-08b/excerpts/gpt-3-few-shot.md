---
chapter: ch-08b
course: llm-training
phase: read
excerpt_of: primary source arXiv:2005.14165v4 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2005.14165
created_at: "2026-09-15"
---

# Excerpt: Language Models are Few-Shot Learners (GPT-3)

**Paper:** Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al. (OpenAI). arXiv v1 2020-05, read at v4 (2020-07-22). Source type: paper / official technical report.

## Definitions (§1, §2, Fig. 1.1, Fig. 2.1)
- "In-context learning" is the inner loop of meta-learning that "occurs within the forward-pass upon each sequence"; pretraining is the outer loop (Fig. 1.1).
- The authors state that the terms zero-, one-, and few-shot do not decide whether the model learns new tasks from scratch at inference time or recognizes patterns seen during training (§1 footnote 1).
- Few-shot: K demonstrations of context and completion, then one context; K typically 10 to 100, as many as fit in n_ctx = 2048; no weight updates. One-shot: K = 1 plus a task description. Zero-shot: instruction only (§2).
- Motivation: "the potential to exploit spurious correlations in training data fundamentally grows with the expressiveness of the model and the narrowness of the training distribution" (§1).

## Models and training (Table 2.1, §2.1-2.3, App. B)
- 8 sizes, all trained on 300 billion tokens, context 2048, d_ff = 4 · d_model, alternating dense and locally banded sparse attention.
- Table 2.1 (params; layers; d_model; heads; d_head; batch in tokens; LR): 125M; 12; 768; 12; 64; 0.5M; 6.0e-4 — 350M; 24; 1024; 16; 64; 0.5M; 3.0e-4 — 760M; 24; 1536; 16; 96; 0.5M; 2.5e-4 — 1.3B; 24; 2048; 24; 128; 1M; 2.0e-4 — 2.7B; 32; 2560; 32; 80; 1M; 1.6e-4 — 6.7B; 32; 4096; 32; 128; 2M; 1.2e-4 — 13.0B; 40; 5140; 40; 128; 2M; 1.0e-4 — 175.0B; 96; 12288; 96; 128; 3.2M; 0.6e-4.
- Batch size chosen with measured gradient noise scale; V100 GPUs on a Microsoft cluster (§2.3).
- App. B: Adam β1 = 0.9, β2 = 0.95, ε = 1e-8; global-norm clip 1.0; cosine decay to 10% over 260B tokens, then 10% constant; linear warmup over 375M tokens; batch ramp from 32k tokens over the first 4-12B tokens depending on size; sampling without replacement until an epoch boundary; weight decay 0.1; full 2048-token sequences with packed documents separated by an end-of-text token and no special masking.

## Data (§2.2, Table 2.2)
- Common Crawl: 41 monthly shards 2016-2019, 45TB compressed before filtering, 570GB after (about 400B BPE tokens); filtered by similarity to high-quality reference corpora; fuzzy document-level dedup within and across datasets.
- Table 2.2 (tokens; weight in mix; epochs at 300B): Common Crawl 410B, 60%, 0.44; WebText2 19B, 22%, 2.9; Books1 12B, 8%, 1.9; Books2 55B, 8%, 0.43; Wikipedia 3B, 3%, 3.4.
- Higher-quality datasets sampled more frequently, "accepts a small amount of overfitting in exchange for higher quality training data".
- Check by ch-08b (derived): weight × 300B / tokens gives 0.44, 3.47, 2.00, 0.44, 3.00; printed WebText2 and Wikipedia epochs differ; not explained in the report.

## Evaluation protocol (§2.4)
- K examples drawn at random from the task's training set, separated by 1 or 2 newlines; K chosen on a development set when one exists; larger K "usually but not always better".
- Multiple choice: per-token-normalized likelihood; for ARC, OpenBookQA, RACE: P(completion|context) / P(completion|answer_context), answer_context = "Answer: " or "A: ".
- Free-form: beam width 4, length penalty α = 0.6.

## Results used in ch-08b
- CoQA 81.5 / 84.0 / 85.0 F1 (zero / one / few-shot) (§1); TriviaQA 64.3 / 68.0 / 71.2 (§1, Table 3.3).
- Table 3.2 LAMBADA accuracy 76.2 / 72.5 / 86.4; fill-in-the-blank format lets the model infer a one-word completion (§3.1.2).
- Table 3.9 (175B, exact match, 2,000 instances per task; zero / one / few): 2D+ 76.9 / 99.6 / 100.0; 2D- 58.0 / 86.4 / 98.9; 3D+ 34.2 / 65.5 / 80.4; 3D- 48.3 / 78.7 / 94.2; 4D+ 4.0 / 14.0 / 25.5; 4D- 7.5 / 14.0 / 26.8; 5D+ 0.7 / 3.5 / 9.3; 5D- 0.8 / 3.8 / 9.9; 2Dx 19.8 / 27.4 / 29.2; 1DC 9.8 / 14.3 / 21.3. The §3.9.1 prose gives 80.2% for 3-digit addition.
- 13B solves 2-digit addition and subtraction about half the time and other operations less than 10% (§3.9.1).
- Memorization spot check: 17 of 2,000 3-digit addition problems (0.8%) and 2 of 2,000 subtraction problems (0.1%) found in training data (§3.9.1).
- Table 3.10 (zero / one / few, K = 100): CL 3.66 / 21.7 / 37.9; A1 2.28 / 8.62 / 15.1; A2 8.91 / 25.9 / 39.7; RI 8.26 / 45.4 / 67.2; RW 0.09 / 0.48 / 0.44. Authors: inability zero-shot suggests the tasks are learned at test time (§3.9.2).
- Fig. 1.3: aggregate over 42 accuracy-denominated benchmarks; few-shot performance increases more rapidly with size than zero-shot.
- ANLI: models smaller than GPT-3 are at almost exactly chance (~33%) even few-shot (§3.8). WiC and ANLI near chance one-shot and few-shot for GPT-3 (§5).

## Contamination (§4)
- A filtering bug left some benchmark overlaps; clean subsets remove examples with 13-gram overlap.
- PIQA: 29% flagged, 3-point absolute drop on the clean subset, asterisked. Winograd: 45% flagged, 2.6% drop, 132 schemas present in another format, asterisked. Four Wikipedia LM benchmarks and CBT almost entirely contained, not reported. LAMBADA: substantial genuine contamination, clean subset within 0.5%.

## Limitations (§5)
- Ambiguity whether few-shot learning learns tasks "from scratch" or recognizes tasks learned during training; may vary by task.
- The objective "weights every token equally and lacks a notion of what is most important to predict".

## Verification
- Read on 2026-09-15 against arXiv PDF v4 (§1-§5, App. B, Tables 2.1, 2.2, 3.2, 3.3, 3.9, 3.10).
