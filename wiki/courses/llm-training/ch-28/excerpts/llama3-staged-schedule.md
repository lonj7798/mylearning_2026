---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Meta Llama Team — "The Llama 3 Herd of Models" (long-context subsection §3.4)
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — the 6-stage 800B-token schedule and RoPE base = 500K

**Source:** `wiki/raw-data/llm-training/papers/long-context-llama3.md`
**Paper:** Meta Llama Team (Grattafiori et al.), 2024
**arXiv:** https://arxiv.org/abs/2407.21783

---

## Bibliographic header

> *"Llama 3's 128K-context extension is split into a staged continued-pretraining schedule (8K → 16K → 32K → 64K → 128K across 800B tokens) with RoPE base rescaled to 500K, followed by a long-context SFT stage on synthetic long-doc QA."*

This is the production-scale recipe — more compute than any other 2024 open recipe by an order of magnitude. The lesson is not "do exactly this"; it's how Meta chose to *spend* 800B tokens of compute on the problem.

---

## The six-stage schedule

From the raw-data:

> *"Stage A — 8K → 16K: moderate data, ~100B tokens. Stage B — 16K → 32K: ~100B tokens. Stage C — 32K → 64K: ~150B tokens. Stage D — 64K → 128K: ~200B tokens. (Additional intermediate stages for stability.) Total: ~800B tokens across all stages."*

The specific design choice: **each stage roughly doubles the context**. This is the NTK-aware position-interpolation literature's recommendation — small rescaling steps give the RoPE sinusoids time to stabilise. Jumping from 8K to 128K in one stage would cause the low-frequency dimensions to alias, producing training instability.

The token-budget profile — 100B, 100B, 150B, 200B — is increasing with context length, which makes sense per-sequence: each training step at 64K context processes 8× fewer sequences than at 8K context, so reaching the same number of sequence-equivalents requires more tokens.

---

## Data mix shift per stage

> *"Proportions shift toward long documents (books, code repos, papers) in later stages. Short-to-long document ratio gradually rebalances from ~80:20 (stage A) to ~40:60 (stage D)."*

The 80:20 → 40:60 shift is the opposite of what a naïve recipe would do. A naïve recipe would say: *"At 128K context, use only 128K-capable documents; why waste capacity on short docs?"* Meta's answer: short docs serve as regularisation that preserves short-context behaviour. Pushing the long-doc fraction too high (>60% at 128K) costs short-context capability.

This is the continued-pretraining analogue of the long-SFT-fraction constraint below.

---

## RoPE base = 500K

> *"Base rescaled from 10K to 500K for the final 128K model. Scaling done progressively alongside the staged training — at each stage, the RoPE base is adjusted to match the new context window."*

The formula change is the most cited single line from §3.4. Llama-3 scales RoPE from `θ = 10,000` (Llama-2 default) to `θ = 500,000` — a **50× base increase** for a **16× context increase** (8K → 128K). The base scaling is *faster than the context scaling*, which is NTK-aware's expected behaviour: the rescale factor applies as `θ → θ · s^(d/(d-2))` where `s` is the context ratio, and for Llama-3's head dim 128, `s^(d/(d-2)) = s^(128/126) ≈ s`. So `s ≈ 50` producing ~50× base rescale for ~50× effective context range (16× claimed + headroom).

No YaRN, no NTK-aware trick beyond the direct rescale. The paper is explicit that simple rescaling combined with the staged training schedule is sufficient — the elaborate position-encoding tricks in other work (YaRN, LongRoPE) are *not* needed at this compute budget. Compute can substitute for position-math cleverness, up to a point.

---

## Long-context SFT at 0.1%

> *"A small fraction (~0.1%) of SFT samples are long-context: synthetic QA over long documents, multi-document summarization, long-context code analysis. Generation uses a larger Llama 3 model as teacher on full documents. Keeping the long-SFT fraction low prevents short-context regression."*

The 0.1% is the binding constraint. Raising it above 1% costs ~1 MMLU point.

Why such a small fraction works: the base model already saw 800B tokens of long-context continued pretraining, so the capability is already present. SFT's job is just to *elicit* long-context instruction-following, not to teach it from scratch. A small fraction is enough to change the output distribution for long prompts without over-shifting behaviour on short prompts.

**Notice:** this is the opposite philosophy from ProLong, which uses 70% long in SFT. The difference is compute: ProLong has 5B SFT tokens and needs to be aggressive; Llama-3 has ~50-100B SFT tokens and can afford a small long-fraction.

Teacher model = Llama 3 405B itself — self-distillation on long-context SFT.

---

## Claimed vs effective

> *"Llama 3.1-70B: NIAH 128K ~99%; RULER 128K ~75% (effective context ~64K)."*

The NIAH-to-RULER gap is ~24 points at 128K; the effective-context metric says the model is only Llama2-7B-4K-equivalent out to 64K, not 128K. Meta acknowledges this explicitly in the paper — it's not a contested number, it's an upstream admission. The gap motivates the reasoning-in-a-haystack evaluations (BABILong) and the multi-needle training ablations (Qwen 1M) that subsequent work adopted.

---

## What scales and what doesn't

> *"Staged schedule is compute-intensive: 800B tokens of CPT is out of reach for smaller labs."*

800B is roughly 5% of Llama-3's full 15T pretraining — a significant fraction. For labs with an order of magnitude less compute, the recipe choices have to change: Fu 2024 (5B CPT on SlimPajama) or LongRoPE (<1B FT with per-dim search) are the reasonable shapes at smaller budgets.

> *"RoPE base = 500K works for Llama 3's architecture; the right value depends on head-dim and pretrain base."*

The 500K figure is *not* transferable to other base models. Llama-3's head dim is 128 and its pretraining base was 10K; a different base model with different pretraining would want a different rescale. LongRoPE's per-dim search is the principled way to avoid hand-tuning this scalar per model.

---

## Connections

- Chapter synthesis: [[ch-28]]
- Data-quality thesis counterpart: [[excerpts/prolong-coherence]]
- Search-based RoPE alternative: [[excerpts/longrope-per-dim-search]]
- Evaluation: [[excerpts/ruler-task-family]], [[excerpts/babilong-pg19-embed]]
- 1M-context successor: [[excerpts/qwen-1m-pipeline]]
