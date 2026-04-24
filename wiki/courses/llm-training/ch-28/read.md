<!-- chapter: ch-28
     track: synthetic
     title: Modality — Long-Context Synthesis
     sources: [[longalpaca]], [[longalign]], [[longchat]], [[longmit]], [[prolong]],
              [[ruler]], [[babilong]], [[needle-in-haystack-data]], [[longbench]],
              [[long-context-llama3]], [[long-context-data-engineering]],
              [[qwen-long-context-synth]], [[longrope-data]], [[pose-synthesis]],
              [[longembed-synth]], [[gemini-long-context-tricks]]
     figures: figures/context-extension.html
-->

# Chapter 28 — Modality: Long-Context Synthesis

> **Core insight.** Long-context capability is not a single-knob extension of short-context training — it is the *co-design* of three independent axes that every 2024+ frontier release has had to solve simultaneously. (i) **Position encoding**: RoPE's base frequency must be rescaled (Llama-3 uses 500K; Fu-2024 uses 200M; LongRoPE uses *per-dimension* factors λ_i found by evolutionary search) or the sinusoids alias past the pretraining range. (ii) **Data**: you need long *coherent* documents, not concatenations of short ones — ProLong's coherence filter is the difference between 512K context on 20B tokens and garbage. (iii) **Evaluation**: a single-needle NIAH pass does not prove long-context; RULER's 13-task generator and BABILong's reasoning-in-a-haystack reveal that "claimed context" and "effective context" routinely differ by 2× on frontier models (Llama-3.1-70B: claimed 128K, effective ~64K per RULER). Synthesis — of instruction data and of evaluation tasks — is what makes each of the three axes tractable.
>
> **Guideline.** Treat long-context training as a three-stage stack: (1) **position extension** via staged RoPE-base rescaling (or LongRoPE per-dim search if you're data-poor); (2) **continued pretraining on coherent long documents** at a modest token budget (5B for Fu, 20B for ProLong) with cross-domain ratios preserved and within-domain length upsampling; (3) **SFT with synthesized long tasks** — LongAlign-style 5-questions-pick-one for cross-span coverage, multi-needle retrieval for explicit retrieval training, LongMIT-style multi-turn for dialog state. Evaluate on RULER + BABILong + realistic LongBench — never NIAH alone.

---

## Why this chapter exists

Chapters 23–27 established the synthetic-data design pattern for short-context SFT: generate → filter → dedup → verify → select → mix. Long-context breaks each step of that loop. A teacher model must *read* 100K tokens of source document before it can generate a useful question-answer pair, so teacher-context ceiling becomes a hard constraint ([[longmit]]). Filter cost scales with sequence length, so MinHash at 32K-shingled docs is 8× more expensive than at 4K. Verification is the steepest: there is no cheap automatic check that a generated answer actually uses evidence spread across 50K tokens of document; LLM-judge + human spot-check is all we have ([[longalign]] validates 94/100 manually).

At the same time, the *architecture* side changed. Context-extension is not just "train longer sequences" — RoPE ([[longrope-data]], [[long-context-data-engineering]]) has a fixed frequency spectrum tied to its base θ, and pushing position indices past the training range produces frequency aliasing unless you rescale. The resulting stack — position fix + long-doc CPT + synthetic long-SFT + synthetic long-eval — is what distinguishes Llama-3-128K, ProLong-512K, and Qwen-2.5-1M from each other.

This chapter walks the stack in the order labs actually build it: synthetic evaluation first (because you can't design data for a capability you can't measure), then SFT-only recipes, then continued-pretraining recipes, then the position-encoding lane that cuts fine-tune data by 10×, then the 1M-context frontier.

The raw-data library backing each section is in `wiki/raw-data/llm-training/papers/` — 16 source files, from the Kamradt NIAH README to the Qwen 2.5-1M technical report. Each section here cites them via `[[wikilink]]`; for deeper walk-throughs of the single most load-bearing claim per paper, see the excerpts under `excerpts/`.

---

## 1. The synthetic-task family — NIAH → RULER → BABILong

Long-context *evaluation* is itself synthetic — because there is no natural benchmark long enough. The lineage starts with a blog post.

### 1.1 Needle-in-a-Haystack (Kamradt, Nov 2023)

The original [[needle-in-haystack-data]] test is a one-liner: hide the sentence *"The best thing to do in San Francisco is eat a sandwich at Dolores Park on a sunny day"* at programmatic depth inside Paul Graham essays padded to target length, ask *"What is the best thing to do in San Francisco?"*, score exact-substring. The output is a 2D heatmap (depth × length, accuracy as colour). No teacher, near-zero cost, visually legible. That last property is why it became the *de facto* long-context marketing metric — every 128K / 200K / 1M-context release now ships a NIAH heatmap.

**What NIAH misses.** It tests retrieval of one fact, at one depth, with one query. Real long-context behaviour requires retrieval of multiple facts, ignoring distractors, aggregating across spans, and reasoning across retrieved items. A model can pass NIAH at 128K while collapsing on multi-hop at 32K. Hence the two successor benchmarks.

### 1.2 RULER and BABILong — the task-family table

| Family | Paper | Primitive | Complexity knobs | What it stresses |
|---|---|---|---|---|
| **S-NIAH** (single-needle) | [[needle-in-haystack-data]], [[ruler]] | 1 key-value injected in haystack | depth, length, needle type (word / 7-digit / UUID), haystack type (noise / essay) | baseline retrieval |
| **MK-NIAH** (multi-key) | [[ruler]] | N keys, query one | N ∈ {4, …, full-haystack-distractors} | distractor resistance |
| **MV-NIAH** (multi-value) | [[ruler]] | one key → k values, return all | k ∈ {2, 4, 8} | recall completeness |
| **MQ-NIAH** (multi-query) | [[ruler]] | multiple independent queries per haystack | number of queries | parallel retrieval |
| **VT** (variable tracing) | [[ruler]] | `X2 = X1`, `X3 = X2`, …; return equivalents | chain count, hops per chain | coreference / state |
| **CWE** (common-word extraction) | [[ruler]] | tokens from common + uncommon word distribution | common count, frequency ratio | aggregation |
| **FWE** (frequent-word extraction) | [[ruler]] | Zeta-distributed tokens, return top-K | Zeta α, K | aggregation tail |
| **QA** | [[ruler]] | SQuAD / HotpotQA + distractor paragraphs | paragraph count | realistic retrieval + reasoning |
| **bAbI-in-PG19** | [[babilong]] | 20 bAbI reasoning tasks inside natural PG19 prose | 0K → 10M length (50M reported), 20 task templates | retrieval + symbolic reasoning |

RULER's three methodology points are quietly load-bearing: (a) **context length and task complexity vary independently** — you can tell whether a model broke because of raw length or because of distractor density; (b) **500 examples per task per length** with an explicit answer prefix in the chat template; (c) **effective context size** is defined as the longest length whose score stays above the Llama2-7B@4K baseline of 85.6. BABILong's complementary choice is to keep the reasoning structure *templated* (contamination-resistant) while the distractors are *real prose* (not artificial filler).

**The result that matters for every training report.** Many models with advertised 128K windows degrade sharply once distractors, multiple targets, or aggregation are introduced — claimed ≠ effective context. Llama-3.1-70B's NIAH @ 128K is ~99%, but its RULER effective context is ~64K ([[long-context-llama3]]). Qwen-2.5-14B-1M's NIAH @ 1M is ~100% but its RULER @ 1M is ~85% ([[qwen-long-context-synth]]). The gap is the reasoning-in-a-haystack tax.

---

## 2. Early SFT-only recipes — LongAlpaca and LongAlign

Before anyone had the compute for 20B-token continued pretraining, the question was: *can we make a short-context base behave like a long-context model using only SFT?*

### 2.1 LongAlpaca-12K ([[longalpaca]], Chen et al. 2023)

The recipe is the minimum viable product. (i) Collect **3,000 long documents** — 40% ArXiv CS papers, 30% public-domain books, 30% GitHub repos. (ii) For each document, prompt **ChatGPT or Claude** (Claude preferred on the longest docs, then still at 100K ceiling) with the full text plus a task-type specification (summarize / QA / extract / analyze). Generate **3 QA pairs per document**. (iii) Filter: document ≥ 8K tokens, answer ≥ 30 tokens, profanity check. (iv) **Mix in 3K random Alpaca samples** (≤2K tokens) to preserve short-chat behaviour. Pair with **LongLoRA's shifted-sparse attention** and you extend Llama-2-7B to 32K/100K at ~$5K in API fees.

The artifact — `Yukang/LongAlpaca-12k` — became the reference baseline for subsequent long-context datasets, not because it was state-of-the-art but because it shipped.

### 2.2 LongAlign-10k ([[longalign]], Bai et al. 2024)

LongAlign is the first recipe to take long-context SFT seriously as a distinct training problem. Four ingredients:

1. **Self-Instruct-style synthesis on 9 source mixes** — ArXiv, Books3, C4, CLUECorpus2020, CommonCrawl, GitHub, StackExchange, Wikipedia, WuDaoCorpora; 90% EN / 10% ZH. Teacher is **Claude 2.1**. Generation is two-stage: *ask Claude for 5 candidate questions covering the whole document, then randomly pick one and ask for the answer.* That pick-one trick forces cross-span coverage — without it, the teacher picks locally-answerable questions, and the student learns long retrieval, not long reasoning.
2. **Packing + block-diagonal mask** using `flash_attn_varlen_func` with `cu_seqlens`. Average pack holds ≈12 sequences; batch-size 8 → global batch 96.
3. **Sequence-level loss weighting** — the naïve packed loss over-weights packs with few (long) sequences and over-weights targets with more tokens. Fix: weight each target token by `1/N` where `N` is that sequence's target length; during training scale by `K/(M·N)` where `K` packs, `M` sequences. On ChatGLM3-6B-64k, LongBench-Chat rises from **5.76 → 6.21**; on Llama-2-7B-64k, **5.89 → 6.10**.
4. **Pre-SFT context extension**: expand RoPE base **10,000 → 2,000,000** (200× rescale) and continually pretrain to 64K on 10B tokens *before* SFT.

LongAlign is therefore a post-extension recipe; it does not claim to solve position extrapolation. Data quality saturates around **10k long examples** — beyond that, diversity matters more than volume, and LongAlign-10k beats the larger LongAlpaca-12k on multi-segment integration.

### 2.3 LongChat and LongMIT — the conversational variants

[[longchat]] (LMSYS, June 2023) mines **ShareGPT's long tail** — real user conversations ≥ 8K tokens, ≥ 4 turns — for 18K long conversations, then fine-tunes Vicuna with **condensed rotary embedding** (position index `i` becomes `i/c` with `c = 8` for 16K). It's the real-log counterpart to LongAlign's synthetic pipeline. Short-lived as a position trick (superseded by NTK-aware → YaRN → LongRoPE) but durable as a *data-sourcing signal*: real long conversations contain topic shifts, backtracking, and cross-turn reference patterns that synthesis struggles to emulate.

[[longmit]] generalizes this to synthesized multi-turn long-context dialogs — 5–10 turns each referencing document spans, full context 20K–100K — and reports adding multi-turn on top of single-turn long-doc SFT lifts LongBench-Chat by **5–10 points** across model families. The limiter is teacher-context ceiling: only frontier closed models can coherently generate a 10-turn conversation conditioned on a 50K-token document.

---

## 3. ProLong — the document-coherence thesis

[[prolong]] (Gao, Wettig, Yen, Chen; Princeton NLP 2024) is the paper that answered "does long-doc quality matter or is volume enough?". Answer: quality matters, and the difference is 10+ points on HELMET.

### 3.1 The coherence filter — what counts as a "long document"

ProLong's training mix is curated against an explicit *coherence* criterion, not just length. The filter threshold is **≥ 64K tokens of coherent content per document**, with coherence judged source-type-by-source-type:

- **Code**: whole *repository* (README → source → tests concatenated in sensible order), not single files.
- **Books**: full-book PDFs parsed with structural fidelity.
- **Academic**: full papers *with references*.
- **Web**: **discarded** — even long web docs are mostly scraped listings with weak long-range dependency.

That last rule is the paper's sting. "Long" and "coherent" are not the same predicate on web data. After the filter, the 30B-token mix is re-weighted per source: **code × 4, books × 2, academic × 2, forum × 1, web × 0.5**, producing a final distribution of ~40% code, 25% books, 15% academic, 10% long forum threads, 10% misc web.

The ablation is the proof. Replacing the curated long documents with *concatenated short documents* of equal token budget — the obvious shortcut — costs **10+ points on HELMET**. The concatenation shortcut teaches a model that "long context" = "sequence of locally-coherent short segments," which is the failure mode that then shows up on RULER's multi-hop tracing and BABILong's reasoning tasks.

### 3.2 The staged schedule

- **Stage 1 — CPT (20B tokens)**: RoPE base rescaled **500K → 128M** (Llama-3.1 NTK-aware style). Train at 64K context initially, expand to 512K in the second half. LR 1e-4 → 1e-5 cosine. **100% long coherent documents**, one document per training sample, *no cross-document packing*.
- **Stage 2 — SFT (5B tokens)**: 70% long-instruction (LongAlign-style, Claude-3-generated) + 30% short-instruction (UltraChat) + synthetic multi-needle NIAH training samples to explicitly teach retrieval.

Total compute ~200K H100-hours produces ProLong-8B (Llama-3-8B base) at 512K context, leading among open 8B models on HELMET and beating Llama-3.1-8B-Instruct and Qwen2-7B-Instruct on InfiniteBench @ 128K.

---

## 4. The production recipe — Llama 3 and Fu 2024

Two parallel 2024 reports nail down the frontier recipe for 128K.

### 4.1 Llama 3's staged schedule ([[long-context-llama3]])

Meta extends Llama 3.1 (405B / 70B / 8B) from 8K to 128K with a **six-stage continued pretraining** schedule, ~800B tokens total:

| Stage | Context | Tokens | RoPE base | Data mix shift |
|---|---|---|---|---|
| A | 8K → 16K | ~100B | adjusted | short:long 80:20 |
| B | 16K → 32K | ~100B | adjusted | 70:30 |
| C | 32K → 64K | ~150B | adjusted | 60:40 |
| D | 64K → 128K | ~200B | **500K** (final) | 40:60 |

**The key formula change.** The RoPE base frequency `θ` is rescaled from Llama-2's `10K` to **500K** for the final 128K model. RoPE's per-dimension frequency is

$$
\theta_i \;=\; \theta^{-2i/d}
$$

with `d` the head dimension. Increasing `θ` *shrinks* every frequency, *stretches* every wavelength, and pushes the sinusoid aliasing point further out along the position axis. Llama 3 ships at `θ = 500K`; Fu 2024 ([[long-context-data-engineering]]) pushes to `θ = 200M` for 128K on Llama-2; LongRoPE generalizes to per-dimension λ_i.

**Post-training integration.** Long-context SFT is kept at **~0.1% of total SFT samples** (~100K out of ~100M). Raising the long-SFT fraction above 1% costs ~1 MMLU point — the short-context regression is the binding constraint, not long-context gain. Teacher is Llama 3 405B itself (self-distillation).

**The claimed-vs-effective gap.** Llama-3.1-405B NIAH@128K is ~99% and RULER effective context is ~96K. Llama-3.1-70B's effective RULER context is ~64K despite 128K support — Meta acknowledges this in the paper. The training-eval co-design gap shows up as soon as RULER replaces NIAH.

### 4.2 Fu 2024 — the 5B-token open recipe ([[long-context-data-engineering]])

A complementary thesis: **cross-domain ratios must be preserved; only within-domain length distribution changes.** Start from SlimPajama with 7 sources (CC, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange). Preserve the original proportions (CC ≈ 67%, Books ≈ 4%, …). *Within each source*, compute a length histogram and reweight sampling so documents longer than 32K get **5× weight**. Train 5B tokens at 80K context window with RoPE base rescaled `10K → 200M` NTK-aware.

The critical ablation: breaking the cross-domain ratio (e.g., globally up-weighting Books) preserves long-context NIAH but drops short-context MMLU by **3–5 points**. Long-context gain must not cost short-context capability; that's what the within-domain-only rule buys you.

---

## 5. LongRoPE and the per-dim search — the position-encoding lane

The data-centric papers share a question they rarely answer: *is there a RoPE rescaling that works better than uniform NTK-aware or YaRN?*

[[longrope-data]] (Ding et al., MSRA 2024) answers with evolutionary search. RoPE applies rotation at frequency `θ_i = θ^(2i/d)` for dimension `i`. Uniform rescaling does `θ_i → θ_i / λ` with a single `λ`. LongRoPE generalizes:

$$
\theta_i' \;=\; \theta_i \,/\, \lambda_i
$$

where each **per-dimension factor** `λ_i` is learned. With `d = 128` RoPE dimensions, that's a 128-dimensional search space.

**Search cost.** Population size **64**, **40 generations**, mutation rate **0.3** — so roughly **64 × 40 = 2560** fitness evaluations in the worst case, each a long-context forward pass. Initial population is seeded from NTK-aware, YaRN, and uniform rescaling — a reasonable prior that stops the first generations from wasting evaluations on garbage. Fitness is a weighted combination of long-context perplexity + NIAH retrieval accuracy on a held-out corpus. Output is an optimized `λ_i` vector per target context (256K, 1M, 2M).

**Why this wins.** A uniform rescale makes every RoPE dimension reach the same effective range; but the high-frequency dimensions (small `i`) saturate much earlier than the low-frequency ones. A per-dimension λ can leave high-frequency dimensions alone while aggressively compressing low-frequency ones. The payoff is a **10× reduction in fine-tune data**: LongRoPE extends LLaMA-2-7B to 2M context with **< 1B tokens of fine-tune data** across two stages (~300M at 256K, ~600M at 2M) — compared to Fu 2024's 5B at 128K. Short-context MMLU and GSM8K stay within 1 point of base.

Complementary to LongRoPE is [[pose-synthesis]]'s **PoSE** — train on 4K-token samples but inject a random position-ID gap `δ ~ Uniform[0, target_ctx − 4K]` between two chunks, so the model learns long-position attention with short-context compute. 4× compute reduction vs full-length, but retrieval-strong / reasoning-weak because PoSE simulates position distribution, not content distribution.

---

## 6. Qwen 2.5-1M and Qwen 3 — the 1M inference stack

[[qwen-long-context-synth]] pushes the frontier from 128K to 1M tokens via a three-leg pipeline that is explicitly a *training-inference split*: train at a reasonable cap, extrapolate at inference.

1. **Gradual continued pretraining** 32K → 128K → 256K (50B + 50B + 100B tokens). Code-repo heavy in stage 2; document-concatenation across topics in stage 3 — explicitly synthesising multi-topic mixed-context sequences.
2. **Synthetic SFT mix generated by Qwen-Max** — multi-needle retrieval (1–8 needles at varied positions), long-doc summarization (50K–200K → summary), RAG-QA (5–20 candidate passages requiring cross-passage fusion), long-code understanding, FILL-IN (masked-segment reconstruction). Filter: answer must reference multiple positions of the source document to avoid shortcut learning.
3. **Dual-Chunk Attention (DCA) at inference** — split long queries into 256K-sized chunks; intra-chunk attention is standard; inter-chunk uses a low-rank formulation that extrapolates RoPE smoothly. This enables 1M-token serving without 1M-token training.

**The eval-curve at 1M tokens for Qwen 2.5-1M.** Qwen-2.5-14B-1M hits **NIAH @ 1M ≈ 100%** and **RULER @ 1M ≈ 85%**. The 15-point NIAH-to-RULER gap is the reasoning-in-a-haystack tax that's become the standard diagnostic. InfiniteBench is strong. Short-context MMLU / GSM8K within 1 point of base Qwen-2.5. Qwen 3 inherits and refines this recipe inside its hybrid-thinking training pipeline.

Gemini ([[gemini-long-context-tricks]]) adds a product-side observation: at million-token scale, *prompt organization and context caching* become part of the eval story. More context does not imply uniform attention quality. The training-side implication: when you synthesize long-context SFT data, varying where in the context the answer-relevant evidence lives (front, middle, back) is itself a diversity axis, not a nuisance variable.

---

## 7. What to remember — the three-lane matrix

| Lane | Knobs | Representative paper | Numbers to memorize |
|---|---|---|---|
| **Position extension** | RoPE base θ; per-dim λ_i; PoSE δ offset | [[long-context-llama3]], [[longrope-data]] | Llama-3: θ = 500K; Fu: θ = 200M; LongRoPE: 2560 search evals → 10× FT data savings |
| **Data (CPT)** | document coherence filter; within-domain length upsample; domain weights | [[prolong]], [[long-context-data-engineering]] | ProLong 20B CPT + 5B SFT; Fu 5B CPT @ 80K; ProLong code × 4 / books × 2 / web × 0.5 |
| **Data (SFT)** | cross-span QA; multi-needle; multi-turn | [[longalign]], [[longmit]], [[longalpaca]] | LongAlign 10k; long-SFT kept at ~0.1% (Llama-3); +5–10 LongBench-Chat from multi-turn |
| **Evaluation** | NIAH heatmap; RULER 13 tasks; BABILong 20 bAbI-in-PG19 | [[needle-in-haystack-data]], [[ruler]], [[babilong]] | RULER 500 ex/task/length; Llama-3.1-70B claimed 128K / effective 64K; Qwen 1M: NIAH 100 / RULER 85 |

The unifying lesson: a long-context capability number is a *claim about a length*, and every such claim has to specify which of the four lanes the number was measured in. "128K context" without stating "NIAH / RULER / BABILong / real" is underspecified.

Play with the axes in **[figures/context-extension.html](figures/context-extension.html)** — the interactive combines a RoPE frequency-band visualization (uniform rescale vs LongRoPE per-dim) with a growing NIAH haystack that reports simulated needle-recovery as context scales. The right panel shows the claimed-vs-effective gap explicitly: crank context from 8K to 1M and watch retrieval stay near 100% while a reasoning-weighted metric collapses around the training-data cap.

---

## Connections and what's next

- **[[longalign]] / [[longalpaca]] / [[longchat]] / [[longmit]]** — early SFT-only recipes; LongAlign's packed-loss correction and LongMIT's multi-turn supplement cover the SFT lane.
- **[[prolong]] / [[long-context-llama3]] / [[long-context-data-engineering]]** — production CPT recipes, each a different point on the data-budget vs curation curve.
- **[[longrope-data]] / [[pose-synthesis]]** — the position-encoding lane; both are orthogonal to the data recipe.
- **[[qwen-long-context-synth]] / [[gemini-long-context-tricks]]** — 1M-context frontier; DCA as training-inference split.
- **[[ruler]] / [[babilong]] / [[needle-in-haystack-data]] / [[longbench]] / [[longembed-synth]]** — the evaluation lanes that make the rest measurable.
- **ch-27** — the synthetic-data design pattern that this chapter specializes to the long-context modality.
- **ch-29 (next)** — the learner-authored pipeline where you generate ~5K instructions end-to-end; long-context synthesis is one of the optional modality specializations.

## Further reading

- [[longalign]] — the paper to read first; it's the most complete end-to-end SFT recipe and the packed-loss derivation is load-bearing.
- [[prolong]] — read §4 (data curation) and the HELMET ablation. The coherence filter vs concatenation ablation is the clearest evidence that volume alone doesn't work.
- [[ruler]] — read Table 5 (13-task configuration) and the effective-context definition. Use it as your standing long-context eval.
- [[longrope-data]] — read §3 (evolutionary search) and the `θ_i' = θ_i / λ_i` equation. It's the cleanest example of position-encoding as an independent lane.
- [[qwen-long-context-synth]] — the frontier 1M recipe; DCA is the training-inference trick that deserves separate study in inference / serving chapters.

## Companion visualization

**[figures/context-extension.html](figures/context-extension.html)** — three interactive panels. **Left:** a RoPE frequency-band chart. Slide the max-context slider and toggle uniform-rescale vs LongRoPE per-dim mode; the frequency bands `θ_i = θ^{-2i/d}` redraw, with the effective "usable-range" band shaded. **Middle:** an NIAH haystack that grows with the context slider; a needle sprite stays at your chosen depth and a simulated recovery-rate counter updates based on distance-from-training-cap and noise. **Right:** the claimed-vs-effective curve — NIAH, RULER, and a BABILong-style reasoning metric plotted on the same axis as context grows, so you can watch the three curves diverge. Use it to fuse §1's eval taxonomy and §5's position-encoding math in one picture.
