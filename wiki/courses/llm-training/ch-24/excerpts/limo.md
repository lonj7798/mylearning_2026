---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/limo.md
source_url: https://arxiv.org/abs/2502.03387
created_at: "2026-04-23"
---

# Excerpt: LIMO — the Less-Is-More Reasoning Hypothesis and hand-curation

**Source library:** `wiki/raw-data/llm-training/papers/limo.md`
**Paper:** Ye et al. 2025, "LIMO: Less is More for Reasoning"

---

## Why this source anchors ch-24 §4

LIMO formalizes what s1 demonstrates: the "Less-Is-More Reasoning Hypothesis" — if the base model already contains domain knowledge from pretraining, a small number of high-quality post-training traces can *activate* strong reasoning. The paper's 817-sample set is hand-curated rather than semi-automatically filtered (s1), so LIMO is the paper to cite when the argument depends on careful human judgment over the traces.

---

## The hypothesis, stated

From the source (§Key Contributions):

> Formalizes the **Less-Is-More Reasoning Hypothesis**: strong latent knowledge from pretraining plus high-quality demonstrations are the two prerequisites.

Two claims packed in:

1. **Latent knowledge** — the base model must have absorbed the domain during pretraining. LIMO does not work on weak bases.
2. **Demonstration quality** — the demonstrations must show the *right cognitive template*: reflective structure, branching, verification. Flat traces do not activate anything.

Ch-24 §4 treats this as the chapter's central tension: scale-is-all (OMI-2) vs activation-via-quality (LIMO). The resolution in ch-24 §8 is "both are right, for different verifier regimes."

---

## The 817 — what got in, what got cut

From the source (§Synthesis pipeline):

- **Seed pool**: competition math, MATH, GSM8K-hard, physics olympiad.
- **Question selection**: keep only problems that strong baselines still find hard — multi-step reasoning, not recall.
- **Solution generation**: multiple candidate traces from strong reasoning models and human editing.
- **Quality scoring**:
  - correctness of final answer vs gold;
  - presence of **self-verification / re-checking segments**;
  - **branching / backtracking markers** (non-linear reasoning structure);
  - **fine-grained step granularity** (not outline-only answers).
- **Manual curation**: hand-filter to remove lucky-guess-correct traces and traces with subtly broken intermediate logic.

The last bullet is the labor-expensive one. Authors explicitly *reject* traces where the final answer is correct but an intermediate step is subtly wrong — the same failure mode OpenMathInstruct-1 catalogued as its chief limitation (ch-24 §2). Hand review catches what symbolic equivalence misses.

**817** is the resulting count. Some traces reach many thousands of tokens; the average is long-reflective.

---

## The evaluation delta

From the source (§Quality evaluation):

> Reaches **63.3% AIME24 and 95.6% MATH500**.

Context for ch-24's Panel-2 table:

| Recipe | AIME24 | MATH500 | Samples |
|---|---|---|---|
| Qwen2.5-32B-Instruct base | ~17 | ~84 | 0 |
| Random 1K (s1 ablation) | ~24 | ~86 | 1000 |
| s1K curated | 56.7 | 93.0 | 1000 |
| **LIMO hand-curated** | **63.3** | **95.6** | **817** |
| OpenMathInstruct-2 on 8B | ~40 on AIME24 | 67.8 MATH (not MATH500) | 14M |

The LIMO → s1 gap (63.3 vs 56.7 on AIME24) is attributed by the LIMO authors to **curator effort**: hand review removes noise that the s1 semi-automated pipeline retains. This is a real ablation; ch-24 §4 flags it as the main reason the "trace quality > count" claim is not a tautology.

---

## The ablation that makes the hypothesis falsifiable

From the source (§Quality evaluation):

> Ablations show that random or low-quality samples do not reproduce the effect; the quality gap is not reducible to volume.

If you could match LIMO's numbers by taking 817 random samples from the same seed pool, "Less-Is-More Reasoning" would be trivial (pretraining did all the work; SFT is noise). The ablation shows that random-817 scores far below curated-817 — so the curation bit is doing real work, even if pretraining does most of the work.

---

## Caveats

From the source (§Risks + gotchas):

- **Curator subjectivity**: exact reproduction depends on matching the curation policy. The paper releases the 817 samples but the *policy* that selected them is not a pipeline.
- **Base-model dependence**: weak bases do not activate from a tiny dataset. Deliberately trying LIMO on a 7B non-reasoning-tuned base mostly fails.
- **Benchmark overlap**: competition-style sources make contamination auditing non-trivial. AIME24 traces may overlap with AIME-previous-years pretraining.

The base-model dependence is the practical caveat for ch-24 §8's guidance: LIMO/s1-style curation is not a substitute for having a strong base; it is an *unlock* on top of one.

---

## Why this sits above s1 on the benchmarks

From the source (§Modality-specific technical details):

> Long-CoT traces, with some examples reaching many thousands of tokens. Reflective long-CoT with verification and backtracking, structurally similar to o1 / R1 style traces.

The style difference is the likely explanation. s1 uses Gemini-generated traces (strong but not explicitly o1-style-reflective). LIMO accepts only traces with explicit reflection and backtracking markers. For AIME24-hard problems, the reflective template is worth 6-7 absolute points at matched dataset size.

---

## Connections

- [[excerpts/s1]] — the twin paper; same hypothesis, semi-automated filter instead of hand curation.
- [[excerpts/openmathinstruct-2]] — the volume-scaled contrast; OMI-2's short-CoT ceiling is why LIMO can win with 1% of the data.
- [[excerpts/rstar-math]] — MCTS as an alternative route to reflective traces without hand curation.
- [[ch-24]] §4 (long-CoT small-N), §8 (when to curate vs when to scale).
