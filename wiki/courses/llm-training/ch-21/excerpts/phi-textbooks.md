---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/phi-textbooks.md
source_url: https://arxiv.org/abs/2306.11644
created_at: "2026-04-23"
---

# Excerpt: Phi-1 — "Textbooks Are All You Need" and what it actually means

**Source library:** `wiki/raw-data/llm-training/papers/phi-textbooks.md`
**Authors:** Gunasekar, Zhang, Aneja, Mendes, Del Giorno, Gopi, Javaheripi, Kauffmann, de Rosa, Saarikivi, Salim, Shah, Behl, Wang, Bubeck, Eldan, Kalai, Lee, Li (Microsoft Research) — 2023.

---

## Why this source anchors ch-21 §4

Phi-1 is the origin of the textbook-synthesis line. Ch-21 §4 tells the story of all four Phi generations; this excerpt grounds the first one — both the actual contribution and the things the title overstates.

From the source's core insight:

> Pretraining loss is *quality*-bounded long before it is *quantity*-bounded; swapping a noisy web crawl for a small "textbook-quality" corpus (filtered real + GPT-3.5-synthesized textbooks + exercises) yields a 1.3B model that rivals models ~10× bigger trained on ~100× more tokens.

The TBAYN marketing has eaten the paper, so this excerpt goes back to the recipe itself.

---

## The Phi-1 recipe, reconstructed

From the source's Synthesis Pipeline section:

**Stage 1 — filtered web (6B tokens).**
- Source: The Stack (deduplicated) + StackOverflow.
- Classifier: a random forest on a small LM's embedding features, trained on a hand-labeled seed of "educational vs non-educational" Python snippets.
- Output: keep the top-scoring slice. ~6B tokens survive.

**Stage 2 — synthetic textbooks (~1B tokens).**
- Teacher: GPT-3.5 (text-davinci-003 era).
- Prompt structure: topic + target audience + requested exposition style.
- Topic list: "breadth of Python stdlib, control flow, algorithms, data processing."
- This is the first proto-taxonomy: a flat topic list, ~thousand entries, hand-curated.

**Stage 3 — synthetic exercises (~180M tokens, SFT).**
- Teacher: GPT-3.5.
- Output shape: `<problem, solution>` pairs.
- Target: topics not well covered in the filtered web (gap filling against the classifier's output).

**Stage 4 — decontamination + dedup.**
- n-gram match against HumanEval / MBPP to remove direct overlaps.
- Standard dedup.
- Pedagogical classifier re-ranking.

**Total** ≈ 7B pretraining tokens + 180M SFT tokens. A 1.3B-parameter Transformer trains on this in 4 days × 8 × A100 = ~800 GPU-hours.

---

## The two-axis data curation idea

From the source's Key Contributions:

> Introduced a **two-axis data curation strategy**: filter real web for pedagogical density, synthesize the rest.

This is the phrase that generalizes. Every Phi generation — and every open reproduction — inherits the same two axes:

- **Axis 1: filter.** Train a classifier (or use a prompt-based classifier, or use perplexity thresholds from a reference model) to score the web for pedagogical density. Keep the top slice.
- **Axis 2: synthesize.** Generate supplementary content in areas the filter undercovers.

Phi-1 used a random-forest classifier because in 2023 it was cheap and interpretable. Phi-3 replaces this with "filtered by a larger LM." Cosmopedia replaces it with "clustered into 145 topic groups, keep the educational clusters." The axis is universal; the implementation varies.

---

## The headline numbers — and the contamination asterisk

From the source:

> **HumanEval:** 50.6% pass@1 (1.3B model, vs ~34% for StarCoder-15B).
> **MBPP:** 55.5% pass@1.
> Ablations show the jump is driven primarily by the synthetic textbook + exercise data, not by model architecture.

And the risk section:

> **HumanEval contamination risk:** later analyses flagged non-trivial overlap between synthetic exercises and HumanEval-style prompts — a perpetual critique of the Phi line.

The honest reading:

- The filter + synthesize recipe definitely helps — the ablation against raw web is unambiguous.
- The *magnitude* of the help — specifically the 50.6% HumanEval number — is inflated by some amount of contamination between the synthetic exercise corpus and the benchmark. Nobody has published a clean decontamination-audited Phi-1 replication that reaches the original number, and the open reproduction ([[cosmopedia]]) lags Phi-1.5 on several tasks.
- The recipe transfers — Cosmopedia, Nemotron's code stage, and Phi-3/4 all show improvement from the same two axes.

Ch-21 §4 presents the headline number with the contamination caveat attached; readers should not propagate the 50.6% as a clean measurement.

---

## TBAYN — what the title actually supports

The title "Textbooks Are All You Need" is stronger than the experiments. A faithful reading of the source:

- **Supported:** for a 1.3B Python-focused model, ~7B tokens of curated + synthetic data is sufficient to compete with 15B-parameter code models trained on much more data.
- **Supported:** filtering + synthesizing pedagogical content is a better data-quality axis than raw web at the same token budget.
- **Not supported by this paper:** that textbooks are sufficient for general reasoning (that is [[phi-1-5]]'s claim and is partially supported).
- **Not supported by this paper:** that textbooks are sufficient for frontier-scale models (Phi-3/4 quietly re-introduce filtered web in Phase 1; they do not run textbook-only).
- **Not supported:** the specific 50.6% number, given the contamination issues.

Ch-21 §4 quotes the paper's title and then immediately qualifies it. The quote is "Textbooks Are All You Need"; the qualification is "for narrow-domain small models with generous contamination assumptions." This is the honest phrasing.

---

## The teacher ceiling

From the source's Risks + gotchas:

> **Teacher-model ceiling:** the student inherits GPT-3.5's code-style tics and mistakes.

Phi-1 is a distillation of GPT-3.5's Python mental model into a 1.3B-parameter package. If GPT-3.5 had a systematic bug in its understanding of, say, Python 3.12 pattern matching (which was new at the time), Phi-1 would inherit it. The paper acknowledges this but does not measure it.

This is the same ceiling that ch-21 §6 argues applies to GLAN and Nemotron: top-down synthesis of any kind produces a student whose upper bound is the teacher. The difference across papers is only where the teacher's bias enters (tree structure vs task families vs prompt templates).

---

## Why Phi-1 matters for ch-21 even though it is a pretraining paper

Ch-21 is labeled "synthetic track" and covers both pretraining and post-training synthesis. Phi-1's position:

1. **Chronologically first** — published before GLAN and before Nemotron; the synthesis-as-pretraining idea preceded synthesis-as-instruction-data.
2. **Establishes the filter + synthesize template** that every later Phi and every Cosmopedia inherits.
3. **Establishes the teacher-ceiling argument** that ch-21 §6 generalizes.
4. **Establishes the contamination-audit burden** that applies to every closed-data synthetic pipeline.

Without Phi-1 there is no Phi-1.5, no 20K-topic list, and therefore no direct precursor to GLAN's tree.

---

## Connections

- [[excerpts/phi-1-5]] — successor; extends the recipe to 20K topics and common-sense reasoning.
- [[excerpts/phi-4]] — current Phi generation; 50 synthetic categories, pivotal-token DPO.
- [[excerpts/cosmopedia]] — open reproduction; includes the contamination analysis Phi-1 did not publish.
- [[excerpts/glan]] — downstream formalization of the taxonomy idea in the instruction-data direction.
- [[ch-20]] — distillation-as-data; Phi-1 is distillation-as-pretraining.
- [[ch-21]] §4.
