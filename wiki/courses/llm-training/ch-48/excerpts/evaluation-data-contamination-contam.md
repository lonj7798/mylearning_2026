<!-- excerpt for ch-48; extract of one primary source. Loci are sections/tables of the arXiv PDF.
     Created 2026-09 (generality revision) from https://arxiv.org/abs/2411.03923 (arXiv v1, 2024-11-06).
-->

# Evaluation data contamination in LLMs: how do we measure it and (when) does it matter?

- **Authors:** Aaditya K. Singh, Muhammed Yusuf Kocyigit, Andrew Poulton, David Esiobu, Maria Lomeli,
  Gergely Szilvasy, Dieuwke Hupkes (UCL; Boston University; Cohere; Meta)
- **Year:** 2024 (arXiv v1 2024-11-06)
- **URL:** https://arxiv.org/abs/2411.03923
- **Source type:** paper
- This is the "Singh et al. (2024)" whose method [[llama-3]] §5.1.4 follows and whose 8-gram setting
  [[tulu-3]] §3.2 cites.

## What the chapter uses

**Estimated performance gain, EPG (§4.1).** For a contamination metric and threshold, EPG is the model's
score on the whole benchmark minus its score on the subset the metric marks clean. It is model-dependent:
the same flagged set has different EPG for different models. A metric is judged by the EPG of what it flags,
which converts "is there overlap" into "does the overlap change the score".

**Four n-gram metrics (§3.1).**
- TOKEN-MATCH (Brown et al. 2020): score = fraction of the evaluation sample's tokens that sit in an n-gram
  found in the pre-training corpus; text lowercased and stripped of punctuation.
- NGRAM-MATCH (Chowdhery et al. 2023): same, counted over n-grams rather than tokens.
- TOKEN-EXTEND (Touvron et al. 2023): match-then-extend with a substitution `skip_budget`; no normalisation.
- LONGEST-MATCH (this paper): fraction of tokens inside the single **longest** matching span, not the union
  of all matches. Introduced so that several unrelated templated spans do not add up to a high score.

**Setup (§3.2–3.3).** 13 benchmarks; Llama 1 at 7B/13B/33B/65B on the Llama 1 corpus; Pythia 1.4B/6.9B/12B on
the Pile. Swept n ∈ {8, 10, 13, 20}, mincount ∈ {1, 5, 10, 20, 100}, skip_budget ∈ {0..5}, and the
contamination threshold.

**Results (§5.2, Figs. 3–4, Table 3).** With the best metric and a per-model, per-benchmark threshold, more
than 50% of samples are flagged for 8 of the 13 benchmarks on the Llama 1 corpus. For Llama 1 65B, EPG is
25 points on BIG-Bench Hard and 18 on HumanEval; HellaSwag, MMLU and PIQA exceed 10. Pythia EPGs span
roughly 2–8. Optimal thresholds sit between 0.05 and 0.40 depending on benchmark (Table 3). Larger Llama
models often extract more gain per percent contaminated than smaller ones (Fig. 4c).

**Hyperparameters (§6).**
- n (§6.1): n > 8 produced false negatives almost across the board. For NGRAM-MATCH on PiQA with Llama 1 65B,
  67.9% of examples have non-zero contamination at n = 8 but only 33.8% at n = 10, and the extra examples do
  carry EPG.
- mincount (§6.3): requiring a match to occur more than once in the corpus also produced false negatives.
- skip_budget (§6.2): little effect; Spearman ρ between skip_budget 0 and 5 orderings is 0.978–0.997
  across benchmarks (Fig. 7a).

**Recommendations (§7.3).** (1) Run several metrics; if only one, use LONGEST-MATCH. (2) Select thresholds
per benchmark and per model with z-scores and ConTAM plots; this is post-hoc and cheap. (3) Report percent
contaminated, EPG, and the selected threshold. Manual inspection of flagged samples is preferred on top.

**Stated conclusion (§7.2).** The impact of contamination has been underestimated in prominent model
releases, which the authors attribute to too-strict metric settings (large n, mincount > 1) producing false
negatives.

## Verification
- Read on 2026-09-15 against the arXiv v1 PDF, §§1–8 and Tables 1–3, Figs. 3–7.
- Not reported: results on models later than Llama 1 and Pythia; post-training-stage contamination;
  paraphrase-level contamination (n-gram metrics only); EPG for agentic benchmarks.
