<!-- scope: Big-Math (arXiv:2502.17387) — 251K-problem RL math set built by filtering open datasets for uniquely verifiable, open-ended, closed-form answers; MC→open reformulation; difficulty by Llama-3.1-8B pass rate
     deps: [[numina-math]]
     see-also: [[deepmath-103k]], [[reasoning-gym]], [[math-verify]], [[verifier-pitfalls]], [[dapo]], [[polaris]], [[rl-design-guide]]
-->

# Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models
- **Core Insight:** An RL math set is defined by three properties of each item, not by its source: a uniquely verifiable solution, an open-ended formulation that cannot be solved by guessing, and a closed-form answer that a program can check (§1); filtering 643,374 open problems for these properties leaves 251,122 (Table 1), and 117,018 matches in the pool were multiple choice (Table 2).
- **Guideline:** When assembling an RL prompt set from existing math corpora, remove multiple-choice, yes/no, true/false, multi-part and proof items (or rewrite multiple-choice items as open-ended questions) and record a per-problem pass rate from N rollouts of a reference policy, because guessable or uncheckable items give reward without reasoning, and the authors' 64-rollout Llama-3.1-8B pass rate puts 36.50% of the final set below a 20% success rate, which lets you select a difficulty window later (§2.2.2, §2.3, §3.1).
- **Authors:** Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, Nick Haber (affiliations not recorded in the pages read)
- **Year:** 2025 (arXiv v1, 24 Feb 2025)
- **URL:** https://arxiv.org/abs/2502.17387
- **Source type:** paper (dataset)
- **Relevant topics:** RL prompt-set construction, answer verifiability, multiple-choice reformulation, decontamination, difficulty measurement by pass rate

## Summary
Big-Math collects open math datasets (NuminaMath, HARP, Omni-MATH, MATH, GSM8K and others; Table 1, 643,374 problems before filtering), applies source-specific cleaning (to 463,426), then source-agnostic filters for answer extractability, item type, solvability and contamination. Multiple-choice items are rewritten as open-ended questions by a four-step Llama-3.1-405B pipeline and added back as Big-Math-Reformulated (47,010 items). The final set has 251,122 problems.

## Key Contributions
- The three desiderata for RL math data: uniquely verifiable, open-ended, closed-form (§1).
- A filter pipeline whose model-based filters were tuned with human-in-the-loop rounds to at least 90% precision and recall (§1) and a recall target above 98% (§2.2).
- Big-Math-Reformulated: multiple-choice → open-ended rewrites, 47,010 items (§2.3, Table 3).
- Per-problem difficulty from 64 Llama-3.1-8B rollouts (§3.1).

## Key Figures/Tables to Study
- Table 1 (sources and counts), Table 2 (filter matches), Table 3 (reformulation funnel), §3.1 difficulty quintiles.

## Technical Details
**Cleaning (§2.2.1).** HARP: Asymptote-figure problems removed (625). Omni-MATH: author/score text removed. NuminaMath: MinHashLSH deduplication and `\boxed{}` answer extraction. Result: 463,426 problems.

**Source-agnostic filters (§2.2.2).** Answer filter (drop items without an extractable answer, e.g., no `\boxed{}`); regex and Llama-3.1-70B classifiers for multiple choice, multi-part, proof and yes/no/true-false items; decontamination (4,229 removed). Table 2 lists 117,018 multiple-choice matches and 40,509 proof matches. Solvability filter: keep an item if Llama-3.1-8B (64 rollouts) or Llama-3.1-405B (5–8 rollouts) produces the reference answer at least once; HARP, Omni-MATH, MATH and GSM8K were exempt. This filter removes items whose reference answer is likely wrong or whose format cannot be matched.

**Filter tuning (§2.2).** Each iteration inspected 100 positive and 100 negative examples; the multiple-choice filter took 5–8 rounds. Final per-filter precision and recall are not reported.

**Reformulation (§2.3).** Steps: key-information extraction → reformulation → judgement → verification (Llama-3.1-405B). 88,983 passed judgement and verification; solvability filtering left 48,698; the full pipeline left 47,010 (Table 3). Composition: cn_k12 35,211; olympiads 9,019; amc_aime 1,878; aops_forum 853; MATH 48; Orca-Math 1.

**Difficulty (§3.1).** Quintiles of Llama-3.1-8B success rate, easiest to hardest: 71,926 (28.64%), 30,533 (12.16%), 25,763 (10.26%), 31,249 (12.44%), 91,647 (36.50%); the hardest bin is success rate below 20%. The reformulated subset has 34.44% in the hardest quintile.

## Verification
- Read the arXiv HTML v1 (first 100K of 149K characters) and the abstract page on 2026-11-24. Counts above come from Tables 1–3 and §2–3.
- Not read: the remaining appendix text; no RL training ablation on Big-Math is cited here, so this card gives no measured effect of these filters on RL accuracy.
- The 204,112 non-reformulated count is arithmetic (251,122 − 47,010), not a stated figure.
