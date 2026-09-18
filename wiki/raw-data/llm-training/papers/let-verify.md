<!-- scope: redirect card — duplicate of [[lets-verify]] (same artifact: Lightman et al., arXiv:2305.20050)
     see-also: [[lets-verify]], [[prm800k]]
-->

# Let's Verify Step by Step (redirect card)

> **Duplicate card.** This file and [[lets-verify]] described the same artifact: "Let's Verify Step by Step",
> Lightman et al., OpenAI, arXiv:2305.20050 (v1, 31 May 2023). The verified extract lives in
> [[lets-verify]]. A second full extract of the same paper, organized figure by figure, is [[prm800k]].
> The file name is kept because other pages link to `[[let-verify]]`; do not cite it for numbers.

- **Exact title:** Let's Verify Step by Step
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, et al. (OpenAI; ten authors, last-listed Karl Cobbe)
- **Year:** 2023 (arXiv v1 2023-05-31; ICLR 2024)
- **URL:** https://arxiv.org/abs/2305.20050
- **Source type:** paper

## Where to find the content
| Topic | Card |
|---|---|
| Headline PRM vs ORM vs majority-voting numbers, PRM800K size and label schema, recipe ledger | [[lets-verify]] |
| Figure-by-figure and table-by-table breakdown, difficulty quintiles, scoring-strategy comparison | [[prm800k]] |
| Automatic step labels that replace the human labels used here | [[math-shepherd]], [[omegaprm]] |
| The outcome-supervised verifier baseline this paper builds on | [[training-verifiers-to-solve-math-word-problems]] |

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2305.20050 (v1, the only arXiv version).
- Corrections to the previous card version:
  - The card was a second full extract of the artifact already covered by [[lets-verify]] and [[prm800k]]; it is reduced to a redirect so recipe and evaluation numbers have one home.
  - "**Year:** 2023 (OpenAI)" is correct; the sibling card's "2024 / OpenReview" header was the erroneous one and has been fixed there.
  - "78.2% vs 72.4% on a MATH test subset at N=1860" → correct, and the third baseline is majority voting at 69.6% (§3, Fig. 3). Retained in [[lets-verify]].
  - "Figure 1 (MATH test-set accuracy vs N)" and "Figure 3 (calibration)" → Fig. 1 is the labeling interface and Fig. 3 is the best-of-N curves; the paper reports no calibration figure.
- Removed as unsupported by the source:
  - "On MATH, PRM + best-of-N gives a ≥10-pt absolute lift over ORM at N=64" — no such number in the paper.
  - "PRM scores every step (1 forward per step) … PRM is ~L× more expensive at inference where L is step count" — §2.6 states one forward pass over the whole solution yields all step predictions.
  - "Figure 6 (active learning): active-PRM reaches the same quality as random with 38% of the labels" — Fig. 6 (App. G) is the difficulty-quintile breakdown; the active-learning estimate is 2.6× data efficiency (§4.2, Fig. 4a).
  - "Figure 3 (calibration): PRM is better calibrated per-step than ORM on full-solution" — the paper reports no calibration measurement.
  - "Active learning: rank unlabeled steps by model uncertainty (entropy on the good/bad head)" — selection is by the current PRM's score on wrong-answer solutions, not by entropy (§2.4, §4.2).
  - "PRM training: binary classifier per step on (prefix, step) → {good, bad}" — the PRM predicts positive/negative/neutral, with neutral counted as positive at scoring time (§2.6, App. F.2).
  - "Base generator: GPT-4 (prompted) and a fine-tuned variant; base PRM and ORM are small-scale fine-tunes" — all large-scale models are fine-tuned from base GPT-4 (§2.2).
  - "Introduced active learning for PRM: route uncertain steps to labelers, yielding ~2.6× label efficiency" — the 2.6× figure comes from the small-scale synthetic ablation, not from the human collection (§4.2).
  - "Provided the credit-assignment = reasoning step operational definition that now underlies all process-reward work" and "any new process-reward method reports numbers on it" — no source given for either claim.
  - "The ORM vs PRM at high N result motivates Best-of-N selection as a deployment strategy — see [[best-of-n]]"; "DeepSeek-R1 … Let-Verify is the paper they ablate against" — R1 lists PRMs among unsuccessful attempts and cites this paper, but reports no ablation against it.
