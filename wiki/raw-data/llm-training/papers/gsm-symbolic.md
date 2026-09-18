<!-- scope: GSM-Symbolic (arXiv:2410.05229, Oct 2024; ICLR 2025) — symbolic templates over 100 GSM8K test questions, the 50-set score distribution, clause-count variants, and the GSM-NoOp irrelevant-clause set
     see-also: [[gsm1k]], [[swe-bench-illusion]], [[reasoning-or-reciting]]
-->

# GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models
- **Core Insight:** Re-instantiating the same 100 GSM8K test questions from symbolic templates produces a score distribution rather than one number — Gemma2-9B's worst-to-best gap across 50 sets exceeds 12% and Phi-3.5-mini's is about 15% — and for 21 of 25 models the score on the 100 original questions lies more than one standard deviation from the centre of that distribution, usually above it (§4.1).
- **Guideline:** When a math score is used as evidence of reasoning, report the mean and standard deviation over several instantiations of the same items plus a variant that adds one inconsequential clause, because names, numbers, clause count, and irrelevant clauses each move the score under an unchanged reasoning requirement (§4.2–§4.4). Otherwise report the single score as specific to those item instances.
- **Authors:** Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar (Apple; Hooman Shahrokhi at Washington State University)
- **Year:** 2024 (arXiv v1 2024-10-07; v2 2025-08-27; ICLR 2025)
- **URL:** https://arxiv.org/abs/2410.05229
- **Source type:** paper
- **Relevant topics:** perturbation evaluation, benchmark contamination, mathematical reasoning, distractor robustness, evaluation variance, GSM8K

## Abstract
Reported GSM8K results may not be reliable indicators of mathematical reasoning. The authors build GSM-Symbolic from symbolic templates that generate many instantiations of the same question, so that evaluation produces a distribution instead of a point. Models show noticeable variance across instantiations of one question, performance declines when only numerical values change, and performance deteriorates further as the number of clauses grows. Adding a single clause that appears relevant but does not enter the solution causes drops of up to 65% across state-of-the-art models. The authors hypothesize that the models replicate reasoning steps seen in training data rather than performing formal logical reasoning.

## Key Contributions
- A template-based generator over 100 GSM8K test questions with typed variables, sampling domains, and correctness conditions, evaluated on 25 open and closed models (§1, §3.1).
- Reporting the score as a distribution over 50 instantiated sets, and locating the original GSM8K items inside it (§4.1).
- Separating the effect of changing proper names from changing numerical values (§4.2).
- Difficulty variants by clause count: GSM-M1 (one clause removed), GSM-P1 and GSM-P2 (one and two clauses added) (§4.3).
- GSM-NoOp: one added clause that is topically related and carries no operation, plus a shot-source ablation that tests whether in-context examples recover the loss (§4.4).

## Key Figures/Tables to Study
- Fig. 1 (a GSM8K question and its template); Fig. 2 (8-shot score distributions over the 50 sets, with the original-items score as a dashed line); Fig. 3 (GSM8K to GSM-Symbolic drop); Fig. 4 (names vs numbers vs both); Fig. 6 (M1 → Symbolic → P1 → P2 shift and spread); Fig. 7 (a GSM-NoOp example); Fig. 8a–c (NoOp drops and the two shot-source ablations); Table 1, Appendix A.2 (full 8-shot results for all models and variants).

## Technical Details
**Template construction (§3.1)**
- A GSM8K test question is annotated with variables, their domains, and conditions that keep the question and answer valid; a common condition is divisibility so that the answer is a whole number.
- Automated checks verify that no original variable value survives in the template, that the original values satisfy the conditions, and that the answer for the original values matches the original question. Ten random samples per template are reviewed manually, and after evaluation any question that no two models answer correctly is reviewed again.
- Numerical ranges are chosen close to the GSM8K test ranges, because the target is logical reasoning rather than arithmetic; Appendix A.6 checks that models keep their arithmetic accuracy over the expanded ranges.

**Evaluation setup (§3.2)**
- More than 20 open models from 2B to 27B, plus GPT-4o-mini, GPT-4o, o1-mini, and o1-preview; 25 models in Table 1. Nearly 500 total evaluations.
- 100 templates × 50 samples = 5,000 examples per benchmark, organized as 50 datasets of 100 examples, each example a mutation of one of the 100 originals.
- 8-shot chain-of-thought prompting with greedy decoding. In preliminary experiments the number of shots did not significantly change performance or conclusions.

**Results**
- Variance across the 50 sets (§4.1): Gemma2-9B's worst-to-best gap is more than 12%; Phi-3.5-mini's is around 15%. The only differences between instances are names and values.
- Position of the original items (§4.1): for 21 of 25 models the original-question score is more than one standard deviation from the centre of the GSM-Symbolic distribution, frequently on the right. The authors offer data contamination as one explanation (Interpretation). Models with a large drop include Gemma2-9B, Phi-3, Phi-3.5, and Mathstral-7B; Llama3-8b and GPT-4o sit near the centre.
- Names vs numbers (§4.2, Fig. 4): variance is lower when only proper names change than when numbers change; with names changed the original GSM8K score sits much closer to the centre; changing both shifts the mean furthest left.
- Clause count (§4.3, Fig. 6): from GSM-M1 to GSM-Symbolic to GSM-P1 to GSM-P2, the distribution shifts left and the variance increases, and the rate of drop grows with difficulty. Footnote 2 states that adding or removing a clause does not necessarily change the number of required reasoning steps by exactly one.
- GSM-NoOp (§4.4, Fig. 8a): accuracy declines across all tested models, with Phi-3-mini experiencing over a 65% drop and o1-preview also declining. The authors report a common failure pattern of converting statements into operations, for example reading "discount" as multiplication regardless of context.
- Shot-source ablation (§4.4, Fig. 8b–c): NoOp-Symb supplies 8 shots of the same question drawn from GSM-Symbolic, and NoOp-NoOp supplies 8 shots from different GSM-NoOp questions. The drop is not recovered in either case; for Phi-3-medium-128k-instruct the bars read 87.3 (GSM8K), 82.5 (Symbolic), 29.4 (NoOp, GSM shots), 30.2 (NoOp-Symb), 22.6 (NoOp-NoOp). For Llama3-8b-instruct: 76.0, 74.6, 18.6, 19.6, 19.2. Fig. 8c shows some weaker models improving under NoOp-Symb, for example Mistral-7b-v0.1 at 44.5, 41.1, 16.2, 62.5, 14.5.

**Table 1 rows used by the chapters (GSM8K Full, GSM8K 100, Symbolic-M1, Symbolic, Symbolic-P1, Symbolic-P2, Symbolic-NoOp; standard deviations in parentheses)**
| Model | GSM8K (Full) | GSM8K (100) | Symbolic-M1 | Symbolic | Symbolic-P1 | Symbolic-P2 | Symbolic-NoOp |
|---|---|---|---|---|---|---|---|
| Gemma2-9b | 85.3 | 87.0 | 71.2 (± 2.81) | 79.1 (± 2.99) | 44.0 (± 5.69) | 41.8 (± 6.00) | 22.3 (± 5.11) |
| Gemma2-9b-it | 85.3 | 87.0 | 84.4 (± 2.36) | 79.1 (± 2.99) | 68.1 (± 4.77) | 41.8 (± 6.00) | 22.3 (± 5.11) |
| Gemma2-27b-it | 89.7 | 92.0 | 90.2 (± 1.86) | 88.3 (± 2.56) | 80.7 (± 4.07) | 63.4 (± 4.14) | 30.0 (± 3.39) |
| Phi-3-mini-128k-instruct | 83.7 | 85.0 | 85.9 (± 2.44) | 80.7 (± 2.94) | 63.4 (± 5.63) | 37.5 (± 5.76) | 18.0 (± 3.83) |
| Phi-3-medium-128k-instruct | 87.3 | 89.0 | 89.6 (± 1.65) | 82.5 (± 2.86) | 75.8 (± 3.89) | 53.1 (± 4.80) | 29.4 (± 4.18) |
| Phi-3.5-mini-instruct | 84.9 | 88.0 | 87.6 (± 1.98) | 82.1 (± 3.38) | 64.8 (± 5.43) | 44.8 (± 6.32) | 22.4 (± 4.03) |
| Mathstral-7b-v0.1 | 80.1 | 80.0 | 82.9 (± 2.87) | 74.0 (± 3.49) | 57.4 (± 5.20) | 35.5 (± 5.07) | 20.4 (± 3.58) |
| Llama3-8b-instruct | 76.0 | 74.0 | 79.5 (± 3.62) | 74.6 (± 2.94) | 53.8 (± 4.54) | 28.3 (± 4.37) | 18.6 (± 3.86) |
| GPT-4o-mini | 94.2 | 95.0 | 92.5 (± 1.63) | 91.7 (± 2.02) | 81.1 (± 3.05) | 72.4 (± 4.57) | 54.1 (± 3.85) |
| GPT-4o | 95.2 | 95.0 | 94.4 (± 1.62) | 94.9 (± 1.87) | 93.9 (± 2.59) | 88.0 (± 3.43) | 63.1 (± 4.53) |
| o1-mini | 95.1 | 93.0 | 94.9 (± 1.49) | 94.5 (± 1.58) | 94.3 (± 2.57) | 89.1 (± 3.56) | 66.0 (± 4.60) |
| o1-preview | 94.9 | 96.0 | 93.6 (± 1.68) | 92.7 (± 1.82) | 95.4 (± 1.72) | 94.0 (± 2.38) | 77.4 (± 3.84) |

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GSM-Symbolic audit, 25 models | 2B–27B open plus GPT-4o-mini, GPT-4o, o1-mini, o1-preview | eval-gate | prompt; decoding | 8-shot chain of thought; greedy | arXiv:2410.05229v2 §3.2 | verified 2026-09-18 | §3.2: the number of shots did not significantly change performance or conclusions in preliminary experiments |
| GSM-Symbolic audit | same | eval-gate | templates; samples; sets | 100 templates × 50 samples = 5,000 examples, reported as 50 datasets of 100 | §3.2 | verified 2026-09-18 | §4.1: the reported quantity is the 50-set distribution, not one score (Fig. 2, Table 1) |
| GSM-Symbolic audit | same | eval-gate | numeric ranges | kept close to the GSM8K test ranges | §3.1, App. A.6 | verified 2026-09-18 | App. A.6: models retain arithmetic accuracy over the expanded ranges |
| GSM-Symbolic audit | same | eval-gate | template QA | automated value/condition/answer checks; 10 random samples per template reviewed manually; questions no two models solve reviewed again | §3.1 | verified 2026-09-18 | no ablation reported |
| GSM-NoOp audit | same | eval-gate | shot source | three settings: original GSM8K shots, NoOp-Symb (8 shots of the same question from GSM-Symbolic), NoOp-NoOp (8 shots from different GSM-NoOp questions) | §4.4, Fig. 8b–c | verified 2026-09-18 | §4.4: the drop is not recovered under either alternative shot source |

## Findings relevant to generality
- Score variance under an unchanged reasoning requirement bounds what a single-run difference on GSM8K can mean: the same model's 50 sets span more than 12 points for Gemma2-9B (§4.1).
- The gap between the original items and the perturbed distribution is a measurement of item-specific advantage; contamination is one of several explanations the authors name and is not separately measured (§4.1, Interpretation).
- GSM-NoOp shows that the failure is not fixed by in-context demonstrations of the same question with its reasoning chain (§4.4, Fig. 8b), which distinguishes it from a prompt-format failure.
- Scope: grade-school arithmetic word problems only; no transfer to competition mathematics, code, or other domains is tested.

## Connections
- [[gsm1k]] — the complementary audit: new items matched in difficulty instead of re-instantiated items.
- [[swe-bench-illusion]] — benchmark-specific advantage measured through memorization probes in agentic coding.

## Verification
- Created on 2026-09-18 from https://arxiv.org/abs/2410.05229 (arXiv v2, 2025-08-27, ICLR 2025 camera-ready).
- Corrections to the previous card version: none (no card existed; ch-24, ch-47a, ch-50, and ch-53 carried chapter excerpts instead).
- Removed as unsupported by the source: none.
- Chapter claims not found in the source:
  - Fig. 8a bar values. ch-24 and ch-53 quote per-model bars (o1-preview −17.5, Phi-3-medium-128k −57.8, Gemma2-9b and Gemma2-9b-it −63.0, Phi-3-small-128k −64.0, Phi-3-mini-128k −65.7) and ch-47a notes that they do not match Table 1 differences. The paper does not state the baseline of the bars. Each bar equals GSM8K (Full) minus Symbolic-NoOp in percentage points (85.3 − 22.3 = 63.0; 83.7 − 18.0 = 65.7; 88.5 − 24.5 = 64.0; 87.3 − 29.4 = 57.9 against a printed 57.8) (**derived** here, 2026-09-18, not stated by the source). Quote Table 1 for absolute scores.
  - "Per-set standard deviations of 2–6 points" (ch-50 question 4) holds only across variant columns. In Table 1 the Symbolic column standard deviations run 1.58 to 4.43; the 6-point end comes from the Symbolic-P1 and P2 columns.
  - "Supplying eight shots of the same question from GSM-Symbolic does not recover performance beyond one standard deviation" (ch-53 excerpt). §4.4 and Fig. 8b state that the drop is not recovered and give the bar values; the source does not express the residual in standard deviations.
- Not reported by the source: the identity of the 100 source questions; any per-template score breakdown; results outside grade-school arithmetic.
