<!-- scope: Iterative Reasoning Preference Optimization — DPO plus an NLL term on the chosen CoT, iterated over rounds
     deps: [[dpo]]
     see-also: [[orpo]], [[self-rewarding-lm]], [[spin]], [[likelihood-displacement]], [[grpo]]
-->

# Iterative Reasoning Preference Optimization
- **Core Insight:** Training Llama-2-70b-chat on correct-vs-incorrect chain-of-thought pairs with plain DPO gives 61.8% on GSM8K, below the 63.5% of SFT on gold CoTs; adding an NLL term on the chosen CoT and iterating four rounds gives 81.6% from a 55.6% zero-shot baseline (Table 1).
- **Guideline:** When preference pairs are built from correct versus incorrect chains of thought, add α·NLL on the chosen chain to the DPO loss with α = 1 and β = 0.1, because without it the chosen-sequence log-probability falls over training and single-iteration accuracy drops from 73.1% to 61.8% (§3.1, Figure 3, Table 1).
- **Authors:** Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, Jason Weston (FAIR at Meta; New York University)
- **Year:** 2024 (arXiv v1 2024-04; v3 2024-06-26)
- **URL:** https://arxiv.org/abs/2404.19733
- **Source type:** paper
- **Relevant topics:** iterative preference optimization, reasoning alignment, DPO+NLL, verifier-labeled pairs, negative-as-gradient

## Abstract
The paper introduces an iterative scheme that builds preference pairs between competing chain-of-thought
candidates and trains on them with a DPO loss plus a negative log-likelihood term on the winning chain.
Correctness of the final answer, checked against the dataset's gold label, is the only supervision. Starting
from Llama-2-70b-chat and using only training-set prompts, accuracy rises from 55.6% to 81.6% on GSM8K
(88.7% with majority voting over 32 samples), from 12.5% to 20.8% on MATH, and from 77.8% to 86.7% on
ARC-Challenge. The authors report that the NLL term is necessary and that gains shrink across iterations.

## Key Contributions
- A loss combining DPO over correct/incorrect CoT pairs with an NLL term on the winner, length-normalized (Eq. 1).
- An iteration scheme in which the reference model for round t+1 is the round-t model M_t, and each round regenerates its own pairs from a fixed prompt set (§2).
- Measurement that the chosen-sequence log-probability declines under DPO without NLL and rises with it, on GSM8K, ARC and MATH (Figures 3, 4).
- Baseline comparisons on identical data: standard DPO, SFT on gold CoTs, SFT on chosen CoTs, and STaR (Table 1).
- Evidence that iterating beats simply doubling the data in one round (Table 1).

## Key Figures/Tables to Study
- **Figure 3:** chosen and rejected log-probability over training steps on GSM8K, DPO vs DPO+NLL, from two initializations.
- **Figure 2:** SFT-only training raises rejected log-probabilities alongside chosen ones; when SFT is on gold CoTs the chosen log-probability barely increases.
- **Table 1:** GSM8K, all baselines and all four iterations.
- **Table 2:** ARC-Challenge and MATH across iterations.
- **Figure 4:** the same log-probability pattern on ARC-Challenge and MATH.

## Technical Details

### Loop per iteration t
1. Sample N responses per training problem from the current model M_t, each a CoT c followed by an answer y.
2. Score each response with a binary reward: r = 1 if the extracted answer exact-matches the gold label, else 0 (§2).
3. Split responses into a winning set G^w and a losing set G^l, then iterate over both simultaneously to form K pairs per problem, restarting from the first element of the shorter set; problems with an empty set are dropped (§2, footnote 2).
4. Train M_{t+1} from M_t's weights with the loss below, using M_t as the reference model.

### Loss (Eq. 1)
`L = − log σ( β log [Mθ(c_w,y_w|x)/M_t(c_w,y_w|x)] − β log [Mθ(c_l,y_l|x)/M_t(c_l,y_l|x)] ) − α · log Mθ(c_w,y_w|x) / (|c_w| + |y_w|)`
- `Mθ` is the model being trained; `M_t` is the previous iteration's model, used as the reference.
- `(c_w, y_w)` is the winning CoT and answer, `(c_l, y_l)` the losing pair, `x` the problem.
- `β` scales the DPO log-ratio margin; `α` weights the NLL term, which is normalized by the winner's total token length `|c_w| + |y_w|`.
- `σ` is the sigmoid.

### Setting per task (§3)
| | GSM8K | ARC-Challenge | MATH |
|---|---|---|---|
| Seed model | Llama-2-70b-chat | Llama-2-70b-chat | Llama-2-70b-chat |
| Prompting | zero-shot | zero-shot | 4-shot |
| Train prompts | ~7.5k | 7.7k (easy + challenge) | 12,500-problem dataset |
| N samples per problem | 30 | 30 | 20 |
| Sampling temperature | 0.8 (iters 1–2), 1.3 (iters 3–4) | 0.8 (iters 1–2), 1.3 (iter 3) | 0.8 (iters 1–2), 1.0 (iter 3) |
| K pairs per problem | 10 | 20 | 15 |
| Pairs per iteration after filtering | ~55–60k | ~20k / 11k / 5k | ~75k |
| Max training steps per iteration | 5,000 | 4,000 | 5,000 |
| Iterations run | 4 | 3 | 3 |

### Results as reported
- GSM8K, greedy decoding (Table 1): zero-shot CoT 55.6; SFT on gold CoT 63.5; SFT on chosen (STaR round 1) 65.2; STaR on twice the data 66.9; DPO from Llama-2-70b-chat 61.8; DPO from the SFT-on-chosen model 60.3; Iterative RPO 73.1 → 78.0 → 81.1 → 81.6, with 88.2 and 88.7 under 32-sample majority voting at iterations 3 and 4.
- Per-iteration gains decay: 17.5, 4.9, 3.1, 0.5 points (§3.1).
- One iteration on twice as much paired data gives 74.8, below the 78.0 of two iterations on the original amount (§3.1).
- ARC-Challenge (Table 2): CoT 77.8; SFT on chosen 79.8; DPO 82.8 (83.5 from the SFT model); Iterative RPO 84.8 → 86.2 → 86.7, and 87.9 with majority voting.
- MATH (Table 2): 4-shot CoT 12.5; SFT on chosen 16.8; DPO 12.4 (10.5 from the SFT model, below the initialization); Iterative RPO 17.7 → 19.9 → 20.8, and 29.1 with majority voting.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Iterative RPO from Llama-2-70b-chat (GSM8K) | 70B | preference | preference loss | DPO + α·NLL(chosen), length-normalized | arXiv:2404.19733v3 §2 Eq. 1 | verified 2026-09-18 | Table 1: 73.1 with NLL vs 61.8 without, same pairs |
| Iterative RPO from Llama-2-70b-chat (GSM8K) | 70B | preference | β | 0.1 | §3.1 | verified 2026-09-18 | tuned over {0.05, 0.1, 0.5, 1.0} (§3.1) |
| Iterative RPO from Llama-2-70b-chat (GSM8K) | 70B | preference | α | 1 | §3.1 | verified 2026-09-18 | tuned over {0.25, 0.5, 1, 2} when training M1 (§3.1) |
| Iterative RPO from Llama-2-70b-chat (GSM8K) | 70B | preference | learning rate, optimizer | 7e-7, AdamW | §3.1 | verified 2026-09-18 | no ablation reported |
| Iterative RPO from Llama-2-70b-chat (GSM8K) | 70B | preference | batch size (pairs) | 16 | §3.1 | verified 2026-09-18 | no ablation reported |
| Iterative RPO from Llama-2-70b-chat (GSM8K) | 70B | preference | reference model | previous iteration's model M_t | §2 Eq. 1 | verified 2026-09-18 | no ablation reported |
| Iterative RPO from Llama-2-70b-chat (GSM8K) | 70B | preference | checkpoint selection | best of ≤5,000 steps on a held-out 1k split of the training set, then retrain including those 1k for the selected step count | §3.1 | verified 2026-09-18 | no ablation reported |
| Iterative RPO from Llama-2-70b-chat | 70B | preference | compute | training on 8 nodes × 8 A100 80GB; generation on 1 node × 8 V100 32GB | §3.1 | verified 2026-09-18 | not applicable |

## Findings relevant to negative feedback
- The negative signal here is negative-as-gradient: the rejected CoT appears in the DPO term, which lowers its likelihood.
- Figure 3 shows that on GSM8K the chosen-sequence log-probability decreases over training under DPO without NLL, and that the decrease is larger when the model is initialized from SFT on chosen sequences. With the NLL term the chosen log-probability increases in all four plotted settings, and the chosen-rejected margin widens in both cases. Figure 4 reports the same pattern on ARC-Challenge and MATH.
- The paper does not derive the mechanism. It states the observation and cites Pal et al. 2024, Xu et al. 2024 and Hong et al. 2024 as prior reports of the same effect in other settings (§3.1). The logit-gradient account of where the displaced probability mass goes is in [[likelihood-displacement]], not in this paper.
- Figure 2 records the complementary failure: training with SFT on the chosen sequences alone raises the rejected sequences' log-probabilities to nearly the chosen level, so excluding negatives entirely also fails to separate them. This is offered as a hypothesis for why SFT-on-chosen (65.2) trails Iterative RPO iteration 1 (73.1).
- The NLL term acts as the positive anchor: it is a cross-entropy term on the winner, not a change to how the rejected sample is treated.

## Findings relevant to generality
- The three tasks are GSM8K, MATH and ARC-Challenge; no evaluation outside reasoning benchmarks is reported, so effects on instruction following, safety or output diversity are not measured.
- On ARC-Challenge, four-option multiple choice means a CoT can reach the right answer by chance about 25% of the time, so some winners carry incorrect reasoning; gains hold anyway (§3.2).
- Gains saturate: the fourth GSM8K iteration adds 0.5 points, which the authors attribute to iterating over a fixed prompt set (§3.1).

## Connections
- [[dpo]] is the base loss; this paper adds the NLL term and the iteration.
- [[likelihood-displacement]] supplies the gradient-level explanation for the chosen-log-probability decline that this paper only measures.
- [[orpo]] folds an SFT signal into a preference objective by a different route (odds ratio).
- [[self-rewarding-lm]] and [[spin]] are the iterative schemes this paper contrasts with; it differs by using gold-answer matching instead of an LLM judge or human-written winners (§2, §4).
- [[rlvr-tulu3]] and [[deepseek-r1]] use the same correctness signal as an online RL reward rather than as pair labels.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2404.19733 (arXiv v3, 2024-06-26).
- Corrections to the previous card version:
  - Title "Iterative Reasoning Preference Optimization (RPO / IRPO)" → the published title is "Iterative Reasoning Preference Optimization"; the paper's own short name is "Iterative RPO".
  - "Learning rate 1e-6 (AdamW)" → 7e-7 with AdamW (§3.1).
  - "**Figure 2:** log-prob of chosen response across training — DPO declines, DPO+NLL increases" → that is Figure 3 for GSM8K and Figure 4 for ARC and MATH; Figure 2 is the SFT comparison.
  - "α ∈ {0.5, 1.0, 2.0} sweep" → tuned over {0.25, 0.5, 1, 2} (§3.1).
  - "N samples per problem 30 / Iterations 3–4 / Batch size 16 / Epochs per iteration 1" given as one table for all tasks → N, K, temperature and iteration count differ per task; see the per-task table above. Epoch count is not reported; the paper caps training at 5,000 steps (4,000 for ARC).
  - "DPO alone drops log-probability of both chosen and rejected responses" → the measured claim is that the chosen log-probability declines under DPO; the paper's argument for the NLL term rests on the chosen side (§3.1, Figure 3).
  - The card did not name the seed model; it is Llama-2-70b-chat for all three tasks (§3.1–§3.3).
  - "Table 2: Round-by-round GSM8K accuracy" → GSM8K rounds are in Table 1; Table 2 holds ARC-Challenge and MATH.
  - Source type field was missing; added as `paper`.
- Removed as unsupported by the source:
  - "the network finds easy gradient directions that push rejected down while also letting chosen drift down" — the paper gives no mechanism; replaced with the measurement and the citations it makes.
  - "π_ref is refreshed each iteration to the previous iteration's policy" was correct and is retained with its locus (Eq. 1); "Epochs per iteration 1" is removed as not stated.
  - "Guideline: iterate 3–4 rounds … accuracy gains compound across iterations" — gains decay rather than compound (17.5, 4.9, 3.1, 0.5 points on GSM8K, §3.1).
- Not reported by the source: results on model sizes other than 70B; results on non-reasoning evaluations; pass@k or diversity measurements; the number of epochs per iteration.
