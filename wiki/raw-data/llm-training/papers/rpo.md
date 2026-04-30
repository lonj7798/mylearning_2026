<!-- scope: Iterative Reasoning Preference Optimization — DPO with added NLL term, iterated over multiple rounds
     deps: [[dpo]]
     see-also: [[orpo]], [[self-rewarding-lm]], [[grpo]]
-->

# Iterative Reasoning Preference Optimization (RPO / IRPO)
- **Core Insight:** DPO alone drops log-probability of *both* chosen and rejected responses — catastrophic for reasoning, where you want to raise chosen-side log-prob; adding a small SFT/NLL term on the winning chain-of-thought fixes this, and iterating the DPO-then-rollout loop compounds gains.
- **Guideline:** For reasoning tasks, use DPO + α · NLL(chosen) with α ≈ 1.0 and iterate 3–4 rounds (sample → label by correctness → DPO); accuracy gains compound across iterations.
- **Authors:** Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, Jason Weston
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.19733
- **Relevant topics:** iterative RLHF, reasoning alignment, DPO+NLL, offline-online hybrid

## Abstract
We introduce Iterative Reasoning Preference Optimization, an approach for optimizing preferences between competing chain-of-thought (CoT) candidates. We include a modified DPO loss with an additional negative log-likelihood term on the winning CoT, which we show is crucial. Iterating the process improves reasoning accuracy on GSM8K from 55.6% → 81.6% (88.7% with majority voting), MATH from 12.5 → 20.8, and ARC-Challenge from 77.8 → 86.7, using only training set prompts.

## Key Contributions
- Identifies DPO's reasoning-regression failure (both chosen and rejected log-probs drop).
- Adds NLL term on the chosen CoT → keeps chosen-side log-prob ascending.
- Iterative generate-and-label pipeline using ground-truth verifier → no human-labeled pairs needed.
- Demonstrates compounding gains across rounds.

## Key Figures/Tables to Study
- **Figure 2:** log-prob of chosen response across training — DPO declines, DPO+NLL increases.
- **Table 2:** Round-by-round GSM8K accuracy.
- **Section 3.2:** Combined loss.

## Technical Details

### Setup per iteration t
1. Sample N CoTs per problem from current policy π_t.
2. Label each CoT by executing its final answer against the gold; chosen = correct, rejected = incorrect (within the same problem).
3. Train π_{t+1} on the preference pairs with the combined loss below.
4. Repeat.

### Combined loss
`L_RPO = L_DPO(π_θ; π_ref) + α · L_NLL(y_w | x)`
where
`L_NLL(y_w | x) = − (1/|y_w|) · log π_θ(y_w | x)`
- α typically 1.0 in the paper (`α ∈ {0.5, 1.0, 2.0}` sweep; 1.0 default).
- π_ref is refreshed each iteration to the previous iteration's policy.

### DPO piece (same as standard)
`L_DPO = − log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) )`

### Hyperparameters (paper)
| Knob | Value |
|------|-------|
| β | 0.1 |
| α | 1.0 |
| N samples per problem | 30 |
| Iterations | 3–4 |
| Learning rate | 1e-6 (AdamW) |
| Batch size | 16 (pairs) |
| Epochs per iteration | 1 |

### Why the NLL term matters
Pure DPO's gradient balances chosen vs rejected *ratios* but doesn't anchor absolute log-probs. On reasoning data, the network finds easy gradient directions that push rejected down while also letting chosen drift down. NLL on the chosen CoT pins chosen log-prob rising → higher sampled accuracy.

## Connections
- Base method: [[dpo]].
- Odds-ratio alternative that also folds in SFT: [[orpo]].
- Iterative self-generated data: [[self-rewarding-lm]], [[spin]].
- Verifiable-reward RL in the same spirit: [[rlvr-tulu3]], [[deepseek-r1]].
