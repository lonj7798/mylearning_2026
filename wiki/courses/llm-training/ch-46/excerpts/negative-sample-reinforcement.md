<!-- scope: Zhu et al. (2025): decomposing RLVR into Positive Sample Reinforcement (PSR) and Negative Sample Reinforcement (NSR), the Pass@k consequences of each, the token-logit gradient of both, and the Weighted-REINFORCE (W-REINFORCE) combination
     deps: [[grpo]], [[ppo]]
     see-also: [[likelihood-displacement]], [[rlvr-beyond-base-model]], [[dr-grpo]]
-->

# The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning
- **Core Insight:** Training Qwen2.5-Math-7B on MATH with only incorrect samples (NSR) reaches Pass@1 75.7 on MATH against 63.2 for the base model and 76.3 for GRPO, and it is the only ablation that keeps Pass@256 at the base-model level (96.9 vs 96.9 base, 95.5 GRPO); training with only correct samples (PSR) raises Pass@1 to 74.1 but lowers Pass@256 to 91.2 (Table 1).
- **Guideline:** When an RLVR run must keep coverage at large k as well as raise Pass@1, down-weight the positive term rather than removing it — W-REINFORCE with λ = 0.1 matched the best Pass@1 (76.6) and the base-model Pass@256 on MATH, and λ ≤ 0.2 gave stable results while λ = 1 (plain REINFORCE) dropped MATH Pass@256 to 92.0 (Table 1; App. E Table 4).
- **Authors:** Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, Yu Meng (University of Virginia; Princeton Language and Intelligence)
- **Year:** 2025 (arXiv v1 2025-06; v2 2025-10-25)
- **URL:** https://arxiv.org/abs/2506.01347
- **Source type:** paper
- **Relevant topics:** RLVR, negative gradient, Pass@k, output diversity, entropy, policy-gradient decomposition, REINFORCE variants

## Abstract
RLVR updates a language model with both correct and incorrect samples. The paper decomposes the learning signal into Positive Sample Reinforcement (PSR) and Negative Sample Reinforcement (NSR) and trains Qwen2.5-Math-7B, Qwen3-4B and Llama-3.1-8B-Instruct on a mathematical reasoning dataset. Training only on negative samples improves performance over the base model across the whole Pass@k spectrum up to k = 256 and often matches or surpasses PPO and GRPO. Reinforcing only correct responses improves Pass@1 but degrades performance at higher k because diversity falls. The authors propose a weighted combination that scales the PSR term by λ.

## Key Contributions
- Decomposition of the RLVR gradient into PSR and NSR, implemented by updating only on correct or only on incorrect rollouts (§3.1, §D.1).
- Pass@k curves for k ∈ {1, …, 256} on MATH, AIME 2025 and AMC23 for base, PPO, GRPO, REINFORCE, PSR, NSR and W-REINFORCE (Table 1).
- Token-logit gradient analysis of PSR and NSR (Eq. 7, Eq. 8) and its extension to PPO and GRPO (§4.3).
- W-REINFORCE (Eq. 9) with a λ ablation over {0, 0.05, 0.1, 0.2, 1} (App. E, Table 4).
- Entropy and diversity tracking on a held-out set across training (§3.3, Figure 5b).

## Key Figures/Tables to Study
- **Table 1:** Pass@k at k = 1…256 for seven training signals on three benchmarks (Qwen2.5-Math-7B).
- **Eq. 7 / Eq. 8 and Figure 6:** the logit-gradient directions of PSR and NSR.
- **Figure 5b:** entropy on a held-out test set over training for PSR, NSR, PPO and GRPO.
- **Table 4 (App. E):** λ sweep for W-REINFORCE.

## Technical Details
- **Setup (§3.1, §D.1).** Training set MATH (7,500 problems); framework verl; prompt batch 1,024; 8 rollouts per prompt; training sampling temperature 1.0; mini-batch 256; learning rate 1e-6; maximum context 4,096 tokens for Qwen2.5-Math-7B and Llama-3.1-8B-Instruct, 32,768 for Qwen3-4B; clip ratio 0.2; entropy bonus coefficient 1e-4 for all objectives; KL penalty 1e-3 for PPO and GRPO and none for PSR and NSR; 8× H200 for one node.
- **Evaluation (§3.1).** 256 samples per prompt at temperature 0.6, top-p 0.95 for Qwen2.5-Math-7B; Pass@k uses the unbiased estimator Pass@k = E[1 − C(n−c, k)/C(n, k)] with n samples and c correct (Eq. 5).
- **Table 1, MATH (Pass@1 / Pass@256):** base 63.2 / 96.9; PPO 76.6 / 96.3; GRPO 76.3 / 95.5; REINFORCE 74.8 / 92.0; PSR 74.1 / 91.2; NSR 75.7 / 96.9; W-REINFORCE 76.6 / 96.7.
- **Table 1, AIME 2025 (Pass@1 / Pass@256):** base 6.1 / 46.7; PPO 8.5 / 43.3; GRPO 10.3 / 50.0; PSR 11.6 / 43.3; NSR 10.0 / 53.3; W-REINFORCE 10.6 / 56.7.
- **Table 1, AMC23 (Pass@1 / Pass@256):** base 41.0 / 100.0; PPO 62.0 / 97.5; GRPO 61.7 / 97.5; PSR 62.6 / 92.5; NSR 60.9 / 100.0; W-REINFORCE 62.0 / 97.5.
- **Gradient of PSR on token logits (Eq. 7):** −∂L_PSR/∂z_v ∝ π_v(1 − π_v) for the sampled token v = y_t, and −π_{y_t}·π_v for every other token v. The paper states that this repeatedly amplifies observed correct sequences and suppresses alternatives, which lowers entropy (§4.2).
- **Gradient of NSR on token logits (Eq. 8):** −∂L_NSR/∂z_v ∝ −π_v(1 − π_v) for the sampled token and +π_{y_t}·π_v for every other token. Three stated properties: the penalty is scaled by (1 − π_{y_t}), so high-probability tokens inside an incorrect response are barely changed; unsampled tokens gain logit in proportion to their current probability, which the authors call prior-guided redistribution; and updates stop once the model stops producing the incorrect response (§4.2).
- **W-REINFORCE (Eq. 9):** L = −E[Σ_{y: r=1} λ·π_θ(y|x)] − E[Σ_{y: r=−1} −π_θ(y|x)]; λ = 1 recovers REINFORCE, λ = 0 recovers NSR. The experiments use λ = 0.1 (§5).
- **λ ablation on MATH (Table 4, Pass@1 / Pass@256):** λ = 0: 75.7 / 96.9; 0.05: 75.6 / 97.1; 0.1: 76.6 / 96.7; 0.2: 75.8 / 95.9; 1: 74.8 / 92.0.
- **Entropy (§3.3, Figure 5b).** NSR keeps held-out entropy high through training; PSR shows the steepest drop; PPO and GRPO fall in between and still decline considerably. The figure is a plot; no numbers are printed.
- **Extension to PPO and GRPO (§4.3).** Clipping preserves the gradient direction; the GRPO advantage (r − mean(r))/std(r) rescales the gradient and keeps the sign of the raw reward; so the sign-specific behaviour of PSR and NSR carries over.

## Findings relevant to generality and negative feedback
- **Negative as gradient (course standard §6.1, type 4).** NSR is a pure negative gradient on incorrect rollouts. Its measured effect is the opposite of what a likelihood-displacement argument alone predicts: coverage at k = 256 was preserved, and the authors attribute this to the (1 − π_{y_t}) scaling and to redistribution proportional to the current prior (§4.2, Interpretation).
- **Size of effect.** The positive term is not useless: on MATH, PSR alone reaches Pass@1 74.1 and NSR alone 75.7, while the λ = 0.1 combination reaches 76.6 (Table 1, Table 4). The paper does not report a percentage split of the gain between positive and negative signals.
- **Limits (App. F).** Extended NSR training over hundreds of gradient steps lowered performance, while W-REINFORCE did not show this; the setting is sparse binary outcome rewards only; results are reported for models with a strong math prior (Qwen family) plus Llama-3.1-8B-Instruct.

## Connections
- [[grpo]] — the group-baseline objective whose advantage sign defines positive and negative samples.
- [[rlvr-beyond-base-model]] — the pass@k coverage critique that motivates measuring k up to 256.
- [[likelihood-displacement]] — the opposite-direction result for negative gradients in preference losses.
- [[dr-grpo]] — normalization terms that change how negative-advantage tokens are weighted.

## Verification
- Checked on 2026-09-15 against the cached primary text of https://arxiv.org/abs/2506.01347 (arXiv v2, 2025-10-25): Abstract, §3.1, §3.3, §4.2, §4.3, §5, Table 1, App. D.1, App. E Table 4, App. F.
- Corrections to the previous card version: no previous card existed in the library for this slug.
- Not reported by the source: number of RL steps per run, non-math held-out evaluations, a measured split of gain between positive and negative signals.
