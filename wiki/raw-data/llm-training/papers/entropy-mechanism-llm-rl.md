<!-- scope: policy-entropy collapse in RLVR for reasoning LLMs; entropy-performance law R = -a exp(H) + b; covariance theorem; Clip-Cov / KL-Cov
     deps: [[grpo]], [[ppo]]
     see-also: [[entropy-regularization-ppo]], [[entropy-collapse-ppo]], [[rlvr-beyond-base-model]], [[kl-control-rlhf]], [[entropy-logging-patterns]]
-->

# The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models
- **Core Insight:** In RL from 11 base models (0.5B-32B) without entropy or KL intervention, validation accuracy R and policy entropy H fit R = -a·exp(H) + b, and 73% of entropy consumption and 76% of the performance gain occur in the first 200 of 2400 gradient steps (§2.3, §2.4).
- **Guideline:** When entropy falls toward zero and validation accuracy plateaus in RLVR, restrict the update of a small fraction of tokens (the paper reports 10^-4 to 10^-3, §4.5) with the largest centered log-probability × advantage product (Clip-Cov or KL-Cov) instead of adding a flat entropy loss, because on Qwen2.5-32B KL-Cov raised the 7-benchmark math average from 45.8 to 52.2 (Table 2), while entropy-loss coefficients either had little effect, caused entropy explosion, or did not outperform the baseline (§4.1, Fig. 9).
- **Authors:** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, et al. (17 authors; Shanghai AI Laboratory, Tsinghua University, UIUC, Peking University, Nanjing University, CUHK)
- **Year:** 2025 (arXiv v1 2025-05; no venue listed on the arXiv page)
- **URL:** https://arxiv.org/abs/2505.22617 (code: https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)
- **Source type:** paper
- **Relevant topics:** policy entropy, entropy collapse, RLVR, GRPO, covariance of log-probability and advantage, Clip-Cov, KL-Cov, clip-higher, predictability of RL performance

## Abstract
Policy entropy, the average token-level uncertainty of the policy on training prompts, drops at the start of RL for reasoning LLMs when no entropy intervention is used and then declines monotonically to near zero, while validation performance rises and then saturates (§2.3). The authors fit R = -a·exp(H) + b between entropy H and downstream performance R, which implies a predictable upper bound R = -a + b at H = 0. They derive that, for a softmax policy, the step-wise entropy change equals the negative covariance between an action's log-probability and its logit change, and that under policy-gradient-like updates the logit change is proportional to the advantage. Measured covariance matches the entropy differences and stays positive during training. They propose Clip-Cov and KL-Cov, which clip or KL-penalize a small set of high-covariance tokens, and report higher entropy and better math-benchmark accuracy.

## Key Contributions
- Entropy-performance law fitted on 11 base models from 4 families, math and code tasks, and 4 RL algorithms, with 2 coefficients per curve over more than 200 data points (§2.2, §2.4).
- Early prediction: fitting on the first 36 training steps of Qwen2.5 models predicts the next 200 steps with RMSE 0.9% (math) and 1.2% (code) (§2.4, Fig. 5).
- Theorems 1-2 linking entropy change to Cov(log π, logit change), with empirical check on Qwen2.5-7B (§3, Fig. 8).
- Evidence that entropy loss and reference-KL penalty do not solve the problem in this setting (§4.1, Figs. 9-10).
- Clip-Cov and KL-Cov, which modify a 10^-4 to 10^-3 fraction of tokens (§4.2, §4.5).

## Key Figures/Tables to Study
- **Fig. 2 / §2.3:** share of entropy consumption and performance gain by training step, 11 runs.
- **Figs. 3-4:** entropy (log scale) vs validation accuracy with fitted curves, math and code. **Fig. 6:** same law under GRPO, RLOO, PRIME, REINFORCE++ on Qwen2.5-7B.
- **Fig. 8:** step-wise entropy difference vs covariance (left); covariance by prompt accuracy 0.125 / 0.5 / 0.875 (right).
- **Table 1:** covariance concentration at training step 1. **Table 2:** GRPO, Clip-higher, Clip-Cov, KL-Cov on 7 math benchmarks.
- **Figs. 9-10:** entropy loss and reference-KL sweeps. **Figs. 11-12:** entropy, response length, accuracy; entropy under varying clip ratio and KL coefficient.

## Technical Details
- **Entropy (Eq. 5):** H(π_θ, D) = -E_{D,π_θ}[log π_θ(y_t | y_<t)], averaged over tokens of responses to training prompts, computed per sampled prompt batch (§2.1).
- **Law (Eq. 6):** R = -a·exp(H) + b. R = validation accuracy, H = entropy, a and b = fitted coefficients. dR/dH = -a·exp(H), so a is the conversion rate of entropy into performance; -a + b is the value at H = 0 (§2.5). The §2 Takeaway box prints "R = -a exp(H + b)", which differs from Eq. 6 and the abstract.
- **Coefficients:** unchanged across GRPO, RLOO, PRIME, REINFORCE++ (§2.5, Fig. 6); log-linear in non-embedding parameter count for Qwen2.5 (Fig. 7); dependent on training data (Fig. 13). Prediction RMSE at the final step: 0.5% (math), 1.9% (code) (§2.4).
- **Lemma 1 (tabular softmax, first order):** H(π^{k+1}|s) - H(π^k|s) ≈ -Cov_{a~π^k(·|s)}(log π^k(a|s), z^{k+1}_{s,a} - z^k_{s,a}). z_{s,a} = logit of action a in state s; k = update step (§3.1).
- **Theorem 1 (vanilla policy gradient):** z^{k+1} - z^k = η·π(a|s)·A(s,a) (Prop. 1), so ΔH ≈ -η·Cov(log π(a|s), π(a|s)·A(s,a)). η = learning rate, A = advantage (§3.2).
- **Theorem 2 (natural policy gradient):** ΔH ≈ -η·Cov(log π(a|s), A(s,a)) (§3.2, proof App. E.4, adapted from Liu 2025).
- **Empirical check:** on-policy GRPO without the PPO surrogate on Qwen2.5-7B, prompt = state, whole response = action, length-normalized log-probability (Eq. 9); -dH and the covariance follow similar curves and the covariance stays positive (§3.3, Fig. 8).
- **Table 1 (Qwen2.5-7B, step 1), mean covariance:** top 0.02% 5.654; top 0.2% 3.112; top 2% 1.385; top 20% 0.351; top 50% 0.152; all 0.003.
- **Token covariance (Eq. 10):** Cov(y_i) = (log π_θ(y_i) - mean_j log π_θ(y_j))·(A(y_i) - mean_j A(y_j)) over the N rollout tokens in a batch.
- **Clip-Cov (Eqs. 11-12):** sample ⌊r·N⌋ indices uniformly from tokens with Cov(y_i) in [ω_low, ω_high] (both bounds >500× the average covariance) and detach their policy gradient. r = clip ratio.
- **KL-Cov (Eqs. 13-14):** tokens with rank(Cov) ≤ k·N (k ≪ 1) get the loss term -β·D_KL(π_θold ‖ π_θ); Listing 1 implements the penalty as |log_prob - old_log_prob|.
- **Results (Table 2, avg of 7 benchmarks):** Qwen2.5-7B GRPO 38.6, Clip-higher 38.8, Clip-Cov 40.4, KL-Cov 40.6. Qwen2.5-32B GRPO 45.8, Clip-higher 47.2, Clip-Cov 50.3, KL-Cov 52.2. On 32B, KL-Cov gains 15.0 (AIME24) and 14.6 (AIME25) points over GRPO (§4.3).
- **Entropy and length:** at the baseline's entropy plateau, KL-Cov keeps entropy more than 10× higher, and response length grows over training (§4.3, Fig. 11). More clipped tokens (Clip-Cov) or a larger β (KL-Cov) gives higher entropy; KL-Cov curves are more stable (§4.4, Fig. 12).
- **Entropy loss and reference KL (§4.1):** entropy-loss coefficients 0.0001 and 0.001 had minor influence, 0.01 caused entropy explosion, 0.005 stabilized entropy but did not outperform other baselines (Fig. 9). Reference-KL coefficients 0.001-0.1 stabilized entropy but lowered accuracy (Fig. 10).

## Recipe ledger
"§2 runs" = Qwen2.5-0.5B/1.5B/3B/7B/32B, Mistral-7B-v0.3, Mistral-Nemo-Base-2407, Mistral-Small-3.1-24B-Base-2501, LLaMA3.2-3B, LLaMA3.1-8B, DeepSeek-Math-7B-Base (§2.2). All loci are arXiv:2505.22617v1.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| §2 runs | 0.5B-32B | RL | Algorithm; framework; start point | GRPO, REINFORCE++, PRIME (Fig. 6 adds RLOO); veRL; base model ("Zero") | §2.2 | verified 2026-09-14 | Fig. 6: fitted curve unchanged across algorithms |
| §2 runs | 0.5B-32B | RL | Policy LR; implicit-PRM LR (PRIME only) | 5×10^-7; 10^-6 | §2.2 | verified 2026-09-14 | no ablation reported |
| §2 runs | 0.5B-32B | RL | Batch size; micro-batch (policy and PRM) | 256; 128 | §2.2 | verified 2026-09-14 | no ablation reported |
| §2 runs | 0.5B-32B | RL | Rollout per step | 512 prompts × 8 responses | §2.2 | verified 2026-09-14 | no ablation reported |
| §2 runs | 0.5B-32B | RL | Reference-KL coefficient; clip ε | 0; 0.2 | §2.2 | verified 2026-09-14 | §4.1 Fig. 10: KL coefficients 0.001-0.1 lowered accuracy |
| §2 runs | 0.5B-32B | RL | Prompt filter | drop prompts whose responses are all correct or all incorrect | §2.2 | verified 2026-09-14 | no ablation reported |
| §2 runs | 0.5B-32B | RL | Training data | math: Eurus-2-RL-Math (Qwen family, Mistral-24B), GSM8K (other models); code: AceCode, Eurus-2-RL-Code, KodCode (Qwen family, Mistral-24B) | App. A | verified 2026-09-14 | chosen by model difficulty "to stabilize the RL process"; no ablation reported |
| §2 runs | 0.5B-32B | RL | Run length; validation interval | 2400 gradient steps (Fig. 2); every 4 rollout steps | §2.3; Fig. 3 caption | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B, Qwen2.5-32B | 7B, 32B | RL | Data; rollout; temperature | DAPO-MATH; 256 prompts × 8 responses; 1 | §4.3 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B, Qwen2.5-32B | 7B, 32B | RL | Policy updates per rollout step; max generation length | 8; 8192 | §4.3 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B, Qwen2.5-32B | 7B, 32B | RL | Clip-higher baseline upper ε | 0.28 (from Yu et al. 2025) | §4.3 | verified 2026-09-14 | no ablation in this paper |
| Qwen2.5-7B, Qwen2.5-32B | 7B, 32B | RL | Clip-Cov r; ω_low; ω_high | 2×10^-4; 1; 5 | §4.3 | verified 2026-09-14 | Fig. 12 (7B): entropy rises with more clipped tokens; accuracy vs r not reported |
| Qwen2.5-7B | 7B | RL | KL-Cov k; β | 2×10^-3; 1 | §4.3 | verified 2026-09-14 | Fig. 12 (7B): entropy rises with β; accuracy vs β not reported |
| Qwen2.5-32B | 32B | RL | KL-Cov k; β | 2×10^-4; 1 | §4.3 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B, Qwen2.5-32B | 7B, 32B | RL | Learning rate; reference-KL coefficient | not reported (the §2.2 values are scoped to §2 runs) | checked §4.3, Table 2, App. A | not reported | — |
| Qwen2.5-7B, Qwen2.5-32B | 7B, 32B | eval-gate | Evaluation decoding | AIME, AMC: temperature 0.6, avg@32; other sets: greedy | §4.3; Table 2 caption | verified 2026-09-14 | — |

## Findings relevant to generality, negative feedback
- **Sign of the update (§3.2):** after Theorem 1 the paper states that "an action a receives both high/low probability and high/low advantage would lower the entropy, and vice versa". Read with Eq. 10, a sampled token with below-mean log-probability and below-mean (for example negative) advantage has a positive centered product, so a push-down on an unlikely token lowers entropy; a negative advantage on an above-mean-probability token gives a negative product and raises entropy. The paper does not report statistics or results split by advantage sign (checked §3-4).
- **Selection is sign-agnostic:** Clip-Cov and KL-Cov select by the centered product, not by p·A or by the sign of A (Eqs. 10-13, Listing 1), so negative-advantage tokens are eligible for selection.
- **Clip-higher (§4.5):** raising the upper ratio bound affects only positive-advantage tokens; it admits low-probability, positive-advantage tokens with average covariance about -0.03.
- **Ceiling (§2.6, Interpretation):** on the question raised by Yue et al. 2025 ([[rlvr-beyond-base-model]]) whether RL only elicits behavior already learned in pre-training and "cannot break the ceiling of the base model", the authors say their results "conditionally support" it: if policy entropy diminishes, the upper bound exists and can be predicted. The authors attribute the bound to the entropy mechanism, not to RL itself.
- **Limits:** the authors state the predictability "is not arguably universal", since other policy models or off-policy data showed different entropy patterns (§2.6). They observe no relationship between the controlled entropy level and performance; the optimal entropy is an open question (§4.5). Evaluation is math (and code in §2); pass@k at large k is not reported (checked §2.2, §4.3, Table 2).

## Connections
- [[grpo]] — group-normalized advantage (Eq. 3) used in most runs.
- [[ppo]] — clipped surrogate (Eq. 4, ε = 0.2) that Clip-Cov and KL-Cov modify.
- [[reinforce-plus-plus]] — one of the four algorithms whose fitted coefficients match in Fig. 6.
- [[rlvr-beyond-base-model]] — pass@k study whose ceiling claim §2.6 discusses.
- [[entropy-regularization-ppo]] — entropy term of the kind tested in §4.1.
- [[entropy-collapse-ppo]] — MuJoCo PPO study that also found entropy regularizers did not help significantly.
- [[kl-control-rlhf]] — reference-KL penalty; §4.1 reports it lowered accuracy here.
- [[entropy-logging-patterns]] — verl `clip_cov` / `kl_cov` loss modes adapted from this paper's repository.
- [[deepseek-r1]] — source of the "Zero" setting (RL from base models) used in §2.2.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2505.22617 (arXiv v1, the only version).
- Corrections to the previous card version:
  - "Across >20 models" → 11 base models from 4 families, plus Qwen2.5 instruct models in App. D (§2.2).
  - "PPO, GRPO, RLOO, Reinforce++" → GRPO, RLOO, PRIME, REINFORCE++ (§2.2, Fig. 6).
  - "entropy change ∝ -Cov(log π, A)" for policy gradient → that is Theorem 2 (natural PG); vanilla PG gives -η·Cov(log π, π·A) (Theorem 1).
  - "Clip-Cov ranks tokens by p_t·A_t and zeroes the top ~2%" → centered log-prob × advantage product (Eq. 10); random sample of r·N tokens with Cov in [1, 5], r = 2×10^-4 (§4.3).
  - "KL-Cov: forward token-level KL, k3 approximation" → D_KL(π_θold ‖ π_θ) (Eq. 14), implemented as |log π - log π_old| (Listing 1).
  - "fit a, b on the first ~10% of training" → first 36 training steps, predicting the next 200 (§2.4); Fig. 5 caption says 15%.
  - "Fig. 1 = entropy across algorithms; Fig. 3 = covariance histogram" → Fig. 1 = collapse and R-H relation; Fig. 6 = algorithms; Fig. 3 = math fitting curves; covariance distribution = Table 1.
  - "Table of interventions includes the entropy bonus" → Table 2 compares GRPO, Clip-higher, Clip-Cov, KL-Cov; entropy loss appears only in Fig. 9.
  - "Qwen2.5-7B / Qwen2.5-Math-7B; ceiling rises several points" → Qwen2.5-7B and Qwen2.5-32B; averages in Table 2.
  - "entropy bonus β in {1e-4, 1e-3, 1e-2}" → {0.0001, 0.001, 0.005, 0.01} with the outcomes in §4.1.
  - "GRPO group size 8-16, rollout up to 8k, LR ~1e-6" → 8 responses per prompt; max generation length 8192 (§4.3); policy LR 5×10^-7 in §2 runs, and 10^-6 is the PRIME implicit-PRM LR (§2.2).
- Removed as unsupported by the source: the "H < 0.1 nats" collapse threshold; the claim that the mechanism explains DeepSeek-R1's cold-start SFT and long rollouts; the SAC temperature-tuning comparison; "treating all tokens symmetrically hurts high-quality trajectories".
- Not reported by the source: pass@k at large k; results split by advantage sign; learning rate for the §4.3 runs.
