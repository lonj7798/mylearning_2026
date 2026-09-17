<!-- chapter: ch-38
     track: preference
     kind: content
     title: KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax
     deps: [ch-37]
     sources: [[ppo]], [[trpo]], [[rlhf-instructgpt]], [[hh-rlhf]], [[mitigating-alignment-tax-rlhf]], [[llama-2]], [[llama-2-recipe]], [[llama-2-ppo]], [[costa-huang-ppo-details]], [[n-implementation-details-rlhf-ppo]], [[kl-control-rlhf]], [[john-schulman-kl-tricks]], [[trl-ppo]], [[trl-ppo-recipe]], [[openrlhf-ppo]], [[reward-model-overoptimization]], [[gpt-4-technical-report]], [[lilianweng-rlhf]]
     figures: figures/ppo-clip.html
     revised: 2026-09 (generality revision)
-->

# Chapter 38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax

> **Core insight.** KL-controlled RLHF trains a policy with PPO on a reward made of a reward-model score and a per-token penalty on the log-ratio to the SFT model. Its optimum is the SFT distribution reweighted by exp(r/β) ([[kl-control-rlhf]], Korbak et al. Eq. 5-6). In InstructGPT, PPO without a pretraining term lowered 175B few-shot SQuADv2 F1 from 69.75 (GPT-3) to 51.95 and FR→EN BLEU from 39.93 to 26.58. Adding pretraining gradients with γ = 27.8 (PPO-ptx) gave 69.93 and 36.76, and the authors report that it "does not lead to large changes in labeler preference" ([[rlhf-instructgpt]] Table 14, §4.1). Raising the KL coefficient 100-fold did not remove these regressions (App. E.6). In Anthropic's 13M-52B study the tax appeared in small models, while the 13B and 52B RLHF models scored higher on zero-shot NLP evaluations ([[hh-rlhf]] §1.1, Figure 3). RLHF also reduced GPT-4's MMLU calibration: expected calibration error went from 0.007 to 0.074 ([[gpt-4-technical-report]] Figure 8).
>
> **Guideline.** When PPO-based RLHF starts from a pretrained base model and the goal is broad capability, add a pretraining (or replay) log-likelihood term to the PPO update. InstructGPT's γ ≥ 20 recovered SQuADv2 and DROP at 1.3B, while a larger β did not ([[rlhf-instructgpt]] App. E.6). If pretraining data is not available, evaluate weight interpolation between the pre-RLHF and post-RLHF checkpoints. For OpenLLaMA-3B, its alignment-forgetting Pareto front was above early stopping, L1/L2 regularization, LoRA, distillation, and KL penalties of 0.05-0.2 ([[mitigating-alignment-tax-rlhf]] §4.1, App. C.2). Tune β against a human or held-out judge rather than against the proxy reward. InstructGPT's Likert score peaked around β = 0.01-0.02 and was poor at 0 and 2 (App. E.7). Llama 2 used 0.01 for 7B/13B and 0.005 for 34B/70B ([[llama-2]] §3.2.3). For every RLHF run, report the held-out capability suite, a borderline-prompt false-refusal rate, and a calibration metric next to the reward, because each of these moved in a published RLHF run while the reward rose.

## Why this chapter matters for a general-purpose model

Pipeline position: pre-training → mid-training → SFT → **preference optimization (this chapter)** → RL → evaluation. [[ch-41]] trained a reward model (RM) from pairwise preferences. [[ch-37]] derived the policy-gradient estimator and baselines. This chapter joins them: PPO optimizes the policy against the RM, and a KL penalty keeps the policy near the SFT model.

The stage matters for generality because its objective contains no term for capabilities the RM does not score. An RM trained on API-style instructions gives no reward for reading comprehension or translation, so no term in the objective penalizes the policy for losing them. InstructGPT named this loss the **alignment tax**: the performance cost on tasks that the alignment procedure did not target ([[rlhf-instructgpt]] §1). The chapter answers the three course questions for this stage:

1. Breadth: RLHF improved instruction following on held-out labelers, and the authors report qualitative transfer to code and non-English instructions (§7). Large models in [[hh-rlhf]] gained on zero-shot NLP evaluations (§6).
2. Narrowing: regressions on public NLP tasks (§6), reduced calibration (§6.5), false refusals on borderline prompts (§8), and over-optimization of the proxy reward (§4, §9).
3. Measurement: a fixed public NLP suite scored against both the base model and the SFT model, held-out labelers, cross-validated RM accuracy, false-refusal rates, and calibration error. The known measurement errors are listed in the Generalization lens.

## §1 TRPO: the trust-region problem PPO approximates

**Definition.** Trust Region Policy Optimization (TRPO) maximizes a local surrogate of expected return under a constraint on the KL divergence between the old and new policy ([[trpo]] §4).

**Problem.** A policy-gradient step computed from samples of π_old is accurate only near π_old. A step that is too large can lower the true return even when the sampled surrogate increases.

**Mechanism and formulas.** Schulman et al. prove a lower bound on the new policy's return (arXiv:1502.05477v5, Theorem 1, Eq. 9):

```
η(π̃) ≥ L_π(π̃) − C · D_KL^max(π, π̃),     C = 4εγ / (1 − γ)²,     ε = max_{s,a} |A_π(s,a)|
```

- η(π̃): expected discounted return of the new policy π̃.
- L_π(π̃): the surrogate η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a), with states weighted by the old policy's visitation ρ_π.
- D_KL^max: the maximum over states of KL(π(·|s) ‖ π̃(·|s)).
- γ: discount factor. A_π: advantage under the old policy.

Maximizing the right-hand side guarantees non-decreasing η (Eq. 10). The authors state that with the theoretical C "the step sizes would be very small", so TRPO uses a constraint instead (Eq. 11-12): maximize L_θold(θ) subject to the average KL over sampled states ≤ δ. The step is computed with conjugate gradient on Fisher-vector products, followed by a line search (§6, App. C).

**Worked example.** With γ = 0.99 and ε = 1, C = 4 × 1 × 0.99 / 0.01² = 39,600. A KL of 0.001 then costs 39.6 units of return in the bound, so such a step is certified to improve η only if its surrogate gain exceeds 39.6. With γ = 1, the undiscounted setting used for LLM responses (§3), C is undefined. This is one reason the bound is used as motivation rather than as a step-size rule for language models (**Interpretation**).

**Evidence and cost.** TRPO used δ = 0.01 in all its experiments (§8, Table 2). It used k = 10 conjugate-gradient iterations. A naive implementation spends more than 90% of computation on Fisher-vector products, which the authors reduce by computing them on 10% of the data (App. C).

**Implication.** PPO keeps the surrogate L and replaces the second-order constrained step with a first-order clipped objective. Engstrom et al., as summarized in [[costa-huang-ppo-details]] (core detail 8), found PPO's clipped objective performed similarly to TRPO's objective when other implementation details were held fixed (control tasks, not language models).

## §2 The PPO clipped objective, term by term

**Definition.** Proximal Policy Optimization (PPO) maximizes a clipped surrogate objective over several minibatch epochs on each batch of samples ([[ppo]] §3).

**Problem.** Maximizing L^CPI(θ) = Ê_t[r_t(θ) Â_t] directly leads to excessively large updates (§3). PPO needs a first-order objective that removes the incentive to move the probability ratio far from 1.

**Formula** (arXiv:1707.06347v2 Eq. 6-7):

```
r_t(θ) = π_θ(a_t | s_t) / π_θold(a_t | s_t)
L^CLIP(θ) = Ê_t[ min( r_t(θ) Â_t ,  clip(r_t(θ), 1 − ε, 1 + ε) Â_t ) ]
```

- r_t(θ): probability ratio of action a_t (for an LLM, token t) under the current and the rollout policy; r_t(θ_old) = 1.
- Â_t: advantage estimate for timestep t. ε: clip range; the paper suggests 0.2.
- Ê_t: empirical mean over timesteps in the batch.

**Mechanism, case by case.**
1. Â_t > 0 and r_t < 1 + ε: the unclipped term is smaller; the gradient raises π_θ(a_t).
2. Â_t > 0 and r_t ≥ 1 + ε: the clipped term (1 + ε)Â_t is smaller and constant in θ; the gradient is zero.
3. Â_t < 0 and r_t > 1 − ε: the unclipped term is smaller (more negative); the gradient lowers π_θ(a_t).
4. Â_t < 0 and r_t ≤ 1 − ε: the clipped term (1 − ε)Â_t is smaller and constant; the gradient is zero.

The min ignores a ratio change only when it would improve the objective. It keeps the change when it makes the objective worse (§3). L^CLIP is therefore a pessimistic (lower) bound on the unclipped surrogate L^CPI, and it equals L^CPI to first order around θ_old (§3). It is not a bound on the true change in return.

**Worked example** (ε = 0.2). Â = +1, r = 1.3: min(1.3, 1.2) = 1.2, gradient zero. Â = −1, r = 0.7: min(−0.7, −0.8) = −0.8, gradient zero. Â = −1, r = 1.5: min(−1.5, −1.2) = −1.5, gradient nonzero. In the third case the objective keeps decreasing linearly as r grows, with no bound. OpenRLHF's optional `dual_clip` caps this loss at c·|Â| for Â < 0 ([[openrlhf-ppo]], loss.py L143-148).

**Evidence.** On 7 MuJoCo tasks × 3 seeds, the average normalized score was 0.82 with ε = 0.2, 0.76 with ε = 0.1, and 0.70 with ε = 0.3. It was −0.39 with no clipping or penalty, 0.74 for the best adaptive-KL variant (d_targ = 0.01), and 0.72 for the best fixed KL penalty (β = 3) ([[ppo]] Table 1). **Result (single study)**, control tasks only.

**The adaptive KL-penalty variant** (Eq. 8): maximize Ê_t[r_t Â_t − β KL[π_θold, π_θ]]. After each update, if d = Ê_t[KL] < d_targ/1.5 then β ← β/2, and if d > 1.5 d_targ then β ← 2β ([[ppo]] §4). This controller acts on the KL to the previous policy. The RLHF penalty in §4 acts on the KL to the SFT model.

**Combined objective** (Eq. 9): L^{CLIP+VF+S} = Ê_t[L^CLIP − c_1 (V_θ(s_t) − V_t^targ)² + c_2 S[π_θ](s_t)], where S is entropy and c_1, c_2 are coefficients. Atari used c_1 = 1 and c_2 = 0.01. The MuJoCo runs shared no parameters between policy and value function and used no entropy bonus ([[ppo]] Table 5, §6.1).

**Conditions.** The paper contains no language-model experiments. Its epochs per batch (10 MuJoCo, 3 Atari) were not tuned for LLMs, and InstructGPT used a single inner epoch (§5).

**Interactive figure.** [figures/ppo-clip.html](figures/ppo-clip.html), panel 1, plots one term of L^CLIP and its gradient against r for either sign of Â, so the four cases above can be checked for any ε.

## §3 Advantages for language models: per-token reward, GAE, and the value model

**Definition.** In RLHF the state s_t is the prompt plus the tokens generated so far, and the action a_t is the next token. The episode ends when the response ends. InstructGPT calls this a bandit environment: the RM produces one reward per prompt-response pair and the episode ends ([[rlhf-instructgpt]] §3.5).

**Problem.** The RM gives one scalar for the whole response, but PPO needs an advantage for every token.

**Mechanism.**
1. Build a per-token reward vector. Every response token receives −β times a per-token KL estimate. The RM score is added at the final position.
2. Run a value model V to predict the return from each position.
3. Compute generalized advantage estimation (GAE) backwards from the end.
4. Set the value target to R_t = Â_t + V(s_t).

**Formulas** ([[ppo]] Eq. 11-12):

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
Â_t = δ_t + (γλ) δ_{t+1} + … + (γλ)^{T−t−1} δ_{T−1}
```

- r_t: per-token reward. V: value model; V after the last token is 0.
- γ: discount; InstructGPT applies no discount (γ = 1) (App. C.4). λ: GAE parameter.
- T: response length. With λ = 1 and γ = 1, Â_t is the sum of remaining rewards minus V(s_t).

TRL's implementation at commit a08e713 ([[trl-ppo]], ppo_trainer.py L775-799):

```python
logr = ref_logprobs - logprobs
kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
non_score_reward = -args.kl_coef * kl
rewards = non_score_reward.clone()
...
rewards[actual_start, actual_end] += scores
...
delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
lastgaelam = delta + args.gamma * args.lam * lastgaelam
...
returns = advantages + values
advantages = masked_whiten(advantages, ~padding_mask)
```

**Worked example.** A 3-token response; β = 0.05; RM score 1.0; γ = 1; λ = 0.95.

| t | log π_θ − log π_ref (k1) | KL reward −β·k1 | RM score | r_t | V(s_t) | δ_t | Â_t |
|---|---|---|---|---|---|---|---|
| 1 | +0.4 | −0.02 | 0 | −0.02 | 0.5 | −0.02 + 0.6 − 0.5 = 0.08 | 0.08 + 0.95 × 0.3905 = 0.4510 |
| 2 | −0.2 | +0.01 | 0 | +0.01 | 0.6 | 0.01 + 0.8 − 0.6 = 0.21 | 0.21 + 0.95 × 0.19 = 0.3905 |
| 3 | +0.2 | −0.01 | 1.0 | 0.99 | 0.8 | 0.99 + 0 − 0.8 = 0.19 | 0.19 |

With λ = 1, Â_1 = 0.08 + 0.21 + 0.19 = 0.48, which equals the reward sum (0.98) minus V(s_1) (0.5). Token 2 has a negative k1 value and therefore a positive KL reward. The per-token k1 term is not a KL value; only its expectation is (§4). Panel 2 of [figures/ppo-clip.html](figures/ppo-clip.html) recomputes this table for any inputs.

**The value model in practice.**
- InstructGPT used a separate 6B value function, initialized from the 6B RM, for policies of all sizes. The value learning rate was 9e-6 for the 1.3B and 6B policies and 5e-6 for the 175B policy ([[rlhf-instructgpt]] App. C.4). A 175B RM was not used partly because its training was less stable, "which made them less suitable for use as initializations for the PPO value functions" (App. C.2).
- TRL's PPO trainer uses a separate value model with a scalar head, and its example script initializes it from the RM ([[trl-ppo]]; [[trl-ppo-recipe]]).
- Huang et al. initialize the value network from the RM. They write that this warm start "can greatly improve initial gradients to the policy and reduce drift / alignment tax over training", citing Noukhovitch et al. 2023, and they report no ablation of it ([[n-implementation-details-rlhf-ppo]] Detail 22).

**Value-loss clipping.** OpenAI's ppo2 clips the value prediction around the previous value: L^V = max[(V_θ − V_targ)², (clip(V_θ, V_old − ε, V_old + ε) − V_targ)²]. This is core detail 9 in [[costa-huang-ppo-details]]. The same post reports that Engstrom et al. found no evidence that it helps and that Andrychowicz et al. suggest it can hurt (control tasks). TRL implements it with a 0.5 factor and `cliprange_value` = 0.2 ([[trl-ppo]] L831-839).

**Implication.** GAE with λ < 1 lowers variance by trusting V. When V is wrong, for example at initialization, the bias enters every token's advantage. This is why value initialization and a warm-up are recurring details (§9).

## §4 The KL penalty: estimator, placement, and the distribution it targets

**Definition.** The KL-regularized RL objective is J(θ) = E_{x∼π_θ}[r(x)] − β D_KL(π_θ, π_0), where π_0 is the initial (SFT) model and β ≥ 0 sets the trade-off ([[kl-control-rlhf]], Korbak et al. arXiv:2205.11275v2 Eq. 3).

**Problem.** An RM is a learned proxy. Without a constraint, PPO can find outputs that score high on the RM and low with humans. Llama 2 reports the penalty "is useful for training stability, and to reduce reward hacking whereby we would achieve high scores from the reward model but low scores from human evaluation" ([[llama-2]] §3.2.3).

**The target distribution.** Korbak et al. show the objective has a closed-form optimum and can be written as a divergence to it (Eq. 5-7):

```
π*(x) = (1/Z) π_0(x) exp(r(x)/β),      J(θ) ∝ −D_KL(π_θ, π*)
```

- x: a full response; Z: normalizing constant; β acts as a temperature on the reward.

**Worked example.** Two responses with π_0 = (0.5, 0.5) and r = (1, 0). With β = 0.5, π*(response 1) = e² / (e² + 1) = 0.881. With β = 2, π* = e^0.5 / (e^0.5 + 1) = 0.622. Smaller β moves more mass toward the higher-reward response. With β → 0 all mass goes to the reward maximum, and with β → ∞ the policy stays at π_0.

**Placement and estimator.** InstructGPT's Eq. 2 subtracts β log(π^RL(y|x)/π^SFT(y|x)) from the reward for each sampled response ([[rlhf-instructgpt]] §3.5). Summed over tokens, this is the sampled log-ratio, called k1 in Schulman's notation. For samples x ∼ q and ratio r = p(x)/q(x), Schulman defines ([[john-schulman-kl-tricks]]):

| Estimator of KL[q, p] | Formula | Bias | Sign | Std/true KL, q = N(0,1), p = N(0.1,1), true KL 0.005 | Std/true KL, p = N(1,1), true KL 0.5 |
|---|---|---|---|---|---|
| k1 | −log r | unbiased | can be negative | 20 | 2 |
| k2 | ½ (log r)² | biased (0.2% and 25% of true KL) | ≥ 0 | 1.42 | 1.73 |
| k3 | (r − 1) − log r | unbiased | ≥ 0 | 1.42 | 1.7 |

In RLHF, q is the policy π_θ and p is π_ref, so k1 = log π_θ − log π_ref per token. TRL's default `kl_estimator` is "k1", with "k3" as an option ([[trl-ppo-recipe]]). OpenRLHF puts KL in the reward by default and prints a recommendation of k1 for reward-side KL and k2 or k3 for loss-side KL ([[openrlhf-ppo]], train_ppo_ray.py L675-680). The k3 estimator inside the loss is the GRPO convention covered in [[ch-40]] and [[ch-43]].

**Fixed or adaptive β.** InstructGPT used a fixed β = 0.02 (App. C.4). OpenRLHF's `AdaptiveKLController`, which cites Ziegler et al. 2019, updates β ← β(1 + clip(KL/target − 1, −0.2, 0.2) · n_steps/horizon). It is used only when a KL target is set, and the default is a fixed coefficient of 0.01 ([[openrlhf-ppo]], kl_controller.py L4-19). The controller acts on measured KL, not on entropy.

**Evidence on choosing β.**
- InstructGPT, human Likert score against β (with pretraining mix): "Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02" ([[rlhf-instructgpt]] App. E.7, Figure 36; model size not stated in E.7).
- Anthropic used λ_KL = 0.001 and wrote that it "likely has a very minor impact during most of RL training (as D_KL < 100 typically), and might actually be wholly unnecessary" ([[hh-rlhf]] §4.1). They also observed an approximately linear relation between preference-model score and √D_KL(π‖π_0) during RLHF training (§4.3).
- In a synthetic gold-RM study with 1.2B policies, a nonzero KL penalty behaved like early stopping and did not raise the gold-score-versus-KL frontier. The authors describe this result as hyperparameter-sensitive ([[reward-model-overoptimization]] §3.6).
- Huang et al.'s 1B TL;DR PPO runs reached a KL of about 50 and 85. They had higher RLHF reward, but GPT-3.5 preferred them over reference summaries less than 20% of the time, and the samples contained concatenated strings without spaces ([[n-implementation-details-rlhf-ppo]] §7.1, β = 0.05).

These results are **Replicated** in direction ([[reward-model-overoptimization]]; [[n-implementation-details-rlhf-ppo]]): a KL penalty alone does not prevent proxy over-optimization in every setting. That the best β depends on RM scale, reward normalization, and model size is an **Interpretation**: Llama 2 used different β per size without an ablation, and no source in this chapter sweeps β across these factors.

**Implication.** The KL penalty keeps the policy near π_SFT in distribution over RL prompts. It does not target abilities that π_SFT has on other inputs, which is why §6 needs a separate mechanism.

## §5 InstructGPT: the recipe and its model names

**Naming.** InstructGPT defines three trained model types ([[rlhf-instructgpt]] §3.5):
- **SFT**: GPT-3 fine-tuned on labeler demonstrations.
- **PPO**: the SFT model fine-tuned with PPO on the RM reward with a per-token KL penalty, with γ = 0.
- **PPO-ptx**: PPO with pretraining gradients mixed in (γ > 0).

"Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models" (§3.5).

**Objective** (Eq. 2):

```
objective(φ) = E_{(x,y)∼D_{π_φ^RL}} [ r_θ(x,y) − β log( π_φ^RL(y|x) / π^SFT(y|x) ) ] + γ E_{x∼D_pretrain} [ log π_φ^RL(x) ]
```

- π_φ^RL: the RL policy with parameters φ. π^SFT: the SFT model (also the KL reference). r_θ: the RM.
- D_{π_φ^RL}: prompts x from the RL prompt set with responses y sampled from the policy. D_pretrain: the GPT-3 pretraining distribution.
- β: KL reward coefficient. γ: pretraining loss coefficient; γ = 0 for "PPO" models.

**Settings stated in the paper** (App. C.3, C.4, E.9, E.11):
1. RL policies are initialized from SFT models trained for 2 epochs on demonstrations with 10% pretraining data mixed in. This differs from the 16-epoch SFT baseline described in §3.5. The init models also serve as the KL reference.
2. β = 0.02. RL runs use 256k episodes covering about 31k unique prompts. The batch is 512 per iteration, split into 8 minibatches of 64, with a single inner epoch.
3. Constant learning rate with warm-up over the first 10 iterations, starting at one tenth of the peak. The policy's peak learning rate is not printed as one value. For 1.3B and 6B, a log-linear sweep from 2.55e-6 to 2.55e-5 was run, and all runs above 8.05e-6 without the pretraining mix diverged. For 175B, 2.55e-6 and 3.74e-6 were tried. Final checkpoints were those with the highest Likert scores (E.9).
4. EMA of weights with decay 0.992; no discount in GAE; clip ratio 0.2; rollout sampling temperature 1.
5. 6B RM and 6B value function; value learning rate 9e-6 (1.3B, 6B) and 5e-6 (175B).
6. PPO-ptx: 8 times more pretraining examples than RL episodes, gradients from PPO and pretraining accumulated in consecutive steps per minibatch, pretraining gradients multiplied by γ = 27.8.
7. The RM is shifted by a bias so that labeler demonstrations score 0 on average before RL (§3.5).

**Derived counts.** 256,000 episodes / 512 per iteration = 500 PPO iterations. 500 × 8 minibatches = 4,000 optimizer steps. 8 × 256,000 = 2,048,000 pretraining examples. These assume "256k" means 256,000 (**derived**).

**Headline result.** Outputs of the 1.3B InstructGPT were preferred to the 175B GPT-3. The 175B InstructGPT was preferred to 175B GPT-3 85 ± 3% of the time and to few-shot 175B GPT-3 71 ± 4% of the time (§4.1). "Adding updates on the pretraining mix during PPO does not lead to large changes in labeler preference" (§4.1). On prompts submitted to GPT-3 models on the API, PPO-ptx models performed slightly worse at larger model sizes (§4.1, Figure 3).

## §6 The alignment tax and its mitigations

### §6.1 What InstructGPT measured

**Definition.** The alignment tax is the drop in performance on tasks that the alignment procedure did not target. InstructGPT measures it as regressions relative to GPT-3 on public NLP datasets ([[rlhf-instructgpt]] §1, §4.2).

**Evidence** (175B columns of Table 14; F1 for SQuADv2 and DROP, accuracy for HellaSwag, BLEU for WMT15 FR→EN):

| Task, setting | GPT-3 | SFT | PPO (γ = 0) | PPO-ptx (γ = 27.8) | PPO − GPT | PPO-ptx − GPT |
|---|---|---|---|---|---|---|
| SQuADv2, few-shot | 69.75 | 65.90 | 51.95 | 69.93 | −17.80 | +0.18 |
| SQuADv2, zero-shot | 64.30 | 57.67 | 43.68 | 59.85 | −20.62 | −4.45 |
| DROP, few-shot | 35.27 | 35.85 | 27.78 | 33.34 | −7.49 | −1.93 |
| DROP, zero-shot | 27.53 | 15.79 | 13.08 | 15.23 | −14.45 | −12.30 |
| HellaSwag, few-shot | 0.791 | 0.741 | 0.759 | 0.820 | −0.032 | +0.029 |
| FR→EN, few-shot | 39.93 | 35.07 | 26.58 | 36.76 | −13.35 | −3.17 |
| FR→EN, zero-shot | 38.92 | 36.90 | 24.16 | 34.28 | −14.76 | −4.64 |

Three readings of the table:
1. PPO-ptx recovers most of the PPO regression but not all. The authors state that PPO-ptx "still lags behind GPT-3 on DROP, SQuADv2, and translation" and surpasses GPT-3 on HellaSwag (§4.2).
2. Part of the drop already appears at SFT. For zero-shot DROP, SFT is 11.74 points below GPT-3, and PPO adds 2.71 more. The size of the "RLHF tax" therefore depends on whether the baseline is the base model or the SFT model.
3. The authors describe the PPO model's regressions as occurring on many datasets, "particularly in the few-shot setting" (App. E.1). At 175B the zero-shot drops on SQuADv2, DROP, and FR→EN are larger in absolute points than the few-shot drops, so this statement does not hold for every task in the table.

Panel 3 of [figures/ppo-clip.html](figures/ppo-clip.html) plots these numbers for comparison across settings.

### §6.2 PPO-ptx: mechanism and ablations

**Mechanism.**
1. For each PPO minibatch, compute the PPO gradient on sampled responses.
2. Sample pretraining text and compute the gradient of its log-likelihood.
3. Multiply the pretraining gradient by γ and accumulate both before the optimizer step (App. C.4).

The ptx term is a positive log-likelihood anchor on the distribution where the lost abilities were learned. The KL penalty only anchors the policy on RL prompts.

**Ablations** (1.3B unless stated, App. E.6-E.11):
- γ ≥ 20 recovered the regressions. Validation reward fell as γ increased, and one value, 27.8, worked from 1.3B to 175B. Human Likert scores were insensitive to the exact γ (E.6).
- With γ = 0, raising β up to 2.0, 100 times the default, did not fix the regressions on DROP and SQuADv2 and caused a significant drop in validation reward (E.6, Figure 34). In this sweep the KL reference was the pretrained GPT model, not the SFT model. **Result (single study)**.
- Training for 512k instead of 256k episodes, with the pretraining mix and three seeds, brought DROP and SQuADv2 slightly below GPT-3 (E.6, Figure 35).
- Pretraining data ratio: at 4× the pretraining log-probability loss often increased during training. Preliminary runs at 32× gave better Likert scores at a few-fold training time. 8× doubled training time relative to no mix (E.11).
- PPO with the pretraining mix was less sensitive to the learning rate (E.9).

**Limits.** The authors state that the ptx term "does not completely mitigate performance regressions, and may make certain undesirable behaviors more likely for some tasks (if these behaviors are present in the pretraining data)" (§5.4).

### §6.3 Scale dependence in Anthropic's HH-RLHF study

[[hh-rlhf]] trained 7 model sizes from 13M to 52B with PPO against preference models (§3.1, §4.1). Findings (§1.1, §4.6.1, Figure 3):
- "Smaller models experience severe 'alignment taxes'". Their mean accuracy on MMLU, Lambada, HellaSwag, OpenBookQA, ARC-Easy, ARC-Challenge, and TriviaQA declined after RLHF.
- The 13B and 52B RLHF models performed better at zero-shot NLP evaluations and the same at few-shot evaluations. In every case except TriviaQA, the 12B and 52B RLHF models performed better than the base models (§1.2).
- Measurement caveat from the authors: the multiple-choice format with explicit choices "tends to improve performance for large models, while decreasing the performance of small models, leading to the arguably misleading appearance of a 'grok' curve" (§4.6.1).

InstructGPT (1.3B-175B, API prompts, KL β = 0.02) found regressions at 175B. Anthropic (13M-52B, HH dialogue data, λ_KL = 0.001) found gains at 13B and 52B. The two studies differ in data, evaluation format, and baseline, so the scale effect is an **Open question**, not a replicated law.

### §6.4 Model averaging as a mitigation

Lin et al. study the tax as forgetting relative to the SFT model θ_0 ([[mitigating-alignment-tax-rlhf]], arXiv:2309.06256v4 §3-4):

```
θ_α = (1 − α) θ_0 + α θ,     α ∈ [0, 1]
```

- θ_0: the instruction-tuned model before RLHF (OpenLLaMA-3B on ShareGPT). θ: the model after RLHF.
- α: interpolation weight. α = 1 is the RLHF model, α = 0 is the SFT model.

**Setting.** Main experiments use rejection-sampling fine-tuning on the HH-RLHF dataset. Findings are also checked with PPO and DPO and extended to Mistral-7B. The tax is measured on ARC Easy/Challenge, RACE, PIQA (accuracy), SQuAD and DROP (F1), and WMT14 FR→EN (BLEU) (§3).

**Findings.**
- As RLHF reward rose, translation and reading comprehension fell, while commonsense QA first rose and then fell (§4, App. E.1).
- Early stopping, L1/L2 regularization toward θ_0, LoRA, knowledge distillation, stochastic moving averaging, and model averaging all reduce the tax at some cost in reward. Model averaging's Pareto front "supersedes nearly all other methods across various hyper-parameters" (§4.1, Figure 3).
- For PPO, KL penalties of 0.05, 0.1, and 0.2 "partially mitigate the forgetting issue", and model averaging was "much more effective" in the alignment-forgetting trade-off (App. C.2, Figure 8).
- Replaying pretraining data at up to 4 times the RLHF data (the paper states 400M tokens for this comparison) outperformed model averaging only on reading comprehension, and underperformed it on commonsense QA and translation. The authors attribute this to replay covering about 0.03% of the 1.2T-token pretraining set (App. C.1; §2 of the same paper gives about 0.01%).
- Averaging only the lower third of the transformer improved both alignment reward and NLP scores. Heterogeneous Model Averaging (HMA), with separate ratios for K = 3 layer groups, pushed the front further (§5-6).

**Conditions.** Results are read from figures, without numeric tables for the main comparison. The models are 3B and 7B. The KL comparison uses a different reward scale from InstructGPT. The replay result conflicts in direction with InstructGPT's PPO-ptx result at 1.3B-175B. The two setups differ in data access (GPT-3's own pretraining distribution against a 0.03% subset) and replay ratio (8× against up to 4×). **Open question** which mitigation is better at a given scale. [[ch-30c]] covers weight averaging in general.

### §6.5 Calibration after RLHF: GPT-4

The GPT-4 technical report states that "the pre-trained model is highly calibrated" and that "after the post-training process, the calibration is reduced" ([[gpt-4-technical-report]] §5). On a subset of MMLU, expected calibration error (ECE) was 0.007 for the pre-trained model and 0.074 for the post-trained model, whose plot is labeled "model=ppo" (Figure 8). On the multiple-choice portions of the exam benchmark, the base model averaged 73.7% and the RLHF model 74.0% (App. B, Table 8). **Result (single study)**. The report does not separate SFT from RL effects on calibration.

**Implication for a general-purpose model.** The two measurements use different benchmarks: exam multiple-choice accuracy was about unchanged (73.7% to 74.0%), and ECE on an MMLU subset rose from 0.007 to 0.074. A capability report that lists accuracy only would not show the calibration change. Calibration and abstention metrics belong in the evaluation gate ([[ch-49]], [[ch-52]]).

## §7 Generalization reported by InstructGPT

**Held-out labelers.** A separate set of labelers who produced no training data ranked InstructGPT outputs similarly to the training labelers, and all InstructGPT models "still greatly outperform the GPT-3 baselines" by their judgment ([[rlhf-instructgpt]] §4.1, Figure 3). Inter-annotator agreement was 72.6 ± 1.5% among training labelers and 77.3 ± 1.3% among held-out labelers (§3.4). Held-out labelers came from the same vendors but did not take the screening test (§3.4).

**RM cross-validation.** Splitting labelers into 5 groups and training on 4, the RMs predicted held-out-group preferences with 69.6 ± 0.9% accuracy, against 72.4 ± 0.4% within the training groups (§4.1). The authors conclude that the models are not only fitting the preferences of the training labelers. They also note that more work is needed on broader user groups and inputs where humans disagree (§1).

**Code and non-English instructions.** The dataset is over 96% English (§3.3). The 175B PPO-ptx model follows instructions in other languages and answers questions about code more reliably than GPT-3, but it "often produces an output in English even when the instruction is in another language" (§4.3). These results are qualitative: "We do not track these behaviors quantitatively" (§4.3, Figure 8, cherry-picked prompts with non-cherry-picked outputs).

**Public NLP task collections are a different distribution.** GPT-3 175B fine-tuned on FLAN or T0++ performed slightly worse than the SFT baseline on API prompts. Against the SFT baseline, InstructGPT had a 73.4 ± 2% win rate, T0 26.8 ± 2%, and FLAN 29.8 ± 2% (§1, §4.1). This result has two consequences for a generalist. Training on academic task collections did not produce better results on real user prompts. In the other direction, the RLHF models regressed on academic tasks (§6.1).

**Failure modes.** The 175B PPO-ptx model accepted false premises and over-hedged simple questions (§4.3, Figure 9).

**Output diversity.** Anthropic reports that RL-finetuned models "typically have much narrower, lower-entropy output distributions" ([[hh-rlhf]] §4.6). Llama 2 reports reduced Self-BLEU diversity on factual prompts after RLHF, with diversity kept on creative prompts ([[llama-2]] §5.1, Figure 21). The trade-off between RL generalization and diversity is the subject of [[ch-38a]].

## §8 Llama 2 PPO: two reward models, per-size KL, and false refusal

**Schedule.** "Until RLHF (V4), we used only Rejection Sampling fine-tuning, and after that, we combined the two sequentially, applying PPO on top of the resulted Rejection Sampling checkpoint before sampling again" ([[llama-2]] §3.2.3). The published PPO comparison is RLHF-V5 with and without PPO, and the PPO version has higher win rates against ChatGPT under both the Meta RM and GPT-4 judges (Figure 11, read from the plot). The wording does not establish whether V4 itself used PPO.

**Reward** (Eq. 4 and the piecewise rule, §3.2.3):

```
R_c(g|p) = R_s(g|p)  if is_safety(p) or R_s(g|p) < 0.15,   else R_h(g|p)
R̃_c(g|p) = whiten( logit( R_c(g|p) ) )
R(g|p) = R̃_c(g|p) − β D_KL( π_θ(g|p) ‖ π_0(g|p) )
```

- p: prompt; g: generation; R_s, R_h: safety and helpfulness RM scores in (0, 1); π_0: the original policy.
- 0.15: threshold for filtering unsafe responses, chosen for precision 0.89 and recall 0.55 on the Meta Safety test set.
- whiten: normalization to zero mean and unit variance. The authors found it "important to whiten the final linear scores ... in order to increase stability and balance properly with the KL penalty term (β)".

**Worked example.** A prompt not tagged as safety-related receives R_s = 0.10 and R_h = 0.90. Because R_s < 0.15, R_c = 0.10 and logit(0.10) = ln(0.10/0.90) = −2.197 before whitening. The same response with R_s = 0.20 would use R_h, with logit(0.90) = +2.197. The rule is a hard switch, not a weighted sum.

**Settings** ([[llama-2-recipe]]; [[llama-2-ppo]]). AdamW with β1 = 0.9, β2 = 0.95, eps = 1e-5, weight decay 0.1, gradient clipping 1.0, constant learning rate 1e-6 for all models. Batch 512, clip 0.2, minibatch 64, one gradient step per minibatch, one generation per prompt. β = 0.01 for 7B and 13B, and β = 0.005 for 34B and 70B. Training ran 200-400 iterations with early stopping on held-out prompts. The report gives no ablation for β or the learning rate, and no maximum response length or sampling temperature for PPO. Each 70B PPO iteration took about 330 seconds.

**False refusal as the helpfulness-safety trade-off.** Llama 2 defines false refusal as refusing a legitimate prompt "due to irrelevant safety concerns", and measures it with a refusal classifier ([[llama-2]] §4.2.3). The measurement comes from the safety-data scaling ablation: helpfulness data fixed at about 0.9M samples, safety data from 0% to 100% of about 0.1M samples. It is not a PPO-only experiment. As safety data rose, the mean helpfulness RM score stayed about constant (Figure 15). False refusal on the helpfulness test set rose from 0.006% (1 occurrence) to 0.05% (8 occurrences). On a 210-prompt borderline set, whose prompts contain sensitive words but are safe, it rose from 15% to 27% (Figure 33). The model "sometimes has difficulty distinguishing whether a prompt is safe when the prompt contains words that frequently occur in unsafe generations (such as 'bomb')" (§4.2.3).

**Implication.** An aggregate helpfulness score did not show the change, while a targeted borderline set showed an increase of 12 percentage points. Over-refusal evaluation is covered in [[ch-52]].

## §9 Implementation details that change results

**From the 37 details of PPO** ([[costa-huang-ppo-details]], ICLR Blog Track 2022). The 13 core details, as numbered in the post, are: (1) vectorized architecture, (2) orthogonal initialization, (3) Adam epsilon 1e-5, (4) Adam learning-rate annealing, (5) GAE, (6) minibatch updates with shuffling, (7) advantage normalization at the minibatch level, (8) clipped surrogate objective, (9) value-function loss clipping, (10) overall loss and entropy bonus, (11) global gradient clipping at norm 0.5, (12) debug variables, (13) shared or separate policy and value networks. The evidence reported in the post comes from control and Atari tasks. Per-minibatch advantage normalization did "not affect performance much", value-loss clipping showed no evidence of helping, and global gradient clipping gave "a small performance boost". None of these was measured on language models in that post.

**From the N+ implementation details of RLHF with PPO** ([[n-implementation-details-rlhf-ppo]], arXiv:2403.17031v1; Pythia 1B, 2.8B, 6.9B on TL;DR summarization, 4 seeds):
- Detail 7: disable dropout. With dropout active, token log-probabilities are not reproducible, the KL penalty becomes unreliable, and PPO ratios are not 1 in the first epoch.
- Detail 22: initialize the value model from the RM (§3).
- Detail 23: the "EOS trick". A completion without an EOS token receives a constant −1 score, because the RM's score is defined only at EOS. This also penalizes long completions that hit the length limit. TRL exposes it as `missing_eos_penalty` ([[trl-ppo]] L759-761).
- Detail 24: optional reward whitening. In their ablation it produced shorter completions and a lower preference rate, and length-controlled comparisons were similar with and without it (§7.1, Figures 11-12).
- Detail 25: advantage whitening.
- Results: GPT-3.5 preferred the best 6.9B PPO model's summaries to reference summaries nearly 80% of the time. PPO responses were longer than SFT responses, and PPO still outperformed SFT at every length ratio after controlling for length (§7.1, Figure 12).

**From the model reports.** InstructGPT used EMA of weights (decay 0.992) and a bias shift of the RM. Llama 2 whitened the logit of the reward. Both used a batch of 512 and a minibatch of 64. InstructGPT found a minibatch of 32 "slightly better than 64" at 1.3B but kept 64 for GPU utilization, and a batch of 512 best among 64-1024 by human evaluation ([[rlhf-instructgpt]] App. E.11).

**Framework defaults are not paper values.** TRL's `PPOConfig` defaults at commit a08e713 are learning rate 3e-6, 4 PPO epochs per batch, `kl_coef` 0.05 with k1, cliprange 0.2, `vf_coef` 0.1, γ = 1.0, λ = 0.95, temperature 0.7, and response length 53 tokens ([[trl-ppo-recipe]]). They follow the N+ TL;DR setup (Table 7) and differ from InstructGPT's single inner epoch and β = 0.02. A value such as a "PPO learning rate of 1.41e-5 with 4 epochs" does not appear in the InstructGPT paper.

## Negative samples and negative feedback

**Which meaning.** PPO-based RLHF uses negatives as gradient (meaning 4 in the course's taxonomy): tokens with Â_t < 0 have their probability pushed down. The RM itself was trained with rejected responses as negative gradient ([[ch-41]]). The pretraining term in PPO-ptx is a positive log-likelihood anchor, not a negative.

**Where negatives come from in this stage.**
1. Low RM score relative to the value baseline gives negative advantages on the tokens of that response.
2. The per-token KL reward −β·k1 is negative wherever log π_θ > log π_ref.
3. Constant penalties: the EOS trick's −1 for truncated completions ([[n-implementation-details-rlhf-ppo]] Detail 23).
4. In Llama 2, the safety RM replaces the helpfulness RM on safety-tagged prompts or when R_s < 0.15, so a response can be pushed down for a safety score on a prompt that is safe. This mechanism is consistent with the false refusals in §8, but the false-refusal ablation did not isolate PPO (**Interpretation**).
No false-negative rate is reported for any of these labels in the sources of this chapter.

**Mechanism.** For a softmax policy with logits z, ∂ log p_y / ∂ z_j = 1[j = y] − p_j. The per-token policy-gradient contribution is Â_t (1[j = y] − p_j). With Â_t < 0, the sampled token's logit falls and every other logit rises in proportion to its current probability p_j. Worked example: p = (0.7, 0.2, 0.1), sampled token 3, Â = −1, one unit step on the logits. The logit change is (+0.7, +0.2, −0.9), and the new probabilities are (0.832, 0.144, 0.024). The removed mass (0.076 from token 3 and 0.056 from token 2) went to the already most likely token 1. A push-down on an unlikely token therefore concentrates mass on the mode. In PPO the ratio for token 3 would be 0.024/0.1 = 0.24, below 1 − ε = 0.8, so later minibatches in the same batch get zero gradient for this token (§2, case 4).

**Controls present in KL-controlled PPO.**
- Clipping bounds the push-down per batch at ratio 1 − ε. It does not bound the loss when a negative-advantage token's ratio rises (§2, case 3). Dual-clip bounds that case ([[openrlhf-ppo]]).
- The KL reward pulls mass back toward π_ref on RL prompts (§4).
- The pretraining term (PPO-ptx) or replay adds a positive NLL anchor on other data (§6.2).
- Reward whitening and the bias shift keep reward scale comparable to β (§8, §9).
- Advantage whitening centers advantages, so about half of the tokens in a batch receive negative advantages regardless of absolute quality (**Interpretation**; follows from mean subtraction).

**Evidence for failure modes with numbers.** Over-optimization at KL ≈ 50-85 with a win rate below 20% ([[n-implementation-details-rlhf-ppo]] §7.1). Borderline false refusal from 15% to 27% as safety data grows ([[llama-2]] Figure 33). ECE from 0.007 to 0.074 after post-training ([[gpt-4-technical-report]] Figure 8). None of these sources separates the effect of negative-advantage tokens from positive ones. The share of the gain or damage attributable to negatives is **not reported** for PPO-based RLHF. [[ch-43a]] covers studies that measure it for other algorithms.

**Diagnostics.** Log mean advantage, clip fraction, and log-probability change separately for Â > 0 and Â < 0 tokens. Log policy entropy, KL to π_ref (sum of k1 over tokens), `approxkl` between consecutive policies ([[trl-ppo]]), and the rate of completions without EOS. For generality, also log the borderline refusal rate, ECE on a held-out multiple-choice set, and pass@k at a large k on a reasoning set.

**Effect on generality.** Negative gradients concentrate mass on already likely outputs. This is consistent with the reported loss of output diversity (§7) and calibration (§6.5), but no source in this chapter isolates negatives as the cause (**Open question**).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| InstructGPT PPO-ptx ([[rlhf-instructgpt]]) | 1.3B, 6B, 175B | RL | KL coefficient β, placement | 0.02; per-token log-ratio to the SFT init model, in the reward | arXiv:2203.02155v1 App. C.4, Eq. 2 | verified 2026-09-15 | App. E.7 Figure 36: Likert optimum "around 0.01 and 0.02"; 0 and 2 poor |
| same | 1.3B, 6B, 175B | RL | pretraining coefficient γ; pretraining examples | 27.8; 8× RL episodes | App. C.4 | verified 2026-09-15 | App. E.6: γ ≥ 20 recovers regressions at 1.3B; E.11: ratio 4 vs 8 vs 32 |
| InstructGPT PPO (γ = 0) | 1.3B, 6B, 175B | RL | pretraining coefficient γ | 0 | §3.5 | verified 2026-09-15 | Table 14 regressions (§6.1) |
| InstructGPT PPO and PPO-ptx | 1.3B, 6B, 175B | RL | episodes; unique prompts | 256k; about 31k | App. C.4 | verified 2026-09-15 | E.11: no benefit beyond 256k at 1.3B with ptx; E.6: 512k regresses |
| same | 1.3B, 6B, 175B | RL | batch; minibatch; inner epochs | 512; 64 (8 minibatches); 1 | App. C.4 | verified 2026-09-15 | E.11: batch 512 best of 64-1024; minibatch 32 slightly better than 64, 64 kept for GPU utilization (1.3B) |
| same | 1.3B, 6B | RL | policy learning rate | not printed as a single value; sweep 2.55e-6 to 2.55e-5, checkpoint with highest Likert chosen | App. C.4, E.9 | not reported (C.4, E.9, Figure 38 checked) | E.9: runs above 8.05e-6 without ptx diverged |
| same | 175B | RL | policy learning rate | not printed; 2.55e-6 and 3.74e-6 tried | App. E.9 | not reported | E.9 Figure 38 |
| same | 1.3B, 6B, 175B | RL | LR schedule; EMA; GAE discount; clip; temperature | constant with 10-iteration warm-up from 1/10 of peak; EMA decay 0.992; no discount; 0.2; 1 | App. C.4 | verified 2026-09-15 | no ablation reported |
| same | 1.3B, 6B / 175B | RL | value model; value learning rate | 6B, initialized from the 6B RM; 9e-6 / 5e-6 | App. C.2, C.4 | verified 2026-09-15 | C.2: 175B RM less stable as value initialization |
| InstructGPT PPO init models | 1.3B, 6B, 175B | SFT | epochs; pretraining mix; batch; selected LR | 2; 10%; 32 (1.3B, 6B), 8 (175B); 5e-6, 1.04e-5, 2.45e-6 | App. C.3 | verified 2026-09-15 | E.8, Figure 37: variants with 0/10/50% pretraining data and 1-2 epochs |
| Llama 2-Chat ([[llama-2-recipe]]) | 7B, 13B | RL | KL coefficient β | 0.01, in the reward (Eq. 4) | arXiv:2307.09288v2 §3.2.3 | verified 2026-09-14 | no ablation reported |
| Llama 2-Chat | 34B, 70B | RL | KL coefficient β | 0.005, in the reward | §3.2.3 | verified 2026-09-14 | no ablation reported |
| Llama 2-Chat | all | RL | optimizer; LR; batch; minibatch; clip; iterations | AdamW (0.9, 0.95, eps 1e-5), wd 0.1, grad clip 1.0; constant 1e-6; 512; 64, one step each; 0.2; 200-400 with early stopping | §3.2.3 | verified 2026-09-14 | no ablation reported |
| Llama 2-Chat | all | RL | reward | R_s if safety-tagged or R_s < 0.15, else R_h; whiten(logit) | §3.2.3 | verified 2026-09-14 | threshold: precision 0.89, recall 0.55 on Meta Safety test set |
| Llama 2-Chat | all | RL | max response length; sampling temperature | not printed | §3.2.3 | not reported (body and A.3 checked) | n/a |
| Anthropic HH RLHF policies ([[hh-rlhf]]) | 13M-52B | RL | KL coefficient λ_KL | 0.001 | arXiv:2204.05862v1 §4.1, Eq. 4.1 | verified 2026-09-15 | authors: "might actually be wholly unnecessary"; no ablation |
| Pythia TL;DR PPO ([[n-implementation-details-rlhf-ppo]]) | 1B, 2.8B, 6.9B | RL | episodes; optimizer; schedule; batch | 1,000,000 (about 8.56 epochs); AdamW eps 1e-5, LR 3e-6; linear; 512 | arXiv:2403.17031v1 Table 7 | verified 2026-09-15 | "closely follows Stiennon et al. (2020), except for a modified learning rate" (Detail 20) |
| same | 1B, 2.8B, 6.9B | RL | β; γ; λ; minibatches; PPO epochs; ε; value clip; c_1; temperature | 0.05; 1.0; 0.95; 1; 4; 0.2; 0.2; 0.1; 0.7 | Table 7 | verified 2026-09-15 | §7.1: 1B runs over-optimized (KL about 50 and 85) at these settings |
| TRL PPOConfig default ([[trl-ppo-recipe]]) | any | RL | LR; PPO epochs; kl_coef; estimator; response length | 3e-6; 4; 0.05; k1; 53 | trl@a08e713 ppo_config.py | verified 2026-09-14 | framework default, no ablation |

**Starting point for a small general-purpose run.** For a 1B-7B policy with a separate RM, the verified rows support the following configuration. KL coefficient in the reward: 0.01-0.02 as a first sweep. This range is InstructGPT's Likert optimum (models up to 175B, API prompts) and Llama 2's 7B/13B value. Batch 512 with minibatch 64 and one step per minibatch (InstructGPT and Llama 2), clip 0.2, γ = 1, λ = 0.95 (N+ TL;DR, 1B-6.9B). A value model initialized from the RM (InstructGPT; N+). A constant learning rate chosen by sweep, because InstructGPT's 1.3B/6B runs without ptx diverged above 8.05e-6 and Llama 2 used 1e-6 at 7B-70B. If pretraining or mid-training data is available, add a log-likelihood term with 8 times as many pretraining examples as episodes and γ = 27.8. That setting was tested at 1.3B-175B on GPT-3's own pretraining distribution. Otherwise, keep the SFT checkpoint and evaluate interpolations θ_α (OpenLLaMA-3B, Mistral-7B). No source tested this combination as a whole.

## Generalization lens

**(a) What increases breadth.**
- RLHF on API prompts improved preference by labelers who produced no training data, and RMs kept 69.6 ± 0.9% accuracy on held-out labeler groups ([[rlhf-instructgpt]] §4.1). **Result (single study)**.
- Instruction following transferred qualitatively to code questions and non-English instructions from a dataset over 96% English ([[rlhf-instructgpt]] §4.3). Not quantified.
- At 13B and 52B, RLHF raised zero-shot NLP accuracy ([[hh-rlhf]] §1.1). Natural-language HH RLHF on Python-finetuned models lowered HumanEval pass@1 for smaller models and raised it for larger models, and at 52B the gain held at large k in pass@k ([[hh-rlhf]] Figure 21). **Result (single study)**, format-sensitive.
- Pretraining-gradient mixing (γ = 27.8) restored or exceeded GPT-3 on few-shot SQuADv2 and HellaSwag at 175B ([[rlhf-instructgpt]] Table 14).
- Interpolating toward the SFT weights, especially in lower layers, raised NLP scores while keeping most of the reward ([[mitigating-alignment-tax-rlhf]] §5-6).

**(b) What causes narrowing or forgetting.**
- PPO without pretraining gradients: −17.80 F1 few-shot SQuADv2 and −13.35 BLEU few-shot FR→EN at 175B ([[rlhf-instructgpt]] Table 14). The tax appeared at small scale in Anthropic's study ([[hh-rlhf]] Figure 3), and it grew with reward in OpenLLaMA-3B ([[mitigating-alignment-tax-rlhf]] §4). **Replicated** in direction across three studies with different scales and data.
- Longer training (512k episodes) with ptx brought DROP and SQuADv2 below GPT-3 ([[rlhf-instructgpt]] App. E.6).
- Proxy over-optimization at high KL: 1B TL;DR runs with KL of about 50-85 had a win rate below 20% ([[n-implementation-details-rlhf-ppo]] §7.1).
- Calibration: ECE 0.007 → 0.074 on MMLU ([[gpt-4-technical-report]] Figure 8).
- Over-refusal: borderline false refusal 15% → 27% with more safety data ([[llama-2]] Figure 33).
- Output diversity: narrower, lower-entropy distributions after RL ([[hh-rlhf]] §4.6); lower Self-BLEU diversity on factual prompts ([[llama-2]] Figure 21).
- A larger β alone did not remove the tax and cost reward ([[rlhf-instructgpt]] App. E.6; [[mitigating-alignment-tax-rlhf]] App. C.2). **Replicated**.

**(c) How to measure it for this stage.**
- Score a fixed public suite against both the base model and the SFT init model, and report PPO − base and PPO − SFT separately. In Table 14, 11.74 of the 14.45-point zero-shot DROP drop at 175B is already present at SFT.
- Report few-shot and zero-shot separately. The authors describe the PPO regressions as "particularly in the few-shot setting" (App. E.1), while at 175B the zero-shot drops on SQuADv2, DROP, and FR→EN are larger ([[rlhf-instructgpt]] Table 14).
- Use held-out labelers or judges, and cross-validated RM accuracy across annotator groups.
- Track gold or human score against √KL, not proxy reward ([[reward-model-overoptimization]]; [[hh-rlhf]] §4.3).
- Add a borderline over-refusal set ([[llama-2]] §4.2.3; [[ch-52]]) and a calibration metric (ECE) ([[gpt-4-technical-report]] Figure 8).
- Control for length when comparing preference win rates ([[n-implementation-details-rlhf-ppo]] Figure 12).
- Known measurement errors: multiple-choice formatting changes small and large models in opposite directions ([[hh-rlhf]] §4.6.1). Qualitative transfer claims are cherry-picked prompts ([[rlhf-instructgpt]] Figure 8). Aggregate refusal rates hide borderline behavior ([[llama-2]] §4.2.3). Figure-only results cannot be compared numerically across papers ([[mitigating-alignment-tax-rlhf]]).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Calling γ = 0 models "InstructGPT" | Claims that InstructGPT had no pretraining mix; tax attributed to the released model | §3.5: "InstructGPT refers to the PPO-ptx models"; γ = 0 models are "PPO" |
| Reading framework defaults as paper settings | "InstructGPT used 4 PPO epochs"; learning rate quoted to three digits | InstructGPT App. C.4: one inner epoch; policy LR not printed as a single value (E.9) |
| Using one KL coefficient for all sizes | Llama 2 70B quoted with β = 0.01 | Llama 2 §3.2.3: 0.01 for 7B/13B, 0.005 for 34B/70B |
| Calling the reward-side KL "k3" | Reward penalty always positive; no negative per-token KL rewards in logs | Eq. 2 and TRL default use k1 = log π_θ − log π_ref; k3 is a loss-side option |
| Raising β to fix capability regressions | Validation reward falls while DROP/SQuADv2 stay low | App. E.6 Figure 34: β up to 2.0 did not recover; add ptx or averaging |
| Measuring the tax only against the base model | RLHF blamed for drops already present after SFT | Score the SFT init checkpoint on the same suite (Table 14 SFT column) |
| Value model initialized randomly or shared without warm-up | Large value loss in early iterations; noisy advantages | Initialize from RM ([[rlhf-instructgpt]] C.4; N+ Detail 22); log explained variance |
| Dropout left on during PPO | Ratio ≠ 1 on the first minibatch; KL estimate nonzero before any update | N+ Detail 7: disable dropout; assert mean ratio = 1 at step 0 |
| No EOS handling | RM scores on truncated text; response length grows to the limit | Log the rate of completions without EOS; apply `missing_eos_penalty` (N+ Detail 23) |
| Judging progress by RM reward | Reward and KL rise while held-out judge win rate falls | Plot judge score against √KL; stop near its peak ([[reward-model-overoptimization]]) |
| Aggregate refusal rate as the only safety-helpfulness metric | 0.05% false refusal reported while borderline refusals are 27% | Separate borderline set ([[llama-2]] Figure 33) |
| Reporting accuracy but not calibration | MCQ accuracy flat, confidence uninformative | ECE on a held-out MCQ set before and after RL ([[gpt-4-technical-report]] Figure 8) |
| Trusting clipping to bound negative-advantage updates | Loss spikes on tokens with Â < 0 and large ratio | Log clip fraction and ratio by advantage sign; consider dual-clip ([[openrlhf-ppo]]) |

## Check your understanding

1. InstructGPT found that raising β up to 2.0 did not recover DROP and SQuADv2, while γ = 27.8 did. Using the definition of the KL penalty (on which inputs is the KL measured?) and of the ptx term, explain why the two regularizers protect different capabilities.
2. In Table 14, zero-shot DROP falls by 11.74 F1 from GPT-3 to SFT and by 2.71 more from SFT to PPO. What does this imply for attributing a tax to RLHF, and which checkpoint would you use as the baseline for a forgetting report in your own pipeline?
3. For a token with Â = −1 and ε = 0.2, the PPO gradient is zero at r = 0.7 and nonzero at r = 1.5. Explain why the min in L^CLIP produces this asymmetry, and describe a training situation in an LLM rollout where r > 1 for a negative-advantage token.
4. The per-token term −β(log π_θ − log π_ref) can be positive for some tokens. Why is this acceptable for the reward, and why would clipping it at zero change the objective's optimum π* ∝ π_ref exp(r/β)?
5. Anthropic found alignment bonuses at 13B and 52B, and InstructGPT found regressions at 175B. List three differences between the setups that could explain the opposite directions, and design one experiment that would separate model scale from evaluation format.
6. Llama 2's helpfulness RM score stayed constant as safety data increased, while borderline false refusals rose from 15% to 27%. Explain how the piecewise reward rule with a 0.15 threshold could produce refusals on safe prompts during PPO, and what data you would add to the RM or PPO prompt set to test that explanation.
7. GPT-4's exam multiple-choice accuracy was 73.7% before and 74.0% after RLHF, while ECE on an MMLU subset rose from 0.007 to 0.074. Explain how a policy-gradient objective on preference rewards can change calibration without changing accuracy. Then explain why the use of two different benchmarks limits this comparison.
8. Replay of 4× RLHF data underperformed model averaging for OpenLLaMA-3B, while InstructGPT's 8× pretraining mix worked at 175B. Using data coverage and anchor strength, explain how both results can hold.

## Connections

- **Previous:** ch-37 — Policy-Gradient Foundations for Language Models (score-function estimator, baselines, and the actor-critic view that §1-§3 build on).
- **Next:** ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity (the diversity and out-of-distribution trade-off raised in §7).
- ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization (the RM that supplies r; over-optimization curves used in §4).
- ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control (the SFT-stage counterpart of §6).
- ch-30c — Weight Averaging and Model Merging for Generalist Models (model averaging, §6.4).
- ch-39 — Offline Preference Optimization: DPO and Its Variants (the same KL-regularized optimum π* without online sampling).
- ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO (critic-free advantages; loss-side KL).
- ch-42 — Reward Hacking and Judge Design (failure modes when the RM is exploited).
- ch-43 — Entropy, Output Diversity, and KL Control in RL (entropy and KL diagnostics).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages (full treatment of negatives as gradient).
- ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting (calibration metrics).
- ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming (borderline false-refusal sets).
- ch-56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls; ch-57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers (framework code for §3-§4).

## Sources

- [[ppo]] — L^CPI, L^CLIP, adaptive KL rule, combined objective, truncated GAE, MuJoCo Table 1 (arXiv:1707.06347v2); excerpt in `excerpts/`.
- [[trpo]] — Theorem 1 bound, Eq. 9-12 constraint, δ = 0.01, conjugate-gradient cost (read in arXiv:1502.05477v5; the library card's hyperparameter table was not used); excerpt in `excerpts/`.
- [[rlhf-instructgpt]] — Eq. 2, model naming, App. C.2-C.4 settings, E.6-E.11 ablations, Table 14, held-out labelers, qualitative transfer (read in arXiv:2203.02155v1; the library card's hyperparameter table was not used); excerpt in `excerpts/`.
- [[hh-rlhf]] — λ_KL = 0.001, √KL relation, scale-dependent tax and bonus, format caveat, lower-entropy outputs (read in arXiv:2204.05862v1); excerpt in `excerpts/`.
- [[mitigating-alignment-tax-rlhf]] — model averaging, comparisons with regularizers, KL penalty and replay, HMA (arXiv:2309.06256v4); excerpt in `excerpts/`.
- [[llama-2]], [[llama-2-recipe]], [[llama-2-ppo]] — PPO schedule, piecewise reward, per-size β, settings, false-refusal study, diversity.
- [[costa-huang-ppo-details]] — numbering and evidence of the 13 core PPO details (read in the ICLR Blog Track post); excerpt in `excerpts/`.
- [[n-implementation-details-rlhf-ppo]] — Details 7 and 22-25, Table 7, over-optimization and length results (arXiv:2403.17031v1); excerpt in `excerpts/`.
- [[kl-control-rlhf]] — Korbak et al. Eq. 3-7, the KL-regularized optimum (read in arXiv:2205.11275v2); excerpt in `excerpts/`.
- [[john-schulman-kl-tricks]] — k1/k2/k3 definitions and Gaussian bias/variance numbers (read in the 2020-03-07 post); excerpt in `excerpts/`.
- [[trl-ppo]], [[trl-ppo-recipe]] — per-token reward and GAE code, value clipping, defaults at commit a08e713; excerpt in `excerpts/`.
- [[openrlhf-ppo]] — KL placement options, estimator recommendation, adaptive controller, dual-clip.
- [[reward-model-overoptimization]] — KL penalty acting like early stopping in the synthetic gold-RM setting.
- [[gpt-4-technical-report]] — calibration before and after post-training, exam accuracy base versus RLHF (arXiv:2303.08774v6); excerpt in `excerpts/`.
- [[lilianweng-rlhf]] — tutorial overview of the RLHF pipeline; not used as evidence for any number in this chapter.
