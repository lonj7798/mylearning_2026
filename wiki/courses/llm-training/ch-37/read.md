<!-- chapter: ch-37
     track: rl
     kind: content
     title: Policy-Gradient Foundations for Language Models
     deps: [ch-41]
     sources: [[vanilla-pg]], [[rloo]], [[trpo]], [[ppo]], [[maximum-entropy-rl]], [[rls-razor]],
              [[sft-memorizes-rl-generalizes]], [[rlhf-generalisation-diversity]], [[rlvr-beyond-base-model]],
              [[raft-reinforce-rej-minimalist]], [[reinforce-plus-plus]], [[dr-grpo]], [[grpo]],
              [[entropy-mechanism-llm-rl]], [[loop-appworld]], [[trl-grpo]], [[john-schulman-kl-tricks]]
     figures: figures/pg-variance.html
     revised: 2026-09 (generality revision)
-->

# Chapter 37 — Policy-Gradient Foundations for Language Models

> **Core insight.** The policy-gradient estimator increases the log-probability of the model's own sampled responses in proportion to an advantage, the reward minus a baseline. The baseline does not change the expected gradient when it does not depend on the sampled response, but it decides which individual samples are pushed up and which are pushed down: with 0/1 rewards and b = 0 no sample is pushed down. Because the gradient is an SFT gradient on self-generated samples, on-policy RL changes the model less than SFT on external targets; in controlled studies this came with less forgetting ([[rls-razor]]) and better transfer to unseen rule variants ([[sft-memorizes-rl-generalizes]]), and with lower output diversity ([[rlhf-generalisation-diversity]]) and lower coverage at large k ([[rlvr-beyond-base-model]]).
>
> **Guideline.** When several responses per prompt can be sampled, use a leave-one-out baseline, because it is exactly unbiased, unlike group-mean subtraction (scaled by (G − 1)/G) and group-std normalization (prompt-dependent scale); at G = 2 it can raise variance above b = 0 (§2), so use it with G ≥ 4. When group mean or group standard deviation is used instead, treat the result as a per-prompt reweighting and report it. When negative advantages are used, bound or anchor them (clipping, a KL or NLL term, on-policy sampling), because a negative weight on log π has no lower limit. When RL is used to build a general model, evaluate three splits (training reward, held-out prompts from the training distribution, other domains) and report pass@1 together with pass@k at large k.

## Why this chapter matters for a general-purpose model

The pipeline in this course runs pre-training → mid-training → SFT → preference optimization → RL → evaluation. This chapter opens the policy-optimization part. ch-41 built the reward model R(x, y). This chapter defines the gradient that turns R into parameter updates; ch-38 adds the KL-controlled PPO recipe, ch-39 removes sampling with offline preference losses, and ch-40 compares group-baseline methods.

Three facts from this chapter decide whether RL makes a model broader or narrower:

1. The gradient only touches prompts that are sampled. Behavior on other prompts changes only through shared parameters. The prompt distribution (ch-16) is therefore a design choice for generality, not a detail.
2. The samples are the model's own. This limits how far each update moves the model (§5), which is the proposed reason RL forgot less than SFT in [[rls-razor]].
3. The objective rewards concentration on high-reward responses. Diversity and large-k coverage can fall even while pass@1 rises (§6).

## §1 The objective and the log-derivative estimator

**Definition.** A policy π_θ(y | x) is the language model's distribution over responses y given a prompt x. The RL objective is the expected reward over a prompt distribution D:

```
J(θ) = E_{x∼D} E_{y∼π_θ(·|x)} [ R(x, y) ]
```

- θ: model parameters. D: the training prompt distribution. R(x, y): scalar reward (reward model, verifier, or environment).

**Problem.** The sampling distribution π_θ depends on θ, so the gradient cannot be obtained by differentiating R inside a fixed expectation. R is often not differentiable (a verifier returns 0 or 1).

**Mechanism.** The log-derivative identity ∇_θ π_θ(y|x) = π_θ(y|x) ∇_θ log π_θ(y|x) moves the gradient onto the log-probability:

```
∇_θ J(θ) = E_{x∼D} Σ_y ∇_θ π_θ(y|x) R(x, y)
         = E_{x∼D} E_{y∼π_θ(·|x)} [ R(x, y) ∇_θ log π_θ(y|x) ]
∇_θ log π_θ(y|x) = Σ_{t=1}^{T} ∇_θ log π_θ(y_t | x, y_<t)
```

- y_t: token t of the response; T: response length; y_<t: tokens before t.

This is the REINFORCE estimator of [[vanilla-pg]] applied to a whole response. [[rloo]] §2.2 models the full generation as one action for this reason: the reward is given only for the complete response (Eq. 6).

**Causal form for per-step rewards.** In a trajectory with rewards r_t at every step, a reward received before step t does not depend on the action at t, so its term has zero expectation. The estimator becomes Σ_t ∇ log π(a_t | s_t) G_t with the return-to-go G_t = Σ_{t′≥t} γ^{t′−t} r_{t′}. With a single terminal reward and γ = 1, G_t = R(x, y) for every t, and the causal form equals the sequence form above.

**Only sampled prompts receive gradient.** The Monte-Carlo estimate averages over a batch of prompts drawn from D. A prompt type that never appears in D contributes no term. Its behavior changes only through parameters shared with the sampled prompts; whether that change helps or hurts is an empirical question (Generalization lens).

**Worked example (checkable by hand).** A prompt has two possible answers. The policy is p = σ(θ) for "correct" (reward 1) and 1 − p for "wrong" (reward 0), with p = 0.2. The true gradient is dJ/dθ = p(1 − p) = 0.16. The score is d log p/dθ = 1 − p = 0.8 for "correct" and d log(1 − p)/dθ = −p = −0.2 for "wrong". The single-sample estimate is 0.8 with probability 0.2 and 0 with probability 0.8. Its mean is 0.2 × 0.8 = 0.16, equal to the true gradient. Its variance is 0.2 × 0.64 − 0.16² = 0.1024.

**Evidence and limits.** The identity is exact for any R. The estimator is unbiased only when y is sampled from the current π_θ; reusing samples from an older policy requires importance weights (§3, ch-38).

**Implication for a general model.** J is defined by D. A model trained on a narrow D is optimized for that D; nothing in the estimator protects prompts outside it.

## §2 Baselines: variance and bias

**Definition.** A baseline b is subtracted from the reward: the estimator uses the advantage A = R(x, y) − b.

**Problem.** The variance of R ∇ log π is large when rewards have a large common offset. In the §1 example, 80% of samples carry no signal and 20% carry a large one.

**Mechanism (unbiasedness condition).** If b does not depend on the sampled y (it may depend on x):

```
E_{y∼π_θ}[ b ∇_θ log π_θ(y|x) ] = b Σ_y ∇_θ π_θ(y|x) = b ∇_θ Σ_y π_θ(y|x) = b ∇_θ 1 = 0
```

So E[(R − b) ∇ log π] = ∇J. The minimum-variance constant baseline per coordinate i is b* = E[R s_i²] / E[s_i²], with s_i the i-th component of the score ∇ log π.

**Worked example (continued).** With b = 0.2 (the expected reward J): correct gives (1 − 0.2)(0.8) = 0.64, wrong gives (0 − 0.2)(−0.2) = 0.04. Mean 0.2 × 0.64 + 0.8 × 0.04 = 0.16, variance 0.0832 − 0.0256 = 0.0576. With b* = E[R s²]/E[s²] = (0.2 × 0.64)/(0.2 × 0.64 + 0.8 × 0.04) = 0.128/0.16 = 0.8: correct gives 0.2 × 0.8 = 0.16 and wrong gives (−0.8)(−0.2) = 0.16, so the variance is 0. All three baselines (0, 0.2, 0.8) give mean 0.16. The zero variance is specific to a one-parameter, two-action case; it shows that b = E[R] is not the minimum-variance constant in general.

**Sample-based baselines in LLM RL.** With G responses y_1..y_G per prompt and rewards r_1..r_G:

| Baseline for sample i | Depends on y_i? | Expected gradient | Source |
|---|---|---|---|
| b = 0 | no | ∇J | [[vanilla-pg]] |
| moving average of past rewards | no (past batches) | ∇J | [[rloo]] Eq. 8 |
| learned value V_φ(x) (critic) | no | ∇J (variance depends on V_φ error) | [[ppo]] Eq. 9–12 |
| leave-one-out: (1/(G−1)) Σ_{j≠i} r_j | no | ∇J | [[rloo]] §2.3 |
| group mean (1/G) Σ_j r_j, includes r_i | yes | ((G−1)/G) ∇J | [[dr-grpo]] App. A |
| (r_i − group mean) / group std | yes | prompt-dependent multiple of ∇J | [[reinforce-plus-plus]] App. A.1, [[dr-grpo]] §3.1 |

The group-mean row follows from r_i − mean = ((G − 1)/G)(r_i − mean_{−i}): the group-mean advantage is the leave-one-out advantage times (G − 1)/G. [[dr-grpo]] App. A states the same relation as "(G/(G−1))·Ã equals the RLOO advantage". A constant factor is equivalent to a smaller learning rate. The std division is not a constant: the std depends on r_i and on how many responses in the group are correct. [[reinforce-plus-plus]] App. A.1 (Theorem 1) proves the resulting advantage is biased for any finite N ≥ 2, and [[dr-grpo]] §3.1 names the effect a difficulty bias: questions whose rewards are almost all 1 or all 0 have small std and receive larger weight.

**Worked example: group of 4 with 0/1 reward.** Prompt P1 has one correct response: rewards (1, 0, 0, 0).
- b = 0: advantages (1, 0, 0, 0).
- Group mean 0.25: (0.75, −0.25, −0.25, −0.25).
- Leave-one-out: correct 1 − 0 = 1; each wrong 0 − 1/3 = −0.333. Multiplying by 3/4 gives the group-mean values.
- Group std with the n − 1 denominator (PyTorch's default, used by TRL's `rewards.view(-1, num_generations).std(dim=1)`, [[trl-grpo]] L2127–2149): sqrt((0.75² + 3 × 0.25²)/3) = 0.5, so advantages (1.5, −0.5, −0.5, −0.5).

Prompt P2 has two correct responses: rewards (1, 1, 0, 0), mean 0.5, std sqrt(4 × 0.25/3) = 0.577, advantage of a correct response 0.5/0.577 = 0.866. After std division a correct response on P1 receives weight 1.5 and one on P2 receives 0.866; after mean subtraction only, the weights are 0.75 and 0.5.

Taking the expectation over groups (enumerated exactly; this course's calculation, not a source result): for G = 8 and binary reward, the std-normalized estimator equals 1.736 ∇J for a prompt with success probability 0.5 and 2.313 ∇J for a prompt with success probability 0.05 or 0.95, a relative weight of 1.33. The group-mean estimator is 0.875 ∇J for every success probability.

**Variance of sample-based baselines.** A baseline estimated from other samples adds its own noise. For one prompt with four responses of probabilities (0.5, 0.3, 0.15, 0.05), responses 2 and 4 correct (J = 0.35), and G = 4, the exact total variance of the logit-gradient estimator is 0.0551 with b = 0, 0.0233 with b = J, and 0.0414 with leave-one-out. For G = 2 leave-one-out gives 0.155, above b = 0 (0.110). With responses 1, 2, 4 correct (J = 0.85) and G = 4, b = 0 gives 0.113 and leave-one-out 0.034. A leave-one-out baseline reduces variance most when the expected reward is far from 0 and G is not small.

**[figures/pg-variance.html](figures/pg-variance.html)** — lets the reader change G, the response probabilities, and the correct set, and read off each baseline's advantages, the exact expected gradient as a multiple of ∇J, and the exact variance.

**Evidence.** On TL;DR (Pythia-6.9B) and Anthropic-HH (Pythia-6.9B, Llama-7B), GPT-4 win rates against reference completions were RLOO k = 4: 77.9 / 43.7 / 64.1, REINFORCE with a moving-average baseline: 70.7 / 37.9 / 55.3, PPO: 67.6 / 29.2 / 32.0 ([[rloo]] Table 1; Result, single study; checkpoint with the highest test reward). In AppWorld with Qwen2.5-32B-Instruct, a leave-one-out PPO variant scored 71.3 task goal completion without std normalization and 61.9 with it ([[loop-appworld]] Table 1, best run; Result, single study).

**Conditions and limits.** The unbiasedness statements are exact for the gradient of J on one prompt. Clipping, length normalization, and several updates per batch change the estimator further (ch-40). The win rates in [[rloo]] compare full methods, not baselines in isolation.

**Implication for a general model.** Std normalization changes how much each prompt contributes to the update. With a mixed-difficulty or multi-domain prompt set, this changes the effective data mixture without any change to D.

## §3 Actor-critic, bootstrapping, and trust regions

**Definition.** An actor-critic method learns a value function V_φ(s) (the critic) and uses it both as a baseline and to bootstrap returns. Bootstrapping replaces future rewards with a value estimate.

**Problem.** Monte-Carlo returns are unbiased, but for long trajectories with rewards at many steps their variance grows with the number of summed rewards.

**Mechanism and formulas.** [[ppo]] uses truncated generalized advantage estimation (GAE, Eq. 11–12):

```
δ_t = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)
Â_t = δ_t + (γλ) δ_{t+1} + … + (γλ)^{T−t+1} δ_{T−1}
```

- r_t: reward at step t; γ: discount; λ ∈ [0, 1]: GAE parameter; T: segment length.

With λ = 1 the estimate is the finite-horizon return minus V_φ(s_t) ([[ppo]] §5); it is unbiased for any V_φ. With λ = 0 it is the one-step residual δ_t, which is biased whenever V_φ ≠ V^π.

**Worked example.** A 3-token response has a single terminal reward 1 at t = 3, γ = 1, and a critic that has not yet learned to distinguish states and predicts 0.6 at every state. With λ = 1: Â_1 = 1 − 0.6 = 0.4. With λ = 0: Â_1 = 0 + V(s_2) − V(s_1) = 0.6 − 0.6 = 0, so the first token receives no signal although the response succeeded. The bias comes from the critic error, not from sampling.

**Variance and sequence length.** A terminal reward does not make the sequence-level estimator's variance independent of length. The estimator is R · Σ_t s_t with s_t = ∇ log π(y_t | x, y_<t). The score terms have zero conditional mean, so E[‖Σ_t s_t‖²] = Σ_t E[‖s_t‖²], which grows with T. A bounded R does not remove this growth.

**Evidence.** For PPO on Llama-7B with Anthropic-HH, training reward decreased monotonically as λ was lowered from 1.0 through 0.95 and 0.5 to 0.0 ([[rloo]] §3.1, Fig. 1; Result, single study). The authors attribute this to the pre-trained initialization concentrating probability on few tokens, so that bias is not worth the variance reduction (Interpretation). In AppWorld, PPO with a learned critic diverged for λ_GAE ∈ {0.95, 0.99, 0.999} and was most stable at 1.0 ([[loop-appworld]] App. C).

**Trust regions.** Several gradient steps on one batch make the samples off-policy. [[trpo]] bounds the true return by a surrogate: η(π̃) ≥ L_π(π̃) − C · D_KL^max(π, π̃), C = 4εγ/(1 − γ)², ε = max |A_π| (Eq. 9), and in practice maximizes the surrogate subject to an average KL constraint ≤ δ, with δ = 0.01 in all experiments (Eq. 12–13, §8.1). [[ppo]] replaces the constraint with a clipped ratio (Eq. 7). Both control the step between consecutive policies; the KL to a fixed reference policy in RLHF is a different quantity (ch-38).

**Conditions and limits.** Target networks and double or clipped-double Q-learning address instability and overestimation when a value function is bootstrapped from its own estimates. They are relevant when a critic is trained and irrelevant for critic-free methods; they do not concern stochastic environment transitions.

**Implication for a general model.** A critic trained on one prompt distribution is another model that must generalize. Critic-free estimators remove that dependency; they do not remove the dependency on D.

## §4 The baseline decides the sign of each update

**Definition.** A sample is pushed down (its log-probability receives negative weight) when its advantage R − b is negative.

**Problem.** Whether a training method uses negative gradients is often described as a property of the algorithm. For the policy-gradient family it is set by the reward coding and the baseline.

**Mechanism.**
1. With rewards in {0, 1} and b = 0, every advantage is 0 or 1. Incorrect samples receive zero weight: the update is SFT on correct self-generated samples. [[rls-razor]] §5.1 calls this "1–0 Reinforce" and states it "is equivalent to sampling from the model and performing SFT on correct answers only". This is the objective of rejection-sampling fine-tuning when samples are fresh ([[raft-reinforce-rej-minimalist]] RAFT, Eq. 1; ch-31).
2. With rewards in {−1, +1} and b = 0, incorrect samples have advantage −1. [[raft-reinforce-rej-minimalist]] §5 describes this as "fine-tuning on the positive samples and unlearning on the negative samples".
3. With any baseline between the lowest and highest reward in the group, low-reward samples receive negative advantages. In the §2 example, the group mean turns three zero-weight samples into three −0.25 samples.
4. Reward design can create negatives without a baseline: GeneralPoints gives r = −1, −2, or −3 for different failures ([[sft-memorizes-rl-generalizes]] App. A.3).

**Asymmetry between positive and negative weights.** Consider one term A log π(y) and a softmax over logits z:

```
∂ log π(y) / ∂ z_j = 1[j = y] − π(j)
```

- For A > 0 the term is maximized at π(y) = 1, where log π(y) = 0. The gradient on z_y is A(1 − π(y)), which shrinks to 0 as π(y) → 1. The positive update saturates.
- For A < 0 the term A log π(y) → +∞ as π(y) → 0. The gradient on z_y is A(1 − π(y)) → A, which does not shrink. The negative update has no fixed point short of π(y) = 0.

**Evidence.**
- On math with Qwen2.5-Math-7B-base and LLaMA-3.2-3B-instruct (average@16 on MATH500, Minerva Math, OlympiadBench), RAFT++ (positives only) reached 56.1 and 27.6, GRPO 56.3 and 28.4, REINFORCE with ±1 reward 53.9 and 24.2 ([[raft-reinforce-rej-minimalist]] Table 1; Result, single study, no seeds reported). RAFT++ entropy fell faster than GRPO's and GRPO overtook it later in training (§5.1, Fig. 2–3).
- On Science Q&A with Qwen 2.5 3B-Instruct, 1–0 Reinforce matched GRPO on the learning–forgetting trade-off, and SimPO (offline, with negatives) matched SFT ([[rls-razor]] §5.1, Fig. 4; Result, single study, values given as plots).

**Conditions and limits.** In expectation the sign pattern does not change the gradient (§2). It changes the finite-sample update, its interaction with clipping, and the entropy trajectory. The two studies above measure different outcomes (benchmark accuracy and entropy versus forgetting); neither measures the share of improvement due to negatives.

**Implication for a general model.** A baseline that creates negative advantages on every group with mixed rewards applies an unbounded push-down to many samples per step. Controls for this are in "Negative samples and negative feedback".

## §5 The RL gradient as SFT on the model's own samples

**Definition.** The SFT loss on a target distribution π_β is L_SFT = −E_{x∼D, y∼π_β}[log π_θ(y|x)]. The policy-gradient loss is L_RL = −E_{x∼D, y∼π_θ}[A(x, y) log π_θ(y|x)], with gradients taken only through log π_θ ([[rls-razor]] §5.1).

**Problem.** Fine-tuning on a new task can lower performance on earlier tasks (forgetting). The question is whether the training method, at equal new-task accuracy, affects how much is forgotten.

**Mechanism.** The two losses differ in two places: the sampling distribution (π_θ versus fixed external targets) and the weight (A, which can be negative, versus 1). An RL update increases the probability of responses that already have non-negligible probability under the current model. SFT can move probability to responses the model would not sample.

**Formula (idealized case).** For a binary reward and a base policy p over a finite set, the distribution closest to p in KL among fully correct distributions is p restricted to correct responses and renormalized, and its KL from p is −log P_p(correct) ([[rls-razor]] App. A, Lemma A.1). [[rls-razor]] Theorem 5.2 states that, under regularity conditions and a convex policy family, policy gradient from π_0 converges to argmin_{π optimal} KL(π ‖ π_0).

**Worked example.** A base model gives four answers probabilities (0.5, 0.3, 0.15, 0.05); answers 2 and 4 are correct, so P(correct) = 0.35.
- KL-minimal correct policy: (0, 0.3/0.35, 0, 0.05/0.35) = (0, 0.857, 0, 0.143). KL to the base = −log 0.35 = 1.05 nats.
- SFT on annotations that always use answer 4: target (0, 0, 0, 1). KL to the base = −log 0.05 = 3.00 nats.
Both policies are fully correct; the SFT target is 2.85× further in KL.

**Evidence.**
- Qwen 2.5 3B-Instruct trained on math (Open-Reasoner-Zero questions), Chemistry L-3 of SciKnowEval, and ToolAlpaca, with GRPO (binary reward, no KL term) and SFT over hyperparameter sweeps: at matched new-task accuracy, RL kept prior-task scores (HellaSwag, TruthfulQA, MMLU, IFEval, WinoGrande, HumanEval) nearly unchanged while SFT lowered them, most strongly on math ([[rls-razor]] §3.1, Fig. 2; Result, single study, values given as plots). Forgetting was predicted by the KL between the fine-tuned and base policy on the new task, quadratic fit R² = 0.71 in the LLM experiments (§4, Fig. 11).
- In the paper's toy setting (ParityMNIST, a 3-layer MLP), SFT on an analytically constructed KL-minimal fully correct distribution forgot less than RL ([[rls-razor]] §4, Fig. 3; the construction requires knowing the base model's full output distribution and was not run on the LLM tasks).
- Llama-3.2-Vision-11B, SFT-initialized, then PPO or further SFT: on GeneralPoints with an unseen face-card rule, out-of-distribution success went from 11.5% to 15.0% after RL and to 3.4% after SFT; on V-IRL-L with an unseen action space, from 80.8% to 91.8% after RL and to 1.3% after SFT ([[sft-memorizes-rl-generalizes]] §5.1; Result, single study). End-to-end RL without the SFT initialization failed to improve because the base model's outputs could not be parsed for reward (§5.4).
- Replicated direction: both studies find RL retains or transfers better than SFT on external targets. They differ in model, task, and metric (forgetting of prior benchmarks versus rule-variant OOD accuracy).

**Conditions and limits.** Both studies use one base model each. The SFT targets were generated by a different model (DeepSeek-R1 or GPT-4o in [[rls-razor]] App. B.1) or task demonstrations; SFT on self-generated targets is a different case (ch-31). [[rls-razor]] §7: "we still lack a mechanistic account of why larger KL shifts on the new task disrupt prior knowledge". ch-30a records a report in which the KL–forgetting relation does not always hold (Open question).

**Implication for a general model.** On-policy sampling is a forgetting control in its own right. A new capability that the model cannot sample at all cannot be reached by on-policy RL; that case needs SFT or distillation first (ch-31, ch-38a).

## §6 Reverse-KL-regularized reward, entropy, and diversity

**Definition.** RLHF maximizes reward minus a KL penalty to a reference policy π_ref (usually the SFT model):

```
max_π E_{x∼D} [ E_{y∼π}[R(x, y)] − β KL(π(·|x) ‖ π_ref(·|x)) ]
π*(y|x) = π_ref(y|x) exp(R(x, y)/β) / Z(x)
```

- β > 0: KL coefficient; Z(x) = Σ_y π_ref(y|x) exp(R(x, y)/β). The second line is the maximizer, obtained by writing the objective as −β KL(π ‖ π*) + β log Z. [[maximum-entropy-rl]] Eq. 4 has the same exponential target with Q in place of R and no reference policy (Interpretation of the relation).

**Why "reverse" and mode-seeking.** KL(π ‖ π_ref) = Σ_y π(y) log(π(y)/π_ref(y)) weights the log-ratio by π. Placing mass where π_ref is small costs a large amount. Removing mass from responses that π_ref supports costs nothing for those responses, because their terms are multiplied by π(y) = 0. The penalty therefore allows the policy to drop responses.

**Worked example.** π_ref = (0.5, 0.3, 0.2), rewards (1, 0, 0). With β = 2: π* = (0.623, 0.227, 0.151), entropy 0.917 nats (π_ref: 1.030), KL 0.030. With β = 0.5: π* = (0.881, 0.072, 0.048), entropy 0.446, KL 0.328. Among the two zero-reward responses the ratio stays 1.5 in both cases: the exact optimum preserves π_ref's shape among equal-reward responses and concentrates mass on higher-reward ones.

**KL does not give a lower bound on entropy.** KL(π ‖ π_ref) = −H(π) + CE(π, π_ref), with CE(π, π_ref) = −Σ_y π(y) log π_ref(y). KL ≥ 0 gives H(π) ≤ CE(π, π_ref), an upper bound. Example: π_ref = (0.9, 0.1), π = (1, 0). KL = log(1/0.9) = 0.105 nats and H(π) = 0. A peaked π_ref allows collapse at low KL cost.

**Evidence on diversity.**
- PPO with β_KL = 0.05 on LLaMA 7B: RLHF generalized better out of distribution than SFT and "significantly reduces output diversity compared to SFT across a variety of measures" ([[rlhf-generalisation-diversity]] Abstract). On OPT-6.7B summarisation, per-input EAD was 0.07 for RLHF and 0.79 for SFT (App. J.4 Table 11); across-input EAD was 0.87 for both (Table 12).
- Increasing the KL coefficient lowered performance and also lowered per-input diversity ([[rlhf-generalisation-diversity]] §6.3, App. I; Result, single study). A larger β did not recover diversity in that setting.
- In RLVR runs on math, adding an entropy loss L − αH(π_θ) with α = 0.0001 or 0.001 had minor effect on entropy, α = 0.01 caused entropy explosion, and α = 0.005 stabilized entropy without outperforming the other baselines; reference-KL coefficients 0.001–0.1 stabilized entropy but lowered accuracy ([[entropy-mechanism-llm-rl]] §4.1, Fig. 9–10; the model used for these two sweeps is not stated). For vanilla policy gradient on a softmax, the entropy change is ≈ −η Cov(log π(a), π(a) A(a)) (Theorem 1): raising an already probable action with positive advantage lowers entropy.

**Entropy bonus in classical RL.** [[ppo]] Eq. 9 includes an entropy term c_2 S[π_θ](s_t). The MuJoCo experiments used no entropy bonus (§6.1); c_2 = 0.01 is the Atari value (Table 5). [[maximum-entropy-rl]] builds entropy into the objective and reports that a small reward scale gave a near-uniform policy and a large scale a near-deterministic policy stuck in poor local minima (§5.2, Fig. 3b).

**KL in practice is a single-sample estimate.** Implementations do not compute the full-vocabulary KL. PPO-style RLHF adds a per-token log-ratio log(π/π_ref) of the sampled token to the reward ([[reinforce-plus-plus]] Eq. 4); GRPO adds π_ref/π − log(π_ref/π) − 1 of the sampled token to the loss ([[grpo]] Eq. 4). These are the k1 and k3 estimators of [[john-schulman-kl-tricks]]. ch-38 and ch-43 compare them.

**pass@k: the metric that detects lost coverage.** Definition: a problem is solved at k if at least one of k samples is correct. With n ≥ k samples and c correct, the unbiased estimator ([[rlvr-beyond-base-model]] App. A.2) is:

```
pass@k = E_x [ 1 − C(n − c, k) / C(n, k) ]
```

Worked example: n = 10, c = 2, k = 5 gives 1 − C(8, 5)/C(10, 5) = 1 − 56/252 = 0.778.

Evidence: Qwen2.5-7B trained with GRPO on 2,000 Omni-MATH-Rule problems raised MATH500 pass@1 from 34.5 to 74.4 while pass@256 went from 96.2 to 97.2; on the in-domain Omni-MATH test, pass@256 went from 69.1 to 68.3 ([[rlvr-beyond-base-model]] Table 3). Longer GRPO training raised Omni-MATH-Train pass@1 from 26.1 (step 150) to 42.5 (step 450) while pass@256 fell from 66.3 to 64.3 (Table 4). On AIME24 at k = 1024, 13.3% of problems were solved by the base model and not by the SimpleRLZoo RL model, and 0.0% the other way (Table 2). The authors interpret RLVR as raising the probability of paths the base model could already sample (Interpretation). [[reinforce-plus-plus]] v9 Table 2 reports that GRPO trained on 30 AIME-24 questions reached 95.0 train pass@1 and 0.4 AIME-25 pass@16 (model not stated; Result, single study).

**Implication for a general model.** Report entropy, per-input diversity, and pass@k at large k alongside reward. A rising pass@1 with a flat or falling pass@k indicates concentration rather than new capability.

## §7 When the environment returns tokens: the agentic exception

**Definition.** In agentic RL the model emits an action (text, a tool call), the environment returns an observation (tool output, verifier message), and the observation is appended to the context before the next action.

**Problem.** The single-turn setting has deterministic transitions: the next state is the prefix plus the sampled token. With tools, the next state depends on the environment, which can be stochastic or stateful. Observation tokens are in the sequence but are not sampled from π_θ.

**Mechanism.**
1. The trajectory probability is p(τ) = Π_t π_θ(a_t | s_t) · P(s_{t+1} | s_t, a_t). The transition factor P has no θ, so its log-derivative is 0 and the §1 estimator remains unbiased.
2. Only action tokens belong in Σ log π_θ. [[loop-appworld]] formulates AppWorld as a POMDP in which environment tokens are appended to the state but only LLM-emitted tokens enter the likelihood (§4.1, Eq. 6–8). TRL sets `tool_mask` to 0 for tool-result tokens and excludes them from the loss and the importance ratio ([[trl-grpo]] L1472, L1638–1648, L2425).
3. Including observation tokens in the loss would add an SFT term on environment text with weight A, which is not part of ∇J.
4. Environment randomness adds reward variance that a prompt-level baseline cannot remove, and a terminal reward is shared by all turns (credit assignment).

**Evidence.** In AppWorld (Qwen2.5-32B-Instruct, LoRA, 72 training tasks), importance weights per token scored 71.3 task goal completion, per turn 64.1, per trajectory 53.3 ([[loop-appworld]] Table 1, best run; three-run means in Table 2 keep the order). In GeneralPoints, the verifier's text is appended to the prompt (sequential revision); more verification steps under the same compute gave larger OOD gains: +0.48% (1 step), +2.15% (3), +2.99% (5), +5.99% (10) ([[sft-memorizes-rl-generalizes]] §3, §5.5).

**Conditions and limits.** AppWorld lacks non-determinism, transient failures, and unsolvable tasks ([[loop-appworld]] §6). Critics, turn-level values, and observation handling are treated in ch-45b.

**Implication for a general model.** Tool outputs vary across environments. A policy trained on one tool environment is evaluated on others (Test-Challenge in AppWorld includes unseen apps: 21.0 → 45.7 after LOOP, [[loop-appworld]] Table 1).

## Negative samples and negative feedback

This section uses the four meanings of "negative" from the course standard. Policy gradient uses meaning (4), **negative as gradient**: a negative advantage lowers the sample's log-probability. Rejection sampling with b = 0 uses the **discard** option.

**Where negatives come from.** A verifier (Math-Verify in [[raft-reinforce-rej-minimalist]] and [[dr-grpo]]), a reward model score below the baseline ([[rloo]]), unit tests passed as a fraction ([[loop-appworld]] App. D), or a shaped environment penalty ([[sft-memorizes-rl-generalizes]] App. A.3). False-negative rates are not reported in these sources. [[rlvr-beyond-base-model]] §2.2 notes the reverse error at large k in math: a wrong chain of thought can reach the correct answer.

**What current practice does.** RAFT/RAFT++ and 1–0 Reinforce discard failures. REINFORCE with ±1 rewards, RLOO, GRPO, and PPO push failures down. Reinforce-Rej discards prompts whose responses are all correct or all incorrect and pushes down failures in mixed groups ([[raft-reinforce-rej-minimalist]] §5).

**Mechanism.** For one sampled token y with advantage A < 0 and step η, the logit update is Δz_k = η A (1[k = y] − p_k). The first-order probability change is

```
Δp_j ≈ η A p_j (1[j = y] − p_j − p_y + Σ_k p_k²)
```

For an unsampled j this is positive only when p_j > Σ_k p_k² − p_y. Worked example: p = (0.6, 0.3, 0.1), sampled y = token 3, A = −1, η = 1. The threshold is 0.46 − 0.1 = 0.36. Exact softmax after the update: (0.710, 0.263, 0.026). Token 3 lost 0.074, token 1 gained 0.110, and token 2 lost 0.037. Mass removed from an unlikely sample moves to the most likely alternative, and other alternatives can also lose mass. Panel C of [figures/pg-variance.html](figures/pg-variance.html) lets the reader repeat this for other probabilities. ch-39 and ch-43a develop this effect (squeezing, likelihood displacement).

**Evidence of benefit and of failure modes.**
- Benefit: GRPO overtook RAFT++ late in training, and RAFT++ entropy fell faster; the authors conclude negatives help maintain exploration (Interpretation; [[raft-reinforce-rej-minimalist]] §5.1).
- Cost without filtering: REINFORCE with ±1 reward scored 53.9 and 24.2 versus 56.4 and 28.5 for Reinforce-Rej on the two models (Table 1). In the ablation on LLaMA-3.2-3B-instruct, removing prompts whose responses were all wrong gave the largest reward gain over vanilla REINFORCE, while removing all-correct prompts changed little (Fig. 4 and its caption).
- Forgetting: on-policy data, not negatives, separated RL from SFT in [[rls-razor]] §5.1.
- Share of improvement attributable to negatives: not measured by any source cited here.

**Controls.**
1. Keep samples on-policy or bound the ratio: PPO clips the ratio at 1 − ε when Â < 0, so lowering a probability below (1 − ε)π_old gives no further gain ([[ppo]] §3, Fig. 1).
2. Use a baseline that yields zero advantage when all rewards are equal (group mean or leave-one-out), so a group in which every sample failed produces no push-down. In TRL the mean subtraction already gives 0 for such a group, and the division adds 1 × 10⁻⁴ to the group std so it stays finite ([[trl-grpo]] L2147–2149); the zero-std flag itself is logged, not used to zero the advantage (L2150, L2186).
3. Anchor with a KL term or a positive NLL term (ch-38, ch-43a).
4. Filter or down-weight uninformative groups (Reinforce-Rej).

**Diagnostics.** Log mean advantage and mean log-probability separately for A > 0 and A < 0 samples; log policy entropy; log the fraction of zero-std groups (`frac_reward_zero_std` in [[trl-grpo]]); report pass@1 and pass@k at large k on held-out prompts.

**Effect on generality.** Negatives may preserve entropy ([[raft-reinforce-rej-minimalist]]), which bears on coverage; unbounded push-down concentrates mass on the most likely alternatives, which bears on diversity. The effect on calibration, hallucination, and over-refusal is not measured in these sources.

## Recipe

Rows quote the policy-gradient settings that the cited sources state. Loci were checked on the dates shown.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| PPO-Clip, MuJoCo 1M-timestep benchmark | MLP 2×64 | RL | entropy coefficient c_2 | not used (no entropy bonus) | arXiv:1707.06347v2 §6.1 ([[ppo]]) | verified 2026-09-14 | not applicable |
| PPO-Clip, MuJoCo | MLP 2×64 | RL | clip ε | 0.2 | arXiv:1707.06347v2 §6.1 Table 1 | verified 2026-09-14 | Table 1: 0.82 vs 0.76 (ε = 0.1), 0.70 (ε = 0.3); 7 tasks × 3 seeds |
| PPO-Clip, Atari (49 games) | not reported | RL | c_1; c_2 | 1; 0.01 | arXiv:1707.06347v2 App. A Table 5 | verified 2026-09-14 | no ablation reported |
| TRPO experiments | not applicable | RL | KL step size δ | 0.01 ("for all experiments") | arXiv:1502.05477 §8.1 ([[trpo]]) | verified 2026-09-15 | no ablation reported |
| RLOO / PPO / RAFT runs, TL;DR | Pythia-6.9B | RL | β (KL in reward); rollout batch; step batch; steps | 0.03; 512; 256; 600 | arXiv:2402.14740v2 App. C ([[rloo]]) | verified 2026-09-15 | no ablation reported |
| same, Anthropic-HH | Pythia-6.9B | RL | β; steps | 0.10; 393 | arXiv:2402.14740v2 App. C | verified 2026-09-15 | no ablation reported |
| same, all datasets | Pythia-6.9B, Llama-7B | RL | LR; warm-up; gradient steps per batch; k | constant 1 × 10⁻⁶; 3% linear; 2; 2 or 4 | App. C; Table 1 | verified 2026-09-15 | LR sweep {1e-6, 1e-5, 2e-5} (RAFT, RLOO) and {1e-6, 1e-5} (PPO, Vanilla PG); Table 1 k = 4 vs k = 2 |
| GRPO for forgetting study | Qwen 2.5 3B-Instruct | RL | KL coefficient; group size; prompts per generation; μ; loss type | 0; 64; 8; {1, 2}; Dr. GRPO | arXiv:2509.04259v1 App. B Table 2 ([[rls-razor]]) | verified 2026-09-15 | Pareto frontier over sweep (Fig. 2) |
| same | Qwen 2.5 3B-Instruct | RL | LR sweep; schedule; warm-up; epochs | {1e-5 … 5e-5}; constant with warm-up; 50 steps; 1 | App. B Table 2 | verified 2026-09-15 | Pareto frontier over sweep |
| RLHF (PPO) for diversity study | LLaMA 7B | RL | β_KL (KL in reward) | 0.05 | arXiv:2310.06452v3 §4 Eq. 1 ([[rlhf-generalisation-diversity]]) | verified 2026-09-15 | App. I sweep: higher β lowered per-input diversity and performance |
| RAFT, RAFT++, GRPO, Reinforce-Rej | Qwen2.5-Math-7B-base, LLaMA-3.2-3B-instruct | RL | prompts per iteration; responses per prompt; mini-batch; LR; max tokens | 1,024; 4; 512; 1 × 10⁻⁶; 4,096 | arXiv:2504.11343v2 §4 ([[raft-reinforce-rej-minimalist]]) | verified 2026-09-15 | Table 1 caption: batch, mini-batch and actor LR were tuned per algorithm and refers to an appendix that v2 does not contain, so the per-algorithm values are not available |
| GRPO / PPO / RLOO comparison | Qwen2.5-7B (base) | RL | KL; LR; prompts per step; responses per prompt; max length; temperature | removed; constant 1 × 10⁻⁶; 256; 8; 8,192; 1.0 | arXiv:2504.13837v5 §4.3 ([[rlvr-beyond-base-model]]) | verified 2026-09-15 | no ablation reported |
| pass@k evaluation, same study | Qwen2.5 and LLaMA-3.1-8B | eval-gate | n samples; temperature; top-p; max tokens | 128 or 1,024 (largest k plotted); 0.6; 0.95; 16,384 | arXiv:2504.13837v5 §3, App. A.2 | verified 2026-09-15 | not applicable |
| Dr. GRPO runs (Oat-Zero) | 1.5B–7B | RL | responses per question; temperature; KL coefficients; clip ε; LR | 8; 1.0; 0.0; 0.2; 1 × 10⁻⁶ constant | arXiv:2503.20783v2 App. G Table 6 ([[dr-grpo]]) | verified 2026-09-14 | no ablation reported |
| Entropy-loss sweep (model not stated) | not reported (checked §4.1, Fig. 9 caption, §2.2) | RL | entropy-loss coefficient α in L − αH | 0.0001, 0.001 minor effect; 0.005 stable, not better; 0.01 entropy explosion | arXiv:2505.22617v1 §4.1 Fig. 9 ([[entropy-mechanism-llm-rl]]) | verified 2026-09-14 | Fig. 9 |

**Starting point for a small general-purpose run.** The following values each come from a verified row above, under the stated conditions. For a 3B–7B model with verifiable rewards: constant LR 1 × 10⁻⁶ (Qwen2.5-7B and Qwen2.5-Math-7B runs in [[rlvr-beyond-base-model]] and [[raft-reinforce-rej-minimalist]]), 4–8 responses per prompt (same rows), temperature 1.0 for rollouts ([[rlvr-beyond-base-model]], [[dr-grpo]]), no entropy bonus (PPO MuJoCo; the entropy-loss sweep found no gain), and a leave-one-out or mean-only baseline (§2). The KL coefficient depends on the goal: 0 was used in the reasoning and forgetting studies above; 0.03–0.10 in the RLHF reward-model runs of [[rloo]]. Evaluate pass@1 and pass@k with n ≥ 128 on held-out prompts. None of these runs tested these values for a multi-domain general model.

## Generalization lens

**(a) What increases breadth.**
- On-policy updates: at matched new-task accuracy, RL forgot less than SFT on external targets ([[rls-razor]] Fig. 2) and transferred to unseen rules and visual variants where SFT did not ([[sft-memorizes-rl-generalizes]] §5.1–5.2). Result, replicated in direction across two studies.
- More environment feedback per episode: more verification steps gave larger OOD gains at equal compute ([[sft-memorizes-rl-generalizes]] §5.5).
- Out-of-distribution robustness from RLHF: better OOD preference scores than SFT, particularly as the distribution shift grows ([[rlhf-generalisation-diversity]] Abstract, §6.1).

**(b) What causes narrowing or forgetting.**
- Reduced per-input diversity after RLHF ([[rlhf-generalisation-diversity]] Table 11), not recovered by a larger KL coefficient (§6.3).
- Reduced large-k coverage after RLVR and with longer training ([[rlvr-beyond-base-model]] Tables 3–4).
- Overfitting to a small prompt set: GRPO on 30 AIME-24 questions reached 95.0 train pass@1 and 0.4 AIME-25 pass@16 ([[reinforce-plus-plus]] v9 Table 2).
- Implicit reweighting of prompts by std normalization (§2), which changes the effective mixture.
- Prompt types absent from D receive no gradient (§1).

**(c) How to measure it for this stage.** Use three evaluation splits for every RL run in this track:
1. **Training reward** on the prompts being optimized (detects optimization failure, not generality).
2. **Held-out prompts from the same distribution**, with pass@1 and pass@k at n ≥ 128 (detects overfitting and coverage loss).
3. **Prompts from other domains and prior-capability benchmarks** (for example the six benchmarks in [[rls-razor]] §3.1), plus per-input diversity (detects forgetting and narrowing).
Known measurement errors: pass@k at large k in math counts lucky guesses ([[rlvr-beyond-base-model]] §2.2); diversity metrics did not separate models on long instruction-following outputs ([[rlhf-generalisation-diversity]] §6.2); win rates from an LLM judge were not checked against human preferences in [[rloo]] (Limitations).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating group-mean or group-std advantages as unbiased | Effective learning rate changes with group size; easy and hard prompts dominate updates | Compute the expected scale (G−1)/G and the per-prompt std weight; compare with leave-one-out (figure panel B) |
| Assuming a KL penalty prevents entropy collapse | Entropy falls while KL to the reference stays small | Log H(π) and CE(π, π_ref) separately; check π_ref's entropy on training prompts |
| Adding an entropy bonus by default | Entropy rises without accuracy gain, or rises without bound | Sweep the coefficient on a held-out split; compare with c_2 = 0 ([[entropy-mechanism-llm-rl]] Fig. 9) |
| Including tool or observation tokens in the policy loss | Model reproduces tool output text; loss decreases on tokens the model did not generate | Verify the loss mask is 0 on observation tokens ([[trl-grpo]] `tool_mask`) |
| Reporting only training reward or pass@1 | Gains on training prompts with flat or falling pass@k on held-out prompts | Report the three splits in the Generalization lens |
| Using ±1 rewards with no filtering or anchor | Log-probability of rejected samples falls fast; probability of unrelated tokens drops | Log log π separately for A > 0 and A < 0; track entropy |
| Stating "the estimator's variance is bounded because the reward is terminal" | Gradient norm variance grows with response length | Plot gradient-norm variance against response length bins |
| Bootstrapping with an untrained critic (λ < 1) | Reward stalls or diverges early | Compare λ = 1 against lower λ, as in [[rloo]] Fig. 1 and [[loop-appworld]] App. C |

## Check your understanding

1. In the §1 two-action example, explain why b = 0.8 gives zero variance and why this would not hold with three possible answers.
2. The group-mean baseline multiplies the expected gradient by (G − 1)/G. Explain why this is harmless on one prompt but std normalization is not harmless across prompts of different difficulty.
3. With 0/1 rewards and b = 0, no sample is pushed down. Explain why the expected gradient is still the same as with a leave-one-out baseline, and what does change in training.
4. Explain why a negative advantage on an already unlikely token can lower the probability of a moderately likely token, using the first-order Δp formula.
5. Explain the causal chain from "samples come from π_θ" to "less forgetting than SFT", and name the evidence that on-policy data, not negative gradients, is the factor in [[rls-razor]].
6. KL(π ‖ π_ref) is small in a run whose entropy collapsed. Explain how this is possible.
7. A run shows pass@1 rising from 30 to 45 while pass@256 falls from 70 to 66. State what the model gained and what it lost, and which evaluation split should come next.
8. In a tool-use trajectory, explain why the transition probabilities do not bias the policy gradient but observation tokens must still be masked.

## Connections

- **Previous: ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization.** Provides R(x, y); reward-model errors enter the estimator as advantage errors.
- **Next: ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax.** Adds the clipped surrogate, the per-token KL in the reward, and the alignment-tax measurements.
- **ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** Full treatment of [[rls-razor]], [[sft-memorizes-rl-generalizes]], and [[rlhf-generalisation-diversity]].
- **ch-39 — Offline Preference Optimization: DPO and Its Variants.** Uses the KL-regularized optimum of §6 without sampling.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.** Implements the baselines of §2 with clipping and length normalization.
- **ch-16 — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse.** Designs D from §1.
- **ch-43 — Entropy, Output Diversity, and KL Control in RL.** Continues §6.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.** Continues the negative-feedback section.
- **ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.** Continues §7.
- **ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control** and **ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation.** Background for §4–§5.

## Sources

- [[vanilla-pg]] — REINFORCE update with a reinforcement baseline (§1–§2).
- [[rloo]] — sequence-as-action form, leave-one-out estimator, λ sweep, win rates, training settings (§1–§3, Recipe).
- [[trpo]] — improvement bound, KL-constrained surrogate, δ = 0.01 (§3, Recipe).
- [[ppo]] — GAE, clipped surrogate, entropy coefficient by domain (§3, §6, negative-feedback controls, Recipe).
- [[maximum-entropy-rl]] — maximum-entropy objective and reward-scale sensitivity (§6).
- [[rls-razor]] — on-policy updates and forgetting, 1–0 Reinforce vs GRPO vs SimPO, KL-minimal theory (§4–§5, Recipe).
- [[sft-memorizes-rl-generalizes]] — rule and visual OOD after RL vs SFT, sequential revision, verification steps, reward design (§4, §5, §7).
- [[rlhf-generalisation-diversity]] — RLHF generalization and diversity, KL sweep (§6, Generalization lens).
- [[rlvr-beyond-base-model]] — pass@k estimator, pass@1 vs pass@256 tables (§6, Recipe).
- [[raft-reinforce-rej-minimalist]] — positives-only vs ±1 REINFORCE vs GRPO, entropy, prompt filtering (§4, negative-feedback section, Recipe).
- [[reinforce-plus-plus]] — bias of the GRPO advantage, KL log-ratio in the reward, small-set overfitting (§2, §6).
- [[dr-grpo]] — (G−1)/G relation to RLOO, difficulty bias, run settings (§2, Recipe).
- [[grpo]] — GRPO advantage and k3 KL term in the loss (§6).
- [[entropy-mechanism-llm-rl]] — entropy change under policy gradient, entropy-loss and KL sweeps (§6, Recipe).
- [[loop-appworld]] — POMDP with observation tokens, importance-weight granularity, std normalization cost (§2, §3, §7).
- [[trl-grpo]] — group std computation, zero-std handling, `tool_mask` (§2, §7, diagnostics).
- [[john-schulman-kl-tricks]] — names of the k1 and k3 KL estimators (§6).
