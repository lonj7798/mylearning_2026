<!-- chapter: ch-38
     track: rl
     kind: content
     title: KL-Controlled RLHF — TRPO, PPO, InstructGPT
     deps: [ch-37]
     sources: [[trpo]], [[ppo]], [[rlhf-instructgpt]], [[kl-control-rlhf]], [[llama-2]], [[costa-huang-ppo-details]], [[hf-rlhf-illustrated]], [[lilianweng-rlhf]], [[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]]
     figures: figures/ppo-clip.html
-->

# Chapter 38 — KL-Controlled RLHF: TRPO, PPO, InstructGPT

> **Core insight.** Modern RLHF is not "run RL on a reward model." It is *KL-regularized* amortized inference from `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)` (the Korbak view, [[kl-control-rlhf]]). The PPO clipped surrogate is the first-order approximation to TRPO's trust region ([[trpo]]), and InstructGPT's Equation 2 ([[rlhf-instructgpt]]) bolts a per-token KL penalty and a pretraining-mix coefficient onto that surrogate. Everything else — Llama-2's dual RM, Costa-Huang's 37 tricks, TRL/OpenRLHF/verl's loss kernels — is plumbing around those two facts.
>
> **Guideline.** Use PPO-clip with ε=0.2, GAE λ≈0.95, γ=1.0 (rewards concentrate at EOS), add `−β·KL(π‖π_ref)` to the per-token *reward* not the loss (k3 estimator), initialize the value head from the RM, whiten advantages per-minibatch, clip value predictions. Start β≈0.02 (InstructGPT) or β≈0.01 (Llama-2). If reward hacks, raise β before touching ε or LR.

---

## §1 TRPO — the ancestor you do not implement

[[trpo]] proves a *monotonic-improvement* bound:

```
η(π_new) ≥ L_{π_old}(π_new)  −  C · D_KL^{max}(π_old, π_new)
```

where `L_{π_old}(π) = η(π_old) + E_{s∼ρ_{π_old}, a∼π}[ (π(a|s)/π_old(a|s)) · A_{π_old}(s,a) ]` is the linearization of the true return around `π_old`, and `C` collapses a worst-case advantage bound. The two terms are the "expected improvement" and the "distribution-shift penalty." TRPO converts the penalty into a constraint:

```
maximize_θ  L_{π_old}(π_θ)    subject to    E_s[ D_KL(π_old(·|s) ‖ π_θ(·|s)) ] ≤ δ
```

with `δ ≈ 0.01`. The Lagrangian is solved with a **natural-gradient step** computed via conjugate-gradient on Fisher-vector products:

```
θ_new = θ_old + sqrt( 2δ / (g^T F^{-1} g) ) · F^{-1} g,      g = ∇L_{π_old}
```

followed by a geometric line-search to enforce both the KL constraint and actual surrogate improvement. This gives provable monotonic improvement.

**Why nobody uses TRPO for LLM-RL.** Two Fisher-vector products per CG iteration × 10 CG iterations = ~20 extra backward passes per update. At 70B scale, prohibitive. The Fisher matrix over 10^10 parameters is ill-conditioned and the natural-gradient direction depends on rare tail tokens whose ratio explodes. PPO keeps the surrogate `L_{π_old}(π_θ)` but swaps the KL constraint for a **clipping trick** that enforces the trust region *per-token* without any second-order structure.

---

## §2 PPO — deriving the clipped surrogate term by term

Define the probability ratio

```
r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t).
```

At the start of each epoch `θ = θ_old`, so `r_t = 1` everywhere. The unclipped surrogate is

```
L^{CPI}(θ) = E_t[ r_t(θ) · Â_t ]
```

("CPI" = conservative policy iteration, [[ppo]]). Maximizing `L^{CPI}` directly is what TRPO bounds with a KL constraint; PPO replaces the constraint with a *pessimistic min* over two terms:

```
L^{CLIP}(θ) = E_t[ min( r_t(θ) · Â_t ,   clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]
```

**Term-by-term.** The `clip(r, 1−ε, 1+ε)` term saturates the ratio outside the trust region `[1−ε, 1+ε]`; its gradient w.r.t. θ is zero outside that window. The `min` combines clipped and unclipped pessimistically — take the lower of the two — so the objective is *lower-bound* on the true improvement. Casework:

- `Â_t > 0` and `r_t < 1+ε`: `r_t · Â_t` is the smaller term; gradient pushes `r_t` up. Good, we raise likelihood of a good action.
- `Â_t > 0` and `r_t ≥ 1+ε`: `clip(r_t) · Â_t = (1+ε)·Â_t` is the smaller (equal) term; its gradient is zero. We refuse to push further, staying in the trust region.
- `Â_t < 0` and `r_t > 1−ε`: `r_t · Â_t` is the smaller (more negative) term; gradient pushes `r_t` down. Good, we lower likelihood of a bad action.
- `Â_t < 0` and `r_t ≤ 1−ε`: `clip(r_t)·Â_t = (1−ε)·Â_t` is the smaller term; its gradient is zero. We refuse to push further.

The *pessimism asymmetry* is deliberate: we only refuse to update when the unclipped term *would* improve the surrogate further than the clip allows. When the ratio has already exploded *against* us, the unclipped term still contributes, and we keep its gradient. This is what the dual-clip trick in [[verl-ppo-loss]] (Ye 2020) later patches — PPO-clip does *not* floor the loss for negative-advantage tokens with huge ratios, and on long LLM rollouts that can blow up.

The combined objective adds a value loss and entropy bonus:

```
L^{CLIP+VF+S}(θ) = E_t[ L^{CLIP}(θ)  −  c_1 · L^{VF}(θ)  +  c_2 · S[π_θ](s_t) ]
```

with `c_1 = 1.0`, `c_2 = 0.01` in the original paper ([[ppo]]). `L^{VF} = (V_θ(s_t) − V_target)^2` is MSE on returns; `S = −Σ_a π log π` is the entropy bonus. In RLHF the entropy term is usually *dropped* ([[lilianweng-rlhf]], [[rlhf-instructgpt]]) because the `−β·KL(π‖π_ref)` added into the reward already regularizes the policy toward the SFT distribution.

---

## §3 GAE — the advantage that makes PPO work

The advantage `Â_t` in `L^{CLIP}` is *generalized advantage estimation* (Schulman 2016). Define the TD residual

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t).
```

GAE is the exponentially weighted sum of future TD residuals:

```
Â_t^{GAE(γ, λ)} = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}
               = δ_t + (γλ)·δ_{t+1} + (γλ)^2·δ_{t+2} + …
```

The `λ` knob interpolates bias–variance: `λ=0` gives `Â_t = δ_t` (pure TD, low variance, biased by value-function error); `λ=1` gives `Â_t = Σ_k γ^k r_{t+k} − V(s_t)` (pure Monte-Carlo return minus baseline, unbiased but high variance). In LLM-RL ([[lilianweng-rlhf]]): `γ = 1.0` (rewards concentrate at EOS; discounting would shrink the signal on long completions) and `λ = 0.95` (Schulman default, keeps variance tractable).

**Value-head ancestry.** `V(s_t)` is a scalar head on the policy's trunk, trained with `L^{VF} = (V_θ(s_t) − R_t)^2` where `R_t = Â_t + V_{old}(s_t)` is the GAE return target. The value head is what turns PPO into an actor-critic: without it, advantages reduce to Monte-Carlo returns minus a constant baseline (REINFORCE), and variance kills training on long sequences. [[trl-ppo]] and [[verl-ppo-loss]] both implement a value-loss clip around `V_old` to mirror the policy-ratio clip — a Costa-Huang trick ([[costa-huang-ppo-details]], item 4).

See **[figures/ppo-clip.html](figures/ppo-clip.html)** — interactive ratio scrubber showing `L^{CLIP}` and its gradient across `r ∈ [0, 2]` for both signs of `Â`, plus a β-vs-reward KL-budget sweep.

---

## §4 InstructGPT — PPO-ptx in one equation

[[rlhf-instructgpt]] Equation 2, verbatim:

```
objective(φ) = E_{(x,y)~D_RL}[ r_φ(x,y)  −  β · log( π_φ^{RL}(y|x) / π^{SFT}(y|x) ) ]
             + γ · E_{x~D_pretrain}[ log π_φ^{RL}(x) ]
```

Three terms, three roles:

1. **`r_φ(x,y)`** — the Bradley-Terry RM score. Scalar, applied at end-of-sequence.
2. **`−β · log(π^{RL}/π^{SFT})`** — the per-token KL penalty. This is the *KL-control* term from [[kl-control-rlhf]]. Implementation-wise, it is **added to the per-token reward**, not to the loss: `r̂_t = −β·(log π^{RL}(y_t|…) − log π^{SFT}(y_t|…))` for `t < |y|`, and `r̂_{|y|} = r̂_{|y|} + r_φ(x,y)` at EOS. Adding KL to the reward keeps the GAE advantage well-defined per token ([[trl-ppo]] §What to notice, [[verl-ppo-loss]] §Context); adding it to the loss breaks advantage-based policy gradient and empirically trains worse.
3. **`γ · E_{D_pretrain}[log π_φ^{RL}(x)]`** — the "ptx" mix. A pretraining cross-entropy term mixed in every update, preventing the alignment tax. When `γ = 0` you get plain PPO against the KL-penalized reward ("InstructGPT"); when `γ > 0` you get PPO-ptx.

**Canonical hyperparameters** ([[rlhf-instructgpt]]):

| Knob | InstructGPT value |
|------|-------------------|
| PPO LR | 1.41e-5 (fixed) |
| PPO batch size | 512 prompts |
| PPO rollout length | ≤ 2048 tokens |
| KL coef β | 0.02 (adaptive controller optional) |
| Pretraining coef γ | 27.8 (PPO-ptx) or 0 (InstructGPT) |
| Clip ε | 0.2 |
| Epochs per rollout K | 4 |

No explicit entropy bonus. Entropy collapse is a *failure signal* tracked by the adaptive-KL controller ([[kl-control-rlhf]]): when observed KL drifts above the target, β is multiplicatively raised; below target, lowered. Korbak's Bayesian-inference view ([[kl-control-rlhf]]) explains why β has a natural "temperature" interpretation: at fixed β, the objective's closed-form optimum is `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)` — exact tilted-posterior sampling.

---

## §5 Llama-2 PPO — dual RM and conservative LR

[[llama-2]] runs five RLHF iterations (V1–V5). V1–V3 are Rejection-Sampling Fine-Tuning only; PPO is added in V4/V5 on top of the RSFT checkpoint. The appendix-quoted hyperparameters (70B policy):

| Knob | Llama-2 value |
|------|---------------|
| PPO LR | 1e-6 (policy) |
| KL coef β | 0.01 |
| Batch size | 512 |
| Sequence length | 4K |
| Value function | standard PPO with clipped ratio + GAE |

**Two reward models.** Instead of the single Bradley-Terry RM from InstructGPT, Llama-2 trains a **Helpfulness RM** and a **Safety RM** separately. At PPO scoring time a rule picks which RM (or a weighted combination) scores each prompt: safety-relevant prompts get the Safety RM (or a max-safety piecewise), helpfulness-relevant prompts get the Helpfulness RM. This resolves the helpfulness-vs-safety tradeoff that a single RM forces into its scalar output.

**Why the conservative LR.** 1e-6 is ~14× smaller than InstructGPT's 1.41e-5. At 70B scale with iterative RLHF (weekly fresh preference batches), a larger LR pushes the policy too far per iteration, and the dual-RM rule destabilizes — helpfulness gains can get clawed back by the safety RM within a single minibatch. Llama-2's recipe is: conservative LR, low β, many iterations. The β=0.01 (half of InstructGPT's 0.02) is load-bearing for the iterative schedule: you want the policy free to move between iterations, compensated by more iterations.

**Margin-weighted RM loss.** Annotators label *margin* ("significantly better / better / slightly better / negligibly better"), and the RM loss upweights large-margin pairs:

```
L_RM = −E[ log σ( r(x, y_w) − r(x, y_l) − m(label) ) ]
```

where `m` is a per-margin scalar. This is not directly a PPO change, but it shapes the *reward surface* PPO optimizes against — and is part of why Llama-2's dual-RM system trains stably at β=0.01.

---

## §6 Costa-Huang — the details that matter for LLM-RL

[[costa-huang-ppo-details]] catalogs 37 tricks from OpenAI Baselines → Stable-Baselines. Most matter for MuJoCo/Atari and not LLMs. The subset that actually shifts training outcomes in RLHF (from the 2024 follow-up and cross-validated in [[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]]):

1. **Advantage normalization (whitening).** Per-minibatch, subtract mean, divide by std. Keeps `L^{CLIP}` well-scaled when reward magnitudes drift across prompts. All three frameworks do this.
2. **Value-loss clipping.** `V_clipped = clamp(V_new, V_old − c, V_old + c)`; `L^{VF} = max((V_new − R)^2, (V_clipped − R)^2)`. Mirrors the policy ratio clip; prevents the value head from overfitting a single rollout across K epochs. [[trl-ppo]] lines 820–840 implement this verbatim.
3. **Ratio clip bounds.** ε = 0.2 symmetric is the canonical default. Modern DAPO/OpenReasonerZero use **asymmetric** `ε_low = 0.2`, `ε_high = 0.28` ([[verl-ppo-loss]], [[openrlhf-ppo]]): clip less aggressively on the upside to let rare positive-advantage tokens upweight faster. Asymmetric clipping is the first thing to try when entropy collapses without reward gain.
4. **Length normalization of the policy loss.** `loss_agg_mode = "token-mean"` averages over all response tokens; `"seq-mean-token-sum"` (Dr.GRPO) sums per-sequence then averages across sequences. Choice materially changes gradients on long-tailed completion-length distributions. Length normalization makes short completions dominate less.
5. **Global gradient clipping.** Clip gradient norm to 1.0 at the policy (higher than the 0.5 used for MuJoCo PPO). Non-negotiable — a single rare token with `Â_t = 10` and `r_t = 5` can otherwise NaN the run.
6. **KL-to-reference in the reward, not the loss.** Canonical implementation; [[trl-ppo]] `non_score_reward`, [[verl-ppo-loss]] external `kl_penalty` using k3 estimator. Adding KL to the loss breaks GAE.
7. **Value head initialization from the RM's value head.** The RM is already trained to predict scalar preferences; its value-head weights give a warmer start than random init and avoid the initial value-loss spike that otherwise burns through 10–30% of the rollout budget.

Not on the LLM-RL critical path: observation normalization (no observations), orthogonal weight init (dominated by pretraining init), LR annealing (RLHF LRs are already tiny; cosine decay is typical but secondary).

---

## §7 The framework-level picture

[[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]] all implement the same loss:

```
pg_losses1 = −advantages · ratio
pg_losses2 = −advantages · clip(ratio, 1−ε_low, 1+ε_high)
pg_loss    = mean( max(pg_losses1, pg_losses2) )     # max because losses are negated
```

and all three add KL outside the loss, through a reward-shaping step:

```
r̂_t = r_t · 1[t = |y|]   −   β · (log π(y_t|…) − log π_ref(y_t|…))
```

The differences: TRL uses symmetric `cliprange` and an adaptive KL controller (InstructGPT style); OpenRLHF exposes `clip_eps_low/high`, dual-clip, GSPO sequence-level ratios, and vLLM importance-sampling correction; verl registers policy-loss variants (`vanilla`, `gspo`, etc.) and plugs loss-aggregation modes. The clipped-surrogate algebra is identical across the three; the ecosystem is the ablation surface.

---

## §8 Acceptance — what you should be able to do after this chapter

1. Derive `L^{CLIP}` from `L^{CPI}` by casework on the sign of `Â_t` and the position of `r_t` relative to the clip window. Explain why `min(·, clip)` is pessimistic.
2. Write the GAE expansion `Â_t = Σ (γλ)^k δ_{t+k}` and state the `λ=0` (TD) and `λ=1` (Monte-Carlo) limits.
3. Quote InstructGPT Equation 2 and label each term (RM score, per-token KL penalty, ptx mix). State why β is applied in the reward not the loss.
4. Quote Llama-2's PPO appendix hyperparameters (LR 1e-6, β=0.01, batch 512, seq 4K). Explain why the LR and β are smaller than InstructGPT's.
5. Name 5–7 Costa-Huang tricks that survive the LLM-RL transfer and justify each.

---

## Connections

- **ch-37 (Policy-Gradient Foundations)** — REINFORCE, score-function estimator, baseline subtraction; ch-38 picks up where the value-function baseline + trust region come in.
- **ch-39 (Offline Preference Optimization)** — DPO replaces the online KL penalty with a closed-form implicit reward `r_θ = β·log(π_θ/π_ref)`. Korbak's tilted-posterior view ([[kl-control-rlhf]]) shows why DPO and PPO-with-KL target the same `π*`.
- **ch-40+ (Critic-Free RL)** — GRPO, RLOO, REINFORCE++ drop the value head and use group-mean baselines; the `L^{CLIP}` surrogate survives, but `Â_t` is computed differently.

## Further reading

- [[trpo]] — trust-region derivation, natural-gradient step, line search.
- [[ppo]] — the clipped surrogate, GAE, canonical hparams.
- [[rlhf-instructgpt]] — PPO-ptx, β=0.02, γ=27.8, labeler protocol.
- [[kl-control-rlhf]] — KL-as-reward, k3 estimator, Korbak Bayesian reformulation.
- [[llama-2]] — dual RM, RSFT then PPO, LR 1e-6, β=0.01.
- [[costa-huang-ppo-details]] — 37 tricks + RLHF-specific follow-up.
- [[hf-rlhf-illustrated]], [[lilianweng-rlhf]] — tutorial framings.
- [[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]] — framework source as the ablation surface.

## Companion visualization

**[figures/ppo-clip.html](figures/ppo-clip.html)** — two-panel interactive. Panel 1: scrub `r ∈ [0, 2]` and toggle `Â` sign; plot `L^{CLIP}` and its gradient. Panel 2: β-sweep of the KL-vs-reward Pareto front with reward-hacking and mode-collapse regimes annotated.
