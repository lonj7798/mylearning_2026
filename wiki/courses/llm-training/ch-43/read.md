<!-- chapter: ch-43
     track: rl
     kind: content
     title: Entropy Dynamics and KL Control
     deps: [ch-42]
     sources: [[entropy-mechanism-llm-rl]], [[entropy-collapse-ppo]], [[entropy-regularization-ppo]],
              [[maximum-entropy-rl]], [[john-schulman-kl-tricks]], [[kl-control-rlhf]],
              [[nathan-lambert-entropy-rl]], [[openrlhf-entropy-debugging]],
              [[entropy-logging-patterns]], [[sampling-temperature-schedule]]
     figures: figures/entropy-dynamics.html
-->

# Chapter 43 — Entropy Dynamics and KL Control

> **Core insight.** RLHF and RLVR are not "reward maximization" — they are *two regularized optimizations sharing the same PPO/GRPO skeleton*. One regularizer is **entropy** H(π), which keeps the conditional token distribution wide so exploration does not die. The other is **KL-to-reference** KL(π‖π_ref), which keeps the policy near the SFT prior so reward hacking does not eat fluency. They answer different questions (distribution width at a state vs distance from a prior *over* states) and they fail in different ways. The two most important empirical facts of 2024–2025 RL are (a) Cui 2025's law `R(step) = −a·exp(H(step)) + b` — reward ceilings rise and fall on entropy dynamics — and (b) Schulman's k3 estimator `(p/q − 1) − log(p/q)` for KL, which replaced the naive k1 in every production stack because it is unbiased *and* non-negative. Everything else in this chapter is commentary on those two facts.
>
> **Guideline.** Log per-token entropy and per-batch KL *every step* and treat them as first-class diagnostics. If entropy drops below 0.1 nats on the last-token distribution, you have collapsed — covariance-targeted interventions ([[entropy-mechanism-llm-rl]]'s Clip-Cov / KL-Cov) beat flat entropy-bonus rescues. If you need a KL penalty, use k3 as a *loss term* (GRPO convention) or as *reward shaping* (PPO convention), never as k1, and never both at once. Rollout temperature is an exploration knob independent of both — raise it to `~1.1` before you retune β. And remember that DPO is not KL-free; it is just KL-absorbed into the implicit reward `r_θ = β·log(π/π_ref)`, so the same overoptimization curves apply.

---

## 1. The entropy mechanism — Cui 2025's law

[[entropy-mechanism-llm-rl]] (Cui, Zhang, Chen, Yuan et al., 2025, arXiv:2505.22617) is the paper that turned "entropy collapse" from folklore into a predictable curve. Across >20 model × algorithm × recipe combinations (PPO, GRPO, RLOO, Reinforce++), they find the same shape: per-token policy entropy `H(π)` decays roughly exponentially from its SFT initialization (2–3 nats) down toward zero within a few hundred updates, and the achievable reward `R` fits

```
R(step) = −a · exp(H(step)) + b
```

with run-specific positive `a, b`. Fit `a, b` on the first ~10% of training and you have a *prediction* of the reward ceiling: as `H → 0`, `R → b − a`. Past that, you are spending gradient on making a nearly-deterministic policy slightly more deterministic, and the marginal reward per entropy nat vanishes.

The mechanistic half of the paper is cleaner than the empirical half. For a softmax policy `π(a|s) ∝ exp(z_a(s))` trained with a policy-gradient objective that uses advantage `A(s,a)`, the expected one-step change in token entropy satisfies

```
E[ΔH(s)] ∝ − Cov_{a~π(·|s)}( log π(a|s),  A(s,a) )
```

This is the load-bearing equation of the chapter. It says entropy dies *in proportion to the covariance* between log-probability and advantage at a state. Large advantages on already-high-probability tokens (positive covariance) burn entropy fast; large advantages on low-probability tokens (negative covariance) *raise* entropy. An A2C-style flat entropy bonus `+ c_H · H(π)` treats every token symmetrically and therefore under-corrects when the collapse is driven by a thin tail of covariance outliers — exactly the LLM regime (see [[entropy-regularization-ppo]] on why LLM-scale vocabularies break the blanket-bonus assumption).

Two targeted interventions fall out of the covariance view:

- **Clip-Cov.** Rank tokens each batch by `p_t · A_t`. Zero the gradient of the top fraction (the paper's default is ~2%). The rest of the distribution keeps updating; the sharp spikes that would have collapsed entropy are silenced. Pseudocode:

  ```python
  # per batch, per token, inside the policy-gradient backward pass
  cov_score = (log_prob.exp() * advantage).detach()     # p_t * A_t
  k = max(1, int(0.02 * cov_score.numel()))             # top 2%
  topk_mask = torch.zeros_like(cov_score, dtype=torch.bool)
  topk_mask.view(-1)[cov_score.view(-1).topk(k).indices] = True
  pg_loss = -(advantage * log_prob).masked_fill(topk_mask, 0.0).mean()
  ```

- **KL-Cov.** Same outlier set, but instead of zeroing the gradient, apply a token-level KL penalty `β_KL · k3(π_new, π_old)` only on those tokens. Gradient still flows; magnitude is dampened.

Attested gains ([[entropy-mechanism-llm-rl]] Table of interventions): on Qwen2.5-7B and Qwen2.5-Math-7B with GRPO, entropy stays meaningfully above `0.1` for the full run, and AIME / MATH accuracy ceilings rise several points over vanilla GRPO — and over vanilla-plus-flat-entropy-bonus, which the paper shows over-corrects.

## 2. Collapse threshold and the triage tree

When does "entropy is decreasing" become "entropy has collapsed"? The practitioner thresholds converge across [[entropy-mechanism-llm-rl]], [[nathan-lambert-entropy-rl]], and [[openrlhf-entropy-debugging]]:

- **Per-token entropy `H < 0.1` nats sustained** for multiple updates → collapse.
- **`H < 0.2` nats on the last-token distribution** → Lambert's "stop and inspect" rule; you are in the danger zone even if reward is still moving.
- **`H` falls ≥ 30% in < 100 steps** → [[entropy-logging-patterns]]'s framework-agnostic collapse signature; usually paired with `ppo_kl > 0.1` and `clipfrac → 1`.

The two regimes disagree on the middle of the curve. [[entropy-collapse-ppo]] (Andrychowicz 2020 on MuJoCo) ranks entropy-coefficient tuning as a second-tier knob — advantage normalization and PPO-clip parameters dominate. At LLM scale the ordering is different: the collapse is driven by a small number of high-`p·A` tokens, so the flat bonus cannot rescue it and the entropy-related levers move up in priority.

Community-standard triage when entropy crashes ([[openrlhf-entropy-debugging]]):

1. Confirm **KL-to-reference is on and finite**. `nan` in `ppo_kl` or a β accidentally set to zero is the #1 cause.
2. Bump **rollout temperature** by 0.1–0.2 (see §5). This reinjects exploration *without* changing any objective coefficient.
3. Raise the **entropy coefficient** `c_H` by an order of magnitude (e.g. 0 → 1e-3, 1e-3 → 1e-2) — if the collapse is driven by the bulk of the distribution, this works; if driven by a thin tail, Clip-Cov is needed instead.
4. Check that **advantage normalization** is per-batch zero-mean unit-variance. Default ON in OpenRLHF / verl, OFF in TRL ([[entropy-logging-patterns]]).
5. Only after the above: suspect the reward signal, then retrain.

## 3. KL estimators — deriving k1, k2, k3

The KL penalty in every LLM-RL framework is a Monte-Carlo estimate. You have sample `a ~ q`, where `q = π_new` is the current policy and `p = π_ref` (or `π_old`) is what you compare to. With ratio `r(a) = p(a) / q(a)`, `KL(q‖p) = E_q[−log r]`. The question is *which one-sample estimator to use* for that expectation. [[john-schulman-kl-tricks]] gives the canonical trio.

**k1 — the direct estimator.** `k1 = −log r`. Unbiased by definition: `E_q[−log r] = KL(q‖p)`. Problems:
  - It can be negative on a single sample whenever `p(a) > q(a)` for that `a`, even though the true KL is ≥ 0. Monte-Carlo KL in a dashboard that occasionally goes negative is not a bug — it is k1.
  - High variance. Consider `q, p` Gaussians with the same mean but different variance: most samples give moderate `|log r|`, but an occasional tail sample gives enormous `|log r|`.

**k2 — the squared-log estimator.** `k2 = ½·(log r)²`. Motivation: Taylor-expand `−log r` around `r = 1`. Writing `r = 1 + ε`,

```
−log(1 + ε) = −ε + ε²/2 − ε³/3 + ...
E_q[−log r]   = 0 + ½·E_q[ε²] − (1/3)·E_q[ε³] + ...
```

because `E_q[ε] = E_q[r − 1] = E_q[p/q] − 1 = ∫ p dx − 1 = 0`. So to leading order, `KL ≈ ½·E_q[ε²] = ½·E_q[(log r)²]` (the second-moment-to-squared-log swap is another Taylor step). That gives k2 as a **biased but low-variance** estimator: its mean is not exactly `KL(q‖p)` but tracks it to O(ε²), and its variance is much smaller than k1's because `ε²` does not have k1's heavy negative tail.

**k3 — the convex unbiased estimator.** `k3 = r − 1 − log r`. Two properties you can verify on paper:

- *Unbiased.* `E_q[r − 1 − log r] = E_q[r] − 1 + KL(q‖p) = (∫ p dx) − 1 + KL = 0 + KL = KL`.
- *Non-negative.* Let `f(r) = r − 1 − log r`. Then `f(1) = 0`, `f'(r) = 1 − 1/r`, `f''(r) = 1/r² > 0` for `r > 0`. So `f` is strictly convex with minimum `0` at `r = 1`, and `f(r) ≥ 0` for every positive `r`. A single k3 sample is never negative.

Variance comparison near `r = 1`: expand `f(r) = ½·(r−1)² − (1/3)·(r−1)³ + ...`. Writing `ε = r − 1`, `f ≈ ½·ε²`. Meanwhile k2 ≈ ½·(log(1+ε))² ≈ ½·(ε − ε²/2)² = ½·ε² − ½·ε³ + O(ε⁴). So k2 and k3 agree to leading order in `ε`; k3's advantage over k2 is zero bias; k3's advantage over k1 is both bounded sign and smaller variance when the policy and reference are close (the operating regime after the first few PPO updates).

Putting the trio in a table:

| Estimator | Formula | Unbiased | Sign | Notes |
|-----------|---------|----------|------|-------|
| `k1` | `−log r` | Yes | any | High variance; can be negative per sample |
| `k2` | `½·(log r)²` | No | ≥ 0 | Lowest-bias only near `r ≈ 1`; stable monitor |
| `k3` | `r − 1 − log r` | Yes | ≥ 0 | Convex, non-negative, preferred default in GRPO |

Costa Huang's caveat ([[john-schulman-kl-tricks]]): `k3` "exploded for some reason" in early TRL experiments. The likely cause is very large `r` in tails — `r − 1` grows linearly while `−log r` grows logarithmically, so a tail event contributes a large positive k3 value. The fix practitioners converge to is *clamp log-ratio to `[−20, 20]` before exponentiation*, as verl does ([[entropy-logging-patterns]] verl excerpt).

## 4. KL-to-reward vs KL-as-loss

There are two places to inject the KL penalty, and they are not interchangeable.

**KL-to-reward (PPO / InstructGPT convention).** [[kl-control-rlhf]] formalizes this lineage from Jaques 2019 → Stiennon 2020 → Ouyang 2022 (InstructGPT). The per-token reward seen by the PPO advantage estimator is shaped:

```
r̂_t = r_t − β · ( log π_φ(y_t | y_<t, x) − log π_ref(y_t | y_<t, x) )
```

where `r_t` is the RM reward (usually zero except at the terminal token) and the KL term is applied on every token. The advantage estimator `A_t = Σ_t' γ^{t'−t} · (r̂_{t'} + ...)` then carries the KL cost into the policy gradient in a per-token, temporally-sensible way. InstructGPT reports `β ≈ 0.02`; [[openrlhf-entropy-debugging]] says production stacks run β in `0.01–0.1` of the reward scale. [[kl-control-rlhf]] emphasizes: *add KL to the reward, not to the loss* — otherwise the advantage-based policy gradient breaks and training empirically degrades.

**KL-as-loss (GRPO / DPO convention).** Modern GRPO (DeepSeekMath 2024, [[entropy-logging-patterns]] TRL GRPO excerpt) places the KL term directly in the per-token loss:

```python
# trl/trainer/grpo_trainer.py
per_token_kl  = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1   # k3
per_token_loss = per_token_loss + self.beta * per_token_kl
```

No reward shaping, no AdaptiveKLController. The justification is (a) GRPO's group-relative advantage is already well-defined without a baseline, so there is nothing to "protect" by routing KL through the reward channel; (b) k3-in-loss gives a correct, non-negative penalty with stable gradients.

**Korbak's Bayesian view** ([[kl-control-rlhf]], Korbak 2022) unifies both. The closed-form optimum of `argmax_π E_π[r] − β·KL(π‖π_ref)` is

```
π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )
```

i.e. the RL-with-KL objective is *exact variational inference* against the tilted posterior `π*`. `β` sets the tilt strength, `π_ref` is the prior, `r/β` is the log-likelihood. DPO inherits the same tilt directly: its implicit reward is `r_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))`, so training a DPO model is equivalent to optimizing an *implicitly* KL-regularized reward whose β is the same β that appears in RLHF-PPO. This is why DPO's overoptimization curves look like PPO's — same objective, different estimator.

**Direction matters.** The penalty as written is `KL(π_new ‖ π_ref)` — *reverse* KL from the reference's perspective. Reverse KL is mode-seeking: it drives `π_new` to put mass only where `π_ref` has mass, not to cover all of `π_ref`'s support. This is the formal reason RLHF-tuned models "sharpen" rather than "broaden" their output distributions; it is also the reason the entropy-collapse story and the KL-penalty story are separate — tightening `KL(π‖π_ref)` does not necessarily stop `H(π)` from shrinking, because the reference itself can be fairly peaked at many states.

## 5. Max-ent RL ancestry — what LM-RL inherited, what it dropped

Every entropy knob in modern LLM-RL descends from Soft Actor-Critic ([[maximum-entropy-rl]], Haarnoja 2018). SAC's objective is

```
J(π) = Σ_t E_{(s_t,a_t)~ρ_π} [ r(s_t, a_t) + α · H(π(·|s_t)) ]
```

— reward *plus* a per-step entropy bonus, not reward alone. The soft-Bellman equations replace `max_a Q` with a log-sum-exp:

```
V(s) = α · log ∫ exp( Q(s, a) / α ) da
Q(s, a) ← r(s, a) + γ · E_{s' ~ p}[ V(s') ]
```

and the optimal policy is the Boltzmann form `π*(a|s) ∝ exp( Q(s,a) / α )` — the same shape as Korbak's RLHF posterior (§4), with `Q/α` playing the role of `r/β`. SAC-v2 adds automatic α tuning: set a target entropy `H̄` (for continuous control, `H̄ = −dim(A)`) and gradient-descent on `L(α) = E[−α·(log π(a|s) + H̄)]` to keep `H(π) ≈ H̄` over the course of training.

What LLM-RL inherited:

- **The loss term itself.** A3C/PPO ([[entropy-regularization-ppo]]) ships the bonus `+ c_H · H(π)` verbatim as the on-policy, small-α limit of SAC. TRL's `entropy_coef`, OpenRLHF's `c_H`, and verl's entropy registry all descend from this line.
- **The target-entropy idea.** Cui 2025's covariance interventions are rediscovering, asymmetrically, what SAC-v2 did symmetrically: keep H above a floor by adjusting the regularization strength online.
- **The Boltzmann-tilt view.** Korbak's `π* ∝ π_ref · exp(r/β)` is structurally identical to SAC's `π* ∝ exp(Q/α)`; the reference policy in RLHF plays the role of the uniform prior in SAC.

What LM-RL dropped:

- **Off-policy replay.** SAC is off-policy (replay buffer + soft Q-learning with twin critics). LLM-RL is predominantly on-policy (PPO/GRPO with fresh rollouts). The trade is sample efficiency for stability at huge context / huge action space; see [[openrlhf-entropy-debugging]] on why async actor-learner does not restore true off-policyness.
- **Soft-Q critics.** Modern GRPO-family methods ([[entropy-logging-patterns]]) have no value head at all — advantages are group-relative z-scores. Entropy regularization now lives only in the actor loss or the reward stream.
- **Automatic α tuning.** Production LLM-RL uses fixed β and fixed c_H (with coarse schedules), not the online temperature controller from SAC-v2. Cui-style Clip-Cov / KL-Cov are the closest active reinventions.
- **True max-ent semantics.** `+ c_H · H(π)` as a loss term is only the small-α limit of max-ent RL. The full soft-Bellman recursion (entropy inside the Q target, log-sum-exp as the soft-max) is not used in any production LLM stack.

Rollout temperature ([[sampling-temperature-schedule]]) is the third independent lever: `P_T(a|s) = softmax(z(s)/T)` rescales logits at *sample time* without changing the parameters, so it widens rollouts without introducing objective bias. Practical recipes: R1 uses `T = 1.0, top_p = 0.95`; Tülu 3 anneals `T = 1.0` during RLVR down to `T = 0.7` for the final DPO polish; OpenRLHF's re-warm rule on collapse is to bump `T` by 0.2 for `N` rollouts before retuning anything else. Avoid `top_p < 1.0` and `top_k` during training — they introduce non-differentiable support truncation that PPO's clip cannot absorb.

## 6. Four-lever mental model and how they interact

The preceding five sections each introduce one lever. The practitioner decision is which to reach for first, and the answer depends on which failure mode you see. A one-table summary:

| Lever | Where it lives | Primary failure it addresses | Collapse effect | Side effect |
|-------|----------------|------------------------------|-----------------|-------------|
| Entropy bonus `c_H · H(π)` | actor loss | bulk-distribution premature sharpening | weak at LLM scale | pushes mass toward low-`p·A` tokens |
| Clip-Cov / KL-Cov | actor loss on top-`k%` tokens | covariance-driven tail collapse | strong; surgical | slight reward-ceiling delay |
| KL-to-reference `β · KL(π‖π_ref)` | reward stream (PPO) or loss (GRPO) | drift from SFT prior; reward hacking | indirect (reference may itself be peaked) | slows overall reward gain if β too large |
| Rollout temperature `T` | sampler | stalled exploration late in training | direct; immediate | off-policy bias if T ≠ 1 without IS correction |

The levers are mostly independent but not orthogonal. Raising `T` increases the effective `KL(π_sampled ‖ π_ref)` at fixed parameters because the sampled distribution widens, which can force the adaptive-KL controller to lower β and indirectly loosen the reference constraint ([[openrlhf-entropy-debugging]] triage step 2 exploits this). Clip-Cov and an entropy bonus roughly compose (they operate on disjoint token sets), but both reach for entropy — running them together without re-tuning is a common over-correction. KL-to-reference and KL-Cov are both KL terms on the same policy; applying both with default β values can cancel the gain from either.

**DPO as an entry in this table.** DPO is *not* KL-free. Its loss `L_DPO = −log σ(β · (log π(y_w|x) − log π(y_l|x) − log π_ref(y_w|x) + log π_ref(y_l|x)))` is exactly the RLHF objective with `r_θ = β · log(π/π_ref)` substituted in, so the β coefficient plays the same KL-budget role it does in PPO. The practical consequence: DPO inherits the same `β · KL(π‖π_ref)` overoptimization curves (ch-39's [[dpo]] / ch-41's [[reward-model-overoptimization]] discussions apply), but loses the per-token diagnostic dashboard — you cannot plot `ppo_kl` per step because there is no rollout. Lambert ([[nathan-lambert-entropy-rl]]) notes that DPO's implicit KL *partly* protects against entropy collapse because the loss shape encodes the reference, but it introduces its own failure mode of preference over-fitting on small pair sets.

**Entropy and reward hacking are different axes.** One of the [[openrlhf-entropy-debugging]] failure patterns to watch: "entropy healthy, rollouts exploding in length". This is the reward-hacking signature from ch-42 — the policy found a loophole in the reward (verbosity, refusal, format padding) that does not touch the distribution's width. The KL-to-reference term in §4 defends against this *to the extent the reference is itself well-behaved*, but does not address entropy collapse; conversely, Clip-Cov fixes entropy collapse without defending against length hacking. They are orthogonal failure axes and therefore need orthogonal diagnostics: the dashboard must include both per-token entropy and response-length histogram, and a collapse diagnosis without looking at length is incomplete.

---

## Connections

- **ch-38** (KL-Controlled RLHF) — derives PPO clip + InstructGPT's KL-to-reward convention; ch-43 extends that line with k1/k2/k3 and the collapse-plus-triage view.
- **ch-40** (Online / Group-Baseline RL) — GRPO uses k3-in-loss as documented here; the Dr.GRPO bias corrections are orthogonal to entropy dynamics.
- **ch-41 / ch-42** (Reward Modeling / Reward Hacking) — KL budgets from §4 bound the overoptimization curves introduced in ch-41; entropy collapse is a separate failure axis from reward hacking but often coincides with it.
- **ch-44** (Process Supervision / RLVR) — verifiable-reward runs still need entropy control; DeepSeek-R1's long rollouts double as exploration budget per [[sampling-temperature-schedule]].
- **ch-55** (verl internals) — the `kl_penalty` switch and `actor/entropy` logging are the files you will read when ch-55's tour lands on `core_algos.py`.

## Further reading

- [[entropy-mechanism-llm-rl]] — the `R = −a·exp(H) + b` law, Clip-Cov / KL-Cov.
- [[entropy-collapse-ppo]] — Andrychowicz 2020 large-scale PPO sweep; collapse as a failure mode.
- [[entropy-regularization-ppo]] — A3C/PPO entropy-bonus loss form, LLM-scale carry-through.
- [[maximum-entropy-rl]] — SAC and auto-α; the ancestor objective.
- [[john-schulman-kl-tricks]] — k1/k2/k3 derivation and practitioner caveats.
- [[kl-control-rlhf]] — Jaques / Stiennon / Ouyang / Korbak lineage, KL-as-reward convention.
- [[nathan-lambert-entropy-rl]] — practitioner synthesis on why entropy is the bottleneck in 2025.
- [[openrlhf-entropy-debugging]] — cross-framework triage protocol.
- [[entropy-logging-patterns]] — verl / OpenRLHF / TRL entropy and KL logging side-by-side.
- [[sampling-temperature-schedule]] — rollout-T as an independent exploration lever.

## Companion visualization

**[figures/entropy-dynamics.html](figures/entropy-dynamics.html)** — interactive two-panel figure. Panel 1: pick an intervention (none / Clip-Cov / KL-Cov / flat entropy bonus) and see entropy-vs-step curves with the `H < 0.1` collapse threshold overlaid; panel 2: sample 1000 `r` values from a toggleable `q` vs `p` regime and watch k1 / k2 / k3 estimator variance side-by-side. Curves are illustrative and match the qualitative shapes attested in [[entropy-mechanism-llm-rl]] and [[john-schulman-kl-tricks]]; absolute numbers are for pedagogy, not benchmarking.
