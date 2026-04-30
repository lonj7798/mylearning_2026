<!-- chapter: ch-39
     track: rl
     kind: content
     title: Offline Preference Optimization Family — DPO, IPO, KTO, SimPO, ORPO, RPO
     deps: [ch-38]
     sources: [[dpo]], [[ipo]], [[kto]], [[simpo]], [[orpo]], [[rpo]], [[hf-dpo-zoo]], [[openrlhf-dpo]], [[trl-online-dpo]], [[llama-3]], [[hh-rlhf]], [[ultrafeedback]]
     figures: figures/dpo-landscape.html
-->

# Chapter 39 — Offline Preference Optimization Family

> **Core insight.** DPO is the analytic solution to the KL-constrained RLHF objective pushed one algebraic step further: if `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)` is the *known* optimal policy for any reward `r`, then `r` can be read off the ratio `β log π*/π_ref` plus a prompt-only term that cancels under Bradley-Terry differences. Every variant in this chapter — IPO, KTO, SimPO, ORPO, RPO — keeps the implicit-reward trick and perturbs exactly one assumption in the sandwich around it. Knowing which assumption a variant trades is how you pick one.
>
> **Guideline.** Default to DPO: β=0.1, LR 5e-7, frozen SFT reference, single epoch on binarized [[ultrafeedback]]-style data. IPO for near-deterministic labels; KTO for unary thumbs; SimPO when reference memory hurts or DPO inflates length; ORPO for single-stage base→aligned; RPO when chosen-logprob collapses on reasoning. None replace online PPO/GRPO for open-ended verifier-driven RL — they are the offline cousins.

---

## §1 Setting the stage

The dataset is triples `{(x, y_w, y_l)}` or — for KTO — unary `{(x, y, b)}` with `b ∈ {desirable, undesirable}`. [[hh-rlhf]] gave the field 161K human pairs; [[ultrafeedback]] made scale viable by scoring multi-model completions with GPT-4. "Offline" means the pairs exist before training and are not resampled from the current policy. Contrast [[trl-online-dpo]], where each step passes `prompts = 2 * prompts` to `.generate()` and a judge picks the winner (ch-40 material).

---

## §2 Deriving DPO from the RLHF objective

### §2.1 The KL-constrained objective ([[dpo]] "Derivation bridge")

```
max_π  E_{x~D, y~π(·|x)}[ r(x, y) ]  −  β · D_KL( π(·|x) || π_ref(·|x) )              (1)
```

For a fixed prompt `x` this is a constrained optimization over a response distribution. Take the Lagrangian with `λ` enforcing `Σ_y π(y|x) = 1`:

```
L(π) = Σ_y π(y|x) r(x,y)  −  β Σ_y π(y|x) log[π(y|x)/π_ref(y|x)]  −  λ(Σ_y π(y|x) − 1).
```

Set `∂L/∂π(y|x) = 0`:

```
r(x,y) − β log[π(y|x)/π_ref(y|x)] − β − λ = 0
  ⇒ π(y|x) = π_ref(y|x) · exp( (r(x,y) − β − λ) / β ).
```

Fold the constant into a normalizer so `π` sums to 1, giving the **Gibbs optimum**:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )                                    (2)
Z(x) = Σ_{y'} π_ref(y'|x) · exp( r(x,y') / β ).
```

This is exact, not variational: the unique optimum of (1) is `π_ref` tilted by the exponentiated reward.

### §2.2 Inverting (2) for the reward

Taking logs:

```
r(x, y) = β · log[ π*(y|x) / π_ref(y|x) ]  +  β · log Z(x)                             (3)
```

Equation (3) says: given any optimal policy, the reward it was optimal for is `β` times the log-ratio to the reference, plus a prompt-only term `β log Z(x)`.

### §2.3 Killing the partition function via Bradley-Terry

The Bradley-Terry preference model (same one ch-38's reward models are trained under):

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) ).                                         (4)
```

Substitute (3) into (4):

```
r(x, y_w) − r(x, y_l)
  = [β log π*(y_w|x)/π_ref(y_w|x) + β log Z(x)]  −  [β log π*(y_l|x)/π_ref(y_l|x) + β log Z(x)]
  = β log[π*(y_w|x)/π_ref(y_w|x)]  −  β log[π*(y_l|x)/π_ref(y_l|x)].                   (5)
```

`Z(x)` cancels. The partition function — the single reason RLHF had to be solved with PPO rollouts — is gone, because both sides share the same prompt.

### §2.4 The DPO loss

Substitute (5) into the Bradley-Terry NLL, treat `π*` as learnable `π_θ`:

```
L_DPO(π_θ; π_ref) = − E_{(x, y_w, y_l) ~ D}[
    log σ( β · log[π_θ(y_w|x)/π_ref(y_w|x)]
         − β · log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]                                         (6)
```

That is [[dpo]] Equation 7. Define the **implicit reward**:

```
r̂_θ(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ]                                           (7)
```

Then (6) reads `L = − E[log σ(r̂_θ(y_w) − r̂_θ(y_l))]` — a Bradley-Terry classifier on the implicit reward. No separate RM, no rollouts.

### §2.5 Gradient behavior

```
∇L_DPO = − β · E[ σ(r̂_l − r̂_w) · ( ∇ log π_θ(y_w|x) − ∇ log π_θ(y_l|x) ) ]             (8)
```

The `σ(r̂_l − r̂_w)` factor is automatic hard-example mining: pairs already ranked correctly (`r̂_w > r̂_l`) get gradient ≈ 0; violations get full weight. No curriculum needed.

### §2.6 β is the KL budget

Larger β scales the sigmoid argument so the same log-ratio buys a bigger preference probability — meaning the policy has to move *less* from `π_ref`. Smaller β permits larger drift. β sets the KL budget without computing a KL penalty at train time. [[llama-3]] and [[hf-dpo-zoo]] both default to **β = 0.1**; [[dpo]] swept `{0.05, 0.1, 1, 5}`.

---

## §3 IPO — identity link, bounded target

[[ipo]] identifies DPO's failure mode: when preferences are deterministic, MLE drives `σ → 1`, so `r̂_w − r̂_l → ∞` and the log-ratio explodes unchecked. Inside the ΨPO family `max_π E[Ψ(P(y_w≻y_l|x))] − τ D_KL(π||π_ref)`, `Ψ(p) = log(p/(1−p))` recovers DPO; `Ψ(p) = p` gives IPO, whose practical loss is squared error around a finite target:

```
L_IPO = E[ ( h_π(y_w, y_l, x) − 1/(2τ) )^2 ]                                           (9)
h_π  = log[π_θ(y_w|x)/π_ref(y_w|x)] − log[π_θ(y_l|x)/π_ref(y_l|x)].
```

Optimum is `h* = 1/(2τ)` — finite, tunable. DPO drops both sides of log-prob; IPO mostly drops only the rejected. Use on distilled / BoN-gated data.

---

## §4 KTO — prospect theory on unary labels

[[kto]] targets a different seam: production feedback is usually one thumbs-up/down, not paired. Same implicit reward as (7). Add a detached batch-level reference point as the prospect-theory "status quo": `z_0 = KL(π_θ(·|x')||π_ref(·|x'))` (minibatch estimate, no grad). Asymmetric value function (concave on gains, steeper on losses):

```
v(x, y) = { λ_D · σ( β · ( r_θ(x,y)  − z_0 ) )       if y desirable
          { λ_U · σ( β · ( z_0 − r_θ(x,y) ) )        if y undesirable                  (10)

L_KTO = E_{(x,y) ~ D}[ λ_y − v(x, y) ].                                                (11)
```

Class-balance recipe: `λ_D / λ_U = N_U / N_D`. Paper default β=0.1, λ_D = λ_U = 1.0, LR 5e-7, batch 32, one epoch. Robust to 90/10 imbalance; matches or beats DPO 1B–30B; uniquely eats unary data.

---

## §5 SimPO — drop the reference, normalize by length

[[simpo]] targets DPO's length bias. `β log π/π_ref` is a sum of per-token log-ratios; uniform shifts grow the margin linearly in `|y|`, so DPO inflates response length 30–60 % over SFT. SimPO replaces the log-ratio with an *average* log-probability and drops `π_ref`:

```
r_SimPO(x, y) = (β / |y|) · Σ_{t=1..|y|} log π_θ( y_t | x, y_<t )                      (12)

L_SimPO = − E[ log σ( (β/|y_w|) log π_θ(y_w|x)
                    − (β/|y_l|) log π_θ(y_l|x)  −  γ ) ]                               (13)
```

Per-token reward means β must be ~20× DPO's: **β ∈ [2, 10]**, γ ∈ [0.3, 1.6], γ/β ∈ [0.25, 0.5]. Dropping `π_ref` halves memory and doubles throughput; length stays within ±5 % of SFT. Failure modes: too-low β → entropy collapse; too-high γ → gradient vanishes; very clean data → add label smoothing or a small SFT loss (that's ORPO).

---

## §6 ORPO — joint SFT and odds-ratio, one stage

[[orpo]]'s motivation: plain SFT raises log-prob of rejected completions alongside chosen (Figure 3). ORPO adds an odds-ratio penalty *inside* the SFT loss — one stage, no `π_ref`, no separate SFT pre-stage.

Odds of a response under the policy:

```
odds_θ(y | x) = π_θ(y|x) / ( 1 − π_θ(y|x) ).                                           (14)
```

Odds-ratio loss (sigmoid on the log odds-ratio difference):

```
L_OR = − log σ( log[ odds_θ(y_w|x) / odds_θ(y_l|x) ] ).                                (15)
```

Total loss:

```
L_ORPO = E[ L_SFT(y_w | x) + λ · L_OR ],      L_SFT = − log π_θ(y_w | x).              (16)
```

Hyperparameters: λ = 0.1 (Mistral-7B), 0.2 (Llama-2-7B), 0.25 (Phi-2); LR 8e-6 (SFT-scale, not DPO-scale), 3–5 epochs, batch 64 prompts. `L_SFT` anchors chosen-side log-prob so the ref-free odds-ratio term is safe; remove either piece and the loss degenerates.

---

## §7 RPO — DPO plus NLL anchor, iterated

[[rpo]] addresses DPO's worst reasoning failure: both chosen and rejected log-probs drift *down*. The sigmoid cares about the ratio, so an easy gradient direction is "push rejected down hard, let chosen drift down slightly" — sampled accuracy drops as loss improves. Fix: add an NLL anchor on the chosen CoT.

```
L_RPO = L_DPO(π_θ; π_ref)  +  α · L_NLL( y_w | x )
L_NLL(y_w|x) = − (1/|y_w|) · log π_θ( y_w | x ).                                       (17)
```

Settings: α=1.0, β=0.1, N=30 samples per problem, 3–4 iterations; each iteration refreshes `π_ref` to the previous one. GSM8K: 55.6 → 81.6 across rounds. [[llama-3]] bakes the same trick in with NLL coef 0.2 on chosen inside one-shot DPO. Iterate the generate→label→train loop with a verifier and you have [[trl-online-dpo]] without the wrapper.

---

## §8 Comparison table

| Variant | Ref-free | β role | Data shape | Length bias | Best-for |
|---------|----------|--------|------------|-------------|----------|
| **DPO** | no | KL budget; smaller = more drift | paired pref, any noise | high (inflates length) | general default; [[ultrafeedback]] / [[hh-rlhf]] |
| **IPO** | no | target = 1/(2τ) | paired, near-deterministic | moderate | distilled / BoN-gated; DPO saturating |
| **KTO** | uses π_ref | reference-point offset | unary (thumbs up/down) | low-moderate | production thumbs; imbalanced classes |
| **SimPO** | yes | per-token scale, β ~2–10 | paired, needs γ tuning | low (length-invariant) | low memory; DPO-length regression; clean offline |
| **ORPO** | yes | λ trades SFT vs OR | paired, starts from base | low-moderate | single-stage base→aligned |
| **RPO** | no | same as DPO; α anchors chosen | paired, verifiable-answer | low | reasoning; DPO collapsing chosen-logprob |

Axes you are trading: **reference memory** (SimPO, ORPO save it); **label shape** (KTO alone eats unary; ORPO starts from base); **deterministic-data safety** (IPO, ORPO tolerate; DPO overfits); **length neutrality** (SimPO wins); **chosen-anchor** (ORPO SFT-term; RPO NLL-term; KTO implicit via z_0).

---

## §9 Framework reality check

The whole family ships behind one module. From [[openrlhf-dpo]]:

```python
# openrlhf/models/loss.py — DPOLoss.forward
pi_logratios  = policy_chosen_logps    - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios                             # this is h of (5)/(9)

if self.ipo:
    losses = (logits - 1 / (2 * self.beta)) ** 2                  # IPO — Eq (9)
else:
    losses = (
        -F.logsigmoid( self.beta * logits) * (1 - self.label_smoothing)
        -F.logsigmoid(-self.beta * logits) *      self.label_smoothing   # cDPO
    )
chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
```

`logits` is exactly `h` from (5); DPO vs IPO differs only in the last line. `chosen_rewards` is the implicit reward (7) — the primary diagnostic. `concatenated_forward` packs chosen and rejected along batch, halving activation memory on 70B+. [[hf-dpo-zoo]] exposes the same switch as `loss_type ∈ {sigmoid, hinge, ipo, kto_pair, ...}` on `trl.DPOTrainer`. [[trl-online-dpo]] swaps offline pairs for on-policy `.generate()` + judge — same loss algebra, different data source.

---

## §10 Llama 3's industrial recipe

From [[llama-3]] §Post-Training / Table 7:

| Knob | Value | Rationale |
|------|-------|-----------|
| β | 0.1 | Community default; KL budget |
| LR | 1e-5 | 20× paper's 5e-7 because 405B absorbs it |
| NLL aux coeff | 0.2 on chosen | Prevents chosen-logprob decay — weakened RPO anchor |
| Epochs | 1 per round | Single pass prevents format drift |
| Rounds | 6 | SFT → Rejection Sampling → DPO, repeated |
| Preference batch | most-recent only | Older batches cause format drift |

Six-round structure is the offline family's answer to stale `π_ref`: refresh to the previous round's checkpoint, resample preferences from the latest model. Offline within a round, online across rounds. Ch-40 makes the inner loop online.

---

## §11 Decision framework and monitoring

Walk in order: (1) **No paired preferences?** → KTO. (2) **Can't afford `π_ref`?** → SimPO or ORPO. (3) **Reasoning task, sampled accuracy metric?** → RPO, or DPO + NLL coef 0.2 per [[llama-3]]. (4) **Near-deterministic data (distilled / BoN / verifier-labelled)?** → IPO. (5) **Skip SFT, go base → aligned?** → ORPO. (6) **Otherwise** → DPO, β=0.1, LR 5e-7, single epoch on [[ultrafeedback]]-binarized. When uncertain: train two variants at matched budget; compare on held-out implicit-reward accuracy + AlpacaEval 2 / Arena-Hard.

Three metrics to watch in [[openrlhf-dpo]] / [[trl-online-dpo]] logs:

1. **`rewards/accuracies = P(r̂_w > r̂_l)`** — rises from ~0.5 to 0.7–0.9 over the first epoch.
2. **`chosen_logps`** absolute — must not drop monotonically; if it does, the variant is wrong (use RPO/ORPO) or β too small.
3. **Length ratio vs SFT checkpoint** — > 1.3 on DPO = length-gaming; switch to SimPO. Clean DPO on [[ultrafeedback]]-binarized: accuracies ≈ 0.75, `chosen_logps` flat, length 1.1–1.4.

---

## Companion visualization

**[figures/dpo-landscape.html](figures/dpo-landscape.html)** — interactive loss landscape over the implicit-reward margin `m = r̂_w − r̂_l` for DPO / IPO / KTO / SimPO. Sliders for β, τ, γ; toggle variants. The overlay shows why IPO pulls toward a finite target while DPO drives `m → ∞`, why SimPO's curve shifts by `γ`, and why KTO's loss shape flips for desirable vs undesirable.

---

## Connections

- **ch-37 / ch-38 (RLHF reward modelling, Bradley-Terry RMs)** — (4) is ch-38's BT likelihood; §2.3 earns the right to skip RM training.
- **ch-40 (Online DPO / iterative)** — continues from §10; offline batches → on-policy samples labelled by judge/verifier.
- **ch-41 (PPO for RLHF)** — the algorithm DPO replaces; contrast is rollout cost, not objective.
- **ch-42 (GRPO)** — group-relative online variant; where offline PO stalls.

## Further reading

[[dpo]], [[ipo]], [[kto]], [[simpo]], [[orpo]], [[rpo]] — primary papers. [[hf-dpo-zoo]] — unified TRL `loss_type` interface. [[openrlhf-dpo]] — canonical offline trainer. [[trl-online-dpo]] — on-policy cousin for ch-40. [[llama-3]] — industrial 6-round recipe. [[hh-rlhf]], [[ultrafeedback]] — canonical data.
