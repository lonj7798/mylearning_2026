<!-- chapter: ch-39
     track: rl
     kind: content
     title: Offline Preference Optimization Family — DPO, IPO, KTO, SimPO, ORPO, RPO
     deps: [ch-38]
     sources: [[dpo]], [[ipo]], [[kto]], [[simpo]], [[orpo]], [[rpo]], [[hf-dpo-zoo]], [[openrlhf-dpo]], [[trl-online-dpo]], [[llama-3]], [[hh-rlhf]], [[ultrafeedback]]
     figures: figures/dpo-landscape.html
-->

# 제39장 — Offline Preference Optimization Family

> **핵심 통찰.** DPO는 KL-constrained RLHF objective의 해석적 해를 대수적으로 한 단계 더 밀어붙인 것이다. 어떤 reward `r`에 대해서도 `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`가 *known* optimal policy라면, `r`은 ratio `β log π*/π_ref`에 prompt-only 항을 더한 것으로 읽어낼 수 있고, 이 항은 Bradley-Terry 차분에서 사라진다. 이 장의 모든 variant — IPO, KTO, SimPO, ORPO, RPO — 는 implicit-reward trick을 유지하고 그 주변의 sandwich에서 정확히 하나의 가정을 흔든다. 어떤 가정을 variant가 trade하는지 아는 것이 선택 방법이다.
>
> **지침.** 기본값은 DPO다. β=0.1, LR 5e-7, frozen SFT reference, binarized [[ultrafeedback]]-style data에서 single epoch. near-deterministic label에는 IPO, unary thumbs에는 KTO, reference memory가 부담되거나 DPO가 length를 부풀릴 때는 SimPO, single-stage base→aligned에는 ORPO, reasoning에서 chosen-logprob이 collapse될 때는 RPO를 쓴다. 이들은 open-ended verifier-driven RL에서 online PPO/GRPO를 대체하지 않는다. offline cousin이다.

---

## §1 무대 설정

Dataset은 triple `{(x, y_w, y_l)}`이거나 — KTO의 경우 — unary `{(x, y, b)}`이며 `b ∈ {desirable, undesirable}`이다. [[hh-rlhf]]는 161K human pair를 제공했고, [[ultrafeedback]]은 GPT-4로 multi-model completion을 scoring하여 scale을 가능하게 했다. "Offline"은 pair가 training 전에 존재하고 current policy에서 다시 sample되지 않는다는 뜻이다. [[trl-online-dpo]]와 대조하라. 거기서는 각 step이 `prompts = 2 * prompts`를 `.generate()`에 넘기고 judge가 winner를 고른다(ch-40의 내용).

---

## §2 RLHF objective에서 DPO 유도하기

### §2.1 KL-constrained objective ([[dpo]] "Derivation bridge")

```
max_π  E_{x~D, y~π(·|x)}[ r(x, y) ]  −  β · D_KL( π(·|x) || π_ref(·|x) )              (1)
```

고정된 prompt `x`에 대해 이는 response distribution 위의 constrained optimization이다. `Σ_y π(y|x) = 1`을 강제하는 `λ`를 둔 Lagrangian을 취한다.

```
L(π) = Σ_y π(y|x) r(x,y)  −  β Σ_y π(y|x) log[π(y|x)/π_ref(y|x)]  −  λ(Σ_y π(y|x) − 1).
```

`∂L/∂π(y|x) = 0`으로 놓으면:

```
r(x,y) − β log[π(y|x)/π_ref(y|x)] − β − λ = 0
  ⇒ π(y|x) = π_ref(y|x) · exp( (r(x,y) − β − λ) / β ).
```

상수를 normalizer에 접어 넣어 `π`가 합 1이 되게 하면 **Gibbs optimum**이 나온다.

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )                                    (2)
Z(x) = Σ_{y'} π_ref(y'|x) · exp( r(x,y') / β ).
```

이는 variational approximation이 아니라 exact다. (1)의 유일한 optimum은 exponentiated reward로 tilt된 `π_ref`다.

### §2.2 (2)를 reward에 대해 뒤집기

log를 취하면:

```
r(x, y) = β · log[ π*(y|x) / π_ref(y|x) ]  +  β · log Z(x)                             (3)
```

Equation (3)은 말한다. 어떤 optimal policy가 주어졌을 때, 그 policy가 최적이었던 reward는 reference 대비 log-ratio에 `β`를 곱한 것에 prompt-only term `β log Z(x)`를 더한 값이다.

### §2.3 Bradley-Terry로 partition function 죽이기

Bradley-Terry preference model(ch-38의 reward model이 학습되는 것과 같은 모델):

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) ).                                         (4)
```

(3)을 (4)에 대입하면:

```
r(x, y_w) − r(x, y_l)
  = [β log π*(y_w|x)/π_ref(y_w|x) + β log Z(x)]  −  [β log π*(y_l|x)/π_ref(y_l|x) + β log Z(x)]
  = β log[π*(y_w|x)/π_ref(y_w|x)]  −  β log[π*(y_l|x)/π_ref(y_l|x)].                   (5)
```

`Z(x)`가 cancel된다. RLHF를 PPO rollout으로 풀어야 했던 단 하나의 이유인 partition function이 사라진다. 양쪽이 같은 prompt를 공유하기 때문이다.

### §2.4 DPO loss

(5)를 Bradley-Terry NLL에 대입하고, `π*`를 learnable `π_θ`로 취급한다.

```
L_DPO(π_θ; π_ref) = − E_{(x, y_w, y_l) ~ D}[
    log σ( β · log[π_θ(y_w|x)/π_ref(y_w|x)]
         − β · log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]                                         (6)
```

이것이 [[dpo]] Equation 7이다. **Implicit reward**를 정의하자.

```
r̂_θ(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ]                                           (7)
```

그러면 (6)은 `L = − E[log σ(r̂_θ(y_w) − r̂_θ(y_l))]`로 읽힌다. implicit reward 위의 Bradley-Terry classifier다. 별도 RM도, rollout도 없다.

### §2.5 Gradient behavior

```
∇L_DPO = − β · E[ σ(r̂_l − r̂_w) · ( ∇ log π_θ(y_w|x) − ∇ log π_θ(y_l|x) ) ]             (8)
```

`σ(r̂_l − r̂_w)` factor는 자동 hard-example mining이다. 이미 제대로 ranked된 pair(`r̂_w > r̂_l`)는 gradient가 ≈ 0이고, violation은 full weight를 받는다. curriculum이 필요 없다.

### §2.6 β는 KL budget이다

β가 크면 같은 log-ratio가 더 큰 preference probability를 사도록 sigmoid argument가 scale된다. 즉 policy가 `π_ref`에서 *덜* 움직여도 된다. β가 작으면 더 큰 drift를 허용한다. β는 train time에 KL penalty를 계산하지 않고도 KL budget을 정한다. [[llama-3]]와 [[hf-dpo-zoo]]는 모두 **β = 0.1**을 기본값으로 둔다. [[dpo]]는 `{0.05, 0.1, 1, 5}`를 sweep했다.

---

## §3 IPO — identity link, bounded target

[[ipo]]는 DPO의 failure mode를 짚는다. preference가 deterministic이면 MLE가 `σ → 1`을 밀고, 따라서 `r̂_w − r̂_l → ∞`가 되어 log-ratio가 unchecked로 폭발한다. ΨPO family `max_π E[Ψ(P(y_w≻y_l|x))] − τ D_KL(π||π_ref)` 안에서 `Ψ(p) = log(p/(1−p))`는 DPO를 회복하고, `Ψ(p) = p`는 IPO를 준다. IPO의 practical loss는 유한 target 주변의 squared error다.

```
L_IPO = E[ ( h_π(y_w, y_l, x) − 1/(2τ) )^2 ]                                           (9)
h_π  = log[π_θ(y_w|x)/π_ref(y_w|x)] − log[π_θ(y_l|x)/π_ref(y_l|x)].
```

Optimum은 `h* = 1/(2τ)`이며, 유한하고 조정 가능하다. DPO는 log-prob 양쪽을 모두 낮추는 경향이 있고, IPO는 대체로 rejected만 낮춘다. distilled / BoN-gated data에 사용하라.

---

## §4 KTO — unary label 위의 prospect theory

[[kto]]는 다른 seam을 겨냥한다. production feedback은 보통 paired가 아니라 thumbs-up/down 하나다. implicit reward는 (7)과 같다. prospect-theory의 "status quo"로 detached batch-level reference point를 더한다. `z_0 = KL(π_θ(·|x')||π_ref(·|x'))`(minibatch estimate, no grad). value function은 asymmetric하다(gain에서는 concave, loss에서는 더 가파름).

```
v(x, y) = { λ_D · σ( β · ( r_θ(x,y)  − z_0 ) )       if y desirable
          { λ_U · σ( β · ( z_0 − r_θ(x,y) ) )        if y undesirable                  (10)

L_KTO = E_{(x,y) ~ D}[ λ_y − v(x, y) ].                                                (11)
```

Class-balance recipe는 `λ_D / λ_U = N_U / N_D`다. 논문 기본값은 β=0.1, λ_D = λ_U = 1.0, LR 5e-7, batch 32, one epoch. 90/10 imbalance에도 robust하고, 1B–30B에서 DPO와 맞먹거나 이기며, unary data를 먹을 수 있는 유일한 방법이다.

---

## §5 SimPO — reference를 버리고 length로 normalize하기

[[simpo]]는 DPO의 length bias를 겨냥한다. `β log π/π_ref`는 token별 log-ratio의 합이다. uniform shift는 margin을 `|y|`에 선형으로 키우므로 DPO는 SFT보다 response length를 30–60% 부풀린다. SimPO는 log-ratio를 *평균* log-probability로 바꾸고 `π_ref`를 버린다.

```
r_SimPO(x, y) = (β / |y|) · Σ_{t=1..|y|} log π_θ( y_t | x, y_<t )                      (12)

L_SimPO = − E[ log σ( (β/|y_w|) log π_θ(y_w|x)
                    − (β/|y_l|) log π_θ(y_l|x)  −  γ ) ]                               (13)
```

token별 reward이므로 β는 DPO의 약 20배여야 한다. **β ∈ [2, 10]**, γ ∈ [0.3, 1.6], γ/β ∈ [0.25, 0.5]. `π_ref`를 버리면 memory가 절반이 되고 throughput이 두 배가 된다. length는 SFT의 ±5% 안에 머문다. Failure mode: β가 너무 낮으면 entropy collapse, γ가 너무 높으면 gradient vanishing, data가 매우 깨끗하면 label smoothing 또는 작은 SFT loss(그것이 ORPO)를 추가한다.

---

## §6 ORPO — joint SFT와 odds-ratio, one stage

[[orpo]]의 동기: plain SFT는 chosen과 함께 rejected completion의 log-prob도 올린다(Figure 3). ORPO는 SFT loss *안에* odds-ratio penalty를 더한다. one stage이며, `π_ref`도 없고 별도 SFT pre-stage도 없다.

policy 아래 response의 odds:

```
odds_θ(y | x) = π_θ(y|x) / ( 1 − π_θ(y|x) ).                                           (14)
```

Odds-ratio loss(log odds-ratio 차이에 대한 sigmoid):

```
L_OR = − log σ( log[ odds_θ(y_w|x) / odds_θ(y_l|x) ] ).                                (15)
```

Total loss:

```
L_ORPO = E[ L_SFT(y_w | x) + λ · L_OR ],      L_SFT = − log π_θ(y_w | x).              (16)
```

Hyperparameters: λ = 0.1(Mistral-7B), 0.2(Llama-2-7B), 0.25(Phi-2); LR 8e-6(SFT-scale, DPO-scale 아님), 3–5 epochs, batch 64 prompts. `L_SFT`가 chosen-side log-prob을 anchor하므로 ref-free odds-ratio term이 안전하다. 둘 중 하나를 제거하면 loss가 degenerate된다.

---

## §7 RPO — DPO plus NLL anchor, iterated

[[rpo]]는 DPO의 최악의 reasoning failure를 다룬다. chosen과 rejected log-prob가 둘 다 *내려간다*. sigmoid는 ratio만 신경 쓰므로 쉬운 gradient 방향은 "rejected를 세게 낮추고, chosen은 조금 내려가게 둔다"가 된다. sampled accuracy는 loss가 개선되는 동안 떨어진다. Fix는 chosen CoT에 NLL anchor를 더하는 것이다.

```
L_RPO = L_DPO(π_θ; π_ref)  +  α · L_NLL( y_w | x )
L_NLL(y_w|x) = − (1/|y_w|) · log π_θ( y_w | x ).                                       (17)
```

Settings: α=1.0, β=0.1, problem당 N=30 samples, 3–4 iterations. 각 iteration은 `π_ref`를 직전 checkpoint로 refresh한다. GSM8K: round를 거치며 55.6 → 81.6. [[llama-3]]는 one-shot DPO 안에서 chosen에 NLL coef 0.2를 넣어 같은 trick을 baked in한다. verifier와 함께 generate→label→train loop를 반복하면 wrapper 없는 [[trl-online-dpo]]가 된다.

---

## §8 Comparison table

| Variant | Ref-free | β role | Data shape | Length bias | Best-for |
|---------|----------|--------|------------|-------------|----------|
| **DPO** | no | KL budget; 작을수록 더 많은 drift | paired pref, any noise | high (length 부풀림) | general default; [[ultrafeedback]] / [[hh-rlhf]] |
| **IPO** | no | target = 1/(2τ) | paired, near-deterministic | moderate | distilled / BoN-gated; DPO saturating |
| **KTO** | uses π_ref | reference-point offset | unary (thumbs up/down) | low-moderate | production thumbs; imbalanced classes |
| **SimPO** | yes | per-token scale, β ~2–10 | paired, γ tuning 필요 | low (length-invariant) | low memory; DPO-length regression; clean offline |
| **ORPO** | yes | λ trades SFT vs OR | paired, starts from base | low-moderate | single-stage base→aligned |
| **RPO** | no | same as DPO; α anchors chosen | paired, verifiable-answer | low | reasoning; DPO collapsing chosen-logprob |

Trade하는 축: **reference memory**(SimPO, ORPO가 절약), **label shape**(KTO만 unary를 먹고, ORPO는 base에서 시작), **deterministic-data safety**(IPO, ORPO가 견딤; DPO는 overfit), **length neutrality**(SimPO가 이김), **chosen-anchor**(ORPO SFT-term; RPO NLL-term; KTO는 z_0을 통해 implicit).

---

## §9 Framework reality check

전체 family는 하나의 module 뒤에 실린다. [[openrlhf-dpo]]에서:

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

`logits`는 정확히 (5)의 `h`다. DPO와 IPO는 마지막 줄에서만 다르다. `chosen_rewards`는 implicit reward (7)이며 primary diagnostic이다. `concatenated_forward`는 chosen과 rejected를 batch 방향으로 pack해 70B+에서 activation memory를 절반으로 줄인다. [[hf-dpo-zoo]]는 같은 switch를 `trl.DPOTrainer`의 `loss_type ∈ {sigmoid, hinge, ipo, kto_pair, ...}`로 노출한다. [[trl-online-dpo]]는 offline pair를 on-policy `.generate()` + judge로 바꾼다. loss algebra는 같고 data source만 다르다.

---

## §10 Llama 3의 industrial recipe

[[llama-3]] §Post-Training / Table 7에서:

| Knob | Value | Rationale |
|------|-------|-----------|
| β | 0.1 | Community default; KL budget |
| LR | 1e-5 | 405B가 흡수하므로 논문의 5e-7보다 20× |
| NLL aux coeff | chosen에 0.2 | chosen-logprob decay 방지 — 약화된 RPO anchor |
| Epochs | round당 1 | single pass가 format drift를 막음 |
| Rounds | 6 | SFT → Rejection Sampling → DPO, 반복 |
| Preference batch | most-recent only | 오래된 batch는 format drift를 유발 |

Six-round structure는 stale `π_ref`에 대한 offline family의 답이다. 직전 round checkpoint로 refresh하고, 최신 model에서 preference를 resample한다. round 안에서는 offline, round 사이에서는 online이다. ch-40은 inner loop를 online으로 만든다.

---

## §11 Decision framework and monitoring

순서대로 걸어가라. (1) **Paired preference가 없는가?** → KTO. (2) **`π_ref`를 감당할 수 없는가?** → SimPO 또는 ORPO. (3) **Reasoning task이고 sampled accuracy metric인가?** → RPO, 또는 [[llama-3]]처럼 DPO + NLL coef 0.2. (4) **Near-deterministic data(distilled / BoN / verifier-labelled)인가?** → IPO. (5) **SFT를 건너뛰고 base → aligned로 가는가?** → ORPO. (6) **그 외** → DPO, β=0.1, LR 5e-7, [[ultrafeedback]]-binarized data에서 single epoch. 확실하지 않으면 같은 budget으로 두 variant를 학습하고 held-out implicit-reward accuracy + AlpacaEval 2 / Arena-Hard에서 비교하라.

[[openrlhf-dpo]] / [[trl-online-dpo]] log에서 볼 세 metric:

1. **`rewards/accuracies = P(r̂_w > r̂_l)`** — 첫 epoch 동안 ~0.5에서 0.7–0.9로 오른다.
2. **`chosen_logps`** absolute — 단조롭게 떨어지면 안 된다. 떨어진다면 variant가 틀렸거나(RPO/ORPO 사용) β가 너무 작다.
3. **Length ratio vs SFT checkpoint** — DPO에서 > 1.3이면 length-gaming이다. SimPO로 바꿔라. [[ultrafeedback]]-binarized에서 깨끗한 DPO는 accuracies ≈ 0.75, `chosen_logps` flat, length 1.1–1.4다.

---

## Companion visualization

**[figures/dpo-landscape.html](figures/dpo-landscape.html)** — implicit-reward margin `m = r̂_w − r̂_l` 위의 DPO / IPO / KTO / SimPO interactive loss landscape. β, τ, γ slider와 variant toggle이 있다. overlay는 왜 IPO가 유한 target으로 당기고 DPO가 `m → ∞`를 미는지, 왜 SimPO curve가 `γ`만큼 shift되는지, 왜 KTO의 loss shape가 desirable vs undesirable에서 뒤집히는지 보여 준다.

---

## Connections

- **ch-37 / ch-38 (RLHF reward modelling, Bradley-Terry RMs)** — (4)는 ch-38의 BT likelihood다. §2.3은 RM training을 건너뛸 권리를 벌어 온다.
- **ch-40 (Online DPO / iterative)** — §10에서 이어진다. offline batch → judge/verifier가 label한 on-policy sample.
- **ch-41 (PPO for RLHF)** — DPO가 대체하는 알고리즘이다. 대비점은 objective가 아니라 rollout cost다.
- **ch-42 (GRPO)** — group-relative online variant; offline PO가 멈추는 곳.

## Further reading

[[dpo]], [[ipo]], [[kto]], [[simpo]], [[orpo]], [[rpo]] — primary papers. [[hf-dpo-zoo]] — unified TRL `loss_type` interface. [[openrlhf-dpo]] — canonical offline trainer. [[trl-online-dpo]] — ch-40의 on-policy cousin. [[llama-3]] — industrial 6-round recipe. [[hh-rlhf]], [[ultrafeedback]] — canonical data.
