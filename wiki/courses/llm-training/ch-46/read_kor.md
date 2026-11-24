# Chapter 46 — Lab: Negative-Signal Ablation과 Held-Out Capability Retention을 포함한 DPO 또는 RLVR 실험

> **핵심 통찰.** preference 또는 RL stage는 target metric을 올리는 동시에 general capability를 낮출 수 있으며, 두 효과는 서로 다른 instrument로 측정한다. GRPO 아래 Qwen2.5-7B에서 Omni-MATH-Train pass@1은 base, step 150, step 300, step 450에 걸쳐 9.9 → 26.1 → 33.6 → 42.5로 오른다. 반면 pass@256은 67.2 → 66.3 → 65.3 → 64.3으로 내려간다([[rlvr-beyond-base-model]], App. C.5 Table 4). Qwen2.5-Math-7B에서는 incorrect rollout만으로 학습하면 MATH pass@1 75.7, pass@256 96.9에 도달하며, 이는 base model과 같다. 반면 correct rollout만으로 학습하면 pass@1 74.1, pass@256 91.2에 도달한다([[negative-sample-reinforcement]], Table 1). pass@1만 보고한 run으로는 이 경우들을 구분할 수 없다.
>
> **지침.** DPO 또는 RLVR stage를 실행할 때 전후로 세 가지를 측정하라. in-domain target, stage가 학습하지 않은 held-out general-capability suite, 그리고 target domain에서 큰 k의 coverage다. 세 번째는 첫 번째가 오르는 동안 내려갈 수 있기 때문이다([[rlvr-beyond-base-model]], Table 4). negative가 기여했다는 claim이 deliverable이면, positive-only, negative-only, full cell을 matched step에서 실행하라. 측정한 ordering이 k에 따라 다르기 때문이다([[negative-sample-reinforcement]], Table 1). run을 base model의 prior가 아니라 reward에 귀속해야 한다면, model family 두 번째에서 cell 하나를 반복하라. Qwen2.5-Math-7B에서 random reward는 ground truth의 +29.1과 비교해 MATH-500에서 +21.4 point를 냈지만 OLMo2는 flat으로 남았다([[spurious-rewards-rlvr]], §2, §3).

---

## 이 챕터가 general-purpose model에 중요한 이유

이것은 RL track의 lab이다. 앞의 chapter들은 objective를 가르쳤다. ch-39의 [[dpo]], ch-40의 group-baseline RL, ch-43의 entropy 및 KL control, ch-43a의 negative gradient, ch-44의 verifiable reward, 그리고 ch-45a부터 ch-45d까지의 stage recipe다. 이 lab은 reader가 이 stage 중 하나를 실행하고, 최적화한 metric뿐 아니라 model 전체에 그것이 무슨 일을 했는지에 관한 evidence를 만들 수 있는지 확인한다.

pipeline에서의 위치는 evaluation 직전 마지막 post-training stage다. 이 stage가 도입한 narrowing은 ch-47의 evaluation suite가 측정하고 실제로 ship되는 것이다. narrowing mechanism 두 가지가 문서화되어 있으며 서로 다르게 측정한다.

1. **target domain 내부의 coverage loss.** pass@1이 커지는 동안 k번 시도 안에 model이 풀 수 있는 problem의 집합이 줄어든다([[rlvr-beyond-base-model]], Table 4). 큰 k의 pass@k로 측정한다.
2. **target domain 밖의 forgetting.** 더 이른 단계에서 얻은 capability가 저하된다. [[rls-razor]]에서 Qwen2.5-3B-Instruct의 RL 및 SFT run 전반 forgetting은, new task에서 평가한 base policy에 대한 forward KL의 함수로 plot하면 하나의 curve로 collapse한다. quadratic fit은 LLM experiment에서 R² = 0.71, 통제된 MNIST setting에서 R² = 0.96이다(§4, Figure 11, Figure 3). held-out suite와 그 KL로 측정한다.

그러므로 이 lab의 deliverable은 target score가 높은 checkpoint가 아니다. sweep cell을 KL axis에 놓고 in-domain gain을 held-out retention에 대비해 보고하는 memo다.

---

## §1 Lab이 만드는 산출물

repository를 가진 reader가 각각 재현할 수 있는 artifact 세 가지다.

1. **Sweep.** cell별 directory 하나이며, 각 directory에는 `metrics.jsonl`(per-step training signal), `eval_before.json`과 `eval_after.json`(retention suite), `passk.json`(target domain의 pass@k), final checkpoint, training code의 commit hash가 들어 있다.
2. **Plot 두 개.** (a) in-domain target 대 held-out suite average, cell당 point 하나. (b) training prompt에서 측정한 reference policy에 대한 forward KL 대 held-out suite average, cell당 point 하나. Plot (b)는 reader 자신의 cell에 적용한 [[rls-razor]] 측정이다(§4, Figure 3 middle).
3. **Memo 하나**, `rl-experiment-memo.md`. setup, three-axis table, negative-signal ablation result, minimal reproduction이 있는 failure post-mortem 하나, data가 반박한 prediction 하나를 포함한다.

모든 cell의 prediction은 첫 run 전에 쓰고 commit한다. 이는 model이 아니라 analysis를 확인한다. 방향을 예측했고 확인된 cell은 mechanism을 뒷받침하며, prediction을 반박한 cell은 그 mechanism의 gap을 드러낸다.

**Option A**는 [[dpo]]를 사용하는 offline preference optimization이다. **Option B**는 group baseline([[grpo]], [[dr-grpo]]의 normalization 포함)을 쓰는 RLVR이다. 둘 다가 아니라 하나를 고른다. ablation과 retention suite가 graded part이며 training run보다 비용이 크다.

---

## §2 Option A — β sweep과 displacement gate를 포함한 DPO

### 2.1 Objective와 그것이 움직이는 대상

DPO loss는 [[dpo]] Eq. 7이다.

```
L_DPO = −E_(x, y_w, y_l) ~ D [ log σ( β log (π_θ(y_w|x)/π_ref(y_w|x)) − β log (π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

- `x`: prompt. `y_w`: preferred response. `y_l`: dispreferred response.
- `π_θ`: 학습 중인 policy. `π_ref`: frozen reference, 보통 SFT checkpoint.
- `β`: implicit KL constraint의 coefficient. `σ`: logistic function.
- implicit reward는 `r̂_θ(x, y) = β log (π_θ(y|x)/π_ref(y|x))`다([[dpo]] §4, §5.1).

gradient는 `∇_θ L = −β·E[ σ(r̂_θ(x,y_l) − r̂_θ(x,y_w)) · (∇_θ log π(y_w|x) − ∇_θ log π(y_l|x)) ]`다([[dpo]] App. A.4, Eq. 21–22). chosen log-probability를 올리는 term 하나와 rejected log-probability를 내리는 term 하나를 포함한다. loss에는 첫 term이 이겨야 한다는 요구가 없다. 차이만 나타난다.

### 2.2 Sweep이 해결해야 하는 측정 문제

대부분 framework가 log하는 quantity는 log-probability가 아니라 implicit reward다. OpenRLHF `DPOLoss`는 `chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()`와 대응 rejected value를 반환한다. `DPOTrainer`는 `loss`, `acc`, `chosen_reward`, `reject_reward`, `nll_loss`를 log하며 raw chosen/rejected log-probability는 log에 없다([[openrlhf-dpo]], loss.py L264–281, dpo_trainer.py L184–197).

계산 예제. pair 하나가 initialization에서 `log π_θ(y_w|x) = log π_ref(y_w|x) = −40`, `log π_θ(y_l|x) = log π_ref(y_l|x) = −42`라고 하자. training 뒤에는 `log π_θ(y_w|x) = −44`, `log π_θ(y_l|x) = −50`이다. β = 0.1이면 다음과 같다.

| quantity | before | after |
|---|---|---|
| implicit reward, chosen | 0.0 | 0.1 × (−44 − (−40)) = −0.4 |
| implicit reward, rejected | 0.0 | 0.1 × (−50 − (−42)) = −0.8 |
| margin (chosen − rejected) | 0.0 | +0.4 |
| accuracy (`chosen_reward > reject_reward`) | — | passes |

margin은 올랐고 accuracy metric도 통과하지만 chosen response는 reference 때보다 `e^{-4} ≈ 0.018`배만큼 likely해졌다. 이것이 likelihood displacement다. [[likelihood-displacement]]에서 on-policy refusal pair의 DPO는 Llama-3-8B-Instruct training-set refusal rate를 74.4%에서 33.4%로, Gemma-2B-IT는 80.5%에서 54.8%로 낮췄다. preferred log-probability의 mean change는 세 run에서 각각 −48.1 ± 22.1, −59.2 ± 5.3이었다(§6.2, Table 16). margin-only acceptance gate는 이런 run을 통과시킨다.

따라서 lab은 framework metric 외에 absolute value인 `logps/chosen`, `logps/rejected`와 model이 계속 생성해야 하는 fixed held-out response 200개의 mean log-probability를 log한다. 예를 들면 non-target math problem의 correct answer, unsafe prompt에 대한 refusal이다.

### 2.3 Sweep axis와 두 toggle

- **Axis:** β ∈ {0.05, 0.1, 0.5}. 세 값 모두 [[dpo]]에 나타난다. 0.05와 0.1은 IMDb reward–KL frontier sweep(§6.1, Fig. 2 left, 22 run), 0.1은 별도 언급 없는 모든 run의 값, 0.5는 GPT-J TL;DR run(App. B)의 값이다.
- **Toggle 1, NLL anchor.** 추가 term `−λ log π_θ(y_w|x)`는 chosen log-probability의 time derivative에 `λ‖∇ log π_θ(y_w|x)‖² ≥ 0`를 기여하므로([[likelihood-displacement]] App. E, Proposition 1), displacement를 더 나쁘게 만들 수 없다. OpenRLHF는 기본값 0인 `--model.nll_loss_coef`로 이를 노출한다([[openrlhf-dpo]], train_dpo.py L246; commit `64c1cc4`의 dpo_trainer.py L169–180). 이를 켠 cell 하나를 실행한다.
- **Toggle 2, CHES filtering.** [[likelihood-displacement]]에서 length-normalized CHES score가 가장 낮은 pair 5%를 유지하면 SFT term보다 refusal rate를 더 회복했고, 15%까지 유지해도 유사한 결과가 났다(§6.3, footnote 9). 이 toggle은 optional이고 dataset 전체에 forward pass 하나가 든다.

β 방향은 추측이 아니라 algebra로 고정된다. implicit reward는 log-ratio에 β를 곱한 것이므로, β가 더 작으면 같은 implicit-reward margin을 만들기 위해 policy가 log-probability space에서 더 멀리 움직여야 한다. 따라서 π_ref 대비 측정한 KL은 β = 0.05 cell에서 가장 커야 한다.

---

## §3 Option B — KL sweep과 negative-signal ablation을 포함한 RLVR

### 3.1 Objective

group-baseline objective는 [[grpo]] Eq. 3이다.

```
J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t { min[ ρ_i,t Â_i,t , clip(ρ_i,t, 1−ε, 1+ε) Â_i,t ] − β D_KL[π_θ ‖ π_ref] } ]
ρ_i,t = π_θ(o_i,t | q, o_i,<t) / π_θold(o_i,t | q, o_i,<t)
Â_i,t = (r_i − mean(r)) / std(r)
```

- `q`: question. `o_i`: G개 sampled response 중 i번째. `|o_i|`: token 길이.
- `ρ_i,t`: per-token importance ratio. `ε`: clip parameter. `β`: KL coefficient.
- `Â_i,t`: `o_i`의 모든 token에 할당한 advantage이며 `r_i`는 verifier의 binary reward다.

[[dr-grpo]]는 `1/|o_i|` factor가 incorrect response의 per-token penalty를 response가 길수록 작게 하므로 policy가 긴 incorrect response를 선호하게 됨을 보인다(§3.1). 또 `1/|o_i|`와 `std(r)` division을 모두 없애면 benchmark에서 incorrect response 길이가 줄어든다(§3.2, Fig. 5). lab은 KL sweep이 length bias와 confound되지 않도록 unbiased normalization을 기본으로 쓴다. TRL에서는 `loss_type="dr_grpo"`, `scale_rewards="none"`이 이를 구현한다(`loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)`, [[trl-grpo]] grpo_trainer.py L2548–2562; advantage scaling L2127–2149).

### 3.2 두 axis

**Axis 1 — KL coefficient β ∈ {0.0, 0.001, 0.04}.** 모든 값은 source가 사용한 값이다. 0.0은 모든 Dr. GRPO run([[dr-grpo]] App. G Table 6) 및 TRL default([[trl-grpo]], L650–653), 0.001은 [[negative-sample-reinforcement]] PPO/GRPO baseline(App. D.1), 0.04는 DeepSeekMath-RL 7B([[grpo]] §4.2)의 값이다. 이 axis는 plot (b)에 필요하다. [[rls-razor]]가 forgetting predictor로 식별한, policy와 reference 사이 거리를 움직인다.

**Axis 2 — matched step에서의 training signal.** cell 네 개다.

| cell | gradient를 만드는 rollout | implementation |
|---|---|---|
| positive-only (PSR) | reward = 1만 | `r = 0` rollout의 loss를 mask |
| negative-only (NSR) | reward = 0만 | `r = 1` rollout의 loss를 mask |
| full | 둘 다, group-normalized advantage | unmodified objective |
| W-REINFORCE, λ = 0.1 | 둘 다, positive term은 0.1배 | positive-rollout loss에 λ를 곱함 |

`L_W-REINFORCE = −E[Σ_{y: r=1} λ·π_θ(y|x)] − E[Σ_{y: r=−1} −π_θ(y|x)]`는 [[negative-sample-reinforcement]] Eq. 9다. λ = 1은 REINFORCE, λ = 0은 NSR을 복원한다.

재현하거나 반박할 결과는 [[negative-sample-reinforcement]] Table 1에서 가져왔다(Qwen2.5-Math-7B, MATH training set, prompt당 rollout 8개, temperature 0.6/top-p 0.95에서 sample 256개 평가).

| signal | MATH pass@1 | MATH pass@256 | AIME 2025 pass@1 | AIME 2025 pass@256 |
|---|---|---|---|---|
| base model | 63.2 | 96.9 | 6.1 | 46.7 |
| PSR (positive only) | 74.1 | 91.2 | 11.6 | 43.3 |
| NSR (negative only) | 75.7 | 96.9 | 10.0 | 53.3 |
| GRPO (full) | 76.3 | 95.5 | 10.3 | 50.0 |
| W-REINFORCE (λ = 0.1) | 76.6 | 96.7 | 10.6 | 56.7 |

[figures/rl-sweep.html](figures/rl-sweep.html)은 reader가 benchmark를 바꾸고 curve에서 단일 k를 읽게 하며, 이것이 이 section이 요구하는 operation이다. Panel 1은 MATH, AIME 2025, AMC23에서 k=1부터 256까지 Table 1의 일곱 row를, panel 2는 evaluation set 세 개에 대해 [[rlvr-beyond-base-model]] Table 4의 GRPO training step 전반 pass@1 대 pass@256을, panel 3은 App. E Table 4의 λ sweep을 그린다. 모든 값은 published table에서 읽은 것이며, lab 자체 ablation curve를 읽는 reference다.

### 3.3 Verifier와 prompt

reward는 {0, 1}을 반환하는 programmatic verifier다. training 전 hand-labeled `(response, gold)` pair set에서 verifier를 점검해 모든 label을 재현해야 한다. hole이 있는 verifier는 training-side diagnostic으로는 보이지 않는 방식으로 reward와 metric을 불일치시킨다. answer extraction은 dataset 자신의 format과 맞아야 한다. solution 끝이 `\boxed{...}`인 dataset에는 `####` separator가 없으며, GSM8K의 `####` convention용 splitter는 whole solution text를 gold answer로 조용히 반환한다.

---

## §4 Instrumentation

두 option 모두에서 step마다 다음을 기록한다.

| signal | Option A source | Option B source |
|---|---|---|
| loss | trainer | trainer |
| implicit reward, chosen 및 rejected | `chosen_reward`, `reject_reward` ([[openrlhf-dpo]] L191–192) | — |
| absolute `logps/chosen`, `logps/rejected` | lab이 추가(default log 아님) | — |
| `π_ref` 대비 greedy-output drift | lab이 추가; N step마다, §6.1 item 3 참조 | lab이 추가; 같은 절차 |
| model이 유지해야 하는 held-out response의 log-probability | lab이 추가, fixed response 200개 | lab이 추가, 같은 set |
| reward mean 및 std | — | reward function output |
| zero-variance group의 비율 | — | `frac_reward_zero_std` ([[trl-grpo]] L2150, L2186) |
| policy entropy | generation 필요; 아래 참조 | 매 step log됨([[trl-grpo]] L2571–2612) |
| advantage sign별 clip fraction | — | low-clip은 A < 0일 때만, high-clip은 A > 0일 때만 count([[trl-grpo]] L2589–2590) |
| reference 대비 KL | generation 필요; 아래 참조 | k3 estimator, `exp(ref − logp) − (ref − logp) − 1` ([[trl-grpo]] L2493–2497) |
| sampler-versus-trainer KL | — | `vllm_kl` = mean(rollout_log_probs − old_log_probs) ([[openrlhf-ppo]] loss.py L173) |

Option A에는 rollout이 없으므로 training loop에서 reference 대비 entropy와 KL을 얻을 수 없다. fixed prompt set에서 N step마다 별도 generation pass로 만든다. current checkpoint와 `π_ref`에서 prompt당 completion 8개를 sample하고, 그 set에서 [[rls-razor]]가 predictor로 쓰는 forward KL `E_{x∼τ}[KL(π_ref‖π_θ)]`을 계산한다(§6, Table 1). training-loop implicit reward를 KL의 대용으로 쓰는 것은 유효하지 않다. 이는 두 특정 response에서의 `β·(log π_θ − log π_ref)`이지 policy에 대한 expectation이 아니다.

evaluation을 slice한다. aggregate improvement는 per-slice regression과 공존할 수 있으므로, held-out target evaluation은 difficulty bucket별(Option B) 또는 prompt source별(Option A)로 보고한다.

---

## §5 Held-out retention suite와 SFT control

### 5.1 Suite

전후에 실행하며, 양쪽에서 decoding setting을 동일하게 한다.

| component | 다루는 범위 | fixed protocol |
|---|---|---|
| [[mmlu-pro]] | target domain 밖의 knowledge와 multi-step reasoning | 5-shot CoT; fixed prompt 하나 |
| [[ifeval]] | 명시적 output-constraint following | prompt-level/instruction-level, strict/loose 모두 보고 |
| code set 하나 | 어느 option도 학습하지 않는 capability | 전후에 같은 set |
| [[xstest]] | over-refusal: safe prompt 250개와 unsafe contrast 200개 | temperature 0, max new token 256 |
| non-target math (Option A) | in-family transfer | fixed sample count |

measurement error도 결과 일부다. [[mmlu-pro]]는 prompt style 24개 전반 model score가 약 2 point(최대 3.74) 변한다고 보고한다. MMLU에서는 4–5 point(최대 10.98)다(§6.3). 그러므로 MMLU-Pro에서 약 2 point보다 작은 retention 차이는 prompt 하나만으로 해결되지 않는다. [[mmlu-pro]]는 CoT가 GPT-4o score를 19.1 point 바꾼다고도 보고한다(§6.2, Table 3). 따라서 generation configuration은 단순히 “유사”하면 안 되고 양쪽이 동일해야 한다. [[xstest]]는 Llama-2 system prompt를 제거하면 safe prompt full refusal이 38%에서 14%로 움직인 반면 unsafe prompt full refusal은 99.5%에서 97.5%로만 움직였다고 보고한다(Table 2). 그러므로 system prompt도 protocol 일부다.

### 5.2 SFT control

추가 cell 하나를 실행한다. preference 또는 RL stage가 쓴 같은 data로 supervised fine-tuning한다. Option A에서는 chosen response, Option B에서는 correct rollout이며, comparable in-domain score에 도달하도록 tuned learning rate를 쓴다. [[rls-razor]]는 math, science Q&A, tool use의 Qwen2.5-3B-Instruct에서 RL Pareto-frontier model은 new-task accuracy가 오르는 동안 prior-benchmark performance를 거의 바꾸지 않지만, SFT model은 같은 new-task accuracy에 substantial forgetting이 있어야 도달한다고 보고한다(§3, Figure 2). 둘은 forward KL에 대해 plot하면 하나의 curve 위에 놓인다(§4). 이 control은 sweep에서 본 retention이 algorithm에서 왔는지, 우연히 취한 distribution shift 크기에서 왔는지를 알려 준다.

### 5.3 Coverage

base, SFT control, 모든 sweep cell에 대해 target domain의 pass@k를 계산한다. [[negative-sample-reinforcement]]의 unbiased estimator(Eq. 5)를 쓴다. problem당 sample `n`개 중 correct `c`개이면 `pass@k = 1 − C(n−c, k)/C(n, k)`다.

계산 예제. `n = 16`, `c = 2`, `k = 8`이면 `C(14,8) = 3003`, `C(16,8) = 12870`이므로 `pass@8 = 1 − 3003/12870 = 1 − 0.2333 = 0.7667`이다. sample 8개의 draw 하나에서 pass@8을 계산하면 이 problem의 값은 0 또는 1이지만, sample 16개에서의 estimator는 0.767이며 variance가 훨씬 작다.

§3.2 ordering이 보이려면 k가 최소 64에 도달해야 한다. MATH에서 PSR과 NSR은 k = 1에서 1.6 point, k = 256에서 5.7 point 차이다([[negative-sample-reinforcement]], Table 1).

---

## §6 Acceptance gate와 failure post-mortem

### 6.1 Acceptance gate

gate는 run 전에 쓰고, run 뒤 평가하며, 통과 여부와 관계없이 보고하는 조건이다. option마다 네 개가 있고 cell을 해당 stage의 successful run이라고 하려면 넷 모두 성립해야 한다.

**Option A (DPO).**

1. **Margin.** held-out pair slice에서 mean implicit-reward margin `r̂(x, y_w) − r̂(x, y_l)`이 positive이고 상승한다. 필요하지만 충분하지는 않다. §2.2는 두 log-probability가 모두 내려가도 margin은 오르는 run을 보였다.
2. **Displacement gate.** training pair에서 `log π_θ(y_w|x) − log π_ref(y_w|x)` 평균을 기록하고 run 전에 정한 threshold와 비교한다. 피할 regime은 [[likelihood-displacement]]가 측정한 것이다. 세 run에서 Llama-3-8B-Instruct는 preferred log-probability mean change −48.1 ± 22.1, Gemma-2B-IT는 −59.2 ± 5.3이었고, 같은 run의 training-set refusal rate는 74.4%→33.4%, 80.5%→54.8%였다(§6.2, Table 16). 논문은 safe/unsafe run을 나누는 threshold를 보고하지 않으므로 lab의 수치는 pre-registered choice이며, 통과 여부와 상관없이 observed value를 보고한다.
3. **Greedy-output drift.** fixed prompt 200개에서 identical setting으로 `π_ref`와 `π_θ`를 greedy decode하고, greedy continuation이 조금이라도 달라진 prompt 비율 및 두 continuation의 mean normalized edit distance를 보고한다. log-probability와 함께 log하는 이유는 둘이 독립적으로 움직일 수 있기 때문이다. positive margin, stable `log π_θ(y_w|x)`, near-zero greedy drift면 model이 output을 바꾸지 않고 preference를 sharpen한 것이다. positive margin, falling chosen log-probability, 다른 response class 쪽의 high greedy drift면 displacement이며 drift가 이동한 class를 명명한다.
4. **Held-out log-probability.** §2.2의 held-out response 200개 mean log-probability가 내려가지 않는다. refusals가 바로 확률을 잃은 response였으므로, 이는 [[likelihood-displacement]] refusal collapse를 잡았을 gate다.

**Option B (RLVR).**

1. **Reward agreement.** held-out slice에서 training reward와 같은 completion의 strict re-scoring이 run 전에 정한 tolerance 안에서 일치한다.
2. **Entropy floor와 group variance.** policy entropy가 사전 고정 floor 이상이고 `frac_reward_zero_std`가 정한 fraction 미만이다. [[entropy-mechanism-llm-rl]]는 base model 11개에서 entropy consumption 73%, performance gain 76%가 gradient step 2400개 중 첫 200개에 일어났다고 보고한다(§2.3). 그러므로 training 끝에서만 점검하는 floor는 이미 그 결정이 끝난 뒤 점검하는 것이다.
3. **Coverage.** target domain pass@64가 base model pass@64보다 stated tolerance 이상 낮지 않다. pass@1과 중복되지 않는 ordering은 §3.2에 있다.
4. **Retention.** held-out suite average가 §5.1 measurement error 이상 before-value보다 낮지 않다.

### 6.2 Failure post-mortem

memo에는 reader가 실행할 reproduction까지 만든 post-mortem 하나를 넣는다. option은 다섯 가지며, 모든 gate를 통과한 cell도 이 signature 중 하나가 나타났다가 제거되면 해당한다.

| Failure | Signal pattern | Minimal reproduction | Reproduction status |
|---|---|---|---|
| Reward hacking | training reward 상승, 같은 completion strict re-scoring은 상승하지 않음, 둘 gap 확대 | verifier exact-answer comparison을 response 어느 곳의 gold string도 허용하는 substring test로 바꾸고 fixed step 재실행 | Lab construction; 이 chapter source set에는 이 specific loophole 보고 없음 |
| Entropy collapse | entropy 단조 하락; `frac_reward_zero_std` 상승; reward plateau | entropy bonus 없이 `ε_high = ε_low = 0.2`로 두고 `ε_high = 0.28` cell과 대조 | Attested: clip-higher는 AIME24 avg@32를 36→38로 옮김([[dapo]], Table 1); Clip-Cov/KL-Cov는 Qwen2.5-32B 7-benchmark average를 45.8→50.3/52.2로 옮김([[entropy-mechanism-llm-rl]], Table 2) |
| Length bias | accuracy flat인데 *incorrect* response mean length 증가 | `loss_type`을 `"dr_grpo"`에서 `"grpo"`로 되돌려 lab default가 제거한 `1/\|o_i\|`, `std(r)` normalization 복원 | mechanism attested([[dr-grpo]], §3.1); 두 term 제거는 incorrect-response length를 낮춤(§3.2, Fig. 5). growth가 보이는 step은 보고되지 않았으니 자신의 run에서 기록 |
| Likelihood displacement / over-suppression | margin 상승, `logps/chosen`/`logps/rejected` 모두 하락, high greedy drift, held-out refusal/correct-answer rate 하락 | chosen/rejected가 near-duplicate(high CHES)인 pair를 만들고 β = 0.05 cell 학습; NLL anchor를 켜서 재실행 | 2B/8B에서 attested([[likelihood-displacement]], §6.2 Table 16); chosen log-probability에 대한 NLL term 기여는 non-negative(App. E, Proposition 1) |
| Non-target eval capability regression | in-domain target 상승; held-out suite가 §5.1 error보다 크게 하락; reference forward KL 큼 | 같은 in-domain score learning rate로 §5.2 SFT control을 실행해 plot (b) KL axis에 두 point 배치 | math/science Q&A/tool use의 Qwen2.5-3B-Instruct에서 attested([[rls-razor]], §3 Figure 2, §4) |

다섯 중 두 쌍은 lab이 추가한 instrumentation으로만 구분된다. reward hacking과 capability regression은 둘 다 “숫자는 올랐는데 model은 나빠졌다”로 보이며, *training* reward와 strict re-scoring의 불일치가 구분한다. entropy collapse와 length bias는 둘 다 plateau로 보이며, falling entropy가 동반되는지 growing incorrect-response length가 동반되는지가 구분한다.

---

## §7 자원 제약 경로

이 lab에서 graded part는 instrumentation, retention suite, post-mortem이며 줄이지 않는다. 줄이는 것은 model과 sweep axis 수다.

| Element | Full path | Constrained path | 변경이 안전한 이유 |
|---|---|---|---|
| Model | 3B–7B | 1B | §1 narrowing mechanism은 0.5B 이상에서 보고되었다. [[entropy-mechanism-llm-rl]] protocol은 0.5B부터 32B까지 포괄한다(§2.2). |
| Option | A 또는 B | B만 | Option B ablation은 chapter 제목이 가리키는 deliverable이며 rollout이 필요한 쪽이다. |
| Sweep axis | KL axis × signal axis | 단일 `β_KL = 0.0`의 signal axis만 | KL axis는 plot (b)가 필요로 하는 축이다. axis 하나면 plot (b)를 RL cell과 SFT control의 두 point로 대체한다. |
| Ablation cell | positive-only, negative-only, full, W-REINFORCE | negative-only와 full | reference result에서 large k 차이가 가장 큰 두 cell이다. NSR MATH pass@256 96.9, GRPO 95.5([[negative-sample-reinforcement]], Table 1). |
| pass@k | sample 64개에서 k ∈ {1, …, 64} | 변경 없음 | §3.2 ordering을 보려면 k가 64까지 가야 한다. |
| Retention suite | [[mmlu-pro]], [[ifeval]], code set 하나, [[xstest]] | 변경 없음 | component를 빼면 그 capability claim의 유일한 evidence가 사라진다. compute 때문에 빼야 하면 남은 것의 average를 보고하는 대신 memo에 현재 unmeasured인 capability를 명시한다. |
| Post-mortem | 다섯 중 하나 | 변경 없음 | — |

constrained path의 run은 scale claim을 뒷받침할 수 없으며 memo에도 그렇게 쓴다. 1B model을 움직이는 reward가 7B에서는 아무 일도 하지 않을 수 있고 반대도 일어난다. [[spurious-rewards-rlvr]]에서 random reward는 Qwen2.5-Math-7B의 MATH-500 accuracy를 ground-truth reward의 +29.1 대비 +21.4 point 올렸지만, 같은 signal에서 OLMo2는 flat이었다(§2, §3). 한 size의 model 하나는 그 size의 그 model에 대한 evidence다.

---

## Negative sample과 negative feedback

두 option 모두 negative를 **gradient로서** 사용한다(course standard type 4). 즉 특정 sample likelihood를 명시적으로 낮춘다. content 또는 conditioning으로서 negative는 쓰지 않으며, 이는 ch-29d와 ch-31에 나온다.

**negative의 출처.** Option A는 human 또는 judge가 label한 preference pair의 `rejected` side다. Option B는 verifier score 0인 rollout이다. verifier false-negative rate는 negative의 false-negative rate다. correctly formatted answer를 reject하는 tolerant grader는 correct reasoning에서 negative를 만들어 낸다.

**메커니즘.** softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이므로 sampled token을 push-down하면 다른 token으로 mass가 이동한다. [[negative-sample-reinforcement]]의 sign-resolved form(Eq. 7, 8)은 다음과 같다.

```
PSR: −∂L/∂z_v ∝  π_v(1 − π_v)   if v = y_t ;  −π_{y_t}·π_v   otherwise
NSR: −∂L/∂z_v ∝ −π_v(1 − π_v)   if v = y_t ;  +π_{y_t}·π_v   otherwise
```

계산 예제: 한 position의 세 token `π = (0.90, 0.07, 0.03)`에서 incorrect response 안 sampled token이 첫 token이다. NSR push-down은 `π(1−π) = 0.90 × 0.10 = 0.09`로 scale한다. 같은 token이 `π = 0.30`이면 `0.30 × 0.70 = 0.21`로 두 배 이상이다. mass는 현재 probability에 비례해 다른 두 token으로 간다. `0.90 × 0.07 = 0.063`, `0.90 × 0.03 = 0.027`, 즉 model이 이미 둔 7:3 비율이다. 그래서 논문은 NSR이 high-confidence pretrained structure를 지우지 않고 error를 suppress한다고 설명한다(§4.2, Interpretation).

**현재 practice와 측정된 효과 크기.** §3.2에 재현한 [[negative-sample-reinforcement]] Table 1에서 Qwen2.5-Math-7B/MATH의 negative-only signal은 full GRPO 76.3에 대해 pass@1 75.7을 내며, base pass@256 96.9를 유지한 유일한 setting이다. positive term도 inert하지 않다. λ = 0.1은 MATH pass@1에서 λ = 0보다 높다(76.6 대 75.7), λ = 1은 모든 k에서 더 나쁘다(App. E Table 4). 논문은 positive/negative signal 사이 gain percentage split을 보고하지 않으므로 이 chapter도 주장하지 않는다.

**반대 결과와 둘 다 성립하는 이유.** [[likelihood-displacement]]는 preference loss negative term이 반대 의미 response로 mass를 옮겨 Llama-3-8B-Instruct refusal rate를 74.4%에서 33.4%로 내렸다고 보고한다(§6.2). setting이 다르다. DPO pair는 흔히 같은 종류 response 두 개이고, 그 논문의 mechanism은 CHES score로 측정한 response embedding similarity에 의해 구동된다(§4.2.2, Definition 2). RLVR negative는 current policy rollout이고 reward는 rollout별 계산이다. 한 논문만 보고 negative가 언제나 safe 또는 harmful이라고 결론내리면 setting 하나를 과잉 일반화한 것이다.

**이 lab의 control.** negative term을 bound/weight한다(W-REINFORCE에서 λ < 1; ε_low는 tight하게 두고 ε_high를 올림, [[dapo]] §3.1). data를 on-policy로 유지한다(Option B는 하고 A는 하지 않음). positive NLL term으로 anchor한다(Option A toggle 1). CHES로 pair filter한다(Option A toggle 2). truncated completion을 mask해 length만으로 penalize하지 않는다([[dapo]] §3.4; [[trl-grpo]] config `mask_truncated_completions` L259–262).

**memo에 필요한 diagnostic.** chosen/rejected log-probability를 separately/absolute term으로, advantage sign별 clip fraction/gradient norm/entropy, step별 `frac_reward_zero_std`, pass@1과 k ≥ 64 pass@k, prompt set에 unanswerable/unsafe item이 있으면 abstention/refusal rate를 기록한다.

**generality에 미치는 효과.** 네 effect는 별도로 측정되고 available source에서 서로 다른 방향으로 움직이므로 구분한다.

1. **Coverage.** RLVR setting에서 retained coverage와 연관된 component는 negative gradient다. NSR은 MATH pass@256을 base 96.9로 유지했지만 PSR은 91.2로 낮췄다([[negative-sample-reinforcement]], Table 1).
2. **Over-refusal과 그 반대.** preference setting에서 rejected response negative gradient는 반대 의미 response로 mass를 옮겨 Llama-3-8B-Instruct training-set refusal rate를 74.4%→33.4%로 낮췄다([[likelihood-displacement]], §6.2). 여기 failure는 over-refusal이 아니라 under-refusal이다. §5.1 [[xstest]]는 safe prompt 250개와 unsafe contrast 200개를 separately 보고해 양쪽을 측정한다.
3. **Forgetting.** negative는 직접이 아니라 policy를 움직인 거리로 forgetting에 영향을 준다. RL/SFT forgetting은 base policy forward KL에 대해 하나의 curve에 놓이므로([[rls-razor]], §4), negative-heavy cell은 loss composition이 아니라 plot (b)의 KL axis에서 비교한다.
4. **Calibration과 hallucination.** 이 chapter source set에는 PSR, NSR, DPO negative term의 calibration/hallucination 측정이 없으므로 주장하지 않는다. 원하면 pass@k에서 추론하지 말고 measurement를 추가한다.

---

## 레시피

lab value가 budget 때문에 source value와 다르면 각각 차이를 명시한다.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 | Lab value | 차이의 이유 |
|---|---|---|---|---|---|---|---|---|---|
| All DPO runs unless noted | task별 | preference | β | 0.1 | arXiv:2305.18290v3 App. B | verified 2026-09-14 | “we did not meaningfully tune DPO's β” (§6.2) | 0.1 (middle cell) | 없음 |
| GPT-J SFT (CarperAI/openai_summarize_tldr_sft), TL;DR | 6B | preference | β | 0.5 | arXiv:2305.18290v3 App. B | verified 2026-09-14 | ablation 보고 없음 | 0.5 (high cell) | 없음 |
| GPT-2-large, IMDb | 명시 없음 | preference | β sweep | {0.05, 0.1, 1, 5} | arXiv:2305.18290v3 §6.1 | verified 2026-09-14 | Fig. 2 left, 22 run, 100 step마다 평가 | 0.05 (low cell) | β=1/5는 chat-model DPO 범위를 크게 벗어나며 high cell은 TL;DR row 사용 |
| All DPO runs | task별 | preference | optimizer; peak LR; warmup | RMSprop; 1e-6; linear 0 → 1e-6/150 step | arXiv:2305.18290v3 App. B | verified 2026-09-14 | ablation 없음 | same | 없음 |
| All DPO runs | task별 | preference | batch size | 64 (unit 명시 없음) | arXiv:2305.18290v3 App. B | verified 2026-09-14 | ablation 없음 | 32 pair | single-GPU memory; β와 batch size는 update 수로 상호작용하므로 deviation 기록 |
| Gemma-2B-IT, Llama-3-8B-Instruct | 2B / 8B | preference | filtered subset | lowest length-normalized CHES 5% | arXiv:2410.08847v4 §6.3 | verified 2026-09-14 | 15%까지 유사, 더 많으면 refusal drop(footnote 9) | 5% (toggle 2) | 없음 |
| Gemma-2B-IT (DPO) | 2B | preference | SFT-term coefficient | 1 | arXiv:2410.08847v4 App. I.3 | verified 2026-09-14 | grid {0.01,0.1,1}, 3 seed에서 highest mean training-set refusal | 1 (toggle 1) | 없음 |
| DeepSeekMath-RL | 7B | RL | policy LR; β; G; max length | 1e-6; 0.04; 64; 1024 | arXiv:2402.03300 §4.2 | verified 2026-09-14 | ablation 없음 | LR 1e-6; β=0.04; G=8; max 1024 | prompt당 rollout 8개에서 G=64는 budget 밖 |
| All Dr. GRPO runs | 1.5B–7B | RL | G; temperature; top-p/top-k; ε; epoch | 8;1.0;(1.0,−1);0.2;1 | arXiv:2503.20783v2 App. G Table 6 | verified 2026-09-14 | ablation 없음 | same | 없음 |
| All Dr. GRPO runs | 1.5B–7B | RL | KL coefficient | 0.0 | arXiv:2503.20783v2 App. G Table 6 | verified 2026-09-14 | ablation 없음 | 0.0 low KL cell | 없음 |
| All Dr. GRPO runs | 1.5B–7B | RL | max response length | 3000 token | arXiv:2503.20783v2 App. G Table 6 | verified 2026-09-14 | 4k Qwen2.5-Math context와 연결(App. B) | 1024 token | 1B–3B short-answer math; length-bias diagnostic은 적용 |
| Qwen2.5-Math-7B PSR/NSR/W-REINFORCE | 7B | RL | prompt batch; rollout; minibatch; LR; temp; KL | 1024;8;256;1e-6;1.0; PPO/GRPO 1e-3, PSR/NSR none | arXiv:2506.01347v2 §3.1, App. D.1 | verified 2026-09-15 | ablation 없음 | prompt batch 64; 나머지 same; KL=0.001 middle cell | one node에 맞게 batch 축소; matched step이므로 ablation에서 absolute batch 상쇄 |
| Qwen2.5-Math-7B | 7B | eval-gate | samples; temp; top-p; k | 256;0.6;0.95;1–256 | arXiv:2506.01347v2 §3.1 | verified 2026-09-15 | unbiased pass@k Eq.5 | 64 sample; same decode; k=1…64 | evaluation cost; Table 1 PSR–NSR ordering이 명확한 최소 k=64 |
| Qwen2.5-32B DAPO | 32B | RL | ε_low; ε_high |0.2;0.28| arXiv:2503.14476v2 §4.1 | verified 2026-09-15 | clip-higher AIME24 avg@32 36→38 | entropy-collapse repair cell만 | sweep axis가 아니며 바꾸면 KL axis confound |
| Qwen2.5-32B DAPO |32B|RL|dynamic sampling|0 < correct rollout < G만 유지|arXiv:2503.14476v2 §3.2 Eq.11|verified 2026-09-15|Table 1 42→50 AIME24 avg@32|`frac_reward_zero_std`로 log; repair cell만 enable|effective batch를 바꾸므로 mid-sweep enable은 matched step 파괴|
| Qwen2.5-32B |32B|RL|KL-Cov token fraction|10⁻⁴–10⁻³ token|arXiv:2505.22617 §4.5|verified 2026-09-14|7-benchmark avg 45.8→52.2|repair option으로 명명, run 안 함|custom loss 필요, one-axis lab 범위 밖|

**작은 general-purpose run의 시작점.** 1B–3B Option B는 Dr. GRPO normalization, G=8, temperature 1.0, top-p 1.0, clip ε=0.2, inner epoch 하나, constant LR 1e-6, 첫 cell KL coefficient 0.0을 쓴다. 이는 [[dr-grpo]] App. G Table 6에서 왔으며 해당 run은 8×A100의 1.5B–7B Qwen2.5 base와 response budget 3000 token이었다. short-answer math에서는 max response length를 1024로 낮춘다. 같은 scale Option A는 [[dpo]] App. B의 β=0.1, RMSprop, 150 warmup step 포함 LR 1e-6을 쓴다. 그 source의 가장 큰 model은 6B다.

---

## 일반화 관점

**(a) breadth를 높이는 것.** positive term을 down-weight하면서 negative term을 유지하는 방법이다. Qwen2.5-Math-7B에서 λ=0.1 W-REINFORCE는 base 96.9에 대해 MATH pass@256 96.7을 유지하고 최고 pass@1 76.6에 도달했지만 λ=1은 pass@256을 92.0으로 낮췄다([[negative-sample-reinforcement]], Table 1, App. E Table 4). prompt당 rollout을 더 sample하는 방법도 있다. fixed prompt batch에서 n을 8→32로 올리면 pass@1은 낮아도 220 step 뒤 pass@128은 더 높았다([[rlvr-beyond-base-model]], Figure 16). policy를 reference에 가깝게 유지하면 equal new-task accuracy에서 작은 forward KL RL cell이 더 많은 prior capability를 보존했다([[rls-razor]], §3, §4).

**(b) narrowing의 원인.** observed correct sequence를 반복 증폭하고 alternative를 suppress하는 positive-only reinforcement([[negative-sample-reinforcement]] Eq. 7)는 MATH pass@256을 96.9→91.2로 낮췄다. fixed configuration에서 더 긴 training도 원인이다. GRPO 150→450 step은 Omni-MATH-Train pass@1 26.1→42.5, pass@256 66.3→64.3, out-of-domain MATH500 pass@256 97.2→95.4였다([[rlvr-beyond-base-model]], Table 4). entropy collapse도 원인이다. 11 base model의 gradient step 2400개 중 첫 200개에서 entropy consumption 73%, performance gain 76%가 일어났고 fitted law `R = −a·exp(H) + b`는 `H = 0` ceiling을 뜻한다([[entropy-mechanism-llm-rl]], §2.3, §2.4 Eq.6). offline preference training의 displacement(§2.2)와 algorithm과 무관한 large distribution shift([[rls-razor]], §4)도 원인이다.

**(c) 이 stage의 측정법.** 서로 대체할 수 없는 instrument 세 개다: coverage는 target domain k ≥ 64 pass@k, target 밖 capability는 §5.1 held-out suite, sweep cell 비교 x-axis는 training prompt forward KL이다([[rls-razor]], §6 Table 1). controlled ParityMNIST에서 forward KL은 reverse KL R²=0.93 ± 0.01, total variation 0.80 ± 0.01, 모든 weight/activation-space distance ≤0.58 ± 0.02 대비 R²=0.96 ± 0.01이었고 LLM fit은 R²=0.71이다(§4 Figure 11). known error: MMLU-Pro prompt style variation 약 2 point, 최대 3.74([[mmlu-pro]] §6.3); IFEval strict/loose gap은 GPT-4 prompt/instruction level 2.41/1.80, PaLM 2 S 3.88/3.35 point([[ifeval]] Table 3); refusal classification에서 string matching은 Llama2.0 safe-prompt refusal 48.4%를 주지만 manual annotation은 38% + 21.6%였다([[xstest]], Table 2). reward가 Qwen2.5에서 작동해도 OLMo2에서 안 될 수 있으므로([[spurious-rewards-rlvr]], §3) model family 하나 결과는 그 family의 evidence다.

---

## 흔한 실수와 탐지 방법

| 실수 | 관측 증상 | 확인 |
|---|---|---|
| pass@1만 보고 | target metric 상승, coverage 진술 없음 | unbiased estimator로 k ∈ {1,…,64} pass@k를 계산하고 largest k에서 base와 비교 |
| rising margin으로 DPO run 수용 | `rewards/margins` 상승, `rewards/accuracies` 높음 | absolute `logps/chosen` log; 단조 하락이면 displacement([[likelihood-displacement]] §6.2) |
| implicit reward를 KL로 사용 | rollout 없는 run에 KL “보고” | fixed prompt set에서 `π_θ`, `π_ref`를 generate해 forward KL 계산([[rls-razor]] §6) |
| step count 다른 ablation cell 비교 | positive-only가 contribution rollout이 적어 epoch당 더 오래 학습 | epoch가 아니라 optimizer step 수 고정; PSR/NSR은 construction상 batch sample이 적음([[negative-sample-reinforcement]] §3.1) |
| KL axis를 length bias와 confound | 모든 KL cell에서 length 증가 | 전체 sweep에 `loss_type="dr_grpo"`, `scale_rewards="none"`; `"grpo"`는 deliberate reproduction에서만([[dr-grpo]] §3.1, [[trl-grpo]] L2548–2562) |
| effective batch 축소를 놓침 | reward mean/gradient noise 상승, 기여하지 않는 group 다수 | step마다 `frac_reward_zero_std` log([[trl-grpo]] L2150); all-correct prompt 수는 training 중 상승([[dapo]] §3.2 Fig.3b) |
| test하지 않은 verifier 신뢰 | training reward가 held-out scored accuracy보다 훨씬 높음 | held-out slice를 strict verifier로 score하고 stated tolerance 내 training reward agreement 요구 |
| before/after decoding config 변경 | retention delta가 training effect보다 큼 | temperature/top-p/max token/system prompt/few-shot format pin; CoT만으로 MMLU-Pro 최대 19.1 point 이동([[mmlu-pro]] §6.2) |
| one-family 결과 일반화 | Qwen-only run으로 reward design이 “works”라 보고 | second family에서 cell 하나 반복([[spurious-rewards-rlvr]] §3) |

---

## 이해 점검

1. DPO cell의 `rewards/margins`는 단조 상승하고 `rewards/accuracies`는 0.82인데 held-out safety refusal rate는 반감했다. implicit reward 정의를 써서 두 관측이 같은 checkpoint에서 어떻게 가능한지 설명하라.
2. target pass@1이 같은 RLVR cell 두 개 중 하나는 reference forward KL이 다른 하나의 두 배다. 어느 쪽이 held-out suite를 더 보존할지, evidence와 prediction이 실패하는 조건을 말하라.
3. NSR은 base-level pass@256을 유지하고 PSR은 낮춘다. Table을 되풀이하지 말고 [[negative-sample-reinforcement]] Eq. 7, 8에서 유도하라.
4. 같은 data SFT control을 빼면 [[rls-razor]] RL-versus-SFT Pareto 결과상 어떤 claim을 뒷받침할 수 없게 되는가?
5. 왜 ablation cell은 epoch가 아니라 optimizer step에서 match해야 하며 mismatch는 negative의 measured contribution에 무엇을 하는가?
6. before evaluation을 생략하고 published base score와 after만 비교하려 한다. [[mmlu-pro]] prompt robustness와 [[ifeval]] strict/loose gap으로 어떤 effect size가 측정 불가능해지는지 말하라.
7. entropy law `R = −a·exp(H) + b`는 서로 다른 advantage estimator를 쓰는 GRPO/RLOO/PRIME에 같은 curve로 fit한다([[entropy-mechanism-llm-rl]], §2.5, Figure 6). plateau를 algorithm 이름으로 설명하는 memo에 무슨 함의가 있는가?
8. [[negative-sample-reinforcement]]와 [[likelihood-displacement]]는 negative gradient의 other-behavior effect에 반대 결론을 낸다. 양쪽을 일관되게 만드는 setting difference 두 가지를 말하라.

---

## 연결

- **ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL** (dependency): 이 lab이 stage 하나에서 실행하는 stage-placement decision을 제공한다.
- **ch-39 — Offline Preference Optimization: DPO and Its Variants**: Option A objective와 variant.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO**: 여기 normalization choice를 포함한 Option B objective.
- **ch-43 — Entropy, Output Diversity, and KL Control in RL**: §4에 log하는 entropy/KL signal.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages**: negative-signal section이 적용하는 derivation.
- **ch-44 — Process Supervision and Verifiable Rewards**: Option B verifier construction.
- **ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split**: held-out split/forgetting report를 재사용하는 이전 lab.
- **ch-00 — What General Capability Means and How It Is Measured**: §5.1 retention suite가 instantiate하는 capability coverage map.
- **ch-46a — Lab: Small Agentic SFT-then-RL Run with a Generality Gate** (next): multi-turn agentic stage의 같은 three-axis deliverable.
- **ch-47 — Evaluation Harness and Suite Design for General Capability**: §5.1 ad-hoc suite를 versioned harness로 전환.
- **ch-54 — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments**: §4 sampler-versus-trainer KL 분석.

---

## 출처

- [[dpo]] — Eq. 7, implicit-reward identity, sign-split gradient, run β 값(App. B, §6.1).
- [[grpo]] — Eq. 3/Eq.4, group-normalized advantage, DeepSeekMath-RL 7B setting(§4.2).
- [[dr-grpo]] — Eq.3 length/difficulty bias, unbiased normalization, App. G Table 6 config.
- [[dapo]] — dynamic sampling/zero-advantage argument, clip-higher, token-level loss, AIME 2024 ablation.
- [[negative-sample-reinforcement]] — PSR/NSR/full/W-REINFORCE comparison(Table 1), logit gradient(Eq.7–8), λ sweep, unbiased pass@k estimator.
- [[likelihood-displacement]] — DPO refusal-rate collapse, preferred-log-probability change, CHES filtering.
- [[rls-razor]] — RL-versus-SFT retention frontier와 forgetting predictor forward KL.
- [[rlvr-beyond-base-model]] — training step별 pass@1/pass@256과 six RL algorithm, rollout-count ablation.
- [[spurious-rewards-rlvr]] — RLVR gain model-family dependence, cell 하나가 증명하는 범위 제한.
- [[entropy-mechanism-llm-rl]] — entropy–performance law, early entropy consumption, Clip-Cov/KL-Cov repair.
- [[mmlu-pro]] — knowledge/reasoning component, prompt robustness/CoT 수치.
- [[ifeval]] — instruction-following component, 네 metric, strict/loose trade-off.
- [[xstest]] — over-refusal component와 trade-off 양쪽에 대한 system prompt 효과.
- [[trl-grpo]] — loss normalizer, advantage scaling, `frac_reward_zero_std`, sign-split clip metric, k3 KL, `mask_truncated_completions`.
- [[openrlhf-dpo]] — `DPOLoss` 및 DPO trainer가 실제 log하는 metric set, §2.2 workaround 근거.
- [[openrlhf-ppo]] — `(loss, clip_ratio, ppo_kl, vllm_kl)` return과 sampler-versus-trainer KL.
- [[verl-grpo]] — DAPO dynamic sampling에 해당하는 group-filtering option.
