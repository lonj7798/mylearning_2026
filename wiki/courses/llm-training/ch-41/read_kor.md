# Chapter 41 — Reward Modeling: Bradley–Terry, 과도 최적화, Reward Model 일반화

> **핵심 통찰.** Bradley–Terry loss로 훈련한 reward model(RM)은 훈련받은 응답 분포에서의 점수 차이를 학습하며, 이에 맞춰 최적화한 policy는 그 분포 밖의 응답으로 이동한다. 6B “gold” RM을 사용한 합성 설정에서, √KL이 커짐에 따라 gold 점수는 먼저 상승했다가 하락한다. 더 큰 proxy RM은 peak를 바깥으로 옮기지만 없애지는 못한다([[reward-model-overoptimization]] §3.2, Fig. 1, Fig. 3). 정적인 RM benchmark의 정확도는 RM이 산출하는 policy의 약한 예측변수다. 통제된 DPO 실험에서 Nemotron-4-340B-Reward(RewardBench 92.0)는 Arena 점수 1172인 policy를 만들었는데, 이는 훈련하지 않은 base 모델(1178)의 confidence interval 안이다. 반면 Athene-RM-8B는 1209를 만들었다([[ppe-reward-model-eval]] Table 3; [[nemotron-4-synthetic]] Table 4).
>
> **지침.** RM을 RL 또는 best-of-n으로 최적화할 예정이면 policy 자신이 생성한 sample 및 held-out domain(PPE 스타일 correctness set, RM-Bench 스타일 제어 pair)에서 측정하고, √KL의 함수로 별도 evaluator를 추적하라. 고정 pair에서의 benchmark 정확도는 downstream 결과를 예측하지 못하기 때문이다([[ppe-reward-model-eval]] §7; [[rm-bench]] §5.2). 하나의 pretrained checkpoint에서 fine-tune한 여러 RM이 비용상 가능하면 conservative ensemble(worst-case 또는 uncertainty-weighted) 또는 weight average를 사용하라. 둘 다 해당 연구에서 과도 최적화를 줄였기 때문이다([[reward-ensembling]] §5; [[warm-weight-averaged-reward-models]] §5.2). RM이 pretraining checkpoint를 공유한다면 어떤 집계도 제거하지 못하는 shared error를 예상하라([[helping-or-herding]] Abstract). 작업에 확인 가능한 답이 있으면 current policy 자신이 만든 정답과 오답 sample로 verifier를 훈련하고 policy가 바뀔 때 갱신하라([[training-verifiers-to-solve-math-word-problems]] §4.2; [[v-star]] §3).

---

## 이 장이 범용 모델에 중요한 이유

이 장의 pipeline 위치는 preference data(ch-15) → reward model(이 장) → policy optimization(ch-37~ch-40)이다. RM은 pairwise human 또는 AI judgment를 optimizer가 최대화할 수 있는 scalar로 바꾼다. 범용 모델에서 RM은 prompt 분포가 포괄하는 모든 domain의 응답을 점수화해야 하며, checkable answer가 없는 domain도 포함한다. 또한 policy가 바뀐 뒤에도 계속 올바르게 점수화해야 한다.

이 위치에서 측정 가능한 세 문제가 따른다.

1. **분포 이동.** RM은 이전 모델의 응답으로 훈련되고 RL은 policy를 새 응답으로 이동시킨다. Gao et al.은 이 실패를 extremal Goodhart이라고 부르며 gold 점수 하락의 대부분을 초래할 것으로 예상한다([[reward-model-overoptimization]] §4.2.2, Interpretation).
2. **고르지 않은 domain 정확도.** RM은 chat에서는 정확하지만 math 또는 code에서는 무작위에 가까울 수 있다. RM-Bench에서 최고 점수 RM의 math hard accuracy는 28.4%, code에서는 30.7%다([[rm-bench]] §4.1, Tables 14-15). 이러한 RM으로 훈련된 policy는 바로 그 domain에서 잘못된 신호를 받는다.
3. **측정.** RM 품질은 보통 고정 preference pair 정확도로 보고되지만, 중요한 양은 RM으로 훈련한 policy의 품질이다. 둘은 부분적으로만 상관한다([[ppe-reward-model-eval]] §7; [[rm-bench]] §5.2).

아래 각 절은 이 문제 중 하나를 다룬다. Loss 및 그것이 식별하는 것(§1), 과도 최적화(§2), robustness 방법(§3), out-of-distribution 평가 및 generalist RM(§4), 불완전한 RM도 전이되는 조건인 policy coverage(§5)다.

---

## §1 Bradley–Terry reward model

### §1.1 정의와 해결하는 문제

Bradley–Terry(BT) reward model은 (prompt, response)마다 scalar 점수 하나를 출력하는 network이며, 두 점수의 차이가 labeler가 선호한 응답을 예측하도록 훈련한다. 이는 사람이 신뢰할 수 있는 상대적 judgment를 주지만 보정된 절대 점수는 주지 않는 반면, RL은 응답마다 scalar를 필요로 하는 문제를 해결한다.

### §1.2 메커니즘

1. `(x, y_w, y_l)` triple을 수집한다. prompt, 선호 응답, 비선호 응답이다(ch-15는 protocol과 agreement를 다룬다).
2. Scorer를 만든다. pretrained 또는 SFT language model에서 unembedding layer를 제거하고 scalar를 출력하는 linear head를 붙인다. InstructGPT는 SFT 모델의 마지막 unembedding layer를 제거한다([[rlhf-instructgpt]] §3.5). Stiennon et al.은 무작위 초기화 linear head를 추가한다([[bradley-terry-rm]], Stiennon §3.4).
3. Pair의 두 응답에 점수를 매기고 점수 차이에 logistic loss를 적용한다.
4. RL 전에 reference set의 평균이 0이 되도록 상수만큼 점수를 이동한다(§1.4).

### §1.3 수식

```
p*(y1 ≻ y2 | x) = exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2))) = σ(r*(x,y1) − r*(x,y2))        (DPO §3 Eq. 1)
L_R(r_φ) = − E_{(x, y_w, y_l) ~ D} [ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]                          (DPO §3 Eq. 2)
```

`x`는 prompt, `y1`, `y2`는 두 응답, `r*`는 잠재 reward, `r_φ`는 parameter `φ`를 가진 RM, `y_w`와 `y_l`은 선호 및 비선호 응답, `D`는 comparison dataset, `σ(z) = 1/(1+e^{−z})`다. Loss는 label 1 및 logit `r_φ(x,y_w) − r_φ(x,y_l)`을 가진 binary cross-entropy다([[bradley-terry-rm]]에서 유도).

각 점수에 대한 미분값은 다음과 같다.

```
∂L/∂r_φ(x,y_w) = −(1 − σ(Δ)),     ∂L/∂r_φ(x,y_l) = +(1 − σ(Δ)),     Δ = r_φ(x,y_w) − r_φ(x,y_l)
```

### §1.4 계산 예시와 식별 가능성

`r(x,y_w) = 1.2`, `r(x,y_l) = 0.4`라고 하자. 그러면 `Δ = 0.8`, `σ(0.8) = 0.690`, loss는 `−ln 0.690 = 0.371`이고, 각 점수는 크기 `1 − 0.690 = 0.310`의 gradient를 받는다. 두 점수에 모두 5를 더해도 `Δ`는 0.8이므로 loss와 gradient는 변하지 않는다.

이것이 식별 가능성 속성이다. Prompt의 임의 함수 `f(x)`만큼 다른 reward function은 동일한 BT preference distribution을 유도한다(DPO §5.1 Definition 1, Lemma 1, [[bradley-terry-rm]]을 통해). 두 결과가 따른다.

- **점수는 prompt 사이에서 비교할 수 없다.** 한 prompt에서 3.0, 다른 prompt에서 1.0이라는 점수는 첫 응답이 더 낫다는 뜻이 아니다. Stiennon et al.은 reference summary의 평균 점수가 0이 되도록 정규화한다. InstructGPT는 RL 전에 labeler demonstration의 평균이 0이 되도록 bias를 더한다. 이는 “RM loss is invariant to shifts in reward”이기 때문이다([[bradley-terry-rm]], Stiennon §3.4; InstructGPT §3.5).
- **RM 전반의 순서 통계에는 공유 offset이 필요하다.** RM A와 RM B가 다른 offset `C(x)`을 고르면 그 점수의 minimum 또는 median은 의미가 없다. Eisenstein et al.은 작은 η > 0에 대해 각 pair 두 reward의 합을 0 쪽으로 regularize하는 항 `η · E[(r(x,y+) + r(x,y−))²]`을 추가한다. 이는 ensemble 전에 offset을 고정한다([[helping-or-herding]] §2.1 Eq. 2).

### §1.5 K-way ranking: Plackett–Luce 대 all-pairs

Labeler가 K개의 응답을 순위화할 때 두 종류의 목적함수를 사용한다.

**Plackett–Luce**는 전체 ranking τ를 남은 항목에서 연속적으로 선택하는 과정으로 모델링한다.

```
p*(τ | y1..yK, x) = ∏_{k=1}^{K}  exp(r*(x, y_τ(k))) / Σ_{j=k}^{K} exp(r*(x, y_τ(j)))            (DPO App. A.3 Eq. 18)
```

`τ(k)`는 k번째로 rank된 응답의 index다. 각 factor는 아직 선택하지 않은 항목의 softmax이므로 log-likelihood는 `log σ` 항의 합이 아니라 log-softmax 항의 합이다. K = 2이면 곱은 BT로 줄어든다([[dpo]] App. A.3).

**All-pairs BT(InstructGPT)**는 ranking을 C(K,2)개 pairwise comparison으로 펼치고, 그들에 대한 BT loss 평균을 사용한다. `loss(θ) = −(1/C(K,2)) E[log σ(r_θ(x,y_w) − r_θ(x,y_l))]`([[rlhf-instructgpt]] §3.5 Eq. 1).

A ≻ B ≻ C이고 점수가 (2, 1, 0)인 세 응답의 계산 예시는 다음과 같다.

| 목적함수 | 계산 | Negative log-likelihood |
|---|---|---|
| Plackett–Luce | 첫 선택 `e²/(e²+e¹+e⁰) = 7.389/11.107 = 0.665`; 두 번째 선택 `e¹/(e¹+e⁰) = 0.731`; 곱 0.486 | `0.408 + 0.313 = 0.721` |
| All-pairs BT, 평균 | pair (A,B): `−ln σ(1) = 0.313`; (A,C): `−ln σ(2) = 0.127`; (B,C): `0.313` | 평균 `0.251` |

두 목적함수는 pair에 서로 다른 가중치를 준다. Plackett–Luce는 top choice를 다른 모든 항목에 한 번에 결합한다. All-pairs BT는 세 comparison을 forward pass를 공유하는 별도 항으로 다룬다. InstructGPT는 모든 comparison을 한 dataset에 섞으면 RM이 한 pass 안에 overfit했다고 보고했으며, 한 prompt의 모든 C(K,2) comparison을 한 batch element에 넣어 이를 고쳤다([[rlhf-instructgpt]] §3.5).

### §1.6 증거

- **RM 및 데이터의 규모(결과, 단일 연구).** Stiennon et al.은 8k~64k comparison으로 160M~13B parameter의 RM 7개를 훈련했다. Data를 두 배로 늘리면 validation accuracy가 약 1.1%, model 크기를 두 배로 늘리면 약 1.8% 상승했다([[bradley-terry-rm]], Stiennon §4.3, Fig. 6).
- **전이(결과, 단일 연구).** TL;DR 훈련 RM은 CNN/DM에서 labeler와 62.4%(1.3B), 66.5%(6.7B) 일치한다. Inter-labeler agreement는 66.9%다(Stiennon §4.3).
- **선호 강도.** Llama 2 helpfulness RM은 “significantly better” pair에서 79.1%, “negligibly better / unsure” pair에서 54.5% 정확도다([[bradley-terry-rm]], Llama 2 App. A.3.3 Table 28). Rating strength에 따라 커지는 margin 항 `m(r)`은 평균 정확도를 62.5(margin 없음)에서 63.0(작은 margin)으로 바꿨다([[llama-2-recipe]], Table 28). Llama 3는 data scaling 뒤 개선이 줄어든다는 이유로 margin을 제거했다([[llama-3-recipe]], §4.1.2).

### §1.7 조건과 한계

- **비추이적 선호.** 응답당 하나의 scalar는 이를 표현하지 못하지만 pairwise preference model은 표현할 수 있다([[bradley-terry-rm]]을 통한 NLHF §3.1).
- **결정적 label.** `p*(y ≻ y′) = 1`이면 BT는 `r(y) − r(y′) → +∞`를 요구하며, 최적 KL-regularized policy는 어떤 KL strength에서든 `π*(y′) = 0`으로 둔다([[bradley-terry-rm]]을 통한 IPO §4.2). 이는 작은 dataset에서 RM을 loss 0까지 훈련하지 말아야 하는 이유다.
- **길이.** Stiennon의 6.7B RM은 shortening edit을 62.6% 선호하며 사람은 76.4% 선호한다(Stiennon §4.3). 따라서 그 설정에서 RM은 더 짧아져 개선된 응답을 사람보다 덜 보상한다.

**범용 모델에 대한 함의.** BT loss는 prompt 내부 및 훈련 분포 내부의 점수 차이만 제약한다. Loss에는 RM이 보지 못한 응답의 점수를 제약하는 것이 없으며, RL은 바로 그곳으로 policy를 보낸다. 이 장의 나머지는 이 간극을 다룬다.

---
## §2 과도 최적화: proxy 점수 대 gold 점수

### §2.1 정의와 측정 가능한 문제

과도 최적화는 추가 최적화가 RM(proxy) 점수는 올리지만 응답의 진정한 품질은 낮추는 체제다. 이를 측정하려면 두 번째로 신뢰할 수 있는 scorer가 필요하다. Gao et al.은 사람을 고정된 6B “gold” RM으로 대체하고, 이 모델로 100,000 comparison에 label을 붙인다(10%는 held out). 이 label로 3M~3B parameter proxy RM을 훈련하고, 각 proxy에 맞춰 1.2B policy를 최적화한 뒤 그 결과 sample을 gold RM으로 점수화한다([[reward-model-overoptimization]] §2.1).

### §2.2 메커니즘과 수식

x축은 초기 policy로부터 KL divergence의 제곱근이다.

```
d := √( D_KL(π ‖ π_init) )
R_bon(d) = d · (α_bon − β_bon · d)            best-of-n
R_RL(d)  = d · (α_RL  − β_RL · log d)          PPO                                  (Gao §1)
KL_bon(n) = log n − (n − 1)/n                                                        (Gao §2, from Stiennon App. G.3)
```

`π`는 최적화한 policy, `π_init`은 초기 SFT policy, `R`은 `R(0) = 0`이 되도록 recentre한 gold 점수, `α`, `β`는 적합 coefficient, `n`은 best-of-n의 sample 수다. RL 형태는 `d = 0`에서 무한한 기울기를 가지므로 origin 근처에서는 성립하지 않을 가능성이 크다(footnote 1).

`dR/dd = 0`으로 두면 peak를 얻는다([[reward-model-overoptimization]]에서 유도했지만 논문에는 수식으로 인쇄되지 않음).

```
d*_bon = α_bon / (2 β_bon)            d*_RL = exp(α_RL / β_RL − 1)
```

### §2.3 계산 예시

**Best-of-n의 KL.** n = 1,000이면 `KL = ln 1000 − 999/1000 = 6.908 − 0.999 = 5.909` nats이고 `d = 2.43`이다. n = 60,000이면 `KL = 11.002 − 1.000 = 10.00` nats이고 `d = 3.16`이다. 이는 fitting 및 validation 실행의 논문 수치 “KL ≈ 6 nats” 및 “KL ≈ 10 nats”와 일치한다(§3.1). Best-of-n의 KL은 `log n`에 따라 증가하므로 `d`의 추가 단위 하나마다 지수적으로 더 많은 sample이 든다.

**Peak 위치(illustrative coefficient, 논문의 값 아님).** 논문은 α와 β를 plot하지만 인쇄하지 않는다. Fig. 3은 proxy RM 크기가 커질수록 α_bon은 증가하고 β_bon과 β_RL은 감소하며, α_RL은 크기 전반에서 고정한다고 보인다(§3.2, Fig. 3). 두 illustrative proxy RM을 보자.

| Proxy RM(illustrative) | α_bon | β_bon | d* = α/(2β) | peak에서 KL = d*² | peak에서 R = d*(α − βd*) |
|---|---|---|---|---|---|
| smaller | 0.50 | 0.12 | 2.08 | 4.34 nats | 0.52 |
| larger | 0.65 | 0.09 | 3.61 | 13.04 nats | 1.17 |

RL 형태에서 α_RL = 0.5를 고정하면 β_RL을 0.15에서 0.12로 낮출 때 `d*_RL = exp(α/β − 1)`는 10.3에서 23.7로 이동한다. RL peak는 α/β에 지수적으로 민감하므로, 짧은 실행에서 best-of-n peak보다 예측하기 어려운 이유 중 하나다.

[figures/rm-overopt.html](figures/rm-overopt.html)에서는 어느 functional form이든 α와 β를 설정하고 peak 위치를 보며 best-of-n sample 수를 `d`로 변환할 수 있다. 두 번째 panel은 §1 및 negatives section에서 쓰는 BT loss와 gradient를 보인다.

### §2.4 증거

- **Proxy RM 크기(결과, 단일 연구).** Coefficient는 proxy RM 크기에 따라 매끄럽게 변한다. 더 큰 RM은 더 작은 β를 가지며, 저자들은 이를 improved robustness로 해석한다(§3.2, §4.2.2, Fig. 3).
- **RM 데이터(결과, 단일 연구).** Data sweep(RM 크기 12M 고정)에서 저자들은 모든 RM 크기에 대해 약 2,000 comparison 미만의 데이터는 chance에 가까운 loss보다 개선이 거의 없다고 보고한다(§3.3, Fig. 6). SGD step 수를 고정할 때 1회가 아닌 4 epoch는 gold 점수를 바꾸지 않았지만, 데이터 4배를 1 epoch 사용하면 더 잘 작동했다. Footnote는 두 데이터 양을 명시하지 않는다(§3.3, footnote 7).
- **Policy 크기(결과, 단일 연구).** 6B policy는 1.2B policy보다 최적화 이득이 작지만 거의 같은 KL에서 peak에 도달하고 거의 같은 proxy–gold gap을 보인다(§3.4, Fig. 7). 논문은 더 큰 policy가 proxy를 더 exploit한다고 보고하지 않는다.
- **KL penalty(결과, 단일 연구; 저자들은 hyperparameter-sensitive라고 부름).** 0이 아닌 KL penalty는 early stopping처럼 행동했으며 gold-score-versus-KL frontier를 올리지 못했다(§3.6, Fig. 9).
- **RL 대 best-of-n.** 저자들은 optimization과 over-optimization 모두에서 RL이 best-of-n보다 훨씬 많은 KL을 소비한다고 보고하지만 비율은 주지 않는다. 따라서 KL 거리는 두 방법의 최적화 양을 비교할 수 없다고 결론낸다(§3.5).
- **Verifier에 대한 test-time search(결과, 단일 연구).** 6B GSM8K verifier에서 ranked completion 수가 400까지 늘면 solve rate가 개선되다가 그 뒤 하락한다. 저자들은 search가 “adversarial solutions that fool the verifier”를 찾기 때문이라고 본다([[training-verifiers-to-solve-math-word-problems]] §5.1, Fig. 7a).

### §2.5 두 항의 해석

Gao et al.은 α를 regressional Goodhart으로 해석한다. Proxy는 independent noise를 더한 true score이므로 selection은 부분적으로 noise를 선택한다. β는 extremal Goodhart이다. 최적화된 sample이 RM 훈련 분포를 벗어나 proxy–gold 관계가 약해지는 현상이다(§4.2.1-4.2.2, Interpretation). 단순화 가정 아래, 재훈련한 RM을 사용해 최적화를 k round로 나누면 gold 점수에 `β_RL · d · log k`를 더하고 α 항은 바꾸지 않는다(§4.3, Interpretation).

### §2.6 조건과 한계

InstructGPT instruction 환경만 시험했다. Gold RM도 모델이므로 이 설정은 label과 human intent 사이의 gap을 포착하지 않으며 adversarial Goodhart도 모델링하지 않는다(§4.2.4, §4.5). Proxy 점수에 대해서는 만족스러운 fit을 찾지 못했다(§3.1).

**범용 모델에 대한 함의.** Peak는 proxy RM, 그 데이터, optimizer의 속성이다. 여러 domain에서 훈련하는 범용 policy는 domain마다 서로 다른 `d`에서 peak에 도달한다. 그러므로 aggregate evaluator에서 고른 단일 stop point는 RM이 가장 약한 domain에서는 이미 peak를 지난 지점일 수 있다.

---
## §3 Reward 신호를 더 견고하게 만드는 방법

### §3.1 보수적 집계를 사용하는 prediction ensemble

**정의.** k개의 RM을 훈련하고 응답별 점수를 결합한다. Coste et al.은 세 결합을 연구한다([[reward-ensembling]] §3, arXiv:2310.02743v2).

```
R_mean(q,a) = (1/k) Σ_i R_i(q,a)                                                   (Eq. 3)
R_WCO(q,a)  = min_i R_i(q,a)                                                        (Eq. 4)
R_UWO(q,a)  = (1/k) Σ_i R_i(q,a) − λ · (1/k) Σ_i ( R_i(q,a) − (1/k) Σ_i R_i(q,a) )²     (Eq. 5)
```

`q`는 prompt, `a`는 응답, `R_i`는 i번째 RM, `λ`는 ensemble 내부 분산의 가중치다. WCO는 worst-case optimization, UWO는 uncertainty-weighted optimization이다. 저자들은 mean optimization은 보수적이지 않다고 지적한다. 과대평가하는 member 하나도 exploit될 수 있기 때문이다(§3).

**계산 예시(λ = 0.5).** 점수 (1.0, 1.2, 2.6)은 평균 1.60, 분산 0.507, UWO `1.60 − 0.25 = 1.35`, WCO 1.0이다. 점수 (1.0, 1.1, 1.2)는 평균 1.10, 분산 0.0067, UWO 1.097, WCO 1.0이다. 한 member만 훨씬 높은 응답은 UWO에서 0.25를 잃지만, member가 동의하는 응답은 0.003을 잃는다.

**설정.** Pythia 1.4B policy, 7M·44M·1.3B proxy RM, gold로 AlpacaFarm 7B human-preference RM, 46k sample로 고정한 RM training set(AlpacaFarm instruction과 각각 두 응답으로 만든 preference pair), random seed(head 초기화와 data order)만 다른 RM, 선택적 25% label noise, 5개 ensemble이다(§4.2-4.3).

**증거(결과, 단일 연구).** Best-of-n에서 ensemble은 평균 single RM보다 label noise 없이 최종 gold 성능을 약 30%까지, 25% label noise에서는 약 75%까지 개선한다. WCO와 UWO는 n = 12,500(≈ 8.4 nats)까지 과도 최적화를 보이지 않았지만, mean optimization은 noisy label에서 과도 최적화했다(§5, §5.1, Fig. 3). PPO에서 WCO와 UWO는 KL penalty 없이는 과도 최적화를 줄이지만 없애지 못한다. KL penalty 0.01에서는 주목할 성능 손실 없이 이를 막지만 single RM은 0.2 penalty가 필요하고 성능을 잃는다(§5.2, Figs. 4-6). Member 4개와 5개의 성능은 비슷하며 3개에서 4개로 갈 때 gap이 있다(§5.4, Fig. 11).

**한계.** 환경 하나, RM refresh 없는 offline RLHF, 합성 gold label뿐이다(§6).

### §3.2 Ensemble member 사이의 공유 오류

Eisenstein et al.은 TL;DR, Anthropic Helpfulness, XSum/NLI에서 5개 pretraining seed × 5개 fine-tuning seed(작업과 규모당 RM 25개)로 T5 RM을 훈련한다([[helping-or-herding]] §2.3). 보고 결과(Abstract; 결과, 단일 연구)는 다음과 같다.

1. In-distribution accuracy가 비슷한 RM은 aligned-policy 출력에 서로 다른 reward를 부여한다(underspecification).
2. 하나의 RM에 맞춰 align해도 같은 data로 훈련한 다른 RM이 측정하는 reward는 개선되지 않는다.
3. Member가 fine-tuning seed만 다른 ensemble보다 pretraining seed가 다른 ensemble이 더 잘 일반화하며, 둘 다 single RM보다 낫다.
4. 모든 member가 오류 패턴을 공유하면 pretrain ensemble도 reward hacking을 제거하지 못한다. Factuality에 맞춰 튜닝한 summarization policy는 너무 짧아졌고 summary quality에 맞춰 튜닝한 것은 너무 장황해졌다. Helpfulness에 맞춘 assistant policy는 정형화된 답변 형식을 과도하게 썼다(§1).

Coste et al.의 ensemble은 fine-tuning seed만 다르므로([[reward-ensembling]] §4.3), Eisenstein et al.의 더 약한 ensemble 유형에 해당한다.

### §3.3 가중치 평균 reward model(WARM)

**정의.** 같은 SFT checkpoint에서 서로 다른 hyperparameter와 data order로 M개 RM을 fine-tuning한 다음, 가중치를 평균한다. `φ_WARM = (1/M) Σ_i φ_i`([[warm-weight-averaged-reward-models]] §3.1). Inference 비용은 RM 하나다.

**메커니즘.** 공유 pretraining에서 fine-tuning한 가중치는 선형적으로 mode connected 상태를 유지하므로, 평균 모델은 OOD pair에서 member accuracy의 interpolation만큼은 정확하다(Observation 1, Eq. 2). 논문의 bag-of-features model에서 prediction ensemble은 feature j에 `p_j`로 가중하고 weight average는 `p_j²`로 가중한다. `p_j`는 한 실행이 해당 feature를 학습할 확률이다(§4.3, Eqs. 4-5). 90% 실행이 학습한 feature는 가중치 0.81을 유지하고, 10% 실행이 학습한 feature는 0.01을 유지한다(card에서 유도). Memorized label noise를 포함하는 실행별 feature는 상대 가중치를 잃는다.

**증거(결과, 단일 연구).** AI label과 PaLM-XXS RM을 쓴 TL;DR에서 WARM(M = 6)으로 훈련한 policy는 best single RM으로 훈련한 policy를 상대로 79.4% oracle 승률을 가진다(§5.2, Fig. 9(c)). 25% swapped label에서 weight average는 prediction ensemble보다 corrupt training pair를 적게 맞추며 OOD test pair에서 더 낫다(Observation 3, Figs. 4-5).

**한계.** 모든 member가 길이 같은 동일 spurious cue에 의존하면 WARM도 그것을 유지한다(§6). 작업 하나와 AI label만 사용했다(§5). Weight average는 uncertainty estimate가 아니다. Disagreement가 신호로 필요하면 prediction ensemble이 여전히 선택지다(§6).

### §3.4 Pairwise ranker(PairRM)

**정의.** Pairwise ranker는 각 응답을 단독으로 점수화하는 대신 prompt와 두 candidate를 하나의 시퀀스로 encode하여 어느 쪽이 더 나은지 예측한다([[pairrm]] §3.1-3.2).

**메커니즘.** 입력은 `<s><source> x </s> <candidate1> y_i </s> <candidate2> y_j </s>`다. Special-token embedding의 MLP head가 `s^i`, `s^j`를 만들며 pair 점수는 `s_ij = s^i − s^j`다(§3.3, App. A). 훈련 중 pair 안의 candidate 순서를 shuffle한다. 모든 pair를 집계하는 MaxLogits는 O(N²) comparison이 필요하고 bubble-sort pass 한 번은 N − 1개가 필요하다(§3.3).

**증거.** MixInstruct(N = 11 candidate)에서 PairRanker의 selection은 평균 GPT-Rank 3.20이며, best single LLM 3.90 및 pointwise SummaReranker 3.66과 대비된다(Table 2). 공개된 0.4B PairRM checkpoint는 Auto-J pairwise에서 UltraRM-13B의 59.85에 비해 59.05, HHH-Alignment에서 83.71에 비해 84.62를 기록한다(model card).

**한계.** Pairwise ranker는 pointwise reward를 주지 않으므로 PPO reward로 직접 쓸 수 없다. Reranking과 pair construction에 맞는다.

### §3.5 Generative reward model 및 직접 LLM scoring

**GenRM(Mahan et al.).** LLM은 선택적으로 rationale 뒤에 preferred-answer token을 출력하도록 훈련한다([[generative-reward-models]] §4). STaR-DPO에서 chosen output은 올바른 verdict에 이르는 rationale이고 rejected output은 잘못된 verdict에 이르는 rationale이다(Eq. 9). UltraFeedback으로 훈련한 Llama-3.1-8B-Instruct에서 STaR-DPO는 in-distribution 73.9%(BT, PairRM, GenRM은 “around 73–74%”) 및 RewardBench 81.9%에 도달하며, reasoning 없는 GenRM은 78.9%다(§5.1). UltraInteract로 훈련하면 명시적 RM은 in-distribution 약 94%에 도달하지만 BT RM은 RewardBench Reasoning에서 무작위보다 낮고 STaR-DPO는 87.2%다(§5.2). 논문은 judge를 RL reward로 사용하지 않는다(§7).

**d-RLAIF(Lee et al.).** Off-the-shelf LLM에 응답을 1~10으로 평가하도록 prompt한다. 각 score token의 likelihood를 distribution으로 정규화하고 reward를 기대 점수 `s(y|x) = Σ_{i=1}^{10} i · P(i | y, x)`로 계산하여 [−1, 1]로 rescale한 뒤, 훈련한 RM 없이 RL에 직접 사용한다([[rlaif-scaling]] §2.2.2). 요약에서 사람 rater는 d-RLAIF를 SFT보다 74% 선호했는데, distilled RM을 쓴 같은 크기 RLAIF는 68%였으며, d-RLAIF를 같은 크기 RLAIF보다 60% 선호했다(Table 1). 저자들은 AI labeling이 human annotation보다 10× 이상 저렴하다고 추정한다(§4.1, App. L). Scorer가 고정 응답 집합에서 훈련되지 않으므로 d-RLAIF는 RM staleness를 피하지만, 그 오류는 prompted judge의 오류가 된다(ch-42, ch-49).

### §3.6 Multi-attribute regression reward model

Nemotron-4-340B-Reward는 Nemotron-4-340B-Base의 마지막 softmax layer를 다섯 HelpSteer attribute(Helpfulness, Correctness, Coherence, Complexity, Verbosity)에 대한 linear projection으로 대체한다. 10K HelpSteer2 예시로 훈련하고 inference에서 weighted sum으로 결합한다([[nemotron-4-synthetic]] §3.1). Report는 ablation 없이 multi-attribute regression이 pairwise ranking model보다 helpfulness와 길이 같은 artifact를 더 잘 분리한다고 말한다(Interpretation). Attribute weight, learning rate, epoch, loss는 보고하지 않는다([[nemotron-4-synthetic-recipe]], §3.1). Preference pair judge로 사용하면 LLM-as-judge 0.54에 비해 Chat-Hard accuracy 0.87에 이르렀다(§3.2.3).

RM-Bench는 correctness와 verbosity head가 분리되는지 시험한다. Safety domain에서는 chosen 및 rejected 응답이 correctness에서 분리되지만, math와 code에서는 겹친다([[rm-bench]] §4.3, Fig. 3).

**범용 모델에 대한 함의.** §3의 각 방법은 오류 원천 하나를 줄인다. Ensemble과 WARM은 seed별 및 label-noise 오류를 줄이고, pairwise 및 generative RM은 응답을 고립해 점수화하는 오류를 줄이며, multi-attribute head는 trade-off를 드러낸다. 이 방법 중 어느 것도 shared pretraining과 shared preference data에서 오는 모든 member 공유 오류를 제거하지 못한다.

---
## §4 Distribution 밖 reward model 평가

### §4.1 RewardBench

**정의.** Chosen 응답에 더 높은 점수를 부여할 때 RM이 정답인 prompt–chosen–rejected trio benchmark다([[rewardbench]] §4.2). Section과 크기는 Chat 358, Chat Hard 456(MT-Bench close rating 및 LLMBar adversarial pair), Safety 740, Reasoning 1,431(PRM800K의 사람 대 buggy LLM 답, 6개 언어의 HumanEvalPack 정답 대 buggy code), Prior Sets 17.2k(Anthropic Helpful, HHH, SHP, Summarize)이며 Prior Sets는 최종 점수에서 0.5 가중치다(Table 1; §4.2 footnote 4).

**한계.** 많은 chosen 및 rejected 응답이 서로 다른 모델에서 왔으며, prior-set test data는 inter-annotator disagreement 때문에 accuracy ceiling이 60~70%다(§1). RM-Bench와 PPE 모두 강도가 다른 모델의 pair는 RM이 style과 source cue로 성공하게 한다고 주장한다([[rm-bench]] §1; [[ppe-reward-model-eval]] §3).

### §4.2 RM-Bench: 미세한 오류와 스타일 제어

**메커니즘.**

1. 같은 모델(gpt-4o)로 chosen 및 rejected 응답을 생성한다. Many-shot jailbreak prompt로 chat 답에 사실 오류를 주입하고, unit test와 answer로 code 및 math의 정답과 오답 sample을 고른다([[rm-bench]] §3.1-3.2).
2. 모든 응답을 세 style로 다시 쓴다. 간결한 `y∅`, 상세한 plain text `y^L`, Markdown이 포함된 상세 `y^{L,M}`다(§3.4).
3. 3 × 3 chosen-style × rejected-style 조합을 모두 점수화한다. Easy accuracy는 하삼각형 평균(chosen의 style이 더 많음), Normal은 대각선, Hard는 상삼각형 평균(rejected의 style이 더 많음)이다(§3.5).

**계산 예시.** Chat에서 FsfairX-LLaMA3-RM-v0.1에 대해 인쇄된 Style-Substance matrix(Fig. 2)는 행이 chosen style(∅, L, L+M), 열이 rejected style이다.

```
            y_r∅      y_rL     y_rL,M
y_c∅       83.61      3.83      2.19
y_cL       99.45     66.12     49.73
y_cL,M    100.00     80.33     66.67
```

Hard = (3.83 + 2.19 + 49.73)/3 = 18.58%. Normal = (83.61 + 66.12 + 66.67)/3 = 72.13%. Easy = (99.45 + 100.00 + 80.33)/3 = 93.26%다. 오답이 더 길고 형식화되어 있으면 이 RM은 chat pair의 80% 이상에서 그것을 선호한다.

**증거.** Top RM(Skywork-Reward-Llama-3.1-8B)은 평균 70.1%, hard accuracy 46.6%다. Nemotron-340B-Reward는 평균 69.5%, hard 56.1%다(Table 3). 네 Tülu-v2.5 RM과 그 PPO policy를 사용했을 때 RM-Bench 점수는 정규화된 downstream policy 성능과 Pearson r = 0.55(p = 0.07)로 상관했으며, RewardBench의 r = 0.21(p = 0.51)과 대비된다(§5.2, App. F). RM 4개 × task family 3개는 작은 표본이다(안정성에 관한 열린 질문).

### §4.3 Preference Proxy Evaluation(PPE): 정확도 대 훈련된 policy

**메커니즘**([[ppe-reward-model-eval]] §4-6).

1. Human-preference set: 121개 언어에 걸친 20개 모델의 Chatbot Arena pair 16,038개, voter 6,120명이다(§4).
2. Correctness set: benchmark prompt(MMLU-Pro, MATH, GPQA, MBPP-Plus, IFEval)에서 네 모델 각각이 prompt당 32개 sample을 생성하고 benchmark verifier가 label을 붙인다. 정답 sample이 10% 미만 또는 90% 초과인 prompt는 제거한다(§5.1).
3. Downstream test: RM 9개는 각각 Llama-3.1-8B-Instruct의 8,000 prompt × 16 sample에 label을 붙인다(chosen = 가장 높은 RM 점수, rejected = 균일하게 샘플한 rank). 각 dataset으로 Llama-3.1-8B-Instruct를 DPO 훈련하고, 9개 policy는 blind Arena vote 12,190개로 rank한다(§6.1-6.2).

**계산 비교.** 다른 곳에서 보고된 RewardBench 점수를 포함한 post-DPO Arena 점수(Table 3)는 다음과 같다.

| DPO pair 구성에 쓴 RM | RewardBench 전체 | Post-DPO Arena 점수(95% CI) |
|---|---|---|
| Athene-RM-8B | 여기 인용한 출처에는 보고되지 않음 | 1209 (1199–1219) |
| Skywork-Reward-Gemma-2-27B | 94.1 ([[deepseek-grm]] Table 2) | 1173 (1163–1182) |
| Nemotron-4-340B-Reward | 92.0 ([[nemotron-4-synthetic]] Table 4) | 1172 (1163–1180) |
| 없음(Llama-3.1-8B-Instruct base) | — | 1178 (1168–1187) |

RewardBench 점수가 92를 넘는 두 RM은 confidence interval이 훈련하지 않은 base model과 겹치는 policy를 만들었다.

**증거(결과, 단일 연구).** Human-preference set의 pairwise accuracy는 post-DPO Arena 점수의 최선 단일 예측변수이며, ranking-correlation metric(Spearman, Kendall)은 상관이 거의 0이다(§7). Category 점수를 low quantile(최솟값에 더 가까움)에서 집계하면 상관이 높아지며 accuracy는 0.80에서 peak를 이룬다(Fig. 5). 저자들은 이를 “reward model의 어떤 domain weakness든 훈련 중 LLM이 exploit할 수 있다”는 뜻으로 해석한다(§7, Interpretation). Top model 사이에서는 RewardBench 점수가 downstream 결과와 음의 상관을 가진다(§2.2, Fig. 4).

**한계.** RM 9개, base model 하나, online RL이 아닌 offline DPO다(§8.2). PPO에서의 과도 최적화는 어떤 metric이 중요한지 바꿀 수 있다.

### §4.4 Generalist generative reward model(DeepSeek-GRM)

**정의.** Query와 하나 이상의 응답이 주어지면 principle을 쓰고 critique를 쓴 뒤 응답별 점수를 쓰는 pointwise generative RM이다([[deepseek-grm]] §3).

**메커니즘: Self-Principled Critique Tuning(SPCT).**

1. Rejective fine-tuning(cold start): RM 예시마다 DeepSeek-V2.5로 principle-and-critique trajectory를 N_RFT = 3회 sample한다. Score가 ground-truth best response를 엄격히 최고로 rank하지 못한 trajectory는 reject하고, 세 trajectory 모두 정답인 예시는 너무 쉽다며 reject한다(§3.2 Eq. 10; App. C.1). 선택적으로 prompt에 “The best response is: Response j”를 붙인다(hinted sampling).
2. GRPO를 이용한 rule-based online RL: 추출한 score가 ground-truth best response를 엄격히 최고로 rank하거나 single-response label과 맞으면 reward `+1`, 그렇지 않으면 `−1`이다. Format reward는 쓰지 않고 대신 더 큰 KL coefficient를 쓴다(§3.2 Eq. 11).
3. Inference-time scaling: k개 trajectory를 sample하여 pointwise score를 합한다. 선택적으로 meta RM이 trajectory마다 점수를 매기고 상위 k_meta만 vote한다(§4, Eq. 14).

**증거(결과, 단일 연구).** Gemma-2-27B 기반 모델이며 모든 baseline의 training data는 같다(§5.1, App. C.2). RewardBench, PPE Preference, PPE Correctness, RMB 전반의 전체 점수는 다음과 같다(Table 2).

| 모델 | RewardBench | PPE Pref. | PPE Correct. | RMB | Overall |
|---|---|---|---|---|---|
| DeepSeek-BTRM-27B (scalar BT) | 81.7 | 68.3 | 66.7 | 57.9 | 68.6 |
| DeepSeek-PairRM-27B | 87.1 | 65.8 | 64.8 | 58.2 | 69.0 |
| DeepSeek-GRM-27B (greedy) | 86.0 | 64.7 | 59.8 | 69.0 | 69.9 |
| DeepSeek-GRM-27B, meta-RM voting @32 | 90.4 | 67.2 | 63.2 | 70.3 | 72.8 |

Scalar BT RM은 PPE Correctness(verifiable task)에서 가장 강하고 RMB에서는 가장 약하다. 저자들은 scalar RM이 domain bias를 보인다고 기술한다(§5.2). Cold start에 rejective-sampling data가 없으면 모델은 RL 전 66.1, RL 후 68.7인 반면 full method는 69.9다. General instruction data를 제거하면 RFT 모델은 68.8에서 63.3으로 낮아진다(Table 4, §5.2). 300-sample RewardBench subset에서 DeepSeek-R1은 236B RFT 모델보다 낮은 점수를 냈다(§5.2).

**한계.** GRM은 policy RL 내부 reward가 아니라 benchmark 위 judge로 평가한다. 저자들은 이 통합을 future work로 제시한다(§7). Multi-domain RL에서 사용하는 rubric과 checklist reward는 ch-44b가 다룬다([[rubrics-as-rewards]]).

**범용 모델에 대한 함의.** RM benchmark 점수는 pair 전반의 평균이다. PPE low-quantile 결과 및 RM-Bench의 math·code 결과는 모두 generalist policy에서 domain 전반 최소값을 모니터링할 양으로 지목한다.

---
## §5 Policy coverage: 불완전한 reward model도 전이되는 때

### §5.1 정의와 문제

Policy coverage는 data-collection policy가 optimal policy가 생성할 응답에 얼마나 많은 확률을 두는지 측정한다. Huang et al.은 coverage coefficient를 다음과 같이 정의한다([[policy-coverage-loss]] Def. 2.2).

```
Cov_{π̃|π} = E_{s~ρ, a~π̃} [ π̃(a|s) / π(a|s) ]
```

`ρ`는 prompt 분포, `π̃`는 target policy, `π`는 data를 수집하는 policy, `a`는 응답이다. Coefficient가 크면 `π`가 `π̃`에 필요한 응답을 드물게 sample한다는 뜻이다.

### §5.2 계산 예시

한 prompt에 두 응답이 있고 target이 `π̃ = (0.5, 0.5)`를 둔다고 하자.

- Collector `π = (0.9, 0.1)`: `Cov = 0.5 · (0.5/0.9) + 0.5 · (0.5/0.1) = 0.278 + 2.5 = 2.78`.
- Collector `π = (0.99, 0.01)`: `Cov = 0.5 · 0.505 + 0.5 · 50 = 25.3`.

두 번째 응답의 확률이 9 point 바뀌면 coefficient는 약 9배 증가한다. 평균적으로 정확한 RM도 policy가 거의 sample하지 않는 응답에서는 policy를 개선할 수 없다.

### §5.3 결과와 조건

KL-regularized 목적함수 `J_β(π; r) = E[r] − β KL(π ‖ π_ref)`에서 reward [0, R]가 유도한 policy의 optimal policy coverage는 `1 + κ(e^{2R/β}) · value gap / β`로 상한 제한된다(Lemma 3.1). `β = 0`이면 거의 최적인 결정적 policy도 무한 coverage coefficient를 가질 수 있으므로 이 상한은 regularization에 의존한다(§3.1). 따라서 저자들은 iteration마다 여러 불완전 RM이 유도한 policy 중 current policy에 대한 estimated win rate로 선택하고, current policy도 candidate로 유지한다(§5, Alg. 3). T5-small(80M) policy와 Llama-3-8B 기반 simulated human reward를 사용한 XSum에서, 이는 3 iteration 뒤 transfer 없는 iterative DPO에 대해 54.0 ± 1.2% 승률에 도달했고, 항상 best source를 사용하는 방법에는 49.1–50.6%였다(Table 1; 결과, 단일 연구). Iteration 3에서는 어느 source도 이기지 못하므로 selection은 일반 online learning으로 돌아간다(§6, Fig. 2).

Verifier 훈련에서의 coverage 관련 근거로 Cobbe et al.은 2 epoch만 훈련한 generator로 verifier training sample을 생성한다. 더 많은 epoch에서는 test@100 성능이 “test@1보다 훨씬 더 급격하게 저하”하고 solution diversity가 붕괴하기 때문이다([[training-verifiers-to-solve-math-word-problems]] §4.1-4.2, Fig. 3).

**한계.** 요약 작업 하나, T5 policy, simulated preference, held-out-task 평가가 없다([[policy-coverage-loss]] §7).

**범용 모델에 대한 함의.** Reward-model accuracy와 policy coverage는 별개 조건이다. RM은 policy가 무시할 수 없을 만큼의 확률로 sample하는 응답에서만 policy로 전이된다. 이는 RM을 ch-43의 diversity control 및 ch-38a의 SFT 대 RL coverage 논의에 연결한다.

---
## 부정 샘플과 부정 피드백

이 절은 reward model과 verifier에 과정의 “negative” 정의를 적용한다(유도식은 ch-43a).

**1. 부정의 출처와 label 방법.**

- Human pairwise label: 비선호 응답 `y_l`이다. Label에는 noise가 있으며 Coste et al.은 보고된 agreement rate에 맞추어 25% flipped label을 모사한다([[reward-ensembling]] §4.3).
- Policy 자신의 sample에 대한 outcome check: 최종 답이 틀리면 solution은 부정이다. Cobbe et al.은 일부 solution이 결함 있는 추론으로 정답에 도달해 positive class false positive를 만든다고 지적한다([[training-verifiers-to-solve-math-word-problems]] §4.2). PRM800K에서 label된 solution의 14.2%는 정답에 도달하고 label된 step의 73.1%는 정답이므로([[prm800k]] App. B Table 3), 오답 solution의 outcome label은 어느 step이 틀렸는지 말하지 않는다.
- RM-labeled pool: West-of-N은 base preference model 아래 N개 policy sample 중 최고 및 최저 점수를 pair로 만든다([[west-of-n]] §4, arXiv:2401.12086v2).
- Preference label로 judge된 model-generated rationale: GenRM에서는 잘못된 verdict에 도달하는 rationale다([[generative-reward-models]] §4).

**2. 현재 관행이 이들을 다루는 방법.**

- BT RM 훈련은 RM의 gradient로 negative를 쓴다. Loss는 `r(y_w)`에 비해 `r(y_l)`을 낮추고 차이만 제약한다(§1.3).
- V-STaR는 language model의 gradient로 쓴다. Generator solution의 Cartesian product에서 나온 (correct, incorrect) pair로 verifier를 DPO 훈련하여, incorrect solution의 verifier likelihood를 낮춘다([[v-star]] §3.1 Eq. 2).
- STaR 스타일 self-training 및 STaR-SFT는 이를 버린다.
- PRM800K는 label할 대상을 선택하는 데 쓴다. Current PRM에서 최고 점수의 wrong-answer solution을 human labeler에게 보낸다([[prm800k]] App. B).

**3. 메커니즘.** RM에서 거부 점수의 gradient는 `∂L/∂r(y_l) = σ(r(y_l) − r(y_w))`다. `Δ = 3`의 쉬운 negative는 `σ(−3) = 0.047`, `Δ = 0.1`의 near-miss negative는 `σ(−0.1) = 0.475`로 10배 더 크다. RM이 이미 분리한 pair는 gradient 기여가 거의 없고 near-miss pair가 update와 label risk 대부분을 가진다.

Token likelihood로 훈련하는 verifier 또는 policy에서 softmax gradient는 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. `p = (0.665, 0.245, 0.090)` logits에서 `log p_3`(unlikely token 3의 unlikelihood step)을 낮추는 크기 η gradient step은 `Δz_3 = −0.910η`, `Δz_1 = +0.665η`, `Δz_2 = +0.245η`로 logits를 바꾼다. 제거된 확률 대부분은 가장 가능성 큰 대안 token 1로 간다. Token 1도 오답이면 이미 unlikely한 negative를 밀어내는 것이 그 오답에 질량을 집중시킨다(ch-43a).

**4. 수치가 있는 근거.**

- *Near-miss negative는 RM이 실패하는 곳이다.* Llama 2 helpfulness RM accuracy는 “significantly better” pair에서 79.1%, “negligibly better” pair에서 54.5%다([[bradley-terry-rm]], Table 28). Rejected 답에 미세한 오류가 있으나 더 많은 style을 가진 RM-Bench hard accuracy는 top RM에서 46.6%다(§4.2).
- *Easy negative는 label이 정확하지만 on-policy가 덜하다.* West-of-N에서 pseudo-preference accuracy는 N과 함께 증가해 high-confidence human data를 넘으며, N = 2(모든 pair label)는 RM을 해친다. N이 커질수록 policy 아래 최고·최저 sample likelihood는 감소하여 OOD 응답이 될 수 있다(§5.2, Fig. 4). West-of-N 한 round는 RM accuracy를 약 2.3%까지 올렸고 human data 두 배는 약 1%였다(§5.1).
- *Current policy의 negative.* Cobbe et al.은 current generator에서 training problem당 100 sample로 verifier를 훈련한다([[training-verifiers-to-solve-math-word-problems]] §4.2). V-STaR는 고정 generator 대신 개선된 generator에서 반복적으로 verifier data를 수집하고, Llama 2 7B·13B의 GSM8K·MBPP에서 이전 self-improvement 및 verification 방법보다 test accuracy 4%~17% 개선을 보고한다([[v-star]] Abstract, §3).
- *설득력 있는 오답 mining.* 설득력 있는 wrong-answer solution을 label하는 active learning은 uniform label보다 약 2.6× 더 data-efficient하다고 추정된다([[prm800k]] §4.2, Fig. 4a).
- *Judge에서 negative 유지 대 버리기.* STaR-SFT는 wrong-verdict rationale을 버리고 UltraFeedback 67.4%에 이른다. STaR-DPO는 이를 rejected output으로 사용해 73.9%, RewardBench 81.9%에 이른다([[generative-reward-models]] §5.1). 두 경우 loss도 바뀌므로 negative 몫의 깨끗한 측정은 아니다.

**5. 제어.**

- Negative를 on-policy로 유지한다. 매 round current policy에서 verifier 및 RM training sample을 재생성한다(V-STaR, West-of-N).
- 불확실한 label을 filter 또는 weight한다. West-of-N은 높은 base-model confidence `P_θ(y+ ≻ y− | x)` pair만 남겨 추가 이득을 얻었다(§5.2, Fig. 5). PPE는 정답 sample이 10% 미만 또는 90% 초과인 prompt를 버린다([[ppe-reward-model-eval]] §5.1).
- Label source가 오답과 검증 불가능한 답(예: 예상 밖 형식의 정답)을 구분하지 못하면 penalize 대신 mask한다. 가장 어려운 mined negative는 positive로 오표기될 가능성도 가장 크다(Interpretation; 이 장 출처 중 mined negative false-negative rate를 측정한 것은 없음).
- 전체 solution 대신 첫 incorrect step까지 label한다([[prm800k]] §2.6; ch-44).
- 알려진 negative 및 positive의 고정 regression set(학습자 ch-20/21 확장의 “negative anchor” set)을 유지하고 refresh 뒤마다 verifier의 false-accept 및 false-reject rate를 보고한다.

**6. 진단.** Chosen과 rejected 응답의 RM 점수를 따로 기록한다. Preference strength 및 style condition(Easy/Normal/Hard)별 accuracy를 보고한다. Verifier에는 known-wrong solution의 false-accept rate, known-correct solution의 false-reject rate를 보고한다. Best-of-k selected correctness를 k에 대해 그리고 큰 k에서 하락을 감시한다([[training-verifiers-to-solve-math-word-problems]] Fig. 7a; [[ppe-reward-model-eval]] §5.2.1).

**7. 일반성에 미치는 효과.** Near-miss 및 style-controlled negative에 실패하는 RM은 확신에 차고 잘 형식화된 오답을 보상하여 RL에서 hallucination을 높인다. 한 domain policy sample의 negative로 훈련한 RM은 다른 domain negative를 포괄하지 않는다. 여기 인용한 출처 중 RM 개선의 얼마가 positive와 별개로 negative에 기인하는지 측정한 것은 없다.

---

## 레시피

모든 행은 인쇄된 reward-model 또는 verifier training 설정이다. Card 행은 2026-09-14에 검증했고 2026-09-15 표시는 이 개정에서 primary text를 확인했다.

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| InstructGPT RM | 6B | reward-model | initialization; head | §3.5: unembedding을 제거한 SFT 모델, scalar 출력; App. C.2: public NLP dataset으로 fine-tune한 GPT-3 6B에서 최종 RM 초기화 | arXiv:2203.02155v1 §3.5, App. C.2 [[rlhf-instructgpt]] | conflict(§3.5 대 App. C.2; App. C.2는 최종 RM 설명) | App. C.2: GPT-3 또는 SFT init에서 유사 결과, 표 없음 |
| InstructGPT RM | 6B | reward-model | training prompt; prompt당 rank된 응답 | 6,623 labeler + 26,584 customer = 33,207 prompt; K = 4~9; tie 제거 | Table 6; §3.5; App. C.2 | derived(Table 6 행 합계) | — |
| InstructGPT RM | 6B | reward-model | LR; schedule; batch; epoch | 9e-6; 초기값 10%까지 cosine; 64 prompt(≤ 2,304 comparison); 1 epoch | App. C.2 | 2026-09-15 검증됨 | App. C.2: ±50% LR 유사, multiple epoch 과적합, 표 없음 |
| Llama 2 Helpfulness RM, Safety RM | 70B / smaller | reward-model | max LR; schedule; epoch; batch | 5e-6(70B), 1e-5(그 외); 10%까지 cosine, warmup 3%; 1 epoch; 512 pair | arXiv:2307.09288v2 §3.2.2 [[llama-2-recipe]] | 2026-09-14 검증됨 | “training longer can lead to over-fitting”; 수치 없음 |
| Llama 2 Helpfulness RM | per size | reward-model | margin m(r) | small {1, 2/3, 1/3, 0} 또는 large {3, 2, 1, 0}; final RM에 쓴 variant는 명시되지 않음 | v2 Eq. 2, Table 27 | 2026-09-14 검증됨 | Table 28: 평균 accuracy 62.5 / 63.0 / 62.9 |
| Llama 3.1 405B RM | 405B | reward-model | objective; ranking data | margin 없는 Llama 2 loss; edited > chosen > rejected | arXiv:2407.21783 §4.1.2 [[llama-3-recipe]] | 2026-09-14 검증됨 | margin은 개선 감소로 제거, 수치 없음 |
| Nemotron-4-340B-Reward | 340B | reward-model | data; head | HelpSteer2 10K; 5 attribute 선형 projection, weighted sum | arXiv:2406.11704v2 §3.1 [[nemotron-4-synthetic-recipe]] | 2026-09-14 검증됨 | RewardBench 92.0(Table 4) |
| Nemotron-4-340B-Reward | 340B | reward-model | LR, epoch, loss, attribute weight | 보고되지 않음 | §3.1 확인 | 보고되지 않음 | — |
| Gao proxy RM | 3M–3B | reward-model | comparison; batch | 100,000(10% held out); 64 | arXiv:2210.10760v1 §2.1, App. C Table 1 [[reward-model-overoptimization]] | 2026-09-14 검증됨 | Fig. 4: 데이터 증가, gold score 증가 |
| Coste proxy RM(Pythia) | 7M / 44M / 1.3B | reward-model | LR; epoch; batch; RM training sample | 1e-5; 5; 32; 46k sample | arXiv:2310.02743v2 §4.3, App. D.1 Table 2 [[reward-ensembling]] | 2026-09-15 검증됨 | 보고된 ablation 없음 |
| Coste ensemble | 44M member | reward-model | member; diversity | 5; random seed만(head init, data order) | §4.3 | 2026-09-15 검증됨 | Fig. 11: 4 ≈ 5 member, 3에서 gap |
| Coste PPO with WCO/UWO | Pythia 1.4B policy | RL | KL penalty | 0.01(single RM은 0.2 필요) | §5.2, Figs. 4, 6 | 2026-09-15 검증됨 | Fig. 7, penalty 전반 |
| WARM RM(PaLM-XXS) | 보고되지 않음 | reward-model | step; batch; LR set; member | 10k; 128; {1e-5, 4e-5, 1e-4}; M = 6 reference | arXiv:2401.12187v1 App. B.3, §5.2 [[warm-weight-averaged-reward-models]] | 2026-09-14 검증됨 | Fig. 9: M = 6 대 best single RM |
| GenRM STaR-DPO | 8B | reward-model | LR; DPO β; STaR iteration | 1.0e-6; 1.0; 3 | arXiv:2410.12832v1 App. A.2.1 Table 2, App. A.3 [[generative-reward-models]] | 2026-09-14 검증됨 | Table 3, iteration별 |
| DeepSeek-GRM-27B | 27B | reward-model(RFT) | data; LR; batch; step | 1256K(1070K general instruction + 186K rejective-sampled); 5e-6; 1024; 900 | arXiv:2504.02495v3 App. C.1 [[deepseek-grm]] | 2026-09-15 검증됨 | Table 4: general instruction data 없으면 63.3 |
| DeepSeek-GRM-27B | 27B | reward-model(RL) | algorithm; data; LR; batch; step; KL β; group size | ±1 rule reward의 GRPO; 237K; 4e-7; 512; 900; 0.08; G = 4 | App. C.1 | 2026-09-15 검증됨 | β grid {0.00, 0.01, 0.02, 0.08}: 작은 β는 일부 subset에서 붕괴 |
| West-of-N self-trained RM | T5-XXL 11B | reward-model | N; data mix; sampling temperature | N = 64; 1:1 base 및 West-of-N pair; 0.7 | arXiv:2401.12086v2 §5 “Methods” [[west-of-n]] | 2026-09-15 검증됨 | Fig. 4a: N에 따라 gain 증가 |
| GSM8K verifier | 6B, 175B | reward-model | generator epoch; 문제당 sample; verifier epoch; aux loss | 2; 100; 1; joint LM objective | arXiv:2110.14168v2 §4.2 [[training-verifiers-to-solve-math-word-problems]] | 2026-09-15 검증됨 | Fig. 3(coverage); Fig. 6b(LM objective) |
| PPE downstream DPO | Llama-3.1-8B-Instruct | eval-gate | pair; β (τ); batch; LR | 8,000 prompt × 16 sample; 0.1; 64; “2.00 × 10−0.6”으로 인쇄 | arXiv:2410.14872v2 §6.1, App. A.3 [[ppe-reward-model-eval]] | 2026-09-15 검증됨; LR conflict(인쇄된 exponent 사용 불가) | Table 3 |

**소규모 범용 실행의 시작점.** 6B~13B 규모의 chat 또는 SFT checkpoint에서 초기화한 pairwise scalar RM에 대해, 검증된 행은 preference data 1 epoch를 지지한다(InstructGPT 6B, Llama 2 모든 크기 모두 더 많은 epoch에서 과적합 보고). Peak LR은 10%까지 cosine decay하는 1e-5(Llama 2, 70B 미만) 또는 9e-6(InstructGPT 6B)이고, K > 2일 때 같은 ranked prompt의 모든 comparison을 한 batch element에 묶는다(InstructGPT). 이 값들은 33K ranked prompt(InstructGPT) 및 약 2.9M comparison(Llama 2) 실행에서 왔고, 더 작은 data에서 ablation되지 않았다. RM을 RL로 최적화할 경우 Coste 행은 5개 fine-tuning seed, worst-case 또는 uncertainty-weighted 결합, 작은 KL penalty(Pythia 1.4B 설정의 0.01)를 지지한다. RM 사용 전 PPE 스타일 downstream 검사를 수행하거나 최소한 domain별 accuracy와 domain 전반 최소값을 보고한다.

---
## 일반화 관점

**(a) 폭을 넓히는 것.**

- RM ensemble member의 diverse pretraining: pretrain-seed ensemble은 fine-tune-seed ensemble보다 잘 일반화한다([[helping-or-herding]] Abstract).
- 한 checkpoint RM의 weight average는 대부분 실행이 학습한 feature를 선호한다([[warm-weight-averaged-reward-models]] §4.3, §5.2).
- Generative-RM 훈련의 general instruction data: 이를 제거하면 DeepSeek-GRM-27B-RFT 전체 점수는 68.8에서 63.3으로 낮아진다([[deepseek-grm]] Table 4).
- On-policy RM 및 verifier data: West-of-N pair는 human-dataset pair보다 policy 아래 orders of magnitude 더 likely하다([[west-of-n]] §5.2, Fig. 4c). V-STaR는 iterative verifier data를 쓴다([[v-star]] §3).
- 더 강한 모델이 아니라 judge 자신에게서 sample한 rationale: Llama 3.1 70B rationale은 GenRM에서 RewardBench accuracy를 낮췄다([[generative-reward-models]] §5.3, Table 1).

**(b) 좁아짐을 일으키는 것.**

- Gold-score peak를 지나 최적화하는 것([[reward-model-overoptimization]] Fig. 1).
- Style의 shared RM error: factuality RM에서는 summary가 너무 짧고, quality RM에서는 너무 장황하며, helpfulness RM에서는 formulaic format이 된다([[helping-or-herding]] §1).
- Domain-biased scalar RM: UltraInteract 훈련 뒤 BT RM은 RewardBench Reasoning에서 무작위보다 낮다([[generative-reward-models]] §5.2). Scalar BT는 PPE Correctness에서 가장 강하고 RMB에서 가장 약하다([[deepseek-grm]] §5.2).
- 낮은 policy coverage: policy가 드물게 sample하는 응답은 어느 RM으로도 개선되지 않으며([[policy-coverage-loss]] §3.1), 과훈련한 generator는 test@100을 잃는다([[training-verifiers-to-solve-math-word-problems]] Fig. 3).

**(c) 이 단계에서 측정하는 방법.**

- Policy sample로 만든 correctness set과 human-preference set의 downstream-linked accuracy를 category 전반 low quantile로 집계한다([[ppe-reward-model-eval]] §7, Fig. 5).
- Domain별 style-controlled hard accuracy([[rm-bench]] §3.5).
- 최적화 중 √KL 대비 gold 또는 independent-judge 점수를 proxy와 같은 축에 plot한다([[reward-model-overoptimization]] Fig. 8).
- RL 중 intra-ensemble variance를 drift signal로 쓴다. PPO mean optimization에서는 label noise 없이 거의 3배, 25% noise에서 약 2.5배 올랐고, noise에서 UWO(λ = 0.1)는 약 20%였다([[reward-ensembling]] §5.5, Fig. 10).
- 알려진 측정 오류: benchmark leakage([[ppe-reward-model-eval]] §8.1), style이 결정하게 하는 서로 다른 모델의 chosen/rejected 응답([[rm-bench]] §1), correlation 연구의 적은 RM 수(RM-Bench §5의 4개, PPE §6의 9개).

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| RewardBench 전체 점수만으로 RM 선택 | RM으로 훈련한 policy가 blind 비교에서 base model을 이기지 못함 | PPE 스타일 downstream test 또는 domain별 최소 accuracy([[ppe-reward-model-eval]] Table 3) |
| Prompt 간 RM 점수 비교 | Raw score 기반 filtering/curriculum이 쉬운 prompt를 선호 | Prompt별 또는 reference 응답에 대해 normalize한다. BT 점수는 `f(x)`까지 식별됨(§1.4) |
| Proxy 점수를 training progress로 보고 | Proxy는 계속 오르나 independent judge는 평평하거나 하락 | √KL에 대해 둘 다 같은 축에 plot([[reward-model-overoptimization]] Fig. 8) |
| Offset 고정 없이 RM의 min 또는 median 사용 | Aggregate가 가장 낮은 `C(x)`를 고른 RM에 지배 | Aggregation 전 offset regularizer 또는 prompt별 centring([[helping-or-herding]] Eq. 2) |
| Seed ensemble을 독립으로 취급 | Ensemble과 member가 같은 길이/형식 오류 | Pretrain-diverse member를 쓰고 최종 policy top-scored 출력을 style feature별로 검사 |
| 고정된 초기 generator로 verifier 훈련 | Offline verifier accuracy는 높지만 policy update 뒤 best-of-k gain 축소 | 매 round current policy에서 verifier data 재생성([[v-star]] §3) |
| 쉬운(best-versus-worst) negative만 mining | 자체 pair에서 RM accuracy는 오르나 미세 오류 hard accuracy는 무작위 근처 | RM-Bench 스타일 Hard accuracy 및 preference strength별 accuracy 보고 |
| Verifier에 대해 best-of-k 무제한 실행 | 큰 k에서 selected correctness 하락 | k 전반 best-of-k curve([[training-verifiers-to-solve-math-word-problems]] Fig. 7a) |
| K-way ranking을 독립 shuffle pair로 사용 | RM이 한 pass 안에 overfit | Prompt의 모든 C(K,2) comparison을 한 batch element에 묶음([[rlhf-instructgpt]] §3.5) |

---

## 이해도 점검

1. BT loss는 pair 두 점수에 상수를 더해도 변하지 않는다. 이것이 추가 제약 없이는 RM ensemble minimum을 정의할 수 없게 하는 이유와 Eisenstein et al. regularizer가 문제를 없애는 방법을 설명하라.
2. 점수 (2, 1, 0)의 세 응답에서 Plackett–Luce와 평균 all-pairs BT는 0.721 및 0.251의 서로 다른 loss를 낸다. 각 목적함수가 더 크게 가중하는 pair comparison과 K = 9 ranking에서 바뀌는 것을 설명하라.
3. Gao et al.은 proxy RM 크기와 함께 β_RL이 감소해도 α_RL은 고정할 수 있다고 한다. `d*_RL = exp(α_RL/β_RL − 1)`로 짧은 실행에서 β_RL 추정의 작은 오류가 예측 stop point의 큰 오류를 만드는 이유를 설명하라.
4. Coste et al.은 25% label noise에서 mean optimization은 과도 최적화하지만 WCO/UWO는 그렇지 않다고 한다. 한 ensemble member의 과대평가 관점에서 메커니즘을 설명하고, 이것이 Eisenstein et al.의 실패를 해결하지 못하는 이유를 설명하라.
5. PPE에서 RewardBench 92 초과 RM 두 개가 base model보다 나은 policy를 만들지 못했다. RM-Bench style-controlled 결과 및 PPE low-quantile 결과와 양립하는 인과 설명 두 가지를 제시하라.
6. 2-epoch generator sample로 훈련한 verifier가 잘 작동했지만 RL 뒤 policy가 새 오답 class를 만든다. Coverage coefficient와 V-STaR data loop으로, 새 class의 verifier false-accept rate가 원래 validation set에서 측정되지 않는 이유를 설명하라.
7. Near-miss negative(`Δ = 0.1`)가 easy negative(`Δ = 3`)보다 RM gradient를 10배 만드는 이유와, 같은 속성이 mislabeled near-miss negative의 비용을 높이는 이유를 설명하라.

---

## 연결

- 이전: **ch-15** — Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage. §1의 ceiling을 정하는 pairwise label, agreement rate, prompt coverage를 제공한다.
- 다음: **ch-37** — Policy-Gradient Foundations for Language Models. RM 점수를 policy-gradient estimator의 return으로 쓴다.
- **ch-38** — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax. §2.4에서 과도 최적화와 상호작용을 측정하는 KL 항을 다룬다.
- **ch-38a** — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity. §5의 coverage 및 on-policy data를 다룬다.
- **ch-39** — Offline Preference Optimization: DPO and Its Variants. §1 BT와 Plackett–Luce에서 유도한 implicit reward `β log π/π_ref`로 explicit RM을 대체한다.
- **ch-42** — Reward Hacking and Judge Design. Optimized policy와 reward로 쓴 LLM judge의 failure mode다.
- **ch-43a** — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Negative section의 전체 유도식을 담는다.
- **ch-44** — Process Supervision and Verifiable Rewards. Step-level label과 verifier를 다룬다.
- **ch-44b** — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. RL에서 쓰는 rubric 및 checklist reward를 다룬다.
- **ch-30c** — Weight Averaging and Model Merging for Generalist Models. RM 대신 policy에 적용한 weight average다.
- **ch-49** — Judge Models: Bias, Calibration, and Judge-Specific Overfitting. §3.5와 §4.4 generative judge의 평가 측면을 다룬다.

---

## 출처

- [[bradley-terry-rm]] — BT 모델과 loss, 식별성, Stiennon RM scaling 및 transfer, preference strength별 Llama 2 accuracy, IPO deterministic-label 한계를 다룬다.
- [[dpo]] — Plackett–Luce model(App. A.3 Eq. 18) 및 BT로의 축소를 다룬다.
- [[rlhf-instructgpt]] — K-way all-pairs RM loss, prompt별 batching, RM hyperparameter(App. C.2), dataset 크기(Table 6)를 다룬다.
- [[reward-model-overoptimization]] — Functional form, coefficient scaling, data 및 policy-size 결과, KL penalty 결과, Goodhart 해석을 다룬다.
- [[reward-ensembling]] — Mean, WCO, UWO aggregation, label noise의 best-of-n/PPO 결과, ensemble 크기, hyperparameter를 다룬다(arXiv:2310.02743v2에서 해당 위치 인용).
- [[helping-or-herding]] — Pretrain 대 fine-tune ensemble, offset regularizer, member 간 공유 reward hack을 다룬다.
- [[warm-weight-averaged-reward-models]] — Weight average, toy `p_j²` 분석, 79.4% 승률, 한계를 다룬다.
- [[pairrm]] — Joint pairwise encoding, aggregation 비용, MixInstruct 및 model-card 결과를 다룬다.
- [[generative-reward-models]] — STaR-DPO judge, RewardBench 및 UltraInteract 일반화 결과, negative 유지 대 discard를 다룬다.
- [[rlaif-scaling]] — d-RLAIF scoring 메커니즘, 승률, 비용 추정치를 다룬다(arXiv:2309.00267v3에서 해당 위치 인용).
- [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]] — Multi-attribute RM architecture, RewardBench 92.0, RM-as-judge 결과, 보고되지 않은 RM 설정을 다룬다.
- [[rewardbench]] — Benchmark 구성 및 scoring을 다룬다.
- [[rm-bench]] — Style-substance matrix, hard accuracy, policy 성능과 상관을 다룬다.
- [[ppe-reward-model-eval]] — Downstream DPO 실험, Arena 점수, metric correlation, low-quantile aggregation을 다룬다.
- [[deepseek-grm]] — SPCT 방법, generalist RM 결과, ablation, training 설정을 다룬다.
- [[policy-coverage-loss]] — Coverage coefficient, Lemma 3.1, empirical TPO 결과를 다룬다.
- [[training-verifiers-to-solve-math-word-problems]] — Verifier recipe, generator coverage, 큰 k에서 test-time 하락을 다룬다(arXiv:2110.14168v2에서 해당 위치 인용).
- [[v-star]] — Policy의 correct/incorrect sample에 대한 DPO verifier, iterative data를 다룬다(arXiv:2402.06457v2에서 해당 위치 인용).
- [[prm800k]] — 설득력 있는 wrong-answer active learning, label 통계, first-incorrect-step label을 다룬다.
- [[west-of-n]] — Best-versus-worst pseudo-preference, N 및 filtering ablation을 다룬다(arXiv:2401.12086v2에서 해당 위치 인용).
- [[llama-2-recipe]], [[llama-3-recipe]] — RM training 행을 다룬다.
- [[rubrics-as-rewards]] — ch-44b에서 다루는 rubric reward pointer다.
