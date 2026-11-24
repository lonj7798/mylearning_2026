<!-- chapter: ch-08a
     track: pretraining
     kind: content
     title: 스케일링 법칙과 연산 배분: 사전학습 손실에서 다운스트림 능력까지
     deps: [ch-00, ch-03]
     sources: [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]], [[resolving-scaling-discrepancies]], [[resolving-scaling-discrepancies-recipe]], [[chinchilla-replication]], [[deepseek-llm]], [[beyond-chinchilla-inference-scaling]], [[overtraining-downstream-scaling]], [[llama-3]], [[llama-3-recipe]], [[smollm2]], [[task-scaling-model-ladders]], [[scaling-laws-unreliable-downstream]], [[predicting-downstream-elusive]], [[data-constrained-scaling]], [[overtrained-lms-harder-to-finetune]], [[cooldown-scaling-beyond-fixed-durations]], [[emergence-loss-perspective]], [[same-loss-better-downstream]], [[datadecide]], [[signal-and-noise-eval]]
     figures: figures/compute-allocation-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 08a — 스케일링 법칙과 연산 배분: 사전학습 손실에서 다운스트림 능력까지

> **핵심 통찰.** 사전학습 loss는 parameter N과 training token D에 대한 적합된 power law를 따르지만 compute-optimal 분할은 law를 어떻게 적합하는지에 달려 있다. Kaplan et al.은 N ∝ C^0.73을 발견했고([[kaplan-scaling-laws]]), Hoffmann et al.은 자신들의 설정에서 N과 D가 같은 비율로, parameter당 약 20 token으로 증가해야 한다고 발견했다([[chinchilla-compute-optimal]]). Porian et al.은 output-head FLOP를 세고, warmup을 줄이고, size별 learning rate·batch size·β2를 tuning하여 Kaplan 설정 재현의 exponent를 0.835에서 0.497로 옮겼다([[resolving-scaling-discrepancies]], 표 1). Loss-optimal 분할은 inference를 무시한다. Serving cost를 세면 parameter당 20 token을 지나 학습한 작은 모델이 같은 loss에 더 낮은 total cost로 도달하며([[beyond-chinchilla-inference-scaling]]), 그 loss는 여전히 예측 가능하다([[overtraining-downstream-scaling]]). Loss는 많은 task의 평균은 잘 예측하지만 개별 task는 잘 못 한다. 한 재분석에서 46개 task 중 18개만 예측 가능하게 scale했고([[scaling-laws-unreliable-downstream]]), 한 연구는 3T token으로 사전학습한 OLMo-1B가 instruction tuning 뒤 2.3T-token checkpoint보다 나쁘다고 발견했다([[overtrained-lms-harder-to-finetune]]).
>
> **지침.** 고정 budget에서 N과 D를 고를 때는 target data·architecture에서 3개 이상 budget의 IsoFLOP curve를 적합하라. Output head를 포함해 FLOP를 세고, model size별 warmup을 설정하며, 각 run token에 맞춘 decay 또는 size별 tuning한 learning rate·batch size·β2를 사용하라. 한 재현에서 처음 세 정정으로 exponent는 0.835에서 0.571로, size별 tuning으로 0.497로 바뀌었고([[resolving-scaling-discrepancies]] 표 1), data quality에 따라 0.450에서 0.578로 바뀌었기 때문이다([[deepseek-llm]] 표 4). Model이 lifetime inference token을 training token과 비슷하거나 더 많이 serve할 때는 inference FLOP를 포함한 뒤 target loss에 도달하는 가장 작은 model을 고르라. Sardana et al. 예시에서 2T inference token의 13B-Chinchilla-quality model은 total FLOP 17%를 절약했다(그림 1). 사전학습 결정을 downstream capability로 판단할 때는 많은 task 평균 또는 bits per byte의 task loss를 forecast하고 held-out larger run에서 forecast를 backtest하며, 긴 over-training run에 착수하기 전에 같은 post-training recipe로 두 token-budget checkpoint를 fine-tune하라. 같은 fit은 17-task 평균을 0.05% 이내 오차로 예측하면서 HellaSwag에는 relative error 25.73%를 냈고([[overtraining-downstream-scaling]] 표 2), 3T-token OLMo-1B base는 instruction tuning 뒤 AlpacaEval에서 2.3T-token checkpoint보다 최대 3% 낮았기 때문이다([[overtrained-lms-harder-to-finetune]] §2). 그렇지 않으면 task별 accuracy forecast를 검증되지 않은 것으로 취급하라.

## 이 장이 범용 모델에 중요한 이유

Pipeline은 pre-training → mid-training → SFT → preference optimization → RL → evaluation이다. Pre-training은 이후 모든 단계가 물려받는 parameter count를 고정한다. Run 전에 내리는 두 결정은 model size N과 training token 수 D다. 둘은 run 뒤에 싸게 수정할 수 없다.

측정 가능한 문제는 세 부분이다.
1. **배분.** Compute budget C에서 하나의 (N, D)가 최저 pre-training loss를 낸다. 이 분할의 발표된 규칙은 서로 다르며, 한 data set과 training setup에서 적합한 규칙이 다른 곳에서는 성립하지 않을 수 있다.
2. **배포 비용.** FLOP당 training loss를 최소화하는 모델은, 같은 loss에서 training과 serving cost 합계를 최소화하는 모델보다 크다.
3. **능력.** 범용 모델은 많은 downstream task(ch-00)로 판단한다. Loss는 law가 예측하는 양이고, 이 장은 loss가 downstream accuracy와 post-training quality를 얼마나 잘 예측하는지 측정한다.

ch-03은 scaling experiment가 의존하는 learning-rate schedule과 batch-size mechanics를 다룬다. ch-08b는 predictability 질문을 in-context learning과 emergence로 더 확장한다. ch-14는 unique data가 binding limit인 경우를, ch-14a는 공개 recipe의 budget 비교를 다룬다.

## §1 양과 계산 관례

**정의.** N은 parameter count다. D는 학습에서 본 training token 수이며 unique token 수가 아니다. C는 floating-point operation(FLOP) 단위의 training compute다. **Compute-optimal** (N, D)는 고정된 C에서 loss를 최소화한다. **IsoFLOP curve**는 같은 C를 쓰는 run들의 N에 대한 loss이며, D = C/(6N)이다.

**문제.** 같은 model도 관례에 따라 서로 다른 N과 C를 배정받을 수 있으며, 이 오류는 law를 적합하는 데 쓰는 small model에서 가장 크다.

**메커니즘.** Token당 forward-pass compute는 약 2N이고 backward pass는 forward pass의 약 두 배이므로, training은 token당 약 6N FLOP를 쓴다([[kaplan-scaling-laws]] §2.1).

```
C ≈ 6 · N · D
```

C는 training FLOP, N은 parameter count, D는 training token이다. N에 무엇을 포함하는지는 관례에 따라 다르다.

```
6·N₁ = 72 · n_layer · d_model²                                   (non-embedding parameter, Kaplan et al.)
6·N₂ = 72 · n_layer · d_model² + 6 · n_vocab · d_model           (all parameter, Hoffmann et al.)
M    = 72 · n_layer · d_model² + 12 · n_layer · d_model · l_seq  (token당 non-embedding FLOP, DeepSeek)
```

n_layer는 layer 수, d_model은 width, n_vocab은 vocabulary size, l_seq는 sequence length다. M은 attention computation은 세지만 vocabulary projection은 세지 않으며, DeepSeek은 C = M·D로 쓴다([[deepseek-llm]] §3.2 식 2). Porian et al.은 네 번째 관례를 쓴다. Input embedding을 제외하고 output head를 포함한 모든 linear layer다([[resolving-scaling-discrepancies]] §2.1).

**계산 예시**(DeepSeek 표 3, 첫 행; n_layer = 8, d_model = 512, n_vocab = 102,400, l_seq = 4,096):
- 72 · 8 · 512² = 150.99M이므로 N₁ = 150.99M / 6 = 25.2M이다.
- 6 · 102,400 · 512 = 314.57M이므로 N₂ = (150.99M + 314.57M) / 6 = 77.6M이다.
- 12 · 8 · 512 · 4,096 = 201.33M이므로 M = 150.99M + 201.33M = token당 352M FLOP다.
- 6N₁/M = 0.43, 6N₂/M = 1.32다. 80 layer, d_model = 8,192 행에서는 비율이 0.92와 0.94다(표 3).

이 작은 크기에서 두 parameter 관례는 FLOP count와 서로 반대 방향으로 2.3배와 1.3배 차이 난다. 80 layer에서는 8% 이내로 일치한다. Size에 따라 달라지는 양만큼 compute를 잘못 쓰는 관례로 이 범위를 가로질러 law를 적합하면 exponent에 bias가 생긴다.

**증거.** Porian et al.에서 head를 FLOP count에서 빼면 가장 작은 model의 compute는 약 90%, 큰 model은 약 10% 과소계산된다. 이를 정정하면 RefinedWeb model-size exponent가 0.835에서 0.706으로 낮아졌다(§3.2, 표 1). Hoffmann et al.은 N과 FLOP 모두에 embedding을 센다(부록 F).

**함의.** 논문 간 exponent 또는 parameter당 token 비율을 비교하기 전에, 각각이 어떤 N과 C를 쓰는지 확인하라.

## §2 Kaplan et al. (2020): power law와 0.73 배분

**정의.** Power law L(X) = (X_c/X)^α_X는 X가 일정한 배수로 늘어날 때마다 loss가 일정한 비율로 떨어진다는 뜻이다.

**수식**([[kaplan-scaling-laws]] §1.2):

```
L(N)      = (N_c / N)^α_N,       α_N ≈ 0.076,  N_c ≈ 8.8 × 10^13          (Eq. 1.1)
L(D)      = (D_c / D)^α_D,       α_D ≈ 0.095,  D_c ≈ 5.4 × 10^13 tokens   (Eq. 1.2)
L(C_min)  = (C_c / C_min)^α_C,   α_C ≈ 0.050,  C_c ≈ 3.1 × 10^8 PF-days  (Eq. 1.3)
L(N, D)   = [ (N_c / N)^(α_N/α_D) + D_c / D ]^α_D                        (Eq. 1.5)
```

L은 WebText2에서 token당 nat 단위 test cross-entropy다. N은 embedding을 제외한다. D는 token 단위 dataset size다. C_min은 critical batch size보다 훨씬 작은 batch size에서 loss에 도달할 compute다. Critical batch size는 더 키워도 optimizer step 수가 점점 덜 줄어드는 batch size다(§5.1). PF-day 하나는 8.64 × 10^19 FLOP다. N_c, D_c, C_c는 tokenizer에 의존하고 “do not have a fundamental meaning”이다(§1.2).

**배분의 메커니즘.** Kaplan et al.은 768~1.5B non-embedding parameter model을 3,000-step warmup과 0까지 cosine decay로 고정 2.5 × 10^5 step 학습했다(§2.2, §3). Training curve에서 loss envelope을 취하고 critical batch size에 맞춰 조정한 뒤 N ∝ C_min^0.73을 적합했다. 조정하지 않은 fixed-batch fit은 C를 PF-day로 하여 N = 1.6 × 10^9 · C^0.88이다(§6.1, 그림 14).

**계산 예시.** N을 두 배로 하면 L(N)은 2^−0.076 = 0.949배가 된다(§1.2). N ∝ C^0.73 아래 100× compute 증가는 N을 100^0.73 = 28.8배, D를 100^0.27 = 3.47배 곱한다(도출됨). Hoffmann et al.은 Kaplan et al. 규칙을 10× budget에 parameter 5.5배와 token 1.8배로 요약한다(§1). 10^0.73 = 5.37, 10^0.27 = 1.86이다(도출됨).

**조건과 한계.** 모든 run은 schedule length 하나를 공유하므로, 일찍 멈춘 model은 learning rate를 decay하지 못했다. §3은 이것이 중요한 이유를 보인다. 저자들은 L(C_min)과 L(D) trend가 약 10^12 parameter와 10^12 token 근처에서 교차하며 값의 불확실성이 한 order of magnitude이고 law는 그 전에 무너져야 한다고 말한다(§6.3). 상태: **결과(단일 연구)**이며 배분에 대해서는 §3–§4가 이를 대체한다.

## §3 Hoffmann et al. (2022): 세 estimate와 동등 scaling

**문제.** Kaplan et al.과 GPT-3 training setup을 따른 2020–2021년의 대형 모델 대부분은 약 300B token으로 학습했다(Gopher 280B, GPT-3 175B, Jurassic 178B; [[chinchilla-compute-optimal]] §1, 표 1). Hoffmann et al.은 같은 compute에서 더 작은 model을 더 많은 token으로 학습하면 loss가 더 낮은지 묻는다.

**메커니즘.** 70M~16B 초과 parameter, 5B~500B token의 400개 넘는 model에 세 estimator를 사용했다(§3).
1. **Training-curve envelope.** 각 size를 16×에 걸친 cosine horizon 4개로 학습하고, FLOP 값 1,500개 각각에서 최저-loss run을 유지한다.
2. **IsoFLOP profile.** 6 × 10^18~3 × 10^21 FLOP의 budget 9개에서 N을 바꾸고 token count에 schedule을 맞추며 parabola를 적합하고 minimum을 취한다.
3. **Parametric fit.** 모든 final loss를 Eq. 2에 적합한다. Log L의 Huber loss를 L-BFGS(quasi-Newton optimizer)로 initialization grid에서 최소화한다. Huber loss는 δ = 10^−3보다 작은 residual에는 quadratic, 그 이상에는 linear이므로 큰 residual point의 weight가 작다(식 3, 부록 D.2).

```
L̂(N, D) = E + A / N^α + B / D^β                                     (Eq. 2)
N_opt(C) = G · (C/6)^a,   D_opt(C) = G^−1 · (C/6)^b                   (Eq. 4)
G = (α·A / (β·B))^(1/(α+β)),   a = β/(α+β),   b = α/(α+β)
```

E는 irreducible loss(data distribution 아래 text entropy), A/N^α는 finite model의 excess loss, B/D^β는 finite training의 excess loss이며 A, B, α, β는 fit constant다. Eq. 4는 C = 6ND 조건에서 Eq. 2를 최소화해 나온다. 인쇄된 fit은 E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28이다(부록 D.2).

**증거.** N_opt ∝ C^a인 exponent a는 세 접근에서 0.50, 0.49, 0.46이며 Kaplan et al.의 0.73과 대비된다(표 2). 접근 1은 1B parameter에 20.2B token, 10B에 205.1B token을 예측한다(표 3). 이는 parameter당 약 20 token이다(도출됨; 논문은 “20” 규칙을 인쇄하지 않는다). Gopher budget 5.76 × 10^23 FLOP에서 estimate는 40B~70B parameter다. Chinchilla 70B는 1.4T token으로 학습했고 5-shot MMLU에서 Gopher 60.0% 대비 67.6%에 도달했다(§4, 표 6; 초록은 67.5%를 인쇄).

**Hoffmann et al.이 불일치에 제시한 설명.** Kaplan et al.은 130B token까지 cosine schedule 하나를 사용했으므로 짧은 run의 loss는 learning-rate decay 전인 schedule 중간에서 읽혔다. 이는 맞춘 schedule이 도달할 값을 과대평가한다(§2). Run의 1×~5× cycle length grid에서 run보다 25% 넘게 긴 cosine cycle은 “leads to clear drops in performance”였다(부록 B, 그림 A1).

**계산 예시.** 인쇄된 coefficient로 a = 0.28/(0.34 + 0.28) = 0.452이다(표 2는 접근 3에 0.46을 인쇄). 100× compute에서 접근 1 exponent는 N × 100^0.50 = 10, D × 10을 주며, Kaplan의 28.8 및 3.47과 대비된다(도출됨).

**조건과 한계.** Large run은 Chinchilla와 Gopher 두 개만 비교했다. Frontier는 높은 compute에서 concavity를 보이고, 모든 run은 한 epoch보다 적게 사용했다. C4와 GitHub code 결과는 “as long as one does not train for more than one epoch”일 때 일치한다(§5, 부록 C, 부록 E). 상태: **재현됨**. Equal-scaling 결과는 [[deepseek-llm]]의 IsoFLOP analysis(현재 data에서 a = 0.524)와 [[resolving-scaling-discrepancies]](a = 0.497)가 재현했다.

## §4 Estimate가 불일치하는 이유와 발표된 law의 정밀도

**Porian et al. (2024).** 저자들은 5M~901M model로 RefinedWeb과 OpenWebText2에서 Kaplan setup을 재현하고 차이를 하나씩 제거했다([[resolving-scaling-discrepancies]] 표 1, RefinedWeb / OpenWebText2).

| 단계 | 변경 | Exponent a |
|---|---|---|
| 1 | Kaplan 재현(fixed warmup 약 1.57B token, 약 131B token에서 0까지 cosine, head 제외) | 0.835 / 0.864 |
| 2 | Output-head FLOP 계산 | 0.706 / 0.699 |
| 3 | Warmup token = N | 0.602 / 0.603 |
| 4 | 각 budget에 맞춘 cosine decay | 0.571 / 0.574 |
| 5 | Constant LR, size별 LR·batch size·β2 tuning(단계 4 대신) | 0.497 / 0.518 |

Fixed warmup에서 small model은 compute-optimal token count에 도달할 때에도 warmup 중이었다(§3.3). Decay를 맞추면 exponent가 약 0.03 바뀌므로, Hoffmann et al.이 제안한 schedule mismatch는 이 재현에서 주원인이 아니다(§3.4). Size별 tuned hyperparameter는 BS = 0.00037 · N^0.703, LR = 3.7 · N^−0.36을 따른다(그림 3). 한계: 가장 큰 model은 Hoffmann et al.의 16B 초과와 비교해 901M이며, 저자들은 compute가 Kaplan et al. 규모 정도라고 설명한다(§5.2). 상태: **결과(단일 연구)**이며 DeepSeek IsoFLOP 결과와 일관된다.

**계산 예시.** 100× compute 증가는 단계 1 exponent에서 N을 100^0.835 = 46.8배, 단계 5 exponent에서 100^0.497 = 9.9배 곱한다(도출됨). Exponent 0.34 차이는 두 order of magnitude에서 권장 model size를 4.7배 바꾼다.

**Besiroglu et al. (2024).** 저자들은 Hoffmann et al. 그림 4의 점을 digitize하고, outlier 5개를 제거한 뒤 남은 240개 점에 Eq. 2를 다시 적합했다([[chinchilla-replication]] §2–§3).

```
L(N, D) = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658            (Eq. 3)
```

이 재적합은 a = 0.5126, standard error 0.018을 주며, parameter당 약 20 token을 뜻한다. 이는 접근 1–2와 Chinchilla의 실제 학습 모두와 일관된다(§3.3). 발표된 접근 3 fit이 달라진 원인은 두 가지다(§3.1–§3.2). 첫째, 본문은 반올림한 값을 인쇄했다. β를 0.2849에서 0.28로 반올림하면 D = 10^11 token에서 data term은 약 (10^11)^0.0049 − 1 ≈ 13% 커진다. 둘째, 원래 optimizer는 Huber loss를 합산하지 않고 평균냈다. 이 때문에 L-BFGS가 일찍 멈췄고, a의 confidence interval (0.454–0.455)을 정당화하려면 약 600,000회 run이 필요하다. 저자들은 두 번째 원인을 Hoffmann et al. 저자의 확인으로 돌린다. 이들의 재적합에서 80% confidence region은 10^26 FLOP 이상으로 학습하는 model에 대해 parameter당 4~40 token과 양립한다(§4). 상태: 재구성한 data 위의 **결과(단일 연구)**이며, loss 정밀도는 약 0.01이다(§2).

**계산 예시: coefficient를 쓰기 전에 논문 자체의 예측을 재현하라.** Gopher budget C = 5.76 × 10^23에서 Eq. 4는 인쇄된 coefficient로 N_opt = 32.2B를, 반올림하지 않은 TeX 값(E = 1.6934, α = 0.3392, β = 0.2849)으로 40.3B를 준다. Hoffmann et al.은 접근 3에 대해 40B라고 적었다(그림 4 caption). 32.2B와 40.3B는 도출값이다. Llama 3에도 같은 점검을 적용할 수 있다. Report는 optimal token count A·C^α에 대해 (α, A) = (0.53, 0.29)를 인쇄하고, 3.8 × 10^25 FLOP에서 16.55T token으로 402B parameter를 학습했다고 말한다. 이는 parameter당 41.2 token이다(도출됨; report는 이 token count를 N*(C)로 쓴다) ([[llama-3]] §3.2.1). C ≈ 6ND로 계산하면 이 plan의 비용은 6 × 402B × 16.55T = 3.99 × 10^25 FLOP로, 명시한 budget보다 5% 높다(도출됨). 인쇄된 pair는 0.29 × (3.8 × 10^25)^0.53 = 10.5T token을 준다. 그림 3 legend의 값 (0.537, 0.299)은 16.3T를 준다(도출됨). log10 C = 25.58이므로 exponent 차이 0.007은 예측에 10^(0.007 × 25.58) = 1.51배를 곱하고, coefficient 비율 0.299/0.29는 추가로 1.03배를 곱한다.

**Data quality도 exponent를 바꾼다.** DeepSeek은 N 대신 M으로 IsoFLOP profile을 적합했다. Main fit은 10^17~3 × 10^20 FLOP의 budget 8개를 썼다. 결과 a는 초기 in-house data에서 0.450, 현재 in-house data에서 0.524, OpenWebText2에서 0.578이었고, 저자들은 이 data들을 quality가 증가하는 순서로 매겼다([[deepseek-llm]] §3.3, 표 4). Quality가 높은 text는 예측하기 덜 어렵다는 설명은 저자들이 speculation이라고 표시한다. 상태: **결과(단일 연구)**. Porian et al.은 0.578 값의 가능한 원인으로 data repetition을 제시한다([[resolving-scaling-discrepancies]] §5.1).

**함의.** Parameter당 token 규칙은 data set, counting convention, hyperparameter protocol의 성질이다. 셋 중 하나라도 바뀌면 다시 적합하라.

## §5 Compute-optimal과 의도적인 과학습

**정의.** **Token multiplier**는 \(M = D/N\)이다. **과학습(over-training)**은 data에 대해 compute-optimal인 \(M^*\)보다 큰 \(M\)으로 학습하는 것을 뜻한다. 즉, loss-per-FLOP 관점에서 권장하는 것보다 작은 model에 더 많은 token을 학습시키는 경우다([[overtraining-downstream-scaling]] §2.1). **Reducible loss**는 \(L - E\)이다. 여기서 \(E\)는 무한한 model과 data에서도 남는 irreducible loss이므로, reducible loss는 model 크기와 학습량을 늘려 줄일 수 있는 loss 부분이다.

**문제.** Training compute는 한 번만 지불하지만 inference compute는 service하는 token마다 계속 발생한다. 따라서 일정한 quality를 만들 때 training FLOP만 최소화하는 model은, training FLOP와 serving FLOP를 합친 total FLOP를 최소화하는 model보다 크다.

**메커니즘** ([[beyond-chinchilla-inference-scaling]] §2).
1. 목표 pre-training loss \(\ell\)를 고정한다.
2. 후보 \(N\)마다 \(L(N, D) = \ell\)을 만족하는 \(D\)를 푼다. 즉, 그 size의 model이 목표 loss에 도달하도록 필요한 training token 수를 구한다.
3. Lifetime cost \(6·N·D + 2·N·D_{inf}\)를 계산한다. \(D_{inf}\)는 모든 request의 input token과 output token을 합한 수이고, token 하나의 forward-pass 비용이 \(2N\)이므로 두 번째 항은 inference cost다.
4. 이 cost가 가장 작은 \(N\)을 택한다. 일반적인 closed form은 없으므로 저자들은 수치적으로 푼다(부록 A).

**증거.**

- 2T inference token을 처리하는 13B-Chinchilla-quality model은 더 많은 data로 학습한 7B model이 같은 quality에 도달하며, total FLOP는 17% 적다. \(10^{13}\) inference token을 처리하는 30B-Chinchilla-quality model은 data를 2.84배 더 쓴 13.6B parameter model로 맞출 수 있고, total FLOP가 28% 적다(§2, 그림 1). Dollar cost에서는 차이가 더 크다. 2T inference token에서 Chinchilla-70B는 FLOP-optimal model보다 FLOP가 1.3%만 더 필요하지만 cost-optimal model보다 비용이 36% 더 든다. 저자들은 generated token의 model FLOPs utilization이 training 때보다 50배 낮기 때문이라고 설명한다(§6).
- 150M~6B의 MPT-style model 47개를 parameter당 10~10,000 token으로 학습했다. 10,000 token/parameter에 도달한 것은 150M model뿐이다. Loss는 10,000까지 plateau를 보이지 않았고, world knowledge, commonsense reasoning, reading comprehension, language understanding, symbolic problem solving task를 동일 가중치로 평균한 Gauntlet average도 ratio와 함께 계속 상승했다(§3, §4, 그림 3). 더 극단적인 run을 차례로 포함하여 Eq. 2를 재적합하면 curve는 평평해졌다. \(\alpha\)는 0.08에서 0.18로, \(\beta\)는 0.13에서 0.24로 변했다. 저자들의 결론은, 통상적인 ratio에서 적합한 law는 아주 높은 ratio에서 extra token이 줄이는 loss를 과대평가한다는 것이다(§5, 표 1).
- Gadre et al.은 \(\alpha = \beta\), \(C = 6ND\)일 때 Eq. 2를 다음처럼 재매개변수화한다.

```
L(C, M) = E + (a · M^η + b · M^−η) · C^−η                             (Eq. 4)
```

여기서 \(\eta = \alpha/2\), \(a = A·6^\eta\), \(b = B·6^\eta\)다([[overtraining-downstream-scaling]] §2.2). Exponent \(\eta\)는 \(M\)에 의존하지 않는다. 그러므로 서로 다른 multiplier의 run은 log-log plot에서 compute에 대한 reducible loss가 평행선이 된다(그림 2). \(2.4 × 10^{19}\) FLOP run으로 적합한 결과는 900B token을 쓴 1.4B model(\(M = 640\))의 C4 loss를 relative error 0.7% 이내로 예측했다(§4, 표 1). \(M = 5\)는 trend에서 벗어났고, 10~80 multiplier는 모두 frontier 근처의 점이었다. 따라서 저자들은 \(M^*\)가 하나의 정확한 값이 아니라 “범위에 놓일 수 있다”고 말한다(§4, 그림 9).

**Eq. 4의 계산 예시** (설명을 위한 값이며, 어떤 source도 적합한 값이 아니다). \(\eta = 0.15\), \(b/a = 20^{0.3} = 2.456\)으로 둔다. 그러면 offset \(aM^\eta + bM^{-\eta}\)의 minimum은 \(M = 20\)에 놓인다. \(a = 1\)이면 offset은 \(M = 20\)에서 3.135, \(M = 320\)에서 3.410이다. 모든 \(C\)에서 320-multiplier run의 reducible loss는 8.8% 더 크다. Reducible loss가 \(C^{-0.15}\)에 비례하므로 같은 reducible loss를 맞추려면 \(1.088^{(1/0.15)} = 1.75\)배 compute가 필요하다. Eq. 4 아래에서 이 \(M^*\)보다 16배 더 학습하는 비용은 어느 budget에서나 일정한 compute 배수이며, \(C\)와 함께 커지는 factor가 아니다. Eq. 4는 \(\alpha = \beta\)를 가정한다. Gadre et al.은 이 가정이 없어도 multiplier를 compute-optimal \(M^*\)에 상대적으로 측정하면 exponent가 과학습으로 변하지 않음을 보인다. 다만 그 경우 \(M^*\) 자체는 \(C\)에 따라 달라진다(부록 B, 식 8).

**실무.** Llama 3.1 405B는 15.6T token, \(3.8 × 10^{25}\) FLOP를 사용했다. 이는 parameter당 38.5 token이며(도출됨), 저자들은 그 size를 approximately compute-optimal이라고 설명한다. 더 작은 model들은 “compute-optimal보다 훨씬 오래” 학습했지만, 8B와 70B의 size별 token 수는 보고되지 않았다([[llama-3]] §1; [[llama-3-recipe]]). SmolLM2는 1.7B를 약 11T token, 360M을 4T, 135M을 2T token으로 학습했다([[smollm2]] 초록, §6). Nominal size에서 도출하면 각각 parameter당 약 6,500, 11,100, 14,800 token이다. DeepSeek LLM 7B는 2.0T token을 사용했다([[deepseek-llm]] 표 2). 그 논문의 자체 Eq. 4는 해당 model의 compute에서 \(D_{opt} = 5.8316 · (8.46 × 10^{22})^{0.4757} = 470B\) token을 준다. 여기서 \(C = M·D\), 그리고 §1의 식으로 구한 token당 \(M = 42.3B\) FLOP다. 따라서 7B는 자신의 law가 주는 \(D_{opt}\)의 4.3배 token으로 학습했다(도출됨). Report는 Eq. 4에서 \(C\)와 \(D\)의 unit을 인쇄하지 않았으며, \(M_{opt}·D_{opt} = 1.000·C\)이므로 FLOP와 token이라는 unit을 추론한 것이다.

**조건과 한계.** Sardana et al.은 demand가 model size에 의존하지 않고 training 전에 추정할 수 있다고 가정한다(§2). Sardana et al.과 Gadre et al. 어느 쪽도 post-training quality를 측정하지 않았다. Gadre et al.은 이를 open question으로 열거한다(§6). 그 측정은 §8에서 다룬다.

[compute-allocation-explorer.html](figures/compute-allocation-explorer.html) figure에서 budget, coefficient set(printed, refit, Sardana analysis 값), \(D_{inf}\)를 선택하고, loss-optimal plan과 inference-aware plan을 비교할 수 있다. 두 번째 preset은 Sardana et al. 그림 1의 예, 즉 6.97B parameter와 total FLOP 17.4% 감소를 재현한다.

## §6 Loss로 downstream accuracy 예측하기

**정의.** **Downstream scaling law**는 compute 또는 text에서 측정한 loss를 benchmark score로 대응시키는 함수다. **Bits per byte**는 string의 negative log-likelihood를 bit 단위로 구한 뒤 string의 byte 길이로 나눈 값이다. Token 수가 아니라 byte 수로 normalize하면 tokenizer가 달라져서 생기는 효과를 줄일 수 있다([[task-scaling-model-ladders]] §2.2).

**문제.** Benchmark accuracy는 item별 likelihood의 step function이다. 정답 choice가 가장 높은 likelihood를 가질 때에만 그 item의 점수는 1이 된다. 따라서 accuracy는 likelihood 자체보다 compute와의 상관이 약하다([[predicting-downstream-elusive]] §4, 그림 3). 그런데 범용 model은 accuracy로 비교한다.

**메커니즘: 연결된 예측.** 아래 세 연구는 모두 두 단계를 사용한다.

1. Compute(또는 N과 D)로 intermediate loss를 예측한다.
2. Saturation하는 fit으로 그 loss에서 accuracy 또는 error를 예측한다.

**Gadre et al.**은 17개 task에서의 평균 top-1 error \(Err\)를 사용한다.

```
Err(L) = ε − k · exp(−γ · L)                                        (Eq. 5)
```

여기서 \(L\)은 C4 validation loss이고, \(ε\), \(k\), \(γ\)는 적합한 값이다. \(Err\)의 상한은 \(ε\)다. 138B RedPajama token으로 학습한 6.9B model에서 이 연결 예측의 relative error는 0.05% 이내였다. 그러나 Eq. 5 fit에서 유일한 1.4B run 하나를 빼자 error가 10.64%로 올랐다([[overtraining-downstream-scaling]] §4). 같은 model에 대해 보고한 개별 task 네 개의 relative error는 RedPajama에서 ARC-Easy 5.21%부터 HellaSwag 25.73%까지였고, RefinedWeb에서는 HellaSwag가 최대 81.96%였다(표 2).

**Llama 3.** Report는 scaling-law model을 \(10^{22}\) FLOP까지 학습하여, FLOPs에 대해 정답 answer의 normalized negative log-likelihood를 선형으로 적합한다. 이어 그 likelihood에서 accuracy까지는 scaling-law model과 Llama 2 model을 사용해 sigmoid를 적합한다. ARC Challenge에서는 네 order of magnitude에 걸친 forecast가 405B 결과를 “only slightly underestimates”한다고 보고한다([[llama-3]] §3.2.1, 그림 4). Report에는 수치 error가 없다.

**Model ladders.** Model ladder는 target model과 같은 architecture와 data mixture로 학습한 작은 model들의 고정 집합이다. 여러 model size와 token multiplier를 포함하며, 그 model들의 loss와 accuracy가 fitting point가 된다([[task-scaling-model-ladders]] §2.1, §3).

```
Step 1:  L(N, D) = A / N^α + B / D^β + E                  (task loss in bits per byte)
Step 2:  Acc(L)  = a / (1 + e^(−k·(L − L₀))) + b
```

\(A\), \(B\), \(α\), \(β\), \(E\)는 task마다 16개 ladder model에 적합한다. Ladder model은 190M~1.3B parameter, \(20·N\) token의 1×~10×, \(5.2 × 10^{21}\) FLOP 범위를 가진다. \(a\), \(b\), \(k\), \(L_0\)는 약 1,400개 checkpoint point에 적합한다. OLMo 2 7B-4T와 13B-5T의 연결 예측은 8개 task에서 average absolute error가 각각 3.8점과 4.2점이었다. MMLU는 7B에서 49.0에 대해 48.4로 예측했고, ARC-Challenge는 61.9에 대해 51.5로 예측했다(§3.3, 그림 2). 가장 큰 ladder model의 마지막 10개 checkpoint에서 task loss의 standard deviation(SD10)은 step-2 error와 상관관계가 있었다. 7B-4T에서는 \(r = 0.821\)이다(§5).

**Error propagation 계산 예시** (설명을 위한 값). \(a = 0.6\), \(b = 0.25\), \(k = −10\), \(L_0 = 1.0\)으로 둔다. \(k\)가 음수이므로 loss가 커지면 accuracy는 떨어진다. \(L = 1.00\)일 때 \(Acc = 0.6/2 + 0.25 = 0.550\)이다. Step 1이 task loss를 5% 과대예측하여 \(L = 1.05\)가 되면, \(Acc = 0.6/(1 + e^{0.5}) + 0.25 = 0.6/2.649 + 0.25 = 0.477\)이다. Loss 5% error가 accuracy 7.3점 error가 된다. 위 ARC-Challenge miss도 이 방향이다. Step 1이 해당 task loss를 과대예측했고, step 2가 accuracy를 과소예측했다([[task-scaling-model-ladders]] §4).

**조건과 한계.** Ladder prediction은 annealing과 post-training 이전의 base checkpoint, 그리고 multiple-choice task만 다룬다(§8). **Decision accuracy**는 small scale에서의 corpus pair ranking이 target scale에서의 ranking과 일치하는 corpus pair의 비율이다. DataDecide는 pre-training corpus 25개와 최대 1B model을 사용했다. Corpus를 single small model size 하나로 ranking하면 1B에서의 winner와 약 80%의 pairwise comparison에서 일치했다. Scaling-law baseline 8개 중 어느 것도 이런 single-scale ranking이 만든 compute-to-decision-accuracy frontier를 넘지 못했다([[datadecide]] §3.2). 따라서 두 data recipe 중 무엇을 선택할지 결정하는 문제에서는 small scale ranking이 extrapolate한 law만큼 유용할 수 있다.

## §7 다운스트림 예측이 실패하는 지점

**실패 1: 지표가 정보를 버린다.** Schaeffer et al.은 다섯 model family 각각에서, 각 benchmark item마다 한 model family의 checkpoint들을 가로질러 compute와 score의 상관을 계산한다([[predicting-downstream-elusive]] §3–§4). ARC-Challenge에서는 정답 choice의 log-likelihood에 대해 약 90% sample의 Spearman(순위) 상관이 0.75보다 크다. answer choice들 사이에서 다시 normalize하면 이 비율은 40%가 되고, Accuracy를 쓰면 상관은 더 낮아진다(그림 3). 그 메커니즘은 Accuracy가 특정 오답 choice들에 놓인 probability mass에 좌우된다는 것이다(§5). 계산 예시(§5): 4지선다 item에서 \(p(정답) = 0.4\)일 때, 남은 0.6을 각 오답에 0.2씩 나누면 Accuracy는 1이다. 반대로 0.6을 오답 하나에 모두 놓으면 Accuracy는 0이다. 두 경우 정답 확률은 완전히 같다.

**실패 2: 많은 task는 loss에 따라 매끄럽게 scaling하지 않는다.** Lourie et al.은 Gadre et al.의 task 46개를 다시 분류했다. 18개(39%)는 validation loss와 함께 예측 가능하게 개선되었지만, 나머지는 inverse, nonmonotonic, noisy, trendless이거나 breakthrough를 보였다([[scaling-laws-unreliable-downstream]] §4, 그림 1). C4 validation loss를 쓰면 C4와 RedPajama pre-training은 하나의 HellaSwag curve를 따른다. 그러나 code validation set을 쓰면 더 나쁜 loss의 C4가 더 좋아 보이고, CoQA에서는 순서가 뒤집힌다(§3, 그림 3). CommonsenseQA는 Gadre et al. 설정(LLM Foundry harness)에서는 nonmonotonic하게 scaling하지만, DataDecide 설정(OLMES)에서는 깔끔하게 scaling한다. 두 설정은 model, prompt, shot 수, answer-choice 수도 서로 다르다(§5, 부록 A).

**실패 3: 작은 scale에는 능력 자체가 없다.** Sardana et al.에서 symbolic problem solving은 150M부터 6B까지의 모든 model에서 거의 0에 머물렀다([[beyond-chinchilla-inference-scaling]] §4, 부록 D). Ladder 연구에서 multiple-choice 형식으로 답하는 능력, 즉 prompt에 option들을 나열하고 model이 answer letter를 출력하는 능력은 수십억 parameter가 될 때까지 나타나지 않는다. 그래서 ladder는 각 option의 likelihood를 채점하는 ranked-classification 형식을 쓴다(§2.2 각주 2). Du et al.은 자신의 corpus에서 training loss가 약 2.2까지 내려가기 전에는 MMLU, C-Eval, GSM8K, GSM8K-Chinese가 random level에 머문다고 보고한다([[emergence-loss-perspective]] §3.1). 이 사례들은 ch-08b에서 다룬다.

**실패 4: 같은 loss라도 transfer는 다르다.** Liu et al.은 probabilistic context-free grammar(PCFG), hidden Markov model(HMM), OPT-125M이 생성한 text로 masked language model을 pre-train했다. 한 data set에는 LSTM도 사용했다. Model들이 같은 pre-training loss에 도달한 경우에도, 더 큰 model은 linear-probe accuracy가 각각 6.9%, 4.5%, 2.0% 높았다. Linear probe는 frozen representation 위에 학습한 linear classifier다([[same-loss-better-downstream]] §3). 따라서 이 설정에서 loss는 transfer에 대한 충분통계량이 아니다. 상태: 단순화한 data와 masked-language-model pre-training에만 해당하는 **결과(단일 연구)**다.

**실패 5: 평가 잡음.** OLMES task들을 가로질러, 60M–750M에서 benchmark의 signal-to-noise ratio, 즉 model 간 spread를 checkpoint-to-checkpoint variation으로 나눈 값은 1B ranking에 대한 그 benchmark의 decision accuracy와 상관을 보였다(\(R = 0.791\)). 또한 bits per byte로 바꾸자 30-task 평균 decision accuracy는 77.0%에서 83.7%로 올랐다([[signal-and-noise-eval]] §4.1, 그림 6).

**함의.** 작은 run으로 frontier model의 task별 accuracy를 예측할 수 있는지는 **열린 질문**이다. 측정된 부분은 더 제한적이다. 많은 task의 평균과 bits per byte로 측정한 task loss는 task별 accuracy보다 더 잘 예측되며, 예측은 hold-out한 더 큰 run으로 검증해야만 한다.

## §8 과도한 pre-training이 plasticity에 주는 비용

**정의.** 여기서 **plasticity(가소성)**는 base model이 이미 획득한 능력을 잃지 않으면서 post-training으로 얼마나 더 좋아질 수 있는지를 뜻한다. Springer et al.은 pre-training token 수가 많아질수록 post-trained model의 quality가 낮아지는 현상을 **catastrophic overtraining(파국적 과학습)**이라고 부른다([[overtrained-lms-harder-to-finetune]] §3).

**공개 checkpoint에서 나온 증거.** OLMo-1B의 intermediate checkpoint들을 Anthropic-HH로 instruction tuning했다. 각 checkpoint마다 learning rate를 따로 tuning했다. 3T-token base는 2.3T-token base보다 AlpacaEval response rate가 최대 3% 낮고 ARC도 2% 낮았다. 그 결과는 base model 성능이 token 수와 함께 단조롭게 좋아졌음에도 1.5T-token base 수준까지 내려간 것이다(§2, 그림 2). Instruction tuning과 LLaVA multimodal tuning 모두에서 2.5T token을 넘은 뒤 degradation이 나타났다(§3.1). 같은 setup으로 3T token까지 시험한 OLMo-2-7B에서는 이 현상이 관찰되지 않았다. 본문 §3.1은 이를 “OLMo-7B”라고 쓰며, Appendix E는 OLMo-2-7B checkpoint를 명시한다. OLMo-1B checkpoint의 nominal ratio는 parameter당 각각 2,300과 3,000 token이고, 3T token OLMo-2-7B는 약 430 token/parameter다(도출됨). 저자들은 한 run에서 나온 checkpoint들이 final learning rate가 서로 다르다는 confound도 명시한다(§3.2). 따라서 token 수만 달라진 완전한 통제 비교는 아니다.

**통제한 run에서 나온 증거.** 15M~90M parameter model을 C4에서 4B~128B token 동안 pre-train했고, 각 run의 learning rate는 zero까지 anneal했다(§3.2). 30M model의 128B token은 parameter당 약 4,300 token이다(도출됨).

1. 고정된 scale \(\gamma\)의 Gaussian noise를 parameter에 더하면, 더 많은 token으로 학습한 model일수록 perplexity가 더 크게 올랐다. 저자들은 이를 “progressive sensitivity”라고 부른다. Noise를 준 뒤의 perplexity는 token 수에 대해 U자형이었다(§3.3, 그림 3).
2. GSM8k, Starcoder-Python, SIQA, 그리고 세 classification data set에서 고정 learning rate로 fine-tuning했을 때도 C4 perplexity는 같은 U자형 변화를 보였다. Fine-tuning learning rate가 클수록 turning point는 더 적은 pre-training token에서 나타났다(§3.4.1, 그림 4–5).
3. Fine-tuning task마다 learning rate를 tuning한 뒤에도 C4 perplexity는 GSM8k, Starcoder-Python, MR, RTE에서 나빠졌다. RTE와 TREC에서는 in-domain perplexity도 나빠졌다(§3.4.2, 그림 6).

**이론.** Feature를 점진적으로 배우는 two-layer linear network에서, pre-training task와 fine-tuning task가 misaligned하면 unregularized fine-tuning 뒤의 pre-training loss는 pre-training 시간이 충분히 길어졌을 때 다시 상승한다. Fine-tuning weight를 pre-trained weight 쪽으로 끌어당기는 regularization은 이 turning point를 늦추지만 fine-tuning loss는 높인다(§4, Theorem 4.7).

**조건과 한계.** 실제 model 증거는 두 family의 세 model, OLMo-1B, OLMo-2-7B, LLM360-Amber 7B에 한정된다. Tokens per parameter와 fine-tuning loss 사이의 fitted relation은 없다. Pre-training 뒤에 RL을 적용한 경우도 시험하지 않았다(§3, §6). 상태는 **결과(단일 연구)**다. 이 결과는 7B 이상 scale에서 over-training이 해롭다고 증명하는 것이 아니다. Training compute를 줄이는 대신 inference compute를 늘리는 pre-training 결정을 할 때, base model만 보고 판단하지 말고 post-training 뒤에도 확인해야 함을 보인다. Forgetting을 측정하는 구체적 protocol은 ch-30a에 있다.

## §9 배분 계산 예제: 하나의 budget, 세 가지 목표

Budget (C = 1.23 × 10^{22}) FLOPs를 둔다. 이는 Chinchilla Table 3에서 100억 parameter와 2,051억 token에 해당하는 행이다. 즉,

\[
6 × N × D = 6 × 10^{10} × 2.051 × 10^{11} = 1.23 × 10^{22}.
\]

이 예제는 Approaches 1–2와 일관되는 Besiroglu et al.의 refit(Eq. 3)을 사용한다. 아래 plan의 모든 수치는 그 식에서 도출했으며, 그림의 첫 preset으로 재현할 수 있다.

**목표 1: pre-training loss를 가장 낮게 만들기.**

- Refit에 Eq. 4를 적용하면 (N_{opt} = 101억), (D_{opt} = 2,038억) token, parameter당 20.3 token, (L = 2.1294)가 나온다. 손으로 (N = 100억), (D = 2,051억)을 대입하면 parameter 항은 (482.01/10^{(10·0.3478)} = 0.1603), data 항은 (2085.43/(2.051 × 10^{11})^{0.3658} = 0.1518)이다. 따라서 (L = 1.8172 + 0.1603 + 0.1518 = 2.1293)이다. 이 점이 optimum보다 0.0001 낮은 것은 (6 × 10^{10} × 2.051 × 10^{11} = 1.2306 × 10^{22})로, 반올림한 budget보다 0.05% 크기 때문이다.
- 같은 budget에 Hoffmann et al. 논문에 인쇄된 coefficient를 쓰면 56.7억 parameter와 3,620억 token(parameter당 63.9 token)이 나온다. (C)를 (C_{min})으로 간주한 Kaplan et al.의 조정 law, (N = 1.3 × 10^9 · (C / (8.64 × 10^{19}))^{0.73})는 485억 parameter와 423억 token을 준다. 이 budget에서 세 published fit의 (N)은 최대 8.6배 차이 난다.
- IsoFLOP curve는 최솟값 근처에서 평평하다. (N_{opt})의 절반과 두 배에서 refit loss는 각각 2.1389와 2.1390으로, 최소값보다 0.45% 높을 뿐이다. Llama 3도 더 큰 budget에서 같은 평탄화를 보고했고, 이를 근거로 law가 예측한 402B 대신 405B를 선택했다([[llama-3]] §3.2.1).

**목표 2: 같은 loss에서 lifetime cost를 최소화하기.** (D_{inf} = 2 × 10^{12}) token을 추론할 예정이라고 둔다.

- §5의 절차를 수치적으로 풀면 (N = 44.5억), (D = 6,480억) token(parameter당 146 token)이다. Training compute는 (1.73 × 10^{22})로 (C)보다 40.7% 크고, inference는 (1.78 × 10^{22})이며, 합계는 (3.51 × 10^{22})이다. Loss-optimal plan의 비용은 (1.23 × 10^{22} + 2 × 1.006 × 10^{10} × 2 × 10^{12} = 5.25 × 10^{22})이므로, 합계 lifetime cost는 33.2% 낮다.
- (N = 50억)에서 손계산으로 확인할 수 있다. Parameter 항은 (482.01/(5 × 10^9)^{0.3478} = 0.2041)이다. Data 항은 (2.1294 − 1.8172 − 0.2041 = 0.1081)이어야 하므로, (D = (2085.43/0.1081)^{(1/0.3658)} = 5.19 × 10^{11}) token이다. 합계는 (6 × 5 × 10^9 × 5.19 × 10^{11} + 2 × 5 × 10^9 × 2 × 10^{12} = 1.56 × 10^{22} + 2.00 × 10^{22} = 3.56 × 10^{22})로 수치적 최솟값에 가깝다.
- Parameter당 146 token은 Sardana et al.이 시험한 범위(최대 10,000, loss plateau 관찰 안 됨)와 Gadre et al.이 시험한 범위(최대 640, (M = 640)에서 C4 loss 예측 오차 0.7% 이내) 안에 있다. 단, refit 자체는 Hoffmann et al.의 run에서 나온 것이다. Sardana et al.은 parameter당 100 token 이하로 제한한 fit이 더 높은 ratio에서 extra token이 주는 loss 감소를 과대평가한다고 발견했다(§5, Table 1). 따라서 6,480억-token 수치는 실제로 필요한 token 수를 과소추정했을 수 있다.

**목표 3: fine-tunability.** 이 목표에 대한 law를 제공하는 source는 없으므로, formula가 아니라 절차를 쓴다(**해석, 이 강의**).

1. Plan의 ratio 146을 계산하고, 유일한 측정값과 비교한다. OLMo-1B에서는 nominal parameter당 2,500 token 이후 degradation이 있었고, OLMo-2-7B에서는 약 430까지 관찰되지 않았다([[overtrained-lms-harder-to-finetune]] §3.1).
2. 이 측정값은 다른 model size에서 나왔으므로 threshold를 그대로 옮기지 않는다. Branch할 수 있는 schedule, 즉 constant learning rate 뒤에 cooldown을 두는 schedule로 학습한다([[cooldown-scaling-beyond-fixed-durations]] §3.2). Token budget 두 개에서 annealed checkpoint를 만들고, 두 checkpoint에 같은 SFT recipe를 적용한 뒤 held-out suite를 paired interval로 비교한다(ch-30a).
3. 더 긴 run의 post-trained model이 held-out suite에서 더 나쁘지 않을 때만, 더 긴 run을 유지한다.

## 레시피

아래 행은 이 장에서 다룬 scaling 연구와 공개 model이 사용한 설정을 인용한다. Excerpt 행은 2026-09-15에 원문 위치에서 읽었고, card 행은 2026-09-14에 검증했다.

| Model(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Kaplan et al. 연구 model([[kaplan-scaling-laws]]) | embedding 제외 768M~1.5B | pretrain-stable | optimizer; step; batch; schedule | Adam(1B 초과는 Adafactor); \(2.5 × 10^5\) step; 512 sequence × 1024 token; 3,000-step linear warmup, cosine으로 0까지 | arXiv:2001.08361v1 §2.2, §3 | 2026-09-15 검증 | “convergence에서의 결과는 learning-rate schedule과 대체로 무관” (§2.2); table 없음 |
| Chinchilla Approach-1 grid([[chinchilla-compute-optimal]]) | 70M~10B 초과 | pretrain-decay/anneal | 최대 LR; decay; schedule 길이 | \(2 × 10^{-4}\)(최소)~\(1.25 × 10^{-4}\)(최대); cosine, 10× decay; cycle 길이는 step 수와 일치 | arXiv:2203.15556v1 App. D.1, App. B | 2026-09-15 검증 | Fig. A1: cycle이 run보다 25% 넘게 길면 loss가 눈에 띄게 악화 |
| Chinchilla 70B | 70B | pretrain-stable | token; 최대 LR; batch(token); optimizer | 1.4T; \(1 × 10^{-4}\); 1.5M → 3M, 중간에 두 배; AdamW | Table 1, Table 4, §4.1 | 2026-09-15 검증 | size는 Approaches 1–3(§4의 40B–70B 범위)에서; AdamW는 더 낮은 LM loss와 더 좋은 fine-tuned 성능을 냄(§4.1, App. G) |
| Gopher 280B | 280B | pretrain-stable | token; 최대 LR; batch(token) | 300B; \(4 × 10^{-5}\); 3M → 6M | Table 1, Table 4 | 2026-09-15 검증 | ablation 보고 없음(baseline) |
| Porian et al. OpenLM grid([[resolving-scaling-discrepancies-recipe]]) | 5M–901M | pretrain-stable | warmup; LR·batch law; \(β_2\) | warmup token = \(N\); \(BS = 0.00037·N^{0.703}\); \(LR = 3.7·N^{-0.36}\); Table 4 batch ≤ 192이면 \(β_2=0.99\), 초과하면 0.95 | arXiv:2406.19146v4 §3.3, §3.5, Fig. 3, Table 4 | 2026-09-14 검증 | App. F Table 5의 108M warmup ablation; 901M에서 지정 LR·batch가 13개 configuration 중 최저 loss(App. G.5 Table 7) |
| DeepSeek LLM 7B([[deepseek-llm]]) | 7B | pretrain-stable | token; batch(sequence); context; peak LR | 2.0T; 2304; 4096; \(4.2 × 10^{-4}\) | arXiv:2401.02954v1 Table 2 | 2026-09-15 검증 | §3 law에서 선택(Table 2 caption) |
| DeepSeek LLM 67B | 67B | pretrain-stable | token; batch(sequence); context; peak LR | 2.0T; 4608; 4096; \(3.2 × 10^{-4}\) | Table 2 | 2026-09-15 검증 | §3 law에서 선택(Table 2 caption) |
| DeepSeek LLM 7B·67B | 7B, 67B | pretrain-stable | optimizer; schedule; clip | AdamW \(β_1=0.9\), \(β_2=0.95\), weight decay 0.1; 2,000 warmup step; token 80% 뒤 LR을 최대의 31.6%, 90% 뒤 10%로; clip 1.0 | §2.3 | 2026-09-15 검증 | Fig. 1a(1.6B, 100B token): multi-step final 성능은 cosine과 “essentially consistent” |
| DeepSeek scaling-law run | \(10^{17}\)–\(3 × 10^{20}\) FLOPs | eval-gate | hyperparameter·allocation law | \(η_{opt}=0.3118·C^{-0.1250}\); \(B_{opt}=0.2920·C^{0.3271}\); \(M_{opt}=0.1715·C^{0.5243}\); \(D_{opt}=5.8316·C^{0.4757}\) | §3.1 Eq. 1, §3.2 Eq. 4 | 2026-09-15 검증 | near-optimal은 minimum loss의 0.25% 이내; \(10^{20}\) FLOPs에서 검사(Fig. 2b); 7B·67B loss 예측(Fig. 5) |
| Llama 3.1 405B([[llama-3-recipe]]) | 405B | pretrain-stable | token; compute; size 선택 | 15.6T; \(3.8 × 10^{25}\) FLOPs; law가 16.55T token에 402B를 예측했고 405B 선택 | arXiv:2407.21783v3 §1, §3.2.1 | 2026-09-14 검증 | \(6 × 10^{18}\)~\(10^{22}\) FLOPs IsoFLOP minimum; 큰 budget에서 minimum이 평평함 |
| Llama 3 scaling-law model | 40M–16B | pretrain-stable | schedule; LR; weight decay; batch | cosine, 2,000-step warmup, peak \(2 × 10^{-4}\)~\(4 × 10^{-4}\), peak의 0.1까지 decay; weight decay = 0.1 × LR; batch 250K–4M(unit 미인쇄) | §3.2.1 | 2026-09-15 검증 | n/a |
| Llama 3.1 8B, 70B | 8B, 70B | pretrain-stable | token | size별 미보고; “compute-optimal보다 훨씬 길게” 학습 | §1 | 미보고(body와 Table 1 확인, card 기준) | n/a |
| SmolLM2-1.7B([[smollm2]]) | 1.7B | pretrain-stable | token; schedule; batch | 약 11T token; WSD, 2,000 warmup step, peak \(5.0 × 10^{-4}\), step의 10%에 걸쳐 0까지 decay; batch당 2M token | arXiv:2502.02737v1 Abstract, §4.1, App. A Table 6 | 2026-09-15 검증 | WSD는 “fixed training duration을 설정하지 않기 위해” 선택(§4.1); token-budget ablation 미보고 |
| SmolLM2-360M; SmolLM2-135M | 360M; 135M | pretrain-stable | token | 4T; 2T | §6 | 2026-09-15 검증 | data ablation을 목표 길이에서 다시 실행(§6) |
| Gadre et al. testbed([[overtraining-downstream-scaling]]) | 0.011B–6.9B | pretrain-stable | multiplier; sweep LR; optimizer; sequence length | \(M ∈ \{5, …, 640\}\); sweep LR \(3e^{-3}\); AdamW \(β_2=0.95\), 독립 weight decay \(1e^{-4}\), z-loss \(1e^{-4}\), cosine으로 \(3e^{-5}\); 2,048 | arXiv:2403.08540v2 §3.2, App. C | 2026-09-15 검증 | \(M=20\)에서 435-model grid search(§3.2) |
| Model ladder([[task-scaling-model-ladders]]) | 190M–1.3B, 16 model | eval-gate | token multiplier; compute | \(20·N\) token의 1×, 2×, 5×, 10×; 총 \(5.2 × 10^{21}\) FLOPs | arXiv:2412.04403v2 §2.1 | 2026-09-14 검증 | App. B.3: 더 큰 size를 추가하면 더 길게 학습하는 것보다 error가 더 낮아짐 |

**작은 범용 run의 출발점.** RefinedWeb에서 parameter당 약 20 token으로 학습하는 5M~901M parameter decoder-only model에는, 검증된 Porian et al. 행을 사용할 수 있다. 즉 warmup token은 \(N\), learning rate와 batch size는 \(BS=0.00037·N^{0.703}\), \(LR=3.7·N^{-0.36}\)이며, 이는 901M parameter와 약 14B token까지 검증되었다. 새 data set에서 \(N\)과 \(D\)를 선택하려면 검증된 Chinchilla 행의 IsoFLOP protocol을 쓴다. 각 budget에서 \(N\)을 바꾸고, cosine cycle을 run의 token 수와 같게 하며 10× decay를 두고, cycle이 run보다 25% 넘게 길어지지 않게 한다. 최종 token budget이 바뀔 수 있다면 SmolLM2-1.7B는 11T token에서 2,000 warmup step과 step의 10%에 걸친 decay를 가진 WSD schedule을 썼다. 다만 source는 그 선택이 loss에 미친 효과를 보고하지 않았다.

## 일반화 관점

**(a) Breadth를 넓히는 것.**

- 같은 compute에서 Kaplan 배분에서 equal scaling으로 옮기자 5-shot MMLU가 60.0%(Gopher 280B, 300B token)에서 67.6%(Chinchilla 70B, 1.4T token)로 올랐다([[chinchilla-compute-optimal]] Table 6). 상태는 **결과(단일 연구)**다. 두 run은 optimizer(AdamW 대 Adam), tokenizer normalization, data-subset 비율도 다르다(§4.1).
- Parameter당 token을 늘리면 150M에서 10,000 token/parameter까지 다섯 downstream category 평균이 계속 올랐고([[beyond-chinchilla-inference-scaling]] §4, Fig. 3), 더 낮은 C4 loss는 17-task 평균 error의 더 낮은 값도 예측했다([[overtraining-downstream-scaling]] §4).
- 더 높은 quality data는 DeepSeek의 optimal allocation을 model size 쪽으로 이동시켰다(\(a\)가 0.450에서 0.524; [[deepseek-llm]] Table 4). 해석하면 data quality는 각 token의 가치를 바꾼다. ch-09와 ch-10a는 data composition과 quality selection을 직접 측정한다.

**(b) Narrowing 또는 forgetting을 일으키는 것.**

- 1B base model을 2.5T token 너머까지 over-train하면 한 family에서 post-trained AlpacaEval과 ARC 점수가 내려갔다([[overtrained-lms-harder-to-finetune]] §2–§3.1).
- 하나의 validation corpus로 data mix를 고르면 task에 잘못된 corpus를 고를 수 있다. CoQA에서 C4와 RedPajama의 순서는 validation set에 따라 뒤집혔다([[scaling-laws-unreliable-downstream]] §3).
- 평균 forecast는 task별 실패를 숨길 수 있다. C4에서 평균 relative error가 0.14%일 때 HellaSwag relative error는 79.58%였다([[overtraining-downstream-scaling]] Table 2).

**(c) 이 단계에서의 측정 방법.**

- 하나의 validation corpus가 아니라 여러 held-out domain의 loss(ch-00 §5), 그리고 forecast용 task loss를 bits per byte로 측정한다([[signal-and-noise-eval]] Fig. 6).
- Backtest한다. 작은 run에 적합하고 가장 큰 held-out run을 예측한 뒤, task별 error와 평균 error를 보고한다([[task-scaling-model-ladders]] Fig. 2).
- Checkpoint noise는 마지막 10개 checkpoint에서 task loss의 standard deviation이다. 값이 크면 forecast할 수 없는 task라는 표시다([[task-scaling-model-ladders]] §5).
- Token budget 두 개에서 같은 SFT를 적용하고 paired held-out comparison으로 post-training probe를 한다(§9 목표 3; ch-30a).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 방법 |
|---|---|---|
| 작은 scale에서 head 또는 attention cost가 큰데 embedding 제외 \(N\)으로 적합 | \(a\) exponent가 0.7 이상(Porian et al.은 warmup·tuning 보정 전 0.835와 0.706을 얻음); 작은 model이 과대 size로 보임 | head와 attention FLOPs를 포함해 \(C\)를 다시 계산(§1)하고, 다시 적합한 \(a\)를 비교([[resolving-scaling-discrepancies]] Table 1) |
| 모든 budget에 하나의 warmup 길이와 schedule | 작은 model이 warmup 중 최저 loss에 도달; mid-schedule checkpoint로 loss envelope를 읽음 | size별 warmup 설정(Porian et al.은 warmup token = \(N\), §3.3); cosine cycle을 run 길이의 25% 이내로 유지([[chinchilla-compute-optimal]] App. B) |
| 하나의 size에서만 hyperparameter tuning | LR 또는 batch를 다시 tuning하면 exponent가 변함 | 두 개 이상 size에서 LR와 batch를 sweep하고, 전후 \(a\)를 비교(Porian step 5) |
| 반올림된 published coefficient를 복사 | Extrapolate한 \(N\) 또는 \(D\)가 논문이 직접 쓴 prediction과 다름 | 재사용 전 논문의 prediction을 재현(Gopher budget에서 40B; Llama 3에서 16.55T)(§4) |
| parameter당 20 token을 상수로 취급 | 계획이 local IsoFLOP minimum과 맞지 않음 | 자기 data에서 budget 세 개로 IsoFLOP minimum을 적합; DeepSeek의 \(a\)는 data set별 0.450~0.578 |
| compute만으로 하나의 benchmark accuracy를 한 단계에 forecast | 큰 backtest error; degenerate fit | task loss를 거치는 two-step fit, 더 큰 held-out run으로 backtest, SD10 보고([[task-scaling-model-ladders]] Table 3, §5) |
| 하나의 validation set loss로 data 선택 | validation set이 바뀌면 ranking이 뒤집힘 | 후보 mix를 두 개 이상 validation corpus와 task loss에서 평가([[scaling-laws-unreliable-downstream]] §3) |
| post-training check 없이 pre-training 연장 | base model은 좋아지나 SFT model은 좋아지지 않음 | annealed token-budget checkpoint 두 개에 같은 SFT를 적용하고 paired held-out comparison(§8, ch-30a) |
| \(D\)가 unique token을 넘을 때 repetition 무시 | training 후반의 loss가 law prediction보다 높음 | \(D\)와 unique token을 비교; 약 4 epoch 초과면 data-constrained law 적용(ch-14; [[data-constrained-scaling]] §6) |

## 이해 점검

1. Kaplan et al.은 모든 model을 \(2.5 × 10^5\) step의 하나의 schedule로 학습했다. 그 schedule의 부분 학습 run에서 loss를 읽으면 적합한 \(N ∝ C^a\) exponent가 왜 위쪽으로 bias되는지 단계별로 설명하고, Porian et al.이 schedule decay가 exponent를 약 0.03만 바꾼다고 여전히 찾은 이유를 설명하라.
2. Output head FLOPs를 세자 Porian et al.의 exponent는 0.835에서 0.706으로 내려갔다. 작은 model에는 크고 큰 model에는 작은 counting error가 \(N_{opt}\) 대 \(C\)의 intercept만이 아니라 slope를 바꾸는 이유를 설명하라.
3. Gadre et al.의 Eq. 4를 사용해, 고정 multiplier로 over-training하는 일이 모든 budget에서 일정한 compute factor를 요구하는 이유와 이 결론이 \(α\), \(β\)에 관해 어떤 가정에 의존하는지 설명하라.
4. §9에서 inference-aware plan은 loss-optimal plan보다 training compute를 40.7% 더 쓰지만 total cost를 33.2% 낮춘다. \(6ND + 2N·D_{inf}\) 중 어느 항이 절감분을 만들며, \(D_{inf}\)가 \(2 × 10^{11}\) token으로 줄면 답이 어떻게 바뀌는지 설명하라.
5. 6.9B model의 17-task 평균 error는 같은 training set에서 0.05% 이내로 예측됐지만, HellaSwag 하나는 25.73% relative error였다. §6–§7에서 평균이 개별 task보다 예측 가능한 두 mechanism을 제시하라.
6. Schaeffer et al.은 correct-answer probability가 같아도 Accuracy가 1에서 0으로 뒤집힐 수 있음을 보인다. 정답의 bits per byte 같은 continuous metric이 decision accuracy를 높이는 이유와, 그래도 generative model에 대해 측정하지 못하는 것을 설명하라.
7. Springer et al.은 instruction tuning 뒤 OLMo-1B가 2.5T token 이후 degrade했지만 OLMo-2-7B는 3T까지 그렇지 않았다고 찾았다. 이것이 15T token으로 학습한 7B model에 대해 무엇을 의미하고 무엇을 의미하지 않는지 쓰고, 자신의 run에서 답할 수 있는 가장 작은 experiment를 설계하라.

## 연결

- **이전 장:** ch-08 — Lab: Target-versus-General Capability Measurement를 갖춘 Minimal Trainer.
- **다음 장:** ch-08b — Why Next-Token Pretraining Produces General Ability: In-Context Learning, Emergence, and Predictability.
- **의존:** ch-00 — What General Capability Means and How It Is Measured(held-out suite, bits per byte, measurement noise); ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization(scaling experiment가 통제해야 할 schedule·batch mechanics).
- **이후 사용:** ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale(작은 scale fit으로 mixture 선택); ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination(\(D >\) unique-tokens 경우); ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures(공개 budget); ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control(§8–§9의 post-training probe); ch-47 — Evaluation Harness and Suite Design for General Capability(task별 대 aggregate 평가).

## 출처

- [[kaplan-scaling-laws]] — Kaplan et al. 2020 발췌: Eq. 1.1–1.8, \(N\) convention, token당 6N FLOPs, \(N ∝ C_{min}^{0.73}\), training protocol.
- [[chinchilla-compute-optimal]] — Hoffmann et al. 2022 발췌: 세 estimator, Eq. 2·Eq. 4, Table 2–4, App. B schedule-length 결과, MMLU 비교, 한계.
- [[resolving-scaling-discrepancies]] — Porian et al. 2024: exponent 보정의 단계별 과정, hyperparameter law, scope 한계.
- [[resolving-scaling-discrepancies-recipe]] — Recipe 행에서 사용한 Porian et al. training setting.
- [[chinchilla-replication]] — Besiroglu et al. 2024 발췌: coefficient 재적합, rounding·optimizer 원인, confidence-interval 분석, 4–40 ratio 범위.
- [[deepseek-llm]] — DeepSeek LLM report 발췌: \(M\) convention, hyperparameter·allocation law, data-quality exponent, 7B·67B setting.
- [[beyond-chinchilla-inference-scaling]] — Sardana et al.: inference-aware objective, 17%·28% 예, parameter당 10,000 token까지의 47 run, extreme ratio에서 fit bias.
- [[overtraining-downstream-scaling]] — Gadre et al. 2024 발췌: Eq. 4·Eq. 5, 0.7% loss·0.05% average-error forecast, task별 error, \(M^*\) 범위.
- [[llama-3]] — §3.2.1 IsoFLOP fit, 402B/16.55T extrapolation, two-step ARC Challenge forecast, 작은 model over-training.
- [[llama-3-recipe]] — Llama 3.1 token 수와 scaling-law run setting, size별 미보고 token 포함.
- [[smollm2]] — SmolLM2 report 발췌: 11T, 4T, 2T token budget과 WSD schedule.
- [[task-scaling-model-ladders]] — two-step task-loss·sigmoid forecast, 3.8·4.2-point error, SD10 predictability indicator.
- [[scaling-laws-unreliable-downstream]] — Lourie et al.: 46 task 중 18개 predictability, validation-corpus·harness reversal.
- [[predicting-downstream-elusive]] — Schaeffer et al.: metric transformation chain, 90% 대 40% correlated sample, incorrect-choice mechanism.
- [[data-constrained-scaling]] — Muennighoff et al.: common-mistakes table의 repetition 한계와 ch-14 연결.
- [[overtrained-lms-harder-to-finetune]] — Springer et al. 2025 발췌: OLMo post-training degradation, progressive sensitivity, controlled run, theory, confound.
- [[cooldown-scaling-beyond-fixed-durations]] — §9 post-training probe를 위한 branch 가능한 constant learning rate + cooldown schedule.
- [[emergence-loss-perspective]] — 여러 benchmark가 random level을 벗어나는 loss threshold(Failure 3).
- [[same-loss-better-downstream]] — 단순화 setting에서 같은 pre-training loss인데 transfer가 다름(Failure 4).
- [[datadecide]] — data decision에서 extrapolated law의 대안인 single-scale ranking.
- [[signal-and-noise-eval]] — 작은 scale decision을 위한 signal-to-noise ratio와 bits per byte(Failure 5, measurement).
