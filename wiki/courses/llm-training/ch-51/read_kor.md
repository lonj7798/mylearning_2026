# Chapter 51 — Metric 노이즈, 신뢰구간, Go/No-Go 결정

> **핵심 통찰.** 보고된 벤치마크 점수의 오차에는 적어도 네 독립 원인이 있으며, 그중 세 개는 평가 세트가 커져도 줄지 않는다. 학습 seed, 생성 샘플링, 프롬프트 형식이다. 측정 크기는 다음과 같다. 동일하게 학습한 7B 모델 10개에서 13개 벤치마크 전반 사전 학습 seed 표준편차 0.21~2.15 정확도 포인트([[benchmark-variance-quantified]] 표 1); 1B 레시피 하나의 재실행 간 표준편차 최대 2포인트([[datadecide]] §2.1); 의미가 동등한 프롬프트 형식 10개 전반의 중앙 spread 7.5 정확도 포인트, 다른 형식에서 모델 쌍의 14.1%가 순위를 뒤집고 그 뒤집힘의 76%가 양 방향 모두 통계적으로 유의함([[prompt-format-sensitivity-formatspread]] §4.2). 군집 평가에서는 표준오차가 naive 값의 3.05배일 수도 있다(DROP, [[adding-error-bars-evals]] 표 4).
>
> **지침.** 동일 항목에서 두 체크포인트를 비교할 때 score correlation과 함께 paired 표준오차 `sqrt(SE²_A + SE²_B − 2·SE_A·SE_B·Corr(s_A, s_B))`를 보고하라. pairing은 공유된 항목 난이도를 제거하며, [[adding-error-bars-evals]] §4.2의 상관 0.5 수치 예에서는 estimator 분산을 3분의 1 낮춘다. 질문이 그룹으로 올 때(하나의 passage, 한 문제가 여러 언어로 있는 경우)는 clustered 형태를 사용하라. DROP에서 naive 값은 3.05배 너무 작았다(표 4). 범용 체크포인트를 출하할지 결정할 때에는 두 검정을 요구하라. 사전 등록 target의 이득 검정과 held-out 일반 suite에서 사전 등록 margin에 대한 non-inferiority 검정이며, gain 쪽 multiplicity를 제어한다. 목표 모델보다 훨씬 작은 모델로 결정할 때는 먼저 그 규모에서 벤치마크가 그 결정을 지지할 수 있는지 점검하라([[signal-and-noise-eval]], [[datadecide]]). 지지하지 못하면 어떤 구간 폭도 구해 주지 못한다.

---

## 이 챕터가 범용 모델에 중요한 이유

파이프라인의 모든 단계(사전 학습 → 중간 학습 → SFT → 선호 최적화 → RL)는 같은 질문으로 끝난다. 이 체크포인트를 남길 것인가, 이전 것을 남길 것인가. 답은 벤치마크 숫자 표에서 읽는다. 하나의 태스크를 목표로 한 모델이면 표에는 행 하나가 있다. 범용 모델은 target 능력, 지시 이행, 지식, 코드, 안전성, 과도 거절, [[ch-50]]의 망각 slice 등 수십 행을 갖고, 결정은 conjunction이다. target은 개선되어야 *하고*, 나머지는 명시한 양보다 더 떨어지면 안 된다.

그 conjunction은 통계를 두 방식으로 바꾼다.

1. 행이 많으면 적어도 하나가 가짜 개선을 보일 확률이 빠르게 늘어난다. 독립 slice 20개를 `α = 0.05`로 검정하면 적어도 한 false positive 확률은 `1 − 0.95²⁰ = 0.642`(계산됨)다. 나중에 가장 좋아 보이는 slice를 고르는 것이 좁은 모델을 넓은 모델로 묘사하는 방식이다.
2. non-regression 쪽은 *부재* 주장이고, 부재 주장은 구간만이 아니라 power가 필요하다. “일반 suite의 paired 구간이 0을 포함한다”는 말은 구간 폭이 3포인트면 3포인트 손실과 양립한다.

이 챕터는 순서대로 구성 요소를 만든다. 노이즈가 어디서 오며 얼마나 큰지(§1), 샘플링 계획을 존중하는 구간(§2), 두 체크포인트 비교(§3), 사전 실행 크기 결정(§4), 많은 slice 동시 처리(§5), 두 부분 gate(§6), 벤치마크가 애초에 지지할 수 있는 결정(§7)이다.

---

## §1 측정 크기를 가진 네 노이즈 원천

보고 점수는 질문, 학습 seed, 생성이라는 세 population 전체에서 모델의 기대 점수를 추정한다. 네 원인이 추정치를 그 양에서 분리하며, 항목을 더 늘려 줄어드는 것은 첫 번째뿐이다.

### 1.1 항목 샘플링 노이즈

평가 세트는 유한한 질문 표본이므로, 같은 크기의 다른 표본에서는 점수가 바뀐다. n개 항목의 이진 항목 점수 평균 `s̄`에 대해 `SE_Bernoulli = sqrt(s̄(1 − s̄)/n)`이다([[adding-error-bars-evals]] 식 2). 95% 구간은 `s̄ ± 1.96 · SE`다.

**수치 예시(손계산).** 10개 항목이 `1,1,0,1,0,1,1,1,0,1`이면 `s̄ = 0.7`, `SE = sqrt(0.7 × 0.3 / 10) = sqrt(0.021) = 0.1449`, halfwidth는 `1.96 × 0.1449 = 0.284`, 구간은 `[0.416, 0.984]`다. 항목 10개는 거의 아무것도 측정하지 않는다.

**실제 벤치마크 크기의 halfwidth**(독립 항목, 항목당 생성 하나에서 `1.96·sqrt(p(1−p)/n)`로 계산됨):

| 벤치마크 | n | p = 0.5에서 | 현실적 p에서 |
|---|---|---|---|
| HumanEval | 164 | 7.65 pp | p = 0.72에서 6.87 pp |
| MATH500 | 500 | 4.38 pp | p = 0.60에서 4.24 pp |
| IFEval | 541 | 4.21 pp | p = 0.78에서 3.49 pp |
| GSM8K test | 1319 | 2.70 pp | p = 0.86에서 1.87 pp |
| MMLU | 14042 | 0.83 pp | p = 0.70에서 0.76 pp |

이 열의 spread 때문에 “2포인트 미만 이득은 노이즈” 같은 고정 규칙은 틀린다. HumanEval에서 5포인트 이득은 구간 안일 수 있고, MMLU에서 1포인트 이득은 구간 밖일 수 있다.

[figures/bootstrap-ci.html](figures/bootstrap-ci.html)의 패널 1은 n, p, 질문당 생성 수, 실행 간 표준편차를 움직이고 결합 halfwidth에서 어느 항이 지배하는지 보여 준다. 위 벤치마크 크기에 marker가 있다.

### 1.2 학습 실행 노이즈(seed와 데이터 순서)

초기화 seed 또는 데이터 순서만 다른 동일 레시피의 두 실행은 점수가 다른 체크포인트를 낸다.

[[benchmark-variance-quantified]]는 seed 외에는 동일하게 각각 210B 토큰으로 학습한 Llama-2-7B architecture 모델 10개를 훈련하고, 21개 체크포인트 전반 seed 표준편차 평균을 측정했다(표 1). HellaSwag 0.21, MMLU-Cloze 0.22, GSM8k 0.41, MMLU 0.57, ARC-C 0.80, HumanEval 1.11, COPA 2.15 정확도 포인트다. 요약은 “COPA와 HumanEval처럼 테스트 예가 적은 벤치마크는 높은 분산(seed 분산과 95% CI 모두)을 보인다. 일반적으로 7B seed 분산은 같은 벤치마크의 95% CI보다 훨씬 낮지만, 둘의 비율은 상당히 가변적이다.”다. 이는 **결과(단일 연구)**이며 사전 학습 seed만 대상이고 fine-tuning seed는 측정하지 않았다. [[datadecide]]는 각 1B 구성을 seed 세 개로 재실행해 “대부분 태스크의 일부 레시피에서 정확도 2 percentage point”에 이르는 실행 간 표준편차를 보고한다(§2.1).

[[signal-and-noise-eval]]은 값싼 proxy를 정의한다. 실행 하나의 마지막 n개 체크포인트에 대한 상대 표준편차 노이즈 `Rel.Std(m) = sqrt((1/(n−1)) Σ_i (m_i − m̄)²)/m̄`다(§3.1). `m_i`는 체크포인트 i의 점수, `m̄`는 그 평균이다. 1B 모델 마지막 30개 체크포인트는 ARC Challenge 정확도 1.7%를 가로지른다. 1B-5xC에서 seed 실행 10개와 data-order 실행 10개에 대해 seed noise, data-order noise, 전체 실행 total variation은 이 final-n 양과 각각 `R² = 0.82`, `0.86`, `0.95`로 상관한다(§3.1, 그림 7). 따라서 실행 하나가 실행 두 개가 보일 것을 추정할 수 있다. gate에 대한 함의는, 같은 가중치를 두 번 평가하는 것이 아니라 따로 학습한 체크포인트 둘을 비교할 때 항목 수준 구간만으로는 오차를 과소평가한다는 것이다.

### 1.3 생성 노이즈

생성 태스크에서는 모델을 샘플링하므로 같은 질문도 생성마다 다른 점수를 받는다. [[adding-error-bars-evals]] §3.1은 항목별 분산을 `Var(s) = Var(x) + E[σ²_i]`로 분해한다. x는 질문의 조건부 평균 점수(질문 성질), `σ²_i`는 샘플링 아래 질문 i의 조건부 분산이다. 질문당 K개 생성을 쓰면 `Var(µ̂ | K) = (Var(x) + E[σ²_i]/K)/n`이다. 논문의 수치 예(이진 점수, 균일 난이도, `Var(x) = 1/12`, `E[σ²_i] = 1/6`)에서는 `Var(µ̂|K) = Var(µ̂|K=1) × (1 + 2/K)/3`다. K=2는 분산 3분의 1, K=4는 절반을 제거하며, K가 커질 때 한계는 K=1 분산의 3분의 2다. 바닥은 `Var(x)/n`이므로 생성은 항목 샘플링 항을 결코 없애지 못한다.

temperature를 0으로 내리면 `σ²_i`는 없어지지만 측정량이 바뀌며 conditional mean 분산을 올릴 수 있다. §3.3 단일 토큰 예에서 `x_{T=1} ~ U[0,1]`이면 `Var(x_{T=1}) = 1/12`, `Var(x_{T=0}) = 1/4`다. `x_{T=1} ~ U[1/3, 1]`이면 기대 점수는 `2/3 → 3/4`, 분산은 `1/27 → 3/16`이다. 저자의 규칙은 “두 경우 모두 점수 분산을 줄이기 위해 sampling temperature를 조정해서는 안 된다”이다. greedy decoding은 측정할 정당한 조건이지 분산 축소 기법이 아니다. K·n개 생성을 K·n개 독립 관측으로 다루는 것은 일관되지 않다(§3.1). 단위는 K개 생성 위 질문당 평균이다.

### 1.4 프롬프트 형식과 harness 노이즈

같은 규칙과 descriptor에서 구성한 두 프롬프트 형식은 같은 태스크를 표현하므로, 형식 간 점수 차이는 태스크가 아니라 모델 성질이다.

[[prompt-format-sensitivity-formatspread]]은 LLaMA-2 7B/13B/70B, Falcon-7B, Falcon-7B-Instruct, GPT-3.5-Turbo에서 Super-NaturalInstructions 태스크 53개를 평가했다. 태스크당 무작위로 뽑은 형식 10개에서 중앙 spread(형식 간 최대−최소 정확도)는 7.5포인트다. 모든 LLaMA-2 설정에서 태스크 20%는 spread가 최소 15포인트이며, 여러 태스크는 70포인트를 넘고, 보고된 최대는 LLaMA-2-13B의 76포인트다. 가능한 형식 중 10개만 샘플했으므로 lower bound다(§4.2). spread는 70B, instruction tuning 뒤, 더 많은 few-shot 예시에서도 남는다.

gate 작성 방식을 바꿔야 하는 결과는 다음이다. 형식 p 아래 M이 M′보다 최소 0.02 정확도 높다고 할 때, 다른 형식 p′ 아래 M′이 M보다 최소 0.02 높을 확률은 LLaMA-2-13B 대 70B에서 0.141, LLaMA-2-7B 대 Falcon-7B에서 0.140이다. 두 모델 비교에서 “두 실험(처음 p, 이어 p′)이 모두 1000개 샘플에서 통계적으로 유의했다(p-value < 0.05)”는 비율도 각각 76%, 47%다(§4.2). 올바르게 paired이고 유의한 비교도 의미를 담지 않은 formatting 선택으로 뒤집힐 수 있으며, 그 뒤집힘도 유의할 수 있다.

같은 효과는 벤치마크 형식화에도 나타난다. [[benchmark-variance-quantified]] §3.3은 모델 집합 하나에서 MMLU 형식 두 개를 비교한다. 210B 토큰 뒤 표준 MMLU는 chance 25에 대해 25.86, monotonicity 0.09, seed 표준편차 0.57이다. MMLU-Cloze는 37.47, seed 표준편차 0.22, monotonicity 0.95에 이른다. 완전히 학습한 모델 70개 전반 두 형식은 Pearson 0.92로 상관한다. 따라서 형식들은 큰 모델 순위에는 동의하지만 작은 모델이 무엇을 할 수 있는지에는 완전히 다르게 말한다.

### 1.5 metric이 judge일 때의 judge 노이즈

LLM judge는 n으로 줄지 않는 항을 더한다. [[judge-llm-bias]] 표 2는 의도적으로 비슷한 답 쌍에서 A/B swap 아래 일관성을 측정한다. 기본 프롬프트에서 GPT-4 65.0%, GPT-3.5 46.2%, Claude-v1 23.8%이며, 판정은 각각 35.0%, 53.8%, 76.2% 쌍에서 바뀐다. 논문의 보수적 규칙은 양 순서 모두에서 답이 이길 때만 win으로 세고 나머지는 tie로 부르는 것이다(§3.4). 전체 처리는 ch-49에 있으며, 여기서 핵심은 swap-and-average가 estimator 일부이고 남은 불일치는 노이즈 budget에 들어간다는 점이다.

### 1.6 나란히 보는 budget

| 원천 | 항목을 더 늘리면 줄어드는가? | 실행 또는 생성을 더 늘리면 줄어드는가? | 측정 크기와 위치 |
|---|---|---|---|
| 항목 샘플링 | 예, `1/sqrt(n)`로 | 아니오 | p=0.5에서 0.83 pp(MMLU, n=14042)~7.65 pp(HumanEval, n=164), 계산됨 |
| 학습 seed / 데이터 순서 | 아니오 | 예, `1/sqrt(runs)`로 | 13개 벤치마크에서 σ 0.21–2.15 pp([[benchmark-variance-quantified]] 표 1); 1B에서 최대 2 pp([[datadecide]] §2.1) |
| 생성 샘플링 | 일부, `Var(x)/n` 바닥까지 | 예, 2/3 바닥까지 | 수치 예의 `Var(µ̂|K) = Var(µ̂|K=1)(1+2/K)/3`([[adding-error-bars-evals]] §3.1) |
| 프롬프트 형식 | 아니오 | 아니오 | 형식 10개 중앙 spread 7.5 pp; 순위 역전 14.1%([[prompt-format-sensitivity-formatspread]] §4.2) |
| Judge 불일치 | 아니오 | 일부, swap-and-average로 | 유사 답 쌍에서 GPT-4 swap 불일치 35.0%([[judge-llm-bias]] 표 2) |

실행당 원천이 항목 샘플링 원천을 넘으면, 항목을 늘리는 것은 계산 낭비이며 budget은 대신 실행, 생성 또는 형식에 배정해야 한다.

---

## §2 샘플링 계획을 존중하는 구간

### 2.1 점수 하나의 구간

[[adding-error-bars-evals]] 식 1–3에서 `s_i`가 질문 i 점수, `s̄`가 n개 질문 평균이면:

```
SE_C.L.T.    = sqrt( Var(s)/n ) = sqrt( (1/(n−1)) Σ_i (s_i − s̄)² / n )      (1)
SE_Bernoulli = sqrt( s̄(1 − s̄)/n )                                          (2)   [이진 점수만]
CI_95%       = s̄ ± 1.96 × SE_C.L.T.                                         (3)
```

식 2는 `s_i ∈ {0,1}`일 때 식 1의 특수 사례다. 이를 fraction 점수(F1, 부분 점수, 항목별 pass rate)에 적용하면 “보수적(너무 넓은) 경향”의 구간을 낸다. 논문은 [[llama-3]] 보고서를 그렇게 한 사례로 든다(§2.1). **결과(단일 연구, 저자 한 명의 재분석).**

### 2.2 군집 항목

reading-comprehension suite는 passage당 여러 질문을 뽑고, 다언어 suite는 문제 하나를 여러 언어로 낸다. 따라서 그룹 안 질문들은 상관하고 식 1, 3 뒤 독립 가정과 naive item bootstrap 가정이 깨진다. cluster c에서 질문 i의 점수가 `s_{i,c}`이고 cluster가 독립이라고 하면([[adding-error-bars-evals]] 식 4):

```
SE_clustered = [ SE²_C.L.T. + (1/n²) Σ_c Σ_i Σ_{j≠i} (s_{i,c} − s̄)(s_{j,c} − s̄) ]^(1/2)     (4)
```

Anthropic 모델에서 측정한 값(표 4, 저자가 fictional이 아니라고 명시)은 DROP `1.34 vs 0.44`, 비율 3.05; MGSM `1.62% vs 0.86%`, 비율 1.88; RACE-H `0.51% vs 0.46%`, 비율 1.10이다. 표 3은 DROP을 588개 cluster의 질문 9,622개로 보고하며, 질문 수 옆에 cluster 수를 출력하라고 권한다.

**수치 예시(계산됨).** 표준오차 비율 3.05는 분산 9.3배를 뜻한다. 그러므로 DROP의 질문 9,622개는 독립 질문 약 `9,622/9.3 ≈ 1,035`개 정보량을 담는다. 질문 수로 이 평가를 sizing하면 resolving power를 한 자릿수 배 과대평가한다.

### 2.3 Bootstrap이 비용을 정당화할 때

percentile bootstrap은 n개 항목 인덱스를 복원 추출로 B번 resample하고, statistic을 다시 계산한 뒤 2.5th와 97.5th percentile을 구간으로 취한다. BCa 변형은 bias factor `z₀ = Φ⁻¹(#{θ̂_b < θ̂}/B)`와 jackknife acceleration 항으로 두 percentile을 조정한다. 이것은 statistic이 skewed하거나 0 또는 1 근처에 있을 때 중요하다. `θ̂`는 전체 표본 statistic, `θ̂_b`는 replicate b에서의 값이다.

plain mean에 이것이 할 만한 일인지 두 출처가 의견이 다르고, 그 불일치는 유익하다. [[adding-error-bars-evals]] §2.1: “복잡한 샘플링 계획 또는 estimator를 사용하지 않는 한 bootstrapping은 불필요하다고 본다.” [[benchmark-variance-quantified]] §3.1은 모든 체크포인트를 bootstrap하고 “bootstrap CI와 analytic CI는 bootstrap sample 수가 크면 수렴한다”고 찾았다. 둘은 같은 방향을 가리킨다. 독립 항목 평균에는 analytic 구간이면 충분하다. estimator가 단순 평균이 아닐 때(크기가 다른 slice의 macro-average, swap-and-average 뒤 승률, pass@k) 또는 샘플링이 clustered/stratified일 때 bootstrap을 써라.

---

## §3 두 체크포인트 비교

### 3.1 항목을 pair하기

두 독립 구간을 빼면 두 체크포인트가 같은 질문에 답했다는 사실을 버린다. 질문 난이도는 공유되므로 차이에서 상쇄된다([[adding-error-bars-evals]] 식 5–7, `s_{A−B,i} = s_{A,i} − s_{B,i}`):

```
unpaired: SE_{A−B}        = sqrt( SE²_A + SE²_B )                                  (5)
paired:   SE_{A−B,paired} = sqrt( Var(s_{A−B})/n )                                 (7)
                          = sqrt( SE²_A + SE²_B − 2·SE_A·SE_B·Corr(s_A, s_B) )
```

pairing 이득은 오직 `Corr(s_A, s_B)`가 정한다. 상관 0에서는 식 7이 식 5와 같고, 논문의 균일 점수 예에서 상관 0.5이면 estimator 분산이 3분의 1 줄어든다(`1/6 → 1/9`, §4.2). 상관 1에서는 차이에 분산이 없다. 일반 multiplier가 없으므로 논문은 구간 옆에 상관을 보고하라고 권한다.

**수치 예시(손계산).** paired 항목 10개.

```
A: 1 1 0 1 1 0 1 1 0 1      s̄_A = 0.7    SE_A = sqrt(0.7·0.3/10) = 0.1449
B: 1 0 0 1 1 0 1 0 0 1      s̄_B = 0.5    SE_B = sqrt(0.5·0.5/10) = 0.1581
d: 0 1 0 0 0 0 0 1 0 0      d̄   = 0.2
Var(d) = [2·(0.8)² + 8·(0.2)²]/9 = 1.6/9 = 0.1778
SE_paired   = sqrt(0.1778/10) = 0.1333          → 95% CI  0.2 ± 0.261 = [−0.061, +0.461]
SE_unpaired = sqrt(0.1449² + 0.1581²) = 0.2145  → 95% CI  0.2 ± 0.420 = [−0.220, +0.620]
implied Corr(s_A, s_B) = (0.2145² − 0.1333²) / (2·0.1449·0.1581) = 0.616
```

이 상관에서 pairing은 구간을 1.61배 좁힌다. 어느 구간도 0을 제외하지 않는다. `n = 10`에서 관측한 20포인트 차이는 근거가 아니다.

항목이 clustered이면 차이에 직접 clustered 형식을 적용한다([[adding-error-bars-evals]] 식 8). cluster 안 쌍 전반에 `(s_{A−B,i,c} − s̄_{A−B})(s_{A−B,j,c} − s̄_{A−B})`를 합한다. 논문의 running example에서는 pairing과 clustering이 결론을 바꾼다. MATH에서는 차이가 유의하고 HumanEval과 MGSM에서는 노이즈와 구별되지 않아, 같은 표를 비공식적으로 읽은 결론을 뒤집는다(§4.2).

### 3.2 Paired bootstrap과 그 p-value가 뜻하는 것

```
# paired_bootstrap(s_A, s_B, B=10000, alpha=0.05)
#   s_A, s_B: 같은 항목 순서에서 checkpoints A, B의 n개 항목 점수 배열
d         = s_A - s_B                            # 항목별 paired 차이
delta_hat = mean(d)
boot = []
for b in range(B):
    idx = sample_with_replacement(range(n), n)   # PAIR를 resample: 인덱스 목록 하나, 두 배열 모두
    boot.append(mean(d[idx]))
boot.sort()
ci_lo, ci_hi = boot[int(0.025*B)], boot[int(0.975*B)]
# interval inversion의 p-value(Efron-Tibshirani achieved significance level이 아님):
p_approx = 2 * min(frac(boot >= 0), frac(boot <= 0))
```

마지막 줄에는 label이 필요하다. Efron과 Tibshirani의 achieved significance level은 **null-centred** bootstrap 분포 아래 계산한다. replicate를 `−delta_hat`만큼 이동해 null이 성립하게 한 뒤, 이동된 replicate 중 `delta_hat`만큼 또는 더 극단적인 것이 몇 개인지 센다. 위처럼 uncentred 분포의 tail mass를 두 배 하는 것은 percentile 구간을 inversion해서 얻는 근사다. bootstrap 분포가 거의 대칭이면 centred 구성과 일치한다. 숫자가 어느 둘에서 왔는지 명시하라.

### 3.3 Sign test

tie를 버리고 `W = #{i : s_A,i > s_B,i}`, `L = #{i : s_A,i < s_B,i}`로 둔다. 차이가 없다는 null 아래 `W ~ Binomial(W + L, 0.5)`다. 정확한 양측 p-value는 `X ~ Binomial(W + L, 0.5)`에서 `2 · P(X ≤ min(W, L))`이며 1로 cap한다. `W + L ≥ 25`이면 continuity correction을 쓴 정규 근사는 `z = (|W − L| − 1)/sqrt(W + L)`, `p = 2(1 − Φ(z))`다.

**수치 예시(손계산).** §3.1 데이터에서 `W = 2`, `L = 0`이므로 `W + L = 2`, `p = 2 · P(X ≤ 0) = 2 · (1/2)² = 0.5`다. 이 검정은 tie 8개를 버리고 관측 두 개만 남긴다. 항목 점수 크기가 비교 불가능하지만(rubric 또는 서로 다른 scale의 judge 점수) 방향은 비교 가능할 때 올바른 도구다. 대부분 항목이 tie일 때는 잘못된 도구다.

[figures/bootstrap-ci.html](figures/bootstrap-ci.html)의 패널 2는 선택한 score correlation에서 paired 데이터를 생성하고 paired 구간, unpaired 구간, sign-test p-value를 함께 보여 준다. 그러므로 고정 n에서 pairing을 무시하는 비용이 보인다.

---

## §4 실행 전 평가 크기 결정

“이 차이는 유의한가?”는 compute를 쓴 뒤에 묻는다. 계획을 바꿀 수 있는 버전은 다음이다. 행동할 가치가 있는 최소 차이에 대해 질문과 생성이 몇 개 필요한가?

[[adding-error-bars-evals]] 식 9–10에서 `ω² = Var(x_A) + Var(x_B) − 2Cov(x_A, x_B)`는 paired conditional-mean 차이의 분산, `σ²_A`, `σ²_B`는 샘플링 아래 평균 conditional variance, `K_A`, `K_B`는 질문당 생성 수, `α`는 유의수준, `1 − β`는 power, `δ`는 minimum detectable effect다.

```
n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²                (9)
δ = (z_{α/2} + z_β) sqrt( (ω² + σ²_A/K_A + σ²_B/K_B) / n )          (10)
```

**공개된 수치 예시.** 생성 분산이 없고(`σ²_A = σ²_B = 0`), `ω² = 1/9`, `δ = 0.03`, `α = 0.05`, `β = 0.20`이면 `n = (1.96 + 0.8416)² × (1/9) / 0.0009 ≈ 969`개 질문이다. 저자는 이를 새 평가에 적어도 1,000개 질문이 있어야 하는 사례로 읽는다(§5).

**질문 대신 생성의 수치 예시.** `σ²_A = σ²_B = 1/6`, `ω² = 1/9`, `Corr(x_A, x_B) = 0.5`, `n = 198`, `α = 0.05`, `β = 0.20`에서 K를 1에서 10으로 올리면 minimum detectable effect는 13.2%에서 7.5%로 움직인다(§5). 평가 세트가 고정되어 작으면 생성이 남은 레버이고, 그 수익은 §1.3의 `Var(x)/n` 바닥에서 평평해진다.

**조건과 한계.** 식 9, 10은 독립 질문을 가정한다. cluster가 있으면 같은 논문 부록 B의 cluster-adjusted variance를 대입한다. 또한 `ω²`, `σ²_A`, `σ²_B`가 알려졌다고 가정하며, 실무에서는 subset pilot run으로 추정한다는 뜻이다.

---

## §5 많은 slice, 많은 벤치마크

범용 gate는 많은 것을 한 번에 검정하며, 하나의 memo에 서로 다른 multiplicity 문제 두 개가 나타난다.

**Gain 쪽: 선택은 false positive를 부풀린다.** m개 독립 slice를 수준 α에서 검정하면 적어도 하나의 가짜 기각 확률은 `1 − (1 − α)^m`이다. `m = 20`, `α = 0.05`에서 0.642다(계산됨). 표준 보정 두 가지가 있다. **Holm–Bonferroni**는 family-wise error rate를 제어한다. m개 p-value를 오름차순으로 정렬하고 `p_(k) ≤ α/(m − k + 1)`인 동안 `p_(k)`를 기각한다. 따라서 `m = 20`, `α = 0.05`에서 가장 작은 것은 0.0025, 다음 것은 0.00263과 비교한다. 출하 결정이나 공개 숫자처럼 거짓 주장 하나의 비용이 클 때 쓴다. **Benjamini–Hochberg**는 false discovery rate를 제어한다. `p_(k) ≤ k·α/m`인 가장 큰 k에 대해 `p_(k)`를 기각한다. output이 조사할 slice의 순위 목록이고 알려진 비율의 false lead를 받아들일 수 있을 때 쓴다.

**Non-inferiority 쪽: 보정은 없지만 power 문제가 있다.** “어느 slice도 δ보다 더 퇴행하지 않았다”는 conjunction이다. intersection-union test는 모든 component test가 기각할 때만 union null을 기각하므로, 보정 없이도 type-I error는 이미 최대 α다. 어려움은 반대다. power는 slice별 power의 곱이므로, under-powered non-inferiority test 20개로 만든 gate는 어디에서도 퇴행하지 않은 체크포인트도 실패시킨다. 200개 항목씩 slice 50개보다 pooled test 하나와 사전 등록 sub-suite 몇 개로 일반 suite를 sizing하라.

**형식도 비교다.** §1.4의 순위 역전 결과는 단일 형식 유의 차이가 형식에 대한 분포의 추출 하나라는 뜻이다. 형식 집합을 미리 고정하고 저자가 권한 대로 그 전반 spread를 보고하면, 그 선택이 10개 검정에 걸친 세지 않은 selection이 되는 것을 막는다.

---

## §6 두 부분 go/no-go gate

**주장 1 — target 이득(superiority).** 실행 전에 target metric 또는 bundle을 사전 등록하고, paired 구간으로 결정하며, §5처럼 사전 등록 target slice 전반을 보정한다. paired 95% 구간이 0을 제외할 때 null을 기각한다.

**주장 2 — held-out 일반 능력에서 퇴행 없음(non-inferiority).** 일반 suite(지식, 지시 이행, 코드, 안전성, 과도 거절, [[ch-50]]의 망각 slice)와 통계가 아니라 제품 결정인 margin `δ_NI`를 사전 등록한다. `Δ = new − old`의 paired 95% 구간 하한이 `−δ_NI`를 넘으면 non-inferiority를 선언한다.

**수치 예시(손계산).** held-out 일반 suite, paired 평가에서 `Δ̂ = −0.6 pp`, `SE_paired = 0.5 pp`, margin `δ_NI = 1.0 pp`다. 구간은 `−0.6 ± 1.96 × 0.5 = [−1.58, +0.38]`이다. 0을 포함하므로 퇴행은 *탐지되지 않았다*. 하한 −1.58은 −1.0 아래이므로 1포인트 margin에서 non-inferiority는 *확립되지 않았다*. 이 둘은 다른 말이며, gate는 두 번째만 쓴다. 이를 확립하려면 `1.96·SE < 0.4`, 즉 `SE < 0.204 pp`가 필요하다. 같은 score correlation에서는 `(0.5/0.204)² ≈ 6.0×` 더 많은 항목을 뜻한다. 실행 전에 이를 찾는 것이 §4의 일이다.

**Memo template.** 한 페이지다. 복사할 부분은 섹션과 trigger line이다.

```
go-no-go-memo.md
----------------
## 1. 사전 등록 주장(평가 실행 전 작성)
C1 (이득): ckpt-round-3가 TARGET = {...}에서 ckpt-round-2보다 개선, paired mean
           difference, 사전 등록 target slice 전반 Holm 보정.
C2 (무퇴행): HELD-OUT GENERAL SUITE(knowledge / instruction following /
           code / safety / over-refusal / ch-50 forgetting slices)에서 pooled paired
           difference가 margin delta_NI = <value> pp에서 non-inferior.

## 2. C1 근거
| slice | n (clusters) | corr(s_A,s_B) | delta_hat | SE_paired | 95% CI | Holm threshold | reject? |
행마다 estimator를 명시: paired CLT, paired clustered, 또는 B를 쓴 paired bootstrap.

## 3. C2 근거
| suite (pooled) | n (clusters) | corr | delta_hat | SE_paired | 95% CI | lower bound vs -delta_NI |
구간이 0을 포함해도 하한이 -delta_NI 아래면 C2 실패.

## 4. 실제 사용한 noise budget
항목 및 cluster 수; estimator; 질문당 생성 K와 sampling temperature(§1.3에 따라 배포할 값);
비교 학습 실행 수, 하나면 출처를 가진 가정 seed sigma; 프롬프트 형식 수와 관측 spread;
judge swap-and-average 및 잔여 불일치율.

## 5. Power 진술
식 10에서 선택한 n, K, alpha = 0.05, power 0.80의 minimum detectable effect.
MDE가 주장 이득을 넘으면 실행은 C1을 지지하지 못함.

## 6. 검토자가 다시 실행할 것
정확한 command, checkpoint hash, eval commit, paired 세트의 item-id 파일, section 4에서 유도한
재현 허용오차.

## 7. 결정
C1이 Holm threshold에서 기각 **그리고** C2 하한이 -delta_NI 초과 **그리고** section 5가
주장 효과를 포괄할 때만 GO. 아니면 누락된 부분을 이름 붙여 NO-GO.
```

trigger line이 memo를 운반한다. 조건 세 개는 모두 필요하고 위 섹션 숫자 하나에 각각 추적 가능하다. 자기 표에서 자기 결정이 따라오지 않는 memo가 이 챕터가 막기 위해 존재하는 실패다.

**학습 중 curve.** 결정이 어떤 체크포인트를 남길지가 아니라 어디에서 멈출지라면 statistic은 curve이며, 이를 요약하는 방법이 보이는 것을 바꾼다. [[signal-and-noise-eval]] §5.2는 한 선택을 측정한다. 실행의 마지막 체크포인트 출력을 마지막 하나만 읽는 대신 평균내면 30태스크 decision accuracy가 68.9%에서 71.3%로, 13B scaling-law prediction error는 절대 percent 1.03에서 0.86으로 낮아졌다. early stopping에서는 exponential moving average가 거의 모든 학습 스텝에서 단일 체크포인트보다 나았다(그림 5). 이것은 출하할 모델을 만드는 체크포인트 평균이며, 그린 curve를 smoothing하는 것과 다른 연산이다. proxy curve를 믿는 별도 위험은 [[reward-model-overoptimization]]의 모양이다. 학습된 proxy reward model에 대해 정책을 최적화하면 proxy 점수는 계속 오르지만 gold 점수는 `√KL`이 커지며 꼭짓점을 찍고 떨어진다(그림 1). 이 결과는 checkpoint 전반 벤치마크 curve가 아니라 최적화 압력 아래 proxy 대 gold reward에 관한 것이다. 이를 옮기는 것은 해석이며, 방어 가능한 좁은 버전은 다음이다. 그린 양이 관심 있는 것의 proxy라면 smoothing한 proxy 대신 held-out 양 자체를 모니터링하라.

---

## §7 벤치마크가 지지할 수 있는 결정

구간은 점수를 얼마나 정밀하게 측정했는지에 답한다. 그 점수가 그 규모에서 결정하려는 것을 예측하는지는 답하지 않는다. 그 질문에는 별도 측정이 있다.

**Decision accuracy.** [[datadecide]]는 corpus 25개 × 크기 14개 × seed 3개(모델 1,050개)를 사전 학습하고, 작은 모델 순위가 1B 순위를 얼마나 자주 예측하는지 물었다. 단일 작은 크기(예: 150M)에서 모델 순위를 매기면 pairwise 비교 약 80%에서 1B에서 이기는 corpus를 고른다. scaling-law 기준선 8개 중 어느 것도 single-scale ranking의 compute 대 decision-accuracy frontier를 넘지 않는다(§3.1–3.2). 태스크별 그림은 크게 다르다. ARC Easy는 compute가 5 orders of magnitude 더 적어도 예측 가능하지만, BoolQ는 target 실행의 intermediate checkpoint가 있어야 trivial decision accuracy를 넘는다(§3.1, 그림 2). continuous likelihood metric을 쓰면 MBPP와 HumanEval은 trivial에서 80% decision accuracy로 이동하지만, Minerva와 GSM8K는 target metric도 continuous가 아니면 trivial 근처에 남는다(§3.4, 그림 6).

**Signal, noise, 그리고 비율.** [[signal-and-noise-eval]]은 signal을 비슷하게 학습한 모델 population 전반 상대 dispersion, noise를 §1.2의 양, SNR을 그 비율로 정의한다(식 2). SNR은 decision accuracy를 `R = 0.791`, `R² = 0.626`으로 예측하며, 두 성분 하나만으로는 그렇지 않다(§4.1, 그림 2). 개입 세 가지는 SNR로 subtask 필터링(상위 MMLU subtask 16개, decision accuracy +2.6%; 상위 AutoBencher subtask 6개, +5%), 마지막 체크포인트 평균(§5.2), bits-per-byte 점수화다. bits-per-byte는 정답의 negative log-likelihood를 UTF-8 byte 수로 나눈 것이며, GSM8K SNR `1.2 → 7.0`, MBPP `2.0 → 41.8`로 올리고 벤치마크 90.0%의 decision accuracy를 개선한다(§5.3). SNR은 scale 전반 전이하지 않는다. ARC Easy는 1.5B-4T의 7.89에서 32B-6T의 5.10으로, SocialIQA는 8.73에서 1.95로 떨어진다. Minerva MATH는 1.5B-4T의 0.91에서 7B-4T의 4.45로 오른다(부록 B.3, 표 4).

**Metric 모양과 겉보기 jump.** checkpoint curve의 계단은 metric 성질일 수 있다. [[emergent-abilities-mirage]]는 고정 InstructGPT/GPT-3 출력에서 산술 태스크가 Accuracy 아래에서는 emergent처럼 보이고 Token Edit Distance 아래에서는 매끄럽게 개선됨을 보인다. BIG-Bench 주장 emergent ability의 92% 초과는 Multiple Choice Grade와 Exact String Match라는 두 metric 아래 나타난다(§3–§4). 두 번째 메커니즘은 이 챕터 주제다. resolution은 `1/(test-set size)`로 정해지므로, 100항목 slice는 1포인트보다 작은 효과를 표현할 수 없고 그 resolution 아래 모델들은 0에 앉은 듯 보인다(§2, 부록 A.1). 두 줄은 bits-per-byte 결과와 일치한다. continuous metric은 더 높은 SNR을 지니며, [[benchmark-variance-quantified]] 표 2에서 continuous SNR은 보고한 10개 벤치마크 모두 discrete SNR을 넘고 더 예측 가능한 curve를 준다.

**더 작은 평가 세트: 서로 다른 두 결과.** item response theory는 항목마다 난이도와 discrimination parameter, 모델마다 ability를 모델링하므로 가장 정보량 큰 항목을 고를 수 있다. [[ai2-benchmirt]]는 16개 벤치마크와 항목 34,301개 전반 open-weight LLM 100개에서 2차원 IRT 모델을 적합했다. 가장 discriminative한 항목 10%를 남기면 “대체로 거의 같은 그림을 보존”하고, 50%는 “종종 전체 벤치마크의 해당 능력 측정을 더 가깝게 맞췄다”고 보고한다. held-out 항목 예측 정확도는 benchmark별 평균 기준선 70%에 대해 79%다. 반대로 [[benchmark-variance-quantified]]는 seed 모델 10개에 100항목 tinyBenchmarks subset을 적용하고 seed 분산이 ARC-C에서 `0.80 → 1.80/1.86`, GSM8k에서 `0.41 → 1.16/1.49`, HellaSwag에서 `0.21 → 2.06/2.42`로 오르는 것을 찾았다. monotonicity는 떨어지고 평균은 drift했다(ARC-C 7% 과대평가). 결론은 초기화와 데이터 순서 seed의 randomness에 더 쉽게 교란되는 비교를 만들기 때문에 이 방법이 “사전 학습 절제에서 제한된 사용”일 수 있다는 것이다(§5). **열린 질문.** 두 연구는 다른 것을 측정한다. 공개 모델 전반 construct alignment 및 ranking fidelity 대, 실행 하나의 intermediate checkpoint 전반 seed 수준 안정성이다. 방어 가능한 입장은 pruned subset이 잘 분리된 모델 간 순위는 보존할 수 있지만 같은 레시피 절제에 필요한 resolution을 잃을 수 있다는 것이다. 사용할 결정에서 pruned subset을 검증하되, seed 분산을 가정하지 말고 측정하라.

**Construct validity.** [[ai2-benchmirt]]은 벤치마크가 라벨과 다른 것을 측정할 수 있다고도 보고한다. BBQ는 회복한 general-reasoning ability와 0.85 상관, safety ability와 −0.06 상관이고, WMDP는 reasoning과 −0.89로 반대로 움직인다. 그런 벤치마크를 평균낸 gate의 “safety가 퇴행하지 않았다”는 줄은 라벨이 말하는 것을 측정하지 않는다. 이는 [[ch-50]] slice analysis를 한 단계 위로 옮긴 것이다. 집계가 통계적으로 건전해도 구성 요소가 다른 construct를 측정할 수 있다.

---

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Signal and Noise 연구(DataDecide 소형 모델) | 60M-750M | eval-gate | decision accuracy용 noise window | 마지막 체크포인트 5개, 소형 모델 전반 평균 | arXiv:2508.13144v1 §4.1; 부록 A.3.2 | 2026-09-14 검증([[signal-and-noise-eval]]) | 표 2: n=5는 거의 모든 벤치마크에서 true noise의 ±1σ 안 |
| Signal and Noise 연구(OLMo 2 target) | 13B | eval-gate | scaling-law error용 noise window | 마지막 체크포인트 30개, 1000스텝 간격 | §4.2, 각주 4 | 2026-09-14 검증 | sample size와 compute 비용의 trade-off로 명시 |
| Signal and Noise 연구 | 150M-1B | eval-gate | 소규모 결정 metric | primary metric 대신 bits-per-byte | §5.3, 그림 6 | 2026-09-14 검증 | 30태스크 평균 SNR `10.0 → 31.5`; decision accuracy `77.0% → 83.7%`; 벤치마크 90.0% 개선 |
| Signal and Noise 연구 | 60M-5xC → 1B-5xC | eval-gate | checkpoint 평균 | 비교 양쪽의 마지막 체크포인트 평균 | §5.2, 표 1 | 2026-09-14 검증 | 30태스크 decision accuracy `68.9% → 71.3%`; 13B prediction error `1.03 → 0.86` |
| DataDecide | 1B 예측 150M | eval-gate | 소규모 결정 규칙 | continuous likelihood metric(CORRECT PROB)으로 소형 크기 하나에서 레시피 순위화 | arXiv:2504.11393v2 §3.1-3.4 | 2026-09-14 검증([[datadecide]]) | 1B 대비 약 80% decision accuracy; 8개 scaling-law baseline 어느 것도 frontier 초과 안 함(그림 3) |
| DataDecide | 1B | eval-gate | target 규모 seed 복제 | 레시피당 full rerun 3회 | §2.1, 표 2 캡션 | 2026-09-14 검증 | 대부분 태스크의 일부 레시피에서 실행 간 std가 정확도 2 pp에 이름 |
| Miller 2024(모델 특이 아님) | — | eval-gate | 새 평가 크기 | 최소 질문 1,000개; 생성 분산 없이 `δ = 3 pp`, `α = 0.05`, power 0.80, `ω² = 1/9`에서 `n ≈ 969` | arXiv:2411.00640v1 §5, 식 9 | 2026-09-15 검증([[adding-error-bars-evals]]) | §5 수치 예가 권고의 명시한 근거 |
| Miller 2024(Anthropic 모델) | — | eval-gate | clustered 평가 표준오차 | clustered SE; DROP 1.34 대 naive 0.44(3.05×); MGSM 1.62% 대 0.86%(1.88×) | 표 4 | 2026-09-15 검증 | 표 4, fictional 아닌 숫자라고 설명 |
| Miller 2024 | — | eval-gate | 질문당 생성 K | `E[σ²_i]/K ≪ Var(x)`가 되게 K 선택; n=198에서 K 1→10은 MDE `13.2% → 7.5%` | §3.1, §5 | 2026-09-15 검증 | §5 수치 예 |
| FormatSpread 연구 | LLaMA-2 7B/13B/70B, Falcon-7B(-Instruct), GPT-3.5-Turbo | eval-gate | 평가 프롬프트 형식 수 | lower bound로 형식 10개 표본; GPT-3.5 연구에서 Thompson sampling 아래 형식 320개 | arXiv:2310.11324 §4.1-4.2 | 2026-09-15 검증([[prompt-format-sensitivity-formatspread]]) | 형식 10개 중앙 spread 7.5포인트; 최대 76포인트 |
| Madaan et al. 2024 | 7B, seed 10개 | eval-gate | 소규모 절제 MCQ 형식화 | 표준 MMLU 대신 MMLU-Cloze | arXiv:2406.10229v1 §3.3, 표 1 | 2026-09-15 검증([[benchmark-variance-quantified]]) | monotonicity `0.09 → 0.95`; seed std `0.57 → 0.22`; 완전 학습 모델 70개에서 표준 형식과 Pearson 0.92 |
| Madaan et al. 2024 | 7B, seed 10개 | eval-gate | 100항목 IRT subset(tinyBenchmarks) | 사전 학습 절제에는 비권장 | §5, 표 3-4 | 2026-09-15 검증 | seed std ARC-C `0.80 → 1.80/1.86`, HellaSwag `0.21 → 2.06/2.42`; monotonicity 하락 |

**작은 범용 실행의 시작점.** 여기의 모든 숫자는 출처가 사용한 조건 아래 위 검증 행에서 왔다. 1B-8B의 체크포인트 비교라면 두 체크포인트를 동일 item id에서 평가하고 score correlation과 paired 표준오차를 보고한다(Miller §4.2). 그룹 질문이 있는 suite에는 clustered estimator를 쓰고 cluster 수를 출력한다(표 3–4). 3포인트 차이를 분해해야 하는 새 평가는 최소 질문 1,000개를 목표로 한다(§5, 식 9). temperature를 내리는 대신 generation-variance 규칙에서 K를 정한다(§3.1, §3.3). 사전 학습 또는 중간 학습 절제에서는 마지막 하나를 읽는 대신 bits-per-byte를 읽고 마지막 체크포인트를 평균낸다(Heineman §5.2–5.3). 소규모 MCQ에는 cloze 형식을 쓴다(Madaan §3.3). seed 복제 행(1B에서 3 rerun)은 사전 학습 데이터 연구에서 왔다. 여기의 어느 출처도 측정하지 않은 post-training 실행에 전이할 값이 아니라 비용 참조로 취급하라.

---

## 일반화 관점

**(a) 폭을 넓히는 것.** 하나 이상을 측정하고 무엇을 측정할지 사전 등록하는 것이다. §6의 두 부분 gate는 non-regression을 post-hoc 점검이 아니라 출하 조건으로 만들고, §5의 multiplicity 제어는 우연히 움직인 slice 중심으로 memo를 쓰는 것을 막는다. 하나가 아니라 의미 동등 형식 전반에서 보고하면([[prompt-format-sensitivity-formatspread]] §4.2) 한 template 아래 능력이 아닌 능력 자체를 측정한다.

**(b) 좁아짐을 만드는 것.** 출처에 보이는 세 메커니즘이다. 선택: `α = 0.05`의 slice 20개에서 적어도 하나 false positive 확률은 0.642이므로, “최고 slice가 개선”하면 출하하는 파이프라인은 운이 좋았던 좁은 능력으로 드리프트한다. 형식 과적합: template 하나에 고정한 gate는 그 template에서만 개선한 모델이 통과할 수 있고, 다른 형식에서 모델 두 개의 순위가 뒤집힐 확률은 약 0.14이며 13B 대 70B reversal의 양쪽 방향 76%가 유의하다(§4.2). Under-powered non-inferiority: 일반 suite가 그럴듯하게 보일 수 있는 어떤 퇴행도 담을 만큼 넓은 구간은 모든 체크포인트를 통과시키므로, 망각은 gate를 실패하지 않은 채 round마다 쌓인다. 이것이 [[ch-50]] forgetting slice의 통계적 대응물이다.

**(c) 이 단계를 측정하는 방법.** 모든 gate 숫자마다 n과 cluster 수, estimator, paired 비교의 score correlation, K와 sampling temperature, 학습 실행 수와 출처를 가진 측정/가정 seed σ, 프롬프트 형식 수와 관측 spread, 선택 설계의 minimum detectable effect를 보고한다. 작은 scale에서 결정을 내려 큰 scale로 옮길 때에는 양 scale에서 같은 순위인 모델 쌍 비율인 decision accuracy를 보고한다([[datadecide]] §2.3, 식 3; tie 제외 `(Kendall's τ + 1)/2`와 같음). 해당 scale의 벤치마크 SNR도 점검한다([[signal-and-noise-eval]] §4.1). 거기서 SNR이 낮은 벤치마크는 구간을 어떻게 계산해도 그 결정을 지지하지 못한다.

---

## 흔한 실수와 감지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| 체크포인트 비교에 두 독립 구간을 빼기 | 차이 구간이 관측 항목별 불일치보다 훨씬 넓고 score correlation이 없음 | 식 7로 재계산하고 `Corr(s_A, s_B)` 보고; 이를 기록하지 않았다면 비교는 paired가 아님 |
| clustering 무시 | passage/번역 기반 suite(DROP, RACE, QuAC, SQuAD, MGSM)에 MMLU와 같은 식 사용 | 식 4 계산; DROP 비율은 3.05×([[adding-error-bars-evals]] 표 4). 항목 수 옆 cluster 수 출력 |
| fraction 점수에 `SE_Bernoulli` 사용 | F1, 부분 점수, 항목별 pass rate가 Bernoulli 구간을 가짐 | 식 1 사용; 여기서는 Bernoulli 형식이 너무 넓음(§2.1) |
| 하나의 학습 seed를 레시피 점수로 취급 | 따로 학습한 체크포인트의 1포인트 차이를 결과로 부름 | 그 벤치마크의 측정 seed σ(0.21–2.15 pp, [[benchmark-variance-quantified]] 표 1)와 비교하거나 두 번째 seed 실행 |
| 노이즈를 줄이려고 temperature 내림 | 평가 temperature가 배포와 다르고 분산을 이유로 정당화 | 둘 다 정당한 측정 조건이지만 §3.3은 T=0이 `Var(x)`를 올릴 수 있음을 보임; 주장이 어느 조건인지 명시 |
| K개 생성을 K·n 독립 항목으로 pooling | 구간의 n이 질문 수를 넘음 | 단위는 K개 생성의 질문당 평균(§3.1) |
| gate 뒤 프롬프트 형식 하나 | harness가 태스크당 template 하나이고 spread 미보고 | 의미 동등 형식 고정 세트 평가; 형식 10개 중앙 spread 7.5포인트, 모델 쌍 14.1% 순위 역전([[prompt-format-sensitivity-formatspread]] §4.2) |
| Slice shopping | 결과를 본 뒤 헤드라인 slice 선택 | target slice 사전 등록 및 전반 Holm 적용; 보정 없으면 `1 − 0.95²⁰ = 0.642` |
| “구간이 0을 포함”을 “퇴행 없음”으로 읽기 | 무퇴행 줄이 넓은 구간과 margin 없음 인용 | `δ_NI` 사전 명시 및 하한을 `−δ_NI`와 검정(§6) |
| swap-and-average 없는 judge win rate | pairwise win rate가 presentation 순서 하나에서 옴 | swap 및 재판정; GPT-4는 유사 답 쌍 35.0%에서 판정 변경([[judge-llm-bias]] 표 2) |
| 지지하지 못하는 벤치마크의 소규모 결정 | 작은 scale에서 chance 근처 벤치마크로 절제 결정 | 해당 scale SNR 또는 decision accuracy 점검([[signal-and-noise-eval]] 부록 B.3; [[datadecide]] 그림 2); MMLU는 7B 210B 토큰 뒤 chance 25에 대해 25.86 |
| 절제에 pruned 항목 subset 사용 | 계산을 아끼려 100항목 subset이 전체 벤치마크 대체 | subset의 seed variance 측정; [[benchmark-variance-quantified]] 표 3–4에서 2~10배 상승 |

---

## 이해 점검

1. §3.1에서 pairing은 구간을 1.61배 줄였다. 두 체크포인트에 무엇이 참이면 pairing이 전혀 줄이지 않으며, 왜 그것이 통계 문제 외 다른 것을 가리키는가?
2. 질문 9,622개 평가가 naive 표준오차 0.44, clustered 표준오차 1.34를 보고한다. clustered 값이 올바른 이유와 세트가 독립 질문 몇 개 가치인지 설명하라.
3. 팀이 노이즈를 없애려 temperature를 0으로 내리고 구간이 좁아졌다. `Var(s) = Var(x) + E[σ²_i]`를 사용해 추정치가 여전히 더 나쁠 수 있는 방법을 설명하라.
4. §1.4는 13B 대 70B rank reversal의 76%가 양 방향 모두 유의했다고 보고한다. 이것이 §2–§3과 모순되지 않는 이유와, 따라서 유의성 진술이 무엇을 허용하지 않는지 설명하라.
5. gate가 multiple-comparison 보정 없이 non-inferiority test 20개를 실행한다. 이 생략이 그쪽에서는 방어 가능하지만 gain 쪽에서는 아닌 이유와, 그 gate의 실제 실패 양상을 설명하라.
6. [[ai2-benchmirt]]은 항목 10% 보존이 벤치마크 그림을 보존한다고, [[benchmark-variance-quantified]]는 100항목 IRT subset이 seed 분산을 2~10배 올린다고 보고한다. 어느 연구도 틀릴 필요 없는 두 결과의 설명 하나와, 각각 어느 결정에 적용되는지 말하라.
7. checkpoint curve가 두 평가점 사이 계단을 보인다. 능력 변화 없이 이를 만들 수 있는 이 챕터 메커니즘 세 개와 각각을 분리할 측정을 이름 대라.
8. non-inferiority test가 구간이 너무 넓어서만 실패했다. §4의 어느 레버에 compute를 먼저 쓰며, 이를 고르기 위해 suite에 대해 무엇을 알아야 하는가?

---

## 연결

- **ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing**(의존): 이 챕터가 구간과 multiplicity 제어를 붙이는 slice를 제공하며, forgetting slice는 gate의 non-inferiority 절반 내용이다.
- **ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting**: §1.5에 도입한 judge noise 항의 전체 처리.
- **ch-51a — Evaluating Agent Generality and Reliability**(다음): agent 태스크의 실행 간 분산, pass@k 대 pass^k, scaffold 민감도, cost-aware gate. 여기서 §1 노이즈 원천은 한 차수 더 커진다.
- **ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming**: §6 gate에 non-inferiority 주장으로 들어가는 safety와 over-refusal suite, [[ai2-benchmirt]] construct validity가 가장 중요한 곳.
- **ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness**: 이 챕터의 estimator, format perturbation, memo를 구현한다.

---

## 출처

- [[benchmark-variance-quantified]] — 같은 레시피 7B 실행 10개, 벤치마크별 측정 seed variance, analytic 대 bootstrap 구간, MMLU 대 MMLU-Cloze, 100항목 IRT subset의 음성 결과.
- [[signal-and-noise-eval]] — signal/noise/SNR 정의, SNR 대 decision accuracy, subtask filtering, checkpoint averaging, bits-per-byte, 벤치마크 및 scale별 SNR.
- [[adding-error-bars-evals]] — 표준오차(CLT, Bernoulli, clustered), paired 비교와 분산 감소, 반복 생성, temperature 논증, §4·§6의 power 식.
- [[datadecide]] — metric으로서의 decision accuracy, 단일 scale 소형 실험이 지지하는 결정, 1B seed rerun과 보고된 실행 간 표준편차.
- [[prompt-format-sensitivity-formatspread]] — 측정한 format spread, 의미 동등 형식 간 순위 역전, 양 방향 유의 reversal 비율.
- [[emergent-abilities-mirage]] — checkpoint curve 겉보기 계단 원천인 metric shape와 `1/(test-set size)` resolution 한계.
- [[ai2-benchmirt]] — IRT 항목 pruning과 held-out 항목 예측, benchmark construct validity(BBQ, WMDP, XSTest).
- [[judge-llm-bias]] — §1.5 judge noise 항에 쓰인 swap-consistency 비율.
- [[reward-model-overoptimization]] — §6에서 proxy curve monitoring의 한계로 인용한 최적화 압력 아래 proxy 대 gold score. 벤치마크 curve로의 전이는 해석으로 표기.
- [[llama-3]] — [[adding-error-bars-evals]] §2.1이 모든 표준오차를 Bernoulli 형식으로 계산한 사례로 이름 붙인 보고서.
