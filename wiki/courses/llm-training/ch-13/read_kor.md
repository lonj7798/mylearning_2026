<!-- chapter: ch-13
     track: data
     title: Domain Mixing and DoReMi
     sources: [[doremi]], [[less]], [[scaling-laws-data-quality]], [[data-constrained-scaling]], [[interplay-pretraining-midtraining-rl]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[deepseek-v3]]
     figures: figures/doremi-reweighting.html
-->

# 13장 — 도메인 믹싱과 DoReMi

> **핵심 통찰.** 필터링([[ch-10]])과 중복 제거([[ch-11]])가 *어떤* 토큰이 살아남을지 결정하고 나면, 마지막이자 가장 영향력이 큰 데이터 레버는 살아남은 토큰을 모델에 *어떤 비율*로 보여줄지다. 고정된 수동 튜닝 믹스(GLaM/PaLM 시대)는 한 번의 추측이다. DoReMi는 믹스를 도메인 가중치를 둘러싼 프록시 언어 모델과 적대자 사이의 **미니맥스 게임**으로 재구성하고, 그룹 분포 강건 최적화(group distributionally robust optimization)로 온라인에서 푼다. 적대자의 균형 가중치는 전체 모델을 한 번도 훈련하지 않고도 30배 스케일 차이를 넘어 전체 모델로 이전된다.
>
> **가이드라인.** 트렁크 스케일에서 직관으로 믹싱 가중치를 고르지 말라. 저렴한 약 280M 파라미터 DoReMi 패스를 실행해 α를 도출하고, α로 코퍼스를 재샘플링한 뒤, 프로덕션 모델을 훈련하라. "믹스"를 복수로 다루라. 넓은 사전학습 단계의 믹스 하나, mid-training cooldown을 위한 두 번째 믹스, SFT를 위한 세 번째 믹스, RLVR 프롬프트 샘플링을 위한 네 번째 믹스가 있다. 각 단계는 목적이 다르므로 α도 다르다.

---

## 이 장이 존재하는 이유

10–12장은 *어떤 토큰이 코퍼스에 들어가는가*에 답했다. 이 장은 *각 토큰이 얼마나 자주 보이는가*에 답한다. 두 질문은 가까워 보이지만 그렇지 않다. 필터링은 0/1 결정이고, 믹싱은 확률 단체(simplex) 위의 연속 노브다. 필터링을 통과한 토큰도 그 도메인이 α = 2%에 머무르는 동안 다른 도메인이 α = 40%를 차지하면 묻힐 수 있다. 반대로 경계선 품질의 도메인도 α가 높게 설정되면 훈련을 지배할 수 있다.

대표 사례들은 이 문제의 중요성을 보여준다.

- **GLaM** (Du et al. 2022)은 여덟 개 소스(Wikipedia, books, C4, web 등)에 대해 혼합 가중치를 수동 튜닝했다. 작은 실행을 스윕하고 다운스트림 정확도로 선택해 가중치를 골랐다. 비용은 후보 믹스마다 작은 격자의 전체 사전학습 실행이다.
- **PaLM** (Chowdhery et al. 2022)은 약간만 조정한 GLaM식 고정 가중치를 재사용했다. 540B 실행은 출시 전 8B ablation에서 고른 믹스로 진행됐고, 다시 열리지 않았다.
- **DoReMi** ([[doremi]])는 280M 파라미터 적대자의 α가 *다운스트림 태스크 지식 없이* 8B(30배 더 큼)로 이전되며, GLaM 자체 코퍼스에서 GLaM의 태스크 튜닝 가중치와 맞먹는다는 것을 보였다.

고정 믹스 전통에는 세 가지 문제가 있었다. 첫째, 수동 튜닝은 샘플 효율이 낮다. 각 후보 믹스가 전체 사전학습 실행 비용을 요구한다. 둘째, 인간은 익숙한 도메인(Wikipedia, books)을 과대가중하고, 눈으로 품질을 판단하기 어려운 기술 코퍼스를 과소가중한다. 셋째, 고정 믹스는 하나의 태스크 가중치 가정을 굳혀 버린다. mid-training과 SFT는 서로 다른 믹스를 필요로 하지만, 고정 믹스 문화는 이 둘을 구분할 어휘를 만들지 못했다.

---

## 1. 고정 믹스 시대 — GLaM과 PaLM이 한 일

GLaM/PaLM 계보는 α를 하이퍼파라미터로 보고 스윕한다.

```
for candidate_alpha in GRID:
    train_8B_proxy(corpus, candidate_alpha, 100B tokens)
    score = downstream_eval(proxy)
store best_alpha
train_production(corpus, best_alpha, full_budget)
```

이 방식은 작동하지만 두 가지 실패 모드가 있다. 첫째, 프록시와 프로덕션 스케일이 갈라진다. 8B에서는 책을 낮게 가중하는 α가 맞을 수 있지만, 540B에서는 책을 높게 가중하는 것이 맞을 수 있다(스케일 의존 선호). 둘째, 다운스트림 평가 점수는 프록시의 프록시다. 믹스는 특정 벤치마크 묶음에 맞춰 최적화되고, 그 벤치마크는 나중에 같은 모델 평가로 새어 들어간다.

GLaM 시대의 믹스는 DoReMi 이후에는 유지되지 않는 방식으로 해석 가능했다. 실무자는 가중치를 보고 "웹은 범용이므로 40% 웹을 골랐다"고 말할 수 있다. DoReMi의 학습된 가중치는 정반대다. 그것들은 *해석 가능하지 않으며*, 그 성공 자체가 믹스에 대한 인간 직관이 체계적으로 잘못 보정돼 있었음을 보여주는 증거다.

---

## 2. DoReMi — 그룹-DRO 미니맥스 게임

**설정.** 코퍼스를 `K`개의 서로소 도메인으로 나눈다(The Pile: 22; [[dolma]]: 약 7). `θ`를 작은 프록시 LM의 파라미터라고 하자. `α ∈ Δ_K`는 도메인 위의 확률 벡터(단체)다. `θ_ref`는 고정된 *참조 모델*이다. 보통 균등 또는 기준 가중치로 사전학습한 작은 LM이며, 도메인별 손실 기준선을 계산하는 데 사용한다.

**목적함수.** DoReMi는 다음 미니맥스 게임을 한다.

```
    min       max     Σ_k α_k · ( L_k(θ) − L_k(θ_ref) )
     θ      α ∈ Δ_K
```

내부 항 `L_k(θ) − L_k(θ_ref)`는 도메인 `k`의 **초과 손실(excess loss)**이다. 프록시가 도메인 `k`에서 참조 모델보다 얼마나 더 나쁜지를 뜻한다. 적대자(`α`)는 프록시가 *상대적으로* 가장 못하고 있는 도메인에 가중치를 집중한다. 학습자(`θ`)는 그 도메인의 손실을 줄이는 방식으로 대응한다.

**왜 원시 손실이 아니라 초과 손실인가.** 원시 손실 기반 그룹 DRO는 α를 절대적으로 가장 어려운 도메인에 집중시킨다. LM에서는 항상 내재 엔트로피가 가장 높은 도메인(예: code, arXiv math)이 된다. 적대자는 거기에 갇히고 나머지는 모두 0으로 낮춘다. 초과 손실은 각 도메인을 기준선에 맞춰 중심화한다. 적대자가 묻는 질문은 *공정한 몫의 훈련이 달성하는 것에 비해, 프록시가 학습하지 못한 도메인은 무엇인가*다. 이 한 가지 변화가 순진한 DRO가 실패하는 곳에서 DoReMi가 작동하는 이유다.

### 2.1 α 업데이트 유도 — 단체 위의 exponentiated gradient

적대자는 `f(α) = Σ_k α_k · ℓ_k`를 최대화한다. 여기서 `ℓ_k := L_k(θ) − L_k(θ_ref)`는 현재 초과 손실이다. 제약 집합은 확률 단체 `Δ_K = { α : α ≥ 0, Σ_k α_k = 1 }`다.

비제약 gradient ascent는 `α_k ← α_k + η · ℓ_k`를 주지만, 이는 즉시 단체를 벗어난다(합이 1이라는 보장이 없고, 비음수성도 없다). 클리핑 뒤 정규화로 `Δ_K`에 투영하는 방법도 가능하지만, DoReMi가 쓰는 알고리즘은 아니다.

DoReMi는 **exponentiated gradient**(Kivinen & Warmuth 1997)를 사용한다. 이는 엔트로피 정규화자 `R(α) = Σ_k α_k log α_k`를 쓰는 mirror descent다. 학습률 `η`의 EG 한 단계는 다음과 같다.

```
  α_k^(t+1)  ∝  α_k^(t) · exp( η · ℓ_k^(t) )
```

정규화 `Σ α = 1`은 각 비정규화 가중치를 `Σ_j α_j · exp(η · ℓ_j)`로 나눠 수행한다. 즉 이 업데이트는 누적 초과 손실의 softmax다. `T`단계에 걸쳐 펼치면 `α_k^(T) ∝ α_k^(0) · exp(η · Σ_{t<T} ℓ_k^(t))`다. 가중치는 누적 초과 손실에 비례하며, 온도 `1/η`로 softmax된다.

**왜 투영이 아니라 EG인가.** 이유는 두 가지다. (1) EG는 단체의 기하를 존중한다. 혼합 사이의 자연스러운 거리는 상대 엔트로피 `KL(α‖α')`다. projected gradient는 유클리드 거리를 사용해 α에 거의 0인 항목이 많을 때 기하를 왜곡한다. (2) EG는 22개 도메인 사례를 매끄럽게 처리한다. 작은 좌표가 많은 고차원 단체에 반복 투영하면 좌표들이 계속 0이 된다. EG는 모든 도메인을 살아 있게 둔다(시작할 때 양수였던 좌표는 항상 엄격히 양수다). 이는 DoReMi의 Pareto 결과, 즉 *모든* 도메인에서 perplexity가 좋아지는 결과가 모든 도메인에 일정한 gradient 신호가 계속 들어가야 가능하기 때문에 중요하다.

∇ 형태는 부호도 명확히 한다. `∇_α L = ℓ`(점별)이다. 적대자가 자기 목적함수에 대해 갖는 gradient가 정확히 초과 손실 벡터다. 그래서 DoReMi 업데이트가 `α ← softmax(α_old + η · ℓ)`처럼 읽힌다. 실제로 그렇기 때문이다.

### 2.2 단계별 루프

```
reference θ_ref pretrained once, frozen.
initialize α^(0) = uniform(K)
initialize θ (proxy weights)

for step t = 1 .. T:
    sample minibatch B_t from corpus with per-domain probs α^(t-1)   # data sampler
    compute per-domain loss ℓ_k = L_k(θ; B_t^k) − L_k(θ_ref; B_t^k)  # no backward through θ_ref
    update α:   α_k^(t) ∝ α_k^(t-1) · exp(η · ℓ_k)                   # EG adversary step
    update θ:   θ ← θ − lr · ∇_θ Σ_k α_k^(t) · L_k(θ; B_t^k)         # SGD learner step
```

사소하지 않은 구현 세부사항이 두 가지 있다. (1) 모든 미니배치에는 *모든* 도메인의 샘플이 들어 있어야 한다. 그렇지 않으면 빠진 도메인의 `ℓ_k`가 정의되지 않는다. DoReMi는 이를 보장하기 위해 단계마다 도메인별 고정 슬라이스를 샘플링한다. (2) 최종 이전되는 α는 마지막 α가 아니라 **시간 평균** `ᾱ = (1/T) Σ_t α^(t)`다. 적대자는 진동하므로 평균이 균형 전략을 준다(표준 saddle-point averaging 논증).

### 2.3 왜 가중치가 스케일을 넘어 이전되는가

DoReMi의 경험적 헤드라인은 이렇다. 280M 프록시에서 계산한 α가 8B(30배)로 이전되고, The Pile에서 기준 정확도까지 2.6배 적은 단계와 +6.5 다운스트림 포인트를 낸다. 원칙적으로 α는 280M 모델의 특정 손실 표면에 묶인 미니맥스 해이므로, 이 이전 주장은 인상적이다.

논문의 해석, 그리고 지금까지 유지된 해석은 다음과 같다. 초과 손실은 *참조 모델 대비 학습 가능성 결핍*을 측정하고, 도메인 간 학습 가능성 순서는 대체로 스케일 불변이다. 280M이 균등 가중치 형제에 비해 어려워하는 도메인은 8B도 균등 가중치 형제에 비해 어려워한다. *절대* 손실은 스케일에 따라 변하지만, *상대적* 도메인별 난이도 순서는 변하지 않는다. EG는 softmax를 통해 그 순서만 읽으므로 α가 이전된다.

이는 정리가 아니라 추측이다. 참조 모델과 프록시가 서로 다른 용량 영역에 있으면 깨진다. 50M 참조와 8B 프록시는 불안정한 α를 만든다. 참조가 의미 있는 기준선을 세우기에는 너무 덜 익었기 때문이다. DoReMi의 기본값(`ref`와 `proxy` 같은 크기, 약 280M)이 경험적으로 안전한 운용점이다.

### 2.4 Pareto 개선의 놀라움

[[doremi]]에서 가장 직관에 반하는 경험적 결과는 이것이다. The Pile에서 DoReMi의 α는 22개 도메인 *모두*에서 도메인별 perplexity를 동시에 개선한다. 가중치가 줄어든 도메인도 포함된다. 단순히 생각하면 도메인을 downweight하면 그 도메인의 perplexity가 나빠져야 한다. 그렇지 않았다. 두 가지 이유가 겹친다.

첫째, 많은 Pile 도메인은 부분적으로 중복된다. "Common Crawl"을 낮추고 "OpenWebText2"를 높이면 토큰이 거의 중복된 분포 사이에서 이동한다. 모델은 다른 도메인 라벨 아래에서 동등한 텍스트를 계속 보므로, downweight된 도메인의 도메인별 손실은 거의 움직이지 않는다. 둘째, 어려운 도메인(예: 수학이 많은 arXiv)을 높이면 더 쉬운 도메인으로도 긍정적으로 전이되는 표현이 생긴다. 겹치는 support를 가진 도메인에서는 이 용량 재배분 효과가 직접 노출 효과를 압도한다.

Pareto 결과는 "고정 믹스"가 Pareto 지배되는 질량을 테이블 위에 남겨 두었음을 보여주는 증거다. 또한 직관 기반 믹싱에 반대하는 가장 강한 논거이기도 하다. 인간은 "Wikipedia perplexity를 개선하기 위해 Wikipedia를 downweight하자"고 제안하지 않는다. 하지만 Wikipedia가 미니맥스 효율 가중치보다 아래에 있을 때 DoReMi가 발견하는 것이 바로 그것이다.

---

## 3. 대안 — DSIR, downstream-reweighting, gradient similarity

DoReMi만 원칙 있는 믹서인 것은 아니다. 세 가지 대안도 내재화할 가치가 있다. 각자 다른 목적을 최적화하기 때문이다.

**비교.**

| Approach | Optimizes for | Needs downstream tasks? | Compute cost | Unit of selection |
|---|---|---|---|---|
| Fixed mix (GLaM/PaLM) | 다운스트림에 대한 수동 설정 프록시 | yes (via sweep) | O(grid · full run) | domain |
| Downstream-reweight | 명시적 다운스트림 벤치마크 | yes | O(sweep) | domain |
| DoReMi | 최악 경우 초과 손실 | **no** | O(1 proxy run, 30× smaller) | domain |
| DSIR (Xie et al. 2023) | 타깃 분포에 대한 KL 근접성 | yes (need target corpus) | O(hashing pass) | document |
| LESS ([[less]]) | 타깃 예제와의 gradient alignment | yes (few-shot target set) | O(warmup + datastore) | example |

**DSIR (Data Selection via Importance Resampling).** *타깃* 분포(예: "고품질 참조"로서 books + Wikipedia)가 주어지면, n-gram 해시 feature 비율로 근사한 중요도 가중치 `p_target(x) / p_source(x)`로 풀 문서 각각에 점수를 매기고, 그 가중치에 비례해 replacement를 허용하여 재샘플링한다. DSIR은 빠르다(해싱 패스, 모델 훈련 없음). 타깃 분포를 미리 알고 있을 때 맞는 도구다. 그러나 "다운스트림 reasoning에 가장 좋은 믹스는 무엇인가" 문제를 풀지는 못한다. 그 문제에는 단일 타깃 코퍼스가 없기 때문이다.

**다운스트림 정확도로 upweight.** 후보 믹스에서 작은 probe 모델을 실행하고, capability 벤치마크별 다운스트림 정확도를 측정하고, 벤치마크 점수를 α에 대해 회귀해, 회귀가 선호하는 α를 고른다. 이는 GLaM의 레시피를 방법론으로 만든 것이다. "우리는 다운스트림에 맞춰 튜닝했다"의 *정직한* 버전이다. Llama 3와 DeepSeek-V3가 category-conditional pretraining subset을 보고할 때 암묵적으로 하는 일이기도 하다.

**LESS — gradient-alignment selection.** [[less]]는 *post-training*을 위한 예제별 selector다. 재사용 가능한 low-rank per-sample gradient datastore를 만들고, 작은 타깃 few-shot set과의 cosine similarity로 순위를 매긴 뒤, 상위 5%를 유지한다. LESS는 "MMLU를 개선하고 싶다면 어떤 SFT 예제를 훈련해야 하는가"에 답한다. DoReMi는 "최악 경우 도메인 손실을 최소화하려면 어떤 도메인을 어떤 가중치로 사전학습해야 하는가"에 답한다. 단계도, 목적도, granularity도 다르다. 하지만 둘 다 *작은 프록시의 loss/gradient 신호*를 selector로 사용한다. DoReMi 쪽 동역학은 `figures/doremi-reweighting.html` 그림을 보라.

**무엇을 쓸 것인가.** 사전학습 믹스를 설정한다면 DoReMi. 타깃 capability를 위한 instruction-tune 데이터를 고른다면 LESS. 깨끗한 참조 코퍼스가 있고 웹을 재샘플링하고 싶다면 DSIR. frontier 스케일에서 튜닝하며 sweep할 compute가 있다면 downstream-reweight.

---

## 4. 믹스는 단계별로 다르다

초보자가 가장 흔히 저지르는 실수는 "데이터 믹스"를 하나의 객체로 취급하는 것이다. 사실은 네 개이고, 목적도 서로 다르다.

| Stage | Objective | Typical α shape | Example |
|---|---|---|---|
| Pretraining | 넓은 coverage, 최악 경우 도메인 손실 | 긴 꼬리: web ~ 40–60%, code 10–20%, books 5–10%, arXiv 2–5%, 나머지는 분산 | [[olmo-2]] OLMo-Mix-1124 (3.9T) |
| Mid-training / cooldown | 이후 단계를 위한 재사용 가능한 prior 설치, 약한 capability sharpening | 집중형: math 20–40%, code 20–40%, high-quality web 20%, IFT-like 10–20% | [[olmo-2]] Dolmino (~50B); [[olmo-3]] Dolma 3 Dolmino (100B) |
| SFT | instruction-following 표면 형식 + capability coverage | capability bucket 기반: code, math, tool use, multilingual 각각 10–20% | [[llama-3]] 6-round SFT mix; Tulu 3 mix on OLMo 2 |
| RLVR prompt distribution | 검증 가능한 보상을 가진 edge-of-competence 태스크 | 좁음: verifier-friendly math, code, IFEval | OLMo 2 / Tulu 3 RLVR prompts |

[[interplay-pretraining-midtraining-rl]] 논문은 이를 명시적으로 주장한다. mid-training은 자기 고유의 믹스를 가진 별도 단계이며, 고정 compute 아래에서는 잘 튜닝된 mid-training mix가 추가 RL을 이길 수 있다. OLMo 3는 이를 릴리스의 조직 원리로 삼는다. 네 단계에 네 이름 붙은 코퍼스(Dolma 3 Mix, Dolmino, Longmino, Dolci)가 있다.

DoReMi 절차는 원칙적으로 어느 단계에서나 실행할 수 있다. 실제로는 도메인 구조가 깨끗한 사전학습에서 가장 많이 쓰인다. SFT에서는 LESS/IFD식 예제별 selector가 도메인별 믹스보다 낫다. "SFT 도메인"은 개념적으로 약하기 때문이다(function-calling trace의 "도메인"은 무엇인가?).

**반복에 대한 경고.** 부족한 도메인을 upweight하는 것은 암묵적 반복이다. [[data-constrained-scaling]]의 약 4 epoch 임계값은 여전히 적용된다. DoReMi의 α가 작은 도메인을 당신의 토큰 예산에서 effective 8 epoch로 밀어 올리면, *그 도메인에 대해서만* data-constrained regime에 들어간 것이며 한계 수익은 평평해진다. 고정하기 전에 α에 대해 도메인별 effective epoch를 확인하라.

---

## 5. Llama 3와 OLMo 2가 실제로 하는 일 — 그리고 보고하지 않는 일

### 5.1 Llama 3 — bucketed capability data, 비공개 가중치

[[llama-3]]에서:

- Pretraining: 15.6T 토큰. 믹스는 가중치 벡터가 아니라 산문으로 설명된다. 논문은 bucket(web, code, math, reasoning, multilingual, long-context)을 이름 붙이지만 `α`를 공개하지 않는다. 독자는 *순서*는 추론할 수 있지만 *크기*는 알 수 없다.
- Post-training mix: 라운드당 약 50–80% synthetic rejection-sampled data, capability별 synthetic pipeline(code/math/multilingual/long-context/tool use/factuality)을 사용한다. 이는 *capability mix*이지 domain mix가 아니며, 가중치도 다시 비공개다.
- 여섯 라운드의 반복 SFT → RS → DPO 구조는 "믹스"가 *동적*이라는 뜻이다. 라운드 `k`의 SFT mix는 라운드 `k-1`의 최고 policy가 생성한다. 하나의 α를 보고하는 것은 개념적으로 부적절하다.

Llama 3가 숨기는 것: 어떤 단계에서도 실제 α 벡터, 라운드별 re-weighting 결정, 각 capability bucket을 통과시키는 데이터 파이프라인 수준 필터.

### 5.2 OLMo 2 — 두 단계 curriculum, 공개 가중치

[[olmo-2]]에서:

- Stage 1 — OLMo-Mix-1124(약 3.9T 토큰): DCLM + Dolma 1.7 + Starcoder + Proof Pile II. 믹스 비율을 공개한다.
- Stage 2 — Dolmino cooldown(약 50B 토큰): LR decay 중 더 고품질로 큐레이션한 subset. 명시적 단계 분리가 pretrain-mix ≠ mid-train-mix 구분을 인코딩한다.
- Post-training은 Tulu 3의 SFT/DPO/RLVR 믹스를 그대로 상속한다.

OLMo 2는 코퍼스를 공개한다. 그 믹스가 DoReMi에서 나온 것인지, 수동 튜닝인지, 직관 설정인지는 덜 명시적이다. 릴리스는 "수동 보정, ablation 검증"처럼 읽힌다.

### 5.3 OLMo 3 — 네 단계에 네 이름 붙은 믹스

[[olmo-3]]는 믹스가 단계별이라는 가장 깨끗한 공개 시연이다. Dolma 3 Mix(약 6T pretrain), Dolmino(100B mid-train), Longmino(50B long-context), Dolci(SFT/DPO/RLVR). 팀은 모든 단계의 mix composition, 모든 intermediate checkpoint, transition decision을 공개한다. 즉 "model flow" 프레이밍이다.

### 5.4 DeepSeek-V3 — 닫힌 믹스의 대조군

[[deepseek-v3]]는 14.8T개의 "diverse, high-quality" 사전학습 토큰을 보고한다. 도메인 분해 없음, α 없음, filter ablation 없음. 사전학습 비용은 정확히 보고한다(2.664M H800 hours). 그러나 그 compute를 소비한 믹스는 산문이다. 연구소가 공개하지 않기로 결정할 때의 현재 산업 기본값이다.

OLMo 3(모든 단계의 믹스 나열)와 DeepSeek-V3(산문뿐) 사이의 간격이 현재 실무의 스펙트럼이다. 연구 목적이라면 frontier lab의 믹스는 held-out 벤치마크의 다운스트림 평가로 수동 보정되고, 새로운 capability가 우선순위가 될 때 재튜닝되며, 실행 중에도 조용히 재가중되는 경우가 많다고 가정하라.

---

## 6. 믹싱 운영 체크리스트

1. DoReMi 실행 전에 각 도메인이 pool의 ≥ 1%를 제공하는지 확인하라. 더 작은 slice는 EG 업데이트에서 noise-dominated가 된다.
2. **reference model**을 균등 도메인 가중치로 훈련하라. 생략하지 말라. task-tuned reference는 DoReMi가 이미 편향된 기준선과 경쟁하게 만들고, 초과 손실 신호를 악화시킨다.
3. EG step `η`를 0.05–1.0 범위로 설정하라. DoReMi 기본값은 약 0.3이다. 너무 작으면 α가 움직이지 않고, 너무 크면 적대자가 심하게 진동해 시간 평균이 무의미해진다.
4. 프록시를 Chinchilla-equivalent small run 하나와 같은 총 토큰으로 훈련하라. 280M 모델이라면 대략 5.6B 토큰이다. DoReMi 가중치는 프록시가 수렴하기 훨씬 전에 안정화된다.
5. 마지막 iterate가 아니라 **시간 평균** `ᾱ`를 취하라(선택적으로 burn-in을 버리고 마지막 `T/2` 단계 평균).
6. 전체 사전학습 코퍼스를 `ᾱ`로 한 번 재샘플링하고, 표준 uniform-within-domain sampling으로 프로덕션 모델을 훈련하라. 프로덕션 모델에서 DoReMi를 온라인으로 실행하지 말라. 30배 이전이 핵심이다.
7. 도메인별 effective epoch 수를 약 4 epoch 임계값과 대조해 audit하라. 초과하는 도메인이 있으면 post에서 downweight하거나 [[dsir]]식 재사용으로 pool을 확장하라.
8. mid-training에서는 cooldown pool을 위해 새로운 α를 다시 도출하라. 목적이 바뀌었기 때문에 pretraining α는 mid-training에 틀리다.

---

## 7. 다음 장이 이어받는 것

14장은 도메인별 토큰 예산이라는 믹싱의 출력을 입력으로 받아 두 가지 질문을 던진다. (1) 전체 토큰 예산이 약 4 epoch 반복 상한에 닿으면 무슨 일이 일어나는가, (2) contamination을 통해 평가 데이터가 믹스 안으로 어떻게 새어 들어가는가. 믹싱은 레버이고, scaling law와 contamination은 어떤 믹스가 정직한지를 제한하는 제약이다.

---

## 연결

- [[doremi]] — Xie et al. 2023; group-DRO 미니맥스 구성, 30배 proxy-to-production 이전, Pareto 도메인별 개선 결과.
- [[less]] — Xia et al. 2024; SFT를 위한 gradient-similarity 예제별 selection, DoReMi의 도메인별 reweighting을 보완한다.
- [[scaling-laws-data-quality]] — "mix"가 "도메인별 effective sample size로 가중된 mix"를 뜻하게 만드는 quality term.
- [[data-constrained-scaling]] — α가 부족한 도메인을 얼마나 밀어 올릴 수 있는지를 제한하는 약 4 epoch 임계값.
- [[interplay-pretraining-midtraining-rl]] — 자기 고유의 mix objective를 가진 별도 단계로서 mid-training.
- [[llama-3]] — 공개된 것(bucket names, 15.6T)과 공개되지 않은 것(α).
- [[olmo-2]] — 공개된 단계별 코퍼스를 가진 두 단계 curriculum.
- [[olmo-3]] — 네 이름 붙은 믹스를 가진 네 단계 model-flow, 가장 깨끗한 공개 사례.
- [[deepseek-v3]] — 닫힌 믹스 대조군. 14.8T 토큰, 분해 없음.
- `figures/doremi-reweighting.html` — 조정 가능한 EG step `η`를 가진 interactive DoReMi convergence animation.
