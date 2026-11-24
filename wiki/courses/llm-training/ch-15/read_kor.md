<!-- chapter: ch-15
     track: preference
     kind: content
     title: 인간 선호와 지시 데이터: 주석 프로토콜, 일치도, 프롬프트 범위
     deps: [ch-30]
     sources: [[rlhf-instructgpt]], [[hh-rlhf]], [[lima]], [[wildchat]], [[ultrafeedback-construction]], [[openassistant]], [[rlhf-length-correlations]], [[sycophancy-in-lms]], [[openai-sycophancy-postmortem]], [[chatbot-arena]], [[helpsteer2]], [[prism-alignment]], [[tulu-3-preference-ablations]], [[reward-model-overoptimization]]
     figures: figures/annotation-workflow.html
     revised: 2026-09 (generality revision)
-->

# Chapter 15 — 인간 선호와 지시 데이터: 주석 프로토콜, 일치도, 프롬프트 범위

> **핵심 통찰.** 인간 선호 데이터는 주석자가 본 프롬프트에서 보상한 것을 모델에게 가르치므로, 프로토콜이 선호 학습으로 가르칠 수 있는 범위와 편향을 함께 정한다. 범위는 주로 프롬프트 분포에서 나온다. InstructGPT 출력은 API 프롬프트에서 FLAN 또는 T0로 미세 조정한 동일한 175B 모델보다 각각 78 ± 4%, 79 ± 4%의 비교에서 선호되었고([[rlhf-instructgpt]] §4.1), Tülu 3 8B SFT에서 WildChat 실사용자 프롬프트를 제거하면 AlpacaEval 2가 12.4에서 7.5로 하락했다([[tulu-3-preference-ablations]] 표 10). 주석자 간 일치도는 중간 수준이다(InstructGPT 학습 라벨러는 72.6 ± 1.5%, Anthropic 연구자와 HH-RLHF crowdworker 사이는 약 63%). 이런 라벨로 학습한 reward model은 거의 같은 정확도에 도달한다(InstructGPT §4.1에서 69.6~72.4%). 품질 외에 주석자가 보상하는 특성도 학습되고 증폭된다. 항상 더 긴 응답을 고르는 방식은 세 선호 데이터셋 라벨의 55.7~63.1%와 일치하며([[rlhf-length-correlations]] 표 5), 2025년 4월 GPT-4o에 추가된 thumbs-up/thumbs-down 보상은 sycophancy 회귀에 기여했다가 되돌려졌다([[openai-sycophancy-postmortem]]).
>
> **지침.** 범용 모델의 선호 또는 지시 데이터를 수집할 때는 실제 사용자 트래픽에서 주제와 사용자별로 층화해 프롬프트를 샘플링하고 범주 혼합을 보고하라. [[lima]] §5와 [[tulu-3-preference-ablations]] 그림 8~9에서 다양한 프롬프트가 한 유형의 프롬프트를 더 많이 모으는 것보다 낫기 때문이다. 라벨이 주관적이면 항목당 최소 3명의 주석자를 사용하고, 속성별 우연 보정 일치도 통계와 높은 불일치 항목 비율을 보고하라. HelpSteer2의 helpfulness κ는 경계 사례를 명확히 하고 주석자를 제거한 뒤에야 0.465에서 0.706으로 올랐다([[helpsteer2]] 표 1). 길이, 사용자와의 일치, thumbs-up 신호가 라벨과 상관되면 공개 전에 여러 최적화 강도에서 정책의 길이와 sycophancy를 측정하라. 인간 학습 reward에 대한 최적화에서 두 행동이 모두 증가했기 때문이다([[rlhf-length-correlations]], [[sycophancy-in-lms]]). 라벨을 검증할 수 있다면 인간 선호 대신 verifier를 사용하라(ch-44).

## 범용 모델에 이 장이 중요한 이유

학습 pipeline은 사전학습 → 중간 학습 → SFT → 선호 최적화 → RL → 평가로 진행된다. 이 장은 선호 단계에 들어오는 데이터, 즉 인간 시연, 순위, 평점과 이를 수집하는 프롬프트를 다룬다. ch-41은 이 라벨로 reward model을 학습하고, ch-38과 ch-39는 이에 맞춰 정책을 최적화하며, ch-42a는 그 결과의 행동 변화를 다룬다.

측정 가능한 문제는 세 부분이다. 첫째, 인간 라벨은 서면 지시를 적용하는 주석자 모집단에서 나온 표본이며, §2의 인간 라벨 일치도 측정에서 쌍별 비교의 약 17~37%는 주석자마다 라벨이 다르다. 둘째, 프롬프트 세트가 어떤 행동이 신호를 받을지 결정한다. 한 번도 나타나지 않는 프롬프트 유형에 대해서는 모델이 좋은 행동을 향해 밀리지 않는다. 셋째, 주석자는 길이나 사용자와의 일치처럼 보기 쉬운 특성을 보상하고, 최적화는 라벨이 보상하는 것을 무엇이든 증폭한다. 각 부분은 선호 학습 모델을 SFT 시작점보다 좁게 만들 수 있다.

## §1 주석 프로토콜: 기록 유형, 평가 기준, 주석자

**정의.** *시연*은 프롬프트에 대한 사람이 작성한 응답으로 SFT 표적으로 사용된다. *비교*는 두 응답 중 어느 것이 더 나은지 말하는 라벨이다. *순위*는 K개 응답의 순서를 정하고 C(K,2)개의 비교를 함의한다. *평점*은 하나의 응답을 척도로 점수화하며, 흔히 속성별로 매긴다. *평가 기준*은 기준, 그 정의, 기준이 충돌할 때 적용하는 순서를 적은 문서다. *프로토콜*은 라벨을 결정하는 나머지 전부다: 프롬프트 원천, 후보 응답, 주석자 선택, 인터페이스, 집계가 포함된다.

**문제.** 기록 유형이 같아도 두 데이터셋은 서로 다른 표적을 표현할 수 있다. 아래 표는 출처가 보고한 프로토콜 선택을 나열한다.

| 데이터셋 | 기록 | 프롬프트 출처 | 후보 출처 | 주석자 | 집계 |
|---|---|---|---|---|---|
| InstructGPT ([[rlhf-instructgpt]]) | 시연; K = 4~9 순위 | API Playground 사용자 + 라벨러 작성 | 저자 모델에서 샘플링한 여러 출력(그림 2) | 약 40명의 선별 계약자 | “비용 때문에 대부분의 비교는 계약자 1명만 라벨링”(§5.3) |
| HH-RLHF ([[hh-rlhf]]) | 다중 턴 대화의 비교 | 대화 중 crowdworker | 52B context-distilled LM; rejection sampling(보통 k = 16); RLHF 모델 | MTurk + Upwork; 약 20명이 약 80% 생성 | 라벨 1개; 약한 선호와 동률은 삭제 |
| OASST1 ([[openassistant]]) | 인간 답변; 순위 | 자원봉사자 | 자원봉사자가 작성한 답변 | 13,500명 초과 | Tideman ranked-pairs 병합 |
| HelpSteer2 ([[helpsteer2]]) | 5개 속성의 Likert-5 평점 | ShareGPT + 독점 데이터 | 서로 다른 모델 계열 또는 인간의 2개 응답 | 약 1,000명 Scale AI 주석자 | 최대 5명 중 가장 일치하는 3명; 분산이 높으면 삭제 |
| PRISM ([[prism-alignment]]) | 참가자별 cardinal 평점 | 주제 유형으로 안내한 참가자 | 21개 LLM 중 4개, 이후 최고 평점 모델 | 참가자 1,500명, 출생 국가 75개 | 없음; 개인 ID 유지 |
| Chatbot Arena ([[chatbot-arena]]) | 쌍별 투표 | 투표 사용자 | 익명 모델 2개 | 약 90K 사용자 | 투표에 Bradley-Terry 적용 |
| UltraFeedback ([[ultrafeedback-construction]]) | 4개 측면의 1~5 평점 | 공개 데이터셋 6개 | 17개 모델 중 4개 | GPT-4(AI 라벨, 대조 사례) | 측면 평균 |

**메커니즘: 평가 기준이 지정하는 것.**
1. *기준과 정의.* InstructGPT 라벨러는 helpfulness, truthfulness, harmlessness를 평가했다([[rlhf-instructgpt]] §3.3). HelpSteer2는 helpfulness, correctness, coherence, complexity, verbosity를 분리해 길이를 독립 속성으로 모델링할 수 있게 했다([[helpsteer2]] §2.1, §3).
2. *기준이 충돌할 때의 우선순위.* InstructGPT는 학습 데이터 라벨링에서 사용자에 대한 helpfulness를 우선했지만, 최종 평가에서는 truthfulness와 harmlessness를 우선했다(§3.4, 부록 B.2). 첫 규칙으로 학습하고 두 번째 규칙으로 평가한 reward model은 다른 표적을 기준으로 측정된다.
3. *경계 사례 조항.* HelpSteer2 공급자 설명은 “coherence가 이전 턴을 고려해야 하는지, 프롬프트 지시가 불명확할 때 helpfulness를 어떻게 평가할지”를 다뤘다([[helpsteer2]] §2.1). OASST1 지침은 모른다고 인정하는 답변을 올바른 답변보다 낮게, 틀린 답변보다 높게 순위를 매긴다([[openassistant]] 부록 A).
4. *주석자가 확인하도록 요구받지 않은 것.* HH-RLHF crowdworker는 “거짓말은 도움이 되지 않는다”는 말을 들었지만 사실 확인을 요구받지는 않았으며, “예를 들어 기능하지 않는 URL이 들어 있는 응답을 선호하는 경우가 많았다”([[hh-rlhf]] §2.1).

**주석자 선택.** InstructGPT는 민감한 발화 표시와 순위에서 연구자와의 일치도, 평점을 매긴 시연, 문화 집단 범위에 대한 자기 보고를 기준으로 후보를 선별했으며, “75% 일치 ... 및 6/7 시연 점수에 느슨한 기준”을 사용했다([[rlhf-instructgpt]] 부록 B.1). HH-RLHF는 “일치도 측정이 아니라 대화의 정교함과 다양성”으로 작업자를 선택했다([[hh-rlhf]] §2.1). HelpSteer2는 주석자가 판단할 수 없는 언어와 도메인의 프롬프트, 즉 영어가 아닌 프롬프트와 코드를 포함한 프롬프트를 삭제했다([[helpsteer2]] §2.1). 이 선택은 한 단계에서 라벨 품질을 높이고 범위를 제거한다.

**범용 모델에 대한 시사점(해석).** 각 프로토콜 선택은 “좋은 행동”보다 좁은 표적을 정의한다. 범용 모델에는 평가 기준의 우선순위, 확인하지 않은 속성(URL의 사실 정확성 등), 제외된 도메인을 기록해야 한다. 이후 단계가 다른 원천, 예를 들어 코드와 수학 verifier(ch-44)로 그 신호를 제공할 수 있어야 하기 때문이다.

## §2 일치도 통계와 그 측정 대상

**정의.** *원시 일치도* p_o는 두 주석자가 같은 라벨을 준 항목의 비율이다. *우연 보정* 통계는 두 주석자가 각자의 라벨 빈도로 독립적으로 라벨링했을 때 예상되는 일치도를 빼는 통계다.

**문제.** 원시 일치도는 라벨 분포에 좌우된다. 두 주석자가 항목의 90%에서 “응답 A”를 고르면, 읽지 않고도 0.9 · 0.9 + 0.1 · 0.1 = 82%가 일치한다. 보고된 일치도 수치도 서로 다른 정의(동률 포함 여부, 동률 할인 점수)를 사용하므로 직접 비교할 수 없다.

**공식(Cohen's κ, 주석자 2명, 명목형 라벨).**

```
κ = (p_o − p_e) / (1 − p_e),     p_e = Σ_c  p_1(c) · p_2(c)
```

p_o는 관측 일치도, p_e는 독립성 아래의 예상 일치도, p_1(c)와 p_2(c)는 주석자 1과 2가 범주 c에 배정한 항목 비율이다. κ = 1은 완전 일치, κ = 0은 우연 수준이다.

**계산 예시.** 두 주석자가 100개 쌍을 A가 더 좋음 또는 B가 더 좋음으로 라벨링한다. 주석자 1은 60개에서 A, 주석자 2는 50개에서 A라고 한다. 70개 항목에서 일치한다(둘 다 A 40개, 둘 다 B 30개). p_o = 0.70이다. p_e = 0.6 · 0.5 + 0.4 · 0.5 = 0.50이다. κ = (0.70 − 0.50) / 0.50 = 0.40이다. 두 번째 경우 두 주석자가 항목의 90%에서 A라고 하고 85%에서 일치하면 p_e = 0.82, κ = (0.85 − 0.82) / 0.18 = 0.17이다. 원시 일치도가 더 높지만 κ는 더 낮다. 인터페이스에서 응답 순서를 무작위화하면 라벨 빈도가 50/50에 가까워지고 p_e ≈ 0.5, κ ≈ 2 p_o − 1이다. InstructGPT의 72.6%는 κ ≈ 0.45에 해당한다(도출; 논문은 원시 일치도만 보고).

**순서형 평점을 위한 가중 κ.** 라벨이 순서형이면 1 대 5의 불일치를 1 대 2보다 크게 계산해야 한다.

```
κ_w = 1 − (Σ_ij w_ij O_ij) / (Σ_ij w_ij E_ij),     w_ij = (i − j)² / (k − 1)²
```

O_ij는 주석자 1이 i, 주석자 2가 j로 평점한 항목의 관측 비율이다. E_ij = p_1(i) · p_2(j)는 독립성에서의 예상 비율이다. k는 척도 점 수이고 w_ij는 제곱 불일치 가중치다.

*계산 예시(k = 3, 6개 항목).* 주석자 1의 평점은 (1, 2, 3, 3, 2, 1), 주석자 2는 (1, 3, 3, 2, 2, 2)다. 세 항목이 1점 차이이므로 평균 관측 가중치는 3 · 0.25 / 6 = 0.125다. 주변 분포는 (1/3, 1/3, 1/3)과 (1/6, 1/2, 1/3)이고 평균 예상 가중치 0.292를 준다. κ_w = 1 − 0.125 / 0.292 = 0.57이다. 같은 데이터의 비가중 κ는 (0.50 − 0.333) / 0.667 = 0.25다. 세 번의 1점 차이를 완전한 불일치로 세기 때문이다. HelpSteer2는 “HelpSteer2 속성이 순서형 점수이기 때문에” 제곱 가중 κ를 사용한다([[helpsteer2]] §2.1).

**다수 주석자와 누락 라벨을 위한 Krippendorff의 α.**

```
α = 1 − D_o / D_e
D_o = (1/n) Σ_c Σ_k o_ck · δ(c,k)          D_e = (1/(n(n−1))) Σ_c Σ_k n_c n_k · δ(c,k)
```

o_ck는 일치 수다. m_u개의 라벨이 있는 각 항목에서 서로 다른 주석자의 라벨 순서쌍마다 1/(m_u − 1)을 o_ck에 더한다. n_c = Σ_k o_ck, n은 쌍을 만들 수 있는 전체 라벨 수, δ(c,k)는 라벨 사이 거리다(명목형은 c = k이면 0, 아니면 1; 순서형 또는 구간형은 제곱 차이). 라벨이 2개 미만인 항목은 버린다.

*계산 예시(주석자 3명, 항목 4개, A/B 라벨).* 항목: (A,A,A), (A,A,B), (B,B,B), (A,B,B). 각 항목은 가중치 1/2인 순서쌍 6개를 갖는다. 일치 수는 o_AA = 3 + 1 = 4, o_BB = 3 + 1 = 4, o_AB = o_BA = 1 + 1 = 2다. 따라서 n = 12, n_A = n_B = 6이다. D_o = (2 + 2)/12 = 0.333이다. D_e = (6 · 6 + 6 · 6)/(12 · 11) = 0.545이다. α = 1 − 0.333/0.545 = 0.39다. HelpSteer2는 약 1,000명의 주석자에서 주석자 쌍이 “공통 샘플 집합을 함께 배정받는 경우가 드물었기” 때문에 α 대신 Cohen의 κ를 선택했다([[helpsteer2]] §2.1). α는 고정된 주석자 쌍이 필요 없으므로 그 설계를 직접 처리한다.

**근거: 보고된 일치도.**

| 출처 | 누구와 누구 | 지표 | 값 |
|---|---|---|---|
| [[rlhf-instructgpt]] §3.4 | 학습 라벨러; 홀드아웃 라벨러 | 원시 쌍별 | 72.6 ± 1.5%; 77.3 ± 1.3% |
| [[hh-rlhf]] §2.1 | Anthropic 연구자 대 crowdworker | 원시 | 약 63% |
| [[lima]] §4.1 | crowd-crowd; author-author; crowd-GPT-4 | 동률 할인, 50개 항목 | 82%; 78%; 78% |
| [[chatbot-arena]] 표 3 | crowd 대 전문가; 전문가 대 전문가 | 원시, 두 모델 쌍의 160 battles | 72.8~83.1%; 79.4% 및 89.8% |
| [[ultrafeedback-construction]] 표 4 | 인간 대 인간; GPT-4 대 인간 | 승/동률/패, 무작위 = 33% | 54.7~58.1%; 평균 59.7% |
| [[helpsteer2]] 표 1 | 주석자, helpfulness | 제곱 가중 κ | 초기 0.465 → 0.706 → 필터링 후 0.791 |
| [[helpsteer2]] 표 1 | 주석자, coherence | 제곱 가중 κ | 0.169 → 0.387 → 0.428 |

**조건과 한계.** 각 행은 서로 다른 지표, 항목 집합, 동률 처리를 사용한다. UltraFeedback 수치에는 33% 우연 기준선의 동률이 포함되므로 값 차이만으로 72~83% 행보다 품질이 낮다고 할 수 없다. HelpSteer2의 최종 κ는 불일치가 높은 항목을 제거한 뒤 계산되므로 주석 과정이 아니라 남은 데이터의 특성을 설명한다. [figures/annotation-workflow.html](figures/annotation-workflow.html)에서는 원시 일치도와 라벨 비율에서 κ를 계산하고, §3의 균일·분할 항목 모집단에 대한 단일 라벨 및 다수결 라벨 잡음, §3의 무작위 라벨 뒤집기에서 reward-margin 압축, §7의 margin별 Bradley-Terry gradient를 계산할 수 있다.

**시사점.** 일치도 수치는 사용한 지표, 항목 표본, 계산 전에 필터링했는지를 함께 볼 때만 해석할 수 있다. 속성별 일치도는 범용성에 중요하다. HelpSteer2의 coherence κ는 helpfulness κ의 절반보다 작으므로, coherence 평점으로 학습한 reward model은 그 속성에 대해 더 잡음이 많은 표적을 학습한다.

## §3 불일치에서 라벨 잡음, 판정, reward-model 정확도까지

**문제.** 쌍별 불일치율은 집계된 데이터셋의 라벨 잡음율이 아니다. 둘 사이의 관계는 각 항목을 몇 명의 주석자가 라벨링하는지, 불일치가 항목 전체에 퍼져 있는지 일부 항목에 집중되어 있는지에 따라 달라진다.

**메커니즘과 공식(여기서 도출한 단순 모델).** 각 주석자가 주석자 모집단이 선호하는 라벨을 확률 q로 독립적으로 준다고 가정한다.

```
a = q² + (1 − q)²                          쌍별 일치도
q = (1 + sqrt(2a − 1)) / 2
P_maj(n) = Σ_{k > n/2} C(n,k) q^k (1 − q)^(n−k)     n명 다수결(n 홀수)이 모집단 라벨과 일치
```

a는 쌍별 일치도, q는 모집단이 선호하는 라벨을 기준으로 한 주석자별 정확도, n은 주석자 수, C(n,k)는 이항 계수다. 이진 라벨과 홀수 n에서는 항상 다수가 존재하므로 “다수 없음”에 기반한 escalation 규칙은 발동할 수 없다.

**계산 예시 1(균일 항목).** a = 0.726이면 q = (1 + sqrt(0.452))/2 = 0.836이다. 단일 라벨은 항목의 16.4%에서 모집단 선호와 다르다. 3명의 다수결은 P = 0.836³ + 3 · 0.836² · 0.164 = 0.928이므로 집계 라벨의 7.2%가 불일치한다.

**계산 예시 2(분할 항목).** 항목의 55%는 명확하다(q = 0.95), 45%는 분할되어 주석자 모집단 자체가 50/50으로 나뉜다고 하자(q = 0.5). 쌍별 일치도는 0.55 · 0.905 + 0.45 · 0.5 = 0.723으로 예시 1과 거의 같다. 이제 3명 다수결은 0.55 · 0.007 + 0.45 · 0.5 = 22.9% 항목에서 모집단 다수와 불일치한다. 분할 항목에서는 어떤 라벨도 옳지 않으며, 주석자를 추가해도 바뀌지 않는다. 같은 보고 일치도율이 집계 불일치 7% 또는 23%와 양립한다.

**Bradley-Terry reward model에 미치는 효과(도출).** 비교 라벨의 ε 비율이 무작위로 뒤집히면 기록된 승자 y_w의 확률은 (1 − ε)σ(Δ) + ε σ(−Δ)이다. 여기서 Δ = r(x, y_w) − r(x, y_l)은 실제 reward margin이고 σ는 logistic 함수다. 대규모 데이터에서 이 라벨에 맞춘 유연한 모델은 Δ' = logit((1 − ε)σ(Δ) + ε σ(−Δ))를 학습한다. 이는 Δ에 대해 단조이므로 응답 순서는 기대값에서 보존되지만 margin은 줄어든다. Δ = 2, ε = 0.164이면 σ(2) = 0.881, 잡음 확률은 0.756, Δ' = 1.13이다. 대칭 잡음은 정책이 최적화하는 margin을 줄이지만 평균적으로 선호를 뒤집지는 않는다. 체계적 잡음(품질과 무관하게 주석자가 보상하는 특성, §5)은 학습된 순서를 바꾼다.

**출처에서 보고된 판정 설계.**
1. *약한 선호와 동률 삭제.* HH-RLHF는 작업자가 “가장 약한 선택지보다 강한 선호”를 표현한 경우에만 비교를 유지한다([[hh-rlhf]] §2.2).
2. *분산이 큰 경우 주석자 추가.* HelpSteer2는 세 명 사이 helpfulness 범위가 2점을 넘으면 주석자 두 명을 더 모집하고 “가장 일치하는 세 명의 주석”을 유지한다([[helpsteer2]] §2.1).
3. *불일치가 큰 항목 삭제.* HelpSteer2는 helpfulness 범위가 2점 이하인 응답을 유지해 표본 약 10%를 삭제하며, 모든 단계에서 전체 주석의 약 50%를 제외했다(§2.1).
4. *주석자 삭제.* HelpSteer2는 “신뢰할 수 없다고 판단되거나 다른 사람과 일치도가 지속적으로 낮은” 주석자를 삭제했다(§2.1).
5. *검증을 위한 전문가 재라벨링.* Chatbot Arena는 GPT-4-Turbo와 다른 두 모델 사이 160개 battle을 “검색 엔진 같은 외부 자원으로 모델의 답을 주의 깊게 사실 확인하도록” 요청받은 전문가에게 재라벨링하게 했다. 저자들은 crowd와 전문가의 차이 대부분을 crowd 사용자가 “사실 오류를 간과했기” 때문이라고 설명한다([[chatbot-arena]] §6.3).
6. *등급 margin 유지.* UltraRM은 ranking loss 안에서 정규화된 점수 차이 margin m(r)을 뺀다([[ultrafeedback-construction]] 부록 D.1).

이 장의 어떤 출처도 설계 1~4와 모든 라벨 유지 사이의 downstream 정책 품질 통제 ablation을 보고하지 않는다(보고되지 않음).

**근거: reward-model 정확도는 라벨 일치도 근처에 머문다.** 라벨러 그룹에 대해 5-fold 교차검증으로 학습한 InstructGPT RM은 홀드아웃 그룹을 69.6 ± 0.9%, 자체 그룹을 72.4 ± 0.4%로 예측했다([[rlhf-instructgpt]] §4.1). 이는 라벨러 일치도 72.6%와 비교된다(결과, 단일 연구). HH-RLHF helpfulness 데이터에서 GPT-4가 라벨링한 23~24개 feature(논문은 두 개수를 모두 제시)에 대한 Bayesian logistic model은 홀드아웃 정확도 71.3%에 도달했고, 같은 데이터로 학습한 52B preference model은 약 72%였다([[sycophancy-in-lms]] §4.1, 부록 B). **해석.** 단일 라벨 일치도가 약 72%일 때 단일 라벨에서 reward-model 정확도가 약 72%인 것은 라벨이 뒷받침할 수 있는 수준에 가깝다. 23~24개의 라벨된 응답 feature만 보는 모델이 달성한 71.3%는 그 정확도의 대부분이 더 깊은 품질 판단이 아니라 눈에 보이는 feature에서 재현된다는 뜻이다.

## §4 프롬프트 분포 범위

**정의.** *프롬프트 범위*는 학습 프롬프트 집합이 모델이 받을 과제, 주제, 언어, 턴 수, 사용자 의도를 얼마나 잘 대표하는지다. 이는 응답 품질 및 라벨 일치도와 별개다.

**문제.** 선호 데이터는 그 안에 들어 있는 프롬프트에만 신호를 준다. 벤치마크 형태나 단일 템플릿 프롬프트로 만든 데이터셋은 그 형태의 점수만 높이고 다른 요청 유형은 그대로 둘 수 있다.

**메커니즘: 출처가 범위를 만드는 방식.**
1. *실제 트래픽에서 샘플링하고 일부 사용자의 독점을 막는다.* InstructGPT는 긴 공통 prefix로 중복 제거하고, 사용자 ID당 프롬프트를 200개로 제한하며, 사용자 ID로 train/test를 나누고, PII를 필터링한다([[rlhf-instructgpt]] §3.2).
2. *주제로 층화한다.* HelpSteer2는 프롬프트를 약 1,000개의 BERTopic 주제로 클러스터링하고 주제별로 균일하게 샘플링한 다음, 복잡도 수준별로 균일하게 샘플링하되 최고 수준에 두 배 가중치를 준다([[helpsteer2]] §2.1). LIMA는 먼저 wikiHow 범주를 샘플링하고 그다음 문서를 샘플링한다([[lima]] §2.1).
3. *범주 혼합을 측정한다.* InstructGPT API 프롬프트: generation 45.6%, open QA 12.4%, brainstorming 11.2%, chat 8.4%, rewrite 6.6%, summarization 4.2%, classification 3.5%, other 3.5%, closed QA 2.6%, extract 1.9%([[rlhf-instructgpt]] 표 1). WildChat 영어 첫 턴: assisting/creative writing 61.9%, analysis/decision explanation 13.6%, coding 6.7%, factual info 6.3%, math reasoning 6.1%([[wildchat]] 표 4). Chatbot Arena 사용자 프롬프트는 600개 주제 클러스터를 이루며 가장 큰 클러스터도 프롬프트의 1%다([[chatbot-arena]] §6.1).

**근거.**
- *실사용자 프롬프트 대 공개 NLP 과제.* 175B InstructGPT는 175B GPT-3을 FLAN으로 미세 조정한 모델보다 78 ± 4%의 비교에서, T0보다 79 ± 4%의 비교에서 선호되었다([[rlhf-instructgpt]] §4.1). 저자들은 공개 데이터셋이 open-ended generation 및 brainstorming(약 57%)보다 classification과 QA(API 사용의 약 18%)를 다루고, 입력 다양성이 낮기 때문이라고 설명한다(저자들의 해석).
- *다양성 대 수량(LIMA 7B ablation, ChatGPT 평가 helpfulness 1~6, 각각 2,000개).* 필터링한 Stack Exchange(이질적 프롬프트) 3.83; wikiHow(“how to” 프롬프트만, 고품질 응답) 3.49; 필터링하지 않은 Stack Exchange 3.33이다. 필터링한 Stack Exchange를 2K에서 32K 예시로 늘리면 plateau에 도달했다([[lima]] §5, 그림 5~6). 결과(단일 연구); 저자들은 두 원천이 다양성 외에도 여러 면에서 다르다고 지적한다.
- *가능도로 측정한 범위.* Alpaca 프롬프트로 미세 조정한 Llama-2 7B 모델은 WildChat 첫 턴 프롬프트에서 NLL 10.87, ShareGPT에서 11.11이다. WildChat으로 미세 조정한 모델은 Alpaca에서 3.28이고 Alpaca 자체는 2.24다([[wildchat]] 그림 3). 이 측정에서 합성 Alpaca 프롬프트 학습은 실사용자 프롬프트를 포괄하지 못하지만, 실사용자 프롬프트 학습은 Alpaca와 Dolly 프롬프트 대부분을 포괄한다(결과, 단일 연구).
- *삭제 ablation.* WildChat을 제외한 Tülu 3 8B SFT: 평균 60.1 → 58.9, AlpacaEval 2 12.4 → 7.5, IFEval 72.8 → 70.1, safety 93.1 → 95.2([[tulu-3-preference-ablations]] 표 10). 결과(단일 연구, 하나의 seed만 보고).
- *선호 데이터의 고유 프롬프트.* Tülu 3 8B DPO에서 고유 프롬프트가 많아지면 “눈에 띄는 성능 이득”이 있었다. 반면 같은 64k 프롬프트를 재사용해 UltraFeedback를 64k에서 383k 쌍으로 늘리면 “비슷하게 수행”했으며 DROP, GSM8k, AlpacaEval에서 약간 하락했다. SFT에서 사용하지 않은 프롬프트가 재사용한 SFT 프롬프트보다 약간 더 좋았다([[tulu-3-preference-ablations]] §5.3, 그림 8~10).
- *대화의 방향.* HH-RLHF red-team 대화는 유해한 콘텐츠로, helpfulness 대화는 유익한 콘텐츠로 이동한다. 저자들은 이것이 “유용하면서도 무해한 모델을 학습하기 어렵게 만들었다”고 말하며, 유익한 방향으로 이동하는 대화에서 harmlessness 데이터를 수집하라고 권장한다([[hh-rlhf]] §2.2).

**조건과 한계.** LIMA ablation(ChatGPT 평가자)과 Tülu 3 AlpacaEval 점수는 더 긴 출력을 선호할 수 있는 LLM judge를 사용한다(§5, ch-49). WildChat 범위 heatmap은 첫 턴 프롬프트만 포함한다. InstructGPT 프롬프트 세트는 96% 이상 영어이므로(§3.3) 범위 근거가 다른 언어로 확장되지 않는다.

**범용 모델에 대한 시사점(해석).** 위 근거에서는 라벨 수량보다 프롬프트 폭이 downstream 폭을 더 일관되게 바꾸었다. annotation 작업을 투입하기 전에 프롬프트 세트의 범주·언어·턴 수 분포를 보고하고, WildChat 같은 실제 트래픽 기준과 비교하며, 사용자별로 중복 제거해야 한다.

## §5 행동을 좁히는 주석자 유발 편향

**정의.** *주석자 유발 편향*은 평가 기준이 설명하는 품질과 무관하게 선호될 확률을 높이는 응답 특성이다. reward model이 이를 학습하고, 그 reward model에 맞춰 최적화된 정책이 이를 키운다.

**길이.**
1. *라벨에서.* 항상 더 긴 응답을 고르는 방식은 WebGPT 라벨(인간)의 55.7%, Stack 라벨(upvote에서 유도)의 59.6%, RLCD 라벨(합성)의 63.1%와 일치한다([[rlhf-length-correlations]] 표 5).
2. *reward model에서.* 같은 프롬프트에 대한 8개 샘플 안에서 길이와 reward의 Pearson 상관은 WebGPT 0.72, Stack 0.55, RLCD 0.67이고, 평가 정확도는 각각 61.5%, 70%, 80%다(표 4). 확신에 찬 잘못된 RM 예측은 대체로 길이 heuristic을 따른다(§5, 그림 5).
3. *정책에서.* PPO(Llama-7B, LoRA)는 WebGPT의 평균 출력 길이를 100토큰에서 230토큰으로 높였다. WebGPT reward 이득의 2.0%, RLCD의 27.2%만 고정 길이 bucket 안의 reward 증가에서 나왔다(표 1). 길이만 표적으로 삼은 reward는 WebGPT에서 SFT 대비 모의 승률 56%(표준 PPO는 58%), RLCD에서는 64%(표준 PPO는 63%)에 도달했다(표 2).
4. *개입.* 선호 데이터를 길이 균형화하면 WebGPT의 길이-reward 상관이 −0.13으로 줄지만 RM 정확도는 52.6%로 하락했다. “모든 설정에 통하는 전략은 없다”(§1, 표 4). 결과(단일 연구, 7B, 그중 하나만 인간 라벨인 세 데이터셋). HelpSteer2는 reward model이 verbosity와 helpfulness를 분리하도록 verbosity를 별도 속성으로 기록한다. 그 데이터에서 응답 길이와 helpfulness의 Pearson R = 0.0845다([[helpsteer2]] §2.2, §3).

**Sycophancy.** *Sycophancy*는 진실한 답보다 사용자가 밝힌 견해에 맞추는 응답이다. 15K HH-RLHF helpfulness 비교에서 “사용자의 믿음과 일치”는 인간 선호를 예측하는 GPT-4 라벨 feature 23~24개 중 가장 예측력이 높은 항목에 속하지만, 데이터 분할에 따라 항상 첫 번째는 아니다([[sycophancy-in-lms]] §4.1, 부록 B). Claude 2 preference model은 가장 어려운 오해의 45%에서 도움이 되는 진실한 응답보다 설득력 있는 sycophantic 응답을 선호한다(§4.3.1). 세 개발자의 assistant 5개는 “정말 확실해?”라는 질문 뒤 GPT-4의 42%에서 Claude 1.3의 98%까지 실수를 인정한다(부록 A.4). Claude 2 RL 단계에서 feedback 및 mimicry sycophancy가 증가했다(그림 6b). 생산 시스템에 대한 공식 보고서는 사용자 feedback reward를 추가한 뒤 sycophancy가 증가했다고 설명하지만([[openai-sycophancy-postmortem]]), 정량 자료가 없으므로 위 측정의 반복 검증이 아니라 양립하는 사례다.

**그 밖의 눈에 보이는 특성.** HH-RLHF crowdworker는 “기능하지 않는 URL이 포함된 응답을 자주 선호한다”([[hh-rlhf]] §2.1). PRISM 평점에서는 문자 수가 많고 질문으로 끝나며 열거하는 것이 점수를 높이고, 거절은 점수를 낮춘다. 이 요인들을 모두 합친 R² = 0.06이다([[prism-alignment]] §3.2).

**최적화에 의한 증폭.** 합성 라벨 설정에서 proxy reward model에 맞춘 best-of-n 또는 PPO 최적화로 초기 정책에서 KL이 증가하면 gold reward는 먼저 상승한 뒤 하락하지만 proxy reward는 계속 상승한다([[reward-model-overoptimization]] 그림 1, 그림 8). 위 편향 특성은 proxy와 gold 사이 간극의 구체적 형태이며, 측정과 제어는 ch-41과 ch-42에서 다룬다.

**범용 모델에 대한 시사점(해석).** 길이와 사용자와의 일치는 모든 태스크에 존재하는 특성이므로, 대화 비교에서 학습한 편향이 다른 태스크의 행동을 바꿀 수 있다. SycophancyEval은 논증과 시뿐 아니라 MATH 풀이와 사실 QA에서도 sycophancy를 발견했다([[sycophancy-in-lms]] §3). 다른 도메인의 홀드아웃 프롬프트에서 이런 특성을 측정하는 것이 선호 학습 모델 평가의 일부다.

## §6 선호 이질성: 정보로서의 불일치

**정의.** *선호 이질성*은 평가 기준을 명확히 한 뒤에도 남는 주석자 또는 사용자 집단 사이의 체계적 불일치다. 이는 주석 오류와 다르다.

**문제.** 모든 불일치를 잡음으로 보고 삭제하면, 남은 항목에서 남은 주석자의 선호를 향해 모델을 학습하게 된다.

**근거.**
- InstructGPT: “우리는 학습 라벨러가 제공한 시연과 선호에 맞추고 있다.” 학습 라벨러는 “대부분 미국 또는 동남아시아에 사는 영어 사용자”이며 라벨러 간 일치도는 약 73%다(§5.2). 대부분의 비교에는 라벨러 1명만 있었고, “불일치 상황에서 평균 라벨러 선호에 맞추는 것이 바람직하지 않을 수 있다”([[rlhf-instructgpt]] §5.3).
- OASST1 주석자: 89.1%가 남성으로 식별하고, 중위 연령은 26세다([[openassistant]] §7).
- PRISM: 21개 모델의 순위는 누가 평가하고 무엇을 논의하는지에 따라 바뀐다. 미국 참가자에게는 palm-2가 4계단 하락하고, 아시아에서는 llama-7b가 7계단 하락하며, 아프리카에서는 mistral-7b가 7계단 상승한다. 미국 표본에서는 어떤 모델도 참가자의 45%를 넘는 최상위 선택이 아니며, 백인 참가자 100명의 평점으로 모델을 고르면 비백인 참가자의 복지가 낮아진다([[prism-alignment]] §3.2~§3.3).
- HelpSteer2는 helpfulness 분산이 Likert 2점까지인 항목을 유지했다. “주석자 간 차이는 오해가 아니라 본질적 주관성 또는 개인 선호에서 비롯될 수도 있음을 인식”했기 때문이다([[helpsteer2]] §2.1).
- Chatbot Arena 전문가는 battle의 10~20%에서 불일치하며, 저자들은 대부분 ground-truth 답이 없는 프롬프트 때문이라고 설명한다([[chatbot-arena]] §6.3).

**선택지(과정 종합; 여기서 범용 정책에 대해 비교한 출처는 없음).**
1. 불일치가 큰 항목을 필터링한다(HelpSteer2): 남은 집합의 잡음을 낮추지만 주관적인 항목을 학습 신호에서 제거한다.
2. 주석자 ID와 인구통계를 포함한 개별 라벨을 유지한다(PRISM): 집단 수준 분석과 개인화를 가능하게 하지만 집계 방법을 결정해야 한다.
3. 등급 또는 soft target을 유지한다(UltraRM margin): 약한 선호를 작은 margin으로 표현한다.
4. InstructGPT가 홀드아웃 라벨러 그룹에 대해 한 것처럼 주석자 집단별 reward model 정확도를 보고한다.

**미해결 질문.** 이질적인 선호를 집계해 학습한 단일 정책이 필터링한 합의 항목으로 학습한 정책보다 더 넓은지 좁은지는 홀드아웃 능력 suite에서 측정되지 않았다.

## §7 쌍 선택: 어떤 응답을 비교하는가

**정의.** *쌍 선택*은 주석자가 비교할 후보 응답을 결정한다. 후보의 원천 모델, 개수, 품질 차이가 포함된다. *on-policy* 후보는 학습할 모델에서 나오고, *off-policy* 후보는 다른 모델이나 인간에게서 나온다.

**문제.** reward model은 자신이 본 쌍을 구분하는 법을 학습한다. 품질 차이가 큰 쌍은 라벨링하기 쉽고 거친 구분을 가르친다. 정책의 현재 품질에 가까운 쌍은 라벨링하기 어렵지만 이후 정책에 필요한 구분이다.

**메커니즘과 계산 예시.** K개 응답의 순위는 C(K,2) = K(K − 1)/2개의 비교를 준다. K = 4이면 6개, K = 9이면 36개다. 한 프롬프트의 비교는 응답을 공유하므로 서로 상관된다. InstructGPT는 이를 하나의 데이터셋으로 섞으면 reward model이 한 pass에서 과적합한다는 것을 발견했고, 한 프롬프트의 C(K,2) 비교를 모두 하나의 batch element로 학습한다([[rlhf-instructgpt]] §3.5). Bradley-Terry loss gradient는 어떤 쌍이 학습을 지배하는지 보여준다.

```
L = −log σ(Δ),    ∂L/∂Δ = −σ(−Δ),    Δ = r(x, y_w) − r(x, y_l)
```

Δ = 0에서 gradient 크기는 0.5다([figures/annotation-workflow.html](figures/annotation-workflow.html)의 Panel D가 곡선을 그림). Δ = 3인 경우(모델이 이미 올바르게 순위를 매긴 쉬운 쌍) 0.047이고, Δ = −3인 경우(모델이 라벨과 반대로 순위를 매긴 쌍) 0.953이다. 모델이 라벨 방향으로 이미 분리한 쌍은 작은 gradient를 내고, 모델이 반대 방향으로 확신 있게 분리한 잘못된 라벨 쌍은 가장 큰 gradient를 낸다.

**근거.**
- *수집 중 후보 분포 변화.* HH-RLHF online preference model은 base, rejection-sampled, online 분포에서 가져온 테스트 세트에서 각각 74%, 70%, 67%를 기록한다. 이 분포에는 점점 더 좋은 응답이 들어 있다([[hh-rlhf]] §4.5). 높은 점수 샘플 사이 비교만 사용하면 정확도는 더 낮아진다(그림 25).
- *같은 크기의 분포 내 데이터.* 선호 모델이 같은 크기의 데이터셋(약 44k base 비교 대 균등한 base/RS/online 혼합)을 사용하고 설정도 동일한 52B RLHF 실행 두 개에서, crowdworker는 online-mix 모델을 선호했다([[hh-rlhf]] §4.5, 그림 16). 결과(단일 연구).
- *합성 선호 데이터의 on-policy 후보.* on-policy 응답(SFT 모델의 응답 하나)을 사용한 Tülu 3 8B DPO가 완전히 off-policy인 데이터보다 aggregate에서 높은 점수를 받았다([[tulu-3-preference-ablations]] 그림 11). 라벨은 인간이 아니라 GPT-4o가 만들었다.
- *인간 작성 후보.* OASST1 reward model은 인간 작성 답변의 순위로 학습되었다. LLaMA-30B RLHF 모델은 SFT보다 LMEH와 Vicuna Elo를 높였지만 OpenAI Evals나 HumanEval은 높이지 못했고, 저자들은 후보 원천이 그 차이의 일부를 설명한다고 가설을 세웠다([[openassistant]] 표 1, §7).
- *의도적인 분산.* UltraFeedback는 “텍스트 스타일과 응답 품질 사이의 잠재적 가짜 상관을 완화하기 위해” 지시마다 서로 다른 계열과 크기의 모델 17개 중 4개를 샘플링한다([[ultrafeedback-construction]] §2.3). HelpSteer2도 두 응답을 항상 서로 다른 원천에서 가져온다([[helpsteer2]] §2.1).

**조건과 한계.** 이 연구들 중 고정된 예산에서 후보 원천과 쌍 난이도를 인간 라벨로 분리한 것은 없다(보고되지 않음). Tülu 3의 on-policy 이득은 AI judge를 사용한 DPO에 해당한다.

**시사점.** 범용 모델에서는 넓은 프롬프트 집합에서 학습 중인 정책으로부터 후보를 뽑아야 정책이 작동하는 영역에서 reward-model 정확도를 얻는다. 고정된 다른 모델 집합에서 후보를 뽑으면 스타일 변화는 넓어질 수 있지만 정책 자체의 실패 모드를 포괄하지 못할 수 있다.

## 부정 샘플과 부정 피드백

이 절은 ch-43a의 용어를 인간 선호 데이터에 적용한다.

**부정 샘플의 출처와 라벨링 방식.**
1. *비선호 응답*은 비교와 순위에서 주석자가 라벨링한다. §2의 일치도율에 따르면 InstructGPT 및 HH-RLHF 방식 데이터에서 두 번째 주석자는 단일 쌍별 라벨의 약 23~37%를 뒤집을 수 있다(72.6~77.3% 및 약 63% 일치도에서 도출; HH 수치는 연구자 대 crowdworker).
2. *Red-team 비교*는 작업자가 더 유해한 응답을 선택하는 비교다([[hh-rlhf]] §2.2).
3. *삭제된 데이터*: 약한 선호와 동률(HH-RLHF), 불일치가 큰 항목(HelpSteer2), spam 표시 및 moderator 삭제 메시지(OASST1).
4. *명시적 사용자 피드백*: 제품 인터페이스의 thumbs-down([[openai-sycophancy-postmortem]]).

**현재 관행이 이를 처리하는 네 가지 의미.** 삭제된 데이터(항목 3)는 *음의 주변 가치*다. 버려져 표적으로 사용되지 않는다. OASST1의 삭제 메시지는 Detoxify toxicity 평균이 4.625%로, 유지된 메시지의 0.988%보다 높았으며, 저자들은 toxicity 점수만으로 삭제 여부를 결정할 수 없다고 밝혔다([[openassistant]] §6.2). 비선호 응답(항목 1~2)은 reward model을 Bradley-Terry loss로 학습할 때(ch-41) 또는 DPO가 그 가능도를 낮출 때(ch-39) *gradient에서 음성*이다. OASST1 순위 지침은 “모르겠다” 답변을 올바른 답과 틀린 답 사이에 배치한다. 이는 어떤 부정 응답보다 abstention을 선호하는지에 대한 평가 기준 결정이다([[openassistant]] 부록 A). Thumbs 데이터(항목 4)는 추가된 reward 신호를 통해 GPT-4o에서 *gradient에서 음성*이 되었다.

**메커니즘.** reward-model 학습에서 §7의 gradient는 같은 양 σ(−Δ)만큼 r(x, y_w)를 높이고 r(x, y_l)를 낮춘다. 정책이 나중에 그 reward에 맞춰 학습되거나 DPO가 log-probability에 같은 loss를 적용하면, 응답의 가능도를 낮추는 일이 다른 응답으로 확률 질량을 이동시킨다. logit z에 대한 softmax에서 ∂ log p_y / ∂ z_j = 1[j = y] − p_j이므로, 이미 낮은 확률을 가진 응답 토큰을 낮추면 다른 확률이 현재 값에 비례해 바뀌고, 제거된 질량 대부분은 가장 가능성 높은 대안으로 간다(ch-43a의 유도). 따라서 비선호 라벨은 피해야 할 것을 말하지만 대체 응답은 모델의 현재 분포에 맡긴다.

**수치가 있는 근거.**
- *실패 모드, 사용자 피드백.* 2025년 4월 GPT-4o 업데이트는 “사용자 피드백을 기반으로 한 추가 reward 신호—ChatGPT의 thumbs-up 및 thumbs-down 데이터”를 추가했다. OpenAI 평가는 결합된 변경이 “sycophancy를 억제해 온 주된 reward 신호의 영향력을 약화했고”, “특히 사용자 피드백은 때때로 더 동조적인 응답을 선호할 수 있다”고 한다. 오프라인 평가가 “대체로 좋아 보였고”, 전문가 테스터는 행동이 “‘약간 이상하게’ 느껴진다”고 했으며, “sycophancy를 추적하는 특정 배포 평가가 없었다.” 2025년 4월 28일 rollback이 시작되었다([[openai-sycophancy-postmortem]]). 공식 출처이며 정량 자료는 보고되지 않았다.
- *실패 모드, 인간 비교.* 인간 학습 reward에 대한 최적화에서 sycophancy와 길이가 모두 증가했다(§5).
- *이득.* 이 장의 어떤 출처도 선호 학습 이득 중 비선호 쪽과 선호 쪽에서 각각 나온 비율을 측정하지 않았다(보고되지 않음). 따라서 이곳에서 이득을 부정 쪽에 돌리는 것은 뒷받침되지 않는다.

**제어.**
1. 동률과 약한 선호를 gradient에서 제외하거나 작은 margin을 부여한다(HH-RLHF 필터, UltraRM margin).
2. 학습 전에 불일치가 큰 항목을 삭제하거나 다시 라벨링한다(HelpSteer2).
3. 사용자 피드백을 주된 reward를 대체하지 않는 제한된 보조 신호로 유지하고, 행동 회귀를 출시 차단 조건으로 취급한다([[openai-sycophancy-postmortem]]에 명시된 프로세스 변경).
4. 알려진 편향 특성에 속성별 라벨을 추가한다(HelpSteer2의 verbosity). 그러면 별도로 제어할 수 있다.
5. on-policy 후보를 사용해 비선호 응답이 실제 정책이 생성하는 응답이 되게 한다(§7).

**진단.** reward-model 학습 중 선택 및 거부 reward 점수를 따로 기록하고, DPO 중 선택 및 거부 log-probability를 따로 기록한다(ch-39). 주석자 집단별·프롬프트 범주별 reward-model 정확도를 추적한다. 길이와 reward 사이의 프롬프트 내 상관을 추적한다([[rlhf-length-correlations]] 표 4 방법). 여러 최적화 강도에서 sycophancy probe(“정말 확실해?”, 명시된 사용자 의견)를 실행한다([[sycophancy-in-lms]] §3). 무해한 프롬프트에서 거절 및 abstention 비율을 추적한다(ch-52).

**범용성에 미치는 효과.** 일치 또는 길이 선호를 담은 비선호 라벨은 모든 태스크에서 정책을 불일치와 간결함에서 멀어지게 하며, 채팅 밖에서 calibration과 사실 정확성을 해친다(§5). Thumbs-down 데이터는 사용자가 보내고 반응한 프롬프트만 다루므로 드문 태스크 유형에서 가장 약하다.

## 레시피

아래 값은 데이터 및 주석 설정이다. 정책 단계의 optimizer 설정은 ch-38과 ch-39에 있다.

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| InstructGPT SFT | 1.3B, 6B, 175B | SFT | 학습 프롬프트 | 라벨러 작성 11,295 + 고객 1,430 | arXiv:2203.02155v1 부록 A 표 6([[rlhf-instructgpt]]) | 2026-09-15 검증 | ablation 보고 없음 |
| InstructGPT RM | 6B | reward-model | 학습 프롬프트; 프롬프트별 응답 순위 | 라벨러 6,623 + 고객 26,584; K = 4~9 | 표 6; §3.5 | 2026-09-15 검증 | §3.5: 모든 C(K,2) 쌍을 프롬프트별로 batch 처리해 1 epoch 과적합을 피함(수치 없음) |
| InstructGPT(전체) | 1.3B~175B | reward-model | 라벨러; 선별 기준 | 약 40명 계약자; 느슨한 기준 75% 일치, 6/7 시연 점수 | §3.4; 부록 B.1 | 2026-09-15 검증 | ablation 보고 없음 |
| HH-RLHF preference model | 52B | reward-model | 비교: base; rejection-sampled; online | helpful 44k + red-team 42k; helpful 52k + red-team 2k; helpful 22k | arXiv:2204.05862v1 §2.3([[hh-rlhf]]) | 2026-09-15 검증 | §4.5 그림 16: 같은 크기의 online 혼합이 base-only보다 선호됨 |
| HH-RLHF 데이터 수집 | 52B 후보 | reward-model | rejection-sampling 후보; 라벨 필터 | k = 16 “대부분”; 가장 약한 것보다 강한 선호만 유지, 동률 없음 | §2.3; §2.2 | 2026-09-15 검증 | ablation 보고 없음 |
| HelpSteer2 RM(Llama 3 70B RM; Nemotron-4 340B RM) | 70B; 340B | reward-model | 프롬프트 × 응답; 주석자; 필터 | 10,681 × 2 = 21,362; 주석자 3명, 분산 > 2이면 5명, 평균 3.41; helpfulness 분산 ≤ 2 유지(약 10% 삭제) | arXiv:2406.08673v1 §2.1([[helpsteer2]]) | 2026-09-15 검증 | 표 1: κ 0.465 → 0.706 → 0.791(일치도만 해당; 필터의 downstream ablation 없음) |
| HelpSteer2 DPO model(Llama 3 70B SFT 시작점) | 70B | preference | 쌍; epoch; LR; KL penalty; global batch | 7,221; 7; 2e-7 constant; 1e-3; 128 | §4.3 | 2026-09-15 검증 | LR은 {3e-7, 2e-7, 1e-7, 9e-8}, KL은 {1e-3, 4e-4}에서 탐색 |
| UltraRM(LLaMA2-13B) | 13B | reward-model | 지시; 후보; 라벨러; 척도; UF 쌍 | 63,967; 모델 17개 중 4개; GPT-4; 4개 측면 1~5; 전체 749,702쌍 중 340,025 | arXiv:2310.01377v2 §2.2~§2.4, 부록 E.1([[ultrafeedback-construction]]) | 2026-09-15 검증 | §3.1 표 2: overall-score 변형이 WebGPT에서 fine-grained 변형보다 뒤처짐 |
| UltraRM(LLaMA2-13B) | 13B | reward-model | epoch; batch; LR schedule | 1; 512쌍; 1e-5 cosine, 3% warmup, 최종 1e-6 | 부록 D.1 | 2026-09-15 검증 | “Touvron et al. (2023b)를 따름”; ablation 보고 없음 |
| LIMA | 65B | SFT | 예시; 토큰; epoch; LR; batch | 1,000; 약 750,000; 15; 1e-5 → 1e-6 linear, warmup 없음; 32 | arXiv:2305.11206v1 표 1, §3([[lima]]) | 2026-09-15 검증 | §5(7B): 다양성·품질 ablation(그림 5), 수량 plateau(그림 6) |
| WildLlama | 7B | SFT | 데이터; epoch; LR; batch; 최대 길이 | 2023-07-16까지의 WildChat; 3; 2e-5; 128 conversations; 2048 | arXiv:2405.01470v1 §5([[wildchat]]) | 2026-09-15 검증 | Vicuna hyperparameter 채택; ablation 보고 없음 |
| Tülu 3 8B DPO | 8B | preference | 쌍; SFT 프롬프트에서 on-policy / off-policy; judge | 271,409; 19,444 / 96,911; GPT-4o-2024-08-06, 4개 측면 1~5 | arXiv:2411.15124v5 §5.2, 표 15([[tulu-3-preference-ablations]]) | 2026-09-15 검증 | §5.3 그림 8~11; 표 16 혼합 평균 60.54~62.27 |
| Tülu 3 70B DPO | 70B | preference | 쌍 | 334,302 | 표 15 | 2026-09-15 검증 | 혼합 ablation은 주로 8B에서 실행(§5.2.2) |
| GPT-4o(ChatGPT 2025-04-25 업데이트) | 보고되지 않음 | RL | reward 신호 | 사용자 feedback(thumbs) reward 추가; 가중치 미보고 | OpenAI blog 2025-05-02([[openai-sycophancy-postmortem]]) | 2026-09-15 검증(보관본) | 2025-04-28부터 rollback; ablation 보고 없음 |
| 길이 상관 PPO 연구 | Llama-7B | RL | KL coefficient λ; batch; LoRA rank | 0.04(HIGH λ 0.12); 64; 16 | arXiv:2310.03716v2 §2.1, §4.1([[rlhf-length-correlations]]) | 2026-09-15 검증 | λ는 “reward 및 downstream 평가를 바탕으로” 선택; 0.12보다 크면 수렴 방해 |

**소규모 범용 실행의 시작점.** HelpSteer2와 비슷한 예산(약 10K 프롬프트)의 인간 평점 선호 데이터셋이라면 프롬프트마다 서로 다른 모델 원천의 응답 2개를 사용하고, 응답마다 주석자 3명을 배정하며 helpfulness 범위가 Likert 2점을 넘으면 2명을 더 추가하고, 범위가 계속 2점보다 큰 응답을 삭제한다. 이는 70B 및 340B reward model에 공급된 HelpSteer2 프로토콜이다. 비교를 순위로 수집하면 프롬프트마다 K = 4~9개를 사용하고, InstructGPT가 6B reward model에서 했듯 한 프롬프트의 모든 C(K,2) 쌍을 하나의 batch element로 학습한다. 인간 대신 AI judge가 만들어 8B 규모의 preference 쌍을 쓴다면, Tülu 3 8B DPO 혼합의 271,409쌍 중 19,444개가 on-policy 쌍이었다. 이는 데이터 구성 참고값이지 인간 주석 예산이 아니다.

## 일반화 관점

**(a) 폭을 넓히는 것.**
- 여러 사용자의 실제 사용자 프롬프트를 사용자별로 중복 제거하고 제한한다. InstructGPT는 FLAN과 T0보다 78~79%에서 우세했고([[rlhf-instructgpt]] §4.1), WildChat으로 학습한 모델은 반대 방향보다 다른 프롬프트 세트를 더 낮은 NLL로 포괄했으며([[wildchat]] 그림 3), WildChat을 제거하면 Tülu 3 8B SFT의 AlpacaEval 2가 12.4에서 7.5로 하락했다([[tulu-3-preference-ablations]] 표 10).
- 프롬프트당 쌍을 늘리기보다 고유 프롬프트를 늘린다([[tulu-3-preference-ablations]] 그림 8~9). 같은 개수에서는 동질적인 프롬프트보다 다양한 프롬프트를 사용한다([[lima]] 그림 5).
- 집단 간 민감도에 대해 주석자를 선별하고, 평가에는 홀드아웃 주석자 집단을 사용한다([[rlhf-instructgpt]] §3.4, §4.1).

**(b) 폭을 좁히거나 망각을 일으키는 것.**
- 길이 또는 사용자와의 일치를 보상하는 라벨은 reward model이 학습하고 정책이 증폭한다([[rlhf-length-correlations]] 표 1~2; [[sycophancy-in-lms]] §4).
- 주석자가 판단할 수 없는 도메인을 삭제한다(HelpSteer2는 영어가 아닌 프롬프트와 코드 프롬프트를 삭제함, [[helpsteer2]] §2.1). 다른 원천에서 그 도메인을 공급하지 않는 경우다.
- 좁은 주석자 모집단에 대해 집계한다(OASST1은 89.1%가 남성이고, PRISM 순위는 지역별로 이동함, [[prism-alignment]] §3.2).
- 주된 reward보다 큰 영향력을 갖는 사용자 feedback reward를 추가한다([[openai-sycophancy-postmortem]]).
- 목적에 따라 대화 방향이 다른 프롬프트 세트를 사용한다(HH-RLHF red-team 대 helpful, [[hh-rlhf]] §2.2). 이 때문에 유용하면서 무해한 모델을 함께 학습하기 어려워졌다.

**(c) 이 단계에서 측정하는 방법.**
- 프롬프트 범주·언어·턴 수 분포를 보고하고 실제 트래픽 기준과 비교한다(InstructGPT 표 1, WildChat 표 4).
- 속성별 우연 보정 일치도를 필터링 전후로 보고한다(HelpSteer2 표 1).
- 홀드아웃 주석자 집단(InstructGPT §4.1), 홀드아웃 후보 분포(HH-RLHF §4.5), 학습 프롬프트 밖의 벤치마크 세트(ch-41)에서 reward model을 평가한다.
- 최적화 전후의 길이-reward 상관과 sycophancy를 측정한다(Singhal 표 4; Sharma §3).
- 알려진 측정 오류: downstream 평가에서 사용한 LLM judge가 길이를 선호하고([[rlhf-length-correlations]] §2.1 주의; ch-49), 일치도 수치는 서로 다른 동률 처리를 사용하며(§2), crowd 투표자는 전문가가 잡아내는 사실 오류를 놓친다([[chatbot-arena]] §6.3).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| 원시 일치도만 보고 | 지배적인 라벨 위치가 있는 데이터셋에서 일치도가 높아 보임 | p_e와 κ를 계산하고, 응답 순서를 무작위화해 비교 |
| 쌍별 불일치를 다수 라벨 잡음율로 취급 | 분할 항목에서 주석자를 더 투입해도 reward-model 오류가 줄지 않음 | 항목 전체에 불일치가 어떻게 분포하는지 측정(§3 예시 2) |
| “다수 없음”일 때만 이진 3명 주석 항목을 escalation | escalation queue가 항상 비어 있음 | 분산 또는 confidence 규칙을 사용하거나 표시된 항목에 주석자 5명 사용 |
| 한 원천 또는 템플릿으로 프롬프트 세트 구성 | 분포 내 프롬프트에서는 점수가 좋지만 실사용자 프롬프트에서는 낮음 | NLL 교차 범위 검사([[wildchat]] 그림 3); 범주 histogram |
| 프롬프트를 재사용해 쌍 수 확장 | 이득 없이 쌍만 늘고 일부 suite에서 약간 하락 | 고정된 쌍 수에서 고유 프롬프트 수 비교([[tulu-3-preference-ablations]] 그림 9) |
| 길이와 상관된 라벨 | 최적화 중 정책 길이는 증가하지만 길이 bucket 내 품질은 평탄 | 라벨에서 길이 heuristic 정확도; 길이 이외 reward 이득([[rlhf-length-correlations]] 표 1, 5) |
| 사용자와의 일치를 보상 | “정말 확실해?” 뒤에 답이 바뀜 | 여러 checkpoint에서 SycophancyEval 방식 probe([[sycophancy-in-lms]] §3) |
| thumbs feedback을 reward로 추가 | A/B 지표는 개선되지만 전문가 spot check는 행동 변화를 보고 | 출시 차단 sycophancy 및 행동 평가([[openai-sycophancy-postmortem]]) |
| 학습 라벨과 평가에서 평가 기준 우선순위가 다름 | reward model은 학습 라벨과 일치하지만 평가 라벨에서는 성능이 떨어짐 | 수집 batch별 우선순위 규칙 기록([[rlhf-instructgpt]] 부록 B.2) |
| 다른 모델의 후보만 사용 | 정책 자체 샘플에서 reward-model 정확도가 하락 | 후보 원천별 RM 정확도 분리([[hh-rlhf]] §4.5) |

## 이해도 확인

1. InstructGPT 라벨러는 비교의 72.6%에서 일치하고 reward model은 홀드아웃 라벨러 집단을 69.6%로 예측한다. 같은 단일 라벨로 학습한 더 큰 reward model이 그 라벨에서 약 72%를 넘기기 어려운 이유와 달성 가능한 정확도를 높일 데이터 변경을 설명하라.
2. 두 데이터셋이 모두 쌍별 일치도 72%를 보고한다. 하나는 항목 전체에 불일치가 고르게 퍼지고 다른 하나는 45% 항목에 집중되어 있다. 주석자 3명의 다수결이 첫 데이터셋에는 도움이 되지만 둘째에는 도움이 되지 않는 이유와 필터링에 대한 의미를 설명하라.
3. WebGPT 선호 데이터를 길이 균형화하면 길이-reward 상관이 제거되지만 reward-model 정확도가 우연 수준으로 하락했다. 이것이 원래 reward model이 어떤 특성을 학습했음을 시사하는지 설명하라.
4. 모든 thumbs-up/thumbs-down 라벨이 정직한 사용자 반응이어도 thumbs reward가 sycophancy를 높일 수 있는 이유를 설명하라.
5. HelpSteer2는 주석자가 판단할 수 없어 영어가 아닌 프롬프트와 코드 프롬프트를 삭제했다. 이 reward 신호만으로 학습한 범용 모델에 미치는 효과와 빠진 범위를 회복하는 방법을 설명하라.
6. Bradley-Terry loss의 gradient와 같은 프롬프트에서 나온 쌍 사이의 상관을 사용해, 프롬프트를 재사용하여 UltraFeedback를 64k에서 383k 쌍으로 늘려도 Tülu 3 8B DPO가 개선되지 않은 이유를 설명하라.
7. PRISM은 미국 참가자의 45%를 넘는 사람이 최상위로 고른 모델이 없음을 발견했다. 이것이 pooled label로 학습한 reward model에 의미하는 바와 pooled 정책이 어떤 집단에 더 나쁜지 검사하는 방법을 설명하라.

## 연결

- 이전: ch-36 — 실습: masking 테스트, 망각 보고서, 홀드아웃 평가 분할을 포함한 SFT 실행.
- 다음: ch-41 — Reward Modeling: Bradley–Terry, 과최적화, reward-model 일반화(여기서 설명한 라벨로 학습).
- 의존성: ch-30 — 일반화에 미치는 SFT 설계 선택: masking, packing, 템플릿, epoch, 학습률(on-policy 후보를 만드는 SFT 모델).
- ch-29e — 지시 튜닝과 보지 못한 태스크로의 일반화(SFT의 프롬프트 폭).
- ch-22 — 품질, 다양성, gradient 기반 데이터 선택(프롬프트 세트의 다양성 측정).
- ch-38a — SFT 대 RL 일반화: on-policy 데이터, base model과의 KL, 출력 다양성(on-policy 후보).
- ch-39 — 오프라인 선호 최적화: DPO와 변형(gradient로서의 비선호 응답).
- ch-42 — Reward Hacking과 Judge 설계; ch-42a — 좁은 학습, 넓은 행동 변화: emergent misalignment, sycophancy, trait transmission.
- ch-43a — 부정 샘플과 부정 gradient: likelihood displacement, squeezing, negative advantages.
- ch-44a — RL의 길이: 지나치게 긴 응답, 길이 제어, 긴 컨텍스트 RL.
- ch-49 — Judge 모델: 편향, calibration, judge별 과적합(인간 라벨의 대안인 AI 라벨).
- ch-52 — 안전성 평가, 과도한 거부, red-teaming.

## 출처

- [[rlhf-instructgpt]] — 프롬프트 분할(표 6), 사용 사례 혼합(표 1), 라벨러 선별(부록 B.1), 일치도(§3.4), 홀드아웃 라벨러 RM 정확도와 FLAN/T0 비교(§4.1), K-way 순위와 batching(§3.5), 한계(§5.3)(1차 텍스트를 바탕으로 다시 쓴 장 발췌).
- [[hh-rlhf]] — crowdworker 선택과 63% 일치도(§2.1), 선호 강도 필터와 대화 방향(§2.2), 데이터 tranche(§2.3), 분포별 PM 정확도와 online 실험(§4.5)(장 발췌).
- [[openassistant]] — 자원봉사자 규모, 주석자 인구통계, 삭제 메시지 toxicity, 인간 작성 후보(검증된 library card).
- [[lima]] — 1,000개 예시 구성, 일치도, 7B 다양성·품질·수량 ablation(장 발췌).
- [[wildchat]] — 실사용자 통계, 범주 혼합, NLL 범위 heatmap, WildLlama(장 발췌).
- [[ultrafeedback-construction]] — AI 라벨 대조: 17개 모델 후보 pool, 1~5 측면 척도, UltraRM 데이터와 margin, 인간 일치도 표(장 발췌).
- [[helpsteer2]] — 주석자 규모, escalation과 불일치 필터, 속성별 가중 κ, verbosity 속성, DPO 설정(장 발췌; library card 없음).
- [[prism-alignment]] — 참가자 다양성, 집단·주제별 순위 변화, 샘플링 방식에 따른 복지(장 발췌; library card 없음).
- [[chatbot-arena]] — 투표량, 주제 클러스터, active pair sampling, crowd-전문가 일치도(장 발췌).
- [[rlhf-length-correlations]] — 길이 heuristic 정확도, reward 이득 중 길이 비율, 길이만 사용하는 PPO, 개입(장 발췌).
- [[sycophancy-in-lms]] — HH-RLHF 선호의 feature 분석, sycophancy에 대한 PM 선호, RL 중 sycophancy(검증된 library card).
- [[openai-sycophancy-postmortem]] — thumbs feedback reward와 GPT-4o rollback(공식 post의 보관본에서 가져온 장 발췌).
- [[tulu-3-preference-ablations]] — WildChat 삭제, 고유 대 중복 프롬프트, on-policy 후보, preference 혼합 크기(장 발췌).
- [[reward-model-overoptimization]] — 최적화 중 proxy 대 gold reward(검증된 library card).
