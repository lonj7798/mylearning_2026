# 23a — 지식 long-tail 논문의 근거 범위

<!-- scope: primary-source setups, measured outcomes, and transfer limits for rarity units, sampling support, retention bias, and source-grounded knowledge coverage
     deps: [[read_kor]]
     see-also: [[read_kor]], [[practice_kor]], [[long-tail-papers]]
     sources: [[long-tail-knowledge]], [[graphgen]], [[knowledge-density-synthesis]], [[data-value-scaling]], [[long-tail-knowledge-survey]]
-->

## 1. 관련 문서와 사실 QA

**Result (single study).** Kandpal 등의 [[long-tail-knowledge]]는 질문·답의 핵심 개체를 연결하고 두 개체가 들어 있는 pretraining 문서 수를 셉니다(§2). TriviaQA/Natural Questions를 빈도 구간별로 평가합니다. BLOOM-176B의 TriviaQA 정확도는 관련 문서 수가 수십에서 수만으로 늘어날 때 약 25%에서 55% 이상으로 올라갑니다(§3.1). 이는 질문별 관련성이지 대화 말투를 교체한 통제 실험은 아닙니다.

관련 문서를 제거한 C4로 4.8B 모델을 재학습하는 실험도 합니다(§3.2). Retrieval은 특히 희소 질문에 도움이 됩니다(§4.3). 부족한 사실을 제공하라는 근거이지 모든 희소 대화 실패가 지식 부족이라는 근거는 아닙니다. 개체 연결과 corpus 선택에 영향을 받으며 두 개체의 동시 등장도 유용한 근거의 대리값입니다.

## 2. GraphGen: 합성 전에 구조를 만든다

**Result (single study).** [[graphgen]]은 원문에서 지식 그래프를 만들고 모델의 이해를 조사하며 atomic·aggregated·multi-hop 질문을 만듭니다. 단일 관계, 관련 관계 묶음, 연결된 여러 사실을 거치는 질문으로 노출을 바꿉니다. 흔한 사실을 표현만 바꾸는 것과 다릅니다.

§4 Step 2 Eq.4는 참·부정문에 대한 cross-entropy로 comprehension을 조사합니다. 초록은 ECE 용어를 사용하지만 방법 전체를 일반적인 구간별 calibration error 계산이라고 바꾸어 설명하면 안 됩니다. 이 probe에서의 높은 loss가 실제로 중요한 long tail을 입증하지는 않습니다.

**Interpretation.** 상품 자료의 관계를 추출·조사하고 근거 질문을 만들 수 있습니다. 그래프 관계와 부정문도 검증해야 합니다. 잘못된 관계나 부정이 정답을 오염시킵니다. 날짜 수정·스킬 범위는 필요한 절차 조건까지 표현하지 않으면 사실 그래프만으로 부족합니다. 세일즈 전이는 **Open question**입니다.

## 3. KDoS와 혼합 데이터 scaling

**Result (single study).** [[knowledge-density-synthesis]]는 knowledge density의 피드백으로 생성 분포를 조절합니다. 검증된 범위는 post-training/SFT이며 pretraining이나 임의 대화 스킬까지 입증하지 않습니다. 부족한 지식을 조사하고 생성 예산을 배분한 뒤 재측정하는 구조를 참고할 수 있지만 최적 희소 비율은 아닙니다.

[[data-value-scaling]]은 실제·합성 혼합에서 head/tail 학습 국면과 가치 평가를 분석합니다. [[long-tail-knowledge-survey]]는 정의·개입을 정리하는 **Review**이지 새 통제 실험이 아닙니다. DataValue §5.2의 세 국면 실험은 ResNet-18/CIFAR-100에 CIFAR-100-C의 corrupted images를 합성 대리값으로 사용합니다. Zipf 가중 클래스와 실제 비율0.0625의 통제 실험이며 자연스러운 LLM 대화 검증은 아닙니다. NLP subset utility 실험은 별도 근거입니다.

## 4. 실험 설정과 측정값

| 연구 | 설정·지표 | 결과·한계 |
|---|---|---|
| GraphGen §5/Fig.3 | Qwen2.5-72B 생성 / 7B-Instruct 학생; 농업·의료·multi-hop QA | 최선 baseline 대비 ROUGE-F 1.08/2.70/4.73점; 사실 정확도 아님 |
| KDoS Table 1 | Qwen3-4B-Base; benchmark 평균 | Random 18.3, Uniform 20.9, Quality 20.6, Difficulty 21.5, KDoS 22.8; SimpleQA-Verified는 Difficulty 8.2 > KDoS 8.0 |

KDoS density는 token 수 / 의미 hypersphere부피이며 반지름은 중심으로부터 평균 embedding거리입니다(§3.1). 사실 관계 수/token이나 보편적 품질이 아닙니다. 그룹과 가까운 두 이웃으로 질문 후보 다섯 개를 만들고 품질 검사와 density rejection을 나눕니다. Cold start에서는 먼저 품질 후보를 받아 density를 구축합니다. Table 2는 전체22.8,품질 제거21.7,품질+density제거17.6,seed 17.4입니다. 상대%가 아닌 해당 실험 점수이며 세일즈 처방이 아닙니다.

GraphGen Table 2의 평균 길이도 657.9 tokens와 LongForm 357.1 tokens로 다릅니다. 어휘·학습 효과 비교에는 길이와 노출도 고려해야 하며 방법 이름만으로 효과 원인이 분리되지 않습니다.

## 5. 근거의 경계

Closed-book QA는 사실 범위의 가설을 만들 수 있지만 말투·화자·도구 정답을 입증하지 않습니다. 공백 점수는 조사 순서를 정하는 신호이며 probe 자체에도 사각지대가 있습니다. 합성 질문의 이름만 바꾼 시험 대신 독립적인 항목을 사용합니다. 결론은 출처의 조건보다 넓게 쓰지 않습니다.

[본문](read_kor.md) · [구성·평가](practice_kor.md) · [English](methods.md).

