<!-- chapter: ch-47
     track: eval
     kind: content
     title: Evaluation Harness and Suite Design for General Capability
     deps: [ch-00, ch-36]
     sources: [[tulu-3]], [[tulu-3-eval-regime]], [[olmes]], [[helm]], [[mmlu-pro]], [[ifeval]], [[prompt-format-sensitivity-formatspread]], [[multi-prompt-evaluation]], [[thinkingmachines-defeating-nondeterminism]], [[signal-and-noise-eval]], [[helmet]], [[nolima]], [[context-length-alone-hurts]], [[chroma-context-rot]], [[lost-in-the-middle]], [[holistic-agent-leaderboard]], [[agentic-benchmark-checklist]]
     figures: figures/eval-config-matrix.html
     revised: 2026-09 (generality revision)
-->

# 제47장 — 범용 능력을 위한 평가 하네스와 스위트 설계

> **핵심 통찰.** 보고된 점수는 세 가지가 함께 만들어 낸다. 항목 집합, 각 항목을 제시하고 채점하는 방식, 답을 생성한 추론 설정이다. 한 모델·한 과제의 발표 수치는 두 번째·세 번째 요소의 크기를 보인다. OLMES Table 1은 Llama2-7B의 ARC-Challenge 점수 7개, 43.2에서 54.2를 나열한다. 각각 shot 수, 정식화, 정규화가 다른 참조에서 왔다([[olmes]] Table 1). *일반* 능력을 측정하려면 ch-00의 네 번째 요구가 더해진다. 스위트는 주장하는 능력 축을 포괄해야 하며 각 축에는 개발 중 전혀 읽지 않은 벤치마크가 필요하다. Tülu 3 최종 8B 체크포인트의 development suite 68.8과 unseen suite 32.4는 서로 다른 질문에 답하며 둘 다 필요하다([[tulu-3-eval-regime]] Table 31).
>
> **지침.** 수치를 보고할 때는 과제 id, 정식화·정규화, matcher, few-shot 예의 수와 원천, 생성 설정, harness 버전, 모델 revision도 함께 보고하라. 아래 소스에서 각 항목이 1점 이상의 측정 효과를 보이기 때문이다. 범용 모델 스위트를 만들 때는 능력 축마다 development와 unseen 벤치마크 하나씩을 할당하고 각 과제가 어떤 결정을 뒷받침하는지 기록하라. unseen 절반 없는 스위트는 기술 이득과 벤치마크 특이 이득을 분리하지 못한다([[tulu-3]] §7.4). 사전학습 중 base 체크포인트를 비교할 때는 cloze 정식화 또는 bits-per-byte를 사용하라. 모델이 형식을 얻기 전에는 multiple-choice 정식화가 chance 근처에 머물기 때문이다([[olmes]] §3.4; [[signal-and-noise-eval]] §5.3). 모델이 multiple-choice 정식화에서 chance를 넘으면 둘 중 높은 값을 보고하라([[olmes]] §3.4). 주장이 최고 능력이 아니라 강건성에 관한 것이면 과제별 prompt template 여러 개를 평가하고 평균·spread를 보고하라. plausible 대안 10개에서 템플릿 하나의 중앙 spread가 정확도 7.5점이기 때문이다([[prompt-format-sensitivity-formatspread]] §4.2). 1점 수준에서 실행 사이 점수를 비교할 때는 추론 스택도 고정하라. 온도 0에서 같은 weights·같은 prompt가 서버 batch 크기를 바꾸자 1,000개 중 80개의 서로 다른 completion을 냈기 때문이다([[thinkingmachines-defeating-nondeterminism]]).

## 이 장이 범용 모델에 중요한 이유

ch-00은 일반 능력을 학습 대상도 개발 중 읽은 대상도 아닌 과제에서 측정한 성능으로 정의하고, 그러한 모든 수치에 적용되는 측정 오류를 제시한다. ch-36은 홀드아웃 분할과 함께 SFT 체크포인트를 만들었다. 이 장은 둘이 의존하는 도구를 만든다. 즉 스위트(어떤 항목, 어떤 분할, 어떤 결정을 뒷받침하는가)와 하네스(체크포인트와 과제 정의를 점수 및 항목별 기록으로 바꾸는 프로그램)다.

두 실패가 이 작업을 이끌며 둘 다 측정 가능하다. 첫째 비교 실패다. 같은 과제·같은 모델의 수치도 설정이 다르면 비교할 수 없고, 설정은 대개 공개되지 않는다. HELM은 자체 표준화 전 연구한 모델 30개가 16개 core scenario 중 평균 17.9%만 평가됐고, 공유 데이터셋조차 서로 다른 조건에서 실행됐음을 발견했다([[helm]] Abstract, §1). 둘째 일반성 실패다. 개발 중 읽은 스위트는 목표가 되므로 점수는 기술이 아니라 벤치마크 적합을 측정한다. Tülu 3는 결정이 끝날 때까지 두 번째 스위트를 읽지 않음으로써 이를 직접 측정했다([[tulu-3]] §2.2, §7.4).

하네스는 파이프라인 끝(사전학습 → 중간학습 → SFT → 선호 최적화 → RL → 평가)에 있지만 이전 모든 단계가 여기서 읽는다. 데이터 혼합 절제, 체크포인트 선택 규칙, 보상 설계는 모두 하네스 출력 차이로 결정된다. 1–3점 수준에서 부정확하거나 기록되지 않은 도구는 그 수준의 결정을 신뢰할 수 없게 만든다. 이는 ch-51의 주제다.

## §1 능력 커버리지 맵에서 스위트까지

**정의.** *능력 커버리지 맵*은 모델이 가진다고 주장하는 능력 축 목록과, 축별로 이를 측정하는 벤치마크 및 각 벤치마크의 분할이다. *평가 스위트*는 실행 가능하게 만든 커버리지 맵이다. 과제 id, 항목 수, 분할, 설정, 각 과제가 뒷받침하도록 허용된 결정을 담는다.

**측정 가능한 문제.** 커버리지 맵이 없으면 벤치마크 선택은 가용성을 따르고 저렴한 벤치마크 없는 축은 측정되지 않는다. 그러면 모델은 보고한 평균을 올리면서 측정하지 않은 축의 능력을 잃을 수 있고, 보고서의 어느 수치도 움직이지 않는다.

**메커니즘.** 스위트 구축은 다섯 단계다.

1. **능력 축을 나열한다.** HELM 분류는 각 scenario를 과제·도메인·언어 삼중항으로 만들며, 모든 메트릭을 재는 16개 *core* scenario와 하나의 기술을 고립하는 26개 *targeted* scenario를 분리한다([[helm]] §1). 이 과정의 축 목록은 ch-00을 따른다. 지식, 추론, 수학, 코드, 지시 따르기, 긴 컨텍스트, 다국어, 도구 사용·에이전트 작업, 사실성·캘리브레이션, 안전·과잉 거부다.
2. **정확도만이 아니라 모든 축에서 잴 메트릭을 정한다.** HELM은 core scenario에서 메트릭 범주 7개(정확도, 캘리브레이션, 강건성, 공정성, 편향, 독성, 효율)를 재며 가능한 (scenario, metric) 쌍 112개 중 98개, 87.5%를 보고한다([[helm]] Abstract, §1).
3. **축별로 같은 기술을 요구하되 원천·형식·제약 집합이 다른 development와 unseen 벤치마크 하나씩을 고른다.** Tülu 3 쌍은 MMLU → MMLU-Pro·GPQA, BBH·DROP → AGIEval English, MATH·GSM8K → DeepMind Mathematics, HumanEval → BigCodeBench, IFEval·AlpacaEval 2 → IFEval-OOD·HREF다([[tulu-3-eval-regime]] Table 3).
4. **과제별로 뒷받침하는 결정을 기록한다.** 데이터 혼합 절제, 체크포인트 선택, release gate, 보고 전용이다. 체크포인트 선택에 쓴 과제는 이름과 무관하게 정의상 development 과제다.
5. **unseen 절반을 동결한다.** Tülu 3는 개발 중 unseen 점수를 보지 않았다고 명시한다. 이 때문에 §7.4의 dev-to-unseen 비교가 해석 가능하다([[tulu-3]] §2.2).

**근거.** 두 절반을 함께 읽으면 일반 이득과 벤치마크 특이 이득을 분리한다. Tülu 3 8B의 SFT → DPO → final에서 development 평균은 64.9 → 68.3 → 68.8, unseen 평균은 29.9 → 31.9 → 32.4로 움직여 파이프라인이 둘 다 개선한다. 축별 모습은 균일하지 않다. 지시 따르기는 IFEval 72.8 → 82.4, IFEval-OOD 17.6 → 24.3인데, 코딩은 8B에서 HumanEval 86.2 → 83.9, BigCodeBench 11.5 → 7.4이고 70B는 12.2 → 21.6이다([[tulu-3-eval-regime]] Table 31). **결과(단일 연구).**

그 격차 중 하나의 메커니즘은 기술 효과가 아니라 형식 효과라서 유익하다. DPO 선호 데이터를 키우면 MATH는 DeepMind Mathematics보다 더 개선했다. 저자들의 설명은 모델이 LaTeX 형식 추론·답을 내도록 학습했지만 DeepMind Mathematics는 그것을 요구하지 않아 추론과 답 추출을 방해했다는 것이다([[tulu-3]] §7.4.1, Fig. 24). **해석(논문 저자).**

**표로 만든 커버리지 맵.** 아래 행은 ch-00 축 목록과 각 소스가 공개한 벤치마크 할당을 결합한다. 처음 6행은 Tülu 3 자체 쌍([[tulu-3-eval-regime]] Table 3, Table 24)이고, 마지막 4행은 Tülu 3 스위트가 이 축을 덮지 않으므로 이 과정이 셀의 소스에서 조합했다.

| 축 | Development 벤치마크 | Unseen 벤치마크 | 형태 | 메트릭 | 뒷받침하는 결정 |
|---|---|---|---|---|---|
| 지식 | MMLU, PopQA, TruthfulQA | MMLU-Pro, GPQA | multiple choice / short answer | EM, MC2 | 데이터 혼합, 체크포인트 선택 |
| 추론 | BBH, DROP | AGIEval English | CoT 생성 | EM, F1 | 데이터 혼합 |
| 수학 | MATH, GSM8K | DeepMind Mathematics | 생성, 검증됨 | flex EM, EM(Sympy) | 데이터 혼합, RL 보상 설계 |
| 코드 | HumanEval, HumanEval+ | BigCodeBench(hard, 1,140 중 148) | 생성, 실행 | T=0.8에서 pass@10 | 데이터 혼합 |
| 지시 따르기 | IFEval, AlpacaEval 2 | IFEval-OOD(제약 52개), HREF | 생성, 프로그램 검사 / 판정 | prompt-level loose, LC 승률 | 체크포인트 선택 |
| 안전·과잉 거부 | Tülu 3 Safety(6 집합 macro) | 소스에 없음 | 생성, classifier 채점 | 거부 / compliance 정확도 | release gate |
| 긴 컨텍스트 | 8K–128K의 RAG·recall 범주([[helmet]] §2.1) | 128K의 downstream 범주; 낮은 overlap needle([[nolima]]) | 여러 길이에서 생성 | SubEM, 판정 점수 | position-scaling·긴 컨텍스트 데이터 결정 |
| 다국어 | 여기 사용한 소스에서 미포괄 | 미포괄 | — | — | — |
| 사실성·캘리브레이션 | 모든 core scenario의 메트릭으로서 calibration([[helm]] §1) | — | any | ECE, selective accuracy | release gate |
| 도구 사용·에이전트 | 고정 scaffold의 과제 특이 벤치마크([[holistic-agent-leaderboard]] §3) | 홀드아웃 환경(§8) | 다단계, 환경 채점 | 비용을 수반한 과제 성공 | scaffold·에이전트 데이터 결정 |

마지막 열은 가장 자주 암묵적으로 남는다. 체크포인트를 고르는 데 쓴 벤치마크는 체크포인트가 일반화한다는 증거를 겸할 수 없으며, 스위트 정의 때 역할을 기록해야 이후 두 용도를 혼동하지 않는다.

**조건과 한계.** Tülu 3에는 unseen 안전 벤치마크가 없으므로([[tulu-3-eval-regime]] Table 3) 안전 수치는 development 전용이다. unseen 벤치마크도 결정에 사용한 순간 unseen이 아니게 된다. 그 뒤에는 development 벤치마크이고 새 벤치마크가 필요하다.

**스위트 크기 정하기.** 항목 수는 예산 결정이며 클수록 항상 좋은 것은 아니다. [[signal-and-noise-eval]]은 ARC Easy 부분집합 1,000문항이 인스턴스 90% 적은 full MMLU보다 높은 결정 정확도를 냈으며, AutoBencher 33K 인스턴스가 연구한 벤치마크 중 최고 체크포인트 noise였다고 보고한다(App. B.2). HELM은 scenario당 1,000 인스턴스로 평가를 제한하고, OLMES는 데이터셋이 1,500을 넘으면 1,000 인스턴스를 샘플링한다. 추가 통계 신호가 다른 점수 변동 원천에 지배된다는 것이 명시한 이유다([[olmes]] §3.1). 계획 예: 축 10개 × 벤치마크 2개 × 항목 1,000개 = 전체 스위트 실행당 항목 20,000개다. 네 저렴한 축에서 template 하나 대신 prompt template 5개(§5)를 실행하면 (축 4개 × 벤치마크 2개 × 항목 1,000개) × 추가 template 4개 = 생성 32,000개를 더하며, base 스위트 비용보다 크다.

## §2 벤치마크 형태, 정식화, 메트릭

**정의.** *정식화(formulation)*는 multiple-choice 항목을 모델에 제시하는 방법이다. *multiple-choice 정식화*(MCF)에서는 선택지가 letter label과 함께 prompt에 나타나며 모델은 label로 채점된다. *cloze 정식화*(CF, completion이라고도 함)에서는 각 답 문자열을 질문이 주어진 답 문자열 확률로 모델이 별도 채점한다([[olmes]] §2.1).

**문제.** 두 정식화는 같은 weights에 서로 다른 수치를 주며 어느 쪽이 높은지는 모델 강도에 따라 달라 참조 간 비교가 순위를 뒤집을 수 있다.

**메커니즘과 수식.** CF는 문자열 raw 확률이 길이와 함께 떨어지므로 길이 정규화가 필요하다. OLMES는 넷을 비교한다([[olmes]] §3.3).

- none: `ln P(a_i | q)`
- token: `ln P(a_i | q) / num_tokens(a_i)`
- character: `ln P(a_i | q) / num_characters(a_i)`
- pmi: `ln [ P(a_i | q) / P(a_i | u) ]`, 여기서 `u = "Answer:"`

여기서 `a_i`는 답 선택지 *i*, `q`는 형식화한 질문 prompt, `u`는 질문을 제거한 unconditional prompt다.

**계산 예**(이 장을 위해 구성; 손으로 확인할 수 있음). 항목에 “iron”(4문자)과 “magma”(5문자) 두 선택지가 있다. `ln P(iron|q) = −6.0`, `ln P(magma|q) = −7.0`이고 unconditional로 `ln P(iron|u) = −5.0`, `ln P(magma|u) = −7.5`라고 하자.

| 정규화 | “iron” 점수 | “magma” 점수 | 예측 |
|---|---|---|---|
| none | −6.00 | −7.00 | iron |
| character | −6.0/4 = −1.50 | −7.0/5 = −1.40 | magma |
| pmi | −6.0 − (−5.0) = −1.0 | −7.0 − (−7.5) = +0.5 | magma |

바뀌지 않은 forward pass 하나에 세 채점 규칙이 두 다른 예측을 낸다. OLMES는 base 모델 15개에 대한 실험에서 과제별 규칙을 고정한다. ARC-Challenge, CommonsenseQA, OpenBookQA에는 pmi, ARC-Easy, HellaSwag, PIQA, Social IQa, MMLU에는 character, BoolQ와 WinoGrande에는 none이다. 과제별 선택은 모델별 최선에서 0.0–1.1점 안이다([[olmes]] Table 2, Table 3).

**정식화 효과의 근거.** ARC-Challenge에서 Llama3-8B는 Hugging Face Open LLM Leaderboard의 25-shot CF에서 60.2, Llama 3 모델 카드의 25-shot MCF에서 78.6이다. 둘 다 각자 설정 안에서는 맞다([[olmes]] Table 1). Llama3-70B는 MCF 93.7, CF 69.0이며 저자들은 오류율이 거의 5배 차이(6.3% 대 31%)라고 설명한다([[olmes]] §3.4). 반대 방향으로 base 모델 15개 중 가장 약한 8개는 ARC-Challenge MCF에서 random 근처, CF에서는 random 위다([[olmes]] §3.4, Fig. 2). HELM은 반대 쪽에서 같은 현상을 보고한다. OPT(175B)는 선택지를 별도 0-shot prompt에서 채점하면 HellaSwag 79.1%, calibrated separate 변형은 54.8%, 선택지를 공동 5-shot multiple-choice prompt에 제시하면 30.2%다([[helm]] §8.2, Fig. 33; Finding 23). **복제됨:** 독립 연구 둘에 걸쳐 있다.

**사전학습 모니터링에서는 시점이 중요하다.** MMLU에서 OLMo-7B-0424는 MCF에서 대략 학습 토큰 400B까지 random 근처에 머물고 그 뒤 더 강한 신호를 내지만 CF는 처음부터 신호를 담는다([[olmes]] Fig. 1). 그래서 OLMES는 두 정식화를 모두 실행하고 더 높은 것을 보고한다([[olmes]] §3.4). loss 같은 대안으로 [[signal-and-noise-eval]]은 bits-per-byte(정답의 negative log-likelihood를 UTF-8 byte 수로 나눈 것) 채점이 30과제 평균 signal-to-noise ratio를 10.0에서 31.5, 작은 규모 결정 정확도를 77.0%에서 83.7%로 올렸다고 보고한다(§5.3, Fig. 6).

**체크포인트를 분리할 수 있는 벤치마크 고르기.** 벤치마크는 보이는 차이가 자체 noise를 넘을 때만 결정에 유용하다. [[signal-and-noise-eval]]은 signal을 비슷하게 학습한 모델 집단의 점수 상대 분산, noise를 한 실행 최종 *n* 체크포인트의 상대 표준편차, SNR을 둘의 비로 정의한다(§3, Eq. 2). SNR은 R = 0.791(R² = 0.626)로 결정 정확도를 예측하지만 signal 또는 noise 단독은 그렇지 않다(§4.1, Fig. 2). MMLU-Pro는 항목 측면에서 같은 문제를 다룬다. 선택지 4개 대신 10개, 질문 12,032개, 전문가 검토 두 차례이며 GPT-4o와 GPT-4-Turbo 격차는 MMLU 1%, MMLU-Pro 9%다([[mmlu-pro]] §1).

**계산 예**(구성; guessing 모델은 측정이 아닌 가정). 모델이 항목 50%의 답을 알고 나머지에서 균일 추측한다고 하자. 선택지 4개에서는 기대 점수 0.5 + 0.5 × 0.25 = 62.5%, 10개에서는 0.5 + 0.5 × 0.10 = 55.0%다. 항목 50%와 55%를 아는 두 모델은 noise 전에도 4선택지에서 3.75점, 10선택지에서 4.5점 차이다. distractor가 더 많으면 점수는 낮아지고 탐지할 격차는 넓어진다.

**범용 모델에 대한 함의.** 벤치마크 형태는 어떤 능력을 측정할지 결정한다. 수학을 multiple-choice ranking으로 채점하면 유도 과정이 아니라 주어진 선택지 간 변별을 잰다. 축-형태 할당은 커버리지 맵의 일부이며, Tülu 3가 코드에는 온도 0.8 샘플 생성의 pass@10, 지시 따르기에는 프로그램 검사 가능 제약 메트릭, 지식에는 exact match를 쓰는 이유다([[tulu-3-eval-regime]] Table 24).

## §3 matcher는 메트릭의 일부다

**정의.** *matcher*는 모델 출력을 항목 하나의 점수로 매핑하는 함수다. exact match, 정규화 extractor, 출력을 실행하는 프로그램, 제약을 검사하는 프로그램, 또는 이를 판정하는 모델이다.

**문제.** matcher에는 거짓 음성(예상하지 못한 형식으로 쓴 정답)과 거짓 양성(검사를 만족하는 오답)이 있다. 둘 다 모델 변화 없이 보고 점수를 움직인다.

**메커니즘과 근거, 측정한 효과 크기.** IFEval은 같은 prompt 541개를 두 matcher로 보고한다. *strict* matcher는 생성된 응답에서 제약을 검사한다. *loose* matcher는 markdown bold marker 제거, 첫 줄 제거, 마지막 줄 제거, 모든 쌍, 세 가지 모두, identity를 포함한 변환 8개를 적용하고 변환 응답 하나라도 통과하면 제약을 따랐다고 센다. GPT-4에서 prompt-level strict 76.89%, prompt-level loose 79.30%, instruction-level strict 83.57%, instruction-level loose 85.37%다([[ifeval]] §2.2, Table 3).

**계산 예.** 541 × 0.7689 = strict matcher에서 통과한 prompt 416.0개, 541 × 0.7930 = loose에서 429.0개다. prompt 13개, 2.41점은 모델이 아니라 텍스트 변환 8개가 결정한다. IFEval 저자들은 loose 기준이 거짓 음성을 줄이지만 거짓 양성도 만든다고 하며 대체가 아니라 보완으로 취급한다([[ifeval]] §2.2). 따라서 인용한 네 수치 중 무엇인지 밝히지 않는 보고서는 한 모델에서 최대 8.5점(76.89에서 85.37) 모호하다.

다른 팀·다른 과제의 두 번째 측정에서 Tülu 3 MATH 평가는 Minerva 형식, 마지막 `<ans>` tag, 마지막 두 `$` delimiter 사이 텍스트를 차례로 시도하는 “flex” 추출을 쓴다. Minerva 전용에서 flex로 가면 “can sometimes improve reported scores by up to 10 points”다([[tulu-3-eval-regime]] §7.2). **복제됨:** 독립 보고서 둘의 현상이지만 효과 크기는 서로 다른 과제에서 왔으므로 비교할 수 없다.

**모델 기반 matcher.** 프로그램으로 답을 검사할 수 없으면 matcher는 모델이며 자체 검증이 필요하다. HELMET 요약 판정기(GPT-4o-2024-05-13)는 두 데이터셋에서 recall Cohen's κ 0.76·0.72, precision 0.91·0.83으로 인간 판정에 대해 검증됐다. 저자들은 부분적으로 뒷받침된 점에서 판정기가 사람보다 관대하다고 말한다([[helmet]] App. B.6). 같은 절은 surface metric이 놓치는 것을 보인다. 문장 하나를 수백 번 반복한 Mistral-7B-Instruct-v0.3 요약은 ROUGE-L 12.3, 판정기 점수 0.0이다. Tülu 3 HREF는 Llama 3.1 405B Instruct 기준선에 대해 Llama 3.1 70B Instruct를 판정기로 쓰며, 인간에 대한 composite agreement 69.4%, 인간 간 67%다([[tulu-3-eval-regime]] §7.3.2). 판정기 편향·캘리브레이션·판정기 특이 과적합은 ch-49의 주제다. 여기서 핵심은 판정 모델, 버전, prompt가 기록에 속하는 하네스 좌표라는 점이다.

**matcher 고르기.** 계열은 검증 가능한 것과 도입하는 오류가 다르다. 능력이 허용하는 가장 강한 것을 쓴다.

| matcher 계열 | 검증하는 것 | 도입하는 것 | 확인된 사용 |
|---|---|---|---|
| Exact 또는 substring match | 적힌 그대로 답 문자열이 나타남 | paraphrase·형식의 거짓 음성 | HELMET RAG 범주는 SubEM 채점([[helmet]] Table 3) |
| 정규화 extractor | 정의한 추출 시도 집합 뒤 답 | 과도하게 허용적인 추출의 거짓 양성 | Tülu 3 MATH flex 추출, Minerva 전용보다 최대 10점([[tulu-3-eval-regime]] §7.2) |
| 제약을 검사하는 프로그램 | 결정 가능한 출력 속성 | 표면 markup의 거짓 음성; 변환 뒤 거짓 양성 | IFEval strict/loose, GPT-4에서 2.41점 차이([[ifeval]] Table 3) |
| 테스트에 대한 실행 | 출력 행동 | 테스트 커버리지 의존; sandbox·timeout 효과 | Tülu 3 스위트 HumanEval·BigCodeBench pass@10([[tulu-3-eval-regime]] Table 24) |
| 환경 상태 predicate | 과제 환경 최종 상태 | 사소한 에이전트도 만족할 grader 결함 | do-nothing 에이전트가 τ-bench Airline 38% 통과([[agentic-benchmark-checklist]] App. E.2) |
| 판정기 모델 | 기준 또는 루브릭과의 일치 | 판정기 편향, 관대함, 버전 드리프트 | HELMET 판정기 인간 대비 κ 0.72–0.91([[helmet]] App. B.6); HREF 판정 일치 인간 대비 69.4%, 인간 간 67%([[tulu-3-eval-regime]] §7.3.2) |

**탐지 규칙.** 오답 채점 항목 50개를 샘플링해 출력을 읽는다. 답은 존재하나 다른 형식이면 matcher에 거짓 음성 문제가 있고, 답이 없는데 정답 처리됐으면 거짓 양성 문제가 있다. 둘 다 모델 결과가 아니라 matcher 버그다.

## §4 추론 설정과 수치 결정성

**정의.** *추론 설정*은 생성을 지배하는 모든 것이다. temperature, top-p, 최대 새 토큰, stop sequence, 샘플 수, chat template, system prompt, thinking 모드·budget, position-scaling 설정, serving stack·precision이다.

**문제.** 이들은 과제 정의가 고정하지 않으므로 같은 벤치마크를 구현하는 두 하네스가 다른 텍스트를 생성하고 다른 점수를 낼 수 있다.

**공개한 보고서의 확인된 설정.** Tülu 3는 HumanEval·HumanEval+를 temperature 0.8, pass@10으로 실행하고, AlpacaEval 2는 최대 8,192토큰 greedy decoding, IFEval은 greedy decoding·prompt-level loose 채점, MMLU는 0-shot CoT로 실행한다([[tulu-3-eval-regime]] Table 24, §7.2). HELMET은 다섯 입력 길이에서 greedy decoding을 쓰며 base 모델이 parseable output을 내도록 ICL·RULER 외 모든 과제에 in-context demonstration 둘을 더한다([[helmet]] §3, §2.3). MMLU-Pro는 5-shot CoT prompting([[mmlu-pro]] §4), OLMES는 completion을 포함한 입력을 2,048토큰으로 제한하고 모델 기본 precision에서 평가한다([[olmes]] §3.5).

형태별로 모은 이 설정은 하네스가 과제별로 노출해야 할 필드다. 아래 모든 값은 마지막 열의 소스에서 인용했고 빈칸은 해당 소스가 필드를 고정하지 않음을 뜻한다.

| 필드 | 로그 확률 채점 multiple choice | 검증 생성 | 판정 생성 | 긴 컨텍스트 |
|---|---|---|---|---|
| temperature | 해당 없음 | HumanEval pass@10 0.8 ([[tulu-3-eval-regime]] Table 24) | AlpacaEval 2 greedy ([[tulu-3-eval-regime]] Table 24) | greedy ([[helmet]] §3) |
| maximum new tokens | 해당 없음 | 미보고 | AlpacaEval 2 8,192 ([[tulu-3-eval-regime]] Table 24) | 미보고 |
| 항목당 샘플 | 1 | pass@10 10 ([[tulu-3-eval-regime]] Table 24) | 1 | 1 |
| few-shot 예 | curated 5개([[olmes]] §3.2) | GSM8K 8, MATH 4 ([[tulu-3-eval-regime]] Table 24) | 0 ([[tulu-3-eval-regime]] Table 24) | ICL·RULER 외 2 ([[helmet]] §2.3) |
| chain of thought | 아니오 | GSM8K·MATH·BBH에 예; MMLU-Pro 5-shot CoT ([[tulu-3-eval-regime]] Table 24; [[mmlu-pro]] §4) | 아니오 | 과제 의존 |
| 입력 cap·truncation | completion 포함 2,048토큰([[olmes]] §3.5) | 미보고 | 미보고 | 131,072토큰까지 5길이, 끝에서 truncation([[helmet]] §2.1, §3) |
| precision | 모델 기본([[olmes]] §3.5) | 미보고 | 미보고 | H100 BF16([[helmet]] App. D) |

**few-shot demonstration은 설정의 일부이며, 측정된 효과는 균일하지 않다.** 128K토큰 HELMET에서 Llama-3.1-8B base는 demonstration 둘을 더하면 JSON KV 77.3 → 98.0, MS MARCO 0.1 → 7.5로 움직인다. 반면 GPT-4o-05는 같은 변경에서 JSON KV 99.3 → 36.7로 떨어진다(시드 3개, [[helmet]] Table 8). 따라서 shot 수 하나는 모델 간 중립적이지 않다.

**test-time compute도 좌표다.** thinking 모드 모델에서 reasoning budget은 양방향으로 점수를 바꾼다. HAL은 벤치마크 전반 네 모델의 low/high reasoning effort를 짝지었고, model-agent-benchmark 조합 36개 중 21개에서 더 많은 reasoning의 정확도가 같거나 낮았다([[holistic-agent-leaderboard]] §4.1 item 5, Fig. 3). **결과(단일 연구).** budget 미보고 thinking 모델과 non-thinking 모델 비교는 교란되어 있다.

**수치 결정성.** temperature 0에서 sampler는 logits가 주어지면 결정적이지만 logits는 요청이 공유한 batch에 의존한다. [[thinkingmachines-defeating-nondeterminism]]은 이를 batch-invariant하지 않은 kernel 탓으로 돌린다. matmul, RMSNorm, attention은 batch 크기로 reduction 순서를 바꾸며 server load도 변한다. 측정: Qwen3-235B-A22B-Instruct-2507, prompt 하나, temperature 0, 각각 1,000토큰인 completion 1,000개는 unique completion 80개를 냈다. 최빈 completion은 78회 나왔다. 1,000개 전부 처음 102토큰은 일치했고, 발산 위치에서 992개는 “Queens, New York”, 8개는 “New York City”로 이어갔다. batch-invariant kernel에서는 1,000 completion 전부 동일하다. Qwen3-8B의 짧은 sequence 1,000개 throughput 비용은 26 s(vLLM default) → 55 s(최적화되지 않은 결정적) → 42 s(개선 attention kernel)다.

**계산 예**(이 과정의 산술, **해석**. 소스는 completion을 측정했지 벤치마크 점수는 측정하지 않음). 관찰 branch의 발산율, 즉 1,000개 중 8개가 다른 continuation을 택한 것을 쓴다. 생성 벤치마크 500항목에서 각 출력이 확률 0.008로 바뀌고 그 변화 절반이 정오를 뒤집으면 기대 이동은 500 × 0.008 × 0.5 = 항목 2개 = 0.4점이다. 이는 보통 행동으로 옮기는 1–3점 차이보다 작지만 0은 아니며 “temperature 0”만 기록한 보고서에서는 보이지 않는다.

[figures/eval-config-matrix.html](figures/eval-config-matrix.html) 그림은 이 절과 §2, §3, §5, §7의 측정 효과를 하나의 패널에 모은다. 정식화, adaptation 방법, matcher 변형, prompt 형식, benchmark 구성, 수치 결정성, perturbation, evidence 위치 좌표를 고르면 weights를 고정한 인용 연구에서 보고 수치가 얼마나 움직였는지 보인다. 두 번째 패널은 세 파이프라인 단계에서 skill별 Tülu 3 8B development·unseen 점수를 인쇄한다.

## §5 하네스가 보고하는 양으로서의 prompt-format 강건성

**정의.** *prompt-format 민감도*는 유능한 실무자가 동등하다고 볼 prompt template, 즉 separator, 대소문자, spacing, descriptor 표현, 열거 스타일 사이 점수 변동이다.

**문제.** template 하나는 의사결정을 내리는 많은 차이보다 spread가 큰 분포에서 샘플 하나를 만든다.

**메커니즘.** FormatSpread는 형식의 문법을 정의하고 각 형식을 multi-armed bandit arm으로 취급한다. Thompson sampling을 사용해 고정 예산 안에서 최고·최저 점수 형식을 찾으며 모델 weights 접근이 필요 없다([[prompt-format-sensitivity-formatspread]] §3.1-3.2). 성능 spread는 형식 `p_1…p_n`, 데이터셋 `D`, 메트릭 `m`에 대해 `max_i m(p_i, D) − min_i m(p_i, D)`다.

**근거.** Super-NaturalInstructions 과제 53개에서 LLaMA-2-13B는 최대 정확도 76점 spread를 보였다. 과제별 sampled format 10개만으로 중앙 spread 7.5점이며 저자들은 이를 하한이라고 부른다. 과제 20%는 모든 LLaMA-2 설정에서 최소 15점 spread를 가진다. spread는 큰 모델, 많은 shot, instruction tuning에서도 지속하고 ranking도 뒤집힌다. LLaMA-2-13B와 -70B는 확률 0.141로 적어도 d = 0.02만큼 뒤집힌다([[prompt-format-sensitivity-formatspread]] §4.2, Fig. 4). [[multi-prompt-evaluation]]은 6.5M 인스턴스, 모델 20개, 과제 39개에서 같은 효과를 잰다. Friedman test는 25과제 중 21개에서 template 간 통계적으로 유의한 차이를 내고 Kendall's W 값 대부분은 0.85 아래이며 LMentry template 하나의 끝 `"."`를 `":"`로 바꾸면 nous-hermes가 0.04에서 0.65로 움직인다(§4, Tables 4-5). **복제됨.**

**하네스가 보고해야 할 것.** [[multi-prompt-evaluation]]은 template 집합 `I`에 대해 세 aggregate를 제안한다. 고정 template 하나를 쓰는 downstream application에는 `MaxP = max_i ε(M,T,i)`, 강건성에는 `AvgP = (1/|I|) Σ_i ε(M,T,i)`, peak와 강건성 모두 중요할 때는 `Sat = 1 − (MaxP − AvgP)`인 `CPS = Sat · MaxP`다(§5). 여기서 `ε(M,T,i)`는 template `i` 아래 과제 `T`에서 모델 `M`의 점수다.

**계산 예.** 모델 A는 template 넷에서 0.74, 0.30, 0.10, 0.05이고 모델 B는 0.60, 0.58, 0.55, 0.52다. 모델 A는 MaxP = 0.74, AvgP = 1.19/4 = 0.2975, Sat = 1 − 0.4425 = 0.5575, CPS = 0.74 × 0.5575 = 0.413이다. 모델 B는 MaxP = 0.60, AvgP = 2.25/4 = 0.5625, Sat = 0.9625, CPS = 0.578이다. 모델 A는 MaxP로 1위, CPS로 2위다. 논문은 실제 데이터에서 같은 양상을 보고한다. LMentry rhyming-word 과제에서 Falcon-Instruct-7b와 Vicuna-13b는 AvgP 각각 0.17, 0.15로 MaxP 0.74에서 선두다([[multi-prompt-evaluation]] §6, Fig. 6).

**벤치마크 자체도 민감도를 낮추도록 만들 수 있다.** reasonable prompt 24개 아래 점수 범위는 MMLU에서 일반적으로 4–5%(최대 10.98%)이고, MMLU-Pro에서는 약 2%(최대 3.74%)다([[mmlu-pro]] §6.3, Fig. 5). option order는 같은 종류의 두 번째 축이며 perturbation audit과 함께 ch-47a에서 다룬다.

**비용 제어.** 완전 다중 template 평가는 template 수만큼 비용을 곱한다. 두 축소법이 확인되어 있다. FormatSpread bandit search는 53과제·320 format에 걸친 GPT-3.5 spread를 과제당 평균 $10 미만에 보고한다([[prompt-format-sensitivity-formatspread]] §1). [[multi-prompt-evaluation]]은 각 template를 무작위 100항목 부분집합에서 평가한다(§3). 실용적 절충은 canonical template에서 전체 항목 집합을 실행하고, 고정 부분집합에서 작은 template 집합을 실행하는 것이다.

## §6 재현성 기록

**정의.** *재현성 기록(reproducibility record)*은 모든 실행(run)에 함께 저장하는 필드들의 집합과 항목별 출력으로, 제3자가 그 수치를 재현할 수 있고 소유자가 다시 생성하지 않고도 그 결과를 재분할(re-slice)할 수 있게 하는 기록이다.

**문제.** 기록이 불완전하면 두 가지 실패가 뒤따른다. 하나는 수치를 재현할 수 없으므로, 나중의 차이를 모델이 아니라 하니스(harness)에 귀속할 수 없다는 것이다. 다른 하나는 나중에 묻게 되는 질문, 즉 점수가 과목별·길이별·근거 위치별·실패 모드별로 어떻게 나뉘는지가 재실행을 요구하며, 그때 전체 생성 예산을 다시 지불해야 한다는 것이다.

**메커니즘: 저장할 항목.**

```jsonl
{"task_id":"arc_challenge","harness":"olmes","harness_commit":"<sha>","format":"mcf",
 "norm":"pmi","shots":5,"shot_source":"curated-train","split":"test","n_items":1172,
 "model_revision":"<hf-revision-sha>","precision":"bf16","server":"vllm-<version>",
 "gen":{"temperature":0.0,"top_p":1.0,"max_tokens":16,"stop":["\n\n"],"n":1},
 "matcher":{"type":"argmax_choice_logprob"},"score":0.793,"decision_role":"report_only"}
{"task_id":"ifeval","harness":"open-instruct","harness_commit":"<sha>","shots":0,
 "matcher":{"type":"constraint_checker","variant":"prompt_level_loose"},
 "gen":{"temperature":0.0,"max_tokens":2048},"score":0.824,"decision_role":"dev_selection"}
```

항목별로는 렌더링된 프롬프트, 원시 출력, 추출된 답, 점수, 그리고 나중에 suite가 분할할 항목 메타데이터(과목, 길이 버킷, 제약 유형, 근거 위치, 환경 ID)를 저장한다. Ch-50은 정확히 이 기록들을 소비한다.

**스택 자체도 드리프트한다는 근거.** HAL은 $40,000 규모의 평가 캠페인에서 다음을 문서화했다. 한 제공자가 같은 이름 아래 DeepSeek R1 엔드포인트를 R1-0528로 교체했고, 한 라우터는 어떤 호출에는 FP4를, 다른 호출에는 FP8을 제공했으며, 조용히 발생한 rate-limit 실패가 오답으로 채점되었고, 한 API는 `stop` 인자를 제거했다([[holistic-agent-leaderboard]] App. A3). 같은 캠페인은 공식 τ-bench few-shot 파일에 테스트 세트 예제가 들어 있음을 약 $1,000의 평가를 수행한 뒤 발견했으며, 그 scaffold에서 나온 모든 결과를 제외했다([[holistic-agent-leaderboard]] App. A5). 기록에 모델 revision ID와 서빙 스택 버전을 넣어야 이러한 사건을 사후에 원인 귀속할 수 있다.

**suite 위생도 기록에 속한다.** Tülu 3는 프롬프트의 8-gram 매칭으로 suite에 대해 학습 데이터를 오염 제거한다. 테스트 인스턴스는 그 인스턴스와 8-gram이 일치하는 토큰이 자기 토큰의 50%를 넘으면 학습 인스턴스와 겹치는 것으로 간주한다. 한 학습 세트는 어떤 평가의 인스턴스 중 2%를 넘게 겹치면 오염된 것으로 간주한다. unseen suite에 대해 오염된 세트는 전부 제거한다([[tulu-3-eval-regime]] §3.2, App. B.2). 보고된 겹침은 무시할 수 없을 만큼 크다. Evol CodeAlpaca는 HumanEval의 70.7%를 포괄하고 LMSys Chat 1M은 AlpacaEval의 46.5%를 포괄한다(Table 37). 탐지 방법과 그 오류율은 ch-48의 주제다. 하니스가 맡을 부분은 적용되는 점수 옆에 오염 제거 보고서 ID를 저장하는 것이다.

**항목별 실패는 다음 단계의 입력이다.** 실패한 항목의 저장 기록은 실패 버킷화(ch-50), rejection-sampling 데이터 구성, ch-47a의 교란 감사(perturbation audit)를 위한 원재료다. 이것이 평가 하니스가 음성 예제를 생산하는 유일한 의미다. 하니스는 음의 그래디언트를 직접 적용하지 않으므로, 음성 피드백 장의 메커니즘은 여기에는 적용되지 않는다.

## §7 긴 컨텍스트 평가: 하니스 좌표

Ch-32c는 주장된 컨텍스트 길이와 유효 컨텍스트 길이, 그리고 긴 컨텍스트 suite 자체를 다룬다. 이 절은 그 수치가 의미를 가지려면 하니스가 무엇을 노출하고 기록해야 하는지를 다룬다.

1. **입력 길이와 절단.** 길이는 설정 하나가 아니라 축이다. HELMET은 8,192 / 16,384 / 32,768 / 65,536 / 131,072 Llama-2 토큰에서 평가하고 긴 문서는 끝에서 절단한다([[helmet]] §2.1, §3). 길이를 정의하는 데 사용한 토크나이저, 최대 입력, 절단 방향을 기록하라.
2. **위치 스케일링 구성.** 추론 시 위치 임베딩을 변경하는 일은 양면 효과를 갖는 평가 하이퍼파라미터다. HELMET은 RoPE base를 500,000에서 16,000,000으로 바꾼 Llama-3-Instruct와 YaRN을 적용한 Qwen2-Instruct가 모두 32,768 토큰을 넘어서 성능이 저하된다고 보고하며, 위치 임베딩 변경이 짧은 길이의 점수도 낮출 수 있다고 보고한다([[helmet]] App. E.3).
3. **데모와 답 형식.** base model은 긴 길이에서 파싱 가능한 출력을 만들기 위해 in-context 예시가 필요하다. 그 효과는 크고 모델 의존적이다([[helmet]] Table 8, 위의 §4에서 인용).
4. **측정지표 선택.** 표면 중첩 지표는 긴 출력에서 무너진다. GPT-4o의 ROUGE-L은 길이에 따라 절대 2점 이내에 머무르지만 judge 점수는 상승하며, 퇴화한 반복 문장 요약은 ROUGE-L 12.3과 judge 점수 0.0을 얻는다([[helmet]] §2.2, App. B.6).
5. **저장할 슬라이스.** 근거 위치(정확도는 입력의 시작 또는 끝에서 가장 높고 가운데에서 가장 낮다. 20개 문서를 사용한 GPT-3.5-Turbo는 답 문서가 첫 번째일 때 75.8%, 열 번째일 때 53.8%, 스무 번째일 때 63.2%를 기록하며, closed-book 점수는 56.1%다. [[lost-in-the-middle]] Table 1, App. G Table 6), 과제 범주, 길이 버킷이다.
6. **과제 계열.** 긴 컨텍스트 수치 하나는 긴 컨텍스트 능력의 요약이 아니다. 128K에서 instruction-tuned 모델 35개에 대해 HELMET의 범주 상관은 최저 0.34(Cite 대 ICL)까지 내려가며, 어떤 synthetic task도 downstream 범주와 평균 0.8 이상의 상관을 보이지 않는다([[helmet]] Fig. 3, Fig. 5). 질문과 needle 사이의 어휘 중첩은 통제해야 할 교란요인이다. 중첩을 제거하면 128K를 주장하는 13개 모델 중 11개가 32K에서 짧은 컨텍스트 점수의 절반 이하로 떨어지고, GPT-4o는 99.3에서 69.7로 내려간다([[nolima]] Abstract, §4.4). 검색이 검증된 경우에도 길이만으로 정확도는 저하된다. 입력이 커질수록 정확도는 13.9%-85% 하락하고, Llama-3.1-8B-Instruct는 30k 토큰에서 MMLU 문제 1,000개 중 970개에 대해 근거를 정확히 낭독하면서도 MMLU 정확도는 24.2점 하락한다([[context-length-alone-hurts]] Abstract, §1, App. Table 6). [[chroma-context-rot]]은 18개 모델에서 두 가지 하니스 요구사항을 추가한다. 길이를 바꾸는 동안 과제 난이도를 고정하고, 기권(abstention)은 오답과 별도로 채점하라.
7. **모든 긴 컨텍스트 실행을 같은 과제의 짧은 컨텍스트 회귀 실행과 짝지어라.** 위치 스케일링 변경은 짧은 입력 점수를 낮출 수 있기 때문이다([[helmet]] App. E.3). 이 쌍의 판단은 ch-51의 구간(interval) 도구로 결정하라.

## §8 에이전트 평가: 하니스 좌표

Ch-51a는 에이전트 일반성과 신뢰성을 다룬다. 에이전트 과제에 특유한 하니스 좌표는 다음과 같다.

1. **scaffold는 측정되는 시스템의 일부다.** HAL은 Online Mind2Web에서 GPT-5 Medium을 사용하는 SeeAct는 $171, Claude Sonnet 4를 사용하는 Browser-Use는 2점 정확도 차이에 $1,577이 든다고 보고한다. Claude 모델은 Browser-Use에서, OpenAI 모델은 SeeAct에서 더 높은 점수를 얻으며, 과제 특화 scaffold는 12개 실행 중 9개(CORE-Bench Hard) 및 12개 중 11개(SWE-bench Verified Mini)에서 generalist scaffold를 이긴다. 한편 generalist는 24개 비교 중 20개에서 비용이 더 낮다([[holistic-agent-leaderboard]] §4.1 items 6-7). 불균등한 scaffold로 모델을 비교하면 모델이 아니라 그 쌍을 측정한다.
2. **예산과 환경 이미지를 기록하라.** HAL의 generalist agent는 planning interval 4단계, 최대 200단계, 이름이 명시된 tool set을 갖는 smolagents CodeAgent로 규정된다([[holistic-agent-leaderboard]] §3, App. A8). 단계·시간·토큰 예산은 점수의 한계를 정하고 환경 image digest는 과제를 고정한다.
3. **정확도 옆에 비용을 보고하라.** HAL에서 가장 비용이 큰 모델은 9개 벤치마크 중 1개에서만 비용-정확도 Pareto frontier에 있으며, 평균적으로 모델의 3분의 1 미만만 frontier에 있다([[holistic-agent-leaderboard]] §4.1 items 1-2).
4. **점수를 신뢰하기 전에 grader를 검증하라.** 아무것도 하지 않는 에이전트와 답을 열거하는 에이전트를 grader에 실행하는 것은 저렴한 검사다. 아무것도 반환하지 않는 에이전트는 τ-bench Airline 과제의 38%를 통과하고, SWE-Lancer의 테스트 파일을 덮어쓰는 에이전트는 100%를 얻는다. 10개 벤치마크 전반에서 checklist는 과제 타당성 결함이 있는 것 7개, 결과 타당성 결함이 있는 것 7개, 보고 한계가 있는 것 10개를 발견했다([[agentic-benchmark-checklist]] §1, §5.2, App. E.2, E.4).
5. **과제만이 아니라 환경도 hold out하라.** 환경이 모두 학습에 등장한 에이전트 suite는 in-distribution 신뢰성을 측정한다. held-out environment split은 §1의 unseen suite에 대응하는 에이전트형이다.
6. **trajectory를 저장하라.** HAL은 transcript 2,184개를 분석해 에이전트가 8개 경우에서 HuggingFace 또는 arXiv로부터 벤치마크 답을 검색했고, 두 벤치마크에서는 하드코딩된 "그럴듯한" 해법을 사용했음을 발견했다. 이들 어느 것도 정확도 열에서는 보이지 않는다([[holistic-agent-leaderboard]] §4.2, App. A7-A8).

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| OLMES standard (15개 base model, 1B-70B) | — | eval-gate | few-shot 예시 | 과제당 training set에서 온 curated shot 5개, label-balanced | arXiv:2406.08446v2 §3.2 | verified 2026-09-15 | §3.2: 5-shot을 넘기면 의미 있는 차이가 없음(Brown et al. 2020; Barton 2024 인용) |
| OLMES standard | — | eval-gate | 인스턴스 샘플링 | 라벨이 공개된 경우 test split, 아니면 validation; 세트가 1,500개를 넘으면 1,000개 샘플, `Random(1234).sample` | §3.1, Table 2 | verified 2026-09-15 | §3.1: 추가 통계 신호는 점수 변동의 다른 원인들에 의해 지배됨 |
| OLMES standard | — | eval-gate | CF 정규화 | ARC-C, CSQA, OBQA에는 pmi; ARC-E, HellaSwag, PIQA, SIQA, MMLU에는 character; BoolQ, WinoGrande에는 없음 | Table 2, Table 3 | verified 2026-09-15 | Table 3: 15개 모델에 대한 승률; 모델별 oracle에서 0.0-1.1점 |
| OLMES standard | — | eval-gate | formulation | MCF와 CF를 실행하고 더 높은 것을 보고 | §3.4 | verified 2026-09-15 | Fig. 1(OLMo-7B-0424 학습 중 MMLU), Fig. 2(15개 모델) |
| OLMES standard | — | eval-gate | 입력 상한; precision; MMLU 집계 | completion을 포함해 2,048 토큰; 모델 기본 precision; 57개 과목의 macro average | §3.5 | verified 2026-09-15 | ablation 보고 없음 |
| Tülu 3 8B / 70B | 8B, 70B | eval-gate | development 설정 | MMLU 0-shot CoT EM; PopQA 15-shot EM; TruthfulQA 6-shot MC2; BBH 3-shot CoT EM; DROP 3-shot F1; GSM8K 8-shot CoT EM; MATH 4-shot CoT flex EM; HumanEval 및 HumanEval+ T=0.8에서 pass@10; IFEval prompt-level loose; AlpacaEval 2 length-controlled win rate, greedy ≤ 8,192 토큰 | arXiv:2411.15124v5 Table 24, §7.2 | verified 2026-09-15 | §7.2: flex extraction은 Minerva-only 대비 보고된 MATH 점수를 최대 10점 개선 |
| Tülu 3 8B / 70B | 8B, 70B | eval-gate | unseen suite | MMLU-Pro, GPQA, AGIEval English, DeepMind Mathematics, BigCodeBench(hard subset, 1,140개 중 148개), IFEval-OOD(제약 52개), HREF; 0-shot, 표시된 곳에서 CoT | Table 3, Table 24, §7.3 | verified 2026-09-15 | §7.4 Table 31: SFT, DPO, final checkpoint의 dev-to-unseen 비교 |
| Tülu 3 | — | eval-gate | 오염 제거 임계값 | 8-gram 프롬프트 매치; 인스턴스는 토큰의 > 50%가 하나의 학습 인스턴스와 매치되면 중첩; 학습 세트는 한 평가의 > 2% 인스턴스와 중첩하면 오염 | §3.2, App. B.2 | verified 2026-09-15 | Table 37: Evol CodeAlpaca가 HumanEval의 70.7%를 덮는 등의 중첩 |
| HELMET(긴 컨텍스트 모델 59개) | 1B-405B 및 closed | eval-gate | 긴 컨텍스트 프로토콜 | 길이 8,192 / 16,384 / 32,768 / 65,536 / 131,072 Llama-2 토큰; greedy decoding; ICL 및 RULER를 제외한 2-shot 데모; judge GPT-4o-2024-05-13; 데이터셋당 100-600개 샘플 | arXiv:2410.02694v3 §2.1-2.3, §3, App. D | verified 2026-09-15 | Table 8: base 및 instruction-tuned 모델의 0-shot 대 2-shot, seed 3개 |
| MMLU-Pro(50개 이상 모델) | — | eval-gate | prompting | 5-shot chain of thought | arXiv:2406.01574v6 §4 | verified 2026-09-15 | §6.2 Table 3: MMLU-Pro에서 direct answering보다 높은 CoT |
| HAL generalist agent | 모델 9개 | eval-gate | agentic scaffold | smolagents CodeAgent; planning interval 4단계; ≤ 200단계; search, browse, Python, bash, text inspector, file editor, VLM tool | arXiv:2510.11977 §3, App. A8 | verified 2026-09-15 | §4.1 item 7: 과제 특화 scaffold가 12회 중 9회와 11회 승리; generalist는 24회 중 20회 더 저렴 |
| Batch-invariant vLLM(Qwen3-8B) | 8B | eval-gate | 수치 결정성 | batch-invariant RMSNorm, matmul, attention kernel | Thinking Machines Lab, Sep 2025, "Performance" and "How nondeterministic are completions?" | verified 2026-09-15 | completion 1,000개 중 1,000개 동일; 짧은 sequence 1,000개에 26 s → 55 s → 42 s |

**작은 범용 실행을 위한 시작점.** 여기의 모든 수치는 그 출처가 사용한 조건과 함께, 위의 검증된 행에서 온다. SFT 도중 및 이후에 평가되는 1B-8B checkpoint의 경우 다음을 사용한다. OLMES standard에 따른 객관식 축(5개 curated shot, 과제별 정규화, 두 formulation 모두 실행하고 더 높은 값을 보고, 1,500개 초과 시 1,000개 인스턴스 샘플링, 2,048-토큰 상한, 기본 precision — 1B부터 70B까지 base model 15개에서 개발됨); Tülu 3 development 설정의 수학과 코드(GSM8K 8-shot CoT, flex extraction을 사용한 MATH 4-shot CoT, temperature 0.8에서 HumanEval pass@10 — 8B와 70B Llama 3.1 기반 post-training에 사용됨); 변형 이름을 명시한 IFEval prompt-level loose의 instruction following; 축별로 Tülu 3 unseen 목록에서 하나의 unseen benchmark를 선택하고 결정 뒤 한 번 읽기; 다섯 HELMET 길이에서 긴 컨텍스트 실행 및 짧은 컨텍스트 회귀 실행을 짝짓기. 이 보고서들이 고정하지 않는 두 좌표, 즉 서빙 스택과 precision, batch-invariant kernel 활성화 여부는 현지에서 고정하고 기록해야 한다.

## 일반화 관점

**(a) 측정의 폭을 넓히는 것.** 고정된 scenario 목록에 대한 다중 지표 포괄은 trade-off를 별도 보고서로 미루지 않고 보이게 한다. HELM은 (scenario, metric) 쌍 112개 중 98개를 측정하고, 예를 들어 TNLG v2(530B)가 robustness 교란 아래 NarrativeQA에서 72.6%에서 38.9%로 하락한다고 보고한다([[helm]] Finding 4). 축별 unseen benchmark는 "모델이 개선되었다"를 거짓일 수 있는 주장으로 바꾼다. Tülu 3의 unseen 평균은 두 크기 모두에서 development 평균과 함께 상승하며, 이는 벤치마크들만이 아니라 pipeline에 대한 근거다([[tulu-3-eval-regime]] Table 31). 다중 템플릿 평가는 형식 운을 보고되는 양으로 바꾼다([[prompt-format-sensitivity-formatspread]]; [[multi-prompt-evaluation]]).

**(b) 협소화를 일으키는 것.** development 동안 한 suite를 읽는 것은 모델을 그 suite 쪽으로 협소화한다. Tülu 3 8B의 코딩 열(HumanEval 86.2 → 83.9, BigCodeBench 11.5 → 7.4)과 MATH에서 DeepMind-Mathematics로의 LaTeX 형식 효과는 측정된 두 사례다([[tulu-3-eval-regime]] Table 31; [[tulu-3]] §7.4.1). 편의에 따라 benchmark를 선택하면 coverage map이 협소화되어 측정되지 않은 축이 남는다. 가장 좋은 수를 주는 formulation, normalization, template, matcher 변형을 고르는 식으로 하니스를 튜닝하면 보고서 자체가 협소화된다. 튜닝할 수 있는 크기는 하나의 ARC-Challenge 행에서 11.0점([[olmes]] Table 1), 네 IFEval 변형에서 최대 8.5점([[ifeval]] Table 3), 열 개 프롬프트 형식 전반에서 중앙값 7.5점이다([[prompt-format-sensitivity-formatspread]] §4.2).

**(c) 이 단계에서 이를 측정하는 법.** 축별 development-to-unseen gap과 학습 단계에 따른 그 변화를 보고하라. 가장 좋은 형식만이 아니라 format spread를 보고하라. 소규모 결정에 사용되는 benchmark의 signal-to-noise를 보고하고, 작은 규모에서는 높은 SNR 과제 또는 bits-per-byte scoring을 선호하라([[signal-and-noise-eval]] §4.1, §5.3). 에이전트의 경우 정확도와 함께 비용을 보고하고 do-nothing grader 검사를 실행하라([[holistic-agent-leaderboard]]; [[agentic-benchmark-checklist]]). Ch-51은 이를 go/no-go 결정으로 바꾸는 interval 도구를 제공한다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| 다른 하니스에서 나온 공개 수치와 수치를 비교함 | 차이가 몇 점이고 checkpoint 전반에서 안정적임 | 자신의 하니스에서 baseline model을 재실행하고, 같은 task ID와 설정으로 나온 수치만 비교한다([[olmes]] Table 1) |
| formulation 없이 객관식 점수를 보고함 | 초기 checkpoint 곡선이 우연 수준에 있다가 뛰어오름 | MCF와 CF를 모두 실행하고 둘 모두를 학습 토큰에 대해 그림으로 그린다([[olmes]] Fig. 1) |
| 변형 없이 IFEval식 수치 하나를 인용함 | 같은 모델에 대한 두 보고가 2.4에서 8.5점 차이남 | metric 이름을 명시한다: prompt-level 또는 instruction-level, strict 또는 loose([[ifeval]] Table 3) |
| matcher의 false negative | 답의 형식이 특이한 항목에 실패가 몰림 | 오답 출력 50개를 읽고, 존재하지만 추출되지 않은 답을 센다([[tulu-3-eval-regime]] §7.2) |
| 보고하는 benchmark에서 프롬프트 템플릿을 튜닝함 | 한 템플릿에서는 이득이 나타나나 paraphrase에서는 나타나지 않음 | subsample에서 템플릿 5-10개를 평가하고 AvgP와 spread를 보고한다([[multi-prompt-evaluation]] §5) |
| temperature 0이면 재현 가능하다고 가정함 | 같은 eval을 재실행하면 점수가 소수점 단위로 움직임 | 서버 부하를 달리하여 한 과제를 두 번 재실행하고 항목별 출력을 비교한다([[thinkingmachines-defeating-nondeterminism]]) |
| unseen benchmark를 영구적인 것으로 취급함 | 방법 변화 없이 수개월에 걸쳐 dev-to-unseen gap이 줄어듦 | unseen suite를 읽을 때마다 기록하고, benchmark가 결정에 정보를 준 뒤에는 폐기한다([[tulu-3]] §2.2) |
| 긴 컨텍스트 수치 하나 | 높은 needle 점수와 약한 downstream 긴 컨텍스트 행동 | 범주별·길이별 결과를 보고하고 낮은 어휘 중첩 검사를 추가한다([[helmet]] Fig. 3; [[nolima]] §4.4) |
| 서로 다른 scaffold에서 에이전트를 비교함 | scaffold가 바뀌면 순위가 바뀜 | scaffold를 고정하거나 각 모델을 여러 scaffold에서 평가하고 비용과 함께 쌍을 보고한다([[holistic-agent-leaderboard]] §4.1) |
| 에이전트 benchmark의 grader를 신뢰함 | 사소한 에이전트가 0보다 훨씬 높은 점수를 얻음 | do-nothing agent와 enumerating agent를 grader에 실행한다([[agentic-benchmark-checklist]] §5.2) |

## 이해도 점검

1. OLMES는 하나의 formulation을 고정하지 않고 MCF와 CF 점수 중 더 높은 것을 보고한다. MCF를 고정하면 왜 pre-training 모니터링 곡선이 오도될 수 있는지, 그리고 최댓값을 취한 뒤 보고된 양이 무엇을 뜻하는지 설명하라.
2. GPT-4의 네 IFEval 수치는 76.89에서 85.37에 걸친다. 541개 프롬프트 중 prompt-level strict와 loose 수치를 가르는 프롬프트가 몇 개인지 도출하고, loose matcher가 false negative를 줄이면서도 false positive를 만들 수 있는 이유를 설명하라.
3. Tülu 3 8B 코딩은 pipeline 전반에서 HumanEval 86.2 → 83.9, BigCodeBench 11.5 → 7.4로 움직이지만, 70B에서 BigCodeBench는 12.2 → 21.6으로 움직인다. 두 크기 사이의 차이가 8B 결과의 설명으로 무엇을 배제하며, 무엇을 배제하지는 않는가?
4. 한 팀이 공유 endpoint에서 하나의 prompt template과 temperature 0으로 평가한 데이터 변경으로 development benchmark에서 2점 이득을 보고한다. 이 장에서 나온, 각각 그 이득을 설명할 수 있는 변동의 측정된 원천 세 가지를 그 연구와 그 크기를 제한하는 수치와 함께 나열하라.
5. MMLU-Pro가 선택지를 4개에서 10개로 바꾸면 두 모델 사이의 측정된 격차가 왜 커지는가? 이 주장은 모델 행동에 관한 어떤 가정 아래 성립하는가?
6. 하니스가 과제별 aggregate score만 저장한다. 재실행 없이는 답할 수 없는 후속 장의 질문 두 가지를 이름 대고, 답하려면 항목별 기록에 무엇이 들어 있어야 하는지 말하라.
7. HAL은 한 scaffold 쌍이 2점 정확도 차이에 약 9배의 비용 차이가 있음을 발견했다. 이것이 scaffold를 구현 세부사항이 아니라 하니스 좌표로 만드는 이유와, 에이전트 점수를 비교 가능하게 하려면 무엇을 보고해야 하는지 설명하라.
8. 모델의 CPS는 개선되지만 MaxP는 하락하는 사례를 구성하고, 그 trade-off가 옳은 배포를 기술하라.

## 연결

- **ch-00 — 일반 능력의 의미와 측정 방법** (의존성): 이 장이 suite와 harness로 구현하는 능력 정의, development-versus-unseen 근거, 측정 오류 원천을 제공한다.
- **ch-36 — Lab: Masking Test, Forgetting Report, Held-Out Evaluation Split을 갖춘 SFT 실행** (의존성): 이 harness가 채점할 checkpoint와 held-out split을 만든다.
- **ch-46a — Lab: Generality Gate를 갖춘 소규모 Agentic SFT-then-RL 실행** (이전 장): 그 generality gate는 여기서 형식화한 suite 설계의 한 사례다.
- **ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation** (다음 장): suite 점수가 교란, 선택지 재정렬, fresh item에서도 유지되는지 시험하는 audit이다.
- **ch-32c — 주장된 대 유효 컨텍스트 길이와 긴 컨텍스트 평가**: §7이 구성하는 긴 컨텍스트 suite와 유효 길이 프로토콜을 맡는다.
- **ch-51a — 에이전트 일반성과 신뢰성 평가**: §8에서 구성하는 에이전트 신뢰성 지표와 held-out environment 설계를 맡는다.
- **ch-48 — 오염 탐지와 보고된 점수에 미치는 영향**: §6에 저장하는 오염 제거 보고서 ID 뒤의 탐지 방법을 다룬다.
- **ch-49 — Judge Model: 편향, 보정, Judge-Specific Overfitting**: §3에서 소개한 모델 기반 matcher의 검증을 다룬다.
- **ch-50 — Slice Analysis, Forgetting Slice, Failure Bucketing**: §6에서 정의한 항목별 기록을 소비한다.
- **ch-51 — Metric Noise, Confidence Interval, Go/No-Go Decision**: 이 harness가 생성한 수치를 명시된 불확실성이 있는 결정으로 바꾼다.
- **ch-52 — Safety Evaluation, Over-Refusal, Red-Teaming**: 단일 refusal 점수가 숨기는 over-refusal 측면을 포함한 coverage map의 safety 축이다.
- **ch-53 — Lab: Held-Out Suite, Forgetting Report, Perturbation Robustness를 갖춘 Evaluation Harness**: 여기서 설명한 harness를 구축한다.

## 출처

- [[olmes]] — formulation(MCF/CF), 네 CF 정규화와 과제별 할당, curated 5-shot prompt, 샘플링과 입력 상한, Table 1의 공개 ARC-Challenge 수치 spread.
- [[helm]] — scenario taxonomy, 다중 지표 포괄((쌍 112개 중 98개), 17.9% → 96.0% coverage 변화, HellaSwag의 adaptation-method 발견.
- [[tulu-3]] — 이 장의 평가 framework가 따르는 post-training 보고서; §7.4의 dev-versus-unseen 분석과 MATH formatting 결과.
- [[tulu-3-eval-regime]] — development 및 unseen suite 표, 과제별 설정, 오염 제거 임계값, Table 31의 dev/unseen 수치.
- [[mmlu-pro]] — 10개 선택지 구성, MMLU에 대한 판별력, 24개 prompt sensitivity 비교.
- [[ifeval]] — matcher 사례로 사용된 검증 가능한 instruction 구성과 strict/loose × prompt/instruction metric grid.
- [[prompt-format-sensitivity-formatspread]] — format-spread 정의와 탐색 절차; 중앙값 7.5점 spread와 순위 역전.
- [[multi-prompt-evaluation]] — 다중 템플릿 metric(MaxP, AvgP, Sat, CPS), 템플릿 전반의 Kendall's W, 단일 편집 효과.
- [[thinkingmachines-defeating-nondeterminism]] — temperature-0 nondeterminism의 원인으로서 batch invariance, completion 1,000개 측정, 결정적 kernel의 throughput 비용.
- [[signal-and-noise-eval]] — suite 크기 정하기에 사용한 signal, noise, SNR, decision accuracy, bits-per-byte scoring, benchmark-size 발견.
- [[helmet]] — 긴 컨텍스트 프로토콜(길이, 데모, judge), 범주 상관, 추론 시 position-extension 효과.
- [[nolima]] — needle test의 교란요인인 lexical overlap과 32K 성능 저하 수치.
- [[context-length-alone-hurts]] — 검증된 retrieval 아래에서 입력 길이에 따른 정확도 손실.
- [[chroma-context-rot]] — 고정 난이도에서 통제된 길이 변화; abstention 별도 채점.
- [[lost-in-the-middle]] — evidence-position slice와 U자형 위치 곡선.
- [[holistic-agent-leaderboard]] — 에이전트 평가의 scaffold, budget, cost 좌표; infrastructure drift와 τ-bench few-shot leak.
- [[agentic-benchmark-checklist]] — grader-validity 검사(do-nothing 및 enumerating agent)와 10개 agentic benchmark 감사.
