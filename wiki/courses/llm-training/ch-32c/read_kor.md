<!-- chapter: ch-32c
     track: midtraining
     kind: content
     title: 주장된 컨텍스트 길이와 실제 유효 컨텍스트 길이, 그리고 긴 컨텍스트 평가
     deps: [ch-32b]
     sources: [[ruler]], [[helmet]], [[nolima]], [[longbench-v2]], [[michelangelo]], [[openai-mrcr-graphwalks]], [[lost-in-the-middle]], [[retrieval-head]], [[context-length-alone-hurts]], [[chroma-context-rot]], [[fiction-livebench]], [[babilong]], [[gradient-llama3-1048k]], [[interconnects-llama-4-long-context]], [[string-effective-context]], [[prolong]], [[controlled-long-context-extension]], [[needle-in-haystack-data]], [[llama-4]]
     figures: figures/effective-length-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32c — 주장된 컨텍스트 길이와 실제 유효 컨텍스트 길이, 그리고 긴 컨텍스트 평가

> **핵심 통찰.** 모델이 받아들이는 컨텍스트 창의 길이는, 일정 정확도를 유지하면서 실제로 사용하는 길이가 아니다. RULER의 13개 합성 과제에서, 주장 길이가 128K인 Llama 3.1 70B는 85.6 임계값을 넘는 길이가 64K까지이고, 주장 길이가 1M인 GradientAI의 Llama-3 70B는 16K까지다([[ruler]] Table 3). 측정한 유효 길이는 과제에 따라 달라진다. 질문과 needle의 단어 중복을 제거하면, 같은 Llama 3.1 70B의 유효 길이는 2K가 된다([[nolima]] Table 3). Needle retrieval은 실제 애플리케이션 성능을 RAG 과제보다 덜 잘 예측한다. 128K에서 instruction-tuned 모델 35개를 비교할 때, NIAH와 ∞Bench QA의 Spearman 상관계수는 ρ = 0.63이지만, RAG 과제 HotpotQA의 상관계수는 0.88이다([[helmet]] Fig. 4). 검색을 통제해도 입력 길이가 길어지면 정확도는 떨어진다. Llama-3.1-8B-Instruct는 30K filler token이 있을 때도 0개일 때와 똑같이 MMLU 근거를 97.0% exact match로 다시 말하지만, 답변 정확도는 여전히 24.2점 하락한다([[context-length-alone-hurts]] App. Table 6).
>
> **지침.** 컨텍스트 확장 단계(ch-32b)를 gate할 때는, 시험한 모든 길이에서 RULER 유효 길이를 보고하고, 비어휘적(non-lexical) 또는 종합(synthesis) 과제 하나(NoLiMa, Michelangelo, MRCR)와 애플리케이션 suite 하나(HELMET category 또는 LongBench v2)를 함께 보고하라. 세 과제군은 같은 모델들의 순위를 서로 다르게 매기기 때문이다([[helmet]] Table 5; [[nolima]] Table 3; [[michelangelo]] Fig. 9). checkpoint나 모델을 비교할 때는 rope_scaling, truncation 방향, answer prefix 또는 prefill, demonstration, chat template, 수치 정밀도, serving stack을 평가 hyperparameter로 기록하라. 아래 출처에서 설정 하나만 바꿔도 점수가 2.7점에서 71점까지 움직였기 때문이다(§6). checkpoint를 승격할 때는 confidence interval을 포함한 짝지은 short-context 비교를 요구하라. 예를 들어 1,000개 항목 중 110개가 답을 바꾼 경우 paired difference의 95% 구간은 ±2.1점이고, unpaired 비교에서는 ±4.0점이다(§6.3 worked example). NIAH heatmap만으로 뒷받침된 1M~10M 창 주장은 유효 길이 측정이 나오기 전까지 일화적 증거로 취급하라(§7).

## 범용 모델에서 이 장이 중요한 이유

범용 모델의 pipeline은 pre-training → mid-training → SFT → preference optimization → RL → evaluation이다. 컨텍스트 길이 확장은 mid-training 단계다(ch-32b). 이 장은 그 단계 끝의 gate를 다룬다. 확장된 모델이 실제로 긴 창을 사용하는지, 그리고 확장 과정에서 short-context 능력이 사라지지 않았는지를 어떻게 판정할지 설명한다.

측정 가능한 문제는 세 부분으로 나뉜다. 첫째, release note는 **주장된 컨텍스트 길이(claimed context length)**, 즉 모델이 받도록 구성한 최대 position 범위를 말한다. 모델은 1M token을 입력으로 받을 수 있어도 32K에서 과제에 실패할 수 있다([[ruler]] Table 3). 둘째, 가장 흔한 시험인 needle-in-a-haystack(NIAH: 삽입된 사실 하나를 질문으로 찾아내기)은 포화돼 있다. 128K에서 [[helmet]] Table 5의 frontier 모델 6개 중 5개는 100.0점이다. 셋째, 긴 컨텍스트 점수는 거의 보고되지 않는 평가 설정에 민감하다(§6). 따라서 같은 checkpoint라도 두 그룹이 서로 다른 수치를 측정할 수 있다.

범용 모델에서 긴 컨텍스트는 단일 능력이 아니다. 이 능력에 의존하는 애플리케이션에는 retrieval-augmented generation(RAG), 여러 문서의 종합, repository 수준 코드 작업, many-shot in-context learning(ICL), 긴 대화, agent trajectory(ch-45c)가 있다. [[helmet]] Fig. 5에서 128K category correlation은 ICL과 Cite 사이에서 0.34까지 내려간다. 이 category 중 하나만 측정하는 gate는 나머지를 측정하지 못한다. ch-32b는 확장 방법과 데이터를, ch-28은 학습 데이터 구성에 쓰이는 합성 과제군을, ch-47은 일반적인 evaluation harness를 다룬다.

## §1 주장 길이, 학습 길이, 유효 컨텍스트 길이

**정의.**
- **주장 길이(claimed length)**: release에서 모델이 지원한다고 밝힌 입력 길이.
- **학습 길이(training length)**: 학습 중 사용한 가장 긴 sequence 길이. inference-time scaling을 쓴 경우 주장 길이보다 짧을 수 있다. RULER 표에서 Qwen2 72B는 32K로 학습했지만 128K를 주장했다([[ruler]] §4).
- **유효 길이(effective length)**: 어떤 과제 점수가 임계값보다 높은 상태로 유지되는 시험 길이 중 가장 큰 길이. 이 값은 과제 suite와 임계값을 기준으로 할 때만 정의된다.

**문제.** 임계값 정의가 없으면 “128K를 지원한다”는 문구는 모델 간 비교도 검증도 할 수 없다.

**메커니즘(RULER protocol, [[ruler]] §4, App. D).**
1. 길이 {4K, 8K, 16K, 32K, 64K, 128K} 각각에서 13개 과제 각각에 대해 500개 예제를 생성한다.
2. 각 입력을 모델의 chat template로 감싸고, “모델이 질의에 답변을 거부하거나 설명을 생성하지 않도록” task answer prefix를 덧붙인다.
3. greedy decoding(vLLM, BFloat16)하고 recall 기반 정확도로 채점한다.
4. 각 길이에서 13개 과제 점수의 평균을 구한다.
5. Llama2-7B-chat의 4K 점수인 85.6을 넘는 평균을 유지하는 가장 큰 길이를 보고한다.

**수식.**

```
L_eff(threshold τ) = max { L ∈ tested lengths : S(L) > τ }
RULER:  τ = 85.6                         (고정값, 모든 모델)
NoLiMa: τ = 0.85 × B,  B = mean over question-needle pairs of max(S_250, S_500, S_1K)
wAvg(inc) = Σ_k k · S(L_k) / Σ_k k ,   wAvg(dec) = Σ_k (7−k) · S(L_k) / Σ_k k ,   k = 1..6
```

여기서 S(L)은 길이 L에서의 suite 평균 점수이고, τ는 임계값이며, B는 NoLiMa의 모델별 base score이고, L_k는 4K부터 128K까지 k번째로 시험한 길이다. 가중치는 sequence 길이에 따라 “선형으로 증가 또는 감소”한다([[ruler]] §4). 가중치 1~6을 쓰면 아래의 인쇄된 값이 재현된다(derived).

**계산 예.** Llama3.1 (70B)의 4K~128K 점수는 96.5, 95.8, 95.4, 94.8, 88.4, 66.6이다([[ruler]] Table 3).
- 유효 길이: 64K에서 88.4 > 85.6이고 128K에서 66.6 < 85.6이다. 따라서 L_eff = 64K다.
- wAvg(inc) = (96.5·1 + 95.8·2 + 95.4·3 + 94.8·4 + 88.4·5 + 66.6·6) / 21 = 1795.1 / 21 = 85.5.
- wAvg(dec) = (96.5·6 + 95.8·5 + 95.4·4 + 94.8·3 + 88.4·2 + 66.6·1) / 21 = 1967.4 / 21 = 93.7. 두 값 모두 표와 일치한다.
- Llama 3.1 70B의 NoLiMa: B = 94.5, τ = 80.3이고, 점수는 1K에서 91.0, 2K에서 81.8, 4K에서 71.2다. 따라서 L_eff = 2K다([[nolima]] Table 3).

그러므로 같은 checkpoint라도 과제 suite에 따라 유효 길이는 64K 또는 2K가 된다. 어느 수치도 틀린 것이 아니다. 각각이 다른 질문에 답한다.

**증거(RULER의 open model leaderboard 행).**

| 모델 | 주장 길이 | 유효 길이 | 32K | 128K | 근거 위치 |
|---|---:|---:|---:|---:|---|
| Llama3.1 8B (Instruct) | 128K | 32K | 87.4 | 77.0 | [[ruler]] Table 3 |
| Llama3.1 70B (Instruct) | 128K | 64K | 94.8 | 66.6 | Table 3 |
| Qwen2 72B (Instruct) | 128K | 32K | 94.1 | 53.7 | Table 3 |
| GradientAI/Llama3 70B | 1M | 16K | 85.4 | 72.1 | Table 3 |
| LWM 7B | 1M | <4K | 69.1 | 65.0 | Table 3 |
| ProLong 8B | 512K | 32K | 89.3 | 81.6 | RULER README @ab17b78 |
| GradientAI/Llama3 8B | 1M | 16K | 79.9 | 69.5 | RULER README @ab17b78 |

논문은 17개 모델 전부가 “32K token 이상의 context size를 주장”하지만, “그중 절반만 32K 길이에서 만족스러운 성능을 유지할 수 있다”고 요약한다(Abstract). 거의 모든 모델이 표준 NIAH에서 거의 완벽한 점수를 낸다(Abstract; §4). 나중에 모델 저자가 추가한 일부 leaderboard 행은 85.6 기준으로 자신이 보고한 점수와 맞지 않는 유효 길이를 싣고 있다. 예를 들어 Qwen3-8B는 64K에서 82.1인데도 effective length를 64K로 기재했다([[ruler]] excerpt). 따라서 확인해야 할 값은 길이별 점수이며, 저자가 보고한 행의 effective-length 열이 항상 그 점수와 일관되지는 않다.

**조건과 한계.**
- 85.6 임계값은 오래된 단일 모델에서 가져온 고정 기준이다. LWM처럼 4K에서도 약한 모델(82.3)은 성능 저하가 완만해도 “<4K”가 된다. 저자들은 이를 “짧은 sequence에서의 절대 성능과 상대적 저하 사이의 평가 trade-off”라고 부른다([[ruler]] §4). NoLiMa의 상대 임계값은 4K 점수 의존성을 없애지만, 유효 길이가 base score에 의존하게 만든다([[nolima]] §4.3).
- [[babilong]]은 과제별 규칙을 쓴다. 85% 초과는 만족스러움, 30% 미만은 실패다(§3.1). 이 규칙에서 시험한 LLM은 “컨텍스트의 10~20%만” 사용하고, GPT-4는 128K 창의 약 10%를 사용한다(Abstract; Fig. 1b).
- 유효 길이는 시험한 길이에서만 측정한다. 64K를 통과하고 128K에서 실패한 모델도 70K에서 실패할 수 있다.

**함의.** 컨텍스트 확장 gate는 반드시 suite, 임계값 규칙, 그리고 길이별 전체 성능 곡선을 명시해야 한다.

[effective-length-explorer.html](figures/effective-length-explorer.html) 그림을 이용하면 독자가 RULER과 NoLiMa 행에 고정 또는 상대 임계값을 적용하여 유효 길이와 weighted average가 어떻게 바뀌는지 볼 수 있다.

## §2 유효 길이가 학습 길이에 못 미치는 이유

측정되어 보고된 요인은 세 가지다. 각 상대 위치가 학습 중 얼마나 자주 나타났는지, 증거가 문서의 어디에 놓였는지, 그리고 어떤 attention head가 retrieval을 수행하는지다.

### §2.1 상대 위치의 빈도(STRING)

**정의.** corpus C와 학습 길이 L에 대해 상대 위치 i의 빈도는 학습 중 거리 i에서 관찰한 query-key pair의 수다.

```
f(i) = Σ_{s ∈ C} max(|s| − i, 0),   0 ≤ i < L
```

|s|는 sequence s의 token 길이다([[string-effective-context]] Eq. 2).

**계산 예.** 길이 4인 sequence 하나는 f(0) = 4, f(1) = 3, f(2) = 2, f(3) = 1을 만든다. 여기에 길이 2 sequence 하나를 더하면 f(0) = 6, f(1) = 4, f(2) = 2, f(3) = 1이다. 짧은 문서는 작은 거리의 count만 더하므로 큰 거리는 언제나 가장 드물게 관찰된다.

**증거.** SlimPajama에서 L = 2048일 때 거리 i ≤ 1024가 position index의 80%를 넘고, i ≥ 1536은 5% 미만을 차지한다(§2.2). 1T token으로 pretraining한 두 1.3B 모델은 training length가 각각 2K와 4K인데도 f(1280)가 같은 count인 100B에 이르면 같은 effective length(1,280)에 도달한다(§3, Fig. 2b). 13개 open model에서는 needle이 문서의 첫 1/3에 있을 때, 즉 query에서 가장 먼 곳에 있을 때 needle-retrieval failure가 최대가 된다(Table 4). 저자들은 open model의 effective length가 종종 “training length의 절반을 넘지 않는다”고 쓴다(Abstract).

**수정의 메커니즘(STRING, inference only).** 학습 길이 L, offset S, local window W를 두면 거리 d ≥ S를 d − S + W로 바꾸고, S 미만의 거리는 그대로 둔다. query position id만 바꾸므로 KV cache는 영향을 받지 않는다(§4.1, Eq. 3-4, App. A.1).

**계산 예(Eq. 3-4에서 도출).** L = 10, S = 4, W = 2라고 하자. 마지막 query row가 원래 사용하는 거리는 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]이다. d ≥ 4인 거리는 d − 2가 되어 [7, 6, 5, 4, 3, 2]가 되고, 나머지는 [3, 2, 1, 0]으로 남는다. 새 row는 [7, 6, 5, 4, 3, 2, 3, 2, 1, 0]이다. 학습에서 가장 드물었던 거리 8과 9를 더 이상 사용하지 않는다.

**결과(single study).** S = L/3, W = 128일 때 Llama 3.1 70B의 128K RULER은 66.6에서 81.7로, Qwen2 72B는 53.7에서 84.6으로 오른다. 7개 base model에서의 4-needle NIAH 평균은 67.8에서 85.7로 오른다(Tables 1-2). **한계:** STRING에는 short-context 또는 general benchmark가 보고되지 않았고, 이 방법은 모델을 training length보다 길게 확장하지 않는다([[string-effective-context]] §4.2; card “Generality”). **함의:** 주장 길이와 유효 길이의 차이 일부는 position-extrapolation failure만이 아니라 data length distribution이 만든 under-training 효과다(ch-32b, ch-04). 이는 저자들의 해석이다.

### §2.2 증거의 위치(lost in the middle)

**정의.** Position sensitivity는 관련 정보의 위치만 바꾸었을 때 accuracy가 바뀌는 정도다.

**증거.** 약 4K token인 retrieved document 20개에서 GPT-3.5-Turbo는 답이 들어 있는 document가 첫 번째일 때 75.8%, 10번째일 때 53.8%, 20번째일 때 63.2%를 답한다. document가 전혀 없는 closed-book accuracy는 56.1%다([[lost-in-the-middle]] App. G Table 6; Table 1). 최고와 최저의 차이는 75.8 − 53.8 = 22.0 point이고, 최저 경우는 closed-book accuracy보다도 낮다. Instruction tuning은 MPT-30B의 gap을 약 10 point에서 약 4 point로 줄였으나 제거하지는 못했다(§4.3). Extended-window variant(GPT-3.5-Turbo 4K와 16K)는 둘 다 들어갈 수 있는 input에서는 거의 동일하다(§2.3). 128K에서 [[helmet]] App. E.4는 model이 일반적으로 가장 최근 context를 선호한다고 보고한다. [[babilong]] App. K에서는 GPT-4-Turbo가 fact가 depth 50%에 있을 때 QA1 성능이 가장 낮다.

**조건.** [[chroma-context-rot]]은 자체 NIAH variant에서 11개 needle position에 따른 뚜렷한 변동이 없다고 보고한다(“Needle-Question Similarity › Results”). 따라서 position effect의 크기는 과제에 따라 달라진다. **함의:** position 평균은 U자형 곡선을 숨기므로 depth별 accuracy와 최고-최저 gap을 보고해야 한다([[lost-in-the-middle]] §1).

### §2.3 Retrieval head

**정의.** Retrieval head는 모델이 needle token을 생성할 때 가장 크게 attention한 input token이 needle 내부의 동일 token인 attention head다([[retrieval-head]] §2).

```
retrieval_score(h) = |g_h ∩ k| / |k|
```

g_h는 head h가 복사한 needle token의 집합이고, k는 needle의 token 집합이다(Eq. 1). score가 0.1보다 큰 head를 retrieval head로 센다(§2).

**계산 예.** 10-token needle에서 head h가 생성된 needle token 9개에 대해 top-attending copier라면 score = 9/10 = 0.9다(toy number; score 정의는 §2).

**증거(Result, single study).** Llama-2, Yi, Qwen1.5, Mistral family 전반에서 0.5를 넘는 head는 전체의 5% 미만이다(Fig. 3 caption). Base model과 그 long-context, chat 또는 MoE derivative의 retrieval-score map은 Pearson correlation 0.8을 넘는다(§3.3). Llama-2-7B-80K의 top 20 retrieval head를 mask하면 NIAH가 63.6으로 떨어지지만, random head 20개를 mask하면 94.7이다(Fig. 1). Retrieval head masking은 Mistral-7B-Instruct-v0.2에서 extractive-QA F1을 9.2%와 23.1% 낮추고 chain-of-thought accuracy도 낮추는 반면, answer-only MMLU는 거의 변하지 않는다(§4.2-4.3).

**함의(이 course의 해석, [[retrieval-head]] §5 기반).** 저자들은 retrieval head가 전체 KV cache를 필요로 한다고 주장한다. KV-cache compression, sliding-window attention, hybrid architecture는 이 head가 attention할 수 있는 대상을 바꾸므로, 이런 변경을 위한 long-context gate에는 perplexity만이 아니라 retrieval 및 context 위의 CoT 과제를 포함해야 한다.

## §3 Retrieval은 필요하지만 충분하지 않다: 길이만 늘어나도 accuracy가 내려간다

**정의.** Length-only effect는 문제, 증거, retrieval 성공을 고정하고 input token 수만 바꾸었을 때 accuracy가 변하는 현상이다.

**메커니즘([[context-length-alone-hurts]] §3.1).**
1. 짧은 과제 하나(GSM8K, MMLU, HumanEval 또는 VarSum: 50개 integer variable 중 3개의 합)를 고른다.
2. input을 [evidence][filler][question]으로 만들고, filler token 수를 0, 7,500, 15,000, 30,000으로 바꾼다.
3. 한 번의 run에서는 모델에게 evidence와 question을 단어 단위로 그대로 recite하라고 요청한다. exact match가 retrieval score다.
4. 다른 run에서는 답을 내라고 요청한다.

**증거.**
- MMLU에서 Llama-3.1-8B-Instruct의 accuracy는 0 token일 때 63.2이고 30,000 token에서는 24.2 point 하락한다. 반면 retrieval은 97.0이고 delta는 0.0이다(App. Table 6). VarSum에서는 0 token일 때 96.0이지만 7,500 token에서 59.0 point, 30,000 token에서 85.0 point 하락하며, 7,500 token에서 retrieval도 8.0 point 하락한다.
- filler를 attention에서 mask해도 30,000 token에서의 하락은 남는다. Llama GSM8K는 19.6 point, Mistral HumanEval은 7.9 point 하락한다(Table 3).
- whitespace filler를 evidence 앞에 두어 evidence가 question 바로 옆에 오게 해도, 30K에서 Llama VarSum의 하락은 20.0 point에 이른다(App. Table 8).
- Recite-then-solve, 즉 evidence를 recite한 뒤 long context 없이 짧은 prompt에서 답하게 하면, 26,250 essay token에서 Mistral GSM8K는 35.5에서 66.7로 오른다(Table 4). GPT-4o의 RULER QA2도 32K에서 68.4에서 72.4로 오른다(Table 5).
- **한계:** open model 2개와 closed model 3개, 과제 4개만 사용했고, 모든 실험은 한 번만 실행했으며 confidence interval이 없다(Limitations; App. A.4).

**실무자 증거.** [[chroma-context-rot]]은 과제를 고정한 채 18개 model을 시험했다. Embedding similarity가 더 낮은 needle-question pair는 길이가 늘어날수록 더 빨리 악화된다. 주제와 관련 있는 distractor는 길이가 길어질수록 accuracy를 더 크게 낮춘다. 동일한 306개 question으로 구성했지만 focused LongMemEval prompt는 약 300 token, full prompt는 약 113k token인데, 모든 model은 full prompt보다 focused prompt에서 더 높은 점수를 받는다. Claude model은 주로 abstention으로 점수를 잃고 GPT model은 wrong answer로 점수를 잃는다(LongMemEval; Impact of Distractors). 보고서 본문에는 model별 수치가 없다.

**함의.** 모델은 어떤 길이에서 retrieval을 통과해도 그 길이에서 reasoning을 더 못할 수 있다. 범용 모델의 effective length는 retrieved fact를 실제로 사용하는 과제로 측정해야 한다([[context-length-alone-hurts]]와 [[chroma-context-rot]] 모두 fixed-task design에서 이를 재현했다).

## §4 Needle retrieval을 넘어선 평가

### §4.1 HELMET: 적용 범주와 NIAH가 예측하는 것

**정의.** HELMET은 8K부터 128K까지 통제된 길이에서 일곱 범주를 평가한다. RAG, 인용을 포함한 생성(Cite), passage re-ranking, many-shot ICL, 긴 문서 QA, 요약, synthetic recall이다([[helmet]] §2.1).

**문제.** 실행 비용이 더 낮은 시험은 application task와 같은 방식으로 model의 순위를 매길 때에만 proxy로 유용하다. 순위 일치는 Spearman의 ρ로 잰다.

**먼저 계산 예.** model 다섯 개를 test A가 1, 2, 3, 4, 5 순위로, test B가 2, 1, 4, 3, 5 순위로 매긴다고 하자. 순위 차이 d는 −1, 1, −1, 1, 0이므로 Σd² = 4다.

```
ρ = 1 − 6 Σ d_i² / (n (n² − 1)) = 1 − 6·4 / (5·24) = 0.8
```

d_i는 model i의 순위 차이, n은 model 수다. 이 식은 tie가 없을 때 성립한다. NIAH처럼 많은 model이 100.0에 tie하면 순위는 평균을 사용하며, 그 test는 model들을 잘 구분하지 못한다.

**증거(Result, single study; 128K에서 instruction-tuned model 35개).** ∞Bench QA와의 Spearman correlation은 NIAH가 0.63, RULER MK needle이 0.81, RAG HotpotQA가 0.88이다(Fig. 4). NIAH와 모든 downstream category의 correlation은 ≤ 0.8이다(§3.1). 순위는 역전된다. RULER는 Gemini-1.5-Flash(86.6)를 Gemini-1.5-Pro(65.3)보다 위에 놓지만, HELMET은 각각 50.7과 62.7을 준다(Table 5). 128K에서 §3.3의 prose는 closed model이 Cite와 Re-rank에서 open model보다 30–40 point 앞선다고 말한다. v3 Fig. 6 값은 best listed open model에 비해 Re-rank 31.8 point, Cite 17.2 point 차이를 준다(card에서 도출). 많은 open model은 ICL에서 앞선다. Jamba-1.5-Mini는 91.0, GPT-4o는 86.3이다(Fig. 6).

**Metric 선택.** 문장 하나를 수백 번 반복한 Mistral-7B summary는 ROUGE-L 12.3, GPT-4o-judge 0.0을 받는다(App. B.6). HELMET은 QA와 summarization에 judge-based scoring을 쓰며, human agreement κ는 0.72부터 0.91 사이다(App. B.6).

**함의.** 개발 중 빠른 확인이 필요하면 HELMET 저자들은 synthetic recall보다 다른 downstream category와 더 높은 상관을 보이는 RAG task를 권한다. model 결정을 내릴 때는 일곱 범주 전체를 권한다([[helmet]] Abstract, §3.1).

### §4.2 NoLiMa: literal overlap 제거

**정의.** Literal overlap은 relevant context에서 찾을 수 있는 question token의 ROUGE-1 precision이다. vanilla NIAH에서는 0.905, RULER S-NIAH에서는 0.571, HELMET RAG에서는 0.689, NoLiMa에서는 0.069다([[nolima]] Table 1).

**메커니즘.** Needle(`Actually, [CHAR] lives next to the Semper Opera House.`)과 question(`Which character has been to Dresden?`)은 keyword를 공유하지 않는다. model은 landmark와 city 사이의 association을 사용해야 한다. Haystack은 distractor word와 의도하지 않은 answer를 걸러낸다. question-needle pair 58개를 haystack 5개에 각각 26번 넣어, 길이 하나당 test 7,540개를 만든다(§3–§4.1).

**증거.** model 13개 중 11개는 32K에서 base score의 절반 이하를 기록한다. GPT-4o는 99.3에서 69.7로 떨어진다(Table 3). Table 3의 어느 model도 8K를 넘는 effective length를 갖지 않으며, 1M–2M을 claim한 model도 포함된다. Llama 3.3 70B의 32K score는 question이 landmark를 직접 명명할 때 98.5, one-hop association일 때 56.2, two-hop일 때 25.9다(Table 6). CoT는 two-hop을 34.3까지 올린다(Table 4). **한계:** effective length는 0.85 × base rule에 의존하며, training intervention은 연구하지 않았다.

### §4.3 LongBench v2: human baseline이 있는 긴 multiple-choice reasoning

[[longbench-v2]]는 여섯 범주(single-document QA, multi-document QA, long ICL, dialogue history, code repository, structured data)에 걸친 8k–2M words context의 four-option question 503개를 담는다. item은 128k-context LLM 세 개가 모두 정답을 맞히지 못하고, human reviewer도 3분 안에 답하지 못할 때만 유지됐다(§1). Human expert는 15분 제한에서 53.7%를 기록한다. GPT-4o-2024-08-06은 직접 답할 때 50.1%, o1-preview는 57.7%다(Table 2). Context 없이 GPT-4o는 33.1%를 기록한다(Table 3). 이는 score 중 parametric knowledge가 설명할 수 있는 양의 상한을 정한다. Qwen2.5-72B-Instruct는 full 128k window보다 32k retrieved context에서 더 높은 score를 낸다(+4.1%, §4.2). **한계:** 저자들은 503개 item이 덜 안정적인 결과를 줄 수 있다고 쓴다(§6). interval width는 이 장 §6.3에서 계산한다. 길이 bucket별 score는 bucket마다 task mix가 다르므로 서로 비교할 수 없다(Table 2 note).

### §4.4 Michelangelo: latent-structure query

[[michelangelo]]는 context 안에 hidden structure에 대한 relevant update와 irrelevant text를 넣고, query가 그 structure를 묻게 하는 task를 만든다. Complexity, 즉 relevant update 수는 고정한 채 length를 늘린다(§3). Task는 다음과 같다. Latent List는 operation을 거치며 Python list를 추적한다. MRCR은 긴 conversation에서 여러 비슷한 writing request 중 i번째를 재생산한다. IDK는 context에 fact가 없을 때 `I don't know`라고 답하며, 이는 instance의 70%다(§2). 저자들은 model이 종종 32K에서 이미 악화한다고 보고하며, 모든 Michelangelo evaluation에 걸친 “very common trend”를 “short-context에서 하나의 초기 급격한 super-linear performance drop” 뒤에 flat 또는 대략 linear한 decline이 오는 것으로 묘사한다(§5.2). 128K에서 model 10개에 대한 rank correlation은 MRCR 대 Latent List 0.64, MRCR 대 IDK 0.043, Latent List 대 IDK −0.25다(Fig. 9). GPT와 Claude는 8K 미만에서 Gemini보다 MRCR을 더 잘하지만 더 빠르게 decay한다(§5.3). **상태:** 평가받은 model family 하나를 training한 조직이 낸 Result(single study)다.

### §4.5 OpenAI MRCR과 Graphwalks

둘 다 official OpenAI dataset이다([[openai-mrcr-graphwalks]]).

**MRCR.** Synthetic conversation이 동일한 request 2개, 4개 또는 8개를 숨긴다. model은 주어진 hash를 prefix로 하여 i번째 response를 반환해야 한다. Distractor response는 needle과 같은 generator에서 온다. Token bin당 sample 100개이며, bin은 4,096부터 1,048,576 token까지다.

```
score = 0                                  if response does not start with hash
score = 2·M / (|response| + |answer|)      otherwise (difflib SequenceMatcher ratio)
```

M은 matched character 수다. **계산 예:** hash 뒤의 answer가 `abcd`, response가 `abce`면 M = 3, ratio = 6/8 = 0.75다. 잘못된 poem을 복사해도 두 poem이 character sequence를 공유하면 0보다 높은 score를 받는다. 따라서 MRCR score는 accuracy가 아니라 similarity score다(metric definition에서 해석).

**Graphwalks.** Prompt는 directed edge를 나열하고 주어진 depth에서 BFS frontier 또는 node의 parent를 묻는다. 마지막 줄은 반드시 `Final Answer: [...]`이어야 한다. **계산 예:** gold {a, b, c}, answer {a, b, d}이면 precision 2/3, recall 2/3, F1 = 0.667이다. `Final Answer:` 줄이 없으면 score는 0이다.

**관리자가 기록한 measurement error.** 2025-12-05까지 MRCR item 약 10%는 target needle이 너무 많았고 약 5%는 ground truth가 틀렸다. 2026-02-27까지 Graphwalks `parents` item 400개 중 24개는 ground truth가 틀렸다. 더 이른 dataset revision에서 계산한 score는 이후 revision과 비교할 수 없다. Model별 결과는 이 장에서 가져오지 않은 GPT-4.1 release post에 있다(여기에는 보고하지 않는다).

### §4.6 BABILong과 Fiction.LiveBench

[[babilong]]은 PG19 book text 안에 bAbI reasoning fact를 숨긴다. Background text가 있으면, 시험한 모든 LLM은 QA2(two supporting fact)에서 85% 아래로 떨어지고 QA3은 80% 아래에 머문다(§3.1). RAG는 모든 length에서 single-fact QA 약 60%에 도달하고 QA2와 QA3에서는 random보다 낮아진다(§3.2). BABILong과 MMLU의 correlation은 0K에서 가장 높고 length와 함께 감소한다. RULER가 MMLU와 가장 높은 correlation을 보이는 length에서 MMLU와의 R²는 RULER의 ≤128K average가 0.928, BABILong이 0.435다(§3.4). 저자들은 이를 BABILong이 짧은 length에서는 general knowledge와 분리된 long-context skill을 가른다는 뜻으로 읽는다(해석).

[[fiction-livebench]](practitioner evidence, private question)는 같은 story question을 최소 `0`-token version과 더 긴 version에서 묻는다. 2025년 9월 table에서 claude-sonnet-4는 0 token에서 100.0, 120k에서 36.4를 기록하고, claude-sonnet-4:thinking은 120k에서 81.3이다. 이 page는 run count나 interval을 보고하지 않는다.

## §5 논쟁 중인 proxy로서의 perplexity

**정의.** N개 token에 대한 perplexity는 \(\mathrm{PPL} = \exp\bigl( -(1/N) \sum_t \log p(x_t \mid x_{<t}) \bigr)\)이다. 여기서 \(x_t\)는 position \(t\)의 token, \(x_{<t}\)는 그 앞선 token들, \(p\)는 model이 예측한 확률이다. **계산 예:** token당 mean negative log-likelihood가 2.0 nats이면 \(\mathrm{PPL}=e^2\approx 7.39\)다.

**엇갈리는 결과.**

- [[controlled-long-context-extension]]은 LLaMA2-7B, 동일한 data 1B token, 32k training length(NTK-64K variant는 64k)를 고정하고 extension method만 바꾼다. Exact-attention fine-tuned method들에서는 PPL@32K가 NIAH, LongBench, RULER와 상관한다(§6, Fig. 4). 저자들은 perplexity가 “remains a general-purpose performance indicator”라고 결론 내린다(Abstract). 예외는 LM-Infinite다. 이 방법은 32k에서 PG19 perplexity 6.71이지만 32k RULER score는 12.34다(Tables 2–3).
- [[prolong]]은 method를 고정하고 Llama-3-8B의 data mixture를 바꾼다. Long-data share가 100%로 올라갈수록 PG19 perplexity는 계속 좋아지지만 downstream long-context average는 내려간다(§2.1, Fig. 1).

**해석(이 코스).** 두 결과는 서로 다른 비교를 측정한다. Data를 고정한 채 position method를 비교하면 perplexity는 먼 token을 전혀 사용하고 있는지를 추적한다. Data mixture를 비교하면 long book text를 더 넣을수록 book perplexity는 낮아질 수 있지만 instruction-following retrieval, RAG, re-ranking은 좋아지지 않는다. ProLong도 이 후자의 능력들이 SFT 뒤에야 측정 가능해진다고 발견했다(§2.2, Fig. 2). 짧은 length에서도 perplexity와 task score는 반대 방향으로 움직일 수 있다. PI, NTK-32K, NTK-64K는 2k에서 LLaMA2보다 높은 PG19 perplexity(6.63–6.88 대 6.61)를 갖지만, 4K RULER score는 더 높다(84.56–86.60 대 80.94)(Tables 2–3).

**시사점.** Data mixture를 고르거나 checkpoint promotion을 결정할 때는 SFT 뒤 downstream task를 사용한다. Perplexity는 fixed data에서 position method를 비교할 때만 쓸 수 있는 early signal이다. 그 setting 밖에서도 통하는지는 Open question이다.

## §6 Hyperparameter로서의 평가 configuration

### §6.1 측정된 효과가 있는 setting

| Setting | 측정된 효과 | 출처와 위치 |
|---|---|---|
| 추론 시 `rope_scaling` | LongBench v2에서 Qwen2.5-72B-Instruct는 YaRN factor 4.0으로 39.4 → 42.1, Qwen2.5-7B는 27.0 → 30.0 | [[longbench-v2]] App. E Table 4 |
| 추론 시 RoPE base 변경 | base를 500,000 → 16,000,000으로 바꾼 Llama-3-Inst는 32,768 token 뒤에서 하락한다. Llama-3-8B-Inst도 짧은 length에서 ODQA와 ICL score를 잃는다 | [[helmet]] App. E.3 |
| Scaling factor는 test length에 의존 | 같은 checkpoint에 Dynamic NTK factor는 32k까지 29, 64k에서 61이다 | [[controlled-long-context-extension]] Tables 7-8 |
| Assistant prefill / answer prefix | NIAH에서 Claude 2.1은 `Here is the most relevant sentence in the context:`라는 prefill로 27% → 98%가 된다. RULER는 모든 model에 answer prefix를 붙인다 | [[needle-in-haystack-data]] Connections (Anthropic post, 2023-12-06); [[ruler]] App. D |
| In-context demonstration | 128K JSON KV에서 Llama-3.1-8B base는 2-shot으로 77.3 → 98.0, GPT-4o는 99.3 → 36.7이다 | [[helmet]] Table 8 (3 seeds) |
| Harness prompt | Llama-3.2 1B/3B는 ∞Bench에서 1.8-2.8, HELMET에서 21.2-36.9다 | [[helmet]] Table 9 |
| Reasoning mode | 120k에서 claude-sonnet-4는 thinking을 켜면 36.4 → 81.3이다 | [[fiction-livebench]] Sept 2025 table |
| Truncation side | LongBench v2는 가운데에서 자르고, HELMET은 LongQA document 끝에서 자른다. 이 출처들에는 controlled comparison이 없다 | [[longbench-v2]] §4.1; [[helmet]] §2.1 |
| Length unit | HELMET은 Llama-2 token, BABILong은 GPT-2 token, LongBench v2는 word, MRCR은 o200k token을 센다 | [[helmet]] §3; [[babilong]] App. F; [[longbench-v2]] §1; [[openai-mrcr-graphwalks]] |
| Serving stack와 precision | RULER와 HELMET은 BFloat16으로 실행한다. Fiction.LiveBench row에는 provider와 fp8이 표시되어 있고, Llama 4 row는 “after inference provider updated with vllm fixes” 뒤 다시 실행됐다. 이 출처에는 이 benchmark들에서 KV-cache quantization을 측정한 연구가 없다 | [[ruler]] §4; [[helmet]] App. D; [[fiction-livebench]] Changelog 4/10/2025 |

효과 크기는 Qwen2.5-72B에서 YaRN의 2.7 point부터 Claude 2.1 prefill의 71 point까지다. 여기서 사용한 출처에서 KV-cache precision이 long-context score에 미치는 효과는 Open question이다. serving된 model의 configuration은 평가한 configuration과 다를 수 있으므로 이를 기록해야 한다. 예를 들어 Gradient 1M model card는 vLLM에서 `--max-model-len 32768`로 serving하라고 권한다([[gradient-llama3-1048k]]).

### §6.2 모든 long-context 수치에 필요한 최소 기록

```yaml
model_revision: <hub id @ commit>
rope_scaling: {type: yarn, factor: 4.0, original_max_position_embeddings: 32768}   # 또는 null
max_model_len: 131072
truncation: middle            # middle | left | right, 그리고 answer에 남겨 둔 token budget
length_unit: tokens:<tokenizer>
chat_template: <file hash>;  answer_prefix: "<text>";  demonstrations: 0 | 2
decoding: {temperature: 0.0, max_new_tokens: <n>, reasoning: off | <budget>}
serving: {engine: vllm==<ver>, dtype: bfloat16, kv_cache_dtype: auto | fp8}
dataset_revision: <commit>    # MRCR과 Graphwalks는 release 뒤 ground truth가 바뀌었다
n_items_per_cell: <n>;  seeds: <k>
```

Field name은 Hugging Face와 vLLM convention을 따른다. 표시한 value는 illustrative example이며, 예외는 [[longbench-v2]] App. E에서 온 YaRN factor 4.0이다.

### §6.3 Confidence interval을 포함한 paired short-context regression

**문제.** Extension stage는 short-context ability를 제거해서는 안 된다(ch-32b). 1,000-item benchmark에서의 single-point difference는 sampling noise 안에 있을 수 있다. 같은 item을 비교하면 그 noise는 더 작아진다.

**절차.**

1. 같은 configuration(§6.2)으로 base checkpoint와 extended checkpoint를 같은 item에 실행한다.
2. 각 item (i)에 대해 \(d_i = correct_{ext}(i) - correct_{base}(i) \in \{-1, 0, 1\}\)를 계산한다.
3. Mean difference와 95% interval을 보고한다.

```
d̄ = (1/n) Σ d_i ,   SE = sqrt( (mean(d²) − d̄²) / n ) ,   CI95 = d̄ ± 1.96 · SE
```

여기서 \(n\)은 item 수, \(d̄\)는 paired difference 평균, \(mean(d^2)\)는 item별 difference 제곱의 평균, \(SE\)는 standard error, 1.96은 양측 95% interval에 쓰는 normal quantile이다.

**Worked example (toy number).** \(n=1{,}000\)이라고 하자. Base는 700개를 맞히고 extended model은 690개를 맞힌다. 60개는 맞음에서 틀림으로, 50개는 틀림에서 맞음으로 바뀐다. \(d̄=(50-60)/1000=-0.010\)이다. \(mean(d^2)=110/1000=0.110\)이다. \(SE=\sqrt{(0.110-0.0001)/1000}=0.0105\)다. 따라서 \(CI95=-1.0 \pm 2.1\) point, 즉 \([-3.1,+1.1]\)이다. Unpaired interval은 \(\sqrt{0.70\cdot0.30/1000+0.69\cdot0.31/1000}=0.0206\)을 쓰므로 ±4.0 point가 된다. 여기서는 890개 item의 결과가 바뀌지 않았기 때문에 pairing이 interval을 절반으로 줄인다.

**왜 필요한가를 보여 주는 근거.** [[context-length-alone-hurts]]는 모든 experiment를 한 번씩만 실행한다(App. A.4). [[longbench-v2]]는 503개 item이 “could also lead to less stable results”라고 쓴다(§6). Accuracy 50%인 503 item의 unpaired SE는 \(\sqrt{0.25/503}=0.022\), 따라서 도출한 interval은 ±4.4 point다. SFT 뒤 ProLong과 Llama-3-8B-Instruct는 five-task short average에서 0.4 point만 다르다(69.4 대 69.8). 그러나 MMLU는 2.4 point(64.6 대 67.0), GSM8K는 9.6 point(58.9 대 68.5) 차이다([[prolong]] App. B.7, Table 25). 따라서 평균 하나는 task별 regression을 가릴 수 있다.

## §7 1M–10M window 주장: 무엇이 증거가 되는가

- **Gradient Llama-3-8B-Instruct-Gradient-1048k.** Model card는 네 stage 전체에서 학습 token이 총 1.4B개였다고 밝히며(RoPE \(\theta\)는 15.3M에서 2.80B까지), 근거로 NIAH heatmap과 RULER ranking image를 제시한다([[gradient-llama3-1048k]]). “최소한의 학습으로 long context에서 작동하는 법을 배울 수 있다”는 capability 주장은 이 코스의 label로는 **anecdotal**이다. Card text에서 이를 뒷받침하는 것은 수치값이 없는 NIAH heatmap과 RULER image뿐이기 때문이다. Independent measurement에서는 8B와 70B Gradient model 모두 RULER effective length가 16K였고, 128K score는 각각 69.5와 72.1이었다([[ruler]] README; Table 3). RULER 저자들은 GradientAI/Llama3를 1M에서 학습한 “less performant models” 목록에 넣는다(§4).
- **Llama 4 Scout.** Meta는 supported token 수가 10M이고 256K에서 학습했다고 밝힌다. Post가 제시한 long-context 근거는 NIAH retrieval과 code의 cumulative NLL이다([[llama-4]]). Interconnects post는 NIAH 외의 평가는 공개되지 않았다고 지적하고, 초기에 나온 third-party result에는 serving issue가 있었을 가능성을 제기한다([[interconnects-llama-4-long-context]]; **anecdotal**, 증거로 사용하지 않음). Practitioner evidence로 Fiction.LiveBench의 2025년 9월 table에서 llama-4-scout:free는 0-token length에서 62.5, 120k에서 27.3을 기록한다. 따라서 낮은 long-context score를 short-context failure와 분리해 볼 수 없다([[fiction-livebench]]).

Supported window라는 주장은 그 길이의 input을 받아들인다는 증거일 뿐이다. Effective-length claim에는 suite, threshold rule, 길이별 score, 그리고 §6.2의 configuration이 필요하다.

## 레시피

출처에 인쇄된 evaluation-gate 값이다. `2026-09-15`로 표시한 행은 이 장을 위해 primary text에서 읽었고, `2026-09-14` 행은 검증된 library card에서 가져왔다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| RULER main run(정렬된 17개 모델, 예: Meta-Llama-3.1-70B-Instruct) | 7B-141B | eval-gate | effective-length threshold | 85.6 = Llama2-7B(chat)의 4K에서 13-task average | arXiv:2404.06654v3 §4, Table 3 ([[ruler]]) | verified 2026-09-15 | ablation 보고 없음. §1에서 “qualitative threshold”라고 설명 |
| RULER main run | 7B-141B | eval-gate | task; example; length | 13 task, task·length당 500개, 4K·8K·16K·32K·64K·128K | §4 | verified 2026-09-15 | task set은 App. C의 correlation study가 선택 |
| RULER main run | 7B-141B | eval-gate | prompt; decoding; precision | model chat template + task answer prefix; VT와 CWE에는 one demonstration; greedy; vLLM의 BFloat16, A100 8장 | §4; App. D Table 6 | verified 2026-09-15 | ablation 보고 없음 |
| RULER main run | 7B-141B | eval-gate | ranking weight | length에 따라 선형으로 증가(inc) 또는 감소(dec) | §4 | verified 2026-09-15; weight 1-6은 §1 worked example에서 도출 | ablation 보고 없음 |
| NoLiMa(13개 모델, 예: GPT-4o, Llama 3.3 70B) | 다양함; open model 8B-405B | eval-gate | effective-length threshold | `0.85 × base score`; base = 250/500/1K에서 pair별 max score의 평균 | arXiv:2502.05167v3 §4.3 ([[nolima]]) | verified 2026-09-14 | ablation 보고 없음 |
| NoLiMa | 위와 같음 | eval-gate | length별 test; decoding | 7,540개(5 haystack × 58 pair × 26 placement); instruction model은 greedy; reasoning model은 generated token을 1,536으로 cap | §4.1; App. C | verified 2026-09-14 | ablation 보고 없음 |
| HELMET(59개 모델) | 다양함(Table 15) | eval-gate | length; decoding; sample | 8,192 / 16,384 / 32,768 / 65,536 / 131,072 Llama-2 token; greedy; 600(JSON KV, NQ, PopQA, TQA), 300(MS MARCO, HotpotQA), 500(ICL), 100(나머지) | arXiv:2410.02694v3 §3; App. D ([[helmet]]) | verified 2026-09-14 | Table 8: 0-shot 대 2-shot, 3 seed |
| HELMET | 위와 같음 | eval-gate | judge | GPT-4o-2024-05-13; human 대비 Cohen's κ 0.72-0.91 | §2.2; App. B.6 | verified 2026-09-14 | App. B.6 human-agreement study |
| LongBench v2(17개 모델) | open 7B-123B; closed model은 보고하지 않음 | eval-gate | item; truncation; setting | 503개 MCQ; model window를 넘으면 가운데를 truncate; zero-shot 및 zero-shot + CoT | arXiv:2412.15204v2 §1, §4.1 ([[longbench-v2]]) | verified 2026-09-15 | Table 2: open model에서 CoT 평균 +3.4 |
| LongBench v2의 Qwen2.5-72B-Instruct | 72B | eval-gate | rope_scaling | YaRN, factor 4.0(model card 기준); overall 39.4 → 42.1 | App. E Table 4 | verified 2026-09-15 | Table 4(one run) |
| Michelangelo(10개 frontier model) | 보고하지 않음 | eval-gate | context subset; prompting | 최대 32K·128K·1M; repetition count로 나눈 bucket; few-shot short demonstration | arXiv:2409.12640v2 §3.2-3.3 ([[michelangelo]]) | verified 2026-09-15 | ablation 보고 없음 |
| OpenAI MRCR dataset | 해당 없음 | eval-gate | needle; sample; bin; metric | 2·4·8; bin당 100개; prompt+answer 합계 4,096-1,048,576 token의 8 bin; hash prefix가 없으면 0인 SequenceMatcher ratio | huggingface.co/datasets/openai/mrcr README @f4c69fa ([[openai-mrcr-graphwalks]]) | verified 2026-09-15 | 보고하지 않음 |
| OpenAI Graphwalks dataset | 해당 없음 | eval-gate | problem type; metric | bfs, parents; 마지막 줄 `Final Answer:` list의 set F1, parse failure = 0 | huggingface.co/datasets/openai/graphwalks README @be6cc6e | verified 2026-09-15 | 보고하지 않음 |
| BABILong(§3.1의 34 LLM) | 다양함(Table 4) | eval-gate | threshold; sample | satisfactory > 85%, failure < 30%; task·length당 100 sample(Table 4: 32K까지 1,000개) | arXiv:2406.10149v2 §3.1; App. N.2; Table 4 caption ([[babilong]]) | verified 2026-09-14 | ablation 보고 없음 |
| ProLong 8B development run | 8B | eval-gate | development protocol | UltraChat SFT 뒤 HELMET subset(recall, RAG, re-rank, ICL, QA, summarization)을 32K와 64K에서 average, plus short task 5개 | arXiv:2410.02660v4 §2; App. A.1 ([[prolong]]) | verified 2026-09-14 | §2.1 Fig. 1(perplexity가 오도함); §2.2 Fig. 2(SFT 전후) |
| Llama-3-8B-Instruct-Gradient-1048k | 8B | long-context | token; stage; RoPE θ; LR | 총 1.4B(1048k stage 838,860,800); 65K → 262K → 524k → 1048k; 15.3M / 207.1M / 1.06B / 2.80B; 2.00E-05 | HF model card @cd3069b ([[gradient-llama3-1048k]]) | verified 2026-09-15 | card 근거: 값 없는 NIAH heatmap과 RULER image(**anecdotal**); RULER README: effective 16K |
| Llama 4 Scout | 17B active / 109B | long-context | trained context; supported context | pre/post-training 256K; supported 10M | Meta post 2025-04-05, Scout paragraph ([[llama-4]]) | verified 2026-09-14 | post 근거는 NIAH와 cumulative NLL뿐 |

**작은 general-purpose run의 출발점.** 128K로 extension한 8B model(ch-32b)에는 검증된 위 행들이 이 gate를 뒷받침한다. RULER이 A100에서 7B-141B aligned model에 했듯이, 4K-128K에서 13 task를 task당 500 example으로 실행하고, chat template + answer prefix, greedy BFloat16을 사용하며 85.6 threshold와 함께 길이별 score를 보고한다. 32K까지 13개 model에 실행한 방식대로 `0.85 × base` rule과 length당 7,540 test를 쓰는 NoLiMa를 추가한다. HELMET의 일곱 category를 8,192-131,072 token에서 dataset별 sample count로 추가하고, ProLong이 8B에서 했듯 SFT 뒤에 평가한다. §6.2의 configuration field를 기록하고 short-context regression은 paired interval(§6.3)로 보고한다. 이 출처들 가운데 1B parameter 미만 model용 gate를 시험한 것은 없다.

## 일반화 관점

**(a) long-context capability의 breadth를 넓히는 것.**

- 여러 task family를 동시에 평가한다. family가 model rank를 다르게 매기기 때문이다. HELMET category correlation은 0.34까지 떨어지고([[helmet]] Fig. 5), Michelangelo task correlation은 −0.25까지 떨어진다([[michelangelo]] Fig. 9). 어느 연구에서도 모든 category에서 선두인 model은 없다.
- perplexity 대신 SFT 뒤 downstream long-context task로 training choice를 고른다([[prolong]] §2.1-2.2).
- 같은 context training이라면 큰 model이 length에 덜 저하된다. Yi-34B-200K 대 Yi-6B-200K가 그 예다([[ruler]] §6). 이는 single study의 결과다.
- inference-time reasoning은 일부 보상한다. LongBench v2의 open model에서 CoT는 평균 +3.4이고([[longbench-v2]] §4.1), 32K two-hop NoLiMa는 25.9에서 34.3으로 오른다([[nolima]] Table 4). 방향은 복제됐지만 effect size는 다르다.

**(b) narrowing 또는 forgetting을 일으키는 것.**

- window를 늘리기 위한 position 변경이 short-context score를 낮출 수 있다. 추론 시 RoPE base를 바꾼 뒤 Llama-3-8B-Inst의 ODQA와 ICL이 그러했으며([[helmet]] App. E.3), NTK-32K 뒤 LongBench LCC는 68.22 → 56.78이었다([[controlled-long-context-extension]] Table 11).
- NIAH형 retrieval에만 최적화하는 경우다. NIAH에서 거의 100%인 model도 RULER의 aggregation과 tracing([[ruler]] §4), NoLiMa association([[nolima]] Table 6)에서는 실패한다. retrieval-only gate suite는 lexical matching을 선택한다.
- 강한 instruction tuning은 ICL을 낮출 수 있다. HELMET ICL에서 많은 open model이 closed model을 이긴다는 저자들의 해석이 있다([[helmet]] §3.3).
- long-only training data도 원인이다. 100% long data는 SFT 뒤 downstream long-context score를 낮추고 short-task score를 단조롭게 낮췄다([[prolong]] §3.2, Fig. 3).

**(c) 이 단계에서 측정하는 법.**

- effective-length 하나가 아니라 길이별 curve를 보고, relative threshold와 fixed threshold를 나란히 둔다(§1).
- depth slice와 best-minus-worst gap을 보고한다([[lost-in-the-middle]] §1).
- low-overlap 및 distractor slice를 둔다([[nolima]]; [[chroma-context-rot]]).
- retrieval-controlled accuracy를 측정한다. 즉 같은 problem을 여러 length에서 출제하고 recitation check를 넣는다([[context-length-alone-hurts]] §3.1).
- memorization의 상한을 정하는 no-context baseline([[longbench-v2]] Table 3)과 leakage를 피하는 regenerable synthetic task([[michelangelo]] §1; [[babilong]] §2)를 둔다.
- abstention은 wrong answer와 분리해 채점한다([[michelangelo]] IDK; [[chroma-context-rot]] LongMemEval).
- interval을 포함한 paired short-context regression(§6.3)을 보고하고 dataset revision을 pin한다(§4.5).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 방법 |
|---|---|---|
| 주장했거나 configured한 window를 usable window로 취급 | release에는 1M이라고 쓰였지만 RULER 길이별 score는 16K-32K에서 85.6 아래 | 이름 붙인 threshold 아래에서 길이별 score로 (L_{eff})를 계산(§1) |
| single-needle NIAH로 extension을 gate | 모든 checkpoint가 100 근처인데 application score는 갈림 | RULER MK/MV/VT/CWE, NoLiMa, HELMET RAG task를 추가하고 checkpoint 간 rank correlation 계산 |
| score 없이 effective-length column만 복사 | leaderboard effective length가 자신의 길이별 score와 불일치 | 길이별 score에서 재계산(Qwen3-8B README row, §1) |
| needle depth 전체를 하나의 average로 보고 | 평균은 괜찮아 보이지만 failure가 middle에 집중 | depth별 accuracy plot과 best-minus-worst gap 보고 |
| perplexity로 long-data mixture 선택 | PG19 perplexity는 좋아지지만 SFT 뒤 RAG·recall은 하락 | downstream task에서 SFT 뒤 candidate mixture 평가([[prolong]] Fig. 1) |
| 다른 `rope_scaling`으로 실행한 수치를 비교 | 같은 model이 report마다 3 point 차이 | run마다 `rope_scaling`과 `max_model_len` 기록(§6.2) |
| instruction-shaped task에서 base model을 zero-shot 평가 | 2 demonstration으로 급상승하는 낮은 score(Llama-3.1-8B base JSON KV, 128K: 77.3 → 98.0) | 0-shot과 2-shot을 모두 실행·보고([[helmet]] Table 8) |
| pin하지 않은 dataset revision에서 채점 | model 변화 없이 MRCR/Graphwalks score가 변함 | dataset commit 기록, changelog 점검 |
| 작은 unpaired difference에서 “short-context regression 없음” 선언 | interval 없이 −1 point 변화를 neutral 또는 loss로 보고 | CI95와 paired per-item difference 사용(§6.3) |
| abstention을 보통 error로 세기 | “context에 없다”고 한 model과 hallucination model의 score가 같음 | abstention과 wrong answer를 별도 채점(IDK; LongMemEval) |

## 이해 점검

1. Llama 3.1 70B의 effective length는 RULER에서는 64K, NoLiMa에서는 2K다. 각 suite의 어떤 속성(task type, literal overlap, threshold rule)이 각각의 숫자를 만드는지 설명하고, paraphrase된 query를 쓰는 RAG application이 어느 쪽에 의존해야 하는지 말하라.
2. (f(i))를 사용해 short document를 128K sequence에 packing해도 128K 근처 distance가 드문 이유를 설명하라. 또한 STRING이 training 없이 128K RULER를 올릴 수 있지만 model을 training length 너머로 extension할 수는 없는 이유를 설명하라.
3. [[context-length-alone-hurts]]는 filler token을 attention에서 mask하고도 30K에서 drop을 발견한다. 이 결과가 배제하는 long-context failure 설명은 무엇이며, 남겨 두는 설명은 무엇인가?
4. 35개 model에서 NIAH는 ∞Bench QA와 0.63, HotpotQA RAG와 0.88 상관한다. tie-handling rule을 사용해 NIAH의 score saturation이 Spearman의 (
ho)를 어떻게 낮추는지 설명하라.
5. Lu et al.은 perplexity가 downstream score를 추적한다고 찾았고 ProLong은 그렇지 않다고 찾았다. 이 불일치가 method variation 때문인지 data variation 때문인지 판별할 experiment를 기술하라.
6. 한 team은 YaRN factor 4가 extended model의 LongBench v2를 다른 lab의 YaRN 없는 수치보다 2.7 point 좋게 만들었다고 보고한다. 비교가 valid하기 전에 맞춰야 할 configuration field를 나열하고, 각 field가 score를 바꿀 수 있는 이유를 설명하라.
7. 128K extension이 1,000-item short benchmark를 70.0에서 68.0으로 바꾸고, right-to-wrong flip 80개와 wrong-to-right flip 60개가 있었다. paired 95% interval을 계산하고, 이것이 unpaired interval보다 좁은 이유와 gate가 내려야 할 결론을 말하라.
8. Llama 4 Scout의 120k Fiction.LiveBench score 27.3은 specifically long-context ability의 약한 증거인 이유는 무엇이며, length effect를 분리하려면 어떤 추가 측정이 필요한가?

## 연결

- **ch-32b (previous)** — *Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression.* 이 장의 gate가 측정하는 method와 data이며, STRING의 position-frequency 관점이 두 장을 연결한다.
- **ch-32d (next)** — *Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training.* repository와 trajectory data는 long input이므로 §4와 §6이 평가에 적용된다.
- **ch-28** — *Long-Context Data Synthesis and Synthetic Evaluation Task Families.* RULER-style generator를 training data로 다룬다.
- **ch-32f** — *Lab: Annealing and Context Extension with a Short-Context Regression Gate.* Recipe paragraph와 §6.3의 gate를 구현한다.
- **ch-44a** — *Length in RL: Overlong Responses, Length Control, and Long-Context RL.* 긴 output과 long-context RL을 다룬다.
- **ch-45c** — *Context Management for Long-Horizon Agents.* §3의 length-only effect로 context management의 동기를 제시한다.
- **ch-47** — *Evaluation Harness and Suite Design for General Capability.* general harness design이며 여기서는 long-context evaluation으로 연결된다.

## 출처

- [[ruler]] — task suite, effective-length definition, 85.6 threshold, Table 3·README leaderboard row, answer-prefix protocol.
- [[nolima]] — literal-overlap 측정, relative threshold, Table 3, association·CoT ablation.
- [[helmet]] — 일곱 category, NIAH·RAG correlation, judge metric, demonstration·RoPE-base configuration effect.
- [[longbench-v2]] — human-verified long MCQ, no-context baseline, middle truncation, YaRN result.
- [[michelangelo]] — latent-structure query, 32K 전의 early degradation, cross-task rank correlation.
- [[openai-mrcr-graphwalks]] — MRCR·Graphwalks task definition, grading code, dataset correction.
- [[lost-in-the-middle]] — position sensitivity와 best-minus-worst gap.
- [[retrieval-head]] — retrieval score, sparsity, stability, masking effect.
- [[string-effective-context]] — position-frequency 설명과 STRING result.
- [[context-length-alone-hurts]] — retrieval을 통제한 length-only accuracy drop, recite-then-solve.
- [[chroma-context-rot]] — similarity, distractor, LongMemEval control이 있는 18-model practitioner evaluation.
- [[babilong]] — reasoning-in-a-haystack threshold, 10-20% context use, RAG limit.
- [[fiction-livebench]] — practitioner comprehension ladder, reasoning-mode·Llama 4 row.
- [[controlled-long-context-extension]] — controlled method comparison 아래 perplexity correlation.
- [[prolong]] — data change 아래 perplexity 대 downstream, SFT 뒤 평가, short-task result.
- [[needle-in-haystack-data]] — original NIAH harness와 Claude 2.1 prefill result.
- [[gradient-llama3-1048k]] — 1M-window training setting과 NIAH-only evidence(**anecdotal** label).
- [[llama-4]] — Scout의 256K training length, 10M supported window, 명시된 근거.
- [[interconnects-llama-4-long-context]] — Llama 4 long-context evidence에 관한 commentary(**anecdotal**).
