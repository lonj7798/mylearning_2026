<!-- chapter: ch-29a
     track: synthetic
     kind: content
     title: Long-Document Synthesis for Continued Pretraining and Long-Context SFT
     deps: [ch-29, ch-32b, ch-32c]
     sources: [[llama-3]], [[long-context-llama3]], [[excerpts/llama-3-long-context-sft]], [[qwen-long-context-synth]], [[yi]], [[excerpts/yi-long-context-recitation]], [[chatqa-2]], [[in2-film]], [[hierarchical-million-token-synth]], [[nemotron-nano-2]], [[quest-query-centric-synthesis]], [[nextlong]], [[ultralong-128k-to-4m]], [[prolong]], [[qwenlong-l1-5]], [[loongrl]], [[artificial-needles-real-haystacks]], [[longmit]], [[excerpts/longmit-mimg]], [[context-synthesis-short-to-long]], [[clipper-compression-synth]], [[bootstrap-context-length]], [[smollm-3]], [[excerpts/smollm3-long-context-merge]], [[longcite]], [[nolima]], [[lost-in-the-middle]], [[agentinstruct]]
     figures: figures/long-sequence-layouts.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29a — 지속 사전학습과 긴 컨텍스트 SFT를 위한 장문서 합성

> **핵심 통찰.** 합성한 긴 학습 예제가 모델에게 긴 컨텍스트 사용법을 가르치려면, 목표(target)가 목표 지점에서 멀리 떨어진 토큰에 실제로 의존해야 하며, 바로 주변의 토큰이나 모델 파라미터에 이미 저장된 지식만으로는 그 목표를 만들 수 없어야 한다. 데이터 양을 고정한 두 비교에서 이 의존성을 강제한 생성기가 더 좋았다. Llama-3-8B를 128K에서 학습할 때, 한 문서의 조각들 사이에 검색으로 찾은 유사하지만 무관한 hard-negative 청크를 넣으면 HELMET+RULER 평균이 62.58이었고, 무작위 연결은 52.85였다([[nextlong]] Table 1). RL에서 UUID 사슬 뒤에 질문을 숨기면, 같은 양의 평범한 긴 multi-hop QA는 66.2인 데 비해 72.4였다([[loongrl]] Table 4). 증거 세그먼트를 임의 깊이로 섞은 QA로 학습하면 7B 모델의 최저 위치와 최고 위치 성능 차도 56.2점에서 13.9점으로 줄었다. 이 비교는 다른 생성기와의 비교가 아니라, 학습하지 않은 backbone을 기준으로 한 변화다([[in2-film]] Table 1). 합성 장문 데이터는 모델의 능력을 좁힐 수도 있다. 사실을 포함한 장문 데이터 학습은 Mistral-7B의 TriviaQA를 2.19~6.33점 낮췄지만, 사실이 없는 key-value 데이터는 +0.11점 변화였다([[artificial-needles-real-haystacks]] Table 2). 긴 컨텍스트 학습 실행에서는 개별 짧은 벤치마크가 0.1~16.9점 떨어진 사례도 보고된다(§7).
>
> **지침.** 목표 길이의 긴 자연 문서가 부족하면, 종속적인 조각 사이를 유사하지만 무관한 텍스트로 떼어 놓는 CPT 시퀀스(NExtLong, Quest)를 만들거나, 무작위 청크에 관한 QA를 문서 뒤에 붙인다(Nemotron Nano 2). 각각 자신의 ablation에서 평범한 연결 또는 합성 데이터 없음보다 좋아졌기 때문이다(§2). 긴 SFT 데이터를 만들 때는 생성기가 문서 전체가 아니라 문서의 일부(청크, 개요, 요약, 검색된 문단)를 읽게 하고, 학생 모델은 전체 문서를 읽고 답하게 한다. 이어서 문서 없이도 어떤 모델이 맞히는 예제는 제거한다. 책 전체를 읽은 생성기는 오류가 있는 claim을 73.1% 만들었지만 outline에서 생성하면 16.7%였고([[clipper-compression-synth]] Table 1), closed-book으로 답할 수 있는 질문은 컨텍스트 사용이 아니라 지식을 측정하기 때문이다([[qwenlong-l1-5]] §3). base model이 이미 광범위한 긴 CPT를 받았다면, 합성 긴 SFT를 추가하기 전에 짧은 SFT만으로 먼저 시험한다. 8B 연구 두 개는 짧은 SFT만으로 충분하다고 했지만([[prolong]] Table 8; [[ultralong-128k-to-4m]] §3.2), 다른 두 연구는 불충분하다고 했다([[excerpts/llama-3-long-context-sft]] §4.3.4; [[bootstrap-context-length]] Table 3). 모든 경우에 학습 전후로 짧은 벤치마크, 지식 벤치마크(TriviaQA, NQ), 지시 이행(IFEval)을 함께 측정한다.

## 범용 모델에서 이 장이 중요한 이유

범용 모델은 200쪽 계약서, 코드 저장소, 검색된 웹페이지 묶음, 긴 도구 로그처럼 긴 입력을 받고, 기억에 의존하지 않고 그 입력에서 답해야 한다. 128K 토큰 이상인 자연 문서는 소수 도메인에만 많다. ProLong은 길이 64K 이상 코드 저장소가 98.8B 토큰인 데 비해, 그런 CommonCrawl 문서는 15.3B 토큰이라고 집계했다([[prolong]] §3.1, Table 3). Quest도 현존 긴 문서의 성능이 더 약한 이유를 긴 문서가 “몇 개 도메인에만 존재한다”는 점으로 돌린다([[quest-query-centric-synthesis]] Table 5). 긴 입력의 사람 주석은 Llama 3에서 “대체로 비현실적”이라고 표현됐고([[excerpts/llama-3-long-context-sft]] §4.3.4), QwenLong-L1.5에서는 32K를 넘는 컨텍스트에서 주석자가 어려운 질문을 만들면서 답을 빠짐없이 검증하기가 어렵다고 설명한다([[qwenlong-l1-5]] §3).

합성은 이 부족분을 세 파이프라인 위치에서 채운다. 긴 시퀀스 길이의 continued pretraining(중간 학습, [[ch-32b]]), 긴 컨텍스트 SFT, 긴 컨텍스트 RL 데이터다. 결과의 평가는 [[ch-32c]]에서 다룬다. [[ch-28]]은 초기 생성기(LongAlign, LongMIT, NIAH-style 학습)와 SFT 비율을 둘러싼 상반된 결과를 소개했다. 이 장은 생성기별로 어떤 종속성을 강제하는지, 어떤 검증을 적용하는지, 범용 능력에 어떤 비용이 측정됐는지를 순서대로 다룬다. 긴 대화와 누적되는 agent 컨텍스트는 [[ch-29b]]에 있다.

이 단계에서 범용성에 던져야 할 질문은 세 가지다. (1) 어떤 생성기 선택이 합성하지 않은 실제 작업으로 긴 컨텍스트 기술을 전이시키는가? (2) 어떤 선택이 새 사실에 따른 환각, 짧은 컨텍스트 또는 지시 이행 능력 상실, 글자 그대로 일치하는 검색만 하는 모델 같은 narrowing을 일으키는가? (3) 그런 효과를 어느 측정으로 포착할 수 있는가?

## §1 합성 시퀀스가 긴 컨텍스트에 의존하게 만드는 조건

**정의.** 학습 예제의 *장거리 종속성(long-range dependency)*이란 목표를 결정하는 토큰이 목표에서 떨어져 있고, 모델이 목표를 예측할 때 쓸 수 있는 local window의 범위를 넘는 경우다. *증거(evidence)*는 정답 목표를 만들기 위해 필요한 span들의 집합이다. *증거 깊이(evidence depth)*는 증거 span의 시작 위치를 전체 컨텍스트 길이로 나눈 값이다.

**측정 가능한 문제.** 자연 텍스트는 “긴 거리 연관이 흔히 약하다”([[qwen-long-context-synth]] §3). 따라서 긴 문서 전체에 next-token loss를 적용해도 손실의 대부분은 가까운 토큰으로 예측된다. IN2 저자들은 SFT 데이터에도 같은 가설을 제시한다. system message와 instruction이 맨 앞에 놓이는 경우가 많기 때문이다([[in2-film]] §1, 저자 해석). 측정되는 증상은 위치 의존성이다. 20개 문서 중 정답 문서를 첫 번째에 둔 GPT-3.5-Turbo의 정확도는 75.8%이고, 10번째에 두면 53.8%이며, 문서를 전혀 주지 않은 closed-book 조건은 56.1%다([[lost-in-the-middle]] App. G Table 6, Table 1). 즉 10번째 위치에서는 문서를 넣은 것이 closed-book 점수보다도 정확도를 낮춘다.

**생성기가 제어할 수 있는 다섯 성질.**

1. *증거 개수(evidence count)*: retrieval처럼 span 하나만 필요하게 할지, integration처럼 여러 span을 필요하게 할지 결정한다.
2. *증거 간격(evidence spread)*: 증거 span끼리의 토큰 거리와 증거에서 목표까지의 토큰 거리다.
3. *깊이 균형(depth balance)*: 증거 깊이가 창 전체에서 균일한지 여부다.
4. *질문과 증거의 어휘 중복(lexical overlap)*: 글자 그대로 겹치는 단어가 많으면 문자열 일치만으로 retrieval이 성공할 수 있다. NoLiMa의 needle 집합에서 질문과 needle 사이 ROUGE-1 precision은 0.069이고, 일반 NIAH에서는 0.905다. 이 조건에서 GPT-4o는 32K에서 99.3에서 69.7로 내려간다([[nolima]] Table 1, Table 3).
5. *closed-book 답변 가능성(closed-book answerability)*: 문서를 빼도 모델이 정답을 낼 수 있는지 여부다.

**계산 예시(어휘 중복).** 증거 문장은 “The Berlin office of the company opened in 1998.”라고 하자. 질문 A는 “What year did the company open its Berlin office?”다. A의 내용어는 {year, company, open, Berlin, office}이고, stemming 뒤 다섯 개 중 네 개가 증거에 나타난다. 그러므로 unigram precision은 4/5 = 0.8이다. 질문 B는 “In which year did the firm begin operating in Germany's capital?”이다. 내용어 {year, firm, begin, operating, Germany, capital}는 증거와 하나도 겹치지 않아 precision이 0/6 = 0이다. 질문 B는 Berlin → 독일의 수도라는 연관을 사용해야 한다. 이는 NoLiMa needle이 요구하는 방식과 같고, QwenLong-L1.5는 “20세기 후반에 5로 끝나는 해”처럼 entity를 난독화하여 같은 효과를 만든다([[qwenlong-l1-5]] §3.2).

대화형 그림 [figures/long-sequence-layouts.html](figures/long-sequence-layouts.html)은 이 장의 다섯 생성기가 만드는 학습 시퀀스를 하나의 정규화된 축에 그린다. 이를 사용해 증거, distractor, 질문이 각각 어디에 놓이는지 비교하고, §2와 §7의 계산 예시 숫자도 재현할 수 있다.

## §2 Continued-pretraining 측 합성

여기서 continued pretraining(CPT)은 주 사전학습 뒤에 긴 시퀀스로 하는 next-token 학습을 뜻한다([[ch-32b]]). 합성에서의 질문은 더 짧은 문서들로 길이 T의 시퀀스를 어떻게 조립해야, 뒤쪽 토큰을 예측할 때 앞쪽 토큰이 필요하게 되는가이다.

### §2.1 연결 기준선과 query-centric 그룹화(Quest)

**정의.** *Standard* 연결은 무작위 문서들을 하나의 시퀀스에 넣는다. *KNN*은 각 문서를 검색으로 찾은 가장 유사한 문서들과 함께 넣는다. *ICLM*은 유사한 문서를 traveling-salesman 경로로 정렬한다([[quest-query-centric-synthesis]] §4.2).

**문제.** 무작위 이웃 문서는 앞을 다시 볼 이유를 만들지 않으며, 거의 중복된 이웃 문서는 중복만 만든다. Quest의 Figure 2는 시퀀스 내부 유사도가 증가할수록 성능이 처음에는 오르다가 다시 떨어지는 모습을 보인다(저자 해석).

**방법(Quest §3).**

1. doc2query 모델이 각 문서에 대한 query를 예측한다.
2. RAKE가 그 query에서 keyword를 추출한다. 점수 3.0 미만 keyword와 “best way” 같은 정보성이 없는 phrase는 제거하고, 남은 keyword 하나를 무작위로 골라 문서의 key로 삼는다.
3. 같은 key를 가진 문서는 inverted index에서 한 그룹을 이룬다.
4. 드문 key는 oversampling한다. oversampling 집합의 최적 split 비율은 key의 10–30%였다(App. A.4).
5. 표본으로 뽑은 key 그룹에서 문서를 비복원으로 꺼내 길이 L까지 연결한다.

**근거.** 모든 방법은 같은 30B Pile 토큰을 사용하고, 달라지는 것은 배열뿐이다. 128K에서 Pythia-6.9B의 Longbook QA는 Standard 14.47, KNN 13.38, ICLM 14.92, Quest 17.95였다. 12B에서는 각각 17.81, 16.42, 18.44, 18.92였다(Table 2). 7개 작업의 짧은 평균은 Pythia가 0.4830, Quest 뒤가 0.4831, KNN 뒤가 0.4769였다(Table 3). 결과는 단일 연구의 Pythia와 Llama-3-8B 한 실행에 관한 것이다.

**한계.** keyword를 공유한다고 해서 문서 j의 어떤 토큰이 문서 j+1을 예측하는 데 필요해지는 것은 아니다. Quest는 유용한 종속성이 생길 확률을 높일 뿐, 종속성을 직접 구성하지는 않는다.

### §2.2 Hard-negative 인터리빙(NExtLong)

**정의.** NExtLong은 하나의 짧은 *meta-document*를 *meta-chunk*로 나눈 뒤, 각 meta-chunk 뒤에 corpus에서 검색한 가장 유사한 top-k 청크를 삽입한다. 삽입된 청크는 관련 있어 보이지만 다른 문서에 속한 *hard negative*다([[nextlong]] §3.1).

**방법.**

1. meta-document를 newline에서 나누어, 최대 granularity s의 meta-chunk로 만든다. 공개 실행에서 s = 2048이며, 단위는 명시되지 않았다. App. D.1은 이를 문자 단위로 계산한다.
2. 같은 granularity의 모든 corpus 청크를 embedding하여 FAISS index에 넣는다.
3. meta-chunk m_i마다 n_i1, ..., n_ik를 검색하고, l_i = [m_i, n_i1, ..., n_ik]를 만든다. 전체 시퀀스는 [l_1, ..., l_p]다. 하나의 negative는 여러 meta-chunk에 재사용하지 않는다.
4. negative를 포함한 모든 토큰에 next-token loss를 적용해 학습한다(§3.2).

**수식(App. D.1, Eq. 8–12).**

  k = (T × E × w − S) / (p × s)

여기서 T는 목표 길이(토큰), E는 tokenizer encoding rate(토큰당 문자 수), w = 1.5는 보정 계수, S는 meta-document 길이(문자), p는 meta-chunk 수, s는 granularity다.

**계산 예시.** T = 131,072, E = 4, S = 20,000, p = 10, s = 2,048이라 하자(E, S, p는 설명을 위한 입력값이다). T × E × w = 131,072 × 4 × 1.5 = 786,432자다. 여기서 S를 빼면 766,432이고, p × s = 20,480으로 나누면 k ≈ 37.4다. 따라서 각 meta-chunk 뒤에 hard negative 약 37개가 따른다. meta-document는 20,000 / 786,432 ≈ 2.5%의 문자만 차지한다. meta-chunk i에서 meta-chunk i+1을 예측하려면 모델은 유사한 텍스트 약 37 × 2,048 ≈ 76,000자를 건너야 한다.

**근거.** Llama-3-8B base를 128K, 약 4B 토큰으로 학습했고 모든 방법의 학습 설정은 같았다. HELMET+RULER 평균은 NExtLong 62.58, Quest 55.25, Standard 52.85, KNN 50.97, ICLM 50.37이었다(Table 1). top-k 유사도로 고른 negative가 다섯 선택 전략 가운데 가장 좋았으나 Figure 7은 수치를 출력하지 않는다. meta-chunk를 l_i의 head에 둘 때 62.58, tail에 둘 때 60.01, 무작위 위치에 둘 때 58.95였다(Table 5). 짧은 텍스트 7개 작업 평균은 base 63.75에 대해 63.83이었지만, LAMBADA는 75.66 → 72.06으로 낮아졌다(Table 4). 결과는 한 base model을 쓴 단일 연구다.

**한계.** 자기 자신을 반복한 meta-chunk는 false negative처럼 작동하여 성능을 낮춘다(§5.4). Artificial Needles는 무작위 distractor를 넣은 key-value 학습이 distractor가 관련 검색 문서일 때 MDQA를 개선하지 못했다고 보고했다([[artificial-needles-real-haystacks]] §4, Fig. 10). NExtLong의 검색 negative는 바로 그 경우를 겨냥하지만, 두 논문은 다른 모델과 다른 작업을 사용한다.

### §2.3 실제 문서 뒤에 붙이는 QA와 구조적 작업

**Nemotron Nano 2(Phase LC).** seed는 32K 토큰보다 긴 학술 문서다. 각 문서를 1,024토큰 청크로 나누고, 그중 10%를 Qwen-2.5-72B-Instruct에 보내 청크당 QA 한 쌍을 작성하게 한다. 모든 QA 쌍을 연결해 문서 뒤에 붙인다([[nemotron-nano-2]] §2.6). 문서-QA 데이터는 blend의 20%를 받고, 모든 Phase 3 source는 자신의 weight가 80%로 축소된다.

*계산 예시.* 64,000토큰 문서에는 ⌈64,000 / 1,024⌉ = 63개 청크가 있다(완전한 청크 62개와 512토큰 청크 하나). 63의 10%는 6.3이므로 QA 쌍 약 6개를 뒤에 붙인다(도출값). 표본 청크가 토큰 2,048–3,071을 덮는다면, 그 마지막 토큰에서 뒤에 붙은 QA block 시작까지 거리는 64,000 − 3,072 = 60,928토큰이다. weight 25%인 Phase 3 source는 25% × 0.8 = 20%가 된다.

*근거.* Nemotron-H 8B ablation의 RULER-128k는 합성 데이터 없이 256k에서 학습했을 때 70.19, 합성 데이터를 넣고 256k에서 학습했을 때 79.04, 합성 데이터를 넣고 512k에서 학습했을 때 81.04, 합성 데이터를 넣고 128k에서 학습했을 때 73.68이었다(Table 4). 단일 연구 결과다. 뒤에 붙은 QA에 loss masking을 했는지와 QA 검증은 보고되지 않았다.

**Qwen2.5-1M 구조적 작업.** 사전학습 데이터에 fill-in-the-middle, keyword·position 기반 문단 retrieval, 문단 순서 재배열을 더한다([[qwen-long-context-synth]] §3). 각 작업의 label은 문서 자체에서 도출되므로 generator model이 필요 없다. 비율과 ablation은 보고되지 않았다.

**CPT에서 Yi의 recitation QA.** Yi의 5B-token CPT mixture에는 “multi-document question-answering synthetic data, where ... the answer contains a recitation of the related paragraph before the answer”가 포함된다([[excerpts/yi-long-context-recitation]] §7.1). 비율과 ablation은 보고되지 않았다.

### §2.4 문서 경계: masking 대 separator

**정의.** *Cross-document masking*은 하나의 시퀀스에 pack한 문서 사이의 attention을 막는다. *Separator*는 문서 사이에 넣는 토큰이며, attention을 mask하지 않으면 모델은 separator를 넘어 attention할 수 있다.

**상충하는 결과.**

- ProLong은 문서 경계를 넘어서는 attention을 mask한다. 5B-token Llama-3-8B ablation에서 mask 사용 시 long 54.6 / short 65.5, mask 없음에서 long 53.6 / short 64.9를 보고했다([[prolong]] App. B.2, Table 20). SmolLM3도 문서 내부 masking을 쓴다([[excerpts/smollm3-long-context-merge]]).
- ChatQA 2는 문서를 "<s>"로 나누는 것이 Llama 3의 BOS/EOS token보다 효과적임을 발견했고, BOS/EOS가 “이전 텍스트 청크를 무시하라”는 신호를 모델에 준다고 가설을 세웠다(저자 해석). App. B는 수치 없이 2B-token 실행의 NIAH heatmap을 보인다([[chatqa-2]] §3.1).
- UltraLong은 "<s>"를 쓰고 cross-document mask를 명시적으로 적용하지 않는다. full attention을 유지한 채 separator를 제거하면 RULER <1M은 80.17에서 79.15, InfiniteBench는 26.25에서 22.75로 낮아진다([[ultralong-128k-to-4m]] Table 3).

**근거가 보이는 것과 보이지 않는 것.** ProLong은 mask와 no mask를 비교한다. UltraLong은 둘 다 mask 없이 separator와 no separator를 비교한다. 이 자료 모음에는 같은 데이터에서 “mask 없는 <s>”와 “mask”를 직접 비교한 연구가 없다. 두 설정은 합성이 하는 일도 다르다. mask를 쓰면, 무관한 문서들을 연결한 시퀀스에는 cross-document 종속성이 없으므로, Quest·NExtLong·Nemotron식 구성은 의존하는 조각들을 하나의 attention span 안에 넣어야 한다. NExtLong은 합성 시퀀스 전체에 next-token loss를 계산하고, 512K 실행은 각 합성 문서 내부에는 full attention을 적용하며, 옆에 pack한 더 짧은 문서 사이에서만 mask한다([[nextlong]] §3.2, App. C.2). **열린 질문:** mask하지 않은 separator가 무관한 컨텍스트에서 retrieval을 훈련하기 때문에 도움이 되는지, 아니면 낮은 비용으로 긴 시퀀스를 추가하기 때문에 도움이 되는지는 아직 분리되어 있지 않다.

**범용 모델에 대한 함의.** 합성 시퀀스를 연결로 만들 때 경계 처리는 구성 방식과 맞아야 한다. 무관한 문서 사이는 mask하고, 의도적으로 만든 종속성 내부는 mask하지 않는다.

## §3 SFT 측 장문서 생성기

SFT 생성기를 가르는 설계 선택은 *생성기가 무엇을 읽는가*와 *학생 모델이 무엇을 읽는가*다. 전체 긴 입력을 읽는 생성기는 자기 자신의 긴 컨텍스트 신뢰성에 의해 제한된다. 일부만 읽는 생성기는 문서 전체를 필요로 하는 질문을 쓸 수 없다.

### §3.1 청크에서 만든 QA를 전체 문서와 함께 학습하기

**Llama 3.** 사전학습 mixture의 문서를 8K 청크로 나눈다. 이전 Llama 3가 “무작위로 선택한 청크에 조건부로” QA 쌍을 작성하고, 학습할 때는 “전체 문서를 컨텍스트로 사용한다.” 표본은 16K, 32K, 64K, 128K 버킷으로 나눈다([[excerpts/llama-3-long-context-sft]] §4.3.4).

**Qwen2.5-1M.** Qwen2.5가 “무작위로 추출한 segment”에서 query를 작성한다. Qwen-Agent framework는 retrieval-augmented generation, 청크별 읽기, 단계별 추론을 이용해 전체 문서 위에서 response를 작성한다([[qwen-long-context-synth]] §4). response 생성 agent는 하나보다 많은 청크를 읽으므로, 답이 문서의 여러 부분을 통합할 수 있다.

**Yi의 recite-then-answer.** 여러 문서를 연결하고 하나 이상의 문단을 표본으로 뽑은 뒤, chat model이 그 문단에서 QA를 작성한다. 이어서 “답을 주기 전에 모델에게 원래 문단을 recite하거나 paraphrase하게 한다”([[excerpts/yi-long-context-recitation]] §7.1). 저자들은 이것이 “환각 행동을 억제한다”고 말하지만 ablation은 보고하지 않았다(저자 해석).

**한계.** 청크 하나에서 쓴 질문은 청크 하나에 의존한다. 목표까지의 거리는 멀지만 증거 개수는 하나이며, 질문이 청크의 단어를 반복할 수 있다(§1, 성질 4).

### §3.2 계층적 요약, 누락 파일 코드, 다중 문서 조립

**Llama 3 계층적 요약.** 가장 강한 8K 모델이 8K 청크를 요약하고, 이어서 그 요약들을 다시 요약한다. 학습에서는 전체 문서를 주고 “모든 중요한 세부 사항을 보존하면서” 요약하라는 prompt를 사용한다. 요약에서 생성한 QA 쌍은 “긴 문서 전체에 대한 전역적 이해”를 묻는다([[excerpts/llama-3-long-context-sft]] §4.3.4).

**Llama 3 누락 파일 코드 추론.** Python import statement에서 “적어도 다섯 개의 다른 파일이 참조하는” 파일을 고른다. 그 파일 하나를 repository에서 제거하고, 모델은 “어떤 파일이 누락 파일에 의존했는지 식별하고 필요한 누락 코드를 생성”해야 한다(§4.3.4). label인 제거된 파일과 그것의 importer는 생성기 판단이 아니라 repository에서 나온다.

**He et al.의 1M까지의 계층적 다중 문서 합성.** Qwen-2-72B-Instruct가 4K·12K 청크와 그 요약을 읽고 계층적 질문(전역 요약 → section → chunk) 및 다양한 질문을 작성한다([[hierarchical-million-token-synth]] §3.1). 책들을 하나의 multi-turn dialogue에 연결한다. 각 책 뒤에 계층적 QA 5개와 다양한 QA 5개, 즉 N1 = 5 계층적 쌍과 5 다양한 쌍을 더한다. 이어서 더 이른 책들에 관한 다양한 쌍 N2 = 9개를 더하고, 각 이전 책에 대해 확률 60%로 후속 질문 N3 = 3개를 추가한다(§3.2, Figure 3; §4.2는 N2와 N3 label을 서로 바꾸어 출력한다).

*계산 예시.* 1M 단계는 200 samples × 1M tokens를 사용한다. 네 단계의 토큰 수 합은 2,000 × 180K + 1,280 × 350K + 600 × 650K + 200 × 1M = 1.398B tokens로, 보고된 “약 1.4B”와 맞는다(card에서 도출한 검산). 4권 표본은 기댓값으로 권별 쌍 4 × 10 = 40개, revisit 쌍 3 × 9 = 27개, follow-up 쌍 (1 + 2 + 3) × 0.6 × 3 = 10.8개, 합계 약 78 QA turn을 가진다(Figure 3 숫자에서 도출).

*근거.* Llama-3.1-8B-Instruct에서 1M RULER은 1M 모델 62.95, zero-shot RoPE scaling 48.81이었다. InfiniteBench는 base 51.31에 대해 54.80, MMLU는 68.21에 대해 65.08이었다(Tables 1, 3, 9). 일곱 가지 180K composition에 걸친 InfiniteBench는 h-h-randomized의 56.63부터 hs-hs-hs-fixed의 59.45까지였다(Table 7). 생성 QA의 filtering 또는 verification은 기술하지 않았다. 단일 연구 결과다.

### §3.3 먼저 압축한 뒤 생성하기(CLIPPER)

**측정한 문제.** 책 전체(평균 90K tokens)를 읽은 Claude-3.5-Sonnet은 error가 있는 claim pair를 73.1% 만들었다. 52개의 사람이 주석한 claim 가운데 invalid 11.5%, 잘못된 chapter를 인용한 것 28.9%, 명시적 chapter reference나 quote가 포함된 것 15.4%, duplicate 17.3%였다([[clipper-compression-synth]] Table 1).

**방법.**

1. 압축한다. chapter별 outline은 chapter당 event 5–7개이며, 책 평균 90,437 tokens에 대해 평균 8,745 tokens, 즉 10.0% compression rate다. book summary는 약 618 tokens다.
2. 적어도 2개 chapter outline의 event에서 true/false claim pair를 만든다(book-level). chapter-level은 outline 하나와 summary 하나에서 만든다. 각각에 chain-of-thought를 붙인다.
3. deduplicate한 다음 GPT-4o가 outline을 기준으로 claim을 validate한다. duplicate라서 59.4%, invalid라서 2.4%를 제거했다.
4. 전체 책을 입력으로, chain-of-thought와 label을 target으로 학습한다.

**근거.** 주석한 CLIPPER claim 66개 중 83.3%는 error-free였고, naive method는 26.9%였다. claim당 비용은 $0.05 대 $0.07이었다. Llama-3.1-8B-Instruct에서 CLIPPER-test는 27.9% → 76.0%, NoCha는 16.5% → 32.2%, NarrativeQA는 47.7% → 49.0%, ∞Bench QA는 47.8% → 46.5%였다. 짧은 WritingPrompts story에서 나온 19K claims로 ProLong-512K-8B를 학습하면 CLIPPER-test는 63.0%, NarrativeQA는 31.0%였다(Table 2). 결과는 7B–8B 모델을 쓴 단일 연구다.

**한계.** claim은 outline model이 보존하기로 선택한 event에서 나온다. 저자들은 사람이 쓴 NoCha claim의 남은 성능 차이를 “이런 outline에는 보통 나타나지 않을 수 있는 low-level detail”에 돌린다(§4.1, 저자 해석).

### §3.4 사람 주석에 고정한 긴 예제

**ChatQA 2 summary 삽입.** NarrativeQA는 긴 source page의 사람이 쓴 summary와 그 summary에 근거한 사람이 쓴 QA 쌍을 제공한다. ChatQA 2는 summary를 “문장 구조가 깨지지 않도록 보장하면서 해당 긴 웹페이지 문서의 무작위 위치에 삽입”하여 32K–128K 예제를 만든다. 답은 사람이 쓴 것이다([[chatqa-2]] §3.2). NarrativeQA는 그 뒤 평가에서 제외한다. InfiniteBench En.Sum에서 ChatQA-2-70B는 16.08이고 Llama3.1-70B-Instruct는 30.94였다. 저자들은 이를 “SFT recipe에 summarization data가 없기 때문”이라고 돌린다(§5.2). 삽입한 summary는 source page를 다시 말하는 증거 span 하나이므로, 작업은 span 하나를 찾는 문제로 줄어든다.

**Context synthesis.** [[context-synthesis-short-to-long]]는 1.6k개의 사람이 쓴 instruction-answer 쌍을 유지하고 GPT-4o-mini가 빠진 context(약 2,000 words)를 쓰게 한 뒤, relevant context 1개와 irrelevant context 9개를 연결한다. UltraChat을 쓴 LLaMA3.1-8B에서 8-task LongBench 평균은 long data 없음 23.35, 같은 1.6k samples에서 instruction synthesis 25.39, context synthesis 38.57이었다(Table 3). instruction은 평가한 dataset의 training split에서 왔으므로 이는 in-distribution 근거다. unseen RULER multi-value NIAH는 91.57 → 96.54였다(Table 5).

### §3.5 짧은 컨텍스트 모델에서 긴 데이터를 bootstrap하는 agent workflow(SelfLong)

**방법** ([[bootstrap-context-length]] §3.1).

1. GPT-4o가 instruction을 작성한다. diversity를 위해 각 call 앞에 무작위 text chunk를 붙인다.
2. E5-mistral-7b가 약 10M개의 Fineweb-Edu 문서에서 문서를 검색한다.
3. 문서를 최대 4K tokens 청크로 나눈다. query-focused summarization agent가 instruction에 맞게 각 청크를 요약하고 irrelevant content를 버린다. summary들이 들어갈 때까지 이 과정을 재귀적으로 반복한다.
4. 모델은 summary에서 response를 작성한다.
5. 학습 입력은 instruction + raw retrieved documents이고, target은 response다. “중간 summary는 학습 중 사용하지 않는다.”

학생 모델은 최대 1M tokens를 한 번에 읽어, workflow가 여러 짧은 call로 수행한 일을 해야 한다. 긴 sample은 input과 output의 모든 token에 평균 loss를 적용하고, 짧은 sample은 output token에만 적용한다(§3.2).

**근거** (Table 3, RULER, SelfLong-8B). 1M에서 full data는 69.6, synthetic data를 뺀 조건은 63.5, short data only(≤4K)는 46.6, 학습 없이 RoPE θ만 조정한 조건은 48.1, GPT-4o 대신 Llama-3.1-8B-Instruct를 generator로 쓴 조건은 66.1이었다. 긴 sample에서 prompt token을 mask하면 1M에서 66.4였다. 짧은 컨텍스트 성능(Table 4)에서 IFEval은 8B 77.4 → 65.9, 3B 73.9 → 57.0으로 낮아졌다. 반면 8B의 GPQA는 26.8 → 30.4, MUSR은 36.9 → 41.5였다. 단일 연구 결과다.

### §3.6 IN2: integration 질문과 균형 잡힌 깊이

**방법** ([[in2-film]] §2.1).

1. C4 realnewslike text를 가져와 128-token segment로 나눈다.
2. fine-grained awareness에서는 GPT-4-Turbo가 segment 하나 s_i에 “매우 구체적인” QA 쌍을 쓴다. context는 무작위 segment r_j들과 함께 섞은 Shuffle(s_i, [r_j])다.
3. integration and reasoning에서는 GPT-4-Turbo가 한 text의 적어도 두 segment를 필요로 하는 질문을 쓴다. 필요한 segment와 무작위 segment를 함께 섞는다.
4. random segment 수에 rejection sampling을 적용하여 context 길이가 4K부터 32K까지 균등하게 분포하게 한다.
5. QA 쌍 약 10%는 원래의 짧은 text를 유지하고, OpenOrca instruction data를 더한다.

**수식.** 섞은 뒤 n개 segment로 된 context에서 segment i의 evidence depth는 d_i = (i − 1) / n이다. uniform shuffle이면 각 d_i는 {0, 1/n, ..., (n − 1)/n}의 n개 값에 균일하게 분포한다.

**계산 예시.** 16,000-token context에는 ⌊16,000 / 128⌋ = 125개 segment가 들어간다. 증거 segment 하나는 context의 128 / 16,000 = 0.8%다. 길이가 4K–32K에 균등하게 분포하면 평균 context 길이는 (4,000 + 32,000) / 2 = 18,000 tokens다(도출값).

**근거.** Mistral-7B-Instruct-v0.2 → FILM-7B(긴 fine-grained 1.1M + integration 300K, 짧은 QA 150K, instruction 200K)에서 VAL Probing 평균은 47.3 → 85.9, min-max position gap은 56.2 → 13.9였다(Table 1). LongBench 평균은 30.6 → 39.9였고 HotpotQA는 42.4 → 62.1이었다(Table 2). MMLU는 59.3 → 59.2, HellaSwag은 83.6 → 79.1, ARC-C는 55.9 → 52.5였다(Fig. 5). 결과는 7B, 32K의 단일 연구다.

**생성기 비교.**

| 생성기 | 생성기가 읽는 것 | 학생이 읽는 것 | 증거 개수 | label 출처 | 보고된 검증 |
|---|---|---|---|---|---|
| Llama 3 chunk QA | 8K 청크 하나 | 전체 문서 | 청크 1개 | 생성기 | 출력 없음 |
| Llama 3 missing file | repository 구조 | 파일 하나를 뺀 repository | importer 여러 개 + 제거된 파일 | repository | 구성상 label이 정확함 |
| Qwen2.5-1M | segment(query), agent를 통한 전체 문서(response) | 전체 문서 | 보고 안 됨 | agent | 보고 안 됨 |
| He et al. | 4K/12K 청크, summary | 연결한 책, multi-turn | summary, 청크 1개, 또는 청크 3개(multi-hop prompt) | 생성기 | 기술 안 됨 |
| CLIPPER | outline + summary | 전체 책 | event 2–3개, book-level은 ≥2 chapter | 생성기 | outline 기준 GPT-4o, 사람 표본 |
| ChatQA 2 | 없음(사람 데이터) | page + 삽입 summary | 삽입 span 1개 | 사람 | 사람이 작성함 |
| SelfLong | summarization agent를 통한 4K 청크 | instruction + raw retrieved documents | 검색 문서 여러 개 | 생성기 | 보고 안 됨 |
| IN2 | 128-token segment 1개 이상 | 섞은 4K–32K context | 1개 또는 ≥2개 | 생성기 | 10-gram decontamination만 |

## §4 장거리 종속성을 강제하는 방법

아래의 각 기법은 §1에서 정의한 성질 하나를 겨냥한다.

1. **문서 전반에 증거를 분산한다(증거 간격).** QwenLong-L1.5는 문서들에 걸쳐 knowledge-graph triplet을 추출하고, random walk와 BFS로 multi-hop path를 표본 추출한 뒤, “path node ...를 여러 문서에 드물게 분산해” 배치한다. 복잡도는 path 길이로 정한다([[qwenlong-l1-5]] §3.2). 이 모델의 학습 입력 평균 길이는 QwenLong-L1의 11,441 tokens에 비해 34,231 tokens다(Table 1).
2. **질문을 숨긴다(indirection).** LoongRL의 KeyChain은 UUID key-value 쌍을 삽입한다. 한 chain은 실제 질문으로 이어지고, 다른 chain들은 다른 item의 질문으로 이어진다. 다음은 prompt 일부다([[loongrl]] §3.1).

```text
Please read the following text.
<Document 0>
<original text> {"UUIDB-n": "distracting question"} <original text>
<Document 1>
{"UUIDA-1": "UUIDA-2"}
...
{"UUIDA-n": "correct question"}
...
In the context above, there is one correct question to answer.
The correct question can only be found by following the correct
consecutive chain of key:value pairs encoded with UUID strings
(e.g., f81d4fae-7dec-11d0-a765-00a0c91e6bf6), starting from
"starting UUIDA-1".
Find the correct question first, then answer it.
```

   질문 위치는 무작위이며 chain을 따라가기 전에는 알 수 없다. 따라서 모델은 질문이 끝부분에 있을 것이라고 가정할 수 없다. Qwen2.5-7B-Instruct에서 데이터 양을 같게 맞추면 KeyChain 데이터의 LongBench v1 multi-hop 평균은 72.4이고, 일반적인 long multi-hop QA는 66.2다(Table 4). 이것은 RL 데이터(GRPO, 16K context)이며 SFT에 사용했을 때의 결과는 시험하지 않았다. 단일 연구 결과다.
3. **깊이를 균형 있게 만든다.** IN2의 shuffle이 이 방법이다(§3.6). 사전학습 쪽 대응물은 NExtLong의 meta-chunk 배치다. 여기서는 head 위치가 tail 및 무작위 위치보다 좋았다(§2.2).
4. **질문–증거 중복을 낮춘다.** QwenLong-L1.5에서는 entity 난독화가 이에 해당한다(§1). LongMIT는 여러 single-hop 질문을 하나의 multi-hop 질문으로 합친다. Qwen2-72B의 일반 Self-Instruct를 쓰면 sample 중 multi-hop은 35% 미만이고 40%를 넘는 sample의 품질이 낮았다. multi-agent framework를 쓰면 multi-hop이면서 high-quality이고 non-duplicate인 sample이 85%를 넘었다([[excerpts/longmit-mimg]] §1).
5. **전체 컨텍스트에서 답을 계산하게 한다.** QwenLong-L1.5는 여러 문서의 table을 하나의 table로 모으고, 생성한 질문을 SQL로 번역한 후 SQL을 실행해 답을 얻는다([[qwenlong-l1-5]] §3.2). 답은 그 query가 닿는 모든 row에 의존한다.

**함의.** retrieval만 하는 데이터(증거 span 하나, 높은 중복)는 이미 포화된 needle 점수를 올린다. LongAlign-13B-64K는 NIAH에서 거의 완벽한 점수를 내지만 VAL Probing에서는 여전히 27.1점의 위치 격차가 있다([[in2-film]] Table 1). §2–§3에서 가장 큰 향상을 낸 생성기를 가르는 성질은 integration, indirection, 낮은 overlap이다.

## §5 사실이 없는 프로그램식 retrieval 데이터

**정의.** 프로그램식(programmatic) 데이터는 정확한 label을 지닌 코드로 생성하며, 세계에 관한 사실을 전혀 포함하지 않는다. 예를 들어 무작위 정수 key와 value를 가진 dictionary 목록을 생성한다([[artificial-needles-real-haystacks]] §2).

**메커니즘.**

1. dictionary 85개를 만들고 각 dictionary에는 key 3–4개를 둔다. key와 value는 무작위 3–4자리 정수이며, 각 prompt는 약 3,900 tokens다.
2. key 하나의 value와 그 key가 들어 있는 dictionary를 답하게 한다. answer template을 제공하므로 loss가 value에 집중된다(Fig. 3–4).
3. 이런 task 350개로 2 epochs, LR 5×10^-6, answer 부분에만 loss를 적용하여 fine-tuning한다(App. A.1).

**계산 예.** dictionary 85개 × 평균 key 3.5개 = 약 297개 key-value 쌍이다. 이를 3,900 tokens 안에 넣으므로 bracket과 dictionary label을 포함하면 쌍 하나당 약 13 tokens다(유도한 값). 정답 쌍은 prompt의 약 13 / 3,900 ≈ 0.3%다.

**근거** (Mistral-7B-Instruct-v0.1, Table 2, dataset별 학습 token 수는 대략 같음). Abstract는 baseline TriviaQA 하락을 2.33%에서 6.19%라고 적지만, Table 2는 −2.19에서 −6.33 points를 적는다. 여기서는 표의 값을 사용한다. TriviaQA 변화는 key-value 데이터가 +0.11, MultidocQA가 −2.43, IN2 데이터가 −2.19, NIAH-style 데이터가 −6.33이다. NQ-Open은 각각 +0.37, −2.91, −1.81, −6.73이다. MMLU 변화는 네 데이터셋 모두에서 ±0.6 안에 머문다. 실제 20-document MDQA task에서는 GPT-3.5 Turbo가 position 10에서 10.5% 개선했다(Abstract). 저자들은 다른 데이터셋의 지식 하락을 그 데이터셋의 factual content로 설명하며, 알려지지 않은 사실로 fine-tuning하면 hallucination을 조장한다는 선행 연구를 인용한다. 이는 해석이며, hallucination 자체를 직접 측정한 결과는 아니다.

**조건과 한계.** 약 4K tokens에서 시험했다(§3.5에 24K run 하나가 있음). key-value 모델은 relevant distractor가 있는 MDQA에서 개선되지 않았고(§4, Fig. 10), 일부 baseline은 MDQA 또는 FLenQA에서 key-value 데이터보다 좋았다(§3.4). 여기서 IN2의 하락은 FILM-7B와 backbone 및 token budget이 다르다. FILM-7B의 MMLU 변화는 0.1이었다([[in2-film]] Fig. 5).

**범용 모델에 대한 함의.** 사실이 없는 데이터는 position과 무관한 lookup을 가르칠 때 위험이 낮은 구성요소다. 그러나 integration이나 의미상 유사한 distractor에 대한 강건성은 가르치지 않는다. 따라서 §2와 §4의 구성을 대체하는 것이 아니라 보완한다.

## §6 장문 답변의 검증

장문서 합성에서 검증은 두 종류의 실패를 찾아내야 한다. 첫째는 예제가 틀린 경우(label error)이고, 둘째는 예제 자체는 맞지만 답을 내는 데 컨텍스트가 필요하지 않은 경우(dependency failure)다.

1. **Closed-book filter(종속성).** 문서를 제거한 뒤 모델에 질문한다. 문서 없이도 정확히 답한 예제는 버린다([[qwenlong-l1-5]] §3, "Knowledge Grounding Check"). 이 필터의 동기는 수치로 확인할 수 있다. 20개 문서 가운데 10번째 위치에서 GPT-3.5-Turbo의 정확도는 자기 자신의 closed-book 정확도보다 낮다([[lost-in-the-middle]] Table 6).
2. **Context-robustness check(label과 종속성).** 관련 없는 문서를 추가하고 pass@k가 0으로 떨어지는 예제는 버린다(QwenLong-L1.5 §3). *계산 예:* 삽입 뒤 어떤 item에 대한 검사 모델의 표본당 정확도가 0.1이고 k = 4라고 하자(k 값은 보고되지 않았으며 여기서는 예시다). 그러면 \(P(\mathrm{pass@4}=0)=0.9^4=0.656\)이므로 이 item은 65.6% 확률로 버려진다. 정확도가 0.5이면 그 확률은 \(0.5^4=0.0625\)다. 이 검사는 성능이 조금 나빠진 item이 아니라 완전히 무너지는 item을 제거한다. QwenLong-L1.5는 모든 필터, 중복 제거, decontamination 뒤 합성 예제 42.7K개 중 14.1K개(33.0%)를 남겼다(§3). 각 검사가 각각 얼마나 제거했는지는 보고하지 않았다.
3. **실행에 근거한 답(label).** 합친 table 위에서 실행하는 SQL(§4, 항목 5)과 Llama 3의 removed-file 구성(§3.2)은 구성 자체로 label을 고정한다. Llama 3가 생성된 missing code를 실행으로 검사하는지는 보고되지 않았다.
4. **인용 target(종속성).** LongCite는 답을 statement 단위로 나누고, 각 sentence마다 128-token chunk를 retrieval하며, sentence-level citation을 추출한다. 그리고 “답의 statement 중 20% 미만에 citation이 있는” instance는 버린다([[longcite]] §3.1). *계산 예:* 12개 statement 중 citation이 달린 statement가 2개면 \(2/12=16.7\%\)이므로 버리고, 3개면 25%이므로 남긴다. LongCite-9B에서는 data filtering으로 citation F1이 61.2에서 63.6으로 올랐고 correctness는 67.4 → 67.6이었다(Table 5). LongCite-8B의 citation F1은 72.0으로, GPT-4o의 65.6보다 높았다(Table 2).
5. **압축된 표현에 대한 검증(label).** CLIPPER는 outline을 기준으로 claim을 검증한다. NoCha에서는 full book을 볼 때 Claude의 검증 정확도가 40.3%였지만 outline을 볼 때는 98.6%였다(§2.2). 표본의 human annotation이 제시하는 오류율은 §3.3에 있다.
6. **데이터셋 수준의 종속성 진단.** long context를 넣은 경우와 넣지 않은 경우로 각각 fine-tuning한다. 두 결과의 차이가 작으면 그 데이터는 context 사용을 가르치지 못한 것이다. instruction-synthesis 데이터에서는 context를 추가해도 context-free tuning보다 개선되지 않았다([[context-synthesis-short-to-long]] §4.3, Figure 4).
7. **오류 감사.** CLIPPER는 naive claim 52개와 CLIPPER claim 66개를 사람이 annotation한 결과를 보고하며, 하나라도 오류가 있는 비율은 각각 73.1%와 16.7%였다([[clipper-compression-synth]] Table 1). LongMIT는 Self-Instruct일 때 multi-hop 비율이 35% 미만이고 poor-quality 비율이 40% 초과라고 보고하지만, label이 사람 할당인지 모델 할당인지는 밝히지 않는다([[excerpts/longmit-mimg]] §1). 이 장의 어떤 source도 사람이 수정해야 했던 생성 장문 예제의 비율은 보고하지 않았다.

## §7 길이 분포와 짧은 예제·긴 예제의 혼합

**정의.** *예제 비율(example share)*은 학습 예제 가운데 긴 예제가 차지하는 비율이고, *token 비율(token share)*은 학습 token 가운데 긴 예제가 차지하는 비율이다. 평균 길이 846-token인 예제와 비교하면 38K-token짜리 긴 예제 하나는 token을 약 45배 더 공급한다.

**계산 예(Llama 3 Table 7).** 장문 context 데이터는 SFT 예제의 0.11%이고, 예제당 평균 38,135.6 tokens다. 전체 데이터의 평균은 846.1 tokens다. 따라서 token 비율은

\[
0.0011 \times \frac{38{,}135.6}{846.1}
= \frac{41.95}{846.1}
\approx 0.050
\]

이다. 여기서 “0.1%”는 예제 비율이며, SFT token의 약 5%에 해당한다(도출값, [[excerpts/llama-3-long-context-sft]] Table 7). 그림은 여섯 행 전체에서 이 값을 다시 계산한다.

**측정된 일반 능력 변화.**

| 연구 | 설정 | 장문 데이터 | 일반 능력 변화 | 위치 |
|---|---|---|---|---|
| IN2 / FILM-7B | Mistral-7B, 4K–32K SFT | 예제의 약 80%가 긴 합성 QA | MMLU −0.1, HellaSwag −4.5, ARC-C −3.4, GSM8K +4.1 | [[in2-film]] Fig. 5 |
| He et al. | Llama-3.1-8B-Instruct → 1M | 1M-token 대화 200개 | MMLU 68.21 → 65.08, LongBench 48.11 → 46.42 | [[hierarchical-million-token-synth]] Tables 2–3 |
| SelfLong-8B-1M | Llama-3.1-8B → 1M | 약 8.3B 학습 tokens 안에 long-input 69K개(4.6B tokens) + long-output 10K개(77M tokens) | IFEval −11.5, MMLU Pro −2.2, GPQA +3.6 | [[bootstrap-context-length]] Table 4 |
| UltraLong-8B-1M | CPT 1B tokens + 짧은 예제만 사용한 SFT | SFT에는 없음 | 5-task 평균 61.45 → 62.47 | [[ultralong-128k-to-4m]] Table 2 |
| ChatQA-2-8B (new) | 1.6M-sample SFT 대 Llama3.1-8B-Instruct | 장문 SFT 단계 | MMLU 65.73 대 67.59, HumanEval 66.46 대 70.73(파이프라인이 다름) | [[chatqa-2]] Table 8 |
| Qwen2.5-14B-Instruct-1M | 128K 버전과 비교 | 최대 길이의 75% CPT + 장문 SFT | GPQA 45.5 → 39.9, IFEval 81.0 → 84.3 | [[qwen-long-context-synth]] Table 6 |
| Yi-34B-200K | 5B-token CPT | CPT에서 recitation QA | MMLU 76.32 → 75.56 | [[excerpts/yi-long-context-recitation]] Table 6 |

표의 하락 폭은 IN2의 MMLU 0.1부터 SelfLong-8B의 IFEval 11.5까지다. SelfLong-3B의 IFEval은 16.9 하락했다. 동시에 같은 실행에서 향상도 나타났다. IN2 GSM8K는 최대 +4.1, SelfLong GPQA는 +3.6, Qwen2.5-14B IFEval은 +3.3이다. 연구마다 base model, 학습 단계, benchmark가 다르므로 이 결과만으로 안전한 장문 데이터 비율 하나를 정할 수는 없다. 일곱 행 중 다섯 행은 적어도 하나의 benchmark에서 2점보다 큰 하락을 보인다. UltraLong-8B-1M의 가장 큰 단일 하락은 MMLU-Pro 44.33 → 42.44, 즉 1.89다(Table 2). Yi는 MMLU만 보고한다(연구 간 결과).

**짧은 예제만 사용하는 SFT: 서로 다른 결과.** Llama 3는 짧은 예제만 사용한 SFT가 long context 성능을 떨어뜨렸다고 보고했다([[excerpts/llama-3-long-context-sft]] §4.3.4). SelfLong은 짧은 예제(≤4K)만으로 학습하면 128K 이후에서 학습을 전혀 하지 않은 경우보다도 나빴다고 보고했다. 1M에서 각각 46.6과 48.1이다([[bootstrap-context-length]] Table 3). 반대로 ProLong은 synthetic long SFT의 token 비율을 1%부터 50%까지 바꾼 모든 경우가 0%보다 낮았다고 보고했고([[prolong]] Table 8), UltraLong은 짧은 예제만 사용한 SFT로도 1M 미만 RULER 79.1을 달성했다([[ultralong-128k-to-4m]] Table 1). 가설은 [[ch-32b]] §7에 기록되어 있으며, 결론은 여전히 **열린 질문(Open question)** 이다. 관찰 가능한 차이 하나는 ProLong과 UltraLong은 짧은 예제만 사용한 SFT 전에 자연 문서로 전용 장문 CPT 단계를 수행한 반면, SelfLong은 장문 SFT 단계에서 context를 확장했다는 점이다.

**Packing은 혼합과 상호작용한다.** Nemotron Nano 2는 “long-context ability를 보존하기 위해” SFT sample을 약 128K tokens까지 packed했고, tool-calling 성능 하락을 그 packing에 귀속시킨다. 다음 단계는 sample을 이어 붙이지 않고 학습했다([[nemotron-nano-2]] §3.2).

**모델 병합으로 회복.** SmolLM3는 RULER 하락의 원인을 reasoning mid-training과 24K-token APO 데이터로 추적했다. 그리고 \(\theta = 0.9\,\theta_{\text{APO-soup}} + 0.1\,\theta_{\text{mid-training}}\)로 병합하여 base model의 RULER를 128K까지 회복했다([[excerpts/smollm3-long-context-merge]]). 병합 전후의 수치는 인쇄되어 있지 않다. 모델 병합은 [[ch-30c]]에서 다룬다.

## §8 합성 장문서 세트의 long-context 검사

이 검사는 학습 **전에** 데이터에 적용하며, 평가 방법은 [[ch-32c]]와 연결된다.

1. **10개 bin의 증거 깊이 histogram.** evidence가 문서 앞 10%, 다음 10%, …, 마지막 10% 중 어디에 있는지 세고 각 bin의 비율을 보고한다. 균일하게 생성하는 generator라면 각 bin은 약 10%여야 한다. 평균 점수만으로는 evidence가 앞부분에 몰려 있는지 알 수 없다.
2. **질문–증거 unigram precision(§1).** 각 question token이 증거 span에 나타나는 비율을 계산하고, precision이 0.5를 넘는 예제의 비율을 보고한다. 이 비율이 높으면 문서 전체를 읽기보다 겹치는 단어로 evidence를 찾을 수 있는 질문이 많다는 뜻이다.
3. **Reference model의 closed-book accuracy.** 500-example sample에서 문서를 주지 않고 reference model에게 질문하여 정확도를 측정한다. 이 값이 높으면 generator가 만든 question이 long document의 evidence에 실제로 의존하지 않을 수 있다.
4. **Evidence-count distribution.** 답이 evidence span 하나에 의존하는지, 여러 span을 결합해야 하는지를 분포로 보고한다. span 하나만 쓰는 retrieval과 여러 span을 합치는 integration은 같은 long-context task가 아니다.
5. **학습에 사용하는 bucket별 length histogram.** Llama 3라면 16K, 32K, 64K, 128K bucket별 example 수 또는 token 수를 보고한다. 평균 길이 하나만 적으면 128K example이 실제로 얼마나 들어갔는지 보이지 않는다.
6. **Generator가 겨냥하지 않은 held-out 평가 family.** 예를 들어 claim 데이터라면 NoCha, QA 데이터라면 LongBench v2 또는 HELMET RAG, literal-match 의존성에는 NoLiMa를 사용한다. 생성기가 최적화한 task와 분리된 family에서 점수가 오르는지를 봐야 한다.

## 음성 샘플과 음성 피드백

장문서 합성은 과정 표준([[ch-31a]], [[ch-43a]])에서 정의한 ‘negative’의 네 가지 뜻 가운데 세 가지를 사용한다.

**1. 음성 항목의 출처.**
- distractor 문서와 hard-negative 청크: 무작위 segment(IN2), 검색된 유사 청크(NExtLong), 방해가 되는 UUID chain 및 다른 항목의 질문(LoongRL), 생성한 무관 context 9개(context synthesis), 목표 길이에 도달하도록 삽입한 무관 문서(QwenLong-L1.5)다. 구성 과정에서 label이 정해진다.
- minimal pair의 거짓 claim(CLIPPER): detail 하나를 바꾼 참 claim이며, generator가 False로 label하고 outline으로 대조한다.
- 거절한 생성 예제: closed-book으로 답할 수 있는 항목(QwenLong-L1.5), robustness가 붕괴하는 항목(QwenLong-L1.5), 인용된 statement가 20%보다 적은 항목(LongCite), duplicate와 invalid claim(CLIPPER: 각각 59.4%, 2.4%)이다. 이 filter들의 false-negative rate는 보고되지 않았다. 예외는 CLIPPER로, GPT-4o filter가 저자와 72개 pair 중 1개에서 불일치했다.
- 장문 context RL의 틀린 rollout(LoongRL, QwenLong-L1.5): rule-based reward 또는 judge reward가 label한다.

**2. 현재 방법이 이를 쓰는 방식.**
- distractor와 hard negative는 *내용으로서의 negative*(type 2)다. 입력 안에 놓이며 target에는 보통의 cross-entropy를 적용한다. NExtLong 같은 CPT 구성에서는 모든 token에서 loss를 계산하므로 hard-negative 청크도 보통 text처럼 예측한다. 이 청크는 dependency에 관해서만 negative다([[nextlong]] §3.2).
- target `False`와 chain-of-thought를 함께 둔 거짓 claim도 type 2다.
- 거절한 example은 *음의 한계 가치*(type 1)를 가지므로 버린다.
- 틀린 RL rollout은 negative advantage를 받는다. 이는 *gradient로서의 negative*(type 4)이며, 이 장의 어느 source도 개선분 가운데 이 negative가 차지한 비율을 측정하지 않는다.

**3. 기제.** type 2의 loss는 다음이다.

\[
L = − \Sigma_t \log p_θ(y_t \mid x_{doc}, x_{neg}, y_{<t})
\]

여기서 \(y_t\)는 t번째 target token, \(x_{doc}\)는 evidence를 담은 context, \(x_{neg}\)는 distractor token, \(y_{<t}\)는 그보다 앞선 target token이다. gradient는 distractor를 포함하는 context에서 올바른 target의 확률을 올린다. 어느 distractor의 확률을 내리는 항은 없다. \(x_{neg}\)에 attention을 주어도 \(L\)이 줄지 않기 때문에, \(x_{neg}\)를 무시하는 attention pattern은 학습 과정에서만 생긴다. RL의 type 4에서 sampled token \(y\)의 logit gradient는 \(∂ \log p_y / ∂ z_j = 1[j = y] − p_j\)다. negative advantage는 여기에 음수를 곱하므로 \(z_y\)를 내리고, \(p_j\)에 비례해 다른 token으로 probability mass를 옮긴다. 따라서 mass는 현재 가장 확률이 높은 대안들에 집중한다([[ch-43a]]).

**4. 수치가 있는 증거.**
- distractor의 이점: distractor 없이(needle만) NIAH SFT를 한 모델은 test에 방해 정보가 나타나면 저하했다. 반면 1K-essay distractor와 학습한 모델은 32K에서 90%를 넘게 유지했다([[context-synthesis-short-to-long]] §3.3). context \(n=10\)개(무관한 것 9개)를 연결하면 \(n=1\)의 37.67에 비해 38.57이었다(Table 11).
- hard negative 대 random: similarity로 고른 negative가 다섯 전략 중 가장 좋았다([[nextlong]] Fig. 7). random-distractor key-value 데이터는 relevant distractor가 있는 MDQA를 개선하지 못했다([[artificial-needles-real-haystacks]] Fig. 10).
- 거짓 claim은 더 어렵다. Qwen-CLIPPER는 book-level pair 1,000개 가운데 false claim 97개, true claim 37개에서 실패했다([[clipper-compression-synth]] §4.6).
- citation coverage에 따른 rejection은 citation F1을 +2.4 올렸다(§6).

**5. control.** 경쟁할 만큼 유사한 negative(검색 결과, 같은 keyword, 같은 task family)를 고르되, false negative가 되는 self-repeat와 near-duplicate는 제외한다([[nextlong]] §5.4). negative를 target 밖에 둔다. SFT에서는 distractor token의 loss를 mask한다. 거짓 claim에는 각각의 false claim을 true counterpart와 짝지어 CLIPPER의 accuracy metric처럼 pair를 함께 채점한다(§2.1).

**6. 진단.** distractor type별(none, random, retrieved) accuracy, position-gap metric(VAL Probing gap), true claim 대 false claim accuracy, 학습 전후 closed-book accuracy를 본다. RL에서는 advantage sign별 entropy와 큰 \(k\)의 pass@k를 나눠 본다([[ch-44a]]).

**7. 일반성에 미치는 효과.** distractor 학습은 RAG형 사용에서 필요한, retrieved지만 무관한 context에 대한 robustness를 만드는 기제다. 정답 target이 abstention인 answerable하지 않은 장문서 질문은 이 장의 어느 generator도 학습하지 않는다. [[agentinstruct]]는 ablation 없이 일부 짧은 reading-comprehension 질문을 answerable하지 않게 만든다(§2.1). evidence가 없을 때 distractor로 학습한 장문 context 모델이 과도하게 답하는지 또는 과도하게 거절하는지는 **열린 질문(Open question)** 이다.

## 레시피

값은 source card 또는 chapter excerpt에서 인용했다. “verified” 날짜는 card 또는 excerpt를 확인한 날짜다.

| Model (정확한 release) | Size | Stage | Setting | 값 | Source 위치 | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama 3 405B | 405B | long-context | stages; tokens | 8K → 128K의 6개 stage, 약 800B tokens | arXiv:2407.21783v3 §3.4.2 | verified 2026-09-15 | stage gate: short eval이 회복되고 NIAH를 해결함, stage별 ablation 없음 |
| Llama 3 (post-trained models) | 8B, 70B, 405B | SFT | long-context share | text의 “0.1%”; example의 0.11%, 평균 38,135.6 tokens(Table 7) | arXiv:2407.21783v3 §4.3.4, Table 7 | verified 2026-09-15 | “careful ablations”, 표는 인쇄되지 않음 |
| Llama 3 | 8B, 70B, 405B | SFT | chunk size; length buckets | 8K chunks; 16K, 32K, 64K, 128K | arXiv:2407.21783v3 §4.3.4 | verified 2026-09-15 | ablation 보고 없음 |
| Nemotron-Nano-12B-v2-Base | 12B | long-context | sequence length; LR; batch; tokens | 524,288; constant 4.5e-6; 12 sequences(약 6M tokens); 18.9B | arXiv:2508.14444v4 §2.6 | verified 2026-09-15 | Table 4(Nemotron-H 8B): 512k 81.04 대 256k 79.04 RULER-128k |
| Nemotron-Nano-12B-v2-Base | 12B | long-context | synthetic doc-QA construction; blend weight | 1,024-token chunks, 10% sample, Qwen-2.5-72B-Instruct, QA append; 20%(Phase 3는 80%로 scale) | arXiv:2508.14444v4 §2.6 | verified 2026-09-15 | Table 4(8B): synthetic 포함 256k에서 79.04, 미포함 70.19 |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | length mix | 현재 max length 75%, 더 짧은 길이 25% | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | ablation 보고 없음 |
| Llama-3-8B-NExtLong-128K | 8B | long-context | granularity; data; LR | \(s=2048\); 128K sample 32,000개(약 4B tokens); 4e-5 cosine | arXiv:2501.12766v2 §4.2, §5.3, Table 8 ([[nextlong]]) | verified 2026-09-14 | Fig. 6: granularity 5개 중 2048이 전체적으로 최상 |
| ProLong-8B | 8B | long-context | document masking | cross-document masking | arXiv:2410.02660 App. B.2 ([[prolong]]) | verified 2026-09-14 | Table 20: mask 없음의 53.6 / 64.9 대비 54.6 / 65.5 |
| UltraLong-8B-1M | 8B | long-context | tokens; separator; masking; LR | 1B tokens; `<s>`; cross-document mask 없음; 3e-5 | arXiv:2504.06214v1 §3.1, Table 5 | verified 2026-09-15 | Table 3: separator 제거 79.15, 유지 80.17 RULER <1M |
| UltraLong-8B-1M-Instruct | 8B | SFT | data; LR; batch | 짧은(<8K) example 100K; 5e-6; 128 | arXiv:2504.06214v1 §3.2 | verified 2026-09-15 | long-SFT ablation 없음 |
| Llama3-ChatQA-2-70B | 70B | long-context | CPT tokens | “10 billion tokens”와 “2000 steps (총 8B tokens)”를 모두 인쇄 | arXiv:2407.14482v3 §3.1 | conflict | App. B: `<s>` 대 BOS/EOS NIAH heatmap, 2B tokens |
| FILM-7B | 7B | long-context SFT | examples; lengths; LR; batch; epochs | 1.1M + long 300K, short QA 150K, instruction 200K; 4K–32K; 1e-6; 128; 1 | arXiv:2404.16811v2 §2.1–2.2 ([[in2-film]]) | verified 2026-09-14 | ablation 보고 없음 |
| Llama-3.1-8B → 1M (He et al.) | 8B | long-context SFT | samples × length; LR; epochs | 200 × 1M; 6e-5; 1 | arXiv:2504.12637v1 §4.2 ([[hierarchical-million-token-synth]]) | verified 2026-09-14 | sample count ablation 없음 |
| SelfLong-8B-1M | 8B | long-context SFT | synthetic data | long-input 69K(4.6B tokens) + long-output 10K(77M tokens); 총 약 8.3B tokens | arXiv:2412.18860v2 §4.1 | verified 2026-09-15 | Table 3: synthetic 없음 1M에서 63.5, 포함 69.6 |
| Llama-3.1-8B-CLIPPER | 8B | SFT | claims; LR; batch; epochs | training claim 16K; 1e-6; 16; 1 | arXiv:2502.14854v2 §3.1 | verified 2026-09-15 | dev-set에서 LR {1e-5, 1e-6, 1e-7}, batch {8, 16, 32, 64} sweep |
| LongCite-9B / 8B | 9B, 8B | SFT | data; filter; LR; steps | LongCite 44,600 + ShareGPT 76K; cited <20% discard; 1e-5; 4,000 steps, batch 8 | arXiv:2409.02897v3 §3.1, §3.3, §4.1 | verified 2026-09-15 | Table 5: filter 63.6, 미사용 61.2 citation F1 |
| Mistral-7B-Instruct-v0.1 (key-value) | 7B | SFT | tasks; epochs; LR; batch | 350; 2; 5e-6; 16 | arXiv:2406.19292v2 App. A.1 | verified 2026-09-15 | ablation 보고 없음 |
| QwenLong-L1.5-30B-A3B | 30B-A3B | RL | data yield; input length | 42.7K → 14.1K; 평균 34,231 tokens | arXiv:2512.12967v1 §3, Table 1 | verified 2026-09-15 | 두 check의 ablation 없음 |
| LoongRL-7B | 7B | RL | KeyChain items; context | 21,024개 중 7,500개; 약 16K tokens | arXiv:2510.19363 Table 1 ([[loongrl]]) | verified 2026-09-14 | Table 4: plain multi-hop 66.2 대비 72.4 |
| SmolLM3-3B | 3B | merge | weights | 0.9 APO soup + 0.1 mid-training checkpoint | huggingface.co/blog/smollm3, “Model Merging” | verified 2026-09-15 | “achieved the best performance”, 값은 인쇄되지 않음 |

**작은 범용 run의 시작점.** 합성 CPT 데이터로 8K에서 128K로 확장하는 8B model이라면, 표에서 NExtLong 128K configuration이 가장 완전하게 문서화된 합성 CPT recipe다. granularity 2048, top-k 유사 negative, 처음에 meta-chunk, 128K에서 약 4B tokens, cosine decay 및 warmup 200 iteration이 있는 LR 4e-5, RoPE base 500,000 → 200,000,000이다. Meta-Llama-3-8B base에서 H100 64장으로 15시간 실행했다. SFT는 UltraLong-8B처럼 short-only SFT(짧은 example 100K, LR 5e-6, batch 128)에서 시작하고, SFT 뒤 long evaluation이 저하할 때만 합성 long share를 더한다. Llama 3의 example 0.11%(약 token 5%) share는 8B–405B 모델에 대해 보고된 유일한 production 값이지만 인쇄된 ablation은 없다. long SFT 데이터를 만들면 closed-book filter와 citation 또는 grounding filter를 적용한다(LongCite의 20% threshold는 9B와 8B model에서 실행했다). 이 setting들을 한 run에서 함께 test한 연구는 없다.

## 일반화 관점

**(a) breadth를 늘리는 요소.**
- 장문서가 있는 소수 domain만 쓰는 대신 다양한 domain에 걸쳐 dependency를 구성한다. 동일 token 수에서 Quest data는 기존 long document보다 LongBench 22.06 대 21.11이었다(1.4B, [[quest-query-centric-synthesis]] Table 5).
- retrieval만 하지 않고 integration과 indirection을 넣는다. KeyChain은 72.4 대 66.2였고([[loongrl]] Table 4), long data에 integration question 300K개가 들어간 IN2 training은 multi-hop LongBench task를 최고 22.7점 올렸다(2WikiMQA 24.3 → 47.0, [[in2-film]] Table 2).
- 위치와 무관한 lookup을 위한 fact-free retrieval data를 쓴다. GPT-3.5 Turbo에서 position 10의 MDQA는 +10.5%였고 Mistral-7B의 TriviaQA는 변하지 않았다([[artificial-needles-real-haystacks]] Table 2).
- 합성 task 밖으로 transfer한다. QwenLong-L1.5는 data target이 아니었던 AIME25를 82.81 → 86.46, corpus에서 simulated dialogue가 작은 부분이었음에도 LongMemEval을 60.80 → 76.40으로 올린다([[qwenlong-l1-5]] Table 8). CLIPPER는 Qwen2.5-7B의 MuSR을 41.2% → 45.2%로 올린다([[clipper-compression-synth]] Table 2). QwenLong-L1.5 비교에서는 RL algorithm과 data가 함께 바뀌었으므로 data만의 효과는 분리되지 않았다([[qwenlong-l1-5]] §4.1).

**(b) narrowing 또는 forgetting을 일으키는 요소.**
- 새 사실을 담은 long data: Mistral-7B에서 TriviaQA −2.19 ~ −6.33([[artificial-needles-real-haystacks]] Table 2).
- long SFT 뒤 instruction-following 손실: IFEval −11.5(8B), −16.9(3B)([[bootstrap-context-length]] Table 4).
- 1M 확장 시 knowledge와 reasoning 손실: MMLU −3.13([[hierarchical-million-token-synth]] Table 3), Qwen2.5-14B-1M의 GPQA −5.6([[qwen-long-context-synth]] Table 6).
- task-format narrowing: summarization data 없는 ChatQA 2 long SFT의 En.Sum은 30.94 대신 16.08이었다([[chatqa-2]] Table 2). Llama-CLIPPER는 ∞Bench QA에서 1.3점을 잃는다([[clipper-compression-synth]] Table 2).
- 다른 skill의 packing-induced regression: 128K packing 뒤 tool calling이 저하했다([[nemotron-nano-2]] §3.2).

**(c) 이 stage에서 측정하는 방법.**
- 각 long stage 전후에 knowledge QA(TriviaQA, NQ-Open), instruction following(IFEval), code(HumanEval), math를 포함한 short suite를 돌린다. 위 손실은 연구마다 서로 다른 benchmark에서 나타난다.
- generator가 target으로 삼지 않은 long evaluation family와 generator 문서에서 분리된 test document를 쓴다(CLIPPER는 book별 split, ChatQA 2는 NarrativeQA 제거). [[context-synthesis-short-to-long]]은 evaluation dataset의 training split에서 training instruction을 가져오므로 main table은 in-distribution이다.
- literal-match control로 NoLiMa식 low-overlap probe를 쓴다([[nolima]] Table 3).
- average에 position-gap metric을 더한다([[in2-film]] Table 1).
- measurement error도 본다. UltraLong은 baseline을 직접 평가하고 ProLong-512k-Instruct HumanEval 10.36을 보고한다([[ultralong-128k-to-4m]] Table 2). paper마다 prompt format과 decoding이 다르면 baseline score가 움직일 수 있으므로, short benchmark의 논문 간 비교에는 evaluation configuration이 필요하다([[ch-32c]]).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| word overlap가 높은 청크에서 QA 생성 | NIAH는 100에 가깝지만 low-overlap probe가 하락 | unigram precision histogram(§1), NoLiMa식 eval |
| 문서 없이도 답할 수 있는 질문을 유지 | knowledge-heavy QA에서는 long-context gain이 보이나 문서를 무관 text로 바꾸면 사라짐 | sample에 closed-book filter(§6) |
| generator가 전체 long input을 읽음 | human audit에서 높은 error 또는 misattribution rate | error type별 50개 이상 audit(CLIPPER Table 1) |
| 구성한 dependency에 cross-document masking 적용 | 합성 CPT가 concatenation보다 이득 없음 | attention mask boundary와 construction boundary 대조(§2.4) |
| long share를 example 수만으로 계산 | “0.1%” long data가 token 기준 약 5%로 드러남 | 평균 길이에서 token share 계산(§7) |
| random text만 distractor로 사용 | random distractor MDQA는 향상하지만 retrieved distractor에는 향상 없음 | retrieved distractor로 평가([[artificial-needles-real-haystacks]] Fig. 10) |
| self-repeat 또는 near-duplicate negative | top-k retrieval보다 낮은 long score | meta-document에 대해 negative deduplicate([[nextlong]] §5.4) |
| short-suite regression check 없음 | release 뒤에야 IFEval, TriviaQA, HumanEval 하락을 발견 | 모든 long stage에서 전후 short suite(§7) |
| training data를 생성한 dataset에서 평가 | independent suite보다 높은 score | source document 또는 dataset별 split(ChatQA 2 §3.2) |
| short tool-call SFT를 long sequence로 pack | tool-calling accuracy 하락 | stage별 tool eval, concatenation 없이 tool data 학습([[nemotron-nano-2]] §3.2) |

## 이해 점검

1. Llama 3는 한 8K chunk에서 QA를 쓰지만 whole document로 학습한다. 이때 어떤 dependency가 만들어지는지, 그리고 §1의 다섯 properties 중 어느 것이 통제되지 않는지 설명하라.
2. NExtLong은 meta-document뿐 아니라 hard-negative chunk에서도 loss를 계산한다. 그럼에도 negative가 long-range modeling을 강제하는 이유와, self-repeated chunk를 negative로 쓰면 무엇이 달라지는지 설명하라.
3. ProLong의 mask ablation과 UltraLong의 separator ablation은 모두 자기 설계를 지지한다. 두 결과가 모순되지 않는 이유와, 이 질문을 해결할 experiment를 설명하라.
4. Key-value retrieval data는 TriviaQA를 그대로 두었지만 NIAH식 data는 6.33점 낮췄다. 저자들이 제시한 causal explanation을 말하고, 그것이 interpretation인 이유 및 시험할 measurement를 설명하라.
5. QwenLong-L1.5의 robustness check는 pass@k가 0으로 떨어질 때만 item을 버린다. §6의 worked example을 이용해 어떤 weak item이 살아남는지, 이를 잡으려면 check를 어떻게 바꿀지 설명하라.
6. CLIPPER의 generator는 book 대신 outline을 읽지만, 학습된 model은 book을 읽는다. 이것이 data quality를 높이는 이유와 생성하지 못하는 claim class를 설명하라.
7. Llama 3 long SFT share는 example의 0.11%다. forgetting에는 왜 token share가 관련량인지 설명하고 Table 7에서 이를 계산하라.
8. SelfLong의 IFEval은 11.5점 하락했지만 GPQA는 3.6점 올랐다. §3.5의 data mixture에 뿌리를 둔 원인 두 가지와 SmolLM3 merge 결과가 시사하는 mitigation을 제안하라.

## 연결

- 이전: [[ch-29]] — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification (여기서는 filter와 verification stage를 long input에 적용한다).
- 다음: [[ch-29b]] — Long-Conversation and Accumulating-Context Synthesis.
- 의존: [[ch-32b]] — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (CPT stage, RoPE, masking, §7의 SFT-share 논쟁).
- 의존: [[ch-32c]] — Claimed versus Effective Context Length and Long-Context Evaluation (RULER, HELMET, NoLiMa).
- 관련: [[ch-28]] — Long-Context Data Synthesis and Synthetic Evaluation Task Families; [[ch-30c]] — Weight Averaging and Model Merging for Generalist Models; [[ch-31a]], [[ch-43a]] — SFT의 negative와 gradient로서의 negative; [[ch-44a]] — Length in RL: Overlong Responses, Length Control, and Long-Context RL.

## 출처

- [[llama-3]], [[excerpts/llama-3-long-context-sft]] — §3.4.2 stage gate와 800B tokens; §4.3.4 chunk QA, hierarchical summarization, missing-file code, 0.1% share; Table 7 token statistics.
- [[long-context-llama3]] — 같은 material의 오래된 library card. 아직 재검증하지 않았고 report에는 stage별 token count와 data ratio가 없다. 이 장의 number는 excerpt에서 왔다.
- [[qwen-long-context-synth]] — structural CPT task, 75/25 length mix, agent-written SFT response, Table 6 short-benchmark 변화.
- [[yi]], [[excerpts/yi-long-context-recitation]] — CPT와 SFT의 recitation QA, 5B-token CPT, Table 6 MMLU(card는 아직 재검증하지 않았고 number는 excerpt에서 왔다).
- [[chatqa-2]] — `<s>` separator, NarrativeQA summary insertion, InfiniteBench, Table 8(chapter excerpt이며 library card는 아직 없음).
- [[in2-film]] — IN2 construction, depth balance, VAL Probing gap, LongBench, short-task 변화.
- [[hierarchical-million-token-synth]] — 1M까지의 hierarchical multi-document dialogue, token budget, MMLU loss.
- [[nemotron-nano-2]] — Phase LC appended doc-QA, 20% blend, Table 4 ablation, SFT packing regression(chapter excerpt).
- [[quest-query-centric-synthesis]] — query-centric grouping, Tables 1–3, 5(chapter excerpt).
- [[nextlong]] — hard-negative interleaving, \(k\) formula, Tables 1, 4, 5.
- [[ultralong-128k-to-4m]] — masking 없는 separator, short-only SFT, Tables 1–3(chapter excerpt).
- [[prolong]] — document-masking ablation, synthetic long SFT share ablation, long-document availability.
- [[qwenlong-l1-5]] — knowledge-graph multi-hop, SQL-grounded answer, grounding/robustness check, Tables 1, 7, 8(chapter excerpt).
- [[loongrl]] — KeyChain construction, Table 4 비교.
- [[artificial-needles-real-haystacks]] — fact-free key-value data, Table 2 knowledge-benchmark 비교, relevant-distractor limit(chapter excerpt).
- [[longmit]], [[excerpts/longmit-mimg]] — Self-Instruct와 MIMG 아래의 multi-hop share(card는 generic dataset class를 설명하고 number는 excerpt에서 왔다).
- [[context-synthesis-short-to-long]] — context synthesis, distractor pilot, context-free diagnostic.
- [[clipper-compression-synth]] — compression-first claim generation, human error audit, Table 2 transfer(chapter excerpt).
- [[bootstrap-context-length]] — agent-workflow synthesis, Table 3 ablation, Table 4 IFEval drop(chapter excerpt).
- [[smollm-3]], [[excerpts/smollm3-long-context-merge]] — RULER regression, 0.9/0.1 merge(card는 아직 재검증하지 않았고 quote는 excerpt에서 왔다).
- [[longcite]] — CoF citation pipeline, 20% filter, Tables 2, 3, 5(chapter excerpt).
- [[nolima]] — question–needle lexical overlap 및 효과.
- [[lost-in-the-middle]] — position dependence, closed-book 비교.
- [[agentinstruct]] — negative content로서 answerable하지 않은 reading question(short context).
