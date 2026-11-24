<!-- chapter: ch-35a
     track: sft
     kind: content
     title: 실전 증류 B: 프롬프트 선택, 교사 샘플링, 품질 필터
     deps: [ch-35, ch-22]
     sources: [[open-thoughts]], [[openmathreasoning]], [[opencodereasoning]], [[qwen-3]], [[light-r1]], [[limo]], [[s1]], [[magistral]], [[magistral-recipe]], [[glm-4-5]], [[llama-4]], [[kimi-k1-5]], [[acereason-nemotron-1-1]], [[distillation-source-matters]], [[small-models-learnability-gap]], [[capacity-gap-law-distillation]], [[deepdistill]], [[naturalthoughts]], [[phi-4]], [[bespoke-stratos]], [[openr1-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[deepseek-v3]], [[llama-nemotron]], [[hermes-4]]
     figures: figures/difficulty-filter-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 35a — 실전 증류 B: 프롬프트 선택, 교사 샘플링, 품질 필터

> **핵심 통찰.** 통제된 ablation을 갖춘 공개 증류 연구에서 질문 출처와 교사는 학생 점수를 13~17점 바꾸었다. Qwen2.5-7B-Instruct 학생에서는 최선과 최악의 코드 출처 사이가 17.2점이었고([[open-thoughts]] Table 3), Qwen2.5-32B 학생에서는 동일한 189만 개 질의에 세 교사를 각각 사용했을 때 AIME2024가 13.4점 차이 났다([[distillation-source-matters]] Table 1). 답변 필터링은 7B~14B 학생에서 모든 답을 유지하는 방법을 이기지 못했다. OpenThoughts 수학 세트에서 같은 크기인 31,600개 답변을 무작위로 고르면 어떤 답변 필터보다 높은 수학 평균을 냈다(GPT 검증 61.4 대비 64.8, [[open-thoughts]] Table 7). 또 151,251개의 단위 테스트를 모두 실패한 해답은 같은 수의 통과 해답보다 더 좋은 14B 학생을 만들었다(LiveCodeBench 52.3 대 47.0). 저자들은 실패 해답이 더 어려운 질문에서 나왔기 때문이라고 추적한다([[opencodereasoning]] Table 3). 점수가 더 높은 모델이 항상 더 좋은 교사는 아니다. 수학 질문 출처를 쓴 7B 학생에서 QwQ-32B trace는 DeepSeek-R1 trace보다 높았다(평균 44.2 대 41.6, [[open-thoughts]] Table 8). 이 파이프라인의 모든 난이도 라벨은 이름이 명시된 기준 모델의 통계량이므로, 필터를 재현하려면 그 모델, 샘플 수, 그리고 유지한 범위를 함께 알아야 한다.
>
> **지침.** 프롬프트를 고를 때는 기준 모델, 프롬프트당 샘플 수, 정답 샘플의 유지 범위를 기록한다. 공개된 규칙인 “4번 안에 풀면 버린다”, “32개 중 1~3개를 유지한다”, “8개 중 통과율이 0.75 이상이면 버린다”는 서로 다른 프롬프트를 남기기 때문이다([[limo]] §3.1.1; [[llama-nemotron]] §5.1). 아직 더 많은 서로 다른 프롬프트를 수집할 수 있다면 프롬프트당 응답 수를 늘리기 전에 프롬프트 수를 늘린다. AceReason-Nemotron 1.1 회귀에서 프롬프트 계수는 4.831이고 프롬프트당 응답 수 계수는 2.635이기 때문이다([[acereason-nemotron-1-1]] §4.4.2). 풀이 가능한 프롬프트 풀이 고갈됐으면 프롬프트당 응답을 더 샘플링하고([[open-thoughts]] §4.4의 16개), epoch plateau까지 학습한다([[acereason-nemotron-1-1]] Fig. 6의 epoch 5~6). 교사를 고를 때는 benchmark 점수로 순위를 매기지 말고 후보별 trace로 작은 학생을 학습한다. 낮은 점수의 QwQ-32B가 7B 학생에는 더 좋은 교사였기 때문이다([[open-thoughts]] Table 8). 학생이 3B parameter 이하이면 long-CoT 데이터에 short-CoT trace를 섞는다. Qwen2.5-3B는 long-CoT SFT 뒤 40.3, short-CoT SFT 뒤 43.4, long:short=1:4 혼합 뒤 45.9를 기록했다([[small-models-learnability-gap]] Tables 1, 3). trace가 cold start 또는 RL 프롬프트 세트의 seed라면 정답을 엄격히 검증한다. 잘못된 기준 답은 잘못된 reward를 만들기 때문이다(해석). 그 외 대규모 SFT 데이터에서는 답변 필터를 채택하기 전에 크기와 난이도를 맞춘 ablation을 수행한다.

## 범용 모델에서 이 장이 중요한 이유

증류 SFT는 더 강한 모델인 교사(teacher)가 작성한 텍스트로 학생 모델을 학습한다. [[ch-35]]는 연구소들이 교사 데이터를 어디에 두는지, 즉 교사 logits를 이용한 사전학습, mid-training, cold start, RL 뒤 rejection-sampling SFT, 소형 모델 증류, on-policy distillation을 정리했다. 이 장은 하나의 증류 데이터셋 안에서 내리는 결정을 다룬다. 어떤 프롬프트를 보낼지, 교사를 어떤 프롬프트와 샘플링 설정으로 호출할지, 어떤 출력을 남길지, 데이터를 어떻게 decontamination할지, 학생에게 어떤 SFT 설정을 적용할지가 대상이다.

파이프라인 위치는 사전학습과 mid-training 뒤, preference optimization과 RL 전의 SFT 및 distillation SFT다. Kimi k1.5, Magistral, LN-Ultra의 RL 프롬프트 규칙 세 가지도 같은 난이도 통계를 사용하므로 포함했다. RL 프롬프트 curriculum은 [[ch-16]]의 주제다. 품질·다양성 점수에 따른 데이터 선택은 [[ch-22]]에서 다룬다.

이 결정들은 일반성에 다음 세 가지 측정 가능한 방식으로 영향을 준다.

1. **도메인 비중.** 공개 증류 세트는 검증 가능한 수학과 코드가 대부분이다. Llama-Nemotron SFT 데이터 33,011,757개 샘플에서 수학은 66.8%, 코드는 30.6%다. 반면 chat은 0.12%, instruction following은 0.17%, safety는 0.10%다([[llama-nemotron]] Table 2).
2. **목표 밖 능력의 손실.** 대부분 수학인 약 3천 개 질문으로 SFT stage 2를 수행한 뒤 Light-R1-32B의 GPQA는 64.3에서 60.6으로 떨어졌다([[light-r1]] Table 2). LN-Super-SFT는 reasoning-on 모드 IFEval에서 81.9점(reasoning-off에서는 83.0점)이었는데, 기반 모델인 Llama-3.3-70B-Instruct는 92.1점이었다. 보고서는 “reasoning-focused SFT causes a noticeable drop in IFEval scores”라고 적는다([[llama-nemotron]] Table 4). OpenThinker3-7B의 HarmBench harmfulness rate는 safety 데이터를 전혀 쓰지 않은 상태에서 Qwen2.5-7B-Instruct 기반의 14.5에서 55.5로 상승했다([[open-thoughts]] App. L).
3. **측정 공백.** [[distillation-source-matters]]는 개수 기준 41.8%가 일반 chat인 질의 세트로 학습하지만, 수학과 코드 결과만 보고한다(Fig. 2; Tables 1-2).

## §1 질문 출처

**질문 출처(question source)**는 forum archive나 competition collection처럼 증류용 프롬프트를 뽑는 프롬프트 풀이다. 출처는 같은 데이터 크기에서도 가르치는 양이 다르다. 코드 출처별로 DeepSeek-R1 답변을 붙인 31,600개 예제 세트를 만들어 Qwen2.5-7B-Instruct 학생을 학습했을 때, 8개 development benchmark 전체 평균에서 최선과 최악 코드 출처의 차이는 17.2점이었다. StackExchange-CodeGolf는 38.8점이었고 최악 출처는 21.6점이었다([[open-thoughts]] §4.1, Table 3).

OpenThoughts의 출처 ablation은 세 단계로 진행된다([[open-thoughts]] §4).

1. 후보 출처별로 31,600개 예제 세트 하나를 만든다. 후보는 코드 27개, 수학 21개, 과학 14개 출처다.
2. 각 세트로 Qwen2.5-7B-Instruct를 fine-tune하고 8개 development benchmark 평균을 낸다.
3. 가장 좋은 선택지를 남긴 뒤, 혼합, 질문 필터링, 질문당 답변 수, 답변 필터링, 교사 선택이라는 다음 파이프라인 단계로 넘어간다.

상위 코드 출처 2개를 섞은 평균은 41.3이었고, 상위 16개를 섞은 평균은 36.4였다(Table 4). 세 도메인 모두 도메인당 출처를 최대 두 개만 쓰는 것이 가장 좋았다(§4.2). 선택된 출처는 수학의 OpenMath-2-Math, 코드의 StackExchange CodeGolf와 OpenCodeReasoning, 과학의 StackExchange Physics와 OrganicChemistry-PDFs다. 상태: 결과(단일 연구)이며, 학생 모델 하나와 reasoning benchmark 평균으로 고른 결과다.

**포럼에서 채굴한 수학.** OpenMathReasoning은 Art of Problem Solving(AoPS) forum에서 풀을 만들며, Qwen2.5-32B-Instruct가 모든 단계를 수행한다([[openmathreasoning]] §2.1, Table 1).

1. 첫 게시물에서 문제를 추출한다. discussion 620K개에서 문제 580K개를 얻는다.
2. 각 문제를 증명 문제인지, 객관식인지, 이진형(yes/no)인지, 유효한지로 분류한다. 객관식·이진형·무효 문제를 제거해 550K개를 남긴다.
3. 증명 문제는 답 기반 질문으로 변환하고, 나머지에서는 최종 답을 추출한다.
4. LLM이 benchmark 항목과 가깝다고 판단한 문제를 제거해 540K개를 남긴다.

최종 풀에는 변환된 증명 260K개, 답을 추출한 문제 190K개, 답이 없는 문제 90K개가 있다(Table 2). “Middle School Math” forum은 “우리의 예비 실험에서 너무 초등적이고 학습에 도움이 되지 않았다”고 하여 제외했지만, 수치는 제시하지 않았다. Llama-Nemotron은 이 파이프라인을 재사용하되 증명 문제를 변환하지 않고 제거한다([[llama-nemotron]] §3.1.1). 따라서 같은 forum도 서로 다른 문제 풀을 만든다.

**경쟁 프로그래밍 집계.** OpenCodeReasoning은 TACO, APPS, CodeContests, OpenR1 CodeForces 세트를 합치고 exact duplicate를 제거해, 10개 platform에서 질문 28,904개를 남긴다. CodeForces는 질문 10,403개와 총 736,712개 샘플 중 388,405개를 기여한다([[opencodereasoning]] §2.1, Table 1). 샘플 수를 25k에서 736k로 늘려도 plateau가 나타나지 않았고, 가장 큰 이득은 “unique, varied, and difficult questions”를 추가했을 때 나왔다(§2.4, Fig. 3). 상태: 서로 다른 프롬프트가 더 많을수록 좋다는 방향은 [[acereason-nemotron-1-1]] §4.4.2가 7B 수학과 코드에서 재현했다.

**연구소 기준점.** DeepSeek-R1의 804,745-sample SFT 세트는 수학 395,285개, 코드 211,129개, STEM 10,124개, 논리 10,395개, 일반 177,812개로 구성된다. 평균 길이는 일반의 1,419.8 token부터 코드의 7,435.7 token까지다([[deepseek-r1-recipe]], v2 B.3.3 Table 5). 일반 샘플은 세트의 22.1%다(177,812 / 804,745, 계산값). DeepSeek-R1-Distill 학생 여섯 종 모두 이 세트에서 학습했다(B.4.3).

범용 모델에 대한 함의: 수학·코드·과학 benchmark 평균으로 출처를 선택하면 바로 그 도메인에 맞춰 최적화한다. OpenThoughts 저자도 전체 평균으로 선택하는 일은 도메인 간 전이를 가정한다고 언급한다(§6). 일반 학생을 위한 출처 ablation에는 선택 점수 안에 held-out chat, instruction-following, safety suite를 포함해야 한다.

## §2 이름을 밝힌 기준 모델에 상대적인 난이도

**난이도 라벨(difficulty label)**은 프롬프트에서 기준 모델(reference model)이 낸 출력의 통계량이다. n개 샘플에서의 통과율, chain of thought(CoT) 없이 답했는지, 응답 길이, 두 모델의 답 일치 여부가 예다. 이는 쉬운 프롬프트가 어려운 프롬프트보다 덜 가르칠 수 있다는 문제를 다룬다. LIMO는 DeepSeek-R1 풀이를 붙인 난이도가 다른 500문제 세 세트(MATH level 1–2, MATH level 3–5, AIME)로 Qwen2.5-32B-Instruct를 학습했다. 저자들은 “문제 선택만 바꾸어도 AIME2024 정확도가 16% 향상되어 51.5%에 도달한다”고 보고했다([[limo]] arXiv:2502.03387v3 §6.3.2, Fig. 4).

| 파이프라인(stage) | 기준 모델 | 통계량 | 유지 규칙과 결과 개수 | 출처 |
|---|---|---|---|---|
| Qwen3 cold start (SFT) | Qwen2.5-72B-Instruct | CoT 없는 정답 | CoT 없이 정답을 맞힌 프롬프트를 버리고, multi-part 및 free-text-generation 질의도 버린다. | [[qwen-3]] §4.1 |
| s1K (distill-SFT) | Qwen2.5-7B-Instruct와 Qwen2.5-32B-Instruct, Claude 3.5 Sonnet이 채점 | 풀었는지 | 둘 중 하나라도 풀면 버린다. 59,029개 수집 → API 오류 뒤 54,116개 → 형식 검사 뒤 51,581개 → 난이도 필터 뒤 24,496개 → 도메인 층화 표본화 뒤 1,000개. 1,000개 중 384개는 고품질로 판단한 출처에서 직접 가져왔다. | [[s1]] §2.1–2.2 |
| LIMO (distill-SFT) | Qwen2.5-Math-7B-Instruct, 이어서 DeepSeek-R1-Distill-Qwen-32B | 4회 시도 정답 여부, 32회 중 정답 수 | 4회 중 0회 정답, 이어서 32회 중 1–3회 정답: 2,125문제 | [[limo]] §3.1.1 |
| Light-R1 (distill-SFT) | DeepScaleR-1.5B-Preview, 이어서 DeepSeek-R1(§3.1.3; §1은 두 번째 stage에 DeepSeek-R1-Distill-Qwen-32B를 명시) | 통과율(n은 인쇄하지 않음) | 약 1000k seed 질문에서 pass rate < α(α는 인쇄하지 않음): 약 76k. 이어서 < α이면서 균일하게 정답 또는 오답이 아닌 것: 약 3k | [[light-r1]] §1, §3.1.1, §3.1.3 |
| Phi-4-reasoning (distill-SFT) | 강한 모델의 plurality 답을 proxy truth로 사용, 약한 모델은 Phi-4 또는 GPT-4o | 약한 모델이 proxy truth와 일치하는 비율, rubric LLM이 세는 reasoning step | “의미 있는 gap”이 있는 seed를 유지한다. threshold는 인쇄하지 않았다. | [[phi-4]] arXiv:2504.21318v1 §2.1 |
| OpenMathReasoning, round 2 (distill-SFT) | tool을 쓰는 Qwen2.5-Math-72B-Instruct | 32회 통과율 | ≤ 0.3, Olympiad forum만, 풀이 길이 ≥ 5000 token: 220만 샘플 | [[openmathreasoning]] §5.1 |
| Magistral math (RL data) | Mistral Large 2, 이어서 24B RL-trained grader | 통과당 16개 샘플 | 한 번도 풀리지 않은 문제와 성공률이 높은 문제, majority 답이 기준 답과 모순되는 문제를 버린다. 699k → 형식 필터 뒤 501k → 38k | [[magistral]] §4.1, Table 1 |
| GLM-4.5 (SFT) | 밝히지 않음 | 응답 길이 | 프롬프트 하위 50%를 버린다. | [[glm-4-5]] §3.1 |
| OpenThoughts 수학·과학 (distill-SFT) | GPT-4.1-mini | 응답 길이 | 가장 긴 것을 유지한다. | [[open-thoughts]] §4.3 |
| AceReason-Nemotron 1.1 (distill-SFT) | DeepSeek-R1 | 응답 길이 | 응답이 약 2,000 token 이하인 프롬프트를 무작위 downsample한다. | [[acereason-nemotron-1-1]] §3.1.1 |
| Llama 4 Maverick (SFT) | Llama-model judge | “easy” tag | easy tag가 붙은 데이터의 50% 이상을 제거한다. Behemoth는 SFT 데이터의 95%를 가지치기했다. “smaller model의 50%와 대조된다.” | [[llama-4]] post |
| Kimi k1.5 (RL 프롬프트, 일반 질의응답) | CoT 없는 모델 | N = 8에서 맞힌 guess 수 | 8회 시도 안에 맞혀 추측할 수 있으면 버린다. | [[kimi-k1-5]] §2.1 |
| LN-Ultra (RL data) | LN-Super | 8회 통과율 | pass rate ≥ 0.75를 버린다. | [[llama-nemotron]] §5.1 |

**수식.** 필터가 프롬프트별로 기준 모델을 n번 샘플링하고, 정답 샘플 수 k가 집합 S에 속할 때 유지한다면 유지 확률은 다음과 같다.

P(keep | p) = Σ_{k ∈ S} C(n, k) · p^k · (1 − p)^(n − k)

p는 기준 모델이 그 프롬프트를 샘플당 풀 확률, n은 샘플 수, k는 정답 샘플 수, S는 유지하는 k 값의 집합, C(n, k)는 이항계수다. 이 식은 프롬프트마다 하나의 고정된 p가 있고 샘플이 독립이라는 가정에 선다. 논문들은 p를 보고하지 않는다.

**계산 예.** LIMO stage 1은 n = 4에서 S = {0}을 유지하므로 P(keep) = (1 − p)^4다. 7B 모델이 p = 0.1로 푸는 문제는 0.9^4 = 0.656 확률로 남고, p = 0.3이면 0.7^4 = 0.240, p = 0.5이면 0.5^4 = 0.0625다. LIMO stage 2는 n = 32에서 S = {1, 2, 3}을 유지한다. P(keep)는 p = 0.01에서 0.275, p = 0.05에서 0.732, p = 0.1에서 0.566, p = 0.2에서 0.092이며 p ≈ 0.057 근처에서 최대 0.740이다. LN-Ultra 규칙은 8개 중 k ≤ 5를 남긴다. LN-Super가 p = 0.75로 푸는 프롬프트도 0.32 확률로 남고, p = 0.9에서는 0.038이다. Kimi guessability 규칙은 8개 중 k = 0을 유지한다. p = 0.1로 추측 가능한 프롬프트는 0.9^8 = 0.430 확률로 살아남는다. n이 작으면 threshold 근처 유지 집합이 noisy해진다.

동반 그림 [difficulty-filter-explorer.html](figures/difficulty-filter-explorer.html)은 이 규칙의 P(keep | p)를 그리고, 한 chart에서 두 규칙을 비교하며, OpenMathReasoning·s1K·LIMO·Magistral·Light-R1의 출판된 선택 단계를 log scale로 보여 준다.

**효과 근거.**

- 32B s1의 AIME24에서 무작위 질문 1,000개는 36.7, 가장 긴 trace 1,000개는 33.3, quality·difficulty·diversity를 합친 s1K는 50.0, 전체 59K 질문은 53.3이었다. 59K와 s1K 차이의 95% bootstrap interval은 [−13.3%, 20.0%]로 0을 포함한다([[s1]] Table 2). 59K 학습은 394 H100 GPU hour, s1K는 7시간이 들었다(§5.1).
- GLM-4.5는 응답 길이로 프롬프트 하위 50%를 버리자 “절반 데이터만 학습했음에도 수학과 과학 task에서 2%–4% 향상”을 얻었다([[glm-4-5]] §3.1). 모델 크기, benchmark, 표는 주지 않았다.
- Light-R1-32B는 약 3k개의 더 어려운 예제로 stage 2를 수행해 AIME24를 69.0에서 73.0, AIME25를 57.4에서 64.3으로 올렸지만 GPQA는 64.3에서 60.6으로 떨어졌다([[light-r1]] Table 2).
- NaturalThoughts의 Llama-3.1-8B-Instruct 학생 GPQA-Diamond는 10k 예제에서 DeepSeek-R1과 Llama-3.3-70B가 불일치하는 프롬프트를 쓰면 39.7, 무작위면 37.6이었다. 500k에서는 45.2 대 무작위 48.3이었다([[naturalthoughts]] Table 1).

상태: 선택된 프롬프트가 500~10k개에서 무작위 또는 쉬운 프롬프트보다 낫다는 결과가 재현됐다. s1K는 quality·difficulty·diversity를 결합하고([[s1]]), LIMO는 질문 난이도 level을 비교하며([[limo]]), NaturalThoughts는 모델 불일치로 고른다([[naturalthoughts]]). 난이도만으로는 충분하지 않다. s1에서 가장 긴 trace 1,000개는 AIME24에서 33.3으로 무작위 36.7보다 낮았다([[s1]] Table 2). 이 이점이 수십만 예제에도 유지되는지는 open question이다. 위 유일한 500k 비교에서는 한 criterion이 뒤집혔다.

**조건과 한계.**

- 유지 집합은 기준 모델에 의존한다. Magistral은 Mistral Large 2를 한 번만 통과시키면 “진짜 어려운 문제 다수를 풀 수 없는 것으로 잘못 분류해 버렸을 가능성이 높다”는 이유로 두 번째 더 강한 grader를 썼다([[magistral]] §4.1).
- s1은 학생 기반 모델이기도 한 Qwen2.5-32B-Instruct로 필터한다. 저자들은 이 필터링이 “our setup에 최적화되었을 수 있다”고 지적한다([[s1]] §2).
- pass rate 0은 어려운 프롬프트와 망가진 프롬프트를 섞는다. Light-R1의 RL 프롬프트 세트에서 pass-rate-0 프롬프트의 “절반 이상”은 검증 불가하거나 잘못된 기준 답을 가졌다([[light-r1]] §4).
- 응답 길이는 난이도와 장황함을 섞는다. 길이가 selection confound인 문제는 [[ch-22]]에서 다룬다.

범용 모델에 대한 함의: pass-rate 필터에는 기준 답이 필요하므로 open-ended chat 프롬프트에는 적용할 수 없다. 정답 검증이 가능한 도메인 밖에서는 Phi-4-reasoning의 plurality proxy answer와 rubric 기반 step counting이 문서화된 대안이다([[phi-4]] §2.1).

## §3 교사에게 보내는 내용: 시스템 프롬프트, 모드 제어, 샘플링

**교사 요청(teacher request)**은 시스템 프롬프트, 선택적 모드 플래그, 사용자 프롬프트, 그리고 샘플링 설정으로 이뤄진다. 샘플링 설정은 temperature T, nucleus threshold top-p(확률의 합이 top-p에 도달하는 가장 작은 토큰 집합에서만 샘플링), 생성할 새 토큰의 최대 수다. **모드 제어(mode control)**는 추론 또는 직접 답변을 고르는 지시다. 두 모드 모두로 학습하면 하나의 학생 모델이 두 행동을 모두 유지할 수 있다.

**Qwen3 플래그.** thinking-mode fusion은 Stage-1 질의에 대해 Stage-2 모델로 rejection sampling한 thinking 데이터와, 코딩·수학·instruction following·다국어 과제·창작 글쓰기·질문 응답·role-play를 포괄하고 자동 생성 checklist로 검사한 non-thinking 데이터를 Stage-2 reasoning 모델에 학습시킨다([[qwen-3]] §4.3). chat template은 다음과 같다(Table 9).

```
<|im_start|>user                     <|im_start|>user
{query} /think<|im_end|>             {query} /no_think<|im_end|>
<|im_start|>assistant                <|im_start|>assistant
<think>                              <think>
{thinking content}
</think>                             </think>
{response}<|im_end|>                 {response}<|im_end|>
```

non-thinking 샘플은 비어 있는 think block을 유지한다. multi-turn 데이터에서는 여러 플래그를 삽입하며 응답은 마지막 플래그 뒤에 온다. 추론 예산은 추론 시점에 다음 문구를 삽입하여 강제한다. `Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n` 보고서는 이 능력이 “명시적으로 학습되지는 않는다”고 말한다(§4.3). Qwen3-0.6B~14B와 Qwen3-30B-A3B로의 증류는 먼저 두 모드의 교사 출력을 사용하고(off-policy), 그 뒤 학생이 생성한 응답에서 학생과 교사 logits 사이 KL divergence를 최소화한다(on-policy)(§4.5).

**Llama-Nemotron 시스템 지시.** 추론 샘플에는 `detailed thinking on`, 비추론 샘플에는 `detailed thinking off`를 넣는다. reasoning 데이터에서 프롬프트를 샘플링한 뒤 Llama-3.1-Nemotron-70B-Instruct(일반 프롬프트) 또는 Llama-3.3-70B-Instruct(그 밖의 프롬프트)로 non-reasoning 응답을 생성하고, ground truth 또는 reward model로 필터해 쌍을 만든다([[llama-nemotron]] §3.2). 수학에서는 reasoning-on 샘플이 2,225,427개, reasoning-off 샘플이 19,840,970개다(Table 2). LN-Nano는 reasoning on에서 AIME24 61.3, off에서 3.0을 기록한다(Table 3). IFEval에서 LN-Nano-SFT는 두 모드 모두 69.9이며, “주로 IFEval 정확도 향상”을 겨냥한 RPO stage 뒤 최종 LN-Nano는 on에서 79.29, off에서 82.1이다(Table 3, §7.2).

**DeepSeek-V3 expert 데이터.** 각 domain expert는 `<problem, original response>`와 `<system prompt, problem, R1 response>`라는 두 유형의 샘플로 학습한다. 후자의 system prompt는 성찰과 검증을 요구한다. high-temperature sampling을 사용하는 RL은 system prompt 없이도 expert가 이 패턴을 생성하도록 학습시키며, 그 뒤 expert들은 rejection sampling으로 SFT 데이터를 생성한다([[deepseek-v3]] §5.1). DeepSeek-V2.5에서 이 데이터는 MATH-500을 74.6에서 83.2로 올렸고, 평균 응답 길이는 769 토큰에서 1510 토큰으로 늘었다(Table 9).

**DeepSeek-R1 cold-start 양식.** 저자들은 “문제를 이해하는 것으로 시작하고, 성찰과 검증을 포함한 상세한 추론이 뒤따르는” 사고 과정을 선호한다. 그 과정은 1인칭으로, 질문의 언어로 작성한다([[deepseek-r1]] v2 B.3.2). DeepSeek-R1-Zero는 temperature 1.0으로 샘플링하고, 정답성(수학에는 sympy)과 가독성(반복 탐지, 언어 혼합 필터)으로 필터한다. DeepSeek-V3는 `Translate the thinking process to the same language as the question.`이라는 지시로 이를 다듬는다. 사람 주석자가 trace를 대화형 양식으로 바꾸고, LLM은 그 예시로 더 많은 데이터를 다시 쓰며, 두 번째 사람 검수가 출력을 확인한다([[deepseek-r1-recipe]] B.3.2). 요약 프롬프트(Listing 1)는 다음으로 시작한다.

```
Based on the above thought process, provide a clear, easy-to-follow, and well-formatted
solution to the question. Use the same language as the question.
The solution must strictly follow these requirements:
- Stay faithful and consistent with the given thought process. Do not add new reasoning
steps or conclusions not shown in the original.
```

**Magistral 및 Phi-4-reasoning 시스템 프롬프트.** Magistral의 system prompt에는 `Write both your thoughts and summary in the same language as the task posed by the user`가 들어가며, 저자들은 RL이 system prompt에 “상당히 민감하다”고 보고한다([[magistral]] §2.2.4). Phi-4-reasoning은 하나의 고정 reasoning system message로 학습한다. 학습 중 이를 일반적 변형 문구로 교체하면 추론 시점의 무작위 system message에는 더 강건해졌지만, 원래 message 아래에서는 “benchmark 점수 변동성이 더 커지고 평균 benchmark 성능이 약간 감소”했다([[phi-4]] arXiv:2504.21318v1 §3.1). 안전 응답은 교사 prompt에 상세 guideline을 넣어 생성했으며, 학습 prompt에서는 guideline을 빼서 “모델이 기대되는 행동을 암묵적으로 학습하도록 유도”했다(§2.2).

| 파이프라인 | 교사 | Temperature | top-p | 새 토큰 최대 수 | 출처 |
|---|---|---|---|---|---|
| OpenMathReasoning CoT | DeepSeek-R1, QwQ-32B | 0.7 | 0.95 | 16,384 | [[openmathreasoning]] §2.3 |
| OpenCodeReasoning | DeepSeek-R1 | 0.6 | 0.95 | 16k | [[opencodereasoning]] §2.2 |
| Llama-Nemotron code | DeepSeek-R1 | 0.6 | 0.95 | 보고하지 않음 | [[llama-nemotron]] §3.1.2 |
| DeepSeek-R1 cold start | DeepSeek-R1-Zero | 1.0 | 보고하지 않음 | 보고하지 않음 | [[deepseek-r1-recipe]] B.3.2 |
| OpenR1-Math-220k | DeepSeek-R1 | `model card's recommended parameters`, 값은 인쇄하지 않음 | 인쇄하지 않음 | 16k | [[openr1-recipe]] Update #2 |
| Bespoke-Stratos-17k, 공개 스크립트 | DeepSeek-R1 API | 0.0 (`"temp": 0.0`) | 보고하지 않음 | 보고하지 않음 | [[bespoke-stratos]] `generate_numina_data.py` L78 |
| OpenThoughts3 | QwQ-32B | 보고하지 않음 | 보고하지 않음 | 보고하지 않음 | [[open-thoughts]] App. D-E |
| Phi-4-reasoning | o3-mini, high reasoning effort | 보고하지 않음 | 보고하지 않음 | 보고하지 않음 | [[phi-4]] §3.2 |

이 출처들 중 SFT 데이터용 교사 temperature를 ablation한 곳은 없다. OpenCodeReasoning이 temperature 0.0, 0.2, 0.6, 0.7, 1.0을 sweep한 것은 학생 추론용이며, 이때 0.6과 0.7이 가장 좋았다([[opencodereasoning]] §3). 상태: 교사 샘플링 temperature는 open question이다.

**교사 출력의 길이 제한.** OpenR1-Math-220k에서 문제의 75%만이 8k 토큰 안에 풀렸고, 나머지 대부분은 16k를 필요로 했다([[openr1-recipe]] Update #2). Hermes 4 데이터에는 최대 16 thousand tokens의 thinking trace가 들어 있지만, Qwen3-14B 학생은 LiveCodeBench 생성의 60%에서 40,960-token 한계에 도달했다. 두 번째 SFT stage는 모델 자신의 생성에서 30,000 토큰에 `</think>`를 삽입하고 `</think>`와 `<eos>`만 학습했다. 그 결과 LiveCodeBench v6는 28.6에서 44.2로 올랐고 AIME'25는 48.7에서 42.5로 내려갔다([[hermes-4]] §3.1, Table 2). RL의 길이 제어는 [[ch-44a]]에서 다룬다.

범용 모델에 대한 함의: mode-paired 데이터는 추론 학생 모델에서 직접 답변 모드를 보존하기 위해 문서화된 방법이며, non-thinking 절반이 다루는 과업 범위가 학습되는 비추론 행동의 범위를 결정한다.

## §4 교사 선택과 학생 용량

**교사 선택**은 학습용 trace를 작성할 모델을 고르는 일이다. 측정 가능한 문제는 교사 자체의 benchmark 점수가 그 교사 출력으로 학습한 학생의 점수를 예측하지 못한다는 점이다.

**근거.**
- OpenThoughts에서 Qwen2.5-7B-Instruct 학생에게 31,600개 예시를 학습시켰다. 코드 질문 출처에서는 QwQ-32B trace의 평균이 44.2이고 DeepSeek-R1 trace는 42.3이었으며, 수학 출처에서는 각각 44.2와 41.6이었다. 그러나 교사 자체 점수는 DeepSeek-R1이 CodeElo 53.7 대 44.3, GPQA-Diamond 73.7 대 65.0, JEEBench 92.3 대 69.9로 더 높다([[open-thoughts]] §4.6, Tables 8 and 29). 같은 비교를 담은 appendix 표는 DeepSeek-R1 평균을 코드 38.0, 수학 40.6으로 더 낮게 인쇄하지만 QwQ-32B 값은 동일하게 44.2이다(Tables 47-48). 따라서 어느 표를 쓰더라도 결과의 방향은 변하지 않는다. Claude 3.7 주석은 코드와 과학에서 DeepSeek-R1보다 약간 나빴다(Tables 19-20).
- [[distillation-source-matters]]에서는 1.89M개의 공유 query를 사용했다. 각 교사는 자기 응답이 category verifier에서 적어도 0.9점을 받을 때까지 다시 생성한다. 그 뒤 Qwen2.5-32B base에 동일한 SFT를 적용했다(LR 8e-5, 2 epochs, batch 64, 32k tokens). AM-Thinking-v1 / Qwen3-235B-A22B / DeepSeek-R1의 학생은 AIME2024에서 84.3 / 79.4 / 70.9, AIME2025에서 72.2 / 62.2 / 52.8, LiveCodeBench에서 65.9 / 59.6 / 57.0을 기록했다(Table 1). 세 학습 세트의 평균 perplexity는 2.5 / 3.0 / 2.9이다(§2.3, Fig. 5). 교사별 실행은 한 번뿐이며, AM-Thinking-v1은 저자들이 만든 모델이다.
- Phi-4-reasoning에서 중간 effort의 o3-mini는 교사로서 DeepSeek-R1과 “similar effect”를 냈고 token 효율은 더 좋았다. high effort는 더 긴 응답을 내면서 “a stronger teacher than medium-effort consistently across tasks”였다([[phi-4]] §3.2). 수치는 인쇄하지 않았다.

상태: prompt와 SFT 설정을 고정했을 때 교사를 바꾸면 학생 점수가 바뀐다는 사실은 재현되어 있다([[open-thoughts]] Table 8; [[distillation-source-matters]] Table 1). 교사 자체 점수가 더 낮아도 더 좋은 교사가 될 수 있다는 결과는 단일 연구의 결과다. [[open-thoughts]]는 교사 자체 점수를 보고한다(Table 29). 반면 [[distillation-source-matters]]는 본문에 세 교사의 점수를 보고하지 않으므로 이 점을 검정하지 않는다.

**학습 가능성 격차(learnability gap).** [[small-models-learnability-gap]]는 다음을 정의한다.

∆Long = P_Long − P_Short,  ∆Large = P_Large − P_Small

여기서 P_Long과 P_Short은 QwQ-32B-Preview의 긴 CoT 또는 Qwen2.5-32B-Instruct의 짧은 CoT로 SFT한 뒤 학생이 다섯 수학 benchmark에서 얻은 평균 점수다. P_Large와 P_Small은 평균 응답 길이가 각각 432.98과 440.70 tokens인 Qwen2.5-72B-Instruct 또는 Qwen2.5-3B-Instruct의 짧은 CoT로 SFT한 뒤의 점수다. Qwen2.5-3B-Instruct에서 ∆Long = 40.3 − 43.4 = −3.1이고, Qwen2.5-32B-Instruct에서 ∆Long = 73.0 − 59.3 = +13.7이다(Table 1). ∆Long은 Qwen2.5 0.5B, 1.5B, 3B와 Llama-3.2 1B, 3B에서 음수이고 7B부터 양수다. 긴 CoT와 짧은 CoT를 1:4 비율로 섞으면 Qwen2.5-3B의 평균은 45.9가 되어 두 순수 세트보다 높았다(Table 3). 설정은 MATH prompt 7,500개, 교사 greedy decoding, 두 epoch, 14B 초과에서는 LoRA, 수학 benchmark만이다. 상태: 단일 연구 결과다.

**용량 격차 법칙(capacity-gap law).** [[capacity-gap-law-distillation]]은 가장 좋은 학생 교사 크기를 다음과 같이 맞춘다.

T* ≈ 2.498 · S − 11.498  (R² = 0.9957)

여기서 S는 학생 크기, T*는 가장 좋은 교사 크기이며 둘 다 parameter 수를 million 단위로 표현한다. S = 300M이면 T* ≈ 2.498 × 300 − 11.498 = 737.9M이다. S = 3,000M이면 T* ≈ 7,482.5M이며, 이는 저자들이 3B 학생에 7B-8B 교사를 선택한 것과 맞는다(§4). 조건은 학생을 교사에서 pruning하고, pretraining-style 데이터에서 teacher distribution 절반과 ground truth 절반으로 이루어진 token-level loss로 학습하는 것이다(§3.2, Eq. 4). 저자들은 생성 텍스트 distillation에서는 “informative distributed probabilities”인 교사 지식이 one-hot label로 축소되기 때문에 이 우려가 “might be rather waived”라고 적었다(§2.1). 그러므로 이 법칙은 logit distillation의 근거다. trace SFT의 직접 근거는 위의 교사 비교와 학습 가능성 격차 연구다(해석).

**SFT가 주는 상한.** Llama-Nemotron은 “distillation inherently sets an upper bound on the student's performance”이며 LN-Ultra-SFT는 “can approach the performance of DeepSeek-R1 but not exceed it”라고 말한다([[llama-nemotron]] §5). LN-Ultra-SFT는 GPQA-Diamond 66.4, DeepSeek-R1은 71.5, RL 뒤 LN-Ultra는 76.0을 기록했다(Table 5). 다른 SFT 대 RL 비교는 다음과 같다.
- DeepSeek-R1-Distill-Qwen-32B는 AIME2024에서 72.6에 도달했다. Qwen2.5-32B base에서 10K step을 넘겨 RL만 한 경우는 47.0이다([[deepseek-r1-recipe]] F.1, Table 16).
- Magistral Small(24B)의 AIME'24 pass@1은 Magistral Medium trace SFT에서 65.4, RL만 하면 65.8, SFT 뒤 RL을 하면 70.7이다. maj@64는 각각 90.0, 86.7, 83.3이다([[magistral]] Table 3).
- 같은 off-policy distilled checkpoint(AIME'24 55.0, pass@64 90.0)에서 출발한 Qwen3-8B는 RL로 17,920 GPU hours에 67.6과 pass@64 90.0에 도달했다. on-policy distillation은 1,800 GPU hours에 74.4와 pass@64 93.3에 도달했다([[qwen-3]] §4.7, Table 21).
- 학생은 단일 benchmark에서는 교사를 넘을 수 있다. OpenThinker3-7B는 JEEBench에서 72.4이고 QwQ-32B는 69.9이다([[open-thoughts]] App. K).

범용 모델에 대한 함의: off-policy trace에서 학생은 교사 텍스트만 보고 자기 오류는 한 번도 보지 않는다. on-policy distillation과 RL은 학생이 낸 sample로 학습한다([[ch-35]], [[ch-38a]]).

## §5 더 많은 프롬프트, 프롬프트당 더 많은 응답, 또는 더 많은 epoch

x를 서로 다른 프롬프트 수, y를 프롬프트당 응답 수, e를 epoch 수라고 하자. SFT 세트에는 x·y개의 예시가 있고, 학습 중에는 x·y·e번의 예시 pass를 본다.

**AceReason-Nemotron 1.1 회귀.** 7개 SFT 세트 v1-v7(36K~2.2M sample, Qwen2.5-Math-7B 학생, DeepSeek-R1 교사)에 다음 식을 fit했다([[acereason-nemotron-1-1]] §4.4.2).

z = a · x̃ + b · ỹ + c

여기서 z는 AIME24, AIME25, LiveCodeBench v5 및 v6의 평균 accuracy다. x̃와 ỹ는 7개 세트에서 평균 0, 단위 분산이 되도록 표준화한 log2 x와 log2 y다. a와 b는 fit한 계수이고 c는 절편이다. fit 결과는 a = 4.831, b = 2.635, R² = 0.989이다.

계산 예시: log2 프롬프트가 한 표준편차 차이 나는 두 세트의 예측 accuracy 차이는 4.831점이다. 프롬프트당 응답의 log2가 한 표준편차 차이 나면 2.635점이며, 비율은 1.83이다. 논문은 표준편차 자체를 인쇄하지 않았으므로, 두 배 증가당 효과는 논문에서 계산할 수 없다. 점 7개와 parameter 3개이므로 자유도는 4개가 남고, v1-v4는 모두 프롬프트당 응답이 하나여서 x와 y를 독립적으로 변화시키지 않았다. 상태: 단일 연구의 결과다.

같은 논문의 다른 근거: v7은 v6와 비슷한 수의 프롬프트를 유지하면서 프롬프트당 응답을 “nearly twice as many”로 늘렸고, AIME25는 41.3에서 49.3으로 올랐다(§4.4.1, §4.4.3). v6와 v7에서는 epoch 1에서 5까지 accuracy가 상승하고 epoch 5-6 근처에서 plateau에 도달했다. 저자들은 어느 정도의 “overfitting”이 주는 이점을 exposure bias로 설명한다(Fig. 6; 해석).

**OpenThoughts의 질문당 응답 수.** 최종 pipeline은 모든 domain에서 질문당 16개 응답을 사용한다(§4.4). 과학에서는 exact deduplication을 적용한 16개 응답이 최선이었다(평균 36.2, Table 43). 코드에서는 deduplication 없이 16개 응답을 쓴 경우가 exact deduplication을 적용한 4개 응답과 41.3으로 동률이었다(Table 41). 수학에서는 exact deduplication을 적용한 응답 하나가 41.7이었고 16개 응답은 40.1이었지만, 저자들은 “as it provides better scalability”라는 이유로 16개를 택했다(§4.4 및 App. S.4, Table 42; 본문은 수학의 승자로 4개 응답을 언급하지만 appendix 표에는 1개라고 적혀 있다). 계산 예시: OpenThoughts3 dataset card는 수학 180k, 코드 60k, 과학 60k 질문으로 filtering한 뒤(합계 300k), deduplication하고 75k 질문으로 downsampling한 다음 각 질문에 QwQ-32B annotation 16개를 붙였다고 적는다. 75,000 × 16 = 1,200,000개 예시이며, 이것이 최종 dataset 크기다. 질문당 응답이 하나라면 같은 filtering pool은 최대 300,000개 예시만 공급할 수 있다(도출).

**그 밖의 다중성 데이터.**

- OpenMathReasoning은 문제당 최대 32개 candidate를 생성하며, 더 어려운 문제에는 더 많이 생성한다. 5.2M개를 생성하여 3.2M개를 유지했으므로 유지 비율은 61.5%다(DeepSeek-R1은 4.2M개 중 2.7M개, 64.3%; QwQ-32B는 1.0M개 중 0.5M개, 50%; [[openmathreasoning]] Table 5에서 도출).
- GLM-4.5에서는 어려운 프롬프트마다 응답 네 개를 쓴 것이 “an additional 1%-2% improvement”를 냈다([[glm-4-5]] §3.1).
- NaturalThoughts의 random selection 및 Llama-3.1-8B-Instruct에서 GPQA-Diamond는 1k일 때 37.1, 100k일 때 42.5, 500k일 때 48.3이었다([[naturalthoughts]] Table 1).
- Hermes 4는 OpenThoughts를 따라 “multiple unique trajectories to the same verified result”를 포함한다([[hermes-4]] §2.2).

**Epoch.** 800~10k개 예시의 curated 세트는 5~15 epoch 학습한다. LIMO는 예시 800개를 15 epoch 학습하고([[limo]] §4), s1은 1,000개를 5 epoch 학습하며([[s1]] App. D), NaturalThoughts는 1k에 10 epoch, 10k에 6 epoch을 쓴다([[naturalthoughts]] §4). Llama-Nemotron은 “extended training over multiple epochs improves performance, particularly for smaller models”라고 보고한다. LN-Super는 compute 제한 때문에 5e-6에서 한 epoch 학습했지만, 더 작은 run은 더 큰 learning rate(5e-5)에서 3-4 epoch까지 성능이 향상됨을 “suggested”했다([[llama-nemotron]] §4.1-4.2). OpenMathReasoning은 “smaller models need to be trained for longer”라고 관찰한다([[openmathreasoning]] Fig. 4).

**RL 이전의 더 강한 SFT.** AceReason SFT v5와 v7 모델에서 시작한 RL은 AIME24 격차를 6.6%에서 1.6%로 줄였다([[acereason-nemotron-1-1]] Fig. 7). LN-Ultra의 RL은 뒤의 SFT checkpoint가 더 높은 점수를 냈음에도 “to improve final RL outcomes”라는 이유로 더 이른 SFT checkpoint에서 시작했다([[llama-nemotron]] §7.4). 상태: RL 전에 SFT를 얼마나 오래 학습해야 하는지는 열린 질문이다. 두 보고서는 서로 다른 방향을 가리키며, 어느 쪽도 통제된 방식으로 SFT 길이를 바꾸지 않았다.

범용 모델에 대한 함의: 이 출처들 중 어느 것도 epoch 수 또는 프롬프트당 응답 수에 따라 target 밖 suite가 어떻게 변하는지 보고하지 않는다. 이 측정은 [[ch-30a]]의 forgetting 분석과 [[ch-36]] lab에 속한다.

## §6 학생 SFT 설정: 학습률, 길이, 단위

**토큰 손실 평균.** SFT 손실은 target 토큰에 대한 평균 cross-entropy이며, 평균을 내는 단위가 각 토큰의 gradient 가중치를 바꾼다. 계산 예시를 보자. 한 batch에 20,000-token trace 하나와 200-token answer 하나가 들어 있다. batch 안의 모든 target 토큰에 걸쳐 손실을 평균 내면, 각 토큰의 가중치는 1/20,200이고 짧은 answer는 전체에서 200/20,200 = 1.0%를 차지한다. 각 sequence 안에서 먼저 평균을 내고 그 뒤 sequence들 사이에서 평균을 내면, 각 trace 토큰의 가중치는 1/(2 × 20,000) = 2.5 × 10⁻⁵이고 각 answer 토큰의 가중치는 1/(2 × 200) = 2.5 × 10⁻³이다. 이는 100배 차이다. Llama-Nemotron은 “models require higher learning rates to effectively learn from long reasoning traces, especially due to sequence-length-dependent token loss averaging”이라고 보고한다([[llama-nemotron]] §4.1). 이 보고서는 어느 평균 규약을 쓰는지 밝히지 않으므로, 그 연결은 해석(Interpretation)이다. 손실 평균은 [[ch-30]]에서 다룬다.

**학습률 근거.**

- DeepDistill, Qwen2.5-72B base, 약 5M sample: LR 8e-5는 AIME2024에서 79.2, LiveCodeBench에서 63.8을 냈고, 8e-6은 각각 72.5와 60.2를 냈다([[deepdistill]] §4.4.2).
- OpenCodeReasoning, Qwen2.5 7B-32B base 및 instruct: 1e-5, 3e-5, 5e-5, 8e-5, 1e-4 grid에서 “across model sizes”로 5e-5를 선택했다([[opencodereasoning]] §3).
- LN-Ultra: 5e-5는 “generally improve[d] outcomes”이었지만 gradient explosion을 일으켰다. 최종 run은 1e-5까지 warmup한 뒤 1e-6으로 decay했으며, 그래도 첫 epoch 뒤 optimizer state를 reinitialize하여 restart해야 했다([[llama-nemotron]] §4.2).
- Phi-4-reasoning, post-trained Phi-4에서 출발: [1e-6, 2e-5] grid에서 1e-5를 골랐다. “higher learning rates result in lower training loss, but saturation and/or degradation across various downstream evaluations”라고 보고했다([[phi-4]] §3.1).
- OlympicCoder, reasoning data에서 instruct model(Qwen2.5 Coder Instruct)을 SFT: figure에서 읽으면 학습률을 두 배로 올릴 때마다 LiveCodeBench가 거의 10점씩 향상됐다. 4e-5를 사용했다([[openr1-recipe]] Update #3).

조건과 한계: 4e-5 이상을 지지하는 결과는 수만~수백만 개 예시와 대부분 base model을 사용한다(OpenCodeReasoning과 OlympicCoder는 instruct model도 학습한다). Phi-4-reasoning은 1.4M pair를 사용한 post-trained model에서 시작해 1e-5가 최선이라고 찾았다. LIMO(5.0e-6)와 Light-R1-DS(5.0e-6)는 800개 또는 약 3k개 예시로 instruction-tuned 또는 이미 distilled된 model에서 시작한다. 이 보고서들에서는 시작 checkpoint, data 크기, 학습률이 서로 confound되어 있고, 어느 연구도 이들을 함께 변화시키지 않았다. 상태: 열린 질문이다.

**단위.** Hermes 4는 global batch 384 sample, 16,384-token context에서 9,000 step과 model당 56B training token을 인쇄한다([[hermes-4]] §3, Table 1). 9,000 × 384 × 16,384 = 56.6B token position(도출)이므로 packed batch의 56B와 일관된다. Phi-4-reasoning은 batch 32, context 32K에서 “roughly 16K steps”와 16B training token을 인쇄한다. 16,000 × 32 × 32,000 = 16.4B이며, 이는 반올림된 수치에서 도출한 값이다([[phi-4]] §3, §3.2). OpenThoughts는 단위 없이 batch 512를 인쇄한다([[open-thoughts]] App. D.2). 보고서들 사이의 batch 값은 step당 token으로 환산한 뒤에만 비교할 수 있다.

## §7 답변 측과 추론 trace 측의 품질 관리

**답변 측 품질 관리**는 최종 답 또는 프로그램을 참조 답과 비교한다. 참조에는 규칙 검증기, 모델 심사자, 또는 실행 결과를 쓸 수 있다. **trace 측 품질 관리**는 추론 텍스트 자체를 확인한다. 확인 대상은 언어, 형식, 반복, 그리고 내부 일관성이다.

**답변 측 메커니즘.**

1. 모델 심사자 fallback을 둔 규칙 검증기. OpenR1-Math-220k에서는 문제의 55%가 Math-Verify를 통과한다. 거부된 문제의 일부에 Llama-3.3-70B-Instruct 심사자를 적용하면 문제 28,000개를 추가로 복구한다([[openr1-recipe]] Update #2).
2. 파서 대신 모델 심사자 사용. Sky-T1의 regex 및 sympy 파서를 gpt-4o-mini 심사자로 바꾸자, 보존된 올바른 수학 풀이의 비율이 25%에서 73%로 올랐다([[bespoke-stratos]] “Data Curation”). 계산 예시: 교사가 낸 올바른 풀이가 10,000개라면 파서는 그중 2,500개만 남기고 올바른 풀이 7,500개를 버린다. 심사자는 7,300개를 남긴다(도출).
3. 동치성 심사자. Qwen2.5-32B-Instruct가 예측 답과 기대 답이 “문제의 문맥에서(in the context of the problem)” 동치인지 판정한다([[openmathreasoning]] §2.3; [[llama-nemotron]] §3.1.1).
4. 구조화된 심사자. DeepSeek-R1의 rejection-sampling SFT 데이터는 DeepSeek-V3와 프롬프트를 사용한다. 이 프롬프트는 답을 correct 또는 incorrect로 분류하고, `'analysis'` 키를 포함하는 JSON을 반환한다([[deepseek-r1]] v2 B.3.3, Listing 4).
5. 합의 레이블. 답을 추출할 수 없을 때 후보들에서 가장 자주 나온 답을 참조 답으로 쓴다([[openmathreasoning]] §2.3). Llama-Nemotron은 ground truth가 없는 과학 문제에 majority voting을 쓴다(§3.1.3).
6. 임계값과 재생성. DeepDistill은 수학·코드·instruction following에는 verifier score가 0.99를 초과한 응답을, 과학에는 0–5 점수에서 4.99를 초과한 응답을, multi-turn 및 기타 데이터에는 0.7을 초과한 응답을 남긴다([[deepdistill]] §2.4.2). [[distillation-source-matters]]는 각 응답이 최소 0.9점을 받을 때까지 다시 생성한다(§2.2).
7. 많은 task verifier. Hermes 4는 “roughly a thousand task-specific verifiers”를 상대로 rejection sample한다. DataForge 심사자는 실패한 답을 통과할 때까지, 또는 최대 iteration 횟수에 도달할 때까지 수정한다([[hermes-4]] §2.1-2.2).

**trace 측 메커니즘.**

- DeepSeek-R1 800K 세트는 언어가 섞인 chain of thought, 긴 문단, 또는 code block을 포함한 chain of thought를 제거한다([[deepseek-r1-recipe]] B.3.3).
- Qwen3 cold start는 다음 응답을 제거한다. (1) 최종 답이 틀린 응답, (2) 상당한 반복을 포함한 응답, (3) 충분한 추론 없이 추측만 보이는 응답, (4) thinking과 summary가 서로 일관되지 않은 응답, (5) 언어가 섞이거나 부적절하게 스타일이 바뀐 응답, (6) validation item과 지나치게 유사한 것으로 의심되는 응답. QwQ-32B가 일관되게 실패하면 human annotator가 응답을 평가한다([[qwen-3]] §4.1).
- OpenCodeReasoning은 `<think>` tag와 풀이 안의 code block을 가진 응답을 남긴다. 반면 reasoning에 code block이 들어간 응답은 제거하고, Tree Sitter로 solution syntax를 검사한다([[opencodereasoning]] §2.3).
- DeepDistill은 perplexity가 20을 초과한 sample, 20-token 문자열이 20회보다 많이 반복된 sample, turn 수가 홀수인 sample, think 또는 answer segment가 빠진 sample을 제거한다([[deepdistill]] §2.5).
- GLM-4.5는 반복적·짧은·잘린·형식이 나쁜 sample을 제거한다. 주관적 응답에는 reward model을 사용하며, call protocol을 위반하거나 기대한 terminal state에 도달하지 못한 tool trajectory를 제거한다([[glm-4-5]] §3.1).
- LIMO는 DeepSeek-R1, DeepSeek-R1-Distill-Qwen-32B, QwQ-32B에서 sample한 각 후보 chain을 점수화한다. 가중치는 길이 30%, 검증 단어(`"check"`, `"verify"`) 빈도 20%, 잠정 표현(`"perhaps"`, `"might"`) 빈도 25%, 연결사(`"therefore"`, `"since"`) 빈도 25%다. keyword count는 텍스트 길이로 정규화한다. 문제당 최고 chain을 남기고, 2,125개 문제 중 상위 800개로 데이터셋을 만든다([[limo]] §3.1.2).

**반증 근거.**

- OpenThoughts, 7B, 수학 질문, 수학 평균: 답 63,200개에서 무작위로 31,600개를 고르면 64.8점이었다. GPT verification은 61.4점이었고, 63,200개 전부로 학습하면 데이터가 두 배이면서 65.6점이었다([[open-thoughts]] §4.5, Tables 7, 45). 같은 크기의 코드 질문에서는 GPT verification(40.7)과 majority consensus(41.3)가 전체 평균에서 random selection(39.8)을 이겼다. 그러나 어떤 filter도 모든 답으로 학습한 경우(42.2)를 0.1점보다 크게 넘지 못했다(Table 44). 저자들의 결론은 “no filtering strategy outperformed the baseline, which uses all the answers”였다(§4.5).
- OpenThoughts의 verified OpenThoughts-114K와 unverified OpenThoughts-114K 비교는 크기를 맞추지 않았다. 7B 평균은 45.0에서 41.9로 떨어졌고, 32B 평균은 62.1에서 64.5로 올랐다(App. H.1.1, Table 15). 같은 16,000개 코드 예시에서 LLM 생성 unit-test filtering은 LiveCodeBench 36.0이었고, 무작위 unfiltered sample은 38.5였다(App. H.1.4, Table 18).
- OpenCodeReasoning, Qwen2.5-14B-Instruct: filtering 없음(445,618개)은 LiveCodeBench 54.1, 통과한 풀이만 사용(151,251개)은 47.0, 모든 test에 실패한 풀이 사용(151,251개)은 52.3이었다([[opencodereasoning]] Table 3).
- s1: grader는 s1K trace의 53.6%를 correct로 판정했고, s1K로 학습한 모델은 AIME24에서 50.0에 도달했다([[s1]] §2, Table 2).

상태: 같은 데이터 크기에서 correctness filtering이 7B–14B 학생에게 이득이 없거나 손실을 냈다는 결과가 수학([[open-thoughts]] Table 45)과 코드([[opencodereasoning]] Table 3; [[open-thoughts]] Table 18)에서 재현되었다. 다만 두 결과가 이 결론을 제한한다. OpenThoughts의 코드 filter는 같은 크기에서 random selection을 이겼고(Table 44), 크기를 맞추지 않은 비교에서는 verification이 32B에 도움이 됐다([[open-thoughts]] App. H.1.1).

메커니즘(해석, OpenCodeReasoning 저자들이 제시): 교사는 어려운 질문에서 더 자주 실패한다. 따라서 고정된 개수만 남긴 상태에서 verified answer만 고르면 집합은 더 쉬운 질문 쪽으로 이동한다. 저자들은 “incorrect solutions span questions that are more challenging than the ones associated with the correct solutions”이라고 썼다(§4.1, Fig. 4). 답변 filter ablation이 correctness의 효과를 측정하려면, 남긴 집합과 버린 집합의 난이도를 맞춰야 한다.

**엄격한 검증을 쓰는 곳.** Cold-start 데이터는 correctness와 readability로 filter한 뒤 사람이 검증한다([[deepseek-r1-recipe]] B.3.2). RL prompt 세트에서는 틀린 reference를 감사한다. Magistral은 majority answer가 reference와 모순되는 문제를 제거하고([[magistral]] §4.1), Light-R1은 pass-rate-0 prompt를 모델 verifier로 다시 검사하며([[light-r1]] §4), AceReason은 “incorrect test cases can lead to false negative rewards”라고 지적한다([[acereason-nemotron-1-1]] §3.2.2). 이 단계에서 틀린 reference answer는 틀린 reward signal을 만든다(해석).

범용 모델에 대한 함의: 대규모 SFT에서 검증되지 않은 답을 남기면 틀린 내용을 positive target으로 학습한다. 위 출처들 중 어느 것도 factuality, calibration, hallucination에 미치는 효과를 측정하지 않았다. 따라서 unfiltered와 filtered를 비교할 때는 reasoning benchmark에 더해 이 suite들도 넣어야 한다.

## §8 오염 제거, 중복 제거, 증류 데이터 카드

**오염 제거(decontamination)** 는 평가 항목과 일치하는 학습 프롬프트를 제거하는 작업이다. **n-gram filter** 는 학습 항목과 benchmark 항목이 연속된 n개 token으로 이루어진 구간을 하나라도 공유하면 그 학습 항목을 표시한다. **embedding filter** 는 항목 embedding과 benchmark 항목 embedding의 유사도가 임계값을 넘으면 표시하며, 그 뒤에 LLM 또는 사람이 검토하는 경우가 많다.

| 파이프라인 | 방법 | Benchmark | 감사 또는 주석 | 출처 |
|---|---|---|---|---|
| s1K | 8-gram overlap | MATH500, GPQA Diamond, AIME24 | — | [[s1]] App. C.5 |
| Open R1 | s1을 따른 8-gram match 및 deduplication | benchmark dataset | `scripts/decontaminate.py` | [[openr1-recipe]] README |
| AceReason-Nemotron 1.1 | 9-gram overlap | 수학 및 코딩 benchmark | — | [[acereason-nemotron-1-1]] §3.1.1 |
| DeepSeek-V3 및 R1 | pre-training 및 post-training 데이터에서 10-gram match | 평가 질문 및 reference solution | 잠재적인 수학 pretraining text 약 600만 개를 제거함; paraphrase는 잡지 못함 | [[deepseek-r1-recipe]] v2 D.1 |
| OpenThoughts | character-level Indel similarity ≥ 75%, 또는 공유된 13-gram 하나 이상(Qwen2-7B-Instruct tokenizer) | evaluation set | 심어 둔 항목 3,092개 중 12개를 놓침; 깨끗한 항목 3,000개 중 1.4%를 버림 | [[open-thoughts]] App. F |
| Light-R1 | 숫자를 무시한 exact match 및 32-gram | AIME24, AIME25, MATH-500, GPQA | 공개 set 감사(아래) | [[light-r1]] §3.1.2, App. C |
| OpenCodeReasoning | cosine similarity로 가장 가까운 benchmark 이웃 선택(threshold 0.7), LLM judge 2개, 수동 검토 | LiveCodeBench, CodeContests, HumanEval, MBPP | 표시된 항목 90개(≤ 0.3%)가 paraphrase가 아니라고 판정됨 | [[opencodereasoning]] §2.1 |
| OpenMathReasoning, Llama-Nemotron | Yang et al. (2023)을 따르는 LLM 기반 비교 | 널리 쓰이는 수학 benchmark; 과학에는 GPQA, MMLU, MMLU-Pro | — | [[openmathreasoning]] §2.1; [[llama-nemotron]] §3.1.3 |
| DeepDistill, distillation-source-matters | exact match 및 bge-m3 similarity > 0.9 | evaluation set, 명시된 AIME2024 | — | [[deepdistill]] §2.2; [[distillation-source-matters]] §2.1 |
| Phi-4-reasoning | Phi-4 오염 제거 절차 | AIME-2024, MATH, GPQA, LiveCodeBench, Codeforces, OmniMATH, SWE-Bench Verified, ArenaHard, MT-Bench 등을 포함한 출력 목록 | AIME-2025는 데이터 확정 뒤에 공개됨 | [[phi-4]] §2.2 |
| Hermes 4 | 보고하지 않음 | — | — | [[hermes-4]] |

**공개 set 감사.** Light-R1은 공개 증류 set에서 MATH-500과 일치하는 학습 프롬프트 수를 센다. Open-Reasoner-Zero 325개, data_ablation_full59K 244개, DeepScaleR-Preview-Dataset 196개, Bespoke-Stratos-17k 125개, OpenThoughts-114k 100개, OpenR1-Math-220k 10개, s1K-1.1 3개, LIMO 0개다. 나열된 모든 set에서 AIME24 및 AIME25 match는 0개다([[light-r1]] App. C, Table 7). 이 표는 서로 다른 benchmark 항목 수가 아니라 일치한 학습 프롬프트 수를 세므로, 영향을 받은 MATH-500 항목 비율은 표만으로 계산할 수 없다. 본문은 MATH-500에 동일하거나 숫자 값만 다른 “tens of compromised questions”가 들어 있다고 말한다(§3.1.2).

**계산 예시.** 30-token benchmark 질문에 training pool 안의 변형이 하나 있다고 하자. 이 변형은 token position 10과 20의 숫자만 바꾼다. 공유하는 token run은 position 1–9(9 token), 11–19(9 token), 21–30(10 token)이다. 9-token run에는 공유된 9-gram이 들어 있으므로 8-gram 또는 9-gram filter는 이 변형을 표시한다. 13-gram 또는 32-gram filter는 표시하지 않는다. OpenThoughts의 Indel 규칙은 similarity = LCS length / max(|s1|, |s2|)이며, 여기서 LCS는 longest common subsequence이고 |s|는 문자열 s의 길이다(App. F, Eq. 1). 논문은 이를 character에 적용한다. 이 예시에 token 단위로 적용하면 longest common subsequence는 28 token이므로 similarity = 28 / 30 = 0.93이다. 이는 0.75보다 크므로 규칙이 변형을 표시한다. Light-R1은 숫자를 무시하는 exact matching으로 이 경우를 포괄한다.

**범위 한계.** 이 filter들은 학습 프롬프트와 benchmark 항목을 비교한다. 어느 것도 교사의 trace가 교사가 암기한 benchmark solution을 재현하는지 시험하지 않는다. 날짜로 구분한 평가는 보완 검사다. AIME 2025는 Phi-4-reasoning 데이터가 확정된 뒤에 공개되었고([[phi-4]] §2.2), AceReason-Nemotron 1.1은 AIME25와 LiveCodeBench v6를 “that carry a lower risk of contamination”인 benchmark로 보고한다([[acereason-nemotron-1-1]] §4.3). 탐지 방법은 [[ch-48]]에서 다룬다.

**중복 제거(deduplication).** OpenThoughts는 수학과 과학에는 exact deduplication을, 코드에는 deduplication을 하지 않는 방법을 택했고, 두 경우 모두 질문당 답변 16개를 사용했다. 저자들은 “there does not seem to be a clear trend in types of deduplication that improve performance”라고 보고한다(§4.4, Table 6 caption; Tables 41–43). OpenCodeReasoning은 exact match로 질문을 deduplicate한다. AceReason-Nemotron 1.1은 각 prompt가 unique하도록 deduplicate하지만 방법은 밝히지 않는다. 여기 인용한 어느 출처도 증류 prompt의 near-duplicate 또는 semantic deduplication을 비교하지 않는다. 방법은 [[ch-12]]에 있다.

**증류 데이터 카드(distillation data card).** 아래 표는 증류 set을 재현하고 점검할 수 있게 만드는 필드를 나열한다. 예시 열은 OpenMathReasoning CoT subset에서 인쇄된 값을 채운 것이다.

| 필드 | 기록할 내용 | 예시: OpenMathReasoning CoT subset ([[openmathreasoning]]) |
|---|---|---|
| 단계와 학생 | 단계 이름; 학생 base와 크기 | distill-SFT; RoPE base 500K인 Qwen2.5-Base 14B 및 32B, Qwen2.5-Math 1.5B 및 7B(§5.1) |
| 프롬프트 출처와 단계별 수 | 각 출처; filter 뒤의 각 수 | AoPS: 620K → 580K → 550K → 540K(Table 1) |
| 도메인과 task 비중 | 개수 및 token 기준 비중 | 수학만; 변환된 proof 260K, 답변 있음 190K, 답변 없음 90K(Table 2) |
| 난이도 통계량 | 기준 모델; n; 남긴 범위 | Qwen2.5-72B-Math-Instruct의 32회 pass rate; 전부 보존, 더 어려운 문제에는 더 많은 sample(§2.3) |
| 교사와 prompt | 교사 모델; system prompt; mode flag | DeepSeek-R1, QwQ-32B; system prompt는 인쇄하지 않음 |
| 샘플링 | T; top-p; max new token; prompt당 sample | 0.7; 0.95; 16,384; 최대 32개(§2.3) |
| 답변 측 QC | verifier; judge; recall audit | Qwen2.5-32B-Instruct equivalence judge; 답이 없으면 majority answer; recall 미보고 |
| trace 측 QC | 규칙과 threshold | CoT subset에 대해서는 보고하지 않음 |
| 보존 비율 | kept / generated | 3.2M / 5.2M = 61.5%(Table 5; 도출) |
| 오염 제거와 dedup | 방법; n 또는 threshold; benchmark 목록 | LLM 기반 비교(§2.1); n-gram 길이는 해당 없음 |
| 학생 SFT | LR; schedule; 단위가 있는 batch; epoch; max length; packing; loss mask | 1e-4(14B, 32B), cosine, 10% warmup, 1,024 sample, 6 epoch, max length 미보고, packing, loss mask 미보고; 최종 model은 일정 간격의 checkpoint 4개를 평균한 값(§5.1) |
| 평가 | held-out 및 날짜로 구분한 set; 비대상 suite | 2024–2025 대회의 Comp-Math-24-25; 비대상 suite 미보고(§2.2) |
| 미보고 감사 | 출처가 제공하지 않는 모든 field | max SFT length; loss mask; trace 측 규칙; verifier recall |

## 음성 샘플과 음성 피드백

이 절은 [[ch-31a]]가 구분한 `negative`의 네 가지 의미를 사용한다. (1) 음의 한계 가치: positive target으로 쓰면 성능을 낮추므로 걸러 내는 샘플, (2) 내용으로서의 음성, (3) conditioning으로서의 음성, (4) gradient로서의 음성이다.

1. **음성이 생기는 곳.** 답변 수준의 label은 rule verifier(Math-Verify, sympy), model judge(DeepSeek-V3, Qwen2.5-32B-Instruct, gpt-4o-mini, Llama-3.3-70B-Instruct), unit test 실행, reward model([[llama-nemotron]] §3.1.4의 일반 데이터용 Llama-3.1-Nemotron-70B), majority-vote label에서 나온다. trace 수준 label은 형식 및 언어 규칙에서 나온다. prompt 수준 label은 기준 모델 통계에서 나온다. 즉 너무 쉽거나, 풀리지 않거나, 추측 가능하거나, 모델 다수가 reference answer를 반박하는 prompt다. 보고된 false negative는 다음과 같다. regex와 sympy parser는 올바른 수학 풀이의 25%만 보존했고 gpt-4o-mini judge는 73%를 보존했다([[bespoke-stratos]]). Llama judge는 Math-Verify가 거부한 문제 28,000개를 복구했다([[openr1-recipe]]). Light-R1의 pass-rate-0 RL prompt 중 절반 초과는 검증할 수 없거나 틀린 reference를 가졌다([[light-r1]] §4).
2. **현재 관행이 하는 일.** OpenMathReasoning, Llama-Nemotron 수학, DeepSeek-R1, Qwen3, Light-R1, OpenR1-Math-220k는 검증에 실패한 답을 가진 샘플을 버린다(의미 1). [[distillation-source-matters]](§2.2)와 Hermes 4 DataForge judge loop([[hermes-4]] §2.1)는 샘플이 통과할 때까지 재생성하거나 수정한다. 일부 set은 검증되지 않은 trace를 positive target으로 남긴다. 53.6%만 correct 판정을 받은 s1K와, 실패한 풀이도 학생을 가르친다는 ablation을 보인 OpenCodeReasoning release가 그렇다([[s1]] §2; [[opencodereasoning]] §4.1). Light-R1은 NCA loss를 쓰는 DPO 단계에서 음성을 gradient로 쓴다(의미 4). rejected response는 SFT-stage-2 model이 만든 검증-오답 sample이고, chosen response는 검증-정답인 DeepSeek-R1 답이다([[light-r1]] §3.2).
3. **메커니즘.** 남긴 trace로 SFT할 때 각 target token `y`는 logit `z`를 `∂ log p_y / ∂z_j = 1[j = y] − p_j`만큼 바꾼다. 여기서 `p_j`는 token `j`에 대한 모델 확률이다. target이 아닌 모든 token의 logit은 자기 확률에 비례해 낮아진다. 틀린 trace를 버리면 그 positive update만 제거하고 아무 logit도 내리지 않는다. negative-gradient term은 rejected token에서 부호를 뒤집는다. 계산 예시: 다음-token 확률이 0.7, 0.2, 0.1이고 step size 1의 gradient step이 0.1 token의 `log p`를 낮춘다고 하자. logit 변화는 `−(1[j = neg] − p_j)`이므로 rejected token은 −0.9, 나머지는 +0.7과 +0.2다. 새 확률은 0.832, 0.144, 0.024다. rejected token은 0.076을 잃고 두 번째 token도 0.056을 잃는 반면, 가장 확률이 높던 token은 0.132를 얻는다. 제거한 모든 확률 질량이 이미 가장 가능성 높던 token으로 간다. 전체 유도는 [[ch-31a]] §2와 [[ch-43a]]에 있다.
4. **수치가 있는 근거.** 같은 크기에서 틀린 답을 버리면 7B–14B에서 수학([[open-thoughts]] Table 7)과 코드([[opencodereasoning]] Table 3) 모두 이득이 없거나 손실이었다. 크기를 맞추지 않은 verified/unverified OpenThoughts-114K 비교에서는 7B가 45.0 → 41.9로 떨어지고 32B가 62.1 → 64.5로 올랐다([[open-thoughts]] Table 15). negative-gradient 단계에서 Light-R1-32B DPO는 AIME24를 73.0에서 75.8로, AIME25를 64.3에서 63.4로 바꿨다(Table 2). 저자들은 어려운 수학에서 “significantly stronger models”의 chosen response가 완전히 on-policy인 DPO보다 더 나은 결과를 냈다고 보고하지만 수치는 제시하지 않는다(§3.2). 여기 인용한 출처 중 이득의 어느 부분이 negative 때문인지 측정한 것은 없다.
5. **통제.** filter 전에 label된 sample에서 verifier recall을 감사한다. parser failure에는 judge fallback을 추가한다. 난이도를 맞춘 상태에서 filter를 비교한다. cold start 또는 RL prompt set의 seed가 되는 데이터는 엄격하게 검증한다. distilled trace에 쓰는 DPO형 단계에서는 [[ch-31a]] §5처럼 positive log-likelihood term을 남기거나 negative term을 제한한다.
6. **진단.** verifier별·난이도 bin별 kept share, kept/discarded sample의 길이 분포, 난이도별 correct/incorrect count([[opencodereasoning]] Fig. 4), DPO 단계의 chosen/rejected log-probability를 각각 기록한다. filter를 바꾼 전후에는 큰 `k`에서 pass@k를 본다.
7. **범용성에 미치는 효과.** coverage 측면에서는 고정된 수에서 correctness로 filter하면 어려운 prompt가 제거된다([[opencodereasoning]] §4.1에서의 해석). Qwen3의 on-policy distillation은 pass@64를 올렸지만 RL은 올리지 못했다([[qwen-3]] Table 21). calibration과 hallucination은 이 출처들이 측정하지 않았다. over-refusal은 학습에 safety data가 없었던 OpenThinker3-7B에서 XSTest 기준 4.4 → 5.6으로 올랐다([[open-thoughts]] App. L). 이는 filtering 효과가 아니라 데이터 coverage 효과다. filtering 함수로서의 forgetting도 측정되지 않았다.

## 레시피

단위: `samples`와 `examples`는 prompt-response pair다. 달리 명시하지 않는 한 batch 값은 sequence 수다. LR은 peak learning rate다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| OpenMath-Nemotron (OpenMathReasoning CoT data) | 1.5B–32B | distill-SFT | 교사; 문제당 sample; temperature; top-p; 최대 token | DeepSeek-R1, QwQ-32B; 최대 32개, 어려운 문제에는 더 많이; 0.7; 0.95; 16,384 | arXiv:2504.16891v1 §2.3 | 2026-09-15 검증 | ablation 미보고 |
| OCR-Qwen (OpenCodeReasoning data) | 7B–32B | distill-SFT | 교사; temperature; top-p; 최대 출력 | DeepSeek-R1; 0.6; 0.95; 16k | arXiv:2504.01943v2 §2.2 | 2026-09-15 검증 | ablation 미보고 |
| LN-Nano, LN-Super, LN-Ultra (수학 data) | 8B, 49B, 253B | distill-SFT | 문제당 생성 수 | DeepSeek-R1 16개(reasoning); Qwen2.5-Math-7B-Instruct 64개(non-reasoning) | arXiv:2505.00949v5 §3.1.1 | 2026-09-15 검증 | ablation 미보고 |
| LN-Nano, LN-Super, LN-Ultra (코드 data) | 8B, 49B, 253B | distill-SFT | temperature; top-p | 0.6; 0.95 | arXiv:2505.00949v5 §3.1.2 | 2026-09-15 검증 | ablation 미보고 |
| OpenThinker3-7B | 7B | distill-SFT | 교사; 문제당 답; 교사 temperature, top-p, 최대 token | QwQ-32B; 16; 미보고 | arXiv:2506.04178v2 §4.4, §4.6; App. D, E, R.3, dataset card 확인 | 2026-09-14 검증; sampling 미보고 | Table 8(교사, 31,600-example run); Tables 41–43(과학에서는 답 16개가 최선, 코드에서는 동률, 수학에서는 두 번째; 확장성을 위해 선택) |
| DeepSeek-R1 (Dev1 cold start) | 671B MoE | SFT | 생성기; temperature | DeepSeek-R1-Zero; 1.0 | arXiv:2501.12948v2 B.3.2 | 2026-09-14 검증 | ablation 미보고 |
| OpenR1-Math-220k | — | distill-SFT | 최대 생성; 문제당 sample; temperature | 16k; 2개, “in some cases, four”; 인쇄하지 않음 | HF blog Open R1 Update #2 | 2026-09-14 검증; temperature 미보고 | ablation 미보고 |
| Bespoke-Stratos-17k curation script | — | distill-SFT | DeepSeek-R1용 generation_params(Numina subset) | `{"temp": 0.0}` | curator main `generate_numina_data.py` L78 | 2026-09-14 검증(released code이며 dataset run으로 확인된 것은 아님) | ablation 미보고 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | distill-SFT | base; data; epoch; 초기 LR; schedule; context; batch | Qwen2.5-Math-7B; 804,745-sample B.3.3 set(Table 5); 2–3; 8×10⁻⁵; 1/10까지 cosine; 32,768; 64 | arXiv:2501.12948v2 B.4.3, Table 6 | 2026-09-14 검증 | ablation 미보고 |
| DeepSeek-R1-Distill-Qwen-32B | 32B | distill-SFT | base; 초기 LR(나머지 설정은 7B 행과 같음) | Qwen2.5-32B; 6×10⁻⁵ | arXiv:2501.12948v2 B.4.3, Table 6 | 2026-09-14 검증 | ablation 미보고 |
| OpenThinker3-7B | 7B | distill-SFT | base; example; LR; batch; epoch; packing; 최대 길이 | Qwen2.5-7B-Instruct; 1.2M; 8e-5; 512(단위 미명시); 5; 예; 미보고 | arXiv:2506.04178v2 App. D.2 Table 9 “Large” | 도출(1.2M은 >31.6K 범위) | scale별 sweep을 언급하지만 결과 미보고 |
| LN-Nano | 8B | distill-SFT | stage-1 data; LR; epoch; global batch; packed length | reasoning data만; 1e-4; 4; 256; 32k | arXiv:2505.00949v5 §4.2 | 2026-09-15 검증 | stated reason: 반복 completion 방지; 표 없음 |
| LN-Super | 49B | distill-SFT | epoch; LR; sequence 길이; global batch | 1; 고정 5e-6; 16k; 256 | arXiv:2505.00949v5 §4.2 | 2026-09-15 검증 | 더 작은 run은 5e-5에서 3–4 epoch까지 향상; 표 없음 |
| LN-Ultra | 253B | distill-SFT | LR schedule; packed 길이; global batch; epoch | 1e-5까지 warmup(10%), 1e-6까지 cosine; 24k; 256; 미보고 | arXiv:2505.00949v5 §4.2 | 2026-09-15 검증; epoch 미보고 | ablation에서는 5e-5가 더 좋지만 불안정; 표 없음 |
| AceReason-Nemotron-1.1 SFT-7B | 7B | distill-SFT | base; sample(v7); epoch; LR; batch; 최대 길이 | Qwen2.5-Math-7B, rope_theta 1,000,000; 2.2M; 미보고; 미보고; 미보고; 미보고 | arXiv:2506.13284v1 §3.1.2, §4.4.3 | 2026-09-15 검증; epoch 및 optimizer 값 미보고(§3.1, §4.4 확인) | Fig. 6: v6/v7 정확도는 epoch 5–6 부근에서 plateau; §4.4.2 regression |
| OpenMath-Nemotron-14B, -32B | 14B, 32B | distill-SFT | data; epoch; peak LR; 최종 LR; batch; warmup; weight decay | 5.5M(3.2M CoT, 1.7M TIR, 566K GenSelect); 6; 1e-4; peak보다 1000× 작음; 1,024 samples; 선형 10%; 0.01 | arXiv:2504.16891v1 §5.1 | 2026-09-15 검증 | Fig. 4(epoch별 정확도, 1.5B/14B) |
| OCR-Qwen-7B/14B/32B(-Instruct) | 7B–32B | distill-SFT | epoch; batch; 최대 길이; LR; warmup; packing | 3; 256; 32,768; 5e-5; 비율 0.1; 예 | arXiv:2504.01943v2 §3 | 2026-09-15 검증 | LR grid 1e-5–1e-4 |
| Light-R1-32B | 32B | distill-SFT | stage 1 LR / batch / length; stage 2 LR / batch / length; epoch | 5.0e-5 / 96 / 20k; 1.0e-5 / 32 / 20k; 미보고 | arXiv:2503.10460v4 App. D Table 8 | 2026-09-15 검증; epoch 미보고 | Table 2(stage별 점수) |
| Light-R1-7B-DS | 7B | distill-SFT | 시작점; example; LR; batch; length | DeepSeek-R1-Distill-Qwen-7B; 약 3k; 5.0e-6; 32; 20k | arXiv:2503.10460v4 §3.4, Table 8 | 2026-09-15 검증 | Table 3 |
| Magistral Small | 24B | distill-SFT | 시작점; data; epoch; checkpoint 규칙; LR, batch | Mistral Small 3 Instruct; Magistral Medium RL trace + OpenThoughts/OpenR1-code prompt 응답 + 일반 instruction data 10%; 4; 가장 좋은 AIME'24; 미보고 | arXiv:2506.10910v1 §5.3 | 2026-09-14 검증; LR, batch 미보고 | Table 3(SFT vs RL vs SFT + RL) |
| Phi-4-reasoning | 14B | distill-SFT | example; 고유 token; step; batch; context; LR; warmup; weight decay | 1.4M pair; 8.3B; 약 16K; 32; 32K; 1e-5; 450 step; 1e-4 | arXiv:2504.21318v1 §3 | 2026-09-15 검증 | §3.1: LR grid [1e-6, 2e-5]; weight decay 0 vs 1e-4는 variance 안 |
| Hermes 4 14B / 70B / 405B | 14B, 70B, 405B | distill-SFT | base; token; LR; step; warmup; batch; context; epoch | Qwen3 14B / Llama 3.1 70B / Llama 3.1 405B; 각 56B; 5e-5 / 1e-5 / 5e-6; 9,000, cosine; 300; 384 samples; 16,384; 미보고 | Hermes 4 Technical Report §3, Table 1 | 2026-09-15 검증; epoch 미보고 | ablation 미보고 |
| LIMO (Qwen2.5-32B-Instruct) | 32B | distill-SFT | example; LR; schedule; warmup; epoch; batch | 800; 5.0e-6; cosine; 없음; 15; 64 | arXiv:2502.03387v3 §4 | 2026-09-15 검증 | ablation 미보고 |
| s1-32B | 32B | distill-SFT | example; epoch; batch; step; LR; warmup; beta; weight decay; loss | 1,000; 5; 16; 315; 1e-5; 5%(16 step), 0까지 cosine; (0.9, 0.95); 1e-4; reasoning 및 solution token만 | arXiv:2501.19393v3 App. D | 2026-09-15 검증 | App. D.1(sequence 길이) |
| Qwen2.5-72B base student (DeepDistill Stage I) | 72B | distill-SFT | sample; LR; epoch; packing; 최대 길이; batch | 5M; 8e-5; 1; 예; 32k; 64 | arXiv:2504.17565v3 §3.2, §4.1 | 2026-09-14 검증 | §4.4.2: 8e-6은 AIME2024 72.5, 8e-5는 79.2 |
| OpenR1-Distill-7B | 7B | distill-SFT | data; epoch; LR; schedule; batch; 최대 길이; packing; grad clip | Mixture-of-Thoughts 349,317행; 5; 4.0e-5; 10%까지 cosine, warmup 0.03; 128(장치당 2 × accumulation 8 × 장치 8); 32,768; 아니오; 0.2 | `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`@1416fa0; model card | 2026-09-14 검증 | model card: 세 도메인 모두 math + code 초과 |
| NaturalThoughts students | 8B, 7B | distill-SFT | LR; epoch(1k / 10k / 100k–500k); batch; 최대 response 길이 | 고정 2e-5; 10 / 6 / 8; 약 400k token; 16,384 | arXiv:2507.01921v1 §4 | 2026-09-15 검증 | ablation 미보고 |

**작은 범용 run의 출발점.** 수학·코드·과학의 DeepSeek-R1 trace 349,317개로 8개 device와 32,768-token context에서 7B base student를 학습한다면, OpenR1-Distill-7B 행은 문서화된 설정이다. 즉 5 epoch, 10%까지 cosine decay와 3% warmup을 포함한 LR 4.0e-5, step당 128 sequence, packing 없음, gradient clipping 0.2다. post-trained model에서 시작하는 14B student에 Phi-4-reasoning은 1.4M pair에 걸쳐 LR 1e-5, batch 32, 32K context, warmup 450 step, weight decay 1e-4를 사용했다. 데이터에서는 7B에서 문제당 교사 답 16개가 OpenThoughts의 선택이었고, 문서화된 교사 sampling은 수학(OpenMathReasoning)에서 temperature 0.7, top-p 0.95, 16,384 token, 코드(OpenCodeReasoning)에서 temperature 0.6, top-p 0.95, 16k token이다. 이 설정들 중 chat, instruction-following, safety behavior의 보존을 시험한 것은 없다. 범용 student에 문서화된 추가 사항은 Llama-Nemotron(§3.2)처럼 mode-paired non-reasoning data를 포함하고 그 suite에서 점수를 측정하는 것이다.

## 일반화 관점

**(a) 폭을 넓히는 것.**
- Mode-paired data는 하나의 모델에 reasoning mode와 direct-answer mode를 준다. LN-Nano는 reasoning on에서 AIME24 61.3, off에서 3.0을 기록한다([[llama-nemotron]] Table 3). 그러나 mode-paired data SFT는 instruction following을 보존하지 못했다. LN-Nano-SFT의 IFEval은 두 mode 모두 69.9였고 Llama-3.1-8B-Instruct는 81.8이었다. RPO 단계는 off를 82.1로 올렸다(Table 3, §7.2). 결과(단일 연구)다. Qwen3도 같은 설계를 쓰지만([[qwen-3]] §4.3) paired ablation은 보고하지 않는다.
- 넓은 question pool: 여러 domain의 2.8M question에서 나온 NaturalThoughts는 Llama-3.1-8B-Instruct의 MMLU-Pro를 무작위 100k example에서 47.7 → 59.8, 500k에서 61.9로 올렸다. 수학·코드·과학 100k OpenThoughts3 example은 MMLU-Pro 59.0이지만, MATH-500에서는 100k NaturalThoughts의 67.5 대비 82.2였다([[naturalthoughts]] Table 1). 결과(단일 연구)다.
- Safety와 alignment prompt를 포함한 mixed-domain reasoning data: Phi-4-reasoning의 IFEval strict는 Phi-4의 62.3에서 83.4로, ArenaHard는 68.1에서 73.3으로 올랐다. 단 Phi-4는 temperature 0.0, Phi-4-reasoning은 0.8에서 평가했다([[phi-4]] §5.2, Table 2).
- 코드 data에 과학 data를 추가하면, 과학이 포함된 두 mix의 GPQA-Diamond는 모두 52.7이었고 code-only set은 47.3과 36.7이었다([[open-thoughts]] App. H.5, Table 24).

**(b) 좁아짐 또는 forgetting을 일으키는 것.**
- 수학 중심 SFT 단계: Light-R1-32B의 GPQA는 stage 2에서 64.3 → 60.6이었다([[light-r1]] Table 2). 동일한 3k stage-2 data는 Light-R1-7B-DS에 domain 밖 이득을 주지 못했다. GPQA는 49.1 → 49.4였고 저자들은 “improvements confined solely to in-domain tasks”라고 했다. 반면 Light-R1-32B-DS의 GPQA는 62.1 → 68.0이었다(§3.4, Table 3).
- Reasoning SFT와 instruction following: LN-Super-SFT의 IFEval은 Llama-3.3-70B-Instruct의 92.1 대비 81.9였다([[llama-nemotron]] Table 4). (a)의 Phi-4-reasoning 결과는 반대 방향이지만, Phi-4-reasoning은 alignment data를 더했고 시작 모델과 평가 temperature도 다르므로 통제된 비교가 아니다.
- Safety data 부재: OpenThinker3-7B의 HarmBench는 14.5 → 55.5였다([[open-thoughts]] App. L).
- Parametric recall: Phi-4-reasoning의 no-context Kitab recall은 8.2 → 4.9로 떨어졌고 precision은 19.3 → 23.2로 올랐다([[phi-4]] Table 2).
- 작은 student의 teacher-student mismatch: Qwen2.5 0.5B–3B에서 음의 ∆Long([[small-models-learnability-gap]] Table 1).
- Robustness: Alice-in-Wonderland 문제의 구조 보존 변형 6개에서, 저자들은 테스트한 모든 distilled reasoning model의 “strong performance fluctuations”를 보고했다. 예를 들어 OpenThinker-32B는 변형 2와 3에서는 1에 가깝고 변형 1에서는 0에 가깝다([[open-thoughts]] App. N, Fig. 13).

**(c) 이 단계에서 측정하는 방법.**
- reasoning score 옆에 non-target suite를 둔다. [[llama-nemotron]] Tables 3–5의 IFEval, BFCL, Arena-Hard와 [[open-thoughts]] App. L의 HarmBench, XSTest가 그 예다.
- 날짜로 filter한 benchmark: Phi-4-reasoning의 AIME 2025, AceReason의 AIME25와 LiveCodeBench v6([[acereason-nemotron-1-1]] §4.3).
- 항목당 충분한 sample: AIME2024 pass@1 standard deviation은 avg@16에서 1.8, avg@32에서 1.2, avg@64에서 0.7이다([[acereason-nemotron-1-1]] §4.1). Light-R1은 sample 16개 이하에서 3점 초과 deviation을 관찰했다([[light-r1]] §2). interval 방법은 [[ch-51]]에 있다.
- [[s1]] Table 2처럼 data ablation에 bootstrap interval을 쓴다.
- [[light-r1]] Table 7처럼 public set별 contamination audit count를 낸다.
- 큰 `k`에서 pass@k를 본다. [[qwen-3]] Table 21의 pass@64가 예다.
- 이 출처들에 알려진 measurement error도 함께 본다. 나중에 보고할 benchmark로 selection한 경우(OpenThoughts는 AIME25, HMMT, HLE, LCB 06/24–01/25에서 이를 피한다, [[open-thoughts]] §4), baseline은 한 번만 평가했는데 student는 64회 run의 평균인 경우([[opencodereasoning]] Table 2), chat benchmark 없이 chat coverage를 주장한 경우([[distillation-source-matters]] §3.2)다.

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 방법 |
|---|---|---|
| 기준 모델, n, 보존 범위를 밝히지 않은 난이도 filter | 재실행할 때 다른 prompt set 또는 다른 개수를 보존함 | 기준 모델, n, S, 단계별 kept count를 기록하고 경계 prompt에 대해 `P(keep \| p)`를 다시 계산한다 |
| 답 감사 없이 pass-rate-0 prompt를 “어려움”으로 보존 | 수동 점검에서 검증 불가 또는 틀린 reference answer 발견; 모든 sample이 zero reward를 받는 RL group | Light-R1처럼 pass-rate-0 prompt 표본을 judge로 재점검한다 |
| 교사를 자기 benchmark 점수로 선택 | 더 낮은 점수의 교사가 낸 trace로 학습한 student보다 성능이 낮음 | 같은 prompt에서 각 교사의 trace로 작은 student를 학습한다 |
| ≤3B student에 long-CoT 전용 data | 동일 prompt의 short-CoT SFT보다 낮은 점수 | long, short, 1:4 mixed data를 비교한다 |
| matching ablation 없이 answer filter 채택 | 어려운 문제 정확도가 떨어짐; kept set이 discarded set보다 짧음 | 난이도 bin 안에서 같은 count로 filtered/unfiltered를 비교한다 |
| parser만 쓰는 verifier | 교사가 보통 푸는 문제에서 kept share가 낮음 | 거부 출력의 무작위 표본에 사람이 label하고 false-rejection rate를 보고한다. judge fallback을 추가한다 |
| 단위 없이 보고서 간 batch size 비교 | 보고서의 LR을 복사했더니 불안정하거나 느린 학습 | LR을 복사하기 전에 batch를 step당 token으로 환산한다 |
| 숫자 변형을 시험하지 않고 n-gram 길이를 선택 | MATH-500 또는 유사 benchmark 점수가 날짜 기반 score보다 높음 | digit-insensitive exact match 또는 similarity filter를 추가하고 set별 match count를 보고한다 |
| 범용 assistant에 reasoning-only SFT | IFEval, BFCL, safety 점수가 떨어짐; direct-answer mode 없음 | mode-paired non-reasoning data를 추가하고 두 mode를 모두 평가한다 |
| 큰 set에 small-data recipe의 epoch 사용 | target score는 plateau인데 non-target suite는 하락 | epoch별 target/non-target suite를 추적한다 |

## 이해 점검

1. LIMO의 stage-2 rule은 32번 중 성공 1–3개를 보존한다. `P(keep | p)`를 사용하여 `p = 0.2`인 문제는 왜 거의 보존되지 않고 `p = 0.05`인 문제는 왜 대개 보존되는지 설명하라. 그리고 DeepSeek-R1-Distill-Qwen-32B를 더 강한 모델로 바꾸면 kept set이 어떻게 바뀌는지 설명하라.
2. OpenCodeReasoning에서는 같은 개수에서 실패한 풀이가 통과한 풀이보다 더 나은 student를 학습시켰다. 난이도 분포를 이용해 인과적 설명을 제시하고 correctness 효과를 분리할 ablation을 설계하라.
3. DeepSeek-R1이 여러 benchmark에서 더 높은 점수인데도 QwQ-32B trace가 더 나은 7B student를 만들었다. [[distillation-source-matters]]에서 측정한 trace set의 어떤 속성이 이를 설명할 수 있는가? verification pass rate로는 왜 그 속성을 탐지할 수 없는가?
4. AceReason 계수는 prompt에 4.831, prompt당 response에 2.635다. 논문만으로 왜 이 값을 doubling당 점수로 바꿀 수 없는가? v1–v4 설계는 왜 비교를 약화하는가?
5. capacity-gap law는 logit distillation에서 적합했다. 저자들이 generated text 학습에서는 효과가 더 약할 것이라 기대하는 이유를 설명하라. 이 장에서 trace SFT의 teacher-student mismatch를 다루는 근거는 무엇인가?
6. DeepDistill은 72B에서 8e-5가 8e-6보다 좋았지만 Phi-4-reasoning은 더 높은 learning rate가 downstream score를 떨어뜨린다고 했다. 두 결과가 양립하도록 만드는 setting 차이를 나열하고, token loss averaging이 논증에 어떻게 들어가는지 설명하라.
7. 30-token 질문과 학습 변형이 숫자 두 개만 다르다. 8-, 13-, 32-gram filter 중 무엇이 이를 표시하는지와 이유를 설명하라. Light-R1은 왜 digit-insensitive exact match를 추가하는가?
8. 범용 8B student가 수학·코드 reasoning trace 95%로 증류된다. 이 장에서 narrowing을 가장 먼저 잡아낼 측정 세 가지와, 각각의 동기가 되는 source result를 말하라.

## 연결

- 이전: [[ch-35]] — *Distillation in Practice A: Where Labs Insert Teacher Data*. 이 장은 pipeline 안에서 teacher data를 어디에 넣는지 배치하고, 이 장은 하나의 distillation dataset을 구성한다.
- 다음: [[ch-36]] — *Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split*. §8의 data card와 non-target measurement가 이 lab으로 이어진다.
- 의존: [[ch-22]] — *Quality, Diversity, and Gradient-Based Data Selection* (selection score와 length confound).
- 관련: [[ch-20]] — *Distillation as Data: Explanation Traces and the R1-Distill Lineage*; [[ch-24]] — *Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data*; [[ch-30]] — *SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate*; [[ch-30a]] — *Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control*; [[ch-31a]] — *Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood*; [[ch-12]] — *Deduplication: Exact, Near-Duplicate, and Semantic*.
- 이후: [[ch-16]] — *RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse*; [[ch-38a]] — *SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity*; [[ch-43a]] — *Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages*; [[ch-44a]] — *Length in RL: Overlong Responses, Length Control, and Long-Context RL*; [[ch-48]] — *Contamination Detection and Its Effect on Reported Scores*; [[ch-51]] — *Metric Noise, Confidence Intervals, and Go/No-Go Decisions*.

## 출처

- [[open-thoughts]] — 출처, mixing, question-filter, answers-per-question, answer-filter, teacher ablation; decontamination rule; safety와 robustness 결과.
- [[openmathreasoning]] — AoPS pool 단계, teacher sampling 설정, consensus label, kept share, student SFT 설정(발췌; library card 없음).
- [[opencodereasoning]] — competitive-programming 집계, embedding+LLM decontamination, 736k까지의 scaling, execution-filtering ablation(발췌; library card 없음).
- [[qwen-3]] — cold-start query filter, 여섯 response criterion, `/think` 및 `/no_think` template, on-policy distillation table. library card는 재검증하지 않았다. 위치는 arXiv:2505.09388v1이다.
- [[light-r1]] — two-stage pass-rate filter, 32-gram 및 digit-insensitive decontamination, public set 감사, DPO negative, GPQA 하락(발췌; library card 없음).
- [[limo]] — two-stage difficulty filter, rule-based chain score, 15-epoch SFT, question-difficulty ablation. library card는 재검증하지 않았다. 위치는 arXiv:2502.03387v3이다.
- [[s1]] — two-model difficulty filter, bootstrap interval이 있는 selection ablation, 53.6% correct trace, 8-gram decontamination, SFT 설정. library card는 재검증하지 않았다. 위치는 arXiv:2501.19393v3이다.
- [[magistral]] 및 [[magistral-recipe]] — two-pass difficulty grading, majority-versus-reference filter, system prompt, Magistral Small SFT와 Table 3.
- [[glm-4-5]] — response-length prompt cut, hard prompt당 네 response, SFT rejection filter.
- [[llama-4]] — easy-tagged SFT data 제거와 Behemoth용 95% pruning.
- [[kimi-k1-5]] — `N = 8`인 guessability rule과 제외한 question format.
- [[acereason-nemotron-1-1]] — prompt-versus-response regression, epoch plateau, 9-gram decontamination, SFT-then-RL gap(발췌; library card 없음).
- [[distillation-source-matters]] — 1.89M query에서 통제한 세 teacher 비교.
- [[small-models-learnability-gap]] — student 크기별 ∆Long/∆Large; Mix Distillation(발췌; library card 없음).
- [[capacity-gap-law-distillation]] — `T* ≈ 2.498·S − 11.498`와 그 적용 범위(발췌; library card 없음).
- [[deepdistill]] — verifier threshold, trace filter, 72B learning-rate ablation.
- [[naturalthoughts]] — 1k–500k selection strategy, random selection scaling, mixed System-1/System-2 distillation(발췌; library card 없음).
- [[phi-4]] — Phi-4-reasoning seed selection, fixed system message, teacher effort, LR grid, decontamination 목록, general benchmark. library card는 재검증하지 않았다. 위치는 arXiv:2504.21318v1이다.
- [[bespoke-stratos]] — parser-versus-judge retention(25% → 73%), 공개된 generation parameter, student setting.
- [[openr1-recipe]] — Math-Verify pass rate와 judge recovery, 16k generation, 8-gram decontamination, OpenR1-Distill-7B configuration, OlympicCoder learning-rate observation.
- [[deepseek-r1]] 및 [[deepseek-r1-recipe]] — cold-start style/filter, V3 judge prompt, 800K domain table, R1-Distill setting, Table 16, 10-gram decontamination.
- [[deepseek-v3]] — reflection-and-verification system prompt를 둔 expert data; Table 9.
- [[llama-nemotron]] — “detailed thinking on/off” pair, data share, model별 SFT setting, distillation upper bound, IFEval 하락, RL prompt filter(발췌; library card 없음).
- [[hermes-4]] — task verifier와 judge loop, training unit, 30k `</think>` length-control stage(발췌; library card 없음).
