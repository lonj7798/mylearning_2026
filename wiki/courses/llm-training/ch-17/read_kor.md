<!-- chapter: ch-17
     track: pretraining
     kind: lab
     title: 실습: 범위 측정을 포함한 필터 및 혼합 ablation
     deps: [ch-14a]
     sources: [[ccnet]], [[fineweb]], [[dclm]], [[paloma]], [[datadecide]], [[signal-and-noise-eval]], [[minhash-lsh]], [[deduplicating-training-data]], [[dolma]], [[c4]], [[weborganizer]], [[data-constrained-scaling]], [[openmathinstruct-2]]
     figures: figures/ablation-grid.html
     revised: 2026-09 (generality revision)
-->

# Chapter 17 — 실습: 범위 측정을 포함한 필터 및 혼합 Ablation

> **핵심 통찰.** 집계 벤치마크 평균만 보고하는 필터 ablation으로는 필터가 능력을 추가한 것인지 학습 분포를 벤치마크 쪽으로 좁힌 것인지 알 수 없다. 둘 다 평균을 높인다. [[weborganizer]]는 1.44B 파라미터와 28.8B 토큰에서 그 차이를 직접 측정했다. FineWeb-Edu는 9태스크 평균을 51.6에서 54.2로 높였지만 필터링하지 않은 기준 말뭉치의 홀드아웃 perplexity도 12.1에서 14.7로 높였고, MMLU만을 기준으로 조정한 혼합은 HellaSwag을 57.5에서 54.1로, PIQA를 71.3에서 69.9로 낮췄다(표 1, 2, 10). 따라서 이 실습은 각 ablation cell에서 세 측정 계열—seed 분산이 있는 신호를 담은 태스크 점수, 도메인별 홀드아웃 perplexity, 분포 비율—을 요구하며, 더 큰 규모에서도 순위가 유지될 것으로 예상하는지 명시해야 한다.
>
> **지침.** 작은 규모에서 ablation cell을 학습할 때는 태스크 평균 옆에 도메인별 홀드아웃 perplexity를 보고하라. [[paloma]]에서 C4만으로 학습한 1B 모델은 RedPajama arXiv 도메인에서 perplexity 391,171, Dolma peS2o 도메인에서 14를 기록했기 때문이다(§2, §4.1). 시험하는 cell이 벤치마크 대상 classifier라면 설계에 사용하지 않은 태스크를 홀드아웃하라. [[weborganizer]] 표 10은 대상에서 얻으면서 HellaSwag을 3.4점 잃는 대상별 혼합을 보여준다. 작은 규모 순위를 큰 실행으로 옮길 때는 전이 근거를 명시하라. [[dclm]]은 10개 방법에서 400M-1x부터 7B-1x까지 Pearson r = 0.838을 보고하고(§3.2), [[datadecide]]는 25개 말뭉치에서 150M부터 1B까지 쌍별 결정 정확도 약 80%를 보고하며 100%가 아니다(§3.2). 그렇지 않으면 해당 규모에서의 관찰로 보고하고 일반화하지 말라.

## 범용 모델에 이 장이 중요한 이유

[[ch-09]]부터 [[ch-14a]]까지의 장은 사전학습 데이터 선택의 효과를 설명한다. 휴리스틱 pipeline([[ch-10]]), 모델 기반 필터([[ch-10a]]), 중복 제거([[ch-12]]), 혼합([[ch-13]]), 다국어 범위([[ch-13a]]), 반복과 오염 제거([[ch-14]]), 레시피 설정([[ch-14a]])이다. 이 장들은 모두 학습자가 직접 수행하지 않은 실행의 결과를 보고한다. 이 실습은 측정 장치를 만드는 곳이다. 이 과정이 중요하게 보는 실패는 앞 장들이 인용한 어떤 단일 수치에서도 보이지 않기 때문이다.

실패는 구체적인 형태를 가진다. 사전학습 필터는 말뭉치에 들어오는 문서를 바꾼다. 이런 변화는 능력의 습득량과 그 능력을 습득하는 분포의 형태를 동시에 바꾼다. 필터의 대상이 벤치마크와 비슷할 때 벤치마크 평균은 능력의 변화와 분포의 변화에 같은 방향으로 함께 움직인다. [[fineweb]] 저자들은 임계값 4가 아니라 3을 선택했다. 더 높은 임계값은 HellaSwag 같은 벤치마크와 지식·추론 벤치마크 사이를 교환했기 때문이다(§4). 이들은 두 종류의 태스크 모두에서 평가했기 때문에만 이 교환을 볼 수 있었다. 이후 [[weborganizer]]는 변화 자체를 측정했고 FineWeb-Edu와 DCLM-fastText가 서로 다른 형식 범주를 증폭하지만 Politics, Health, Science & Tech., History, Knowledge Articles, Tutorials, Academic Writing, Q&A Forums는 둘 다 증폭한다는 것을 발견했다(§5, 그림 4).

이 실습은 사전학습 단계의 끝, 중간 학습([[ch-32]]) 직전에 위치한다. 이 실습이 만드는 산출물—범위 열과 전이 진술이 있는 ablation 메모—은 이후 모든 단계에 필요한 동일한 산출물이다. SFT, 선호 최적화, RL도 작은 proxy 실행에서 데이터 선택을 하기 때문이다.

## §1 실습이 만드는 것

`ch-17/lab-artifacts/`에 네 가지 산출물이 있다.

1. `pipeline/` — 필터 cascade를 실행해 필터링된 텍스트가 아니라 문서별 **속성**을 기록한다. keep/drop은 속성 파일을 대상으로 하는 별도 pass다. 이 분리는 Dolma toolkit 문서(github.com/allenai/dolma, main의 `docs/getting-started.md`, 2026-09-14 확인)를 따른다. `dolma tag`는 tagger를 실행하고, `dolma dedupe`는 `attributes/` 아래에 중복 span 속성을 기록하며, `dolma mix`는 주석된 파일에 대한 config에서 데이터셋을 만든다. [[dolma]] 논문 자체는 문서별 속성 형식이 아니라 두 작업을 설명한다. “filtering”은 언어·품질·콘텐츠 필터를 통합하고, Rust “mixing” 모듈은 Bloom filter로 up/down-sampling, 중복 제거, 오염 제거를 구현한다(§4.1). 속성을 디스크에 유지하면 threshold sweep에서 언어 식별이나 perplexity 점수를 다시 실행하지 않아도 된다.
2. `runs/` — 각 ablation cell마다 하나의 학습 모델, cell마다 최소 두 seed.
3. `measurements.json` — cell별 태스크 점수, 도메인별 홀드아웃 perplexity, 분포 비율, 오염 개수와 각각의 seed 간 spread.
4. `ablation-memo.md` — 작성한 결과. 필수 내용은 §8에 있다.

함께 제공되는 [`figures/ablation-grid.html`](figures/ablation-grid.html)에서는 직접 선택한 모델 크기, cell당 토큰, 유지한 고유 토큰을 입력하고 cell당 FLOPs, 도출한 H100 시간, 각 필터 설정이 암시하는 epoch 수, 선택한 매개변수에 대한 MinHash banding 곡선을 볼 수 있다. [[dclm]] 표 1과 [[fineweb]] 부록 E.1의 공개 기준 행도 나란히 표시된다.

## §2 규모 선택과 그 규모의 cell이 결정할 수 있는 것

**정의.** *ablation cell*은 기준 실행과 데이터가 정확히 하나의 문서화된 방식으로만 다르고 나머지 설정은 모두 고정된 하나의 완전한 학습 실행이다.

**측정 가능한 문제.** Cell에는 계산량이 든다. 너무 작은 cell은 점수가 우연 수준이거나 그 근처여서 필터 효과를 seed 잡음과 구별할 수 없게 만들고, 너무 큰 cell은 cell 수를 적게 만든다.

**메커니즘.** 규모를 고정한 다음 예산을 도출한다.

1. 파라미터 `N`과 토큰 `D`를 고른다. 학습 FLOPs ≈ `6ND`다([[dclm]] §3.2, 표 1의 이 열에 이 정의를 사용).
2. cell 수와 cell당 seed 수를 곱한다.
3. 공개 기준값과 비교해 시간으로 변환하고, 이 변환을 도출값이라고 표시한다.

**계산 예시.** [[dclm]]의 가장 작은 competition 규모인 400M-1x는 412M 파라미터와 8.2B 토큰이다. FLOP 열을 확인하면 `6 × 412e6 × 8.2e9 = 2.03e19`이고 표에는 2.0e19로 표시되어 있다. 표는 OpenLM framework에서 해당 실행에 26 H100 시간도 표시한다. 두 seed의 8-cell grid는 16개 실행이므로 `16 × 26 = 416` H100 시간이다(인쇄된 실행당 값에서 곱셈으로 도출; 자체 framework의 처리량은 다르다).

자원 제약 경로에서 공개 기준값은 [[weborganizer]]의 RegMix proxy model이다. 실행당 50M 파라미터, 1B 토큰, “512회 실행에 약 360 H100 시간”(부록 C)이며 실행당 0.70 H100 시간이다. 이 실행의 `6ND = 3.0e17` FLOPs는 400M-1x FLOPs의 1.5%인 반면, 0.70 / 26 = 2.7% 시간이다. 따라서 FLOP 비율은 두 공개 실행에서 작은 cell의 wall-clock 비용을 1.8배 과소예측한다. 일반적인 설명은 작은 모델 크기에서 하드웨어 활용률이 낮다는 것이며(두 출처가 측정한 것이 아닌 **해석**), 16개의 제약 경로 실행은 약 11 H100 시간이다.

**조건과 한계.** 두 기준값은 하나의 cluster에서 하나의 framework가 보고한 단일 값이다. 어느 것도 당신 코드의 측정값이 아니다. 자신의 측정 시간을 메모에 기록하라.

## §3 파이프라인

아래 단계 순서는 [[dolma]]가 웹 부분집합에 사용한 순서다(§5.5): CCNet 출력 → URL 중복 제거 → 문서 중복 제거 → 품질 필터 → 콘텐츠 필터 → 문단 중복 제거. 먼저 upstream CCNet 단계에 대해 자주 잘못 말하는 두 가지를 기억하라. [[ccnet]]은 언어 식별 *전에* 문단을 중복 제거하며, 그 이유는 ablation(§4.1, 그림 3)에 있다. 반복 영어 boilerplate를 먼저 제거하면 저자원 언어 문서가 영어로 분류되거나 버려지는 것을 막는다. 또한 [[ccnet]]은 언어 모델 perplexity로 문서를 전혀 제거하지 않는다. 각 언어를 head·middle·tail 3분위로 나누고 세 가지를 모두 공개한다(§3.4, §5.2). 저자들은 Wikipedia와 어휘가 다른 텍스트, 예컨대 “말하듯 쓰인 텍스트의 블로그 댓글 또는 특정 전문용어가 있는 매우 전문화된 포럼”이 tail에 들어가기 때문이라고 설명한다(§5.2). Perplexity 필터는 기준과의 유사도 필터이며, 이 실습은 이를 하나로 취급한다.

### §3.1 언어 식별

각 문서에 fastText `lid.176`을 적용한다. [[ccnet]]은 점수 > 0.5인 최상위 언어로 페이지를 유지하고(§3.3), [[dolma]]는 영어 점수 ≥ 0.5를 사용해 바이트의 61.7%를 제거했다고 보고하며(§5.1), [[fineweb]]은 ≥ 0.65를 사용한다(§3.3). cell에서 사용한 임계값과 전후 언어 비율을 기록하라. 비율 표는 이 실습의 필수 출력 중 하나다(§5.3).

### §3.2 휴리스틱 품질 필터

재현 가능한 공개 세트 두 가지가 있다.

- [[c4]] 규칙. [[fineweb]]은 2019-18 crawl에서 규칙을 하나씩 ablation했다(§3.5, 그림 6). 문장부호로 끝나는 규칙이 단일 이득 중 가장 컸지만 토큰의 약 30%를 제거했다. 나머지 C4 필터를 모두 사용하면 약 7%를 제거하면서 문장부호 규칙만 쓴 경우보다 점수가 높아 FineWeb은 나머지를 채택했다. [[dolma]]는 C4 규칙에서 반대 선택을 해 문장부호 규칙인 “C4 NoPunc”만 유지했고, 이는 문자의 22.73%를 표시했다. 동시에 전체 Gopher 규칙 세트는 15.23%를 표시했다(§5.2, 부록 N.4).
- [[fineweb]]이 동일 crawl의 고품질·저품질 버전을 구분하는 통계에서 도출한 세 custom filter(§3.6): 구두점으로 끝나는 줄의 비율 ≤ 0.12(토큰 10.14% 제거), 중복 줄의 문자 비율 ≥ 0.1(12.47%), 길이 30자 미만 줄의 비율 ≥ 0.67(3.73%). 합치면 토큰 약 22%를 제거하고 aggregate 점수를 약 +1% 바꾼다.

같은 C4 규칙 세트를 두 팀이 반대 방식으로 채택했다는 사실 자체가 이 실습의 주제다. 두 결정 모두 서로 다른 pool과 평가 suite에서 ablation으로 뒷받침되었다.

### §3.3 정확한 부분 문자열 중복 제거

[[deduplicating-training-data]] §4.1: 모든 예시를 하나의 토큰 시퀀스로 이어 붙이고 suffix array를 만든다. 접두사가 최소 **50 BPE 토큰** 공유되는 인접 suffix-array 항목을 찾아 두 예시 중 하나에서 반복 부분 문자열을 제거한다. 임계값은 부록 B에서 나왔다. 10토큰보다 짧은 일치는 흔하고, 25토큰 일치 25개를 수동 검사했을 때 거짓 양성이 없었으며, 여유를 위해 저자들은 25를 50으로 두 배로 했다.

보고된 비용(부록 B): 96코어·768GB 기계 한 대에서 350GB C4 suffix array는 wall-clock 12시간 이내(약 1,000 CPU-hour)가 걸리고 1.5TB를 차지하며, 제거 pass는 1시간 이내다. 계획 전에 pool 크기에 맞춰 축소하라. 구현이 지켜야 할 규칙은 두 가지다. 제거는 반복 span의 두 발생 모두가 아니라 한 발생에만 적용한다. 또한 **cross-split 규칙**(§5)에 따라 텍스트가 둘 이상의 split에 있으면 validation 또는 test 복사본을 유지하고 training 복사본을 제거한다. split 경계를 넘는 중복 제거는 의도된 동작이며 오류가 아니다.

### §3.4 준중복 제거와 banding 산술

준중복 검출은 작은 sketch에서 Jaccard 유사도 `r_w(A,B) = |S(A,w) ∩ S(B,w)| / |S(A,w) ∪ S(B,w)|`를 추정한다. 여기서 `S(D,w)`는 문서 D의 w-token shingle 집합이고 w는 shingle 크기다([[minhash-lsh]] §2). [[minhash-lsh]] 정리 1은 무작위 permutation 하나에서 편향 없는 추정량을 준다. 최신 pipeline이 사용하는 *banding* 확률은 그 논문에 없으며 Leskovec, Rajaraman, Ullman의 *Mining of Massive Datasets* 3장 §3.4.2에서 도출된다([[minhash-lsh]], Connections).

Banding 확률을 다음처럼 쓴다.

```
P(candidate | s) = 1 − (1 − s^h)^B
```

여기서 `s`는 쌍의 Jaccard 유사도, `h`는 한 bucket 안에서 모두 일치해야 하는 hash 수, `B`는 bucket 수다. 공개 설정 두 가지는 다음과 같다.

| 설정 | 전체 hash | bucket `B` | bucket당 hash `h` | 위치 |
|---|---|---|---|---|
| [[fineweb]] | 112 | 14 | 8 | §3.4, 부록 E.1 |
| [[deduplicating-training-data]] NearDup | 9,000 | 450 | 20 | §4.2, 부록 A |

**계산 예시(손으로 확인).** FineWeb의 `h = 8`, `B = 14`에서 `s = 0.8`이면 `s^8 = 0.16777`, `(1 − 0.16777)^14 = 0.0765`이므로 `P = 0.9235`다. `s = 0.7`이면 `s^8 = 0.05765`, `P = 0.5645`다. `s = 0.75`에서는 `P = 0.7716`, `s = 0.85`에서는 `P = 0.9884`다. FineWeb은 이 네 유사도에서 정확히 56%, 77%, 92%, 98.8%를 인쇄하므로(부록 E.1) 산술이 공개 표를 재현한다.

이제 Lee 등의 `h = 20`, `B = 450`에서 같은 산술을 한다. `s = 0.8`이면 `s^20 = 0.01153`, `P = 0.9946`; `s = 0.7`이면 `P = 0.3018`; `s = 0.6`이면 `P = 0.0163`이다. FineWeb 설정에서 `s = 0.6`의 `P = 0.2111`이다. 450 × 20 설정은 0.8 부근에서 더 날카롭다. 0.6 유사 쌍의 1.6%를 표시하는 반면 14 × 8 설정은 21%를 표시한다. 중간 유사도에서의 recall은 (B, h) 선택이 정하며, 메모는 cell이 어느 설정을 사용했는지와 준중복 paraphrase recall에 미치는 의미를 밝혀야 한다.

[`figures/ablation-grid.html`](figures/ablation-grid.html)의 Panel B는 임의의 (B, h)에 대해 이 공식을 계산하고 네 FineWeb 값을 자신의 값 옆에 표시한다. 따라서 설정을 확정하기 전에 위 재현을 확인하고 곡선을 탐색할 수 있다.

[[deduplicating-training-data]]는 후보 단계에서 멈추지 않는다. Jaccard > 0.8 **그리고** edit similarity > 0.8일 때만 후보 쌍을 중복으로 본다. `EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j) / max(|x_i|, |x_j|)`다(§4.2). 중복 cluster는 결과 graph의 connected component다.

**어느 중복 제거 단계에서 이득을 기대할 것인가.** [[dclm]] 부록 L.2.1 표 17은 76B 토큰 RefinedWeb-filtered pool에서 1B-1x로 정확히 이 ablation을 실행했다. exact only는 13% 제거에서 CORE +1.3, MinHash only는 18%에서 +0.9, suffix array only는 33%에서 +1.9, Bloom filter only는 26%에서 +2.1, exact + MinHash + suffix array 세 가지는 41% 제거에서 +2.1이었다. 저자들은 suffix array와 MinHash 사이의 겉보기 차이가 “데이터셋 subsampling과 모델 학습의 비결정성에 따른 CORE 점수 분산 범위 안”이라고 말했다. 같은 이유로 단일 seed에서 자신의 차이가 약 1점이면 noise와 구분할 수 없다고 예상하라.

### §3.5 시험할 두 품질 필터

Cell 설계는 **broad** 필터와 **benchmark-targeted** 필터를 나란히 요구한다.

- Broad: [[dclm]]의 fastText classifier. OpenHermes-2.5와 점수가 높은 r/ExplainLikeImFive 게시물을 positive로, RefinedWeb 재현에서 샘플링한 것을 negative로 사용하고 문서 상위 10%를 유지한다. 7B-1x에서 CORE 41.0, MMLU 29.2, EXTENDED 21.4이며, Wikipedia positive로 학습한 동일 classifier의 35.7 / 27.0 / 19.1과 비교된다(표 5). 임계값이 중요하다. 상위 15%는 CORE 39.8, 상위 20%는 38.7이다.
- Benchmark-targeted: FineWeb-Edu 방식 classifier([[fineweb]] §4). 고정된 Snowflake-arctic-embed-m encoder 위 linear regression head이며, 0~5 척도의 Llama-3-70B-Instruct 교육 평점으로 학습하고 score ≥ 3을 유지한다(binary F1 82%). 전체 15T FineWeb 토큰에 적용하는 데 6,000 H100 GPU 시간이 걸렸으므로(§4) 학습 실행뿐 아니라 classifier pass도 예산에 포함하라.

두 필터가 모두 필요한 이유는 벤치마크 주제 쪽으로 학습한 classifier가 그 벤치마크를 두 메커니즘으로 높일 수 있기 때문이다. 유지한 문서가 모델에 없던 내용을 가르치거나, 내용을 추가하지 않고 학습 분포를 벤치마크 분포에 가깝게 옮길 수 있다. 첫 번째는 관련 없는 태스크로 전이하지만 두 번째는 전이하지 않으며, §5.4가 이를 측정한다.

## §4 Ablation 프로토콜

**Cell의 단일 변경 외에는 모두 고정한다.** 아키텍처, tokenizer, 시퀀스 길이, optimizer, 학습률 schedule, 초기화 seed set, 평가 harness를 grid 전체에서 고정한다. [[fineweb]]의 프로토콜(§3.1)을 기준으로 삼는다. 데이터만 다르고 나머지가 동일한 모델을 데이터 버전마다 서로 다른 데이터 부분집합과 seed로 두 번 실행하고 점수를 평균한다.

**Cell당 토큰을 같게 하고, 여기서 생기는 반복 교란을 관리한다.** 각 cell은 같은 수의 학습 토큰을 봐야 한다. 그렇지 않으면 비교가 예산도 함께 측정하기 때문이다. pool의 60%를 제거하는 필터는 cell이 데이터를 반복하게 만든다. [[data-constrained-scaling]]은 GPT-2 아키텍처 모델에서 반복 비용을 측정했다. 약 4 epoch까지 홀드아웃 손실 불이익은 무시할 수 있다(8.7B 모델의 4 epoch 종료 손실은 1 epoch 손실보다 0.5% 높음). 적합 상수 R*_D ≈ 15.4 반복으로 가치가 감소하고, downstream 점수는 약 4 epoch 후 하락하기 시작한다(초록, §6, 적합 상수는 부록 A, downstream 결과는 부록 L과 그림 6). 같은 논문은 이 실습에 필요한 필터 후 반복 설정도 실행했다(§7, 부록 N). 4.2B 파라미터와 84B 토큰에서 perplexity 필터는 perplexity가 가장 낮은 샘플의 25%를 유지해 고유 토큰 44B를 약 2 epoch 반복하게 했고, 100자 span 중복 제거는 21B 토큰을 4 epoch 반복하게 했다.

운영 규칙은 다음과 같다. 가장 공격적인 필터를 포함한 **모든** cell이 4 epoch 이하가 되도록 pool 크기를 정한다. 불가능하면 메모에 cell별 epoch을 열로 보고하고 공격적인 cell이 교란되어 있음을 명시해야 한다.

**필터가 유지한 것뿐 아니라 제거한 것도 보고한다.** [[fineweb]] §3.4는 snapshot 2013-48에서 이 실험을 수행했다. 전역 중복 제거가 유지한 약 31B 토큰으로 학습한 모델은, 제거된 약 460B 토큰만 중복 제거해 얻은 171B 토큰으로 학습한 모델보다 *나빴다*. 96개 snapshot 전체에서 전역 중복 제거는 이전 snapshot의 최대 90%를 제거하고 4T 토큰을 남겼으며, 그 데이터의 350B 토큰 실행은 중복 제거하지 않은 데이터보다 조금만 나아졌다. Snapshot별 중복 제거는 20T 토큰을 만들어 RefinedWeb과 일치했다(§3.4, 그림 5).

**Cell.** 방어 가능한 8-cell grid는 다음과 같다.

| Cell | 데이터 | 분리하는 것 |
|---|---|---|
| A | 원시 추출, 필터 없음 | 기준 |
| B | A + language ID | 언어 비율 효과 |
| C | B + 휴리스틱 필터(C4 규칙 + FineWeb custom 3개) | 규칙 기반 품질 |
| D | C + 정확한 부분 문자열 중복 제거(≥ 50토큰) | 그대로 반복 |
| E | D + MinHash 준중복 제거 | 준중복 반복 |
| F | E + broad classifier(fastText OH-2.5 + ELI5, 상위 10%) | 범용 모델 기반 필터 |
| G | E + benchmark-targeted classifier(교육 점수 ≥ 3) | 대상 모델 기반 필터 |
| H | E + 주제·형식 라벨에 대한 도메인별 quota 혼합 | 고정 필터에서 혼합 |

Cell H가 있는 이유는 [[weborganizer]]가 품질 필터 위에 도메인별 토큰 quota를 설정하면 1.44B 파라미터에서 FineWeb-Edu 9태스크 평균에 2.0점(54.2 → 56.2), DCLM-fastText에 1.0점(55.1 → 56.1)을 추가한다는 것을 발견했기 때문이다(표 1). 따라서 필터링과 혼합은 서로 바꿔 쓸 수 없다.

## §5 범위 측정: 세 계열 모두 필수

### §5.1 도메인별 홀드아웃 perplexity

[[paloma]]를 따른다. 단일 pool perplexity가 아니라 도메인에 대한 **macro average** `|D|⁻¹ Σ_{d∈D} perplexity(d)`와 최악의 도메인을 보고한다. Pool perplexity는 토큰으로 가중한 micro average라 토큰이 적은 도메인을 숨긴다(§4.1). 범용 모델이 포괄해야 할 공간을 가로지르는 부분집합을 사용한다—C4-100-domains, RedPajama 7개 도메인, M2D2 S2ORC, M2D2 Wikipedia, Dolma-100-programming-languages, Dolma-100-subreddits, WikiText-103, Penn Treebank, Twitter AAE(표 1).

논문에서 제시한 두 프로토콜 규칙은 수치를 모두 바꾼다.

- **G5, 문서를 따로 평가한다.** 각 문서는 `<BOS>` 뒤에 독립적으로 점수화한다. 입력을 이어 붙이면 분산 추세가 깨진다. 4M 평가 토큰에서 2B 토큰을 학습한 Pythia 1.4B는 이어 붙인 경우 92.23 ± 17.33, 분리한 경우 42.57 ± 0.29를 준다(표 17).
- **G1, perplexity를 읽기 전에 평가 세트에 대해 학습 데이터를 오염 제거한다.** Paloma 자체의 삭제율은 Dolma 0.062%, RedPajama 0.099%, The Pile 2.753%, Falcon RefinedWeb 0.733%, C4 0.010%, mC4-en 0.002%다(표 4).

이 계열이 검출하는 것은 태스크 평균은 올랐지만 하나 이상의 도메인이 나빠진 cell이다. 공개된 극단 사례는 C4-only 1B 기준선이다. RedPajama arXiv perplexity 391,171, Dolma peS2o 14다(§2, §4.1). [[dolma]]도 150B 토큰의 1.2B에서 같은 패턴을 찾았다. 단일 원천 웹 말뭉치(C4, mC4-en, RefinedWeb)는 선별된 비웹 원천을 가진 말뭉치보다 평균 Paloma perplexity가 높았다(§9.2, 그림 5).

**필터링하지 않은 기준 pool**에 대한 perplexity도 기록한다. [[weborganizer]]는 base 12.1, FineWeb-Edu 14.7, DCLM-fastText 14.0, 암묵적 도메인 혼합 12.2~12.9를 보고하며(표 2), 그 차이를 도메인 재균형이 아닌 필터의 분포 변화로 해석한다(§5; 저자들의 해석).

### §5.2 이 규모에서 신호를 전달하는 태스크와 seed 분산

400M~1B 규모의 태스크 선택에는 공개된 제약 세 가지가 있다.

1. [[dclm]]의 CORE는 “작은 규모에서도 분산이 낮은 신호를 제공”하도록 선택한 22태스크 부분집합이고, 각 태스크를 0이 무작위 추측이 되도록 재조정했다(§3.5). 그래도 [[weborganizer]] 부록 E는 일부 DCLM CORE 태스크가 1b-1x에서 거의 무작위라고 말한다.
2. [[fineweb]]은 낮은 seed 분산, 학습 중 거의 단조로운 향상, 1.71B에서 무작위보다 높은 점수를 기준으로 벤치마크를 선택했다(§3.1).
3. [[datadecide]]는 작은 계산량으로 어느 태스크가 올바르게 결정되는지 측정했다. ARC-Easy는 계산량 5자릿수보다 적어도 예측 가능하고, HellaSwag·SocialIQA·WinoGrande는 계산 임계값 전까지 둔감하며, BoolQ는 대상 실행의 중간 checkpoint에서만 trivial decision accuracy를 초과한다(§3.1, 그림 2).

**태스크 선택보다 metric 선택이 답을 더 크게 바꾼다.** [[signal-and-noise-eval]]은 noise를 한 실행의 마지막 `n` checkpoint에 대한 상대 표준편차 `sqrt((1/(n−1)) Σ_i (m_i − m̄)²) / m̄`로 정의하고, signal을 비슷하게 학습된 모델 모집단의 상대 dispersion `max_{j,k} |m_j − m_k| / m̄`로 정의하며, SNR을 둘의 비율로 정의한다(§3, 식 2). 30개 벤치마크에서 primary metric을 bits-per-byte(정답의 negative log-likelihood를 UTF-8 byte 수로 나눈 것)로 바꾸면 평균 SNR이 10.0에서 31.5로, decision accuracy가 77.0%에서 83.7%로 올랐다. 태스크별로 Minerva MATH는 51.0 → 90.0, HellaSwag은 74.3 → 95.3, MMLU는 89.0 → 92.0이다(§5.3, 그림 6). 마지막 checkpoint 하나 대신 마지막 checkpoint들을 평균하면 30태스크 decision accuracy가 2.4점 추가된다(68.9% → 71.3%, 표 1). 같은 연구는 OLMo 2 7B에서 마지막 checkpoint `n = 5`개가 거의 모든 벤치마크에서 표본 표준편차를 실제 noise의 ±1σ 안에 놓는다고 보고한다(부록 A.3.2).

**필수 보고.** Cell별·태스크별로 score, bits-per-byte score, 최소 두 seed에 대한 spread를 보고한다. seed spread보다 작은 차이는 신호로 보고하지 말고 “이 규모에서 해상되지 않음”으로 보고한다.

### §5.3 분포 비율

각 cell에서 *유지한* 문서에 대해 세 가지 비율 표를 보고한다.

1. 언어 식별 pass의 **언어 비율**.
2. **주제 및 형식 비율.** [[weborganizer]]는 Llama-3.1-405B-Instruct annotation에서 distill한 24개 주제와 24개 형식 범주에 대해 140M-parameter classifier(gte-base-en-v1.5)를 공개했다. 405B annotator가 75% 이상 확신하는 validation page에서 주제는 평균 93.5 / 최악 집단 87.1, 형식은 91.8 / 80.5 정확도다(표 7). 두 taxonomy 사이 normalized mutual information은 약 0.10이므로 §2.3에서 거의 독립적인 축이다.
3. §3.2에서 이미 속성으로 계산한 **길이와 형식 proxy**(구두점으로 끝나는 줄, 중복 줄 비율, 짧은 줄 비율)를 pass rate가 아닌 분포로 보고한다.

이 계열이 검출하는 것은 점수 변화 뒤의 메커니즘이다. [[fineweb]]은 50k + 50k embedding sample의 cluster로 자체 classifier의 주제 변화를 측정했다. FineWeb-Edu는 “Education, Learning, Teaching”을 +3.2%, “History, Culture, Politics”를 +2.2% 늘리고 “Business, Finance, Law”, “Entertainment, Film, Theater”, “Places, Travel, Real Estate”를 down-sample한다(§4.1, 그림 18).

### §5.4 설계하지 않은 태스크로 대상 필터 전이

Cell G의 classifier는 교육 평점으로 학습한다. classifier 설계에 사용하지 않은 태스크에서 cell G를 평가하고 별도 열로 보고한다. 공개된 결과 형태는 다음과 같다. [[weborganizer]] 표 10에서 MMLU에 맞춘 Topic × Format 혼합은 MMLU 33.2에 도달하지만 HellaSwag은 57.5에서 54.1로, PIQA는 71.3에서 69.9로 하락한다. 표 2는 암묵적 도메인 혼합만으로 FineWeb-Edu 태스크 평균 이득의 84%를 회복하지만 DCLM-fastText 이득의 35%만 회복한다고 보여준다. 즉 FineWeb-Edu 이득 대부분은 도메인 재균형으로 재현되고, DCLM-fastText 이득 대부분은 재현되지 않는다.

## §6 Paraphrase를 포함한 오염

모든 cell이 하나의 평가 suite를 공유하므로 오염은 전체 grid를 한 번에 무효화할 수 있는 교란이다. 비용이 증가하는 세 수준은 다음과 같다.

1. **문단 또는 n-gram 정확 일치.** [[dolma]]는 Paloma에 나타나는 13토큰 초과 문단을 포함한 문서를 제거한다(부록 L, 문서의 ≤ 0.02%). [[paloma]] G1은 줄바꿈으로 분리된 문단에 Bloom filter를 사용하며, 13 Unicode-segmented 토큰 미만 또는 구두점·공백·emoji만으로 된 문단은 무시한다(부록 C.1.1). 저자들은 이 규칙이 길고 자주 인용되는 문서를 제거한다고 말하며 주의를 권한다.
2. **질문과 선택지의 일치.** [[dclm]] §4.6은 질문 텍스트와 최소 하나의 답 선택지를 포함한 페이지를 표시한다. 긴 MMLU 질문에서는 recall을 높이기 위해 마지막 문장만 검출하며, 이것이 “여전히 많은 거짓 양성을 낳는다”고 명시한다. 7B-2x에서 검출된 겹침을 제거해도 점수가 낮아지지 않았다(MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4, 표 7). 이는 그들의 이득이 오염에서 나온 것이 아님을 보여주는 근거다.
3. **Paraphrase 인식.** [[openmathinstruct-2]] §3.1은 embedding 유사도(multi-qa-MiniLM-L6-cos-v1)로 테스트 질문 상위 5개를 검색한 뒤 Llama-3.1-405B-Instruct에 양방향 paraphrase 동등성을 판단하게 한다. 질문당 10회 호출하며 새 질문 569K가 삭제 후 519K가 되었다. 부록 C.2는 유사하지만 동등하지 않은 예시도 표 11에 유지한다.

평가 세트별로 level 1에서 삭제한 문서, level 2에서 추가로 삭제한 문서, level 3에서 추가로 삭제한 문서를 보고하라. 출처 자체의 설명에 따라 level 2와 3에는 거짓 양성이 있으므로 삭제 수만 보고하지 말고 개수, 손라벨링에서 표본 추출한 거짓 양성 비율, 삭제 후 점수 변화를 보고한다. 비교할 기준율은 [[deduplicating-training-data]]가 제공한다. C4 validation 예시의 4.60%, LM1B validation 예시의 4.92%가 학습 데이터에 준중복을 가진다(표 2). 오염 제거 방법의 자세한 내용은 [[ch-14]]에 있다.

## §7 순위는 전이되는가?

두 연구가 답의 범위를 정하며, 메모는 둘 중 하나를 인용해야 한다.

- [[dclm]] §3.2: 10개 curation 방법을 세 가지 작은 규모 각각에서 7B-1x와 비교해 순위를 매겼다. Pearson r은 400M-1x에서 0.838, 1B-1x에서 0.956, 3B-1x에서 0.982다. 작은 규모가 클수록 순위를 더 잘 매긴다.
- [[datadecide]] §3.2: 25개 말뭉치, 14개 크기, 3개 seed. 150M 같은 단일 작은 규모의 순위는 쌍별 비교 약 80%에서 1B 승자를 예측한다. 8개 scaling-law 기준선 중 어느 것도 단일 규모 순위의 compute-to-decision-accuracy frontier를 넘지 못했다. Decision accuracy는 `(1/|P|) Σ_{(A,B)∈P} 1[sign(ŷ_A − ŷ_B) = sign(y_A − y_B)]`이며, 여기서 `P`는 말뭉치 쌍 집합, `y`는 3개 seed 평균의 관측 1B 결과, `ŷ`는 예측이다(식 3).

**계산 예시.** 8-cell grid에는 순위를 매길 서로 다른 cell 쌍이 `C(8,2) = 28`개 있다. Decision accuracy가 80%라면 대상 규모에서 약 28개 쌍별 순서 중 5.6개가 틀릴 것으로 예상한다. 메모의 권고가 하나의 쌍별 차이에 의존한다면, 그 권고가 뒤집힐 확률이다.

**출처가 명시한 조건과 한계.** [[datadecide]]는 하나의 토큰-파라미터 비율(100), 최대 1B까지 14개 설정, 객관식 cloze 태스크만 사용했다. 또한 작은 규모와 대상 규모 사이에서 관측된 scaling 추세가 자주 교차하며, 진짜 crossover를 noise와 구분하기 어렵다고 보고한다(§3.2, §5). [[dclm]]의 상관은 하나의 대상 규모에서 10개 방법에 대한 것이다. 따라서 메모는 보장이 아니라 범위를 진술한다.

## §8 Ablation 메모

메모는 이 장의 산출물이다. 표를 나열한 문서가 아니며 다음 일곱 부분을 담으면 완성된다.

1. **설정.** Pool 원천과 크기, extractor, tokenizer, 모델 설정, cell당 토큰, cell당 seed, 평가 harness와 버전, cell당 측정 wall-clock 시간을 적는다.
2. **Cell 표.** §4의 cell A~H마다 한 행을 둔다. 포함할 항목은 유지한 토큰, 해당 단계의 제거율, 유지 pool에서의 epoch, 태스크 평균과 seed spread, bits-per-byte 점수와 seed spread, macro 도메인별 perplexity, 최악 도메인 perplexity, 필터링하지 않은 기준 pool의 perplexity다.
3. **비율 표.** Cell별 언어·주제·형식 비율과 cell A에서의 변화를 담는다.
4. **홀드아웃 전이 열.** Cell G에서 classifier 설계에 역할이 없었던 태스크의 점수를 설계에 사용한 태스크와 분리해 보고한다.
5. **오염 보고.** §6의 세 수준 각각에서 삭제한 개수, level 2와 3의 손라벨 거짓 양성 비율, 삭제 후 점수 변화를 기록한다.
6. **전이 진술.** 의존한 공개 결과([[dclm]] §3.2 또는 [[datadecide]] §3.2), 사용한 grid 크기에서 순서가 틀릴 것으로 예상되는 쌍의 수, 그 결과 권고에 대한 신뢰도를 적는다. 이 문단 없이 cell을 권고하는 메모는 불완전하다.
7. **범위를 좁힌 Cell.** 태스크 평균은 올랐지만 macro 도메인별 perplexity가 나빠졌거나, 비율 표에서 어떤 범주가 seed spread보다 크게 이동한 모든 cell을 명시한다. 그런 cell이 없으면 없다고 쓰고 측정 해상도를 제시한다. 0.5점 해상도의 null 결과는 0.05 해상도의 null 결과와 다른 주장이다.

이 장의 근거에서 따르는 금지 사항 두 가지가 있다. Seed spread보다 작은 차이는 [[dclm]] 부록 L.2.1처럼 절대 부호로 쓰지 말고 “해상되지 않음”으로 쓴다. 더 큰 실행에 대한 권고는 단순한 순위가 아니라 전이 근거와 오류율을 함께 적는다([[datadecide]] §3.2).

## 부정 샘플과 부정 피드백

이 실습은 과정 표준 §6.1의 네 가지 의미 중 (1)의 부정을 사용한다. **음의 주변 가치**—학습 표적으로 포함하면 성능을 낮춘다고 여겨 삭제한 문서다. 삭제된 문서에는 gradient를 적용하지 않으며 아래로 밀어내지도 않는다. (2), (3), (4)의 의미는 이 단계에 나타나지 않는다.

- **부정의 출처.** 휴리스틱 속성, classifier 점수, 또는 중복 cluster 포함 여부에 대한 임계값이다. 라벨은 인간이 검증하지 않는다. 여기서 보고된 일치도 수치가 있는 유일한 경우는 [[fineweb]] classifier다. held-out Llama-3-70B-Instruct annotation을 기준으로 임계값 3에서 binary F1 82%다(§4). 논문은 임계값 근처 문서의 오류율을 보고하지 않으므로 이 필터의 거짓 음성율도 보고되지 않았다.
- **관행이 처리하는 방식.** 버린다. 그래서 §4의 *유지 대 삭제* 실험이 필요하다. [[fineweb]] §3.4는 snapshot 2013-48에서 삭제된 절반으로 학습한 모델이 유지된 절반보다 더 좋다는 것을 발견했으며, 이는 필터가 음의 주변 가치를 잘못 부여한 것을 직접 측정한다.
- **효과 크기의 근거.** [[dclm]] 표 17은 1B-1x에서 토큰 13~41%를 제거한 중복 제거 설정에 대해 CORE +0.3~+2.1을 준다. 저자들은 약 1점 차이가 실행 간 분산 범위 안에 있다고 직접 적었다. [[data-constrained-scaling]] §7은 perplexity filtering이 C4에서는 도움이 되지만 중복 제거는 C4 downstream 결과를 높이지 못하며, 둘 다 더 잡음이 많은 OSCAR에서 더 효과적이라고 보고한다. 버림의 가치는 pool의 잡음 정도에 의존한다.
- **제어.** 버린 세트만으로 하나의 cell을 학습한다. 버린 세트에 과대표집된 도메인에서 유지 세트 모델의 도메인별 perplexity를 보고한다. pipeline을 다시 실행하지 않고 임계값을 옮길 수 있도록 속성 파일을 유지한다.
- **진단.** 단계별 삭제율, 도메인별 삭제율, 언어 비율 변화, 버린 세트 모델의 점수다.
- **범용성에 미치는 효과.** [[ccnet]] §5.2는 구체적인 위험을 명시한다. perplexity 기반 삭제는 말하듯 쓰인 텍스트와 전문용어 포럼을 tail로 보내며, 저자들은 이런 이유로 tail을 삭제하지 않고 bucket을 공개하기로 했다.

## 레시피

공개 행에서 도출한 실습 설정이다. `실습 값`은 이 장이 처방하는 값이고, `차이의 이유`는 출처와 달라지는 이유다.

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 | 실습 값 | 차이의 이유 |
|---|---|---|---|---|---|---|---|---|---|
| DCLM 400M-1x 기준 실행 | 412M | pretrain-stable | 학습 토큰; 학습 FLOPs; H100 시간 | 8.2B; 2.0e19; 26 | arXiv:2406.11794v4 §3.2 표 1 | 2026-09-17 검증 | ablation 보고 없음; 규모는 20 × N × multiplier로 설정 | 동일(전체 경로): cell당 412M / 8.2B | 없음; 전체 예산 cell로 변경 없이 채택 |
| WebOrganizer RegMix proxy model | 50M | pretrain(proxy) | 실행당 토큰; 계산량 | 1B; 512회 실행에 약 360 H100 시간(0.70 h/run) | arXiv:2502.10341v3 §3, 부록 C | 2026-09-17 검증 | 표 8: 50개 홀드아웃 혼합에서 Spearman ≈ 0.90 | 동일(제약 경로): cell당 50M / 1B | 없음; 제약 cell로 변경 없이 채택 |
| WebOrganizer RegMix proxy model | 50M | pretrain(proxy) | 아키텍처 | hidden 512, intermediate 1536, SwiGLU, head 8개, block 8개, RoPE base 10000 | arXiv:2502.10341v3 부록 C 표 6 | 2026-09-17 검증 | ablation 보고 없음 | 동일 | 없음 |
| WebOrganizer RegMix proxy model | 50M | pretrain(proxy) | optimizer; peak LR; cooldown; warmup; batch | Adam β (0.9, 0.95); 3e-3; 3e-4까지 cosine; 10%; 128 | arXiv:2502.10341v3 부록 C 표 6 | 2026-09-17 검증 | ablation 보고 없음 | 동일, 모든 cell에서 고정 | 없음; 고정은 ablation 요구사항 |
| FineWeb data-ablation model | 1.71B | eval-gate | 데이터 버전당 seed | 2(서로 다른 데이터 부분집합 및 초기화), 평균 | arXiv:2406.17557v2 §3.1 | 2026-09-17 검증 | ablation 보고 없음 | cell당 최소 2 seed | 없음 |
| FineWeb data-ablation model | 1.71B | pretrain-stable | ablation 실행당 토큰 | 약 28B(필터 ablation) | arXiv:2406.17557v2 §3.1 | 2026-09-17 검증 | 1.71B에 대략 Chinchilla 최적이라고 설명 | 8.2B(전체) / 1B(제약) | 더 작은 모델; 토큰 예산은 cell 자체의 Chinchilla 기준을 따름 |
| FineWeb MinHash | — | data dedup | hash; bucket; bucket당 hash; shingle | 112; 14; 8; 5-gram | arXiv:2406.17557v2 §3.4, 부록 E.1 | 2026-09-17 검증 | 부록 E.1은 s = 0.70/0.75/0.80/0.85에서 P = 56/77/92/98.8%를 표시 | 동일 | 없음 |
| Lee et al. NearDup | — | data dedup | hash; bucket; bucket당 hash; 임계값 | 9,000; 450; 20; Jaccard > 0.8 및 edit similarity > 0.8 | arXiv:2107.06499v2 §4.2, 부록 A | 2026-09-17 검증 | 0.8 및 0.9 유사도 histogram이 같은 지점에서 사라짐(부록 A, 그림 4) | 그림의 두 번째 preset으로 제공 | 실습 pool 크기에서 문서당 9,000 hash는 비용이 큼 |
| Lee et al. ExactSubstr | — | data dedup | 최소 일치 길이 | 50 BPE 토큰 | arXiv:2107.06499v2 §4.1.2, 부록 B | 2026-09-17 검증 | 수동 검사에서 25토큰에 거짓 양성 없음; 여유를 위해 두 배 | 동일 | 없음 |
| DCLM fastText filter | — | data filter | positive; 임계값 | OpenHermes-2.5 + ELI5 게시물, RefinedWeb 재현에서 샘플링한 negative; 상위 10% | arXiv:2406.11794v4 §4.4, 표 5 | 2026-09-17 검증 | 7B-1x 표 5: 상위 10% CORE 41.0 대 15% 39.8, 20% 38.7 | 동일, cell F로 | 없음 |
| FineWeb-Edu classifier | 보고되지 않음 | data filter | 점수 임계값 | ≥ 3 | arXiv:2406.17557v2 §4, 부록 F.2 | 2026-09-17 검증 | 28B 토큰에서 FW-Edu-2/3/4; 3이 aggregate에서 최고이며 HellaSwag trade-off | 동일, cell G로 | 없음 |
| Dolma web pipeline | — | data pipeline | 단계 순서 | CCNet 출력 → URL dedup → 문서 dedup → quality → content → 문단 dedup | arXiv:2402.00159v2 §5.5 | 2026-09-17 검증 | 논문은 필터 순서 자체를 ablation하지 않음 | 동일 순서 | 없음; 출처가 순서를 ablation하지 않았음을 명시 |
| Paloma evaluation | 1B 기준선 | eval-gate | 평가 형식 | 각 문서를 `<BOS>` 뒤에서 따로 점수화 | arXiv:2312.10523v2 §3 G5, 부록 H | 2026-09-17 검증 | 표 17: 이어 붙임 92.23 ± 17.33 대 분리 42.57 ± 0.29 | 동일 | 없음 |
| Signal-and-Noise checkpoint rule | 60M~750M | eval-gate | noise window; metric | 마지막 5 checkpoint 평균; bits-per-byte | arXiv:2508.13144v1 §4.1, §5.3 | 2026-09-17 검증 | 표 1: decision accuracy +2.4점; 그림 6: 77.0% → 83.7% | 동일 | 없음 |

**소규모 범용 실행의 시작점.** 제약 경로: 파라미터 50M(hidden 512, intermediate 1536, head 8개, block 8개, SwiGLU, RoPE base 10000), cell당 1B 토큰, Adam β (0.9, 0.95), peak LR 3e-3에서 3e-4까지 cosine cooldown, warmup 10%, batch 128을 사용한다. 모든 값은 H100에서 각각 1B 토큰으로 학습한 50M proxy model에 사용된 [[weborganizer]] 부록 C 표 6에서 왔다. Cell당 seed 2개를 사용하고 마지막 5 checkpoint 평균의 bits-per-byte로 점수화한다([[signal-and-noise-eval]] §4.1, §5.3). 어떤 cell도 4 epoch를 넘지 않도록 pool 크기를 정한다([[data-constrained-scaling]] §6).

## 일반화 관점

**(a) 폭을 넓히는 것.**
- 필터링한 웹 텍스트와 함께 선별된 비웹 원천을 사용한다. 150B 토큰에서 1.2B를 학습할 때 단일 원천 웹 말뭉치는 Dolma 또는 The Pile보다 평균 Paloma perplexity가 높았다([[dolma]] §9.2, 그림 5).
- 품질 필터 위에 도메인별 quota를 둔다. 1.44B에서 FineWeb-Edu 9태스크 평균에 +2.0점, DCLM-fastText에 +1.0점을 준다([[weborganizer]] 표 1).
- 중간 정도의 삭제율로 중복 제거한다. 1B-1x에서 토큰 13~41%를 제거한 8개 중복 제거 설정의 CORE가 +0.3~+2.1이었다([[dclm]] 표 17). Perplexity를 악화시키지 않으면서 그대로 암기를 약 10배 줄였다([[deduplicating-training-data]] 표 4: 1.926% → 0.189% NearDup, 0.138% ExactSubstr).

**(b) 폭을 좁히는 것.**
- 벤치마크 대상 혼합. MMLU 대상 선택은 HellaSwag을 57.5 → 54.1, PIQA를 71.3 → 69.9로 낮췄다([[weborganizer]] 표 10).
- 기준과의 유사도에 기반한 perplexity 필터. 말하듯 쓰인 텍스트와 전문용어가 많은 텍스트가 삭제 tail에 들어간다([[ccnet]] §5.2).
- 공격적인 전역 중복 제거. Snapshot 2013-48에서 유지한 데이터가 삭제한 데이터보다 나빴다([[fineweb]] §3.4).
- 하나의 aggregate를 최대화하도록 임계값 설정. FineWeb은 HellaSwag류 벤치마크와의 trade-off 때문에 임계값 4가 아닌 3을 선택했다([[fineweb]] §4).

**(c) 이 단계에서 측정하는 방법.**
- 오염 제거한 말뭉치에서 문서를 따로 점수화하고, 도메인별 macro 평균 perplexity와 최악 도메인을 사용한다([[paloma]] §3, §4.1).
- 분포 변화의 판독값으로 *필터링하지 않은* 기준 pool의 홀드아웃 perplexity를 사용한다([[weborganizer]] §5).
- 신호를 담은 태스크를 seed spread와 함께 bits-per-byte로 점수화하고, seed spread보다 작은 차이는 해상되지 않음으로 보고한다([[signal-and-noise-eval]] §5.3; [[dclm]] 부록 L.2.1).
- 유지 세트의 주제·형식·언어 비율을 측정한다([[weborganizer]] §2.2; [[fineweb]] §4.1).
- 출처와 오류율이 포함된 전이 진술을 작성한다([[dclm]] §3.2; [[datadecide]] §3.2).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| Cell당 토큰 수가 다름 | 공격적인 필터 cell이 더 좋아 보이고 optimizer step도 더 많이 실행함 | cell별 본 토큰과 step을 기록하고 점수화 전에 동일성을 assert |
| 같은 토큰 안에 반복 교란이 숨음 | 공격적인 cell이 자체 유지 pool에서 4 epoch을 초과 | cell별 `tokens_per_cell / unique_tokens_kept`를 계산해 열로 보고([[data-constrained-scaling]] §6) |
| 단일 seed로 부호를 해석 | seed를 바꾸면 약 1점의 보고 delta가 뒤집힘 | 최소 2 seed, spread 보고, spread보다 작은 차이는 해상되지 않음으로 처리([[dclm]] 부록 L.2.1) |
| 벤치마크가 우연 수준 | Cell 점수가 무작위 기준선 근처에 모이고 학습과 함께 움직이지 않음 | 무작위에 맞춰 재조정한 CORE 형식을 사용하고 계속 우연 수준인 태스크를 제거([[dclm]] §3.5; [[weborganizer]] 부록 E) |
| Aggregate만 보고 | 태스크 평균은 오르지만 어느 도메인이 나빠졌는지 표시되지 않음 | macro 도메인별 perplexity와 최악 도메인 열([[paloma]] §4.1) |
| 대상 classifier를 자체 대상에서 평가 | 대상 태스크에서만 큰 이득 | classifier 설계에 사용하지 않은 태스크를 홀드아웃하고 별도 보고([[weborganizer]] 표 10) |
| Perplexity를 이어 붙여 평가 | 같은 checkpoint 실행 사이 perplexity 수치가 수십씩 변함 | `<BOS>` 뒤에서 각 문서를 따로 점수화([[paloma]] 부록 H, 표 17) |
| 오염이 없다고 가정 | pool에 텍스트가 있는 바로 그 태스크에서 점수가 상승 | §6의 level 1~3을 실행하고 [[dclm]] 표 7처럼 제거 후 점수 변화를 보고 |
| Banding parameter를 하나의 수치로 보고 | bucket 없이 “112 hash MinHash”라고만 씀 | (전체 hash, bucket `B`, bucket당 hash `h`)와 s = 0.6/0.7/0.8에서 `P(s)`를 보고 |
| 필터 순서가 검증됐다고 가정 | 논문의 단계 순서를 복사하고 근거로 인용 | [[dolma]]는 필터 순서 자체를 ablation하지 않았다고 명시한다(§5.5). 결과가 아닌 선택으로 인용 |

## 이해도 확인

1. 두 cell이 필터 하나로 다르다. Cell F는 태스크 평균에서 1.1점 높고 macro 도메인별 perplexity에서 0.8 높다(더 나쁨). 두 수치 중 어느 것이 필터의 능력 추가로, 어느 것이 분포 변화로 생길 수 있는지, 둘을 구분하는 세 번째 측정은 무엇인지 설명하라.
2. 어떤 cell이 pool의 70%를 제거한 뒤 기준 cell과 같은 토큰 예산으로 학습된다. epoch 수를 도출하고, [[data-constrained-scaling]] 결과를 사용해 비교가 더 이상 필터 비교가 아닌 이유를 설명하라.
3. `P = 1 − (1 − s^h)^B`를 사용해 (B = 450, h = 20)이 s = 0.6에서 쌍의 1.6%를 표시하지만 (B = 14, h = 8)은 21%를 표시하는 이유를 설명하라. pool에 기계가 paraphrase한 spam이 있을 때 둘 중 어느 것을 더 안전한 기본값으로 삼을지 말하라.
4. [[fineweb]]은 snapshot 2013-48의 *삭제된* 절반으로 학습한 모델이 유지된 절반보다 낫다는 것을 발견했다. 이것이 삭제 기반 필터의 양의 ablation delta 해석에 의미하는 바를 설명하고 이를 검증하는 추가 cell 하나를 설계하라.
5. [[datadecide]]는 150M에서 1B까지 쌍별 decision accuracy 약 80%를 보고한다. 8-cell grid에서 대상 규모의 예상 오순위 쌍 수를 계산하고, 단일 cell을 권고하는 메모에 이것이 의미하는 바를 설명하라.
6. [[signal-and-noise-eval]]은 모델을 바꾸지 않고 metric을 bits-per-byte로 바꿔 30태스크 decision accuracy를 77.0%에서 83.7%로 높였다. signal과 noise 정의를 사용해 metric에서 decision accuracy로 이어지는 인과 경로를 설명하라.
7. [[dclm]]은 검출된 MMLU·HellaSwag 겹침을 삭제해도 7B-2x에서 점수가 낮아지지 않음을 발견했다. 이것이 오염에 대한 근거인 이유와 pool에 오염이 없다는 증명이 아닌 이유를 설명하라.
8. Cell G의 교육 classifier는 MMLU를 높이고 HellaSwag을 낮춘다. 필터가 지식을 추가했다는 해석과 분포를 좁혔다는 해석을 모두 주장하고, 둘을 결정하는 측정을 말하라.

## 연결

- **이전 장 — [[ch-14a]]: 사전학습 레시피 나란히 비교: 예산, 배치, 스케줄, 단계 혼합.** 이 실습의 모든 cell이 공유하는 고정 학습 설정을 제공한다.
- **의존성 — [[ch-14a]]**가 유일하게 선언된 의존성이다. pipeline 단계는 [[ch-10]](휴리스틱 선별 pipeline: CCNet, C4, Dolma, FineWeb), [[ch-10a]](모델 기반 품질 필터링과 벤치마크 대상 데이터 선택), [[ch-12]](정확·준중복·의미 중복 제거), [[ch-13]](도메인 혼합: DoReMi, 혼합 법칙, 규모별 검증), [[ch-14]](데이터 제약 scaling, 반복, 사전학습 오염 제거)에서 온다.
- **다음 장 — [[ch-32]]: 중간 학습: annealing 데이터, 단계 게이트, 후속 SFT·RL 효과.** 여기서 만든 측정 규율을 annealing 혼합에 적용한다. 훨씬 작은 토큰 예산에서 같은 범위 축소 위험이 나타난다.

## 출처

- [[dclm]] — DataComp-LM 장 발췌(arXiv:2406.11794 v4): competition 규모와 6ND FLOP 열, CORE 22태스크 정의, 규모 간 순위 상관, 필터·임계값 표, 중복 제거 ablation 표, 오염 제거 결과.
- [[fineweb]] — 1.71B ablation 프로토콜, C4·custom filter 삭제율, 인쇄된 `P(s)` 값이 있는 MinHash banding parameter, snapshot별 대 전역 중복 제거 결과, FineWeb-Edu classifier·임계값·주제 변화.
- [[paloma]] — 범위 metric으로서 도메인별 perplexity: macro 평균, 문서별 평가, 오염 제거율, C4 기준선 도메인 급증.
- [[datadecide]] — 25개 말뭉치에서 150M~1B 단일 규모 순위의 decision accuracy, 태스크별 예측 가능성, 전이의 명시된 한계.
- [[signal-and-noise-eval]] — signal·noise·SNR 정의, bits-per-byte 및 checkpoint 평균 개입, 마지막 checkpoint `n` 선택.
- [[ccnet]] — 언어 ID 전 중복 제거 순서, 언어 점수 임계값, perplexity tercile가 품질 필터가 아니라 유사도 필터인 이유.
- [[deduplicating-training-data]] — ExactSubstr 50토큰 임계값과 비용, NearDup parameter와 이중 임계값, cross-split 규칙, 겹침·암기 수치.
- [[minhash-lsh]] — 유사도 추정량과 banding 공식이 Broder 1997이 아니라 *Mining of Massive Datasets* 3장에서 온다는 기록.
- [[dolma]] — 웹 pipeline 단계 순서, attributes 후 mix toolkit 설계, Gopher·C4-NoPunc 필터율, 오염 제거 규칙, 다원천 도메인 적합 결과.
- [[c4]] — Cell C에서 ablation한 휴리스틱 규칙 세트.
- [[weborganizer]] — 주제·형식 classifier와 정확도, 암묵적 도메인 혼합으로서 품질 필터, 분포 변화 판독값인 홀드아웃 perplexity, 대상별 전이 손실, 제약 경로에 사용한 50M proxy 실행 레시피.
- [[data-constrained-scaling]] — 4-epoch 반복 한계, 필터 후 반복 설정, 필터 가치가 pool의 잡음 정도에 의존한다는 결과.
- [[openmathinstruct-2]] — level 3에 사용한 embedding retrieval + LLM judge paraphrase 오염 제거 절차.
