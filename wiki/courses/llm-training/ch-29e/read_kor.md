<!-- chapter: ch-29e
     track: sft
     kind: content
     title: 명령어 튜닝과 미관측 과제로의 일반화
     deps: [ch-00, ch-29]
     sources: [[flan]], [[t0-multitask-prompted-training]], [[super-natural-instructions]], [[flan-palm-scaling-instruction-finetuning]], [[flan-collection]], [[opt-iml]], [[instruction-diversity-unseen-tasks]], [[lima]], [[urial]], [[revisiting-superficial-alignment-hypothesis]], [[instruction-following-without-instruction-tuning]], [[false-promise-imitating-proprietary-llms]], [[tulu-1-how-far-can-camels-go]], [[tulu-3]], [[ifbench]]
     figures: figures/heldout-protocols.html
     revised: 2026-09 (일반성 개정)
-->

# Chapter 29e — 명령어 튜닝과 미관측 과제로의 일반화

> **핵심 통찰.** 여러 과제를 섞은 명령어 튜닝은 학습에서 제외한 과제 유형의 zero-shot 정확도를 올릴 수 있지만, 측정된 세 조건이 충족되어야 했다. 첫째, 서로 다른 학습 과제가 많이 있어야 한다. FLAN에서 학습 클러스터가 1개일 때 held-out 평균은 49.9였고 7개일 때 63.5였다([[flan]] 그림 6). 둘째, 모델이 충분히 커야 한다. FLAN의 8B 이하 모델은 held-out 정확도가 오히려 낮아졌다(그림 7). 셋째, 학습 형식에 평가 형식이 들어 있어야 한다. chain-of-thought 데이터를 빼고 학습한 Flan-PaLM은 chain-of-thought 정확도를 잃었다([[flan-palm-scaling-instruction-finetuning]] §4.2). 통제된 rewrite-rule 연구에서는 명령어 하나당 예제 수가 아니라 서로 다른 명령어의 수가, 모델이 미관측 명령어로 일반화하는지를 결정했다([[instruction-diversity-unseen-tasks]]). 응답의 문체는 학습 예제 100개에서 1,000개로 늘리는 것만으로도 바뀌며([[lima]]; [[revisiting-superficial-alignment-hypothesis]] 그림 3), 학습 없이 in-context 예제 세 개만 주어도 바뀔 수 있다([[urial]]). 반면 객관적 과제 정확도는 과제 데이터가 늘수록 power law를 따라 계속 상승한다([[revisiting-superficial-alignment-hypothesis]]). 따라서 문체에 민감한 승률만으로는 일반성을 증명할 수 없다.
>
> **지침.** 폭넓은 능력이 목표이고 데이터 예산이 고정돼 있다면, 과제 하나당 예제를 더 넣기 전에 서로 다른 과제, 명령어, 프롬프트 형식(zero-shot, few-shot, chain-of-thought)을 추가하라. FLAN, Super-NaturalInstructions, Flan-PaLM, rewrite-rule 연구에서 held-out 개선은 과제·명령어 수와 함께 나타났고, Super-NaturalInstructions에서는 과제당 예제 수의 효과가 64개에서 포화됐기 때문이다([[super-natural-instructions]] §7.1). 일반성 주장을 보고할 때는 과제 클러스터 전체와 그 원천 데이터셋을 학습에서 빼고, 별도의 development suite로 checkpoint를 선택하며, judge 승률 옆에 미관측 suite의 객관적 정확도를 보고하라. GPT-4o judge는 GSM8K에서 chat 문체 모델을 84.4%의 경우에 더 선호했지만, 그 모델의 정확도는 14.7%였고 과제 학습 모델은 46.5%였기 때문이다([[revisiting-superficial-alignment-hypothesis]] 표 2). 이 절차가 없다면 단일 suite의 향상은 development-set 결과로만 다뤄야 한다. FLAN의 조건에서 기반 모델이 8B 이하라면 전이를 가정하기 전에 held-out 성능 하락을 측정하라. 명령어 튜닝 뒤 FLAN의 8B 이하 모델은 held-out 정확도를 잃었기 때문이다([[flan]] 그림 7).

## 범용 모델에서 이 장이 중요한 이유

범용 모델의 학습 파이프라인에는 여섯 단계가 있다. pretraining, mid-training, 지도 미세조정(supervised fine-tuning, SFT), 선호 최적화(preference optimization), 강화학습(reinforcement learning, RL), 평가다. 이 장은 SFT 단계를 시작한다. SFT는 학습 분포를 코퍼스 단위가 아니라 과제 단위로 고르는 첫 단계다. 그러므로 여기서 처음으로 “모델이 좋아졌다”는 말이 “우리가 고른 과제에서 모델이 좋아졌다”는 뜻일 수 있다. 측정 가능한 문제는 전이다. 과제 집합 `T`로 학습한 모델의 정확도가 `T` 밖 과제에서 어떻게 변하는지, 그리고 기반 모델이 원래 갖고 있던 능력에서 어떻게 변하는지를 묻는다.

이 장 전체에서 세 용어를 쓴다. **과제(task)** 는 하나의 데이터셋과 하나의 과제 설정이 정의하는 입력에서 출력으로의 사상이다. 예를 들어 SQuAD의 추출형 질의응답이 하나의 과제다. **과제 클러스터(task cluster)** 또는 범주(category)는 자연어 추론처럼 같은 유형에 속하는 과제의 묶음이다. **템플릿(template)** 은 예제를 자연어 명령어와 목표 답으로 표현하는 자연어 패턴이다. “미관측(unseen)”은 어느 수준을 말하는지 밝히기 전에는 모호하다. 학습한 과제의 새 인스턴스일 수도 있고, 학습에 포함된 클러스터에서 나온 새 과제일 수도 있으며, 학습에서 빠진 새 클러스터일 수도 있다([[opt-iml]] §2.3). 이 장은 세 수준 모두에서 주장을 측정한다. §8은 [[ch-30]]부터 [[ch-36]]까지 재사용할 held-out 프로토콜을 고정한다. 범용 능력과 그 측정의 과정 전체 정의는 [[ch-00]]에 있고, [[ch-29]]가 만든 synthetic pool은 이 프로토콜이 평가하는 데이터다.

## §1 명령어 튜닝: 정의, 손실, 그리고 “미관측”의 세 수준

**정의.** 명령어 튜닝은 사전학습 언어 모델을 여러 과제의 혼합물로 지도 미세조정하는 방법이다. 각 입력은 자연어로 과제를 명시하고, target은 원하는 응답이다([[flan]] §1).

**해결하려는 문제.** GPT-3의 zero-shot 정확도는 독해, 질의응답, 자연어 추론에서 few-shot 정확도보다 낮았다([[flan]] §1). 저자들은 few-shot 예시가 없으면 pretraining 데이터의 형식과 닮지 않은 프롬프트를 모델이 더 어려워한다는 이유를 제안한다(해석).

**손실.** OPT-IML은 target 토큰에만 손실을 주는 지도 목적함수를 다음처럼 쓴다([[opt-iml]] §3.1, 식 1). 이 장에서 masking 방식을 밝힌 다른 decoder-only 실행도 같은 형태를 쓴다.

```
L(D; θ) = − Σ_i Σ_j log p_θ( t_ij | s_i , t_i,<j )
```

- `D`는 인스턴스 `i`로 이루어진 미세조정 데이터셋이고, `θ`는 모델 파라미터다.
- `s_i`는 인스턴스 `i`의 source token, 즉 명령어와 입력을 합친 token이다. 이 token들에는 loss를 주지 않는다.
- `t_ij`는 인스턴스 `i`의 `j`번째 target token이며, `t_i,<j`는 그보다 앞에 나온 target token들이다.

[[instruction-following-without-instruction-tuning]]은 식 2에서 명령어를 빈 문자열로 바꾸어 **response tuning**을 정의한다. 그러면 모델은 `p(response | instruction)`이 아니라 `p(response)`를 학습한다. §7에서 이 변형을 사용한다.

**“미관측”의 세 수준.**

| 수준 | 학습에서 제외하는 것 | 이를 측정하는 프로토콜 | 출처 |
|---|---|---|---|
| L1 held-out 인스턴스 | 학습한 과제의 새 예제 | OPT-IML의 “fully supervised” split | [[opt-iml]] §2.3 |
| L2 관측된 클러스터의 held-out 과제 | 학습한 과제 유형에서 나온 새 데이터셋 | OPT-IML의 “partially supervised” split | [[opt-iml]] §2.3 |
| L3 held-out 클러스터 | 한 과제 유형에 속한 모든 데이터셋 | FLAN leave-one-cluster-out, T0의 네 held-out 과제 | [[flan]] §2.2; [[t0-multitask-prompted-training]] §3 |

같은 변경이라도 세 수준의 반응은 다르다. OPT-IML 30B에서 학습 과제 수를 16개에서 1,024개로 늘렸을 때 L3와 L2 과제는 개선됐지만 L1 과제는 변하지 않았다([[opt-iml]] §4.4, 그림 2). 세 수준을 하나의 평균으로 합쳐 보고하면 이 차이를 숨기게 된다.


## §2 FLAN, T0, Super-NaturalInstructions의 held-out 프로토콜

함께 제공하는 그림 [heldout-protocols.html](figures/heldout-protocols.html)은 FLAN, T0, Super-NaturalInstructions, OPT-IML, Tülu 3의 split을 전환해 볼 수 있게 한다. 독자는 이 그림에서 어떤 클러스터가 학습에 들어가고, 평가에만 쓰이며, leakage를 막기 위해 제거되는지 확인할 수 있다.

### 2.1 FLAN: 클러스터 하나를 통째로 제외하기(arXiv 2021-09)

**메커니즘**([[flan]] §2.1–2.4):

1. 공개 텍스트 데이터셋 62개를 모아 각각을 12개 과제 클러스터 가운데 하나에 배정한다(그림 3).
2. 데이터셋마다 템플릿 10개를 작성한다. 이 가운데 최대 3개는 “과제를 뒤집는다.” 예를 들어 영화 리뷰를 분류하는 대신 영화 리뷰를 생성하게 만든다.
3. 데이터셋 `D`가 속한 어떤 클러스터의 데이터셋도 학습에 쓰이지 않았을 때만 `D`를 미관측으로 취급한다. 따라서 `c`개 클러스터를 평가하려면 각각 따로 학습한 모델 `c`개가 필요하다.
4. 이웃 클러스터도 함께 제거한다. NLI를 평가할 때는 paraphrase 클러스터를, 그 반대의 경우도 마찬가지로 제거한다. 어느 부모 클러스터를 평가하든 commonsense를 포함한 reading comprehension도 제거한다(§2.2, 각주 1).
5. 데이터셋의 예제 수에 비례해 섞되, 데이터셋별 예제 수는 30k로 제한하고 mixing-rate 최댓값은 3k로 둔다.

FLAN이 Raffel et al.에서 가져와 사용한 mixing 규칙은 다음과 같다(§2.4, 각주 2).

```
r_m = min(e_m, K) / Σ_n min(e_n, K)
```

- `r_m`은 데이터셋 `m`에서 표본을 뽑을 확률이다.
- `e_m`은 그 데이터셋의 예제 수다.
- `K`는 mixing-rate 최댓값이며 FLAN에서는 3,000이다.

**계산 예.** 세 데이터셋의 예제 수가 각각 1,000, 3,000, 30,000개이고 `K = 3,000`이라고 하자. 상한을 적용한 크기는 1,000, 3,000, 3,000이므로 표본 추출 확률은 `1/7 = 0.143`, `3/7 = 0.429`, `0.429`이다. 상한이 없으면 각각 `0.029`, `0.088`, `0.882`가 되어 가장 큰 데이터셋 하나가 batch의 88%를 차지한다.

**근거**([[flan]], LaMDA-PT 137B):

- development set에서 가장 좋았던 템플릿을 사용하면, zero-shot FLAN은 25개 데이터셋 중 20개에서 zero-shot GPT-3 175B를 앞섰다(§3). **결과(단일 연구).**
- NLI, closed-book QA, commonsense를 held out한 클러스터 ablation에서, 학습 클러스터 1개부터 7개(데이터셋 11개부터 39개)까지의 held-out 평균 점수는 차례대로 49.9, 55.0, 59.3, 59.2, 60.8, 61.9, 63.5였다. 유일한 하락인 59.3에서 59.2로의 변화는 sentiment 클러스터를 더했을 때 일어났다(§4.1, 그림 6).
- 규모 효과: held-out 과제 13개에서 422M, 2B, 8B, 68B, 137B 모델을 평가했을 때, 명령어 튜닝은 68B와 137B 모델은 개선했지만 8B 이하 모델의 held-out 정확도는 낮췄다(§4.2, 그림 7; 값은 그래프에만 있고 수치로 인쇄되지 않았다). 저자들은 약 40개 학습 과제를 배우는 일이 작은 모델의 용량을 채운다고 설명한다(해석).
- 명령어의 역할(App. B.2, 표 3, 네 클러스터 평균): FLAN은 55.2였고, 템플릿 없이 학습하면 37.3, 데이터셋 이름만 주고 학습하면 명령어로 평가했을 때 46.6, 이름으로 평가했을 때 47.0이었다.
- 데이터셋과 템플릿의 효과(App. B.1, 그림 11): 클러스터당 데이터셋 수를 늘리면 held-out 성능이 거의 10% 개선됐다. 반면 클러스터당 데이터셋 하나일 때 데이터셋당 템플릿 1개, 4개, 10개의 효과는 “비교적 무시할 만하다”고 기술됐고, 클러스터당 데이터셋 네 개에서는 그 효과가 사라졌다.

**조건과 한계.** 이미 문장 완성으로 표현된 과제 대부분에서는 명령어 튜닝이 개선을 만들지 못했다. FLAN은 commonsense 및 coreference 과제 7개 중 3개에서만 LaMDA-PT를 이겼다(§3). contamination은 pretraining corpus와의 대략적인 13-gram overlap으로 사후 점검했다. DROP, SQuADv2, ANLI R1/R2는 주로 passage context에서 거의 완전한 overlap을 보였다. ReCoRD와 PIQA는 clean subset에서 점수가 1% 넘게 낮았다(App. C).

### 2.2 T0: held-out 과제, 프롬프트 다양성, 프롬프트 선택 금지(arXiv 2021-10)

**메커니즘**([[t0-multitask-prompted-training]] §3–5):

1. NLI, coreference resolution, sentence completion, word sense disambiguation의 네 과제에 속한 모든 데이터셋을 held out한다. 또한 GPT-3가 평가에 사용한 모든 데이터셋을 주 모델의 학습에서 제외한다.
2. held-out 데이터셋 11개와 BIG-bench 과제 14개를 평가한다.
3. T5+LM 11B를 P3의 프롬프트로 학습한다. T5+LM은 T5를 100B C4 token으로 language-modeling 목적함수에 따라 추가 학습한 모델이다. P3에는 177개 데이터셋용 프롬프트 2,073개, 데이터셋당 평균 11.7개가 들어 있다.
4. 학습 데이터셋의 validation split 점수로 checkpoint를 고른다. 따라서 held-out 예제는 checkpoint 선택에 영향을 주지 않는다.
5. validation data를 사용해 프롬프트를 고르지 않는다. held-out 데이터셋마다 모든 프롬프트의 median과 사분위 범위(IQR, `Q3 − Q1`)를 보고한다.

**근거:**

- T0는 자신보다 약 16배 큰 GPT-3(최대 175B)와 비교해 held-out 데이터셋 11개 중 9개에서 같거나 더 좋았고, 큰 baseline language model보다 BIG-bench 14개 중 13개에서 개선됐다(§1, §6.1).
- 데이터셋당 프롬프트 수 `p`(§6.2, 그림 6): `p = 1`은 `p = 0`보다 median을 높였다. `p = 1 → 5.7`은 데이터셋 11개 중 8개에서 median을 올리고 7개에서 IQR을 낮췄다. 원래 과제를 변형하는 프롬프트까지 추가한 `p = 8.03`은 11개 중 9개에서 median을 높이고 8개에서 IQR을 낮췄다.
- 데이터셋 수 `d`(그림 7): 39개에서 49개로 늘리면 held-out NLI 데이터셋 5개 모두에서 median이 올랐고, 5개 중 1개에서 IQR이 내려갔다. 49개에서 55개로 늘리면 모두에서 median이 올랐고, 5개 중 2개에서 IQR이 내려갔다.
- baseline의 프롬프트 견고성: RTE의 프롬프트 10개에서 GPT-3가 공개한 프롬프트 점수는 58.8%였다(Brown et al.은 63.5%를 보고했다). 나머지 9개 프롬프트의 median은 52.96%, IQR은 1.28%였다(§6.2).

**조건과 한계.** T0의 3B 모델은 held-out 과제에서 개선됐지만 FLAN의 8B decoder-only 모델은 그렇지 않았다. 저자들은 가능한 원인 두 가지, 즉 masked language modeling으로 사전학습한 encoder-decoder 모델과 더 다양한 프롬프트를 든다(§7; 검증하지 않은 해석). HellaSwag에서는 명령어를 제거했을 때 T0 median이 33.65%에서 57.93%로 올랐다(§7). 이것은 문장 완성형 과제에 관한 FLAN의 결과를 반복한다.

### 2.3 Super-NaturalInstructions: 1,616개 과제와 원천 데이터셋 제외(arXiv 2022-04)

**메커니즘**([[super-natural-instructions]] §3, §5.1):

1. 76개 과제 유형과 55개 언어를 포괄하는 과제 1,616개를 수집한다. 각 과제에는 평균 56.6단어의 정의, 평균 positive 예제 2.8개와 negative 예제 2.4개가 있으며, 인스턴스는 최대 6.5K개다.
2. 12개 범주, 154개 과제(영어 119개, cross-lingual 35개)로 평가 세트를 고정하고, 과제당 인스턴스는 최대 100개를 뽑는다. test instance 수는 15,310개다.
3. test 과제와 같은 source dataset에서 만들어진 모든 과제를 학습에서 제거한다(각주 6). 그러면 영어 학습 과제는 757개가 남는다.
4. 답 후보의 순위를 매기지 않고 open-ended generation을 ROUGE-L로 채점한다.

**근거**(표 3, English track의 ROUGE-L): Tk-Instruct 11B는 62.0, InstructGPT 175B는 52.1, GPT-3 175B는 45.0, T0 11B는 32.3, T5-LM 11B는 30.2, demonstration output을 그대로 복사하는 방법은 28.5, test 과제에서 지도학습한 upper bound는 74.3이었다. Scaling 분석(§7.1, 그림 5; 과제와 인스턴스 sweep에는 T5-3B, 크기 sweep에는 T5 small부터 xxl을 사용)은 성능이 학습 과제 수에 대해 log-linear하게 증가하고, 과제당 인스턴스 64개에서 포화하며, 모델 크기에 대해서도 log-linear하게 증가한다고 보였다. 757개 과제로 학습한 T5-Large의 48.0은 128개 과제로 학습한 T5-3B의 48.4와 맞먹었다.

**조건과 한계.** 정의만으로 학습한 모델은 예제만 주는 test encoding으로 일반화하지 못했고, 반대 방향도 실패했다. 정의와 예제를 함께 사용해 학습한 모델은 두 형식 모두에 견고했다(§7.2, 표 4). 여기서 T0는 32.3을 기록했으며, 저자들은 T0의 다른 프롬프트 문체를 원인으로 든다(§6.1; 해석). 따라서 명령어 collection 사이의 형식 전이 자체가 일반화 축이다.

**범용 모델에 대한 함의.** 세 프로토콜은 세부 사항에서 다르다. 하나는 클러스터를, 다른 하나는 과제를 held out하며, template에 대해 하나는 mean, 다른 하나는 median을 쓰고, best-template 보고와 all-template 보고도 다르다. 각각의 세부 사항이 보고되는 수치를 바꾼다. 따라서 일반성 주장은 그 주장을 산출한 프로토콜과 함께 제시해야 한다.

## §3 과제 수, 모델 크기, chain-of-thought 데이터의 scaling(Flan-PaLM, arXiv 2022-10)

**설정**([[flan-palm-scaling-instruction-finetuning]] §2). 혼합물은 데이터셋 473개에서 나온 과제 1,836개로 구성한다. Muffin 80개 과제, T0-SF 193개, NIV2 1,554개, 그리고 사람이 작성한 rationale이 있는 데이터셋 9개의 chain-of-thought(CoT) 혼합물이다(그림 2). NIV2에서는 MMLU와 관련된 과제 44개를 제거했고(각주 4), GPT-3 평가 세트는 거의 모두가 이 혼합물에 학습 세트를 갖기 때문에 사용하지 않았다(§2.3). held-out suite는 MMLU(과제 57개), BBH(23개), TyDiQA(언어 8개), MGSM(언어 10개)이다.

**정규화 지표.** 점수는 무작위 추측을 기준으로 정규화한 뒤 평균낸다(§2.3, 각주 5).

```
norm = 100 · (raw − r) / (max − r)
```

- `raw`는 퍼센트 단위 정확도다.
- `r`은 그 과제에서 무작위 추측했을 때의 정확도다.
- `max`는 100이다.

각주의 계산 예에서 `r = 50`이면 raw 55는 10, raw 45는 −10이 된다. 선택지 네 개인 benchmark에서는 `r = 25`이므로 raw 49.3은 `100 · 24.3 / 75 = 32.4`가 된다(계산으로 도출).

**근거**(표 3, held-out 점수 여섯 개의 정규화 평균).

| 모델 | 과제 0개 | 9개(CoT) | 89개(+Muffin) | 282개(+T0-SF) | 1,836개(+NIV2) |
|---|---:|---:|---:|---:|---:|
| PaLM 8B | 6.4 | 8.3 | 14.8 | 20.5 | 21.9 |
| PaLM 62B | 28.4 | 29.0 | 33.4 | 37.9 | 38.8 |
| PaLM 540B | 49.1 | 52.6 | 57.0 | 57.5 | 58.5 |

대부분의 향상은 과제 282개까지 추가됐을 때 이미 도착했다(§3). 저자들은 두 가지 설명을 제시한다. 추가한 과제들이 충분히 다양하지 않거나, 미세조정은 주로 모델이 pretraining에서 이미 얻은 지식을 표현하는 법을 가르친다는 설명이다. pretraining에는 780B token을 썼지만 fine-tuning에는 1.4B token, 즉 0.2%만 썼다(해석).

**계산 예: 절대 향상과 상대 향상.** 8B 모델은 6.4에서 21.9로 15.5점 올랐다. 이는 남아 있던 오차 93.6 가운데 `15.5 / 93.6 = 16.6%`를 제거한 것이다. 540B 모델은 49.1에서 58.5로 9.4점 올랐으며, 반올림된 표의 값으로 계산하면 남은 오차 50.9 가운데 `9.4 / 50.9 = 18.5%`를 제거한 것이다. 저자들은 8B에 16.6%, 540B에 18.4%를 보고한다(§3). 따라서 “누가 더 많이 향상됐는가”라는 모델 크기 순위는 어떤 지표를 쓰느냐에 따라 뒤집힌다.

**Chain-of-thought 데이터**(§4.2, 그림 5). CoT가 아닌 데이터만으로 미세조정하면 held-out CoT 성능이 낮아졌고, CoT 데이터셋 9개를 더하면 CoT 평가가 개선되면서 non-CoT 평가는 낮아지지 않았다. 저자들은 명령어 미세조정이 학습 프롬프트 형식을 공유하는 미관측 과제를 개선한다고 결론짓는다(해석). 좁은 혼합물은 반대 방향으로도 성능 범위를 좁힐 수 있다. CoT 과제 9개만 학습하면 direct-prompting MMLU가 62B에서는 55.1에서 48.5로, 540B에서는 71.3에서 68.8로 낮아졌다. 반면 8B에서는 24.3에서 26.3으로 올랐다(표 3). CoT 미세조정은 학습하지 않은 PaLM이 성공적으로 사용하지 못한 “let's think step-by-step” 프롬프트를 BBH에서 zero-shot CoT로 사용할 수 있게도 했다(§4.3, 그림 6).

**사용성.** open-ended 프롬프트 190개에서 평가자는 Flan-PaLM 540B를 PaLM 540B보다 79%의 경우에 선호했다(§6, 그림 8).

**조건과 한계.**

- 각 모델의 fine-tuning step은 held-out 과제를 주기적으로 평가해 골랐다. 모델 크기에 따라 2k에서 10k step마다 평가했다(§2.2). 그러므로 held-out suite는 checkpoint 선택 신호 역할도 했으며, 이는 T0의 규칙이 금지하는 방식이다.
- 표 3과 표 5는 서로 다른 혼합 비율, 즉 “Proportion A”와 “Proportion B”를 쓴다(App. E 표 23). 그래서 540B 모델 점수는 표 3에서는 58.5, 표 5에서는 58.4다.
- GSM8K 83.9%(self-consistency를 적용한 CoT)는 held-out 결과가 아니다. GSM8K 학습 세트가 혼합물에 들어 있기 때문이다(§4.1).

---

## §4 혼합 프롬프트 형식, input inversion, 혼합 비율의 균형

### 4.1 Flan Collection ablation(arXiv 2023-01)

**설정**([[flan-collection]] §3). LM-adapted T5-XL(3B)을 사용한다. 검증에는 Held-In 과제 8개, CoT 검증 세트 5개를 사용하고, MMLU와 BBH는 held-out suite로 둔다. **Input inversion**은 예를 들어 답에서 질문을 생성하도록, 입력과 출력을 뒤바꾼 쌍으로 학습하는 방법이다(§3.4).

**근거**(표 1, zero-shot / few-shot).

| 변형 | Held-In | CoT | MMLU | BBH | BBH-CoT |
|---|---|---|---|---|---|
| Flan 2022(모든 방법) | 73.8 / 74.8 | 35.8 / 34.1 | 50.3 / 52.4 | 26.2 / 39.3 | 33.9 / 35.2 |
| − CoT 데이터 | 73.3 / 73.2 | 28.8 / 24.6 | 47.5 / 46.9 | 18.2 / 30.0 | 18.2 / 12.0 |
| − input inversion | 73.8 / 74.1 | 32.2 / 23.5 | 41.7 / 41.2 | 18.4 / 24.2 | 15.7 / 13.0 |
| − mixture balancing | 71.2 / 73.1 | 32.3 / 30.5 | 45.4 / 45.8 | 15.1 / 24.3 | 13.8 / 15.4 |
| − few-shot template | 72.5 / 62.2 | 38.9 / 28.6 | 47.3 / 38.7 | 27.6 / 30.8 | 18.6 / 23.3 |
| T5-XL on Flan 2021 | 68.4 / 56.3 | 24.6 / 22.7 | 41.4 / 34.8 | 28.1 / 28.3 | 26.0 / 26.9 |
| T5-XL on Super-NaturalInstructions | 50.3 / 42.2 | 13.8 / 14.3 | 35.6 / 31.1 | 10.4 / 15.6 | 8.0 / 12.5 |

한 행을 실제로 읽어 보자. few-shot template을 제거하면 few-shot Held-In은 74.8에서 62.2로 12.6점, few-shot MMLU는 52.4에서 38.7로 13.7점 낮아진다. 그러나 zero-shot CoT는 35.8에서 38.9로 3.1점, zero-shot BBH는 26.2에서 27.6으로 1.4점 오른다. Input inversion은 zero-shot Held-In을 0.0점 바꾸지만 zero-shot MMLU는 41.7에서 50.3으로 8.6점 이동시킨다. 형식 또는 augmentation 하나가 어떤 지표에는 도움이 되고 다른 지표에는 손해를 줄 수 있으므로, 평균 하나가 아니라 지표별 표 자체가 결과다.

추가 근거: few-shot template을 5% 넣으면 zero-shot 결과가 좋아졌고, Held-In 및 held-out 점수는 모두 few-shot 학습 데이터 비율 10%와 90% 사이에서 최고점에 도달했다(§3.2, 그림 3). Held-In 정확도는 학습 과제 약 200개 근처에서 최고점 뒤 하락했지만, held-out MMLU는 모든 과제를 쓸 때까지 log-linear하게 상승했다. 전체 세트 전에 최고점에 도달한 것은 T5-Small뿐이었다(§3.3, 그림 4). 동일 가중치 혼합물에서 한 source를 제거하면 MMLU는 47.3에서 44.7(T0-SF 제거) 또는 45.7(Flan 2021 제거)로 변했고, CoT를 제거하면 CoT 평가는 41.4에서 29.1로 변했다(표 2). 각각 **결과(단일 연구)**다. Flan Collection 저자들은 OPT-IML의 mixing 실험도 Flan 2021, T0-SF, T5 혼합물이 가장 폭넓게 이롭다고 찾았다고 쓴다(§3.5). 그러나 이 장은 OPT-IML 본문에서 그 순위가 명시된 것을 찾지 못했으므로, 이 주장을 재현된 결과로 세지 않는다.

### 4.2 OPT-IML: 세 일반화 수준과 혼합물의 trade-off(arXiv 2022-12)

**메커니즘**([[opt-iml]] §2–4):

1. collection 8개를 benchmark 하나로 통합한다. filtering 뒤에는 학습 과제 1,545개, validation 과제 35개, test 과제 87개가 남는다(표 1).
2. 평가는 §1의 일반화 수준 세 개로 나눈다.
3. leakage를 제거한다. 모든 train–evaluation 과제 쌍마다, 평가 예제 중 어떤 13-gram이라도 겹치는 예제가 차지하는 비율을 계산한다. 1%를 넘은 약 14,000개 쌍은 사람이 직접 검사한다(§2.3).
4. OPT 30B에서 각 요인을 4,000 step 동안 조정하고, 세 수준과 0-shot·5-shot 점수의 평균으로 설정을 고른다(§4.1).

**근거**(별도 언급이 없으면 OPT 30B):

- Mixing cap(EPS): cap을 둔 모든 모델은 평균적으로 cap 없는 모델보다 좋았고, 4,096보다 작은 cap들은 비슷한 성능을 보였다. 따라서 4,096을 골랐다(§4.2, 표 4).
- Benchmark 비율: 예제 수 비례 sampling을 쓰면 Super-NaturalInstructions가 예제의 71%, PromptSource가 18%, FLAN이 5%를 공급한다(§4.3). CrossFit, ExMix, T5, UnifiedSKG를 모두 0%로 두면 가장 나쁜 모델이 되었다(표 5).
- Fine-tuning에 pretraining 데이터를 섞으면 L3 및 L2 과제는 10%까지 개선된 뒤 하락했고, 5%를 선택했다(§4.5).
- Reasoning 데이터: held-out validation reasoning 과제 14개 중 2개에서 ROUGE-L이 12.2%에서 31.6%로 상승했다. 전체 평균 향상은 reasoning 데이터 1%일 때 가장 컸다(§4.6).
- Dialogue 데이터를 0.5%(BlenderBot 3 대화 320,543개) 넣으면 zero-shot validation 평균은 46.0에서 44.8로 내려갔다. stereotype-detection 및 word-analogy 과제가 요구하는 짧은 출력 형식을 모델이 덜 따르게 되었기 때문이다. 그래서 dialogue 데이터는 넣지 않았다(§4.7, 표 7).
- 표준 held-out 과제 14개에서 최종 모델의 zero-shot 평균(표 9)은 OPT 30B가 59.2에서 OPT-IML 30B 66.3으로, OPT 175B가 61.4에서 OPT-IML 175B 68.2로 상승했다. 개별 과제는 퇴보하기도 했다. OPT-IML 30B의 PIQA 32-shot은 78.8에서 69.2로, OpenBookQA 0-shot은 57.2에서 50.6으로 떨어졌다.

**조건과 한계.** 각 요인은 30B에서 독립적으로 바꿨다. 저자들은 요인들 사이에 상호작용이 있을 수 있고, 30B에서의 trade-off가 더 큰 scale에도 성립하지 않을 수 있다고 명시한다(§6.2). OPT-IML-Max 175B는 MMLU zero-shot 49.1, BBH 35.7을 기록해 Flan-T5 11B의 53.7과 45.3보다 낮았다(표 14). 저자들은 pretraining token 수(OPT: 180B), architecture, fine-tuning 방법의 차이를 가능한 원인으로 든다(§6.1; 해석).

## §5 통제된 실험에서의 instruction 다양성

**정의.** Instruction 다양성은 전체 예제 수를 고정했을 때 학습 세트 안에서 서로 다른 instruction의 개수와 의미적 분산을 말한다.

**문제.** 자연어 instruction 데이터셋에서는 instruction 수, instruction당 예제 수, 품질이 함께 변한다. 따라서 각각이 미치는 효과를 분리할 수 없다([[instruction-diversity-unseen-tasks]] §1).

**메커니즘**([[instruction-diversity-unseen-tasks]] §3–4, arXiv v1 2024-02):

1. 각 instruction은 rewrite rule `x → y`다. 모델은 문자열 `z`에서 `x`가 가장 왼쪽에 처음 나타나는 위치를 `y`로 바꾸거나, `x`가 없으면 `z`를 그대로 반환해야 한다. 후자를 “no-op”이라고 부른다.
2. 6 layer, 4 head, hidden size 256인 GPT-2 스타일 모델을 scratch부터 학습한다. 전체 예제 수는 `S × I = 10^6`이며, `I`는 서로 다른 rule 수, `S`는 rule당 예제 수다.
3. 학습 중 한 번도 나타나지 않은 rule로 만든 예제 `10^5`개에서 시험한다.

**계산 예.** 예산을 `10^6`개 예제로 고정하면, `I = 100`일 때 `S = 10,000`, `I = 10,000`일 때 `S = 100`이다. 논문은 instruction이 300개보다 적게 학습된 모델은 rule당 예제가 많아도 일반화하지 못했고, 1,000개 이상 instruction으로 학습된 모델은 rule당 예제가 적어도 항상 일반화했다고 보고한다. 전이는 instruction 약 400개 근처에서 나타났다(§4.1, 그림 2a). 따라서 첫 configuration은 실패하고, 두 번째 configuration은 성공한다.

추가 근거:

- **의미적 분산.** 하나의 제약된 rule family만으로 학습한 모델(예: 문자를 `k`번 반복하며 `k`가 큰 경우)은 작은 `k`로 일반화하지 못했다. 제약된 family 세 개를 섞으면 일반화했다(§4.3, 그림 4).
- **불균등 분포.** instruction 1,000개일 때 instruction당 예제 수가 가파른 power law를 따르면 정확도가 떨어졌다. instruction이 10,000개 또는 100,000개일 때 그 손실은 작았다(§4.4, 그림 3a).
- **입력 다양성.** instruction 1,000개에서 한 번만 출현하는 input으로 학습하면 출현 횟수 1~20에서 정확도는 0.41이었다. 출현 횟수 1, 5, 10, 15, 20을 섞으면 0.94였다(표 1).
- **Pretrained model.** 암호화된 rewrite 과제에서 Llama-2-7B에 LoRA(rank 512, `α 1024`)를 적용했다. 학습 sequence 40,000개, test sequence 5,000개, no-op 비율 40%를 사용했다. 그림 5의 더 작은 instruction 수에서는 no-op 사례만 풀었다. instruction 9,000개에서 데이터를 균일하게 절반으로 줄여도 성능은 떨어지지 않았지만, 균일하지 않게 subsample하면 떨어졌다(§5, 그림 5).

**조건과 한계.** 과제는 symbolic이며, 대부분의 모델은 6 layer·hidden size 256이고 scratch부터 학습한다. 저자들은 실제 세계의 instruction-following 데이터셋이 없다는 점을 한계로 든다. **결과(단일 연구)**다. 방향은 자연어 결과와 일치한다. Super-NaturalInstructions에서는 task당 예제가 64개에서 포화된 반면 task 수가 늘수록 정확도가 증가했고([[super-natural-instructions]] §7.1), 이질적인 Stack Exchange 예제 2,000개로 학습한 7B 모델은 동질적인 wikiHow “how to” 예제 2,000개로 학습했을 때의 3.49보다 높은 3.83을 기록했다([[lima]] §5, 그림 5). 한 task 안에서의 phrasing 다양성에 대한 근거는 충돌한다. T0는 dataset당 prompt가 많을수록 향상됐지만, FLAN은 cluster마다 dataset이 네 개가 되면 dataset당 template 수의 효과가 작았다고 찾았다(§2). T0 저자들은 이 불일치를 언급하며, 자신들의 prompt가 길이와 문구에서 더 다양했을 것이라고 가설을 제시한다([[t0-multitask-prompted-training]] §7; 검증하지 않은 해석). **열린 질문:** 현대 scale에서 task, cluster, phrasing, input 중 어느 종류의 다양성이 가장 중요한가.

## §6 피상적 정렬 가설: 근거와 반대 근거

**정의.** LIMA는 **피상적 정렬 가설(Superficial Alignment Hypothesis)**을 다음처럼 제시한다. “모델의 지식과 능력은 거의 전부 pretraining 중에 학습되며, alignment는 사용자와 상호작용할 때 써야 하는 형식의 부분분포가 무엇인지 가르친다”([[lima]] §2, arXiv 2023-05).

### 6.1 LIMA

설정([[lima]] §2–3): LLaMA 65B를 예제 1,000개(약 750,000 token)로 fine-tuning했다. 구성은 Stack Exchange STEM 200개, Stack Exchange 기타 200개, wikiHow 200개, r/WritingPrompts 150개, Super-NaturalInstructions 50개, 저자가 직접 쓴 예제 200개다(표 1). Stack Exchange 답변은 길이(1,200~4,096 character), 1인칭 표현, 다른 답변을 참조하는 표현을 기준으로 filtering했다(§2.1).

**근거:**

- 300개 test prompt에서 human rater는 LIMA가 GPT-4보다 같거나 낫다고 한 비교를 43%, Claude 대비 46%, Bard 대비 58%, DaVinci003 대비 65%로 판단했다(§1, §4.2, 그림 1).
- 50개 prompt의 절대 평가에서는 응답 50%가 excellent였고, 88%가 prompt 요구사항을 충족했다. Out-of-distribution prompt 20개에서는 20%가 실패, 35%가 통과, 45%가 excellent였다(§4.3).
- ChatGPT가 1~6점으로 매긴 LLaMA 7B ablation에서 filtered Stack Exchange는 3.83, unfiltered는 3.33, wikiHow는 3.49였다. Filtered Stack Exchange를 2K에서 32K 예제로 늘려도 점수는 오르지 않았다(§5, 그림 5–6).
- Dialogue 데이터가 없으면 실제 대화 10개 중 6개가 3 turn 안에 실패했다. Dialogue chain 30개를 추가하면 excellent turn 비율은 45.2%에서 76.1%로 올랐고, 실패는 42 turn 중 15개에서 46 turn 중 1개로 줄었다(§6, 그림 7).

**한계.** 주 평가는 open-ended prompt의 pairwise preference이며 객관적 task benchmark가 없다. 저자들은 학습 prompt와 test prompt를 쓴 두 집단 사이에 “significant contact”가 있었다고 적었다(§2.2, 각주 1). Held-out perplexity는 올랐지만 생성 품질도 올랐으므로, 저자들은 50예제 development set에서 epoch 5와 10 사이 checkpoint를 사람이 직접 골랐다(§3, 부록 B).

### 6.2 URIAL: token 분포 이동과 tuning 없는 alignment(arXiv 2023-12)

**메커니즘**([[urial]] §2):

1. Aligned model에서 응답 `o`를 greedy decoding한다.
2. 위치 `t`마다 같은 prefix를 base model에 넣고, 그 다음-token 분포의 순위를 매긴다.
3. Aligned token `o_t`의 **base rank** `η`를 기록한다. `η = 1`이면 unshifted, `1 < η ≤ 3`이면 marginal, `η > 3`이면 shifted다.

**근거**(그림 3, 예제 1,000개): Llama-2-7B에서 Llama-2-7B-chat으로 갈 때 unshifted 77.7%, marginal 14.5%, shifted 7.8%였다. Llama-2-7B에서 Vicuna-7B-v1.5로 갈 때는 82.4% / 12.8% / 4.8%, Mistral-7B에서 Mistral-7B-Instruct로 갈 때는 82.2% / 12.5% / 5.2%였다. Shifted token은 주로 담화 표지와 안전 문구(`"However"`, `"cannot"`, `"Here"`, `"sorry"`)였고, 이동은 응답 초반 위치에서 더 컸다(§2.2, 그림 4). **계산 예(평균에서 유도).** Llama-2-7B-chat token 1,000개 전체에서는 약 777개가 base model의 top choice이기도 하고, 145개는 2위 또는 3위이며, 78개는 더 낮은 순위다. Shifted token은 초반 위치에 몰리므로, 단일 응답 하나가 이 개수를 그대로 따를 필요는 없다.

URIAL 자체는 system prompt와 문체를 바꾼 예시 `K = 3`개(1,011 token)를 base model에 prompt로 주며 weight update는 하지 않는다(§3.3). AlpacaEval, MT-Bench, LIMA test set, HH-RLHF red-team prompt, MaliciousInstruct를 합친 저자들의 1,000-prompt 평가 set에서, GPT-4가 다섯 측면을 1~5점으로 매기고 ChatGPT가 안전성을 채점했다(§4.1–4.2, 표 1). 평균은 Mistral-7B + URIAL 4.63 대 Mistral-7B-Instruct 4.44, Llama-2-70B + URIAL 4.74 대 Llama-2-70B-chat 4.67(둘 다 4-bit quantized), Llama-2-7B + URIAL 4.33 대 Llama-2-7B-chat 4.47이었다.

**한계.** 점수는 GPT-4 및 ChatGPT judge에서 나왔다. 저자들은 coding, mathematics, interactive agent에는 tuning이 여전히 필요할 수 있다고 명시한다(§5.4). SFT 뒤 지식이 사라진 사례(예: Mistral-7B-Instruct가 Facebook이 이름을 바꿨다는 사실을 부정함)는 단일 case study다(§4.3, 부록 B).

### 6.3 반대 근거: post-training의 객관적 scaling(arXiv 2024-09)

[[revisiting-superficial-alignment-hypothesis]]는 base Llama-3, Llama-2, Mistral model을 task 예제 0~10,000개(GSM8K, SubQA, StarCoder Self-Align, Conifer, Dolly)로 fine-tuning하고 표준 benchmark를 채점한다(§3.1, 표 1). Task 정확도는 power law를 따랐다(§3.2).

```
P = a · D^(1/b)
```

- `P`는 percent 단위 benchmark accuracy, `D`는 fine-tuning example 수, `a`와 `b`는 model과 task별로 fit한 값이다(부록 A.6, 표 7).

**계산 예.** Llama-3-8B의 GSM8K는 `a = 19.47`, `b = 7.66`이다. `D`가 10배 늘 때마다 `P`는 `10^(1/7.66) = 1.351`배가 된다. `D = 100`이면 35.5, `D = 1,000`이면 48.0, 전체 training set인 `D = 7,500`이면 62.4다. 1,000에서 fit한 값은 표 2의 1,000-example GSM8K model 실측값 46.5%와 가깝다. 그림의 세 번째 panel은 표 7의 모든 fit에 이 곡선을 계산한다.

**근거:**

- Llama-3-8B를 LIMA 예제 1,000개로 학습하면 GSM8K 14.7%, SubQA 21%였지만, task 예제 1,000개로 학습하면 각각 46.5%, 36%였다. LIMA의 comparison prompt를 쓴 GPT-4o judge는 GSM8K에서 LIMA 학습 응답을 84.4% 선호했고 task 학습 응답은 0.24%만 선호했다(표 2).
- Style 및 formatting error는 100예제에서 포화 수준에 도달했다(표본 크기 0, 100, 1,000, 전체 training split). 반면 전체 mistake는 reasoning error를 추적했다(Llama-3-8B에서 math `r² = 0.98`, multihop QA `r² = 0.99`)(§4.2, 그림 3).
- Llama-3-8B cutoff 이후의 news event 100개에서, 먼저 multihop reasoning으로 post-training한 model은 event SFT 후 직접 질문 81개와 multihop 질문 55개에 답했다. Base model은 각각 65개와 37개였다. Event를 prompt에 넣으면 전자는 86개와 71개, 후자는 49개와 34개에 답했다(표 4).

**한계.** 각 run은 task 하나를 목표로 하며, 저자들은 한 task를 위한 fine-tuning이 다른 task 성능에 미치는 영향을 모른다고 말한다(§7). **결과(단일 연구)**다.

### 6.4 모방 격차(the imitation gap)(arXiv 2023-05)

[[false-promise-imitating-proprietary-llms]]는 GPT-2 1.5B와 LLaMA 7B·13B를 ChatGPT output 최대 150M token으로 fine-tuning했다. Crowdworker는 13B imitation model output의 약 70%를 ChatGPT와 같거나 낫다고 평가했다(그림 1). 폭넓은 imitation data는 Natural Questions 정확도를 낮췄다(7B: 17→10, 13B: 20→15). 반면 target을 맞춘 NQ-style 예제 6,000개는 높였다(7B: 22, 13B: 27, ChatGPT: 31)(표 1). 문체는 수렴했다. ChatGPT가 list를 쓴 경우 imitation model도 list를 쓴 비율은 학습 전 13%에서 150M token 후 81%가 됐다(표 2). Imitation data를 더 넣는 것보다 base model을 키우는 편이 benchmark accuracy를 더 높였다(§4.3). 이 결과의 method 수준 처리는 [[ch-19]]에 있다. [[tulu-1-how-far-can-camels-go]](§6)는 imitation data가 다양한 skill을 포괄할 때 도움이 된다고 주장한다(해석).

### 6.5 근거를 함께 해석하기

이 결과들은 format과 capability가 분리된다는 해석과 양립한다(해석). Response format과 assistant style은 100~1,000개 예제에서 학습되고([[lima]] §4–6; [[revisiting-superficial-alignment-hypothesis]] 그림 3), aligned model의 token 위치 중 4.8~7.8%는 base model에서 3위보다 낮은 순위다([[urial]] 그림 3). Reasoning 또는 특정 지식이 필요한 task의 정확도는 task-specific data가 있을 때 계속 향상한다([[revisiting-superficial-alignment-hypothesis]] 그림 1; [[flan-palm-scaling-instruction-finetuning]] §4.2). SFT data가 그 task를 포괄하지 않으면 향상하지 않거나 퇴보한다([[false-promise-imitating-proprietary-llms]] 표 1; [[tulu-1-how-far-can-camels-go]] §5.1). 범용 model에서는 작게 curated한 set만으로 응답 형식을 정할 수 있지만, 어떤 capability가 향상하는지는 task data의 breadth가 결정한다.

## §7 명령어 튜닝 없이도 나타나는 instruction following(arXiv 2024-09)

**응답 튜닝(response tuning)**([[instruction-following-without-instruction-tuning]] §4). Llama-2-7B와 OLMo-7B-Feb2024를 LIMA의 명령어를 제거한 응답 1,030개로 학습했다. 저자들은 이 모델을 같은 base model에 LIMA 전체를 사용해 instruction tuning한 경우와 비교했고, 길이를 통제한 AlpacaEval 승률로 평가했다(표 1). Response-tuned Llama-2-7B의 승률은 43.3% ± 1.1%였고 base model은 2.4%였다. Response-tuned OLMo-7B-Feb2024의 승률은 43.7% ± 1.7%였고 base model은 4.7%였다. 승률 50%는 두 응답의 품질이 같다는 뜻이다.

**응답 순위화 능력(response ranking capability)**(§4.2, 식 3). 서로 독립인 두 쌍 `(instruction, response)`과 `(instruction′, response′)`에 대해, 다음이 성립하면 이 능력이 있다고 정의한다.

```
p_θ(response | instruction) > p_θ(response′ | instruction)
```

- `p_θ`는 model이 부여하는 sequence probability다. `response′`는 다른 instruction에 대한 바람직한 응답이다.

Alpaca 쌍에서 이 조건은 base Llama-2-7B에서는 80.4%, instruction-tuned version에서는 77.4%의 경우에 성립했다. OLMo에서는 각각 74.5%와 74.3%였다(표 2). **계산 예(설명용 값).** Base model이 올바른 recipe에 `10^-40`, 무관한 답에 `10^-45`의 확률을 주면 올바른 응답의 순위는 더 높으므로 위 조건은 성립한다. 하지만 이어서 나올 문자열인 `"Give me a recipe for cake"`의 확률이 `10^-5`라면 greedy decoding은 그 문자열을 출력한다. Response tuning은 명령어에서 응답으로의 mapping을 가르치지 않아도 응답 같은 문자열의 확률을 올려 출력 결과를 바꿀 수 있다.

**단일 과제 fine-tuning**(§5, 표 3). 좁은 과제 하나만으로 학습해도 일반적인 instruction following이 나타났다. Llama-2-7B의 승률은 GSM 23.7%, poetry 22.9%, MBPP 16.9%, recipes 14.6%, chess 2.1%였다. OLMo-7B-Feb2024의 승률은 GSM 30.3%, poetry 21.9%, recipes 21.5%, MBPP 10.4%, chess 6.3%였다. 단, 응답은 fine-tuning에 사용한 명령어와 비슷한 instruction에 대해서만 그 fine-tuning 형식을 따랐다(§5.2, 그림 5).

**규칙 기반 adapter**(§6, 식 4–5, 표 4). 저자들은 Llama-2-7B의 확률에 손으로 쓴 세 규칙을 곱했다. 규칙은 응답 길이에 따라 end-of-sequence score를 올리고, 고정 token 15개의 확률을 바꾸며, 반복 token에는 penalty를 주는 것이다. 이 adapter의 승률은 24.4% ± 0.40%였다. EOS 규칙을 제거하면 10.4%, 다양성 규칙을 제거하면 14.3%, token 규칙을 제거하면 16.3%로 내려갔다.

**조건과 한계.** Llama-2-7B는 pretraining 중 instruction data를 보았을 수 있다. OLMo-7B-Feb2024에는 의도적으로 넣은 instruction data가 없었고, 두 model은 비슷한 결론을 보였다(§1.1). 평가는 greedy decoding과 LLM judge를 사용한다.

**범용 model에 대한 함의.** Base model보다 높은 승률만으로 SFT data가 측정한 행동을 가르쳤다고 결론 내릴 수 없다. 응답만으로 학습하거나 좁은 task만으로 학습한 경우도 승률을 올리기 때문이다. 좁은 task로 fine-tuning한 model도 무관한 instruction을 따를 수 있으므로, 저자들은 safety testing을 포함해 범용 chatbot으로 시험하라고 권고한다(§7).

## §8 ch-30~ch-36에서 사용하는 held-out task 및 template protocol

**정의.** 이 protocol은 어떤 task, dataset, template, evaluation setting을 SFT data와 설계 의사결정에서 제외할지를 결정하는 규칙의 묶음이다. 보고한 score가 목표로 삼지 않은 task에서의 성능을 추정하게 하려는 목적이다.

**규칙.**

1. **Cluster holdout.** Task type별로 task를 묶고, L3 평가에서는 cluster 전체를 holdout한다. Holdout한 cluster와 task type이 비슷한 cluster도 제거한다. FLAN은 NLI를 holdout할 때 paraphrase를 제외한다([[flan]] §2.2, 각주 1). 또 §4.1 ablation에서는 reading comprehension이 commonsense reasoning과 “too similar”하다는 이유로 commonsense와 함께 제외한다(각주 3).
2. **Source-dataset exclusion.** Evaluation task와 같은 source dataset에서 만든 모든 training task를 제거한다([[super-natural-instructions]] 각주 6). 같은 input-output pair를 다시 쓰는 input-inverted variant도 포함한다([[flan-collection]] §3.4).
3. **N-gram decontamination.** 모든 training source를 모든 evaluation set과 대조한다. OPT-IML은 13-gram overlap을 검사하고 1%를 넘는 pair는 사람이 검토한다([[opt-iml]] §2.3). Tülu 3는 test instance token의 50% 초과가 training instance 하나와 8-gram을 공유하면 overlap으로 보며, evaluation instance의 2% 초과와 겹치는 training set은 flag한다([[tulu-3]] §3.2).
4. **Development suite와 unseen suite의 분리.** Mixture, epoch, learning rate, checkpoint를 포함한 모든 설계 선택은 development suite에서 하고, 선택을 고정한 뒤에만 unseen-suite score를 본다. Tülu 3는 model을 개발하는 동안 unseen score를 보지 않았다([[tulu-3]] §2.2, §7.3). Checkpoint는 training-task validation split 또는 development suite에서 고른다(T0 규칙, [[t0-multitask-prompted-training]] §5). Unseen suite에서 고르면 안 된다.
5. **Template holdout.** Training에 없던 template과 constraint type으로 평가하고, 가장 높은 template score 대신 template 전체의 median과 IQR을 보고한다([[t0-multitask-prompted-training]] §5). 이 course는 IFEval의 constraint template 25개를 이 의미의 development set으로 취급한다(해석). GPT-4.1과 Claude 3.7 Sonnet 같은 선도 model도 IFBench의 보지 못한 constraint 58개에서는 50% 미만을 기록했기 때문이다([[ifbench]] §1, 그림 1).
6. **세 level과 세 setting.** L1, L2, L3를 분리해 보고하고([[opt-iml]] §2.3), 각 level을 zero-shot, few-shot, CoT setting으로 각각 보고한다([[flan-collection]] 표 1).
7. **Judge 옆의 objective metric.** LLM-judge win rate를 보고할 때는 exact-match 또는 execution accuracy도 함께 보고한다([[revisiting-superficial-alignment-hypothesis]] 표 2; [[tulu-1-how-far-can-camels-go]] §5.4).
8. **Base-model row.** 같은 suite와 같은 template에서 base model도 평가하고, task별 regression을 보고한다([[opt-iml]] 표 9). [[ch-30a]]는 이를 forgetting report로 바꾼다.

**계산 예: development-set overfitting 탐지** ([[tulu-3]] §7.4.1, 표 32, Tülu 3 8B SFT). Data source `s`에 대해 다음처럼 정의한다.

```
Δ_dev(s)    = score_dev(with s)    − score_dev(without s)
Δ_unseen(s) = score_unseen(with s) − score_unseen(without s)
```

- `score_dev`는 skill의 development metric(IFEval)이고, `score_unseen`은 같은 skill의 unseen metric(IFEval-OOD)이다.

Persona instruction data의 경우 full SFT model은 IFEval에서 72.8, IFEval-OOD에서 17.6을 기록한다. Persona data를 빼면 53.6과 18.0이다. 따라서 `Δ_dev = +19.2`, `Δ_unseen = −0.4`다. WildChat data의 경우 이를 뺀 model은 70.1과 20.8을 기록하므로 `Δ_dev = +2.7`, `Δ_unseen = −3.2`다. 두 source 모두 development score는 올렸지만 unseen score는 올리지 못했다. Tülu 3 저자들은 이 data 선택이 precise instruction following의 development evaluation에 overfit했다고 결론 낸다(§7.4.1). 이 protocol에서는 `Δ_dev > 0`이면서 `Δ_unseen ≤ 0`인 source를 flag한다(course 규칙; 해석).

## 음성 표본과 음성 피드백

이 단계에서 “음성(negative)”은 style standard가 정의한 네 가지 뜻 중 두 가지로 나타난다. (1) 음의 한계 가치, (2) 내용으로서의 음성이다. 이 장의 어느 출처도 음성을 conditioning (3)이나 gradient (4)로 사용하지 않는다. SFT cross-entropy는 softmax를 통해 target이 아닌 모든 token의 확률을 계속 낮춘다. 그 기제는 [[ch-30]]이, 명시적 음성 gradient는 [[ch-31a]]가 다룬다.

1. **음성의 출처.** (a) Quality filter가 표본을 버린다. 예를 들어 LIMA의 Stack Exchange 길이·문체 filter가 그렇다([[lima]] §2.1). (b) source 전체가 어떤 capability에는 음의 한계 가치를 가질 수 있다. 13B에서 Super-NaturalInstructions만 사용하면 base model 대비 GSM은 14.5에서 4.0으로, BBH는 39.3에서 4.5로 낮아졌다([[tulu-1-how-far-can-camels-go]] Table 3). Dialogue data 0.5%는 OPT-IML 30B의 0-shot validation 평균을 46.0에서 44.8로 낮췄다([[opt-iml]] Table 7). (c) 잘못된 output은 instruction 안의 내용으로 들어간다. Super-NaturalInstructions의 각 task에는 설명을 포함한 negative example이 평균 2.4개 있다([[super-natural-instructions]] Table 2). (d) Refusal target도 있다. LIMA는 응답이 일부 또는 전부 거절인 training prompt 13개를 포함한다([[lima]] §2.2).
2. **현재 관행.** 유형 (1) 표본은 버리거나 ablation 뒤 source 자체를 제거한다. 유형 (2) example은 input 안이나 target으로 남기며, target token에 대한 보통의 cross-entropy로 학습한다.
3. **기제.** Instruction에 넣은 negative example은 §1의 conditioning context `s_i`만 바꾼다. Loss는 여전히 target에 대해 계산한다. Refusal target은 내용이 거절인 positive target이다.
4. **근거.** Stack Exchange filtering은 7B ablation score를 3.33에서 3.83으로 높였다([[lima]] Fig. 5). T5-3B에서 negative example 두 개를 넣어 train·test하면 ROUGE-L은 negative 없이도 54.3, 넣어도 54.3이었고, 설명까지 넣으면 52.6이었다([[super-natural-instructions]] Table 4 diagonal). 저자들은 negative가 “a little bit” 도움을 준다고 기술한다(§7.2). LIMA는 sensitive test prompt 30개 중 80%에 안전하게 응답했고, 그중 malicious intent가 있는 10개에서는 6개에 안전하게 응답했다([[lima]] §4.3).
5. **통제.** Source를 제거하거나 유지할지 결정할 때는 development와 unseen suite 모두에서 ablate해야 한다. Source가 development score는 올리면서 unseen score는 그대로이거나 낮출 수 있기 때문이다([[tulu-3]] Table 32; [[flan-collection]] Table 2는 source별 제거를 보고한다). Refusal target을 추가하면 benign prompt에서의 refusal rate도 측정한다. Tülu 저자들은 Tülu 13B의 ToxiGen toxic generation rate 0.1%를 refusal behavior에 대한 overfitting 때문이라고 본다(item 7).
6. **진단.** Development·unseen suite의 source별 leave-one-out table, filtering 전후 category별 count([[ch-29]] survival report), benign prompt에서의 refusal 비율을 본다.
7. **범용성에 미치는 효과.** Filter와 source 제거는 task type을 삭제해 coverage를 좁힐 수 있다. Refusal-heavy target은 over-refusal을 높일 수 있다. Tülu 13B는 ToxiGen에서 toxic generation 0.1%를 냈고 ChatGPT는 27.7%였으며, 저자들은 이를 refusal behavior overfitting으로 해석한다([[tulu-1-how-far-can-camels-go]] §5.3; 해석).

## 레시피

| Model (정확한 release) | 크기 | 단계 | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---:|---|---|---|---|---|---|
| FLAN (LaMDA-PT) | 137B | SFT | steps; batch; optimizer; LR | 30k gradient steps; 8,192 tokens; Adafactor; 3e-5 | arXiv:2109.01652v5 §2.4 | verified 2026-09-15 | ablation 보고 없음 |
| FLAN (LaMDA-PT) | 137B | SFT | mixture; input/target length; packing | examples-proportional, dataset당 ≤30k examples, mixing-rate max 3k; 1024 / 256 tokens; EOS separator packing | §2.4, footnote 2 | verified 2026-09-15 | ablation 보고 없음 |
| FLAN (LaMDA-PT) | 137B | SFT | templates; compute; checkpoint | dataset당 10개; TPUv3 128 core에서 약 60시간; 30k step final checkpoint | §2.1, §2.4 | verified 2026-09-15 | App. B.1 Fig. 11: template 1·4·10개 효과는 “comparatively negligible” |
| T0 (T5+LM) | 11B | SFT | LR; batch; optimizer; dropout | 1e-3; 1024 sequences (2^20 input tokens); Adafactor; 0.1 | arXiv:2110.08207v3 §5 | verified 2026-09-15 | ablation 없음(“standard practice for fine-tuning T5”) |
| T0 (T5+LM) | 11B | SFT | truncation; mixing cap; checkpoint rule | 1024 / 256 tokens, packing; 500,000개 초과 dataset은 `500,000 / num_templates`; training-dataset validation split 최고 점수 | §5 | verified 2026-09-15 | ablation 없음 |
| Tk-Instruct (T5.1.1-xxl) | 11B | SFT | batch; LR; steps; encoding | 1,048,576 tokens (1,024 examples); constant 1e-5; 1,000 steps; definition + positive example 2개 | arXiv:2204.07705v3 App. D, §4 | verified 2026-09-15 | Table 4 (T5-3B): Def+Pos(2) 54.3 대 Def 45.0 ROUGE-L |
| Tk-Instruct analysis model (T5 ≤ 3B) | ≤3B | SFT | epochs; batch; LR; lengths | 2 epochs; 16; constant 1e-5; input 1024, output 128 | App. D | verified 2026-09-15 | ablation 없음 |
| Flan-PaLM | 540B | SFT | batch; dropout; LR; steps; schedule | 32; 0.1; 1e-3; 21k; constant, Adafactor, packed example boundary를 넘는 attention masking | arXiv:2210.11416v5 App. E Table 22, §2.2 | verified 2026-09-15 | App. E는 LR·batch·dropout을 가장 중요하다고 명시; table 없음 |
| Flan-PaLM | 8B, 62B | SFT | batch; dropout; LR; steps | 32; 0.05; 3e-3; 40k | App. E Table 22 | verified 2026-09-15 | ablation 없음 |
| Flan-T5-XL / Flan-T5-XXL | 3B / 11B | SFT | batch; dropout; LR; steps | 64; 0.05; 5e-4; 38k / 14k | App. E Table 22 | verified 2026-09-15 | ablation 없음 |
| Flan-PaLM (all sizes) | all | SFT | per-task caps; mixture proportions (sampling weight) | Muffin 30,000, T0-SF 20,000, CoT 100,000, NIV2 5,000; A=52% / 15% / 3% / 30%(§3-4); B=46.0% / 27.9% / 1.8% / 24.2%(other runs) | App. E Table 23 | verified 2026-09-15 | App. E: “Proportion A signaled that the T0 mixture was good for performance” |
| Flan-PaLM | 540B | SFT | compute; tokens; checkpoint rule | pretraining compute의 0.2%(약 512 v4 TPU chip, 37시간); fine-tuning token 1.4B(PaLM runs); periodic held-out-task 평가로 step 선택 | §2.2, §3, Table 2 | verified 2026-09-15 | ablation 없음 |
| OPT-IML 30B / 175B | 30B / 175B | SFT | GPUs; batch; LR; steps; warmup; tokens | 64 / 128; 256 / 128; 5e-05; 4,000 / 8,000; 60; 2B | arXiv:2212.12017v3 §3.3 Table 3 | verified 2026-09-15 | validation split에서 LR `{1e-5,3e-5,5e-5,6e-5}`, per-GPU batch `{2,4,8}` sweep; 결과 table 없음 |
| OPT-IML 30B / 175B | 30B / 175B | SFT | optimizer; schedule; regularization; sequence | Adam (0.9, 0.95); linear warmup 뒤 0으로 linear decay; dropout 0.1, grad clip 1.0; document attention masking으로 2048 token packing; loss는 target token만 | §3.1-3.3 | verified 2026-09-15 | §3.2: document masking이 stability·performance 향상(table 없음) |
| OPT-IML 30B / 175B | 30B / 175B | SFT | mixture | EPS 4,096; CrossFit/ExMix/FLAN/NIV2/PromptSource/T5/UnifiedSKG = 4/2/20/25/45/2/2 (%); pretraining 5%, reasoning 1%, dialogue 0% | §4.2-4.7 | verified 2026-09-15 | Tables 4, 5, 7 및 Figs. 4-5 (OPT 30B, 4,000 steps) |
| LIMA | 65B | SFT | data; epochs; optimizer; LR; batch | 1,000 examples(약 750,000 tokens); 15 epochs; AdamW β1 0.9, β2 0.95, weight decay 0.1; warmup 없음, 1e-5→1e-6 linear decay; 32 examples; 2048 token 초과 text trim | arXiv:2305.11206v1 §3, Table 1 | verified 2026-09-15 | ablation 없음 |
| LIMA | 65B | SFT | dropout; checkpoint rule | residual dropout bottom 0.0→top 0.3; 50-example development set에서 epoch 5~10 수동 선택 | §3, App. B | verified 2026-09-15 | App. B Fig. 9: perplexity와 generation quality가 함께 상승 |
| Revisiting-SAH task runs (Llama-3-8B, Mistral-7B, Llama-2-7B; 70B variants) | 7B-70B | SFT | epochs; LR; batch; PEFT | 3 epochs; 1e-5 cosine to 0; batch 2 / 8 / 16 (0-100 / 101-1,000 / 1,001+ examples, 8B·13B), 16 / 32 / 128(70B); PEFT 없음 | arXiv:2410.03717v1 §3.1, App. A.5 Table 6 | verified 2026-09-15 | A.5: small dataset에서 batch size가 “has a big effect”(table 없음) |
| Response-tuning and single-task runs | 7B | SFT | LR; schedule; batch; epochs | Llama-2-7B 1e-5, OLMo-7B-Feb2024 3e-6; cosine to 0, 10% linear warmup; 64; `{5,7,10,15,20}`에서 epochs 선택 | arXiv:2409.14254v1 App. A, §4.1 | verified 2026-09-15 | 저자 validation set의 GPT-3.5-turbo 대비 AlpacaEval win rate로 epoch 선택 |
| Tülu (LLaMA) | 7B, 13B | SFT | epochs; LR; loss masking | 2 epochs; 2e-5; assistant token에만 loss | [[tulu-1-how-far-can-camels-go]] ledger (arXiv:2306.04751v2 App. D, §3.2) | verified 2026-09-14 | ablation 없음 |
| URIAL (base model, training 없음) | 7B-70B | eval-gate | prefix; decoding | system prompt + restyled example `K = 3`, 1,011 tokens; greedy | arXiv:2312.01552v1 §3.3, §4.1 | verified 2026-09-15 | Table 1: `K = 1, 3, 8` 비교 |

**작은 범용 run의 출발점.** 7B-8B base model과 task당 1,000~10,000 example에서는 3 epoch full fine-tuning, learning rate 1e-5, 0까지 cosine decay, 1,000 example 초과 시 batch 16을 사용한다. 이는 Llama-3-8B·Mistral-7B·Llama-2-7B의 single-task Revisiting-SAH setting을 재현한다. OPT-IML·Tülu처럼 response token에만 loss를 계산한다. Examples-proportional sampling과 per-dataset cap으로 source를 섞는다. FLAN은 137B에서 mixing-rate max 3,000을 썼고, OPT-IML은 30B에서 4,096 미만의 모든 cap이 비슷했다고 찾았다. 어느 쪽도 7B에서 시험하지 않았다. Suite에 CoT evaluation이 있으면 CoT data를 넣는다. Flan-PaLM Proportion B의 CoT는 1.8%(App. E는 Flan-T5를 포함한 §3-4 밖 run에서 B를 썼다고 명시)이고 OPT-IML은 30B·175B에서 1%를 썼다. Checkpoint는 §8 rule 4의 T0 rule로 선택한다.

## 일반화 관점

**(a) 범위를 넓히는 요인.**

- 더 많은 task cluster와 task: FLAN은 cluster 1개에서 7개로 늘리며 49.9에서 63.5로 올랐다([[flan]] Fig. 6). Task 수에 따라 log-linear하게 성장했으며([[super-natural-instructions]] §7.1; [[flan-collection]] Fig. 4), Flan-PaLM 8B의 normalized score는 6.4에서 21.9로 올랐다([[flan-palm-scaling-instruction-finetuning]] Table 3). **반복 재현됨.**
- 고정된 budget에서 더 많은 서로 다른 instruction([[instruction-diversity-unseen-tasks]] §4.1), dataset당 더 많은 prompt([[t0-multitask-prompted-training]] §6.2).
- Zero-shot, few-shot, CoT 형식을 섞고, held-out task에 대해 input inversion을 포함하는 것([[flan-collection]] Table 1).
- 더 크거나 더 잘 pretrain된 base model([[flan]] Fig. 7; [[false-promise-imitating-proprietary-llms]] §4.3; [[urial]] Table 1).

**(b) 범위를 좁히거나 forgetting을 일으키는 요인.**

- 작은 model: FLAN setting에서는 8B 이하에서 held-out loss가 발생했다([[flan]] Fig. 7).
- 빠진 형식: CoT data가 없으면 CoT accuracy가 낮아지고([[flan-palm-scaling-instruction-finetuning]] §4.2), CoT-only data는 direct MMLU를 낮춘다(Table 3).
- 형식 지배적인 source: dialogue data 0.5%는 short-answer format 준수를 낮췄고([[opt-iml]] §4.7), single source는 GSM·BBH·TyDiQA를 base model보다 낮췄다([[tulu-1-how-far-can-camels-go]] Table 3).
- Held-in task가 너무 많은 경우: 약 200개 task 이후 Held-In accuracy가 떨어졌다([[flan-collection]] Fig. 4).
- 폭넓은 imitation data는 knowledge QA를 낮췄고([[false-promise-imitating-proprietary-llms]] Table 1), multitask tuning 뒤에는 task별 regression이 나타났다([[opt-iml]] Table 9).

**(c) 이 단계에서의 측정 방법.**

- §8의 여덟 규칙을 적용하고 L1/L2/L3를 따로 보고하며 base-model row를 둔다.
- 가장 좋은 template이 아니라 unseen template 전체에서의 median과 IQR을 보고한다([[t0-multitask-prompted-training]] §5).
- Data source별 development 대 unseen delta를 보고한다([[tulu-3]] Table 32).
- 알려진 측정 오류도 함께 본다. Judge win rate는 문체를 보상한다([[revisiting-superficial-alignment-hypothesis]] Table 2; [[tulu-1-how-far-can-camels-go]] §5.4에서 AlpacaEval win rate와 unique token 사이의 `r = 0.96`). Perplexity는 generation quality를 추적하지 못한다([[lima]] App. B). Held-out suite score로 checkpoint를 선택하면 그 score가 selection signal이 되어, 더 이상 목표로 삼지 않은 task의 성능을 편향 없이 추정하지 못한다(해석; [[flan-palm-scaling-instruction-finetuning]] §2.2는 이 방식으로 step을 골랐고 그 효과 크기는 측정하지 않았다). N-gram contamination check는 false positive를 만들며([[flan]] App. C), paraphrase된 test item을 겨냥하지 못한다. Tülu 3는 embedding matching으로 이를 찾으려 했지만 distributional similarity와 구분하지 못했다([[tulu-3]] §3.2). Contamination 방법은 [[ch-48]]이 다룬다.

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 방법 |
|---|---|---|
| Dataset은 holdout했지만 같은 cluster의 다른 dataset으로 학습한 뒤 결과를 L3라고 부름 | Cluster 전체를 학습에서 빼면 “unseen” task의 gain이 줄어듦 | Cluster holdout과 neighbor-cluster removal로 split을 다시 구성한다(§8 규칙 1-2) |
| Evaluation dataset의 input-inverted 또는 reformat된 version으로 학습함 | 그 dataset의 unseen score가 같은 cluster의 다른 held-out task 점수보다 높음 | 원래 dataset ID를 key로 한 source-dataset exclusion list; 13-gram 또는 8-gram overlap report |
| 가장 좋은 template만 보고함 | 보고 점수가 template 전체 median보다 IQR 이상 높음 | Held-out template 전체의 median과 IQR을 보고한다 |
| Unseen suite에서 checkpoint나 mixture를 선택함 | 모든 의사결정 과정에서 unseen·development curve가 함께 움직임 | 각 결정이 어느 suite를 읽었는지 log로 남기고, 선택을 고정한 뒤에만 unseen score를 읽는다 |
| Judge win rate만 SFT metric으로 사용함 | Win rate는 오르는데 exact-match accuracy는 그대로이거나 떨어짐 | 모든 win rate 옆에 objective metric을 보고한다 |
| 단일 format의 대형 source를 추가함 | Label이 필요한 short-answer·classification task에 여러 문장 output이 나옴 | Source 추가 전후 category별 format-compliance rate |
| Filtering 중 CoT example을 제거함 | Direct-prompt score는 안정적인데 CoT score가 떨어짐 | CoT와 non-CoT evaluation을 별도 column으로 유지한다 |
| 좁은 SFT run은 행동도 좁다고 간주함 | Model이 unsafe instruction을 포함한 무관한 instruction도 따름 | 모든 narrow fine-tune에 general chat·safety suite를 실행한다 |
| Validation perplexity로 checkpoint를 선택함 | 선택한 checkpoint의 judged quality가 더 늦은 checkpoint보다 낮음 | LIMA App. B처럼 generation-based development metric을 쓴다 |
| Distinct instruction 대신 example 수를 셈 | Data를 두 배로 늘려도 held-out score가 변하지 않음 | Example count와 함께 distinct instruction·cluster 수를 보고한다 |

## 이해 점검

1. FLAN의 8B model은 instruction tuning 뒤 held-out accuracy를 잃었지만 T0의 3B model은 얻었다. 두 setup의 차이를 두 가지 열거하고, 각각이 어떻게 반대 결과를 낼 수 있는지 설명하라.
2. Cap 없는 examples-proportional mixing은 model을 실질적으로 형성하는 task 수를 왜 줄이는가? §2.1의 cap은 이를 어떻게 바꾸는가?
3. Flan-PaLM에서 8B model은 더 많은 point를 얻었고 540B model은 error를 더 많이 제거했다. 크기 간 instruction-tuning benefit을 비교할 때 어느 quantity를 사용하며, 그 이유는 무엇인가?
4. Flan Collection에서 few-shot template을 제거하자 zero-shot CoT accuracy는 올랐지만 few-shot MMLU는 낮아졌다. 이것은 단일 average를 보고하는 일에 대해 무엇을 뜻하는가?
5. Rewrite-rule 결과(300개 미만 instruction은 `10^6` example에서도 일반화하지 않음)와 LIMA의 2K~32K example plateau는 training set의 같은 어떤 성질 때문에 함께 일어날 수 있는지 설명하라.
6. 1,000개 LIMA example으로 학습한 model이 GSM8K comparison에서 84.4%를 이기지만 점수는 14.7%다. Training data에서 두 숫자로 이어지는 인과 사슬을 설명하라.
7. Response-tuned model은 instruction을 보지 않고도 43% win rate에 도달한다. 이는 범용 model을 위한 SFT data가 반드시 공급해야 하는 것과 공급하지 않아도 되는 것에 대해 무엇을 뜻하는가?
8. 한 data source가 IFEval을 19.2 point 올리고 IFEval-OOD를 0.4 point 낮춘다. 이 pattern이 capability gain이 아니라 development suite overfitting을 나타내는 이유를 설명하라.

## 연결

- 이전: [[ch-29d]] — User Simulators, Trajectory Verification, Failed Trajectory.
- 다음: [[ch-30]] — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, Learning Rate.
- 의존성: [[ch-00]] — What General Capability Means and How It Is Measured; [[ch-29]] — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification.
- 여기에서 사용하는 앞 장: [[ch-19]] — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing(imitation 대 capability); [[ch-22]] — Quality, Diversity, and Gradient-Based Data Selection.
- §8 protocol을 재사용하는 뒤 장: [[ch-30a]] — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares; [[ch-31a]] — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split.
- Evaluation 단계: [[ch-47]] — Evaluation Harness and Suite Design for General Capability; [[ch-48]] — Contamination Detection and Its Effect on Reported Scores.

## 출처

- [[flan]] — leave-one-cluster-out protocol, mixing rule, cluster·scale·instruction·template ablation, contamination appendix, recipe row.
- [[t0-multitask-prompted-training]] — held-out task selection, held-out data 없이 하는 checkpoint rule, median/IQR 보고, prompts-per-dataset·dataset ablation.
- [[super-natural-instructions]] — 1,616-task benchmark, source-dataset exclusion, Table 3 result, task/instance/size scaling, encoding·negative-example ablation.
- [[flan-palm-scaling-instruction-finetuning]] — task·size scaling(Table 3), normalized metric, CoT ablation, checkpoint-selection rule, hyperparameter·mixture cap.
- [[flan-collection]] — method ablation(Table 1), mixed prompt setting, Held-In 대 held-out task scaling, source removal(Table 2).
- [[opt-iml]] — 세 generalization level, 13-gram leakage check, EPS, benchmark proportion, pretraining/reasoning/dialogue data effect, recipe.
- [[instruction-diversity-unseen-tasks]] — 통제된 instruction count, semantic spread, distribution, LoRA 실험.
- [[lima]] — superficial alignment hypothesis, data composition, preference·absolute result, diversity/quality/quantity ablation, dialogue, recipe(숫자는 arXiv:2305.11206v1에서 가져옴).
- [[urial]] — token distribution shift, tuning-free alignment result, 명시된 한계.
- [[revisiting-superficial-alignment-hypothesis]] — power-law fit, LIMA-1k 대 task-1k accuracy·win rate, error analysis, new-knowledge experiment, recipe.
- [[false-promise-imitating-proprietary-llms]] — imitation style 대 capability 수치(Tables 1-2).
- [[instruction-following-without-instruction-tuning]] — response tuning, response ranking capability, single-task tuning, rule-based adapter, recipe.
- [[tulu-1-how-far-can-camels-go]] — dataset별 capability regression, AlpacaEval length bias, over-refusal 관찰, recipe row.
- [[tulu-3]] — development 대 unseen suite 설계, decontamination threshold, Table 32 data-source ablation(숫자는 arXiv:2411.15124v5에서 가져옴).
- [[ifbench]] — IFEval template 대 unseen constraint type.
