# Chapter 49 — Judge 모델: 편향, 보정, Judge별 과적합

> **핵심 통찰.** judge 점수는 측정 가능하고 재현 가능한 편향을 지닌 모델이 산출하는 추정치이며, 그 편향의 크기는 순위를 바꿀 만큼 크다. MT-Bench position-bias probe에서 두 답변을 교환한 후에도 GPT-4는 65.0%의 경우 같은 판정을 내렸고, GPT-3.5는 46.2%, Claude-v1은 23.8%였다([[judge-llm-bias]] Table 2). AlpacaEval에서는 장황함에 대한 지시만 바꾸어 한 모델의 승률이 22.9%에서 64.3%로 이동했다([[length-controlled-alpacaeval]] §4.1). 지시를 무시하는 상수 문자열은 AlpacaEval 2.0에서 86.5%의 length-controlled 승률과 MT-Bench에서 9.55점을 기록했다([[null-model-cheating-benchmarks]] Table 2). Judge benchmark 정확도 역시 downstream policy 품질을 자동으로 예측하지 않는다. [[ppe-reward-model-eval]]은 최상위 reward model의 RewardBench 점수와 post-RLHF Arena 점수 사이의 음의 상관관계를 측정했다(§1, Fig. 4).
>
> **지침.** judge가 두 checkpoint 사이를 결정할 때는 모든 쌍을 두 순서로 실행하고, 승률과 함께 position-consistency rate를 보고하라. consistency는 judge별 특성이며, [[judge-llm-bias]] Table 2에서 세 judge에 걸쳐 23.8%부터 65.0%까지 범위가 있었기 때문이다. 응답의 길이 또는 markdown이 다르면 style-controlled 점수([[length-controlled-alpacaeval]] Eq. 1의 AlpacaEval GLM 또는 [[lmsys-style-control]]의 Bradley–Terry style feature)를 보고하라. style control은 verbosity prompt 전반의 정규화 표준편차를 25%에서 10%로 줄이고 Chatbot Arena와의 Spearman 상관을 0.94에서 0.98로 높였기 때문이다. Judge family가 훈련용 preference label도 제공했다면, 다른 family의 두 번째 judge를 추가하고 두 수치를 모두 보고하라. 동일 프롬프트에서 Arena-Hard-Auto의 human ranking과의 일치는 GPT-4-Turbo judge일 때 90.9%, Claude-3-Opus judge일 때 66.7%, ensemble일 때 91.5%로 달라졌기 때문이다([[arena-hard-benchbuilder]] Table 4). 결정이 내부 비교가 아니라 release gate라면 judge 점수 leaderboard 하나에만 의존하지 말고 held-out human-labeled set과 verifiable-correctness set을 사용하라. null-model exploit과 선택적 private testing 모두 capability를 높이지 않고 leaderboard 점수를 높이기 때문이다([[null-model-cheating-benchmarks]], [[leaderboard-illusion]]).

---

## 이 장이 범용 모델에 중요한 이유

능력의 폭은 정답 참조가 없는 작업, 즉 개방형 글쓰기, 다중 턴 지원, 설명 품질, 스타일에 민감한 instruction following에서 측정된다. 이런 작업의 측정 도구는 다른 언어 모델이다. 이 과정에서 “모델이 RL 이후에도 일반 능력을 유지했다”는 형태의 모든 주장은 일부나마 judge 수치에 근거한다.

이는 범용 모델에 두 가지 뚜렷한 위험을 만든다.

1. **측정 위험.** judge가 두 checkpoint의 차이와 상관된 편향을 지니면, 예를 들어 한쪽은 더 장황하고, 한쪽은 judge 자신과 같은 family이며, 한쪽은 markdown을 더 많이 생성하면, 측정된 차이의 일부는 그 편향이다. 이는 도구의 속성이며, 누구도 game하려 하지 않아도 존재한다.
2. **최적화 위험.** 평가 점수를 매기는 동일 judge family에서 훈련 신호를 유도하면, 훈련 루프는 capability를 높이지 않고 judge 점수를 높일 수 있다. Chapter 42는 훈련 시 reward model에 대해 이를 다뤘다. 여기서는 보상 곡선을 검사할 수 없어서 더 알아차리기 어려운 평가 시점에 같은 메커니즘이 나타난다.

이 장은 [[ch-48]]이 benchmark 점수가 contamination으로 부풀 수 있음을 확립한 뒤의 평가 단계에 놓인다. Contamination은 모델이 시험 항목을 봤기 때문에 점수를 부풀린다. Judge별 과적합은 모델이 채점자의 선호에 맞았기 때문에 점수를 부풀린다. 둘은 전달되지 않는 수치라는 동일한 증상을 만들지만, 탐지 방법은 서로 다르다.

---

## §1 Judge 점수의 의미와 judge 채점 benchmark의 구성 방법

**정의.** LLM judge는 task 입력과 하나 또는 두 개의 후보 응답을 받도록 prompt된 언어 모델이다. 이 모델은 한 응답의 점수(single-answer grading) 또는 둘 중 더 나은 답을 지목하는 판정(pairwise comparison)을 출력한다. Judge 채점 benchmark는 prompt 집합, judge 모델, judge prompt, baseline 또는 reference 모델, 개별 판정을 후보 모델 하나당 하나의 수치로 바꾸는 aggregation rule을 고정한다.

**해결하는 문제.** 개방형 응답에 대한 사람 평가는 느리다. [[chatbot-arena]] §6.3에서 외부 자원으로 사실 확인을 포함한 expert의 pairwise comparison 하나 라벨링에는 3~5분이 걸렸다. Judge는 같은 판정을 몇 초 만에 낸다. Arena-Hard-Auto는 500개 prompt에서 모델당 평가 비용을 $20으로 보고한다. 모델당 10,000개가 넘는 prompt를 사용하는 Chatbot Arena의 비용은 “very high”다([[arena-hard-benchbuilder]] Table 1).

**세 가지 기준점.** 다음은 이 장에서 사용하는 구성이다. 각 행은 출판된 그대로의 구성이며, judge 수치는 행 사이에서 비교할 수 없다.

| Benchmark | 프롬프트 | Judge와 baseline | 집계 | 출처 위치 |
|---|---|---|---|---|
| MT-Bench | 8개 범주에 걸친 80개 질문, 각각 2 turn | GPT-4; pairwise 및 single-answer (1–10) grading | 평균 점수 또는 사람 투표에 대한 pairwise agreement | [[judge-llm-bias]] §3.1, §4.1 |
| Chatbot Arena | 실시간 crowdsourced 사용자 prompt; 논문 시점에 약 90K 사용자로부터 240K표 이상 | Judge가 아닌 사람 crowd 투표 | active sampling 및 confidence interval을 사용하는 Bradley–Terry 계수 | [[chatbot-arena]] §1, §4 |
| Arena-Hard-Auto | Chatbot Arena 데이터에서 BenchBuilder가 선별한 500개 prompt | judge로 gpt-4-1106-preview; baseline으로 gpt-4-0314 | baseline에 대한 pairwise comparison의 Bradley–Terry fit, 95% CI를 위한 100 bootstrap round 포함 | [[arena-hard-benchbuilder]] §5, §6.1 |
| AlpacaEval 2.0 | 805개 instruction | annotator 및 baseline 모델로 gpt-4-1106-preview | 승률 및 GLM에서 얻는 length-controlled 승률 | [[length-controlled-alpacaeval]] §3; [[null-model-cheating-benchmarks]] Table 1 |

**구성 기준.** [[arena-hard-benchbuilder]] §3은 모델 비교에 쓰는 benchmark의 두 요구사항, 즉 separability와 human preference와의 agreement를 제시하고, 이를 위한 세 metric을 정의한다.

- **Separability with Confidence**: bootstrapping으로 계산한, benchmark 점수의 confidence interval이 겹치지 않는 모델 쌍의 비율이다.
- **Agreement with Confidence Interval**: 두 benchmark A와 B 및 모델 쌍에 대하여, 둘 다 그 쌍을 신뢰 있게 분리하고 순서에도 동의하면 +1, 둘 다 분리하지만 순서에는 동의하지 않으면 −1, 어느 하나라도 분리할 수 없으면 0을 준다.
- **Pair Rank Brier Score**: benchmark가 예측한 pairwise 순서 확률의 제곱 오차다.

*계산 예시.* [[arena-hard-benchbuilder]] Table 1의 20-모델 집합에는 20 × 19 / 2 = 190개의 모델 쌍이 있다. separability 87.4%는 0.874 × 190 ≈ 166개 쌍의 95% interval이 겹치지 않았음을 뜻한다. MT-Bench의 22.6%는 약 43개 쌍을 뜻한다. 나머지 147개 쌍에서 MT-Bench 차이는 bootstrap noise와 구별되지 않는다. 이것이 MT-Bench만으로 checkpoint를 비교하는 것이 약한 정량적 이유다. 같은 실행에서 나온 두 checkpoint는 near-tie 쌍이고, near-tie 쌍은 MT-Bench가 분리하지 못하는 쌍이기 때문이다.

해당 20-모델 집합에서 보고된 값(Table 1)은 다음과 같다. Arena-Hard-Auto separability 87.4%, MT-Bench 22.6%, AlpacaEval 2.0 LC 83.2%, Chatbot Arena 85.8%; Brier score는 0.069 / 0.09 / 0.11 / n.a. **결과(단일 연구).**

**조건과 한계.** 이 metric들은 2024-04-13의 Chatbot Arena reference ranking에 대해, 두 leaderboard에 모두 있던 20개 모델로 계산했다. 이것들은 benchmark가 출시된 모델의 human *ranking*을 얼마나 잘 재현하는지는 말해 주지만, 내부 평가가 수행하는 비교인 한 훈련 실행 내 checkpoint 순위를 재현하는지는 말해 주지 않는다.

---
## §2 측정된 편향 목록

명시적인 측정 절차를 갖춘 네 가지 편향에는 출판된 측정치가 있다. 각각을 judge의 일반 속성이 아니라 “probe, 수치, 보정”으로 읽어야 한다.

### 2.1 위치 편향

**정의.** Judge의 판정이 어느 후보가 먼저 제시되는지에 따라 달라진다.

**측정 절차**([[judge-llm-bias]] §3.3): 각 첫 번째 turn MT-Bench 질문에 대해 temperature 0.7에서 GPT-3.5를 두 번 호출하여 비슷한 두 답변을 구성하고, 각 쌍을 두 순서로 judge한다. *Consistency*는 교환해도 판정이 바뀌지 않는 경우의 비율이다.

| Judge | Prompt | Consistency | 첫 번째 쪽 편향 | 두 번째 쪽 편향 | 오류 |
|---|---|---|---|---|---|
| Claude-v1 | default | 23.8% | 75.0% | 0.0% | 1.2% |
| Claude-v1 | rename | 56.2% | 11.2% | 28.7% | 3.8% |
| GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-3.5 | rename | 51.2% | 38.8% | 6.2% | 3.8% |
| GPT-4 | default | 65.0% | 30.0% | 5.0% | 0.0% |
| GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

([[judge-llm-bias]] Table 2. “rename” prompt는 assistant의 이름을 바꾸므로 위치와 이름을 분리한다. Claude-v1이 23.8%에서 56.2%로 오른 것은 “Assistant A”라는 이름에 대한 선호의 증거다.)

**보정.** 논문의 보수적 규칙은 다음과 같다. 순서를 바꾸어 judge를 두 번 호출하고, 두 번 모두 같은 답을 선호할 때만 승리로 선언하며, 일관되지 않은 쌍은 무승부로 센다(§3.4). 대안으로 위치를 무작위로 배정할 수 있는데, 많은 쌍에 대해 기대값상 편향은 없지만 쌍별로는 noisy하다.

*계산 예시.* 200개 평가 쌍과 GPT-4 default 행을 생각하자. 보수적 규칙에서는 0.650 × 200 = 130개 쌍이 결정을 내고 70개는 무승부가 된다. GPT-3.5를 judge로 쓰면 0.462 × 200 ≈ 92개 쌍이 결정을 내고 108개는 무승부가 된다. 따라서 더 약한 judge에서는 동일 평가가 결정된 쌍을 약 40% 더 적게 가지며, 이 judge에서 계산하는 모든 confidence interval이 넓어진다. 교환은 공짜가 아니다. Judge 호출 수가 두 배가 되고 유효 표본 크기도 judge에 의존하여 줄어든다. 따라서 승률을 해석 가능하게 하려면 consistency rate를 승률 옆에 보고해야 한다.

**같은 효과의 두 번째 측정.** [[self-preference-recognition]] §2.5는 요약 쌍에서 옵션 순서를 반대로 했을 때 evaluator가 pairwise 선호를 뒤집는 비율을 측정했다. task와 dataset 전반 평균으로 GPT-4 25%, GPT-3.5 58%, Llama 2 89%였다(Fig. 5). 이들의 보정은 일관되지 않은 쌍을 버리는 대신 두 순서의 logit 기반 confidence score를 평균하는 것이다. **재현됨**: 서로 다른 데이터의 독립된 두 연구가 모두 수십 퍼센트의 순서 반전을 발견했으며, 시험한 모델 중 GPT-4가 가장 일관됐다.

**완전한 해결책이 아닌, 보고된 완화책.** 세 judgment 예시를 넣은 few-shot judge prompt는 GPT-4의 consistency를 65.0%에서 77.5%로 높였다([[judge-llm-bias]] Table 12). prompt 비용은 약 4×였으며, 저자들은 더 높은 consistency가 더 높은 정확도를 뜻하지 않고 few-shot 예시가 새 편향을 도입할 수 있다고 명시한다(§3.4).

### 2.2 장황함 편향

**측정 절차**([[judge-llm-bias]] §3.3): 번호 목록이 들어 있는 MT-Bench 답변 23개를 가져와 GPT-4에 정보를 추가하지 않고 목록을 바꾸어 쓰게 한 뒤, 바꾸어 쓴 항목을 앞에 붙인다. 따라서 5개 항목 답변은 같은 내용을 지닌 10개 항목 답변이 된다. Judge가 더 긴 버전을 선호하면 공격은 성공한다.

| Judge | Claude-v1 | GPT-3.5 | GPT-4 |
|---|---|---|---|
| “repetitive list”에서의 실패율 | 91.3% | 91.3% | 8.7% |

([[judge-llm-bias]] Table 3.) 동일한 두 답변에는 같은 judge들이 무승부를 반환한다. 그러므로 실패는 깨진 비교 절차가 아니라 추가된 중복 길이에 특이적이다.

**조건과 한계.** N = 23개 답변과 한 가지 공격 형태다. 이 수치는 일반적인 길이 계수가 아니라 이 조작에 대한 취약성을 정량화한다. 일반 길이 계수는 §3이 회귀로 추정하는 값이다.

### 2.3 자기 선호

**약한 버전과 그 주의사항.** [[judge-llm-bias]] §3.3은 같은 쌍에서 각 judge의 승률을 사람 승률과 비교했고(Fig. 3b), GPT-4는 자신에게 10% 높은 승률을, Claude-v1은 자신에게 25% 높은 승률을 주었다고 밝혔다. 그러나 저자들은 “제한된 데이터와 작은 차이 때문에, 본 연구는 모델들이 self-enhancement bias를 보이는지 결정할 수 없다”고 썼다. 통제된 연구에는 품질을 바꾸지 않고 응답을 다른 모델의 스타일로 바꾸어 쓰는 작업이 필요하기 때문이다. 두 수치는 확립된 효과 크기가 아니라 관찰로 취급해야 한다. 그 논문이 제시한 **열린 질문**이다.

**통제된 버전.** [[self-preference-recognition]]은 두 속성을 분리한다. *self-preference*는 evaluator가 자기 출력을 다른 사람이 매긴 것보다 높게 점수화하는 것이고, *self-recognition*은 evaluator가 어떤 출력이 자신이 쓴 것인지 알아낼 수 있는 것이다. Llama-2-7b-chat, GPT-3.5, GPT-4를 사용한 XSUM 및 CNN/DailyMail 요약에서의 결과는 다음과 같다.

1. 세 evaluator 모두 fine-tuning 없이 자신의 요약과 다른 출처의 요약을 구별하는 정확도 50%를 넘는다. GPT-4는 다른 두 LLM 및 사람을 상대로 73.5%에 이른다(§1, finding 2).
2. 500개 예시에 대한 fine-tuning은 GPT-3.5와 Llama 2의 self-recognition 정확도를 90% 이상으로 올린다(finding 3).
3. self-recognition을 높이거나 낮추도록 모델을 fine-tuning하면 self-preference도 선형 추세를 따라 이동한다(Fig. 1, finding 4).
4. 응답 길이, 모음 수, Flesch–Kincaid 가독성, 항상 “1”로 답하기, 무작위 답하기처럼 무관한 속성으로 fine-tuning한 모델도 같은 선형 추세에 놓인다. 이것은 fine-tuning이 도입한 confound에 대한 저자들의 통제다(§3.4).

**범용 모델에 대한 함의.** 운영 규칙은 인과 질문을 해결하는 데 의존하지 않는다. Judge family와 후보 family가 같으면 점수의 한 성분은 응답 품질 이외의 것에 기인하며, 그 성분의 크기는 당신의 설정에서 알 수 없다. 측정상의 해결책은 다른 family의 두 번째 judge이며, 보고할 양은 재귀속한 점수가 아니라 두 judge 사이의 불일치다. 두 judge의 차이는 다른 rubric, 다른 calibration, 또는 품질에 대한 진짜 불일치에서 올 수 있다. 따라서 leakage term을 분리하기보다 불확실성을 상한 제한한다.

### 2.4 수학 및 추론 답안의 채점

**측정 절차**([[judge-llm-bias]] §3.3–§3.4): 10개 수학 질문, LLaMA-13B 및 Vicuna-13B의 답변, 각 쌍을 두 순서로 judge하여 20개 판정을 얻는다. GPT-4가 오답을 정답이라고 선언하면 실패다.

| Prompt | Default | Chain-of-thought | Reference-guided |
|---|---|---|---|
| 20개 중 judge 실패 | 14 | 6 | 3 |

([[judge-llm-bias]] Table 4. 본문은 같은 변화를 “70%에서 15%로”라고 보고한다.) 재사용에 중요한 세부사항이 두 가지 있다. 첫째, reference answer는 gold solution이 아니라 **후보를 보기 전에 독립적으로 생성한 judge 자신의 답**이다(§3.4). 이 절차는 judge가 후보에 anchoring되지 않게 하여 작동한다. 둘째, chain-of-thought만으로는 효과를 제거하지 못한다. 저자들은 judge가 자기 추론 안에서 주어진 답변과 같은 실수를 자주 재현한다고 보고한다(§3.4).

**조건과 한계.** N = 10개 질문이다. 보정은 judge가 스스로 답을 만들 수 있는 작업에 적용된다. 개방형 글쓰기에는 reference로 생성할 것이 없다.

### 2.5 agreement 수치의 정확한 진술

흔히 인용되는 “GPT-4는 사람이 서로 동의하는 만큼 사람과 동의한다”는 말은 설정에 의존한다. [[judge-llm-bias]] Table 5(MT-Bench, 첫 turn)는 두 설정을 보고한다. S1은 non-tie, tie, position-inconsistent vote를 포함하고 inconsistent vote를 tie로 센다(random baseline 33%). S2는 non-tie vote만 포함한다(random baseline 50%).

| 쌍 | S1 (R = 33%) | S2 (R = 50%) |
|---|---|---|
| GPT-4 pairwise 대 사람 | 66% | 85% |
| GPT-4 single-answer 대 사람 | 60% | 85% |
| 사람 대 사람 | 63% | 81% |

두 번째 turn에서는 GPT-4 pairwise 대 사람이 66% / 85%, 사람 대 사람이 67% / 82%다. Chatbot Arena 데이터(Table 6)에서는 GPT-4 대 사람이 64%(S1)와 87%(S2)다. Agreement는 두 모델 사이 gap의 크기와 함께 증가한다. 승률 차이가 커질수록 70%에서 거의 100%까지 오른다(Fig. 2).

이 마지막 지점이 checkpoint 비교에 가져가야 할 지점이다. Judge–human agreement는 답이 명백한 바로 그곳에서 가장 높고, 두 후보가 가까운 곳에서 가장 낮다. 후자는 한 실행에서 나온 두 checkpoint를 비교하는 체제다.

동반 그림 **[figures/judge-bias.html](figures/judge-bias.html)**은 위의 측정된 position-consistency rate 중 하나를 선택하고 평가 쌍 수를 설정하여, 보수적 swap 규칙을 통과하는 쌍이 몇 개인지 볼 수 있게 한다. 두 번째 panel은 §3.1의 AlpacaEval length-controlled GLM을 평가하므로 raw 및 length-controlled 예측을 서로 대조해 읽을 수 있다. 두 panel은 이 절에서 인용한 수치를 사용한다.

---
## §3 회귀로 알려진 교란 요인을 제거하는 스타일 제어

### 3.1 길이 제어 AlpacaEval

**측정된 문제.** 모델에 “가능한 한 자세히 답하라”라고 prompt하는 것과 “가능한 한 간결하게 답하라…”라고 prompt하는 것은 gpt4_1106_preview의 raw AlpacaEval 승률을 22.9%에서 64.3%로 움직였다([[length-controlled-alpacaeval]] §4.1). 따라서 이 metric은 어떤 능력도 바꾸지 않는 지시에 반응한다.

**메커니즘.** 모델 정체성, 길이 차이, instruction 난이도라는 세 항에서 annotator의 선호를 예측하는 logistic regression을 적합한 다음, 길이 항을 0으로 설정하여 평가한다. 적합된 모델(Eq. 1)은 다음과 같다.

```
q(y = 1 | z_m, z_b, m, b, x) =
    logistic( θ_m − θ_b  +  φ_{m,b} · tanh( (len(z_m) − len(z_b)) / std(len(z_m) − len(z_b)) )  +  (ψ_m − ψ_b) · γ_x )
```

- `z_m`, `z_b`: instruction `x`에 대한 평가 모델 `m`과 baseline `b`의 출력이다.
- `θ_m`, `θ_b`: 모델별 strength coefficient다.
- `φ_{m,b}`: 모델 쌍의 길이 차이에 대한 민감도다.
- `len(·)`: 출력 길이다. 차이는 단위 분산으로 표준화된 다음 `tanh`를 통과하므로 길이 차이는 log odds에 감소하는 효과를 낸다.
- `ψ_m`, `ψ_b`: 모델별 instruction 난이도 민감도이고, `γ_x`는 instruction `x`의 난이도다.
- `logistic(u) = 1 / (1 + e^{−u})`.

Length-controlled 승률(Eq. 2)은 길이 항을 제거한다.

```
winrate_LC(m, b) = 100 · E_x [ logistic( θ_m − θ_b + (ψ_m − ψ_b) γ_x ) ]
```

이 featurization은 raw 승률이 갖는 두 항등식을 만족한다. 모델을 자기 자신과 비교하면 `q = 0.5`이고, `m`과 `b`를 바꾸면 `tanh`가 홀함수이고 나머지 항이 부호를 바꾸므로 `q`는 `1 − q`로 바뀐다(§3). `M`개 모델과 `N`개 instruction이면 GLM은 `M·N`개 예시에서 적합된 `3M + N`개 parameter를 가진다. AlpacaEval은 `M > 128`, `N = 805`다.

*계산 예시.* 한 instruction에 대해 `θ_m − θ_b = 0.30`, `φ_{m,b} = 0.80`, 표준화된 길이 차이가 `1.0`이어서 `tanh(1.0) = 0.762`이고 instruction 항이 `0`이라고 하자. Raw 예측은 `logistic(0.30 + 0.80 × 0.762) = logistic(0.910) = 0.713`이다. 길이 항을 0으로 설정하면 `logistic(0.30) = 0.574`가 된다. 이 instruction에서는 71% 승률 중 14 point가 모델이 baseline보다 길다는 데 기인한다. 이를 805개 instruction에 걸쳐 반복하고 평균하면 보고된 LC 승률을 얻는다.

**증거.** 길이 제어는 세 verbosity prompt 전반에서 모델 승률의 정규화된 표준편차를 25%에서 10%로 낮췄다. gpt4_1106_preview의 범위는 22.9–64.3%에서 41.9–51.6%로 좁아졌다. 둘 모두에 존재한 38개 모델로 계산한 Chatbot Arena와의 Spearman 상관은 0.94에서 0.98로 올랐다(§4.1–§4.2, Table 1). 대안 제어의 점수는 더 낮았다. length-normalized는 상관 0.96 / gameability 15%, length-balanced는 0.95 / 15%였고 GLM의 8.5 대비 40.8-point adversarial 승률 상승이 있었다(Table 1). **결과(단일 연구).**

### 3.2 Chatbot Arena 및 Arena-Hard-Auto에서의 스타일 제어

[[lmsys-style-control]]은 같은 생각을 사람 투표에 적용한다. Arena 점수는 모델 indicator vector `X_i`에 대한 battle outcome `Y_i`의 Bradley–Terry logistic regression이며, style control은 자체 coefficient를 갖는 style feature `Z_i`를 더한다.

```
β̂, γ̂ = argmin_{β, γ}  (1/n) Σ_i BCELoss( sigmoid( X_iᵀ β + Z_iᵀ γ ), Y_i )
```

각 style feature는 `normalize( (feature_A − feature_B) / (feature_A + feature_B) )`로 정의한다. 네 feature는 답변 token 길이, markdown header 수, markdown bold 수, markdown list 수다. 길이와 markdown을 모두 제어할 때 적합된 coefficient는 길이 0.249, markdown list 0.031, header 0.024, bold 0.019다. 길이만 제어하면 0.267이다. 저자들은 길이가 지배적인 style factor이며 markdown 효과는 2차적이라고 말한다. 보고된 ranking 효과는 다음과 같다. GPT-4o-mini 및 Grok-2-mini는 대부분 frontier 모델 아래로 내려가고, Claude 3.5 Sonnet, Opus, Llama-3.1-405B는 올라가며, Hard Prompt subset에서 Claude 3.5 Sonnet은 공동 1위가 된다.

[[arena-hard-benchbuilder]] §6.5는 같은 네 style feature를 Arena-Hard-Auto의 Bradley–Terry fit에 추가하고 style-controlled Arena ranking과 비교한다. style control을 적용하면 confidence agreement는 98.6%, 적용하지 않으면 94.4%였고, Spearman은 98.6% 대 94.9%, Kendall은 93.7% 대 85.3%였다(Table 3). 또한 GPT-3.5-Turbo, Llama-3.1-70B-Instruct, Gemini-1.5-Flash에 verbosity와 markdown 사용을 늘리라고 지시했다. Raw 점수는 상승했고, style control은 그 이점을 제거했다(§6.5, App. Table 12).

**조건과 한계.** Style control은 이름을 붙인 feature가 설명하는 분산을 제거한다. 모델링하지 않은 style 효과는 제거하지 못하며 adversarial 조작을 상한 제한하지도 못한다. §4는 null model이 length-controlled metric을 이기는 사례를 보인다.

---
## §4 Judge별 과적합

“Judge별 과적합”은 judge가 대신 서 있던 작업에서 capability가 증가하지 않았는데도 judge 채점 metric이 증가하는 것을 뜻한다. 출판된 세 가지 실증은 그 크기가 얼마나 클 수 있는지 상한을 제시한다.

### 4.1 Null 모델

[[null-model-cheating-benchmarks]]는 입력과 아무 관계 없이 모든 instruction에 하나의 상수 문자열을 반환하는 “null model”을 제출한다. 세 benchmark 모두에서 annotator로 gpt-4-1106-preview를 사용한 결과는 다음과 같다(Table 2).

| 대상 | AlpacaEval 2.0 LC 승률 | AlpacaEval 2.0 raw | Arena-Hard-Auto | MT-Bench 점수 |
|---|---|---|---|---|
| 검증된 SOTA 모델(2024-10-01 이전) | 57.5 | 51.3 | 82.6 | 8.96 |
| Structured cheating 응답 | 76.8 | 64.2 | 67.2 | 7.75 |
| Structured 응답 + random-search prefix | 86.5 | 84.0 | 83.0 | 9.55 |

두 메커니즘은 분리해서 볼 가치가 있다. *Structured* 응답은 annotator의 template을 exploit한다. Judge가 이것을 평가 지시의 끝과 새로운 빈 비교로 parse하게 작성되어 있으며, 이 시점에서 annotator는 첫 번째 identifier를 반환한다(§3, Fig. 2). 이것만으로 76.8% LC를 얻는다. 그다음 *random-search prefix*는 public instruction set을 상대로 최적화한다. Benchmark 자체의 805개 AlpacaEval instruction을 private인 것처럼 취급하고, 그 효과는 세 benchmark 모두로 전이된다(§3, Table 2). 승리 응답의 평균 token 수는 205이므로 §3의 길이 제어가 활성화되어 있으며 결과를 막지 못한다.

**함의.** 길이 제어와 스타일 제어는 비적대적 조건에서 알려진 교란 요인을 고친다. Judge를 겨냥한 최적화에 대한 방어는 아니다. 어떤 제출자든 query할 수 있는 judge에서 나온 leaderboard 수치는 직접 최적화할 수 있는 수치다.

### 4.2 사람 leaderboard에서의 선택적 시험

[[leaderboard-illusion]]은 Chatbot Arena 자체를 감사한다. 보고된 발견은 다음과 같다. Llama-4 출시 전 한 provider가 시험한 private LLM variant 27개, 전체 arena 데이터 중 Google과 OpenAI가 각각 받았다고 추정되는 19.2%와 20.4%(83개 open-weight 모델을 합친 29.7%와 대조), 공식적으로 목록에 있는 47개에 비해 조용히 deprecated된 public 모델 243개 중 205개(조용히 deprecated된 모델의 64%는 open-weight 또는 open-source), 그리고 실험에서 추가 arena-distribution 데이터에 접근했을 때 Arena-Hard에서 최대 112%의 상대적 이득이다(Abstract, §1). 저자들의 해석은 여러 private variant 중 최선만 선택적으로 공개하면 Bradley–Terry fit의 unbiased-sampling 가정을 위반하며, 이 동역학이 일반 품질이 아니라 arena별 행동으로의 과적합을 만든다는 것이다.

이는 적대적 문자열이 없는 §4.1과 같은 실패다. 측정한 `k`개 variant 중 어느 것을 공개할지 고르는 것은 noise에 대한 maximum이고, maximum은 위쪽으로 편향된다.

### 4.3 Judge 선택 자체가 결과의 일부인 이유

[[arena-hard-benchbuilder]] Table 4는 Arena-Hard-Auto의 500개 prompt를 네 judge로 실행하고, 각 결과 ranking을 같은 human reference와 비교한다.

| Judge | Confidence agreement | Separability | Spearman | Brier |
|---|---|---|---|---|
| gpt-4-1106-preview | 90.9% | 87.4% | 93.2% | 0.069 |
| Claude-3-Opus | 66.7% | 83.68% | 77.0% | 0.170 |
| gemini-1.5-pro-0514 | 84.8% | 82.11% | 95.2% | 0.064 |
| llama-3-70b-instruct | 65.6% | 81.6% | 70.5% | 0.196 |
| Ensemble (GPT-4-Turbo + Gemini-1.5-Pro) | 91.5% | 89.5% | 96.5% | 0.065 |

저자들은 기본 judge의 self-bias도 확인했다. GPT 모델은 human preference가 부여한 것보다 약간 높은 평균 ranking을 받고 Claude 모델은 더 낮게 rank되며, 두 judge ensemble은 이를 줄인다(§6.6, App. Table 10). **결과(단일 연구).** 실무적 결과는 “Arena-Hard-Auto 점수”가 모델의 속성이 아니라 모델, judge, style-control 설정을 함께 묶은 속성이라는 것이다.

### 4.4 이 장이 다루지 않는 것

위 측정은 단일 응답 및 pairwise text judging에 대한 것이다. Agent trajectory를 judge할 때에는 다른 실패 표면이 추가된다. outcome predicate, environment state, tool-call sequence가 모두 별도로 채점할 대상이다. 이는 [[ch-51a]]의 주제다. Judge 자신이 채점 전에 긴 입력을 읽어야 하는 long-context judging은 이 장의 출처 집합 중 어느 것도 측정하지 않는다.

---
## §5 Judge 또는 reward model은 일반화하는가?

Judge가 범용 모델에 유용하려면 정확도가 검증된 분포 밖에서도 유지되어야 하고, 점수가 downstream의 무언가를 예측해야 한다. 세 benchmark가 이 질문의 서로 다른 부분을 측정한다.

### 5.1 RewardBench: 범주 전반의 포괄성

[[rewardbench]]는 Chat, Chat Hard, Safety, Reasoning, Prior Sets(이전 preference dataset의 test split)라는 다섯 section의 prompt–chosen–rejected trio 집합이다. 점수는 정확도다. 모델이 chosen 응답에 더 높은 점수를 부여하면 정답이다. Section 안에서는 prompt별 가중 평균을 사용하고, Prior Sets 안에서는 비가중 평균을 사용하며, 최종 수치에서 Prior Sets는 다른 section의 0.5로 down-weight한다(§4.2). 무작위 성능은 50%다. 저자들은 일부 subset은 정확도 100%에 접근하는 반면 Chat Hard와 Reasoning subset은 ceiling이 낮고 분산이 크며, Reasoning은 모델에 따라 35%(무작위 미만)에서 97%까지 걸친다고 보고한다(§5).

이 section 구조가 이것을 generality probe로 만든다. Chat에서는 정확하지만 Reasoning에서는 무작위보다 낮은 judge는 좁은 competence를 가지며, aggregate 수치는 이를 숨긴다.

### 5.2 RM-Bench: 내용과 스타일의 분리

[[rm-bench]]는 각 prompt에 대해 세 style 수준의 chosen 및 rejected 응답을 구성한다. `y∅`는 짧고 평이한 응답, `y^L`은 상세하고 평이한 응답, `y^{L,M}`은 markdown을 포함한 상세 응답이다. 모든 chosen style을 모든 rejected style과 비교하면 3×3 Style–Substance matrix가 생기며, 여기서 세 metric을 도출한다(§3.3).

- **Easy accuracy** = 하삼각형의 평균(chosen 응답도 더 유리한 style을 가짐).
- **Normal accuracy** = 대각선의 평균(두 응답이 style을 공유함).
- **Hard accuracy** = 상삼각형의 평균(*rejected* 응답이 더 유리한 style을 가짐).

*계산 예시.* Chat 도메인에서 `sfairXC/FsfairX-LLaMA3-RM-v0.1`의 출판된 matrix(Fig. 2)를 사용한다. 행은 chosen style이고 열은 rejected style이다.

```
              y_r      y_r^L    y_r^{L,M}
y_c         83.61%    3.83%     2.19%
y_c^L       99.45%   66.12%    49.73%
y_c^{L,M}  100.00%   80.33%    66.67%
```

Easy = (99.45 + 100.00 + 80.33) / 3 = 93.26%. Normal = (83.61 + 66.12 + 66.67) / 3 = 72.13%. Hard = (3.83 + 2.19 + 49.73) / 3 = 18.58%. 같은 reward model이 style과 내용이 일치할 때는 거의 완벽하지만, 충돌할 때는 무작위보다 훨씬 낮다.

거의 40개 reward model 전반에서 최고 평균 정확도는 70.1%(Skywork-Reward-Llama-3.1-8B)이고 hard accuracy는 46.6%로 50% 무작위 baseline보다 낮다(Abstract, Table 3). **결과(단일 연구).**

Downstream 상관에서 [[rm-bench]] §5는 일치한 PPO hyperparameter로 서로 다른 60k preference set에서 훈련한 네 Tülu-v2.5 reward model을 사용한다. reward-model 점수와 GSM8k, Big Bench Hard, HumanEval+, MBPP+, ToxiGen, XSTest에서의 policy 성능을 상관시킨 결과는 Pearson r = 0.55 (p = 0.07)다. 같은 비교에서 RewardBench에 보고된 r = 0.21 (p = 0.51)과 대비된다. 둘 다 작은 표본 상관이며, 논문 자체의 limitations section도 downstream 상관이 제한적이라고 말한다.

### 5.3 PPE: reward-model metric과 post-RLHF 결과의 상관

[[ppe-reward-model-eval]]은 여기서 end-to-end 검사를 실행하는 유일한 출처다. 구성은 Chatbot Arena의 16,038개 human preference pair와 verifiable-correctness preference set, 12개 도메인에 걸친 12개 metric이다(Abstract, §3). 이후 reward model마다 policy를 훈련한다. Llama-3.1-8B-Instruct의 응답으로 만든 8,000 DPO 행을 사용한다. 결과 모델을 Chatbot Arena에 배포하고 6일 동안 사람 투표 12,190개, 모델당 약 2,032 battle을 수집한다(§6.1–§6.2, Table 3).

발견(§7):

1. Human preference set에서의 reward-model **정확도가 post-DPO Arena 점수의 최선 예측변수**다. Row-wise Pearson correlation, confidence agreement, separability에도 일부 예측력은 있지만 더 약하다.
2. **Spearman과 Kendall rank correlation은 최종 Arena 점수와 거의 0의 상관**을 가진다. 저자들의 해석은 정확도는 preference pair별로 측정하는 반면 rank correlation은 모델 전반으로 집계하여 세분성을 잃는다는 것이다.
3. 범주 점수를 **낮은 quantile**에서 집계하는 것은 평균보다 더 잘 예측한다. 낮은 quantile 집계에서 정확도는 downstream Arena 점수와 Pearson 상관 0.80으로 최고에 이른다(Fig. 5). 저자들의 해석은 약한 도메인은 훈련 중 exploit될 수 있으므로 reward model이 모든 입력 분포에서 robust해야 한다는 것이다.
4. Verifiable-correctness 도메인 중 math가 가장 예측력 있는 단일 도메인이며, 그곳에서는 ROC AUC가 정확도보다 더 잘 상관한다.
5. 최상위 reward model에 대해서는 RewardBench 점수와 downstream RLHF 성능 사이의 **음의 상관**을 보고한다(§1, Fig. 4).

**조건과 한계.** Reward model 9개, PPO가 아닌 DPO, 하나의 policy 크기, 하나의 6일 battle window다. 논문은 leakage 위험과 DPO 대 PPO 주의사항을 모두 명시한다(§8).

**범용 모델에 대한 함의.** Judge 또는 reward model에 대한 신뢰 순서는 다음과 같다. policy에 대한 측정된 downstream 효과 > 신선한 human-labeled set에서의 pair별 정확도 > style-robust benchmark 정확도 > aggregate leaderboard rank correlation. 마지막만 보고하는 것은 이용 가능한 증거 중 가장 약하며, [[ppe-reward-model-eval]]은 그것이 가장 예측력이 낮다고 발견했다.

---
## §6 Judge 훈련과 그것이 바꾸는 것

Judge를 소유하면 두 문제가 해결된다. API version이 바뀔 때 instrument가 바뀌는 일이 멈추고, 평가 시 judge를 훈련 시 reward model과 다르게 만들 수 있다. 검증된 결과를 가진 네 구성은 다음과 같다.

### 6.1 Self-Taught Evaluators — 사람 preference label 없는 반복 fine-tuning

Pipeline([[self-taught-evaluators]] §3–§4.1):

1. 라벨 없는 instruction을 가져온다. 이 실행은 Mixtral 22Bx8 Instruct가 분류한 WildChat instruction을 사용하며, “reasoning” 예시 20,582개를 유지한다.
2. Instruction `x`에 대해 좋은 응답 `y_w`를 생성한다. 그다음 `x`와 “highly relevant but not semantically identical”한 **modified instruction** `x′`와 `x′`에 대한 고품질 응답 `y_l`을 작성한다. `y_l`은 다른 질문에 답하므로 쌍 `y_w ≻ y_l`은 사람 annotation이 없는 construction label을 가진다.
3. 현재 모델에서 예시당 `N = 15`개 judgment를 temperature 0.7, top-p 0.9로 샘플한다. 판정이 construction label과 불일치하는 judgment는 버리고, 올바른 judgment 하나를 무작위로 유지하며, 올바른 것이 하나도 없으면 예시를 버리고, A-better와 B-better label의 균형을 맞춘다.
4. Seed 모델에서 **judgment token에 대해서만** negative log-likelihood로 fine-tuning한다. 각 iteration은 이전 checkpoint를 계속하는 대신 Llama-3-70B-Instruct에서 다시 시작한다.

결과(Table 1): RewardBench 전체 점수는 5회 iteration에 걸쳐 75.4(seed) → 83.9, 86.0, 87.5, 87.7, 88.3이며, 32-sample majority vote에서는 88.7이다. GPT-4-0125 점수는 84.3이며 **iteration 2**(86.0)에서 처음 넘어섰다. 사람 label이 있는 HelpSteer2로 훈련한 같은 seed는 85.6에 이른다. 범주별로 seed → iteration 5는 Chat 97.6 → 96.6, Chat Hard 58.9 → 84.2, Safety 69.2 → 91.5, Reasoning 78.5 → 81.0이다.

이 장에 중요한 ablation 결과는 두 가지다. 첫째, construction이 중요하다. “답을 더 나쁘게 바꾸어 쓰라” prompt는 modified-instruction construction의 83.8에 비해 80.7을 준다(§6.2). 둘째, **훈련 종료 시점에도 평가는 순서에 민감하다**. Iteration 5에서 RewardBench 정확도는 승자를 항상 먼저 두면 85.5, 패자를 항상 먼저 두면 91.1이고 평균은 88.3이다(App. A.3, Table 9). Judge로 특화 훈련한 모델도 5.6-point 위치 효과를 지니므로, judge 훈련 후에도 §2.1의 swap protocol은 선택 사항이 아니다.

**저자들이 진술한 조건과 한계**: 70B 모델만 사용, pairwise judgment만 사용, seed가 이미 합리적인 평가를 생성해야 함, compute는 연구하지 않음.

### 6.2 SFR-Judge — judgment pair에 대한 DPO

[[direct-judgement-preference]]는 CoT critique 70%, verdict-only 15%, response deduction 15%의 세 유형으로 된 680K judgment preference pair에서 8B, 12B, 70B judge를 훈련한다(§4.1). Loss는 chosen judgment에 대한 길이 정규화 SFT 항과 DPO 항을 결합한다(§3.4).

```
L = − log M_s(y_w | x) / (|y_w| + |x|)
    − log σ( β log[ M_s(y_w|x) / M_ref(y_w|x) ] − β log[ M_s(y_l|x) / M_ref(y_l|x) ] )
```

- `M_s`: 훈련 중인 judge, `M_ref`: 같은 instruction-tuned initialization의 동결 사본이다.
- `x = (p, i, r)`: protocol(task description 및 rubric), task input, 하나 또는 두 응답이다.
- `y = {c, j}`: critique와 verdict이며, `y_w`는 ground-truth verdict를 갖고 `y_l`은 그렇지 않다.
- `|·|`: token 길이, `σ`: logistic function, `β`: DPO temperature(주 실행에는 보고되지 않음)다.

결과: 7개 benchmark의 pairwise 평균은 GPT-4o의 76.78 및 Self-taught-evaluator-Llama-3.1-70B의 82.26에 비해 84.25(70B), 81.49(12B), 80.91(8B)다(Table 1). RewardBench는 GPT-4o-2024-08-06의 86.7에 비해 92.7 / 90.3 / 88.7이다(Table 7). Inference에서 critique를 제거하면 8B single-rating Pearson은 0.68에서 0.58로, pairwise 평균은 80.97에서 80.05로 낮아진다(App. E.6).

**가져가야 할 측정 주의사항.** RewardBench가 아닌 여섯 pairwise benchmark에 대해서 논문은 응답 순서를 바꾸어 각각 두 번 실행하고 **두 실행 중 더 좋은 결과**를 보고한다(§4.3). 순서 평균 정확도는 보고하지 않는다. 다른 논문의 수치와 비교할 때 어떤 관례가 그 수치를 만들었는지 확인해야 한다.

### 6.3 Generative reward model — 훈련 신호의 출처

[[generative-reward-models]]는 UltraFeedback(61k pair)과 UltraInteract에서 Llama-3.1-8B-Instruct judge를 훈련하고 in-distribution 및 RewardBench에서 평가한다. 일반화에 관해 기억할 결과는 다음이다. 명시적 Bradley–Terry reward model은 UltraInteract에서 in-distribution 약 94%에 이르지만 **RewardBench Reasoning에서는 무작위보다 낮게** 떨어진다. 반면 자기의 correct-versus-incorrect reasoning chain으로 DPO 훈련한 judge(STaR-DPO)는 그곳에서 87.2%에 이른다(§5.2). UltraFeedback에서는 in-distribution에서 Bradley–Terry, PairRM, GenRM이 약 73–74%이고 STaR-DPO는 73.9%다. RewardBench에서는 STaR-DPO가 81.9%, GenRM이 78.9%다(§5.1). 32개 sample의 majority vote는 설정에 따라 1.6~4.9 point를 추가한다(§5.4).

이 논문은 훈련된 judge를 PPO 또는 online preference optimization 안의 reward로 시험하지 않았고, generative RM의 reward hacking을 future work로 든다(§7). 그러므로 이는 judge 정확도에 관한 증거이지 최적화 압력하에서 judge robustness에 관한 증거는 아니다.

### 6.4 루프 안에서 훈련할 때 judge는 drift한다

[[meta-rewarding-lm]]은 Llama-3-8B-Instruct 하나를 actor, judge, meta-judge로 사용하여 네 DPO iteration을 훈련한다. 표제 결과는 AlpacaEval 2 LC 22.92% → 39.44%(Table 1), Arena-Hard 20.6% → 29.1%(Table 2)다. 이 장에 필요한 측정치는 실패 진단이다.

- Meta-judge가 더 높은 점수의 judgment를 선호하는 비율은 iteration 1의 63.04%에서 iteration 2의 **97.68%**로 올랐다(Table 5).
- 모든 judgment pair의 위치 편향은 43.92%에서 68.11%로 올랐다(Table 5). 저자들은 위치 편향이 iteration 3의 추가 개선을 제한했다고 보고한다(§5).
- Judge 훈련 두 iteration 뒤 5-point rubric에서 평균 judge 점수는 약 4.1에서 4.7 이상으로 올랐다(Fig. 5). 점수 분포가 ceiling 쪽으로 압축된다.
- 190개 held-out Open Assistant sample의 Spearman correlation으로 측정한 사람과의 judge agreement는 seed 0.315, iteration 2 0.382, iteration 4 0.326이다(Table 7). 이 이득은 지속되지 않는다.

이 pipeline의 length control은 **rubric 항이 아니라 pair-selection rule**이다. Chosen 응답은 top tier `[(1−ρ)S_max + ρS_min, S_max]` 안에 점수가 드는 응답 중 가장 짧은 것이고, rejected 응답은 bottom tier에서 가장 긴 것이다(§2.1). Iteration 4에서 `ρ = 0`이면 응답은 2212 character에 이르며, `ρ = 0.4`에서는 2003이다(Table 4). Rubric 자체에는 길이 항이 없다(App. A.1).

**함의.** 점수를 매기는 루프 안에서 다시 훈련하는 judge에는 고정된 human-labeled set을 가진 외부 anchor가 필요하다. Anchor statistic, 즉 사람과의 agreement 또는 Spearman은 judge version마다 다시 계산해야 한다. 이 연구에서 anchor statistic은 훈련이 멈추기 두 iteration 전에 최고점에 도달했다.

---
## 부정 샘플과 부정 피드백

Judge 훈련은 과정 표준이 정의한 네 가지 의미 중 세 가지 의미에서 부정 예시를 사용한다(유도식은 [[ch-43a]] 참조). 한 방법이 사용하는 의미를 명시하면 한 의미의 제어를 다른 의미로 잘못 옮기는 일을 막을 수 있다.

**부정 샘플의 출처.** [[self-taught-evaluators]]에서 부정은 판정이 construction label과 일치하지 않는 샘플된 judgment이며, 거부된 *응답* `y_l`은 modified instruction에 대한 좋은 답이다. [[direct-judgement-preference]]에서 부정 judgment는 dataset annotation과 판정이 일치하지 않는 teacher critique다. [[generative-reward-models]]에서 부정은 잘못된 verdict로 끝나는 reasoning chain이다. [[meta-rewarding-lm]]에서 거부 judgment는 meta-judge의 battle matrix에서 Elo가 가장 낮은 judgment다. **이 논문 중 어느 것도 construction의 false-negative rate를 보고하지 않는다.** [[self-taught-evaluators]] §3은 `y_l`을 “likely of lower quality”라고만 기술한다.

**현재 관행이 이들을 다루는 방법.**

| 방법 | “부정”의 의미 | 처리 |
|---|---|---|
| [[self-taught-evaluators]] | (1) 부정 marginal value | 잘못된 judgment는 **버린다**. 훈련은 예시당 유지한 올바른 judgment 하나에 대한 NLL이다(§3.4, §4.1) |
| [[direct-judgement-preference]] | (4) 그래디언트로서의 부정 | 거부 judgment는 DPO 항에 들어가고 chosen judgment의 SFT 항이 이를 anchor한다(§3.4) |
| [[generative-reward-models]] | (4) 그래디언트로서의 부정 | STaR-DPO는 wrong-verdict chain을 거부 출력으로 사용한다(§4, Eq. 9) |
| [[meta-rewarding-lm]] | (4) 그래디언트로서의 부정 | 최저 Elo judgment는 judge DPO의 거부 항이다(§2.2) |

**메커니즘.** Verdict token에 대한 softmax에서, 올바른 verdict `y`의 log-probability를 logit `z_j`에 관해 미분한 값은 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. DPO 스타일 항은 추가로 `log π(y_l | x)`를 낮춘다. `y_l`에서 제거한 질량은 모델이 현재 가장 높게 rank한 곳으로 가며, judge에서는 이것이 같은 verdict의 더 잘 추론된 버전이 아니라 흔히 반대 verdict다. 이 때문에 verdict-only pair 유형이 존재한다. [[direct-judgement-preference]] §3.2는 긴 CoT critique에서 verdict를 결정하는 token은 몇 개뿐이라 critique 길이 목표가 정확히 그 token에서 신호를 희석한다고 말하며, critique를 제거한 `D_Std` pair를 추가한다.

**수치가 있는 증거.**

- 버리기 대 그래디언트: [[generative-reward-models]] §5.1은 wrong-verdict chain을 버리는 STaR-SFT가 UltraFeedback에서 67.4%이고 base model 대비 이득이 없다고 보고한다. 이들을 rejected term으로 쓰는 STaR-DPO는 in-distribution에서 73.9%, RewardBench에서 81.9%다. 논문은 부정 사용과 loss 변경을 분리하는 ablation을 **수행하지 않았으므로**, 두 원인 사이의 기여 분할은 측정되지 않았다.
- 부정 난이도: [[direct-judgement-preference]] App. E.8 Table 11은 8B 규모에서 70B teacher가 생성한 negative와 8B teacher negative를 비교한다. Pairwise accuracy 78.83 대 77.56, pairwise consistency 85.94 대 80.70, Pearson 0.68 대 0.67, classification 85.48 대 84.54이다.
- 거부 응답의 구성: [[self-taught-evaluators]] §6.2는 modified-instruction negative가 83.8, “답을 더 나쁘게 바꾸어 쓰라” prompt가 80.7이라고 보고한다.

**여기서 부정을 안전하게 만드는 제어.** 양의 likelihood 항으로 anchor한다(SFR-Judge loss의 SFT 항). 부정을 결정을 지니는 token으로 제한한다(verdict-only pair). Loss가 “더 짧은 것이 오답”을 encode하지 않도록 길이로 negative를 filter한다([[meta-rewarding-lm]] App. A.3의 judgment-length filter: iteration 1에서는 1100 character 초과 chosen judgment를, iteration 2에서는 1000 character 초과 chosen judgment를 버린다). A-better와 B-better에 걸쳐 label 균형을 유지한다([[self-taught-evaluators]] §3.4).

**진단.** Accuracy와 별도로 position-consistency를 기록한다. [[self-taught-evaluators]] Table 3은 HelpSteer2 validation accuracy가 65.5 → 71.0으로 오르는 동안 position-consistent accuracy는 56.5 → 60.6으로만 오른다고 보인다. Scale 상단으로의 압축([[meta-rewarding-lm]] Fig. 5)은 평균 agreement를 바꾸지 않고 separability를 제거하므로 judge 점수 분포를 기록한다. Judge version마다 고정 human set과의 agreement를 기록한다.

**일반성에 미치는 효과.** 한 prompt 분포에서 judge를 훈련하면 category profile이 이동한다. [[self-taught-evaluators]] §5.1은 Chat Hard 58.9 → 84.2, Safety 69.2 → 91.5인 반면 Chat은 97.6 → 96.6이라고 보고하며, 저자들은 더 어려운 reasoning 훈련 데이터에 기인한다고 해석한다(Interpretation). [[direct-judgement-preference]] §5.5는 specialization trade-off를 명시적으로 보인다. `β = 0.01`에서 8B judge를 continual fine-tuning하면, 저자들이 일곱 general pairwise benchmark에서 최소 하락이라고 설명한 상태로 ContextualJudgeBench 55.6%에 도달한다. 더 작은 `β`가 specialization을 만든다(Fig. 6).

---

## 레시피

아래 값은 주어진 위치의 source card에서 인용했다. `not reported` 행은 확인한 대상을 열거한다.

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Self-Taught Evaluator (Llama-3-70B-Instruct seed) | 70B | reward-model | prompt | 20,582 WildChat “reasoning” instruction | arXiv:2408.02666v2 §4.1 | 2026-09-14 검증됨 | Table 4: reasoning 83.5 대 coding 79.4, GSM8K 79.3, hh_rlhf 79.6 |
| 동일 | 70B | reward-model | 예시당 샘플 judgment; 유지 | N = 15; 올바른 것 하나, label-balanced | v2 §4.1 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| 동일 | 70B | reward-model | judgment sampling | temperature 0.7, top-p 0.9 | v2 App. A.2 Table 8 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| 동일 | 70B | reward-model | iteration; 초기화 | 5; 매 iteration seed에서 재초기화 | v2 §3.5, §4.1, Table 1 | 2026-09-14 검증됨 | Table 1: iteration 1–5에서 83.9 → 88.3 |
| 동일 | 70B | reward-model | loss; lr; warmup; epoch | judgment token만 NLL; final_lr_ratio 0.2에서 1.0e-06; 100 step; 최대 2 | v2 §4.1, Table 7 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| 동일 | 70B | eval-gate | RewardBench decoding; majority vote | temperature 0.0, top-p 1.0; 0.7 / 0.9에서 32 sample | v2 Table 8, Table 1 | 2026-09-14 검증됨 | Table 1: voting으로 88.3 → 88.7 |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | initialization(및 frozen reference) | Llama-3.1-8B-Instruct / NeMo-Instruct-12B / Llama-3.1-70B-Instruct | arXiv:2409.14664v3 §4.1, §3.4 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | preference pair; 유형 혼합 | 680K; D_CoT 70% : D_Std 15% : D_Ded 15% | v3 §4.1 | 2026-09-14 검증됨 | Fig. 4 (8B): D_Std 또는 D_Ded 제거 시 결과 변화, 비율 자체는 ablation하지 않음 |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | D_CoT sampling | Llama-3.1-70B-Instruct; prompt당 20 sample; temperature 0.7 | v3 §4.1 | 2026-09-14 검증됨 | App. E.8 Table 11: 70B negative가 8B negative를 이김 |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | DPO β; LR; epoch; batch; max length; compute | 보고되지 않음 | arXiv v1과 v3 본문 및 appendix 확인 | 보고되지 않음 | — |
| CoT-GenRM STaR-DPO | 8B | reward-model | initialization; 데이터; β; peak LR; iteration | Llama-3.1-8B-Instruct; UltraFeedback 61k pair; β = 1.0; 1.0e-6; 3, iteration당 새 데이터 1/3 | arXiv:2410.12832v1 §5, App. A.2.1 Table 2, App. A.3 | 2026-09-14 검증됨 | App. B Table 3: Maj@32 71.28 → 72.98 → 73.58 |
| CoT-GenRM STaR-DPO | 8B | eval-gate | sampling; majority vote | temperature 1.0, top-p 0.95; 32 sample | v1 App. A.2.2, Fig. 2 caption | 2026-09-14 검증됨 | Fig. 5 majority-vote curve |
| Meta-Rewarding LLM | 8B | preference | 응답당 judgment; scoring | N = 11 sample, temperature 0.8, top-p 0.95; 유효한 5-point 점수의 평균 | arXiv:2407.19594v2 §2.1, §3.1 fn. 2 | 2026-09-14 검증됨 | fn. 2: 큰 N은 사람 judgment와 유사하거나 더 나쁜 상관(표 없음)을 보임 |
| Meta-Rewarding LLM iteration 4 | 8B | preference | pair-selection length control ρ | 0.4 | v2 §3.1, App. A.3, Table 4 | 2026-09-14 검증됨 | Table 4: ρ = 0.4는 39.44% LC, 길이 2003 character; ρ = 0은 길이 2212 |
| Meta-Rewarding LLM, 모든 iteration | 8B | preference | loss; β; epoch; LR; batch; schedule | DPO; 0.1; iteration별 checkpoint 선택을 포함한 10; 5×10⁻⁶; 32; cosine | v2 App. A.3 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| Arena-Hard-Auto (v0.1) | — | eval-gate | prompt; judge; baseline; aggregation; 비용 | 500; gpt-4-1106-preview; gpt-4-0314; 95% CI를 위한 100 bootstrap round의 Bradley–Terry; 모델당 $20 | arXiv:2406.11939v2 §5, §6.1, Table 1 | 2026-09-15 검증됨 | Table 1: separability 87.4% 대 MT-Bench 22.6% |
| Style control 적용 Arena-Hard-Auto | — | eval-gate | style feature | 답변 token 길이, markdown header 수, markdown bold 수, markdown list 수를 BT fit에 추가 | v2 §6.5, App. A.2 | 2026-09-15 검증됨 | Table 3: style control 없을 때 94.4% 대비 confidence agreement 98.6% |
| AlpacaEval 2.0 LC | — | eval-gate | instruction; baseline; control | 805; gpt4_1106_preview; 길이 항을 0으로 둔 Eq. 1 GLM, `3M + N` parameter | arXiv:2404.04475v2 §3, §4 | 2026-09-15 검증됨 | Table 1: Arena 상관 0.98 대 0.94, gameability 10% 대 26% |
| Chatbot Arena | — | eval-gate | aggregation; 규모 | active sampling 및 multiplicity-corrected interval의 Bradley–Terry; 약 90K 사용자에서 240K표 | arXiv:2403.04132v1 §1, §4, §7.1 | 2026-09-15 검증됨 | §6.3: crowd–expert agreement 72.8–83.1%, expert–expert 79.4% 및 89.8% |
| RM-Bench | — | eval-gate | 보고할 metric | 3×3 style matrix의 상삼각형인 Hard accuracy | arXiv:2410.16184v1 §3.3, §5 | 2026-09-15 검증됨 | §5.2: downstream policy 성능과 r = 0.55 (p = 0.07), RewardBench는 r = 0.21 (p = 0.51) |
| PPE | — | eval-gate | 보고할 metric; aggregation | human preference set의 pair별 accuracy; 범주 전반 low-quantile aggregation | arXiv:2410.14872v2 §7 | 2026-09-15 검증됨 | Fig. 5: low quantile에서 post-DPO Arena 점수와 accuracy의 Pearson 0.80이 최고 |

**소규모 범용 judging 설정의 시작점.** 여기의 모든 수치는 원래 조건을 포함한 위 `verified` 행에서 왔다. 개방형 prompt에서 8B급 모델 checkpoint를 비교하려면, 80개 대신 자신의 traffic 분포에서 뽑은 500개 prompt를 사용한다. 20-모델 집합에서 500개의 curated prompt는 160개 MT-Bench turn의 22.6%에 비해 87.4% separability를 주었기 때문이다([[arena-hard-benchbuilder]] Table 1). Baseline checkpoint 하나를 고정하고 이에 대해 pairwise 채점하며, 두 순서를 실행하고 position consistency를 보고한다. 95% interval을 위한 100 bootstrap round의 Bradley–Terry fit으로 집계하고 [[lmsys-style-control]]의 네 style feature를 fit에 추가한다. 같은 500개 prompt가 두 judge에서 90.9%와 66.7% confidence agreement를 만들었으므로([[arena-hard-benchbuilder]] Table 4), 다른 family의 두 judge를 사용하고 둘 다 보고한다. 자체 judge를 훈련한다면 출판된 구성 중 소규모 실행에 가장 가까운 것은 8B SFR-Judge 레시피다(70/15/15에서 680K pair, prompt당 teacher sampling 20회, temperature 0.7). 이 실행의 DPO β, learning rate, batch size는 보고되지 않았으므로 로컬에서 조정해야 한다.

---
## 일반화 관점

**(a) 폭을 넓히는 것.**

- 난이도와 주제 포괄성을 위해 prompt 집합을 선별하면 모델을 구분하는 benchmark 능력이 높아진다. BenchBuilder의 curated prompt 500개는 separability 87.4%, confidence agreement 90.9%였으며, 무작위로 뽑은 WildChat prompt 250개의 confidence agreement 36.4%, separability 75.6%와 대비된다([[arena-hard-benchbuilder]] Tables 1–2).
- 둘 이상의 model family로 judge하면 human ranking과의 agreement가 높아진다. GPT-4-Turbo + Gemini-1.5-Pro ensemble은 단일 기본 judge의 90.9% / 87.4%에 비해 confidence agreement 91.5%, separability 89.5%에 이르며 측정된 self-bias를 줄였다([[arena-hard-benchbuilder]] Table 4, §6.6).
- 더 어려운 prompt로 judge를 훈련하면 category coverage가 넓어진다. 5 iteration 동안 Chat Hard 58.9 → 84.2, Safety 69.2 → 91.5였다([[self-taught-evaluators]] Table 1).
- 판정 전에 추론하도록 judge를 훈련하는 것은 Bradley–Terry head보다 distribution 밖에서 더 잘 일반화한다. RewardBench Reasoning에서 Bradley–Terry는 무작위보다 낮은 반면 STaR-DPO는 87.2%에 이른다([[generative-reward-models]] §5.2).

**(b) 좁아짐을 일으키는 것.**

- Judge 채점 leaderboard에 대한 policy 최적화. 상수 응답은 AlpacaEval 2.0에서 86.5% LC, Arena-Hard-Auto에서 83.0에 이르렀다([[null-model-cheating-benchmarks]] Table 2). 이는 capability 0으로 judge-directed optimization이 만들 수 있는 것의 상한이다.
- 비공개로 시험한 여러 variant 중 최선을 선택하는 것. [[leaderboard-illusion]]은 한 출시 전 private variant 27개와 추가 arena-distribution 데이터에서 Arena-Hard 상대 이득 최대 112%를 보고한다.
- 훈련 시 labeler와 평가 시 judge 사이에 model family를 공유하는 것. Self-preference 효과에는 통제된 측정이 있으며([[self-preference-recognition]] Fig. 1), judge 선택의 ranking 효과는 confidence agreement 24 point다([[arena-hard-benchbuilder]] Table 4).
- Judge를 자체 루프 안에서 재훈련하는 것. Judge–human Spearman은 iteration 2에서 0.382로 정점에 이른 뒤 iteration 4에는 0.326으로 떨어졌으며, positional bias는 43.92%에서 68.11%로 올랐다([[meta-rewarding-lm]] Tables 5, 7).
- 작은 DPO β로 judge를 특화하는 것. `β = 0.01`은 [[direct-judgement-preference]] Fig. 6에서 가장 특화된 judge를 만들었다.

**(c) 이 단계에서 측정하는 방법.**

1. 승률 옆에 보고하는 benchmark별 judge별 **position consistency**([[judge-llm-bias]] Table 2, [[self-taught-evaluators]] Table 9는 judge 훈련 뒤에도 이것이 남음을 보인다).
2. **Style-controlled와 raw 점수를 함께** 보고한다. 이 둘의 큰 차이는 style 성분의 크기다([[length-controlled-alpacaeval]] Table 1, [[lmsys-style-control]] coefficient).
3. 한 원인에 재귀속하지 않고 범위로 보고하는 **cross-family judge 불일치**([[arena-hard-benchbuilder]] Table 4).
4. Rejected 응답이 더 유리한 style을 갖는 [[rm-bench]]의 hard accuracy인 **style-stressed reward-model accuracy**.
5. **Downstream 검사**. 신선한 human-labeled set의 pair별 accuracy는 rank correlation보다 post-training Arena 점수를 더 잘 예측하며([[ppe-reward-model-eval]] §7), judge benchmark와 downstream 결과 사이의 음의 상관도 가능하다(§1, Fig. 4).
6. 가능한 곳에서는 **verifiable-correctness slice**를 사용하여 평가 일부가 judge에 전혀 의존하지 않게 한다.

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| Swap protocol 없이 승률 보고 | 같은 평가를 두 번 실행하면 가까운 쌍에서 불일치하며 승자가 어느 모델을 먼저 넣었는지와 상관 | 200-pair sample에서 두 순서를 실행하고 consistency를 계산한다. [[judge-llm-bias]] Table 2의 23.8–65.0% 범위와 비교한다 |
| “사람과 85% agreement”를 자신의 설정에서 judge 정확도로 취급 | Judge가 결정적이라고 한 checkpoint 비교가 사람에게 재현되지 않음 | 수치의 설정을 확인한다. S2는 tie를 제외하고 random baseline이 50%이며, S1은 33% baseline에서 66%다([[judge-llm-bias]] Table 5) |
| 80-question MT-Bench로 checkpoint 비교 | Checkpoint 간 점수 차이가 bootstrap interval 안에 있음 | Separability를 계산한다. MT-Bench는 20-모델 집합에서 쌍의 22.6%를 분리했다([[arena-hard-benchbuilder]] Table 1) |
| Style control이 metric을 안전하게 만든다고 가정 | 짧은 상수 문자열을 제출해도 상위에 가까운 점수 | Metric을 신뢰하기 전에 자신의 annotator template에 [[null-model-cheating-benchmarks]] structured response를 실행한다 |
| 훈련 시 labeler와 평가 시 judge에 같은 model family 사용 | Shared-family judge에서는 이득이 크고 다른 judge에서는 작음 | 같은 checkpoint를 두 번째 judge family로 점수화하고 범위를 보고한다([[arena-hard-benchbuilder]] Table 4) |
| 평가한 여러 variant 중 최선만 선택하여 보고 | 보고된 점수가 신선한 prompt set에서 재현되지 않음 | 몇 variant를 점수화했는지 기록한다. maximum이 아닌 분포를 보고한다([[leaderboard-illusion]] §3) |
| Aggregate leaderboard 점수로 reward model 순위화 | 선택한 reward model이 최선의 policy를 만들지 못함 | Pair별 accuracy와 low-quantile aggregation을 사용한다. [[ppe-reward-model-eval]] §1의 음의 RewardBench 상관을 점검한다 |
| Style 제약이 없는 rubric으로 채점한 후 길이 증가를 품질 증가로 해석 | Iteration에 따라 응답 길이는 늘지만 사람 agreement는 평평함 | 점수와 함께 iteration별 평균 응답 길이를 추적한다([[meta-rewarding-lm]] Table 4: ρ = 0 대 ρ = 0.4에서 2212 대 2003 character) |
| Bare judge로 수학 또는 추론 쌍 채점 | Judge가 확신에 찬 오답을 확인 | 후보를 읽기 전에 judge가 생성한 reference를 추가한다. 실패는 14/20에서 3/20으로 떨어졌다([[judge-llm-bias]] Table 4) |
| 외부 anchor 없이 루프에서 judge 재훈련 | Judge 점수 분포가 scale 상단으로 압축되는데 benchmark 점수는 계속 상승 | Judge version마다 고정 human-labeled set과의 agreement를 다시 계산한다([[meta-rewarding-lm]] Table 7) |

---

## 이해도 점검

1. MT-Bench probe에서 GPT-4의 position consistency는 65.0%, GPT-3.5의 것은 46.2%다. 보수적 swap 규칙이 승률 confidence interval을 judge에 의존하게 만드는 이유를 설명하고, 300개 중 각 judge가 남기는 decided pair 수를 계산하라.
2. AlpacaEval GLM에는 `tanh`를 통과하는 antisymmetric 길이 항이 있다. 이 길이 항이 대신 raw token 차이에 선형이었다면 무엇이 깨질지, 그리고 baseline과 모델을 비교하는 metric에서 항등식 `q(y = 1 | z_b, z_b, b, b, x) = 0.5`가 중요한 이유를 설명하라.
3. Length control은 gameability를 26%에서 10%로 낮췄지만([[length-controlled-alpacaeval]] Table 1), 상수 문자열은 86.5% LC 승률에 이르렀다([[null-model-cheating-benchmarks]] Table 2). 두 결과가 양립하는 이유를 설명하고 GLM이 제거하는 조작의 클래스를 정확히 말하라.
4. [[ppe-reward-model-eval]]은 post-DPO Arena 점수의 예측변수로서 Spearman과 Kendall 상관이 거의 0인 반면 pair별 accuracy가 최선이라고 발견했다. 저자들이 제안한 인과적 이유를 말하고, 이것이 reward-model 평가 보고 방식에 무엇을 시사하는지 설명하라.
5. [[rm-bench]]에서 같은 reward model은 §5.2에 인용한 chat matrix에서 easy accuracy 93.26%, hard accuracy 18.58%를 기록한다. 이 reward model에 맞춰 훈련한 policy가 무엇을 학습할지와, 그 행동을 예측하는 두 수치 중 어느 것인지 설명하라.
6. [[self-taught-evaluators]]는 유지한 올바른 judgment에 NLL로 judge를 훈련하고, [[direct-judgement-preference]]는 잘못된 judgment에 DPO 항과 SFT 항을 더해 judge를 훈련한다. 확률 질량이 어디로 이동하는지 관점에서, 두 번째 방법이 verdict-only pair도 추가하는 이유를 설명하라.
7. [[meta-rewarding-lm]]은 AlpacaEval 2 LC를 22.92%에서 39.44%로 올리지만 judge–human Spearman은 iteration 2에서 정점에 이르고 iteration 4에 하락한다. 둘 다 참일 수 있는 이유와 benchmark 이득이 capability를 반영하는지 판단하기 위해 추가할 측정을 설명하라.
8. 한 팀이 같은 checkpoint에서 GPT-4 judge benchmark +12 point, Claude judge +4 point를 보고한다. “8 point는 judge leakage”가 유효한 추론이 아닌 이유와, 8-point gap이 말할 수 있게 하는 것을 설명하라.

---

## 연결

- **이전 장** — [[ch-48]] — Contamination Detection and Its Effect on Reported Scores. Contamination은 시험 항목을 훈련에서 보았기 때문에 점수를 부풀리고, judge-specific overfitting은 응답이 grader에 맞았기 때문에 점수를 부풀린다. 둘 다 전이되지 않는 수치를 만들며 별도 탐지가 필요하다.
- **다음 장** — [[ch-50]] — Slice Analysis, Forgetting Slices, and Failure Bucketing. Slice 전반에 균일한 judge bias는 모든 점수를 동일하게 이동시키지만, slice를 추적하는 편향(코드 답은 더 길고 safety 답은 refusal)은 slice ranking을 바꾸며 그곳에서 탐지된다.
- [[ch-47]] — Evaluation Harness and Suite Design for General Capability. §1의 judge 구성은 harness가 기록해야 하는 benchmark별 specification이다.
- [[ch-47a]] — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation. 이 장의 §4는 해당 audit의 judge-specific 사례다.
- [[ch-41]] — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization. Arena 점수와 reward model에 쓰는 Bradley–Terry fit은 같은 모델이며, 여기의 §5는 이를 훈련 신호가 아니라 instrument로 평가한다.
- [[ch-42]] — Reward Hacking and Judge Design. Judge의 훈련 시 hacking을 다루며, 이 장은 평가 시 사례다.
- [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. 부정 피드백 섹션이 judge 훈련에 적용하는 유도식을 담는다.
- [[ch-44]] — Process Supervision and Verifiable Rewards. Verifiable prompt에서는 checker가 judge를 대체하여 이 장의 bias inventory를 측정 경로에서 제거한다.
- [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions. Judge sampling variance와 swap으로 결정된 pair가 줄어드는 현상은 그 장의 noise budget 입력이다.
- [[ch-51a]] — Evaluating Agent Generality and Reliability. §4.4가 여기서 제외한 trajectory와 environment state 채점이다.

---

## 출처

- [[judge-llm-bias]] — Zheng et al. 2023 (arXiv:2306.05685). Position, verbosity, self-enhancement, math-grading 측정, swap·few-shot·CoT·reference-guided 보정, S1/S2 구분을 가진 MT-Bench agreement 표를 다룬다.
- [[chatbot-arena]] — Chiang et al. 2024 (arXiv:2403.04132). 사람 투표 platform, active sampling을 포함한 Bradley–Terry, crowd–expert agreement rate, expert 라벨링 비용을 다룬다.
- [[arena-hard-benchbuilder]] — Li et al. 2024 (arXiv:2406.11939). Separability, confidence agreement, Brier metric, 500-prompt pipeline, judge-choice 비교, automatic benchmark에 적용한 style control을 다룬다.
- [[length-controlled-alpacaeval]] — Dubois et al. 2024 (arXiv:2404.04475). Length-control GLM, 그 항등식 및 대칭 성질, gameability 측정, Chatbot Arena와의 상관을 다룬다.
- [[lmsys-style-control]] — LMSYS 2024 blog. Arena Bradley–Terry fit에 추가한 style feature, 적합 coefficient, ranking 변화를 다룬다.
- [[null-model-cheating-benchmarks]] — Zheng et al. 2024 (arXiv:2410.07137). 세 judge-scored benchmark에서의 null-model 승률과 전이 가능한 random-search prefix를 다룬다.
- [[leaderboard-illusion]] — Singh et al. 2025 (arXiv:2504.20879). Private-variant testing, data-access 비대칭, silent deprecation, arena-distribution 데이터의 측정된 이득을 다룬다.
- [[self-preference-recognition]] — Panickssery et al. 2024 (arXiv:2404.13076). Self-recognition accuracy, self-preference와의 선형 관계, confounder control, order-reversal rate를 다룬다.
- [[rewardbench]] — Lambert et al. 2024 (arXiv:2403.13787). Section 구조, scoring 및 weighting rule, Reasoning 안의 accuracy 범위를 다룬다.
- [[rm-bench]] — Liu et al. 2024 (arXiv:2410.16184). Style–substance matrix, easy/normal/hard accuracy, 46.6% hard-accuracy 결과, policy-correlation 비교를 다룬다.
- [[ppe-reward-model-eval]] — Frick et al. 2024 (arXiv:2410.14872). 실제 post-training Arena 배포를 포함한 end-to-end reward-model 평가와 어떤 offline metric이 downstream preference score를 예측하는지 다룬다.
- [[self-taught-evaluators]] — Wang et al. 2024 (arXiv:2408.02666). Modified-instruction pair 구성, rejection-sampled judgment, RewardBench 5 iteration, 지속되는 순서 효과를 다룬다.
- [[direct-judgement-preference]] — Wang, Xu et al. 2024 (arXiv:2409.14664). DPO+SFT judge loss, 세 pair 유형, hard-negative ablation, better-of-two-orders 보고 관례를 다룬다.
- [[generative-reward-models]] — Mahan et al. 2024 (arXiv:2410.12832). GenRM 및 CoT-GenRM 훈련 변형, in-distribution 대 RewardBench 일반화, majority-vote 효과를 다룬다.
- [[meta-rewarding-lm]] — Wu et al. 2024 (arXiv:2407.19594). Judge와 meta-judge 훈련, pair selection에서의 length control, drift 진단(meta-judge bias, positional bias, score compression, iteration별 human Spearman)을 다룬다.
