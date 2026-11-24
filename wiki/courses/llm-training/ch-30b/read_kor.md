<!-- chapter: ch-30b
     track: sft
     kind: content
     title: 다중 능력 SFT 혼합: 간섭, 전이, 에이전트형 및 긴 컨텍스트 비중
     deps: [ch-30a, ch-18]
     sources: [[sft-data-composition-dmt]], [[tulu-1-how-far-can-camels-go]], [[tulu-3]], [[tulu-3-sft-mix]], [[llama-3]], [[llama-3-recipe]], [[long-context-llama3]], [[glm-4-5]], [[qwen-3]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[agenttuning]], [[agent-flan]], [[agent-data-protocol]], [[limi]], [[llama-nemotron]], [[prolong]], [[prolong-recipe]], [[qdit-data-diversity]], [[deita]], [[instag]], [[mix-data-or-merge-models]], [[am-reasoning-answers-for-non-reasoning]], [[nemotron-ultra]]
     figures: figures/mixture-share-calculator.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30b — 다중 능력 SFT 혼합: 간섭, 전이, 에이전트형 및 긴 컨텍스트 비중

> **핵심 통찰.** 모든 능력에 가장 좋은 SFT 데이터셋 하나나 혼합 비율 하나는 없다. 통제된 LLaMA 7B–33B 실험에서 수학·코드·채팅 데이터를 섞으면 각 소스가 작을 때는 각 능력이 올라가지만(LLaMA-7B 코드의 경우는 저자들이 명시한 예외), 각 소스를 전량 사용하면 각 능력이 내려간다([[sft-data-composition-dmt]]). Tülu 3 8B 혼합에서 한 하위 집합을 제거하면 그 집합이 직접 겨냥하지 않는 능력의 점수도 바뀐다([[tulu-3]]). 에이전트형 및 긴 컨텍스트 비율을 수치로 보고한 자료는 AgentTuning의 에이전트 샘플링 비율 η = 0.2와 Llama 3.1의 긴 컨텍스트 데이터 예제 비율 0.11%를 사용한다. 긴 컨텍스트의 근거는 서로 충돌한다. Llama 3는 짧은 예제만으로 SFT하면 긴 컨텍스트 능력이 후퇴했다고 보고한 반면, ProLong에서는 시험한 모든 합성 긴 SFT 토큰 비율(1%, 3%, 10%, 50%)이 긴 컨텍스트 평균을 낮췄다.
>
> **지침.** 전문 능력에 큰 데이터셋이 있다면(DMT의 GSM8K RFT 예제 110,142개) 일반 단계 전에 먼저 학습하고 최종 단계에서 일부를 재생(replay)하라(DMT에서는 1/256). LLaMA-7B에서 k = 1/256인 DMT 일정은 replay 없는 경우의 GSM8K 32.60, MT-Bench 6.02에 비해 GSM8K 41.92, MT-Bench 6.08을 기록했기 때문이다. 단일 단계 혼합을 수정할 때는 모든 능력과 능력별 최소 하나의 미관측 평가를 다시 평가하라. Tülu 3 8B에서 Persona 데이터를 제거하자 개발 IFEval은 19.2점 낮아진 반면 IFEval-OOD는 0.4점 올랐다. 에이전트 trajectory를 넣을 때는 일반 instruction 데이터를 혼합 안에 유지하고 held-out 에이전트 과제로 에이전트 비중을 고르라. AgentLM-7B의 에이전트 전용 학습은 held-out 에이전트 과제에서 0.09였지만 η = 0.2에서는 0.67이었다. 혼합 비율을 보고하거나 다른 레시피에 복사할 때는 단위를 밝혀야 한다. Llama 3.1 SFT에서 예제의 0.11%는 토큰의 4.95%이기 때문이다.

## 범용 모델에서 이 장이 중요한 이유

지도 미세조정(supervised fine-tuning, SFT)은 사전학습 또는 중간학습된 모델을 대상으로, 목표 응답에 대한 cross-entropy를 사용해 학습한다. 범용 모델에는 채팅, 지시 따르기, 수학, 코드, 도구 사용, 다국어 텍스트, 안전성, 긴 입력, 그리고 때로는 두 응답 모드까지 포괄하는 하나의 SFT 단계가 필요하다. SFT 혼합이란 데이터셋들의 집합과 각 데이터셋을 얼마나 자주 샘플링할지를 정하는 규칙이다.

여기서 세 문제가 생기며, 각각 측정할 수 있다.

1. **간섭(interference).** 한 능력을 위한 데이터를 추가했더니 다른 능력 점수가 내려갈 수 있다. 추가 없는 상태에서 다른 능력을 학습한 run과의 차이로 이를 측정한다.
2. **단위(units).** “예제의 %”로 인용한 비율은 토큰으로 계산한 같은 비율과 최대 45배까지 다를 수 있다(이 장 §1). 따라서 한 보고서의 비율을 다른 단위로 세는 레시피에 그대로 복사하면 실제 혼합이 달라진다.
3. **협소화(narrowing).** 개발 벤치마크에 맞춰 조정한 혼합은 그 벤치마크는 올리면서도 같은 능력의 미관측 평가는 올리지 못할 수 있다(이 장 §3).

사전학습 → 중간학습 → SFT → 선호 최적화 → RL → 평가라는 파이프라인에서 SFT 혼합은 이후 단계가 출발하는 능력 프로필을 정한다. 이후 단계도 그 프로필을 바꿀 수 있다. Llama 3는 짧은 컨텍스트 direct preference optimization(DPO, ch-39)이 긴 컨텍스트 SFT의 효과를 되돌리지 않았다고 보고하며, DPO 레시피의 optimizer step 수가 SFT보다 적기 때문일 수 있다고 추정한다(이 장 §6). Qwen3는 mode-fusion SFT 및 일반 RL 뒤에 thinking mode 성능이 내려갔다고 보고한다(이 장 §4). ch-30은 masking, packing, template, epoch 같은 예제별 설계 선택을 다룬다. ch-30a는 forgetting 측정법을 다룬다. 이 장은 무엇을 혼합에 넣을지와 각 항목을 어떤 비율로 넣을지를 다룬다. ch-30c는 별도 모델을 학습한 뒤 가중치를 합치는 대안을 다룬다.

## §1 혼합의 단위: 예제 비율, 토큰 비율, 손실 가중치

**정의.** 혼합 비율(mixture share)은 한 도메인 d에 배정한 학습 신호의 비율이다. 예제 수, 전체 토큰 수, 또는 손실 토큰 수(masking 뒤 cross-entropy를 받는 응답 토큰)로 셀 수 있다.

**문제.** 보고서는 비율을 서로 다른 단위로 출력한다. 각 도메인의 gradient 가중치를 실제로 정하는 단위는 손실을 어떤 방식으로 평균내는지에 달려 있다. 한 도메인의 토큰 비율은 그 도메인의 평균 예제 길이를 혼합 전체의 평균 길이로 나눈 비율만큼 예제 비율보다 커진다. Llama 3.1의 긴 컨텍스트 데이터에서는 그 배율이 45다.

**메커니즘.**
1. 각 데이터셋 d에는 원본 예제 수 n_d와 샘플링 multiplier w_d가 있으므로, epoch마다 m_d = n_d · w_d개의 예제를 뽑는다.
2. 도메인 d의 각 예제는 평균 L_d개의 토큰과 평균 ℓ_d개의 손실 토큰을 가진다.
3. sequence별 평균 손실(per-sequence mean loss)에서는 각 예제의 손실을 해당 예제 자신의 토큰 수로 평균낸 다음 예제 간에 평균내므로, 모든 예제가 같은 가중치를 받는다.
4. batch 전체의 token mean 또는 token sum에서는 각 손실 토큰이 같은 가중치를 받으므로, 응답이 긴 도메인이 더 큰 가중치를 받는다.

**수식.**

```
example share_d = m_d / Σ_j m_j
token share_d   = m_d · L_d / Σ_j m_j · L_j
loss share_d    = m_d · ℓ_d / Σ_j m_j · ℓ_j
```

m_d는 도메인 d에서 샘플링된 예제 수, L_d는 도메인 d의 예제당 평균 토큰 수, ℓ_d는 도메인 d의 예제당 평균 손실 토큰 수다. 합은 혼합 안의 모든 도메인 j에 대해 계산한다.

**계산 예 1(Llama 3.1, [[llama-3]] Table 7).** 표는 “예제의 %”, 예제당 평균 토큰 수, 최종 응답의 평균 토큰 수를 출력한다.

| 도메인 | 예제의 % | 평균 토큰 | 최종 응답 평균 토큰 |
|---|---:|---:|---:|
| 일반 영어 | 52.66 | 974.0 | 317.1 |
| 코드 | 14.89 | 753.3 | 374.5 |
| 다국어 | 3.01 | 520.5 | 289.7 |
| 시험형 | 8.14 | 297.8 | 173.4 |
| 추론 및 도구 | 21.19 | 661.6 | 301.9 |
| 긴 컨텍스트 | 0.11 | 38,135.6 | 740.5 |

- 평균 예제 하나에 해당하는 긴 컨텍스트 토큰: 0.0011 × 38,135.6 = 41.95.
- 여섯 행 전체: 0.5266 × 974.0 + 0.1489 × 753.3 + 0.0301 × 520.5 + 0.0814 × 297.8 + 0.2119 × 661.6 + 41.95 = 847.1토큰이다. 인쇄된 총합은 846.1인데, 백분율이 반올림되었기 때문이다.
- 긴 컨텍스트 토큰 비율: 41.95 / 847.1 = 4.95%(도출값). 이는 예제 비율의 45배다.
- 긴 컨텍스트 최종 응답 비율: 0.0011 × 740.5 = 310.4토큰 중 0.81토큰 = 0.26%(도출값).

이 셋 중 어떤 수가 gradient 가중치를 정하는지는 loss mask와 aggregation에 달려 있으며, Table 7은 이를 밝히지 않는다.

**계산 예 2(DeepSeek-R1 Dev3 SFT, [[deepseek-r1-recipe]], arXiv:2501.12948v2 Table 5).** General은 804,745개 표본 중 177,812개로 표본의 22.10%다. 토큰으로는 177,812 × 1,419.8 = 252.5M이며, 전체 804,745 × 5,355.3 = 4,309.7M토큰 중 5.86%다(도출값). Math는 표본의 49.12%이지만 토큰의 55.90%다.

**aggregation이 중요하다는 근거.** ProLong은 SFT 손실을 “모든 유효 학습 토큰에 대한 평균”으로 바꿨다. sequence별 평균이 “최적화를 왜곡하고 도메인 비율에 대한 통제도 왜곡하기” 때문이라고 설명한다([[prolong]], App. A.3). Llama-Nemotron은 특히 sequence-length-dependent token loss averaging 때문에 긴 reasoning trace에서 효과적으로 학습하려면 더 높은 learning rate가 필요하다고 보고한다([[llama-nemotron]], §4.1). 이 두 보고서는 aggregation 선택 자체의 ablation을 출력하지는 않는다. Tülu 3는 훈련 설정 차이의 원인을 loss averaging으로 추적한 뒤 SFT를 sum loss로 바꿨고, 이는 “사실상 모든 토큰에 같은 가중치를 준다”고 설명한다. Fig. 5는 Tülu 2 혼합에서 Llama 3.0을 여러 learning rate로 mean loss와 sum loss로 비교하며, sum loss의 5e-6이 가장 좋다([[tulu-3]] §4.3.2; 그림만 있고 표는 없음). ch-30은 sum 대 mean aggregation을 자세히 다룬다.

[혼합 비율 계산기](figures/mixture-share-calculator.html): count, multiplier, 평균 크기를 입력하면( Llama 3.1 Table 7, DeepSeek-R1 Table 5, ADP agentic mixture preset 제공) 예제·토큰·손실 토큰 비율과 각 aggregation 규칙에서 어느 비율이 손실 가중치를 정하는지 볼 수 있다.

**조건과 한계.** Table 7은 길이 분포가 아니라 평균만 제공한다. 예제 길이의 변동이 큰 도메인에서는 평균이 토큰 비율을 설명하지만, batch마다의 토큰 비율은 달라진다. 작은 비율은 반올림된 백분율 때문에 약 두 자리 유효숫자 정도로만 정밀하게 계산할 수 있다.

**함의.** 보고서의 혼합 비율을 재현하려면 그 비율의 단위뿐 아니라 loss mask와 aggregation 규칙도 함께 알아야 한다.

## §2 데이터 양, 비율, 순서의 함수인 간섭과 전이

**정의.** 도메인 B에서 능력 A로의 음의 전이(간섭)는, A 데이터를 같은 양으로 사용했을 때 A만 학습한 경우보다 A+B를 학습한 뒤 A 점수가 낮아지는 현상이다. 양의 전이는 A 점수가 더 높아지는 경우다.

**문제.** 범용 모델용 혼합은 많은 도메인을 결합한다. 따라서 능력 A의 점수 변화는 B의 데이터 양, B 대 A의 비율, 또는 A와 B를 학습한 순서에서 올 수 있다.

**1차 설명 모델(해석; 인용한 어떤 출처도 gradient를 측정하지 않았다).** g_A와 g_B를 각각 능력 A와 B의 손실 gradient라고 하자. learning rate η로 B를 한 step 학습하면 A의 손실은 근사적으로 다음만큼 변한다.

```
ΔL_A ≈ −η · (g_A · g_B)
```

여기서 g_A · g_B는 두 gradient vector의 inner product다. inner product가 음수이면 L_A가 올라가므로 간섭이고, 양수이면 L_A가 내려가므로 전이다. 예를 들어 g_A = (1, 2), g_B = (−2, 0.5), η = 0.1이면 g_A · g_B = −2 + 1 = −1이고 ΔL_A ≈ +0.1이다. g_B = (1, 0)일 때는 g_A · g_B = 1이고 ΔL_A ≈ −0.1이다. 이 모델은 한 step을 설명할 뿐이며, 효과가 여러 epoch에 걸쳐 어떻게 누적되는지는 예측하지 못한다.

**근거: DMT([[sft-data-composition-dmt]], LLaMA 7B/13B/33B; GSM8K RFT 110,142개, Code Alpaca 20,022개, ShareGPT 86,060개 예제; 3 epoch, LR 2e-5).** GSM8K RFT는 rejection sampling으로 수집한 여러 reasoning path를 가진 7.5K GSM8K 학습 문제다(App. A). DMT(dual-stage mixed fine-tuning)는 논문이 제안한 schedule이며, 아래 4번에 설명한다.

1. **양(§3.2, Fig. 2).** 세 모델 크기 모두에서 GSM8K는 데이터 양과 함께 오른다. HumanEval은 7B와 13B에서 불규칙한 scaling curve를, 33B에서 대략 log-linear curve를 따른다. MT-Bench는 “약 1k data sample만으로 출현(emerges)”한 뒤 천천히 오른다.
2. **혼합 대 단일 출처(§3.3, Fig. 3).** 세 출처를 각각 같은 fraction으로 혼합하면 1/256에서는 각 점수가 단일 출처 run보다 올라가지만, 전체 양에서는 각 점수가 내려간다. 저자들은 1/256에서의 LLaMA-7B code를 예외(outlier)로 든다. 7B의 교차점은 1/64와 1/16 사이에 있다. 저자들은 저자원 조건에서 수학과 일반 능력의 이득이 모델 크기와 함께 커진다고 말한다(결과, 단일 연구).
3. **비율(Fig. 4).** 한 출처를 고정하고 다른 출처를 scaling하면 수학 점수는 비율에 따라 거의 변하지 않는다. HumanEval은 출렁이며, 저자들은 이를 ShareGPT 내부의 코드 데이터 때문이라고 본다(해석). InsTag(지시에 open-set intent tag를 붙이는 LLM 기반 tagger, [[instag]])로 식별한 ShareGPT 내 코드·수학 예제를 제거해 86K에서 63K로 줄이면 고자원 충돌이 “어느 정도(to some extent)” 줄어든다(§4.2, Fig. 6).
4. **순서(Table 1).** 순차 학습(code → math → general)은 먼저 학습한 능력을 망각한다. DMT는 먼저 code와 math를 학습한 뒤, ShareGPT와 code·math 데이터의 일부 fraction k를 함께 학습한다.

**계산 예(Table 1, GSM8K / HumanEval / MT-Bench).**

| LLaMA-7B run | GSM8K | HumanEval | MT-Bench |
|---|---:|---:|---:|
| 각 열의 능력에 대한 단일 출처(math only, code only, general only) | 49.10 | 18.40 | 5.88 |
| multi-task learning(하나의 혼합 stage) | 47.53 | 14.63 | 5.76 |
| sequential training | 31.39 | 15.85 | 5.72 |
| mixed sequential training(code+math, 그다음 general) | 32.60 | 15.24 | 6.02 |
| DMT, k = 1/256 | 41.92 | 17.68 | 6.08 |

- multi-task에서 single source를 뺀 값은 GSM8K −1.57, HumanEval −3.77, MT-Bench −0.12다. 33B에서 같은 차이는 각각 −1.22, −7.92, −0.56이다.
- sequential training에서 math only를 뺀 값은 31.39 − 49.10 = −17.71 GSM8K다. 이는 나중 stage에서 생긴 망각이다.
- DMT에서 mixed sequential을 뺀 값은 GSM8K +9.32, HumanEval +2.44, MT-Bench +0.06이다. k = 1/256일 때 두 번째 stage는 86,060개 ShareGPT 예제에 GSM8K RFT 430개와 Code Alpaca 78개(Table 2)를 더한다. 즉 stage-2 예제의 508 / 86,568 = 0.59%다(도출값).
- DMT는 7B에서 여전히 math-only run보다 GSM8K 7.18점 낮다. 33B에서 그 차이는 1.55점(57.91 대 56.36)이다.
- k를 1/256에서 1/4 쪽으로 올리면 general 점수와 specialist 점수 사이에 trade-off가 생기며, k를 1/4에서 1로 올리면 general 점수가 내려간다(§4.3, Fig. 5).

**조건과 한계.** LLaMA-1 모델, 능력당 하나의 benchmark, GPT-4로 채점한 MT-Bench라는 조건이다. 학습 seed와 반복 학습 run은 보고하지 않았다. 논문은 서로 다른 두 batch size를 인쇄한다(§3.1의 16, App. C의 128). MT-Bench 점수는 “fluctuate할 것”이므로(App. C) 세 번의 run을 평균냈지만, 변동량을 보고하지 않았다. 따라서 MT-Bench의 −0.12 차이를 run 간 변동과 비교할 수 없다.

**함의.** 혼합 효과는 비율뿐 아니라 데이터 양에 따라 달라진다. 따라서 한 데이터 크기에서 고른 비율이 다른 데이터 크기에서도 맞다는 근거가 되지는 않는다. 학습 순서는 망각을 만들며, DMT에서는 1/256의 replay fraction이 이를 줄였다. ch-30a는 망각 제어 방법으로서 replay를 다룬다.

## §3 단일 최고 데이터셋은 없다: Tülu 1 비교와 Tülu 3 제거 ablation

**근거 1: Tülu 1([[tulu-1-how-far-can-camels-go]], LLaMA 13B, 데이터셋 12개, 하나의 recipe, Table 3).**

- Human+GPT 혼합은 평균 점수 45.2로 가장 높지만, 6개 능력 평가 중 최고인 것은 2개뿐이다.
- ShareGPT는 AlpacaEval 승률 70.5%로 가장 높으며, TyDiQA F1은 base의 43.2에서 30.5로 낮춘다.
- 12개 데이터셋 중 6개는 GSM을, 8개는 TyDiQA를 base model보다 낮춘다.
- CoT 데이터는 GSM을 14.5에서 40.0으로 올리고, SuperNI는 BBH를 39.3에서 4.5로 낮춘다.
- 저자들은 GSM과 TyDiQA의 하락을 해당 데이터셋들에 CoT 및 다국어 데이터가 없기 때문이라고 설명한다(해석, §5.1).

**근거 2: Tülu 3 8B SFT 제거 ablation([[tulu-3]], arXiv:2411.15124v5 Table 10).** 최종 혼합은 939,344개 prompt다(Table 7). 각 행은 하위 집합 하나를 제거하고 다시 학습한 결과다.

| Run | Avg | BBH | DROP | GSM | MATH | IFEval | AE 2 | TQA | Safety |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tülu 3 8B SFT | 60.1 | 67.9 | 61.3 | 76.2 | 31.5 | 72.8 | 12.4 | 46.8 | 93.1 |
| WildChat 제외 | 58.9 | 65.6 | 59.3 | 75.8 | 31.8 | 70.1 | 7.5 | 45.2 | 95.2 |
| Safety 제외 | 58.0 | 68.3 | 59.4 | 76.9 | 32.6 | 71.0 | 12.4 | 45.5 | 74.7 |
| Persona 제외 | 58.6 | 68.3 | 62.2 | 76.8 | 30.1 | 53.6 | 13.5 | 48.9 | 93.9 |
| Math 제외 | 58.2 | 68.9 | 60.9 | 64.1 | 23.5 | 70.6 | 12.0 | 47.1 | 93.5 |

- 실제 사용자 대화인 WildChat을 제거하면 12개 benchmark 중 7개의 점수가 최소 1점 바뀐다. 여기에는 AlpacaEval 2의 −4.9와 BBH의 −2.3이 포함되며, Safety는 2.1점 오른다.
- Safety 하위 집합을 제거하면 Safety는 18.4점 낮아진다. 저자들은 safety를 다른 능력과 “대체로 직교적(generally orthogonal)”이라고 표현하지만, HumanEval(−1.7), DROP(−1.9), IFEval(−1.8)도 함께 낮아진다.
- 합성 수학·코드·instruction-following prompt인 Persona 데이터를 제거하면 IFEval은 19.2점 낮아지고 TruthfulQA는 2.1점 오른다.
- Math 데이터를 제거하면 GSM8K는 12.1점, MATH는 8.0점 낮아지고 BBH는 1.0점 오른다.

**seed noise의 크기.** Table 14는 Tülu 3 8B SFT를 seed 다섯 개로 학습한다. 평균은 59.8부터 60.1까지이며 spread는 0.3점이다. 평균에서의 제거 delta(−1.2부터 −2.1)는 이 spread의 4배에서 7배다. benchmark별 seed variance는 보고하지 않았으므로, Math 제외 시 BBH +1.0처럼 1점 부근의 단일 benchmark 변화는 공개된 데이터만으로 seed noise와 구분할 수 없다.

**development 평가와 unseen 평가([[tulu-3]] §7.4, Table 32).** IFEval-OOD는 IFEval의 25개에 포함되지 않는 검증 가능한 constraint 52개로 구성된 unseen instruction-following test다(§7.3.1). IFEval-OOD 점수는 전체 혼합에서 17.6, Persona 제외에서 18.0, WildChat 제외에서 20.8이다. Persona 하위 집합은 development IFEval을 19.2점 올렸지만 IFEval-OOD를 올리지는 않았다. 저자들은 “Precise Instruction Following에서, 그리고 어느 정도 Knowledge Recall과 Reasoning에서 우리의 선택이 development evaluations에 overfit된다”고 쓴다. 그럼에도 unseen 평균은 ablation의 27.4부터 29.7보다 최종 혼합에서 29.9로 더 높다.

**혼합을 만든 방식(§4.1.2).** 저자들은 “먼저 능력별 데이터 혼합과 모델을 만들고, 다른 평가들은 무시한 채 개별 능력에서 최고의 성능으로 이어진 혼합을 유지했다”고 설명한다. 그 뒤 이를 preview mix로 합쳐 데이터셋을 추가하거나 제거하고, decontamination을 수행하고, 큰 데이터셋을 downsample하는 방식으로 반복했다. [[tulu-3-sft-mix]]는 공개한 component를 나열하지만, 그 component 목록에는 Table 7의 OpenMathInstruct 2 50,000개 prompt가 없고 합계는 889,344다. 같은 dataset card는 prose에서 939,344개 sample이라고 하며 viewer는 939,343행을 보고한다. dataset card는 이 50,000개 차이의 이유를 말하지 않으므로, 누락 component의 정체는 출처만으로 확정할 수 없다.

**상태.** Replicated: Tülu 1, Tülu 3, DMT는 서로 다른 모델과 데이터에서, 모든 능력에 대해 단일 source 또는 단일 ratio가 최고가 아니라는 결론에 동의한다.

**함의.** 모든 하위 집합에는 능력들에 걸친 효과 vector가 있으며, 어떤 성분은 양수이고 어떤 성분은 음수다. development benchmark에서 내린 mixture 결정에는, development 점수만 올리는 하위 집합을 찾아내기 위해 능력별 paired unseen evaluation(ch-47)이 필요하다.

## §4 능력 전문가를 하나의 범용 모델로 결합하기

**정의.** 능력 전문가는 하나의 능력(코드, 다국어, 추론, 에이전트, 채팅)을 주로 학습한 모델이다. 전문가에서 범용 모델로 가는 설계는 데이터(주석, rejection sampling, distillation SFT)를 통해, 또는 가중치(merge, ch-30c)를 통해 전문가 능력을 한 모델로 옮긴다.

**문제.** 처음부터 단일 혼합으로 학습하면 모든 능력이 하나의 training run을 공유해야 한다. 따라서 한 능력의 데이터와 RL은 다른 능력에 영향을 주지 않고 따로 조정할 수 없다.

**공개된 세 가지 설계.**

| 보고서 | 전문가 | 전문가 능력이 범용 모델에 도달하는 방법 | 전문 능력에 대해 측정한 비용 |
|---|---|---|---|
| Llama 3 ([[llama-3]] §4.3.1, §4.3.2) | 코드 전문가: pre-training branch를 85% 초과가 코드인 1T token에서 계속 학습하고, 16K까지 long-context fine-tuning한 뒤 post-training했다. 다국어 전문가: 90%가 다국어 token인 데이터에서 계속 학습한 뒤 post-training했다 | 코드 전문가는 coding prompt에 대한 human annotation을 수집하고 응답을 rejection-sample하는 데 사용한다. 다국어 전문가는 pre-training이 끝날 때까지 non-English annotation을 수집하는 데 사용한다 | 보고하지 않음 |
| GLM-4.5 ([[glm-4-5]] §3, §3.1) | Reasoning, Agent, General-chat 전문가이며, 각각 cold-start SFT 뒤 RL을 수행한다 | “Overall SFT”: long-context task를 포함하는 수백만 개의 expert-generated sample을 최대 128K token 길이로 학습한다. 명시적 thinking이 있는 데이터와 없는 데이터는 “meticulously balanced”했다고 한다 | 보고하지 않음 |
| Qwen3 ([[qwen-3]] arXiv:2505.09388v1 §4.3, §4.7) | 한 계보: long-CoT cold start → reasoning RL(Stage 2) | Stage 3 “Thinking Mode Fusion”: Stage 2 모델을 대상으로 해당 모델이 rejection-sample한 thinking data와 curate한 non-thinking data로 continual SFT하고, Stage 4에서 general RL을 수행한다 | 아래 Table 22 |

**계산 예(Qwen3-32B, Table 22).** thinking mode에서 Stage 2 → Stage 3 → Stage 4의 결과는 다음과 같다.

| Benchmark | Stage 2 | Stage 3 | Stage 4 |
|---|---:|---:|---:|
| AIME'24 | 83.8 | 81.9 | 81.4 |
| LiveCodeBench v5 | 68.4 | 67.2 | 65.7 |
| MMLU-Redux | 91.4 | 91.0 | 90.9 |
| IFEval strict prompt | 73.0 | 78.4 | 85.0 |
| ToolUse(in-house) | 63.3 | 70.4 | 85.5 |
| ThinkFollow(in-house mode switching) | 해당 없음 | 88.7 | 98.9 |

- 두 stage를 거치며 AIME'24는 2.4점, LiveCodeBench는 2.7점 낮아진다. 반면 IFEval은 12.0점, ToolUse는 22.2점 오른다.
- AIME'24는 30문항이며 각 문항을 64회 sample한다(§4.6). 한 문항은 평균 정확도 3.33점이므로, 2.4점 하락은 한 문항 정확도의 0.72배에 해당한다(계산으로 도출). confidence interval은 보고하지 않았다.
- 보고서는 다음과 같이 쓴다. “We conjecture this degradation is due to the model being trained on a broader range of general tasks, which may compromise its specialized capabilities in handling complex problems. During the development of Qwen3, we choose to accept this performance trade-off to enhance the model's overall versatility”(§4.7). 즉 저자들은 복잡한 문제를 다루는 전문 능력의 손상 가능성을 더 넓은 일반 task 학습 때문이라고 추정하며, 전체 활용성을 높이기 위해 이 trade-off를 받아들였다고 설명한다(해석).

**조건과 한계.** 세 보고서 모두 전문가 경로를 같은 데이터를 직접 혼합해 학습한 단일 모델과 비교하지 않았다. 따라서 전문가를 썼을 때의 이득은 측정되지 않았다. GLM-4.5와 Llama 3는 범용 모델 옆에 전문가별 점수를 제시하지 않는다. 별도의 8B 연구에서는, pre-release Aya 23 8B 모델의 general-only DPO checkpoint와 safety-only DPO checkpoint를 SLERP(두 weight vector의 spherical linear interpolation)로 merge했을 때, 15% safety mixture로 학습한 단일 모델의 71.0% win rate와 −54.7% relative harm에 비해 78.0% win rate와 −57.8% relative harm을 얻었다([[mix-data-or-merge-models]], Table 1). 이 비교는 ch-30c에서 다룬다.

**함의.** 전문가 경로는 결합 전에 도메인별 RL을 수행할 수 있게 한다. 다만 결합 단계 자체가 이 장 §2-§3에서 다룬 간섭 문제를 가진 SFT mixture가 된다. ch-35는 연구소들이 전문가 데이터를 어느 지점에 넣는지 지도로 보여 준다.

## §5 혼합에서 agentic 데이터가 차지하는 비율

**정의.** Agentic SFT 데이터는 target에 tool call 또는 action이 포함되고, context에는 환경 observation이 포함되는 multi-turn trajectory다.

**문제.** Agent trajectory는 형식과 길이가 chat과 다르다. agent 데이터만으로 이루어진 mixture는 그 형식에 overfit하여 general 능력과 held-out agent 능력을 모두 낮출 수 있다. 반대로 agent 데이터가 전혀 없는 mixture는 held-in agent task 성능을 낮게 남긴다(AgentLM-7B의 general-only held-in score는 0.38, mixed는 1.96; [[agenttuning]] Table 5).

**AgentTuning ([[agenttuning]], Llama-2-chat 7B/13B/70B).** Mixture objective는 다음과 같다.

```
J(θ) = η · E_(x,y)~D_agent [log π_θ(y | x)] + (1 − η) · E_(x,y)~D_general [log π_θ(y | x)]
```

π_θ(y | x)는 instruction과 history x가 주어졌을 때 응답 y를 낼 모델의 확률이다. D_agent는 6개 AgentBench task에서 얻어 주로 GPT-4가 생성한, filter를 거친 trajectory 1,866개다. D_general은 ShareGPT다. η는 agent sampling ratio다(§2.2.2, Eq. 1; App. A). 7B에서 η를 0부터 1까지 0.1 간격으로 scan했으며, held-out agent task에서는 η = 0.2가 가장 좋았다. Table 6은 단위를 쓰지 않은 batch size 64를 제시한다. 이를 example 수로 읽으면 η = 0.2에서 batch당 기대 agent example 수는 12.8개다(계산으로 도출).

Table 5(held-in agent / held-out agent / general, normalized score):

| Size | Mixed η = 0.2 | General only | Agent only |
|---|---|---|---|
| 7B | 1.96 / 0.67 / 0.63 | 0.38 / 0.64 / 0.61 | 1.34 / 0.09 / 0.22 |
| 70B | 2.55 / 1.40 / 0.96 | 0.99 / 0.98 / 1.00 | 2.47 / 0.87 / 0.83 |

Agent-only 데이터는 held-in agent task를 올리지만 held-out agent task와 general ability는 낮춘다. 7B에서 mixture의 held-out score는 general-only와 가깝다(0.67 대 0.64). Mixing의 held-out 이점은 70B에서 나타난다(1.40 대 0.98).

**Agent-FLAN ([[agent-flan]], Llama2 7B ablation).** T-Eval은 이 논문의 평가에서 held-out tool-utilization benchmark다(§4). Agentic 데이터에 세 가지 변경을 가했다. ReAct-format turn(논문이 부르는 Thought-Action-Observation 형식)을 multi-turn chat으로 다시 썼고(T-Eval 61.8 → 64.9), 데이터를 capability별로 분해하고 reweight했으며, negative sample을 추가했다(아래 negative-samples 절 참고). Reasoning data를 절반으로 줄이면 T-Eval은 63.8, understanding data를 절반으로 줄이면 64.6이 된다. 반면 retrieval data(65.3)나 instruction-following data(65.9)를 절반으로 줄여도 점수가 낮아지지 않는다(Table 2). Weighted mixture는 reasoning : retrieval : understanding = 1 : 0.25 : 0.75를 쓰며, 37.3M 대신 18.1M token으로 T-Eval 66.3을 얻는다(Table 2; App. A). ShareGPT는 agent corpus와 1:1로 혼합하지만 단위는 밝히지 않았다(App. A). 7B general score는 MMLU 50.0에서 49.7, GSM8K 21.9에서 22.1, HumanEval 15.1에서 15.5로 움직인다(Table 4).

**Agent Data Protocol ([[agent-data-protocol]], Qwen2.5-Coder-Instruct 7B-32B, Qwen2.5-7B-Instruct, Qwen3-8B).** 각 dataset은 m_d = ⌈w_d · n_d⌉로 resample한다(App. C). 다음은 OpenHands와 SWE-Agent에 사용한 non-web subset의 계산 예다(Table 1의 count, Table 9의 multiplier). App. C.1은 Mind2Web, Go-Browse, NNetNav, Synatra를 제외하지만 AgentInstruct 안의 Mind2Web 및 WebShop subset을 보존하는지는 말하지 않는다. 이 예에서는 AgentInstruct 전체를 보존한다.

- Orca AgentInstruct: raw trajectory 1,046.1K × 0.001 = sampled 1.05K. Raw non-web trajectory 중 share는 91.71%였지만 sampled example 중 share는 3.74%가 된다(계산으로 도출).
- Code-Feedback: 66.4K × 0.1 = 6.64K. Nebius SWE trajectory: 13.4K × 0.2 = 2.68K. SWE-Gym: 0.5K × 3 = 1.5K. AgentInstruct: 1.9K × 2 = 3.8K. CodeActInstruct 7.1K, SWE-smith 5.0K, openhands-feedback 0.2K는 w = 1을 유지한다.
- 합계: sampled example 27.97K개(Table count가 0.1K 단위로 반올림됐으므로 계산으로 도출). App. C.1은 “around 30K”라고 쓴다.
- Web dataset은 “web-specific interaction patterns”의 잠재적 간섭을 피하기 위해 이 subset에서 제외했다(App. C.1). Multiplier에 대한 ablation은 보고하지 않았다.

결과(Table 6; App. E.1 Table 10): OpenHands를 사용한 SWE-bench Verified에서 Qwen3-8B는 CodeActInstruct + Code-Feedback 후 0.2%, SWE-smith만 학습한 뒤 11.0%, ADP mixture 뒤 16.6%를 기록한다. 약 30K sample로 규모를 맞추고 SWE-smith를 up-sample해도 11.0%다. GAIA에서 Qwen2.5-7B-Instruct는 AgentInstruct만 학습한 뒤 0.6%, ADP 뒤 9.1%다.

**LIMI ([[limi]], GLM-4.5 355B).** GPT-5와 human collaborator가 수집한 성공 trajectory는 78개다. 평균 42.4k token, 최대 152k token이며 총 약 3.3M token이다(계산으로 도출). Table 2와 Table 3은 GLM-4.5를 네 set으로 fine-tuning한 결과를 비교한다. AgencyBench는 multi-round vibe-coding 및 research task 10개를 점수화한다. tau2 airline과 retail은 τ²-bench tool-use task이고, Pass^4, 즉 독립 run 네 번 중 성공한 비율로 점수화한다(§3.4).

| Training set | Samples | AgencyBench avg | tau2 airline | tau2 retail |
|---|---:|---:|---:|---:|
| none (GLM-4.5) | none | 45.1 | 28.0 | 36.8 |
| CC-Bench trajectories | 260 | 29.2 | 38.0 | 39.6 |
| AFM-WebAgent-SFT | 7,610 | 36.7 | 18.0 | 13.2 |
| AFM-CodeAgent-SFT | 10,000 | 47.8 | 20.0 | 16.7 |
| LIMI | 78 | 73.5 | 34.0 | 45.6 |

- 7,610-sample 및 10,000-sample single-domain set은 각각 한 run에서 tau2 retail을 36.8에서 13.2와 16.7로, tau2 airline을 28.0에서 18.0과 20.0으로 낮춘다. 이 과정은 해당 set이 target으로 삼지 않는 tool-use task에서 capability가 좁아진 사례로 읽는다(해석).
- Abstract의 “53.7% improvement”는 relative 수치다. (73.5 − 47.8) / 47.8 = 53.8%이며, 차이는 25.7 point다(계산으로 도출). §4.3은 이를 “percentage point” improvement라고 부른다.
- Printed Table 3의 LIMI average(57.2)는 printed column의 unweighted mean으로 재현되지 않는다. AgencyBench를 제외하면 45.6, 포함하면 49.1이다(계산으로 도출).
- AgencyBench는 저자 그룹의 task 10개로, training query와 같은 두 domain에서 출제됐다. Sample은 token-matched가 아니고 seed도 보고하지 않았다. Non-agentic benchmark는 coding 및 scientific-computing set(EvalPlus, DS-1000, SciCode)뿐이며 knowledge, chat, safety benchmark는 보고하지 않았다. CLI tool이 없을 때 GLM-4.5는 tau2 retail에서 LIMI보다 높다(52.6 대 49.1, Table 4).

**Llama 3.** “Reasoning and tools”는 SFT example의 21.19%다(Table 7). Tool data에는 tool이 필요 없는 query도 넣어 모델이 불필요하게 tool을 호출하지 않도록 했다([[llama-3]] §4.3.5). Tool-only share는 보고하지 않았다.

| Source | 인쇄된 agentic share | 단위 | 보고된 non-agentic ability |
|---|---|---|---|
| AgentTuning | η = 0.2 | sampling ratio | 예(MMLU, HumanEval, GSM8K, MT-Bench) |
| Agent-FLAN | ShareGPT : agent 1:1 | 밝히지 않음 | 예(MMLU, GSM8K, HumanEval) |
| ADP | multiplier w_d; 약 30K non-web sample | epoch당 example | 아니오 |
| LIMI | agentic sample 78개만 | example | coding 및 scientific computing만(EvalPlus, DS-1000, SciCode) |
| Llama 3.1 | Reasoning and tools 21.19% | example의 % | 예(full report) |

**함의.** Agentic evidence는 세 가지 조건부 rule을 뒷받침한다. Agent data는 general data와 섞고(AgentTuning), raw count가 아니라 agent source와 capability를 균형 잡으며(Agent-FLAN, ADP), single-domain agent set(LIMI 비교의 7,610개와 10,000개 sample)은 다른 agent task를 좁힐 위험으로 취급해야 한다(LIMI, ADP). ADP는 non-agentic benchmark를 전혀 보고하지 않고, LIMI는 coding 및 scientific-computing benchmark만 보고한다. 따라서 이 mixture가 knowledge, chat, safety에 주는 효과는 Open question이다. ch-45d는 stage별 agentic recipe를 비교한다.

## §6 혼합에서 long-context 데이터가 차지하는 비율

**정의.** Long-context SFT 데이터는 모델의 확장된 context window를 사용하는 example이다. Llama 3.1의 long-context example은 전체 SFT mixture의 평균 846.1 token에 비해 평균 38,135.6 token으로, 45배다(Table 7). 이 example은 16K, 32K, 64K, 128K token bucket으로 나뉜다.

**문제.** 아래에서 비교하는 두 report 모두 SFT 전에 continued pre-training으로 context window를 확장한다(ch-32b). 측정 가능한 질문은, short example만 포함한 SFT mixture가 synthetic long example을 포함한 mixture보다 SFT 후 long-context benchmark 점수를 더 낮추는가이다.

**Evidence A: Llama 3 ([[llama-3]] §4.3.4; Table 7).** “기존 SFT recipe를 short-context 데이터만으로 순진하게 적용하면 pre-training에서 얻은 long-context capability가 크게 퇴보했다.” Long 데이터는 full document를 context로 둔 8K chunk의 synthetic QA, hierarchical summary, repository code reasoning으로 이루어지며, 이전 Llama 3 모델이 생성했다. “신중한 ablation을 통해, synthetic long-context data 0.1%를 원래 short-context data와 섞는 것이 short-context 및 long-context benchmark 전반의 성능을 최적화한다는 점을 관찰했다.” Ablation table은 인쇄하지 않았다. SFT 모델이 long context에서 강할 때는 short-context DPO가 long-context performance를 해치지 않았다. 저자들은 DPO가 SFT보다 optimizer step을 적게 쓰기 때문이라고 추정한다.

**Evidence B: ProLong ([[prolong]] §5, Table 8; ProLong base model은 512K까지 long-context continued training을 40B token 수행한 뒤 UltraChat SFT를 1B token 수행한 Llama-3-8B-Instruct다).** Llama-3-8B-Instruct가 생성한 synthetic long data(QA 40%, RAG 30%, summarization 30%)를 UltraChat과 섞었으며, share의 단위는 token이다.

| Synthetic share (token) | JsonKV | RAG | Re-rank | ICL | QA | Summ. | Avg |
|---|---|---|---|---|---|---|---|
| 0% | 65.7 | 58.1 | 38.5 | 80.3 | 49.7 | 42.1 | 55.7 |
| 1% | 61.5 | 57.0 | 38.3 | 80.8 | 45.3 | 41.5 | 54.1 |
| 3% | 62.0 | 56.4 | 37.9 | 80.6 | 44.8 | 39.5 | 53.5 |
| 10% | 70.3 | 55.5 | 36.1 | 80.6 | 41.7 | 39.4 | 53.9 |
| 50% | 45.8 | 48.8 | 18.8 | 70.5 | 42.3 | 33.3 | 43.3 |

각 열은 HELMET task(§2.1)다. JsonKV는 긴 JSON file에서 key에 대응하는 value를 찾고, RAG는 retrieve한 Wikipedia passage를 바탕으로 질문에 답하며, Re-rank는 섞인 document의 순서를 매긴다. ICL은 많은 in-context example으로 classification을 학습하고, QA는 책 전체에 관한 질문에 답하며, Summ.은 긴 legal document를 요약한다. JsonKV, QA, Summ.은 512K에서 평가했고, 나머지 열은 32K와 64K의 평균이다. Llama-3-70B-Instruct generator를 사용하면 1%, 3%, 10%, 50%에서 각각 54.2, 55.4, 53.6, 49.7을 얻어(App. B.5, Table 23), 모두 55.7보다 낮다.

**조건을 명시한 충돌.** Llama 3의 example share 0.11%는 이 장 §1의 환산에서 token의 4.95%이며, ProLong의 3%와 10% token share 사이에 있다. 그러나 ProLong에서는 둘 다 0% run보다 평균이 낮다(55.7 대비 53.5와 53.9). 단위를 맞춰도 이 불일치는 해소되지 않는다.

| 조건 | Llama 3.1 | ProLong |
|---|---|---|
| Model | 8B, 70B, 405B(0.1% ablation의 size는 밝히지 않음) | Llama-3-8B-Instruct initialization |
| Long-context continued training | 약 800B token을 여섯 stage로 수행, 8K → 128K(405B에 대해 명시, §3.4.2) | 40B token, 64K 다음 512K |
| SFT data | internal, 여러 round, rejection-sampled 및 synthetic | UltraChat, 1B token |
| Long SFT data | QA, summarization, repository code; 이전 Llama 3 모델 | QA, RAG, summarization; Llama-3-8B 또는 70B-Instruct |
| Evaluation | 인쇄하지 않음 | 32K와 64K HELMET subset, 일부 task는 512K |

ProLong 저자는 두 가설을 제시한다. 이전 연구는 long-context training이 충분하지 않아 synthetic data가 추가 long-context training 역할을 했을 수 있으며, “대규모 short instruction dataset을 사용할 때 synthetic long data를 섞으면 모델이 long-context task에서 퇴화하는 일을 막을 수 있다”(§5)는 것이다. **상태: Open question.** 이 문제를 가르는 experiment는 한 모델에서 short SFT size와 long SFT share를 모두 변화시키고, seed를 포함해 long 및 short benchmark를 보고해야 한다.

GLM-4.5는 unified SFT에 최대 128K token의 long-context understanding task를 포함하지만 share는 인쇄하지 않는다([[glm-4-5]] §3.1). Library card [[long-context-llama3]]는 다시 검증하지 않았으며, 이 절의 수치는 [[llama-3]] 및 [[llama-3-recipe]]를 경유해 Llama 3 report에서 인용했다.

**함의.** Base model이 long-context continued training을 거쳤고 SFT set이 약 1B token의 short open-chat data라면, 이 조건이 ProLong과 맞으므로 short-only SFT mixture에서 시작한다. SFT set이 rejection-sampled 및 synthetic data를 포함하는 multi-round 구성이라면, 이 조건이 Llama 3와 맞으므로 example의 0.1% 부근 long-context share를 넣고 측정한다. 어느 경우든 mixture를 바꿀 때마다 long-context benchmark는 필수 gate다(ch-32c).

## §7 하나의 mixture 안의 reasoning 및 non-reasoning mode

**정의.** Reasoning mode는 답을 내기 전에 명시적인 reasoning trace를 생성한다. Non-reasoning mode는 reasoning trace 없이 곧바로 답한다. Inference 시 mode control(system prompt 또는 flag)이 둘 중 어느 mode를 선택할지 결정한다.

**문제.** Reasoning 비중이 큰 domain의 sample은 길다. DeepSeek-R1 Dev3 SFT에서 Math, Code, STEM, Logic domain은 sample의 77.90%이면서 token의 94.14%다(Table 5와 §1에서 계산으로 도출). Reasoning 중심 SFT는 Llama-Nemotron에서 IFEval도 낮췄다(아래). 따라서 two-mode mixture는 두 mode 모두에서 instruction-following 손실이 없는지 점검해야 한다.

**Llama-Nemotron ([[llama-nemotron]], arXiv:2505.00949v5).** Reasoning sample에는 system prompt `detailed thinking on`을, non-reasoning sample에는 `detailed thinking off`를 붙인다. Paired data에서는 reasoning dataset에서 무작위로 뽑은 prompt에 non-reasoning 응답을 추가한다. 이 응답은 general domain에서는 Llama-3.1-Nemotron-70B-Instruct가, 다른 domain에서는 Llama-3.3-70B-Instruct가 생성했다(§3.2). Table 2에는 SFT sample 33,011,757개가 있으며 reasoning-on 행의 합은 3,934,627개, 즉 11.9%다(계산으로 도출). Math reasoning-off만 sample의 60.1%를 차지한다.

- LN-Nano(8B)는 세 stage로 학습한다. 먼저 reasoning data만 learning rate 1e-4로 4 epoch 학습하는데, 이는 “repetitive completions 같은 failure mode를 방지”한다. 그 뒤 reasoning 및 non-reasoning data를 함께 학습하고, 마지막으로 더 작은 chat·instruction-following·tool-calling blend를 학습한다(§4.2).
- LN-Nano는 reasoning on에서 AIME24 61.3, reasoning off에서 3.0을 기록한다. 반면 BFCL V2 Live(Berkeley Function-Calling Leaderboard의 live split)는 각각 63.9와 63.6이다(Table 3). Toggle은 AIME24를 58.3 point 바꾸지만 function calling은 0.3 point만 바꾼다.
- LN-Super-SFT(49B)는 IFEval에서 on 81.9, off 83.0을 기록한다. 이는 LN-Super가 architecture search와 knowledge-distillation 40B token을 거쳐 파생된 모델인 Llama-3.3-70B-Instruct의 92.1보다 낮다(§2.1-§2.2). Report는 “reasoning-focused SFT가 IFEval score를 눈에 띄게 낮춘다”고 쓰며, 전용 IFEval RL run을 추가한다(§7.3). 그 run, Llama-3.1-Nemotron-70B-Reward에 대한 online RPO(reward-aware preference optimization) RLHF 두 iteration, model merging 뒤의 released LN-Super는 89.2(on)와 89.0(off)을 기록한다(§6, §7.3, Table 4).

**DeepSeek-R1 ([[deepseek-r1]], [[deepseek-r1-recipe]]).** Dev3 SFT set은 reasoning sample 약 600k개와 non-reasoning sample 약 200k개를 섞는다. Non-reasoning data에는 writing, factual QA, self-cognition, translation, software engineering이 포함된다. “hello 같은 더 단순한 query”에는 chain of thought를 제공하지 않는다(Supp. B.3.3). Dev2에서 Dev3으로 갈 때(Table 3), AlpacaEval 2.0은 55.8 → 62.1, IF-Eval은 72.0 → 78.1, Aider-Polyglot은 25.6 → 44.8, AIME 2024는 74.0 → 78.1로 오른다. 반면 SimpleQA는 28.2 → 24.9, C-Eval은 91.9 → 86.4로 떨어진다. Dev3는 새 data로 DeepSeek-V3-Base에서 SFT를 다시 시작했으므로, 이 차이는 non-reasoning share의 controlled ablation이 아니다.

**Qwen3 ([[qwen-3]] §4.3, Table 9).** Non-thinking sample은 `/no_think` flag를 쓰고 비어 있는 think block을 유지한다.

```
<|im_start|>user
{query} /no_think<|im_end|>
<|im_start|>assistant
<think>

</think>

{response}<|im_end|>
```

Thinking은 default이므로 일부 thinking sample은 `/think` flag를 생략한다. Multi-turn data에서는 flag를 여러 개 삽입하며, 응답은 마지막 flag를 따른다.

**관련 결과.** Non-reasoning Qwen2.5-32B student에서 DeepSeek-R1 output의 answer segment만으로 SFT하면 IFEval 76.9를 기록했다. Original community response로 학습했을 때는 73.4다. Thinking의 summary를 앞에 붙이면 61.2로 낮아진다([[am-reasoning-answers-for-non-reasoning]], Table 1).

**함의.** Two-mode mixture에는 mode별 평가(같은 suite에서 두 mode 모두), ThinkFollow 같은 mode-control metric, 각 share의 token 단위 accounting이 필요하다. DeepSeek-R1 Dev3에서 General domain은 sample의 22.10%, token의 5.86%다. Table 5는 reasoning/non-reasoning data가 아니라 domain별로 나뉘어 있으므로, 약 200k개 non-reasoning sample의 token share는 보고하지 않았다.

## §8 혼합 구성을 균형 잡는 데 쓰는 data diversity 측정값과 그 한계

**정의.** Diversity measure는 example 집합이 어떤 공간을 얼마나 넓게 덮는지에 수치를 부여한다. 그 공간은 embedding space, tag space, 또는 task label일 수 있다.

**문제.** Quality만으로 selection하면 높은 점수를 받은 거의 중복인 example을 고를 수 있다. 그러면 instruction distribution의 일부는 덮이지 않은 채 남는다.

**Facility location (QDIT, [[qdit-data-diversity]] §3).**

```
d(A) = Σ_{v∈V} max_{a∈A} sim(a, v)
f(a | A, α) = (1 − α) · d(a | A) + α · q(a)
```

V는 candidate pool이고, A는 선택한 subset이며, sim은 instruction embedding(all-mpnet-base-v2)의 cosine similarity다. `d(a | A) = d(A ∪ {a}) − d(A)`는 marginal diversity gain이고, `q(a)`는 reward model 또는 ChatGPT rating에서 얻은 quality score다. `α ∈ [0, 1]`은 두 값 사이의 trade-off를 정한다. Selection은 매 step에서 f가 가장 큰 example을 더한다.

**계산 예(네 prompt 중 둘 선택).** Similarity는 `sim(v1, v2) = 0.9`(거의 중복인 math prompt 둘), `sim(v1, v3) = 0.1`, `sim(v1, v4) = 0.2`, `sim(v2, v3) = 0.1`, `sim(v2, v4) = 0.2`, `sim(v3, v4) = 0.3`, 그리고 `sim(v, v) = 1`이다. Quality는 `q = 0.9, 0.8, 0.5, 0.6`이다.

1. Step 1에서 `α = 0.5`라면, `d({v1}) = 1 + 0.9 + 0.1 + 0.2 = 2.2`이므로 `f(v1) = 0.5 × 2.2 + 0.5 × 0.9 = 1.55`다. `f(v2) = 1.50`, `f(v3) = 1.00`, `f(v4) = 1.15`이므로 v1을 선택한다.
2. Step 2에서 `d({v1, v2}) = 1 + 1 + 0.1 + 0.2 = 2.3`, gain은 0.1, `f = 0.05 + 0.40 = 0.45`다. `d({v1, v3}) = 1 + 0.9 + 1 + 0.3 = 3.2`, gain은 1.0, `f = 0.50 + 0.25 = 0.75`다. `d({v1, v4}) = 3.2`, gain은 1.0, `f = 0.50 + 0.30 = 0.80`이므로 v4를 선택한다.
3. Quality-only selection(`α = 1`)은 `{v1, v2}`를 고른다. mean quality는 0.85, diversity는 2.3이다. `α = 0.5`의 QDIT는 `{v1, v4}`를 골라 mean quality 0.75, diversity 3.2를 얻는다.

논문은 d와 q를 공통 scale로 어떻게 맞추는지 밝히지 않는다. 위 예는 raw value를 쓴다.

**근거.** LLaMA-1 7B에서 Ultrachat 1.3M → 10K를 줄일 때, lowest-10% reward-model score는 random 2.620, quality-only 3.405, QDIT 3.497이고, mean은 각각 6.219, 6.961, 6.993이다(Table 1). Main result에서 QDIT는 quality-only selection보다 worst-case reward score를 5.2%, Alpaca 52K에 대한 worst-case winning score를 6.26% 높인다(§4.3). `α ∈ {0.5, 0.7, 0.9}`가 보통 가장 좋고, `α = 0.1`과 `α = 1`은 둘 다 더 낮은 성능을 낸다(§4.4).

**Tag와 embedding threshold.** InsTag([[instag]])는 instruction에 open-set intent tag를 부여한다. DMT는 이 tag를 사용해 ShareGPT에서 code와 math example을 제거했다(이 장 §2). Llama 3는 InsTag intention 수를 difficulty score로 사용하고, 인쇄하지 않은 cosine threshold로 RoBERTa cluster 안에서 semantic deduplication을 실행했다([[llama-3-recipe]] §4.2.3). DEITA의 embedding-distance filter(threshold 0.9)는 LLaMA-1 13B에서 300K pool에서 6K sample을 선택할 때 MT-Bench 6.17을 기록했다. 이는 InsTag diversity 6.10, random selection 5.82와 비교된다([[deita]], Table 4). QDIT는 threshold deduplication과 cluster-balanced selection이 facility location보다 낮은 worst-case performance를 낸다고 보고한다(§4.4, Fig. 8a).

**한계.**

- 네 measure 모두 candidate pool의 coverage를 서술한다. Deployment distribution의 coverage나 pool에 존재하지 않는 skill의 coverage는 서술하지 않는다.
- 평가는 7B–13B에서 judge 또는 reward model에 기반하며, maximum length가 짧다(QDIT는 512 token).
- 어느 연구도 math, agentic, long-context data 같은 skill domain 전체에 걸쳐 selection하지 않는다. Domain balancing에는 여전히 이 장 §2–§3의 per-skill evaluation이 필요하다.
- ADP는 matched-scale gain을 corpus의 “the greater diversity and unified structure”로 돌린다(App. E.1; Interpretation). 그러나 그 비교가 control한 것은 sample count이며 domain composition은 아니다. Agent-FLAN은 data-scaling curve에서 volume을 더하는 것보다 diversity 또는 quality를 높이는 일이 다음 단계라고 결론 내린다(§5.1.1; Interpretation). 두 논문 모두 diversity를 숫자로 측정하지 않는다.

**함의.** Diversity measure는 quality score와 결합했을 때 하나의 pool 안에서 selection을 균형 잡을 수 있다(QDIT의 `α ∈ {0.5, 0.7, 0.9}`). General-purpose mixture에서 각 skill domain이 차지할 share는 이 measure로 정해지지 않으며, per-skill 및 unseen evaluation으로 여전히 선택해야 한다.

## 부정 표본과 부정 피드백

이 장에서 말하는 “부정(negative)”은 style-guide §6.1의 뜻을 따른다. SFT mixture에는 유형 1과 2가 들어가며, 이 장의 출처는 부정 gradient(유형 4)를 사용하지 않는다.

1. **부정 표본의 출처.** 실행과 task 성공(AgentTuning은 reward `r = 1` trajectory와 Mind2Web에서는 `r ≥ 2/3`인 trajectory를 보존하며, Table 1은 35,341개 중 1,866개를 보존한다), judge agent(GLM-4.5는 여러 judge agent가 완료했다고 표시한 agent trajectory를 보존, §3.1), reward model(Llama 3의 상위 사분위 quality, §4.2.3), 그리고 어떤 skill에는 negative marginal value를 갖는 subset을 드러내는 제거 ablation(Tülu 3 WildChat이 safety score에 보이는 효과, 이 장 §3)이 있다. False-negative rate는 어느 경우에도 보고하지 않았다.
2. **유형 1, negative marginal value, 폐기.** Filter하지 않은 trajectory로 학습한 AgentTuning은 7B에서 held-in 1.34, held-out 0.47을 기록하며, filter한 경우의 1.96, 0.65보다 낮다(Table 2). DMT ablation에서는 InsTag로 code·math tag를 붙인 ShareGPT example을 제거하자 high-resource conflict가 “어느 정도” 줄었다(§4.2). 한 subset은 한 skill에는 negative value이고 다른 skill에는 positive value일 수 있으므로, “폐기” 결정은 모든 skill을 대상으로 해야 한다.
3. **유형 2, 내용으로서의 negative.** Agent-FLAN은 tool call을 유도하지만 정답 target은 일반 text reply인 prompt를 추가한다(tool이 없거나 관련 없는 tool만 주어진 경우). 이는 ToolBench query 761개로 만들었다(App. C). 7B에서 ReAct-keyword hallucination은 15.6에서 9.9로, general-keyword hallucination은 13.5에서 11.9로 낮고, T-Eval은 66.3에서 66.0으로 움직인다([[agent-flan]] Table 3). Llama 3는 tool이 필요 없는 query(§4.3.5)와 knowledge-probe refusal(§4.3.6)을 더한다. Tülu 3는 safe prompt의 over-refusal을 막는 데 도움이 됐다고 저자들이 보고한 CoCoNot contrastive prompt를 추가한다(§4.2; 수치 없음).
4. **기제.** 유형 2 표본은 교정된 target에 ordinary cross-entropy로 학습한다. logit `z_j`에 대한 `log p_y`의 gradient는 `1[j = y] − p_j`이므로, 일반 text target token의 확률을 높이는 일은 shared softmax를 통해서만 경쟁 tool-call token의 확률을 낮춘다. 어떤 표본의 likelihood도 직접 아래로 밀지 않는다(ch-30은 이 implicit suppression을 다룬다).
5. **Control.** Agent-FLAN은 source query 761개에서 negative-as-content 표본을 만들고 이를 agent sample 24,703개와 비교하며, positive capability에 주는 영향도 확인한다(T-Eval 변화 −0.3).
6. **Diagnostic.** Tool이 필요 없는 prompt에서 tool-call rate, safe prompt에서 refusal rate(CoCoNot-style), filter를 바꾼 뒤의 per-skill delta를 측정한다.
7. **범용성에 미치는 효과.** 유형 2 표본은 over-triggering(tool call, refusal)을 겨냥하며, 이는 narrowing failure다. Unseen tool-use task에 미치는 효과는 보고하지 않았다.

## 레시피

| 모델(정확한 release) | 규모 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DMT run (LLaMA-1) | 7B, 13B, 33B | SFT | epoch; peak LR; warmup | 3; 2e-5; 3% | arXiv:2310.05492v4 §3.1, App. C | verified 2026-09-15 | ablation 보고 없음 |
| DMT run (LLaMA-1) | 7B, 13B, 33B | SFT | batch size | 16(§3.1); 128(App. C) | §3.1; App. C | conflict | 논문은 Table 1을 만든 값이 어느 것인지 말하지 않는다 |
| DMT run (LLaMA-1) | 7B, 13B, 33B | SFT | 전체 dataset 크기 | GSM8K RFT 110,142; Code Alpaca 20,022; ShareGPT 86,060 examples | App. A Table 2 | verified 2026-09-15 | 해당 없음 |
| DMT (`k = 1/256`) | 7B, 13B, 33B | SFT | stage-2 replay | ShareGPT + GSM8K RFT 및 Code Alpaca의 1/256(430 + 78 examples) | Table 1; Table 2 | verified 2026-09-15 (`k`); derived(count) | Table 1: 7B GSM8K 41.92 대 mixed sequential 32.60; Fig. 5 `k` sweep |
| Tülu 1 Human+GPT mix | 7B-65B | SFT | mixing rule; instance | sampling weight 없이 7 dataset을 concatenate; 490,694 | arXiv:2306.04751v2 §3.3, Table 1 | verified 2026-09-14 ([[tulu-1-how-far-can-camels-go]]); total derived | Table 3: mix average 45.2 대 best single dataset 42.0(13B) |
| Tülu 3 8B SFT | 8B | SFT | prompt | 939,344 | arXiv:2411.15124v5 Table 7 | verified 2026-09-15 | Table 10 removal ablation |
| Tülu 3 8B SFT | 8B | SFT | peak LR; schedule; warmup ratio; epoch; effective batch; max length; loss aggregation | 5e-6; linear; 0.03; 2; 128; 4,096; sum loss | Table 11; §4.3; §4.3.2 | verified 2026-09-15 | §4.3.2 Fig. 5-6(Llama 3.0의 Tülu 2 mixture이며 final mix가 아님): sum loss에서 5e-6이 LR와 loss type 전반에서 최상, 2 epoch가 최상; figure만 있고 seed는 밝히지 않음 |
| Tülu 3 70B SFT | 70B | SFT | peak LR(다른 설정은 8B와 같음) | 2e-6 | Table 11 | verified 2026-09-15 | 위와 같음 |
| Tülu 3 8B SFT | 8B | SFT | compute | 32 GPU로 6시간(8×H100 node) | §4.3 | verified 2026-09-15 | 해당 없음 |
| Tülu 3 8B / 70B SFT | 8B, 70B | eval-gate | checkpoint selection | soup이 아닌 best single-seed run | §4.3.1, Table 14 | verified 2026-09-15 | Table 14: 8B seed 59.8-60.1, best soup 60.2 |
| Llama 3.1 | 범위 미정 | SFT | mixture | General English 52.66%, Code 14.89%, Multilingual 3.01%, Exam-like 8.14%, Reasoning and tools 21.19%, Long context 0.11% of examples | arXiv:2407.21783v3 Table 7 | verified 2026-09-14 ([[llama-3-recipe]]) | mix를 매 round “폭넓은 benchmark에 걸친 성능을 조정”하도록 변경(§4.2.2); 표 없음 |
| Llama 3.1 | 범위 미정 | SFT | long-context share | synthetic long-context data 0.1%; 16K, 32K, 64K, 128K bucket | §4.3.4 | verified 2026-09-14 ([[llama-3-recipe]]) | “careful ablation”; 표 없음 |
| Llama 3.1 | 범위 미정 | SFT | long-context token share | token의 4.95% (`0.0011 × 38,135.6 ÷ 847.1`) | Table 7 | derived | 해당 없음 |
| Llama 3.1 405B | 405B | SFT | LR; step | 1e-5; 8.5K-9K step(“our largest models”) | §4.1.3 | verified 2026-09-14 ([[llama-3-recipe]]) | “서로 다른 round와 data mix에서 잘 작동”; 수치 없음 |
| AgentLM-7B/13B/70B | 7B, 13B, 70B | SFT | agent sampling ratio `η`; general data | 0.2; ShareGPT에서 GPT-4 : GPT-3.5 = 0.2 : 0.8 | arXiv:2310.12823v2 §2.2.1-§2.2.2; App. A | verified 2026-09-14 ([[agenttuning]]) | 7B에서 `η`를 0-1, 0.1 step으로 scan, held-out 최상(per-`η` score는 표로 없음); Table 5 endpoint |
| AgentLM-7B / 13B; AgentLM-70B | 7B, 13B; 70B | SFT | peak LR | 5e-5; 1e-5 | Table 6 | verified 2026-09-14 ([[agenttuning]]) | ablation 보고 없음 |
| Agent-FLAN (Llama2) | 7B, 13B, 70B | SFT | ShareGPT : agent; format mix; capability weight | 1:1(unit 미기재); ReAct 10%, conversation 90%; reasoning : retrieval : understanding = 1 : 0.25 : 0.75 | arXiv:2403.12881v1 App. A | verified 2026-09-14 ([[agent-flan]]) | Table 2의 7B half-data ablation; 1:1은 App. A가 따른다고 말하는 AgentTuning `η = 0.2`와 충돌 |
| Agent-FLAN (Llama2) | 7B, 13B, 70B | SFT | epoch; peak LR | 1; 2e-5 | App. A Table 5 | verified 2026-09-14 ([[agent-flan]]) | ablation 보고 없음 |
| ADP (OpenHands, SWE-Agent run) | 7B-32B | SFT | dataset별 multiplier `w_d` | orca agentinstruct 0.001; synatra 0.01; code feedback 0.1; nebius SWE 0.2; agenttuning subset 2; swe-gym openhands 3; 나머지 1 | arXiv:2510.24702v2 App. C Table 9 | verified 2026-09-15 | ablation 보고 없음 |
| ADP (OpenHands, SWE-Agent run); ADP (AgentLab run) | 7B-32B | SFT | subset; sampled size | non-web만 약 30K; web만 약 20K | App. C.1 | verified 2026-09-15 | Table 10: matched scale에서 SWE-bench Verified 16.6 대 11.0(Qwen3-8B) |
| ADP | 7B-32B | SFT | LR; epoch; sequence length | 보고 없음(LLaMA-Factory pipeline) | §5.1, App. C-E 확인 | not reported | 해당 없음 |
| LIMI(from GLM-4.5) | 355B | SFT | sample; token per sample | 78; mean 42.4k, max 152k | arXiv:2509.17567v2 §3.2, Fig. 4 | verified 2026-09-15 | Table 2 대 260 / 7,610 / 10,000-sample set, 각 한 run |
| LIMI | 355B | SFT | LR; epoch; batch | 보고 없음(slime framework) | §4.1 확인 | not reported | 해당 없음 |
| LN-Nano | 8B | SFT | stage; stage-1 LR와 epoch; batch; packed length | 3 stage; 1e-4, 4 epoch; 256; 32k | arXiv:2505.00949v5 §4.2 | verified 2026-09-15 | stage 1이 “repetitive completions 같은 failure mode를 막는다”; 표 없음 |
| LN-Super | 49B | SFT | LR; epoch; length; batch | fixed 5e-6; 1; 16k; 256 | §4.2 | verified 2026-09-15 | 더 작은 run은 5e-5에서 최대 3-4 epoch gain을 시사; 표 없음 |
| LN-Ultra | 253B | SFT | LR schedule; packed length; batch | 10% 동안 1e-5까지 warmup, cosine으로 1e-6; 24k; 256 | §4.2 | verified 2026-09-15 | 초기 ablation: 5e-5 등 LR은 대체로 outcome 개선, 일관되게 높은 LR은 gradient explosion을 포함한 instability 발생; 표 없음 |
| Llama-Nemotron SFT data | 8B-253B | SFT | sample; reasoning-on share | 33,011,757; reasoning-on 3,934,627(11.9%) | Table 2 | verified 2026-09-15; share derived | ablation 보고 없음 |
| DeepSeek-R1 Dev3 | 671B MoE | SFT | domain별 sample | Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812; total 804,745 | arXiv:2501.12948v2 Supp. B.3.3 Table 5 | verified 2026-09-14 ([[deepseek-r1-recipe]]) | Table 3 Dev2 → Dev3이며 controlled ablation 아님 |
| DeepSeek-R1 Dev3 | 671B MoE | SFT | General token share | 5.86% (`177,812 × 1,419.8 ÷ 804,745 × 5,355.3`) | Table 5 | derived | 해당 없음 |
| DeepSeek-R1 Dev3 | 671B MoE | SFT | epoch | 2-3(v2 B.4.2); 2(v1 §2.3.3) | v2 B.4.2; v1 §2.3.3 | conflict | 어느 version도 released checkpoint를 만든 count를 말하지 않음 |
| Qwen3-32B Stage 3 | 32B | SFT | data size; thinking : non-thinking ratio | 보고 없음 | arXiv:2505.09388v1 §4.3, Table 9, §4.7 확인 | not reported | Table 22 stage별 score |
| GLM-4.5 unified SFT | 355B | SFT | sample; maximum context | “millions of samples”; 128K token | arXiv:2508.06471v1 §3.1 | verified 2026-09-14 ([[glm-4-5]]) | ablation 보고 없음; thinking balance ratio도 보고 없음 |
| ProLong-512k-Instruct | 8B | SFT | data; synthetic long share; peak LR; global batch | UltraChat 1B token; 0%(token share); 2e-5; 4M token | arXiv:2410.02660v4 Table 9; Table 8 | verified 2026-09-14 ([[prolong-recipe]]) | Table 8: 0% avg 55.7 대 1%, 3%, 10%, 50%의 54.1, 53.5, 53.9, 43.3 |
| QDIT selection (LLaMA-1) | 7B | SFT | `α`; selected size; LR; epoch; batch; max length | dataset별 `α`(App. Table 5), 보통 `{0.5, 0.7, 0.9}`가 최상; 10K; 2e-5; 3; 128; 512 | arXiv:2311.14736v3 §4.2, §4.4, App. D Table 4 | verified 2026-09-15 | Fig. 4 `α` sweep |

**작은 general-purpose run의 시작점.** Verified row는 서로 다른 setup에서 나왔고 함께 시험된 적이 없다. Open data를 쓰는 7B-8B model이라면 Tülu 3 8B SFT setting이 문서화된 baseline이다. Llama 3.1 8B에서 prompt 939,344개, sum loss와 peak LR 5e-6(LR은 sum loss를 위해 선택), linear schedule, warmup ratio 0.03, 2 epoch, effective batch 128, maximum length 4,096이며, 32 GPU로 6시간을 쓴다. Specialist skill을 앞 stage에서 학습했다면 DMT의 final-stage replay fraction `k = 1/256`은 LLaMA-1 7B-33B에서 3 epoch, LR 2e-5로 시험됐다. General chat data에 agent trajectory를 추가한다면 AgentTuning의 `η = 0.2`는 ShareGPT를 general data로 쓰고 Llama-2-chat-7B의 held-out agent task에서 선택됐다. Long-context data는 verified value 두 개가 충돌한다(Llama 3.1에서 model size를 밝히지 않은 example의 0.1%, long-context training 40B token 뒤 8B ProLong에서 0%). 따라서 share는 default가 아니라 측정할 hyperparameter다.

## 일반화 관점

**(a) 폭을 넓히는 것.**
- Skill별 데이터가 적을 때 source를 섞기: 각 dataset의 1/256에서 mixed source는 single source보다 높은 점수를 내며, 저자들은 LLaMA-7B code를 outlier로 적었다([[sft-data-composition-dmt]] §3.3, Fig. 3).
- 실제 사용자 대화: Tülu 3 8B에서 WildChat을 제거하면 AlpacaEval 2는 4.9, BBH는 2.3 낮아진다([[tulu-3]] Table 10).
- Agent data와 함께 쓰는 general data: AgentLM-70B held-out agent score는 `η = 0.2`에서 1.40이고 agent-only에서는 0.87이다([[agenttuning]] Table 5).
- Single-domain set보다 cross-domain agentic mixture: matched sample count에서 SWE-bench Verified 16.6% 대 11.0%([[agent-data-protocol]] Table 10).
- Worst-case prompt를 위한 diversity-aware selection: quality-only selection보다 worst-case reward score +5.2%([[qdit-data-diversity]] §4.3).

**(b) Narrowing 또는 forgetting을 일으키는 것.**
- High resource에서 full-size mixing: 33B에서 multi-task는 code-only보다 HumanEval −7.92([[sft-data-composition-dmt]] Table 1).
- Replay 없는 sequential skill stage: 7B GSM8K 31.39 대 49.10(같은 표).
- Single-domain agent set: GLM-4.5에서 tau2 retail 36.8 → 13.2(web set), 16.7(code set)([[limi]] Table 3); AgentLM-7B general score는 agent-only 0.22, mixed 0.63([[agenttuning]] Table 5).
- Reasoning RL 뒤의 더 넓은 general SFT와 RL: Qwen3-32B thinking AIME'24 83.8 → 81.4, LiveCodeBench 68.4 → 65.7([[qwen-3]] Table 22).
- Reasoning-focused SFT: LN-Super-SFT IFEval은 on 81.9-83.0이고, 그 기반인 Llama-3.3-70B-Instruct는 92.1이다. 이 비교에는 49B architecture search와 distillation도 들어가며, report는 하락을 reasoning-focused SFT에 돌린다([[llama-nemotron]] §7.3, Table 4).
- Development-benchmark tuning: Tülu 3 8B에서 Persona data는 IFEval +19.2, IFEval-OOD −0.4([[tulu-3]] Table 32).
- RL stage에서 Nemotron 3 white paper는 staged single-task approach가 일부 capability를 흔히 낮춘다는 DeepSeek-V3.2-Exp의 관찰을 인용하고 모든 environment를 함께 학습한다. Staged-versus-simultaneous 수치는 인쇄하지 않았다([[nemotron-ultra]] §2.6).

**(c) 이 stage에서의 측정법.**
- Added 또는 removed subset마다 unchanged mixture 대비 모든 skill의 delta와 seed를 보고한다. Tülu 3의 8B average seed spread는 0.3 point(Table 14)이며 per-benchmark spread는 로컬에서 측정해야 한다.
- 각 development benchmark에 같은 skill의 unseen benchmark를 짝지어 둔다(Tülu 3 §7.4는 IFEval과 IFEval-OOD, HumanEval과 BigCodeBench를 짝짓는다).
- Agentic data에서는 held-in agent task, held-out agent task, non-agentic general benchmark를 모두 보고한다(AgentTuning의 세 group). ADP는 셋째 group을 생략하고, LIMI는 그 용도로 coding과 scientific-computing benchmark만 보고한다.
- Long-context data는 SFT 전이 아니라 SFT 후 long-context task(ProLong §2.2)와, 같은 run의 short-context task를 함께 보고한다.
- Two-mode model은 같은 suite에서 두 mode와 mode-switching metric(Qwen3 ThinkFollow)을 보고한다.
- Mixture share를 example과 token 모두로 보고하고 loss aggregation을 명시한다(이 장 §1).
- 알려진 measurement error: DMT의 GPT-4 judge MT-Bench; LIMI의 relative-versus-point wording과 재현되지 않는 average; AIME'24는 30문항이므로 3.33 point가 한 문항인 Qwen3 §4.6.

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 |
|---|---|---|
| 단위 없이 share를 복사 | “0.1% long-context” mixture를 token share로 복사하면 source long-context token의 1/45가 됨 | 학습 전 calculator로 example, token, loss-token share를 계산 |
| Mixture는 유지한 채 loss aggregation 변경 | Long-response domain의 weight가 바뀌고, 데이터 변경 없이 per-skill score가 이동 | Step마다 domain별 loss-token count를 log하고 per-sequence와 token-mean run 비교 |
| Average만 보고 mixture 변경 판단 | Average는 1 point 움직이나 한 skill은 19 point 움직임 | 모든 변경에 Tülu 3 Table 10 형식의 per-skill delta table |
| 1-point single-benchmark change를 결과로 취급 | Rerun 사이에 delta 부호가 뒤집힘 | Mixture마다 적어도 두 seed를 실행하고 benchmark별 spread 보고 |
| Subset을 자기 development benchmark에서 tuning | Development score는 오르지만 paired unseen benchmark는 flat 또는 하락 | Development benchmark마다 같은 skill의 unseen benchmark를 짝지음 |
| Replay 없이 skill을 순서대로 학습 | Later stage 뒤 earlier skill 하락(DMT에서 GSM8K −17.71) | 각 stage 뒤 모든 skill 평가; replay fraction 추가(DMT에서는 1/256) |
| Agent-only 또는 single-domain agent SFT | Held-in agent task는 오르나 held-out agent task와 chat benchmark는 하락 | Held-out agent task와 non-agentic benchmark를 함께 평가 |
| Raw count로 agent source 균형화 | 한 synthetic source가 sampled example을 지배(Orca는 raw non-web ADP row의 91.71%) | Dataset별 multiplier를 적용하고 sampled share를 출력 |
| Short SFT 뒤 long-context 능력이 남는다고, 또는 synthetic long SFT가 도움이 된다고 가정 | SFT 뒤 long-context benchmark 하락, 또는 synthetic long data를 넣은 뒤 하락 | 같은 setup에서 0%와 example의 0.1% long share를 두고 SFT 후 long-context task 측정 |
| Two-mode model을 한 mode로 평가 | Reasoning-off 또는 chat regression이 보고되지 않음 | 두 mode와 mode-switch accuracy 보고 |
| Quality-only data selection | Near-duplicate selection, worst-case prompt 실패 | Pool coverage(facility location)와 bottom-10% score 측정 |

## 이해 확인

1. DMT는 각 dataset의 1/256에서는 mixing이 각 skill에 도움을 주고 full size에서는 해를 준다고 찾았다. 두 결과 모두와 양립하는 mechanism 하나를 제시하고, 당신의 mechanism을 저자들의 “다른 source의 data는 noise로 볼 수 있다” 해석과 구별할 measurement를 하나 적어라.
2. Llama 3.1 long-context example은 example의 0.11%, token의 4.95%, final-response token의 0.26%다. 어느 loss mask와 aggregation rule에서 각각의 수가 gradient weight를 설명하는가? 그 답이 ProLong과의 비교를 왜 바꾸는가?
3. Tülu 3 8B에서 Persona data를 제거하면 IFEval은 19.2 point 낮고 IFEval-OOD는 +0.4 움직인다. 이것이 Persona instruction-following data가 무엇을 가르쳤다는 뜻인지 설명하고, 유지 여부를 어떻게 시험할지 결정하라.
4. AgentTuning의 agent-only model은 held-in agent task에서 1.34, held-out agent task에서 0.09이고, mixed model은 각각 1.96, 0.67이다. General chat data를 제거하면 unseen agent task 성능도 낮아질 수 있는 이유와, general-only 대비 mixing advantage가 70B에서만 나타나는 이유를 설명하라.
5. Llama 3와 ProLong setting의 차이 세 가지를 들어 상반된 long-context SFT 결과를 설명하고, 가장 가능성 높은 설명을 시험할 experiment 하나를 설계하라.
6. LIMI의 78 sample은 AgencyBench에서 10,000-sample set보다 높다. Sample count 외에 달라진 variable을 적고, 이 비교가 general agentic ability에 대해 curation이 scale보다 효과적임을 입증하지 못하는 이유를 설명하라.
7. Qwen3-32B는 mode fusion과 general RL 뒤 thinking mode AIME'24에서 2.4 point를 잃었다. Qwen3 report §4.6의 evaluation protocol을 사용해, 이 변화가 sampling noise와 구별되는지와 결정하려면 report가 어떤 data를 인쇄해야 하는지 설명하라.
8. Facility-location example에서 quality-only selection은 near-duplicate prompt 두 개를 고른다. Average quality가 더 높아도 이것이 worst-case instruction-following performance를 낮출 수 있는 이유와 quality-only selection이 더 나은 경우를 설명하라.

## 연결

- 의존성: ch-18 — *The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix*(여기서 가르치는 evidence를 쓰는 Mix step); ch-30a — *Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control*(forgetting 측정과 replay).
- 이전 장: ch-30a — *Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control*.
- 다음 장: ch-30c — *Weight Averaging and Model Merging for Generalist Models*(data mixing의 대안인 별도 학습 model의 weight merging).
- 관련: ch-30 — *SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate*(loss aggregation, implicit negative); ch-29e — *Instruction Tuning and Generalization to Unseen Tasks*(task diversity); ch-32b — *Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression*; ch-32c — *Claimed versus Effective Context Length and Long-Context Evaluation*; ch-34 — *Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports*(맥락 속 Qwen3 mode fusion); ch-35 — *Distillation in Practice A: Where Labs Insert Teacher Data*(expert-to-unified placement); ch-35a — *Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters*(thinking on/off pair); ch-45d — *Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL*; ch-47 — *Evaluation Harness and Suite Design for General Capability*; ch-49 — *Judge Models: Bias, Calibration, and Judge-Specific Overfitting*.

## 출처

- [[sft-data-composition-dmt]] — DMT의 amount·ratio·order experiment, Table 1 strategy, replay fraction `k`(arXiv:2310.05492v4 chapter excerpt).
- [[tulu-1-how-far-can-camels-go]] — LLaMA 13B의 12-dataset 비교, 어느 한 dataset도 전체에서 최상이 아님.
- [[tulu-3]] — SFT mixture 구성, Table 10 removal ablation, Table 14 seed, Table 32 unseen evaluation, Table 11 setting, §4.3.2 sum loss(arXiv:2411.15124v5에서 2026-09-15에 값 확인).
- [[tulu-3-sft-mix]] — Released SFT mixture component list; 이 장의 component count는 논문의 Table 7에서 가져옴.
- [[llama-3]] — Table 7 SFT composition, code·multilingual expert, 0.1% long-context share, tool need가 없는 tool-use query.
- [[llama-3-recipe]] — Verified SFT ledger row와 data-pruning signal(InsTag difficulty, semantic deduplication).
- [[long-context-llama3]] — Llama 3 long-context material의 owner card; 아직 재검증하지 않아 수치에는 쓰지 않음.
- [[glm-4-5]] — Reasoning, Agent, General-chat expert를 하나의 hybrid model로 distill.
- [[qwen-3]] — Thinking Mode Fusion template와 Table 22 stage별 score(arXiv:2505.09388v1에서 2026-09-15에 값 확인).
- [[deepseek-r1]] — Dev3 reasoning·non-reasoning SFT data와 Table 3 stage result.
- [[deepseek-r1-recipe]] — Table 5 domain count와 SFT epoch conflict.
- [[agenttuning]] — `η`-weighted mixture objective와 agent-only·general-only·mixed ablation.
- [[agent-flan]] — Chat-format alignment, capability weight, negative-as-content sample.
- [[agent-data-protocol]] — Dataset별 sampling multiplier, web exclusion, diverse 대 single-domain result(chapter excerpt).
- [[limi]] — 78-sample agentic SFT, tau2의 single-domain narrowing, measurement caveat(chapter excerpt).
- [[llama-nemotron]] — Reasoning on/off paired data, Table 2 split, staged SFT, IFEval 하락과 회복(chapter excerpt).
- [[prolong]] — Table 8 synthetic long SFT share, 충돌 가설, token-averaged loss.
- [[prolong-recipe]] — Verified ProLong SFT setting.
- [[qdit-data-diversity]] — Facility-location diversity, quality-diversity trade-off, worst-case evaluation(chapter excerpt).
- [[deita]] — InsTag diversity와 비교한 embedding-threshold diversity filter.
- [[instag]] — DMT와 Llama 3가 쓰는 intent-tagging method.
- [[mix-data-or-merge-models]] — Mix-versus-merge 비교 하나, ch-30c를 가리키는 pointer.
- [[am-reasoning-answers-for-non-reasoning]] — Non-reasoning student용 answer-only 대 summary target.
- [[nemotron-ultra]] — Staged 대 simultaneous multi-capability RL에 대한 qualitative statement.
