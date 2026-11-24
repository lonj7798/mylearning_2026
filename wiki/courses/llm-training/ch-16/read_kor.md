<!-- chapter: ch-16
     track: rl
     kind: content
     title: RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse
     deps: [ch-40]
     sources: [[rlvr-tulu3]], [[deepseek-r1]], [[kimi-k1-5]], [[dapo]], [[verl-dapo-recipe]], [[magistral]], [[llama-nemotron]], [[olmo-3]], [[skywork-or1-rl-data]], [[deepmath-103k]], [[multi-domain-rlvr-data-centric]], [[reasoning-or-memorization-rl-contamination]], [[replay-buffer-rlhf]], [[trl-grpo]], [[reinforcement-learning-with-one-training-example]], [[rlvr-beyond-base-model]]
     figures: figures/rollout-passrate.html
     revised: 2026-09 (generality revision)
-->

# Chapter 16 — RL 프롬프트 분포: 난이도 필터링, 도메인 breadth, 프롬프트 재사용

> **핵심 통찰.** 그룹 기준선 RL(ch-40)에서 프롬프트는 표본 그룹에 성공과 실패가 모두 있을 때만 gradient에 기여한다. 그러므로 프롬프트 세트는 인간 난이도 라벨이 아니라 측정한 pass rate로 선택한다. 공개된 2025년 레시피는 같은 cut을 서로 다른 위치에 적용한다. DAPO는 각 batch가 정확도 0과 1 사이에 엄격히 있는 그룹만 포함할 때까지 resample하고, 이 단계를 추가하면 AIME 2024 avg@32가 42 → 50이라고 보고한다([[dapo]], 표 1). Llama-Nemotron은 8개 표본에서 pass rate가 0.75 이상인 프롬프트를 버린다([[llama-nemotron]], §5.1). Olmo 3 Think 7B는 8개 rollout 중 62.5%를 초과해 푼 프롬프트를 버린다([[olmo-3]], §4.4.2). Skywork-OR1은 N = 16에서 0/N 및 N/N 프롬프트를 제거하며, 그 결과 7B에서는 수학 pool의 46.2%, 32B에서는 37.3%가 남는다고 보고한다([[skywork-or1-rl-data]], §6.2).
>
> **지침.** RL 알고리즘이 그룹 기준선이면 학습하려는 checkpoint로, 학습에 쓸 sampling 설정에서 pass rate를 측정하고 pass rate가 1인 프롬프트를 제거한다. rollout 예산이 구속 조건일 때만 pass-rate-0 프롬프트를 제거한다. 오늘의 0은 나중에 0.2가 될 수 있기 때문이다. rollout throughput이 병목이 아니면 고정 offline cut보다 online filtering(DAPO dynamic sampling, Olmo 3 active sampling)을 선호한다. offline cut은 첫 업데이트 뒤 더 이상 policy가 아닌 checkpoint에 대해 측정되기 때문이다. 필터링한 pool을 복수 도메인에 쓰면, 일방향 cut이 모델이 이미 푸는 도메인을 제거하므로 필터링 뒤 도메인별 점유율을 다시 확인한다. gradient 단계 사이에 표본을 재사용하면 optimizer 단계 기준으로 얼마나 오래됐는지 세고 importance ratio를 기록한다. Skywork-OR1은 rollout batch 하나를 2회 또는 4회 재사용하면 단일 on-policy 단계보다 entropy collapse가 빨라지고 최종 test 성능이 낮아진다고 보고한다([[skywork-or1-rl-data]], §4.4).

---

## 범용 모델에서 이 장이 중요한 이유

파이프라인 위치는 pretraining → mid-training → SFT → preference optimization → **RL** → evaluation이다. RL이 시작할 때 policy 능력은 앞 단계가 정하며, RL은 policy가 이미 표본 추출할 수 있는 출력의 확률을 바꾼다. RL 단계가 닿는 범위는 프롬프트 세트로 제한된다. 절대 표본 추출하지 않는 프롬프트는 gradient를 내지 않으며, 프롬프트 세트에 없는 도메인은 직접 신호를 전혀 받지 않는다.

일반적으로 유능한 모델에는 두 실패 방향이 중요하다.

1. **구성에 의한 좁아짐.** 경쟁 수학으로 제한한 프롬프트 세트는 측정 가능한 개선이 경쟁 수학에 있는 policy를 만든다. [[magistral]]은 같은 checkpoint에서 24B 모델을 수학 전용 및 코드 전용 RL로 학습하고(AIME'24 32.2, LiveCodeBench v5 22.7), 수학 전용 RL은 두 벤치마크에서 62.5 / 38.3, 코드 전용 RL은 49.7 / 42.7에 도달한다고 보고한다(§6.1, 표 5). 두 실행 모두 수치를 움직이지만 더 큰 변화는 학습한 도메인에 있다.
2. **필터링에 의한 좁아짐.** 난이도 필터링은 도메인 중립적이지 않다. 모델이 이미 푸는 프롬프트를 제거하면 모델이 잘하는 도메인에서 더 많은 프롬프트를 제거한다. [[rlvr-tulu3]]은 405B에서 이를 보인다. SFT와 DPO 뒤 포화했으므로 GSM8K를 RLVR 혼합물에서 제외하고, 도움이 되지 않는 IFEval 프롬프트도 제외하여 MATH만 RLVR 학습 도메인으로 남겼다(§8.1).

이 단계의 측정 질문은 보고한 이득이 프롬프트에서 왔는지 평가 세트에서 왔는지다. [[reasoning-or-memorization-rl-contamination]]은 Qwen2.5-Math-7B에 MATH-500 문제의 처음 60%를 주면 나머지 40%를 exact match 54.60으로 재구성하고 그 부분 문제의 53.6%에 맞게 답하는 반면, Llama3.1-8B는 같은 프로토콜에서 3.80과 훨씬 낮은 답 정확도를 보임을 보인다(§4.2, 표 2; §1). 따라서 Qwen2.5 base를 쓴 MATH-500만의 RL 데이터 주장은 프롬프트 세트 설계의 근거가 아니다.

---

## §1 RL 프롬프트 세트란 무엇인가

**정의.** RL 프롬프트 세트는 `(x, g)` 튜플의 모음이다. `x`는 질의이고 `g`는 표본 응답 `y`를 scalar reward로 매핑하는 채점 메커니즘이다. target response는 없다. gradient는 표본 응답에서 오므로 프롬프트의 기여는 현재 policy가 그 프롬프트에 대해 무엇을 표본 추출하는지에 달렸다.

**공개된 레시피의 grader.**

| Grader | 공개 레시피의 예 | 위치 |
|---|---|---|
| 문자열/기호 답 일치 | Tülu 3 RLVR 수학: 최종 답을 추출해 reference와 비교 | [[rlvr-tulu3]] §6.1 |
| 제약 조건 검사기 | Tülu 3 IF verifiable: 제약 조건 template마다 검증 함수 하나 | [[rlvr-tulu3]] §6.1 |
| Unit test | DeepSeek-R1 코딩: 경쟁 문제 17k개와 실패 unit test가 있는 GitHub bug-fixing 과제 8k개 | [[deepseek-r1]] B.3.1 |
| 객관식 선택지 일치 | DeepSeek-R1 STEM: 선택지 4~8개 질문 22k개, 일치한 선택지의 binary reward | [[deepseek-r1]] B.3.1 |
| reference에 대한 judge 모델 | Llama-Nemotron: Llama-3.3-70B-Instruct가 예측이 ground-truth 답과 일치하는지 판정 | [[llama-nemotron]] §5.1 |
| Reward 모델 | DeepSeek-R1 일반: helpfulness 질문 66k개, harmlessness 질문 12,000개, reward 모델 2개 | [[deepseek-r1]] B.3.1 |

**공개된 프롬프트 세트 크기.** 이는 각 보고서가 쓰는 단위 그대로 인쇄된 크기다.

| 릴리스 | 프롬프트 | 보고된 구성 | 위치 |
|---|---|---|---|
| Tülu 3 8B/70B RLVR | 29,946 | GSM8K train 7,473; MATH train 7,500; IF verifiable 14,973 | [[rlvr-tulu3]] 표 22 |
| DeepSeek-R1 | 수학 26K, 코드 17K(+bug fixing 8k), STEM 22K, 논리 15K, 일반 66K | 데이터 유형별 행 하나, 프롬프트 수만 | [[deepseek-r1]] 표 4, B.3.1 |
| Olmo 3 Think RL(Dolci-Think-RL) | 104,869 | precise IF 30,186; 5개 출처 수학 30,186; 코드 23,110; 일반 chat 21,387 | [[olmo-3]] 표 20 |
| Skywork-OR1-RL-Data | 수학 105,055, 코드 14,057 | HF dataset card split 크기; 보고서는 약 수학 105K, 코드 13.7K(LeetCode 2.7K, TACO 11K)라고 말함 | [[skywork-or1-rl-data]] 카드 + §6.1 |
| DeepMath-103K | 103K | 원본 2,869K → decontamination → 난이도 ≥ 5 → 답 검증 = 95K, SimpleRL에서 8K 추가 | [[deepmath-103k]] §3, 그림 7 |
| Magistral 수학 | 38k | 초기 699k → format filtering 후 501k → difficulty filtering 후 38k | [[magistral]] §4.1, 표 1 |

mechanics 이전에 이름 붙일 만한 속성 두 가지가 있다.

**(i) grader는 질문만큼 프롬프트를 정의한다.** [[kimi-k1-5]]는 정답이 잘못된 추론으로도 도달될 수 있으므로 객관식, 참/거짓, 증명 기반 질문을 RL 프롬프트 세트에서 제거하며, chain-of-thought 없는 모델이 N = 8번 안에 답을 추측하면 그 프롬프트도 제거한다(§2.1). [[deepseek-r1]]은 일치 선택지에 binary reward를 주는 객관식 STEM 질문 22k개를 유지하고, 같은 검증 가능성 이유로 수학 증명은 제외한다(B.3.1). 두 보고서는 객관식에 동의하지 않지만 검증 불가능한 형식은 들어오지 않는다는 데 동의한다. 객관식은 열린 질문으로 취급한다. 어느 보고서도 이를 절제하지 않는다.

**(ii) 프롬프트 가치는 (프롬프트, 현재 policy) 쌍의 속성이다.** 사람 또는 judge가 쓴 난이도 라벨은 gradient를 예측하지 못한다. [[kimi-k1-5]]는 SFT 모델이 비교적 높은 sampling temperature에서 각 프롬프트에 열 번 답할 때의 pass rate로 난이도를 정의한다(§2.1). [[llama-nemotron]], [[olmo-3]], [[skywork-or1-rl-data]], [[magistral]]도 각각의 checkpoint로 같은 양을 측정한다. 이유는 §2에 있다.

---

## §2 Pass rate가 프롬프트가 gradient를 내는지 결정한다

**문제 진술.** 그룹 기준선(ch-40)에서 trainer는 프롬프트 `x`에 대해 `G`개 응답을 표본 추출하고 reward `r_1..r_G`를 계산하며, 응답 `i`의 advantage를 `A_i = r_i − r̄`로 설정한다(선택적으로 그룹 표준편차로 나눔). 여기서 `r̄ = (1/G) Σ_i r_i`다. `G`개 reward가 모두 같으면 모든 `i`에서 `A_i = 0`이며, `G`개 rollout을 소비했어도 프롬프트는 policy-gradient 항에 기여하지 않는다.

**binary reward의 메커니즘.** `p`를 현재 policy가 학습 sampling 설정에서 `x`를 풀 확률, `k`를 크기 `G` 그룹의 성공 수라 하자.

```
A_i = r_i − k/G
Σ_i |A_i| = k(1 − k/G) + (G − k)(k/G) = 2k(G − k)/G
P(zero-variance group) = p^G + (1 − p)^G     (독립 표본)
```

`A_i`: 응답 i의 advantage; `r_i ∈ {0,1}`: verifier 출력; `k`: 그룹의 성공 수; `G`: 그룹 크기; `p`: 표본당 성공 확률.

**계산 예시(G = 8).** 절대 advantage 질량 `2k(G−k)/G`는 k = 4에서 4.00, k = 2 또는 6에서 3.00, k = 1 또는 7에서 1.75, k = 0 또는 8에서 0이다. 그룹이 낭비될 확률 `p^8 + (1−p)^8`은 p = 0.5에서 0.0078, p = 0.75에서 0.100, p = 0.9에서 0.430, p = 0.95에서 0.663이다. 따라서 policy가 90% 확률로 푸는 프롬프트는 100개 그룹 중 약 43개에서 gradient를 내지 않으며, 남은 그룹의 advantage 질량은 한두 응답에 집중된다.

공개된 cut point를 같은 축에 그린 이 계산의 interactive 버전은 [figures/rollout-passrate.html](figures/rollout-passrate.html)에 있다. 그룹 크기와 pass rate를 움직여 wasted-group 확률, advantage 질량, 2도메인 pool에 대한 일방향 cut 효과를 읽어라.

**훈련 중 낭비 비율이 증가한다는 근거.** [[dapo]]는 정확도가 1인 프롬프트 수가 학습 내내 증가해 batch당 유효 프롬프트 수가 계속 감소한다고 보고한다(§3.2, 그림 3b). 따라서 cut은 한 번이 아니라 계속 적용한다.

**조건과 한계.** zero-variance 공식은 독립 표본과 binary verifier를 가정한다. 연속 reward([0,1]의 LM judge 점수, 비례적인 test-pass 비율)에서는 정확한 동점이 더 드물고 zero-gradient 경우가 low-variance 경우로 바뀐다. Olmo 3는 filter를 “rewards are all identical”이라고 진술하며 둘을 포괄한다([[olmo-3]] §4.4.1).

**범용 모델에 대한 함의.** filter가 최대화하는 것은 capability breadth가 아니라 rollout당 gradient다. §3과 §4의 모든 내용은 두 목적이 다르다는 사실에서 따른다.

---

## §3 난이도 필터링: offline cut, online cut, 그리고 각각의 비용

### 3.1 Offline filtering(학습 전 한 번 측정)

보고서가 서술하는 순서의 메커니즘:

1. RL을 시작할 checkpoint를 선택한다.
2. 학습 temperature에서 프롬프트당 `N`개 응답을 표본 추출한다.
3. pass rate를 계산한다.
4. cut을 적용하고 나머지를 유지한다.

공개된 사례:

| 릴리스 | 측정을 위한 sampling | Cut | 위치 |
|---|---|---|---|
| Kimi k1.5 | SFT 모델, 답 10개, “상대적으로 높은” temperature | curriculum 및 사소한 사례 prefiltering의 난이도 라벨로 pass rate 사용 | [[kimi-k1-5]] §2.1 |
| Llama-Nemotron LN-Ultra | LN-Super 응답 8개 | pass rate ≥ 0.75 폐기 | [[llama-nemotron]] §5.1 |
| Olmo 3 Think 7B | DPO checkpoint의 rollout 8개, temperature 1.0, top-p 1.0, RL sampling과 일치 | pass rate > 62.5% 폐기(즉, 8개 중 5개 초과) | [[olmo-3]] §4.4.2 |
| Skywork-OR1 | 수학 rollout 16개, 코드 8개, temperature 1.0, 최대 32K 토큰 | 0/N 및 N/N 폐기 | [[skywork-or1-rl-data]] §6.2 |
| Magistral | Mistral Large 2 표본 16개, 뒤이어 RL 학습 24B grader 표본 16개 | 두 pass에서 한 번도 풀리지 않거나 높은 성공률인 문제 제거; 표본 다수가 reference와 다른 답에 동의한 문제도 제거 | [[magistral]] §4.1 |
| DeepMath-103K | 문제별 GPT-4o 난이도 평정 6회를 질의해 평균 | level ≥ 5 유지(Omni-MATH/AoPS 척도) | [[deepmath-103k]] §3 |

보고된 유지율에서 Skywork-OR1은 DeepSeek-R1-Distill-Qwen-7B의 수학 프롬프트 중 21.4%가 전부 오답, 32.4%가 전부 정답이어서 46.2%가 남는다고 한다. 32B에서는 20.7%와 42.0%로 37.3%가 남으며, 코드 유지율은 48%와 37.6%다([[skywork-or1-rl-data]] §6.2). 더 강한 모델에 대해 같은 pool을 필터링하면 더 적은 프롬프트와 다른 부분집합이 남는다. filter는 프롬프트가 아니라 모델의 측정이다.

offline cut의 두 비용:

- **첫 업데이트 뒤 stale해진다.** pass rate는 trainer가 즉시 떠나는 checkpoint로 측정했다. Olmo 3는 7B에 offline cut을 쓰고 32B에는 active sampling을 쓴다([[olmo-3]] §4.4.2).
- **약한 grader는 실제로 유용한 프롬프트를 버린다.** [[magistral]]은 Mistral Large 2의 단일 pass가 풀기 어렵지만 풀 수 있는 많은 문제를 풀 수 없는 것으로 버렸을 것이므로 두 번째 pass에 RL 학습 24B grader를 쓴다고 주장한다(§4.1). 저자가 절제 없이 진술한 주장이다.

### 3.2 Online filtering(학습 loop 내부에서 측정)

[[dapo]] dynamic sampling은 over-sample하고, 정확도 0 또는 1인 모든 그룹을 버리며, 정확도가 엄격히 0과 1 사이인 그룹으로 batch가 찰 때까지 표본 추출한다(§3.2, 식 11, 알고리즘 1의 6–8행). 목적함수의 제약은 `s.t. 0 < |{o_i : is_equivalent(a, o_i)}| < G`다.

verl 구현은 비용을 명시한다([[verl-dapo-recipe]]):

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc          # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10   # Non-positive values mean no upper limit
```

generation은 학습 batch의 세 배로 실행하며, 충분한 자격 그룹이 생기거나 `max_num_gen_batches`에 도달할 때까지 반복한다.

[[olmo-3]]은 oversample-and-discard를 연속 resampling으로 대체한다. asynchronous actor가 completion을 계속 가져오고 prompt queue를 채워 batch가 목표 수 non-zero-gradient completion을 가질 때까지 계속한다(§4.4.3). 보고서는 이것이 일관되게 가득 찬 batch를 유지하고 RL-Zero 수학 실행의 loss variance를 낮춘다고 말한다(§6.2, 그림 26). arm당 한 실행 비교다.

[[skywork-or1-rl-data]]는 둘을 결합한다. base-model 정확도 1 또는 0의 프롬프트는 학습 전 제거하고, actor가 이전 stage에서 완전히 푼 프롬프트는 각 stage 경계에서 제거하며, batch에서는 non-zero advantage가 적어도 하나인 그룹만 유지한다(§3.1).

**online cut의 근거.** `avg@k`는 문제당 `k` 독립 표본의 평균 정확도이며 단일 수치로 보고한다. [[dapo]] 표 1은 기법을 누적 추가한 Qwen2.5-32B base의 AIME 2024 avg@32를 naive GRPO 30, + overlong filtering 36, + clip-higher 38, + soft overlong punishment 41, + token-level loss 42, + dynamic sampling 50으로 보고하며 DeepSeek-R1-Zero-Qwen-32B는 47이다. 저자들은 더 적은 학습 단계가 필요해 추가 sampling이 총 시간을 유의미하게 늘리지 않는다고 말한다(§4.2, 그림 6). verl 재현 표는 16×8×H800에서 DAPO 52%, dynamic sampling 없음 50%를 보고한다([[verl-dapo-recipe]], Reproduction Runs). 구성당 단일 실행이고, 점진적 절제이므로 dynamic-sampling 행은 모든 이전 행 위에서 측정했다.

### 3.3 Cut이 하지 않는 일

공개 보고서 중 threshold 자체를 절제한 것은 없다. [[llama-nemotron]]의 0.75, [[olmo-3]]의 62.5%, [[dapo]]의 엄격한 0-or-1 cut은 세 가지 다른 답이고, 통제된 설정에서 서로 비교되지 않았다(열린 질문). [[dapo]], [[olmo-3]], [[skywork-or1-rl-data]], [[magistral]]에 공통인 것은 분포의 해결된 끝을 제거하는 일이다. 해결되지 않은 끝 제거는 보편적이지 않다. [[llama-nemotron]]은 쉬운 쪽만 자르고 pass-rate-0 프롬프트는 pool에 남긴다(§5.1).

---

## §4 도메인 구성과 프롬프트 세트의 breadth

**공개된 구성.** [[deepseek-r1]]은 추론 RL 데이터(수학, 코드, STEM, 논리)를 일반 RL 데이터(helpfulness 66k, harmlessness 12,000)와 분리하고, 1,700단계 두 번째 RL stage의 최종 400단계에서만 일반 instruction 데이터와 preference 기반 reward를 넣는다. 모델 기반 preference reward 아래에서 단계가 너무 많으면 reward hacking이 생길 수 있다고 말한다(§3.2.2). [[olmo-3]]은 precise instruction following, 수학, 코드, 일반 chat을 섞고 도메인별 대략 같은 양, 수학과 instruction following은 약간 더 많이 쓴다고 보고한다(§4.4.2). [[kimi-k1-5]]는 STEM·코딩·일반 추론의 다양한 coverage, 균형 난이도, 정확한 evaluability를 큐레이션 속성으로 밝히고 subject area 균형을 위한 domain tagging system을 쓴다(§2.1). 혼합물 절제 수치는 없다.

**혼합의 근거.** [[multi-domain-rlvr-data-centric]]은 Qwen2.5-7B-Base를 수학·코드·퍼즐 데이터로 GRPO 학습한다(수학 20k = DeepScaleR 10k + CountDown 10k; 코드 12k = CodeR1-12k; 퍼즐 7.8k = Knights-and-Knaves 5.4k + Logic Puzzle Baron 2.4k, 표 1). 각 실행 뒤 세 도메인을 모두 평가한다(표 9, base 행: 수학 22.48 / 코드 67.46 / 퍼즐 9.07 / 전체 31.50).

| 학습 데이터 | 수학 평균 | 코드 평균 | 퍼즐 평균 | 전체 |
|---|---|---|---|---|
| 수학 | 47.48 | 64.23 | 22.42 | 45.11 |
| 퍼즐 | 29.47 | 71.35 | 61.98 | 50.72 |
| 코드 | 19.17 | 73.95 | 22.55 | 35.78 |
| 수학 + 퍼즐 | 49.72 | 44.90 | 49.78 | 48.36 |
| 퍼즐 + 코드 | 32.06 | 74.88 | 55.15 | 50.89 |
| 수학 + 코드 | 47.22 | 75.06 | 25.34 | 48.92 |
| 수학 + 코드 + 퍼즐 | 49.75 | 73.63 | 49.73 | 56.57 |

세 도메인 혼합물은 최고 전체 평균 56.57을 가지며 Math + Puzzle의 코드 붕괴(기준 67.46에 비해 44.90)를 피하지만, 퍼즐 전용 실행은 여전히 최고 퍼즐 모델(61.98)이다. 결과(단일 연구)이며 한 모델 family, 구성당 한 seed다. transfer와 interference의 메커니즘은 ch-44b가 다룬다. 이 표가 보이는 점은 프롬프트 세트 구성이 양방향의 측정 가능한 cross-domain 결과를 갖는 결정이라는 것이다.

**필터링은 구성을 바꾼다.** 혼합 pool의 pass-rate cut은 시작 checkpoint가 이미 처리하는 도메인을 더 많이 제거한다. 반올림 계산 예시에서 수학 10,000개, instruction-following 10,000개 중 수학 60%, IF 20%가 cut 위라면 수학 4,000개, IF 8,000개가 남아 수학 점유율은 선택 없이 50%에서 33%가 된다. 공개 사례는 [[rlvr-tulu3]] §8.1이다. 405B에서 SFT와 DPO가 포화시킨 GSM8K와 초기 실행에서 도움이 안 된 IFEval을 제거해 MATH만 남겼다. [[olmo-3]]은 의도적 반대 조정을 보고한다. offline filtering 뒤 모델이 어려워한 OMEGA 하위 과제 8개를 50% downsample했다(§4.4.2, 각주 39).

**함의.** 필터링 전이 아니라 후의 도메인별 수를 측정하고, 필터링 후 혼합물을 실제로 학습하는 혼합물로 취급한다.

---

## §5 프롬프트 공간에서의 curriculum

**정의.** curriculum은 프롬프트 분포를 균등 표본 추출하는 대신 학습 시간에 따라 순서를 정한다.

공개된 형태:

1. **추적 성공률의 prioritized sampling.** [[kimi-k1-5]]는 RL 동안 각 문제의 성공률 `s_i`를 추적하고 문제 `i`를 `1 − s_i`에 비례해 표본 추출한다(§2.3.4). 별도 재측정 pass 없이 sampling 분포가 policy를 따른다.
2. **재측정이 있는 단계적 난이도.** [[kimi-k1-5]]는 전체 혼합물로 warm up한 뒤 어려운 질문에 집중하며 uniform sampling보다 높은 정확도를 보고한다(§3.5, 그림 9; 두 곡선 수치는 인쇄하지 않음). [[magistral]]은 성능이 오르면 이전 stage가 필터링했던 문제를 되넣고 완전히 푼 문제를 제거해 더 어려운 split을 만든다(§5.2).
3. **분포 형상화 batching.** [[llama-nemotron]]은 미리 계산한 pass rate의 Gaussian target distribution으로 batch를 만들고, 중심을 batch 전반에 높은 pass rate에서 낮은 pass rate로 옮기며, 남는 capacity는 남은 pool이 가장 큰 pass-rate bin에서 채운다(§5.1). 효과는 200단계 GPQA-Diamond Avg@4에서 curriculum 대 random ordering이다(그림 6).
4. **reference-policy refresh가 있는 curriculum.** [[multi-domain-rlvr-data-centric]]은 Knights-and-Knaves 퍼즐을 175단계씩 6 난이도 stage로 학습해 standard curriculum 97.29, mixed training 94.29의 최종 평균을 보고한다. stage 경계마다 reference model을 현재 actor로 바꾸고 optimizer state를 reset하면 99.71이 된다(§6, 그림 6). 결과(단일 연구), 데이터셋 하나다.

**좁아짐의 위험.** 한 도메인의 가장 어려운 slice에서 끝나는 curriculum은 좁은 분포에서 학습을 끝내고 마지막 단계가 최종 weight를 정한다. 두 측정이 위험을 제한한다. `pass@k`는 `k`개 표본 응답 중 적어도 하나가 맞는 문제 비율이므로 `pass@1`은 한 번 시도에서 맞는 빈도, 큰 `k`의 `pass@k`는 policy가 아직 전혀 도달할 수 있는 문제 수를 측정한다. [[rlvr-beyond-base-model]]은 RLVR 모델이 시험한 수학·코드·시각 추론에서 base보다 높은 pass@1이나, 큰 k에서는 base가 pass@k에서 같거나 더 높음을 보고하고, 저자는 RLVR이 base가 이미 표본할 수 있던 solution path 확률을 높인 것으로 해석한다. [[olmo-3]]은 다른 설정에서 반대 부호를 보인다. Olmo 3 Base에서 수학 RL-Zero를 학습하면 AIME 2024·2025 pass@32가 pass@1과 함께 좋아지며, 보고서는 diversity를 유지하고 초기 능력 너머로 밀었다고 읽는다(§6.2, 그림 24). (그림 20 오른쪽은 RL 전후가 아니라 DPO와 SFT checkpoint 비교이므로 RL 근거가 아니다.) 두 결과는 다른 base와 프롬프트 세트의 단일 설정이다. 자신의 curriculum에서는 held-out 세트의 전후 큰 k pass@k를 측정한다.

---

---

## §6 프롬프트와 trajectory 재사용

이 절은 흔히 반복되는 두 주장을 바로잡는다. 하나는 framework가 프롬프트만 replay하고 trajectory는 절대 replay하지 않는다는 주장이고, 다른 하나는 sequence 수준 importance ratio의 기댓값이 `exp(Tσ²/2)`라는 주장이다.

### 6.1 TRL의 replay buffer가 실제로 저장하는 것

실험용 trainer에 관한 TRL 문서는 이것이 “GRPO로 모델을 학습하되, 표준편차가 0인 group(및 그에 대응하는 completion)을 prior batch에서 모델 학습에 사용되었고 reward와 표준편차가 높은 group으로 교체한다”고 설명한다([[replay-buffer-rlhf]], `docs/source/grpo_with_replay_buffer.md`). config field는 buffer가 “가장 높은 advantage score와 group당 variance를 가진 rollout을 저장한다”고 명시한다.

`trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py`의 cached copy(2026-09-14 fetch, `excerpts/replay-buffer-rlhf.md`에서 검증한 발췌)에는 다음이 있다.

```python
buffered_output = {
    "prompt_ids": group_prompt_ids,
    "completion_ids": group_completion_ids,
    "advantages": group_advantages[group_idx].tolist(),
    "prompt_mask": group_prompt_mask,
    "completion_mask": group_completion_mask,
}
...
replay_buffer_scores = (group_advantages.abs() * group_std_rewards).sum(dim=-1)[groups_with_variance]
self.replay_buffer.add(replay_buffer_scores.tolist(), buffered_outputs)
```

새 batch의 group이 reward 표준편차 0일 때는 다음과 같이 한다.

```python
prompt_ids[idx_range] = sampled_data["prompt_ids"][i]
completion_ids[idx_range] = sampled_data["completion_ids"][i]
group_advantages[group_idx] = sampled_data["advantages"][i]
```

따라서 buffer는 completion과 그 advantage를 저장하며, stale group 하나가 새 zero-variance group 하나를 통째로 대체한다. `old_per_token_logps`와 `ref_per_token_logps`도 함께 저장하지만, 해당 step에서 그것들을 계산했을 때만 저장한다. trainer는 generation step과 optimization step이 어긋났거나 vLLM importance-sampling correction을 켠 경우에만 `old_per_token_logps`를 계산하고, 그 밖에는 `None`으로 둔다. 상위 `GRPOTrainer` loss는 없는 값을 `per_token_logps.detach()`로 대체한다([[trl-grpo]], commit a08e713). 그러면 매 token의 ratio는 1이 된다. 이 구성에서는 replay된 completion이 importance correction 없이 그대로 gradient에 들어간다.

**buffer의 scoring function은 §2 기준으로 환원된다.** 기본값 `scale_rewards="group"`에서는 advantage가 buffer에 도달하기 전에 group 표준편차로 나뉜다. 따라서 `Σ_i |A_i| · σ_group ≈ Σ_i |r_i − r̄| = 2k(G − k)/G`다. G = 8이면 k = 4에서 score는 4.00이고, k = 1 또는 7에서는 1.75다. 즉 buffer는 pass rate가 0.5에 가장 가까운 group을 보관한다.

### 6.2 importance-sampling ratio를 정확히 쓰기

`π_old`에서 표본 추출한 response `y`에 대해 `ρ(y) = π_θ(y|x) / π_old(y|x)`라 하자. 그러면

```
E_{y∼π_old}[ρ(y)] = Σ_y π_old(y) · π_θ(y)/π_old(y) = Σ_y π_θ(y) = 1
```

이며, drift의 양과 무관하게 정확히 성립한다. 평균은 stale 정도에 관한 정보를 주지 않는다. 정보를 주는 것은 분산이다.

```
Var_{π_old}[ρ] = E_{π_old}[ρ²] − 1 = χ²(π_θ ‖ π_old)
```

`χ²`는 chi-square divergence다. token-factorized model에서 `T`개 token에 대해 `log ρ = Σ_t log ρ_t`라고 쓰자. token별 log-ratio를 서로 독립인 `N(μ, σ²)`로 모델링하면, `E[ρ] = 1` 제약은 `μ = −σ²/2`를 강제한다. 그러면

```
log ρ ~ N(−Tσ²/2, Tσ²)
E[ρ] = 1,  E[ρ²] = exp(Tσ²),  Var[ρ] = exp(Tσ²) − 1
ESS / n ≈ 1 / E[ρ²] = exp(−Tσ²)      (Kish effective sample size)
median ρ = exp(−Tσ²/2)
```

`T`는 response 길이(token 수), `σ²`는 token별 log-ratio variance, `ESS/n`은 batch가 독립적인 on-policy sample처럼 행동하는 비율이다.

**계산 예시.** `σ = 0.01`, `T = 1,000`이면 `Tσ² = 0.1`, variance는 0.105, ESS는 약 `0.90 n`, median ratio는 0.95다. `σ = 0.01`, `T = 10,000`이면 `Tσ² = 1`, variance는 1.72, ESS는 약 `0.37 n`, median ratio는 0.61이다. `σ = 0.03`, `T = 10,000`이면 `Tσ² = 9`, variance는 8,102, ESS는 약 `1.2 × 10⁻⁴ n`, median ratio는 0.011이다. 평균 weight는 여전히 정확히 1이지만, 소수 sequence가 그 평균을 떠받친다. 따라서 stale해지며 나빠지는 양은 ratio의 평균이 아니라 ratio의 퍼짐과 effective sample size다. 이 Gaussian-independence model은 설명을 위한 것이다(Interpretation). 어떤 source도 token별 log-ratio variance를 이 방식으로 측정하지 않았다.

실무상 두 결과가 따른다. 첫째, PPO형 objective는 ratio를 token별로 clip한다(또는 length-normalized form에서는 sequence별로 clip한다). 따라서 estimator는 유계가 되지만 biased해지고, clipped token 비율이 클수록 bias가 커진다. 둘째, 기록할 diagnostic은 평균 ratio가 아니다. 평균은 구성상 1이다. 대신 퍼짐을 기록해야 한다. vLLM correction을 켜면 TRL은 `sampling/importance_sampling_ratio/{min,mean,max}`를 기록하고, 모든 실행에서 `frac_reward_zero_std`를 기록한다([[replay-buffer-rlhf]] extract, `_generate_and_score_completions`).

### 6.3 재사용에 대해 측정값이 말하는 것

[[skywork-or1-rl-data]]는 rollout batch 하나에 수행하는 SGD step 수를 절제한다(DeepSeek-R1-Distill-Qwen-7B, rollout batch `D_R` = 64, group size 16, 16K context, temperature 1.0, KL loss 없음). `N_SGD = (D_R/D_T) · N_reuse`일 때 `N_SGD ∈ {2,4}`인 구성은 rollout batch를 더 작은 minibatch로 나누어서든 rollout buffer를 2회 또는 4회 순회해서든 도달한다. 이 구성들은 몇 step 안에 entropy가 작은 값으로 감소하고 test performance가 더 이상 개선되지 않는다. 반대로 on-policy 구성 `(1,64,64,1)`은 더 천천히 개선하지만 마지막 성능은 더 높다(§4.4, 그림 16–17). 보고서는 추가 control로 원인을 분리한다. 같은 작은 minibatch 크기의 on-policy 실행은 조기 entropy collapse를 보이지 않는다. 그러므로 저자들은 저하 원인을 minibatch 크기가 아니라 reuse가 도입한 off-policy data에 귀속한다(§4.4, Ablation 7). `N_SGD` = 4에서 rollout batch를 64에서 256으로 키워도 collapse를 막지 못했다(§4.4, 그림 18).

[[olmo-3]]은 infrastructure 수준에서 반대 접근을 취한다. generation을 멈추지 않고 actor에 weight를 push한다. 따라서 in-flight sequence는 더 새로운 weight 아래에서 계속 생성된다. 명시한 목적은 sample이 얼마나 off-policy인지 최소화하는 것이다(§4.4.3).

[[kimi-k1-5]]는 trajectory *segment*를 재사용한다. iteration당 output token budget을 고정한 채 미완료 trajectory를 replay buffer에 저장하고 다음 iteration에서 이어서 생성한다. 보고서는 현재 segment만 on-policy computation을 요구하며, 일부 segment는 loss 계산에서 제외할 수 있다고 명시한다(§2.6.2). 하지만 재사용 segment가 gradient에 어떻게 들어가는지는 말하지 않는다. 따라서 정확한 요약은 이렇다. partial rollout은 long-context throughput 기법이며, gradient 처리 방식은 일부만 공개되어 있다.

**권고.** trainer가 synchronous이고 generation이 병목이면 저장한 response보다 *prompt*를 재사용해 새 response를 다시 표본 추출한다. 다시 표본 추출한 group에는 correction이 필요 없다. 저장한 response를 재사용한다면 그것을 optimizer step 하나 전의 것으로 제한한다(표준 `num_iterations = 1` 설정). 생성 policy의 log-probability를 저장하고 실제로 사용해 ratio가 1이 아닌 실제 ratio가 되게 하며, ratio의 퍼짐과 entropy를 기록한다. 몇 step보다 오래된 response만 쓸 수 있다면 Skywork-OR1이 보고한 entropy-collapse pattern을 예상하고, 채택 전에 on-policy arm과 비교한다.

---

## §7 프롬프트 세트 주장 평가에서의 함정

1. **오염된 평가 세트.** [[reasoning-or-memorization-rl-contamination]]은 partial-prompt completion과 partial-prompt answer accuracy를 측정한다. 전자는 모델에 문제의 앞부분(prefix)을 주고 나머지를 완성하게 하는 측정이고, 후자는 같은 종류의 일부 프롬프트에 대한 최종 답 정확도다. MATH-500에서 문제 prefix의 60%를 주었을 때 Qwen2.5-Math-7B의 exact-match(EM) completion은 54.60, Qwen2.5-7B는 21.20이며, Llama3.1-8B는 3.80이다. 반면 LiveMathBench에서는 세 모델 모두 completion EM이 0.00이다(§4.2, 표 2). 모델 출시 뒤에 생성한 산술 문제인 RandomCalculation에서는 잘못된 보상을 주는 실험도 수행했다. 올바른 reward만 성능을 개선했고, 무작위 reward와 반전된 reward는 성능을 개선하지 못했다(§4.3).
2. **오염된 학습 pool.** [[deepmath-103k]]는 filtering 전 자기 raw pool의 오염을 측정한다. raw pool에서 발견된 benchmark test sample의 비율로 정의했을 때 Omni-MATH는 92.6%, AIME24와 AMC23은 각각 90.0%, MATH-500은 76.6%, Minerva Math는 35.7%, OlympiadBench는 33.6%다(§2, 그림 3). decontamination 절차는 먼저 embedding retrieval로 각 benchmark 항목의 가장 가까운 항목 상위 5개를 찾고, 이어 LLM judge가 paraphrase를 버리는 방식이다(§3, Stage 2).
3. **protocol을 보고하는 decontamination.** [[deepseek-r1]]은 post-training용 수학 SFT data와 RL prompt를 2023년 이전 대회에서 가져왔고, pretraining과 같은 10-gram filter를 통과시켰다고 명시한다. 동시에 n-gram filtering은 paraphrase를 잡아내지 못한다는 한계도 인정한다(Supplementary, Decontamination).
4. **negative control.** [[olmo-3]]은 negative control로 base model을 RL-Zero mixture에서 random binary reward로 학습한다. 평가 세트가 이미 memorization되어 있었다면, 실제 정답과 무관한 spurious reward도 그 평가 성능을 끌어낼 것이라는 논리다(§6.2).
5. **single-prompt 결과.** [[reinforcement-learning-with-one-training-example]]는 RLVR을 단 하나의 example로 수행해 Qwen2.5-Math-1.5B의 MATH-500 성능을 36.0%에서 73.6%로 올렸다고 보고한다. example은 그 example에 대한 과거 training accuracy의 variance를 기준으로 선택했다(§1, §3.2). 이를 항목 1과 함께 읽어야 한다. base model은 Qwen2.5-Math checkpoint이고 benchmark는 MATH-500이므로, 이 결과는 오염된 base가 이미 가진 것을 RL이 어느 정도 끌어낼 수 있는지를 보여준다. 프롬프트 세트 크기가 중요하지 않다는 증명은 아니다.

**요구 사항.** 프롬프트 세트 설계에 관한 주장은 base model 이후에 공개된 benchmark를 적어도 하나 포함해야 하며, 가능하면 서로 다른 base-model family 두 개에서 확인해야 한다.

---

## 음성 샘플과 음성 피드백

여기서 적용하는 “negative(음성)”의 뜻은 무엇인가(저술 표준 §6.1). 이 장은 **(1) 음의 한계 가치(negative marginal value)**, 즉 아무것도 기여하지 않거나 noise를 기여하므로 버리는 prompt와 group을 다룬다. 그리고 유지한 group 내부에서 평균보다 낮은 response, 즉 **(4) gradient로서의 음성(negative as gradient)**에 쓸 원재료를 제공한다. (4)의 유도는 ch-43a에 있고, 여기서는 prompt-set 단계가 그 response들에 무엇을 하는지만 말한다.

**음성 사례는 어디에서 나오는가.** verifier, test suite, judge model, 또는 reward model이 표본 추출한 response마다 label을 붙인다. false label은 두 방향 모두에 존재한다. [[kimi-k1-5]]는 잘못된 추론으로도 정답에 도달할 수 있는 format을 제거한다(false positive, §2.1). [[magistral]]은 sample 다수가 reference와 다른 답에 합의한 문제를 제거한다. reference가 잘못되었을 가능성이 더 높다는 판단에 따른 것이며, 이 경우는 false negative다(§4.1). [[skywork-or1-rl-data]]는 model이 16개 rollout 중 한 번 이상 “정답”을 냈기 때문에 난이도 추정을 통과했지만, 실제로는 무효이거나 불완전한 문제를 발견했다(§6.3). 이 보고서들 중 어느 것도 false-label rate를 제시하지 않는다.

**현재 관행이 이 사례들을 처리하는 방식.** reward variance가 0인 group은 학습하지 않고 버린다([[dapo]] §3.2; [[olmo-3]] §4.4.1; [[skywork-or1-rl-data]] §3.1; [[magistral]] §2.1). 유지한 group 안에서는 실패 response를 negative advantage `A_i = r_i − r̄ < 0`를 통한 gradient로 사용한다.

**작동 기제.** negative advantage를 가진 response의 policy-gradient 항은 `A_i ∇ log π_θ(y_i|x)`이고 `A_i < 0`이다. 따라서 그 response의 모든 token log-probability를 낮춘다. 한 position에서 logits를 `z`, softmax probability를 `p`라 하고 방출한 token을 `y`라 하면 log-probability의 gradient는 다음과 같다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

따라서 `log p_y`를 낮추면 `y`에서 probability mass가 빠지고, 다른 token들에는 `p_j`에 비례하여 재분배된다. 즉 주로 model이 이미 차선으로 가장 가능성 높다고 보던 continuation으로 mass가 간다. 그것이 정답 continuation이라는 보장은 없다. ch-43a가 이 결과인 likelihood displacement와 squeezing을 전개한다.

**전부 틀린 group이 아무것도 가르치지 않는 이유.** 모든 `i`에 대해 `r_i = 0`이면 baseline `r̄ = 0`이 reward와 같아서 `A_i = 0`이다. 실패는 baseline에 상대적일 때만 음성이다. group에 성공이 하나도 없으면 baseline이 실패까지 내려가 signal이 사라진다. 이것이 pass rate가 0인 prompt가 “어려운 학습 data”가 아닌 이유다. policy가 가끔이라도 풀 수 있게 되기 전에는 학습 data가 전혀 아니다.

**Control.** 낮은 pass-rate prompt도 가끔 성공을 내도록 group size를 충분히 크게 둔다. Skywork-OR1은 추정 시 수학에 16 rollout, code에 8 rollout을 사용하고([[skywork-or1-rl-data]] §6.2), [[deepseek-r1]]은 RL에서 문제당 output 16개를 표본 추출하며(§3.2.1), [[llama-nemotron]]은 rollout prompt size 72에서 prompt당 16개를 표본 추출한다(§5.1). 음성이 신뢰할 수 없는 format은 penalize하지 말고 filter out한다. negative advantage가 current policy에 대해 계산되도록 reuse를 낮게 유지한다(§6.3).

**Diagnostic.** reward standard deviation이 0인 group의 비율을 기록한다(TRL은 `frac_reward_zero_std`를 기록한다). batch의 평균 pass rate만이 아니라 pass-rate histogram을 기록한다. response length와 entropy 통계를 advantage sign별로 나눈다. 그리고 coverage loss를 탐지하려고 held-out set에서 큰 `k`의 pass@k를 추적한다.

**범용성에 미치는 효과와 규모에 대한 정직성.** 어떤 prompt-set 보고서도 개선 중 얼마가 negative-advantage sample에서 왔는지 측정하지 않는다. [[kimi-k1-5]]는 잘못된 response를 penalize하지 않고 가장 좋은 표본 response만 fit하는 ReST와 자신의 objective를 비교한다. 11개 evaluation set에서 더 나은 sample complexity를 negative gradient에 귀속하지만(§3.5, Figure 10), 그 몫을 정량화하지는 않는다. 그 몫이 측정되었다고 제시해서는 안 된다.

---

## 레시피

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값에 대한 근거 |
|---|---|---|---|---|---|---|---|
| Tülu 3 8B RLVR | 8B | RL | RLVR prompt set(ground-truth label이 있는 prompt) | 29,946 = GSM8K 7,473 + MATH 7,500 + 검증 가능한 IF 14,973 | arXiv:2411.15124v5 Table 22 | 2026-09-15 확인됨 | mixture에 대한 ablation은 보고하지 않음 |
| Tülu 3 8B RLVR | 8B | RL | 총 episode(prompt 방문 수); 결합 set에 대한 유효 epoch | 100,000 episodes; ≈ 3.3 epochs | arXiv:2411.15124v5 Table 21; §6.2는 GSM8K-only ablation에서 100,000/7,473 ≈ 13 epochs라고 명시 | 도출됨(100,000 / 29,946) | ablation을 보고하지 않음 |
| Tülu 3 8B RLVR | 8B | RL | PPO: LR, effective batch, response length, temperature, K, β, warmup | 3×10⁻⁷; 224; 2,048(GSM8K-only는 1,024); 1.0; 4; β = 0.05; ω = 0.0 | arXiv:2411.15124v5 Table 21 및 caption | 2026-09-15 확인됨 | β를 [0.1, 0.05, 0.03, 0.01]에서 sweep함(§6.2) |
| Tülu 3 70B RLVR | 70B | RL | LR, episodes, batch, response length, β | 1×10⁻⁷; 400,000; 640; 2,048; β = 0.07(Table 21 caption) 대 β = 0.7(§6.4 본문) | arXiv:2411.15124v5 Table 21 caption; §6.4 | 충돌(caption과 본문; caption 값은 released checkpoint에 붙어 있음) | 이전 70B development run(§6.4) |
| Tülu 3 405B RLVR | 405B | RL | prompt set; 실행한 step | MATH train만 사용(GSM8K는 saturated되어 제거, IFEval은 도움이 되지 않아 제거); 75 steps | arXiv:2411.15124v5 §8.1 | 2026-09-15 확인됨 | 405B에서의 초기 RLVR run(§8.1) |
| DeepSeek-R1 | 보고되지 않음 | RL | 유형별 prompt 수 | math 26K, code 17K(+8k bug fixing), STEM 22K, logic 15K, general helpfulness 66K와 harmlessness 12,000 | arXiv:2501.12948v2 Table 4, B.3.1 | 2026-09-15 확인됨 | ablation을 보고하지 않음 |
| DeepSeek-R1 | 보고되지 않음 | RL | 첫 단계: LR, KL coefficient, GRPO clip ε, temperature, 문제당 sample 수, batch | 3e-6; 0.001; 10; 1.0; 16; step당 32문제 = 512 | arXiv:2501.12948v2 §3.2.1 | 2026-09-15 확인됨 | ε가 낮으면 gradient를 잘라내고, 높으면 불안정해짐(§3.2.1) |
| DeepSeek-R1 | 보고되지 않음 | RL | 둘째 단계: steps; preference reward가 들어가는 위치; temperature | 1,700 steps; 마지막 400 step에 general instruction data와 preference reward; 0.7 | arXiv:2501.12948v2 §3.2.2 | 2026-09-15 확인됨 | preference-reward step이 더 많으면 reward hacking 위험(§3.2.2, B.5) |
| Kimi k1.5 | 보고되지 않음 | RL | 난이도 추정; easy-to-hack 제거; prioritized sampling | 고온도에서 SFT model 답 10개; no-CoT model이 N = 8 안에서 guess하면 제거; sample ∝ 1 − s_i | arXiv:2501.12599v4 §2.1, §2.3.4 | 2026-09-15 확인됨 | curriculum 대 uniform sampling(§3.5, Figure 9; 숫자는 인쇄되지 않음) |
| DAPO(Qwen2.5-32B base) | 32B | RL | dynamic sampling constraint; prompt batch; prompt당 sample; ε_low/ε_high; LR | `0 < #correct < G`; 512; 16; 0.2/0.28; 20-step linear warm-up 뒤 1e-6 고정 | arXiv:2503.14476v2 §3.2, §4.1 | 2026-09-15 확인됨 | Table 1: +dynamic sampling으로 AIME24 avg@32가 42 → 50(단일 run, 누적 ablation) |
| verl의 DAPO 레시피 | 32B | RL | generation 대 training batch; retry cap | `gen_batch_size` 1536, `train_batch_size` 512, `max_num_gen_batches` 10 | verl docs `algo/dapo.md`, 최종 갱신 2025-06-19 | 2026-09-15 확인됨 | 재현 표: DAPO 52%, dynamic sampling 없이 50% |
| Llama-Nemotron LN-Ultra | 253B | RL | offline difficulty filter; rollout prompt size; prompt당 sample; global batch; rollout당 update | LN-Super response 8개로 측정한 pass rate ≥ 0.75를 discard; 72; 16; 576; 2 | arXiv:2505.00949v5 §5.1 | 2026-09-15 확인됨 | GPQA-D Avg@4에서 curriculum 대 random batching(Figure 6) |
| Olmo 3 Think 7B | 7B | RL | offline difficulty filter | temperature 1.0, top-p 1.0에서 DPO checkpoint로 rollout 8개; pass rate > 62.5%를 discard | arXiv:2512.13961v2 §4.4.2 | 2026-09-15 확인됨 | threshold ablation을 보고하지 않음 |
| Olmo 3 Think 32B | 32B | RL | filtering strategy | offline cut 대신 active sampling; 7B-filtered data를 재사용 | arXiv:2512.13961v2 §4.4.2 | 2026-09-15 확인됨 | compute와 시간 제약을 명시(§4.4.2) |
| Skywork-OR1(R1-Distill-Qwen-7B / -32B) | 7B / 32B | RL | offline difficulty estimation 및 retention | temperature 1.0, 32K에서 math 16 rollout / code 8 rollout; 0/N과 N/N discard; 7B는 46.2%/48%, 32B는 37.3%/37.6% retained | arXiv:2505.22312v2 §6.2 | 2026-09-15 확인됨 | zero-advantage group은 policy gradient를 주지 않음(§3.1) |
| Skywork-OR1 ablation baseline | 7B | RL | rollout batch, minibatch, reuse, group size, context, temperature, LR | D_R 64, D_T 64, N_reuse 1, gs 16, T 16K, τ 1.0, 1e-6 | arXiv:2505.22312v2 Table 5 | 2026-09-15 확인됨 | N_SGD ∈ {2,4}가 entropy를 collapse시키고 test score를 낮춤(§4.4) |
| Magistral Medium | 보고되지 않음 | RL | math data funnel; difficulty grader | 699k → 501k(format) → 38k(difficulty); 문제당 16 sample을 두 번(Mistral Large 2, 이후 24B RL-trained grader) | arXiv:2506.10910v1 §4.1, Table 1 | 2026-09-15 확인됨 | 논증되었으나 ablation은 보고하지 않음 |
| DeepMath-103K | dataset | RL data | funnel; difficulty rating; answer verification | raw 2,869K → decontamination → level ≥ 5 → verification = 95K, + SimpleRL 8K; GPT-4o가 6회 rating해 평균; DeepSeek-R1 solution 3개가 일치해야 함 | arXiv:2504.11456v2 §3, Figure 7 | 2026-09-15 확인됨 | raw-pool contamination rate(§2, Figure 3)가 decontamination 단계를 동기화함 |

**작은 범용 run의 출발점.** 아래의 모든 수치는 위 `확인됨` 행에서, 그 수치가 실제로 쓰인 setting과 함께 가져왔다. group size는 16을 쓴다(DeepSeek-R1은 보고되지 않은 규모에서, Llama-Nemotron은 253B에서 사용). training temperature로 8 rollout을 했을 때 pass rate가 1인 prompt는 제거한다(Olmo 3 Think 7B는 62.5% 초과를 제거하고, Llama-Nemotron은 253B에서 0.75 이상을 제거한다). rollout budget이 허용하면 pass-rate-0 prompt를 포함한 나머지 prompt는 유지한다. loop 안에는 DAPO constraint `0 < #correct < G`를 넣고, generation은 training batch의 세 배로 둔다(32B, 16×8×H800에서 verl의 `gen_batch_size` 1536 대 `train_batch_size` 512). sample은 optimizer step 하나만 지난 것으로 유지한다(7B에서 Skywork-OR1의 on-policy baseline `(1,64,64,1)`). 이 수치들은 각각의 조건에서 측정된 것이며, 어느 것도 ablation으로 확인한 최적값은 아니다.

---

## 일반화 관점

**(a) breadth를 넓히는 요인.**
- 프롬프트 세트에 여러 도메인을 섞는 것: [[multi-domain-rlvr-data-centric]]의 표 9와 §4.2에서 Qwen2.5-7B-Base의 세 도메인 혼합은 전체 평균 56.57을 기록한다. 이는 가장 좋은 두 도메인 쌍의 50.89보다 높다.
- 일반적인 도움성(general helpfulness)을 위해 검증 불가능한 slice를 유지하는 것: [[deepseek-r1]]은 reward model로 채점한 일반 프롬프트 66k개를 포함하고, 총 1,700 step 중 마지막 400 step에 이를 넣는다(B.3.1, §3.2.2). [[olmo-3]]은 수학과 코드 외에 일반 채팅 및 instruction-following 도메인도 함께 포함한다(표 20).
- 균형 잡힌 coverage를 명시적 큐레이션 목표로 두는 것: [[kimi-k1-5]] §2.1은 diverse coverage, balanced difficulty, accurate evaluability를 나열하며, 그중 첫째 목표인 다양한 coverage를 강제하기 위해 domain tag를 사용한다.

**(b) 좁아짐 또는 망각을 일으키는 요인.**
- 단일 도메인 프롬프트 세트: [[magistral]] §6.1 표 5에서 code-only RL은 LiveCodeBench v5를 22.7에서 42.7로 올린다. 그러나 수학은 32.2에서 49.7까지만 오른다. 이는 math-only RL에서의 62.5에 못 미친다.
- 혼합 안에서의 도메인 간섭: [[multi-domain-rlvr-data-centric]] 표 9에서 Math + Puzzle 조합은 code 평균을 67.46에서 44.90으로 낮춘다.
- 이미 푼 도메인을 제거하는 filtering: [[rlvr-tulu3]] §8.1에서 405B RLVR mixture는 결국 MATH만 남는 구성으로 축소된다.
- 마지막 단계가 하나의 좁은 slice로 끝나는 curriculum: §5의 경우이며, [[rlvr-beyond-base-model]]이 제시한 pass@k 주의점을 함께 적용해야 한다.

**(c) 이 단계에서 측정하는 방법.**
- filtering 뒤의 도메인별 prompt 수와, 학습 뒤의 도메인별 평가를 모두 기록한다. 목표 도메인만 평가해서는 안 된다([[multi-domain-rlvr-data-centric]]은 모든 run 뒤 세 도메인 전체를 평가한다).
- RL 전후 held-out set에서 큰 `k`의 pass@k를 측정한다([[rlvr-beyond-base-model]]). 단, 이를 당연한 개선으로 가정하면 안 된다. [[olmo-3]] §6.2 그림 24는 RL-Zero 수학 run에서 반대 방향의 부호를 보고한다. 그러므로 이는 결과가 아니라 반드시 수행할 측정이다.
- 프롬프트 세트 선택의 근거로 쓰는 모든 benchmark에서 contamination을 점검한다. partial-prompt completion과 answer accuracy([[reasoning-or-memorization-rl-contamination]] §4.2), spurious-reward negative control([[olmo-3]] §6.2), 그리고 base model 출시 뒤에 공개된 benchmark를 적어도 하나 사용한다.

---

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 징후 | 점검 방법 |
|---|---|---|
| RL을 시작하는 checkpoint가 아닌 다른 checkpoint로 filtering함 | step 1에서 batch pass-rate histogram이 filter band에서 벗어나 이동해 있음 | 처음 50개 batch의 pass-rate histogram을 기록하고 offline 측정값과 비교한다 |
| policy가 이미 푼 prompt를 계속 유지함 | 표준편차가 0인 group의 비율이 증가하고, 학습 중 effective batch size가 감소함 | step마다 `frac_reward_zero_std`를 기록한다([[trl-grpo]]). DAPO constraint 또는 Olmo 3 active sampling을 적용한다 |
| rollout이 병목이 아닌데도 쉬운 쪽뿐 아니라 어려운 쪽도 자름 | pool이 중간 난이도로 축소되고, 학습 후반에는 아직 풀지 못한 prompt가 고갈됨 | 같은 pool에서 one-sided cut([[llama-nemotron]]) 실행과 two-sided cut([[skywork-or1-rl-data]]) 실행을 비교한다 |
| filtering 전 mixture만 보고 도메인 균형을 판단함 | filtering 뒤 도메인별 비율이 의도한 mixture와 달라지고, 한 도메인이 gradient를 지배함 | filtering 뒤와 각 학습 stage마다 도메인별 prompt 수를 센다 |
| difficulty label을 실제 난이도로 취급함 | 사람 또는 judge가 "hard"로 표시한 prompt가 zero-variance group을 만듦 | 학습 시작 checkpoint로 pass rate를 다시 측정한다. label은 prior이고, pass rate가 신호다 |
| generation 시간을 아끼려고 rollout batch를 재사용함 | 몇 step 안에 entropy가 떨어지고 test score가 더 이상 개선되지 않음 | 나머지를 모두 고정한 상태에서 `N_reuse = 1`과 2를 비교한다([[skywork-or1-rl-data]] §4.4). step별 entropy를 기록한다 |
| 저장한 log-probability 없이 저장 completion을 replay함 | sample이 stale한데도 모든 token의 importance ratio가 정확히 1이고 clipping이 전혀 발생하지 않음 | replayed group에서 `old_per_token_logps`가 `None`인지 확인한다([[replay-buffer-rlhf]] extract, [[trl-grpo]] loss) |
| 평균 importance ratio로 staleness를 진단함 | 학습은 나빠지는데 평균은 1 근처에 머무름 | ratio의 min/max와 clipped-token 비율을 기록한다. 평균은 구성상 1이다(§6.2) |
| Qwen2.5 base에서 MATH-500 결과로 prompt-set gain을 주장함 | control run에서 random reward 또는 inverted reward로도 큰 gain이 나옴 | partial-prompt completion test와 base 출시 후 benchmark 하나를 실행한다([[reasoning-or-memorization-rl-contamination]]) |
| hack 가능한 format을 평가에서만 제거함 | reward는 오르는데 held-out score는 오르지 않음 | curation 시점에 no-CoT guessing test를 적용한다([[kimi-k1-5]]에서 `N = 8`) |

---

## 이해도 점검

1. 현재 policy에서 어떤 prompt의 실제 pass rate가 0.9이고 group size가 8이라고 하자. §2의 두 수식을 사용하여, 이 prompt를 유지하면 proportional한 gradient를 만들지 못하면서 rollout만 소모하는 이유를 설명하고, 두 값을 모두 계산하라.
2. Llama-Nemotron은 pass rate가 0.75 이상인 prompt를 제거하지만 pass-rate-0 prompt는 유지한다. 반면 Skywork-OR1은 양쪽 끝을 모두 제거한다. run에서 무엇이 부족한가를 기준으로, 각각의 선택이 더 나은 조건을 제시하라.
3. TRL의 replay buffer는 group-scaled advantage를 사용하여 저장한 group의 점수를 `Σ|A_i| · σ_group`으로 매긴다. binary reward일 때 이것이 왜 대략 `2k(G−k)/G`인지 보이고, 따라서 buffer가 어느 pass rate를 선호하는지 말하라.
4. 누군가 replay한 trajectory가 안전하다고 보고하면서 batch 전체에 기록한 평균 importance ratio가 1.02라고 말한다. 왜 이것이 근거가 아닌지 설명하고, 대신 어떤 두 통계를 봐야 하는지 제시하라.
5. filtering 전 pool은 수학 50%, instruction-following 50%이고, filtering 뒤에는 수학이 33%가 되었다. filter 자체에는 변화가 없었다. 이 현상이 생기는 기제를 설명하고, 이 효과가 한 도메인 전체를 제거한 공개 사례를 이름으로 제시하라.
6. DeepSeek-R1은 22k개의 multiple-choice STEM prompt를 유지하지만, Kimi k1.5는 multiple-choice prompt를 제거한다. 각 쪽의 논리를 재구성하고, 둘 중 어느 쪽이 맞는지 결정할 측정이 무엇인지 말하라.
7. Qwen2.5-Math-1.5B에서 one-shot RLVR이 MATH-500을 36.0%에서 73.6%로 올렸다. §7을 사용하여, 이 결과만으로 RL prompt-set 크기가 중요하지 않다고 주장할 수 없는 이유와, 그 주장을 하려면 어떤 증거가 필요한지 설명하라.
8. 어떤 curriculum이 수학 pool에서 가장 어려운 5%만으로 마지막 500 step을 끝낸다. 범용성에 대한 위험을 이름으로 말하고, 이를 탐지할 구체적인 측정과 그 `k` 값을 제시하라.

---

## 연결

- **이전 장:** ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO. 이 장의 filter가 작동시키는 advantage 정의와 clipped objective를 제공한다.
- **다음 장:** ch-43 — Entropy, Output Diversity, and KL Control in RL. §6.3이 sample reuse 아래에서 관찰하는 entropy-collapse 양상을 다룬다.
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. 이 장에서 적용한 negative-feedback section의 유도를 담고 있다.
- ch-44 — Process Supervision and Verifiable Rewards. `(x, g)` pair의 grader 측을 전개한다.
- ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL. partial rollout과 overlong filtering을 length mechanism으로 다룬다.
- ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. §4가 일부만 다룬 transfer와 interference 근거를 이어서 다룬다.
- ch-37 — Policy-Gradient Foundations for Language Models. §6.2에서 쓰는 importance-sampling identity의 출처다.
- ch-48 — Contamination Detection and Its Effect on Reported Scores. §7의 논의를 evaluation 단계까지 확장한다.

---

## 출처

- [[rlvr-tulu3]] — Tülu 3 RLVR prompt set(표 22), PPO setting(표 21), 그리고 405B mixture가 MATH로 축소된 과정(§8.1). 장 내부에서 검증한 발췌: `excerpts/rlvr-tulu3.md`. verifier가 과도한 최적화를 “기계적으로 0”으로 만든다는 library card의 주장은 보고서가 뒷받침하지 않는다. 보고서는 IFEval 과도 최적화를 문서화한다(App. B.4). 이 장은 그 주장을 사용하지 않는다.
- [[deepseek-r1]] — 유형별 RL data 및 prompt 수(표 4, B.3.1), 두 단계 RL setting(§3.2.1–3.2.2), decontamination protocol.
- [[kimi-k1-5]] — prompt-set curation 속성, SFT sample 10개로 측정한 pass-rate 난이도, `N = 8`에서의 easy-to-hack 제거, curriculum과 prioritized sampling, partial rollout(§2.1, §2.3.4, §2.6.2). 장 내부 발췌: `excerpts/kimi-k1-5.md`.
- [[dapo]] — dynamic sampling constraint 및 표 1의 기여도, DAPO-Math-17K 구성. 장 내부 발췌: `excerpts/dapo.md`.
- [[verl-dapo-recipe]] — dynamic sampling을 구현하는 configuration 및 재현 수치. 장 내부 발췌: `excerpts/verl-dapo-recipe.md`.
- [[magistral]] — 2단계 difficulty filtering funnel(표 1), 단계적 difficulty schedule, math-only 대 code-only cross-domain 표(표 5).
- [[llama-nemotron]] — sample 8개로 pass rate 0.75에서 수행한 one-sided cut와 Gaussian progressive batching(§5.1). 장 내부 발췌: `excerpts/llama-nemotron.md`.
- [[olmo-3]] — Dolci-Think-RL 구성(표 20), rollout 8개 중 62.5%에서 하는 offline filtering, active sampling, inflight weight update, spurious-reward negative control.
- [[skywork-or1-rl-data]] — dataset card의 split 크기와 model-aware difficulty estimation 및 retention 비율, `N_reuse`를 측정하는 reuse ablation. 장 내부 발췌: `excerpts/skywork-or1-rl-data.md`.
- [[deepmath-103k]] — raw pool contamination rate 및 decontamination 뒤 difficulty, verification으로 이어지는 funnel. 장 내부 발췌: `excerpts/deepmath-103k.md`.
- [[multi-domain-rlvr-data-centric]] — Qwen2.5-7B의 single-, dual-, triple-domain RL 결과와 policy-refresh를 포함한 curriculum 비교. 장 내부 발췌: `excerpts/multi-domain-rlvr-data-centric.md`.
- [[reasoning-or-memorization-rl-contamination]] — partial-prompt memorization 측정과 clean RandomCalculation benchmark. 장 내부 발췌: `excerpts/reasoning-or-memorization-rl-contamination.md`.
- [[replay-buffer-rlhf]] — TRL의 experimental replay trainer와 documentation을 장 내부에서 검증해 발췌한 `excerpts/replay-buffer-rlhf.md`를 통해서만 사용했다. framework가 prompt만 replay하고 trajectory는 절대 replay하지 않는다는 library card의 진술은 그 code와 모순되며, 이 장은 그 진술을 사용하지 않는다.
- [[trl-grpo]] — commit a08e713의 parent GRPO loss: `old_per_token_logps`가 없을 때 ratio가 1이 되는 방식과 기록 metric.
- [[reinforcement-learning-with-one-training-example]] — one-example RLVR 결과와 historical-variance selection rule을 limit case로 읽는다. 장 내부 발췌: `excerpts/reinforcement-learning-with-one-training-example.md`.
- [[rlvr-beyond-base-model]] — RL-trained policy의 coverage 측정으로서 큰 k에서의 pass@k.
