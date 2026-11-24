<!-- chapter: ch-44a
     track: rl
     kind: content
     title: RL에서의 길이: 과도하게 긴 응답, 길이 제어, 장문맥 RL
     deps: [ch-44, ch-32b]
     sources: [[dapo]], [[verl-dapo-overlong-buffer]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[kimi-k1-5]], [[kimi-k1-5-recipe]], [[magistral]], [[magistral-recipe]], [[grpo]], [[dr-grpo]], [[demystifying-long-cot]], [[l1-lcpo]], [[overthinking-o1-like-llms]], [[sky-t1-flash-overthinking]], [[inverse-scaling-test-time-compute]], [[scaling-reasoning-losing-control-mathif]], [[qwenlong-l1]], [[qwenlong-l1-5]], [[loongrl]], [[loongrl-recipe]], [[memagent]], [[spell-self-play-long-context]], [[randomized-yarn]], [[longer-context-deeper-thinking]], [[minimax-01]], [[minimax-01-recipe]], [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]], [[nemotron-ultra-recipe]], [[deepswe]], [[demystifying-agentic-rl]], [[rollout-training-mismatch-tis]], [[qwen-long-context-synth]], [[context-length-alone-hurts]], [[ruler]], [[tokenskip]]
     figures: figures/length-reward-explorer.html
     revised: 2026-09 (일반성 수정)
-->

# 제44a장 — RL에서의 길이: 과도하게 긴 응답, 길이 제어, 장문맥 RL

> **핵심 통찰.** 모든 RL 실행에서는 두 길이를 사람이 정한다. 하나는 롤아웃이 생성할 수 있는 범위를 제한하는 최대 응답 길이이고, 다른 하나는 롤아웃이 읽을 수 있는 범위를 제한하는 최대 입력 길이이다. 명시적으로 길이 항을 쓰지 않아도 둘 다 보상 항처럼 작동한다. 응답 제한에서 잘린 롤아웃에는 답이 없으므로 정확성 검증기는 이를 오답으로 채점하며, [[dapo]]에서 이 점수를 loss mask로 바꾸자 Qwen2.5-32B의 AIME 2024 avg@32가 30에서 36으로 올랐다(Table 1). 정확성만 쓰는 보상에서는 [[demystifying-long-cot]]가 응답 길이가 문맥 창에 닿을 때까지 증가하고 그 뒤 학습 정확도가 하락한다고 보고한다(§4.1, Figure 2). 명시적 길이 제어는 RL 단계에서 작동한다. [[kimi-k1-5]]는 long2short RL 뒤 평균 3,272토큰에서 AIME 2024 pass@1 60.8을 보고했고(§3.4), [[l1-lcpo]]는 1.5B 모델이 프롬프트로 지정한 예산을 수학에서 평균 약 3% 오차, 예산 위반 0.3–2.3%로 지킨다고 보고했다. 입력 축에서는 중간 길이에서 한 RL이 더 긴 길이로 전이된다. [[loongrl]]은 약 16K에서 학습하고 128K RULER를 개선했으며(Table 7), [[memagent]]는 8K 창 안에서 학습하고 350만 토큰 질문에 정확도 71.09로 답한다(Table 1).
>
> **지침.** 응답 제한에서 롤아웃이 잘렸다면 이를 오답으로 채점한 채 두지 말라. [[dapo]]의 Overlong Filtering처럼 loss를 mask하거나, [[allenai-olmo3-open-instruct-scripts]]의 `mask_truncated_completions`를 쓰거나, [[verl-dapo-overlong-buffer]]처럼 버퍼 안에서 보상을 shaping하라(`overlong_buffer.len = 4096`, penalty factor 1.0). 정답일 수 있지만 끝나지 않은 추론 연쇄를 오답으로 채점하면, 실행이 만들려는 바로 그 행동을 벌주기 때문이다. 추론 시 토큰 예산을 반드시 지켜야 하면 서빙 단계에서 생성을 자르지 말고 정책에 예산을 학습시켜라. 예산 강제는 512와 1024토큰에서 LCPO보다 정확도가 20–25점 낮았다([[l1-lcpo]] §5). 긴 CoT 모델을 짧게 만들 때는 긴 정답뿐 아니라 짧은 오답도 negative로 넣어라. 길이만 다른 쌍은 AIME 2024에서 6.6점을 잃게 했기 때문이다([[sky-t1-flash-overthinking]] Ablations table). 목표 입력 길이가 감당 가능한 롤아웃 길이보다 길다면 입력 길이를 단계적으로 늘려라([[qwenlong-l1]] 20K → 60K, [[qwenlong-l1-5]] 20K/12K → 60K/20K → 120K/50K). 또는 distractor가 많은 데이터로 중간 길이에서 학습하라([[loongrl]]). 반대로 배포 목표에 토큰 예산도 장문 입력도 없다면 길이 항을 넣지 말고, 진단 지표로 truncation rate와 길이 분위수만 기록하라.

## 범용 모델에서 이 장이 중요한 이유

길이는 RL에 독립적인 두 경로로 들어간다.

**응답 축**은 롤아웃이 생성할 수 있는 토큰 수이다. 이는 비용을 가진 하이퍼파라미터다. 생성 시간과 key-value cache 메모리는 이 값과 함께 증가하고, 동기식 step에서는 배치 내 가장 긴 샘플들이 wall-clock 시간을 결정한다([[dapo]] §3.2). 또한 이는 능력 축이기도 하다. 추론 RL은 정확도와 응답 길이를 함께 올리고([[deepseek-r1]] §2.3; [[kimi-k1-5]] §3.3), 예산을 너무 낮게 자르면 학습하려던 행동 자체가 사라진다.

**입력 축**은 프롬프트가 담을 수 있는 문맥 토큰 수이다. mid-training은 모델에 문맥 창을 제공하고([[ch-32b]]), RL 단계는 모델이 그 창에 대해 추론할지 아니면 단지 저장만 할지를 정한다. 긴 입력은 RL 동역학을 직접 바꾼다. [[qwenlong-l1]]은 짧은 입력 길이보다 긴 입력 길이에서 보상 수렴이 더 느리고, 출력 엔트로피는 더 낮으며, KL spike는 더 크다고 보고한다(§1, Figure 2).

범용 모델에서는 이 두 축 때문에 이 장이 다루는 네 가지 실패 양식이 생긴다. (1) truncation을 실패로 취급하는 보상은 모델에게 일찍 멈추거나, 끝내 보상받지 못할 토큰을 계속 생성하도록 가르친다. (2) 길이에 보상하는 보상은 반복을 가르친다. [[demystifying-long-cot]]는 “충분한 compute가 주어지면 length rewards will be hacked”라고 보고한다(§4.5). (3) 길게 쓰는 법만 학습한 모델은 길이가 필요 없는 질문에도 토큰을 쓰며, 일부 과제에서는 더 긴 추론이 정확도를 낮춘다([[inverse-scaling-test-time-compute]]). (4) 긴 입력에만, 또는 짧은 입력에만 학습한 모델은 다른 구간의 능력을 잃는다. [[qwenlong-l1-5]]의 memory-specialized checkpoint는 memory-agent MRCR을 7.7점 얻는 대신 full-context average를 3.06점 잃었고(Table 10), [[scaling-reasoning-losing-control-mathif]]에서 RL rollout limit를 1k에서 8k토큰으로 올리자 수학 정확도는 28.73에서 39.82로 상승했지만 hard constraint 정확도는 19.05에서 14.29로 하락했다(Table 5).

이 장은 process 및 outcome reward 설계([[ch-44]])와 context extension([[ch-32b]]) 뒤의 RL 단계에 놓인다. 순서대로 응답 제한을 하이퍼파라미터로 보는 법(§1), 잘린 롤아웃과 보상 shaping(§2), 길이가 증가하는 이유와 이를 멈추는 요소(§3), 명시적 길이 제어와 long2short(§4), 긴 입력에 대한 RL(§5), 두 길이 구간에서의 회귀(§6), 진단 지표(§7)를 다룬다.

## §1 최대 응답 길이는 메모리 예산을 갖는 하이퍼파라미터다

**정의.** 최대 응답 길이 `L_max`는 sampler가 중단시키기 전까지 롤아웃이 생성할 수 있는 토큰 수다. 이렇게 중단된 응답은 *truncated* 상태다. end-of-sequence token 없이 끝나며, 추론 과제라면 최종 답도 없다.

**측정 가능한 문제.** `L_max`의 상한은 context window이다. 프롬프트와 응답이 함께 들어가야 하기 때문이다. 동시에 진행 중인 sequence가 차지하는 key-value(KV) cache도 상한을 만든다. layer 수가 `n_layers`, key-value head 수가 `n_kv`, head 차원이 `d_head`, 원소당 byte 수가 `b`인 decoder에서 cache는 다음과 같다.

```
bytes_per_token = 2 · n_layers · n_kv · d_head · b        (keys and values)
bytes_per_sequence = bytes_per_token · (prompt_len + L_max)
```

예시 설정은 `n_layers = 32`, `n_kv = 8`, `d_head = 128`, bf16이므로 `b = 2`이다. 이때 `bytes_per_token = 2 · 32 · 8 · 128 · 2 = 131,072` bytes, 즉 128 KiB이다. DAPO 설정인 프롬프트 2,048토큰 및 `L_max = 20,480`([[dapo]] §4.1)에서, 전체 길이에 도달한 sequence 하나는 `128 KiB · 22,528 = 2.75 GiB`를 보유한다. 롤아웃 step 하나가 512 prompts × 16 samples이면 8,192 sequences이다. 모두 제한까지 생성한다면 cache는 약 22 TiB가 필요하다. 그래서 롤아웃을 여러 wave로 나누어 생성하고, 길이 제한을 올리면 batch를 줄인다. [[magistral]]은 이 trade-off를 직접 적는다. Magistral Medium 학습 동안 penalty를 받지 않는 길이 `l_max − l_cache`는 16k → 24k → 32k로 늘고, batch size는 8k → 4k → 2k로 줄어든다. 목적은 “to limit KV-cache memory”이다(§5.2, [[magistral-recipe]]).

**공개된 실행에서 사용한 값.** 이 값들은 공통 default가 아니라 모델 유형에 따라 다르다. DeepSeek-R1의 추론 RL은 32,768토큰이고 R1-Zero는 step 8.2k 뒤 65,536토큰이다([[deepseek-r1-recipe]]). DAPO는 20,480([[dapo]] §4.1), Nemotron 3 Nano의 RLVR은 overlong filtering과 함께 49K([[nemotron-ultra-recipe]], NR §3.2.5), QwenLong-L1은 20K–60K 입력에서 10K 출력([[qwenlong-l1]] §3.2), L1은 학습 중 4K([[l1-lcpo]] §4)를 사용한다. Olmo 3 script는 규칙을 명시한다. 한 script 계열은 “model type에 맞춘 response length(8,192 Instruct, 32,768 Think, 16,384 RL-Zero)”를 사용한다([[allenai-olmo3-open-instruct-scripts-recipe]], starting-point paragraph). 모든 값은 아래 Recipe 표에 있다.

**실행 도중 제한을 올리는 것은 알려진 개입이다.** DeepSeek-R1-Zero의 제한은 step 8.2k에서 32,768에서 65,536으로 올랐고, 보고서는 성능과 응답 길이가 “exhibit a significant jump at the 8.2k step”이라고 적는다([[deepseek-r1]] §2.1, Figure 1). 하지만 이 보고서는 더 긴 제한의 효과와 계속된 학습의 효과를 분리하지 않는다. 따라서 통제 비교가 아니라 단일 관찰이다(Result, single study).

**조건과 한계.** 위 숫자들은 모델 크기 사이에 그대로 옮길 수 없다. [[kimi-k1-5]]는 같은 데이터로 두 모델 크기를 학습했고, 작은 모델은 더 긴 chain of thought로 비슷한 정확도에 도달한 반면 큰 모델은 토큰 효율이 더 높았다고 보고한다(§3.5, Figure 8). 따라서 다른 보고서의 모델에서 가져온 예산은 추정일 뿐이다.

**범용 모델에 대한 함의.** `L_max`는 context window가 아니라 학습 중인 모델의 측정된 길이 분포(§7)와 배포 예산에서 정해야 한다. p99 응답 길이보다 훨씬 큰 제한은 gradient를 바꾸지 않은 채 메모리와 throughput만 소모한다. 반대로 median 근처의 제한은 롤아웃 대부분을 자르고 §2의 문제를 지배적인 학습 신호로 만든다.

## §2 잘린 롤아웃: 끝나지 않은 추론에 보상이 하는 일

**정의.** truncation은 모델이 멈춘 것이 아니라 `L_max`가 생성을 멈춘 것이다. 검증기는 답을 보지 못하므로, 정확성만 쓰는 보상은 그 롤아웃을 오답으로 채점한다.

**측정 가능한 문제.** [[dapo]]는 이를 이렇게 적는다. “we assign a punitive reward to truncated samples. This approach may introduce noise into the training process, as a sound reasoning process can be penalized solely due to its excessive length”(§3.4). 이렇게 벌을 받는 롤아웃은 기대상 가장 긴 추론을 한 롤아웃이며, 어려운 문제에서는 끝까지 갔다면 정답이었을 가능성이 가장 큰 롤아웃이다. 이 label은 모델이 만든 것이 아니라 학습 설정이 만든 **false negative**다.

**출처가 제시한 네 가지 처리법.**

1. **징벌 보상(punitive reward, 기본값).** 잘린 샘플은 오답 보상을 그대로 받는다([[dapo]] Eq. 7에서는 `−1`, binary-reward 실행에서는 `0`). 그리고 다른 오답과 똑같이 loss에 들어간다.
2. **Overlong filtering.** 잘린 샘플의 loss를 mask하여 gradient에 전혀 기여하지 않게 한다([[dapo]] §3.4). [[allenai-olmo3-open-instruct-scripts]]는 이를 `mask_truncated_completions`로 구현한다. 이는 “removes rollouts whose finish reason is not 'stop'”이다. 공개된 Olmo 3 script는 RL-Zero code, instruction-following 실행, general 실행에서만 이를 True로 설정하고, Think, Instruct, RL-Zero math 실행에서는 False로 설정한다. 후자의 경우 잘린 롤아웃은 검증기 점수를 가진 채 batch에 남는다.
3. **부드러운 overlong 벌점(soft overlong punishment).** hard limit 바로 아래의 buffer 안에서, 길이에 비례하면서 범위가 제한된 벌점을 준다([[dapo]] Eq. 13).

```
R_length(y) = 0                                      if |y| ≤ L_max − L_cache
              ((L_max − L_cache) − |y|) / L_cache    if L_max − L_cache < |y| ≤ L_max
              −1                                     if |y| > L_max
```

   `|y|`는 토큰 단위 응답 길이, `L_max`는 최대 생성 길이, `L_cache`는 buffer 폭이다. 이 항은 정확성 보상에 더해진다. verl 구현은 실제 생성 길이에서 같은 양을 계산하고 0에서 clamp한다([[verl-dapo-overlong-buffer]]).

```python
overlong_buffer_len = self.overlong_buffer_cfg.len
expected_len = self.max_resp_len - overlong_buffer_len
exceed_len = valid_response_length - expected_len
overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
reward += overlong_reward
```

4. **agent용 compact filtering.** [[deepswe]]는 overlong filtering을 최대 environment step 수에 도달했거나 20분 생성 timeout에 걸린 trajectory까지 확장한다. agent가 그 뒤 무작위 파일을 수정하다 우연히 test를 통과할 수 있고, 그런 trajectory에 보상하면 “leads to collapse”하기 때문이다(§2.3, Figure 6).

**계산 예시: label 변경이 group에 하는 일.** prompt 하나, `G = 8`개 롤아웃, binary reward, 모집단 표준편차를 쓴 GRPO advantage `A_i = (r_i − mean(r)) / std(r)`([[grpo]])를 생각하자. 롤아웃 2개는 정답이면서 짧고, 5개는 오답이면서 짧으며, 1개는 현재까지의 추론은 정답이지만 `L_max`에서 잘렸다고 하자.

- **잘린 롤아웃을 0으로 채점:** `r = [1, 1, 0, 0, 0, 0, 0, 0]`, 평균은 0.25, 표준편차는 `√(0.25 · 0.75) = 0.433`이다. 정답 롤아웃은 `A = 0.75/0.433 = +1.73`을 받고, 잘린 롤아웃을 포함한 모든 오답 롤아웃은 `A = −0.25/0.433 = −0.58`을 받는다.
- **잘린 롤아웃을 mask:** group은 `r = [1, 1, 0, 0, 0, 0, 0]`이 되고, 평균은 0.286, 표준편차는 `√(0.286 · 0.714) = 0.452`이다. 정답 롤아웃은 `A = +1.58`, 오답 롤아웃은 `A = −0.63`을 받으며, 잘린 롤아웃은 gradient를 전혀 받지 않는다.
- **`L_max = 20,480`, `L_cache = 4,096`에서 soft punishment:** 18,000토큰인 *정답* 응답은 `(16,384 − 18,000)/4,096 = −0.39`를 받는다. 따라서 보상은 `1`이 아니라 `1 − 0.39 = 0.61`이지만, 계속 positive example으로 남는다.

세 번째 경우가 DAPO 저자가 최종 실행에서 유지한 경우다. verl FAQ는 “most experiments in the paper, including the best-performant one, are run without Overlong Filtering because it's somehow overlapping with Overlong Reward Shaping”이라고 적는다([[verl-dapo-overlong-buffer]]). [figures/length-reward-explorer.html](figures/length-reward-explorer.html)에서는 group의 길이와 정확성을 설정하여 각 처리법에서의 보상과 advantage를 읽을 수 있다.

**길이 정규화와의 상호작용.** 롤아웃이 받는 토큰당 weight는 loss를 집계하는 방식에 따라 달라진다. sample-level GRPO는 각 롤아웃의 token loss를 해당 롤아웃 길이 `|o_i|`로 나눈다. 따라서 `A = −0.58`인 20,480토큰의 잘린 롤아웃은 각 token에 `−0.58/20,480`을 걸지만, 2,000토큰짜리 오답 롤아웃은 자신의 각 token에 그 열 배를 건다. [[dr-grpo]]는 이를 length bias라고 부른다. negative advantage에서는 `|o_i|`가 큰 긴 응답이 token당 덜 벌을 받으므로, 정책은 더 긴 오답 응답을 선호하게 된다(§3.1). DAPO의 token-level loss는 대신 `Σ_i |o_i|`로 나눈다([[dapo]] Eq. 12). 이는 token당 weight를 같게 만들며, 따라서 잘못 label된 잘린 롤아웃에 가해지는 *전체* push를 그 길이에 비례하게 만든다. 위 숫자에서는 잘린 롤아웃의 negative gradient mass가 짧은 오답 하나의 약 열 배다. loss가 token-level이 되면 잘린 보상을 mask하거나 shaping하는 일은 덜 중요한 것이 아니라 더 중요해진다(Interpretation. 출처는 ablation 결과를 보고하지만 이 분해를 직접 보고하지는 않는다).

**근거.** [[dapo]] Table 1의 Qwen2.5-32B base, AIME 2024 avg@32, 누적 단일 실행 결과는 다음과 같다. naive GRPO 30, overlong filtering 추가 36, clip-higher 추가 38, soft overlong punishment 추가 41, token-level loss 추가 42, dynamic sampling 추가 50이다. 비교 지점인 DeepSeek-R1-Zero-Qwen-32B는 47이다. overlong filtering은 이 사다리에서 가장 큰 단일 단계이며, 새로운 positive 신호를 추가한 것이 아니라 negative 신호를 *제거*해서 작동한다(Result, single study, math only).

**조건과 한계.** 이 사다리는 누적되어 있으므로 overlong filtering의 +6점은 naive GRPO 위에서만 측정되었다. [[demystifying-agentic-rl]]은 agentic setting에서 overlong reward shaping을 다시 쓰지만(§4.1, Eq. 9), `L_cache`를 제시하지 않는다. 또 이 항을 token-level loss와 clip-higher와 함께 하나의 recipe(GRPO-TCR)로 묶으므로 별도 효과를 측정하지 않는다. ledger는 GRPO baseline의 최대 응답 길이를 16,384토큰으로 제시한다(App. A.1).

## §3 응답 길이는 왜 늘어나며, 무엇이 그것을 멈추게 하는가

**메커니즘.** 정확성만 보상하는 목적에서는 토큰에 가격을 매기는 항이 없다. 따라서 정답을 낼 확률을 높이는 행동은, 그 행동에 드는 토큰 수와 무관하게 강화된다. [[dapo]]는 유용한 측면을 다음처럼 설명한다. “The increase in length provides the model with a larger space for exploration, allowing more complex reasoning behaviors to be sampled and gradually reinforced through training.” 즉 길이가 늘면 더 복잡한 추론 행동을 표본으로 뽑고, 훈련을 통해 조금씩 강화할 탐색 공간도 커진다. 다만 같은 절(§4.3)은 “length does not always maintain a continuous upward trend during training. In some considerable periods, it can exhibit a trend of stagnation or even decline”라고 즉시 덧붙인다. 길이는 훈련 내내 단조롭게 증가하지 않으며, 상당한 기간 동안 정체하거나 오히려 감소할 수 있다.

**암묵적 벌점.** [[demystifying-long-cot]]은 16K context window에서 Llama-3.1-8B와 Qwen2.5-Math-7B를 정답이면 `+1`만 주는 보상으로 실행했다. 두 모델 모두 window에 닿을 때까지 길이가 늘었고, 이 때문에 “CoTs exceeding the allowable window size”가 발생하여 training accuracy가 하락했다. exceed 비율은 이후 “leveled off at a certain threshold below 1”, 즉 1보다 작은 어떤 임계값에서 평탄해졌다. 저자들의 해석은 window 자체가 길이 벌점으로 작동한다는 것이다. 명시적인 exceed-length penalty가 없어도, “a trajectory might be penalized even without an explicit exceed-length penalty due to reward or advantage normalization”될 수 있다(§4.1). 이는 §2를 반대쪽에서 본 같은 메커니즘이다. 길이 항이 없는 실행에서 길이 증가를 막는 것은 결국 truncation label뿐이다.

**형태를 만든 대안: cosine 보상.** [[demystifying-long-cot]]은 평평한 보상을 길이와 정답 여부에 의존하는 보상으로 바꾼다(§4.2, App. C.1).

```
R(C, L_gen) = CosFn(L_gen, L_max, r0^c, rL^c)   if C = 1 (정답)
              CosFn(L_gen, L_max, r0^w, rL^w)   if C = 0 (오답)
              r_e                                if L_gen = L_max

CosFn(t, T, η_min, η_max) = η_min + 0.5 · (η_max − η_min) · (1 + cos(t π / T))
```

`C`는 정답 여부, `L_gen`은 생성한 길이, `L_max`는 최대 길이, `r0`과 `rL`은 길이 0 및 `L_max`에서의 보상이다. 위첨자 `c`와 `w`는 각각 정답(correct) 응답과 오답(wrong) 응답의 branch를 가리키며, `r_e`는 exceed-length penalty다. 논문이 보고한 값은 `r0^c = +2`, `rL^c = +1`, `r0^w = −10`, `rL^w = 0`, `r_e = −10`이다. 여기에 N-gram repetition penalty `P = −0.05`, `N = 40`도 적용했다(App. E.5.1).

`L_max = 14,336`으로 계산해 보자. 이는 그 실행에서 쓴 generation length다. 정답 응답이 `L_gen = 7,168`이면 `1 + 0.5 · (2 − 1) · (1 + cos(π/2)) = 1.5`를 받는다. 같은 정답을 1,000토큰에 냈다면 약 1.99를 받고, `L_gen`이 `L_max`에 가까워질수록 정답 branch는 `rL^c = 1`로 접근한다. 오답 응답이 7,168토큰이면 `0 + 0.5 · (−10 − 0) · (1 + 0) = −5`를 받는다. 1,000토큰 오답은 약 −9.9를 받고, `L_max` 근처에서는 오답 branch가 `rL^w = 0`으로 접근한다. 실제로 `L_max`에 도달한 응답은 truncate되어 세 번째 branch를 타므로, 내용과 무관하게 `r_e = −10`을 받는다. 잘린 롤아웃이 받을 수 있는 보상은 exceed-length penalty뿐이다. 따라서 짧고 정답인 경우가 최상, 짧고 오답인 경우가 최악이다. 이 보상 순서는 모델이 오답일 가능성이 클 때에만 더 긴 생각의 비용을 지불하게 한다.

**그 모양이 하는 일.** cosine 보상은 평평한 보상보다 “more stable (a) training accuracy and (b) response length”를 보였다(Figure 4). 하이퍼파라미터는 길이가 늘어나는 방향도 통제한다. 정답의 보상이 CoT 길이와 함께 증가하면, 즉 `r0^c < rL^c`이면 “the CoT length increases explosively”한다. 또 정답 보상이 오답 보상보다 낮을수록 CoT가 길어진다. 저자들은 이를 학습된 risk aversion으로 해석한다(§4.3, Interpretation).

**길이 보상은 해킹된다.** 같은 논문은 “Length rewards will be hacked with enough compute ... but this can be mitigated using a repetition penalty”라고 쓴다(§4.5). 관찰된 해킹은 어려운 문제에서의 반복이며, 동시에 분기용 단어 `alternatively`의 등장 횟수가 감소했다(Figure 10).

**더 긴 window에는 더 많은 훈련 계산량이 든다.** 훈련 샘플 수를 같게 둔 실험에서 8K-window 실행은 4K와 16K 실행을 모두 이겼다. 저자들은 이를 “models need more training compute to learn to fully utilize longer context window sizes”라는 뜻으로 읽는다(§4.4, Figure 6). 이는 Llama-3.1-8B, math prompt에서 수행한 단일 연구 결과다.

**가중치에서 성장이 보이는 위치.** [[magistral]]은 checkpoint들을 따라 PCA 방향을 잡았을 때 mean reward와 output length가 함께 증가한다고 보고한다. 평균 output length가 1,500~8,000토큰인 구간에서는 raw reward가 mean output length에 대해 로그 스케일로 증가한다(§7.1, Figures 8-9). Magistral의 자체 길이 항은 `l_max − l_cache`와 `l_max` 사이에서 0부터 −0.1까지의 벌점이고, `l_max`를 넘으면 −0.1이다([[magistral-recipe]]). 이 벌점은 0.9인 정확성 component에 비하면 작다.

**범용 모델에 주는 함의.** 길이 증가는 목표가 아니라 목적함수의 부작용이다. 안전한 조절 수단은 두 가지다. 하나는 limit 근처에 범위가 제한된 벌점을 두는 방법(DAPO, Magistral)이고, 다른 하나는 보상 순서 제약을 명시한 shaped reward(cosine reward)다. 둘 다 repetition diagnostic이 필요하다. 길이에 보상하는 어떤 보상도 반복이 가장 싸게 만족시키는 방법이기 때문이다. 같은 실패를 entropy 관점에서 다루는 내용은 [[ch-43]]를 보라.

## §4 길이 제어: budget, long2short, 그리고 과도한 생각(overthinking) 문헌

**정의.** 길이 제어는 배포 모델의 응답 길이를 조절 가능한 양으로 만드는 훈련 개입이다. 그 방식은 고정된 길이 축소(long2short)일 수도 있고, prompt마다 다른 budget을 주는 방식(LCPO)일 수도 있다.

### §4.1 RL 보상 안에 넣는 길이 항(Kimi k1.5)

[[kimi-k1-5]]는 task reward에 길이 보상을 추가한다(§2.3.3). 문제 `x`에서 표본으로 뽑은 `k`개 응답의 길이를 `len(i)`라 하고, `min_len = min_i len(i)`, `max_len = max_i len(i)`라 하자.

```
λ = 0.5 − (len(i) − min_len) / (max_len − min_len)
r_length(i) = λ              if the response is correct
              min(0, λ)      if the response is wrong
              0              for all i if max_len = min_len
```

계산 예를 들면 `k = 4`, 길이 1,000 / 2,000 / 3,000 / 5,000에서는 `λ = 0.5 / 0.25 / 0.0 / −0.5`다. 정답인 1,000-token 응답은 `+0.5`를 얻고, 정답인 5,000-token 응답은 0.5를 잃는다. 오답인 1,000-token 응답은 짧다는 이유로 아무 보상도 얻지 못한다(`min(0, 0.5) = 0`). 반면 오답인 5,000-token 응답은 0.5를 잃는다. 이 항은 짧은 *정답*에 보상하며, 오답이 짧다는 이유로 보상하지는 않는다. 훈련은 먼저 penalty 없이 실행하고 이어서 상수 penalty를 켠다. 보고서는 penalty가 초반 훈련을 늦출 수 있다고 쓰지만, 전환 시점과 weight는 제공하지 않는다([[kimi-k1-5-recipe]]).

### §4.2 long2short: 긴 CoT 모델을 짧은 budget으로 옮기는 네 가지 방법

[[kimi-k1-5]] §2.4는 네 가지 방법을 비교한다. 첫째, long-CoT 모델과 short-CoT 모델을 weight-averaging으로 merge한다. 둘째, shortest rejection sampling이다. `n = 8`개 표본을 뽑고, 그중 가장 짧은 정답 하나를 SFT에 남긴다. 셋째, DPO다. 가장 짧은 정답 응답을 chosen으로, 오답 응답들과 chosen보다 1.5× 이상 긴 정답 응답들을 rejected로 둔다. 넷째, 성능 대비 token 균형이 가장 좋은 checkpoint에서 시작하여 별도의 long2short RL phase를 돌린다. 이 phase에는 length penalty와 줄인 최대 rollout 길이를 쓴다.

보고된 결과에서 long2short RL은 네 방법 중 token efficiency가 가장 높다(Figure 7). `k1.5-short w/ rl`은 평균 3,272 tokens에서 AIME 2024 pass@1 60.8을 달성한다(8회 실행 평균). `k1.5-shortest`는 다른 short 모델과 비슷한 token 수에서 MATH-500 88.2에 이른다(§3.4). 보고서에는 방법별 수치 표가 없으므로, 방법 순위는 Figure 7 하나에 근거한다.

### §4.3 Budget-conditioned RL(L1 / LCPO)

[[l1-lcpo]]는 budget을 prompt 안에 넣는다. 각 훈련 prompt에는 `"Think for n_gold tokens."`를 덧붙이며, `n_gold`는 `U(100, 4000)`에서 균등하게 뽑는다. GRPO가 최적화하는 식은 다음과 같다.

```
LCPO-Exact:  r = I(y = y_gold) − α · |n_gold − n_y|
LCPO-Max:    r = I(y = y_gold) · clip(α · (n_gold − n_y) + δ, 0, 1)
```

`I(·)`는 정답 indicator, `n_y`는 실제 생성 길이, `α = 0.0003`은 길이 weight다. `δ = 0.5`는 작은 초과 길이를 가진 정답이 어떤 오답보다도 높은 보상을 유지하게 하는 offset이다.

`n_gold = 1,000`, `α = 0.0003`인 예를 보자. 1,400 tokens의 정답 응답은 Exact에서 `1 − 0.0003 · 400 = 0.88`을, Max에서 `1 · clip(0.0003 · (−400) + 0.5, 0, 1) = 0.38`을 받는다. 1,000 tokens의 정답은 Exact에서 1.0, Max에서 `clip(0.5) = 0.5`다. 길이와 무관하게 오답이면 Exact에서는 1,400-token일 때 `−0.12`, Max에서는 `0`이다. Exact는 요청 길이보다 짧아도 길어도 모두 요청 길이 쪽으로 밀어 준다. Max는 일찍 끝내는 것은 허용하고, budget을 초과할 때만 보상을 잃게 한다.

4K context에서 훈련하고 8K에서 평가한 DeepScaleR-1.5B-Preview 결과는 다음과 같다. budget에서 멈추고 `"Final Answer"`를 삽입하는 S1-style budget forcing과 비교하여, L1은 512 및 1024-token budget 모두에서 “over 100-150% relative and 20-25% absolute performance gains”를 얻었다. accuracy-vs-log-length slope는 L1이 0.24, S1이 0.37이다. 평균 길이 오차는 math benchmark에서 약 3%, out-of-domain에서 20--40%다. L1-Max가 budget을 500 tokens 넘게 위반한 비율은 0.3--2.3%(평균 1.3%)다. 같은 짧은 budget에서 L1-Max는 동급 크기 non-reasoning 모델 Qwen2.5-1.5B-Instruct의 41.0에 대해 평균 46.2를 기록한다(L1의 base는 DeepScaleR-1.5B-Preview). 같은 math set에서 GPT-4o의 45.6에 대해서는 47.8을 기록한다(Table 1). 이는 1.5B 모델 하나와 math 훈련 데이터에 대한 단일 연구 결과다.

### §4.4 RL 없이 줄이기: 모델 자신의 sample로 만든 preference pair

두 연구는 RL 대신 preference optimization으로 long-CoT 모델을 줄인다. 둘 다 **길이로 정의한 negative**를 쓴다는 점이 여기서 중요하다(§ "Negative samples and negative feedback").

[[overthinking-o1-like-llms]]는 QwQ-32B-Preview에서 PRM12K 문제마다 응답 10개를 sample하고, 정답만 유지한 뒤 네 종류의 positive를 만든다. Table 3의 통계는 평균 solution rounds / 평균 tokens / outcome efficiency / process efficiency 순서로 다음과 같다. shortest response는 2.5 / 1051.3 / 69.8% / 80.3%, first-correct solutions(FCS)는 1.1 / 681.0 / 99.5% / 99.1%, FCS+Reflection은 1.9 / 878.7 / 78.4% / 82.4%다. negative는 greedy response가 아니라 가장 긴 sample 응답이다. 저자들은 이것이 “provides a clearer contrastive signal”이라고 설명한다. MATH500(Table 4)에서 FCS positive를 쓴 SimPO는 tokens를 2407.9에서 1016.0으로 줄이지만 정확도는 93.0에서 91.0으로 내린다. FCS+Reflection은 1330.7 tokens에서 정확도 92.8이다. AIME24에서는 같은 recipe가 정확도 46.7 → 43.3, tokens 9480.9 → 5154.5로 바꾼다. GPQA에서는 59.6 → 59.1, 3228.4 → 2085.7이다.

[[sky-t1-flash-overthinking]]은 Sky-T1-32B-Preview에서 self-generated pair 10K개를 만들어 SimPO를 적용하고, 길이는 줄면서 정확도는 유지되었다고 보고한다. MATH500은 2124 → 1417 tokens에서 88.6 → 88.6, AIME24는 6881 → 4365에서 43.3 → 43.3, LiveCodeBench-Hard는 14564 → 6199에서 17.9 → 17.9, MMLU는 1087 → 799에서 82.4 → 81.7, GPQA Diamond는 3503 → 2148에서 56.8 → 56.6이다. 가져가야 할 ablation은 다음이다. `longest correct vs shortest correct` pair만 쓰면 AIME24는 36.7, LCB-Hard는 13.5까지 떨어진다. rejected가 *짧은 오답*인 pair 1K개를 더하면 AIME24는 43.3으로 회복한다.

[[tokenskip]]은 supervised compression 대안이다. token importance로 pruning한 모델 자신의 정답 chain에 LoRA fine-tuning을 하며, compression ratio를 입력에 넣는다. Qwen2.5-14B-Instruct의 GSM8K 결과는 313.11 → 180.68 tokens, 정확도 93.1 → 92.7이다.

### §4.5 길수록 언제나 더 좋은 것은 아니다

[[inverse-scaling-test-time-compute]]는 추론을 길게 할수록 정확도가 낮아지는 task를 만든다. Claude Opus 4는 Misleading Math에서 거의 100%에서 약 85--90%로 떨어진다. DeepSeek R1은 natural overthinking 조건에서 distractor가 다섯 개일 때 70%에서 30%로 떨어진다(Fig. 3). 반면 Claude Sonnet 4, o4-mini, o3는 MultiArith, ASDiv, GSM8K, GSM-IC에서 inverse scaling을 보이지 않는다(App. F.3). 이 추세의 부호는 모델 family와 task에 따라 달라진다(Table 1). 저자들의 권고는 측정 protocol이다. 여러 reasoning budget과 자연스럽게 sample된 여러 길이에서 평가하라는 것이다. [[deepseek-r1]]은 같은 문제의 비용 측면을 보고한다. 2024 competition 366문제에서 R1은 평균 8,793 thinking tokens로 61.8%를 푼다. 쉬운 문제는 7,000 아래, 가장 어려운 문제는 18,000 이상이다(E.4). 저자들은 단순한 질문에서 overthinking도 언급한다(§6).

**범용 모델에 주는 함의.** Budget 제어는 효율성만의 문제가 아니라 generality 속성이다. `2 + 3`에 1,000 tokens를 쓰는 모델은 혼합 traffic에 배포하기 더 나쁘다. [[overthinking-o1-like-llms]]는 이 질문에서 conventional 모델보다 1,953% 많은 tokens를 쓴다고 보고한다. 더 오래 생각할수록 정확도가 떨어지는 모델이라면 budget은 비용 제어가 아니라 안전 제어가 되어야 한다.

## §5 긴 입력에 대한 RL: 단계, distractor, 그리고 memory

응답 축은 모델이 **얼마나 많이 쓰는지**에 관한 것이다. 이 절은 모델이 **얼마나 많이 읽는지**에 관한 것이다. 공개된 네 가지 RL 설계는 긴 입력 롤아웃의 비용을 서로 다른 방식으로 처리한다.

**점진적 context 확장([[qwenlong-l1]]).** RL을 목표 입력 길이가 점점 커지는 여러 phase로 나눈다. phase `k`에서는 `L_{k−1} < |x| + |c| ≤ L_k`를 만족하는 예제만 훈련한다(Eq. 9). 여기서 `x`는 질문, `c`는 함께 읽는 context이며, `|x| + |c|`가 해당 phase의 입력 길이 구간에 들어가야 한다. 이전 phase의 어려운 예제는 `diff(x, c) = 1 / mean({r_i})`라는 가중치로 다시 넣는다. 즉 group rollout의 평균 보상이 낮을수록 가중치가 크다. base 모델이 한 번도 풀지 못한 항목은 평균 보상이 0에 가까우므로 가장 큰 가중치를 받는다(Eq. 10). RL 전에는 DeepSeek-R1 demonstration으로 warm-up SFT를 수행한다. 설정은 입력 20K 뒤 60K, 출력 10K, rollout 8개, batch 128, learning rate `2e-6`, `32×A100-80G`다. 7개 DocQA benchmark에서 R1-Distill-Qwen-32B는 65.6 → SFT 68.7 → QwenLong-L1-32B-DAPO 70.7이었고, 14B 모델은 64.2 → 65.0 → 68.3이었다(Table 4). 점수 외에 중요한 관찰은 두 가지다. single-stage RL에는 phase RL에는 없던 “fluctuating KL divergence and entropy collapse”가 나타났다(§4.2, Figure 5c). 또 long-context SFT 모델은 출발점은 더 높았지만 RL로 0.3점만 얻은 반면, short-context SFT 모델에서 RL을 시작하면 3.2점을 얻었다(§4.3).

**입력과 출력 길이를 함께 확장([[qwenlong-l1-5]]).** “입력 context 길이가 늘어날수록 reasoning content 길이도 대체로 양의 증가 추세를 보인다”는 관찰에 따라, stage마다 둘을 함께 올린다. 순서는 입력 20K / 출력 12K → 60K / 20K → 120K / 50K이며, 난이도 filtering은 다음 stage 설정에서 다시 계산한다(§4.1). stage별 full-context 평균은 base 61.92 → 69.59 → 70.46 → 71.59였다. MRCR은 76.35 → 82.69으로 올랐지만, DocMath와 Frames처럼 짧은 입력 benchmark는 Stage-1 뒤에 평평하게 유지되었다(Table 10). 최종 모델은 평균 71.82, base 대비 +9.90, MRCR +31.72를 기록했다(Table 7).

**중간 길이에서 훈련하고 긴 길이에서 평가([[loongrl]]).** KeyChain 항목은 약 16K token의 distractor document 안에 UUID key-value chain으로 실제 질문을 숨긴다. 따라서 모델은 multi-hop 추론 전에 먼저 질문의 위치를 찾아야 한다. 학습은 two-way substring reward, `G = 8`, `β = 0.001`, 최대 출력 4,096, 약 16K 입력의 GRPO다([[loongrl-recipe]]). LongBench v1 multi-hop 평균은 Qwen2.5-7B-Instruct에서 48.9 → 72.4, Qwen2.5-14B-Instruct에서 53.1 → 74.2였다(Table 2). 16K 훈련은 더 긴 길이로도 전이되었다. NarrativeQA 32K--64K는 7B에서 42.4 → 57.2, YaRN을 쓴 RULER 128K는 7B에서 69.41 → 76.84, 14B에서 73.57 → 79.92였다(Tables 3, 7). KeyChain을 같은 양의 평범한 long multi-hop QA로 바꾸면 72.4 대신 66.2가 나왔다(Table 4). 저자들은 이 차이를 입력 길이 자체보다 과제 난이도의 효과로 해석한다. 다만 카드에 기록된 한계가 있다. 평가 task 다섯 개 중 세 개가 RL seed와 같은 dataset에서 왔고, overlap 검사는 보고되지 않았다.

**길이를 context window 밖으로 옮기기([[memagent]]).** 문서를 chunk 단위로 stream하고, 모델은 각 chunk 뒤에 고정 길이 memory를 덮어쓴다. 따라서 chunk 하나당 compute는 일정하고, 전체 compute는 chunk 수에 선형 비례한다. 훈련은 8K window를 query 1,024 + chunk 5,000 + memory 1,024 + output 1,024로 배분한다. Multi-Conv DAPO는 답변 conversation의 보상에서 group mean을 뺀 하나의 advantage를 sample의 **모든** conversation에 준다. 표준편차로 나누지는 않는다. 이것이 memory-update 단계까지 credit이 전달되는 방식이다. RULER-HotpotQA accuracy에서 RL-MemAgent-14B는 7K에서 80.47, 3.5M token에서 71.09이었다. 비교 대상 QwenLong-L1-32B는 7K에서 72.66이지만 896K에서 11.72로, Qwen2.5-Instruct-14B-1M은 60.16에서 0.00으로 떨어졌다(Table 1). [[qwenlong-l1-5]]도 같은 paradigm을 채택했고 specialization 비용을 보고했다. memory-RL expert는 MRCR 512K--1M에서 20.34에 도달했으나 full-context 평균은 68.53으로 내려갔다. SCE merging 후에는 21.68을 유지하면서 평균 71.18을 복원했다(Table 10).

**annotation 병목 제거([[spell-self-play-long-context]]).** 하나의 policy가 raw document 위에서 questioner, responder, verifier 역할을 맡는다. questioner 보상은 responder 성공률을 입력으로 하는 Gaussian이며 중심은 `µ = 0.5`, `σ = 0.5/3`다. 성공률이 0 또는 1이면 보상은 0, grounded되지 않은 질문은 −0.5, formatting error는 −1이다(Eq. 7). 따라서 curriculum은 responder의 현재 능력을 따라간다. 학습은 최대 입력 16K, 최대 출력 4K(추론 모델은 20K)를 쓴다. 6개 long-context QA benchmark 평균에서 Qwen2.5-14B는 16K에서 37.3 → 51.7, 100K에서 36.1 → 51.1이었다. Qwen3-30B-A3B-Thinking은 각각 60.7 → 62.7, 63.6 → 65.9였다. 같은 모델에서 R1-synthesized data를 쓴 RLVR baseline은 각각 0.0과 0.9만 올랐다. 100K 입력의 pass@8은 SPELL 74.5, RLVR 68.1, base 모델 66.9였다. 이는 한 번의 답을 더 날카롭게 고르는 변화보다, 여러 sample 중 정답이 나올 coverage가 넓어진 변화다([[ch-43]], [[ch-43a]]와 비교).

**같은 목표를 위한 non-RL 대안.** [[randomized-yarn]]은 길이 curriculum 아래에서 더 긴 범위로부터 sample한 position index를 가진 YaRN encoding으로 훈련한다. 5K개보다 적은 short-context 예제에 LoRA를 사용했으며, Qwen2.5-7B-Instruct의 BABILong OOD 평균은 vanilla LoRA 83.6에 비해 90.3이었다. 가장 큰 이득은 128K에서 나왔고, curriculum을 빼면 MRCR에서 6.2점을 잃었다. [[longer-context-deeper-thinking]]은 SFT 측의 선행 조건을 제시한다. reasoning-SFT 데이터가 동일해도 32K needle 점수가 더 좋은 LLaMA3-8B-Instruct variant가 SFT 뒤 더 높은 MATH500 accuracy를 얻었다. `θ×1`은 54.80, `θ×16`은 59.36이었다(Table 3). 긴 입력을 처리하는 능력과 긴 출력을 이용해 추론하는 능력은 독립된 capability가 아니다.

**장문맥에 대해 RL stage가 놓이는 위치.** [[minimax-01]]은 하나의 긴 stage를 연속 수행하지 않고 post-training 전체에서 길이를 번갈아 쓴다. 순서는 SFT 8,192 → long-context prompt 50%를 포함한 1,032,192-token training → DPO 8,192 → DPO 1,032,192 → online RL 8,192이며, 전 구간에서 RoPE base 10M을 사용한다(§5.6, Table 7, [[minimax-01-recipe]]). [[qwen-long-context-synth]]는 Qwen2.5-1M에서 같은 비대칭을 보고한다. 길이 최대 8,192 token pair에 대한 offline RL만으로도 LongBench-Chat이 7B에서 7.32 → 8.08, 14B에서 8.56 → 8.76으로 올랐다. 반대로 [[kimi-k1-5]]는 RL 자체를 128K context까지 밀어 올리고 partial rollout으로 비용을 처리한다. token budget을 넘긴 rollout은 replay buffer에 저장해 다음 iteration에서 이어서 생성한다. 따라서 현재 segment에 대해서만 on-policy compute가 필요하다. 감지된 repetition은 일찍 종료한다(§2.6.2).

## §6 양방향의 regression 실행

**길이 중심 RL 뒤의 짧은 context regression.**
- [[scaling-reasoning-losing-control-mathif]]는 outcome reward를 전혀 주지 않고, 최대 rollout 길이만 바꾸어 너무 긴 rollout을 잘라낸 상태에서 DeepSeek-R1-Distill-Qwen-1.5B의 RL을 계속했다. hard-constraint accuracy / soft accuracy / math accuracy는 다음과 같았다. 1k: 19.05 / 39.88 / 28.73, 2k: 16.43 / 36.75 / 36.32, 4k: 16.91 / 35.87 / 40.03, 8k: 14.29 / 34.13 / 39.82(Table 5). 길이 budget은 instruction following과 math accuracy 사이를 교환하며, 각 단계에서 추세가 단조롭지는 않다.
- [[loongrl]]은 보존(retention) 측면을 보고한다. MMLU는 +2.8(7B), +1.1(14B), MATH-500은 76.0 → 78.0 및 83.4 → 83.2, IF-Eval은 −0.3 및 −2.6이었다(Table 2). 한편 QwenLong-L1-32B의 MMLU는 자기 base보다 낮았다(80.5 대비 78.5).
- [[qwenlong-l1-5]]는 long-context RL 뒤 MMLU-PRO 81.03 → 81.33, AIME24 90.31 → 90.0, AIME25 82.81 → 86.46, GPQA-Diamond 75.88 → 76.78, LongMemEval 60.80 → 76.40을 보고한다(Table 8). 이는 long-input RL이 short-context 능력을 반드시 희생시킬 필요는 없다는 가장 강한 공개 사례다. 다만 단일 모델의 결과이며 평가는 저자들 자신의 평가다.
- [[spell-self-play-long-context]]는 base 모델에 대해서만 short-context 전이를 보고한다. long-context self-play 뒤 Qwen2.5-7B/14B/32B는 math 및 MMLU-Pro suite 평균에서 각각 6.93 / 8.74 / 7.24점을 얻었다(App. F.2, Table 7). base 모델의 출발점이 낮으므로, 이 이득의 일부는 reasoning 자체보다 answer formatting일 수 있다(Interpretation).
- [[sky-t1-flash-overthinking]]는 math와 code pair로 응답을 짧게 만든 뒤 학습하지 않은 domain을 측정했다. GPQA Diamond는 56.8 → 56.6, MMLU는 82.4 → 81.7이었고, 길이는 각각 39%와 17% 줄었다.

**짧은 context 또는 specialized RL 뒤의 long-context regression.**
- memory 또는 agent 특화 RL stage는 full-context 능력을 잃을 수 있다. [[qwenlong-l1-5]] Table 10에서는 71.59 → 68.53으로 떨어졌고, merging으로 71.18까지 회복했다.
- 길이 중심 post-training은 long-input 능력을 측정하지 않은 채 남길 수 있다. [[minimax-01]]은 “NIAH is inadequate for effectively monitoring the model's performance throughout the training process”라고 적는다. 그 이유는 “NIAH metric performance reaches its peak score early on, specifically within the initial 128K training steps”이기 때문이다. 따라서 저자들은 중간 checkpoint를 더 어려운 task로 평가했다(§4.2). NIAH만으로 만든 long-context gate는 regression을 발견하지 못한다.
- [[context-length-alone-hurts]]는 측정 자체에 대한 주의점이다. 모델이 evidence를 정확히 그대로 되말해도 입력 길이가 늘면서 accuracy는 13.9%--85% 떨어진다. 거의 모든 task-length cell에서 accuracy의 하락 폭은 retrieval score의 하락 폭보다 크다. retrieval-only gate는 long-context RL 실행이 실제로 얻은 것을 과대평가한다.

**여기서 따라오는 gate.** 길이 중심 RL stage를 매번 마친 뒤 다음을 실행한다. (1) 훈련 budget이 아니라 **배포** generation budget에서 short-suite panel을 실행한다. 여기에는 MMLU 또는 MMLU-Pro, GSM8K 또는 MATH, 그리고 IF-Eval 또는 MathIF 같은 instruction-following benchmark가 들어간다. (2) 훈련 길이보다 하나 더 긴 길이를 포함해 여러 길이에서 [[ruler]]-style suite를 실행한다. (3) NIAH는 포화되므로, LongBench-v2, MRCR, 또는 QA set처럼 retrieval만이 아닌 long task를 적어도 하나 넣는다. (4) [[inverse-scaling-test-time-compute]]의 protocol을 따라 같은 평가를 두세 개의 reasoning budget에서도 실행한다.

## §7 진단: entropy 옆에 무엇을 기록할 것인가

| 수량 | 정의 | 알려 주는 것 |
|---|---|---|
| 절단율(truncation rate) | 종료 이유가 stop token이 아니라 길이 제한인 rollout의 비율 | 1보다 작은 값에서 plateau를 향해 상승하면, [[demystifying-long-cot]] §4.1의 암묵적 길이 패널티가 작동하고 있음을 뜻한다 |
| 응답 길이 백분위수 | step별 p50 / p90 / p99를 정답과 오답으로 나누어 기록 | 데이터로 `L_max`를 정한다. p99가 `L_max`에 가까우면, 절단 label이 negative gradient를 지배한다는 뜻이다 |
| advantage 부호별 길이 | positive-advantage rollout과 negative-advantage rollout의 평균 길이 | [[dr-grpo]] §3.1의 길이 편향은 오답 응답이 정답 응답보다 더 빠르게 길어지는 형태로 나타난다 |
| 보상 구성 | correctness 항과 length 항을 따로 기록 | correctness 항은 평평한데 총보상만 떨어진다면 capability 손실이 아니라 길이 패널티 효과다 |
| 반복 통계 | 응답 내부 N-gram 반복률 | 길이에 보상 또는 비용을 부과하는 보상에서 문서화된 hack([[demystifying-long-cot]] §4.5)이다 |
| zero-variance group 비율 | `G`개 rollout이 모두 같은 보상을 받은 prompt의 비율 | 절단과 상호작용한다. 모든 rollout이 절단되면 group에는 gradient가 없다([[dapo]] §3.2) |
| 길이와 함께 보는 validation accuracy | 두 curve를 한 축에 함께 표시 | [[dapo]] §4.3은 이 쌍으로 실행이 악화되는지 판단한다. training reward만으로는 validation accuracy와 “little correlation”을 보인다 |

## 부정 sample과 부정 feedback

**네 종류 중 무엇인가.** 이 장은 그중 세 가지를 사용한다. (1) 오답으로 채점된 절단 rollout은, 실제로는 맞을 수도 있는 trajectory에 적용하는 **gradient로서의 negative**다. 이는 설정이 만들어 낸 false negative다. (2) overlong filtering과 `mask_truncated_completions`는 그 sample을 **신호 없음**으로 바꾼다. 이는 masking이지 negative가 아니다. (3) 길이 preference pair(Kimi long2short DPO, [[overthinking-o1-like-llms]], [[sky-t1-flash-overthinking]])는 **gradient로서의 negative**다. reject된 응답은 완전히 정답일 수 있지만 길이만 이유로 덜 선호된다.

**negative는 어디서 오며 어떻게 label되는가.** 절단 label은 verifier가 아니라 sampler가 만든다. 절단 rollout 중 실제로 정답이었을 비율을 보고한 source는 없으므로, 이 label의 false-negative rate는 측정되지 않았다(Open question). 길이-preference negative는 policy 자체 sample에서 온다. 정답이 하나도 없는 sample은 버린 뒤 10개 sample 중 가장 긴 응답([[overthinking-o1-like-llms]] §3.1), 8개 중 가장 긴 정답과 가장 짧은 오답([[sky-t1-flash-overthinking]] Stage 1), 또는 선택 응답보다 1.5배 긴 정답과 오답([[kimi-k1-5]] §2.4)을 reject로 쓴다.

**작동 기제.** sample된 sequence의 log-probability를 낮추는 모든 objective에서 softmax logit gradient는 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. `z_j`는 token `j`의 logit, `p_j`는 그 확률이다. token `y`를 낮추면 `z_y`는 내려가고 다른 모든 logit은 `p_j`에 비례해 올라간다. 따라서 제거된 mass는 선택된 응답일 필요가 없는 현재의 고확률 alternative로 간다. 길이로 정의된 negative에서는 이것이 정확한 위험이다. reject trajectory가 올바른 추론이므로, 그로부터 뺀 mass는 모델이 보존해야 할 추론 행동에서 빠진다. 유도는 [[ch-43a]]에 있으며, 이 장은 이를 적용한다.

**source에서 쓴 control.**
- label이 신뢰할 수 없으면 *penalize 대신 mask*한다. [[dapo]]의 Overlong Filtering, Olmo 3 RL-Zero의 instruction-following 및 general script에 있는 `mask_truncated_completions`([[allenai-olmo3-open-instruct-scripts]]), timeout 또는 step 소진 agent trajectory에 대한 compact filtering([[deepswe]])이 이에 해당한다.
- *penalty를 상한으로 묶는다.* soft overlong punishment는 최대 1 reward unit이고 hard limit에서만 그 값에 도달한다([[dapo]] Eq. 13). Magistral의 length penalty는 0에서 −0.1이며 0.9 correctness 항과 함께 쓴다([[magistral-recipe]]).
- *오답을 짧다는 이유만으로 보상하지 않는다.* Kimi는 incorrect response에 `min(0, λ)`를 쓴다(§2.3.3). LCPO-Max는 length 항에 correctness indicator를 곱한다([[l1-lcpo]] Eq. 2).
- *두 번째 negative 유형을 추가한다.* [[sky-t1-flash-overthinking]]는 1K short-incorrect pair를 추가했고, length-only pair 대비 AIME24를 36.7에서 43.3으로 복원했다.
- *entropy로 negative를 gate한다.* [[qwenlong-l1-5]]의 AEPO는 batch entropy가 `H_high`를 넘으면 모든 negative-advantage sample을 mask하고 `H_low` 아래에서 복원한다. ablation에서 high-entropy negative sequence를 clipping한 경우 평균 57.36이었고 low-entropy clipping은 55.47--56.66이었으며, 저자들은 “too many negative gradient signals”를 제거하면 entropy collapse가 날 수 있다고 보고한다(§4.3, Table 4). Qwen3-4B-Thinking-2507에서 AEPO는 GRPO보다 평균 3.29점 높다(Table 5).

**효과 크기를 정직하게 말하면.** 이 장의 어느 source도 long 또는 truncated sample의 negative gradient가 개선분 중 얼마를 차지하는지 측정하지 않는다. 보고된 가장 큰 길이 관련 효과는 negative의 *제거*다. truncated sample masking만으로 +6 AIME점([[dapo]] Table 1, 누적 단일 run)이었다. AEPO 비교는 알고리즘 변경이지 분해 실험이 아니다. “negative signal이 gain을 만든다”는 주장은 여기서는 뒷받침되지 않는다.

**이 절의 진단.** 길이-preference training에서는 chosen/rejected log-probability를 따로 기록한다. advantage 부호별 길이와 reward를 기록한다. 길이 단축이 pass@1은 유지하면서 coverage를 좁힐 수 있으므로 큰 `k`의 pass@k를 추적한다. [[spell-self-play-long-context]]가 반대 방향의 coverage 증가를 보이기 위해 pass@8을 명시한 이유도 이것이다.

**generality에 미치는 효과.** 두 self-training 연구에서 길이-only negative는 가장 어려운 benchmark에서 underthinking을 만들었다(AIME24 36.7, LCB-Hard 13.5: [[sky-t1-flash-overthinking]]; FCS positive는 MATH500을 93.0에서 91.0으로 하락: [[overthinking-o1-like-llms]]). 어느 연구도 refusal 또는 calibration 효과를 재지 않아 length shaping의 abstention 측면은 측정되지 않았다(Open question).

## 설정 recipe

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DAPO Qwen2.5-32B base | 32B | RL | 최대 생성; 기대 최대; soft-punish cache | 20,480; 16,384; 4,096 | arXiv:2503.14476v2 §4.1 ([[dapo]]) | 2026-09-15 검증 | Table 1: +overlong filtering 30 → 36; +soft overlong punishment 38 → 41(누적, 단일 run) |
| DAPO Qwen2.5-32B base | 32B | RL | prompt × sample; mini-batch; LR; warmup; ε_low/ε_high | 512 × 16; 512; 상수 1e-6; rollout 20 step; 0.2 / 0.28 | v2 §4.1 | 2026-09-15 검증 | Table 1(clip-higher 36 → 38) |
| verl DAPO recipe | n/a | RL | `data.max_response_length`; `overlong_buffer.len`; `penalty_factor` | 20480; 4096; 1.0 | verl@753aed3 `docs/algo/dapo.md` ([[verl-dapo-overlong-buffer]]) | 2026-09-15 검증 | reproduction table: 52% AIME 2024(16×8×H800) |
| DAPO reproduction script(Flash-RL post) | 32B | RL | 최대 prompt / response; overlong buffer; penalty | 2,048 / 20,480; 4,096; 1.0 | [[rollout-training-mismatch-tis]] L20-24 | 2026-09-14 검증 | ablation 미보고 |
| DeepSeek-R1-Zero | 671B MoE | RL | 최대 응답 길이 | 8.2k step 전 32,768; 후 65,536 | v2 §2.1 ([[deepseek-r1-recipe]]) | 2026-09-14 검증 | §2.1: 8.2k step에서 성능과 길이가 “significant jump” |
| DeepSeek-R1(Dev1 → Dev2) | 671B MoE | RL | temperature; 출력; 최대 길이; step당 질문 | 1; 16; 32,768; 32 | v2 §3.2.1 | 2026-09-14 검증 | ablation 미보고 |
| DeepSeek-R1 | 671B MoE | eval-gate | 최대 output token; decoding | 32,768; temperature 0.6, top-p 0.95, k개 sample의 pass@1 | v2 D.1 | 2026-09-14 검증 | greedy decoding은 repetition이 더 높음 |
| Kimi k1.5 | 미보고 | RL | 길이 보상 | `λ = 0.5 − (len(i) − min_len)/(max_len − min_len)`; 정답이면 λ, 오답이면 min(0, λ); weight 미공개 | arXiv:2501.12599v4 §2.3.3 ([[kimi-k1-5-recipe]]) | 2026-09-14 검증 | ablation 수치 없음; 저자들은 penalty가 초기 학습을 늦출 수 있다고 설명 |
| Kimi k1.5 | 미보고 | RL | length-penalty schedule; 최대 RL context | 처음 off, 이후 상수(switch point 미공개); 128k | §2.3.3, §3.3 | 2026-09-14 검증 | Figure 5--6(더 작은 internal model) |
| Kimi k1.5 short-CoT | 미보고 | RL | long2short RL | 최고 performance/token checkpoint에서 시작; length penalty; 줄인 최대 rollout 길이(값 미공개) | §2.4, §3.4 | 2026-09-14 검증 | §3.4: 3,272 token에서 AIME 2024 60.8; Figure 7 최고 token efficiency |
| Kimi k1.5 short-CoT | 미보고 | preference | long2short DPO pair | chosen=가장 짧은 정답; rejected=오답 및 1.5배 더 긴 정답 | §2.4 | 2026-09-14 검증 | Figure 7(long2short RL 아래) |
| Magistral Medium | 미보고 | RL | 길이 penalty; penalty 없는 길이; batch | 0--−0.1(Eq. 1); `l_max − l_cache` 16k → 24k → 32k; 8k → 4k → 2k sequence | arXiv:2506.10910v1 §2.2, §5.2 ([[magistral-recipe]]) | 2026-09-14 검증 | KV-cache memory가 이유; penalty ablation 없음 |
| Olmo 3 7B Think / Instruct / RL-Zero math | 7B | RL | 응답 길이 | 32,768 / 8,192 / 16,384 | open-instruct olmo3 scripts ([[allenai-olmo3-open-instruct-scripts-recipe]]) | 2026-09-14 검증 | model type에 맞춤; ablation 미보고 |
| Olmo 3 RL runs | 7B, 32B | RL | truncated-rollout 처리 | `7b_rlzero_code.sh`, `7b_rlzero_instruction_following.sh`, `7b_rlzero_general.sh`에서 `mask_truncated_completions` True; Think, Instruct, `7b_rlzero_math.sh`에서는 False | `open_instruct/data_loader.py` L1126-1132; 각 script L70/L69/L77 ([[allenai-olmo3-open-instruct-scripts]]) | 2026-09-14 검증 | ablation 미보고 |
| Nemotron 3 Nano 30B-A3B | 31.6B / 3.2B active | RL | 최대 생성 길이; 절단 처리 | 49K; overlong filtering | NR §3.2.5 ([[nemotron-ultra-recipe]]) | 2026-09-14 검증 | ablation 미보고 |
| Demystifying-long-CoT run(Llama-3.1-8B) | 8B | RL | 생성 길이; cosine reward; repetition penalty | 14,336; r0^c +2, rL^c +1, r0^w −10, rL^w 0, r_e −10; P −0.05, N 40 | arXiv:2502.03373v1 App. E.5.1--E.5.3 ([[demystifying-long-cot]]) | 2026-09-15 검증 | Figure 4(고전 보상 대비 안정성); Figure 6(동일 sample에서 8K > 4K, 16K) |
| L1-Exact / L1-Max(DeepScaleR-1.5B-Preview) | 1.5B | RL | train / eval context; step; α; δ; n_gold 범위; LR; batch | 4K / 8K; 700 뒤 120; 0.0003; 0.5; U(100, 4000); 1e-6; 128 | arXiv:2503.04697v2 §4 ([[l1-lcpo]]) | 2026-09-15 검증 | §5: math 평균 length error 약 3%; 평균 budget violation 1.3%; 광범위한 hyperparameter tuning은 하지 않음 |
| QwenLong-L1-14B / -32B | 14B, 32B | RL | phase(입력); 최대 출력; rollout; batch; LR | 20K 뒤 60K; 10K; 8; 128(mini 32); 2e-6 | arXiv:2505.17667v2 §3.2 ([[qwenlong-l1]]) | 2026-09-15 검증 | Table 4(평균 +4.1 / +5.1); Figure 5(phased vs single-stage) |
| QwenLong-L1.5-30B-A3B | 30B-A3B | RL | stage 입력 / 출력 길이 | 20K/12K → 60K/20K → 120K/50K; memory-RL 128K 입력, 32K chunk, 15K memory; Stage-4 120K/50K | arXiv:2512.12967v1 §4.1, Figure 6 ([[qwenlong-l1-5]]) | 충돌: Figure 6은 Stage-1 입력 32K, §4.1은 20K | Table 10: 61.92 → 69.59 → 70.46 → 71.59 → (merge) → 71.82 |
| QwenLong-L1.5-30B-A3B | 30B-A3B | RL | algorithm; group 크기; batch; LR; sampling | AEPO(entropy-gated negative-gradient masking); G = 8; 128; 상수 2e-6; T 0.7, top-p 0.95 | §4.4, §5.1 | 2026-09-15 검증 | Table 5: Qwen3-4B-Thinking-2507에서 56.07(GRPO) → 59.36(AEPO) |
| LoongRL-7B / -14B | 7B, 14B | RL | 입력 약; 최대 출력; G; β; LR; batch | ~16K; 4,096; 8; 0.001; cosine 1e-6; 512 / 256 | arXiv:2510.19363v2 §4.1 ([[loongrl-recipe]]) | 2026-09-14 검증 | Table 4: KeyChain 72.4 vs plain long QA 66.2(7B) |
| RL-MemAgent-7B / -14B | 7B, 14B | RL | 8K window 분할; algorithm; group 크기; rollout batch | 1,024 query + 5,000 chunk + 1,024 memory + 1,024 output; Multi-Conv DAPO, std division 없음; 16; 256 | arXiv:2507.02259v2 §3.1, App. A.3 ([[memagent]]) | 2026-09-15 검증 | Table 1: 3.5M token에서 71.09 |
| SPELL(Qwen2.5/Qwen3 base) | 7B--30B | RL | 최대 입력; 최대 출력; G; batch; LR | 16K; 4K(reasoning model은 20K); 8; 128; 상수 2e-6 | arXiv:2509.23863v4 §4.1 ([[spell-self-play-long-context]]) | 2026-09-15 검증 | Table 1(16K/100K); Table 2(verifier ablation −3.2) |
| Sky-T1-32B-Flash | 32B | preference | loss; pair; LR; γ; β; epoch | SimPO; 10K(1K short-incorrect, 500 TACO 포함); 5e-7; 0.3; 2.0; 1 | [[sky-t1-flash-overthinking]] Stage 1--3 | 2026-09-14 검증 | Ablation: math-only length pair AIME24 36.7 vs short-incorrect pair 포함 43.3 |
| QwQ-32B-Preview + SimPO FCS+Reflection | 32B | preference | prompt당 sample; temperature; positive; negative | 10; 1.0; 첫 정답 solution + reflection 하나; 가장 긴 sample 응답 | arXiv:2412.21187v2 §3.1--§3.2 ([[overthinking-o1-like-llms]]) | 2026-09-15 검증 | Table 4: MATH500 1,330.7 token에서 92.8 vs 2,407.9 token에서 93.0 |
| MiniMax-Text-01 | 456B / 45.9B active | RL | post-training stage 길이 | SFT 8,192 → 1,032,192(긴 prompt 50%) → DPO 8,192 → DPO 1,032,192 → online RL 8,192 | arXiv:2501.08313v1 §5.6, Table 7 ([[minimax-01-recipe]]) | 2026-09-14 검증 | ablation 미보고 |

**작은 범용 run의 출발점.** 여기의 모든 수치는 위 검증된 행에서 왔다. math 또는 code verifier를 쓰는 32B급 base의 reasoning RL에는 DAPO 설정이 가장 상세히 문서화되어 있다. 최대 생성 20,480 token은 기대 길이 16,384와 penalty factor 1.0의 soft-punish buffer 4,096으로 이루어진다. 512 prompt × 16 sample, mini-batch 512, warm-up rollout step 20회의 상수 LR 1e-6, ε_low 0.2와 ε_high 0.28, token-level loss aggregation을 쓰며 verl에서 16×8×H800으로 실행했다. 짧게 유지해야 하는 instruct-style model에는 검증된 Olmo 3 Instruct 설정이 대안이다. 응답 8,192 token, 64 prompt × 8 sample, 상수 LR 1e-6, β = 0.0(KL term 없음)다. 이 run은 `mask_truncated_completions`를 False로 두므로 원한다면 절단 masking을 의도적으로 추가해야 한다. 긴 입력에는 QwenLong-L1의 phase 구조(입력 20K 뒤 60K, 출력 10K, rollout 8, batch 128, LR 2e-6, 32×A100-80G, 먼저 warm-up SFT)가 가장 작은 검증 long-context RL recipe다. 목표 길이가 60K를 넘으면 [[qwenlong-l1-5]]처럼 입력과 출력을 함께 늘린다.

## 일반화 관점

**(a) 무엇이 범위를 넓히는가.**
- 절단 false negative 제거: masking만으로 +6 AIME점([[dapo]] Table 1)을 얻었고, soft punishment 아래에서도 긴 정답 추론을 positive example으로 남긴다.
- serving에서 강제하는 대신 budget에 조건화: LCPO의 길이 제어는 그 task가 RL 데이터에 없어도 GPQA, LSAT, MMLU로 전이됐다([[l1-lcpo]] §5, Figure 3).
- 어려운 예제를 보존하는 phased input-length scaling: 7개 DocQA benchmark 평균이 +4.1 / +5.1점이었고 SFT만으로는 +0.8 / +3.2였다([[qwenlong-l1]] Table 4).
- 원시 길이보다 task 난이도: 같은 크기의 plain long QA보다 KeyChain data가 6.2점 높았으며([[loongrl]] Table 4), [[qwenlong-l1-5]] Table 8에서 long-context RL은 AIME25(+3.65), LongMemEval(+15.60)로 전이됐다.
- responder 성공률을 0.5 근처에 두는 self-play curriculum: 같은 base에서 RLVR 68.1 대비 pass@8 74.5([[spell-self-play-long-context]] §4.3).

**(b) 무엇이 narrowing 또는 forgetting을 만드는가.**
- 길이에 돈을 주는 reward: 충분한 compute에서 repetition hacking과 branching frequency 하락([[demystifying-long-cot]] §4.5), 그리고 `r0^c < rL^c`일 때 explosive growth(§4.3).
- 길이-only negative: hardest set에서 underthinking(AIME24 43.3 → 36.7; LCB-Hard 17.9 → 13.5, [[sky-t1-flash-overthinking]] Ablations table).
- instruction-following gate 없이 rollout budget 올리기: 1k에서 8k로 갈 때 hard-constraint accuracy 19.05 → 14.29([[scaling-reasoning-losing-control-mathif]] Table 5).
- 한 길이 regime만 특화: merging 전 memory-RL은 full-context 평균 3.06점을 잃었고([[qwenlong-l1-5]] Table 10), QwenLong-L1-32B는 base의 MMLU 80.5에 비해 78.5였다([[loongrl]] Table 2).
- 더 많은 생각이 항상 낫다고 가정: constructed task와 natural overthinking 아래 nine model 전체의 Zebra Puzzle에서 inverse trend([[inverse-scaling-test-time-compute]] Table 1).

**(c) 이 단계에서 generality를 재는 법.** 배포 generation budget에서 short suite를 실행한다. 훈련 길이보다 긴 길이를 포함한 여러 길이에서 [[ruler]]-style retrieval 및 retrieval만이 아닌 long task를 하나 이상 실행한다. instruction-following benchmark, 큰 `k`의 pass@1과 pass@k, [[inverse-scaling-test-time-compute]]에 따른 두세 reasoning budget의 accuracy를 함께 재고, [[sky-t1-flash-overthinking]] 및 [[l1-lcpo]]처럼 모든 accuracy 옆에 token count를 보고한다. 알려진 측정 오류는 다음과 같다. NIAH는 빨리 포화한다([[minimax-01]] §4.2). Retrieval score는 long-context 능력을 과대평가한다([[context-length-alone-hurts]]). [[loongrl]]은 RL seed와 evaluation-set overlap을 검사하지 않았다. L1의 length error는 domain 밖에서 20--40%까지 커지므로 math에서 확인한 budget이 검증 없이 전이되지는 않는다.

## 흔한 실수와 발견 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| 절단 rollout을 오답으로 남김 | 절단율이 1 아래에서 plateau하는데 training accuracy는 하락; 길이가 `L_max`에 고정 | 종료 응답과 별도로 절단율·accuracy 기록([[demystifying-long-cot]] §4.1; [[dapo]] §3.4) |
| 데이터를 보지 않고 context window로 `L_max` 선택 | 몇 sample이 rollout step 시간을 지배; KV-cache 압박으로 batch를 줄임 | p99 응답 길이와 `L_max` 비교; Magistral은 `l_max` 증가 시 batch를 줄임([[magistral-recipe]]) |
| step 0부터 길이 penalty 활성화 | 초기에 reward가 평평, 짧은 degenerate 답 | Kimi는 penalty 없이 훈련 후 상수 penalty 추가([[kimi-k1-5]] §2.3.3) |
| 더 긴 정답에 더 많이 주는 길이 reward | accuracy 없이 길이 증가; stop-token 확률 하락 | `r0^c`와 `rL^c` 검사; `r0^c < rL^c`에서 explosive growth([[demystifying-long-cot]] §4.3) |
| 길이 보상 reward에 repetition control 없음 | N-gram repeat rate 상승; branching word 하락 | N-gram penalty(`P = −0.05`, `N = 40`)를 넣고 repeat rate 기록([[demystifying-long-cot]] §4.5) |
| 한 domain에서만 훈련한 budget control | math에서는 budget 유지, 다른 곳에서는 실패 | out-of-domain length error 측정; L1은 20--40% 보고([[l1-lcpo]] §5) |
| long-correct negative만으로 단축 | easy set은 좋아도 hardest benchmark accuracy 하락 | short-incorrect pair 추가; AIME급 점수 비교([[sky-t1-flash-overthinking]] Ablations) |
| heavy truncation에서 sample-level loss | 긴 오답이 정답보다 더 빨리 자람 | advantage 부호별 길이 기록; [[dr-grpo]] §3.1 length bias |
| NIAH로 long-context RL 판정 | needle 점수만 포화, 실제 task는 개선 없음 | LongBench-v2 / MRCR / CorpusQA-style task 사용([[minimax-01]] §4.2; [[qwenlong-l1-5]] Table 7) |
| merge/mixed stage 없이 specialized long-context stage 배포 | 특화 stage 뒤 full-context 평균 하락 | stage checkpoint 비교; merge 후 재측정([[qwenlong-l1-5]] Table 10) |
| rollout 길이를 올리고 IF 재실행 안 함 | math는 상승, constraint compliance 하락 | budget별 MathIF 또는 IF-Eval([[scaling-reasoning-losing-control-mathif]] Table 5) |

## 이해 점검

1. 한 run이 binary reward, token-level loss aggregation, 길이 항 없음을 쓴다. 절단율이 18%이며 증가 중이다. advantage 계산과 token별 weighting을 사용하여, 왜 절단 rollout이 짧은 오답보다 더 큰 negative gradient mass를 갖는지 설명하고, loss를 sample별로 aggregate하면 무엇이 바뀌는지 설명하라.
2. DAPO는 overlong filtering만으로 AIME 2024 avg@32가 30에서 36으로 올랐고, 최종 run에는 soft overlong punishment를 남겼다고 보고한다. 둘 다 punitive default보다 개선일 수 있는 인과적 설명을 하고, 새 모델에 무엇을 쓸지 결정할 실험을 제시하라.
3. [[demystifying-long-cot]]는 명시적 penalty term이 없어도 context window가 길이 penalty로 작동한다고 보고한다. group-normalized reward에서 절단 rollout이 길이에 대한 downward pressure로 이어지는 경로를 설명하고, exceed rate가 왜 1에 도달하지 않고 1 미만에서 plateau하는지 말하라.
4. Kimi의 length reward는 정답에 `λ`, 오답에 `min(0, λ)`를 준다. 풀지 못한 문제에서 policy가 어떤 행동을 학습할 수 있는지로 설명하며, 오답도 full `λ`를 받으면 무엇이 잘못되는지 설명하라.
5. L1-Exact는 길이 제약 없는 같은 모델보다 약 1점 낮고 gap은 AIME에 집중된다. LCPO-Exact와 LCPO-Max의 차이로 이를 설명하고, 난이도가 매우 다양한 문제의 benchmark에서는 둘 중 어느 것이 덜 하락할지 예측하라.
6. LoongRL은 약 16K 입력으로 훈련해 128K RULER를 개선했고, QwenLong-L1은 입력 길이를 phase로 60K까지 늘린다. 각 전략이 더 적합한 조건을 제시하고, 16K 전략이 전이되지 않았음을 training curve에서 어떻게 알지 설명하라.
7. QwenLong-L1.5의 memory-RL stage는 memory-agent MRCR을 올리고 full-context 평균을 내렸으며, merge는 memory gain을 유지하며 평균을 복원했다. 이 순서가 두 능력이 어디에 저장되는지에 대해 뜻하는 바를 설명하고, “merging이 모든 것을 복구한다”가 한 run이 아닌 일반 주장이라면 결과가 어떻게 나와야 하는지 설명하라.
8. 팀이 길이-preference pair로 reasoning model을 짧게 만들고 MATH500 pass@1을 유지한 채 배포한다. pass@1이 숨기는 손실을 보일 측정 두 가지와, 각각이 겨냥하는 기제를 설명하라.

## 연결

- **이전:** [[ch-44]] — Process Supervision and Verifiable Rewards. 이 장이 변형하는 reward는 그 장에서 정의한 verifiable outcome reward이며, truncation은 verifier가 검사할 것이 없는 경우다.
- **다음:** [[ch-44b]] — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. reward가 verifier가 아니라 model judge이면 judge 자신도 length preference를 가지므로 length term은 다르게 동작한다.
- **기반:** [[ch-32b]] — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression(이 장이 소비하는 window); [[ch-40]] — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO(§2의 loss aggregation과 advantage formula); [[ch-43]] — Entropy, Output Diversity, and KL Control in RL(길이 증가 옆의 entropy collapse); [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages(이 장 negative 절에 적용한 유도).
- **후속:** [[ch-45b]] — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability; [[ch-45c]] — Context Management for Long-Horizon Agents; [[ch-32c]] — Claimed versus Effective Context Length and Long-Context Evaluation(§6의 측정 suite); [[ch-29a]] — Long-Document Synthesis for Continued Pretraining and Long-Context SFT(§5 long-input RL data의 출처).

## 출처

- [[dapo]] — overlong filtering 및 soft overlong punishment, Eq. 13, Table 1 ladder, token-level loss, 길이를 training metric으로 쓰는 법, RL setting.
- [[verl-dapo-overlong-buffer]] — 공개된 `overlong_buffer` configuration과 penalty code, overlong filtering을 거기 구현하지 않은 이유에 대한 FAQ.
- [[deepseek-r1]], [[deepseek-r1-recipe]] — 8.2k step에서 최대 응답 길이 32,768 → 65,536, evaluation budget, 문제 난이도별 thinking-token count, 단순 질문 overthinking.
- [[kimi-k1-5]], [[kimi-k1-5-recipe]] — 길이 reward, long2short 방법과 결과, 128K partial rollout, model-size별 token efficiency 비교.
- [[magistral]], [[magistral-recipe]] — length penalty 크기, batch-size 감소가 있는 `l_max − l_cache` schedule, weight space에서 reward와 length의 방향.
- [[dr-grpo]] — negative advantage에서 per-sample normalization의 length bias.
- [[demystifying-long-cot]] — correctness-only reward의 길이 불안정성, cosine reward formula/hyperparameter, context-window 비교, 길이 reward hacking과 repetition penalty.
- [[l1-lcpo]] — LCPO-Exact/LCPO-Max reward, budget adherence, budget forcing 비교, short-CoT 결과.
- [[overthinking-o1-like-llms]] — outcome efficiency metric, FCS/FCS+Reflection positive, 가장 긴 response negative, Table 4 accuracy·length 결과.
- [[sky-t1-flash-overthinking]] — SimPO length pair, short-incorrect ablation, 7 benchmark의 accuracy와 length.
- [[tokenskip]] — 제어 가능한 chain-of-thought compression의 supervised 대안.
- [[inverse-scaling-test-time-compute]] — 긴 추론일수록 accuracy가 낮아지는 task와 multi-budget evaluation protocol.
- [[scaling-reasoning-losing-control-mathif]] — instruction-following accuracy에 대한 RL rollout-length sweep(Table 5).
- [[qwenlong-l1]] — long-context RL dynamics, curriculum phase, difficulty-aware retrospective sampling, hybrid reward, SFT와 RL의 trade-off.
- [[qwenlong-l1-5]] — joint input/output length schedule, AEPO entropy-gated negative gradient, memory-RL specialization 및 SCE merging, out-of-distribution generality table.
- [[loongrl]], [[loongrl-recipe]] — KeyChain data, 16K training/128K evaluation, short-context retention, RL setting.
- [[memagent]] — chunked memory의 8K training window, Multi-Conv DAPO advantage sharing, 3.5M token까지 결과.
- [[spell-self-play-long-context]] — self-play role, Gaussian questioner reward, 16K → 100K transfer, pass@8, base model의 short-context transfer.
- [[randomized-yarn]] — 길이 generalization을 위한 supervised route로서 length curriculum을 쓴 randomized YaRN position.
- [[longer-context-deeper-thinking]] — SFT 이후 long reasoning의 선행 조건으로서 long-context ability.
- [[minimax-01]], [[minimax-01-recipe]] — 8,192와 1,032,192 token을 번갈아 쓰는 post-training stage, NIAH saturation.
- [[qwen-long-context-synth]] — long-context SFT 뒤 short-only offline RL과 LongBench-Chat gain.
- [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]] — `mask_truncated_completions`와 model type별 response length.
- [[nemotron-ultra-recipe]] — 2025-12 RLVR recipe의 overlong filtering 포함 49K generation length.
- [[deepswe]] — timeout 또는 step 소진 agent trajectory의 compact filtering.
- [[demystifying-agentic-rl]] — 길이 미보고 상태로 agentic RL에서 재사용한 overlong reward shaping.
- [[rollout-training-mismatch-tis]] — overlong-buffer setting이 있는 공개 DAPO reproduction script.
- [[grpo]] — worked example에서 쓰는 group-normalized advantage.
- [[ruler]] — length-wise gate로 쓰는 synthetic long-context suite.
- [[context-length-alone-hurts]] — retrieval이 완벽해도 input length에 따라 accuracy가 하락함.
