<!-- chapter: ch-03
     track: foundations
     kind: content
     title: 학습률 스케줄, 배치 크기, 초기화, 정규화
     deps: [ch-02]
     sources: [[lr-schedules]], [[weight-init]], [[batch-vs-layer-norm]], [[minicpm]], [[critical-batch-size-pretraining]], [[large-batch-training-noise-scale]], [[deepseek-llm]], [[olmo-2]], [[llama-3]], [[llama-3-recipe]], [[overtrained-lms-harder-to-finetune]], [[cooldown-scaling-beyond-fixed-durations]], [[cooldown-scaling-beyond-fixed-durations-recipe]], [[small-scale-proxies-instabilities]], [[resolving-scaling-discrepancies]], [[resolving-scaling-discrepancies-recipe]], [[datadecide]], [[continual-pretraining-rewarm-replay]], [[prolong]], [[prolong-recipe]], [[kimi-k1-5-recipe]], [[yarn]], [[rope-nope-hybrid-attention]], [[llama-4]], [[qwen-2.5]]
     figures: figures/lr-schedules.html
     revised: 2026-09 (generality revision)
-->

# Chapter 3 — 학습률 스케줄, 배치 크기, 초기화, 정규화

> **핵심 통찰.** 학습률 스케줄의 마지막 단계는 loss와 downstream score에 측정 가능한 영향을 미친다. WSD와 cooldown 스케줄에서는 앞선 constant phase보다 decay 동안 loss가 더 빠르게 떨어진다(MiniCPM §4.3; Hägele et al. 그림 3). Constant learning rate에서 20% cooldown은 1B parameter에서 8개 downstream task의 길이 일치 cosine과 같았다(46.20 대 46.26; [[cooldown-scaling-beyond-fixed-durations]]). Decay 중 데이터 변경도 변경하지 않은 mixture에서의 decay와 비교해 측정할 수 있다. OLMo 2 7B의 10-task 평균은 mid-training 뒤 53.0에서 62.9로 올랐고(50B-token anneal 3회의 평균), held-out MMLU-Pro도 27.4 → 31.0이었다. 다만 변경 없는 mixture의 anneal도 최종 mixture가 OLMES에서 얻은 6.1점 향상 중 4.4점을 설명한다([[olmo-2]], 표 9, 11). Batch size·초기화·정규화는 안정한 learning rate를 정한다. Critical batch size는 85M–1.2B에서 model size보다 학습한 token에 주로 따라 증가했고([[critical-batch-size-pretraining]]), qk-layernorm과 z-loss는 최대 1.2B 모델이 3e-4~0.3 learning rate에서 낮은 loss까지 학습하게 했다([[small-scale-proxies-instabilities]]). 이 선택은 후속 단계가 바꿀 수 있는 범위도 제한한다. 3T token으로 사전학습한 OLMo-1B checkpoint는 instruction tuning 뒤 2.3T-token checkpoint보다 여러 표준 benchmark에서 2% 넘게 나빴다([[overtrained-lms-harder-to-finetune]]).
>
> **지침.** 최종 token budget이 고정되지 않았거나 한 run에서 여러 budget이 필요하면 warmup 뒤 constant learning rate를 사용하고 저장 checkpoint에서 token의 10–20%에 해당하는 decay branch를 실행하라. MiniCPM과 Hägele et al.이 이 방식으로 cosine을 맞췄기 때문이다([[minicpm]], [[cooldown-scaling-beyond-fixed-durations]]). 그렇지 않으면 실제로 학습할 token에 맞춘 cosine을 사용하라. Decay 중 고품질 또는 target 데이터를 더할 때는 변경 없는 mixture의 anneal과 비교하고 held-out suite를 보고하라. Learning-rate decay만으로 OLMo 2 7B OLMES가 69.6 → 74.0이었기 때문이다([[olmo-2]], 표 11). 큰 run용 small-scale tuning에서는 parametrization(μP 또는 적합한 learning-rate·batch rule)을 먼저 고정하고 각 proxy를 learning-rate crossover를 지나도록 충분히 길게 실행하라. OLMo 2의 3e-4 대 6e-4 crossover는 200B token 뒤에 왔다([[olmo-2]], §4.1). 높은 learning rate에서 깊은 pre-norm Transformer를 학습하면 QK-norm과 z-loss를 넣고 maximum attention logit과 gradient RMS를 추적하라. 이 값들이 작은 모델에서 4.8B divergence를 예측했기 때문이다([[small-scale-proxies-instabilities]]). Long-context retrieval이 목표이면 QK-norm이 attention을 평탄화하지 않았는지도 평가하라([[rope-nope-hybrid-attention]]).

## 이 장이 범용 모델에 중요한 이유

사전학습은 이후 모든 단계가 수정하는 representation을 정한다. 이 장은 사전학습의 안정성과 mid-training, SFT, preference optimization, RL에 남기는 capability를 정하는 네 설정, 즉 learning-rate schedule, batch size, initialization, normalization layer를 다룬다. Optimizer·precision 선택(ch-01, ch-02)과 scaling-law·data 장(ch-08a 이후) 사이에 위치하며, mid-training(ch-32), continual pre-training(ch-32a), context extension(ch-32b)이 쓰는 schedule 어휘를 제공한다.

문제는 측정 가능하다. Loss spike 또는 divergence는 연산을 낭비하고 더 이른 checkpoint에서 재시작하게 할 수 있다. 잘못된 시점에 decay하는 schedule은 올바른 시점의 decay가 제거했을 loss를 남긴다. Critical batch size를 넘는 batch는 같은 loss에 더 많은 token을 쓴다. 전이되지 않는 small-scale hyperparameter search는 large run에 잘못된 learning rate를 고른다. 범용 모델에서는 두 번째 질문이 있다. 많은 held-out task를 개선하는가, 아니면 development 때 쓴 benchmark만 개선하는가? 그리고 checkpoint를 나중에 쉽게 adapt할 수 있게 남기는가? 이 질문의 측정 규칙은 ch-00에 있다.

## §1 학습률 스케줄

**정의.** Learning-rate schedule은 step t에서 optimizer step size를 정하는 함수 η(t)다. 현대 LLM schedule은 warmup phase, main phase, decay phase를 가진다.

**문제.** Schedule에는 두 가지 측정 가능한 failure mode가 있다. 초기에 optimizer 통계와 activation이 충분히 scale되지 않은 상태에서 step size가 커 발생하는 불안정(loss spike 또는 divergence), 그리고 완결된 decay가 도달할 수 있는 값보다 높은 최종 loss(감소가 끝나기 전에 멈춘 run)다.

### 1.1 Warmup

**메커니즘.**
1. Warmup은 W step 동안 η를 0(또는 작은 값)에서 peak까지 선형으로 올린다. t < W에서 η(t) = η_peak · t / W다.
2. AdamW(ch-01)는 first-moment estimate를 second-moment estimate의 제곱근으로 나눈다. m과 v를 0으로 초기화해 생기는 과소추정을 bias correction m̂_t = m_t / (1 − β₁ᵗ), v̂_t = v_t / (1 − β₂ᵗ)가 제거한다.
3. t = 1에서 보정한 estimate는 m̂₁ = g, v̂₁ = g²이므로 update는 |g| ≫ ε인 모든 coordinate에서 −η · g / (|g| + ε) ≈ −η · sign(g)다. Coordinate당 step size는 gradient 크기와 무관하게 η이고, v̂는 sample 하나로 추정된다.

**계산 예시(도출됨).** β₁ = 0.9, β₂ = 0.95, ε = 1e-8이고 g = 0.01인 coordinate를 보자. m₁ = 0.001, v₁ = 5e-6, m̂₁ = 0.01, v̂₁ = 1e-4이므로 update는 η · 0.01 / (0.01 + 1e-8) ≈ η다. g = 1e-5인 coordinate는 η · 1e-5 / (1e-5 + 1e-8) = 0.999 η를 받는다. Warmup 없이 η = 3e-4이면 모든 weight는 step 1에서 약 3e-4만큼 움직인다. 이는 크기 0.02 weight(OLMo 2 init standard deviation)의 1.5%다. 2,000-step warmup에서는 step 1에 η = 1.5e-7을 사용한다. 초기 불안정이 이런 sign-like, high-variance 첫 update에서 나온다는 설명은 **해석**이고, 아래 증거는 경험적이다.

**증거.**
- IWSLT14 De-En의 Post-LN Transformer: warmup 없는 Adam은 BLEU 8.45, warmup이 있으면 “around 34”였다. T_warmup = 500에서 lr_max 5e-4의 BLEU는 31.16, 1e-3에서는 2.77이었다([[batch-vs-layer-norm]] 발췌; Xiong et al. 2020 §3.2, 그림 2). 결과(단일 연구).
- C4에서 2.4M~1.2B decoder-only model, 총 1e5 step: warmup 50~25,000 step을 시험했고 긴 warmup이 learning-rate sensitivity와 loss를 낮췄다. 특히 LR 0.3에서 긴 warmup 없이 안정하지 못한 큰 모델에서 효과가 가장 컸다([[small-scale-proxies-instabilities]], §3.2.1, 그림 5). 결과(단일 연구).
- OLMo 2는 spike를 재현하는 baseline을 “mainly [by] reducing the warmup period”로 만들었다([[olmo-2]], §3.2). 위 두 결과와 함께 재현됨: 짧은 warmup은 불안정을 높인다.

**조건과 한계.** Xiong et al.은 Pre-LN Transformer가 자신들의 과제에서 warmup 없이 학습한다고 보였으므로(§4), warmup 길이는 normalization placement(§5)와 상호작용한다. Large run이 사용한 warmup 길이(OLMo 2 2,000 step, Llama 3 405B 8,000)는 해당 보고서에서 ablation하지 않았다.

### 1.2 Cosine과 inverse square root

**수식.**

```
cosine:      η(t) = η_min + ½ (η_peak − η_min) (1 + cos(π · (t − W) / (T − W)))        for W ≤ t ≤ T
inverse sqrt: η(t) = d_model^-0.5 · min(t^-0.5, t · W^-1.5)
```

η_peak은 peak learning rate, η_min은 final learning rate, W는 warmup length, T는 schedule horizon, d_model은 model width다. Inverse square root 형태는 W = 4000인 Vaswani et al. §5.3 식 3이다([[lr-schedules]] 발췌).

**계산 예시.** d_model = 512인 inverse square root에서 step 4,000은 512^-0.5 · 4000^-0.5 ≈ 6.99e-4, step 16,000은 ≈ 3.49e-4다. Llama 3 405B의 cosine(η_peak 8e-5, η_min 8e-7, [[llama-3-recipe]])에서 decay 중간의 η는 8e-7 + ½ · 7.92e-5 = 4.04e-5다. Decay의 75%에서는 cos(0.75π) = −0.707이므로 η = 1.24e-5다. Final learning rate는 peak의 1%다. OLMo 2는 peak의 10%까지 decay한다(`alpha_f: 0.1`, [[olmo-2]] 발췌). 따라서 “min LR”은 고정된 비율이 아니라 run별 선택이다.

**Horizon.** MiniCPM의 0.036B 실험에서 S = 20N, 40N, 60N, 80N token에서 최저 loss는 “always achieved by the Cosine(T) where T = S. Both T < S and T > S are not optimal”이었다([[minicpm]], §4.1, 그림 4). Hägele et al.도 같은 양상을 보인다. 각 cosine은 자기 endpoint에서만 최고다([[cooldown-scaling-beyond-fixed-durations]], 그림 1). **재현됨.** 두 출처 모두 mismatch horizon의 단일 백분율 penalty는 제공하지 않는다. Penalty는 mismatch와 model에 따라 달라진다.

### 1.3 Warmup-stable-decay, cooldown, multi-step schedule

**정의.** WSD(warmup-stable-decay)는 warmup 뒤 η를 peak에서 유지한 다음 학습 마지막 일부에서 decay한다. WSD(T; s) = (s/W)η(s < W), η(W < s < T), f(s − T)η(T < s < S)이며 f는 0 < f ≤ 1인 감소 함수다([[minicpm]], §4.2, 식 1). Hägele et al.은 같은 모양을 “constant LR + cooldown”이라 부르고, f = 1 − √((n − (N − N_decay)) / N_decay), 즉 (1-sqrt) cooldown을 제안한다([[cooldown-scaling-beyond-fixed-durations]], §3.2). Multi-step schedule은 η를 이산적인 drop으로 낮춘다.

**이것이 해결하는 문제.** Cosine schedule은 T를 미리 고정해야 한다. 나중에 더 많은 데이터를 받는 run이나 여러 token count가 필요한 scaling study는 그렇지 않으면 budget당 full run 하나가 필요하다.

**증거.**
- Decay 중 loss: MiniCPM(0.036B)에서 loss는 decay stage에서 “experiences a significant rapid decline”하고 T = S의 cosine loss에 도달하거나 이를 넘는다(§4.3, 그림 5). Hägele et al.(210M)에서도 “the cooldown phase initiates a sharp decrease in loss”로 cosine을 맞춘다(그림 3). **재현됨.** 두 출처는 decay 동안 전체 loss 감소 중 어느 비율이 일어나는지 보고하지 않는다.
- Decay length: MiniCPM은 전체 token의 10% decay가 “sufficient to achieve the best results”이고 2.5%는 부족하다고 했다(§4.3). Hägele et al.은 124M model에서 약 20% step에서 이득이 plateau하고, 200k-step run에서 5% (1-sqrt) cooldown이 cosine에 거의 맞는다고 했다(그림 5–6). 10–20%에서 **재현됨**이지만, 가장 짧은 충분 길이는 두 연구에서 다르다.
- Downstream: FineWeb 100B token의 1B model은 8개 task 평균에서 cosine to 10% 46.26, (1-sqrt) 20% 46.23, linear 20% 46.20, cosine to 0 45.88이었다([[cooldown-scaling-beyond-fixed-durations]], 표 4). 결과(단일 연구, seed 보고 없음).
- Multi-step: DeepSeek LLM은 2,000 step warmup 뒤 token의 80%에서 peak의 31.6%, 90% 뒤 10%로 낮춘다. 100B token의 1.6B model에서 최종 결과는 cosine과 “essentially consistent”했다([[deepseek-llm]], §2.3, 그림 1a). WSD는 MiniCPM에서 나왔고 DeepSeek LLM은 WSD를 사용하지 않는다.

**계산 예시(인쇄된 표기의 해석).** MiniCPM은 f(s − T) = 0.5^((s−S)/T)로 decay하며, “T is set to be 5000 steps (20B tokens)”라고 쓴다([[minicpm]], §6.2). 이를 5,000 step half-life로 읽으면 η는 5,000 step 뒤 50%, 10,000 step 뒤 25%, log₂(10) · 5,000 ≈ 16,610 step 뒤 10%로 떨어진다. MiniCPM은 learning rate가 max learning rate의 10% 아래로 떨어진 뒤에도 loss가 “still drops”한다고 보고한다(§6.4).

[figures/lr-schedules.html](figures/lr-schedules.html) 도표는 발표된 이 schedule들을 하나의 normalized axis에 그린다. 독자가 stop point와 decay length를 움직일 수 있으며, 그 stop point에서 각 schedule의 learning rate를 표에서 읽을 수 있다. 두 번째 panel은 §3.1의 batch-size trade-off를 계산한다.

### 1.4 계획된 schedule과 실제 실행한 schedule

보고서의 schedule 행은 중간에 잘린 계획을 설명할 수 있다. OLMo 2 7B는 5T token 동안 3e-4의 10%까지 cosine하는 계획과 8,388,608,000 token warmup(1024 × 4096의 2,000 step)을 제시했지만, cosine을 “after 4T”에 멈췄다([[olmo-2]], 표 3). Stage-2 config는 checkpoint `step928646`을 load하고 `learning_rate: 0.000061499`에서 시작한 뒤 50B token에 걸쳐 0까지 선형 decay한다.

**계산 예시(도출됨).** Step 928,646 × 4,194,304 token = 3.895T token이다. Cosine progress는 (3.895e12 − 8.39e9) / (5e12 − 8.39e9) = 0.7786이므로 η = 3e-5 + ½ · 2.7e-4 · (1 + cos(0.7786π)) = 6.14e-5다. 이는 config 값의 0.3% 이내다. Linear decay는 50e9 / 4,194,304 = 11,921 step이고 config의 `stop_at: 11931`은 10을 더한다. 따라서 공개된 7B는 표 3의 5T cosine이 아니라, 중단된 cosine 뒤 서로 다른 데이터에서 별도 linear decay로 학습했다. OLMo 2는 cosine을 자르고 tail을 linear decay로 대체하는 것은 수치 없이 OLMo-0424 경험상 “little loss of performance”라고 말한다(§4.1).

같은 검사는 OLMo 2 13B의 충돌을 드러낸다. 표 3은 peak 9.0e-4를 인쇄하지만 공개 stage-1 YAML은 `learning_rate: 3.0e-4`다. 13B stage-2 config는 `step596057`을 load한다(596,057 × 2048 × 4096 = 5.0T token, 도출됨). 시작값 `learning_rate: 9e-5`는 5T horizon에서 9e-4 cosine의 10% floor다. 따라서 stage-2 값은 stage-1 YAML이 아니라 표 3과 일치하므로, checkpoint를 만든 값을 추정할 때 표 값이 더 낫다(도출됨).

Llama 3 405B는 η를 0까지 선형으로 하여 “on the final 40M tokens”에 anneal한다([[llama-3-recipe]]). §3.4.1의 16M-token batch에서는 optimizer step 2.5개다(도출됨). 보고서는 annealing 중 batch size를 쓰지 않으므로, 이는 annealing의 증거가 아니라 인쇄된 수치에 대한 **미해결 질문**이다.

## §2 데이터 품질이 작용하는 장소로서의 decay phase

**정의.** Annealing(cooldown 또는 데이터가 바뀔 때 mid-training이라고도 함)은 고품질 또는 target 데이터를 upsample한 mixture에서 실행하는 decay phase다. ch-32는 mid-training 데이터를 다루고, 이 절은 schedule이 기여하는 것과 결과 측정법을 다룬다.

**문제.** 데이터가 바뀌지 않아도 decay phase는 loss를 낮춘다. 따라서 decay 중 데이터 변경 뒤의 gain은 learning-rate decay와 데이터라는 두 부분을 가진다. 두 번째 문제는 추가 데이터가 이를 판정하는 benchmark와 자주 겹친다는 것이다.

**측정 메커니즘.**
1. Stable-phase checkpoint에서 시작한다.
2. 변경하지 않은 pre-training mixture에서 decay를 실행한다(control).
3. Candidate mixture에서 같은 decay를 실행한다.
4. 둘 다를 checkpoint와 비교한다. Development decision에 한 번도 쓰지 않은 held-out task, 그리고 데이터가 target한 capability에서 비교한다.

**증거: decay와 데이터를 분리(OLMo 2, [[olmo-2]] 표 11).** 4T token의 7B checkpoint에서 50B-token run은 다음을 보였다(OLMES / OLMES-Gen / MMLU / GSM*. OLMES는 OLMo 2 evaluation suite의 multiple-choice evaluation average, OLMES-Gen은 generative-task average, GSM*는 development에 쓴 GSM8K 문제 200개다).

| Run | OLMES | OLMES-Gen | MMLU | GSM* |
|---|---|---|---|---|
| Checkpoint, anneal 없음 | 69.6 | 63.2 | 59.8 | 28.5 |
| Pre-training mix에서 anneal(learning-rate decay만) | 74.0 | 64.5 | 61.8 | 27.0 |
| Filtered web + reference data에서 anneal | 75.2 | 63.8 | 63.1 | 28.5 |
| + math data | 75.7 | 69.7 | 62.3 | 52.0 |
| + math + instruction data | 75.7 | 70.2 | 63.1 | 46.5 |

Learning-rate decay만으로 최종 mix의 OLMES 6.1점 gain 중 4.4점을 설명하며 GSM* gain은 전혀 설명하지 않는다. Math data가 GSM* gain을 설명한다. 결과(단일 연구, 행당 run 하나).

**증거: held-out 대 development task(OLMo 2, 표 9).** 최종 mid-training 뒤(anneal 3회의 평균), 7B development score는 MMLU 59.8 → 63.7, GSM8K 24.1 → 67.5로 올랐다. Held-out score는 MMLU-Pro 27.4 → 31.0, TriviaQA 74.6 → 78.0으로 올랐다. Held-out suite는 “not used for model development decisions”였지만, GSM8K는 부분적으로만 held out이었다(1,319개 중 200개 예제를 development에 사용). Held-out gain은 target GSM8K gain보다 작지만 0은 아니다. 결과(단일 연구).

**증거: SFT 스타일 데이터로 annealing(MiniCPM, [[minicpm]] 표 1).** MiniCPM-2.4B는 고품질 및 SFT 데이터를 pre-training data에 섞어 decay한 뒤 SFT를 4B token 수행했다. C-Eval 52.6, MMLU 50.9, GSM8K 42.3, HumanEval 30.4이었고, decay에 pre-training data만 쓴 경우는 40.0, 44.6, 27.7, 27.7이었다. 1.2B 모델에서 annealing data 없이 SFT token을 두 배(6B → 12B)로 늘리면 MMLU는 47.9에 머물고 GSM8K는 34.2 → 34.4였다. Annealing data와 SFT 6B token을 쓰면 MMLU 49.6, MATH 10.5(7.9에서)를 기록했지만 GSM8K는 31.8(34.2에서)로 떨어졌다. 추가 데이터가 benchmark와 같은 도메인(지식, 수학, 코드)을 target하고 seed가 보고되지 않았으며 benchmark 하나가 하락했다. 따라서 이는 폭을 보인 증명이 아니라 target gain에 대한 결과(단일 연구)다.

**증거: benchmark training set에서 annealing(Llama 3, [[llama-3]] §3.1.3).** GSM8K와 MATH training set으로 annealing하면 사전학습 Llama 3 8B validation score가 24.0%, 6.4% 올랐지만, “the improvements on the 405B model are negligible”였다. Llama 3는 이후 “true few-shot learning capabilities and out-of-domain generalization”을 평가하려고 annealing data에서 “any training sets from commonly used benchmarks”를 제외했다. 보고서는 작은 dataset의 가치를 판단하는 데도 annealing을 쓴다. 50% 학습한 8B를 새 데이터에 30% weight로 하여 40B token 동안 선형으로 0까지 anneal한다. 결과(단일 연구, seed 없음).

**증거: peak learning rate와 anneal(OLMo 2 §4.1, 표 8).** 7B에서 peak LR 3, 6, 9, 12 ·10^-4 run을 0까지 decay한 뒤 비교했다. 300B + 50B token 뒤 OLMES는 62.5, 63.9, 64.1, 63.6이었다. 2T + 100B high-quality token 뒤 3e-4와 6e-4는 73.8, 73.9였다. 저자들은 높은 learning rate가 “does make mid-training more effective, but it does so by exactly the amount that the pretraining is worse”라고 결론 낸다. 높은 learning rate에서 GSM8K는 2.8점 높았으며, 저자들은 더 연구해야 한다고 말한다.

**조건과 한계.** OLMo 2와 MiniCPM ablation은 single run이다. 이 연구들 중 어느 것도 annealing data가 데이터와 evaluation suite 양쪽에 없는 capability의 성능을 바꾸는지 측정하지 않는다.

**범용 모델에 대한 함의.** Decay phase는 측정 기회다. Control anneal은 schedule을 분리하고 held-out suite는 target gain과 넓은 gain을 분리한다. Llama 3의 8B 대 405B 결과는 같은 annealing data가 작은 모델의 benchmark score를 바꾸면서 큰 모델은 바꾸지 않을 수 있음을 보인다. 따라서 한 size에서의 gain은 다른 size의 증거가 아니다.

## §3 Batch size와 learning-rate scaling

**정의.** Batch size B는 gradient step 하나에서 평균하는 example(또는 token) 수다. Critical batch size는 B를 키워도 필요한 step 수가 비례해서 더 줄지 않는 batch size다.

**문제.** 큰 batch는 더 많은 data parallelism과 더 적은 step을 허용하지만, 어떤 크기를 넘으면 같은 loss에 도달하려고 더 많은 총 token이 필요하다. 최적 learning rate도 B에 따라 바뀐다.

### 3.1 Gradient noise scale과 steps–tokens trade-off

**메커니즘**([[large-batch-training-noise-scale]], McCandlish et al. 2018).
1. Batch gradient는 example별 gradient B개를 평균한다. Covariance는 Σ/B이며, Σ는 example별 gradient covariance다.
2. True gradient G와 Hessian H를 갖는 loss의 quadratic model에서 최적 step size와 step당 loss 감소는 모두 1/(1 + B_noise/B)에 비례한다.
3. Target loss에 필요한 step S와 example E는 hyperbolic trade-off를 만족한다.

**수식.**

```
B_simple = tr(Σ) / |G|²
ε_opt(B) = ε_max / (1 + B_noise / B)
S / S_min − 1 = (E / E_min − 1)^(−1),     B_crit = E_min / S_min
```

B_simple은 simple noise scale(H가 identity의 상수배라고 가정), B_noise = tr(HΣ)/(GᵀHG), ε_max는 full-batch gradient에서 최적 step size, S_min과 E_min은 최소 step과 example, B_crit은 critical batch size다(식 2.6–2.12).

**계산 예시(card에서 도출, B_noise = 1,000).** Minimum 대비 step은 1 + B_noise/B, example은 1 + B/B_noise다.

| B | Steps / S_min | Examples / E_min | ε_opt / ε_max |
|---|---|---|---|
| 250 | 5.0 | 1.25 | 0.20 |
| 500 | 3.0 | 1.5 | 0.33 |
| 1,000 | 2.0 | 2.0 | 0.50 |
| 2,000 | 1.5 | 3.0 | 0.67 |
| 4,000 | 1.25 | 5.0 | 0.80 |

B를 250에서 500으로 두 배로 하면 최적 step size는 1.67배여서 linear scaling(×2)에 가깝다. 2,000에서 4,000으로 두 배로 하면 1.2배다. 이 quadratic SGD model에서 ε_opt/ε_max = B/(B + B_noise)이므로, linear-scaling prediction B/B_noise가 10% 이내인 것은 B ≤ B_noise/9일 때뿐이다(도출됨). Adam의 측정 exponent는 0.5~1.0이다(아래).

**증거.** B_simple은 B_crit가 20~1천만 초과인 8개 task에서 B_crit을 한 order of magnitude 이내로 예측했고, 둘 다 학습 중 적어도 한 order of magnitude 상승했다([[large-batch-training-noise-scale]], §3, 그림 4). Adam 또는 RMSProp에서 최적 learning rate는 ε ∝ B^α를 따랐고 α는 0.5(square-root scaling)~1.0(linear scaling) 사이였으며, 이후 평탄해졌다(§3.1, 부록 E.2). 결과(8개 task에 대한 단일 연구).

### 3.2 Language-model pre-training의 critical batch size

Zhang et al.은 critical batch size B*를, 최적 batch에서 linear-scaling prediction보다 step count가 최대 20% 높은 가장 큰 batch로 정의한다([[critical-batch-size-pretraining]], §3.1). Context length 512에서 C4로 학습한 85M–1.2B 모델(batch은 sequence 단위)에서:

- Chinchilla 설정(model size에 따라 token scaling): B* = 93.20 · N^0.47, N은 million 단위(§3.2). 도출하면 151M은 약 985 sequence(약 0.5M token), 1.2B는 약 2,610 sequence(약 1.3M token)다.
- 고정 3.072B token, model size 변화: B* = 621.341 · N^0.087(§3.3). 도출하면 151M은 약 961, 1.2B는 약 1,151이다.
- 고정 302M model, Chinchilla count의 0.28×~4×로 token 변화: B*는 token과 함께 상승한다(§3.3, 그림 1).

저자들은 compute-optimal training에서 B* 증가가 “is more strongly attributed to extended data size or training durations rather than the increase of model size”라고 결론 낸다. 결과(단일 연구; C4만; 1.2B까지). 더 큰 LSTM이 낮은 loss를 통해서만 더 큰 noise scale에 도달한다고 발견한 McCandlish et al.과 일치한다(§3.2, 그림 8).

**함의.** 고정된 모델을 더 많은 token에 학습하는 run은 학습 후반에 batch size를 올릴 수 있다. Llama 3 405B는 4M token에서 시작해 252M token 뒤 8M, 2.87T token 뒤 16M으로 올렸으며, 작은 초기 batch는 “to improve training stability”를 위한 것이었다([[llama-3-recipe]]). MiniCPM-1.2B는 batch를 2M에서 4M으로 올리고, learning rate를 낮추는 것과 “might have a similar effect”인 loss drop을 관측했다([[minicpm]], §6.4).

### 3.3 적합한 learning-rate 및 batch-size 규칙

- DeepSeek LLM은 1e17~2e19 FLOPs budget에 맞춰 η_opt = 0.3118 · C^−0.1250, B_opt = 0.2920 · C^0.3271을 적합했고, 최저 loss의 0.25% 이내 설정을 near-optimal로 처리했다([[deepseek-llm]], §3.1, 식 1). 계산 예시(도출됨): 7B 모델에서 C = M · D = 4.23e10 FLOPs/token · 2.0e12 token = 8.46e22이므로 η = 4.25e-4, B = 9.2M token이다. 표 2는 4.2e-4, 2304 × 4096 = 9.4M token을 쓴다.
- Porian et al.은 RefinedWeb에서 parameter당 20 token으로 각각 학습한 최대 221M-parameter 모델 sweep에 BS = 0.00037 · N^0.703, LR = 3.7 · N^−0.36(batch은 2048-token sequence 단위)을 적합했다. 이를 최대 901M 모델에 적용했고, batch size 128 이하에서 AdamW β₂ = 0.95가 suboptimal임을 발견했다([[resolving-scaling-discrepancies]], §3.5, 그림 3). 이들의 901M 처방(LR 0.0024, batch 640)은 13개 configuration 중 최저 loss를 냈다([[resolving-scaling-discrepancies-recipe]]).
- DeepSeek과 Porian et al. 모두 그보다 작은 batch에서 성능이 저하되는 optimal batch size를 발견했다. Porian et al.은 이것이 critical size 아래의 어떤 batch도 동등하게 좋다는 견해와 “appears to contradict”한다고 말한다([[resolving-scaling-discrepancies]], §3.5). 두 learning-rate exponent는 0.11 차이고 prediction은 2–3배 차이다. **미해결 질문:** 어느 규칙이 새 data mixture로 전이되는가.

## §4 Initialization과 hyperparameter transfer

### 4.1 Variance-preserving initialization

**정의.** Initialization은 학습 전 weight의 distribution을 정한다. Variance-preserving scheme은 activation과 gradient variance가 layer를 지나도 대략 일정하도록 weight variance를 고른다.

**수식**([[weight-init]] 발췌).

```
Glorot (Xavier):  Var(W) = 2 / (n_in + n_out)
He (Kaiming):     Var(W) = 2 / n_in               (ReLU)
LeCun / fan-in:   Var(W) = 1 / n_in
```

n_in과 n_out은 layer의 fan-in과 fan-out이다.

**계산 예시(도출됨).** 4096 × 4096 matrix에서 Glorot과 fan-in은 std √(1/4096) = 0.0156, He는 √(2/4096) = 0.0221을 준다. OLMo 2 7B(d_model 4096)와 13B(5120)는 모두 모든 parameter에 std 0.02를 쓴다. 이는 4096에서는 fan-in과 He 값 사이이고, 5120에서는 fan-in(0.0140)보다 높다.

### 4.2 Residual scaling: GPT-2, OLMo-0424, OLMo 2

- GPT-2는 initialization에서 “the weights of residual layers”를 “a factor of 1/√N where N is the number of residual layers”로 scale했고(§2.3), 각 sub-block 입력으로 layer normalization을 옮기고 마지막 layer norm을 뒀다([[weight-init]] 발췌). 논문은 N이 block 수인지 sub-layer 수인지를 말하지 않는다. 48-layer GPT-2 XL에서 N이 block이면 1/√48 = 0.144, block당 sub-layer 두 개면 1/√96 = 0.102다(도출됨).
- OLMo-0424는 input projection을 1/√d_model, output projection을 1/√(2 · d_model · layer_idx)로 scale했다. OLMo 2는 이를 모든 parameter에 3 standard deviation에서 truncate한 N(0, 0.02)로 교체했다. Spike를 재현하는 reduced-warmup test에서 gradient spike score(1,000-step rolling mean에서 7 standard deviation 이상인 값의 비율)는 0.40에서 0.03으로 내려갔고, “the new initialization converges slightly slower”였다([[olmo-2]], §3.2, 그림 4). 결과(단일 연구, 각각 run 하나).

**조건과 한계.** OLMo 2 결과는 depth-scaled init가 일반적으로 해롭다는 뜻이 아니다. Normalization도 바꾼 한 architecture에서 두 specific scheme을 비교한다. Depth scaling은 보편적 관행이 아니다. OLMo 2는 이를 버렸고 DeepSeek LLM은 단일 init std 0.006을 보고한다([[deepseek-llm]], §2.3).

### 4.3 μP: 작은 proxy가 큰 모델의 learning rate를 고를 수 있는 이유

**정의.** μP(maximal update parametrization)는 width가 커져도 최적 hyperparameter가 대략 일정하도록 layer type별 initialization variance, learning rate, output multiplier를 width의 함수로 정한다. μTransfer는 작은 proxy를 tuning하고 hyperparameter를 큰 model에 복사한다([[weight-init]] 발췌; Yang et al. 2022).

**수식**(Tensor Programs V, 표 3. 다른 standard parametrization은 괄호 안).

|  | Input weights and biases | Output weights | Hidden weights |
|---|---|---|---|
| Init. variance | 1/fan_in | 1/fan_in² (1/fan_in) | 1/fan_in |
| SGD LR | fan_out (1) | 1/fan_in (1) | 1 |
| Adam LR | 1 | 1/fan_in (1) | 1/fan_in (1) |

Transformer μP는 attention logit도 1/√d 대신 1/d로 scale한다. Adam에서 hidden-weight learning rate는 1/fan_in으로 scale하고, SGD에서는 width-independent다.

**계산 예시(도출됨, 가상 width).** MiniCPM operation(2-D tensor LR = base LR / (d_m / d_base), 2-D init std = init_std / √(d_m / d_base))으로, proxy에서 찾은 base LR 0.01, init_std 0.1, target d_m / d_base = 4096 / 256 = 16이면 hidden matrix는 LR 0.01 / 16 = 6.25e-4, init std 0.1 / 4 = 0.025를 받는다([[minicpm]], 부록 A.1, 표 7).

**증거.**
- GPT-3 6.7B는 40M-parameter width-256 proxy에서 tuning했고, 비용은 “only 7% of total pretraining cost”였다. Validation loss는 원래 hyperparameter로 재실행한 2.03 대비 1.98이다. 저자들은 두 confound를 쓴다. 재실행은 실수로 absolute attention을 썼고, μP model은 divergence를 피하려 FP32로, baseline은 FP16으로 학습했다(Tensor Programs V §7.4, 표 7). 결과(명시된 confound가 있는 단일 연구).
- MiniCPM: μP에서 best base LR은 0.04B~0.5B에서 “remains around 0.01”이고 2.1B에서 확인했다([[minicpm]], §3.3, 그림 3). Learning-rate stability 주장을 재현한다.
- Wortsman et al.: 간단한 μP variant는 width 간 optimal learning rate를 안정화했지만 loss 또는 learning-rate sensitivity를 낮추지 못했고, 높은 learning rate에서 qk-layernorm 필요성을 없애지도 못했다([[small-scale-proxies-instabilities]], §3.2.4).

**조건과 한계.** Tensor Programs V 표 1은 regularization(dropout, weight decay)을 transferable하지 않다고 열거하고, depth·batch size·training time·sequence length 간 transfer를 “empirically validated only on Transformers”라고 한다. μP는 batch size나 token budget을 고르지 않는다. 이 장 §3이 이를 다룬다.

### 4.4 작은 실험이 큰 실험을 예측할 때와 아닐 때

- DataDecide는 25개 pre-training corpus를 14개 크기(4M–1B)에서 seed 3개로 학습했다. 150M model로 corpus의 순위를 매기면 pairwise comparison의 약 80%에서 1B에서 이기는 corpus를 골랐다([[datadecide]], §3.2, 그림 3). Continuous likelihood metric은 MMLU, ARC, code task의 small-scale decision accuracy를 높였지만 math task는 trivial 근처에 머물렀다(§3.3–3.4).
- Wortsman et al.은 small model에서 maximum attention logit을 model size에 대해 fit하고 qk-layernorm 없이 LR 1e-2에서는 4.8B model이 diverge할 것으로 예측했다. 4.8B run은 diverge했다([[small-scale-proxies-instabilities]], §3.3, 그림 9).
- 7B 규모 OLMo 2에서는 높은 learning rate가 초기에 더 낮은 loss를 냈지만 3e-4가 6e-4를 “well past 200B tokens”에서 앞질렀다. “A shorter hyperparameter experiment might come to the wrong conclusion”이다([[olmo-2]], §4.1).

함의: Proxy는 parametrization이 전이되고(§4.3), crossover를 지나도록 충분히 길게 학습하며, metric이 small scale에서 후보를 분리할 때에만 large run을 예측한다.

## §5 Normalization

### 5.1 LayerNorm과 RMSNorm

**수식**([[batch-vs-layer-norm]] 발췌).

```
LayerNorm: y = γ ⊙ (x − μ) / √(σ² + ε) + β
RMSNorm:   y = γ ⊙ x / √((1/d) Σ_i x_i² + ε)
```

x는 token 하나의 d-dimensional activation, μ와 σ²는 d feature에 대한 mean과 variance, γ와 β는 학습하는 scale과 shift, ε은 작은 상수다(OLMo 2 7B config: 1e-6).

**증거.** RMSNorm은 “achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models”라고 보고한다(Zhang & Sennrich 2019, 초록). 이 범위는 normalization op의 speedup이 아니라 RNN 및 Transformer 실험의 전체 model running time이다. OLMo 2는 ablation에서 non-parametric LayerNorm과 RMSNorm 사이에 “no difference”를 발견했다([[olmo-2]], §3.3.1). Qwen2.5는 pre-normalization과 RMSNorm을 사용한다([[qwen-2.5]], §2).

### 5.2 Post-norm 대 pre-norm

```
Post-LN: x_{l+1} = LN(x_l + F(x_l))
Pre-LN:  x_{l+1} = x_l + F(LN(x_l)),   with a final LN before the output layer
```

F는 attention 또는 MLP sub-layer다.

**메커니즘**(Xiong et al. 2020, [[batch-vs-layer-norm]] 발췌).
1. Initialization에서 Post-LN은 모든 layer의 expected squared hidden-state norm을 (3/2)d로 유지한다. 반면 Pre-LN hidden state는 (1 + l/2)d ≤ E‖x_l‖² ≤ (1 + 3l/2)d를 만족하며 depth l에 따라 선형으로 증가한다(Lemma 2).
2. Layer normalization의 Jacobian은 √d / ‖x‖로 scale한다(Lemma 3). 따라서 input norm이 클수록 더 작은 gradient가 통과한다.
3. 마지막 FFN layer gradient는 Post-LN에서 depth L과 무관한 O(d√(ln d))이고 Pre-LN에서는 O(d√(ln d / L))다(Theorem 1). Initialization에서 측정한 Post-LN 값은 “around 1.6”에 머물렀고 Pre-LN 값은 depth가 6에서 14 layer로 늘며 줄었다(§3.4, 그림 3).

**계산 예시(도출됨).** l = 24 layer에서 Pre-LN E‖x‖²는 13d~37d이므로 ‖x‖는 √d의 3.6~6.1배다. Post-LN의 √1.5 = 1.22배와 비교된다. 그 depth의 layer norm은 Post-LN layer norm보다 약 3~5배 작은 gradient를 통과시킨다.

**함의.** Near-output Post-LN의 큰 gradient가 warmup이 필요한 이유라는 것이 저자 설명이다(§1.1). Pre-LN model은 output layer 전 final layer norm을 더한다(Xiong et al. 표 1; GPT-2 §2.3). Xiong et al.은 이 final layer norm의 input이 L에 따라 선형으로 scale하므로 모든 parameter gradient가 √L로 normalize된다고 말한다(§3.3). Depth에 따른 residual-stream norm 증가는 pre-norm 특성이다. Post-LN은 add마다 다시 normalize한다.

### 5.3 OLMo 2의 reordered norm과 QK-norm

**정의.** Reordered norm은 각 sub-layer 출력을 residual stream에 더하기 전에 normalize한다.

```
OLMo-0424 (pre-norm):  h = x + Attention(LN(x));        h_out = h + MLP(LN(h))
OLMo 2 (reordered):    h = x + RMSNorm(Attention(x));   h_out = h + RMSNorm(MLP(h))
```

Residual stream x, h는 add 뒤에 정규화되지 않으므로 post-norm이 아니다([[olmo-2]], §2.1, §3.3.2; Liu et al. 2021이 제안). QK-norm은 attention dot product 전에 query와 key projection에 RMSNorm을 적용한다.

**증거.** “In isolation, neither of these changes yield good results, but together they improve both the growth and the spikiness of the L2 norm of the gradient.” 둘을 함께 적용하면 gradient spike score가 0.108에서 0.069로 떨어졌다([[olmo-2]], §3.3.2, 그림 7). 결과(단일 연구, 각각 run 하나; downstream comparison은 인쇄하지 않음).

### 5.4 Logit growth를 제어하는 QK-norm, z-loss, ε

**수식**([[small-scale-proxies-instabilities]]).

```
attention logit:  z_ij = ⟨q_i, k_j⟩ / √d_h
z-loss:           L_total = L_CE + α · log² Z,    Z = Σ_j exp(y_j)
AdamW update:     Δ = m̂ / (√v̂ + ε)
```

q_i와 k_j는 query와 key vector, d_h는 head dimension, y는 output logit, Z는 softmax normalizer, α는 z-loss coefficient, ε은 AdamW 상수다.

**증거.**
- Attention-logit growth는 LR 0.1의 9.4M model에서 나타났다. QK-layernorm이 없을 때 diverge하는 learning rate는 model이 커질수록 낮아졌다. 10M model에서 maximum attention logit을 약 1e4로 강제하면 loss는 zero-layer bigram baseline보다 나빴다(§3.1.1, §3.3, 그림 10). Growth는 cosine similarity 증가가 아니라 더 큰 query와 key norm에서 왔다(그림 E.1).
- Weight decay 없는 model에서 training 후반 매우 음의 logit인 output-logit divergence가 일어났다. α = 1e-4의 z-loss가 이를 해결했다(§3.1.2, 그림 3–4).
- QK-layernorm과 z-loss가 있으면 최대 1.2B model이 LR 3e-4~3e-1 전반에서 낮은 loss까지 학습했다(그림 1). 결과(단일 연구, 2.4M–4.8B, C4).
- Qwen3는 QKV-bias를 제거하고 안정 학습을 위해 “introduce[d] QK-Norm”했다(arXiv:2505.09388 §2, [[batch-vs-layer-norm]] 발췌). Qwen2.5 architecture 설명은 QKV bias를 열거하고 QK-norm은 열거하지 않는다([[qwen-2.5]], §2). MiniCPM은 QK-norm에서 더 낮은 learning-rate sensitivity를 측정했지만 채택하지 않았다([[minicpm]], 부록 A.1).
- ε: Scale과 learning rate가 커지며 gradient RMS는 ε = 1e-8 쪽으로 떨어졌다. LR 0.3의 4.8B model에서 ε = 1e-15는 loss를 개선했고 ε = 1e-6은 diverge했다([[small-scale-proxies-instabilities]], §3.4). OLMo 2는 ε를 1e-5에서 1e-8로 낮췄으며, 이후 “the gradient norm settles much more quickly and remains permanently lower”였다([[olmo-2]], §3.4.1).
- OLMo 2의 z-loss coefficient는 §3.3.3에 1e-4, 표 1과 공개 7B config에는 1e-5로 인쇄됐다. Fused(Flash Attention) z-loss와 PyTorch z-loss implementation은 forward pass에서는 맞았지만 backward pass에서는 맞지 않았다(§3.3.3, 그림 8).

## §6 Long-context continued pre-training: optimization과 normalization 선택

Context extension은 ch-32b에서 다룬다. 이 절은 extension stage가 재사용하거나 변경하는 schedule 및 normalization 설정을 다룬다.

- **Learning rate.** ProLong은 Llama-3-8B-Instruct를 64K에서 peak LR 1e-5, warmup 10%, cosine to 1e-6, 4M-token batch로 20B token 계속 학습한다([[prolong-recipe]]). 이는 Llama 3 보고서가 8B pre-training에 인쇄한 peak 3e-4의 1/30이다(표 3은 Llama 3.1 model을 설명하며, 비율은 도출됨). 512K stage에서 논문은 1e-5를 인쇄하지만 공개 script default는 5e-6이다(충돌, [[prolong-recipe]]). YaRN은 LR 2e-5, warmup 20 step, 400 step, batch 64로 Llama 2를 extension한다([[yarn]], §4.1). Kimi k1.5의 128K에서 1e-5 재-warm은 SFT stage에 속한다(32K stage 2e-5 → 2e-6, 이후 128K stage 1e-5 → 1e-6). Long-context activation pre-training learning rate는 보고하지 않는다([[kimi-k1-5-recipe]]). Llama 3 405B는 약 800B token에 걸쳐 6 stage로 8K에서 128K로 extension하고, short-context evaluation이 “recovered completely”하며 needle-in-a-haystack retrieval(긴 문서에 삽입한 사실 하나 찾기)을 풀면 전진한다. 이 stage의 learning rate는 보고하지 않는다([[llama-3]], §3.4.2).
- **Re-warming.** Decay한 checkpoint에서 새 data로 재개할 때 learning rate를 re-warm하고 re-decay하면 adaptation이 늘었고, 더 높은 re-warm peak는 “more forgetting and more adaptation”을 낳았다(405M, Pile → SlimPajama 및 Pile → German; [[continual-pretraining-rewarm-replay]], §6.1.2). 이전 data를 5% 또는 25% replay하면 re-warmed run은 union에서의 re-training과 맞먹었다(§6.3). Hägele et al.은 pre-cooldown checkpoint에서 높은 learning rate로 계속하면 re-warming을 피한다고 주장하지만 새-distribution continuation은 시험하지 않았다([[cooldown-scaling-beyond-fixed-durations]], §2).
- **Attention temperature.** YaRN은 extension factor s에 대해 √(1/t) = 0.1 ln(s) + 1을 query와 key 모두에 적용해 attention logit을 scale한다([[yarn]], §3.3, 식 15). 계산 예시(도출됨): s = 8이면 √(1/t) = 1.208이므로 logit은 1/t = 1.459배이고, s = 32이면 1.347, 1.813이다. Llama 4 Scout는 발표된 수식 없이 inference-time attention temperature scaling을 쓴다([[llama-4]]).
- **QK-norm과 retrieval.** 750B token으로 학습하고 SFT 뒤 평가한 8B variant 세 개에서 QK-Norm variant는 MMLU에서 RoPE 48.55와 0.34점 이내인 48.21이었지만, 65K needle에서 최저 7.93이었다(RoPE 9.82, NoPE 9.03). 128K aggregated attention entropy는 RoPE 7.62와 비교해 14.14였다([[rope-nope-hybrid-attention]], 표 2, 표 9). 저자들은 normalization이 query–key dot product의 magnitude information을 제거하기 때문이라고 귀속한다(**해석**). 결과(단일 연구, seed 없음). 이것이 QK-norm이 다른 architecture에서 long-context ability를 막는다는 뜻은 아니지만, 안정성을 위해 QK-norm을 채택할 때 long-context retrieval은 평가 요건이 된다.

## §7 Over-training과 fine-tuning plasticity

**정의.** 여기서 plasticity는 사전학습 checkpoint가 이미 가진 capability를 잃지 않으면서 fine-tuning으로 얼마나 바뀔 수 있는지를 뜻한다. Over-training은 모델 크기의 compute-optimal token count보다 더 많은 token으로 학습하는 것이며, Hoffmann et al.은 이를 parameter당 약 20 token으로 둔다([[minicpm]] §4.5에서 인용).

**증거**([[overtrained-lms-harder-to-finetune]]).
- Anthropic-HH에서 checkpoint별로 tuning한 learning rate를 써 instruction-tune한 OLMo-1B intermediate checkpoint: 3T-token base는 2.3T-token base보다 AlpacaEval response rate가 최대 3% 낮고 ARC는 약 2% 낮았으며 1.5T base 수준으로 떨어졌다(§2.2, 그림 2). Base model은 token에 따라 단조로 개선됐다. Onset은 2.5T token 너머였다(§3.1). OLMo-7B는 3T까지 이런 저하가 없었다(§3.1). Nominal size로 계산하면 3T token은 1B에서 parameter당 약 3,000 token, 7B에서 약 430 token이다(도출됨). 논문은 threshold ratio를 확립하지 않는다.
- 저자들은 confound를 지적한다. 한 run의 public checkpoint는 annealing schedule 때문에 서로 다른 final learning rate를 가진다. 이 confound는 각 0까지 anneal한 controlled run(15M–90M, C4 4B–128B token)에서 제거했다(§3.2).
- 메커니즘(“progressive sensitivity”): 고정된 Gaussian perturbation size 또는 고정 fine-tuning learning rate에서 modification이 일으키는 perplexity 증가는 pre-training token에 따라 단조 증가하며, 더 큰 fine-tuning learning rate는 더 적은 token에서 degradation point에 도달한다(§3.3–3.4, 그림 3–5). Optimal보다 작은 fine-tuning learning rate는 degradation을 늦추지만 in-distribution performance도 낮춘다(§3.4.2).

**조건과 한계.** 결과(단일 연구). Public checkpoint scale은 1B–7B, controlled run은 15M–90M이며 RL과 pruning은 시험하지 않는다.

**범용 모델에 대한 함의.** Pre-training token budget과 final learning rate는 prior ability를 잃기 전에 이후 단계가 모델을 얼마나 움직일 수 있는지 바꾼다. SFT와 RL이 뒤따를 pre-training plan은 candidate checkpoint의 base checkpoint만 보지 말고 held-out task에서 post-trained version도 평가해야 한다(ch-01은 update size와 retention, ch-32a는 replay를 다룬다).

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 증거 |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-stable | peak LR; warmup; decay | 8 × 10^-5; 선형 8,000 step; 1,200,000 step에 걸쳐 8 × 10^-7까지 cosine | arXiv:2407.21783v3 §3.4.1([[llama-3-recipe]]) | 검증됨 2026-09-14 | ablation 보고 없음 |
| Llama 3.1 405B | 405B | pretrain-stable | batch schedule | sequence 4,096에서 4M token → 252M token 뒤 “8M sequences of 8,192 tokens” → 2.87T token 뒤 16M | §3.4.1 | 검증됨 2026-09-14 | 작은 초기 batch는 “to improve training stability”; 수치 없음 |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | anneal | 마지막 40M token, 128K context에서 LR 선형 0; 고품질 source upsample; Polyak averaging(checkpoint 평균) | §3.4.3 | 검증됨 2026-09-14 | ablation 보고 없음; anneal 중 batch 보고 안 함 |
| Llama 3.1 8B / 70B | 8B / 70B | pretrain-stable | peak LR; schedule | 3 × 10^-4 / 1.5 × 10^-4; warmup, decay, batch 인쇄 안 함 | 표 3; §3.4 | 검증됨 2026-09-14(schedule 보고 안 함) | ablation 보고 없음 |
| Llama 3 scaling-law 모델 | 40M–16B | pretrain-stable | schedule | cosine, warmup 2,000 step, size별 peak 2 × 10^-4~4 × 10^-4, peak의 0.1까지 decay; batch 250K–4M(단위 인쇄 안 함) | §3.2.1 | 검증됨 2026-09-14 | n/a |
| Llama 3 8B(실험) | 8B | eval-gate | anneal-as-data-evaluation | 50%-학습 model, 40B token 동안 LR 선형 0, 새 data 30% / 기본 mix 70% | §3.1.3 | 검증됨 2026-09-14 | 방법; 수치 인쇄 안 함 |
| OLMo 2 7B / 32B | 7B / 32B | pretrain-stable | peak LR | 3.0 × 10^-4 / 6.0 × 10^-4 | arXiv:2501.00656 표 3; `OLMo2-7B-stage1.yaml` `learning_rate: 3.0e-4`([[olmo-2]] 발췌) | 검증됨 2026-09-15 | §4.1 표 8(7B): 2T + 100B anneal 뒤 3e-4 대 6e-4가 OLMES 0.1 이내 |
| OLMo 2 13B | 13B | pretrain-stable | peak LR | 9.0 × 10^-4 | 표 3; §4.1(“higher peak learning rate from the start”) | 충돌(stage-2 config는 5T horizon의 9e-4 cosine 10% floor인 9e-5에서 시작; 이 행의 도출된 지지) | ablation 보고 없음 |
| OLMo 2 13B | 13B | pretrain-stable | peak LR | 3.0 × 10^-4 | `OLMo2-13B-stage1.yaml` `learning_rate: 3.0e-4` | 충돌 | ablation 보고 없음 |
| OLMo 2 7B / 13B / 32B | 7B / 13B / 32B | pretrain-stable | warmup; decay; horizon; truncation | 2,000 step; peak의 10%까지 cosine; 5T / 5T / 6.5T token; 4T 뒤 / n/a / 6T 뒤 | 표 3, §2.3 | 검증됨 2026-09-15 | §4.1: truncation과 linear decay는 “with little loss of performance”(OLMo-0424 경험, 수치 없음) |
| OLMo 2 7B / 13B / 32B | 7B / 13B / 32B | pretrain-stable | global batch | 1024 / 2048 / 2048 sequence × 4096 token(4.19M / 8.39M / 8.39M token, 도출됨) | 표 3 | 검증됨 2026-09-15; token 도출됨 | ablation 보고 없음 |
| OLMo 2 7B | 7B | pretrain-stable | optimizer; clipping; init | AdamW β (0.9, 0.95), wd 0.1, embedding에 wd 없음, ε 1e-8; clip 1.0; 3σ에서 truncate한 N(0, 0.02) | `configs/official-1124/OLMo2-7B-stage1.yaml`; §3.2, §3.4 | 검증됨 2026-09-15 | 그림 4: spike score 0.40 → 0.03; 그림 9(ε); 그림 10: 0.16 대 0.092(embedding wd) |
| OLMo 2 7B | 7B | pretrain-stable | z-loss coefficient | 10^-4 | §3.3.3 | 충돌 | ablation 보고 없음 |
| OLMo 2 7B | 7B | pretrain-stable | z-loss coefficient | 10^-5 | 표 1; `OLMo2-7B-stage1.yaml` `auxiliary_loss_multiplier: 1e-5` | 충돌(공개 config가 checkpoint를 생성) | ablation 보고 없음 |
| OLMo 2 7B | 7B | mid-train | LR; token; merge | step 928,646에서 6.1499 × 10^-5로 시작, 50B token에 걸쳐 선형 0; data order 3개 평균 | `OLMo2-7B-stage2-seed42.yaml`; §2.3 | 검증됨 2026-09-15 | 표 11: LR decay만으로 OLMES 69.6 → 74.0; 표 9 |
| OLMo 2 13B, 32B | 13B, 32B | mid-train | token; merge | 100B run 3개 + 300B run 1개, 평균 | §2.3, 표 9 caption | 검증됨 2026-09-15 | 표 9: 13B 평균 58.9 → 68.3 |
| DeepSeek LLM 7B / 67B | 7B / 67B | pretrain-stable | peak LR; batch | 4.2 × 10^-4 / 3.2 × 10^-4; 2304 / 4608 sequence × 4096 | arXiv:2401.02954v1 표 2([[deepseek-llm]] 발췌) | 검증됨 2026-09-15 | 1e17–2e19 FLOPs의 식 1 fit; 도출 검사가 표 2와 일치 |
| DeepSeek LLM 7B, 67B | 7B, 67B | pretrain-stable | schedule; optimizer; init | warmup 2,000 step; token 80% 뒤 max의 31.6%, 90% 뒤 10%; AdamW (0.9, 0.95), wd 0.1, clip 1.0; init std 0.006 | §2.3 | 검증됨 2026-09-15 | 그림 1a(1.6B, 100B token): cosine과 “essentially consistent” |
| MiniCPM-2.4B | 2.4B | pretrain-stable | schedule; batch; 최대 LR | WSD; “batch size of 3.93 million”(단위 인쇄 안 함; 표 2: 4M); 0.01(μP base LR) | arXiv:2404.06395v3 §6.2, 표 2([[minicpm]] 발췌) | 검증됨 2026-09-15 | §3.3 그림 3: 0.04B~0.5B에서 optimal base LR 약 0.01, 2.1B 확인 |
| MiniCPM-2.4B | 2.4B | pretrain-decay/anneal | decay shape; 길이; 데이터 | exponential 0.5^((s−S)/T), T = 5000 step(20B token); pre-training data + 고품질 SFT data | §6.2 | 검증됨 2026-09-15 | 표 1: A-2 대 A-1(MMLU 50.9 대 44.6) |
| MiniCPM(μP search) | 0.009B proxy | pretrain-stable | μP 설정 | scale_depth 1.4, scale_emb 12, init_std 0.1, lr 0.01 | 부록 A.1 | 검증됨 2026-09-15 | Bayesian search, 그림 14 |
| Hägele et al. 1B(FineWeb, 100B token) | 1B | pretrain-decay/anneal | peak LR; warmup; cooldown | 8e-4; 2,000 step; (1-sqrt) 20% 또는 linear 20% | arXiv:2405.18392v3 표 3, 표 4([[cooldown-scaling-beyond-fixed-durations-recipe]]) | 검증됨 2026-09-14 | 표 4: cosine-to-10% 46.26 대비 46.23 / 46.20 |
| Porian et al. 901M | 901M | pretrain-stable | LR; batch(2048-token sequence); β₂ | 0.0024; 640; 0.95 | arXiv:2406.19146v4 표 4([[resolving-scaling-discrepancies-recipe]]) | 검증됨 2026-09-14 | 부록 G.5 표 7: 13 configuration 중 최저 loss 2.943 |
| Wortsman et al. default | 2.4M–1.2B | pretrain-stable | warmup; qk-layernorm; z-loss; ε | 1e5 step 중 5e3; on(head별); 1e-4; 1e-8 | arXiv:2309.14322v2 §2.1([[small-scale-proxies-instabilities]]) | 검증됨 2026-09-14 | 그림 5(warmup), 그림 1(qk-layernorm), 그림 3(z-loss) |
| ProLong-64k-Base | 8B | long-context | LR; warmup; decay; batch; RoPE base | 1e-5; 10%; 1e-6까지 cosine; 4M token; 8 × 10^6 | arXiv:2410.02660v4 표 9; train_64K.sh([[prolong-recipe]]) | 검증됨 2026-09-14 | RoPE base: 부록 B.1 표 18(54.6 대 48.7 대 29.1); LR: ablation 보고 없음 |
| ProLong-512k-Base | 8B | long-context | peak LR | 1e-5(“each stage”) | v4 표 9 | 충돌 | ablation 보고 없음 |
| ProLong-512k-Base | 8B | long-context | peak LR | 5e-6(script default) | train_512K.sh `lr=${LR:-5e-6}` | 충돌 | ablation 보고 없음 |
| Yarn-Llama-2 s = 16 | 7B, 13B | long-context | LR; warmup; step; batch | 2 × 10^-5; 선형 20 step; 400; 64 | arXiv:2309.00071v3 §4.1([[yarn]]) | 검증됨 2026-09-14 | 부록 B.2 표 6: YaRN 400 step ≈ PI 1000 step |
| Kimi k1.5 | 보고하지 않음 | SFT | 길이 stage별 LR | 32K: 2e-5 → 2e-6; 128K: 1e-5로 re-warm → 1e-6 | arXiv:2501.12599v4 §2.5.2([[kimi-k1-5-recipe]]) | 검증됨 2026-09-14 | ablation 보고 없음 |
| Kimi k1.5 base | 보고하지 않음 | long-context | LR | 보고하지 않음(§2.5.1, 부록 B 확인) | — | 보고하지 않음 | — |
| Continual pre-training, 405M | 405M | mid-train | re-warm peak; decay; warmup; replay | 3 × 10^-4(pre-training peak); 3 × 10^-5까지 cosine; iteration의 1%; 5%(Pile → SlimPajama), 25%(Pile → German) | arXiv:2403.08763v4 §6.1.2, §6.3([[continual-pretraining-rewarm-replay]] 발췌) | 검증됨 2026-09-15 | 그림 4: 높은 re-warm peak는 더 많은 adaptation과 forgetting |

**작은 범용 run의 시작점.** 약 100B web token에 학습하는 1B 근처 decoder-only model에 대해 검증된 행은 다음을 뒷받침한다. β (0.9, 0.95)의 AdamW, gradient clipping 1.0, 2,000-step linear warmup, 다른 size에서 복사하지 않고 fitted rule로 정한 peak learning rate(Hägele et al.은 1B/100B run에서 DeepSeek LLM law로 추정한 8e-4를 썼고, Porian et al.은 약 14B token의 901M에 batch 640 × 2048과 0.0024를 처방), 그리고 20% (1-sqrt) cooldown이 붙은 constant learning rate다. 이 cooldown은 1B setting에서 downstream cosine과 맞았다. 210M run에서 cooldown schedule의 최적 constant learning rate는 최적 cosine peak의 약 절반이었다(그림 3 caption). Wortsman et al.의 qk-layernorm과 z-loss 1e-4는 C4의 2.4M–1.2B에서 stable learning-rate range를 넓혔다. OLMo 2의 truncate한 N(0, 0.02) init, AdamW ε 1e-8, embedding에 weight decay 없음, QK-norm을 갖는 reordered norm은 4096-token sequence의 7B에서 gradient-spike 측정으로 선택됐다. 이 값 중 어느 것도 독자의 data에서 시험되지 않았고, 다른 출처 행을 함께 시험하지도 않았다.

## 일반화 관점

**(a) 폭을 늘리는 것.**
- Learning-rate decay를 완료하는 것: OLMo 2 7B의 변경 없는 mixture anneal은 OLMES 69.6 → 74.0, MMLU 59.8 → 61.8이었다([[olmo-2]], 표 11). 1B에서 cooldown과 cosine은 동일한 8-task 평균을 냈다([[cooldown-scaling-beyond-fixed-durations]], 표 4).
- Decay에서 넓은 고품질 데이터를 쓰고 held-out task로 평가하는 것: OLMo 2 7B held-out MMLU-Pro 27.4 → 31.0, TriviaQA 74.6 → 78.0([[olmo-2]], 표 9).
- 선택한 learning rate에서 큰 run을 가능하게 하는 설정: learning rate 3 order of magnitude 전반의 QK-norm과 z-loss([[small-scale-proxies-instabilities]]), 그리고 OLMo 2에서 gradient spike를 줄인 reordered norm, QK-norm, init 변경([[olmo-2]], §3).

**(b) 협소화 또는 forgetting을 일으키는 것.**
- Benchmark 인접 데이터에 annealing: Llama 3 8B는 GSM8K와 MATH training set anneal에서 GSM8K validation 24.0%를 얻었지만 405B의 gain은 negligible했다([[llama-3]], §3.1.3). MiniCPM-1.2B annealing data는 같은 SFT token에서 MATH와 MMLU를 올리지만 GSM8K는 34.2에서 31.8로 낮췄다([[minicpm]], 표 1).
- Fine-tuning이 뒤따르는 extended pre-training: OLMo-1B는 3T token에서 instruction tuning 뒤 2.3T보다 2% 넘게 나빴다([[overtrained-lms-harder-to-finetune]]).
- 새 데이터에서 re-warming: 더 높은 re-warm peak는 원래 distribution의 forgetting을 높였다([[continual-pretraining-rewarm-replay]], §6.1.2).
- Long context의 QK-norm: RoPE variant와 MMLU 0.34점 이내이면서 8B variant 세 개 중 최저 65K needle score([[rope-nope-hybrid-attention]]).

**(c) 이 단계에서의 측정 방법.**
- 데이터 변경 decay마다 변경 없는 mixture에 대한 control decay 실행([[olmo-2]], 표 11 설계).
- 실험 전 development와 held-out suite 선언. OLMo 2는 held-out suite를 “not used for model development decisions”로 유지했다([[olmo-2]], §2.5).
- 단일 learning rate가 아니라 최소 3 order of magnitude에 걸친 learning-rate sensitivity curve로 hyperparameter 선택 측정([[small-scale-proxies-instabilities]], §2.2).
- Small-scale data와 recipe decision에는 small scale에서 candidate를 분리하는 continuous likelihood metric을 쓰고, larger run에 대해 decision accuracy 확인([[datadecide]]).
- Post-training용 checkpoint는 OOD task에서 fine-tuned version을 평가한다. Base-model score는 단조 개선됐지만 fine-tuned score는 하락했기 때문이다([[overtrained-lms-harder-to-finetune]], §2.2).
- 알려진 측정 오류: single-seed ablation(OLMo 2, MiniCPM 표), partial held-out contamination(OLMo 2 GSM8K), 한 run checkpoint 비교의 confounded final learning rate([[overtrained-lms-harder-to-finetune]], §3.2).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| 실제 학습하지 않을 계획 budget에 cosine horizon 설정 | η가 η_min보다 높은데 학습이 멈춤; stop 시 loss가 완료된 decay의 loss보다 높음 | 최종 step에서 η 기록; 같은 checkpoint에서 branch한 짧은 decay와 비교([[minicpm]] §4.3 설계) |
| Decay가 만든 gain을 데이터 변경에 돌림 | 새 데이터와 무관한 suite를 포함한 모든 suite에서 anneal gain | 변경 없는 mixture에서 같은 decay 실행(OLMo 2 표 11 PT Mix 행) |
| Development benchmark에서만 annealing gain 보고 | Target benchmark 큰 gain, held-out 결과 없음 | 실험 전에 선언한 held-out suite 보고; annealing data와 evaluation set overlap 확인([[llama-3]] §3.1.3) |
| Noise scale을 한참 넘어서 batch size와 함께 LR 선형 scaling | Batch 증가 뒤 divergence 또는 speedup 없음 | Local/global gradient norm에서 B_simple 추정([[large-batch-training-noise-scale]] 부록 A.1); 두 batch size에서 steps-to-target 비교 |
| 학습한 token에 대해 critical batch size 초과 | 같은 loss에 더 많은 token 필요; 절약한 step이 batch 증가보다 적음 | Batch size에 대한 target loss step plot; 20%-overhead point 찾기([[critical-batch-size-pretraining]] §3.1) |
| μP 또는 fitted rule 없이 width 간 learning rate 복사 | Proxy의 최적 LR가 target width에서 diverge 또는 underperform | 두 width에서 LR sweep; μP 아래에서는 optimum이 이동하지 않아야 함([[weight-init]] 발췌, Tensor Programs V 그림 1) |
| Learning-rate crossover를 지나기 전에 proxy run 종료 | Large run 후반에 learning-rate ranking이 역전 | 관측 crossover 뒤까지 최소 두 candidate 연장; OLMo 2는 200B token 뒤([[olmo-2]] §4.1) |
| Reordered norm을 post-norm으로 설명하거나 pre-norm model의 final norm 제거 | Final hidden-state norm이 depth와 함께 자라고 output logit도 자라거나, 구현이 add마다 stream을 normalize | Code에서 block equation 출력; initialization에서 layer별 residual norm 기록([[batch-vs-layer-norm]] 발췌) |
| 높은 LR에서 attention-logit 또는 output-logit monitoring 없음 | Training 후반의 느린 divergence; 매우 음의 output logit mean | Layer별 max attention logit과 output logit mean 기록; max attention logit이 1e4 초과한 모든 run은 diverge([[small-scale-proxies-instabilities]] 그림 9) |
| AdamW ε이 gradient RMS에 가까움 | Scale이 커지며 update size 감소; gradient RMS collapse | Layer별 gradient RMS를 ε과 비교해 기록([[small-scale-proxies-instabilities]] §3.4) |
| Run 중 fused와 manual z-loss implementation 전환 | Cross-entropy는 같아 보이지만 전환 뒤 z-loss curve가 갈라짐 | Run마다 implementation 하나 유지; batch 하나에서 backward value 비교([[olmo-2]] §3.3.3) |
| Long-context evaluation 없이 QK-norm 채택 | Short-task score는 그대로인데 needle/retrieval score 하락 | SFT 뒤 target length에서 retrieval 평가; attention entropy 기록([[rope-nope-hybrid-attention]]) |
| Base-model score만으로 pre-training checkpoint 선택 | 후속 fine-tuned model이 이전 checkpoint보다 held-out task에서 나쁨 | Tuned LR로 candidate checkpoint 둘 이상 fine-tune하고 OOD score 비교([[overtrained-lms-harder-to-finetune]]) |

## 이해 점검

1. OLMo 2 표 11에서 변경 없는 mixture의 anneal은 MMLU를 2.0점 올리고 GSM*를 1.5점 낮췄다. 이 행이 최종 mixture(46.5)와 control(27.0)의 GSM* 19.5점 차이에 대해 무엇을 결론 내리게 하는지, held-out math task에 대해서는 무엇을 결론 내리지 못하게 하는지 설명하라.
2. ε_opt(B) = ε_max / (1 + B_noise/B)를 사용하여, batch 256에서 tuning한 learning rate가 한 run에서는 batch 512에 거의 선형 scaling될 수 있지만 같은 model의 다른 run에서는 선형과 크게 다를 수 있는 이유를 설명하라. 두 run에서 어떤 quantity가 다른가?
3. Zhang et al.은 critical batch size가 주로 학습한 token에 의존한다고 발견한다. 이것이 fixed model에서 학습 중 성장하는 batch-size schedule이 정당화될 수 있음을 뜻하는 이유와, 2.87T token에서 Llama 3의 batch 증가가 critical batch size 아래인지 시험할 측정을 설명하라.
4. Step 1에서 AdamW update는 대략 −η · sign(g)다. 이것이 Xiong et al. Theorem 1과 함께, 실험에서 Post-LN은 warmup이 필요하고 Pre-LN은 warmup 없이 학습한 이유를 어떻게 설명하는지 말하라. 설명의 어느 부분이 derivation이고 어느 부분이 interpretation인가?
5. OLMo 2의 reordered norm은 sub-layer output을 normalize하지만 residual stream은 normalize하지 않는다. 이것이 depth에 따른 residual-stream norm 효과에서 post-norm과 다른 이유와, QK-norm도 함께 적용했을 때만 spike score 0.108 → 0.069가 나온 논문 결과가 두 변경 중 어느 쪽이 더 중요한지 보여 주지 못하는 이유를 설명하라.
6. Width 256의 μP proxy가 hidden matrix에 Adam LR 0.01을 찾았다. Tensor Programs V 표 3 아래 width 4096의 hidden-matrix learning rate를 유도하고, proxy에서 찾은 weight decay가 전이하지 않을 수 있는 이유를 설명하라.
7. Springer et al.은 한 training run에서 2.3T 및 3T token의 OLMo-1B checkpoint를 비교한다. Checkpoint별 learning rate가 이 비교를 왜 confound하는지와 controlled experiment가 confound를 어떻게 제거하는지 설명하라.
8. 팀이 높은-learning-rate divergence를 고치려고 QK-norm을 넣고 128K로 context를 확장하려 한다. Wortsman et al.과 Cohere 8B 결과를 사용해 각 normalization 선택이 attention logit에서 무엇을 바꾸는지 설명하고, extension 전에 long-context regression을 탐지할 최소 평가를 설계하라.

## 연결

- 의존: ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch(AdamW ε, z-loss implementation 차이, logit에 대한 precision effect).
- 순서상 이전: ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch.
- 순서상 다음: ch-04 — Sequence Packing, Loss Masking, and Chat Templates.
- ch-00 — What General Capability Means and How It Is Measured(§2와 Generalization lens에서 쓰는 development 대 held-out suite).
- ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability(§1.1과 §7의 Adam moment 및 update size).
- ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen(§3 batch size를 정하는 parallelism).
- ch-07 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures(§5에서 진단한 spike와 divergence).
- ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability(DeepSeek LLM, Porian et al., cooldown 기반 scaling sweep).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL(§2 decay phase의 data 선택).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining(§6 re-warming과 replay).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression(§6의 RoPE base, PI, YaRN, data).
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate(§7 plasticity 결과 뒤 fine-tuning learning rate).

## 출처

- [[lr-schedules]] — 장 발췌: inverse square root(Vaswani §5.3), WSD 정의(MiniCPM 식 1), multi-step(DeepSeek LLM §2.3), truncated cosine(OLMo 2 config), Llama 3 405B schedule; 지지되지 않는 수치 목록 제거.
- [[minicpm]] — 장 발췌: WSD, decay 길이, cosine horizon, SFT data annealing(표 1), μP operation과 search 결과, batch-size fit.
- [[cooldown-scaling-beyond-fixed-durations]], [[cooldown-scaling-beyond-fixed-durations-recipe]] — (1-sqrt) cooldown, cooldown 길이, 1B downstream 비교, re-warming 주장.
- [[deepseek-llm]] — 장 발췌: multi-step schedule, 표 2 설정, 도출 검사가 있는 η_opt·B_opt fit.
- [[large-batch-training-noise-scale]] — noise scale, steps–examples trade-off, ε_opt(B), Adam learning-rate exponent.
- [[critical-batch-size-pretraining]] — 장 발췌: 20%-overhead 정의 및 model/data size에 대한 B* fit.
- [[resolving-scaling-discrepancies]], [[resolving-scaling-discrepancies-recipe]] — 적합한 batch-size·learning-rate law, small batch의 β₂, 901M 처방.
- [[weight-init]] — 장 발췌: Glorot·He variance, GPT-2 residual scaling, OLMo 2 init ablation, μP 표 1·표 3, confound가 있는 GPT-3 6.7B μTransfer.
- [[batch-vs-layer-norm]] — 장 발췌: LayerNorm·RMSNorm, RMSNorm running-time 주장, Xiong et al. Lemma 2·Theorem 1, warmup 실험, OLMo 2 norm placement, Qwen3 QK-Norm 설명.
- [[olmo-2]] — 장 발췌: stability ablation(spike score), 표 3 schedule·config, z-loss 충돌, learning-rate crossover·표 8, mid-training 표 9·11.
- [[small-scale-proxies-instabilities]] — attention-logit growth, output-logit divergence, qk-layernorm, z-loss, warmup, μP, ε, 4.8B divergence forecast.
- [[llama-3]], [[llama-3-recipe]] — 405B schedule·batch ramp, annealing, 8B 대 405B의 GSM8K/MATH annealing, anneal-as-data-evaluation, long-context stage.
- [[datadecide]] — pre-training data 선택에 대한 small-scale ranking의 decision accuracy.
- [[overtrained-lms-harder-to-finetune]] — 장 발췌: OLMo-1B catastrophic overtraining, progressive sensitivity, learning-rate dependence.
- [[continual-pretraining-rewarm-replay]] — 장 발췌: re-warming peak 대 adaptation·forgetting, replay fraction.
- [[prolong]], [[prolong-recipe]] — long-context continued-training learning rate, batch, RoPE base, 512K learning-rate 충돌.
- [[kimi-k1-5-recipe]] — 128K에서 SFT learning-rate re-warm; long-context pre-training learning rate 보고 안 함.
- [[yarn]] — extension fine-tuning 설정과 attention temperature 수식.
- [[rope-nope-hybrid-attention]] — 장 발췌: 8B QK-Norm variant의 needle score와 attention entropy.
- [[llama-4]] — Scout의 inference-time attention temperature scaling(발표된 수식 없음).
- [[qwen-2.5]] — Qwen3 QK-Norm과 비교할 QKV bias가 있는 pre-norm RMSNorm.
