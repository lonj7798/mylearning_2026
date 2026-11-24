<!-- chapter: ch-32a
     track: midtraining
     kind: content
     title: Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining
     deps: [ch-32]
     sources: [[continual-pretraining-rewarm-replay]], [[paloma]], [[grpo]], [[grpo-recipe]], [[qwen-2-5-coder]], [[qwen-2-5-math]], [[instruction-pretraining]], [[synthetic-continued-pretraining]], [[merging-in-pretraining]], [[cooldown-scaling-beyond-fixed-durations]], [[kimi-k2-5]], [[llama-3]], [[rephrasing-the-web]]
     figures: figures/replay-tradeoff.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32a — 망각 없이 계속 사전학습하기: replay, 학습률 재-warming, 합성 continued pretraining

> **핵심 통찰.** 새 데이터로 continued pretraining을 하면 새 데이터 적응과 기존 데이터 능력 손실 사이에 교환관계가 생기며, re-warming/re-decaying과 replay가 이를 제어한다. 통제된 405M run에서 새 데이터 적응에는 학습률을 다시 올린 뒤 다시 감쇠시키는 과정이 필요했다. 기존 데이터 일부를 replay하면 기존 데이터 망각은 줄었고, 최대 25%까지 시험한 각 replay 비율에서 새 데이터 loss 증가는 0.00–0.05였다. 이 replay 결과는 같은 언어 안의 shift에서도 10B에서 유지됐다. 300B Pile token 후 German token 200B를 학습하면 25% replay는 Pile loss 증가를 1.39에서 0.16으로 줄이고 German loss는 1.11에서 1.16으로 올렸다. 이는 두 데이터셋을 함께 처음부터 학습한 모델의 평균 loss와 같다([[continual-pretraining-rewarm-replay]], Table 2). Specialist mixture에서도 같다. 100% code Qwen2.5-Coder-7B는 MMLU 42.8이지만 code:text:math = 70:20:10 mixture는 수학 점수도 높이면서 MMLU 62.9를 기록했다([[qwen-2-5-coder]], Table 3). 새 corpus가 약 1M token이면 next-token 학습만으로 쓸 수 있는 지식이 추가되지 않을 수 있다. raw 1.3M token은 closed-book QA를 39.49%에서 38.15%로 낮췄지만 EntiGraph 합성 455M token은 56.22%로 올렸다([[synthetic-continued-pretraining]], Fig. 2).
>
> **지침.** cosine schedule로 낮은 학습률까지 학습된 base model을 100B개 이상 새 token으로 계속 학습하고 pre-decay checkpoint가 없다면, 원래 peak 근처까지 re-warm한 뒤 cosine re-decay하라. Ibrahim et al.에서 일정한 낮은 학습률은 새 데이터에 가장 적게 적응했다([[continual-pretraining-rewarm-replay]] §6.1.2). pre-decay 또는 constant-phase checkpoint가 있으면 거기서 시작하라. re-warming은 데이터가 바뀌지 않아도 loss를 높였고(§7.1), constant-phase checkpoint는 같은 분포 경계에서 망각이 거의 없었다(§7.4; 분포 shift에서는 미시험). 언어·domain이 달라지면 기존 또는 일반 데이터를 replay하고 짧은 sweep으로 비율을 정하라. 405M에서 1% replay도 망각을 줄였고 약한 shift에는 5%, 언어 shift에는 25%가 선택됐다. 약 1M-token corpus는 source-grounded synthetic entity-relation text로 확장하라. raw 학습은 QA를 낮췄고 paraphrase-only 확장은 규모 증가 효과가 더 나빴다([[synthetic-continued-pretraining]], Fig. 2). Instruction-augmented format([[instruction-pretraining]])은 1B token domain run에서만 시험됐고 1M-token 규모에서는 시험되지 않았다. 정확한 초기화 checkpoint에 대한 domain별 held-out perplexity와 downstream delta를 보고하라. §4의 7개 CPT run 중 5개는 시작 checkpoint의 general score를 싣지 않는다.

## 범용 모델에서 이 장이 중요한 이유

이 장은 이미 넓은 능력을 가진 base model에 modality(Kimi K2.5), code/math specialization(Qwen2.5-Coder, Qwen2.5-Math, DeepSeekMath), 새 언어 또는 최신 web data(Ibrahim et al.), 작은 문서 모음(EntiGraph, instruction pre-training)을 추가하는 run을 다룬다. 이 과정은 기존 능력 일부를 제거할 수 있다. 따라서 이 장은 모델을 좁히지 않고 새 데이터를 더하는 mid-training(ch-32)의 부분을 다룬다. 자체 regression 문제가 있는 context-length 확장은 ch-32b에서 다룬다.

Pipeline 위치는 `pre-training → mid-training 및 continued pretraining(이 장) → SFT → preference optimization → RL → evaluation`이다. 여기서 생긴 망각은 이후 stage 모두에 전달된다. SFT 전 multilingual 또는 general-knowledge 능력을 잃은 모델은 그 능력이 없는 post-training mixture만으로 회복할 수 없다(해석). 측정 질문은 다음과 같다.

1. **Adaptation:** 새 데이터 loss가 얼마나 내리고 target benchmark가 얼마나 오르는가?
2. **Forgetting:** 기존 데이터 loss가 domain별로 얼마나 오르고, 시작 checkpoint가 하던 어떤 downstream score가 떨어지는가?
3. **Cost:** 전체 데이터를 처음부터 재학습하는 것에 비해 token과 compute를 얼마나 쓰는가?

SFT/RL 뒤의 fine-tuning 망각 mechanism과 alignment-tax 측정은 ch-30a에서 다룬다. 이 장은 같은 논리를 사전학습 규모에 적용한다. Run 규모는 Raw CPT의 약 5M tokens seen(raw 1.3M × 4 epoch)부터 Kimi K2.5의 15T token까지다.

## §1 Continued pretraining의 망각과 측정

**정의.** *Continued pretraining*(또는 *continual pretraining*, CPT)은 pretrained model이 이전 데이터셋 D0 뒤에 D1에서, 모든 token의 next-token prediction이라는 사전학습 objective로 추가 학습하는 것이다. *Adaptation*은 held-out D1 loss 감소 또는 D1 관련 task score 상승이다. *Forgetting*은 held-out D0 loss 증가 또는 시작 checkpoint가 풀던 task score 하락이다. *Domain-adaptive* CPT는 D1이 D0보다 좁은 code, math, biomedicine인 경우이며 범용 모델의 망각 위험이 가장 크다.

**문제.** Ibrahim et al.은 naive continued training의 두 실패를 새 데이터셋 최적화 실패(poor adaptation)와 기존 데이터셋에서의 상당한 capability loss(catastrophic forgetting)로 이름 붙인다([[continual-pretraining-rewarm-replay]] §1). 이 장의 control은 한쪽을 개선하며 다른 쪽을 희생시킬 수 있으므로 둘 다 측정해야 한다.

**수식.**

```
Δforget(d) = L_d(θ_after) − L_d(θ_before)
Δadapt     = L_D1(θ_before) − L_D1(θ_after)
Δtask(k)   = s_k(θ_after) − s_k(θ_before)
```

- `L_d(θ)`는 parameter θ가 domain d held-out 문서에서 내는 token당 평균 cross-entropy(nat)다.
- `θ_before`는 CPT를 초기화한 정확한 checkpoint, `θ_after`는 CPT 후 checkpoint다.
- `D1`은 새 데이터셋이고 `s_k`는 고정 prompt format과 few-shot count로 측정한 downstream task k score다.

Paloma는 문서 집합 N의 perplexity를 `exp(−ℓ / T(N))`로 정의한다. ℓ은 token log-likelihood 합, `T(N)`은 token 수다. Paloma는 token-weighted micro average 대신 domain별 perplexity를 동등 가중하는 macro average `|D|⁻¹ Σ_d perplexity(d)`를 보고한다([[paloma]] §3, §4.1).

**계산 예(도출).** Base model의 A loss가 2.0(perplexity `e^2.0 = 7.39`), B loss가 3.0(perplexity 20.09)이고 validation에는 A 900K token, B 100K token이 있다고 하자.

1. 이전 micro perplexity는 `exp(0.9·2.0 + 0.1·3.0) = exp(2.10) = 8.17`, macro perplexity는 `(7.39 + 20.09)/2 = 13.74`다.
2. CPT 뒤 A는 그대로지만 B loss가 3.5(perplexity 33.12)가 된다.
3. 이후 micro는 `exp(0.9·2.0 + 0.1·3.5) = exp(2.15) = 8.58`, 즉 5% 증가한다. Macro는 `(7.389 + 33.115)/2 = 20.25`, 즉 47% 증가한다.

Token-weighted 값은 작은 B domain의 망각을 숨긴다. 이것이 `Δforget(d)`를 domain별로 추적해야 하는 이유다.

**Domain별 측정이 결론을 바꾼다는 증거.**

- 1B controlled baseline 6개 중 C4 및 mC4-en model은 약 20B-token과 약 150B-token checkpoint 사이에 각각 65개, 43개 Paloma domain에서 나빠졌다. Curated non-web source를 포함한 baseline은 안정적 격차를 유지하며 향상됐다([[paloma]] App. D.1.1). 저자들은 domain별 perplexity가 하나의 held-out loss가 감추는 격차를 드러낸다고 제시한다. 단일 연구 결과다.
- Paloma에서 단일 perplexity source는 시험한 downstream task 8개 전체와 상관하지 않았다. 예를 들어 c4-en 대 HellaSwag Spearman `ρ`는 −0.77이고 RedPajama 대 HellaSwag는 0.94다. 그러므로 perplexity panel은 downstream delta와 함께 읽어야 한다([[paloma]] App. A, Table 3).

**CPT 특유의 측정 오류.**

1. **잘못된 기준 checkpoint.** DeepSeekMath-Base 7B는 learning-rate decay 직전 DeepSeek-Coder-Base-v1.5로 초기화됐다. MMLU는 그 checkpoint에서 42.9%, 출시된 decay 완료 coder base에서 49.1%, math-heavy 500B token 후 54.9%다. 실제 초기화 대비 +12.0, 출시 모델 대비 +5.8이다([[grpo]], Table 4). 출시 모델만 비교하면 decay 효과와 새 데이터 효과를 섞는다.
2. **기준 row 누락.** Qwen2.5-Math base는 Qwen2.5 base에서 초기화했지만 Table 2에는 Qwen2.5가 아닌 Qwen2 base가 실려 있다([[qwen-2-5-math]]). Retention delta가 보고되지 않았다.
3. **작은 규모 평가 noise.** 405M에서 5개 모델 English evaluation average 범위는 1.19점이며 저자들은 차이가 "likely not significant"하다고 썼다. 모든 run은 single-seed다([[continual-pretraining-rewarm-replay]] §6.3.2, §8).
4. **Train-validation contamination.** Ibrahim et al.은 German validation을 German train과 deduplicate하지 않았다([[continual-pretraining-rewarm-replay]] §8). Paloma G1 rule은 evaluation paragraph와 일치하는 training document를 제거한다([[paloma]] §3). D1 validation을 decontaminate하지 않으면 adaptation을 과대평가할 수 있다(해석).

**범용 모델에 대한 함의.** Continued run의 최소 망각 report는 이전 mixture의 주요 source별 held-out perplexity(macro-averaged, decontaminated), D1 held-out loss, general suite downstream score다. 모두 동일 harness로 `θ_before`와 `θ_after`에서 계산해야 한다.

## §2 학습률 재-warming과 재-decaying

**정의.** *Re-warming*은 base model의 schedule이 낮은 값에서 끝난 뒤 continued pretraining을 시작하면서 학습률을 다시 올리는 것이다. *Re-decaying*은 continued run 동안 학습률을 다시 내리는 것으로, 보통 새 token budget 크기에 맞춘 cosine schedule을 쓴다.

**문제.** Ibrahim et al.은 Llama, Llama 2, Mistral, Gemma를 예로 들어 성능이 높은 open-source LLM 대부분이 학습 끝에 학습률을 작은 값까지 감쇠시킨다고 설명한다([[continual-pretraining-rewarm-replay]] §1). 이들의 run에서 최종 최소 학습률로 계속 학습한 경우는 D1 적응이 가장 적었다(§6.1.2). 학습률을 올리면 적응은 생기지만, parameter도 D0 solution에서 멀어진다.

**Mechanism.**

1. D0 학습의 최종 checkpoint를 불러온다. Ibrahim et al.은 open-weight model에는 optimizer state가 없는 경우가 많으므로 optimizer state를 reset한다(§5.2).
2. `T_warmup` step 동안 0에서 `η_max`까지 선형 warmup한다.
3. D1 budget 끝에서 `η_min`에 도달하도록 cosine으로 감쇠시킨다.
4. 선택적으로 모든 batch에 replay data를 섞는다(§3).

**수식**(Ibrahim et al. Eq. 1–2):

```
η_t = η_max · t / T_warmup                                                   for t ≤ t_ann = T_warmup
η_t = η_min + (η_max − η_min)/2 · ( cos(π · (t − t_ann)/(t_end − t_ann)) + 1 )   for t_ann < t ≤ t_end
```

- `η_t`는 iteration t의 학습률, `η_max`는 peak, `η_min`은 최종값이다. 논문의 run에서 `η_min = 0.1 · η_max`다.
- `T_warmup`은 warmup iteration 수(D1 iteration의 1%)이고, `t_end`는 D1 budget의 마지막 iteration이다.

**계산 예(도출).** 405M weak-shift run은 132,366 iteration 동안 `η_max = 3·10^-4`, `η_min = 3·10^-5`를 쓴다. Batch는 token 2048개짜리 sequence 1104개다(batch와 sequence length는 Table 13, 학습률은 Table 14, §5.3, §6.1.1).

1. Iteration당 token 수는 `1104 × 2048 = 2,260,992`다. 132,366 iteration이면 `299.3B` token으로 SlimPajama subset 크기다.
2. Warmup은 `132,366`의 1%, 즉 `1,324` iteration이다.
3. 감쇠 구간 25% 지점에서는 `cos(π/4) = 0.7071`이고, `η = 3·10^-5 + 1.35·10^-4 · 1.7071 = 2.60·10^-4`다.
4. 감쇠 구간 50% 지점에서는 `cos(π/2) = 0`이고, `η = 3·10^-5 + 1.35·10^-4 = 1.65·10^-4`다.
5. 끝에서는 `cos(π) = −1`이므로 `η = 3·10^-5`다.

**증거.**

- **Re-warm 대 constant.** 405M에서 두 shift 모두 constant `η_min` model이 D0에서 망각이 가장 적었고, re-warm/re-decay model은 새 데이터셋에 유의하게 더 잘 적응했다([[continual-pretraining-rewarm-replay]] §6.1.2, Fig. 4). `η_max ∈ {1.5, 3, 6}·10^-4`인 re-warm run 사이에서는 `η_max`가 클수록 망각도 적응도 더 컸다. 단일 연구, one seed 결과다.
- **Warmup 길이.** 0%, 0.5%, 1%, 2% warmup은 50B token 뒤에 비교적 비슷한 망각과 적응을 보였다. Warmup이 없으면 일시적인 loss spike가 생겼다(§6.1.1, Fig. 3).
- **데이터가 바뀌지 않아도 생기는 re-warming 효과.** Pile 자체에서 계속 학습할 때 3·10^-4까지 re-warm하면 Pile validation loss peak가 0.1, 6·10^-4까지 올리면 0.2 증가했다. SlimPajama에서는 각각 0.35, 0.45였다(§7.1, Fig. 8). 저자들은 data shift와 무관하게 re-warming 자체가 초기 loss 증가의 중요한 원인으로 보인다고 결론낸다(해석, 단일 연구).

**Re-warming을 피하는 대안.**

- **Infinite learning-rate schedule.** Warmup 뒤 `η_const`까지 cooldown하고, 향후 run이 재개할 수 있는 constant phase를 유지한 뒤 release 전 짧게 final anneal한다. Shift가 없는 SlimPajama 100B-token split 세 개에서 405M으로 실험하면 이 schedule은 repeated cosine과 유사한 loss에 도달하면서 dataset boundary 사이 망각은 거의 없었다(§7.4, Fig. 10). Distribution shift는 시험하지 않았다(§8). Shift된 data에는 열린 질문이다.
- **실무의 pre-decay checkpoint.** DeepSeekMath는 learning-rate decay 전 coder checkpoint에서 시작했고([[grpo-recipe]]), Kimi K2.5의 joint vision-text stage는 near-end Kimi K2 checkpoint에서 계속 학습한다([[kimi-k2-5]] §4.3). Hägele et al.은 constant-plus-cooldown schedule의 pre-cooldown checkpoint라면 re-warming 없이 높은 학습률로 계속할 수 있다고 주장한다. 그러나 새 data로 continual training한 실험은 보고하지 않는다([[cooldown-scaling-beyond-fixed-durations]] §2, §3.1).

**조건과 한계.** Controlled evidence는 dense 405M 및 10B(embedding 포함 9.6B, §5.3) model, 두 transition, one seed, optimizer-state reset만 다룬다. 10B model은 논문 안에서도 충돌한다. §6.4는 두 크기 모두 3·10^-4까지 re-warm한다고 하지만, Table 14는 10B cosine schedule의 `η_max`를 1.2·10^-4, `η_min`을 1.2·10^-5로 적는다. 이 source의 학습률 값은 모델 크기 사이로 옮겨 쓰면 안 된다.

**함의.** 범용 모델에서 학습률 peak는 data 선택 전부터 adaptation-forgetting 교환관계를 정한다. Decay됐는지 여부를 포함해 어느 checkpoint에서 시작했는지 기록해야 한다. 그것이 필요한 schedule과 `Δforget`의 기준 둘 다를 바꾸기 때문이다.

## §3 일반 데이터 replay

**정의.** *Replay*는 이전 분포 D0의 데이터 또는 D0를 대신하는 일반 corpus를 continued pretraining batch에 섞는 방법이다. *Compute-equivalent replay*는 총 token 수를 고정한다. 즉 replay token을 D1 token에 더하는 것이 아니라 D1 token을 그만큼 대체한다([[continual-pretraining-rewarm-replay]] §4.2).

**문제.** Re-warming은 망각을 증가시킨다(§2). 질문은 그 망각을 제거하려면 이전 데이터가 얼마나 필요한지, 그리고 그 대가로 새 데이터 적응을 얼마나 잃는지다.

**수식.**

```
tokens_D1_new = (1 − x) · B
tokens_replay = x · B
```

- `x`는 replay 비율, 즉 각 batch에서 D0로부터 뽑는 비중이다. `B`는 continued run의 전체 token budget이다.

**계산 예.** 강한 shift run의 `B = 200B` German token이다(§6.2). 대화형 그림 [figures/replay-tradeoff.html](figures/replay-tradeoff.html)에서 weak shift와 strong shift를 바꾸면 각 replay 비율의 token 분할과 loss 표를 볼 수 있다.

1. `x = 0.25`이면 German token 150B와 Pile token 50B를 학습한다.
2. Pile loss는 300B Pile-only model에서 2.17, replay 없이 German을 학습한 뒤 3.56, 25% replay에서 2.33이다(Tables 2–3). 따라서 replay 없이는 `Δforget = +1.39`, 25% replay에서는 `+0.16`이다. Perplexity로 바꾸면 `exp(2.17) = 8.76`, `exp(3.56) = 35.2`, `exp(2.33) = 10.3`이다.
3. German loss는 replay 없이 1.11, replay를 넣으면 1.16이다(`Δ = +0.05`; perplexity `3.03 → 3.19`).
4. 두 loss의 평균은 replay 없이 2.34, replay를 넣으면 1.75이며, Pile ∪ German으로 처음부터 500B token을 학습한 model도 1.75다.

**증거(405M, Table 2).**

| Shift | Replay | Pile loss | D1 loss | 평균 |
|---|---|---|---|---|
| Pile → SlimPajama (300B) | 0% / 0.5% / 1% / 5% / 10% / 50% | 2.44 / 2.27 / 2.26 / 2.23 / 2.21 / 2.16 | 2.50 / 2.50 / 2.50 / 2.51 / 2.51 / 2.54 | 2.47 / 2.39 / 2.38 / 2.37 / 2.36 / 2.35 |
| 600B Pile ∪ SlimPajama from scratch | — | 2.17 | 2.53 | 2.35 |
| Pile → German (200B) | 0% / 1% / 5% / 10% / 25% / 50% | 3.56 / 2.83 / 2.57 / 2.46 / 2.33 / 2.24 | 1.11 / 1.12 / 1.12 / 1.13 / 1.16 / 1.22 | 2.34 / 1.97 / 1.85 / 1.80 / 1.75 / 1.73 |
| 500B Pile ∪ German from scratch | — | 2.26 | 1.25 | 1.75 |

- “시험한 가장 낮은 replay 비율인 1%조차 replay 없는 baseline과 비교해 Pile의 망각을 유의하게 줄인다”라고 저자들은 썼다(§6.2). 비슷한 평균 loss에서 더 높은 비율보다 더 많은 새 token을 볼 수 있기 때문에 weak shift에는 5%, strong shift에는 25%를 선택했다(§6.3). 결과는 단일 연구다.
- Downstream(Table 3)에서 English evaluation average는 5% replay의 Pile → SlimPajama가 35.14, union model이 34.30이었다. 25% replay의 Pile → German은 32.48, union model은 32.43, replay 없는 경우는 29.20이었다.
- *Scale.* Replay가 없으면 Pile loss는 10B에서 0.23, 405M에서 0.27 올랐다. 5% replay는 그 증가를 각각 0.19와 0.21 줄였다. 10B에서 5%-replay model의 평균 loss는 1.89, union model은 1.87이고 evaluation average는 47.68 대 48.00이었다. 그러나 MMLU는 28.79 대 37.78이었다(Tables 4–5, §6.4.2). 저자들은 제한된 training data를 의심하지만 그 격차를 설명하지 못했다. 열린 질문이다.

**다른 run의 replay 및 일반 데이터 비율.**

- DeepSeekMath-Base 7B(500B tokens)는 math corpus 56%, AlgebraicStack 4%, arXiv 10%, GitHub code 20%, Common Crawl natural language 10%를 사용했다([[grpo-recipe]], §2.3). HumanEval은 초기 checkpoint에서 40.2%, 이후 40.9%였고 저자들은 이를 유지됐다고 기술한다([[grpo]], Table 4).
- Llama 3 8B의 EntiGraph CPT는 “1B RedPajama token을 사용해 rate 0.1로 replay”했다([[synthetic-continued-pretraining]] App. C).
- Ibrahim et al. §2는 Glorioso et al.(50B-token decay phase에서 60% replay)과 DeepSeek-AI(DeepSeek-V2 continued pretraining 6T token에 걸쳐 30% replay)를 인용한다. 이 수치는 저자들의 기술이며, 이 course에서 다시 검증하지 않았다.
- Kimi K2.5는 text model에 vision token을 추가한다. 고정 budget ablation에서 text knowledge는 training 시작 시점부터 vision을 넣고 vision:text = 10:90으로 한 경우 45.5, 50% 지점부터 넣고 20:80으로 한 경우 43.9, 80% 지점부터 넣고 50:50으로 한 경우 43.1이었다. Mid-fusion과 late-fusion run은 vision data가 들어온 시점에 text performance가 떨어졌다가 다시 회복하는 “dip-and-recover”를 보였다([[kimi-k2-5]] Table 1, App. B.1). 이 ablation의 model size와 budget은 보고되지 않았다.

**조건과 한계.** Replay는 원래 D0 순서대로 읽었으며 replay sample 선택은 연구하지 않았다(§4.2). 공개되지 않은 pretraining data를 가진 open-weight model처럼 D0를 확보할 수 없으면 일반 corpus가 이를 대체한다. Ibrahim et al.은 비공개 데이터를 위한 replay buffer 구축을 future work로 든다(§9). Domain을 순차적으로 하나씩 갱신하면 좋지 않은 결과가 났고, 저자들은 domain mixture에서 계속 학습하라고 권한다(§5.2, App. A.1).

**함의.** 이 run에서 replay는 선택된 비율에서 D1 loss를 0.01–0.05 올리는 비용으로 D0 망각을 줄이는 control이었다(Table 2). 범용 model의 replay source는 무작위 web text 일부만이 아니라, release가 약속하는 capability인 언어·code·math·knowledge를 덮어야 한다.

## §4 Specialist 보고서의 domain-adaptive continued pretraining

**문제.** 이 course의 자료 중 code 및 math specialist 보고서는 data mixture를 공개한 domain-adaptive run이다. 아래 표에 넣은 7개 run 중 5개는 초기 checkpoint의 일반 benchmark 점수를 싣지 않는다. 따라서 그 5개에는 retention delta를 계산할 수 없다.

**Qwen2.5-Coder.** Qwen2.5에서 시작해, file-level pretraining은 8,192-token context에서 5.2T token으로, 이어서 repo-level pretraining은 약 300B token을 32,768-token context에서 수행했다. 논문에 적힌 RoPE base는 `10,000 → 1,000,000`이다([[qwen-2-5-coder]] §3.2). Text data는 “Qwen2.5-Coder의 general capabilities를 보존하기 위해” Qwen2.5 pretraining corpus에서 가져왔다. Qwen2.5-Coder-7B의 mixture ablation(Table 3)은 다음과 같다.

| Code:Text:Math | Coding: Common | MATH | GSM8K | MMLU | CEval | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|
| 100:0:0 | 49.8 | 10.3 | 23.8 | 42.8 | 35.9 | 58.3 |
| 85:15:5 | 43.3 | 26.1 | 52.5 | 56.8 | 57.1 | 70.0 |
| 70:20:10 | 48.3 | 33.2 | 64.5 | 62.9 | 64.0 | 73.5 |

Table 3은 Coding 아래에 `Common`과 `BCB`도 싣지만, 위 표에서는 BCB 점수 `(40.3, 36.2, 38.3)`를 생략했다. 70:20:10 행은 100% code 행과 비교해 code 열에서 1.5점 낮고 MMLU에서는 20.1점 높다. 본문은 가운데 행을 85:10:5라고 설명하지만 표에는 85:15:5로 인쇄되어 있다. 100:0:0의 인쇄된 average도 31.3인데, 일곱 점수의 산술평균은 37.3이다. Ablation run의 token 수는 적혀 있지 않다. 단일 연구 결과다.

**DeepSeekMath.** Learning-rate decay 전의 code checkpoint에서 시작했고, 500B token 중 code 20%, natural language 10%를 썼다([[grpo-recipe]]). 일반 점수는 초기 checkpoint 대비 올라갔다. MMLU는 `42.9% → 54.9%`, BBH는 `42.9% → 59.5%`였다([[grpo]], Table 4). 1.3B ablation에서 code를 먼저, math를 나중에 학습하는 two-stage training은 HumanEval을 `25.0% → 12.2%`, MBPP를 `40.0% → 17.0%`로 낮췄다. 반면 one-stage mixed training은 각각 29.3%와 39.4%를 유지했다(Tables 6–7). 저자들은 1.3B model에는 두 영역을 함께 담을 capacity가 부족할 수 있다고 추측한다. 이는 해석이다.

**Qwen2.5-Math.** Qwen Math Corpus v2는 4K context에서 “over 1T” token이며, model은 Qwen2.5 base model에서 초기화한다([[qwen-2-5-math]] §2). 보고서는 mixture, general-data 비중, learning rate, math 밖의 일반 benchmark를 제시하지 않는다. 이전 세대인 Qwen2-Math-7B는 MMLU-STEM 65.7, Qwen2-7B는 67.6이었지만 Qwen2-Math는 중간 Qwen2 checkpoint에서 초기화했다. 그러므로 이것은 깨끗한 before/after delta가 아니다(Table 2).

**Run별 retention 보고** — 각 cell에는 source가 실제로 인쇄한 정보를 적었다.

| Run | 초기 checkpoint | CPT token | 일반 데이터 비중 | 초기값 대비 일반 benchmark |
|---|---|---:|---|---|
| Ibrahim et al. 405M/10B, Pile → SlimPajama | Pile checkpoint (decayed) | 300B | 5% replay | 보고됨(Tables 3–5) |
| DeepSeekMath-Base 7B | Coder-v1.5 pre-decay | 500B | code 20%, NL 10% | 보고됨(Table 4) |
| Qwen2.5-Coder-7B | Qwen2.5 (§1, Fig. 2) | 5.2T + 약 300B | text 20%(math 10% 추가) | 보고 안 됨(다른 coder와만 비교) |
| Qwen2.5-Math-7B | Qwen2.5 base (§2) | >1T | 보고 안 됨 | 보고 안 됨 |
| Llama 3 code expert | main pre-training run의 branch; checkpoint 불명([[llama-3]] §4.3.1) | 1T, code >85% | 보고 안 됨 | 보고 안 됨 |
| Instruction PT Llama3-8B (biomed, finance) | Llama3-8B | 4K steps × 0.25M tokens (도출 시 1B) | Cheng et al. (2023)과 “같은 mixing ratio”의 general instruction; 이 논문에는 비율 미기재 | 보고 안 됨 |
| EntiGraph CPT Llama 3 8B | Llama 3 8B Base | 455M × 2 epochs | RedPajama replay 10% | 보고 안 됨 |

**함의.** Qwen2.5-Coder Table 3 같은 mixture ablation은 specialist run에서 general data가 일반 점수를 보호하고 target과 인접한 점수도 올릴 수 있음을 보인다. 하지만 초기 checkpoint의 before/after 행이 없으면 specialist가 base model의 일반 능력을 실제로 보존했는지 판별할 수 없다. Specialist에서 merge하거나 distill하는 범용 program(ch-30c, ch-35)에는 그 행이 필요하다.

## §5 작은 corpus에서 새 지식을 추출 가능하게 만들기

**문제.** §4의 domain-adaptive run은 수천억 token을 사용한다. Yang et al.은 크기가 수백만 token일 수 있는 proprietary document store를 adaptation target으로 든다(§1). EntiGraph의 Table 1은 공개된 CPT corpus의 고유 token 수가 14.9B–620B인 것과 자신의 1.3M-token 설정을 대조한다([[synthetic-continued-pretraining]] Table 1). 이런 corpus에 직접 학습하면 실패할 수 있다. 1.3M QuALITY token으로 한 Raw CPT(4 epoch, 10% replay)는 closed-book QA 38.15%를 기록했으며, base model의 39.49%보다 낮았다(§4.2, App. C). 저자들은 좁은 corpus가 model의 “overall English capabilities”를 해칠 수 있고 raw corpus 안의 “limited diversity of knowledge representations”가 지식 획득을 제한한다고 가정한다. 일반 capability는 측정하지 않았으므로 이는 해석이다.

### 5.1 Instruction pre-training

**정의.** Instruction pre-training은 각 raw document를 그 document에서 생성한 instruction-response pair로 확장하고, 모든 token에 next-token loss를 적용해 학습한다([[instruction-pretraining]] §2).

**작동 방식.**

1. Context-based task dataset으로 Mistral-7B-v0.1에서 출발한 7B synthesizer를 fine-tune하여 raw text를 instruction-response pair로 바꾸게 한다.
2. Domain corpus에 여러 round로 실행한다. 각 training sequence가 few-shot example이 되도록 앞선 text와 pair를 앞에 붙인다. Text 하나마다 약 52 token의 pair가 약 5개 생성된다(§3.1).
3. 확장한 corpus를 general instruction data와 섞어 continued pretraining한다.

**근거.** PubMed Abstracts에서 continued training한 Llama3-8B의 biomedicine 평균은 61.3이었다. 같은 token 수의 vanilla continued pretraining은 58.4, 학습 전에는 53.6이었다. Financial news에서는 각각 74.7, 72.0, 70.1이었다(Table 3). Llama3-70B의 점수는 63.9와 71.9였다. Domain run은 batch당 0.25M token으로 4K step, LR `1e-5`를 사용했다(Table 14). Adapted model의 general-benchmark 점수는 보고하지 않았다. GPT-4는 합성 response의 정확도를 finance에서 69.8%, biomedicine에서 86.2%로 판단했다(Table 6). Limitations는 부정확한 response가 “may potentially mislead the pre-trained model”이라고 경고한다. 단일 연구 결과다.

### 5.2 EntiGraph를 사용한 synthetic continued pretraining

**정의.** Synthetic CPT는 generator `A_synth: D_source → D_synth`를 적용하고 `D_source` 대신 `D_synth`에서 continued pretraining한다([[synthetic-continued-pretraining]] Eq. 1). EntiGraph는 두 prompt로 구성된 generator다. 먼저 document에서 salient entity를 뽑고, 이어서 그 document에서 entity subset들이 어떻게 관계하는지 LM에게 분석하게 한다(§2.2).

**계산 예(도출).** Document 하나에서 `n = 20` entity가 나왔다고 하자. EntiGraph는 모든 pair와 triplet에 text를 만든다. `C(20,2) = 190`, `C(20,3) = 1,140`이므로 document 하나에 relation-analysis prompt가 1,330개다. 다양성의 원인은 같은 문장을 반복 paraphrase하는 것이 아니라 이 조합 수다. 전체 corpus에서는 1.3M source token이 455M synthetic token으로 바뀌었으므로 350배 확장이다. 보고된 한 개 8×H100 node의 6,090 tokens/s에서 455M token을 2 epoch 학습하면 `910M / 6,090 s ≈ 41.5 h`이며, App. C의 약 41시간 보고와 맞는다.

**근거.**

- Llama 3 8B의 closed-book QuALITY accuracy는 base 39.49%, Raw CPT 38.15%, EntiGraph CPT 56.22%였다. EntiGraph CPT는 455M token, 2 epoch, 10% replay, peak LR `5e-6` 설정이며 GPT-4의 51.30%도 넘었다(Fig. 2). Accuracy는 synthetic token 455M까지 log-linear하게 증가했다. Maini et al.에서 변형한 Rephrase baseline([[rephrasing-the-web]])은 38M token에서 멈췄고 scaling이 더 나빴다.
- Retrieval을 붙이면 base + RAG는 60.35%, EntiGraph CPT + RAG는 62.60%였다(Table 3). EntiGraph 단독은 16.73점을 더했고, RAG가 더한 20.86점의 80.2%다(`16.73 / 20.86`).
- 250M UltraChat token으로 instruction tuning한 뒤, 책의 closed-book summary는 Raw-CPT model보다 salient claim을 더 많이 포함하면서 새 false claim은 더 적었다(Fig. 3).

**조건과 한계.** Generator는 `gpt-4-turbo`였다. 저자들은 일부 책의 일부 subset을 fact-check했고, 더 어려운 content에서는 hallucinated relation이 나올 수 있으며 generator 지식이 distill될 수 있다고 적었다(§7.1). 8B model의 general-capability retention은 측정하지 않았다. 단일 corpus, 단일 연구 결과다.

**함의.** 시험한 한 corpus에서는 1.3M-token corpus의 지식이 455M token의 서로 다른 entity-relation text로 다시 표현된 뒤에야 closed-book으로 답할 수 있게 되었다(Fig. 2의 해석). 여기의 두 방법은 corpus를 다양하고 task 같은 text 또는 relation text로 바꾸고 general data를 mixture에 남긴다. 하지만 어느 방법도 general ability를 보존했는지 보고하지 않았으므로, 사용할 때는 §1의 retention panel이 필요하다.

## §6 Pretraining 중 모델 병합

**정의.** Pre-trained Model Averaging(PMA)은 pretraining run 하나에서 나온 N개 checkpoint를 평균낸다([[merging-in-pretraining]] §3).

```
M_avg = Σ_{i=1..N} w_i · M_i          V = T_{i+1} − T_i
SMA: w_i = 1/N     WMA: w_i ∝ i     EMA: M_avg^(i) = α·M_i + (1 − α)·M_avg^(i−1)
```

- `M_i`: checkpoint i의 parameter, `w_i`: 그 checkpoint의 weight, `T_i`: checkpoint i 시점의 누적 token, `V`: 고정 token interval, `α`: EMA smoothing factor이다.

**문제.** Constant-learning-rate checkpoint는 annealed checkpoint보다 loss가 높다. 그러나 annealing을 하면 run을 높은 learning rate로 계속 학습할 능력이 끝난다(§2). 병합은 stable phase checkpoint에서 annealed model에 가까운 품질을 얻는 방법으로 시험되었다.

**작동 원리**(§4.6). θ* 주변의 quadratic loss를 `δ = θ − θ*`라 놓으면 `L(θ) ≈ L(θ*) + ½ δᵀHδ`이다. `Σ_i Σ_{j≠i} δ_iᵀHδ_j < (k − 1) Σ_i δ_iᵀHδ_i`(Eq. 15)일 때 k개 checkpoint 평균의 loss는 그 checkpoint들의 평균 loss보다 낮다. Deviation이 서로 다른 방향을 가리킬수록 이 조건은 더 충족하기 쉽다.

**계산 예(도출).** 1차원에서 `H = 2`, `θ* = 0`이라고 하자.

1. Checkpoint `θ1 = 0.3`, `θ2 = −0.1`의 loss는 각각 0.09와 0.01이고 평균은 0.05이다. 평균 `θ = 0.1`의 loss는 0.01이다. Cross term은 `2 · (0.3 · 2 · −0.1) = −0.12`이고, 이는 `(2 − 1)(0.18 + 0.02) = 0.20`보다 작다.
2. Checkpoint `θ1 = 0.3`, `θ2 = 0.1`이 같은 쪽에 있으면 평균 loss는 0.05이다. 평균 `θ = 0.2`의 loss는 0.04로, 더 좋은 checkpoint의 0.01보다 높다.

Positive-definite H에서는 unequal deviation이면 Cauchy–Schwarz에 따라 이 inequality가 성립하므로, 이 model에서 merge는 항상 *평균* checkpoint보다 낫다. 실제로 중요한 비교인 *latest* checkpoint보다 나은지는 deviation의 방향에 달려 있다.

**근거.**

- Stable phase에서 병합은 Seed-MoE-1.3B/13B의 HumanEval을 31.1에서 36.6으로, Seed-MoE-10B/100B는 54.3에서 61.6으로 올렸다(Fig. 1). 단일 연구의 internal model·data 결과다.
- *Annealing 대체.* 1.4T token 지점의 Seed-MoE-1.3B/13B fork 두 개를 250B token 더 학습했다. 하나는 constant LR, 하나는 annealed였고, merged constant-LR model은 초반에는 둘 다보다 “significantly outperformed”했으며 후반에는 “comparable to the annealed models”였다(Fig. 3, 값은 graph로만 제시).
- 설정: 204B token에서는 WMA가 가장 좋았고, 이후에는 method 간 차이가 줄었다. 이후 실험은 “for its simplicity and stability” 때문에 SMA를 사용했다. 0.7B/7B, 1.3B/13B, 10B/100B model의 interval은 각각 약 4B, 8B, 80B token이었다. 학습 종료 시 `N = 3`은 `N = 15`보다 “nearly 1 point lower”였고, 저자들은 `N = 10`을 골랐다(Figs. 4–5).
- *대조 결과.* Hägele et al.에서는 constant-LR run의 stochastic weight averaging이 raw checkpoint보다 loss를 낮췄지만, perplexity로 측정한 33M–360M parameter에서 명시적 cooldown의 loss에는 도달하지 못했다([[cooldown-scaling-beyond-fixed-durations]] §4.1). 두 연구는 scale, architecture, metric이 다르다. 열린 질문이다.
- *계속 학습을 위한 merged initialization.* 약 1T stable token 뒤 Seed-MoE-0.7B/7B의 PMA-init은 continued training 초기 phase에서 “marginally lower loss”를 냈고, 학습이 진행되면서 comparable loss로 수렴했다. 최종적으로는 overall parity였다(§4.4, Fig. 6; 값은 graph로만 제시).

**조건과 한계.** 저자들은 PMA에 대해 shifted D1에서 forgetting을 측정하지 않았고, model architecture와 data는 공개되지 않았다. 저자들에 따르면 learning rate를 매우 낮게 anneal한 checkpoint를 병합하면 gain이 더 작다(§4.6).

**함의.** Continued pretraining에서는 아직 decay되지 않은 stable-phase checkpoint들을 병합한 model을 시작점 후보로 둘 수 있다. 이는 §2의 pre-decay-checkpoint 관행과 맞는다. 다만 distribution shift 아래 forgetting에 미치는 효과는 시험되지 않았다. Continued pretraining 뒤 specialist를 병합하는 내용은 ch-30c에서 다룬다.

## §7 의사결정 규칙과 평가 gate

아래 규칙은 §1–§6을 이 강좌에서 종합한 것이다(해석). 각 단계는 근거로 삼는 evidence를 명시한다. 인용하지 않은 threshold는 어느 source에서 가져온 값도 아니다.

1. **새 data의 규모를 정한다.** D1이 100B token 이상이고 language, domain, modality 가운데 하나가 D0와 다르면 replay를 포함한 continued-pretraining run을 계획한다(§3). D1이 수백만 token 규모라면 먼저 synthetic expansion을 계획하고(§5), EntiGraph가 했듯 10% general replay를 유지한다.
2. **시작 checkpoint를 고른다.** Pre-decay checkpoint(DeepSeekMath), near-end checkpoint(Kimi K2.5), 또는 존재한다면 stable-phase checkpoint를 우선한다. 그런 checkpoint가 없다면 원래 `η_max`에 가까운 peak까지 re-warm한 뒤 cosine re-decay한다(§2). 사용한 checkpoint를 기록한다. 이것이 모든 delta의 `θ_before`다.
3. **Replay source와 비율을 고른다.** Release가 약속하는 capability를 replay한다. 제한된 token budget에서 1%, 5%, 10%, 25% 같은 fraction을 짧게 sweep한다. Ibrahim et al.은 “relative differences between them appear very early during training”이라고 보고했기 때문이다(§2의 rules of thumb). Domain별로 차례대로 update하지 말고 한 stage에서 domain들을 mix한다(§3).
4. **Release 또는 SFT로 hand-off하기 전에 gate를 통과시킨다.** `θ_after`를 `θ_before`와 비교하고, 비용이 허용되면 no-replay run과도 비교한다.

| Gate 항목 | 측정값 | 통과 조건(강좌 종합) | 방법의 source |
|---|---|---|---|
| Adaptation | D1 held-out loss, target benchmark | `θ_before`보다 낮고, 이 harness에서 측정한 run-to-run noise보다 더 나쁘지 않음 | [[continual-pretraining-rewarm-replay]] Table 2 |
| Domain별 forgetting | 각 주요 D0 source의 `Δforget(d)`, macro-average, decontaminated | 모든 domain의 increase를 보고하고, harness의 seed-to-seed spread보다 큰 increase는 trade-off로 명시 | [[paloma]] §3–§4 |
| General downstream | MMLU-style knowledge, reasoning, code, multilingual score | `Δtask`가 측정한 evaluation noise 이내이거나 trade-off를 문서화함 | [[grpo]] Table 4 |
| Reference correctness | 정확한 initialization checkpoint의 score | 같은 table에 보고 | [[qwen-2-5-math]], [[qwen-2-5-coder]] |
| Modality 또는 language shift | Run 중 text-only curve | 회복되지 않는 dip이 없음 | [[kimi-k2-5]] App. B.1 |

5. **보고 panel을 다음 단계까지 가져간다.** Continued pretraining과 post-training에서 생기는 forgetting은 release model에서 합쳐지므로, SFT(ch-30a) 뒤와 RL 뒤에도 같은 panel을 다시 실행한다.

## Negative samples와 negative feedback

Continued pretraining은 모든 token에 next-token loss를 적용한다. 따라서 sample의 likelihood를 아래로 밀어 내리는 항(gradient로서의 negative)은 없고, 이 단계에서 likelihood displacement 위험도 없다. Negative는 positive target으로 학습했을 때 성능을 낮추는 data, 즉 **negative marginal value**라는 첫 번째 의미로만 나타난다. 이 장에서 측정된 경우는 두 가지다. 1.3M-token corpus에 raw CPT를 적용하면 base model보다 낮은 38.15%를 기록했다(base 39.49%, [[synthetic-continued-pretraining]] Fig. 2). Qwen2.5-Coder mixture ablation에서는 100% code data가 가장 낮은 MMLU와 math score를 냈다([[qwen-2-5-coder]] Table 3). 세 번째는 측정되지 않은 위험이다. Finance용 synthesized instruction-response pair는 69.8%만 정확하다고 판정됐는데([[instruction-pretraining]] Table 6), filtering 없이 positive target으로 학습했다. 학습 전 filtering 또는 verification이 control이며, Qwen2.5-Coder의 synthetic code에서는 “실행 가능한 code만 유지했다(only executable code was retained)”([[qwen-2-5-coder]] §3.1.1). Negative gradient는 ch-31a와 ch-43a에서 다룬다.

## 레시피

| Model(정확한 release) | 크기 | Stage | Setting | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Ibrahim et al. Pile → SlimPajama | 405M | mid-train | D0 / D1 tokens; batch; seq | 300B Pile / 300B SlimPajama; 1104 sequences; 2048 | arXiv:2403.08763v4 §5.1–§5.3 | verified 2026-09-15 | ablation 미보고 |
| same | 405M | mid-train | LR schedule | 1% iteration 동안 3e-4까지 re-warm, 3e-5까지 cosine; optimizer state reset | Table 14; §5.2; §6.3 | verified 2026-09-15 | §6.1.2 Fig. 4: re-warm vs constant, `η_max ∈ {1.5, 3, 6}e-4` |
| same | 405M | mid-train | replay fraction | 5% | §6.3 | verified 2026-09-15 | Table 2: 0.5–50% sweep, single seed |
| Ibrahim et al. Pile → German | 405M | mid-train | D1 tokens; replay | 200B; 25% | §6.2–§6.3 | verified 2026-09-15 | Table 2: 1–50% sweep, single seed |
| Ibrahim et al. Pile → SlimPajama | 10B | mid-train | peak → final LR(Table 14) | 1.2e-4 → 1.2e-5, warmup 1% | Table 14 | conflict(Table 14 vs §6.4 text; 논문 config 미공개) | ablation 미보고 |
| Ibrahim et al. Pile → SlimPajama | 10B | mid-train | peak LR(§6.4 text) | 두 크기 모두 “pre-training의 `ηmax`(3·10−4)까지 re-warm” | §6.4 | conflict(위 행 참조) | ablation 미보고 |
| Ibrahim et al. Pile → SlimPajama | 10B | mid-train | replay fraction | 5% | §6.4 | verified 2026-09-15 | 405M weak-shift sweep에서 선택(§6.4) |
| DeepSeekMath-Base 7B | 7B | mid-train | init; tokens; mixture | Coder-v1.5 pre-decay; 500B; math corpus 56%, AlgebraicStack 4%, arXiv 10%, code 20%, NL 10% | arXiv:2402.03300v3 §2.3 ([[grpo-recipe]]) | verified 2026-09-14 | 이 split의 ablation 미보고; §5.1 Tables 6–7은 1.3B |
| DeepSeekMath-Base 7B | 7B | mid-train | peak LR; batch | 4.2e-4; 10M tokens | §2.3 ([[grpo-recipe]]) | verified 2026-09-14 | ablation 미보고 |
| Qwen2.5-Coder(0.5B–32B) | all | mid-train | init; tokens; seq; mixture | Qwen2.5; 5.2T file-level at 8,192; code 70%, text 20%, math 10% | arXiv:2409.12186v3 §3.1.2, §3.2.1 | verified 2026-09-15 | Table 3: 100:0:0 vs 85:15:5 vs 70:20:10 on 7B |
| Qwen2.5-Coder(0.5B–32B) | all | long-context | tokens; seq; RoPE base | ~300B; 8,192 → 32,768; 10,000 → 1,000,000(as printed) | §3.2.2 | verified 2026-09-15 | ablation 미보고 |
| Qwen2.5-Coder | all | mid-train | LR, batch, schedule | 미보고(§3, §4, Table 1 확인) | — | not reported | — |
| Qwen2.5-Math-7B | 7B | mid-train | init; tokens; seq | Qwen2.5 base; >1T(Qwen Math Corpus v2); 4K | arXiv:2409.12122v1 §2 | verified 2026-09-15 | Table 2: v2 vs v1 comparison across models |
| Qwen2.5-Math-7B | 7B | mid-train | mixture; general share; LR | 미보고(§2, §5.1 확인) | — | not reported | — |
| Instruct PT Llama3-8B(biomed / finance) | 8B | mid-train | steps; batch; seq; LR | 4K; 0.25M tokens; 4096; 1e-5 cosine, 1000 warmup | arXiv:2406.14491v2 Table 14 | verified 2026-09-15 | Table 4 data ablation; LR ablation 없음 |
| same | 8B | mid-train | total tokens | 1B | 4K × 0.25M | derived | — |
| EntiGraph CPT Llama 3 8B | 8B | mid-train | data; epochs; replay | 455M EntiGraph tokens; 2; replay rate 0.1(RedPajama batch loading probability; 1B-token pool) | arXiv:2409.07431v2 §4.1, App. C | verified 2026-09-15 | Fig. 2 token-count scaling; 이 run의 replay ablation 없음 |
| same | 8B | mid-train | seq; batch; LR | 2048; 16; peak 5e-6, warmup 5%, cosine | App. C | verified 2026-09-15 | ablation 미보고 |
| Seed-MoE-1.3B/13B(PMA) | active/total 1.3B/13B | merge | method; N; V | SMA; 10; ~8B tokens | arXiv:2505.12082v3 §4.2–§4.3 | verified 2026-09-15 | Figs. 4–5: WMA/SMA/EMA, `V ∈ {4, 8, 16, 32}B`, `N ∈ {3, 6, 10, 15}` |
| Kimi K2.5 | total 1.04T / active 32B | mid-train | init; tokens; seq | near-end K2 checkpoint; 15T vision-text; 4K | arXiv:2602.02276v2 §4.3, Table 3 | verified 2026-09-15 | Table 1: vision timing/ratio ablation(size 미보고) |
| Kimi K2.5 | same | mid-train | vision:text ratio; LR | 미보고(“constant ratio”, §2) | — | not reported | — |

**작은 범용 run의 시작점.** Optimizer state를 reset하고 200–300B token을 continued training하는 약 405M-parameter dense model에서는, 검증된 행이 다음 설정을 준다. Original peak(그 연구에서는 3e-4)까지 iteration의 1% 동안 re-warm하고, peak의 0.1배까지 cosine을 적용하며, 짧은 sweep 뒤 같은-language data refresh면 compute-equivalent replay 5%, new language면 25%를 쓴다([[continual-pretraining-rewarm-replay]] §6.3, Tables 13–14). 8B model과 약 1M-token corpus에서는 검증된 EntiGraph 설정이 peak LR 5e-6, warmup 5%, synthetic corpus 2 epoch, general replay 10%다([[synthetic-continued-pretraining]] App. C). 어느 설정도 다른 scale의 general benchmark에서 검증되지 않았다.

## 일반화 관점

**(a) Breadth를 늘리는 것.**
- General data를 mixture에 남긴다. Qwen2.5-Coder-7B의 70:20:10 mixture는 100% code의 MMLU 42.8에 비해 62.9, GSM8K 23.8에 비해 64.5를 얻었다([[qwen-2-5-coder]] Table 3). Code 20%, NL 10%를 포함한 DeepSeekMath의 500B-token run은 initialization 대비 MMLU +12.0, BBH +16.6이었다([[grpo]] Table 4).
- Re-warming 뒤 replay한다. Language-shift run에서 25% replay는 두 dataset 모두에서 from-scratch 500B-token training의 average loss와 일치했다([[continual-pretraining-rewarm-replay]] Table 2). Qwen2.5-Coder mixture ablation도 다른 setting에서 같은 방향, 즉 general data가 general score를 보호함을 가리킨다. 따라서 방향은 Replicated지만 효과 크기는 setting-specific이다.
- Small corpus에는 다양한 knowledge representation을 쓴다. EntiGraph의 combinatorial entity-relation text는 paraphrase보다 나쁜 scaling과 달리 log-linear scaling을 보였다([[synthetic-continued-pretraining]] Fig. 2).

**(b) Narrowing 또는 forgetting을 일으키는 것.**
- Re-warming 뒤 replay 없이 계속한 경우: 405M에서 German 200B token 뒤 Pile loss +1.39([[continual-pretraining-rewarm-replay]] Table 2).
- Learning-rate increase 자체: Pile에서 `η_max = 3e-4`로 계속했을 때 Pile loss peak +0.1(§7.1).
- Sequential single-domain stage: 1.3B에서 code → math는 HumanEval을 25.0%에서 12.2%로 낮췄다([[grpo]] Tables 6–7).
- New modality를 늦고 높은 ratio로 도입: K2.5 ablation의 text-performance “dip-and-recover”와 더 낮은 final text knowledge([[kimi-k2-5]] Table 1, App. B.1).
- Small raw corpus 학습: closed-book QA가 base model보다 낮음([[synthetic-continued-pretraining]] Fig. 2).

**(c) 이 단계에서의 측정법.**
- Decontamination, fixed vocabulary 또는 bits per byte, document를 한 번에 하나씩 score하는 macro-averaged per-domain perplexity([[paloma]] §3 G1–G5).
- 정확한 initialization checkpoint에 대한 downstream delta. Reference 선택은 DeepSeekMath의 MMLU gain을 +5.8에서 +12.0으로 바꿨다([[grpo]] Table 4).
- 알려진 error: single seed 및 405M에서 1.19-point spread([[continual-pretraining-rewarm-replay]] §6.3.2), reference row 없는 intermediate-checkpoint initialization([[qwen-2-5-math]] Table 2), 다른 specialist와만 비교하는 specialist report([[qwen-2-5-coder]] Tables 13–14).

## Common mistakes와 탐지 방법

| 실수 | 관측 가능한 증상 | Check |
|---|---|---|
| Base model의 최종 낮은 LR에서 계속함 | 같은 token의 re-warmed run보다 D1 loss가 덜 떨어지고 target benchmark 변화도 작음 | 같은 token으로 짧은 re-warmed run과 비교(§2, Fig. 4 pattern) |
| Shift된 D1에서 replay 없이 re-warm | Old-domain loss가 일찍 오르고 계속 높음 | 학습 중 domain별 `Δforget(d)` plot; 1–5% replay run 추가 |
| Initialization checkpoint 대신 released model에 대해 delta 보고 | Reference를 바꾸면 gain/loss 크기가 달라짐 | `θ_before` 자체를 평가; DeepSeekMath의 +5.8 vs +12.0 MMLU가 예 |
| Token-weighted validation loss 하나만 사용 | Aggregate loss는 평평하지만 작은 domain은 degrade | Per-domain perplexity를 macro-average(§1 worked example) |
| Run 비교에서 replay를 extra token으로 계산 | Replay run이 더 많은 compute를 써서 좋아 보임 | Compute-equivalent replay 사용: `(1 − x)·B` new tokens |
| Domain 하나씩 차례로 update | 각 transition에서 early domain degrade | 한 stage에서 domain을 mix(§3; [[grpo]] Tables 6–7) |
| Small raw corpus에 직접 training | Closed-book QA가 flat 또는 base 이하 | Equal step에서 raw CPT와 synthetic expansion 비교(§5.2) |
| 검증하지 않은 synthetic target 신뢰 | Closed-book answer에 hallucinated fact | Generated pair를 sample/grade하고 학습 전 filter(§5.1 Table 6) |
| Check 없이 merged constant-LR checkpoint를 annealed checkpoint와 동등시 | Annealed branch보다 낮은 loss/score | 짧은 cooldown branch를 만들고 같은 suite에서 비교(§6 contrasting result) |

## 이해 점검

1. Language-shift run에서 25% replay는 German loss를 0.05 올리고 Pile loss를 1.23 내렸다. No-replay run보다 German token을 50B 적게 봤는데도 이 run의 average loss가 500B token으로 from scratch 학습한 model과 같을 수 있는 이유를 설명하라.
2. Pile 자체에서 re-warming하면 Pile loss가 올랐다. 이 결과는 SlimPajama에서 계속 학습할 때 보인 early forgetting의 원인에 무엇을 뜻하는가? 두 원인을 분리하는 run을 어떻게 설계하겠는가?
3. DeepSeekMath의 MMLU gain은 reference checkpoint에 따라 +5.8 또는 +12.0이다. 차이 중 어느 부분이 base model 자신의 learning-rate decay에서 오며, math data가 general ability를 개선하는지 판단할 때 왜 중요한지 설명하라.
4. Qwen2.5-Coder의 70:20:10 mixture는 100% code보다 code column은 낮지만 MMLU와 GSM8K는 높다. Math/knowledge gain을 설명할 수 있는 mechanism 두 개를 제시하고, 이를 시험하려면 어떤 추가 table이 필요한지 말하라.
5. 같은 document에서 나온 1.3M raw token의 continued pretraining은 closed-book QA를 높이지 못했지만 455M EntiGraph token은 성공한 이유는 무엇인가? 각 fact가 나타나는 distinct form 수와 연결해 설명하라.
6. One-dimensional merging example에서 merge는 두 경우 모두 mean checkpoint보다 좋지만 두 번째에서는 best checkpoint보다 좋지 않다. Real training trajectory의 어떤 성질이 PMA가 latest checkpoint보다 좋아지는지를 결정하는지 설명하라.
7. New-language continued-pretraining report가 stable aggregate validation loss와 unchanged MMLU를 보인다. 이 report가 여전히 숨길 수 있는 forgetting 두 가지와 각각을 드러낼 measurement를 기술하라.

## 연결

- **Depends on:** ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL(annealing, stage placement, decay-phase data).
- **Previous in outline:** ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL.
- **Next:** ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression(sequence length에서도 같은 adaptation-versus-regression trade-off).
- **Related:** ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization(cosine/WSD schedule); ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale; ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing; ch-29a — Long-Document Synthesis for Continued Pretraining and Long-Context SFT; ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; ch-30c — Weight Averaging and Model Merging for Generalist Models; ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing.

## 출처

- [[continual-pretraining-rewarm-replay]] — re-warming/re-decaying result, replay sweep(Table 2), downstream/10B result(Tables 3–5), infinite schedule, limitation.
- [[paloma]] — perplexity/macro-average 정의, guideline G1–G5, web-only baseline의 per-domain degradation, perplexity–downstream correlation.
- [[grpo]] — pre-decay initialization 대비 DeepSeekMath-Base general/code retention(Table 4), 1.3B code-then-math forgetting ablation(Tables 6–7).
- [[grpo-recipe]] — DeepSeekMath-Base 7B initialization, 500B-token mixture, learning rate, batch.
- [[qwen-2-5-coder]] — Qwen2.5 initialization, stage token count, 70:20:10 mixture ablation, initialization row 없는 general-benchmark report.
- [[qwen-2-5-math]] — math corpus size, Qwen2.5/intermediate Qwen2 checkpoint initialization, Qwen2.5 row 없는 base-model table.
- [[instruction-pretraining]] — synthesizer method, domain continued-pretraining result, synthesized-pair accuracy, hyperparameter.
- [[synthetic-continued-pretraining]] — EntiGraph method, raw vs rephrase vs EntiGraph scaling, RAG comparison, replay/training setting.
- [[merging-in-pretraining]] — PMA 정의, stable-phase/annealing-substitute result, interval/count ablation, quadratic analysis, PMA-init.
- [[cooldown-scaling-beyond-fixed-durations]] — SWA가 cooldown loss에 도달하지 못함, pre-cooldown checkpoint에서 계속할 근거.
- [[kimi-k2-5]] — near-end K2 checkpoint에서의 continuation, vision injection timing/ratio ablation, text dip-and-recover.
- [[llama-3]] — 85% 초과 code로 1T token continued pretraining한 Llama 3 code expert.
- [[rephrasing-the-web]] — EntiGraph의 Rephrase baseline으로 적용한 rephrasing approach.
