<!-- chapter: ch-01
     track: foundations
     kind: content
     title: LLM 학습용 옵티마이저: AdamW, 업데이트 크기, 기존 능력 유지
     deps: [ch-00]
     sources: [[adam]], [[gradient-clipping]], [[pytorch-adamw-clip-amp]], [[heavy-tailed-class-imbalance-adam]], [[why-weight-decay-modern-dl]], [[small-scale-proxies-instabilities]], [[gpt-3-few-shot]], [[rlhf-instructgpt]], [[rlhf-instructgpt-optimizer-settings]], [[llama-2-recipe]], [[llama-3]], [[llama-3-recipe]], [[olmo-2]], [[olmo-2-optimizer-settings]], [[deepseek-v3-recipe]], [[kimi-k2]], [[kimi-k2-recipe]], [[glm-4-5-recipe]], [[qwen-2.5-recipe]], [[lora-learns-less-forgets-less]], [[lora-without-regret]], [[rls-razor]], [[online-merging-optimizer]], [[fantastic-pretraining-optimizers]], [[muon-scalable-moonlight]], [[openrlhf-ppo-recipe]]
     figures: figures/beta2-memory.html, figures/clip-unscale-order.html
     revised: 2026-09 (generality revision)
-->

# 01장 — LLM 학습용 옵티마이저: AdamW, 업데이트 크기, 기존 능력 유지

> **핵심 통찰.** 일반적인 경우 Adam은 그래디언트의 크기와 무관하게 한 스텝에서 각 좌표를 대략 학습률 이하만큼 이동시킨다([[adam]] §2.1). 따라서 학습률 스케줄과 스텝 수가 한 학습 단계가 모델을 얼마나 멀리 이동시키는지를 정한다. 사전학습에서는 이 좌표별 정규화가 경사하강법으로는 도달하지 못하는 속도로 희귀 토큰의 손실을 낮추게 하며([[heavy-tailed-class-imbalance-adam]] Fig. 1), weight decay는 정규화 항이 아니라 유효 학습률과 bf16 안정성을 통해 작동한다([[why-weight-decay-modern-dl]] §3). 미세조정에서는 동일한 업데이트 예산이 망각을 좌우한다. 코드 지시문으로 Llama-2-7B를 전체 미세조정하면 3개 벤치마크의 유지 평균이 1 epoch 뒤 0.595에서 16 epoch 뒤 0.414로 낮아졌다([[lora-learns-less-forgets-less]] Table S6). 소규모에서 사전학습 손실을 낮추는 대안 옵티마이저는 각 옵티마이저를 조정한 뒤 1.2B 파라미터에서 약 1.1× 속도 향상과 다운스트림 이득 없음만 보였다([[fantastic-pretraining-optimizers]] §4.1, Table 5).
>
> **지침.** 옵티마이저 sweep 예산 없이 dense Transformer를 사전학습할 때는 β₁ = 0.9, β₂ = 0.95, ε = 1e-8, PyTorch 형식의 weight decay 0.1, 토큰 임베딩에는 decay 없음, global-norm clipping 1.0을 가진 AdamW를 사용하라. 이 조합이 공개된 OLMo 2 7B 구성이고([[olmo-2-optimizer-settings]]), OLMo 2 ablation은 ε = 1e-5보다 ε = 1e-8에서 더 낮은 그래디언트 norm을, 임베딩 decay가 없을 때 더 적은 spike를 발견했기 때문이다(arXiv:2501.00656v3 §3.4.1-§3.4.2). 이 장의 어떤 보고서도 규모에서 β₂를 ablation하지 않으므로, 소규모 sweep이 가능하면 β₂는 조정할 값으로 취급하라([[fantastic-pretraining-optimizers]] Table 36). 일반 checkpoint를 미세조정할 때는 목표 이득 대 held-out 유지율의 측정 곡선에서 peak learning rate와 step 또는 epoch 수를 선택하라. epoch와 함께 망각이 늘었고([[lora-learns-less-forgets-less]] Table S6), toy setting에서는 weight-space 거리가 새 과제 입력에서의 KL divergence보다 망각을 더 나쁘게 예측했기 때문이다([[rls-razor]] Table 1). RLHF 단계가 held-out 과제를 후퇴시키면 더 큰 KL 계수만 쓰지 말고 사전학습 분포 또는 SFT delta에 대한 anchor를 추가하라. InstructGPT에서 기본값의 100배인 KL 계수는 사전학습 그래디언트 mix가 제거한 회귀를 제거하지 못했다([[rlhf-instructgpt-optimizer-settings]] App. E.6). fp16 loss scaling을 사용할 때는 clipping 전에 그래디언트를 unscale하라. PyTorch 문서에 따르면 scale된 그래디언트를 clipping하면 threshold가 무효가 되며([[pytorch-adamw-clip-amp]]), bf16에서 loss scaling을 쓰지 않는다면 이 순서는 적용되지 않는다. 대안 옵티마이저가 제안되면, 학습 종료 시 breadth suite에서 조정된 AdamW baseline과 비교하고 두 옵티마이저로 SFT를 시험하라. 그렇지 않으면 AdamW를 유지하라.

## 일반 목적 모델에서 이 장이 중요한 이유

옵티마이저는 그래디언트를 파라미터 변화로 변환한다. 일반 목적 모델에서 이 변환은 파이프라인의 세 지점(pre-training → mid-training → SFT → preference optimization → RL → evaluation)에서 중요하다.

1. **사전학습 커버리지.** 다음 토큰 예측은 어휘에 대한 분류 문제이며(Llama 3의 경우 128,000개 토큰, [[llama-3-recipe]]), 토큰 빈도는 heavy-tailed다([[heavy-tailed-class-imbalance-adam]] §1.1). 고정된 토큰 예산 안에서 옵티마이저가 어떤 토큰을 학습하는지가 base model이 긴 꼬리(희귀 단어, 식별자, 저자원 언어)를 얼마나 포괄하는지를 결정한다(§2).
2. **긴 실행의 안정성.** 손실 spike와 발산은 연산을 낭비하고 더 나쁜 checkpoint를 남길 수 있다. weight decay, ε, β₂, gradient clipping은 이를 제어하기 위해 공개 보고서가 변경하는 옵티마이저 설정이다(§3-§5).
3. **post-training 중 유지.** 모든 post-training 단계는 이미 넓은 능력을 가진 checkpoint에서 시작한다. 학습률, step 수, 업데이트 형식(full, low-rank, merged), 최적화 중 사용되는 anchor가 그 능력 중 얼마나 살아남는지를 정한다(§6). ch-30a가 망각을 자세히 측정하고, 이 장은 옵티마이저 측면을 다룬다.

측정 가능한 문제는 희귀 토큰 클래스에서 더 느린 손실 감소, 손실 및 gradient-norm spike, 미세조정 후 held-out benchmark 하락이다. 아래에서 쓰는 용어는 다음과 같다. *파라미터(parameter)* θ는 학습 가능한 하나의 수(좌표) 또는 모든 수의 벡터이고, *그래디언트(gradient)* g는 θ에 대한 학습 손실의 미분이며, *학습률(learning rate)* α(η 또는 lr로도 표기)는 각 업데이트를 스케일하고, *step*은 한 번의 옵티마이저 업데이트이며, *global norm*은 모든 파라미터에 걸친 그래디언트의 L2 norm이다.

## §1 AdamW: 기호별 업데이트 규칙

**정의.** Adam은 파라미터마다 두 개의 지수이동평균, 즉 그래디언트 하나와 제곱 그래디언트 하나를 유지하고 전자를 후자의 제곱근으로 나누는 확률적 옵티마이저다([[adam]] Algorithm 1). AdamW는 *분리된(decoupled)* weight decay를 적용한 Adam이다. decay 항은 파라미터를 직접 줄이고 moment 추정치를 통과하지 않는다([[adam]] AdamW Algorithm 2).

**해결하는 문제.** 좌표들 사이에서 그래디언트 크기는 자릿수 단위로 다르다. 하나의 학습률만 사용하면 경사하강법은 그래디언트가 큰 좌표는 멀리, 작은 좌표는 조금 이동시킨다. Adam의 좌표별 나눗셈은 모든 좌표의 step 크기를 α에 가깝게 만든다.

**메커니즘.**
1. minibatch에서 확률적 그래디언트 g_t를 계산한다.
2. 1차 moment m_t(g의 running mean)와 2차 moment v_t(g²의 running mean)를 갱신한다.
3. 둘 다 0으로 초기화해 생긴 편향을 제거하기 위해 각각을 (1 − β^t)로 나눈다.
4. 각 좌표를 α·m̂_t/(√v̂_t + ε)만큼 이동한다.
5. AdamW에서는 decay 대상 각 파라미터도 λ에 의존하는 계수만큼 수축한다.

**수식** ([[adam]] Algorithm 1; AdamW Algorithm 2, line 12):

```
m_t = β₁·m_{t−1} + (1 − β₁)·g_t
v_t = β₂·v_{t−1} + (1 − β₂)·g_t²
m̂_t = m_t / (1 − β₁^t)          v̂_t = v_t / (1 − β₂^t)
θ_t = θ_{t−1} − η_t·( α·m̂_t / (√v̂_t + ε) + λ·θ_{t−1} )      # AdamW, paper form
```

θ_t: t step 뒤의 파라미터. g_t: t step의 그래디언트. m_t, v_t: 1차 및 2차 moment 추정치(t = 0에서 0). β₁, β₂ ∈ [0, 1): 두 평균의 감쇠율. m̂_t, v̂_t: 편향 보정된 추정치. α: 기본 step 크기. ε: 분모를 양수로 유지하는 작은 상수. η_t: 스케줄 multiplier. λ: weight-decay 계수. 모든 연산은 원소별이다.

Adam step에는 근사 상한이 있다. ε = 0이면 각 좌표에서 |Δθ_t| ≲ α이고, m̂는 c배, √v̂도 c배가 되므로 모든 그래디언트에 상수 c를 곱해도 step은 불변이다([[adam]] §2.1). 논문은 (1 − β₁) > √(1 − β₂)인 경우에 더 느슨한 상한 α·(1 − β₁)/√(1 − β₂)를 제시하며, 이 상한은 좌표의 그래디언트가 모든 이전 step에서 0이었던 가장 심한 sparsity에서만 도달한다고 말한다. 상한 α는 근사다. β₂ = 0.999에서는 조건이 성립하고(0.1 > 0.032), §4의 계산 예 2는 한 step spike 뒤 1.81α, 지속적인 그래디언트 상승 중 5.30α의 step을 보인다.

**대부분의 실행이 사용하는 구현.** PyTorch v2.5.0 `AdamW`는 Adam step 전에 파라미터에 (1 − lr·weight_decay)를 곱한다([[pytorch-adamw-clip-amp]], `torch/optim/adamw.py` L368-L429).

```python
# Perform stepweight decay
param.mul_(1 - lr * weight_decay)
# Decay the first and second moment running average coefficient
exp_avg.lerp_(grad, 1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
bias_correction1 = 1 - beta1**step
bias_correction2 = 1 - beta2**step
step_size = lr / bias_correction1
bias_correction2_sqrt = bias_correction2**0.5
denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
param.addcdiv_(exp_avg, denom, value=-step_size)
```

여기서 `lr`에는 이미 스케줄이 포함되어 있으므로 step별 decay는 lr·λ다. 논문 형식에서 decay는 η_t·λ이며 기본 α에 따라 스케일되지 않는다. [[small-scale-proxies-instabilities]]는 논문 형식을 *independent* weight decay라고 부른다(§3.2.2). 반면 `weight_decay`를 사용하는 PyTorch `Adam`은 L2 regularization을 구현한다. 즉 `grad = grad.add(param, alpha=weight_decay)` (`torch/optim/adam.py` L366-L367)이므로, decay 항이 손실 그래디언트와 함께 √v̂로 나뉜다.

**계산 예 1(AdamW 두 step, 한 좌표).** θ₀ = 1.0, α = 1e-3, β₁ = 0.9, β₂ = 0.95, ε = 1e-8, λ = 0.1, PyTorch 형식, 그래디언트 g₁ = 0.2와 g₂ = −0.1을 두자.

- Step 1: decay θ ← 1.0 × (1 − 1e-3 × 0.1) = 0.9999. m₁ = 0.1 × 0.2 = 0.02; v₁ = 0.05 × 0.04 = 0.002. m̂₁ = 0.02/0.1 = 0.2; v̂₁ = 0.002/0.05 = 0.04; √v̂₁ = 0.2. 비율은 1.0이므로 θ₁ = 0.9999 − 0.001 = 0.9989다. 어떤 g₁에 대해서도 편향 보정은 m̂₁ = g₁ 및 v̂₁ = g₁²를 주므로 첫 step의 크기는 항상 α다.
- Step 2: decay θ ← 0.9989 × 0.9999 = 0.998800. m₂ = 0.9 × 0.02 + 0.1 × (−0.1) = 0.008; v₂ = 0.95 × 0.002 + 0.05 × 0.01 = 0.0024. m̂₂ = 0.008/0.19 = 0.0421; v̂₂ = 0.0024/0.0975 = 0.0246; √v̂₂ = 0.1569. 비율은 0.268이므로 θ₂ = 0.998800 − 0.000268 = 0.998532다. 그래디언트의 부호 변화가 step을 α의 27%로 줄였다.

**계산 예 2(L2 대 분리된 decay).** 두 좌표 모두 θ = 1, α = 1e-3이라고 하자. 좌표 A의 전형적인 |g| = 10이므로 √v̂ ≈ 10이고, 좌표 B의 |g| = 0.01이므로 √v̂ ≈ 0.01이다. Adam 내부의 L2 계수 1e-4를 사용하면(L2 항이 g에 비해 작다고 하자) step의 decay 부분은 α·λθ/√v̂다. 즉 A에서는 1e-3 × 1e-4 / 10 = 1e-8, B에서는 1e-3 × 1e-4 / 0.01 = 1e-5로 1,000배 차이 난다. PyTorch AdamW와 λ = 0.1에서는 두 좌표 모두 step당 α·λ·θ = 1e-4만큼 decay한다. 이는 [[adam]]이 말한 비동치성(AdamW §2, Proposition 2)이다. L2 아래에서 “전형적인 그래디언트 크기 s가 큰 가중치는 더 작은 상대량만큼 정규화된다.”

**근거.** AdamW 논문이 보고한 이득은 CIFAR-10 및 ImageNet32x32 이미지 분류에서 Adam에 대해 15%의 상대 test-error 개선이다(AdamW §1, Figs. 2-3). 언어 모델 실험은 없다. 한 epoch GPT-2-124M 학습에서는 손실에 추가한 L2 penalty가 weight decay의 학습 손실 이득을 재현했고(Fig. 21), momentum SGD도 비슷한 이득을 보였다(Fig. 22)([[why-weight-decay-modern-dl]]) (Result, 단일 연구).

**조건과 한계.** decoupling 자체가 중요하다는 근거는 이미지 분류에서 왔다. LLM 사전학습에서 측정된 효과는 decoupled 대 L2가 아니라 일반적인 weight decay의 효과다(§3).

**일반 목적 모델에 대한 함의.** 보고서에서 가져온 λ는 학습 코드가 같은 decay 형식을 구현할 때만 이전될 수 있다. 동일한 λ = 0.1도 논문 형식과 PyTorch 형식에서 step별 decay가 다르다(§3에서 계산).

## §2 Adam이 경사하강법보다 언어 모델을 빠르게 학습하는 이유: heavy-tailed 토큰 빈도

**정의.** *Heavy-tailed 클래스 불균형*은 클래스 빈도가 거듭제곱 법칙을 따르므로 희귀 클래스들이 함께 표본의 큰 몫을 차지하는 현상이다. 다음 토큰 예측에서 어휘의 각 항목은 하나의 클래스다([[heavy-tailed-class-imbalance-adam]] §1.1).

**측정으로 진술한 문제.** WikiText-103으로 학습한 GPT2-Small을 빈도 순으로 정렬한 클래스 그룹별 학습 손실로 평가했다. 각 그룹은 표본의 약 10%를 담는다. “SGD는 저빈도 클래스에서 거의 또는 전혀 진전하지 못하는 반면 Adam은 모든 그룹에서 진전한다”(Fig. 1). 대부분의 표본은 빈도가 낮은 토큰에서 오므로, SGD에서는 평균 손실도 천천히 낮아진다(Abstract).

**메커니즘**(가중 이차 모형, §3.1).
1. 각 클래스 k에 빈도 π_k로 가중한 손실 f_k(w_k) = ½‖w_k‖²를 부여하며, Σπ_k = 1이다.
2. w_k에 대한 그래디언트는 π_k·w_k이므로, 경사하강법은 step마다 (1 − απ_k) 계수로 w_k를 수축한다.
3. 가장 빈번한 클래스를 불안정하게 만들지 않으려면 step 크기는 약 1/π₁을 넘을 수 없으므로, 희귀 클래스는 최댓값의 약 π_k/π₁인 step을 받는다.
4. Adam의 단순화 형태인 sign descent는 그래디언트를 그 크기로 나누므로, 각 클래스는 π_k와 무관하게 step마다 α만큼 이동한다.

**수식**(§3.1): 경사하강법은 w_k^(t) = (1 − απ_k)^t · w_k^(0)를 주며, sign descent는 w_k^(t) = w_k^(t−1) − α·sign(f_k′(w_k^(t−1)))를 준다. w_k^(t): t step 뒤 클래스 k의 파라미터. π_k: 클래스 k의 빈도. α: step 크기.

**계산 예.** π₁ = 0.5, 희귀 클래스 π_k = 0.001로 두고 §3.1이 명시한 step 크기 한계 α = 1/π₁ = 2를 사용하자. 빈번한 클래스는 한 step 만에 0이 된다. (1 − 2 × 0.5) = 0. 희귀 클래스는 100 step 뒤 초기 오차의 (1 − 2 × 0.001)^100 = 0.998^100 = 0.819를 유지하고, 1,000 step 뒤에는 0.998^1000 = 0.135를 유지한다. step 0.01인 sign descent는 두 클래스 모두를 w = 1에서 100 step 뒤 0으로 이동시킨다.

**근거.** 빈도 격차는 각 5개 표본으로 된 약 10,000개 추가 클래스를 붙인 MNIST의 CNN(Fig. 2), π_k ∝ 1/k인 ImageNet 부분집합의 ResNet18(Fig. 3), 무작위 입력의 softmax 선형 모델(Fig. 4)에서 재현되었다. 이 현상은 full-batch 경사하강법에서도 나타나므로 minibatch noise는 필요하지 않다(§2.3). Sign descent는 Adam의 행동을 재현하고(Fig. 5), 희귀 클래스 손실의 가중치를 높이면 SGD가 개선된다(Appendix F)([[heavy-tailed-class-imbalance-adam]]). 상태: Result(여러 architecture를 쓴 단일 연구).

**조건과 한계.** 모든 측정은 학습 손실이고 held-out 능력은 평가하지 않았다. 가장 큰 언어 모델은 GPT2-Small이다. 저자들은 “클래스 불균형이 Adam이 SGD보다 뛰어난 유일한 이유라고 주장하지 않는다”고 명시한다(§1.1).

**일반 목적 모델에 대한 함의(해석).** 사전학습 말뭉치의 긴 꼬리(전문 어휘, 코드 식별자, 저자원 언어 텍스트)는 희귀 토큰으로 이루어진다. 좌표별로 정규화하는 옵티마이저는 경사하강법이 주로 빈번한 클래스를 맞추는 것과 같은 토큰 예산 안에서 이 클래스들을 맞춘다. 이것이 downstream coverage를 바꾸는지는 측정되지 않았다. ch-12a(지식 습득)와 ch-13a(다국어 커버리지)가 데이터 측면을 다룬다. Adam의 비용은 파라미터마다 state tensor 두 개를 추가한다는 점이며, training-memory course가 optimizer-state memory를 다룬다.

## §3 Weight decay: one-pass 사전학습에서의 유효 학습률과 안정성, multi-epoch 단계에서만의 과적합 제어

**정의.** Weight decay는 매 step에서 decay 대상 파라미터에 1보다 약간 작은 계수를 곱한다. 실무에는 세 가지 형식이 등장한다.

| 형식 | step별 효과 | 등장 위치 |
|---|---|---|
| 손실 안의 L2 | g ← g + λ_L2·θ, 그 뒤 √v̂로 나눔 | PyTorch `Adam(weight_decay=…)` ([[pytorch-adamw-clip-amp]]) |
| 독립형(논문 AdamW) | θ ← θ − η_t·λ·θ | [[adam]] Algorithm 2 line 12; [[small-scale-proxies-instabilities]] §3.2.2 |
| LR 결합형(PyTorch AdamW) | θ ← θ·(1 − lr_t·λ) | PyTorch `AdamW`; OLMo 2: “모든 파라미터에 매 step 1 − (0.1 · lr)를 곱한다”([[olmo-2-optimizer-settings]], §3.4.2); Llama 3 scaling-law 실행: “각 step의 weight decay는 해당 step의 learning rate의 0.1배로 설정한다”([[llama-3-recipe]], §3.2.1) |

**해결하는 문제.** 고전 학습 이론에서 weight decay는 과적합을 줄이기 위해 모델 용량을 제한한다. LLM 사전학습은 데이터 전체를 거의 한 번 통과하며, [[why-weight-decay-modern-dl]]은 GPT-2-124M의 일반화 gap이 weight decay 없이도 0에 가깝다고 보고한다(Fig. 19). 이 영역에서 weight decay의 측정 가능한 효과는 최종 학습 손실과 발산이다.

**메커니즘**([[why-weight-decay-modern-dl]] §3, 저자들의 해석).
1. sign과 비슷한 업데이트에서는 decay가 파라미터 norm ‖w_t‖₂를 더 작게 유지한다.
2. 그러면 방향 w/‖w‖₂는 decay가 없을 때보다 큰 *유효 학습률(effective learning rate)* η_t/‖w_t‖₂로 변한다.
3. 학습 초반에는 더 큰 유효 LR이 손실을 높인다(gradient-noise 항이 step 크기와 함께 커진다). 학습 후반에는 LR이 decay한 상태이므로, weight decay 실행은 초기 조건 항(initial-condition term)에서 더 빠르게 진전했기에 더 낮은 손실에 도달한다.
4. Weight decay는 bfloat16 가중치에서 학습 후반 발산도 방지한다.

**수식.** 유효 LR은 η_eff ∝ η_t/‖w_t‖₂다. PyTorch AdamW의 step별 decay는 d_t = lr_t·λ이고, lr이 일정할 때 decay 시간척도는 1/(lr·λ) step이다. η_t, lr_t: t step의 learning rate. ‖w_t‖₂: 가중치의 L2 norm. λ: decay 계수.

**계산 예.** OLMo 2 7B는 결합형에서 peak LR 3e-4와 λ = 0.1을 사용한다([[olmo-2-optimizer-settings]]). peak LR에서 step별 decay는 3e-4 × 0.1 = 3e-5이고, 시간척도는 1/3e-5 = 33,333 step이다. 그 LR에서 10,000 step 뒤 decay만으로 가중치에 남는 비율은 (1 − 3e-5)^10,000 = 0.741이다. OLMo 2 7B의 한 step은 1,024 × 4,096 = 4,194,304 토큰을 담으므로, 4T-token stage 1은 953,674 step이다(Table 3에서 도출). peak LR에서는 약 29개 시간척도이며 cosine decay 뒤에는 더 적다. λ = 1e-4인 독립형([[small-scale-proxies-instabilities]] §2.1)에서는 스케줄 multiplier가 1일 때 모든 LR에서 step당 decay가 1e-4다. λ = 0.1인 결합형에서는 LR 3e-4일 때 3e-5, LR 0.1일 때 1e-2다. 결합형에서 LR을 올리면 decay도 올라가므로 두 형식의 LR 민감도는 다르다.

**근거.**
- OpenWebText의 GPT-2-124M, 50,000 iteration, batch 256, context 256, AdamW LR 6e-4, β₂ = 0.95, clipping 1.0에서 λ = 0.1과 0.3은 λ = 0보다 최종 학습 손실이 낮다(Fig. 6). λ = 0.1 또는 0.3 실행의 유효 LR에 맞춘 LR 스케줄을 가진 decay 없는 실행은 float32에서 전체 손실 곡선을 재현하지만, 동일한 실행은 bfloat16에서 발산한다(Fig. 7). context 1,024와 LR 1e-3에서 decay 없이 실행하면 세 bfloat16 seed 모두 학습 후반에 발산하고 회복하지 않으나 λ > 0은 이를 방지한다(Fig. 8)([[why-weight-decay-modern-dl]]). 상태: Result(단일 연구, 124M).
- 독립 decay(λ = 1e-4)는 C4에서 2.4M-1.2B 모델에 대해 결합형 기본값(λ = 0.1)보다 낮은 LR 민감도를 주었다. Weight decay는 시험한 더 큰 모델에서 output-logit 발산도 완화했다([[small-scale-proxies-instabilities]] §3.1.2, §3.2.2, Fig. 3, Fig. 6). 상태: Result(단일 연구).
- OLMo 2 실행에서 토큰 임베딩을 decay하면 embedding norm은 줄고 gradient norm은 증가했으며, embedding decay가 없을 때 0.092인 gradient spike score가 0.16이었다([[olmo-2-optimizer-settings]], §3.4.2, Fig. 10). 공개된 OLMo 2 7B config는 `decay_embeddings: false`, `decay_norm_and_bias: true`를 설정하므로 normalization weight를 decay한다. norm 또는 bias decay의 ablation은 보고되지 않았다. GLM-4.5는 word embedding, bias, RMSNorm weight를 제외한 모든 파라미터에 Muon을 적용하며, 그 제외된 항목에 사용한 optimizer 또는 decay는 출력하지 않는다([[glm-4-5-recipe]], §2.4).
- Weight decay 없는 Muon은 “가중치와 layer output의 RMS 모두가 큰 scale로 계속 증가하여 bf16의 고정밀 범위를 넘게” 했다. 100B 토큰으로 학습한 800M 모델은 decay가 있을 때 없는 경우보다 낮은 validation loss에 도달했다([[muon-scalable-moonlight]] §2.2, Fig. 2). 상태: 안정성 역할에 대해 Replicated(독립된 두 그룹, [[why-weight-decay-modern-dl]] 및 [[muon-scalable-moonlight]]).

**multi-epoch 단계와의 대조.** 데이터가 반복되면 과적합은 측정 가능하다. InstructGPT SFT 모델은 “1 epoch 후 validation loss에서 과적합”했지만, 16 epoch는 RM score와 인간 선호 평점을 개선했다. 6B reward model은 몇 epoch 안에 과적합했고 “validation loss의 명백한 악화”가 있었다([[rlhf-instructgpt-optimizer-settings]] §3.5, App. C.1-C.2). Llama-2-7B의 코드 instruction tuning에서 5e-5 또는 1e-4의 weight decay는 decay 없는 full fine-tuning과 거의 같은 정도로 학습하고 망각했으며, 목표 성능은 epoch 8과 16에서 악화됐다([[lora-learns-less-forgets-less]] §4.5, Fig. 4). 분리형 decay는 시작 checkpoint θ₀가 아니라 0을 향해 가중치를 당긴다(수식 θ ← θ(1 − lr·λ)에서). 그러므로 fine-tuned model을 base에 anchor하지 않는다. Llama 2-Chat SFT 값 lr = 2e-5와 λ = 0.1([[llama-2-recipe]] §3.1)에서, 그 LR로 가정한 10,000 step은 데이터와 무관하게 모든 decay 대상 가중치를 (1 − 2e-6)^10,000 = 0.980배로 수축한다.

**조건과 한계.** 유효 LR 설명은 124M 파라미터와 50,000 iteration에서 시험됐다([[why-weight-decay-modern-dl]] §4는 대규모 실험 부족을 언급한다). decay에서 제외할 파라미터 group은 임베딩에 대해서만 ablation 근거가 있다.

**일반 목적 모델에 대한 함의.** 사전학습에서 weight decay는 LR 스케줄 및 precision과 상호작용하는 안정성과 최종 손실 설정이므로 함께 조정한다. 미세조정에서 인용된 근거는 weight decay를 retention 제어로 뒷받침하지 않는다. §6이 측정된 retention 효과가 있는 제어를 열거한다.

## §4 β₂와 ε: 공개 보고서가 선택한 값과 측정된 결과

**정의.** β₂는 2차 moment 평균의 기억 길이를 정하고, ε은 분모 √v̂ 아래에 바닥값을 둔다. 비율 β인 지수이동평균의 *평균 시간척도(averaging timescale)*는 1/(1 − β) step이고, *반감기(half-life)*는 ln 0.5 / ln β step이다.

**계산 예 1(시간척도).** β = 0.9: 시간척도 10 step, 반감기 6.6 step. β = 0.95: 20과 13.5. β = 0.98: 50과 34.3. β = 0.999: 1,000과 692.8.

**계산 예 2(그래디언트 변화 뒤 한 좌표).** g = 1인 정상상태에서 시작하므로 m = v = 1이고 업데이트는 α라고 하자. step 0에서 그래디언트는 한 step 동안 10이 되었다가 다시 1로 돌아오며, β₁ = 0.9이다.

- β₂ = 0.999: v = 0.999 + 0.001 × 100 = 1.099, √v = 1.048; m = 0.9 + 1.0 = 1.9다. spike step은 1.9/1.048 = 1.81α다. 업데이트는 step +28까지 α보다 큰 채로 유지되고, 이후 α보다 약간 작아진다(+50과 +100에서 0.96α). 그동안 v는 수백 step 동안 높은 채로 남아 있다.
- β₂ = 0.95: v = 0.95 + 0.05 × 100 = 5.95, √v = 2.44다. spike step은 1.9/2.44 = 0.78α다. 업데이트는 α보다 작은 상태로 유지되며(+14에서 최솟값 0.65α), step +108에서 α의 1% 이내로 돌아온다.
- spike 대신 점진 상승(그래디언트가 20 step에 걸쳐 1에서 10으로 커지고 그 상태로 유지됨)이면, peak update는 β₂ = 0.999에서 5.30α이고 β₂ = 0.95에서 1.15α다.

그러므로 긴 β₂ 기억은 그래디언트가 *커지는 동안* α보다 큰 step을 만든다. 한 step spike 뒤에는 높아진 v가 후속 step을 α보다 *작게* 만든다. 대화형 그림 [figures/beta2-memory.html](figures/beta2-memory.html)은 독자가 spike 크기를 바꾸고, spike와 점진 상승을 전환하고, β₂ = 0.95, 0.98, 0.999의 업데이트 크기를 읽을 수 있게 한다.

**계산 예 3(ε).** 정상상태에서 좌표별 업데이트는 약 α·r/(r + ε)이며, r은 그 좌표의 gradient RMS다. r = 1e-6일 때 ε = 1e-8은 0.990α, ε = 1e-5는 0.091α를 준다. r = 1e-8, ε = 1e-8이면 업데이트는 0.5α다.

**근거: 보고서 속 값.**
- GPT-3(모든 버전, arXiv 2020-05): Adam β₁ = 0.9, β₂ = 0.95, ε = 1e-8, global-norm clipping 1.0, weight decay 0.1([[gpt-3-few-shot]] App. B). 따라서 β₂ = 0.95는 아래의 더 늦은 보고서들보다 앞선 2020년부터 공개 기록에 있다.
- InstructGPT(arXiv 2022-03): 모든 모델이 β₁ = 0.9, β₂ = 0.95인 Adam, fp32 master copy를 가진 fp16 weight를 사용한다([[rlhf-instructgpt-optimizer-settings]] App. C).
- Llama 2(2023-07): 사전학습과 PPO에서 AdamW β₂ = 0.95, eps = 1e-5, weight decay 0.1, clipping 1.0([[llama-2-recipe]] §2.2, §3.2.3).
- DeepSeek-V3: AdamW β₂ = 0.95, weight decay 0.1, clipping 1.0([[deepseek-v3-recipe]] §4.2). Llama 3.1 405B: AdamW이나 betas와 weight decay는 출력하지 않았다([[llama-3-recipe]] §3.4.1).
- OLMo 2 7B: betas (0.9, 0.95), ε 1e-8([[olmo-2-optimizer-settings]], config L44-L53). 이 보고서들 가운데 β₂ ablation을 제공하는 것은 없다.

**근거: 측정값.**
- OLMo 2는 ε을 1e-5에서 1e-8로 낮췄다. “더 낮은 값은 학습 초기에 더 큰 업데이트를 허용하고 … 그래디언트 norm이 훨씬 더 빨리 안정되며 영구적으로 더 낮게 유지된다”고 보고한다(§3.4.1, Fig. 9, 약 8,000 step; ablation 모델 크기는 명시되지 않음)([[olmo-2-optimizer-settings]]). 상태: Result(단일 연구).
- 첫 MLP layer의 gradient RMS는 모델 크기와 LR에 따라 낮아지며, 시험한 가장 큰 규모와 LR에서는 기본 ε = 1e-8 근처다. LR 0.3의 4.8B 모델에서 ε = 1e-15는 손실을 개선했고 ε = 1e-6은 발산했다([[small-scale-proxies-instabilities]] §3.4, Figs. 11-12; 값당 한 실행). 상태: Result(단일 연구). OLMo 2와 이 논문은 방향(더 작은 ε이 도움이 됨)에는 동의하지만 서로 다른 설정을 측정한다.
- 130M 파라미터, 1× Chinchilla 데이터, C4-EN 손실에서 조정된 AdamW의 한 번에 한 hyperparameter씩 바꾸는 sweep에서 β₂ = 0.98은 3.529, 0.95는 3.535, 0.9는 3.545를 얻었다. β₁ = 0.95는 3.539, β₁ = 0.98은 3.882를 얻었으며, ε을 1e-25에서 1e-10으로 바꾼 손실 변화는 최대 0.002였다([[fantastic-pretraining-optimizers]] Table 36). 이 논문은 validation loss가 3e-3보다 많이 개선될 때 hyperparameter 변경을 받아들인다(§3.2). β₂ = 0.95와 0.98의 0.006 차이는 그 임계값의 두 배이고 seed는 보고되지 않았다. 상태: Result(단일 연구, 소규모).

**조건과 한계.** frontier 규모에서 어느 β₂가 가장 좋은지는 이 장의 출처에서 Open question이다. 소규모 sweep은 130M 파라미터에서 수행됐다. [[small-scale-proxies-instabilities]]는 β₂를 related work(footnote 1, §4)에서만 언급한다.

**일반 목적 모델에 대한 함의.** β₂ = 0.95는 문서화된 시작값이지 규모에서 측정된 최적값은 아니다. 큰 실행에서는 ε과 나란히 기록한 layer별 gradient RMS가 ε이 업데이트를 제한하는지를 보인다. 단일 spike가 아니라 그래디언트 크기의 지속적 상승이 긴 β₂ 기억이 큰 step을 만드는 경우다.

## §5 Global norm에 의한 gradient clipping

**정의.** Global-norm clipping은 L2 norm이 임계값 c를 넘을 때 전체 그래디언트 벡터를 재스케일하고, 그 방향은 유지한다([[gradient-clipping]] Algorithm 1).

**해결하는 문제.** 학습 중 gradient norm은 spike를 일으킨다. OLMo-0424에는 “손실에서, 그리고 더 자주 gradient norm에서” spike가 있었다([[olmo-2-optimizer-settings]] §3). 그런 그래디언트 뒤 Adam의 첫 step은 α를 넘을 수 있으며(§4, 계산 예 2), SGD의 step은 그래디언트에 비례해 커진다.

**수식**([[gradient-clipping]] §3.2):

```
ĝ ← ∂E/∂θ
if ‖ĝ‖ ≥ threshold:  ĝ ← (threshold / ‖ĝ‖) · ĝ
```

ĝ: 모든 파라미터에 대한 그래디언트. ‖ĝ‖: 그 L2 norm, 즉 모든 tensor의 모든 항목 제곱합의 제곱근. threshold: c. PyTorch는 분기 없이 같은 계수를 계산한다([[pytorch-adamw-clip-amp]], `clip_grad.py` L94-L109).

```python
total_norm = torch.linalg.vector_norm(
    torch.stack([norm.to(first_device) for norm in norms]), norm_type
)
clip_coef = max_norm / (total_norm + 1e-6)
clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
```

함수는 clipping 전의 `total_norm`(L124)을 반환하며, 이것이 기록할 값이다.

**계산 예 1(global 대 tensor별).** 두 tensor의 gradient norm이 3과 4이면 global norm은 √(9 + 16) = 5다. c = 1에서 계수는 0.2이고 norm은 0.6과 0.8이 되어 3:4 비율이 보존된다. 각 tensor를 따로 1로 clipping하면 norm은 1과 1이 되어 비율이 1:1로 바뀌고, 따라서 업데이트 방향도 바뀐다. Pascanu 등은 RNN 실험에서 자신들의 norm 기반 변형과 Mikolov의 원소별 변형이 “비슷하게 작동한다”고 보고한다(§3.2). Transformer에 대해서는 이 장의 출처가 비교를 제시하지 않는다.

**clipping이 loss unscaling에 놓이는 위치.** fp16 학습은 작은 그래디언트가 표현 가능하게 남도록 loss에 loss scale S를 곱한다(ch-02). PyTorch 문서는 unscaling 없이 clipping하면 threshold가 “무효(invalid)”가 된다고 말하며, 다음 순서를 제시한다. `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step(optimizer)` → `scaler.update()`. 축적된 모든 그래디언트가 존재한 뒤 step마다 `unscale_`을 한 번 호출해야 한다([[pytorch-adamw-clip-amp]], `amp_examples.rst` L67-L117). bf16 학습은 GradScaler를 사용하지 않으므로 S = 1이고 순서는 중요하지 않다.

**계산 예 2(unscale 전 clip).** GradScaler의 초기 scale은 S = 2^16 = 65,536이다(`grad_scaler.py` L122). 실제 gradient norm이 0.8이고 c = 1이라고 하자.

1. 문서화된 순서: unscaling 뒤 norm은 0.8 < 1이므로 계수는 1이고 옵티마이저는 norm 0.8을 받는다.
2. unscale 전 clip: clipping은 65,536 × 0.8 = 52,428.8을 보고, 계수는 1/52,428.8 = 1.91e-5가 된다. `scaler.step`이 unscale한 뒤 옵티마이저는 norm 0.8 × 1.91e-5 = 1.53e-5를 받는다. Clipping은 매 step마다 발동하며 no-op이 아니다.
3. AdamW에 대한 효과(단순화: 모든 좌표의 크기가 같고 파라미터 N = 1e9개): 좌표별 그래디언트는 1.53e-5/√1e9 = 4.83e-10으로 ε = 1e-8보다 작다. 따라서 업데이트는 약 4.83e-10/(4.83e-10 + 1e-8) = 0.046α다. Adam의 scale invariance(§1)는 상수 인자를 상쇄하겠지만 ε은 scale되지 않으므로, 이 실행은 의도한 step 크기의 약 5%로 학습한다.

계산기 [figures/clip-unscale-order.html](figures/clip-unscale-order.html)는 독자가 S, gradient norm, c, N, ε을 바꾸고 두 순서에서 옵티마이저가 받는 값을 비교할 수 있게 한다.

**임계값 선택과 보고서 속 값.** Pascanu 등은 “충분히 많은 업데이트에서 평균 norm의 통계”로 임계값을 정하라고 제안했고, RNN 실험에서는 학습이 그 값에 “매우 민감하지 않았다”고 발견했다([[gradient-clipping]] §3.2). GPT-3, Llama 2, DeepSeek-V3, OLMo 2 7B는 1.0에서 clipping한다(§4 인용). 130M AdamW sweep에서 clipping 없음과 임계값 2.0은 모두 3.534를 주었고, 1.0에서는 3.529였다([[fantastic-pretraining-optimizers]] Table 36). Kimi K2와 GLM-4.5는 clipping 값을 출력하지 않는다([[kimi-k2-recipe]]; [[glm-4-5-recipe]]).

**모니터링.** OLMo 2는 OLMo-0424에 “손실에서, 그리고 더 자주 gradient norm에서” spike가 있었다고, “gradient norm의 더 극적인 spike가 학습 손실 spike에 자주 선행했다”고 보고한다. 또 *spike score*를 “최근 1,000개 값의 rolling average에서 적어도 7 표준편차 떨어진 시계열 값의 백분율”로 정의한다. 새 initialization은 gradient-norm spike score를 0.40에서 0.03으로 바꿨다([[olmo-2-optimizer-settings]], §3, §3.2). 이 장의 어떤 출처도 gradient-norm spike와 loss spike 사이의 고정 lead time을 주지 않는다.

**RL reward outlier와의 상호작용.** REINFORCE식 그래디언트는 (1/N)·Σ_i A_i·∇log π(y_i | x)이며, A_i는 표본 y_i의 advantage, π는 policy, N은 표본 수다(ch-37). 그래디언트 norm은 advantage의 크기와 함께 커지고, outlier reward는 다른 표본 advantage의 *부호*도 바꾼다. 보상이 [1, 0, 1, 0, 20]인 한 prompt의 표본 다섯 개를 계산 예로 든다.

1. 평균 baseline: advantage는 [−3.4, −4.4, −3.4, −4.4, 15.6]이다. 정답 두 표본(reward 1)은 음의 advantage를 받으며, 이는 ch-43a의 의미에서 음의 그래디언트다(그 likelihood가 아래로 밀린다).
2. 그룹 정규화(population 표준편차 7.81로 나눔, sample 표준편차는 8.74; GRPO, ch-40): [−0.435, −0.563, −0.435, −0.563, 1.997]. 크기는 줄지만 부호는 바뀌지 않는다.
3. Global-norm clipping은 합산 그래디언트를 재스케일하며, 이 부호도 바꾸지 않는다.
4. 10에서 reward clipping(OpenRLHF 기본 범위는 (−10, 10), [[openrlhf-ppo-recipe]] `train_ppo_ray.py` L437): reward [1, 0, 1, 0, 10], advantage [−1.4, −2.4, −1.4, −2.4, 7.6]이며 부호는 여전히 뒤집혀 있다. 2에서 clipping하면 [0.2, −0.8, 0.2, −0.8, 1.2]가 되어 정답 표본이 다시 양수가 된다.

Gradient clipping은 step 크기에 한계를 두며, reward 및 advantage 설계는 step 방향을 정한다. OpenRLHF의 기본 gradient clip은 1.0(L474), Llama 2-Chat PPO는 1.0([[llama-2-recipe]] §3.2.3), RL's Razor LLM 실험은 max grad norm 1을 사용하며 weight decay와 max gradient norm을 “수동으로 ablation”했을 때 “결과에 유의한 효과가 없었다”고 명시한다([[rls-razor]] App. B.1 Table 2).

**일반 목적 모델에 대한 함의.** clipping 전 global norm과 clip 계수는 매 step 기록할 두 값이다. 대부분 step에서 clip 계수가 1 미만이면 threshold, loss-scaling 순서, reward scale을 점검하라는 신호다. 이 각각은 모든 업데이트가 모델을 얼마나 멀리 이동시키는지를 바꾸기 때문이다.

## §6 미세조정에서의 업데이트 크기와 기존 능력 유지

**정의.** *업데이트 크기(update size)*는 학습 단계가 시작 checkpoint θ₀에서 모델을 얼마나 멀리 이동시키는지다. 이는 weight space(예: ‖θ_T − θ₀‖) 또는 output space(예: 새 과제 τ의 입력 x에서 base policy π₀와 fine-tuned policy π 사이 KL divergence인 E_{x∼τ}[KL(π₀ ‖ π)], [[rls-razor]] §1)로 잴 수 있다. *망각(forgetting)*은 그 단계가 목표로 하지 않은 held-out 과제에서의 하락이다. *정렬 세금(alignment tax)*은 alignment 단계가 일으킨 망각이다([[rlhf-instructgpt-optimizer-settings]] §1). *Anchor*는 모델을 기준 분포(예: 사전학습 텍스트) 또는 기준 가중치(예: SFT checkpoint) 쪽으로 당기는 학습 항이다.

**문제.** 미세조정 단계는 목표를 개선하면서 held-out 점수를 낮출 수 있다. Magicoder-Evol-Instruct-110K에서 Llama-2-7B를 전체 미세조정하면 HumanEval pass@1은 epoch 1 뒤 0.302에서 epoch 8 뒤 0.497로 올랐다가 epoch 16 뒤 0.416으로 떨어졌다. 반면 망각 평균(HellaSwag, ARC-Challenge, WinoGrande의 평균이며 높을수록 망각이 적음)은 0.595에서 0.446, 0.414로 낮아졌다([[lora-learns-less-forgets-less]] Tables S5-S6).

**메커니즘.**
1. 일반적인 경우 Adam은 각 좌표의 step을 대략 α_t로 제한하므로(§1; β₂ = 0.999일 때 §4에서 5.30α까지의 예외가 있음), T step 뒤 한 좌표의 총 이동은 약 Σ_{t=1..T} α_t 이하이다.
2. 그래디언트 부호가 번갈아 나타나면 실제 이동은 더 작다(§1, 계산 예 1).
3. Weight decay는 가중치를 θ₀가 아니라 0 쪽으로 수축한다(§3).
4. 옵티마이저는 업데이트 형식(LoRA의 low rank)을 제한하거나 anchor 방향(사전학습 그래디언트, SFT delta)을 섞을 수 있다.

**수식(상한).** |θ_{T,i} − θ_{0,i}| ≲ Σ_{t=1..T} α_t. θ_{T,i}: T step 뒤 좌표 i. α_t: t step의 학습률. T step에 걸쳐 α_max에서 0으로 가는 cosine schedule에서 Σα_t ≈ α_max·T/2다.

**계산 예.** 일정한 α = 1e-5에서는 1,000 step이 좌표별 상한 0.01을, 16,000 step이 0.16을 준다. 2e-5에서 0으로 가는 10,000 step cosine schedule은 약 2e-5 × 10,000 / 2 = 0.1을 준다. 같은 schedule 모양에서 epoch를 두 배로 늘리면 T와 상한도 두 배가 된다.

**근거.**
- *Step과 epoch.* 망각은 “학습 기간에 따라 악화되는 경향”이 있고, instruction fine-tuning은 continued pretraining보다 더 잊으며, 코드는 수학보다 더 잊는다([[lora-learns-less-forgets-less]] §4.2). Llama 3는 short-context 데이터의 DPO가 long-context SFT model의 long-context 성능을 해치지 않았다고 관찰하고, 이는 “우리 DPO recipe가 SFT보다 optimizer step이 적기 때문”일 수 있다고 추정했다([[llama-3]] §4.3.4; 저자들의 해석).
- *학습률.* 1.3B와 6B InstructGPT PPO에서 2.55e-6부터 2.55e-5까지 scan한 결과, pretraining mix가 없으면 “학습률이 8.05e-6보다 큰 모든 실행이 발산”했다. mix가 있는 PPO는 “학습률 변화에 덜 민감”했다([[rlhf-instructgpt-optimizer-settings]] App. E.9).
- *업데이트 형식.* 코드 instruction tuning에서 LoRA rank 256은 epoch 4에 HumanEval 0.498, 망각 평균 0.631에 도달했고, full fine-tuning은 epoch 8에 0.497, 0.446에 도달했다. Rank 16은 더 적게 학습했으며(epoch 4에 0.358) 0.652를 유지했다([[lora-learns-less-forgets-less]] Tables S5-S6). 최선의 LoRA learning rate는 full fine-tuning의 약 10×였다(App. B). [[lora-without-regret]](practitioner-evidence)는 용량 제한이 없을 때 모든 layer의 LoRA가 Tulu3 및 OpenThoughts3 subset에서 full fine-tuning 손실과 일치하고, policy-gradient RL에서도 “rank가 1만큼 낮아도” full fine-tuning과 일치하며, 최적 LR이 full fine-tuning의 약 10×(fit multiplier 9.8)라고 보고한다. 망각은 측정하지 않는다. 10× LR 비율의 상태: Replicated(두 출처). 일치한 목표 정확도에서 LoRA가 망각을 줄인다는 상태: Result(단일 연구, 코드).
- *Weight distance 대 output distance.* MNIST 기반 toy task에서 망각을 forward KL로 이차 적합하면 R² 0.96, reverse KL로는 0.93, weight change L1으로는 0.34, Fisher-weighted L2와 spectral-norm weight change로는 0.58이었다. Qwen 2.5 3B-Instruct에서 KL 적합은 R² 0.71이었다([[rls-razor]] §4, §6 Table 1). 저자들은 겉보기로 희소한 RL weight update가 bfloat16 가중치에서 “작은 파라미터 업데이트가 … 표현 임계값을 넘지 못할 수” 있기 때문에 생기며 float32에서는 사라짐도 발견했다(§6). ch-30a는 KL-망각 연결이 “항상 성립하지는 않는다”는 연구를 기록한다. 상태: LLM에 대해서는 Open question.
- *최적화 내부의 anchor.* InstructGPT PPO-ptx는 γ = 27.8 및 RL episode보다 8× 많은 사전학습 예제를 사용하여 RL 목적에 γ·E_{x∼D_pretrain}[log π(x)]를 더한다. γ ≥ 20일 때 public NLP dataset의 회귀는 1.3B에서 회복됐지만, γ = 0에서 KL 계수를 “기본값의 100배”인 2.0으로 올려도 회복되지 않았다([[rlhf-instructgpt-optimizer-settings]] §3.5, App. C.4, App. E.6). PPO-ptx도 DROP, SQuADv2, 번역에서 GPT-3보다 뒤처졌다(§4.2).
- *Online Merging Optimizer.* 매 step에 업데이트를 SFT delta τ_r = θ_SFT − θ_base와 결합한다. θ^(t) = θ^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r). 여기서 Δθ는 Adam update, F_R은 각 항목을 확률 p로 남기는 무작위 sparsification, α는 merging weight다([[online-merging-optimizer]] Eq. 3-4). UltraFeedback의 off-policy DPO에서 OnDARE는 AdamW에 비해 benchmark 평균(Table 1 “Benchmark Avg.”)을 +0.5(Qwen1.5-1.8B, AdamW 41.8), +1.1(Qwen1.5-7B, 58.4), +1.3(LLaMa-3-8B, 58.0) 바꿨고, MT-Bench를 +0.24, +0.12, +0.19 바꿨다. 저자들은 MT-Bench 표준편차를 약 0.05로 추정한다(Table 1, App. B). KL-penalty baseline은 평균을 +0.4, +0.1, +1.0 바꿨다. 구성별 seed는 보고되지 않았다. Qwen2.5 Instruct는 hyperparameter를 출력하지 않고 LR 7e-7로 DPO 1 epoch에 이 옵티마이저를 사용했다([[qwen-2.5-recipe]] §4.2). 상태: Result(단일 그룹).
- *옵티마이저 불일치.* 1.2T 토큰의 Moonlight checkpoint를 tulu-3-sft-mixture로 2 epoch 미세조정했을 때, Muon 사전학습과 Muon SFT는 MMLU 55.7 및 GSM8K 68.0, Muon 사전학습과 AdamW SFT는 50.2 및 64.9를 기록했다. AdamW 사전학습 checkpoint는 AdamW SFT에서 52.0 및 64.6을 기록했다([[muon-scalable-moonlight]] Table 6). AdamW 사전학습 Qwen2.5-7B에서 Muon SFT는 GSM8K 85.8 대 Adam SFT 89.8, MMLU 70.8 대 71.4를 기록했으며, 저자들은 이를 동등한 수준으로 표현한다(Table 7, §3.5.2). Kimi K2는 Muon 사전학습 뒤 SFT와 RL에 Muon을 사용했다([[kimi-k2-recipe]] §3.1, §3.2.3). 상태: Result(단일 그룹).

**조건과 한계.** epoch 및 LoRA 근거는 하나의 base model(Llama-2-7B), AdamW가 아닌 decoupled LionW, 객관식 retention benchmark 3개를 사용한다([[lora-learns-less-forgets-less]] App. A). Online-merging 이득은 보고된 seed 없이 평균에서 0.5-1.3점이다. Σα_t 상한은 느슨하고 어느 방향이 이동하는지는 말해주지 않는다.

**일반 목적 모델에 대한 함의.** Peak LR × step은 측정으로 선택할 양이다. 연산으로 sweep이 가능하면, 각 후보 값은 목표 이득과 ch-00에서 고정한 suite의 held-out retention을 위해 평가한다. 목표 prompt에서 starting policy에 대한 KL은 held-out 데이터 없이도 쓸 수 있는 모니터다. 그러나 KL-망각 관계는 LLM에 대해 Open question이므로, 중단 결정은 held-out 평가에 둔다. 한 단계가 넓은 능력을 후퇴시킬 것으로 예상되면, weight decay보다 측정된 retention을 보이는 메커니즘(사전학습 mix, low-rank 업데이트, online merging, 더 적은 step)을 선호하라.

## §7 AdamW의 대안: downstream 폭으로 판단하기

**정의**([[fantastic-pretraining-optimizers]] §3; [[muon-scalable-moonlight]] §2).
- *Lion*: w_{t+1} = w_t − η·sign(β₂·m_t + (1 − β₂)·g_t); 1차 moment만 유지한다.
- *Muon*: M_t = μM_{t−1} + ∇L_t; O_t = Newton-Schulz(M_t)이며, M_t = UΣVᵀ일 때 UVᵀ를 근사한다. A × B 행렬에 대해서는 W_t = W_{t−1} − η_t(0.2·O_t·√max(A, B) + λW_{t−1})다. 0.2·√max(A, B) 인자는 AdamW의 전형적인 update RMS 0.2-0.4에 맞추므로 AdamW의 learning rate와 weight decay를 재사용할 수 있다. 임베딩, LM head, normalization weight는 여전히 AdamW가 업데이트한다.
- *행렬 기반 preconditioner*(Shampoo family, Soap, Kron)는 항목별 scalar 대신 행렬을 그래디언트에 곱한다.

**문제.** 새 옵티마이저를 제안한 논문들은 목표 손실까지의 토큰으로 측정하여 AdamW에 대해 1.4-2× 사전학습 속도 향상을 보고했으며, 흔히 고정되었거나 조정되지 않은 hyperparameter를 가진 baseline을 상대로 했다([[fantastic-pretraining-optimizers]] §1).

**근거.**
- 100M 모델에서 GPT-3 recipe의 learning rate만 조정해도 AdamW 자체가 최대 2× 속도 향상을 얻었다. Lion의 최적 weight decay는 약 0.6이며 AdamW는 약 0.1이다. 조정된 AdamW와 비교하면 최선의 추정 속도 향상은 1.4×였다. 행렬 기반 방법은 520M 파라미터 미만에서 약 1.3×, 1.2B 및 8× Chinchilla 데이터에서는 약 1.1×를 주었고, 후자의 경우 downstream 개선을 “더 이상” 주지 않았다. 1.2B와 193B 토큰에서 10개 과제 평균은 AdamW 67.15, NAdamW 66.70, Muon 66.98이었다. Scaling-law 적합은 7B와 1× Chinchilla에서 Muon이 AdamW보다 약간 더 높은 손실을 낼 것으로 예측한다. 옵티마이저 간 순위는 learning-rate decay 중 뒤집힐 수 있다([[fantastic-pretraining-optimizers]] Abstract, §4.1, Table 5, Fig. 1).
- Moonshot AI는 compute-optimal 학습에서 Muon이 AdamW 학습 FLOPs의 약 52%를 필요로 한다고 보고한다([[muon-scalable-moonlight]] §3.2, Fig. 1a). cooldown 전 1.2T 토큰에서, 같은 2.24B-activated MoE의 Muon 대 AdamW는 MMLU 60.4 대 60.2, BBH 43.2 대 45.3, HumanEval 37.2 대 29.3, MATH 19.8 대 16.1, CMMLU 58.8 대 58.2였다(Table 4). 이득은 코드와 수학에 집중되어 있고 BBH는 더 낮다.
- Production use: Kimi K2(총 1.04T, 활성 32B)는 15.5T 토큰으로 MuonClip을 사용해 사전학습됐다. MuonClip은 한 head의 최대 attention logit이 τ = 100을 넘으면 query 및 key weight를 재스케일한다. 중간 규모 vanilla Muon 실행은 1,000을 넘는 logit에 도달했다([[kimi-k2]] §2.1). GLM-4.5(총 355B, 활성 32B)는 update RMS 0.2의 Muon을 사용했다([[glm-4-5-recipe]] §2.4). 두 보고서 모두 AdamW 대응군을 학습하지 않았으므로, 어느 쪽도 옵티마이저가 일으킨 downstream 차이를 측정하지 않는다.

상태: frontier 규모에서 Muon 이득의 크기는 Open question이다. 두 통제된 연구는 서로 다르다([[muon-scalable-moonlight]]에서 약 2× compute efficiency, [[fantastic-pretraining-optimizers]]에서 옵티마이저별 조정 뒤 1.2B까지 1.1-1.4×). 또한 이들은 조정 프로토콜, 데이터-대-모델 비율, 규모가 다르다.

**조건과 한계.** 통제된 비교는 1.2B 파라미터에서 멈추며 downstream suite는 base-model 객관식 및 cloze 과제다. Moonlight의 Table 4는 중간 checkpoint에서 옵티마이저당 한 실행을 비교한다.

**일반 목적 모델에 대한 함의.** 옵티마이저 변경은 pipeline 변경이다. 이는 hyperparameter 최적값, downstream 점수의 과제별 profile, 이후 가장 잘 작동하는 fine-tuning 옵티마이저를 바꾼다(§6). 사전학습 손실만 보지 말고, 학습 종료 시 및 SFT 뒤에 전체 breadth suite에서 평가하라.

## 레시피

모든 행은 인쇄된 옵티마이저 관련 설정이다. 상태 날짜: 2026-09-14 행은 연결된 library card에서 검증했고, 2026-09-15 행은 이 장을 위해 해당 위치에서 읽었다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| GPT-3(모든 버전) | 보고서의 모든 크기 | pretrain-stable | Adam β₁, β₂, ε; clipping; weight decay | 0.9, 0.95, 1e-8; global norm 1.0; 0.1 | arXiv:2005.14165v4 App. B ([[gpt-3-few-shot]]) | verified 2026-09-15 | ablation 보고 없음 |
| InstructGPT SFT | 1.3B, 6B / 175B | SFT | optimizer; LR; batch; epoch; schedule | Adam (0.9, 0.95); 9.65e-6 / 5.03e-6; 32 / 8; 16 epoch; 10%까지 cosine, warmup 없음 | arXiv:2203.02155v1 App. C, C.1 ([[rlhf-instructgpt-optimizer-settings]]) | verified 2026-09-15 | 7개 LR(1.3B, 6B) 및 5개 LR(175B)을 기하 탐색하고 epoch도 기하 탐색; 최종 모델은 RM score로 선택(§3.5) |
| InstructGPT PPO / PPO-ptx | 1.3B, 6B | RL | LR scan; 발산 | 2.55e-6부터 2.55e-5; ptx 없는 PPO는 8.05e-6 초과에서 발산 | arXiv:2203.02155v1 App. E.9 | verified 2026-09-15 | Fig. 38 인간 평가; 최종 checkpoint는 Likert score로 선택; 단일 최종 LR은 출력하지 않음 |
| InstructGPT PPO-ptx | 1.3B-175B | RL | KL β; ptx γ; 사전학습 예제 | 0.02; 27.8; RL episode의 8× | arXiv:2203.02155v1 App. C.4 | verified 2026-09-15 | App. E.6: γ ≥ 20은 1.3B에서 회귀를 회복; γ = 0에서 β를 2.0까지 올려도 회복하지 못함 |
| Llama 2 | 7B-70B | pretrain-stable | AdamW β₁, β₂, eps; weight decay; clipping | 0.9, 0.95, 1e-5; 0.1; 1.0 | arXiv:2307.09288v2 §2.2 ([[llama-2-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| Llama 2-Chat | 전체 | SFT | LR; weight decay; batch; epoch | 2e-5 cosine; 0.1; 64; 2 | arXiv:2307.09288v2 §3.1 ([[llama-2-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| Llama 2-Chat | 전체 | RL | optimizer; LR | AdamW (0.9, 0.95), eps 1e-5, weight decay 0.1, clipping 1.0; constant 1e-6 | arXiv:2307.09288v2 §3.2.3 ([[llama-2-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| Llama 3 scaling-law 모델 | 40M-16B | pretrain-stable | weight decay | “해당 step learning rate의 0.1배” | arXiv:2407.21783v3 §3.2.1 ([[llama-3-recipe]]) | verified 2026-09-14; 문구는 2026-09-15에 재독 | 해당 없음 |
| Llama 3.1 405B | 405B | pretrain-stable | optimizer; betas, weight decay, clipping | AdamW; 출력하지 않음 | arXiv:2407.21783v3 §3.4.1 ([[llama-3-recipe]]) | verified 2026-09-14(optimizer); not reported(betas, decay, clipping; 본문은 card가 확인) | 해당 없음 |
| Llama 3.1(“가장 큰 모델”) | 405B | SFT; preference | LR; step | SFT 8.5K-9K step에 1e-5; DPO 1e-5 | arXiv:2407.21783v3 §4.1.3, §4.1.4 ([[llama-3-recipe]]) | verified 2026-09-14 | SFT 값은 “서로 다른 round와 data mix 전반에서 잘 작동”; 수치는 없음 |
| OLMo 2 7B | 7B | pretrain-stable | AdamW betas; ε; weight decay; decay group; clipping; peak LR | (0.9, 0.95); 1e-8; 0.1; `decay_norm_and_bias: true`, `decay_embeddings: false`; 1.0; 3.0e-4 | github.com/allenai/OLMo@600a8d0 configs/official-1124/OLMo2-7B-stage1.yaml L44-L53, L90 ([[olmo-2-optimizer-settings]]) | verified 2026-09-15 | ε: arXiv:2501.00656v3 §3.4.1 Fig. 9(1e-8 대 1e-5); 임베딩: §3.4.2 Fig. 10(spike score 0.092 대 0.16) |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | AdamW β₁, β₂; weight decay; clipping | 0.9, 0.95; 0.1; 1.0 | arXiv:2412.19437v2 §4.2 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | optimizer; weight decay; QK-Clip τ; gradient clipping | MuonClip; 0.1; 100; 출력하지 않음 | arXiv:2507.20534v2 §2.1, §2.5 ([[kimi-k2-recipe]]) | verified 2026-09-14; clipping not reported | App. D: 0.5B-activated / 3B-total MoE에서 τ = 30, 무시할 수 있는 손실 변화; Muon 대 AdamW는 재-ablation하지 않음 |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT; RL | optimizer | Muon | arXiv:2507.20534v2 §3.1, §3.2.3 ([[kimi-k2-recipe]]) | verified 2026-09-14 | Muon 사전학습 checkpoint를 Muon으로 미세조정한 Moonlight를 인용 |
| GLM-4.5 | 355B / 32B act. | pretrain | optimizer; weight decay | word embedding, bias, RMSNorm weight를 제외한 Muon(N = 5, μ = 0.95, update RMS 0.2); 0.1 | arXiv:2508.06471v1 §2.4 ([[glm-4-5-recipe]]) | verified 2026-09-14 | Muon이 “수렴을 가속하고 더 큰 batch size를 견딜 수 있게 한다”; 수치는 없음 |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | preference | optimizer; LR; epoch | Online Merging Optimizer; 7e-7; 1; 그 hyperparameter는 주어지지 않음 | arXiv:2412.15115v2 §4.2 ([[qwen-2.5-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| Moonlight | 2.24B act. / 총 15.29B | pretrain-stable | optimizer; update scale; weight decay; peak LR | 0.2·√max(A, B) scaling을 가진 Muon; 0.1; 4.2e-4 | arXiv:2502.16982v1 §2.2 Eq. 4, §3.3 ([[muon-scalable-moonlight]]) | verified 2026-09-15 | Table 4: 1.2T 토큰에서 Moonlight-A(AdamW)와 비교, 각각 한 실행 |
| Small-scale proxy 기본값(Wortsman et al.) | 2.4M-1.2B | pretrain-stable | AdamW β₁, β₂, ε; clipping; weight decay | 0.9, 0.95, 1e-8; global norm 1; independent 1e-4 | arXiv:2309.14322v2 §2.1 ([[small-scale-proxies-instabilities]]) | verified 2026-09-14 | Fig. 6(independent decay: 더 낮은 LR 민감도); ε = 1e-15는 LR 0.3의 4.8B 실행에서만(Fig. 12) |
| 조정된 AdamW baseline(Wen et al.) | 130M, 1× Chinchilla | pretrain-stable | β₁, β₂, ε, LR, clipping, weight decay, batch, warmup | 0.9, 0.98, 1e-20, 0.008, 1, 0.1, 4,096 토큰 시퀀스 128개, 2,000 step | arXiv:2509.02046v2 Table 36 ([[fantastic-pretraining-optimizers]]) | verified 2026-09-15 | Table 36에서 C4-EN loss를 한 번에 하나씩 sweep; seed 미보고 |
| Llama-2-7B code IFT(Biderman et al.) | 7B | SFT | optimizer; LoRA LR; weight decay; clipping; batch | decoupled LionW (0.9, 0.95); 2e-4(r = 16, 64), 1e-4(r = 256); 0; norm 1; 192 | arXiv:2405.09673v2 App. A ([[lora-learns-less-forgets-less]]) | verified 2026-09-15 | App. B Fig. S1: 최선 full fine-tuning LR 5e-5(코드), 최선 LoRA 5e-4, 2 epoch |
| RL's Razor LLM 실행 | Qwen 2.5 3B-Instruct | SFT; RL | AdamW; clipping; weight decay; LR sweep | max grad norm 1; 0; SFT {1e-5 … 9e-5}; GRPO {1e-5 … 5e-5}, KL 0 | arXiv:2509.04259v1 App. B.1 Table 2 ([[rls-razor]]) | verified 2026-09-15 | weight decay와 clipping을 “수동으로 ablation”, 유의한 효과 없음 |
| OpenRLHF 기본값(모델 없음) | 모든 경우 | RL | gradient clipping; reward clip | 1.0; (−10, 10) | OpenRLHF@64c1cc4 `openrlhf/cli/train_ppo_ray.py` L474, L437 ([[openrlhf-ppo-recipe]]) | verified 2026-09-14 | ablation 보고 없음(framework 기본값) |

**소규모 일반 목적 실행의 시작점.** OLMo 2 7B 구간의 dense decoder-only 사전학습 실행(7B 파라미터, 1,024 × 4,096 토큰 batch, 4T에서 절단한 5T-token cosine schedule)의 경우, 검증된 OLMo 2 7B 행은 betas (0.9, 0.95), ε 1e-8, 임베딩을 제외한 결합형 weight decay 0.1, global-norm clipping 1.0, peak LR 3.0e-4의 AdamW를 준다. 1× Chinchilla 데이터로 학습하는 약 130M 파라미터 모델의 경우 검증된 조정 행은 다르다(β₂ 0.98, LR 0.008, batch 128). 따라서 7B 행에서 값을 복사하지 말고 그 규모에서 재조정해야 한다. 미세조정에 대해서도 검증된 행은 크기 범위가 정해져 있다. Llama 2-Chat SFT는 LR 2e-5의 cosine decay, weight decay 0.1, batch 64, 2 epoch를 사용했고, PPO는 clipping 1.0과 constant LR 1e-6를 사용했다(7B-70B). Qwen2.5 DPO는 Online Merging Optimizer로 LR 7e-7에서 1 epoch를 사용했다(0.5B-72B). 검증된 어떤 행도 held-out retention 측정으로 고른 fine-tuning LR을 진술하지 않으므로, 예산은 §6에서 설명한 대로 sweep해야 한다.

## 일반화 관점

**(a) 폭을 늘리는 것.**
- 좌표별 정규화 업데이트는 경사하강법이 같은 step 수에서 높은 손실로 남기는 희귀 토큰 클래스를 학습한다([[heavy-tailed-class-imbalance-adam]] Fig. 1; 학습 손실이며 downstream 폭은 아님).
- Weight decay는 유효 LR을 통해 최종 사전학습 손실을 낮추고 늦은 bf16 발산을 막는다([[why-weight-decay-modern-dl]] Figs. 6-8). 더 낮은 C4 손실은 옵티마이저 benchmark에서 HellaSwag 정확도와 함께 움직였다([[fantastic-pretraining-optimizers]] §4.1).
- 1.2T 토큰에서 Muon은 같은 모델의 AdamW보다 코드와 수학 점수를 높였다(HumanEval 37.2 대 29.3, MATH 19.8 대 16.1). MMLU 변화는 더 작았고 BBH는 하락했다([[muon-scalable-moonlight]] Table 4).

**(b) 협소화 또는 망각을 일으키는 것.**
- 더 많은 full fine-tuning epoch: 망각 평균은 1에서 16 epoch까지 0.595 → 0.414([[lora-learns-less-forgets-less]] Table S6), full fine-tuning 뒤 prompt당 고유 HumanEval generation도 더 적다(§4.5).
- 높은 RL learning rate: ptx 없는 InstructGPT PPO는 8.05e-6 초과에서 발산했다([[rlhf-instructgpt-optimizer-settings]] App. E.9).
- 사전학습 anchor 없는 RLHF: SQuAD, DROP, HellaSwag, WMT French-English에서 회귀([[rlhf-instructgpt-optimizer-settings]] §1).
- 사전학습 옵티마이저와 다른 fine-tuning 옵티마이저: Muon 사전학습 Moonlight는 AdamW SFT 뒤 MMLU 50.2, Muon SFT 뒤 55.7([[muon-scalable-moonlight]] Table 6).
- 인용된 설정에서 retention에 도움이 되지 않는 메커니즘: fine-tuning의 weight decay 및 dropout([[lora-learns-less-forgets-less]] §4.5), InstructGPT에서 더 큰 KL 계수만 쓰는 것([[rlhf-instructgpt-optimizer-settings]] App. E.6).

**(c) 이 단계에서 측정하는 방법.**
- 사전학습: 토큰 빈도 bucket별 손실([[heavy-tailed-class-imbalance-adam]] Fig. 1의 grouping), base-model 형식의 최종 checkpoint downstream suite(ch-00), gradient-norm spike score([[olmo-2-optimizer-settings]] §3.2), ε과 비교하는 layer별 gradient RMS([[small-scale-proxies-instabilities]] §3.4).
- 옵티마이저 비교: LR decay 중 순위가 뒤집히므로 옵티마이저별 조정 후 학습 종료 시 비교([[fantastic-pretraining-optimizers]] Abstract).
- 미세조정: LR과 epoch에 걸친 목표 이득 대 held-out retention 곡선([[lora-learns-less-forgets-less]] Fig. 3), monitor로서 target prompt에서 KL(π₀ ‖ π)([[rls-razor]] §4), 출력 다양성(prompt당 고유 generation, [[lora-learns-less-forgets-less]] §4.5). ch-00이 정의한 paired uncertainty를 보고하라. Online-merging의 0.5-1.3점 이득은 이를 필요로 하는 크기다([[online-merging-optimizer]] Table 1).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| AdamW를 의도하면서 `torch.optim.Adam(weight_decay=…)` 사용 | 오류는 없지만 작은 그래디언트 파라미터가 큰 그래디언트 파라미터보다 더 빨리 수축(§1 계산 예 2에서 1,000×) | 옵티마이저 class를 확인한다. `adam.py` L366-L367은 decay를 그래디언트에 더한다 |
| decay 형식을 맞추지 않고 보고서의 λ를 복사 | step별 decay가 LR만큼 다름(LR 3e-4에서 independent 1e-4 대 coupled 3e-5) | 학습 코드에서 lr_t·λ 또는 η_t·λ를 계산하고 parameter norm을 기록한다 |
| λ = 0.1로 토큰 임베딩을 decay | embedding norm 하락, gradient norm 상승, 더 많은 spike(spike score 0.16 대 0.092) | embedding L2 norm과 gradient-norm spike score를 기록한다([[olmo-2-optimizer-settings]] §3.4.2) |
| 큰 사전학습 실행에서 ε = 1e-5 | 초기 손실 감소가 느리고 gradient norm이 더 높음([[olmo-2-optimizer-settings]] Fig. 9) | layer별 gradient RMS를 기록해 ε과 비교한다 |
| fp16에서 `scaler.unscale_` 전 clipping | clipping 전 norm이 S·‖g‖ 근처(예: 5e4)로 기록되고, 매 step clip 계수가 1/(S·‖g‖), 예: 1.9e-5 근처이며 손실 감소가 느림 | `backward → unscale_ → clip → step → update` 순서를 assert하고 clip 계수를 기록한다([[pytorch-adamw-clip-amp]]) |
| gradient accumulation이 끝나기 전에 `unscale_` 호출 | 두 번째 호출에서 `RuntimeError`, 또는 partial gradient에 clipping 적용 | 축적된 그래디언트에서 optimizer step당 한 번 clipping한다(`amp_examples.rst` L114-L117) |
| β₂ = 0.999를 “spike 뒤 step이 너무 큼”으로 해석 | 점진적 그래디언트 상승을 따르는 발산에 잘못된 수정을 선택 | layer별 m̂/√v̂를 simulation 또는 logging한다. spike 뒤 step은 작고 상승하는 그래디언트는 큰 step을 준다(§4) |
| global clipping을 reward outlier 보호로 취급 | 올바른 RL 표본이 음의 advantage를 받고 clip 계수는 정상 | group별 advantage 부호와 reward 범위를 기록하고 reward를 clip 또는 normalize한다(§5) |
| 목표 metric이 계속 개선된다는 이유로 epoch 추가 | HumanEval이 peak 뒤 하락하는 동안 held-out retention은 0.595 → 0.414로 하락 | 매 epoch retention suite를 평가한다([[lora-learns-less-forgets-less]] Tables S5-S6) |
| weight-space 거리만 망각 monitor로 사용 | 망각 없이 큰 weight 변화, 또는 그 반대 | target prompt에서 KL(π₀ ‖ π)을 기록하고 held-out 평가를 실행한다([[rls-razor]] Table 1) |
| mid-run loss 또는 조정되지 않은 baseline으로 새 옵티마이저가 낫다고 선언 | LR decay 뒤 또는 AdamW LR 조정 뒤 순위가 역전 | 학습 종료 시 옵티마이저별 sweep과 breadth suite로 비교한다([[fantastic-pretraining-optimizers]] Fig. 1, §4.2) |
| 확인 없이 사전학습 때와 다른 옵티마이저로 checkpoint 미세조정 | 일치하는 옵티마이저보다 MMLU 및 GSM8K가 낮음 | 두 옵티마이저 모두로 짧은 SFT를 실행한다([[muon-scalable-moonlight]] Tables 6-7) |
| 낮은 LR RL에서 순수 bf16 weight 사용 | 일부 파라미터가 전혀 바뀌지 않음(sparse update) | float32 또는 master-weight 실행과 update sparsity를 비교한다([[rls-razor]] §6; ch-02) |

## 이해도 점검

1. Adam의 step은 모든 그래디언트에 상수를 곱해도 불변이다. 그런데 fp16에서 unscaling 전에 clipping하면 §5 예에서 AdamW 업데이트가 의도한 크기의 약 5%로 줄어든다. 업데이트의 어떤 항이 불변성을 깨며, 파라미터 수와 함께 효과가 커지는 이유는 무엇인지 설명하라.
2. §2의 가중 이차 모형을 사용해 경사하강법의 learning rate를 올려도 희귀 클래스 격차를 좁힐 수 없는 이유와, sign descent의 어떤 성질이 π_k 의존성을 제거하는지 설명하라.
3. [[why-weight-decay-modern-dl]]은 유효 learning rate를 따르는 LR schedule을 가진 no-decay 실행으로 weight-decay 실행을 재현하지만 float32에서만 가능하다. 이것이 weight decay의 두 효과 중 schedule로 대체 가능한 것은 무엇인지, bf16 학습에서 왜 중요한지를 무엇을 뜻하는가?
4. §4 예에서 β₂ = 0.999는 spike 시 β₂ = 0.95보다 큰 step을 주지만 100 step 뒤에는 더 작은 step을 준다. 두 phase를 통해 m과 v를 추적하고 긴 β₂ memory가 위험해지는 gradient 패턴이 무엇인지 말하라.
5. InstructGPT는 100× 큰 KL 계수가 pretraining-gradient mix가 제거한 회귀를 제거하지 못함을 발견했다. 각 항이 제약하는 분포의 관점에서 두 anchor의 차이를 설명하고, Online Merging Optimizer의 SFT delta 사용과 연결하라.
6. [[rls-razor]]는 weight-change L1이 망각의 약한 예측변수(R² 0.34)임을 발견하는 반면, §6의 Σα_t 상한은 weight space로 진술된다. 두 진술이 모두 성립할 수 있는 이유와, 중단 시점을 결정하기 위해 미세조정 실행 동안 무엇을 기록할지 설명하라.
7. Moonlight는 Muon의 약 2× compute efficiency를 보고했으나 [[fantastic-pretraining-optimizers]]는 1.2B 파라미터에서 약 1.1×를 보고한다. 두 결과를 모두 낳을 수 있는 protocol 차이를 나열하고, SFT 단계를 포함해 일반 목적 사전학습 실행을 Muon으로 전환할지 결정할 평가를 설계하라.

## 연결

- 이전: **ch-00 — 일반 능력의 의미와 측정 방법.** §6 및 Generalization lens에서 사용하는 held-out suite, retention 보고, paired-uncertainty 형식을 제공한다.
- 다음: **ch-02 — 수치 정밀도, 결정성, 학습-추론 불일치.** fp16 loss scaling, bf16, fp8을 다루며, 이들이 §5 clipping 순서 적용 여부 및 작은 업데이트 표현 가능 여부(§6)를 결정한다.
- **ch-03 — Learning-Rate Schedule, Batch Size, Initialization, Normalization.** schedule이 §6 예산의 α_t를 정하며, μP와 initialization은 여기 아닌 그 장에서 다룬다.
- **ch-05 — Batch Size, Sequence Length, Tokens Seen을 바꾸는 분산 학습 선택.** shard 전반에서 global gradient norm을 계산한다.
- **ch-12a — 사전학습 중 암기, 지식 습득, 일반화** 및 **ch-13a — 능력 축으로서 다국어 커버리지와 어휘.** 희귀 토큰 학습(§2)의 데이터 측면.
- **ch-14a — 나란히 본 사전학습 레시피: 예산, batch, schedule, stage mixture.** 완전한 사전학습 레시피 비교.
- **ch-30 — SFT 설계 선택과 일반화에 미치는 효과: Masking, Packing, Template, Epoch, Learning Rate** 및 **ch-30a — Fine-Tuning의 망각 및 Alignment Tax: 측정과 제어.** §6에서 요약한 망각 근거의 측정 및 제어.
- **ch-30c — Generalist Model의 Weight Averaging 및 Model Merging.** Online Merging Optimizer의 사후 대응물인 offline merging.
- **ch-37 — 언어 모델의 Policy-Gradient 기초**, **ch-38 — KL-Controlled RLHF: PPO, InstructGPT, Alignment Tax**, **ch-38a — SFT 대 RL 일반화: On-Policy Data, Base Model KL, Output Diversity**, **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, GSPO**, **ch-43a — Negative Sample 및 Negative Gradient: Likelihood Displacement, Squeezing, Negative Advantage.** §5-§6에서 사용한 RL objective, PPO-ptx, KL-망각 문제, group normalization, negative advantage.

## 출처

- [[adam]] — Adam arXiv:1412.6980v9(Algorithm 1, §2.1 step bound 및 scale invariance, §3 bias correction)과 AdamW arXiv:1711.05101v3(Algorithm 2, Proposition 1-2, 이미지 분류 근거)를 대조한 장 발췌. library card `classics/adam.md`는 2026-09-15에 재검증하지 않았다.
- [[gradient-clipping]] — Pascanu, Mikolov, Bengio arXiv:1211.5063v2의 장 발췌: Algorithm 1과 threshold heuristic(§3.2). library card는 재검증하지 않았다.
- [[pytorch-adamw-clip-amp]] — PyTorch v2.5.0 AdamW 및 Adam step, `clip_grad_norm_`, GradScaler 초기 scale, 문서화된 unscale-then-clip 순서.
- [[heavy-tailed-class-imbalance-adam]] — SGD 대 Adam의 빈도 그룹 학습 손실, 가중 이차 모형, sign descent, reweighting.
- [[why-weight-decay-modern-dl]] — one-pass GPT-2-124M 학습의 weight decay: 유효 LR, 최종 손실, bf16 발산, 이 효과에 대한 L2 동치.
- [[small-scale-proxies-instabilities]] — independent 대 coupled weight decay, ε 근처 gradient RMS, 4.8B에서 ε = 1e-15, 기본 AdamW 및 clipping 설정.
- [[gpt-3-few-shot]] — GPT-3 App. B의 optimizer, clipping, weight-decay 값.
- [[rlhf-instructgpt]] — InstructGPT library card(2026-09-15에 미검증; 그 hyperparameter 표는 사용하지 않음).
- [[rlhf-instructgpt-optimizer-settings]] — InstructGPT Adam betas, SFT epoch와 과적합, RM epoch, PPO LR 발산, PPO-ptx 계수 및 ablation.
- [[llama-2-recipe]] — Llama 2 사전학습, SFT, PPO의 optimizer 행.
- [[llama-3]] — short-context DPO가 long-context 능력을 해치지 않았다는 Llama 3 관찰. 저자들은 이를 더 적은 optimizer step에 귀속했다(§4.3.4).
- [[llama-3-recipe]] — Llama 3 scaling-law weight decay 문구, 405B optimizer 공개, SFT 및 DPO learning rate.
- [[olmo-2]] — OLMo 2 보고서의 library card(2026-09-15에 미검증); 보고서 자체에 연결됨.
- [[olmo-2-optimizer-settings]] — OLMo 2 7B 공개 optimizer config, ε 및 embedding-decay ablation, spike score, LR crossover.
- [[deepseek-v3-recipe]] — DeepSeek-V3 AdamW betas, weight decay, clipping.
- [[kimi-k2]] — MuonClip 메커니즘 및 vanilla Muon logit 증가 관찰.
- [[kimi-k2-recipe]] — Kimi K2 optimizer, weight decay, τ, SFT 및 RL의 Muon.
- [[glm-4-5-recipe]] — GLM-4.5 Muon 설정과 parameter group.
- [[qwen-2.5-recipe]] — LR 7e-7에서 Online Merging Optimizer를 사용한 Qwen2.5 DPO.
- [[lora-learns-less-forgets-less]] — LoRA와 full fine-tuning의 epoch별 학습 및 망각, 망각 제어로서 weight decay, LR sweep.
- [[lora-without-regret]] — SFT와 RL에서 LoRA 대 full fine-tuning, 10× learning-rate 비율(practitioner-evidence).
- [[rls-razor]] — 망각 예측변수로서 새 과제 입력의 KL 대 weight-space 측정, bf16 update sparsity, LLM optimizer 설정.
- [[online-merging-optimizer]] — OnDARE/OnTIES update rule, AdamW 및 regularization baseline 대비 DPO 결과, hyperparameter 효과.
- [[fantastic-pretraining-optimizers]] — 1.2B까지 11개 옵티마이저의 조정된 비교, downstream Table 5, AdamW sweep Table 36.
- [[muon-scalable-moonlight]] — weight decay 및 RMS matching을 갖는 Muon update, compute-efficiency 주장, Moonlight 대 Moonlight-A downstream score, SFT optimizer mismatch.
- [[openrlhf-ppo-recipe]] — OpenRLHF 기본 gradient clipping 및 reward clipping.
