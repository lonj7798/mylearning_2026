<!-- chapter: ch-01
     track: foundations
     title: Optimization Fundamentals for Transformers
     sources: [[adam]], [[gradient-clipping]], [[mixed-precision]], [[lr-schedules]], [[weight-init]]
     figures: figures/beta2-memory.html
-->

# 1장 — 트랜스포머를 위한 최적화 기초

> **핵심 통찰.** 파라미터별 적응형 스텝 크기, *분리된(decoupled)* 가중치 감쇠, 전역 노름 그래디언트 클리핑을 갖춘 AdamW가 2025년 LLM 학습에서 옵티마이저 이야기의 전부다. 학습 루프의 다른 모든 요소는 이 세 축 위에 세워진다.
>
> **지침.** 사전학습에는 `clip_grad_norm_=1.0`과 함께 `AdamW(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)`을 사용하라. SFT와 RL에서는 옵티마이저가 아니라 LR을 낮춰라. 근거가 있을 때만 여기서 벗어나라.

---

## 이 장이 필요한 이유

대부분의 실무자는 별생각 없이 `torch.optim.AdamW`를 집어 든다. 보통은 잘 작동한다. 작동하지 않을 때까지는. 실패할 때는 조용히 실패한다. loss가 평평해지고, 3000 스텝에서 norm이 폭발하고, fp16에서 그래디언트가 NaN이 되고, embedding row에서 `v_hat`이 0으로 언더플로하고, FSDP가 잘못된 전역 norm을 클리핑한다. 이 버그들은 하나하나가 모두 옵티마이저 수준의 버그다. 이 장은 하이퍼파라미터를 고르는 데서 그치지 않고, 이런 문제를 *알아보고* *고칠* 수 있는 mental model을 만든다.

이 장을 읽고 가져가야 할 것은 세 가지다.

1. Adam과 AdamW의 정확한 알고리즘 박스, 그리고 둘을 가르는 한 줄.
2. LLM 기본값(`β₂=0.95`, `wd=0.1`)이 여러분이 배웠을 수 있는 ImageNet 기본값과 다른 이유, 그리고 이를 섞어 썼을 때 생기는 일.
3. 프로덕션 trainer 절반에서 조용히 틀리는 `unscale → clip → step` 순서.

이 모든 내용은 raw-data 라이브러리의 [[adam]]과 [[gradient-clipping]]에서 온다. 이 장은 그것들을 하나의 일관된 이야기로 묶는다.

---

## 1. SGD → Adam → AdamW로 이어지는 흐름

일반 SGD with momentum은 하나의 스칼라 learning rate를 모든 파라미터에 똑같이 적용한다.

```
theta_t = theta_{t-1} - alpha * grad_t                             # SGD
theta_t = theta_{t-1} - alpha * (rho * m_{t-1} + grad_t)           # SGD + momentum
```

이는 볼록에 가까운 landscape에서는 잘 작동한다. 하지만 트랜스포머의 loss surface에서는 좋지 않다. attention weight, embedding, layer-norm scale, FFN projection 사이에서 파라미터별 curvature가 몇 자릿수씩 다르기 때문이다. 하나의 스칼라 LR이 이 모두를 감당할 수는 없다.

**Adam**(Kingma & Ba 2014)은 *파라미터마다* 그래디언트 분산의 running estimate를 유지하고 각 스텝을 그 분산으로 재스케일링해 이 문제를 해결한다. 분산이 큰 그래디언트를 보는 파라미터는 작은 스텝을 받고, 깨끗한 신호를 받는 파라미터는 온전한 스텝을 받는다. 업데이트는 여전히 gradient descent다. 다만 파라미터별 적응형 스텝 크기를 쓸 뿐이다.

**AdamW**(Loshchilov & Hutter 2017)는 미묘하지만 중요한 버그를 고친다. 원래 Adam 논문은 loss에 `λ * θ`를 더하는 방식, 즉 L2 regularization으로 weight decay를 구현하라고 권했다. 하지만 그 항은 gradient를 통해 흘러가고, second-moment estimate `v_t`에 섞이며, *적응적으로 재스케일링*된다. 즉 `v_t`가 큰 파라미터는 사실상 더 약한 weight decay를 받는다. AdamW는 decay 항을 파라미터 업데이트에 직접 적용하여 `m_t`와 `v_t`를 완전히 우회함으로써 이를 고친다.

수정은 코드 한 줄이다. 그 결과가 충분히 커서 2024년 이후의 모든 LLM 보고서에서 "Adam"이라고 쓰인 것은 실제로 AdamW를 뜻한다.

---

## 2. Adam과 AdamW, 수식과 코드

[[adam]]에서 가져온, timestep `t`와 gradient `g_t`에 대한 update rule은 다음과 같다.

```
m_t    = β₁ · m_{t-1} + (1 - β₁) · g_t                 # 1st moment (momentum)
v_t    = β₂ · v_{t-1} + (1 - β₂) · g_t²                # 2nd moment (variance)
m_hat  = m_t / (1 - β₁ᵗ)                               # bias correction
v_hat  = v_t / (1 - β₂ᵗ)
θ_t    = θ_{t-1} - α · m_hat / (√v_hat + ε)            # Adam update
θ_t    = θ_{t-1} - α · (m_hat / (√v_hat + ε) + λ·θ_{t-1})   # AdamW update  ← the whole change is λ·θ
```

나머지는 모두 공유된다. bias correction(`m_hat`, `v_hat`)이 중요한 이유는 `m_t`와 `v_t`가 0으로 초기화되기 때문이다. 보정하지 않으면 초기 스텝은 실제 gradient statistics를 `1 − βᵗ` 배만큼 과소평가한다.

다음은 실제 PyTorch의 핵심 AdamW step 소스다(`torch.optim.adamw._single_tensor_adamw`에서 가져와 단순화).

```python
# decoupled weight decay happens BEFORE the adaptive step
param.mul_(1 - lr * weight_decay)

# momentum + second-moment EMAs
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

# bias correction
bias_correction1 = 1 - beta1 ** step
bias_correction2 = 1 - beta2 ** step
step_size = lr / bias_correction1
bias_correction2_sqrt = bias_correction2 ** 0.5

# adaptive step
denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
param.addcdiv_(exp_avg, denom, value=-step_size)
```

첫 줄이 AdamW의 기여 전부다. 이 줄을 지우면 Adam이 된다. 대신 EMA 전에 `grad`에 `weight_decay * param`을 더하면 L2-Adam이 된다. 역사적으로 AdamW가 대체한, 문제가 있는 옵티마이저다.

**실무상 함정.** PyTorch에서 `torch.optim.Adam(..., weight_decay=0.1)`은 AdamW가 *아니다*. 그것은 L2-Adam이며, Loshchilov & Hutter가 망가져 있다고 보인 바로 그 방식이다. 항상 `torch.optim.AdamW`를 명시적으로 사용하라. 취미 규모 학습 코드에서 가장 흔한 옵티마이저 버그가 이것이다.

---

## 3. LLM 하이퍼파라미터 기본값과 그 차이의 이유

다음은 [[adam]]의 표에 각 숫자가 운영상 무엇을 의미하는지 주석을 붙인 것이다.

| 하이퍼파라미터 | 사전학습 | SFT | RL (PPO/GRPO) | 하는 일 |
|---|---|---|---|---|
| `β₁` | 0.9 | 0.9 | 0.9 | momentum time scale(약 10스텝 half-life) |
| `β₂` | **0.95** | 0.95 – 0.999 | 0.95 | variance-memory time scale(약 20스텝 vs 약 1000스텝) |
| `ε` | 1e-8 | 1e-8 | 1e-8 (또는 fp16에서는 1e-5) | denominator floor |
| `weight_decay` | **0.1** | 0.0 – 0.01 | 0.0 | 파라미터 shrinkage rate |
| Peak `lr` | 1e-4 to 6e-4 | 1e-5 to 5e-5 | 1e-6 to 1e-5 | step-size envelope |

두 숫자는 고전적 기본값과 다르므로 설명이 필요하다.

**`β₂ = 0.95`(0.999가 아님).** second moment `v_t`는 파라미터별 gradient variance를 근사한다. `β₂ = 0.999`는 약 1000스텝의 기억을 준다. LLM 사전학습 규모에서는 loss landscape가 non-stationary하기 때문에(curriculum, data shift, LR warmup) 그 기억은 너무 길다. `v_t`가 현실보다 늦게 따라오고 effective step size가 stale해진다. `β₂ = 0.95`는 약 20스텝의 기억을 가지며 현재 gradient distribution을 추적한다. 이는 Llama 1의 선택이었고, 이제 GPT, Qwen, DeepSeek, OLMo 전반의 표준이다.

`β₂ ∈ {0.9, 0.95, 0.999}`에서 gradient spike에 `v_hat`이 어떻게 반응하는지 보여주는 대화형 시각화는 `figures/beta2-memory.html`을 보라.

**`weight_decay = 0.1`(1e-4가 아님).** Transformer는 image classifier보다 훨씬 더 과파라미터화되어 있다. 일반화 prior로서 weight decay의 유용성은 데이터 토큰 대비 파라미터 비율과 함께 커진다. `0.1`은 100B-token 사전학습 run에 경험적으로 맞다. 하지만 보통 0 또는 `0.01`로 낮추는 10K-sample SFT에는 지나치게 공격적이다.

**제외 대상.** LayerNorm scale, bias, embedding scale parameter는 항상 weight decay에서 제외하라. 관례는 두 그룹 optimizer다.

```python
decay_params, no_decay_params = [], []
for n, p in model.named_parameters():
    (no_decay_params if any(k in n for k in ("bias", "norm.weight", "norm.bias"))
     else decay_params).append(p)
optim = AdamW([
    {"params": decay_params,    "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

이 패턴은 코드 5줄이면 되고 사전학습 종료 perplexity를 0.1–0.5% 바꾼다. 사실상 공짜다.

---

## 4. Gradient clipping — 안전망

Adam의 adaptive rescaling은 exploding gradient를 막아주지 *않는다*. 직관은 이렇다. `v_hat`은 *running* average이므로, 100배짜리 gradient가 한 스텝 들어오면 `v_t`가 따라잡기 전에 현재 scale을 뚫고 지나간다. 해결책은 Pascanu 2013([[gradient-clipping]]) 이후 존재해 왔고, LLM 학습에서는 필수다.

**Global-norm clipping** — 유일하게 올바른 변형:

```
g_norm = sqrt(sum over all params of ||g_i||²)
if g_norm > c:
    g_i ← g_i · (c / g_norm)        # same rescale factor applied to every tensor
```

방향은 보존되고 크기만 제한된다. 주변에서 볼 수 있는 두 대안은 둘 다 더 나쁘다.

- **Clip-by-value**(`g_i ← clip(g_i, -c, c)` elementwise)는 descent direction을 망가뜨린다. language model에는 쓰지 마라.
- **Per-tensor norm clip**(각 파라미터에 대해 `clip_grad_norm_(p, c)`를 loop)은 optimizer를 작은 tensor 쪽으로 편향시킨다. 이것도 쓰지 마라.

**현대적 기본값**: 사전학습과 SFT에서는 `max_grad_norm = 1.0`, reward spike가 advantage outlier를 만드는 RL rollout에서는 `0.5 – 1.0`.

**조용히 깨지는 순서.** mixed precision과 FSDP에서는 pipeline이 정확히 다음과 같아야 한다.

```
1. loss.backward()                    # accumulates scaled gradients
2. scaler.unscale_(optimizer)         # divide gradients by loss-scale S
3. clip_grad_norm_(params, 1.0)       # NOW the threshold is meaningful
4. scaler.step(optimizer)             # step + update scaler state
5. scaler.update()
```

unscale 전에 clip하면 threshold가 S(보통 2^15 이상) 배만큼 어긋나서 clip이 아무것도 하지 않는다. step 뒤에 clip하면, 말 그대로 clip하지 않은 것이다.

**FSDP/ZeRO-3의 경우.** global norm은 scaling 전에 *모든 shard에 걸쳐* 계산되어야 한다. FSDP에서 순진하게 `clip_grad_norm_(local_shards, 1.0)`를 호출하면 norm을 과소계산하고 rank마다 서로 다른 rescale을 만들어 조용한 divergence로 이어진다. PyTorch에서는 `FullyShardedDataParallel.clip_grad_norm_`를, DeepSpeed에서는 `engine.clip_grad_norm_`를 사용하라. 직접 만들지 마라.

**계측 팁.** `pre_clip_grad_norm`을 training metric으로 추적하라. `pre_clip_grad_norm`의 100배 spike는 보통 loss spike를 1–5스텝 앞서 예고한다. 이것은 사전학습 run에서 가장 좋은 early-warning signal이다.

---

## 5. "트랜스포머에는 momentum이 필요 없다"는 신화

몇 년마다 누군가가 적절한 schedule을 붙인 SGD+momentum이 language modelling에서 AdamW와 맞먹을 수 있다고 주장한다. 작은 모델과 특정 schedule에서는 기술적으로 맞다. 프로덕션 규모에서는 틀리다.

이유는 이렇다. Transformer 학습은 질적으로 다른 세 파라미터 그룹을 업데이트한다.

1. **Embedding** — 매우 sparse한 gradient(보인 토큰의 row만 업데이트됨). adaptive scaling으로 보정하지 않으면 scalar LR은 rare token의 신호를 낭비한다.
2. **Attention QKV projection** — head마다 curvature가 다르다. 어떤 head는 일찍 specialize되어 섬세한 update가 필요하고, 다른 head는 아직 탐색 중이다.
3. **FFN projection** — dense gradient이며 고전적인 MLP 학습에 더 가깝다.

AdamW는 이 셋을 하나의 knob으로 다룬다. SGD+momentum은 layer-group별 LR tuning, group별 gradient clipping, embedding blow-up을 피하기 위한 정교한 warmup이 필요하다. 70B+ 규모에서 아무도 이렇게 하지 않는다. AdamW가 기본값인 이유는 어떤 한 설정에서 가장 빠르기 때문이 아니라 *robust*하기 때문이다.

현대적 미묘함: Lion(Chen 2023), Sophia(Liu 2023), Shampoo, Muon은 모두 사전학습에서 AdamW를 이기려는 시도다. 2025년 말 기준 frontier report(Llama 3, DeepSeek V3, Qwen 3, OLMo 3)는 여전히 AdamW를 사용한다. Muon은 일부 MoE context(Kimi K2가 MuonClip 사용)에서 traction이 있지만, dense transformer 사전학습에서는 AdamW가 여전히 기본값이다.

---

## 6. 2025년 옵티마이저 지형 — 실제로 쓰이는 것

Frontier 채택 순서로 정리한 대안 빠른 가이드:

- **AdamW** — 모든 곳의 기본값. 2025년 공개 frontier recipe 전부(Llama 3/4, Qwen 2.5/3/3.5, DeepSeek V3/R1, OLMo 2/3, Phi-3/4, Tülu 3, Nemotron-Ultra).
- **MuonClip / Muon** — Kimi K2(Moonshot)는 MoE 사전학습 단계에서 MuonClip을 사용한다. 실험적이지만 1T-parameter 규모에서 production-validated.
- **Lion**(Chen 2023) — sign-of-momentum optimizer. AdamW보다 step당 빠르지만 최종 model quality의 variance가 더 높다. SFT에서 일부 community 채택은 있으나 frontier는 아니다.
- **Sophia**(Liu 2023) — Gauss-Newton approximation을 통한 second-order 방식. 인상적인 scaling-law 논문이지만 production evidence는 제한적이다.
- **Shampoo** — distributed preconditioning. Google-scale에서는 잘 작동하지만 sharding 복잡성 때문에 대부분의 open stack에는 들어오지 못했다.
- **Online-Merging Optimizer**(Qwen 2.5) — DPO 중 checkpoint를 평균하는 변형. task-specific이며 범용 AdamW 대체재가 아니다.

2026년 경험칙: 보고서가 어떤 optimizer를 쓰는지 말하지 않는다면 AdamW를 쓴 것이다.

---

## 7. 실무자용 cheat-sheet

```python
# The 90% case: pretraining, SFT, and RL, with the right defaults baked in.

import torch
from torch.optim import AdamW

def build_optimizer(model, lr, stage="pretrain"):
    betas = {"pretrain": (0.9, 0.95), "sft": (0.9, 0.95), "rl": (0.9, 0.95)}[stage]
    wd    = {"pretrain": 0.1,         "sft": 0.01,        "rl": 0.0}[stage]

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if any(k in n for k in ("bias", "norm", "embed_tokens")) else decay).append(p)

    return AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8,
    )

# Training step — the canonical pipeline.
for batch in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch).loss
    loss.backward()
    # bf16 does NOT need a scaler; fp16 does. See ch-02.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## 연결과 다음 내용

- **[[mixed-precision]] / ch-02** — fp16 loss scaling + clip 전에 `GradScaler.unscale_(optimizer)`. bf16은 이 고통의 대부분을 없애며, 2025년 run이 기본적으로 bf16을 쓰는 이유다.
- **[[lr-schedules]] / ch-03** — warmup이 필요한 이유 중 하나는 AdamW의 bias correction 때문에 처음 약 100스텝이 사실상 더 높은 LR로 실행되기 때문이다. warmup은 `v_hat`이 안정화되도록 해준다.
- **[[weight-init]] / ch-03** — μP width-transfer rule은 AdamW의 파라미터별 update scale이 width와 무관하게 `O(α)`라는 점에 의존한다. μP와 SGD를 섞지 마라.
- **[[ppo]] / ch-36** — PPO fine-tuning은 policy gradient가 noisy하기 때문에 같은 AdamW를 LR ≈ 1e-6에서 사용한다. Reward spike는 RL에서 embedding blow-up에 해당한다. 같은 이유로 clip하라.
- **ch-05 / ch-07** — distributed training과 failure-mode diagnosis는 FSDP clip-norm correctness 문제와 잘못된 clip/step 순서가 만드는 silent-drift bug를 다시 다룬다.

## 더 읽을거리

- [[adam]] — Kingma-Ba 2014 + Loshchilov-Hutter 2017의 전체 발췌.
- [[gradient-clipping]] — Pascanu 2013; canonical treatment.
- [[karpathy-training-neural-net-recipe]] — "monitor and clip the gradient norm"를 타협 불가능한 규칙으로 다룬다.
- [[mixed-precision]] — 이 장이 의도적으로 ch-02로 미룬 loss-scaling interaction.

## 함께 보는 시각화

**[figures/beta2-memory.html](figures/beta2-memory.html)** — `β₂ ∈ {0.9, 0.95, 0.999}`에서 second-moment EMA `v_hat`이 단일 gradient spike에 어떻게 반응하는지 보여주는 대화형 슬라이더. Llama가 왜 0.95를 골랐는지에 대한 직관을 만드는 데 사용하라.
