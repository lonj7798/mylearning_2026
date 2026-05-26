<!-- chapter: ch-03
     track: foundations
     title: LR Schedules, Weight Init, Norms
     sources: [[lr-schedules]], [[weight-init]], [[batch-vs-layer-norm]], [[adam]], [[mixed-precision]]
     figures: figures/lr-schedules.html
-->

# 3장 — LR Schedule, Weight Init, Norm

> **핵심 통찰.** 서로 분리되어 보이는 세 관심사, 즉 step size가 *언제* 큰지, weight가 *어떻게* 시작하는지, activation을 *무엇이* bounded하게 유지하는지는 같은 문제를 세 각도에서 본 것이다. 그 문제는 *학습 전반과 깊이 전반에서 잘 conditioned된 forward/backward pass를 유지하는 것*이다. 하나가 깨지면 나머지가 구해주지 못한다.
>
> **지침.** 2025년에 새 Transformer를 만든다면: linear warmup(2000–8000 step)을 붙인 cosine 또는 WSD schedule; 모든 linear layer는 `N(0, 0.02)`로 초기화하고 residual-projection scaling은 `1/√(2L)`; RMSNorm + pre-norm + final `ln_f`; norm reduction은 fp32.

---

## 이 장이 필요한 이유

ch-01 / ch-02에서 optimizer([[adam]])와 precision stack([[mixed-precision]])을 배웠다. 하지만 학습의 *shape*가 step 1부터 틀리면 그 어떤 것도 의미 없다. `v̂`가 안정화되기 전에 LR이 너무 높거나, init scale 때문에 첫 forward pass가 saturation되거나, norm placement가 residual-stream variance를 depth와 함께 키우면 그렇다. 이 세 knob은 서로 상호작용한다. 이 장은 기본값을 못 박고, 왜 그런 값인지 설명하며, failure mode를 표시한다.

주요 출처: [[lr-schedules]], [[weight-init]], [[batch-vs-layer-norm]].

---

## 1. Learning-rate schedule — 실제로 중요한 네 계열

출처: [[lr-schedules]]. 현대 LLM 학습을 지배하는 네 계열이 있다.

**Linear warmup** — 항상 앞에 붙는다.

```
lr(t) = peak_lr · t / warmup_steps     for  t < warmup_steps
```

**Cosine annealing** — LLM 사전학습 기본값:

```
lr(t) = min_lr + 0.5 · (peak_lr − min_lr) · (1 + cos(π · (t − warmup) / (T − warmup)))
```

Min LR은 보통 `0.1 × peak_lr`(Llama, GPT-3) 또는 `0.0`(일부 Qwen run)이다. `T`는 전체 step budget이다. Cosine의 유일한 failure mode는 horizon mismatch다(Chinchilla Figure A1). step T에서 cosine-to-zero를 하도록 잡았는데 0.5T에서 학습을 멈추면 val loss가 약 ~0.5% 나빠진다.

**Inverse-square-root** — Vaswani 2017의 원래 방식:

```
lr(t) = d_model^(−0.5) · min(t^(−0.5), t · warmup_steps^(−1.5))
```

model width에 따라 self-scale되고 이론적으로 우아하다. 고정 budget 실무에서는 cosine이 약 ~0.3% 이긴다. 일부 encoder pretrain(T5 variant)에는 여전히 쓰이지만 frontier LLM에는 쓰이지 않는다.

**WSD — Warmup-Stable-Decay**(Hu 2024, DeepSeek):

```
phase 1 (warmup):  lr ramps 0 → peak_lr           [0, warmup]
phase 2 (stable):  lr = peak_lr                   [warmup, T − decay]
phase 3 (decay):   lr → min_lr  (10–20% of T)     [T − decay, T]
```

WSD의 대표 장점은 stable phase가 *checkpoint-able*하다는 것이다. 어떤 stable-phase checkpoint에서도 10% decay run을 fork할 수 있어, 재학습 없이 여러 training length에서 "final loss"를 얻는다. DeepSeek와 MiniCPM이 하나의 trunk에서 많은 model variant를 만드는 방식이다.

대화형 비교는 `figures/lr-schedules.html`을 보라. knob을 움직이면 세 schedule이 함께 그려지는 것을 볼 수 있다.

### 현대적 기본값

| 설정 | 사전학습 | SFT | RL (PPO/GRPO) |
|---|---|---|---|
| Warmup steps | 2000–8000 | 100–500(전체의 약 3%) | 0–50 |
| Schedule | cosine 또는 WSD | cosine 또는 constant | constant |
| Peak LR | 3e-4 (1B) → 1.2e-4 (70B) → 8e-5 (405B) | 2e-5 (Llama-3 SFT) | 1e-6 – 1e-5 |
| Min LR | 0.1 × peak | 0.1 × peak | — |

**AdamW에서 warmup이 타협 불가능한 이유.** `v̂ = v_t / (1 − β₂ᵗ)`는 처음 몇 step 동안 추정이 좋지 않다. bias-correction denominator가 작으므로 effective LR이 부풀려진다. warmup이 없으면 첫 update에서 NaN이 날 수 있다. GPT-3는 375M token warmup을 썼고, Llama-3는 8000 step을 썼다. 7B+ model에서 zero warmup은 divergence한다. ch-01의 bias-corrected `v̂` 논의를 보라.

**흔한 함정.** 잘못된 horizon에 맞춘 cosine → Chinchilla식 penalty. 높은 LR에서 너무 짧은 warmup → step ~150 근처 loss spike. Constant-LR finetuning을 "영원히" 하는 것 → 적절한 decay보다 final loss가 1–3% 나쁨. WSD decay phase < 5% → cosine보다 성능이 떨어짐.

---

## 2. Weight initialization — 건너뛸 수 없는 세 규칙

출처: [[weight-init]]. 모든 것을 떠받치는 원리는 두 가지다. layer 전반의 **variance preservation**, 그리고 depth 전반의 **residual-stream budget**.

**Variance preservation(Xavier/He).** Linear layer `y = Wx`에 대해:

```
Xavier (tanh/linear):  Var(W) = 2 / (fan_in + fan_out)
He     (ReLU):         Var(W) = 2 / fan_in
LeCun  (SELU):         Var(W) = 1 / fan_in
```

Transformer는 linear와 ReLU 사이에 있는 GELU / SwiGLU variant를 쓴다. 실무에서 현대 코드는 activation으로부터 variance를 유도하지 않고 **GPT-2 / Megatron rule**을 직접 사용한다.

```
all linear layers:   N(0, 0.02)
all embeddings:      N(0, 0.02)   (some recipes use N(0, 1e-5) for shared LM-head)
residual projections: additionally scaled by 1 / √(2L)    (L = # of residual blocks)
```

`1/√(2L)` scale은 GPT-2 trick이다. 이것이 없으면 residual-stream variance는 depth에 대해 *선형으로* 증가한다. 100개 이상의 block에서는 LM head가 unnormalised logit을 보게 된다. 모든 현대 frontier LLM이 이를 적용한다(Llama, Qwen, Megatron at 530B, OLMo-2).

**Embedding init.** Shared LM-head-with-embedding은 embedding이 기본 PyTorch scale로 초기화되면 큰 gradient를 만든다. Linear layer와 맞추려면 `N(0, 0.02)`를 쓰고, tying한다면 `N(0, 1e-5)`까지 낮춰라.

**빠른 init audit.** 학습 전에 un-trained model에 batch를 forward하라. Activation variance는 block 전반에서 대략 보존되어야 한다(2배 이내). Backward gradient norm은 layer 전반에서 한 자릿수 이내여야 한다. **Initial loss는 `ln(vocab_size)`와 같아야 한다**. uniform-prediction baseline이다. 그렇지 않다면 init이 틀렸고 학습은 자기 자신과 싸우게 된다.

### μP (muTransfer) — hyperparameter transfer

출처: [[weight-init]]. 요지는 간단하다. network를 다시 parameterize하여 *최적* LR, init scale, beta가 **width-invariant**가 되게 한다. 40M-parameter proxy model에서 sweep하고, 6B+에 그대로 transfer한다. Cerebras-GPT와 GPT-4의 HP sweep 일부가 compute를 절약한 방식이다.

abc-parametrisation(단순화):

```
input layer init:    O(1)
hidden layer init:   O(1/√d)
output layer init:   O(1/d)
hidden LR (AdamW):   O(1)    (SGD requires O(1/d))
output multiplier:   1/d applied to logits
```

**Transfer되는 것**: peak LR, beta, init scale, schedule shape.
**Transfer되지 않는 것**: depth-dependent quantity, batch size(여전히 Chinchilla-scaled), data mix.

함정: μP layer와 non-μP layer를 섞는 것(LM head scaling을 빼먹는 것) → transfer property가 파괴된다.

---

## 3. Norm — placement와 formula

출처: [[batch-vs-layer-norm]]. 세 가지 선택이 중요하다. 어떤 normalizer인지, residual에 대해 어디에 놓이는지, reduction이 어떤 precision을 쓰는지.

### LayerNorm vs RMSNorm

```python
# LayerNorm (Ba 2016)
mu     = x.mean(dim=-1, keepdim=True)
sigma2 = x.var(dim=-1, keepdim=True, unbiased=False)
y      = (x - mu) / torch.sqrt(sigma2 + eps) * gamma + beta

# RMSNorm (Zhang & Sennrich 2019)
rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
y   = x * rms * gamma      # no mean subtract, no beta
```

RMSNorm은 mean subtraction과 learnable bias를 없앤다. reduction 하나, subtract 하나, parameter group 하나를 절약한다. 원래 RMSNorm 논문은 quality loss 없이 norm-op speedup 7–64%를 보고했다. Llama, Qwen, DeepSeek, OLMo, Mistral, Gemma가 모두 RMSNorm을 쓴다. **자체 norm을 발명하지 마라.**

### Pre-norm vs post-norm

```
# Post-norm (Vaswani 2017 original)
x = LN(x + Sublayer(x))

# Pre-norm (every modern LLM)
x = x + Sublayer(LN(x))
```

Pre-norm은 residual-stream gradient를 depth에 대해 `O(1)`로 유지한다. post-norm의 gradient는 depth에 대해 선형으로 커지고, 폭발을 피하려면 섬세한 warmup이 필요하다. 24개 이상의 layer에서는 exotic trick 없는 post-norm training이 step 1k 전후에서 divergence한다. pre-norm의 대가는 residual-stream magnitude가 depth와 함께 커진다는 점이다. 그래서 LM head 전에 final `ln_f`가 필요하고, 모든 현대 architecture가 이를 가진다.

### 2024–2025 변형

- **QK-norm.** Attention dot product 전에 `LayerNorm(Q)`와 `LayerNorm(K)`를 적용한다. Long context에서 attention-logit explosion을 막는다. ViT-22B, OLMo-2, Qwen-2.5가 사용한다.
- **Reordered-norm(OLMo-2).** MLP의 residual add *뒤*에 두 번째 norm을 둔다. 경험적으로 mid-training loss spike를 제거했다.
- **Sandwich-norm.** 각 sub-layer 전후에 모두 norm을 둔다. 이득은 작고 niche하다.
- **DeepNorm** — 매우 깊은 *post-norm* stack용. 2025년 mainstream LLM에는 쓰이지 않는다.

### Precision rule(ch-02와 교차 링크)

LayerNorm/RMSNorm 안의 `mean`과 `var` reduction은 주변 compute가 bf16 또는 fp8이어도 **반드시** fp32에서 실행되어야 한다. ch-02 §3의 `RMSNorm.forward` snippet을 보라. 이것은 home-grown training code에서 가장 잦은 precision bug다.

---

## 4. 세 요소의 상호작용 — failure case 하나 따라가기

기본 PyTorch init(`kaiming_uniform_`), post-norm LayerNorm, warmup 없음, `β₂=0.999`, 2T token에 맞춘 cosine schedule이지만 실제로는 800B에서 멈추는 70-layer model을 생각해 보자.

Failure sequence:

1. **Step 0–10.** Residual-stream variance가 depth에 대해 거의 선형으로 증가(post-norm without residual scale) → logit이 O(√L). Initial loss가 `ln(vocab_size)`보다 훨씬 높다.
2. **Step 10–50.** AdamW의 `v̂`가 잘 추정되지 않음(`β₂=0.999`, no warmup). Effective LR이 peak_lr 설정의 5–10배. Gradient spike.
3. **Step 50.** Norm reduction 하나가 실수로 bf16에서 실행됨(`.float()` cast 누락) → normalisation에 0.1% bias → block마다 작은 추가 drift → 70 layer에 걸쳐 compound.
4. **Step ~150.** grad spike 때문에 loss가 NaN이 되거나 plateau. clip_grad_norm 1.0이 일부 damage를 흡수하지만 training dynamics는 이미 off-manifold.
5. **Step 800B tokens.** 학습은 깨끗하게 멈추지만 cosine은 2T에 맞춰져 있었으므로 LR이 실제로 decay되지 않았다. 최종 val loss는 competent tuning run보다 1–1.5% 나쁘다.

이 failure의 모든 단계는 이 장의 기본값으로 막을 수 있다.

---

## 5. Drop-in reference code

```python
# ----- init (GPT-2 / Llama style) -----
def init_weights(module, n_layer):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

def scale_residual_projections(model, n_layer):
    scale = 1.0 / math.sqrt(2 * n_layer)
    for name, p in model.named_parameters():
        # residual projections: attention output W_O and MLP W_2
        if any(k in name for k in ("attn.out_proj", "mlp.down_proj", "w2.weight")):
            p.data.mul_(scale)

# ----- LR schedule (warmup + cosine) -----
def lr_at(step, warmup, total, peak, min_lr_ratio=0.1):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (min_lr_ratio + (1 - min_lr_ratio) * cosine)

# ----- RMSNorm with fp32 reduction -----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps
    def forward(self, x):
        in_dtype = x.dtype
        x_fp32   = x.float()
        rms      = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(in_dtype) * self.weight
```

---

## 연결과 다음 내용

- **[[adam]] / ch-01** — warmup은 AdamW의 bias correction *때문에* 필요하다. μP LR rule은 AdamW semantics를 가정한다.
- **[[mixed-precision]] / ch-02** — norm reduction은 fp32; fp8 run은 norm을 bf16/fp32에 유지한다.
- **ch-04 (packing + masking)** — position ID는 packed sub-sequence마다 reset된다. RoPE table init은 자체 rule을 따른다.
- **ch-05 (FSDP)** — μP의 LR rule은 FSDP sharding strategy와 그대로 결합된다.
- **[[ppo]] / ch-36** — RL은 policy가 이미 좋기 때문에 아주 작은 warmup(0–50 step)과 constant LR을 쓴다. drift가 적이다.

## 더 읽을거리

- [[lr-schedules]] — cosine, inverse-sqrt, WSD, 전체 hyperparameter table.
- [[weight-init]] — Glorot → He → GPT-2 → μP, residual-scaling derivation 포함.
- [[batch-vs-layer-norm]] — LayerNorm / RMSNorm / QK-norm / reordered-norm.
- [[olmo-2]] — QK-norm + reordered-norm의 공개 recipe와 documented ablation.

## 함께 보는 시각화

**[figures/lr-schedules.html](figures/lr-schedules.html)** — linear-warmup + cosine / inverse-sqrt / WSD curve를 interactive total-step, warmup-fraction, decay-phase slider와 함께 나란히 그린 plot. 잘못된 horizon에 맞춘 cosine이 왜 perplexity 비용을 만드는지 체감하는 데 사용하라.
