<!-- chapter: ch-03
     track: ledger
     phase: read
     title: Activations and Gradient Checkpointing
     deps: [[ch-01]], [[ch-02]]
     sources: [[gradient-checkpointing-chen]], [[selective-recompute-korthikanti]]
     created_at: 2026-07-16
-->

# 3장 — Activation과 Gradient Checkpointing

> **핵심 통찰.** Transformer의 forward pass는 backward pass가 chain rule을 따라 곱셈할 때 필요한 모든 intermediate tensor를 보관해야 한다. 이 tensor들은 정적 항에서 O(L·s·b·h), attention 항에서 O(L·a·s²·b/h)로 scaling하므로, activation memory는 training ledger에서 sequence length에 민감하고 지배적인 항목이 된다. Gradient checkpointing(Chen 2016)은 √L개의 경계 activation만 저장하고 backward 중 그 사이의 모든 값을 recompute하여 O(L) footprint를 O(√L)로 바꾼다. 그 대가는 정확히 한 번의 추가 forward pass(약 33%의 추가 compute)다. Selective recomputation(Korthikanti 2022)은 이 tradeoff를 개선한다. 저장 비용이 낮은 activation(MLP, LayerNorm)은 유지하고, 크지만 recompute 비용이 낮은 activation(attention score matrix)만 버려서 4% 미만의 recompute overhead로 5×의 memory 절감을 얻는다.

> **지침.** Activation은 bandwidth가 아니라 compute를 지불하여 memory를 되찾을 수 있는 유일한 ledger 항목이다(즉, 값을 저장하는 대신 나중에 다시 계산한다). 우선 memory budget을 지키기 위한 baseline으로 full gradient checkpointing을 활성화하라. throughput이 중요하면 selective recomputation(Megatron의 default)으로 전환하여 작은 compute tax(추가 recompute 비용)만으로 대부분의 memory 절감 효과를 얻어라. Tensor parallelism에서 남는 activation 복제를 제거하려면 sequence parallelism을 함께 사용하라.

---

## 1. Backward Pass가 보관해야 하는 것

Neural network의 모든 parameter는 chain rule로 계산한 gradient에 의해 update된다. 합성 함수 f(g(x))의 chain rule로 ∂f/∂g를 계산하려면 최종 output f뿐 아니라 g(x)의 값도 사용할 수 있어야 한다. Attention softmax, GELU activation, layer normalization처럼 여러 non-linear operation을 포함하는 Transformer layer에서는 backward pass가 각 값을 multiplier로 사용할 수 있도록 forward pass가 operation 경계마다 intermediate tensor를 보관해야 한다.

표준 training run(checkpointing 없음)에서는 이로 인해 layer 수와 layer당 activation volume의 곱에 비례하는 memory footprint가 발생한다.

```
Total activation memory (no recompute) = L * per_layer_activation_bytes
```

Tensor-parallel degree가 *t*인 Transformer의 layer당 항은 다음과 같다([[transformer-math-101]], [[selective-recompute-korthikanti]]).

```
per_layer_bytes = s * b * h * (10 + 24/t + 5*a*s / (h*t))
```

여기서:
- `s` = sequence length(token 수)
- `b` = GPU당 micro-batch size
- `h` = hidden dimension
- `L` = layer 수
- `a` = attention head 수
- `t` = tensor-parallel degree(1 = TP 없음)

이 formula에는 구조적으로 다른 두 항이 있다.

**Linear term** `(10 + 24/t)·s·b·h`: MLP activation, LayerNorm output, residual connection을 저장한다. s와 b에 대해 linear하게 증가한다.

**Quadratic term** `5·a·s²·b / (h·t)`: softmax와 dropout 이후의 `[b, a, s, s]` tensor인 attention score matrix를 저장한다. 모든 token이 다른 모든 token을 attend하므로 s에 대해 quadratic하게 증가한다.

이 quadratic attention 항 때문에 long-context training 비용이 커진다. s=8,192이면 현실적인 어떤 h에서도 attention 항이 linear 항을 지배하며, s=32,768이면 그 차이가 압도적이다. Formula가 이를 정확히 보여 준다. TP=8(attention 항 내부를 t로 나눔)이어도 algorithm의 도움 없이는 s² scaling을 피할 수 없다(FlashAttention은 [[ch-04]]와 [[ch-05]]에서 다룬다).

### Activation Memory와 Static Floor의 비교

[[ch-01]]에서 static memory floor(activation을 제외하고도 반드시 필요한 최소 memory)를 확립했다. Parameter가 P개인 model에 mixed-precision AdamW를 사용하면 floor는 18 bytes/param이다.
- 2 B: BF16 working weights
- 4 B: FP32 master weights
- 4 B: FP32 gradients
- 8 B: FP32 Adam momentum + variance

Activation은 이 floor 위에 전부 추가된다. 실질적인 sequence length와 batch size에서는 activation이 지배적이다. 대략적인 예로 h=4096, L=32, a=32, s=4096, b=1, t=1인 7B-parameter model은 다음과 같은 activation memory를 생성한다.

```
32 * 4096 * 1 * 4096 * (10 + 24 + 5*32*4096/4096) = 32 * 4096 * 4096 * (34 + 160)
  ≈ 32 * 4096 * 4096 * 194 ≈ 104 GB
```

Static floor가 약 7B * 18 = 126 GB인 것과 비교하면 s=4096의 activation은 이미 static floor에 맞먹는다. s=16384에서는 attention 항이 16×가 된다.

---

## 2. Gradient Checkpointing — Compute 33%를 추가하여 Memory를 O(√n)으로

[[gradient-checkpointing-chen]](Chen et al., 2016)은 activation memory와 recompute를 맞바꾸는 기반 algorithm이다. 핵심 관찰은 *모든* intermediate activation을 저장할 필요가 없다는 것이다. 전략적으로 선택한 일부인 **checkpoint**만 저장하고, 그 사이의 activation은 backward pass에서 필요한 순간 처음부터 recompute할 수 있다.

> **▶ 인터랙티브 companion — [`figures/checkpointing.html`](figures/checkpointing.html)**
> *checkpoint이 물리적으로 무엇이고, 나머지는 어떻게 되는가.* 패널 1은 transformer block 하나를 개별 저장 tensor로 펼쳐서 실제 shape과 바이트 비례 막대로 보여준다 — checkpoint은 **hidden state tensor `[B, T, h]` 하나 = 33.55 MB**이고, 그 대가로 버리는 내부 tensor 약 **1.64 GB** 안에는 `[B, heads, T, T]` attention probability 괴물 **1.07 GB**가 들어있다 (비율 49×). 패널 2는 forward 저장 / backward 재계산을 step별로 애니메이션하며 memory gauge가 `2√n`을 넘지 않는 걸 보여주고, 패널 3은 `k`를 직접 끌어 `k + n/k` 곡선에서 양 끝이 모두 `n`이고 `√n`에서만 최소가 되는 걸 확인하게 한다. 패널 4는 실무의 "block마다 checkpoint"가 왜 `k = n`이 **아닌지**, 패널 5는 그것이 `torch.utils.checkpoint`의 `no_grad` forward / 재forward 동작에 어떻게 대응되는지를 보여준다.

### √n 방식

n-layer network를 크기가 √n인 segment로 나눈다. Segment 경계마다 activation tensor 하나만 저장한다(총 √n개 tensor). Backward pass가 한 segment에 도달하면 저장된 경계부터 그 segment의 내부 activation을 forward 방향으로 recompute한 뒤 local gradient 계산에 사용한다.

```
Forward pass (store phase):
  For i = 0, sqrt(n), 2*sqrt(n), ..., n:
    x[i] = forward_segment(x[i - sqrt(n)])   # recompute the segment
    checkpoint[i] = x[i]                     # store only the boundary

Backward pass (recompute phase):
  For each segment s (in reverse):
    recompute all activations in s from checkpoint[s.start]
    compute gradients using recomputed activations
    free the recomputed activations
```

Memory cost는 저장된 checkpoint O(√n)과 한 번에 하나의 live segment에서 recompute되는 activation O(√n)의 합이다.

Compute cost는 "mini-batch당 추가 forward pass 한 번의 computational cost만" 필요하다([[gradient-checkpointing-chen]]). Layer마다 한 번이 아니라 전체에서 한 번이다. Backward pass는 어차피 gradient를 계산하며, overhead는 segment recomputation이다. 크기가 √n인 segment가 √n개이므로 추가 작업의 총합은 n개 operation, 즉 forward pass 한 번이다. 따라서 overhead는 33%다(일반적인 Transformer에서 forward pass가 전체 forward+backward FLOPs의 약 1/3).

### 실험 결과

[[gradient-checkpointing-chen]]은 ImageNet의 1,000-layer ResNet에서 다음 결과를 검증했다.
- **Checkpointing 없음:** activation memory 48 GB
- **√n checkpointing 사용:** activation memory 7 GB(6.8× 절감)
- **Runtime overhead:** 30%(2× 또는 100%가 아님)

극단적인 variant인 recursive checkpointing은 이 scheme을 재귀적으로 적용하여 O(n log n)의 compute cost로 O(log n) memory를 달성한다. √n 방식의 33% overhead가 이미 수용 가능하고 logarithmic scheme은 구현을 복잡하게 하므로 실무에서는 거의 사용하지 않는다.

### 실무의 Per-layer Checkpointing

현대 framework는 이 scheme을 단순화한다. √n layer마다 checkpoint하는 대신 모든 layer 경계에서 checkpoint한다(layer당 activation 하나를 저장하고 backward 중 각 layer 내부를 recompute). Checkpoint tensor 수는 O(n)이지만 각각은 layer input일 뿐이므로 전체 internal activation 집합보다 훨씬 작다. PyTorch는 `torch.utils.checkpoint.checkpoint()`로, HuggingFace는 `model.gradient_checkpointing_enable()`로 이를 제공한다.

Full gradient checkpointing에서는 activation memory formula가 다음과 같이 축소된다.

```
Full recompute activation memory = 2 * s * b * h * L   (bytes)
```

Recompute가 없을 때는 `s·b·h·L·(34 + 5·a·s/h)`이다. `2sbhL` floor는 checkpoint로 유지해야 하는 layer-input tensor일 뿐이며, 나머지는 필요할 때 recompute한다.

---

## 3. Selective Recomputation — 5× Memory 절감, 4% 미만의 Compute Overhead

[[selective-recompute-korthikanti]](Korthikanti et al., 2022)은 full gradient checkpointing의 비효율을 지적한다. 이 방식은 모든 activation을 대칭적으로 취급하지만 실제 특성은 서로 다르다.

| Activation type | Memory cost | Recompute cost |
|---|---|---|
| Attention score matrix (s×s per head) | High (quadratic in s) | Low (matmul + softmax) |
| MLP intermediate activations | Medium | High (large FFN matmuls) |
| LayerNorm / residual outputs | Low | Medium |

Full recomputation은 비용이 큰 MLP activation까지 모두 버리고 모두 recompute한다. 이는 낭비다. 선택적으로 처리하면 훨씬 낮은 비용으로 되찾을 수 있는 memory를 위해 30–40%의 compute tax(추가 계산 비용) 전체를 지불하기 때문이다.

### 비대칭적 선택

Selective recomputation은 크지만 recompute 비용이 낮은 activation만 "discard" 목록에 두고, 저장 비용이 낮은 activation은 유지한다.

**버렸다가 backward 중 recompute:**
- `[b, a, s, s]` attention score matrix(post-softmax, post-dropout) — 위 formula의 quadratic 항 전체에 해당한다.

**Memory에 유지:**
- MLP activation output(recompute 비용이 높고 element당 크기는 더 작음)
- LayerNorm output
- Linear projection output

Attention score matrix는 가장 큰 tensor(quadratic s²)이지만 Q와 K 사이의 matmul 뒤에 softmax를 적용하면 되므로 reconstruct 비용은 가장 낮다. 따라서 이것이 올바른 비대칭이다. [[ch-04]]에서 소개하는 FlashAttention은 자체 IO-awareness의 일부로 backward pass에서 이미 attention score를 recompute한다. Full-precision attention 문맥의 selective recomputation도 같은 통찰을 활용한다.

### 대규모 결과

[[selective-recompute-korthikanti]]이 NVIDIA A100 GPU 2,240개에서 530B GPT-3-style model로 검증한 결과는 다음과 같다.

- **Activation memory 절감:** full storage 대비 5×
- **Recompute overhead:** full recomputation의 30–40%와 비교해 4% 미만
- **MFU 향상:** full recomputation 대비 42.1% → 54.2%(throughput +29%)

Selective recomputation 이후의 formula:

```
Selective recompute activation memory = s * b * h * L * (10 + 24/t)   (bytes)
```

Quadratic `5·a·s²·b/(h·t)` 항은 완전히 사라진다. 이 항은 이제 필요할 때 recompute하는 attention matrix이기 때문이다. 남는 것은 sequence length에 따라 유리하게 scaling하는 linear 항이다.

---

## 4. Sequence Parallelism — 마지막 복제 제거

Tensor parallelism(TP)은 weight matrix를 여러 GPU에 나누지만 LayerNorm과 Dropout activation은 복제된 채로 둔다. Weight dimension을 따라 직접 shard할 수 없으므로 TP group의 모든 GPU가 이 tensor의 완전한 copy를 보유한다. 긴 sequence에서는 복제된 LayerNorm activation도 무시할 수 없다.

[[selective-recompute-korthikanti]]은 함께 사용할 기법으로 **sequence parallelism**을 소개한다. LayerNorm과 Dropout은 sequence position 사이에서 독립적이므로 TP group 전체에서 sequence dimension을 따라 shard할 수 있다. 결과는 다음과 같다.

```
Activation memory before SP  = s * b * h * (10 + 24/t + 5*a*s/(h*t))   per layer
Activation memory after SP   = (s * b * h / t) * (34 + 5*a*s/h)         per layer
```

Weight와 결합된 부분만이 아니라 activation formula 전체를 t(tensor-parallel degree)로 나눈다. Communication 관점에서는 TP all-reduce를 AllGather + ReduceScatter pair로 교체한다. Bandwidth cost는 동일하지만 activation은 더 이상 복제되지 않는다.

Sequence parallelism은 동일한 GPU group에 tensor parallelism communicator가 이미 존재해야 하므로 새로운 communication topology를 추가하지 않는다. Megatron-LM과 NeMo-Megatron에서는 TP와 함께 사용하는 flag로 제공된다.

---

## 5. 전체 Ledger에서 Activation의 위치

[[ch-01]], [[ch-02]], 그리고 이 chapter를 종합하면 다음과 같다.

```
Total GPU memory = static_floor + activation_memory + logit_spike + overheads

static_floor     = 18 bytes/param  (weights 2B + masters 4B + grads 4B + Adam 8B)
                 [with FP8 on H100: 6 bytes/param; see [[ch-02]]]

activation_memory (no recompute) = L * s * b * h * (10 + 24/t + 5*a*s/(h*t))
activation_memory (selective)    = L * s * b * h * (10 + 24/t)
activation_memory (full recompute) = 2 * L * s * b * h

logit_spike      = B*T * V * dtype_bytes   [eliminated by Liger; see [[ch-02]]]
                   e.g., s=16384, V=32000, BF16 → 1.05 GB
```

Activation은 실무자가 compute와 맞바꾸는 유일한 ledger 항목이다. Static floor는 model size와 precision 선택(FP32 Adam, BF16 working weights)에 의해 고정된다. Logit spike는 kernel 교체(Liger, [[ch-02]])로 제거한다. 근본적인 memory-compute tradeoff가 존재하는 곳은 activation이다.

```
            Memory               Compute overhead
─────────────────────────────────────────────────────
No checkpointing      O(L)                +0%
Full checkpointing    O(√L) → O(1/L)*    +33%
Selective recompute   O(L, -quadratic)    +~4%
─────────────────────────────────────────────────────
* Per-layer ckpt variant: O(L) checkpoints of cheap tensors
```

---

## 6. Activation Memory의 연쇄 효과: Mental Model

Backward pass를 공장 assembly line을 거꾸로 걸어가는 작업자에 비유해 보자. 각 작업대(layer)에서 gradient를 계산하려면 작업자는 forward run 당시 작업대의 정확한 states, 즉 "in-flight" configuration을 알아야 한다. Checkpointing이 없으면 공장은 다음으로 이동하기 전에 모든 작업대를 촬영한다(O(n) storage; 모든 intermediate activation 저장). √n checkpointing에서는 √n번째 작업대마다 하나만 촬영하고, 해당 지점에 도착할 때 각 segment를 다시 forward로 실행하여 intermediate states를 복원한다(O(√n) storage, forward sweep 1회 추가). Selective recomputation에서는 저장 비용이 낮은 작업대만 촬영하고, reconstruct 비용이 높아 보이지만 실제로는 빠르게 다시 측정할 수 있는 값(attention score)은 필요할 때 재측정할 수 있다고 기억한다(즉, 큰 attention matrix만 버리고 recompute한다).

핵심 통찰은 비대칭성이다. 모든 activation의 저장 비용과 recompute 비용이 같지 않다. Selective recomputation은 가장 큰 activation(attention s×s matrix)이 정확히 가장 저렴하게 reconstruct할 수 있는 값이기 때문에 이를 버림으로써 이 비대칭성을 온전히 활용한다.

---

## 문헌에서 얻은 핵심 통찰

**1. Activation에서는 memory와 compute를 서로 교환할 수 있다(Chen 2016).** [[gradient-checkpointing-chen]]은 activation memory가 고정 비용이 아니라 memory/compute Pareto frontier 위의 한 점임을 확립했다. O(√n) 결과는 명확한 이론적 bound를 제시한다. Memory를 절반으로 줄이는 대가는 두 번이 아니라 한 번의 추가 forward pass다. 이 통찰이 매우 깊은 network의 training을 가능하게 한 개념적 돌파구였다.

**2. 버려야 할 올바른 대상은 quadratic attention 항이다(Korthikanti 2022).** [[selective-recompute-korthikanti]]의 selective recomputation은 단순한 engineering optimization을 넘어 올바른 비대칭성을 식별한다. Attention score matrix는 O(s²) scaling 때문에 크지만 matmul + softmax로 reconstruct할 수 있어 비용은 낮다(low arithmetic intensity이므로 빠른 recompute에 적합). MLP activation은 반대로 element당 크기는 더 작지만 큰 intermediate dimension의 dense GEMM이 필요해 비용이 높다. Full checkpointing은 이 비대칭을 무시하지만 selective recomputation은 이를 활용해 4% 미만의 compute cost로 5× memory 절감을 달성한다.

**3. Sequence parallelism은 올바른 activation을 대상으로 해야 한다(Korthikanti 2022).** Activation formula의 10·s·b·h "non-TP" 항(LayerNorm, Dropout)은 tensor parallelism이 나머지를 분할한 뒤에도 남는 항이다. Sequence parallelism은 이 특정 activation을 추가 communication-bandwidth overhead 없이 sequence dimension을 따라 shard하는 표적화된 해결책이다. SP가 없으면 TP만으로는 TP group의 모든 rank에 이 항이 완전히 복제된다.

**4. Long context에서는 activation이 곧 ledger다.** Static floor(18 B/param)는 고정되어 있다. s=4096이면 7B model의 activation memory가 이미 static floor와 비슷하다. s=32768에서는 quadratic attention 항으로 인해 activation이 큰 차이로 지배적인 항이 된다. 실무적으로 어떤 long-context training run이 "memory에 들어가는가"라는 질문은 거의 전적으로 activation strategy, 즉 full recompute, selective recompute, recompute 없음 중 무엇을 선택하는가에 관한 질문이다.

---

## 핵심 정리

- **Activation memory는 O(L·s·b·h) + O(L·a·s²·b/h)로 scaling한다.** 긴 sequence length에서는 quadratic attention 항이 지배하며 large-s training의 주된 OOM 위험이다.
- **Full gradient checkpointing은 약 33%의 추가 compute를 대가로 activation memory를 O(L·s·b·h·34)에서 2·L·s·b·h로 줄인다**(layer input만 저장). Chen 2016의 정리는 일반 network의 최적 조건에서 O(√L) checkpoint를 제시하며, 현대의 per-layer checkpointing은 실용적인 variant다.
- **Chen 2016의 실험 검증:** 1,000-layer ResNet에서 48 GB → 7 GB, runtime overhead 30%.
- **Selective recomputation(Korthikanti 2022)은 4% 미만의 compute overhead로 activation memory를 5× 줄인다.** 크고 recompute 비용이 낮은 attention score matrix만 버리고, 더 작지만 recompute 비용이 높은 MLP/LayerNorm activation은 유지한다.
- **Sequence parallelism**은 tensor parallelism과 함께 LayerNorm/Dropout activation을 TP group 전체에 shard하여 layer 전체에서 실제 t× activation 절감을 달성한다.
- **530B-parameter 검증:** A100 2,240개에서 full recomputation과 비교해 selective recompute + sequence parallelism의 MFU가 42.1% → 54.2%(29% 더 빠름)로 향상되었다.
- Activation은 **compute와 맞바꾸는 유일한 memory ledger 항목**이다. Static floor와 logit spike는 precision 선택과 kernel 교체로 해결하지만 activation에는 algorithmic decision이 필요하다.

---

## 질문

1. Activation memory formula에는 `s·b·h·(10 + 24/t)`(s에 대해 linear)와 `5·a·s²·b/(h·t)`(s에 대해 quadratic)이라는 두 항이 있다. h=4096, a=32, t=1인 model에서 quadratic 항과 linear 항이 같아지는 sequence length는 얼마인가? 이는 checkpointing strategy가 중요해지는 sequence length에 관해 무엇을 의미하는가?

2. [[gradient-checkpointing-chen]]은 compute overhead가 layer당 한 번이 아니라 "mini-batch당 추가 forward pass 한 번"이라고 설명한다. √n scheme을 단계별로 살펴보라. n=64 layer(√n=8 segment)라면 backward pass 중 정확히 몇 번의 layer-forward operation이 발생하며, 왜 그 합이 n√n이 아니라 총 n개의 추가 operation이 되는가?

3. Selective recomputation(Korthikanti 2022)은 post-softmax attention score matrix를 버리고 MLP activation은 유지한다. 이제 [[ch-04]]에서 소개하는 FlashAttention을 고려하라. FlashAttention의 backward pass는 저장된 softmax statistic(logsumexp)으로부터 attention score를 이미 recompute한다. FlashAttention과 selective recomputation을 함께 사용하면 recompute overhead가 이중으로 계산되는가? 이는 memory와 compute accounting에 무엇을 의미하는가?

4. Sequence parallelism은 TP all-reduce를 AllGather + ReduceScatter로 교체한다. 논문은 이에 "추가 communication bandwidth cost가 전혀 없다"고 주장한다. 이 주장을 검증하라. Layer당 all-reduce volume이 V bytes라면 AllGather + ReduceScatter volume은 얼마인가? (t개 rank의 ring all-reduce에서 rank당 bandwidth cost는 2V·(t-1)/t다. t개 rank의 AllGather + ReduceScatter에서 rank당 cost는 얼마인가?)

5. [[gradient-checkpointing-chen]] 발췌문은 O(n log n) compute로 O(log n) memory를 달성하는 극단적인 variant를 언급한다. 이를 달성하는 recursive structure를 설명하라. 이론적으로 O(√n)보다 우수함에도 이 variant가 실무에서 거의 사용되지 않는 이유는 무엇인가?

6. Selective recomputation의 activation formula인 `s·b·h·L·(10 + 24/t)`와 full recomputation의 formula인 `2·s·b·h·L`을 비교하라. 어떤 tensor-parallel degree t에서 selective recomputation이 full recomputation보다 bytes 기준으로 더 저렴해지는가?

---

## 참고자료

- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174. https://arxiv.org/abs/1604.06174 ([[gradient-checkpointing-chen]])
- Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., & Catanzaro, B. (2022). Reducing Activation Recomputation in Large Transformer Models. MLSys 2023. https://arxiv.org/abs/2205.05198 ([[selective-recompute-korthikanti]])
- Anthony, Q. et al. (2023). Transformer Math 101. EleutherAI Blog. https://blog.eleuther.ai/transformer-math/ ([[transformer-math-101]])
- Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR 2018. https://arxiv.org/abs/1710.03740 ([[mixed-precision-training]])
- Shoeybi, M. et al. (2019) + Korthikanti et al. (2022). Megatron-LM (TP + SP). https://arxiv.org/abs/1909.08053 + https://arxiv.org/abs/2205.05198 ([[megatron-tp-sp]])
