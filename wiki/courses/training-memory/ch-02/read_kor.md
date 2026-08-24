<!-- chapter: ch-02
     track: ledger
     title: Optimizer States, Precision, and the Loss-Head Spike
     deps: [[ch-01]]
     sources: [[mixed-precision-training]], [[fp8-training]], [[liger-fused-ce]]
-->

# 2장 — Optimizer States, precision 및 Loss-Head Spike

> **핵심 통찰.** parameter당 12 bytes인 Adam tax(Adam을 사용하기 위해 필수로 추가되는 memory 비용)는 구현상의 우연이 아니라 fp16의 5-bit exponent가 직접 초래한 결과입니다. fp32 master weights가 없으면 각 step에서 gradient update의 약 5%가 underflow로 0이 되고, model은 뚜렷한 오류 없이 서서히 발산합니다. BF16는 exponent를 8 bits(fp32와 동일)로 늘려 dynamic-range trap(표현 가능 범위가 좁아 값이 overflow/underflow되는 문제)을 피하지만, optimizer accumulation에는 fp32 numerical fidelity가 여전히 필요하므로 bf16 mixed precision에서도 fp32 master copy는 필수입니다. loss-head logit tensor는 이 static floor와 완전히 별개로 발생하는 transient spike(짧은 순간에 memory 사용량이 급증하는 peak)이며, 긴 sequence length와 큰 vocabulary에서는 static floor를 큰 폭으로 초과할 수 있습니다.

> **지침.** mixed-precision AdamW의 static training floor로 parameter당 18 bytes(2-byte bf16 working weights + 4-byte fp32 master weights + 4-byte fp32 gradients + 8-byte fp32 optimizer states)를 책정하십시오. A100 및 그 이전 hardware에서는 이 구성을 사용하십시오. H100+에서는 FP8 training(Transformer Engine, E4M3/E5M2)이 GPT-175B급 model의 해당 floor를 약 39% 줄이지만 Hopper tensor cores가 필요하므로, A100에서는 FP8를 시도하지 마십시오. transient `B·T×V` logit spike를 제거하려면 fused/chunked cross-entropy kernel(Liger)을 항상 사용하십시오. 이는 large-vocabulary training에서 한 step 동안 발생하는 가장 큰 단일 memory event이며 accuracy 손실도 없습니다.

---

## 1. Adam의 12바이트/Param: 수학이 이를 강제하는 이유

[[ch-01]]의 ledger는 Adam optimizer bucket에 12 bytes/parameter를 배정했다. 다음은 정확한 accounting과 mixed precision에서 각 byte가 필수인 이유다.

**혼합 precision AdamW의 3tensor 레이아웃:**

| tensor | Dtype | 바이트/parameter |
|--------|-------|-------------|
| Working weights(compute 복사) | bf16 / fp16 | 2 |
| fp32 master weights(optimizer 복사) | fp32 | 4 |
| fp32 gradients (누적) | fp32 | 4 |
| 1차 moment m(momentum) | fp32 | 4 |
| Second-order moment v (variance) | fp32 | 4 |
| **Static floor 합계** | | **18** |

[[ch-01]]의 "Adam은 12 B/param"이라는 수치는 optimizer에만 존재하는 fp32 master weights(4) + m(4) + v(4) = 12를 뜻한다. Working-weight copy(2)와 gradient copy(4)가 나머지 6 bytes를 차지하여 합계가 18이 된다. [[ultrascale-playbook]]과 [[transformer-math-101]]도 이 18 B/parameter floor를 제시하며, fp32 gradient accumulation을 적용하면 20 B/parameter가 된다고 설명한다.

**fp32 master weights가 필수인 이유 - 두 가지 실패 모드([[mixed-precision-training]]):**

1. **Gradient underflow.** "Magnitude가 2^−24보다 작은 값은 FP16에서 0이 된다." 실험적으로 대표 model의 weight-gradient magnitude 중 약 5%가 매 step 이 threshold 아래에 놓인다. 해당 gradient는 아무 경고 없이 0이 되고 그 parameter는 학습을 멈춘다. 이 현상이 step 전체에 누적되면 model이 발산한다.

2. **Mantissa cancellation(weight update rule).** Adam은 `w ← w − lr · m̂ / (√v̂ + ε)`를 적용한다. `|w| / |update| > 2048`이면 fp16의 10-bit mantissa로는 그 차이를 표현할 수 없다. `w + Δw`가 다시 정확히 `w`로 round되어 update가 사라진다. fp32 master weights에서는 23-bit mantissa로 accumulate하므로 이런 cancellation이 발생하지 않는다.

이를 무시한 실험적 대가는 크다. 논문의 Mandarin speech model은 fp32 master를 사용한 경우와 비교해 fp16-only weights로 training했을 때 "80% relative accuracy loss"를 보였다. fp32 master는 보수적인 engineering 선택이 아니라 training의 numerical stability를 유지하는 최소 precision이다.

---

## 2. BF16 대 FP16: Exponent가 변수입니다

BF16가 FP16를 기본 training dtype로 대체한 이유를 이해하려면 각 format의 장단점이 무엇인지 확인해야 합니다.

**format 비교:**

| Format | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| FP32 | 1 | 8비트 | 23비트 | ±3.4×10³⁸ | ~7자리 |
| BF16 | 1 | 8비트 | 7비트 | ±3.4×10³⁸ | 10진수 2자리 |
| FP16 | 1 | 5비트 | 10비트 | ±65,504 | 10진수 ~3자리 |

BF16는 잘린 fp32입니다. 동일한 exponent 너비(8비트), 더 적은 mantissa 비트(7 대 23)입니다. FP16는 ​​fp32와 비교하여 exponent 비트를 mantissa 비트로 교환합니다.

**실질적인 결과** fp16의 5비트 exponent는 representable range를 ±65,504로 제한하는 반면, BF16의 8비트 exponent는 fp32(±3.4×10³⁸)와 동일한 range를 커버합니다. Training 중 gradient magnitude는 여러 order of magnitude에 걸쳐 분포하며 fp16의 range를 빈번하게 벗어난다. BF16에는 fp32와 동일한 exponent가 있으므로 dynamic range 문제가 발생하지 않습니다. fp32 master weights는 optimizer 누적 충실도를 위해 계속 유지되지만 fp16의 별도 loss scaling 메커니즘은 제거됩니다.

**FP16용 Loss scaling([[mixed-precision-training]]):**

개입하지 않으면 backward 중 FP16 gradient가 overflow 또는 underflow하므로 표준 해결책은 loss scaling이다.

```
# Forward pass
loss_scaled = loss * scale_factor   # e.g. scale_factor = 8 to 32768

# Backward pass (chain rule scales all gradients by scale_factor)
loss_scaled.backward()

# Before optimizer step: unscale and clip
for param in model.parameters():
    param.grad /= scale_factor
clip_grad_norm_(model.parameters(), max_norm)

optimizer.step()
```

Scale factor는 gradient histogram 전체를 magnitude가 큰 쪽으로 이동시켜 FP16에서 0이 될 underflow 값을 representable range 안으로 가져온다. Chain rule에 따라 이는 loss를 scaling하는 것과 수학적으로 동일하므로 computation graph를 다시 작성할 필요가 없다. Backward 이후 clipping과 optimizer step 전에 같은 factor로 나눈다. 이것이 없으면 SSD 객체 감지의 경우: "gradient 값의 67%는 scaling이 없는 FP16의 zero입니다. 8배 scaling을 사용하면 training이 fp32 accuracy와 일치합니다." bigLSTM에는 128배 scaling이 필요했습니다. Dynamic loss scaling은 training 중 factor를 조정한다. Gradient에서 NaN/Inf overflow가 감지되면 scale을 절반으로 줄이고, N step 동안 overflow가 없으면 두 배로 늘린다.

**현대의 default는 BF16 + fp32 master이며 loss scaling은 사용하지 않는다**([[mixed-precision-training]]). BF16는 fp32와 동일한 8-bit exponent width를 사용하므로 FP16의 5-bit exponent에서 발생하는 dynamic-range 문제를 제거한다. 따라서 이 논문의 loss-scaling 기법은 BF16에서는 덜 중요하다.

---

## 3. Hopper의 FP8: 18 B/Param floor 압축

FP8 training은 H100/Hopper hardware를 대상으로 하는 precision의 다음 step입니다. memory 인수는 간단합니다. weights와 gradients를 모두 1바이트 format으로 저장하고 compute할 수 있으면 static floor가 극적으로 줄어듭니다.

**두 가지 FP8 format([[fp8-training]]):**

- **E4M3**: sign 1개 + exponent 4개 + mantissa 비트 3개; range ±448; 더 높은 precision. **weights 및 activations**(forward pass)에 사용됩니다. range가 아닌 precision가 필요하기 때문입니다.
- **E5M2**: sign 1개 + exponent 5개 + mantissa 비트 2개; range ±57,344; 더 넓은 range. gradients는 overflow 없이 넓은 크기 스펙트럼을 나타내야 하므로 **gradients** (backward pass)에 사용됩니다.

비대칭 allocation은 의도적인 것입니다. weights는 수치적으로 안정적이고 해당 값은 제한되어 있으므로 precision가 range(E4M3)보다 중요합니다. Gradients는 layer와 step에 따라 크기가 크게 달라지므로 precision보다 range가 더 중요합니다(E5M2).

**FP8 최적화 memory 레이아웃(총 6B/parameter vs BF16의 경우 18B/parameter)([[fp8-training]]):**

| tensor | FP8 레이아웃 | 바이트/parameter |
|--------|------------|-------------|
| master weights | scaling 기능이 있는 FP16 | 2 |
| Gradients | FP8 (E5M2) | 1 |
| 1차 moment m | FP8 | 1 |
| Second-order moment v | FP16 | 2 |
| **토탈 optimizer** | | **6** |

BF16 AdamW의 18B/param과 비교: 이는 **2.6× optimizer states 감소**입니다(optimizer 단독에서만 ~16B에서 ~6B로, 소스마다 경계를 다르게 compute함).

**Per-tensor dynamic scaling**은 FP8의 좁은 range를 처리합니다. 각 tensor는 동적 scale factor μ를 얻습니다. overflow 비율이 0.001%를 초과하는 경우: μ → μ/2. 1,000개의 training step에 대해 overflow가 없는 경우: μ → μ×2. GPU 전체의 distributed all-reduce의 경우 global minimum scale `s_g' = min(s'₁, ..., s'_n)`가 사용됩니다. 이는 규모에 따라 지배적인 tensor별 동기화 overhead를 제거합니다.

**전체 memory reduction 대 BF16 mixed precision([[fp8-training]]):**

| model | Memory reduction |
|-------|-----------------|
| GPT-7B | 29% |
| GPT-13B | 28% |
| GPT-175B | **39%** |

175B에서 절감 폭이 더 큰 이유는 parameter count가 커질수록 sequence/batch에 의해 정해지는 activation에 비해 optimizer states가 memory budget을 더 강하게 지배하기 때문이다.

**H100에서의 throughput 이점([[fp8-training]]):** GPT-175B는 BF16 Megatron-LM보다 **75% 더 빠르게**, NVIDIA Transformer Engine의 자체 구현보다 37% 더 빠르게 실행됩니다. GPT-7B: 38% 더 빠릅니다. FP8는 또한 더 긴 sequence를 가능하게 합니다. GPT-175B는 FP8에서 seq=4,096으로 training할 수 있으며, 여기서 BF16는 동일한 H100 cluster에서 seq=2,048로 제한됩니다.

**A100이 FP8을 사용할 수 없는 이유.** FP8 tensor core는 Hopper architecture의 기능이며 Ampere(A100)에는 없는 hardware-accelerated 8-bit matrix-multiplication unit에 직접 mapping된다. FP8은 구형 hardware에서 단순히 성능만 낮아지는 software trick이 아니다. 전용 tensor core가 없으면 compute 이점은 사라지고 precision loss만 남는다. A100에서는 BF16 + fp32 master를 사용하라.

---

## 4. Loss-Head Logit Spike

Static 18 B/parameter floor는 weights, gradients, optimizer states를 포함한다. 그러나 각 forward pass 끝에는 이를 압도할 수 있는 transient peak인 logit tensor가 나타난다.

**출처.** language model의 마지막 layer는 hidden dimension h에서 vocabulary 크기 V까지의 linear projection입니다.

```
# Standard PyTorch pattern (what NOT to do at scale):
logits = hidden_states @ lm_head.weight.T   # shape: [B, T, V]
loss = F.cross_entropy(
    logits.view(B*T, V),                     # must exist in full
    labels.view(B*T)
)
```

Logit tensor에는 `B × T × V`개의 element가 있다. Element당 fp32는 4 bytes, bf16은 2 bytes다. Cross-entropy loss를 계산하기 전에 이 tensor 전체가 materialize된다.

**spike의 크기([[liger-fused-ce]]):**

BF16에서 seq=16,384 토큰 및 vocab=32,000인 경우:
```
16,384 × 32,000 × 2 bytes = 1.05 GB
```

이 1GB 이상의 tensor는 모든 forward pass에서 일시적으로 나타납니다. 즉, linear projection과 loss reduction 사이에만 존재하므로 "spike"처럼 갑자기 증가했다가 해제됩니다(짧은 시간 동안 memory 사용량의 최고점을 만든다는 뜻입니다). 그러나 forward-pass activations와 logit tensor가 공존하는 step 경계에서는 이것이 최대 memory event입니다. 이는 18B/param static floor의 **완전히 외부**에 위치하며, parameter-level 상수가 아니라 activation-level transient입니다. [[ch-09]] capstone 사례(seq=32k, vocab=248k)는 상황을 극적으로 악화시킵니다.

```
32,768 × 248,000 × 2 bytes = ~16 GB
```

16GB의 단일 logit tensor는 다른 activations를 compute하기 전에 80GB H100에서 지배적인 memory 소비자가 됩니다.

**OOM failure mode.** Memory profiler에서는 step 0을 통과한 job이 step 1이나 2에서 OOM이 되는 모습을 볼 수 있다. Logit spike와 optimizer-state materialization([[ultrascale-playbook]]의 "peak memory warning")이 maximum allocation 시점에 겹치기 때문이다. 계산상 4 GB의 headroom이 있어 보이는 job도 이 transient 때문에 OOM이 될 수 있다.

**모니터링 신호: grad_norm 및 NaN.** 정밀 문제 — fp16 overflow, fp8 스케일 구성 오류 또는 loss scaling 버그 — 먼저 NaN 또는 Inf gradients로 나타납니다. 표준 monitoring loop:

```python
# After loss.backward() and before optimizer.step():
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0
)
if torch.isnan(grad_norm) or torch.isinf(grad_norm):
    # Skip optimizer.step(); reduce loss scale (fp16) or investigate
    scaler.update()   # GradScaler will halve the scale
    continue
```

`grad_norm`의 NaN는 가장 먼저 관찰 가능한 증상입니다. loss divergence를 기다리는 것은 낭비되는 step가 많다는 것을 의미합니다.

---

## 5. Liger-Kernel: Fused Chunked CE를 통해 spike 제거

Liger의 fused cross-entropy kernel는 linear projection를 loss compute 및 청크 단위([[liger-fused-ce]]) 처리 토큰과 융합하여 materialization 문제를 해결합니다.

**세 가지 kernel 내 전략:**

```
# Pseudocode for LigerFusedLinearCrossEntropyLoss (Triton kernel)

for chunk in split(hidden_states, chunk_size=65536):   # ≤65,536 tokens/chunk
    # 1. Incremental projection: project only this chunk
    logit_chunk = chunk @ lm_head.weight.T             # shape: [chunk, V]
    
    # 2. Compute loss contribution from this chunk
    loss_chunk = cross_entropy(logit_chunk, labels[chunk])
    loss_accum += loss_chunk
    
    # 3. In-place gradient accumulation:
    #    grad_weight accumulates across chunks, no full logit buffer needed
    grad_weight += chunk.T @ d_logit_chunk
    d_hidden[chunk] = d_logit_chunk @ lm_head.weight

# Never materializes the full [B*T, V] logit tensor
```

CUDA의 플랫폼별 청크 크기는 일반적으로 청크당 `65,536 ÷ 2 = 32,768` 토큰입니다. 이전 예(seq=16,384, vocab=32,000)의 경우:

```
Peak per-chunk allocation: 2,048 × 32,000 × 2 bytes = 131 MB
```

전체 materialization의 경우 1,050MB — 이 한 번의 작업으로 최대 과도 allocation이 **8배 감소**됩니다.

**보고된 memory 감소([[liger-fused-ce]]):**

| training 모드 | 전체 memory reduction |
|---------------|--------------------------|
| Pretraining / SFT | ~60% |
| Alignment(DPO, ORPO, CPO) | 최대 80% |

Alignment에서 절감 폭이 더 큰 이유는 DPO와 ORPO가 같은 vocabulary head에 대해 chosen sequence와 rejected sequence의 forward pass를 각각 실행하기 때문이다. Vanilla implementation에서는 spike가 두 배지만 Liger는 양쪽 모두 chunk로 처리한다.

**Exact semantics 보장:** approximation은 사용하지 않는다. Chunked accumulation은 전체 logit matrix에 대한 CE 계산과 수학적으로 동일하다. Backward pass는 Triton kernel 내부에 통합되어 hidden states와 `lm_head.weight` 모두에 올바른 gradient를 반환한다. `nn.Linear + CrossEntropyLoss`를 `LigerFusedLinearCrossEntropyLoss`로 교체하는 것은 accuracy에 영향을 주지 않는 drop-in substitution이다.

**API:**

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

# Replace the standard PyTorch combination:
# loss_fn = nn.CrossEntropyLoss()
# logits = lm_head(hidden_states)
# loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

# With:
loss_fn = LigerFusedLinearCrossEntropyLoss()
loss = loss_fn(lm_head.weight, hidden_states, labels)
# lm_head bias optional; backward() works normally
```

모든 training run의 시작부터 이를 적용하라. Large vocabulary를 사용하는 memory-constrained training에서 가장 효과가 큰 단일 operator 교체다.

---

## 6. 조각들이 어떻게 맞춰지는가: 완전 정밀 스택

training 루프의 각 지점에서 precision 선택에 대한 전체 그림:

```
┌─────────────────────────────────────────────────────────────────┐
│  FORWARD PASS                                                    │
│  Input tokens → Embedding (bf16) → Transformer layers (bf16)   │
│  → Hidden states (bf16) → [Liger fused CE, chunked]            │
│                            ↑ no full logit tensor               │
│  Loss scalar (fp32) ←──────┘                                    │
├─────────────────────────────────────────────────────────────────┤
│  BACKWARD PASS                                                   │
│  Gradients in bf16 (or fp16+scaling, or fp8 E5M2)             │
│  Accumulated into fp32 gradient buffers                         │
├─────────────────────────────────────────────────────────────────┤
│  OPTIMIZER STEP (Adam)                                          │
│  fp32 gradients → update fp32 m, fp32 v                        │
│  → compute Δw in fp32                                           │
│  → apply to fp32 master weights                                 │
│  → cast fp32 masters back to bf16 working weights              │
└─────────────────────────────────────────────────────────────────┘
```

fp32 master weights는 이 마지막 step에 사용되며, 이 때문에 optimizer compute에 필요한 memory 외에 4 bytes/parameter가 추가로 필요하다. 다음 forward pass를 위해서만 bf16으로 cast한다.

---

## 7. 교차 precision 비교표

| 측면 | FP16 혼합 정밀 | BF16 혼합 정밀 | FP8(H100 전용) |
|--------|-----------------|-----------------|-----------------|
| working weights dtype | fp16(2B) | bf16(2B) | fp8 E4M3 (1B) |
| master weights dtype | fp32(4B) | fp32(4B) | fp16(2B) |
| Gradient dtype | fp16(확장) | bf16 | fp8 E5M2 (1B) |
| 최적화 순간 dtype | fp32+fp32(8B) | fp32+fp32(8B) | fp8+fp16(~3B) |
| Static floor(B/parameter) | ~18 | ~18 | ~6 |
| Loss scaling 필요 | 예(동적) | 아니요 | tensor별 μ |
| hardware 요구사항 | 모든 CUDA | 모든 CUDA | H100+ (Hopper) |
| Throughput 이득 대 fp32 | ~2× | ~2× | bf16 대비 +75% |
| Memory reduction 대 fp32 | ~50% | ~50% | ~89% |

**세 가지 모두에 걸쳐 불변**: weights의 fp32(또는 동등한 precision) copy이 optimizer update 경로 어딘가에 있습니다. 이는 numerical representation이 강제하는 조건이다. update `w ← w − lr·m̂/(√v̂+ε)`는 forward pass가 사용하는 precision에 관계없이 mantissa cancellation를 피하기 위해 충분한 precision로 compute되어야 합니다. FP8 training도 fp8가 아닌 fp16 master weights(4B 대신 2B)를 유지합니다.

**변형**: gradients 및 moment에 사용되는 precision와 dynamic range 문제가 해결되는 방법(fp16의 dynamic loss scaling, bf16에서 제거, fp8의 tensor당 μ). 이는 hardware 기능을 고려할 때 자유로운 디자인 선택입니다.

---

## 문헌의 핵심 통찰력

**[[mixed-precision-training]](Micikevicius et al., ICLR 2018)에서:** 두 가지 뚜렷한 수치 현상 때문에 fp32 master weights가 필요합니다. fp16의 floor인 2^−24 아래에 있는 gradient underflow와 weights 대 update 비율이 2,048을 초과할 때 mantissa cancellation입니다. 둘 다 step에 걸쳐 복합적인 자동 zero update를 유발합니다. loss scaling는 underflow 케이스를 수정하지만 취소 케이스는 수정하지 않으므로 scaling을 적용하더라도 master weights가 여전히 필요합니다.

**[[fp8-training]](Peng et al., 2023)에서:** 비대칭 FP8 format allocation(forward의 경우 E4M3, backward의 경우 E5M2)은 weights/activations 대 gradients의 다양한 수치 요구 사항을 반영합니다. weights는 제한되어 있으며 mantissa precision가 필요합니다. gradients는 수십 배에 걸쳐 있으며 exponent range가 필요합니다. 39% memory reduction에서 GPT-175B의 75% throughput 이득은 컴퓨팅 가속(8비트 matmuls)과 해제된 대역폭(HBM의 더 작은 tensor)의 복합 효과를 나타냅니다.

**[[liger-fused-ce]](Liu et al., 2024)에서:** B·T×V logit materialization은 static memory floor와 독립적으로 발생하는 병리학적 spike입니다(다른 memory 항목의 크기와 무관하게 별도의 peak를 만든다는 뜻입니다). 다른 항목은 충분히 들어가는 model도 이로 인해 OOM이 될 수 있습니다. Fused chunked kernel은 exact computation(근사 없음)을 수행하는 drop-in replacement이며, 일반적인 구성에서 이 spike를 1GB 이상에서 131MB로 줄입니다. alignment training에서 80%가 절감되는 이유는 DPO/ORPO가 동일한 head를 통해 두 sequence를 전달하여 spike를 두 배로 늘리기 때문입니다.

**[[transformer-math-101]] 및 [[ultrascale-playbook]]에서:** 18B/parameter floor(또는 bf16 gradients를 사용하는 playbook의 accounting에서 16B)은 시작 추정치이지만 activations formula `L·seq·bs·h·(34 + 5·n_heads·seq/h)`는 sequence length에서 quadratic적으로 증가하고 긴 컨텍스트에서 static floor를 10-100×만큼 초과할 수 있습니다. logit spike는 activations 위에 추가되는 transient이므로 step당 최악의 최대 memory 이벤트가 됩니다.

---

## 주요 시사점

- **18바이트/parameter**는 정적 혼합 precision AdamW 층(2 bf16 working weights + 4 fp32 마스터 + 4 fp32 gradients + 8 fp32 optimizer)입니다. [[ch-01]]의 "Rule of 16"는 2B bf16 gradients를 사용하여 16B를 얻습니다. 18 B는 fp32 gradients를 사용한 보수적인 floor입니다.
- **fp32보다 낮은 precision으로 training할 때 fp32 master weights는 필수다.** Adam의 subtraction에서 mantissa cancellation이 update를 아무 경고 없이 0으로 만들 수 있으므로 수치 표현 자체가 이를 강제한다.
- **BF16 대 FP16 축**: BF16는 loss scaling를 제거합니다(fp32와 동일한 exponent range). FP16에는 dynamic loss scaling이 필요하며, 사용하지 않으면 training이 발산할 수 있다. 최신 hardware(A100+)에서는 BF16가 기본값입니다.
- **FP8는 ​​H100 전용**: GPT-175B의 경우 39% memory reduction 및 75% throughput 게인이지만 Hopper tensor core가 있어야만 사용할 수 있다. A100의 경우: BF16를 사용합니다.
- **logit spike**(`B·T×V × dtype_size`)는 training step에서 가장 큰 transient이며 static floor 외부에 위치합니다. BF16의 seq=32k, vocab=248k에서는 최대 16GB로 30B 미만인 대부분의 model parameter memory보다 큽니다. Liger의 fused CE를 사용해 chunking하십시오.
- **모니터링**: `grad_norm`의 NaN/Inf는 정밀 실패(fp16 스케일 구성 오류, fp8 μ 폭주 또는 수치 불안정)에 대해 처음으로 관찰할 수 있는 신호입니다. 모든 step를 모니터링하세요.
- **다음 장** [[ch-03]]는 activation memory 및 gradient checkpointing를 다룹니다. 이는 quadratic sequence 길이 항이 지배적인 memory 방정식의 다른 측면입니다.

---

## 참고자료

- Paulius Micikevicius et al. "Mixed Precision Training." ICLR 2018. https://arxiv.org/abs/1710.03740 ([[mixed-precision-training]])
- Houwen Peng et al. "FP8-LM: Training FP8 Large Language Models." arXiv:2310.18313, 2023. https://arxiv.org/abs/2310.18313 ([[fp8-training]])
- Austin Liu et al. "Liger Kernel: Efficient Triton Kernels for LLM Training." arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel ([[liger-fused-ce]])
- Quentin Anthony et al. "Transformer Math 101." EleutherAI Blog, 2023. https://blog.eleuther.ai/transformer-math/ ([[transformer-math-101]])
- HuggingFace / nanotron team. "The Ultra-Scale Playbook." 2025. https://nanotron-ultrascale-playbook.static.hf.space/ ([[ultrascale-playbook]])

---

## 질문

1. **Mantissa cancellation:** fp32 master weights는 `|w| / |Δw| > 2048`일 때 mantissa cancellation를 방지합니다. Adam이 `√v̂`로 update를 정규화(거의 단위 규모로 만듦)한 경우 `|w| / |Δw|` 비율이 2048을 초과할 가능성이 가장 높은 training 조건(초기 training, 늦은 training 또는 전혀 없음)은 무엇입니까? 숫자를 살펴보세요.

2. **Exponent 산술:** FP16에는 5비트 exponent가 있습니다(편향 표현: range 2^−14 ~ 2^15, 즉 ±65,504). BF16에는 8비트 exponent가 있습니다(fp32와 동일: range 2^−126 ~ 2^127). FP16에 대한 underflow 임계값은 2^−24(비정상 floor)입니다. gradient 크기에서 fp16는 bf16에 대한 정보를 잃기 시작하며 이것이 2^−14가 아닌 underflow 임계값인 이유는 무엇입니까?

3. **Loss scaling 파생:** gradient 히스토그램 증거에서 "SSD gradient 값의 67%는 scaling이 없는 fp16의 zero입니다. 8× scaling은 전체 accuracy를 복구합니다." 8× scaling이 적용되면 표현 가능한 가장 작은 fp16 gradient는 ~2^−24에서 ~2^−24 × 8 = 2^−21로 이동합니다. 이는 zero에 가까운 SSD gradients의 대부분이 exponent range에 집중되어 있음을 알려줍니다.

4. **FP8 format allocation:** 논문에서는 E4M3(고precision, 좁은 range)를 weights/activations에 allocation하고 E5M2(넓은 range, 낮은 precision)를 gradients에 allocation합니다. [[liger-fused-ce]] 발췌문에 따르면 logit tensor spike는 BF16의 seq=16k, vocab=32k에서 1.05GB에 도달합니다. Transformer Engine가 activations에 대해 E4M3를 사용하는 경우(1바이트 대 2바이트 bf16), logit spike는 FP8 forward pass에서 무엇이 되며, 이로 인해 Liger에 대한 인수가 변경됩니까?

5. **청킹 산술:** Liger kernel은 CUDA에서 `chunk_size ≤ 65,536 ÷ 2 = 32,768` 토큰을 사용합니다. [[ch-09]] 캡스톤 사례(seq=32,768, vocab=248,000, BF16)의 경우 (a) 전체 logit tensor 크기, (b) Chunk_size=32,768의 청크당 피크 allocation 및 (c) memory reduction element를 compute합니다. Liger 없이 단일 A100-80GB가 전체 spike를 처리할 수 있습니까?

6. **[[fp8-training]] 발췌:** 39% memory reduction 및 75% throughput 이득에서 GPT-175B. 530B의 Selective recomputation + sequence parallelism(Korthikanti 2022, [[ch-03]]에서 다루어짐)는 42.1%에서 54.2% MFU로 29% throughput 이득을 제공합니다. 이것은 H100의 추가 기술입니다. 둘 다 H100의 175B model에 동시에 적용되는 경우 최대 memory에 대한 예상 결합 효과는 무엇이며 어떤 제약 조건(static floor 대 activations spike 대 logit spike)이 구속력이 됩니까?
