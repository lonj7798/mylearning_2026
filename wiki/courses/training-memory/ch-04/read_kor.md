<!-- chapter: ch-04
     track: attention
     kind: content
     title: Attention Is a Memory Problem: O(N²) and Why the Kernel Decides
     deps: [[ch-03]]
     sources: [[self-attention-no-n2-memory]], [[online-softmax]]
-->

# 4장 — Attention은 Memory 문제다: O(N²), 그리고 Kernel이 결정하는 이유

> **선수 chapter.** 이 chapter는 attention의 memory 비용을 분석하므로 mechanism 자체는 이미 안다고 가정합니다.
> Q/K/V, head split, causal mask, RoPE, transformer block의 tensor ledger가 아직 확실하지 않다면
> [[ch-extra]] — [Attention과 Transformer, 처음부터](../ch-extra/read.md) — 를 먼저 읽으세요. [[ch-03]]과 이 chapter 사이에 위치합니다.

> **핵심 통찰력.** 표준 multi-head attention은 N×N score matrix와 해당 softmax를 activation로 materialize합니다. layer당 head당 O(N²) 바이트는 sequence 길이에 따라 2차적으로 증가하고 long context에서 주요 제한 요소로 FLOP가 아닌 GPU memory를 지배합니다. Rabe & Staats 2021는 이것이 수학적으로 필요하지 않다는 것을 증명합니다. 전체 matrix을 작성하지 않고 실행 중인 softmax normalizer로 key를 스트리밍하여 query당 O(1) 추가 memory에서 exact self-attention을 계산할 수 있습니다. model architecture가 아닌 attention *kernel*의 선택에 따라 training 비용이 layer당 O(N²) 또는 O(N) activation memory인지 여부가 결정됩니다.

> **가이드라인.** training job이 긴 sequence length에서 OOM되면 가장 먼저 조정할 수단(문제를 해결하기 위해 우선 변경해야 하는 설정)은 gradient checkpointing이나 batch size가 아니라 attention kernel입니다. PyTorch SDPA가 MATH backend(전체 N×N matrix를 할당함)로 조용히 fallback하지 않았는지 확인하세요. fallback했다면 FlashAttention 또는 xFormers를 강제하세요. online-softmax recurrence(Milakov & Gimelshein 2018)은 O(N)-memory tiling을 numerically stable하게 만드는 수학적 primitive입니다. 이를 이해하면 kernel이 왜 안전하게 tiling할 수 있는지, 그리고 언제 그렇게 할 수 없는지를 정확히 알 수 있습니다.

---

## 1. Standard Attention: O(N²) memory 예산

### 1.1 바닐라 forward pass, 단계별

query matrix Q ∈ ℝ^(N×d), key matrix K ∈ ℝ^(N×d) 및 value matrix V ∈ ℝ^(N×d)을 사용하는 단일 head에 대한 multi-head self-attention은 다음을 계산합니다.

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

실행 순서로 설명:

```
Step 1.  S = Q Kᵀ           # shape: N×N  (the score matrix)
Step 2.  S_scaled = S / √d   # elementwise divide; same shape
Step 3.  P = softmax(S_scaled, dim=-1)  # row-wise softmax; shape N×N
Step 4.  O = P · V           # shape: N×d  (the output)
```

1-3 단계는 memory에 동시에 유지되어야 하는 N×N 모양의 두 개의 tensor, 즉 ​​`S_scaled`(softmax를 통한 backward pass용)와 `P`(단계 4의 matmul용)를 생성합니다. bf16(요소당 2 bytes)에서 하나의 head에 대한 하나의 N×N activation tensor 비용은 다음과 같습니다.

```
bytes = N² × 2
```

N = 32,768(32,000개 token)의 경우 head당 32768² × 2 = **2,147,483,648 bytes ≒ 2 GB**입니다.

layer당 head가 H개이고 layer가 L개인 경우 attention scores에서 얻은 총 activation footprint은 다음과 같습니다.

```
attention_activations = 2 × H × L × N² × dtype_bytes
                        ↑ stores both S_scaled and P
```

N=32k, bf16에서 70B 모델(L=80, H=64)의 경우:

```
= 2 × 64 × 80 × (32768²) × 2 ≈ 549 TB
```

그 수치는 터무니없이 크며, 바로 그것이 핵심입니다. 긴 sequence에서 Standard attention은 어떤 GPU에도 들어가지 않으며 어떤 training cluster에도 들어가지 않습니다. N² 항은 단순한 계수 차이가 아니라 qualitative regime change입니다(입력 규모가 커질 때 memory 요구량의 성장 양상 자체가 달라지는 근본적 변화입니다).

### 1.2 FLOPs가 아니라 Memory가 한계인 이유

표준 설명은 O(N²) *compute*에 중점을 둡니다. 그러나 [[self-attention-no-n2-memory]]는 더 날카로운 주장을 합니다.

> "compute 기능보다는 device memory가 최신 가속기의 제한 요소인 경우가 많습니다."

Standard attention의 FLOPs도 O(N²)이지만, 최신 GPU는 compute throughput이 충분하므로 실제 병목은 *memory bandwidth*(HBM에서 N×N matrix를 읽고 쓰는 속도)와 *memory capacity*(backward pass을 위해 저장해야 하는 activation tensor를 수용하는 용량)입니다. activation을 recompute하면 더 적은 memory로 더 긴 sequence를 training할 수 있습니다. 하지만 Standard attention에서는 *recomputation* 자체도 N×N matrix를 다시 materialize해야 하므로, algorithm을 바꾸지 않는 한 memory는 O(N²) 아래로 내려가지 않습니다.

이는 나머지 attention 트랙을 설정하는 논거입니다. 근본적인 문제는 N×N materialized tensor의 존재이며, 이를 수정하려면 softmax 계산 *방법*에 대한 algorithm 변경이 필요합니다.

---

## 2. online softmax: 수학적 열쇠

### 2.1 클래식 Softmax에는 세 번의 패스가 필요합니다.

길이가 N인 점수 행 x = [x₁, x2, ..., xₙ]가 주어지면 표준 numerically stable softmax는 다음과 같이 진행됩니다.

```
Pass 1:  m = max(x₁, ..., xₙ)            # find global max for stability
Pass 2:  d = Σᵢ exp(xᵢ − m)              # compute partition function
Pass 3:  pᵢ = exp(xᵢ − m) / d  for all i  # normalize
```

각 패스는 memory에서 전체 행을 읽습니다. 이는 전체 행을 storage해야 함을 의미합니다. 이는 Attention 점수의 경우 Attention 계산 중에 전체 N×N matrix 행이 HBM에 상주해야 함을 의미합니다. [[online-softmax]] 문서에 따르면: "3단계 클래식 softmax: (1) 모든 x를 읽어 최대 m을 찾습니다. (2) 모든 x를 다시 읽어 합계 d를 계산합니다. (3) 모든 x를 세 번째 읽어 출력합니다." 이는 softmax에 대한 전체 점수 vector에 대한 3× memory bandwidth입니다.

### 2.2 원패스 온라인 recurrence

Milakov & Gimelshein 2018([[online-softmax]])는 단일 스트리밍 패스에서 numerically stable softmax를 계산하는 recurrence을 도출하며, 두 개의 scalar(running maximum m과 실행 합계 d)만 유지합니다.

```
Initialize: m₀ = -∞,  d₀ = 0

For each element xₖ  (k = 1 … N):
    m_new = max(m_old, xₖ)
    d_new = exp(m_old − m_new) · d_old + exp(xₖ − m_new)
    m_old ← m_new,  d_old ← d_new

Final output:  pᵢ = exp(xᵢ − m_final) / d_final  for all i
```

재조정 항 `exp(m_old − m_new)`가 핵심입니다. m_new > m_old(즉, 더 큰 요소를 찾았을 때) 이전에 누적된 모든 기여는 추가 storage 비용 없이 최대 빼기 트릭과 수치적 동등성을 유지하면서 정확히 올바른 요소로 소급하여 하향 재조정됩니다. 각 단계에서 m과 d만 register에 있어야 합니다.

측정된 효과는 상당합니다. [[online-softmax]]는 "Softmax 단독으로 최대 1.3×까지 가속화되고, fused Softmax+TopK는 GPU에서 최대 5×까지 가속화됩니다."라고 보고합니다. 그러나 training-memory 관점의 효과는 단순한 speedup보다 더 중요합니다.

> "1회 recurrence은 전체 N×N score matrix를 HBM에 기록하지 않고도 FlashAttention tile이 Q 및 K block에 대한 attention computation을 수행할 수 있게 하는 핵심입니다. online softmax가 없으면 각 tile은 renormalization를 위해 이전에 작성된 부분 결과를 다시 읽어야 합니다. 즉, O(N²) HBM 쓰기가 필요합니다." ([[online-softmax]])

online recurrence에서는 running (m, d) scalar가 register에 들어가고 attention output이 SRAM에 누적됩니다. attention의 backward-pass activation storage는 O(N²)에서 O(N)로 축소됩니다.

### 2.3 recurrence이 numerically exact 이유

정확하게 말할 가치가 있습니다. 이는 근사치가 아닙니다. 온라인 recurrence은 기존의 3단계 계산과 bit-identical 결과를 생성합니다(floating-point associativity까지). 유일한 구조적 요구 사항은 각 xₖ가 순서대로 정확히 한 번 표시되어야 한다는 것입니다. 이는 차단된 attention tile을 통한 스트리밍이 제공하는 것과 정확히 같습니다. 이러한 정확성 덕분에 [[self-attention-no-n2-memory]]는 스트리밍 attention이 "exact, not approximate"고 주장할 수 있습니다.

---

## 3. O(N²) memory가 없는 Self-Attention — Rabe & Staats 2021

### 3.1 핵심 관찰

Rabe & Staats 2021 ([[self-attention-no-n2-memory]])는 문제를 정확하게 설명합니다.

> **핵심 통찰력.** exact self-attention은 softmax normalization를 연기하여 query당 O(1) 추가 memory로 계산할 수 있습니다. 즉, K/V block에 대한 단일 외부 루프 패스에서 weighted values과 실행 중인 normalizer를 축적하고 전체 N×N 점수 matrix을 materialize하지 않습니다.

이 메커니즘은 online softmax recurrence을 attention computation에 직접 적용한 것입니다. S = QKᵀ를 materialize하는 대신 한 번에 하나의 query 출력을 계산합니다.

```python
# Pseudocode for O(1)-per-query streaming attention
# Q: (N, d),  K: (N, d),  V: (N, d)

for i in range(N):                     # outer loop: queries
    q_i = Q[i]                         # (d,)
    m_i, d_i = -inf, 0.0              # running max, running sum
    o_i = zeros(d)                     # running output accumulator

    for j in range(N):                 # inner loop: keys (streamed)
        s_ij = dot(q_i, K[j]) / sqrt(d)   # scalar score
        m_new = max(m_i, s_ij)
        d_i = exp(m_i - m_new) * d_i + exp(s_ij - m_new)
        o_i = exp(m_i - m_new) * o_i + exp(s_ij - m_new) * V[j]
        m_i = m_new

    O[i] = o_i / d_i                  # normalize output
```

임의 순간의 memory: q_i(d scalar), m_i 및 d_i(2 scalar), o_i(d scalar) 및 현재 K[j] / V[j] 행(2d scalar). 총계: O(d) = N과 관계없이 query당 O(1) 추가 memory.

### 3.2 세 가지 변형

[[self-attention-no-n2-memory]]는 세 가지 exact memory regime를 증명합니다.

| Variant | Memory | Mechanism |
|---------|--------|-----------|
| O(1) query별 | 단일 query 외부 루프, 전체 K/V 내부 루프; 실행 중인 scalar만 storage합니다. | 순수 스트리밍; query에 대한 직렬 |
| 완전한 self-attention을 위한 O(log N) | query당 하나의 scalar 인덱스 | 임의의 self-attention을 위한 이론적 최솟값 |
| O(√N) 실용적인 accelerator variant | Q와 K를 √N 크기로 chunking | GPU/TPU tile parallelism을 활용하며, SRAM에 두 chunk를 모두 수용합니다. |

O(√N) variant는 실무적으로 중요합니다. Q와 K를 block size √N으로 chunking하여 두 block을 SRAM에 동시에 수용하고, 각 chunk 안에서 GPU가 효율적으로 수행하도록 설계된 matrix-level GEMM operations를 실행할 수 있게 합니다. xFormers가 구현하고 FlashAttention이 지향하는 variant가 이것입니다.

### 3.3 측정된 memory 감소

N = 16,384(현실적인 long context sequence 길이)에서:

- **inference memory: Standard attention 대비 59× 감소**
- **backpropagation memory: Standard attention 대비 32× 감소**
- **런타임 오버head: 기준선의 몇 퍼센트 이내**

[[self-attention-no-n2-memory]]에서:

> "training-memory 관점: 일반적으로 O(N²·B·H) bytes(batch × heads × N²이며 backward pass을 위해 저장되는 attention scores)로 증가하는 activation tensor를 제거합니다. attention의 backward-pass memory는 O(N²)에서 O(N·d)로 감소합니다. 이는 long-sequence training에서 qualitative regime change입니다."

"qualitative regime change"라는 표현은 정확합니다. N=16k에서 O(N²)와 O(N·d)의 차이는 head dimension이 512일 때 대략 16384/512 ≈ 32×이며, 측정된 32× backpropagation memory 감소와 일치합니다. N이 두 배가 되면 O(N²)는 네 배가 되지만 O(N)는 두 배만 됩니다. 따라서 격차는 계속 커집니다.

### 3.4 backward pass: storage 대신 checkpointing

순진한 backward pass은 모든 중간 attention scores를 storage하여 그라디언트를 계산하므로 memory 이점이 제거됩니다. Rabe & Staats는 selective checkpointing를 통해 이 문제를 해결합니다.

> "이 논문은 청크 요약 기능에 대해 selective checkpointing를 적용합니다. 즉, 기울기는 즉석에서 recompute되며 backpropagation 중에 N×N matrix을 storage하지 않습니다." ([[self-attention-no-n2-memory]])

이는 [[ch-03]]의 gradient checkpointing와 동일한 recomputation 대 storage 트레이드오프이지만, layer 간 activation가 아닌 attention kernel의 내부 상태에 특별히 적용됩니다. storage된 주요 수량은 query당 (m, d) scalar(logsumexp)이며 이는 O(N)이며 N×N matrix을 작성하지 않고도 backward pass에서 attention weights를 recompute할 수 있습니다.

---

## 4. kernel이 결정하는 이유

이 장의 중심 구성: softmax(QKᵀ/√d)V를 계산하기 위한 *algorithm*은 memory 체계를 결정하지만 *kernel*은 하드웨어에서 해당 algorithm을 implementation하는 것입니다.

```
Model architecture:   defines Q, K, V, head count, head dimension
       ↓
Attention algorithm:  standard (O(N²)) vs streaming (O(N))
       ↓
Kernel:               PyTorch MATH / FlashAttention / xFormers / SDPA dispatch
       ↓
Actual GPU memory:    N² bytes vs N·d bytes per head per layer
```

모델은 어떤 경로가 선택되는지 제어하지 않습니다. 이를 결정하는 것은 kernel입니다. [[ch-05]]에서 살펴보듯이:

- **PyTorch SDPA MATH backend** 입력 구성이 빠른 kernel에서 지원되지 않는 경우(잘못된 dtype, non-contiguous layout, causal mask 형식 불일치) O(N²) 경로(전체 점수 matrix 할당)로 자동으로 돌아갑니다.
- **FlashAttention 1/2/3**는 SRAM에서 online softmax tiling을 implementation하고 N×N matrix을 HBM에 쓰지 않습니다.
- **xFormers `memory_efficient_attention`**는 Rabe & Staats O(N) algorithm을 implementation하는 CUTLASS FMHA kernel입니다. PyTorch SDPA의 EFFICIENT_ATTENTION backend입니다.

결정 지점은 모델 코드에 없습니다. backend가 실행되는 곳입니다. 여기에서 수학을 이해하면(online softmax가 O(N) tiling을 activation하는 이유) O(N²) 공간을 피하기 위해 kernel에 필요한 속성이 무엇인지 정확히 알 수 있습니다. kernel은 K/V block을 통해 스트리밍하고 실행 중인 (m, d) normalizer로 출력을 축적할 수 있어야 하며 O(d)보다 큰 버퍼에 전체 attention 행을 쓰지 않아야 합니다.

---

## 5. 종합: Long Context에서의 Memory 구도

O(N²) 비용을 구체적으로 만들려면 표준 및 streaming attention를 사용하여 다양한 sequence 길이에서 7B 모델(L=32, H=32, d_head=128)을 training하는 것을 고려하세요. attention만으로 activation memory, layer당, head 1개, bf16:

```
N=2,048:   2048²  × 2 B = 8  MB  per head   →  32 heads × 32 layers = 8   GB
N=8,192:   8192²  × 2 B = 128 MB per head   →  32 heads × 32 layers = 128 GB
N=32,768:  32768² × 2 B = 2  GB  per head   →  32 heads × 32 layers = 2   TB
```

(이것은 attention activation 숫자입니다. 총 activation memory에는 feedforward layer도 포함되며 [[ch-03]]의 gradient checkpointing는 이 중 일부를 recomputation하기 위해 교환할 수 있습니다.)

O(N) 스트리밍 attention(Rabe & Staats / FlashAttention)을 사용하면 N×N 용어가 사라집니다. Attention의 activation footprint은 O(N·d·H·L·B)가 되며, N=32k, d=128, H=32, L=32, B=1의 동일한 모델에 대해 다음과 같습니다.

```
= 32768 × 128 × 32 × 32 × 1 × 2 B = 8.6 GB
```

N=32k 및 B=1에서 2 TB와 8.6 GB의 차이는 실제 regime change입니다(문제의 규모가 같은데도 실행 가능성 자체가 달라지는 근본적 변화입니다). 32k context에서 kernel 선택은 단순한 최적화가 아니라, training job이 실행되느냐 실행되지 않느냐를 가르는 차이입니다.

---

## 문헌의 핵심 통찰력

**[[self-attention-no-n2-memory]](Rabe & Staats 2021)에서:**
Self-attention에는 O(N²) memory가 필요하지 않습니다. "전체 score matrix를 materialize하는 것"과 "Standard attention을 계산하는 것"이 동등하다는 생각은 수학적 요구 사항이 아니라 implementation 가정입니다. 단일 query로 exact self-attention을 계산할 때의 information-theoretic minimum은 O(d), 즉 running normalizer (m, d)와 output accumulator뿐입니다. 이로써 long-context training의 설계 질문은 "matrix 전체를 memory에 넣을 수 있는가?"에서 "어떤 kernel이 streaming을 올바르게 구현하는가?"로 바뀝니다(저장 용량 문제가 kernel의 streaming implementation 선택 문제로 전환된다는 뜻입니다).

**[[online-softmax]](Milakov & Gimelshein 2018)에서:**
1회 온라인 recurrence — m_new = max(m_old, xₖ), d_new = exp(m_old − m_new)·d_old + exp(xₖ − m_new) —는 activation 기본 요소입니다. approximate 것이 아니라 정확합니다. training memory 맥락에서 그 중요성은 1.3× softmax 속도 향상이 아니라 kernel이 renormalization를 위해 두 번째 HBM 읽기를 요구하지 않고 SRAM 크기의 Q 및 K block에 대한 attention computation을 tiling할 수 있게 한다는 것입니다. 이러한 recurrence이 없으면 tiling에서는 나중에 normalization하기 위해 모든 부분 점수를 storage해야 하므로 memory 절약 효과가 사라집니다.

**substrate은 스트리밍을 강제합니다. kernel은 이를 사용할지 여부를 선택합니다:**
HBM bandwidth(compute 아님)은 long context throughput을 제어합니다. O(N²) score matrix는 HBM 쓰기/읽기 병목 현상입니다. 즉, layer당 head당 2 GB를 HBM에 쓴 다음 출력 matmul에 대해 다시 읽으면 MFU를 제한하는 것과 동일한 bandwidth을 소비합니다. online softmax recurrence은 normalizer를 register에 유지하여 이러한 roundtrip을 제거합니다. substrate(HBM bandwidth 대 SRAM 크기 대 register 파일)은 스트리밍 접근 방식이 승리하는 이유를 정확하게 설명합니다.

**신뢰에는 정확성이 중요합니다.**
Rabe & Staats O(1) 변형과 [[online-softmax]]를 기반으로 구축된 FlashAttention kernel은 모두 Standard attention를 사용하여 bit-consistency 있는 결과를 생성한다는 점에서 *정확*합니다(최대 floating-point associativity, fused kernel과 동일). 이는 approximate attention(Linformer 또는 Longformer와 같은)가 아닙니다. N×N 계산은 여전히 ​​발생합니다. HBM에서 materialize되는 것이 아니라 SRAM tile에서만 발생합니다. 이러한 차이 때문에 FlashAttention을 model-accuracy regression 없이 drop-in replacement으로 사용할 수 있습니다.

---

## 주요 시사점

- Standard attention은 S = QKᵀ와 P = softmax(S)를 head당 O(N²) activation으로 materialize합니다. N=32k 및 70B 모델에서는 물리적으로 저장할 수 없습니다.
- Milakov & Gimelshein 2018 1회 recurrence((m, d) 요소당 업데이트된 scalar 실행)은 전체 점수 행에 대한 두 번째 패스의 필요성을 제거하여 O(1) 추가 상태를 사용하여 단일 패스 numerically stable softmax를 가능하게 합니다.
- Rabe & Staats 2021는 이것을 self-attention에 적용합니다. 실행 중인 (m, d) normalizer를 사용하여 K/V block을 통해 스트리밍하면 query당 attention activation memory가 O(N²)에서 O(1)로, 실제 GPU 병렬 변형의 경우 O(√N)로 줄어듭니다. N=16k에서 측정됨: 59× inference memory 감소, 32× backpropagation 감소.
- backward pass는 query당 logsumexp scalar의 selective checkpointing를 사용하여 즉석에서 attention weights를 recompute하며 backpropagation 중에 N×N matrix을 storage하지 않습니다.
- 모델이 아닌 kernel이 적용되는 regime를 결정합니다. 이는 중앙 설계 축 [[ch-05]](FlashAttention) 및 [[ch-06]](SDPA/xFormers/SageAttention/Ring/Paged)가 자세히 매핑됩니다.
- 7B 모델의 N=32k에서 Standard attention와 streaming attention의 차이는 attention-only activation memory에서 대략 2 TB 대 8.6 GB입니다. 이는 최적화가 아니며 작업을 실행하기 위한 전제 조건입니다.

---

## 참고자료

- 마커스 N. 라베(Markus N. Rabe)와 찰스 스타츠(Charles Staats). "self-attention에는 O(n²) memory가 필요하지 않습니다." arXiv:2112.05682, 12월 2021. https://arxiv.org/abs/2112.05682 - [[self-attention-no-n2-memory]]
- 막심 밀라코프와 나탈리아 기멜세인. "softmax에 대한 온라인 normalizer 계산." arXiv:1805.02867, 5월 2018. https://arxiv.org/abs/1805.02867 - [[online-softmax]]

**연관 장:** [[ch-03]](gradient checkpointing 및 activation 예산), [[ch-05]](이 수학을 기반으로 구축된 FlashAttention 1/2/3), [[ch-06]](전체 attention kernel zoo와 각 kernel의 training-memory profile), [[ch-09]](capstone: long-context MoE 예산).
