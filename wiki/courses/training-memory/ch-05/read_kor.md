<!-- chapter: ch-05
     track: attention
     kind: content
     title: FlashAttention (1/2/3): IO-Aware Exact Attention
     deps: [[ch-04]]
     sources: [[flash-attention-1]], [[flash-attention-2]], [[flash-attention-3]]
-->

# 5장 — FlashAttention (1/2/3): IO-Aware Exact Attention

> **핵심 통찰.** Attention이 느리고 memory를 많이 사용하는 이유는 arithmetic complexity가 아니라 HBM bandwidth다. Naive implementation(Q·Kᵀ → softmax → ·V)은 N×N score matrix를 HBM에 쓰고 서로 다른 세 단계에서 다시 읽는다. FlashAttention은 Q, K, V를 SRAM에 들어가는 block으로 tiling하고 numerically exact한 online softmax recurrence를 on-chip에서 유지하여 세 번의 HBM round-trip을 모두 제거한다. N×N matrix는 materialize되지 않으므로 activation memory가 O(N)이 되며, FLOPs를 줄여서가 아니라 IO를 제거함으로써 wall-clock이 2–3× 빨라진다.

> **지침.** Sequence length가 약 512 token을 넘는 모든 training run에서 FlashAttention을 활성화하라(`FLASH_ATTENTION` backend를 지정한 `torch.nn.functional.scaled_dot_product_attention` 또는 `flash-attn` package를 직접 사용). A100에서는 동일한 O(N) memory로 FA1보다 throughput이 약 2× 높은 FA2를 사용하라. H100에서는 740 TFLOPs/s(FP16) 또는 약 1.2 PFLOPs/s(FP8)를 내는 FA3를 사용하라. Training이 `MATH` backend로 silent fallback하도록 두지 마라. 이 backend는 전체 O(N²) score matrix를 allocation하며 약 38× 느리다. `torch.backends.cuda.flash_sdp_enabled()`를 점검하고 지원하지 않는 dtype/mask로 인한 fallback을 감시하라.

---

## 1. 문제: Bottleneck은 HBM이다

Sequence length N과 head dimension d에 대한 naive attention algorithm은 다음과 같이 HBM에 access한다([[flash-attention-1]]).

```
# Standard attention — what actually runs without FA
S = Q @ K.T              # shape (N, N) — WRITE to HBM: Θ(N²) bytes
P = softmax(S)           # READ S from HBM, WRITE P: Θ(N²) bytes again
O = P @ V                # READ P from HBM: Θ(N²) bytes a third time
# Total HBM traffic: Θ(Nd + N²)
```

GPU의 on-chip SRAM(A100에서 약 20 MB)은 약 19 TB/s지만 off-chip HBM은 약 1.5–2 TB/s이므로 bandwidth 차이가 10×다. N=2048, batch=32, head=32, d=64이면 layer 하나의 score matrix 크기는 다음과 같다.

```
32 (batch) × 32 (heads) × 2048² (scores) × 2 (bytes/fp16) = 8 GB per layer
```

이 8 GB를 forward pass마다 세 번 쓰고 읽어야 한다. 32개 layer에서는 step마다 768 GB의 N×N buffer가 2 TB/s인 pipe를 통과한다(pipe는 제한된 HBM bandwidth를 비유하며, 실제로는 그만큼의 memory traffic이 발생한다는 뜻이다). 진짜 bottleneck은 tensor core throughput이 아니라 이 이동이다.

**FlashAttention의 해법:** 각 tile이 SRAM에 들어가도록 Q, K, V를 tiling하고 output O와 softmax statistic을 그 안에서 accumulate하며 score matrix는 HBM에 전혀 쓰지 않는다.

> **대화형 보조 자료:** [figures/flash-tiling.html](figures/flash-tiling.html) — tiling loop를 tile별로 실행하면서 data가 SRAM과 HBM 중 어디에 존재하는지, online softmax running statistic(m, l)이 각 step에서 어떻게 update되는지 보여 준다.

---

## 2. FlashAttention-1: Tiling + Online Softmax (Dao et al., 2022)

### 2.1 Tile Size

Block size는 Q/K/V tile 두 개가 동시에 SRAM에 들어가도록 정한다([[flash-attention-1]]).

```
Br = ⌈M / 4d⌉    # Q tile rows   (M = SRAM size, e.g. 20 MB)
Bc = min(⌈M / 4d⌉, d)  # K/V tile columns
```

각 (Br × Bc) pair는 compute 도중 HBM에 접근하지 않고 계산할 수 있는 독립적인 attention sub-problem이다.

### 2.2 Online softmax recurrence

Single-pass tiling을 numerically stable하게 만드는 mathematical primitive는 Milakov & Gimelshein(2018)의 **online softmax**다. K score의 새로운 tile이 vector x_new로 들어올 때마다 다음 recurrence를 적용한다.

```
m_new = max(m_old, max(x_new))           # running row maximum
d_new = exp(m_old - m_new) * d_old       # rescale old denominator
      + sum(exp(x_new - m_new))           # add new tile's contribution
O_new = (d_old / d_new) * exp(m_old - m_new) * O_old  # rescale accumulated output
      + exp(x_new - m_new) * V_tile / d_new            # add new tile's contribution
```

모든 K/V tile을 처리한 뒤 `O_new`는 전체 N×N matrix를 materialize하고 global softmax를 적용했을 때 얻는 값과 정확히 같은 softmax-weighted value다. 이 recurrence가 없으면 renormalize하기 위한 두 번째 HBM read를 피할 수 없다.

**HBM traffic:** FlashAttention은 전체 HBM access를 Θ(Nd + N²)에서 **Θ(N²d²/M)**으로 줄인다. Nd에 비해 M이 충분히 크면 round-trip 수가 훨씬 적다. 실제로 N×N write/read cycle은 완전히 제거된다.

### 2.3 Memory Complexity: O(N) Activation

FlashAttention의 forward pass가 저장하는 것은 다음뿐이다([[flash-attention-1]]).

| Tensor | Shape | Cost |
|--------|-------|------|
| Output O | (N, d) | O(Nd) |
| Softmax row-max m | (N,) | O(N) |
| Softmax denominator l | (N,) | O(N) |
| Input Q, K, V | 각각 (N, d) | O(Nd) — activation에 이미 포함됨 |

N×N score matrix S와 probability matrix P는 **저장하지 않는다**. 위 예(batch 32, head 32, N=2048, d=64)에서는 attention weight의 layer당 activation memory가 8 GB에서 거의 0으로 감소한다. O(N) logsumexp statistic m과 l만 유지하며 합계는 약 8 MB다.

### 2.4 Backward Pass: 저장보다 Recomputation이 저렴하다

Backward pass는 gradient 계산에 S와 P가 필요하다. FlashAttention은 이를 저장하여 layer당 8 GB를 사용하는 대신 저장된 Q, K, V, m, l로부터 **recompute**한다([[flash-attention-1]]).

```
# Backward: recompute S and P on-the-fly from stored (Q, K, V, l, m)
# Cost: ~33% extra FLOPs vs a hypothetical "store everything" backward
# Saving: 8 GB/layer of N×N buffers that would otherwise stay resident
#         in GPU memory across the entire forward pass
```

이는 gradient checkpointing([[ch-03]])과 같은 recomputation-vs-storage tradeoff를 attention-weight tensor에 한정해 적용한 것이다. 추가 FLOPs의 비용이 저장된 tensor가 유발할 HBM traffic보다 낮다.

### 2.5 측정 결과(FA1)

- BERT-large(N=512): MLPerf 1.1 baseline 대비 end-to-end 15% speedup
- GPT-2 (N=1024): **3× speedup**
- Long-range Arena(N=1K–4K): 2.4× speedup
- Block-sparse FA: 64K-token sequence를 지원하며 Path-256(accuracy 63.1%)을 달성한 최초의 Transformer

GPU MFU는 25–40%다. Warp-level work partitioning이 bottleneck이며 FA2에서 해결한다.

---

## 3. FlashAttention-2: Warp Work Partitioning (Dao, 2023)

FA1의 MFU가 25–40%에 그친 이유는 서로 다른 warp가 softmax rescaling 단계에서 shared memory에 쓰기 위해 경쟁했기 때문이다. "split-K" layout에서는 warp 네 개가 각각 K/V strip을 담당하지만 공통 softmax denominator 계산을 위해 **synchronize**해야 한다. 이 inter-warp communication이 shared-memory round-trip을 유발하며 FA2는 이 구조를 수정한다([[flash-attention-2]]).

### 3.1 세 가지 Algorithm 개선

**1. Non-matmul FLOPs 감소 — Q-outer Loop Reordering**

FA1은 `각 Q tile → 각 K/V tile → rescale` 순서로 iterate한다. Tile pair당 O(d) operation인 rescaling은 non-matmul instruction을 사용하며 A100에서 throughput이 tensor-core matmul의 약 1/16이다. FA2는 Q tile별로 모든 K/V tile의 rescaling을 batch하도록 재구성하여 non-matmul operation 수를 비례해서 줄인다.

**2. Sequence Dimension에 대한 Parallelism**

Causal masking으로 column strip 사이에 data dependency가 생기므로 FA1은 단일 batch×head 내부를 parallelize할 수 없었다. FA2는 **Q의 서로 겹치지 않는 row range를 서로 다른 thread block에 배정**한다. 각 block은 독립적으로 parallel 실행되므로 특히 small-batch 또는 few-head configuration에서 SM occupancy가 직접 높아진다.

**3. Warp-per-output-row Partitioning**

FA1: warp 네 개가 각각 K/V strip을 담당하므로 softmax denominator를 위한 inter-warp communication이 필요하다.
FA2: 각 warp가 **Q output row의 partition**을 담당하여 softmax를 위해 shared memory에 접근하지 않고 완전한 output row를 생성한다. Softmax 단계의 inter-warp communication은 0이다.

### 3.2 성능

| Metric | FA1 | FA2 | Ratio |
|--------|-----|-----|-------|
| A100 MFU (attention) | 25–40% | 50–73% | ~2× |
| Peak TFLOPs/s (A100) | ~110 | **225** | ~2× |
| Memory (activation) | O(N) | O(N) | same |
| Wall-clock vs FA1 | — | **~2×** | — |

Activation memory 체계는 FA1과 같다. O(N) logsumexp storage와 backward recomputation이 동일하다. 이득은 더 나은 warp utilization에서 나오는 **순수한 throughput**이다. Memory budget이 고정되어 있으면 attention이 2× 빨라질 때 시간당 token 수가 2×가 되거나 같은 wall-clock step time에 2× 긴 sequence를 처리할 수 있다.

---

## 4. FlashAttention-3: Hopper-Specific Async + FP8 (Shah et al., 2024)

FA2는 synchronous하다. 각 warp가 GEMM을 issue하고 완료를 기다린 뒤 softmax를 실행하고 다음 GEMM을 issue한다. Hopper H100에서는 Tensor Memory Accelerator(TMA)의 async pipeline이 idle 상태로 남는다. GPU가 data loading과 compute를 overlap할 수 있지만 FA2는 이를 활용하지 않기 때문이다. FA3는 **producer-consumer warp specialization**을 중심으로 재구성한다([[flash-attention-3]]).

### 4.1 세 가지 Hopper-Specific 기법

**1. Producer-consumer ping-pong pipeline (TMA async)**

```
# FA3 warp layout on H100
Producer warps:  drive TMA async loads → double-buffer tiles in shared memory
Consumer warps:  drive tensor-core GEMM (wgmma) on the already-loaded tile
# Overlap: while consumer runs GEMM on tile i, producer loads tile i+1
# Result: consumer never stalls waiting for data
```

TMA는 H100 전용 unit이므로 Hopper에서만 가능하다. A100에는 동등한 unit이 없으며 FA2 design이 A100에는 적합하다.

**2. Interleaved Block-wise GEMM + Softmax**

Consumer warp group 안에서 두 GEMM으로 이루어진 attention sequence `(S = QKᵀ, 이후 P·V)`를 pipeline한다. Partial P의 softmax rescaling이 진행되는 동안 다음 GEMM stage를 tensor core에 issue한다. Softmax latency가 matmul throughput 뒤에 숨겨진다(두 작업을 overlap하여 별도 대기 시간이 드러나지 않는다는 뜻이다).

**3. FP8 Block Quantization + Incoherent Processing**

- **Per-block quantization:** tensor 전체가 아니라 각 attention tile 내부의 작은 tile을 quantize하여 각 단계가 다뤄야 하는 dynamic range를 제한한다.
- **Incoherent processing:** quantization 전에 random Hadamard rotation을 적용하여 outlier activation을 모든 element에 균일하게 분산하고 FP8 range를 효율적으로 사용한다.
- 결과: naive per-tensor FP8 attention보다 **numerical error가 2.6× 낮다**.

### 4.2 H100 SXM5 측정 성능

| Precision | TFLOPs/s | Utilization | vs FA2 |
|-----------|----------|-------------|--------|
| FP16 | **740** | 75% | ~2× |
| FP8 | **~1200** | ~75% | — |

H100의 FA2는 synchronous kernel design이 TMA pipelining을 활용하지 못하므로 약 370 TFLOPs/s(utilization 약 40%)에 그친다. FA3는 utilization을 거의 두 배로 높인다.

### 4.3 FA3의 Training-Memory Profile

FA3는 FA1/2가 확립한 O(N) activation memory 체계를 바꾸지 **않는다**([[flash-attention-3]]). Memory 측면의 의미는 bandwidth level에 있다.

- TMA가 더 적은 warp stall로 tile을 SRAM에 빠르게 전달한다 → SRAM을 더 연속적으로 사용한다 → stall로 인한 re-fetch에 낭비되는 HBM bandwidth가 줄어든다.
- FP8은 FP16 대비 attention tile당 bytes를 절반으로 줄인다 → tile당 HBM bandwidth 사용량이 절반이 된다 → 같은 step당 HBM bandwidth budget에서 더 긴 sequence를 처리한다.

O(N) logsumexp storage와 backward recomputation strategy는 FA1과 같다.

---

## 5. Training-Memory Ledger에 미치는 영향

Long-context training에서 가장 큰 memory lever(memory 사용량을 줄이는 수단)는 O(N²) attention activation의 제거다. L=32 layer, B=8 batch, H=32 head, N=8192, d=64, FP16인 model을 예로 보자.

```
# Naive attention — score matrix per layer per forward pass:
  B × H × N × N × 2 bytes = 8 × 32 × 8192 × 8192 × 2 = 34 GB per layer
  × 32 layers = 1.1 TB (impossible on any current GPU)

# FlashAttention — logsumexp statistics only:
  B × H × N × 2 (for m and l) × 4 bytes (fp32) = 8 × 32 × 8192 × 2 × 4 = 134 MB per layer
  × 32 layers = 4.3 GB
```

FlashAttention은 N=8192에서 attention activation footprint를 TB 단위에서 GB 단위로 줄인다. 따라서 FA는 long-context fine-tuning이나 pretraining의 단순한 optimization이 아니라 prerequisite다.

[[ch-03]]의 gradient checkpointing과는 절감 효과가 더해진다. Checkpointing은 MLP, layer norm, residual 같은 다른 activation을 줄이고 FA는 attention score를 줄인다. N≥4096에서는 FA의 절감량이 aggressive checkpointing의 효과보다도 크다.

---

## 6. FA1 → FA2 → FA3 발전 과정 요약

| Version | Key innovation | MFU (A100) | MFU (H100) | Memory | Hardware target |
|---------|---------------|-----------|-----------|--------|----------------|
| FA1 (2022) | Tiling + online softmax, O(N) memory | 25–40% | — | O(N) | 모든 CUDA GPU |
| FA2 (2023) | Warp-per-row, Q-outer loop, seq parallelism | 50–73% | 40% | O(N) | A100 (Ampere) |
| FA3 (2024) | TMA async, producer-consumer, FP8 | — | 75% FP16, ~75% FP8 | O(N) | H100 (Hopper) |

**변하지 않는 것**(hardware substrate가 강제하는 조건):
- Online softmax recurrence — streaming에 필수이며 두 번째 HBM pass 없이는 생략할 수 없다.
- O(N) activation footprint — tiling의 수학적 결과이며 세 version이 모두 공유한다.
- Logsumexp를 이용한 backward recomputation — 어떤 bandwidth 체계에서도 N×N을 저장하는 것보다 저렴하다.

**달라지는 것**(version에 따라 발전한 design choice):
- Warp/thread-block work partition — FA1은 synchronization overhead가 있는 split-K를 사용했고 FA2는 row별로 독립적인 split-Q로 전환했다.
- Data loading strategy — FA2는 software-managed prefetch에 의존하고 FA3는 Hopper 전용 TMA hardware unit을 사용한다.
- Numeric precision — FA1/2는 FP16/BF16만 지원하고 FA3는 error control을 위한 incoherent processing과 함께 FP8을 추가한다.

---

## 7. Online Softmax Recurrence — Derivation 개요

Recurrence가 numerically stable한지 검증하려는 독자를 위해 유도 과정을 살펴본다(핵심 시험 문제다).

Row x ∈ ℝᴺ의 standard softmax:
```
softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)
```

x를 두 부분 x = [a; b]로 나눈다.
```
m_a = max(a)
m_b = max(b)
m   = max(m_a, m_b)

# Numerically stable: subtract global max before exp
softmax([a; b])_i = exp(x_i - m) / [Σⱼ∈a exp(aⱼ - m) + Σⱼ∈b exp(bⱼ - m)]
                  = exp(x_i - m) / [exp(m_a - m)·Σⱼ exp(aⱼ - m_a) + exp(m_b - m)·Σⱼ exp(bⱼ - m_b)]
```

이것이 recurrence다. `d = Σⱼ exp(xⱼ - m)`인 running (m, d)를 유지하고 새로운 score가 들어오면 m과 d를 모두 update한다. Rescaling factor `exp(m_old - m_new)`가 exact하며 tile당 O(1)에 계산되므로 두 번째 pass가 필요 없다. O도 같은 방식으로 accumulate한다.

임의의 split에서도 이것이 가능하다는 최초 증명은 Rabe & Staats(2021)에 있으며 [[ch-04]]에서 mathematical foundation으로 다룬다. FA1은 이를 최초의 실용적이고 hardware-efficient한 구현으로 실현했다.

---

## 문헌에서 얻은 핵심 통찰

**1. Attention에는 IO가 올바른 cost model이다**([[flash-attention-1]]). Standard attention은 Θ(Nd + N²), FA는 Θ(N²d²/M)의 HBM access를 수행한다. N=2048에서 layer당 8 GB인 score matrix 때문에 standard attention은 compute-bound가 아니라 I/O-bound이며 tensor core FLOPs는 data를 기다리며 idle 상태가 된다. Output correctness를 유지하면서 HBM traffic을 줄이는 optimization은 반드시 이득이다.

**2. GPU utilization은 warp-level work partitioning에서 결정된다**([[flash-attention-2]]). FA1의 25–40% MFU는 tiling algorithm의 이론적 한계가 아니라 implementation 결함이다. Data를 담당하는 warp를 split-K에서 split-Q로 재구성하면 mathematical algorithm을 바꾸지 않고 throughput이 거의 두 배가 된다. Algorithmic correctness와 hardware efficiency는 별개의 문제라는 일반적인 kernel-design 교훈이다.

**3. Hopper에는 단순히 더 빠른 chip이 아니라 새로운 programming model이 필요하다**([[flash-attention-3]]). H100의 FP16 FLOPs/s는 A100의 2×지만 H100의 FA2는 약 370 TFLOPs/s에 그쳐 FA3의 740 TFLOPs/s보다 낮다. Synchronous kernel design이 TMA를 사용할 수 없기 때문이다. Hardware generation이 바뀌면 기존 software pattern이 무효가 될 수 있으므로 parameter tuning이 아니라 새로운 kernel generation이 필요하다.

**4. 긴 sequence에서는 backward recomputation이 N×N storage보다 명확히 우수하다**([[flash-attention-1]]). N×N matrix를 저장하고 다시 읽는 HBM bandwidth cost는 Q, K, V, l, m으로 recompute하는 FLOPs cost보다 훨씬 크다. 이는 [[ch-03]]의 gradient checkpointing 원리를 single-layer, single-head attention-weight tensor라는 가장 세밀한 granularity에 적용한 것이다.

---

## 핵심 정리

- FlashAttention의 speedup은 FLOP 감소가 아니라 **HBM read/write 제거**에서 나온다. 대상은 N×N score matrix다.
- Tile 전체에서 running m, d, O를 유지하는 **online softmax recurrence**는 single-pass tiling을 exact하게 만드는 mathematical primitive다. 이것이 없으면 두 번째 HBM pass를 피할 수 없다.
- 세 version 모두 **O(N) activation memory**와 (l, m)을 이용한 **backward recomputation**을 공유한다. Version별 변화는 warp partitioning(FA2)과 async pipelining + FP8(FA3)이다.
- N=8192에서 naive attention은 score matrix에만 layer당 약 34 GB가 필요하지만 FA는 이를 약 134 MB로 줄여 250× 절감한다. 이는 aggressive gradient checkpointing보다 효과가 큰 **주요 long-context training memory lever**다.
- FA2 on A100: **225 TFLOPs/s, 50–73% MFU**. FA3 on H100: **740 TFLOPs/s (FP16), ~1.2 PFLOPs/s (FP8), 75% MFU**.
- PyTorch SDPA가 `MATH` backend로 silent fallback하면 전체 O(N²) score matrix를 allocation하고 약 38× 느리게 실행된다. 모든 training setup에서 이를 점검하라.

---

## 참고 문헌

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. https://arxiv.org/abs/2205.14135 ([[flash-attention-1]])
- Tri Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. https://arxiv.org/abs/2307.08691 ([[flash-attention-2]])
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608, July 2024. https://arxiv.org/abs/2407.08608 ([[flash-attention-3]])

**관련 chapter:** [[ch-03]] (gradient checkpointing — the same recompute-vs-store tradeoff), [[ch-04]] (online softmax math + Rabe & Staats O(N) theory), [[ch-06]] (the broader attention kernel zoo: SDPA backends, xFormers, SageAttention, Ring Attention, PagedAttention)
