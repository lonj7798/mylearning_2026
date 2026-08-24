<!-- chapter: ch-06
     track: attention
     title: The Attention Kernel Zoo: SDPA, xFormers, SageAttention, Ring, Paged
     deps: [[ch-05]]
     sources: [[pytorch-sdpa]], [[xformers-mem-efficient]], [[sage-attention]], [[ring-attention]], [[paged-attention]]
-->

# 6장 — Attention Kernel 동물원(여러 종류의 kernel 모음): SDPA, xFormers, SageAttention, Ring, Paged

> **핵심 통찰.** [[ch-05]]에서 FlashAttention이 확립한 O(N) memory 체제는 하나의 고정된 경로가 아니다. PyTorch `scaled_dot_product_attention`은 서로 다른 네 backend로 보내는 runtime dispatcher이며, 각 backend에는 서로 다른 hardware 제약이 있다. 제약을 충족하지 못했을 때 math backend로 조용히 fallback하는 것은 FlashAttention을 활성화했다고 믿는 codebase에서 예상치 못한 OOM이 발생하는 가장 흔한 원인이다. 이 dispatcher를 넘어서면 kernel 계열은 quantized inference 가속(SageAttention: INT8 QKᵀ), device 간 sequence sharding(Ring Attention: GPU당 O(L/D)), inference-serving KV-cache 관리(PagedAttention: paged physical blocks)로 가지를 뻗는다(서로 다른 기술 문제의 갈래로 나뉜다). 이 기법들은 근본적으로 서로 다른 문제를 해결하며, 그중 일부만 training memory와 관련된다.

> **지침.** training loop에서는 조용한 dispatch에 의존하지 말고 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`으로 backend를 명시적으로 고정한 뒤 `RuntimeError`를 처리하라. device당 O(N) KV memory조차 HBM을 넘칠 때는 Ring Attention(context parallelism, `--context-parallel-size D`)을 사용하라. SageAttention과 PagedAttention은 inference 도구로 취급하고, 그 memory 기법이 training으로 이전되지 않는 이유를 이해하라. GDN linear-attention MoE처럼 model이 standard attention을 완전히 금지한다면 전체 kernel 동물원(사용 가능한 여러 kernel 선택지)은 무의미해지고, activation memory 문제에는 다른 분석이 필요하다([[ch-09]] capstone 연결부 참조).

---

## 1. 기반: 애초에 Kernel 동물원(여러 종류의 kernel 모음)이 존재하는 이유

[[ch-04]]에서는 긴 sequence에서 attention의 n×n score matrix가 activation memory를 가장 많이 소비한다는 점을 확립했다. [[ch-05]]에서는 FlashAttention이 SRAM에서 computation을 tile로 나누고 backward에서 score matrix를 다시 계산하여 activation memory를 O(N²)에서 O(N)으로 줄인다는 점을 보였다. 그러나 하나의 kernel이 모든 hardware, 모든 dtype, 모든 attention bias pattern을 포괄하지는 못한다. 그 결과, 모두 동일한 수학 연산인 softmax(QKᵀ/√d)V를 목표로 하지만 다음 항목에서 서로 다른 구현의 동물원(다양한 구현 집합)이 생겼다.

- 지원하는 hardware 세대와 dtype
- 어떤 제약 위반 때문에 실행을 거부하는지(그리고 거부할 때 무슨 일이 일어나는지)
- training 중에도 동작하는지(backward pass 필요), 아니면 inference 중에만 동작하는지
- computation을 여러 device에 걸쳐 shard하는지(ring topology 필요)

이를 지배하는 tradeoff는 더 유능한 kernel일수록 input에 대해 더 강한 가정을 한다는 것이다. 이러한 가정이 위반되면 kernel은 정상적으로 실패하거나(error 발생), 조용히 성능이 저하된다(38× 느린 math backend로 fallback). 가장 위험한 결과는 error message 없이 조용히 성능이 저하되는 것이다.

> **대화형 동반 자료:** [figures/kernel-memory.html](figures/kernel-memory.html) — 전환 가능한 설정(batch size, head dim, Ring Attention의 device 수)을 사용하여 다섯 kernel 모두의 device당 memory cost와 sequence length를 나란히 시각화하며, 각 kernel의 O(N²), O(N), O(L/D) 곡선이 정확히 어디에서 갈라지는지 보여 준다.

---

## 2. PyTorch SDPA: Dispatcher — [[pytorch-sdpa]]

**정의.** `torch.nn.functional.scaled_dot_product_attention` (SDPA) 자체는 attention kernel이 아니다. 이것은 hardware, dtype, head dimension, input 속성에 따라 runtime에 네 backend 중 하나를 선택하는 dispatch layer다. `F.scaled_dot_product_attention`을 호출하는 모든 Hugging Face transformer는 이 dispatcher를 거친다.

**네 가지 backend** ([[pytorch-sdpa]]):

| Backend | Algorithm | Memory | 제약 |
|---|---|---|---|
| `MATH` | 순수 PyTorch C++; 전체 N×N score matrix를 materialize함 | O(N²) | 항상 사용 가능 |
| `FLASH_ATTENTION` | FlashAttention-2 kernel | O(N) | CUDA, fp16/bf16, head_dim ≤ 128, 임의의 bias 불가 |
| `EFFICIENT_ATTENTION` | xFormers CUTLASS FMHA (Rabe & Staats O(N)) | O(N) | 더 넓은 범위: custom bias, 더 큰 head_dim |
| `CUDNN_ATTENTION` | cuDNN SDPA graph; autograd 호환 | O(N) | 지원되는 cuDNN + CUDA version 필요 |

**성능 격차.** [[pytorch-sdpa]] 발췌문은 구체적인 benchmark를 제시한다. 동일한 input에서 MATH는 ~87,478 µs인 반면 optimized backend는 ~2,274 µs다. 이는 ~38× 차이이며, 전적으로 N×N score matrix의 HBM bandwidth cost에서 비롯된다. memory 관점에서 n=4096, batch=16, 32 heads일 때 math backend는 layer당 ~8 GB의 attention activation을 할당하지만, FlashAttention은 이를 ~200 MB로 줄인다. fallback이 조용히 발생하면 경고 없이 한 order of magnitude(약 10배 규모)의 regression이 일어난다.

**조용한 fallback 함정(예상치 못하게 느리고 메모리를 많이 쓰는 backend로 전환되는 조건).** 명시적으로 backend를 선택하지 않으면 다음 중 어느 것이든 PyTorch가 error나 warning 없이 조용히 MATH로 되돌아가게 한다.
- 지원되지 않는 dtype (fp32, int8)
- head_dim > 128 (FlashAttention 제약)
- 표준이 아닌 shape의 Attention bias tensor
- 일부 PyTorch version에서 explicit mask tensor와 결합된 `is_causal=True`
- 특정 cuDNN configuration에서 홀수 sequence length

올바른 pattern ([[pytorch-sdpa]]):

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# Explicit: raises RuntimeError if flash unavailable — the safe failure mode
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

backend를 사용할 수 없을 때 `RuntimeError`를 발생시키는 것이 안전한 결과다. 위험한 결과는 default dispatch가 조용히 MATH까지 fall through하는 것이다. 모든 production training codebase는 backend를 고정하고 error를 명시적으로 처리해야 한다. dispatcher를 검증하지 않은 채 "어딘가에서 use_flash_attention=True를 설정했다"에 의존하는 것으로는 충분하지 않다.

**[[ch-05]]와의 연결.** SDPA의 FLASH_ATTENTION backend는 FA2의 O(N) 체제가 user code에 도달하는 경로다. EFFICIENT_ATTENTION backend는 다음에 설명할 xFormers 경로다. CUDNN_ATTENTION은 cuDNN graph API를 감싼다. 세 가지 모두 O(N) activation memory를 제공하며, MATH는 O(N²) 함정(의도치 않은 memory 복잡도 증가)이다.

---

## 3. xFormers memory_efficient_attention — [[xformers-mem-efficient]]

**정의.** Rabe & Staats (2021)의 O(N) streaming attention algorithm을 CUTLASS FMHA kernel로 fuse한 Meta의 production 구현이다. 이것은 PyTorch SDPA의 `EFFICIENT_ATTENTION` backend로, FlashAttention의 hardware 제약을 충족하지 못했을 때 사용하는 fallback O(N) 경로다.

**Algorithm.** [[self-attention-no-n2-memory]] (Rabe & Staats 2021, arXiv:2112.05682)은 세 가지 exact memory variant를 증명했다.

- **query당 O(1)**: Q row를 순회하는 단일 outer loop와 streaming K/V inner loop를 사용한다. weighted value와 running normalizer를 SRAM에서만 누적하며 N×N score matrix를 HBM에 절대 쓰지 않는다.
- **전체 self-attention에 O(log N)**: 임의의 self-attention에 대한 이론적 최솟값이다.
- **O(√N) 실용적 accelerator variant**: Q와 K를 모두 √N 크기로 chunk하여 tile을 SRAM에 유지하면서 GPU/TPU tile parallelism을 활용한다.

N=16,384에서 Rabe & Staats는 standard attention 대비 inference memory가 59× 감소하고 backpropagation memory가 32× 감소하는 것을 측정했다([[self-attention-no-n2-memory]]). xFormers CUTLASS FMHA는 hardware별 dispatch, 즉 Ampere (sm80) CUTLASS kernel, `_get_use_fa3` toggle을 통한 Hopper의 FlashAttention-3, AMD ROCm용 Composable Kernel을 사용해 이 streaming algorithm을 구현한다([[xformers-mem-efficient]]).

**FA2와 나란히 존재하는 이유.** 두 kernel은 coverage와 peak throughput 사이에서 차이가 난다.

- FA2/FA3: standard configuration(fp16/bf16, head_dim ≤ 128, standard causal mask, A100에서 theoretical peak의 ~60–70%)에서 A100/H100의 throughput이 더 높다.
- xFormers FMHA: 임의의 attention bias tensor(per-head ALiBi, relative position bias, `BlockDiagonalMask`, `PagedBlockDiagonalGappyKeysMask`), 더 큰 head_dim, 더 오래된 hardware를 더 폭넓게 지원한다. standard causal attention의 경우 A100에서 FA2 속도의 약 60–70%로 실행되지만, FA2를 사용할 수 없을 때 유일한 대안이 38× 느린 MATH backend이므로 올바른 선택이다([[xformers-mem-efficient]]).

**Training-memory 관점.** xFormers FMHA와 FA2는 모두 O(N) activation memory를 제공하며, 이는 [[ch-05]]와 동일한 체제 변화다. 둘 중 하나를 고르는 것은 memory 문제가 아니라 throughput 문제다. 실용적 규칙은 FA2를 사용할 수 있고 configuration이 그 제약을 충족하면 FA2를 선호하고, 그렇지 않으면 MATH보다 훨씬 나은 EFFICIENT_ATTENTION으로 넘어가는 것이다.

---

## 4. SageAttention — [[sage-attention]]

**정의.** 두 attention matrix multiplication을 mixed-precision 연산으로 대체하여 memory bandwidth를 줄이고 throughput을 개선하는 quantized attention kernel이다(Jintao Zhang et al., ICLR 2025, arXiv:2410.02367). **Inference 전용**이며, backward-pass 구현은 존재하지 않는다.

**Quantization 방식** ([[sage-attention]]). attention의 두 matrix multiplication은 서로 다른 수치적 속성을 지닌다.

- **QKᵀ matmul**: multiplication 전에 Q와 K를 **INT8**로 quantize한다. 논문은 이 연산에서 INT8 accuracy가 FP8(E4M3와 E5M2 모두)을 능가하고, RTX4090/RTX3090에서 INT8 matmul이 FP8 matmul보다 2× 빠르다는 것을 보여 준다. 주의점은 K에 naive INT8 quantization을 망칠 per-channel outlier가 있다는 것이다.
- **P·V matmul**: P(softmax output)와 V를 FP16 accumulator와 함께 **FP16**으로 유지한다. FP16은 "accuracy를 보존하면서 이 단계에서 INT8보다 2× 더 빠르다"이기 때문에 이 matmul은 quantize하지 *않는다*.

**K channel smoothing.** K에 INT8을 적용할 수 있게 하는 핵심 통찰이다. K에는 “뚜렷한 channel별 outlier”가 있다. 즉, 각 token의 key vector는 큰 shared bias와 작은 token별 signal의 합이다. 다음을 적용한다.

```
γ(K) = K − mean(K)    # mean across the token dimension
```

softmax는 logits에 일률적으로 더해지는 shift에 대해 불변이므로, 이는 attention output을 변경하지 않는다.

```
σ(q · (K − mean(K))ᵀ) = σ(q · Kᵀ)
```

shared bias는 softmax에서 상쇄된다. overhead는 runtime의 <0.2%다. smoothing 후 K의 quantization range가 크게 좁아져 INT8이 정확해진다([[sage-attention]]).

**RTX4090에서 측정한 결과** ([[sage-attention]]):
- 341 TOPS (theoretical INT8 peak의 52%)
- **FlashAttention2 대비 2.1×**, **xFormers 대비 2.7×**
- full precision 대비 Cosine similarity: 1.0; Relative L1: 0.019

SageAttention2 (arXiv:2411.10958)는 per-warp quantization과 Smooth Q/Smooth V centering을 사용해 INT4로 확장하며, RTX4090에서 **FA2 대비 3.1×**를 달성한다.

**이것이 inference 전용인 이유와 그 점이 training에 중요한 이유.** SageAttention에는 backward pass가 없다. INT8 quantization은 forward pass에 작은 error(Relative L1 ≈ 0.019)를 유발하며, 이는 inference에는 허용할 수 있지만 training 중에는 누적되어 gradient에 bias를 줄 수 있다. 더 근본적으로 training memory 문제는 backward pass를 위해 activation, 즉 N×N score matrix를 저장하는 문제다. SageAttention의 quantization은 forward pass의 memory bandwidth를 줄이지만 backward-pass activation 저장 필요성을 없애지는 않는다. FA1/FA2/FA3의 recomputation 기법(logsumexp만 저장하고 backward에서 score를 다시 계산)은 이와 독립적이며 training 문제를 해결한다. SageAttention의 INT8 기법은 inference throughput 문제를 해결한다([[sage-attention]]).

**올바른 mental model:** SageAttention은 ~2%의 numerical error를 대가로 얻는 더 빠른 attention *evaluation*이다. FlashAttention은 N×N materialization을 피하는 정확히 동등한 attention *algorithm*이다. 둘은 stack의 서로 다른 수준에서 동작한다.

---

## 5. Ring Attention / Context Parallelism — [[ring-attention]]

**정의.** 논리적 ring으로 배치된 D개 device에 sequence를 shard하여 single-device FlashAttention이 지원할 수 있는 것보다 D배 긴 sequence를 가능하게 하는 multi-device attention algorithm이다(Hao Liu, Matei Zaharia, Pieter Abbeel, ICLR 2024, arXiv:2310.01889). 이것은 production framework가 "context parallelism"(CP)이라고 부르는 것의 algorithmic foundation이다.

**해결하는 문제.** [[ch-05]]의 FlashAttention은 N×N score matrix를 제거하여 device당 O(N) KV activation memory를 제공한다. 그러나 N이 매우 크면(예: 128k 또는 1M token), device당 O(N · d) memory조차 40 GB HBM을 초과할 수 있다. 여기서는 FlashAttention이 도움이 되지 않는다. 이미 불필요한 N×N storage를 모두 제거했기 때문이다. Ring Attention은 이 장벽(device당 HBM capacity limit)을 해결한다.

**Ring topology와 communication** ([[ring-attention]]). D개 device를 논리적 ring으로 배치한다. 각 device는 길이가 L/D인 sequence의 연속된 slice를 소유한다. algorithm의 각 "round"에서는 다음이 일어난다.

1. Device i는 자신의 local Q slice와 현재 보유한 K/V slice 사이의 blockwise attention을 계산한다.
2. 동시에 device i는 자신의 K/V slice를 ring의 device i+1로 보내면서 이전 device i−1의 K/V slice를 받는다.
3. D round가 끝나면 모든 query가 전체 sequence의 모든 key에 attention을 수행한 상태가 된다.

communication(K/V ring rotation)은 GEMM compute 뒤에 pipelining된다. 일반적인 head dimension에서는 compute time ≥ communication time이므로, rotation이 critical path에 추가하는 latency는 0이다. 즉, "추가적인 communication 및 computation overhead가 없다"([[ring-attention]]).

**Memory scaling.** device당 KV activation memory는 O(L/D · d)이며, D가 고정되어 있을 때 L에 대해 constant다. 달성 가능한 전체 context length는 L ∝ D로 scaling된다. 이 논문은 single-device memory-efficient transformer보다 "최대 device 수 배 더 긴" sequence를 "수백만 token의 context size"에서 입증한다. 정확성은 보존된다. D round가 완료되어 모든 query가 모든 key에 attention을 수행하며, blockwise softmax accumulation은 FlashAttention과 동일한 online normalization recurrence를 사용한다([[ring-attention]]).

**Megatron-LM CP와의 연결.** Ring Attention은 이후 production framework가 "context parallelism"이라고 부르는 것의 algorithmic basis다. Megatron-LM의 `--context-parallel-size` flag는 동일한 ring-KV communication pattern을 구현한다. [[ch-07]]에서 다루는 world-size factorization에서는 다음과 같다.

```
world_size = TP × PP × CP × DP
```

CP > 1이면 Ring Attention이 활성 상태라는 뜻이다. CP를 2×씩 늘릴 때마다 D회의 ring-rotation communication round를 대가로 device당 KV memory가 절반으로 줄어든다.

**사용 시점.** FlashAttention을 이미 활성화한 뒤에도 sequence length가 너무 커서 device당 O(N · d) KV memory조차 HBM을 overflow할 때 Ring Attention이 올바른 지렛대(직접 조정할 parallelism 수단)다. 이 시점에서 approximation 없이 선택할 수 있는 방법은 batch size를 줄이거나(training이 비효율적이 될 수 있음), CP로 sequence를 shard하는 것뿐이다. tradeoff는 ring rotation을 위한 all-to-all communication overhead다. 이 overhead는 표준 head dimension에서는 compute 뒤에 숨지만, head dim이 매우 작으면 드러난다.

**Capstone 연결부.** [[ch-09]] capstone은 GDN linear-attention MoE를 다룬다. Linear attention은 softmax kernel을 associative kernel로 대체하여 본래부터 O(N) time과 O(d) memory를 가능하게 한다. 따라서 이 architecture에는 ring-attention 논의 전체가 무관해진다. Linear attention을 사용하면 KV memory가 이미 O(N)이 아니라 O(d)이므로 CP는 이점을 제공하지 않는다. capstone에서는 이 차이를 처리해야 한다.

---

## 6. PagedAttention — [[paged-attention]]

**정의.** KV cache에 OS virtual-memory paging을 적용하여 internal fragmentation을 60–80%의 낭비에서 4% 미만으로 줄이는 inference serving용 KV-cache memory management scheme이다(Woosuk Kwon et al., SOSP 2023, arXiv:2309.06180). **Training 기법이 아니다.** 여기 포함한 이유는 training과 serving의 memory 경계를 정확히 그리기 위한 대조다.

**Inference-serving 문제** ([[paged-attention]]). autoregressive decoding 중에 기존 system(FasterTransformer, Orca)은 request가 시작될 때 request마다 `max_seq_len × d_model × 2 × num_layers × 2 (K+V)` byte의 contiguous buffer를 미리 할당한다. 13B model의 1024-token max context에서는 request당 ~1.7 GB다. sequence가 아직 짧으므로 그 대부분이 낭비된다. internal fragmentation(contiguous buffer에서 사용하지 않는 suffix)과 external fragmentation(길이가 다른 request가 공간을 공유할 수 없음)을 합치면 KV HBM의 60–80%가 낭비된다.

**PagedAttention solution.** KV cache를 고정 크기 **physical block**(~16 tokens/block, 연속으로 저장됨)으로 나눈다. 각 request에는 logical block index → physical block index를 mapping하는 **block table**이 있다. block은 on demand로 할당되며, 공통 prefix가 있는 request 사이에서 공유된다(prefix caching). Beam search의 beam은 갈라질 때까지 physical block을 공유하여(copy-on-write), beam-search KV memory를 O(beam_width × length)에서 O(length + 갈라진 작은 suffix)로 줄인다. 결과는 **≤4% fragmentation**, 동일 latency에서 **FasterTransformer와 Orca 대비 2–4× throughput 개선**이다([[paged-attention]]).

**이것이 training 기법이 아닌 이유.** 차이는 각 context에서 "KV cache"가 의미하는 바에 있다([[paged-attention]]).

- **Training**: batch의 K tensor와 V tensor는 forward pass에서 계산되고 backward pass를 위해 activation으로 저장될 수 있지만, *decoding step에 걸쳐 누적되지는 않는다*. 각 training step은 전체 sequence에 대한 완전한 forward-backward다. page로 관리할, 계속 커지는 KV buffer가 없다.
- **Inference serving**: autoregressive decoding은 모든 이전 token의 K tensor와 V tensor를 누적한다. KV cache는 step마다 커지며 각 active request에 대해 decoding step 사이에도 지속된다. PagedAttention은 이것을 관리한다.

FlashAttention의 기법(backward에서 score matrix를 다시 계산하고 logsumexp만 저장)은 *training activation* optimization이다. PagedAttention은 *serving KV accumulation* optimization이다. 둘은 서로 독립적인 문제를 해결한다. 하나의 model이 둘 다 사용할 수 있다. training 중에는 FlashAttention kernel을, serving 중에는 PagedAttention을 사용한다. 학습자가 두 context 모두에서 "KV cache" 논의를 접할 때 둘을 혼동하는 것은 흔한 오류다.

---

## 7. 구현 간 종합

### Memory behavior 비교표

| Kernel | Setting | Memory complexity | N×N materialized | Backward pass | 주요 제약 |
|---|---|---|---|---|---|
| SDPA/MATH | Training + Inference | O(N²) | Yes — HBM의 전체 score matrix | Yes | 없음(항상 사용 가능) |
| SDPA/FLASH_ATTENTION | Training | O(N) | No — backward에서 재계산 | Yes | CUDA, fp16/bf16, head_dim ≤ 128, 임의의 bias 불가 |
| SDPA/EFFICIENT_ATTENTION (xFormers) | Training | O(N) | No — streaming accumulation | Yes | 더 넓은 범위: custom bias, 큰 head_dim |
| SDPA/CUDNN_ATTENTION | Training | O(N) | No | Yes | cuDNN version 요구 사항 |
| SageAttention INT8 | Inference only | O(N) forward | No — streaming | **No** | Inference only; ~2% numerical error |
| Ring Attention / CP | Training | device당 O(L/D) | No — blockwise | Yes | Multi-device ring; communication overhead |
| PagedAttention | Inference serving only | O(KV cache, paged) | N/A | **No** | KV accumulation 관리, training 아님 |

### Invariant와 variant

**Invariant**(substrate가 강제함): 모든 exact attention kernel은 softmax(QKᵀ/√d)V를 계산해야 한다. N² work는 compute에서 줄일 수 없다. 줄일 수 있는 것은 *HBM memory*, 즉 N×N score matrix를 HBM에 아예 쓰는지 여부다. online softmax recurrence(Milakov & Gimelshein 2018, [[online-softmax]])는 single-pass streaming을 가능하게 하는 보편적 primitive다. `m_new = max(m_old, x_k)`, `d_new = exp(m_old − m_new) · d_old + exp(x_k − m_new)`. 모든 O(N) kernel(FA1/FA2/FA3, xFormers FMHA, Ring Attention blockwise)은 이 recurrence를 사용한다.

**Variant**(구현을 구분하는 자유로운 design choice):
- *work partitioning의 granularity*: FA1은 K/V를 warp 사이에 나눈다(softmax denominator를 위해 shared-memory sync 필요). FA2는 Q를 warp 사이에 나눈다(각 warp가 output row를 독립적으로 소유하여 sync 제거). 이로써 A100에서 MFU가 25–40%에서 50–73%로 두 배가 된다.
- *Hardware specialization*: FA3는 Hopper TMA producer-consumer warp specialization을 추가하여 FP16에서 740 TFLOPs/s(75% utilization), FP8에서 ~1.2 PFLOPs/s에 도달한다. xFormers FMHA는 더 낮은 peak throughput으로 더 넓은 범위를 지원하기 위해 CUTLASS 2.x를 사용한다.
- *Precision*: SageAttention은 inference를 위해 QKᵀ를 INT8로 quantize한다(K-channel smoothing 사용). FA3는 training을 위해 incoherent processing(quantization 전 Hadamard rotation, naive FP8보다 2.6× 낮은 error)을 적용한 FP8 block quantization을 사용한다.
- *sharding의 범위*: Ring Attention은 O(N) 체제를 D개 device에 걸쳐 확장하여 device당 memory가 O(L/D)가 되게 하며, single-device FlashAttention이 해결할 수 없는 문제를 해결한다.
- *Dispatch policy*: PyTorch SDPA는 silent fallback semantics를 가진 indirection layer를 추가한다. 그 결과 fail-fast behavior보다 backward compatibility를 우선한 결과로 silent-MATH-fallback OOM 함정(오류 없이 MATH로 전환되어 OOM을 유발하는 조건)이 생긴다.

이 동물원(여러 kernel 구현의 집합)이 답하는 근본적인 design question은 다음과 같다. 고정된 수학 computation(exact softmax attention)이 주어졌을 때 느린 HBM memory를 통과해야 하는 data의 최소량은 얼마이며, hardware constraint, multi-device topology, inference-vs-training context를 추가하면 그 최소량은 어떻게 달라지는가?

---

## 문헌의 핵심 통찰

1. **Silent fallback은 transformer training에서 가장 위험한 default다.** [[pytorch-sdpa]]는 명시적인 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`이 없으면 지원되지 않는 input 때문에 PyTorch가 경고 없이 38× 느리고 O(N²) memory를 사용하는 MATH backend로 조용히 fallback한다고 설명한다. 올바른 contract는 명시적 backend selection이 지원되지 않는 input에서 `RuntimeError`를 발생시키고, default dispatch는 error를 삼키는 것이다. 실제 dispatch를 검증하지 않고 config flag를 통해 "FlashAttention을 활성화"하는 codebase는 edge-case input에서 아무 표시 없이 OOM을 일으킨다.

2. **Online softmax recurrence는 보편적 primitive다.** 모든 O(N) attention kernel, 즉 FA1, FA2, FA3, xFormers FMHA, Ring Attention의 blockwise accumulation은 Milakov & Gimelshein의 single-pass recurrence에 의존한다([[online-softmax]]). 이것이 없으면 tiling은 renormalization을 위해 두 번째 HBM read를 필요로 하여 O(N²) bandwidth를 다시 도입하게 된다. recurrence는 element당 네 번의 arithmetic operation이다. 이것이 single-pass streaming을 수치적으로 exact하게 만든다.

3. **Training memory와 inference memory는 서로 독립적인 문제다.** [[paged-attention]]과 [[sage-attention]]은 둘 다 중요한 방식으로 memory를 줄이지만, 어느 것도 training과 관련되지 않는다. PagedAttention은 decoding step에 걸쳐 누적되는 *persisted* KV cache를 page로 관리한다. 이 structure는 training에는 존재하지 않는다. SageAttention은 attention의 *forward-pass evaluation*을 가속하지만 backward pass가 없다. 따라서 training loop에 참여할 수 없다. [[flash-attention-1]]의 backward recomputation(logsumexp를 저장하고 Q, K, V로 score를 재계산)은 attention layer의 training activation memory를 줄이는 것으로 알려진 유일한 exact 기법이다.

4. **Sequence sharding(Ring Attention)은 device당 O(N) memory 자체가 OOM을 일으킬 때 유일한 해결책이다.** N이 매우 클 때 FlashAttention의 O(N) 체제가 이야기의 끝(더 이상의 memory 문제가 없는 최종 상태)은 아니다. [[ring-attention]]은 ring의 D개 device에서 device당 memory가 O(L/D)로 scaling되어 approximation과 추가 latency 없이 sequence를 D× 더 길게 만들 수 있음을 보여 준다(communication은 GEMM 뒤에 pipelining됨). 이것은 Megatron-LM context parallelism(`--context-parallel-size`)의 algorithmic basis이며, [[ch-07]]에서 전체 parallelism taxonomy 안에서 다룬다.

---

## 핵심 요점

- `torch.nn.functional.scaled_dot_product_attention`은 kernel이 아니라 dispatcher다. production code에서는 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`으로 backend를 명시적으로 고정하라. MATH로의 silent fallback = silent OOM이다.
- 네 backend는 MATH(O(N²), 항상 사용 가능), FLASH_ATTENTION(O(N), 엄격한 제약), EFFICIENT_ATTENTION/xFormers(O(N), 더 넓은 범위 지원), CUDNN_ATTENTION(O(N), cuDNN 의존)이다. MATH와 optimized backend 사이의 ~38× performance gap은 전적으로 N×N score matrix의 HBM bandwidth cost 때문이다.
- xFormers `memory_efficient_attention`은 Rabe & Staats 2021(arXiv:2112.05682)을 구현한다. O(√N) practical variant이며, N=16,384에서 inference memory는 59×, backpropagation memory는 32× 줄인다. FA2의 제약(head_dim, bias shape)을 충족할 수 없을 때 올바른 선택이다.
- SageAttention: QKᵀ에는 INT8(K-channel smoothing 사용: channel별 mean을 빼면 softmax에서 상쇄됨), P·V에는 FP16을 사용하며, RTX4090에서 FA2보다 2.1× 빠르다. **Inference only** — backward pass가 없어 training에는 적용할 수 없다.
- Ring Attention: ring의 D개 device, device당 KV memory O(L/D), communication overhead 0(rotation이 GEMM 뒤에 pipelining됨), exact(D round 후 모든 Q가 모든 K에 attention 수행). 이것은 `world_size = TP × PP × CP × DP`에서 Megatron-LM의 `--context-parallel-size` = CP다.
- PagedAttention: 고정 크기 physical block(~16 token)을 on demand로 할당하며, 기존 system의 60–80% 낭비와 비교해 fragmentation이 ≤4%이고 serving throughput은 2–4×다. **Inference serving only** — 대응하는 training 기법은 없다. 이것이 page로 관리하는 "KV cache"는 decoding step에 걸쳐 커지는 persisted structure이며, training에는 존재하지 않는다.
- [[ch-09]] capstone의 경우 GDN linear attention은 본래 O(d) KV memory를 사용하므로 Ring Attention / CP는 이점을 제공하지 않는다. 전체 FlashAttention kernel 계열은 무의미하다. 그러나 같은 model의 standard attention layer에는 여전히 SDPA dispatch와 그 OOM 함정(조용한 MATH fallback으로 인한 OOM)이 적용된다.

---

## References

- PyTorch SDPA Tutorial — https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html ([[pytorch-sdpa]])
- facebookresearch/xformers — https://xformers.org/ · https://github.com/facebookresearch/xformers ([[xformers-mem-efficient]])
- Markus N. Rabe and Charles Staats. "Self-attention Does Not Need O(n²) Memory." arXiv:2112.05682 (2021) — https://arxiv.org/abs/2112.05682
- Jintao Zhang et al. "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration." ICLR 2025, arXiv:2410.02367 — https://arxiv.org/abs/2410.02367 ([[sage-attention]])
- Jintao Zhang et al. "SageAttention2." arXiv:2411.10958 — https://arxiv.org/abs/2411.10958
- Hao Liu, Matei Zaharia, Pieter Abbeel. "Ring Attention with Blockwise Transformers for Near-Infinite Context." ICLR 2024, arXiv:2310.01889 — https://arxiv.org/abs/2310.01889 ([[ring-attention]])
- Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023, arXiv:2309.06180 — https://arxiv.org/abs/2309.06180 ([[paged-attention]])
- Maxim Milakov and Natalia Gimelshein. "Online normalizer calculation for softmax." arXiv:1805.02867 (2018) — https://arxiv.org/abs/1805.02867 ([[online-softmax]])
- Tri Dao et al. "FlashAttention." NeurIPS 2022, arXiv:2205.14135 — https://arxiv.org/abs/2205.14135 ([[flash-attention-1]])
- Tri Dao. "FlashAttention-2." ICLR 2024, arXiv:2307.08691 — https://arxiv.org/abs/2307.08691 ([[flash-attention-2]])
- Jay Shah et al. "FlashAttention-3." arXiv:2407.08608 (2024) — https://arxiv.org/abs/2407.08608 ([[flash-attention-3]])

---

## 질문

1. [[pytorch-sdpa]] 발췌문에 따르면 MATH fallback은 ~38× 느리고 전체 O(N²) score matrix를 할당하지만, fallback은 조용히 일어난다. bug가 아니라 일부 model의 정당한 architectural choice인 어떤 구체적인 input 속성이 "FlashAttention을 활성화한" codebase에서도 이 fallback을 유발하는가? production에서 OOM을 만나기 전에 이를 어떻게 감지하겠는가?

2. [[sage-attention]]은 QKᵀ를 INT8로 quantize하여 RTX4090에서 FA2 대비 2.1×를 달성하지만, A100의 FA2는 N×N HBM write를 제거하여 이미 50–73% MFU에 도달한다. 이 두 speedup 주장은 양립 가능한가? 각각 무엇을 optimize하며, SageAttention의 기법이 A100 training run으로 이전되지 않는 이유는 무엇인가?

3. Ring Attention은 KV rotation을 GEMM 뒤에 pipelining하므로 "추가적인 communication 및 computation overhead가 없다"고 보장한다. 이 pipelining이 무너져 communication이 실제로 bottleneck이 되는 조건을 설명하고, 회복하기 위해 어떤 parameter를 조정할지 설명하라.

4. Rabe & Staats 논문은 세 가지 exact memory variant인 query당 O(1), O(log N), O(√N)을 제시한다. xFormers FMHA는 GPU/TPU tile parallelism을 위해 O(√N) variant를 구현한다. [[ch-04]]에서 배운 online softmax recurrence를 바탕으로, O(1) variant를 O(√N) tile variant와 같은 방식으로 warp에 걸쳐 효율적으로 parallelize할 수 없는 이유를 기계적 동작 수준에서 설명하라.

5. PagedAttention은 ~16 token의 physical block을 사용하여 KV fragmentation을 60–80%의 낭비에서 ≤4%로 줄인다. block이 1 token인 naive re-implementation은 fragmentation 0%를 달성할 것이다. 실용적인 선택이 1 token이 아니라 ~16 token의 block size인 이유는 무엇이며, 어떤 두 hardware constraint가 이 수치를 결정하는가?

6. [[ch-09]] capstone model은 GDN linear attention을 사용한다. Linear attention은 본래 O(N)이 아닌 O(d) KV memory를 사용한다는 점을 고려하여, §7 비교표의 어떤 항목이 무관해지고 어떤 항목이 여전히 관련되는지 추적하라. 또한 이 architecture에서 O(N) activation 우려를 대체하는 새로운 memory 문제가 있다면 무엇인지 설명하라.
