<!-- chapter: ch-09
     track: capstone
     kind: content
     title: Capstone: Modeling a 27B MoE Memory Budget End-to-End
     deps: [[ch-01]], [[ch-02]], [[ch-03]], [[ch-04]], [[ch-05]], [[ch-06]], [[ch-07]], [[ch-08]]
     sources: [[zero-memory-optimization]], [[deepspeed-moe-ep]], [[transformer-math-101]], [[megatron-tp-sp]], [[pipeline-parallelism-1f1b]], [[selective-recompute-korthikanti]], [[liger-fused-ce]], [[training-oom-failure-modes]], [[pytorch-fsdp]], [[memory-calculator-notes]], [[ultrascale-playbook]]
-->

# Chapter 9 — 종합 과제: 27B MoE Memory Budget을 End-to-End로 모델링하기

> **핵심 통찰.** 이 과정에서 다루는 모든 memory 제약은 기반 환경(substrate, 시스템을 떠받치는 하드웨어·아키텍처 기반)의 결과다. 여기에는 A100의 40 GB HBM 한계, TP degree의 상한을 정하는 NVLink bandwidth, 순진한 ZeRO-3 + MoE 조합을 파국적으로 만드는 all-to-all topology, CP를 엄격히 금지하는 linear-attention contract가 포함된다. 하나의 구체적인 model을 대상으로 여섯 항목 ledger에서 최종 per-GPU 판정까지 차근차근 계산하면, 이러한 기반 환경의 작용을 명확히 읽을 수 있고 과정 전체가 하나의 일관된 decision tree로 통합된다.

> **지침.** 다음 순서로 budget을 산정하라. (1) Rule of 16으로 static model-state floor를 계산한다. (2) MoE topology 제약을 진단한다(EP가 필요하며, ZeRO-3 expert gather는 OOM을 일으킨다). (3) 주어진 architecture에서 허용되는 유일한 attention kernel을 선택한다. (4) activations + logit spike의 크기를 산정한다. (5) 전체 parallelism plan을 구성한다. (6) per-GPU 적합 여부를 검증한다. 이 hardware에서 마주칠 모든 OOM은 이 여섯 단계 중 하나에 대응한다.

---

## 1. Model과 Cluster

**Model 사양:**
- 총 27B parameters (Ψ = 27 × 10⁹)
- 256-expert MoE. parameters 대부분은 expert FFN blocks에 있다(일반적인 분할: ~3B dense attention + embedding, ~24B expert weights).
- GDN linear-attention blocks — **hard-assert CP = 1** (이 architecture에서는 context parallelism이 허용되지 않는다. §4 참조)
- Vocabulary size V = 248,000 (대규모 multilingual vocab)
- Training sequence length S = 32,768 tokens (32k)
- Cluster: A100-40GB (H100이 아님 — **fp8을 사용할 수 없으며**, Transformer Engine은 Hopper+에서만 실행됨)

**도출된 geometry** (표준 MoE layer sizing을 가정):
- Hidden dimension h = 4,096
- Attention heads a = 32, head dim = 128
- Layer 수 L = 32 (16 attention + 16 MoE blocks가 번갈아 배치됨)
- FFN expansion: 각 expert의 d_ff = 4h = 16,384

---

## 2. Step 1: 여섯 항목 Ledger와 Rule of 16

[[ch-01]]의 taxonomy는 memory에 상주하는 여섯 항목을 제시한다. [[ch-08]]의 calculator formula는 다음과 같다.

```
M_total = (16Ψ/N) + A_layers + L_spike + FSDP_allgather_buffer + CUDA_overhead
```

### Rule of 16

[[transformer-math-101]]과 [[ultrascale-playbook]]에 따르면 mixed-precision Adam은 다음과 같이 분해된다.

```
bf16 working weights:     2 bytes/param
bf16 gradients:           2 bytes/param
fp32 master copy:         4 bytes/param
fp32 Adam momentum:       4 bytes/param
fp32 Adam variance:       4 bytes/param
─────────────────────────────────────────
Static floor (model states): 16 bytes/param
```

ZeRO-3를 사용하는 N개의 DP ranks에 Ψ = 27B parameters가 있을 때:

```
Model states = 16 × 27×10⁹ / N  bytes
             = 432 GB / N
```

A100-40GB 한 장에서는 이 값이 **432 GB**로, card capacity의 10.8배다. Model states를 처리하기 위해 실행 가능한 최소 전략은 ZeRO-3(또는 FSDP FULL_SHARD)다.

**ZeRO scaling 확인** ([[zero-memory-optimization]]의 7.5B@Nd=64를 기준으로 사용):
ZeRO-3 formula는 `16Ψ/Nd`다. 27B model과 Nd=64 DP ranks일 때:
```
16 × 27B / 64 = 6.75 GB  of model states per GPU
```
이는 activations와 logit spike를 위한 여유를 남기면서 40 GB card에 들어간다.

### Static Floor에서의 LoRA vs Full Fine-Tuning

[[memory-calculator-notes]]에 따르면:

> "LoRA의 핵심 통찰: LoRA memory에서는 activations가 지배적이다. 'LoRA fine-tuning 중 주된 memory 소비는 LoRA parameters가 아니라 frozen weights의 activation gradients에서 발생한다.' LoRA optimizer states는 무시할 수 있을 정도로 작으며, 동일한 s와 b에서 activation budget은 full fine-tune과 같다."

| 전략 | Model states/GPU | Activations/GPU |
|----------|-----------------|-----------------|
| Full-FT, ZeRO-3, N ranks | 16Ψ/N | 동일함(같은 fwd pass) |
| LoRA, frozen base | ~2Ψ (전체가 load됨) + 작은 adapter | full-FT와 **동일함** |

이것이 [[ch-08]]의 LoRA node-invariance 교훈이다. DP replicas를 추가하면 ZeRO-3에서 model-state cost가 절반으로 줄지만, **모든 replica가 동일한 S와 b로 같은 forward pass를 실행**하므로 per-GPU activation memory는 줄어들지 않는다. Model states 때문에 발생한 Full-FT OOM은 nodes(더 큰 N)를 추가하면 해결되지만, activations 때문에 발생한 LoRA OOM은 nodes를 추가해도 **해결되지 않는다**.

---

## 3. Step 2: 순진한 ZeRO-3가 OOM을 일으키는 이유 — Expert Parallelism 필요

### Expert Gather 문제

ZeRO-3는 모든 parameters를 전체 N ranks에 걸쳐 shard한다. Dense model에서 "forward 이전 all-gather"는 transformer layer 하나의 weights(~수백 MB)를 재구성한다는 뜻이며, 이는 감당할 수 있다. 하지만 24B expert weights를 가진 256-expert MoE에서는 ZeRO-3의 all-gather가 각 MoE layer의 forward pass 전에 모든 rank에서 전체 expert parameters를 동시에 재구성한다. [[deepspeed-moe-ep]]에 따르면:

> "Expert parallelism (EP)은 expert weight matrices를 EP_size GPUs에 걸쳐 shard하여 각 rank가 E/EP_size experts만 보유하게 한다. tokens는 한 쌍의 all-to-all collectives(dispatch + combine)를 통해 올바른 rank로 routing된다."

순진한 ZeRO-3(EP 없음)를 사용하면:
- MoE forward에서 각 rank는 전체 24B expert weights를 all-gather하며, bf16에서 약 **48 GB**다. Layer가 실행되기도 전에 40 GB card capacity를 초과한다.
- Peak는 `2 × P_unit`(FSDP all-gather buffer)이다. 24B experts에서 buffer는 ~48 GB이므로 OOM이 반드시 발생한다.

### Expert Parallelism이 이를 해결한다

EP_size = 8(A100 8개로 구성된 node 하나)을 사용하면:
- 각 rank는 256/8 = **32 experts**를 보유한다.
- Per-GPU expert weight memory = 24B / 8 = **3 GB**로, budget 안에 충분히 들어간다.
- Routing: MoE layer마다 두 번의 all-to-all collectives가 발생하며, buffer size = `tokens × d_model × 2 bytes`다.
  ```
  At S=32768, b=1, d_model=4096:
  Buffer = 32768 × 4096 × 2 × 2 bytes ≈ 0.5 GB (two collectives)
  ```
  감당할 수 있는 크기다. Sequence × batch에 비례하므로 b > 1에서는 주의해야 한다.

### EP에는 Hybrid Parallelism Plan이 필요하다

[[deepspeed-moe-ep]]에 따르면:

> "DeepSpeed-MoE hybrid parallelism (Rajbhandari 2022): sparse experts에는 EP + DP를, dense layers에는 TP를, optimizer states에는 ZeRO를 결합한다. Trillion-parameter 규모에서 MoE를 훈련한 최초의 framework다."

Non-expert layers(attention, embeddings, layer norms)는 sparse하지 않으므로 EP를 사용할 수 없다. 이들은 TP가 필요하거나 DP에서 복제된다. 실용적인 plan은 dense layers에 대해 node 내부에서 TP_size를 적용하고, expert layers에 대해 같은 node(또는 여러 nodes)에 걸쳐 EP_size를 적용하며, node groups 바깥쪽에 DP를 적용하는 것이다.

---

## 4. Step 3: Attention Kernel — GDN Linear Attention이 강제하는 사항

### CP Hard-Assert

Context parallelism(Ring Attention)은 각 device가 sequence의 1/D을 보유한 상태에서 D개 devices로 구성된 ring을 따라 K/V blocks를 회전시키는 방식으로 작동한다. [[ring-attention]]에 따르면:

> "각 device는 길이 L/N인 연속된 sequence slice를 소유한다. 각 'round'에서 각 device는 (a) 자신의 local Q slice와 현재 보유한 K/V slice 사이의 blockwise attention을 계산하고, (b) 자신의 K/V slice를 ring의 다음 device로 보낸다."

GDN linear-attention은 sequence 전체에 걸친 recurrent formulation으로 attention을 계산한다. 각 state는 순서상 앞선 모든 states에 의존하며, 위치 t의 output은 key-value outer products의 누적 합에 대한 함수다. 이 recurrence는 sequential dependency를 깨뜨리지 않고서는 **block으로 나누어 ring을 따라 회전시킬 수 없다**. Hard-assert CP=1은 선택 사항이 아니라 architectural contract다.

**A100-40GB의 GDN linear attention에서 허용되는 kernels:**

| Kernel | GDN에서 허용되는가? | 이유 |
|--------|---------------|--------|
| Standard O(N²) | 예 | Ring이 필요 없지만 32k에서 OOM 발생 |
| FlashAttention (1/2/3) | 아니요 | FA는 표준 softmax attention을 가정하므로 linear-attention formulation과 호환되지 않음 |
| xFormers memory-efficient | 아니요 | 동일하게 softmax attention block structure를 가정함 |
| Ring/Context Parallel | 아니요 | Hard-assert CP=1이 이를 금지함 |
| SageAttention | 아니요 | Softmax attention을 위해 설계된 quantized approximation임 |
| PagedAttention | 아니요 | Inference 전용 KV-cache management이며 training kernel이 아님 |
| **GDN-specific linear-attention kernel** | **예** | Model이 자체 kernel을 제공하며, per-layer memory cost는 O(N²)이 아니라 O(N×d)임 |

GDN kernel의 memory cost는 layer마다 O(N×d)이며, 이는 recurrent state(variant에 따라 d×d matrix 또는 d-dim vector)다. d=4096, S=32768일 때:

```
Linear-attention state per layer: 4096² × 2 bytes ≈ 32 MB per layer
Across L=16 attention layers: ~512 MB total
```

이를 32k의 vanilla attention과 비교하면 32768² × 2 × 32 heads × 16 layers × 2 (S and P) ≈ **4 TB**다. 여기서는 linear-attention architecture 자체가 memory 절감 역할을 한다. Kernel은 중간 O(N²) tensors를 materialize하지 않고 recurrence를 올바르게 구현하기만 하면 된다.

**Activation memory에 대한 핵심 결과**: GDN kernel 자체의 memory formula를 사용하라. Megatron의 `sbh(10 + 24/t + 5as/ht)` formula([[megatron-tp-sp]])는 softmax attention에 맞춰 보정되었으며, 5as/h·t 항(N² attention score 영역)은 여기에는 **적용되지 않는다**. 다음을 사용한다.

```
A_layer ≈ s × b × h × C_linear  (where C_linear is the kernel-specific constant, typically 10-14)
```

Quadratic term이 없으면 32k에서 activation memory를 지배하는 것은 attention scores가 아니라 MLP / FFN activations다.

---

## 5. Step 4: Logit Spike — 32k Sequence에서 248k Vocab

### Spike Formula

[[liger-fused-ce]]와 [[memory-calculator-notes]]에 따르면 logit buffer spike는 forward-backward 경계에서 발생한다.

```
L_spike = vocab_size × seq_len × batch × dtype_bytes
         = 248,000 × 32,768 × 1 × 2  bytes (bf16)
         = 16,273,408,000 bytes
         ≈ 30.4 GB
```

**이는 파국적이다.** 40 GB A100에서는 model weights, activations, optimizer states를 계산하기도 전에 logit spike만으로 card memory의 76%를 소비한다.

[[training-oom-failure-modes]]에 따르면:

> "Logit spike 식별: `lm_head` / `output_projection`에서 OOM 발생 → 원인은 `vocab_size × seq_len × batch × 2` bytes 크기의 logit buffer다. 해결책: ZeRO stage가 아니라 seq_len 또는 batch를 줄인다."

ZeRO-3, TP, PP는 logit spike를 줄이지 **못한다**. 이는 lm_head를 보유한 rank에서 locally-assembled activation을 바탕으로 계산되며, sharding strategy와 관계없이 일시적으로 할당되기 때문이다.

### 필수 해결책: Fused Cross-Entropy

[[liger-fused-ce]]에 따르면:

> "표준 cross-entropy loss는 loss를 계산하기 전에 전체 `(B·T × V)` logit tensor를 materialize한다. 이는 크기가 보통인 batch/sequence/vocab에서도 1+ GB에 이르는 spike다. Liger의 fused kernel은 token dimension을 따라 chunking하고 하나의 Triton kernel 안에서 linear projection과 loss computation을 fuse하여 이 spike를 제거하며, peak activation memory를 ~80% 줄인다."

Liger `LigerFusedLinearCrossEntropyLoss`를 사용하면 최대 transient allocation은 chunk 하나다.

```
Chunk size (typical CUDA): 65,536 tokens
Max transient = 65536 × 248000 × 2 bytes ≈ 32.5 GB
```

그래도 card의 81%다. 더 작은 chunk size(예: 이 제약이 큰 hardware에서 2048 tokens)를 사용하면:

```
Chunk = 2048 × 248000 × 2 bytes ≈ 1.0 GB
```

**이 model에서 Fused-CE는 선택 사항이 아니다.** Liger 또는 동등한 chunked kernel 없이 32k sequence에서 248k vocab을 사용하면, 다른 모든 optimization과 관계없이 어떤 40 GB card에서도 OOM이 발생한다.

> **주의.** Logit spike는 lm_head가 tensor-parallel인 경우에도 TP sharding이 해결하지 **못하는** 유일한 memory 항목이다. TP는 lm_head의 weight matrix를 shard하지만, output logits는 softmax를 위해 gather되어야 한다(또는 fused kernel이 shard를 고려해야 한다). Fused-CE kernel이 TP-aware인지 확인하거나, 각 rank 내부에서 TP-shard와 fuse를 수행하라(vocabulary parallelism).

---

## 6. Step 5: Activation Memory와 Recomputation

### MLP / FFN Activations (지배적인 항)

GDN model에서는 FFN blocks가 activation memory를 지배한다. TP degree t일 때:

```
A_FFN_per_layer = s × b × h × (C_mlp / t)   per rank
```

MoE FFN에서는 각 token이 expert 하나(top-1) 또는 둘(top-2)로 routing된다. Memory는 d_ff = 16,384 크기의 dense FFN과 비슷하지만 routing에 의해 gate된다. 최악의 경우 모든 tokens가 local experts에 도달하며 activation peak는 다음과 같다.

```
A_MoE_expert_layer ≈ (tokens_local × d_ff × 2 bytes) per EP rank
                   = (32768 / EP_size) × 16384 × 2 bytes
                   = (32768/8) × 16384 × 2 = 4096 × 16384 × 2 ≈ 0.13 GB
```

16개 MoE layers에서 expert activations는 **~2 GB**다(EP가 tokens를 분산하므로 비교적 작음).

### Selective Recomputation

[[selective-recompute-korthikanti]]에 따르면:

> "Selective activation recomputation: 각 transformer layer 안에서 memory cost는 높지만 recompute cost는 낮은 operations를 식별한다. Attention score computation(s×s matrices에 대한 softmax, dropout)은 버렸다가 다시 계산하고, MLP, LayerNorm, projection outputs는 유지한다. Memory 결과: activation memory 소비가 5배 감소한다. Compute 결과: recomputation으로 인한 execution time overhead가 90% 이상 감소한다(~30–40%에서 <4%로)."

GDN linear attention에서 "attention score computation"은 recurrent state update이며, MLP matmuls보다 recompute 비용이 낮다. Selective recomputation을 적용하여 각 attention block 내부의 intermediate recurrence states는 버리고 MLP activations는 유지한다. 그 결과 <2% compute overhead로 attention activation 부분에서 ~5배를 절감한다.

### Activation 추정치 (Recompute 없음, TP=t, SP 활성화)

[[megatron-tp-sp]]에 따르면 SP를 사용할 때:

```
A_layer (non-linear-attn portion) = (s × b × h / t) × 34  bytes
```

s=32768, b=1, h=4096, t=4(node 내부의 TP degree)일 때:

```
A_layer = (32768 × 1 × 4096 / 4) × 34 = 33554432 × 34 ≈ 1.08 GB per layer
```

L=32 layers 전체에서 recomputation이 없으면 **~34.6 GB**다. 5× selective recompute를 적용하면 **~7 GB**다. 이는 model states와 함께 여유 있게 들어간다.

---

## 7. Step 6: 대표적인 OOM 진단

### Scenario: Full-FT, ZeRO-3만 사용, EP 없음, Batch=1

[[ch-08]]의 debugging loop를 사용한다.

**Step 1 — OOM message 읽기:**
```
RuntimeError: CUDA out of memory. Tried to allocate 48.00 GiB
(GPU 0; 39.59 GiB total capacity; 6.75 GiB already allocated;
 32.59 GiB free; ...)
Traceback: ... inside MoE forward, expert_linear.weight ...
```

**Step 2 — 추정:**
요청량은 48 GB다. 이는 `2 × P_experts = 2 × 24B params × 2 bytes/param = 96 GB`... 잠깐, 그 값은 전체 expert weight block에 대한 FSDP all-gather buffer다. 48 GB의 단일 allocation은 정확히 expert all-gather tensor에 해당한다. 즉, bf16으로 전체 24B expert params를 재구성한 크기인 `24×10⁹ × 2 = 48 GB`다.

**Step 3 — Lever 식별:**
[[training-oom-failure-modes]]에 따르면:

> "`lm_head` / `output_projection`에서 traceback → logit spike. `loss.backward()`에서 traceback → activation peak. `optimizer.step()`에서 traceback → optimizer-state memory."

여기서 traceback은 expert weight all-gather가 수행되는 MoE forward 내부에 있다. Phase: forward pass, expert layer. 원인: ZeRO-3 all-gather가 모든 expert weights를 동시에 재구성함.

**Step 4 — 정확한 lever 적용:**
Lever는 ZeRO stage 조정이 아니라 **Expert Parallelism**(EP)이다. 나머지 parameters에는 ZeRO-3를 피할 수 없지만, expert weights는 ZeRO-3 shard에서 분리하여 EP로 관리해야 한다. DeepSpeed-MoE와 NeMo는 이 hybrid를 native로 구현한다.

---

## 8. 최종 Parallelism Plan과 Per-GPU 판정

### 제약 요약

| 제약 | 출처 | 강제되는 사항 |
|------------|--------|--------|
| A100 40 GB 한계 | Cluster | ZeRO-3 필수, headroom 없음 |
| fp8 없음 | Volta/Ampere limitation | bf16 master copy 필요, 16 bytes/param floor |
| 256-expert MoE | Architecture | EP 필요, 순진한 ZeRO-3는 OOM 발생 |
| GDN CP=1 | Architecture hard-assert | Ring Attention 없음, CP dimension 없음 |
| V=248k, S=32k | Model 사양 | Fused-CE 필수(unfused 시 ~30 GB spike) |

### Parallelism 할당

```
world_size = TP × PP × EP × DP
```

- **TP = 4** (NVLink node 내부의 dense attention/embedding layers에 적용. 일반적인 최댓값은 8×이지만, 4는 node 내부에 EP를 위한 여유를 남김)
- **PP = 1** (pipeline parallelism은 bubble overhead를 추가하고 MoE all-to-all routing을 복잡하게 하므로, 이 cluster size에서는 flat 구성을 선호함)
- **EP = 8** (8 GPUs로 구성된 A100 node 하나가 전체 256 experts를 보유하며, GPU당 32개를 보유함)
- **DP = 8** (8 nodes = 총 64 GPUs. 각 EP domain 내부의 8-GPU DP groups에 걸쳐 optimizer states를 ZeRO-3로 sharding하는 outer data parallelism)

```
Total GPUs: TP × EP × DP = 4 × 8 × 8 = 256 GPUs (32 nodes of 8 A100-40GB)
```

조정안: budget이 더 작다면 PP=2를 사용하여 per-rank weight memory를 절반으로 줄이되, m=20 microbatches에서 (PP-1)/microbatches ≈ 5%의 bubble penalty를 감수한다.

### Per-GPU Memory 세부 내역 (DP=8, TP=4, EP=8, PP=1)

```
Component                                    GB/GPU
─────────────────────────────────────────────────────
Dense model states (ZeRO-3, N=DP×TP=32):  16×3B/32     = 1.5  GB
Expert model states (EP=8, no ZeRO-3):    16×24B/8/...
  ↳ Only 1 expert replica per rank: 24B×2B/8           = 6.0  GB  (bf16 params only)
  ↳ Adam states (fp32) on shard:    24B×12B/8/DP_ep    = 4.5  GB  (if ZeRO-3 over DP)
  ↳ Expert subtotal:                                   ≈ 10.5 GB
Activations (selective recompute, TP=4):                ≈ 7.0  GB
All-to-all MoE buffer (tokens×d×2):                    ≈ 0.5  GB
FSDP AllGather buffer peak (transient):                 ≈ 1.5  GB
Logit spike (fused-CE, chunk=2048):                     ≈ 1.0  GB
CUDA context + fragmentation:                           ≈ 1.0  GB
─────────────────────────────────────────────────────
TOTAL:                                                 ≈ 23.0 GB
```

**판정: ~17 GB headroom을 남기고 A100-40GB에 들어간다.** 따라서 40 GB 한계 안에 머물면서 batch_size=2를 사용할 수 있다(activation과 logit-chunk cost가 두 배가 되어 ~8 GB 추가).

> **주의.** 여기서 full-FT 대신 LoRA를 실행하면 optimizer state와 gradient 항목은 거의 0으로 줄지만 activation 항목과 logit-spike 항목은 그대로다. 총량은 ~23 GB에서 ~18 GB로 줄어든다. 이 5 GB 절감은 activations가 아니라 ZeRO-3 model states에서 나온다. LoRA는 이 hardware에서 더 큰 sequence나 batch를 가능하게 하지 않는다.

---

## 9. LoRA vs Full-FT: 1→2 Nodes가 Full-FT는 해결했지만 LoRA는 해결하지 못한 이유

이는 [[ch-08]]의 node-invariance 통찰을 구체적으로 검증하는 사례다. 원래 run이 **1 node (8 GPUs, TP=4, DP=2, EP=4)**에서 수행된다고 가정하자.

### 1 Node에서의 Full-FT OOM

```
N_DP = 2 (two DP ranks within the node)
Dense model states = 16 × 3B / 2 = 24 GB    ← OOM
```

두 번째 node를 추가하면 N_DP가 4로 두 배가 되고 dense model states는 12 GB로 줄어 OOM이 해결된다. 이것이 가능한 이유는 **ZeRO-3가 model states를 N_DP로 나누고**, DP scaling이 넘치는 항을 직접 줄이기 때문이다.

### 1 Node에서의 LoRA OOM

동일한 1-node config에서 LoRA를 사용하면 optimizer states와 gradients는 ~0.1 GB(adapters만 해당)로 줄어든다. OOM은 activations에서 발생한다.

```
Activations (selective recompute, TP=4, s=32k, b=1) ≈ 7 GB per rank
+ logit spike (unfused, 248k×32k×1×2) ≈ 30 GB      ← OOM
```

두 번째 node를 추가해도 각 DP replica는 **같은 tokens**를 **같은 sequence length**와 **같은 batch size**로 처리한다. Per-GPU activations는 7 GB로 변하지 않고, per-GPU logit spike도 30 GB로 변하지 않는다. 추가 node는 activation / logit 문제에 도움이 되는 DP ranks를 하나도 추가하지 않은 셈이다. 이미 LoRA에서 문제가 없던 model states에만 도움이 되었다.

**LoRA OOM 해결책:** fused-CE(30 GB spike 제거) + selective recomputation(7 GB를 ~1.5 GB로 축소)이며, 추가 nodes는 필요하지 않다.

---

## 10. 이 Architecture를 위한 Attention Kernel 선택

**GDN linear attention은 독자적인 category다.** Chapter 6의 kernels 중 어느 것도 직접 적용되지 않는다.

- **FlashAttention 1/2/3** ([[ch-05]]): 표준 `softmax(QKᵀ/√d)V`를 위한 online softmax + tiling을 구현한다. GDN recurrence는 수학적 형태가 다르므로 FA로 대체할 수 없다.
- **xFormers memory-efficient** ([[ch-06]]): Rabe & Staats streaming attention의 CUTLASS FMHA 구현으로, 동일하게 softmax를 가정한다.
- **Ring Attention** ([[ch-06]], [[ring-attention]]): CP=1 hard-assert로 인해 architecture 수준에서 허용되지 않는다. GDN이 호환된다고 해도 sequential recurrence dependency 때문에 sequence sharding이 불가능하다.
- **SageAttention** ([[ch-06]]): quantized softmax attention으로, 잘못된 kernel family다.
- **PagedAttention** ([[ch-06]]): inference 전용 KV-cache management이므로 적용할 수 없다.
- **PyTorch SDPA** ([[ch-06]]): softmax attention을 위해 FA/math backends로 dispatch하며, custom linear-attention recurrences로 dispatch하지 않는다.

**허용되는 선택:** GDN model이 linear-attention recurrence용으로 자체 제공하는 fused CUDA/Triton kernel이다. Kernel의 memory profile은 layer마다 O(N²)이 아니라 O(N×d)(recurrent state)다. 이것이 A100에서 32k sequence를 처리할 수 있는 근본적인 이유다. Linear-attention architecture가 없다면 32k에는 Ring Attention이 필요하지만, 여기서는 금지되어 있다.

---

## 11. 이 Model을 위한 한 화면 Memory Memo

```
MODEL:   27B MoE (3B dense + 24B experts), 256 experts, GDN linear-attn
CLUSTER: A100-40GB, 32 nodes × 8 GPUs = 256 GPUs
PLAN:    TP=4 · EP=8 · DP=8 · PP=1  (world_size = 256)

FLOOR:   16Ψ/N = 16×27B / 64  ≈ 6.75 GB  (model states, ZeRO-3 DP+TP shard)
         Expert weights         ≈ 6.0 GB   (bf16, EP=8)
         Expert Adam states     ≈ 4.5 GB   (fp32, sharded over DP)

ACTIVATIONS:
         Selective recompute, SP, TP=4, s=32k, b=1  ≈ 7 GB
         (5as²/ht attention term = 0: linear attn has no softmax matrix)

SPIKES:
         MoE all-to-all buffer (2×all-to-all, s=32k, d=4096)  ≈ 0.5 GB
         Logit spike UNFUSED: 248k×32k×1×2  ≈ 30.4 GB  ← KILLS 40GB card
         Logit spike FUSED (Liger, chunk=2k): 248k×2k×2 ≈ 1.0 GB

MANDATORY KNOBS:
  [1] EP=8: without it, ZeRO-3 all-gathers 48 GB of expert weights → OOM
  [2] Fused-CE: without it, logit spike = 30 GB → OOM regardless
  [3] CP=0: architecture bans Ring Attn; use GDN's own kernel (O(Nd) memory)
  [4] No fp8: A100, not H100; stay bf16 mixed precision
  [5] Selective recompute: 5× activation saving at <2% FLOPs

FULL-FT PER-GPU ESTIMATE: ~23 GB of 40 GB (leaves room for b=2)
LORA PER-GPU ESTIMATE:    ~18 GB (saves 5 GB in optimizer states only;
                           activations and logit spike unchanged)

SCALING LAW:
  Adding DP nodes → divides model states (helps full-FT OOM)
  Adding DP nodes → does NOT divide activations or logit spike
  LoRA OOM from activations/logit → fix with fused-CE + recompute, not more nodes
```

---

## 문헌에서 얻은 핵심 통찰

**[[zero-memory-optimization]]에서 (Rajbhandari 2020):**
ZeRO-3는 per-GPU model states를 `16Ψ/N` bytes로 만들며, 이는 data-parallel world size에 선형적으로 비례한다. 7.5B와 Nd=64에서 DDP=120 GB vs ZeRO-3=1.9 GB라는 구체적인 기준점은 이 감소가 실제로 크다는 것을 입증한다. 그러나 논문의 각주는 중요하다. "ZeRO는 activations와 logit-buffer spike에 영향을 주지 않으며, 이들은 별도의 처리가 필요하다." ZeRO는 의자를 받치는 다리 하나일 뿐, 의자 전체는 아니다(전체 해결책 가운데 한 부분일 뿐이라는 비유).

**[[deepspeed-moe-ep]]에서 (Rajbhandari 2022):**
Memory 제약이 있는 hardware에서 대규모 MoE models를 실행할 때 Expert parallelism은 선택 사항이 아니다. EP가 도입하는 all-to-all communication topology(MoE layer마다 두 collectives)는 sequence × batch에 따라 증가하는 cost다. 따라서 설계자는 이를 static weight memory와 별도로 budget에 반영해야 한다. S=32k와 큰 batches에서는 all-to-all buffers가 expert weight memory에 맞먹을 수 있다.

**[[selective-recompute-korthikanti]]에서 (Korthikanti 2022):**
Attention score matrices는 버렸다가 다시 계산하고(저렴함: matmul + recurrence뿐임), MLP activations는 유지하면(recompute 비용이 큼: 대규모 FFN matmuls) <2% compute overhead로 activation을 5배 절감할 수 있다. GDN model에도 같은 비대칭성이 적용된다. Recurrent state update는 recompute 비용이 낮지만 expert FFN matmuls는 그렇지 않다. Selective recompute는 attention blocks를 대상으로 해야 한다.

**[[liger-fused-ce]]에서:**
Logit spike는 이 과정에서 가장 자주 간과되는 치명적 요인이다. 큰 vocab(≥32k)에서는 각주 정도로 여겨지지만, 248k에서는 지배적인 OOM trigger가 되며 40 GB card에서 model weights, optimizer states, activations를 모두 합친 것보다 크다. 이는 forward-backward 경계에서 일시적으로 발생하고 ZeRO의 영향을 받지 않으므로 static 16-byte ledger에는 보이지 않는다. 큰 vocab에서 Fused-CE는 performance optimization이 아니라 run이 시작되기 위한 correctness requirement다.

---

## 핵심 요점

- **Rule of 16**은 ZeRO-3에서 model states를 `16Ψ/N`으로 계산한다. 27B에서는 8 GPUs(N=8)를 사용해도 54 GB/GPU이므로 너무 크다. N≥64(8 nodes) 또는 보완 기법이 필요하다. 이것이 최소 cluster를 결정하는 기본 sizing 제약이다.
- **대규모 MoE에는 EP가 필수다**: A100-40GB에서 ZeRO-3의 all-gather는 rank와 layer마다 48 GB의 expert weights를 재구성한다. EP는 experts를 ranks에 걸쳐 shard하고 all-gather를 크기가 제한된 두 번의 all-to-all collectives로 대체한다.
- **V=248k, S=32k에서는 Fused-CE가 필수다**: unfused logit buffer는 30 GB로 전체 card memory의 76%다. lm_head computation을 fuse하지 않는 한 어떤 parallelism strategy도 이 spike를 피할 수 없다.
- **CP=1은 long-sequence toolkit을 무력화한다(사용 가능한 도구를 크게 제한한다는 비유)**: GDN architecture contract로 인해 Ring Attention이 제외된다. 허용되는 유일한 long-sequence strategy는 GDN kernel 자체의 O(N·d) recurrence다. 이 때문에 32k의 처리 가능 여부는 cluster가 아니라 model architecture가 결정한다.
- **LoRA node-invariance**: DP nodes를 추가해도 per-GPU activation memory는 줄지 않는다. Activations 또는 logit spikes로 인한 LoRA OOM에는 더 많은 hardware가 아니라 kernel-side 해결책(fused-CE, selective recompute)이 필요하다. Model states로 인한 Full-FT OOM은 nodes를 늘려 해결한다.
- **Per-GPU 판정 (TP=4, EP=8, DP=8, PP=1, 256 GPUs):** b=1의 full-FT에서 ~23 GB이며, b=2를 위한 여유를 남긴 채 A100-40GB 안에 들어간다.

---

## 참고 문헌

- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC 2020. https://arxiv.org/abs/1910.02054 — [[zero-memory-optimization]]
- Rajbhandari, S. et al. "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale." ICML 2022. https://proceedings.mlr.press/v162/rajbhandari22a — [[deepspeed-moe-ep]]
- Korthikanti, V. et al. "Reducing Activation Recomputation in Large Transformer Models." arXiv:2205.05198, 2022. https://arxiv.org/abs/2205.05198 — [[selective-recompute-korthikanti]]
- Shoeybi, M. et al. "Megatron-LM." arXiv:1909.08053, 2019. https://arxiv.org/abs/1909.08053 — [[megatron-tp-sp]]
- Huang, Y. et al. "GPipe." NeurIPS 2019. https://arxiv.org/abs/1811.06965; Narayanan et al. SC 2021. https://arxiv.org/abs/2104.04473 — [[pipeline-parallelism-1f1b]]
- Liu, A. et al. "Liger Kernel." arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel — [[liger-fused-ce]]
- Liu, H. et al. "Ring Attention." ICLR 2024. https://arxiv.org/abs/2310.01889 — [[ring-attention]]
- Anthony, Q. et al. "Transformer Math 101." EleutherAI Blog, 2023. https://blog.eleuther.ai/transformer-math/ — [[transformer-math-101]]
- Penedo, G. et al. "The Ultra-Scale Playbook." HuggingFace, 2025. https://nanotron-ultrascale-playbook.static.hf.space/ — [[ultrascale-playbook]]
- Bekman, S. "ML Engineering." https://github.com/stas00/ml-engineering — [[ml-engineering-memory]], [[training-oom-failure-modes]]
- Zhao, Y. et al. "PyTorch FSDP." VLDB 2023. https://arxiv.org/abs/2304.11277 — [[pytorch-fsdp]]

**이전의 모든 chapters:** [[ch-01]] (ledger + Rule of 16), [[ch-02]] (optimizer states + fused-CE), [[ch-03]] (activation checkpointing), [[ch-04]] (O(N²) attention memory), [[ch-05]] (FlashAttention), [[ch-06]] (kernel zoo + Ring Attention), [[ch-07]] (parallelism taxonomy), [[ch-08]] (calculator + OOM loop).
