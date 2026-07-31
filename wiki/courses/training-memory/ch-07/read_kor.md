<!-- chapter: ch-07
     track: scaling
     title: Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP
     deps: [[ch-01]]
     sources: [[zero-memory-optimization]], [[megatron-tp-sp]], [[pipeline-parallelism-1f1b]], [[pytorch-fsdp]], [[deepspeed-moe-ep]]
-->

# 7장 — 병렬화 분류 체계: DP / ZeRO / FSDP / TP / SP / PP / EP / CP

> **핵심 통찰.** 모든 병렬화 primitive는 training computation의 한 축을 기준으로 내리는 partitioning 결정이다. 그 축에는 batch (DP), optimizer state / gradient / weight (ZeRO), weight matrix의 column/row (TP), sequence position (SP / CP), layer stage (PP), expert parameter (EP)가 있다. 각 분할은 GPU당 memory 사용량을 줄이는 대신 communication overhead와 synchronization complexity를 감수한다. `world_size = TP × PP × CP × DP`라는 항등식(여기서 EP는 DP-expert dimension을 다시 세분한다)은 전체 설계 공간을 나타내며, 모든 production configuration은 이 곱 공간 안의 한 점이다.

> **지침.** DP rank가 4개 이상인 모든 job에서는 ZeRO-1(추가 비용 없이 optimizer-state를 sharding하는 방식)부터 시작하라. 단일 ZeRO-2 rank가 layer 하나의 activation조차 담을 수 없을 때 TP=8(NVLink node 하나)을 추가하라. 여러 node에 걸쳐 확장하려면 PP를 추가하라. TP만으로는 normalization layer에서 놓치는 t× activation 절감 효과를 되찾기 위해 TP ≥ 2일 때 SP를 함께 추가하라. 모든 expert weight를 ZeRO-3로 all-gather하는 작업 자체가 OOM을 일으킬 sparse MoE model에만 EP를 사용하라. sequence length가 단일 node에서 SP가 처리할 수 있는 범위를 넘어설 때만 CP를 추가하라.

---

## 1. 각 Primitive가 Sharding하는 대상 — 분류 표

| 기법 | 분할 대상 | GPU당 memory 감소 | step당 communication cost |
|-----------|----------------|-------------------------|-----------------------------|
| **DP** (Data Parallel) | Batch | 없음 — 전체 model이 복제됨 | gradient All-reduce: **2Ψ** |
| **ZeRO-1** | Optimizer state만 | N이 클 때 Adam state 4× | DDP와 동일: **2Ψ** |
| **ZeRO-2** | + Gradient | N이 클 때 약 8× (state + grad) | DDP와 동일: **2Ψ** |
| **ZeRO-3 / FSDP** | + Parameter | **16Ψ/N** — DP rank 수에 선형 비례 | DDP의 1.5×: AllGather + ReduceScatter = **3Ψ** |
| **TP** (Tensor Parallel) | Weight matrix의 column/row | weight는 1/t + linear-op activation은 1/t | transformer layer당 AllReduce 2회(MLP + Attn) |
| **SP** (Sequence Parallel) | Non-TP activation(norm, dropout)을 sequence를 따라 분할 | layer 전체에 걸쳐 완전한 t× activation 절감 | AllGather + ReduceScatter (AllReduce를 대체 — BW는 동일) |
| **PP** (Pipeline Parallel) | Layer를 stage로 분할 | weight는 1/p; activation은 local stage의 것만 | microbatch마다 stage 경계에서 P2P send/receive |
| **EP** (Expert Parallel) | Expert weight matrix를 MoE rank에 분산 | GPU당 E/EP_size개의 expert | MoE layer당 All-to-All 2회(dispatch + combine) |
| **CP** (Context Parallel) | Sequence position을 CP rank에 분산 | sequence dim을 따라 activation 분할 | attention step마다 KV block에 대한 Ring-AllReduce |

> **대화형 보조 자료:** [figures/parallelism-sharding.html](figures/parallelism-sharding.html) — 각 parallelism type 위에 마우스를 올리면 어떤 tensor axis가 sharding되는지, communication pattern은 무엇인지, 설정 가능한 model size에 따른 실시간 GPU당 memory 추정치를 볼 수 있다.

---

## 2. Data Parallelism — 복제를 기준으로 삼는 Baseline

DP는 아무것도 분할하지 않는 경우다. 모든 rank가 model의 완전한 사본을 보유하고 서로 다른 mini-batch slice를 처리한다. backward pass는 rank별 gradient를 생성하며, 이 gradient는 optimizer step 전에 **all-reduce**를 통해 전역적으로 평균 내야 한다. DDP (PyTorch DistributedDataParallel)는 이 all-reduce를 bucket 단위로 backward pass와 겹쳐 수행한다.

**Memory:** DP는 아무것도 줄이지 않는다. mixed precision의 7.5B model에서 GPU당 footprint는 완전한 `16Ψ` = 120 GB다. 80 GB A100을 사용하더라도 sharding 없이는 single-node 실행이 이미 불가능하다.

**Communication:** 모든 parameter의 bf16 gradient에 해당하는 **2Ψ bytes**를 전송하는 all-reduce를 step마다 한 번 수행한다. 이는 다른 모든 scheme을 비교하는 baseline이다.

따라서 DP만으로는 memory에 중립적인 distribution scheme이다. memory상 이점은 전혀 없이 data throughput scaling만 제공한다. 이 chapter의 다른 모든 primitive는 16Ψ라는 장벽을 허무는 방법을 다룬다(즉, GPU마다 복제되는 16Ψ의 model-state memory를 sharding하여 per-GPU 요구량을 줄인다).

---

## 3. ZeRO-1/2/3: Optimizer의 중복을 Partition하기

ZeRO ([[zero-memory-optimization]], Rajbhandari et al. SC 2020)는 모든 rank에 optimizer component 세 가지를 전부 복제하는 것이 DP의 memory 낭비라고 본다. 각 rank가 모든 state를 보유해야 할 수학적 필요는 없다. 각 rank는 gradient의 자기 shard를 parameter의 자기 shard에 적용하고 그 결과를 broadcast하기만 하면 된다.

### 3.1 16Ψ의 분해

Mixed-precision AdamW에는 정확히 다음이 필요하다([[zero-memory-optimization]]에서 발췌).

```
2Ψ fp16 weights  +  2Ψ fp16 gradients  +  4Ψ fp32 master params
+  4Ψ fp32 momentum  +  4Ψ fp32 variance  =  16Ψ bytes/param
```

`K=12`인 optimizer component(마지막 세 항: 4+4+4 = 12 B/param)는 DP rank 전반에 걸친 순수한 중복이다. 각 rank는 update를 계산하고 적용하기 위해 자신의 1/N slice만 필요하다.

### 3.2 세 단계의 Partitioning

**ZeRO-1 (Pos):** optimizer state만 partition한다. 각 rank는 optimizer state `KΨ` 전체 대신 `KΨ/Nd`를 저장하며, 완전한 `4Ψ` weight + gradient block은 계속 복제한다.

- GPU당 수식: `4Ψ + KΨ/Nd` — K=12, Nd=64일 때: parameter당 `(4 + 12/64)Ψ ≈ 4.19Ψ`.
- Communication: DDP와 동일(**2Ψ** all-reduce). 이 stage는 언제나 활성화할 가치가 있다.

**ZeRO-2 (Pos+g):** gradient도 추가로 partition한다. backward 중에 gradient를 all-reduce하는 대신 그 gradient를 담당하는 shard로 reduce-scatter한다.

- GPU당 수식: `2Ψ + (2+K)Ψ/Nd`.
- Communication: 여전히 **2Ψ**(reduce-scatter는 all-reduce와 volume이 동일하다).

**ZeRO-3 (Pos+g+p):** parameter도 partition한다. 각 rank는 weight의 `Ψ/Nd`만 보유한다. 각 layer의 forward와 backward 전에 해당 layer의 parameter를 all-gather해야 한다.

- GPU당 수식: `16Ψ/Nd` — DP degree에 따른 linear scaling.
- Communication: step당 **3Ψ**(forward 전 AllGather Ψ + backward 전 AllGather Ψ + backward 중 ReduceScatter Ψ) = **DDP의 1.5×**.

### 3.3 구체적인 수치: 7.5B Model, Nd = 64

[[zero-memory-optimization]]에서 발췌:

| Scheme | 수식 | GPU당 memory |
|--------|---------|----------------|
| DDP | 16Ψ | **120 GB** |
| ZeRO-1 | 4Ψ + 12Ψ/64 | **31 GB** |
| ZeRO-2 | 2Ψ + 14Ψ/64 | **17 GB** |
| ZeRO-3 | 16Ψ/64 | **1.9 GB** |

40 GB A100은 어떤 batch size에서도 이 model을 DDP로 실행할 수 없다. ZeRO-3는 weight+optimizer footprint를 2 GB 미만으로 낮춰 GPU의 거의 전부를 activation과 attention kernel workspace에 사용할 수 있게 한다.

**[[zero-memory-optimization]]에서 제시한 중대한 주의점:** "activation과 logit-buffer spike는 ZeRO의 영향을 받지 않는다. 이들은 별도의 처리(activation checkpointing, sequence parallelism)가 필요하다." ZeRO는 state 중복을 제거하지만 forward-pass computation graph에는 손대지 않는다.

---

## 4. FSDP — PyTorch Primitive로 구현한 ZeRO-3

PyTorch FSDP ([[pytorch-fsdp]], Zhao et al. VLDB 2023)는 ZeRO-3를 native 방식으로 구현하여 PyTorch dispatcher를 통해 TP, PP, autograd와 조합할 수 있게 한다.

### 4.1 Step별 Lifecycle

[[pytorch-fsdp]]에 따르면 `FULL_SHARD`에서는 다음과 같이 동작한다.

```
1. AllGather parameters for current FSDP unit    → forward pass
2. Free gathered parameters immediately
3. AllGather parameters again                    → backward pass
4. ReduceScatter gradients into shard
5. Optimizer step on local shard only
```

Communication = AllGather 2회(2Ψ) + ReduceScatter 1회(Ψ) = **총 3Ψ** — ZeRO-3와 동일하다.

### 4.2 AllGather의 일시적 Peak

FSDP는 parameter를 **FSDP unit**으로 감싼다(일반적으로 transformer layer 하나가 unit 하나다). 각 unit의 forward/backward 중에는 gather된 full-precision parameter를 위해 `≈ 2 × P_unit` 크기의 AllGather buffer가 일시적으로 할당된다. 완전한 GPU당 memory 수식은 다음과 같다.

```
Memory = 16Ψ/N  +  2 × P_unit   (transient peak during largest unit's forward/backward)
```

`2 × P_unit` buffer는 FSDP에서 실질적인 OOM 위험 요소다. steady-state ZeRO-3 수식에는 나타나지 않으므로 별도로 예산을 잡아야 한다. 더 세밀한 granularity(더 작은 FSDP unit)로 wrapping하면 이 buffer는 줄지만 AllGather 호출 횟수가 늘어난다.

### 4.3 Multi-Node를 위한 HYBRID_SHARD

여러 node에 걸친 job에서 FSDP의 `HYBRID_SHARD` mode는 NVLink로 연결된 각 intra-node group 안에서는 `FULL_SHARD`를 수행하고(빠른 all-gather), node 사이에서는 `REPLICATE`를 수행한다(비싼 inter-node AllGather 회피). 이 방식은 inter-node bandwidth 소비를 줄이는 대신 memory 절감 효과 일부를 포기하며, production multi-node training의 default다.

### 4.4 clip_grad_norm_의 함정

[[pytorch-fsdp]]에 따르면 FSDP에서 local shard에 `torch.nn.utils.clip_grad_norm_`을 호출하면 "global norm이 √N만큼 작게 보고되어 조용한 divergence를 일으킨다." 언제나 FSDP module 자체에서 `model.clip_grad_norm_(max_norm)`을 호출하라. 이 method는 모든 rank에 걸쳐 distributed norm computation을 수행한다.

---

## 5. Tensor Parallelism — Weight Matrix 분할

Tensor parallelism ([[megatron-tp-sp]], Shoeybi et al. 2019)은 다른 종류의 중복을 겨냥한다. 단일 transformer layer 안에서도 각 GPU가 여전히 완전한 weight matrix를 보유한다는 점이다. TP는 이 matrix를 t개의 GPU에 걸쳐 sharding한다(일반적으로 NVLink node 하나이며 t ≤ 8).

### 5.1 Column-Then-Row 분할

MLP의 경우:

- 첫 번째 linear (up-projection): **column-wise**로 분할한다. 각 rank는 서로 다른 output feature subset을 계산한다. input은 복제되어 있으므로 이 op 전에는 communication이 필요 없다.
- 두 번째 linear (down-projection): **row-wise**로 분할한다. 각 rank는 input feature의 subset을 보유한다. 이 op 후 **AllReduce**가 partial result를 합산한다.

Attention도 같은 pattern을 따른다. Q/K/V projection은 attention head별로 column-split하고, output projection은 row-split하며, AllReduce 한 번으로 동기화한다.

총 communication은 **transformer layer당 AllReduce 2회**(MLP에 한 번, attention에 한 번)다. 이 AllReduce는 critical path에 있으며 작은 t에서는 throughput-bound가 아니라 latency-bound이므로 TP에는 빠른 intra-node interconnect (NVLink)가 필요하다.

### 5.2 Activation Memory: 10sbh 문제

TP는 weight memory와 linear-op activation memory를 각각 1/t로 줄인다. 하지만 normalization layer (LayerNorm, Dropout)는 **TP로 분할되지 않는다**. 모든 rank에 계속 복제된다. [[megatron-tp-sp]]에 따르면 TP=t이고 SP가 없을 때 activation memory는 다음과 같다.

```
activation memory per layer = sbh(10 + 24/t + 5as/ht) bytes
```

`10sbh` 항은 LayerNorm + Dropout 영역으로, t개의 모든 rank에 복제되어 TP를 적용해도 변하지 않는다. 긴 sequence에서는 이 항이 지배적이므로 TP만으로 얻는 activation 감소 폭은 t×보다 훨씬 작다.

---

## 6. Sequence Parallelism — 10sbh 문제의 해결책

Sequence parallelism ([[megatron-tp-sp]], Korthikanti et al. 2022)은 복제된 `10sbh` activation을 동일한 t개의 TP rank에 걸쳐 **sequence dimension**을 따라 sharding한다. 추가 bandwidth는 필요 없다. SP는 TP 영역 전후의 AllReduce를 AllGather + ReduceScatter pair로 대체한다. communication volume은 같지만 이제 non-TP activation도 1/t로 분할된다.

TP=t와 함께 SP를 활성화하면 activation memory는 다음과 같다.

```
activation memory per layer = (sbh/t)(34 + 5as/h) bytes
```

이는 TP만 사용할 때와 비교해 **진정한 t× 감소**다. t=8(node 하나)일 때 SP는 linear-op 영역뿐 아니라 모든 layer 전체에서 activation을 8× 절감한다.

[[megatron-tp-sp]]에 따르면 A100 2,240개에서 실행한 530B GPT-3 training은 SP + selective recompute를 사용했을 때 MFU 54.2%를 달성했으며, 사용하지 않았을 때는 42.1%였다. 이는 SP가 확보한 memory 여유에서 비롯된 29%의 throughput 향상이다.

**SP는 TP > 1일 때만 활성화된다.** SP는 TP가 이미 만든 intra-node communication group을 그대로 재사용하므로(별도의 communication topology를 만들지 않으므로), TP=1에서 SP를 활성화해도 아무 효과가 없다.

### 6.1 Selective Recomputation (`5as²b/ht` 항)

[[megatron-tp-sp]], Korthikanti 2022에 따르면, 전체 layer를 checkpointing하여 30–40%의 FLOP overhead를 부담하는 대신 attention softmax/score 영역만 recompute한다. 이 영역은 compute density가 낮은 activation(attention weight 및 intermediate score) `5as²b/ht` bytes에 해당한다.

- Memory 절감: 모든 것을 저장할 때보다 activation footprint 약 5× 감소
- FLOP overhead: 추가 computation < 2%
- full-layer recompute와 비교: memory 이점은 같고 FLOP 측면에서는 15–20× 저렴함

Selective recompute가 SP보다 더 유리해지는 threshold는 model에 따라 달라진다. 긴 sequence(a > h/5)에서는 attention 영역이 지배적이므로 효과를 극대화하기 위해 selective recompute를 SP와 함께 사용하는 것이 일반적이다.

---

## 7. Pipeline Parallelism — Layer 분할

Pipeline parallelism ([[pipeline-parallelism-1f1b]])은 서로 다른 **layer**를 서로 다른 GPU에 배치하는 유일한 primitive다. 각 pipeline stage p는 전체 layer의 1/p을 보유하여 weight memory를 즉시 p로 나눈다. stage 간 communication은 stage 경계에서 point-to-point tensor send로 이루어진다.

### 7.1 Bubble

Stage를 구성하면 pipeline 시작 및 배출 bubble(파이프라인이 충분히 채워지거나 비워지는 동안 생기는 유휴 구간)이 발생한다. GPipe(all-forward-then-all-backward)와 1F1B(interleaved microbatch scheduling)는 모두 동일한 bubble fraction을 갖는다.

```
bubble fraction = (p - 1) / (m + p - 1)  ≈  (p - 1) / m  when m >> p
```

여기서 m은 microbatch 수(= total batch / microbatch size)다. bubble을 5% 미만으로 유지하는 경험 법칙은 **m ≥ 20 × (p - 1)**이다. 4-stage pipeline에는 gradient accumulation microbatch가 최소 60개 필요하다. 이것이 PP의 대가다. bubble을 상쇄할 만큼 충분한 microbatch를 누적해야 하며, 이 누적 자체도 batch-effective throughput에 영향을 미친다.

### 7.2 GPipe와 1F1B 비교: Memory 차이

두 schedule의 bubble fraction은 동일하지만 **activation memory**에서는 중대한 차이가 있다.

| Schedule | stage당 activation peak | 비례 대상 |
|----------|--------------------------|-------------|
| GPipe | m개 microbatch의 activation (all-forward-then-backward) | O(m) — 무제한 증가 |
| 1F1B | p개 microbatch의 activation (동시에 진행 중인 것은 최대 p개) | O(p) — 일정함 |

[[pipeline-parallelism-1f1b]]에 따르면 1F1B는 "동시에 진행 중인 activation memory를 p개의 microbatch(pipeline depth)로 제한한다. 따라서 memory는 m과 무관하게 O(p)로 증가한다." GPipe에 비해 1F1B가 갖는 이점은 **전적으로 memory에 있으며 bubble time에는 없다**.

### 7.3 Interleaved 1F1B (Virtual Stage)

Megatron의 interleaved schedule은 각 device에 연속된 stage 하나 대신 서로 연속되지 않은 virtual stage v개를 할당한다. bubble fraction은 다음과 같이 감소한다.

```
bubble fraction (interleaved) = (p - 1) / (v × m)
```

그 대가는 microbatch당 v회의 추가 P2P send/receive와 rank당 activation peak의 소폭 증가(stage 내부의 standard 1F1B 대비 약 v×)다. Megatron scale에서 PP degree가 클 때 사용하는 production setting이다.

---

## 8. Expert Parallelism — 대규모 Sparse MoE

Mixture-of-Experts model ([[deepspeed-moe-ep]])에서 expert parallelism은 dense parameter가 아니라 expert weight matrix를 sharding한다.

### 8.1 Memory Model

총 expert 수가 E이고 EP_size rank가 있을 때:

```
expert weights per GPU = W_expert / EP_size
```

각 GPU는 E/EP_size개의 expert weight matrix를 보유한다. Non-expert layer(attention, embedding, non-expert FFN)는 평소처럼 DP에서는 복제되고 TP/PP에서는 sharding된다. expert가 256개인 MoE에서 EP_size=64 configuration을 사용하면 GPU당 expert는 4개다. DP로 복제하는 baseline보다 극적으로 작다.

**대규모 MoE model에서 ZeRO-3가 EP를 대체할 수 없는 이유:** ZeRO-3는 각 layer의 forward pass 전에 모든 parameter를 all-gather한다. expert가 dense parameter count의 16×를 차지하는 256-expert model에서는 ZeRO-3 all-gather가 일시적으로 완전한 `W_expert`를 각 GPU에 올린다. 이는 EP가 해결하려던 job을 즉시 OOM으로 만든다.

### 8.2 All-to-All Routing — MoE Layer당 두 번의 Collective

Expert parallelism을 사용하려면 token을 target expert가 있는 GPU로 물리적으로 routing해야 한다.

```
Forward MoE layer:
  1. Dispatch all-to-all: each GPU sends its assigned tokens to EP ranks with target experts
  2. Run expert FFN on received tokens (local computation)
  3. Combine all-to-all: each GPU receives expert outputs and aggregates by routing score
```

[[deepspeed-moe-ep]]에 따르면 각 all-to-all은 rank당 `tokens × d_model` bytes를 전송한다. 고정 volume인 all-reduce와 달리 all-to-all volume은 **sequence length × batch size에 비례**하며, 긴 context에서는 주된 communication bottleneck이 될 수 있다.

**All-to-all buffer:** 두 collective에는 각각 `tokens × d_model × 2 bytes` 크기의 일시적 buffer가 필요하다. steady-state expert weight footprint와 함께 이 buffer의 예산도 잡아야 한다.

### 8.3 Capacity Factor

```
capacity = capacity_factor × (tokens_per_batch / num_experts)
```

Capacity factor가 1.0 미만이면 **token dropping**(과부하된 expert로 routing된 token을 조용히 버리는 현상)이 발생하여 training instability를 유발한다. capacity_factor ≥ 1.0이 안전한 최소 threshold이며, 일반적인 production 값은 1.0–2.0이다.

---

## 9. Context Parallelism — 긴 Sequence를 위한 Ring Attention

Context parallelism은 **sequence dimension**을 CP rank에 걸쳐 sharding하며, 각 rank는 연속된 subsequence를 담당한다. Cross-rank attention에는 Ring Attention pattern이 필요하다. 각 rank가 자신의 KV block을 ring을 따라 전달해 결국 모든 rank가 모든 position에 attention하도록 한다.

CP와 SP는 서로 겹치는 문제를 서로 다른 scale에서 해결한다.

- **SP**는 단일 node(동일한 TP group, NVLink) 안에서 동작하며 normalization activation을 sharding한다.
- **CP**는 여러 node(또는 큰 intra-node group)에 걸쳐 동작하며 sequence 자체와 attention computation을 sharding한다.

주어진 TP degree에서 sequence length가 SP만으로 처리할 수 있는 범위를 넘어설 때만 CP를 활성화한다. **world_size = TP × PP × CP × DP**라는 항등식에서 CP와 TP는 서로 독립적인 parallel dimension이므로(각각 sequence/context dimension과 tensor dimension을 별도로 shard하므로) 함께 조합할 수 있다.

---

## 10. World-Size 항등식과 조합 규칙

EP를 제외한 다섯 dimension은 모두 곱셈 방식으로 조합된다.

```
world_size = TP × PP × CP × DP
```

EP는 DP dimension을 세분한다. EP rank는 DP group GPU의 subset을 구성하므로 EP_size ≤ DP다. 64-GPU MoE job의 실제 조합은 다음과 같을 수 있다.

```
TP=8, PP=4, CP=1, DP=2  →  world_size = 64
EP=2  (subdivides DP=2)
```

**적용 순서(heuristic):**

1. TP = NVLink node당 GPU 수(일반적으로 8)로 설정한다. intra-node이며 bandwidth가 높다.
2. TP ≥ 2일 때마다 SP를 활성화한다.
3. PP = layer를 분산하는 데 필요한 node 수로 설정한다. inter-node P2P다.
4. DP = 남은 dimension으로 설정한다(`DP = world_size / (TP × PP × CP)`).
5. DP group 안에서 ZeRO-3 / FSDP를 적용한다.
6. MoE라면 expert를 sharding하도록 EP ≤ DP로 설정한다.
7. 현재 TP×SP에서도 sequence length가 GPU당 memory를 초과할 때만 CP를 추가한다.

---

## 11. LoRA의 Node 불변성과 Full-FT의 World-Size Scaling 비교

Full fine-tuning과 LoRA 사이에는 중대한 비대칭이 있다.

**ZeRO-3를 적용한 full fine-tuning:** weight + optimizer state memory는 DP degree N에 따라 `16Ψ/N`으로 scaling된다. DP를 두 배로 늘리면(GPU를 더 추가하면) GPU당 state memory는 절반이 된다. training job의 memory footprint는 world size에 민감하며, GPU를 추가하면 GPU당 load가 직접 줄어든다.

**LoRA:** adapter parameter(rank r ≪ Ψ)만 gradient와 optimizer state를 축적한다. 고정된 base model도 여전히 forward/backward에 참여한다. 각 GPU는 완전한 frozen weight(inference에서 2 B/param)를 보유하고 adapter로 gradient를 전파하기 위한 전체 activation graph를 계산한다.

따라서 **LoRA activation memory는 world size에 불변이다.** 다음 이유로 DP replica를 추가해도 GPU당 activation footprint는 줄지 않는다.

1. adapter backward pass를 위해 frozen base의 activation을 계속 저장해야 한다.
2. DP는 sequence나 layer를 분할하는 것이 아니라 batch 사본을 추가한다.

그 결과 activation 때문에 OOM이 발생하는 LoRA job은 DP rank를 늘려도 해결되지 않는다. 해결책은 activation checkpointing, 더 짧은 sequence, 또는 더 작은 batch size이며 scale-out이 아니다.

LoRA에서는 training하는 것이 rank `r` matrix뿐이므로 optimizer state와 gradient memory는 감소하며, base model optimizer에서 비롯되는 full-FT와 유사한 overhead는 중요하지 않게 된다. 그러나 LoRA가 일반적으로 사용되는 규모(큰 base model, 중간 정도의 batch size)에서는 activation이 지배적이므로 node 불변성 제약이 실제 구속 조건이 된다.

---

## 12. Logit-Buffer Spike — 어떤 Parallelism으로도 줄어들지 않음

[[ch-02]]의 loss-head spike를 다시 살펴보자. forward-backward 경계에서 할당되는 logit buffer는 다음과 같다.

```
spike = vocab_size × seq_len × batch_size × 2 bytes (bf16)
```

32K-vocab model에서 seq=2048, batch=1일 때는 `32768 × 2048 × 1 × 2 ≈ 0.13 GB`로 무시할 만하다. 그러나 128K-vocab model에서 seq=4096, batch=4일 때는 `131072 × 4096 × 4 × 2 ≈ 4.3 GB`이며, 256K vocab의 production batch size에서는 spike가 step당 10–20 GB에 이를 수 있다.

**중요하게도 이 spike는 ZeRO, TP, PP, SP 어느 것으로도 줄어들지 않는다.** [[zero-memory-optimization]]에 따르면 ZeRO는 activation이나 logit buffer에 손대지 않는다. TP는 lm_head의 weight matrix를 분할하지만 output logit tensor는 cross-entropy loss 전에 다시 gather된다. 유일한 수단은 loss를 tile 단위로 계산하여 완전한 logit tensor를 materialize하지 않는 fused/chunked cross-entropy (Liger kernel, [[ch-02]])다.

forward 중 `lm_head`에서 발생하는 OOM은 logit spike다. 해결책은 parallelism 변경이 아니라 chunked cross-entropy다.

---

## 문헌에서 얻는 핵심 통찰

**1. ZeRO의 1.5× communication overhead는 parameter 중복을 제거하기 위한 이론적 최소 비용이다** ([[zero-memory-optimization]]). DDP에는 2Ψ(all-reduce 한 번)가 필요하고, ZeRO-3에는 3Ψ(all-gather 두 번 + reduce-scatter 한 번)가 필요하다. parameter를 분산하는 모든 scheme은 parameter를 복제하는 scheme보다 최소한 Ψ만큼의 추가 communication을 치러야 한다. 논문은 이를 "ZeRO는 computational efficiency를 유지하면서 memory efficiency를 달성한다"고 표현한다.

**2. SP는 독립적인 기법이 아니라 TP communication group 안에서만 존재한다** ([[megatron-tp-sp]]). SP는 TP의 AllReduce를 AllGather + ReduceScatter로 대체한다. 이는 추가 bandwidth cost 없이 sharding할 수 있도록 sequence dimension을 여는 재구성이다. 따라서 TP > 1이면 SP는 추가 communication bandwidth를 거의 요구하지 않으면서 activation memory를 줄일 수 있다("사실상 무료"라는 말은 별도의 communication group이나 추가 bandwidth cost가 거의 없다는 뜻이다).

**3. 1F1B의 혁신은 scheduling efficiency가 아니라 전적으로 memory에 있다** ([[pipeline-parallelism-1f1b]]). GPipe와 1F1B는 모두 `(p-1)/m` bubble fraction을 달성한다. GPipe의 결함인 O(m) activation memory는 microbatch k의 forward가 끝나는 즉시 그 backward를 시작하여 동시에 진행 중인 activation을 p개로 제한하는 1F1B 규칙으로 해결된다.

**4. MoE에서 EP와 ZeRO-3는 중복되는 것이 아니라 상호 보완적이다** ([[deepspeed-moe-ep]]). ZeRO-3는 DP rank에 걸쳐 parameter를 sharding하고 layer마다 이를 all-gather하므로 dense model에는 적합하다. dense parameter count의 4–16×에 달하는 sparse expert에서는 ZeRO-3의 all-gather transient peak가 GPU memory를 초과한다. EP는 expert를 할당된 GPU에 영구적으로 배치하고, ZeRO-3는 나머지 dense parameter를 처리한다. DeepSpeed-MoE는 둘을 결합한다.

---

## 핵심 요점

- 16Ψ 수식은 정확히 `2Ψ + 2Ψ + 12Ψ`(weight, gradient, Adam state)로 분해된다. ZeRO-3는 세 가지 모두를 partition하여 `16Ψ/N`에 도달한다. activation과 logit spike는 ZeRO의 범위 밖이다.
- FSDP는 한 가지 추가 위험이 있는 ZeRO-3다. `2 × P_unit` AllGather transient buffer는 steady-state 수식에는 보이지 않지만 실제 OOM을 일으키는 요인이다.
- LayerNorm/Dropout이 복제되므로 TP만으로는 t× activation 감소를 달성하지 못한다. SP는 해당 op를 sequence를 따라 sharding하여 이 문제를 해결한다. 올바른 수식 쌍은 TP-only `sbh(10 + 24/t + 5as/ht)`와 SP+TP `(sbh/t)(34 + 5as/h)`다.
- GPipe와 1F1B의 bubble fraction은 동일하다. 1F1B의 이점은 activation memory가 O(m)이 아니라 O(p)라는 점이다.
- 대규모 MoE(GPU당 expert 환산 수가 >~16)에는 EP가 필수다. MoE layer당 두 번의 all-to-all collective는 model size가 아니라 sequence × batch에 비례한다.
- LoRA activation memory는 node 수에 불변이다. DP replica를 늘려도 도움이 되지 않는다. LoRA OOM의 구속 조건은 activation size이며 checkpointing이나 sequence 축소로만 해결된다.
- 어떤 parallelism도 logit-buffer spike를 줄이지 못한다. chunked cross-entropy를 사용하라.
- 조합 항등식 `world_size = TP × PP × CP × DP`가 설계 공간을 이룬다. EP ≤ DP다. 먼저 TP(intra-node, 최고 bandwidth)를 적용하고, 그다음 PP(inter-node layer)를 적용한 뒤 나머지를 DP로 채운다.

---

## 참고 문헌

- Rajbhandari, S. et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC '20. https://arxiv.org/abs/1910.02054 ([[zero-memory-optimization]])
- Shoeybi, M. et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. https://arxiv.org/abs/1909.08053 ([[megatron-tp-sp]])
- Korthikanti, V. et al. (2022). Reducing Activation Recomputation in Large Transformer Models. MLSys 2023. https://arxiv.org/abs/2205.05198 ([[megatron-tp-sp]])
- Huang, Y. et al. (2019). GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. NeurIPS 2019. https://arxiv.org/abs/1811.06965 ([[pipeline-parallelism-1f1b]])
- Narayanan, D. et al. (2021). Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. SC '21. https://arxiv.org/abs/2104.04473 ([[pipeline-parallelism-1f1b]])
- Zhao, Y. et al. (2023). PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. VLDB 2023. https://arxiv.org/abs/2304.11277 ([[pytorch-fsdp]])
- Lepikhin, D. et al. (2021). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. ICLR 2021. https://arxiv.org/abs/2006.16668 ([[deepspeed-moe-ep]])
- Rajbhandari, S. et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale. ICML 2022. https://proceedings.mlr.press/v162/rajbhandari22a ([[deepspeed-moe-ep]])
