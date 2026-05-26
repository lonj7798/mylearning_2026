<!-- chapter: ch-05
     track: foundations
     title: Distributed Training — FSDP, TP, PP, ZeRO
     sources: [[fsdp-sft]], [[sequence-packing]], [[loss-masking-prompt]], [[mixed-precision]], [[adam]], [[gradient-clipping]]
     figures: figures/fsdp-memory.html
-->

# 5장 — 분산 학습: FSDP, TP, PP, ZeRO

> **핵심 통찰.** 하나의 parallelism primitive만으로는 70B+ model을 end-to-end로 학습할 수 없다. 실제 학습은 **nD parallelism**이다. memory를 위한 data sharding(FSDP / ZeRO-3), layer 내부 compute를 위한 tensor parallelism, node를 가로지르기 위한 pipeline parallelism, MoE를 위한 expert parallelism이 함께 쓰인다. 각 축에는 memory/bandwidth tradeoff가 있고, 이를 잘못 잡으면 training job이 hang한다.
>
> **지침.** 13B 미만 model에는 FSDP(FULL_SHARD) + bf16 + activation checkpointing + packed SFT면 충분하다. 13–70B에는 NVLink가 있으면 within-node TP(=8)를 추가하고, node 간에는 FSDP를 유지하라. 200B+에는 3D parallelism(DP × TP × PP)과 MoE용 expert parallelism이 필요하다. 이 primitive를 직접 작성하지 마라. verl / OpenRLHF / torchtitan / Megatron / DeepSpeed를 사용하라.

---

## 이 장이 필요한 이유

분산 학습은 frontier lab compute가 가장 많이 낭비되는 영역이다. 알고리즘이 새롭기 때문은 아니다. 그렇지 않다. sharding, mixed precision, gradient clipping, checkpointing 사이의 상호작용이 64+ GPU 규모에서만 드러나는 silent bug를 만들기 때문이다. FSDP local shard에 대한 naive clip-norm 호출은 global norm을 √N만큼 과소계산하고 조용히 divergence한다. node를 가로지르는 TP group은 inter-node bandwidth에 눌린다. TP group 내부의 activation checkpointing은 경계를 잘못 잡으면 communication cost를 두 배로 만든다.

이 장은 correctness model을 준다. 무엇이 어디로 가는지, 어떤 communication이 언제 일어나는지, 어떤 framework call이 올바른지 설명한다. 그래서 실제 training codebase 앞에 앉았을 때 primitive의 이름이 이미 머릿속에 있게 한다.

주요 출처는 FSDP mechanics의 [[fsdp-sft]], interaction의 [[sequence-packing]]과 [[mixed-precision]]이다.

---

## 1. Memory formula — 왜 70B를 DDP로 할 수 없는가

출처: [[fsdp-sft]]. `N`개 data-parallel rank, `P`개 parameter, AdamW에서 per-GPU memory:

| Component | DDP(replicate) | ZeRO-2(SHARD_GRAD_OP) | ZeRO-3 / FSDP(FULL_SHARD) |
|---|---|---|---|
| Parameters(bf16) | 2P | 2P | 2P / N |
| Gradients(bf16) | 2P | 2P / N | 2P / N |
| Optimizer state(fp32: m, v, master) | 12P | 12P / N | 12P / N |
| Temporary AllGather buffer | 0 | 0 | ≈ 2P(largest FSDP unit) |
| **Steady-state total** | **16P** | **(4P / N) + 12P** | **(16P / N) + 2P** |

70B × 8 GPU를 넣어보면:

- DDP: 16 · 70 = GPU당 1120 GB. 불가능.
- ZeRO-3 / FSDP: (16 · 70 / 8) + 2 · 70 ≈ GPU당 280 GB. **80 GB에서도 여전히 불가능.**
- Activation checkpointing(activation memory 약 2배 이상 절약)과 ZeRO-3 offload를 더하면 8 × 80 GB에서 가능하다.

70B SFT에서 FSDP FULL_SHARD + activation-ckpt + bf16 + fp32 master는 있으면 좋은 것이 아니라 단단한 *floor*다. 어떤 (P, N, precision) 조합에서도 이를 계산해 보는 interactive memory calculator는 `figures/fsdp-memory.html`을 보라.

---

## 2. FSDP / ZeRO — data-parallel sharding axis

FSDP와 DeepSpeed ZeRO는 같은 수학을 구현한다. 구현은 독립적이다. 중요한 strategy는 세 가지다.

```
NO_SHARD          ≡ DDP                → AllReduce grads
SHARD_GRAD_OP     ≡ ZeRO-2            → ReduceScatter grads; shard grads + opt state
FULL_SHARD        ≡ ZeRO-3            → AllGather params per-block + ReduceScatter grads
HYBRID_SHARD      ≡ intra-node FULL,   → FULL_SHARD within node, REPLICATE across nodes
                    inter-node REPL
```

**FULL_SHARD에서 step마다 일어나는 일**([[fsdp-sft]]):

1. **Forward.** 각 transformer block의 forward 전에 sharded parameter를 AllGather하여 full tensor를 만들고, block을 실행한 뒤 gathered copy를 해제한다.
2. **Backward.** weight에 대해 같은 AllGather를 수행하고, gradient를 shard로 ReduceScatter한다.
3. **Optimizer step.** 각 rank는 자신의 local shard의 parameter와 optimizer state만 업데이트한다.

step당 communication volume: 2P AllGather + 2P ReduceScatter = 4P. DDP의 AllReduce는 2P다. FSDP의 추가 bandwidth는 N배 memory saving을 사는 비용이다.

**HYBRID_SHARD**는 실용적인 타협이다. node 내부(8 GPU, NVLink)는 FULL_SHARD를 하고, node 간(slow Ethernet / IB)은 REPLICATE한다. 이렇게 하면 fast-NVLink node 내부의 FSDP memory saving은 유지하면서, 느린 cross-node communication은 backward 끝의 고전적 DDP AllReduce로 바뀐다.

**Wrapping policy** — "sharding unit"이 무엇인지:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer},   # one unit per block
)
```

각 transformer block이 FSDP unit이므로 AllGather는 해당 block의 parameter만 가져온다. 전체 model을 하나의 unit으로 만들면 모든 것을 AllGather하게 되어 memory saving을 잃는다.

### Mixed precision + FSDP — 올바른 주문

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp import ShardingStrategy

bf16_mp = MixedPrecision(
    param_dtype   = torch.bfloat16,   # weights live in bf16 after AllGather
    reduce_dtype  = torch.bfloat16,   # gradient ReduceScatter in bf16
    buffer_dtype  = torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy = ShardingStrategy.FULL_SHARD,
    auto_wrap_policy  = auto_wrap_policy,
    mixed_precision   = bf16_mp,
    # activation checkpointing applied separately via apply_activation_checkpointing
)
```

Optimizer state(AdamW `m`, `v`, master weight)는 fp32에 남는다. FSDP는 `param_dtype`와 무관하게 이를 fp32로 sharding한다. 교차 규칙은 [[mixed-precision]]을 보라.

### FSDP-specific gradient-clipping bug

[[gradient-clipping]]에서: clip-norm은 **모든 shard에 걸쳐** 계산되어야 한다. FULL_SHARD에서 local parameter에 naive `torch.nn.utils.clip_grad_norm_`를 호출하면 √N배 과소계산한다. 대신 FSDP method를 사용하라.

```python
model.clip_grad_norm_(max_norm=1.0)     # correct — computes global norm
```

DeepSpeed에는 유사한 `engine.clip_grad_norm_`가 있다. 직접 만들지 마라. home-rolled FSDP training loop에서 가장 큰 silent divergence bug다.

---

## 3. Tensor Parallelism — layer 내부를 나누기

FSDP는 rank 전반에 whole parameter를 shard한다. **Tensor Parallelism(TP)**은 rank 전반에 단일 parameter matrix *내부*를 나눈다. single layer의 compute(memory가 아니라)가 bottleneck일 때 쓰는 도구다. 70B+ model, 넓은 MLP, 긴 sequence가 그렇다.

Mechanic(Megatron-LM convention):

```
For Y = X · W where W is [d_in, d_out]:

  Column parallelism:  W → [W_1, W_2, ..., W_TP]   (split columns)
                       Y = X · W = concat(X · W_1, X · W_2, ...)
                       → no communication in forward, AllReduce in backward
  Row parallelism:     W^T → [W_1^T, W_2^T, ...]   (split rows)
                       Y = X · W, X itself is partitioned
                       → AllReduce in forward, no comm in backward
```

Attention sub-layer는 column-parallel `Q, K, V` projection 뒤에 row-parallel `W_O`를 쓴다. 따라서 attention block마다 AllReduce 하나, MLP마다 하나가 있다. `W_gate / W_up` + `W_down` MLP도 같다.

**TP는 bandwidth-bound다.** AllReduce가 모든 transformer block 안에서 두 번 일어난다. NVLink로 연결된 GPU에서는 괜찮지만 PCIe나 Ethernet을 가로지르면 throughput이 무너진다. **경험칙: TP는 반드시 하나의 node 안에 들어가야 한다.** 일반적인 선택은 단일 8-GPU NVLink node에서 TP=8이다.

**TP × FSDP 결합.** TP는 `TP` rank의 group 내부에서 parallelize하고, FSDP는 `N / TP` group 전반에서 parallelize한다. 이것이 "2D parallelism"이다. Framework(Megatron, NeMo, torchtitan)는 2D process group을 설정해 이를 처리한다.

---

## 4. Pipeline Parallelism — depth를 따라 나누기

model이 단일 TP group에 들어가지 않거나(TP=8에서도), cross-node communication이 TP를 망가뜨릴 때 model을 *depth를 따라* 자른다. layer 1–10은 node A, layer 11–20은 node B 식이다. 이것이 **Pipeline Parallelism(PP)**이다.

Naive PP에는 "bubble problem"이 있다. node A가 micro-batch 1의 forward를 하는 동안 node B/C/D는 idle이다. **1F1B scheduling**(one-forward-one-backward)은 micro-batch를 interleave하여 모든 node를 바쁘게 유지한다.

```
Time →
Stage 1:  F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
Stage 2:     F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
Stage 3:        F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
Stage 4:           F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
```

bubble fraction은 대략 `(PP − 1) / (num_microbatches + PP − 1)`이다. PP=8, 64 micro-batch이면 bubble ≈ 10%다. Interleaved 1F1B는 activation memory를 대가로 이를 더 낮춘다.

PP는 latency와 cross-node bandwidth를 맞바꾼다. stage 사이에는 weight/grad(큰 것) 대신 activation(작은 것)이 이동한다. Frontier scale(DeepSeek-V3, Llama-3 405B)에서는 PP가 node를 span하는 방법이다.

---

## 5. Expert Parallelism — MoE 전용

Mixture-of-Experts model(DeepSeek-V3, Mixtral, Qwen-MoE)에서 각 expert는 token subset만 처리하는 별도 FFN이다. **Expert Parallelism(EP)**은 서로 다른 expert를 서로 다른 rank에 배치한다. Routing은 각 token을 선택된 expert rank로 AllToAll을 통해 보낸다.

EP의 bottleneck은 AllToAll bandwidth다. DeepSeek-V3 scale(총 671B, active 37B)에서는 expert parallelism이 TP × PP × DP 위에 얹히는 자체 dimension이다. DeepSeek는 이 communication을 compute 뒤에 숨기기 위해 custom AllToAll kernel(DualPipe)을 개발했다.

2025년 MoE training run의 parallelism recipe는 `DP × TP × PP × EP`처럼 생겼고, 각 축은 cluster의 bandwidth tier에 맞게 선택된다.

---

## 6. Activation checkpointing — 직교하는 memory knob

§1의 distributed-memory formula는 *parameter* memory를 budget한다. **Activation memory**(backward를 위해 보관하는 forward activation)는 더 클 수 있다. token당 parameter보다 두 자릿수 작지만 sequence length × batch × layer가 곱해진다.

Activation checkpointing은 layer의 activation을 저장하는 대신 backward 중에 forward pass를 다시 실행한다. 절약되는 memory: O(L · seqlen · hidden). 비용: 25–35% 더 많은 compute.

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, CheckpointImpl,
)
apply_activation_checkpointing(
    model,
    check_fn = lambda m: isinstance(m, LlamaDecoderLayer),   # every block
    checkpoint_impl = CheckpointImpl.NO_REENTRANT,
)
```

현대 기본값은 모든 transformer block을 checkpoint하는 것이다. NO_REENTRANT variant는 FSDP와 올바르게 compose된다. reentrant variant는 그렇지 않다.

---

## 7. 전형적인 2025 SFT recipe — 8×80GB에서 70B

[[fsdp-sft]]에서:

| Knob | Value |
|---|---|
| Strategy | FSDP FULL_SHARD |
| Precision | bf16 params + bf16 reduce + fp32 optimizer master |
| Activation ckpt | transformer block별(NO_REENTRANT) |
| Micro-batch / GPU | 1 |
| Gradient accumulation | 16(effective batch = 128) |
| Sequence packing | yes(ch-04 / [[sequence-packing]] 참조) |
| Max seq length | 4096 |
| Optimizer | AdamW β=(0.9, 0.95), wd=0.0(SFT), ε=1e-8 |
| Learning rate | 1e-5, cosine, warmup 3% |
| Gradient clip | `model.clip_grad_norm_`를 통한 1.0 |
| Chat template | `train_on_response_only=True`와 함께 tokenizer의 `apply_chat_template` |

configuration을 고정하라. 이 표의 모든 knob은 1–4장에 문서화된 failure mode를 가진다.

---

## 8. Framework — 2025 field guide

경험칙: **직접 만들지 마라**. 하나를 고르고 wrapping convention을 이해하라.

| Framework | Sharding | TP | PP | MoE / EP | 가장 적합한 용도 |
|---|---|---|---|---|---|
| **PyTorch FSDP2** | FULL / HYBRID / NO_SHARD | DeviceMesh 경유 | third-party | limited | research, ~70B까지의 SFT |
| **torchtitan** | FSDP2 + native TP + PP | yes | yes | yes | modern PyTorch reference for pretraining |
| **Megatron-LM** | DDP + ZeRO-1 | native TP | 1F1B PP | EP | Nvidia frontier pretraining |
| **DeepSpeed ZeRO-3** | FSDP와 같은 수학 | Megatron 경유 | yes | yes | legacy production, bf16/fp16 offload |
| **NeMo** | 내부는 Megatron | yes | yes | yes | Nvidia enterprise stack |
| **verl / OpenRLHF** | FSDP + TP | yes | limited | some | actor + ref + RM + critic orchestration을 갖춘 RL |
| **TRL** | Accelerate + FSDP | limited | no | no | single-node SFT / preference tuning |

Infrastructure track의 53–56장은 verl / OpenRLHF / TRL internals를 source-level로 end-to-end 살펴본다.

---

## 9. Scale에서 흔한 silent-failure mode

- **FSDP local clip.** FULL_SHARD에서 `model.clip_grad_norm_` 대신 `clip_grad_norm_`를 호출하면 global norm을 √N만큼 과소계산한다. 조용한 divergence.
- **Reentrant impl로 TP group 내부 activation checkpointing.** recompute 비용이 두 배가 된다. NO_REENTRANT를 사용해 고친다.
- **Shard 전반의 mixed fp16 / bf16.** `param_dtype=bf16`인데 `reduce_dtype=fp16`이면 모든 gradient reduction마다 type-cast가 생긴다. numerical drift.
- **TP group이 node를 가로지름.** 모든 block 내부의 inter-node AllReduce가 throughput을 죽인다. TP를 단일 NVLink node로 제한하라.
- **Skewed batch 아래 NCCL timeout.** 어떤 rank의 straggler(pack 안의 유난히 긴 sequence)가 전체 collective를 지연시킨다. timeout은 hang으로 나타난다. `py-spy dump --pid <rank>`로 진단하라.
- **rank 0에만 optimizer checkpointing.** FSDP에서 각 rank는 optimizer state의 shard를 소유한다. sharded-save API를 사용하라.
- **Resume across data loader desync.** 각 rank의 sampler는 local이 아니라 같은 global step에서 resume해야 한다. "training starts at a different loss than expected"로 드러나는 resume bug다.

---

## 연결과 다음 내용

- **[[fsdp-sft]] / ch-05** — FSDP mechanics와 memory formula(이 장).
- **[[sequence-packing]] / [[loss-masking-prompt]] / ch-04** — packing + masking은 FSDP의 per-step memory 이득을 실제로 만드는 요소다.
- **[[mixed-precision]] / ch-02** — `MixedPrecision(param_dtype, reduce_dtype)`가 per-shard precision을 선택한다.
- **[[gradient-clipping]] / ch-01** — distributed-clip-norm bug.
- **ch-06 (checkpointing + resume)** — sharded checkpoint save/load와 bit-exact resume.
- **ch-07 (failure modes)** — silent-failure list를 확장하고 위 checklist를 debug한다.
- **ch-52–56 (infrastructure internals)** — verl / OpenRLHF / TRL source-level deep dive.

## 더 읽을거리

- [[fsdp-sft]] — Zhao 2023; PyTorch의 industry-scale FSDP implementation.
- DeepSeek-V3 technical report([[deepseek-v3]]) — DualPipe AllToAll hiding을 갖춘 frontier 3D + EP parallelism.
- Megatron-LM paper — tensor + pipeline parallelism foundation.
- Karpathy의 "recipe"([[karpathy-training-neural-net-recipe]]) — "scale up slowly; add parallelism after the single-GPU recipe works."

## 함께 보는 시각화

**[figures/fsdp-memory.html](figures/fsdp-memory.html)** — DDP / ZeRO-2 / ZeRO-3 / FSDP FULL_SHARD 전반의 interactive per-GPU memory calculator. parameter count, GPU count, precision(bf16 / fp16 / fp32), activation-checkpointing toggle slider가 있다. 왜 DDP가 ~7B에서 포기하고, 70B에서 FSDP + activation-ckpt가 floor인지 확인하는 데 사용하라.
