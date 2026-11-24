# 05장 — Batch Size, Sequence Length, Tokens Seen을 바꾸는 분산 학습 선택

> **핵심 통찰.** 병렬성 layout은 배치 결정이며, 그 대부분은 모델이 학습하는 내용을 바꾸지 않는다. 바꾸는 수량은 네 가지다. global batch의 token 수, sequence length, 총 tokens seen, 각 batch의 구성이다. Llama 3.1 405B는 사전학습 중 서로 다른 세 layout, 즉 8,192 GPU에서 TP 8 / CP 1 / PP 16 / DP 64, 16,384 GPU에서 TP 8 / CP 1 / PP 16 / DP 128, 그리고 sequence length 131,072에서 TP 8 / CP 16 / PP 16 / DP 8을 사용했고, 세 경우 모두 batch당 token을 16M으로 유지했다([[llama-3]] Table 4). 이 수량을 고정해야 세 layout을 하나의 학습 실행으로 비교할 수 있다.
>
> **지침.** 병렬 degree를 바꿀 때는 무엇보다 먼저 global batch를 token으로, data-parallel group당 sequence 수를 다시 진술하라. Llama 3는 8,192-GPU와 16,384-GPU layout 사이 MFU가 43%에서 41%로 떨어진 이유를 batch당 global token을 일정하게 유지하는 데 필요했던 DP group당 더 작은 batch로 돌리기 때문이다([[llama-3]] §3.3.2). Tensor parallelism과 data parallelism을 설정한 뒤 목표 sequence length가 더는 들어가지 않으면 batch를 줄이지 말고 context parallelism을 추가하라. Llama 3는 CP 16으로 batch당 16M token을 유지하면서 sequence length 131,072에 도달했다(Table 4). Context-parallel rank 사이 interconnect가 intra-node가 아니라 inter-node이면 ring attention보다 all-gather-K/V 또는 all-to-all 방법을 선호하라. Ring attention은 A100 InfiniBand에서 통신을 숨기려면 약 149,500 token의 local sequence가 필요하지만 A100 NVLink에서는 약 6,200 token이면 되기 때문이다([[ring-attention]] Table 2).

---

## 일반 목적 모델에서 이 장이 중요한 이유

파이프라인상 위치는 pre-training과 mid-training이며 같은 산술은 SFT와 RL에 다시 나타난다. [[ch-03]]은 schedule, batch size, initialization을 학습 결정으로 고정했다. [[ch-04]]는 training sequence 안에 무엇이 들어가는지를 고정했다. 이 장은 그 결정을 많은 device에서 실행하는 machinery를 다루며, 그 machinery 중 결정을 바꿀 수 있는 부분으로 범위를 제한한다.

이 장의 수량 중 두 가지는 결과 모델의 breadth를 직접 정한다.

- **Tokens seen**은 domain, language, format의 coverage에 지불하는 예산이다. Llama 3의 scaling-law fit은 3.8 × 10^25 FLOPs에서 compute-optimal 점을 402B 파라미터와 16.55T token으로 두었고, 공개된 405B 모델은 15.6T token으로 학습했다([[llama-3]] §3.2.1, Abstract). Throughput은 고정된 cluster-time 예산이 산출하는 token 수를 정하므로, MFU를 절반으로 만드는 layout은 고정된 예산으로 지불할 수 있는 breadth도 절반으로 만든다. 배분 문제 자체는 [[ch-08a]]에 속한다.
- **Sequence length**는 무엇을 학습할 수 있는지 자체의 경계를 정한다. Layout이 131,072-token sequence 하나를 담지 못하면 long-context stage는 가능하지 않으며, context parallelism이 제거하는 제약이 바로 그것이다.

남은 수량인 **batch composition**은 오류가 발생하지 않은 채 가장 쉽게 손상된다. Intended mixture는 corpus의 속성이고, realized mixture는 shard를 data-parallel rank에 배정한 방식과 resume 뒤 data loader가 복원한 것의 속성이다.

Component별 byte ledger, attention kernel, activation checkpointing 정책, parallelism taxonomy 자체를 포함한 memory mechanics는 학습자의 별도 training-memory course, ch-06(“The Attention Kernel Zoo: SDPA, xFormers, SageAttention, Ring, Paged”) 및 ch-07(“Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP”)에서 다룬다. 이 장에서는 다시 유도하지 않는다.

---

## §1 참고용 다섯 축

각 축은 training computation의 한 차원을 분할한다. 표는 무엇이 나뉘는지, 무엇이 통신되는지, 그 축이 학습 수량을 바꿀 수 있는지를 진술한다.

| 축 | 분할되는 것 | layer 또는 step당 collective | 학습 수량을 바꾸는가? |
|---|---|---|---|
| Data parallel(DP) | batch, 그리고 sharding 시 parameter·gradient·optimizer state | AllGather + ReduceScatter(sharded) 또는 AllReduce(replicated) | 예 — DP degree × rank당 batch가 global batch |
| Tensor parallel(TP) | layer 안 개별 weight matrix | 각 block 안 AllReduce | global batch를 고정하면 아니오 |
| Pipeline parallel(PP) | layer를 stage로 | stage 사이 point-to-point activation | 간접적 — micro-batch 수가 batch와 상호작용 |
| Context parallel(CP) | 한 sequence 안 position | K/V all-gather, all-to-all, 또는 ring rotation | 예 — CP가 학습 가능한 최대 sequence length를 정함 |
| Expert parallel(EP) | rank에 걸친 MoE layer의 expert | all-to-all dispatch 및 combine | 예 — routing balance가 어떤 expert가 어떤 data를 보는지 바꿈 |

**정의.** *Global batch*는 하나의 optimizer step에 평균되는 그래디언트의 token 수다. *Tokens per step*은 같은 수량이다. *Tokens seen*은 global batch에 optimizer step 수를 곱한 값이다. *World size*는 참여 device 수다.

이들을 연결하는 항등식은 다음과 같다.

```
world_size = TP × PP × CP × DP
global_batch_tokens = DP × sequences_per_DP_group × sequence_length
tokens_seen = global_batch_tokens × optimizer_steps
```

`TP`, `PP`, `CP`, `DP`는 네 축의 degree다. `sequences_per_DP_group`은 한 data-parallel group이 optimizer step마다 소비하는 training sequence 수이며, 하나 이상의 micro-batch에 걸쳐 누적될 수 있다.

**하나의 수식으로 본 sharded data parallelism.** FSDP의 단일 control은 sharding factor `F`로, 하나의 parameter가 shard되는 rank 수다. `F = 1`은 replication, `F = W`는 full sharding, `1 < F < W`는 hybrid sharding이다([[fsdp-sft]] §3.2). `Ψ`개 element의 모델이 element 수 `ψ_1 … ψ_N`인 `N`개 unit으로 나뉜다면 논문은 다음을 준다.

```
peak parameter memory ∈ O( Σ_i ψ_i / F  +  max_i ψ_i )
collectives per iteration ∈ O(N)
```

`Ψ`는 전체 parameter 수, `ψ_i`는 FSDP unit `i`의 element 수, `F`는 sharding factor, `N`은 unit 수다. 두 번째 항은 모델 전체가 아니라 **가장 큰 단일 unit**이다([[fsdp-sft]] §3.2.1). Mixed precision에서는 첫 항을 full-precision byte로, 두 번째 항을 low-precision byte로 센다(§4.4).

`G`개 accelerator가 있는 host들 위 `W`개 accelerator에서 `M` byte 모델 하나에 대해, GPU당 cross-host traffic은 full replication에서 `2M(W−1)/W`, full sharding에서 `3M(W−1)/W`, `F = W/G`인 hybrid sharding에서 `2M(W−1)/(GW)`다([[fsdp-sft]] §3.2.2). 처음 두 비율은 논문이 DDP 대비 “1.5x communication overhead and volume”이라고 진술한 값이다(§3.2.1).

[figures/fsdp-memory.html](figures/fsdp-memory.html)은 선택한 `(Ψ, N, F, W, G)`에 대해 두 수식을 계산하고 세 traffic regime를 나란히 보인다.

**배치.** Llama 3는 네 group을 `[TP, CP, PP, DP]` 순서로 배치한다. “가장 안쪽 parallelism은 가장 높은 network bandwidth와 가장 낮은 latency를 요구하므로 보통 같은 server 안으로 제한”되는 반면, 가장 바깥 FSDP는 “sharded model weight를 비동기적으로 prefetch하고 gradient를 reduce하여 더 긴 network latency를 견딜 수” 있기 때문이다([[llama-3]] §3.3.2). 405B 실행은 TP 8을 하나의 8-GPU server 안에 배치한다.

---

## §2 Global batch, step당 token, tokens seen

### 2.1 측정 가능하게 진술한 문제

동일한 `global_batch_tokens`, 동일한 sequence length, 동일한 data order를 만드는 두 layout은 수치적 재정렬을 제외하면 동일한 optimizer state sequence를 만든다. 그렇지 않은 두 layout은 서로 다른 실험을 실행한다. 측정 가능한 실패는 실제 tokens per step이 구성한 값과 다른 실행이다. 이는 schedule 구성의 어느 줄도 바꾸지 않고 [[ch-03]]의 learning-rate-per-token schedule을 바꾼다.

### 2.2 계산 예: Llama 3.1 405B, Table 4

인쇄된 [[llama-3]] Table 4는 다음과 같다.

| GPU | TP | CP | PP | DP | Seq. Len. | Batch size/DP | Tokens/Batch | TFLOPs/GPU | BF16 MFU |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38% |

각 행에서 항등식을 확인한다.

- 행 1: `8 × 1 × 16 × 64 = 8,192` device. `64 × 32 × 8,192 = 16,777,216` token으로 ≈ 16M이다.
- 행 2: `8 × 1 × 16 × 128 = 16,384`. `128 × 16 × 8,192 = 16,777,216` token이다.
- 행 3: `8 × 16 × 16 × 8 = 16,384`. `8 × 16 × 131,072 = 16,777,216` token이다.

행 1에서 행 2로 device 수는 두 배가 된다. Global batch는 16M으로 유지하므로 DP group당 sequence는 32에서 16으로 줄어든다. Llama 3는 MFU가 43%에서 41%로 낮아진 원인을 정확히 이것, 즉 “학습 중 batch당 global token을 일정하게 유지하는 데 필요했던 DP group당 더 낮은 batch size”로 돌린다([[llama-3]] §3.3.2). 행 2에서 행 3은 sequence length를 16× 높이고 CP를 1에서 16으로 올린다. 따라서 동일한 world size에서 DP는 128에서 8로 낮아지며, batch는 131,072 token sequence 16개로 DP group당 16 sequence를 유지해 16M으로 남는다.

**함의.** Global batch를 고정하고 cluster를 두 배로 늘리면 학습 문제를 바꾸지 않고 wall-clock hour당 완료하는 optimizer step 수를 높이며, 효율에는 일부 비용이 든다. Cluster와 batch를 모두 두 배로 늘리는 것은 다른 결정이고 §2.4에서 평가한다.

[figures/parallelism-layout.html](figures/parallelism-layout.html)은 네 degree와 batch 정의를 설정하게 하고 world size, DP group당 sequence, 누적 micro-step, divisibility flag를 보고한다. Llama 3의 세 행과 OLMo 3의 다섯 stage는 preset으로 불러올 수 있다.

### 2.3 계산 예: DP degree가 유도되는 OLMo 3

OLMo 3는 pre-training, mid-training, long-context stage라는 세 단계로 학습하며([[olmo-3]]), 단계별 script가 공개되어 있다. OLMo-core script는 batch와 CP degree는 출력하지만 DP degree는 출력하지 않는다. GPU 수는 repository README의 stage table에서 온다([[olmo-core-olmo3-configs]]). 이 장의 모든 수치 OLMo 3 값은 그 script와 README에서 가져온다.

OLMo 3 32B long-context: H100 1,024개, `DEFAULT_SEQUENCE_LENGTH = 65536`, `GLOBAL_BATCH_SIZE = 8 * 1024 * 1024`, `cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4)`, `rank_microbatch_size=sequence_length`.

1. DP degree: script에 TP와 PP가 없으므로 `1024 / 8 = 128`.
2. Step당 sequence: `8,388,608 / 65,536 = 128`.
3. DP group당 sequence: `128 / 128 = 1`.
4. Micro-step: `rank_microbatch_size`가 sequence 하나와 같으므로 micro-step 하나이며 gradient accumulation이 없다.
5. Attention 내부 device당 token: `65,536 / 8 = 8,192`. 이것은 script 자체 주석의 값이기도 하다. `# 64k tokens per instance -> 8k tokens per device`.

OLMo 3 7B long-context: H100 256개, CP 8, `GLOBAL_BATCH_SIZE = 65536 * 64`. DP는 `256 / 8 = 32`, step당 sequence는 `4,194,304 / 65,536 = 64`, DP group당 sequence는 `64 / 32 = 2`다. `rank_microbatch_size=sequence_length`이므로 이 실행은 optimizer step당 micro-step 2개를 누적한다. Script는 같은 줄에 `# for CP we want only 1 instance per rank`라고 주석을 단다.

**조건과 한계.** 이 유도는 README의 GPU 수와 script가 동일 실행에 속한다는 가정을 둔다. README는 각 stage 행을 해당 script에 연결하여 이를 주장한다. Script는 MFU와 world-size product를 출력하지 않으므로 위 DP degree는 보고된 값이 아니라 유도된 값이다.

### 2.4 추가 width가 optimizer step 절약을 멈추는 batch size

[[large-batch-training-noise-scale]]은 고정 목표 도달에 필요한 optimizer step `S`와 처리한 example `E`의 쌍곡선 trade-off에서 critical batch size `B_crit`을 정의한다.

```
S/S_min − 1 = (E/E_min − 1)^(−1),   B_crit = E_min / S_min
```

`S_min`은 모든 batch size에 걸친 최소 step, `E_min`은 최소 example 수, `B_crit`은 실행이 최소 step의 2×와 최소 example의 2×를 사용하는 batch다(Eq. 2.11–2.12, §2.3).

논문 자체 관계식(App. D.1, 최소값 대비 `δS = 1 + B_noise/B`, `δE = B + B_noise`)의 계산 예로 `B_noise = 1,000`을 둔다.

| Batch B | 최소값 대비 step | 최소값 대비 example |
|---|---|---|
| 250 | 5× | 1.25× |
| 1,000 | 2× | 2× |
| 4,000 | 1.25× | 5× |

각 쌍은 Eq. 2.11을 만족한다. `B_crit`를 넘으면 추가 data-parallel width는 절약되는 step이 아니라 소비되는 example로 바뀐다. 논문의 `B_crit` 추정량은 simple gradient noise scale `B_simple = tr(Σ)/|G|²`이다. 이는 `B_crit`이 20에서 1천만 초과까지인 8개 과제에서 order-of-magnitude 수준으로 `B_crit`과 맞았다(§3, Table 1, Fig. 4). `B_crit`은 손실이 내려갈수록 올라간다. Billion Word는 시작 시 700에서 학습 평균 100,000으로 이동했다(Table 1).

**조건과 한계.** Table 1의 과제는 frontier LLM pre-training이 아니라 MNIST부터 Dota까지 규모이며, `B_simple / B_crit` 비율 자체도 과제 간 약 한 order of magnitude로 달라진다(§5). Production run은 대신 batch를 ramp한다. Llama 3.1 405B는 sequence 4,096에서 4M token으로 시작해 2.87T token 뒤 16M으로 갔다([[llama-3-recipe]], v3 §3.4.1). DeepSeek-V3는 첫 469B token에 걸쳐 batch를 3,072에서 15,360으로 높였다([[deepseek-v3-recipe]] §4.2; 보고서는 이 수의 단위를 출력하지 않는다).

**일반 목적 모델에 대한 함의.** Batch size는 systems decision이 아니라 data-efficiency decision이다. 오류를 내지 않은 채 global batch를 `B_crit`보다 높이는 cluster 확장은, domain breadth에 지불하는 같은 token budget을 중복된 gradient averaging에 사용한다.

---

## §3 Rank 간 data sharding과 mixture 보존 여부

### 3.1 정의와 측정 가능한 문제

**Intended mixture**는 각 source가 corpus에 기여하는 token share다. **Realized mixture**는 optimizer가 실제로 본 global batch sequence에 각 source가 기여하는 token share다. Shard assignment, sampling, resume logic이 rank가 읽는 file을 바꿀 때 둘은 달라진다. 측정 가능한 증상은 mixture configuration의 변화 없이 domain별 held-out loss가 움직이는 것이다.

### 3.2 메커니즘

1. Corpus를 source subset마다 하나의 file인 번호 매긴 shard로 token화한다.
2. Data-loader seed가 instance 전체의 permutation을 고정하고, 각 DP rank는 그 permutation의 strided 또는 contiguous slice를 가져간다.
3. 각 rank의 slice를 순서대로 읽으며, 한 step의 global batch는 그 step에서 모든 DP rank slice의 합집합이다.
4. Resume 시 loader position을 복원해야 하며, 그렇지 않으면 permutation이 다시 시작되어 이미 본 instance를 재소비한다.

OLMo 3 7B long-context script는 네 step을 모두 명시한다([[olmo-core-olmo3-configs]]; 32B script는 구조가 같고 mixture 이름과 `num_workers`만 다르다).

```python
dataset_config = NumpyPackedFSLDatasetConfig.from_data_mix(
    DataMix.OLMo_longmino_mix_0625,
    ...
    generate_doc_lengths=True,  # enables intra-document masking
    source_group_size=8,
    source_permutation_seed=123,
)

data_loader_config = NumpyDataLoaderConfig(
    global_batch_size=GLOBAL_BATCH_SIZE,
    seed=34521,
    num_workers=4,
)
```

`source_group_size=8`은 source를 permutation 전에 묶고, `source_permutation_seed=123`은 그 permutation을 고정한다. `seed=34521`은 instance 순서를 고정한다. 이 세 정수 중 어느 하나가 다른 두 실행은 같은 corpus를 서로 다른 step별 composition으로 본다.

### 3.3 Resume은 correctness 결정만이 아니라 mixture 결정이다

같은 script는 stage-2 checkpoint를 다음과 같이 불러온다.

```python
load_path=".../Olmo-3-1025-7B/stage2/step47684/",
load_strategy=LoadStrategy.always,
load_trainer_state=False,
load_optim_state=True,
```

`load_trainer_state=False`와 `load_optim_state=True`는 optimizer moment는 유지하고 data-loader position을 포함한 trainer 자체 state는 버린다. 여기서는 stage 3가 stage 2와 다른 mixture(`OLMo_longmino_mix_0625`)로 학습하므로 의도된 동작이다. 동일 stage의 *restart*에서 같은 두 flag를 사용하면 optimizer state를 유지하면서 data order를 instance zero에서 다시 시작한다. 따라서 실행은 permutation의 앞부분을 재소비하고 tail에 절대 도달하지 않는다. 재소비 data가 in-distribution이므로 error가 발생하지 않고 loss curve는 매끄럽게 계속된다(해석; flag와 그 semantics는 공개 script에서, failure mode는 이 course가 그 flag를 읽어낸 결과다).

### 3.4 DP degree 변경이 composition에 하는 일

Global batch가 고정되면 DP를 높일수록 DP group당 sequence는 줄어든다(§2.2). 그 수가 1에 도달하면 optimizer step은 정확히 `DP`개의 sequence를 뽑는다. 따라서 step별 mixture는 permutation에서 크기 `DP`인 sample이며, global batch를 높이지 않고서는 더 representative하게 만들 수 없다. OLMo 3 32B pre-training은 이 경계에 있다. Sequence 8,192에서 global batch 8,388,608 token은 1,024 sequence를 주며 `DP = 1,024`이므로, 각 rank가 step당 정확히 sequence 하나를 기여한다([[olmo-core-olmo3-configs]]). 두 가지 결과가 따른다. `sequences_per_step`이 `DP`의 정수배가 아닌 layout은 loader가 padding 또는 dropping을 해야 하며, 어느 선택도 실제 tokens per step을 바꾼다. 또 rank당 sequence 하나일 때는 자기 shard가 우연히 단일 source가 된 rank를 완화할 per-rank averaging이 남지 않는다. 따라서 mixture specification만이 아니라 shard assignment가 한 step에 무엇이 들어가는지를 정한다.

**일반 목적 모델에 대한 함의.** Domain breadth는 mixture에 명시되고([[ch-13]]), loader가 전달한다. 전달 단계에는 자체 seed, 자체 sharding, 자체 resume semantics가 있으며, 이들 어느 것도 mixture specification에는 보이지 않는다.

---

## §4 긴 컨텍스트: context 및 sequence parallelism

이 절은 한 device가 담을 수 있는 것보다 긴 sequence를 global batch를 바꾸지 않고 학습하는 방법을 다룬다.

### 4.1 세 가지 방법

**Ring attention**([[ring-attention]]). 각 host는 query block 하나를 보유하고 key-value block은 ring을 따라 회전한다. Host는 보유한 K/V block에 대해 blockwise attention을 계산하는 동안 그 block을 다음으로 보내고 다음 block을 받으므로 통신은 계산과 겹친다(§3, Figure 2a). Abstract는 이 방법이 “근사에 의존하거나 추가 통신 및 계산 overhead를 발생시키지 않고, 이전 memory-efficient Transformer가 달성 가능한 것보다 device 수배 더 긴 sequence의 학습 및 추론을 가능하게 한다”고 말한다.

**DeepSpeed Ulysses**([[deepspeed-ulysses]]). Sequence는 `P` device에 걸쳐 `N/P`로 나뉜다. Attention 전에 Q, K, V의 all-to-all은 각 device에 서로 겹치지 않는 attention head subset에 대한 전체 sequence를 준다. Head별 attention을 계산한 뒤 두 번째 all-to-all이 output을 sequence-parallel layout으로 돌려보낸다(§3.1, Figure 2).

**All-gather-K/V context parallelism**([[llama-3]] §3.3.2). 각 rank가 K와 V를 all-gather한 뒤 local Q chunk의 attention을 계산한다. Llama 3는 ring 구조 대신 이를 고른 두 이유를 진술한다. “document mask와 같은 여러 attention mask 유형을 all-gather 기반 CP attention에서 지원하기가 더 쉽고 유연하다.” 또 GQA 사용으로 gathered K와 V가 “Q tensor보다 훨씬 작으므로”, `O(S²)` attention과 `O(S)` all-gather에서는 “all-gather overhead가 무시 가능하다.”

### 4.2 통신: 각 출처가 인쇄한 수치

Ulysses는 닫힌 형식을 준다(§3.2). Hidden size `h`, sequence length `N`, parallel degree `P`에서 `P` GPU에 걸친 aggregate message `M`의 all-to-all GPU link당 volume은 `M/P`다. Ulysses는 layer마다 `3Nh`(QKV) all-to-all 하나와 `Nh`(output) all-to-all 하나를 수행하므로 다음과 같다.

```
Ulysses per-link volume     = 4Nh / P      →  O(N/P)
Megatron-LM SP per-link vol = 4Nh          →  O(N), P times larger
```

Megatron 비교는 논문 자체의 것이다. Layer마다 `Nh` all-gather 두 번과 reduce-scatter 두 번을 수행하며, `P ≫ 1`이면 각각 `M/P`가 아니라 `M`의 비용이다(§3.2). Ulysses volume은 “N과 P를 비례하게 함께 높이면 일정하다.”

Ring attention은 volume 대신 hardware 조건을 준다(§3). Hidden size `d`에서 block 크기 `c`인 blockwise attention의 비용은 `4dc²` FLOP이고, K/V block 비용은 `4cd` byte다. 따라서 host당 `F` FLOPS와 bandwidth `B`에서는 overlap을 위한 최소 block 크기가 `c = F/B`, 최소 local sequence가 `s = 6c`다(Table 2 caption). Ring-attention 논문은 FLOPS에 `F`를 쓰며, 이 장 §1의 FSDP sharding factor `F`와는 무관하다.

Table 2의 다섯 행 중 세 행을 재현한다. TPU v4 및 TPU v5e 행과 HBM 열은 생략한다.

| Host당 사양 | FLOPS(TF) | interconnect bandwidth(GB/s) | 최소 block 크기(×1e3) | 최소 local sequence(×1e3) |
|---|---:|---:|---:|---:|
| A100 NVLink | 312 | 300 | 1.0 | 6.2 |
| A100 InfiniBand | 312 | 12.5 | 24.5 | 149.5 |
| TPU v3 | 123 | 112 | 1.1 | 6.6 |

계산해서 읽으면 FLOPS 대 bandwidth 비는 NVLink에서 `312e12 / 300e9 ≈ 1.04e3`, InfiniBand에서 `312e12 / 12.5e9 ≈ 25.0e3`이므로 bandwidth 비 `300 / 12.5 = 24`는 block 크기로 곧장 이어진다. 최소 local sequence는 `6c`로 이어진다. 재계산한 InfiniBand 항목은 `25.0e3`, `149.8e3`이며 표가 인쇄한 `24.5`, `149.5`와의 차이는 출처의 반올림 차이이지 다른 주장이 아니다. 이 local sequence length 아래에서는 K/V 전송이 숨겨지지 않아 방법의 “추가 overhead 없음” 성질이 성립하지 않는다([[ring-attention]] Table 2, §3).

### 4.3 Production run이 선택한 것

Llama 3.1 405B long-context stage: GPU 16,384개, TP 8, **CP 16**, PP 16, DP 8, sequence 131,072, batch당 16M token, 380 TFLOPs/GPU, BF16 MFU 38%([[llama-3]] Table 4). Sequence는 `2 × CP` chunk로 분할되며, “i번째 CP rank는 i번째 및 (2 × CP − 1 − i)번째 chunk 둘 다를 받았다.” 이는 causal mask가 불균등하게 만드는 일을 균형화한다(§3.3.2).

OLMo 3 long-context stage: sequence 65,536에서 7B(H100 256개)와 32B(H100 1,024개) 모두 `cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4)`를 쓴다([[olmo-core-olmo3-configs]]). Constructor 이름은 ring attention이 아니라 all-gather-K/V 방법을 가리킨다.

**통제된 결과가 아닌 재현된 관행**: Meta(Llama 3.1 405B)와 Ai2(OLMo 3)라는 두 독립 그룹은 long-context stage에 all-gather-K/V context parallelism을 선택했고, 둘 모두 그와 함께 document mask를 진술하거나 구성한다([[llama-3]] §3.3.2; OLMo 3 script는 `generate_doc_lengths=True  # enables intra-document masking`를 설정).

**조건과 한계.** 어느 출처도 CP 방법을 throughput과 대조해 ablation하지 않는다. Ulysses의 보고된 이득, 즉 “기존 방법 SOTA baseline보다 4× 긴 sequence length에서 2.5× 빠름”, peak의 54%를 넘는 175 TFLOPs/GPU, 최대 A100 256개, 1.2B GPT에서 1M-token sequence는 all-gather CP가 아니라 GPT model에서 Megatron-LM sequence parallelism을 대상으로 측정한 것이다([[deepspeed-ulysses]] Abstract, §1, §4.1).

**해석.** Ulysses는 `P` device 전반에서 attention을 head-parallel하게 계산하므로 `P`는 나눌 수 있는 attention head 수에 의해 제한된다. 논문은 head-parallel 구조를 진술하지만(§3.1, §3.4) divisibility rule은 인쇄하지 않으므로, 이 한계는 보고된 값이 아니라 함의로 다룬다. KV head가 8개인 GQA, 즉 OLMo 3 32B 구성([[olmo-core-olmo3-configs]])에서는 그 한계가 query-head 수에서 보이는 것보다 더 빡빡하다.

**일반 목적 모델에 대한 함의.** CP는 긴 컨텍스트 능력 자체를 학습 가능하게 만드는 요소다. [[ch-32b]]는 context-extension stage의 data mixture와 short-context regression을 다루고, [[ch-32c]]는 claimed context length와 effective context length의 차이를 다룬다. 이 장은 layout이 sequence를 담을 수 있는지만 다룬다.

---

## §5 Expert parallelism과 load balancing

### 5.1 정의와 문제

Mixture-of-Experts layer에서 각 token은 expert FFN 일부로 routing된다. **Expert parallelism**은 서로 다른 expert를 서로 다른 rank에 배치하므로 routing은 all-to-all dispatch와 all-to-all combine이 된다. 측정 가능한 문제는 load imbalance다. Routing이 token을 몇 expert에 집중시키면 그 rank가 step의 latency가 되고, 흔한 해결책인 auxiliary balancing loss는 모델이 달리 학습했을 routing을 바꾼다.

### 5.2 메커니즘: auxiliary-loss-free balancing

DeepSeek-V3는 affinity score에 expert별 bias `b_i`를 **top-K 선택에만** 더한다. Gate 값은 여전히 편향 없는 affinity `s_{i,t}`를 사용한다. 매 step 뒤 overloaded expert의 `b_i`는 `γ`만큼 감소하고 underloaded expert의 것은 `γ`만큼 증가한다([[deepseek-v3]] §2.1.2, Eq. 16). 한 sequence 안의 극단적 imbalance를 막기 위해 작은 `α`의 sequence-wise balance loss는 유지한다(Eq. 17–20). 구성 값은 첫 14.3T token에서 `γ = 0.001`, 남은 500B에서 `0.0`, 그리고 `α = 0.0001`이다([[deepseek-v3-recipe]] §4.2).

Bias는 selection에 들어가지만 gate에는 들어가지 않으므로, balancing pressure는 language-modelling loss의 gradient에 나타나지 않는다.

### 5.3 근거

[[deepseek-v3]] Table 5는 두 규모, 즉 1.33T token의 15.7B-total(2.4B active) 모델과 578B token의 228.7B-total(20.9B active) 모델에서 auxiliary-loss-based와 auxiliary-loss-free를 비교한다. 표의 benchmark 10개 행 중 7개를 여기 재현하며 DROP, TriviaQA, NaturalQuestions는 생략한다.

| Benchmark(metric) | Small, aux-loss | Small, aux-free | Large, aux-loss | Large, aux-free |
|---|---|---|---|---|
| Pile-test(BPB, 낮을수록 좋음) | 0.727 | 0.724 | 0.656 | 0.652 |
| BBH(EM, 3-shot) | 37.3 | 39.3 | 66.7 | 67.9 |
| MMLU(EM, 5-shot) | 51.0 | 51.8 | 68.3 | 67.2 |
| HumanEval(pass@1) | 22.0 | 22.6 | 40.2 | 46.3 |
| MBPP(pass@1, 3-shot) | 36.6 | 35.8 | 59.2 | 61.2 |
| GSM8K(EM, 8-shot) | 27.1 | 29.6 | 70.7 | 74.5 |
| MATH(EM, 4-shot) | 10.9 | 11.1 | 37.2 | 39.6 |

**Result(단일 연구).** Auxiliary-loss-free 전략은 두 규모 모두에서 대부분 행에서 앞서며, large 규모 MMLU와 small 규모 MBPP에서는 뒤처진다. Cell당 seed 하나이며 보고서는 error bar를 주지 않는다.

보고서가 제시하는 메커니즘은 scope다. Batch-wise balancing은 “각 sequence에서 in-domain balance를 강제하지 않으므로”, “expert가 서로 다른 domain에 더 잘 specialize할 수 있게 한다.” 16B에서 Pile domain별 expert load는 auxiliary-loss-free 모델에서 더 큰 specialization을 보인다(§4.5.3, Figure 9). Batch-wise auxiliary loss는 auxiliary-loss-free 방법과 같은 validation loss에 이른다. 1B MoE는 2.258(sequence-wise) 대 2.253(auxiliary-loss-free) 대 2.253(batch-wise), 3B MoE는 2.085 대 2.080 대 2.080이다(§4.5.3).

**조건과 한계.** 보고서는 batch-wise balancing의 효율 과제 두 개를 지명하며, 첫째는 batch 안의 load imbalance다(§4.5.3). 1B 및 3B validation-loss 비교는 downstream 평가가 아니라 validation loss만 비교한다.

### 5.4 Layout, 그리고 routing 선택이 통신 선택인 이유

DeepSeek-V3의 학습 layout은 H800 GPU 2,048개 cluster에서 “16-way Pipeline Parallelism(PP), 8 node에 걸친 64-way Expert Parallelism(EP), ZeRO-1 Data Parallelism(DP)”이며 **tensor parallelism은 없다**. Memory optimization이 “비용이 큰 Tensor Parallelism(TP)을 사용하지 않고 DeepSeek-V3를 학습할 수 있게 한다”([[deepseek-v3]] §3.1, §3.2; [[deepseek-v3-recipe]] §3.2, §4.2).

Node-limited routing은 token마다 `M = 4` node로 제한한다(§2.1.2). Kernel 설계는 bandwidth 차이를 직접 사용한다. “NVLink는 160 GB/s bandwidth를 제공하며 IB(50 GB/s)의 약 3.2배”이므로, 각 token은 target node마다 일치하는 in-node index로 IB를 통해 보내고, 거기서 NVLink를 통해 전달된다. 보고서가 이로부터 도출한 결과는 token 하나가 “추가 overhead 없이 node당 평균 3.2 expert를 선택”한다는 것이다. 실제로 routed expert 8개를 선택하지만, 동일 통신 비용을 유지하면서 “최대 13 expert(4 node × 3.2 expert/node)까지 이 수를 확장할 수” 있고, “IB와 NVLink bandwidth를 완전히 활용하는 데는 SM 20개면 충분하다”(§3.2.2). DualPipe는 forward와 backward chunk pair 전반에서 computation과 communication을 겹치는 pipeline schedule(§3.2.1)이고, cross-node all-to-all kernel은 별도 기여다(§3.2.2).

Balancing이 성공하므로 “DeepSeek-V3는 학습 중 어떤 token도 drop하지 않는다”(§2.1.2). 즉 routing layer가 버리는 training token은 없다.

**일반 목적 모델에 대한 함의.** Load balancing은 효율 메커니즘이 모델이 무엇을 학습하는지에 닿는 지점이다. Sequence별 balancing pressure는 모든 sequence가 모든 expert를 쓰도록 강제하며, 이는 batch별 balancing이 허용하는 domain specialization을 억제한다. DeepSeek-V3는 sequence별 버전에 대해 작지만 일관된 downstream 비용을 측정한다(§4.5.3, Table 5).

---
## 레시피

행은 공개 실행에서 인쇄한 값이다. `상태`는 명시된 날짜에 해당 위치를 읽었음을 기록한다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-stable | 병렬성, 8K context | 8,192 GPU: TP 8, CP 1, PP 16, DP 64; seq 8,192; DP당 32 seq; batch당 16M token; 430 TFLOPs/GPU; BF16 MFU 43% | arXiv:2407.21783 v3 Table 4 | verified 2026-09-17 | ablation 보고 없음 |
| Llama 3.1 405B | 405B | pretrain-stable | 병렬성, 8K context, 더 큰 cluster | 16,384 GPU: TP 8, CP 1, PP 16, DP 128; DP당 16 seq; batch당 16M token; 400 TFLOPs/GPU; MFU 41% | v3 Table 4, §3.3.2 | verified 2026-09-17 | §3.3.2는 global token 고정 시 DP group당 더 작은 batch를 MFU 43%→41% 하락 원인으로 돌림 |
| Llama 3.1 405B | 405B | long-context | 병렬성 | 16,384 GPU: TP 8, CP 16, PP 16, DP 8; seq 131,072; DP당 16 seq; batch당 16M token; 380 TFLOPs/GPU; MFU 38% | v3 Table 4 | verified 2026-09-17 | ablation 보고 없음 |
| Llama 3.1 405B | 405B | pretrain-stable | CP 방법 | K와 V all-gather 후 local Q chunk attention; sequence는 2×CP chunk, rank i는 chunk i 및 2·CP−1−i 보유 | v3 §3.3.2 | verified 2026-09-17 | mask 유연성과 GQA 아래 작은 K/V 때문에 선택; ablation 출력 없음 |
| Llama 3.1 405B | 405B | pretrain-stable | PP bubble ratio | rank당 V stage interleaved schedule; bubble ratio (PP−1)/(V·M), M = micro-batch | v3 §3.3.2 | verified 2026-09-17 | 측정된 bubble 수치는 출력하지 않음 |
| Llama 3.1 405B | 405B | pretrain-stable | batch schedule | seq 4,096에서 4M token → 252M token 후 “8M sequences of 8,192 tokens” → 2.87T token 후 16M | v3 §3.4.1 ([[llama-3-recipe]]) | verified 2026-09-14 | 초기의 더 작은 batch는 “학습 안정성 개선” 목적 |
| DeepSeek-V3 | 671B total / 37B active | pretrain-stable | 병렬성 | 16-way PP; 8 node에 걸친 64-way EP; ZeRO-1 DP; TP 없음; 2,048 H800 cluster | arXiv:2412.19437 v2 §3.1, §3.2 | verified 2026-09-17 | ablation 보고 없음 |
| DeepSeek-V3 | 671B / 37B active | pretrain-stable | node-limited routing | token당 최대 M = 4 node; node당 평균 3.2 expert; 통신에 SM 20개; NVLink 160 GB/s 대 IB 50 GB/s | v2 §2.1.2, §3.2.2 | verified 2026-09-17 | ablation 보고 없음 |
| DeepSeek-V3 | 671B / 37B active | pretrain-stable | balancing | 첫 14.3T token에서 bias update γ = 0.001, 마지막 500B에서 0.0; sequence-wise balance loss α = 0.0001 | v2 §4.2 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | 15.7B/1.33T 및 228.7B/578B의 Table 5; 1B 및 3B validation loss §4.5.3 |
| DeepSeek-V3 | 671B / 37B active | pretrain-stable | batch schedule | 첫 469B token 동안 3,072 → 15,360, 이후 15,360(단위 출력 안 함) | v2 §4.2 | verified 2026-09-14 | ablation 보고 없음 |
| OLMo 3 7B | 7B | long-context | layout 및 batch | H100 256개; CP degree 8(`...llama3(degree=8, head_stride=4)`); `hsdp`, `shard_degree=1`; seq 65,536; global batch 65536×64 ≈ 4.19M token; `rank_microbatch_size=sequence_length` | OLMo-core `src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py` + `README.md` ([[olmo-core-olmo3-configs]]) | verified 2026-09-17 | ablation 보고 없음 |
| OLMo 3 32B | 32B | long-context | layout 및 batch | H100 1,024개; CP degree 8; `hsdp`, `shard_degree=8`; seq 65,536; global batch 8,388,608 token; AC budget 0.3 | `OLMo-3-1025-32B-long-context.py` + `README.md` | verified 2026-09-17 | ablation 보고 없음 |
| OLMo 3 32B | 32B | pretrain-stable | layout 및 batch | H100 1,024개; `hsdp`, `shard_degree=64`; seq 8,192; global batch 8,388,608 token; AC budget 0.5 | `OLMo-3-1025-32B-pretrain.py` + `README.md` | verified 2026-09-17 | ablation 보고 없음 |
| OLMo 3 7B | 7B | long-context | resume flag | `load_trainer_state=False`, `load_optim_state=True`, `load_strategy=LoadStrategy.always` | `OLMo-3-1025-7B-long-context.py` | verified 2026-09-17 | stage 3는 stage 2와 다른 mixture 사용 |
| OLMo 3 7B / 32B | 7B, 32B | long-context | data-loader seed | `source_group_size=8`, `source_permutation_seed=123`, loader `seed=34521` | 같은 script | verified 2026-09-17 | ablation 보고 없음 |
| minGPT 175B(FSDP 논문) | 175B | benchmark | throughput | 128–512 A100 80GB에서 batch 1과 2에 각각 >173, >186 TFLOPS/GPU; 312 TFLOPS BF16 peak의 약 55%, 60% | arXiv:2304.11277 §5.4, Figure 7b | verified 2026-09-17 | full sharding + prefetch + rate limiter + activation checkpointing + BF16 + Adam(§5.4) |
| GPT 1.2B(Ulysses 논문) | 1.2B | benchmark | sequence scaling | 최대 1M token; 최대 A100 256개 평가; >175 TFLOPs/GPU(peak의 >54%); Megatron-LM SP 대비 throughput 2.5×와 sequence length 4× | arXiv:2309.14509 v2 Abstract, §1, §4.1 | verified 2026-09-17 | Megatron-LM sequence parallelism하고만 측정 |

**소규모 일반 목적 실행의 시작점.** 단일 8-GPU NVLink node에는 위 공개 행 중 적용되는 것이 없다. 모든 행은 multi-node frontier 실행이다. 이전 가능한 부분은 §1의 항등식과 검증된 두 조건이다. 첫째, Llama 3.1 405B가 세 layout에서 batch당 16M token으로 유지한 것처럼, device 수가 바뀌면 global batch token을 고정한다(v3 Table 4). 둘째, node에서 context parallelism을 사용한다면 [[ring-attention]] Table 2의 NVLink 행은 A100급 hardware(312 TF, 300 GB/s)에서 ring attention의 최소 local sequence를 약 6,200 token으로 준다. 그보다 낮으면 all-gather-K/V 또는 all-to-all 방법을 택한다. 위 표의 나머지 수는 모두 해당 실행 범위에 한정되며 이전해서는 안 된다.

---

## 일반화 관점

**(a) 폭을 늘리는 것.**

- Layout이 바뀌어도 step당 global token을 고정하면 cluster 크기 전반에서 학습 문제는 동일하므로, throughput 개선은 다른 최적화 trajectory가 아니라 tokens seen으로 전환된다([[llama-3]] Table 4: 세 layout 모두 batch당 16M token).
- Context parallelism은 batch를 줄이지 않고 긴 sequence 학습을 가능하게 한다. Llama 3.1 405B는 CP 16으로 sequence 131,072에서 batch당 16M token을 유지했다(Table 4). Long-context 능력은 아무리 많은 short-context data도 공급할 수 없는 breadth 축이다.
- Sequence별이 아니라 batch별 expert balancing은 domain specialization을 허용하며, 두 ablation 규모 모두에서 Table 5의 열 행 대부분에서 앞선다([[deepseek-v3]] Table 5, §4.5.3).

**(b) 협소화 또는 망각을 일으키는 것.**

- Critical batch size보다 큰 global batch는 token budget을 중복 averaging으로 바꾼다. `B = 4·B_noise`에서는 최소 step의 1.25×를 위해 최소 example의 5×가 필요하다([[large-batch-training-noise-scale]] Eq. 2.11–2.12, App. D.1).
- 목표 sequence를 담지 못하는 layout은 truncation 또는 더 짧은 stage를 강제하여, 모델이 보게 될 corpus에서 long-context 부분을 제거한다.
- Sharding 및 resume 오류는 specification을 바꾸지 않고 realized mixture를 바꾸므로, domain coverage는 loader가 반복하는 shard 방향으로 조용히 협소화된다(§3.3; 해석).
- Sequence별 balancing loss는 모든 sequence가 expert 전체에 분산되게 하며, [[deepseek-v3]] §4.5.3은 이를 expert specialization 감소 및 1B·3B에서 작은 validation-loss 비용과 연관 짓는다.

**(c) 이 단계에서 측정하는 법.**

- 매 step 실제 optimizer step당 token을 기록하고, 시작 시만이 아니라 매번 구성된 global batch와 비교한다.
- Step마다 source 또는 shard별 소비 token 수를 기록한다. Specification에서 mixture가 drift하면 평가지표보다 먼저 여기에서 드러난다.
- Llama 3가 Table 4 각 행에서 한 것처럼, layout이 바뀔 때마다 MFU 또는 TFLOPs/GPU를 기록하여 placement regression과 data regression을 분리한다.
- Aggregate loss 하나는 더 이상 도착하지 않는 domain을 숨긴다. Domain별 held-out loss를 추적한다. In-loop evaluation 설계는 [[ch-06]]에 있다.

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| Global batch를 다시 진술하지 않고 DP degree 변경 | schedule 구성은 바뀌지 않았는데 cluster resize 뒤 loss curve가 이동 | 전후 `DP × sequences_per_DP × sequence_length`를 재계산하고 기록한 step당 token과 비교 |
| Global batch가 `DP × sequence_length`의 정수배가 아님 | 실제 step당 token이 구성 값과 다르고 일부 rank가 idle | 시작 시 divisibility를 assert한다. [figures/parallelism-layout.html](figures/parallelism-layout.html)의 flag는 두 division 중 무엇이 실패했는지 보인다 |
| FSDP가 all-gather 중 항상 모델의 full copy를 보유한다고 가정 | 실행이 실제 사용하는 값보다 memory estimate가 훨씬 큼 | `O(Σψ_i/F + max_i ψ_i)`를 쓴다. 두 번째 항은 가장 큰 unit이며 block-wise wrapping이면 대략 block 하나다([[fsdp-sft]] §3.2.1) |
| TP group을 host 간에 배치 | batch 또는 data 변화 없이 MFU 하락 | group 순서 `[TP, CP, PP, DP]`와 TP group size가 host당 accelerator 수를 나누는지 확인([[llama-3]] §3.3.2) |
| overlap threshold 아래에서 ring attention 사용 | CP degree가 커질수록 throughput이 유지되지 않고 하락 | 실제 interconnect에서 `c = FLOPS/Bandwidth`, `s = 6c`를 계산해 local sequence length와 비교([[ring-attention]] Table 2) |
| all-to-all 방법에서 CP degree를 분할 가능한 head 수보다 크게 설정 | initialization 실패 또는 예상 밖 fallback path | model config에서 head 수를 읽고 GQA에서는 KV-head 수를 쓴다([[deepspeed-ulysses]] §3.1, §3.4; 해석) |
| optimizer state는 복원하지만 data-loader position은 복원하지 않음 | loss curve는 부드럽지만 corpus tail에 도달하지 않음 | source별 누적 소비 token을 기록하고 mixture specification과 비교(§3.3) |
| model family 사이에서 parallelism layout 복사 | report가 TP 없이 학습했다고 하는 model에 TP 적용 | 해당 위치의 layout을 읽는다. DeepSeek-V3는 PP 16, EP 64, ZeRO-1 DP, TP 없음을 사용했다([[deepseek-v3]] §3.2) |
| frontier MoE 규모에서 sequence별 balancing loss 적용 | 모든 sequence 안 expert load는 uniform, domain별 specialization은 없음 | [[deepseek-v3]] §4.5.3, Figure 9처럼 domain별 expert load를 기록 |

## 이해도 점검

1. Llama 3.1 405B는 8,192 GPU에서 16,384 GPU로 옮기면서 batch당 token을 16M으로 유지했다. MFU가 43%에서 41%로 하락한 이유와 batch를 고정하려고 무엇이 바뀌어야 했는지 설명하라.
2. Table 4의 행 3은 동일한 world size와 동일한 batch당 token에서 sequence length를 8,192에서 131,072로 높인다. 세 제약을 일관되게 만드는 CP와 DP degree를 유도하고 DP가 낮아져야 한 이유를 설명하라.
3. OLMo 3 7B long-context script는 “for CP we want only 1 instance per rank”라는 주석과 함께 `rank_microbatch_size=sequence_length`를 설정한다. GPU 256개, CP 8, sequence 65,536, global batch 65536×64 token일 때 optimizer step 하나가 누적하는 micro-step 수는 얼마이며, 왜 1이 아닌가?
4. 한 실행이 단일 stage 중간에 `load_optim_state=True`, `load_trainer_state=False`로 resume한다. Loss curve는 연속적이고 error도 없다. 모델이 무엇을 더 이상 보지 못하는지 설명하고 이를 드러낼 metric 하나를 이름 대라.
5. Ring attention은 추가 통신 overhead가 없다고 주장하며 Table 2에서 A100 NVLink는 약 6,200 token, A100 InfiniBand는 약 149,500 token의 최소 local sequence를 제시한다. 24× 차이를 만드는 요인과 local sequence가 threshold 아래일 때 무엇이 깨지는지 설명하라.
6. DeepSeek-V3의 auxiliary-loss-free bias는 top-K selection에는 들어가지만 gate value에는 들어가지 않는다. 이 위치가 그래디언트에 왜 중요한지, sequence별 대신 batch별 balancing의 downstream 결과로 무엇을 측정했는지 설명하라.
7. 두 engineer가 같은 throughput 개선을 제안한다. 한 명은 global batch 고정 상태로 cluster를 두 배로 하고, 다른 한 명은 cluster 고정 상태로 global batch를 두 배로 한다. [[large-batch-training-noise-scale]]의 steps-versus-examples 관계를 이용해 fixed target loss에서 각각이 tokens seen에 무엇을 바꾸는지 말하라.
8. Mixture specification이 code 17%라고 한다. Error 없이 optimizer가 실제 본 batch의 realized code share가 17%와 달라질 수 있는 이 장의 서로 다른 메커니즘 두 가지를 설명하라.

## 연결

- 이전 장: [[ch-04]] — *Sequence Packing, Loss Masking, and Chat Templates*. Packing은 sequence 하나에 들어가는 것을 정하며, 이 장은 그런 sequence가 optimizer step 하나에 몇 개 들어가는지 정한다.
- 의존성: [[ch-03]] — *Learning-Rate Schedules, Batch Size, Initialization, and Normalization*. 그 장에서 고른 batch size가 이 장의 layout으로 실제 구현되는지 여부를 다룬다.
- 다음 장: [[ch-06]] — *Checkpointing, In-Loop Evaluation, and Checkpoint Selection*. 이 장이 요구하는 per-domain 및 per-shard instrumentation과 resumable checkpoint가 포함해야 할 것을 설계한다.
- 이후: [[ch-08a]] — *Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability*. Tokens seen은 이 allocation의 입력이고, throughput은 compute budget을 그것으로 변환한다.
- 이후: [[ch-13]] — *Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale*. Intended mixture는 그 장에서 명시되고 §3의 sharding mechanics가 전달한다.
- 이후: [[ch-32b]] — *Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression* — 및 [[ch-32c]] — *Claimed versus Effective Context Length and Long-Context Evaluation*. §4는 layout이 sequence를 담을 수 있는지만 다룬다.
- 동위 course: training-memory ch-06(*The Attention Kernel Zoo*) 및 ch-07(*Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP*)는 이 장이 반복하지 않고 참조하는 memory ledger와 primitive별 유도를 담는다.

## 출처

- [[fsdp-sft]] — §1, Recipe: sharding factor `F`, peak parameter memory `O(Σψ_i/F + max_i ψ_i)`, 세 cross-host traffic 식, DDP 대비 1.5× 비율, GPT-175B throughput(장 사본: `excerpts/fsdp-sft.md`, 2026-09 재작성).
- [[llama-3]] — §1, §2, §4, Recipe: Table 4 layout 및 MFU, `[TP, CP, PP, DP]` 배치 규칙, `2 × CP` chunk assignment를 갖는 all-gather-K/V CP, interleaved pipeline bubble ratio.
- [[llama-3-recipe]] — §2.4, Recipe: 해당 위치와 함께 405B batch-size ramp.
- [[deepseek-v3]] — §5, Recipe: TP 없는 PP 16 / EP 64 / ZeRO-1 DP, node-limited routing 및 IB-to-NVLink kernel, auxiliary-loss-free balancing, Table 5, §4.5.3.
- [[deepseek-v3-recipe]] — §2.4, §5: 위치를 포함한 balancing coefficient와 batch-size schedule.
- [[olmo-core-olmo3-configs]] — §2.3, §3, §4.3, Recipe: 공개 OLMo 3 script 및 README stage table의 장 발췌(GPU 수, batch·sequence literal, `cp_config`, `dp_config`, loader seed, resume flag).
- [[olmo-3]] — §2.3: 세 단계 학습 절차만. 이 장의 수치 OLMo 3 값은 모두 그 대신 공개 script에서 왔다.
- [[ring-attention]] — §4: ring mechanism, `c = F/B`와 `s = 6c` overlap 조건, Table 2를 담은 장 발췌.
- [[deepspeed-ulysses]] — §4: 두 all-to-all step, link당 `4Nh/P` 대 `4Nh` volume, 결합된 DP 및 SP group에 대한 ZeRO-3, 보고된 scaling을 담은 장 발췌.
- [[large-batch-training-noise-scale]] — §2.4: steps-versus-examples 관계, `B_crit`, `B_simple = tr(Σ)/|G|²` 추정량, 8개 과제 전반의 Table 1 범위.

<!-- 인용 추적: [[fsdp-sft]] [[llama-3]] [[llama-3-recipe]] [[deepseek-v3]] [[deepseek-v3-recipe]] [[olmo-3]] [[olmo-core-olmo3-configs]] [[ring-attention]] [[deepspeed-ulysses]] [[large-batch-training-noise-scale]] -->
