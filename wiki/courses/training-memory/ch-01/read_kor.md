<!-- chapter: ch-01
     track: ledger
     title: The Memory Ledger: What Fills a GPU
     deps: []
     sources: [[transformer-math-101]], [[ultrascale-playbook]], [[ml-engineering-memory]]
-->

# 1장 - memory 원장: GPU를 채우는 것

> **핵심 통찰.** training memory는 하나의 단일 항목이 아니라, 각각 정확한 bytes-per-parameter formula를 따르는 여섯 가지 명시적 resident(각각 GPU memory를 점유하는 독립적인 구성요소)로 이루어집니다. mixed precision에서 AdamW를 사용하는 full fine-tuning의 static floor(activation을 제외하고도 반드시 필요한 최소 memory)는 단 하나의 activation도 저장하기 *전*부터 parameter당 16–18 bytes입니다. 따라서 27B model에는 optimizer, precision 또는 distribution strategy를 바꾸지 않는 한 줄일 수 없는 432–486 GB의 states가 필요합니다.

> **지침.** 모든 training run 전에 여섯 항목의 ledger(구성요소별 memory 사용량 명세)를 작성하십시오. weights(2 B/parameter, bf16) + gradients(2 B/parameter, trainable parameter에만 해당) + Adam states(12 B/parameter, fp32) + activations(아래 formula) + loss-head logit spike(B·T·V·2 B, transient) + overhead(CUDA context, NCCL, ZeRO gather buffers, 수 GiB)입니다. 이 합계를 통해 해당 training job이 GPU memory에 들어가는지 판단할 수 있습니다. Rule of 16(항목 1–3의 합이 16 B/parameter)은 빠른 1차 판별 기준입니다.

---

## 1. 6개 항목 원장

모든 training step은 정확히 여섯 가지 서로 다른 resident(독립적으로 memory를 점유하는 구성요소)에 memory를 allocation합니다. 어떤 strategy(LoRA, ZeRO, activation checkpointing, FP8)가 memory 사용량에 유의미한 변화를 만드는 이유를 이해하려면, 먼저 각 항목을 개별적으로 이해해야 합니다.

### 1.1 Weights — parameter당 2 B

bf16 mixed precision에서는 모든 parameter의 *working copy*가 2바이트를 차지합니다. 이는 모든 forward 및 backward matrix multiplication에 사용되는 copy입니다. 27B model → **54GB** working weights.

optimizer가 유지 관리하는 fp32 *master copy*도 항목 3에서 다룹니다. master copy는 별도의 원장 라인이 아닙니다. 이는 optimizer state allocation 내에 번들로 제공됩니다.

### 1.2 Gradients — 2B/parameter(trainable parameters만 해당)

Gradients는 working weights: bf16, 2B/param과 동일한 dtype입니다. 결정적으로, *training 가능한* parameter만 gradients를 accumulation합니다.

- **Full fine-tune:** 모든 parameter는 학습 가능합니다 → 2 B × N_total. 27B → **54GB** gradients.
- **LoRA (rank *r*):** adapter matrices(대상 linear layer당 2개의 low-rank projections)만 학습 가능합니다. Q/V projections에 rank 16이 있는 일반적인 LoRA의 경우 training 가능한 parameter count는 전체 model의 < 0.5%입니다. model 규모에서 Gradient memory ≒ 0입니다.

이것이 LoRA가 parameter-efficient training을 달성하는 주된 메커니즘이다. Weight bytes만 줄어드는 것이 아니라 gradient allocation이 거의 0이 되고, 그 결과 optimizer-state allocation도 거의 0이 된다.

[[ml-engineering-memory]]는 규칙을 정확하게 명시합니다: weights에 대해 "Mixed precision: 6 B/param (2 B bf16 working + 4 B fp32 마스터)", 별도로 "Gradient 바이트: 4 B/param (fp32 또는 혼합 절반), 혼합되지 않은 fp16인 경우 2 B." bf16 gradients를 사용하는 현대의 실제 수치는 2B/param입니다.

### 1.3 Adam Optimizer States — parameter당 12 B

AdamW는 training 가능한 parameter당 3개의 fp32 tensor를 유지합니다.

| Component | Dtype | B/parameter |
|---|---|---|
| fp32 master weights | fp32 | 4 |
| First moment(momentum,m) | fp32 | 4 |
| Second moment (variance, v) | fp32 | 4 |
| **합계** | | **12** |

27B model을 full fine-tune할 경우 → optimizer states **324GB**.

fp32 master copy는 bf16으로 training할 때도 필수다. Optimizer update rule(`weight -= lr * m / sqrt(v + eps)`)이 bf16의 quantization step보다 작은 update를 만들 수 있으며, 이 경우 update가 소실되기 때문이다. fp32 copy는 작은 update를 precision loss 없이 누적하고, 그 결과인 fp32 값만 bf16 working copy로 downcast한다. 전체 precision 설명은 [[mixed-precision-training]]을 다루는 [[ch-02]]를 참조하라.

> **▶ 인터랙티브 companion — [`figures/adamw-derivation.html`](figures/adamw-derivation.html)**
> *왜 정확히 tensor 3개이고, 왜 그중 하나는 optimizer의 것이 아닌가.* SGD → +momentum(`m`) → +second moment(`v`) → +bias correction → AdamW 순서로 항을 하나씩 붙이며, 매 step 숫자를 실제로 대입하고 step multiplier를 그래프로 보여줍니다. 여기서 드러나는 memory 논점: **state를 새로 들고 있는 항만 bytes를 낸다** — `m`과 `v`가 각각 4 B/param(따라서 Adam 고유 비용은 8 B/param)이고, bias correction과 decoupled weight decay는 공식만 길어질 뿐 **0 B**입니다. 나머지 4 B(fp32 master)는 Adam이 아니라 mixed precision에 청구됩니다. 그래서 `4+4+4+4 = 16`(순수 fp32) `= 2+2+12`(bf16 mixed) — §2의 Rule of 16은 precision의 산물이 아니라 *optimizer가 만드는 바닥*입니다.

12B/param 수치는 표준 AdamW formula에서 나온 것입니다. [[ml-engineering-memory]]는 대안을 항목화합니다.

- BF16 AdamW: 4B(quantization된 moment)
- SGD(momentum/LION/Adafactor 포함): 4B
- 8-bit quantized(bitsandbytes): 2 B

이는 distributed strategy를 적용하기 전에 static ledger *내부에서* 조정할 수 있는 optimizer-side lever(optimizer 선택으로 memory를 줄이는 수단)다.

### 1.4 Activations — 동적 항목

Activations는 forward pass 동안 생성된 중간 tensor이며 backward pass 동안 gradients를 compute하기 위해 유지되어야 합니다. 이것은 유일한 비정적 resident입니다. batch size 및 sequence length로 확장됩니다.

표준 formula([[transformer-math-101]], tensor-parallel 차수 *t* = 1):

```
m_act = s · b · h · L · (10 + 24 + 5·a·s/h) bytes
      = s · b · h · L · (34 + 5·a·s/h) bytes
```

어디:
- s = sequence length
- b = GPU당 micro-batch size
- h = hidden dimension
- L = layer 수
- a = attention heads의 수

[[ultrascale-playbook]]는 더 깔끔한 format인 `m_act = L · seq · bs · h · (34 + 5·n_heads·seq/h)`를 제공합니다.

`5·a·s/h · s` = `5·a·s²/h` 항은 sequence length에 대한 quadratic 종속성을 명시적으로 만듭니다. (34 + ...) 상수는 projections, MLP 및 LayerNorm activations를 포함합니다. `5·a·s/h` 용어는 attention score matrix(head당 `s×s`, head당 `a`)입니다.

**구체적인 예** ([[ml-engineering-memory]]): Llama-3-8B, batch=1, seq=32,768:
- checkpointing 제외: ~240GB
- 전체 gradient checkpointing 포함: ~31GB

이 8× 차이는 전적으로 attention의 quadratic 항에서 발생합니다. batch=1에서 sequence length는 유일한 자유 변수이며, memory 사용량은 linear가 아니라 quadratic하게 증가합니다.

> **대화형 자료:** [figures/memory-ledger.html](figures/memory-ledger.html) — (N_params, batch, sequence, hidden dimension, layer, head)의 여섯 항목 ledger를 채우고 slider를 움직일 때 각 resident가 어떻게 변하는지 보여주는 실시간 calculator입니다.

### 1.5 Loss-Head Logit Spike

모든 forward pass 끝에서 최종 linear layer는 cross-entropy를 compute하기 전에 hidden states를 vocabulary logits로 투영합니다. 이 과정에서 shape이 `[B·T, V]`인 tensor가 fp32로 materialize된다.

```
logit bytes = B × T × V × 4 B
```

일반적인 설정의 경우(seq = 16,384, vocab = 32,000, CE 안정성을 위해 FP32로 저장되는 BF16):

```
16,384 × 32,000 × 2 = 1.05 GB   (bf16)
16,384 × 32,000 × 4 = 2.10 GB   (fp32 CE inputs)
```

[[liger-fused-ce]]는 이것을 "vocabulary가 큰 model의 표준 training step에서 가장 큰 transient memory event"라고 부릅니다. 이는 일시적입니다(CE reduction 이후 allocation되고 즉시 해제됩니다). 그러나 memory timeline에서 단일 최대 peak이며, static memory를 올바르게 allocation한 *후에도* OOM을 일으키는 가장 흔한 trigger입니다.

완화([[ch-02]]에서 다룸): Liger의 fused chunked cross-entropy kernel은 전체 `B·T×V` tensor를 결코 구체화하지 않습니다. CUDA에서 2,048 이하의 청크로 토큰을 처리하며 sequence length에 관계없이 `2,048 × 32,000 × 2 = 131 MB`에서 spike를 제한합니다. kernel은 수치적으로 정확합니다.

### 1.6 Overhead — 단순한 수학적 계산에는 포함되지 않는 몇 GiB

Parameter가 하나도 없는 model조차 GPU memory를 사용한다.

- **PyTorch CUDA init:** "PyTorch가 처음으로 CUDA를 사용하는 경우 model이 로드되기 전에 최대 0.5–2GB의 GPU memory([[ml-engineering-memory]])를 사용할 수 있습니다."
- **NCCL communication buffers:** 각 collective(all-reduce, all-gather)에는 staging buffers가 필요합니다. 대규모 cluster training에서는 1~4GB를 소비할 수 있습니다.
- **ZeRO all-gather transient:** ZeRO-3 / FSDP는 여러 rank에 걸쳐 parameter를 shard하고 forward pass 중 layer별로 이를 all-gather합니다. gather된 unsharded parameter는 해당 layer를 compute하는 동안 GPU memory를 차지합니다. 이는 한 layer의 전체 parameter tensor와 같은 크기일 수 있는 transient memory입니다.
- **Kernel workspace memory:** cuBLAS 및 cuDNN는 kernel별 작업 공간을 allocation합니다. batch=1의 대형 GEMM tiles는 100~500MB를 allocation할 수 있습니다.

총 overhead 예산은 일반적으로 80GB GPU에서 **2~8GB**이며 이는 용량의 2.5~10%입니다. 이는 parameter 수 formula에 표시되지 않으며 "종이에 여유 공간이 5GB 부족" 오류 클래스를 담당합니다.

[[ultrascale-playbook]]는 관련 현상에 대해 언급합니다. "첫 번째 training step는 후속 step와 다른 memory 패턴을 보여줍니다. optimizer states는 1step 후에만 구체화됩니다. OOM는 1step가 성공하더라도 2step에 나타날 수 있습니다." 이것은 Adam states materialization spike입니다. Adam은 `.step()`가 처음 호출될 때만 fp32 tensor를 allocation합니다.

---

## 2. Rule of 16: Activations 이전의 Static States

항목 1~3(working weights + gradients + Adam states)은 forward pass가 실행되기 전에 allocation됩니다. 요약하면 다음과 같습니다.

```
2 B (weights, bf16)
+ 2 B (gradients, bf16)
+ 12 B (Adam states, fp32 master + m + v)
= 16 B/param
```

**Rule of 16**입니다. 단일 activations가 저장되거나 단일 토큰이 처리되기 전에 완전 precision AdamW를 사용하여 parameter를 training하는 데 *일부* 장치에 필요한 최소 memory입니다.

[[ml-engineering-memory]]은 약간 다른 accounting으로 같은 floor를 도출한다. "6 B/param(2 B bf16 working + 4 B fp32 master) + 4 B gradients + 8 B AdamW optimizer = 18 B/param." 16과 18의 차이는 gradient를 bf16(2 B)으로 계산하는지 fp32(4 B)로 계산하는지다. Numerical stability를 위해 optimizer step 전에 gradient를 fp32로 accumulate하는 현대 framework는 18 B이고, bf16-gradient training은 16 B다. 두 숫자 모두 널리 인용됩니다. 이 장에서는 16B를 기준으로 사용하지만 [[ml-engineering-memory]]를 직접 인용할 때는 18B를 사용합니다.

**27B model의 계산 예:**

| Component | Formula | 27B result |
|---|---|---|
| Working weights (bf16) | 2×27B | 54GB |
| Gradients(bf16, 전체 FT) | 2×27B | 54GB |
| Adam optimizer states (fp32) | 12×27B | 324GB |
| **정적 합계(Rule of 16)** | **16 × 27B** | **432GB** |
| Activations (다양함) | formula §1.4 | + 다양함 |
| Loss-head spike | B·T·V | + 다양함 |
| Overhead | fixed | + 2–8 GB |

432GB states만으로도 5×A100-80GB의 용량을 초과합니다. 이것이 27B full fine-tune에 최소 6~8개의 GPU, 일반적으로 16개 이상의 GPU에 걸쳐 distributed training가 *필요*하는 이유입니다.

Rule of 16은 빠른 filter다. `16 × N_params > GPU_memory × GPU_count`이면 distributed strategy로 states를 shard하거나(ZeRO, FSDP), trainable parameter count를 줄이거나(LoRA), optimizer states를 압축해야 한다(FP8, Adafactor, 8-bit Adam).

---

## 3. 원장 수준의 Full Fine-Tune 대 LoRA

Full fine-tuning과 LoRA의 ledger 차이는 단순한 weight count에 그치지 않고 여섯 항목 중 세 항목으로 연쇄적으로 이어진다.

| 아이템 | Full fine-tune | LoRA(rank 16, Q/V 전용) |
|---|---|---|
| Working weights | 2 B × N_total | 2 B × N_total(memory에 전체 model 상주) |
| Gradients | 2 B × N_total | 27B 규모에서 2 B × N_LoRA ≈ 0 |
| Adam states | 12 B × N_total | 27B 규모에서 12 B × N_LoRA ≈ 0 |
| Activations | 동일 | 동일(동일한 forward pass) |
| Logit spike | 동일 | 동일 |
| Overhead | 동일 | 동일 |

중요한 관찰: **LoRA는 여전히 전체 model을 작업 weights memory에 로드합니다.** 27B LoRA 실행에는 frozen된 model weights에만 54GB가 필요합니다. LoRA가 제거하는 것은 gradient 및 optimizer states 부담입니다. dynamic state의 54 + 54 + 324 = 432GB 대신 54GB(frozen weights) + ~200MB(0.1~0.5% training 가능한 parameter에 대해 어댑터 gradients + Adam states)가 필요합니다.

이 때문에 LoRA는 다른 방식으로는 감당하기 어려운 model도 single-GPU에서 fine-tuning할 수 있게 한다. Frozen weights가 ceiling이 아니라 floor다(LoRA memory의 최소치는 base model을 load하는 데 필요한 memory라는 뜻). Quantization(QLoRA, 4-bit NF4)을 적용하면 54 GB인 frozen-weight floor가 더 낮아져 27B LoRA를 24 GB consumer GPU 두 개에서 실행할 수 있다.

---

## 4. Per-GPU 대 총 회계

Rule of 16은 *전체* states 크기를 제공합니다. Distributed training은 선택한 strategy에 따라 이 합계를 여러 GPU에 나눕니다. 주요 차이점:

**Replicated(바닐라 DDP, ZeRO-0):** 모든 GPU는 전체 16B × N states를 보유합니다. cluster 전체의 총계 = 16B × N × GPU_count. 낭비적이지만 간단합니다.

**Partitioned(ZeRO-1/2/3, FSDP):** states를 shard하여 각 GPU가 이상적으로 16 B × N / GPU_count의 states와 forward 중 all-gather transient를 보유한다. 전체 states는 16B × N으로 유지됩니다. per-GPU states는 비례적으로 떨어집니다.

**핵심 함의:** "이 model이 GPU에 들어가는가?"를 판단하려면 *어떤 component*가 *어떤 GPU*에 놓이는지 알아야 한다. GPU 8개에서 ZeRO-3/FSDP를 사용하면 432 GB의 static states가 GPU당 432/8 = 54 GB로 줄어 27B full fine-tune의 static states가 80 GB H100에 들어간다. 이는 activation과 overhead를 포함하기 전의 값이다. 조금이라도 유의미한 sequence length의 activation을 추가하면 다시 capacity를 초과할 수 있으므로 gradient checkpointing 또는 sequence parallelism이 필요하다.

[[ultrascale-playbook]]는 이를 구체적으로 설명합니다. "7B(!)에 도달하자마자 weights와 최적화 요구 사항이 이미 크게 증가하기 시작하여 일반적인 GPU memory의 크기(예: H100 GPU의 경우 80GB)를 초과합니다." 최대 precision(7 × 16 = 112GB)의 7B 임계값은 단일 H100-80GB라도 ZeRO/FSDP 샤딩 없이는 7B full fine-tune의 static states를 보유할 수 없음을 확인합니다.

activations 및 로짓 spike 예산은 샤딩 전략에 관계없이 *per-GPU*로 유지됩니다. 각 GPU는 자체 마이크로 batch를 처리하므로 activations 및 logit spike는 global batch가 아닌 per-GPU batch size 및 sequence length로 확장됩니다. 이는 ZeRO가 정적 states 문제를 완전히 해결한 후에도 activations 감소 전략(gradient checkpointing, selective recomputation)이 여전히 필요하다는 것을 의미합니다.

---

## 문헌의 핵심 통찰력

**1. static floor는 16-18B/param이며 이는 시작점일 뿐입니다**([[transformer-math-101]], [[ml-engineering-memory]]). 대부분의 엔지니어링 직관은 model weights(2B/parameter)에 초점을 맞추고 Adam이 optimizer 예산만 3배로 늘린다는 사실(12B/parameter)을 망각하여 training memory를 과소평가합니다. weights 전용과 전체 Adam-FT 사이의 8× 비율은 내면화할 숫자입니다.

**2. Activation은 memory budget을 무너뜨리는 변수다**(sequence length 증가로 예상 memory를 초과하게 만드는 항목)([[ultrascale-playbook]], [[transformer-math-101]]). Model과 optimizer가 정해지면 static floor는 일정하지만 activation은 sequence length에 대해 quadratic하게 scaling한다. seq=32,768에서 Llama-3-8B의 activation은 checkpointing이 없으면 240 GB로, static states인 8 × 16 = 128 GB보다 훨씬 크다. Long-sequence training은 optimizer-state 문제가 아니라 activation 문제다.

**3. logit spike는 별도의 보이지 않는 피크**([[liger-fused-ce]])입니다. 이는 16B/param formula에는 나타나지 않고 activations 추정에도 나타나지 않지만 높은 vocabulary model에 대한 표준 training step에서 가장 큰 단일 tensor allocation입니다. `B·T·V` 바이트를 포함하지 않는 memory 예산은 실제 최고치를 놓친 것입니다.

**4. LoRA의 memory 절약은 항목 1**이 아닌 항목 2와 3에 전적으로 적용됩니다([[ml-engineering-memory]], [[transformer-math-101]]에서 파생됨). frozen weights는 GPU memory에 전체 크기로 유지됩니다. LoRA는 N_total − N_LoRA개의 frozen parameter에 대한 gradient 및 optimizer-state allocation을 없앤다. 여기에는 중요한 결과가 있습니다. LoRA의 memory footprint는 LoRA rank가 아닌 기본 model의 inference footprint에 의해 아래에서 경계를 이룹니다.

---

## 주요 시사점

- 모든 training step에는 여섯 가지 명시적 resident가 allocation됩니다. Rule of 16(2 + 2 + 12 = 16 B/param)은 full fine-tune이 memory에 들어가는지를 빠르게 판별하는 기준입니다.
- 27B full fine-tune에는 activations, logit spike 또는 overhead 이전에 54 + 54 + 324 = 432GB의 static states가 필요합니다. 이는 distributed sharding를 요구합니다.
- Activation은 `5·a·s²/h` attention 항을 통해 sequence length에 대해 quadratic하게 scaling한다. Long context에서는 activation memory가 static floor를 지배한다.
- LoRA는 frozen된 parameter에 대해 항목 2와 3을 0으로 설정하지만 항목 1을 줄일 수는 없습니다. inference-memory floor는 남아있습니다.
- logit spike(`B·T·V × dtype`)는 16B/param formula으로 포착되지 않는 일시적인 피크입니다. 이는 정적 예산 책정이 올바르게 표시된 후 OOM의 가장 일반적인 소스입니다.
- overhead(CUDA 초기화, NCCL 버퍼, ZeRO 과도)는 model 크기와 관계없이 2~8GB를 소비합니다. 항상 사용 가능한 GPU 용량에서 이를 빼십시오.
- Adam states는 model load 시점이 아니라 첫 번째 optimizer step의 *끝*에서 materialize된다. 따라서 step 1이 성공하고 step 2에서 OOM이 발생할 수 있다.

---

## 참고자료

- Quentin Anthony et al. (EleutherAI), "Transformer Math 101," 2023. https://blog.eleuther.ai/transformer-math/
- Guilherme Penedo et al. (HuggingFace / nanotron team), "The Ultra-Scale Playbook: Training LLMs on GPU Clusters," 2025. https://nanotron-ultrascale-playbook.static.hf.space/
- Stas Bekman, "Machine Learning Engineering Open Book," stas00/ml-engineering, ongoing. https://github.com/stas00/ml-engineering
- Tianqi Chen et al., "Training Deep Nets with Sublinear Memory Cost," arXiv:1604.06174, 2016. https://arxiv.org/abs/1604.06174 (gradient checkpointing, [[ch-03]])
- Austin Liu et al., "Liger Kernel: Efficient Triton Kernels for LLM Training," arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel (logit spike, [[ch-02]])
