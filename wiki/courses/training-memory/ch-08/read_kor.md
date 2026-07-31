<!-- chapter: ch-08
     track: scaling
     title: Memory Formulas, the Calculator, and the OOM Debugging Loop
     deps: [[ch-01]], [[ch-02]], [[ch-03]], [[ch-07]]
     sources: [[memory-calculator-notes]], [[training-oom-failure-modes]]
-->

# 8장 — Memory 공식, Calculator, 그리고 OOM Debugging Loop

> **핵심 통찰.** Training 중 peak GPU memory에는 서로 더해지는 다섯 가지 구성 요소, 즉 weights (W), gradients (G), optimizer states (O), activations (A), logit-buffer spike (L)가 있으며, 이들이 모두 동시에 peak에 도달하는 것은 아니다. ZeRO-3는 W+G+O를 N (world size)으로 나눈다. activations와 logit spike는 *N과 무관하며* 별도로 예산을 잡아야 한다. 모든 OOM에는 forward, backward, optimizer-step 중 하나인 정확한 phase label이 있으며, traceback은 방금 어떤 구성 요소가 한도를 넘었는지 알려 준다.

> **지침.** 어떤 training run이든 시작하기 전에 `M = 16Ψ/N + A_per_layer × L_layers + L_spike + overhead`를 계산하고, 2 GiB 이상의 safety margin을 둔 GPU capacity와 비교하라. Run에서 OOM이 발생하면 CUDA error message를 읽어 세 수치(requested, free, reserved)를 추출하고 traceback에서 phase를 식별한 다음, 다음 순서로 필요한 최소한의 lever를 적용하라. nodes (full-FT model states용), TP (activations 및 MoE용), fused-CE (logit spike용), batch/seq 축소, gradient checkpointing, 마지막으로 offload.

---

## 1. Full-FT Per-GPU 공식 구성하기 (ZeRO-3)

ZeRO-3를 사용하는 full fine-tuning(또는 이름만 다른 ZeRO-3인 FSDP FULL_SHARD)은 세 가지 model-state 항을 모두 N개의 DP ranks에 분산한다. Per-GPU 원장은 다음과 같다.

| 구성 요소 | Per-GPU 공식 | Dtype | 설명 |
|-----------|----------------|-------|-------|
| Weights | `2Ψ / N` | bf16 | 일시적으로 AllGather되며, 유휴 상태에서는 shard를 보유 |
| Gradients | `2Ψ / N` | bf16 | local shard로 ReduceScatter됨 |
| Adam optimizer | `12Ψ / N` | fp32 | fp32 master `4Ψ` + momentum `4Ψ` + variance `4Ψ` |
| **Model-state 소계** | **`16Ψ / N`** | | shard된 Rule of 16 |
| AllGather buffer (FSDP peak) | `≈ 2 × P_unit` | bf16 | 가장 큰 FSDP unit의 forward/backward 중 일시적으로 존재 |
| Activations (checkpointing 없음) | `(sbh/t)(34 + 5as/h) × L` | mixed | s=seq, b=batch, h=hidden, a=heads, t=TP; layer당 계산 후 L layers를 곱함 |
| Activations (checkpointing 사용) | `~2sbh × L` bytes | bf16 | layer 경계마다 tensor 하나 — 실무자를 위한 단순화된 추정치 |
| **Logit-buffer spike** | `vocab × s × b × 2` bytes | bf16 | forward-backward 경계에서 구체화되며 ZeRO, TP, PP로 줄어들지 않음 |
| Overhead | `~1–2 GiB` | — | CUDA context, NCCL, allocator fragmentation |

16Ψ 하한을 결정하는 **baseline mixed-precision Adam 항등식** ([[zero-memory-optimization]]):

```
2Ψ (fp16/bf16 weights)
+ 2Ψ (fp16/bf16 gradients)
+ 4Ψ (fp32 master params)
+ 4Ψ (fp32 momentum)
+ 4Ψ (fp32 variance)
= 16Ψ bytes per parameter, unreduced
```

ZeRO-3는 이 다섯 항을 각각 N으로 나누어 `16Ψ/N`을 만든다. 이 절감에 따르는 communication 비용은 **DDP 대비 1.5배**이다. forward 전 all-gather (Ψ), backward 중 reduce-scatter (Ψ), backward 전 두 번째 all-gather (Ψ)를 합쳐 총 3Ψ이며, DDP의 2Ψ all-reduce와 대비된다 ([[pytorch-fsdp]]).

**N=64 DP ranks에서 7.5B model의 구체적인 수치** ([[zero-memory-optimization]]):

| Strategy | Per-GPU model states |
|----------|---------------------|
| DDP (sharding 없음) | 120 GB |
| ZeRO-1 (optimizer만 sharding) | ~31 GB |
| ZeRO-2 (optimizer + grad sharding) | ~17 GB |
| ZeRO-3 (모두 sharding) | ~1.9 GB |

"ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware" (ZeRO는 오늘날의 hardware를 사용하여 1 Trillion parameters를 넘어 확장할 잠재력이 있다) (Rajbhandari et al., SC 2020)라는 말의 구체적인 의미가 바로 이것이다. 16Ψ 하한이 16Ψ/N이 되므로, N을 확장할수록 model-state memory는 거의 비용이 들지 않는 자원이 된다.

**계산 예시: 7B model, N=8, s=2048, b=1, h=4096, a=32, t=1, vocab=32K** ([[memory-calculator-notes]]):
- Model states: `16 × 7×10⁹ / 8 ≈ 14 GB`
- Activations (checkpointing 없음, 32 layers): `(2048 × 1 × 4096 / 1) × (34 + 5 × 32 × 2048 / 4096) × 32 layers ≈ 16 GB`
- Logit spike: `32000 × 2048 × 1 × 2 ≈ 0.13 GB`
- AllGather buffer: `≈ 2 × (7B/32 layers × 2) ≈ 0.44 GB`
- **추정 합계: ~31 GB** — 2× A100-40GB에는 여유를 두고 들어가지만, checkpointing 없이 GPU 하나에서는 빠듯함

---

## 2. Logit-Buffer Spike와 이것이 World-Invariant인 이유

Logit spike는 **forward-backward 경계**에서 이루어지는 단일 allocation이다. 모든 forward pass는 loss를 계산하기 전에 bf16으로 `[batch, seq, vocab_size]` shape의 `output_projection(hidden) → logits`를 계산해야 한다. 그 크기는 다음과 같다.

```
vocab_size × seq_len × batch_size × 2 bytes
```

이 allocation은 다음과 같은 특성이 있다.
- **ZeRO로 줄어들지 않는다** — ZeRO는 parameter memory를 shard하며, activations나 computation buffers를 shard하지 않는다.
- **PP로 줄어들지 않는다** — output layer를 소유한 모든 pipeline stage가 전체 크기를 allocation한다.
- **DP/FSDP로 줄어들지 않는다** — 각 DP replica가 자신의 micro-batch에 대해 자체 logits를 계산한다.

적당한 vocab (32K)과 seq (2048)에서는 0.13 GB에 불과해 무시할 만하다. 하지만 vocab (128K+)이 크거나 batch/seq가 크면 **지배적인 OOM 유발 요인**이 된다. seq=16,384, vocab=32K이면 `16384 × 32000 × 2 = 1.05 GB`이다 ([[liger-fused-ce]]). seq=16K, vocab=128K이면 `16384 × 131072 × 2 ≈ 4.3 GB`로, step마다 정확히 한 지점에서 발생하는 단일 allocation이다.

**식별 signature**: `lm_head`, `output_projection`, `_get_per_token_logps_and_entropies`를 가리키는 OOM traceback은 logit spike를 뜻한다 ([[training-oom-failure-modes]]). 해결책은 nodes를 늘리거나 ZeRO stage를 높이는 것이 **아니다**. Token dimension을 따라 chunk로 나누어 가장 큰 임시 allocation을 전체 곱이 아닌 `chunk_size × vocab_size × 2`로 만드는 fused cross-entropy (Liger [[liger-fused-ce]])를 사용하거나 seq/batch를 줄여야 한다.

> **Interactive companion:** [figures/memory-calculator.html](figures/memory-calculator.html) — model size, world config (N, TP, PP), sequence/batch를 입력하면 calculator가 다섯 가지 원장 구성 요소를 나란히 출력하고 지배적인 항을 강조하며 각 lever가 결과를 얼마나 바꾸는지 보여 준다(‘moves the needle’, 즉 측정값에 의미 있는 변화를 만든다).

---

## 3. LoRA Memory: Node-Invariant Activations

LoRA는 base model을 freeze하고 rank r의 low-rank adapter matrices만 training한다. Optimizer-state와 gradient 절감 효과는 분명하지만, activation 예산은 동일한 sequence length와 batch size의 full fine-tuning과 **동일**하며, 결정적으로 **world-invariant**이다.

| 구성 요소 | Full-FT (ZeRO-3) | LoRA |
|-----------|-----------------|------|
| Frozen/working weights | `2Ψ / N` | `2Ψ` (모든 replicas가 전체 base를 보유) |
| Adapter weights | — | `≈ 2rΨ_target / d` (매우 작음) |
| Gradients | `2Ψ / N` | adapter만: `≈ 2r × Ψ_target / d` |
| Optimizer states | `12Ψ / N` | adapter만: `≈ 12r × Ψ_target / d` |
| **Activations** | `(sbh/t)(34 + 5as/h) × L` | **동일** |
| Logit spike | world-invariant | world-invariant |

Activations가 변하지 않는 이유는 frozen base model도 여전히 전체 forward pass를 수행하고, 그 forward pass가 adapter backward를 위해 activations를 cache해야 하기 때문이다. Adapter는 몇몇 weight matrices에만 삽입될 뿐, forward 중 cache되는 tensors의 수를 줄이지 않는다.

**World-invariant 특성** ([[memory-calculator-notes]]): DP ranks를 더 추가하면(nodes를 늘리고 world size를 키우면) `16Ψ` model-state 항을 N으로 나누므로 full-FT에 도움이 된다. LoRA에서는 `16Ψ` 항이 이미 매우 작다(adapter states뿐이므로 무시할 만하다). 지배적인 예산은 activations이며, 이는 **N으로 나뉘지 않는다**. 각 replica는 자체 tokens를 처리하므로 모두 동일한 activation peak를 보유한다. 따라서 LoRA OOM을 해결하려고 nodes를 추가해도 실제로 한도를 넘은 구성 요소에는 아무런 유용한 효과가 없다. LoRA activation OOM에 맞는 lever는 다음과 같다.
1. Batch 또는 sequence length 축소
2. Gradient/activation checkpointing 활성화
3. TP degree t 증가(activation 공식을 t로 나눔)

이것이 full-FT와 LoRA의 OOM 동작을 가르는 핵심 차이이다. **Full-FT OOM은 nodes를 늘리면 확장을 통해 해소되지만, LoRA OOM은 그렇지 않다.**

---

## 4. Activation Memory 심층 분석

### 4.1 Megatron activation 공식

Tensor parallelism degree t이고 sequence parallelism이 활성화된 경우 ([[megatron-tp-sp]]):

```
Per-layer activation memory (TP only, no SP):
  sbh(10 + 24/t + 5as/ht)  bytes

Per-layer activation memory (TP + SP):
  (sbh/t)(34 + 5as/h)  bytes
```

TP-only 공식의 `10sbh` 항은 layer-norm과 dropout activations이며, TP만 사용할 때는 모든 t ranks에 **복제된** 상태로 남는다. Sequence parallelism은 이를 sequence dimension을 따라 shard하여 `10sbh`를 `10sbh/t`로 바꾼다. 이는 layer 전체에서 실제로 t배 줄어드는 것이다. 최종적으로 SP+TP는 추가 communication bandwidth 비용 없이 TP만 사용할 때보다 총 activation memory를 실질적으로 t배 줄인다(all-reduce가 동일한 volume의 AllGather + ReduceScatter로 대체된다).

### 4.2 Activation checkpointing: 실무자를 위한 공식

Full activation checkpointing (Chen 2016 ([[gradient-checkpointing-chen]]))은 layer 경계마다 activation tensor 하나만 저장하고 backward pass 중 layer 내부의 모든 activations를 다시 계산한다. Memory는 O(L × sbh × factor)에서 대략 다음 수준으로 감소한다.

```
~2sbh × L  bytes  (simplified practitioner estimate with checkpointing)
```

Compute overhead는 2배가 아니라 **약 33%**이다(mini-batch마다 forward pass 한 번을 추가). Chen 2016의 `sqrt(n)` scheme은 n이 layers 수일 때 더 정확하게 적용된다. sqrt(L) layers마다 checkpoint하고 각 segment가 내부에서 다시 계산한다. PyTorch의 `torch.utils.checkpoint.checkpoint()`가 이를 구현한다.

### 4.3 Selective recomputation (Korthikanti 2022)

Full recomputation에는 FLOPs가 30–40% 더 든다. [[selective-recompute-korthikanti]]의 통찰은 비대칭적이다. 모든 activations의 성격이 같지는 않다.

- **Attention score/softmax matrices** (`s × s` 또는 `s²` 항): seq에 대해 제곱으로 증가하므로 크지만, 다시 계산하는 비용은 저렴하다(matmul + softmax이며, 어차피 FlashAttention도 backward에서 이를 다시 계산한다).
- **MLP activations**: 더 작지만 다시 계산하는 비용이 크다(대규모 FFN matmuls).

Selective recomputation은 attention score 영역, 즉 `5as²b/ht` 항만 버리고 MLP, LayerNorm, projection outputs는 유지한다. 그 결과는 다음과 같다.

- **activation memory 약 5배 감소**
- **추가 FLOPs 2% 미만** (full recompute의 30–40%와 대비)
- 2240개의 A100에서 530B scale일 때 MFU가 42.1% (full recompute)에서 54.2% (selective recompute + SP)로 향상됨 — throughput 29% 개선

Ultra-Scale Playbook ([[ultrascale-playbook]])의 `2.7% compute cost로 activation memory 70% 감소` 수치는 GPT-3 175B에 이 selective scheme을 적용한 결과를 가리킨다.

---

## 5. Phase 경계에서의 Allocator Peak

가장 흔하면서도 뜻밖인 OOM 상황 중 하나는 **step 1은 성공하지만 step 2에서 OOM이 발생하는 것**이다. 이는 PyTorch CUDA caching allocator에서 memory 사용량이 시간에 따른 구조를 갖기 때문이다.

| Phase | 구체화되는 항목 |
|-------|-------------------|
| Model load | Weights (`2Ψ/N`) |
| Step 1 forward | Activations가 layer별로 증가 |
| Step 1 fwd-bwd 경계 | Logit spike가 나타나고 모든 activations가 동시에 cache됨 |
| Step 1 backward | Gradients가 뒤로 흐르면서 activations가 감소 |
| **Step 1 종료 시점** | **Adam optimizer states가 처음으로 구체화됨** (`12Ψ/N`) |
| Step 2 forward | 이제 weights + 전체 optimizer states + 새로운 activations가 공존해야 함 |

[[ultrascale-playbook]]은 이를 명시적으로 설명한다. "The first training step shows different memory patterns than subsequent steps — optimizer states materialize only after step 1; OOM can appear on step 2 even if step 1 succeeds." (첫 training step은 이후 steps와 다른 memory pattern을 보인다. Optimizer states는 step 1 이후에야 구체화되므로 step 1이 성공해도 step 2에서 OOM이 발생할 수 있다.) 안전한 실무 방식은 step 1이 아니라 **항상 step 2의 peak를 profile하는 것**이다.

두 번째 원인은 **allocator fragmentation**이다. PyTorch의 caching allocator는 해제된 blocks를 계속 보유하므로 `free_memory > requested`인 경우에도 새로운 allocation을 충족하지 못할 수 있다. 진단 기준은 `reserved_memory - allocated_memory`가 큰 것이다. 해결책은 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch ≥ 2.2)이다 ([[training-oom-failure-modes]]).

FSDP per-step lifecycle을 보면 이를 구체적으로 이해할 수 있다 ([[pytorch-fsdp]]). 가장 큰 FSDP unit을 지나는 각 forward/backward pass마다 `≈ 2 × P_unit` bytes의 **AllGather buffer**가 일시적으로 allocation되고, 해당 unit의 compute가 진행되는 동안 유지된 뒤 해제된다. 이 buffer는 `16Ψ/N` steady state 위로 솟는 "peak excursion"(일시적인 peak 상승분)이므로 명시적으로 예산에 포함해야 한다.

```
Peak FSDP memory ≈ 16Ψ/N + 2 × P_largest_unit + activations + logit_spike + overhead
```

---

## 6. OOM Debugging Loop

모든 CUDA OOM은 다음 형식의 message를 생성한다 ([[training-oom-failure-modes]]).

```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
(GPU 0; Y GiB total capacity; Z GiB already allocated;
 W MiB free; PyTorch memory managed V GiB...)
```

세 수치를 추출하라. X (requested), Z (already allocated), Y (total)이다. 필요한 최소 memory 감소량은 `Z + X − Y`이다. Traceback은 **phase**를 특정한다.

| Traceback 위치 | Phase | 지배적인 소비 항목 | 올바른 lever |
|-------------------|-------|-------------------|---------------|
| `lm_head` / `output_projection` | Fwd-bwd 경계 | Logit spike | Fused-CE (Liger), vocab 또는 seq 축소 |
| `loss.backward()` 또는 layer backward | Backward pass | 모든 activations가 cache됨 | Checkpointing, seq/batch 축소 |
| `optimizer.step()` | Optimizer phase | Adam states가 모두 활성 상태 | ZeRO-3 / FSDP, LoRA |
| Forward 내부에서 증가 | Forward pass | Activations 증가 | seq/batch 축소, TP, checkpointing |
| Model load | Load | Weights + states | ZeRO-3가 필수 |

### 6단계 debugging loop

```
1. Run 전에 추정(ESTIMATE):
   M = 16Ψ/N + activations(s, b, t) + logit_spike(vocab, s, b) + overhead
   M > GPU_capacity − 2 GiB safety margin이면 → 실행하지 말고 먼저 조정한다.

2. SMOKE-RUN (batch=1, seq=128, 5 steps):
   여기서 OOM이면 → model-state memory (weights + optimizer)가 한도를 넘는다.
   ZeRO-3 (nodes 추가) 또는 LoRA를 적용한다.
   통과하면 → model-state 예산은 문제없으므로 계속 진행한다.

3. OOM message 읽기(READ):
   X (requested), Z (allocated), Y (total)를 추출한다.
   Traceback을 읽어 → phase (fwd / bwd-seam / bwd / opt-step)를 식별한다.

4. Phase label에서 지배적인 구성 요소 식별(IDENTIFY)(위 표 참조).

5. 영향과 비용 순서에 따라 lever 적용(APPLY):
   a. Nodes 추가 (ZeRO-3 N↑) — full-FT model states에만 사용
   b. Tensor parallelism (t↑) — activations와 MoE weights용
   c. Fused-CE — logit spike용, 비용 없음
   d. Batch 또는 sequence 축소 — activations + logit spike용
   e. Gradient/activation checkpointing — activations용, compute +15–40%
   f. LoRA — full-FT에서 전환해도 괜찮은 경우(gradient signal이 달라짐)
   g. Offload (CPU/NVMe) — 최후의 수단: optimizer step이 10–100배 느려짐

6. Calculator에 결과를 다시 입력(FEED back):
   Lever를 적용한 뒤 새로운 config로 M을 다시 계산한다.
   다음 smoke-run에서 측정한 Z를 ground-truth oracle로 사용하여
   공식을 보정한다(공식이 놓친 항목을 찾아낸다).
```

**Offload가 최후의 수단인 이유**: CPU offload는 optimizer states를 RAM(또는 NVMe)으로 옮겨 `12Ψ/N` GPU allocation을 제거한다. 하지만 이제 optimizer step마다 양방향으로 각각 `12Ψ/N` bytes를 PCIe를 통해 전송해야 한다. Ψ=7B, N=8이면 step당 약 21 GB의 PCIe traffic이 약 32 GB/s로 이동하므로 step당 PCIe 전송에 약 0.65초가 걸린다. 이는 GPU 내 optimizer step의 1 ms 미만과 대비된다. 대규모 models에서는 throughput이 100–1000배 악화된다 ([[training-oom-failure-modes]]).

---

## 7. Batch Semantics와 MFU

### 7.1 Batch 계산

각 weight update에서 optimizer가 보는 **effective batch size**는 다음과 같다.

```
global_batch_size = per_device_batch × grad_accum_steps × DP_replicas
```

Gradient accumulation은 optimizer step 전에 `grad_accum_steps`개의 micro-batches에 걸쳐 gradients를 누적한다. Memory에 미치는 영향은 다음과 같다. Per-step *peak* memory는 `per_device_batch`를 사용하므로(한 번에 micro-batch 하나씩), grad accumulation은 peak activation memory를 늘리지 않으면서 effective batch를 키우는 거의 무료에 가까운 방법이다. 하지만 model-state memory를 줄이지는 않는다.

Epoch당 steps:

```
steps_per_epoch = dataset_tokens / (global_batch_size × seq_len)
```

### 7.2 Model FLOPs Utilization

**6PD FLOPs/token 규칙** ([[ultrascale-playbook]], [[transformer-math-101]]):

```
FLOPs per training token ≈ 6 × Ψ
```

(forward matmuls에 2 + backward에 2 + weights를 통한 gradient accumulation에 2이며, 모두 근삿값이다. Activation checkpointing은 recompute에 약 1/3을 추가하므로 checkpointing을 사용하는 경우 약 8× Ψ가 된다.)

MFU:

```
MFU = (achieved_FLOPs_per_second) / (peak_hardware_FLOPs_per_second)
    = (6Ψ × tokens_per_second) / (GPU_TFLOPS × N_GPUs × 10¹²)
```

A100에서 잘 조정된 run의 양호한 MFU는 40–55%이다. 30% 미만이면 communication saturation, activation checkpointing overhead, data-loading bottleneck을 의심해야 한다. 530B에서 selective-recompute + SP 조합은 MFU 54.2%에 도달했다 ([[selective-recompute-korthikanti]]).

---

## 8. Parallelism Levers와 각 Lever가 실제로 해결하는 문제

| Lever | 나누어 줄이는 항목 | 도움이 되지 않는 항목 |
|-------|----------------|---------------|
| DP nodes 추가 (ZeRO-3) | Model states: `16Ψ → 16Ψ/N` | Activations, logit spike |
| Tensor Parallelism (TP=t) | Activations (SP 사용 시 완전한 t배), weight matmuls | Model states 자체 |
| Pipeline Parallelism (PP=p) | Weight memory (각 rank가 `Ψ/p` layers를 보유) | Per-layer activations; bubble 추가 |
| Gradient checkpointing | Activations: `A_layers → ~2sbh × L` | Model states, logit spike |
| Fused cross-entropy | Logit spike: materialization 제거 | 그 밖의 모든 항목 |
| LoRA | Frozen params의 gradients + optimizer | Activations (world-invariant!) |
| Seq length 축소 | Activations (제곱 관계), logit spike | Model states |
| Offload (CPU/NVMe) | Model states (극심한 throughput 비용을 감수) | Activations, logit spike |

**Pipeline parallelism bubble**: GPipe와 1F1B의 bubble fraction은 `(p-1)/m`으로 동일하다(여기서 m = microbatches). 하지만 동시에 처리 중인 activation memory의 상한은 GPipe가 m microbatches인 데 비해 1F1B는 p microbatches이다 ([[pipeline-parallelism-1f1b]]). GPipe보다 1F1B를 선호할 유일한 이유는 이 activation상의 이점이다. Bubble time은 동일하다.

**MoE용 Expert parallelism (EP)**: 각 GPU는 `E/EP_size` experts를 보유한다. Routing에는 MoE layer마다 두 번의 all-to-all collectives가 필요하다(dispatch + combine). All-to-all buffer size는 rank당 `tokens × d_model × 2` bytes이며 sequence length와 batch에 따라 증가한다. 이는 정적인 expert weight allocation에 더하여 예산에 포함해야 하는 일시적인 peak이다 ([[deepspeed-moe-ep]]).

---

## 문헌에서 얻은 핵심 통찰

1. **하나가 아닌 다섯 가지 구성 요소** ([[memory-calculator-notes]]): Peak GPU memory는 step 중 서로 다른 시점에 peak에 도달하는 다섯 항(W, G, O, A, L)의 합이다. Optimizer states (O)는 step 1 종료 시점에야 구체화된다. Step 1을 완료한 smoke-run은 해당 run이 계속 유지될 수 있다는 증거가 아니다. 항상 step 2까지 검증하라.

2. **Logit spike는 구조적으로 ZeRO 밖에 있다** ([[training-oom-failure-modes]], [[liger-fused-ce]]): `vocab × seq × batch × 2` bytes가 forward-backward 경계에서 발생하며 어떤 data-parallel 또는 model-state strategy도 이를 다루지 못한다. Vocab이 128K 이상이면 model states보다 먼저 제한 요인이 된다. Fused cross-entropy (Liger kernel)가 이 문제를 정확히 겨냥한 해결책이다. Projection + loss를 token 단위 loop의 chunks로 나누고 전체 tensor를 구체화하지 않아 peak를 `B·T·V × 2`에서 `chunk_size × V × 2`로 줄인다.

3. **LoRA activation 불변성** ([[memory-calculator-notes]]): "LoRA uses less memory"(LoRA는 memory를 덜 사용한다)라는 통념은 optimizer states에는 맞지만 activations에는 틀리다. Adapter backward가 frozen forward pass를 통해 gradients를 흘려보내야 하므로 frozen base model은 full fine-tuning과 동일한 activations를 cache한다. LoRA OOM에 DP replicas를 추가하는 것은 잘못된 lever이며, TP와 checkpointing이 올바른 선택이다.

4. **Selective recompute는 Pareto-optimal이다** ([[selective-recompute-korthikanti]]): Activation memory를 5배 줄이기 위해 full activation checkpointing의 30–40% compute overhead를 감수할 필요는 없다. Attention score matrices(s에 대해 제곱 관계이고 recompute가 저렴함)만 버리고 MLP activations를 유지하면 compute cost 2% 미만으로 동일하게 memory를 5배 절감한다. 절감된 byte당 compute 비율이 full recompute보다 15–20배 우수하다.

---

## 핵심 요점

- **Full-FT ZeRO-3의 per-GPU 공식은 `16Ψ/N + A + L_spike + overhead`이다**. Model states는 N에 따라 작아지지만 activations와 logit spike는 그렇지 않다.
- **LoRA memory는 activation 수준에서 node-invariant이다**. Nodes를 늘리면 full-FT OOM은 해결되지만 LoRA OOM은 해결되지 않는다. TP 또는 seq/batch 축소만 도움이 된다.
- **Step-2 함정**: optimizer states는 step 1 종료 시점에 구체화된다. 항상 step 2까지 profile해야 하며, 그렇지 않으면 smoke-run이 false negative(문제가 있는데 없다고 판단하는 결과)를 낸다.
- **OOM traceback이 첫 번째 diagnostic tool이다**: `lm_head` → logit spike, `loss.backward()` → activations, `optimizer.step()` → model states.
- **Lever 순서**: nodes → TP → fused-CE → batch/seq → checkpointing → LoRA → offload (마지막, step-time 비용 10–100배).
- **MFU = 6Ψ × tok/s / (GPU_TFLOPS × N × 10¹²)**. Selective recompute + SP는 530B scale에서 MFU를 42%에서 54%로 끌어올린다.
- **Offload는 최후의 수단이다** — PCIe bandwidth (~32 GB/s)는 HBM (~2 TB/s)보다 100–1000배 느리므로, 다른 대안이 있는 model에서 CPU offload를 사용하면 throughput에 재앙적인 결과를 초래한다.

---

## 참고 문헌

- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC 2020. https://arxiv.org/abs/1910.02054 — [[zero-memory-optimization]]
- Zhao, Y. et al. "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel." VLDB 2023. https://arxiv.org/abs/2304.11277 — [[pytorch-fsdp]]
- Korthikanti, V. et al. "Reducing Activation Recomputation in Large Transformer Models." MLSys 2023. https://arxiv.org/abs/2205.05198 — [[selective-recompute-korthikanti]], [[megatron-tp-sp]]
- Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost." arXiv:1604.06174, 2016. https://arxiv.org/abs/1604.06174 — [[gradient-checkpointing-chen]]
- Liu, A. et al. "Liger Kernel: Efficient Triton Kernels for LLM Training." arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel — [[liger-fused-ce]]
- Huang, Y. et al. "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." NeurIPS 2019. https://arxiv.org/abs/1811.06965 — [[pipeline-parallelism-1f1b]]
- Rajbhandari, S. et al. "DeepSpeed-MoE." ICML 2022. https://proceedings.mlr.press/v162/rajbhandari22a — [[deepspeed-moe-ep]]
- Penedo, G. et al. "The Ultra-Scale Playbook." HuggingFace, 2025. https://nanotron-ultrascale-playbook.static.hf.space — [[ultrascale-playbook]]
- Bekman, S. "ML Engineering Open Book." https://github.com/stas00/ml-engineering — [[ml-engineering-memory]], [[training-oom-failure-modes]]
- 종합한 공식 및 계산 예시: [[memory-calculator-notes]]

---

*다음: [[ch-09]] — Capstone: 27B MoE Memory Budget을 End-to-End로 Modeling하기. 이 장의 공식을 기본 worksheet로 사용한다.*
