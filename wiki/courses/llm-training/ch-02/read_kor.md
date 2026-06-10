<!-- chapter: ch-02
     track: foundations
     title: Numerical Precision and Stability
     sources: [[mixed-precision]], [[adam]], [[gradient-clipping]], [[batch-vs-layer-norm]], [[deepseek-v3]]
     figures: figures/precision-range.html
-->

# 2장 — 수치 정밀도와 안정성

> **핵심 통찰.** 현대 LLM 학습은 multi-precision pipeline이다. 계산에는 bf16, reduction에는 fp32, H100/B100의 matmul tensor core에는 fp8을 쓴다. run이 조용히 divergence하지 않게 지키는 것은 *어떤 op가 어떤 precision에 있어야 하는가에 대한 규칙*이다.
>
> **지침.** 2025년에는 가능한 모든 곳에 bf16을 써라(loss scaler 불필요). norm reduction, softmax, loss, optimizer state는 fp32로 유지하라. fp8은 fp32 master weight와 per-tensor 또는 per-block scaling을 갖춘 matmul path에만 남겨두라.

---

## 이 장이 필요한 이유

Precision bug는 스스로를 알리지 않는다. fp16에서 AdamW의 잘못된 `eps`로 학습한 run은 step 3000까지 괜찮아 보이는 loss curve를 만들다가, 단 한 번의 나쁜 step에서 `inf`를 만들고 training scaler가 친절하게 그 step을 건너뛴다. 그다음 500스텝 동안 계속. LayerNorm reduction이 실수로 fp32가 아니라 bf16에서 일어난 run은 수렴하지만 최종 perplexity가 기대보다 0.5포인트 나쁘고, 이유를 끝내 찾지 못한다. per-tensor scaling 없는 fp8 run은 amax history가 saturation되어 무너지기 전까지 수백 스텝 동안 drift한다.

precision을 올바르게 다루는 비용은 작다. 틀렸을 때의 비용은 compute 80% 지점에서 예산 밖 restart를 하는 것이다. 이 장은 training stack의 각 layer가 무엇을 필요로 하는지, 그리고 왜 그런지 열거한다.

주요 자료는 [[mixed-precision]]이며, 상호작용은 [[adam]]과 [[gradient-clipping]]을 교차 참조한다.

---

## 1. 네 가지 format과 운영상 의미

부동소수점 수의 layout은 `sign | exponent | mantissa`다. 중요한 축은 두 가지다. **range**(exponent bit)와 **precision**(mantissa bit). 나머지는 여기서 나온다.

| Format | Bits | Exp | Mant | Range | Precision | 일반적 사용 |
|---|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1e-38 to 3e38 | 약 7자리 십진수 | master weights; norm reductions; softmax; loss |
| fp16 | 16 | 5 | 10 | ~6e-8 to 65504 | 약 3자리 십진수 | Volta 시대 compute; **loss scaling 필요** |
| bf16 | 16 | **8** | 7 | ~1e-38 to 3e38 | 약 2자리 십진수 | 2025년 기본 compute format; loss scaling 없음 |
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 | coarse | H100+ forward matmul(activation × weight) |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 | coarser | H100+ backward matmul(gradient) |

각 format이 표현 가능한 값이 real line 어디에 놓이는지 나란히 보여주는 visualizer는 `figures/precision-range.html`을 보라.

핵심 tradeoff: **fp16은 bf16보다 precision이 높지만 range가 훨씬 좁다**. LLM gradient에서는 특히 range가 지배적이다. gradient는 여러 자릿수 규모를 가로지르고, fp16의 floor(약 ~6e-8) 아래의 값은 조용히 0으로 underflow한다. bf16은 fp32의 range와 일치한다. 2023년 이후 frontier run이 전환한 이유다.

**bf16 / fp16 결정을 한 문장으로.** Volta(V100) hardware를 지원해야 한다면 fp16 + dynamic loss scaling을 쓰고, 아니면 bf16을 쓰고 scaler code를 지워라.

---

## 2. fp16 recipe와 bf16이 이를 거의 퇴역시키는 이유

[[mixed-precision]]에서 가져온 Micikevicius et al. 2017의 세 부분짜리 fp16 recipe:

1. **fp32 master weights.** Optimizer state와 깨끗한 parameter copy는 fp32에 둔다. forward에서 쓰는 bf16/fp16 weight는 *view*다.
2. **Loss scaling.** `.backward()` 전에 loss에 `S`를 곱해 작은 gradient가 fp16의 representable range 안에 들어오게 한다. optimizer step 전에 gradient를 `S`로 나눈다.
3. **fp32 matmul accumulation.** Tensor core는 fp16 input으로 `C += A @ B`를 계산하되 fp32 accumulator를 쓴다. 이것이 없으면 긴 reduction이 saturation된다.

전체 pipeline을 코드로 쓰면 다음과 같다.

```python
# fp16 + dynamic loss scaler (PyTorch AMP)
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(batch).loss          # forward in fp16
    scaler.scale(loss).backward()         # scaled backward → fp16 grads ×S

    scaler.unscale_(optimizer)            # ← grads ÷ S back to real scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # NOW valid
    scaler.step(optimizer)                # step, or skip if any grad is inf/NaN
    scaler.update()                       # adjust S dynamically
```

Dynamic scaling은 `S = 2^15`에서 시작한다. 어떤 gradient든 inf/NaN이면 `S`를 절반으로 줄이고 step을 skip한다. 약 2000번의 성공 step마다 `S`를 두 배로 늘린다. scaler state는 checkpoint를 빼먹으면 곧바로 training-resume bug가 되는 한 줄짜리 상태다.

**bf16의 단순화.** bf16으로 바꾸면 scaler code가 사라진다.

```python
for batch in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch).loss
    loss.backward()                       # grads live in bf16 range = fp32 range
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

공개된 2023년 이후 모든 frontier run(Llama 3, GPT-NeoX, Mistral, Qwen, DeepSeek V2/V3, OLMo, Tülu 3)은 이 bf16 + fp32-master setup을 쓴다. 오늘 training codebase를 시작하고 V100 compatibility가 필요 없다면 fp16을 아예 지원하지 마라.

---

## 3. 반드시 fp32에 있어야 하는 op들(끝나지 않는 목록)

bf16 compute에서도 **네 종류의 op는 무조건 fp32에 남아야 한다**.

1. **Norm reduction**(LayerNorm / RMSNorm). 4096-dim vector에 대한 `mean(x²)`는 bf16에서 error가 누적되어 normalisation을 bias한다. Framework는 여기서 기본적으로 fp32 reduction을 쓴다. 절대 override하지 마라. [[batch-vs-layer-norm]]을 보라.
2. **Softmax.** Exponential은 range-sensitive하다. attention logit에 대해 bf16으로 계산한 `softmax`는 right tail의 token을 잃는다.
3. **Cross-entropy loss.** 최종 log-softmax + NLL은 numerically stable gradient를 위해 fp32에서 일어나야 한다.
4. **Optimizer state + master weights.** AdamW의 `m_t`, `v_t`, 그리고 fp32 shadow copy. [[adam]]을 보라.

```python
# The universal pattern for a norm under mixed precision
class RMSNorm(nn.Module):
    def forward(self, x):                          # x: bf16 activations
        in_dtype = x.dtype
        x_fp32 = x.float()                         # promote for reduction
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(in_dtype) * self.weight
```

`.float()` cast는 선택 사항이 아니다. 빼면 느리고 조용한 quality regression을 보게 된다. 진단하기 가장 어려운 종류의 버그다.

---

## 4. fp8 — 2024년 이후 frontier path

H100과 B100 tensor core는 bf16 throughput의 2배로 fp8 matmul을 계산한다. 하지만 deployment detail이 flop 수보다 더 중요하다.

**Per-tensor scaling.** 각 matmul 전에 `amax(x)`를 계산하고 scale `s = fmax_E4M3 / amax(x)`를 고른 다음 `x_fp8 = (s · x).to(float8_e4m3)`로 cast한다. matmul은 bf16/fp32로 accumulate하고, product는 나오는 길에 unscale된다. 이것이 "storage format이 아니라 accelerator feature로서의 fp8" 패턴이다.

**E4M3 vs E5M2 split.** [[mixed-precision]]은 forward/backward split을 문서화한다. precision이 range보다 더 중요한 forward activation과 weight에는 E4M3(4 exponent bit), range가 더 중요한 gradient에는 E5M2(5 exponent bit)를 쓴다. NVIDIA의 Transformer Engine library가 이를 자동화한다.

**DeepSeek-V3의 block-wise 변형.** DeepSeek-V3는 per-tensor가 아니라 1×128(weight row)과 128×128(activation tile) block-wise scaling을 사용한다. 각 block은 자체 scale factor를 가진다. 이는 더 많은 scale-factor bookkeeping을 대가로 weight matrix 내부의 heterogeneous distribution을 견딘다. 논문은 [[deepseek-v3]]에 있다.

**Delayed scaling.** 매 call마다 `amax(x)`를 새로 계산하는 것은 비싸다. 대부분의 구현은 *delayed* scaling을 쓴다. step t의 scale은 t-16부터 t-1까지의 amax history에서 계산되어 1스텝 늦게 적용된다. 안정적인 학습에서는 괜찮다. 변동성이 큰 phase(warmup, LR change, RL rollout)에서는 drift할 수 있다. 그래서 많은 frontier fp8 run은 bf16 fallback path를 유지하고 auto-switch한다.

**보편 fp8 규칙.** full fp8 training에서도 residual stream, norm, softmax는 bf16 또는 fp32에 남는다. fp8은 matmul-specific feature다. 누군가 "완전히 fp8로 학습했다"고 말한다면 거의 틀림없이 "matmul은 fp8이고 나머지는 bf16"이라는 뜻이다.

---

## 5. Precision stack이 만드는 안정성 함정

아래 버그들은 모두 precision-coupled다. 보자마자 알아야 한다.

- **fp16에서 작은 `eps`로 생기는 NaN.** `v̂`가 0에 가깝고 `ε = 1e-8`이 fp16에서 underflow하면 AdamW의 `1 / (√v̂ + ε)`가 overflow할 수 있다. 해결책: bf16으로 옮기거나 `eps`를 `1e-5`로 올려라.
- **Loss-scaled clipping은 조용히 틀린다.** `clip_grad_norm_`를 호출하기 *전에* gradient를 unscale하라. [[gradient-clipping]]을 보라. 순서는 `unscale → clip → step`이다.
- **한 run에서 fp16과 bf16이 섞임.** 우연히 일어난다. 예를 들어 forward는 fp16인데 FSDP `reduce_dtype=bfloat16` 때문에 gradient reduction은 bf16인 경우다. 두 half-precision format은 서로 바꿔 쓸 수 없다. 조용한 divergence가 난다.
- **bf16으로 loss logging.** bf16은 precision이 약 2자리 십진수뿐이라 loss curve가 quantized/jaggy해 보인다. `log()`나 `.item()` 전에 `loss.float()`로 cast하라.
- **norm에서 fp32 reduction을 빼먹음.** §3을 보라. 항상 promote하라.
- **fp8 amax collapse.** 극단적인 gradient spike 아래에서 amax history가 saturation된다. 이후 scale은 쓸모없다. outlier-prone layer(embedding, head)에는 bf16 fallback을 유지하라.

---

## 6. 권장 표

| Context | Compute dtype | Reductions | Grads | Opt state | Loss scaling |
|---|---|---|---|---|---|
| 2025 pretrain(기본) | bf16 | fp32 | bf16 | fp32 | 없음 |
| Pre-H100 hardware(V100) | fp16 | fp32 | fp16(scaled) | fp32 | dynamic |
| H100+ frontier | bf16 + fp8 matmul | fp32 | bf16(+ fp8 backward) | fp32 | per-tensor / block |
| SFT / DPO | bf16 | fp32 | bf16 | fp32 | 없음 |
| RL(PPO/GRPO) | bf16 | fp32 | bf16 | fp32 | 없음; reward spike는 별도 추적 |
| Inference / eval | bf16 | bf16 OK | — | — | — |

---

## 연결과 다음 내용

- **[[adam]] / ch-01** — optimizer state는 fp32에 남는다. bf16 `v̂`는 작은 gradient에서 약 100스텝 안에 underflow한다.
- **[[gradient-clipping]] / ch-01** — `unscale → clip → step` 순서는 특히 mixed-precision concern이다.
- **[[batch-vs-layer-norm]] / ch-03** — RMSNorm / QK-norm placement를 논의할 때 norm-reduction precision rule이 다시 나온다.
- **ch-05 (FSDP)** — FSDP의 `MixedPrecision(param_dtype, reduce_dtype, buffer_dtype)`가 shard 전반의 이 policy를 선언하는 방법이다.
- **[[deepseek-v3]]** — 2024년 canonical fp8 training recipe. block-wise scaling의 reference implementation.

## 더 읽을거리

- [[mixed-precision]] — Micikevicius 2017 전체 발췌와 bf16 / fp8 successor context.
- [[deepseek-v3]] — production 671B MoE run에서의 per-block fp8 recipe.
- Karpathy의 "recipe"([[karpathy-training-neural-net-recipe]] 참조) — "start in fp32; add mixed precision only once training is stable".

## 함께 보는 시각화

**[figures/precision-range.html](figures/precision-range.html)** — fp32 / fp16 / bf16 / fp8-E4M3 / fp8-E5M2가 real line에서 값을 표현할 수 있는 위치를 나란히 그린 plot. Hover하면 각 format의 dynamic range 경계와, mantissa bit가 7개뿐인데도 bf16이 range에서 fp32와 일치하는 이유를 볼 수 있다.
