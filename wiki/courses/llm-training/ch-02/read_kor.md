<!-- chapter: ch-02
     track: foundations
     kind: content
     title: Numerical Precision, Determinism, and Train–Inference Mismatch
     deps: [ch-01]
     sources: [[mixed-precision]], [[fp8-formats-for-deep-learning]], [[pytorch-amp-autocast-gradscaler]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[scaling-laws-for-precision]], [[numerical-nondeterminism-llm-inference]], [[thinkingmachines-defeating-nondeterminism]], [[rollout-training-mismatch-tis]], [[minimax-m1]], [[small-scale-proxies-instabilities]], [[olmo-2]], [[olmo-2-official-configs]]
     figures: figures/precision-range.html
     revised: 2026-09 (generality revision)
-->

# Chapter 02 — 수치 정밀도, 결정성, 학습–추론 불일치

> **핵심 통찰.** Floating-point format은 어떤 magnitude가 살아남는지(range, exponent bit가 정함)와 얼마나 거칠게 반올림되는지(precision, mantissa bit가 정함)를 고정한다. Mixed-precision training은 master weight, gradient reduction, 선택 operation을 높은 precision으로 유지해 full-precision loss와 맞춘다. 공개된 FP8 근거는 loss 비교다. DeepSeek-V3 FP8 run은 약 16B parameter에서 1.33T token, 약 230B에서 0.9T token을 학습했을 때 BF16 대비 relative loss 0.25% 이내이며 downstream benchmark 비교는 보고하지 않는다([[deepseek-v3]] App. B.1). 같은 rounding은 학습 뒤 측정도 바꾼다. BF16 greedy decoding에서 DeepSeek-R1-Distill-Qwen-7B의 AIME'24 accuracy는 GPU type, GPU count, batch size configuration 12개에 걸쳐 standard deviation 9.15점이고 FP32에서는 0이다([[numerical-nondeterminism-llm-inference]] Table 3). RL에서는 같은 weight에서도 sampler와 learner가 같은 token에 다른 probability를 준다(DAPO, Qwen2.5-32B에서 maximum difference 1.0). 따라서 명목상 on-policy training은 off-policy가 된다([[rollout-training-mismatch-tis]] Fig. 1).
>
> **지침.** Hardware가 bf16을 지원하면 fp32 master weight 및 fp32 gradient reduction과 함께 bf16 autocast로 학습하라. Bf16은 fp32 exponent range를 가져 gradient underflow와 loss scaling이 없지만 relative spacing \(2^{-7}\)은 작은 weight update를 지운다(§2, §3; [[olmo-2-official-configs]]). 그렇지 않고 fp16이면 dynamic loss scaling을 쓰고 gradient clipping 전에 unscale하라([[mixed-precision]] §3.2). Lower-precision format(FP8 또는 quantized deployment)이 higher-precision format을 대체할 때는 ch-00의 development 및 unseen suite에서 둘을 비교하라. 인용된 parity 근거는 loss 또는 perplexity뿐이기 때문이다([[deepseek-v3]] App. B.1; [[fp8-formats-for-deep-learning]] Table 4; [[scaling-laws-for-precision]] §6). Greedy decoding checkpoint 두 개를 비교하면 한 recorded configuration과 FP32 또는 FP32 computation(LayerCast) 아래에서 둘 다 실행하거나, 여러 run(AIME'24 연구는 16 또는 64회)을 sample하고 mean을 보고하라. BF16 configuration만으로 AIME'24 standard deviation 9.15점이 나왔기 때문이다([[numerical-nondeterminism-llm-inference]] §3). RL rollout이 inference engine에서 오고 gradient가 training backend에서 오면 sampler log-probability를 기록하고 매 step sampler–learner mismatch를 log하라. ch-54처럼 correction을 적용하거나 numerical source를 제거하라. Correction 없는 mismatch 뒤 [[thinkingmachines-defeating-nondeterminism]]에서는 reward collapse, [[minimax-m1]] §3.2에서는 reward growth 없음이 나타났기 때문이다.

## 범용 모델에 이 장이 중요한 이유

Numerical format은 한 번 선택하면 pipeline의 모든 stage(pre-training → mid-training → SFT → preference optimization → RL → evaluation)에 영향을 준다. 네 측정 가능한 문제가 따른다.

1. **학습 결과.** Zero로 underflow하거나 infinity로 overflow하는 값은 update를 바꾼다. Original mixed-precision 연구에서 loss scaling 없는 FP16 Multibox SSD detector는 diverge했고, FP16 weight를 직접 update한 Mandarin speech model은 relative accuracy 80%를 잃었다([[mixed-precision]] Table 2, §3.1).
2. **Deployment 결과.** Checkpoint는 흔히 학습 뒤 quantize된다. [[scaling-laws-for-precision]] model에서 post-training quantization의 loss degradation은 parameter당 token과 함께 증가한다(Finding 1). 따라서 training token을 추가하면 full-precision model loss는 낮아지지만 quantized version loss는 높아질 수 있다(그 논문 §3.1).
3. **측정.** General capability의 claimed gain은 score difference다. Runtime configuration이 claimed difference보다 score를 더 움직이면 그 difference는 측정된 것이 아니다(§6; ch-00 §6).
4. **RL 정확성.** Policy-gradient method는 sample이 update되는 policy에서 온다고 가정한다. Inference engine과 trainer의 numerical difference는 error 없이 이 가정을 깨뜨린다(§7).

이 장에서 쓰는 용어. *dtype*은 numeric storage format(fp32, bf16, fp16, fp8)이다. *Mixed precision*은 한 training step에서 서로 다른 tensor를 서로 다른 dtype으로 저장하고 계산한다. *Autocast*는 operation별 dtype을 고르는 framework mechanism이다. *Master weight*는 optimizer가 update하는 full-precision parameter copy다. *Reduction*은 많은 element를 더 적은 수로 결합하는 operation(sum, mean, softmax normalizer)이다. *Rollout*은 RL 동안 policy에서 sample한 response이며, *sampler*가 rollout을 만들고 *learner*가 gradient를 계산한다.

## §1 Floating-point format: range, precision, rounding

**정의.** Binary floating-point number는 sign bit, \(e\)개 exponent bit, \(m\)개 mantissa bit를 가진다. Range는 format이 담을 수 있는 magnitude interval이며, precision은 인접한 representable value 사이 spacing이다.

**문제.** Range 밖 gradient, update, activation은 0 또는 infinity가 된다. Range 안 값은 가장 가까운 representable value로 round되며 rounding error는 그 magnitude의 spacing에 좌우된다.

**수식.** Biased exponent field \(E\)(IEEE-style format에서는 \(1\le E\le2^e-2\); E4M3도 finite value에 \(E=15\) 사용)와 mantissa field \(f\)를 둔 normal number는 다음이다.

`x = (−1)^s · (1 + f / 2^m) · 2^(E − bias)`,  `bias = 2^(e−1) − 1`

Subnormal number(\(E=0\))는 `x = (−1)^s · (f / 2^m) · 2^(1 − bias)`다.

- \(s\)는 sign bit, \(E\)는 stored exponent field, \(f\)는 \(m\) mantissa bit의 integer value, bias는 exponent range를 중앙에 두는 offset이다.
- 같은 exponent를 가진 인접 normal number의 간격은 \(2^{E-bias-m}\)이며 1.0에서는 \(2^{-m}\)이다. Rounding은 nearest value, tie는 even으로 한다.

**LLM training에서 쓰는 format.** E4M3와 E5M2 value는 [[fp8-formats-for-deep-learning]] Table 1에서 왔다. fp32, bf16, fp16 value는 formula에서 따라온다(status: derived). PyTorch 2.12.0 `torch.finfo`와도 일치한다.

| Format | sign/exp/mantissa bit | smallest subnormal | smallest normal | largest | 1.0의 spacing |
|---|---|---|---|---|---|
| fp32 | 1/8/23 | \(2^{-149}\approx1.40e-45\) | \(2^{-126}\approx1.18e-38\) | \(\approx3.40e38\) | \(2^{-23}\approx1.19e-7\) |
| bf16 | 1/8/7 | \(2^{-133}\approx9.18e-41\) | \(2^{-126}\approx1.18e-38\) | \(\approx3.39e38\) | \(2^{-7}\approx7.81e-3\) |
| fp16 | 1/5/10 | \(2^{-24}\approx5.96e-8\) | \(2^{-14}\approx6.10e-5\) | 65,504 | \(2^{-10}\approx9.77e-4\) |
| fp8 E5M2 | 1/5/2 | \(2^{-16}\approx1.53e-5\) | \(2^{-14}\approx6.10e-5\) | 57,344 | 0.25 |
| fp8 E4M3 | 1/4/3 | \(2^{-9}\approx1.95e-3\) | \(2^{-6}\approx1.56e-2\) | 448(infinity 없음) | 0.125 |

E4M3는 special-value bit pattern 대부분을 finite number에 재사용한다. 따라서 largest value가 240에서 448로 올라가고 infinity가 없어진다([[fp8-formats-for-deep-learning]] §3.1). E5M2는 IEEE convention을 따르며 “fewer mantissa bits를 가진 IEEE half precision으로 볼 수 있다.” fp16의 10 대신 2 bit다(§3).

**Worked example**(PyTorch 2.12.0 cast로 확인).

1. *Precision.* 값 1.00012는 1과 다음 fp16 value \(1+2^{-10}=1.000977\) 사이이며 1에 더 가까우므로 fp16은 1.0으로 저장한다. Bf16도 1.0으로 저장하며 fp32는 1.00012004로 저장한다([[numerical-nondeterminism-llm-inference]] Table 1).
2. *Range.* Gradient entry \(2^{-26}\approx1.49e-8\)은 fp16 smallest subnormal \(2^{-24}\)의 절반 \(2^{-25}\)보다 작으므로 fp16은 0으로 저장한다. Bf16은 smallest normal이 \(2^{-126}\)이므로 정확히 저장한다.
3. *Exponential overflow.* fp16은 result가 65,504을 넘을 때 overflow한다. \(\ln65{,}504=11.09\)이므로 fp16에서 계산한 `exp(12)`는 infinity다. `exp(11)` \(\approx59{,}874\)는 finite이며 fp16에 59,872로 저장된다. Bf16 threshold는 \(\ln(3.39e38)=88.7\)이다. 따라서 max logit을 빼지 않고 11 초과 logit에서 계산한 softmax는 fp16에서 overflow한다.

**Figure.** [figures/precision-range.html#formats](figures/precision-range.html#formats)는 reader가 임의 value를 넣고 다섯 format 각각의 stored value, rounding error, underflow/overflow 여부를 보게 한다.

**범용 모델에 대한 함의.** Range error는 update를 바꾸는 zero와 infinity를 낸다. Precision error는 operation당 작지만 operation order에 의존하며(§6), 이것이 이 장 뒤의 evaluation 및 RL effect 원천이다.

## §2 혼합 정밀도 학습: 어떤 tensor를 더 높은 정밀도로 유지하는가

**정의.** Mixed-precision training은 대부분 activation을 저장하고 대부분 matrix multiplication을 16-bit 또는 8-bit format으로 실행하되, 결과가 full-precision training과 맞도록 더 작은 tensor 및 operation 집합은 wider format으로 유지한다.

**문제.** Micikevicius et al.은 training memory를 대략 절반으로 줄이면서 hyperparameter를 바꾸지 않고 FP32 accuracy를 맞추는 것을 목표로 삼았다([[mixed-precision]] Abstract, §3.1).

**메커니즘**([[mixed-precision]] §3).

1. *FP32 master weight.* Optimizer는 FP32 copy를 update하고 FP16 copy는 forward/backward pass에 쓴다(§3.1).
2. FP16 gradient를 위한 *loss scaling*(이 장 §3).
3. *Arithmetic precision.* Dot product는 FP16으로 변환하기 전 FP32 value에 accumulate한다. “large reduction(vector element 전반의 sum)은 FP32로 수행해야” 하며, 주로 batch-normalization statistic 및 softmax다. Point-wise operation은 어느 dtype이든 쓸 수 있다(§3.3).

논문은 master weight에 두 이유를 든다(§3.1). 첫째 \(2^{-24}\)보다 작은 update는 FP16에서 zero가 된다. Mandarin speech model에서 weight-gradient exponent의 약 5%가 −24보다 작다(Fig. 2b). 둘째 weight가 update보다 적어도 2048배 크면 FP16 addition은 update를 round away할 수 있다. 이 model에서 FP16 weight를 직접 update하는 일은 relative accuracy 80%를 잃게 했다(Fig. 2a).

**Worked example: 16-bit weight의 update rounding**(course calculation; status: derived, PyTorch cast로 확인).

- fp16: \(1+2^{-12}\)는 1.0으로 round된다. 1에서 spacing이 \(2^{-10}\)이고 \(2^{-12}\)가 그 절반보다 작기 때문이다. \(1/2^{-12}=4096\) ratio는 [[mixed-precision]] §3.1의 2048 threshold를 넘는다.
- Bf16에는 mantissa bit 7개가 있으므로 해당 threshold는 \(2^8=256\)이다. Weight 0.02는 bf16에 0.02001953125로 저장되며, 그 magnitude에서 spacing은 \(2^{-13}\approx1.22e-4\)다. Element별 크기가 learning rate에 가까운 Adam-type update(ch-01)는 learning rate \(1e-5\)에서(비교를 위해 DeepSeek-V3 SFT learning rate는 \(5e-6\)에서 \(1e-6\)으로 decay하므로 [[deepseek-v3-recipe]], element별 update는 더 작다) half spacing \(6.1e-5\)보다 작다. 따라서 `w + u`는 같은 bf16 value를 반환하여 update가 사라진다. \(3e-4\) update는 3e-4 대신 2.44e-4만큼 stored value를 바꾸며 error 19%다. Fp32에서는 두 update 모두 유지된다.
- 결론: bf16은 loss scaling 필요성(§3)은 없애지만 full-precision master weight 필요성은 없애지 않는다.

**Source별 현재 관행.**

- *PyTorch autocast(framework policy)* ([[pytorch-amp-autocast-gradscaler]] amp.md L194-272). CUDA에서 `linear`, `matmul`, `mm`, `bmm`, `addmm`, convolution은 lower-precision dtype으로 실행한다. `softmax`, `log_softmax`, `cross_entropy`, `nll_loss`, `layer_norm`, `group_norm`, `sum`, `prod`, `cumsum`, `exp`, `log`, `pow`, `rsqrt`는 float32로 실행한다. 목록 밖 operation은 input dtype으로 실행한다(L186-188). `scaled_dot_product_attention`, `rms_norm`은 CUDA list에 없으므로 이들을 통한 fused attention 및 RMSNorm kernel은 input dtype으로 실행한다. 이전 operation과 parameter가 bf16이면 bf16이다. Float32 list는 한 framework default일 뿐 모든 training stack의 규칙은 아니다.
- *OLMo 2 7B stage 1(released config)* ([[olmo-2-official-configs]]). `precision: amp_bf16`, `fsdp.precision: mixed`이며, code는 이를 computation 동안 bf16 parameter/buffer와 gradient reduction용 `reduce_dtype=torch.float32`로 정의한다(config.py L836-840, L1339-1344).
- *DeepSeek-V3 FP8 framework* ([[deepseek-v3]] §3.3.1, §3.3.3). 각 Linear의 세 GEMM(general matrix multiplication)은 FP8로 실행한다. Embedding module, output head, MoE gating, normalization operator, attention operator는 BF16 또는 FP32에 남는다. Master weight와 batch size accumulation에 “used”되는 gradient는 FP32에 남는다. AdamW moment는 “observable performance degradation 없이” BF16에 저장한다.

```python
# github.com/allenai/OLMo@090253d olmo/config.py L1339-1344
            elif self.fsdp.precision == FSDPPrecision.mixed:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=torch.float32,
                    buffer_dtype=self.autocast_precision,
                )
```

**조건과 한계.** 세 source는 higher precision에 남기는 집합이 다르다. Micikevicius et al.(CNN/RNN, 2017)은 reduction을 FP32에 둔다. PyTorch는 이름 붙은 list를 유지한다. DeepSeek-V3는 attention/normalization을 BF16 또는 FP32, optimizer moment를 BF16에 둔다. 어느 source도 LLM downstream capability의 operation별 ablation을 보고하지 않는다(열린 질문). DeepSeek-V3의 BF16 moment 설명에는 수치가 없다.

**범용 모델에 대한 함의.** Gradient와 master weight는 작은 update를 전달하는 tensor다. SFT, RL, annealing의 낮은 learning rate는 weight 대비 update를 더 작게 하므로, 이미 capable한 model을 fine-tune하는 stage에서는 이 tensor가 wider format에 있어야 한다(위 worked example에서 도출).

## §3 fp16 loss scaling과 bf16에 필요 없는 이유

**정의.** Loss scaling은 작은 gradient가 fp16 range 안에 들어가도록 backpropagation 전에 loss에 factor \(S\)를 곱하고, 쓰기 전에 gradient를 \(S\)로 나누는 일이다.

**문제.** Fp16 normalized exponent range는 \([-14,15]\)인데 activation gradient는 작은 magnitude가 지배한다. Multibox SSD에서는 \([2^{-27},2^{-24})\) gradient value를 보존해야 했으며 scaling 없이는 model이 diverge했다([[mixed-precision]] §3.2, Fig. 3).

**메커니즘**([[mixed-precision]] §3.2; [[pytorch-amp-autocast-gradscaler]]).

1. Loss \(L\)을 계산해 \(S\)를 곱한다.
2. Backpropagate한다. Chain rule에 따라 모든 gradient에 \(S\)가 곱해진다.
3. Weight gradient를 \(S\)로 나누어 unscale한다. 이는 “right after the backward pass but before gradient clipping or any other gradient-related computations”에 하므로 clipping threshold와 weight decay는 바꿀 필요가 없다.
4. Gradient 하나라도 infinity 또는 NaN이면 “irreversibly damage the weights”할 수 있으므로 적용하지 않는다. 논문은 update skip을 한 option으로 든다(§3.2).
5. Dynamic scaling이 \(S\)를 조정한다. PyTorch 2.12.0 `GradScaler` default는 initial \(S=2^{16}\)이다. inf/NaN이면 \(S\)에 0.5를 곱하고 step을 skip하며, inf/NaN 없는 연속 2000 step 뒤에는 \(S\)에 2.0을 곱한다(grad_scaler.py L123-131, L96-101). Micikevicius et al.은 8부터 32K까지 constant factor를 썼고 automatic selection은 future work로 든다(§3.2, §5).

**수식.** fp16에서 \(\hat g=S\cdot\nabla_\theta L\), fp32에서 \(g=\hat g/S\)다. Entry는 \(|g|\cdot S>2^{-25}\)(fp16 smallest subnormal의 절반)일 때 보존되고, \(S\cdot\max_i|g_i|\le65{,}504\)일 때 step이 valid하다.

- \(\theta\)는 parameter, \(L\)은 loss, \(\nabla_\theta L\)은 true gradient, \(\hat g\)는 scaled fp16 gradient, \(g\)는 recovered gradient, \(i\)는 gradient entry index다.

**Worked example**(PyTorch cast로 확인. [figures/precision-range.html#loss-scaling](figures/precision-range.html#loss-scaling)의 default \(g=10^{-7.83}\approx1.48e-8\), largest gradient 1.0, \(S=2^{16}\)은 아래 \(1.49e-8\) entry와 같은 outcome을 낸다).

- \(g=1.49e-8\)(약 \(2^{-26}\)): scaling 없는 fp16은 0으로 저장한다.
- \(S=2^{16}\)이면 \(Sg\approx2^{-10}\approx9.8e-4\)로 normal fp16 value다. Fp32에서 \(S\)로 나누면 1.49e-8을 회복한다.
- 같은 step의 largest gradient entry가 1.0이면 \(S\cdot1.0=65{,}536>65{,}504\)이라 fp16은 infinity가 된다. `GradScaler`는 step을 skip하고 \(S=2^{15}\)로 설정한다. 다음 step은 \(S\cdot1.0=32{,}768\)로 finite다.
- Figure는 reader가 \(g\), largest gradient, \(S\)를 움직여 어느 entry가 살아남는지와 step skip 여부를 보게 한다.

**근거**([[mixed-precision]] Table 2, §4.5). Faster R-CNN mAP는 FP32 69.1%, scaling 없는 mixed precision 68.6%, scaling 적용 69.7%였다. Multibox SSD는 FP32 76.9%, scaling 없이는 divergence, factor 8에서 77.1%였다. 1B-word dataset의 bigLSTM은 scaling 없는 FP16에서 300K iteration 뒤 diverge했지만 factor 128은 FP32와 맞았다. Table 1의 ResNet-50을 포함한 ILSVRC CNN 6개는 scaling이 필요 없었다(ResNet-50 FP32 75.92%, mixed 76.04%).

**Bf16에 필요 없는 이유.** Bf16은 fp32의 exponent bit 8개를 유지하므로 smallest normal이 \(1.18e-38\)이며 위 \(2^{-26}\) entry를 정확히 저장한다(§1). 비용은 mantissa bit 7개로 precision에 영향을 줄 뿐 range에는 영향을 주지 않는다. PyTorch 문서는 CPU의 bfloat16 mixed precision을 `GradScaler` 없이 autocast만 쓰는 것으로 설명한다([[pytorch-amp-autocast-gradscaler]] amp.md L34-39).

**조건과 한계.** Loss-scaling 근거는 2017년 CNN, RNN, LSTM에서 왔으며, 이 장 source 중 Transformer LLM에서 loss scaling을 측정한 것은 없다. FP8의 경우 FP8-format 저자들은 narrower range에서 overflow가 더 자주 나므로 overflow update skip은 “not a good choice”라고 말한다. FP8은 saturation과 함께 per-tensor 또는 더 fine한 scaling을 쓴다([[fp8-formats-for-deep-learning]] §2).

**범용 모델에 대한 함의.** Fp16 run에서 skipped step 수는 training-health metric이다. Skipped step은 해당 batch data가 model을 update하지 못한 것이다.

## §4 FP8 학습: format, scaling granularity, 검증이 보이는 것

**정의.** FP8 training은 matrix multiplication input을 8-bit format(E4M3 또는 E5M2)으로 quantize하고 곱한 뒤 output을 wider format으로 쓴다.

**문제.** E4M3는 binade 18개, E5M2는 32개를 포괄한다([[fp8-formats-for-deep-learning]] §3.1). Low-precision training에서 “overflows and underflows are common challenges due to the limited dynamic range of the FP8 format”이다([[deepseek-v3]] §3.3.2). 따라서 cast 전에 value에 scale을 곱하며, outlier 하나가 그 scale을 공유하는 모든 value의 scale을 정한다(§3.3.2).

**메커니즘.**

1. Value group(whole tensor, 1×128 tile, 128×128 block)을 고른다.
2. \(s_b=F_{max}/\max_{i\in b}|x_i|\)를 계산한다. E4M3에서 \(F_{max}=448\)이다([[fp8-formats-for-deep-learning]] §2).
3. \(q_i=\operatorname{round}_{E4M3}(s_b\cdot x_i)\)를 저장하며 overflow value는 saturate한다.
4. FP8로 multiply하고 output은 BF16 또는 FP32에 쓰며 scale로 나눈다.

- \(b\)는 scaling group, \(x_i\)는 group value, \(s_b\)는 scale, \(q_i\)는 stored 8-bit value, \(q_i/s_b\)는 recovered value다.

**보고된 선택.**

- *Format 분리.* FP8-formats 논문은 weight/activation에 E4M3, gradient에 E5M2를 권한다([[fp8-formats-for-deep-learning]] §3). DeepSeek-V3는 fine-grained scaling 때문이라고 하며 모든 tensor에 E4M3를 쓴다([[deepseek-v3]] §3.3.2).
- *Granularity.* DeepSeek-V3는 activation을 1×128 tile별, weight를 128×128 block별 scale한다. Scale은 이전 iteration history(delayed scaling)가 아니라 current maximum에서 online으로 계산한다(§3.3.2).
- *Accumulation.* H800의 FP8 GEMM accumulation은 약 14 bit를 유지한다. Inner dimension \(K=4096\)에서 maximum relative error는 거의 2%였고, DeepSeek-V3는 128 element마다 partial result를 FP32 register에 copy한다(§3.3.2).
- *예외.* Attention 뒤 Linear input은 power-of-2 scale의 custom E5M6 format을 쓴다(§3.3.3).

**Worked example: tensor당 scale 하나 대 group당 scale 하나**(PyTorch `float8_e4m3fn` cast course calculation; [figures/precision-range.html#block-scaling](figures/precision-range.html#block-scaling)에서 상호작용 가능). Group A = [0.004, −0.001, 0.0025, 0.0006], group B = [1000, −3.2, 0.75, 12]라고 하자.

- Tensor당 scale 하나: \(s=448/1000=0.448\)이다. 그러면 \(0.001\cdot0.448=4.48e-4\)이며 E4M3 smallest subnormal 절반(\(2^{-10}\approx9.8e-4\))보다 작아 0으로 저장된다. 0.0006도 0으로 저장된다. 0.0025는 1.12e-3이 되어 \(2^{-9}\approx1.95e-3\)로 round되고 0.00436으로 recovery된다(error 74%).
- Group당 scale 하나: \(s_A=448/0.004=112{,}000\)이다. −0.001은 −112가 되며 E4M3가 정확히 저장한다. 0.0025는 280이 되어 288로 round되고 0.00257로 recovery된다(error 2.9%). 0.0006은 67.2가 되어 64로 round되고 0.000571로 recovery된다(error 4.8%). 잃는 value는 없다.

**근거와 이것이 확립하는 것.**

| 비교 | 설정 | 지표 | 결과 | 위치 |
|---|---|---|---|---|
| FP8 대 BF16 | MoE 약 16B parameter, 1.33T token | training loss | relative error < 0.25% | [[deepseek-v3]] App. B.1 |
| FP8 대 BF16 | MoE 약 230B parameter, 약 0.9T token | training loss | relative error < 0.25% | App. B.1 |
| Activation gradient의 block-wise quantization | MoE 약 16B, 약 300B token | training | diverged | App. B.2 |
| Simulated FP8 대 16-bit | GPT 126M–175B | training perplexity | 175B: FP8 6.68 대 bfloat16 baseline 6.65(baseline은 training 75%에서 보고) | [[fp8-formats-for-deep-learning]] Table 4 |

- 각 행은 결과(단일 연구)다. DeepSeek-V3 framework의 두 scale에서 0.25% 이내 loss parity는 확립됐으며, 저자들은 seed variance를 보고하지 않고 이를 “well within the acceptable range of training randomness”라고 부른다(§3.3).
- 이 source들이 확립하지 않은 것: downstream benchmark, unseen evaluation suite, long-context/agentic task, base model의 SFT/RL 반응에서의 parity다. 어느 source도 FP8과 16-bit GPT-style 또는 MoE language model의 downstream task comparison을 보고하지 않는다(열린 질문).
- DeepSeek-V3 report는 FP8 GEMM이 “theoretically” computational speed를 두 배로 한다고 말하지만 measured end-to-end speedup은 보고하지 않는다(§3.3.1).

**범용 모델에 대한 함의.** Average validation loss는 모든 token에 걸쳐 합산된다. Precision change는 FP8와 16-bit checkpoint 모두를 같은 evaluation configuration에서 ch-00 development 및 unseen suite에 score할 때만 general capability에 대해 평가된다.

## §5 Precision과 parameter당 token

**정의.** *Parameter당 token*은 training token을 parameter로 나눈 \(D/N\)이다. *Post-training quantization(PTQ)*은 추가 training 없이 trained model weight를 더 적은 bit로 round한다. \(\delta_{PTQ}\)는 PTQ가 일으킨 loss 증가다.

**문제.** [[scaling-laws-for-precision]]은 Llama-3-8B의 \(D/N\approx2000\)을 인용한다(§2.2). 이는 compute-optimal training과 연관된 parameter당 약 20 token(ch-08a)의 약 100배다. Quantization cost가 \(D/N\)과 함께 커지면 training budget과 deployment precision을 독립적으로 선택할 수 없다.

**설정**([[scaling-laws-for-precision]] §2.3). Dolma V1.7 token 1.5B–26B로 학습한 OLMo-style model, non-embedding parameter 30M–220M; 3–16 bit precision의 run 465개; validation은 parameter 최대 1.7B, token 26B까지다. PTQ에는 GPTQ를 쓰고 다른 두 method로 복제했다(§3).

**수식**(Finding 1, Eq. 2).

`δ_PTQ(N, D, P_post) = C_T · (D^γ_D / N^γ_N) · e^(−P_post / γ_post)`

- \(N\)은 parameter, \(D\)는 training token, \(P_{post}\)는 post-training weight precision(bit), \(C_T,\gamma_D,\gamma_N,\gamma_{post}\)은 positive fitted constant다. \(\delta_{PTQ}\)는 \(D\)와 함께 상승하고 \(N\)과 함께 하락하며, \(P_{post}\)이 작아질수록 exponential하게 상승한다(§3.1).

**Worked example**(course calculation; status: App. K constant에서 도출했으며 저자들은 이 constant가 자기 setting 밖에서 “are unlikely to be useful”하다고 말한다). \(\gamma_D=0.5068\)이다. \(N,P_{post}\) 고정에서 \(D/N\)을 20에서 200으로 올리면 \(\delta_{PTQ}\)는 \(10^{0.5068}=3.21\)배, 20에서 2000으로 올리면 \(100^{0.5068}=10.3\)배가 된다. Quantization 전 loss는 \(D\)가 커지며 하락하므로 어느 effect가 큰지가 질문이다. \(D/N\gg10^3\)에서 저자들은 quantization effect가 “nontrivial around 5-bits, and dominant below that”이라고 발견한다(App. E.1).

**추가 결과.**

- Finding 2(§4.2): Training precision은 effective parameter count에 작용한다. \(N_{eff}=N(1-e^{-P_w/\gamma_w})(1-e^{-P_a/\gamma_a})(1-e^{-P_{kv}/\gamma_{kv})\)이며 \(P_w,P_a,P_{kv}\)는 weight, activation, KV cache training precision이다.
- Finding 3(§4.3): Cost \(\propto N\cdot D\cdot P\) 아래 \(N,D\), precision을 함께 optimize하면 integer-type quantization fit은 7-8 bit가 compute-optimal임을 준다. \(N\)을 고정하면 optimal precision은 \(\log C\)와 함께 증가한다.
- §5: Lower precision으로 학습한 model은 PTQ 아래 degradation이 더 작다(unified fit \(R^2=0.90\)).

**조건과 한계**(§6). Architecture 하나, “without downstream model evaluations”인 loss만, model 최대 1.7B parameter, experiment는 \(D/N\approx10^3\)까지다. 상태: 결과(단일 연구). Large over-trained model의 quantized loss에 대한 \(D/N\) effect가 downstream capability로 이어지는지는 열린 질문이다.

**범용 모델에 대한 함의.** Model이 높은 \(D/N\)에서 학습되고 quantize되어 serving될 때 보호할 capability는 quantized model의 것이다. Deployment precision을 고정하기 전에 actual \(D/N\), bit width에서 held-out loss와 ch-00 suite의 \(\delta_{PTQ}\)를 측정하라.

## §6 평가의 비결정성

**정의**([[thinkingmachines-defeating-nondeterminism]]). *Run-to-run nondeterminism*은 같은 input의 같은 kernel이 다른 output을 내는 것이다. *Batch invariance*는 request output이 batch size 또는 sequence split 방식에 의존하지 않는 성질이다. Floating-point addition은 associative하지 않다. `(0.1 + 1e20) − 1e20`은 0이지만 `0.1 + (1e20 − 1e20)`은 0.1이다.

**문제.** Greedy decoding(temperature 0)은 prompt마다 하나의 answer를 낼 것으로 기대된다. 실제로 answer는 serving load와 hardware에 따라 바뀌므로 benchmark score는 model뿐 아니라 evaluation configuration의 함수다.

**메커니즘.**

1. LLM GPU forward pass는 atomic add를 쓰지 않으므로, 같은 batch의 같은 input에 같은 kernel은 bitwise-equal result를 낸다([[thinkingmachines-defeating-nondeterminism]], “When are atomic adds needed?”).
2. RMSNorm, matrix multiplication, attention kernel은 shape에 따라 reduction strategy를 고른다. Split reduction, 서로 다른 tensor-core instruction, query token 처리 수에 따른 KV split이 이에 해당한다(“How do we make kernels batch-invariant?”).
3. Server load가 batch size를 바꾸므로 한 request의 reduction order와 rounding도 바뀐다.
4. Candidate token 두 개 probability가 가까우면 rounding difference가 greedy choice를 뒤집고 이후 response도 달라진다. BF16 pair 하나에서는 configuration 하나에서 top two가 “know” 49.75%, “have” 43.91%인데 다른 configuration에서는 “have” 46.65%, “know” 46.64%였다([[numerical-nondeterminism-llm-inference]] Fig. 3).
5. Bf16의 mantissa bit 7개는 FP16 10개 또는 FP32 23개보다 operation당 rounding이 크다. 저자들은 BF16 top-1 probability variation이 더 큰 원인을 이것으로 본다(§3.2, Fig. 4; 해석).

**근거.**

- Qwen3-235B-A22B-Instruct-2507에서 prompt 하나의 temperature 0 completion 1000개는 distinct completion 80개였다. 102 token까지는 모두 일치한 뒤 992개는 “Queens, New York”, 8개는 “New York City”로 계속됐다. Batch-invariant kernel에서는 1000개가 모두 동일하다([[thinkingmachines-defeating-nondeterminism]], “How nondeterministic are completions?”).
- vLLM에서 네 7B–8B model, benchmark 5개, configuration 12개(L40S/A100, GPU 2/4개, batch 8/16/32)를 썼다([[numerical-nondeterminism-llm-inference]] §3.1).

| Model | Benchmark | 12 configuration의 greedy accuracy 표준편차: BF16 / FP16 / FP32 | 위치 |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | AIME'24 | 9.15% / 5.74% / 0 | Table 3 |
| DeepSeek-R1-Distill-Llama-8B | AIME'24 | 4.60% / 6.00% / 5.8e-17 | Table 3 |
| Qwen2.5-7B-Instruct | AIME'24 | 1.71% / 1.45e-17 / 1.45e-17 | Table 3 |
| DeepSeek-R1-Distill-Qwen-7B | MATH500 | 1.04% / 1.12% / 0.12% | Table 3 |

- DeepSeek-R1-Distill-Qwen-7B의 AIME'24 output length는 BF16에서 standard deviation 9,189.53 token, FP32에서 0이었다(Table 4). MATH500에서는 example 96.6%가 BF16, 73.0%가 FP16, 2.2%가 FP32에서 configuration 간 divergence를 보였다(Fig. 5).
- Sampling도 영향받는다. DeepSeek-R1-Distill-Qwen-7B에서 6 configuration 간 AIME'24 Pass@1(16 sample) standard deviation은 BF16 1.7151, FP16 0.8273이었다. 64 sample에서는 ordering이 역전됐다(BF16 0.3749, FP32 0.7377). 저자들은 이를 AIME'24 문제 30개 탓으로 본다(Table 5, §3.3).

**Mitigation과 비용.**

- *Batch-invariant kernel*: Qwen-3-8B에서 sequence 1000개는 default vLLM 26 s, unoptimized deterministic vLLM 55 s, improved attention kernel 42 s였다. Batch-invariant matmul은 cuBLAS보다 약 20% 느리다([[thinkingmachines-defeating-nondeterminism]], “Performance”). Batch invariance는 tensor-parallel size 또는 GPU type 변경을 포괄하지 않는다([[numerical-nondeterminism-llm-inference]] §2.2).
- *LayerCast*: BF16 weight storage와 FP32 computation이다. Configuration 간 divergence는 3.4% 아래이며, full FP32보다 memory 34%가 적다. 저자들은 full FP32가 BF16 대비 memory와 inference time을 두 배로 만든다고 말한다(§4).

**Worked example**(course calculation). AIME'24에는 문제 30개가 있으므로 한 문제는 3.33점이다([[numerical-nondeterminism-llm-inference]] §3.3). Configuration 간 standard deviation 9.15점은 약 2.7문제에 해당한다. Checkpoint A가 4 A100, batch 32의 BF16 greedy에서 40.0(30개 중 12개), checkpoint B가 2 L40S, batch 8에서 46.7(30개 중 14개)이라고 하자. 6.7점 차이는 DeepSeek-R1-Distill-Qwen-7B에서 configuration 간 측정된 9.15점 standard deviation보다 작으므로 comparison은 이를 checkpoint 탓으로 돌릴 수 없다. 같은 configuration에서 둘을 평가하면 각 score는 repeatable하지만 dependence는 사라지지 않는다. 9.15점 spread가 fixed configuration들 사이 spread이므로 다른 fixed configuration은 두 checkpoint 순서를 바꿀 수 있다.

**조건과 한계.** 근거는 output 최대 32,768 token의 7B–8B model([[numerical-nondeterminism-llm-inference]] appendix의 Qwen3-32B 포함)을 포괄한다. Long reasoning output이 가장 큰 effect를 보인다. Status: configuration-dependent greedy output 존재는 위 독립 source 두 개가 복제했다. Benchmark별 magnitude는 결과(단일 연구)다.

**범용 모델에 대한 함의.** Reasoning 및 long-output task가 BF16 configuration noise가 가장 큰 capability axis다. 이 axis에 근거한 breadth claim에는 FP32-computation evaluation, batch-invariant serving, 또는 configuration을 보고한 multi-sample average가 필요하다(ch-51).

## §7 RL의 sampler–learner mismatch

**정의.** Hybrid RL framework에서 rollout은 probability \(\pi_{sampler}(a;\theta)\)의 inference engine이 만들고, gradient는 probability \(\pi_{learner}(a;\theta)\)의 training backend가 계산한다. \(\theta\)는 shared weight, \(a\)는 token이다([[rollout-training-mismatch-tis]] Summary).

**문제와 수식.** 계산되는 REINFORCE gradient는 다음이다.

`E_{a ∼ π_sampler(θ)} [ R(a) · ∇_θ log π_learner(a; θ) ]`

- \(R(a)\)는 sampled response reward다. 이는 \(\pi_{sampler}=\pi_{learner}\)일 때만 on-policy gradient와 같다. Logged reward는 \(\pi_{learner}\)가 아니라 \(\pi_{sampler}\) 아래 expectation이다(Fig. 6).

**Mismatch의 numerical source.**

1. *Kernel과 precision.* MiniMax-M1 RL에서 training-mode와 inference-mode token probability는 “a precision mismatch between the training and inference kernels” 때문에 달랐다. Layer-by-layer analysis는 error를 LM head의 high-magnitude activation에서 찾았다([[minimax-m1]] §3.2).
2. *Parallelism.* DAPO-Math prompt에서 vLLM TP1 sampler와 FSDP SP1 learner는 maximum token-probability mismatch 0.5 초과 response 1개, TP2 sampler는 2개를 냈다. Learner에 Ulysses SP8을 더하자 count가 “from two to double digits”로 올랐다. Same-parallelism TP2 또는 TP4는 small number를 냈으나 저자들은 exact count를 보고하지 않는다([[rollout-training-mismatch-tis]] Fig. 7-8).
3. *Response length.* 20K token cap response는 mean mismatch가 비슷한 4K cap보다 maximum mismatch가 높다. 20K response 첫 4K token 안 maximum mismatch는 독립적인 4K response보다 흔히 높다(Fig. 9-10).
4. *Batch composition.* Batch-invariant가 아닌 kernel은 token probability를 batch 내 다른 sequence에 의존하게 한다(§6).
5. *Quantized rollout.* 저자들은 FP8/INT8 rollout을 gap이 큰 사례, BF16 rollout을 INT8보다 gap이 작은 사례로 설명한다(Fig. 4 paragraph; Fig. 5 뒤 BF16 paragraph).

- Sampling에 쓴 probability를 반환하도록 vLLM을 patch하고 LM head를 fp32로 cast해도 DAPO-32B mismatch는 제거되지 않았다(Fig. 1).

**Effect의 근거.**

- Qwen2.5-32B의 DAPO: maximum token-probability difference 1.0(일부 token은 \(\pi_{sampler}=1\), \(\pi_{learner}=0\)); Qwen2.5-0.5B의 GSM8K PPO: 약 0.4였다([[rollout-training-mismatch-tis]] Fig. 1-2). DAPO-32B의 INT8 rollout에서는 entropy가 0.2 아래로 떨어지고 response가 비정상적으로 길어졌으며 k1 KL estimate가 자주 negative였다. Truncated importance sampling은 이를 되돌렸다(Fig. 5). BF16 rollout에서 correction이 없으면 logged reward는 더 높지만 AIME accuracy는 correction 적용보다 낮았다(Fig. 6).
- MiniMax-M1(총 456B, activated 45.9B): mismatch가 “prevented reward growth”했다. LM head를 FP32로 계산하자 training/inference probability의 Pearson correlation이 0.987319에서 0.997135로 올랐고(Fig. 3), 이후 reward가 증가했다([[minimax-m1]] §3.2). 이 문제는 더 작은 dense softmax-attention model에는 나타나지 않았다.
- Qwen 2.5-VL instruct 8B에서 시작한 Bigmath RLVR, rollout 최대 4096 token: importance weighting 없이 reward가 step 318 부근 KL spike와 함께 collapse했다. Importance weighting을 쓰면 sampler–trainer KL은 0.001 부근에 머물렀고, bitwise-identical sampler/trainer에서는 KL 0, training은 진행됐다([[thinkingmachines-defeating-nondeterminism]], “True on-policy RL”).
- 상태: uncorrected mismatch가 RL training을 저해했다는 것은 복제됨(독립 source 세 개: entropy collapse와 lower accuracy, reward growth 없음, reward collapse). 각 수치 magnitude는 결과(단일 연구)다.

**Worked example: correction이 truncated인 이유**([[rollout-training-mismatch-tis]] callout). Token importance ratio \(r=\pi_{learner}/\pi_{sampler}=16\)이면 untruncated importance weight는 그 token의 gradient variance 기여를 \(r^2=256\)배로 만든다. \(C=2\)에서 weight를 truncate하면 4배, \(C=8\)에서는 64배다. \(\pi_{learner}=0\), \(\pi_{sampler}=1\)인 token weight는 0이다. Truncated estimator와 PPO form은 ch-54에서 다룬다.

**Negative advantage**(gradient로서 negative: negative advantage는 sampled token likelihood를 명시적으로 낮춘다). TIS 저자들의 entropy-collapse 설명은 negative advantage 및 \(\pi_{learner}/\pi_{sampler}<1\) rollout에서 \(\pi_{learner}\) 하락이 \(\pi_{sampler}\)에 나타나지 않을 수 있어 update가 \(\pi_{learner}\)를 계속 내린다는 것이다. Exact mechanism은 열려 있다고 말한다([[rollout-training-mismatch-tis]], mechanism paragraph; 해석). Ch-43a가 negative gradient를 다룬다.

**조건과 한계.** BF16 rollout의 DeepSeek-R1-Distill-Qwen-1.5B에서는 mismatch가 작았고 correction도 gain이 없었다([[rollout-training-mismatch-tis]] Fig. 3). Consistently 더 좋은 sampler backend는 없었다(Fig. 11).

**범용 모델에 대한 함의.** Entropy collapse와 abnormal response length는 output diversity를 좁히고 pass@k를 바꾼다(ch-43). Held-out accuracy와 맞지 않는 logged reward는 training signal과 capability의 연결을 없앤다. Sampler–learner KL, maximum mismatch, mean mismatch는 reward와 함께 RL health metric이다.

## §8 불안정성 증상과 수치 원인ㆍ데이터 원인 분리

**수치 mechanism이 있는 증상.**

1. *Non-finite gradient.* 65,504 초과 fp16 product는 infinity가 되며 loss scaling이면 step을 skip한다(§3).
2. *Attention logit growth.* High learning rate의 2.4M–1.2B parameter model에서 query/key norm이 커진다. Maximum attention logit이 \(1e4\)를 넘은 run은 diverge했고, learning rate \(1e-2\)의 4.8B run도 작은 model에서 예측한 대로 diverge했다. QK-layernorm이 이를 완화한다([[small-scale-proxies-instabilities]] §3.1.1, §3.3). 이 run은 TPU에서 bfloat16을 썼다(§2.1).
3. *Output logit divergence.* Weight decay 없는 model은 training 끝으로 갈수록 output logit이 매우 negative가 된다. Softmax normalizer \(Z\)에 대해 \(1e-4\cdot\log^2Z\)를 더하는 z-loss가 이를 해결한다(§3.1.2).
4. *AdamW \(\epsilon\) 부근 gradient RMS.* First MLP layer gradient RMS는 model size와 learning rate가 커질수록 낮아지고, 검사한 최대 scale/LR에서 \(1e-8\) 근처다. 4.8B, learning rate 0.3에서 \(\epsilon=1e-15\)은 loss를 낮추고 \(\epsilon=1e-6\)은 diverge했다(§3.4, Figs. 11-12). MiniMax-M1은 RL gradient \(1e-18\)–\(1e-5\), 대부분 \(1e-14\) 아래를 보고하며 \(\epsilon=1e-15\)을 쓴다([[minimax-m1]] §3.2). OLMo 2는 \(\epsilon\)을 \(1e-5\)에서 \(1e-8\)로 낮췄고, 이는 “lowers and stabilizes the norm of the gradient early in training”한다([[olmo-2]] §3.4.1).
5. *Implementation-dependent backward.* OLMo 2 FlashAttention z-loss와 PyTorch z-loss는 같은 forward value이지만 다른 backward result였고 z-loss curve도 diverge했다. 저자들은 “root cause lies in differences in precision”을 의심했고 cross-entropy/downstream task에는 effect가 없었지만 divergence point에서 재학습했다([[olmo-2]] §3.3.3, Fig. 8).

**Worked example: \(\epsilon\)과 dtype**(course calculation). Element별 Adam update는 \(\hat m/(\sqrt{\hat v}+\epsilon)\)이다(ch-01). Constant gradient \(g\)에서 bias correction 뒤 \(\sqrt{\hat v}\approx|g|\)이므로 \(\epsilon=0\) 대비 update size는 \(|g|/(|g|+\epsilon)\)이다. \(g=1e-8,\epsilon=1e-8\)이면 ratio 0.5, \(\epsilon=1e-15\)이면 0.9999999다. MiniMax-M1이 보고한 magnitude \(g=1e-14\)에서 \(\epsilon=1e-8\)이면 \(1e-6\)이므로 parameter가 움직이지 않으며, \(\epsilon=1e-15\)이면 0.91이다. \(\epsilon=1e-15\)은 fp16에서는 0으로 저장된다(smallest subnormal \(5.96e-8\)). Bf16에서는 \(9.99e-16\)으로 저장된다. 따라서 optimizer computation dtype이 존재 가능한 \(\epsilon\) value를 제한한다.

**데이터 원인이 있는 증상.** OLMo 2는 loss/gradient-norm spike batch에 긴 repeated n-gram sequence가 자주 있음을 발견했다. 그러나 관계는 deterministic하지 않았다. 같은 sequence가 큰 model에서는 spike하고 작은 model에서는 spike하지 않았으며, 한 data order에서는 spike하고 reshuffle 뒤에는 spike하지 않았고, spike하지 않은 batch에도 나타났다. 1-13 token n-gram이 32회 이상 반복된 document를 제거하고 그런 sequence를 loss에서 mask하면 spike는 줄었지만 사라지지 않았고 gradient norm의 느린 성장은 바꾸지 않았다([[olmo-2]] §3.1, Fig. 3).

**지표: spike score**([[olmo-2]] §3.2). Time series에서 직전 1,000 value rolling average의 7 standard deviation 이상인 value 비율이다. 예를 들어 step 20,000에서 gradient-norm value 80개가 flag되면 spike score는 0.40이다. OLMo 2는 initialization change에서 gradient-norm spike score 0.40 → 0.03(§3.2), reordered norm + QK-norm에서 0.108 → 0.069(§3.3.2, Fig. 7), embedding weight decay 유/무에서 0.16 대 0.092(§3.4.2, Fig. 10)를 보고한다.

**Diagnostic procedure**(위 source에서 구성한 course procedure; 어떤 source도 full sequence를 시험하지 않음; 해석).

1. Non-finite loss/gradient norm, fp16이면 skipped step 및 current loss scale을 확인한다.
2. Spike batch에서 repeated n-gram과 다른 degenerate sequence를 조사한다([[olmo-2]] §3.1).
3. 같은 batch/kernel로 이전 checkpoint에서 step을 replay한다. Spike가 재현되지 않으면 복원하지 않은 state 또는 nondeterministic kernel이 원인에 관여한다. FlashAttention backward는 determinism을 위해 atomic add 또는 recomputation이 필요한 흔한 LLM operation이다([[thinkingmachines-defeating-nondeterminism]], “When are atomic adds needed?”).
4. Logit, loss, normalization을 fp32에서 계산해 replay한다. Fp32에서만 사라지는 spike는 numerical cause를, fp32에서도 유지되는 spike는 data/optimization cause를 지지한다.
5. Mechanism-specific metric, 즉 maximum attention logit, output softmax mean \(\log Z\), \(\epsilon\) 대비 gradient RMS를 읽는다([[small-scale-proxies-instabilities]] §3.1, §3.4).
6. OLMo 2처럼 loss/kernel implementation은 comparison run이 있는 restart point에서만 바꾼다(§3.3.3).

## 레시피

| 모델(정확 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 총 671B / active 37B | pretrain-stable | FP8 scope/format | Fprop, Dgrad, Wgrad GEMM FP8; 모든 tensor E4M3 | arXiv:2412.19437v2 §3.3.1-3.3.2 ([[deepseek-v3]]) | verified 2026-09-15 | App. B.1: 약 16B/1.33T 및 약 230B/약 0.9T token에서 BF16 대비 relative loss error <0.25%; downstream 비교 없음 |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | scaling granularity | activation 1×128 tile; weight 128×128 block; online max-abs | §3.3.2 | verified 2026-09-15 | App. B.2: activation gradient 128×128 block은 약 16B, 약 300B token에서 diverge |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | BF16/FP32 유지 | embedding, output head, MoE gating, normalization, attention | §3.3.1 | verified 2026-09-15 | component별 ablation 없음 |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | optimizer/gradient dtype | AdamW moment BF16; master weight FP32; gradient FP32 | §3.3.3 | verified 2026-09-15 | “without incurring observable performance degradation”; 수치 없음 |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | FP32 accumulation interval | \(N_C=128\) element | §3.3.2 | verified 2026-09-15 | \(K=4096\) random matrix: promotion 없으면 max relative error 거의 2%; table 없음 |
| OLMo 2 7B(stage 1 config) | 7B | pretrain-stable | autocast dtype; FSDP precision | amp_bf16; mixed(parameter/buffer bf16, gradient reduction fp32) | github.com/allenai/OLMo@090253d configs/official-1124/OLMo2-7B-stage1.yaml L84-88; olmo/config.py L1339-1344 ([[olmo-2-official-configs]]) | verified 2026-09-15 | ablation 없음 |
| OLMo 2 7B(stage 1 config) | 7B | pretrain-stable | AdamW \(\epsilon\); grad clip | 1e-8; max_grad_norm 1.0 | yaml L48, L90; arXiv:2501.00656v3 §3.4.1 | verified 2026-09-15 | §3.4.1 Fig. 9: \(\epsilon\) 1e-8 대 1e-5는 early gradient norm을 더 낮고 안정적으로; downstream 수치 없음 |
| OLMo 2 7B | 7B | pretrain-stable | z-loss coefficient | report: \(1e-4\cdot\log^2Z\) | arXiv:2501.00656v3 §3.3.3 | conflict | yaml L39의 auxiliary_loss_multiplier 1e-5는 090253d train.py가 전달하지 않고 loss function default 1e-4(L130,L162); report value는 code path와 일치 |
| Wortsman et al. default | 2.4M-1.2B | pretrain-stable | dtype; z-loss; \(\epsilon\) | TPU bfloat16; 1e-4; 1e-8 | arXiv:2309.14322v2 §2.1 ([[small-scale-proxies-instabilities]]) | verified 2026-09-14 | z-loss Fig.3-4; \(\epsilon\)은 다음 행 |
| Wortsman et al. intervention | 4.8B | pretrain-stable | LR 0.3 AdamW \(\epsilon\) | 1e-15 | §3.4, Fig. 12 | verified 2026-09-14 | 1e-8보다 lower loss, 1e-6 diverged; value당 한 run |
| MiniMax-M1 | 총 456B / active 45.9B | RL | LM head precision | FP32 | arXiv:2506.13585v1 §3.2 ([[minimax-m1]]) | verified 2026-09-15 | train–inference probability correlation 0.987319 → 0.997135; reward growth resumed(곡선 수치 없음) |
| MiniMax-M1 | 456B / 45.9B | RL | AdamW \(\beta_1,\beta_2,\epsilon\) | 0.9, 0.95, 1e-15 | §3.2 | verified 2026-09-15 | gradient 1e-18–1e-5(대부분 <1e-14); VeRL default (0.9,0.999,1e-8)는 non-convergence를 “can result in”; ablation table 없음 |
| Qwen2.5-32B DAPO(TIS blog) | 32B | RL | TIS cap \(C\); sampler/learner parallelism | 8; vLLM TP2 / FSDP Ulysses SP8 | github.com/yaof20/verl@fd02f7b recipe/flash_rl (card ledger, [[rollout-training-mismatch-tis]]) | verified 2026-09-14 | Fig.1: 250 step TIS 유/무; \(C\) sweep 없음 |
| DeepSeek-R1-Distill-Qwen-7B 및 다른 7B-8B 3개(eval study) | 7B-8B | eval-gate | runtime config; max output; sampling | L40S/A100 × GPU 2/4 × batch 8/16/32; reasoning 32,768 token/non-reasoning 2,048; \(T=0.7\), top-p 0.95 | arXiv:2506.09501v2 §3.1 ([[numerical-nondeterminism-llm-inference]]) | verified 2026-09-15 | Table 3: BF16 Std@Acc 최대 9.15%, FP32 약 0 |
| PyTorch 2.12.0 GradScaler(framework default, model run 아님) | n/a | pretrain-stable | init scale; backoff; growth; growth interval | \(2^{16}\); 0.5; 2.0; 2000 | torch/amp/grad_scaler.py L123-131(git 7661cd9) ([[pytorch-amp-autocast-gradscaler]]) | verified 2026-09-15 | ablation 없음; Micikevicius et al.은 constant factor 8-32K 사용([[mixed-precision]] §3.2) |

**소규모 범용 실행의 시작점.** 아래 모든 값은 원래 조건이 있는 위 verified 행에서 왔다. OLMo 2 7B stage-1 config(7B dense, FSDP)처럼 computation 중 parameter/buffer bf16, gradient reduction fp32의 bf16 autocast로 학습한다. DeepSeek-V3(671B MoE FP8 framework)가 한 것처럼 master weight/gradient는 fp32에 둔다(§2가 rounding 이유를 준다). 그 config처럼 AdamW \(\epsilon=1e-8\), gradient clip 1.0을 쓰고 gradient RMS를 log한다. RMS가 \(\epsilon\)에 접근하면 Wortsman et al.(TPU bfloat16, C4)은 high LR에서 \(\epsilon=1e-15\)이 더 좋았고, MiniMax-M1은 456B MoE RL에 1e-15를 썼다. Fp16만 가능하면 PyTorch GradScaler default(initial \(2^{16}\), backoff 0.5, 2000 clean step마다 growth 2.0)에서 시작한다. Separate inference engine RL에서는 MiniMax-M1처럼 LM head FP32를 계산하고 truncated correction이면 DAPO-32B는 \(C=8\)을 썼다. Evaluation에는 Yuan et al. 12-configuration design처럼 GPU type, GPU count, batch size, dtype을 기록한다.

## 일반화 관점

**(a) 무엇이 breadth를 보존하거나 넓히는가.**

- Numerical choice는 capability를 추가하지 않지만 update/measurement가 올바른지를 정한다. FP16 weight update가 relative accuracy 80%를 잃은 곳에서 full-precision master weight가 FP32 accuracy를 회복했다([[mixed-precision]] §3.1, Fig. 2a).
- Sampler–learner mismatch 제거는 RL을 계속하게 했다. FP32 LM head fix 뒤 MiniMax-M1 reward가 증가했고([[minimax-m1]] §3.2), bitwise-identical sample/training은 KL 0으로 실행됐다([[thinkingmachines-defeating-nondeterminism]]).
- Lower precision training은 model을 PTQ에 더 robust하게 만들며([[scaling-laws-for-precision]] §5), served model이 quantized일 때 중요하다.

**(b) 무엇이 narrowing, forgetting, false breadth claim을 일으키는가.**

- Over-trained model은 PTQ에서 더 많은 loss를 잃는다. Fitted setting에서 \(\delta_{PTQ}\propto D^{0.5068}\)(\(N\) fixed)다([[scaling-laws-for-precision]] Eq. 2, App. K).
- Correction 없는 INT8 rollout은 DAPO-32B에서 entropy 0.2 아래 및 abnormal long response를 냈다. Output diversity 감소다([[rollout-training-mismatch-tis]] Fig. 5).
- Correction 없이는 logged RL reward가 더 높고 AIME accuracy가 더 낮아 reward curve가 progress를 과장했다(Fig. 6).
- Loss parity만으로 accept한 FP8/precision change는 downstream parity를 측정하지 않는다([[deepseek-v3]] App. B.1).
- Bf16 weight에서는 spacing 절반보다 작은 fine-tuning update가 지워져(§2 worked example), low-learning-rate stage가 effect 없는 듯 보일 수 있다.

**(c) 이 단계에서 어떻게 측정하는가.**

- Training precision change: checkpoint 둘의 ch-00 development/unseen suite 및 domain별 loss를 evaluation configuration 하나에서 측정.
- Quantized deployment: deployed bit width의 held-out loss와 같은 suite에서 \(\delta_{PTQ}\) 측정.
- Evaluation reproducibility: runtime configuration 두 개 이상에서 Std@Acc, divergence rate, greedy 비교의 FP32-computation 또는 batch-invariant rerun 보고([[numerical-nondeterminism-llm-inference]] §3-4).
- RL: response별 sampler–learner KL, max/mean token-probability mismatch, entropy, response length, logged reward와 함께 held-out accuracy([[rollout-training-mismatch-tis]] Figs. 5-11).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| Unscale 전 fp16 gradient clipping | clipping이 매 step 또는 전혀 발생하지 않음; effective threshold가 \(S\)배 틀림 | `clip_grad_norm_` 전 `unscale_` 호출, unscaled norm log([[mixed-precision]] §3.2) |
| Bf16/fp16 weight에 optimizer update 적용 | low LR에서 loss flat; step 뒤 parameter bitwise 불변 | stored value가 바뀐 parameter 수를 세고 fp32 master copy와 비교(§2) |
| Optimizer dtype이 저장 못 하는 \(\epsilon\) 선택 | fp16에서 \(\epsilon=1e-15\)이 0; zero second moment parameter의 inf/NaN update | \(\epsilon\)과 dtype smallest subnormal 절반 비교(§1 table, §8 worked example) |
| Framework 하나의 fp32 op list를 universal로 취급 | “norm은 fp32”라 했지만 fused RMSNorm/attention은 bf16 | autocast 아래 actual op dtype 읽기([[pytorch-amp-autocast-gradscaler]]) |
| Loss curve만 보고 FP8/lower precision accept | loss <0.25%, evaluation comparison 기록 없음 | checkpoint 둘을 development/unseen suite에서 score(§4) |
| Outlier가 있는 tensor당 scale 하나 | FP8 cast 후 nonzero value가 0 | quantization 뒤 tensor/block별 zero fraction(§4 worked example) |
| 측정 없이 high-\(D/N\) checkpoint를 5 bit 이하 quantize | 적은 token checkpoint보다 quantized loss 증가가 큼 | deployed bit width에서 \(\delta_{PTQ}\), suite score 측정(§5) |
| BF16에서 GPU count/type/batch가 다른 greedy score 비교 | rerun에서 difference sign 변화; output length 수천 token 차이 | fixed recorded config + FP32 computation 또는 n-sample average([[numerical-nondeterminism-llm-inference]]) |
| Sampler log-probability를 learner log-probability로 사용 | reward collapse, KL spike, negative k1 KL, rising reward/flat held-out accuracy | 두 log-prob, max/mean mismatch, sampler–learner KL log(§7) |
| Sampler/learner sharding 다름 | max mismatch >0.5 response 증가 | sampler/learner tensor parallelism 일치([[rollout-training-mismatch-tis]] Figs. 7-8) |
| Run 중 loss/kernel implementation 전환 | forward value는 같은데 switch 뒤 curve diverge | comparison run이 있는 restart에서만 switch([[olmo-2]] §3.3.3) |
| 모든 loss spike를 numerics 탓으로 함 | repeated sequence batch에서 spike, data order에 따라 이동 | spike batch inspect, fp32 step replay(§8) |

## 이해 점검

1. Bf16과 fp16은 모두 16 bit다. 왜 fp16 training에는 loss scaling이 필요하고 bf16에는 필요 없는지, bf16도 fp32 master weight를 유지하는 이유를 설명하라.
2. Learning rate \(1e-5\) fine-tuning run이 어떤 evaluation score도 바꾸지 않는다. Bf16 spacing으로 numerical cause와 이를 confirm/rule-out할 measurement를 설명하라.
3. DeepSeek-V3는 FP8 training loss가 BF16의 0.25% 이내라고 보고한다. 이것이 지지하는 general capability claim과 지지하지 않는 claim을 설명하고 나머지를 시험할 comparison을 설계하라.
4. §4 worked example에서 1000 outlier 하나가 −0.001을 0으로 저장하게 하는 mechanism과, FP8-format 논문이 gradient E5M2를 권하는데 DeepSeek-V3가 모든 tensor에 E4M3를 쓸 수 있는 이유를 설명하라.
5. 같은 checkpoint greedy decoding이 2/4 GPU에서 AIME'24 accuracy가 다른 이유, long reasoning output에서 더 큰 이유, configuration 하나를 고정해도 문제를 제거하지 못하는 이유를 설명하라.
6. RL run이 rising reward, flat held-out accuracy, falling entropy를 보인다. §7의 numerical cause 두 개, 구분 metric, 각 fix가 gradient estimator에 바꾸는 것을 제시하라.
7. [[scaling-laws-for-precision]] PTQ degradation이 \(D/N\)과 함께 커지는 이유와 deployed 4-bit model의 training token budget 선택 함의를 설명하라.
8. Bf16 run step 40,000에서 loss spike가 났다. Data/numerical cause를 나누는 check sequence와 각 step result가 conclusion을 어떻게 바꾸는지 설명하라.

## 연결

- **이전:** ch-01 — LLM Training Optimizer: AdamW, Update Size, Prior Ability Retention. §2/§8의 AdamW update와 \(\epsilon\)을 정의한다.
- **다음:** ch-03 — Learning-Rate Schedule, Batch Size, Initialization, Normalization. §8 QK-norm/normalization placement의 stability evidence를 다룬다.
- **한 절을 확장하는 장:** §1-§3/§8 → ch-07 — Training Failure Mode: Numerical, Masking, Capability-Level Failure; §2 gradient-reduction dtype → ch-05 — Batch Size, Sequence Length, Seen Token을 바꾸는 Distributed Training Choice; §5 → ch-08a — Pretraining Loss에서 Downstream Capability까지 Scaling Law/Compute Allocation; §6 → ch-47 — General Capability 평가 Harness/Suite 설계 및 ch-51 — Metric Noise, Confidence Interval, Go/No-Go; §7 → ch-54 — Off-Policy Data, Asynchronous RL, Agentic Environment의 Rollout Infrastructure, ch-43 — RL Entropy/Output Diversity/KL Control, ch-43a — Likelihood Displacement/Squeezing/Negative Advantage, ch-44a — RL의 Length: Overlong Response, Length Control, Long-Context RL.
- **측정 계약:** ch-00 — General Capability의 의미와 측정(development/unseen suite, §6 measurement error).

## 출처

- [[mixed-precision]] — FP32 master weight, loss scaling, FP32 reduction, FP16 result(arXiv:1710.03740v3 §3-4). Card는 아직 re-verify되지 않았고 value는 primary text 및 같은 이름 chapter excerpt에서 읽음.
- [[fp8-formats-for-deep-learning]] — E4M3/E5M2 limit(Table 1), scaling/overflow handling(§2), simulated FP8 training(Table 4). Chapter excerpt.
- [[pytorch-amp-autocast-gradscaler]] — PyTorch v2.12.0 CUDA autocast op list 및 GradScaler default. Chapter excerpt.
- [[deepseek-v3]] — FP8 framework, fine-grained quantization, accumulation precision, BF16 optimizer moment, FP8-BF16 loss ablation(§3.3, App. B).
- [[deepseek-v3-recipe]] — DeepSeek-V3 model size/token count/SFT LR schedule.
- [[scaling-laws-for-precision]] — \(D/N\) 대비 PTQ degradation, effective parameter count, compute-optimal precision, stated limit(arXiv:2411.04330v2). Chapter excerpt.
- [[numerical-nondeterminism-llm-inference]] — configuration-dependent greedy/sampled score, divergence rate, LayerCast(arXiv:2506.09501v2). Chapter excerpt.
- [[thinkingmachines-defeating-nondeterminism]] — inference nondeterminism 원인인 batch invariance, completion count, kernel cost, bitwise on-policy RL. Chapter excerpt; planned library card는 2026-09-15에 없었음.
- [[rollout-training-mismatch-tis]] — sampler–learner mismatch 측정, parallelism/length factor, INT8 effect, truncated importance sampling.
- [[minimax-m1]] — train–inference mismatch source인 LM-head precision, FP32 fix, RL AdamW \(\epsilon\)(arXiv:2506.13585v1 §3.2). Chapter excerpt.
- [[small-scale-proxies-instabilities]] — attention logit growth, output logit divergence, z-loss, \(\epsilon\) 대 gradient RMS.
- [[olmo-2]] — data-caused spike/repeated n-gram, spike score, z-loss implementation divergence, \(\epsilon\) change(arXiv:2501.00656v3 §3; card 아직 verified 안 됨, locus는 primary text에서 읽음).
- [[olmo-2-official-configs]] — release config/code commit 090253d의 OLMo 2 7B stage-1 precision/FSDP/\(\epsilon\)/z-loss setting. Chapter excerpt.
