<!-- chapter: ch-07
     track: foundations
     kind: content
     title: Training Failure Modes: Numerical, Masking, and Capability-Level Failures
     deps: [ch-06]
     sources: [[small-scale-proxies-instabilities]], [[gradient-clipping]], [[sequence-packing]], [[packing-position-id-evidence]], [[hf-gradient-accumulation-fix]], [[tulu-3]], [[llama-3]], [[data-constrained-scaling]], [[catastrophic-forgetting-continual-finetuning]], [[mitigating-alignment-tax-rlhf]], [[sft-memorizes-rl-generalizes]], [[entropy-mechanism-llm-rl]], [[reward-model-overoptimization]], [[reward-hacking-taxonomy]], [[swe-bench-illusion]], [[paloma]]
     figures: figures/failure-modes-tree.html
     revised: 2026-09 (generality revision)
-->

# 7장 — 학습 실패 모드: 수치·마스킹·능력 수준 실패

> **핵심 통찰.** 학습 실패는 서로 다른 도구가 필요한 세 부류로 나뉜다. 수치 실패(NaN, Inf, 발산)는 loss와 gradient log에서 보이고 작은 scale에서 재현할 수 있다. Attention-logit growth와 output-logit divergence는 높은 learning rate에서 2.4M~1.2B parameter model에 나타났으며, small model의 quadratic fit은 4.8B 발산을 예측했고 4.8B run이 이를 확인했다([[small-scale-proxies-instabilities]], §3.1, §3.3). Masking, packing, loss-aggregation bug는 loss가 유한한지와 관계없이 loss를 계산하는 대상을 바꾸므로 loss curve에는 보이지 않는다. Position ID와 step count를 유지했을 때 padding 대 packing validation loss 1.129 대 1.127 같은 reference batching과의 equality test로 잡는다([[packing-position-id-evidence]] item 2, Table 2). Capability-level failure—반복 data 과적합, forgetting, contamination, template narrowing, entropy collapse—는 두 log 모두에서 보이지 않으며 held-out capability 측정에서만 관측된다. BLOOMZ-7.1b는 다섯 continual instruction task에서 target task는 모두 개선되는 동안 MMLU score의 18.37%를 잃었다([[catastrophic-forgetting-continual-finetuning]], Tables 3–4).
>
> **지침.** Run에서 non-finite 또는 diverging loss가 나면 attention logit → output logit → optimizer state 순서로 점검한다. 각 stage에는 공개 근거가 있는 별도 완화책이 있기 때문이다(attention logit에는 qk-layernorm, output logit에는 z-loss 1e-4, update collapse에는 더 작은 AdamW ε; [[small-scale-proxies-instabilities]] §3.1, §3.4). Numerically healthy하지만 reference보다 성능이 낮으면 hyperparameter를 조정하기 전에 batching equivalence를 시험한다. 같은 data와 같은 global batch는 gradient accumulation 1과 4에서 같은 loss를 내야 하며([[tulu-3]] §4.3.2, [[hf-gradient-accumulation-fix]]), position ID를 reset했을 때 packed와 unpacked가 일치해야 한다. Checkpoint가 target task를 개선해도 training curve 자체가 아니라 held-out capability suite와 이전 checkpoint에 대해 판단한다. Target-task 개선과 general-capability 손실은 함께 일어날 수 있기 때문이다([[catastrophic-forgetting-continual-finetuning]] Tables 3–4; [[mitigating-alignment-tax-rlhf]] §4).

---

## 일반 목적 모델에서 이 장이 필요한 이유

1장에서 6장까지는 기계적으로 올바른 training step을 구성했다. Optimizer(ch-01), precision policy(ch-02), schedule(ch-03), packing과 masking(ch-04), distributed layout(ch-05), checkpoint/evaluation loop(ch-06)다. 이 장은 그 step이 일반 능력을 지닌 model을 더 이상 만들지 못하게 되는 방식과, 각 방식을 탐지할 도구를 다룬다.

세 부류는 pipeline의 서로 다른 지점에서 중요하다. Numerical failure는 run이 길고 발산의 wall-clock 비용이 가장 큰 pre-training에서 주로 발생한다. Masking, packing, loss-aggregation failure는 sequence가 짧고 이질적이며 padded이고, token 일부에 대해 loss를 계산하는 SFT에서 주로 발생한다. Capability-level failure는 pre-training 이후의 모든 단계에서 중요하다. Mid-training, SFT, preference optimization, RL은 모두 broad model 위에서 narrow objective를 최적화하므로 objective를 높이면서 breadth를 낮출 수 있다.

처음에 바로잡을 믿음이 하나 있다. Large run은 느린 drift로만 실패하지 않는다. Llama 3 405B pre-training의 54일 snapshot에는 job interruption이 466건 있었고, 그중 419건은 unexpected였으며, unexpected interruption의 약 78%는 confirmed 또는 suspected hardware issue로 귀속되었다([[llama-3]] §3.3.4, Table 5). 이 failure들은 갑작스럽고 뚜렷하다. 이 장의 failure는 job을 계속 실행되게 남겨 두는 보완적인 집합이다. Crash, straggler, distributed hang의 operational handling은 infrastructure material과 별도 training-memory course에 속한다. 이 장은 detection이 *measurement* 질문인 failure만 다룬다.

아래 diagnostic order의 interactive version은 [figures/failure-modes-tree.html](figures/failure-modes-tree.html)에 있다. Symptom을 고르면 page가 cost 순서의 check, mitigation, 그리고 표시하는 모든 수치의 source locus를 보여 준다.

---

## §1 수치 실패: 진단 순서

Non-finite value 자체는 정보가 충분하지 않다. Diagnostic question은 어느 stage가 다음 stage가 표현할 수 없는 값을 만들었는가다. 구분할 가치가 있는 stage는 세 가지이며 각각 자기 근거를 가진다.

### 1.1 Attention-logit growth

**정의.** Attention logit은 `z_ij = ⟨q_i, k_j⟩ / √d_h`다. `q_i`는 position `i`의 query vector, `k_j`는 position `j`의 key vector, `d_h`는 head dimension이다([[small-scale-proxies-instabilities]] §3.1.1). Attention-logit growth는 학습 중 `max_ij z_ij`가 bounded 상태에 머무르지 않고 상승하는 경우다.

**측정 가능한 문제로서 중요한 이유.** 큰 maximum을 가진 logit row의 `softmax`는 포화된다. Row가 one-hot이 되고 그 row를 통과하는 gradient는 0으로 가며, fp16에서는 큰 값의 `exp`가 overflow한다. Paper는 경계를 경험적으로 보고한다. Maximum attention logit이 약 `1e4`를 넘은 모든 run은 발산했다(§3.3, Fig. 9). 10M model의 maximum attention logit을 값 `κ`로 *강제*하면 `κ = 1e3` 부근에서 loss가 나빠지고, `κ = 1e4`에서는 zero-layer bigram baseline보다 나빠진다(§3.3, Fig. 10). 따라서 logit 값은 발산의 증상이 아니라 발산을 일으키기에 충분한 원인이다.

**성장의 mechanism.** 성장은 query/key 사이 cosine similarity가 높아져서가 아니라 query와 key norm이 커져서 생긴다(Fig. E.1). Softmax가 없는 pointwise attention variant에서도 생긴다(§3.2.5, Fig. E.11).

**완화와 근거.** qk-layernorm은 logit을 계산하기 전에 query와 key에 LayerNorm을 적용한다. qk-layernorm과 z-loss를 사용하면 non-embedding parameter 2.4M~1.2B model이 peak learning rate 3e-4~3e-1에서 낮은 loss까지 학습했으며, LR 0.3의 1.2B model도 포함한다(§3.1.1, Fig. 1). Head별 qk-layernorm은 전체 model dimension에 적용한 qk-layernorm보다 더 좋았다(Fig. E.8). **결과(단일 연구).**

**진단뿐 아니라 예측.** Learning rate별로 step 2e3의 maximum attention logit을 model size의 quadratic으로 fit하면, LR 1e-2에서 다음 scale이 `1e4`를 넘을 것이라고 예측했다. 이어 LR 1e-2로 학습한 4.8B model은 실제 발산했고, fit은 maximum attention logit을 가깝게 extrapolate했다(§3.3, Fig. 9). 이 장이 세 failure 중 이 failure로 시작하는 이유는 small run series가 large run을 시작하기 전에 이를 예측할 수 있기 때문이다.

### 1.2 Output-logit divergence

Output logit `y`에 대해 `p_i = e^{y_i} / Z`이고 `Z = Σ_j e^{y_j}`다. Output-logit divergence는 logit이 log-probability에서 크게 drift하는 경우다. Paper의 run에서는 late training에 매우 음수가 되었다(§3.1.2, Fig. 4). 완화책은 coefficient `1e-4`를 갖는 auxiliary term `log²Z`인 **z-loss**다. Weight decay가 없는 model은 시험한 모든 scale에서 divergence가 발생했고, z-loss가 이를 해결했다. Weight decay도 시험한 더 큰 model에서는 완화했다(§3.1.2, Fig. 3). **결과(단일 연구).**

### 1.3 Optimizer-state stage: AdamW ε

Learning rate를 곱하기 전 AdamW update는 `Δ = v / (√u + ε)`다. `v`와 `u`는 first/second gradient moment의 exponential moving average이고, `ε`는 numerical floor다. Model size와 learning rate가 커질수록 첫 MLP layer의 gradient RMS가 낮아지며, 가장 큰 시험 scale/LR에서 기본값 `ε = 1e-8`과 같은 order가 된다(§3.4, Figs. 11, 13). 그러면 `Δ`가 0을 향해 수축한다. Run은 NaN이 나는 것이 아니라 움직임을 멈춘다. LR 0.3의 4.8B model에서 `ε = 1e-15`는 loss를 개선하고 gradient RMS collapse를 없앴지만, `ε = 1e-6`은 발산했다(§3.4, Fig. 12, Fig. E.15).

Practical consequence는 흔한 습관의 교정이다. `ε`을 높이는 것은 때때로 NaN fix로 설명되지만, 이 연구에서 측정한 failure는 반대 방향이다. Gradient scale에 비해 `ε`이 너무 *크면* update가 collapse한다. `ε`을 바꾸고 loss를 보는 대신 layer별 gradient RMS를 `ε`과 함께 log하여 진단한다.

### 1.4 0의 logarithm

`torch.log(torch.softmax(x))`로 계산한 cross-entropy는 underflow된 zero의 log를 취할 수 있다. Fused `log_softmax`와 `F.cross_entropy` path는 내부에서 row maximum을 빼므로 그렇지 않다. Cross-entropy를 손으로 작성할 때는 `torch.log`와 `torch.softmax`를 합성하지 말고 `F.cross_entropy` 또는 `log_softmax`를 쓴다. 이는 coding rule이며 empirical result가 아니고, 여기에는 수치가 붙지 않는다.

느슨한 형태의 주장이 퍼져 있으므로, 관련된 두 주장을 정확히 적을 가치가 있다.

1. `torch.softmax`는 이미 max-subtracted form을 구현한다. 그러므로 이를 호출하는 attention implementation에는 별도 `amax` subtraction이 필요 없다. Hand-rolled hazard는 raw logit을 직접 exponentiate하는 code에서만 실제다.
2. RL에서 쓰는 k3 KL estimator는 **sampled token의** log-probability를 계산하며, 이는 `log_softmax`에서 오므로 유한하다. Sampled token은 그것을 sample한 policy 아래 probability가 정확히 0일 수 없다. 이 estimator의 실제 hazard는 큰 log-ratio의 exponential이 low precision에서 overflow하는 것이며, 이는 `log(0)` 문제가 아니라 range 문제다.

### 1.5 Advantage normalization의 zero variance: worked example

Group-normalized RL은 standard deviation으로 나눈다. `A ← (A − mean) / (std + ε)`. Binary reward를 갖는 한 prompt의 rollout 8개 group을 생각하자.

*Mixed group*: reward `[1, 1, 0, 0, 0, 0, 0, 0]`. Mean `= 0.25`, variance `= 0.25 − 0.25² = 0.1875`, `std = 0.4330`다. `ε = 1e-6`일 때 correct rollout은 `A = (1 − 0.25)/0.4330 = +1.732`, incorrect rollout은 `A = (0 − 0.25)/0.4330 = −0.577`을 받는다. 두 sign이 모두 존재하므로 group은 gradient에 기여한다.

*Uniform group*: reward `[1, 1, 1, 1, 1, 1, 1, 1]`. Mean `= 1`, `std = 0`다. 모든 numerator는 정확히 `1 − 1 = 0`이므로, 모든 rollout에서 `A = 0 / (0 + 1e-6) = 0`이다. 어떤 `ε > 0`에 대해서도 결과는 **zero advantage이며 NaN이 아니다**. 이 group은 numerical failure가 아니라 positive/negative 어느 learning signal도 전혀 운반하지 않는 group이다. NaN은 `ε = 0`이거나 arithmetic이 fp16에서 underflow할 때만 나타난다.

이 구분은 완화책이 다르므로 중요하다. NaN에는 numerical guard가 필요하다. Dead group에는 *sampling* 변화가 필요하다. Entropy study는 update 전에 response가 모두 correct이거나 모두 incorrect인 prompt를 drop한다([[entropy-mechanism-llm-rl]] §2.2, Recipe ledger). Step마다 `std = 0`인 group의 fraction을 log한다. 이 fraction이 늘면 prompt set이 current policy에 비해 너무 쉽거나 너무 어려워졌다는 뜻이며, 이는 arithmetic signal이 아니라 curriculum signal이다.

---

## §2 Masking, packing, loss-aggregation의 정확성

이 bug들은 loss가 finite하고 smooth하게 남으므로 numerical alarm이 울리지 않는다는 공통점이 있다. Reference computation과의 equality test로 탐지한다.

### 2.1 Label shift와 실제로 일어나는 off-by-one

Causal-LM loss는 logit position `i`를 label index `i + 1`과 짝짓는다. `p*`를 prompt token, `r*`를 response token이라 하고, `input_ids = [p0, p1, p2, r0, r1, r2]`, `prompt_len = 3`을 취하자. Shift된 pair는 다음과 같다.

| logit position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| target label index | 1 | 2 | 3 | 4 | 5 |
| target token | `p1` | `p2` | `r0` | `r1` | `r2` |

올바른 masking은 **shift하지 않은** label에 `-100`을 적용한다.

```python
labels = input_ids.clone()
labels[:, :prompt_len] = -100          # masks label indices 0, 1, 2 -> p0, p1, p2
loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V),
                       labels[:, 1:].reshape(-1),
                       ignore_index=-100)
```

이는 *target*이 prompt token인 pair를 버리고 `p2 → r0`, `r0 → r1`, `r1 → r2` 세 trained pair를 남긴다. 마지막 prompt token에서 첫 response token을 예측하는 것은 leak이 아니라 inference에서 model이 해야 하는 transition이므로 올바른 학습의 일부다.

실제 발생하는 bug는 **shift 뒤에** masking하는 것이다.

```python
labels_shift = labels[:, 1:].clone()      # [p1, p2, r0, r1, r2]
labels_shift[:, :prompt_len] = -100       # masks p1, p2, r0  <- removes the first response token
```

효과는 흔한 설명의 반대다. 첫 response token이 loss에서 제거되므로 model은 prompt에서 response로 들어가는 법을 학습하지 않는다. Chat template에서 첫 response token이 role 또는 format marker라면, 이 token의 누락은 나중에 malformed turn start로 나타난다.

**탐지.** 싸고 정확한 check 두 개다.

```python
active = (labels[:, 1:] != -100).sum(dim=-1)          # per sample
assert (active == response_len).all()                  # off-by-one shows up as response_len - 1
print(tok.decode(input_ids[0][labels[0] != -100]))     # must be exactly the assistant text
```

주의할 두 번째 variant는 2-D tensor에 `labels[:prompt_len] = -100`을 쓰는 것이다. Batched tensor에서 이는 **row**를 index하므로 처음 `prompt_len` sample 전체를 mask하고, 나머지 sample의 모든 prompt는 loss에 남긴다. 위 `active` assertion은 masked row가 active token 0을 보고하므로 즉시 잡는다.

### 2.2 Packing: mask가 하는 일과 position ID가 실제로 하는 일

Packing은 여러 sample을 하나의 sequence로 concatenate한다. Correctness에는 block-diagonal attention mask가 필요하며, 보통 variable-length kernel에 cumulative sequence length(`flash_attn_varlen_func(..., cu_seqlens_q, cu_seqlens_k, max_seqlen)`)로 공급한다. 이것이 없으면 sample 2 token의 softmax denominator에 sample 1 token에 대한 score가 포함되고, 두 sample의 gradient가 바뀐다([[sequence-packing]] §3.2.1).

각 adjustment의 근거는 분리되어 있다([[packing-position-id-evidence]] item 1, Krell et al. §4.2.1, Fig. 4, BERT pre-training).

- **Mask** adjustment가 없으면 loss와 accuracy가 크게 악화되고, 더 오래 학습해도 회복되지 않는다.
- **Position** adjustment가 없으면 loss와 accuracy는 baseline에 거의 맞지만 MLM accuracy는 target 72.1%에 대해 71.8%에서 멈춘다.

이 비대칭과 setting을 유의하라. BERT는 learned absolute positional embedding을 쓰므로 reset하지 않은 position index는 잘못된 embedding vector를 lookup한다. **RoPE** decoder의 올바른 risk model은 다르다. 이는 이 장의 이전 version을 포함한 여러 곳의 주장에 대한 correction이다.

> RoPE attention score는 relative offset `i − j`에 의존한다. 올바른 block-diagonal mask에서는 attended pair가 모두 같은 document 안에 있고, document가 pack position 0에서 시작하든 pack position `L₁`에서 시작하든 document 안 offset은 변하지 않는다. 따라서 reset하지 않은 position은 intra-document relative-position math를 왜곡하지 **않는다**.

Reset하지 않은 position이 깨뜨리는 것은, 중요한 빈도 순서로 다음과 같다.

1. **Boundary derivation.** `position_ids`에서 *cu_seq_len*을 재구성하는 implementation—`DataCollatorWithFlattening`의 documented mechanism([[packing-position-id-evidence]] item 2, §3.3)—은 position이 reset되지 않으면 boundary를 전혀 찾지 못한다. 따라서 block-diagonal mask가 조용히 dense causal mask가 된다.
2. **Trained-range overflow.** Pack length가 model이 학습한 position range를 넘으면 reset하지 않은 position은 token을 model이 본 적 없는 index에 둔다. Pack을 trained context length 이하로 만들면 이는 발생하지 않는다.
3. 위 BERT 결과 같은 **learned absolute position embedding**.

Decoder에서 이것을 잘못했을 때의 측정 비용은 다음과 같다([[packing-position-id-evidence]] item 2, FLAN_20k에서 Mistral-7B, Table 2). Padding은 validation loss 1.129에서 742 tokens/s, position ID 없는 packing은 1.306에서 2986 tokens/s, position ID 있는 offline packing은 1.284에서 3010 tokens/s, position ID 있는 online minibatch packing은 1.127에서 1408 tokens/s였다. 저자는 offline packing의 잔여 gap을 attention contamination이 아니라 한 epoch의 optimizer step이 훨씬 적은 탓으로 돌렸다. Minibatch packing은 step count를 유지하고 padding loss와 일치한다(§4.1, §4.3).

**조건과 한계.** 세 번째 study는 packed conversation 사이 attention을 reset한다고 명시하지 않은 채 LLaMA-3-8B/70B에서 padding, random packing, greedy packing을 비교했으며, greedy-packing average가 model-dataset setting 8개 모두에서 padding보다 높다고 보고했다(Table 3; [[packing-position-id-evidence]] item 3). 그러므로 decoder SFT의 cross-contamination effect 크기는 **열린 질문**이다. Mask가 correctness에 필요하다는 논증은 그렇지 않다.

**주어진 implementation에서 이를 판정하는 equality test:**

```python
out_packed   = model(input_ids=packed, position_ids=pos, cu_seqlens=cu).logits
out_unpacked = torch.cat([model(input_ids=s).logits for s in split_by_cu(packed, cu)], dim=1)
assert (out_packed - out_unpacked).abs().max() < 1e-4
```

### 2.3 Gradient accumulation과 data parallelism 아래 loss aggregation

Non-padding token에 대한 mean loss는 batch를 어떻게 split하는지에 불변이 아니다. Tülu 3는 두 경우를 직접 제시한다([[tulu-3]] §4.3.2, Eqs. 1–2). Non-padding token 수 `n₁`, `n₂`와 summed token loss `l₁`, `l₂`인 sample 두 개에서:

- One forward pass: `L = (l₁ + l₂) / (n₁ + n₂)` — 모든 **token**을 동일하게 weight한다.
- Accumulate한 micro-batch 둘: `L = (l₁/n₁ + l₂/n₂) / 2` — 모든 **sample**을 동일하게 weight한다.

*Worked example.* `n₁ = 10`, `l₁ = 20`; `n₂ = 90`, `l₂ = 90`이다. Token-weighted는 `110 / 100 = 1.10`, sample-weighted는 `(2.0 + 1.0) / 2 = 1.50`이다. 같은 data와 같은 global batch인데도 loss가 36% 다르고 short sample의 effective weighting도 다르다. Cross-device averaging이 per-rank mean을 average하므로 data-parallel rank 사이에서도 같은 effect가 나타난다.

Hugging Face는 2024년 10월 해당 fix를 `transformers`에 배포했다([[hf-gradient-accumulation-fix]]). “the correct loss should be computed by the total loss across all batches in a gradient accumulation step divided by the total number of all non padding tokens in those batches. This is not the same as the average of the per-batch loss values.” Patch는 causal-LM default loss를 `reduction="sum"` 뒤 `num_items`로 나누도록 바꾼다.

Tülu 3 자체의 대응은 **sum loss**로 학습하고 learning rate를 다시 tuning하는 것이었다. Tülu 2 mixture에서 Llama 3.0을 fine-tune할 때 LR 5e-6의 sum loss가 최상이었고, 2 epoch가 3~7 epoch보다 좋았다([[tulu-3]] §4.3.2, Figs. 5–6). 이는 공식 framework fix와 독립 model report에서 **복제됨**이다.

**탐지.** 같은 global batch와 같은 data order로 `grad_accum = 1` 및 `grad_accum = 4`에서 50 step을 실행한다. Loss curve는 floating-point tolerance 안에서 일치해야 한다. 일치하지 않으면 어딘가에서 aggregation이 sample-weighted다.

---

## §3 발산, spike, plateau

발산, spike, plateau는 원인이 서로 다른 세 가지 loss shape이며, 각각은 서로 다른 logged quantity로 구분한다.

**높은 learning rate에서의 발산**은 small-proxy study가 측정하는 shape다. Summary statistic은 **LR sensitivity**다. `E_{η∈[a,b]}[min(ℓ(A(η)), ℓ₀) − ℓ*]`이며, `η`는 warmup-plus-cosine schedule의 peak LR, `A(η)`는 `η`로 학습한 뒤 weight, `ℓ`은 validation loss, `ℓ₀`은 initialization의 loss, `ℓ*`는 sweep range에서의 best loss다(default `[3e-4, 3e-1]`, [[small-scale-proxies-instabilities]] §2.2). 더 긴 warmup은 LR sensitivity와 loss를 낮췄으며, LR 3e-1에서 long warmup 없이는 stable하지 않았던 큰 model에서 가장 큰 효과가 있었다(§3.2.1, Fig. 5). `λ = 1e-4`의 independent weight decay는 `λ = 0.1`의 coupled PyTorch/Optax form보다 LR sensitivity가 낮았다(§3.2.2, Fig. 6). 이 metric에는 문서화된 한계가 있다. Optimal LR의 shift는 고려하지 않으며 loss scale에 불변이므로, 모든 LR에서 random performance인 model은 score 0을 받는다(App. B).

**Spike**—한두 step의 jump—는 다른 현상이며, 이 course의 verified source는 folklore가 시사하는 것보다 이 부분에서 얇다. 다음 두 사실을 유지하라.

- Small-proxy study는 fast loss spike가 아니라 slow divergence를 일으키는 instability만을 명시적으로 범위에 둔다(footnote 1, §4).
- 15.6T token에서 Llama 3 405B는 spike-intervention stack의 반대를 보고한다. “We found this training recipe to be very stable: we observed few loss spikes and did not require interventions to correct for model training divergence”([[llama-3]] §3.4.1). 명시한 stability measure는 batch-size ramp다. Sequence 4,096에서 4M token으로 시작해 252M token 뒤 sequence 8,192에서 8M으로 두 배, 2.87T token 뒤 16M으로 다시 두 배로 늘린다. 이와 함께 peak LR 8e-5와 8,000 warmup step을 쓴다.

Library에서 수치를 제공하는 유일한 spike-adjacent ablation은 optimizer setting이다. Data-constrained study에서 `β₂ = 0.95`는 FLOP budget 둘에서 `0.999`보다 final loss가 약간 낮고 loss spike도 더 적었다([[data-constrained-scaling]] App. S). 이 source들에는 controlled ablation이 없으므로 skip-step heuristic은 engineering practice로 취급하고 사용할 때 그 사실을 밝힌다.

**Plateau**는 같은 세 logged scalar로 구분하는 세 원인이 있다.

| 관측 | `lr` | `clipped_fraction` | batch당 active token | 원인 |
|---|---|---|---|---|
| flat loss | 0 | any | normal | schedule 소진 또는 잘못된 step에서 resume(ch-06) |
| flat loss | normal | ≈ 1.0 | normal | natural gradient scale보다 낮은 clip threshold |
| flat loss | normal | ≈ 0 | ≈ 0 | masking 또는 data bug: 학습하는 대상이 없음 |
| flat loss | normal | ≈ 0 | normal | update collapse: gradient RMS를 AdamW `ε`와 비교(§1.3) |

Global-norm clipping은 `‖g‖ > threshold`일 때 전체 gradient를 `threshold / ‖g‖`로 rescale한다. 이는 direction을 보존하고 step size를 bound한다([[gradient-clipping]]). Threshold가 run의 통상 gradient norm보다 훨씬 낮으면 모든 step은 gradient direction의 fixed-length step이 된다. 이를 식별하는 quantity는 loss 자체가 아니라 1.0에 고정된 `clipped_fraction`이다. 이 장 source의 verified pre-training recipe 둘 모두 global-norm 1.0을 사용한다([[small-scale-proxies-instabilities]] §2.1; [[data-constrained-scaling]] App. S).

---

## §4 능력 수준 실패

이들은 이 장의 개정을 이끈 failure다. 어느 것도 abnormal loss curve를 만들지 않는다. 모두 training objective가 개선되는 동안 breadth를 줄인다.

### 4.1 반복 data에서의 과적합

**측정 가능한 문제.** Data 반복은 held-out fit보다 training-set fit을 더 빨리 높인다. 따라서 train-loss만 보는 관점은 개선을 보고하지만 model은 더 나빠진다.

**Pre-training scale의 근거.** Data-constrained study(GPT-2 architecture, C4 subset, 9B parameter와 900B token까지 400여 run)에서, 최대 **4 epoch** 반복 data 학습은 held-out loss를 거의 바꾸지 않는다. 4 epoch의 8.7B model은 1 epoch 대비 validation loss가 0.5% 더 높게 끝나며, 19 task downstream performance는 약 4 epoch 뒤 하락하기 시작한다([[data-constrained-scaling]] Abstract, §6, App. L, Fig. 6). Fitted effective-data formula는 `D′ = U_D + U_D · R*_D · (1 − e^{−R_D/R*_D})`이고 `R*_D = 15.387756`이다. `U_D`는 unique-token budget, `R_D = D/U_D − 1`은 repetition 수다.

*Worked example.* 4 epoch는 `R_D = 3`이므로 `D′ = U_D(1 + 15.39 × (1 − e^{−3/15.39})) = 3.73 U_D`다. `D = 4 U_D`에 비해 repeated token은 unique token의 93% 가치다. 16 epoch(`R_D = 15`)에서는 `D′ = 10.58 U_D`, 즉 66%다. 이 study는 model이 repeated data에 overfit하기 때문에 training loss가 아닌 held-out test loss를 쓴다고도 명시한다(App. H, Fig. 14).

**SFT scale의 근거.** Tülu 2 mixture에서 Llama 3.0을 fine-tune하면 evaluation average에서 2 epoch가 3~7 epoch를 이겼다([[tulu-3]] §4.3.2, Fig. 6). 각 경우 **결과(단일 연구)**지만 direction은 scale 전반에서 일치한다.

**Detector.** 같은 distribution data의 held-out loss와 broad benchmark average를 함께 쓴다. Training loss만으로는 이를 볼 수 없다.

### 4.2 Continual fine-tuning 중 catastrophic forgetting

**정의와 metric.** Forgetting은 새 task를 배우는 동안 이전에 획득한 ability를 잃는 것이다. Forgetting metric은 average *relative* decrease다. `FG_i = (1/|E_i|) Σ_e (1/N) Σ_m (R_e^0 − R_e^m)/R_e^0 × 100%`. `E_i`는 evaluation set 하나, `R_e^m`은 계속 학습한 task `m`개 뒤 score, `R_e^0`은 initial model score다([[catastrophic-forgetting-continual-finetuning]] Eq. 1).

**근거.** 고정된 순서의 task 다섯 개에서 BLOOMZ를 continual instruction-tuning하고, MMLU, commonsense reasoning, RACE를 general suite로 쓴 결과(Table 4)는 다음과 같다.

| Model | Domain knowledge FG | Reasoning FG | Reading comprehension FG |
|---|---|---|---|
| BLOOMZ-1.1b | 9.54% | 6.73% | 18.04% |
| BLOOMZ-1.7b | 10.72% | 6.48% | 24.29% |
| BLOOMZ-3b | 14.63% | 11.09% | 27.56% |
| BLOOMZ-7.1b | 18.37% | 13.62% | 26.75% |

시험한 1.1b–7.1b 범위에서 forgetting은 scale과 함께 늘었다. 저자는 큰 model의 더 높은 시작 score가 부분 원인이라고 돌린다. 한편 target task는 개선됐다(BLOOMZ-7.1b explanation task 51.47 → 68.71, Table 3). 두 사실은 같은 run에서 왔기에 이 pairing은 정보가 있다. Target-task 개선은 retained breadth의 근거가 아니다.

같은 paper의 mitigation 하나는 다음과 같다. LLAMA-7b는 세 set에서 FG 34.57 / 31.33 / 31.72를 보였지만, general instruction tuning 뒤 같은 base model인 ALPACA-7b는 18.14 / 7.56 / 10.31을 보였다(Table 6). **결과(단일 연구).**

### 4.3 Alignment tax

같은 failure의 RLHF version에는 고유 이름이 있다. ShareGPT로 instruction-tuning한 뒤 rejection-sampling fine-tuning, PPO, DPO로 align한 OpenLLaMA-3B에서 reward는 오르는 반면 WMT14 French→English BLEU와 SQuAD/DROP F1은 지속적으로 떨어지고, commonsense QA는 오른 뒤 떨어진다([[mitigating-alignment-tax-rlhf]] §4, App. E.1, Fig. 12).

시험한 모든 mitigation은 tax **와** reward를 함께 줄였다. Early stopping, pre-RLHF weight를 향한 L1/L2 regularization, LoRA, pre-RLHF policy의 distillation이다(§4.1, Fig. 3). Alignment-forgetting Pareto front가 가장 좋았던 method는 `α ∈ [0, 1]`에서 `π_{(1−α)θ₀ + αθ}`인 weight interpolation이었다. Heterogeneous Model Averaging—block별 별도 ratio, default `K = 3`—은 이 front를 더 밀었다(§6, Fig. 5). `K`가 6과 9로 늘면 reported curve는 나빠지며, 저자는 이를 in-domain reward 과적합으로 돌린다.

전이 가능한 lesson은 axis choice다. Checkpoint 또는 `α`마다 한 point를 놓고 general-capability metric을 alignment metric **대비** plot할 때만 tax가 보인다. Training step 대비 plot하면 각 curve가 progress처럼 보인다.

### 4.4 겉보기 generality를 부풀리는 contamination

Benchmark score가 오르는 이유는 model이 더 좋아져서일 수도, benchmark가 training data에 있어서일 수도 있다. 가장 깔끔한 최근 demonstration은 다음이다. Repository content 없이 repository name과 issue text만 주었을 때 OpenAI/Anthropic model 10개는 SWE-Bench Verified instance의 60–76%에서 gold patch가 바꾼 file을 이름 댔다. 하지만 SWE-Bench에 없는 repository 7개의 SWE-Bench-style task 245개에서는 53% 미만이었다. Function-reproduction 5-gram overlap maximum은 SWE-Bench Verified에서 34.9%였고, outside repository에서는 13.9%, RefactorBench와 SWE-Bench Extra에서는 18.1–18.2%였다. 후자는 여전히 SWE-Bench repository를 쓰므로 clean held-out set이 아니다([[swe-bench-illusion]] §4.1, §4.3). **결과(단일 연구).**

Detector는 일반화된다. Benchmark builder가 publish할 수 없었던 slice—다른 repository set, benchmark를 만든 뒤 생성된 issue, re-templated variant—를 hold out하여 비교한다. Ch-14는 pretraining-side decontamination, ch-47은 suite design을 다룬다.

### 4.5 Template과 format narrowing

Narrow input format으로 학습하면 task뿐 아니라 format도 배운다. 측정된 사례 하나는 single-turn-only SFT set(OpenHermes 2.5의 filtered 200K subset)을 packing한 경우다. MATH가 유의하게 하락했지만 sample 20~40개당 multi-turn conversation 하나를 추가하자 정상으로 돌아왔다. 다른 internal 200K single-turn set에서는 하락이 없었다([[packing-position-id-evidence]] item 3, §5.3, Fig. 3). Effect는 data-dependent다. 따라서 detector는 rule이 아니라 evaluation이어야 한다. Training에 나타나지 않는 형식 하나를 포함해 적어도 두 prompt format에서 같은 capability를 평가한다.

---

## §5 출력 다양성 손실로서의 RL 실패

Verifiable reward 또는 learned reward에 대한 RL은 training objective를 올리면서 model이 만들 수 있는 출력을 좁힐 수 있다. 두 mechanism에는 quantitative evidence가 있다.

### 5.1 Entropy collapse

**정의.** Policy entropy는 `H(π_θ, D) = −E[log π_θ(y_t | y_<t)]`이며, training prompt response token 전반에 average한다([[entropy-mechanism-llm-rl]] §2.1, Eq. 5).

**측정된 법칙.** Base model 11개, family 4개(0.5B–32B), math/code task와 RL algorithm 4개 전반에서 validation accuracy `R`과 entropy `H`는 `R = −a·exp(H) + b`에 fit한다(§2.2–2.4). 직접 보고된 결과 둘은 다음과 같다. `H = 0`에서 accuracy는 `−a + b`로 bound되며, **entropy consumption의 73%와 performance gain의 76%가 2400 gradient step 중 첫 200 step에서 발생한다**(§2.3–2.4). 첫 36 step에서 `a`, `b`를 fit하면 다음 200 step을 math RMSE 0.9%, code RMSE 1.2%로 예측했다(§2.4, Fig. 5).

**Mechanism.** Softmax policy에서 step별 entropy 변화는 action의 log-probability와 그 logit change 사이 negative covariance에 대략 비례한다. Natural-policy-gradient update에서는 `ΔH ≈ −η·Cov(log π(a|s), A(s,a))`로 줄어든다(§3.2, Theorems 1–2). Covariance는 집중되어 있다. Qwen2.5-7B의 training step 1에서 top 0.02% token의 mean covariance는 5.654였고, 전체 token 평균은 0.003이었다(Table 1).

**수치와 함께, 작동하지 않는 것.** Entropy-loss coefficient 0.0001과 0.001은 영향이 작았고, 0.01은 entropy explosion을 일으켰으며, 0.005는 entropy를 안정화했지만 baseline을 이기지 못했다(§4.1, Fig. 9). 0.001~0.1의 reference-KL coefficient도 entropy는 안정화했지만 accuracy를 낮췄다(§4.1, Fig. 10). **이 study에서 작동한 것:** centered log-probability × advantage product가 가장 큰 token `1e-4`~`1e-3` fraction의 update를 제한했다. Qwen2.5-32B의 benchmark 7개 math average는 GRPO 45.8에서 Clip-Cov 50.3, KL-Cov 52.2가 되었다(Table 2). **결과(단일 연구).**

이 장이 **주장하지 않는** 것을 유의하라. “0.1 nats 아래면 collapse” 같은 공개 entropy threshold는 없다. Paper는 0을 향한 monotone decline과 fitted relation을 보고할 뿐 cut-off는 보고하지 않는다. Threshold가 아니라 trajectory와 fit을 쓴다.

### 5.2 Reward over-optimization

Reward가 learned model이면, 어느 지점을 넘어 이를 최적화하면 true objective가 낮아진다. 6B “gold” reward model이 3M–3B proxy reward model 학습에 쓸 data를 label하고, 1.2B policy가 proxy에 대해 optimize되는 synthetic setup에서 `d = √KL(π‖π_init)`가 커지면 gold score는 올랐다가 내려간다. Best-of-n에는 `R_bon(d) = d(α − β·d)`, RL에는 `R_RL(d) = d(α − β·log d)`가 fit한다([[reward-model-overoptimization]] §1, §3.2, Fig. 1). 미분하면 peak는 `d* = α_bon/(2β_bon)` 및 `d* = exp(α_RL/β_RL − 1)`이다.

같은 study에서 가져갈 결과 둘은 다음이다. RL은 같은 optimization 양에 대해 best-of-n보다 훨씬 많은 KL을 쓰므로 KL은 method를 비교하는 common currency가 아니다(§3.5). Nonzero KL penalty는 early stopping처럼 동작했고 gold-score-versus-KL frontier를 올리지 못했다. 저자 자신도 이를 hyperparameter-sensitive result라고 부른다(§3.6, Fig. 9).

더 좋은 proxy를 찾는 대신 구조적 이유가 있다. 모든 stochastic policy의 class에서는 proxy improvement가 true reward를 결코 낮추지 않는 두 reward function, 즉 “unhackable”한 두 reward function은 degenerate case에서만 가능하다. 본질적으로 하나가 다른 하나의 positive affine transform이거나 하나가 constant일 때다([[reward-hacking-taxonomy]]). 따라서 쓸 수 있는 control은 구조적이다. Policy class를 제한하거나, optimization을 bound하거나, verifier에 reward를 ground한다(ch-44).

### 5.3 Memorization과 generalization을 가르는 detector

Training distribution에서의 held-out data는 rule을 배운 model과 training instance를 memorized한 model을 가르지 못한다. 둘 다 거기서는 score가 좋기 때문이다. Task를 고정한 채 **rule** 또는 **visual attribute**를 바꾸면 구분된다. Llama-3.2-Vision-11B backbone의 GeneralPoints와 V-IRL에서 RL은 OOD success rate를 네 variant 모두에서 높였다(예: V-IRL-L 80.8% → 91.8%). SFT는 네 variant 모두에서 낮췄다(V-IRL-L 80.8% → 1.3%, GP-L 11.5% → 3.4%)([[sft-memorizes-rl-generalizes]] §5.1). Visual variation에서는 RL이 +17.6, +61.1 points를 얻은 반면 SFT는 9.9, 5.6 points를 잃었다(§5.2).

같은 paper가 붙인 두 조건은 이를 “SFT 대신 RL”로 만들지 않는다. Base model은 task instruction을 따를 수 없었으므로 RL이 학습되려면 먼저 SFT가 필요했다(§5.4). 또한 매우 과적합한 SFT checkpoint에서 시작한 RL은 success 1% 아래에 머물렀다(App. D.3).

---

## Negative sample과 negative feedback

이 장은 standard taxonomy의 의미 (4), 즉 sample likelihood를 명시적으로 낮추는 **gradient로서의 negative** 의미로 negative를 쓴다. 여기서 RL failure가 그 형태로 나타나기 때문이다. 의미 (1)~(3)은 data/SFT chapter에 속한다.

**이 stage에서 negative가 오는 곳.** Verifier 또는 reward model이 rollout마다 label을 붙이고, group-normalized advantage가 이를 sample별 sign으로 바꾼다. Label quality가 이 stage의 failure 지점이다. Reward model은 over-optimization을 측정할 수 있는 proxy(§5.2)이며, verifier는 output-format trick으로 game될 수 있다.

**진단에 고유한 failure mode.** Reward variance가 0인 group은 모든 member에 `A = 0`을 만든다(§1.5). 이를 division-by-zero hazard로 설명하곤 하지만, 더 중요한 사실은 **positive/negative 어느 gradient도 공급하지 않는다는 것**이다. Zero-variance group fraction이 늘면 effective batch size가 조용히 줄어든다. Entropy study처럼 response가 모두 correct 또는 모두 incorrect인 prompt를 drop하는 prompt selection이 fix다([[entropy-mechanism-llm-rl]] §2.2).

**Sign asymmetry의 mechanism.** Softmax head에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. 이미 unlikely한 sampled token을 아래로 밀면 그 mass는 현재 가장 likely한 alternative로 옮겨가 distribution이 sharpen된다. Entropy study Eq. 10과 함께 읽으면, below-mean log-probability와 below-mean advantage를 가진 sampled token은 centered product가 **positive**이므로 entropy를 낮춘다. 반면 above-mean-probability token의 negative advantage는 entropy를 높인다(§3.2 및 Eq. 10). Paper가 advantage sign별 result는 보고하지 않았으므로, 그 split은 **열린 질문**이라는 caveat가 있다.

**근거가 있는 control.** Covariance가 가장 큰 token `1e-4`–`1e-3`에서 update를 제한하면 flat entropy bonus와 달리 entropy와 accuracy를 모두 높였다(§4.1, §4.3, Table 2). Gold 또는 held-out score를 `√KL`에 대해 추적하고 peak 근처에서 중단하여 total optimization을 bound하는 것은 reward-model analog다([[reward-model-overoptimization]] §3.2, Fig. 8).

**기록할 diagnostic.** Zero-variance group fraction, update당 entropy, sign별 advantage distribution, 큰 k에서 pass@1과 pass@k를 함께 기록한다. Entropy study는 pass@k를 보고하지 않았으므로 그 intervention의 diversity cost는 그곳에서 측정되지 않았다.

---

## 레시피

위 failure를 예방하거나 탐지하는 데 관련된 값이다. 모든 row는 각 locus에서 인용했으며 model size/stage 사이로 transfer한 row는 없다.

| Model (정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Small-scale-proxies default Transformer (NanoDO, C4) | 2.4M–1.2B | pretrain-stable | gradient clipping | global norm 1 | arXiv:2309.14322v2 §2.1 | verified 2026-09-14 (card) | ablation 보고 없음 |
| same | 2.4M–1.2B | pretrain-stable | `log²Z`의 z-loss coefficient | 1e-4 | §2.1, §3.1.2 | verified 2026-09-14 (card) | Figs. 3–4: z-loss와 weight decay 모두 없으면 output logit diverge |
| same | 2.4M–1.2B | pretrain-stable | qk-layernorm | on, shared parameter의 per head | §2.1, §3.2.5 | verified 2026-09-14 (card) | Figs. 1–2; Fig. E.8(per-head가 whole dimension보다 나음) |
| same | 2.4M–1.2B | pretrain-stable | warmup; total step | 5e3 linear; 1e5 | §2.1 | verified 2026-09-14 (card) | Fig. 5: 긴 warmup은 LR sensitivity와 loss를 낮춤 |
| same | 2.4M–1.2B | pretrain-stable | weight decay | independent, 1e-4(coupled이면 0.1) | §2.1, §3.2.2 | verified 2026-09-14 (card) | Fig. 6: independent form이 더 낮은 LR sensitivity |
| Small-scale-proxies 4.8B intervention run at LR 0.3 | 4.8B | pretrain-stable | AdamW ε | 1e-15 | §3.4, Fig. 12 | verified 2026-09-14 (card) | 1e-8보다 loss 낮음; 1e-6은 발산(Figs. 12, E.15); 값당 run 하나 |
| datablations GPT-2-architecture study model | ≤ 9B | pretrain-stable | Adam β₂ | FLOP budget 9.3e20/2.1e21에서 0.95; 그 외 0.999 | arXiv:2305.16264v5 App. S | verified 2026-09-14 (card) | 0.999보다 final loss 약간 낮고 loss spike 적음(App. S, table 없음) |
| datablations GPT-2-architecture study model | ≤ 9B | pretrain-stable | 무시할 수 있는 loss penalty의 epoch | 최대 4(`R_D = 3`) | §6, App. L | verified 2026-09-14 (card) | 4 epoch 8.7B: 1 epoch 대비 validation loss +0.5%; 19-task average는 ~4 뒤 하락 |
| Llama 3 405B | 405B | pretrain-stable | peak LR; warmup; batch ramp | 8e-5; 8,000 step; seq 4,096의 4M token → 252M token 뒤 seq 8,192의 8M → 2.87T 뒤 16M | arXiv:2407.21783 §3.4.1 | verified 2026-09-17 | report가 stability를 ramp에 귀속: “few loss spikes … no interventions”(§3.4.1); printed ablation 없음 |
| Tülu 3 SFT study run(Llama 3.0 base, Tülu 2 SFT mixture) | §4.3.2에 미기재 | SFT | loss aggregation; LR; epoch | sum loss; 5e-6; 2 | arXiv:2411.15124v5 §4.3.2, Figs. 5–6 | verified 2026-09-17 | Fig. 5: 5e-6 sum loss가 sweep 최고; Fig. 6: 2 epoch가 3–7보다 좋음 |
| Mistral-7B on FLAN_20k (Kundu et al.) | 7B | SFT | correctness/throughput batching | example별 position ID의 online minibatch packing | arXiv:2407.09105v6 §4.1, Table 2 | verified 2026-09-17 | validation loss 1.127 대 padding 1.129; position ID 없는 packing 1.306; position ID 있는 offline packing 1.284 |
| Entropy-mechanism §2 run | 0.5B–32B | RL | prompt filter | response가 모두 correct 또는 모두 incorrect인 prompt drop | arXiv:2505.22617v1 §2.2 | verified 2026-09-14 (card) | ablation 보고 없음 |
| Qwen2.5-32B (entropy-mechanism §4.3) | 32B | RL | KL-Cov `k`; `β` | 2e-4; 1 | §4.3 | verified 2026-09-14 (card) | 7-benchmark math average 45.8(GRPO) → 52.2(Table 2) |
| Gao et al. proxy-RM study policy | 1.2B | RL | KL penalty coefficient | §3.6 외 모든 RL experiment에서 0 | arXiv:2210.10760 §2, App. C | verified 2026-09-14 (card) | §3.6, Fig. 9: nonzero penalty가 early stopping처럼 행동, frontier를 올리지 못함 |

**작은 범용 run의 시작점.** Web text로 학습하는 약 1B parameter 이하 dense decoder에는 위 verified row가 다음을 뒷받침한다. Global-norm clipping 1.0, z-loss coefficient 1e-4, per-head qk-layernorm, 1e5-step cosine schedule 안에서 5e3 step linear warmup, independent weight decay 1e-4다. 이는 sequence length 512, batch 256 sequence의 C4에서 2.4M–1.2B model로 모두 측정됐다([[small-scale-proxies-instabilities]] §2.1). 그 위 SFT stage에는 sum-loss aggregation, LR 5e-6, 2 epoch가 Tülu 3가 Llama 3.0 base/Tülu 2 SFT mixture sweep으로 선택한 setting이다([[tulu-3]] §4.3.2, Figs. 5–6). 그 sweep의 model size는 locus에 인쇄되지 않았으므로 sum loss가 gradient scale을 바꾸는 만큼 자신의 size에서 LR sweep을 다시 수행한다. 4.8B의 `ε = 1e-15` row를 smaller run으로 옮기지 말라. 이는 model 하나의 LR 0.3에서 선택됐다.

---

## 일반화 렌즈

**(a) 무엇이 breadth를 늘리는가.** Pre-training scale에서 unique data를 도는 epoch 수를 약 4 이하로 유지하는 것. 이때 held-out loss는 single-epoch 값의 0.5% 안이고 19-task average는 변하지 않는다([[data-constrained-scaling]] §6, Fig. 6). SFT scale에서는 Tülu 3가 측정한 2 epoch다(§4.3.2, Fig. 6). Task-specific continual tuning 전에 general instruction tuning을 수행하는 것. ALPACA-7b는 18.14% / 7.56% / 10.31% forgetting을 보이는 반면 LLAMA-7b는 34.57% / 31.33% / 31.72%다([[catastrophic-forgetting-continual-finetuning]] Table 6). Post-alignment weight와 pre-alignment weight를 interpolate하는 것은 측정된 alignment-forgetting Pareto front가 가장 좋았다([[mitigating-alignment-tax-rlhf]] §4.1). Flat entropy bonus 대신 highest-covariance token을 제한해 RL entropy collapse를 막는다([[entropy-mechanism-llm-rl]] §4.3).

**(b) 무엇이 narrowing 또는 forgetting을 일으키는가.** 측정된 knee를 넘는 repetition(약 4 epoch 뒤 19-task average 하락). 모든 target task가 개선되는 동안 BLOOMZ size 전반 MMLU의 9.54%–18.37%를 잃게 하는 task sequence continual fine-tuning. Reward가 오르는 동안 translation/reading comprehension을 단조롭게 낮추는 alignment optimization. Proxy score는 계속 오르지만 gold score가 peak 뒤 하락하는 reward over-optimization([[reward-model-overoptimization]] Fig. 1). Entropy consumption, 즉 2400 step 중 첫 200에서 entropy의 73%를 쓰고 `H = 0` accuracy ceiling이 `−a + b`인 현상. Multi-turn data를 다시 넣기 전 MATH를 움직였던 single-turn packing처럼 single-format training data.

**(c) 이 stage에서의 측정 방법.** 서로 놓치는 것을 잡는 도구 넷이다.

1. **Held-out multi-domain perplexity.** Held-out loss 하나는 domain gap을 숨긴다. Controlled 1B baseline 6개 중 C4-only model은 RedPajama arXiv domain에서 perplexity가 최대 391,171에 이르고, perplexity source 하나도 시험한 downstream task 8개 모두와 correlate하지 않는다([[paloma]] §4.1–4.2). Fixed vocabulary로 decontaminated data의 domain별 perplexity를 macro-average하여 보고한다.
2. **이전 checkpoint 대비 broad benchmark average.** Training curve 대비가 아니다. Forgetting과 alignment tax를 드러내는 axis다.
3. **Target task의 out-of-distribution variant.** Rule, surface attribute, template을 바꾸며, rule learning과 memorization을 가르는 방법이다([[sft-memorizes-rl-generalizes]] §5.1–5.2).
4. **Contamination probe.** Benchmark builder가 publish할 수 없었던 material에서 같은 capability를 측정한다([[swe-bench-illusion]] §4.1).

위 failure에 매핑하면 regression suite는 failure당 row 하나이며, previous checkpoint와 각 row를 대조하기 전에는 run을 promote하지 않아야 한다.

| Failure | 이를 드러내는 instrument | 비교 |
|---|---|---|
| Repeated-data overfitting | 같은 distribution의 held-out loss; 19-task형 broad average | current 대 1-epoch 또는 2-epoch run([[data-constrained-scaling]] §6) |
| Continual tuning 뒤 forgetting | FG로 score하는 fixed general suite(MMLU, commonsense, reading comprehension) | current 대 pre-tuning checkpoint([[catastrophic-forgetting-continual-finetuning]] Eq. 1) |
| Alignment tax | alignment metric **대비** plot한 general-capability metric | checkpoint 또는 interpolation ratio당 point 하나([[mitigating-alignment-tax-rlhf]] §4) |
| Contamination | unpublished material의 matched task | public benchmark 대 matched slice([[swe-bench-illusion]] §4.1) |
| Template narrowing | 두 번째, 학습하지 않은 prompt format에서 같은 capability | format A 대 format B, 같은 checkpoint |
| Rule learning 대신 memorization | Task의 rule- 또는 attribute-changed variant | in-distribution 대 variant([[sft-memorizes-rl-generalizes]] §5.1) |
| Entropy collapse/reward over-optimization | fitted `R = −a·exp(H) + b`의 entropy trajectory; `√KL` 대비 held-out/gold score | 끝에서만이 아니라 update 전반([[entropy-mechanism-llm-rl]] §2.4; [[reward-model-overoptimization]] §3.2) |
| 한 average에 숨은 domain-specific loss regression | macro-averaged per-domain perplexity | current 대 previous checkpoint([[paloma]] §4.1) |

---

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| Label shift 뒤 masking 적용 | 첫 response token이 학습되지 않음; inference에서 malformed turn start | sample별 `(labels[:, 1:] != -100).sum(-1) == response_len` |
| Batched(2-D) tensor에 `labels[:prompt_len] = -100` | Sample 전체가 조용히 제외; 나머지에서는 prompt로 학습 | 첫 `prompt_len` row의 active-token count가 0 |
| `cu_seqlens` 없이 packing 또는 이를 무시하는 kernel | Unpacked보다 loss 약간 나쁨; 오류 없음 | Packed 대 sample별 forward logit equality test, tolerance 1e-4 |
| Reset하지 않는 `position_ids`에서 boundary 도출 | Varlen kernel이 조용히 dense causal attention으로 fallback | Packed row별 `position_ids.min() == 0`, 복원한 `cu_seq_len` count가 sample count와 같은지 assert |
| Gradient accumulation 아래 mean loss | `grad_accum`에 따라 loss/result 변화; short sample overweight | 같은 data에서 `grad_accum` 1 대 4, loss 일치해야 함([[tulu-3]] §4.3.2) |
| AdamW `ε`을 NaN remedy로 올림 | Update가 수축하고 loss plateau | Layer별 gradient RMS를 log해 `ε`과 비교([[small-scale-proxies-instabilities]] §3.4) |
| Zero-variance RL group을 numerical bug로 취급 | `std = 0` group fraction 상승; reward curve flat | Advantage는 NaN이 아니라 정확히 0; clamp 대신 prompt filter |
| 상승하는 proxy reward를 progress로 해석 | Proxy는 오르나 gold/human score는 peak 뒤 하락 | `√KL` 대비 held-out score plot, peak 근처에서 중단([[reward-model-overoptimization]] §3.2) |
| Fine-tune을 target task로 판단 | Target task 개선; MMLU/RACE/translation 하락 | Fixed general suite에서 pre-tuning checkpoint 대비 FG 보고 |
| Benchmark gain을 capability로 해석 | Public set 고득점, matched unpublished set 저득점 | Held-out repository 또는 post-cutoff slice([[swe-bench-illusion]] §4.1) |

---

## 이해를 점검하세요

1. Rollout 8개 group이 모두 reward 1을 받고 run이 `(A − mean)/(std + 1e-6)`을 쓴다. 결과가 NaN이 아니라 정확히 0인 이유와, numerical guard가 아닌 prompt-selection change가 올바른 대응인 이유를 설명하라.
2. Colleague가 packing에서 `position_ids`를 reset하지 않으면 “RoPE relative position을 깨뜨린다”고 한다. 올바른 block-diagonal mask의 intra-document attention에서 이것이 왜 틀렸는지, reset하지 않은 position이 그래도 run을 깨는 상황 두 가지를 제시하라.
3. 같은 data/global batch/seed의 run 두 개가 `grad_accum = 1`, `grad_accum = 4`이며 loss가 30% 다르다. 무엇이 다르게 weight되는지 도출하고 equality를 복원할 fix를 제시하라.
4. High LR의 4.8B run이 healthy `lr`, 0 근처 `clipped_fraction`, normal active-token count로 plateau한다. 다음으로 할 optimizer-level measurement와 diagnosis를 확인하는 result는 무엇인가?
5. SFT checkpoint가 target benchmark를 6 point 개선하고 MMLU 3 point를 잃는다. FG 정의를 이용해 “3 points”와 “FG = x%”가 checkpoint 두 개의 순위를 왜 다르게 매길 수 있는지, 무엇을 report할지 설명하라.
6. RLVR run의 entropy가 꾸준히 떨어지고 accuracy는 오른 뒤 flat해진다. `R = −a·exp(H) + b`와 covariance mechanism을 사용해 flat entropy bonus 0.01이 entropy explosion을 일으킨 반면 token `1e-4` fraction을 제한하면 entropy와 accuracy를 모두 높인다고 보고된 이유를 설명하라.
7. Training distribution held-out split score가 같은 checkpoint 두 개가 있다. 어느 것이 rule을 배웠는지 말해 주는 가장 싼 measurement 하나를 설계하고, 어떤 result가 memorization의 근거인지 말하라.

---

## 연결

- **이전:** ch-06 — Checkpointing, In-Loop Evaluation, and Checkpoint Selection. 이 장 diagnostic을 가능하게 하는 instrumentation/checkpoint-selection rule이 그곳에 있다.
- **다음:** ch-08 — Lab: Minimal Trainer with a Target-versus-General Capability Measurement. Lab은 여기의 equality test와 general-capability comparison을 구현한다.
- ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability: §1.3의 AdamW `ε` 및 update-scale material.
- ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch: 도달 가능한 numerical failure를 결정하는 precision choice.
- ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization: §3에서 참조한 warmup/batch-ramp setting.
- ch-04 — Sequence Packing, Loss Masking, and Chat Templates: 이 장이 correctness를 시험하는 mechanics.
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination: repetition limit/decontamination 상세.
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate: §2/§4의 SFT-side treatment.
- ch-43 — Entropy, Output Diversity, and KL Control in RL: §5.1의 전체 treatment.
- ch-47 — Evaluation Harness and Suite Design for General Capability: generalization lens가 가정하는 suite 구축.

---

## 출처

- [[small-scale-proxies-instabilities]] — attention-logit growth, output-logit divergence, LR sensitivity, warmup/weight-decay effect, AdamW-ε result; §1/§3의 evidence base.
- [[gradient-clipping]] — §3에서 쓴 global-norm clipping 정의.
- [[sequence-packing]] — packed training의 block-diagonal masking과 sequence별 position index.
- [[packing-position-id-evidence]] — §2.2/§4.5에서 쓴 locus를 담은 세 primary packing study(Krell 2021, Kundu 2024, Wang 2024)의 chapter-local extract.
- [[hf-gradient-accumulation-fix]] — October 2024 `transformers` fix의 chapter-local extract; §2.3에서 인용한 correctness rule/patch.
- [[tulu-3]] — batch-aggregation equation, sum-loss decision, epoch/LR sweep(§4.3.2).
- [[llama-3]] — 405B pre-training schedule/batch ramp, loss-spike statement, opening section interruption count.
- [[data-constrained-scaling]] — repetition limit, effective-data formula, β₂ note.
- [[catastrophic-forgetting-continual-finetuning]] — §4.2에서 쓴 FG metric 및 BLOOMZ/LLAMA/ALPACA forgetting table의 chapter-local extract.
- [[mitigating-alignment-tax-rlhf]] — alignment-tax setup, mitigation comparison(§4.1), heterogeneous model averaging(§6)의 chapter-local extract.
- [[sft-memorizes-rl-generalizes]] — §5.3/generalization lens의 OOD rule/visual variation result chapter-local extract.
- [[entropy-mechanism-llm-rl]] — §5.1의 entropy-performance law, covariance mechanism, intervention result.
- [[reward-model-overoptimization]] — §5.2의 `√KL` gold-versus-proxy curve 및 KL-penalty result.
- [[reward-hacking-taxonomy]] — 모든 stochastic policy에 대해 non-trivial unhackable proxy가 존재하지 않는다는 formal statement.
- [[swe-bench-illusion]] — §4.4의 contamination probe와 수치.
- [[paloma]] — generalization lens에서 held-out instrument로 쓰는 per-domain perplexity.
