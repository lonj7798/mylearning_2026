<!-- chapter: ch-07
     track: foundations
     title: Common Training Failure Modes
     sources: [[gradient-clipping]], [[mixed-precision]], [[adam]], [[loss-masking-prompt]], [[sequence-packing]], [[fsdp-sft]], [[karpathy-training-neural-net-recipe]], [[olmo-2]], [[olmo-3]], [[llama-3]], [[openrlhf-entropy-debugging]], [[entropy-collapse-ppo]]
     figures: figures/failure-modes-tree.html
-->

# 7장 — 흔한 학습 실패 모드

> **핵심 통찰.** 현대 LLM 학습은 처음부터 *크래시*하지 않는다. 먼저 *드리프트*한다. 비용이 큰 모든 사고, 즉 84k 스텝의 치명적 NaN, 712번째 `all_reduce`에서의 행, 하루 동안 `<|pad|>` 토큰으로 조용히 학습한 SFT 실행은 손실보다 먼저 형태가 바뀐 로그 값이 선행했다. 학습 루프를 불변식들의 파이프라인(유한한 손실, 비어 있지 않은 배치, 프롬프트가 마스킹된 라벨, 동기화된 collective, 제한된 엔트로피)으로 다루면 모든 실패 모드는 "어떤 불변식이 깨졌고, 그것이 얼마나 오래전인가"가 된다. 이 장의 실패들은 그 불변식들을 반복해서 조용히 깨뜨리는 것들이다.
>
> **가이드라인.** 전체 실행을 시도하기도 전에 모든 새 파이프라인에서 Karpathy의 값싼 두 가지 테스트를 실행하라. *초기 손실은* `ln(V)`와 같아야 하고, *그다음 단일 배치에 거의 0까지 과적합*해야 한다([[karpathy-training-neural-net-recipe]]). 매 스텝마다 `pre_clip_grad_norm`, 토큰별 엔트로피, 활성 토큰 수, 랭크별 heartbeat를 계측하고, liveness 제약을 인라인으로 assert하라. 실행을 저렴하게 멈추는 `assert torch.isfinite(loss)`는 버그를 숨기는 영리한 복구보다 가치가 크다.

---

## 이 장이 존재하는 이유

1-6장은 기계적으로 올바른 학습 스텝을 만들었다. 혼합 정밀도([[mixed-precision]])를 존중하는 옵티마이저([[adam]]), 노름 클리핑된 backward pass([[gradient-clipping]]), 패킹되고 마스킹된 SFT 배치([[sequence-packing]], [[loss-masking-prompt]]), FSDP 샤딩 forward([[fsdp-sft]]), 그리고 비트 단위로 정확한 재개([ch-06])가 그것이다. 그 장들의 모든 내용은 루프가 *건강하다*고 가정한다. 이 장은 그 루프가 어떻게 죽는지에 대한 목록이다.

이 목록의 형태는 프런티어 보고서들이 정한다. Llama 3의 405B 실행은 15.6T 토큰에 걸쳐 3.8e25 FLOP를 소모했다([[llama-3]]). 이 정도 사고 노출 구간이면 *어떤* 저확률 실패 모드도 반드시 발생한다. OLMo 2([[olmo-2]])는 OLMo 1을 죽인 loss spike 표현형을 선제적으로 막기 위해 아키텍처 ablation을 할애했다. OLMo 3([[olmo-3]])는 1,024개 H100에서 pretraining을, 128개에서 mid-training을, 256개에서 post-training을 수행했다. 단일 랭크가 collective에서 빠지면 전체 world group이 행에 빠지는 세 개의 별도 클러스터다. 이 프로젝트들은 이전 세대의 실패를 목록화하고 그에 대한 assertion을 배선함으로써 안정성을 얻었다. 그 목록이 이 장이다.

아래의 모든 절은 같은 방식으로 구성된다. **증상 → 깨진 불변식 → 진단 트리 → 수정 → 더 일찍 잡았을 로그.** Karpathy의 격언, 즉 *"신경망 학습은 조용히 실패하므로 유일한 방어는 모든 스텝을 명시적 예측에 대해 검증하는 집요하고 데이터 우선적이며 점진적인 워크플로"*([[karpathy-training-neural-net-recipe]])라는 말이 구성 원칙이다. 아래의 모든 수정은 그 원칙의 사례다.

---

## 1. NaN / Inf — 세 가지 산술적 원천

NaN은 결코 자발적으로 생기지 않는다. Transformer 스텝 안의 세 연산 중 하나에서 나오며, 진단 트리는 짧다.

**1a. Softmax / logit overflow.** Attention은 `softmax(QKᵀ / √d_k)`를 계산한다. `QKᵀ`가 커지면, 즉 query/key 크기가 제어되지 않을 때, `exp()`는 fp16에서 오버플로(max ≈ 6.5e4)하고, 더 미묘하게는 bf16에서 모든 정밀도를 잃는다(mantissa 7비트, `exp(89)`는 이미 포화된다). 언어 모델 head의 output-softmax도 vocabulary 축에서 같은 실패 모드를 갖는다. OLMo 1의 spike 표현형이 정확히 이것이었다. 논문은 그 치료가 단일 수정이 아니라 QK-Norm on attention, output logits에 대한 Z-loss 같은 logit scaling 개입의 *스택* 덕분이었다고 설명한다([[olmo-2]]).

> *"아키텍처 안정성 recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. OLMo 1을 괴롭힌 training-spike 표현형을 방지한다."* — [[olmo-2]]

커널 내부의 산술적 수정은 표준적이다. `exp` 전에 row-max를 빼라.

```python
# numerically stable softmax — the only one you should ever see
m = x.amax(dim=-1, keepdim=True)       # per-row max
z = (x - m).exp()
p = z / z.sum(dim=-1, keepdim=True)
```

모든 production attention kernel(FlashAttention, SDPA, xFormers)은 이것을 수행한다. 직접 구현하면서 `amax` subtraction을 잊으면 128-token row에서 fp16 softmax는 처음 몇 스텝 안에 NaN을 낸다.

**1b. KL / log-softmax / CE의 `log(0)`.** 두 번째 원천은 `p`가 정확히 0이 될 수 있는 모든 `log(p)`다. RL의 KL-to-reference penalty(ch ~40)가 가장 흔한 범인이다. k3 estimator `(π_ref/π) − 1 − log(π_ref/π)`([[openrlhf-entropy-debugging]])는 `π`가 정확히 0인 단일 vocab entry만 있어도 즉시 NaN이 된다. policy가 붕괴했다면 쉬운 일이다. `torch.nn.functional.cross_entropy(logits, targets)`를 쓰는 cross-entropy는 안전하다(log-softmax와 CE를 fuse하기 때문). 하지만 손으로 작성한 `torch.log(softmax(x)) * y`는 안전하지 않다. 방어적 guard는 `log_softmax`를 직접 호출하거나 clamp하는 것이다. `logp = torch.log(p.clamp_min(1e-9))`. fp16에서는 `1e-9` 자체가 underflow한다. [[mixed-precision]]은 *"softmax 계산은 fp32로 유지하라. cross-entropy loss도 fp32로 유지하라."*고 경고한다. 이것이 그 이유다.

**1c. Advantage / reward normalization의 division by zero.** RL 학습은 배치별로 advantage를 정규화한다. `A ← (A − mean) / (std + eps)`. 한 배치가 우연히 K개의 동일한 reward를 포함하면(모든 rollout이 같은 binary score를 얻은 경우), `std = 0`이고 `eps`가 너무 작아 배치의 모든 advantage가 NaN이 된다. 같은 패턴은 `v̂`가 underflow할 때 Adam update `α · m̂ / (√v̂ + ε)`에도 나타난다([[adam]]).

> *"fp16에서 `eps`를 너무 작게 설정하면 division-by-zero NaN이 난다. optimizer step에서 NaN이 보이면 `1e-5`로 올려라."* — [[adam]]

방어 패턴은 `std.clamp_min(1e-6)`이다. RL에서는 추가로 "`std < 1e-4`인 배치의 비율" 로그 카운터가 필요하다. 이 카운터가 갑자기 상승하는 것은 엔트로피 붕괴(§6)의 가장 이른 신호다. policy가 다양한 completion 생성을 멈췄고, 한 prompt의 모든 rollout이 같은 reward를 얻는다는 뜻이다.

| NaN 증상 | 가장 가능성 높은 원천 | 첫 확인 |
|---|---|---|
| `loss.backward()` 출력에서만 NaN | softmax / attention overflow | `logits.abs().max()`를 로그하고 QK-Norm 또는 Z-loss 추가 |
| 손실 scalar 자체에서 NaN | CE / KL의 `log(0)` | `log(softmax(...))`가 아니라 `log_softmax`인지 확인하고 clamp |
| `optimizer.step()`에서 NaN | `v̂` underflow 또는 advantage norm의 /0 | `eps`를 올리고, `std.clamp_min(1e-6)`을 쓰며, fp32 master 확인 |
| 재개 후에만 NaN 발생 | loss-scaler 상태 누락([[mixed-precision]]) | `GradScaler.state_dict()`를 저장( ch-06 §5.2 참조) |

**Liveness assertion.** 어떤 production trainer에서든 가장 값싼 bug catcher는 다음이다.

```python
loss = model(**batch).loss
assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss.item()}"
```

이는 gradient history가 오염된 300스텝 뒤가 아니라 발생 지점에서 NaN을 잡는다. 매 스텝 로그되는 `pre_clip_grad_norm`([[gradient-clipping]])과 결합하라. 그 scalar의 100배 spike는 NaN을 1-5스텝 앞서 예측하므로 skip-step mitigation을 위한 여유를 준다.

---

## 2. 손실 발산 vs 손실 spike vs 손실 plateau — 진단 트리

세 가지 손실 병리는 모두 "선이 예상한 일을 멈췄다"는 시각적 형태를 공유하지만, 원인과 수정은 서로 다르다. 오진은 몇 시간을 잃게 만든다. 트리는 다음과 같다.

**Loss spike.** 단일 스텝 또는 몇 스텝의 점프(1-20배) 뒤 회복 또는 발산으로 이어지는 형태다. 거의 보편적인 근본 원인은 out-of-distribution micro-batch가 이미 큰 weight step과 충돌하는 것이다. [[gradient-clipping]]: *"갑작스러운 100배 spike는 보통 임박한 loss-spike나 NaN을 예측한다."* OLMo 2 mitigation stack은 이 형태를 위해 구체적으로 층층이 쌓여 있다.

> *"Pretraining의 loss spike: 표준 Llama-3 / OLMo-2 mitigation stack은 (1) global-norm clip 1.0, (2) loss-spike에서 skip-step, (3) embedding-norm monitoring이다."* — [[gradient-clipping]] (cross-ref)

Skip-step의 의미는 다음이다. `loss > running_mean + k·running_std`(k≈5)이면 gradient를 버리고, dataloader를 전진시키며, optimizer state는 유지한다. 이는 한 배치의 compute만 잃는다. 대안은 몇 시간을 잃는 divergence rollback이다.

**Loss divergence.** 곡선이 단일 점프 없이 수백 스텝에 걸쳐 단조롭게 오른다. Spike mitigation은 여기서 쓸모없다. 단일 배치가 범인이 아니기 때문이다. 흔한 원인은 learning rate가 너무 높음(Karpathy sanity `ln(V)` check를 시도하라. init loss가 이미 `ln(V)`보다 높다면 LR만의 문제가 아니다), warmup이 너무 짧음(ch-03), softmax / logits가 overflow하지만 bf16이 이를 숨김(NaN은 없는데 bf16의 range는 fp32급이고 정밀도는 사라졌기 때문)이다. 진단: `||W_embed||`와 layer별 weight norm을 로그하라. 발산 중 weight norm이 위로 추세를 보이면 weight decay가 부족하거나 꺼진 것이다. 아래로 추세를 보이면 gradient가 noise에 지배되고 있다.

**Loss plateau.** 곡선이 데이터가 정당화하는 값보다 높은 곳에서 평평해진다. 서로 다른 세 원인이 있다.

1. **죽은 learning rate** — schedule이 0으로 decay됨(cosine이 0에 도달, WSD가 cooldown을 지나 decay), 또는 LR scheduler off-by-one([ch-06 §5.3]).
2. **너무 낮은 clip threshold** — [[gradient-clipping]]: *"clipping threshold가 너무 낮으면(예: 0.1) optimizer가 hard example에서 진짜 step을 만들지 못하고 loss가 plateau에 빠진다."*
3. **죽은 data pipeline** — 배치가 구조적으로 잘못됨(전부 padding, 전부 같은 label, loss가 전부 mask됨). §3이 이 원인을 자세히 다룬다.

가장 빠른 구분 신호는 `pre_clip_grad_norm`이다. 죽은 LR → norm은 건강하지만 optimizer step이 0으로 scale됨(`lr`도 로그하라). 과도한 clipping → raw norm은 크지만 매 스텝 threshold로 clipped됨. 죽은 pipeline → *손실에 들어가는 실제 label이 없기 때문에* norm이 0에 가깝다.

**진단 call-flow(위에서 아래로):**

```
손실 병리 관측
├── 단일 스텝 점프 → SPIKE branch
│   ├── grad_norm pre-clip > running의 100배 → skip-step + 배치 조사
│   └── logits |max| > 50 (bf16) → QK-Norm / Z-loss / 더 낮은 LR
├── 단조 상승 → DIVERGENCE branch
│   ├── ||W|| 상승 → LR이 너무 높거나 WD 꺼짐
│   ├── ||W|| 하락 → gradient noise가 지배; init 확인, LR 낮춤
│   └── 뚜렷한 추세 없음 → 마지막 코드 변경 되돌리기 시도(Karpathy의 규칙)
└── 예상 floor보다 높은 곳에서 flat → PLATEAU branch
    ├── lr == 0 → scheduler bug (ch-06 §5.3 참조)
    ├── clipped_fraction == 1.0 → clip threshold 올리기
    └── active_tokens_per_batch ≈ 0 → §3 dead pipeline
```

각 branch에는 그것을 구분하는 persistent metric이 있다. 이것이 ch-06의 instrumentation 투자가 회수되는 지점이다. `pre_clip_grad_norm`, `clipped_fraction`, `active_tokens_per_batch`, `||W_embed||`, `lr`가 매 스텝 로그되지 않으면 트리는 추측으로 무너진다.

---

## 3. 죽은 data pipeline — 조용히 padding으로 학습하기

가장 사기를 꺾는 버그다. 손실 곡선은 그럴듯해 보인다(평평하거나 가짜 신호에서 천천히 내려간다). 모델은 아무것도 배우지 않는다. 메커니즘은 다음과 같다. upstream filter가 비어 있거나 all-padding인 배치를 내보내고, collate function이 그것을 직사각형 tensor로 padding하고, loss mask가 모든 위치를 0으로 만들며, un-masked token이 0개인 상태에서 `reduction="mean"`은 framework에 따라 조용히 `0.0` 또는 `nan`을 만든다. 그다음 divide-by-zero guard에 걸려 "training-loss-equals-epsilon"이 영원히 계속된다.

[[loss-masking-prompt]]의 구체적인 SFT 버전에서 canonical masking code는 다음이다.

```python
labels = input_ids.clone()
labels[:prompt_len] = -100          # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

이것이 조용히 죽는 세 가지 방식이 있다.

- generator가 response가 length filter에 의해 떨어진 truncated example을 반환해 `prompt_len == input_ids.size(-1)`가 된다. mask가 모든 것을 덮고 `loss`는 0개 token에서 계산된다.
- chat-template renderer가 raw record의 assistant field가 비어 있어 assistant turn 없이 `<|system|> ... <|end|>`만 반환했다.
- deduplication filter가 K token보다 긴 모든 completion을 제거했고, SFT mix에 모든 prompt가 긴 completion만 가진 cluster가 있어 배치가 구조적으로 빈 상태가 된다.

**불변식.** SFT 배치는 다음을 만족해야 한다.

```python
active = (labels != -100).sum()
assert active > 0, f"batch {step} has zero active tokens"
logger.log_scalar("tokens/active", active.item(), step=step)
```

매 스텝 `active_tokens_per_batch`를 로그하라. `~batch_size · 512`에서 `~batch_size · 5`로 갑자기 떨어지는 것은 모호하지 않다. OLMo 3의 자체 보고서는 이 부류와 관련된 운영상 불안을 기록한다. *"SFT를 Open Instruct에서 Olmo Core로 옮기자 throughput이 8배 개선되었다고 보고되었다"*([[olmo-3]]) 같은 것이다. 이런 throughput jump에는 거의 항상 올바른 설명과 잘못된 설명이 모두 있다. 올바른 설명은 새 kernel이고, 모든 팀이 먼저 확인하는 잘못된 설명은 "collator가 바뀌면서 대부분 padding으로 학습하기 시작했다"는 것이다. OLMo 3 공개에서 두 팀 모두 swap을 shipping하기 전에 `tokens/active` assertion을 실행했다.

**Data layer의 인접 실패:**

- **Rank-local filter로 인한 empty-batch.** FSDP에서 global batch는 `micro_batch · grad_accum · dp_size`다. DDP 아래에서 단일 rank가 빈 micro-batch를 만들면 다른 rank들이 결코 나타나지 않을 gradient를 기다리므로 `all_reduce`가 행에 빠진다. rank별 `active_tokens`를 로그하고 0에 alarm하라.
- **Iterator exhaustion.** `StopIteration` 처리가 없는 single-epoch iterator가 cycled될 때 조용히 처음으로 재시작되어 첫 epoch 데이터를 같은 label로 재생한다. 합법적인 second epoch처럼 보이지만 그렇지 않은 느린 손실 감소로 관측된다. 첫 100개 sample을 hash하고 이전 epoch hash와의 반복 없음(non-repetition)을 assert해서 감지할 수 있다(ch-06 §5.1 참조).
- **All-one-label batch.** Pure-RL(모든 rollout이 reward-verifier batch에서 성공하거나 모두 실패)이다. §1c에서 advantage-normalization /0로 다뤘다. 이것은 dead pipeline의 RL 사례다.

---

## 4. Masking bug — off-by-one과 cross-sample attention leakage

두 masking bug는 거의 보이지 않는다. 손실을 절대값으로 약 0.5-2%만 악화시키기 때문이다. 짧은 실행에서는 "약간 나쁜 hyperparameter"처럼 보일 만큼 작고, 긴 실행에서는 benchmark 순위를 잃을 만큼 크다.

**4a. Prompt-masking off-by-one.** Causal LM loss의 shift-for-next-token 패턴은 `logits[..., :-1, :]` 대 `labels[..., 1:]`이다. prompt mask는 shift *전에*, 길이 T의 원래 `labels`에 적용되어야 한다. 그래야 shift 후 index `i`의 prompt token이 logit position `i-1`에서 mask된다. 흔한 잘못된 형태:

```python
# WRONG #1 — mask applied post-shift
labels_shift = labels[..., 1:].clone()
labels_shift[:, :prompt_len] = -100         # off by one: position prompt_len-1 not masked

# WRONG #2 — prompt_len computed on packed block
labels[:prompt_len] = -100                  # in a packed block, this only masks the first pack's prompt

# RIGHT
labels = input_ids.clone()
labels[:prompt_len] = -100                  # full-length mask
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

Wrong #1은 마지막 prompt token을 loss에 누출한다. 그 token의 label은 response의 첫 token이고, gradient는 mask되어야 했던 위치를 통해 흐른다. 순효과: 모델은 "직전 predecessor에서 첫 response token을 예측하라"는 작지만 일관된 신호를 받는다. 단일 turn prompt에서는 무해하지만, user↔assistant boundary를 붕괴시키는 multi-turn에서는 *적극적으로 해롭다*. [[loss-masking-prompt]]는 multi-turn 규칙을 명시한다.

> *"turn이 `[u_1, a_1, u_2, a_2, …, u_k, a_k]`인 conversation에서는 **모든** user turn을 mask하고, **모든** 이전 assistant turn(a_1..a_{k−1})을 mask하며, a_k token에만 train한다."*

디버깅 방법: 배치에서 세 개의 random sample을 고르고, tokenizer의 `decode`를 `input_ids[labels != -100]`에 실행하라. 출력은 정확히 의도한 assistant text여야 한다. `<|user|>`나 template token을 포함한다면 mask는 적어도 하나 이상 어긋나 있다.

**4b. Packing에서의 cross-sample attention leakage.** [[sequence-packing]]은 올바른 kernel을 형식화한다. `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)`는 block-diagonal causal attention을 계산한다. 다음 세 가지 중 하나라도 틀리면 sample들이 서로의 attention으로 새어 들어간다.

1. **`cu_seqlens`가 생략됨** 그리고 kernel이 기본 dense causal로 fallback → sample 2가 lower-triangular를 통해 sample 1의 token에 attention한다.
2. **`position_ids`가 sub-sequence boundary에서 reset되지 않음** → RoPE가 pack 전체의 절대 offset을 사용해 rotate하므로 sample 2의 token 0은 position ≈ L₁을 가져 relative-position math가 왜곡된다.
3. **Custom attention override**(Liger, TorchTune, HF `attn_implementation="eager"`)가 손으로 작성한 path에서 `cu_seqlens`를 전달하지 않아 조용히 무시한다.

[[sequence-packing]]:

> *"Masking이 없으면 sequence 2의 token t가 sequence 1의 token에 attend할 수 있다 → softmax의 partition function이 document 사이로 leak된다 → sequence 1 token의 gradient까지 바꾼다."*

영향은 정량적이다. ablation report([[sequence-packing]])는 *올바른 masking이 있는* packing은 2배 throughput에서 unpacked와 수학적으로 동등함(metric change 없음)을 보인다. *masking 없는* packing은 GLUE급 metric을 0.5-2 point 악화시킨다. 짧은 SFT 실행에서는 noise처럼 보일 만큼 작고, 공식 eval에서는 드러날 만큼 크다. 모든 변형을 잡는 unit test는 다음이다.

```python
# at unit-test time, with deterministic weights
out_packed   = model(input_ids_packed, cu_seqlens=cu)
out_unpacked = torch.stack([model(s) for s in split_by_cu(input_ids_packed, cu)])
assert (out_packed - out_unpacked).abs().max() < 1e-4, "cross-sample leak"
```

이것을 CI에 넣어라. ch-06 시기의 모든 OLMo 3 framework swap은 이것을 다시 통과해야 했다.

---

## 5. Distributed hang — NCCL, all-reduce rank drop, py-spy / gdb attach

가장 무서운 실패 부류다. 실행은 크래시하지 않았고, 진행하지 않으며, 로그하지 않는다. 모든 rank가 결코 완료되지 않을 collective 안에 앉아 있다. FSDP FULL_SHARD([[fsdp-sft]]) 아래에서:

> *"FSDP는 model parameter, gradient, optimizer state를 data-parallel group 전체에 shard하고, 필요할 때 AllGather로 full parameter를 재구성한다. Gradient는 ReduceScatter로 reduce된다."*

각 transformer block의 forward는 `AllGather`를 내보내고, 각 backward는 `AllGather` + `ReduceScatter`를 내보낸다. 32-layer model이면 스텝당 약 96개의 collective다. 어떤 rank라도 그중 하나를 놓치면 다른 모든 rank는 예외 없이 NCCL wait에 갇힌다. Incident log에서 빈도순으로 정렬한 원인은 다음과 같다.

1. **Rank-local control-flow divergence.** `if batch["has_images"]:` 같은 branch가 일부 rank에서만 trigger된다. branch를 탄 rank는 다른 rank가 내보내지 않는 추가 collective(예: vision-tower forward)를 내보낸다. mismatch, hang이다.
2. **한 rank의 empty micro-batch.** Per-rank data filter가 zero-token batch를 내보내고, forward가 short-circuit되며, 이후 `AllGather`가 절대 발생하지 않는다. §3 empty-batch bug의 distributed form이다.
3. **한 rank의 OOM.** collator 이후 한 rank가 약간 더 긴 sequence를 받아 OOM이 나고, CUDA caching allocator가 NCCL을 깨끗하게 내리지 않은 채 그 process를 abort한다. peer들은 30분 NCCL timeout을 기다린다.
4. **Infra: 끊어진 network link, sled reboot, GPU ECC error.** Llama 3의 405B 실행([[llama-3]])과 OLMo 3의 1,024-H100 pretraining([[olmo-3]])에서는 scale상 이것들이 *흔하다*. Llama 3는 수천 개 GPU에 걸쳐 15.6T token을 학습했다. MTBF가 몇 달인 어떤 component도 그 fleet size에서는 주당 여러 번 발생한다.

**NCCL timeout.** PyTorch의 기본 NCCL timeout은 30분이다. 즉 단일 stuck collective가 첫 rank가 raise하기 전에 *전체 cluster*의 30분을 낭비한다는 뜻이다. 1,024-H100 cluster에서는 hang당 512 H100-hour다. 2024 cloud rate로는 hang당 수백 달러다. 타이트하게 설정하라.

```python
import datetime
torch.distributed.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(minutes=5),   # not the 30-minute default
)
```

5분은 느린 checkpoint save를 흡수할 만큼 길고, 실제 hang이 빠르게 실패해 행동 가능한 py-spy trace를 dump할 만큼 짧다.

**py-spy / gdb attach 의식.** Hang이 감지되면(loss curve cadence 없음, dashboard에서 tokens/sec → 0) 첫 대응 프로토콜은 다음이다.

```bash
# On the node you suspect is the straggler (or all nodes, in parallel via pdsh):
py-spy dump --pid $(pgrep -f "train.py" | head -1)   # native Python stack
# If py-spy shows a C-level wait:
gdb -p $(pgrep -f "train.py" | head -1)
(gdb) thread apply all bt
(gdb) detach
```

보고 싶은 것은 모든 rank가 대략 같은 layer의 `ncclAllReduce` 또는 `ncclAllGather`에서 멈춰 있는 모습이다. rank 3이 다른 모두보다 step의 *더 이른* 지점에서 멈춰 있다면 rank 3이 collective를 놓친 rank다. 그 rank를 디버그해야 한다. rank 3이 *더 늦은* 지점(다른 rank가 건너뛴 추가 op 내부)에 있다면 rank 3이 divergent branch를 탄 rank다. 이 구분은 dataloader code를 `git blame` 한 번 하는 가치가 있다.

**Heartbeat instrumentation.** Liveness-invariant 관점에서 hang은 "per-rank heartbeat counter가 전진을 멈춤"이다. 배경 thread가 `step, rank, wall_time`을 매초 shared file에 쓰게 구현하라. Monitoring job은 "max rank wall_time – min rank wall_time > 60s"에 alarm한다. 60초 stale rank는 NCCL의 5분 timeout이 발동하기 전의 straggler다. OLMo 3는 *"continuous batching과 threading 작업이 RL training을 약 4배 더 효율적으로 만들었다"*([[olmo-3]])고 보고한다. 같은 threading infrastructure가 heartbeat thread의 자리다.

---

## 6. RL 전용: 엔트로피 붕괴와 reward explosion

두 가지 RL 전용 실패는 pretraining이나 SFT에 나타나지 않기 때문에 별도 절이 필요하다. 둘 다 손실 곡선은 건강하다. 일반적인 §2 트리는 실패한다.

**Entropy collapse.** [[entropy-collapse-ppo]]:

> *"Per-token entropy `H(π)`가 몇백 번의 PPO update 안에 ~2-3 nats에서 0.1 nats 아래로 떨어지고, reward가 plateau에 머물며, rollout이 반복적으로 변한다."*

Fingerprint는 entropy의 점진적 감소가 아니라 *갑작스러운 변곡*이다. 처음 200 update에서 reward가 상승하는 속도보다 entropy가 더 빠르게 떨어지면 policy는 수렴하는 것이 아니라 붕괴하고 있다. [[openrlhf-entropy-debugging]]은 community-standard triage를 정확히 제시한다.

> *"(1) KL-to-reference term이 켜져 있고 finite인지 확인하고, (2) rollout temperature를 0.1-0.2 올리고, (3) entropy coefficient를 한 자릿수 올리고, (4) advantage normalization이 batch별 zero-mean unit-var인지 확인하고, (5) 그다음에야 reward signal을 의심하라."*

Advantage-norm ON/OFF default는 반복되는 footgun이다. OpenRLHF와 verl은 기본 ON이고, TRL은 기본 OFF다([[openrlhf-entropy-debugging]]). 프로젝트 중간의 framework swap은 normalization을 조용히 disable하고, entropy를 0으로 밀어 넣으며, 왜 reward가 plateau됐는지 궁금하게 만들 수 있다. 위의 check 순서는 probability가 아니라 cost 기준으로 정렬되어 있다.

**Reward explosion / PPO ratio의 NaN.** [[openrlhf-entropy-debugging]]: *"PPO ratio의 `NaN` → 매우 공격적인 update이므로 LR과 clip range를 낮춰라."* PPO ratio `r = π(a|s)/π_old(a|s)`는 `π_old`가 어떤 token에서 underflow하여 0이 되었는데 `π`는 여전히 mass를 부여할 때 NaN이 된다. Mitigation: division 전에 `π_old`를 `exp(-50)`에서 clamp하고, 매 스텝 *pre-clip PPO ratio*를 로그하라. 어떤 token에서든 ratio가 10을 넘으면 clip ε = 0.2 regime에 대해 out-of-distribution이므로 skip-step을 trigger해야 한다.

**Entropy dashboard.** 모든 RL 실행에서 최소한 살아남을 수 있는 로그 표면은 다음이다.

| Metric | Cadence | Alarm condition |
|---|---|---|
| per-token entropy | every update | `H < 0.1` 또는 갑작스러운 변곡 |
| KL(π ‖ π_ref) | every update | target을 지나 단조 상승 |
| PPO ratio mean / max | every update | max > 10 |
| advantage std | every batch | std < 1e-4 |
| response-length histogram | every 50 updates | bimodal 또는 mean diverging |
| clipped-fraction | every update | > 0.5 |

이것들은 OpenRLHF, verl, TRL이 기본으로 노출하는 metric들이다. 세 개의 독립 framework가 같은 표면으로 수렴했다는 사실은 이 표면이 preference가 아니라 minimum이라는 증거다([[openrlhf-entropy-debugging]]).

---

## 7. Silent-failure checklist — 모든 새 trainer 상단에 붙일 한 페이지

위 모든 절과 Karpathy의 maxim-list([[karpathy-training-neural-net-recipe]])에서 증류했다. 이것들을 unit test 또는 inline assert로 실행하라. 각각은 실제 실행을 죽인 버그의 증류된 형태다.

```python
# --- one-time, pre-run ---
assert initial_loss == pytest.approx(math.log(vocab_size), rel=0.02)  # Karpathy's init check
overfit_single_batch_to_near_zero(model, batch, steps=200)            # Karpathy's pipeline check

# --- every step ---
assert torch.isfinite(loss), f"non-finite loss at step {step}"
active = (labels != -100).sum().item()
assert active > 0, f"zero-active-token batch at step {step}"
assert grad_norm < cfg.hard_ceiling, f"runaway grad_norm {grad_norm}"     # e.g. 1000× clip threshold

# --- periodic (every N=100) ---
assert packed_vs_unpacked_max_diff(model, batch) < 1e-4                   # §4b leak check
assert all_ranks_heartbeat_within(60)                                     # §5 hang guard
assert abs(embed_norm - embed_norm_prev) / embed_norm_prev < 1e-4         # §1a logit drift

# --- on resume (ch-06) ---
assert bit_exact_resume_loss_delta(ckpt, steps=1) < 1e-6
assert scheduler.last_epoch == saved_step
```

목록이 짧은 데에는 이유가 있다. 각 assertion은 step time의 << 1%만 들고, 각 assertion은 적어도 한 연구실에서 적어도 1 person-week를 태운 실패 모드를 잡는다. 경제성은 명확하다.

---

## 연결과 다음 내용

- **[[gradient-clipping]] / ch-01** — `pre_clip_grad_norm`은 이 장의 모든 warning signal 중 가장 이르다.
- **[[mixed-precision]] / ch-02** — bf16 vs fp16은 어떤 NaN mode가 보이는지 자체를 지배한다. fp32 master weight는 baseline safety다.
- **[[adam]] / ch-01** — `eps` placement와 `v̂` underflow, optimizer-step NaN surface.
- **[[loss-masking-prompt]] / ch-04** — §4a의 off-by-one surface.
- **[[sequence-packing]] / ch-04** — §4b의 cross-sample attention leak.
- **[[fsdp-sft]] / ch-05** — §5의 hang을 가능하게 만드는 collective topology.
- **[[karpathy-training-neural-net-recipe]]** — 구성 원칙. 여기의 모든 절은 "neural net training fails silently"의 사례다.
- **[[olmo-2]] / [[olmo-3]] / [[llama-3]]** — 세 개의 frontier-scale incident log. 이 장의 실패들에 대한 engineering response.
- **[[openrlhf-entropy-debugging]] / [[entropy-collapse-ppo]]** — §6의 RL-specific extension.
- **ch-06 (checkpointing)** — §1, §2의 resume-time subset(scaler drop, LR off-by-one, data-iter desync).
- **ch-08 (lab)** — 필수 첫 산출물: trainer에 대해 §7의 assertion을 정확히 열거하는 failure-mode-checklist.md.

## 더 읽을거리

- [[gradient-clipping]] — Pascanu 2013; 100배 pre-spike signal과 FSDP global-norm pitfall.
- [[mixed-precision]] — Micikevicius 2017; bf16이 7 mantissa bit 비용으로 대부분의 §1 failure를 제거하는 이유.
- [[adam]] — Kingma 2014 / Loshchilov 2017; fp16 아래 `eps` placement와 `v̂` underflow.
- [[loss-masking-prompt]] — Shi 2024; response-only loss와 multi-turn mask rule.
- [[sequence-packing]] — Krell 2021; `cu_seqlens`와 block-diagonal invariant.
- [[fsdp-sft]] — Zhao 2023; 깨지면 §5 hang을 일으키는 AllGather/ReduceScatter topology.
- [[karpathy-training-neural-net-recipe]] — Karpathy 2019; `ln(V)` init check와 "overfit a single batch."
- [[olmo-2]] — 세 층의 spike-mitigation stack, QK-Norm + Z-loss logit-overflow defense.
- [[olmo-3]] — 1,024-H100 pretraining incident surface, staged model flow가 §5 hang risk를 곱하는 이유.
- [[llama-3]] — 15.6T-token 405B run; chosen-logprob-collapse fix로서의 DPO NLL stabilizer.
- [[openrlhf-entropy-debugging]] — OpenRLHF / verl / TRL 전반에서 수렴한 framework-level entropy triage.
- [[entropy-collapse-ppo]] — Andrychowicz 2020 + LLM-RL derivative; sudden-inflection fingerprint.

## 동반 시각화

**[figures/failure-modes-tree.html](figures/failure-modes-tree.html)** — interactive diagnostic tree. 증상(Loss NaN, Loss Spike, Loss Plateau, NCCL Hang, Grad Clip Triggers Every Step, Entropy Collapse)을 클릭하면, 어떤 logged metric이 진단을 확인하는지, 어떤 불변식이 깨졌는지, 어떤 fix가 적용되는지, 어떤 log가 더 일찍 잡았을지를 decision branch로 안내한다. 이 tree는 §2 flowchart를 tactile하게 만든 것이다. 실제 incident에서 잘못된 branch의 비용은 cluster-hour로 측정되므로, diagnostic ordering이 reflex가 될 때까지 연습하는 데 사용하라.
