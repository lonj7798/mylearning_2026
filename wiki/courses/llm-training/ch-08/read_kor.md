<!-- chapter: ch-08
     track: foundations
     kind: lab
     title: Lab — Systems Tour and Minimal Trainer
     deps: [ch-07]
     sources: [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[sequence-packing]], [[loss-masking-prompt]], [[fsdp-sft]], [[mixed-precision]], [[gradient-clipping]], [[lr-schedules]], [[olmo-2]], [[karpathy-training-neural-net-recipe]]
     figures: figures/trainer-map.html
-->

# 8장 — 랩: 시스템 둘러보기와 최소 Trainer

> **핵심 통찰.** Masking이 적용되는 줄, grad-norm이 계산되는 줄, optimizer가 실제로 step하는 줄을 가리킬 수 있기 전까지는 trainer를 이해한 것이 아니다. 내가 본 모든 production post-training 실패는 *누군가 그 세 줄 중 하나에서 trainer가 올바른 일을 한다고 가정했고 확인하지 않았다*는 것으로 환원된다. 이 장은 그 가정을 닫는다.
>
> **가이드라인.** TRL의 `SFTTrainer`를 reference로 고르고, 처음부터 끝까지 읽은 뒤, 각 Track-1 개념(ch-01..ch-07)을 특정 함수에 매핑하라. 재현 가능한 실행을 만들라. 최소한 single-GPU 125M, 가능하다면 8×H100에서 7B로 실행하라. 그리고 다음에 계측할 세 가지 silent failure 지점을 열거하는 한 페이지짜리 failure-mode memo를 제출하라.

---

## 목표

이 장을 마치면 동료가 재현할 수 있는 세 가지 artifact가 있어야 한다.

1. **Map.** 1-7장의 모든 Track-1 개념이 TRL의 `SFTTrainer` 안의 구체적인 call에 고정되어 있다. "trainer가 처리한다"는 식의 얼버무림은 없다.
2. **Run.** 작동하는 SFT job. 최소 한 번의 optimizer step, 권장 100 step. FSDP FULL_SHARD(또는 한 node의 DDP), bf16, packed sequence, response-only masking, `max_grad_norm = 1.0`을 사용한다.
3. **Memo.** 조용히 실패할 가능성이 가장 높은 세 곳, 계측한 것, 다음에 계측할 것을 나열한 한 페이지짜리 `failure-mode-checklist.md`.

랩의 규율은 Karpathy의 것([[karpathy-training-neural-net-recipe]])이다. *"단일 배치를 overfit할 수 없다면 training set도 overfit할 수 없다."* Multi-GPU, dataset mixing, evaluation을 건드리기 **전에** 한 배치를 near-zero loss까지 overfit하라. One-batch overfit이 수렴하지 않으면 멈추고 디버그하라. 세 개의 silent-failure line 중 하나가 틀렸다.

---

## Trainer 선택

Outline은 네 가지 선택지를 제시한다. TRL의 `SFTTrainer`를 고르라. 이유는 다음과 같다.

- **가장 작은 end-to-end reference.** 하나의 class(`trl.SFTTrainer`)가 HF `Trainer`를 확장하고 packing, response-only masking, Accelerate/FSDP, training loop를 약 1k line 안에서 배선한다. `torchtune`은 더 크고 distributed-first다. `nanotron`은 pretraining-oriented다. HF `examples/` script들은 파일 여러 개에 흩어져 있으며 masking story가 없다.
- **인용된 upstream.** [[hf-alignment-handbook]]은 이것으로 Zephyr-7B를 만들고, [[allenai-tulu-sft-recipe]]는 같은 loss surface를 가진 fork(`open-instruct`)를 사용한다. 여러분은 두 open-source SOTA chat model의 수치적 동작을 만든 바로 그 코드를 읽는 것이다.
- **HF `Trainer`를 확장한다.** Optimizer construction, LR scheduler, checkpointing(ch-06)은 parent class에 있다. 두 stack이 모두 보이는 하나의 object를 얻는다.
- **세 개의 silent-failure line이 한 repo 안에 있다.** Masking(`trl/trainer/`의 `DataCollatorWithPacking` / `DataCollatorForCompletionOnlyLM`), clipping(`Trainer._maybe_log_save_evaluate`와 `accelerator.clip_grad_norm_`에서 상속), optimizer step(`transformers/trainer.py`의 `training_step`). 세 가지 모두를 한 오후에 읽을 수 있다.

다른 선택도 정당하다. 다음 장의 RL이 연속적으로 느껴지길 원한다면 장기적으로 `torchtune`이 더 나은 선택일 수도 있다. 이 랩에서는 TRL에 머물라.

---

## Full-budget path

Target: 8 × H100(80 GB), 7B base model, 50K-prompt subset에서 약 1 GPU-hour SFT.

**Model.** `meta-llama/Llama-3.1-8B` 또는 `Qwen2.5-7B` base. 랩에서는 선택이 중요하지 않다. trainer는 동일하다.

**Data.** 단일 source에서 50K prompt를 가져오라. UltraChat-200K-filtered 또는 Tülu-3-SFT-mix([[allenai-tulu-sft-recipe]])가 예다. 랩에서는 source를 mix하지 **마라**. Mixing은 debug surface를 하나 더 추가한다. `tokenizer.apply_chat_template`으로 model의 chat template을 적용하라. `packing=True`로 4096 token에 pack하라. Prompt를 mask하라. `train_on_response_only=True`.

**Config.** 검증된 Zephyr-7B recipe([[hf-alignment-handbook]])를 인용하면:

```python
# attested from hf-alignment-handbook.md lines 70-78
from trl import SFTTrainer, SFTConfig
config = SFTConfig(
    packing=True,
    max_seq_length=2048,             # raise to 4096 for Llama-3.1
    train_on_response_only=True,     # masks user tokens automatically
    dataset_kwargs={"add_special_tokens": False},
    # hparams from the handbook's Zephyr table
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    max_grad_norm=1.0,               # ch-01 clipping default
    gradient_checkpointing=True,
    # FSDP wiring (attested: FULL_SHARD per [[fsdp-sft]] §"Typical SFT recipe")
    fsdp="full_shard auto_wrap",
    fsdp_config={"transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]},
)
```

8개 GPU에서는 `accelerate launch --config_file fsdp.yaml`로 실행하라. 7B + packed 4096의 깨끗한 100-step 실행은 bf16에서 8×H100 기준 약 15분이 걸린다.

**Full-budget path의 acceptance.** 한 optimizer step이 OOM 없이 완료된다. 제대로 초기화된 base model이라면 step 1 loss는 대략 `ln(V)` ≈ `ln(128k)` = 11.76이다. `pre_clip_grad_norm`은 O(1)이다. 이것이 O(1e2)라면 LR이 너무 높거나, warmup이 너무 짧거나, masking이 어긋나 있다.

---

## Resource-constrained path

Target: 1 × GPU(어떤 것이든 ≥ 16 GB), 또는 인내심 있는 CPU.

**Model.** `HuggingFaceTB/SmolLM-135M` 또는 `Qwen2.5-0.5B`. 135M은 8 GB에서 bf16 fine-tune이 가능하다. 500M은 `gradient_checkpointing=True`로 16 GB에 들어간다.

**Data.** Alpaca-cleaned에서 2K prompt, 또는 UltraChat의 2K slice. Overfit하기에 충분하다. 목표는 유용한 model이 **아니다**. 읽기 쉬운 trainer가 목표다.

**Full-budget 대비 config 변경.** `packing=True`, `train_on_response_only=True`, `bf16=True`, `max_grad_norm=1.0`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.1`은 유지하라. FSDP를 DDP(`--num_processes=1`)로 낮추거나, sharding path를 연습하고 싶다면 FSDP-1-node를 사용하라. Pipeline/tensor parallel은 완전히 건너뛰라. `max_seq_length=1024`, `gradient_accumulation_steps=4`를 설정하라.

**Karpathy의 one-batch check.** 2K-prompt 실행 전에 *하나의* batch를 잡고, `num_train_epochs=200`으로 설정한 뒤 loss가 200 step 안에 ~11에서 < 0.1로 떨어지는지 확인하라. 그렇지 않다면 masking / tokenization / packing / chat-template 중 하나가 틀렸다. 실제 실행 전에 고쳐라.

**Memo requirement는 변하지 않는다.** CPU에서 135M을 실행해도 8xH100에서 7B를 실행할 때와 같은 세 개의 silent-failure line을 연습한다. 여러분이 사는 것은 model이 아니라 memo다.

---

## Code를 개념에 매핑하기(§1..§7)

이 절이 하중을 지탱한다. 모든 Track-1 개념에 대해 TRL 안의 concrete call을 고정하라. Companion HTML([figures/trainer-map.html](figures/trainer-map.html))은 같은 mapping을 clickable call-graph로 렌더링한다. 이 목록과 함께 열어두라.

### §1 — Tokenization + chat template (ch-04)

- `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)` — `SFTTrainer._prepare_dataset`에서 사용된다.
- **Silent failure:** 잘못된 template. [[hf-alignment-handbook]]은 명시적이다. *"packed batch를 decode해 chat template을 검증하라. Template mismatch는 #1 silent bug다."* 학습 전에 one-shot script에서 `tokenizer.decode(batch["input_ids"][0])`를 decode해야 한다. special token(`<|im_start|>`, `<|eot_id|>` 등)이 존재하고 모호하지 않은지 눈으로 확인하라.

### §2 — Packing (ch-04 / [[sequence-packing]])

- `trl/trainer/sft_trainer.py`의 `DataCollatorWithPacking` / `ConstantLengthDataset` — `packing=True`로 활성화된다.
- `cu_seqlens`(cumulative sequence-start offsets)가 있는 `input_ids`를 만들고, sub-sequence마다 `position_ids`를 reset한다([[sequence-packing]] §Mechanics).
- Attention은 `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)`를 사용한다. O(L_max²)가 아니라 O(Σ L_i) memory를 쓰는 block-diagonal causal attention이다.
- **Silent failure:** `cu_seqlens`가 attention으로 전달되지 않아 token들이 sub-sequence를 가로질러 attend한다. Unit test는 §Deliverables에 있다.

### §3 — Loss masking (ch-04 / [[loss-masking-prompt]])

- `train_on_response_only=True`이면 `DataCollatorForCompletionOnlyLM`이 각 sub-sequence를 따라가 instruction-template boundary를 찾고 `labels[:response_start] = -100`으로 설정한다.
- Canonical masking snippet([[loss-masking-prompt]] §Implementation):

```python
# attested from loss-masking-prompt.md lines 46-52
labels = input_ids.clone()
labels[:prompt_len] = -100            # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

- **Silent failure:** multi-turn mismatch. 이전 assistant turn도 mask되어야 한다([[loss-masking-prompt]] §Multi-turn). 맨 처음 user turn만 mask하면 모델은 과거 assistant output에 gradient-match한다. SFT가 "자기 자신을 복사하기" task로 변한다.

### §4 — Mixed precision (ch-02 / [[mixed-precision]])

- `SFTConfig`의 `bf16=True` → HF `Trainer`가 forward를 `torch.autocast("cuda", dtype=torch.bfloat16)`로 감싼다.
- Optimizer state는 fp32로 유지된다(AdamW는 fp32 `m`, `v`, master copy를 유지한다. [[fsdp-sft]]는 이것이 FSDP `MixedPrecision` default임을 확인한다).
- **Silent failure:** loss를 bf16으로 logging. `loss.item()`은 upcast하지만, `loss.to(torch.bfloat16).cpu().numpy()`를 수행하는 custom callback은 quantized-jaggy curve를 만든다. 항상 fp32로 log하라.

### §5 — Gradient clipping (ch-01 / [[gradient-clipping]])

- `max_grad_norm=1.0` → HF `Trainer.training_step`이 `self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)`를 호출한다. FSDP 아래에서 Accelerate는 global norm을 계산한 뒤 rescale하는 sharded-aware version인 `FSDP.clip_grad_norm_`으로 dispatch한다([[fsdp-sft]] §"Distributed-training pitfall").
- **Silent failure:** parameter별로 `clip_grad_norm_(p, c)`를 loop하는 custom code path는 optimizer를 작은 tensor 쪽으로 bias한다([[gradient-clipping]] §"Per-tensor norm clip"). Clipping을 monkey-patch한다면 이것이 그 버그다.

### §6 — Optimizer step, scheduler, clipping ordering (ch-01/ch-02/ch-03)

[[mixed-precision]]과 [[gradient-clipping]]을 인용하면 반드시 지켜야 하는 ordering은 다음이다.

```
loss.backward()           # produces grads (scaled by S if fp16)
[unscale_ if fp16]
clip_grad_norm_(params, max_norm)           # direction-preserving, global
optimizer.step()
scheduler.step()                            # AFTER optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

- HF `Trainer.training_step`은 이것을 구현한다. bf16에서는 `unscale_`이 없다. FSDP 아래에서 clip은 sharded-aware다.
- **Silent failure:** `optimizer.step()` *전에* `scheduler.step()`을 호출함. 첫 step이 step-(k+1)의 LR로 실행되고, cosine phase가 영구적으로 하나 어긋난다([[ch-06]] §5.3 참조).

### §7 — LR schedule (ch-03 / [[lr-schedules]])

- `lr_scheduler_type="cosine"` + `warmup_ratio=0.1` → `transformers.optimization.get_cosine_schedule_with_warmup`. 공식([[lr-schedules]] §Technical Details): `lr(t) = min_lr + 0.5*(peak_lr - min_lr)*(1 + cos(pi*(t - warmup)/(T - warmup)))`.
- **Silent failure:** total `T`가 `num_train_epochs * steps_per_epoch`에서 계산되지만, packing은 pack ratio에 의해 `steps_per_epoch`를 바꾼다([[allenai-tulu-sft-recipe]] 기준 약 2.5배). Scheduler의 `T`가 packing 적용 전에 설정되면 cosine은 `T/2.5` step에서 끝나고 마지막 60%의 training은 `min_lr`에서 돈다. 치명적이지는 않지만, 측정 가능하게 나쁘다.

---

## Deliverables checklist

아래의 모든 것은 여러분의 lab output directory에 둔다. 이 wiki가 **아니다**. Gist, run directory, 또는 첨부 PR로 제출하라.

- [ ] `run.sh` — `accelerate launch` invocation, 옆에 `fsdp.yaml`(또는 single-GPU)을 둔다.
- [ ] `sft_config.py` — 위의 `SFTConfig` 그대로, hyperparameter delta를 주석으로 표시.
- [ ] `chat_template_check.py` — packed batch를 decode하고 첫 200 token을 출력한다. 한 번 실행하고 output을 commit한다.
- [ ] `masking_unit_test.py` — 아래 참조. 반드시 pass해야 한다.
- [ ] `packing_unit_test.py` — 아래 참조. 반드시 pass해야 한다.
- [ ] `overfit_one_batch.py` — 한 batch에서 200 step, loss plot. loss < 0.1에 도달해야 한다.
- [ ] `failure-mode-checklist.md` — memo, 1 page, 아래 구조.

**Masking unit test([[loss-masking-prompt]]에서).** 하나의 batch를 구성하고 `loss.backward()`를 실행한 뒤, `embed_tokens.weight.grad`가 prompt-token row에서 zero mass인지 확인하라. 한 줄로:

```python
prompt_ids = batch["input_ids"][labels == -100]
assert model.get_input_embeddings().weight.grad[prompt_ids].abs().sum() == 0.0
```

이것이 실패하면 masking이 잘못되었다. Full run으로 진행하지 **마라**.

**Packing unit test([[sequence-packing]]에서).** 같은 batch를 두 번 실행하라. 한 번은 `cu_seqlens`가 attention으로 전달된 packed 상태로, 한 번은 `cu_seqlens`를 `[0, L_total]`로 바꾼 packed 상태로 실행한다(즉 block-diagonal mask가 없는 cross-contamination case). Loss는 > 1e-3만큼 달라야 한다. 그렇지 않다면 packing이 `cu_seqlens`를 전달하지 않거나 batch에 sub-sequence가 하나뿐인 것이다. batch를 다시 만들라.

**Failure-mode memo(구조).** 한 페이지, 네 절:

1. *Picks.* 이 trainer에서 가장 가능성 높은 세 가지 silent failure. 내 선택은 (a) base model tokenizer와의 chat-template mismatch, (b) `cu_seqlens`가 attention에 연결되지 않아 생기는 cross-contamination, (c) custom callback의 `scheduler.step()` 순서다. 여러분의 선택은 다를 수 있다. 정당화하라.
2. *What you instrumented.* 매 step 로그한 metric 목록: `loss`, `pre_clip_grad_norm`, `lr`, `tokens/sec`. Packed-batch decode의 output path, masking unit test output.
3. *What you would instrument next.* Per-shard loss breakdown([[ch-06]] §4), checkpoint별 embedding-norm([[olmo-2]]), 알려진-good batch와 byte-identical한 canary set의 loss(resume across data-loader drift를 감지).
4. *Reproduction recipe.* `git rev-parse HEAD`, `pip freeze`, 정확한 `accelerate launch` line, total wall-clock, final loss.

---

## Acceptance criteria

Hard gate, 순서대로. 건너뛰지 마라.

1. `chat_template_check.py` output을 손으로 검증했다. 모든 special token이 올바르게 렌더링되고, stray `<s>` 또는 doubled BOS가 없다.
2. `masking_unit_test.py`가 pass한다. Prompt-token embedding gradient가 정확히 zero다.
3. `packing_unit_test.py`가 pass한다. Cross-contamination loss delta > 1e-3.
4. `overfit_one_batch.py`가 batch of 1에서 200 step 안에 loss < 0.1에 도달한다.
5. 100-step(또는 ≥ 1 optimizer-step) real run이 OOM 없이 완료되고, `pre_clip_grad_norm`은 < 10으로 유지되며, step-1 loss는 `ln(V)`의 20% 이내다.
6. `failure-mode-checklist.md`가 존재하고, 한 페이지이며, 세 가지 구별되는 silent failure를 각각 특정 metric과 함께 나열한다.

어떤 gate라도 실패하면 lab은 incomplete다. 그 이전 gate로 돌아가 디버그하라. 이것이 Karpathy([[karpathy-training-neural-net-recipe]])의 "이번에는 다르다며 step을 건너뛰지 말라"는 원칙이다.

---

## 연결

- **ch-04 (SFT mechanics)** — packing + masking + chat-template surface. 이 lab은 그 장의 claim에 대한 unit-test다.
- **ch-05 (FSDP)** — `fsdp=full_shard` + `LlamaDecoderLayer` wrap policy. 같은 code path, 다른 knob.
- **ch-06 (checkpointing)** — 100-step run은 DCP checkpoint를 만들어야 한다. Resume-bit-exact는 full-budget path의 stretch gate다.
- **ch-07 (silent failure catalog)** — 이 lab의 memo는 ch-07의 더 넓은 catalog를 장별로 구체화한 것이다.
- **ch-09 (first real end-to-end SFT)** — 이 trainer를 default로 활용한다. 아래 모든 것은 세 unit test가 pass한다고 가정한다.
- **Track 2 (synthetic data) / Track 3 (SFT-at-scale) / Track 4 (RL)** — 세 track 모두 이 trainer를 상속한다. Track 4의 RL은 loss function을 바꾸지만 masking, packing, clipping, optimizer-step ordering은 동일하게 유지한다.

## 더 읽을거리

- [[hf-alignment-handbook]] — reference `SFTConfig`와 FSDP wiring, lab-scale default의 authoritative source.
- [[allenai-tulu-sft-recipe]] — 939K-prompt / 8B-70B scale에서 바뀌는 것(LR down, epochs up, 70B에서 HYBRID_SHARD), memo extension.
- [[sequence-packing]] — Krell 2021; `cu_seqlens` contract, cross-contamination unit-test idea.
- [[loss-masking-prompt]] — Shi 2024 + Alpaca/InstructGPT canon; response-only loss, multi-turn masking.
- [[fsdp-sft]] — Zhao 2023; `FULL_SHARD` + `MixedPrecision` + `clip_grad_norm_` sharded contract.
- [[mixed-precision]] — Micikevicius 2017; bf16 default, fp32 accumulation rule.
- [[gradient-clipping]] — Pascanu 2013; global-norm clip 1.0.
- [[lr-schedules]] — cosine + warmup 10%, lab default.
- [[olmo-2]] — reference 2025 post-training number. Scale up할 경우 sanity-check target으로 사용하라.
- [[karpathy-training-neural-net-recipe]] — lab-memo tradition. "단일 배치 overfit"이 gate 4다.

## 동반 시각화

**[figures/trainer-map.html](figures/trainer-map.html)** — TRL `SFTTrainer`의 interactive call-graph. 어떤 box(`apply_chat_template`, `DataCollatorWithPacking`, `DataCollatorForCompletionOnlyLM`, `FSDP wrap`, `autocast bf16`, `clip_grad_norm_`, `optimizer.step`, `scheduler.step`)에 hover하면 그것이 구현하는 concept-chapter(ch-01..ch-07)와 한 줄짜리 silent-failure mode를 볼 수 있다. Box를 클릭하면 detail panel이 고정된다. 각 node의 colour band는 그것이 세 가지 most-likely-to-fail-silently line 중 어디에 놓이는지 나타낸다. Legend는 memo의 §1 short-list template 역할도 한다.
