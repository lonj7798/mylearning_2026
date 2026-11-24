# Chapter 57 — TRL 내부: SFT, DPO, GRPO, Distillation Trainer

> **핵심 통찰.** TRL의 post-training 동작은 기본값이 논문 설정과도 다르고 릴리스 사이에서도 안정적이지 않은 configuration field가 결정한다. commit `a04ffd3`(2026-09-14)에서 GRPO 기본값은 DeepSeekMath 목적함수가 아니라 `beta=0.0`인 `loss_type="dapo"`다([[trl-repo-a04ffd3]], `grpo_config.py` L796, L676; loss 코드는 [[trl-grpo]]). `SFTConfig`는 `assistant_only_loss=True`가 아니면 렌더한 대화 전체에서 학습한다([[trl-repo-a04ffd3]], `sft_config.py` L271). `DPOConfig.loss_type`은 `loss_weights`로 결합하는 이름 붙은 손실 15개의 **list**다([[trl-repo-a04ffd3]], `dpo_config.py` L81-L88). `PPOTrainer`는 더는 존재하지 않는다. PR #7020이 2026-09-04 삭제했고 v1.13.0에 릴리스됐다([[trl-ppo]] 검증). 이 field 각각은 어느 토큰이 얼마나 gradient를 받는지 바꾸므로 post-training 실행이 보존하는 일반 능력의 양도 바꾼다.
>
> **지침.** TRL에서 공개 결과를 재현할 때는 commit을 pin하고 `loss_type`, `beta`, `scale_rewards`를 명시하라. 기본값이 [[grpo]], [[dpo]]의 목적함수와 다르기 때문이다. 대화 데이터셋을 fine-tuning할 때 `assistant_only_loss=True`로 두고 template의 end-of-turn token이 mask 안에 있는지 검증하라. 없으면 TRL은 예외 대신 warning만 기록한다([[trl-repo-a04ffd3]], `sft_trainer.py` L1271-L1277). 여러 reward function의 수치 scale이 다르면 `multi_objective_aggregation="normalize_then_sum"`을 써라. 기본값인 `"sum_then_normalize"`은 raw spread가 큰 reward가 advantage 부호를 정하게 한다(§4.3 수치 예). fine-tuning 도메인 밖 능력 보존이 우선이면 full fine-tuning 대신 rank 256, α = 2r LoRA를 써라. Llama-2-7B code instruction fine-tuning에서 full fine-tuning은 forgetting metric 0.414, LoRA r=64는 0.509였다([[lora-learns-less-forgets-less]] §4.2, 표 S6). code continued pretraining에서 target 도메인 정확도가 우선이면 full fine-tuning을 써라. 그 regime에서는 LoRA가 시험한 모든 rank에서 뒤처졌다(§4.1).

---

## 이 챕터가 범용 모델에 중요한 이유

파이프라인 위치는 사전 학습 → 중간 학습 → **SFT → 선호 최적화 → RL** → 평가다. [[ch-55]]는 verl, [[ch-56]]은 OpenRLHF를 다뤘다. TRL은 이 그룹의 세 번째 프레임워크이며 trainer가 `transformers.Trainer`의 subclass이므로 distributed layer는 Ray가 아니라 `accelerate`다.

프레임워크 챕터의 일반성 질문은 throughput이 아니다. **오류를 내지 않고 base 모델 breadth가 실행 뒤 얼마나 남는지를 결정하는 configuration field가 무엇인가**다. 측정 가능한 문제 세 가지가 따른다.

1. **Loss-mask 및 template 과적합.** loss가 렌더한 chat의 user/system token을 덮으면 모델은 inference에서 낼 필요가 없는 turn header를 생성하도록 학습한다. assistant turn end-of-turn token이 mask 밖이면 serving configuration이 기대하는 stop token을 내는 법을 배우지 않는다. 둘 다 silent다. 학습 loss curve는 정상처럼 보인다.
2. **논문과 프레임워크 사이 objective drift.** TRL 기본값으로 구성한 실행은 trainer 이름에 붙은 논문과 다른 목적함수를 최적화한다. `beta=0.0`은 reference model 자체가 없다는 뜻이므로, 켜지지 않으면 [[rlhf-instructgpt]]가 의존하는 pre-RL 정책 KL anchor가 없다.
3. **단일 reward 아래 좁아짐.** 한 도메인의 검증 가능 reward 하나는 정책을 좁힌다. TRL의 multi-reward/multi-domain 표면(`reward_funcs`, `reward_weights`, `multi_objective_aggregation`)과 평가 hook(`eval_dataset`, `num_generations_eval`, `LogCompletionsCallback`)은 breadth를 학습 뒤가 아니라 학습 중 측정하게 하는 API 부분이다.

구성은 다음과 같다. pin한 commit의 repository(§1); SFT mask, packing, template(§2); DPO 계열과 PEFT reference trick(§3); GRPO loss aggregation, KL estimator, multi-reward 학습(§4); online/asynchronous trainer(§5); distillation trainer(§6); 측정된 LoRA 대 full fine-tuning(§7); in-loop 평가와 merging(§8).

**이 챕터의 모든 내용은 commit 하나, `a04ffd337108285002df29d9ae73ef7e855e03f1`, 2026-09-14의 `main`에서 읽는다.** 파일과 line 참조는 그 tree에 대한 것이다. TRL API는 최근 minor release마다 바뀌었으므로 commit 없는 line 참조는 점검할 수 없다.

---

## §1 pin한 commit의 repository

### §1.1 tree에 들어 있는 것

`a04ffd3`에서 `trl/` package는 supported 표면과 experimental 표면으로 나뉜다. 전체 listing은 [[trl-repo-a04ffd3]]에 있고, 여기 중요한 부분은 다음과 같다.

```
trl/trainer/          sft_trainer.py  dpo_trainer.py  grpo_trainer.py
                      rloo_trainer.py  kto_trainer.py  reward_trainer.py
                      distillation_trainer.py  callbacks.py  utils.py
trl/experimental/     async_grpo/  async_distillation/  server_distillation/
                      online_dpo/  nash_md/  xpo/  gkd/  minillm/  gold/
                      sdft/  sdpo/  ssd/  iw_opd/  bco/  cpo/  orpo/  kto/
                      prm/  a2po/  gmpo/  gspo_token/  tpo/  harbor/
                      merge_model_callback.py
trl/losses/           dpo_loss.py  grpo_loss.py  jsd_loss.py
                      fused_linear_distillation.py  fused_linear_ppo.py
trl/rewards/          accuracy_rewards.py  format_rewards.py  other_rewards.py
trl/generation/       vllm_client.py  vllm_generation.py
```

listing에서 직접 따라오는, 오래된 TRL 설명과 모순하는 사실 세 가지가 있다.

1. **`ppo_trainer.py`도 `modeling_value_head.py`도 없다.** `trl/models/`에는 `activation_offloading.py`, `utils.py`만 있다. PR #7020(commit `700b845`, 2026-09-04)이 `PPOTrainer`, `PPOConfig`, value-head model을 삭제했고 v1.13.0에 배포됐다([[trl-ppo]] 검증). actor-critic trainer가 필요한 코드는 v1.13.0 이전 TRL release를 pin하고 `trl.experimental.ppo`에서 import해야 한다.
2. **공유 value head가 아닌 별도 value model.** PPOTrainer가 존재했을 때 value function은 `.score` head가 있는 별도 `nn.Module`이었다. `PolicyAndValueWrapper`로 policy와 감싸 `accelerator.prepare` 호출 하나가 둘을 덮게 했다(`a08e713`의 `ppo_trainer.py` L283-L302, [[trl-ppo]]). `AutoModelForCausalLMWithValueHead`를 import하지 않았다.
3. **`ORPO`, `CPO`, `BCO`, `KTO`(experimental copy), `GKD`는 `trl/experimental/` 아래다.** `DPOTrainer`의 `loss_type` 값이 아니라 별도 trainer다. [[hf-dpo-zoo]]는 “모든 variant가 `loss_type` parameter를 가진 `trl.DPOTrainer`로 노출된다”고 하지만, 이 commit에서는 사실이 아니며 §3.1이 고친다.

### §1.2 stable 대 experimental

구분은 quality가 아니라 import path다. `trl/experimental/`은 새 연구 trainer(`a2po`, `gmpo`, `tpo`, `iw_opd`)와 supported surface에서 옮긴 오래된 algorithm(`online_dpo` v0.26.0, PR #4473; `ppo` v0.26.0, PR #4482, 이후 삭제)을 모두 담는다. 실무 결과 두 가지:

- import path는 재현 레시피 일부다. 이 commit에서 `from trl import OnlineDPOTrainer`는 실패하고, 경로는 `from trl.experimental.online_dpo import OnlineDPOTrainer`다([[trl-online-dpo]]).
- audits가 찾은 어떤 표면에도 repository는 compatibility guarantee text를 공개하지 않았다. 그러므로 여기서 “stable”은 versioning 약속이 아니라 “`trl/trainer/` 안”을 뜻한다.

이 commit의 trainer 지도를 보려면 **[figures/trl-stack.html](figures/trl-stack.html)**을 보라. 제거되고 relocated된 trainer도 표시하며, 주어진 algorithm이 오늘 어떤 import path를 갖는지 점검하게 한다.

## §2 SFT: 어떤 토큰이 gradient를 받는가

### §2.1 정의와 측정할 수 있는 문제

Supervised fine-tuning은 target token에 대해 cross-entropy를 최소화한다. 설계에서 결정해야 할 것은 **어느** token을 target으로 둘 것인가이다. 측정 가능한 문제는 다음과 같다. 전체 렌더링 대화로 학습한 모델은 user turn의 header와 본문에도 gradient를 할당한다. 따라서 inference에서 자기 답변 뒤를 계속 이어서, 만들어 낸 user turn을 생성할 수 있다. end-of-turn token이 mask 밖에 있는 모델은 serving configuration이 기대하는 stop token을 내보내도록 학습되지 않는다.

### §2.2 메커니즘: 세 가지 masking 모드

`a04ffd3`의 `SFTConfig`는 masking field 두 개를 노출하며, 그 동작은 dataset shape에 따라 달라진다([[trl-repo-a04ffd3]], `sft_config.py` L96-L105).

- `completion_only_loss` (기본값 `None`): “`None`(기본값)이면 동작은 dataset에 따라 달라진다. prompt-completion dataset에서는 completion에 대해 loss를 계산하고, language modeling dataset에서는 전체 sequence에 대해 계산한다.”
- `assistant_only_loss` (기본값 `False`, L271): “assistant response에 대해서만 loss를 계산하며, conversational dataset에서만 지원된다.”

따라서 `messages` column을 가진 conversational dataset에서 두 field를 모두 기본값으로 두면, **렌더링된 대화의 모든 token이 target**이다. TRL에는 `train_on_response_only`라는 field가 없다. 그 이름은 Unsloth helper에 속한다. `max_seq_length`라는 field는 `max_length`로 대체되었고(기본값 1024, L203), 현재 field는 후자이다.

`assistant_only_loss`는 chat template의 `{% generation %}` marker를 통해 구현된다. `apply_chat_template(..., return_assistant_tokens_mask=True)`는 이를 `assistant_masks` column으로 변환한다(`sft_trainer.py` L1551). tokenizer template에 marker가 없으면 TRL은 training template으로 교체한다.

```python
# trl/trainer/sft_trainer.py @a04ffd3 L1262-L1277
# assistant_only_loss가 활성화되어 있고, 현재 template에 {% generation %} marker가 아직 없다면
# {% generation %} marker가 있는 training chat template으로 교체한다.
if args.assistant_only_loss and not has_generation_markers(processing_class.chat_template):
    self.chat_template = get_training_chat_template(processing_class)
else:
    self.chat_template = None

# template이 generation marker를 정의하고 있어도 assistant의 end-of-turn token을 다음 message에 귀속시킬 수 있다.
# 그러면 해당 token은 assistant mask에서 빠지고, 모델은 stop하도록 학습되지 않는다.
if args.assistant_only_loss and not is_chat_template_stop_token_trained(
    processing_class, chat_template=self.chat_template
):
    logger.warning(
        "The chat template does not include the assistant turn's end-of-turn token in the loss mask; "
        "the model may not learn to stop."
    )
```

이는 **경고(warning)**이지 error가 아니다. marker를 잘못 지정한 template로도 run은 정상적으로 완료되며, 결과 모델은 stop하지 않을 수 있다.

### §2.3 예제: non-assistant token으로 가는 gradient의 양

다음 token count를 갖는 렌더링된 2-round 대화 하나를 생각하자. system 20, user₁ 30, assistant₁ 60, user₂ 25, assistant₂ 80이다. 총 215 token이다.

| 설정 | Target token | inference에서 모델이 절대 생성하지 않는 token에 걸리는 gradient의 비율 |
|---|---|---|
| `assistant_only_loss=False` (기본값, conversational data) | 215 | 75 / 215 = 34.9% |
| `assistant_only_loss=True` | 140 | 0% |

기본 설정에서는 이 예제에서 optimizer의 작업 중 약 3분의 1이 turn header와 user text의 확률을 높이는 데 쓰인다. 기본 설정에서 loss 값은 더 낮다(template token은 예측하기 매우 쉽다). 따라서 loss curve는 이 문제를 드러내지 않는다.

### §2.4 Packing

`packing`(기본값 `False`, L223)은 sequence를 `max_length` 크기의 고정 block으로 묶는다. `packing_strategy`(기본값 `"bfd"`, L230)는 docstring(L80-L82)에 인용된 세 가지 option을 제공한다. `"bfd"`(best-fit decreasing, overflow를 truncate), `"bfd_split"`(best-fit decreasing, overflow sequence를 split), `"wrapped"`(aggressive하며 sequence 중간을 자름)이다. 현재 경로에는 `ConstantLengthDataset`이 없다. strategy `"bfd"`로 `packing`을 켜면, 자체 값과 관계없이 `padding_free`가 활성화된다(L83-L88). 즉 batch가 하나의 연속 sequence로 평탄화되며, example별 attention boundary를 유지하려면 FlashAttention 2 또는 3이 필요하다.

breadth와의 관련성은 다음과 같다. `"wrapped"`는 example을 sequence 중간에서 자른다. 따라서 한 example의 답변이 한 training block에서 시작하고 prompt가 전혀 없는 다른 block에서 끝날 수 있다. instruction data에서는 instruction이 없는 target이 만들어진다. 반면 `"bfd"`는 truncate한다. 따라서 긴 example의 tail을 잃으며, 이는 long-context SFT에서 중요한 실패이다([[ch-32c]]).

### §2.5 조건과 한계

이 library의 어떤 source도 TRL SFT 기본값의 ablation을 보고하지 않는다. Alignment Handbook에 공개된 Zephyr-7B SFT config는 `max_seq_length: 2048`, `learning_rate: 2.0e-05`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.1`, `num_train_epochs: 1`, `per_device_train_batch_size: 16`, `gradient_accumulation_steps: 1`, `bf16: true`를 설정한다. 그러나 `packing`도 response-only masking field도 **어느 것도** 설정하지 않는다([[hf-alignment-handbook]]; `recipes/zephyr-7b-beta/sft/config_full.yaml`, commit `1de1fc9`에서 읽은 config). [[hf-alignment-handbook]] card가 “`packing: true` 및 `train_on_response_only: true`를 유지하라”고 지시하지만, 그 주장은 이 config가 뒷받침하지 않으므로 여기서 바로잡는다.

---

## §3 DPO 계열

### §3.1 `loss_type`은 list이고, 그 list는 오래된 요약이 말하는 내용과 다르다

```python
# trl/trainer/dpo_config.py @a04ffd3 L81-L88 (docstring)
loss_type (`list[str]`, *optional*, defaults to `["sigmoid"]`):
    사용할 loss의 type이다. 가능한 값은 다음과 같다: `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`,
    `'nca_pair'`, `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`, `'aot_unpaired'`, `'apo_zero'`,
    `'apo_down'`, `'discopop'`, `'sft'`, `'sigmoid_norm'`. 여러 loss type을 제공하면,
    `loss_weights`에 지정한 weight를 사용하여 이들을 결합한다.
loss_weights (`list[float]`, *optional*):
    multi-loss 조합에서 쓸 loss weight의 list. ... 예: MPO에서 `[0.8, 0.2, 1.0]`.
```

`'kto'`, `'simpo'`, `'orpo'`, `'cpo'`는 **이 list에 없다**. SimPO는 `CPOTrainer(loss_type="simpo")`다. KTO, ORPO, CPO, BCO는 별도의 trainer이며, 이 commit에서는 `kto_trainer.py`를 제외한 모두가 `trl/experimental/` 아래에 있다. 이는 “TRL implementation consistency” 절에서 단일 `DPOTrainer`가 모든 variant를 노출한다고 주장하는 [[hf-dpo-zoo]]를 바로잡는다.

결합 loop는 이름 붙은 loss들을 가중합한다(`dpo_trainer.py` L1470-L1472).

```python
loss = 0.0
for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
    if loss_type == "sigmoid":
        per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
```

여기서 `delta_score = chosen_scores - rejected_scores`(L1468)이며, [[dpo]]의 log-ratio 차이다. `'sft'` member는 chosen completion에만 적용하는 일반 cross-entropy다(L1581-L1587). 따라서 `loss_type=["sigmoid", "sft"]`, `loss_weights=[1.0, 0.2]`는 positive NLL term으로 anchor를 둔 preference loss다. 이것이 §Negatives가 “positive NLL term으로 anchor를 둔다”고 부르는 메커니즘이다.

### §3.2 PEFT reference-model trick을 정확하게 진술하기

DPO는 frozen reference π_ref 아래의 log-probability를 필요로 한다. PEFT adapter를 사용할 때 TRL은 두 번째 model을 load하지 않는다.

```python
# trl/trainer/dpo_trainer.py @a04ffd3 L919-L924
if ref_model is None:
    if is_peft_model(self.model) or args.precompute_ref_log_probs:
        # PEFT를 사용하면 adapter를 비활성화하여 initial model로 되돌릴 수 있으므로 reference model은 필요 없다.
        # precompute_ref_log_probs가 True이면 training 중 reference model을 memory에 유지할 필요가 없다.
        self.ref_model = None
```

그리고 loss 계산 시(L1412-L1418)는 다음과 같다.

```python
if is_peft_model(model) and self.ref_model is None:
    # PEFT adapter를 학습할 때 reference를 얻는 방식은 setup에 따라 달라진다.
    # - 새 adapter: adapter를 비활성화하면 base model을 얻는다.
    # - 기존 adapter 재학습: initial copy를 "ref"라는 이름으로 load한다.
    model = self.accelerator.unwrap_model(model)
    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
        ref_outputs = self.model(**ref_model_kwargs)
```

절약되는 것은 base weight의 frozen copy 하나이지 forward pass 하나가 아니다. reference forward는 여전히 `torch.no_grad()` 아래에서 실행된다. 두 번째 branch인 `precompute_ref_log_probs`는 preprocessing pass에서 π_ref log-probability를 한 번 계산해 training loop에서 reference forward를 완전히 제거한다. 이는 `sync_ref_model`과 호환되지 않으며, docstring도 PEFT와는 아직 호환되지 않는다고 표시한다(`dpo_config.py` L119-L124).

`sync_ref_model`이 하는 일과 하지 않는 일을 구분하라. 이것은 주기적으로 현재 policy를 reference에 섞는다(TR-DPO, `ref_model_mixup_alpha` 기본값 0.6, `ref_model_sync_steps`). 이것이 없으면 reference는 전체 run 동안 frozen이다. **online DPO**에서도 reference는 frozen이다. on-policy가 되는 것은 sampled pair이지 reference가 아니다([[trl-online-dpo]] Corrections).

---

## §4 GRPO: aggregation, KL, multiple reward

### §4.1 여덟 가지 loss type은 normalizer에서 다르다

`GRPOTrainer._compute_loss`는 per-token surrogate를 공유하고 masked per-token loss를 scalar로 줄이는 방식에서 다른, 여덟 `loss_type` 값을 구현한다([[trl-grpo]], `grpo_trainer.py`의 `a08e713` L2548-L2562).

```python
if self.loss_type in ["grpo", "sapo"]:
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "bnpo":
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
elif self.loss_type in ["cispo", "dapo", "vespo"]:
    normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
    loss = (per_token_loss * mask).sum() / normalizer
```

**계산 예제.** completion이 G = 2개인 group 하나를 보자. completion A는 completion token 4개이고 각 token의 per-token loss는 1.0이다. completion B는 completion token 8개이고 각 token의 per-token loss는 0.0이다. `max_completion_length = 16`이다.

| `loss_type` | 산술식 | Scalar loss |
|---|---|---|
| `"grpo"` | ((4·1.0)/4 + (8·0.0)/8) / 2 | 0.500 |
| `"bnpo"` | (4·1.0 + 8·0.0) / 12 | 0.333 |
| `"dr_grpo"` | (4·1.0 + 8·0.0) / (2 × 16) | 0.125 |

동일한 rollout batch인데도 scalar 세 개는 4배 차이가 난다. `"grpo"` 아래에서는 A의 각 token이 계수 1/(4·2)로 sum에 들어가고 B의 각 token은 1/(8·2)로 들어간다. group 평균 전에 per-sequence mean이 해당 sequence 자신의 길이로 나누기 때문이다. per-sequence normalization은 short completion의 각 token에 더 큰 gradient 비중을 준다. config docstring은 그 결과를 명시한다. `grpo` normalization은 “positive advantage를 가진 더 짧은 completion과 negative advantage를 가진 더 긴 completion을 선호하는 경향이 있다”(`grpo_config.py` L239-L241). `dr_grpo`는 length term을 완전히 제거한다([[dr-grpo]]).

`a04ffd3`에서 위 세 normalizer expression은 바뀌지 않았다. 각각 뒤에 현재 gradient-accumulation step count로 나누는 연산이 있고, `dapo`/`cispo`/`vespo` normalizer는 추가로 `gradient_accumulation_steps / steps_per_generation`로 rescale된다(`grpo_trainer.py` L3246-L3266). 이 factor들은 batch의 모든 completion에 공통이므로, 계산 예제의 상대 weight는 pin한 commit에서도 유지된다.

**기본값은 `"grpo"`가 아니라 `"dapo"`다** (`grpo_config.py` L796, `default="dapo"`). `beta` 기본값은 `0.0`이다(L676). 이 값에서는 reference model이 생성되지 않는다. 그러므로 기본값을 그대로 둔 run은 KL anchor 없이 DAPO normalizer를 최적화한다. 문서는 Open-Reasoner-Zero, Dr. GRPO, DAPO를 인용하여 이것이 의도한 선택임을 정당화한다([[trl-grpo]]). 그러나 이는 [[grpo]] Eq. 3의 objective는 아니다.

### §4.2 KL estimator를 정확히 진술하기

TRL의 GRPO KL term은 `exp(ref − logp) − (ref − logp) − 1`이다(`a04ffd3`의 `grpo_trainer.py` L3193-L3194; `a08e713` L2493-L2497, [[trl-grpo]]). `logr = log π_ref − log π_θ`, `r = exp(logr)`라 두면 `r − logr − 1`이며, Schulman의 **k3** estimator다. [[john-schulman-kl-tricks]]에서 명명한 사실은 다음과 같다.

| Estimator | Formula | Bias | Variance | Sign |
|---|---|---|---|---|
| k1 | `−log r` | unbiased | high | negative일 수 있음 |
| k2 | `0.5 · (log r)²` | **biased** | low | non-negative |
| k3 | `(r − 1) − log r` | unbiased | low | non-negative |

**계산 예제.** 두 token 모두 π_θ에서 sample한다.

- Token 1: log π_θ = −2.0, log π_ref = −2.3 → `logr` = −0.3, `r` = 0.7408. k1 = 0.300, k3 = 0.7408 + 0.300 − 1 = **0.041**.
- Token 2: log π_θ = −2.5, log π_ref = −2.2 → `logr` = +0.3, `r` = 1.3499. k1 = **−0.300**, k3 = 1.3499 − 0.300 − 1 = **0.050**.

두 token에서 mean k1은 0.000이고 mean k3은 0.045다. 단일 sample k1은 KL이 그럴 수 없음에도 negative일 수 있으며, 두 sample은 서로 상쇄된다. k3은 per-sample로 non-negative다. step당 token 수가 적으면 k1 기반 penalty는 일시적으로 policy가 reference에서 멀어지도록 reward할 수 있다.

오래된 PPO trainer는 `kl_estimator`(기본값 `"k1"`)로 k1과 k3 중 선택했고, 그 결과를 loss가 아니라 reward 내부에 적용했다. `non_score_reward = −kl_coef · kl`을 GAE 전에 더했다([[trl-ppo]], `ppo_trainer.py` L775-L781). 따로 log한 `approxkl`은 k2 형식이며, 현재 policy를 **rollout** policy와 비교한다. clip fraction은 별도 metric `pg_clipfrac`다([[trl-ppo]] Corrections). 둘은 서로 다른 diagnostic이므로 하나로 읽으면 안 된다.

### §4.3 Multiple reward와 multiple domain

`GRPOTrainer`는 `reward_funcs` list, `reward_weights` list(기본값: 모두 1.0, `grpo_config.py` L213-L215), 그리고 두 order를 가진 `multi_objective_aggregation`(L216-L224)을 받는다. 코드(`grpo_trainer.py` L2787-L2826)는 다음과 같다.

```python
if self.multi_objective_aggregation == "sum_then_normalize":
    rewards = (rewards_per_func * self.reward_weights...).nansum(dim=1)
    mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)...
    advantages = rewards - mean_grouped_rewards
    if self.scale_rewards != "none":
        advantages = advantages / (std_rewards + 1e-4)
elif self.multi_objective_aggregation == "normalize_then_sum":
    grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
    mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
    std_k = nanstd(grouped, dim=1, keepdim=True) ...
    reward_k = (grouped - mean_k) / (std_k + 1e-4)
    rewards = (reward_k * self.reward_weights...).nansum(dim=1)
    advantages = (rewards - torch.nanmean(rewards)) / (std_rewards + 1e-4)
```

`nanstd`는 unbiased correction `count / (count − 1)`을 사용한다(`trl/trainer/utils.py` L881-L885). 그러므로 아래 divisor는 sample standard deviation이다.

**계산 예제.** prompt 하나, G = 4 completion이다. Reward 1은 binary correctness `[1, 0, 1, 0]`이다. Reward 2는 raw token unit의 length penalty `[−120, −100, −60, −80]`다. `reward_weights = [1.0, 1.0]`, `scale_rewards="group"`이다.

*`sum_then_normalize`*: 합친 reward는 `[−119, −100, −59, −80]`, mean은 −89.5, sample std는 25.826이다. Advantage는 **[−1.14, −0.41, +1.18, +0.37]**이다.

*`normalize_then_sum`*: reward 1 normalized → `[0.866, −0.866, 0.866, −0.866]`; reward 2 normalized → `[−1.162, −0.387, +1.162, +0.387]`; weighted sum `[−0.296, −1.253, +2.028, −0.479]`; 다시 normalized → **[−0.21, −0.89, +1.43, −0.34]**.

completion 1과 completion 4를 읽어 보자. completion 1은 **정답**이면서 가장 길다. `sum_then_normalize`에서는 group에서 가장 negative advantage(−1.14)를 받으므로, 길다는 이유로 correct answer에서 policy가 멀어지도록 밀린다. completion 4는 **오답**이고 중간 길이다. `sum_then_normalize`에서는 positive advantage(+0.37)를 받는다. `normalize_then_sum`에서는 completion 4의 advantage 부호가 −0.34로 바뀌고, completion 1은 negative(−0.21)를 유지하지만 그것보다 높아진다. 따라서 네 advantage의 순서는 두 correct completion을 모두 두 wrong completion보다 위에 둔다(+1.43 및 −0.21 대 −0.34 및 −0.89). `sum_then_normalize`의 순서는 +1.18, +0.37, −0.41, −1.14이며 wrong completion을 correct completion보다 위에 둔다. 원인은 scale이다. reward 2의 raw spread는 60 unit이고 reward 1의 것은 1 unit이므로 `reward_weights=[1.0, 1.0]`은 기본 aggregation에서 같은 influence를 뜻하지 않는다. **[figures/multi-reward-aggregation.html](figures/multi-reward-aggregation.html)**로 두 reward vector와 weight를 바꾸고 두 advantage vector가 함께 갱신되는 것을 보라. 두 order가 sign에서 언제 불일치하는지를 가장 빨리 볼 수 있다.

**범용 모델에 대한 함의.** multi-domain RL은 검증 가능한 task 하나로 좁아지는 것을 피하는 표준 방법이다. math, code, instruction-following prompt를 하나의 `train_dataset`에서 섞고 domain별 reward function을 각각 준다. 이는 각 domain의 reward가 비교 가능한 scale로 기여할 때에만 동작한다. `sum_then_normalize`와 heterogeneous scale에서는 한 domain의 reward가 advantage를 지배하고, mixture가 조용히 single-domain RL로 퇴화한다.

### §4.4 내장 reward function

`trl/rewards/`는 `accuracy_reward`, `reasoning_accuracy_reward`, `get_cosine_scaled_reward`(`accuracy_rewards.py`), `think_format_reward`(`format_rewards.py`), `get_repetition_penalty_reward`, `get_soft_overlong_punishment`(`other_rewards.py`)를 제공한다([[trl-repo-a04ffd3]]). 이것들은 verifier를 직접 작성하지 않고 쓸 수 있는, recipe literature에 나타나는 format 및 length control이다.

---

## §5 Online 및 asynchronous trainer

### §5.1 Online DPO

`trl.experimental.online_dpo.OnlineDPOTrainer`는 prompt마다 정확히 completion 2개를 sample한다(`self.num_generations = 2`, `a08e713`의 L339). 각 completion을 `reward_funcs`로 score하고, 더 높은 score의 것을 chosen으로 label한 뒤, 동일한 `training_step` 안에서 그 pair에 DPO 또는 IPO step을 적용한다([[trl-online-dpo]]). pair는 저장하지 않는다. 오래된 설명이 잘못 말하는 사실 두 가지는 다음과 같다.

- **이 commit에는 judge interface가 없다.** TRL v1.1.0(PR #5485)이 judge support를 제거했으며, scoring은 reward-function만으로 한다.
- **reference model은 frozen으로 남는다.** on-policy인 것은 pair이지 π_ref가 아니다.

tie 동작은 이 trainer의 결함이다. `mask = first_half >= second_half`(`a08e713`의 L1197, [[trl-online-dpo]])이므로, 두 completion이 같은 score를 받을 때(binary correctness reward에서는 흔하다) 첫 sample이 chosen으로 label되고 sigmoid DPO loss에는 여전히 nonzero gradient가 있다. binary reward에 이 trainer를 쓰기 전에 tie를 filter하라.

### §5.2 Asynchronous GRPO

`trl/experimental/async_grpo/`는 rollout generation을 optimizer step과 분리한다. staleness bound는 명시적이다(`async_grpo_config.py` L120-L133).

```
max_inflight_tasks (defaults to -1): ... auto, which sets it to
    max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes
max_staleness (defaults to 4): Maximum number of weight update steps a rollout sample can lag
    behind the current model version before being discarded.
weight_sync_steps (defaults to 1): Number of training steps between weight synchronizations.
```

따라서 policy version *v*에서 생성한 rollout은 policy가 *v* + 4에 도달할 때까지 사용할 수 있다. 그 뒤에는 importance correction을 하는 대신 discard한다. 이 commit에서 “TRL은 synchronous하다”는 주장은 성립하지 않으므로, TRL concurrency model에 대한 진술에는 commit을 붙여야 한다.

config는 `dtype="float32"`도 기본값으로 두고, docstring(L36-L46)에서 그 이유를 설명한다. trainer는 “Defeating the Training-Inference Mismatch via FP16”(arXiv:2510.26788)에 서술된 training-inference numerical mismatch를 기준으로 측정되며, gap을 닫으려면 vLLM 쪽 serving을 같은 dtype으로 해야 한다. mismatch는 train 시작 시 warning으로 log된다. `vllm_mode`는 `GRPOConfig`에서 여전히 `"server"` 또는 `"colocate"`다(L589, 기본값 `"colocate"`). 이 commit에서 어느 쪽도 deprecated가 아니다.

---

## §6 Distillation trainer

### §6.1 stable on-policy trainer

`trl/trainer/distillation_trainer.py`는 supported trainer이며, docstring은 방법을 다음과 같이 진술한다.

```
Trainer for knowledge distillation. The student is trained on-policy — it generates the completions itself — to
match the teacher's next-token distribution under a generalized Jensen-Shannon divergence (interpolating forward
KL, reverse KL, and JSD via `beta`), as introduced in On-Policy Distillation of Language Models
(https://huggingface.co/papers/2306.13649).
```

이 논문은 [[gkd-on-policy-distillation]](Agarwal et al., ICLR 2024)이다. Eq. 1은 mixture M = βP + (1−β)Q를 이용한 generalized JSD를 정의한다. P는 teacher, Q는 student다.

  D_JSD(β)(P ‖ Q) = β · D_KL(P ‖ M) + (1 − β) · D_KL(Q ‖ M)

- β — (0, 1)의 interpolation coefficient. P — 한 position에서 teacher의 next-token distribution; Q — 같은 position에서 student distribution. M — 두 distribution의 β-mixture.
- 양 끝점은 식의 값이 아니라 limit이다. 논문은 Huszár (2015)를 인용하여 lim_{β→0} D_JSD(β)(P ‖ Q)/β = D_KL(P ‖ Q)라고 하고, “β가 각각 0과 1에 가까울 때 JSD(β)의 gradient는 forward KL과 reverse KL처럼 행동한다”고 말한다(§2). 식 자체는 양 끝에서 0으로 간다.
- TRL은 끝점에서 mixture를 평가하지 않는다. `beta=0.0`은 정확한 forward KL D_KL(P ‖ Q)로, `beta=1.0`은 정확한 reverse KL D_KL(Q ‖ P)로 dispatch된다. mixture form은 0 < β < 1일 때만 쓴다. 따라서 아래 표는 각 설정에서 TRL이 계산하는 값을 보고한다.

endpoint와 mixture branch는 `distillation_trainer.py` L142-L153에 있고, 기본값은 `beta=1.0`이다(`distillation_config.py` L354). 즉 mode-seeking 쪽 끝인 reverse KL이다.

**계산 예제.** vocabulary 크기 3, position 하나다. teacher P = (0.7, 0.2, 0.1), student Q = (0.4, 0.4, 0.2)이다.

| β | Divergence | 값(nats) |
|---|---|---|
| 0.0 | forward KL, D_KL(P ‖ Q) | 0.7·ln1.75 + 0.2·ln0.5 + 0.1·ln0.5 = **0.184** |
| 0.5 | M = (0.55, 0.30, 0.15)인 JSD | 0.5·0.047 + 0.5·0.045 = **0.046** |
| 1.0 | reverse KL, D_KL(Q ‖ P) | 0.4·ln(4/7) + 0.4·ln2 + 0.2·ln2 = **0.192** |

이 산술에서 두 가지를 가져가야 한다. 첫째, β range의 내부는 방향 면에서만 endpoint의 “중간”이 아니다. 여기서는 magnitude가 약 4배 작으므로 β = 1.0에서 tuning한 learning rate는 β = 0.5로 transfer되지 않는다. 둘째, breadth에는 방향이 중요하다. forward KL은 mass-covering이다(teacher가 mass를 둔 곳에 student가 near-zero probability를 두면 penalty를 준다). reverse KL은 mode-seeking이다.

### §6.2 divergence 선택에 관한 근거

[[gkd-on-policy-distillation]]은 둘 다 측정한다.

- **Task-agnostic distillation.** FLAN T5-XL teacher, FLAN T5-Base student, FLAN2021(62개 task에 걸쳐 5.36M example), training 50K step을 사용했다. held-out MMLU(57 task)와 BBH(23 task)를 few-shot prompting으로 평가하고 task들에 대해 평균을 냈다. §4.4는 reverse KL을 쓴 on-policy GKD가 “supervised KD와 ImitKD를 상당히 능가한다”고, 그리고 이 setting에서 reverse KL이 forward KL보다 더 잘 수행한다고 보고한다. gain 크기는 §1에서만 “held-out BBH 및 MMLU benchmark suite에서 2%와 1% absolute accuracy improvement(Figure 10)”라고 주어지며, 어느 baseline 대비인지는 명명하지 않는다. Fig. 10 caption은 시작점을 준다. teacher FLAN T5-XL은 MMLU 52.4%, BBH 41%이고 student는 35.6%, 31.25%다. 저자들의 설명은 논문에서 **Interpretation**으로 label되어 있으며, reverse KL의 mode-seeking behavior가 model이 “instruction이 지정하는 main intent 또는 behavior에 집중하게 한다”고 한다.
- **Diversity cost.** forward KL에서 generalized JSD를 거쳐 reverse KL로 갈수록 “divergence의 강화된 mode-seeking characteristic에 기인한 diversity 감소”가 생긴다. sampling temperature 전반에서 Self-BLEU로 측정했다(Fig. 4 caption). mode-seeking divergence는 high temperature에서 더 좋은 quality를 보였다.
- **Aggregate task-specific gain.** T5 student size(77M, 250M, 800M)에 대해 평균을 내면, on-policy GKD는 initial student 대비 summarization(XSum)에서 2.1×, translation(WMT)에서 1.7×, arithmetic reasoning(GSM8K)에서 1.9× 개선되었다. 이는 baseline KD method가 같은 student에서 달성한 개선에 상대적인 수치다(§1, Fig. 1).

세 경우 모두 **결과(단일 연구)**다. 800M까지 T5-family student, 네 task family에서 얻은 한 논문 결과이며, 현대 decoder-only scale에서는 시험하지 않았다.

### §6.3 GKD와 다른 distillation trainer

`trl/experimental/gkd/`는 student-data fraction을 `lmbda`로 노출한 논문의 Algorithm 1을 구현한다(기본값 0.5, `gkd_config.py` L59). λ = 1은 fully on-policy이고 λ = 0은 fixed dataset에서의 supervised KD다. `beta` 기본값은 0.5, `temperature`는 0.9, `max_new_tokens`는 128, `seq_kd`는 `False`다(sequence-level KD, “teacher-generated output에 대한 supervised FT”). `GKDConfig`는 `SFTConfig`를 subclass하므로 §2의 모든 masking 및 packing field가 적용된다.

GKD와 함께 experimental tree에는 `async_distillation/`, `server_distillation/`, `minillm/`, `gold/`, `sdft/`, `sdpo/`, `ssd/`, 그리고 importance-weighted off-policy distillation인 `iw_opd/`가 있다. on-policy-distillation course는 자체 ch-06(`wiki/courses/on-policy-distillation/ch-06/`)에서 algorithm family를 다룬다. 이 장은 이 commit에서 어느 trainer가 어느 variant를 구현하는지만 다룬다.

**Distillation이 generality chapter에 속하는 이유.** Distillation은 target이 single token 하나가 아니라 full distribution인 유일한 post-training stage다. 따라서 training label에 나타나지 않는 input에서도 teacher behavior를 transfer한다. 위 task-agnostic 결과가 그 근거다. gain은 distillation data가 아니라 held-out benchmark **suite**에서 측정된다.

---

## §7 측정한 LoRA 대 full fine-tuning

### §7.1 측정 가능한 trade-off

[[lora-learns-less-forgets-less]]는 두 regime, 즉 instruction fine-tuning(약 100K prompt-response pair)과 continued pretraining(약 20B token)에서 Llama-2-7B를 대상으로 LoRA와 full fine-tuning을 비교한다. target domain은 code와 math 두 가지다. metric도 두 가지다.

- **Learning**: HumanEval pass@1(code)과 GSM8K strict match(math).
- **Forgetting**: HellaSwag, ARC-Challenge, WinoGrande score의 평균(§3.2, Fig. 2 caption).

보고한 수치와 각 setting은 다음과 같다.

| Regime | Target metric | Forgetting metric |
|---|---|---|
| Code IFT | LoRA r=16: 0.358; r=64: 0.417; r=256: 0.498(모두 epoch 4); full FT: 0.497(epoch 8) | epoch 16에서 full FT 0.414 대 LoRA r=64 **0.509**(§4.2 prose; Table S6). epoch 4에서는 full FT 0.512, LoRA r=64 0.632, r=256 0.631 |
| Code CPT, 20B token | 모든 rank에서 LoRA가 full FT보다 낮음(§4.1) | full FT 0.545 대 LoRA r=256 **0.617**(Table S2) |
| Math CPT | LoRA r=256은 16B에서 0.203 peak; full FT는 20B에서 0.293 | LoRA 0.616(20B) 대 full FT 0.613(16B) — gap 없음 |
| Math IFT | LoRA r=256: 8 epoch에서 0.634; full FT: 4 epoch에서 0.642 | epoch 16에서 LoRA 0.567 대 full FT 0.559 — gap 없음 |

**조건과 한계.** base model 하나(Llama-2-7B), domain 둘, multiple-choice benchmark 세 개로 된 forgetting suite 하나다. forgetting advantage는 **code**에서 크고 **math**에서는 없다. 논문은 OpenWebMath가 English sentence가 지배적이기 때문에 domain shift가 원인이라고 본다(§4.2). code 수치를 math나 dialogue run으로 가져가면 안 된다.

### §7.2 Diversity

HumanEval에서 generation 50개 중 unique output string 수를 세면, full fine-tuning은 pass solution과 fail solution 모두에서 base model보다 적은 unique generation을 만들었고, LoRA는 둘 사이에 있었다(§4.5, Fig. 5). 저자들은 metric의 한계를 직접 말한다. “generation 사이 exact string matching은 predictive diversity의 민감한 metric이 아니다.” **결과(단일 연구)**이며, benchmark 하나와 string-level metric 하나다.

### §7.3 논문의 configuration 권고

§4.7에서 인용하면 다음과 같다. “(a) continued pretraining이 아니라 instruction finetuning에 LoRA를 사용한다. (b) GPU memory가 허용하면 rank 256으로 ‘All’ transformer module을 target으로 삼는다. rank 16 − 64는 code task에 충분하지 않은 경향이 있기 때문이다. (c) α = 2r를 쓴다. (d) [1e−5, 5e−4] 사이 learning rate를 sweep하고 stable training을 가능하게 하는 가장 높은 값을 고른다.” α = 2r는 중요하다. PEFT가 adapter를 α/r로 scale하므로 fixed α는 high rank를 scale down하기 때문이다. “Attention”만 target으로 삼는 것은 “MLP” 및 “All”보다 낮았다(Fig. 7).

### §7.4 TRL run에서 forgetting을 측정하는 법

논문의 metric은 추가 infrastructure 없이 TRL run 안에서 재현할 수 있다. fine-tuning data가 target으로 하지 않는 multiple-choice benchmark 세 개를 hold out하고, base checkpoint에서 score를 낸 후 저장한 checkpoint마다 다시 score를 내고, 평균을 보고하라. metric은 log-likelihood-scored multiple choice의 평균이므로 generation이나 sampling parameter가 필요 없다(§3.2). 수치가 공개 결과와 비교 가능하도록 `compute_metrics` 내부가 아니라, 저장한 checkpoint에서 external harness(`lm-eval-harness`, `lighteval`)를 통해 실행하라.

---

## §8 loop 내부의 평가와 merging

`a04ffd3`의 `trl/trainer/callbacks.py`는 `SyncRefModelCallback`, `RichProgressCallback`, `LogCompletionsCallback`, `WeaveCallback`, `BEMACallback`을 정의한다. breadth 측정에 관련 있는 것은 다음과 같다.

- **`LogCompletionsCallback`** (`callbacks.py` L263; `freq` docstring L283-L284, L314에서 resolve) — trainer의 `eval_dataset`에 있는 prompt의 completion을 매 `freq` step마다 생성한다. `freq`의 기본값은 trainer의 `eval_steps`이며, step / prompt / completion table을 Weights & Biases 또는 Comet에 쓴다. `prompt` column이 있는 eval dataset이 필요하다. qualitative하지만, template breakage 및 stop하지 않는 model을 일찍 잡아내는 수단이다.
- **`eval_dataset`와 `num_generations_eval`** (`grpo_config.py` L67-L69 docstring, L486 field, 기본값 `None` → `num_generations`) — GRPO는 training 때보다 prompt당 sample 수를 적게 하여 평가할 수 있다. 그러므로 out-of-domain eval prompt set을 매 eval step마다 감당할 수 있다.
- **`MergeModelCallback`** (`trl/experimental/merge_model_callback.py` L294, constructor L322-L331) — `mergekit`로 training checkpoint와 target model을 merge한다. `merge_at_every_checkpoint=True`(기본값 `False`)이면 모든 checkpoint에서, 그렇지 않으면 training 끝에 수행하며 optional `push_to_hub`가 있다. `MergeConfig`(L48-L112)는 `{"linear", "ties", "dare_ties", "slerp"}` 중 `method`를 받는다. 기본값은 weight 0.5 / 0.5의 `linear`, policy density `[1.0, 0.7, 0.1]`의 `ties`와 `dare_ties`, `t_values` 0.5의 `slerp`이고, 모두 float16이다. 이는 fine-tuning 중 잃은 capability를 복구하는 표준 post-hoc 조치인, fine-tuned checkpoint weight를 시작 model 쪽으로 average하는 in-framework hook이다. TRL은 mechanism을 제공하지만 이에 대한 evaluation은 보고하지 않는다.

---

## Negative sample과 negative feedback

“negative”의 어느 의미가 적용되는지는 trainer에 따라 다르다. 네 가지 구분을 사용하면 다음과 같다.

| Trainer | negative의 의미 | 메커니즘 |
|---|---|---|
| `SFTTrainer` | (1) negative marginal value | failure는 upstream에서 dataset에서 제외된다. trainer에는 이를 위한 메커니즘이 없다. `loss_type="dft"`는 token을 reweight하지만 어떤 것도 아래로 밀지 않는다. |
| `DPOTrainer` | (4) gradient로서의 negative | `delta_score = chosen_scores − rejected_scores`(L1468)의 rejected term이 log π_θ(y_r)를 낮춘다. |
| `OnlineDPOTrainer` | (4) | on-policy sample 두 개 중 reward가 낮은 것이 y_r가 된다. |
| `GRPOTrainer` | (4) | A < 0인 completion은 token의 log-probability가 아래로 밀린다. |
| `GRPOTrainer` tool loop | (2) content로서의 negative | tool exception은 model이 다음 turn에 읽는 `{"error": ...}` message가 된다. 이 token들은 `tool_mask`로 loss에서 mask out된다([[trl-grpo]]). |

**메커니즘.** softmax에 대해 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. token y에 대한 negative-advantage step은 이 gradient를 빼므로 probability mass가 y에서 빠져나오고 현재 p_j에 비례해 재분배된다. 즉 이미 가장 likely한 token에 집중된다. y가 이미 unlikely하면 이동하는 mass는 작지만 destination은 통제되지 않는다. [[ch-43a]]에 derivation과 displacement evidence가 있다.

**TRL이 노출하는 control**과 위치는 다음과 같다.

1. **negative에서 ratio를 bound한다.** `delta`는 importance ratio에 upper clamp를 적용한다. clip한 두 term의 `min`과 결합하면 A < 0일 때만 bind한다. [[trl-grpo]]의 계산 예제: ε = 0.2, δ = 3, A = −1, r = 5이면 `min(−3, −1.2) = −3`이다. 즉 loss 3이며 r에 대한 gradient는 0이다. δ가 없으면 loss는 5다. A = +1, r = 5일 때 objective는 δ 유무와 관계없이 1.2다.
2. **stale negative를 버린다.** off-policy sequence mask는 A < 0 **이고** sampling policy와 current policy 사이 mean KL이 `off_policy_mask_threshold`를 넘을 때만 sequence를 discard한다(`grpo_trainer.py` L3038-L3062, L3161에서 적용).
3. **truncation을 벌하지 않는다.** `mask_truncated_completions`(기본값 `False`)는 truncated completion을 loss에서 제거한다. 따라서 length budget에 닿았다는 이유로 completion을 아래로 밀지 않는다(`grpo_config.py` L267-L270 docstring, L831 field; docstring은 DAPO를 인용).
4. **부호 비대칭 처리.** `sapo`는 positive advantage의 `sapo_temperature_pos` = 1.0과 달리 non-positive advantage에 `sapo_temperature_neg` = 1.05를 쓴다(`grpo_config.py` L188-L193 docstring, L709 및 L717 field).
5. **positive NLL term으로 anchor를 둔다.** `DPOConfig(loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2])`는 chosen completion에 cross-entropy를 추가한다(§3.1).

**TRL이 log하는 diagnostic.** `DPOTrainer`는 `rewards/chosen`, `rewards/rejected`를 **따로** log하며, `rewards/margins`, `rewards/accuracies`, `logps/chosen`, `logps/rejected`도 log한다(`dpo_trainer.py` L1346-L1363). margin만 보면 chosen/rejected log-probability가 모두 내려가지만 margin은 계속 오르는 failure mode를 숨긴다. `GRPOTrainer`는 sign-split clipping metric을 log한다. low-clip metric은 A < 0인 곳에서만 r < 1 − ε_low를 세고, high-clip은 A > 0인 곳에서만 r > 1 + ε_high를 센다(`a04ffd3`의 `grpo_trainer.py` L3367-L3368). 또 모든 completion이 같은 reward를 받아 A = 0이 된 group의 비율 `frac_reward_zero_std`도 log한다([[trl-grpo]]). entropy는 매 step log된다. cross-framework convention과 각 framework의 “entropy”가 실제로 무엇을 합산하는지는 [[entropy-logging-patterns]]를 보라.

**False negative.** 어떤 TRL source도 reward function의 false-negative rate를 보고하지 않는다. 그 수치는 framework가 아니라 verifier에 속한다.

**generality에 미치는 효과.** size-of-effect 질문은 어떤 TRL source도 답하지 않는다. 해당 setting에서 그 split을 측정한 연구가 없다면 negative term이 DPO 또는 GRPO gain의 주된 동인이라고 주장하지 말라.

---

## 레시피

모든 framework row는 **commit 하나에서의 framework default**다. authoring standard의 §5.3은 이를 paper value나 released run config와 구별되는 사실 class로 다룬다. 별도 언급이 없으면 위치는 `a04ffd337108285002df29d9ae73ef7e855e03f1`의 file과 line이다.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| TRL `a04ffd3` (v1.13.0.dev) | — | SFT | max_length; packing; packing_strategy; completion_only_loss; assistant_only_loss; loss_type; learning_rate | 1024; False; "bfd"; None; False; "chunked_nll"; 2e-5 | `trl/trainer/sft_config.py` L203, L223, L230, L259, L271, L281+L332, L141 [[trl-repo-a04ffd3]] | verified 2026-09-17 | ablation 보고 없음 |
| TRL `a04ffd3` | — | preference | loss_type; loss_weights; beta; f_divergence_type; learning_rate | ["sigmoid"]; None (→ 모두 1.0); 0.1; "reverse_kl"; 1e-6 | `trl/trainer/dpo_config.py` L241, L250, L293, L267, L155 | verified 2026-09-17 | ablation 보고 없음 |
| TRL `a04ffd3` | — | RL | loss_type; beta; scale_rewards; num_generations; max_completion_length; epsilon; top_entropy_quantile; multi_objective_aggregation; use_vllm; vllm_mode | "dapo"; 0.0; "group"; 8; 512; 0.2; 1.0; "sum_then_normalize"; False; "colocate" | `trl/trainer/grpo_config.py` L796, L676, L784, L479, L493, L688, L861, L772, L582, L589 [[trl-repo-a04ffd3]]; loss code [[trl-grpo]] | verified 2026-09-17 | ablation 보고 없음; docstring은 DAPO ε_high 0.28 및 “Beyond the 80/20 Rule” ρ 0.2를 **paper** 값으로 인용 |
| TRL `a04ffd3` | — | distill-SFT | beta; temperature; max_completion_length; learning_rate; gradient_checkpointing | 1.0 (reverse KL); 1.0; 512; 1e-6; True | `trl/trainer/distillation_config.py` L354, L245, L222, L171, L155-L160 | verified 2026-09-17 | ablation 보고 없음 |
| TRL `a04ffd3` | — | distill-SFT | GKD: temperature; lmbda; beta; max_new_tokens; seq_kd | 0.9; 0.5; 0.5; 128; False | `trl/experimental/gkd/gkd_config.py` L55, L59, L66, L74, L96 | verified 2026-09-17 | ablation 보고 없음 |
| TRL `a04ffd3` | — | RL | async GRPO: dtype; num_generations; max_completion_length; max_staleness; weight_sync_steps; queue_maxsize; lr_scheduler_type | "float32"; 8; 2048; 4; 1; 1024; constant | `trl/experimental/async_grpo/async_grpo_config.py` L177, L232, L236, L358, L369, L365, L143-L149 | verified 2026-09-17 | docstring이 float32 default에 arXiv:2510.26788을 인용 |
| TRL `a08e713` (v1.13.0에서 삭제) | — | RL | PPO: kl_coef; vf_coef; cliprange; num_ppo_epochs; gamma; lam; response_length; temperature; kl_estimator | 0.05; 0.1; 0.2; 4; 1.0; 0.95; 53; 0.7; "k1" | `trl/experimental/ppo/ppo_config.py` L232-L264 [[trl-ppo]], [[trl-ppo-recipe]] | verified 2026-09-14 | 문서화된 benchmark run 하나: Pythia-1B, TL;DR (docs L173-L202) |
| Zephyr-7B SFT (Alignment Handbook) | 7B | SFT | base; max_seq_length; LR; schedule; warmup; epochs; per-device batch; grad accum; precision | Mistral-7B-v0.1; 2048; 2.0e-05; cosine; 0.1; 1; 16; 1; bf16 | `recipes/zephyr-7b-beta/sft/config_full.yaml` @`1de1fc9` [[hf-alignment-handbook]] | verified 2026-09-17 | ablation 보고 없음; config는 packing도 response-only masking도 설정하지 않음 |
| GKD (on-policy distillation) | T5-Base student | distill-SFT | teacher; data; steps; divergence; λ | FLAN T5-XL; FLAN2021 5.36M examples / 62 tasks; 50K steps; reverse KL; 1 (on-policy) | arXiv:2306.13649v3 §4.4, Fig. 10 [[gkd-on-policy-distillation]] | verified 2026-09-17 | Fig. 10: initial student 대비 BBH +2%, MMLU +1% absolute |
| LoRA IFT (paper recommendation) | Llama-2-7B | SFT | target modules; rank; α; LR sweep | "All"; 256; 2r = 512; [1e−5, 5e−4], highest stable | arXiv:2405.09673v2 §4.7 [[lora-learns-less-forgets-less]] | verified 2026-09-17 | Fig. 7(module sweep); Fig. S3(r = 256에서 α × LR sweep) |

**작은 general-purpose run의 출발점.** node 하나의 8B model에서는 위 verified row가 다음을 뒷받침한다. `SFTConfig(max_length=2048, packing=True, packing_strategy="bfd", assistant_only_loss=True, learning_rate=2e-5, lr_scheduler_type="cosine", warmup_ratio=0.1, num_train_epochs=1)`를 쓴다. length, learning rate, schedule, warmup, epoch count는 약 200K UltraChat dialogue의 7B Mistral base에 대한 Zephyr-7B 값이다. `packing`과 `assistant_only_loss`는 Zephyr config가 설정하지 않은 TRL field이므로, 재현한 값이 아니라 이 course의 선택으로 취급한다. 다음으로 TRL 자체 default인 `DPOConfig(loss_type=["sigmoid"], beta=0.1, learning_rate=1e-6)`를 쓴다. 그 뒤 RL로 가면 `"dapo"`/`0.0`/`"group"`을 받아들이지 말고 `GRPOConfig(loss_type=..., beta=..., scale_rewards=...)`를 명시한다. reward function이 둘 이상이면 언제든 `multi_objective_aggregation="normalize_then_sum"`으로 설정한다. 이 TRL default 중 published ablation이 있는 것은 없으며, supporting evidence가 있는 위 숫자는 모두 마지막 column에 표시했다.

---

## 일반화 관점

**(a) breadth를 높이는 것.**

- **더 강한 teacher에 대한 on-policy distillation.** held-out-suite evidence: reverse KL on-policy GKD는 supervised KD 및 ImitKD보다 BBH와 MMLU에서 높았고, gain은 2%와 1% absolute로 보고되었다. training data는 FLAN2021이며 어느 suite도 그 안에 없다([[gkd-on-policy-distillation]] §1 및 §4.4, Fig. 10; 논문은 2% / 1% 수치의 baseline을 명시하지 않는다).
- **mixed-domain prompt에 대한 multiple reward function**을 common scale(`normalize_then_sum`)로 aggregate하면, 둘 이상의 domain gradient가 advantage에서 보이게 유지한다(§4.3 계산 예제; 메커니즘이며 published TRL ablation 없음).
- **α = 2r인 rank 256 LoRA**는 code IFT target accuracy에서 full fine-tuning과 맞먹었다(epoch 4에서 0.498, full FT epoch 8의 0.497). 두 point 모두에서 forgetting metric은 더 높았다(0.631 대 0.446)([[lora-learns-less-forgets-less]] §4.1-§4.2, Tables S5 및 S6). **결과(단일 연구)**, Llama-2-7B다.
- weight가 있는 **`loss_type=["sigmoid", "sft"]`**는 preference objective에 positive NLL anchor를 유지한다(`dpo_trainer.py` L1581-L1587). 메커니즘이며 TRL은 evaluation을 보고하지 않는다.

**(b) narrowing 또는 forgetting을 일으키는 것.**

- **먼 domain에서 full fine-tuning.** 20B token Code CPT에서 forgetting metric은 full FT 0.545, LoRA r = 256은 0.617이다([[lora-learns-less-forgets-less]] Table S2). output diversity도 감소했다. pass/fail solution 모두에서 50개 중 unique HumanEval generation이 base model보다 적었다(§4.5, Fig. 5).
- **mode-seeking distillation.** forward KL에서 reverse KL 쪽으로 이동하면 Self-BLEU로 측정한 generation diversity가 감소한다([[gkd-on-policy-distillation]] Fig. 4). TRL의 `DistillationConfig` 기본값 `beta=1.0`은 reverse-KL 끝이다.
- **전체 렌더링 대화로 학습.** 일반적인 two-round example에서 gradient의 약 3분의 1이 template와 user token으로 가며(§2.3), 이는 model이 답변이 아니라 turn structure를 생성하도록 학습한다.
- **scale이 지배하는 multi-reward aggregation.** `sum_then_normalize`에서 raw spread가 큰 reward는 correct completion의 advantage sign을 뒤집을 수 있다(§4.3).
- **단일 domain에서 단일 verifiable reward.** 이것이 일반적인 narrowing 메커니즘이다. TRL의 기여는 대안을 싸게 만드는 `reward_funcs` list다.

**(c) 이 stage에서 측정하는 방법.**

- **Forgetting**: [[lora-learns-less-forgets-less]] metric, 즉 HellaSwag, ARC-Challenge, WinoGrande 평균을 base checkpoint와 모든 saved checkpoint에서 score한다. generation이 없으므로 sampling-parameter confound도 없다(§3.2).
- **Coverage**: held-out task에서 pass@1이 아니라 큰 k의 pass@k를 쓴다. 같은 논문의 Appendix F는 full fine-tuning이 k < 64에서는 LoRA r = 256보다 높고 그 이상에서는 같음을 보였다. k = 1에서는 보이지 않는 차이다.
- **Diversity**: fixed sample count에서 unique generation을 세되, 저자들이 말한 caveat(string matching은 coarse함)를 적용한다. 또는 [[gkd-on-policy-distillation]]처럼 Self-BLEU를 쓴다.
- **In-loop breadth**: out-of-domain prompt의 held-out `eval_dataset`에 `num_generations_eval`을 `num_generations`보다 작게 설정하고, `LogCompletionsCallback`으로 실제 output을 읽는다(§8).
- **알려진 measurement error**: TRL 자체 entropy와 KL metric은 trainer 간 mask가 균일하지 않다. `a08e713`에서 PPO trainer의 `objective/entropy`는 `INVALID_LOGPROB = 1.0` sentinel인 padded position을 포함해 `−logprobs`를 sum했으므로 commit `190c39c` 전까지 padded token마다 −1.0을 더했다([[trl-ppo]]). `OnlineDPOTrainer`의 `objective/kl` 및 `objective/entropy`는 padding mask 없이 모든 completion position을 합산하고, `mean_entropy`는 token distribution의 entropy가 아니라 negative sampled log-probability다([[trl-online-dpo]]). 이 수치는 trainer 하나와 commit 하나 안에서만 비교하라.

---

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 방법 |
|---|---|---|
| `SFTConfig`에 `train_on_response_only=True` 또는 `max_seq_length=` 사용 | unexpected keyword의 `TypeError`, 또는 wrapper가 field를 조용히 받아 무시 | pin한 commit에서 `SFTConfig.__dataclass_fields__.keys()`를 print한다. field는 `assistant_only_loss`, `completion_only_loss`, `max_length`다. |
| conversational dataset에서 masking을 기본값으로 둠 | answer 뒤 fabricated user turn 생성; training loss가 예상보다 낮음 | batch 하나의 `labels`에서 `-100` position을 제거해 decode하고 assistant text만 남는지 확인 |
| `{% generation %}` 없는 chat template, 또는 mask 밖 end-of-turn token | model이 stop하지 않고 generation이 `max_new_tokens`까지 감 | training log에서 “does not include the assistant turn's end-of-turn token in the loss mask”를 검색(`sft_trainer.py` L1271-L1277) |
| `DPOTrainer`에서 `loss_type="kto"` / `"simpo"` / `"orpo"`라고 가정 | 15 valid value를 나열하는 `ValueError: Unknown loss type` (`dpo_trainer.py` L1597-L1600) | `dpo_config.py` L81-L88을 읽고, `CPOTrainer(loss_type="simpo")` 및 별도 experimental trainer 사용 |
| default config로 GRPO paper 재현 | paper와 다른 결과; memory에 reference model 없음 | train 시작에 `args.loss_type`, `args.beta`, `args.scale_rewards`를 log하고 paper objective와 비교 |
| `approxkl`을 policy-to-reference KL로 읽음 | policy가 SFT model에서 멀리 drift했는데 KL은 작아 보임 | `approxkl`은 policy-vs-rollout-policy(k2 form)이고 reference KL은 `objective/kl`이다([[trl-ppo]] Corrections) |
| k1을 biased estimator로 취급 | KL penalty가 noisy한 이유를 잘못 추론 | k1은 unbiased/high variance, k2는 biased/low variance, k3은 unbiased/low variance/non-negative([[john-schulman-kl-tricks]]) |
| reward scale이 다른데 `reward_weights`를 같게 둠 | small-scale reward curve는 flat, advantage는 large-scale reward만 추적 | reward function별로 log하고, `multi_objective_aggregation="normalize_then_sum"`으로 바꾼 뒤 advantage vector 비교 |
| binary reward에 `OnlineDPOTrainer` 사용 | `rewards/accuracies`가 0.5 근처, loss가 내려가지 않음 | L1197에서 tie가 chosen-first로 label됨; step 전 zero-margin pair filter |
| pin한 최근 version에서 `PPOTrainer` import | `ImportError` | 2026-09-04 PR #7020에서 삭제되어 v1.13.0에 release됨; 이전 release를 pin하거나 `RLOOTrainer`/`GRPOTrainer` 사용 |
| trainer나 commit 간 entropy 비교 | artifact인 entropy “collapse” 또는 “growth” | 사용하는 commit에서 각 trainer mask 확인([[entropy-logging-patterns]]) |

---

## 이해 점검

1. conversational SFT run이 `assistant_only_loss`, `completion_only_loss`를 기본값으로 둔다. 이 configuration에서 inference 시 fabricated user turn을 내는 model에 이르는 mechanism을 추적하고, training loss curve가 이를 드러내지 않는 이유를 말하라.
2. 같은 rollout batch가 `"grpo"`, `"bnpo"`, `"dr_grpo"`에서 scalar loss 0.500, 0.333, 0.125를 낸다. normalizer expression에서 어느 completion이 각 경우 relative weight를 얻는지 설명하고, positive advantage일 때 short completion 쪽으로 가장 강하게 압력을 주는 normalizer를 예측하라.
3. KL divergence는 그럴 수 없는데 single-sample k1 KL estimate는 왜 negative일 수 있는가? optimizer step당 token이 적은 run에서 k1-based penalty를 적용하면 무엇을 의미하는가?
4. 같은 weight의 reward function 두 개가 두 `multi_objective_aggregation` order에서 같은 completion에 반대 advantage sign을 만든다. 두 order가 sign에서 불일치하는 두 reward vector의 조건을 유도하라.
5. TRL `DistillationConfig`는 `beta=1.0`이 기본이고 GKD 논문은 reverse KL이 held-out instruction-tuning suite에서 이기면서 generation diversity도 줄인다고 보고한다. 한 mechanism이 두 결과를 어떻게 만드는지 설명하고, general-purpose student에 default를 받아들이기 전에 무엇을 측정할지 말하라.
6. Biderman et al. experiment에서 LoRA는 code에서는 full fine-tuning보다 forgetting이 적지만 math에서는 그렇지 않다. 그 차이에 관한 causal account를 주고 dialogue-domain fine-tune에 무엇을 예측하는지 말하라.
7. Online DPO는 매 step fresh pair를 sample하면서 frozen reference model을 유지한다. DPO loss의 어느 quantity가 정확히 on-policy가 되고 어느 것은 그렇지 않은지, 그리고 loss가 policy drift를 constrain하지 못하기 전 policy가 얼마나 멀어질 수 있는지에 주는 함의를 설명하라.
8. commit pin 없는 TRL training script를 받고 그것이 2025 GRPO paper를 재현하는지 묻는다. 순서대로 먼저 확인할 configuration fact 네 가지와, 각각이 조용히 바꿀 수 있는 것을 나열하라.

---

## 연결

- 이전: **ch-56** — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls. §1 및 §5.2의 Ray 기반 비교를 제공한다.
- 다음: **ch-58** — Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability. §7 및 §8을 선택 및 instrumentation 절차로 바꾼다.
- **ch-55** — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation. TRL이 하나의 `loss_type` switch에 합친 advantage와 loss computation의 registry-based split.
- **ch-43a** — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. negative section이 적용한 logit-gradient derivation을 담는다.
- **ch-40** — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO. §4.1 normalizer가 구현하는 algebra.
- **ch-43** — Entropy, Output Diversity, and KL Control in RL. §4.2의 β와 KL-estimator 논의.
- **ch-32c** — Claimed versus Effective Context Length and Long-Context Evaluation. §2.4의 `"bfd"` packing strategy가 long example에서 만드는 truncation failure.
- **ch-58a** — Open General-Model Recipes End to End: Pretraining to Merge. §8 merging callback이 full recipe 안에 나타나는 곳.

---

## 출처

- [[trl-repo-a04ffd3]] — pin한 commit의 repository tree, `SFTConfig`, `DPOConfig`, `DistillationConfig`, `GKDConfig`, `AsyncGRPOConfig` field quote와 line number를 담은 chapter excerpt.
- [[trl-grpo]] — `_compute_loss` loss-type normalizer, advantage 및 multi-reward code, k3 KL term, entropy mask, off-policy sequence mask, tool-token masking, sign-split clip metric, 기본 `loss_type="dapo"` / `beta=0.0`.
- [[trl-grpo-recipe]] — `GRPOConfig` framework-default ledger.
- [[trl-ppo]] — 삭제된 actor-critic trainer: 별도 value model, KL-in-reward, k1/k3 선택, `approxkl` 대 `pg_clipfrac`, entropy-masking defect, PR #7020 삭제.
- [[trl-ppo-recipe]] — PPO framework default와 문서화된 Pythia-1B TL;DR run.
- [[trl-online-dpo]] — prompt당 two-sample pairing, L1197 tie handling, frozen reference, judge interface 부재, unmasked KL/entropy metric.
- [[trl-online-dpo-recipe]] — online-DPO framework default와 문서화된 run.
- [[gkd-on-policy-distillation]] — generalized JSD 정의, Algorithm 1과 λ, held-out MMLU/BBH에서의 task-agnostic FLAN 결과, divergence-diversity trade-off를 담은 chapter excerpt.
- [[lora-learns-less-forgets-less]] — learning/forgetting metric, code/math IFT/CPT 수치, diversity count, §4.7 configuration 권고를 담은 chapter excerpt.
- [[hf-alignment-handbook]] — Recipe table에 쓴 Zephyr-7B SFT config 값. card의 “`packing: true` 및 `train_on_response_only: true`를 유지”라는 guidance와 `train_on_response_only` code snippet은 config 및 current TRL API와 맞지 않아 §2.5에서 바로잡는다.
- [[hf-dpo-zoo]] — DPO-variant survey. 모든 variant가 `DPOTrainer`의 `loss_type`으로 노출된다는 주장을 `dpo_config.py` L81-L88 및 repository tree와 대조하여 §3.1에서 바로잡는다.
- [[grpo]] — `loss_type="grpo"` 및 k3 term이 따르는 DeepSeekMath objective와 KL estimator.
- [[dr-grpo]] — `loss_type="dr_grpo"`로 구현한 length-free normalizer.
- [[dpo]] — `delta_score` 뒤의 closed-form preference loss.
- [[ipo]] — TRL length normalization note를 포함해 `loss_type="ipo"`로 구현한 squared loss.
- [[ppo]] — 삭제된 PPO trainer가 구현했던 clipped surrogate.
- [[john-schulman-kl-tricks]] — k1/k2/k3 정의와 bias/variance property.
- [[rlhf-instructgpt]] — `beta=0.0`이 제거하는 reference-KL-anchored RLHF pipeline.
- [[verl-grpo]], [[openrlhf-ppo]], [[openrlhf-dpo]] — [[ch-55]]와 [[ch-56]]의 두 framework에서 같은 loss.
- [[entropy-logging-patterns]] — TRL 수치를 verl 또는 OpenRLHF와 비교하기 전에 필요한, framework별 entropy metric이 합산하는 대상.
