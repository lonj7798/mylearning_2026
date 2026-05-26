<!-- chapter: ch-57
     track: infra
     kind: content
     title: TRL Internals
     deps: [ch-56]
     sources: [[trl-ppo]], [[trl-grpo]], [[trl-online-dpo]], [[hf-alignment-handbook]], [[hf-dpo-zoo]], [[hf-rlhf-illustrated]], [[grpo]], [[dpo]], [[ppo]]
     figures: figures/trl-stack.html
-->

# 57장 — TRL 내부 구조

> **핵심 통찰.** TRL은 distributed-systems layer를 쓰지 않겠다고 결정했을 때 생기는 결과물이다. 모든 trainer는 `transformers.Trainer`의 subclass이고, 모든 distributed primitive는 `accelerate`에서 오며, 모든 model은 `PreTrainedModel`이고, 모든 dataset은 `datasets.Dataset`이다. 전체 저장소는 RL 모자를 쓴 HF ecosystem이다. 그래서 chat-template 아이디어에서 Zephyr 규모 실행까지 가는 가장 빠른 경로가 되며, 동시에 single node를 넘어서면 절벽에서 떨어진다. scheduler도, weight-transfer protocol도, rollout-worker pool도 없다. verl([[ch-55]])은 Ray + custom controller에 걸었고, OpenRLHF([[ch-56]])는 Ray + colocated vLLM에 걸었다. TRL은 `accelerate launch`에 걸었고, 그 선택이 저장소의 모든 tradeoff를 형성한다.
>
> **가이드라인.** 한 node에 맞는 모든 작업(≤ 8×H100)은 기본적으로 TRL을 선택하라. `packing=True, train_on_response_only=True`로 `SFTTrainer`를 쓰고, offline preference에는 `loss_type="sigmoid"`의 `DPOTrainer`, verifiable reward에는 `loss_type="dr_grpo"`의 `GRPOTrainer`를 사용하라. `experimental/` 폴더는 unstable하지만 canonical한 곳으로 취급하라. active RL algorithm이 거기에 들어온다. (a) multi-node rollout, (b) asynchronous actor/rollout decoupling, (c) `loss_type` switch에 맞지 않는 custom advantage estimator가 필요해지는 순간 **TRL을 졸업하라**. 그 시점에서는 TRL을 patch하는 비용보다 verl로 porting하는 비용이 더 작다.

---

## §1 저장소 둘러보기 — stable vs experimental

2026년 4월 기준 `huggingface/trl`은 `trl/` 아래 몇 개의 top-level package로 구성되어 있다.

```
trl/
├── trainer/                   # stable, supported trainers
│   ├── sft_trainer.py         # SFTTrainer + SFTConfig
│   ├── dpo_trainer.py         # DPOTrainer + DPOConfig (offline, sigmoid/ipo/kto/simpo/orpo/bco/cpo)
│   ├── grpo_trainer.py        # GRPOTrainer + GRPOConfig (monolithic, ~2700 LOC)
│   ├── rloo_trainer.py        # RLOOTrainer (k-sample leave-one-out)
│   ├── kto_trainer.py         # unary-label KTO (thin wrapper)
│   ├── orpo_trainer.py        # ORPO: SFT+preference joint
│   ├── reward_trainer.py      # reward model training (Bradley-Terry)
│   └── utils.py               # selective_log_softmax, masked_mean, etc.
├── experimental/              # new / unstable algorithms
│   ├── ppo/ppo_trainer.py     # classic actor-critic PPO, value head
│   ├── online_dpo/online_dpo_trainer.py
│   ├── nash_md/               # Nash-MD self-play
│   ├── xpo/                   # exploratory preference optimization
│   └── self_distillation/
├── models/                    # thin model wrappers
│   ├── modeling_value_head.py # AutoModelForCausalLMWithValueHead
│   └── utils.py
├── core/                      # PPO utilities (legacy, still used by experimental/)
└── extras/                    # BoN sampler, datasets helpers
```

**stable/experimental split은 중요하다.** `trl/trainer/`의 코드는 semver guarantee와 함께 배포된다. `GRPOTrainer`, `DPOTrainer`, `SFTTrainer`, `RLOOTrainer`, `RewardTrainer`, `KTOTrainer`, `ORPOTrainer`가 여기에 있다. `trl/experimental/`은 active research가 어떤 약속을 받기 전에 도착하는 곳이다. 여기에는 actor-critic PPO trainer(예전에는 mainline이었지만 2024년 critic-free method로 pivot한 뒤 demoted되었다. [[trl-ppo]] 참조), online-DPO trainer, Nash / XPO self-play variant가 포함된다.

`trl/core/` module은 shared PPO utility를 담는다. reward-shaping helper, adaptive-KL controller, advantage whitening 같은 것들이다. 이들은 `accelerate`가 distributed backbone이 되기 전부터 있던 코드다. 상당수는 experimental trainer가 살려두는 dead code다. `ppo_trainer.py`에서 import되는 것을 볼 수 있고, 그 외에는 거의 보이지 않는다.

**`trl.DPOTrainer` trick**([[hf-dpo-zoo]]). 하나의 trainer class가 `loss_type` string으로 전체 DPO family를 처리한다. `"sigmoid"`(vanilla DPO), `"ipo"`(Azar identity), `"kto"`(prospect theory unary), `"simpo"`(reference-free length-normalized), `"orpo"`(joint SFT+odds-ratio), `"bco"`(binary classifier), `"cpo"`(contrastive)가 모두 포함된다. 대수는 다르지만 dataloader와 distributed code path는 동일하다. 이것이 TRL의 전체 design philosophy다. 하나의 trainer skeleton을 고르고, 모든 variant를 string argument로 노출한 뒤, HuggingFace의 `Trainer`가 나머지를 처리하게 둔다.

---

## §2 세 가지 RL trainer — signature와 실제 동작

### SFTTrainer(stable anchor, [[hf-alignment-handbook]])

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    packing=True,                 # concat + reshape to max_seq_length
    max_seq_length=2048,
    train_on_response_only=True,  # masks user tokens to label = -100
    dataset_kwargs={"add_special_tokens": False},
)
trainer = SFTTrainer(
    model=base_model,             # AutoModelForCausalLM or path string
    args=config,
    train_dataset=ds_train,
    processing_class=tokenizer,   # was `tokenizer=` pre-0.14
)
trainer.train()
```

내부적으로 이것은 (a) chat-template-aware preprocessing, (b) `ConstantLengthDataset`을 통한 optional packing, (c) 모든 non-response position의 label tensor를 `-100`으로 설정하는 prompt-token masking을 더한 `transformers.Trainer`일 뿐이다. 실제 forward/backward/optimizer step은 HF `Trainer`에서 통째로 상속되므로, `accelerate`가 DDP / FSDP / DeepSpeed를 투명하게 처리한다.

### DPOTrainer

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(beta=0.1, loss_type="sigmoid", max_length=2048,
                max_prompt_length=1024)
trainer = DPOTrainer(model=policy, ref_model=ref_policy, args=cfg,
                     train_dataset=preference_ds, processing_class=tokenizer)
```

DPO는 memory 안에 *두* model이 필요하다. training 중인 policy `π_θ`와 [[dpo]] Eq. 7의 log-ratio denominator에 쓰이는 frozen reference `π_ref`다. PEFT/LoRA를 사용할 때는 `ref_model`을 `None`으로 설정한다. TRL은 base model에서 adapter를 비활성화해 `π_ref`를 계산하므로 full model 하나만큼의 VRAM을 절약한다. 이 trick은 `dpo_trainer.py`의 `null_ref_context()`에 있다.

### GRPOTrainer(거대한 파일, [[trl-grpo]])

```python
from trl import GRPOTrainer, GRPOConfig
cfg = GRPOConfig(
    per_device_train_batch_size=8,
    num_generations=8,                 # G in the GRPO paper
    max_completion_length=1024,
    beta=0.04,
    loss_type="dr_grpo",               # or "grpo", "dapo", "cispo", ...
    use_vllm=True,
    vllm_mode="colocate",              # or "server"
    top_entropy_quantile=0.2,          # DAPO-style entropy filter
)
trainer = GRPOTrainer(
    model=policy, reward_funcs=[verifier_fn],
    args=cfg, train_dataset=prompt_ds,
    processing_class=tokenizer,
)
```

`GRPOTrainer`는 약 2700 LOC 파일이다. `_compute_loss`(약 L2418–2610)는 TRL에서 가장 큰 단일 method이며 GRPO zoo의 모든 loss variant를 포함한다. 이 class는 하나의 loop에서 세 가지 일을 한다.

1. **Generate.** HF `.generate()`(느리지만 정확함), `server` mode의 vLLM(remote), 또는 `colocate` mode의 vLLM(같은 GPU, shared tensor) 중 하나를 사용한다. Generation은 `completion_ids`와 선택적으로 IS ratio용 `old_per_token_logps`를 만든다.
2. **Score.** `reward_funcs`의 각 reward function은 `(prompts, completions)`로 호출되어 `(B,)` reward tensor를 반환한다. 여러 reward function은 configurable weight로 합산된다.
3. **Update.** `_compute_loss`는 per-token logprob를 다시 계산하고, group-relative advantage `(r_i − mean(r_{1..G})) / std`를 만들며, clipped PPO ratio를 형성하고, 선택적으로 top-entropy quantile로 filter하고, reference에 대한 K3 KL을 더하고, `loss_type`별 denominator로 aggregate한 뒤 scalar를 반환한다.

이것은 advantage computation과 loss computation을 두 registry hook으로 나누는 verl과 다른 design axis다. TRL은 둘을 fuse하고, verl은 split한다([[ch-55]]). `loss_type="grpo"`의 대수적 결과는 동일하다.

### Online DPO(experimental, [[trl-online-dpo]])

```python
from trl.experimental.online_dpo import OnlineDPOTrainer, OnlineDPOConfig
cfg = OnlineDPOConfig(beta=0.1, loss_type="sigmoid",
                      missing_eos_penalty=1.0, use_vllm=True)
trainer = OnlineDPOTrainer(model=policy, ref_model=ref,
                           reward_model=rm, judge=None,
                           args=cfg, train_dataset=prompt_ds,
                           processing_class=tokenizer)
```

매 step마다 prompt당 *두* completion을 sampling하고(`prompts = 2 * prompts`), 둘 다 `reward_funcs` 또는 `Judge`로 score한 뒤, 더 높은 점수를 `chosen`, 낮은 점수를 `rejected`로 선언하고 그 fresh pair에 DPO gradient step을 수행한다. pair는 저장되지 않는다. 이것은 offline DPO의 on-policy gap을 닫는다. reference가 실제로 current policy의 neighborhood가 되기 때문이다. 비용은 step마다 completion마다 RM forward가 하나 필요하다는 것이다. [[trl-online-dpo]]에는 argmax pair selection과 sequence-level log-ratio loss를 보여주는 `training_step` excerpt가 포함되어 있다.

---

## §3 Accelerate 기반 orchestration — scale에서 깨지는 것

모든 TRL trainer는 `transformers.Trainer` subclass다. 따라서 distributed story는 `accelerate`의 story다. 이것은 portability에는 훌륭하고 scale에는 좋지 않다. 한 node를 넘어서 밀어붙일 때의 failure chain은 다음과 같다.

### 3.1 Accelerate가 잘하는 것

`accelerate launch`는 YAML config를 기반으로 `{DDP, FSDP, DeepSpeed ZeRO-1/2/3}` 중 하나를 고르고 model을 감싼다. GPU당 하나의 Python process, 외부 scheduler 없음, Ray 없음, MPI 없음. training loop는 단순한 Python `for batch in dataloader`다. single-node run에서는 이것이 **가장 빠른 setup time**이다. config를 쓰고 launch하면 끝이다. Alignment Handbook([[hf-alignment-handbook]])은 즉시 동작하는 Zephyr-7B SFT와 DPO용 8×A100 FSDP config를 제공한다.

### 3.2 rollout이 들어오는 순간 깨지는 것

trainer가 *generate*해야 하는 순간 single-process model은 깨진다. 1024 token × 8 completion × prompt batch에 대한 policy full forward pass는 step마다 몇 초가 걸린다. 모든 rank가 어떤 rank의 `.generate()`를 synchronously 기다리면 throughput은 무너진다. TRL의 답은 세 escape hatch인데, 어느 것도 균일하게 만족스럽지 않다.

1. **HF `.generate()`** — 단순하고 느리며, training과 같은 sharded model을 사용한다. FSDP에서는 generation마다 모든 parameter를 다시 gather한 뒤 다시 shard한다. correctness path이자 가장 느린 path다.
2. **vLLM server mode**(`vllm_mode="server"`). 별도 vLLM process가 dedicated GPU에서 실행된다. TRL은 optimizer step마다 updated weight를 push하는 weight-sync client(`vllm_client`)를 제공한다. rollout은 빨라지지만 이제 server용 extra GPU를 provision해야 하고, weight transfer가 step마다 O(model size)다.
3. **vLLM colocate mode**(`vllm_mode="colocate"`). vLLM이 training process *안에서* 같은 GPU 위에 실행된다. shared weight memory라 network transfer는 없지만, backward 중에도 vLLM의 paged-attention memory manager 비용을 치른다. 2025년에 적극적으로 유지보수되는 mode가 이것이며, server mode는 phased out 중이다.

이를 [[ch-55]] verl architecture와 비교하라. dedicated `RolloutWorker` pool이 자체 GPU set에서 vLLM/SGLang을 실행하고, `CriticWorker`와 `ActorWorker`는 별도 set에 산다. Ray는 `DistributedRollout`과 `HybridEngine`을 통해 scheduling과 weight sync를 처리한다. 또는 [[ch-56]] OpenRLHF의 colocated-Ray setup with unified actor-rollout worker를 보라. 이 framework들은 *distributed systems primitive*(Ray)를 선택하고 그 위에 구축했다. TRL은 이를 거부했다. 그 거부가 TRL의 scaling limit이 어디서 오는지 설명하는 전부다.

### 3.3 Straggler behavior와 single-controller problem

모든 rank가 같은 Python loop를 실행하므로, 어떤 rank든 더 오래 걸리면 전체 collective가 막힌다. sampled completion이 더 길었든, vLLM batch 운이 나빴든, reward function이 느린 HTTP call을 했든 이유는 중요하지 않다. Accelerate에는 built-in straggler mitigation이 없다. 실제 workaround는 다음과 같다.

- **모든 completion을 `max_completion_length`까지 pad**해서 rank들이 generation을 동시에 끝내게 한다. compute를 낭비한다.
- **hard generation timeout**을 설정하고 timed-out rollout을 버린다. distribution을 bias한다.
- **reward function을 background thread로 이동**한다. I/O-bound인 경우(LLM-as-judge, API call)에 한정된다.

어느 것도 우아하지 않다. verl은 hybrid engine으로 straggler를 처리하고, OpenRLHF는 Ray가 reschedule할 수 있어 버틴다. TRL은 앉아서 기다린다.

### 3.4 `accelerate launch` + SLURM을 넘어서는 native multi-node가 없음

Multi-node TRL은 "각 node에서 올바른 `--machine_rank`로 `accelerate launch`를 호출하는 SLURM script 작성"이다. built-in fault tolerance도, NCCL을 넘어서는 weight-transfer protocol도, run 중 worker를 추가하거나 제거하는 방법도 없다. node 하나가 죽으면 전체 run이 죽는다. < 16 GPU에서는 괜찮다. 64+ GPU에서는 심각한 operational burden이 된다.

---

## §4 HF ecosystem integration — PEFT, datasets, transformers

### 4.1 PEFT-LoRA as the eval-time reference-policy trick

TRL의 DPO/GRPO trainer는 policy가 `PeftModel`인지 감지한다. 그렇다면 별도 `ref_model`을 loading하지 않는다. 대신 base model에서 LoRA adapter를 일시적으로 비활성화(`with adapter_model.disable_adapter()`)하여 reference policy를 계산한다. 전체 model 하나만큼의 VRAM을 절약한다. 야생에서 가장 흔한 TRL trick이다. LoRA + DPO = memory 안의 7B model 하나, 둘이 아니다. [[dpo]] loss는 변하지 않는다.

```python
# simplified pattern used by DPOTrainer / GRPOTrainer when ref_model=None
def null_ref_context(self):
    with self.accelerator.unwrap_model(self.model).disable_adapter():
        yield
# later:
with self.null_ref_context():
    ref_logits = self.model(**batch).logits
```

이 trick은 **PEFT 전용**이다. full fine-tuning은 여전히 두 copy가 필요하다. 70B model에서는 이것이 DPO full-FT가 매우 비싸고 LoRA-DPO가 지배적이 된 이유 중 하나다.

### 4.2 preprocessing용 `datasets`

모든 TRL trainer는 `datasets.Dataset` 또는 `IterableDataset`을 받는다. SFT path는 `datasets.map(tokenize, batched=True, remove_columns=...)`를 사용한 뒤 선택적으로 packing에 `ConstantLengthDataset`을 쓴다. DPO는 `{prompt, chosen, rejected}` column을 기대한다. GRPO는 `{prompt}`와 prompt별 `reward_funcs` call을 기대한다. Online DPO는 `{prompt}`를 기대하고 나머지는 모두 online으로 생성한다.

eval-time implication은 이렇다. 모든 것이 `datasets.Dataset`이므로 RAM보다 큰 corpora에 대한 streaming, built-in Arrow-backed caching, `split_dataset_by_node`를 통한 cross-process sharding을 distributed-IO code 없이 얻는다. 엄청난 productivity win이다. 비용은 non-standard data format을 먼저 `{column: value}` row로 강제로 맞춰야 한다는 것이다.

### 4.3 `transformers` chat template

TRL의 SFT와 DPO trainer는 `tokenizer.apply_chat_template(...)`를 호출해 message를 하나의 string으로 format한다. training과 inference의 chat template이 하나라도 어긋나면 silent bug다. [[hf-alignment-handbook]]의 최상위 lesson은 "packed batch를 항상 decode해 chat template을 검증하라"다. TRL은 이를 강제하지 않고 그대로 통과시킨다.

### 4.4 Eval-time implication

TRL artifact는 `transformers`-native이므로 model이 HF inference code(`pipeline`, `.generate()`, vLLM의 HF-compatible loader)로 conversion step 없이 바로 전달된다. 자체 format으로 checkpoint를 저장하고 vLLM inference에 conversion script가 필요한 verl과 대조된다. HF ecosystem story는 TRL이 prototyping에서 이기는 이유의 나머지 절반이다. 훈련한 것이 곧 serving하는 것이다.

---

## §5 PPO, legacy reference(아직 experimental/에 있음)

classic actor-critic PPO는 `trl/experimental/ppo/ppo_trainer.py`에 있다. inner loop(약 L820–870, [[trl-ppo]])는 open source에서 InstructGPT([[hf-rlhf-illustrated]]) recipe를 가장 깔끔하게 한 파일로 보여준다.

```python
# trl/experimental/ppo/ppo_trainer.py, ≈ L820-870
logprobs_diff = new_logprobs - mb_logprobs
ratio        = torch.exp(logprobs_diff)
pg_losses    = -mb_advantage * ratio
pg_losses2   = -mb_advantage * torch.clamp(ratio, 1 - args.cliprange, 1 + args.cliprange)
pg_loss      = masked_mean(torch.max(pg_losses, pg_losses2), ~padding_mask[mb_inds])

vpredclipped = torch.clamp(vpred, mb_values - args.cliprange_value,
                                  mb_values + args.cliprange_value)
vf_loss      = 0.5 * masked_mean(torch.max((vpred - mb_return)**2,
                                            (vpredclipped - mb_return)**2),
                                  ~padding_mask_p1[mb_inds])
loss = pg_loss + args.vf_coef * vf_loss
```

핵심 architectural fact: policy와 value head는 `AutoModelForCausalLMWithValueHead`를 통해 **base transformer를 공유한다**. 한 번의 forward pass가 logits와 token별 scalar value head output을 모두 반환한다. 이는 별도 model 두 개와 비교해 FLOPs를 절반으로 줄이지만, value estimate가 policy의 gradient flow에 coupling되는 비용을 낸다.

KL은 **loss가 아니라 reward에 적용된다**. `non_score_reward = −β · (logprobs − ref_logprobs)`가 GAE 전에 per-token reward에 더해져 `mb_advantage`의 일부가 된다. 두 KL metric이 logging된다. K1(`logprobs − ref_logprobs`, biased but cheap, reward shaping에 사용)과 K2(`0.5·Δlogp²`, clipfrac diagnostic에 쓰이는 Schulman estimator)다. GRPO는 K3로 전환했다([[trl-grpo]]와 ch-40 §4 참조). 이 PPO trainer가 `experimental/`로 demotion된 것은 field가 critic-based method에서 멀어진 pivot을 반영한다.

---

## §6 TRL이 올바른 선택인 경우 — decision gate

다음이 **모두** 참이면 TRL을 고른다.

- Run이 ≤ 1 node(≤ 8 GPU, 또는 < 70B에서는 2 node)에 맞는다.
- Algorithm이 기존 trainer의 `loss_type`으로 표현 가능하거나 direct subclass다.
- Rollout engine이 `.generate()` 또는 vLLM이다. SGLang이나 custom inference stack이 필요하지 않다.
- Distributed-systems control보다 HF-ecosystem integration(chat template, PEFT, Hub)을 더 중시한다.
- checkpoint를 conversion 없이 `transformers` inference로 내보내고 싶다.

다음 중 **하나라도** 참이면 TRL을 졸업한다.

- **Multi-node rollouts.** TRL의 vLLM colocate/server mode는 소수 rank를 넘어서면 우아하게 scale하지 않는다. verl의 `RolloutWorker` pool이 canonical next step이다.
- **Async actor-rollout decoupling.** TRL은 synchronous다. actor가 batch N으로 train하는 동안 rollout worker가 batch N+1을 만들어야 한다면 Ray(OpenRLHF / verl)가 필요하다.
- **Custom advantage 또는 rollout orchestration.** GRPO의 `_compute_loss`는 `loss_type`으로 switch한다. 확장하려면 200줄 conditional을 patch해야 한다. verl의 registry(`compute_advantage` / `compute_policy_loss` hook)가 더 싸게 확장된다.
- **Straggler-sensitive workloads**(긴 completion, variable reward-model latency). Accelerate는 reschedule할 수 없다.
- **> 70B full-FT RL** — step마다 weight-sync cost가 지배한다. weight transfer를 first-class scheduled operation으로 다루는 framework가 필요하다.

decision gate는 "TRL이 verl보다 나쁘다"가 아니다. "TRL은 productivity-vs-scale Pareto curve의 다른 지점에 최적화되어 있다"다. 한 node 아래에서는 TRL이 중요한 모든 축에서 이긴다. 두 node 위에서는 점점 진다. 어느 쪽 선에 있는지 알아야 한다.

---

## Companion visualization

**[figures/trl-stack.html](figures/trl-stack.html)** — interactive TRL stack diagram. 다섯 clickable layer(`datasets` → `tokenizer` → `SFTTrainer/DPOTrainer/GRPOTrainer` → `Accelerate` → `torch`)가 있다. layer를 클릭하면 (a) 관련 API surface(class / config / method), (b) 그 layer에서 먼저 부딪히는 scaling pain point가 표시된다. top row는 frictionless하다(`datasets` streams, `tokenizer.apply_chat_template` is one call). middle(`Trainer` subclass)은 세 trainer가 모두 공존하는 곳이다. bottom(`Accelerate`)은 scale limit에 부딪히는 곳이다.

---

## Further reading

- [[trl-ppo]] — classic actor-critic PPO, value-head co-location, K1/K2 KL. demoted reference.
- [[trl-grpo]] — `_compute_loss` monolithic body; `loss_type` switch와 K3 KL inline.
- [[trl-online-dpo]] — two-sample-per-prompt training_step; Judge plug-in; sigmoid vs IPO branch.
- [[hf-alignment-handbook]] — Zephyr-7B reference recipe; FSDP + ZeRO configs; eval protocol.
- [[hf-dpo-zoo]] — `loss_type` parameter zoo(sigmoid / ipo / kto / simpo / orpo / bco / cpo).
- [[hf-rlhf-illustrated]] — TRL이 구현하도록 만들어진 three-stage diagram.
- [[grpo]] — `GRPOTrainer`가 그대로 구현하는 Eq. 3.
- [[dpo]] — Eq. 7 closed-form loss; `DPOTrainer` default branch.
- [[ppo]] — experimental trainer가 한 파일에 fuse한 PPO-clip objective.

## Connections

- **ch-55 (verl)** — Ray-based counterfactual. TRL이 거부한 split-worker architecture.
- **ch-56 (OpenRLHF)** — middle-ground Ray setup. TRL이 Ray를 받아들였다면 어떻게 보였을지 보여준다.
- **ch-58 (framework comparison)** — 이 장의 §6이 공급하는 feature matrix + decision tree.
- **ch-37 to ch-41** — TRL이 구현하는 대수의 RL algorithm chapter들.
- **ch-44+ (DeepSeek-R1 recipe)** — frontier scale에서 GRPO production run이 어떤 모습인지. 보통 TRL 위에서는 아니다.
