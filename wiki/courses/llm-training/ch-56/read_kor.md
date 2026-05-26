<!-- chapter: ch-56
     track: infra
     kind: content
     title: OpenRLHF Internals
     deps: [ch-55]
     sources: [[openrlhf-ppo]], [[openrlhf-dpo]], [[entropy-logging-patterns]], [[openrlhf-entropy-debugging]], [[async-rollout]], [[ppo]], [[dpo]], [[rlhf-instructgpt]], [[kl-control-rlhf]]
     figures: figures/openrlhf-ray.html
-->

# 56장 — OpenRLHF 내부 구조

> **핵심 통찰.** OpenRLHF는 `(token-level KL-into-reward) + (clipped policy loss) + (Ray-pool orchestration)` factorization을 70B scale에서 처음 실용적으로 만든 open-source stack이다. 모든 설계 선택, 즉 하나가 아니라 두 KL을 반환하는 `PolicyLoss`, loss *밖에* 존재하는 `AdaptiveKLController`, DPO activation memory를 절반으로 줄이는 `concatenated_forward`, generation과 training을 weight-swap race 없이 overlap하게 해주는 `rollout_queue` / `vllm_lock` pair는 한 가지 태도로 이어진다. *전문화된 Ray actor들을 조합하고, loss는 단순하게 유지하며, reward shaping과 KL controller를 trainer의 first-class citizen으로 만든다.*
>
> **가이드라인.** OpenRLHF는 monolithic trainer가 아니라 queue로 통신하는 Ray actor들의 집합으로 읽어라. `PolicyLoss`는 작다. clip + dual-clip + 선택적 IS correction이 전부다. KL controller는 `ppo_trainer.py`에 있고 loss가 아니라 *reward*를 변경한다. DPO는 frozen reference를 eval mode로 유지하며, memory가 빡빡하면 ZeRO-3-offloaded 상태로 둔다. verl과 비교하면 OpenRLHF는 단순성과 Ray-native orchestration에서 강하고, verl은 dataflow-graph flexibility와 partial-rollout에서 강하다. Ray-managed cluster에서 batteries-included PPO/DPO가 필요하면 OpenRLHF를 고르고, scale에서 continuous-batching RL이 필요하면 verl을 고른다.

---

## 1. 저장소 둘러보기 — 무엇이 어디에 있는가

OpenRLHF([[openrlhf-ppo]])는 `openrlhf/` 아래 세 개의 top-level package를 중심으로 구성되어 있다.

```
openrlhf/
  models/
    loss.py            # PolicyLoss, ValueLoss, DPOLoss, KDLoss
    actor.py, model.py # policy wrapper + RM / critic head wrappers
  trainer/
    ppo_trainer.py, ppo_trainer_async.py, dpo_trainer.py,
    sft_trainer.py, rm_trainer.py, kd_trainer.py
    ray/
      launcher.py, ppo_actor.py, ppo_critic.py, vllm_worker.py
  utils/
    deepspeed/         # ZeRO-3 integration
    kl_controller.py   # Adaptive + Fixed KL controllers
```

구성 원리는 이렇다. `models/loss.py`는 `nn.Module` loss object들의 library다. `trainer/`의 trainer들은 data를 loss에 연결하고 distributed concern을 처리한다. `trainer/ray/`는 각 역할(actor, critic, reward, reference, vLLM rollout)을 자체 GPU placement가 있는 Ray actor class로 승격한다. 이것은 TRL(single `PPOTrainer`, Accelerate single-process)의 반대편이며, verl(HybridFlow dataflow graph, [[async-rollout]])의 사촌이다. OpenRLHF의 입장은 명확하다. scheduling은 Ray가 맡고, loss primitive는 `nn.Module`이며, trainer는 glue다.

---

## 2. PPO — PolicyLoss와 KL이 실제로 있는 곳

`openrlhf/models/loss.py`의 `PolicyLoss.forward` 전체 본문(68–168행)을 [[openrlhf-ppo]]에서 그대로 인용한다.

```python
class PolicyLoss(nn.Module):
    def __init__(self, clip_eps_low=0.2, clip_eps_high=0.2, dual_clip=None,
                 token_level_loss=True, policy_loss_type="ppo",
                 enable_vllm_is_correction=False,
                 vllm_is_truncated_threshold=None,
                 vllm_is_correction_type="tis"):
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.dual_clip = dual_clip
        self.token_level_loss = token_level_loss
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type

    def forward(self, log_probs, old_log_probs, advantages,
                action_mask=None, rollout_log_probs=None):
        if self.policy_loss_type == "ppo":
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            base = rollout_log_probs if self.enable_vllm_is_correction else old_log_probs
            log_ratio = log_probs - base
            ratio = (log_ratio * action_mask).sum(-1) / action_mask.sum(-1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
        if self.dual_clip is None:
            loss = -torch.min(surr1, surr2)
        else:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            loss = -torch.where(advantages < 0, clip2, clip1)

        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low, high = self.vllm_is_truncated_threshold
            log_ratio_v = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                vllm_is = torch.exp(log_ratio_v).detach()
                vllm_is = vllm_is * ((vllm_is >= low) & (vllm_is <= high))
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                seq_log_ratio = masked_mean(log_ratio_v, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low) & (seq_is <= high)
                vllm_is = torch.exp(log_ratio_v).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:  # "tis"
                vllm_is = torch.exp(log_ratio_v).clamp(min=low, max=high).detach()
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        loss = (masked_mean(loss, action_mask, dim=None) if self.token_level_loss
                else masked_mean(loss, action_mask, dim=-1).mean())
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl
```

눈여겨볼 네 가지가 있다. **(1) loss는 네 scalar를 반환한다.** `(loss, clip_ratio, ppo_kl, vllm_kl)`가 OpenRLHF의 idiom, 즉 loss-object-as-logger다. `ppo_kl`은 K1-style train-vs-old-logprob([[entropy-logging-patterns]])이고, `vllm_kl`은 rollout-vs-train-forward, 즉 sampler-drift diagnostic([[async-rollout]] failure signature)이다. **(2) loss 안에는 KL-to-reference가 없다.** 표준 RLHF([[rlhf-instructgpt]] Eq. 2)는 advantage computation *전에* `β · KL(π‖π_ref)`를 reward에 접어 넣는다. `PolicyLoss`가 보는 advantage에는 이미 penalty가 들어 있다. `β` knob은 한 abstraction 위의 controller에 있다([[kl-control-rlhf]]). **(3) Asymmetric + dual clip은 first-class다.** `clip_eps_low` / `clip_eps_high`는 DAPO / verl과 맞고, `dual_clip`은 negative-advantage token loss를 `dual_clip * advantages`에서 floor하여 한 trajectory가 거대한 negative update를 만들지 못하게 한다. **(4) vLLM IS correction은 세 갈래다.** `tis`(`[low, high]`로 per-token ratio clamp), `seq-mask-tis`(mean IS-weight가 band를 벗어나면 whole sequence mask), `icepop`(band 밖 token을 zero). 이는 bf16 vLLM이 trainer의 fp32 forward에서 diverge할 때 async rollout을 안전하게 만든다.

### 2.1 Per-token KL-to-reward 통합

KL penalty는 loss가 아니라 `openrlhf/trainer/ppo_trainer.py`에서 적용된다.

```python
# openrlhf/trainer/ppo_trainer.py  — KL controller and reward shaping
self.kl_ctl = (AdaptiveKLController(init_coef=init_beta, target=target_kl, horizon=horizon)
               if adaptive else FixedKLController(init_coef=init_beta))
# ... during make_experience, for each rollout token t:
#     reward_t = base_reward_t - kl_ctl.value * (log_pi(y_t|..) - log_pi_ref(y_t|..))
# after a trainer step:
self.kl_ctl.update(current=status["kl"], n_steps=rollout.batch_size * n_samples_per_prompt)
```

`AdaptiveKLController`([[kl-control-rlhf]] Technical Details)는 InstructGPT rule이다.

```
beta_new = beta_old * (1 + K_beta * clamp((KL_obs - KL_target) / KL_target, -0.2, 0.2))
```

관측 KL이 target보다 높으면 β가 올라가고, 낮으면 β가 내려간다. 이는 `ppo_trainer.py` 약 172행에서 instantiate된다([[openrlhf-ppo]] Context). `FixedKLController`는 `update()`를 무시하고 상수 β를 반환하는 stub이다. 이전 run에서 안정적인 β를 이미 아는 reward function에는 이것을 사용한다.

### 2.2 왜 KL-into-loss가 아니라 KL-into-reward인가

[[kl-control-rlhf]]는 이 점을 강하게 말한다. KL을 reward에 더하면 per-token advantage estimator가 잘 정의되어, GAE와 GRPO의 경우 group-relative z-score가 KL cost를 environment의 일부로 본다. TRL GRPO처럼 KL을 loss에 더하면([[entropy-logging-patterns]] comparison table) 그 깔끔한 분리가 깨지고, 경험적으로 다른 β regime이 필요해진다. OpenRLHF는 이 논쟁에서 InstructGPT 쪽을 택했고, 지금까지 그 입장을 유지했다. 이 codebase에서 가장 많이 복제된 결정이 바로 이것이다.

---

## 3. DPO — concatenated forward와 reference-policy 관리

`openrlhf/models/loss.py`의 `DPOLoss.forward`(231–257행)를 [[openrlhf-dpo]]에서 그대로 인용한다.

```python
class DPOLoss(nn.Module):
    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(self, policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps):
        pi_logratios  = policy_chosen_logps  - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2          # Azar 2023 IPO
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()
        chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return loss, chosen_rewards, rejected_rewards
```

training step(`openrlhf/trainer/dpo_trainer.py`, 약 150–185행, [[openrlhf-dpo]]에서 그대로 인용):

```python
chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens,
)
with torch.no_grad():
    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens,
    )
preference_loss, chosen_reward, reject_reward = self.loss_fn(
    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps,
)
if not self.aux_loss:  aux_loss = 0
if not self.nll_loss:  nll_loss = 0
loss = (
    preference_loss
    + aux_loss * self.args.model.aux_loss_coef
    + nll_loss * self.args.model.nll_loss_coef
)
self.strategy.backward(loss, self.model, self.optimizer)
self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
acc = (chosen_reward > reject_reward).float().mean().item()
```

하나의 loop 안에 production-grade trick 네 가지가 들어 있다([[openrlhf-dpo]] Context). **(1) `concatenated_forward`**는 chosen과 rejected를 batch axis에 stack한다. 한 번의 model forward가 두 logprob set을 모두 계산하므로 activation memory가 절반이 된다. `c_mask` / `r_mask`가 output logps를 다시 slice한다. **(2) Reference는 frozen + eval-mode + no-grad다**(`ref.eval()`, 전체 forward가 `torch.no_grad()` 안). ZeRO-3 사용자는 CPU로 offload한다. 이는 8xH100에서 70B DPO를 할 때 결정적이다. **(3) Optional NLL mixing** — `nll_loss_coef > 0`이면 chosen response에 대한 추가 CE term(RPO / SimPO-Mix, [[dpo]] length-hack fix)이 `logπ_chosen` collapse를 막는다. **(4) MoE aux-loss가 보존된다** — Mixtral / DeepSeek-MoE router는 이것 없이는 수백 step 안에 unbalanced해진다. 그 collapse는 `preference_loss`에는 나타나지 않는다.

Pair batching: dataset은 `(prompt, chosen, rejected)`를 산출한다. collator는 chosen과 rejected를 독립적으로 pad하고(variable length), `chosen_ids / c_mask / reject_ids / r_mask / prompt_id_lens`를 내보낸다. `concatenated_forward`는 dim 0을 따라 concatenate한다.

---

## 4. Ray actor-pool orchestration

OpenRLHF의 대표적인 움직임은 RLHF stage의 각 역할을 자체 GPU placement가 있는 Ray actor로 승격하는 것이다. interactive graph는 [[figures/openrlhf-ray.html]]을 보라. canonical PPO job은 다섯 actor class를 instantiate한다.

| Actor | 역할 | Count | 분리하는 이유 |
|---|---|---|---|
| `PolicyActor` (`ppo_actor.py`) | π_θ를 train. forward + backward + ZeRO-3 partition | 1 pool, N GPUs | Main compute. training-grade memory layout 필요 |
| `ValueActor` (`ppo_critic.py`) | V_ψ를 train. returns에 대한 MSE | 1 pool | backward가 policy와 경쟁하지 않게 분리 |
| `ReferenceActor` | `logπ_ref` 제공, frozen | 1(offload 가능) | eval-mode only. footprint가 작음 |
| `RewardActor` | rollout에 RM forward 제공 | 1 | trainer와 분리해 forward-step contention 방지 |
| `vLLMWorker` (`vllm_worker.py`) | vLLM으로 rollout 생성 | M workers | vLLM은 자체 KV-cache manager가 있음. trainer와 co-tenant하면 throughput이 죽음 |

async PPO variant([[async-rollout]], `ppo_trainer_async.py`)는 세 가지 Ray primitive를 추가한다.

- `rollout_queue`: `ray.util.queue.Queue`이며 capacity는 1–2다. 완료된 rollout을 `vLLMWorker`에서 `PolicyActor`로 운반한다.
- `rollout_slots`: `global_step` token을 운반하는 companion queue다. rollout worker는 시작 전에 `slots.get()`에서 block하고, 결과를 `rollout_queue`에 push한다. 이것이 textbook backpressure다. 없으면 rollout worker가 앞서 달려 trainer가 stale data를 받는다.
- `vllm_lock`: Ray `asyncio.Lock`으로, weight-broadcast-to-vLLM과 in-flight generation을 serialize한다. 이것이 없으면 generation 중간의 broadcast 때문에 response의 일부 token은 W_t에서, 나머지는 W_{t+1}에서 sampling되어 rollout을 조용히 bias한다.

Partial rollout(`strategy.args.train.partial_rollout_enable`)은 새 weight set이 broadcast될 때 in-flight response를 interrupt하게 해준다. response는 다음 cycle에서 새 policy로 마무리되며, IS-correction path에 표시되어 `vllm_kl`이 의미를 유지한다.

Throughput: OpenRLHF paper(Hu 2024, §3.3)는 async가 sync mode 대비 7B에서 1.9×, 70B에서 1.6× speedup을 보인다고 보고한다([[async-rollout]] Key Figures).

### 4.1 OpenRLHF가 우회해야 하는 Ray quirk들

자주 문제가 되는 것은 네 가지다. **placement groups**(ZeRO-3 partition이 NVLink-local로 남도록 PACK. many-small-node cluster에서는 manual bundle과 STRICT_PACK), **weight broadcast는 Ray object-store가 아니라 NCCL**(OpenRLHF는 vLLM 내부 NCCL broadcast를 `ray.get(worker.update_weights.remote(...))`로 감싸고 `vllm_lock`으로 serialize한다), **actor restart는 KV cache를 잃는다**(production에서는 vLLM auto-restart를 꺼라. long-reasoning rollout에서는 otherwise multi-minute hole이 생긴다), **`ray.init(address="auto")` race**(worker `ray start` 전에 head가 떠 있어야 한다. bundle misconfiguration은 silently-empty placement group을 만든다).

---

## 5. Defaults + 코드를 건드리지 않고 조정할 수 있는 것

[[openrlhf-entropy-debugging]] Technical Details 기준:

| Knob | OpenRLHF default | 위치 | 이유 |
|---|---|---|---|
| `entropy_coef` | `0.0` | PPO trainer | KL-to-ref가 이미 regularize함. LLM RL에서 entropy bonus는 드묾 |
| KL mode | `adaptive` (AdaptiveKLController) | `ppo_trainer.py` 약 172행 | 새 reward function에 더 안전 |
| `β_init` | 0.01(adaptive) / 0.1(fixed) | `kl_controller.py` | [[kl-control-rlhf]] range |
| Clip ε | 0.2 symmetric(또는 asymmetric 0.2/0.28) | `PolicyLoss` | DAPO-style asymmetric 노출 |
| Rollout T | 1.0, top_p=1.0 | vLLM worker | [[openrlhf-entropy-debugging]] Technical Details |
| Advantage norm | on, per-batch zero-mean unit-var | trainer | 기본 ON(TRL은 OFF — recurring footgun) |
| DPO β | 0.1 | DPO trainer | [[dpo]] practical default |
| Ref offload | ZeRO-3에서 on | DeepSpeed utils | 70B-class memory |

---

## 6. OpenRLHF vs verl — tradeoff table

어떤 framework를 고를지 결정하는 infra-track 비교([[openrlhf-ppo]] "Comparison to paper / to other frameworks", [[async-rollout]] Key Contributions):

| Concern | OpenRLHF | verl | Notes |
|---|---|---|---|
| Communication model | Ray actors + queues | HybridFlow dataflow graph | Verl은 DAG를 compile하고, OpenRLHF는 actor를 procedural하게 compose |
| Sharding | DeepSpeed ZeRO-1/2/3 | FSDP2 + Megatron(3D) | Verl은 tensor-parallel trainer를 out of the box 지원 |
| KL placement | reward shaping(token별) | reward shaping, K1/K2/K3 switch | 같은 입장. verl은 estimator를 string으로 노출 |
| PPO loss object | `PolicyLoss` nn.Module → `(loss, clip, ppo_kl, vllm_kl)` | `compute_policy_loss_vanilla` + `agg_loss` | 대수적으로 동등 |
| GSPO / dual-clip / IS corr | `PolicyLoss`에 built-in | policy-loss registry | 2025년 기준 parity |
| Rollout mode | sync 또는 Ray-queue async | sync, async, partial-rollout | verl의 partial-rollout이 unique |
| Weight broadcast | NCCL + `vllm_lock` | `engine.pause()` mid-decode | verl은 draining 없이 broadcast |
| Config / multi-node | Python dataclasses; Ray-native | Hydra YAML; Ray 또는 torchrun | Ray cluster에서는 OpenRLHF가 빛남 |
| Defaults (entropy / adv-norm) | 0.0 / ON | 0.0(일부 1e-3) / ON | [[openrlhf-entropy-debugging]] |
| Codebase size (2026) | ~60K LOC | ~80K LOC | verl의 scope가 더 넓음 |

정성적 요약: **OpenRLHF = Ray-native simplicity + batteries-included PPO/DPO**; **verl = dataflow-graph flexibility + aggressive async / partial-rollout tricks**. KubeRay에서는 OpenRLHF가 friction이 낮다. continuous-batching RL이나 3D-parallel trainer에서는 verl이 여전히 앞서 있다.

---

## 7. Failure-mode map — OpenRLHF가 깨지는 곳

issue tracker와 [[openrlhf-entropy-debugging]]이 catalog한 네 가지 pattern: **<100 step entropy collapse**(β가 너무 작거나 KL-to-ref misconfigured. `ppo_kl` > 0.1이고 `clipfrac`가 1에 붙었는지 확인. `β_init`을 올리고 controller `K_beta`를 낮춘다), **`vllm_kl` divergence**(rollout-vs-train sampler drift. `enable_vllm_is_correction=True`, `vllm_is_correction_type="tis"`, `vllm_is_truncated_threshold=(0.5, 2.0)`를 켜고 async queue depth를 줄인다), **70B DPO에서 reference-model OOM**(ref의 ZeRO-3 CPU-offload가 기본으로 꺼져 있음. DeepSpeed config에서 켠다), **Ray placement-group hang**(worker 전에 head node가 떠 있지 않거나 `gpus_per_actor` 합이 cluster를 초과. `ray status`로 진단한다. launcher는 misconfigured bundle에 fail fast하지 않는다).

---

## 8. Connections

- **ch-55 / ch-57 / ch-58** — verl internals(나란히 읽을 것), TRL internals(Accelerate single-process contrast), framework-comparison matrix(§6이 그 시작 draft).
- **ch-38 / ch-39 / ch-40 / ch-41** — DPO(`concatenated_forward`가 production template), PPO, GRPO, vLLM-drift([[openrlhf-ppo]]의 IS-correction branch가 ch-41의 failure mode를 고친다).
- **ch-42 / ch-43** — entropy collapse와 reward hacking. OpenRLHF의 네 logged signal은 instrumentation substrate다.
- **[[kl-control-rlhf]]** — KL-into-reward vs KL-into-loss. OpenRLHF는 reward 쪽을 택했다. **[[async-rollout]]** — queue + `vllm_lock` architecture.

## Companion visualization

**[figures/openrlhf-ray.html](figures/openrlhf-ray.html)** — 각 Ray actor를 클릭하면 responsibility, message, GPU footprint, failure mode를 볼 수 있다. §1의 module tree와 §2.1의 KL-placement diagram을 physical cluster layout 위에 매핑한다.
