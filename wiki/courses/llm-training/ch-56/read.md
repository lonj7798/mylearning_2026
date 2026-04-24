<!-- chapter: ch-56
     track: infra
     kind: content
     title: OpenRLHF Internals
     deps: [ch-55]
     sources: [[openrlhf-ppo]], [[openrlhf-dpo]], [[entropy-logging-patterns]], [[openrlhf-entropy-debugging]], [[async-rollout]], [[ppo]], [[dpo]], [[rlhf-instructgpt]], [[kl-control-rlhf]]
     figures: figures/openrlhf-ray.html
-->

# Chapter 56 — OpenRLHF Internals

> **Core insight.** OpenRLHF is the open-source stack that first made the `(token-level KL-into-reward) + (clipped policy loss) + (Ray-pool orchestration)` factorization practical at 70B scale. Every design choice — `PolicyLoss` returning two KLs instead of one, the `AdaptiveKLController` that lives *outside* the loss, `concatenated_forward` halving DPO activation memory, the `rollout_queue` / `vllm_lock` pair letting generation and training overlap without weight-swap races — traces back to one stance: *compose specialist Ray actors, keep the loss dumb, make reward shaping and the KL controller first-class citizens of the trainer*.
>
> **Guideline.** Read OpenRLHF as Ray actors communicating through queues, not a monolithic trainer. `PolicyLoss` is minimal — clip + dual-clip + optional IS correction. The KL controller sits in `ppo_trainer.py` and mutates the *reward*, not the loss. DPO keeps a frozen reference in eval mode, ZeRO-3-offloaded when memory is tight. vs verl: OpenRLHF wins on simplicity and Ray-native orchestration; verl wins on dataflow-graph flexibility and partial-rollout. Pick OpenRLHF for a Ray-managed cluster with batteries-included PPO/DPO; pick verl for continuous-batching RL at scale.

---

## 1. Repo tour — what lives where

OpenRLHF ([[openrlhf-ppo]]) is organized around three top-level packages under `openrlhf/`:

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

Organizing principle: `models/loss.py` is a library of `nn.Module` loss objects; trainers in `trainer/` wire data into the loss and handle distributed concerns; `trainer/ray/` promotes each role (actor, critic, reward, reference, vLLM rollout) into a Ray actor class with its own GPU placement. This is the opposite of TRL (single `PPOTrainer`, Accelerate single-process) and a cousin of verl (HybridFlow dataflow graph, [[async-rollout]]). OpenRLHF's stance: Ray schedules, `nn.Module` is the loss primitive, the trainer is glue.

---

## 2. PPO — PolicyLoss and where the KL actually lives

The full `PolicyLoss.forward` body from `openrlhf/models/loss.py` (lines 68–168), quoted verbatim from [[openrlhf-ppo]]:

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

Four things worth staring at: **(1) The loss returns four scalars**, `(loss, clip_ratio, ppo_kl, vllm_kl)` — the OpenRLHF idiom, loss-object-as-logger. `ppo_kl` is K1-style train-vs-old-logprob ([[entropy-logging-patterns]]); `vllm_kl` is rollout-vs-train-forward, the sampler-drift diagnostic ([[async-rollout]] failure signature). **(2) No KL-to-reference inside the loss.** Standard RLHF ([[rlhf-instructgpt]] Eq. 2) folds `β · KL(π‖π_ref)` into the reward *before* advantage computation; `PolicyLoss` sees advantages that already encode the penalty. The `β` knob lives on a controller one abstraction up ([[kl-control-rlhf]]). **(3) Asymmetric + dual clip are first-class.** `clip_eps_low` / `clip_eps_high` match DAPO / verl; `dual_clip` floors negative-advantage token loss at `dual_clip * advantages`, preventing one trajectory from producing a huge negative update. **(4) vLLM IS correction is three-way**: `tis` (clamp per-token ratio to `[low, high]`), `seq-mask-tis` (mask whole sequence if mean IS-weight exits the band), `icepop` (zero tokens outside the band). This keeps async rollout safe when vLLM at bf16 diverges from the trainer's forward at fp32.

### 2.1 The per-token KL-to-reward integration

The KL penalty is applied in `openrlhf/trainer/ppo_trainer.py`, not in the loss:

```python
# openrlhf/trainer/ppo_trainer.py  — KL controller and reward shaping
self.kl_ctl = (AdaptiveKLController(init_coef=init_beta, target=target_kl, horizon=horizon)
               if adaptive else FixedKLController(init_coef=init_beta))
# ... during make_experience, for each rollout token t:
#     reward_t = base_reward_t - kl_ctl.value * (log_pi(y_t|..) - log_pi_ref(y_t|..))
# after a trainer step:
self.kl_ctl.update(current=status["kl"], n_steps=rollout.batch_size * n_samples_per_prompt)
```

The `AdaptiveKLController` ([[kl-control-rlhf]] Technical Details) is the InstructGPT rule:

```
beta_new = beta_old * (1 + K_beta * clamp((KL_obs - KL_target) / KL_target, -0.2, 0.2))
```

If observed KL runs above target, β rises; below target, β falls. This is instantiated around line 172 of `ppo_trainer.py` ([[openrlhf-ppo]] Context). The `FixedKLController` is a stub that ignores `update()` and returns a constant β — use it for reward functions where you already know the stable β from a prior run.

### 2.2 Why KL-into-reward, not KL-into-loss

[[kl-control-rlhf]] is emphatic: adding KL to the reward keeps the per-token advantage estimator well-defined, so GAE (and for GRPO, the group-relative z-score) sees the KL cost as part of the environment. Adding KL to the loss, as TRL GRPO does ([[entropy-logging-patterns]] comparison table), breaks that clean separation and empirically requires a different β regime. OpenRLHF picked the InstructGPT side of this debate and has stayed there — it's the single most copied decision from this codebase.

---

## 3. DPO — concatenated forward and reference-policy management

`DPOLoss.forward` from `openrlhf/models/loss.py` (lines 231–257), quoted verbatim from [[openrlhf-dpo]]:

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

The training step (`openrlhf/trainer/dpo_trainer.py`, lines ~150–185, verbatim from [[openrlhf-dpo]]):

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

Four production-grade tricks in one loop ([[openrlhf-dpo]] Context): **(1) `concatenated_forward`** stacks chosen and rejected along the batch axis — one model forward computes both sets of logprobs, activation memory halved; `c_mask` / `r_mask` slice the output logps back. **(2) Reference is frozen + eval-mode + no-grad** (`ref.eval()`, whole forward inside `torch.no_grad()`); ZeRO-3 users offload to CPU — critical for 70B DPO on 8xH100. **(3) Optional NLL mixing** — with `nll_loss_coef > 0`, an extra CE term on the chosen response (RPO / SimPO-Mix, [[dpo]] length-hack fix) stops `logπ_chosen` from collapsing. **(4) MoE aux-loss preserved** — Mixtral / DeepSeek-MoE routers go unbalanced in a few hundred steps without this; the collapse doesn't show up in `preference_loss`.

Pair batching: dataset yields `(prompt, chosen, rejected)`; collator pads chosen and rejected independently (variable lengths), emits `chosen_ids / c_mask / reject_ids / r_mask / prompt_id_lens`; `concatenated_forward` concatenates along dim 0.

---

## 4. Ray actor-pool orchestration

OpenRLHF's signature move is promoting each role in an RLHF stage to a Ray actor with its own GPU placement. See [[figures/openrlhf-ray.html]] for the interactive graph. The canonical PPO job instantiates five actor classes:

| Actor | Role | Count | Why separate |
|---|---|---|---|
| `PolicyActor` (`ppo_actor.py`) | Trains π_θ; forward + backward + ZeRO-3 partition | 1 pool, N GPUs | Main compute; needs training-grade memory layout |
| `ValueActor` (`ppo_critic.py`) | Trains V_ψ; MSE on returns | 1 pool | Separate so backward doesn't contend with policy |
| `ReferenceActor` | Serves `logπ_ref`, frozen | 1 (offloadable) | Eval-mode only; tiny footprint |
| `RewardActor` | Serves RM forward on rollouts | 1 | Sharded away from trainer to avoid forward-step contention |
| `vLLMWorker` (`vllm_worker.py`) | Generates rollouts via vLLM | M workers | vLLM has its own KV-cache manager; co-tenanting with a trainer kills throughput |

The async PPO variant ([[async-rollout]], `ppo_trainer_async.py`) adds three Ray primitives:

- `rollout_queue`: a `ray.util.queue.Queue` with capacity 1–2 carrying finished rollouts from `vLLMWorker` to `PolicyActor`.
- `rollout_slots`: a companion queue carrying `global_step` tokens; the rollout worker blocks on `slots.get()` before starting, then pushes the result to `rollout_queue`. This is textbook backpressure — without it, the rollout worker runs ahead and the trainer gets stale data.
- `vllm_lock`: a Ray `asyncio.Lock` that serializes weight-broadcast-to-vLLM against in-flight generation. Without it, a broadcast mid-generation means some tokens in a response were sampled from weights W_t and others from W_{t+1}, silently biasing the rollout.

Partial rollout (`strategy.args.train.partial_rollout_enable`) lets an in-flight response be interrupted when a new weight set broadcasts; the response is finished by the new policy on the next cycle and is marked in the IS-correction path so `vllm_kl` remains meaningful.

Throughput: the OpenRLHF paper (Hu 2024, §3.3) reports 1.9× speedup for 7B, 1.6× for 70B in async vs sync mode ([[async-rollout]] Key Figures).

### 4.1 Ray quirks OpenRLHF has to work around

Four that bite: **placement groups** (PACK so ZeRO-3 partition stays NVLink-local; STRICT_PACK with manual bundles for many-small-node clusters); **weight broadcast is NCCL, not Ray object-store** (OpenRLHF wraps `ray.get(worker.update_weights.remote(...))` around an NCCL broadcast inside vLLM, serialized by `vllm_lock`); **actor restart costs the KV cache** (disable vLLM auto-restart in production — long-reasoning rollouts pay multi-minute holes otherwise); **`ray.init(address="auto")` race** (head must be up before worker `ray start`; misconfigured bundles yield silently-empty placement groups).

---

## 5. Defaults + what OpenRLHF lets you tune without touching code

From [[openrlhf-entropy-debugging]] Technical Details:

| Knob | OpenRLHF default | Where | Why |
|---|---|---|---|
| `entropy_coef` | `0.0` | PPO trainer | KL-to-ref already regularizes; entropy bonus rare for LLM RL |
| KL mode | `adaptive` (AdaptiveKLController) | `ppo_trainer.py` ~line 172 | Safer for new reward functions |
| `β_init` | 0.01 (adaptive) / 0.1 (fixed) | `kl_controller.py` | [[kl-control-rlhf]] range |
| Clip ε | 0.2 symmetric (or asymmetric 0.2/0.28) | `PolicyLoss` | DAPO-style asymmetric exposed |
| Rollout T | 1.0, top_p=1.0 | vLLM worker | [[openrlhf-entropy-debugging]] Technical Details |
| Advantage norm | on, per-batch zero-mean unit-var | trainer | ON by default (TRL is OFF — recurring footgun) |
| DPO β | 0.1 | DPO trainer | [[dpo]] practical default |
| Ref offload | on with ZeRO-3 | DeepSpeed utils | 70B-class memory |

---

## 6. OpenRLHF vs verl — the tradeoff table

The infra-track comparison that governs which framework to pick ([[openrlhf-ppo]] "Comparison to paper / to other frameworks", [[async-rollout]] Key Contributions):

| Concern | OpenRLHF | verl | Notes |
|---|---|---|---|
| Communication model | Ray actors + queues | HybridFlow dataflow graph | Verl compiles a DAG; OpenRLHF composes actors procedurally |
| Sharding | DeepSpeed ZeRO-1/2/3 | FSDP2 + Megatron (3D) | Verl supports tensor-parallel trainer out of the box |
| KL placement | reward shaping (per-token) | reward shaping, K1/K2/K3 switch | Same stance, verl exposes estimator as a string |
| PPO loss object | `PolicyLoss` nn.Module → `(loss, clip, ppo_kl, vllm_kl)` | `compute_policy_loss_vanilla` + `agg_loss` | algebraically equivalent |
| GSPO / dual-clip / IS corr | built into `PolicyLoss` | policy-loss registry | parity as of 2025 |
| Rollout mode | sync or Ray-queue async | sync, async, partial-rollout | verl's partial-rollout is unique |
| Weight broadcast | NCCL + `vllm_lock` | `engine.pause()` mid-decode | verl broadcasts without draining |
| Config / multi-node | Python dataclasses; Ray-native | Hydra YAML; Ray or torchrun | OpenRLHF shines on Ray clusters |
| Defaults (entropy / adv-norm) | 0.0 / ON | 0.0 (some 1e-3) / ON | [[openrlhf-entropy-debugging]] |
| Codebase size (2026) | ~60K LOC | ~80K LOC | verl is broader in scope |

Qualitative read: **OpenRLHF = Ray-native simplicity + batteries-included PPO/DPO**; **verl = dataflow-graph flexibility + aggressive async / partial-rollout tricks**. On KubeRay, OpenRLHF is lower-friction; for continuous-batching RL or 3D-parallel trainers, verl is still ahead.

---

## 7. Failure-mode map — where OpenRLHF breaks

Four patterns the issue tracker and [[openrlhf-entropy-debugging]] catalog: **entropy collapse in <100 steps** (β too small / KL-to-ref misconfigured — check `ppo_kl` > 0.1 with `clipfrac` pegged at 1; raise `β_init`, lower controller `K_beta`); **`vllm_kl` divergence** (rollout-vs-train sampler drift — enable `enable_vllm_is_correction=True` with `vllm_is_correction_type="tis"` and `vllm_is_truncated_threshold=(0.5, 2.0)`, reduce async queue depth); **reference-model OOM on 70B DPO** (ZeRO-3 CPU-offload of the ref is off by default — turn it on in the DeepSpeed config); **Ray placement-group hangs** (head node not up before workers, or summed `gpus_per_actor` exceeds cluster — `ray status` diagnoses; the launcher does not fail fast on misconfigured bundles).

---

## 8. Connections

- **ch-55 / ch-57 / ch-58** — verl internals (read side-by-side), TRL internals (Accelerate single-process contrast), framework-comparison matrix (§6 is its starting draft).
- **ch-38 / ch-39 / ch-40 / ch-41** — DPO (`concatenated_forward` is the production template), PPO, GRPO, vLLM-drift ([[openrlhf-ppo]]'s IS-correction branches fix ch-41's failure mode).
- **ch-42 / ch-43** — entropy collapse and reward hacking; OpenRLHF's four logged signals are the instrumentation substrate.
- **[[kl-control-rlhf]]** — KL-into-reward vs KL-into-loss; OpenRLHF took the reward side. **[[async-rollout]]** — queue + `vllm_lock` architecture.

## Companion visualization

**[figures/openrlhf-ray.html](figures/openrlhf-ray.html)** — click each Ray actor to see its responsibilities, messages, GPU footprint, and failure modes. Maps the module tree in §1 and the KL-placement diagram in §2.1 onto the physical cluster layout.
