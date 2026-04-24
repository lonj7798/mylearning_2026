<!-- chapter: ch-57
     track: infra
     kind: content
     title: TRL Internals
     deps: [ch-56]
     sources: [[trl-ppo]], [[trl-grpo]], [[trl-online-dpo]], [[hf-alignment-handbook]], [[hf-dpo-zoo]], [[hf-rlhf-illustrated]], [[grpo]], [[dpo]], [[ppo]]
     figures: figures/trl-stack.html
-->

# Chapter 57 — TRL Internals

> **Core insight.** TRL is what happens when you refuse to write a distributed-systems layer. Every trainer is a subclass of `transformers.Trainer`; every distributed primitive comes from `accelerate`; every model is a `PreTrainedModel`; every dataset is a `datasets.Dataset`. The entire repo is the HF ecosystem wearing an RL hat. That is why it is the fastest path from a chat-template idea to a running Zephyr-sized run, and why it falls off a cliff past a single node: there is no scheduler, no weight-transfer protocol, no rollout-worker pool. verl ([[ch-55]]) bet on Ray + a custom controller; OpenRLHF ([[ch-56]]) bet on Ray + colocated vLLM; TRL bet on `accelerate launch`, and that bet shapes every tradeoff in the repo.
>
> **Guideline.** Default to TRL for anything that fits on one node (≤ 8×H100). Use `SFTTrainer` with `packing=True, train_on_response_only=True`; use `DPOTrainer` with `loss_type="sigmoid"` for offline preferences or `GRPOTrainer` with `loss_type="dr_grpo"` for verifiable rewards. Treat the `experimental/` folder as unstable but canonical — it is where the active RL algorithms live. **Outgrow TRL** the moment you need (a) multi-node rollouts, (b) asynchronous actor/rollout decoupling, or (c) a custom advantage estimator that does not fit the `loss_type` switch. At that point the cost of porting to verl is smaller than the cost of patching TRL.

---

## §1 Repo tour — stable vs experimental

As of April 2026 `huggingface/trl` is organized into a few top-level packages under `trl/`:

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

**The stable/experimental split is load-bearing.** Code in `trl/trainer/` ships with semver guarantees: `GRPOTrainer`, `DPOTrainer`, `SFTTrainer`, `RLOOTrainer`, `RewardTrainer`, `KTOTrainer`, `ORPOTrainer`. Code in `trl/experimental/` is where the active research lands before anything is promised. That includes the actor-critic PPO trainer (which *used to* be mainline, then was demoted after the 2024 pivot to critic-free methods — see [[trl-ppo]]), the online-DPO trainer, and the Nash / XPO self-play variants.

The `trl/core/` module holds shared PPO utilities — reward-shaping helpers, adaptive-KL controller, advantage whitening — that predate `accelerate` becoming the distributed backbone. A lot of it is dead code kept alive by the experimental trainers. You will see it imported from `ppo_trainer.py` and nowhere else.

**The `trl.DPOTrainer` trick** ([[hf-dpo-zoo]]). One trainer class handles the entire DPO family via a `loss_type` string: `"sigmoid"` (vanilla DPO), `"ipo"` (Azar identity), `"kto"` (prospect theory unary), `"simpo"` (reference-free length-normalized), `"orpo"` (joint SFT+odds-ratio), `"bco"` (binary classifier), `"cpo"` (contrastive). The algebra differs; the dataloader and distributed code path are identical. This is the entire design philosophy of TRL: pick one trainer skeleton, expose every variant as a string argument, and let HuggingFace's own `Trainer` handle the rest.

---

## §2 The three RL trainers — signatures and what they actually do

### SFTTrainer (the stable anchor, [[hf-alignment-handbook]])

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

Under the hood this is just `transformers.Trainer` with (a) chat-template-aware preprocessing, (b) optional packing via `ConstantLengthDataset`, (c) prompt-token masking that sets the label tensor to `-100` on all non-response positions. The actual forward/backward/optimizer step is inherited wholesale from HF `Trainer`, which means `accelerate` handles DDP / FSDP / DeepSpeed transparently.

### DPOTrainer

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(beta=0.1, loss_type="sigmoid", max_length=2048,
                max_prompt_length=1024)
trainer = DPOTrainer(model=policy, ref_model=ref_policy, args=cfg,
                     train_dataset=preference_ds, processing_class=tokenizer)
```

DPO needs *two* models in memory — the policy `π_θ` being trained and the frozen reference `π_ref` for the log-ratio denominator in [[dpo]] Eq. 7. When using PEFT/LoRA, `ref_model` is set to `None`: TRL computes `π_ref` by disabling the adapters on the base model, saving one full model's worth of VRAM. That trick lives in `dpo_trainer.py` under `null_ref_context()`.

### GRPOTrainer (the monster, [[trl-grpo]])

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

`GRPOTrainer` is a ~2700-LOC file. Its `_compute_loss` (~L2418–2610) is the single largest method in TRL and contains every loss variant in the GRPO zoo. The class does three jobs in one loop:

1. **Generate.** Either via HF `.generate()` (slow, correct), or via vLLM in `server` mode (remote), or `colocate` mode (same GPU, shared tensors). Generation produces `completion_ids` and optionally `old_per_token_logps` for the IS ratio.
2. **Score.** Each reward function in `reward_funcs` is called with `(prompts, completions)` and returns a `(B,)` reward tensor. Multiple reward functions sum with configurable weights.
3. **Update.** `_compute_loss` recomputes per-token logprobs, builds the group-relative advantage `(r_i − mean(r_{1..G})) / std`, forms the clipped PPO ratio, optionally filters to the top-entropy quantile, adds K3 KL to the reference, aggregates with the `loss_type`-specific denominator, and returns a scalar.

This is a different design axis from verl, which splits advantage computation and loss computation into two registry hooks. TRL fuses them; verl splits them ([[ch-55]]). Algebraic result is identical for `loss_type="grpo"`.

### Online DPO (experimental, [[trl-online-dpo]])

```python
from trl.experimental.online_dpo import OnlineDPOTrainer, OnlineDPOConfig
cfg = OnlineDPOConfig(beta=0.1, loss_type="sigmoid",
                      missing_eos_penalty=1.0, use_vllm=True)
trainer = OnlineDPOTrainer(model=policy, ref_model=ref,
                           reward_model=rm, judge=None,
                           args=cfg, train_dataset=prompt_ds,
                           processing_class=tokenizer)
```

Every step samples *two* completions per prompt (`prompts = 2 * prompts`), scores both with `reward_funcs` or a `Judge`, declares the higher-scoring one `chosen` and the lower one `rejected`, and runs a DPO gradient step on that fresh pair. Pairs are never stored. This closes the on-policy gap of offline DPO — the reference is actually the current policy's neighborhood — at the cost of one RM forward per completion per step. [[trl-online-dpo]] includes the `training_step` excerpt showing the argmax pair selection and the sequence-level log-ratio loss.

---

## §3 Accelerate-based orchestration — and what breaks at scale

Every TRL trainer is a `transformers.Trainer` subclass, which means the distributed story is `accelerate`'s. That is great for portability and terrible for scale. Here is the failure chain when you push past one node.

### 3.1 What Accelerate does well

`accelerate launch` picks one of `{DDP, FSDP, DeepSpeed ZeRO-1/2/3}` based on a YAML config and wraps your model. A single Python process per GPU, no external scheduler, no Ray, no MPI. The training loop is just a plain Python `for batch in dataloader`. For single-node runs this is **the fastest possible setup time**: write the config, launch, done. The Alignment Handbook ([[hf-alignment-handbook]]) ships 8×A100 FSDP configs for Zephyr-7B SFT and DPO that work out of the box.

### 3.2 What breaks the moment rollouts enter

The moment the trainer needs to *generate*, the single-process model breaks. A full forward pass through the policy at 1024 tokens × 8 completions × batch-of-prompts takes seconds per step. If every rank is synchronously waiting for one rank's `.generate()`, throughput collapses. TRL's answers are three escape hatches, none of them uniformly satisfying:

1. **HF `.generate()`** — straightforward, slow, uses the same sharded model as training. With FSDP, every generation gathers all parameters back, then re-shards. This is the correctness path; it is also the slowest.
2. **vLLM server mode** (`vllm_mode="server"`). A separate vLLM process runs on dedicated GPUs; TRL ships a weight-sync client (`vllm_client`) that pushes updated weights after every optimizer step. Faster rollouts, but now you need to provision extra GPUs for the server, and weight transfer is O(model size) per step.
3. **vLLM colocate mode** (`vllm_mode="colocate"`). vLLM runs *inside the training process* on the same GPUs. Shared weight memory, no network transfer, but you now pay the cost of vLLM's paged-attention memory manager even while doing backward. This is the mode actively maintained in 2025; the server mode is being phased out.

Compare this with [[ch-55]] verl's architecture: a dedicated `RolloutWorker` pool runs vLLM/SGLang on its own GPU set; a `CriticWorker` and an `ActorWorker` live on separate sets; Ray handles scheduling and weight sync through `DistributedRollout` and `HybridEngine`. Or [[ch-56]] OpenRLHF's colocated-Ray setup with a unified actor-rollout worker. Those frameworks chose a *distributed systems primitive* (Ray) and built around it. TRL refused to. That refusal is the whole story of where TRL's scaling limits come from.

### 3.3 Straggler behaviour and the single-controller problem

Because every rank runs the same Python loop, any rank that takes longer — because its sampled completions happened to be longer, because its vLLM batch was unlucky, because its reward function made a slow HTTP call — blocks the whole collective. Accelerate has no built-in straggler mitigation. Workarounds in practice:

- **Pad every completion to `max_completion_length`** so ranks finish generation at the same time. Wastes compute.
- **Set a hard generation timeout** and discard timed-out rollouts. Biases the distribution.
- **Move reward functions to background threads** when they are I/O-bound (LLM-as-judge, API calls).

None of these is elegant. verl handles stragglers through its hybrid engine; OpenRLHF tolerates them because Ray can reschedule. TRL sits and waits.

### 3.4 No native multi-node beyond `accelerate launch` + SLURM

Multi-node TRL is "write a SLURM script that calls `accelerate launch` on each node with the right `--machine_rank`". There is no built-in fault tolerance, no weight-transfer protocol beyond NCCL, no way to add or remove workers mid-run. If a node dies, the whole run dies. For < 16 GPUs this is fine. For 64+ GPUs it becomes a serious operational burden.

---

## §4 HF ecosystem integrations — PEFT, datasets, transformers

### 4.1 PEFT-LoRA as the eval-time reference-policy trick

TRL's DPO/GRPO trainers detect when the policy is a `PeftModel`. If so, they skip loading a separate `ref_model`: instead, the reference policy is computed by temporarily disabling the LoRA adapters on the base model (`with adapter_model.disable_adapter()`). Saves a whole model's worth of VRAM. This is the single most common TRL trick in the wild: LoRA + DPO = one 7B model in memory instead of two, [[dpo]] loss unchanged.

```python
# simplified pattern used by DPOTrainer / GRPOTrainer when ref_model=None
def null_ref_context(self):
    with self.accelerator.unwrap_model(self.model).disable_adapter():
        yield
# later:
with self.null_ref_context():
    ref_logits = self.model(**batch).logits
```

This trick is **exclusive to PEFT**. Full fine-tuning still needs two copies. For 70B models this is one of the reasons DPO full-FT is so expensive and LoRA-DPO has become dominant.

### 4.2 `datasets` for preprocessing

Every TRL trainer accepts `datasets.Dataset` or `IterableDataset`. The SFT path uses `datasets.map(tokenize, batched=True, remove_columns=...)` then optionally `ConstantLengthDataset` for packing. DPO expects columns `{prompt, chosen, rejected}`. GRPO expects `{prompt}` plus a per-prompt `reward_funcs` call. Online DPO expects `{prompt}` and generates everything else online.

The eval-time implication: because everything is a `datasets.Dataset`, you get streaming (for corpora too big for RAM), built-in Arrow-backed caching, and cross-process sharding via `split_dataset_by_node` without writing any distributed-IO code. That is a huge productivity win. The cost is that any non-standard data format has to be coerced into `{column: value}` rows first.

### 4.3 `transformers` chat templates

TRL's SFT and DPO trainers call `tokenizer.apply_chat_template(...)` to format messages into a single string. Every mismatch between the chat template in training and inference is a silent bug — [[hf-alignment-handbook]]'s top-listed lesson is "always decode a packed batch to verify your chat template." TRL does not enforce this; it just passes through.

### 4.4 Eval-time implications

Because TRL artifacts are `transformers`-native, the model ships directly to HF inference code (`pipeline`, `.generate()`, vLLM's HF-compatible loader) with zero conversion step. Contrast with verl, which ships checkpoints in its own format and requires a conversion script for vLLM inference. The HF ecosystem story is the other half of why TRL wins for prototyping: the thing you train is the thing you serve.

---

## §5 PPO, the legacy reference (still in experimental/)

The classic actor-critic PPO lives at `trl/experimental/ppo/ppo_trainer.py`. The inner loop (~L820–870, [[trl-ppo]]) is the cleanest single-file statement of the InstructGPT ([[hf-rlhf-illustrated]]) recipe in open source:

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

Key architectural fact: the policy and value head **share the base transformer** via `AutoModelForCausalLMWithValueHead`. One forward pass returns both logits and a scalar value head output per token. That halves the FLOPs compared to two separate models, at the cost of the value estimate being coupled to the policy's gradient flow.

KL is applied **on the reward, not in the loss** — `non_score_reward = −β · (logprobs − ref_logprobs)` is added to the per-token reward before GAE, so it becomes part of `mb_advantage`. Two KL metrics are logged: K1 (`logprobs − ref_logprobs`, biased but cheap, used for reward shaping) and K2 (`0.5·Δlogp²`, the Schulman estimator used in clipfrac diagnostics). GRPO switched to K3 (see [[trl-grpo]] and ch-40 §4). The demotion of this PPO trainer into `experimental/` tracks the field's pivot away from critic-based methods.

---

## §6 When TRL is the right choice — decision gate

Pick TRL when **all** of the following hold:

- Run fits on ≤ 1 node (≤ 8 GPUs, or 2 nodes for < 70B).
- Your algorithm is expressible as a `loss_type` in an existing trainer, or is a direct subclass.
- Your rollout engine is `.generate()` or vLLM; you do not need SGLang or a custom inference stack.
- You value HF-ecosystem integration (chat templates, PEFT, Hub) over distributed-systems control.
- You want to ship a checkpoint to `transformers` inference without conversion.

Outgrow TRL when **any** of the following is true:

- **Multi-node rollouts.** TRL's vLLM colocate/server modes do not elegantly scale past a handful of ranks. verl's `RolloutWorker` pool is the canonical next step.
- **Async actor-rollout decoupling.** TRL is synchronous. If you need the actor to train on batch N while the rollout workers produce batch N+1, you need Ray (OpenRLHF / verl).
- **Custom advantage or rollout orchestration.** GRPO's `_compute_loss` switches on `loss_type`; extending it means patching a 200-line conditional. verl's registry (`compute_advantage` / `compute_policy_loss` hooks) is cheaper to extend.
- **Straggler-sensitive workloads** (long completions, variable reward-model latency). Accelerate cannot reschedule.
- **> 70B full-FT RL** — weight-sync cost per step dominates. You want a framework that treats weight transfer as a first-class scheduled operation.

The decision gate is not "TRL is worse than verl." It is "TRL is optimized for a different point on the productivity-vs-scale Pareto curve." Below one node, TRL wins on every axis that matters. Above two nodes, it progressively loses. Know which side of the line you are on.

---

## Companion visualization

**[figures/trl-stack.html](figures/trl-stack.html)** — interactive TRL stack diagram. Five clickable layers (`datasets` → `tokenizer` → `SFTTrainer/DPOTrainer/GRPOTrainer` → `Accelerate` → `torch`). Clicking a layer shows (a) the relevant API surface (class / config / method), and (b) the scaling pain point that hits at that layer first. The top row is frictionless (`datasets` streams, `tokenizer.apply_chat_template` is one call); the middle (`Trainer` subclass) is where all three trainers coexist; the bottom (`Accelerate`) is where scale limits bite.

---

## Further reading

- [[trl-ppo]] — classic actor-critic PPO, value-head co-location, K1/K2 KL; the demoted reference.
- [[trl-grpo]] — `_compute_loss` monolithic body; `loss_type` switch and K3 KL inline.
- [[trl-online-dpo]] — two-sample-per-prompt training_step; Judge plug-in; sigmoid vs IPO branch.
- [[hf-alignment-handbook]] — Zephyr-7B reference recipe; FSDP + ZeRO configs; eval protocol.
- [[hf-dpo-zoo]] — the `loss_type` parameter zoo (sigmoid / ipo / kto / simpo / orpo / bco / cpo).
- [[hf-rlhf-illustrated]] — the three-stage diagram TRL was built to implement.
- [[grpo]] — the Eq. 3 that `GRPOTrainer` implements verbatim.
- [[dpo]] — Eq. 7 closed-form loss; `DPOTrainer` default branch.
- [[ppo]] — PPO-clip objective the experimental trainer fuses into one file.

## Connections

- **ch-55 (verl)** — the Ray-based counterfactual; split-worker architecture TRL refuses.
- **ch-56 (OpenRLHF)** — the middle-ground Ray setup; shows what TRL would look like if it accepted Ray.
- **ch-58 (framework comparison)** — the feature matrix + decision tree this chapter's §6 feeds into.
- **ch-37 to ch-41** — the RL algorithm chapters whose algebra TRL implements.
- **ch-44+ (DeepSeek-R1 recipe)** — what a GRPO production run looks like at frontier scale; usually not on TRL.
