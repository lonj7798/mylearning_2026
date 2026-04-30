<!-- chapter: ch-58
     track: infra
     kind: content
     title: Framework Comparison and When to Pick Which
     deps: [ch-57]
     sources: [[verl-ppo-loss]], [[verl-grpo]], [[verl-rollout]], [[openrlhf-ppo]], [[openrlhf-dpo]], [[trl-ppo]], [[trl-grpo]], [[trl-online-dpo]], [[entropy-logging-patterns]], [[openrlhf-entropy-debugging]], [[async-rollout]]
     figures: figures/framework-compare.html
-->

# Chapter 58 — Framework Comparison and When to Pick Which

> **Core insight.** verl, OpenRLHF, and TRL are not three "PPO libraries" — they are three *architectural bets* on the same algebra. TRL bets on a single Python process, Accelerate, and pedagogical clarity. OpenRLHF bets on Ray actors, an `nn.Module` loss, and an `AdaptiveKLController`. verl bets on registries (`@register_policy_loss`, `@register_adv_est`), an async vLLM engine with per-request priority, and SPMD HybridFlow. The PPO surrogate is the same in all three; the *deployment envelope* that each can survive is not. Pick the framework whose bet matches the run you're launching — not the one whose README you read last.
>
> **Guideline.** Use the feature matrix (§2) and the decision tree (§5) as a pair. The matrix answers "can this framework do X at all?"; the tree answers "given my scale + algo + engineering budget, which framework is the cheapest bet?". Match scale to backend: single-node experiments and online DPO live in TRL; production Ray-based RLHF with `AdaptiveKLController` and clean async semantics lives in OpenRLHF; 128-GPU+ HybridFlow runs with partial rollout and MoE routing live in verl. The graduation path (§6) — TRL → OpenRLHF → verl — is the path a team actually travels as its runs outgrow one bet after another.

---

## §1 What the three frameworks agree on

Before contrasting, make the invariants explicit. On a 2026 reading of all three codebases:

- **Same PPO-clip algebra.** verl `compute_policy_loss_vanilla` ([[verl-ppo-loss]] L1080–1140), OpenRLHF `PolicyLoss.forward` ([[openrlhf-ppo]] L68–168), and TRL's inline PPO ([[trl-ppo]] L820–870) all minimize `−min(r·A, clip(r, 1−ε_low, 1+ε_high)·A)`. The differences are ornamentation.
- **Same GRPO advantage.** verl's `compute_grpo_outcome_advantage` ([[verl-grpo]] L290–335) and TRL's upstream advantage builder both z-score outcome reward within a prompt group; OpenRLHF computes the group baseline in the experience buffer pre-processing. Algebraically identical.
- **Three canonical KL estimators** (K1 / K2 / K3 per [[entropy-logging-patterns]]): K1 = `Δlogp` (biased, verl reward shaping, TRL `objective/kl`), K2 = `0.5·Δlogp²` (Schulman, TRL `approxkl`), K3 = `exp(−Δlogp)+Δlogp−1` (unbiased, verl `kl_penalty="k3"`, TRL GRPO loss term).
- **Entropy is never a loss term by default.** All three log it; none regularize with an entropy bonus in the default config ([[openrlhf-entropy-debugging]] defaults).
- **vLLM for rollouts.** All three integrate vLLM. verl runs an *async* vLLM engine ([[verl-rollout]]); OpenRLHF's async mode uses Ray + `vllm_lock` ([[async-rollout]]); TRL offers `_generate_vllm_server` and `_generate_vllm_colocate` in-process.

What makes them differ is not the loss — it is where KL enters, how rollouts are scheduled, how parallelism is expressed, and which algorithm zoo ships in-tree.

---

## §2 The feature matrix

A 16-row attested matrix. Every cell is a fact from the raw-data sources; "n/a" means the feature genuinely does not exist in that framework as of `main` fetched 2026-04-21.

| # | Dimension | verl | OpenRLHF | TRL |
|---|---|---|---|---|
| 1 | **Rollout backend** | async vLLM `AsyncLLMEngine` ([[verl-rollout]] `vllm_async_server.py` L440–525); SPMD sync retired PR #4411 | Ray actor wrapping vLLM; `vllm_lock` gates weight broadcast ([[async-rollout]]) | HF `.generate()` default; `_generate_vllm_server` + `_generate_vllm_colocate` ([[trl-online-dpo]] L585–893) |
| 2 | **Rollout orchestration** | `DataProto` + worker groups; tokens-in-tokens-out; per-request `priority` | Ray `rollout_queue` + `rollout_slots`; partial-rollout via `strategy.args.train.partial_rollout_enable` ([[async-rollout]]) | Single Python process; Accelerate `gather()` for DDP/FSDP coordination |
| 3 | **Parallelism** | FSDP + Megatron-LM tensor/pipeline parallel via HybridFlow dataflow ([[verl-rollout]]); multi-node native | DeepSpeed ZeRO-1/2/3 + Ray distribution; actor/critic split across GPUs ([[openrlhf-dpo]]) | Accelerate + DeepSpeed/FSDP; single worker-group pattern; no Megatron integration in mainline |
| 4 | **PPO (actor-critic)** | `@register_policy_loss("vanilla")` + value loss; asymmetric `clip_ratio_low/high`, `clip_ratio_c=3.0` dual-clip ([[verl-ppo-loss]]) | `PolicyLoss` module with dual_clip + asymmetric clip + three IS modes ([[openrlhf-ppo]]) | `trl/experimental/ppo/ppo_trainer.py`; symmetric `cliprange` only; value clipping; now experimental ([[trl-ppo]]) |
| 5 | **GRPO / Dr.GRPO** | `@register_adv_est(GRPO)` + `norm_adv_by_std_in_grpo` toggle (False = Dr.GRPO) ([[verl-grpo]]); Pass@k variant | Group baseline computed in experience-buffer pre-processing; reuses `PolicyLoss` | `GRPOTrainer._compute_loss` with `loss_type ∈ {grpo, dr_grpo, bnpo, dapo, cispo, sapo, luspo, vespo}` ([[trl-grpo]]) |
| 6 | **DPO (offline)** | Not the focus; preference loss plugins exist but the track is RL | `DPOLoss` + `concatenated_forward` + MoE aux + NLL-mix + label smoothing + IPO ([[openrlhf-dpo]] L231–257) | `DPOTrainer` with `loss_type ∈ {sigmoid, hinge, ipo, kto_pair, ...}` |
| 7 | **Online DPO / judge-driven** | n/a in mainline | n/a in mainline | `trl/experimental/online_dpo/online_dpo_trainer.py` + sibling `nash_md`, `xpo`, `self_distillation` ([[trl-online-dpo]]) |
| 8 | **KL-to-ref location** | Reward shaping (subtracted per-token before GAE); K1/K2/K3 switch in `kl_penalty()` | Reward shaping via `AdaptiveKLController` / `FixedKLController` around `ppo_trainer.py` L172 | PPO: reward shaping as `non_score_reward` + adaptive controller. GRPO: `β·per_token_kl` added *to loss* with K3 ([[entropy-logging-patterns]]) |
| 9 | **Entropy logging** | `actor/entropy` via `verl_F.entropy_from_logits` (true H); optional registry entropy loss | Per-step mean `−logp` (biased proxy) | PPO: two fields (`objective/entropy` biased, `policy/entropy_avg` true H). GRPO: `_metrics[mode]["entropy"]` (true H) |
| 10 | **vLLM-vs-train IS correction** | `rollout_is_weights` passed to `compute_policy_loss_vanilla`; per-token multiplier | `enable_vllm_is_correction` with three modes: `tis` (truncated IS), `seq-mask-tis`, `icepop` ([[openrlhf-ppo]] L60–76); exposes `vllm_kl` metric | `vllm_importance_sampling_correction` in `GRPOTrainer`; `importance_sampling_ratio` multiplied into per-token loss ([[trl-grpo]]) |
| 11 | **DAPO-style asymmetric clip** | Native: `clip_ratio_low`, `clip_ratio_high`, `delta` upper cap | Native: `clip_eps_low`, `clip_eps_high` | GRPO only: `epsilon_low`, `epsilon_high`, `delta` cap |
| 12 | **GSPO (sequence-level ratio)** | Via loss registry (custom) | Native one-line branch: `policy_loss_type="gspo"` ([[openrlhf-ppo]] L43–48) | `importance_sampling_level="sequence"` in `GRPOTrainer` |
| 13 | **Distributed checkpointing (DCP)** | FSDP DCP + Megatron checkpoints; sharded-save mandatory at scale | DeepSpeed ZeRO checkpoints; CPU-offload for ref model | Accelerate / safetensors sharded save; no native Megatron path |
| 14 | **Async rollout / partial rollout** | Yes: `priority` reorders in-flight, pause-state during broadcast (~L628) | Yes: `ppo_trainer_async.py` + partial-rollout flag; 1.6–2.0× throughput ([[async-rollout]]) | No mainline async trainer; "forthcoming" per [[async-rollout]] |
| 15 | **Multi-node / 128-GPU+** | Primary target; HybridFlow dataflow scales sync→async→partial-rollout with no code change | Supported via Ray cluster; production runs at 70B reported in Hu 2024 | Practical up to 8×H100; 70B feasible with Accelerate+FSDP but no partial-rollout knob |
| 16 | **MoE support** | Megatron MoE routing + async-aware weight broadcast; captures expert selection | DPO: MoE aux loss preserved ([[openrlhf-dpo]]); PPO: supported via DeepSpeed-MoE | Basic (HF `transformers` MoE forward); no specialized router-capture path |
| 17 | **VLM / multimodal** | `multi_modal_data` plumbed through async server (`image_data`, `video_data`) ([[verl-rollout]] L73) | Experimental extensions; not mainline | Mainline `VisionDPOTrainer` / `VisionGRPOTrainer` (transformers integration) |
| 18 | **Maintenance velocity (2026-04)** | Active (volcengine/ByteDance Seed production); monthly tagged releases | Active; monthly releases; Hu et al. community | Most active of the three on HF side; weekly merges; GRPO zoo expands fastest |

---

## §3 Logging-pattern crib sheet

When the dashboard breaks, you need to know *exactly* what each framework names the three signals. From [[entropy-logging-patterns]]:

| Signal | verl field | OpenRLHF field | TRL PPO field | TRL GRPO field |
|---|---|---|---|---|
| **KL(π‖π_old)** | `actor/ppo_kl` (K1, logged) | `ppo_kl` from `PolicyLoss.forward` | `objective/kl` (K1) + `policy/approxkl_avg` (K2) | implicit via K3 in loss |
| **KL(π‖π_ref)** | shaped into reward; monitored via `kl_penalty` | `kl_ctl.value · kl_t` per-token; `status["kl"]` | `objective/kl` (for PPO); GRPO: `β·per_token_kl` loss term |
| **Rollout-vs-train KL** | via `rollout_is_weights` per-token ratio | `vllm_kl` returned from `PolicyLoss.forward` | n/a in PPO; GRPO: `importance_sampling_ratio` |
| **Entropy** | `actor/entropy` (true H from logits) | mean `−logp` per step (biased) | `objective/entropy` (biased) + `policy/entropy_avg` (true H) | `_metrics[mode]["entropy"]` (true H) |
| **Clip fraction** | `actor/pg_clipfrac` + `actor/pg_clipfrac_lower` (dual-clip hits) | `clip_ratio` returned from `PolicyLoss.forward` | `policy/clipfrac_avg` | computed in-loop, no single canonical key |

**Collapse signature is identical across the three** ([[openrlhf-entropy-debugging]]): entropy drops ≥30% in <100 steps, PPO-KL spikes ≥0.1, clipfrac pegs to 1. Use whichever framework-specific key maps to these three.

---

## §4 Performance envelope

Attested throughput from [[async-rollout]] and the verl blog:

- **Sync (TRL default).** Rollout idle during optimizer step; optimizer idle during rollout. For 7B on 8×H100, rollout dominates ≥70% wall time. No straggler handling — slowest rollout in the batch gates the step.
- **OpenRLHF async.** `rollout_queue` decouples the two; staleness bounded at `k = queue_depth + partial_rollout_depth`, typically 1–2. Hu 2024 Figure 5: 1.9× throughput at 7B, 1.6× at 70B. `vllm_lock` serializes weight broadcasts; partial-rollout mode interrupts in-flight generation when weights update.
- **verl async HybridFlow.** vLLM's internal scheduler + per-request `priority` + pause-state. The "continuous batching RL" pattern: newly-weighted requests jump ahead of stragglers; in-flight generations finish under old weights then resume. Claimed to saturate GPUs more completely than OpenRLHF's queue-based model because the scheduling is at vLLM's granularity, not Ray's.

**Straggler handling** is where the three diverge most. TRL has none — slowest rollout in batch gates the step. OpenRLHF's queue isolates the trainer from rollout variance but does not reorder within a rollout batch. verl's `priority` reorders *within* the vLLM engine, letting short completions finish first and long completions continue under stale weights (IS-correction handles the bias).

**vLLM-vs-train logprob mismatch** is the hidden tax. bf16 vLLM inference vs fp32 actor forward produces systematic ratio bias; without IS correction, PPO destabilizes within ~50 steps on long completions ([[openrlhf-ppo]]). verl + OpenRLHF both ship IS correction; TRL's GRPO path has `vllm_importance_sampling_correction`, but TRL PPO does not.

---

## §4.5 Three concrete run profiles

To make §4's abstractions concrete, three attested envelopes — each a different framework lands at a different leaf of §5:

**Profile A — "Qwen-2.5-3B RLVR on 1×A100, 5K math prompts."** 3B base, G=8 rollouts, max_completion_length=1024, β_KL=0.05, Dr.GRPO aggregation. Rollout is ~60% wall-time but you can't meaningfully async on a single GPU. Pick TRL: `GRPOTrainer(loss_type="dr_grpo", vllm_importance_sampling_correction=True)`. No Ray, no DeepSpeed dance, `_compute_loss` fits on a single screen for debugging. This is the [[openrlhf-entropy-debugging]] "resource-constrained path" shape.

**Profile B — "Llama-3-70B PPO on 32×H100, 3-day run, ultrafeedback prompts."** Rollout is ~80% wall-time; sync wastes ≥1 day of GPU-hours. Pick OpenRLHF: `PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.28, dual_clip=3.0, enable_vllm_is_correction=True, vllm_is_correction_type="tis")` with `AdaptiveKLController(target=10.0)`. Ray-managed actor/critic split; queue-level async recovers the ~1.6× throughput attested in [[async-rollout]] Figure 5. `vllm_kl` in the dashboard catches the bf16/fp32 mismatch before PPO destabilizes.

**Profile C — "Mixtral-8x22B GRPO on 128×H100, 2-week run, multi-modal agent rollouts."** Straggler variance dominates; MoE routing + VLM `image_data` in-flight; partial-rollout is load-bearing. Pick verl: `@register_policy_loss("vanilla")` with `clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=3.0`, `@register_adv_est(GRPO)` with `norm_adv_by_std_in_grpo=False` (Dr.GRPO mode). Async vLLM with `priority` reordering, pause-state during weight broadcast, Megatron TP=8/PP=2 for the 8x22B. The same run does *not* fit in OpenRLHF's queue-level scheduling — slowest sample gates the step.

All three profiles share the same PPO algebra; the framework choice is driven entirely by §4's performance envelope + §2's feature matrix, not by the loss.

---

## §5 Decision tree — which framework for which run

Read top-to-bottom; stop at the first match. Every leaf is a concrete recommendation with a one-line justification.

```
Q1. Is this a <8-GPU experimental run, pedagogical exercise, or offline DPO?
    YES → TRL.
         Rationale: single Python process, `Accelerate` handles the parallelism,
         `DPOTrainer` + `GRPOTrainer` cover 90% of single-node runs, monolithic
         `_compute_loss` is the clearest code to read when debugging.
    NO  → Q2.

Q2. Do you need ONLINE DPO, self-rewarding LM, SPIN, Nash-MD, or XPO?
    YES → TRL (mandatory).
         Rationale: `trl/experimental/online_dpo/` + sibling `nash_md`, `xpo`,
         `self_distillation` -- no equivalent exists in OpenRLHF or verl
         mainline ([[trl-online-dpo]]).
    NO  → Q3.

Q3. Is the run 8–64 GPUs, standard PPO/GRPO/DPO, with ByteDance / Qwen-style
    asymmetric clip and partial-rollout async?
    Q3a. Do you primarily want a clean Ray-based async pattern with
         AdaptiveKLController and proven 7B–70B track record?
         YES → OpenRLHF.
              Rationale: `PolicyLoss` nn.Module + `AdaptiveKLController` +
              `rollout_queue` is the smallest production-grade surface;
              1.9x/1.6x throughput attested; IS-correction (tis/icepop/
              seq-mask-tis) ships in-tree ([[openrlhf-ppo]], [[async-rollout]]).
         NO  → Q3b.
    Q3b. Do you need MoE with router capture, VLM multi-modal rollouts, GRPO
         Pass@k, or Megatron tensor/pipeline parallel?
         YES → verl.
              Rationale: `vllm_async_server.py` plumbs `multi_modal_data`;
              registry-pluggable adv estimators include GRPO Pass@k; Megatron
              integration is first-class ([[verl-rollout]], [[verl-grpo]]).

Q4. Is the run 128-GPU+, multi-node, partial-rollout with per-request priority,
    and will it run for weeks?
    YES → verl.
         Rationale: HybridFlow dataflow graph is the only one of the three
         designed for "same code, sync→async→partial-rollout"; per-request
         priority + pause-state is the only in-vLLM scheduler; production
         references at ByteDance Seed ([[async-rollout]]).
    NO  → revisit Q3.

Q5. Do you need a feature TRL has and the others do not (CISPO, SAPO, LUSPO,
    entropy-quantile masking, online judge-driven DPO, KTO, SimPO, ORPO)?
    YES → TRL, even at scale.
         Rationale: the GRPO zoo (`loss_type` switch covers 8 variants) and
         the DPO zoo (`loss_type` covers 6+ variants) are TRL-exclusive; the
         scaling cost is real but the algorithm cost of re-implementing in
         verl/OpenRLHF is larger for one-off research runs ([[trl-grpo]]).

Q6. Is debugging / understanding the algorithm the primary goal this quarter?
    YES → TRL (read), then port to OpenRLHF or verl.
         Rationale: one-file implementations; no Ray / dataflow indirection.
```

---

## §6 Graduation criteria — TRL → OpenRLHF → verl

A team doesn't "pick one" — a team *graduates* from one to the next as the run outgrows the previous bet. The criteria are not opinions; they're concrete signals from the codebases.

**Graduate from TRL to OpenRLHF when:**
- You need `AdaptiveKLController` (TRL has one for PPO but it's minimal; OpenRLHF's is the InstructGPT-faithful implementation).
- You hit the vLLM-IS-correction wall: PPO destabilizes within ~50 steps on long completions, and you need `tis` / `icepop` / `seq-mask-tis` ([[openrlhf-ppo]] L60–76).
- Rollout becomes ≥70% of wall time and you need async to recover it. TRL's in-process vLLM cannot overlap rollout with optimizer step; OpenRLHF's Ray-based async can.
- You need MoE aux-loss preservation during DPO on Mixtral/DeepSeek-MoE. OpenRLHF's `DPOTrainer` handles this natively; TRL's does not in mainline.
- You need a clean actor/critic split across GPUs (Ray-distributed) instead of single-process FSDP.

**Graduate from OpenRLHF to verl when:**
- You hit straggler-dominated rollouts where the slowest sample in the batch gates the step, and you need per-request `priority` to reorder within the engine. OpenRLHF's queue-level isolation is not enough; you need vLLM-scheduler-level reordering ([[verl-rollout]]).
- You move beyond 64 GPUs and need HybridFlow dataflow: same code running sync on 8 GPUs, async on 32, partial-rollout on 128.
- You need Megatron tensor/pipeline parallelism (70B+ with activation recomputation pathologies).
- You add VLM modalities (image/video tokens) that require the rollout engine to accept `multi_modal_data`.
- You want plug-in algorithm research: writing a new advantage estimator or policy loss as a 30-line `@register_*` function without modifying the trainer is a verl idiom.
- Debugging KL-as-reward-shaping at token granularity: verl's `kl_penalty()` with `k1/k2/k3` switch is the cleanest instrumentation surface.

**When *not* to graduate:**
- Don't graduate to OpenRLHF just for Ray. Ray adds debugging difficulty; only adopt it when you need async + distributed actor/critic simultaneously.
- Don't graduate to verl "because it's the newest". The registry pattern is powerful but adds indirection; if you will never write a custom `@register_policy_loss`, OpenRLHF's `PolicyLoss` module is simpler.
- Don't graduate forward then regress: once a run is verl-scale, reproducing it in TRL requires either Megatron integration work or accepting lower throughput — plan the migration as one-way.

---

## §6.5 Anti-patterns observed in 2025–2026 runs

Common failure modes where a team picked the wrong framework and paid for it:

- **TRL at 128 GPUs.** Attempting to scale `PPOTrainer` or `GRPOTrainer` past ~8 H100s by layering more DeepSpeed configuration. The failure mode: rollout stays ≥70% of wall time, no partial-rollout knob exists, and the fix ("add async") requires rewriting the training loop. The attested signal: `rollout_time / total_time > 0.7` sustained across 500+ steps. Graduate per §6.
- **OpenRLHF for online DPO.** Building a "judge callback" on top of `DPOTrainer` to sample fresh pairs. The failure mode: the pair harness, EOS penalty, KL-canary, and judge-interface plumbing all have to be reinvented; [[trl-online-dpo]] already ships them. The attested signal: the first `self_rewarding` commit in a non-TRL repo invariably copies TRL's `_calculate_rewards_from_functions`.
- **verl for a pedagogical one-file read.** Pulling the ByteDance HybridFlow stack to understand GRPO, when `trl/trainer/grpo_trainer.py` is a single-file `_compute_loss` that covers every variant. The failure mode: three days of Ray + Megatron debugging before the learner can execute one backward pass. §5 Q6 routes these reads to TRL for a reason.
- **Mixing KL estimators across frameworks.** Running a verl sweep with K1, porting to TRL GRPO (K3 in loss), and comparing KL numbers directly. [[entropy-logging-patterns]] flags this: K1 is biased and can be negative; K3 is unbiased and ≥0. The curves aren't comparable without renormalization.
- **Disabling `vllm_importance_sampling_correction` "to match a TRL PPO baseline".** TRL PPO doesn't have the correction; OpenRLHF and verl do. Turning it off on the production framework to match the reference is the wrong direction — PPO will destabilize within ~50 steps on long completions. Upgrade the baseline instead.

Each anti-pattern has the same root cause: treating the framework as a commodity substitution instead of as an architectural bet. The matrix in §2 plus the tree in §5 are jointly designed to make the substitution cost explicit.

---

## §7 The companion interactive

See [framework-compare.html](figures/framework-compare.html) — pick scale, algo, and special requirements (MoE / VLM / online DPO / async); the recommendation updates with the justification citing the matrix row(s) behind it. Use it as a checklist: every recommendation lands on a §2 cell you can cross-reference.

---

## §8 What this chapter is not

This is not a benchmark chapter — no framework has a single "throughput number" independent of algo / model size / rollout length / straggler distribution. Any reproducible benchmark requires specifying all four, and the numbers from Hu 2024 ([[async-rollout]] Figure 5) are the only attested ones in the raw-data library. This is not a recommendation chapter either — the recommendation is always "the framework whose architectural bet matches this run"; §5's tree produces that match, but a team that knows its constraints can route itself without the tree.

The synthesis to carry forward: **the algorithm is in the paper; the framework is the deployment envelope**. Chapter 59 (the capstone) will ask you to reproduce an open recipe end-to-end — this chapter is the decision input for that capstone's "which framework" question.

---

## Connections

- **ch-55 (verl internals)** — the framework-specific deep read for `[[verl-ppo-loss]]`, `[[verl-grpo]]`, `[[verl-rollout]]`.
- **ch-56 (OpenRLHF internals)** — deep read for `[[openrlhf-ppo]]`, `[[openrlhf-dpo]]`, `[[async-rollout]]`.
- **ch-57 (TRL internals)** — deep read for `[[trl-ppo]]`, `[[trl-grpo]]`, `[[trl-online-dpo]]`.
- **ch-43 (entropy/KL control)** — the theory behind the §3 logging crib sheet.
- **ch-59 (capstone)** — the "pick one and reproduce" run that uses this chapter's decision tree as its framework-selection input.
- [[entropy-logging-patterns]], [[openrlhf-entropy-debugging]] — the cross-framework tables this chapter cites for §3.
- [[async-rollout]] — the throughput + staleness-bound evidence for §4 and §5's scale thresholds.
