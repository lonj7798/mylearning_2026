<!-- chapter: ch-01
     track: rl-pipeline
     kind: content
     title: Why RL-from-Base Needs Its Own Infrastructure
     deps: []
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[slime]], [[verl-rollout]], [[olmo-3-rl-pipeline]], [[scalerl]], [[areal-async-rl]], [[open-instruct-rlvr-pipeline]], [[pipelinerl]]
     figures: none
-->

# Chapter 1 — Why RL-from-Base Needs Its Own Infrastructure

> **Core insight.** The infrastructure an RL run needs is fixed by the training recipe it serves. A model distilled from a stronger teacher begins RL near its ceiling, so RL is a short refinement: a failed run can be restarted, and small sampler–learner numerical differences "have little time to compound" ([[mai-rl-infrastructure]] p. 1). A climb from a base model through RL alone runs for "many thousands of training steps spanning weeks to months on thousands of GPUs" (p. 1) and removes both escapes. Three properties then stop being edge cases and become design constraints: policy staleness (agentic rollouts last hours while GRPO waits for the whole group), numerical consistency between a lower-precision sampler and the learner, and recovery and resizing without restarting the job. Microsoft AI's answer, Rocket, owns the three tightly coupled systems — learner, RL loop, and an inference engine on an MAI-maintained fork of SGLang — and reuses commodity layers: Kubernetes or Slurm for node allocation and Ray actors for components (p. 2).
>
> **Guideline.** When RL is a short phase after distillation (hundreds of steps, restartable), adopt an open framework and its default placement, because the open recipes that report their RL length run 60–2,000 steps ([[areal-async-rl]] Table 1; [[olmo-3-rl-pipeline]] Table 49) and the failure modes in this course have little time to accumulate. When RL is the main capability climb and lasts weeks, plan for staleness control, sampler–learner parity, and component-level recovery from the first day, because Rocket's Table 1 estimates that targeted recovery and runtime reconfiguration saved 432.5 of 2,490 training hours (17.37%) in a portion of one run ([[mai-rl-infrastructure]] p. 11), and because MAI-Thinking-1 reports that resuming from a pre-collapse checkpoint "is often not viable" ([[mai-thinking-1]] Sec. 3.1.4), so restart is not a cheap fallback. When you adopt an open framework for a long run, keep the trainer and the inference engines in separate allocations, because slime's own placement code separates them so that "a restarted trainer [can] resize without moving the serving engines" (slime@79aa41a `slime/ray/placement_group.py` L127–128).

---

## Why this chapter matters

Every later chapter of this course describes a mechanism that exists because of one of the three challenges named here. Staleness gets ch-03 to ch-05, numerics get ch-06 and ch-07, agentic rollouts get ch-08, and operations get ch-20 and ch-21. If you do not know which recipe regime you are in, you cannot decide which of those mechanisms you need. Building all of them for a 300-step refinement run wastes engineering time. Skipping them for a multi-week climb wastes GPU time. In Table 1 the waste is counted in millions of GPU-hours.

The chapter also sets out the build-versus-adopt question in the form Rocket states it. The open frameworks are not described as wrong. They are described as being exercised in a different regime. You will meet that regime boundary again whenever this course compares Rocket with slime, verl, AReaL, or open-instruct.

---

## §1 Two recipe regimes

### 1.1 What the Rocket post states

The argument is on pages 1–2 of [[mai-rl-infrastructure]]. In order:

1. **Distilled start.** "A model distilled from a stronger teacher begins RL close to its ceiling, so RL becomes a comparatively short refinement phase." In that regime "a failed run can simply be restarted, and small mismatches between sampler and learner have little time to compound" (p. 1).
2. **Climb from base.** Rocket "is built for the more demanding case of climbing from base capabilities through RL itself, without a distilled head start." In this regime "hardware failures are a certainty, small numerical mismatches accumulate over thousands of updates, and the best data mixture and GPU allocation shift as the policy improves" (p. 2).
3. **The workload.** MAI Base 2, a "three-trillion-parameter base model", is being post-trained with Rocket, "climbing from the base model through RL without distillation from third-party models". The post states that this work "is separate from the MAI Code 2 models" (p. 1).

The predecessor run used the same regime. MAI-Thinking-1 (35B active / 1T total parameters) started RL from a checkpoint "with no prior exposure to reasoning traces" ([[mai-thinking-1]] Sec. 3, p. 30). Its Sec. 3.6 introduced Rocket as the framework behind that climb.

### 1.2 Why the regime changes the infrastructure, step by step

The post gives two escapes that a distilled start provides. A climb from base removes both.

- **Escape 1: restart.** If RL lasts a few days, a crashed or diverged run costs a few days. In a climb that lasts weeks, a restart costs the time since the last good checkpoint, plus cluster requeue and container start-up. There is also a second cost. MAI-Thinking-1 reports that resuming from a checkpoint saved before a collapse "is often not viable" because instabilities are embedded "many steps before the actual collapse" ([[mai-thinking-1]] Sec. 3.1.4, p. 34–36). Its remedy was self-distillation to restart the climb, which is a recipe step and not an infrastructure step. So the infrastructure has to avoid losing progress in the first place. That is the job of the recovery and elasticity machinery in ch-20 and ch-21.
- **Escape 2: little time to compound.** A small per-token log-probability difference between sampler and learner changes the gradient a little at every step. Over a few hundred steps the accumulated effect may stay small. Over thousands of steps it can grow. The post also reports that the gap itself grew: "In a controlled comparison with identical inputs, we observed a larger sampler–learner log-probability discrepancy at a later RL checkpoint than at initialization", which "suggests that optimization may amplify or exploit numerical differences" (p. 9). The authors wrote "suggests". This is one observation, not a measured mechanism.

### 1.3 How long open recipes run

The post says open frameworks are "typically exercised" in a different regime (p. 2). It gives no step counts for that regime. The open reports in this library give the following numbers, collected here only for context:

| Run | RL length | Source |
|---|---|---|
| AReaL 1.5B / 7B math, 14B / 32B code | 250 / 250 / 80 / 60 PPO steps | [[areal-async-rl]] Table 1 |
| Olmo 3 7B Think / 32B Think / 7B Instruct / 7B RL-Zero | 1,400 / 750 / 450 / 2,000 steps | [[olmo-3-rl-pipeline]] Table 49 |
| Olmo 3 7B Think wall clock | ≈15 days first run, 6 days on newer infra | [[olmo-3-rl-pipeline]] §4.4.2 |
| PipelineRL Qwen2.5-7B math | 1,000 optimizer steps | [[pipelinerl]] §5 |
| ScaleRL 8B | 100K GPU-hours, about 7,400 steps | [[scalerl]] Fig. 1 |
| Rocket, MAI model (Fig. 8, "a portion") | x-axis 40–2,000 steps, log-linear fit R² = 0.974 | [[mai-rl-infrastructure]] p. 14 |

Two cautions. First, step counts do not compare directly, because batch sizes and rollout lengths differ. Second, the ScaleRL run shows that open groups also run long climbs, so the boundary is not "open versus closed". What these runs show is that open RL experiments usually report hundreds to a few thousand steps on models of 1.5B–32B dense or 17B×16 MoE. Rocket targets a 3T-parameter model with hour-long agentic rollouts.

---

## §2 Challenge 1 — staleness

The post states the problem in one sentence: "Agentic rollouts can take hours while the learner keeps updating the policy. Because GRPO scores each rollout against the others for the same problem, a single slow or failed rollout holds back its entire group, so training drifts further off-policy" (p. 1).

Two facts combine here.

1. **Rollout durations spread over two orders of magnitude.** Fig. 2 reports medians over 1.13M rollouts from one day of a production run: instruction following 2 min / 1 turn; visual Q&A 2 min / 1 turn; math & science 7 min / 1 turn; tool use 12 min / 13 turns; competitive coding 34 min / 1 turn; web research 51 min / 89 turns; web development 58 min / 104 turns; software engineering 93 min / 249 turns (p. 4). These are medians, so the tails are longer. The single GRPO group traced in Fig. 4 has a slowest rollout with 784 turns (p. 6).
2. **GRPO needs the whole group.** The advantage of each rollout is computed from the rewards of all rollouts of the same problem (llm-training ch-40). A completed rollout therefore cannot be trained until its slowest sibling finishes. In Fig. 4 the fastest rollout waits 7.6 h (p. 6).

In a synchronous loop the cost of this would be idle GPUs. In an asynchronous loop GPUs stay busy, but the data becomes stale. Rocket chose asynchrony "by design" (p. 5), so it has to measure staleness and bound it. ch-03 defines the quantity and ch-04 and ch-05 describe the levers.

---

## §3 Challenge 2 — numerical consistency

The post: "Fitting a larger model and longer contexts into the sampler requires lower-precision storage, and differences in precision, kernels, and parallelism between the sampler and learner can slow learning and destabilize training" (p. 1).

The two Rocket generations made different choices here:

| | MAI-Thinking-1 (Rocket v1) | MAI Base 2 (Rocket, Nov 2026) |
|---|---|---|
| Sampler precision | "bf16 for both the learner and the inference engine" ([[mai-thinking-1]] Sec. 3.6.4) | MXFP8 storage for MoE experts and most global-attention KV, BF16 compute (p. 7–9; ch-06) |
| Extra consistency measures | MoE routing replay, top-p mask replay (Sec. 3.6.4, 3.1.3) | bitwise sampler–learner parity for identical checkpoints and inputs (p. 10; ch-07) |

The change of model size forced the change of sampler precision. A 3T-parameter model does not fit the same number of sampler GPUs at BF16. Lower precision adds a new source of mismatch, so Rocket added the parity work.

Open frameworks handle the same gap mainly with an algorithmic correction. open-instruct multiplies the loss by a clamped ratio ρ = π_train_old / π_infer_old with upper bound 2.0 (open-instruct@f020e9a `open_instruct/grpo_utils.py` L125–137, matching Olmo 3 Table 49). slime's production low-precision path is BF16 training with FP8 rollout and an optional TIS weight (slime@79aa41a `docs/en/advanced/low-precision.md` L3–7). ch-07 compares parity with correction.

---

## §4 Challenge 3 — fault tolerance and elasticity

The post: "Runs that span thousands of GPUs for weeks or months will encounter hardware failures, while the best runtime configuration and data mixture change. Restarting the whole job for every failure or reconfiguration wastes GPU time" (p. 1).

Table 1 (p. 11) is the only quantitative statement of this cost. It covers "a portion of RL training, assuming 6,000 GPUs":

| Event type | Saved per event | Time saved (h) | % | GPU-hours saved |
|---|---|---|---|---|
| In-job recovery (n = 21) | 0.5 h | 10.5 | 0.42% | 63,000 |
| Learner recovery (n = 88) | 2 h | 176 | 7.07% | 1,056,000 |
| Runtime elasticity (n = 123) | 2 h | 246 | 9.88% | 1,476,000 |
| Overall training time | — | 2,490 | 100.00% | 14,940,000 |
| Total estimated savings | — | 432.5 | 17.37% | 2,595,000 |

How to read it:

- **Every value is an estimate.** The caption says "Per-event savings are estimates that vary with the workload mix and the point at which a failure or reconfiguration occurs" (p. 11). The per-event saving (0.5 h or 2 h) is an assumed counterfactual, not a measured A/B difference. Each GPU-hour cell is hours × 6,000, and each % cell is hours / 2,490 ([[mai-rl-infrastructure]] Verification).
- **Event counts give a failure rate.** 88 learner recoveries in 2,490 hours is about one every 28 hours. Without learner-only recovery, each of those events would have been a job restart.
- **Elasticity is the largest row.** 123 reconfigurations saved more than recovery did. This supports the post's claim that "the best data mixture and GPU allocation shift as the policy improves" (p. 2). The table measures how often the configuration changed. It does not measure whether each change improved the model.

ch-20 and ch-21 cover the mechanisms. For this chapter, the point is the scale: about 17% of the hours in the portion they report.

---

## §5 Build versus adopt

### 5.1 What Rocket owns and what it reuses

The post lists three systems MAI "develops and operates": "the learner and training framework, Rocket, and an inference engine built on an MAI-maintained fork of SGLang". The reason given is co-design: "Owning all three allows us to co-design the interfaces between them rather than optimize each component in isolation" (p. 2).

It also lists what it does not build: "Rocket leverages Kubernetes or Slurm to allocate nodes to a job, and Rocket's components run on Ray as actors. We adopted SGLang as our inference foundation and have substantially adapted it" (p. 2). The cost is stated as well: "Owning these integrations carries a maintenance cost" (p. 2).

MAI-Thinking-1 gives the original reason for building: the requirement to "support asynchronous RL across several thousand GPUs at the scale MAI-Thinking-1 demanded" was "not served by available open-source RL frameworks", citing TRL, OpenRLHF and verl ([[mai-thinking-1]] Sec. 3.6, p. 49). The largest Rocket v1 job used 4,864 GB300 chips: 4,096 for inference and 768 for the learner (Sec. 3.6.4).

**What is not stated.** Neither source names a specific capability that verl or slime lacked. The post calls them "strong foundations for RL post-training, including distributed execution, customizable agent workflows, and asynchronous sampling", then says long climbs "push these capabilities beyond the regime they are typically exercised in" (p. 2). Read this as a statement about scale and duration of use. It is not a list of missing features.

### 5.2 Why co-design matters in this course

Several later mechanisms span two of the three owned systems at once:

| Mechanism | Systems it spans | Chapter |
|---|---|---|
| Two-stage weight transfer (engines pull from learner, then fan out) | learner + inference engine | ch-05 |
| Learner fake-quantizes what the sampler stores in MXFP8 | learner + inference engine | ch-06 |
| Shared FA4 kernel with aligned tiles in learner forward, sampler prefill, sampler decode | learner + inference engine | ch-07 |
| Request priority inherited from rollout start time | RL loop + inference engine | ch-04 |
| Learner recovery reconnects to the live controller | learner + RL loop | ch-20 |

Each row requires changing code on both sides of an interface. A team that controls only one side can still apply the idea, but must do it through the other side's public API. For bitwise parity this is hard, because parity depends on kernel tiling and reduction order inside both engines.

### 5.3 How the open frameworks draw the same boundaries

The open frameworks own the same two integration points. They differ in how many backends they support behind each one.

**slime** couples Megatron (training) and SGLang (rollout) through one loop, and the same `train.py` runs synchronous and fully asynchronous RL ([[slime]] Core Insight). The weight-update step sits in that loop:

```python
# slime@79aa41a train.py L85–93
offload_train(actor_trains)
if args.offload_rollout and not release_train:
    ray.get(rollout_manager.onload_weights.remote())
was_paused = ray.get(rollout_manager.pause_rollout_admission.remote())
actor_model.update_weights()
# Final evaluation uses the synchronized engines directly. Keep training
# producers paused when no later rollout will consume their new work.
if rollout_id + 1 < args.num_rollout:
    ray.get(rollout_manager.resume_rollout_admission.remote(was_paused))
```

slime's placement code also states the reason for keeping learner and inference in separate allocations. The reason is the same one Rocket gives for learner-only recovery:

```python
# slime@79aa41a slime/ray/placement_group.py L126–138
if not args.colocate and not args.rollout_external and not args.debug_train_only and not args.debug_rollout_only:
    # Separate allocations let a restarted trainer resize without moving
    # the serving engines that still hold rollout state.
    actor_pg = _create_placement_group(args.actor_num_nodes * args.actor_num_gpus_per_node)
    try:
        rollout_pg = _create_placement_group(args.rollout_num_gpus)
    except BaseException:
        from ray.util.placement_group import remove_placement_group

        if actor_pg[0] is not None:
            Cleanup().run("remove partial training placement", remove_placement_group, actor_pg[0])
        raise
    return {"actor": actor_pg, "critic": actor_pg if args.use_critic else None, "rollout": rollout_pg}
```

**verl** puts the boundary in an enum of rollout modes. The comments state the use case of each:

```python
# verl@8357f1e verl/workers/rollout/replica.py L68–81
class RolloutMode(Enum):
    # Rollout engine and training engine(fsdp/megatron) fused in same process
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"
```

verl supports several training backends (Megatron, FSDP, TorchTitan, VeOmni) and several rollout engines (vLLM, SGLang, TRT-LLM) behind these interfaces. Its low-precision doc describes the target as "the Cartesian product of training backends … and rollout engines" (verl@8357f1e `docs/low_precision/qat.md` L54–59). Supporting a product of backends is the opposite of Rocket's choice to own one learner and one inference fork. The trade-off is generality against the ability to change both sides of an interface at once.

**open-instruct** and **AReaL** make the same split. open-instruct runs a DeepSpeed or OLMo-core learner against vLLM actors, coordinated by Ray ([[open-instruct-rlvr-pipeline]] Summary). AReaL runs rollout and training on separate GPU groups with a rollout controller ([[areal-async-rl]] §4.1).

**What every stack shares with Rocket.** All of them, Rocket included, use Ray to place components and an open inference engine (SGLang or vLLM) as the starting point for the sampler. The decision you face is therefore not whether to write a distributed runtime from scratch. It is whether you need to modify the learner and the inference engine together. You need to if you want bitwise parity, quantization that matches on both sides, or a weight transfer co-designed with the engine's sharding.

---

## §6 What a separated control plane buys

Rocket separates supervision from the RL loop: "Separating supervision from the RL loop lets Rocket recover or resize individual components without restarting the whole job" (p. 10–11). Two concrete choices follow from this.

1. **Each component is a Ray actor with its own lifecycle.** The controller, problem workers and rollout workers are each "a single Python process / Ray actor" ([[mai-thinking-1]] Sec. 3.6, p. 50). The learner can fail while the controller, rollout workers and inference fleet keep running (p. 11).
2. **The control plane runs on CPU-only nodes.** "Ray heads and Rocket's CPU actors run on CPU-only nodes, which have been more stable than GPU nodes in our deployments" (p. 12). This is an operational observation. No failure rates are given.

The causal chain: if the controller holds the curriculum state and the list of valid pending work, and it does not share a process or node with the learner, then a learner crash does not destroy that state. Recovery becomes "rebuild learner, restore checkpoint, reconnect" ("roughly five minutes", p. 11) instead of a job restart. Table 1's learner-recovery row (88 events) is the measured frequency of the case this design covers.

---

## §7 Evidence: measured, estimated, or argued

| Claim | Type | Location |
|---|---|---|
| Distilled starts are near their ceiling, so restart is cheap and mismatches have little time to compound | argued, no measurement | [[mai-rl-infrastructure]] p. 1 |
| The sampler–learner gap was larger at a later checkpoint than at initialization | one controlled observation; no numbers | p. 9 |
| Rollout durations: 2 min to 93 min median by type | measured, 1.13M rollouts, one day | p. 4, Fig. 2 |
| Learner recovery takes roughly five minutes | measured ("in our runs"), no distribution | p. 11 |
| 432.5 h / 17.37% saved | estimate with assumed per-event savings | p. 11, Table 1 |
| Bitwise-matched run climbs higher | two runs launched about one month apart; confounded by time | p. 10, Fig. 6 (ch-07) |
| Model "recently passed Claude Opus 4.8" on the aggregate benchmark | outcome; infrastructure is "one of the ingredients" | p. 13–14, Fig. 8 |
| Resuming from a pre-collapse checkpoint is often not viable | observation from MAI-Thinking-1 climbs | [[mai-thinking-1]] Sec. 3.1.4 |

Fig. 8's steady climb (R² = 0.974) is not evidence that any single infrastructure choice was necessary. The post itself calls infrastructure "one of the ingredients in this climb" (p. 14).

---

## Recipe

1. **Classify the run.** Write down the expected number of RL steps, the wall-clock duration, the starting checkpoint (distilled or base), and the longest rollout type. If the run lasts days and starts from a distilled model, a restart is an acceptable fallback.
2. **Estimate the failure budget.** From your cluster's history, estimate failures per day per 1,000 GPUs. Multiply by the run length and by the cost of one restart (time since the last checkpoint plus requeue time). Compare the result with the run length. Table 1's 88 learner failures in 2,490 hours is one reference point.
3. **Measure the rollout-length spread.** Run a small batch of each task type and record median and p95 wall-clock time and turns, as in Fig. 2. If the ratio between the longest and shortest median is large (Rocket's is 93 / 2 ≈ 46), plan for asynchrony and staleness control (ch-02 to ch-04).
4. **Decide the sampler precision.** If the model fits the sampler fleet in BF16 at the target concurrency, keep BF16 on both sides, as MAI-Thinking-1 did. If it does not, plan for low-precision storage and matching learner quantization (ch-06).
5. **Separate allocations from the start.** Put learner and inference in separate placement groups (slime without `--colocate`; verl `STANDALONE`), and run the controller on stable nodes.
6. **Decide how much to own.** List the cross-boundary mechanisms you need (table in §5.2). If you need bitwise parity or co-designed weight transfer, budget for maintaining forks of both the learner and the inference engine. Otherwise adopt an open stack and use its correction mechanisms (TIS, staleness bounds).
7. **Instrument before scaling.** Log staleness per token or per sample, sampler–learner log-prob difference, restarts, and time lost per restart from the first run. Without these numbers you cannot produce your own version of Table 1.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating a multi-week climb like a short refinement | Each node failure costs hours of requeue and recomputation | Count restarts × hours lost per restart; compare with Table 1's per-event estimates |
| Assuming a restart from an earlier checkpoint fixes a collapse | The resumed run collapses again at a similar step | Track the sampler–learner gap and entropy before the collapse; [[mai-thinking-1]] reports instabilities embedded many steps earlier |
| Colocating learner and inference for an agentic run | The learner waits for the longest rollout every step; GPU utilization falls as rollout-length spread grows | Log per-step generation time and training time separately (verl `timing_s/gen`); compare with Fig. 2-style duration spread |
| Reading Table 1 as measured savings | Overconfident cost model | The caption labels per-event savings as estimates; rebuild the table from your own counted events and your own per-event costs |
| Reading Fig. 6 or Fig. 8 as proof that one infrastructure feature caused the improvement | Copying one feature and expecting the same curve | Fig. 6 compares runs launched a month apart; Fig. 8 has no ablation |
| Choosing an open framework's colocated default for long-tail rollouts | Training step time set by the slowest sample | slime's docs recommend separate pools plus fully async for widely varying lengths ([[slime]] Guideline) |

---

## Check your understanding

1. The post argues that a distilled start makes small sampler–learner mismatches harmless because they "have little time to compound". State the mechanism by which a per-token mismatch could compound over training steps, and explain why the post's later observation (a larger gap at a later checkpoint on identical inputs) is evidence for a stronger claim than "accumulation over time".
2. Why does GRPO, specifically, turn a long tail of rollout durations into a staleness problem? Would a method with a learned value baseline and per-sample advantages have the same group barrier? Give the reason.
3. Table 1 reports 123 runtime-elasticity events. Explain why a run that climbs from base would need to change its data mixture and GPU allocation more often than a short refinement run, using the post's own statement about what shifts as the policy improves.
4. slime's placement comment and Rocket's learner recovery give the same reason for keeping trainer and inference in separate allocations. State that reason, and explain what state the inference side holds that would be lost if both were restarted together.
5. Rocket owns the learner and an SGLang fork, while verl supports a product of training backends and rollout engines. For bitwise sampler–learner parity, explain why owning both sides changes what is feasible, using one concrete mechanism from §5.2.
6. MAI-Thinking-1 ran with BF16 on both sides, and MAI Base 2 moved the sampler to MXFP8 storage. What property of the new model forced this change, and what new requirement did the change create for the learner?

---

## Connections

- **Next: ch-02** — the Rocket loop in detail (controller, problem workers, rollout workers, sampler, learner) and how colocated and fully asynchronous open designs differ from it.
- **ch-03, ch-04, ch-05** — the staleness challenge of §2: definition, levers, weight transfer.
- **ch-06, ch-07** — the numerical-consistency challenge of §3: low-precision storage and bitwise parity.
- **ch-08** — agentic rollouts, the reason rollout durations reach hours.
- **ch-20, ch-21** — the fault-tolerance and elasticity mechanisms behind Table 1.
- **llm-training ch-40** (group-baseline RL) — why GRPO needs the whole group before any rollout can be trained.
- **llm-training ch-54** (rollout infrastructure, off-policy data, async RL) — the earlier survey of async rollout systems that this course extends.
- **llm-training ch-44** (verifiable rewards) and **ch-45b** (multi-turn agentic RL) — the reward and rollout types that produce Fig. 2's duration spread.

---

## Sources

- [[mai-rl-infrastructure]] — anchor: the recipe argument (p. 1–2), the three challenges, the ownership split, Table 1, Fig. 2, Fig. 8.
- [[mai-thinking-1]] — Rocket v1: why MAI built it (Sec. 3.6), 4,864-chip job split, BF16 numerics, self-distillation finding on resuming after collapse.
- [[slime]] — single train loop for sync and async; separate placement groups so a restarted trainer can resize; code quoted from slime@79aa41a.
- [[verl-rollout]] — verl rollout modes (HYBRID / COLOCATED / STANDALONE); code quoted from verl@8357f1e.
- [[olmo-3-rl-pipeline]] — open RL lengths (Table 49) and wall clock (≈15 days → 6 days) for context.
- [[scalerl]] — a long open climb (100K GPU-hours, about 7,400 steps) for context.
- [[areal-async-rl]] — open async system and its run lengths (Table 1).
- [[open-instruct-rlvr-pipeline]] — ρ correction with upper bound 2.0 as the open alternative to parity.
- [[pipelinerl]] — 1,000-step in-flight-update run for context.
