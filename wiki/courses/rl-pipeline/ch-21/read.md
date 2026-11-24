<!-- chapter: ch-21
     track: rl-pipeline
     kind: content
     title: Elasticity: Resizing Components and Changing Config Mid-Run
     deps: [ch-20]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[olmo-3-rl-pipeline]], [[slime]], [[verl-rollout]], [[areal-async-rl]], [[open-instruct-rlvr-pipeline]]
     figures: figures/elasticity.html
-->

# Chapter 21 — Elasticity: Resizing Components and Changing Config Mid-Run

> **Core insight.** A weeks-long RL run cannot be configured correctly at launch, because the quantities that decide the right configuration change during the run: rollout lengths grow, the domain mixture shifts, GPUs fail and come back, and the inference-to-learner balance that kept the job inference-bound at step 100 is wrong at step 1,000. Rocket treats two things as changeable without a restart. **Compute elasticity**: learner, inference and controller scale independently; inference resizes immediately, the learner only after its next checkpoint save, with topology-aware placement and a spare node group sized to one learner slice. **Runtime elasticity**: a fixed set of typed, validated settings (inference capacity and concurrency, admission, staleness limits, batch packing, mixture weights) can be changed and take effect at step or sampling boundaries ([[mai-rl-infrastructure]] p. 12–13). Table 1 attributes the single largest saving in the post to runtime elasticity: 123 events, 2 h saved each, 246 h, 9.88% of training time, 1,476,000 GPU-hours on a 6,000-GPU basis (p. 11).
>
> **Guideline.** When a setting must change during a run, apply it at the boundary where the quantity it controls is defined (a step boundary for batch packing and staleness limits, a sampling boundary for admission and mixture weights), because a change in the middle of a step would mix two configurations in one batch; Rocket applies typed settings only at those boundaries (p. 13). When the learner and inference fleet are separate allocations, resize inference freely but resize the learner only from a checkpoint, because the learner's optimizer state is sharded by its parallel layout and Rocket resizes the learner "only after the next checkpoint save" (p. 12–13); slime's equivalent requires `torch_dist` checkpoints with reshardable optimizer state (`training_recovery.py` L54–57). When adding inference capacity, re-check staleness and batch size together, because adding inference servers at fixed batch size can *increase* staleness (p. 7; ch-04).

---

## Why this chapter matters

The two Rocket runs described in the library differ by an order of magnitude in what they resize:

- **Rocket v1 (MAI-Thinking-1).** Up to 5:1 inference:learner GPUs; the largest job used 4,864 GB300 (4,096 inference, 768 learner) ([[mai-thinking-1]] Sec. 3.6). Its weight-transfer plan is "computed once at job start" by intersecting learner and inference sharding layouts per parameter, then expanded to live replicas.
- **Rocket for MAI Base 2.** A three-trillion-parameter base, many thousands of steps over weeks to months on thousands of GPUs ([[mai-rl-infrastructure]] p. 1), with independent scaling of each component and runtime settings (p. 12–13).

The open-source reference points show why a fixed allocation drifts out of balance. In Olmo 3's 32B Think run, the learner used 8 H100 nodes and inference 20 nodes, and the learner spent 75% of the time waiting for data; a representative step took 1,000 s, of which 125 s was training ([[olmo-3-rl-pipeline]] §4.4.3). Rollout lengths in Rocket's production mixture span 2 min (instruction following) to 93 min (SWE, 249 turns) at the median ([[mai-rl-infrastructure]] Fig. 2). As the mixture or the maximum length changes, the generation work per training step changes, and the allocation that matched them no longer does.

Restarting the job to change any of this costs the recovery time of ch-20 plus the loss of in-flight rollouts. Runtime elasticity is the mechanism that avoids those restarts; Table 1 values it above all fault-tolerance mechanisms combined (246 h vs 186.5 h).

---

## §1 Two kinds of elasticity

| | Compute elasticity | Runtime elasticity |
|---|---|---|
| What changes | number and placement of GPUs per component | values of loop settings |
| Examples (Rocket) | inference replicas added or removed; learner resized; spare learner slice used | inference capacity and concurrency, admission, staleness limits, batch packing, mixture weights |
| When it takes effect | inference: immediately; learner: after the next checkpoint save | at step or sampling boundaries |
| What makes it safe | independent components; weight transfer to new replicas (ch-05); resharding from checkpoint | typed, validated settings |
| Table 1 row | none separate | n = 123, 2 h each, 246 h, 9.88%, 1,476,000 GPU-hours |
| Source | [[mai-rl-infrastructure]] p. 12–13 | p. 11, 13 |

The card does not say whether the 123 runtime-elasticity events include resizes, or only setting changes. It also does not record how the 2 h per-event saving was estimated (ch-20 §7).

> Interactive: [figures/elasticity.html](figures/elasticity.html) — Rocket's two-layer scheduling with resize and slice-failure events on an illustrative node pool, a timeline showing where each runtime setting and resize takes effect (after a training step, before a sampling operation, after the next checkpoint save), and the Table 1 elasticity vs fault-tolerance savings.

---

## §2 Compute elasticity: why inference resizes immediately and the learner does not

**Inference.** A new inference replica needs three things: GPUs, the current weights, and registration with the router. Rocket's two-stage weight transfer (selected sampler engines pull packed chunks from the learner, then distribute to matching ranks across replicas) makes the learner-side cost independent of fleet size; a transfer takes 15 s "in our current configuration" (p. 7; ch-05). A replica added between transfers can receive weights from a peer replica of matching layout. Nothing in the learner or controller must change, so the resize can happen immediately. Removing a replica is the same operation as replacing a failed one (ch-20 rung 1), without the failure.

**Learner.** The learner's state is weights plus optimizer moments, sharded across data, tensor, pipeline and expert parallel ranks. Changing the number of learner GPUs changes the sharding, so the state must be re-laid-out. The point at which a consistent, complete copy of that state exists on durable storage is a checkpoint. Rocket resizes the learner only after the next checkpoint save (p. 12–13). The post gives no resize time.

**slime's version of the same rule.** slime keeps actor (learner) and rollout placement groups separate when not colocated, with the stated reason that a restarted trainer can resize without moving serving engines:

```python
# slime@79aa41a slime/ray/placement_group.py:L123–L138
def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

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

When a new trainer attaches to the retained serving cluster, slime replaces the actor placement group if the GPU count changed, and refuses to grow a colocated trainer beyond the retained allocation:

```python
# slime@79aa41a slime/ray/serving.py:L137–L148
        reused = self.attachments > 0
        count = args.actor_num_nodes * args.actor_num_gpus_per_node
        actor_pg = self.placements["actor"]
        if args.colocate:
            # This placement is shared with live serving, so it cannot be resized.
            if count > len(actor_pg[1]):
                raise ValueError("Restarted colocated trainer exceeds the retained GPU placement")
        elif not args.debug_rollout_only and count != len(actor_pg[1]):
            if actor_pg[0] is not None:
                remove_placement_group(actor_pg[0])
            self.placements["actor"] = _create_placement_group(count)
            self.placements["critic"] = self.placements["actor"] if args.use_critic else None
```

The resharding requirement is enforced at configuration time:

```python
# slime@79aa41a slime/ray/training_recovery.py:L47–L57
def configure_recovery_checkpoint(args):
    """Apply the same recovery policy before and after role-specific overrides."""
    stateless = getattr(args, "use_stateless_adam", False)
    if getattr(args, "no_save_rng", False) or (getattr(args, "no_save_optim", False) and not stateless):
        raise ValueError(
            "Trainer fault tolerance requires optimizer and RNG state; only --use-stateless-adam may omit optimizer state"
        )
    if args.ckpt_format != "torch_dist":
        raise ValueError("Trainer fault tolerance requires --ckpt-format torch_dist for parallelism changes")
    args.ckpt_fully_parallel_save = True
    args.dist_ckpt_optim_fully_reshardable = True
```

The slime learner resize is therefore a restart-time operation (resubmit with new TP/CP/EP or GPU count; `fault-tolerance.md` L35–43), tied to the last checkpoint, which is the same boundary Rocket uses. The difference is that Rocket's is automatic inside the running job and slime's requires a new submission.

**Other frameworks.**

| Framework | Learner resize | Inference resize | Source |
|---|---|---|---|
| Rocket (Base 2) | after next checkpoint save, in-job | immediate | [[mai-rl-infrastructure]] p. 12–13 |
| slime | on trainer resubmit, from a `torch_dist` checkpoint; colocated trainer cannot exceed retained allocation | engines restarted after failure; resize of serving not documented in the cards | `placement_group.py` L126–138; `serving.py` L140–148 |
| AReaL | not supported on recovery: DCP checkpoints "require identical parallelism configuration" | allocation fixed per engine by its `backend` string (e.g. `sglang:d4t2`) at launch | `checkpointing.md` L254–255; `alloc_mode.md` L21–38 |
| verl v1 async | not documented | `separate_async` with switch lends trainer GPUs to generation between steps (hybrid replicas), described as a way to "reduce trainer idle time when it's hard to set a perfect rollouter-trainer ratio" | `v1_async_trainer.md` L8, L17–22 |
| open-instruct | not documented | not documented | — |

verl's hybrid switch is a different answer to the same problem: instead of resizing the inference fleet, it changes what the trainer GPUs do between steps, so a ratio that is wrong at launch costs less idle time. verl's doc reports one measurement (`v1_async_trainer.md` L308–322): two otherwise identical runs for the first 100 steps on 3 × 8 H100 (two trainer/hybrid nodes, one standalone rollout node), Qwen3.5-35B-A3B, DAPO-Math-17k, max response length 32,768, decoupled PPO disabled.

| Mode | 100-step training time | Aggregate tokens/s | Mean response length | Mean reward |
|---|---|---|---|---|
| no-switch | 13.15 h | 14,604 | 13,343 | 0.7755 |
| switch | 11.79 h (−10.3%) | 16,445 (+12.6%) | 13,478 | 0.7762 |

This is one run per arm over 100 steps; it shows a throughput gain with equal mean reward at that horizon, not an effect on final quality.

---

## §3 Placement: topology-aware and a spare learner slice

**Topology-aware placement.** Rocket places components topology-aware on rack-scale Blackwell (p. 12). The card gives no detail of the placement rule beyond Fig. 7, "two-layer scheduling" (p. 12), whose content was not transcribed. The general reason for topology awareness in this design is that the learner's collectives and the MoE all-to-all of ch-07 run fastest within a rack-scale NVLink domain, so a learner slice that straddles domains is slower than one that does not. That is the standard motivation; the post's own stated reason is not in the card.

**A spare learner slice.** Rocket keeps a spare node group sized to one learner slice (p. 12–13). The reason is not stated in the card. One consequence follows from ch-20: a learner-only recovery needs healthy GPUs in the right topology for the rebuilt learner. If a node failure removed part of a learner slice, the rebuild must either wait for the scheduler to return a replacement node or shrink the learner (which needs a checkpoint and a resharding). A pre-allocated spare slice of the right shape removes that wait. Read this as an inferred rationale.

**Sizing guidance in the sources.**

| Run | Inference : learner | Observation | Source |
|---|---|---|---|
| Rocket v1 largest job | 4,096 : 768 GPUs (about 5.3:1) | "up to 5:1" stated as the regime | [[mai-thinking-1]] Sec. 3.6 |
| Olmo 3 32B Think | 20 : 8 nodes; ≈5× inference vs training GPU use | learner waited 75% of the time | [[olmo-3-rl-pipeline]] §4.4.3 |
| Olmo 3 7B Think | 7 : 2 nodes; ≈14× | — | same |
| Rocket Base 2 | not stated | deliberately inference-bound; learner overallocated, partly idle | [[mai-rl-infrastructure]] p. 7 |

Rocket chooses to be inference-bound (ch-04) because a learner that waits for data costs idle learner GPUs, while an inference fleet that waits for the learner raises staleness. Elasticity is what lets the run stay in the chosen regime as the workload changes.

---

## §4 Runtime elasticity: typed settings at boundaries

**The settings.** Rocket lists inference capacity and concurrency, admission, staleness limits, batch packing, and mixture weights as typed, validated runtime settings applied at step or sampling boundaries (p. 13). Each maps to an earlier chapter:

| Setting | What it controls | Natural boundary | Chapter |
|---|---|---|---|
| Inference capacity and concurrency | in-flight rollout count per replica; total generation throughput | sampling (admission of new requests) | ch-04 |
| Admission | which problems are admitted and when; rollout-age priority | sampling | ch-04 |
| Staleness limits | maximum policy-version gap at training time | step (batch formation) | ch-03 |
| Batch packing | how training segments are packed into a learner batch | step | ch-02 |
| Mixture weights | share of each domain in the problems the controller selects | sampling | ch-17 |

**Why boundaries.** Each setting is read at one place in the loop. A staleness limit is applied when the controller forms a batch; changing it while a batch is half-formed would mean that part of the batch was filtered by the old limit and part by the new one. Mixture weights are read when the controller selects the next problem; changing them between selections is clean, changing them while a GRPO group is being dispatched is not, because the group's members must all come from one problem. Applying the change at the boundary where the setting is read means every unit of work (one batch, one group) sees exactly one configuration. That is the reasoning; the post states the rule without an ablation.

**Why typed and validated.** A runtime setting that crashes the controller turns a configuration change into a recovery event (ch-20). Validation before application keeps a bad value from reaching the loop. The post gives no details of the validator. The open frameworks validate at launch, not live:

- open-instruct's MILES integration provides `plan`, `validate`, and `train` with repeatable `--set SECTION.KEY=TOML_VALUE` overrides, applied "before validation" (`docs/miles/run-controls.md` L24–37). `plan` checks option names, basic types and choices without importing the runtime; `validate` invokes the real parser. This is the closest open analogue to typed, validated settings, applied before the run starts.
- slime freezes the rollout configuration across trainer restarts. A retained serving session refuses to reattach if any rollout-side setting changed:

```python
# slime@79aa41a slime/ray/training_recovery.py:L241–L251
        if self.journal.exists():
            state = torch.load(self.journal, weights_only=False)
            changed = [
                name
                for name in state["configuration"].keys() | self.configuration.keys()
                if name not in state["configuration"]
                or name not in self.configuration
                or state["configuration"][name] != self.configuration[name]
            ]
            if changed:
                raise ValueError("Retained rollout session requires unchanged configuration: " + ", ".join(changed))
```

  The set of settings allowed to change on reattach is an explicit allowlist of trainer and operational fields (parallel sizes, micro-batch size, token limits per GPU, recompute settings, save/eval intervals, health-check timings; `training_recovery.py` L60–156). Sampling, filters, reward hooks and custom arguments are frozen (`fault-tolerance.md` L45). This is the reverse of Rocket's list: in slime, loop-side settings (admission, mixture, staleness) are the ones that cannot change without a new rollout session, and trainer-side settings are the ones that can, at restart.
- verl and AReaL: no live configuration change is documented in the cards or docs read; configuration is set at launch (Hydra config; per-engine `backend` strings).

**What runtime elasticity is used for.** The card does not list what the 123 events changed. From the settings list and the earlier chapters, examples consistent with the design are: raising concurrency when rollouts get longer (ch-04); tightening a staleness limit after the log-prob gap grows (ch-03, Fig. 3: mean |log-prob difference| rises with staleness, R² = 0.76); and shifting mixture weights as domains saturate (ch-17). These are examples, not reported events.

---

## §5 Interactions with staleness and recovery

1. **More inference can raise staleness.** At fixed batch size, adding inference servers makes training steps complete faster while a long rollout's wall-clock time is unchanged, so the long rollout spans more learner versions (p. 7; ch-04). An inference resize should therefore be paired with a check of the staleness distribution and, if needed, a change to the batch size or staleness limit, which are runtime settings in Rocket. This is why the two kinds of elasticity are useful together.
2. **Learner resize and weight transfer.** In Rocket v1 the weight-transfer plan was computed once at job start from both sharding layouts ([[mai-thinking-1]] Sec. 3.6). A learner resize changes the learner layout, so a plan of that kind must be recomputed. The Base 2 post does not describe how its two-stage transfer handles a learner resize. Tying the learner resize to a checkpoint boundary gives a natural point to recompute it; that link is inferred.
3. **Recovery and resize share machinery.** slime's resize path is its recovery path (resubmit to retained serving). Rocket's learner-only recovery and learner resize both restore from a checkpoint into a rebuilt learner. Designing one gives most of the other.

---

## §6 Deciding what to make runtime-mutable

A setting should be runtime-mutable when (a) the right value depends on quantities that drift during the run, (b) it is read at a well-defined boundary, and (c) a wrong value is recoverable by changing it back. A setting should be restart-only when changing it invalidates state already accumulated.

| Setting | Drifts during run? | Boundary | Invalidates state? | Rocket | slime |
|---|---|---|---|---|---|
| Inference concurrency | yes (rollout lengths) | sampling | no | runtime | frozen (rollout config) |
| Admission / priority | yes | sampling | no | runtime | frozen |
| Staleness limit | yes (gap grows, Fig. 3) | step | no | runtime | frozen |
| Batch packing | yes (length mix) | step | no | runtime | frozen |
| Mixture weights | yes (domain saturation) | sampling | no | runtime | frozen |
| Learner parallel layout | yes (GPU availability) | checkpoint | optimizer sharding | after checkpoint | on resubmit |
| Reward function / grader | — | — | yes: changes what earlier advantages meant | not listed | frozen |
| Sampling temperature / top-p | — | — | changes logged log-probs semantics; top-p mask replay (ch-07) | not listed | frozen |
| Model architecture, tokenizer | no | — | yes | not listed | frozen |

The last three rows are not in Rocket's list. The card does not say they are forbidden; they are not listed as runtime settings. A reward change mid-run changes what the policy is optimizing and makes before/after curves incomparable, which is a reason to treat it as a new run segment rather than a setting.

---

## Recipe

1. Separate learner, inference and controller into independent allocations (slime `placement_group.py` L126–138 pattern).
2. Make inference replicas join by receiving weights from the transfer path and registering with the router; make removal identical to failure replacement (ch-20).
3. Save learner checkpoints in a reshardable format with optimizer and RNG state (slime: `torch_dist`, reshardable optimizer); resize the learner only from a checkpoint.
4. Keep a spare learner-slice-sized node group in the right topology if learner-only recovery is a goal.
5. Define the runtime settings explicitly with types and ranges: concurrency, admission, staleness limit, batch packing, mixture weights. Validate every change before applying it.
6. Apply each setting at the boundary where it is read: step boundary for staleness and packing; sampling boundary for admission, concurrency and mixture.
7. After any inference resize, check the staleness distribution and adjust batch size or staleness limit (ch-04).
8. Keep reward, sampling parameters and architecture restart-only; treat a change as a new run segment.
9. Log each elasticity event with the setting, old value, new value, step, and the reason; estimate its saving with a stated counterfactual so that a Table 1 equivalent is reproducible.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Adding inference without adjusting batch or staleness limit | staleness rises after a capacity increase | staleness histogram before and after each resize (ch-03) |
| Changing a setting mid-batch | one batch mixes two staleness filters or two mixtures | tag every batch and group with a config version; assert one version per unit |
| Non-reshardable learner checkpoints | learner cannot resize, or resize requires a fresh start | test a resize from a checkpoint before the long run (slime enforces `torch_dist`) |
| Mutable reward or sampling settings | reward curve jumps without a policy change | config-version tag on every logged reward |
| Unvalidated runtime settings | a config change triggers a recovery event | count recovery events following config changes |
| Fixed inference:learner ratio for the whole run | learner idle fraction drifts upward as lengths grow (Olmo: 75% waiting) | learner idle fraction and inference utilization per step |

---

## Check your understanding

1. Why can an inference replica be added immediately while the learner must wait for a checkpoint? State the answer in terms of what state each component must reconstruct.
2. Explain why adding inference servers at fixed batch size can raise staleness. Which runtime setting would you change with the resize, and in which direction?
3. Rocket makes mixture weights runtime-mutable and slime freezes them across a retained session. What failure is each design protecting against?
4. Why should a staleness limit be applied at a step boundary rather than whenever the operator changes it?
5. verl lends trainer GPUs to generation between steps instead of resizing the fleet. Under what workload is that better or worse than Rocket's independent scaling?
6. Table 1 values runtime elasticity above both fault-tolerance rows combined. What would you need to log in your own run to check whether that ordering holds for you?

---

## Connections

- Course chapters: ch-02 (components and batch packing), ch-03 (staleness definition and Fig. 3), ch-04 (inference-bound sizing; more inference can raise staleness), ch-05 (weight transfer to new replicas), ch-07 (sampling-mask replay; why sampling settings are restart-only), ch-17 (mixture weights as runtime config), ch-20 (recovery machinery shared with resize), ch-22 (capstone recovery/elasticity row).
- llm-training: ch-16 (RL prompt distribution; why mixture weights drift), ch-44a (length growth that changes the inference load).
- Sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[olmo-3-rl-pipeline]], [[slime]], [[verl-rollout]], [[areal-async-rl]], [[open-instruct-rlvr-pipeline]].

---

## Sources

- [[mai-rl-infrastructure]] — compute and runtime elasticity (p. 12–13); Table 1 runtime-elasticity row; staleness counter-example (p. 7).
- [[mai-thinking-1]] — Rocket v1 inference:learner ratio and job size; weight-transfer plan computed at job start.
- [[olmo-3-rl-pipeline]] — learner/inference node counts and 75% learner wait time.
- [[slime]] — separate placement groups, trainer resize on reattach, reshardable checkpoints, frozen rollout configuration (code at 79aa41a).
- [[verl-rollout]] — v1 async trainer modes, hybrid replica switch (docs at 8357f1e).
- [[areal-async-rl]] — per-engine allocation strings; identical-parallelism requirement for recovery (docs at bcef492).
- [[open-instruct-rlvr-pipeline]] — MILES plan/validate/`--set` run controls (docs at f020e9a).
