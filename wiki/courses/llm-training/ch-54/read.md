<!-- chapter: ch-54
     track: infra
     kind: content
     title: Rollout, Replay, Async Infrastructure
     deps: [ch-53]
     sources: [[best-of-n]], [[on-off-policy-rlhf]], [[replay-buffer-rlhf]], [[async-rollout]],
              [[minibatch-sharing-rl]], [[verl-rollout]], [[openrlhf-ppo]], [[trl-ppo]],
              [[policy-coverage-loss]], [[rejection-sampling-finetuning]]
     figures: figures/async-pipeline.html
     opens_track: infra (ch-54..ch-60)
-->

# Chapter 54 — Rollout, Replay, Async Infrastructure

> **Core insight.** RL for LLMs is not a loss problem, it is a pipeline problem. Once [[dpo]] / [[ppo]] / [[grpo]] are five-line objectives (ch-37..ch-40) and the reward signals are wired (ch-42..ch-44), the remaining 70% of wall-time is **rollout**, and the remaining 90% of engineering risk is **how stale the data is when the gradient finally sees it**. The infra track opens here by laying a four-axis map over that pipeline: *who generates* (policy vs frozen teacher), *how fresh* (on-policy ↔ off-policy staleness `k`), *how reused* (no replay, prompt replay, trajectory replay), *how parallel* (sync, async queue-1, async queue-2, partial rollout). Every production stack — TRL, OpenRLHF, verl — is one corner of that cube, and moving between corners has a price denominated in KL, variance, and clipfrac.
>
> **Guideline.** Your first move on a new RL run is not an algorithm choice — it is: (1) try Best-of-N against your RM to get a KL-matched baseline ceiling ([[best-of-n]]); (2) if you must train, start on-policy (iterative DPO or PPO/GRPO) because [[on-off-policy-rlhf]] shows ~80% of the gap offline DPO takes is pure distribution shift; (3) do **not** replay trajectories — replay *prompts*, keyed by group-reward variance ([[replay-buffer-rlhf]]); (4) once the run exceeds ~30 min, go async ([[async-rollout]]) with queue depth 1–2 and watch `vllm_kl`; (5) set `n` rollouts per prompt ≥ 4 so the group baseline is not noise ([[minibatch-sharing-rl]]). Every later infra chapter (ch-55 verl, ch-56 OpenRLHF, ch-57 distributed training) is a deep dive into *one* of those five moves.

---

## 1. The four-axis map

Everything in this chapter sits on one of four axes of the rollout → train loop:

| Axis | Endpoint A | Endpoint B | Paid by |
|---|---|---|---|
| Generator | policy π_t | frozen teacher π_ref / stronger model | distribution shift (DPO vs iterative DPO) |
| Freshness | on-policy (k=0) | off-policy (k=queue_depth + partial) | IS correction, ratio blow-up, `vllm_kl` |
| Reuse | no replay | prompt replay / trajectory replay | bias, ratio explosion, stale KL term |
| Parallelism | sync (trainer idle during rollout) | async queue + partial rollout | weight-broadcast pause, engine lock |

[[trl-ppo]] sits at (π_t, k=0, no replay, sync) — the reference corner. [[openrlhf-ppo]] async and [[verl-rollout]] vllm_async_server.py both move along the parallelism axis with correction on the freshness axis. [[replay-buffer-rlhf]] adds the reuse axis. Figure [[figures/async-pipeline.html]] lets you slide along parallelism + freshness and watch throughput vs staleness in real time.

---

## 2. Best-of-N — the baseline your PPO has to beat

Before any RL run, run [[best-of-n]]. Stiennon 2020 (OpenAI TL;DR summarization) established the closed form that governs every subsequent BoN analysis:

```
KL(BoN || base) = log N − (N−1)/N        # analytic, tight for well-calibrated RM
                ≈ log N   for large N
```

So BoN-64 sits at KL ≈ ln 64 − 63/64 ≈ 3.17 nats — exactly the budget PPO gets if you set β so that the attained KL is ~3. At that matched budget, Stiennon's Figure 6 shows BoN-64 within ~2 human-preference points of a well-tuned PPO, with **zero training cost**. That is the ceiling your PPO run has to beat to justify the engineering. It often does not: BoN-64 is frequently chosen as the deployment strategy (Anthropic's test-time compute work, Cohere's chat models; see [[best-of-n]] §Connections).

**The ceiling.** BoN's gold-vs-proxy curve on Stiennon's Figure 4 is the first documented **reward-model overoptimization**: past the critical KL, RM score keeps rising while human preference falls. This is the same law formalized by [[reward-model-overoptimization]] (ch-41 §2). BoN is monotone in N only if the RM is faithful; above `d* = α_bon / (2 β_bon)` nats^0.5 even argmax-over-N degrades. So BoN's ceiling is the RM's ceiling.

**RSFT — one rung up.** [[rejection-sampling-finetuning]] is BoN applied to *training data* instead of inference: sample N per prompt, filter, SFT on survivors, optionally iterate. Llama 2's post-training is the canonical recipe; it captures part of BoN's gain without the per-query inference cost, but caps at the same RM ceiling. Treat RSFT as "BoN cached into weights" — useful bridge between SFT and full RLHF.

**Decision rule.** If your RM is trustworthy and you can afford N = 16–64 at inference, BoN is the move. If the RM is shaky (overopt peak at `d ≈ 3`), BoN-8 plus a tighter RM is still usually cheaper and safer than full PPO.

---

## 3. On-policy vs off-policy — the only number that matters is `k`

[[on-off-policy-rlhf]] (Tang 2024, DeepMind) settled a long dispute: **offline DPO underperforms PPO not because DPO is algorithmically weaker, but because its training data is drawn from a different distribution than the current policy**. The decomposition of the gap on TL;DR / HH / GSM8K (Gemma-2B and 7B):

- ≈ 80% distribution-shift contribution
- ≈ 20% variance-reduction contribution

Iterative DPO — sample chosen/rejected pairs from π_t at each step, label with a frozen RM, DPO-update with β=0.1, keep π_0 as the frozen reference — **matches PPO** on all three tasks, with lower seed variance. The lesson generalizes:

```
gap(algorithm) = gap(distribution_shift) + gap(variance) + gap(algebra)
              ≈         0.8               +      0.2      +      ~0
```

The algebra of PG vs closed-form does not matter at this scale. *Which distribution you sample from* does.

**[[policy-coverage-loss]] sharpens this.** Distribution shift is not a scalar; it is a shape. A source policy / dataset is useful iff its **support covers** the target-optimal policy's support — you can afford an imperfect reward model if it induces a policy that reaches the right regions. This is why offline DPO on Anthropic-HH helps some tasks and not others: it depends on whether the HH pair-distribution covers your target's action regions. For production: measure source-policy **win rate** against a held-out reference before transferring.

**Staleness as a scalar.** In async RL (§5), off-policyness is parameterized by `k = queue_depth + partial_rollout_depth`. Tang's "on-policy bonus" collapses monotonically in `k` — at `k=1` (1-step queue) the gap is negligible; at `k=5` it approaches offline-DPO territory. This is why [[async-rollout]] keeps queue depth at 1–2 and monitors `vllm_kl` as the canary.

---

## 4. Replay — why trajectories break and prompts don't

Classical RL (DQN, Ape-X, R2D2) relies on experience replay. LLM RL does not, and [[replay-buffer-rlhf]] explains exactly why:

**Why trajectory replay fails for PPO/GRPO** (four mechanisms, compounding):

1. **Ratio explosion.** The IS weight `exp(Σ_t Δ log π_t)` accumulates token-wise. With 100-token responses and 0.01-nat drift per token per update, the ratio exceeds any clip bound within ~20 updates. The PPO clipfrac pegs at 1; the gradient is ~all-zero.
2. **Stale advantages.** GRPO's group baseline `μ_i = mean_k R_{i,k}` is computed against the policy at rollout time; replaying yesterday's advantages trains you against yesterday's baseline, a form of reward hacking against your old self.
3. **Compute asymmetry.** In LLM RL, rollouts are cheap relative to update cost (vLLM prefill is bandwidth-bound, forward+backward is compute-bound). Trajectory reuse saves little.
4. **KL-to-ref mismatch.** The per-token KL reward (§6) is computed against `π_ref` at rollout time; if you freeze `π_ref`, that is fine, but if you adapt it, replayed rewards are now against the wrong reference.

DeepSeek-R1 §3.2 reports attempting trajectory replay and abandoning it for fresh rollouts + larger batch — that is the attested negative result.

**What works: prompt-level replay.** TRL's experimental `GRPOWithReplayBufferTrainer` (`trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py`) keeps a FIFO buffer of `(prompt_ids, rewards, variance)`:

```
# TRL prompt-replay schema (simplified from the real trainer)
@dataclass
class BufferEntry:
    prompt_ids: torch.Tensor
    rewards:    torch.Tensor          # (n,) per-rollout outcome rewards
    variance:   float                  # rewards.var().item()
    seen_steps: int
```

Sampling probability `p_i ∝ var_i + ε`: prompts with all-same rewards (all pass or all fail) have group variance 0, contribute zero GRPO gradient, and get downweighted in replay. The old *completions* are discarded — only the *prompt* is replayed; the engine resamples fresh responses under the current π_t. No IS correction needed, because there is no old-policy content in the gradient.

**Verdict on replay buffers.** OpenRLHF ships no buffer (`rollout → train → discard`). verl's `experience_makers/experience_buffer.py` is an intra-step scratch pad for PPO epochs, not cross-step replay. TRL's is experimental and gated behind `p_replay=0.25`. Treat prompt-replay as a diagnostic for curriculum (see hard prompts twice) and variance reduction, not an off-policy correction trick.

---

## 5. Async actor-learner — the queue is the architecture

Synchronous RL (the [[trl-ppo]] inner loop) wastes half the machine: the rollout GPUs idle while the trainer runs `backward()`, and vice versa. [[async-rollout]] (OpenRLHF + verl HybridFlow) recovers 1.6×–2.0× throughput by decoupling the two processes across a bounded queue, at the price of a bounded staleness `k ≤ queue_depth + partial_rollout_depth`.

**Architecture in prose.**

```
  prompts ─▶ [RolloutActor]───generate(π_{t−k})──▶ ┌────────────┐
               x R workers                          │ rollout_   │
               (vLLM AsyncLLMEngine)                │   queue    │
                                                    │  depth d=1 │
                                                    └─────┬──────┘
                                                          │
                                                  (FIFO pop, k≤d+partial)
                                                          ▼
                                          ┌─────────────────────────┐
                                          │   TrainingActor         │
                                          │   score → GAE / GRPO    │
                                          │   → clipped PG step     │
                                          │   → optimizer.step()    │
                                          └─────────┬───────────────┘
                                                    │ broadcast_weights
                                                    │ (holds vllm_lock)
                                                    ▼
                                          [engine.pause() — wait
                                           in-flight to drain or
                                           interrupt via partial-
                                           rollout interrupt hook]
                                                    │
                                                    └──▶ back to RolloutActor
```

**OpenRLHF primitives** (from `openrlhf/trainer/ppo_trainer_async.py`):

- `rollout_queue`: `ray.util.queue.Queue` capacity 1–2.
- `rollout_slots`: companion backpressure queue carrying `global_step` tokens (one slot per allowed in-flight rollout).
- `vllm_lock`: `ray` asyncio lock serializing weight-broadcast vs generate.
- `strategy.args.train.partial_rollout_enable`: let in-flight generates be interrupted on weight update (continuous-batching RL).

**verl primitives** (from `verl/workers/rollout/vllm_rollout/vllm_async_server.py`):

- `engine.generate(..., priority=…)`: per-request priority so newly-weighted requests jump the queue.
- `engine to paused state` block (≈ line 628): hard-pause before weight broadcast, resume after.
- SPMD sync vLLM was **retired** (PR #4411) — new deployments must use async; the sync path is only `HFRollout` (reference, debug).
- Tokens-in / tokens-out — the async server never touches the tokenizer (cheap multi-turn tool-use).

**Staleness bound and the canary.** `k = queue_depth + partial_rollout_depth` — typically 1–2. Correction runs in [[openrlhf-ppo]]'s `PolicyLoss.forward`:

- Per-token truncated-IS weight `exp(log π_train − log π_rollout)` clipped to `[low, high]` (mode `tis`).
- Sequence-level mask dropping sequences whose seq-IS escapes `[low, high]` (`seq-mask-tis`).
- Or per-token mask on tokens outside the range (`icepop`).
- Hard-cap drop on entire rollouts with seq-IS over a threshold.

The canary metric returned from `PolicyLoss.forward` as `vllm_kl` is `masked_mean(rollout_log_probs − old_log_probs, action_mask)`. Spec: `vllm_kl > 0.1` together with PPO clipfrac pegged at 1 is the attested signal that staleness has exceeded what IS correction handles — either the queue depth grew (producer is faster than consumer) or the vLLM fp-precision differs from the trainer's (bf16 rollout vs fp32 train).

**Ancestry.** Both stacks descend from IMPALA V-trace (Espeholt 2018) and Ape-X / R2D2. The IS-weight clip is *the same mathematical object* as V-trace's ρ̄ and c̄; only the policy parameterization changed.

**Companion.** [figures/async-pipeline.html](figures/async-pipeline.html) animates the pipeline: slide queue depth, rollout/learner speed ratio, partial-rollout on/off, and watch throughput + staleness (`k`) + `vllm_kl` proxy evolve.

---

## 6. Mini-batch sharing — B × n, not B

For critic-free objectives (GRPO, RLOO, REINFORCE++), the advantage is a **group-relative** statistic — you need ≥ 2 rollouts per prompt just for a baseline. [[minibatch-sharing-rl]] synthesizes the ablations:

```
A_{i,k}^{GRPO} = (R_{i,k} − μ_i) / (σ_i + ε),   μ_i = mean_k R_{i,k}   (need n ≥ 2; stable n ≥ 4)
A_{i,k}^{RLOO} = R_{i,k} − (1/(n−1)) · Σ_{j≠k} R_{i,j}                 (variance ∝ 1/(n−1))
```

**Framework defaults — attested.**

| Framework | Config knob | Default `n` | Batch recipe |
|---|---|---|---|
| verl | `actor_rollout_ref.rollout.n` | 8 | B=128 prompts × n=8 = 1024 seqs |
| TRL (GRPO) | `num_generations` | 8 | B=64 × n=8 = 512 seqs |
| OpenRLHF (PPO) | `n_samples_per_prompt` | 4 | B=128 × n=4 = 512 seqs |
| TRL (PPO, classic) | N/A (critic available) | 1 | B=256 × n=1 = 256 seqs |

**Why `n` is cheaper than linear.** vLLM batches same-prompt requests together and shares the prompt-prefix forward pass — only the decode half scales with `n`. Going from `n=1` to `n=8` on a 1024-token prompt with 512-token response costs ~1.4× wall time, not 8×.

**Per-prompt vs pooled advantages.** Two patterns attested:

- *Per-prompt* (GRPO, RLOO): compute `μ_i`, `σ_i` within each prompt's group. Removes prompt-difficulty bias (easy prompts have high baselines; hard ones low). The defensible default.
- *Pooled*: subtract the global batch mean. REINFORCE++ uses this when prompts are homogeneous enough that prompt-level variance is noise. Saves one reduction, but conflates "hard prompt" with "good response".

**The n=1 special case.** With `n=1` there is no group baseline; you either accept REINFORCE's high variance or bring a critic (PPO's value head; [[trl-ppo]] `AutoModelForCausalLMWithValueHead`). The entire critic-free family (GRPO/RLOO/REINFORCE++) exists to avoid that critic; `n ≥ 4` is the price.

**Knee in the curve.** DeepSeekMath appendix ablations show GRPO pass@1 gain plateaus past `n=16` on MATH but keeps climbing on AIME — harder tasks want more rollouts. Lambert's Interconnects sweep converges on `n=8` as the pareto point.

---

## 7. Putting the corners together — where each stack sits

| Stack | Generator | Freshness | Reuse | Parallelism |
|---|---|---|---|---|
| [[trl-ppo]] (actor-critic) | π_t | k=0 | none | sync |
| TRL GRPO (default) | π_t | k=0 | none | sync + colocate vLLM |
| TRL `GRPOWithReplayBufferTrainer` | π_t | k=0 | **prompt replay** (var-weighted) | sync |
| TRL Online DPO | π_t | k=0 | none | sync |
| [[openrlhf-ppo]] async | π_{t−k} | k∈[1,2], IS-corrected | none | **async queue + partial** |
| [[verl-rollout]] (vllm_async_server) | π_{t−k} | k∈[1,2], IS-corrected, priority | none | **async + priority + partial** |
| DeepSeek-R1 production | π_t | k=0 | none (replay rejected §3.2) | sync, huge batch |
| BoN deployment (Anthropic, Cohere) | π_t | N/A (no gradient) | N/A | trivial parallelism |

The infra track's job over ch-55..ch-60 is to walk each of these corners: ch-55 reads verl end-to-end (ppo_loss, grpo, rollout, entropy logging); ch-56 does the same for OpenRLHF; ch-57 hits distributed training (FSDP, Ray); ch-58 covers KV-cache reuse + speculative decoding in rollout; ch-59 is partial-rollout / continuous-batching RL; ch-60 is cost accounting. ch-54 is the map; ch-55+ are the terrain.

---

## Connections

- **ch-37 / ch-38 ([[dpo]])** — offline-DPO baseline that [[on-off-policy-rlhf]] dissects.
- **ch-39 ([[ppo]])** — the loss that `PolicyLoss.forward` computes; async is the *wrapper*, not the gradient.
- **ch-40 ([[grpo]])** — the group baseline is why `n ≥ 4` matters.
- **ch-41 ([[reward-model-overoptimization]])** — BoN's ceiling is this inverted-U.
- **ch-43 ([[entropy-mechanism-llm-rl]])** — entropy-collapse shows up first as `vllm_kl` spike in async stacks.
- **ch-55** — verl internals (deep dive into [[verl-rollout]] + [[verl-ppo-loss]] + [[verl-grpo]]).
- **ch-56** — OpenRLHF internals (deep dive into [[openrlhf-ppo]], including Ray actor topology).
- **ch-57..ch-60** — distributed training, KV reuse, partial rollout, cost accounting.

## Further reading

- [[best-of-n]] — Stiennon 2020; the BoN KL closed form; first overoptimization curve.
- [[on-off-policy-rlhf]] — Tang 2024; 80/20 decomposition of the DPO-vs-PPO gap.
- [[replay-buffer-rlhf]] — trajectory-replay failure modes; TRL prompt-replay schema.
- [[async-rollout]] — OpenRLHF async + verl HybridFlow; IS correction and `vllm_kl`.
- [[minibatch-sharing-rl]] — RLOO 1/(n−1) variance; `n=8` defaults.
- [[verl-rollout]] — `vllm_async_server.py` line-by-line; engine-pause around line 628.
- [[openrlhf-ppo]] — `PolicyLoss.forward` body; `tis` / `icepop` / `seq-mask-tis`.
- [[policy-coverage-loss]] — coverage as the right lens on "distribution shift".
- [[rejection-sampling-finetuning]] — BoN-in-weights; the SFT-side alternative.

## Companion visualization

**[figures/async-pipeline.html](figures/async-pipeline.html)** — animated async pipeline. Controls: **queue depth** `d` (1–4), **rollout:learner speed ratio** `ρ` (0.5–4), **partial-rollout** toggle, **IS-correction mode** (off / tis / icepop). Readouts: instantaneous throughput (rollouts/s), effective staleness `k`, estimated `vllm_kl` proxy, PPO clipfrac. Start at d=1, ρ=2, partial-off and watch throughput climb as you raise d — then watch clipfrac peg at 1 when `k` exceeds ~3 with IS off. The canary story in prose (§5) becomes kinesthetic here.
