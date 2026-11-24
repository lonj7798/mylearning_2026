<!-- chapter: ch-03
     track: rl-pipeline
     kind: content
     title: Policy Staleness: Definition, Decomposition, and Why It Hurts
     deps: [ch-02]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[areal-async-rl]], [[async-rollout]], [[slime]], [[pipelinerl]], [[areal-agent-pipelines]], [[truncated-importance-sampling]], [[scalerl]], [[slime-agentic-and-mismatch-examples]], [[open-instruct-rlvr-pipeline]]
     figures: figures/staleness-timeline.html
-->

# Chapter 3 — Policy Staleness: Definition, Decomposition, and Why It Hurts

> **Core insight.** Rocket defines staleness per token: "the gap between the policy version used to generate a token and the learner's version when it trains on that token" ([[mai-rl-infrastructure]] p. 5). When learner steps are roughly evenly spaced, this gap splits into two terms: staleness ≈ sampler lag + (t_trained − t_generated) / (time between training steps) (p. 6). The first term is how far the sampler's weights trailed the learner's at generation time, and frequent weight transfer bounds it. The second is wall-clock time between generating a token and training on it, counted in learner steps. In an agentic GRPO run the second term dominates: in the single group traced in Fig. 4, the slowest rollout samples from six weight versions (steps 104–124) across 784 turns, the fastest rollout waits 7.6 h for it, and all three are trained in step 127 (p. 6–7). Staleness matters because the sampler's and learner's token probabilities drift apart as it grows: Fig. 3 fits a line with R² = 0.76 between staleness and mean |log-prob difference|, with an exact match at staleness 0 (p. 5). That figure is a correlation. The causal evidence that staleness degrades learning comes from controlled sweeps in open systems, most directly AReaL's Table 2.
>
> **Guideline.** When rollouts last more than a few learner steps, log staleness per token (or at least the minimum weight version per sample) from the first run, because Fig. 4 shows that one group can span 23 learner steps between its first generated token and its training step (p. 6), and slime's docs state that "a single age statistic is only a diagnostic" for trajectories that contain several versions (slime@79aa41a `docs/en/advanced/rl-systems.md` L41). When the objective is plain PPO, keep maximum staleness at 2 or below, and when you need more asynchrony, use a decoupled objective that separates behaviour and proximal policies, because AReaL measured AIME24 23.3 with naive PPO at η = 4 against 42.2 with the decoupled objective and 42.0 for the synchronous oracle ([[areal-async-rl]] Table 2, 1.5B math, one run per cell). When you bound staleness, bound it finitely, because the same table shows 36.9 at η = ∞ even with the decoupled objective. When you compare staleness across frameworks, convert units first: MAI-Thinking-1's limit of "8 inference updates" equals 40 gradient steps (Sec. 3.1.5), while slime and verl count weight-update versions.

---

## Why this chapter matters

ch-02 described a loop in which no component waits for a global barrier. This chapter puts a number on what that buys and what it costs. Every lever in ch-04 and ch-05 reduces one term of the decomposition below, so you need the decomposition to know which lever to pull. The open frameworks also measure staleness in different units and correct for it at different points (admission, consumption, loss). You need the definitions to compare their settings with each other and with Rocket's.

**Interactive figure.** [figures/staleness-timeline.html](figures/staleness-timeline.html) reproduces the Fig. 4 group on a learner-step axis. A slider picks the generation time of one token in the slowest rollout and shows its sampler lag, its waiting term, and their sum. A second panel converts the fastest and median rollouts' waits into learner steps for an assumed step time. Use it with §2 and §3.

---

## §1 Definition and units

### 1.1 Rocket's definition

"We measure policy staleness as the gap between the policy version used to generate a token and the learner's version when it trains on that token" (p. 5). Three properties follow from the wording.

1. **It is per token.** A multi-turn trajectory generated across several weight refreshes has a different staleness for each turn. In Fig. 4 the slowest rollout "samples from six successive weight versions before it completes" (p. 7).
2. **It is measured at training time.** The same token becomes staler every learner step it waits. Waiting in a queue after completion counts.
3. **Waiting before the first token does not count.** "A long wait before a rollout's first token is generated does not contribute to staleness" (p. 7). No token exists yet, so no token can be stale.

### 1.2 Units differ between systems

Rocket's Fig. 3 x-axis (0–40) and Fig. 4 are in learner steps. Other systems count different things:

| System | Unit | Granularity | Source |
|---|---|---|---|
| Rocket (Base 2) | learner steps | per token | [[mai-rl-infrastructure]] p. 5–6 |
| Rocket v1 (MAI-Thinking-1) | inference updates; 5 gradient steps per update | per rollout (drop rule) | [[mai-thinking-1]] Sec. 3.1.5: "more than 8 inference updates stale (i.e., 40 gradient steps behind)" |
| slime | weight-update versions | per sample: current − min over the sample's versions | slime@79aa41a `slime/utils/staleness.py` L10–17 |
| verl V1 | model versions | per trajectory: span, newest-version gap, oldest-version gap | verl@8357f1e `docs/advance/v1_async_trainer.md` L74–78 |
| AReaL | policy versions | admission bound per sample; optional per-token mask | AReaL@bcef492 `areal/infra/staleness_manager.py` L92–113; `areal/api/cli_args.py` L3312–3319 |
| open-instruct | learner steps (`model_step`) | per result | open-instruct@f020e9a `open_instruct/data_loader.py` L996–1005 |

When weights are synced every k learner steps, one weight version equals k learner steps. Comparing slime's `staleness/max = 2` with Rocket's Fig. 3 value of 2 is therefore wrong unless both sync every step.

slime's per-sample metric shows how a framework reduces per-token staleness to one number:

```python
# slime@79aa41a slime/utils/staleness.py L10–17
def sample_staleness(sample: Sample, current_weight_version: int) -> int | None:
    versions = sample.weight_versions or []
    if not versions or any(not str(v).isascii() or not str(v).removeprefix("-").isdigit() for v in versions):
        return None
    numeric_versions = [int(v) for v in versions]
    if max(numeric_versions) > current_weight_version:
        return None
    return current_weight_version - min(numeric_versions)
```

The sample's staleness is the age of its oldest version, so it is the worst token's staleness. A mean over tokens would be lower. slime's docs warn: "A long trajectory may contain several versions; a single age statistic is only a diagnostic" (`docs/en/advanced/rl-systems.md` L41). verl logs three statistics for this reason: the number of versions inside one trajectory (`trajectory_spans`), and the gap from the current version to the newest (`trajectory_staleness`) and to the oldest (`trajectory_staleness_worst`) version used (verl `v1_async_trainer.md` L74–78).

---

## §2 The decomposition

### 2.1 The formula

The post states: "When the time between training steps is roughly constant, a token's staleness is approximately"

```
staleness ≈ sampler lag + (time when token is trained on − time when token was generated) / (time between training steps)
```

(p. 6). The terms are defined in the next paragraph:

- **Sampler lag** "measures how far the sampler's policy version lags the learner's when a token is generated".
- The **second term** "captures staleness accumulated between generation and training: while the token's rollout finishes, while the completed rollout waits for the rest of its GRPO group, and while the group waits for the learner" (p. 6).

### 2.2 Why it holds

Let v be the sampler's weight version when a token is generated, s the learner's step at that moment, and T the learner step that trains on the token. All three are in learner-step units. By definition staleness = T − v. Add and subtract s:

```
T − v = (s − v) + (T − s)
```

The first bracket is the sampler lag. The second is the number of learner steps that pass between generation and training. If steps take a constant time Δt, then T − s = (t_trained − t_generated) / Δt. The identity is exact in step units. Only the conversion of the second term to wall-clock time needs the constant-Δt assumption. When Δt varies (for example, steps get slower as responses get longer), use step counts directly.

### 2.3 What each term depends on

| Term | Set by | Bounded by | Lever (chapter) |
|---|---|---|---|
| Sampler lag | sync cadence k and transfer time | 0 … k − 1 steps if transfer is fast | frequent weight transfer (ch-05) |
| Rollout still running | number of turns, tool latency, inference queueing | rollout timeout | priority to in-flight rollouts (ch-04) |
| Waiting for the group | slowest sibling; retries | drop failed rollouts (ch-04) | dropping failures (ch-04) |
| Waiting for the learner | controller queue length | learner capacity | inference-bound sizing (ch-04) |
| Denominator Δt | batch size / inference throughput when inference-bound | learner step time | more inference at fixed batch size *raises* staleness (ch-04) |

The last row is the counter-intuitive one, and ch-04 derives it.

---

## §3 Fig. 4: one GRPO group, term by term

Fig. 4 (p. 6) follows three rollouts of one GRPO group "from dispatch to the training step that consumes them", "colored by the version of the model weights that inference servers sampled from at each turn". Values from the figure text layer and the prose on p. 6–7:

| Item | Value | Location |
|---|---|---|
| Weight versions used by the slowest rollout | learner steps 104, 108, 112, 116, 120, 124 ("six successive weight versions") | Fig. 4 header; p. 7 |
| Refresh cadence | "inference servers get new weights every 4 steps"; "sampler lag is 0–3 steps" | Fig. 4 axis; p. 6 |
| Wait before the first inference request | "Each rollout in Figure 4 waits 24 minutes" | p. 7 |
| Slowest rollout length | 784 turns of "model inference, tool calls and waits" | Fig. 4 |
| Fastest rollout's wait for the slowest | 7.6 h | Fig. 4; p. 7 |
| Median rollout's wait for the slowest | 5.6 h | Fig. 4 |
| Controller queue after the group completes | 12 s | Fig. 4; p. 7 |
| Training | "all trained together in step 127 · 3 min" | Fig. 4 |
| x-axis | 0–12 hours since dispatch | Fig. 4 |

**Card note.** The library card [[mai-rl-infrastructure]] summarizes Fig. 4 as "slowest rollout queued 24 min on step-104 weights". The prose on p. 7 says each of the three rollouts waits 24 minutes before its first inference request. This chapter uses the prose.

### 3.1 The slowest rollout

The slowest rollout's first tokens are generated with step-104 weights, and it is trained in step 127. Counting the training step as the learner's version, its first tokens have staleness 127 − 104 = 23 learner steps. The post does not say whether the learner's version during step 127 is counted as 126 or 127, so the value is 22 or 23 depending on that convention.

Split by §2.2: suppose a token is generated while the learner is at step 105 and the sampler still serves 104. Its sampler lag is 1 and its waiting term is 127 − 105 = 22. A token generated at learner step 107 has lag 3 and waiting term 20. The sum stays 23 for every token generated with step-104 weights. The decomposition does not change the total. It tells you which part a lever can reach. Faster sync can only remove the 0–3 lag. The other 20–23 steps come from the rollout's own duration and the group wait.

Tokens generated with step-124 weights, near the end of the rollout, have staleness 3. Within one trajectory, staleness therefore ranges from about 3 to 23. PipelineRL describes the same structure for in-flight updates: earlier tokens in a long sequence have higher lag ([[pipelinerl]] Fig. 3a).

### 3.2 The fastest and median rollouts

The prose states that all three rollouts wait 24 minutes before their first request, and they belong to one group dispatched together. Their first tokens were therefore generated around the same time as the slowest rollout's, presumably with the same step-104 weights. This is a deduction. The text layer does not label the versions of the fastest and median bars.

Their extra staleness comes from waiting. The fastest rollout's tokens accumulate 7.6 h of waiting for the group. Converting to steps needs Δt, which the post does not state. A bound follows from the axis: at least the 19 step intervals from step 108 to step 127 fit inside the 12-hour axis, after the 24-minute wait, so Δt ≤ 11.6 h / 19 ≈ 37 min. Then 7.6 h corresponds to at least 7.6 × 60 / 37 ≈ 12 learner steps of staleness accumulated purely by waiting for a sibling. These numbers are derived from the axis range, not stated by the post.

### 3.3 What the group spends where

| Phase | Staleness effect | Value in Fig. 4 |
|---|---|---|
| Queue before first token | none | 24 min |
| Generation across refreshes | sampler lag 0–3 per token, plus aging | 784 turns, six versions |
| Fast rollouts waiting for the slowest | aging only | 7.6 h (fastest), 5.6 h (median) |
| Grading | aging only | shown, duration not stated |
| Controller queue | aging only | 12 s |
| Training step | — | 3 min |

The controller queue (12 s) is negligible next to the group wait (hours). This is the outcome Rocket designs for by keeping the job inference-bound (ch-04).

---

## §4 Why staleness hurts

### 4.1 What Rocket shows: Fig. 3

Fig. 3 (p. 5) plots, for one point per rollout, the mean absolute difference between sampler and learner log-probabilities (y-axis 0–0.05) against staleness (x-axis 0–40). A linear fit has R² = 0.76. At staleness 0 the figure is annotated "Same weights: exact match". The caption: "Higher staleness is correlated with larger differences between learner and sampler token probabilities." The text: "When staleness is zero, there is no difference between token probabilities in the sampler and learner (more details in the Numerical parity section below)" (p. 5).

Three readings:

1. **Under parity, the gap is pure staleness.** slime's docs split the train–rollout log-prob difference into "parameter staleness" (log p_{v_t} − log p_{v_s}) and "same-version engine mismatch" (log p_{v_s} − log q_{v_s}), and note that deterministic kernels "can reduce the second term, not the first" (slime `rl-systems.md` L33–41). Rocket's bitwise parity (ch-07) makes the second term zero, so the exact match at 0 is expected, and every nonzero value in Fig. 3 is weight drift. In a stack without parity, the curve would start above zero. verl's MXFP8 doc reports `rollout_corr/kl` of 0.0006–0.0008 for a BF16 rollout at synchronous training (verl@8357f1e `docs/low_precision/mxfp8.md` L85–86).
2. **It is a correlation across rollouts.** Rollouts with high staleness are also the long agentic ones (Fig. 2). They differ from short rollouts in task type, length and token distribution, not only in weight age. The slope therefore mixes weight drift with task composition. The post does not separate the two.
3. **It shows a symptom, not damage.** A larger log-prob difference means a larger importance ratio in an off-policy objective. Whether that harms learning depends on the objective. The post's stated mechanism is qualitative: off-policy training "increases the risk of instability due to token probability and activation differences between the two policies" (p. 5).

### 4.2 Controlled evidence from open systems

| Study | Setup | Result | Type |
|---|---|---|---|
| [[areal-async-rl]] Table 2 | 1.5B math, 250 steps; η = max staleness | AIME24 naive PPO / decoupled: η = 1 → 41.8 / 42.1; η = 2 → 40.0 / 41.8; η = 4 → 23.3 / 42.2; η = 8 → 35.7 / 41.0; η = 16 → 35.8 / 38.7; η = ∞ → 34.0 / 36.9; oracle (η = 0) 42.0 | controlled sweep, one run per cell |
| [[areal-async-rl]] Fig. 5c | same | effective throughput 128.7k tok/s (η = 0) → 371.7k (η = 8) → 396.8k (η = ∞) | throughput side of the trade |
| [[scalerl]] Fig. 4b | 8B, PipelineRL | max off-policyness k = 8 best among tested values | controlled, fitted curves |
| [[pipelinerl]] Fig. 6 | Qwen2.5-7B math | some tokens lag by more than 50K samples, yet ESS similar to G = 8 conventional; ESS for G = 16 and 32 drops | controlled |
| [[mai-thinking-1]] Sec. 3.1.5 | Rocket v1 production | drop rollouts more than 8 inference updates (40 steps) stale; "slightly relaxes on-policy freshness in exchange for substantially higher throughput"; LR 9e−7 at higher lengths "to reduce off-policiness" | practice, no ablation reported |

Two observations from AReaL's table:

- **The damage is non-monotone in η for naive PPO** (23.3 at η = 4, 35.7 at η = 8). Single runs per cell cannot separate a real effect from run-to-run variance. The paper reports no seed count ([[areal-async-rl]] Verification).
- **The decoupled objective moves the tolerance, it does not remove it.** At η = ∞ it scores 36.9 against 42.0. Staleness is still a cost after correction.

All of these runs are math or code on 1.5B–8B models with rollouts up to 32K tokens. None tests hour-long agentic rollouts. The evidence transfers in direction, not in threshold values.

---

## §5 How open frameworks bound and correct staleness

Corrections act at four points. A run can combine several.

### 5.1 At admission: do not start work that will be too stale

AReaL admits a new rollout only while the number of accepted plus running samples stays below what the next η versions can consume:

```python
# AReaL@bcef492 areal/infra/staleness_manager.py L99–113
with self.lock:
    current_version = self.version_provider.get_version()
    # Calculate concurrency-based capacity
    max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
    concurrency_capacity = max_concurrent_rollouts - self.rollout_stat.running

    # Calculate staleness-based capacity
    ofp = self.max_staleness
    sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
    consumer_bs = max(1, self.consumer_batch_size)
    staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

    # Return the minimum of both constraints
    capacity = min(concurrency_capacity, staleness_capacity)
    return capacity
```

With current version i and batch size B, at most (η + i + 1) · B samples may exist. Any admitted sample is consumed within η versions of its admission. The bound counts samples, not wall-clock time. A rollout that runs for many steps can still exceed η if it is admitted early and finishes late, because the formula assumes samples are consumed roughly in admission order. AReaL's search-agent config uses `max_head_offpolicyness: 4` with `max_concurrent_rollouts: 160` ([[areal-agent-pipelines]] Pipeline step 2). The default is 0, which is synchronous (`cli_args.py` L3304–3311).

### 5.2 At consumption: drop what is too stale

open-instruct drops results whose generating step is too far behind and replaces the prompt:

```python
# open-instruct@f020e9a open_instruct/data_loader.py L996–1005
def result_is_stale(model_step: int | None, training_step: int | None, max_result_age_steps: int | None) -> bool:
    """Whether an async rollout result is too far behind the current training step.

    Returns False (never stale) when staleness checking is disabled or the inputs are
    unavailable. A result is stale when ``training_step - model_step`` exceeds
    ``max_result_age_steps``.
    """
    if max_result_age_steps is None or training_step is None or model_step is None:
        return False
    return training_step - model_step > max_result_age_steps
```

The caller drops the result and pushes a fresh prompt (`data_loader.py` L1086–1105). verl V1 offers the choice between dropping and waiting: `max_off_policy_threshold` (default 8 model versions, "from first generation to being trained") with `max_off_policy_strategy` `drop` (evict the stale prompt group, "for GRPO, one stale trajectory, the whole sample dropped") or `wait` (block until threshold-reaching in-flight groups finish) (verl `v1_async_trainer.md` L67–71). MAI-Thinking-1 used the drop rule at 40 gradient steps. The Base 2 post lists "policy-staleness limits" among Rocket's runtime-configurable settings (p. 13) without giving values.

Dropping at consumption wastes the generation compute already spent. Bounding at admission avoids that waste, but it can idle the sampler. Waiting keeps the data, but it stalls the learner.

### 5.3 In the loss: correct the ratio

If stale data is trained on, the objective has to account for the behaviour policy that generated it.

- **Decoupled PPO (AReaL).** The clipped ratio is π_θ / π_prox, where π_prox is the policy immediately before the current update, and an outer weight π_prox / π_behav corrects for the behaviour policy ([[areal-async-rl]] Technical Details 3). The implementation applies the clip to the proximal ratio and then multiplies by the behaviour weight:

```python
# AReaL@bcef492 areal/utils/functional/functional.py L556–579
clipped_ratio = torch.clamp(
    ratio,
    1.0 - eps_clip,
    1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
)

pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * clipped_ratio
clip_mask = pg_loss1.detach() < pg_loss2.detach()
pg_loss = torch.max(pg_loss1, pg_loss2)
if c_clip is not None:
    assert c_clip > 1.0, c_clip
    pg_loss3 = torch.sign(advantages) * c_clip * advantages
    dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
    pg_loss = torch.min(pg_loss, pg_loss3)
else:
    dual_clip_mask = torch.zeros_like(clip_mask)

# Apply behavioural importance weight from rejection sampling
if rejection_sampling is not None:
    behave_approx_kl = proximal_logprobs.detach() - old_logprobs.detach()
    behave_mask = (behave_imp_weight > 0).logical_and(loss_mask.bool())
    behave_approx_kl = torch.where(behave_mask, behave_approx_kl, 0.0)
    pg_loss = pg_loss * behave_imp_weight
```

  Here `ratio` is exp(logprobs − proximal_logprobs) (L546–549), and `behave_imp_weight` is π_prox / π_behav, clamped or masked by `apply_rejection_sampling` (L189–441). The trust region is centred on a recent policy, not on the stale one. This is why the decoupled column of Table 2 tolerates η = 8.
- **Truncated importance sampling.** Multiply the PPO gradient by min(π_learner(θ_old) / π_sampler(θ_old), C), outside the clip ([[truncated-importance-sampling]] Key Contributions). It was introduced for same-version engine mismatch, but the same factor absorbs staleness when the sampler log-probs come from an older version. open-instruct clamps ρ at 2.0 by default (`grpo_utils.py` L129–137), and PipelineRL clamps importance weights at 5 ([[pipelinerl]] Setup).
- **Masking stale tokens.** AReaL's optional `mask_stale_tokens` "mask[s] actor loss targets whose token policy-version age exceeds `max_head_offpolicyness`" (`cli_args.py` L3312–3319), passed through to `ppo_update` (`areal/trainer/rl_trainer.py` L1172–1177). slime's `--mask-offpolicy-in-partial-rollout` zeroes the loss on a continued prefix ([[slime-agentic-and-mismatch-examples]] Fully async staleness).

What Rocket does in the loss is not stated in the Base 2 post. MAI-Thinking-1 describes an importance-weighted GRPO objective with an outer ratio clip, clip(r, r_min, r_max) with r_min = 0 and r_max = 50 on all branches ([[mai-thinking-1]] Sec. 3.1.1, 3.1.5), and states that log-prob discrepancies "can destabilize the importance-sampling correction in off-policy RL" (Sec. 3.6.4).

### 5.4 Mixed-version trajectories

A trajectory generated across refreshes has a behaviour distribution that is a product of per-version factors. slime's docs write it for one switch at token k:

```
q(y | x) = Π_{t ≤ k} q_{v0}(y_t | x, y_<t) · Π_{t > k} q_{v1}(y_t | x, y_<t)
```

and conclude that "a token-level importance ratio uses p_θ(y_t | h_t) / q_{v(t)}(y_t | h_t); it cannot substitute one current sampler for the entire mixed-version response" (slime@79aa41a `docs/en/advanced/rollout-scheduling.md` L34–41). This is why the sampler must return a version with every token (ch-02) and why the learner must read the sampler's stored log-probs rather than recompute them under one version. Rocket's Fig. 1 lists "versions" next to tokens and log-probs for this reason, and its rollout store holds "sampling log probabilities" (p. 3).

AReaL's Proposition 1 states the complementary result: a trajectory made of segments from several policy versions is equivalent to a sample from a single behaviour policy, so the decoupled objective applies to it unchanged ([[areal-async-rl]] Key Contributions).

---

## Recipe

1. **Tag every generated token with its weight version** at the sampler, and store it with the token IDs and log-probs.
2. **Log three staleness statistics per batch:** mean token staleness, worst token staleness (slime-style min version), and versions per trajectory (verl `trajectory_spans`).
3. **Log the decomposition.** For each training step, record mean sampler lag at generation, mean time from rollout completion to group completion, and mean controller-queue time. This tells you which lever of ch-04 or ch-05 to use.
4. **State units.** Record the sync cadence k so that version-based and step-based numbers can be converted.
5. **Choose the bound location.** Admission bound (AReaL) if wasted generation is expensive (long agentic rollouts). Consumption drop (open-instruct, verl `drop`, MAI-Thinking-1) if the bound must be strict. `wait` if data is scarce and learner idle time is acceptable.
6. **Pair any bound > 2 with a decoupled or truncated-IS objective,** per AReaL Table 2, and keep the bound finite.
7. **Compute ratios per token with the generating version's log-prob,** never with one recomputed sampler distribution for a mixed-version trajectory.
8. **Plot your own Fig. 3** (mean |Δ log-prob| against staleness) early. Its intercept at staleness 0 measures same-version engine mismatch (ch-07), and its slope measures weight drift per step.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Counting queue time before the first token as staleness | Timeouts or admission tuned against a wrong number | Rocket p. 7: pre-first-token wait adds no staleness; start the clock at the first sampled token |
| Comparing staleness across frameworks without converting units | A "staleness 8" setting copied from verl (versions) into a step-based system | Multiply versions by the sync cadence k (MAI-Thinking-1: 8 updates = 40 steps) |
| Using one age per trajectory as if it were per token | Mean staleness looks small while early tokens are very stale | Log worst-token and span statistics (slime `staleness/max`, verl `trajectory_staleness_worst`) |
| Recomputing behaviour log-probs under the current sampler for a mixed-version trajectory | Importance ratios wrong for early turns | Compare stored per-token sampler log-probs with recomputed ones by version |
| Reading Fig. 3's slope as the causal effect of staleness on learning | Over- or under-investing in staleness control | Fig. 3 is correlational and mixes task type; use controlled sweeps (AReaL Table 2) |
| Raising η with naive PPO | Benchmark drop with no training-loss alarm (AReaL η = 4: 23.3 vs 42.0) | Track held-out accuracy across η; switch to a decoupled objective |
| Unbounded staleness with a correction | Slow degradation (AReaL η = ∞: 36.9 vs 42.0) | Keep a finite bound and log the fraction of data dropped or masked |

---

## Check your understanding

1. Rocket says that waiting before a rollout's first token adds no staleness. Using the definition, explain why, and explain why the same wait still matters for rollout timeouts and resource holding.
2. Derive the decomposition T − v = (s − v) + (T − s) and explain which part needs the constant-step-time assumption. What happens to the wall-clock form of the formula when step time grows during training?
3. In Fig. 4, every token the slowest rollout generated with step-104 weights has total staleness 23 regardless of when it was generated within steps 104–107. Explain why, and why the decomposition is still useful if the total does not change.
4. Fig. 3 shows an exact match at staleness 0. Explain why this would not be true in a stack without bitwise parity, and what the intercept of such a curve would measure.
5. Fig. 3's fit is across rollouts of different task types. Describe a confound in reading its slope as "log-prob drift per learner step", and design a measurement that removes it.
6. AReaL's decoupled objective scores 42.2 at η = 4 where naive PPO scores 23.3. Explain, from where the trust-region centre sits in each objective, why the decoupled form tolerates more staleness.
7. Why must a mixed-version trajectory use per-token behaviour log-probabilities from the generating version? What would a single recomputed behaviour distribution get wrong?

---

## Connections

- **Previous: ch-02** — the loop whose waits produce each staleness term.
- **Next: ch-04** — the levers on the waiting terms: priority, dropping failures, inference-bound sizing, and the effect of adding inference at fixed batch size.
- **ch-05** — weight transfer, the lever on sampler lag.
- **ch-07** — bitwise parity, which makes Fig. 3's intercept zero.
- **ch-08** — agentic rollouts, the reason the second term reaches hours.
- **ch-18** — choosing an algorithm by domain, including how much off-policyness each objective tolerates.
- **llm-training ch-40** — GRPO and the clipped objective that the decoupled form modifies.
- **llm-training ch-43** — entropy and KL control; staleness interacts with both through the importance ratio.
- **llm-training ch-54** — earlier treatment of off-policy data in async RL.

---

## Sources

- [[mai-rl-infrastructure]] — definition (p. 5), formula and term definitions (p. 6), Fig. 3 (R² = 0.76), Fig. 4 group timeline, 24-min pre-first-token wait (p. 7).
- [[mai-thinking-1]] — Rocket v1 staleness limit (8 updates = 40 steps), outer ratio clip, LR reduction to limit off-policiness.
- [[areal-async-rl]] — η admission rule, decoupled objective, Table 2 staleness sweep, Fig. 5c throughput.
- [[areal-agent-pipelines]] — `StalenessManager` code and `mask_stale_tokens`; AReaL@bcef492.
- [[slime]] — staleness metric code, parameter-staleness vs engine-mismatch split; slime@79aa41a.
- [[slime-agentic-and-mismatch-examples]] — fully async staleness logging and `--mask-offpolicy-in-partial-rollout`.
- [[open-instruct-rlvr-pipeline]] — `result_is_stale` and ρ clamp; open-instruct@f020e9a.
- [[pipelinerl]] — lag structure inside long sequences, ESS, clamped weights at 5.
- [[scalerl]] — max off-policyness k = 8 best among tested values.
- [[truncated-importance-sampling]] — truncated ratio outside the PPO clip.
- [[async-rollout]] — OpenRLHF's queue-bounded async trainer, which has no policy-version field (comparison).
