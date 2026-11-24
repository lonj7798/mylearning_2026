<!-- chapter: ch-10
     track: rl-pipeline
     kind: content
     title: Designing the Rollout: Sampling, Group Size, Length, and Multi-Turn
     deps: [ch-09]
     sources: [[rl-design-guide]], [[scalerl]], [[pipelinerl]], [[polaris]], [[skywork-or1]], [[brorl]], [[dapo]], [[kimi-k1-5-recipe]], [[async-rollout]], [[verl-rollout]], [[slime]], [[tito-token-in-token-out]], [[mai-thinking-1]], [[mai-rl-infrastructure]], [[olmo-3-rl-pipeline]], [[rstar2-agent]], [[fp16-train-infer-mismatch]], [[verl-rollout-correction-and-lowprec]], [[deepseek-train-infer-consistency]], [[areal-agent-pipelines]], [[slime-agentic-and-mismatch-examples]], [[recipe-crosswalk]], [[deepmath-103k]], [[scaling-reasoning-losing-control-mathif]]
     figures: figures/rollout-design.html
-->

# Chapter 10 — Designing the Rollout: Sampling, Group Size, Length, and Multi-Turn

> **Core insight.** Rollout settings decide which samples carry gradient and how off-policy they are, before the loss function sees them. Temperature and top-p set entropy, and therefore how many groups contain both successes and failures; the group size n sets the probability that a group has zero reward variance (p^n + (1 − p)^n for a binary reward with pass rate p) and the variance of the group baseline; the maximum length and the treatment of truncated responses decide whether long responses are pushed toward completion or toward silence; oversampling and dynamic sampling decide what fraction of generated rollouts is discarded; partial rollouts and asynchronous scheduling trade throughput for staleness; and in multi-turn rollouts, the way tokens are recorded decides whether the learner trains on what the sampler produced. The measured evidence is unusually concrete here, but much of it is single-model and single-seed, and early reward is a poor guide: ScaleRL found that smaller batches and shorter generations looked better early and reached lower asymptotes ([[scalerl]] Figs. 9–10).
>
> **Guideline.** Compare rollout settings by fitting held-out pass rate against compute and comparing the fitted asymptote, because ScaleRL's run-to-run error on the asymptote is about ±0.02 and early curves reversed ([[scalerl]] Fig. 8a, Figs. 9–10). Set temperature per model by sweeping it and taking the highest value before accuracy drops, because Qwen3-4B was stable from T = 0.6 to 1.4 and collapsed above 1.55 ([[polaris]]), while from a distilled 7B model T = 0.6 collapsed entropy within about 100 steps and T = 1.0 did not ([[skywork-or1]] Exp. 4). If top-p < 1, replay the sampler's top-p mask in training, because MAI-Thinking-1 saw "divergence within a few training steps" without it ([[mai-thinking-1]] Sec. 3.1.3). At fixed total batch, do not spend tuning effort on 8 versus 32 rollouts per prompt (curves "essentially unchanged", [[scalerl]] App. A.13); when a long run plateaus, test a much larger n, because BroRL's N = 512 beat N = 16 at equal compute (math average 63.03 vs 62.02) and doubled throughput from 36.5 to 72.4 samples/s ([[brorl]] Tables 2–3). Penalize length inside a buffer before the hard limit rather than giving truncated responses the full failure reward ([[dapo]] Table 1: +3 for soft overlong punishment), but check masking choices against your own run, because Skywork-OR1 and Olmo 3 RL-Zero found that masking truncated responses hurt ([[skywork-or1]] Exp. 3; [[olmo-3-rl-pipeline]] §6). In multi-turn rollouts, train on the engine's token ids and log-probs and tokenize tool and user turns as deltas with loss mask 0 ([[tito-token-in-token-out]]).

---

## Why this chapter matters

The rollout is where most of the compute goes and where the data distribution of RL is created. Olmo 3's 32B Think run used 8 learner nodes and 20 inference nodes, and the learner spent 75% of its time waiting for data; a representative step took 1,000 s of which 125 s were training ([[olmo-3-rl-pipeline]] §4.4.3). Every rollout setting in this chapter changes both that cost and what the learner receives.

This chapter follows [[rl-design-guide]] §2 and cites the underlying cards. It assumes the GRPO/DAPO mechanics of [llm-training ch-40](../../llm-training/ch-40/read.md) and the entropy material of [llm-training ch-43](../../llm-training/ch-43/read.md). The interactive figure [rollout-design.html](figures/rollout-design.html) accompanies §2–§4.

---

## §1 How to compare rollout settings

ScaleRL fits held-out pass rate R_C against RL compute C with a saturating curve, R_C − R_0 = (A − R_0) / (1 + (C_mid / C)^B), where A is the asymptote, B the efficiency exponent and C_mid the compute at half the gain ([[scalerl]] Core Insight). It classifies each design choice by whether it changes A or only B:

| Choice | Effect | Location |
|---|---|---|
| PipelineRL vs PPO-off-policy-k | similar A, higher B | §3.1, Fig. 4a |
| Max off-policyness k | k = 8 best | Fig. 4b |
| FP32 LM-head logits | A 0.52 → 0.61 | Fig. 5b |
| Zero-variance filtering | higher A | Fig. 6a |
| No-Positive-Resampling (drop pass ≥ 0.9) | higher A | Fig. 6b |
| Generation length 14K → 32K | lower B, higher C_mid, higher A | Fig. 9 |
| Batch of 2,048 prompts vs smaller | higher A | Fig. 10 |
| 8–32 generations per prompt at fixed total batch | essentially unchanged | App. A.13 |

The method matters more than any single row: the authors fitted after about 1.5K GPU-hours and found run-to-run error on A of about ±0.02 (Fig. 8a). Differences smaller than that are not evidence. A power-law fit to 1.5K–50K GPU-hours predicted A = 1.0, while the sigmoid gave 0.645 against an observed saturation near 0.65 (Summary), so the choice of fit family also matters. All ScaleRL results come from an 8B dense model and a Llama-4 Scout 17B×16 MoE.

---

## §2 Temperature, top-p and entropy

**Per-model temperature.** POLARIS swept rollout temperature for Qwen3-4B and found three zones: stable AIME24 accuracy from 0.6 to 1.4 ("Robust Generation"), 1.4–1.55 ("Controlled Exploration"), and collapse from 1.55; it started RL at T = 1.4, and at 0.7 for DeepSeek-R1-Distill-Qwen-7B, and raised the temperature at each stage: 7B 0.7 → 1.0 → 1.1; 4B 1.4 → 1.45 → 1.5 ([[polaris]] Temperature). It measures diversity by the distinct 4-gram ratio and reports that the common defaults 0.6 and 1.0 degraded performance for its models. These are single runs.

**Temperature and entropy collapse.** Skywork-OR1 chose T = 1.0 over 0.6 for distilled R1 models; at 0.6, entropy "starts very low or collapses within about 100 steps" and learning stalled ([[skywork-or1]] Exp. 4). Its §4 study found that more SGD steps per rollout batch (more off-policy updates) accelerated entropy collapse and lowered test scores. The causal direction is supported by intervention (temperature and update count were changed), but on one model family.

**Top-p and mask replay.** MAI-Thinking-1 samples with top-p = 0.97 and replays the sampler's top-p truncation mask in training by setting excluded logits to −∞ before the softmax. Without it, "continuing to backpropagate through logits corresponding to tokens outside the sampled nucleus can lead to catastrophic off-policy mismatch, causing divergence within a few training steps" ([[mai-thinking-1]] Sec. 3.1.3, read from the PDF). The cost is storage and transfer of the mask; the hyperparameter table notes that a larger p raises mask-transfer overhead (Sec. 3.1.5). The report also states that top-p sampling was "similarly effective" as its language-consistency reward at preventing single low-probability foreign-language tokens (Sec. 3.1.2). Without mask replay, a top-p sampler and a full-softmax learner are two different policies, which is the ch-07 problem in a different form.

**Framework defaults.** verl: `temperature: 1.0`, `top_p: 1` (`verl/trainer/config/rollout/rollout.yaml` L17, L23 at 8357f1e). slime's Qwen3-4B example: temperature 1 ([[slime]] Recipe ledger); its coding-agent example: temperature 1, top-p 0.95 ([[slime-agentic-and-mismatch-examples]]). Olmo 3: temperature 1.0 in all four runs ([[olmo-3-rl-pipeline]] Table 49).

**Controlling entropy without changing temperature.** MAI-Thinking-1 controls entropy through an integral controller on the upper clip bound toward a target entropy H* = 0.3 instead of an entropy bonus; Skywork-OR1 switches on an entropy loss only when entropy falls below 0.2. These are loss-side controls covered in [llm-training ch-43](../../llm-training/ch-43/read.md); they interact with rollout temperature because both change the entropy the next rollouts are sampled at.

---

## §3 Group size n

**What n controls.** With binary reward and pass rate p for a prompt, the probability that all n rollouts get the same reward is p^n + (1 − p)^n. Such a group has zero advantage under any group baseline and contributes no gradient ([llm-training ch-40](../../llm-training/ch-40/read.md) §5). Larger n lowers that probability for prompts with p near 0 or 1 and lowers the variance of the baseline estimate. The figure's first panel computes the zero-variance probability and the expected number of prompts needed per informative group for any n and pass-rate distribution.

**Measured effects.**
- ScaleRL: at fixed total batch, 8, 16 and 32 generations per prompt gave essentially unchanged fitted curves (App. A.13).
- BroRL: from a ProRLv2 1.5B checkpoint already trained for 3,000 steps, N = 512 with 128 prompts per step was compared with N = 16 with 512 prompts per step at equal compute ([[brorl]] §4.2, Tables 3–5):

| Method | N | Extra steps | Math avg | Code avg | Reasoning Gym avg |
|---|---|---|---|---|---|
| Baseline | 16 | 0 | 61.69 | 52.00 | 61.29 |
| ProRL | 16 | +535 | 62.02 | 52.74 | 61.45 |
| BroRL | 512 | +191 | 63.03 | 54.20 | 63.09 |

The dynamic-sampling pass rate (share of groups with non-zero variance) was 41% at N = 16 and 62% at N = 512, and generation throughput was 36.5 vs 72.4 samples/s (Table 2), which the authors attribute to generation moving from memory-bound toward compute-bound and to more prefix-cache hits. The paired per-instance gain is small (Δ = 0.0033, t = 4.84, §4.2.2), and only N ∈ {16, 512} were tested.

The two studies do not conflict: ScaleRL varied n in the 8–32 range from a fresh run; BroRL varied it by 32× from a plateaued run. Neither measures the region between.

**Values used in practice.** MAI-Thinking-1: G = 128 with an early-exit group of G_early = 16 ([[mai-thinking-1]] Sec. 3.1.5). DAPO: 16 ([[dapo]] §4.1). DeepMath: 16 ([[deepmath-103k]] Table 6). Olmo 3: 8 ([[olmo-3-rl-pipeline]] Table 49). POLARIS: 8. slime coding agent: 8 trajectories per group. AReaL SWE example: `n_samples: 4` ([[areal-agent-pipelines]]). Larger groups appear where rollouts are cheap relative to the value of a correct sample (MAI STEM); smaller groups where each rollout costs an hour of sandbox time.

**Group size and staleness.** In asynchronous RL the group trains only when its slowest member finishes. Rocket's Fig. 4 group waited 7.6 hours between the fastest and slowest rollout ([[mai-rl-infrastructure]] p. 6). Each extra rollout in a group raises the expected maximum duration of the group, so larger n raises staleness for long-tailed tasks. No source measures this trade-off directly.

---

## §4 Maximum response length and overlong handling

**The decision.** A response that hits the length limit has no final answer. Giving it the failure reward pushes the policy away from long responses even when the reasoning was correct so far; masking it from the loss avoids that push but also removes the signal that the response was too long.

**DAPO's two measured steps** ([[dapo]] Table 1, cumulative, single seed, Qwen2.5-32B): overlong filtering (mask truncated samples) raised AIME24 avg@32 from 30 to 36; soft overlong punishment later raised it from 38 to 41. The soft punishment is (Eq. 13): R_length(y) = 0 if |y| ≤ L_max − L_cache; ((L_max − L_cache) − |y|)/L_cache if L_max − L_cache < |y| ≤ L_max; −1 if |y| > L_max. DAPO used 16,384 expected tokens plus a 4,096-token buffer, 20,480 total (§4.1). verl's DAPO reward manager implements the same buffer with a penalty factor:

```python
# verl@8357f1e verl/workers/reward_manager/dapo.py L121-127
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
```

**Evidence against masking.** Skywork-OR1 started Stage I with about 40% of responses truncated and compared no advantage mask with two masking variants; masking raised accuracy on non-truncated responses but lowered overall accuracy and clip ratio and gave no test-time scaling benefit, and the final recipe uses no mask ([[skywork-or1]] Exp. 3). Olmo 3 RL-Zero 3.1 improved partly by *not* masking truncated sequences, "because masking varied batch size and reduced stability" ([[olmo-3-rl-pipeline]] §6). These conflict with DAPO's overlong filtering result. The setups differ (DAPO from a base model at 20K; Skywork from distilled models with multi-stage context; Olmo RL-Zero from a base model at 16K), so the evidence supports "test both" rather than a rule.

**Length schedules.** Three measured patterns:
- Start at the maximum: Qwen3-4B trained at 24K "caused a sharp irreversible drop", while starting at 40K improved steadily ([[polaris]] Length).
- Stage up: Skywork-OR1 switched from 8K to 16K at step 540 and reached the same AIME24 accuracy (about 60) as training at the long context throughout, saving about 100 training hours over 1,000 steps ([[skywork-or1]] Exp. 2).
- Double from short: MAI-Thinking-1 caps rollouts at 8k tokens early and doubles in powers of two to 128k, reporting that this "significantly" reduced inference cost in the low-performance regime (Sec. 3.1.3; no number).

The model's starting state appears to decide which works: POLARIS's 4B model already produced long responses; Skywork's 7B distilled model and MAI's from-scratch climb did not need long budgets early. ScaleRL measured that 32K rather than 14K generation raised the asymptote at the cost of efficiency (Fig. 9).

**Length penalties tied to difficulty.** MAI-Thinking-1 uses R_len = ρ_q · |y| / ℓ_max with ρ_q the problem's pass rate, so easy problems are penalized more for length; w_len = 0.25 up to 64k and 0 at 128k ([[mai-thinking-1]] Sec. 3.1.2, 3.1.5). Kimi k1.5's length reward λ = 0.5 − (len − min_len)/(max_len − min_len), given as λ to correct responses and min(0, λ) to incorrect ones, was off at first and constant afterwards; the authors state it can slow early training ([[kimi-k1-5-recipe]] §2.3.3). ch-12 treats length terms as reward components.

**Length versus other capabilities.** Raising the RL maximum rollout length from 1k to 8k tokens on DeepSeek-R1-Distill-Qwen-1.5B raised math accuracy from 28.73 to 39.82 while lowering hard instruction-following accuracy from 19.05 to 14.29 ([[scaling-reasoning-losing-control-mathif]] Table 5). Track an instruction-following metric when lengths grow.

The figure's second panel plots DAPO's soft punishment and MAI's difficulty-scaled penalty for adjustable L_max, buffer and pass rate.

---

## §5 Oversampling and dynamic sampling for zero-variance groups

**Dynamic sampling.** DAPO drops groups whose responses are all correct or all wrong and samples more prompts until the batch is full; this was the largest single step in its ablation, AIME24 42 → 50 ([[dapo]] Table 1). slime implements the filter as a pluggable group function with oversampling: the rollout loop keeps submitting `over_sampling_batch_size` prompts until `rollout_batch_size` groups pass, and cancels outstanding requests when the batch is full ([[slime]] `sglang_rollout.py` L492–618):

```python
# slime@79aa41a slime/rollout/filter_hub/dynamic_sampling_filters.py L9-25
def check_reward_nonzero_std(args, samples: list[Sample | list[Sample]], **kwargs):
    # Agent generators return one list of context-window segments per trajectory.
    # Count its terminal reward once, independent of the number of segments.
    rewards = [(sample[-1] if isinstance(sample, list) else sample).get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_reward_nonzero_std_with_fallback(args, samples: list[Sample], **kwargs):
    """Prefer non-zero-std groups without triggering another sampling round."""

    output = check_reward_nonzero_std(args, samples, **kwargs)
    output.keep_when_insufficient = True
    return output
```

The `_with_fallback` variant keeps a rejected group when dropping it would force another sampling round, which trades signal for latency.

**Active sampling vs up-front oversampling.** Olmo 3 keeps the batch full of non-zero-advantage groups by resampling as groups are filtered (active sampling) instead of oversampling 3× up front, and states that this reduced loss variance and kept batch size constant ([[olmo-3-rl-pipeline]] §4.4.2, §6 "Active sampling stabilizes training", Fig. 26). In open-instruct the filter is one line in the group builder:

```python
# open-instruct@f020e9a open_instruct/data_loader.py L812-818
    percent_solved = np.mean(result.reward_scores).item() / max_possible_score
    if no_resampling_pass_rate is not None and percent_solved >= no_resampling_pass_rate:
        assert iter_dataloader is not None
        iter_dataloader.exclude_index(result.index)

    if filter_zero_std_samples and np.std(result.reward_scores) == 0:
        return None
```

The first branch is ScaleRL's No-Positive-Resampling (ch-12): a prompt solved at or above the threshold is excluded from future sampling.

**Early exit.** MAI-Thinking-1 samples G_early = 16 responses first and continues to the full G = 128 only if the early pass rate is in [0.05, 0.8]; after the full group it applies a second filter [0.1, 0.8] "to remove very low-variance groups" ([[mai-thinking-1]] Sec. 3.1.3, 3.1.5). Early exit spends 16 rollouts to decide whether to spend 128 more. In Rocket v1 the problem worker implements it as 16 requests, then 128 more, then advantage postprocessing (Sec. 3.6.2).

**Reusing signal instead of sampling more.** POLARIS's Rollout Rescue keeps a correct response per prompt from an earlier epoch and substitutes it for one failed rollout when a group is 0/8; Intra-Batch Informative Substitution duplicates mixed-outcome groups to replace zero-advantage groups, which the authors state matches DAPO's benefit "without extra rollouts" ([[polaris]] Zero-advantage groups). A rescued response was sampled by an older policy, so it is off-policy; the blog does not discuss an importance correction.

**Selecting within a group.** rStar2-Agent oversamples 2G = 32 rollouts and keeps G = 16: negatives sampled uniformly, positives sampled with probability inversely proportional to a tool-error-plus-format penalty, so correct trajectories with failed tool calls are reinforced less often ([[rstar2-agent]] §2.2.3). This moves quality control from the reward into rollout selection; the authors chose it because penalizing tool errors in the reward risked reward hacking.

**Cost.** BroRL's 41% pass rate at N = 16 means 59% of groups were discarded (Table 2). Dynamic sampling's gain is bought with that generation cost.

---

## §6 Partial rollouts

A partial rollout is generation that is stopped before completion and continued later, possibly under newer weights.

- **Kimi k1.5** gives each rollout a fixed output-token budget per iteration; unfinished parts continue in the next iteration, with repeat detection and early stopping ([[kimi-k1-5-recipe]] §2.6.2; budget value not printed).
- **slime** with `--partial-rollout`: when a batch is full, aborted groups with non-empty responses are returned to the buffer with their start rollout id; `--buffer-sort-by-staleness` resumes the oldest first, and `--mask-offpolicy-in-partial-rollout` excludes continued-prefix tokens from the loss ([[slime]]; [[slime-agentic-and-mismatch-examples]]).
- **verl fully_async_policy** with `partial_rollout=True` interrupts generation before parameter sync and resumes afterwards. On 128 H20 GPUs with Qwen2.5-Math-7B and DAPO, 400 steps took 1d 16h 48m colocated vs 17h 22m fully async with 64:64 GPUs (2.35×); AIME-2024 mean@1 at the last step 0.2958 vs 0.3094 and max 0.3573 vs 0.3521 ([[verl-rollout]] `fully_async_policy/README.md` L333–369). One run each.
- **PipelineRL** goes further: weights change between decoding steps of the same sequence (in-flight update), and the KV cache computed under old weights is kept. Mixed-policy sequences with stale KV showed only slightly higher KL than with recomputed KV (Fig. 7); time to the same reward was about 2× faster than conventional RL with G = 32 optimizer steps per RL step, and ESS was similar to G = 8 ([[pipelinerl]] §5). Conventional RL beyond G = 32 was unstable (G = 64 diverged, Fig. 10). The paper clamps importance weights at 5.

**Why not regenerate interrupted rollouts?** DeepSeek-V4 argues that regenerating from scratch after preemption "is mathematically incorrect" because shorter responses are more likely to survive interruption; it resumes from a token-granular write-ahead log and saved KV cache ([[deepseek-train-infer-consistency]] Sec. 5.2.3). The same argument applies to aborting long responses at every weight sync: the surviving completed set is biased toward short responses. Partial rollout avoids that bias at the cost of mixed-version sequences.

---

## §7 Multi-turn loops and tool-call parsing

**Parsing.** The rollout worker turns generated tokens into tool calls. verl's tool parser extracts calls from `response_ids` (default format `hermes`) ([[verl-rollout]]; `tool_agent_loop.py` L302–304 at 8357f1e). AReaL's search agent uses `<tool_call>{json5}</tool_call>` and returns results as a user message `<tool_response>…</tool_response>` ([[areal-agent-pipelines]]). rStar2-Agent ends an assistant turn at EOS, executes a `<tool_call>` JSON block, and appends the result under the user role as `<tool_response>`, until an `<answer>` tag or T turns ([[rstar2-agent]] §2.1).

**What to do with malformed calls.** The sources differ:
- GLM-4.5 gives reward 0 for bad tool-call format ([[recipe-crosswalk]] GLM-4.5 row).
- rStar2-Agent keeps the reward answer-only and lowers the selection probability of correct trajectories with tool errors or format problems (§5 above).
- rStar2's environment returns errors with tracebacks to the model as observations, so the model can recover inside the episode.

**Turn and time limits.** verl: `max_assistant_turns`, `max_user_turns`, and response-mask length (`tool_agent_loop.py` L291–297). AReaL search agent: 100 LLM calls, 150 minutes, forced answer at 80% of 27,648 tokens. slime ReTool: 16 turns. slime coding agent: no turn cap; up to two context compactions, a third stops the agent and grades the existing patch (`README.md` L226–228). MAI-Thinking-1: step, context-length and time budgets (Sec. 3.3). A limit that ends the episode without an answer produces a truncated rollout, so §4's choices apply to it.

**Tool output length.** verl truncates tool responses longer than `max_tool_response_length` (default 256 characters) in the middle (`rollout.yaml` L223). Truncated observations change what the model sees; a model trained with 256-character observations may fail with full ones. The default suits short tool outputs, not search results or file contents.

---

## §8 Token-in/token-out

ch-08 §7 describes the three implementations (slime CLEAN/REALIGN/FORK, verl Continuous Token builders, AReaL concat export). The rollout-design rules are:

1. Store the engine's token ids and per-token log-probs for every model turn; never rebuild them from decoded text. slime's Search-R1 example warns that re-tokenizing a post-processed string "may produce different tokens than what the engine generated, leading to misalignment between tokens and their log probabilities" ([[tito-token-in-token-out]], `generate_with_search.py` L96–97).
2. Tokenize only new tool or user content as a delta; give it loss mask 0. Give template boundary tokens the engine did not sample loss mask 0 and no log-prob.
3. Keep the template-consistency check on (verl default `tokenization_sanity_check_mode: strict`, `rollout.yaml` L242), and add a per-model-family test that the trained ids equal the sampled ids.
4. When a turn's prompt diverges from the held tokens by more than a short realignment can absorb, start a new sample (slime FORK, default threshold 1,024 tokens) instead of training on a sequence the sampler never produced.

The only reported effect size is verl's statement that applying the chat template to the final history made PPO "not even converged in single-turn" (experience report, no numbers) ([[verl-rollout]]).

---

## §9 Recording sampler log-probs and other replay data

The learner needs from the rollout everything required to compute the correct ratio:
- **Per-token sampler log-probs.** verl: `calculate_log_probs: True` (`rollout.yaml` L252); slime adapters call SGLang with `return_logprob=True`; open-instruct stores vLLM log-probs for its ρ correction ([[olmo-3-rl-pipeline]]). Rocket's rollout store holds tokens and sampling log-probs that the learner fetches ([[mai-rl-infrastructure]] Fig. 1).
- **Top-p masks** if top-p < 1 ([[mai-thinking-1]] Sec. 3.1.3).
- **MoE routed experts** for routing replay: verl returns `routed_experts` when `enable_rollout_routing_replay` is set ([[verl-rollout]]); MAI-Thinking-1 uses routing replay (Sec. 3.6.4).
- **Per-token policy version** for staleness masks and metrics (AReaL `versions`, slime `weight_versions`).

With sampler log-probs stored, two training modes are possible. Decoupled mode recomputes π_old with the learner and applies a sampler-correction weight; [[fp16-train-infer-mismatch]] estimates the extra forward at "approximately 25%" more training cost (Sec. 2.1.1). Bypass mode uses the sampler log-probs as π_old and skips that forward ([[verl-rollout-correction-and-lowprec]]). ch-07 covers which correction to apply. ScaleRL measured that computing LM-head logits in FP32 raised the asymptote from 0.52 to 0.61 (Fig. 5b), a rollout-precision choice with a measured effect.

---

## §10 Synchronous vs asynchronous rollout scheduling

**Measured effects of asynchrony.**
- ScaleRL: PipelineRL with max off-policyness k = 8 reached a similar asymptote to PPO-off-policy-k and a higher efficiency B; k = 8 was best among the values tried (Fig. 4).
- PipelineRL: about 2× faster time-to-reward than the most aggressive stable conventional setting, driven by about 2× sample throughput; the conventional baseline's throughput was estimated from a 4-node run plus a correction, not measured at 128 GPUs ([[pipelinerl]] §5).
- Olmo 3 infrastructure ablation on a fixed 2-node benchmark: 881 tokens/s (OLMo 2) → 975 with continuous batching → 1,358 with better threading → 2,949 with in-flight updates; MBU 12.90% → 43.21% ([[olmo-3-rl-pipeline]] Table 23). Static batching at 32K max and 14,628 mean length would waste up to 54% of compute (§4.4.3). The 7B Think run first took about 15 days without PipelineRL-style updates or TIS; the replication on newer infrastructure took 6 days (§4.4.2).
- verl fully async: 2.35× wall-clock speedup at similar AIME accuracy (§6 above).

**Bounds used.** MAI-Thinking-1: 5 gradient steps per inference update; rollouts more than 8 updates (40 steps) stale are dropped (Sec. 3.1.5). Olmo 3: max asynchrony 1 (7B Think) or 8 (others) (Table 49). AReaL: admission bound `max_head_offpolicyness`, default 0, 4 in the agent examples ([[areal-agent-pipelines]]). slime: no hard bound found; staleness logged ([[slime-agentic-and-mismatch-examples]]).

**Placement rule.** slime's documentation recommends `--colocate` when rollout lengths are similar and GPU count is the constraint, and separate pools with the fully async rollout function when lengths vary widely, and states that the async speedup bound max(t_R, t_T) + t_W is optimistic ([[slime]] Guideline). PipelineRL's authors also state that gains shrink when generation lengths are uniform ([[pipelinerl]] §7). Asynchrony helps in proportion to length variance. The infrastructure for this choice is ch-02 to ch-05.

---

## Recipe

1. **Decide the comparison method first**: held-out pass rate vs compute, fitted asymptote, at least two seeds for any difference near ±0.02.
2. **Sweep temperature** on the starting model before RL; take the highest value before accuracy drops; plan to raise it between stages if entropy falls.
3. **If top-p < 1, store and replay the mask**; otherwise use top-p = 1.
4. **Choose n** from the pass-rate distribution of your prompts (figure panel 1): enough that most groups are informative, small enough that long-tailed tasks do not hold groups open for hours. Use early exit (16 first, then the rest) if n is large.
5. **Set the length schedule** from the starting model: start at the maximum if the model already produces long responses; otherwise stage up. Add a soft overlong buffer; test masking vs not masking truncated responses on your setup.
6. **Filter zero-variance groups** with active sampling rather than a fixed oversampling factor; log the discard rate.
7. **For multi-turn**, set turn, token and time limits; return tool errors as observations; size tool-output truncation for the actual tools.
8. **Record** token ids, sampler log-probs, top-p masks, routed experts and per-token policy versions for every rollout.
9. **Choose scheduling** by length variance: colocated synchronous for uniform lengths; asynchronous with a staleness bound (k ≈ 8 in ScaleRL and Olmo 3) for long-tailed lengths. Resume interrupted rollouts rather than regenerating them.
10. **Track** entropy, response length, truncation rate, zero-variance discard rate, and one instruction-following metric.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Picking settings by early reward | worse final asymptote | fit compute curves; compare A |
| Default temperature for every model | entropy collapse within ~100 steps or degenerate text | temperature sweep before RL; entropy trace |
| top-p < 1 without mask replay | divergence within a few steps | mismatch metric; MAI Sec. 3.1.3 |
| Small n on prompts with extreme pass rates | most groups zero-variance; small effective batch | discard rate; figure panel 1 |
| Full failure reward for truncated responses | response length collapses or accuracy on long problems falls | truncation rate vs accuracy by length bucket |
| Masking truncated responses without checking | unstable batch size; lower overall accuracy | compare masked vs unmasked on a short run |
| Starting at a short limit for a model that already reasons long | irreversible drop | POLARIS 24K vs 40K pattern; check initial length distribution |
| Regenerating interrupted rollouts | length distribution biased short | compare lengths of resumed vs completed-first-try rollouts |
| Tool-output truncation sized for toy tools | model never sees full observations | distribution of observation lengths vs limit |
| Re-tokenizing multi-turn history | log-prob/token misalignment | trained-ids-equal-sampled-ids test |

---

## Check your understanding

1. ScaleRL found that smaller batches and shorter generations look better early and worse late. Explain why early reward is a biased guide for these two settings specifically.
2. Using p^n + (1 − p)^n, explain why BroRL's dynamic-sampling pass rate rose from 41% to 62% when N went from 16 to 512, and why ScaleRL could still see no difference between 8 and 32 at fixed total batch.
3. Why does sampling with top-p < 1 and training with the full softmax create off-policy updates even when sampler and learner are bitwise identical otherwise? What does mask replay change?
4. DAPO found that masking truncated samples helped, while Skywork-OR1 and Olmo 3 RL-Zero found it hurt. Identify the differences in setup that could explain the conflict, and design a run that would separate them.
5. MAI-Thinking-1's length penalty scales with the problem's pass rate. What behavior does this encourage on easy problems and on hard problems, and why would a constant penalty work worse in a climb from base?
6. DeepSeek argues that regenerating preempted rollouts biases toward short responses. Explain the selection mechanism, and say whether aborting rollouts at every weight sync has the same bias.
7. PipelineRL keeps the KV cache computed under old weights after an in-flight update. Why is the resulting KL only slightly higher than with recomputation, and where in a long sequence is the lag largest?

---

## Connections

- ch-02 to ch-05 (async loop, staleness, weight transfer that §6 and §10 depend on), ch-07 (sampler–learner mismatch, the reason §9's replay data must be stored), ch-08 (multi-turn agent loop and TITO implementations), ch-09 (pass rates measured at build time), ch-11 (graders that produce the rewards filtered in §5), ch-12 (prompt selection and length terms as reward components).
- Algorithm mechanics: [llm-training ch-40](../../llm-training/ch-40/read.md) (zero-variance groups and group baselines), [llm-training ch-43](../../llm-training/ch-43/read.md) (entropy and KL control), [llm-training ch-44a](../../llm-training/ch-44a/read.md) (overlong responses and length control), [llm-training ch-45b](../../llm-training/ch-45b/read.md) (multi-turn agentic RL), [llm-training ch-16](../../llm-training/ch-16/read.md) (prompt distribution).
- Sources: [[rl-design-guide]] (§2 outline), [[scalerl]], [[pipelinerl]], [[polaris]], [[skywork-or1]], [[brorl]], [[dapo]], [[kimi-k1-5-recipe]], [[async-rollout]], [[verl-rollout]], [[slime]], [[tito-token-in-token-out]], [[mai-thinking-1]], [[olmo-3-rl-pipeline]], [[rstar2-agent]], [[deepseek-train-infer-consistency]], [[areal-agent-pipelines]], [[scaling-reasoning-losing-control-mathif]].

---

## Sources

- [[rl-design-guide]] — §2 rollout rules used as this chapter's outline.
- [[scalerl]] — compute-curve fitting, A vs B classification, FP32 head, length and batch scaling, generations per prompt.
- [[pipelinerl]] — in-flight weight updates, stale KV, ESS and speed vs conventional RL, G = 64 divergence.
- [[polaris]] — per-model temperature zones and stage temperatures, length start, rollout rescue and substitution.
- [[skywork-or1]] — T = 1.0 vs 0.6, SGD steps and entropy collapse, truncation-mask ablation, staged context.
- [[brorl]] — N = 512 vs 16 at equal compute, dynamic-sampling pass rate, throughput.
- [[dapo]] — dynamic sampling, overlong filtering, soft overlong punishment, setup.
- [[kimi-k1-5-recipe]] — partial rollout with token budget, length reward.
- [[async-rollout]] — asynchronous rollout designs and their queue/staleness semantics.
- [[verl-rollout]] — rollout defaults, agent-loop limits, fully async partial-rollout timings, TITO report.
- [[slime]] — dynamic-sampling filter and oversampling loop, placement guideline, partial rollout.
- [[tito-token-in-token-out]] — token-delta construction rules.
- [[mai-thinking-1]] — top-p mask replay, early exit and pass-rate windows, G = 128, length doubling, difficulty-scaled length penalty, staleness cutoff (read from the PDF).
- [[olmo-3-rl-pipeline]] — active sampling, RL-Zero unmasked truncation, infrastructure throughput ablation, Table 49 settings.
- [[rstar2-agent]] — Resample-on-Correct selection, tool-call format.
- [[deepseek-train-infer-consistency]] — WAL resumption and short-response survival bias.
- [[areal-agent-pipelines]] — tool-call format, episode limits, admission bound.
- [[scaling-reasoning-losing-control-mathif]] — rollout length vs instruction following.
- [[mai-rl-infrastructure]] — Fig. 4 group waiting time; rollout store of tokens and sampling log-probs.
- [[fp16-train-infer-mismatch]], [[verl-rollout-correction-and-lowprec]] — cost of the extra old-log-prob forward; decoupled vs bypass mode.
- [[slime-agentic-and-mismatch-examples]] — coding-agent sampling settings and staleness logging.
- [[recipe-crosswalk]] — GLM-4.5 zero reward for bad tool-call format.
- [[deepmath-103k]] — 16 rollouts per prompt in its RL setup.
