<!-- chapter: ch-16
     track: rl
     kind: content
     title: RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse
     deps: [ch-40]
     sources: [[rlvr-tulu3]], [[deepseek-r1]], [[kimi-k1-5]], [[dapo]], [[verl-dapo-recipe]], [[magistral]], [[llama-nemotron]], [[olmo-3]], [[skywork-or1-rl-data]], [[deepmath-103k]], [[multi-domain-rlvr-data-centric]], [[reasoning-or-memorization-rl-contamination]], [[replay-buffer-rlhf]], [[trl-grpo]], [[reinforcement-learning-with-one-training-example]], [[rlvr-beyond-base-model]]
     figures: figures/rollout-passrate.html
     revised: 2026-09 (generality revision)
-->

# Chapter 16 — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse

> **Core insight.** In group-baseline RL (ch-40) a prompt contributes gradient only when the sampled group contains both successes and failures, so the prompt set is selected by measured pass rate rather than by human difficulty labels. Every disclosed 2025 recipe applies the same cut in a different place: DAPO resamples until each batch contains only groups with accuracy strictly between 0 and 1 and reports AIME 2024 avg@32 42 → 50 when that step is added ([[dapo]], Table 1); Llama-Nemotron discards prompts whose pass rate over 8 samples is 0.75 or higher ([[llama-nemotron]], §5.1); Olmo 3 Think 7B discards prompts solved in more than 62.5% of 8 rollouts ([[olmo-3]], §4.4.2); Skywork-OR1 removes 0/N and N/N prompts at N = 16 and reports that this leaves 46.2% of its math pool for the 7B model and 37.3% for the 32B ([[skywork-or1-rl-data]], §6.2).
>
> **Guideline.** When the RL algorithm is a group baseline, measure pass rate with the checkpoint you are about to train, at the sampling settings you will train with, and remove prompts whose measured pass rate is 1; remove pass-rate-0 prompts only if the rollout budget is the binding constraint, because a 0 today can become a 0.2 later. When rollout throughput is not the bottleneck, prefer online filtering (DAPO dynamic sampling, Olmo 3 active sampling) over a fixed offline cut, because the offline cut is measured against a checkpoint that stops being the policy after the first update. When a filtered pool is used for more than one domain, re-check the per-domain share after filtering, because a one-sided cut removes the domains the model already solves. When reusing samples across gradient steps, count how many optimizer steps old they are and log the importance ratio; Skywork-OR1 reports that reusing one rollout batch 2 or 4 times accelerates entropy collapse and lowers final test performance relative to a single on-policy step ([[skywork-or1-rl-data]], §4.4).

---

## Why this chapter matters for a general-purpose model

The pipeline position is: pretraining → mid-training → SFT → preference optimization → **RL** → evaluation. By the time RL starts, the policy's abilities are fixed by the earlier stages, and RL changes the probabilities of outputs the policy can already sample. What the RL stage can touch is bounded by the prompt set: a prompt that is never sampled produces no gradient, and a domain that is absent from the prompt set gets no direct signal at all.

Two failure directions matter for a generally capable model.

1. **Narrowing by construction.** A prompt set restricted to competition mathematics gives a policy whose measurable improvements are in competition mathematics. [[magistral]] trains a 24B model on math-only and on code-only RL from the same checkpoint (AIME'24 32.2, LiveCodeBench v5 22.7) and reports math-only RL reaching 62.5 / 38.3 and code-only RL reaching 49.7 / 42.7 on those two benchmarks (§6.1, Table 5) — each run moves both numbers, but the larger movement is in the trained domain.
2. **Narrowing by filtering.** Difficulty filtering is not domain-neutral. Removing prompts the model already solves removes more prompts from domains the model is good at. [[rlvr-tulu3]] shows the endpoint of this at 405B: GSM8K was dropped from the RLVR mixture because the model had saturated it after SFT and DPO, and the IFEval prompts were dropped as unhelpful, leaving MATH as the only RLVR training domain (§8.1).

The measurement question for this stage is whether reported gains come from the prompts or from the evaluation set. [[reasoning-or-memorization-rl-contamination]] shows that on Qwen2.5-Math-7B, given the first 60% of a MATH-500 problem, the model reconstructs the remaining 40% with 54.60 exact match and answers 53.6% of those partial problems correctly, while Llama3.1-8B scores 3.80 and much lower answer accuracy on the same protocol (§4.2, Table 2; §1). RL data claims measured only on MATH-500 with a Qwen2.5 base are therefore not evidence about prompt-set design.

---

## §1 What an RL prompt set is

**Definition.** An RL prompt set is a collection of tuples `(x, g)` where `x` is a query and `g` is a grading mechanism that maps any sampled response `y` to a scalar reward. There is no target response. The gradient comes from the sampled responses, so the prompt's contribution depends on what the current policy samples for it.

**Graders in disclosed recipes.**

| Grader | Example in a disclosed recipe | Locus |
|---|---|---|
| String/symbolic answer match | Tülu 3 RLVR math: extract the final answer, compare to the reference | [[rlvr-tulu3]] §6.1 |
| Constraint checker | Tülu 3 IF verifiable: one verification function per constraint template | [[rlvr-tulu3]] §6.1 |
| Unit tests | DeepSeek-R1 coding: 17k competition problems plus 8k GitHub bug-fixing tasks with failing unit tests | [[deepseek-r1]] B.3.1 |
| Multiple-choice option match | DeepSeek-R1 STEM: 22k questions with four to eight options, binary reward on the matched option | [[deepseek-r1]] B.3.1 |
| Judge model against a reference | Llama-Nemotron: Llama-3.3-70B-Instruct judges whether the prediction matches the ground-truth answer | [[llama-nemotron]] §5.1 |
| Reward model | DeepSeek-R1 general: 66k helpfulness questions plus 12,000 harmlessness questions, two reward models | [[deepseek-r1]] B.3.1 |

**Disclosed prompt-set sizes.** These are the sizes as printed, with the unit each report uses.

| Release | Prompts | Composition as reported | Locus |
|---|---|---|---|
| Tülu 3 8B/70B RLVR | 29,946 | GSM8K train 7,473; MATH train 7,500; IF verifiable 14,973 | [[rlvr-tulu3]] Table 22 |
| DeepSeek-R1 | 26K math, 17K code (+8k bug fixing), 22K STEM, 15K logic, 66K general | one row per data type, prompt counts only | [[deepseek-r1]] Table 4, B.3.1 |
| Olmo 3 Think RL (Dolci-Think-RL) | 104,869 | precise IF 30,186; math 30,186 across five sources; code 23,110; general chat 21,387 | [[olmo-3]] Table 20 |
| Skywork-OR1-RL-Data | 105,055 math, 14,057 code | HF dataset card split sizes; the report states about 105K math and 13.7K code (2.7K LeetCode, 11K TACO) | [[skywork-or1-rl-data]] card + §6.1 |
| DeepMath-103K | 103K | 2,869K raw → decontamination → difficulty ≥ 5 → answer verification = 95K, plus 8K from SimpleRL | [[deepmath-103k]] §3, Figure 7 |
| Magistral math | 38k | 699k initial → 501k after format filtering → 38k after difficulty filtering | [[magistral]] §4.1, Table 1 |

Two properties are worth naming before the mechanics.

**(i) The grader defines the prompt as much as the question does.** [[kimi-k1-5]] removes multiple-choice, true/false, and proof-based questions from its RL prompt set because a correct final answer can be reached through incorrect reasoning, and additionally removes a prompt when a model without chain-of-thought guesses the answer within N = 8 attempts (§2.1). [[deepseek-r1]] keeps 22k multiple-choice STEM questions with a binary reward on the matched option and excludes mathematical proofs for the same verifiability reason (B.3.1). The two reports disagree on multiple choice; they agree that unverifiable formats do not enter. Treat the multiple-choice question as open: neither report ablates it.

**(ii) A prompt's value is a property of the pair (prompt, current policy).** Difficulty labels written by humans or by a judge do not predict the gradient. [[kimi-k1-5]] defines difficulty as the pass rate of an SFT model that answers each prompt ten times at a relatively high sampling temperature (§2.1). [[llama-nemotron]], [[olmo-3]], [[skywork-or1-rl-data]] and [[magistral]] all measure the same quantity with their own checkpoint. §2 is why.

---

## §2 Pass rate decides whether a prompt produces a gradient

**Problem statement.** In a group baseline (ch-40), for a prompt `x` the trainer samples `G` responses, computes rewards `r_1..r_G`, and sets the advantage of response `i` to `A_i = r_i − r̄` (optionally divided by the group standard deviation), where `r̄ = (1/G) Σ_i r_i`. If all `G` rewards are equal, then `A_i = 0` for every `i` and the prompt contributes no policy-gradient term, having consumed `G` rollouts.

**Mechanism with binary rewards.** Let `p` be the probability that the current policy solves `x` at the training sampling settings, and let `k` be the number of successes in a group of size `G`.

```
A_i = r_i − k/G
Σ_i |A_i| = k(1 − k/G) + (G − k)(k/G) = 2k(G − k)/G
P(zero-variance group) = p^G + (1 − p)^G     (independent samples)
```

`A_i`: advantage of response i; `r_i ∈ {0,1}`: verifier output; `k`: successes in the group; `G`: group size; `p`: per-sample success probability.

**Worked example (G = 8).** Absolute advantage mass `2k(G−k)/G`: k = 4 gives 4.00; k = 2 or 6 gives 3.00; k = 1 or 7 gives 1.75; k = 0 or 8 gives 0. Probability that a group is wasted, `p^8 + (1−p)^8`: p = 0.5 gives 0.0078; p = 0.75 gives 0.100; p = 0.9 gives 0.430; p = 0.95 gives 0.663. A prompt the policy solves 90% of the time therefore produces no gradient in about 43 of every 100 groups, and in the remaining groups its advantage mass is concentrated in one or two responses.

The interactive version of this calculation, with the published cut points drawn on the same axis, is in [figures/rollout-passrate.html](figures/rollout-passrate.html): move the group size and the pass rate and read off wasted-group probability, advantage mass, and the effect of a one-sided cut on a two-domain pool.

**Evidence that the wasted fraction grows during training.** [[dapo]] reports that the number of prompts with accuracy equal to 1 keeps increasing during training, so the effective number of prompts per batch keeps decreasing (§3.2, Figure 3b). This is the reason the cut is applied continuously rather than once.

**Conditions and limits.** The zero-variance formula assumes independent samples and a binary verifier. With a continuous reward (an LM judge score in [0,1], a proportional test-pass fraction), exact ties are rarer and the zero-gradient case is replaced by a low-variance case; Olmo 3 states its filter as "rewards are all identical", which covers both ([[olmo-3]] §4.4.1).

**Implication for a general-purpose model.** The quantity being maximized by the filter is gradient per rollout, not capability breadth. Everything in §3 and §4 follows from the fact that these two objectives are different.

---

## §3 Difficulty filtering: offline cuts, online cuts, and what each costs

### 3.1 Offline filtering (measure once, before training)

Mechanism, in the order the reports describe it:

1. Choose the checkpoint that will start RL.
2. Sample `N` responses per prompt at the training temperature.
3. Compute the pass rate.
4. Apply a cut and keep the rest.

Disclosed instances:

| Release | Sampling for the measurement | Cut | Locus |
|---|---|---|---|
| Kimi k1.5 | SFT model, 10 answers, "relatively high" temperature | pass rate used as a difficulty label for curriculum and prefiltering of trivial cases | [[kimi-k1-5]] §2.1 |
| Llama-Nemotron LN-Ultra | 8 responses from LN-Super | discard pass rate ≥ 0.75 | [[llama-nemotron]] §5.1 |
| Olmo 3 Think 7B | 8 rollouts from the DPO checkpoint, temperature 1.0, top-p 1.0, matching RL sampling | discard pass rate > 62.5% (that is, more than 5 of 8) | [[olmo-3]] §4.4.2 |
| Skywork-OR1 | 16 rollouts for math, 8 for code, temperature 1.0, 32K max tokens | discard 0/N and N/N | [[skywork-or1-rl-data]] §6.2 |
| Magistral | 16 samples from Mistral Large 2, then 16 samples from a 24B RL-trained grader | remove never-solved and high-success-rate problems in both passes; also remove problems where the sample majority agrees on an answer that differs from the reference | [[magistral]] §4.1 |
| DeepMath-103K | GPT-4o difficulty rating queried 6 times per problem and averaged | keep level ≥ 5 (scale from Omni-MATH/AoPS) | [[deepmath-103k]] §3 |

Retention, where reported: Skywork-OR1 gives, for DeepSeek-R1-Distill-Qwen-7B, 21.4% of math prompts all-incorrect and 32.4% all-correct, leaving 46.2%; for the 32B, 20.7% and 42.0%, leaving 37.3%; code retention is 48% and 37.6% ([[skywork-or1-rl-data]] §6.2). The same pool filtered against a stronger model retains fewer prompts and retains a different subset — the filter is a measurement of the model, not of the prompts.

Two costs of the offline cut:

- **It is stale after the first update.** The pass rates were measured with a checkpoint that the trainer immediately leaves. Olmo 3 uses the offline cut for the 7B and relies on active sampling for the 32B ([[olmo-3]] §4.4.2).
- **A weak grader discards genuinely useful prompts.** [[magistral]] argues that a single pass with Mistral Large 2 would have discarded many hard-but-solvable problems as unsolvable, which is why the second pass uses the RL-trained 24B grader (§4.1). That argument is stated by the authors without an ablation.

### 3.2 Online filtering (measure inside the training loop)

[[dapo]] dynamic sampling: over-sample, drop every group whose accuracy is 0 or 1, and keep sampling until the batch is filled with groups whose accuracy is strictly between 0 and 1 (§3.2, Eq. 11, Algorithm 1 lines 6–8). The constraint appears in the objective as `s.t. 0 < |{o_i : is_equivalent(a, o_i)}| < G`.

The verl implementation makes the cost explicit ([[verl-dapo-recipe]]):

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc          # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10   # Non-positive values mean no upper limit
```

Generation is run at three times the training batch size, and the trainer repeats generation until enough qualified groups exist or `max_num_gen_batches` is reached.

[[olmo-3]] replaces the oversample-and-discard pattern with continuous resampling: the asynchronous actor keeps pulling completions and refilling the prompt queue until the batch holds the target number of non-zero-gradient completions (§4.4.3). The report states this maintains a consistently full batch and reduces loss variance in an RL-Zero math run (§6.2, Figure 26); the comparison is one run per arm.

[[skywork-or1-rl-data]] combines both: prompts with base-model correctness 1 or 0 are removed before training, prompts the actor solved completely in the previous stage are removed at each stage boundary, and within a batch only groups with at least one non-zero advantage are kept (§3.1).

**Evidence for the online cut.** `avg@k` is the mean accuracy over `k` independent samples per question, reported as a single number. [[dapo]] Table 1 reports AIME 2024 avg@32 for Qwen2.5-32B base with techniques added progressively: naive GRPO 30, + overlong filtering 36, + clip-higher 38, + soft overlong punishment 41, + token-level loss 42, + dynamic sampling 50, against DeepSeek-R1-Zero-Qwen-32B at 47. The authors note the extra sampling does not significantly increase total training time because fewer training steps are needed (§4.2, Figure 6). The verl reproduction table reports 52% with DAPO and 50% without dynamic sampling on 16×8×H800 ([[verl-dapo-recipe]], Reproduction Runs). These are single runs per configuration, and the progressive-ablation format means the dynamic-sampling row is measured on top of all previous rows.

### 3.3 What the cut does not do

No disclosed report ablates the threshold value itself. The 0.75 of [[llama-nemotron]], the 62.5% of [[olmo-3]], and the strict 0-or-1 cut of [[dapo]] are three different answers, and none is compared against the others in a controlled setting (Open question). What is replicated across [[dapo]], [[olmo-3]], [[skywork-or1-rl-data]] and [[magistral]] is the removal of the solved end of the distribution; removal of the unsolved end is not universal — [[llama-nemotron]] cuts only the easy side and keeps pass-rate-0 prompts in the pool (§5.1).

---

## §4 Domain composition and the breadth of the prompt set

**Disclosed compositions.** [[deepseek-r1]] separates reasoning RL data (math, code, STEM, logic) from general RL data (66k helpfulness, 12,000 harmlessness) and introduces the general instruction data and preference-based rewards only in the final 400 steps of a 1,700-step second RL stage, stating that more steps under a model-based preference reward can lead to reward hacking (§3.2.2). [[olmo-3]] mixes precise instruction following, math, code, and general chat and reports roughly equal amounts per domain with slightly more math and instruction following (§4.4.2). [[kimi-k1-5]] states the three properties it curates for — diverse coverage across STEM, coding, and general reasoning; balanced difficulty; accurate evaluability — and uses a domain tagging system to keep subject areas balanced (§2.1); it gives no ablation numbers for the mixture.

**Evidence on mixing.** [[multi-domain-rlvr-data-centric]] trains Qwen2.5-7B-Base with GRPO on math, code, and puzzle data (math 20k = DeepScaleR 10k + CountDown 10k; code 12k = CodeR1-12k; puzzle 7.8k = Knights-and-Knaves 5.4k + Logic Puzzle Baron 2.4k, Table 1) and evaluates all three domains after each run (Table 9, base row 22.48 math / 67.46 code / 9.07 puzzle / 31.50 overall):

| Training data | Math avg | Code avg | Puzzle avg | Overall |
|---|---|---|---|---|
| Math | 47.48 | 64.23 | 22.42 | 45.11 |
| Puzzle | 29.47 | 71.35 | 61.98 | 50.72 |
| Code | 19.17 | 73.95 | 22.55 | 35.78 |
| Math + Puzzle | 49.72 | 44.90 | 49.78 | 48.36 |
| Puzzle + Code | 32.06 | 74.88 | 55.15 | 50.89 |
| Math + Code | 47.22 | 75.06 | 25.34 | 48.92 |
| Math + Code + Puzzle | 49.75 | 73.63 | 49.73 | 56.57 |

The three-domain mixture has the highest overall average (56.57) and avoids the code collapse that Math + Puzzle produces (44.90 against a base of 67.46), while the puzzle-only run remains the best puzzle model (61.98). Result (single study), one model family, one seed per configuration. The mechanisms behind these transfers and interferences are the subject of ch-44b; what this chapter takes from the table is that the composition of the prompt set is a decision with measurable cross-domain consequences, in both directions.

**Filtering changes composition.** A pass-rate cut applied to a mixed pool removes more of the domains the starting checkpoint already handles. Worked example with round numbers: a pool of 10,000 math and 10,000 instruction-following prompts, where 60% of the math prompts and 20% of the IF prompts sit above the cut, becomes 4,000 math and 8,000 IF — the math share falls from 50% to 33% without anyone choosing that. The disclosed instance of this effect is [[rlvr-tulu3]] §8.1: at 405B, GSM8K was removed from the RLVR mixture because SFT and DPO had saturated it, and IFEval prompts were removed because they did not help in initial runs, leaving MATH alone. [[olmo-3]] reports the reverse adjustment made deliberately — after offline filtering, eight OMEGA subtasks were downsampled by 50% because the model struggled with them (§4.4.2, footnote 39).

**Implication.** Measure the per-domain counts after filtering, not before, and treat the post-filter mixture as the mixture you are training on.

---

## §5 Curricula in prompt space

**Definition.** A curriculum orders the prompt distribution over training time instead of sampling it uniformly.

Disclosed forms:

1. **Prioritized sampling by tracked success rate.** [[kimi-k1-5]] tracks the success rate `s_i` of each problem during RL and samples problem `i` with probability proportional to `1 − s_i` (§2.3.4). This makes the sampling distribution follow the policy without a separate re-measurement pass.
2. **Staged difficulty with re-measurement.** [[kimi-k1-5]] warms up on the full mixture and then focuses on hard questions, reporting higher accuracy than uniform sampling (§3.5, Figure 9; no numbers are printed for the two curves). [[magistral]] raises data difficulty as performance increases, constructing harder splits by adding back problems that earlier stages filtered out and removing completely solved problems (§5.2).
3. **Distribution-shaped batching.** [[llama-nemotron]] builds each batch from a Gaussian target distribution over pre-computed pass rates whose center moves from high pass rates to low pass rates across batches, filling any remaining batch capacity from the pass-rate bins with the largest remaining pools (§5.1). The reported effect is on GPQA-Diamond Avg@4 over 200 steps, curriculum against random ordering (Figure 6).
4. **Curriculum with reference-policy refresh.** [[multi-domain-rlvr-data-centric]] trains on the Knights-and-Knaves puzzle set in six difficulty stages of 175 steps each and reports a final average of 97.29 for a standard curriculum against 94.29 for mixed training; replacing the reference model with the current actor and resetting the optimizer state at each stage boundary raises the final average to 99.71 (§6, Figure 6). Result (single study), one dataset.

**Risk of narrowing.** A curriculum that ends on the hardest slice of one domain ends training on a narrow distribution, and the last steps are the ones that set the final weights. Two measurements bound this risk. Both use `pass@k`: the fraction of problems for which at least one of `k` sampled responses is correct, so `pass@1` measures how often the policy is right on a single try and `pass@k` at large `k` measures how many problems the policy can still reach at all. [[rlvr-beyond-base-model]] reports that RLVR-trained models have higher pass@1 than their base models on the math, code, and visual-reasoning benchmarks tested, while at large k the base models match or exceed them on pass@k, which the authors read as RLVR raising the probability of solution paths the base model could already sample. [[olmo-3]] reports the opposite sign in a different setting: in RL-Zero training from Olmo 3 Base on the math domain, pass@32 on AIME 2024 and 2025 improves over training alongside pass@1, which the report reads as the run maintaining diversity and RLVR pushing the model beyond its initial capabilities (§6.2, Figure 24). (Olmo 3's other pass@k figure, Figure 20 right, compares the DPO and SFT checkpoints, not before and after RL, and is not evidence about RL.) Both are single-setting results with different bases and different prompt sets; the measurement to run for your own curriculum is pass@k at large k on held-out sets, before and after.

---

## §6 Prompt and trajectory reuse

This section corrects two claims that are commonly repeated: that frameworks replay prompts and never trajectories, and that the sequence-level importance ratio has expectation `exp(Tσ²/2)`.

### 6.1 What TRL's replay buffer actually stores

The TRL documentation for the experimental trainer states that it "trains a model with GRPO but replaces groups (and corresponding completions) that have 0 standard deviation with groups with high rewards and standard deviation that've been used to train a model in prior batches" ([[replay-buffer-rlhf]], `docs/source/grpo_with_replay_buffer.md`). The config field says the buffer "stores the rollouts with the highest advantage scores and variance per group".

From the cached copy of `trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py` (fetched 2026-09-14; verified extract in `excerpts/replay-buffer-rlhf.md`):

```python
buffered_output = {
    "prompt_ids": group_prompt_ids,
    "completion_ids": group_completion_ids,
    "advantages": group_advantages[group_idx].tolist(),
    "prompt_mask": group_prompt_mask,
    "completion_mask": group_completion_mask,
}
...
replay_buffer_scores = (group_advantages.abs() * group_std_rewards).sum(dim=-1)[groups_with_variance]
self.replay_buffer.add(replay_buffer_scores.tolist(), buffered_outputs)
```

and, when a group in the new batch has zero reward standard deviation:

```python
prompt_ids[idx_range] = sampled_data["prompt_ids"][i]
completion_ids[idx_range] = sampled_data["completion_ids"][i]
group_advantages[group_idx] = sampled_data["advantages"][i]
```

So the buffer stores completions and their advantages, and a stale group replaces a fresh zero-variance group wholesale. `old_per_token_logps` and `ref_per_token_logps` are stored alongside, but only when they were computed for that step: the trainer computes `old_per_token_logps` only when generation and optimization steps are misaligned or when vLLM importance-sampling correction is on, and otherwise sets it to `None`. In the parent `GRPOTrainer` loss the missing value is replaced by `per_token_logps.detach()` ([[trl-grpo]], commit a08e713), which makes the ratio 1 for every token. In that configuration, replayed completions enter the gradient with no importance correction at all.

**The buffer's scoring function reduces to the §2 criterion.** With the default `scale_rewards="group"`, advantages are divided by the group standard deviation before they reach the buffer, so `Σ_i |A_i| · σ_group ≈ Σ_i |r_i − r̄| = 2k(G − k)/G`. For G = 8 the score is 4.00 at k = 4 and 1.75 at k = 1 or 7: the buffer keeps the groups whose pass rate is nearest 0.5.

### 6.2 The importance-sampling ratio, stated correctly

Let `ρ(y) = π_θ(y|x) / π_old(y|x)` for a response `y` sampled from `π_old`. Then

```
E_{y∼π_old}[ρ(y)] = Σ_y π_old(y) · π_θ(y)/π_old(y) = Σ_y π_θ(y) = 1
```

exactly, for any amount of drift. The mean carries no information about staleness; the variance does:

```
Var_{π_old}[ρ] = E_{π_old}[ρ²] − 1 = χ²(π_θ ‖ π_old)
```

`χ²`: the chi-square divergence. For a token-factorized model, write `log ρ = Σ_t log ρ_t` over `T` tokens. If the per-token log-ratios are modelled as independent `N(μ, σ²)`, the constraint `E[ρ] = 1` forces `μ = −σ²/2`, so

```
log ρ ~ N(−Tσ²/2, Tσ²)
E[ρ] = 1,  E[ρ²] = exp(Tσ²),  Var[ρ] = exp(Tσ²) − 1
ESS / n ≈ 1 / E[ρ²] = exp(−Tσ²)      (Kish effective sample size)
median ρ = exp(−Tσ²/2)
```

`T`: response length in tokens; `σ²`: per-token log-ratio variance; `ESS/n`: fraction of the batch that behaves like an independent on-policy sample.

**Worked example.** With `σ = 0.01` and `T = 1,000`: `Tσ² = 0.1`, variance 0.105, ESS ≈ 0.90 n, median ratio 0.95. With `σ = 0.01` and `T = 10,000`: `Tσ² = 1`, variance 1.72, ESS ≈ 0.37 n, median ratio 0.61. With `σ = 0.03` and `T = 10,000`: `Tσ² = 9`, variance 8,102, ESS ≈ 1.2 × 10⁻⁴ n, median ratio 0.011 — the average weight is still exactly 1, carried by a small number of sequences. The quantity that degrades with staleness is therefore the spread of the ratio and the effective sample size, not its mean. The Gaussian-independence model is an illustration (Interpretation); no source measures per-token log-ratio variance this way.

Two consequences for practice. First, PPO-style objectives clip the ratio per token (or per sequence, in the length-normalized form), so the estimator stays bounded but becomes biased, and the bias grows with the fraction of clipped tokens. Second, the diagnostic to log is not the mean ratio (which is 1 by construction) but its spread: TRL logs `sampling/importance_sampling_ratio/{min,mean,max}` when the vLLM correction is on, and logs `frac_reward_zero_std` in every run ([[replay-buffer-rlhf]] extract, `_generate_and_score_completions`).

### 6.3 What the measurements say about reuse

[[skywork-or1-rl-data]] runs an ablation over the number of SGD steps taken on one rollout batch (DeepSeek-R1-Distill-Qwen-7B, rollout batch `D_R` = 64, group size 16, 16K context, temperature 1.0, no KL loss). With `N_SGD = (D_R/D_T) · N_reuse`, configurations with `N_SGD ∈ {2,4}` — reached either by splitting the rollout batch into smaller minibatches or by traversing the rollout buffer 2 or 4 times — show entropy decaying to small values within a few steps and test performance that stops improving, while the on-policy configuration (1,64,64,1) improves more slowly and ends higher (§4.4, Figures 16–17). The report isolates the cause with a further control: on-policy runs at the same small minibatch size do not show premature entropy collapse, so the degradation is attributed to the off-policy data that reuse introduces rather than to the minibatch size (§4.4, Ablation 7). Raising the rollout batch from 64 to 256 at `N_SGD` = 4 did not prevent the collapse (§4.4, Figure 18).

[[olmo-3]] takes the opposite approach at the infrastructure level: weights are pushed to the actors without pausing generation, so in-flight sequences are continued under newer weights, with the stated goal of minimizing how off-policy the samples are (§4.4.3).

[[kimi-k1-5]] reuses trajectory *segments*: with a fixed output token budget per iteration, an unfinished trajectory is saved to a replay buffer and continued in the next iteration, and the report states that only the current segment requires on-policy computation and that certain segments can be excluded from loss computation (§2.6.2). The report does not state how reused segments enter the gradient, so the correct summary is: partial rollouts are a long-context throughput technique whose gradient treatment is only partly disclosed.

**Recommendation.** When the trainer is synchronous and generation is the bottleneck, reuse the *prompt* (resample fresh responses) rather than the stored responses, because a resampled group needs no correction. When stored responses are reused, keep them one optimizer step old (the standard `num_iterations = 1` setting), store and use the generating policy's log-probabilities so the ratio is real rather than 1, and log the ratio spread and the entropy. When responses older than a few steps are the only option, expect the entropy-collapse pattern Skywork-OR1 reports and compare against an on-policy arm before adopting it.

---

## §7 Evaluation pitfalls for prompt-set claims

1. **Contaminated evaluation sets.** [[reasoning-or-memorization-rl-contamination]] measures partial-prompt completion (give the model a prefix, ask for the rest) and partial-prompt answer accuracy. On MATH-500 at a 60% prefix, Qwen2.5-Math-7B reaches 54.60 EM completion and Qwen2.5-7B 21.20, against 3.80 for Llama3.1-8B; on LiveMathBench the completion EM is 0.00 for all three (§4.2, Table 2). With spurious rewards on RandomCalculation — arithmetic problems generated after the models' release — only the correct reward improves performance, while random and inverted rewards do not (§4.3).
2. **Contaminated training pools.** [[deepmath-103k]] measures the contamination of its own raw pool before filtering and reports rates of 92.6% for Omni-MATH, 90.0% for AIME24 and AMC23, 76.6% for MATH-500, 35.7% for Minerva Math and 33.6% for OlympiadBench, defined as the percentage of benchmark test samples found in the raw pool (§2, Figure 3). Decontamination is embedding retrieval of the top-5 nearest benchmark items followed by an LLM judge that discards paraphrases (§3, Stage 2).
3. **Decontamination that reports its protocol.** [[deepseek-r1]] states that post-training mathematical SFT data and RL prompts were sourced from pre-2023 competitions and passed through the same 10-gram filter as pretraining, and acknowledges that n-gram filtering does not catch paraphrases (Supplementary, Decontamination).
4. **Negative controls.** [[olmo-3]] trains its base model on the RL-Zero mixture with random binary rewards as a negative control, on the argument that if the evaluation sets were memorized, a spurious reward would elicit them (§6.2).
5. **Single-prompt results.** [[reinforcement-learning-with-one-training-example]] reports that RLVR on one example raises Qwen2.5-Math-1.5B from 36.0% to 73.6% on MATH-500, with examples selected by the historical variance of their training accuracy (§1, §3.2). Read together with item 1 — the base model is a Qwen2.5-Math checkpoint and the benchmark is MATH-500 — this is a result about what RL can elicit from a contaminated base, not a demonstration that prompt-set size is irrelevant.

**Requirement.** A claim about prompt-set design needs at least one benchmark released after the base model, and preferably two base-model families.

---

## Negative samples and negative feedback

Which sense of "negative" applies here (§6.1 of the authoring standard): this chapter handles **(1) negative marginal value** — prompts and groups discarded because they contribute nothing or contribute noise — and it supplies the raw material for **(4) negative as gradient** — the below-mean responses inside a kept group. The derivations for (4) live in ch-43a; this section states what the prompt-set stage does to them.

**Where negatives come from.** A verifier, a test suite, a judge model, or a reward model labels each sampled response. False labels exist in both directions: [[kimi-k1-5]] removes formats where a correct answer can be reached by incorrect reasoning (false positives, §2.1); [[magistral]] removes problems where the sample majority agrees on an answer that disagrees with the reference, on the argument that the reference is more likely wrong (false negatives, §4.1); [[skywork-or1-rl-data]] found invalid or incomplete problems that survived difficulty estimation because the model produced the "correct" answer at least once in 16 rollouts (§6.3). None of these reports gives a false-label rate.

**What current practice does with them.** Zero-variance groups are discarded, not trained on ([[dapo]] §3.2; [[olmo-3]] §4.4.1; [[skywork-or1-rl-data]] §3.1; [[magistral]] §2.1). Inside a kept group, failures are used as gradient through the negative advantage `A_i = r_i − r̄ < 0`.

**Mechanism.** For a response with a negative advantage, the policy-gradient term is `A_i ∇ log π_θ(y_i|x)` with `A_i < 0`, which lowers the log-probability of every token of that response. At a single position, with logits `z` and softmax probabilities `p`, the gradient of the log-probability of the emitted token `y` is

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

so decreasing `log p_y` removes mass from `y` and redistributes it in proportion to `p_j` across the other tokens — that is, mostly to whatever the model already considered the next-most-likely continuation, which is not necessarily a correct one. ch-43a develops the consequences (likelihood displacement, squeezing).

**Why an all-wrong group teaches nothing.** With `r_i = 0` for all `i`, the baseline `r̄ = 0` equals the rewards, so `A_i = 0`. A failure is negative only relative to a baseline; with no success in the group, the baseline moves down to the failures and the signal disappears. This is the reason a pass-rate-0 prompt is not "hard training data" — it is not training data at all until the policy can occasionally solve it.

**Controls.** Keep the group size large enough that a low-pass-rate prompt still produces the occasional success (Skywork-OR1 uses 16 rollouts for math and 8 for code at estimation time, [[skywork-or1-rl-data]] §6.2; [[deepseek-r1]] samples 16 outputs per question in RL, §3.2.1; [[llama-nemotron]] samples 16 per prompt with a rollout prompt size of 72, §5.1). Filter out formats whose negatives are unreliable rather than penalizing them. Keep reuse low so that negative advantages are computed against the current policy (§6.3).

**Diagnostics.** Log the fraction of groups with zero reward standard deviation (TRL logs `frac_reward_zero_std`); log the pass-rate histogram of the batch, not only its mean; split response-length and entropy statistics by advantage sign; and track pass@k at large k on held-out sets to detect coverage loss.

**Effect on generality, and honesty about size.** None of the prompt-set reports measures what share of the improvement comes from negative-advantage samples. [[kimi-k1-5]] compares its objective against ReST, which fits the best sampled response without penalizing incorrect ones, and attributes better sample complexity across 11 evaluation sets to the negative gradients (§3.5, Figure 10) without quantifying the share. Do not present the share as measured.

---

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Tülu 3 8B RLVR | 8B | RL | RLVR prompt set (prompts, with ground-truth labels) | 29,946 = GSM8K 7,473 + MATH 7,500 + IF verifiable 14,973 | arXiv:2411.15124v5 Table 22 | verified 2026-09-15 | no ablation reported for the mixture |
| Tülu 3 8B RLVR | 8B | RL | total episodes (prompt visits); effective epochs over the combined set | 100,000 episodes; ≈ 3.3 epochs | arXiv:2411.15124v5 Table 21; §6.2 states 100,000/7,473 ≈ 13 epochs for the GSM8K-only ablation | derived (100,000 / 29,946) | no ablation reported |
| Tülu 3 8B RLVR | 8B | RL | PPO: LR, effective batch, response length, temperature, K, β, warmup | 3×10⁻⁷; 224; 2,048 (1,024 for GSM8K-only); 1.0; 4; β = 0.05; ω = 0.0 | arXiv:2411.15124v5 Table 21 and caption | verified 2026-09-15 | β swept over [0.1, 0.05, 0.03, 0.01] (§6.2) |
| Tülu 3 70B RLVR | 70B | RL | LR, episodes, batch, response length, β | 1×10⁻⁷; 400,000; 640; 2,048; β = 0.07 (Table 21 caption) vs β = 0.7 (§6.4 text) | arXiv:2411.15124v5 Table 21 caption; §6.4 | conflict (caption vs body; the caption value is attached to the released checkpoint) | prior 70B development runs (§6.4) |
| Tülu 3 405B RLVR | 405B | RL | prompt set; steps run | MATH train only (GSM8K removed as saturated, IFEval removed as unhelpful); 75 steps | arXiv:2411.15124v5 §8.1 | verified 2026-09-15 | initial RLVR runs at 405B (§8.1) |
| DeepSeek-R1 | not reported | RL | prompt counts by type | math 26K, code 17K (+8k bug fixing), STEM 22K, logic 15K, general 66K helpfulness plus 12,000 harmlessness | arXiv:2501.12948v2 Table 4, B.3.1 | verified 2026-09-15 | no ablation reported |
| DeepSeek-R1 | not reported | RL | first stage: LR, KL coef, GRPO clip ε, temperature, samples per question, batch | 3e-6; 0.001; 10; 1.0; 16; 32 questions/step = 512 | arXiv:2501.12948v2 §3.2.1 | verified 2026-09-15 | ε: lower truncates gradients, higher destabilizes (§3.2.1) |
| DeepSeek-R1 | not reported | RL | second stage: steps; where preference rewards enter; temperature | 1,700 steps; general instruction data and preference rewards in the final 400; 0.7 | arXiv:2501.12948v2 §3.2.2 | verified 2026-09-15 | more preference-reward steps risk reward hacking (§3.2.2, B.5) |
| Kimi k1.5 | not reported | RL | difficulty estimation; easy-to-hack removal; prioritized sampling | 10 SFT-model answers at high temperature; remove if a no-CoT model guesses within N = 8; sample ∝ 1 − s_i | arXiv:2501.12599v4 §2.1, §2.3.4 | verified 2026-09-15 | curriculum vs uniform sampling (§3.5, Figure 9; no numbers printed) |
| DAPO (Qwen2.5-32B base) | 32B | RL | dynamic sampling constraint; prompt batch; samples per prompt; ε_low/ε_high; LR | `0 < #correct < G`; 512; 16; 0.2/0.28; 1e-6 constant with 20-step linear warm-up | arXiv:2503.14476v2 §3.2, §4.1 | verified 2026-09-15 | Table 1: +dynamic sampling 42 → 50 AIME24 avg@32 (single run, cumulative ablation) |
| DAPO recipe in verl | 32B | RL | generation vs training batch; retry cap | `gen_batch_size` 1536, `train_batch_size` 512, `max_num_gen_batches` 10 | verl docs `algo/dapo.md`, last updated 2025-06-19 | verified 2026-09-15 | reproduction table: 52% with DAPO, 50% without dynamic sampling |
| Llama-Nemotron LN-Ultra | 253B | RL | offline difficulty filter; rollout prompt size; samples per prompt; global batch; updates per rollout | discard pass rate ≥ 0.75 measured with 8 LN-Super responses; 72; 16; 576; 2 | arXiv:2505.00949v5 §5.1 | verified 2026-09-15 | curriculum vs random batching on GPQA-D Avg@4 (Figure 6) |
| Olmo 3 Think 7B | 7B | RL | offline difficulty filter | 8 rollouts from the DPO checkpoint at temperature 1.0, top-p 1.0; discard pass rate > 62.5% | arXiv:2512.13961v2 §4.4.2 | verified 2026-09-15 | no threshold ablation reported |
| Olmo 3 Think 32B | 32B | RL | filtering strategy | active sampling instead of an offline cut; reuses the 7B-filtered data | arXiv:2512.13961v2 §4.4.2 | verified 2026-09-15 | compute and time constraints stated (§4.4.2) |
| Skywork-OR1 (R1-Distill-Qwen-7B / -32B) | 7B / 32B | RL | offline difficulty estimation and retention | 16 rollouts (math) / 8 (code) at temperature 1.0, 32K; discard 0/N and N/N; retained 46.2%/48% (7B), 37.3%/37.6% (32B) | arXiv:2505.22312v2 §6.2 | verified 2026-09-15 | zero-advantage groups give no policy gradient (§3.1) |
| Skywork-OR1 ablation baseline | 7B | RL | rollout batch, minibatch, reuse, group size, context, temperature, LR | D_R 64, D_T 64, N_reuse 1, gs 16, T 16K, τ 1.0, 1e-6 | arXiv:2505.22312v2 Table 5 | verified 2026-09-15 | N_SGD ∈ {2,4} collapses entropy and lowers test score (§4.4) |
| Magistral Medium | not reported | RL | math data funnel; difficulty grader | 699k → 501k (format) → 38k (difficulty); 16 samples per problem, twice (Mistral Large 2, then a 24B RL-trained grader) | arXiv:2506.10910v1 §4.1, Table 1 | verified 2026-09-15 | argued, no ablation reported |
| DeepMath-103K | dataset | RL data | funnel; difficulty rating; answer verification | 2,869K raw → decontamination → level ≥ 5 → verification = 95K, + 8K SimpleRL; GPT-4o rated 6× and averaged; 3 DeepSeek-R1 solutions must agree | arXiv:2504.11456v2 §3, Figure 7 | verified 2026-09-15 | raw-pool contamination rates (§2, Figure 3) motivate the decontamination stage |

**Starting point for a small general-purpose run.** Every number below comes from a `verified` row above, with the setting it was used in. Use a group size of 16 (DeepSeek-R1 at unreported scale; Llama-Nemotron at 253B), remove prompts whose pass rate over 8 rollouts at the training temperature is 1 (Olmo 3 Think 7B removes above 62.5%; Llama-Nemotron removes at or above 0.75 at 253B), keep the remaining prompts including pass-rate-0 ones if the rollout budget allows, and add the DAPO constraint `0 < #correct < G` in the loop with generation set to three times the training batch (verl `gen_batch_size` 1536 against `train_batch_size` 512, at 32B on 16×8×H800). Keep samples one optimizer step old (Skywork-OR1's on-policy baseline (1,64,64,1) at 7B). These are the conditions the numbers were measured under; none of them is an ablated optimum.

---

## Generalization lens

**(a) What increases breadth.**
- Mixing domains in the prompt set: [[multi-domain-rlvr-data-centric]] Table 9 and §4.2 — the three-domain mixture reaches the highest overall average (56.57 against 50.89 for the best pair) on Qwen2.5-7B-Base.
- Keeping a non-verifiable slice for general helpfulness: [[deepseek-r1]] carries 66k general prompts scored by reward models and adds them in the last 400 of 1,700 steps (B.3.1, §3.2.2); [[olmo-3]] carries general chat and instruction-following domains alongside math and code (Table 20).
- Balanced coverage as an explicit curation target: [[kimi-k1-5]] §2.1 lists diverse coverage, balanced difficulty, and accurate evaluability, and uses domain tagging to enforce the first.

**(b) What causes narrowing or forgetting.**
- Single-domain prompt sets: [[magistral]] §6.1 Table 5 — code-only RL moves LiveCodeBench v5 from 22.7 to 42.7 but math from 32.2 only to 49.7, against 62.5 for math-only RL.
- Domain interference inside a mixture: [[multi-domain-rlvr-data-centric]] Table 9 — Math + Puzzle drops the code average from 67.46 to 44.90.
- Filtering that removes solved domains: [[rlvr-tulu3]] §8.1 — the 405B RLVR mixture collapses to MATH alone.
- Curricula that end on one narrow slice: §5, with the pass@k caution from [[rlvr-beyond-base-model]].

**(c) How to measure it for this stage.**
- Per-domain counts after filtering, and per-domain evaluation after training, not only the target domain ([[multi-domain-rlvr-data-centric]] evaluates all three domains after every run).
- pass@k at large k on held-out sets before and after RL ([[rlvr-beyond-base-model]]; [[olmo-3]] §6.2 Figure 24 reports the opposite sign on its RL-Zero math run, so this is a measurement to make, not a result to assume).
- Contamination checks on every benchmark used to argue for a prompt-set choice: partial-prompt completion and answer accuracy ([[reasoning-or-memorization-rl-contamination]] §4.2), a spurious-reward negative control ([[olmo-3]] §6.2), and at least one benchmark released after the base model.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Filtering with a checkpoint other than the one that starts RL | Batch pass-rate histogram is shifted away from the filter band on step 1 | Log the pass-rate histogram of the first 50 batches and compare with the offline measurement |
| Keeping prompts the policy has solved | Rising fraction of zero-standard-deviation groups; effective batch size falls during training | Log `frac_reward_zero_std` per step ([[trl-grpo]]); apply the DAPO constraint or Olmo 3 active sampling |
| Cutting the hard end as well as the easy end when rollouts are not the bottleneck | Pool shrinks toward the middle; late training runs out of unsolved prompts | Compare a run with the one-sided cut ([[llama-nemotron]]) against the two-sided cut ([[skywork-or1-rl-data]]) on the same pool |
| Reading domain balance from the pre-filter mixture | Post-filter domain shares differ from the intended mixture; one domain dominates the gradient | Count prompts per domain after filtering, per training stage |
| Treating a difficulty label as difficulty | Human- or judge-assigned "hard" prompts produce zero-variance groups | Re-measure pass rate with the training checkpoint; the label is a prior, the pass rate is the signal |
| Reusing rollout batches to save generation time | Entropy falls within a few steps and test scores stop improving | Compare `N_reuse` = 1 against 2 with everything else fixed ([[skywork-or1-rl-data]] §4.4); log entropy per step |
| Replaying stored completions without stored log-probabilities | The importance ratio is exactly 1 for every token; no clipping is triggered although the samples are stale | Inspect whether `old_per_token_logps` is `None` for replayed groups ([[replay-buffer-rlhf]] extract, [[trl-grpo]] loss) |
| Diagnosing staleness by the mean importance ratio | The mean stays near 1 while training degrades | Log ratio min/max and the clipped-token fraction; the mean is 1 by construction (§6.2) |
| Claiming a prompt-set gain on MATH-500 with a Qwen2.5 base | Large gains from random or inverted rewards in a control run | Run the partial-prompt completion test and one post-release benchmark ([[reasoning-or-memorization-rl-contamination]]) |
| Removing a hackable format only from evaluation | Reward rises while held-out scores do not | Apply the no-CoT guessing test at curation time ([[kimi-k1-5]] N = 8) |

---

## Check your understanding

1. A prompt has a true pass rate of 0.9 under the current policy and the group size is 8. Explain, using the two formulas in §2, why keeping this prompt costs rollouts without producing a proportional gradient, and compute both quantities.
2. Llama-Nemotron removes prompts with pass rate ≥ 0.75 but keeps pass-rate-0 prompts, while Skywork-OR1 removes both ends. Give the condition under which each choice is the better one, in terms of what is scarce in the run.
3. TRL's replay buffer scores stored groups by `Σ|A_i| · σ_group` with group-scaled advantages. Show why this is approximately `2k(G−k)/G` for binary rewards, and say what pass rate the buffer therefore prefers.
4. Someone reports that their replayed trajectories are safe because the mean importance ratio logged over the batch is 1.02. Explain why this is not evidence, and state which two statistics would be.
5. A pool is 50% math and 50% instruction-following before filtering and 33% math afterwards, with no change to the filter. Explain the mechanism, and name the disclosed case in which this effect removed an entire domain.
6. DeepSeek-R1 keeps 22k multiple-choice STEM prompts; Kimi k1.5 removes multiple-choice prompts. Reconstruct the argument on each side and state what measurement would decide between them.
7. One-shot RLVR raises MATH-500 from 36.0% to 73.6% on Qwen2.5-Math-1.5B. Using §7, explain why this result cannot be used to argue that RL prompt-set size does not matter, and what evidence would be needed to make the argument.
8. A curriculum ends its last 500 steps on the hardest 5% of a math pool. Name the generality risk, and give the specific measurement, with its k, that would detect it.

---

## Connections

- **Previous chapter:** ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO. Supplies the advantage definition and the clipped objective that this chapter's filters act on.
- **Next chapter:** ch-43 — Entropy, Output Diversity, and KL Control in RL. Takes up the entropy-collapse pattern that §6.3 observes under sample reuse.
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Holds the derivations the negative-feedback section applies here.
- ch-44 — Process Supervision and Verifiable Rewards. Develops the grader side of the `(x, g)` pair.
- ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL. Covers partial rollouts and overlong filtering as length mechanisms.
- ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. Carries the transfer and interference evidence that §4 only samples.
- ch-37 — Policy-Gradient Foundations for Language Models. Source of the importance-sampling identity used in §6.2.
- ch-48 — Contamination Detection and Its Effect on Reported Scores. Extends §7 to the evaluation stage.

---

## Sources

- [[rlvr-tulu3]] — Tülu 3 RLVR prompt set (Table 22), PPO settings (Table 21), and the 405B mixture collapse to MATH (§8.1). Chapter-local verified extract: `excerpts/rlvr-tulu3.md`. The library card's claim that the verifier makes over-optimization "mechanically zero" is not supported by the report, which documents IFEval over-optimization (App. B.4); this chapter does not use that claim.
- [[deepseek-r1]] — RL data by type and prompt counts (Table 4, B.3.1), two-stage RL settings (§3.2.1–3.2.2), decontamination protocol.
- [[kimi-k1-5]] — prompt-set curation properties, pass-rate difficulty with 10 SFT samples, easy-to-hack removal at N = 8, curriculum and prioritized sampling, partial rollouts (§2.1, §2.3.4, §2.6.2). Chapter-local extract: `excerpts/kimi-k1-5.md`.
- [[dapo]] — dynamic sampling constraint and its Table 1 contribution; DAPO-Math-17K construction. Chapter-local extract: `excerpts/dapo.md`.
- [[verl-dapo-recipe]] — the configuration that implements dynamic sampling and its reproduction numbers. Chapter-local extract: `excerpts/verl-dapo-recipe.md`.
- [[magistral]] — two-stage difficulty filtering funnel (Table 1), staged difficulty schedule, math-only vs code-only cross-domain table (Table 5).
- [[llama-nemotron]] — one-sided pass-rate cut at 0.75 with 8 samples and Gaussian progressive batching (§5.1). Chapter-local extract: `excerpts/llama-nemotron.md`.
- [[olmo-3]] — Dolci-Think-RL composition (Table 20), offline filtering at 62.5% of 8 rollouts, active sampling, inflight weight updates, spurious-reward negative control.
- [[skywork-or1-rl-data]] — dataset card split sizes and model-aware difficulty estimation with retention percentages; the reuse ablations that measure `N_reuse`. Chapter-local extract: `excerpts/skywork-or1-rl-data.md`.
- [[deepmath-103k]] — the raw-pool contamination rates and the decontamination-then-difficulty-then-verification funnel. Chapter-local extract: `excerpts/deepmath-103k.md`.
- [[multi-domain-rlvr-data-centric]] — single-, dual-, and triple-domain RL results on Qwen2.5-7B and the curriculum-with-policy-refresh comparison. Chapter-local extract: `excerpts/multi-domain-rlvr-data-centric.md`.
- [[reasoning-or-memorization-rl-contamination]] — partial-prompt memorization measurements and the clean RandomCalculation benchmark. Chapter-local extract: `excerpts/reasoning-or-memorization-rl-contamination.md`.
- [[replay-buffer-rlhf]] — used only through the chapter-local verified extract `excerpts/replay-buffer-rlhf.md` of TRL's experimental replay trainer and its documentation; the library card's statement that frameworks replay prompts and never trajectories is contradicted by that code and is not used here.
- [[trl-grpo]] — the parent GRPO loss at commit a08e713: how a missing `old_per_token_logps` becomes a ratio of 1, and the logged metrics.
- [[reinforcement-learning-with-one-training-example]] — the one-example RLVR result and its historical-variance selection rule, read as a limit case. Chapter-local extract: `excerpts/reinforcement-learning-with-one-training-example.md`.
- [[rlvr-beyond-base-model]] — pass@k at large k as the coverage measurement for RL-trained policies.
