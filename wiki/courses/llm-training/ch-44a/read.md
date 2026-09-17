<!-- chapter: ch-44a
     track: rl
     kind: content
     title: Length in RL: Overlong Responses, Length Control, and Long-Context RL
     deps: [ch-44, ch-32b]
     sources: [[dapo]], [[verl-dapo-overlong-buffer]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[kimi-k1-5]], [[kimi-k1-5-recipe]], [[magistral]], [[magistral-recipe]], [[grpo]], [[dr-grpo]], [[demystifying-long-cot]], [[l1-lcpo]], [[overthinking-o1-like-llms]], [[sky-t1-flash-overthinking]], [[inverse-scaling-test-time-compute]], [[scaling-reasoning-losing-control-mathif]], [[qwenlong-l1]], [[qwenlong-l1-5]], [[loongrl]], [[loongrl-recipe]], [[memagent]], [[spell-self-play-long-context]], [[randomized-yarn]], [[longer-context-deeper-thinking]], [[minimax-01]], [[minimax-01-recipe]], [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]], [[nemotron-ultra-recipe]], [[deepswe]], [[demystifying-agentic-rl]], [[rollout-training-mismatch-tis]], [[qwen-long-context-synth]], [[context-length-alone-hurts]], [[ruler]], [[tokenskip]]
     figures: figures/length-reward-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL

> **Core insight.** Two lengths are set by hand in every RL run: the maximum response length, which bounds what a rollout may
> generate, and the maximum input length, which bounds what it may read. Both act as reward terms even when no length term is
> written down. A rollout cut off at the response limit has no answer, so a correctness verifier scores it as wrong, and in
> [[dapo]] replacing that score with a loss mask raised AIME 2024 avg@32 from 30 to 36 on Qwen2.5-32B (Table 1). Under a
> correctness-only reward, [[demystifying-long-cot]] reports response length rising until it reaches the context window and
> training accuracy then falling (§4.1, Figure 2). Explicit length control works at the RL stage: [[kimi-k1-5]] reports 60.8
> AIME 2024 pass@1 at 3,272 average tokens after long2short RL (§3.4), and [[l1-lcpo]] reports a 1.5B model holding a
> prompt-specified budget with about 3% mean error on math and 0.3-2.3% budget violation. On the input axis, RL at a moderate
> length transfers upward: [[loongrl]] trains at about 16K and improves RULER at 128K (Table 7), and [[memagent]] trains inside
> an 8K window and answers 3.5M-token questions at 71.09 accuracy (Table 1).
>
> **Guideline.** When rollouts are truncated at the response limit, do not leave them scored as wrong: mask their loss
> ([[dapo]] Overlong Filtering; `mask_truncated_completions` in [[allenai-olmo3-open-instruct-scripts]]) or shape the reward
> inside a buffer (`overlong_buffer.len = 4096`, penalty factor 1.0 in [[verl-dapo-overlong-buffer]]), because scoring a
> correct-but-unfinished chain as wrong penalizes exactly the behaviour the run is trying to produce. When a token budget must
> hold at inference, train the budget into the policy rather than cutting generation at serving time, because budget forcing
> lost 20-25 accuracy points to LCPO at 512 and 1024 tokens ([[l1-lcpo]] §5). When shortening a long-CoT model, add negatives
> that are short and wrong, not only long and correct, because length-only pairs cost 6.6 points of AIME 2024
> ([[sky-t1-flash-overthinking]] Ablations table). When the target input length is beyond the affordable rollout length, scale
> input length in phases ([[qwenlong-l1]] 20K → 60K; [[qwenlong-l1-5]] 20K/12K → 60K/20K → 120K/50K) or train at a moderate
> length with distractor-heavy data ([[loongrl]]). Otherwise, when neither a budget nor a long input is part of the deployment
> target, leave the length terms out and log truncation rate and length percentiles as diagnostics only.

## Why this chapter matters for a general-purpose model

Length enters RL twice, and the two entries are independent.

The **response axis** is the number of tokens a rollout may generate. It is a hyperparameter with a cost: generation time and
key-value cache memory grow with it, and the longest samples in a batch set the wall-clock time of a synchronous step
([[dapo]] §3.2). It is also a capability axis: reasoning RL raises accuracy and response length together
([[deepseek-r1]] §2.3; [[kimi-k1-5]] §3.3), and cutting the budget too low removes the behaviour being trained.

The **input axis** is the number of context tokens a prompt may carry. Mid-training gives a model a context window
([[ch-32b]]); the RL stage decides whether the model reasons over that window or only stores it. Long inputs change RL
dynamics directly: [[qwenlong-l1]] reports slower reward convergence, lower output entropy, and larger KL spikes at long
input lengths than at short ones (§1, Figure 2).

For a general-purpose model the two axes create four failure modes that this chapter addresses. (1) A reward that treats
truncation as failure teaches the model to stop early, or to keep producing tokens that are never rewarded. (2) A reward that
rewards length teaches repetition; [[demystifying-long-cot]] reports that "length rewards will be hacked with enough compute"
(§4.5). (3) A model trained only to be long spends tokens on questions that do not need them, and on some tasks longer
reasoning lowers accuracy ([[inverse-scaling-test-time-compute]]). (4) A model trained only on long inputs, or only on short
ones, loses the other regime: the memory-specialized checkpoint in [[qwenlong-l1-5]] gained 7.7 points of memory-agent MRCR
and lost 3.06 points of full-context average (Table 10), and raising the RL rollout limit from 1k to 8k tokens in
[[scaling-reasoning-losing-control-mathif]] raised math accuracy from 28.73 to 39.82 while lowering hard constraint accuracy
from 19.05 to 14.29 (Table 5).

The chapter sits at the RL stage, after process and outcome reward design ([[ch-44]]) and after context extension
([[ch-32b]]). It covers, in order: the response limit as a hyperparameter (§1), truncated rollouts and how their reward is
shaped (§2), why length grows and what stops it (§3), explicit length control and long2short (§4), RL over long inputs (§5),
regression in both length regimes (§6), and diagnostics (§7).

## §1 The maximum response length is a hyperparameter with a memory budget

**Definition.** The maximum response length `L_max` is the number of tokens a rollout may generate before the sampler stops
it. A response stopped this way is *truncated*: it ends without an end-of-sequence token and, for a reasoning task, without a
final answer.

**The measurable problem.** `L_max` is bounded above by the context window (prompt plus response must fit) and by the
key-value (KV) cache that in-flight sequences occupy. For a decoder with `n_layers` layers, `n_kv` key-value heads, head
dimension `d_head`, and `b` bytes per element, the cache holds

```
bytes_per_token = 2 · n_layers · n_kv · d_head · b        (keys and values)
bytes_per_sequence = bytes_per_token · (prompt_len + L_max)
```

Worked example with an illustrative configuration (`n_layers = 32`, `n_kv = 8`, `d_head = 128`, bf16 so `b = 2`):
`bytes_per_token = 2 · 32 · 8 · 128 · 2 = 131,072` bytes, or 128 KiB. At the DAPO setting of 2,048 prompt tokens and
`L_max = 20,480` ([[dapo]] §4.1), one sequence at full length holds `128 KiB · 22,528 = 2.75 GiB`. A rollout step of 512
prompts × 16 samples is 8,192 sequences; if every one ran to the limit, the cache would need about 22 TiB, which is why
rollouts are generated in waves and why the batch is reduced when the length limit is raised. [[magistral]] states the
trade-off directly: across Magistral Medium training the non-penalized length `l_max − l_cache` grows 16k → 24k → 32k while
the batch size falls 8k → 4k → 2k, "to limit KV-cache memory" (§5.2, [[magistral-recipe]]).

**Values used in published runs.** They differ by model type rather than by a common default: 32,768 tokens for DeepSeek-R1's
reasoning RL and 65,536 for R1-Zero after step 8.2k ([[deepseek-r1-recipe]]); 20,480 for DAPO ([[dapo]] §4.1); 49K with
overlong filtering for Nemotron 3 Nano's RLVR ([[nemotron-ultra-recipe]], NR §3.2.5); 10K of output at 20K-60K input for
QwenLong-L1 ([[qwenlong-l1]] §3.2); 4K during training for L1 ([[l1-lcpo]] §4). The Olmo 3 scripts make the rule explicit:
one family of scripts, "a response length matched to the model type (8,192 Instruct, 32,768 Think, 16,384 RL-Zero)"
([[allenai-olmo3-open-instruct-scripts-recipe]], starting-point paragraph). Every value is in the Recipe table below.

**Raising the limit mid-run is a known intervention.** DeepSeek-R1-Zero's limit was raised from 32,768 to 65,536 at step 8.2k,
and the report states that performance and response length "exhibit a significant jump at the 8.2k step"
([[deepseek-r1]] §2.1, Figure 1). The report does not separate the effect of the longer limit from continued training, so this
is a single observation, not a controlled comparison (Result, single study).

**Conditions and limits.** The numbers above are not transferable across model sizes: [[kimi-k1-5]] trained two model sizes on
the same data and found the smaller model reaching comparable accuracy with longer chains of thought while the larger one was
more token efficient (§3.5, Figure 8). A budget set from another report's model is therefore a guess.

**Implication for a general-purpose model.** `L_max` should come from the measured length distribution of the model being
trained (§7) and from the deployment budget, not from the context window. A limit far above the p99 response length costs
memory and throughput without changing the gradient; a limit near the median truncates a large share of rollouts and turns
§2's problem into the dominant training signal.

## §2 Truncated rollouts: what the reward does to unfinished reasoning

**Definition.** Truncation is generation stopped by `L_max`, not by the model. The verifier sees no answer, so a
correctness-only reward scores the rollout as wrong.

**The measurable problem.** [[dapo]] states it: "we assign a punitive reward to truncated samples. This approach may introduce
noise into the training process, as a sound reasoning process can be penalized solely due to its excessive length" (§3.4).
The rollout that is punished is, in expectation, the one with the longest reasoning, which on hard problems is the one most
likely to have been correct had it finished. The label is a **false negative** produced by the training configuration rather
than by the model.

**Four treatments, from the sources.**

1. **Punitive reward (the default).** The truncated sample keeps the wrong-answer reward (`−1` in [[dapo]] Eq. 7, `0` in
   binary-reward runs) and enters the loss like any wrong answer.
2. **Overlong filtering** — mask the loss of truncated samples so they contribute no gradient ([[dapo]] §3.4).
   [[allenai-olmo3-open-instruct-scripts]] implements the same idea as `mask_truncated_completions`, which "removes rollouts
   whose finish reason is not 'stop'"; the released Olmo 3 scripts set it True only in the RL-Zero code, instruction-following
   and general runs, and False in the Think, Instruct and RL-Zero math runs, where truncated rollouts stay in the batch with
   the verifier score.
3. **Soft overlong punishment** — a bounded, length-proportional penalty inside a buffer below the hard limit
   ([[dapo]] Eq. 13):

```
R_length(y) = 0                                      if |y| ≤ L_max − L_cache
              ((L_max − L_cache) − |y|) / L_cache    if L_max − L_cache < |y| ≤ L_max
              −1                                     if |y| > L_max
```

   `|y|` is the response length in tokens; `L_max` is the maximum generation length; `L_cache` is the buffer width. The term is
   added to the correctness reward. The verl implementation computes the same quantity from the realized length and clamps it
   at zero ([[verl-dapo-overlong-buffer]]):

```python
overlong_buffer_len = self.overlong_buffer_cfg.len
expected_len = self.max_resp_len - overlong_buffer_len
exceed_len = valid_response_length - expected_len
overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
reward += overlong_reward
```

4. **Compact filtering** for agents — [[deepswe]] extends overlong filtering to trajectories that hit the maximum number of
   environment steps or a 20-minute generation timeout, with the reason that an agent can pass tests by chance while editing
   random files afterwards, and rewarding such trajectories "leads to collapse" (§2.3, Figure 6).

**Worked example: what the label change does to the group.** Take one prompt, `G = 8` rollouts, binary reward, GRPO advantages
`A_i = (r_i − mean(r)) / std(r)` with the population standard deviation ([[grpo]]). Suppose 2 rollouts are correct and short,
5 are wrong and short, and 1 is truncated at `L_max` with correct reasoning so far.

- **Truncated scored 0:** `r = [1, 1, 0, 0, 0, 0, 0, 0]`, mean 0.25, std `√(0.25 · 0.75) = 0.433`. Correct rollouts get
  `A = 0.75/0.433 = +1.73`; every wrong rollout, including the truncated one, gets `A = −0.25/0.433 = −0.58`.
- **Truncated masked:** the group is `r = [1, 1, 0, 0, 0, 0, 0]`, mean 0.286, std `√(0.286 · 0.714) = 0.452`. Correct rollouts
  get `A = +1.58`, wrong rollouts `A = −0.63`, and the truncated rollout gets no gradient at all.
- **Soft punishment with `L_max = 20,480`, `L_cache = 4,096`:** a *correct* response of 18,000 tokens receives
  `(16,384 − 18,000)/4,096 = −0.39`, so its reward is `1 − 0.39 = 0.61` instead of `1`, and it stays a positive example.

The third case is the one the DAPO authors kept for the final run; the verl FAQ states that "most experiments in the paper,
including the best-performant one, are run without Overlong Filtering because it's somehow overlapping with Overlong Reward
Shaping" ([[verl-dapo-overlong-buffer]]). [figures/length-reward-explorer.html](figures/length-reward-explorer.html) lets you
set the group's lengths and correctness and read off the rewards and advantages under each treatment.

**The interaction with length normalization.** The per-token weight a rollout receives depends on the loss aggregation.
Sample-level GRPO divides each rollout's token losses by its own length `|o_i|`, so a truncated 20,480-token rollout with
`A = −0.58` puts `−0.58/20,480` on each of its tokens while a 2,000-token wrong rollout puts ten times more on each of its
own. [[dr-grpo]] names this the length bias: for a negative advantage, a longer response is penalized less per token because
`|o_i|` is larger, so the policy comes to prefer longer incorrect responses (§3.1). DAPO's token-level loss divides by
`Σ_i |o_i|` instead ([[dapo]] Eq. 12), which equalizes per-token weight and therefore makes the *total* push on a mislabeled
truncated rollout proportional to its length: with the numbers above, the truncated rollout carries about ten times the
negative gradient mass of a short wrong one. Masking or shaping the truncated reward matters more, not less, once the loss is
token-level (Interpretation; the sources report the ablation result, not this decomposition).

**Evidence.** [[dapo]] Table 1, Qwen2.5-32B base, AIME 2024 avg@32, cumulative single runs: naive GRPO 30, + overlong
filtering 36, + clip-higher 38, + soft overlong punishment 41, + token-level loss 42, + dynamic sampling 50. The comparison
point DeepSeek-R1-Zero-Qwen-32B is 47. Overlong filtering is the largest single step in that ladder, and it works by
*removing* a negative signal rather than adding one (Result, single study, math only).

**Conditions and limits.** The ladder is cumulative, so the +6 points for overlong filtering were measured on top of naive
GRPO only. [[demystifying-agentic-rl]] reuses overlong reward shaping in an agentic setting (§4.1, Eq. 9) but does not print
`L_cache`, and bundles the term with token-level loss and clip-higher in one recipe (GRPO-TCR), so its separate effect is not
measured; the ledger gives a 16,384-token maximum response length for the GRPO baseline (App. A.1).

## §3 Why response length grows, and what stops it

**Mechanism.** Under a correctness-only reward, no term prices tokens, so any behaviour that raises the probability of a
correct answer is reinforced regardless of the tokens it costs. [[dapo]] describes the useful side: "The increase in length
provides the model with a larger space for exploration, allowing more complex reasoning behaviors to be sampled and gradually
reinforced through training", and immediately adds that "length does not always maintain a continuous upward trend during
training. In some considerable periods, it can exhibit a trend of stagnation or even decline" (§4.3).

**The implicit penalty.** [[demystifying-long-cot]] ran Llama-3.1-8B and Qwen2.5-Math-7B with a +1-for-correct reward at a 16K
context window. Both grew until they hit the window, "which led to a decline in training accuracy due to CoTs exceeding the
allowable window size", and the exceed rate then "leveled off at a certain threshold below 1". The authors' reading is that
the window itself acts as a length penalty, and that "a trajectory might be penalized even without an explicit exceed-length
penalty due to reward or advantage normalization" (§4.1). This is the same mechanism as §2 seen from the other side: the
truncation label is the only thing stopping length growth in a run with no length term.

**Shaped alternative: the cosine reward.** [[demystifying-long-cot]] replaces the flat reward with one that depends on length
and correctness (§4.2, App. C.1):

```
R(C, L_gen) = CosFn(L_gen, L_max, r0^c, rL^c)   if C = 1 (correct)
              CosFn(L_gen, L_max, r0^w, rL^w)   if C = 0 (wrong)
              r_e                               if L_gen = L_max

CosFn(t, T, η_min, η_max) = η_min + 0.5 · (η_max − η_min) · (1 + cos(t π / T))
```

`C` is correctness; `L_gen` the generated length; `L_max` the maximum length; `r0` and `rL` the rewards at length 0 and at
`L_max` for correct (`c`) and wrong (`w`) responses; `r_e` the exceed-length penalty. Reported values: `r0^c = +2`,
`rL^c = +1`, `r0^w = −10`, `rL^w = 0`, `r_e = −10`, with an N-gram repetition penalty `P = −0.05`, `N = 40` (App. E.5.1).

Worked example with `L_max = 14,336` (the generation length in those runs). A correct answer at `L_gen = 7,168` receives
`1 + 0.5 · (2 − 1) · (1 + cos(π/2)) = 1.5`; the same answer at 1,000 tokens receives about 1.99, and the correct branch tends
to `rL^c = 1` as `L_gen` approaches `L_max`. A wrong answer at 7,168 receives `0 + 0.5 · (−10 − 0) · (1 + 0) = −5`; a wrong
answer at 1,000 tokens receives about −9.9, and the wrong branch tends to `rL^w = 0` near `L_max`. A response that actually
reaches `L_max` is truncated and takes the third branch instead, `r_e = −10`, whatever its content: the exceed-length penalty
is the only reward a truncated rollout can receive. Correct-and-short is best, wrong-and-short is worst: the ordering pays for extra
thinking only when the model is likely to be wrong.

**What the shape does.** The cosine reward gave "more stable (a) training accuracy and (b) response length" than the flat
reward (Figure 4), and the hyperparameters control the direction: "if the reward for a correct answer increases with CoT
length (`r0^c < rL^c`), the CoT length increases explosively", and "the lower the correct reward relative to the wrong reward,
the longer the CoT length", which the authors interpret as trained risk aversion (§4.3, Interpretation).

**Length rewards get hacked.** The same paper: "Length rewards will be hacked with enough compute ... but this can be
mitigated using a repetition penalty" (§4.5). The observed hack is repetition on hard questions, together with a falling count
of the branching word "alternatively" (Figure 10).

**A longer window costs training compute.** With the same number of training samples, the 8K-window run outperformed both the 4K and the
16K run, which the authors read as "models need more training compute to learn to fully utilize longer context window sizes"
(§4.4, Figure 6). Result (single study), Llama-3.1-8B, math prompts.

**Where the growth is visible in weights.** [[magistral]] reports a PCA direction over checkpoints along which mean reward and
output length grow together, with raw reward scaling logarithmically with mean output length between 1,500 and 8,000 tokens
(§7.1, Figures 8-9). Magistral's own length term is a penalty of 0 to −0.1 between `l_max − l_cache` and `l_max`, and −0.1
beyond ([[magistral-recipe]]); the penalty is small relative to the 0.9 correctness component.

**Implication for a general-purpose model.** Length growth is a side effect of the objective, not a goal. The two safe levers
are a bounded penalty near the limit (DAPO, Magistral) and a shaped reward whose ordering constraints are stated (cosine
reward). Both need a repetition diagnostic, because repetition is the cheapest way to satisfy any reward that pays for length
(see [[ch-43]] for the entropy side of the same failure).

## §4 Length control: budgets, long2short, and the overthinking literature

**Definition.** Length control is a training intervention that makes the deployed model's response length a controllable
quantity: either a fixed reduction (long2short) or a per-prompt budget (LCPO).

### §4.1 A length term inside the RL reward (Kimi k1.5)

[[kimi-k1-5]] adds a length reward to the task reward (§2.3.3). For `k` sampled responses of problem `x` with lengths
`len(i)`, `min_len = min_i len(i)` and `max_len = max_i len(i)`:

```
λ = 0.5 − (len(i) − min_len) / (max_len − min_len)
r_length(i) = λ              if the response is correct
              min(0, λ)      if the response is wrong
              0              for all i if max_len = min_len
```

Worked example, `k = 4`, lengths 1,000 / 2,000 / 3,000 / 5,000: `λ = 0.5 / 0.25 / 0.0 / −0.5`. A correct 1,000-token response
gains +0.5; a correct 5,000-token response loses 0.5; a wrong 1,000-token response gains nothing (`min(0, 0.5) = 0`), and a
wrong 5,000-token response loses 0.5. The term rewards short *correct* answers and never rewards a wrong answer for being
short. Training runs without the penalty first and then with a constant penalty; the report states the penalty can slow early
training and does not give the switch point or the weight ([[kimi-k1-5-recipe]]).

### §4.2 long2short: four ways to move a long-CoT model to a short budget

[[kimi-k1-5]] §2.4 compares four methods: weight-averaging merge of a long-CoT and a short-CoT model; shortest rejection
sampling (`n = 8` samples, keep the shortest correct one for SFT); DPO with the shortest correct response as chosen and both
wrong responses and correct responses 1.5× longer than the chosen one as rejected; and a separate long2short RL phase started
from the checkpoint with the best performance-per-token balance, with the length penalty and a reduced maximum rollout length.
Reported outcome: long2short RL has the highest token efficiency of the four (Figure 7); `k1.5-short w/ rl` reaches 60.8
AIME 2024 pass@1 (average of 8 runs) at 3,272 average tokens, and `k1.5-shortest` reaches 88.2 on MATH-500 at a token count
similar to other short models (§3.4). The report gives no per-method table, so the ranking rests on one figure.

### §4.3 Budget-conditioned RL (L1 / LCPO)

[[l1-lcpo]] puts the budget in the prompt. Each training prompt is augmented with "Think for `n_gold` tokens.", with `n_gold`
drawn uniformly from `U(100, 4000)`, and GRPO optimizes

```
LCPO-Exact:  r = I(y = y_gold) − α · |n_gold − n_y|
LCPO-Max:    r = I(y = y_gold) · clip(α · (n_gold − n_y) + δ, 0, 1)
```

`I(·)` is the correctness indicator, `n_y` the realized length, `α = 0.0003` the length weight, and `δ = 0.5` the offset that
keeps a correct answer with a small overrun above any wrong answer.

Worked example, `n_gold = 1,000`, `α = 0.0003`. A correct answer of 1,400 tokens: Exact gives `1 − 0.0003 · 400 = 0.88`; Max
gives `1 · clip(0.0003 · (−400) + 0.5, 0, 1) = 0.38`. A correct answer of 1,000 tokens: Exact 1.0, Max `clip(0.5) = 0.5`. A
wrong answer of any length: Exact `−0.12` at 1,400 tokens, Max `0`. Under Exact the model is pushed toward the requested
length from both sides; under Max it is free to finish early, and it loses reward only for overrunning.

Results, DeepScaleR-1.5B-Preview trained at a 4K context and evaluated at 8K: against S1-style budget forcing (stop at the
budget and insert "Final Answer"), L1 gains "over 100-150% relative and 20-25% absolute performance gains at both 512 and 1024
token budgets"; the accuracy-vs-log-length slope is 0.24 for L1 against 0.37 for S1. Mean length error is about 3% on math
benchmarks and 20-40% out of domain; L1-Max violates its budget by more than 500 tokens in 0.3-2.3% of cases (average 1.3%).
At matched short budgets L1-Max averages 46.2 against 41.0 for Qwen2.5-1.5B-Instruct, the non-reasoning model of the same
size (L1's own base is DeepScaleR-1.5B-Preview), and 47.8 against GPT-4o's 45.6 on
the same math set (Table 1). Result (single study), one 1.5B model, math training data.

### §4.4 Shortening without RL: preference pairs over the model's own samples

Two studies shorten a long-CoT model with preference optimization instead of RL, which matters here because both use
**length-defined negatives** (§ "Negative samples and negative feedback").

[[overthinking-o1-like-llms]] samples 10 responses per PRM12K question from QwQ-32B-Preview, keeps correct ones, and builds
positives of four kinds. Their statistics (Table 3: mean solution rounds / mean tokens / outcome efficiency / process
efficiency): shortest response 2.5 / 1051.3 / 69.8% / 80.3%; first-correct solutions (FCS) 1.1 / 681.0 / 99.5% / 99.1%;
FCS+Reflection 1.9 / 878.7 / 78.4% / 82.4%. The negative is the longest sampled response, chosen over the greedy response
because it "provides a clearer contrastive signal". On MATH500 (Table 4), SimPO with FCS positives cuts tokens from 2407.9 to
1016.0 but drops accuracy from 93.0 to 91.0; FCS+Reflection gives 92.8 accuracy at 1330.7 tokens. On AIME24 the same recipe
moves 46.7 → 43.3 accuracy and 9480.9 → 5154.5 tokens; on GPQA 59.6 → 59.1 and 3228.4 → 2085.7.

[[sky-t1-flash-overthinking]] applies SimPO to 10K self-generated pairs from Sky-T1-32B-Preview and reports accuracy held with
length cut: MATH500 88.6 → 88.6 at 2124 → 1417 tokens; AIME24 43.3 → 43.3 at 6881 → 4365; LiveCodeBench-Hard 17.9 → 17.9 at
14564 → 6199; MMLU 82.4 → 81.7 at 1087 → 799; GPQA Diamond 56.8 → 56.6 at 3503 → 2148. The ablation is the part to carry: with
only "longest correct vs shortest correct" pairs, AIME24 falls to 36.7 and LCB-Hard to 13.5; adding 1K pairs whose rejected
response is a *short incorrect* answer restores AIME24 to 43.3.

[[tokenskip]] is the supervised-compression alternative: LoRA fine-tuning on the model's own correct chains pruned by token
importance, with the compression ratio in the input (Qwen2.5-14B-Instruct GSM8K 313.11 → 180.68 tokens, 93.1 → 92.7 accuracy).

### §4.5 Longer is not uniformly better

[[inverse-scaling-test-time-compute]] constructs tasks where extending reasoning lowers accuracy: Claude Opus 4 falls from
near 100% to about 85-90% on Misleading Math, and DeepSeek R1 falls from 70% to 30% with five distractors under natural
overthinking (Fig. 3), while Claude Sonnet 4, o4-mini and o3 show no inverse scaling on MultiArith, ASDiv, GSM8K or GSM-IC
(App. F.3). The sign of the trend depends on model family and task (Table 1). The authors' recommendation is a measurement
protocol: evaluate at several reasoning budgets and across naturally sampled lengths. [[deepseek-r1]] reports the cost side of
the same point: on 366 problems from 2024 competitions R1 solves 61.8% with 8,793 thinking tokens on average, under 7,000 for
easy problems and over 18,000 for the hardest (E.4), and the authors note overthinking on simple questions (§6).

**Implication for a general-purpose model.** Budget control is a generality property, not only an efficiency property: a model
that spends 1,000 tokens on "2 + 3" ([[overthinking-o1-like-llms]] reports 1,953% more tokens than conventional models on that
question) is worse to deploy across mixed traffic, and a model whose accuracy falls as it thinks longer needs the budget as a
safety control rather than a cost control.

## §5 RL over long inputs: phases, distractors, and memory

The response axis is about how much the model writes; this section is about how much it reads. Four published RL designs
handle the cost of long-input rollouts in different ways.

**Progressive context scaling ([[qwenlong-l1]]).** RL is split into phases with increasing target input lengths, and in phase
`k` only examples with `L_{k−1} < |x| + |c| ≤ L_k` are trained (Eq. 9). Hard examples from earlier phases are re-inserted with
weights `diff(x, c) = 1 / mean({r_i})` — the inverse mean group reward, so an item the base model never solves gets the
largest weight (Eq. 10). A warm-up SFT stage on DeepSeek-R1 demonstrations precedes RL. Settings: 20K then 60K input, 10K
output, rollout 8, batch 128, LR 2e-6, 32×A100-80G. Results over seven DocQA benchmarks: R1-Distill-Qwen-32B 65.6 → SFT 68.7 →
QwenLong-L1-32B-DAPO 70.7; the 14B model 64.2 → 65.0 → 68.3 (Table 4). Two findings matter beyond the scores: single-stage RL
showed "fluctuating KL divergence and entropy collapse" where phased RL did not (§4.2, Figure 5c), and a long-context SFT model
started higher but gained only 0.3 points from RL, against 3.2 points when RL started from the short-context SFT model (§4.3).

**Input and output lengths scaled together ([[qwenlong-l1-5]]).** Because "as the input context length increases, the reasoning
content length exhibits a generally positive growth trend", the stages raise both: 20K input / 12K output → 60K / 20K → 120K /
50K, with difficulty filtering recomputed under the next stage's setting (§4.1). Stage-by-stage full-context averages:
61.92 base → 69.59 → 70.46 → 71.59, with MRCR rising 76.35 → 82.69 while short-input benchmarks such as DocMath and Frames
stay flat after Stage-1 (Table 10). The final model scores 71.82 average, +9.90 over its base, with MRCR +31.72 (Table 7).

**Train at a moderate length, evaluate long ([[loongrl]]).** KeyChain items hide the real question behind a UUID key-value
chain inside about 16K tokens of distractor documents, so the task requires locating the question before multi-hop reasoning.
GRPO with a two-way substring reward, `G = 8`, `β = 0.001`, max output 4,096, inputs about 16K ([[loongrl-recipe]]). LongBench
v1 multi-hop average: Qwen2.5-7B-Instruct 48.9 → 72.4, Qwen2.5-14B-Instruct 53.1 → 74.2 (Table 2). Training at 16K transfers
upward: NarrativeQA 32K-64K 42.4 → 57.2 (7B), RULER at 128K with YaRN 69.41 → 76.84 (7B) and 73.57 → 79.92 (14B) (Tables 3, 7).
Replacing KeyChain data with the same amount of plain long multi-hop QA gives 66.2 against 72.4 (Table 4), so the gain is
attributed to task difficulty rather than to input length alone. Caveat recorded on the card: three of the five evaluation
tasks come from the same datasets as the RL seeds, and no overlap check is reported.

**Move the length out of the context window ([[memagent]]).** The document is streamed in chunks and the model overwrites a
fixed-length memory after each chunk, so per-chunk compute is constant and total compute is linear in the number of chunks.
Training uses an 8K window allocated as 1,024 query + 5,000 chunk + 1,024 memory + 1,024 output. Multi-Conv DAPO assigns one
advantage — the answer conversation's reward minus the group mean, with no standard-deviation division — to every conversation
of the sample, which is how the memory-update steps receive credit. RULER-HotpotQA accuracy: RL-MemAgent-14B 80.47 at 7K and
71.09 at 3.5M tokens, against QwenLong-L1-32B 72.66 → 11.72 at 896K and Qwen2.5-Instruct-14B-1M 60.16 → 0.00 (Table 1).
[[qwenlong-l1-5]] adopts the same paradigm and reports the specialization cost: a memory-RL expert reached 20.34 on
MRCR 512K-1M but dropped the full-context average to 68.53, and SCE merging restored 71.18 while keeping 21.68 (Table 10).

**Removing the annotation bottleneck ([[spell-self-play-long-context]]).** One policy plays questioner, responder and verifier
over raw documents. The questioner's reward is a Gaussian in the responder's success rate centred at `µ = 0.5` with
`σ = 0.5/3`, zero when the success rate is 0 or 1, −0.5 for an ungrounded question and −1 for a formatting error (Eq. 7), so
the curriculum tracks the responder's ability. Training uses 16K maximum input and 4K (20K for reasoning models) maximum
output. Averages over six long-context QA benchmarks: Qwen2.5-14B 37.3 → 51.7 at 16K and 36.1 → 51.1 at 100K;
Qwen3-30B-A3B-Thinking 60.7 → 62.7 and 63.6 → 65.9, where the RLVR baseline on R1-synthesized data gains 0.0 and 0.9 on the
same model. pass@8 at 100K input is 74.5 with SPELL against 68.1 for RLVR and 66.9 for the base model, which is coverage
growth rather than sharpening (compare [[ch-43]] and [[ch-43a]]).

**Non-RL alternatives for the same goal.** [[randomized-yarn]] trains with YaRN encodings whose position indices are sampled
from a longer range under a length curriculum, using LoRA on fewer than 5K short-context examples, and reports BABILong OOD
averages of 90.3 against 83.6 for vanilla LoRA on Qwen2.5-7B-Instruct, with the largest gains at 128K; removing the curriculum
costs 6.2 points on MRCR. [[longer-context-deeper-thinking]] gives the prerequisite result from the SFT side: with identical
reasoning-SFT data, LLaMA3-8B-Instruct variants with better 32K needle scores reach higher MATH500 accuracy after SFT (54.80 at
θ×1 against 59.36 at θ×16, Table 3). Long-input ability and long-output reasoning are not independent capabilities.

**Where the RL stage sits relative to long context.** [[minimax-01]] alternates lengths across post-training instead of
running one long stage: SFT at 8,192 → training at 1,032,192 tokens with 50% long-context prompts → DPO at 8,192 → DPO at
1,032,192 → online RL at 8,192, with RoPE base 10M throughout (§5.6, Table 7, [[minimax-01-recipe]]). [[qwen-long-context-synth]]
reports the same asymmetry for Qwen2.5-1M: offline RL on pairs of at most 8,192 tokens still raised LongBench-Chat
(7B 7.32 → 8.08; 14B 8.56 → 8.76). [[kimi-k1-5]] instead pushes RL itself to a 128K context and pays for it with partial
rollouts: a rollout that exceeds its token budget is saved to a replay buffer and continued in the next iteration, so only the
current segment needs on-policy computation, and detected repetitions are terminated early (§2.6.2).

## §6 Regression runs in both directions

**Short-context regression after length-focused RL.**
- [[scaling-reasoning-losing-control-mathif]] continued RL of DeepSeek-R1-Distill-Qwen-1.5B with truncated overlong rollouts
  given no outcome reward, varying only the maximum rollout length. Hard-constraint accuracy / soft accuracy / math accuracy:
  1k 19.05 / 39.88 / 28.73; 2k 16.43 / 36.75 / 36.32; 4k 16.91 / 35.87 / 40.03; 8k 14.29 / 34.13 / 39.82 (Table 5). The length
  budget trades instruction following against math accuracy, and the trend is not monotone at every step.
- [[loongrl]] reports the retention side: MMLU +2.8 (7B) and +1.1 (14B), MATH-500 76.0 → 78.0 and 83.4 → 83.2, IF-Eval −0.3 and
  −2.6 (Table 2), while QwenLong-L1-32B scores lower MMLU than its own base (78.5 against 80.5).
- [[qwenlong-l1-5]] reports MMLU-PRO 81.03 → 81.33, AIME24 90.31 → 90.0, AIME25 82.81 → 86.46, GPQA-Diamond 75.88 → 76.78, and
  LongMemEval 60.80 → 76.40 after long-context RL (Table 8), which is the strongest published case that long-input RL need not
  cost short-context ability. It is a single model and the evaluations are the authors' own.
- [[spell-self-play-long-context]] reports short-context transfer on base models only: Qwen2.5-7B/14B/32B gain 6.93 / 8.74 /
  7.24 average points on a math and MMLU-Pro suite after long-context self-play (App. F.2, Table 7). Base models start low, so
  part of the gain may be answer formatting rather than reasoning (Interpretation).
- [[sky-t1-flash-overthinking]] shortened with math and code pairs and measured untrained domains: GPQA Diamond 56.8 → 56.6 and
  MMLU 82.4 → 81.7, with lengths down 39% and 17%.

**Long-context regression after short-context or specialized RL.**
- A memory- or agent-specialized RL stage can lose full-context ability: [[qwenlong-l1-5]] Table 10, 71.59 → 68.53, recovered
  to 71.18 by merging.
- Length-focused post-training can leave long-input ability untested: [[minimax-01]] states that "NIAH is inadequate for
  effectively monitoring the model's performance throughout the training process", because "NIAH metric performance reaches its
  peak score early on, specifically within the initial 128K training steps", so intermediate checkpoints were evaluated with
  more demanding tasks instead (§4.2). A long-context gate built on NIAH alone will not detect regression.
- [[context-length-alone-hurts]] is the measurement caution: accuracy falls by 13.9%-85% as input length grows even when the
  model recites the evidence exactly, and accuracy drops by a larger margin than retrieval scores in almost all task-length
  cells. A retrieval-only gate overstates what a long-context RL run has achieved.

**Gates that follow from this.** Run, after every length-focused RL stage: (1) a short-suite panel (MMLU or MMLU-Pro, GSM8K or
MATH, an instruction-following benchmark such as IF-Eval or MathIF) at the *deployment* generation budget, not at the training
budget; (2) a [[ruler]]-style suite at several lengths including one above the training length; (3) at least one non-retrieval
long task (LongBench-v2, MRCR, or a QA set), since NIAH saturates; (4) the same evaluations at two or three reasoning budgets,
following [[inverse-scaling-test-time-compute]]'s protocol.

## §7 Diagnostics: what to log next to entropy

| Quantity | Definition | What it tells you |
|---|---|---|
| Truncation rate | share of rollouts whose finish reason is the length limit, not the stop token | Rising toward a plateau below 1 is the implicit length penalty of [[demystifying-long-cot]] §4.1 acting |
| Response-length percentiles | p50 / p90 / p99 per step, split by correct and wrong | Sets `L_max` from data; a p99 near `L_max` means the truncation label dominates the negative gradient |
| Length by advantage sign | mean length of positive-advantage and negative-advantage rollouts | [[dr-grpo]] §3.1's length bias appears as wrong responses growing faster than correct ones |
| Reward composition | correctness term and length term logged separately | A falling total reward with a flat correctness term is a length-penalty effect, not a capability loss |
| Repetition statistics | N-gram repeat rate inside responses | The documented hack of any length-paying reward ([[demystifying-long-cot]] §4.5) |
| Zero-variance group share | share of prompts where all `G` rollouts get the same reward | Interacts with truncation: if every rollout truncates, the group has no gradient ([[dapo]] §3.2) |
| Validation accuracy next to length | both curves on one axis | [[dapo]] §4.3 uses the pair to decide whether a run is deteriorating; training reward alone "exhibits little correlation" with validation accuracy |

## Negative samples and negative feedback

**Which of the four kinds.** This chapter uses three of them. (1) A truncated rollout scored as wrong is **negative as
gradient** applied to a possibly correct trajectory — a configuration-induced false negative. (2) Overlong filtering and
`mask_truncated_completions` convert that sample into **no signal at all** (masking, not a negative). (3) Length-preference
pairs (Kimi long2short DPO, [[overthinking-o1-like-llms]], [[sky-t1-flash-overthinking]]) are **negative as gradient** where
the rejected response may be entirely correct and is dispreferred only for its length.

**Where the negatives come from and how they are labeled.** Truncation labels come from the sampler, not from a verifier; no
source reports the share of truncated rollouts that would have been correct, so the false-negative rate of this label is
unmeasured (Open question). Length-preference negatives come from the policy's own samples: the longest of 10 sampled
responses, after samples with no correct answer are discarded ([[overthinking-o1-like-llms]] §3.1), the longest correct of 8 and the shortest incorrect ([[sky-t1-flash-overthinking]]
Stage 1), or wrong responses plus correct responses 1.5× longer than the chosen one ([[kimi-k1-5]] §2.4).

**Mechanism.** For any objective that lowers the log-probability of a sampled sequence, the softmax logit gradient is
`∂ log p_y / ∂ z_j = 1[j = y] − p_j`, where `z_j` is the logit of token `j` and `p_j` its probability. Pushing down token `y`
lowers `z_y` and raises every other logit in proportion to `p_j`, so the removed mass goes to the currently most likely
alternatives, which need not be the chosen response. For a length-defined negative this is the exact risk: the rejected
trajectory is correct reasoning, so the mass removed from it is removed from reasoning behaviour that the model should keep.
The derivations are in [[ch-43a]]; this chapter applies them.

**Controls used in the sources.**
- *Mask instead of penalize* when the label is unreliable: [[dapo]] Overlong Filtering; `mask_truncated_completions` in the
  Olmo 3 RL-Zero code, instruction-following and general scripts ([[allenai-olmo3-open-instruct-scripts]]); compact filtering
  for timed-out or step-exhausted agent trajectories in
  [[deepswe]].
- *Bound the penalty*: the soft overlong punishment is at most 1 reward unit and reaches it only at the hard limit
  ([[dapo]] Eq. 13); Magistral's length penalty spans 0 to −0.1 against a 0.9 correctness term ([[magistral-recipe]]).
- *Never reward a wrong answer for being short*: Kimi's `min(0, λ)` for incorrect responses (§2.3.3); LCPO-Max multiplies the
  length term by the correctness indicator ([[l1-lcpo]] Eq. 2).
- *Add a second negative type*: [[sky-t1-flash-overthinking]] adds 1K short-incorrect pairs, which restored AIME24 from 36.7
  to 43.3 relative to length-only pairs.
- *Gate negatives on entropy*: AEPO in [[qwenlong-l1-5]] masks all negative-advantage samples when batch entropy exceeds
  `H_high` and reinstates them below `H_low`; ablations show clipping high-entropy negative sequences at 57.36 average against
  55.47-56.66 for low-entropy clipping, and the authors report that "removing too many negative gradient signals can cause
  entropy collapse" (§4.3, Table 4). AEPO gains 3.29 average points over GRPO on Qwen3-4B-Thinking-2507 (Table 5).

**Size of effect, stated honestly.** No source in this chapter measures the share of an improvement that is attributable to
negative gradients on long or truncated samples. The largest length-related effect reported is the *removal* of a negative:
+6 AIME points from masking truncated samples ([[dapo]] Table 1, one cumulative run). The AEPO comparison is an algorithm
change, not a decomposition. Claims of the form "the negative signal drives the gain" are not supported here.

**Diagnostics for this section.** Log chosen and rejected log-probabilities separately for length-preference training; log
length and reward split by advantage sign; track pass@k at large k, because length shortening can preserve pass@1 while
narrowing coverage — [[spell-self-play-long-context]] reports pass@8 explicitly for the opposite reason, to show coverage
growth.

**Effect on generality.** Length-only negatives produced underthinking on the hardest benchmarks in both self-training studies
(AIME24 36.7 and LCB-Hard 13.5 in [[sky-t1-flash-overthinking]]; FCS positives dropping MATH500 from 93.0 to 91.0 in
[[overthinking-o1-like-llms]]). Neither study measures refusal or calibration effects, so the abstention side of length
shaping is unmeasured (Open question).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DAPO Qwen2.5-32B base | 32B | RL | max generation; expected max; soft punish cache | 20,480; 16,384; 4,096 | arXiv:2503.14476v2 §4.1 ([[dapo]]) | verified 2026-09-15 | Table 1: +overlong filtering 30 → 36; +soft overlong punishment 38 → 41 (cumulative, one run) |
| DAPO Qwen2.5-32B base | 32B | RL | prompts × samples; mini-batch; LR; warmup; ε_low/ε_high | 512 × 16; 512; 1e-6 constant; 20 rollout steps; 0.2 / 0.28 | v2 §4.1 | verified 2026-09-15 | Table 1 (clip-higher 36 → 38) |
| verl DAPO recipe | n/a | RL | `data.max_response_length`; `overlong_buffer.len`; `penalty_factor` | 20480; 4096; 1.0 | verl@753aed3 `docs/algo/dapo.md` ([[verl-dapo-overlong-buffer]]) | verified 2026-09-15 | reproduction table: 52% AIME 2024 (16×8×H800) |
| DAPO reproduction script (Flash-RL post) | 32B | RL | max prompt / response; overlong buffer; penalty | 2,048 / 20,480; 4,096; 1.0 | [[rollout-training-mismatch-tis]] L20-24 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Zero | 671B MoE | RL | max response length | 32,768 before step 8.2k; 65,536 after | v2 §2.1 ([[deepseek-r1-recipe]]) | verified 2026-09-14 | §2.1: performance and length "exhibit a significant jump at the 8.2k step" |
| DeepSeek-R1 (Dev1 → Dev2) | 671B MoE | RL | temperature; outputs; max length; questions/step | 1; 16; 32,768; 32 | v2 §3.2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 | 671B MoE | eval-gate | max output tokens; decoding | 32,768; temperature 0.6, top-p 0.95, pass@1 over k samples | v2 D.1 | verified 2026-09-14 | greedy decoding gave higher repetition |
| Kimi k1.5 | not reported | RL | length reward | λ = 0.5 − (len(i) − min_len)/(max_len − min_len); λ if correct, min(0, λ) if wrong; weight not printed | arXiv:2501.12599v4 §2.3.3 ([[kimi-k1-5-recipe]]) | verified 2026-09-14 | no ablation numbers; authors state the penalty can slow early training |
| Kimi k1.5 | not reported | RL | length-penalty schedule; max RL context | off first, then constant (switch point not printed); 128k | §2.3.3, §3.3 | verified 2026-09-14 | Figures 5-6 (smaller internal model) |
| Kimi k1.5 short-CoT | not reported | RL | long2short RL | start from best performance/token checkpoint; length penalty; reduced max rollout length (value not printed) | §2.4, §3.4 | verified 2026-09-14 | §3.4: AIME 2024 60.8 at 3,272 tokens; highest token efficiency in Figure 7 |
| Kimi k1.5 short-CoT | not reported | preference | long2short DPO pairs | chosen shortest correct; rejected = wrong responses and correct responses 1.5× longer | §2.4 | verified 2026-09-14 | Figure 7 (below long2short RL) |
| Magistral Medium | not reported | RL | length penalty; non-penalized length; batch | 0 to −0.1 (Eq. 1); l_max − l_cache 16k → 24k → 32k; 8k → 4k → 2k sequences | arXiv:2506.10910v1 §2.2, §5.2 ([[magistral-recipe]]) | verified 2026-09-14 | reason given: KV-cache memory; no ablation of the penalty |
| Olmo 3 7B Think / Instruct / RL-Zero math | 7B | RL | response length | 32,768 / 8,192 / 16,384 | open-instruct olmo3 scripts ([[allenai-olmo3-open-instruct-scripts-recipe]]) | verified 2026-09-14 | matched to model type; no ablation reported |
| Olmo 3 RL runs | 7B, 32B | RL | truncated-rollout handling | `mask_truncated_completions` True in `7b_rlzero_code.sh`, `7b_rlzero_instruction_following.sh`, `7b_rlzero_general.sh`; False in the Think, Instruct and `7b_rlzero_math.sh` runs | `open_instruct/data_loader.py` L1126-1132; `7b_rlzero_code.sh` L70, `7b_rlzero_instruction_following.sh` L69, `7b_rlzero_general.sh` L77 ([[allenai-olmo3-open-instruct-scripts]]) | verified 2026-09-14 | no ablation reported |
| Nemotron 3 Nano 30B-A3B | 31.6B / 3.2B act. | RL | max generation length; truncation handling | 49K; overlong filtering | NR §3.2.5 ([[nemotron-ultra-recipe]]) | verified 2026-09-14 | no ablation reported |
| Demystifying-long-CoT runs (Llama-3.1-8B) | 8B | RL | generation length; cosine reward; repetition penalty | 14,336; r0^c +2, rL^c +1, r0^w −10, rL^w 0, r_e −10; P −0.05, N 40 | arXiv:2502.03373v1 App. E.5.1-E.5.3 ([[demystifying-long-cot]]) | verified 2026-09-15 | Figure 4 (stability vs classic reward); Figure 6 (8K > 4K and > 16K at equal samples) |
| L1-Exact / L1-Max (DeepScaleR-1.5B-Preview) | 1.5B | RL | training / eval context; steps; α; δ; n_gold range; LR; batch | 4K / 8K; 700 then 120; 0.0003; 0.5; U(100, 4000); 1e-6; 128 | arXiv:2503.04697v2 §4 ([[l1-lcpo]]) | verified 2026-09-15 | §5: about 3% mean length error on math; 1.3% average budget violation; "did not conduct extensive hyperparameter tuning" |
| QwenLong-L1-14B / -32B | 14B, 32B | RL | phases (input); max output; rollouts; batch; LR | 20K then 60K; 10K; 8; 128 (mini 32); 2e-6 | arXiv:2505.17667v2 §3.2 ([[qwenlong-l1]]) | verified 2026-09-15 | Table 4 (+4.1 / +5.1 average); Figure 5 (phased vs single-stage) |
| QwenLong-L1.5-30B-A3B | 30B-A3B | RL | stage input / output lengths | 20K/12K → 60K/20K → 120K/50K; memory-RL 128K input, 32K chunk, 15K memory; Stage-4 120K/50K | arXiv:2512.12967v1 §4.1, Figure 6 ([[qwenlong-l1-5]]) | conflict: Figure 6 prints 32K input for Stage-1 while §4.1 prints 20K | Table 10: 61.92 → 69.59 → 70.46 → 71.59 → (merge) → 71.82 |
| QwenLong-L1.5-30B-A3B | 30B-A3B | RL | algorithm; group size; batch; LR; sampling | AEPO (entropy-gated negative-gradient masking); G = 8; 128; 2e-6 constant; T 0.7, top-p 0.95 | §4.4, §5.1 | verified 2026-09-15 | Table 5: 56.07 (GRPO) → 59.36 (AEPO) on Qwen3-4B-Thinking-2507 |
| LoongRL-7B / -14B | 7B, 14B | RL | input ≈; max output; G; β; LR; batch | ~16K; 4,096; 8; 0.001; 1e-6 cosine; 512 / 256 | arXiv:2510.19363v2 §4.1 ([[loongrl-recipe]]) | verified 2026-09-14 | Table 4: KeyChain 72.4 vs plain long QA 66.2 (7B) |
| RL-MemAgent-7B / -14B | 7B, 14B | RL | 8K window split; algorithm; group size; rollout batch | 1,024 query + 5,000 chunk + 1,024 memory + 1,024 output; Multi-Conv DAPO, no std division; 16; 256 | arXiv:2507.02259v2 §3.1, App. A.3 ([[memagent]]) | verified 2026-09-15 | Table 1: 71.09 at 3.5M tokens |
| SPELL (Qwen2.5 and Qwen3 bases) | 7B-30B | RL | max input; max output; G; batch; LR | 16K; 4K (20K for reasoning models); 8; 128; 2e-6 constant | arXiv:2509.23863v4 §4.1 ([[spell-self-play-long-context]]) | verified 2026-09-15 | Table 1 (16K and 100K); Table 2 (verifier ablation −3.2) |
| Sky-T1-32B-Flash | 32B | preference | loss; pairs; LR; γ; β; epochs | SimPO; 10K (incl. 1K short-incorrect, 500 TACO); 5e-7; 0.3; 2.0; 1 | [[sky-t1-flash-overthinking]] Stage 1-3 | verified 2026-09-14 | Ablations table: math-only length pairs AIME24 36.7 vs 43.3 with short-incorrect pairs |
| QwQ-32B-Preview + SimPO FCS+Reflection | 32B | preference | samples per prompt; temperature; positive; negative | 10; 1.0; first-correct solution plus one reflection; longest sampled response | arXiv:2412.21187v2 §3.1-§3.2 ([[overthinking-o1-like-llms]]) | verified 2026-09-15 | Table 4: MATH500 92.8 at 1330.7 tokens vs 93.0 at 2407.9 |
| MiniMax-Text-01 | 456B / 45.9B act. | RL | post-training stage lengths | SFT 8,192 → 1,032,192 (50% long prompts) → DPO 8,192 → DPO 1,032,192 → online RL 8,192 | arXiv:2501.08313v1 §5.6, Table 7 ([[minimax-01-recipe]]) | verified 2026-09-14 | no ablation reported |

**Starting point for a small general-purpose run.** Every number here is a verified row above. For a reasoning RL run on a
32B-class base with a math or code verifier, the DAPO configuration is the most fully documented: maximum generation 20,480
tokens made of a 16,384 expected length plus a 4,096 soft-punish buffer with penalty factor 1.0, 512 prompts × 16 samples,
mini-batch 512, constant LR 1e-6 with 20 warm-up rollout steps, ε_low 0.2 and ε_high 0.28, token-level loss aggregation; it was
run on 16×8×H800 in verl. For an instruct-style model that must stay short, the Olmo 3 Instruct setting is the verified
alternative: 8,192-token responses, 64 prompts × 8 samples, LR 1e-6 constant, β = 0.0 (no KL term). That run leaves
`mask_truncated_completions` False, so truncation masking has to be added deliberately if it is wanted. For long inputs,
the QwenLong-L1 phase structure (20K then 60K input, 10K output, rollout 8, batch 128, LR 2e-6, 32×A100-80G, warm-up SFT
first) is the smallest verified long-context RL recipe; scale input and output together, as [[qwenlong-l1-5]] does, if the
target length exceeds 60K.

## Generalization lens

**(a) What increases breadth.**
- Removing the truncation false negative: +6 AIME points from masking alone ([[dapo]] Table 1), and the same mechanism keeps
  long correct reasoning as a positive example under soft punishment.
- Conditioning on a budget rather than enforcing it at serving time: LCPO's length control transferred to GPQA, LSAT and MMLU
  without those tasks being in the RL data ([[l1-lcpo]] §5, Figure 3).
- Phased input-length scaling with hard-example retention: +4.1 / +5.1 average points over seven DocQA benchmarks, where SFT
  alone gave +0.8 / +3.2 ([[qwenlong-l1]] Table 4).
- Task difficulty rather than raw length: KeyChain data beat equal-size plain long QA by 6.2 points ([[loongrl]] Table 4), and
  long-context RL transferred to AIME25 (+3.65) and LongMemEval (+15.60) in [[qwenlong-l1-5]] Table 8.
- Self-play curricula that keep the responder near a 0.5 success rate: pass@8 74.5 against 68.1 for RLVR on the same base
  ([[spell-self-play-long-context]] §4.3).

**(b) What causes narrowing or forgetting.**
- Rewards that pay for length: repetition hacking with enough compute, with falling branching frequency
  ([[demystifying-long-cot]] §4.5) and explosive growth when `r0^c < rL^c` (§4.3).
- Length-only negatives: underthinking on the hardest sets (AIME24 43.3 → 36.7; LCB-Hard 17.9 → 13.5,
  [[sky-t1-flash-overthinking]] Ablations table).
- Raising the rollout budget without an instruction-following gate: hard-constraint accuracy 19.05 → 14.29 from 1k to 8k
  ([[scaling-reasoning-losing-control-mathif]] Table 5).
- Specializing on one length regime: memory-RL cost 3.06 points of full-context average before merging
  ([[qwenlong-l1-5]] Table 10); QwenLong-L1-32B scores 78.5 MMLU against its base's 80.5 ([[loongrl]] Table 2).
- Assuming more thinking is better: inverse trends on constructed tasks and on Zebra Puzzles for all nine models tested under
  natural overthinking ([[inverse-scaling-test-time-compute]] Table 1).

**(c) How generality is measured at this stage.** Short suite at the deployment budget; [[ruler]]-style retrieval plus at least
one non-retrieval long task at several lengths, including above the training length; instruction-following benchmark; pass@1
and pass@k at large k; accuracy at two or three budgets per [[inverse-scaling-test-time-compute]]; token counts reported next
to every accuracy, as [[sky-t1-flash-overthinking]] and [[l1-lcpo]] do. Known measurement errors: NIAH saturates early
([[minimax-01]] §4.2); retrieval scores overstate long-context ability ([[context-length-alone-hurts]]); evaluation-set overlap
with RL seeds is unchecked in [[loongrl]]; L1's length errors grow to 20-40% out of domain, so a budget verified on math does
not transfer unverified.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Truncated rollouts left scored as wrong | Truncation rate plateaus below 1 while training accuracy falls; length pinned at `L_max` | Log truncation rate and accuracy on terminated responses separately ([[demystifying-long-cot]] §4.1; [[dapo]] §3.4) |
| `L_max` chosen from the context window, not from data | Rollout step time dominated by a few samples; KV-cache pressure forces batch cuts | Compare p99 response length with `L_max`; the Magistral schedule cuts batch as `l_max` rises ([[magistral-recipe]]) |
| Length penalty enabled from step 0 | Reward flat early, short degenerate answers | Kimi trains without the penalty first, then adds a constant one ([[kimi-k1-5]] §2.3.3) |
| Length reward that pays more for longer correct answers | Length grows without accuracy; stop-token probability falls | Inspect `r0^c` vs `rL^c`; [[demystifying-long-cot]] §4.3 reports explosive growth for `r0^c < rL^c` |
| No repetition control with any length-paying reward | N-gram repeat rate rises; branching words fall | Add the N-gram penalty (`P = −0.05`, `N = 40`) and log repeat rate ([[demystifying-long-cot]] §4.5) |
| Budget control trained on one domain only | Budget held on math, missed elsewhere | Measure length error out of domain; L1 reports 20-40% ([[l1-lcpo]] §5) |
| Shortening with long-correct negatives only | Accuracy falls on the hardest benchmarks while easy sets look fine | Add short-incorrect pairs; compare AIME-class scores ([[sky-t1-flash-overthinking]] Ablations) |
| Sample-level loss with heavy truncation | Long wrong answers grow faster than correct ones | Log length by advantage sign; [[dr-grpo]] §3.1 length bias |
| Long-context RL judged by NIAH | Needle score saturates while real tasks do not improve | Use LongBench-v2 / MRCR / CorpusQA-style tasks ([[minimax-01]] §4.2; [[qwenlong-l1-5]] Table 7) |
| Specialized long-context stage shipped without a merge or mixed stage | Full-context average drops after the specialized stage | Compare stage checkpoints; merge and re-measure ([[qwenlong-l1-5]] Table 10) |
| Raising rollout length without re-running instruction-following | Math up, constraint compliance down | MathIF or IF-Eval at each budget ([[scaling-reasoning-losing-control-mathif]] Table 5) |

## Check your understanding

1. A run uses binary rewards, token-level loss aggregation, and no length term. Truncation rate is 18% and rising. Explain,
   using the advantage computation and the per-token weighting, why the truncated rollouts now carry more negative gradient
   mass than the short wrong ones, and what changes if the loss is aggregated per sample instead.
2. DAPO reports that overlong filtering alone raised AIME 2024 avg@32 from 30 to 36, and that the soft overlong punishment was
   kept for the final run instead. Give a causal account of why both can be improvements over the punitive default, and state
   the experiment that would decide which to use for a new model.
3. [[demystifying-long-cot]] reports that the context window acts as a length penalty even with no explicit penalty term.
   Explain the path from a truncated rollout to downward pressure on length when rewards are group-normalized, and say why the
   exceed rate plateaus below 1 instead of reaching it.
4. Kimi's length reward gives correct responses `λ` and wrong responses `min(0, λ)`. Explain what would go wrong if wrong
   responses also received the full `λ`, in terms of what the policy could learn to do on unsolved problems.
5. L1-Exact scores about 1 point below the same model trained without length constraints, and the gap is concentrated on AIME.
   Explain this with the difference between LCPO-Exact and LCPO-Max, and predict which of the two would degrade less on a
   benchmark whose problems vary widely in difficulty.
6. LoongRL trains at about 16K input and improves RULER at 128K, while QwenLong-L1 scales input length in phases up to 60K.
   Give the conditions under which each strategy is preferable, and describe how you would tell from training curves that the
   16K strategy was not transferring.
7. QwenLong-L1.5's memory-RL stage raised memory-agent MRCR and lowered the full-context average, and merging restored the
   average while keeping the memory gain. Explain what this ordering implies about where the two abilities are stored, and
   what the result would have to look like for "merging recovers everything" to be a general claim rather than one run.
8. A team shortens a reasoning model with length-preference pairs, keeps pass@1 on MATH500, and ships it. State two
   measurements that would reveal a loss the pass@1 number hides, and explain the mechanism each one targets.

## Connections

- **Previous:** [[ch-44]] — Process Supervision and Verifiable Rewards. The reward this chapter reshapes is the verifiable
  outcome reward defined there; truncation is the case where the verifier has nothing to check.
- **Next:** [[ch-44b]] — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. Length terms behave
  differently when the reward is a model judge rather than a verifier, since judges have their own length preference.
- **Builds on:** [[ch-32b]] — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (the window this
  chapter spends); [[ch-40]] — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO (the loss aggregation and advantage
  formulas used in §2); [[ch-43]] — Entropy, Output Diversity, and KL Control in RL (entropy collapse next to length growth);
  [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages (the
  derivations applied in this chapter's negatives section).
- **Feeds:** [[ch-45b]] — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability; [[ch-45c]] — Context
  Management for Long-Horizon Agents; [[ch-32c]] — Claimed versus Effective Context Length and Long-Context Evaluation
  (the measurement suites used in §6); [[ch-29a]] — Long-Document Synthesis for Continued Pretraining and Long-Context SFT
  (where the long-input RL data of §5 comes from).

## Sources

- [[dapo]] — overlong filtering and soft overlong punishment, Eq. 13, Table 1 ladder, token-level loss, length as a training
  metric, RL settings.
- [[verl-dapo-overlong-buffer]] — the released `overlong_buffer` configuration and penalty code, and the FAQ on why overlong
  filtering is not implemented there.
- [[deepseek-r1]], [[deepseek-r1-recipe]] — max response length 32,768 → 65,536 at step 8.2k, evaluation budget, thinking-token
  counts by problem difficulty, overthinking on simple questions.
- [[kimi-k1-5]], [[kimi-k1-5-recipe]] — length reward, long2short methods and results, partial rollouts at 128K, model-size
  token-efficiency comparison.
- [[magistral]], [[magistral-recipe]] — length penalty magnitude, `l_max − l_cache` schedule with batch-size reduction, the
  reward-versus-length direction in weight space.
- [[dr-grpo]] — the length bias of per-sample normalization for negative advantages.
- [[demystifying-long-cot]] — length instability under a correctness-only reward, cosine reward formula and hyperparameters,
  context-window comparison, length reward hacking and the repetition penalty.
- [[l1-lcpo]] — LCPO-Exact and LCPO-Max rewards, budget adherence, comparison with budget forcing, short-CoT results.
- [[overthinking-o1-like-llms]] — outcome efficiency metric, FCS / FCS+Reflection positives, longest-response negatives,
  Table 4 accuracy-and-length results.
- [[sky-t1-flash-overthinking]] — SimPO length pairs, the short-incorrect ablation, accuracy and length on seven benchmarks.
- [[tokenskip]] — supervised alternative for controllable chain-of-thought compression.
- [[inverse-scaling-test-time-compute]] — tasks where longer reasoning lowers accuracy and the multi-budget evaluation
  protocol.
- [[scaling-reasoning-losing-control-mathif]] — RL rollout-length sweep against instruction-following accuracy (Table 5).
- [[qwenlong-l1]] — long-context RL dynamics, curriculum phases, difficulty-aware retrospective sampling, hybrid reward, SFT
  versus RL trade-off.
- [[qwenlong-l1-5]] — joint input/output length schedule, AEPO entropy-gated negative gradients, memory-RL specialization and
  SCE merging, out-of-distribution generality table.
- [[loongrl]], [[loongrl-recipe]] — KeyChain data, training at 16K with evaluation to 128K, short-context retention, RL
  settings.
- [[memagent]] — 8K training window with chunked memory, Multi-Conv DAPO advantage sharing, results to 3.5M tokens.
- [[spell-self-play-long-context]] — self-play roles and the Gaussian questioner reward, 16K → 100K transfer, pass@8, short-
  context transfer on base models.
- [[randomized-yarn]] — randomized YaRN positions with a length curriculum as a supervised route to length generalization.
- [[longer-context-deeper-thinking]] — long-context ability as a prerequisite for long reasoning after SFT.
- [[minimax-01]], [[minimax-01-recipe]] — post-training stages alternating 8,192 and 1,032,192 tokens, NIAH saturation.
- [[qwen-long-context-synth]] — short-only offline RL after long-context SFT, with LongBench-Chat gains.
- [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]] — `mask_truncated_completions` and
  response lengths by model type.
- [[nemotron-ultra-recipe]] — 49K generation length with overlong filtering in a 2025-12 RLVR recipe.
- [[deepswe]] — compact filtering for agent trajectories that time out or exhaust steps.
- [[demystifying-agentic-rl]] — overlong reward shaping reused in agentic RL, with lengths unreported.
- [[rollout-training-mismatch-tis]] — a released DAPO reproduction script with the overlong-buffer settings.
- [[grpo]] — the group-normalized advantage used in the worked examples.
- [[ruler]] — the synthetic long-context suite used as a length-wise gate.
- [[context-length-alone-hurts]] — accuracy falling with input length even under perfect retrieval.
