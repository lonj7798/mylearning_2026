<!-- chapter: ch-45b
     track: rl
     kind: content
     title: Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability
     deps: [ch-45a, ch-29d]
     sources: [[search-r1]], [[deepswe]], [[deepswe-recipe]], [[loop-appworld]], [[gigpo-verl-agent]], [[turn-level-credit-assignment]], [[sweet-rl]], [[ragen-starpo]], [[simpletir]], [[agentrl]], [[agentrl-recipe]], [[agentgym-rl]], [[agentgym-rl-recipe]], [[skyrl-agent]], [[skyrl-agent-recipe]], [[kimi-researcher]], [[minimax-m2-aligning-to-what]], [[minimax-m2-interleaved-thinking]], [[demystifying-agentic-rl]], [[deepseek-v3.1]], [[glm-5]], [[fission-grpo]], [[grpo]], [[dr-grpo]], [[rloo]], [[ppo]], [[r2e-gym]], [[bfcl]]
     figures: figures/rollout-gradient-anatomy.html, figures/gigpo-anchor-grouping.html
     revised: 2026-09 (generality revision)
-->

# Chapter 45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability

> **Core insight.** When the policy acts many times in an environment, the training sequence contains tokens the
> policy did not emit, and the environment decides what those tokens are. Three consequences follow, each with a
> measured effect. (1) Environment tokens must be excluded from the loss: with retrieved-token masking a
> Qwen2.5-7B-base PPO run averages 0.431 exact match over seven QA datasets against 0.343 without it
> ([[search-r1]], Table 4). (2) The outcome reward alone is enough to train strong agents when the advantage
> baseline and the filtering are right: DeepSWE-Preview reaches 42.2% Pass@1 on SWE-Bench-Verified from Qwen3-32B
> with RL only, no KL term and no entropy loss ([[deepswe]], §4); LOOP reaches 71.3 task goal completion on
> AppWorld Test-Normal from 72 training tasks ([[loop-appworld]], Table 1). (3) Multi-turn RL has failure modes
> that single-turn RL does not: tool feedback pushes the policy off its pre-training distribution, later turns
> emit low-probability tokens, and the gradient norm explodes; masking feedback tokens does not prevent this, but
> discarding whole trajectories that contain a turn with neither a tool call nor an answer raises AIME24 from 20.8
> to 50.5 ([[simpletir]], Table 2).
>
> **Guideline.** When the environment returns tokens into the context, mask them from the loss and from the KL
> term, and feed the trainer the exact token ids the inference engine produced, because re-tokenizing the text
> "can corrupt step alignment between actions and rewards/advantages" ([[glm-5]], §4.1.2). When trajectories can
> end by context limit, step limit, environment crash or generation timeout, remove them from the gradient rather
> than scoring them as failures, because they measure the harness and not the policy ([[deepswe]], §2.3;
> [[skyrl-agent]], §4.2; [[glm-5]], §4.1.2). When a group of rollouts shares a task and an initial state, use a
> leave-one-out baseline without dividing by the return standard deviation, because that division cost 9.4 points
> of AppWorld Test-Normal task goal completion ([[loop-appworld]], Table 1) and is the difficulty bias that
> [[dr-grpo]] removes. When the environment revisits states across rollouts, add a step-level advantage over
> actions taken from the same state, because that adds more than 12 points of ALFWorld success over GRPO at 1.5B
> and 7B for 0.01 s of grouping plus 0.53 s of advantage arithmetic per iteration, against 362.83 s of shared
> cost ([[gigpo-verl-agent]], Table 1, §5.6).

## Why this chapter matters for a general-purpose model

Every RL chapter up to here assumed one prompt in, one response out, one reward. [[ch-40]] built group baselines
on that assumption, [[ch-43]] analysed entropy under it, and [[ch-44]] attached verifiers to it. An agent breaks
the assumption in one specific way: between the policy's own tokens, an environment inserts tokens the policy did
not choose and cannot predict, and it does this repeatedly. That single change makes the transition stochastic,
makes the sequence partly non-policy text, and stretches the horizon from one action to tens of actions.

The stage sits here: pre-training → mid-training → SFT → preference optimization → reasoning RL →
**agentic RL** → general RL → evaluation. [[glm-5]] runs the post-SFT part of that order without a separate
preference-optimization stage: "a sequential Reinforcement Learning pipeline—starting with Reasoning RL,
followed by Agentic RL, and finishing with General RL" (§1). [[deepseek-v3.1]] does not separate the stages at
all: "we merge reasoning, agent, and human alignment training into one RL stage" (§3).

For a generally capable model, agentic RL is the stage with the widest spread between the best and the worst
outcome. On the positive side, SWE-only RL raised Terminal-Bench from 13.75 to 16.25, BrowseComp-Plus accuracy
from 18.1 to 19.4 and WebArena from 15.8 to 17.0 for the same 32B checkpoint ([[skyrl-agent]], Table 3), and RL
on synthetic general-agent tasks improved three environments that were never in the RL mixture
([[deepseek-v3.1]], §4.3). On the negative side, [[agentgym-rl]] states that its trained agents "perform well
within in-domain settings" and lists transfer to novel environments as future work (§7); [[ragen-starpo]] reports
that reasoning traces shrink during outcome-only multi-turn RL until the model produces "hallucinated reasoning"
(§4.4); and [[minimax-m2-aligning-to-what]] reports from deployment that "the same model can feel brilliant in
one framework and useless in another". The mechanics in this chapter determine which of the two you get.

## §1 The agentic MDP: what actually changes

**Definition.** In *single-turn LM-RL* the policy emits one token sequence `y` for prompt `x`, and the episode
ends. In *multi-turn agentic RL* the policy emits an action, an environment returns an observation, and the pair
is appended to the context; the episode ends when the policy stops, a verifier fires, or a limit is reached.

Three formalizations appear in the sources, and they differ in what counts as an action.

1. **Turn-level MDP with token-level execution.** [[gigpo-verl-agent]] defines a trajectory
   `τ = {(s_1, a_1, r_1), …, (s_T, a_T, r_T)}` where each action `a_t ∈ V^n` is a whole token sequence and the
   policy is `π_θ(a_t | s_t, x)` (§3).
2. **Hierarchical MDP.** [[simpletir]] splits the same object into a high-level MDP over turns, with state
   `S_k = (q, l_0, f_0, …, l_{k−1}, f_{k−1})`, transition `S_{k+1} = S_k ∘ (l_k, f_k)` and a terminal reward
   `R(o)`, and a low-level MDP over tokens with `R_L = 0` and a deterministic append transition (§2.1). `l_k` is
   the model's turn-`k` response, `f_k` the tool feedback, and `∘` is concatenation.
3. **POMDP.** [[loop-appworld]] models the digital environment as partially observed, because the agent sees API
   responses rather than the environment's state, which reaches 30M text tokens in AppWorld (§4.1, §5.1).

The three formalizations agree on the operational point: **the transition is stochastic and the randomness is not
the policy's sampling noise.** In single-turn RL, `p(y | x)` is the policy; in agentic RL the sequence
probability factorizes over the policy's tokens only, and the environment supplies the rest. [[loop-appworld]]
writes this explicitly: environment tokens are appended to the state but only LLM-emitted tokens `a(x)` enter the
likelihood and the gradient (Eq. 6-8).

**Why this matters for the loss.** A rollout is one long sequence with two kinds of tokens interleaved. Consider
a five-turn SWE rollout in which the model emits 900, 1,100, 700, 1,500 and 300 tokens, and the tools return
2,000, 3,500, 1,200, 800 and 0 tokens. The model emitted 4,500 tokens; the sequence is 12,000 tokens. Environment
text is 62.5% of the sequence. Two facts follow:

- If the environment tokens enter the loss, 62.5% of the gradient trains the model to predict its own tool
  outputs. That is the quantity [[search-r1]] ablates (§2 below).
- If the loss is averaged over the sequence length rather than over the mask, the same per-token gradient is
  scaled by `4,500 / 12,000 = 0.375` for this rollout and by a different factor for the next one. Trajectories
  with terse tool output then receive a larger effective learning rate than trajectories with verbose output,
  for reasons unrelated to their quality. [[simpletir]] normalizes by `Σ_t m_{i,t}`, the number of unmasked
  tokens, for this reason (Eq. 1).

The figure [rollout-gradient-anatomy.html](figures/rollout-gradient-anatomy.html) lets you set the number of
turns, the observation lengths, the number of successful rollouts and the filtering policy, and shows which
tokens carry gradient and what advantage each trajectory receives.

## §2 Rollout fidelity: masking, token-in-token-out, template

### 2.1 Observation masking

**Problem.** Training on environment tokens changes the objective from "act well" to "act well and predict the
environment", and the environment text is out of the model's pre-training distribution.

**Mechanism.** Every source in this chapter applies the same fix: a binary mask over token positions.
[[search-r1]] writes it as an indicator inside the PPO and GRPO objectives, "I(y_t) is the token loss masking
operation such that I(y_t)=1 if y_t is a LLM generated token" (§3.1, Eq. 2), and applies the same mask inside the
GRPO KL term. [[simpletir]] writes `m_{i,t} = 1` if the token belongs to a model response (Eq. 1). [[glm-5]]
states it in prose: "only model-generated tokens are used for optimization, and the environment feedback is
ignored in loss computation" (§4.1). [[deepswe]] masks the environment observations that appear as ChatML user
messages (§2.3).

**Evidence (Result, single study).** [[search-r1]] trains Qwen2.5-7B-base with PPO on merged NQ and HotpotQA and
evaluates exact match on seven QA datasets (Table 4):

| Setting | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | Avg |
|---|---|---|---|---|---|---|---|---|
| With retrieved-token mask | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | **0.431** |
| Without mask | 0.388 | 0.567 | 0.391 | 0.325 | 0.321 | 0.108 | 0.304 | **0.343** |

The masked run is higher on all seven datasets. This is one model, one algorithm and one tool; no source repeats
the ablation at another scale.

**Condition and limit.** Masking removes the environment tokens from the *gradient*; it does not remove their
effect on the *policy*, because they remain in the context that conditions every later turn. [[simpletir]] states
this directly: masking "is insufficient. The distributional drift it induces contaminates subsequent model
generations" (§3.1). §5 takes up what that drift does.

### 2.2 Token-in-token-out

**Definition.** *Token-in-token-out* (TITO) means the trainer consumes the exact token ids the inference engine
emitted. *Text-in-text-out* means the trainer receives decoded text and re-tokenizes it.

**Problem.** Re-tokenization is not the identity. [[glm-5]] states the failure: "re-tokenization can introduce
subtle mismatches in token boundaries, whitespace/normalization handling, truncation, or special-token placement,
which in turn can corrupt step alignment between actions and rewards/advantages—especially when rollouts are
streamed, truncated, or interleaved across many actors" (§4.1.2). The report implements a "TITO Gateway that
intercepts all generation requests from rollout tasks and records each trajectory's token IDs and metadata".
[[skyrl-agent]] records input tokens, output tokens and log-probabilities per LLM call for the same reason, and
states that the transition format "naturally supports token-in/token-out processing, eliminating off-policy drift
caused by re-tokenization or text postprocessing" (§3.3).

This is a multi-turn-specific problem. In single-turn RL a re-tokenization mismatch affects one boundary; in a
40-turn rollout it affects 40 boundaries, and each one shifts the mask by a token or two, so the mask stops
matching the actions.

A second, related quantity is which log-probabilities the trainer treats as `π_old`. [[loop-appworld]] recomputes
log-probabilities under the generating policy rather than taking them from vLLM (App. D). [[glm-5]] does the
opposite on purpose: it reuses the rollout log-probabilities as the behavior policy and drops `π_θ_old` entirely,
`r_t(θ) = exp(log π_θ(a_t|s_t) − log π_rollout(a_t|s_t))`, with tokens outside `[1 − ε_ℓ, 1 + ε_h]` masked from the
gradient (§4.1.2, Eqs. 3-5). Both are defensible; recording the wrong one silently is not.

### 2.3 Template fidelity across turns

**Problem.** The chat template decides where a turn starts, where thinking is placed, and whether earlier
thinking survives into the next turn. A change between training and deployment produces no error.

Three sources treat previous-turn reasoning as part of the state.

- [[skyrl-agent]]: "The Qwen3 chat template is modified to preserve the 'thinking' from previous turns,
  maintaining reasoning continuity across steps" (§4.2); the released config sets `qwen3_acc_thinking: true`
  ([[skyrl-agent-recipe]]).
- [[deepseek-v3.1]]: "Historical reasoning content is discarded only when a new user message is introduced to
  the conversation. If only tool-related messages (e.g., tool outputs) are appended, the reasoning content is
  retained throughout the interaction", and "when reasoning traces are removed, the history of tool calls and
  their results remains preserved in the context" (§3.2.1). The report recommends non-thinking mode for frameworks that
  deliver tool results as user messages (§3.2.1).
- [[minimax-m2-interleaved-thinking]] names the deployment cause: "the widely-used OpenAI Chat Completion API
  does not support passing reasoning content back in subsequent requests". [[minimax-m2-aligning-to-what]] states
  the same requirement from the training side: "its context is its memory. For best performance, you must retain
  the full session history, including the thinking steps".

**Evidence (Result, single study; inference-time comparison, not an RL ablation).**
[[minimax-m2-interleaved-thinking]] evaluates one model under two history-construction policies:

| Benchmark | Prior-round thinking retained | Discarded | Δ |
|---|---|---|---|
| SWE-Bench Verified | 69.4 | 67.2 | +2.2 |
| τ² | 87 | 64 | +23 |
| BrowseComp | 44.0 | 31.4 | +12.6 |
| GAIA | 75.7 | 67.9 | +7.8 |
| xBench | 72.0 | 66.0 | +6.0 |

The spread across benchmarks (from +2.2 to +23) is itself the point: the same template change is close to free on
one benchmark and decisive on another, so a single benchmark will not tell you whether your harness matches your
training template. Seeds and variance are not reported.

## §3 Outcome-only recipes that work

**Definition.** An *outcome-only* recipe gives the trajectory one scalar at the end — tests pass, answer matches,
task goal completed — and derives every token's advantage from it.

### 3.1 The common shape

Three independent SWE and tool-use recipes converge on nearly the same configuration:

| | [[deepswe]] (GRPO++) | [[skyrl-agent]] (SA-SWE-32B) | [[loop-appworld]] (LOOP token) |
|---|---|---|---|
| Base | Qwen3-32B, no SFT | Qwen3-32B, no SFT | Qwen2.5-32B-Instruct, LoRA r=16 |
| Advantage baseline | leave-one-out | leave-one-out | leave-one-out (Eq. 3) |
| Divide by return std | no (from [[dr-grpo]]) | no | no |
| KL term | none | disabled | none in the objective |
| Entropy loss | none | disabled | not reported |
| Importance weight | per token | per token | per token |
| Truncated / timed-out rollouts | masked (compact filtering) | masked; reward and advantage unchanged | rollouts with `|Â| < 0.01` dropped |
| Reward | 1 if selected tests pass, else 0 | R2E-Gym tests | fraction of unit tests passed, in [0,1] |
| Result | 42.2% Pass@1 SWE-Bench-Verified, 71.0% Pass@16 | 39.4% Pass@1 under a simple ReAct scaffold | 71.3 Test-N TGC, 45.7 Test-C TGC |

[[glm-5]] states the same objective for its agentic RL stage without the std term:
`L(θ) = E_{x∼D}[(1/K) Σ_{i=1..K} (r(x, y_i) − r̄(x))]` with `r̄(x) = (1/K) Σ_i r(x, y_i)` (§4.1), where `K` is the
number of traces per problem and `r` the trajectory reward. [[deepseek-v3.1]] defines the advantage
"by subtracting the average reward of the group from the reward of output `o_i`", `Â_{i,t} = R_i − mean(R)`, with
no standard-deviation term in the definition (§3.1, below Eq. 6).

### 3.2 Worked example: the leave-one-out advantage on a group of 8

Take `n = 8` rollouts of one SWE task with a binary reward, of which 2 pass. With the leave-one-out baseline
`A_i = r_i − (1/(n−1)) Σ_{j≠i} r_j` ([[deepswe]], derived in the card):

- A passing rollout: `A = 1 − (1/7)(1) ≈ +0.86`.
- A failing rollout: `A = 0 − (1/7)(2) ≈ −0.29`.
- If all 8 fail, every advantage is exactly 0 and the group contributes nothing.

Now divide by the return standard deviation, as GRPO does, and compare three groups of the same size:

| Successes out of 8 | Return std | Leave-one-out (pass / fail) | GRPO (pass / fail) | Ratio of the two |
|---|---|---|---|---|
| 4 (balanced) | 0.500 | +0.571 / −0.571 | +1.000 / −1.000 | 1.75× |
| 2 | 0.433 | +0.857 / −0.286 | +1.732 / −0.577 | 2.02× |
| 1 (near-uniform) | 0.331 | +1.000 / −0.143 | +2.646 / −0.378 | 2.65× |

Every advantage in a group is scaled by the same factor `((n−1)/n) / std`, and that factor grows as the group
becomes more uniform. Relative to the undivided baseline, the division moves training weight from tasks the model
sometimes solves toward tasks it almost always solves or almost never solves. [[loop-appworld]] states exactly
this as the reason to remove it — "The largest training signal thus comes from scenarios that the LLM either
fully solves or fails on. For AppWorld this seems less beneficial than considering scenarios that can sometimes
be solved" (§5.4) — and measures the cost: LOOP with reward normalization scores 61.9 against 71.3 without,
on AppWorld Test-Normal task goal completion (Table 1), a difference of 9.4 points.
[[gigpo-verl-agent]] reaches a conditional version of the same conclusion: `F_norm = std` hurt on the harder
ALFWorld subtasks and on WebShop but was similar elsewhere, so "`F_norm = std` can still be beneficial when
reward variance is stable" (§5.2).

### 3.3 Where the importance weight is applied

[[loop-appworld]] is the only source that ablates this axis directly. All three variants use the same rollouts and
the same leave-one-out advantage; they differ in whether the clipped ratio is formed per token, per turn or per
trajectory. Each Table 1 number is the mean over 5 evaluation runs of one checkpoint, "selected based on best dev
set performance" (Table 1 caption); Test-N is AppWorld test-normal and Test-C is test-challenge:

| Variant | Test-N TGC | Test-C TGC |
|---|---|---|
| LOOP token | 71.3 | 45.7 |
| LOOP turn | 64.1 | 40.8 |
| LOOP bandit (per trajectory) | 53.3 | 27.7 |

The three-run replication keeps the order: token 66.4 ± 4.8, and the per-trajectory variant is called unstable
with clipped importance weights (App. E, Table 2). One environment, one model size.

### 3.4 What outcome-only does not fix

[[deepswe]] gives the reason it needs compact filtering: "LLM agents may stumble upon correct patches and pass
all tests without knowing. Training with these positives rewards reinforces undesired behaviors across steps
(e.g. LLM answers correctly in first 10 steps but patches random files later on), leading to collapse when such
behaviors accumulate" (§2.3). The outcome reward cannot distinguish the useful prefix from the harmful suffix. §4 is the
set of methods that try to.

## §4 Credit assignment beyond the outcome reward

[[turn-level-credit-assignment]] gives the vocabulary: reward structures are **terminal**, **delayed**, or
**per-turn**, each corresponding to a different turn-level MDP, and the paper derives GRPO and PPO variants for
each. Its finding is that dense per-turn rewards beat terminal and delayed rewards for both algorithms on
multi-turn search and game tasks. Most frontier recipes in §3 nevertheless use terminal rewards, because a
per-turn reward requires a per-turn verifier that usually does not exist.

Three methods produce step-level signal without one.

### 4.1 GiGPO: anchor-state grouping

**Mechanism** ([[gigpo-verl-agent]], §4).

1. Sample `N` trajectories for one task from the same initial state.
2. Compute the episode advantage `A^E(τ_i) = (R(τ_i) − mean{R(τ_j)}) / F_norm({R(τ_j)})` (Eq. 3).
3. Collect the set `U` of distinct environment states seen anywhere in the group. For each anchor state `s̃ ∈ U`,
   build `G^S(s̃) = {(a_t^{(i)}, R_t^{(i)}) | s_t^{(i)} = s̃}` (Eq. 4, Eq. 6), where
   `R_t^{(i)} = Σ_{k=t..T} γ^{k−t} r_k^{(i)}` is the discounted return from that step (Eq. 5).
4. Compute the step advantage inside each anchor group,
   `A^S(a_t^{(i)}) = (R_t^{(i)} − mean{R_t ∈ G^S(s̃)}) / F_norm({R_t ∈ G^S(s̃)})` (Eq. 7).
5. Train on `A(a_t^{(i)}) = A^E(τ_i) + ω · A^S(a_t^{(i)})` with the clipped PPO surrogate (Eq. 8, Eq. 9).

Symbols: `N` rollouts per task; `T` steps per trajectory; `γ ∈ (0,1]` the discount; `ω ≥ 0` the weight on the step
term, set to 1 "with no further tuning" (§5.1); `F_norm` either `std` (GRPO) or `1` (leave-one-out).

**Worked numeric example.** Four rollouts of a shopping task reach the same search-results page at step 3.
Terminal reward 1 for buying the right item, 0 otherwise; `γ = 0.95`.

| Rollout | Action at the anchor state | Ends at step | Terminal reward | `R_3 = γ^(T−3) · r` |
|---|---|---|---|---|
| τ₁ | click 1st item | 5 | 1 | `0.95² = 0.9025` |
| τ₂ | click 2nd item | 7 | 1 | `0.95⁴ ≈ 0.8145` |
| τ₃ | click Next Page | 8 | 0 | 0 |
| τ₄ | click Next Page | 8 | 0 | 0 |

Anchor-group mean is `(0.9025 + 0.8145 + 0 + 0)/4 = 0.4293`. With `F_norm = 1` the step advantages are `+0.4732`,
`+0.3852`, `−0.4293`, `−0.4293`. The episode advantages, with mean return 0.5, are `+0.5`, `+0.5`, `−0.5`, `−0.5`.
Summing with `ω = 1`: τ₁ gets `+0.973`, τ₂ gets `+0.885`, τ₃ and τ₄ get `−0.929`.

Read the τ₁ / τ₂ gap: both rollouts succeeded, so GRPO would give their actions at this state the identical
advantage `+0.5`. GiGPO separates them because τ₁ reached the goal in fewer steps, so its discounted return from
the shared state is higher. The paper gives the same example in Figure 3 and states the resulting ordering
`A^S(1st Item) > A^S(2nd Item) > A^S(Next Page)`. The interactive version is in
[gigpo-anchor-grouping.html](figures/gigpo-anchor-grouping.html), where `γ` and `ω` are adjustable.

**Evidence (Result, single study; 3 seeds).** Averages over 3 random seeds, Qwen2.5-Instruct (Table 1):

| Model | Method | ALFWorld (all) | WebShop success |
|---|---|---|---|
| 1.5B | GRPO | 72.8 ± 3.6 | 56.8 ± 3.8 |
| 1.5B | GiGPO w/o std | 86.1 ± 4.7 | 67.4 ± 4.5 |
| 7B | GRPO | 77.6 ± 5.2 | 66.1 ± 3.7 |
| 7B | GiGPO w/o std | 90.2 ± 2.3 | 75.2 ± 3.8 |

On search-augmented QA trained on NQ and HotpotQA, the seven-dataset average is 42.1 at 3B and 47.2 at 7B against
32.5 and 38.5 for [[search-r1]] (Table 2); the out-of-domain Bamboogle column moves from 36.8 to 68.9 at 7B.

**Conditions and limits.** Anchor grouping needs states to repeat. In ALFWorld, groups of size 1 stay below 35%
of anchor states throughout training, so over 65% of states recur (§5.5); by iteration 140 the distribution
concentrates at group sizes 6 to 8 with `N = 8`, which the authors read as the policy converging. When no state
repeats, `A^S = 0` and the method "naturally degrades to GRPO" (§6). The cost is 0.01 s for hashmap grouping plus
0.53 s for the step advantages against 362.83 s per iteration for rollout, log-probability computation and the
policy update — under 0.002% (§5.6). The environments tested are text games and a shopping simulator with exactly
reproducible states; no source applies anchor grouping to a SWE repository or a live browser.

### 4.2 SWEET-RL: a critic that sees training-time information

[[sweet-rl]] trains a turn-level critic on information available only at training time — the reference solution,
for example — and uses it to supply step-level credit to the policy. The policy never sees that information, so
the asymmetry is between critic and actor. On ColBench, a benchmark where the agent works with a simulated human
partner on backend programming and frontend design, the reported gain is "a 6% absolute improvement in success
and win rates ... compared to other state-of-the-art multi-turn RL algorithms", enough for Llama-3.1-8B to match
or exceed GPT-4o on those tasks. The user-simulator half of this setup is the subject of [[ch-29d]].

### 4.3 Shaping the terminal reward by length

[[kimi-researcher]] keeps a single outcome reward and redistributes it over steps: "the reward of step i becomes
r×γ^(T − i), where r is the outcome reward, T is the number of steps, and 0<γ<1". Shorter successful trajectories
receive more credit per step. `γ` is not published, and no ablation isolates the decay from the rest of the
recipe.

## §5 Stability: what breaks in multi-turn RL and what stops it

### 5.1 The mechanism: drift → low-probability tokens → gradient explosion

[[simpletir]] runs the controlled comparison: single-turn tool use trains smoothly and multi-turn tool use on the
same data collapses with gradient spikes (§3.1, Fig. 2). The causal chain the paper gives:

1. Tool feedback is out of the model's pre-training distribution and enters the context (§3.1).
2. Conditioned on it, later generations become more stochastic and assign low probabilities to sampled tokens.
   Figure 3 shows high probabilities in turn 1 and a collapsed, near-zero-probability response in turn 4.
3. Those low-probability tokens are appended to the context, which worsens the drift for the next turn.
4. The gradient norm explodes.

**Proposition 3.1** makes step 4 precise. For a token `c` at position `t`,

`‖∇_{z_t} J_TIR‖₂ = (m_{i,t} / Σ_j m_{i,j}) · ρ_{i,t}(θ) · g_{i,t} · |Â_i| · sqrt(1 − 2P(c) + Σ_{j∈A} P(j)²)`

where `z_t` are the pre-softmax logits, `P = π_θ(· | o_{i,<t})`, `ρ_{i,t}` is the importance ratio, `m_{i,t}` the
feedback mask, `Â_i` the trajectory advantage, and `g_{i,t}` a gate that is active when the PPO update is not
clipped.

**Worked numeric example.** Compare two tokens in a distribution that is otherwise sharp, with
`Σ_j P(j)² ≈ 0.81`:

- Confident token, `P(c) = 0.90`: `sqrt(1 − 1.80 + 0.81) = sqrt(0.01) = 0.10`.
- Drifted token, `P(c) = 0.001`: `sqrt(1 − 0.002 + 0.81) ≈ sqrt(1.808) ≈ 1.34`.

The probability-dependent factor is about 13× larger. Now add the ratio. For a negative-advantage trajectory the
upper clip does not bind, so `ρ` is unbounded above: if `π_old(c) = 0.001` and one update raises `π_θ(c)` to
`0.01`, then `ρ = 10`. The two factors multiply to roughly 130× the confident-token gradient, from one token, in
one trajectory. This is the same softmax-gradient identity used in the negative-feedback chapter, seen from the
norm side rather than the mass-movement side.

Step 3 is also where credit assignment fails: a trajectory that reasoned correctly for three turns and collapsed
in the fourth receives one negative reward for all four, which "unfairly penalizes valid multi-turn behavior,
causing the policy to collapse toward safer, single-turn generations" (§3.2).

### 5.2 Void-turn filtering

**Definition.** A *void turn* is "an LLM response that contains neither a complete code block nor a final answer"
— partial code, repetitive text, or a premature end-of-sequence token ([[simpletir]], §3.3).

**Mechanism.** If any turn of a trajectory is void, mask the policy loss for the **entire** trajectory and remove
it from the batch before the GRPO update. This blocks the large gradients and removes the misassigned negative
credit at the same time.

**Evidence (Result, single study; highest score within 1,000 gradient steps, Table 2):**

| | SimpleTIR-7B | Naive multi-turn | Low-probability-token filtering | High-ratio filtering | Stop generation, no filtering |
|---|---|---|---|---|---|
| AIME24 | 50.5 | 20.8 | 23.3 | 26.3 | 26.1 |
| MATH500 | 88.4 | 73.1 | 72.8 | 75.0 | 77.3 |

Two of these columns are the obvious alternatives, and both fail: filtering by token probability or by importance
ratio "cannot resolve the issue of gradient explosion" (§4.3), because "their thresholds are difficult to tune and
they do not solve the credit assignment problem" (§3.3).
The last column stops generation on a void turn but still trains on the trajectory, and lands 24 points below the
full method — so the filtering, not the early stop, carries the effect. Final results: SimpleTIR-7B reaches 50.5
on AIME24 from a Qwen2.5-7B base at 3.2 and a text-only Zero-RL baseline at 22.1 (Table 1, Abstract).

### 5.3 Echo Trap and variance-based filtering

[[ragen-starpo]] reports a recurring failure mode in multi-turn RL that it calls the **Echo Trap**: the model
"repeatedly reuses memorized reasoning paths when trained on self-generated trajectories", reward variance
collapses, and gradient norms spike (Abstract, §4.1). The three monitored signals are stated as rules of thumb
(§4.1, Fig. 4): "Reward standard deviation is an early indicator of convergence"; "Gradient norm spikes indicate
irreversible collapse"; "Entropy should follow a stable decay trend during effective learning".

StarPO-S is the stabilized variant, with three parts (§4.2, App. D):

1. **Uncertainty-based trajectory filtering.** "retain only the top p% highly-uncertain prompts at each training
   step", ranked by reward standard deviation, default `p = 25%`. Reported effect: "retaining 75% of rollouts
   extends stability in FrozenLake from 100 to 140 steps, while 50% avoids collapse entirely".
2. **Critic incorporation**, i.e. a PPO value model.
3. **Gradient stabilization**: KL term removed, asymmetric clipping with `ε_low = 0.2`, `ε_high = 0.28`.

Note what part 1 discards: a group in which all rollouts succeed or all fail has zero reward variance. Those are
the same groups whose leave-one-out advantages are exactly 0 (§3.2), so filtering them costs nothing in signal and
saves the compute. The setup is small — Qwen2.5-0.5B-Instruct, 3B on WebShop, `P = 8` prompts × `N = 16` rollouts,
at most 5 turns and 10 actions (§3.2) — and the StarPO-vs-StarPO-S comparisons are published as curves, so this
chapter cites no success-rate numbers from them.

### 5.4 Masking the harness, not the policy

A trajectory can end for reasons that say nothing about the policy's quality. Four sources handle these the same
way — remove the gradient, keep the reward accounting — and the differences are in what counts as such a reason.

| Source | Masked or dropped | Kept |
|---|---|---|
| [[deepswe]] §2.3 ("compact filtering") | max context, max environment steps, 20-minute generation timeout | everything else; reward is still 0/1 |
| [[skyrl-agent]] §4.2 | 32K-context truncation, 50-turn limit; "this masking does not modify the reward or advantage estimation" | recoverable tool errors, injected into the context as feedback (§3.4) |
| [[glm-5]] §4.1.2 | samples whose oldest rollout weight version is staler than `τ`; samples that failed by environment collapse | model-caused failures |
| [[deepseek-v3.1]] §3.1 (Eq. 9) | `M_{i,t} = 0` when `Â_{i,t} < 0` **and** `(1/|o_i|) Σ_t log(π_old/π_θ) > δ` | negative samples that are still near-policy |

[[glm-5]] also states the group-repair rule that follows from dropping members of a group: "we pad the group by
repeating valid samples if the number of valid samples exceeds half of the group size; otherwise, we drop the
entire group" (§4.1.2). This matters because the leave-one-out baseline is computed over the group; silently
shrinking a group from 8 to 3 changes every advantage in it.

[[deepseek-v3.1]]'s rule is the sharpest statement of the principle: the authors state that "highly off-policy
negative samples can be detrimental" and mask only those, leaving positive-advantage samples and near-policy
negatives alone (§3.1).

### 5.5 Horizon as a schedule

[[agentgym-rl]] treats the maximum number of interaction turns as a curriculum: a monotonic schedule
`h_1 < h_2 < … < h_n` with `h_{t+1} = h_t + δh` every `Δ` training steps (§3.3). In Deep Search, a cap of 10 turns
"achieves higher performance in the early stage compared to a shorter-turn setting (e.g., 5), but rapidly
collapses as training progresses", while the shorter cap "restrict[s] early exploration but provide[s] more
stable learning signals"; the staged schedule is the design that follows from the pair (§4.2, Fig. 7). The released scripts use `[5, 8, 10]` every 100 steps for Deep Search,
`[8, 12, 15]` every 80 for WebArena, and `[10, 20, 30]` every 100 for TextCraft and SciWorld
([[agentgym-rl-recipe]]). One environment supports the comparison; seeds and variance are not reported.

## §6 Multi-task agentic RL and what it buys

**Problem.** A single-environment agent is a specialist. [[agentrl]] measures how narrow: a 14B policy trained on
the DB environment alone scores 0.2 on ALFWorld, and a WebShop-only policy scores 0.0 there (Table 4).

**Mechanism.** [[agentrl]] trains one policy on five AgentBench-FC environments at once, with two additions:

1. **Task advantage normalization** (§3.2, Eq. 1): `Ã_{i,s,g,t,k} = (Â_{i,s,g,t,k} − μ_i) / σ_i`, where `i` indexes
   the task and `μ_i, σ_i` are the mean and standard deviation of all token-level advantages of task `i` in the
   current batch. Without it, tasks with longer trajectories and different difficulty learn at different rates.
2. **Cross-policy sampling** (§3.1): actions inside one trajectory are drawn from more than one model — in
   practice the current policy and a "stale" rollout engine updated every several steps.

**Evidence (Result, single study; Table 4, 14B, task success):** the multi-task policy averages 67.7 against 67.8
for the best of the five single-task policies, while none of the single-task policies is close on tasks it was
not trained on. **Ablations (Table 6, 14B, average 65.0):** without cross-policy sampling 60.7; without task
advantage normalization 59.4. **Held-out (Table 5, 32B):** BFCL-v3 overall 59.9 → 61.4, multi-turn 16.2 → 19.2
([[bfcl]]). The paper does not reconcile the 65.0 ablation baseline with the 67.7 of Table 3.

Three other systems cover the same ground at different scales:

- [[agentgym-rl]] trains Qwen2.5-7B-Instruct on five scenarios with outcome rewards and no agent SFT:
  ScalingInter-7B scores 26.00 on WebArena, 38.25 on Deep Search, 91.00 on TextCraft, 96.67 on BabyAI and 57.00
  on SciWorld against 9.76, 18.75, 42.00, 66.67 and 1.50 for the untrained instruct model (Tables 1-5).
- [[skyrl-agent]] makes the tool the unit of composition: the agent acts only through OpenAI-style function
  calls, a Gym environment is added by wrapping `step()` as a tool, and each dataset binds its own tools and
  verifiers, so one training job can mix tasks (§3.1). Its asynchronous pipeline dispatcher is about 1.55× faster
  than bounded asynchronous batching at batch 64 × 8 rollouts on 2×8 H100 (Fig. 1b).
- [[glm-5]] scales the environments rather than the algorithm: "over 10k verifiable environments across thousands
  of repositories spanning 9 programming languages" for SWE, thousands of synthesized terminal environments with
  "Docker construction accuracy exceeding 90%", and a Multi-Task Rollout Orchestrator that standardizes all
  trajectories into "a unified message-list representation" and "supports over 1k concurrent rollouts"
  (§4.1.1, §4.2).
- [[kimi-researcher]] trains a search agent end to end with REINFORCE on strictly on-policy trajectories and
  reports HLE pass@1 rising "from 8.6% to 26.9%", pass@4 40.17%, and 69% pass@1 on xbench-DeepSearch averaged over
  4 runs, with about 23 reasoning steps and over 200 URLs explored per task.

One caution applies to every row above: the environments are also the evaluation. [[agentgym-rl]] states it
plainly (§7), and [[skyrl-agent]] reports the case where it shows: for the computer-use agent, training reward
rose while validation accuracy showed "little to no gain", and the training tasks were a subset of the OSWorld
benchmark (§5.3).

## Negative samples and negative feedback

Under the four senses of "negative" used in this course, agentic RL uses three of them, and the distinction
decides what each source's numbers mean.

**1. Where negatives come from.** A trajectory is negative when its verifier returns 0 (tests fail, answer does
not match, task goal not completed), when it terminates abnormally, or when it exceeds a limit. The false-negative
rate is not reported by any source in this chapter. Two sources make the labeling explicit: [[agentrl]] gives −0.2
for abnormal termination, exceeding maximum interaction rounds, or exceeding maximum response length, and 0 for
an ordinary failure (App. E.1-E.2); [[deepswe]] gives 0 when at least one test fails or the run times out (§2.2).

**2. What practice does with them.**

| Treatment | Sense | Sources |
|---|---|---|
| Below-baseline advantage through the clipped surrogate | negative as gradient | [[deepswe]], [[skyrl-agent]], [[loop-appworld]], [[agentrl]], [[glm-5]], [[gigpo-verl-agent]] |
| Masked from the gradient when the cause is the harness | none (removed) | [[deepswe]] §2.3, [[skyrl-agent]] §4.2, [[glm-5]] §4.1.2, [[simpletir]] §3.3, [[ragen-starpo]] §4.2 |
| Masked when the sample is both negative and far off-policy | bounded negative as gradient | [[deepseek-v3.1]] §3.1, Eq. 9 |
| Discarded to control entropy | none (removed) | [[kimi-researcher]] |
| Failed prefix plus diagnostic feedback becomes a new training input | negative as content | [[skyrl-agent]] §3.4, [[fission-grpo]] |

**3. Mechanism.** Only the gradient sense removes probability mass. The softmax logit gradient
`∂ log p_y / ∂ z_j = 1[j = y] − p_j` moves the mass removed from a pushed-down token onto the currently most
likely alternatives. [[simpletir]]'s Proposition 3.1 is the multi-turn consequence: the norm of that same gradient
carries a factor `sqrt(1 − 2P(c) + Σ_j P(j)²)` that grows as `P(c)` falls, and a factor `ρ_{i,t}` that is unbounded
above precisely when `Â_i < 0`. A push-down on an already-unlikely token in a late turn is therefore the largest
single gradient contribution in an agentic batch, which is why every stabilizer in §5 targets exactly that case.

**4. Evidence with numbers.**

- Benefit of *removing* harness negatives: AIME24 50.5 with void-turn filtering against 20.8 for naive multi-turn
  training, and 26.1 when generation stops on a void turn but the trajectory is still trained
  ([[simpletir]], Table 2).
- Benefit of *removing* zero-variance groups: FrozenLake stability extends from 100 to 140 steps at 75% retention
  and collapse is avoided at 50% ([[ragen-starpo]], §4.2).
- Benefit of *reusing* negatives as content: Fission-GRPO raises Qwen3-8B's error-recovery rate by 5.7 points and
  overall accuracy from 42.75% to 46.75% on BFCL v4 Multi-Turn ([[fission-grpo]], Abstract).
- Cost of discarding too much: [[kimi-researcher]] discards "some negative samples strategically" because
  "negative samples lead to a decrease in token probabilities, which increases the risk of entropy collapse", but
  reports no split of the gain and no discard fraction.

**Honesty about size of effect.** No source in this chapter measures what share of the improvement comes from
negative-advantage updates against positive ones. The closest measurements are one-sided: [[simpletir]] and
[[ragen-starpo]] measure what *removing* certain negatives buys, and [[fission-grpo]] measures what *reusing*
them as content buys; none of the three reports the positive-versus-negative split of the remaining gain, and
neither do [[deepswe]], [[loop-appworld]], [[skyrl-agent]] or [[agentrl]]. The single-turn
splits collected in [[ch-43a]] were not re-measured in any agentic setting. Treat claims that negatives drive
agentic RL as unmeasured for this stage.

**5. Controls that make negatives safe here.** Keep the update near-policy and mask negatives that are not
([[deepseek-v3.1]] Eq. 9; [[glm-5]] staleness threshold `τ`). Bound the ratio on both sides rather than only
above ([[glm-5]] Eqs. 3-5). Localize the negative to the step that failed rather than the whole trajectory
([[gigpo-verl-agent]] Eq. 7). Remove instead of penalizing when the cause is the harness (§5.4). Repair the group
after removals ([[glm-5]] §4.1.2).

**6. Diagnostics.** Log reward standard deviation per group, gradient norm, and policy entropy on the same axis —
[[ragen-starpo]] treats the first as the early signal and the second as the point of no return (§4.1). Add the
minimum token probability per turn index, which is [[simpletir]]'s diagnostic (Fig. 3), the fraction of
trajectories masked by each filter, and the fraction of groups with zero variance.

**7. Effect on generality.** [[demystifying-agentic-rl]] is the source that measures coverage: its
exploration-friendly recipes raise pass@k and average@k together on AIME2024/2025, while the plain GRPO baseline
shows the trade-off where pass@k is suppressed (§4.2, Fig. 4). [[agentgym-rl]] reports that at 64 samples
its RL model is 5.5 points above the untrained base model on Deep Search and 7.05 points above it on SciWorld
(§5.1, Fig. 9). [[ragen-starpo]] reports the opposite direction for reasoning
content: under outcome-only reward, traces shrink and become mismatched to environment states (§4.4).

## Recipe

Values as printed at the stated locus. Paper values, released-config values and blog values are separate rows.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSWE-Preview | 32B | RL | Algorithm | GRPO++: clip high, no KL loss, no reward std, length normalization by max context length, leave-one-out advantage, compact filtering, no entropy loss | together.ai/blog/deepswe §2.3 ([[deepswe-recipe]]) | verified 2026-09-14 | Blog Fig. 5 (FrozenLake), Fig. 6 (Qwen3-14B); curves only |
| DeepSWE-Preview | 32B | RL | Masked trajectories | max context, max environment steps, 20-minute generation timeout | Blog §2.3 | verified 2026-09-14 | Fig. 6: compact filtering prevents or delays reward collapse |
| DeepSWE-Preview | 32B | RL | LR; clip_ratio_high; entropy coefficient | 1e-6; 0.28; 0.0 | rllm@709dec43b740 `scripts/agent/swe/deepswe_32b.sh` L24, L34, L53 | verified 2026-09-14 (script only) | no ablation reported |
| DeepSWE-Preview | 32B | RL | max env steps; max response length | 50; 32,768 | same script L69, L18-19 | verified 2026-09-14 (script only) | conflicts with the blog, which prints neither |
| SA-SWE-32B | 32B | RL | batch; mini-batch; rollouts per task | 64; 64 (fully on-policy); 8 | arXiv:2511.16108v1 §4.2 | verified 2026-09-14 | §4.2: chosen "for stable training"; no ablation |
| SA-SWE-32B | 32B | RL | truncated trajectories | masked from the gradient; reward and advantage estimation unchanged | §4.2 | verified 2026-09-14 | §4.2 reason: avoid bias against trajectories with more actions |
| SA-SWE-32B | 32B | RL | compute | 4,601 H100 hours | Table 2 | verified 2026-09-14 | Table 2 lists DeepSWE at 9,180 H100 hours |
| LOOP (token) on Qwen2.5-32B-Instruct | 32B | RL | importance weight | per token | arXiv:2502.01600v3 §5.4, Table 1 | verified 2026-09-14 | Table 1: token 71.3, turn 64.1, trajectory 53.3 (Test-N TGC) |
| LOOP (token) | 32B | RL | advantage | leave-one-out, no std normalization; rollouts with `|Â| < 0.01` dropped | Eq. 3, §4.2, App. D | verified 2026-09-14 | Table 1: RwNorm 61.9 vs 71.3 |
| LOOP (token) | 32B | RL | tasks per iteration; rollouts per task; temperature | 40; K = 6 (240 rollouts); 1.0 | App. D, §5.2 | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | interaction and length limits | 40 interactions (train), 50 (eval); 1,500 output tokens per turn; API responses truncated above 3K tokens | §5.2, App. D | verified 2026-09-14 | no ablation reported |
| LOOP (token) | 32B | RL | log-probabilities | recomputed under the generating policy, not taken from vLLM | App. D | verified 2026-09-14 | no ablation reported |
| Search-R1 (Qwen2.5-7B-base, PPO) | 7B | RL | retrieved-token masking | loss indicator 0 on `<information>` tokens, in the objective and in the KL term | arXiv:2503.09516 §3.1, Eq. 2 | verified 2026-09-15 | Table 4: seven-dataset average EM 0.431 with mask vs 0.343 without |
| Search-R1 | 3B, 7B | RL | reward | `r_φ(x, y) = EM(a_pred, a_gold)`; no format reward, no learned RM | §3.4, Eq. 4 | verified 2026-09-15 | no ablation reported |
| GiGPO (Qwen2.5-1.5B/3B/7B-Instruct) | 1.5B-7B | RL | step-advantage weight ω; rollouts per task N | 1 ("with no further tuning"); 8 (5 for search QA, max 4 turns) | arXiv:2505.10978v3 §5.1 | verified 2026-09-15 | Fig. 4: removing either advantage level degrades every task |
| GiGPO (search QA) | 3B, 7B | RL | anchor matching | similarity-based grouping, longest-matching-subsequence threshold 0.9 | §5.1 | verified 2026-09-15 | no ablation reported |
| SimpleTIR (Qwen2.5-7B / 32B base) | 7B, 32B | RL | rollout batch; mini update; max response; max turns | 512; 128; 16K then 24K; 5 then 10 | arXiv:2509.02479v2 §4.1 | verified 2026-09-15 | Fig. 5 top: MATH500 and length scale with turns; AIME24 does not |
| SimpleTIR | 7B, 32B | RL | trajectory filter | mask the policy loss of the whole trajectory if any turn is void | §3.3 | verified 2026-09-15 | Table 2: AIME24 50.5 vs 20.8 naive multi-turn |
| SimpleTIR | 7B, 32B | RL | template | no chat template; tool outputs prefixed "Code Execution Result:"; generation stopped after a complete code block | §3.4 | verified 2026-09-15 | no ablation reported |
| StarPO-S | 0.5B (3B on WebShop) | RL | trajectory filter; clipping | keep top 25% of prompts by reward std; KL removed; ε_low 0.2, ε_high 0.28 | arXiv:2504.20073 §4.2, App. D | verified 2026-09-15 | §4.2: 75% retention extends FrozenLake stability 100 → 140 steps; 50% avoids collapse |
| StarPO / StarPO-S | 0.5B, 3B | RL | rollout shape | P = 8 prompts × N = 16 rollouts; at most 5 turns and 10 actions | §3.2 | verified 2026-09-15 | no ablation reported |
| ScalingInter-7B (released scripts) | 7B | RL | turn-cap schedule | Deep Search [5,8,10] / 100 steps; WebArena [8,12,15] / 80; TextCraft and SciWorld [10,20,30] / 100; BabyAI [6,13,20] / 100 | `examples/train/ScalingInter-RL/*.sh` ([[agentgym-rl-recipe]]) | verified 2026-09-14 | Fig. 7 (Deep Search): fixed cap 10 collapses, fixed cap 5 plateaus |
| AgentGym-RL (paper values) | 3B, 7B | RL | algorithm; KL coefficient; temperature; trajectories per query | GRPO; 1e-3; 1.0; 8 (4 for WebArena) | arXiv:2509.08755 App. B.1-B.5 | conflict (scripts set n = 4) | Table 6: GRPO above REINFORCE++ at both sizes |
| AgentRL | 14B | RL | task advantage normalization | `Ã = (Â − μ_i)/σ_i` over all loss-masked token advantages of task `i` in the batch | arXiv:2510.04206 §3.2, Eq. 1 | verified 2026-09-14 | Table 6: five-task average 65.0 → 59.4 without it |
| AgentRL | 3B-32B | RL | reward for abnormal termination | −0.2; ordinary failure 0; success 1 | App. E.1-E.2 | verified 2026-09-14 | no ablation reported |
| AgentRL | 3B-32B | RL | samples per rollout; temperature | 8; 0.8 | §4.1, App. E.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2 | 671B total / 37B active | RL | advantage; off-policy sequence mask | `Â_{i,t} = R_i − mean(R)`, no std; `M_{i,t} = 0` when `Â_{i,t} < 0` and `(1/|o_i|) Σ_t log(π_old/π_θ) > δ` | arXiv:2512.02556 §3.1, Eq. 9 | verified 2026-09-14 | §3.1: masking improved stability in otherwise unstable runs; δ not printed |
| DeepSeek-V3.2 | — | RL | agentic task counts | 24,667 code-agent; 50,275 search-agent; 4,417 general-agent over 1,827 synthesized environments; 5,908 code-interpreter | Table 1, §3.2.3 | verified 2026-09-14 | Fig. 5: synthetic-only RL improves three unseen environments |
| GLM-5 | not reported | RL (agentic) | objective; token clipping; staleness filter | group-mean advantage without std; `f(r; ε_ℓ, ε_h)` zeroes tokens outside `[1−ε_ℓ, 1+ε_h]`; drop when `w′ − w_0 > τ` | arXiv:2602.15763v2 §4.1, §4.1.2 | verified 2026-09-15 | no ablation reported; ε_ℓ, ε_h and τ not printed |
| GLM-5 | not reported | RL (agentic) | group repair after removals | pad by repeating valid samples if valid > half the group; otherwise drop the group | §4.1.2 | verified 2026-09-15 | no ablation reported |
| GLM-5 | not reported | RL (agentic) | environments | over 10k SWE environments, 9 languages; thousands of terminal environments, Docker build accuracy > 90% | §4.2.1, §4.2.2 | verified 2026-09-15 | no ablation reported |
| Kimi-Researcher | not reported | RL | algorithm; per-step reward | REINFORCE, strictly on-policy; `r × γ^(T − i)` | moonshotai.github.io/Kimi-Researcher (2025-06-20) | verified 2026-09-15 | context-management ablation: 30% more iterations; γ not printed |

**Starting point for a small general-purpose agentic run.** Every number here comes from a `verified` row above,
with its conditions. From the released DeepSWE script for Qwen3-32B on 4.5K R2E-Gym tasks with 16-64 GPUs: LR
1e-6, `clip_ratio_high` 0.28, entropy coefficient 0, 8 trajectories per problem, temperature 1.0, no KL loss,
max 50 environment steps, max response length 32,768. From [[skyrl-agent]] §4.2 at the same model size: fully
on-policy (batch equals mini-batch), leave-one-out advantage without standard-deviation or length normalization,
and truncated trajectories masked with reward and advantage unchanged. From [[search-r1]] §3.1: mask environment
tokens in the objective and in the KL term. From [[simpletir]] §3.3, verified at 7B and 32B on a code
interpreter: drop any trajectory containing a turn with neither a tool call nor a final answer. None of these
values was tuned below 7B by their sources, and no source reports a small-scale sweep.

## Generalization lens

**(a) What increases breadth.**

- **Multi-task training in one policy.** One 14B AgentRL policy over five environments averages 67.7 against 67.8
  for the best of five specialists, while specialists score near zero off their own task ([[agentrl]], Table 4).
  Held-out BFCL-v3 rises 59.9 → 61.4 overall and 16.2 → 19.2 on the multi-turn split at 32B (Table 5).
- **Transfer from one agentic domain to others.** SWE-only RL moved Terminal-Bench 13.75 → 16.25, BrowseComp-Plus
  accuracy 18.1 → 19.4 and WebArena 15.8 → 17.0 on the same 32B checkpoint ([[skyrl-agent]], Table 3), with one
  evaluation run per cell and no variance reported.
- **Synthetic environment diversity.** RL on V3.2-SFT with only synthetic general-agent tasks improved Tau2Bench,
  MCP-Mark and MCP-Universe, none of which were in the RL mixture, while V3.2-Exp — trained with RL only in search
  and code environments — did not improve on them ([[deepseek-v3.1]], §4.3, Fig. 5; values shown in the figure
  only). The synthetic tasks span 1,827 environments (Table 1).
- **Robustness to perturbations along the trajectory**, rather than tool count alone. [[minimax-m2-aligning-to-what]]
  lists five axes — tool info, system prompt, user prompt, environment, tool responses — and states that "our old
  'tool scaling' approach only addressed the first item". Official but qualitative: the post prints no numbers.
- **Exploration-preserving RL settings.** Clip-higher plus overlong reward shaping raised pass@k and average@k
  together, where the plain GRPO baseline suppressed pass@k ([[demystifying-agentic-rl]], §4.2, Fig. 4).

**(b) What causes narrowing or forgetting.**

- **Training only where you evaluate.** [[agentgym-rl]] states its agents "perform well within in-domain
  settings" and lists unseen environments as future work (§7). [[skyrl-agent]]'s computer-use agent raised
  training reward with "little to no gain" on validation, on training tasks drawn from the evaluation benchmark
  (§5.3).
- **Outcome-only reward over long horizons.** Reasoning traces shrink and become mismatched to environment
  states; "without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge"
  ([[ragen-starpo]], Abstract, §4.4).
- **Collapse toward single-turn behaviour.** Terminal reward plus a late-turn failure penalizes the correct early
  turns, "causing the policy to collapse toward safer, single-turn generations" ([[simpletir]], §3.2).
- **Starting from a long-CoT model.** RL from Qwen3-4B-Thinking-2507 drove tool calls toward zero on reasoning
  tasks ([[demystifying-agentic-rl]], §5.2, Fig. 9).
- **Scaffold overfitting.** Kimi-dev and SWE-Swiss, trained under the Agentless workflow scaffold, "struggle to
  follow tool-call instructions" under a simple ReAct scaffold ([[skyrl-agent]], §4.3).

**(c) How to measure it for this stage.**

1. **Held-out environments, not held-out tasks.** [[agentrl]] uses BFCL-v3 (Table 5); [[deepseek-v3.1]] names
   MCP-Universe, MCP-Mark and Tool-Decathlon as environments not seen in RL (§4.1); [[loop-appworld]] uses
   AppWorld test-challenge (Test-C), whose tasks "require more complex sequences of interactions and involve new
   apps not seen during training" (§5.1).
2. **A second scaffold.** [[skyrl-agent]] evaluates every model under one simple ReAct scaffold and warns that
   "because different works use distinct scaffolds and evaluation setups, the numbers may not be strictly
   comparable" (Table 2 caption). A model that drops under a scaffold change learned the harness.
3. **pass@k at large k**, to separate sharpening from new coverage ([[demystifying-agentic-rl]] §4.2;
   [[agentgym-rl]] §5.1, Fig. 9 at K = 64; the argument is [[rlvr-beyond-base-model]]'s).
4. **Non-agentic benchmarks.** None of [[deepswe]], [[agentgym-rl]], [[kimi-researcher]] or [[agentrl]] reports a
   general chat, instruction-following or knowledge evaluation after agentic RL. For a general-purpose model this
   is the measurement gap of this stage.
5. **Contamination checks on the environment set.** [[deepswe]] removed repositories shared with
   SWE-Bench-Verified, such as sympy, from its 4.5K R2E-Gym training problems (§2.1).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Environment tokens enter the loss | Training loss falls while task reward is flat; the model starts emitting text that looks like tool output | Sum the loss mask over one batch and compare to the number of assistant tokens the rollout recorded |
| Loss averaged over sequence length instead of over the mask | Effective learning rate varies with observation verbosity; gradient norm correlates with average observation length | Log `Σ_t m_{i,t} / |o_i|` per trajectory; [[simpletir]] Eq. 1 divides by `Σ_t m_{i,t}` |
| Text-in-text-out re-tokenization | No error; advantages drift out of alignment with actions as turns accumulate | Re-encode the decoded rollout and assert equality with the recorded token ids ([[glm-5]] §4.1.2) |
| Harness-caused endings scored as failures | Reward collapses after an initial rise; average environment steps rise while per-step response length falls | Log termination cause per trajectory and the share masked by each filter ([[deepswe]] §2.3, Fig. 7) |
| Dividing the advantage by the return standard deviation | Near-uniform groups dominate the update; easy and impossible tasks produce the largest gradients | Compare a run with `F_norm = 1`; [[loop-appworld]] measures 61.9 vs 71.3 (Table 1) |
| Group shrunk by filtering without repair | Advantage magnitudes jump between steps for no visible reason | Log group size after filtering; apply [[glm-5]]'s pad-or-drop rule (§4.1.2) |
| Void turns trained as ordinary failures | Gradient-norm spikes; minimum token probability per turn falls with turn index; turn count drifts toward 1 | Plot minimum token probability by turn index ([[simpletir]] Fig. 3) and the share of trajectories with a void turn |
| Zero-variance groups kept in the batch | Compute spent on groups whose advantages are all 0; reward std trends to 0 before collapse | Log the fraction of groups with zero reward std; [[ragen-starpo]] filters to the top 25% (§4.2) |
| Chat template differs between RL and deployment | Benchmarks that need carried-forward reasoning drop far more than others | Compare the two histories on a reasoning-heavy agent benchmark; [[minimax-m2-interleaved-thinking]] measures +2.2 on SWE-Bench Verified and +23 on τ² |
| Turn cap fixed at the final value from step 0 | Early reward rises faster than a staged schedule, then collapses | Run the staged cap as a control ([[agentgym-rl]] Fig. 7) |
| Evaluating only in the training environments | Every number rises; nothing transfers | Add one held-out environment and one alternative scaffold before reading any result |

## Check your understanding

1. Masking environment tokens removes them from the gradient but not from the context. Explain, using
   [[simpletir]]'s causal chain, why the second half of that sentence is what produces gradient explosion, and why
   the explosion appears in multi-turn but not single-turn tool use.
2. A group of 8 rollouts has 7 failures and 1 success. Compute the leave-one-out advantages, then the GRPO
   advantages after dividing by the return standard deviation. Explain which design choice [[loop-appworld]]'s
   9.4-point measurement is evidence about, and why the direction of the effect depends on the task difficulty
   distribution rather than on the algorithm alone.
3. [[deepswe]] masks trajectories that hit the context limit but keeps their reward at 0. State what breaks if you
   instead (a) drop them from the group entirely without repair, or (b) score them as failures and train on them.
4. GiGPO adds a step-level advantage for 0.54 s per iteration against 362.83 s of shared cost. Explain why it
   needs no extra rollouts, and construct an environment in which the method provably reduces to GRPO.
5. [[ragen-starpo]] discards groups with low reward variance and [[simpletir]] discards trajectories with void
   turns. One of these removals costs no training signal at all under a leave-one-out baseline. Say which, and
   prove it from the advantage formula.
6. [[minimax-m2-interleaved-thinking]] reports +2.2 on SWE-Bench Verified and +23 on τ² from the same change to
   history construction. Give a mechanism that predicts this spread, and say what it implies about choosing a
   benchmark to detect a template mismatch.
7. [[agentrl]] reports that removing task advantage normalization drops the five-task average from 65.0 to 59.4,
   with KG dropping most. Explain how per-task normalization changes the relative gradient scale across tasks, and
   why the effect is larger when tasks differ in trajectory length.
8. [[skyrl-agent]] reports that SWE-only RL improved Terminal-Bench, BrowseComp-Plus and WebArena. Give two
   explanations for this result that the reported evidence cannot distinguish, and name the measurement that would
   separate them.

## Connections

- **Previous:** [[ch-45a]] — Preference-Optimization and RL Stage Recipes Side by Side. That chapter compares the
  stage recipes as whole pipelines; this one opens the agentic RL stage inside them.
- **Depends on:** [[ch-29d]] — User Simulators, Trajectory Verification, and Failed Trajectories, for where
  agentic trajectories and their verifiers come from, and for the simulator that [[sweet-rl]] trains against.
- **Next:** [[ch-45c]] — Context Management for Long-Horizon Agents. Every recipe here hits a context limit; that
  chapter covers what the agent does when it does, including the Discard-all strategy that raises
  [[deepseek-v3.1]]'s BrowseComp score from 51.4 to 67.6 (§4.1, §4.4).
- **Builds on:** [[ch-40]] for the group baselines (RLOO, GRPO, Dr. GRPO, DAPO) that every algorithm here
  modifies; [[ch-43]] for entropy and KL control, which most agentic recipes remove; [[ch-44]] for the verifiers
  that supply the outcome reward.
- **Followed by:** [[ch-45d]] — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL,
  which places this stage inside the full open recipes; and [[ch-44b]] — Multi-Domain RL for General Capability:
  Non-Verifiable Rewards and Domain Mixing, the general RL stage that usually runs after this one.

## Sources

- [[search-r1]] — retrieved-token masking, the `<think>/<search>/<information>/<answer>` template, the exact-match
  reward, and the masking ablation (0.431 vs 0.343 average EM).
- [[deepswe]], [[deepswe-recipe]] — GRPO++, compact filtering, the leave-one-out advantage arithmetic, and the
  42.2% / 71.0% SWE-Bench-Verified results; the recipe card separates blog values from released-script values.
- [[loop-appworld]] — the POMDP formulation, the per-token / per-turn / per-trajectory importance-weight ablation,
  and the reward-normalization measurement on AppWorld.
- [[gigpo-verl-agent]] — anchor-state grouping, the two-level advantage equations, the ALFWorld and WebShop
  results, and the cost breakdown.
- [[turn-level-credit-assignment]] — the terminal / delayed / per-turn taxonomy of reward granularity.
- [[sweet-rl]] — the turn-level critic trained on training-time-only information, and the ColBench result.
- [[ragen-starpo]] — the Echo Trap failure mode, its three warning signals, and StarPO-S's variance-based
  trajectory filtering.
- [[simpletir]] — the low-probability-token analysis, Proposition 3.1, the void-turn definition, and the filtering
  ablation table.
- [[agentrl]], [[agentrl-recipe]] — multi-task agentic RL, task advantage normalization, cross-policy sampling,
  and the BFCL-v3 held-out result.
- [[agentgym-rl]], [[agentgym-rl-recipe]] — the five-scenario framework, the ScalingInter turn-cap schedule, and
  the in-domain-only scope statement.
- [[skyrl-agent]], [[skyrl-agent-recipe]] — the tool-centric agent loop, transition-based recording, truncated-
  trajectory masking, and the out-of-domain evaluation table.
- [[kimi-researcher]] — end-to-end REINFORCE for a search agent, gamma-decay step rewards, negative-sample
  discarding, and the HLE and xbench results.
- [[minimax-m2-aligning-to-what]] — the five perturbation axes and the statement that tool scaling alone did not
  produce agent generalization.
- [[minimax-m2-interleaved-thinking]] — the keep-versus-discard-thinking benchmark comparison.
- [[demystifying-agentic-rl]] — pass@k under agentic RL, entropy behaviour, and the long-CoT starting-model result.
- [[deepseek-v3.1]] — off-policy sequence masking, the no-std advantage, thinking in tool use, the agentic task
  counts, and the unseen-environment ablation.
- [[glm-5]] — the agentic RL objective, the TITO gateway, double-sided token clipping, the staleness and
  environment-failure filters, the group-repair rule, and environment scaling.
- [[fission-grpo]] — converting failed tool-call trajectories into on-policy corrective training instances.
- [[grpo]], [[dr-grpo]], [[rloo]], [[ppo]] — the base algorithms every recipe in this chapter modifies.
- [[r2e-gym]] — the executable SWE environments behind [[deepswe]] and [[skyrl-agent]].
- [[bfcl]] — the held-out function-calling benchmark used by [[agentrl]] and [[fission-grpo]].
- [[rlvr-beyond-base-model]] — the pass@k-at-large-k argument used in the generalization measurements.
