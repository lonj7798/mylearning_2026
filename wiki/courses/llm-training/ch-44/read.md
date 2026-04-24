<!-- chapter: ch-44
     track: rl
     kind: content
     title: Process Supervision and Verifiable Rewards
     deps: [ch-43]
     sources: [[prm800k]], [[lets-verify]], [[let-verify]], [[math-shepherd]], [[omegaprm]], [[rlvr-tulu3]], [[tulu-3]], [[swe-rl]], [[training-verifiers-to-solve-math-word-problems]], [[step-dpo]], [[prorl]], [[rlvr-beyond-base-model]]
     figures: figures/prm-vs-orm.html
     excerpts: excerpts/prm800k-label-protocol.md, excerpts/lets-verify-prm-vs-orm.md, excerpts/math-shepherd-mc-formula.md, excerpts/omegaprm-divide-and-conquer.md, excerpts/rlvr-tulu3-config.md, excerpts/swe-rl-difflib-reward.md, excerpts/prm-vs-rlvr-contrast.md
-->

# Chapter 44 — Process Supervision and Verifiable Rewards

> **Core insight.** Once a task has a ground-truth check — a grader, a unit test, a reference diff — the learned reward model is a liability, not an asset. Either decompose the trajectory and reward each step against a process-level signal (a PRM), or collapse the whole reward function to `v(x, y) in {0, 1}` and let a deterministic program do the judging. Both moves sidestep Goodhart by removing the proxy; their costs and signal densities are different, and the right choice per task is the chapter's real content.
>
> **Guideline.** For every RL prompt, ask first: "Is the answer checkable?" If yes, skip the RM and use the verifier (RLVR, Tulu-3 LR `3e-7`, KL `0.05`; SWE-RL difflib ratio, GRPO KL `0.02`). If the answer is checkable but the chain is long, add a PRM on top of the outcome check so credit lands on the first bad step — build it with OmegaPRM's divide-and-conquer MC labels (`O(K log L)`) rather than Math-Shepherd's per-step scan (`O(K L)`). Only fall back to a learned preference RM (ch-41) when no verifier exists.

---

## Why this chapter exists

Chapter 40 built PPO and GRPO, ch-41 trained the Bradley-Terry RM, ch-42 catalogued reward hacking, and ch-43 gave you the KL / entropy controls that keep a run alive. Every one of those techniques assumes a learned scalar reward. The last three years of post-training progress — [[prm800k]], [[math-shepherd]], [[omegaprm]], [[rlvr-tulu3]], [[swe-rl]], [[deepseek-r1]] — come from *not assuming that*. When the task is verifiable, a learned RM is strictly worse: it eats labels, it is subject to reward-model over-optimisation ([[reward-model-overoptimization]] carried forward from ch-42), and it teaches the policy to exploit its drift. Process supervision and verifiable rewards are the two escape hatches. This chapter fixes the taxonomy so ch-45 (self-improvement loops) and ch-46 (the RL lab) can pick between them cleanly.

---

## §1. The problem outcome RMs leave on the table

[[training-verifiers-to-solve-math-word-problems]] (Cobbe 2021) was the first paper to frame the issue: a 6B GPT model that pass@1s 20% of GSM8K pass@100s 60%. The generator already produces the correct answer one time in five; the deficit is *selection*. Cobbe's fix was an outcome reward model (ORM) — train a separate verifier to score `(question, candidate-solution)` pairs, sample 100, pick the top-scoring one.

| Method | GSM8K | MATH-500 (subset) | Signal density |
|--------|-------|--------------------|----------------|
| Finetuning only (6B, Cobbe 2021) | ~20 pass@1 | — | dense (every token) |
| ORM best-of-100 (Cobbe 2021) | +strong gain | — | one scalar per solution |
| Majority vote (Lightman 2023) | — | 69.6 | — |
| ORM best-of-1860 (Lightman 2023) | — | 72.4 | one scalar per solution |
| **PRM best-of-1860 (Lightman 2023)** | — | **78.2** | one scalar *per step* |

The last two rows are the pivot. [[prm800k]] showed that for MATH problems, outcome labels leave 5.8 absolute points on the table *even at matched compute* against a PRM — and the gap grows with N. The reason is blunt: a long reasoning trace can arrive at the right answer by cancellation of two wrong steps, and it can also arrive at the wrong answer through a clean chain that slips at step 14. Outcome labels cannot tell these apart. Process labels can.

---

## §2. PRM800K — step-level human labels as the reference protocol

[[prm800k]] is the canonical supervised PRM. The protocol is worth memorising because every automated variant (Math-Shepherd, OmegaPRM, R*-Math) is engineered to approximate it cheaply.

**Data unit.** One labeled example is `(problem, partial-solution-prefix, next-step, label)` where `label in {+1 correct, -1 incorrect, 0 neutral}`. The generator is told to produce newline-delimited steps so the labeler sees one step at a time; the labeler marks the first `-1` they encounter and stops. Neutral captures filler ("Let me re-read the problem...") that carries no correctness signal.

**Scale.** 800K step labels over ~75K GPT-4 generations to 12K MATH training problems. Paper explicitly reports ~**10x per-example cost** vs outcome labeling — active learning is not optional if you want a publishable budget.

**PRM training.** A binary classifier head over `{good, bad}` fires at the token position immediately after each step separator. Loss is cross-entropy on non-neutral steps only:

```python
# Conceptual; reference: prm800k schema + let-verify section 3.
# labels_step[t] in {+1, -1, 0}; 0 is ignored.
step_logits = prm_head(hidden[step_end_positions])          # [num_steps, 2]
step_targets = (labels_step[labels_step != 0] == +1).long() # {0, 1}
loss = F.cross_entropy(step_logits[labels_step != 0], step_targets)
```

**Aggregation to a solution score.** Lightman's paper uses the product of per-step `p_correct`:

```
S_prod(y) = prod_{t in steps} p_correct(step_t | prefix_t)
```

Equivalently, `exp(sum_t log p_correct)`. [[math-shepherd]] later argued that `min_t p_correct(step_t)` is a better aggregator in practice — "a solution is only as good as its worst step" — and reports `min` Pareto-dominating `prod` and `mean` on GSM8K and MATH (Math-Shepherd Table 4). The training loss does not change; only the inference-time aggregation does.

**Active learning.** The paper surfaces solutions the current PRM scores highly even though the final answer is wrong (so-called "convincing-wrong" cases). Labeling that slice yields a ~2.6x data-efficiency multiplier — a 38% label budget reaches the same PRM quality as uniform at 100%.

---

## §3. Let's-Verify — process vs outcome head-to-head

The PRM-vs-ORM table at matched compute is the single most cited result from [[lets-verify]]. Numbers are verbatim from their MATH-500 representative subset (Best-of-N with N=1860):

| Selector | MATH-500 acc | Delta vs majority |
|----------|--------------|--------------------|
| Majority vote | 69.6 | — |
| ORM (outcome labels) | 72.4 | +2.8 |
| **PRM (process labels)** | **78.2** | **+8.6** |

Two things to notice. First, the PRM curve *dominates* the ORM curve at every N, not only at large N — the gap is already visible at N=16 and widens through N=1860. Second, the calibration picture (paper Figure 3) shows that the PRM's per-step `p_correct` is calibrated; the ORM's per-solution score is not. That is the mechanistic reason the PRM selector keeps paying off: the aggregator `min_t p_correct` is a well-ordered statistic only when the per-step probabilities are themselves meaningful.

Inference cost is the asymmetry that nothing in the paper hides: ORM is `O(1)` forwards per solution, PRM is `O(L)` where `L` is the step count. On MATH-500 the median `L` is ~10, so PRM inference is ~10x slower at Best-of-N time — an acceptable hit when N itself is 1860.

---

## §4. Math-Shepherd — MC rollouts as automatic step labels

Paying ~10x outcome-label cost does not scale past 800K. [[math-shepherd]] is the first paper to make PRMs cheap. The idea is one line: replace the human labeler with a rollout policy and the label with empirical reach-probability.

**The definition.**

```
For a step s_t in trajectory (s_1, ..., s_L):

    MC(s_t) = (1/K) * sum_{i=1..K} I[rollout(policy | s_1..s_t) reaches gold]
```

`K` is the number of completions sampled from the prefix ending at step `t`. `I[...]` is 1 if that completion's final answer matches the gold answer. The paper uses `K = 8` or `16`; [[omegaprm]] argues `K >= 16` is needed for trajectories deeper than ~10 steps.

**Hard vs soft.** The "hard" label `y_hard(s_t) = 1[MC(s_t) > 0]` is a binary "at least one rollout survives"; the "soft" label `y_soft(s_t) = MC(s_t)` is the fraction directly. Math-Shepherd Table 5 shows soft wins on MATH, hard is comparable on GSM8K — soft is the default.

**What it buys.** Mistral-7B on GSM8K moves 77.9 -> 84.1 with step-level PPO (PRM as dense reward) and 77.9 -> 89.1 when the same PRM is used only at inference for Best-of-N. On MATH: 28.6 -> 33.0 (PPO) -> 43.5 (verify). The verify-only number beating PPO is the lesson: a good PRM is worth more as a ranker than as a dense reward, at least at the scale Math-Shepherd tested.

**Step-level PPO reward composition.**

```
R_total(trajectory) = r_final + lambda * sum_{t in steps} PRM(step_t)
```

with `lambda ~ 0.1 - 1.0`; `r_final` is the outcome 0/1 reward. The PRM terms act as a shaped, dense reward so the policy gets gradient for correct *intermediate* steps even when the final answer is wrong — the failure mode that pure RLVR handles poorly on long chains.

---

## §5. OmegaPRM — divide-and-conquer labeling at `O(K log L)`

Math-Shepherd's `O(K * L)` rollouts per trajectory (every prefix gets K completions) is still expensive. [[omegaprm]] drops it to `O(K * log L)` by binary-searching the first bad step:

1. Start with a seed trajectory of `L` steps.
2. Measure `MC(s_{L/2})` with K rollouts.
3. If `MC(s_{L/2})` is close to `MC(s_0)`, the first error is in the second half — recurse on `steps[L/2 : L]`.
4. Else the first error is in the first half — recurse on `steps[0 : L/2]`.
5. Terminate when the interval has length 1; the isolated step is the first wrong one.

The labels populated along the way form the PRM training set. The threshold for "MC dropped sharply" is `tau ~ 0.2` (a step is marked bad when `MC(s_t) < tau` and its parent is above `tau`). The full paper regresses the PRM onto soft `MC(s_t)` values via MSE rather than thresholding, so `tau` only matters for the recursion decision, not the final labels.

**Result.** 1.5M step labels on ~80K problems generated fully automatically. Gemini Pro 1.0 on MATH moves `51.0 -> 69.4` with PRM-weighted Best-of-N — `+18.4` absolute, `+69.4%` relative. Beats Math-Shepherd PRM by ~5 MATH points at equal compute.

Put Math-Shepherd and OmegaPRM side by side:

| Method | Rollouts per trajectory | Human labels | GSM8K lift (best) | MATH lift (best) |
|--------|-------------------------|--------------|-------------------|-------------------|
| PRM800K | 0 (human annotates) | 800K | — | 69.6 -> 78.2 |
| Math-Shepherd | `O(K * L)` | 0 | 77.9 -> 89.1 | 28.6 -> 43.5 |
| OmegaPRM | `O(K * log L)` | 0 | — | 51.0 -> 69.4 (Gemini Pro) |

The rollout-count line is the interesting one. Trajectories with `L = 10` steps and `K = 16` cost 160 completions under Math-Shepherd and ~64 under OmegaPRM; at `L = 20`, 320 vs ~80; at `L = 40`, 640 vs ~96. For deep reasoning the gap is substantial and is the reason open reasoning recipes after 2024 default to OmegaPRM-style divide-and-conquer.

---

## §6. RLVR — skip the RM entirely

[[rlvr-tulu3]] makes the opposite move: when the task has a verifier, do not build a PRM at all. Use the verifier *as* the reward.

```
r(x, y) = v(x, y) in {0, 1}
```

That is the whole contribution. Three verifier domains ship in the Tulu-3 open-instruct RLVR pipeline:

- **Math.** Extract the final numeric or symbolic answer from the completion; grade against the reference with SymPy equivalence on MATH or normalized string match on GSM8K.
- **Constrained instruction following.** IFEval-style constraints ("respond in JSON", "use exactly three bullets") checked by regex and format parsers.
- **Code.** Execute the model's code against unit tests in a sandboxed runner; reward is `1` iff every test passes, `0` otherwise.

The Tulu-3 model report (verbatim from [[tulu-3]] §RLVR) fixes the PPO configuration at:

```
LR                3e-7
beta (KL coef)    0.05
clip epsilon      0.2
PPO epochs (K)    4
minibatches (N)   1
GAE lambda        0.95
gamma             1.0   (episodic)
local mini batch  32
local rollout     32
total episodes    10,000,000
reward            verifier output in {0, 1}
```

Connect these to ch-43. `beta = 0.05` is a mid-range KL coefficient: the paper's PPO is adding `-beta * log(pi / pi_ref)` per token to the verifier reward, which is exactly the k3 estimator discussion from ch-43. `gamma = 1.0` with `GAE lambda = 0.95` is standard episodic PPO — there is no discounting within a solution because the reward only arrives at termination. The `LR = 3e-7` is an order of magnitude below a typical SFT LR precisely because a 0/1 reward with small KL is a high-variance, low-signal stream; large steps blow up entropy fast (ch-43 again). Measured RLVR gain over the DPO-only Tulu-3 checkpoint: `+5 - 10pp` on GSM8K, `+~4pp` on IFEval, neutral-to-positive elsewhere.

**Why it sidesteps hacking.** The verifier is a fixed, interpretable function; there is no proxy RM to drift and no out-of-distribution region where the reward spuriously rises. Goodhart's gap (ch-42) is *mechanically zero* on verifiable prompts. It is not zero on verifier *bugs* — Tulu-3 notes that string-match math graders that accept "42" inside prose can be gamed, and the failure mode is to treat verifier engineering like unit-test engineering.

**Prompt curation.** Only prompts with (a) a verifier implementation and (b) a known reference answer enter the RLVR set. Everything else goes through DPO (ch-41) or is dropped.

---

## §7. SWE-RL — difflib as a scalable RL substrate

The biggest scaling question for RLVR is: where does the verifier come from? Math and code have obvious graders; most tasks do not. [[swe-rl]] (Meta, 2025) answers with a surprising trick for software-engineering RL.

**Reward.**

```python
import difflib
# predicted_patch, ground_truth_patch: unified diffs (strings)
r = difflib.SequenceMatcher(None, predicted_patch, ground_truth_patch).ratio()
# r in [0, 1]; continuous, no execution required.
```

No unit-test execution. No sandbox. No RM. Just the Python stdlib's longest-common-subsequence-style similarity ratio between the model's output patch and the human PR diff. The paper ablates continuous `r in [0,1]` vs binary-thresholded `r > tau`; continuous wins because it gives dense signal on every sample whereas execution reward is sparse (many tests fail for unrelated reasons).

**Training.** GRPO (ch-40) with group size `G = 8`, KL `beta = 0.02`, LR `1e-6`, 11M scraped (issue, code-context, ground-truth-patch) triples, on Llama-3.1-70B.

**Headline result.** Llama3-SWE-RL-70B reaches 41.0% on SWE-Bench Verified — open SOTA at release, beating DeepSeek-Coder-V2-Instruct (18.0%) and matching SWE-Gym-32B. The provocative finding is transfer: the same RL run moves HumanEval+ by +6, MATH by +4, BBH by +3 — training on GitHub patches generalises out-of-domain.

**Why it matters for the taxonomy.** SWE-RL proves that "verifiable reward" does not have to mean "unit test passes." Any rule that you can compute automatically from `(prediction, reference)` counts. A template match, an AST equivalence check, a diff ratio, a regex match — each is a candidate RLVR reward for a domain that previously required a learned RM.

---

## §8. The decision tree — when to use which

| Task property | Choose | Why |
|---------------|--------|-----|
| No verifier, short outputs, preferences available | Preference RM + PPO/DPO (ch-41) | No cheaper signal. |
| Verifier exists, short outputs (math, code, constraints) | RLVR (Tulu-3 hyperparameters) | Zero Goodhart gap; no RM to train. |
| Verifier exists, long chain-of-thought | RLVR + PRM as shaped reward (Math-Shepherd / OmegaPRM) | Dense intermediate signal; locates the bad step. |
| Reference output exists but no grader (patches, summaries) | Rule-based similarity reward (SWE-RL pattern) | Cheap, continuous, executes without sandbox. |
| Preference data + verifier both exist | RLVR on verifiable slice + DPO on rest (Tulu-3) | Mix prompt-by-prompt; same policy network. |

Two calibrations carried from adjacent literature. [[rlvr-beyond-base-model]] (Yue 2025) reports that RLVR often lifts `pass@1` while the base model wins at large `k` — treat verifiable-reward gains as *sampling-efficiency* improvements by default, and track both `pass@1` and `pass@large-k`. [[prorl]] (Liu 2025) counters that prolonged RL + KL control + reference-policy resets can expand the reasoning boundary; do not conclude RL has saturated from a short run. Both calibrations hit the ch-46 lab hardest, where the sweep across `KL in {0.01, 0.05, 0.1}` makes or breaks the memo.

---

## §9. Where process supervision meets preference optimisation

[[step-dpo]] is the bridge between this chapter and ch-41. Step-DPO keeps DPO's functional form but feeds it `(prefix, good-step, bad-step)` triples instead of full trajectories:

```
L_StepDPO = -log sigma( beta * log[pi_theta(y_w | x) / pi_ref(y_w | x)]
                       - beta * log[pi_theta(y_l | x) / pi_ref(y_l | x)] )
```

where `x` is a multi-step prefix and `y_w`, `y_l` are single step continuations. Qwen2-7B-Instruct on MATH: 53.0 -> 58.6 with 10K step-preference pairs — beats full-trajectory DPO on 100K pairs (54.3). The gradient concentrates on the actual point of disagreement, not on the identical tokens before and after. Step-DPO is not a PRM; it is a PRM's step-preference cousin. You can build it from OmegaPRM output by pairing each step where `MC(s_t)` dropped sharply with a neighbouring step where it did not.

---

## §10. Companion figure

The `prm-vs-orm.html` companion (`figures/prm-vs-orm.html`) is the chapter's hands-on deliverable. Two panels:

1. **Eval-delta panel.** Three sliders — `base_model_capability in [0, 100]` (the pass@1 before RL), `prm_quality in [0, 1]` (roughly the AUC of the PRM as a step classifier), `N_best_of in [1, 2048]`. It plots PRM, ORM, and RLVR predicted eval scores under a simple monotonic model calibrated to the Lightman 2023 and Tulu-3 numbers in this chapter. The point of the panel is not quantitative prediction; it is to build intuition for *which* lever matters most at which base-model regime.
2. **Label-cost panel.** A bar chart of `label_cost_per_1pct_eval_gain` for four methods: PRM800K (human), Math-Shepherd (MC rollouts), OmegaPRM (divide-and-conquer MC), RLVR (zero labels, only verifier engineering). Lets the reader see that the no-label verifier beats every PRM on `$/pt` *when the task is verifiable* — and does nothing when it is not.

Open it alongside this page and drag the sliders before finishing §8.

---

## Key takeaways

- **Outcome labels are leaky on long chains.** A wrong step plus a cancelling wrong step looks correct; a pure chain with a late slip looks wrong. PRMs decompose credit at step granularity.
- **[[prm800k]] sets the label schema** — `{+1, -1, 0}` per step, product (paper) or min (Math-Shepherd) aggregation. Human labels cost ~10x outcome labels — active learning is not optional.
- **[[math-shepherd]] automates the labels** via `MC(s_t) = (1/K) sum I[rollout reaches gold]`; `K = 8-16`; soft labels; PPO shaping `R_total = r_final + lambda * sum PRM(step_t)`.
- **[[omegaprm]] divides-and-conquers** the scan from `O(K L)` to `O(K log L)`; enables 1.5M labels at Google-DeepMind scale; soft MC regression makes the `tau` threshold affect only the recursion, not the targets.
- **[[rlvr-tulu3]] skips the RM.** `r(x, y) = v(x, y) in {0,1}` with PPO LR `3e-7`, KL `0.05`, clip `0.2`, 10M episodes. Goodhart gap is mechanically zero on verifiable prompts; the new risk is verifier bugs.
- **[[swe-rl]] proves rule-based rewards scale** via `difflib.ratio()` over patches: 11M triples, GRPO, 41.0% SWE-Bench Verified, out-of-domain transfer to math and BBH.
- **Decision tree:** verifier + short output -> RLVR; verifier + long chain -> RLVR + PRM shaping; reference-only -> similarity reward; neither -> fall back to preference RM (ch-41). Always track `pass@1` and `pass@large-k` ([[rlvr-beyond-base-model]]).
