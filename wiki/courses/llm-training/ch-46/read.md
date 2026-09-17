<!-- chapter: ch-46
     track: rl
     kind: lab
     title: Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention
     deps: [ch-45d]
     sources: [[dpo]], [[grpo]], [[dapo]], [[negative-sample-reinforcement]], [[likelihood-displacement]], [[rls-razor]], [[rlvr-beyond-base-model]], [[mmlu-pro]], [[ifeval]], [[xstest]], [[dr-grpo]], [[entropy-mechanism-llm-rl]], [[spurious-rewards-rlvr]], [[trl-grpo]], [[openrlhf-dpo]], [[openrlhf-ppo]], [[verl-grpo]]
     figures: figures/rl-sweep.html
     revised: 2026-09 (generality revision)
-->

# Chapter 46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention

> **Core insight.** A preference or RL stage can raise its target metric and lower general capability at the
> same time, and the two effects are measured with different instruments. On Qwen2.5-7B under GRPO, pass@1 on
> Omni-MATH-Train rises 9.9 → 26.1 → 33.6 → 42.5 across base, step 150, step 300 and step 450 while pass@256
> falls 67.2 → 66.3 → 65.3 → 64.3 ([[rlvr-beyond-base-model]], App. C.5 Table 4). On Qwen2.5-Math-7B, training
> only on incorrect rollouts reaches MATH pass@1 75.7 with pass@256 96.9, equal to the base model, while
> training only on correct rollouts reaches pass@1 74.1 with pass@256 91.2
> ([[negative-sample-reinforcement]], Table 1). A run reported as pass@1 alone cannot distinguish these cases.
>
> **Guideline.** When a DPO or RLVR stage is run, measure three quantities before and after: the in-domain
> target, a held-out general-capability suite that the stage did not train on, and coverage at large k on the
> target domain, because the third can fall while the first rises ([[rlvr-beyond-base-model]], Table 4).
> When the deliverable is a claim that negatives contributed, run the positive-only, negative-only and full
> cells at matched steps, because the measured ordering differs by k ([[negative-sample-reinforcement]],
> Table 1). When the run must be attributed to the reward rather than to the base model's prior, repeat one
> cell on a second model family, because on Qwen2.5-Math-7B random rewards gave +21.4 MATH-500 points against
> +29.1 for ground truth while OLMo2 stayed flat ([[spurious-rewards-rlvr]], §2, §3).

---

## Why this chapter matters for a general-purpose model

This is the lab for the RL track. The chapters before it taught the objectives — [[dpo]] in ch-39,
group-baseline RL in ch-40, entropy and KL control in ch-43, negative gradients in ch-43a, verifiable rewards
in ch-44 — and the stage recipes in ch-45a through ch-45d. The lab checks whether the reader can run one of
those stages and produce evidence about what it did to the model as a whole, not only to the metric it
optimized.

Position in the pipeline: this is the last post-training stage before evaluation. Whatever narrowing it
introduces is what the evaluation suite of ch-47 will measure and what ships. Two narrowing mechanisms are
documented and are measured differently:

1. **Coverage loss inside the target domain.** The set of problems the model can solve within k attempts
   shrinks as pass@1 grows ([[rlvr-beyond-base-model]], Table 4). Measured with pass@k at large k.
2. **Forgetting outside the target domain.** Capability acquired earlier degrades. In [[rls-razor]], forgetting
   across RL and SFT runs on Qwen2.5-3B-Instruct collapses onto a single curve when plotted against the
   forward KL to the base policy evaluated on the new task, with a quadratic fit of R² = 0.71 for the LLM
   experiments and R² = 0.96 in a controlled MNIST setting (§4, Figure 11, Figure 3). Measured with a
   held-out suite plus that KL.

The deliverable of this lab is therefore not a checkpoint with a higher target score. It is a memo that
reports in-domain gain against held-out retention, with the sweep cells placed on a KL axis.

---

## §1 What the lab produces

Three artifacts, each reproducible by a reader who has the repository.

1. **A sweep.** One directory per cell, each containing `metrics.jsonl` (per-step training signals),
   `eval_before.json` and `eval_after.json` (the retention suite), `passk.json` (pass@k on the target
   domain), the final checkpoint, and the commit hash of the training code.
2. **Two plots.** (a) in-domain target against held-out suite average, one point per cell; (b) held-out suite
   average against forward KL to the reference policy measured on the training prompts, one point per cell.
   Plot (b) is the [[rls-razor]] measurement applied to the reader's own cells (§4, Figure 3 middle).
3. **A memo**, `rl-experiment-memo.md`, with: setup; the three-axis table; the negative-signal ablation
   result; one failure post-mortem with a minimal reproduction; one prediction that the data contradicted.

Predictions for every cell are written and committed before the first run. This is a check on the analysis,
not on the model: a cell whose direction was predicted and confirmed supports a mechanism, and a cell that
contradicted the prediction identifies a gap in it.

**Option A** is offline preference optimization with [[dpo]]. **Option B** is RLVR with a group baseline
([[grpo]], with the normalization of [[dr-grpo]]). One option is chosen, not both; the ablation and the
retention suite are the graded parts and they cost more than the training run.

---

## §2 Option A — DPO with a β sweep and a displacement gate

### 2.1 The objective and what it moves

The DPO loss is (Eq. 7 of [[dpo]]):

```
L_DPO = −E_(x, y_w, y_l) ~ D [ log σ( β log (π_θ(y_w|x)/π_ref(y_w|x)) − β log (π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

- `x`: prompt. `y_w`: preferred response. `y_l`: dispreferred response.
- `π_θ`: the policy being trained. `π_ref`: the frozen reference, normally the SFT checkpoint.
- `β`: the coefficient of the implicit KL constraint. `σ`: the logistic function.
- The implicit reward is `r̂_θ(x, y) = β log (π_θ(y|x)/π_ref(y|x))` (§4, §5.1 of [[dpo]]).

The gradient is `∇_θ L = −β·E[ σ(r̂_θ(x,y_l) − r̂_θ(x,y_w)) · (∇_θ log π(y_w|x) − ∇_θ log π(y_l|x)) ]`
(App. A.4, Eq. 21–22 of [[dpo]]). It contains one term that raises the chosen log-probability and one that
lowers the rejected log-probability. Nothing in the loss requires the first term to win: only the difference
appears.

### 2.2 The measurement problem the sweep must solve

The quantity most frameworks log is the implicit reward, not the log-probability. OpenRLHF's `DPOLoss`
returns `chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()` and the matching
rejected value, and `DPOTrainer` logs `loss`, `acc`, `chosen_reward`, `reject_reward`, `nll_loss`; the raw
chosen and rejected log-probabilities are not in the logs ([[openrlhf-dpo]], loss.py L264–281,
dpo_trainer.py L184–197).

Worked example. Suppose for one pair, at initialization, `log π_θ(y_w|x) = log π_ref(y_w|x) = −40` and
`log π_θ(y_l|x) = log π_ref(y_l|x) = −42`. After training, `log π_θ(y_w|x) = −44` and
`log π_θ(y_l|x) = −50`. With β = 0.1:

| quantity | before | after |
|---|---|---|
| implicit reward, chosen | 0.0 | 0.1 × (−44 − (−40)) = −0.4 |
| implicit reward, rejected | 0.0 | 0.1 × (−50 − (−42)) = −0.8 |
| margin (chosen − rejected) | 0.0 | +0.4 |
| accuracy (`chosen_reward > reject_reward`) | — | passes |

The margin rose and the accuracy metric passes, while the chosen response became `e^{-4} ≈ 0.018` times as
likely as it was under the reference. This is likelihood displacement. In [[likelihood-displacement]], DPO on
on-policy refusal pairs lowered the training-set refusal rate of Llama-3-8B-Instruct from 74.4% to 33.4% and
of Gemma-2B-IT from 80.5% to 54.8%, with mean change in preferred log-probability −48.1 ± 22.1 and
−59.2 ± 5.3 over three runs (§6.2, Table 16). A margin-only acceptance gate passes such a run.

The lab therefore logs, in addition to the framework metrics: `logps/chosen` and `logps/rejected` as absolute
values, and the mean log-probability of a fixed held-out set of 200 responses the model should keep
producing (for example correct answers to non-target math problems, and refusals of unsafe prompts).

### 2.3 Sweep axis and the two toggles

- **Axis:** β ∈ {0.05, 0.1, 0.5}. All three values appear in [[dpo]]: 0.05 and 0.1 in the IMDb reward–KL
  frontier sweep (§6.1, Fig. 2 left, 22 runs), 0.1 as the value used in all runs unless noted, and 0.5 for
  the GPT-J TL;DR run (App. B).
- **Toggle 1, the NLL anchor.** An added term `−λ log π_θ(y_w|x)` contributes `λ‖∇ log π_θ(y_w|x)‖² ≥ 0` to
  the time derivative of the chosen log-probability (App. E, Proposition 1 of [[likelihood-displacement]]),
  so it cannot make displacement worse. OpenRLHF exposes it as `--model.nll_loss_coef`, default 0
  ([[openrlhf-dpo]], train_dpo.py L246; dpo_trainer.py L169–180, at commit 64c1cc4). Run one cell with it on.
- **Toggle 2, CHES filtering.** In [[likelihood-displacement]], keeping the 5% of pairs with the lowest
  length-normalized CHES score restored refusal rates more than the SFT term did, and keeping up to 15% gave
  analogous results (§6.3, footnote 9). This toggle is optional; it costs one forward pass over the dataset.

Direction of β is fixed by the algebra, not by a guess: the implicit reward is β times a log-ratio, so at a
smaller β the policy must move further in log-probability space to produce the same implicit-reward margin.
The KL measured against `π_ref` should therefore be largest in the β = 0.05 cell.

---

## §3 Option B — RLVR with a KL sweep and a negative-signal ablation

### 3.1 The objective

The group-baseline objective is Eq. 3 of [[grpo]]:

```
J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t { min[ ρ_i,t Â_i,t , clip(ρ_i,t, 1−ε, 1+ε) Â_i,t ] − β D_KL[π_θ ‖ π_ref] } ]
ρ_i,t = π_θ(o_i,t | q, o_i,<t) / π_θold(o_i,t | q, o_i,<t)
Â_i,t = (r_i − mean(r)) / std(r)
```

- `q`: question. `o_i`: the i-th of G sampled responses. `|o_i|`: its length in tokens.
- `ρ_i,t`: the per-token importance ratio. `ε`: the clip parameter. `β`: the KL coefficient.
- `Â_i,t`: the advantage assigned to every token of `o_i`; `r_i` is the verifier's binary reward.

[[dr-grpo]] shows that the `1/|o_i|` factor makes the per-token penalty on an incorrect response smaller the
longer that response is, so the policy comes to prefer long incorrect responses (§3.1), and that removing
both `1/|o_i|` and the `std(r)` division lowers the length of incorrect responses on benchmarks (§3.2,
Fig. 5). The lab uses the unbiased normalization by default so that the KL sweep is not confounded with the
length bias; `loss_type="dr_grpo"` with `scale_rewards="none"` in TRL implements it
(`loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)`,
[[trl-grpo]] grpo_trainer.py L2548–2562; advantage scaling L2127–2149).

### 3.2 The two axes

**Axis 1 — KL coefficient β ∈ {0.0, 0.001, 0.04}.** Every value is a value some source used:
0.0 in all Dr. GRPO runs (App. G Table 6 of [[dr-grpo]]) and as TRL's default ([[trl-grpo]], L650–653);
0.001 for the PPO and GRPO baselines of [[negative-sample-reinforcement]] (App. D.1); 0.04 for
DeepSeekMath-RL 7B (§4.2 of [[grpo]]). This axis is what plot (b) needs: it moves the policy's distance from
the reference, which is the variable [[rls-razor]] identifies as the predictor of forgetting.

**Axis 2 — the training signal, at matched steps.** Four cells:

| cell | which rollouts produce gradient | implementation |
|---|---|---|
| positive-only (PSR) | reward = 1 only | mask the loss of rollouts with `r = 0` |
| negative-only (NSR) | reward = 0 only | mask the loss of rollouts with `r = 1` |
| full | both, group-normalized advantage | unmodified objective |
| W-REINFORCE, λ = 0.1 | both, positive term scaled by 0.1 | multiply the positive-rollout loss by λ |

`L_W-REINFORCE = −E[Σ_{y: r=1} λ·π_θ(y|x)] − E[Σ_{y: r=−1} −π_θ(y|x)]`, Eq. 9 of
[[negative-sample-reinforcement]]; λ = 1 recovers REINFORCE and λ = 0 recovers NSR.

The result to reproduce or contradict, from Table 1 of [[negative-sample-reinforcement]] (Qwen2.5-Math-7B,
MATH training set, 8 rollouts per prompt, evaluated with 256 samples at temperature 0.6, top-p 0.95):

| signal | MATH pass@1 | MATH pass@256 | AIME 2025 pass@1 | AIME 2025 pass@256 |
|---|---|---|---|---|
| base model | 63.2 | 96.9 | 6.1 | 46.7 |
| PSR (positive only) | 74.1 | 91.2 | 11.6 | 43.3 |
| NSR (negative only) | 75.7 | 96.9 | 10.0 | 53.3 |
| GRPO (full) | 76.3 | 95.5 | 10.3 | 50.0 |
| W-REINFORCE (λ = 0.1) | 76.6 | 96.7 | 10.6 | 56.7 |

[figures/rl-sweep.html](figures/rl-sweep.html) lets the reader switch benchmark and read any single k off the
curves, which is the operation this section asks for. Panel 1 plots all seven rows of Table 1 at every k from
1 to 256 on MATH, AIME 2025 and AMC23; panel 2 plots pass@1 against pass@256 across GRPO training steps from
Table 4 of [[rlvr-beyond-base-model]] for three evaluation sets; panel 3 is the λ sweep of App. E Table 4.
Every value in it is read from a published table, and it is the reference against which the lab's own
ablation curves are read.

### 3.3 Verifier and prompts

The reward is a programmatic verifier returning {0, 1}. Before any training, the verifier is checked on a set
of hand-labeled `(response, gold)` pairs and must reproduce every label; a verifier with a hole makes the
reward and the metric disagree in a way that no training-side diagnostic will show. Answer extraction must
match the dataset's own format: a dataset whose solutions end in `\boxed{...}` has no `####` separator, and a
splitter written for GSM8K's `####` convention silently returns the whole solution text as the gold answer.

---

## §4 Instrumentation

Per step, for both options:

| signal | Option A source | Option B source |
|---|---|---|
| loss | trainer | trainer |
| implicit reward, chosen and rejected | `chosen_reward`, `reject_reward` ([[openrlhf-dpo]] L191–192) | — |
| absolute `logps/chosen`, `logps/rejected` | added by the lab (not logged by default) | — |
| greedy-output drift against `π_ref` | added by the lab; every N steps, see §6.1 item 3 | added by the lab; same procedure |
| log-probability of the held-out responses the model should keep | added by the lab, 200 fixed responses | added by the lab, same set |
| reward mean and std | — | reward function output |
| fraction of zero-variance groups | — | `frac_reward_zero_std` ([[trl-grpo]] L2150, L2186) |
| policy entropy | requires generation; see below | logged every step ([[trl-grpo]] L2571–2612) |
| clip fraction split by advantage sign | — | low-clip counted only where A < 0, high-clip only where A > 0 ([[trl-grpo]] L2589–2590) |
| KL to reference | requires generation; see below | k3 estimator, `exp(ref − logp) − (ref − logp) − 1` ([[trl-grpo]] L2493–2497) |
| sampler-versus-trainer KL | — | `vllm_kl` = mean(rollout_log_probs − old_log_probs) ([[openrlhf-ppo]] loss.py L173) |

Option A has no rollouts, so entropy and KL to the reference are not available from the training loop. They
are produced by a separate generation pass every N steps on a fixed prompt set: sample 8 completions per
prompt from the current checkpoint and from `π_ref`, and compute the forward KL
`E_{x∼τ}[KL(π_ref‖π_θ)]` on that set, which is the quantity [[rls-razor]] uses as its predictor (§6,
Table 1). Using the training-loop implicit reward as a stand-in for KL is not valid: it is
`β·(log π_θ − log π_ref)` on two specific responses, not an expectation over the policy.

Slice the evaluation. Aggregate improvement can coexist with per-slice regression, so the held-out target
evaluation is reported per difficulty bucket (Option B) or per prompt source (Option A).

---

## §5 The held-out retention suite and the SFT control

### 5.1 Suite

Run before and after, with identical decoding settings on both sides:

| component | what it covers | fixed protocol |
|---|---|---|
| [[mmlu-pro]] | knowledge and multi-step reasoning outside the target domain | 5-shot CoT; one fixed prompt |
| [[ifeval]] | explicit output-constraint following | report prompt-level and instruction-level, strict and loose |
| a code set | a capability neither option trains | the same set used before and after |
| [[xstest]] | over-refusal: 250 safe prompts plus 200 unsafe contrasts | temperature 0, 256 max new tokens |
| non-target math (Option A) | in-family transfer | fixed sample count |

Measurement error is part of the result. [[mmlu-pro]] reports that across 24 prompt styles a model's score
varies by about 2 points (maximum 3.74), against 4–5 points (maximum 10.98) on MMLU (§6.3), so a retention
difference below roughly 2 points on MMLU-Pro is not resolved by a single-prompt measurement. [[mmlu-pro]]
also reports that chain-of-thought changes GPT-4o's score by 19.1 points (§6.2, Table 3), which is why the
generation configuration must be identical on both sides rather than merely "similar". [[xstest]] reports
that removing Llama-2's system prompt moved full refusal of safe prompts from 38% to 14% while full refusal
of unsafe prompts moved only from 99.5% to 97.5% (Table 2), so the system prompt is part of the protocol.

### 5.2 The SFT control

Run one additional cell: supervised fine-tuning on the same data the preference or RL stage used — the chosen
responses for Option A, the correct rollouts for Option B — at a learning rate tuned to reach a comparable
in-domain score. [[rls-razor]] reports that on Qwen2.5-3B-Instruct across math, science Q&A and tool use, RL
Pareto-frontier models keep prior-benchmark performance nearly unchanged as new-task accuracy rises while SFT
models reach the same new-task accuracy only with substantial forgetting (§3, Figure 2), and that both fall
on one curve when plotted against forward KL (§4). The control tells the reader whether the retention
observed in the sweep came from the algorithm or from the size of the distribution shift it happened to take.

### 5.3 Coverage

Compute pass@k on the target domain for the base, the SFT control and every sweep cell, using the unbiased
estimator of [[negative-sample-reinforcement]] (Eq. 5): with `n` samples per problem and `c` correct,
`pass@k = 1 − C(n−c, k)/C(n, k)`.

Worked example. `n = 16` samples, `c = 2` correct, `k = 8`: `C(14,8) = 3003`, `C(16,8) = 12870`, so
`pass@8 = 1 − 3003/12870 = 1 − 0.2333 = 0.7667`. Computing pass@8 from a single draw of 8 samples would give
either 0 or 1 for this problem; the estimator over 16 samples gives 0.767 and has far lower variance.

k must reach at least 64 for the ordering in §3.2 to be visible: on MATH, PSR and NSR differ by 1.6 points at
k = 1 and by 5.7 points at k = 256 ([[negative-sample-reinforcement]], Table 1).

---

## §6 Acceptance gates and the failure post-mortem

### 6.1 Acceptance gates

A gate is a condition written before the run, evaluated after it, and reported whether or not it passed. Each
option has four; all four must hold for a cell to be described as a successful run of that stage.

**Option A (DPO).**

1. **Margin.** The mean implicit-reward margin `r̂(x, y_w) − r̂(x, y_l)` is positive and rising on a held-out
   slice of pairs. This is necessary and not sufficient: §2.2 shows a run in which the margin rises while both
   log-probabilities fall.
2. **Displacement gate.** The mean of `log π_θ(y_w|x) − log π_ref(y_w|x)` over the training pairs is recorded
   and compared against a threshold fixed before the run. The regime to avoid is the one measured in
   [[likelihood-displacement]]: mean change in preferred log-probability of −48.1 ± 22.1 for
   Llama-3-8B-Instruct and −59.2 ± 5.3 for Gemma-2B-IT over three runs, in the same runs whose training-set
   refusal rates fell from 74.4% to 33.4% and from 80.5% to 54.8% (§6.2, Table 16). The paper does not report
   a threshold that separates safe from unsafe runs, so the lab's number is a pre-registered choice, and the
   observed value is reported regardless of whether it passed.
3. **Greedy-output drift.** On a fixed set of 200 prompts, decode greedily from `π_ref` and from `π_θ` with
   identical settings, and report two quantities: the fraction of prompts whose greedy continuation changed at
   all, and the mean normalized edit distance between the two continuations. The reason to log this alongside
   the log-probabilities is that the two can move independently. A run with a positive margin, a stable
   `log π_θ(y_w|x)`, and near-zero greedy drift has sharpened a preference without changing what the model
   emits. A run with a positive margin, a falling `log π_θ(y_w|x)`, and high greedy drift toward a different
   response class is the displacement case, and the drift measurement is what names the class it moved to.
4. **Held-out log-probabilities.** The mean log-probability of the 200 held-out responses of §2.2 does not
   fall. This is the gate that would have caught the refusal collapse in [[likelihood-displacement]], because
   refusals are exactly the responses that lost probability there.

**Option B (RLVR).**

1. **Reward agreement.** Training reward and a strict re-scoring of the same completions agree within a
   tolerance stated before the run, on a held-out slice.
2. **Entropy floor and group variance.** Policy entropy stays above a floor fixed before the run, and
   `frac_reward_zero_std` stays below a stated fraction. Both are pre-registered because
   [[entropy-mechanism-llm-rl]] reports that 73% of the entropy consumption and 76% of the performance gain
   occurred in the first 200 of 2400 gradient steps across 11 base models (§2.3), so a floor checked only at
   the end of training is checked after the decision it governs has already been made.
3. **Coverage.** pass@64 on the target domain is not below the base model's pass@64 by more than a stated
   tolerance. The ordering that makes this gate non-redundant with pass@1 is in §3.2.
4. **Retention.** The held-out suite average is not below its before-value by more than the measurement error
   of §5.1.

### 6.2 The failure post-mortem

The memo contains one post-mortem, worked to a reproduction the reader can run. Five options; a cell that
passed every gate still qualifies if one of these signatures appeared and was then removed.

| Failure | Signal pattern | Minimal reproduction | Status of the reproduction |
|---|---|---|---|
| Reward hacking | Training reward rises; strict re-scoring of the same completions does not; the gap between them widens | Replace the verifier's exact-answer comparison with a substring test that accepts the gold string anywhere in the response, and rerun a fixed number of steps | Lab construction; no source in this chapter's set reports this specific loophole |
| Entropy collapse | Entropy falls monotonically; `frac_reward_zero_std` rises; reward plateaus | Set `ε_high = ε_low = 0.2` with no entropy bonus, and contrast with a cell at `ε_high = 0.28` | Attested: clip-higher moved AIME24 avg@32 from 36 to 38 ([[dapo]], Table 1); Clip-Cov and KL-Cov moved the Qwen2.5-32B 7-benchmark average from 45.8 to 50.3 and 52.2 ([[entropy-mechanism-llm-rl]], Table 2) |
| Length bias | Mean length of *incorrect* responses grows while accuracy is flat | Switch `loss_type` from `"dr_grpo"` back to `"grpo"`, restoring the `1/\|o_i\|` and `std(r)` normalizations that the lab default removes | Mechanism attested ([[dr-grpo]], §3.1); removing the two terms lowered incorrect-response length (§3.2, Fig. 5). The step at which the growth becomes visible is not reported; record it from your own run |
| Likelihood displacement or over-suppression | Margin rises while both `logps/chosen` and `logps/rejected` fall; greedy drift high; held-out refusal or correct-answer rate falls | Build pairs whose chosen and rejected responses are near-duplicates (high CHES) and train at the β = 0.05 cell; then re-run with the NLL anchor on | Attested at 2B and 8B ([[likelihood-displacement]], §6.2 Table 16); the NLL term's contribution to the chosen log-probability is non-negative (App. E, Proposition 1) |
| Capability regression on non-target evals | In-domain target rises; held-out suite falls by more than the §5.1 measurement error; forward KL to the reference is large | Run the SFT control of §5.2 at a learning rate that reaches the same in-domain score, and place both points on the KL axis of plot (b) | Attested on Qwen2.5-3B-Instruct across math, science Q&A and tool use ([[rls-razor]], §3 Figure 2, §4) |

Two of these five are distinguished only by instrumentation the lab adds. Reward hacking and capability
regression both present as "the number went up and the model got worse", and the measurement that separates
them is whether the *training* reward and a strict re-scoring disagree. Entropy collapse and length bias both
present as a plateau, and the measurement that separates them is whether the plateau is accompanied by falling
entropy or by growing incorrect-response length.

---

## §7 Resource-constrained path

The instrumentation, the retention suite and the post-mortem are the graded parts of this lab and they do not
shrink. What shrinks is the model and the number of sweep axes.

| Element | Full path | Constrained path | Why the change is safe |
|---|---|---|---|
| Model | 3B–7B | 1B | The narrowing mechanisms of §1 are reported at 0.5B upward: [[entropy-mechanism-llm-rl]]'s protocol spans 0.5B to 32B (§2.2) |
| Option | A or B | B only | Option B's ablation is the deliverable the chapter is named for, and it is the one that needs rollouts |
| Sweep axes | KL axis × signal axis | signal axis only, at a single `β_KL = 0.0` | The KL axis is the one that plot (b) needs; with one axis, plot (b) is replaced by two points, the RL cell and the SFT control |
| Ablation cells | positive-only, negative-only, full, W-REINFORCE | negative-only and full | These are the two cells that differ most at large k in the reference result: MATH pass@256 96.9 for NSR against 95.5 for GRPO ([[negative-sample-reinforcement]], Table 1) |
| pass@k | k ∈ {1, …, 64} from 64 samples | unchanged | k must reach 64 for the §3.2 ordering to be visible |
| Retention suite | [[mmlu-pro]], [[ifeval]], a code set, [[xstest]] | unchanged | Dropping a component removes the only evidence for a claim about that capability; if compute forces a drop, the memo states which capability is now unmeasured rather than reporting the average of what remains |
| Post-mortem | one of five | unchanged | — |

A run on the constrained path cannot support a claim about scale, and the memo says so. A reward that moves a
1B model may do nothing at 7B, and the reverse also occurs: in [[spurious-rewards-rlvr]], random rewards raised
Qwen2.5-Math-7B's MATH-500 accuracy by 21.4 points against 29.1 points for ground-truth rewards, while
OLMo2 stayed flat under the same signals (§2, §3). One model at one size is evidence about that model at
that size.

---

## Negative samples and negative feedback

Both options use negatives **as gradient** (type 4 of the course standard): an explicit decrease in the
likelihood of a specific sample. Neither uses negatives as content or as conditioning; those appear in ch-29d
and ch-31.

**Where the negatives come from.** Option A: the `rejected` side of a preference pair, labeled by a human or a
judge. Option B: rollouts the verifier scored 0. The false-negative rate of a verifier is the false-negative
rate of the negatives; a tolerant grader that rejects a correctly formatted answer manufactures negatives out
of correct reasoning.

**Mechanism.** For the softmax, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, so a push-down on the sampled token
moves mass to the other tokens. [[negative-sample-reinforcement]] gives the sign-resolved form (Eq. 7, 8):

```
PSR: −∂L/∂z_v ∝  π_v(1 − π_v)   if v = y_t ;  −π_{y_t}·π_v   otherwise
NSR: −∂L/∂z_v ∝ −π_v(1 − π_v)   if v = y_t ;  +π_{y_t}·π_v   otherwise
```

Worked example, single position, three tokens with `π = (0.90, 0.07, 0.03)` and the sampled token being the
first one inside an incorrect response. The NSR push-down on it scales as `π(1−π) = 0.90 × 0.10 = 0.09`; the
same token at `π = 0.30` would receive `0.30 × 0.70 = 0.21`, more than twice as much. The mass goes to the
other two in proportion to their current probabilities: `0.90 × 0.07 = 0.063` and `0.90 × 0.03 = 0.027`, a
ratio of 7:3, the same ratio the model already held. This is why the paper describes NSR as suppressing an
error without erasing high-confidence pretrained structure (§4.2, Interpretation).

**What current practice does with them, and the measured size of the effect.** Table 1 of
[[negative-sample-reinforcement]], reproduced in §3.2, is the measurement: at Qwen2.5-Math-7B scale on MATH,
the negative-only signal alone reaches pass@1 75.7 against 76.3 for full GRPO, and it is the only setting
that leaves pass@256 at the base model's 96.9. The positive term is not inert — λ = 0.1 beat λ = 0 at pass@1
on MATH (76.6 against 75.7) and λ = 1 was worse at every k (App. E Table 4). The paper does not report a
percentage split of the gain between positive and negative signals, so this chapter makes no such claim.

**The opposite result, and why both can hold.** [[likelihood-displacement]] reports that the negative term of
a preference loss can move mass to responses of the opposite meaning, dropping Llama-3-8B-Instruct's refusal
rate from 74.4% to 33.4% (§6.2). The settings differ: DPO pairs are often two responses of the same kind, and
the paper's mechanism is driven by the similarity of the two responses' embeddings, measured by the CHES
score (§4.2.2, Definition 2); RLVR negatives are rollouts the current policy produced and the reward is
computed per rollout. A reader who concludes from either paper alone that negatives are always safe or always
harmful has over-generalized one setting.

**Controls available in this lab.** Bound or weight the negative term (λ < 1 in W-REINFORCE; ε_low kept tight
while ε_high is raised, §3.1 of [[dapo]]); keep the data on-policy (Option B does, Option A does not); anchor
with a positive NLL term (Option A's toggle 1); filter pairs by CHES (Option A's toggle 2); localize by
masking truncated completions so length alone is not penalized (§3.4 of [[dapo]];
`mask_truncated_completions` in [[trl-grpo]], config L259–262).

**Diagnostics required in the memo.** Chosen and rejected log-probabilities logged separately and in absolute
terms; clip fraction, gradient norm and entropy split by advantage sign; `frac_reward_zero_std` per step;
pass@1 and pass@k at k ≥ 64; and, if the prompt set includes unanswerable or unsafe items, the abstention and
refusal rates.

**Effect on generality.** Four effects are separated because they are measured separately and, in the sources
available here, move in different directions.

1. **Coverage.** The negative gradient is the component associated with *retained* coverage in the RLVR
   setting: NSR held MATH pass@256 at the base model's 96.9 while PSR lowered it to 91.2
   ([[negative-sample-reinforcement]], Table 1).
2. **Over-refusal and the reverse.** In the preference setting, a negative gradient on rejected responses
   moved probability mass toward responses of the opposite meaning, lowering training-set refusal rates from
   74.4% to 33.4% on Llama-3-8B-Instruct ([[likelihood-displacement]], §6.2). The failure here is under-refusal
   rather than over-refusal, and the [[xstest]] component of §5.1 measures both sides: 250 safe prompts and
   200 unsafe contrasts, reported separately.
3. **Forgetting.** Negatives affect forgetting through the distance they move the policy, not directly.
   Forgetting across RL and SFT runs fell on one curve against forward KL to the base policy
   ([[rls-razor]], §4), so a negative-heavy cell is compared with the others on the KL axis of plot (b), not
   on its loss composition.
4. **Calibration and hallucination.** No source in this chapter's set reports a calibration or hallucination
   measurement for PSR, NSR or the DPO negative term, so this chapter makes no claim about them. A lab that
   wants one adds the measurement rather than inferring it from pass@k.

---

## Recipe

Lab values differ from source values only where budget forces it, and each difference is stated.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| All DPO runs unless noted | per task | preference | β | 0.1 | arXiv:2305.18290v3 App. B | verified 2026-09-14 | "we did not meaningfully tune DPO's β" (§6.2) | 0.1 (middle cell) | none |
| GPT-J SFT (CarperAI/openai_summarize_tldr_sft), TL;DR | 6B | preference | β | 0.5 | arXiv:2305.18290v3 App. B | verified 2026-09-14 | no ablation reported | 0.5 (high cell) | none |
| GPT-2-large, IMDb | not stated | preference | β sweep | {0.05, 0.1, 1, 5} | arXiv:2305.18290v3 §6.1 | verified 2026-09-14 | Fig. 2 left, 22 runs, evaluated every 100 steps | 0.05 (low cell) | β = 1 and 5 are far outside the range used for chat-model DPO; the high cell comes from the TL;DR row instead |
| All DPO runs | per task | preference | optimizer; peak LR; warmup | RMSprop; 1e-6; linear 0 → 1e-6 over 150 steps | arXiv:2305.18290v3 App. B | verified 2026-09-14 | no ablation reported | same | none |
| All DPO runs | per task | preference | batch size | 64 (unit not stated) | arXiv:2305.18290v3 App. B | verified 2026-09-14 | no ablation reported | 32 pairs | single-GPU memory; recorded as a deviation because β and batch size interact through the number of updates |
| Gemma-2B-IT, Llama-3-8B-Instruct | 2B / 8B | preference | filtered subset | 5% of samples with lowest length-normalized CHES | arXiv:2410.08847v4 §6.3 | verified 2026-09-14 | up to 15% analogous; more caused refusal drops (footnote 9) | 5% (toggle 2) | none |
| Gemma-2B-IT (DPO) | 2B | preference | SFT-term coefficient | 1 | arXiv:2410.08847v4 App. I.3 | verified 2026-09-14 | highest mean training-set refusal rate, grid {0.01, 0.1, 1}, 3 seeds | 1 (toggle 1) | none |
| DeepSeekMath-RL | 7B | RL | policy LR; β; G; max length | 1e-6; 0.04; 64; 1024 | arXiv:2402.03300 §4.2 | verified 2026-09-14 | no ablation reported | LR 1e-6; β = 0.04 (high KL cell); G = 8; max length 1024 | G = 64 is out of budget at 8 rollouts per prompt per step |
| All Dr. GRPO runs | 1.5B–7B | RL | G; temperature; top-p, top-k; clip ε; epochs | 8; 1.0; (1.0, −1); 0.2; 1 | arXiv:2503.20783v2 App. G Table 6 | verified 2026-09-14 | no ablation reported | same | none |
| All Dr. GRPO runs | 1.5B–7B | RL | KL loss coefficient | 0.0 | arXiv:2503.20783v2 App. G Table 6 | verified 2026-09-14 | no ablation in that paper | 0.0 (low KL cell) | none |
| All Dr. GRPO runs | 1.5B–7B | RL | max response length | 3000 tokens | arXiv:2503.20783v2 App. G Table 6 | verified 2026-09-14 | tied to the 4k Qwen2.5-Math context (App. B) | 1024 tokens | short-answer math prompts at 1B–3B; the length-bias diagnostic still applies |
| Qwen2.5-Math-7B (PSR/NSR/W-REINFORCE) | 7B | RL | prompt batch; rollouts; mini-batch; LR; temperature; KL | 1024; 8; 256; 1e-6; 1.0; 1e-3 for PPO and GRPO, none for PSR and NSR | arXiv:2506.01347v2 §3.1, App. D.1 | verified 2026-09-15 | no ablation reported | prompt batch 64; other values unchanged; KL = 0.001 as the middle cell | prompt batch scaled to one node; the ablation compares cells at matched steps, so the absolute batch cancels |
| Qwen2.5-Math-7B | 7B | eval-gate | samples per prompt; temperature; top-p; k range | 256; 0.6; 0.95; 1–256 | arXiv:2506.01347v2 §3.1 | verified 2026-09-15 | Pass@k unbiased estimator, Eq. 5 | 64 samples; same decoding; k ∈ {1, …, 64} | evaluation cost; k = 64 is the smallest k at which the PSR–NSR ordering in Table 1 is already clear |
| Qwen2.5-32B (DAPO) | 32B | RL | ε_low; ε_high | 0.2; 0.28 | arXiv:2503.14476v2 §4.1 | verified 2026-09-15 | Table 1: clip-higher 36 → 38 AIME24 avg@32 | only in the entropy-collapse repair cell | not a sweep axis; changing it would confound the KL axis |
| Qwen2.5-32B (DAPO) | 32B | RL | dynamic sampling | keep only prompts with 0 < correct rollouts < G | arXiv:2503.14476v2 §3.2 Eq. 11 | verified 2026-09-15 | Table 1: 42 → 50 AIME24 avg@32 | logged as `frac_reward_zero_std`; enabled only in the repair cell | it changes the effective batch, so enabling it mid-sweep breaks matched steps |
| Qwen2.5-32B | 32B | RL | KL-Cov token fraction | 10⁻⁴ to 10⁻³ of tokens | arXiv:2505.22617 §4.5 | verified 2026-09-14 | Table 2: 7-benchmark average 45.8 → 52.2 | named as a repair option; not run | requires a custom loss; out of scope for a one-axis lab |

**Starting point for a small general-purpose run.** Option B at 1B–3B: Dr. GRPO normalization, G = 8,
temperature 1.0, top-p 1.0, clip ε = 0.2, one inner epoch, LR 1e-6 constant, KL coefficient 0.0 for the first
cell — all from App. G Table 6 of [[dr-grpo]], whose runs were Qwen2.5 base models of 1.5B to 7B on 8× A100
with a 3000-token response budget. Maximum response length is reduced to 1024 for short-answer math. Option A
at the same scale: β = 0.1, RMSprop, LR 1e-6 with 150 warmup steps, from App. B of [[dpo]], whose largest
model was 6B.

---

## Generalization lens

**(a) What increases breadth.** Keeping the negative term while down-weighting the positive one: on
Qwen2.5-Math-7B, W-REINFORCE at λ = 0.1 held MATH pass@256 at 96.7 against the base model's 96.9 and reached
the best pass@1 of 76.6, while λ = 1 dropped pass@256 to 92.0 ([[negative-sample-reinforcement]], Table 1,
App. E Table 4). Sampling more rollouts per prompt: raising n from 8 to 32 at fixed prompt batch gave a
higher pass@128 after only 220 steps despite lower pass@1 ([[rlvr-beyond-base-model]], Figure 16).
Keeping the policy close to the reference: RL cells with smaller forward KL retained more prior capability at
equal new-task accuracy ([[rls-razor]], §3, §4).

**(b) What causes narrowing.** Positive-only reinforcement, which repeatedly amplifies observed correct
sequences and suppresses alternatives (Eq. 7 of [[negative-sample-reinforcement]]) — PSR dropped MATH
pass@256 from 96.9 to 91.2. Training for more steps at fixed configuration: GRPO steps 150 → 450 raised
Omni-MATH-Train pass@1 from 26.1 to 42.5 and lowered pass@256 from 66.3 to 64.3, with out-of-domain MATH500
pass@256 falling from 97.2 to 95.4 ([[rlvr-beyond-base-model]], Table 4). Entropy collapse: in
[[entropy-mechanism-llm-rl]], 73% of entropy consumption and 76% of the performance gain occur in the first
200 of 2400 gradient steps across 11 base models, and the fitted law `R = −a·exp(H) + b` implies a ceiling at
`H = 0` (§2.3; Eq. 6 in §2.4). Displacement in offline preference training (§2.2 above). Large distribution shift,
whatever the algorithm ([[rls-razor]], §4).

**(c) How to measure it for this stage.** Three instruments, none of which substitutes for another:
pass@k at k ≥ 64 on the target domain for coverage; the held-out suite of §5.1 for capability outside the
target; forward KL on the training prompts as the x-axis that makes the sweep cells comparable
([[rls-razor]], §6 Table 1, measured in the controlled ParityMNIST setting: forward KL R² = 0.96 ± 0.01
against reverse KL 0.93 ± 0.01, total variation 0.80 ± 0.01, and ≤ 0.58 ± 0.02 for every weight- or
activation-space distance tested; the LLM version of the same fit is R² = 0.71, §4 Figure 11). Known
measurement errors: prompt-style variation of about 2 points on MMLU-Pro, maximum 3.74 points
(§6.3 of [[mmlu-pro]]); the strict-versus-loose gap on IFEval, which is 2.41 and 1.80 points for GPT-4 and
3.88 and 3.35 points for PaLM 2 S at prompt level and instruction level (Table 3 of [[ifeval]]); judge and string-match disagreement on refusal classification, where string matching
gave 48.4% safe-prompt refusal for Llama2.0 against 38% + 21.6% under manual annotation ([[xstest]], Table 2).
Model-family dependence is a fourth: a reward that works on Qwen2.5 may do nothing on OLMo2
([[spurious-rewards-rlvr]], §3), so a single-family result is evidence about that family.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reporting pass@1 only | Target metric up, no statement about coverage | Compute pass@k for k ∈ {1, …, 64} with the unbiased estimator; compare the cell to the base model at the largest k |
| Accepting a DPO run on a rising margin | `rewards/margins` rises, `rewards/accuracies` high | Log absolute `logps/chosen`; a run where it falls monotonically is displacing ([[likelihood-displacement]] §6.2) |
| Using the implicit reward as a KL measurement | KL "reported" for a run with no rollouts | Generate from `π_θ` and `π_ref` on a fixed prompt set and compute forward KL ([[rls-razor]] §6) |
| Comparing ablation cells at different step counts | The positive-only cell trained longer per epoch because fewer rollouts contribute | Fix the number of optimizer steps, not the number of epochs; PSR and NSR see fewer samples per batch by construction (§3.1 of [[negative-sample-reinforcement]]) |
| Confounding the KL axis with the length bias | Length grows across all KL cells | Use `loss_type="dr_grpo"` with `scale_rewards="none"` for the whole sweep; run `loss_type="grpo"` only as the deliberate length-bias reproduction ([[dr-grpo]] §3.1, [[trl-grpo]] L2548–2562) |
| Shrinking effective batch without noticing | Reward mean rises while gradient noise rises; many groups contribute nothing | Log `frac_reward_zero_std` each step ([[trl-grpo]] L2150); the count of all-correct prompts rises through training (§3.2, Figure 3b of [[dapo]]) |
| Trusting a verifier that was never tested | Training reward far above held-out scored accuracy | Score a held-out slice with the strict verifier and require agreement with the training reward within a stated tolerance |
| Changing the decoding configuration between the before and after evaluations | Retention delta larger than the training effect | Pin temperature, top-p, max tokens, system prompt and few-shot format; CoT alone moves MMLU-Pro by up to 19.1 points (§6.2 of [[mmlu-pro]]) |
| Generalizing a one-family result | A reward design reported as "works" from Qwen-only runs | Repeat one cell on a second family ([[spurious-rewards-rlvr]] §3) |

---

## Check your understanding

1. A DPO cell reports `rewards/margins` rising monotonically and `rewards/accuracies` at 0.82, while the
   model's refusal rate on a held-out safety set has halved. Explain, using the definition of the implicit
   reward, how both observations can be true of the same checkpoint.
2. Two RLVR cells reach the same pass@1 on the target set. One has forward KL to the reference twice as large
   as the other. Predict which retains more on the held-out suite, name the evidence, and state the condition
   under which that prediction fails.
3. NSR keeps pass@256 at the base-model level while PSR lowers it. Derive this from Eq. 7 and Eq. 8 of
   [[negative-sample-reinforcement]] rather than restating the table.
4. The lab asks for an SFT control trained on the same data. What claim becomes unsupportable if the control
   is omitted, given the RL-versus-SFT Pareto result in [[rls-razor]]?
5. Why does the chapter require the ablation cells to be matched on optimizer steps rather than on epochs,
   and what would a mismatch do to the measured contribution of negatives?
6. A reader proposes to skip the before-evaluation and compare the after-evaluation against the published
   score for the base model. Using the prompt-robustness numbers in [[mmlu-pro]] and the strict/loose gap in
   [[ifeval]], say what sizes of effect this makes unmeasurable.
7. The entropy law `R = −a·exp(H) + b` fits the same curve for GRPO, RLOO and PRIME, which use different
   advantage estimators ([[entropy-mechanism-llm-rl]], §2.5, Figure 6).
   What does that imply about a memo that explains a plateau by naming the algorithm?
8. Both [[negative-sample-reinforcement]] and [[likelihood-displacement]] study a negative gradient and reach
   opposite conclusions about its effect on the model's other behaviour. State the two setting differences
   that make both results consistent.

---

## Connections

- **ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL** (dependency):
  supplies the stage-placement decision this lab executes for one stage.
- **ch-39 — Offline Preference Optimization: DPO and Its Variants**: Option A's objective and its variants.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO**: Option B's objective, including the
  normalization choices used here.
- **ch-43 — Entropy, Output Diversity, and KL Control in RL**: the entropy and KL signals logged in §4.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative
  Advantages**: the derivations this lab's negative-signal section applies.
- **ch-44 — Process Supervision and Verifiable Rewards**: verifier construction for Option B.
- **ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split**: the
  earlier lab whose held-out split and forgetting report this lab reuses.
- **ch-00 — What General Capability Means and How It Is Measured**: the capability coverage map that the
  retention suite in §5.1 instantiates.
- **ch-46a — Lab: Small Agentic SFT-then-RL Run with a Generality Gate** (next): the same three-axis
  deliverable for a multi-turn agentic stage.
- **ch-47 — Evaluation Harness and Suite Design for General Capability**: turns the ad-hoc suite of §5.1 into
  a versioned harness.
- **ch-54 — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments**: where the
  sampler-versus-trainer KL signal of §4 is analysed.

---

## Sources

- [[dpo]] — Eq. 7, the implicit-reward identity, the sign-split gradient, and the β values used in the runs (App. B, §6.1).
- [[grpo]] — Eq. 3 and Eq. 4, the group-normalized advantage, and the DeepSeekMath-RL 7B settings (§4.2).
- [[dr-grpo]] — the length and difficulty biases of Eq. 3, the unbiased normalization, and the App. G Table 6 configuration.
- [[dapo]] — dynamic sampling and the zero-advantage argument, clip-higher, token-level loss, and the AIME 2024 ablation.
- [[negative-sample-reinforcement]] — the PSR / NSR / full / W-REINFORCE comparison (Table 1), the logit gradients (Eq. 7–8), the λ sweep, and the unbiased pass@k estimator.
- [[likelihood-displacement]] — the refusal-rate collapse under DPO, the preferred-log-probability changes, and CHES filtering.
- [[rls-razor]] — the RL-versus-SFT retention frontiers and forward KL as the predictor of forgetting.
- [[rlvr-beyond-base-model]] — pass@1 against pass@256 across training steps and six RL algorithms; the rollout-count ablation.
- [[spurious-rewards-rlvr]] — model-family dependence of RLVR gains, used to bound what one cell proves.
- [[entropy-mechanism-llm-rl]] — the entropy–performance law, early entropy consumption, and the Clip-Cov / KL-Cov repair.
- [[mmlu-pro]] — the knowledge-and-reasoning component and its prompt-robustness and CoT numbers.
- [[ifeval]] — the instruction-following component, its four metrics, and the strict/loose trade-off.
- [[xstest]] — the over-refusal component and the effect of the system prompt on both sides of the trade-off.
- [[trl-grpo]] — loss-type normalizers, advantage scaling, `frac_reward_zero_std`, sign-split clip metrics, the k3 KL term, `mask_truncated_completions`.
- [[openrlhf-dpo]] — `DPOLoss` and the exact metric set a DPO trainer logs, which is what §2.2 works around.
- [[openrlhf-ppo]] — the `(loss, clip_ratio, ppo_kl, vllm_kl)` return and the sampler-versus-trainer KL.
- [[verl-grpo]] — the group-filtering option corresponding to DAPO dynamic sampling.
