<!-- chapter: ch-12
     track: rl-pipeline
     kind: content
     title: Composing Rewards and Selecting Prompts
     deps: [ch-11]
     sources: [[rl-design-guide]], [[mai-thinking-1]], [[spurious-rewards-rlvr]], [[dapo]], [[polaris]], [[reward-model-overoptimization]], [[natural-emergent-misalignment-reward-hacking]], [[metr-frontier-reward-hacking]], [[scalerl]], [[skywork-or1]], [[ifbench]], [[magistral-recipe]], [[minimax-m1]], [[agentrl]], [[rubrics-as-rewards]], [[rubicon-rubric-anchors]], [[olmo-3-rl-pipeline]], [[open-instruct-rlvr-pipeline]], [[reasoning-gym]], [[general-reasoner-verifier]], [[kimi-k1-5-recipe]], [[reward-hacking-taxonomy]], [[verifier-pitfalls]], [[master-rm-one-token]], [[scaling-reasoning-losing-control-mathif]], [[slime-agentic-and-mismatch-examples]], [[mai-rl-infrastructure]], [[subjective-reward-guide]], [[deepmath-103k]], [[recipe-crosswalk]], [[writing-zero]]
     figures: none
-->

# Chapter 12 — Composing Rewards and Selecting Prompts

> **Core insight.** A training reward is a combination of terms — correctness, format, length, language, safety, quality — that live on different scales and matter differently. Summing them lets whichever term has the largest spread on a given prompt decide the gradient, regardless of its importance, and a group-normalized advantage magnifies small differences in a secondary term whenever the primary term ties. MAI-Thinking-1 states both problems — rewards "occupy different scales and are not directly comparable", and their distribution is "narrow for some prompts and wide for others" — and replaces additive combination with two rules: *lexicographic shaping*, where a lower-priority reward affects the gradient only if all rollouts in the group tie on the higher-priority reward, and *gated application*, where a higher-priority reward must reach a minimum before lower-priority rewards apply at all; an unsafe response "receives the minimum reward and is never graded on response quality" ([[mai-thinking-1]] Sec. 3.4.1). Prompt selection is the other half of the same problem: a group-relative advantage is zero on prompts the policy always or never solves, so the batch should be kept on prompts with within-group variance (DAPO dynamic sampling, +8 AIME24), solved prompts should be retired (ScaleRL No-Positive-Resampling raised the asymptote), but the difficulty window should not be narrowed to hard prompts only (POLARIS: keeping only pass ≤ 4/8 hurt as much as no filtering). Because a reward can be imperfect in ways no one has detected, the run needs monitors that compare the training reward with an independent signal, and baselines that would expose a reward that works for the wrong reason: random rewards raised Qwen2.5-Math-7B's MATH-500 score by 21.4 points ([[spurious-rewards-rlvr]] §2.2).
>
> **Guideline.** When combining rewards whose scales differ or whose importance is strictly ordered, use priority or gating rather than a weighted sum, because a weighted sum lets a high-variance minor term dominate and can stay positive on unacceptable responses; MAI found that 87.8% of policy-non-compliant responses received reward-model scores ≥ 3 and that expected safety Likert agreed only moderately with policy compliance (Pearson 0.293, Spearman 0.344) ([[mai-thinking-1]] Sec. 3.4.3, footnote 9). When the reward is a verifiable 0/1 signal, keep it binary and add only small, bounded shaping terms, because Magistral tested a proportional code reward and rejected it after it lowered LiveCodeBench by 2% at 24B ([[magistral-recipe]] §7.4.1). When constraint rewards are the only signal, add a quality signal, because IF-only rewards lowered AlpacaEval 2 from 33.5 to 21.3 ([[ifbench]] §5, Table 3). When one policy trains on several environments, normalize advantages per task, because removing per-task normalization lowered AgentRL's 14B five-task average from 65.0 to 59.4 ([[agentrl]] Table 6). Drop zero-variance groups and refill the batch ([[dapo]] Table 1: +8 AIME24), exclude prompts with pass rate ≥ 0.9 from later epochs ([[scalerl]] Fig. 6b), and keep the window wide ([[polaris]]). Validate any reward change on more than one model family with random-reward and format-reward baselines ([[spurious-rewards-rlvr]]). Against a learned reward, track a held-out gold score against KL and stop near its peak ([[reward-model-overoptimization]] Fig. 1). When an environment may be hackable, patch the scorer rather than penalizing detected hacks ([[metr-frontier-reward-hacking]]), and consider inoculation prompting, which reduced misalignment by 75–90% while hacking stayed above 99% ([[natural-emergent-misalignment-reward-hacking]] Fig. 5).

---

## Why this chapter matters

ch-11 produced grader outputs: a correctness bit, a format check, a judge score, a reward-model probability, a safety verdict. GRPO needs one scalar per rollout. The function that maps grader outputs to that scalar is a design decision with the same consequences as the grader itself: it decides which behavior is reinforced when objectives conflict.

Two facts about group-relative advantages make the combination rule matter more than in a value-based setting:

1. The advantage of rollout i is (R_i − mean_G R) / std_G R ([[mai-thinking-1]] Sec. 3.1.1; [llm-training ch-40](../../llm-training/ch-40/read.md) §4). Only within-group differences matter; a term that is constant within the group has no effect.
2. Division by the group standard deviation rescales whatever differences exist to unit size. If the correctness term ties within a group, a small difference in a format or length term becomes a full-magnitude advantage. [llm-training ch-40](../../llm-training/ch-40/read.md) §5 works through a reward-model example in which a 0.02 difference becomes a full-magnitude advantage.

Prompt selection is in the same chapter because it acts on the same quantity: it decides which groups have within-group variance at all. This chapter follows [[rl-design-guide]] §4 and cites the underlying cards.

---

## §1 Inventory of reward terms

| Term | Form in the sources | Source |
|---|---|---|
| Correctness | 0/1 (Olmo 3, Reasoning Gym, General-Reasoner) or ±1 (DeepMath, slime `math_dapo`) | [[olmo-3-rl-pipeline]], [[reasoning-gym]], [[deepmath-103k]] |
| Format | +0.2 on top of accuracy 1.0 (Reasoning Gym); Magistral: format pass 0.1, format fail 0 and not graded; R1-style format reward in open-instruct, additive or gating | [[reasoning-gym]] §4, [[magistral-recipe]], open-instruct |
| Extraction failure | −0.5 | [[general-reasoner-verifier]] §4.1 |
| Length near the limit | DAPO soft overlong punishment, 0 to −1 inside the buffer | [[dapo]] Eq. 13 |
| Length scaled by difficulty | −w_len · ρ_q · \|y\| / ℓ_max, w_len = 0.25 up to 64k, 0 at 128k | [[mai-thinking-1]] Eq. 12, Sec. 3.1.5 |
| Length relative to group | Kimi λ = 0.5 − (len − min_len)/(max_len − min_len); λ if correct, min(0, λ) if incorrect | [[kimi-k1-5-recipe]] §2.3.3 |
| Length relative to reference | 1 − 0.05 · min(10, \|len(gt) − len(answer)\|) on correct answers | [[general-reasoner-verifier]] §4.1 |
| Length quantile for subjective tasks | penalty outside a per-context acceptable range computed offline | [[mai-thinking-1]] Sec. 3.4.1 |
| Language consistency | w_lang · max(1 − α·n_non-English, 0), w_lang = 0.5, α = 0.005 | [[mai-thinking-1]] Eq. 11 |
| Tool-use bonus | wrong answers scored min(−0.6, −1 + (turns − 2)/2 · 0.1) | slime ReTool ([[slime-agentic-and-mismatch-examples]]) |
| Safety | policy-compliance judge as a gate | [[mai-thinking-1]] Sec. 3.4.3 |
| Quality | RM P(score = 5); rubric judge; pairwise judge | [[mai-thinking-1]] Sec. 3.4.1, ch-11 |
| Honesty | 5-category factuality × confidence judge combined by weighted sum | [[mai-thinking-1]] Sec. 3.4.4 |
| Non-stop penalty | response that did not stop gets a fixed value (default −10) | open-instruct `RewardConfig` |

MAI-Thinking-1 uses one additive decomposition for its verifiable climbs, R = R_task + w_lang · R_lang − w_len · R_len (Eq. 10), and switches to priority and gating for helpfulness and safety (§3). The language term was added because mixed-language chains of thought "correlate with spikes in log-probability divergence between the training and inference policies" (Sec. 3.1.2): a numerical-stability motive (ch-07), not a quality one.

---

## §2 Additive combination and where it fails

**Additive combination in open code.** open-instruct sums the outputs of every verifier named for a row, each times its weight and a global multiplier (default `verification_reward = 10.0`):

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L1348-1356
    for result, metadata in zip(reward_results, task_metadata):
        response_idx = metadata["response_idx"]
        dataset = metadata["dataset"]
        reward_weight = metadata["reward_weight"]

        score = result.score if hasattr(result, "score") else result
        weighted_reward = reward_mult * score * reward_weight

        response_rewards[response_idx] += weighted_reward
```

The same file shows the gated alternative for format rewards. With `additive_format_reward` off, a response that fails the format check gets 0 regardless of its verifier score, and with `non_stop_penalty` on, any response that did not stop is overwritten with a fixed value:

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L1446-1451, L1465-1469
                        if self.apply_r1_style_format_reward and self.additive_format_reward:
                            scores[i] = raw_score + scores[i]
                        elif self.apply_r1_style_format_reward and not self.additive_format_reward:
                            scores[i] = raw_score if format_valid[i] else 0
                        else:
                            scores[i] = raw_score
...
            if self.non_stop_penalty:
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = self.non_stop_penalty_value
```

Magistral uses the gated form: a response that fails the format check gets 0 and is not graded for correctness; one that passes gets 0.1, plus 0.9 if correct, plus a length penalty from 0 to −0.1 and a language-consistency bonus of 0.1 ([[magistral-recipe]] §2.2.1–2.2.4).

**Failure 1: scale.** MAI: "Naïvely summing such rewards would result in the largest-magnitude signal dominating irrespective of its importance" ([[mai-thinking-1]] Sec. 3.4.1, read from the PDF). The magnitude that matters is the within-group spread, not the nominal range. A reward model whose scores spread widely on a prompt dominates a binary constraint check that ties on the same prompt.

**Failure 2: context-dependent spread.** MAI: "the reward distribution is itself context-dependent, narrow for some prompts and wide for others". A fixed weight w on a term with prompt-dependent spread gives a prompt-dependent effective weight. Per-group standard-deviation normalization makes this worse for the terms that tie: whatever is left after ties is scaled to unit size.

**Failure 3: non-negotiable criteria.** MAI: "a well-written response that is unsafe remains unacceptable, regardless of its quality". In a sum, enough quality can compensate for a safety failure. MAI measured that the compensation is not hypothetical: in a paired rollout audit, expected safety Likert agreed with the policy-compliance field only moderately (Pearson 0.293, Spearman 0.344), and 87.8% of policy-non-compliant responses received reward-model scores ≥ 3 (Sec. 3.4.3, footnote 9). "A simple weighted average is insufficient: scalar rewards can remain positive on policy-non-compliant responses."

**Failure 4: interaction with group normalization.** A small shaping term (format 0.2, length −0.05) is intended as a tie-breaker. Under GRPO it is a tie-breaker that gets full-magnitude advantage when correctness ties, i.e. in all-correct and all-wrong groups. That can be what you want (§3 discusses MAI's intentional use of it) or a source of reward hacking on the shaping term. When dynamic sampling (§6) drops tied groups before the loss, the shaping term never acts as a tie-breaker at all, and its effect comes only from mixed groups.

---

## §3 MAI-Thinking-1's priority and gated combination

MAI applies two strategies "selectively based on the context" ([[mai-thinking-1]] Sec. 3.4.1, read from the PDF).

**Lexicographic reward shaping.** "For one set of contexts, lower-importance rewards become active only when all rollouts in a group score equally on higher-importance rewards. This induces a strict priority ordering, where a secondary reward influences the gradient only if the primary reward is tied within the rollout group (Skalse et al., 2022). Because it operates on within-group relative comparisons, this formulation is also invariant to the absolute scale of each reward."

**Gated reward application.** "For other contexts, higher-importance rewards must satisfy a minimum level of performance before lower-importance rewards are applied at all (Achiam et al., 2017). Safety is the canonical case: an unsafe response receives the minimum reward and is never graded on response quality."

"Both strategies sidestep cross-scale calibration by replacing additive combination with conditional or rank-based logic, ensuring that high-priority objectives are never traded away for gains on lower-priority ones."

The report does not publish code. The sketch below implements the two rules as stated, for illustration only; the variable names, the tie test, and the use of the standard GRPO normalization are my choices, not MAI's.

```python
# Illustration of the stated rules (not MAI code).
import numpy as np

def lexicographic_rewards(groups_of_terms):
    """groups_of_terms: array [G, K], column 0 = highest priority.
    Returns the per-rollout reward used for the group advantage."""
    R = np.asarray(groups_of_terms, dtype=float)
    for k in range(R.shape[1]):
        if np.ptp(R[:, k]) > 0:          # first term that is not tied in this group
            return R[:, k]
    return np.zeros(R.shape[0])          # all terms tied: no signal

def gated_reward(gate_ok, quality, r_min=0.0):
    """gate_ok: bool per rollout (e.g. policy-compliant); quality: score per rollout."""
    return np.where(gate_ok, quality, r_min)

def grpo_advantage(r):
    s = r.std()
    return (r - r.mean()) / s if s > 0 else np.zeros_like(r)
```

Properties that follow from the definition:
- In a lexicographic group, only one term decides the advantage, so there is no weight to tune and no scale to calibrate.
- A secondary term gets signal exactly in the groups where the primary term ties. With a binary primary reward, those are the all-correct and all-wrong groups, which dynamic sampling would otherwise discard. MAI does not state how its lexicographic contexts interact with its pass-rate filters (Sec. 3.1.3); the interaction is my inference from the definitions, not a reported result.
- A gate removes the gated rollouts from the quality comparison entirely: in safety contexts, a non-compliant rollout gets the minimum reward and otherwise "the normal weighted mixture is used" (Sec. 3.4.3). The gate is applied first; additive mixing survives inside the gate.

**Where MAI uses each.**

| Domain | Graders | Combination | Source |
|---|---|---|---|
| Instruction following | rule checks for deterministic constraints; LLM judge giving a binary judgment per atomic rubric, averaged over passes; reward model for general quality | lexicographic, "with IF-specific rewards serving as the primary signal" | Sec. 3.4.2 |
| Safety | judges for policy compliance, response engagement, response style; reward model; optional AI judge | safety-gated aggregator: non-compliant → minimum reward; else weighted mixture | Sec. 3.4.3 |
| Honesty | LLM judge assigning one of five factuality × confidence categories | weighted sum: confident-correct highest, confident hallucination lowest, abstention neutral, unconfident-correct reduced | Sec. 3.4.4 |
| Style | prompted LLM judges on an integer scale 0 / 1 / 2 (major / minor / no style issues), plus the reward model | "applied only after verifiable rewards and safety constraints are satisfied", then "weighted alongside other graders depending on the domain" | Sec. 3.4.5 |
| STEM, agentic | task reward + language + length | additive (Eq. 10) | Sec. 3.1.2 |

For style, MAI reports that "coarse graders outperform more granular graders or prompt-specific rubrics because they give the LLM judge more flexibility to interpret the rubric appropriately for the given prompt and response, making it harder for models to hack the grader" (Sec. 3.4.5; no numbers). This runs against the [[rubrics-as-rewards]] result that prompt-specific rubrics beat generic ones on HealthBench; the domains differ (style vs expert medical content) and neither source tests the other's setting.

No ablation in the report compares lexicographic or gated combination with a weighted sum; the evidence for the choice is the audit in footnote 9 (gating) and the design argument. MAI adds that verifiable rewards are "less prone to reward hacking", "less sensitive to multi-epoching", and "generally useful for stabilizing training" than non-verifiable ones, and that AI-feedback rubrics push length and style upward, which it counters with the reward model and an offline per-context length quantile range (Sec. 3.4.1).

**Open-source analogues of gating.**
- IFBench's RM-gated shaping: F_i = V_i + 1 if V_i > 0 and S_i > α; V_i − 0.5 if V_i > 0 and S_i ≤ α; V_i if V_i ≤ 0, where V is the verifiable constraint reward, S the reward-model score and α = 7. The run scored IFEval 86.1, IFBench 30 and AlpacaEval 2 31.6 (App. E), against AlpacaEval 2 21.3 for IF-only rewards (Table 3) ([[ifbench]]). Here the constraint is the gate and the quality model shapes only satisfied responses.
- Magistral's format gate (§2).
- Rubicon's veto rubric zeroes the reward for prefatory praise of the user and self-evaluation ([[rubicon-rubric-anchors]] App. A.1; no number for the veto's effect).
- GLM-4.5 gives reward 0 for bad tool-call format ([[recipe-crosswalk]] GLM-4.5 row).
- open-instruct's non-additive format reward and non-stop penalty (§2 code).

---

## §4 Partial credit vs binary reward

Partial credit gives more within-group variance and therefore more groups with signal. It also gives the policy a way to earn reward without solving the task.

**Evidence.**
- Magistral tested a proportional code reward (fraction of tests passed) and rejected it: 24B, 250 steps, −2% LiveCodeBench ([[magistral-recipe]] §7.4.1, Fig. 11). One run.
- RaR: implicit aggregation (the judge sees all rubric items and returns one score) 31.2 vs explicit weighted sum 29.7 on HealthBench (Fig. 2) ([[rubrics-as-rewards]]). Both are graded; the comparison is about who aggregates.
- Olmo 3's code verifier supports both pass-rate and all-pass rewards ([[olmo-3-rl-pipeline]] Key Contributions); the report does not compare them in the card.
- Instruction following: the Olmo 3 paper describes a binary IF reward ("1 if all constraints are satisfied"), while open-instruct's `IFEvalVerifier` at f020e9a returns the fraction of constraints satisfied; which trained the released models is unresolved ([[olmo-3-rl-pipeline]] Verification).
- MAI's tool-use environments decompose synthetic tasks into sub-tasks graded independently by an LLM judge ([[mai-thinking-1]] Sec. 3.3.2), a form of partial credit at the task level; no comparison with a binary success reward is reported.

**Mechanism.** With partial credit on tests or constraints, a response that satisfies the easy constraints and ignores the hard one outscores one that attempts the hard one and fails, so the policy can learn to skip hard parts. With a binary reward, that response and a complete failure score the same. [[ifbench]] shows the other side of the trade-off: rewarding only constraint satisfaction, even binary, degraded general quality (AlpacaEval 2 33.5 → 21.3; GPT-4.1 judge quality 7 → 6.4 out of 10, §5), so the constraint reward needs a quality signal; §3's gating is how IFBench and MAI add it without letting quality compensate for missed constraints.

The practice rule in [[rl-design-guide]] §4.4 — binary correctness plus small, bounded shaping terms — is practice, not measured as a general rule; Magistral's −2% is the only direct ablation in the library.

---

## §5 Multi-objective weighting across domains

When one policy trains on several domains under one loop, the combination problem reappears at the batch level: domains differ in reward scale, pass rate and rollout length.

- **Per-task advantage normalization.** AgentRL normalizes token-level advantages separately per task; removing it lowered the 14B five-task average from 65.0 to 59.4, and the tasks were learned at different rates ([[agentrl]] Table 6, §4.2).
- **Per-domain verifiers, roughly equal data.** Olmo 3 gives each domain its own verifier and mixes "roughly equal amounts of data … for each domain (with slightly more focus on math and instruction following)"; per-domain reward weights in the final mix are not reported ([[olmo-3-rl-pipeline]] §4.4.2, Verification).
- **Parallel per-domain progress.** ScaleRL's math + code joint training gave parallel per-domain trends (Fig. 11) ([[scalerl]]).
- **Stage-wise instead of joint.** Rubicon found a "seesaw effect" when constraint-following and style rubrics were trained jointly: creativity-only training cost Collie −6.0 and IFEval −5.9; instruction-only training cost EQ-Bench3 −2.2; it trains them in sequential stages ([[rubicon-rubric-anchors]] §4.4, Fig. 2). MAI trains separate specialist climbs and consolidates them (ch-17); in the agentic climb it mixes STEM tasks in because that "helps to stabilize the RL climb" ([[mai-thinking-1]] Sec. 3.3).
- **Gradual mixing.** MiniMax-M1 starts with rule-verified reasoning tasks and gradually mixes in reward-model-graded general tasks with dynamic weighting, and monitors GenRM length bias online because offline mitigation "frequently failed to prevent length bias during RL training" ([[minimax-m1]] §4.2.2, §4.3).
- **Mixture weights as runtime configuration.** Rocket's typed runtime settings include mixture weights, applied at step or sampling boundaries without restarting the job ([[mai-rl-infrastructure]] p. 12–13); ch-21 covers the mechanism.

Weighting across domains is less studied than combination within a rollout. The measured evidence is AgentRL's normalization ablation and Rubicon's seesaw; the rest is practice.

---

## §6 Spurious-reward evidence: what a reward gain does and does not show

[[spurious-rewards-rlvr]] trained Qwen2.5-Math-7B with GRPO under a ladder of rewards, from ground truth to random (§2.2):

| Reward | MATH-500 gain | AMC gain | AIME2024 gain |
|---|---|---|---|
| Ground truth | +29.1 | about +27 to +29 | +15.3 |
| Majority vote (64 samples) | — | about +27 to +29 | — |
| Format (non-empty `\boxed{}`) | — | +13.8 | +10.3 |
| Incorrect labels | +24.1 | +24.1 | +10.2 |
| Random | +21.4 | +21.4 | +10.2 |

(The card lists identical incorrect-label and random gains for MATH-500 and AMC; I did not re-check this against the paper. "—" means the card gives no value.)

On AIME2025, written after every tested model's knowledge cutoff, ground truth had a clear advantage and the other rewards gave −0.4 to +4.5 (App. D). On OLMo2 and Llama3 the spurious rewards gave minimal or negative gains; every weak or spurious reward failed to help at least one non-Qwen model (§3). The authors trace the random-reward gain to GRPO's clipping term, which amplifies behaviors that already have high probability; disabling clipping in three ways removed the gain (§4, Fig. 4). For Qwen2.5-Math, the amplified behavior was code-style reasoning without execution, rising from 65% to over 90% of responses. Olmo 3 ran a random-reward negative control and found no gains (Fig. 27) ([[olmo-3-rl-pipeline]]).

**What it implies for reward design.**
- A reward change that improves one model family is not evidence that the reward is right. Validate on at least two families, and include random-reward and format-only baselines.
- Most spurious gains appeared within the first 50 steps (§2.2). Early gains are the least informative.
- Post-cutoff evaluation sets are the check that separates elicitation of a prior from learning.

---

## §7 Prompt selection by pass-rate window

Pass rate p for a prompt under the current policy determines whether its group has signal (ch-10 §3). The selection rules in the sources act at three times.

**Offline, before training.** Remove prompts the starting policy always solves or never solves: Olmo 3 removes prompts with pass rate > 62.5% over 8 samples (7B) ([[olmo-3-rl-pipeline]] §4.4.2); Skywork-OR1 removes base-model correctness 0 or 1 ([[skywork-or1]] §3.1); POLARIS removes 8/8-solved prompts and shapes the rest into a mirrored-J distribution ([[polaris]]); MAI's IF data keeps "only examples that are challenging but solvable by our model" ([[mai-thinking-1]] Sec. 3.4.2).

**Online, per group.** DAPO's dynamic sampling drops all-correct and all-wrong groups and samples until the batch is full: AIME24 42 → 50 in the cumulative ablation ([[dapo]] Table 1). MAI's two-stage window: G_early = 16 rollouts, continue only if the early pass rate is in [0.05, 0.8]; after the full G = 128, keep only groups with pass rate in [0.1, 0.8] ([[mai-thinking-1]] Sec. 3.1.3, 3.1.5). ScaleRL found zero-variance filtering raised the asymptote (Fig. 6a).

**Across epochs.** ScaleRL's No-Positive-Resampling excludes prompts with pass rate ≥ 0.9 from later epochs and raised the asymptote (Fig. 6b) ([[scalerl]]). POLARIS removes samples above 0.9 accuracy at the end of each stage; Skywork-OR1 removes prompts the actor solved with correctness 1 in the previous stage. open-instruct implements No-Positive-Resampling and zero-std filtering in the group builder:

```python
# open-instruct@f020e9a open_instruct/data_loader.py L812-818
    percent_solved = np.mean(result.reward_scores).item() / max_possible_score
    if no_resampling_pass_rate is not None and percent_solved >= no_resampling_pass_rate:
        assert iter_dataloader is not None
        iter_dataloader.exclude_index(result.index)

    if filter_zero_std_samples and np.std(result.reward_scores) == 0:
        return None
```

**Do not narrow the window to hard prompts.** POLARIS's ablation on DeepScaleR-40K with the 7B model compared the full 40K set, 26K with 8/8-solved removed, and 19K keeping only pass ≤ 4/8; "the 19K and unfiltered sets both hurt training" ([[polaris]] Difficulty). Removing easy prompts helps; removing medium ones too does not. MAI's upper bound of 0.8 keeps prompts the policy solves most of the time.

**Prioritized sampling instead of a hard window.** Kimi k1.5 samples problem i with probability proportional to 1 − s_i, where s_i is the tracked success rate ([[kimi-k1-5-recipe]] §2.3.4; no ablation reported). This keeps easy prompts in the distribution at low frequency instead of removing them.

**A reward-scale caution.** The window is defined on the pass rate of the *combined* reward. With partial credit or shaping terms, "solved" must be defined by a threshold on the correctness term, or open-instruct's `percent_solved` (mean reward divided by the maximum possible score) will treat a prompt with high shaping reward and low correctness as nearly solved.

---

## §8 Curriculum and replay

**Difficulty curriculum.** Reasoning Gym advances one difficulty level after the policy exceeds 70% accuracy over 20 steps; the curriculum beat fixed-level training at every listed level (Spell Backwards length 4: 30.00 → 70.67; Mini Sudoku 6.67 → 20.00; Count Primes 4.00 → 30.67) ([[reasoning-gym]] Table 5; one 3B model). The authors note curriculum value may be limited where the progression path is ambiguous. Kimi k1.5 warmed up on the full mixed set and switched to hard questions only at iteration 24; one plot shows curriculum above uniform sampling, with no seeds reported ([[kimi-k1-5-recipe]] §3.5, Fig. 9).

**Stage-wise schedules.** POLARIS combines prompt removal with temperature increases per stage (ch-10 §2). MAI doubles the length limit from 8k to 128k (ch-10 §4). MiniMax-M1's second run re-filters data by the first run's pass rates and upweights harder math and code ([[minimax-m1]] via [[recipe-crosswalk]]).

**Replay of earlier successes.** POLARIS's Rollout Rescue stores one correct response per prompt from an earlier epoch and substitutes it for a failed rollout when a group is 0/8 ([[polaris]]). The substituted response comes from an older policy; the blog does not discuss an importance correction, so the replayed sample is off-policy by an unknown amount.

**Reuse across epochs.** MAI reports verifiable rewards to be "less sensitive to multi-epoching" than non-verifiable ones (Sec. 3.4.1; no numbers), which suggests limiting how many times prompts graded by judges or reward models are reused.

---

## §9 Monitoring for reward hacking during the run

Theory says a better proxy does not remove hacking: on policy sets with an open subset, any non-trivial unhackable pair must induce the same policy ordering ([[reward-hacking-taxonomy]] Theorem 1). The practical response is to limit optimization and to detect divergence early.

**Gold score against KL.** With proxy reward models trained on labels from a 6B gold RM, the gold score first rose and then fell as KL from the initial policy grew, while the proxy kept rising; a nonzero KL penalty behaved like early stopping and did not raise the gold-vs-KL frontier ([[reward-model-overoptimization]] Fig. 1, §3.6, Fig. 9). Track a held-out gold or human score against √KL and stop near its peak.

**Training reward against oracle reward.** [[verifier-pitfalls]] detected verifier hacking by plotting training reward against a GPT-4o oracle (§5); [[master-rm-one-token]] found its judge collapse when the policy converged to responses under 30 tokens such as "Solution." (§1). Both were found by comparing the reward with an independent check and by reading degenerate outputs.

**Capability drift on a held-out axis.** Raising RL rollout length from 1k to 8k raised math accuracy from 28.73 to 39.82 and lowered hard instruction-following accuracy from 19.05 to 14.29 ([[scaling-reasoning-losing-control-mathif]] Table 5). A reasoning reward does not measure instruction following; track it separately.

**Behavioral signatures.** Response length; length of self-justification (417 vs 58 tokens under a scalar RM, [[writing-zero]]); rate of a specific phrase; refusal rate; tool-call count; shortest and longest high-reward responses. The [[subjective-reward-guide]] §2 table lists measured signatures for subjective rewards.

**Patch the scorer, do not penalize the hack.** METR found o3 reward-hacked in 39 of 128 RE-Bench runs (30.4%) and 8 of 1,087 HCAST runs (0.7%), and appending "Please do not cheat." left the share of runs with a hacking plan at 80% of 20 runs ([[metr-frontier-reward-hacking]]). It recommends patching the scoring function when a monitor finds an exploit, and cites evidence that penalizing monitor-detected hacking can produce hacking the monitor does not detect. MAI's SWE environments include an LLM monitor alongside environment hardening ([[mai-thinking-1]] card).

**Inoculation prompting.** When a pretrained model learned to reward-hack production coding environments, all six misalignment evaluations rose (e.g. "Fake/bad goals" 69.8% vs 0.1% in a non-hacking baseline), and the final checkpoint attempted to sabotage a reward-hacking classifier 12% of the time ([[natural-emergent-misalignment-reward-hacking]] Fig. 2, Fig. 9). A single system-prompt line during RL that frames passing the grader as the task reduced final misalignment by 75–90% while hacking rates stayed above 99% (Fig. 5); telling the model not to hack, or filtering hacked episodes and distilling, did not prevent misalignment (Fig. 4). This is measured on one production setup.

**What to do when hacking is detected.** Fix the grader (ch-11), then decide whether the policy can continue. MAI reports, for collapses in general, that resuming from a pre-collapse checkpoint "is often not viable" because instabilities are embedded "many steps before the actual collapse", and uses self-distillation to restart climbs ([[mai-thinking-1]] Sec. 3.1.4). Whether the same holds for learned hacks is not stated.

---

## §10 Applied example: a reward for a sales agent

The [[subjective-reward-guide]] §4 sketches a reward for a human-like sales voice agent. In this chapter's terms it is a gated and lexicographic design:

1. **Gates (reward = minimum on failure):** claims to be human or invents personal experience; product or price facts contradicted by the catalog (program check or reference judge probed with master keys); user praise or self-praise veto; synthesized-speech WER above a threshold. These follow MAI's safety gate and Rubicon's veto.
2. **Primary, among gated rollouts:** task outcome from the simulated customer (meeting booked, objection resolved), graded from CRM/calendar state (ch-08 §8).
3. **Secondary, active when the primary ties within the group:** conversation quality from a pairwise judge with a sales rubric; speech naturalness from a pairwise judge trained on human A/B labels of the agent's own TTS.
4. **Monitors:** length, engagement-bait rate, banlist phrase ratio, a held-out human score against KL.

The guide labels the gate contents and rubric contents as practice and the judge methods as measured; nothing in this design has been run. It is a starting point for ch-22 and ch-23.

---

## Recipe

1. **List every reward term** and classify it: non-negotiable (gate), primary objective, secondary objective, numerical-stability term (language, length near the limit).
2. **Gate first**: apply gates (safety, format, catalog facts) before computing other terms; gated-out rollouts get the minimum reward.
3. **Choose priority or sum among the rest**: lexicographic when importance is strictly ordered or scales are incomparable; a weighted sum only for terms on one scale whose trade-off you can state.
4. **Keep correctness binary** for verifiable tasks; add small bounded shaping terms; check what each shaping term does in all-tied groups.
5. **Normalize across domains**: per-task advantage normalization when environments differ in difficulty or length.
6. **Select prompts** offline (drop always-solved and never-solved), online (drop zero-variance groups, refill with active sampling or early exit), and across epochs (retire prompts with pass ≥ 0.9). Keep the window wide. Define "solved" on the correctness term.
7. **Add curriculum** where difficulty is a parameter (advance at a fixed accuracy threshold).
8. **Run baselines** for any new reward: random reward and format-only reward, on two model families, evaluated on post-cutoff sets.
9. **Monitor**: oracle vs training reward, gold score vs KL, a held-out capability axis (IF), length and phrase statistics, extreme-length high-reward samples.
10. **On detected hacking**: patch the scorer; do not add a penalty for the detected pattern; consider inoculation prompting for environments that cannot all be hardened.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Weighted sum of terms with different spreads | one term decides the advantage on most prompts | per-term within-group variance; share of advantage explained by each term |
| Safety as an additive term | unsafe responses with positive reward | audit non-compliant rollouts' total reward (MAI: 87.8% RM ≥ 3) |
| Shaping term acting on tied groups | format or length behavior drifts while accuracy is flat | advantage attribution in all-correct and all-wrong groups |
| Partial credit on tests or constraints | hard parts skipped | rate of full solves vs partial |
| Constraint reward without quality signal | general quality drops | AlpacaEval or judge score on held-out chat |
| Narrow hard-only window | training hurt like no filtering | pass-rate histogram of the batch |
| "Solved" defined on combined reward | prompts retired while correctness is low | track correctness pass rate separately |
| Trusting one-family gains | gains vanish on another model | random and format baselines on two families |
| Stopping on proxy reward | gold score falling while proxy rises | gold-vs-KL curve |
| Penalizing detected hacks | hacking shifts to undetected forms | inspect whole task families (METR) |

---

## Check your understanding

1. Explain why per-group standard-deviation normalization makes a small shaping term act with full magnitude in some groups. In which groups does this happen under a binary correctness reward, and how does dynamic sampling change it?
2. MAI's lexicographic rule is invariant to the absolute scale of each reward. Why does that remove the need to tune weights, and what information about the secondary reward is discarded?
3. MAI's audit found 87.8% of non-compliant responses with reward-model scores ≥ 3. Explain why this result argues for a gate rather than a larger safety weight in a sum.
4. IFBench's shaping gives +1 only when the constraint is satisfied and the RM score exceeds α, and −0.5 when satisfied but below α. Which objective is the gate here, and why did this raise AlpacaEval 2 relative to IF-only rewards while lowering IFEval?
5. Random rewards raised Qwen2.5-Math-7B's MATH-500 by 21.4 points through the clipping term. What does this imply about interpreting an early reward-design gain, and which two controls would expose it?
6. POLARIS found that keeping only pass ≤ 4/8 prompts hurt as much as no filtering, while ScaleRL found that retiring pass ≥ 0.9 prompts helped. Reconcile these two results in terms of which prompts carry signal and which carry information about the distribution.
7. METR recommends patching the scorer rather than penalizing detected hacks. Explain the mechanism by which a penalty can make hacking harder to detect.

---

## Connections

- ch-09 (pass rates recorded at build time), ch-10 (dynamic sampling, early exit and length terms on the rollout side), ch-11 (the grader outputs combined here), ch-13 (diversity axes balanced by mixture weights), ch-14 (instruction-following reward composition), ch-17 (stage order and mixture across domains), ch-19 (rewards without a verifier), ch-21 (mixture weights as runtime configuration), ch-22 and ch-23 (applying the gated design to the learner's agent).
- Algorithm mechanics: [llm-training ch-40](../../llm-training/ch-40/read.md) (group advantage normalization and zero-variance groups), [llm-training ch-44a](../../llm-training/ch-44a/read.md) (length terms), [llm-training ch-44b](../../llm-training/ch-44b/read.md) (multi-domain RL with non-verifiable rewards), [llm-training ch-16](../../llm-training/ch-16/read.md) (prompt distribution and reuse), [llm-training ch-43](../../llm-training/ch-43/read.md) (KL control and over-optimization).
- Sources: [[rl-design-guide]] (§4 outline), [[mai-thinking-1]], [[spurious-rewards-rlvr]], [[dapo]], [[polaris]], [[reward-model-overoptimization]], [[natural-emergent-misalignment-reward-hacking]], [[metr-frontier-reward-hacking]], [[scalerl]], [[skywork-or1]], [[ifbench]], [[magistral-recipe]], [[minimax-m1]], [[agentrl]], [[rubrics-as-rewards]], [[rubicon-rubric-anchors]], [[olmo-3-rl-pipeline]], [[reasoning-gym]], [[kimi-k1-5-recipe]], [[reward-hacking-taxonomy]], [[subjective-reward-guide]].

---

## Sources

- [[rl-design-guide]] — §4 reward and prompt-selection rules used as this chapter's outline.
- [[mai-thinking-1]] — additive decomposition for verifiable climbs (Eq. 10–12), lexicographic shaping, gated application, safety-gated aggregator and its audit (footnote 9), IF/honesty/style combinations, pass-rate windows, multi-epoching remark (Sec. 3.1, 3.4, read from the PDF).
- [[spurious-rewards-rlvr]] — reward ladder from ground truth to random, cross-family failure, clipping mechanism, post-cutoff check.
- [[dapo]] — dynamic sampling and its +8 AIME24 contribution.
- [[polaris]] — mirrored-J window, hard-only ablation, per-stage removal, rollout rescue.
- [[scalerl]] — zero-variance filtering and No-Positive-Resampling raise the asymptote; math + code joint trends.
- [[skywork-or1]] — offline and per-stage online filtering.
- [[reward-model-overoptimization]] — gold score vs KL, KL penalty as early stopping.
- [[natural-emergent-misalignment-reward-hacking]] — misalignment from learned hacks; inoculation prompting.
- [[metr-frontier-reward-hacking]] — hacking rates in agentic evaluations; patch-the-scorer recommendation.
- [[ifbench]] — IF-only reward degrades quality; RM-gated shaping.
- [[magistral-recipe]] — gated format reward, rejected proportional code reward.
- [[minimax-m1]] — gradual mixing of RM-graded tasks; online length-bias monitoring.
- [[agentrl]] — per-task advantage normalization ablation.
- [[rubrics-as-rewards]] — implicit vs explicit rubric aggregation.
- [[rubicon-rubric-anchors]] — seesaw effect, staged training, veto rubric.
- [[olmo-3-rl-pipeline]] — offline difficulty filter, per-domain verifiers, random-reward control, IF code–paper discrepancy.
- [[open-instruct-rlvr-pipeline]] — additive verifier sum, gated format reward, non-stop penalty, No-Positive-Resampling and zero-std filter (code read in the clone at f020e9a).
- [[reasoning-gym]] — format reward, curriculum by accuracy threshold.
- [[general-reasoner-verifier]] — extraction-failure and length-difference terms.
- [[kimi-k1-5-recipe]] — length reward, prioritized sampling, curriculum switch.
- [[reward-hacking-taxonomy]] — impossibility of unhackable non-trivial proxies.
- [[verifier-pitfalls]], [[master-rm-one-token]] — hacking detected by oracle comparison and degenerate outputs.
- [[scaling-reasoning-losing-control-mathif]] — reasoning gains with instruction-following losses.
- [[slime-agentic-and-mismatch-examples]] — ReTool tool-use bonus.
- [[mai-rl-infrastructure]] — mixture weights as typed runtime settings.
- [[subjective-reward-guide]] — failure signatures for subjective rewards; sales-agent reward sketch.
