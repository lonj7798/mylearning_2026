<!-- chapter: ch-53
     track: eval
     kind: lab
     title: Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness
     deps: [ch-52]
     sources: [[tulu-3]], [[gsm-symbolic]], [[gsm1k]], [[ifbench]], [[mmlu-pro]], [[adding-error-bars-evals]], [[harmbench-data]], [[xstest]], [[deduplicating-training-data]], [[signal-and-noise-eval]], [[swe-bench-illusion]]
     figures: figures/eval-pipeline.html, figures/lsh-band-curve.html
     excerpts: excerpts/held-out-suite-tulu3.md, excerpts/perturbation-robustness-sets.md, excerpts/paired-statistics-and-power.md, excerpts/minhash-contamination-gate.md, excerpts/harmbench-behavior-layering.md
     revised: 2026-09 (generality revision)
-->

# Chapter 53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness

> **Core insight.** A harness that reports one averaged score per checkpoint cannot tell an increase in general capability from an increase in benchmark-specific behaviour. Tülu 3 separated a development suite from an unseen suite and never looked at the unseen scores while making decisions; at 8B the pipeline gained +3.9 points on the development average (64.9 → 68.8) but +2.5 on the unseen average (29.9 → 32.4), and on coding the development score moved 86.2 → 83.9 while the unseen score moved 11.5 → 7.4 (arXiv:2411.15124 Table 31). The same checkpoint's 6-task safety average moved 93.1 → 85.5 between SFT and the final RL model while GSM8K moved 76.2 → 87.6 (Table 6). A harness detects this only if it holds out tasks, pairs the comparison per question, varies the prompt template, and gates on contamination before any score is read.
>
> **Guideline.** When you compare two checkpoints on the same questions, use the paired difference and its standard error, because pairing removes the shared question-difficulty variance and, in the worked example of §5, narrows the 95% interval from ±6.1 pp to ±3.8 pp on identical data ([[adding-error-bars-evals]] §4.2, Eq. 7). When the training data of both checkpoints is available, run the contamination gate before scoring and refuse to report scores if it trips, because Tülu 3's 8-gram check found 70.7% of HumanEval instances overlapping Evol CodeAlpaca and 18.2% of MATH instances overlapping NuminaMath-TIR (arXiv:2411.15124 Table 37). When the training data is not available, substitute a memorization probe, because that is the only contamination evidence obtainable from a closed checkpoint ([[swe-bench-illusion]] §4.1). When the target task is mathematics, add a perturbation set, because GSM8K accuracy on 100 template questions sits more than one standard deviation above the centre of the same questions' perturbed distribution for 21 of 25 models ([[gsm-symbolic]] §4.1).

---

## Why this chapter matters for a general-purpose model

This is the last lab of the eval track. It runs after the training labs and before the infrastructure track, and its output is the acceptance test those later chapters use. The SFT lab ([[ch-36]]) and the preference/RL lab ([[ch-46]]) each produced checkpoints and each reported a held-out capability-retention number for its own stage. This chapter builds the harness that measures all of those checkpoints at once, on tasks that were not used while developing either lab, and turns the measurement into a go/no-go decision.

The failure mode the harness exists to detect is narrowing: the checkpoint gets better at the task that was trained and worse at something that was not measured. Narrowing is invisible to a single average for three reasons.

1. The tasks used during development are the tasks the data mixture was tuned against, so their scores are partly a measurement of the tuning loop rather than of capability ([[tulu-3]] §7.4).
2. Scores are reported as point estimates on a few hundred questions, and the noise on that estimate is large enough to hide several points of regression ([[adding-error-bars-evals]] §5; [[signal-and-noise-eval]] §3.1).
3. Accuracy on a fixed benchmark can rise through memorization of that benchmark's items, which no amount of statistics on that benchmark will reveal ([[gsm1k]] §5.4; [[swe-bench-illusion]] §4.1).

The harness answers each with a separate mechanism: a held-out suite (§2), template variation and paired statistics (§3, §5), and a contamination gate plus perturbation sets (§4, §7).

Position in the pipeline: this lab consumes checkpoints from SFT ([[ch-36]]) and RL ([[ch-46]]), applies the measurement theory of [[ch-50]] and [[ch-51]] and the safety protocol of [[ch-52]], and emits the memo that the infrastructure track ([[ch-54]] onward) uses to decide whether a run is worth scaling.

---

## §1 Deliverables and the two budget paths

The lab produces five artifacts.

| Artifact | Content | Section |
|---|---|---|
| `suite.yaml` | task ids, split label (`dev` or `heldout`), ≥3 prompt templates per task, decoding settings, scorer | §2, §3 |
| `runs/<ckpt>/<task>/<template>.jsonl` | one row per question: id, cluster id, prompt, completion, score, slice labels | §1 |
| `contamination.json` | per-task overlap rate against each training set, plus the gate verdict | §7 |
| `forgetting.md` | base → SFT → RL deltas with paired intervals, per capability group | §6 |
| `memo.md` | verdict, gate status, non-inferiority table, regressions, recommendation | §9 |

**Full-budget path.** Live generation for every checkpoint × task × template. Six held-out tasks, three templates each, 500 questions per task, K = 4 samples per question on generative tasks. About 10 GPU-hours for three checkpoints at 8B (estimate from the token counts, not a measured figure).

**Resource-constrained path.** Reuse the generations already written by the ch-36 and ch-46 labs (both labs save rollouts to JSONL), add only the held-out tasks that those labs did not run, and set K = 1. This path runs on one workstation, since it decodes nothing that the earlier labs already decoded.

What may be dropped on the constrained path: live decoding, K > 1, and the agentic slice ([[ch-51a]]). What may not be dropped: the held-out split, ≥3 templates per task, the paired analysis, the multiple-comparison correction, and the contamination gate. Those four decide whether the output is a score with a stated error rate or a point number with none, and none of them consumes GPU time.

A note on scope: this harness scores single-turn and multi-turn text tasks with programmatic scorers. Pairwise judge tasks (AlpacaEval-style win rates) are measured by the machinery of [[ch-49]], whose bias controls are a precondition for trusting any judge number; this lab uses judges only where the scorer is a released classifier with published agreement rates (§8).

The stage-by-stage flow, including which gate blocks which report, is drawn in [figures/eval-pipeline.html](figures/eval-pipeline.html) — the toggles let you trip each gate in turn and watch the memo verdict change.

### The data model

Four record types are enough. Slices and templates are fields on the sample, not filters applied afterwards, because a slice that is reconstructed after the run cannot be paired across checkpoints reliably.

```python
# harness/core.py
@dataclass(frozen=True)
class Sample:
    sample_id: str                 # stable across checkpoints and templates
    cluster_id: str                # questions drawn together share this (see §5)
    prompt_vars: Mapping[str, str] # rendered by a template, not a fixed string
    gold: str                      # or a list of accepted answers
    slices: Mapping[str, str]      # {"subject": "algebra", "length_bucket": "long"}

@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str                     # "dev" | "heldout"
    samples: Sequence[Sample]
    templates: Sequence[str]       # >= 3, see §3
    scorer: Callable[..., float]      # (completion, gold) -> score in [0,1]
    decoding: Mapping[str, object] # temperature, max_new_tokens, K
    contamination_scope: Sequence[str]   # training-set dumps to check against

@dataclass(frozen=True)
class Score:                       # one row per (checkpoint, task, template, sample, k)
    checkpoint: str; task: str; template: str
    sample_id: str; cluster_id: str; k: int; value: float
```

Everything downstream is an aggregation over `Score` rows. Keeping the per-sample rows is what makes §5's paired analysis possible; a harness that stores only task means cannot compute a paired standard error afterwards.

---

## §2 The held-out generality suite

**Definition.** A held-out suite is a set of evaluation tasks whose scores are not inspected while the training data, hyperparameters, or checkpoints are being chosen. It is not the same thing as a test split of a benchmark that was used during development: reading any score from a benchmark during development couples the development decisions to that benchmark's idiosyncrasies.

**The measurable problem.** Development scores overstate capability by an amount that is not knowable from the development scores themselves. Tülu 3 quantified this by constructing the unseen suite through an independent design process and reporting both splits for the same checkpoints ([[tulu-3]] §7.3, §7.4).

**Mechanism.**
1. Assign each core capability a development task (used freely during development) and a held-out task that tests the same capability with a different formulation.
2. Freeze the held-out suite before the first training run, and record its hash in `suite.yaml`.
3. Read held-out scores only at the gate, never inside a tuning loop.
4. Report the pair (development, held-out) for every checkpoint, and the gap between them.

**Evidence.** Tülu 3 8B, one development and one held-out task per skill (arXiv:2411.15124 Table 31; extract in [[held-out-suite-tulu3]]):

| Skill (dev → held-out task) | 8B SFT | 8B DPO | 8B Final | Held-out: SFT | DPO | Final |
|---|---|---|---|---|---|---|
| Average over skills | 64.9 | 68.3 | 68.8 | 29.9 | 31.9 | 32.4 |
| Knowledge (MMLU → GPQA) | 65.9 | 68.7 | 68.2 | 31.9 | 31.2 | 35.7 |
| Reasoning (BBH → AGIEval English) | 67.9 | 65.8 | 66.0 | 56.2 | 61.8 | 59.3 |
| Math (MATH → DeepMind Mathematics) | 31.5 | 42.0 | 43.7 | 32.3 | 33.0 | 35.4 |
| Coding (HumanEval → BigCodeBench) | 86.2 | 83.9 | 83.9 | 11.5 | 9.5 | 7.4 |
| Instruction following (IFEval → IFEval-OOD) | 72.8 | 81.1 | 82.4 | 17.6 | 23.9 | 24.3 |

Two readings matter. The pipeline improved both averages, which is the authors' headline (§7.4.1). The coding row moved the other way on the held-out task at every stage, ending 4.1 points below the SFT checkpoint while the development score moved 2.3 points and then stopped. A report that carried only HumanEval would describe the coding trajectory as flat.

**Conditions and limits.** Tülu 3's unseen suite has no safety task (§2.2), so §8 of this lab supplies one. The authors also note they cannot verify that the comparison models never trained on GPQA, MMLU-Pro, AGIEval, DeepMind Mathematics, or BigCodeBench, so cross-model held-out comparisons are weaker than within-pipeline ones (§7.4.2). Result (single study).

**The lab's suite.** Six held-out tasks, chosen so that no task overlaps the data used in [[ch-36]] or [[ch-46]]:

| Capability | Held-out task | Why this one |
|---|---|---|
| Knowledge | MMLU-Pro | 12,032 items with up to ten answer options (83% have ten; mean 9.47); prompt sensitivity about 2% against MMLU's 4–5% across the same 24 prompts ([[mmlu-pro]] §3.1, §3.2, §6.3) |
| Knowledge, hard | GPQA | Tülu 3's held-out knowledge task alongside MMLU-Pro ([[tulu-3]] Table 3) |
| Reasoning | BBH or AGIEval English | BBH if it was not used in development; AGIEval English otherwise ([[tulu-3]] Table 24) |
| Instruction following | IFBench | 58 constraints disjoint from IFEval's 25; GPT-4.1 and Claude 3.7 Sonnet score below 50% ([[ifbench]] §1, Fig. 1) |
| Math robustness | GSM-Symbolic-style variants or GSM1k | §4 |
| Multilingual | MGSM | translated copies of one question form clusters, which forces the clustered standard error of §5 ([[adding-error-bars-evals]] §2.2) |

IFBench is the strongest single addition for a model that has been through instruction-following RL. On the Tülu-3-8B-DPO policy, IF-RLVR training raised IFEval 81.1 → 92.2 and IFBench 25.2 → 44.6 ([[ifbench]] §4.7, Table 6), so the two benchmarks move together but at different rates; a pipeline that reports only IFEval cannot separate constraint-following ability from IFEval-template fitting.

---

## §3 Prompt templates as an axis of the suite

**Definition.** A template is the fixed text that surrounds a question: system message, instruction phrasing, answer-format request, and few-shot block. Two templates that request the same answer in different words define two different measurements of the same underlying skill.

**The measurable problem.** Score differences between checkpoints can be smaller than score differences between templates for one checkpoint. MMLU-Pro's authors evaluated models under 24 different reasonable prompts: on MMLU the spread was generally 4–5% with a maximum of 10.98%, and on MMLU-Pro generally about 2% with a maximum of 3.74% ([[mmlu-pro]] §6.3). A 3-point improvement claimed from one template on an MMLU-like task is inside the template spread.

**Mechanism.**
1. Write three templates per task: the benchmark's canonical one, a plain-instruction variant with no few-shot block, and a variant that changes the answer-format request (for example `Answer: X` instead of `Therefore, the answer is X`).
2. Run every checkpoint under all three. Never mix templates across checkpoints.
3. Report, per task, the mean over templates and the range across templates.
4. Gate on the minimum over templates, not the maximum, so that the reported claim holds under the least favourable of the three.

**Worked example.** Checkpoint B on the knowledge task scores 48.2, 51.0, 45.9 under templates T1–T3; checkpoint A scores 47.0, 47.6, 46.6. Means: B 48.37, A 47.07, difference +1.30. Range: B 5.1, A 1.0. The template range for B is larger than the mean difference, and B's advantage disappears under T3 (−0.7). The memo reports "+1.3 mean, −0.7 worst-template" and does not claim an improvement.

**Conditions and limits.** Template variance is a property of the benchmark and of the checkpoint. MMLU-Pro reduced it by making items harder and adding options; the authors treat model robustness itself as outside their scope (§6.3). Result (single study). Note also that answer-extraction code is part of the template: Tülu 3 reports that its models formatted answers in LaTeX on DeepMind Mathematics, where LaTeX was not requested, and that this interfered both with the reasoning and with the extraction logic ([[tulu-3]] §7.4.1).

---

## §4 Perturbation robustness

**Definition.** A perturbation set is a set of questions generated from the same reasoning templates as a benchmark, with surface details changed: names, numeric values, added clauses that do not change the answer, or an entirely new set of items written to the same specification.

**The measurable problem.** After SFT or RL on math data, accuracy on GSM8K or MATH can rise for two different reasons: the model solves more grade-school arithmetic problems, or the model has become better at the specific items and phrasings in those benchmarks. The two are distinguished by holding the reasoning constant and changing the surface.

**Mechanism (template instantiation).** GSM-Symbolic converts a GSM8K question into a template with typed variables and a constraint that keeps answers integral, then samples instantiations ([[gsm-symbolic]] §3.1, Fig. 1). The evaluation protocol uses 100 templates × 50 samples = 5,000 items, treated as 50 datasets of 100 items each, scored with 8-shot CoT and greedy decoding (§3.2).

**Evidence.**
- Across the 50 sets, the gap between a model's worst and best set exceeds 12% for Gemma2-9B and is around 15% for Phi-3.5-mini ([[gsm-symbolic]] §4.1).
- The accuracy on the 100 original GSM8K items is often more than one standard deviation above the centre of that model's GSM-Symbolic distribution; this holds for 21 of 25 models (§4.1). The authors name data contamination as one possible explanation.
- Changing only names produces less variance than changing numbers, and changing both produces the largest drop (§4.2). Adding one clause that looks relevant but changes nothing (GSM-NoOp) reduces accuracy by up to about 65%, with Phi-3-mini-128k-instruct at −65.7 and Gemma2-9b-it at −63.0 (§4.4, Fig. 8a).
- The independent-items variant: GSM1k is 1,205 new grade-school problems written by human annotators without model assistance. The worst models score 8% lower on GSM1k than on GSM8K; the per-character log-likelihood of generating the GSM8K test set correlates with a model's GSM8K−GSM1k gap at Spearman 0.36 (p = 0.03) ([[gsm1k]] §5.4). Difficulty was matched: annotators solved 4.07 ± 0.93 GSM8K problems and 4.36 ± 1.11 GSM1k problems in 15 minutes (§3.2.2). **Replicated** with [[gsm-symbolic]] in the sense that both find benchmark-specific inflation; the mechanisms measured differ (item novelty vs surface perturbation).

**What the lab builds.** GSM1k is not public, so the lab generates its own perturbation set: take 100 items from the math data used in [[ch-46]]'s evaluation, rewrite each as a template with two to four variables, and sample 30 instantiations per template. Report three numbers per checkpoint: the original-items accuracy, the mean over the 30 sets, and the standard deviation over the 30 sets. The quantity that goes in the memo is the **perturbation gap** = original − perturbed mean, expressed in standard deviations of the perturbed distribution.

**Worked example.** Checkpoint B scores 71.0 on the original 100 items. The 30 instantiation sets give a mean of 63.4 with standard deviation 3.1. The gap is 7.6 points = 2.45 σ. Checkpoint A (the SFT model) scores 58.0 original, 56.4 perturbed mean, σ = 2.9, gap 1.6 points = 0.55 σ. B gained 13 points on the original items and 7 points on the perturbed mean; 6 of the 13 points, slightly under half of B's apparent gain, do not survive perturbation. The memo reports both, and the RL stage is recorded as having increased the perturbation gap by 1.9 σ.

**Conditions and limits.** GSM-Symbolic's templates are grade-school arithmetic; the result has not been shown to transfer to competition mathematics or to code. The interpretation offered by the authors — that models perform in-distribution pattern matching rather than formal reasoning — is an **Interpretation** of the accuracy drops, not a separate measurement.

---

## §5 The statistics the memo needs

All four items below come from [[adding-error-bars-evals]] and are quoted with their equation numbers in [[paired-statistics-and-power]].

### 5.1 Paired differences

Miller writes the difference as model minus baseline. In this lab the baseline is A (the earlier checkpoint) and the candidate is B, so for the same question i let `d_i = s_{B,i} − s_{A,i}`. Then

```
SE_paired = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )        (Eq. 7)
```

where `n` is the number of questions, `d̄` the mean difference over questions, and `d_i` the per-question difference. The variance relation is `Var(µ̂_paired) = Var(µ̂_unpaired) − 2 Cov(x_A, x_B)/n`, so pairing helps whenever the two checkpoints agree about which questions are hard (§4.2).

**Worked example.** n = 500 questions, binary scoring. The 2×2 table: both correct 165, A correct and B wrong 35, A wrong and B correct 60, both wrong 240.
- `p_A = (165+35)/500 = 0.400`, `p_B = (165+60)/500 = 0.450`, difference `p_B − p_A = +0.050`.
- Per-question differences are +1 on 60 questions, −1 on 35, 0 on 405. Mean = 25/500 = 0.050. Because each difference squares to 0 or 1, `E[d²] = 95/500 = 0.190` and `Var(d) = 0.190 − 0.050² = 0.1875`.
- `SE_paired = sqrt(0.1875/500) = 0.0194` → 95% interval `+5.0 ± 3.8 pp = [+1.2, +8.8]`.
- Unpaired: `SE_A = sqrt(0.4·0.6/500) = 0.0219`, `SE_B = sqrt(0.45·0.55/500) = 0.0222`, `SE_unpaired = sqrt(SE_A² + SE_B²) = 0.0312` → `+5.0 ± 6.1 pp = [−1.1, +11.1]`.

Same data, same questions: the paired interval excludes zero and the unpaired interval does not. The variance ratio here is 2.6×. This is why the regression rule of this lab is stated on the paired interval and why a rule of the form "B regresses if B's interval lies below A's mean" is not used — that rule mixes two unpaired intervals and both its false-positive and false-negative rates are unknown.

### 5.2 Clustered standard errors

When questions arrive in groups — one passage with several questions, or one question translated into several languages — the independence assumption behind the ordinary standard error and behind the ordinary bootstrap fails. Compute the standard error over clusters (Eq. 8 for the paired version, in [[paired-statistics-and-power]]). Measured on Anthropic models, the clustered standard error was 3.05× the naive one on DROP, 1.88× on MGSM, and 1.10× on RACE-H ([[adding-error-bars-evals]] Table 4). The lab's MGSM task therefore carries `cluster_id` = source question id, and its interval is roughly twice as wide as the naive computation suggests.

### 5.3 Power and the minimum detectable effect

```
n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²                  (Eq. 9)
```

`n` = number of independent questions, `α` = false-positive rate, `1−β` = power, `δ` = minimum detectable effect, `ω²` = variance of the difference of conditional means, `σ²_A, σ²_B` = expected conditional variances of the two models' scores, `K_A, K_B` = samples drawn per question. Miller's worked case: `σ²_A = σ²_B = 0`, `ω² = 1/9`, `δ = 0.03`, `α = 0.05`, `β = 0.20` gives `n = (1.96 + 0.84)² (1/9) / 0.03² ≈ 969`, from which the author concludes that new evals should contain at least 1,000 questions (§5).

Inverted for this lab's 500-question tasks and the discordance rate of §5.1 (`Var(d) = 0.1875`):

```
δ_min = (1.96 + 0.84) · sqrt(0.1875/500) = 2.80 · 0.0194 = 0.054
```

A 500-question task with that much disagreement between checkpoints can detect 5.4 pp differences at 80% power and nothing smaller. If the checkpoints agree more — say only 5% of questions discordant, `Var(d) ≈ 0.05` — the same 500 questions detect 2.8 pp. Two consequences for the memo. Detecting a 1 pp difference at 80% power would need about 14,700 questions at `Var(d) = 0.1875`. Declaring a group non-inferior at a 1 pp margin is a different and slightly weaker requirement — the interval's lower bound must clear −1 pp, so `1.96·SE < 0.01` and `n ≈ 7,200` — and it is also out of reach at n = 500, where the half-width is 3.8 pp. The margin is therefore chosen from the measured half-width, and a margin narrower than it returns `inconclusive` for every group, including groups whose delta is zero.

Raising `K` reduces only the conditional-variance term: going from K = 1 to K = 2 cuts the total variance by 1/3 in Miller's uniform-difficulty example, K = 4 by 1/2, with an upper limit of 2/3 (§3.1). Lowering the sampling temperature to reduce variance is not a substitute. In the paper's first single-token example it moves the variance of the conditional means from 1/12 to 1/4 while leaving the expected score unchanged; in the second, where the conditional means are uniform on [1/3, 1], it moves variance from 1/27 to 3/16 and also shifts the expected score from 2/3 to 3/4 (§3.3). Fix the temperature at the value the deployment uses and buy precision with `K` and `n`.

### 5.4 Multiple comparisons

The suite produces one test per (task × slice). With 20 such tests and α = 0.05, the expected number of false rejections under a true null is 1. Use the Holm step-down procedure: sort the p-values ascending as `p_(1) ≤ … ≤ p_(K)`, compare `p_(i)` against `α/(K − i + 1)`, and stop at the first failure; everything from there on is not rejected. This is a standard procedure, not an empirical claim.

**Worked example.** K = 6 slice tests with p-values 0.004, 0.011, 0.030, 0.040, 0.200, 0.500. Thresholds: 0.05/6 = 0.00833, 0.05/5 = 0.0100, 0.05/4 = 0.0125, 0.05/3 = 0.0167, 0.05/2 = 0.025, 0.05/1 = 0.05. The first test passes (0.004 < 0.00833); the second fails (0.011 > 0.0100) and the procedure stops. The two slices that would have looked significant at an uncorrected 0.05 (p = 0.030, 0.040) are not reported as regressions. They are reported in the memo as unresolved at this sample size, with their `δ_min`.

Slices with n < 30 are labelled `insufficient` and take no part in the correction or the verdict; including them inflates K and weakens every other test.

---

## §6 The forgetting report

**Definition.** A forgetting report is a table of paired deltas between consecutive checkpoints of one pipeline, over capability groups that the later stages did not train, with intervals and a non-inferiority decision per group.

**The measurable problem.** Each stage optimizes an objective defined on a subset of capabilities. Whether the other capabilities survive is an empirical question that the stage's own training signal cannot answer.

**Mechanism.**
1. Fix three checkpoints: base, post-SFT ([[ch-36]]), post-RL ([[ch-46]]).
2. Score all three on the same questions, same templates, same decoding.
3. For each capability group, compute the paired delta and its interval against the previous stage and against the base.
4. Apply Holm across groups (§5.4).
5. Mark a group `non-inferior` if the paired interval clears the pre-registered margin m from §5.3: for a higher-is-better metric the lower bound must exceed −m; for a lower-is-better metric such as attack success rate or over-refusal, the upper bound must fall below +m. The direction is a field on the task, not a convention the reader is expected to remember.

**Reference numbers.** Tülu 3 8B, development-suite scores, SFT → DPO → Final RLVR (arXiv:2411.15124 Table 6):

| Benchmark | SFT | DPO | Final | Final − SFT |
|---|---|---|---|---|
| GSM8K (8-shot CoT) | 76.2 | 84.3 | 87.6 | +11.4 |
| MATH (4-shot CoT, Flex) | 31.5 | 42.0 | 43.7 | +12.2 |
| IFEval (prompt loose) | 72.8 | 81.1 | 82.4 | +9.6 |
| AlpacaEval 2 (LC win) | 12.4 | 33.5 | 34.5 | +22.1 |
| MMLU (0-shot CoT) | 65.9 | 68.7 | 68.2 | +2.3 |
| BigBenchHard (3-shot CoT) | 69.7 | 68.7 | 69.0 | −0.7 |
| HumanEval (pass@10) | 86.2 | 83.9 | 83.9 | −2.3 |
| PopQA (15-shot) | 29.3 | 29.3 | 29.1 | −0.2 |
| Safety (6-task average) | 93.1 | 87.2 | 85.5 | −7.6 |

Read the last row against the first. The stages that produced +11.4 on GSM8K and +9.6 on IFEval also moved the safety average down 7.6 points from the SFT checkpoint. Tülu 3 does not report intervals for these numbers, so the magnitudes are single-run point estimates; the lab's own version of this table carries paired intervals, which is the addition this chapter makes. Note also that most of the GSM8K and IFEval gain is at the DPO stage (+8.1 and +8.3), and the RLVR stage adds +3.3 and +1.3. Attributing the whole delta to the RL stage is a common misreading of this table.

**A second reference, for the instruction-following case.** After IF-RLVR from the Tülu-3-8B-DPO policy, AlpacaEval 2 moved 33.5 → 21.3 and MMLU 68.7 → 66.4 while IFEval and IFBench rose; GPT-4.1 judge scores of the same responses with the constraint removed moved from 7 to 6.4 out of 10 ([[ifbench]] §5, Table 3). Constraint-only rewards buy constraint satisfaction and spend general response quality. **Result (single study)**, and the paper's own remedy — mixing a reward-model score into the verifiable reward — lowered IF scores and raised AlpacaEval 2 to 31.6 (App. E).

**Report format** (`forgetting.md`):

```
| group        | base  | SFT   | RL    | Δ(RL−SFT) 95% CI      | margin | verdict        |
|--------------|-------|-------|-------|-----------------------|--------|----------------|
| knowledge    | 41.2  | 44.0  | 43.6  | −0.4 [−3.9, +3.1]     | −3.0   | inconclusive   |
| math         | 22.8  | 31.5  | 44.1  | +12.6 [+8.2, +17.0]   | −3.0   | improved       |
| math-perturb | 21.9  | 29.9  | 37.1  | +7.2 [+3.0, +11.4]    | −3.0   | improved       |
| instr-follow | 15.1  | 17.6  | 24.3  | +6.7 [+2.9, +10.5]    | −3.0   | improved       |
| safety-ASR   | 31.0  | 8.2   | 14.9  | +6.7 [+2.1, +11.3]    | +3.0   | REGRESSION     |
| over-refusal | 4.0   | 11.2  | 9.6   | −1.6 [−5.4, +2.2]     | +5.0   | non-inferior   |
```

`inconclusive` is a verdict, not a pass. A group whose interval spans the margin has not been shown non-inferior, and the memo says so.

---

## §7 The contamination gate

**Definition.** Contamination is overlap between the evaluation items and the training data of the checkpoint under test. The gate is a check that runs before scores are read and that blocks reporting when the overlap rate exceeds a stated threshold.

### 7.1 When the training data is available

Tülu 3's procedure, which this lab copies, works on prompts only, because completions in training sets are often regenerated ([[tulu-3]] §3.2):

1. Tokenize the evaluation prompt and the training prompt.
2. A token of the evaluation instance counts as matched if the two instances share an 8-gram containing that token.
3. The evaluation instance overlaps the training instance if more than 50% of its tokens are matched against that same training instance.
4. A training set is contaminated with respect to an evaluation if any of its instances overlap more than 2% of that evaluation's instances.

**Worked example.** An evaluation prompt has 40 tokens. A training instance shares a verbatim span of 25 tokens. Every token inside a 25-token span belongs to some shared 8-gram, so 25/40 = 62.5% > 50% and the instance is flagged. If the shared span were 12 tokens, 12/40 = 30% and it is not flagged. On a 500-item held-out task, the dataset-level rule trips at 2% × 500 = 10 flagged items.

**Base rates, so the threshold is not mistaken for a small number.** Tülu 3 lists the public datasets it found contaminated at >5% evaluation overlap (Table 37): Evol CodeAlpaca vs HumanEval 70.7%; LMSys Chat 1M vs AlpacaEval 46.5%, vs HumanEval 17.7%, vs MMLU 10.3%, vs GSM8K 8.9%; NuminaMath-TIR vs MATH 18.2%; DaringAnteater vs MATH 30.7%. Decontaminating those sets removed 3.5% of Evol CodeAlpaca and 11.3% of NuminaMath-TIR (Table 8). The 2% rule is Tülu 3's own dataset-level decision rule; it is not a figure from the deduplication literature.

### 7.2 Candidate generation with MinHash, set correctly

Exhaustive 8-gram comparison against a multi-million-document training set is not affordable, so candidates are generated with a banded MinHash index and then verified exactly. The parameters must be read carefully. [[deduplicating-training-data]] §4.2 uses 5-gram shingles and a signature of k = 9,000 hash values **split into r = 450 buckets of b = 20 hashes each**, with candidate probability

```
P(candidate | Jaccard s) = 1 − (1 − s^b)^r ,   b = 20 hashes per bucket, r = 450 buckets
```

and confirms candidates by requiring Jaccard > 0.8 **and** edit similarity > 0.8, where `EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j)/max(|x_i|, |x_j|)`.

**Worked example.** With b = 20 and r = 450: at s = 0.5, `s^b = 9.5e−7` and `P = 0.0004`; at s = 0.6, `P = 0.016`; at s = 0.7, `P = 0.302`; at s = 0.8, `P = 0.995`. The curve's midpoint is near `(1/r)^(1/b) = (1/450)^(1/20) = 0.737`, below the 0.8 confirmation threshold, so the index proposes almost every true near-duplicate and the exact check removes the rest.

Swapping b and r inverts the behaviour: with 20 buckets of 450 hashes the midpoint is `(1/20)^(1/450) = 0.993`, and a pair at Jaccard 0.9 is proposed with probability under 1e−15. Most library APIs take the pair as `(bands, rows_per_band)`, so the assignment has to be checked against the S-curve rather than against the paper's prose. [figures/lsh-band-curve.html](figures/lsh-band-curve.html) plots both settings and lets you move b and r to see which pairs survive.

```python
# harness/contamination.py
NUM_PERM, HASHES_PER_BUCKET, BUCKETS = 9000, 20, 450     # Lee et al. 2021 §4.2
assert HASHES_PER_BUCKET * BUCKETS == NUM_PERM
lsh = MinHashLSH(num_perm=NUM_PERM, params=(BUCKETS, HASHES_PER_BUCKET))  # (bands, rows)

def confirm(a_tokens, b_tokens):                          # exact verification
    j = jaccard(set(five_grams(a_tokens)), set(five_grams(b_tokens)))
    e = 1 - edit_distance(a_tokens, b_tokens) / max(len(a_tokens), len(b_tokens))
    return j > 0.8 and e > 0.8
```

MinHash here is a candidate generator for the near-duplicate case. The decision rule that trips the gate remains the 8-gram token-coverage rule of §7.1, because it catches the cases the Tülu 3 authors cared about: instances that are trivially different, such as a math problem in which only the numbers changed (§3.2).

### 7.3 When the training data is not available

For a closed checkpoint or a distillation teacher, no string matching is possible, and the substitute is a behavioural probe. [[swe-bench-illusion]] ran three probes, of which two apply here. Given only the repository name and the issue text, ten OpenAI and Anthropic models named a file touched by the gold patch in 60–76% of SWE-Bench Verified instances but in under 53% of 245 comparable tasks from repositories outside the benchmark (§4.1.1–4.1.2). Function-reproduction 5-gram overlap reached 34.9% on SWE-Bench versus 13.9–18.2% elsewhere (§4.3). [[gsm1k]] used a different probe: the per-character log-likelihood of the benchmark's test items, which correlates with the benchmark-to-fresh-set gap at Spearman 0.36 (§5.4). Both are gap measurements, and both need a comparable fresh set — which is what §4 builds.

The gate's verdict field takes three values: `clean` (checked, under threshold), `contaminated` (over threshold; scores are withheld), `unknown-training-data` (probe result reported, no string check possible). `unknown-training-data` never counts as a pass.

---

## §8 Policy-level safety and over-refusal

The measurement here is the behaviour of the checkpoint itself, not its ability to classify text. Feeding a moderation test set to the policy and scoring its labels against the moderation gold labels measures the checkpoint as a classifier; that is a different quantity from whether the checkpoint complies with harmful requests. The lab measures two policy behaviours, following [[ch-52]].

**Attack success rate.** Prompts from HarmBench's behaviour library are sent to the policy; the policy's completions are scored by HarmBench's released classifier ([[harmbench-data]]; extract in [[harmbench-behavior-layering]]).
- Scoring: the fine-tuned test classifier reaches 93.2% agreement with human labels, the separate validation classifier 88.6%; the test classifier makes 41 errors and the validation classifier 51, with 26 in common (App. B.2). Agreement against human judgments on standard behaviours: HarmBench classifier 94.53%, GPT-4 89.8%, GPTFuzz 77.36%, AdvBench substring matching 71.14% (Table 3). Substring matching is not forbidden by the paper; it is 23 points less accurate than the released classifier on standard behaviours, which is the reason not to use it.
- Generation length is part of the measurement: the number of tokens generated during evaluation changes ASR by up to 30%, and HarmBench standardizes it at N = 512 (§3.2, Fig. 2). Record it in `suite.yaml`.
- Use the validation classifier inside any optimization loop and the test classifier only at the gate; the paper does not allow direct optimization against the test metric (App. B.2).

**Over-refusal.** XSTest sends 250 safe prompts that resemble unsafe ones plus 200 unsafe contrasts and labels responses as full compliance, full refusal, or partial refusal ([[xstest]] §3, §4.2). The reference numbers show why both halves are needed: Llama-2-70b-chat with its original system prompt fully refuses 38% of the safe prompts and partially refuses 21.6%, while refusing 99.5% of the unsafe ones; Mistral-7B-Instruct-v0.1 without a system prompt refuses 0.8% + 0.8% of safe prompts and only 23.5% + 12.5% of unsafe ones (Tables 1, 2). Removing Llama-2's system prompt moved safe-prompt full refusal 38% → 14% and unsafe-prompt full refusal 99.5% → 97.5%, so the system prompt is part of the measurement and must be pinned in `suite.yaml`.

The two numbers are reported together as a pair (ASR, over-refusal). A checkpoint that improves one while worsening the other has moved along the trade-off between them rather than improving both. Automated refusal scoring has known error: XSTest's string matching called "I'm sorry to hear about your hamster" a refusal, and zero-shot GPT-4 classification failed on partial refusals, though both preserved the model ranking from manual annotation (§4.5).

---

## §9 The go/no-go memo

The memo is the artifact the rest of the course reads. It has a fixed header so that two memos from different weeks can be diffed.

```
# Eval memo: <run_id>            date: <YYYY-MM-DD>   suite hash: <sha>
## Verdict: GO | NO-GO | NEEDS-FIX | INCONCLUSIVE
## Checkpoints
  A = ch-36-sft <sha>     B = ch-46-rl <sha>     base = <sha>
## Gates (all must pass before any score below is read)
  contamination: clean | contaminated(task=<>, rate=<>) | unknown-training-data(probe=<>)
  template coverage: <n_templates per task, min 3>
  power: delta_min and CI half-width per task = <...>; non-inferiority margin m = <...>
## Held-out suite
  per task: worst-template score, mean over templates, paired delta vs A [95% CI], n, cluster count
  held-out average, development average, and the gap between them
## Forgetting report      (table from §6, Holm-corrected)
## Perturbation           original, perturbed mean, sigma, gap in sigma, per checkpoint
## Safety                 (ASR, over-refusal) pairs per checkpoint
## Open items             slices marked insufficient or inconclusive, with the n they would need
## Recommendation         ship | re-run with n=<> | revisit ch-46 <arm> | investigate slice <>
```

Verdict rules, in order:

1. `NO-GO` if the contamination gate or the template-coverage gate fails, including `unknown-training-data` when the checkpoint is one the course trained.
2. `NO-GO` if any capability group fails non-inferiority after Holm correction.
3. `NEEDS-FIX` if the held-out average rises but one or more groups are `inconclusive` at the pre-registered margin. A margin narrower than 1.96·SE puts every group in this state; the memo then prints the sample size that would resolve it rather than a verdict on the checkpoint.
4. `INCONCLUSIVE` if the held-out average change lies inside its own paired interval.
5. `GO` otherwise: the held-out average rises with an interval excluding zero and every group is non-inferior.

Rule 2 is the part that distinguishes this memo from a benchmark table. A checkpoint that gains 12 points on the trained task and loses 7 on safety does not pass, and the ordering of the rules means no reader has to weigh the two against each other by intuition.

---

## Negative samples and negative feedback

This lab does not train, so it neither produces nor applies a training gradient. It touches negatives in two ways, and the distinction from §6.1 of the course standard matters.

**1. The harness's own negative outputs are negatives as content, not as gradient.** Every failing `Score` row, with its slice labels, prompt, and completion, is an input to the next data round: failure buckets from [[ch-50]] become targeted prompts for [[ch-36]] or [[ch-46]]. They are used with ordinary cross-entropy on a corrected target, or as prompts for a new rollout. No likelihood is pushed down by this lab.

**2. The lab measures the consequences of negatives used as gradient elsewhere.** The RL stage of [[ch-46]] applies negative advantages, and the preference stage applies the rejected-response term. The known side effects of those are coverage loss and calibration change, and they are visible only in specific measurements:
- **Coverage.** Report pass@1 and pass@k at a large k on at least one task. A stage that raises pass@1 while lowering pass@k at large k has concentrated probability mass rather than added solutions ([[ch-46]]).
- **Abstention and over-refusal.** The XSTest full-refusal rate on safe prompts is the direct measurement; it rises when safety negatives are pushed down too hard (§8).
- **Diversity within a question.** With K samples per question, record the number of distinct final answers. Collapse to a single answer across all K is the sampling-level signature of the same pressure.
- **General response quality.** [[ifbench]] §5 measured it with a judge score on responses with the constraint removed: 7 → 6.4 out of 10 after constraint-only RLVR, with AlpacaEval 2 at 33.5 → 21.3.

**Honesty about size of effect.** None of these measurements attributes a share of the gain or the loss to the negative term by itself; separating positive from negative contributions requires the ablation run built in [[ch-46]], not this harness. The harness's job is to make the side effect visible with an interval attached.

---

## Recipe

Rows are the settings this lab inherits from published harnesses. Lab values differ only where budget forces it.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| Tülu 3 8B / 70B | 8B, 70B | eval-gate | suite split | development suite (MMLU, PopQA, TruthfulQA, BBH, DROP, GSM8K, MATH, HumanEval(+), IFEval, AlpacaEval 2, Safety); unseen suite (MMLU-Pro, GPQA, AGIEval English, DeepMind Mathematics, BigCodeBench, IFEval-OOD, HREF); unseen scores never examined during development | arXiv:2411.15124 §2.2, Table 3, Table 24 | verified 2026-09-17 | Table 31: final checkpoints best on both split averages; coding unseen 11.5 → 7.4 | 6 held-out tasks (§2), same rule on inspection | no safety task in Tülu's unseen suite; budget caps task count |
| Tülu 3 (all sizes) | n/a | eval-gate | decontamination | 8-gram prompt matching; instance flagged if >50% of its tokens match one training instance; training set contaminated if >2% of an eval's instances overlap | arXiv:2411.15124 §3.2 | verified 2026-09-17 | Table 37 overlap rates; Table 8 removal rates (3.5% Evol CodeAlpaca, 11.3% NuminaMath-TIR) | identical | none |
| Lee et al. 2021 NearDup | n/a | eval-gate | MinHash index | 5-gram shingles; k = 9,000 hashes as r = 450 buckets of b = 20; confirm Jaccard > 0.8 and edit similarity > 0.8 | arXiv:2107.06499 §4.2, App. A | verified 2026-09-17 | App. A Fig. 4 also reports a 0.9/0.9, b = 20, r = 40, k = 800 setting | identical, candidate generation only | the gate decision stays the 8-gram rule |
| HarmBench | n/a | eval-gate | generation length; scorer | N = 512 generated tokens; fine-tuned Llama-2-13B-Chat test classifier; Mistral-7B validation classifier for in-loop use | arXiv:2402.04249 §3.2, §4.3, App. B.2 | verified 2026-09-17 | Fig. 2: ASR changes up to 30% with token count; Table 3: 94.53% vs 71.14% substring agreement | identical | none |
| XSTest | n/a | eval-gate | prompts; decoding; labels | 250 safe + 200 unsafe prompts; temperature 0, max 256 tokens; full compliance / full refusal / partial refusal | arXiv:2308.01263 §3, §4.2, App. B | verified 2026-09-14 ([[xstest]]) | Table 2: system prompt moves safe-prompt refusal 38% → 14% | identical; system prompt pinned per checkpoint | the lab's checkpoints share one system prompt |
| GSM-Symbolic protocol | n/a | eval-gate | template count; samples; decoding | 100 templates × 50 samples = 5,000 items as 50 sets of 100; 8-shot CoT, greedy | arXiv:2410.05229 §3.2 | verified 2026-09-17 | §4.1: worst-to-best set gap >12% (Gemma2-9B) | 100 templates × 30 samples as 30 sets | generation budget; σ over 30 sets is still reported |
| MMLU-Pro | n/a | eval-gate | prompt variants | 24 prompts; spread about 2% (max 3.74%) vs MMLU 4–5% (max 10.98%) | arXiv:2406.01574 §6.3 | verified 2026-09-17 | §6.3 | 3 templates per task | 24 templates × 6 tasks × 3 checkpoints is out of budget; 3 bounds the worst case |
| Miller 2024 | n/a | eval-gate | sample size | n = (z_{α/2}+z_β)²(ω² + σ²_A/K_A + σ²_B/K_B)/δ²; n ≈ 969 for δ = 0.03, α = 0.05, β = 0.20, ω² = 1/9 | arXiv:2411.00640 §5, Eq. 9 | verified 2026-09-17 | §5: "new evals should contain at least 1,000 questions" | n = 500, with δ_min and the interval half-width printed per task | budget; the margin is derived from the measured half-width rather than fixed in advance |
| IF-RLVR evaluation | 7-8B | eval-gate | decoding | temperature 0 (DeepSeek-R1: 0.6 / top-p 0.95; o3: 1); prompt-level loose accuracy | arXiv:2507.02833 §3, App. G | verified 2026-09-14 ([[ifbench]]) | not applicable | identical for IFBench | none |
| OLMo-2-era signal/noise study | 60M–32B | eval-gate | checkpoint noise window | final 5 checkpoints (small models), final 30 spaced 1,000 steps (13B target) | arXiv:2508.13144 §4.1, §4.2 | verified 2026-09-14 ([[signal-and-noise-eval]]) | App. A.3.2 Table 2: n = 5 within ±1σ for almost all benchmarks | final 3 checkpoints of each stage, averaged | the labs save fewer checkpoints; the reduction is recorded as a limitation |

**Starting point for a small general-purpose run.** Six held-out tasks at n = 500 questions, three templates each, K = 1 for multiple-choice and K = 4 for generative tasks, temperature fixed at the deployment value, HarmBench at N = 512 tokens, XSTest with a pinned system prompt, contamination gate at the Tülu 3 8-gram/50%/2% rule with the Lee et al. MinHash index for candidate generation. Every number in this paragraph comes from a `verified` row above. The conditions under which the sources used them: Tülu 3's split protocol at 8B–405B on Llama 3.1 bases; Lee et al.'s index on C4-scale corpora (360M documents); HarmBench's N = 512 on open 7B–13B chat models; Miller's n ≈ 969 under an assumed `ω² = 1/9` with zero conditional variance.

---

## Generalization lens

**(a) What increases breadth of measurement.**
- A suite split whose held-out half is never inspected during development. Tülu 3's final 8B checkpoint was best on both split averages, which is evidence that its data decisions generalized; the same table shows where they did not (coding, unseen 11.5 → 7.4) ([[tulu-3]] Table 31).
- Constraint sets disjoint from the training constraints. IFBench's 58 unseen constraints separate models that follow constraints from models that follow IFEval; GPT-4.1 and Claude 3.7 Sonnet fall below 50% on it ([[ifbench]] §1).
- Perturbation sets over the same reasoning templates, which hold the skill constant and vary the surface ([[gsm-symbolic]] §3.1).
- Benchmarks with better signal-to-noise for the decision at hand; subtask filtering by SNR raised decision accuracy by +2.6% on MMLU and +5% on AutoBencher ([[signal-and-noise-eval]] §5.1).

**(b) What causes narrowing, and how it shows up here.**
- Training on data that overlaps the evaluation. Public mixtures reach 70.7% overlap with HumanEval and 46.5% with AlpacaEval ([[tulu-3]] Table 37), and benchmark-specific memorization shows as a 60–76% versus <53% gap on file-path identification ([[swe-bench-illusion]] §4.1).
- Optimizing a narrow verifiable reward. Constraint-only IF-RLVR moved AlpacaEval 2 33.5 → 21.3 and MMLU 68.7 → 66.4 ([[ifbench]] Table 3).
- Safety tuning without an over-refusal measurement: Llama-2-70b-chat refuses 38% of safe prompts under its own system prompt ([[xstest]] Table 1).
- Reporting a single template. A 3-point claim on an MMLU-like task is inside the 4–5% spread across 24 reasonable prompts ([[mmlu-pro]] §6.3).

**(c) How generality is measured at this stage.** Four numbers, each with an interval: the held-out average and its gap to the development average; the worst-template score per task; the perturbation gap in standard deviations; and the forgetting table's non-inferiority verdicts under Holm correction. A checkpoint is recorded as more general than its predecessor only when the held-out average rises **and** no capability group fails non-inferiority.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Held-out scores consulted during tuning | held-out and development curves move together across every ablation | grep the experiment log for held-out task names before the gate timestamp; the suite hash in `suite.yaml` must predate the first run |
| Unpaired comparison on paired data | wide intervals; regressions never reach significance; sign flips between runs | recompute with Eq. 7; the paired SE must be smaller than `sqrt(SE_A² + SE_B²)` whenever the checkpoints correlate |
| Clustered questions scored as independent | intervals implausibly narrow on MGSM, DROP, RACE-style tasks | ratio of clustered to naive SE; expect up to 3.05× ([[adding-error-bars-evals]] Table 4) |
| MinHash bands and rows transposed | contamination rate near zero on every task, including known-dirty training sets | evaluate `1 − (1 − s^b)^r` at s = 0.8; it must exceed 0.9 ([figures/lsh-band-curve.html](figures/lsh-band-curve.html)) |
| Safety measured as classification accuracy | the "safety" score rises when the checkpoint becomes a better labeller of text | the scorer's input must be the policy's own completion to a behaviour prompt, not a gold label comparison |
| Generation length not pinned | ASR moves between runs with no training change | diff `max_new_tokens` across runs; HarmBench standardizes N = 512 ([[harmbench-data]] §3.2) |
| Non-inferiority margin narrower than the interval | every group returns `inconclusive`, including groups whose delta is near zero | compare the margin m with the half-width 1.96·SE; a zero-delta group can pass only when m exceeds it |
| Twenty slice tests at α = 0.05 with no correction | one or two "regressions" appear in every comparison and do not reproduce | apply Holm; count how many survive |
| Temperature lowered to tighten intervals | the conditional variance disappears but the variance of the conditional means rises, and the mean can move with it | recompute both the mean and the standard error at the deployment temperature; in Miller §3.3 the total variance rises from 1/12 to 1/4 and, in the second example, the expected score also moves |
| Perturbation set built from the same items the model trained on | perturbation gap near zero for a checkpoint with known contamination | templates must come from items outside the training mixture, checked by §7's gate |

---

## Check your understanding

1. Tülu 3's coding row moves 86.2 → 83.9 on the development task and 11.5 → 7.4 on the held-out task. Explain why the second number can fall faster than the first even though both measure coding, and what would have to be true for the opposite pattern to appear.
2. In §5.1 the paired interval excludes zero and the unpaired interval does not, on identical data. Derive which quantity the pairing removes, and construct a case where pairing would not help.
3. A colleague proposes tightening the non-inferiority margin from 3 pp to 0.5 pp "to be strict". Using the interval half-width and Eq. 9, explain what happens to the lab's verdicts and why a tighter margin does not make the gate more protective at this sample size.
4. GSM-Symbolic finds that a model's GSM8K score sits more than one standard deviation above its perturbed distribution for 21 of 25 models. Give two mechanisms that produce this pattern and describe a measurement that distinguishes them.
5. Why does the gate treat `unknown-training-data` as different from `clean`, given that both report no detected overlap? What would a checkpoint have to show for the probe result to substitute for the string check?
6. The safety row in §6 moves −7.6 while GSM8K moves +11.4 across the same stages. Explain the causal path from a verifiable-reward RL stage to a lower safety average, and name the one measurement in this harness that would distinguish that path from a decoding-configuration difference.
7. A run reports a +2.1 pp gain on the held-out average and four slice regressions, of which one survives Holm correction. State the memo verdict this harness produces and justify each part of it.

---

## Connections

- Previous: [[ch-52]] — Safety Evaluation, Over-Refusal, and Red-Teaming. Supplies the behaviour library, the released-classifier protocol, and the over-refusal contrast set that §8 runs.
- Next: [[ch-54]] — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments. Consumes this memo as the acceptance test for a scaled run.
- [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split. Supplies the SFT checkpoint and its saved generations.
- [[ch-46]] — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention. Supplies the RL checkpoint, the ablation arm, and the pass@k measurement referenced in the negative-feedback section.
- [[ch-47]] — Evaluation Harness and Suite Design for General Capability. Defines the task-shape taxonomy that `TaskSpec` implements.
- [[ch-48]] — Contamination Detection and Its Effect on Reported Scores. Derives the detection methods that §7 turns into a gate.
- [[ch-49]] — Judge Models: Bias, Calibration, and Judge-Specific Overfitting. Required before any judge-scored task is added to this suite.
- [[ch-50]] — Slice Analysis, Forgetting Slices, and Failure Bucketing. Defines the slice and failure-bucket vocabulary used in `forgetting.md`.
- [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions. Derives the paired bootstrap and the decision rule that §5 and §9 apply.
- [[ch-51a]] — Evaluating Agent Generality and Reliability. The agentic slice, omitted here, belongs to that chapter's protocol.

---

## Sources

- [[tulu-3]] — development/unseen suite design (§2.2, Table 3, Table 24), stage-by-stage 8B results (Table 6), dev-vs-unseen per skill (Table 31), decontamination rule and overlap base rates (§3.2, Tables 8, 37).
- [[adding-error-bars-evals]] — paired standard error (Eq. 7), clustered and paired-clustered forms (Eq. 4, 8, Table 4), resampling and temperature guidance (§3), sample-size formula and the n ≈ 969 example (Eq. 9).
- [[gsm-symbolic]] — template instantiation protocol (§3.1–3.2), across-set variance and the GSM8K offset for 21 of 25 models (§4.1), name-versus-number perturbations (§4.2), GSM-NoOp drops (§4.4).
- [[gsm1k]] — 1,205 human-written items, difficulty matching (§3.2), up-to-8% gap, and the log-likelihood correlation at Spearman 0.36 (§5.4).
- [[ifbench]] — 58 unseen constraints and the IFEval/IFBench gap (§1), IF-RLVR gains (Table 6), and the general-quality cost of constraint-only rewards (§5, Table 3).
- [[mmlu-pro]] — 24-prompt sensitivity comparison against MMLU (§6.3) and the benchmark's construction.
- [[harmbench-data]] — behaviour/attack/scorer separation, released classifier agreement rates (Table 3, App. B.2), and the N = 512 generation-length standardization (§3.2).
- [[xstest]] — 250 safe plus 200 unsafe prompts, three-way response taxonomy, reference refusal rates, and system-prompt sensitivity (Tables 1, 2).
- [[deduplicating-training-data]] — MinHash NearDup banding parameters (k = 9,000 as r = 450 buckets of b = 20) and the Jaccard/edit-similarity confirmation thresholds used by §7.2 (§4.2, App. A).
- [[signal-and-noise-eval]] — checkpoint-noise definition and window sizes (§3.1, §4), SNR-based subtask filtering (§5.1).
- [[swe-bench-illusion]] — memorization probes usable when training data is unavailable (§4.1, §4.3).
- [[held-out-suite-tulu3]] — extract: the Tülu 3 split tables and the inspection rule.
- [[perturbation-robustness-sets]] — extract: GSM-Symbolic and GSM1k construction and numbers.
- [[paired-statistics-and-power]] — extract: Equations 4, 7, 8, 9 with symbols defined.
- [[minhash-contamination-gate]] — extract: the 8-gram gate and the corrected banding arithmetic.
- [[harmbench-behavior-layering]] — extract: policy-level ASR protocol and classifier agreement table.
