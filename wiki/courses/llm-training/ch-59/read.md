<!-- chapter: ch-59
     track: capstone
     kind: capstone
     title: Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate
     deps: [ch-58a, ch-53, ch-46a]
     sources: [[tulu-3]], [[olmo-3]], [[smollm-3]], [[phi-4]], [[deepseek-r1]], [[deepseek-r1-recipe]],
              [[deepswe]], [[deepswe-recipe]], [[open-instruct-allenai-recipes]],
              [[open-instruct-allenai-recipes-recipe]], [[allenai-olmo3-open-instruct-scripts]],
              [[allenai-olmo3-open-instruct-scripts-recipe]], [[r2e-gym]], [[swe-gym]], [[skyrl-agent]],
              [[agenttuning]], [[reasoning-trap-tool-hallucination]], [[rlvr-beyond-base-model]],
              [[signal-and-noise-eval]], [[swe-bench-illusion]], [[agentic-benchmark-checklist]],
              [[holistic-agent-leaderboard]], [[ties-merging]], [[mix-data-or-merge-models]], [[ruler]],
              [[prolong]], [[rlvr-tulu3]], [[olmo-2]], [[grpo]],
              [[swe-smith-rft]], [[toucan-sft]], [[tulu3-reproduction-tolerance]],
              [[olmo3-think-stage-deltas]], [[phi4-reasoning-recipe-and-gate]],
              [[smollm3-longcontext-merge]]
     figures: figures/recipe-option-selector.html, figures/paired-gate-calculator.html
     revised: 2026-09 (generality revision)
-->

# Chapter 59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate

> **Core insight.** A reproduction is a measurement, and it is interpretable only with a stated tolerance and
> a stated off-target check. The tolerance is knowable from the reports themselves: the five random
> seeds of the Tülu 3 8B SFT recipe span 0.3 points of the 12-benchmark average, while the three 70B seeds
> span 2.6 points, and changing only the chat template moves an intermediate Tülu 3 SFT model by 1.4 points
> ([[tulu3-reproduction-tolerance]], Tables 13-14). The off-target check is necessary because every recipe in
> this chapter moves benchmarks its own target metric does not name: Olmo 3 Think 32B gains 20 points of
> IFBench between its 3.0 and 3.1 RL runs and loses 5 points of AlpacaEval over the same step
> ([[olmo3-think-stage-deltas]], Table 14), and SWE-agent-LM-32B reaches 40.2% on SWE-bench Verified while
> reaching 8.4% on SWE-bench Multilingual against its own base model's 6.5% ([[swe-smith-rft]], Table 3,
> App. F.4). The capstone is one stage, run end to end, with a recipe ledger, a tolerance band, and a gate
> that can fail the run.
>
> **Guideline.** When compute allows only one stage, choose the stage whose released artifacts pin the most
> values, because a value the source never printed cannot be reproduced or refuted: the Tülu 3 8B SFT and DPO
> rows are pinned to launch commands at a named commit ([[open-instruct-allenai-recipes-recipe]]), while the
> DeepSWE blog and its released script disagree on batch size, GPU count, step limit and trajectory timeout
> ([[deepswe-recipe]]). When setting a pass band, derive it from the source's own seed or run variance rather
> than from a round figure, because the reported Olmo 3 per-benchmark standard deviations range from 0.1554
> (PopQA) to 1.4798 (GPQA) over three runs of fourteen models ([[olmo3-think-stage-deltas]], §4.1.1). When a
> stage trains on a single domain, budget for the mixed-domain comparison, because Olmo 3 reports that RL on
> IFEval alone raised IFEval and lowered AlpacaEval while a mixed RL mixture raised both (§4.5). When the run
> is agentic, require a held-out environment family in the gate, because agentic SFT at state of the art on
> the trained distribution moved an untrained language family by 1.9 points ([[swe-smith-rft]], App. F.4).

---

## Why this chapter matters for a general-purpose model

The pipeline this course has followed runs pre-training → mid-training → SFT → preference optimization →
RL → evaluation. [[ch-58a]] laid the six open recipes side by side across that whole pipeline. This capstone
takes one stage out of one of them and runs it, at a scale the reader can afford, under conditions that make
the result interpretable.

Three properties make a reproduction worth the compute for someone building a general-purpose model.

1. **It measures the noise floor of the setup.** Every later decision — how much agent data, which preference
   loss, how many RL steps — is a comparison between two runs, and a difference between two runs cannot be
   read as an effect until the size of a no-change difference is known.
2. **It exposes what the report did not print.** A reproduction attempt forces every missing value into view,
   because the run cannot start without one.
3. **It forces the off-target question.** The stage is trained against one objective. Whether the model that
   comes out is more generally capable is a separate measurement, and every source in this chapter that
   looked outside its target metric found movement there in both directions.

The measurable problem of this capstone: **for one stage of one open recipe, at a stated budget, does a
small-scale re-run land inside a tolerance band derived from the source's own variance, and does its
generality gate — held-out suite, forgetting report, perturbation robustness, paired intervals — pass?**

---

## §1 The six options and what each one can demonstrate

Each option below is a stage that has been published with enough detail to attempt. The
[interactive option selector](figures/recipe-option-selector.html) lets you compare them on four axes at
once: what the source pins, what it leaves unreported, the smallest honest downscale, and the gate slices
that option makes mandatory.

| Option | Stage to reproduce | What the source pins | What it does not print | Anchor number to compare against |
|---|---|---|---|---|
| A. Tülu 3 | SFT then length-normalized DPO on a Llama-3.1 base | Full launch commands at a named commit: SFT LR 5e-6 linear, warmup 0.03, 2 epochs, batch 128 sequences, 4,096 tokens; DPO with `dpo_norm` (the DPO loss on per-token-averaged response log-probabilities) at β = 5, LR 5e-7, 1 epoch, 128 pairs, 2,048 tokens ([[open-instruct-allenai-recipes-recipe]]) | Which RLVR checkpoint step was released; the loss-reduction flag changed between commits | Tülu 3 8B SFT 12-benchmark average 60.1 ([[tulu3-reproduction-tolerance]], Table 9) |
| B. Olmo 3 | Think DPO, or one RL-Zero domain from the base | Per-run scripts with start checkpoint, data mix names and counts, LR, lengths, batch and GPU layout ([[allenai-olmo3-open-instruct-scripts-recipe]]) | Several script values conflict with the paper tables (32B Think RL: 64 prompts per step in the script, 128 in Table 49) | Per-stage columns of Table 14 / Table 15 ([[olmo3-think-stage-deltas]]) |
| C. SmolLM3 | Context extension 4k → 32k → 64k, then the repair merge | Two 50B-token stages, RoPE base frequency (theta) 1.5M then 5M, YaRN position-interpolation at inference to 128k, linear merge at 0.9 / 0.1 ([[smollm3-longcontext-merge]]) | No RULER scores before and after the merge; no LR or batch in the blog text | RULER at 64k, recovered to the base model's score after the merge (qualitative) |
| D. Phi-4-reasoning | Distill-SFT on long reasoning traces, then a 90-step GRPO | SFT 1.4M pairs / 8.3B unique tokens / 16B trained tokens, batch 32, 32K context, LR 1e-5, warmup 450 steps; GRPO batch 64 on 32 H100, LR 5e-8, G = 8, β = 0.001, entropy 0.001 ([[phi4-reasoning-recipe-and-gate]]) | The prompt-filter thresholds; the 50 synthetic-data categories of the Phi-4 base | Table 2 general-purpose row set, and "more than 10%" AIME gain from the 90 GRPO steps |
| E. R1-distill | SFT-only distillation of a large reasoner into a small base | Student bases and the ~800K-sample SFT set; SFT data creation cost 5K of 147K total GPU hours ([[deepseek-r1]], Table 7, Supp. B.4.3) | Per-student batch sizes and schedules beyond the bases and learning rates in Table 6 | Distill-Qwen-32B AIME 2024 72.6 / MATH-500 94.3 / GPQA 62.1 / LiveCodeBench 57.2 against RL-from-base 47.0 / 91.6 / 55.0 / 40.2 (Table 16) |
| F. Agentic | DeepSWE RL, SWE-smith SFT, or Toucan SFT plus a small multi-turn RL | SWE-smith: full-parameter SFT, LR 5e-5, ≤ 3 epochs, 32,768 context, 2-8 H100 ([[swe-smith-rft]]); Toucan: LR 2e-5, 2 epochs, effective batch 64, 32,768 context ([[toucan-sft]]) | DeepSWE: LR, clip values and step limits appear only in the script and conflict with the blog ([[deepswe-recipe]]) | SWE-agent-LM-32B 40.2% SWE-bench Verified; Toucan Table 2 per-slice deltas |

**Selection rule.** When the objective is to calibrate a setup, choose A or F, because both have a published
number produced by a command you can read line by line. When the objective is to study a mechanism — what a
stage adds and what it removes — choose B or D, because both publish the intermediate checkpoints on either
side of the stage. When the budget is under a few hundred GPU-hours, choose E or the Toucan variant of F,
because both are single SFT passes whose data is released.

---

## §2 The plan memo

The plan memo is written before any training job is launched and is not revised afterwards; revisions go in
the comparative memo of §8 with their reasons. It has six parts.

**1. Budget, in the units the run consumes.** For an RL stage the consumed unit is generated tokens, not
optimizer steps. Olmo 3 measured its own RL throughput on two 8×A100 nodes, one for training and one for
inference: 881 tokens/second with the OLMo 2 stack and 2,949 tokens/second after continuous batching, better
threading, and in-flight weight updates ([[olmo3-think-stage-deltas]], Table 23).

> *Worked example.* Plan an RL stage of 64 prompts × 8 samples per step with an average generation of 4,000
> tokens. Tokens per step = 64 × 8 × 4,000 = 2,048,000. At 2,949 tokens/second one step takes
> 2,048,000 / 2,949 = 694 seconds, so 200 steps take 138,800 seconds = 38.6 hours on those 16 GPUs. At 881
> tokens/second the same 200 steps take 2,325 seconds per step and 465,000 seconds = 129.1 hours. The rollout
> stack changes the budget by a factor of 3.3 before any hyper-parameter is chosen.
>
> *Conditions and limits.* Table 23 is a two-hour benchmark on two 8×A100 nodes, one training and one
> inference, started from a reasoning SFT or DPO checkpoint whose average generation lengths the report
> describes as extremely long; the report does not print the response-length cap used for that benchmark.
> A run with 4,000-token responses on other hardware has a different rate, so the arithmetic above gives the
> form of the estimate only, and the rate must be re-measured on the cluster that will be used.

**2. Evaluation budget, stated separately.** Olmo 3 reports that during recipe development on 7B models,
"evaluation costs between 10 and 20% of our compute budget" ([[olmo3-think-stage-deltas]], §4.1.1). A memo
that allocates zero compute to evaluation produces a run with no gate.

**3. Stages.** Name the starting checkpoint by its public identifier, the stage, and the stopping rule, and
separate the planned schedule from the stop point: the Tülu 3 8B RLVR command passes
`total_episodes 10,000,000` while the paper's Table 21 gives 100,000 and the released model is "an earlier
than final checkpoint" ([[open-instruct-allenai-recipes-recipe]]).

**4. Recipe ledger with lab values.** The table in the [Recipe](#recipe) section below, with **Lab value** and
**Reason for difference** filled in before the run. A blank reason is a defect.

**5. Failure-mode checklist.** The failure each stage is known to produce and the signal that detects it, from
[Common mistakes](#common-mistakes-and-how-to-detect-them).

**6. Go / no-go criteria, written as numbers.** The reproduced anchor must land inside the §3 band; every
slice in §4 must clear its stated margin with a paired interval; and a stated wall-clock and GPU-hour ceiling
stops the run, which is then reported as incomplete rather than extended.

---

## §3 Reproduction targets: what counts as the same number

**Definition.** The *reproduction band* is the interval around a published number inside which a re-run is
treated as agreeing with it. It is derived from variance the source itself measured, not chosen.

**The problem it addresses.** Published numbers are single draws. Tülu 3 trained its 8B SFT recipe under five
random seeds and reports 59.9, 60.1, 59.8, 59.8, 59.8 on the 12-benchmark average, and released the 60.1 run
([[tulu3-reproduction-tolerance]], Table 14). A reproduction landing at 59.7 differs from the published 60.1
by more than the entire seed range and is still inside the distribution the recipe produces. The same table's
three 70B seeds span 71.8, 70.0, 72.6 — a 2.6-point range against 0.3 at 8B, from the same pipeline. Band
width is therefore a property of the scale and the benchmark as much as of the recipe (Interpretation).

**Mechanism, step by step.**
1. Collect every source-side variance estimate for the anchor metric: seed repeats, run-to-run standard
   deviations, evaluation-sampling variance.
2. Add the variance the re-run introduces and the source did not have: a different base checkpoint revision,
   chat template, or loss reduction.
3. Set the band to cover the union. A template change alone moved an intermediate Tülu 3 SFT model across a
   1.4-point range (Table 13), so a band narrower than 1.4 points is not defensible for a re-run that did not
   pin the template byte for byte.
4. State the band in the plan memo before the run.

**Per-benchmark bands.** A single band over a multi-task average hides that the tasks differ by an order of
magnitude in noise. Olmo 3's per-benchmark standard deviations over three runs of fourteen models are 1.4798
for GPQA, 1.2406 for AlpacaEval 3 and 0.8835 for IFEval at the top, and 0.2219 for MMLU and 0.1554 for PopQA
at the bottom ([[olmo3-think-stage-deltas]], §4.1.1). A 2-point GPQA difference is inside that suite's noise;
a 2-point PopQA difference is not. [[signal-and-noise-eval]] reports the same effect as a selection
criterion: across the tasks of the OLMES evaluation suite a benchmark's signal-to-noise ratio at small scale
correlates with its decision accuracy for predicting 1B rankings (R = 0.791, R² = 0.626, §4.1) while signal
or noise alone does not, and switching the metric to bits-per-byte raised the 30-task average SNR from 10.0 to 31.5 and decision
accuracy from 77.0% to 83.7% (Fig. 6). **Result (single study).**

**Implication for a general-purpose model.** The band decides which stage-level claims survive. A capstone
reporting "reproduced within 0.5 points" on a benchmark whose measured standard deviation is 1.48 has
reported nothing.

---

## §4 The generality gate

The gate is four measurements and one decision rule, taken from [[ch-53]] and [[ch-51a]] and applied to the
reproduced stage.

**(a) Held-out suite.** Tasks from families the stage never trained on, fixed before the run. For an agentic
stage this must be a held-out environment family, not held-out instances of the trained family: SWE-smith's
model, trained to state of the art on Python SWE-bench instances, scored 8.4% on 300 non-Python tasks against
its own base model's 6.5% and Claude 3.7 Sonnet's 43% ([[swe-smith-rft]], App. F.4).

**(b) Forgetting report.** Every benchmark the starting checkpoint was strong on, re-run on the trained
checkpoint. Olmo 3 Think 32B shows why this is not optional: DPO raised AlpacaEval 2 LC from 69.1 to 78.6 and
lowered IFEval from 83.9 to 80.6; the following RL stage raised IFEval to 89.0 and lowered AlpacaEval to 74.2;
extended RL in the 3.1 run raised IFBench from 47.6 to 68.1 and returned AlpacaEval to 69.1, the SFT level
([[olmo3-think-stage-deltas]], Table 14). Each trade is visible only because every column was re-run at every
stage.

**(c) Perturbation robustness.** The same tasks under a changed surface: reordered options, renamed tools, a
different system prompt, a paraphrased instruction. [[swe-bench-illusion]] reports that ten OpenAI and
Anthropic models named a file changed by the gold patch in 60-76% of SWE-Bench Verified instances given only
the repository name and issue text, against under 53% on 245 SWE-Bench-style tasks from repositories outside
the benchmark (§4.1). [[agentic-benchmark-checklist]] reports that an agent returning nothing passes 38% of
τ-bench Airline tasks and an agent that overwrites SWE-Lancer's test files scores 100% (§1, App. E.2, E.4);
running a do-nothing agent and an enumerating agent against the grader belongs to this measurement.

**(d) Paired confidence intervals.** Both checkpoints are evaluated on the same task instances with the same
seeds, and the interval is computed on the per-task difference.

**Formula.** For binary per-task outcomes with one trial per task, let *n* be the number of tasks, *b* the
proportion of tasks the reproduced model solves and the reference does not, and *c* the proportion where the
reverse holds. The 95% interval half-width for the accuracy difference is

    h_paired = 1.96 · sqrt((b + c) / n)

where *b + c* is the discordant proportion — the share of tasks on which the two models disagree. The
unpaired half-width for the same two accuracies *p* is

    h_unpaired = 1.96 · sqrt(2 · p · (1 − p) / n)

where *p* is the accuracy of either model under the null that they are equal.

> *Worked example.* A held-out suite of n = 200 tasks, both models near p = 0.40, 20% of tasks discordant
> (b + c = 0.20). Paired: 1.96 · sqrt(0.20 / 200) = 1.96 · 0.03162 = 0.062, a half-width of 6.2 points.
> Unpaired: 1.96 · sqrt(2 · 0.40 · 0.60 / 200) = 1.96 · sqrt(0.0024) = 1.96 · 0.04899 = 0.096, a half-width of
> 9.6 points. Pairing cuts the minimum detectable effect from 9.6 to 6.2 points at the same cost. If the
> expected stage effect is 4 points, neither design detects it at n = 200, and the plan memo has to raise *n*,
> raise trials per task, or stop claiming the effect.

The [paired-interval calculator](figures/paired-gate-calculator.html) recomputes both half-widths as *n*, the
discordant share, the base accuracy and the trials per task change, and marks where a stated target effect
becomes detectable.

**Decision rule.** The gate passes when the held-out suite, every forgetting-report column, and the perturbed
score each fall by no more than their stated margins, and each statement is backed by a paired interval that
excludes its margin. Any one failure fails the run. A failed gate is a result and is reported as one.

---

## §5 Long context in the capstone

Two of the six options make context length part of the stage, and both report that the context work and the
rest of post-training interfere.

**SmolLM3 (option C).** After 11.2T tokens of three-stage pretraining, the context window was extended in two
50B-token stages, 4k → 32k with the RoPE base frequency (theta) raised to 1.5M and 32k → 64k with theta
raised to 5M, with YaRN position interpolation at inference to reach 128k ([[smollm3-longcontext-merge]]). The reported ablation is negative: upsampling code
repositories, books and long web pages beyond the naturally long samples in the mixture "didn't further boost
performance on RULER and HELMET benchmarks". The regression appears later — after reasoning mid-training and
APO (Anchored Preference Optimization), RULER degraded, and the blog traces it to the reasoning mid-training
stage plus an APO dataset capped at 24k tokens. The repair is a linear merge of the APO model soup with a mid-training checkpoint at weights 0.9
and 0.1, which recovered the base model's RULER score up to 128k. The blog does not print the RULER numbers on
either side of the merge. **Result (single study), effect size not reported.**

**Olmo 3 (option B).** The long-context stage trains on Dolma 3 Longmino Mix, 50B tokens for the 7B model and
100B for the 32B, drawn from a 639B-token pool ([[olmo-3]], §3.6, Table 11). The 7B Think SFT run starts from
a long-context checkpoint at step 11921, so any reproduction of that stage inherits whatever the context
extension left behind ([[allenai-olmo3-open-instruct-scripts-recipe]]).

**What the capstone must do.** Any option whose stage touches context length, and any option that trains an
extended model on short sequences, adds two gate rows: a short-context regression check and a long-context
check at the trained length. [[ruler]] supplies the synthetic length-controlled slice and [[prolong]] the
natural-document slice. When the long-context check fails after a successful short-context stage, a merge with
the pre-stage long-context checkpoint is the documented repair ([[smollm3-longcontext-merge]]);
[[ties-merging]] gives the trim-and-sign-elect variant and reports that merging does not match multitask
training (73.1 against 66.4 for (IA)³, Table 5), so the merge is a repair, not an upgrade.

---

## §6 The agentic option in detail

**Three sub-options, ordered by cost.**

1. **Toucan SFT (cheapest).** A single supervised pass: 119.3K selected instances, LR 2e-5, 2 epochs,
   effective batch 64, max sequence 32,768, AdamW with β = (0.9, 0.999), DeepSpeed ZeRO-3, on 8 or 64 H100s
   ([[toucan-sft]], Table 5). Its published per-slice deltas are a ready-made gate exercise, because the
   overall number and the slice numbers move in opposite directions at small scale. BFCL V3 (Berkeley Function
   Calling Leaderboard v3) scores several slices separately, among them non-live AST, which matches a
   generated call against a reference by abstract-syntax-tree comparison on curated prompts, and relevance,
   which scores whether the model calls a tool at all when one applies. At 7B the BFCL V3 overall
   rises 3.16 points while the non-live AST slice falls 5.67 and the relevance slice falls 5.55; at 14B
   relevance falls 11.11 while overall rises 7.40; only at 32B does every column rise (Table 2). The τ²-bench
   Telecom column falls at 7B (16.70% → 10.50%) and at 32B (21.11% → 20.20%) while both averages rise
   (Table 3). **Result (single study).**
2. **SWE-smith rejection-sampling SFT.** Expert trajectories from Claude 3.7 Sonnet inside SWE-agent at a
   75-step and $2.00 limit; only trajectories that resolved the instance are trained on; full-parameter
   fine-tuning with torchtune at LR 5e-5, at most 3 epochs, 32,768 context, on 2-8 H100s ([[swe-smith-rft]],
   App. F.1). Two design facts transfer to a small run: repeated easy trajectories were capped at 3 per task
   instance because they "degrade model performance" (§4), and the filtered-against-unfiltered ablation at
   100 to 1,600 training points gives 14.3/22.4/27.8/30.1/33.4 percent resolved against
   10.2/19.7/18.3/23.4/27.8 (Fig. 23).
3. **Multi-turn RL (most expensive).** DeepSWE trained Qwen3-32B with RL only on 4.5K R2E-Gym problems for
   six days on 64 H100 GPUs, reaching 42.2% Pass@1 on SWE-Bench-Verified ([[deepswe]], intro, §4). The
   reproduction hazard is in its own ledger: the blog and the released script disagree on batch size (64 × 8
   against `train_batch_size=8`), GPU count (64 against 16), trajectory timeout (20 minutes against 5,400
   seconds) and step limit ([[deepswe-recipe]]). This option requires recording which of the two sources each
   lab value follows, row by row.

**The trajectory-ending decision.** A trajectory can end by success, failure, context limit, step limit, or
wall-clock timeout. DeepSWE's compact filtering masks the loss of trajectories that end by context, step, or a
20-minute generation timeout; the stated reason is that an agent can pass all tests by chance and that
rewarding such trajectories leads to collapse ([[deepswe]], §2.3, Fig. 6 — curve only, no number). The
decision changes the gradient of every other rollout in the group, because the group baseline is computed over
the surviving members.

> *Worked example (derived; the post gives no formula).* With a leave-one-out baseline and no standard-
> deviation normalization, A_i = r_i − (1/(n−1)) Σ_{j≠i} r_j, where A_i is the advantage of rollout i, r_i is
> its reward in {0, 1}, and n is the number of rollouts for that problem. With n = 8 and 2 successes, a
> failure gets 0 − 2/7 ≈ −0.29 and a success gets 1 − 1/7 ≈ 0.86. If two of the six failures are masked
> instead, n = 6 with 2 successes: a failure gets 0 − 2/5 = −0.40 and a success gets 1 − 1/5 = 0.80. The same
> rollouts produce different gradients depending on a configuration flag ([[deepswe-recipe]], masked-
> trajectories row).

**Required gate slices for option F.** A held-out environment family; a tool-calling slice; an abstention
check on prompts whose required tool is absent; and two non-agentic suites. The abstention check is required
because [[reasoning-trap-tool-hallucination]] reports that think-then-act GRPO on tool data raised
Qwen2.5-7B-Instruct's no-tool-available hallucination rate from 34.8% to 90.2% and its distractor-tool rate
from 54.7% to 100.0%, while BFCL Multi-Turn rose from 13.6 to 23.5 and IFEval moved from 62.4 to 59.8
(Tables 2-3) — the tool-calling and instruction-following benchmarks did not reveal the change. The
mixture-share axis is required because [[agenttuning]] reports that at 7B, agent-only SFT scored 0.09 on
held-out agent tasks and 0.22 on general tasks against 0.67 and 0.63 for the same trajectories mixed at
η = 0.2 with ShareGPT (Table 5). Cost and scaffold are reported next to accuracy, because
[[holistic-agent-leaderboard]] found two Online Mind2Web agents differing ninefold in cost at a two-point
accuracy difference, and a leaked few-shot file that invalidated one scaffold's results (§4.1, App. A5).

---

## §7 The distillation option in detail

**Definition.** Distillation here means SFT of a student on sequences produced by a stronger teacher, scored
or filtered by a verifier where one exists. Four of the six options use it, in different roles.

| Recipe | Teacher | What is distilled | Locus |
|---|---|---|---|
| R1-distill | DeepSeek-R1 | ~800K SFT samples into six students (Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B, Llama-3.1-8B, Llama-3.3-70B-Instruct) | [[deepseek-r1]], Supp. B.4.3, Table 6 |
| Phi-4-reasoning | o3-mini (high thinking, 32K) | 1.4M prompt-response pairs, 8.3B unique tokens | [[phi4-reasoning-recipe-and-gate]], §3 |
| SmolLM3 | Qwen3-32B in thinking mode; Qwen3-0.6B for rejected responses | 0.8B reasoning-mode SFT tokens; APO preference pairs | [[smollm3-longcontext-merge]] |
| SWE-smith | Claude 3.7 Sonnet inside SWE-agent | 5,016 resolved trajectories | [[swe-smith-rft]], §3-4 |

**The comparison that makes option E worth running.** DeepSeek-R1 ran distillation and RL from the same base
at the same size. From Qwen2.5-32B-Base, RL for over 10K steps reached AIME 2024 47.0, MATH-500 91.6, GPQA
55.0, LiveCodeBench 40.2; SFT-only distillation of the ~800K set into the same base reached 72.6, 94.3, 62.1,
57.2 ([[deepseek-r1]], Table 16, Supp. F.1). The authors conclude that distillation is economical but that
exceeding current limits may still require stronger bases and larger-scale RL (Interpretation).

**Cost.** R1 puts SFT-data creation at 5K GPU-hours against 101K for R1-Zero and 41K for R1, a total of 147K
([[deepseek-r1]], Supp. B.4.4, Table 7) — data generation was 3.4% of the total. In a capstone the ratio is
usually worse, because the teacher is a paid API rather than an in-house checkpoint, so the plan memo carries
teacher-token cost as its own line.

**The generality caution.** [[rlvr-beyond-base-model]] reports that RLVR models have higher pass@1 than their
base models while base models match or exceed them at large k on the benchmarks tested, and interprets this
as increased probability of paths the base could already sample. A distilled student raises the mirror-image
question: whether the gain is coverage the student did not have, or concentration on the teacher's preferred
path. The gate answers it with pass@k at a large k alongside pass@1; SWE-smith reports pass@1 40.2 rising to
54.8 at k = 6 on SWE-bench Verified ([[swe-smith-rft]], Fig. 22).

---

## §8 The ablation table and the comparative memo

The comparative memo is the deliverable. It has four parts.

**1. Reproduced against reported.** One row per anchor metric: reported value with its locus, reproduced
value, difference, band from §3, and inside/outside.

**2. Explained deltas.** Every difference outside the band gets a named cause and its evidence, drawn from
the ledger's "Reason for difference" column. A difference with no named cause is recorded as unexplained
rather than attributed.

**3. Ablation table.** At least two rows besides the main run, each pair isolating one decision. Useful
pairs, each with a published counterpart:

| Ablation | Published counterpart |
|---|---|
| Single-domain stage against mixed-domain stage | Olmo 3: RL on IFEval alone raised IFEval and lowered AlpacaEval; the mixed run held both ([[olmo3-think-stage-deltas]], §4.5, Fig. 20 left) |
| RL after SFT against RL after DPO | Olmo 3 7B: "our RL framework yields greater improvements when applied after contrastive learning with DPO rather than directly following SFT" (§4 contributions list; the comparison itself is Table 22 and Fig. 19 in §4.5) |
| Filtered against unfiltered training trajectories | SWE-smith Fig. 23: 33.4% against 27.8% resolved at 1,600 training points ([[swe-smith-rft]]) |
| Agent-data share η in the SFT mixture | [[agenttuning]] Table 5 at 7B: 0.09 / 0.22 for agent-only against 0.67 / 0.63 at η = 0.2 |
| Merge against mixed training for two objectives | [[mix-data-or-merge-models]] Table 1: SLERP merge 78.0% win-rate and −57.8% relative harm against 71.0% and −54.7% for the 15% safety mix |

**4. What could not be reproduced and why.** Every `not reported` row in the ledger that blocked an exact
re-run, and every `conflict` row where a choice had to be made. This section is what lets a later reader
repeat the run without rediscovering the same gaps.

---

## Negative samples and negative feedback

Which meaning of "negative" is in play depends on the option, and the capstone must name it per stage
([[ch-43a]] holds the derivations for negative gradients; [[ch-31a]] for negatives used as content).

**Where negatives come from.** Option A's DPO stage takes rejected responses from a preference mixture.
Option B's RL stage takes them from per-domain verifiers: math and code checkers, an instruction-following
verifier, and a Qwen3-32B judge for general prompts ([[allenai-olmo3-open-instruct-scripts]]). Option D's
GRPO takes them from a rule-based math checker ([[phi4-reasoning-recipe-and-gate]], §4). Option E discards
them: R1's rejection-sampling SFT keeps only correct responses out of about 600K reasoning samples
([[deepseek-r1]], Supp. B.3.3). Option F takes them from test execution ([[swe-smith-rft]]; [[deepswe]]).
None of these sources reports a false-negative rate for its own verifier.

**What current practice does with them.** Three of the four meanings appear across the options:
- **Negative marginal value** (a sample that lowers performance as a positive target): SWE-smith caps
  repeated easy trajectories at 3 per instance because repeatedly-solved instances "degrade model
  performance" ([[swe-smith-rft]], §4); its filtered-against-unfiltered ablation is the measurement.
- **Negative as gradient through a preference loss**: `dpo_norm` lowers the length-normalized log-ratio of
  the rejected response ([[allenai-olmo3-open-instruct-scripts]], `dpo_utils.py` L83-84, L556-559).
- **Negative as gradient through an advantage**: GRPO gives every below-group-mean sample A < 0. Under
  Olmo 3's `centered` normalization the advantage is the raw reward gap, not divided by the group standard
  deviation; the card's example is a verifiable reward of 10.0 for 1 of 8 samples giving A = 8.75 and −1.25,
  against 2.646 and −0.378 under `standard` (`data_loader.py` L421, L1417-1420).

**Mechanism.** For a softmax over logits z with target y, ∂ log p_y / ∂ z_j = 1[j = y] − p_j, where p_j is the
model's probability of token j. A negative-gradient update pushes down the logits of the tokens in the sampled
sequence; the removed probability mass is redistributed over the remaining tokens in proportion to their
current probabilities, so the largest share goes to the currently most likely alternative. When the
pushed-down sequence already had low probability, the redistribution concentrates on one or two alternatives
that were never evaluated.

**Controls the sources use.** Bound the set that receives negative gradient: Olmo 3 drops groups whose reward
standard deviation is zero, so all-correct and all-wrong groups contribute nothing, and excludes prompts
solved at a rate of at least 0.875 of the maximum from later sampling in two of its scripts
(`data_loader.py` L419 and L814 for the zero-variance drop, L809-812 for the pass-rate exclusion). Mask rather than penalize uncertain failures: DeepSWE's compact filtering ([[deepswe]], §2.3).
Bound the ratio: the Olmo 3 default clamps a train/inference ratio at 2.0 and multiplies the loss by it
(`grpo_utils.py` L104-113, L400-415). Anchor with a positive term: option E's rejection-sampling SFT is a
pure positive-NLL stage by construction.

**Diagnostics the capstone must log.** Chosen and rejected log-probabilities separately for any preference
stage; per-step counts split by advantage sign for any RL stage; the share of groups dropped for zero
variance; the share of trajectories masked and the reason for each mask; policy entropy; pass@1 and pass@k at
a large k; and abstention rate on the no-tool-available slice for option F.

**Effect on generality.** Coverage: [[rlvr-beyond-base-model]] reports base models matching or exceeding RLVR
models at large k on the benchmarks tested. Hallucination and over-refusal: the abstention measurement in
[[reasoning-trap-tool-hallucination]] moved 55.4 points toward fabricating tools while the target benchmarks
improved. Over-optimization: Olmo 3's single-domain RL raised the trained domain and lowered an untrained one,
and the mixed run produced lower train reward with equal or better downstream scores
([[olmo3-think-stage-deltas]], §4.5, Figs. 20-21).

**Size of effect.** No source in this chapter's set measures the share of the stage's improvement
attributable to the negative-gradient term as opposed to the positive term. Claims of the form "the negatives
drove the gain" are therefore not supported here. **Open question.**

---

## Recipe

Rows are the settings a capstone has to fill in before launching. "Lab value" and "Reason for difference" are
written by the reader in the plan memo; the examples given are illustrative of the format, and the reason
column is what makes a row auditable. All source values below are carried from a `verified` row of the linked
ledger card.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR, schedule, warmup ratio, weight decay | 5e-06, linear, 0.03, 0.0 | `docs/tulu3.md` L39-42 @ open-instruct 098424c | verified 2026-09-14 | paper Table 11; §4.3.2 sum loss with 5e-6 best in mix sweep | (fill) | (fill) |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | epochs; max length; effective batch (sequences) | 2; 4,096 tokens; 128 = 64 GPUs × 1 × 2 | `docs/tulu3.md` L35, L43, L18-21, L65 | verified 2026-09-14 | §4.3.2: "training for longer did not yield further improvements" | (fill) | e.g. 8 GPUs × 1 × 16 to hold batch 128 at lower GPU count |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss reduction | `--reduce_loss sum` at commit 8781471; flag absent at 098424c, default mean since PR #1024 | `docs/tulu3.md`@8781471 L49; commit bb98dbc | conflict | §4.3.2: sum loss chosen over mean | (fill) | must be stated; the two commits are different recipes |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | loss; β; peak LR; epochs; batch; max length | `dpo_norm`; 5; 5e-07 linear warmup 0.1; 1; 128 pairs; 2,048 | `docs/tulu3.md` L134-159 | verified 2026-09-14 | paper §5.4.1 Table 18; Table 20 | (fill) | (fill) |
| Llama-3.1-Tulu-3-8B | 8B | RL | algorithm; LR; KL β; batch; no-EOS reward | PPO; 3e-7; 0.05; 7 actor GPUs × 32 = 224 responses; −10.0 | `docs/tulu3.md` L313-346 | verified 2026-09-14 | paper Table 21 (224; β = 0.05 for the final 8B) | (fill) | (fill) |
| Llama-3.1-Tulu-3-8B | 8B | RL | total episodes | 10,000,000 (planned) | `docs/tulu3.md` L329 | conflict | paper Table 21: 100,000; §6.3: "an earlier than final checkpoint" released | (fill) | record schedule and stop point separately |
| Olmo 3 7B Think DPO | 7B | preference | pairs; LR and schedule; max length; batch | 150,000; 8e-8 linear, weight decay 0.0; 16,384; 128 | `7b_think_dpo.sh` L3-40 @ open-instruct d8a7f1c | verified 2026-09-14 | matches paper Table 48; A.6.2: LR and dataset size were swept | (fill) | (fill) |
| Olmo 3 7B Think (released RL run) | 7B | RL | unique prompts × samples; minibatches; LR; prompt / response / pack length; temperature | 64 × 8; 1; 1e-6 constant; 2,048 / 32,768 / 35,840; 1.0 | `7b_think_rl_no_pipeline.sh` L20-49 | verified 2026-09-14 | matches paper Table 49 | (fill) | (fill) |
| Olmo 3 (all RL runs) | 7B, 32B | RL | clip lower / upper; loss; advantage normalization; ρ cap | 0.2 / 0.272; `dapo`; `centered`; 2.0 (defaults at d8a7f1c) | `grpo_utils.py` L100-112, L139; `data_loader.py` L413, L421 | conflict | scripts that printed no flag before PR #1547 ran with different defaults (0.2 / 0.2, `standard`, ρ off) | (fill) | pin the framework commit, not the flag list |
| SmolLM3-3B | 3B | long-context | stages; tokens; RoPE theta | 4k → 32k then 32k → 64k; 50B tokens each; theta 1.5M then 5M | HF blog, "Long context extension" | verified 2026-09-17 | ablation: upsampling extra long-context data gave no further RULER/HELMET gain | (fill) | (fill) |
| SmolLM3-3B | 3B | merge | method and weights | linear merge, APO model soup 0.9 / mid-training checkpoint 0.1 | HF blog, "Model merging" | verified 2026-09-17 | stated to recover the base model's RULER score to 128k; scores not printed | (fill) | (fill) |
| Phi-4-reasoning | 14B | distill-SFT | pairs; unique tokens; trained tokens; steps; global batch; context; LR; warmup; weight decay | 1.4M; 8.3B; 16B; ~16K; 32; 32K; 1e-5; 450 steps linear; 1e-4 | Phi-4-reasoning report §3 | verified 2026-09-17 | §3: grid search over [1e-6, 2e-5]; 1e-5 chosen | (fill) | (fill) |
| Phi-4-reasoning-plus | 14B | RL | algorithm; global batch; GPUs; LR; group size; KL β; entropy coeff; steps; max length | GRPO (verl); 64; 32 H100; 5e-8 cosine, 10-step warmup; G = 8; 0.001; 0.001; 90; 32K with 31K output clip | Phi-4-reasoning report §4, §4.2 | verified 2026-09-17 | §4.2: checkpoint chosen by best AIME 2024; further steps gave no additional gains | (fill) | (fill) |
| SWE-agent-LM-32B | 32B | distill-SFT | trajectories; LR; epochs; context; GPUs; selection | 5,016 resolved trajectories (≤ 3 per task instance); 5e-5; ≤ 3; 32,768; 2-8 H100; torchtune full-parameter | SWE-smith App. F.1, §4 | verified 2026-09-17 | Fig. 23: rejection sampling beats unfiltered at 100-1,600 points | (fill) | (fill) |
| Toucan-tuned Qwen2.5-Instruct | 7B/14B/32B | SFT | instances; LR; epochs; effective batch; context; optimizer | 119.3K; 2e-5; 2; 64; 32,768; AdamW β = (0.9, 0.999), ε = 1e-8, ZeRO-3 | Toucan App. C.2 Table 5 | verified 2026-09-17 | Table 6 ablation of the three extensions on BFCL V3 | (fill) | (fill) |
| DeepSWE-Preview | 32B | RL | problems per iteration × trajectories; compute; timeout | blog: 64 × 8, 64 H100 for six days, 20-minute generation timeout / script: `train_batch_size=8`, 16 GPUs, `trajectory_timeout=5400` | Blog §2.2-2.3 / `scripts/agent/swe/deepswe_32b.sh` @ rllm 709dec4 | conflict | no ablation reported for either | (fill) | state which source each lab value follows |

**Starting point for a small general-purpose run.** Option A at reduced scale is the only configuration in
this table whose every value comes from `verified` rows of one pinned commit: SFT on a Llama-3.1 base at LR
5e-6 linear with warmup ratio 0.03 for 2 epochs at an effective batch of 128 sequences of up to 4,096 tokens,
followed by `dpo_norm` DPO at β = 5 and LR 5e-7 for 1 epoch at 128 pairs of up to 2,048 tokens
([[open-instruct-allenai-recipes-recipe]]). The conditions attached to those values: the SFT learning rate
was selected under a sum loss reduction that the pinned commit no longer applies, the doc's SFT run used 64
H100 GPUs while the paper states 32 GPUs for 6 hours, and the values are for Llama 3.1 bases — the OLMo 2
report states that OLMo 2 bases needed higher learning rates than the Llama 3.1 recipe ([[olmo-2]], §5).

---

## Generalization lens

**(a) What increases breadth.** Each item is a **Result (single study)**.
- **Mixing domains inside the stage.** Olmo 3: single-domain RL raised the trained domain and lowered an
  untrained one, the mixed mixture improved across domains, and mixing gave lower train reward with equal or
  better downstream scores ([[olmo3-think-stage-deltas]], §4.5, Figs. 20-21).
- **Mixing agent data with general instruction data.** At 7B, agent trajectories at η = 0.2 against ShareGPT
  scored 0.67 held-out agent and 0.63 general, against 0.09 and 0.22 for agent-only data ([[agenttuning]],
  Table 5).
- **Ordering stages so RL follows a preference stage.** At 7B, Olmo 3's RL gains were larger after DPO than
  directly after SFT, and the report states that the SFT model never overtook the DPO model after RL
  ([[olmo3-think-stage-deltas]], §4.5, Table 22, Fig. 19).
- **Selecting training sequences by an execution verifier.** SWE-smith: 33.4% against 27.8% resolved at 1,600
  training points ([[swe-smith-rft]], Fig. 23).

**(b) What causes narrowing or forgetting.**
- **Training one benchmark's objective.** IFEval-only RL raised IFEval and lowered AlpacaEval in Olmo 3
  (§4.5); extended RL between Think 3.0 and 3.1 raised IFBench 20 points and lowered AlpacaEval 5 (Table 14).
- **Training one language or one environment family.** SWE-agent-LM-32B: 40.2% SWE-bench Verified against
  8.4% SWE-bench Multilingual, with its base at 6.5% ([[swe-smith-rft]], App. F.4).
- **Training tool use without an abstention target.** No-tool-available hallucination 34.8% → 90.2% on
  Qwen2.5-7B-Instruct ([[reasoning-trap-tool-hallucination]], Table 2).
- **Mid-training that competes with an earlier stage.** SmolLM3's reasoning mid-training degraded RULER and
  the repair was a merge, not more training ([[smollm3-longcontext-merge]]).
- **Slice-level losses hidden by an average.** Toucan at 7B: overall +3.16 with non-live AST −5.67 and
  relevance −5.55 ([[toucan-sft]], Table 2).

**(c) How to measure it for this stage.** The four-part gate of §4, plus three reproduction-specific
additions: per-benchmark bands from the source's own measured variance rather than one band for the average
([[olmo3-think-stage-deltas]], §4.1.1; [[signal-and-noise-eval]]); pass@k at a large k alongside pass@1,
because concentration and coverage move differently ([[rlvr-beyond-base-model]]; [[swe-smith-rft]], Fig. 22);
and a contamination and triviality check on the anchor benchmark before any number is reported, using the
repository-level leakage indicators of [[swe-bench-illusion]] and the trivial agents of
[[agentic-benchmark-checklist]].

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing against the released number without a band | A 0.4-point difference is reported as a failed reproduction | Compute the band from the source's seed or run variance first ([[tulu3-reproduction-tolerance]], Table 14) |
| One band applied to every benchmark | A GPQA difference of 2 points is called a regression | Use per-benchmark standard deviations ([[olmo3-think-stage-deltas]], §4.1.1) |
| Chat template not pinned to the source's bytes | Loss curve looks normal; instruction-following and stop-token behaviour shift | Diff the rendered template against the source's; Tülu 3 measured a 1.4-point spread across templates (Table 13) |
| Copying a value from a paper table and a config that disagree | Two rows of the ledger cannot both be true | Record paper value, released-config value and framework default as separate facts ([[deepswe-recipe]]; [[allenai-olmo3-open-instruct-scripts-recipe]]) |
| Treating the planned schedule as the run | `total_episodes 10,000,000` in the memo, 100,000 in the paper | Record schedule horizon, stop point and released checkpoint in separate rows ([[open-instruct-allenai-recipes-recipe]]) |
| Framework defaults changed between commits | The same script produces different clip bounds or advantage normalization | Pin the framework commit; PR #1547 moved β, loss type, clip and normalization from flags into defaults ([[allenai-olmo3-open-instruct-scripts]]) |
| Evaluation budget not allocated | The run finishes and the gate cannot be run | Allocate the 10-20% Olmo 3 measured for development evaluation ([[olmo3-think-stage-deltas]], §4.1.1) |
| Unpaired comparison on a small held-out suite | Interval too wide to support the claim | Pair the evaluation; §4's example moves the half-width from 9.6 to 6.2 points at n = 200 |
| Averaging over slices that move in opposite directions | Overall score rises, deployment behaviour worsens | Report every BFCL-style slice separately ([[toucan-sft]], Table 2) |
| Held-out set drawn from the trained family | Held-out score tracks the training score exactly | Hold out an environment family or a language, not instances ([[swe-smith-rft]], App. F.4) |
| Grader accepts trivial agents | Scores look plausible and are not | Run a do-nothing agent and an answer-enumerating agent ([[agentic-benchmark-checklist]], §5.2) |
| Trajectory-ending policy left at the framework default | Advantages change without any recipe change | Log the share of masked trajectories per step and the reason for each mask ([[deepswe]], §2.3) |
| Accuracy reported without cost or scaffold | Two results are not comparable | Report tokens and cost next to accuracy ([[holistic-agent-leaderboard]], §4.1) |

---

## Check your understanding

1. Tülu 3's 8B SFT recipe produced 59.8 to 60.1 across five seeds and its 70B recipe 70.0 to 72.6 across
   three. Explain why the band must be set per model size rather than once for the recipe, and what that
   implies for a capstone run at 1B.
2. Olmo 3 Think 32B gains 20 IFBench points and loses 5 AlpacaEval points between its 3.0 and 3.1 RL runs.
   Construct the argument that this is over-optimization and the argument that it is not, and state the one
   measurement that separates them.
3. A paired evaluation on 200 tasks with a 20% discordant share gives a 6.2-point half-width against 9.6
   unpaired. Explain the mechanism by which pairing removes variance, and name a case where it is unavailable.
4. SWE-agent-LM-32B scores 40.2% on SWE-bench Verified and 8.4% on SWE-bench Multilingual, where its base
   scores 6.5%. Explain why "learned agentic interaction but not the language" and "contaminated at the
   repository level" predict different results, and design the measurement that distinguishes them.
5. Using the leave-one-out advantage arithmetic of §6, explain why the choice between masking limit-ended
   trajectories and scoring them as failures changes the gradient of the successful rollouts in the group.
6. R1's 32B comparison gives distillation 72.6 AIME against RL-from-base 47.0. Explain why this does not
   establish that distillation is preferable for a general-purpose model, and name the omitted measurement.
7. SmolLM3 repaired a RULER regression by merging the APO model soup with a mid-training checkpoint at
   0.9 / 0.1. Explain what this implies about where the long-context ability was held, and why the repair is
   not a substitute for training the two objectives together.
8. A capstone reports that its stage matched the published number within band and that its held-out suite did
   not move. Explain why this is compatible with a narrowing of general capability, and name the two gate
   components that would catch it.

---

## Connections

- **Previous chapter:** [[ch-58a]] — Open General-Model Recipes End to End: Pretraining to Merge. Supplies
  the six recipes compared stage by stage; this capstone takes one stage out of one of them.
- **Dependency:** [[ch-53]] — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and
  Perturbation Robustness. Supplies the gate's harness and the slice and regression machinery of §4.
- **Dependency:** [[ch-46a]] — Lab: Small Agentic SFT-then-RL Run with a Generality Gate. Supplies the
  agentic option's mixture axis, trajectory-ending policy, and abstention slice.
- **Related:** [[ch-51a]] — Evaluating Agent Generality and Reliability. Supplies the held-out environment
  family and the reliability measurements used in §4(a) and §6.
- **Next chapter:** none. ch-59 closes the course.

---

## Sources

- [[tulu-3]], [[rlvr-tulu3]] — option A's stages and RLVR component; the seed and template tables are
  extracted in [[tulu3-reproduction-tolerance]].
- [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]] — option A's launch commands and
  verified ledger rows, including the loss-reduction and episode-count conflicts.
- [[olmo-3]] — the long-context stage sizes used in §5 (50B tokens for the 7B model, 100B for the 32B, from
  a 639B-token pool; §3.6, Table 11, read from the report itself because the card does not carry them).
- [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]] — per-run Olmo 3 DPO
  and RL settings, framework defaults, and the PR #1547 default drift in the Recipe table.
- [[smollm-3]], [[phi-4]] — options C and D; the verified extracts are in [[smollm3-longcontext-merge]] and
  [[phi4-reasoning-recipe-and-gate]].
- [[deepseek-r1]] (full card `model-reports/deepseek-r1.md`; `papers/deepseek-r1.md` under the same slug is a
  redirect stub), [[deepseek-r1-recipe]] — option E: the six students, the ~800K SFT set, the Table 16
  distillation-against-RL comparison, and the Table 7 GPU-hour accounting.
- [[deepswe]], [[deepswe-recipe]] — option F's RL variant, compact filtering, and the blog-against-script
  conflicts; [[r2e-gym]], [[swe-gym]], [[skyrl-agent]] — its environments and comparison points.
- [[swe-smith-rft]], [[toucan-sft]] — chapter excerpts of SWE-smith (arXiv:2504.21798v2) and Toucan
  (arXiv:2510.01179v1); no library card exists for either artifact yet.
- [[tulu3-reproduction-tolerance]], [[olmo3-think-stage-deltas]] — chapter excerpts holding the tables quoted
  in §1, §3, §4 and §8.
- [[agenttuning]], [[reasoning-trap-tool-hallucination]] — the agent-data-share evidence and the abstention
  slice used in §6 and the Generalization lens.
- [[rlvr-beyond-base-model]] — pass@k against pass@1 as the coverage measurement in §7.
- [[signal-and-noise-eval]] — benchmark signal-to-noise as the basis for per-benchmark bands in §3.
- [[swe-bench-illusion]], [[agentic-benchmark-checklist]], [[holistic-agent-leaderboard]] — contamination
  indicators, trivial-agent checks, and cost and scaffold reporting used in §4(c) and §6.
- [[ruler]], [[prolong]] — the synthetic and natural long-context slices required in §5.
- [[ties-merging]], [[mix-data-or-merge-models]] — merge methods and the merge-against-mixing ablation.
- [[olmo-2]] — the base-model caution in the Recipe section's starting-point paragraph.
- [[grpo]] — the group-advantage objective shared by options B, D and F.
