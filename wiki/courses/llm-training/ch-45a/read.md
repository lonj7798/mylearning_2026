<!-- chapter: ch-45a
     track: rl
     kind: content
     title: Preference-Optimization and RL Stage Recipes Side by Side
     deps: [ch-45, ch-39]
     sources: [[zephyr]], [[llama-3-recipe]], [[qwen-2.5-recipe]], [[tulu-3]], [[tulu-3-hyperparameter-tables]],
              [[rlvr-tulu3]], [[tulu-3-1]], [[olmo-3]], [[allenai-olmo3-open-instruct-scripts-recipe]],
              [[smollm-3]], [[smollm3-apo]], [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]],
              [[rlhf-instructgpt]], [[instructgpt-ppo-appendix]], [[llama-2-recipe]], [[hf-dpo-zoo]], [[dpo]],
              [[simpo]], [[grpo]], [[grpo-recipe]], [[deepseek-r1-recipe]], [[dapo]], [[verl-dapo-recipe]],
              [[skywork-or1]], [[magistral-recipe]], [[phi-4]], [[phi-4-reasoning-rl]], [[acereason-nemotron]],
              [[mimo-7b]], [[scalerl]], [[simplerl-zoo]]
     figures: figures/step-budget-normalizer.html, figures/scalerl-compute-curve.html
     revised: 2026-09 (generality revision)
-->

# Chapter 45a — Preference-Optimization and RL Stage Recipes Side by Side

> **Core insight.** Published post-SFT settings look comparable and are not. Two reports can both print "batch
> 512" and mean different things: DAPO's rollout batch is 512 prompts while its training mini-batch is 512
> samples, and 512 prompts × 16 responses ÷ 512 samples is exactly the 16 gradient updates per rollout step the
> paper states ([[dapo]] §4.1). Two reports can both print "β" and mean different things: Zephyr's β = 0.1 is
> the DPO temperature on an unnormalized log-ratio ([[zephyr]] §4.4), Tülu 3's β = 5 is the same coefficient on
> a per-token-normalized log-ratio ([[tulu-3-hyperparameter-tables]] Table 20), DeepSeek-R1-Zero's 0.001
> multiplies a KL estimator inside the loss ([[deepseek-r1-recipe]] §2.1), and Tülu 3's RLVR β = 0.05 subtracts
> from a reward in {0, 1} ([[tulu-3-hyperparameter-tables]] Table 21). A third of the values that circulate for
> these runs are not in the primary sources at all: InstructGPT never prints a PPO policy learning rate, only
> the sweep range 2.55e-6 to 2.55e-5 and the selection rule ([[instructgpt-ppo-appendix]] App. E.9).
>
> **Guideline.** When you copy a number from a report, copy its unit, its stage, its model size, and its locus in
> the same row, because scoped-away values are the failure mode these ledgers exist to prevent. When a paper table, the
> paper's prose, and a released launch script disagree, prefer the script for questions about the released
> checkpoint and keep the other rows, because for Tülu 3 70B RLVR the caption's β = 0.07 and the script's
> `--beta 0.07` agree against the prose's "β = 0.7" ([[tulu-3-hyperparameter-tables]] Table 21, §6.4). When you
> are choosing a configuration rather than reporting one, prefer axes that raise the asymptote over axes that
> raise early efficiency, because in the 400,000-GPU-hour ScaleRL study loss aggregation, curriculum, length
> penalty and advantage normalization mainly moved the efficiency exponent B while loss type, logit precision and
> batch size moved the asymptote A ([[scalerl]] Abstract, §3.2). When a single row must transfer to a new base
> model, re-tune the prompt difficulty first, because holding one RL configuration fixed across ten open base
> models made Mistral-7B collapse on MATH levels 3-5 and Qwen2.5-7B stagnate on easy data ([[simplerl-zoo]] §3.2).

## Why this chapter matters for a general-purpose model

Every chapter from ch-37 to ch-45 introduced one objective or one pipeline shape. This chapter is the place where
those shapes are pinned to the numbers the labs actually ran, in a form that can be checked. That matters for a
general-purpose model in three concrete ways.

First, most published post-SFT stages were tuned on a narrow evaluation. Tülu 3's DPO algorithm choice was
decided on one dataset, UltraFeedback, against a development average ([[tulu-3-hyperparameter-tables]] Table 18).
DAPO, Skywork-OR1, AceReason-Nemotron, MiMo-7B and Phi-4-reasoning-plus all selected their RL settings on
mathematics, and in three of those five the report evaluates nothing outside mathematics and code. A row copied
from such a run carries the narrowness of the run that produced it, and the only defence is to know what the run
measured.

Second, the two stages differ in what they can damage. A preference stage consumes a fixed set of pairs and
changes the policy by at most what those pairs express; the danger is the direction of the gradient on the
rejected side (ch-43a) and the breadth of the pair distribution. An RL stage generates its own data, so its
prompt filter, its length cap and its reward shape decide which behaviours the model ever sees reinforced. The
ledger columns differ for that reason: pairs and their source on one side, prompts-per-step and samples-per-prompt
on the other.

Third, a configuration is only meaningful together with the compute it was run at. The ScaleRL study fits a
saturating curve to reward against RL compute and shows that the ordering of two recipes can invert between a
16,000-GPU-hour budget and a 100,000-GPU-hour budget ([[scalerl]] §5, Fig. 10). A row without a compute budget
attached is a row you cannot use.

This chapter sits after ch-45, which covered the pipelines these stages compose into, and before ch-45b, which
takes the RL stage into multi-turn agentic settings where the same columns acquire a turn dimension.

## §1 What a recipe row must carry, and the unit traps

**Definition.** A recipe row is one setting of one stage of one named checkpoint, with a locus in a primary
source and a status saying how it was checked. The column set is fixed by style-guide §5.2: model, size, stage,
setting, value, source location, status, and the ablation that selected the value.

**The measurable problem.** Values migrate between rows. The 2026-09 audit of this course's own cards found an
InstructGPT PPO learning rate that the paper does not print, a DeepSeekMath "16 prompts × 64 completions" batch
decomposition that the paper does not print, and a Tülu 3 RLVR episode count taken from a launch script rather
than the report. Each of these is a single number that silently changes what a reader would run.

### 1.1 Prompts, samples, pairs, episodes, steps

Five quantities are routinely printed with the same word "batch".

1. **Prompts per step** — how many distinct questions the rollout covers.
2. **Samples per prompt** (`G`, `n`, "responses per query", "group size") — how many completions per question.
3. **Samples per step** = prompts × samples per prompt.
4. **Mini-batch** — the unit of one gradient update. Dividing samples per step by the mini-batch gives the number
   of gradient updates per rollout, which is the degree of off-policyness inside a step.
5. **Episodes** — a cumulative count of generated sequences used as a stopping budget, not a per-step quantity.

**Worked example (DAPO, checkable by hand).** The paper prints "the prompt batch size is 512 and we sample 16
responses for each prompt. For training, the mini-batch size is set to 512, i.e., 16 gradient updates for each
rollout step" ([[dapo]] §4.1). Then:

- samples per step = 512 × 16 = 8,192;
- gradient updates per step = 8,192 ÷ 512 = 16, which matches the printed sentence;
- therefore the first 512 counts prompts and the second 512 counts samples.

Reading both as prompts would give one gradient update per step and a 16× error in the amount of off-policy
reuse. The verl configuration for the same recipe adds a quantity the paper does not print: `gen_batch_size:
1536` against `train_batch_size: 512`, so up to 1,536 prompts are generated to yield 512 groups that survive the
all-correct / all-wrong filter, i.e. up to 24,576 generated samples per 8,192 trained samples
([[verl-dapo-recipe]]).

**Worked example (Tülu 3 8B RLVR).** The report's effective batch of 224 ([[tulu-3-hyperparameter-tables]]
Table 21) is reproduced exactly by the launch command: 7 actor GPUs × `--local_rollout_batch_size 32` = 224. With
`PPO update iterations K = 4` and one mini-batch, each rollout of 224 samples produces 4 gradient updates. The
table's "Total Episodes 100,000" then corresponds to 100,000 ÷ 224 ≈ 446 rollout steps and about 1,786 gradient
updates — while the released script says `--total_episodes 10000000`, a budget the run did not exhaust, since
checkpoints were evaluated "every 100 training steps" and selected on MATH and IFEval. The number of steps
actually run is not reported.

The interactive companion [figures/step-budget-normalizer.html](figures/step-budget-normalizer.html) carries out
this arithmetic for every RL row in the chapter and shows which quantities each source printed and which are
derived.

### 1.2 Status is part of the value

Four statuses appear in the ledgers below. `verified` means read at the stated locus on the stated date.
`not reported` means the listed places were checked and the value is absent — for example DeepSeekMath's clip ε,
which appears only as a symbol in Eq. 1, 3 and 15 ([[grpo-recipe]]). `derived` means computed from printed
values, with the formula shown. `conflict` means two sources disagree and both rows are kept.

A fifth category has no status because it has no row: values that circulate but are in no source. Those are
listed after each table.

## §2 The preference stage: what actually varies

Four axes separate the published preference stages. Everything else in the ledger follows from them.

### 2.1 Pair source: off-policy, on-policy, or a capability gap

**Definition.** A pair is *on-policy* when both responses were sampled from the checkpoint being trained (or its
immediate predecessor), and *off-policy* when they come from other models or from a fixed corpus.

- Fully off-policy: Zephyr takes UltraFeedback's four responses per prompt, makes the highest GPT-4 mean score
  the chosen response and "one of the remaining three at random" the rejected one — deliberately not the lowest,
  "to encourage diversity and make the DPO objective more challenging" ([[zephyr]] §4.1).
- Capability-gap pairs: SmolLM3 takes the chosen response from Qwen3-32B and the rejected response from
  Qwen3-0.6B ([[smollm3-apo]]). The pair encodes a size difference, not a judgement about the policy.
- On-policy resampling: Qwen2.5 has the SFT model resample responses for new queries in mathematics, coding,
  instruction following and logical reasoning, keeps passing responses as positives and failing ones as
  negatives, for about 150,000 pairs ([[qwen-2.5-recipe]] §4.2).
- Mixed, round-based: Llama 3.1 collects new preference annotations in each of 6 rounds and trains DPO
  "primarily on the most recent batches of preference data" from the best models of previous rounds, keeping only
  pairs labelled "significantly better" or "better" ([[llama-3-recipe]] §4.1.4, §4.1.6, §4.2.1).

**Implication for a general-purpose model.** The pair source bounds what the stage can teach. A
capability-gap pair teaches the policy to look more like the larger teacher, which is a distillation signal
(ch-35), not a correction of the policy's own errors. An on-policy pair whose rejected side is the policy's own
failure is the only construction where the pushed-down probability mass belongs to the policy in the first place
(ch-43a).

### 2.2 Loss variant and what β means in each one

The DPO objective ([[dpo]]) is

```
L_DPO = −log σ( β · [ (log π_θ(y_c|x) − log π_ref(y_c|x)) − (log π_θ(y_r|x) − log π_ref(y_r|x)) ] )
```

where `x` is the prompt, `y_c` and `y_r` the chosen and rejected responses, `π_θ` the policy, `π_ref` the frozen
reference, `σ` the logistic function, and `β` a scalar temperature. The bracketed terms are **sums over tokens**.
Length-normalized DPO (the "DPO-norm" of [[tulu-3-hyperparameter-tables]] Table 18, following the normalization
in [[simpo]]) divides each bracketed term by its response length in tokens before multiplying by β.

**Worked example.** Take a pair where the chosen response is 200 tokens with a summed log-ratio of +4.0 nats, and
the rejected response is 400 tokens with a summed log-ratio of −6.0 nats.

- Plain DPO at β = 0.1: argument = 0.1 × (4.0 − (−6.0)) = 1.0; σ(1.0) = 0.731; loss = −log 0.731 = 0.313.
- Length-normalized DPO at β = 5: argument = 5 × (4.0/200 − (−6.0/400)) = 5 × (0.020 + 0.015) = 0.175;
  σ(0.175) = 0.544; loss = 0.609.

The same pair, the same policy, and a β that is fifty times larger produce a *weaker* margin, because
normalization shrank the log-ratio difference from 10.0 to 0.035. Matching the plain-DPO margin of 1.0 under
normalization would need β ≈ 1.0 / 0.035 ≈ 28.6. This is why β = 5 (Tülu 3, Olmo 3) and β = 0.1 (Zephyr,
Llama 3.1, plain DPO) are not two points on one scale.

**Evidence that the variant mattered here.** In [[tulu-3-hyperparameter-tables]] Table 18, on UltraFeedback over
an early Tülu 3 SFT checkpoint scoring 55.7, plain DPO at β = 0.1 gives 55.2 and SimPO gives 51.8 and 52.9, while
length-normalized DPO at β = 5 for 1 epoch gives 57.3. The same variant at β = 2 gives 46.8 and at 3 epochs
gives 53.4. One run per row, one dataset, one development average — the selection is real but narrow
(**Result (single study)**).

Other variants in the ledger: Nemotron-4-340B adds an SFT loss on chosen responses to DPO after observing both
chosen and rejected likelihoods fall, then replaces DPO with Reward-aware Preference Optimization for three
iterations ([[nemotron-4-synthetic]] §3.3.2). Llama 3.1 adds an NLL term on the chosen sequence with coefficient
0.2 for the same stated reason ([[llama-3-recipe]] §4.1.4). SmolLM3 uses Anchored Preference Optimization
([[smollm3-apo]]). The variant survey is [[hf-dpo-zoo]].

### 2.3 Epochs, and the one place overfitting was measured

Zephyr trained DPO for one to three epochs and reports that "after one epoch of DPO training, the model would
strongly overfit, as indicated by perfect training set accuracies", yet the best MT-Bench and AlpacaEval model
was one SFT epoch followed by three DPO epochs; and that if the SFT model is trained for more than one epoch,
"the DPO step actually induces a performance regression with longer training" ([[zephyr]] §5, Fig. 3). Tülu 3
went the other way and settled on 1 epoch, with its own Table 18 showing the same β = 5 configuration at 57.3 for
1 epoch and 53.4 for 3 epochs on a different base and dataset. Both are single studies and they point in opposite
directions, so epochs is a value to re-measure, not to copy (**Open question**).

### 2.4 Reference model and the memory trick that changes nothing else

Tülu 3 pre-computes and caches the reference log-probabilities over the dataset instead of holding a reference
model in memory ([[tulu-3-hyperparameter-tables]] §5.4.2). This is exactly equivalent as long as `π_ref` is
frozen, which it is for every offline preference row in this chapter. It is not equivalent for online preference
methods where the reference moves, which is why the row records the reference policy separately from the loss.

## §3 The RL stage: the five columns that decide behaviour

### 3.1 KL: coefficient, placement, and target

Three placements appear, and they are not interchangeable.

1. **KL inside the reward.** InstructGPT: `r(x,y) = r_φ(x,y) − β log(π^RL(y|x) / π^SFT(y|x))` with β = 0.02
   ([[instructgpt-ppo-appendix]] App. C.4). Llama 2: `R = R̃_c − β D_KL(π_θ ‖ π_0)` with β = 0.01 for 7B and 13B
   and 0.005 for 34B and 70B ([[llama-2-recipe]] §3.2.3, Eq. 4). Tülu 3 RLVR: the same shape with the verifier's
   binary output in place of the RM score, β = 0.05 for 8B and 0.07 for 70B.
2. **KL inside the loss.** DeepSeek-R1-Zero and R1 stage 1 add the Eq.-2 KL estimator to the loss with
   coefficient 0.001 ([[deepseek-r1-recipe]] §2.1, §3.2.1). DeepSeekMath uses 0.04 ([[grpo-recipe]] §4.2).
   SimpleRL-Zoo uses 1e-4 for 0.5B–14B models and 1e-3 above 14B ([[simplerl-zoo]] App. B.5). Tülu 3.1 uses
   `--beta 0.01` with the `kl3` estimator ([[tulu-3-hyperparameter-tables]], open-instruct command).
3. **No KL at all.** DAPO removes the term ([[dapo]]); Skywork-OR1 removes it because "the KL penalty hinders
   further improvements in test performance during multi-stage training" ([[skywork-or1]] §3.2.6);
   AceReason-Nemotron sets β = 0 ([[acereason-nemotron]] §3.2.2); MiMo-7B removes the KL loss ([[mimo-7b]]
   §3.3.2); Magistral removes it at a "compute cost we find unjustified" ([[magistral-recipe]] §2.1); the Olmo 3
   RL scripts pass β = 0.0 against a framework default of 0.05
   ([[allenai-olmo3-open-instruct-scripts-recipe]]).

A fourth case is easy to misread. Phi-4-reasoning-plus prints `− β D_KL(π_θ ‖ π_θold)` with β = 0.001, where the
target is the **rollout policy**, not a frozen reference ([[phi-4-reasoning-rl]] §4.2). A KL to π_θold constrains
the size of one update; a KL to π_SFT constrains total drift from the starting checkpoint. Only the second is a
retention control.

**Worked example of what β = 0.05 buys in a reward.** With a binary verifier reward and Tülu 3's β = 0.05, a
correct response that has drifted 4 nats from the reference scores 1 − 0.05 × 4 = 0.8, and an incorrect response
at 0.5 nats scores 0 − 0.025 = −0.025. Twenty nats of drift cancels the entire reward for a correct answer. The
Tülu 3 sweep over β ∈ {0.1, 0.05, 0.03, 0.01} found "more KL divergence typically results in lower average
scores" with one exception, and the released 8B model used β = 0.05
([[tulu-3-hyperparameter-tables]] Table 21; report §6.3).

### 3.2 Clip range: symmetric, asymmetric, or absent

PPO and GRPO clip the importance ratio `ρ = π_θ / π_θold` into `[1 − ε_low, 1 + ε_high]`. The published values:

- symmetric 0.2: InstructGPT, Llama 2, Tülu 3 RLVR, Skywork-OR1, SimpleRL-Zoo;
- asymmetric: DAPO at ε_low = 0.2, ε_high = 0.28 ([[dapo]] §4.1, confirmed by verl's `clip_ratio_low: 0.2`,
  `clip_ratio_high: 0.28`, [[verl-dapo-recipe]]); Olmo 3 at 0.2 / 0.272 (scripts) against a paper table printing
  0.272 ([[allenai-olmo3-open-instruct-scripts-recipe]]); Magistral at ε_high between 0.26 and 0.28 adjusted
  during training for Medium and 0.3 for Small after a cold start with "far lower entropy"
  ([[magistral-recipe]] §2.1, §5.3); the ScaleRL reconstruction of DAPO at ε_max = 0.26 ([[scalerl]] §A.16);
- very large: DeepSeek-R1 stage-1 RL at ε = 10, with the stated reason that a lower ε "truncates gradients for
  many tokens and degrades performance" while a higher one may cause instability ([[deepseek-r1-recipe]] §3.2.1).
  DeepSeek-R1-Zero's ε is not reported.

Four different numbers on the same axis, chosen by four different arguments, none of them by a printed sweep.
This is the clearest case in the chapter where the "Evidence for this value" column is "no ablation reported".

### 3.3 Inner updates: strictly on-policy or not

Dividing samples per step by the mini-batch gives the number of gradient updates taken on one rollout.

| Run | Samples per step | Mini-batch | Updates per rollout |
|---|---|---|---|
| AceReason-Nemotron | 128 prompts × 8 or 16 | one update, `ρ = 1` fixed | 1 |
| Skywork-OR1-7B stage 1 | 256 × 16 | 256 (batch = mini-batch) | 1 |
| DeepSeekMath-RL 7B | not stated in these units | — | 1 ("a single update following each exploration stage") |
| DeepSeek-R1-Zero | 8,192 outputs per rollout | 16 mini-batches | 16, 1 inner epoch |
| MiMo-7B | 512 | 32 | 16 |
| DAPO | 8,192 | 512 | 16 |
| Tülu 3 8B RLVR | 224 | 224, `K = 4` | 4 |
| SimpleRL-Zoo | 1,024 × 8 | 256 | 32 if the mini-batch counts samples (unit not stated) |

AceReason-Nemotron is the one row with a measurement attached: "applying multiple (2 or 4) gradient updates after
model generation with a group of G rollouts per prompt led to rapid entropy collapse around 100 steps... using
exactly one gradient update after model generation... consistently prevented collapse"
([[acereason-nemotron]] §3.2.2, Fig. 3c). Skywork-OR1 reports the same direction: "On-policy training mitigates
entropy collapse and leads to higher test performance" (§4). ScaleRL disagrees about the size of the effect,
finding the off-policy algorithm to be one of the axes that changes efficiency rather than the asymptote
([[scalerl]] §3.1, Fig. 4a) — but ScaleRL's off-policy variants are paired with a truncated-importance-sampling
loss (CISPO), which the two former runs did not use. The claim "strictly on-policy is required" is therefore
**Result (single study)** twice over, not **Replicated**, because the settings differ.

### 3.4 Length: a cap, a schedule, or a shaped reward

- A fixed cap: MiMo-7B at 32,768 ([[mimo-7b]] §3.3.3); Tülu 3 RLVR at 2,048 response tokens, 1,024 for GSM8K only.
- A schedule during training: DeepSeek-R1-Zero at 32,768 rising to 65,536 after step 8.2k
  ([[deepseek-r1-recipe]] §2.1); AceReason-Nemotron at 8K → 16K → 24K → 32K, where "directly starting from 16K or
  24K resulted in suboptimal results" ([[acereason-nemotron]] §3.2.2); Skywork-OR1-7B at 16K → 32K and
  Skywork-OR1-32B at 16K → 24K, with the batch and group size changing at the same boundary ([[skywork-or1]]
  Tables 11, 12); Magistral Medium at 16k → 24k → 32k with the batch shrinking 8k → 4k → 2k as generations grew
  ([[magistral-recipe]] §5.2).
- A shaped reward: DAPO's soft overlong punishment, a linear penalty over the last 4,096 of 20,480 tokens
  reaching −1.0 at the cap ([[verl-dapo-recipe]]); Phi-4-reasoning-plus's length-aware cosine accuracy reward,
  which reduces the reward for a *correct* answer beyond 25,600 tokens and reduces the penalty for an *incorrect*
  answer beyond 3,702 tokens ([[phi-4-reasoning-rl]] §4.1); Magistral's length penalty from 0 to −0.1
  ([[magistral-recipe]] §2.2).
- Forced interruption instead of a penalty: ScaleRL appends an end-of-thinking phrase to stop a long generation,
  and reports that adding a length penalty on top "does not improve performance" ([[scalerl]] §2, App. A.15).

### 3.5 Prompt filter, and why it changes the gradient

**Mechanism.** With group-relative advantages, a prompt whose `G` samples all receive the same reward has zero
advantage for every sample and contributes no policy gradient. Prompts near the extremes contribute little and
are asymmetric.

**Worked example.** Take `G = 8` samples, a binary reward, and standardized group-relative advantages
`Â = (r − mean) / std` with the population standard deviation `√(p(1−p))` at pass rate `p`.

| Pass rate p | std | Â of a correct sample | Â of an incorrect sample |
|---|---|---|---|
| 0/8 | 0 | — | 0 (no gradient) |
| 1/8 | 0.331 | +2.646 | −0.378 |
| 4/8 | 0.500 | +1.000 | −1.000 |
| 7/8 | 0.331 | +0.378 | −2.646 |
| 8/8 | 0 | 0 (no gradient) | — |

At a low pass rate the rare success dominates; at a high pass rate the rare failure dominates. That is what the
published filters are managing:

- Drop the degenerate groups: DAPO's dynamic sampling removes groups with accuracy 1 or 0 and resamples until the
  batch is full ([[dapo]] §3.3, [[verl-dapo-recipe]]); MiMo-7B does the same ([[mimo-7b]] §3.3.2); Skywork-OR1
  calls it rejection sampling and gives the reason that zero-advantage samples still enter the entropy and KL
  terms and so distort their relative weight ([[skywork-or1]] §3.1); Olmo 3's scripts enable active sampling with
  `no_resampling_pass_rate 0.875` ([[allenai-olmo3-open-instruct-scripts-recipe]]).
- Shift the difficulty distribution as the model improves: AceReason-Nemotron filters out prompts with pass rate
  above 6/16 at the 24K and 32K stages ([[acereason-nemotron]] §3.2.2, Table 3).
- Filter by model-specific difficulty before training: Skywork-OR1's dataset carries a difficulty score from 0 to
  16 per problem **and per model variant**, and the released preprocessing keeps 1 ≤ difficulty ≤ 15 for the
  DeepSeek-R1-Distill-Qwen-32B column, so the same 105,055 math and 14,057 code problems produce a different
  training set for each policy size ([[skywork-or1]]).
- Reintroduce easy prompts deliberately: MiMo-7B keeps a pool of perfectly-solved problems and samples from it
  with probability α = 10%, after finding that re-using easy data directly was unstable ([[mimo-7b]] §3.3.2).

**Conditions and limits.** The pass-rate thresholds above were measured on mathematics and code with binary
verifiers. None of these sources reports a pass-rate filter tuned on a non-verifiable domain.

## §4 Conflicts, and how each was resolved

Six disagreements in this chapter's sources are recorded as separate rows rather than averaged away.

1. **Tülu 3 70B RLVR, β.** Table 21's caption says the final 70B model used β = 0.07; §6.4's prose says β = 0.7;
   the released launch command says `--beta 0.07`. Two of three agree and one of those two is the artefact that
   produced the checkpoint, so the ledger's released-checkpoint row uses 0.07 and keeps the prose as a conflict
   note ([[tulu-3-hyperparameter-tables]]).
2. **Tülu 3 RLVR, episodes.** Table 21 prints 100,000; §6.4 prose prints 400,000 for 70B; the 8B script prints
   `--total_episodes 10000000` and the 70B script prints `400000`. The script value is a stopping budget, not a
   count of episodes consumed, so the two are different quantities and both rows stay.
3. **Tülu 3 405B RLVR, table caption.** Table 35's caption reads "PPO used for optimizing against a general RM"
   while its column header reads "405B RLVR" and §8.1 introduces it as the RLVR table. The column header and the
   surrounding text agree against the caption.
4. **DeepSeekMath, batch unit.** §4.2 prints "training batch size is 1024" and "max length is set to 1024" with
   no unit for either. [[grpo-recipe]] records both as "unit not stated" and explicitly forbids the common
   decomposition into "16 prompts × 64 completions", which the paper does not print.
5. **DAPO paper versus verl recipe.** The paper prints prompt batch 512 and mini-batch 512; verl's recipe
   documentation prints `train_batch_size: 512` with `gen_batch_size: 1536` and `overlong_buffer.penalty_factor:
   1.0`, neither of which the paper prints. The documentation does not claim to be the released run's exact
   configuration, so the two are separate rows ([[dapo]], [[verl-dapo-recipe]]).
6. **Olmo 3 scripts versus paper tables.** The recipe card records seven rows where `scripts/train/olmo3` at
   commit d8a7f1c and arXiv:2512.13961v2 Tables 47–49 disagree on GPU counts, unique prompts per step, dataset
   sizes and maximum asynchrony, and one row where a flag removed in PR #1547 means a script now silently runs
   with a different framework default than it did at release time
   ([[allenai-olmo3-open-instruct-scripts-recipe]]).

A seventh case is not a conflict between sources but between a comment and the code beneath it: Skywork-OR1's
preprocessing script says in a comment that math data is duplicated when mixing, while the file list it builds
duplicates the code split ([[skywork-or1]]).

**Values that circulate and are in no primary source.** InstructGPT's PPO policy learning rate of 1.41e-5 and
"4 epochs per rollout" (App. C.4 states one inner epoch over 8 mini-batches; E.9 gives only the sweep range and
the selection rule) ([[instructgpt-ppo-appendix]]). DeepSeekMath's clip ε = 0.2 and rollout temperature 1.0
([[grpo-recipe]]). Phi-4's "+1 correct, −0.5 incorrect" reward, which the report replaces with a cosine-scaled
length-aware term ranging over [+0.5, +1.0] and [−1.0, −0.5] ([[phi-4-reasoning-rl]] §4.1, against [[phi-4]]).
A Tülu 3 RLVR row combining "lr ~1e-6, β_KL ~0.04, ~128 prompts, 4 rollouts each" ([[rlvr-tulu3]] Technical
Details), none of which appears in Table 21.

## §5 What a row is worth: RL compute scaling

**Definition.** ScaleRL fits reward against RL compute with a saturating curve
([[scalerl]] Eq. 1):

```
R_C − R_0 = (A − R_0) × 1 / (1 + (C_mid / C)^B)
```

`R_C` is the pass rate (mean@16) on an i.i.d. validation set after `C` GPU-hours of RL; `R_0` is the starting
checkpoint's pass rate; `A` is the asymptotic pass rate; `B` is a scaling exponent, larger meaning faster ascent;
`C_mid` is the compute at the curve's midpoint. Model and training data are held fixed.

**Worked example.** Two fitted rows from [[scalerl]] Table 1: ScaleRL at batch 512 has `C_mid = 2,818`,
`B = 1.77`, `A = 0.605`; at batch 2048 it has `C_mid = 10,909`, `B = 1.70`, `A = 0.645`. The fraction of each
run's own available gain realized at a budget `C` is `1 / (1 + (C_mid/C)^B)`:

- at C = 16,000 GPU-hours: batch 512 gives `(2818/16000)^1.77 = 0.046`, so 0.956 of its gain; batch 2048 gives
  `(10909/16000)^1.70 = 0.522`, so 0.657 of its gain. With a common `R_0`, the smaller batch is ahead.
- at C = 100,000 GPU-hours: batch 512 gives 0.998 of 0.605; batch 2048 gives 0.977 of 0.645. The larger batch is
  ahead.

The ordering inverts, which is the paper's stated finding: "larger batch size is slower in training but settles
at a higher asymptote. Batch size show an inverse trend initially where smaller values seem better at lower
compute budget" (Fig. 10). [figures/scalerl-compute-curve.html](figures/scalerl-compute-curve.html) lets you move
A, B and C_mid and watch where two curves cross.

**What moved which parameter** (8B dense, verifiable mathematics, ±0.02 error margin on A from three repeated
runs, [[scalerl]] Fig. 8a):

| Axis | Effect reported |
|---|---|
| Loss type (CISPO, GSPO vs DAPO) | raises A; inside ScaleRL the leave-one-out shows similar A and B 2.01 vs 1.77 |
| FP32 precision at the LM head | A 0.52 → 0.61 in the forward ablation |
| Batch size | A 0.605 (512) → 0.645 (2048) |
| Loss aggregation, advantage normalization, curriculum, length penalty, off-policy algorithm | "primarily modulate compute efficiency without materially shifting the asymptote" |
| Model size | Llama-4 Scout 17B×16 reaches A = 0.710 vs 0.645 for the 8B dense model |

**Conditions and limits.** Every number above is one model family, one domain, and one codebase. The comparison
curves labelled DeepSeek (GRPO), Qwen2.5 (DAPO), Magistral and MiniMax in [[scalerl]] Fig. 2 are the authors'
reconstructions of those recipe *shapes* at 8B with deliberate deviations (the DAPO reconstruction uses
ε_max = 0.26 and drops zero-variance prompts from a larger batch of 1,280 instead of refilling), not
re-evaluations of those labs' released checkpoints ([[scalerl]] App. A.16). Reading Fig. 2 as "DAPO is worse than
ScaleRL" overstates it; reading it as "this recipe shape, in this codebase, at this size, saturated lower"
is what the appendix supports.

## Negative samples and negative feedback

Both stages in this chapter use negatives **as gradient** in the sense of ch-43a §6.1 case 4: an explicit
decrease of a sample's likelihood. The preference stage does it through the rejected term of the DPO-family loss;
the RL stage does it through a negative group-relative advantage. Nothing in this chapter uses negatives as
content or as conditioning; those constructions belong to ch-31 and ch-43a.

**Where the negatives come from, per stage.**

| Stage | Negative source | Labeller | False-negative rate |
|---|---|---|---|
| Preference, off-policy | a non-top response from a fixed corpus | GPT-4 scores (Zephyr, UltraFeedback) | not reported |
| Preference, capability gap | a smaller model's response | none — the gap is assumed | not reported |
| Preference, on-policy | the policy's own failing response | quality checks plus human and automated review (Qwen2.5 §4.2) | not reported |
| Preference, RM-ranked | the lower-scored of two responses | reward model | Nemotron-4 reports RM-as-judge Chat-Hard 0.87 vs LLM-as-judge 0.54 ([[nemotron-4-synthetic-recipe]]) |
| RL, verifier | any sample the verifier marks wrong | exact match, sympy, unit tests, constraint checkers | not reported by any source in this chapter |

**Mechanism.** The softmax gradient is `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, so pushing down a sample moves its
mass to whatever the model already ranks highest. Two consequences visible in these ledgers:

1. The magnitude of a negative gradient in the RL stage is set by the group's pass rate, not by how wrong the
   sample is. The table in §3.5 shows an incorrect sample at pass rate 7/8 receiving `Â = −2.646` while an
   incorrect sample at pass rate 1/8 receives `−0.378`. A prompt filter is therefore also a negative-gradient
   magnitude control.
2. In the preference stage, the rejected term's gradient does not vanish when the rejected response is already
   unlikely; Nemotron-4 observed "both chosen and rejected likelihoods fall" under DPO and responded with an SFT
   loss on chosen responses, and Llama 3.1 added the same anchor with coefficient 0.2
   ([[nemotron-4-synthetic]] §3.3.2; [[llama-3-recipe]] §4.1.4). Both report the change as stabilizing without a
   printed ablation.

**Controls the ledgers actually contain.**

- Anchor with a positive NLL term: Llama 3.1's coefficient 0.2 on chosen; Nemotron-4's SFT-loss weight tuned in
  [1e-5, 1e-3] for DPO and fixed at 1e-5 for RPO.
- Bound the push-down by normalizing: length-normalized DPO divides each log-ratio by response length, which caps
  how much a long rejected response can dominate the margin (§2.2 worked example).
- Scale the target by the reward gap instead of using a constant one: Nemotron-4's RPO fits `β(log-ratio gap)` to
  `η(r*(y_c) − r*(y_l))`, with the stated purpose of not "unlearning" a high-quality rejected response
  ([[nemotron-4-synthetic]] §3.3.2).
- Keep the negative on-policy: Qwen2.5 resamples negatives from the SFT model rather than importing them
  ([[qwen-2.5-recipe]] §4.2).
- Do not penalize what the verifier could not judge — or do. Skywork-OR1 tested an advantage mask for truncated
  responses and removed it: "penalizing truncated responses does not hinder later-stage improvements and enhances
  token efficiency" ([[skywork-or1]] §3.1, §3.2.3). DAPO's overlong filtering goes the other way and is worth
  +6 points on AIME24 avg@32 in its progressive table (30 → 36, [[dapo]] Table 1). The two runs differ in base
  model and stage, and no source compares them directly (**Open question**).
- Mask the token positions that carry no content: Llama 3.1 masks header and termination tokens in both chosen
  and rejected responses, reporting tail repetition or abrupt terminations without it ([[llama-3-recipe]] §4.1.4).
- Penalize a specific failure explicitly rather than through the advantage: Tülu 3's RLVR gives a response with no
  EOS token a reward of −10.0 ([[tulu-3-hyperparameter-tables]] Table 21); Phi-4-reasoning-plus sets the accuracy
  term to −1.0 for a missing or malformed `<think>` block ([[phi-4-reasoning-rl]] §4.1).

**Diagnostics these sources logged.** Entropy over steps (Skywork-OR1 §4, AceReason Fig. 3c, Phi-4-reasoning
Fig. 7f); KL divergence against the reference over episodes (Tülu 3 Figs. 19–23, which is how the 70B run's KL
"remaining well below 1 over the duration of run" was noticed); response length split by correct and incorrect
(Phi-4-reasoning Fig. 7d); the up-clipped token probability and the share of prompts with accuracy 1 (DAPO
Figs. 3a, 3b); truncation rate, where ScaleRL reports that "truncations in the range of 10–15% typically
destabilized training" at batch 768 while ScaleRL runs stayed below 5% for over 90% of training
([[scalerl]] App. A.15).

**Share of the effect.** No source in this chapter measures what fraction of its gain comes from the negative
term. The one course-wide number that does exist — roughly 80% positives to 20% negatives at 32B in the NFT
analysis — belongs to ch-43a and must not be transferred to these runs.

**Effect on generality.** Two measurements in this chapter bear on it directly. Math-only RL raised
LiveCodeBench v5 by 6.8 points at 7B and 5.8 points at 14B, while math-only SFT on a comparable model scored 19.3
on the same benchmark ([[acereason-nemotron]] Table 1) — negatives delivered through a verifier on one domain
transferred; imitation on one domain did not. In the other direction, DeepSeek-R1's final RL stage, which adds
preference rewards for general instruction data only in the last 400 of 1,700 steps, is accompanied by a figure
showing reward rising while Codeforces pass@1 falls ([[deepseek-r1-recipe]] §3.2.2, B.5 Fig. 6).

## Recipe

Units for both tables: "pairs" are preference pairs; "prompts/step" and "samples/prompt" are per rollout step;
"samples/step" is their product; "episodes" is a cumulative generated-sequence budget as each source defines it.
A row describes one named checkpoint. Statuses follow style-guide §5.2; dates are the date this chapter's author
read the locus.

### Preference stage

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Zephyr-7B-β | 7B | preference | loss; β; peak LR; schedule; warmup; global batch; epochs | dDPO (plain DPO); 0.1; 5e-7; linear; 10%; 32; 3 (final model) | arXiv:2310.16944 §4.4 | verified 2026-09-15 | §5 Fig. 3: best MT-Bench at 1 SFT epoch + 3 DPO epochs; >1 SFT epoch causes a DPO regression |
| Zephyr-7B-β | 7B | preference | pairs and their source | UltraFeedback, 64k prompts × 4 responses; chosen = highest mean GPT-4 score; rejected = one of the other three at random; fully off-policy | arXiv:2310.16944 §4.1 | verified 2026-09-15 | random rejection chosen "to encourage diversity and make the DPO objective more challenging"; no ablation |
| Llama 3.1 405B | 405B | preference | loss; LR; β; auxiliary NLL coefficient; token masking; rounds | DPO; 1e-5; 0.1; 0.2 on chosen sequences; header and termination tokens masked; 6 rounds | arXiv:2407.21783v3 §4.1.4, §4.1.6 | verified 2026-09-14 ([[llama-3-recipe]]) | DPO preferred over PPO for compute and "instruction following benchmarks like IFEval"; no numbers; NLL term cites Pang et al. 2024 and Pal et al. 2024 |
| Llama 3.1 405B | 405B | preference | pair filter; composition; epochs, batch, steps | only "significantly better"/"better" pairs; General English 81.99%, Coding 6.93%, Multilingual 5.19%, Reasoning and tools 5.89% of comparisons; epochs/batch/steps not printed | v3 §4.2.1, Table 6, §4.1.4 | verified 2026-09-14; epochs/batch/steps not reported | no ablation reported |
| Qwen2.5-Instruct (all 7 dense sizes) | 0.5B–72B | preference | loss; pairs; pair source; epochs; LR; optimizer; β | DPO; ~150,000 pairs; SFT model resamples responses, passing = chosen, failing = rejected; 1 epoch; 7e-7; Online Merging Optimizer; β not given | arXiv:2412.15115v2 §4.2 | verified 2026-09-14 ([[qwen-2.5-recipe]]); β not reported | no ablation reported |
| Tülu 3 8B / 70B | 8B, 70B | preference | loss; β; LR; schedule; warmup; effective batch; max length; epochs | length-normalized DPO; 5; 5e-7 / 2e-7; linear; 0.1; 128; 2,048; 1 | arXiv:2411.15124 Table 20 | verified 2026-09-15 | Table 18 (UltraFeedback, one run per row): DPO-norm β 5, 1 epoch = 57.3 vs plain DPO β 0.1 = 55.2, SimPO 51.8/52.9, SFT base 55.7; Table 19 (70B LR) |
| Tülu 3 405B | 405B | preference | LR; batch; max length; β; warmup; epochs | 2e-7; 256; 2,048; 5; 0.1; 1 | arXiv:2411.15124 Table 34 | verified 2026-09-15 | "we opted to lower the LR for larger models" (§8.1); no ablation at this size |
| Olmo 3 7B Think / 7B Instruct / 32B Think / 32B Instruct | 7B, 32B | preference | loss; β; epochs; warmup; LR; pairs; max length; batch | `dpo_norm`; 5; 1; 0.1; 8e-8 / 1e-6 / 7e-8 / 1e-6; 150,000 / 260,000 / 200,000 / 260,000; 16,384 (7B), 8,192 (32B); 128 | open-instruct `scripts/train/olmo3/*.sh` at d8a7f1c, flags at 00d42c4; arXiv:2512.13961v2 Table 48 | verified 2026-09-14 ([[allenai-olmo3-open-instruct-scripts-recipe]]); 7B Instruct GPU count is a conflict | App. A.6.2: LR and dataset size were swept per model; other settings fixed |
| SmolLM3-3B | 3B | preference | loss; pairs; pair source; context cap; β, LR, batch, epochs | APO (arXiv:2408.06266); Tülu 3 preference mixture plus synthetic reasoning pairs; chosen = Qwen3-32B, rejected = Qwen3-0.6B; 24k tokens; not reported | huggingface.co/blog/smollm3, alignment section | verified 2026-09-15; hyper-parameters not reported (blog checked) | "we also observed higher downstream performance in our internal ablations" — no numbers |
| SmolLM3-3B | 3B | merge | repair of the long-context regression | model soup of APO checkpoints, then a linear merge with weights 0.9 (soup) and 0.1 (a mid-training checkpoint) | same blog, merging section | verified 2026-09-15 | "We were able to recover the base model's RULER score on contexts up to 128k tokens"; no numbers printed |
| Nemotron-4-340B-Instruct | 340B | preference | DPO pairs; epochs; batch; LR; KL coefficient; SFT-loss weight on chosen | 160K; 1; 256; constant, tuned in [3e-8, 3e-7]; tuned in [3e-4, 3e-3]; tuned in [1e-5, 1e-3] (selected values not given) | arXiv:2406.11704 §3.3.2 | verified 2026-09-14 ([[nemotron-4-synthetic-recipe]]) | Table 6: MT-Bench 7.99 → 7.90, GSM8K 87.9 → 88.5 across the DPO stage |
| Nemotron-4-340B-Instruct | 340B | preference | RPO pairs; iterations; LR; η; SFT coefficient; β | 300K per iteration; 3; 3e-7; 1; 1e-5; tuned in [1e-3, 1.0] | arXiv:2406.11704 §3.3.2 | verified 2026-09-14 | Table 6: MT-Bench 7.90 → 8.21 → 8.31 → 8.22; IFEval prompt-strict 61.7 → 78.2 → 79.9 → 79.9; released checkpoint is iteration 3 |
| InstructGPT (PPO-ptx) | 1.3B, 6B, 175B | preference (RM + PPO) | KL coefficient and placement; clip ε; batch; mini-batch; inner epochs; episodes; unique prompts; temperature; EMA; pretraining coefficient γ | β = 0.02 subtracted from the reward; 0.2; 512; 64; 1 inner epoch over 8 mini-batches; 256k; ~31k; 1.0; decay 0.992; 27.8 | arXiv:2203.02155 App. C.4 | verified 2026-09-15 | App. E.7: "the optimal value is around 0.01 and 0.02"; App. E.11: batch 512 best of {64…1024} by human evaluation; mini-batch 32 slightly better than 64, 64 kept for GPU utilization |
| InstructGPT (PPO-ptx) | 1.3B, 6B, 175B | preference (RM + PPO) | policy learning rate | not reported; swept log-linearly from 2.55e-6 to 2.55e-5 (175B: 2.55e-6 and 3.74e-6), final models selected by Likert score | App. C.4, E.9 checked | not reported | E.9: "All runs with learning rate greater than 8.05e-6 diverged, for PPO models without pretraining data mix" |
| Llama 2-Chat | 7B, 13B | preference (RM + PPO) | KL coefficient and placement; LR; batch; mini-batch; clip ε; samples per prompt; iterations | β = 0.01 in the reward, `R = R̃_c − β D_KL(π_θ ‖ π_0)`; constant 1e-6; 512; 64; 0.2; 1; 200–400 with early stopping | arXiv:2307.09288v2 §3.2.3, Eq. 4 | verified 2026-09-14 ([[llama-2-recipe]]) | no ablation reported |
| Llama 2-Chat | 34B, 70B | preference (RM + PPO) | KL coefficient | β = 0.005, same placement | v2 §3.2.3, Eq. 4 | verified 2026-09-14 | no ablation reported |

### RL stage (verifier or rule reward)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeekMath-RL 7B | 7B | RL | algorithm; KL placement and coefficient; LR; samples per question G; max length; training batch; updates per exploration stage; clip ε | GRPO; KL estimator added to the loss, 0.04; 1e-6; 64; 1024 (unit not stated); 1024 (unit not stated); 1; not reported | arXiv:2402.03300v3 §4.1.1, §4.2 | verified 2026-09-14 ([[grpo-recipe]]); clip ε not reported | Fig. 5 (1.3B): GRPO above Online RFT and RFT; prompts are ~144K CoT questions taken from the SFT data so that non-RL benchmarks stay measurable |
| DeepSeek-R1-Zero | 671B MoE (37B active) | RL | algorithm; KL; LR; questions/step; samples/question; samples/step; rollout split; temperature; max length; reference refresh; steps | GRPO; 0.001 in the loss; 3e-6; 32; 16; 512; 8,192 outputs per rollout into 16 mini-batches, 1 inner epoch; 1.0; 32,768 then 65,536 after step 8.2k; every 400 steps; 10,400 steps (1.6 epochs) | arXiv:2501.12948v2 §2.1 | verified 2026-09-14 ([[deepseek-r1-recipe]]) | §2.1: performance and length "exhibit a significant jump at the 8.2k step"; A.3 Fig. 4 compares PPO with GAE λ 0.95 against GRPO on a 16B MoE |
| DeepSeek-R1 (stage-1 RL) | 671B MoE | RL | clip ε; other settings | 10; LR 3e-6, KL 0.001, temp 1, 16 outputs, 32,768, 32 questions/step, batch 512 as above | v2 §3.2.1 | verified 2026-09-14 | §3.2.1: a lower ε "truncates gradients for many tokens and degrades performance"; a higher ε may cause instability; no table |
| DeepSeek-R1 (final RL) | 671B MoE | RL | temperature; steps; reward composition | 0.7; 1,700 steps, with general instruction data and preference rewards only in the final 400; rule + RM + format + language rewards | v2 §3.2.2, Eq. 8–10 | verified 2026-09-14 | §3.2.2: higher temperatures give incoherent generation; B.5 Fig. 6: with the helpful RM, reward rises while Codeforces pass@1 falls |
| Qwen2.5-Instruct | 0.5B–72B | RL | algorithm; samples per query; samples per episode; queries per episode; query ordering; LR, KL, clip ε, temperature, max length, episodes | GRPO; 8; 2,048; 256 (derived: 2,048 ÷ 8); by descending variance of the RM scores within a query's responses; not given | arXiv:2412.15115v2 §4.3 | verified 2026-09-14; derived row marked; remaining settings not reported | ordering aim stated as "to ensure more effective learning"; no ablation |
| Llama-3.1-Tulu-3-8B | 8B | RL | algorithm; KL placement and β; LR; clip ε; γ, λ; effective batch; K; response length; temperature; episodes; EOS penalty; prompts | PPO; β = 0.05 subtracted from the verifier reward; 3e-7 linear, warmup 0.0; 0.2; 1.0, 0.95; 224 (= 7 GPUs × 32); 4; 2,048 (1,024 for GSM8K); 1.0; 100,000 (table) / `--total_episodes 10000000` (script); −10.0; 29,946 verifiable prompts | arXiv:2411.15124 Tables 21–22; open-instruct `docs/tulu3.md` | verified 2026-09-15; episodes is a conflict | report §6.3: β swept over {0.1, 0.05, 0.03, 0.01}, "more KL divergence typically results in lower average scores"; Table 23: MATH 42.0 → 43.7, GSM8K 84.3 → 87.6, IFEval 81.1 → 82.4 over the DPO checkpoint |
| Llama-3.1-Tulu-3-70B | 70B | RL | β; LR; batch; response length; episodes; warmup | 0.07 (caption and script) / 0.7 (prose); 1e-7; 640 (= 40 GPUs × 16); 2,048; 400,000; 0.1 | Table 21 caption, §6.4, `docs/tulu3.md` | conflict (β); verified 2026-09-15 for the rest | Table 23: IFEval 82.6 → 83.2, MATH 62.3 → 63.0, GSM8K unchanged at 93.5 ("already close to saturation") |
| Llama-3.1-Tulu-3-405B | 405B | RL | LR; batch; K; response length; episodes; β; prompts; steps run | 1e-7; 1,856; 1; 1,024; 300,000; 0.05; MATH train only; 75 steps | arXiv:2411.15124 Table 35, §8.1 | verified 2026-09-15; Table 35's caption conflicts with its column header | §8.1: "even with as few as 25 RLVR steps, MATH performance improved by over 5 points"; run stopped early for compute reasons |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | algorithm; KL estimator and β; LR; samples/prompt; rollout and mini-batch; response length; temperature; EOS penalty; schedule | GRPO; `kl3`, 0.01; 5e-7 constant; 16; 4 prompts per process, mini-batch = half the rollout (2 updates); 2,048; 1.0; 0.0; constant | open-instruct `docs/tulu3.md`, legacy `grpo_vllm_thread_ray_gtrl.py` at commit 745bf58d321c | verified 2026-09-15 | the doc preserves the command "for historical reference"; no ablation in the doc |
| Olmo 3 7B Think (released RL run) | 7B | RL | algorithm; β; prompts × samples; minibatches; LR; prompt/response length; temperature; clip low/high; loss | open-instruct GRPO; 0.0; 64 × 8; 1; 1e-6 constant; 2,048 / 32,768; 1.0; 0.2 / 0.272; `dapo` | `7b_think_rl_no_pipeline.sh` at d8a7f1c; arXiv:2512.13961v2 Table 49 | verified 2026-09-14 ([[allenai-olmo3-open-instruct-scripts-recipe]]) | script comment: the released model did not use pipelinerl, "using it should do just as well" (no numbers) |
| Olmo 3 7B Instruct | 7B | RL | prompts × samples; minibatches; response length; active sampling; `no_resampling_pass_rate`; episodes | 64 × 8; 4; 8,192; on; 0.875; 1,024,000 | `7b_instruct_rl.sh` L42–85 | conflict: Table 49 prints dataset size 171,950 against the script's 172,000 | no ablation reported |
| Olmo 3 7B RL-Zero Math | 7B | RL | start; prompts × samples; LR; response length; episodes | `allenai/Olmo-3-1025-7B` (base); 32 × 8; 1e-6 constant; 16,384; 768,000 | `7b_rlzero_math.sh` L4–62 | conflict: Table 49 prints 8 learner and 64 actor GPUs against the script's 16 and 56 | no ablation reported |
| DAPO (Qwen2.5-32B base) | 32B | RL | algorithm; KL; optimizer and LR; warmup; prompts × samples; mini-batch; updates/step; ε_low / ε_high; max generation; overlong shaping; data | GRPO variant; removed; AdamW constant 1e-6; linear over 20 rollout steps; 512 × 16; 512 samples; 16; 0.2 / 0.28; 20,480 (16,384 expected + 4,096 soft cache); linear penalty over the last 4,096 tokens; DAPO-Math-17K, 17K prompts | arXiv:2503.14476v2 §4.1 | verified 2026-09-15 | Table 1 (AIME24 avg@32, one run per row): naive GRPO 30, +overlong filtering 36, +Clip-Higher 38, +soft overlong punishment 41, +token-level loss 42, +dynamic sampling 50; DeepSeek-R1-Zero-Qwen-32B listed at 47 |
| verl DAPO recipe | not scoped | RL | `gen_batch_size`; `train_batch_size`; `max_response_length`; `overlong_buffer.len` / `penalty_factor`; `max_num_gen_batches` | 1536; 512; 20480; 4096 / 1.0; 10 | verl `docs/algo/dapo.md` (fetched 2026-09-14) | verified 2026-09-15; separate row from the paper | documentation example; no run link and no ablation |
| Skywork-OR1-7B | 7B | RL | LR; clip ε; target entropy; temperature; KL; stages (steps / context / batch / mini-batch / group); released checkpoint | 1e-6 constant; 0.2; 0.2 with adaptive coefficient; 1.0; none; stage 1 = 0–660 / 16K / 256 / 256 / 16, stage 2 = 660–1320 / 32K / 160 / 160 / 32; step 1320 | arXiv:2505.22312v2 §8.1, Table 11 | verified 2026-09-15 | §3.2.6: "The KL penalty hinders further improvements in test performance during multi-stage training"; §4: on-policy training mitigates entropy collapse |
| Skywork-OR1-32B | 32B | RL | stages; released checkpoint | stage 1 = 0–760 / 16K / 256 / 256 / 16, stage 2 = 760–1130 / 24K / 160 / 160 / 32; step 1000 | v2 Table 12 | verified 2026-09-15 | Table 13: AIME24 82.2, AIME25 73.3, LiveCodeBench 63.0 |
| Skywork-OR1 (all) | 7B, 32B | RL | prompt filter | model-specific difficulty 0–16; the released 32B script keeps 1 ≤ difficulty ≤ 15; source pool 105,055 math and 14,057 code problems | dataset card; `skywork-or1-filter-32b.py` | verified 2026-09-15 | dataset card: "For each model variant, we excluded problems with difficulty values of 0 and 16 specific to that model" |
| Magistral Medium | not reported | RL | algorithm; KL; ε_high; ε_low; batch schedule; non-penalized length schedule; LR, G, steps | GRPO with no KL, group-token loss normalization, zero-advantage groups removed; 0.26–0.28 adjusted during training; not reported; 8k → 4k → 2k sequences; 16k → 24k → 32k; not reported | arXiv:2506.10910v1 §2.1, §5.2 | verified 2026-09-14 ([[magistral-recipe]]); several values not reported | §6.4 Fig. 7: no significant difference between minibatch, group and no advantage normalization; §6.3 Fig. 6 (3B): reward degrades with more than 2 minibatches per batch |
| Magistral Small | 24B | RL | ε_high; batch; non-penalized length; temperature | 0.3; 2,048 sequences; 32k; 1.0 | v1 §5.3 | verified 2026-09-14 | ε_high raised because the cold-started model had "far lower entropy"; temperature 1.0 stated as the "best balance"; no numbers |
| Phi-4-reasoning-plus | 14B | RL | framework; global batch; G; optimizer and LR; warmup; KL target, placement and β; entropy coefficient; max length; clip; steps; seed data | verl; 64 problem seeds per iteration on 32 H100s; 8; Adam 5e-8; cosine over the first 10 steps; `D_KL(π_θ ‖ π_θold)` in the loss, 0.001; 0.001; 32k with outputs clipped at 31,744; ε not reported; 90; 72,401 math problems, ~6,400 seen | arXiv:2504.21318v1 §4, §4.2 | verified 2026-09-15 | §4.2: 90 steps selected by best AIME 2024 score; "additional GRPO training for only 90 steps boosts AIME performance by more than 10% (Figure 7a). Further training for more steps does not translate to additional gains" |
| Phi-4-reasoning-plus | 14B | RL | reward | length-aware cosine accuracy: +0.5 to +1.0 correct (penalty beyond L_pos = 25,600 of L_max = 31,744), −1.0 to −0.5 incorrect (penalty lighter beyond L_neg = 3,702); −0.5 for a missing `<im_end>`; −1.0 for a malformed `<think>` block; 5-gram repetition penalty; `R = (8/13)·R_acc + (1/13)·R_rep` | v1 §4.1 | verified 2026-09-15 | stated aim: concise when correct, longer when incorrect; no ablation of the weights |
| AceReason-Nemotron 7B / 14B | 7B, 14B | RL | algorithm; importance weight; KL and entropy coefficients; batch; G; LR; length stages; prompt filter | GRPO with `ρ = 1` fixed (one update per generation); β = 0, entropy 0; 128 prompts; 8 at 8K, 16 otherwise; 1e-6 AdamW (math), 5e-6 (code); 8K → 16K → 24K → 32K; drop prompts with pass rate > 6/16 at 24K and 32K | arXiv:2505.16400 §3.2.2, §3.3.2 | verified 2026-09-15 | §3.2.2 Fig. 3c: 2 or 4 updates per generation collapsed entropy around step 100; Fig. 3b: starting directly at 16K or 24K was worse; Table 3: the pass-rate filter "significantly improves model performance" |
| MiMo-7B-RL | 7B | RL | batch; mini-batch; updates/step; LR; max sequence length; temperature; top-p; KL; dynamic sampling; easy-pool probability; clip values; G | 512; 32; 16; 1e-6; 32,768; 1.0; 1.0; removed; on; α = 10%; not reported; not reported | arXiv:2505.07608v2 §3.3.2, §3.3.3 | verified 2026-09-15; ε and G not reported | §3.3.2: re-using easy data directly "introduces significant instability in policy updates"; the 10% pool is the stated fix; no numbers |
| SimpleRL-Zoo (10 open base models) | 0.5B–32B | RL | prompts/step; samples/prompt; mini-batch; max rollout length; temperature; clip ε; KL placement and coefficient; LR | 1,024; 8; 256; 8,192; 1.0; 0.2; in the loss, 1e-4 for 0.5B–14B and 1e-3 above 14B; not reported | arXiv:2503.18892v3 App. B.5 | verified 2026-09-15; LR not reported | §3.2 Fig. 7: the same configuration collapses Mistral-7B on MATH levels 3-5 and understretches Qwen2.5-7B on easy data |
| ScaleRL (8B dense; Llama-4 Scout 17B×16) | 8B; 17B×16 MoE | RL | recipe components; sequence budget; batch; compute | PipelineRL asynchrony, forced length interruption, CISPO loss, prompt-level loss averaging, batch-level advantage normalization, FP32 logits, zero-variance filtering, no-positive-resampling; 16,384 tokens (12,288 thinking + 2,048 solution); 768 default, 512 and 2,048 studied; 100,000 GPU-hours (8B) and 50,000 (Scout) | arXiv:2510.13786v1 §1, §2, §4, §5 | verified 2026-09-15 | §4 leave-one-out at 16k GPU-hours per variant; Table 1 fitted (C_mid, B, A); Fig. 8a: ±0.02 error margin on A from three repeated runs |

### Starting point for a small general-purpose run

Every number here is a `verified` row above, with the conditions of the run that produced it.

For a **preference stage** on a 7B–8B SFT checkpoint with on-policy or mixed pairs: length-normalized DPO,
β = 5, 1 epoch, linear schedule with warmup ratio 0.1, effective batch 128 pairs, maximum sequence length 2,048,
learning rate 5e-7. That is the Tülu 3 8B configuration, selected on UltraFeedback over an early Tülu 3 SFT
checkpoint and then run on a mixture of more than 270k pairs (Tables 18, 20). If the pairs are fully off-policy
and the loss is plain DPO instead, the verified alternative is Zephyr's β = 0.1, peak LR 5e-7 linear with 10%
warmup, global batch 32, on Mistral-7B-v0.1 with UltraFeedback. Do not carry β across the two rows. If chosen
log-probabilities fall during training, the two anchors with verified coefficients are Llama 3.1's NLL term at
0.2 on chosen sequences (405B) and Nemotron-4's SFT loss weight of 1e-5 in RPO (340B); neither was printed with
an ablation.

For an **RLVR stage** on a 7B model with a binary verifier: the Olmo 3 7B script values — β = 0.0, 64 prompts ×
8 samples, learning rate 1e-6 constant, temperature 1.0, clip 0.2 / 0.272, response length matched to the model
type (8,192 for an instruct model, 32,768 for a thinking model) — on 8–16 learner GPUs with 32–56 vLLM engines.
If a KL term is wanted, the verified small-model value is SimpleRL-Zoo's 1e-4 in the loss for models of 0.5B–14B.
For the prompt set, the two verified filters are dropping all-correct and all-wrong groups (DAPO, MiMo-7B,
Skywork-OR1) and excluding prompts above a pass-rate threshold measured with the policy itself
(AceReason-Nemotron at 6/16; Skywork-OR1 by a per-model difficulty score). Neither a learning rate for
SimpleRL-Zoo nor a group size for MiMo-7B is available, so those gaps must be filled from a different verified
row rather than assumed.

Compute conditions to state alongside any of these: Tülu 3's 8B DPO ran 10 hours on 8 H100s and the 70B DPO 19
hours on 64; Olmo 3's post-training used 256 H100s; DAPO, Skywork-OR1, Phi-4-reasoning and MiMo-7B all ran on
verl; and ScaleRL's ablations cost 16,000 GPU-hours per variant at 8B.

## Generalization lens

**(a) What increases breadth.**

- A verifier reward on one domain transferred to another in the one place it was measured: math-only RL moved
  LiveCodeBench v5 from 37.6 to 44.4 at 7B and from 53.1 to 58.9 at 14B, "across all problem topics—not just
  math-related coding tasks", while a math-only SFT model of comparable strength scored 19.3
  ([[acereason-nemotron]] Table 1, Fig. 4) (**Result (single study)**).
- Mixing a general RL stage after a reasoning stage recovered non-reasoning ability: DeepSeek-R1's Dev2 → Dev3
  step moved AlpacaEval 2.0 from 55.8 to 62.1 and Aider-Polyglot from 25.6 to 44.8
  ([[deepseek-r1-recipe]], v2 Table 3).
- Keeping non-RL benchmarks measurable by construction: DeepSeekMath excluded the rest of its SFT questions from
  the RL prompt set specifically so that the effect of RL on benchmarks absent from RL data could be read
  ([[grpo-recipe]] §4.2). This is a measurement design, and it is cheap.
- Repairing a stage-induced regression by merging instead of retraining: SmolLM3's 0.9 / 0.1 linear merge of the
  APO soup with a mid-training checkpoint recovered the base model's RULER score to 128k ([[smollm3-apo]]).

**(b) What causes narrowing or forgetting.**

- A preference stage trained only on short sequences narrows long-context behaviour when the SFT model is not
  already strong there. SmolLM3 states the cause directly: "the APO training data was limited to 24k tokens".
  Llama 3.1 ran DPO on short-context data only and reports no long-context damage, but attributes that to the SFT
  model already being strong on long context ([[llama-3-recipe]] §4.3.4) — the two are consistent and the
  condition is the SFT checkpoint, not the stage.
- Optimizing against a learned reward beyond a point: DeepSeek-R1's final-stage helpfulness RM produces rising
  reward with falling Codeforces pass@1 ([[deepseek-r1-recipe]] B.5 Fig. 6). Tülu 3's β sweep shows the same
  shape from the KL side: "more KL divergence typically results in lower average scores".
- Entropy collapse from off-policy inner updates ([[acereason-nemotron]] Fig. 3c) and from low sampling
  temperature ([[skywork-or1]] §3.2.4) both reduce the diversity of what is sampled, which is the mechanism by
  which pass@k at large k falls while pass@1 rises (ch-45 §"rlvr-beyond-base-model").
- A prompt set mismatched to the base model: Mistral-7B on MATH levels 3-5 shows "a significant increase in
  response length without any corresponding improvement in accuracy" ([[simplerl-zoo]] §3.2).

**(c) How to measure generality for these two stages.**

- Preference stage: evaluate on a panel that includes at least one benchmark none of the pairs targeted. Tülu 3's
  Table 18 is an average across development evaluations, which is why a 1.1-point difference between variants
  should not be read as decisive; its final tables separate development from held-out ("unseen") suites.
- RL stage: report pass@k at a large k alongside pass@1, and evaluate the domains the reward never covered.
  AceReason-Nemotron and DeepSeek-R1 both do this; DAPO, MiMo-7B and Phi-4-reasoning-plus report no benchmark
  outside mathematics and code.
- Both: state the compute budget. ScaleRL's Fig. 10 crossover means a comparison at one budget can reverse at
  another, so "recipe X beat recipe Y" without a budget is not a claim that can be checked ([[scalerl]] §5).
- Decontamination is part of the measurement, not a separate chore: Tülu 3, Qwen2.5 (LCS-based filter, §5),
  DeepSeek-R1 (10-gram filter, D.1) and Skywork-OR1 (removal of problems similar to AIME 24/25 and
  LiveCodeBench) all describe one, and the Skywork dataset card notes that an early release briefly carried
  incorrect difficulty fields — a reminder that released artefacts have versions.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reading a "batch" value in the wrong unit | The configured run does 1 gradient update per rollout where the source did 16, or 16 where the source did 1; wall-clock per step is far from the source's | Compute prompts × samples ÷ mini-batch and compare with the updates-per-step number the source prints, as in §1.1 |
| Carrying β across loss variants | Training either barely moves (β = 0.1 under length normalization) or the margin saturates immediately (β = 5 without it) | Print the mean bracketed log-ratio difference before and after normalization; the §2.2 example gives 10.0 versus 0.035 for the same pair |
| Treating a KL-in-loss coefficient as a KL-in-reward coefficient | The KL curve is an order of magnitude away from the source's, with no error | Locate the equation, not the number: KL-in-reward appears inside `r(x,y)`, KL-in-loss appears as a separate additive term |
| Using a KL to `π_θold` as a retention control | KL to the SFT checkpoint grows without bound while the logged KL stays small | Log both KLs separately; Phi-4-reasoning-plus's β = 0.001 is against `π_θold` ([[phi-4-reasoning-rl]] §4.2) |
| Copying a launch script's `total_episodes` as the run length | Reported steps are far larger than the source's evaluation cadence implies | Compare against the paper's table and against the checkpoint-selection rule; Tülu 3's script budget is 10,000,000 against a table value of 100,000 |
| Copying a framework default that changed | A flag no longer printed in a script silently takes a new default after a refactor | Pin the commit; Olmo 3's defaults changed at commit 7917d41 from `dpo`/0.1/2 epochs/0.03 warmup to `dpo_norm`/5.0/1/0.1 ([[allenai-olmo3-open-instruct-scripts-recipe]]) |
| Assuming a prompt filter transfers between model sizes | The filtered set is degenerate — nearly all groups all-correct or all-wrong — and the effective batch collapses | Log the share of zero-advantage groups per step; Skywork-OR1's difficulty score is per model variant for exactly this reason |
| Comparing two recipes at one compute budget | The ranking reverses when either run is extended | Fit the §5 curve on the first half and extrapolate; ScaleRL's batch-512 and batch-2048 rows cross between 16k and 100k GPU-hours |
| Quoting a reconstruction as the original | A "DeepSeek recipe" or "DAPO recipe" number that no DeepSeek or ByteDance document contains | Check whether the figure's appendix describes a re-implementation; [[scalerl]] App. A.16 does |
| Citing a duplicate card as evidence | The locus in the citation points to a redirect page rather than a source | `deepseekmath` redirects to [[grpo]]/[[grpo-recipe]] and `nemotron` to [[nemotron-4-synthetic]]; follow the redirect |

## Check your understanding

1. DAPO prints "prompt batch size is 512" and "mini-batch size is set to 512". Explain, using the printed
   sentence about gradient updates, why these two 512s cannot both be prompts, and state what would go wrong in a
   reproduction that read them as the same unit.
2. Tülu 3's DPO β is 5 and Zephyr's is 0.1. Construct a pair of responses for which both configurations give
   approximately the same loss, and explain why no single "correct β" exists across the two.
3. InstructGPT applies its KL penalty inside the reward; DeepSeek-R1-Zero applies it inside the loss. Derive one
   consequence of this difference for what happens when the advantage estimates are normalized within a group.
4. AceReason-Nemotron takes exactly one gradient update per generation; DAPO takes sixteen. Both report good
   results. Give the mechanism by which the second can be stable, and name the component in DAPO's configuration
   that does the work the first achieves by construction.
5. A prompt at pass rate 7/8 gives an incorrect sample an advantage of −2.646 and a correct sample +0.378. Explain
   why AceReason-Nemotron's filter removes prompts *above* a pass rate rather than below it, and what would happen
   to the entropy if it removed the opposite end instead.
6. ScaleRL's batch-512 and batch-2048 fits cross between 16k and 100k GPU-hours. Given only the three fitted
   parameters per run, explain how you would decide which to use for a project with a fixed 30,000-GPU-hour
   budget, and what additional quantity you would need to compare absolute rather than relative gains.
7. SmolLM3's long-context regression was traced to two causes, only one of which is in the preference stage.
   Explain why the 0.9 / 0.1 merge repairs the symptom, and what measurement would tell you whether the
   preference-stage cause still needs fixing.
8. Three sources in this chapter print a value for a run that their own released artefact contradicts. Pick one
   and argue, from the nature of the artefacts, which value belongs in a row describing the released checkpoint.

## Connections

- **Previous:** ch-45 — Self-Improvement Loops and Multi-Stage Reasoning Pipelines. Supplies the pipeline shapes
  (reasoning RL → rejection-sampling SFT → general RL) whose individual stages are tabulated here.
- **Dependency:** ch-39 — Offline Preference Optimization: DPO and Its Variants. Supplies the DPO objective, its
  variants, and the reference-model mechanics that §2 scopes to named runs.
- **Next:** ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. Takes the RL
  columns of §3 into settings where a rollout contains environment observations, which adds a masking column that
  none of the single-turn rows above carries.

## Sources

Library cards:

- [[llama-3-recipe]] — Llama 3.1 405B DPO row: learning rate, β, NLL coefficient, token masking, round structure.
- [[qwen-2.5-recipe]] — Qwen2.5 DPO row (pairs, Online Merging Optimizer) and the GRPO row with its derived
  queries-per-episode value.
- [[tulu-3]] — the report's pipeline framing; its Technical Details block is superseded on numbers by the primary
  tables quoted in [[tulu-3-hyperparameter-tables]].
- [[rlvr-tulu3]] — RLVR as a method; cited here only for the verifier definition, since its hyper-parameter
  paragraph does not match Table 21.
- [[tulu-3-1]] — what changed and did not change between Tülu 3 and the 3.1 refresh.
- [[olmo-3]] — model-flow framing for the Olmo 3 rows.
- [[allenai-olmo3-open-instruct-scripts-recipe]] — every Olmo 3 DPO and RL value, the framework defaults, and the
  script-versus-table conflicts.
- [[smollm-3]] — the SmolLM3 release and its stage structure.
- [[nemotron-4-synthetic]] and [[nemotron-4-synthetic-recipe]] — DPO-plus-SFT-term and the three RPO iterations,
  with per-stage Table 6 numbers. (The `nemotron` card is a duplicate redirect and is not cited as evidence.)
- [[rlhf-instructgpt]] — the three-stage RLHF template; its canonical-hyperparameter table is superseded on
  numbers by [[instructgpt-ppo-appendix]].
- [[llama-2-recipe]] — Llama 2-Chat PPO rows, including the per-size KL coefficients.
- [[hf-dpo-zoo]] — the map of preference-loss variants referenced in §2.2.
- [[dpo]] and [[simpo]] — the objective and the length normalization used by the DPO-norm rows.
- [[grpo]] and [[grpo-recipe]] — DeepSeekMath's GRPO run, including the unit-less batch and the values the paper
  does not print. (The `deepseekmath` card is a duplicate redirect and is not cited as evidence.)
- [[deepseek-r1-recipe]] — R1-Zero, R1 stage-1 and R1 final-stage RL rows.
- [[magistral-recipe]] — Magistral Medium and Small RL rows and the settings the paper leaves out.
- [[phi-4]] — the Phi-4 family framing; its reward description is corrected by [[phi-4-reasoning-rl]].

Chapter excerpts (primary text read for this chapter):

- [[instructgpt-ppo-appendix]] — App. C.3, C.4, E.7, E.9, E.11 verbatim, and the list of values the paper does
  not contain.
- [[tulu-3-hyperparameter-tables]] — Tables 18–22, 34, 35 and the open-instruct launch commands for 8B, 70B, 405B
  and Tülu 3.1.
- [[zephyr]] — pair construction, DPO settings, and the overfitting observation.
- [[dapo]] — §4.1 training details and Table 1.
- [[verl-dapo-recipe]] — the released configuration for the same recipe shape, including `gen_batch_size`.
- [[skywork-or1]] — shared settings, the per-stage tables, the zero-advantage filter, and the difficulty filter.
- [[acereason-nemotron]] — strict on-policy updates, length staging, the pass-rate filter, and the cross-domain
  transfer table.
- [[mimo-7b]] — batch and mini-batch in one unit, dynamic sampling, and the easy-pool resampling probability.
- [[phi-4-reasoning-rl]] — the 90-step GRPO stage, its KL target, and the actual reward shape.
- [[smollm3-apo]] — APO pair construction, the 24k context cap, and the merge that restored RULER.
- [[scalerl]] — the sigmoid fit, the A-versus-B ablations, and the appendix that scopes the recipe comparison.
- [[simplerl-zoo]] — one configuration across ten base models, and the difficulty-matching finding.
