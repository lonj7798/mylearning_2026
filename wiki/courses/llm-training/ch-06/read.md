<!-- chapter: ch-06
     track: foundations
     kind: content
     title: Checkpointing, In-Loop Evaluation, and Checkpoint Selection
     deps: [ch-05]
     sources: [[early-stopping-and-checkpointing]], [[paloma]], [[olmes]], [[signal-and-noise-eval]], [[datadecide]], [[merging-in-pretraining]], [[model-soups]], [[wise-ft]], [[llama-3]], [[tulu-3]], [[olmo-2]], [[olmo-3]], [[olmo-core-olmo3-configs]], [[smollm3-training-configs]]
     figures: figures/checkpoint-state.html, figures/checkpoint-selection.html
     revised: 2026-09 (generality revision)
-->

# Chapter 6 — Checkpointing, In-Loop Evaluation, and Checkpoint Selection

> **Core insight.** A training run does not produce one model; it produces a sequence of checkpoints, and the
> choice of which checkpoint (or which average of checkpoints) is released changes measured capability by more
> than most hyperparameter choices at this stage. Two measurements decide that choice well: per-domain held-out
> loss, which exposes fit gaps that a single validation loss hides — the C4-only 1B baseline in [[paloma]]
> reaches perplexity 391,171 on the RedPajama arXiv domain (§4.1) — and downstream evaluation run under a fixed
> format with a known signal-to-noise ratio, since in [[signal-and-noise-eval]] a benchmark's small-scale SNR
> predicts whether a small-scale ranking transfers to a larger model (R² = 0.626, §4.1).
>
> **Guideline.** When a run is long enough that it will be resumed, save the data-loader position, RNG state,
> step counter, and scheduler state alongside model and optimizer state, and check the loader position against
> a dataset fingerprint on load ([[olmo-core-olmo3-configs]]). When a checkpoint must be selected, prefer an
> average over the last several checkpoints to the single best one, because in [[signal-and-noise-eval]]
> averaging raised 30-task decision accuracy from 68.9% to 71.3% and lowered 13B scaling-law prediction error
> from 1.03 to 0.86 absolute points (§5.2), and in [[olmo-2]] a 3-checkpoint soup equalled or beat the best
> single mid-training run on all six mixes tested (§4.5, Table 14). When a stage is fine-tuning rather than
> pre-training, stop on a held-out suite that was never used for development decisions, because in [[tulu-3]]
> every model tested scores higher on IFEval than on the structurally matched IFEval-OOD (Tülu 3 8B: 82.4 vs
> 24.3, §7.4.1, Table 31).

---

## Why this chapter matters for a general-purpose model

[[ch-05]] settled how many tokens a step actually sees. This chapter covers what happens between steps: what is
written to disk, what is measured, and how the measurement decides which weights become the released model.
The stage is pre-training and mid-training for §1–§5 and §7 and SFT for §6, but the mechanism recurs later —
[[tulu-3]] selects RLVR checkpoints by the same procedure it uses for SFT.

A model is generally capable when it performs on tasks and domains that were not targeted. Every part of the
measurement loop can hide a loss of that property:

1. A single held-out loss averages over domains, so a fit gap on one domain is invisible in the average.
2. A downstream benchmark whose score moves by more between adjacent checkpoints than between two candidate
   recipes cannot rank the recipes, however many instances it has.
3. A development suite that is used to make decisions stops measuring generalization once enough decisions
   have been made against it.
4. Selecting the single checkpoint with the best score on a development suite selects partly for capability
   and partly for the noise of that suite.

§2 through §6 address these in order. §1 and §7 cover the mechanical prerequisite: a resume that silently
changes the data order invalidates every measurement above it.

---

## §1 What a resumable checkpoint contains

**Definition.** A resumable checkpoint is the state that lets a stopped run continue the same trajectory:
model parameters, optimizer state, and the bookkeeping state that determines what the next step will see.

**The measurable problem.** A resume that restores parameters and optimizer state but not the bookkeeping
state produces no error and no visible jump in the loss curve. It changes which data the model sees next, so
the tokens-seen ledger from [[ch-05]] and every per-domain measurement in §2 no longer describe the run.

**Mechanism.** OLMo-core separates the two halves. Model and optimizer state are written by handing the train
module to the checkpointer; the rest is an ordinary Python dictionary
(`src/olmo_core/train/trainer.py` L87–96, [[olmo-core-olmo3-configs]]):

```python
class TrainerStateDict(TypedDict):
    global_step: int
    global_train_tokens_seen: int
    global_train_petaflops: float
    max_steps: Optional[int]
    data_loader: Dict[str, Any]
    epoch: int
    world_size: int
    rng: Dict[str, Any]
    callbacks: Dict[str, Dict[str, Any]]
```

The data-loader entry is itself a dictionary (`src/olmo_core/data/data_loader.py` L441–449):

```python
def state_dict(self) -> Dict[str, Any]:
    return {
        "dataset_fingerprint_version": self.dataset.fingerprint_version,
        "dataset_fingerprint": self.dataset.fingerprint,
        "batches_processed": self.batches_processed,
        "tokens_processed": self.tokens_processed,
        "seed": self.seed,
        "epoch": self._epoch,
    }
```

Three properties of this code are worth reading carefully, because each is a condition under which a resume is
not equivalent to not having stopped.

1. **The dataset fingerprint is checked; the seed in the checkpoint overrides the seed in the config.** On load,
   a differing `dataset_fingerprint` raises `RuntimeError` ("Dataset fingerprint does not match the fingerprint
   in the checkpoint … This will probably result in a different data order!") unless
   `ignore_fingerprint_mismatch=True`, which downgrades it to a warning with the same text (L451–467). A
   differing `seed` is handled the other way round: the loader warns ("Restoring data loading state with a
   different data seed, will use data seed from state dict for data order consistency.") and then assigns
   `self.seed = state_dict["seed"]` (L469–473). Changing the data seed in the config at resume time therefore
   does not change the data order; the fingerprint is the only one of the two that can stop the run.
2. **RNG restoration is conditional on world size.** `trainer.py` L838–848 restores per-rank RNG only when
   `state_dict["world_size"] == get_world_size()`; otherwise it logs "Trainer will not restore rank RNG states
   since the RNG states in the checkpoint were saved with a different world size." Resuming a run on a
   different number of GPUs therefore gives up exact reproduction of the RNG stream by design.
3. **Position is stored in tokens and recomputed into batches.** The fixed-sequence-length loader recomputes
   `self.batches_processed = self.tokens_processed // self.global_batch_size` on load (L757–764), so a resume
   with a different global batch size lands on the nearest batch boundary rather than failing.

**Worked numeric example.** A run has `global_batch_size = 4,194,304` tokens (8,192 × 512, the Olmo 3 7B
setting) and stops after `tokens_processed = 419,430,400` tokens, that is 100 batches. Resume it with the batch
size doubled to 8,388,608 tokens. Then `batches_processed = 419,430,400 // 8,388,608 = 50`. The loader reports
the correct token position, and the step counter restored from `global_step` does not match it: the run
continues from token 419,430,400 but at step 100 rather than step 50. Any schedule expressed in steps is now
offset relative to the token ledger, which is why [[ch-05]]'s rule — fix the unit before the run — applies at
resume time as well as at launch time.

**A resume that changes the schedule on purpose.** Olmo 3 7B pre-training ran as two scripts. The first used
`CosWithWarmup(warmup_steps=2000)` with `max_duration=Duration.tokens(int(5e12))`. The second resumed from
`step596047` with `load_trainer_state=True, load_optim_state=True` and
`max_duration=Duration.tokens(int(7e12))  # Changed from 5T -> 7T`, replacing the scheduler with
`HalfCosWithWarmup(warmup_steps=original_max_steps // 2 + original_warmup_steps // 2)`, commented in the file
as "Scheduler updated to extend lr from where we left off" ([[olmo-core-olmo3-configs]]). The paper describes
the result: "The first half of the learning rate schedule is a cosine schedule over 5T tokens. We stretch the
second half of the schedule to reach a target length of one epoch (5.93T tokens)" ([[olmo-3]],
arXiv:2512.13961 Fig. 3 caption). The 7T in the script is the horizon the schedule is drawn over, not the
stop point: the run stopped at one epoch of the corpus, 5.93T tokens. The scheduler state and the scheduler
definition are different facts: the checkpoint carries the step count, the script carries the shape.

**Configurable, not automatic.** SmolLM3's nanotron configs set `load_optimizer: true` and
`load_lr_scheduler: true` as independent switches next to `checkpoint_interval: 2000`
([[smollm3-training-configs]]). Both being true is what makes a resume a continuation; either being false makes
it a new run initialized from old weights.

The interactive inventory in [figures/checkpoint-state.html](figures/checkpoint-state.html) lets you toggle each
state component off and read what the code above does in that case, with the file and line for each entry.

**Implication for a general-purpose model.** The checkpoint's bookkeeping half is what makes the mixture
proportions you chose the mixture proportions the model actually saw. A loader reset after a crash re-serves
the beginning of the shuffled order, which over-represents whatever sources appear early in it; §7 gives the
detection method.

---

## §2 In-loop measurement I: per-domain held-out loss

**Definition.** Per-domain held-out loss evaluates the model on several disjoint validation sets, one per data
source or domain, and reports them separately rather than as one number.

**The measurable problem.** [[paloma]] (Magnusson et al., arXiv:2312.10523, v1 2023-12) states it as a fit
question: perplexity on one held-out set is a micro-average dominated by whichever domain contributes the most
sampled tokens, so a model can improve on the aggregate while getting worse on specific domains.

**Formula.** Paloma's macro average over a domain set D is

  macro-ppl(D) = |D|⁻¹ · Σ_{d ∈ D} perplexity(d)

where perplexity(d) = exp(−ℓ_d / T_d), ℓ_d is the summed log-likelihood over the documents of domain d, and
T_d the token count of domain d (§3, §4.1). When vocabularies differ between the models being compared, Paloma
uses bits per byte instead: BPB = −ℓ / (B · ln 2), with B the number of UTF-8 bytes (App. B).

**Worked numeric example.** Two domains, 100 tokens each. Model A: summed NLL 250 nats on domain 1 and 250 on
domain 2, so perplexity 12.18 on each and macro average 12.18. Model B: 150 nats on domain 1 and 400 on domain
2, giving perplexity 4.48 and 54.60, micro average over the 200 pooled tokens
exp(550/200) = 15.64, macro average (4.48 + 54.60)/2 = 29.54. Pooling the tokens makes B look 28% worse than
A; the macro average makes it 142% worse, which is the statement that matters if domain 2 is a capability you
intend to keep.

**Evidence.** Paloma trains six 1B baselines that differ only in pre-training corpus, holding architecture,
token budget, tokenizer, decontamination, and data order fixed (§4.1, App. G: 1B, 2048 sequence length, LionW,
peak LR 2.0e-4, cosine to 70k steps, stopped at 35k steps ≈ 150B tokens). Reported results:

- The C4-only baseline reaches perplexity 391,171 on the RedPajama arXiv domain and 14 on Dolma peS2o (§2, §4.1).
- Between the ~20B-token and ~150B-token checkpoints, the C4 and mC4-en baselines get **worse** on 65 and 43
  domains respectively; outside those two baselines, only 6 model-domain pairs worsen (App. D.1.1). A single
  aggregate loss curve for those runs is monotone decreasing over the same interval.
- Perplexity does not rank models the way downstream tasks do: Spearman ρ between the ranking by a source's
  perplexity and by a task's accuracy includes −0.77 (c4-en vs HellaSwag) and 0.94 (RedPajama vs HellaSwag)
  over the six baselines (App. A, Table 3). **Result (single study).**

**Conditions and limits stated by the source.** Paloma covers English and code only; code sources are not
decontaminated; fringe-source perplexity tracks document length; and evaluation must score documents
separately rather than concatenated — Pythia 1.4B at 2B tokens on 4M evaluation tokens gives 92.23 ± 17.33
concatenated against 42.57 ± 0.29 separate (App. H, Table 17). That variance difference is large enough to
swamp the effect being measured.

**What this looks like in a real training loop.** The Olmo 3 7B scripts attach a held-out-loss callback with
its own multi-domain dataset, `LMEvaluatorCallbackConfig(eval_dataset=... DataMix.v3_small_ppl_validation ...,
eval_interval=10_000)` ([[olmo-core-olmo3-configs]]), so held-out loss is measured every 10,000 steps against a
named validation mixture rather than a single file.

**Implication for a general-purpose model.** Per-domain loss is the cheapest in-loop signal that separates "the
model is improving" from "the model is improving on the domains that dominate the mixture." It catches
narrowing before any benchmark does, because it needs no task format and no emergent ability.

---

## §3 In-loop measurement II: downstream evaluation under a fixed format

**Definition.** In-loop downstream evaluation scores intermediate checkpoints on task benchmarks during
training, under a formatting and scoring procedure fixed in advance.

**The measurable problem.** [[olmes]] (Gu et al., arXiv:2406.08446, v1 2024-06) documents that the same model
on the same dataset produces different published numbers depending on the setup. Its Table 1 lists
ARC-Challenge scores for Llama2-7B between 43.2 and 53.7 across six references, and for Llama3-8B 60.2 on the
Hugging Face Open LLM Leaderboard against 78.6 in the Llama 3 model card. The difference is formulation, shot
count, and normalization, not model quality.

**Mechanism, step by step.**
1. Choose the **formulation**. In MCF (multiple-choice formulation) the options appear in the prompt with
   letter labels and the model is scored on the label token. In CF (cloze formulation) each answer string is
   scored separately by its own probability.
2. Choose a **normalization** for CF: `none` = ln P(a|q); `character` = ln P(a|q) / characters(a);
   `pmi` = ln[P(a|q) / P(a|u)] with the uninformative prompt u = "Answer:".
3. Fix the prompt, the split, the sample (1,000 instances when the split exceeds 1,500,
   `Random(1234).sample(...)`), and fixed curated 5-shot examples.
4. Report the better of MCF and CF (§3.4).

**Why the formulation matters during training and not only at release.** [[olmes]] Fig. 1 evaluates
OLMo-7B-0424 on the MMLU validation set in both formats across training: MCF stays near random until about
400B training tokens, then becomes the stronger signal, while CF gives signal from early on and levels off
late. A run instrumented only with MCF sees a flat line for its first 400B tokens. Across 15 base models on
ARC-Challenge, the weakest 8 are near random under MCF but above random under CF (§3.4, Fig. 2).
**Result (single study); the effect is a property of the models, not of the benchmark.**

**Evidence that the choice is made deliberately in production.** The Olmo 3 7B downstream callback runs 26
tasks every 10,000 steps ([[olmo-core-olmo3-configs]]): twelve OLMES tasks as paired `*_bpb_5shot` and
`*_mc_5shot_fast` variants of ARC-Challenge, ARC-Easy and the four MMLU splits, plus `hellaswag_bpb_5shot`
without an MC counterpart; six `basic_skills_*_rc_5shot` tasks; six generative tasks scored in bits per byte
(`codex_humaneval_gold_bpb_3shot`, `codex_mbpp_gold_bpb_3shot`, `minerva_math_500_gold_bpb_0shot`,
`mt_mbpp_{cpp,java,rust}_gold_bpb_3shot`); and `copycolors_10way_fast`, commented in the file as "Sanity check
for MCQA ability". The paired BPB and MC variants implement OLMES's "evaluate with both and use the better one"
as two metrics logged side by side.

**Cadence is a cost decision, and the costs differ by framework.** OLMo-core evaluates in-process every 10,000
steps and checkpoints every 1,000 ([[olmo-core-olmo3-configs]]). SmolLM3 checkpoints every 2,000 steps and
launches evaluation as a separate Slurm job that reads the written checkpoint, at `eval_interval: 4000` in
stages 1–2, 6000 in stage 3, and 400000 against `train_steps: 20000` in the long-context stage, which disables
it ([[smollm3-training-configs]]). The asynchronous design removes evaluation cost from training throughput at
the price of measuring a checkpoint rather than the live model.

**Implication for a general-purpose model.** The task list is the operational definition of "generally capable"
for that run. Olmo 3's list spans multiple-choice STEM and non-STEM, generative QA, math, code and code
infilling so that a data change targeted at one capability shows its cost on the others ([[olmo-3]] §3.3.1).

---

## §4 Signal, noise, and which benchmarks can decide anything at ablation scale

**Definition.** For a benchmark and a model scale, **noise** is the variability of the score across adjacent
checkpoints of one run, and **signal** is the spread of scores across a population of models trained with
similar compute. Their ratio is the signal-to-noise ratio, SNR.

**The measurable problem.** Decisions at this stage are made with small runs. If the score difference between
two candidate recipes is smaller than the score difference between two consecutive checkpoints of the same
run, the comparison carries no information.

**Formulas** ([[signal-and-noise-eval]], Heineman et al., arXiv:2508.13144, v1 2025-08, §3.1–3.2, Eq. 2):

  Rel. Std.(m) = sqrt( (1/(n−1)) · Σ_i (m_i − m̄)² ) / m̄        (noise)
  Rel. Dispersion(M) = max_{j,k} |m_j − m_k| / m̄               (signal)
  SNR = Rel. Dispersion(final checkpoints of a model population) / Rel. Std.(final n checkpoints of one run)

Here m_i is the score of the i-th of the final n checkpoints of a single run, m̄ their mean, and m_j, m_k the
final-checkpoint scores of two models in the population M.

**Worked numeric example.** Final 5 checkpoints of one run score 40, 42, 41, 43, 39; mean m̄ = 41, sample
standard deviation = sqrt((1+1+0+4+4)/4) = sqrt(2.5) = 1.58, so Rel. Std. = 1.58/41 = 0.0386. A population of
four models trained at similar compute scores 38, 41, 44, 45, mean 42, so Rel. Dispersion =
(45 − 38)/42 = 0.167. SNR = 0.167/0.0386 = 4.3. Now suppose a second benchmark has the same spread across
models but final-checkpoint scores 36, 44, 40, 46, 39: standard deviation 4.0, Rel. Std. = 0.0976, SNR = 1.7.
The two benchmarks separate the same models equally well at a fixed checkpoint, and only the first one can be
trusted to do so when the checkpoint is chosen arbitrarily.

**Evidence.**
- Across 30 benchmarks, small-scale SNR predicts decision accuracy — the share of model pairs ranked the same
  at small and large scale — with R = 0.791, R² = 0.626, while signal alone and noise alone do not (§4.1,
  Fig. 2). Setup: [[datadecide]] models of 60M–750M predicting 1B rankings, noise from the final 5 checkpoints.
- Noise at the target scale tracks scaling-law prediction error: R = 0.653, R² = 0.426 across 30 tasks, with
  the 13B target's noise taken from its final 30 checkpoints spaced 1,000 steps apart (§4.2).
- Changing the metric from accuracy to bits per byte raises the 30-task average SNR from 10.0 to 31.5 and
  decision accuracy from 77.0% to 83.7%; per task, Minerva MATH goes 51.0 → 90.0 and HellaSwag 74.3 → 95.3
  (§5.3, Fig. 6). This is the reason the Olmo 3 in-loop task list in §3 is written in BPB variants.
- SNR does not transfer across scales: ARC-Easy falls from 7.89 at 1.5B-4T to 5.10 at 32B-6T, SocialIQA from
  8.73 to 1.95, while Minerva MATH 500 rises from 0.91 at 1.5B-4T to 4.45 at 7B-4T (App. B.3, Table 4).
  **Result (single study).**

**The complementary result on whether small runs decide at all.** [[datadecide]] (Magnusson et al.,
arXiv:2504.11393, v1 2025-04) trains 25 corpora × 14 sizes × 3 seeds (1,050 models, up to 100B tokens) and
measures how often a ranking at one small size predicts the 1B winner. Ranking at 150M gets about 80% of
pairwise comparisons right, and none of 8 scaling-law baselines beats the compute-to-decision-accuracy frontier
of single-scale ranking (§3.1–3.2). Per task the picture differs: ARC-Easy is predictable with five orders of
magnitude less compute, while BoolQ exceeds trivial decision accuracy only with intermediate checkpoints of the
target run, and SocialIQA is unreliable at every scale tested (§3.1, Fig. 2). Using a continuous likelihood
metric instead of accuracy moves MBPP and HumanEval from trivial to 80% decision accuracy (§3.4, Fig. 6).

**Conditions and limits.** DataDecide fixes the token-to-parameter ratio at 100 and evaluates multiple-choice
cloze tasks only (§2.4, §5). Signal and Noise studies training-time noise only; noise from evaluation
configuration is named as future work (§6).

**How a frontier run applies this.** [[olmo-3]] built OlmoBaseEval by running the same SNR procedure on the
final 50 checkpoints of OLMo 2 13B training and 10 external base models at roughly 4·10²³ FLOPs, then removing
benchmarks that were too noisy — binary benchmarks such as BoolQ, "as we found that models usually oscillate
between predicting the majority and minority class" — and moving CruxEval out of the macro-average while
keeping it as a separate number (§3.3.3). The suite has 43 tasks, "over 4 times more benchmarks than OLMo 2",
split into a Base Easy suite in BPB for decisions below 1B parameters and a Base Main suite for the final run,
plus a Held-out set of 4 benchmarks — MMLU Pro, DeepMind Math, LBPP, BBH — "to prevent overfitting on the
development suite" (§3.3.4).

---

## §5 Checkpoint selection: averaging along a run and merging across runs

**Definition.** Checkpoint averaging replaces the released parameters θ with the arithmetic mean of several
checkpoints: θ_S = (1/|S|) · Σ_{i ∈ S} θ_i. When the θ_i come from one trajectory it is usually called
averaging or merging; when they come from separate runs that share an initialization it is called souping.

**The measurable problem.** Selecting the single checkpoint with the best development score selects for
capability and for the noise quantified in §4 at the same time. The final-n noise of a 1B model on ARC
Challenge spans 1.7 percentage points ([[signal-and-noise-eval]] §3.1), which is the size of many recipe
effects.

**Worked numeric example.** Two parameters, two checkpoints: θ_1 = (1.0, −0.5), θ_2 = (1.4, −0.1), average
(1.2, −0.3). Averaging acts on weights, not on outputs, so the averaged model costs one forward pass where an
output ensemble of the two costs two ([[model-soups]] §2). It is also a different function from the average of
the two functions, and nothing guarantees its accuracy lies between theirs; the empirical claim below is that
it is often above both.

**Evidence, ordered from the most to the least direct language-model setting.**

1. **Averaging the last checkpoints of a run improves decisions.** [[signal-and-noise-eval]] §5.2, Table 1:
   30-task decision accuracy for 60M-5xC → 1B-5xC rises from 68.9% (final checkpoint) to 71.3% (average of the
   final checkpoints for both the small and the target model), improving on all but two tasks; 13B scaling-law
   prediction error falls from 1.03 to 0.86 absolute percentage points, improving on 20 of 30 tasks. For early
   stopping, an exponential moving average gives higher decision accuracy than a single checkpoint at nearly
   any training step (Fig. 5).
2. **Souping runs that differ only in data order.** [[olmo-2]] §4.5, Table 14, 7B model, 50B-token
   mid-training stage, six candidate mixes, three permutations each:

   | Mix | best single OLMES (MCF) | 3× soup | best single GSM* | 3× soup GSM* |
   |---|---|---|---|---|
   | A | 75.6 | 77.0 | 71.0 | 74.0 |
   | B | 75.3 | 77.3 | 73.0 | 77.0 |
   | C | 76.3 | 76.8 | 66.0 | 66.0 |
   | D | 77.5 | 77.8 | 59.5 | 60.0 |
   | E | 73.4 | 75.3 | 60.5 | 43.0 |
   | F | 77.1 | 77.9 | 73.5 | 74.5 |

   The soup equals or beats the best single run on OLMES for all six mixes. On GSM*, which is not part of the
   OLMES averages but a sample of 200 GSM8K questions used as a math development set (§4.5, §A.1), mix E drops
   17.5 points. The aggregate claim and the per-task claim are different claims; the source makes only the
   first. **Result (single study).** OLMo 2 7B ships as the average of three
   checkpoints on the 50B Dolmino sample; 13B and 32B as the average of four (three on the 100B sample and one
   on the 300B sample), which the report states is empirically better than averaging the three 100B runs alone.
3. **Merging inside pre-training, at scale.** [[merging-in-pretraining]] (ByteDance Seed, arXiv:2505.12082,
   2025-05) merges checkpoints from one trajectory under a WSD schedule for dense models 411M–70B and MoE
   models up to 20B/200B. Merging during the constant-LR phase raised Seed-MoE-1.3B/13B HumanEval from 31.1 to
   36.6 and Seed-MoE-10B/100B from 54.3 to 61.6 (§4.1). Merging early in the cosine-decay phase reached results
   comparable to the end of annealing, and forking a run at 1.4T tokens showed merged constant-LR checkpoints
   matching the annealed model (§4.1, Figs. 2–3). The two settings the paper gives: the interval V between
   merged checkpoints scales with model size — about 4B tokens at 0.7B/7B, 8B at 1.3B/13B, 80B at 10B/100B —
   and merging more checkpoints helps once training is complete, with N = 3 about one point below N = 15 (§4.3).
4. **Production practice.** [[llama-3]] §3.4.3: "During pre-training on the final 40M tokens, we linearly
   annealed the learning rate to 0 … Finally, we compute the average of model checkpoints (Polyak (1991)
   averaging) during annealing to produce the final pre-trained model." §4.1.5 extends it to post-training:
   "we average models obtained from experiments using various versions of data or hyperparameters at each RM,
   SFT, or DPO stage." The Olmo 3 32B README states the same for two stages: "we soup (with simple averaging
   of parameters) the outputs of two separate midtraining runs and we soup the final three checkpoints produced
   by the long-context stage" ([[olmo-core-olmo3-configs]]). **Replicated across organizations at the level of
   the practice; the controlled numbers are OLMo 2's Table 14 and the two papers below.**

**Where the method comes from, and its stated conditions.** [[model-soups]] (Wortsman et al.,
arXiv:2203.05482, ICML 2022) is the controlled study, in image and text classification. For CLIP ViT-B/32 with
72 models from a random hyperparameter search (Table 3): best individual model 80.38 ImageNet / 47.83 average
over five distribution shifts; uniform soup 79.97 / 51.45; greedy soup 81.03 / 50.75. The uniform soup is
0.41 points below the best model on the target distribution and 3.62 above it under shift. The greedy
procedure — sort by held-out validation accuracy and add an ingredient only if the average does not get worse —
exists because a uniform soup fails when some ingredients are poor, which the paper attributes to an error
barrier between models fine-tuned at high learning rates (§3.3.1). Two stated conditions: every ingredient is
a model "fine-tuned independently from a shared initialization θ0 with different hyperparameter
configurations" (§3.2, Fig. 3 caption), so the method is defined only for models that share an initialization;
and §5 names the failure of soups to substantially improve calibration as a limitation.

**The special case of interpolating with the pre-fine-tuning model.** [[wise-ft]] (Wortsman et al.,
arXiv:2109.01903, CVPR 2022) interpolates between the model before fine-tuning (θ0) and after (θ1):
θ_α = (1 − α)·θ0 + α·θ1. For CLIP ViT-L/14@336px (Table 1), fine-tuning moves ImageNet from 76.6 to 86.2 and
the five-shift average from 73.4 down to 68.6; α = 0.5 gives 86.8 and 76.9, above the fine-tuned model on both.
The authors recommend α = 0.5 when the target shift is unknown (§4). Both papers are image classification, and
§7 of WiSE-FT names the absence of language-model experiments as a limitation, so for a language model this is
a mechanism with a hypothesis attached, not a transferred result. The mechanism this course reads into both
results — linear mode connectivity between models that share an initialization — is the one [[olmo-2]] and
[[llama-3]] rely on (Interpretation).

[figures/checkpoint-selection.html](figures/checkpoint-selection.html) plots the OLMo 2 Table 14 rows above so
that the aggregate gain and the mix-E GSM* drop can be read from the same chart.

**Implication for a general-purpose model.** Averaging is the one selection procedure in this chapter with
evidence that it improves breadth rather than the selected metric: the soup gains in [[model-soups]] are larger
under distribution shift than on the target distribution, and the [[wise-ft]] result is entirely a breadth
result. The matching risk is that an average can lose a narrow capability that one ingredient had, which is
what the mix-E GSM* row shows.

---

## §6 Early stopping and checkpoint selection in fine-tuning

**Definition.** Early stopping halts training when a monitored metric stops improving and returns the
checkpoint that achieved the best value. The classical procedure ([[early-stopping-and-checkpointing]])
evaluates a held-out set every `eval_every` steps, saves a checkpoint whenever the validation loss improves by
more than `delta`, increments a counter otherwise, and breaks when that counter reaches `patience` (typically
3 to 5 evaluations). The three parameters that define it are the metric, `delta`, and `patience`; the rest of
this section is about the first one.

**The measurable problem for a general-purpose model.** Target validation loss and target-benchmark accuracy
both keep improving while the capabilities not represented in the fine-tuning mixture degrade. A stopping rule
that reads only the target metric cannot see that.

**Evidence that the stopping point matters.** [[tulu-3]] swept epochs while fine-tuning Llama 3.0 on the
Tülu 2 SFT mixture with a sum loss at LR 5e-6, and reports: "Surprisingly, we additionally found that training
for longer did not yield further improvements, and so used 2 epochs for training" (§4.3, Fig. 6, which plots
average performance for 2 through 7 epochs). The final Tülu 3 SFT runs use 2 epochs at LR 5e-6 for 8B and
2e-6 for 70B. **Result (single study), one mixture, one base model.**

**Evidence that the metric matters more than the stopping point.** Tülu 3 split its evaluation suite into a
development set used for decisions and an unseen set that "we did not examine … when developing our models"
(§2.1, §7.2–7.4). The unseen set pairs each development benchmark with a benchmark testing the same skill
(§7.4.1, Table 31):

| Skill (dev → unseen) | 8B SFT dev / unseen | 8B DPO dev / unseen | 8B Final dev / unseen |
|---|---|---|---|
| Average | 64.9 / 29.9 | 68.3 / 31.9 | 68.8 / 32.4 |
| Reasoning (BBH → AGIEval) | 67.9 / 56.2 | 65.8 / 61.8 | 66.0 / 59.3 |
| Coding (HumanEval → BigCodeBench) | 86.2 / 11.5 | 83.9 / 9.5 | 83.9 / 7.4 |
| Inst. following (IFEval → IFEval-OOD) | 72.8 / 17.6 | 81.1 / 23.9 | 82.4 / 24.3 |

Two readings. First, the pipeline generalizes on average: the final checkpoint has the best average on both
suites. Second, the coding column moves the other way — development coding stays near 84–86 while unseen
coding falls from 11.5 to 7.4 across the same stages. IFEval-OOD was built to "test precise instruction
following abilities of LLMs and whether they are able to follow constraints that go beyond the 25 constraints
included in IFEval", and consists of 52 constraints across six categories (§7.3.1); Tülu 3's reading is that "those models that do well on
IFEval are likely overfitting to the specific set of constraints included in the dataset" (§7.3.1, §7.4.2).
The SFT data ablations in Table 32 show the same split: the data choices generalize on average, and the authors
state that their choices overfit the development evaluations in precise instruction following and to some
extent in knowledge recall and reasoning.

**The selection rule Tülu 3 actually runs at the RL stage.** "For our final runs, we examine model checkpoints
every 40-100 steps and choose the best checkpoint on our development evaluation set" (§6, RLVR implementation
details, item 3). Note the asymmetry: selection uses the development suite, and the unseen suite is reserved
to audit the selection afterwards. That is the only arrangement in which the unseen suite keeps its meaning.

**Conditions and limits.** The Tülu 3 epoch result is one mixture at one scale; the unseen-suite comparison
cannot rule out that the other models compared had trained on those benchmarks (§7.4.2, Table 33 caption).

**Implication for a general-purpose model.** Two disjoint suites, with one of them never consulted during
development, is the operational definition of a generalization measurement at fine-tuning time. A single suite
used for both stopping and reporting measures how well the stopping rule optimized that suite.

---

## §7 Per-domain and per-shard loss tracking after a resume

**Definition.** Per-shard loss tracking logs the training loss broken down by data source, alongside the
identity of the shards the loader is serving.

**The measurable problem.** §1 gives three resume paths that change what the next step sees without raising an
error: the loader state is not restored at all (`load_trainer_state=False` in OLMo-core, or a training loop
that saves only weights and optimizer state); a fingerprint mismatch that was suppressed with
`ignore_fingerprint_mismatch=True`; and a
world-size change, after which per-rank RNG is not restored. In the first two the aggregate loss curve stays
continuous, because the model's loss on data of the same mixture is the same regardless of order.

**The detection method.** Data already seen has a lower loss than unseen data of the same source. A loader
that restarts its order therefore produces a downward step in the per-source loss on the sources that appear
early in the shuffled order, and no change on the others. Two quantities make this visible:

1. Per-source training loss, which shows the step for the affected sources only.
2. The multi-domain held-out loss of §2, evaluated on data never trained on, which does **not** move. The
   divergence between the two is the signature: training loss down, held-out loss flat.

**Worked numeric example.** A mixture is 50% web, 30% code, 20% math, and a loader reset re-serves the web
shards it already served. Write δ for the loss reduction that a second pass over those documents produces;
δ is not a measured constant, so treat it as an assumption of the example and set δ = 0.05 nats. The web
curve then steps down by 0.05, code and math do not move, and the mixture-weighted total moves by
0.5 × 0.05 = 0.025 nats — smaller than ordinary step-to-step variation in a pre-training loss curve. The
per-source split is what makes a change of this size legible, whatever the true value of δ.

**Direct check available in code.** The fingerprint comparison in [[olmo-core-olmo3-configs]] converts the
undetected version of this failure into an exception, provided `ignore_fingerprint_mismatch` is left at its
default. This is the cheapest of the three checks and belongs at load time rather than in a later diagnosis.

**Implication for a general-purpose model.** Repeating a subset of the data changes the effective mixture, and
the mixture is what determines breadth. A per-source loss panel is the same instrument as the per-domain
held-out loss of §2, applied to training data rather than held-out data, and it is the only one that can
attribute a change to a specific source.

---

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Olmo 3 Base 7B | 7B | pretrain-stable | checkpoint save interval | 1000 steps (`save_interval=1000`, `save_async=False`) | OLMo-core@66f768b `src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py` | verified 2026-09-17 | no ablation reported |
| Olmo 3 Base 7B | 7B | pretrain-stable | in-loop held-out-loss interval; dataset | 10,000 steps; `DataMix.v3_small_ppl_validation` | same file, `LMEvaluatorCallbackConfig` | verified 2026-09-17 | no ablation reported |
| Olmo 3 Base 7B | 7B | pretrain-stable | in-loop downstream interval; task set | 10,000 steps; 26 tasks (OLMES subsets in BPB and MC 5-shot, 6 BasicSkills, 6 generative BPB, `copycolors_10way_fast`) | same file, `DownstreamEvaluatorCallbackConfig` | verified 2026-09-17 | SNR selection procedure in [[olmo-3]] §3.3.3 |
| Olmo 3 Base 7B | 7B | pretrain-stable | metrics collection interval | 10 steps (`metrics_collect_interval=10`) | same file, `TrainerConfig` | verified 2026-09-17 | no ablation reported |
| SmolLM3-3B | 3B | pretrain-stable | checkpoint interval; resume switches | 2000 steps; `load_optimizer: true`, `load_lr_scheduler: true` | huggingface/smollm SmolLM3 `stage1_8T.yaml` L1–9 | verified 2026-09-17 | no ablation reported |
| SmolLM3-3B | 3B | pretrain-stable / decay | downstream eval interval | 4000 steps (stages 1–2); 6000 (stage 3); 400000 vs 20000 train_steps in the long-context stage, i.e. disabled | `stage1_8T.yaml`, `stage2_8T_9T.yaml`, `stage3_9T_11T.yaml`, `long_context_4k_to_32k.yaml`, `lighteval.eval_interval` | verified 2026-09-17 | no ablation reported |
| Llama 3 405B | 405B | pretrain-decay/anneal | final-checkpoint rule | average of checkpoints during annealing over the final 40M tokens (Polyak averaging), LR linear to 0 at 128K context | arXiv:2407.21783 §3.4.3 | verified 2026-09-17 | no ablation reported |
| Llama 3 (all sizes) | 8B–405B | reward-model / SFT / preference | model averaging | average of models from experiments with different data versions or hyperparameters at each RM, SFT, or DPO stage | arXiv:2407.21783 §4.1.5 | verified 2026-09-17 | no ablation reported; averaging weights not reported |
| OLMo 2 7B | 7B | mid-train | final-checkpoint rule | average of 3 checkpoints trained on 3 permutations of the 50B Dolmino Mix 1124 sample | OLMo 2 report §4.5 | verified 2026-09-17 | §4.5 Table 14: soup ≥ best single on all 6 mixes (OLMES MCF) |
| OLMo 2 13B, 32B | 13B, 32B | mid-train | final-checkpoint rule | average of 4 checkpoints: 3 on the 100B sample and 1 on the 300B sample | OLMo 2 report §4.5 | verified 2026-09-17 | §4.5 states this is empirically better than averaging the three 100B runs alone |
| Olmo 3 Base 32B | 32B | mid-train; long-context | merge rule | simple parameter average of two midtraining runs; average of the final three long-context checkpoints | OLMo-core@66f768b `src/scripts/official/OLMo3/README.md` | verified 2026-09-17 | no ablation reported |
| Seed-MoE-1.3B/13B (PMA) | 1.3B active / 13B total | pretrain-stable | merge interval V; number of checkpoints N | V ≈ 8B tokens; N = 10 | arXiv:2505.12082 §4.3 | verified 2026-09-17 | §4.3 Fig. 5: V = 16B/32B below baseline at 204B tokens; N = 3 about 1 point below N = 15 |
| Tülu 3 SFT (Llama 3.0 on the Tülu 2 mixture) | 8B | SFT | epochs; loss; LR | 2 epochs; sum loss; 5.00e-6 | Tülu 3 report §4.3, Figs. 5–6 | verified 2026-09-17 | Fig. 6: average performance over 2–7 epochs, 2 best; "training for longer did not yield further improvements" |
| Tülu 3 8B, 70B | 8B, 70B | SFT | epochs; LR | 2 epochs; 5e-6 (8B), 2e-6 (70B) | Tülu 3 report §4.3 | verified 2026-09-17 | same sweep |
| Tülu 3 (RLVR final runs) | 8B, 70B | RL | checkpoint selection rule | inspect checkpoints every 40–100 steps; select the best on the development evaluation set | Tülu 3 report §6, RLVR implementation details, item 3 | verified 2026-09-17 | not applicable |
| Signal and Noise (DataDecide small models) | 60M–750M | eval-gate | noise window | final 5 checkpoints | arXiv:2508.13144 §4.1; App. A.3.2 | verified 2026-09-17 | Table 2: n = 5 lands within ±1σ for almost all benchmarks on OLMo 2 7B |
| Signal and Noise (OLMo 2 13B target) | 13B | eval-gate | noise window | final 30 checkpoints, 1000 steps apart | arXiv:2508.13144 §4.2, footnote 4 | verified 2026-09-17 | stated as a sample-size/compute trade-off |

**Starting point for a small general-purpose run.** For a run below about 10B parameters on a single node or a
small cluster, the verified rows above support: checkpoint every 1,000–2,000 optimizer steps (Olmo 3 7B and
SmolLM3-3B both sit in this range); evaluate multi-domain held-out loss and a
downstream suite at a coarser interval than checkpointing (10,000 steps in-process for Olmo 3 7B, 4,000 steps
asynchronously for SmolLM3); score small-scale downstream tasks in bits per byte rather than accuracy
([[signal-and-noise-eval]] §5.3, 30-task SNR 10.0 → 31.5); and release the average of the final checkpoints
rather than the single best one ([[signal-and-noise-eval]] §5.2). The conditions attached to these numbers:
Olmo 3 7B is 5.93T tokens at a ~4M-token global batch (8,192 tokens × 512 sequences; the GPU count for the 7B
run is not reported — the runtime breakdown in [[olmo-3]] §2.4 covers the 32B), SmolLM3 is 11T tokens at
sequence length 4,096, and the averaging result is measured on DataDecide models of 60M–1B.

---

## Generalization lens

**(a) What increases breadth.**
- Averaging the last checkpoints of a run rather than picking one: +2.4 points of 30-task decision accuracy and
  1.03 → 0.86 absolute scaling-law error at 13B ([[signal-and-noise-eval]] §5.2, Table 1).
- Souping runs that differ only in data order: equal or better than the best single run on the OLMES aggregate
  for all six mid-training mixes tested at 7B ([[olmo-2]] §4.5, Table 14).
- Merging inside pre-training under a constant learning rate: HumanEval 31.1 → 36.6 at Seed-MoE-1.3B/13B and
  54.3 → 61.6 at 10B/100B ([[merging-in-pretraining]] §4.1).
- Interpolating with the model as it was before fine-tuning, in the setting where it has been measured:
  +8.3 points of shift accuracy over the fine-tuned model at α = 0.5, with target accuracy also up
  ([[wise-ft]] Table 1, image classification).
- Measuring per domain rather than in aggregate, so that a decision that costs a domain is visible before the
  run ends ([[paloma]] §4.1).

**(b) What causes narrowing or forgetting, and how it hides.**
- Training past the point where the target metric still improves: Tülu 3's epoch sweep found 2 epochs best over
  2–7 on average performance ([[tulu-3]] §4.3, Fig. 6).
- Optimizing against a development suite until it stops being a measurement: Tülu 3 8B scores 82.4 on IFEval
  and 24.3 on IFEval-OOD, which changes the constraint set and keeps the task structure, and the report finds the same gap for every
  model it compares ([[tulu-3]] §7.3.1, §7.4, Tables 31 and 33).
- Selecting on an aggregate while a component degrades: Tülu 3's unseen coding number falls 11.5 → 9.5 → 7.4
  across SFT, DPO and final while development coding stays near 84 ([[tulu-3]] Table 31).
- Averaging itself, when a single ingredient held a narrow capability: OLMo 2 mix E loses 17.5 points of GSM*
  under the same 3× soup that gains 1.9 points of OLMES ([[olmo-2]] Table 14).
- A resume that re-serves data already seen, which changes the effective mixture without changing the loss
  curve (§7).

**(c) How to measure it at this stage.**
- Multi-domain held-out loss, macro-averaged, documents scored separately, on decontaminated data
  ([[paloma]] §3 G1–G5; the concatenated-input variance in App. H is the failure to avoid).
- A downstream suite under a fixed format, with both CF and MCF reported early in training because MCF is near
  random before roughly 400B tokens ([[olmes]] §3.4, Fig. 1).
- Benchmark-level SNR measured on the final n checkpoints before trusting any small-scale comparison, with
  n = 5 for small models and bits per byte as the metric ([[signal-and-noise-eval]] §3.1, §5.3).
- A held-out suite that is never consulted for development decisions, reported next to the development suite
  ([[tulu-3]] §7.4; [[olmo-2]] §2.5, which advocates "a standard practice of declaring development vs held-out
  evaluation tasks"; [[olmo-3]] §3.3.4).
- Per-source training loss next to per-domain held-out loss, so that a divergence between them identifies a
  data-pipeline change rather than a capability change (§7).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Resuming without the data-loader position | Aggregate training loss continues smoothly; per-source loss steps down on the sources served first | Log per-source training loss; compare against the multi-domain held-out loss, which does not move |
| Suppressing a dataset-fingerprint mismatch to get a run restarted | No error at load; data order differs from the pre-crash run | Leave `ignore_fingerprint_mismatch` at its default and treat the `RuntimeError` as a blocker ([[olmo-core-olmo3-configs]]) |
| Resuming on a different number of GPUs and assuming reproducibility | Log line "Trainer will not restore rank RNG states …"; losses differ from the original run at the same step | Read the resume log; record world size in the run metadata and compare |
| Changing global batch size at resume while tracking the schedule in steps | Token position correct, step counter inconsistent with it | Assert `global_step == tokens_processed // global_batch_size` at load, or track the schedule in tokens |
| Judging a small-scale ablation on a low-SNR benchmark | Ranking flips between adjacent checkpoints of the same run | Compute Rel. Std. over the final 5 checkpoints and compare it to the difference being claimed ([[signal-and-noise-eval]] §3.1) |
| Evaluating only in MCF early in pre-training | Downstream curve flat near chance for hundreds of billions of tokens | Log CF (or BPB) and MCF variants of the same task ([[olmes]] Fig. 1) |
| Scoring an in-loop benchmark by accuracy at ablation scale | Task sits at trivial decision accuracy; no ranking signal | Switch the metric to bits per byte or another continuous likelihood metric ([[signal-and-noise-eval]] §5.3; [[datadecide]] §3.3) |
| Concatenating documents for held-out perplexity | Held-out loss noisy enough to hide the effect under test | Score documents separately; Pythia 1.4B 92.23 ± 17.33 concatenated vs 42.57 ± 0.29 separate ([[paloma]] App. H) |
| Reporting the development suite as evidence of generalization | Development and unseen scores diverge across stages | Keep a disjoint unseen suite and report both ([[tulu-3]] §7.4) |
| Releasing the single best-scoring checkpoint | Released score is not reproduced by the next checkpoint of the same run | Average the final checkpoints and compare ([[signal-and-noise-eval]] §5.2) |
| Souping checkpoints from different initializations or very different learning rates | Merged model far below every ingredient | Check the shared initialization; use the greedy procedure with a held-out set ([[model-soups]] §3.3.1, Recipe 1) |

---

## Check your understanding

1. The aggregate training loss of a resumed run is continuous, and the multi-domain held-out loss is unchanged,
   but per-source training loss dropped for two of six sources at the resume step. Explain what happened and
   why the two aggregate curves could not reveal it.
2. Paloma's C4 baseline gets worse on 65 domains between the 20B and 150B checkpoints. Explain how a pooled
   held-out loss over the same data could keep improving across that interval, using the difference between
   the micro and macro averages.
3. A benchmark has high signal and high noise; another has low signal and low noise. Which is the better
   instrument for choosing between two data mixtures at 150M parameters, and what quantity settles the
   question?
4. Why does changing the scoring metric from accuracy to bits per byte raise a benchmark's SNR, given that the
   underlying model and data are unchanged? Argue from the definitions of signal and noise.
5. A soup of three mid-training runs beats the best single run on the OLMES aggregate and loses 17.5 points on
   GSM*. Give a mechanism that produces this pattern, and state what you would measure to distinguish your
   mechanism from noise.
6. Tülu 3 selects RLVR checkpoints on the development suite but reports the unseen suite. Explain why selecting
   on the unseen suite would destroy the property that makes it worth reporting.
7. WiSE-FT's result is measured only in image classification. Give the argument for expecting it to transfer to
   an SFT'd language model, and give the reason that argument is not evidence.
8. A resume restores parameters, optimizer state, and the step counter, but the run is moved from 512 to 256
   GPUs. List the quantities that are no longer reproducible and say, for each, whether it affects the
   trajectory or only the ability to reproduce it.

---

## Connections

- **Previous — [[ch-05]] Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen.**
  ch-05 fixes the units (tokens per step, sequences per step, world size). §1 of this chapter shows those units
  reappearing inside the checkpoint, where a changed batch size or world size silently changes what a resume
  means.
- **Next — [[ch-07]] Training Failure Modes: Numerical, Masking, and Capability-Level Failures.** ch-07 uses the
  instrumentation built here. The per-domain and per-source curves of §2 and §7 are the inputs to the
  capability-level failure detection in that chapter.

---

## Sources

- [[paloma]] — per-domain held-out perplexity; the six controlled 1B corpus baselines; the macro-average
  definition, the five evaluation guidelines, and the separate-document evaluation requirement.
- [[olmes]] — the fixed evaluation format; CF vs MCF and the 400B-token MCF acquisition curve; the spread of
  published ARC-Challenge numbers for one model.
- [[signal-and-noise-eval]] — definitions of signal, noise and SNR; SNR predicting decision accuracy;
  checkpoint averaging and bits-per-byte as interventions; noise windows of 5 and 30 checkpoints.
- [[datadecide]] — whether a small-scale ranking predicts the large-scale winner; per-task decision accuracy;
  continuous likelihood metrics for code and math.
- [[olmo-2]] — the six-mix souping comparison (Table 14) and the released 7B/13B/32B averaging rules; the
  development-versus-held-out evaluation recommendation.
- [[olmo-3]] — OlmoBaseEval: task clustering, the Base Easy BPB proxy suite, the SNR filtering procedure, and
  the 4-benchmark Held-out set; the stretched second half of the 7B learning-rate schedule.
- [[olmo-core-olmo3-configs]] — excerpt: the trainer and data-loader state dictionaries, the fingerprint and
  world-size guards, checkpoint and evaluation intervals, and the 32B souping statement in the README.
- [[smollm3-training-configs]] — excerpt: checkpoint interval, the `load_optimizer` / `load_lr_scheduler`
  switches, and asynchronous checkpoint-driven evaluation with per-stage intervals.
- [[llama-3]] — checkpoint averaging during annealing (§3.4.3) and model averaging at the RM, SFT and DPO
  stages (§4.1.5).
- [[tulu-3]] — the epoch sweep and the 2-epoch SFT setting; the development/unseen suite split, Table 31, and
  IFEval versus IFEval-OOD; the RLVR checkpoint-selection rule.
- [[merging-in-pretraining]] — excerpt: PMA during the stable and decay phases; merge interval and checkpoint
  count as a function of model size; PMA-init for recovering a broken trajectory.
- [[model-soups]] — excerpt: uniform and greedy soups, the CLIP ViT-B/32 table, and the conditions under which
  averaging fails.
- [[wise-ft]] — excerpt: interpolation between the pre-fine-tuning and fine-tuned models, and the recovery of
  out-of-distribution accuracy at α = 0.5.
- [[early-stopping-and-checkpointing]] — the classical early-stopping loop with patience, used in §6 as the
  baseline procedure that the held-out-suite rule replaces for general-purpose fine-tuning.
