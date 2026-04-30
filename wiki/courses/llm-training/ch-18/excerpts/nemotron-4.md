---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "Nemotron-4 340B Technical Report (NVIDIA, 2024) — synthetic alignment pipeline"
source_url: https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
created_at: "2026-04-23"
---

# Excerpt: Nemotron-4 — industrial-scale alignment, and the reward model as stage 4 + stage 5

**Authors:** NVIDIA
**Year:** 2024
**URL:** https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
**Raw-data source:** [[raw-data/nemotron-4-synthetic]]

---

## Why this paper is the flagship stage-4-heavy pipeline

Nemotron-4 is the synthetic-data paper that proves at industrial scale what Lambert calls the operating principle of 2025: **synthetic data can do almost all of the work, given a strong base model and a robust verifier**. The raw-data file states the headline number:

> "The paper says over 98% of the training data for alignment is synthetic, while only about 20K human-annotated examples are used overall, split between SFT and HelpSteer2 reward-model data."

98% synthetic, 20K human anchors. That ratio is only possible because stage 4 of the loop is industrialised — the Nemotron-4-340B-Reward model scores every candidate response, and stage 5 uses those scores to pick chosen/rejected pairs automatically. Self-Instruct left stage 4 empty; Nemotron makes it the centre of the pipeline.

---

## Mapping Nemotron to the six-stage loop

The raw-data file lays out the pipeline stage by stage. Map it:

- **(1) Generate.** "The pipeline synthesizes prompts by task family, then uses the current model to generate responses or multi-turn dialogues. For impossible tasks, it uses few-shot prompting with human-written examples to elicit explicit rejections." Genetic Instruct for code combines self-instruct and WizardCoder-style mutations plus an LLM-based fitness function.
- **(2) Filter.** Staged — code filter separate from general filter. Topic-following track intentionally keeps distractor turns (a reminder that filtering is sometimes about *not* over-cleaning).
- **(3) Dedup.** Light, cross-prompt. The paper leans on task-family partitioning for diversity more than on aggressive near-dup removal. This is a notable contrast with Self-Instruct (ROUGE-L-central).
- **(4) Verify.** "**A reward model scores responses for quality; when ground truth is missing, Nemotron-4-340B-Reward selects high-quality chosen responses. The preference pipeline prefers RM-based ranking over raw model self-selection.**" This single sentence is the paper's claim to fame.
- **(5) Select.** RM-score-driven: highest-scoring response becomes the "chosen" in a preference pair, lowest-scoring becomes "rejected." Selection IS the preference-pair construction step.
- **(6) Mix.** Staged: code SFT first (800K), then general SFT (200K); then DPO (160K), then RPO (300K). The paper also adds a small amount of alignment-style QA in continued pretraining "to steer low-accuracy areas."

Notice: every stage is populated and stage 4 is the most sophisticated slot. That is the Nemotron fingerprint.

---

## The numbers that define the pipeline

From the raw-data extract:

> "Output shape: approximately 800K code SFT samples, 200K general SFT samples, 160K DPO preference examples, and 300K RPO preference examples, all within a pipeline that is more than 98% synthetic overall."

- **Total alignment data:** ~1.46M examples (0.8M + 0.2M + 0.16M + 0.3M).
- **Human anchor:** ~20K examples (~1.4% of the total).
- **Amplification factor:** ~73x — one human-curated example supports 73 synthetic ones.

Compare with Self-Instruct's 175 seed -> 52K amplification (~300x). Self-Instruct is actually the *higher*-amplification pipeline; Nemotron's lower ratio buys you a reward-model-calibrated verifier instead. The trade is real money (the ~20K HelpSteer2 annotations are expensive) for real quality.

---

## Staged SFT: why code comes first

The raw-data file:

> "Implements staged SFT: first a code-focused SFT stage, then a broader general SFT stage."

This is a stage-6 design decision. Mixing 800K code examples with 200K general examples homogeneously would dilute the code signal (code is 4x the volume but has narrower objectives). Staging preserves the code capability before letting general SFT smooth it. The same reasoning shows up in Tulu-3 and in Llama-3's post-training recipe.

Notice: the mix ratio is not just "how much synthetic vs real" (Nemotron is >98% synthetic regardless); it is "in what order and in what proportions within synthetic." That is the stage-6 axis ch-27 will treat in depth.

---

## DPO then RPO: iterating at stage 5

The raw-data file:

> "Implements preference fine-tuning with DPO followed by RPO, with the reward model used to select higher-quality chosen responses."

Two things to notice. First, DPO-then-RPO is itself a stage-6 curriculum. Second, the same RM serves stage 4 (filter by quality) AND stage 5 (rank pairs) AND stage 6 (ordering decisions — which preference data comes first). A single component doing three jobs.

This is also where the paper flags the corresponding failure mode:

> "Reward-model errors compound when the same scorer is reused across iterations."

The defence, from the paper: "DPO alone can overfit to reward gaps; the paper adds SFT loss and then RPO to reduce that effect." Which is to say — the iteration-loop danger is real, and the mitigation lives at stages 5 and 6.

---

## Genetic Instruct: stage 1 is not just prompting

Most synthetic-data papers treat stage 1 as "prompt the teacher." Nemotron's code track uses Genetic Instruct, which the raw-data file describes:

> "The code alignment stage uses Genetic Instruct, which combines self-instruction and WizardCoder-style mutations plus an LLM-based fitness function to grow a population from a limited number of seeds."

This is stage 1 with a population-evolution mechanism bolted on. It is the same loop slot but filled with a richer procedure. The point for ch-18: the six stages are *slots*, and each slot admits a range of implementations from trivial (few-shot prompting) to elaborate (evolutionary search with fitness functions). Understanding which slot you are upgrading is the first step; choosing the implementation is the second.

---

## The anchor-set pattern, made explicit

The paper's most generalisable insight is compressed in the guideline of the raw-data file:

> "Use a category-seeded synthetic pipeline, quality-filter with a reward model, keep a small human anchor set for the reward model and hard cases, then iterate the generator checkpoint."

Four moves: category seeds (stage 1), RM filter (stage 4), human anchor (what supports stage 4), iteration (the loop-around-the-loop). The small human anchor is the piece most replicators get wrong — they either skip it entirely (pure self-distillation; collapse risk) or swell it to the point where it dominates the mix (defeating the synthetic-scale thesis). Nemotron's ~1.4% anchor ratio is a concrete datapoint for the right scale.

---

## Risks the paper flags

From the raw-data extract:

> "- Reward-model errors compound when the same scorer is reused across iterations.
> - DPO alone can overfit to reward gaps; the paper adds SFT loss and then RPO to reduce that effect.
> - The synthetic majority is a strength for scale, but it also makes quality filtering and judge calibration critical."

All three are stage-4 / stage-5 risks. The synthetic-heavy pipeline concentrates risk at the verifier; get the RM wrong and the entire 1.4M-example edifice degrades in lockstep. This is the cost of the design: when stage 4 is your leverage, it is also your single point of failure.

---

## Iterating the generator checkpoint — the loop-around-the-loop

The raw-data file notes:

> "The paper iterates through Nemotron checkpoints for alignment, and uses Nemotron-4-340B-Reward as the scorer/judge during filtering and ranking."

This is the iteration arrow of the loop — the dashed line from stage 6 back to stage 1 in the ch-18 figure. Nemotron does it twice over, in two distinct ways:

- **Teacher rotation.** After an SFT stage, the newer checkpoint becomes the generator for the next round. The synthetic data shifts distribution; the RM has to track it.
- **Judge reuse.** The 340B-Reward model is re-used as both filter (stage 4 scoring) and judge (stage 5 preference selection) across iterations.

This reuse is explicitly flagged as risky in the paper ("reward-model errors compound when the same scorer is reused across iterations"). The defence is the ~20K human anchor for RM calibration — a stage-adjacent invariant that does not itself iterate. In other words: the loop iterates, but the anchor does not, and that asymmetry is the pipeline's stability lever.

For ch-18: iteration is a legitimate move, but it converts stage-4 error into systemic error. The mitigation is to pin one component (the anchor, the human-curated prompts, the static seed distribution) so the loop has something to snap back to. Nemotron pins the anchor; Self-Instruct pinned the 175 seed tasks; OMI-2 pins the MATH/GSM8K source problems. Every successful iterated synthesis pipeline has a non-iterated pin.

## The code-then-general staging, once more

Worth a second look because it is the single stage-6 decision most teams forget to ablate. Nemotron's 800K code SFT stage comes *before* the 200K general SFT. Why not the other way around, or interleaved?

The stated reason (from the raw-data file): "the paper separates code SFT from general SFT, keeps topic-following data intentionally noisy with distractor turns, and adds a small amount of alignment-style QA in continued pretraining to steer low-accuracy areas."

Read the sequence: *pretraining-continued QA* -> *code SFT* -> *general SFT* -> *DPO* -> *RPO*. Four curriculum boundaries, each separating data with different objective structures. Homogenising any pair of adjacent stages is an ablation the paper implies (without publishing the numbers) would hurt. The ch-18 takeaway: **curriculum ordering is a stage-6 hyperparameter that deserves explicit ablation**; treating it as "just stack the datasets" is the default that under-performs.

For ch-27 we will revisit whether this specific ordering generalises. For ch-18, the operational point is that a staged curriculum is the norm at industrial scale, not an exotic choice.

## Connections

- [[excerpts/self-instruct]] — the empty-stage-4 baseline; Nemotron is the industrial-scale completion.
- [[excerpts/openmathinstruct-2]] — cheap modality-specific stage-4 (SymPy) in contrast to Nemotron's expensive general-purpose stage-4 (340B RM); both are valid design points.
- [[excerpts/apigen]] — stage 4 done differently: three independent layers instead of one RM. Complementary design.
- [[excerpts/openmathinstruct-2]] — stage 4 via SymPy (cheap, modality-specific) vs Nemotron's stage 4 via 340B RM (expensive, general).
- [[excerpts/nathan-lambert-synth]] — "98% synthetic" as a Lambert-era talking point; this paper is where it comes from.
- [[ch-18]] — parent. Nemotron-4 is the ch-18 flagship for stage 4 + 5, and for the anchor-set principle.
