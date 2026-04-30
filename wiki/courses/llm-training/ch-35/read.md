<!-- chapter: ch-35
     track: sft
     kind: content
     title: Case Studies C — Nemotron, Distillation SFT
     deps: [ch-34]
     sources: [[nemotron-4-synthetic]], [[nemotron]], [[nemotron-ultra]], [[deepseek-r1]], [[deepseek-r1-followup]], [[deepseek-r1-distill-synth]], [[bespoke-stratos]], [[sky-t1]], [[openr1]], [[s1]], [[limo]], [[open-thoughts]]
     figures: figures/distill-sft-compare.html
-->

# Chapter 35 — Case Studies C: Nemotron, Distillation SFT

> **Core insight.** The cases in this chapter sit at two opposite ends of one continuum. Nemotron-4 340B builds the world's heaviest *synthetic-SFT apparatus* — a custom multi-attribute reward model (HelpSteer2) drives a pipeline that produces >98% of the alignment data, with only ~20K human examples anchoring the whole stack. At the other end, Bespoke-Stratos / Sky-T1 / s1 / LIMO build the world's *lightest* post-training — copy a frontier reasoning model's trace distribution, filter, SFT, ship. The lesson worth extracting is that both extremes work, but only for the thing they were built for. Nemotron's machinery is the only viable path if you want to be the trace generator for everyone else; the lean distillation recipes are the only viable path if you want a reasoner for <$1K and your teacher happens to be publicly redistributable.
>
> **Guideline.** Pick the recipe from the decision tree in §6 — not from the benchmark you want to win. If you have your own base model, a compute budget in the tens of millions, and intend to produce open data for others, build a HelpSteer2-style multi-attribute RM and the synthetic pipeline around it. If you have a strong open base, a weekend, and a permissive teacher (R1 or QwQ, not GPT-4), run the 17K-trace distillation recipe. If your base is already reasoning-rich and you have exceptional curators, s1/LIMO-style 1K hand-selection beats both. The boundary between "SFT is enough" and "still need RL" is attested by OpenR1's +3-5 AIME gap between `OpenR1-Qwen-7B-SFT` and its GRPO follow-up — on saturated distilled traces, SFT has a ceiling and RL pushes past it.

---

## Why this chapter exists

Ch-33 and ch-34 walked through the mainstream heavy recipes: Tülu 3's 939K-sample mix, Llama 3's 6-round RSFT, Qwen 2.5's 1M-SFT + 150K-DPO, Qwen 3 hybrid-thinking, Phi-4-reasoning. Ch-20 established the **distillation-as-data** primitive and catalogued three open R1-distill reproductions. This chapter does two things ch-20 did not:

1. Foregrounds **Nemotron-4 340B** as the *producer* of frontier synthetic data — the model whose whole alignment story is "if you build a better RM, 98% of your SFT and preference data can be synthetic." Ch-20 mentioned taxonomy-driven synthesis in passing; here Nemotron gets its own treatment because its HelpSteer2 5-attribute RM is the cleanest public disclosure of a production reward-model recipe and is the reference point ch-41 (reward modeling) will come back to.
2. Asks the **sufficiency question**: at what point does distillation-SFT hit a ceiling that only RL crosses? The s1/LIMO/Bespoke-Stratos/Sky-T1/Open-R1 comparison makes this an empirical, not philosophical, question — and the 5-way table in §5 is the chapter's unit of analysis.

---

## 1. Nemotron-4 340B — the synthetic-alignment apparatus

[[nemotron-4-synthetic]] and [[nemotron]] together describe a 340B dense model whose distinctive contribution is not the base but the **apparatus** wrapped around it: a reward model, a generator, and a critic that together synthesize over 98% of all alignment data. The stated human budget is ~20K annotations total, split between SFT seeds and HelpSteer2 preference labels. Everything else is machine-made.

### 1.1 The HelpSteer2 5-dimensional rubric (verbatim)

This is the chapter's load-bearing artifact. [[nemotron]] trains a regression reward model that outputs a **5-vector**, one score per attribute, instead of a single preference logit. The five attributes, from the HelpSteer2 rubric:

| # | Attribute | What it measures |
|---|---|---|
| 1 | **Helpfulness** | Does the response address what the user asked for? |
| 2 | **Correctness** | Are the factual, logical, or code claims correct? |
| 3 | **Coherence** | Is the response internally consistent and well-structured? |
| 4 | **Complexity** | Does the response match the intellectual depth the prompt demands? |
| 5 | **Verbosity** | Is the response length calibrated to the task? |

The RM architecture is a shared trunk (the 340B base) plus **five linear heads** trained with per-attribute L2 regression on HelpSteer2's 10,000 human-labeled examples. For preference use at RL time, the five scores are combined by a **weighted sum** (weights documented in the NeMo-Aligner config). Nemotron-4-340B-Reward ranked #1 on RewardBench at release.

Why 5 scores instead of 1 matters operationally:

- **Compositional preference at RL time.** You can reweight verbosity down without retraining the RM — the policy gets a different objective for free. A scalar-only RM forces a new annotation round.
- **Single-attribute Goodhart is visible.** If the policy starts over-optimizing verbosity, the other four attributes still give a signal; a scalar RM collapses all four into the hacked one.
- **Synthetic filtering at scale.** Per-attribute scores let the pipeline keep, say, high-correctness-low-verbosity traces for math and high-complexity traces for science without retraining.

This is the response to the question ch-41 will pose: *"why does Nemotron get more out of 10K human preferences than Llama 3 gets out of much more?"* Answer: richer label schema amortizes the human cost across five training signals.

### 1.2 The >98%-synthetic alignment pipeline

[[nemotron-4-synthetic]] itemizes the pipeline; [[nemotron]] names the approximate slice sizes. In words, the six-stage loop:

```
seed (task-family prompts)
   -> prompt generation (Nemotron-4-Instruct_{t-1} synthesizes task prompts per family)
   -> response generation (Instruct_{t-1} emits 1-N candidate responses / dialogues)
   -> RM filtering (340B-Reward scores each response on 5 attributes)
   -> selection (keep high-score; for DPO pair the high vs low)
   -> stage-specific training (SFT -> DPO -> RPO, iteratively)
```

The task families covered: coding, general QA, topic-following, document-based reasoning, function-calling, and *incapable tasks* (prompts that should be refused — few-shot seeded with human-written rejections). For topic-following, the pipeline *intentionally injects distractor turns* so the student learns to steer back. Approximate output volumes:

- **~800K code SFT** (generated via Genetic Instruct: Self-Instruct + WizardCoder-style mutations + LLM fitness function growing a small seed into a 1000x-scale population).
- **~200K general SFT** (category-seeded; RM-filtered).
- **~160K DPO preference pairs.**
- **~300K RPO preference pairs** (reward-preference optimization — DPO with an added SFT loss term to prevent the policy from "flying off" from the reference).

SFT is staged: **code SFT first**, then **general SFT**. The paper's justification is that code SFT sharpens format discipline before general-domain SFT introduces looser objectives. Preference optimization runs **DPO followed by RPO**; Nemotron argues DPO alone overfits to the reward gap between chosen and rejected, and RPO's auxiliary SFT term on the chosen response counteracts that drift.

### 1.3 Why a small human anchor suffices

The counter-intuitive claim is that ~10K HelpSteer2 labels plus ~10K SFT seeds are enough to sustain an alignment loop generating tens of millions of downstream tokens. The mechanism:

1. The 10K human labels train the RM, not the policy directly.
2. The RM then scores an arbitrary-size synthetic pool — the RM's *coverage*, not the human set's, bounds what can be filtered.
3. The filtered synthetic pool trains the policy. Each RM query is a machine operation; the human cost amortizes.

The risk Nemotron flags: **reward-model errors compound when the same scorer is reused across iterations.** If the RM systematically underscores a correct-but-unusual reasoning step, iteration 2's policy stops emitting those steps, iteration 3's RM is now tuned on a narrower distribution, and the collapse compounds. The mitigation is partial: Nemotron periodically adds fresh human preferences to the HelpSteer2 pool and re-trains the RM, but the paper does not claim this fully eliminates the compounding risk. Ch-23 (model collapse) is the direct continuation of this failure mode.

---

## 2. Nemotron-Ultra / Nemotron 3 — multi-environment RL succession

[[nemotron-ultra]] describes the 2025 successor. Nemotron 3 ships as Nano (3.2B active / 31.6B total MoE), with Super and Ultra tech reports to follow. The two deltas from Nemotron-4 that matter for this chapter:

- **Multi-environment RL** replaces sequential stages. Nemotron-4 ran reasoning-RL then tool-use-RL then alignment-RL; Nemotron 3 collapses these into a single RL run spanning reasoning, multi-step tool use, and agentic environments with the reward model (now a **GenRM** — generative reward model) scoring across all of them. The claim is better generalization to agentic tasks than the staged recipe.
- **GenRM is publicly released** alongside the policy. Nemotron-4's 340B-Reward was open-weight but the recipe for training it was not fully reproducible from the paper; Nemotron 3's GenRM release lets downstream users resume RLHF without retraining the RM.

What [[nemotron-ultra]] *does not* disclose is as telling as what it does: RL algorithm (PPO vs GRPO vs DPO unspecified), KL β, LR, batch size, clip ε, group size G, rollouts per prompt, step counts, GenRM loss form, preference-data sizes, multi-environment reward-mixing weights. The white paper is thin on hyperparameters — a reminder that "open release" is a spectrum and Nemotron 3 sits closer to "reproducible artifact bundle" than to "reproducible recipe."

The Nemotron-4 -> Nemotron 3 shift that matters for ch-35: the synthetic-data pipeline is *carried forward* but no longer the headline. The headline is the RL environment coverage. Synthetic SFT is now the *substrate*, not the finish line.

---

## 3. R1-distill as SFT-consumption — what changed from ch-20

Ch-20 covered the teacher-side R1 pipeline in detail. This chapter revisits R1-distill from the **student-side** angle: once DeepSeek emits the 800K trace pool, what does SFT-consuming it look like, and why does it work without RL?

From [[deepseek-r1]] / [[deepseek-r1-distill-synth]] / [[deepseek-r1-followup]]:

- The distill corpus is produced by the **rejection-sampling SFT stage** in the teacher's own pipeline — stage-1 RL model samples N traces per prompt, V3-judge filters for readability + correctness, kept set is ~600K reasoning + 200K non-reasoning.
- Six distilled students are released: Qwen-2.5-Math 1.5B, Qwen-2.5 7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B. **All are pure SFT on the 800K** — no RL, no RM, no DPO.
- The explicit claim from the report: *dense students benefit more from copied reasoning structure than from rediscovering that structure via their own RL.* A dense 32B student running GRPO from scratch needs more compute than one-epoch SFT on R1 traces and gets a weaker model.

The R1-0528 refresh ([[deepseek-r1-followup]]) is *R1-with-more-compute* — same V3 base, same recipe, more RL steps. R1-Distill family is unchanged. V3.1 (Aug 2025) then *absorbs* R1's reasoning into the V3 line, ending R1 as a standalone family. The practical implication for ch-35: R1-distill is likely the **terminal** version of "straight SFT transfer of reasoning"; future recipes will be hybrid-thinking-mode (ch-34 Qwen 3) rather than separate distilled reasoners.

---

## 4. Bespoke-Stratos, Sky-T1 — "cheap frontier reasoning" as an operational claim

Ch-20 catalogued these. Here we extract the specific cost claims and what exactly each team filtered, because the cost numbers are the chapter's concrete evidence that distillation SFT can be extremely cheap.

### 4.1 Bespoke-Stratos — contamination checks and the $4.8K run

[[bespoke-stratos]]: 17,000 `(prompt, R1-trace)` pairs covering math (~7K problems from NuminaMath-CoT, MATH, AIME/AMC archive), code (~5K from APPS, CodeContests, TACO, LeetCode), science (~5K from STILL-2 curated prompts + CoTLogic).

**Trace generation.** Query DeepSeek-R1 (official API) at temperature 0.6, request `<think>...</think><answer>...</answer>` format, retry up to 3× on failure.

**Three-layer verifier (rejection-sampling filter):**

1. **Math.** Extract boxed answer; compare to gold via SymPy canonicalization; reject on mismatch.
2. **Code.** Extract candidate solution; run public unit tests; reject on any failure.
3. **Science.** GPT-4o as LLM-judge; require "correct" verdict against reference.

Reject rate ~30-50% of raw R1 outputs; majority of rejections are code test failures and math extraction errors. MinHash dedup cross-prompt; per-source cap enforces domain balance. **Contamination check** — because AIME and MATH are public and R1 may have memorized solutions, Bespoke explicitly holds out AIME25 as a clean eval; the Stratos-32B paper reports AIME24 ~63% but flags that AIME25 numbers are weaker and represent the "post-contamination-gap" reality.

**Cost.** ~$800 DeepSeek-R1 API credits (teacher) + ~$4,000 student training (8×H100, few hours on Qwen2.5-32B-Instruct). Ablation: removing code-verification halves LiveCodeBench gain; removing math symbolic equivalence halves MATH gain. **Every verifier layer is load-bearing.**

### 4.2 Sky-T1 — $450 QwQ recipe and the reformatting trick

[[sky-t1]]: 17K traces, mostly distilled from **QwQ-32B-preview** (Alibaba's open-weights reasoner, no API lock-in). Qwen2.5-32B-Instruct base, 3 epochs × 19 hours on 8×H100 ≈ $450 on rental hardware at listed rates.

**Pulled from QwQ.** Local vLLM inference (teacher cost ≈ 0), temperature 0.7, max 8K tokens per trace, ~10K math seeds (NuminaMath-CoT + AIME/AMC) + ~5K code (APPS + TACO) + ~2K science (STILL-2).

**Filtered out.**

- **Math mismatch.** SymPy on `\boxed{}`; reject non-matching.
- **Code failure.** Unit-test execution; reject any-test-fail.
- **Science incorrect.** GPT-4o-mini LLM-judge; reject "incorrect" verdicts.
- **Format noise.** QwQ emits "Alright, let me think", "Hmm, okay so", and other filler preambles; Sky-T1 runs a GPT-4o rewriter pass that converts QwQ's native format to `<|im_start|>…<|im_end|>` chat template *and* strips fillers. The paper reports this cleanup *alone* lifted AIME by +4 points — the rewriter is not cosmetic.

**Training config attested.** LR 1e-5, 3 epochs, sequence length 32K (to fit long traces), BF16, FSDP across 8 GPUs, Llama-Factory framework.

**Results.** MATH500 82.4% (matches o1-preview), AIME24 43.3% (within 2 pts of o1-preview), LiveCodeBench-Easy 86.3% (beats o1-preview), GPQA-Diamond 56.8%. **AIME25 drops significantly** — QwQ as a teacher has a lower ceiling than R1, and Sky-T1 inherits it. This is the clearest public evidence that *teacher quality is the ultimate ceiling of SFT-only distillation*.

---

## 5. Comparison table — 5 distillation recipes

This is the chapter's headline table. The interactive version is [figures/distill-sft-compare.html](figures/distill-sft-compare.html) — click any recipe for the full filter breakdown and attested hyperparameters.

| Recipe | # traces | Teacher | Filter stack | Cost (full) | AIME24 | MATH500 | GPQA-Dia |
|---|---:|---|---|---:|---:|---:|---:|
| **R1-Distill-Qwen-32B** (official) | 800,000 | DeepSeek-R1 (671B MoE) | V3-judge readability + correctness | not disclosed | ~72% | ~94% | ~62% |
| **Bespoke-Stratos-32B** | 17,000 | DeepSeek-R1 | SymPy + unit tests + GPT-4o-judge + MinHash | ~$4.8K ($800 API + $4K compute) | ~63% | ~93% | ~59% |
| **Sky-T1-32B-Preview** | 17,000 | QwQ-32B-preview (open) | SymPy + unit tests + GPT-4o-mini judge + GPT-4o rewriter | ~$450 (local QwQ + 8×H100) | ~43% | ~82% | ~57% |
| **OpenR1-Qwen-7B** | ~440,000 (220K×2) | DeepSeek-R1 | Math-Verify SymPy only (math-only corpus) | ~$10K + multi-day H100 | ~40% | ~80% | n/a |
| **s1-32B** | 1,000 | Gemini (CoT traces) + hand-curation | difficulty + diversity + quality (hand-filter from 59K pool) | ~26 min × 16 H100 ≈ $50 | 56.7% | 93.0% | 59.6% |
| **LIMO** (817 traces) | 817 | mix + hand-edited | manual correctness + reflective-structure filter | hand-curation labor | 63.3% | 95.6% | (strong OOD) |

Three non-obvious readings of this table:

**5.1 The trace count and the benchmark are not monotone.** s1's 1,000 and LIMO's 817 beat Sky-T1's 17,000 on AIME24 despite Sky-T1 having ~17× more data. The mechanism per [[s1]] and [[limo]]: hand curation removes low-signal traces that *corrupt* rather than augment the student; mass distillation inevitably smuggles them in even after SymPy/unit-test filtering. s1 goes further with **budget forcing** at inference (suppress early stopping by appending `"Wait"`) — which lifts AIME24 from 50% to 57% on the same checkpoint, no additional training. Budget forcing is not a training trick; it reallocates latent compute.

**5.2 17K is a regime, not a target.** Bespoke-Stratos, Sky-T1, and (coincidentally) several OpenThoughts intermediate checkpoints all landed around 17K. The pattern is not mystical: the smallest pool that survives a three-layer verifier over math + code + science from a public seed set, with ~30-50% reject rate, is approximately 17K. If you filter harder you drop below 10K and lose domain coverage; if you filter softer you keep bad traces. The 17K-number is an equilibrium, not a hyperparameter.

**5.3 OpenR1's GRPO delta is the "still need RL?" evidence.** OpenR1-Qwen-7B gets MATH ~80% / AIME24 ~40% with pure SFT on 440K traces. Adding a GRPO stage on a 40K-subset (binary Math-Verify reward) adds +3-5 AIME points. This is the single cleanest public ablation isolating "SFT ceiling vs RL residual." The gap is not huge (5 points on AIME), but it is real and persists after the SFT budget is already generous.

---

## 6. When distillation SFT is enough vs when you still need RL

The decision tree the chapter is built around. Three axes: base-model reasoning capacity, teacher-trace license, target evaluation.

```
Is your base model already reasoning-rich (Qwen2.5-32B / Llama-3.1-70B)?
|-- YES: distillation SFT is likely enough
|     |
|     Can you hand-curate?
|     |-- YES + strong curators  -> s1 / LIMO regime (1K traces, $50, hand review)
|     \-- NO  -> Bespoke-Stratos / Sky-T1 regime (17K traces, ~$500-$5K)
|
\-- NO (base is weaker / smaller / general-purpose):
      |
      Is your target eval well-verifiable (math, code)?
      |-- YES -> SFT + GRPO/RLVR
      |         (OpenR1 pattern: 440K SFT + 40K-prompt GRPO; s1 budget forcing
      |          stops being cheap recovery and RL is needed to break the ceiling)
      \-- NO (agentic, tool-use, open-ended)
                -> full Nemotron-style synthetic-pipeline + multi-attribute RM + RL
                   (reasoning-only SFT ceiling is lower on open-ended tasks)
```

The asymmetry this tree encodes is the chapter's take-home: **SFT saturates faster on the verifiable domains it was supposed to be good at**. Math and code are where outcome rewards and unit tests work, and where the verifier is strong enough to catch the "wrong-question-correctly" failure ch-20 §5.5 flagged. On open-ended tasks the verifier is an LLM-judge which is itself distribution-shift-brittle; SFT's ceiling there is lower but RL's ceiling is also lower. Nemotron-style multi-attribute RM is the only lever that reliably works across all regimes — which is why Nemotron-Ultra multi-environment RL is the 2025 direction, not distillation.

See [figures/distill-sft-compare.html](figures/distill-sft-compare.html) for the full recipe switch — click a scenario and the tree highlights the recipe row.

---

## 7. What this chapter leaves open

- **Per-attribute RM weights.** [[nemotron]] says "weighted sum of 5 attributes (weights in NeMo-Aligner config)" — the weights themselves are not in the paper. Practitioners must read the NeMo-Aligner source.
- **R1-Distill's per-domain slice ratios.** DeepSeek gives 600K + 200K split but not the per-domain breakdown of the 600K reasoning set.
- **Budget-forcing transferability.** [[s1]] shows `"Wait"`-appending boosts AIME but not on every prompt class; the paper does not characterize which prompt distributions respond.
- **OpenR1 GRPO on non-math.** The GRPO stage is math-only (Math-Verify as reward); the generalization to code / science is an open question ch-44 (verifiable rewards) partially addresses.

---

## Connections and what's next

- **ch-20** — Distillation-as-data origin chapter; Orca lineage + R1-distill mechanics. This chapter assumes ch-20 and uses its vocabulary.
- **ch-33 / ch-34** — Tülu 3, Llama 3, Qwen 2.5/3, OLMo 2/3, Phi 3/4 — the mainstream SFT case studies. Nemotron is the **synthetic-only** peer; the 5-dim RM is what distinguishes it.
- **ch-36 (lab)** — Packed SFT run with masking tests; the practical checkpoint after this case-study block.
- **ch-23 (model collapse)** — Nemotron's self-bootstrapping RM loop is the archetypal test case; the compounding-error risk §1.3 flagged is made precise there.
- **ch-41 (reward modeling)** — HelpSteer2 5-attribute regression is the reference recipe; scalar-only baselines are the comparison.
- **ch-40 (GRPO)** — R1's GRPO hyperparameters and DeepSeek's loose-clip (ε=10) philosophy are the case study.
- **ch-44 (verifiable rewards)** — OpenR1's Math-Verify is the smallest working verifier; Bespoke-Stratos's three-layer stack is the canonical extended version.

## Further reading

- [[nemotron-4-synthetic]] — NVIDIA 2024; >98%-synthetic alignment; staged SFT + DPO + RPO; Genetic Instruct for code.
- [[nemotron]] — Nemotron-4 340B model report; HelpSteer2 5-attribute RM; 10K human preferences.
- [[nemotron-ultra]] — Nemotron 3 Nano white paper; multi-environment RL; GenRM release; reasoning budget control.
- [[deepseek-r1]] / [[deepseek-r1-followup]] / [[deepseek-r1-distill-synth]] — R1 pipeline, R1-0528 refresh, 800K distill corpus.
- [[bespoke-stratos]] — 17K curated, $800 API + $4K compute, three-layer verifier, AIME24 ~63%.
- [[sky-t1]] — $450 QwQ recipe, GPT-4o rewriter +4 AIME, teacher-ceiling lesson.
- [[openr1]] — 220K×2 math corpus, Math-Verify, GRPO-adds-+3-5-AIME evidence.
- [[s1]] — 1K curated + budget-forcing at inference; 26-minute SFT.
- [[limo]] — 817 traces; Less-is-More Reasoning Hypothesis; latent-capability activation.
- [[open-thoughts]] — 1000+ ablations; QwQ > R1 as teacher; no-answer-side-filter finding.

## Companion visualization

**[figures/distill-sft-compare.html](figures/distill-sft-compare.html)** — self-contained interactive comparator. Five recipe cards (R1-Distill-Qwen-32B / Bespoke-Stratos / Sky-T1 / OpenR1 / s1) arranged side by side. Click any card to expand the filter stack (which verifier layers, reject rate, dedup policy, rewriter step) and see the attested evaluation numbers. The decision-tree panel on the right lights up the row matching the current selection so you can see which recipe your answer-profile points to. Use it before a distillation run to pick your recipe, and after a run to check your numbers against the reference grid.
