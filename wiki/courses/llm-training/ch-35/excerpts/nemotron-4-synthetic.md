---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md
source_url: https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
created_at: "2026-04-23"
---

# Excerpt: Nemotron-4 340B synthetic alignment pipeline

**Source library:** `wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md`
**Paper:** NVIDIA, *Nemotron-4 340B Technical Report*, 2024.

---

## Why this source anchors ch-35

Nemotron-4 340B is the cleanest public example of **building the whole alignment apparatus around a custom reward model**, then using that RM to synthesize almost all of the training data. The raw-data header states:

> **Core Insight:** NVIDIA compresses alignment into a strong reward model plus a synthetic prompt/response/pair pipeline; over 98% of post-training data is synthetic, and the same pipeline feeds SFT, DPO, and RPO.

Ch-35 cites this as the counter-example to Tülu 3 (ch-33) and Llama 3 — both of which retain substantial human-annotated preference data. Nemotron shows what is achievable when the RM is the leverage point.

---

## The pipeline, in concrete numbers

The raw-data entry's "Synthesis Pipeline" section lists slice sizes:

> - Approximately **800K code SFT samples**, **200K general SFT samples**, **160K DPO preference examples**, and **300K RPO preference examples**, all within a pipeline that is more than 98% synthetic overall.
> - Uses a reward model both as a filter and as a judge for preference ranking when ground truth is unavailable.
> - Implements staged SFT: first a code-focused SFT stage, then a broader general SFT stage.
> - Implements preference fine-tuning with DPO followed by RPO, with the reward model used to select higher-quality chosen responses.

The sizes add to ~1.46M examples; paired with ~20K human anchor examples, the synthetic ratio is >98%. The staged structure — code SFT first, then general SFT, then DPO, then RPO — is deliberate: code SFT sharpens format discipline before the general-SFT stage broadens behavior, and DPO+RPO is preferred over DPO alone because DPO alone can over-fit to the reward gap between chosen and rejected responses.

## Task-family coverage

The raw-data entry enumerates the seed categories:

> **Seed input:** task families for coding, general question answering, topic-following, document-based reasoning, function calling, and incapable tasks that need explicit refusal behavior.

Two details worth flagging for ch-35 readers:

- **Topic-following is intentionally noisy.** The pipeline inserts distractor turns so the student learns to steer back to the topic. This is the reverse of "clean data" — noise is a feature because the eval target requires robustness to off-topic turns.
- **Incapable tasks use human few-shot.** For prompts the model *should* refuse (harmful requests, impossible tasks, hallucination bait), the pipeline uses few-shot prompting with human-written examples to elicit explicit rejections. This is the one place where the human anchor is load-bearing at inference design time, not just at RM training time.

## Genetic Instruct — the code-SFT generator

For code alignment specifically:

> the code alignment stage uses Genetic Instruct, which combines self-instruction and WizardCoder-style mutations plus an LLM-based fitness function to grow a population from a limited number of seeds.

Genetic Instruct is a self-bootstrapping loop: seed a small pool of coding problems, apply Evol-Instruct-style mutations (complicate-input, add-constraints, deepen), score candidates with an LLM-based fitness function (does the test pass, is the solution readable), keep the highest-fitness mutants as new seeds, iterate. This is the mechanism that turns <1K hand-written code seeds into ~800K code-SFT samples.

## Why the apparatus works

From the "Quality / Diversity Evaluation" section of the source:

> The pipeline is meant to preserve behavior diversity across task families while still using synthetic data at very high scale.

The non-obvious engineering claim: the multi-attribute RM's per-dimension scores let the filter *target* different attribute profiles for different task families (high-correctness for math, high-helpfulness for QA, controlled-verbosity for code), which preserves diversity that a scalar RM would collapse. This is what ch-35 §1.1 calls out as the operational payoff of the 5-head design.

## Risks the source explicitly flags

> - Reward-model errors compound when the same scorer is reused across iterations.
> - DPO alone can overfit to reward gaps; the paper adds SFT loss and then RPO to reduce that effect.
> - The synthetic majority is a strength for scale, but it also makes quality filtering and judge calibration critical.

The first point is the ch-23 (model collapse) connection — a synthetic-only pipeline with a fixed RM is structurally vulnerable. Nemotron's mitigation is partial: periodic fresh human-preference injection into HelpSteer2, followed by RM re-training. The paper does not claim this fully eliminates the compounding risk.

---

## Why ch-35 leads with this

Of the case studies in the course, Nemotron's is the heaviest *infrastructure* commitment for the smallest *human* commitment. It is the counterexample to both Tülu 3 ("full recipe disclosure of mixed human + synthetic") and to distillation-SFT ("just copy a teacher's output distribution"). It sits at a distinct point in the design space: *build your own RM, then amortize it across every downstream stage.* Ch-41 (reward modeling) picks up where this excerpt leaves off.
