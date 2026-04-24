---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sky-t1.md
source_url: https://novasky-ai.github.io/posts/sky-t1/
created_at: "2026-04-23"
---

# Excerpt: Sky-T1 — the $450 recipe and what it filtered

**Source library:** `wiki/raw-data/llm-training/papers/sky-t1.md`
**Team + date:** Sky Computing Lab / NovaSky (UC Berkeley), January 2025.

---

## Why this source anchors ch-35

Sky-T1 is the *open-teacher* cost-floor entry. From the source header:

> **Core Insight:** A 32B reasoning model matching o1-preview on math and code is trainable for ~$450 by distilling 17K traces from QwQ-32B-preview and fine-tuning Qwen2.5-32B-Instruct for 19 hours on 8×H100 — dramatically lowering the entry cost for o1-class open research.
>
> **Guideline:** If your budget is <$1K, use QwQ-32B-preview (fully open weights, no API) as the reasoning teacher, curate ~10–20K traces across math/code/science, run SFT for ~3 epochs, and skip RL entirely; this is the cheapest path to a credible open reasoning model.

Ch-35 §4.2 uses this as the cheap-budget anchor. The excerpt records what was pulled from the teacher, what was filtered out, and where the teacher-ceiling showed up.

---

## What they pulled from QwQ

From "Synthesis pipeline":

> - **Seed input:** problems drawn from multiple sources:
>   - APPs + TACO (code) — ~5K.
>   - NuminaMath-CoT + AIME/AMC — ~10K math.
>   - STILL-2 + miscellaneous science/reasoning — ~2K.
> - **Trace generation:** query **QwQ-32B-preview** locally via vLLM for each seed, temperature 0.7, max 8K tokens per trace.
> - **Teacher model:** QwQ-32B-preview (open-weights).

Note the temperature difference from Bespoke-Stratos: Sky-T1 uses T=0.7 (vs Stratos's 0.6). Higher temperature on a weaker teacher is a conscious choice — slightly broader sampling to make up for the teacher's lower ceiling. The max-tokens cap at 8K (vs R1's 32K) is also teacher-driven: QwQ's traces are shorter, so 8K is rarely the limiting factor and saves inference time.

## What they filtered out

Three verifier layers plus a distinctive reformatting step:

> - **Math:** parse `\boxed{}` and compare to gold via SymPy; reject mismatches.
> - **Code:** execute against public unit tests; reject any-test-failure.
> - **Science / open-ended:** use GPT-4o-mini as LLM-judge to verify final-answer agreement with reference.
> - **Reformatting trick:** QwQ's raw output uses a non-standard thinking/answer format; NovaSky reformats to `<|im_start|>…<|im_end|>`-style chat template and removes redundant "Alright, let me think" preambles via a GPT-4o rewriter (this cleanup lifted AIME by +4 points).

The three verifier layers parallel Bespoke-Stratos. The **GPT-4o rewriter pass** is the Sky-T1 distinctive: QwQ's native format has verbose preambles and non-canonical `<|im_start|>` placement; the rewriter normalizes these. The attested +4 AIME points from reformatting alone is the clearest published evidence that *trace formatting matters for SFT* — the student learns the wrapper as part of the skill, and noisy wrappers dilute the signal.

## The training config, attested

> **Training config:** 3 epochs, LR 1e-5, seq len 32K (to fit long traces), BF16, FSDP across 8 GPUs via Llama-Factory.

Every number is in the source: 3 epochs, LR 1e-5, seq 32K, BF16, FSDP×8, Llama-Factory framework. Wall-time 19 hours on 8×H100 ≈ $450 at listed rental rates. This is one of the few 2025 public recipes with every hyperparameter disclosed — contrast with Nemotron-Ultra which hides almost all RL hparams.

## The teacher ceiling

From "Risks + gotchas":

> - **QwQ teacher quality is lower than R1** on hardest problems — Sky-T1 underperforms Bespoke-Stratos and R1-Distill on AIME25.

AIME25 is the clean OOD eval (post-teacher-cutoff). Sky-T1 lands ~20 points behind Stratos on AIME, which is the 2025 reference datapoint for *how much teacher quality matters when everything else is held fixed*. The seed pool, filter stack, student base, and training config are nearly identical between Sky-T1 and Stratos; the only change is the teacher (QwQ vs R1). The ~20-point gap on AIME is the answer.

For ch-35 the implication is: **teacher quality is the ultimate ceiling of SFT-only distillation**. If you cannot use R1 (budget, licensing, API access), you cannot close the gap to R1-Distill with any amount of filter engineering. You would need to add RL (next chapter block) to break past the teacher ceiling.

## Scores, verbatim

> - Sky-T1-32B-Preview: MATH500 ~82.4%, AIME24 ~43.3%, LiveCodeBench-Easy ~86.3%, GPQA-Diamond ~56.8%.
> - Matches o1-preview on MATH500; beats it on LiveCodeBench Easy; within 2 points on AIME24.

Note the asymmetry: beats o1-preview on LCB-Easy, matches on MATH500, trails on AIME24. QwQ is a better code teacher than reasoner-in-general; this asymmetry carries through to the student.

---

## The follow-up direction

> Follow-up **Sky-T1-32B-Flash** adds RL stage (GRPO on verifiable math) for +3 AIME points; future **Sky-T1-mini** distills to 7B.

The +3 AIME points from adding GRPO-on-verifiable-math to Sky-T1 is the same pattern OpenR1 shows (+3-5 from GRPO). Two independent datapoints now support the claim that SFT on distilled reasoning traces has a saturation point around 3-5 AIME-points below what RL can reach. Ch-44 (verifiable rewards) picks this up.

## Why ch-35 treats Sky-T1 as the cost-floor reference

Every other distillation recipe in ch-35 §5 costs more (Stratos $4.8K, OpenR1 $10K+, R1-Distill undisclosed but presumably much higher). Sky-T1 at $450 is the floor — below this cost point either the teacher is stronger (so free via open weights) or the dataset is already pre-curated (so you are paying zero for generation and only for training). Any "$50 reasoning model" claim after this is either reusing someone else's dataset or running on tiny models.
