<!-- scope: reasoning-trace synthesis — $450 full recipe for 32B reasoning model from QwQ traces
     deps: [[deepseek-r1]]
     see-also: [[bespoke-stratos]], [[openr1]], [[qwen-qwq-traces]]
-->

# Sky-T1: Train your own O1 preview model within $450
- **Core Insight:** A 32B reasoning model matching o1-preview on math and code is trainable for ~$450 by distilling 17K traces from QwQ-32B-preview and fine-tuning Qwen2.5-32B-Instruct for 19 hours on 8×H100 — dramatically lowering the entry cost for o1-class open research.
- **Guideline:** If your budget is <$1K, use QwQ-32B-preview (fully open weights, no API) as the reasoning teacher, curate ~10–20K traces across math/code/science, run SFT for ~3 epochs, and skip RL entirely; this is the cheapest path to a credible open reasoning model.
- **Authors:** Sky Computing Lab / NovaSky (UC Berkeley) — Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica
- **Year:** 2025
- **URL:** https://novasky-ai.github.io/posts/sky-t1/
- **Relevant topics:** low-cost reasoning distillation, QwQ distillation, open recipe

## Abstract
Sky-T1-32B-Preview is a 32B reasoning model fine-tuned from Qwen2.5-32B-Instruct on 17K curated reasoning traces, most of which are distilled from Alibaba's QwQ-32B-Preview open-weights reasoning model. The NovaSky team released full data, code, and model, and reported a total compute cost of ~$450 (3 epochs × 19 hours on 8×H100). Sky-T1 matches o1-preview on MATH500, beats it on LiveCodeBench Easy, and comes within a few points on AIME.

## Key Contributions
- **Sky-T1-data-17K** — open curated dataset.
- **$450 end-to-end cost disclosure** with full recipe; proof of the "cheap reasoning" feasibility.
- Demonstration that **QwQ-32B-preview**, an open-weights reasoner, is a viable teacher at similar quality to closed R1 at that time.
- Full open release: data, training config (Llama-Factory), model weights.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** problems drawn from multiple sources:
  - APPs + TACO (code) — ~5K.
  - NuminaMath-CoT + AIME/AMC — ~10K math.
  - STILL-2 + miscellaneous science/reasoning — ~2K.
- **Trace generation:** query **QwQ-32B-preview** locally via vLLM for each seed, temperature 0.7, max 8K tokens per trace.
- **Filtering:**
  - **Math:** parse `\boxed{}` and compare to gold via SymPy; reject mismatches.
  - **Code:** execute against public unit tests; reject any-test-failure.
  - **Science / open-ended:** use GPT-4o-mini as LLM-judge to verify final-answer agreement with reference.
- **Reformatting trick:** QwQ's raw output uses a non-standard thinking/answer format; NovaSky reformats to `<|im_start|>…<|im_end|>`-style chat template and removes redundant "Alright, let me think" preambles via a GPT-4o rewriter (this cleanup lifted AIME by +4 points).
- **Output shape:** 17,000 (prompt, long-CoT) pairs; median trace length ~3K tokens.
- **Teacher model:** QwQ-32B-preview (open-weights).
- **Cost / compute:** teacher inference cost negligible (local QwQ on H100s); training ~$450 on RunPod at listed H100 rates.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** median 3K tokens; tail ~10K.
- **Trace style:** QwQ-native long-CoT with reflection markers ("Wait", "Hmm"); post-processed to remove filler.
- **Correctness verifier:** SymPy (math) + unit tests (code) + GPT-4o-mini judge (open-ended).
- **Error mode:** reformatting failures are ~5% of raw traces; these are re-generated not patched.
- **Training config:** 3 epochs, LR 1e-5, seq len 32K (to fit long traces), BF16, FSDP across 8 GPUs via Llama-Factory.

## Quality / diversity evaluation
- Sky-T1-32B-Preview: MATH500 ~82.4%, AIME24 ~43.3%, LiveCodeBench-Easy ~86.3%, GPQA-Diamond ~56.8%.
- Matches o1-preview on MATH500; beats it on LiveCodeBench Easy; within 2 points on AIME24.
- Follow-up **Sky-T1-32B-Flash** adds RL stage (GRPO on verifiable math) for +3 AIME points; future **Sky-T1-mini** distills to 7B.

## Risks + gotchas
- **QwQ teacher quality is lower than R1** on hardest problems — Sky-T1 underperforms Bespoke-Stratos and R1-Distill on AIME25.
- **Narrow eval:** reported on math+code only; no safety or instruction-following eval.
- **Reformatting brittleness:** the GPT-4o-mini rewriter occasionally distorts reasoning steps.
- **$450 is training-only:** excludes teacher-inference compute (which is negligible only because they ran QwQ locally).

## Connections
- Parallel 2025 R1/QwQ distill efforts: [[bespoke-stratos]], [[openr1]].
- Teacher: [[qwen-qwq-traces]] (QwQ-32B-preview).
- Curated-small-set lineage: [[s1]], [[limo]].
- RL follow-up: [[rlvr-tulu3]] / [[grpo]] for Sky-T1-Flash.
