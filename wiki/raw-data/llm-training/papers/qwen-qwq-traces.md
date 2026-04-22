<!-- scope: reasoning-trace synthesis — QwQ-32B-preview as an open-weight reasoning teacher
     deps: [[deepseek-r1]]
     see-also: [[sky-t1]], [[bespoke-stratos]], [[openr1]]
-->

# QwQ reasoning-trace synthesis (QwQ-32B-Preview / QwQ-32B)
- **Core Insight:** The Qwen team's QwQ family is the first fully open-weights reasoning model producing long-CoT o1-style traces at 32B scale; its open checkpoints make it the cheapest high-quality reasoning teacher available, catalyzing the $450-recipe distill wave of early 2025.
- **Guideline:** When budget rules out R1 API calls or you need a self-hostable reasoning teacher, run QwQ-32B locally via vLLM, sample at temperature 0.7 with max_new_tokens ≈ 16K, and use `<thought>…</thought>` parsing to separate reasoning from final answer.
- **Authors:** Qwen team (Alibaba DAMO)
- **Year:** 2024 (QwQ-32B-Preview Nov 2024) / 2025 (QwQ-32B final March 2025)
- **URL:** https://qwenlm.github.io/blog/qwq-32b-preview/ ; https://qwenlm.github.io/blog/qwq-32b/
- **Relevant topics:** open-weight reasoning teacher, long-CoT, distillation source, QwQ

## Abstract
QwQ-32B-Preview (released 2024-11) and the subsequent QwQ-32B (released 2025-03) are Alibaba Qwen-team reasoning models that produce o1-style reflective long-CoT. Both are fully Apache-2.0 open-weight. QwQ-32B-Preview reaches MATH500 90.6%, AIME24 50.0%, and LiveCodeBench 50.0%; the final QwQ-32B (post-RL, 32B dense) rivals DeepSeek-R1-671B on math and code benchmarks. The models have become the standard open teacher for 2025-era reasoning distillation.

## Key Contributions
- **First open-weight o1-style reasoner at 32B.** Full checkpoint release, not distill-only.
- Long-CoT output with explicit reflection tokens ("Wait", "Hmm", "But…").
- QwQ-32B uses a two-stage RL recipe (outcome-reward then general-reward) detailed in the blog.
- De-facto teacher model for Sky-T1, Open-R1, Bespoke-Stratos (as alternative to R1).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### QwQ's own training (self-synthesis)
- **Base:** Qwen2.5-32B.
- **Stage 1 — outcome-reward RL:** RL on math + coding problems with a rule-based verifier (exact-match math, unit-test code). No PRM. Reports 8,000+ RL training steps.
- **Stage 2 — general-reward RL:** add a general reward model + rule-based verifiers for instruction-following and tool-use; continue RL.
- QwQ's own synthetic training data is not released as an SFT corpus — the model itself is the artifact.

### Using QwQ as a teacher (distillation pipeline)
- **Seed input:** any problem with a ground-truth verifier (math with gold answer, code with tests).
- **Trace generation:** `vLLM serve QwQ-32B`, then prompt with the QwQ chat template. Sample at T=0.6–0.7, top_p=0.95, max_new_tokens ≥ 16K.
- **Filtering:** same downstream verifier used in Sky-T1 / Open-R1 / Stratos — SymPy for math, test execution for code.
- **Parsing:** QwQ emits thinking wrapped in `<|im_start|>assistant\n<thought>…</thought>\n<answer>…</answer>`; downstream training usually reformats.
- **Output shape:** trace length median ~3K tokens, heavy tail to 20K. Reflective long-CoT.
- **Teacher model:** QwQ-32B-Preview (earlier) / QwQ-32B (final, stronger).
- **Cost / compute:** ~8 sec/trace on 1×H100 at 3K tokens; fully local inference is feasible.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** median 3K, P95 ~10K, max ~30K tokens.
- **Trace style:** reflective long-CoT. More English-dominant than R1 (which sometimes code-switches into Chinese).
- **Correctness verifier:** external to QwQ — downstream users apply rule-based verifiers. QwQ itself was RL'd with similar verifiers.
- **Comparison to R1:** QwQ traces are ~30% shorter on average, less likely to code-switch, but slightly lower accuracy on hardest AIME problems.
- **Known quirks:** QwQ occasionally falls into reasoning loops ("Wait, let me reconsider" repeated); downstream filtering should cap trace length and drop loop-detected samples.

## Quality / diversity evaluation
- QwQ-32B-Preview: MATH500 90.6, AIME24 50.0, LiveCodeBench 50.0, GPQA-Diamond 65.2.
- QwQ-32B (final): AIME24 ~79, MATH500 ~97, LiveCodeBench ~63 — roughly matches DeepSeek-R1-671B despite 20× fewer parameters.
- Alternate teacher in Sky-T1 ($450 recipe) and in many small-lab distillations.

## Risks + gotchas
- **Reasoning-loop failure mode:** QwQ can generate 20K tokens of repeated self-correction without progress; always cap max tokens and detect loops.
- **Chinese leakage:** especially in QwQ-32B-Preview; downstream training on English tasks should filter on language detection.
- **Not a chat model:** QwQ is reasoning-specialized; general-chat quality is weaker than Qwen2.5-Instruct.
- **Preview model is inferior teacher:** prefer the final QwQ-32B (March 2025) over QwQ-32B-Preview for distillation.

## Connections
- Cheaper alternative teacher to [[deepseek-r1]].
- Primary teacher in [[sky-t1]]; alternative teacher in [[openr1]] and [[bespoke-stratos]].
- Related: [[qwen-long-context-synth]] (Qwen family long-context data pipeline).
- RL recipe cousin: [[deepseek-r1]] two-stage RL.
