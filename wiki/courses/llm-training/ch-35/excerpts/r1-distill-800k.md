---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/deepseek-r1.md + wiki/raw-data/llm-training/model-reports/deepseek-r1.md + wiki/raw-data/llm-training/blogs/deepseek-r1-distill-synth.md
source_url: https://arxiv.org/abs/2501.12948 ; https://github.com/deepseek-ai/DeepSeek-R1
created_at: "2026-04-23"
---

# Excerpt: R1-Distill — what the 800K corpus actually is and why SFT-only works

**Source libraries:** `wiki/raw-data/llm-training/papers/deepseek-r1.md`, `wiki/raw-data/llm-training/model-reports/deepseek-r1.md`, `wiki/raw-data/llm-training/blogs/deepseek-r1-distill-synth.md`
**Paper + release:** DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability via RL*, 2025 (Nature 645:633-638).

---

## Why this source anchors ch-35

Ch-20 already covered the teacher-side R1 pipeline. This excerpt captures what R1-distill looks like from the **student** side: once DeepSeek emits the 800K corpus, what does consuming it via pure SFT actually look like, and why is it sufficient without any RL on the student.

---

## The 4-stage teacher pipeline that produces the 800K

From `model-reports/deepseek-r1.md`:

> 1. **Cold-start SFT** on ~800K curated reasoning examples with human-readable CoT format — fixes readability.
> 2. **Stage-1 Reasoning RL** with GRPO + rule-based rewards:
>    - **Learning rate:** 3e-6
>    - **KL coefficient:** 0.001
>    - **GRPO clip ratio (eps):** 10 (intentionally loose — DeepSeek argues tight clipping destroys exploration)
>    - **Sampling temperature:** 1.0 for rollouts
>    - **Rollouts:** 16 samples per prompt (group size G=16)
>    - **Max generation length:** 32,768 tokens
>    - **Batch size:** 32 unique prompts/step -> 32 * 16 = 512 training samples/step
> 3. **Rejection-sampling SFT:** Use stage-1 RL model to generate data; filter via V3 judge; ~600K reasoning + 200K non-reasoning.
> 4. **Stage-2 Alignment RL:** second RL run with helpfulness + harmlessness preference rewards for general-purpose alignment.

Stage 3 is the one that produces the 800K distill corpus. The stage-1 RL model (already a strong reasoner) samples multiple traces per prompt; a V3 judge filters for readability + correctness; the kept set is the union of 600K reasoning + 200K non-reasoning.

The GRPO hyperparameters are attested: LR 3e-6, KL 0.001, clip ε = 10 (intentionally loose), T = 1.0 at rollout, G = 16, max seq 32,768, 32 unique prompts per step. These are the ch-40 (GRPO) reference numbers.

## What's actually in the 800K

From `blogs/deepseek-r1-distill-synth.md`:

> The repo states that the Qwen-based distills use 800k samples curated with DeepSeek-R1, and that the released student family spans 1.5B, 7B, 8B, 14B, 32B, and 70B models.

The per-source breakdown is not publicly itemized, but the rough composition inferred from R1's task coverage is:

| Slice | Approx size | What it is | Verifier used (teacher-side) |
|---|---|---|---|
| Math (problem -> R1 long CoT) | ~200-300K | NuminaMath / olympiads / AIME-style | Exact-match on final boxed answer (SymPy) |
| Code | ~200-300K | LeetCode / APPS / CodeContests | Public unit-test execution |
| Logic / science reasoning | ~50-100K | GPQA-style, logical puzzles | LLM-judge (V3 or V3-reasoning) |
| Non-reasoning SFT | ~200K | Writing, roleplay, translation, Q&A | No verifier; V3-judge for quality filtering |

## The students — pure SFT, six bases

From `model-reports/deepseek-r1.md`:

> 800K reasoning traces from R1 used to SFT Qwen-2.5 (1.5B/7B/14B/32B) and Llama-3 (8B/70B) students. No RL on students; pure SFT. Distilled-R1-Qwen-32B beats o1-mini on MATH-500 and AIME.

The explicit claim: **pure SFT on R1 traces transfers reasoning**. Six dense bases, six distilled students, no RL, no RM, no DPO on the student side. The report attributes this to the fact that dense students with enough base-model capacity (32B+ on Qwen-Math) can absorb R1's reasoning *pattern* from the trace format alone.

From `blogs/deepseek-r1-distill-synth.md`:

> dense students benefit more from copied reasoning structure than from trying to rediscover that structure via their own RL.

This is the chapter-worthy claim. Running GRPO from scratch on a dense 32B student needs enormous compute for a weaker result than one-epoch SFT on R1 traces.

## Why SFT-only suffices for the distilled students

Three mechanisms, synthesizing across the three sources:

1. **R1 already did the RL.** The cost of discovering long-CoT structure was paid in Stage 2 of the teacher pipeline. The student inherits the discovered structure; it does not need to re-discover it.
2. **The traces are on-distribution for a strong base.** When the student base is Qwen2.5-Math or Llama-3.x-70B, the R1 traces land in a reasoning capacity the base already partially possesses. SFT activates it; RL would only refine.
3. **The rejection-sampling filter is tight enough.** V3-judge + correctness filter drops obviously broken traces. What survives is high-signal supervision that SFT loss can learn from directly.

## Why R1-Distill is likely the **terminal** version of straight SFT reasoning transfer

From `model-reports/deepseek-r1-followup.md` (companion source):

> DeepSeek-V3.1 (Aug 2025) subsequently merged V3 and R1 into a single hybrid checkpoint — making R1-0528 likely the final standalone R1 descendant.

The industry shift after R1-Distill is toward **hybrid thinking mode** (Qwen 3, ch-34) where one checkpoint toggles between reasoning and non-reasoning modes. R1-Distill as a separate family of reasoning-only models is unlikely to be continued. For ch-35 this is the historical framing — distillation-SFT-only is the *2025 peak*, not the future baseline.

---

## What the sources do *not* disclose

Per `blogs/deepseek-r1-distill-synth.md`:

> The public docs do not fully enumerate the distill corpus construction, but they make the key invariant clear: keep only teacher outputs that preserve the reasoning pattern and final answer quality.

Missing: per-domain slice ratios inside the 600K reasoning set, rejection-sample yield per domain, V3-judge prompt, data-order within SFT epochs. Open reproductions (Stratos, OpenR1, Sky-T1 — ch-20 §4 and ch-35 §4-5) were built to fill these gaps; reading them is how you reconstruct what DeepSeek did not publish.
