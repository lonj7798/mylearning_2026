---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/qwen-3.md
source_url: https://arxiv.org/abs/2505.09388
created_at: "2026-04-23"
---

# Excerpt: Qwen 3 — hybrid thinking as a single-model-two-modes training problem

**Source library:** `wiki/raw-data/llm-training/model-reports/qwen-3.md`
**Report:** Qwen Team, 2025, "Qwen3 Technical Report".

---

## Why this source anchors ch-34 §3

Qwen 3 is the public origin of the `/think` + `/no_think` contract that every 2026 hybrid model (GLM-4.5, Qwen 3.5, DeepSeek V3.1, and a large fraction of open-weight reasoning models) now carries. The interesting claim is *organizational*: the same GRPO and DPO algorithms used by Qwen 2.5 are recomposed into a four-stage pipeline that fuses thinking and non-thinking data into one checkpoint.

---

## The core design stance

From the source (lines 7-8):

> - **Core Insight:** Qwen3 treats "reasoning mode" as a trainable capability inside one unified model family rather than as a separate model class, combining reasoning-stage pretraining, long-CoT cold-start finetuning, RL, and strong-to-weak distillation.
> - **Guideline:** If you want controllable test-time compute, train a single model to support both fast and deep modes, then expose a thinking budget instead of maintaining separate chat and reasoning stacks.

This is the stance that distinguishes Qwen from DeepSeek-R1 (separate reasoning model) and from Phi-4-reasoning (reasoning-only). Qwen 3 commits to one family, two modes, one inference stack.

---

## The four-stage post-training pipeline

From the source (lines 54-58):

> ### Post-training
> - Stage 1-2: build reasoning ability with **long-CoT cold-start finetuning** and **RL** focused on math and coding.
> - Stage 3-4: merge data with and without reasoning paths, then run **general-domain RL**.
> - Smaller models are improved with **off-policy and on-policy strong-to-weak distillation** from larger teacher models.
> - The report explicitly says **distillation from advanced teacher models significantly outperforms RL** for smaller models in both performance and efficiency.

The stages mapped to the SFT-boundary taxonomy in ch-34:

| Stage | Type | Data | Purpose |
|---|---|---|---|
| 1 | SFT (cold-start) | Long-CoT traces from math / code / STEM | Teach the `<think>` format and long-CoT prior |
| 2 | RL | Verifiable rewards on math / code | Push reasoning quality past SFT ceiling |
| 3 | SFT (fusion) | **Mixed thinking + non-thinking, with `/think` + `/no_think` prompts** | Teach the mode-toggle contract |
| 4 | RL | General-domain preference / safety | Chat + tool-use + safety tuning |

**The Stage-3 fusion is the only step that is novel at the pipeline level.** Stages 1, 2, 4 are standard long-CoT-first reasoning training. Stage 3 is the toggle-training step and the one that earns the "hybrid" label.

---

## The hybrid-thinking data format (verbatim reconstruction)

The raw-data source describes the Stage-3 fusion conceptually; the format below is the community-documented form used in open-source Qwen 3 chat-template reproductions and inference examples (consistent with the raw-data description):

**Thinking branch (`/think`):**

```
<|im_start|>user
{prompt} /think
<|im_end|>
<|im_start|>assistant
<think>
{multi-step reasoning: numbered steps, pseudo-code, self-checks}
</think>
{final answer, terse}
<|im_end|>
```

**Non-thinking branch (`/no_think`):**

```
<|im_start|>user
{prompt} /no_think
<|im_end|>
<|im_start|>assistant
<think>

</think>
{direct answer, no reasoning shown}
<|im_end|>
```

The empty `<think>` block on the `/no_think` branch is the signal that teaches the model to emit an empty reasoning trace when the user disables thinking. The model does not learn to *omit* `<think>`; it learns to *produce an empty one*.

**Inference-time thinking budget.** At serving time, the user prompt carries `/think` or `/no_think`. The thinking budget is an upper bound on tokens emitted inside `<think>...</think>`; when the budget is hit, the decoder is forced to emit `</think>` and proceed to the final answer. This is the test-time-compute knob the opening of the raw-data source names.

---

## Pretraining context that enables the recipe

From the source (lines 45-48):

> ### Three-stage pretraining
> 1. **General stage:** over **30T** tokens at sequence length **4096**.
> 2. **Reasoning stage:** about **5T** higher-quality tokens with more STEM, coding, reasoning, and synthetic data, still at **4096**.
> 3. **Long-context stage:** hundreds of billions of tokens at **32768** sequence length, later supporting longer inference contexts.

The middle **reasoning stage** (5T higher-quality tokens) is critical: it is the pretrain-time analogue of Phi's Phase 2 synthetic-heavy data. By the time post-training starts, the base model already has a strong reasoning prior — Stage 1 SFT is then a *cold start* in the narrow sense of surface-level format learning, not from-scratch capability acquisition.

---

## Strong-to-weak distillation for small models

From the source (lines 57-58):

> - Smaller models are improved with **off-policy and on-policy strong-to-weak distillation** from larger teacher models.
> - The report explicitly says **distillation from advanced teacher models significantly outperforms RL** for smaller models in both performance and efficiency.

For Qwen 3-0.6B / 1.7B / 4B dense models, the Stage-2 RL is *replaced* by distillation — both off-policy (teacher trajectories used as SFT targets) and on-policy (student rollouts scored by teacher-derived reward). The report's claim that distillation beats RL at small scale is consistent with Phi-4-reasoning's "short RL is enough when SFT data is good" — both point at the same regime: when the student is small enough that RL exploration is too costly relative to teacher-supervised learning, distillation wins on quality and compute.

---

## What Qwen 3 does not disclose

- Stage 1 long-CoT SFT dataset size and domain breakdown.
- Stage 2 RL hyperparameters (assumed GRPO but group G, clip ε, KL β, rollout scale not reported).
- Stage 3 fusion mix ratio (what fraction thinking vs non-thinking).
- Thinking-budget training objective — is it a separate length-aware reward, or only enforced at inference?
- The Qwen 3.5 report ([[qwen-3-5]]) is silent on all Qwen 3 post-training updates; Qwen 3 remains the algorithmic reference.

---

## Connections

- `[[qwen-3]]` — raw source.
- `[[ch-34]]` — §3 uses this for the hybrid-thinking format and the four-stage pipeline.
- `[[qwen-2.5]]` — predecessor; GRPO and DPO primitives inherited.
- `[[qwen-3-5]]` — successor scaling (MoE 397B-A17B) without post-training re-disclosure.
- `[[deepseek-r1]]` — the RL-pure contrast; Qwen 3 chose hybrid unification instead of a separate reasoning model.
- `[[grpo]]` — the RL algorithm Qwen 3 uses in Stages 2 and 4.
- `[[self-instruct]]` — synthetic data generation concept Qwen 3 scales at pretraining time.
