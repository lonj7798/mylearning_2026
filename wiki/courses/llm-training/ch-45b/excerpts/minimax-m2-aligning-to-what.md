---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: "Aligning to What? Rethinking Agent Generalization in MiniMax M2 (MiniMax-AI, 2025-10-30)"
source_url: https://huggingface.co/blog/MiniMax-AI/aligning-to-what
created_at: "2026-09-15"
---

# Excerpt: Aligning to What? Rethinking Agent Generalization in MiniMax M2

- **Organization:** MiniMax-AI
- **Year:** 2025 (post dated 2025-10-30)
- **Source type:** official blog (qualitative; the post prints no numbers)
- **Used in:** [[read]] §§2.3 and 6, "Why this chapter matters", Generalization lens

## The generalization claim
The post defines agent generalization as robustness to perturbations along the whole trajectory, and lists five
places where a perturbation can enter:

1. "The **Tool Info** and available toolset"
2. "The **System Prompt** defining the agent's persona and rules"
3. "The **User Prompt** and its specific goal"
4. "The **Environment** itself (files, codebases, APIs)"
5. "The **Tool Responses** returned at each step"

On its own earlier approach (verbatim): "Our old 'tool scaling' approach only addressed the first item. It
ignored perturbations in all the other parts of the process."

The symptom the post opens with (verbatim): "the same model can feel brilliant in one framework and useless in
another."

## Interleaved thinking
Thinking can happen "at any point during a task, not just once at the beginning", which the post ties to handling
"constant, unpredictable perturbations from the outside world (i.e., tool outputs)". Callers must "retain the
full session history, including the thinking steps"; discarding it degrades performance, which the team observed
in community deployments.

## Status of the evidence
This is an official post-mortem with no ablation, no benchmark table and no numbers. Under the course evidence
rules it is **official** but qualitative: it can support a design claim, not a quantitative one. The companion
post [[minimax-m2-interleaved-thinking]] (2025-11-03) supplies the numeric comparison for the
keep-versus-discard-thinking question.

## Verification
- Checked on 2026-09-15 against https://huggingface.co/blog/MiniMax-AI/aligning-to-what (post dated 2025-10-30);
  re-read in full on the same date when the chapter's citations were re-verified.
- Correction carried into [[read]] §2.3: the statement that the OpenAI Chat Completion API has no field for
  returning reasoning content is **not** in this post; it is in [[minimax-m2-interleaved-thinking]]. This post
  states only that callers "must retain the full session history, including the thinking steps".
- Not reported by the post: model size, data pipeline sizes, any ablation of the perturbation axes, and any
  evaluation number.
