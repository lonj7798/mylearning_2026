---
chapter: ch-51a
course: llm-training
phase: read
excerpt_of: "Aligning to What? Rethinking Agent Generalization in MiniMax M2 (MiniMax-AI, post dated 2025-10-30)"
source_url: https://huggingface.co/blog/MiniMax-AI/aligning-to-what
created_at: "2026-09-15"
note: "No library card exists for this slug on 2026-09-15; the same excerpt is held by ch-45b. This copy records only what ch-51a cites."
---

# Excerpt: the five perturbation axes of agent generalization

- **Organization:** MiniMax-AI
- **Source type:** official blog (qualitative; the post prints no numbers)
- **Used in:** [[read]] §5, Generalization lens, Common mistakes

## The five places a perturbation can enter

The post defines agent generalization as robustness to perturbations along the whole trajectory and lists five
axes (verbatim labels):

1. "The **Tool Info** and available toolset"
2. "The **System Prompt** defining the agent's persona and rules"
3. "The **User Prompt** and its specific goal"
4. "The **Environment** itself (files, codebases, APIs)"
5. "The **Tool Responses** returned at each step"

On the team's own earlier approach (verbatim): "Our old 'tool scaling' approach only addressed the first item. It
ignored perturbations in all the other parts of the process."

The symptom the post opens with (verbatim): "the same model can feel brilliant in one framework and useless in
another."

## Status of the evidence

An official post-mortem with no ablation, no benchmark table, and no numbers. Under the course evidence rules it
is **official** but qualitative: it can support a design claim about which axes to perturb, not a quantitative
claim about how much each axis moves a score.

## Verification

- Checked on 2026-09-15 against https://huggingface.co/blog/MiniMax-AI/aligning-to-what (post dated 2025-10-30).
- Not reported by the post: model size, any ablation of the perturbation axes, any evaluation number.
