---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prosocial-dialog.md
source_url: https://arxiv.org/abs/2205.12688
created_at: "2026-04-23"
---

# Excerpt: Prosocial-Dialog — rules-of-thumb as a rubric anchoring layer

**Source library:** `wiki/raw-data/llm-training/papers/prosocial-dialog.md`
**Year / authors:** 2022 / Kim, Jiang, Sap, Yejin Choi et al. (AI2 / UW Yejin Choi group).

---

## Why this source anchors ch-15

Prosocial-Dialog's operational innovation is the **rules-of-thumb (RoT) layer** — every problematic dialog is anchored to a short, explicit moral/ethical guideline before the assistant response is drafted. This is an answer to a specific ch-15 §1 problem: what do you do when the rubric cannot be written as a flat list of criteria because the right behavior depends on a cultural or situational norm? Prosocial-Dialog's answer is to make the norm itself part of the annotation schema, so that annotators label against a *specific* rule rather than against a generic "be helpful / be safe."

---

## The five-step pipeline — human-in-the-loop construction

```
# prosocial-dialog.md, §4
Step 1 — Problematic-prompt collection
  Crowdworkers + domain experts author prompts covering
  10 problem categories (stereotypes, insults, conspiracies,
  safety risks, unethical plans, ...). ~10K initial prompts.

Step 2 — RoT annotation
  For each prompt, workers identify applicable rules-of-thumb:
    "It's rude to mock someone's appearance."
    "Planning to hurt yourself is a safety concern."
    "Stereotyping based on race is harmful."
  300+ unique RoTs emerge across the dataset.

Step 3 — Response drafting
  Teacher LLM (GPT-3) generates candidate assistant responses
  grounded in the RoT. Multiple candidates per prompt.

Step 4 — Human refinement
  Workers edit/rank responses for prosocial quality, clarity,
  RoT alignment. Only high-quality responses retained.

Step 5 — Multi-turn extension
  Single-turn → multi-turn via follow-up generation + human edit.
```

Notice the order. The RoT annotation happens *before* the response drafting. The workflow explicitly constructs the rubric anchor for each item, and only then does the teacher model produce a response. This is the inverse of UltraFeedback, where responses come first and the judge rates them against a fixed aspect rubric. The prosocial ordering is expensive (two rounds of human work per item, one before and one after generation) but it forecloses a failure mode: a teacher model producing a response grounded in the wrong norm.

For ch-15 §1, RoTs are the extreme case of "enumerate positive and negative exemplars." Each RoT is a micro-rubric; the full dataset uses 300+ of them. A generic "helpful + safe" rubric could not express them all.

---

## The 10-category harm taxonomy

> 10 top-level categories: stereotyping, insults, self-harm, violence
> planning, misinformation, safety risks, ...

This is a *hierarchical* rubric: 10 categories → 300+ RoTs → per-item label. Each level constrains the next. For ch-15 §2 (agreement metrics), the hierarchy changes the κ computation: annotators can disagree at the category level (is this "stereotyping" or "insults"?) and still agree at the RoT level (both mapped to "It's rude to mock appearance"). A good hierarchical κ reports both levels; papers that report only the item-level label lose the structure.

The 2023-2025 successor line ([[wildguard-data]] mentioned in the source, plus Safe-RLHF, BeaverTails) all inherit the hierarchical structure. The number of top-level categories varies (WildGuard has 7, others have ~15), but the pattern is the same: top-level taxonomy for coarse allocation, leaf-level norm for label resolution.

---

## CANARY-400M — what the rubric buys

The paper's trained model, CANARY-400M, is evaluated against a no-anchor baseline (BlenderBot-3B):

> CANARY-400M engages constructively with 89% of problematic prompts, vs BlenderBot-3B at 32%.

The delta is not just from scale (CANARY is smaller). It is from the annotation structure. A generic safety-SFT model trained on refusal data will *refuse* problematic prompts at high rate — which fails the engagement criterion. CANARY learned that problematic prompts are anchored to RoTs, and the RoTs tell the model *what to say instead of refusing*. This is the ch-15 §5 data point: for safety-critical data where the right behavior is nuanced engagement (not refusal), human-annotated RoT data dominates judge-generated refusal data.

---

## The generalization claim — RoT-based reasoning transfers

> Generalization to unseen problem categories strong (RoT-based reasoning transfers).

The hypothesis: by training on explicit norms rather than implicit good/bad labels, the model learns to compose norms for novel situations. The paper's evidence is in-distribution vs out-of-distribution RoT coverage; the 2024 follow-ups (Constitutional AI [[constitutional-ai]], Anthropic's Claude safety training) generalize this hypothesis to principle-based alignment writ large.

For ch-15 §1 (rubric design), the takeaway is that **a rubric written as a list of high-level principles rather than a list of low-level criteria composes better at test time**. This is in tension with ch-15 §2's κ-maximization — high-level principles are more ambiguous, so κ is lower. The tradeoff is real and per-project: safety-critical data typically chooses principle-based (lower κ, better generalization); bulk preference data typically chooses criterion-based (higher κ, worse generalization).

---

## Risks the paper flags

> Crowdsourcing biases: RoT selection reflects US/English-speaking contributor norms.
> Prosocial ≠ refusal: the dataset deliberately engages; downstream users who want
>   pure refusal behavior need additional data.
> RoTs can be over-specified: real conversations blur multiple norms.
> Size (58K) is small vs modern dialog corpora.

Three of these are operational ch-15 §6 issues: cultural bias in the rubric itself (not just the labels), a rubric that optimizes for one use pattern (engagement) at the expense of another (refusal), and over-specification that fails on blended norms. The last one — over-specification — is the ch-15 §1 tradeoff again: fine-grained RoTs fit the annotator pool's cases but may be too narrow for real-world deployment.

---

## Connections

- [[excerpts/hh-rlhf]] — the red-team-preference counterpart; Prosocial chooses principle-anchoring over adversarial preference.
- [[excerpts/openassistant]] — the comparable crowdsourced scale but without the RoT layer; the contrast shows what RoT anchoring buys.
- [[excerpts/tulu-3-sft-mix]] — consumer of Prosocial-style safety data (WildJailbreak, Tulu-3 Safety); the 2024 descendant structure.
- [[excerpts/ultrafeedback-construction]] — the non-safety, judge-driven alternative; Prosocial is what UF cannot be for safety slices.
- [[ch-15]] — this excerpt supports §1 (principle-based rubrics), §2 (hierarchical κ), §5 (when humans override for cultural/contextual safety), §6 (rubric translation as a second rubric problem).
