# Weaknesses
<!-- scope: observed areas where the learner needs extra push or scaffolding
     deps: [[learning-style]]
     see-also: [[strengths]], [[push-tactics]]
-->

## Concept gaps

### Factual specifics in comparison tables (ch-18)

Learner holds the structural frame of a pipeline reliably but drops per-paper factual specifics under recall pressure. In ch-18 discuss Probe 1, the learner incorrectly assumed Alpaca had a Stage 4 verifier — a direct contradiction of the chapter's central claim (read.md line 97: "Self-Instruct: Minimal — no gold-answer checker"). The framework was installed; the specific cell in the 4×6 table was not retained.

**Pattern**: when the chapter's claim is "this pipeline is the canonical example of a *missing* stage," the learner is at risk of assuming the stage exists because the framework exists. Teacher should probe negative-space facts ("what does this pipeline *not* have?") as a standard check.

### Stage-1 sub-type naming under cold recall

Two of four Raschka stage-1 sub-types (bootstrap, full-generate) required prompting or were initially punted in the ch-18 summary. The taxonomy was present in the Q&A file (Q&A file covers bootstrap and full-generate directly) but was not fluently recalled in free-form writing. The names are retrieved under probe; they are not yet automatic.

## Avoidance patterns

### Imprecise vocabulary substitution (self-identified 2026-05-08, llm-training ch-18 summary)

The learner tends to substitute near-synonyms for chapter terminology when writing summaries, even when they correctly used the precise term in their own [[qa]] notes earlier in the same chapter. Examples from ch-18 summary:

- "Sampling" instead of **Generate** (loses rewrite/backtranslate cases)
- "minimum bar" instead of **surface check**, "more complicated" instead of **ground-truth check** (loses the cheap-vs-expensive axis; misses that the real axis is connection to ground truth, not LLM involvement)
- "seed data" instead of **anchor** (regresses qa.md Q7's anchor ⊇ seed correction)

Quality regresses from `qa.md` → `summary.md`. The learner *can* hold the precise distinctions when probed, but defaults to imprecise vocabulary when writing alone. This gap closes quickly under real-time flagging (summary v2 after mid-session corrections was markedly more precise than v1).

### Punting on incomplete recall

When the learner does not immediately remember an item, they default to "I heard we will talk about this later" rather than re-reading the relevant section. In ch-18 summary, two of four Raschka stage-1 sub-types (bootstrap, full-generate) were punted this way despite both being explicitly covered in `read.md` lines 158-159.

**Refinement from ch-18 discuss**: the punting pattern does not persist into the discuss phase — when forced to engage (Probe 1), the learner attempts an answer rather than deflecting. Punting appears to be a summary-writing habit under low-stakes conditions, not a universal avoidance behaviour.

### Skipping operational deliverables

The chapter's most operational sections (section 7 reading checklist, section 8 connections) were initially left blank or marked "skip" before being filled in after pushback. Pattern: the learner reads the prose-heavy parts and skips the structured-list parts that are meant to be carried forward as reusable tools.

## Scaffolding needs

### Real-time flagging during summary writing

The learner asked the teacher to point out these patterns *as they occur*, not wait for end-of-summary review. When the learner produces a summary or writes free-form prose, the teacher should:

1. **Term check**: flag substitutions for chapter-defined vocabulary in real-time. Do not let "sampling" stand for "generate."
2. **Count check**: when the chapter says "three corollaries / four sub-types / six stages," confirm the learner's section has the matching count. Flag punts immediately.
3. **qa.md cross-reference**: when summary regresses from qa.md precision, name the regression and point to the qa.md entry that was sharper.
4. **Negative-space probe**: for chapters where the key claim is "this pipeline *lacks* stage X," probe that gap explicitly during discuss.

This turns the post-summary critique into mid-summary corrections, which the learner has explicitly requested as preferred guidance style.

See [[session-log]] for session evidence used to identify these weakness patterns.
