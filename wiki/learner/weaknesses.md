# Weaknesses
<!-- scope: observed areas where the learner needs extra push or scaffolding
     deps: [[learning-style]]
     see-also: [[strengths]], [[push-tactics]]
-->

## Concept gaps

*(populated by profiler after discuss phases)*

## Avoidance patterns

### Imprecise vocabulary substitution (self-identified 2026-05-08, llm-training ch-18 summary)

The learner tends to substitute near-synonyms for chapter terminology when writing summaries, even when they correctly used the precise term in their own [[qa]] notes earlier in the same chapter. Examples from ch-18 summary:

- "Sampling" instead of **Generate** (loses rewrite/backtranslate cases)
- "minimum bar" instead of **surface check**, "more complicated" instead of **ground-truth check** (loses the cheap-vs-expensive distraction; misses the real axis)
- "seed data" instead of **anchor** (regresses qa.md Q7's anchor ⊇ seed correction)

Quality regresses from `qa.md` → `summary.md`. The learner *can* hold the precise distinctions when probed, but defaults to imprecise vocabulary when writing alone.

### Punting on incomplete recall

When the learner does not immediately remember an item, they default to "I heard we will talk about this later" rather than re-reading the section. In ch-18 summary, two of four Raschka stage-1 sub-types (bootstrap, full-generate) were punted this way despite both being explicitly covered in `read.md` lines 158-159.

### Skipping operational deliverables

The chapter's most operational sections (section 6 corollaries, section 7.1 reading checklist) were left blank or filled with vague observations. Pattern: the learner reads the prose-heavy parts and skips the structured-list parts that are meant to be carried forward.

## Scaffolding needs

### Real-time flagging during summary writing

The learner asked the teacher to point out these patterns *as they occur*, not wait for end-of-summary review. When the learner produces a summary or writes free-form prose, the teacher should:

1. **Term check**: flag substitutions for chapter-defined vocabulary in real-time. Do not let "sampling" stand for "generate."
2. **Count check**: when the chapter says "three corollaries / four sub-types / six stages," confirm the learner's section has the matching count. Flag punts immediately.
3. **qa.md cross-reference**: when summary regresses from qa.md precision, name the regression and point to the qa.md entry that was sharper.

This turns the post-summary critique into mid-summary corrections, which the learner has explicitly requested as preferred guidance style.

See [[session-log]] for session evidence used to identify these weakness patterns.
