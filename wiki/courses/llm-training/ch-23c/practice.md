# 23c — Valid Transformations and Length-Controlled Evaluation

<!-- scope: proposed implementation records, controlled comparisons, metrics, and failure repair for policy-conditioned expression, emotion, lexical metrics, and valid tone transformations
     deps: [[read]]
     see-also: [[read]], [[methods]], [[long-tail-papers]]
     sources: [[realtalk]], [[deepdialogue]], [[persona-lexical-diversity]]
-->

## 1. A transformation contract

Before rewriting, store source meaning, required constraints, permitted action, product evidence, operator strategy, assistant tone instruction, and the transformation being requested. Mark what must stay invariant. For the price objection, policy A's clarifying action and the supplied price remain fixed under a conversational-tone rewrite. Rewriting into a documented cost explanation changes the strategy and requires policy B, not only a different tone label.

After rewriting, inspect whether facts, negation, certainty, consent, and references changed. ‘Could we perhaps do Thursday?’ and ‘Book Thursday’ may differ in commitment. ‘Not Thursday’ is not a lexical variant of ‘Thursday’. A generator can accidentally turn uncertainty into a confirmed request; validate semantics before grading stylistic quality.

## 2. Expression bundles with legitimate variation

For each semantic case, generate a small bundle of direct, indirect, fragmented, corrected, and colloquial expressions where coherent. Add code switching only where the intended meaning and interpretation are validated by a competent reviewer. Do not assume a language label guarantees correct bilingual phrasing. Human examples can supply patterns the generator misses, but remove or revise invalid targets rather than automatically treating observed behavior as desired behavior.

An assistant-tone bundle can separately vary formality, brevity, and wording while obeying the same strategy. If a brevity requirement conflicts with a mandatory disclosure, define the priority; the model should not delete required information merely to meet a word limit. If multiple phrasings are valid, use content/action rubrics rather than a single exact string.

## 3. Measure lexical, trajectory, and behavioral coverage separately

Lexical evaluation can use matched-length samples or windows, with tokenization and preprocessing declared. Compare the same tasks across generation methods. Product names and boilerplate can dominate overlap; record whether they were retained or normalized. A lexical score is a descriptive audit, not a complete validity gate. Do not remove all repeated task-critical wording solely to increase novelty.

For trajectories, count which verified transitions and lengths remain after filtering, and manually inspect a sample of classifier assignments. For behavior, test whether equivalent expressions yield the same supported tool arguments and whether changed policies yield the appropriate different strategy. For tone, attach explicit constraints such as brief, formal, or conversational and assess them independently of action validity.

## 4. Validate the validators

Use a human-labeled sample covering common and unusual expressions to examine false rejection and false acceptance. A sentiment classifier trained on neat written language may be unreliable on fragments or code switching. A judge may mistake excessive empathy for quality or penalize an appropriate short answer. Report agreement and disagreement by slice; do not infer population truth from the model's labels.

If a classifier is used only for descriptive counts, preserve uncertain labels as uncertain. If it controls filtering, its errors directly change the training distribution and deserve stricter checks. Keep raw candidates and reasons so an unexpected thin slice can be investigated. Fine-tuning to the same judge that later scores the model risks optimizing a proxy rather than the intended behavior.

## 5. A controlled sales comparison

Compare a baseline semantic dataset with a version adding validated expression/tone bundles, keeping supervised-token budget and task mixture comparable. Reserve unseen expression families and operator-strategy/tone combinations. Report policy adherence, factual validity, argument accuracy, and requested-tone adherence separately; include common baseline cases to detect regressions.

The desired result is not simply a higher diversity score. It is stable valid behavior across relevant language changes and correct adaptation when instructions change. If style scores improve but tool accuracy drops, inspect whether rewrites altered state or changed supervised exposure. If only seen expressions improve, broaden the held-out pattern design before claiming generalization.

[Main lesson](read.md) · [Methods](methods.md) · [Next chapter](../ch-23d/read.md) · [한국어](practice_kor.md).

