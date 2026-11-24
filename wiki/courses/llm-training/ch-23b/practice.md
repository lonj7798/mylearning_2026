# 23b — Dataset Records and a Skill-Boundary Experiment

<!-- scope: proposed implementation records, controlled comparisons, metrics, and failure repair for state updates, policy scope, counterfactual skill families, and causal supervision
     deps: [[read]]
     see-also: [[read]], [[methods]], [[long-tail-papers]]
     sources: [[user-oriented-dialogue-generation]], [[consistentchat]], [[josh-dialogue-self-training]], [[multichallenge]]
-->

## 1. A record that explains its gold action

Proposed fields: `family_id`, visible messages with turn IDs, active skill text/version/activation turn, global policy, product evidence version, tool schema, customer state with field provenance, missing fields, allowed next actions, selected target, and validation notes. Task and coverage tags are metadata; the model need not receive the expected action or annotation state if those are labels. Avoid putting target-derived fields into inference inputs.

A valid callback record points the date to the correction turn, the time to the time turn, and the action eligibility to the active policy. An explicit state representation can help annotation and validators even if the deployed model consumes only text. Keep the original messages so a stale or faulty state summary can be detected. If an annotation updates the state incorrectly, a perfectly learned target still teaches the wrong behavior.

## 2. Construct semantic cases before rewriting tone

Begin with coherent requirements and a declared policy. Derive the target and verify required fields. Then generate surface variations. Recompute the target after a material change: an added cancellation changes eligibility; a politeness change usually does not. Validate after rewriting because a generator can accidentally add consent or erase a correction. Keep cases where the correct response asks a question rather than calls a tool.

Do not maximize all combinations blindly. Some policy/state combinations are impossible or lack a unique allowed action. If several answers are acceptable, record the constraint set rather than pretending one exact sentence is mandatory. Human review should focus on whether the action and content follow the instructions, with style evaluated separately.

## 3. Different training and selection choices

SFT on corrected calls teaches a positive target under the visible context. DPO uses a preferred and rejected output for the same input; its value depends on valid preference labels. A rejected call with an unsupported date can be useful, but an alternative equally valid response is not automatically rejected. Include the relevant policy in both arms.

LESS targets examples aligned with a reference task's learning direction. G-Vendi characterizes within-set gradient diversity. They neither create missing scenarios nor change the objective by themselves. A targeted correction problem makes LESS plausible, not automatically superior. Compare with random and balanced selection under the same budget and preserve varied valid contexts. Separate selection gain from SFT/DPO gain rather than changing both without a control.

## 4. Holdouts that test actual generalization

Split families before variant generation. Create separate tests for new values, new dialogue patterns, new products with supplied evidence, new skill descriptions, and unseen policy/state combinations. New values test extraction; new patterns test context use; new policies test instruction interpretation. Keep selection-reference and development cases out of the final test.

Evaluate the callback family using call/no-call accuracy, schema validity, per-argument exactness under your canonicalization rules, provenance support, and policy timing. If multiple tools/actions are legitimate, grade against an allowed set. Check the absence of an unauthorized call, not only the correctness of a call that happened. Fluent replies should not rescue invalid actions.

## 5. Failure analysis and repair

If dates are wrong only after interruptions or topic changes, inspect context and field provenance. If all v2 cases are wrong, inspect policy scope and activation timing. If the response follows an expired local rule, add verified switch-boundary families rather than merely more examples of the new skill alone. If new products fail but old ones work, inspect source grounding and product split leakage.

The test cannot prove arbitrary future-skill competence. It can identify specific dependencies and show whether your intervention improves them. Report common and rare/stress slices separately, including valid no-action cases, and preserve uncertainty for small counts. No mixture proportion or data volume in these papers guarantees your desired sales behavior.

[Main lesson](read.md) · [Methods](methods.md) · [Next chapter](../ch-23c/read.md) · [한국어](practice_kor.md).

