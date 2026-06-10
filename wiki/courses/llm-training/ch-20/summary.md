<!-- chapter: ch-20 — learner summary
     deps: [[read]], [[qa]], [[qa-deep]], [[qa-deep-2]]
     scope: learner-authored distillation; written alone, no teacher input on content
-->

# Ch-20 — Learner Summary

## Core insight (one sentence)

sample-filter-SFT pipeline and this type of distillation are not just extracting the knowledge from teacher model, but also mimicking the behavior of the model. 

## What distillation actually is in 2023-2025

- 2017-2022 logit-matching distillation
- 2023-2025 sample-filter-SFT distillation (output distribution = teacher value)
previous logit-matching distillation provides the knowledge only, not a behavior, reasoning and thinking path. 
but 2023-2025 sample-filter-SFT distillation provides reasoning path. 
this shift shows teacher model's performance is come from not only knowledge, but also the reasoning chain, and other behaviors.
so we need to design the pipeline to extract the reasoning chain, and others. 
and also we need to filter out the wrong reasoning path. 

## The teacher-side recipe (3 stages)

1. Elicit reasoning trace from teacher model, 
2. Filtering the trace, 
3. Inherit/avoid teacher's quirks

## §1 Orca — explanation traces via system-prompt scaffolding

Orca
instead of using a single system message, adapts 16 different types of system message. 
this kind of method helps dstribute the model's reasoning path. providing the diverse veiw points. 
basically extract the pattern of reasoning process, instead of extracting the surface style. 
the problem of Orca dataset is the model use kind if over-reasoning every turn (no matter how the problem is easy)

Orca 2
prompt erasing + 5-strategy variety
Previous problem of Orca was over-reasoning. and Orca 2 resolve this issue with prompt erasing + 5-strategy variety.
by removing the prompt, the model can learn to choose the reasoning strategy by itself.


## §2 DSBS — multi-task rationale supervision

consist two types of datasets. one with [label] and second one with [rationale]
with [label] incudes simple answer, and with [rationale] includes reason chain. 
and when train the model, use mixture of both [label] and [rationale] datasets.
and also with prefix [label] during inference, can control the reasoning. 
also with [rationale] prefix data, it improve the performance of the model with [label]-only inference. 
training the single model with both dataset, improve the [label] performance

## §3 R1-distill — what the teacher actually generated

R1 is reasoning model, not only chat model. 
4-stage R1 teacher pipeline → student SFT

## §4 Open-reproductions — Stratos / Open-R1 / Sky-T1 비교

difference is comming from how to curate the data, and what kind of filter they use.

- math filter: \boxed{} + SymPy (all)
- code filter: unit test (Stratos and Sky-T1 only)
- science filter: LLM judge (Stratos: gpt-4o, Sky-T1: gpt-4o-mini)
- Format filter: existance of <think> tag (Stratos, Open-R1)
- "curation beats scale" — Stratos 17K가 R1-distill 800K에 근접
- "verifier determines upper bound"
- "stronger ≠ better teacher" — QwQ > R1

## §5 Teacher-bias inheritance

Output format, Aha point, Refusal patterns are inherited from teacher to student model. 
so the key point is how we build the good filter and verifier.
outcome-only filter let wrong reasoning pass. and it also belong to teacher bias inheritance.

## §6 OpenThoughts — recipe meta-experiment

- single question with multi-answers sampling. 
- Source concentration > diversity
- No answer-filter over-filtering
- Q-side > A-side filtering
- Domain-sensitive dedup
- QwQ > R1 as teacher (if student is qwen family)

## §7 Licensing reality

most of closed model, they don't allow the community to use their model's output.
so we can't use them as teacher model. that's why R1 and QwQ are dominant teacher model.

## Control-axis framework (Q6 lineage 9 axes)

  1. Reasoning expression style (Orca v1)
  2. Training/inference separation (DSBS)
  3. Strategy selection (Orca-2)
  4. Reasoning emergence + grounding (R1)
  5. Verifier coverage (Stratos)
  6. Trace length (Open-Thoughts §4.3)
  7. Teacher-student distribution match (Open-Thoughts §4.3)
  8. Multi-turn / format preservation (uncontrolled)
  9. Question-side curation (OpenThoughts §6)

## Failure modes (cross-cutting)

Cannot verify the reasoning path (from math), if we verify the final answer only. 
Clear celing of teacher Model. 
if teacher and student model has a huge gap in vector space, no matter how teacher model is good, it may not learn from teacher model a lot. 
if we cannot filter out the reasoning path, student model may learn the wrong reasoning path, useless trace, aha-moments, etc. 
also most of the dataset we talked, is focusing on single round instead of multi-round conversation. so it may easly regress to single-turn conversation. 

## Connections to other chapters

[[ch-19]] pass@k → OpenThoughts ≥16× sampling instantiation
[[ch-21]] Phi/textbook distillation
[[ch-22]] quality/gradient selection
[[ch-24]] process reward (wrong-question-correctly 방어)
[[ch-25]] long-conversation (single-turn regression 대처)
[[ch-26]] tool-calling (multi-turn format 필수)
[[ch-44]] RLVR (verifier-grounded reward의 RL 적용)

## Open questions / what I'm still unsure about

still question left. How to break the ceiling of the teacher model clearly? 
and how to mitigate the teacher bias? (honestly looks like ch-18 to ch-20 don't provide clear answe how to break celing.)
