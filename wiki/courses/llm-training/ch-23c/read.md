---
chapter: ch-23c
course: llm-training
phase: read
---

# Chapter 23c — Tone, Expression, Personas, and Emotion

<!-- scope: policy-conditioned expression, emotion, lexical metrics, and valid tone transformations
     deps: [[../ch-23b/read]]
     see-also: [[methods]], [[practice]], [[long-tail-papers]]
     sources: [[realtalk]], [[deepdialogue]], [[persona-lexical-diversity]]
-->

## Overview

**Core insight.** Language variation is not one axis. The user's wording, the user's changing affect, the assistant's requested tone, and the allowed task behavior can vary independently. Generating many persona names or emotional labels does not establish broad human expression coverage.

**Guideline.** Fix the semantic facts and policy before rewriting expression. Validate that the rewrite preserves or intentionally changes the supported action, and measure task correctness separately from style. Use human reference language when possible, with explicit population limits.

Read the main lesson → [Methods and evidence](methods.md) → [Practice: transformations and measurement](practice.md). [한국어](read_kor.md). The sales cases below are proposed examples, not results from the cited datasets.

## Key Concepts

### 1. Separate what happened from how it was said

‘Thursday at three’, ‘Uh… could we do Thursday, around three?’, and ‘Tuesday—sorry, Thursday, three pm’ can describe the same final schedule under a declared interpretation. Their language differs in hesitation, indirectness, and self-repair. The last utterance also introduces an explicit update, so the history-sensitive task differs even if the final value matches. A dataset should retain both expression tags and state-transition tags.

More words do not necessarily mean more behaviors. Twenty paraphrases of one direct request can increase lexical diversity while leaving cancellation, missing consent, or policy switches unrepresented. Conversely, nearly identical sentences under two policies can require different actions. This is why a single diversity score cannot tell you whether a sales assistant covers the interactions you care about.

### 2. Worked objection case: two operator preferences

Assume the supplied product document states a monthly price of $40 and a one-time activation fee of $20, with no authorized discount. The customer says, ‘That seems expensive.’ Policy A instructs the assistant to first ask a concise clarifying question about the customer's budget, without proposing an unapproved offer. Policy B instructs it to first explain the documented cost components briefly, without assuming the customer cannot afford them.

A valid A response could be, ‘What monthly budget would you like me to work within?’ A valid B response could be, ‘The monthly charge is $40, with a one-time $20 activation fee.’ These are different positive targets for the same objection because the policy is different. The lesson is not that one response is universally better; the operator instruction supplies the missing criterion.

Now change only assistant tone. A formal A response might be, ‘Could you tell me your preferred monthly budget?’ A conversational A response might be, ‘What monthly budget works for you?’ Both perform the same requested clarification. Changing tone must not introduce a discount, invent affordability, or switch to a different objection-handling strategy. If the operator wants a different strategy, change the strategy instruction explicitly.

### 3. User emotion is evidence with uncertainty

The customer could be cautious, frustrated, or simply making a comparison. The text ‘expensive’ does not identify their emotion or purchasing intent uniquely. A user simulator can be assigned an emotional trajectory as a generation condition, but the deployed model observes language and audio rather than an oracle label. Do not put a hidden label into the student input unless deployment supplies an equivalent signal.

Emotion may change across turns: a clear explanation can precede a calmer question, or a repeated misunderstanding can precede sharper wording. Record observable cues and intended generation conditions separately. A dataset with every concern resolved into cheerful agreement is narrow even if it contains many initial emotion labels. Also avoid equating appropriate response with exaggerated empathy; usefulness depends on the task and policy.

### 4. Personas are a generation control, not demographic proof

A persona prompt can encourage different interests and wording, but a generated label such as ‘older customer’ does not verify age or actual language behavior. Identity, dialect, and preference should not be inferred from spelling, accent, or politeness. If demographic evaluation is needed, use supported metadata and a justified collection design, and report the sampling limits.

For expression coverage, observable properties are often more useful than fixed personality classes: terse versus elaborated requests, indirect versus direct wording, code switching, hesitation, corrections, and tolerance for long replies when explicitly stated. Tags can overlap and evolve. Do not force an unusual utterance into an invented identity category merely to fill a taxonomy.

### 5. Sentiment diversity is not emotional appropriateness

REALTALK §4.3 uses sentiment-polarity entropy H=−Σp(x)log₂p(x) over positive, negative, and neutral labels. For a distribution (0.5,0.25,0.25), H=1.5 bits. Equal thirds give about 1.585 bits; all one label gives 0. These illustrations use the source formula and are calculated for this lesson. They measure the spread of classifier labels, not whether an assistant responded appropriately.

A dataset can reach high entropy by inserting irrelevant negative sentences. A classifier can also mistake an indirect request for negative sentiment. Therefore verify labels on human-rated samples and inspect trajectories, not only aggregate entropy. High lexical or polarity entropy does not check product facts, tool calls, policy scope, or real population frequency.

### 6. Filtering may polish away useful variation

A generic quality judge can prefer complete sentences and long explanations. Natural fragments or concise replies can then disappear even when they are fully meaningful and correct. Alternatively, a fragment may genuinely omit a required condition. Inspect which case you have rather than exempting all messy language from verification.

For every slice, record candidate length, retained length, validity rate, and rejection reason. If longer emotional conversations fail more often, the retained dataset may have less trajectory variation than the generator plan suggests. The count of labels in the generation prompt is not the count of valid behaviors represented after filtering.

## Recipe

Build a semantic family with policy-conditioned gold behavior; produce user-expression and assistant-tone variants independently; validate facts, action, and scope again; then audit length-controlled expression coverage. Use actual human wording as a reference rather than assuming persona detail guarantees realism.

## Generalization lens

Test meaning-preserving rewrites for behavioral invariance and strategy-policy changes for required differences. Hold out unfamiliar expressions and policy/expression combinations. A model that varies tone beautifully but invents a price has failed factual correctness; a correct tool call with a disallowed tone is a separate failure.

## Questions

1. Why can two responses to the same objection both be valid training targets?
2. Why does a high sentiment entropy fail to prove appropriate emotion handling?
3. When does a wording rewrite also change the underlying task?

[Methods](methods.md) → [Practice](practice.md) → [23d](../ch-23d/read.md). [Catalog](../../../raw-data/llm-training/long-tail-papers.md).


## Source cards

- [[realtalk]] — [card](../../../raw-data/llm-training/papers/realtalk.md)
- [[deepdialogue]] — [card](../../../raw-data/llm-training/papers/deepdialogue.md)
- [[persona-lexical-diversity]] — [card](../../../raw-data/llm-training/papers/persona-lexical-diversity.md)
