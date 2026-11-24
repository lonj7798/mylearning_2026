---
chapter: ch-23a
course: llm-training
phase: read
---

# Chapter 23a — Rarity, Knowledge, and Synthetic Distribution

<!-- scope: rarity units, sampling support, retention bias, and source-grounded knowledge coverage
     deps: [[../ch-23/read]]
     see-also: [[methods]], [[practice]], [[long-tail-papers]]
     sources: [[long-tail-knowledge]], [[graphgen]], [[knowledge-density-synthesis]], [[data-value-scaling]], [[long-tail-knowledge-survey]]
-->

## Overview

**Core insight.** Long-tail coverage is a property of a distribution and a counting unit. It cannot be established by counting categories or by making a dataset large. Generation changes what candidates exist; selection changes which candidates survive; training changes what the model learns. Failure at any of these points can leave a tail unserved.

**Guideline.** Define the reference population first, examine candidate and retained distributions separately, and evaluate actual behavior on held-out slices. Preserve a real-data anchor where available, but do not treat a fixed real/synthetic percentage as a universal solution.

Read this lesson first, then [Methods: evidence and mechanisms](methods.md), then [Practice: construction and controlled comparisons](practice.md). [한국어](read_kor.md). These three pages form one chapter. Published findings are labeled **Result (single study)**, cross-domain applications **Interpretation**, and untested transfer **Open question**.

## Key Concepts

### 1. What exactly is rare?

Suppose your sales assistant receives 10,000 conversations. In 80 conversations a customer changes an appointment date twice. That is 0.8% of conversations. If those conversations contain 240 correction turns among 100,000 total turns, the turn share is 0.24%. Neither denominator is wrong; they answer different questions. If deployment logs are unavailable, you cannot infer either rate from 80 examples created by your generator. Call the scenario *hypothesized rare* and record why it matters.

Rarity is also relative to time, product, and customer population. A pricing question can be common for one product and rare for another. Aggregate rarity can conceal a frequent problem for a small customer group. Count both the overall frequency and relevant conditional frequency when the data support them; do not guess a customer's group from tone or spelling.

**Difficulty** concerns whether a model succeeds. **Novelty** concerns whether a condition appeared in training. **Diversity** concerns the spread of a set. These need not move together. A rare request to repeat a price can be easy. A common request combining several constraints can be difficult. A new skill description can be novel to training but routine after a client rollout. None should silently substitute for frequency.

### 2. Three different probability distributions

Let P be the reference distribution of actual conversations, Q the generator's candidate distribution, and R the retained distribution after checking and selection. The model is trained on R, not P or Q. A good average score on R does not establish good performance on P. For example, P may contain many fragments, while Q produces polished paragraphs and the filter further favors neat text.

A mismatch can occur before filtering: the generator rarely creates a twice-corrected date. It can also occur after filtering: such candidates exist but fail more often because their tool arguments use the first date. Keeping those wrong calls is not the repair. Keep the scenario blueprint, produce a correct call from the latest supported date, and verify it again. This is an application proposal rather than a published sales result.

### 3. Why generation scale helps only within support

For an event generated independently with probability q, its probability of appearing at least once in M samples is 1−(1−q)^M. If q=0.001 and M=1,000, this is about 63.2%; the expected count is one. At M=100,000, the expected count is 100 and absence is extremely unlikely. For q=0.00001, M=1,000 gives only about a 1.0% chance of seeing it; even M=100,000 gives about 63.2%, with expected count one.

These sampling examples are calculated for this lesson, not LLM benchmark outcomes. They assume a fixed q and independent samples. Related prompts, deterministic decoding, repeated templates, and a shared generator can violate independence. If q=0, increasing M cannot produce the event. If the generator can produce an event but not the correct answer, more candidates do not by themselves solve correctness.

Explicit scenario conditions and source documents can change Q instead of merely increasing M. Asking for a supported pricing exception can expose a fact supplied in the product document. Asking for a coherent date-correction sequence can expose an interaction omitted by generic dialogue generation. The generated case still needs validation; naming a scenario does not guarantee its realization.

### 4. A correct filter can still shrink a tail

Imagine candidates are 5% correction-heavy conversations. The filter retains 90% of common candidates but only 30% of correction-heavy ones. The retained rare share is (0.05×0.30)/(0.05×0.30+0.95×0.90)=1.72%. Thus the rare share falls from 5% to 1.72%, even if every retained answer is correct. This Bayes calculation concerns retention, not trained-model accuracy.

The rate difference has at least two explanations: rare candidates really contain more errors, or the verifier wrongly rejects valid unusual answers. Audit both. Repairing faulty targets addresses the first; correcting the verifier or using a better rubric addresses the second. Relaxing the threshold indiscriminately can retain errors. Oversampling rare candidates without fixing labels can amplify them.

### 5. Knowledge tails and conversation tails require different evidence

[[long-tail-knowledge]] studies factual QA versus relevant-document frequency. It motivates source grounding when the missing element is a fact. A model's failure to use the latest appointment date is different: all necessary facts may already be in the conversation. In that case, more product knowledge can leave the state-update error unchanged. The next chapter separates those behavioral conditions.

## Recipe

Use a three-stage audit: measure Q against a reference or explicitly designed coverage plan; measure acceptance and rejection reasons by slice; then train and evaluate on held-out P-like and stress-test cases. Store the counting unit and evidence behind each rarity claim. A taxonomy helps discover empty cells, but an empty cell is not automatically an important or even coherent scenario.

## Generalization lens

Recursive collapse in chapter 23 concerns repeated generations of training data. A single-round narrowing of Q or R can happen without recursion. A low rare-slice score may also reflect limited capacity, wrong labels, or noisy measurement. Keep those explanations separate before calling an outcome collapse.

## Questions

1. If a filter keeps only correct answers, why can rare coverage still decline?
2. What changes when you specify a rare scenario rather than asking for more generic conversations?
3. Does a failure-selected test establish how frequently customers encounter the failure?

Continue: [Methods](methods.md) → [Practice](practice.md) → [23b](../ch-23b/read.md). [Catalog](../../../raw-data/llm-training/long-tail-papers.md).


## Source cards

- [[long-tail-knowledge]] — [card](../../../raw-data/llm-training/papers/long-tail-knowledge.md)
- [[graphgen]] — [card](../../../raw-data/llm-training/papers/graphgen.md)
- [[knowledge-density-synthesis]] — [card](../../../raw-data/llm-training/papers/knowledge-density-synthesis.md)
- [[data-value-scaling]] — [card](../../../raw-data/llm-training/papers/data-value-scaling.md)
- [[long-tail-knowledge-survey]] — [card](../../../raw-data/llm-training/papers/long-tail-knowledge-survey.md)
