# 23a — Coverage Records, Mixtures, and Fair Comparisons

<!-- scope: proposed implementation records, controlled comparisons, metrics, and failure repair for rarity units, sampling support, retention bias, and source-grounded knowledge coverage
     deps: [[read]]
     see-also: [[read]], [[methods]], [[long-tail-papers]]
     sources: [[long-tail-knowledge]], [[graphgen]], [[knowledge-density-synthesis]], [[data-value-scaling]], [[long-tail-knowledge-survey]]
-->

## 1. A usable coverage record

The proposed record should include a family ID, task, current policy version, evidence source, expected action, candidate-generation conditions, validity result, rejection reason, and rarity provenance. Rarity provenance can be a measured deployment rate with a counting unit, an expert report, or a deliberately constructed stress case. Preserve that distinction. A stress test has value without claiming to represent production frequency.

For a twice-corrected appointment, store the full correction sequence, the latest supported date, whether consent to book exists, and the source of each field. A missing field is a separate state, not an invitation to fill the most frequent value. Track candidate counts and valid retained counts. If 200 candidates yield 20 valid targets, the solution may be to improve the generator or evidence, not simply raise the slice quota.

## 2. Data shares have different denominators

Suppose 100 common dialogues average 200 training tokens and 10 rare dialogues average 2,000. Rare dialogue share is 10/110=9.1%, but rare token share is 20,000/40,000=50%. A mixture expressed only as conversation counts hides half of the learning exposure. If response-only loss is used, additionally report supervised assistant tokens; input tokens affect context and compute without necessarily receiving the same loss.

This is a teaching example calculated for this lesson, not a recommended mixture. Record raw conversations, tokens seen after sampling/repetition, and supervised tokens separately. Epoch count alone is not comparable when datasets differ in length. A larger accumulated pool trained for one epoch can consume much more compute than a small replacement pool.

## 3. Controlled comparison

Start with a base checkpoint and a common reference set. Compare generic generated data against targeted generated data at a fixed training-token budget. A second comparison can test targeted data with versus without a retained real-data anchor. Keep tokenizer, context length, masking, optimizer, schedule, inference settings, and evaluation items fixed. Report generation and filtering cost separately; equal training cost does not imply equal pipeline cost.

Use both deployment-like tests and explicitly labeled rare/stress slices. Split whole families before generating close variants. Reserve a validation set for mixture choices and a final test for reporting; selection reference examples are part of tuning, not an untouched test. If you change the generator, policy, and mixture together, the comparison measures the combined pipeline and cannot isolate the cause.

## 4. Read the outcomes carefully

Higher retained-target correctness with flat model performance may mean the original labels were not the binding problem. Improved rare performance with worse common performance is a trade-off requiring an application decision. Better scores only on generator-like wording can be a shortcut. If all arms plateau, capacity, compute, redundancy, or evaluation ceiling remain possible explanations; the plateau alone does not establish model collapse.

The next practical action is to inspect slice counts and errors, not automatically add more synthetic data. Preserve valid rare families, repair invalid targets with evidence, and evaluate a revised pipeline under the same budget. Unknown future skills remain an open coverage problem; temporary multilabel taxonomies should evolve as real cases appear.

[Back to lesson](read.md) · [Methods](methods.md) · [Next chapter](../ch-23b/read.md) · [한국어](practice_kor.md).

