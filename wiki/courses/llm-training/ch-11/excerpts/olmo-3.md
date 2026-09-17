---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten from the arXiv:2512.13961 PDF text read on 2026-09-15; the earlier version attributed MosaicML MDS shards, lineage assertions, and an 8x throughput gain from dataloader state to OLMo 3
---

# Excerpt: Olmo 3 — tokenizer, special tokens, PII filtering, decontamination, tracing

Team Olmo, arXiv 2025-12. Library card: [[olmo-3]]. Section numbers refer to the PDF read on 2026-09-15.

## Tokenizer (§3.2)
"We process data for each stage using the same tokenizer as OLMo 2, which is derived from OpenAI's cl100k." The OLMo-core configuration gives vocab_size 100,278, padded to 100,352 ([[olmo-core-olmo3-configs]]).

## Special tokens in mid-training data (§3.5.4, "Leave special tokens for SFT stage")
Microanneals on Tulu3-SFT data with and without chat special tokens such as `<|im_start|>` and `<|im_end|>`: "when training on data containing chat templates and special tokens, models consistently output these special tokens at inference time, resulting in evaluation scores that are [strongly] reduced (e.g. GSM8K drops from 49.43 to 0, and CruxEval drops from 32.89 to 18.91)." A chat template with ordinary text in place of the special tokens gave 46.02 on GSM8K and 29.65 on CruxEval. The authors attribute the drop "specifically to the introduction of special tokens to the embedding vocabulary when they have not been seen in pretraining", note that the score loss is "primarily" answer-parsing disruption, and removed both the template and the special tokens from mid-training instruct data in favor of newline-based formatting.

## Tool-call tokens in post-training (§5.2.1)
For Olmo 3 Instruct function calling, "We also extend the tokenizer's vocabulary with dedicated special tokens corresponding to these tags." Preliminary results suggested this was more effective than encoding the tags as regular text; no numbers are given.

## PII (§3.4.2, science PDFs only)
Document-type-aware filtering: Gemma 3 12B classifies the first page for sensitive standalone PII or sensitive information linked to an individual; Gemma 3 4B assigns document-type flags from the first 5,000 characters; rules decide which document types with PII remain. "Ultimately this removes 4.9% of the remaining pool and yields a pool of 148 million documents." The rule: "is this document type intended for public dissemination?" The web pool description (§3.4.1: URL blocklist, heuristics, language ID, exact, fuzzy, and substring deduplication, topic and quality classifiers) does not describe a PII step.

## Decontamination (§3.5.3-3.5.4)
- The `decon` tool samples n-grams at a stride, then expands matches on both sides and removes a document when the contaminated-n-gram count exceeds a threshold; applied to mid-training and long-context data "in light of results suggesting that memorization occurs most strongly near the end of training".
- Removing all splits of DROP removed over 60,000 training examples from sources such as Flan. Some benchmarks (DROP, Minerva, SQuAD) scored lower after decontamination; GSM8K showed complete leakage yet scored higher with decontaminated data.

## Deduplication tokenizer (App. A.2.2)
Web MinHash deduplication "tokenize[s] documents using the p50k tokenizer and construct[s] sets of 5-gram token sequences", with 26 bands of size 11 targeting Jaccard similarity 0.80, on 32 shards of the 12.7B documents left after exact deduplication. The deduplication tokenizer (p50k) differs from the training tokenizer (dolma2).

## Tracing (§1)
"The Olmo 3 release also enables reasoning chains to be traced back to their original training data." See [[olmotrace]] for the tracing system.

## How ch-11 uses it
§1 (tokenizer lineage), §5 (special tokens added after pre-training), §6 (PII and decontamination as pipeline stages), Recipe rows.
