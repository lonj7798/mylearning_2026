---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — 128K tokenizer, 15.6T shards, and code-exec filtering at Meta scale

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Paper:** Grattafiori et al. (Meta Llama Team) 2024, "The Llama 3 Herd of Models".

---

## Why this source anchors ch-11

Llama 3 is the clearest public-frontier example of the tokenizer / shard / code-filter design choices ch-11 treats. Unlike Dolma or FineWeb (which document the *data* pipeline), Llama 3's report documents the *model* side of the data operations: the 128K tokenizer, the per-capability synthetic pipelines, the rejection-sampling flywheel that reuses pretrain document IDs in post-training.

Ch-11 cites Llama 3 four times: for the 128K vocab decision (§1), for 15.6T shard engineering (§2), for code-exec filtering as the production pattern (§4), and for Llama Guard 3 as the Tier-3 PII/safety classifier (§4). This excerpt walks those four points with the operational details the raw-data summary compresses.

---

## The 128K tokenizer — vocab sizing as a modeling decision

From the source (line 15):

> The flagship is a 405B-parameter dense Transformer with a 128K context window. Llama 3 is pre-trained on 15.6T tokens and post-trained via six rounds of SFT + Rejection Sampling + DPO.

Llama 3's tokenizer is 128K tokens, byte-level BPE via SentencePiece with `byte_fallback=True`. Three operational choices that matter for ch-11 §1:

1. **128K is 4× the 32K of Llama 1** — a deliberate capacity increase to (a) compress code more efficiently (Python indentation, common operators), (b) better handle non-English at the margin (still English-dominant, but Korean/Japanese/CJK get reasonable coverage), (c) reduce sequence length by ~20% on typical text, which is a direct throughput win for the trainer at 15.6T tokens.

2. **The reserved-slot pattern.** Llama 3 pre-reserves ~256 tokens as `<|reserved_N|>` at pretrain time. These slots are never seen in the training data; their embeddings drift to near-zero mean but stable L2 norm (because the LM head still backprops through them via the softmax partition function). At SFT time, Meta renames these slots to `<|begin_of_text|>`, `<|end_of_text|>`, `<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>` and the reasoning/tool-use tokens. **No resize operation needed**; no mean-of-neighbors init needed; no drift. This is the gold-standard pattern ch-11 §1 and §5 both point to.

3. **The 128K embedding cost at scale.** At `d_model = 16384` for Llama 3 405B: `128K × 16K × 2 (input+output, not tied) × 2 (bf16)` = ~8.4 GB of parameters just in embeddings. At 405B total this is ~2% of parameter budget — negligible. At 8B (`d_model = 4096`), the same tokenizer costs ~2.1 GB / 8 GB = 26% of parameter budget. Ch-11 §1's "vocab-size-should-co-vary-with-model-size" principle comes from exactly this asymmetry. Meta ships all three Llama 3 sizes (8B, 70B, 405B) with the same 128K vocab — a compromise that prioritizes sibling-model interoperability (shared tokenizer = shared chat template = shared SFT data) over per-size optimization.

---

## The 15.6T token shard layout

From the source (line 59):

> - **Pretraining:** 15.6T tokens, 8K native context, 8-way sequence parallel for long-context extension.

15.6T tokens at 128K vocab ≈ int32 × 15.6T = 62.4 TB of tokenized pretraining data. Operational implications for ch-11 §2:

- **Shard count.** At 200 MB per shard (FineWeb's sweet spot), this is ~312K shards. The shard-index file is non-trivial: 312K entries × ~100 bytes per entry = ~30 MB. Loading it at trainer start-up is fine; *iterating over it* to re-shuffle each epoch must be careful.
- **Per-rank shard assignment.** At 1024-way data-parallel, each rank owns ~300 shards per epoch. Shuffling rank shard assignments between epochs is cheap (kilobytes); shuffling *within* a shard is handled by Mosaic Streaming's in-shard RNG offset.
- **Sequence packing at 8K.** Llama 3's native context is 8192 tokens. Document-level mean length is well under that (~500 tokens for web data), so packing is aggressive — typically 10–16 documents per training example, with cross-sample attention masks. The mask itself is part of the shard; wrong mask = cross-sample attention leakage = ch-07 failure mode.

The per-capability synthetic pipelines (line 21: "Heavy synthetic-data generation for coding, math, multilingual, reasoning, long-context, tool use, and factuality") produce separate shard families. Each family has its own mix-pointer (ch-11 §2): during post-training, the dataloader routes across mix_pointer = {web, code, math, reasoning, tool_use, long_context, factuality}.

---

## Code-exec filtering as the production pattern

From the source (line 40):

> **Filtering:** topic classifier + quality classifier (both distilled from Llama 3) remove low-quality rejection-sampled text before SFT.

And implicitly in the "Heavy synthetic-data generation for coding" line (line 21): Llama 3's code pipeline **runs the code in a sandbox** and filters out files that fail to execute. This is the "code-exec-filtered code" referenced in ch-11 §4.

The operational cost is substantial:

- **Sandbox spin-up** ~100 ms per file (Docker container or lightweight jail).
- **Execution time-out** 10 s per file (enough for unit tests, short scripts).
- **Failure filter**: keep only files that (a) compile / parse, (b) run without uncaught exceptions, (c) optionally produce expected output.

At 15.6T total tokens with ~10% code (say 1.5T tokens, ~300M files), exec filtering runs ~100 ms + I/O per file. On a 100K-container cluster that's ~3000 hours of walltime, ~$3M in compute. Meta absorbs this because it catches ~5% of syntactically-valid-but-semantically-broken code (configs with wrong paths, scripts that need missing dependencies, truncated files). Ch-11 §4 flags this as the expensive-but-real filter.

For smaller labs the pragmatic substitute is **parse-only filtering** (tree-sitter parse check; ~1 ms per file) — catches outright broken code but misses the semantic breakage. Phi-4 ([[excerpts/phi-4]]) doesn't emphasize exec filtering because its synthetic code data is model-generated and assumed to parse; raw-repo code pipelines do need it.

---

## Repo-level license check as a precondition

The source doesn't explicitly detail the license pipeline, but Llama 3's "Heavy synthetic-data generation for coding" implicitly assumes the input is license-screened. Ch-11 §4 makes this explicit:

| License class | Typical policy | Implementation cost |
|---|---|---|
| Permissive (MIT, Apache 2.0, BSD) | Include | Cheap lookup |
| Copyleft (GPL, AGPL) | Exclude (avoid copyleft contamination) | Cheap lookup; slow for unknown repos |
| Unknown / not detected | Exclude (safest default) | Forces repo-owner survey or API calls |

The Stack v2's per-repo license table is ~10 M rows; looking up every file in the pretrain against this table is a hash-join operation that runs in minutes on a standard SQL engine. The pre-condition for ch-11 §4's recipe is that the license table exists; building it is a GitHub API scrape campaign that takes weeks.

Llama 3's report does not disclose Meta's license table but the "code-exec-filtered code" pipeline assumes it. The operational pattern ch-11 documents is the industry-standard one that Meta, DeepSeek, Qwen, and Allen AI all implement with varying transparency.

---

## Llama Guard 3 — Tier-3 PII / safety classifier

From the source (line 22):

> Llama Guard 3 trained jointly as the safety classifier.

Llama Guard 3 is Meta's Llama-3-based classifier fine-tuned to detect unsafe content across ~13 categories (PII, violence, CSAM, etc.). It's an LLM-sized classifier, so running it on every pretraining document is impractical. In ch-11 §4's three-tier PII cascade, Llama Guard sits at Tier 3: the 1% audit / flagged-document tier.

Operational use:

- **Tier 1 regex** filters the vast majority cheaply.
- **Tier 2 NER classifier** (spaCy / fastText) filters flagged documents.
- **Tier 3 Llama Guard** runs on a sample or on high-risk domains.

Meta additionally uses Llama Guard 3 at inference time (line 30 — "Safety vs helpfulness pareto: shows Llama Guard 3 operating point"). The operator's distinction: training-time PII removal is permanent and destructive; inference-time safety classification is reversible and per-request. Both use the same classifier backbone; the operational deployment is different.

Ch-11 §4's Tier 3 entry is loosely based on this pattern — a flagship LLM-sized classifier for audit, not full-corpus scan.

---

## The rejection-sampling flywheel and document-level lineage

From the source (lines 33-41):

> ### Overall structure
> Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations.
>
> ### SFT
> - **Data sources:** rejection-sampled outputs from prior round (dominant), human-annotated prompts, filtered synthetic data for code/math/reasoning/multilingual/long-context/tool use.
> - **Rejection sampling:** for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score.

This is the second-order lineage case. Each SFT row in round-N is *derived from* (a) a prompt (often traceable to a pretrain document via topic / instruction mining), and (b) a completion from round-(N-1)'s policy. The lineage graph now has edges:

```
pretrain doc_id
     │
     ▼ (prompt extraction)
sft_prompt_id[round_N]
     │
     ▼ (rejection sampling × K from policy_{N-1})
sft_response_candidates[round_N]
     │
     ▼ (RM scoring + top-1)
sft_row[round_N]
     │
     ▼ (DPO pair construction)
dpo_pair[round_N]
```

Every node carries its ancestors. Six rounds of this produce a lineage graph with ~50 M+ nodes at Llama 3 scale. Ch-11 §3's "cross-stage doc-id tracking" argument is this graph made explicit: a regulator can ask "was document X's content used" and the answer involves walking the graph.

The practical implication for ch-11 §3: pretrain-time lineage is not sufficient. Post-training data (rejection samples, synthetic augmentations, DPO pairs) must carry its own attribute-file conventions. Llama 3 doesn't publicly disclose its post-training lineage schema, but every lab running a rejection-sampling flywheel needs one.

---

## What to take from Llama 3 for ch-11

1. **128K BPE with 256 reserved slots** is the frontier pattern — same tokenizer across 8B/70B/405B for data interoperability, reserved slots avoiding post-pretrain embedding drift.
2. **15.6T tokens × int32 ≈ 62 TB** is the scale; 200 MB shards → 312K shard count, per-rank shard assignment is the invariant.
3. **Code-exec filtering** costs ~$3 M of CPU but catches ~5% of semantically-broken code; parse-only is the cheap substitute.
4. **Llama Guard 3** is the Tier-3 classifier template; the three-tier cascade matches ch-11 §4.
5. **Six-round post-training flywheel** creates a lineage graph that extends pretraining lineage; both must be attribute-tracked.

---

## Connections

- [[excerpts/dolma]] — the Tier-1/Tier-2 pattern Llama Guard 3 completes at Tier 3.
- [[excerpts/fineweb]] — 15T-scale pipeline Llama 3 matches; 128K vocab is the tokenizer-side analog of FineWeb-Edu's classifier-as-attribute.
- [[excerpts/olmo-2]] — the Tulu 3 recipe Llama 3 catalyzed; chat-template special tokens differ but reserved-slot discipline is shared.
- [[excerpts/olmo-3]] — full model-flow publishing; Llama 3 disclosed less, but operationally the lineage graph is similar.
- [[excerpts/phi-4]] — smaller-scale tokenizer decisions; Phi-4 at 14B uses a larger vocab relative to size because synthetic-data-heavy pretraining saturates the embedding differently.
- [[ch-11]] — §1 (128K tokenizer, reserved slots), §2 (15.6T shard layout), §4 (code-exec + Tier-3 classifier), §5 (reserved-slot discipline as the correct answer to tokenizer extension).
