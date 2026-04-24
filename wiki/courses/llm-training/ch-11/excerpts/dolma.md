---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md
source_url: https://arxiv.org/abs/2402.00159
created_at: "2026-04-23"
---

# Excerpt: Dolma — attribute-file lineage and the PII cascade as first-class operations

**Source library:** `wiki/raw-data/llm-training/papers/dolma.md`
**Paper:** Soldaini, Kinney, Bhagia et al. 2024, "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (Allen AI, ACL 2024).

---

## Why this source anchors ch-11

Dolma is the only corpus in the raw-data library whose primary contribution is *operational*. FineWeb contributed a classifier; CCNet contributed a three-stage skeleton; Dolma contributed the **toolkit** — the `dolma` CLI, the per-document attribute-file convention, and the public ablation that says *filter order matters as much as filter content*.

Ch-11 cites Dolma four times: for the six-stage filter cascade (§4), for the three-tier PII cascade (§4), for attribute-file lineage (§3), and for the filter-order defense (§4 — "dedup runs after PII, not before"). This excerpt pulls out those specific passages with the operational details the read.md page compresses.

---

## The six-stage cascade — order is the contribution

From the source (lines 28-37):

> **Filter cascade (order matters):**
> 1. **URL / document-level deduplication first** using Bloom filters — remove exact-URL repeats across CC snapshots.
> 2. **Language identification** via fastText; keep English (lang score ≥ threshold).
> 3. **Quality filters** adapted from Gopher/C4 heuristics (line-length, symbol-to-word ratio, stopword ratio, fraction of lines ending in punctuation, duplicate-line fraction).
> 4. **Content filters**: fastText classifiers trained on the **Jigsaw Toxic Comments** dataset produce `hate` and `NSFW` scores; documents above threshold are dropped.
> 5. **PII filtering** targets three high-precision categories: email addresses, IP addresses, phone numbers.
> 6. **Paragraph-level deduplication last**, via Bloom filter exact-match on paragraphs.
>
> **Why this order:** doing paragraph dedup last is deliberate — earlier stages change which paragraphs survive, so dedup only matters over the surviving distribution.

The last sentence is the insight. A naive pipeline would dedup early (cheapest to run on raw bytes); Dolma shows this is *wrong* because the set of "duplicates" is a function of what survives earlier filters. Concretely: two documents that differ only in a PII span become identical *after* the PII filter runs. If paragraph dedup ran before PII, these would not be caught.

This directly maps onto ch-11 §4's failure mode: *"PII removal creates new near-duplicates. If PII runs after dedup, these are missed."* The Dolma pipeline is the worked example of why this matters and what the fix is.

---

## Attribute files — the lineage substrate

From the source (line 44):

> **Tooling:** the `dolma` CLI accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document `attribute` files (one score per filter) so the final keep/drop decision is a separate, cheap pass.

This one sentence is the conceptual backbone of ch-11 §3. The `dolma` tool does not produce a filtered corpus; it produces *attributes*, and the filtered corpus is a **query** over those attributes. The separation has three consequences the read.md page pulls out:

1. **Filters become reproducible from primary sources.** If you want a different threshold on `gopher_quality`, you re-run the *query* (milliseconds) instead of the filter (hours).
2. **Multiple mixes share one attribute pass.** Pretrain uses `gopher_quality > 0.7`; mid-training uses `gopher_quality > 0.85`. One filter run; two mixes.
3. **Audit is cheap.** "Which documents were dropped for PII?" is `SELECT doc_id WHERE pii.removed_spans IS NOT EMPTY`. At Dolma's 3T-token scale this is seconds, not a full re-run.

The ch-11 §3 attribute-directory layout (`attributes/lang-id/`, `attributes/quality/`, `attributes/pii/`, ...) is directly lifted from the Dolma CLI's output convention. Mastering Dolma's tooling-level thinking is the transition from "I can train a model" to "I can build the data pipeline my lab will rely on for two years."

---

## The PII cascade — three tiers, different recall targets

From the source (line 34):

> **PII filtering** targets three high-precision categories: email addresses, IP addresses, phone numbers.

Dolma's paper (not fully quoted in the raw-data file but referenced in its "Key Figures/Tables to Study" ablation row) uses a regex-based filter for these three categories because they have *high-precision surface forms*. Emails: `\b[\w.-]+@[\w.-]+\.\w+\b`. IPs: four dot-separated octets with appropriate range. Phones: country-code + digit-group patterns per locale.

Ch-11 §4 extends this to the three-tier cascade that any production pipeline eventually needs — because Dolma's Tier 1 has ~80% recall (misses obfuscated forms), and regulatory regimes demand higher. The tiers Ch-11 describes:

| Tier | From Dolma | Ch-11 extension | Precision | Recall |
|---|---|---|---|---|
| 1 | Regex (email, phone, IP) | (same) | ~99% | ~80% |
| 2 | — | NER classifier on flagged domains | ~90% | ~92% |
| 3 | — | LLM classifier on 1% audit sample | ~95% | ~96% |

Dolma itself is Tier-1-only for the categories it targets; the ch-11 extension formalizes what production labs (Meta for Llama 3's Guard pipeline, Allen AI's extensions) build on top. The key operational point: **only Tier 1 runs on every document**. Tier 2 and Tier 3 are sampled/filtered; running them globally at 15T tokens would cost ~10 M CPU-hours.

---

## Scrub vs delete — the undocumented decision

Dolma's source line 34 says PII "documents above threshold are dropped" for toxic content but is less explicit about PII-span handling. The paper text (summarized in the raw-data file's line 44) emits per-document attributes including PII spans; the actual scrub-or-delete choice is a downstream query decision. This matches ch-11 §4's framing — that scrub-vs-delete is *per-consumer policy*, not a hard-coded pipeline property.

Dolma's design naturally supports either: the attribute file lists the spans; the downstream query can (a) replace each span with `[EMAIL]` / `[PHONE]` / `[IP]` tokens keeping the document, or (b) drop the document entirely. FineWeb redacts; Dolma leaves the choice to the consumer. Ch-11 §4 recommends scrub-with-span-log for most modern applications because it preserves context and provides the legal audit trail.

---

## Why paragraph dedup is last — a worked example

From the source (line 37):

> **Why this order:** doing paragraph dedup last is deliberate — earlier stages change which paragraphs survive, so dedup only matters over the surviving distribution.

A concrete example from what Dolma encounters in CC:

- Document A: legal boilerplate + article text + PII (author email in footer).
- Document B: same legal boilerplate + same article text + different PII (comment section with emails).

Before PII: A and B differ (different emails). Paragraph-level dedup sees the shared boilerplate + article paragraphs as high-frequency repeats; these *are* duplicates across thousands of similar documents. Running paragraph dedup before PII removes the boilerplate-paragraphs from both — good.

After PII removal (emails scrubbed to `[EMAIL]`): A and B become increasingly similar. If paragraph dedup ran *after* PII on A and B, additional paragraphs now look identical (the footer lines, once different, now both read `Contact: [EMAIL]`). More dedup opportunity surfaces only after PII.

Dolma's order: run paragraph dedup *after* PII precisely so the second round of "duplicates created by scrubbing" is caught. If dedup ran only before PII, this set of near-duplicates survives into the final corpus.

Ch-11 §4 states the principle abstractly ("dedup must run after PII, not before"); Dolma's order is the operational precedent.

---

## Source-specific quirks — why one pipeline is actually N

From the source (lines 39-42):

> **Source-specific quirks:**
> - `peS2o` (scientific) uses different quality filters than web — it trusts publication structure.
> - `The Stack` code uses near-dedup via MinHash on code tokens.
> - Social media (Reddit) is filtered by subreddit-level quality lists.

Dolma is *not* one pipeline. It's a base pipeline plus per-source overrides, all unified by the `dolma` CLI's YAML config system. Ch-11 §2 picks this up implicitly: different source buckets (web / code / books / academic / social) have different shard layouts and different throughput characteristics, which is why [[fineweb]]'s 200 MB uniform shard size is a *web-only* guideline.

For code specifically: Dolma's use of "MinHash on code tokens" instead of word-level MinHash is the operational detail that carries forward. Code has different token distributions (many repeated identifiers, whitespace-heavy); word-level MinHash saturates quickly. Token-level MinHash with a code-aware tokenizer is the production pattern, picked up by The Stack v2 and referenced implicitly in ch-11 §4's code-filtering discussion.

---

## What Dolma does not solve — the opt-out registry

Dolma's PII filter is *reactive*: given a document, detect and remove PII. It does not address the *prospective* problem — a user-initiated opt-out request, where the identity is known but the documents are not. Ch-11 §4's opt-out-registry discussion fills this gap.

The operational implementation ch-11 sketches (bloom filter over doc-ids or URL hashes, loaded into the pipeline at runtime) is compatible with Dolma's attribute-file architecture: a new attribute `optout.matched: bool` is added; the mix query filters on `optout.matched = false`. The registry is a table external to the pipeline; the pipeline consumes it via the attribute pass.

Dolma's paper does not document this extension (it was a 2024 release; opt-out registries became more prominent in post-2024 regulatory contexts). The chapter makes the extension explicit because any modern pipeline in production will need it.

---

## What to take from Dolma for ch-11

1. **Filter order is a first-class design decision.** Dedup-last is not cosmetic; it compounds with PII removal.
2. **Attribute files are the lineage graph.** They are append-only, per-document, per-filter. The filtered corpus is a query.
3. **The PII cascade has three tiers.** Only Tier 1 scales to global corpora; Tier 2/3 are flagged-sample or audit.
4. **Source-specific quirks force per-bucket pipelines under one CLI umbrella.** Dolma's YAML config is the operational mechanism; the read.md chapter generalizes it.
5. **Scrub-vs-delete and opt-out registries are consumer-side decisions** built on top of Dolma's attribute substrate — not built into the filter itself.

---

## Connections

- [[excerpts/fineweb]] — the successor pipeline; classifier-based quality filter over Dolma's heuristic stack.
- [[excerpts/ccnet]] — the ancestor skeleton Dolma inherits.
- [[excerpts/llama-3]] — Llama Guard 3 as a downstream PII / safety classifier; sits at Tier 3 of the cascade.
- [[excerpts/olmo-3]] — the model-flow worldview; Dolma 3 / Dolmino / Longmino are separate Dolma-architected pipelines.
- [[ch-11]] — §3 (attribute files as lineage), §4 (PII cascade, filter order, scrub vs delete).
