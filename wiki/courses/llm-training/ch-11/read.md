<!-- chapter: ch-11
     track: data
     title: Data Operations — Tokenizers, Shards, Lineage, PII
     sources: [[dolma]], [[fineweb]], [[ccnet]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[qwen-3]], [[deepseek-v3]], [[phi-4]], [[physics-of-lm-3]]
     external: https://arxiv.org/abs/1508.07909 (BPE), https://arxiv.org/abs/1808.06226 (SentencePiece)
     figures: figures/pipeline-ops.html
-->

# Chapter 11 — Data Operations: Tokenizers, Shards, Lineage, PII

> **Core insight.** Curation decides *what* text enters the model; **operations** decide *whether you can ever build the same model again*. A 15T-token corpus is not a dataset — it is a `(content-hash → doc-id → filter-attributes → shard-offset → tokenizer-id)` graph that must survive six months of pipeline drift, two tokenizer migrations, three PII-removal incidents, and a hardware change. Lose one edge of the graph and the run is unreproducible; lose the tokenizer → shard edge and you cannot even *resume* training.
>
> **Guideline.** Treat every byte as content-addressed (BLAKE3 of the raw document) from the first crawl stage. Store filter verdicts as *per-document attribute vectors*, not as filtered shards — so you can re-derive any subset without re-running the upstream filters. Size your vocabulary to the worst-case language you actually train on (120K+ if multilingual, 128K if code-heavy, 32K only if English-only base). Never extend a tokenizer post-pretraining without initializing new embeddings from the mean of their sub-word decomposition; otherwise the new `<think>` and `<tool_call>` tokens drift the chat template into non-reproducibility within ~1000 SFT steps.

---

## Why this chapter exists

Chapters 9–10 argued that the *shape* of the training corpus — its filters, its dedup, its quality cascade — is load-bearing for final model quality. That was the curator's view. This chapter takes the operator's view: once the filters exist, how do you **run** them on 15T tokens ([[fineweb]]) or 36T tokens ([[qwen-3]]) without losing the lineage of any single document, without leaking a customer's phone number into a checkpoint, and without silently drifting the tokenizer when Stage 2 of pretraining adds a new language?

The cost of getting operations wrong is not a worse model — it is a model that is *suspicious*. Dolma's contribution ([[dolma]]) was not primarily a 3T-token corpus; it was the *toolkit* that emits a `.attributes.jsonl` file per document so any subset of the final corpus can be re-derived from the raw shards by replaying the attribute pass. FineWeb's contribution ([[fineweb]]) was not its 15T tokens; it was the `datatrove` codebase that re-runs the pipeline deterministically, per-snapshot, with a classifier-score threshold that's a *parameter*, not a hard-coded filter. OLMo 3's contribution ([[olmo-3]]) was not Dolma 3 — it was the *model-flow worldview* where Dolma 3 / Dolma 3 Mix / Dolmino / Longmino / Dolci are each separately-tracked lineages with their own filter gates, their own tokenizer-consumer assertions, and their own audit trails.

This chapter is organized around the four operational artifacts that survive beyond the training run: the **tokenizer**, the **shard layout**, the **lineage graph**, and the **PII / secrets audit log**. Each has its own failure mode; each has been the root cause of at least one well-documented training incident.

---

## 1. Tokenizer construction at scale — BPE, WordPiece, SentencePiece

The tokenizer is the only artifact that sits on *both* sides of the trainer: it produces the training shards, and it is shipped with the final weights. Getting it wrong post-hoc is impossible without a full retrain.

**The three algorithm families, briefly.**

- **Byte-Pair Encoding (BPE)** — Sennrich, Haddow, Birch 2015 ([arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)). Greedy bottom-up merges. Start from bytes (or chars); repeatedly merge the most frequent adjacent pair until the vocabulary reaches the target size. The trained tokenizer is a *merge table* (an ordered list of pairs) plus a base vocab. Decoding is a stack-walk over the merge table. BPE is what GPT-2 / GPT-3 / GPT-4 / Llama use.
- **WordPiece** — Schuster & Nakajima 2012, used by BERT. Same bottom-up merge idea, but the merge criterion is likelihood-gain under a unigram LM, not raw frequency. Produces slightly different segmentations in practice; the algorithmic difference rarely matters for LLM pretraining, and most labs have converged on BPE or its byte-level variant.
- **SentencePiece** — Kudo & Richardson 2018 ([arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)). Two critical moves: (a) **operate on raw bytes**, treating whitespace as a special character (`▁`), which means the tokenizer is fully reversible — a round-trip `decode(encode(text))` is bit-exact; and (b) **unigram LM** as an alternative training algorithm where the vocab is learned by EM over a large candidate set. Almost every modern tokenizer in production is a SentencePiece or SentencePiece-flavored byte-level BPE: Llama 3 ([[llama-3]]) uses a 128K BPE built with SentencePiece-style byte fallback; OLMo 2 / 3 ([[olmo-2]], [[olmo-3]]) the same; Qwen 3 ([[qwen-3]]) extends the byte-level scheme to 151K for 119-language coverage.

**Vocab-size tradeoffs — 32K vs 128K vs 200K.** The choice is a three-way tension between (a) sequence length (bigger vocab → fewer tokens per document → cheaper training), (b) embedding-table cost (`vocab × d_model × 2` for input+output embeddings), and (c) downstream quality on under-represented scripts or programming languages.

| Vocab size | Typical choice | Sequence compression | Embedding cost at d=4096 | When it's right |
|---|---|---|---|---|
| 32K | GPT-2, early Llama 1, OLMo 1 | baseline | 0.26 B params | English-only base; compute-scarce; simple chat |
| ~100K | GPT-4, Claude, Gemini | ~20% fewer tokens than 32K on English, ~2× fewer on code | 0.82 B | code-heavy, multilingual, production frontier |
| 128K | Llama 3 ([[llama-3]]), OLMo 2/3, DeepSeek-V3 ([[deepseek-v3]]) | ~15% fewer than 100K; better code and non-Latin scripts | 1.05 B | frontier default in 2024+ |
| 151K | Qwen 3 ([[qwen-3]]) | best on CJK, Arabic, Korean | 1.24 B | 119-language coverage |
| 200K | experimental (none of the cited reports ship this) | diminishing returns beyond ~150K | 1.64 B | only if you genuinely have long-tail scripts |

The 32K → 128K jump is the non-trivial one. [[physics-of-lm-3]] reframes the cost: parameters devoted to the embedding table are parameters *not* devoted to factual-knowledge storage. Allen-Zhu & Li argue knowledge capacity scales linearly with parameter count. At 7B total parameters, a 128K vocab eats 1.05 B = ~15% of the budget on embeddings; at 70B it's ~1.5%. Thus small models should have smaller vocabularies (Phi-3 used 32K; [[phi-4]] uses a larger vocab but still well below frontier because the model is 14B), and the vocab-size decision *co-varies with model size* — a fact that is routinely violated when a 70B tokenizer is reused for a 1B distill.

**Code-vs-natural BPE merges.** The same training text produces very different merge tables depending on domain mix. On code, BPE aggressively merges `    ` (four-space indent) into a single token, learns `    def ` / `    return ` / `, self):` as high-frequency merges, and segments `camelCaseIdentifiers` at case boundaries. On natural text, the merges are prefixes (`un-`, `pre-`, `re-`), common suffixes (`-tion`, `-ing`, `-ed`), and whole-word high-frequency tokens.

The operational consequence: a tokenizer trained on a 50/50 code/natural mix under-tokenizes each. GPT-2's 50K tokenizer famously wastes vocab on long rare English words (`disestablishmentarianism` is a single token) at the expense of `    elif ` which it splits into `    ` + `el` + `if ` + space — a ~50% sequence-length inflation on Python. Llama 3 ([[llama-3]]) addresses this by training the 128K tokenizer on a mix that explicitly up-weights code: the result is that Python and Go see ~25% fewer tokens per line than GPT-3.5's tokenizer on the same inputs.

The recipe used by frontier labs, inferred from [[llama-3]] / [[olmo-2]] / [[deepseek-v3]]:

1. Sample a **tokenizer training corpus** of ~500 GB text. Domain ratios should match the *final* pretraining mix, not the raw-web distribution — otherwise the 80%+ web-dominated BPE drowns code and math.
2. Train byte-level BPE with SentencePiece, target vocab = 128K (code-heavy) or 151K (multilingual), `byte_fallback=True` so any byte is representable.
3. **Pre-reserve** a block of special tokens: `<|begin_of_text|>`, `<|end_of_text|>`, `<|im_start|>`, `<|im_end|>`, plus 200–500 unused-but-reserved ids (`<|reserved_0|>` … `<|reserved_500|>`) for future extensions. This is the "leave space for `<think>`" decision; §5 explains why omitting it costs you a full pretrain.
4. Run the full pretrain tokenizing on-the-fly *or* pre-tokenize and shard (§2 covers the tradeoff). Freeze the tokenizer; emit `tokenizer.json` with a BLAKE3 hash. Every shard in §2 is tagged with this hash.

---

## 2. Shards and streaming — Mosaic, WebDataset, tar vs parquet

A modern pretrain reads ~15 TB of tokenized data during a full epoch. At 1024 H100s each consuming ~40 MB/s of tokens, the aggregate read bandwidth is ~40 GB/s. The file-system layout has to match that.

**The three shard formats in production.**

- **WebDataset (`.tar` shards)** — originally from NVIDIA DALI. Each shard is a plain tar file containing thousands of `.json` or `.txt` documents, addressed by a synthetic name (`{split}-{shard:06d}.tar`). Reads are streaming-sequential: you open a tar and pull documents in order. Random access is `O(shard-size)` — seek to a document mid-shard is expensive. Good for pretraining (sequential sweep is natural), bad for ablation runs that want per-document lookup.
- **MosaicML Streaming (`.mds` shards)** — MDS = "Mosaic Dataset Shards." Each shard is a binary file with an explicit in-file index so random access is `O(log n)` within a shard. Critically, the format supports **deterministic resumable streaming** — dataloader state = (shard, offset, rng) fits in a few bytes and round-trips cleanly. This is what makes [[olmo-3]] model-flow possible: every stage (pretrain → mid-train → long-context) resumes cleanly from a different mix.
- **Parquet** — columnar, compressed, with row-group indexes. Read throughput via PyArrow / Polars is excellent for wide schemas (text + 20 attribute columns). The `datatrove` pipeline that produced [[fineweb]] emits parquet natively — it's the right format for the *curation* pipeline because each row is `(doc_id, text, lang_score, classifier_score, minhash_signature, dedup_cluster_id, ...)`. Most frontier pretrains *do* then repack parquet into MDS or tar at the last stage for pure streaming throughput.

**Read-throughput vs seek-cost.** For a 1024-GPU pretrain that reads ~40 GB/s, the relevant metric is sustained sequential read per worker node, not random-access latency. A pathological shard layout — one shard containing 1% of the corpus — creates a straggler where one dataloader finishes 100× slower than its peers. The operational fix is **uniform shard size**: [[fineweb]] reports 200 MB per shard as the sweet spot for their 15T release (99% of shards within 1.5× the median). Llama 3 uses a similar ~500 MB shard size for its 15.6T tokens ([[llama-3]]). Too-small shards (10 MB) inflate open/close overhead; too-large (5 GB) make any read-sample a long-tail event.

**Tokenize-offline vs tokenize-on-the-fly.** The two regimes:

| Mode | When | Pros | Cons |
|---|---|---|---|
| Tokenize offline once, store as `int32[]` | Tokenizer is frozen; you plan to run multiple epochs or ablation sweeps | 4× faster dataloader (no re-tokenization); deterministic replay | Re-tokenization when you change the tokenizer is a full shard rewrite (~$10K at 15T tokens, 1 week of a 128-node cluster) |
| Tokenize on-the-fly from text shards | Early experimentation; tokenizer still in flux | Cheap tokenizer changes; text remains human-readable | Dataloader CPU-bound; 15–25% throughput penalty |

DeepSeek-V3 ([[deepseek-v3]]), because of its FP8 training and extreme GPU utilization, tokenizes offline — the trainer cannot afford the 20% CPU tax. OLMo 2 / 3 tokenize offline but release the raw-text shards too, so downstream researchers can re-tokenize with their own vocab.

**Per-rank dataloader state.** Under FSDP with a 1024-way data-parallel dimension, each rank holds a *portion* of the shard list. The ch-06 silent-failure mode returns: if the per-rank shard assignment is not preserved across a resume, you silently re-train on shards other ranks already consumed. Mosaic Streaming solves this by making the dataloader state a compact serializable `(rank_shard_list, within_shard_offset, rng_seed)` tuple; [[olmo-3]]'s 8× SFT throughput improvement ("Moving SFT from Open Instruct to Olmo Core reportedly improved throughput by 8x") rests on this.

---

## 3. Dataset lineage and versioning — content-addressed hashing, doc-id tracking

**The problem statement.** Six months after a pretrain, a regulator asks: "Did you train on document `X`?" Or a dedup bug is discovered: you need to identify and remove all 400K documents that were clustered with document `Y`. Or [[physics-of-lm-3]] hints that repeating a subset of high-quality data 4× helps factual recall, and you need to re-derive "the top-10% FineWeb-Edu classifier subset" from your current shards without rerunning the classifier. None of these are tractable without lineage.

**Content-addressed hashing.** Every raw crawled document gets a **stable primary key** = BLAKE3 hash of its original UTF-8 bytes. This is immutable: re-crawling the same URL tomorrow produces the same doc-id iff the bytes are identical. [[ccnet]]'s hash-based dedup was the earliest large-scale operational use of this pattern — "exact dedup via hash" scaled to Common Crawl because hashes fit in 16 bytes and bloom filters could process the stream. [[dolma]] generalized it: every filter stage reads `(doc_id, text)` and emits `(doc_id, text, attribute_k)`; the filtered corpus is just `SELECT doc_id WHERE attributes_satisfy(...)`.

**Attribute files as the lineage substrate.** The operational pattern from [[dolma]]:

```
raw/cc-2024-30/doc-{doc_id}.json          ← raw text + URL + crawl date
attributes/lang-id/doc-{doc_id}.json      ← {lang: "en", score: 0.94}
attributes/quality/doc-{doc_id}.json      ← {gopher_quality: 0.82, c4_rules: [pass, pass, fail]}
attributes/pii/doc-{doc_id}.json          ← {emails: 2, phones: 0, ips: 1, removed_spans: [[44,63],...]}
attributes/classifier/doc-{doc_id}.json   ← {fineweb_edu: 4, toxic: 0.02}
attributes/dedup/doc-{doc_id}.json        ← {minhash_sig: [...], cluster_id: 17234}
```

The attribute files are append-only. A new filter does not rewrite the corpus — it emits a new attribute directory. Building a training mix is then a *query*:

```python
docs = filter(
    lambda d: d.lang == "en"
          and d.quality.gopher_quality > 0.7
          and d.classifier.fineweb_edu >= 3
          and d.dedup.cluster_rank == 0
          and d.pii.removed_spans  # PII was scrubbed, not deleted
    , all_docs)
```

The query is the *mix specification*. Reproducing a mix months later = rerun the query against the same attribute files. The mix version = `hash(query_source + attribute_file_hashes)`.

**Cross-stage doc-id tracking.** The doc-id is the foreign key. When FineWeb ([[fineweb]]) runs per-snapshot MinHash and produces `cluster_id` attributes, those clusters reference doc-ids from the original CC dump. When Llama 3 ([[llama-3]]) generates rejection-sampled SFT responses conditioned on a pretrain document, the SFT row still carries `source_doc_id`. This is the ["reproducibility graph"] — you can trace any token in any SFT sample back to its pretrain origin. The [[olmo-3]] model-flow is a six-stage lineage graph: `raw → Dolma 3 → Dolma 3 Mix → Dolmino → Longmino → Dolci`. Each arrow is a filter; each filter emits attributes; the graph is the corpus.

**Reproducing a mix months later.** The three things that must survive:

1. **The raw shards** (content-addressed). Cost: raw, at frontier scale, is 100+ TB. Cheap object storage.
2. **The attribute files.** Cost: a few MB per filter per million docs. Negligible.
3. **The mix query** (versioned in git alongside the training config). Cost: kilobytes.

What *must* be rejected: storing the filtered, tokenized shards as the "corpus." That is an *output*, not a *source*. It loses the ability to re-derive a different subset. [[fineweb]] deliberately ships both (raw parquet for re-query + classifier scores as separate columns); [[dolma]] does the same with its `.attributes.jsonl` emission.

---

## 4. Code-repo filtering and PII removal as operations

**Code repos are a different beast.** Code is licensed, and licenses are per-repo, not per-file. [[llama-3]]'s pretrain includes code from many repos; the pipeline must:

1. **Repo-level license check.** For every file, look up its source repo's license. Permissive (MIT, Apache, BSD) → include. Restrictive (GPL, AGPL) → exclude (most labs exclude to avoid copyleft contamination). Unknown → exclude. The Stack v2 (used by DeepSeek-Coder, OLMo 3) builds this table from GitHub API scrape; the lookup is the slow step.
2. **Line-length heuristics.** Machine-generated minified JS (one line, 100K chars), long base64 blobs, SQL dumps — all look like code but carry no learning signal. The filter: drop files with `max_line_length > 1000` or `mean_line_length > 100`. ([[dolma]]'s heuristic; most pipelines now use similar thresholds.)
3. **Executability screens.** Does the code parse? Does it run the file's own doctests? Tree-sitter parse-check filters syntactically broken files. Llama 3's code pipeline ([[llama-3]]) goes further: "code-exec-filtered code" means the file was executed in a sandbox and produced non-error output. The filter is expensive (~10s per file at 10K containers/s) but catches ~5% of syntactically-valid but semantically-broken code.
4. **Secret detection.** API keys, AWS credentials, private SSH keys. TruffleHog / detect-secrets regex suites. Any file with a match → drop *the file*, or optionally scrub the key and keep the rest. [[dolma]]'s pipeline drops the document; the more aggressive approach is to scrub and audit.

**PII removal as an operation, not a policy.** The policy question ("should we remove phone numbers?") is settled upstream of the pipeline. The operational question is: given that policy, *how do you execute it reliably over 15 T tokens*?

[[dolma]]'s PII filter is a three-tier cascade:

| Tier | Method | Precision | Recall | Latency |
|---|---|---|---|---|
| 1 | Regex (email, phone, IP) | ~99% on standard formats | ~80% (misses obfuscated `john [at] gmail [dot] com`) | ~1 µs/doc |
| 2 | Named-Entity-Recognition classifier (spaCy / fastText) | ~90% | ~92% | ~5 ms/doc |
| 3 | LLM classifier (e.g. Llama Guard 3, [[llama-3]]) on flagged documents only | ~95% | ~96% | ~100 ms/doc |

Only Tier 1 runs on every document. Tier 2 runs on documents that pass Tier 1 unflagged but come from high-risk domains (forums, social media). Tier 3 runs on a ~1% sample for audit. The cascade is because at 15 T tokens (~3 B documents) running Tier 3 globally is prohibitive (~10 M CPU-hours).

**The three operational decisions that are usually undocumented.**

1. **Scrub vs delete.** Scrub (replace the span with `[EMAIL]`) preserves the document; delete drops it. [[dolma]] scrubs emails/IPs/phones; FineWeb redacts. The tradeoff: deleting is safer (no chance of partial PII surviving); scrubbing preserves surrounding context. Most modern pipelines scrub.
2. **Opt-out registries.** GDPR and analogous regimes require the ability to honor "remove my data" requests. The operational implementation: a list of doc-ids (or URLs, or content hashes) maintained in a registry; every pipeline run filters against the current registry. Registry size at frontier scale: ~10K entries, growing. The cost is the lookup on every shard read; at 40 GB/s this must be a bloom filter, not a database query.
3. **Audit log for removal events.** Every `(doc_id, removed_span, reason, timestamp, pipeline_version)` tuple is appended to a log shard. Log cardinality ~10 M rows per pretrain run. The log is the *legal artifact* — when a regulator asks what PII was removed, the log is the answer. [[dolma]]'s attribute-file design makes this natural: `attributes/pii/` *is* the audit log.

---

## 5. Tokenizer extension pitfalls — the post-pretrain `<think>` trap

Every modern chat/reasoning model adds special tokens *after* pretraining. [[phi-4]] has a `<think>` block wrapping reasoning traces; Qwen 3 ([[qwen-3]]) exposes a "thinking budget"; OLMo 2 Instruct ([[olmo-2]]) reuses Tulu 3's chat template with `<|im_start|>` / `<|im_end|>`. These tokens did not exist in the pretrain tokenizer. Adding them without care is one of the most common SFT regressions.

**The problem.** When you extend a tokenizer from `V` to `V + k` tokens, you have to extend the input embedding matrix `E_in ∈ R^{V × d}` and the output projection `E_out ∈ R^{V × d}` to `(V + k) × d`. The new rows are *uninitialized*. Typical library defaults initialize them with `N(0, 0.02)` — the same scheme used at pretrain for fresh embeddings.

This is almost always wrong. At pretrain init, *every* embedding is drawn from `N(0, 0.02)`, and the model co-trains with the LayerNorms, biases, and attention weights that were also initialized to small values. The model learns to handle small-magnitude embeddings because everything is small-magnitude. After 15 T tokens of pretraining, the learned embeddings for existing tokens have settled at typical L2 norms of ~1.0–1.5. Injecting new `N(0, 0.02)` embeddings — L2 norm ~0.04 — means the new token's embedding is 30× smaller than its neighbors. The first time `<think>` is fed into the model, attention basically ignores it: its query / key projections produce near-zero logits, so the post-softmax attention weights to and from `<think>` collapse to the uniform distribution minus epsilon. The chat template silently behaves as if the special tokens were not there for the first ~1000 SFT steps, after which the embedding finally grows and the model starts attending to `<think>` — at which point the *other* weights have been partly updated under the wrong assumption, producing measurable downstream regression on reasoning benchmarks.

**The recipe — mean-of-neighbors initialization.** The fix, cited by [[olmo-2]]'s Tulu 3 application and by the Phi-4 report ([[phi-4]]) implicitly through its "no chat-template regression" claim:

```python
def extend_tokenizer_and_model(model, tokenizer, new_tokens):
    """Extend vocab + init new rows from mean of sub-word decomposition."""
    old_vocab_size = len(tokenizer)
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    E_in  = model.get_input_embeddings().weight.data
    E_out = model.get_output_embeddings().weight.data
    for tok in new_tokens:
        new_id = tokenizer.convert_tokens_to_ids(tok)
        # Decompose "new-token" into its pre-extension sub-word pieces.
        surface = tok.lstrip("<|").rstrip("|>").replace("_", " ")
        sub_ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
        if not sub_ids:              # "<think>" may decompose to <, think, > under old vocab
            sub_ids = tokenizer("think", add_special_tokens=False)["input_ids"]
        E_in[new_id]  = E_in[sub_ids].mean(0)
        E_out[new_id] = E_out[sub_ids].mean(0)
```

The new token's embedding now has approximately the correct L2 norm — because it's the mean of vectors that already have the right norm. The semantic content is a reasonable prior: `<think>` initialized from the mean of `think`, `thought`, `reason` is already "in the right neighborhood" before SFT starts. SFT steps then refine a good initial state rather than climbing out of a pit.

**The tokenizer-freezing discipline.** §1 said: "Pre-reserve 200–500 unused tokens." This is why. If you know in advance that future SFT phases will add `<think>`, `<tool_call>`, `<search>`, `<code>`, etc., reserve the ids at pretrain time. The embedding slots train to some near-zero mean (they're never seen in pretrain), but at SFT time you *rename* the reserved slot from `<|reserved_7|>` to `<think>` without resizing the matrix. No new rows; no initialization question; no drift. This is what Llama 3 ([[llama-3]]) does with its 256 reserved slots in the 128K vocab.

**Diagnosis — how to detect the drift when it happens.** Two cheap logs:

- `||E_in[new_id]||` vs `median(||E_in[existing_id]||)`. Alarm if ratio < 0.3 at any SFT step.
- Attention probability mass assigned to the new token position, averaged over a validation batch, across layers. Alarm if it's significantly below `1/seq_len`.

Both signals show the drift as it happens; both can catch a silent regression before the downstream eval does.

---

## 6. The full pipeline as one graph

Combining §§1–5 into the operator's mental model — see `figures/pipeline-ops.html` for the interactive version.

```
              [raw shards, content-addressed]
                         │
                   ┌─────┴─────┐
                   ▼           ▼
            [filter stage 1]  [code filter]
            lang-id, quality   license, exec, secrets
                   │           │
                   └─────┬─────┘
                         ▼
                   [PII scrub]  ← opt-out registry, audit log
                         │
                   [dedup]       MinHash per-dump; doc-id cluster assignment
                         │
                   [tokenize]    tokenizer.json locked by BLAKE3; reserved slots live
                         │
                   [shuffle]     per-rank deterministic seed; data-iter state
                         │
                   [pack]        MDS shards, ~200 MB each; uniform size
                         │
                         ▼
                      [trainer]
```

Every edge is a `(doc_id, attribute) → (doc_id, attribute')` transformation that appends to the attribute graph. Every edge has a failure mode:

- raw → filter1: **new crawl date, different snapshot; doc-id collision if content changed silently**.
- filter1 → PII: **regex false-negatives on obfuscated spans; audit log entry missing → regulator-facing risk**.
- PII → dedup: **removing PII creates new near-duplicates that did not exist before; dedup must run after PII, not before**.
- dedup → tokenize: **tokenizer version mismatch between pretrain and SFT; BLAKE3 check catches this**.
- tokenize → shuffle: **non-deterministic shuffle; resume desync returns** (ch-06 §5.1).
- shuffle → pack: **shard size imbalance; straggler worker blocks the step**.
- pack → trainer: **dataloader state-dict missing `mix_pointer`; §3 mix version lost**.

The interactive figure walks through each stage, exposes the typical throughput on the 15 T-token [[fineweb]] / 14.8 T-token [[deepseek-v3]] scale, and lists the lineage attributes that must be present at each handoff.

---

## Connections and what's next

- **[[dolma]] / [[fineweb]] / [[ccnet]]** — the three pipelines this chapter operationalizes; all earlier chapters treat them as *designs*, here as *running systems*.
- **[[llama-3]] / [[olmo-2]] / [[olmo-3]] / [[qwen-3]] / [[deepseek-v3]] / [[phi-4]]** — six production tokenizer decisions, six shard-layout choices, six lineage conventions.
- **[[physics-of-lm-3]]** — the capacity argument that forces small-model vocab-size rethinks.
- **ch-06 (checkpointing)** — data-iterator state lives in the checkpoint; a bad shard layout silently breaks resume.
- **ch-10 (curation pipelines)** — the design; this chapter is the operations.
- **ch-12 (dedup)** — the next stage after tokenize; MinHash lives on the `doc_id → signature` lineage edge.
- **ch-17 (lab)** — you will implement a minimal CCNet-style pipeline end-to-end; the lineage and tokenizer discipline here are the interfaces.

## Further reading

- Sennrich, Haddow & Birch 2015 — [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) — canonical BPE.
- Kudo & Richardson 2018 — [arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226) — SentencePiece; byte fallback.
- [[dolma]] — §3 of the paper is the attribute-file design; §4 is the PII cascade.
- [[fineweb]] — `datatrove` codebase; the per-dump MinHash ablation is the canonical reference on shard-aware dedup.
- [[ccnet]] — hash-based lineage as a scalability argument.
- [[llama-3]] — 128K tokenizer; reserved-slot discipline; code-exec filter.
- [[olmo-3]] — model-flow worldview; OLMES + OlmoTrace + decontam tooling.
- [[qwen-3]] — multilingual tokenizer sizing at 36 T tokens, 119 languages.
- [[deepseek-v3]] — Chinese-English tokenizer; FP8 training → shard throughput is decisive.
- [[phi-4]] — textbook synthetic data; special-token extension for reasoning traces.

## Companion visualization

**[figures/pipeline-ops.html](figures/pipeline-ops.html)** — interactive production-pipeline diagram. Click any stage (raw → filter1 → filter2 → dedup → tokenize → shuffle → pack → train) to see: (a) the lineage attributes emitted at that edge, (b) typical throughput on a 15 T-token pretrain scale, (c) the common operational failure modes for that stage and their observable symptoms. Use it to internalize how the lineage graph accumulates: every stage appends attributes, no stage rewrites the doc-id, and the final trainer consumes a compact `(doc_id → shard_offset)` map rather than a filesystem of text.
