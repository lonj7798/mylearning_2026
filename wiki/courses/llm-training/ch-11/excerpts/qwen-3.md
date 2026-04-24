---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/qwen-3.md
source_url: https://arxiv.org/abs/2505.09388
created_at: "2026-04-23"
---

# Excerpt: Qwen 3 — multilingual tokenizer for 119 languages and the three-stage mix

**Source library:** `wiki/raw-data/llm-training/model-reports/qwen-3.md`
**Paper:** Qwen Team 2025, "Qwen3 Technical Report".

---

## Why this source anchors ch-11

Qwen 3 is the frontier multilingual counterpoint to Llama 3 / OLMo 3. The source's one-line hook — "**36T** tokens across **119** languages and dialects" — forces every tokenizer and lineage decision to confront problems that an English-centric pipeline sidesteps. Ch-11's §1 vocab-size table names 151K as Qwen 3's sizing; §3's three-stage pretraining curriculum frames the multi-mix lineage argument. This excerpt expands both.

Ch-11 cites Qwen 3 three times: for the 151K multilingual vocab (§1), for instance-level mixing as a lineage problem (§3), and for the three-stage pretraining mix schedule (§3, §5). This excerpt pulls those forward.

---

## 36T tokens × 119 languages — the vocab-size forcing function

From the source (lines 36-40):

> ### Pretraining data
> - Total of **36T tokens** across **119 languages and dialects**.
> - Data expansion includes:
>   - OCR-style text extraction from large PDF corpora using **Qwen2.5-VL**
>   - synthetic math data from **Qwen2.5-Math**
>   - synthetic code/data variants from **Qwen2.5-Coder** and related models
> - The report says the data is annotated at large scale for educational value, domain, and safety, then mixed at the **instance level** using proxy-model ablations.

119 languages breaks a 32K or 50K vocab decisively. Consider:

- **Chinese (simplified + traditional)** alone requires 2K+ high-frequency characters as single tokens for reasonable compression; without them, each Chinese character is 2–3 bytes (UTF-8) tokenized individually = 3× sequence inflation.
- **Korean / Japanese** similarly need 500–1000 per-language high-frequency tokens.
- **Arabic, Hebrew, Devanagari, Thai, etc.** each need a base vocab of ~200–500 common glyphs.
- **Long-tail languages** (Swahili, Vietnamese, Tagalog) often tokenize reasonably under shared Latin-script sub-word merges but still require some language-specific frequency coverage.

The math of 119 languages × ~500 language-specific tokens = ~60K tokens just for non-English coverage. Add English (20K typical in a well-balanced tokenizer), code (~10K), and special tokens / reserved slots (~500), and the natural target is **~100–150K vocab**. Qwen 3 reports ~151K, which is consistent with this calculation.

Ch-11 §1's vocab-size table names 151K for Qwen 3 with embedding cost 1.24B parameters at `d=4096`. For Qwen 3's larger models (32B dense, 235B MoE) this is a small fraction of budget; for the 0.6B small model it's ~20% of parameter budget — one of the reasons the Qwen 3 family keeps a *consistent* tokenizer across sizes despite the proportional cost.

---

## Byte fallback as a multilingual-pipeline precondition

Qwen 3 uses SentencePiece-style byte-level BPE with `byte_fallback=True` (implicit in the report; standard practice for modern multilingual models). Ch-11 §1 emphasizes this. For a 119-language pipeline the alternative — requiring every possible byte sequence to be tokenizable via learned merges — is infeasible: the long tail (rare scripts, mixed-language documents, unicode combining characters, emoji-as-text) would produce OOV errors.

With byte fallback, any input is encoded (possibly as individual UTF-8 bytes for the rare cases). The cost is sequence-length inflation for rare content; the benefit is **zero training-time OOV crashes**, which is the operational requirement at 36T tokens / ~9 billion documents.

Ch-11 §2's "tokenizer version mismatch" failure mode lands differently here: under byte fallback, a mismatch produces *drifted but valid* token sequences (the same text tokenizes differently under two vocabularies, but never fails). This is more dangerous than an OOV crash — the trainer doesn't alert, so the mismatch silently corrupts the training run. The BLAKE3(tokenizer.json) assertion ch-11 §2 recommends catches this before the silent drift starts.

---

## The three-stage pretraining — a multi-mix lineage case

From the source (lines 45-49):

> ### Three-stage pretraining
> 1. **General stage:** over **30T** tokens at sequence length **4096**.
> 2. **Reasoning stage:** about **5T** higher-quality tokens with more STEM, coding, reasoning, and synthetic data, still at **4096**.
> 3. **Long-context stage:** hundreds of billions of tokens at **32768** sequence length, later supporting longer inference contexts.

Three stages, three mixes:

| Stage | Tokens | Context | Mix emphasis |
|---|---|---|---|
| General | 30T | 4096 | Broad web + code + multilingual |
| Reasoning | 5T | 4096 | STEM, code, synthetic reasoning |
| Long-context | 100B+ | 32768 | Long documents, staged RoPE extension |

For ch-11 §3's lineage argument, each stage is a separate query over the raw pool. The General-stage mix query might look like:

```python
docs = filter(
    lambda d: d.lang in ALL_119_LANGS
           and d.quality.qwen_edu >= 2
           and d.length_tokens <= 4096
           and d.decontam.overlapped == []
    , all_docs)
```

The Reasoning-stage query bumps the quality threshold and restricts the domain:

```python
docs = filter(
    lambda d: d.domain in {"stem", "code", "math", "reasoning"}
           and d.quality.qwen_edu >= 4
           and d.length_tokens <= 4096
    , all_docs)
```

Each query hashes to a different mix identity. The dataloader's mix-pointer (ch-11 §2) tracks which query is active; the resume-state must include the pointer.

This is conceptually identical to [[excerpts/olmo-3]]'s Dolma 3 / Dolmino / Longmino structure. The operational pattern is now well-established across frontier labs: **pretraining is a sequence of mix queries, not a single dataset**.

---

## Instance-level mixing as a proxy-model lineage problem

From the source (line 40):

> ...data is annotated at large scale for educational value, domain, and safety, then mixed at the **instance level** using proxy-model ablations.

"Instance-level" is the critical word. Earlier approaches (GLaM, PaLM) used *bucket-level* mixing — the entire code bucket gets weight 0.15, the entire web bucket gets 0.6, etc. Qwen 3's instance-level mixing means each document has its own sampling probability, derived from its attribute vector:

```
p(sample doc_i) ∝ f(quality_i, domain_i, safety_i, language_i)
```

where `f` is learned via proxy-model ablations (train a small model on various weightings, optimize for downstream eval).

Ch-11 §3 does not belabor instance-level mixing (that's ch-13's territory) but the *operational* requirement does touch §3: every document needs a rich-enough attribute vector to drive the sampling function. Educational-value score, domain label, safety flag, language, and more. The attribute file is not optional for instance-level mixing — it's the substrate.

Qwen 3's choice forces a more elaborate attribute schema than Llama 3's or OLMo 3's (which emphasize bucket-level mixing with finer decontam / classifier gates). For a pipeline operator, the implication is: **if you plan to do instance-level mixing later, emit the attributes now**. Retrofitting attribute files against a trained classifier six months later costs a full pipeline re-run.

---

## The reasoning + general mix merge — a post-training lineage challenge

From the source (lines 53-56):

> ### Post-training
> - Stage 1-2: build reasoning ability with **long-CoT cold-start finetuning** and **RL** focused on math and coding.
> - Stage 3-4: merge data with and without reasoning paths, then run **general-domain RL**.

Stages 3-4 — "merge data with and without reasoning paths" — is the operational story behind the "thinking budget" interface. Qwen 3's single unified model must produce both long-CoT and short-response outputs depending on inference-time toggle. The training data is a mix of (prompt, short-response) and (prompt, long-CoT-response) pairs, with a system-level controller deciding which style to emit.

For ch-11, the lineage implication: **every training example carries a `reasoning_mode` attribute** — whether it's a thinking or non-thinking pair. Inference-time control requires that the training set is balanced across both. The attribute cannot be inferred post-hoc from the response length (a long non-CoT response is not the same as a long CoT response); it must be set at data-generation time and tracked through the pipeline.

This exemplifies ch-11 §3's "attribute-is-part-of-data" argument: for Qwen 3's thinking-budget interface to work in deployment, the *operational decision* to tag each training example's reasoning mode must be made at pipeline build time. Retrofitting it later is impossible.

---

## YARN and Dual Chunk Attention — downstream-of-tokenizer context extension

From the source (line 51):

> - Uses **YARN** and **Dual Chunk Attention** to increase usable context during inference.

This is a ch-11-adjacent detail. YARN (Yet Another RoPE extensioN) reconfigures positional encodings to enable longer context without retraining; Dual Chunk Attention restructures the attention pattern for long sequences. Both are *inference-time* changes that do not touch the tokenizer or the pipeline.

But they imply operational decisions at training time: the long-context stage (100B tokens at 32K) generates the data that YARN extrapolates from. If that stage's shards are mis-laid-out (ch-11 §2 straggler failure) or mis-mix-pointered (§3 resume failure), the YARN extension starts from a weaker base. Ch-11 §2's "uniform shard sizes" guideline applies with extra force to long-context stages where a single 32K-example shard is ~128 KB — much smaller than typical pretrain examples.

---

## What Qwen 3 does not solve — the 0.6B model's vocab

The Qwen 3 family spans 0.6B to 235B (MoE). All share the same ~151K vocab. For the 0.6B model this is operationally painful:

- Embedding table: `151K × d=1024 × 2` = 310 M parameters.
- Total model: 0.6B.
- Embeddings are **52% of the model**.

Qwen 3 accepts this because *maintaining a unified family tokenizer* has its own benefits: shared chat templates, shared SFT data, shared deployment infrastructure. The tradeoff is explicit in the report's family-design philosophy (line 33: "Dense models from 0.6B to 32B"); ch-11 §1 mentions but does not resolve it, because the decision is one of lab strategy, not operational optimality.

The Phi-4 line ([[excerpts/phi-4]]) takes the other approach: smaller models get smaller vocabs, accepting the interoperability cost for the capacity budget. Neither choice is wrong; both cases are instances of ch-11 §1's "vocab-size-should-co-vary-with-model-size" principle, with the compromise drawn at different thresholds.

---

## What to take from Qwen 3 for ch-11

1. **119 languages force ~150K+ vocab**; byte fallback is not optional at this scale.
2. **Three-stage pretraining** (general → reasoning → long-context) is the multi-mix pattern; each stage is a query with its own mix-pointer.
3. **Instance-level mixing requires rich attribute vectors** emitted at pipeline build time; retrofitting is infeasible.
4. **Reasoning-mode tagging** (thinking vs non-thinking) is a training-time attribute that enables inference-time control — the attribute is part of the data, not inferable from the response.
5. **Unified-family tokenizer cost vs per-size optimization** is a strategic decision; for small models in a multilingual family, the cost can be 50%+ of parameter budget.

---

## Connections

- [[excerpts/ccnet]] — multilingual-from-day-1 ancestor; Qwen 3 scales CCNet's per-language operational discipline to 119 languages.
- [[excerpts/llama-3]] — 128K vocab + reserved slots; Qwen 3 at 151K is the multilingual upgrade.
- [[excerpts/olmo-3]] — three-stage base training (pretraining / mid-training / long-context); Qwen 3 is structurally similar.
- [[excerpts/deepseek-v3]] — Chinese-English tokenizer at 14.8T tokens; subset of Qwen 3's scope.
- [[excerpts/phi-4]] — the opposite family-design decision (per-size vocab optimization).
- [[ch-11]] — §1 (151K multilingual vocab, byte fallback), §3 (three-stage mix lineage, instance-level attributes), §5 (unified tokenizer across family vs per-size compromise).
