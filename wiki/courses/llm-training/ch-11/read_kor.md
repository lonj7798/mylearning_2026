<!-- chapter: ch-11
     track: data
     title: Data Operations — Tokenizers, Shards, Lineage, PII
     sources: [[dolma]], [[fineweb]], [[ccnet]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[qwen-3]], [[deepseek-v3]], [[phi-4]], [[physics-of-lm-3]]
     external: https://arxiv.org/abs/1508.07909 (BPE), https://arxiv.org/abs/1808.06226 (SentencePiece)
     figures: figures/pipeline-ops.html
-->

# 11장 — Data Operations: Tokenizer, Shard, Lineage, PII

> **핵심 통찰.** Curation은 model에 *어떤* text가 들어가는지를 결정한다. **Operation**은 *같은 model을 다시 만들 수 있는지*를 결정한다. 15T-token corpus는 dataset이 아니다. 그것은 `(content-hash → doc-id → filter-attributes → shard-offset → tokenizer-id)` graph이며, 6개월의 pipeline drift, 두 번의 tokenizer migration, 세 번의 PII-removal incident, hardware change를 견뎌야 한다. Graph의 edge 하나를 잃으면 run은 재현 불가능해진다. tokenizer → shard edge를 잃으면 training을 *resume*할 수도 없다.
>
> **가이드라인.** 첫 crawl stage부터 모든 byte를 content-addressed(raw document의 BLAKE3)로 다루라. Filter verdict는 filtered shard가 아니라 *document별 attribute vector*로 저장하라. 그래야 upstream filter를 다시 실행하지 않고 어떤 subset도 다시 도출할 수 있다. Vocabulary size는 실제로 학습하는 최악의 language에 맞춰라(multilingual이면 120K+, code-heavy면 128K, English-only base에서만 32K). Post-pretraining에 tokenizer를 확장할 때는 새 embedding을 sub-word decomposition의 평균으로 초기화하지 않고는 절대 확장하지 마라. 그렇지 않으면 새 `<think>`와 `<tool_call>` token이 약 1000 SFT step 안에 chat template을 non-reproducibility로 drift시킨다.

---

## 이 장이 존재하는 이유

9-10장은 training corpus의 *shape*, 즉 filter, dedup, quality cascade가 최종 model quality에 load-bearing이라고 주장했다. 그것은 curator의 관점이었다. 이 장은 operator의 관점을 취한다. Filter가 존재한 뒤, 어떻게 15T token([[fineweb]]) 또는 36T token([[qwen-3]])에서 **그것들을 실행**하면서 단일 document의 lineage도 잃지 않고, 고객의 phone number를 checkpoint에 leak하지 않으며, pretraining Stage 2가 새 language를 추가할 때 tokenizer가 조용히 drift하지 않게 할 것인가?

Operation을 잘못하면 더 나쁜 model이 나오는 정도가 아니다. *의심스러운* model이 나온다. Dolma의 기여([[dolma]])는 주로 3T-token corpus가 아니었다. Raw shard에서 attribute pass를 replay해 final corpus의 어떤 subset도 다시 도출할 수 있도록 document별 `.attributes.jsonl` file을 emit하는 *toolkit*이었다. FineWeb의 기여([[fineweb]])는 15T token이 아니었다. Per-snapshot으로 deterministically pipeline을 다시 실행하고, classifier-score threshold를 hard-coded filter가 아니라 *parameter*로 두는 `datatrove` codebase였다. OLMo 3의 기여([[olmo-3]])는 Dolma 3만이 아니었다. Dolma 3 / Dolma 3 Mix / Dolmino / Longmino / Dolci 각각을 별도로 tracked lineage로 보고, 각자 filter gate, tokenizer-consumer assertion, audit trail을 갖는 *model-flow worldview*였다.

이 장은 training run 이후에도 살아남는 네 가지 operational artifact, 즉 **tokenizer**, **shard layout**, **lineage graph**, **PII / secrets audit log**를 중심으로 구성된다. 각각은 자신의 failure mode가 있고, 각각은 적어도 하나의 잘 문서화된 training incident의 root cause였다.

---

## 1. Scale에서의 tokenizer construction — BPE, WordPiece, SentencePiece

Tokenizer는 trainer의 *양쪽*에 놓인 유일한 artifact다. Training shard를 만들고, final weight와 함께 shipping된다. Post-hoc으로 잘못을 고치는 것은 full retrain 없이 불가능하다.

**세 algorithm family, 간단히.**

- **Byte-Pair Encoding (BPE)** — Sennrich, Haddow, Birch 2015([arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)). Greedy bottom-up merge. Byte(또는 char)에서 시작해 vocabulary가 target size에 도달할 때까지 가장 빈번한 adjacent pair를 반복해서 merge한다. 학습된 tokenizer는 *merge table*(pair의 ordered list)과 base vocab이다. Decoding은 merge table 위의 stack-walk다. GPT-2 / GPT-3 / GPT-4 / Llama가 쓰는 것이 BPE다.
- **WordPiece** — Schuster & Nakajima 2012, BERT가 사용. 같은 bottom-up merge idea지만 merge criterion이 raw frequency가 아니라 unigram LM 아래 likelihood-gain이다. 실제로 약간 다른 segmentation을 만든다. Algorithmic difference는 LLM pretraining에서는 드물게 중요하며, 대부분의 lab은 BPE 또는 byte-level variant로 수렴했다.
- **SentencePiece** — Kudo & Richardson 2018([arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)). 두 가지 중요한 움직임이 있다. (a) **raw byte 위에서 작동**하고 whitespace를 special character(`▁`)로 다룬다. 따라서 tokenizer가 완전히 reversible하다. `decode(encode(text))` round-trip은 bit-exact다. (b) Alternative training algorithm으로 **unigram LM**을 제공하며, 큰 candidate set 위에서 EM으로 vocab을 학습한다. 거의 모든 modern production tokenizer는 SentencePiece 또는 SentencePiece-flavored byte-level BPE다. Llama 3([[llama-3]])는 SentencePiece-style byte fallback으로 만든 128K BPE를 사용한다. OLMo 2 / 3([[olmo-2]], [[olmo-3]])도 같다. Qwen 3([[qwen-3]])은 119-language coverage를 위해 byte-level scheme을 151K로 확장한다.

**Vocab-size tradeoff — 32K vs 128K vs 200K.** 선택은 세 가지 긴장 사이에 있다. (a) sequence length(bigger vocab → document당 token 수 감소 → cheaper training), (b) embedding-table cost(input+output embedding에 대해 `vocab × d_model × 2`), (c) under-represented script 또는 programming language에서의 downstream quality.

| Vocab size | Typical choice | Sequence compression | Embedding cost at d=4096 | When it's right |
|---|---|---|---|---|
| 32K | GPT-2, early Llama 1, OLMo 1 | baseline | 0.26 B params | English-only base; compute-scarce; simple chat |
| ~100K | GPT-4, Claude, Gemini | English에서 32K보다 ~20% fewer tokens, code에서 ~2× fewer | 0.82 B | code-heavy, multilingual, production frontier |
| 128K | Llama 3 ([[llama-3]]), OLMo 2/3, DeepSeek-V3 ([[deepseek-v3]]) | 100K보다 ~15% fewer, code와 non-Latin script에서 더 좋음 | 1.05 B | 2024+ frontier default |
| 151K | Qwen 3 ([[qwen-3]]) | CJK, Arabic, Korean에서 최고 | 1.24 B | 119-language coverage |
| 200K | experimental(인용된 report 중 ship한 곳 없음) | ~150K 이후 diminishing returns | 1.64 B | long-tail script가 정말 있을 때만 |

32K → 128K jump가 중요하다. [[physics-of-lm-3]]는 비용을 다시 framing한다. Embedding table에 들어간 parameter는 factual-knowledge storage에 들어가지 *않는* parameter다. Allen-Zhu & Li는 knowledge capacity가 parameter count에 선형으로 scale한다고 주장한다. Total 7B parameter에서 128K vocab은 embedding에 1.05B = budget의 ~15%를 쓴다. 70B에서는 ~1.5%다. 따라서 small model은 더 작은 vocabulary를 가져야 한다(Phi-3는 32K를 썼고, [[phi-4]]는 더 큰 vocab을 쓰지만 model이 14B라 frontier보다 여전히 훨씬 작다). Vocab-size decision은 *model size와 함께 변한다*. 70B tokenizer를 1B distill에 재사용할 때 일상적으로 위반되는 사실이다.

**Code-vs-natural BPE merge.** 같은 training text도 domain mix에 따라 매우 다른 merge table을 만든다. Code에서 BPE는 `    `(four-space indent)를 single token으로 공격적으로 merge하고, `    def ` / `    return ` / `, self):`를 high-frequency merge로 학습하며, `camelCaseIdentifiers`를 case boundary에서 segment한다. Natural text에서는 merge가 prefix(`un-`, `pre-`, `re-`), common suffix(`-tion`, `-ing`, `-ed`), high-frequency whole-word token이다.

Operational consequence: 50/50 code/natural mix로 학습한 tokenizer는 둘 다 under-tokenize한다. GPT-2의 50K tokenizer는 긴 rare English word(`disestablishmentarianism`이 single token)를 vocab에 낭비하는 것으로 유명하고, 그 대가로 `    elif `를 `    ` + `el` + `if ` + space로 split한다. Python에서 약 50% sequence-length inflation이다. Llama 3([[llama-3]])는 code를 명시적으로 up-weight한 mix로 128K tokenizer를 학습해 이를 해결한다. 결과적으로 같은 input에서 Python과 Go는 GPT-3.5 tokenizer보다 line당 token 수가 ~25% 적다.

Frontier lab들이 사용하는 recipe는 [[llama-3]] / [[olmo-2]] / [[deepseek-v3]]에서 추론하면 다음과 같다.

1. 약 500 GB text의 **tokenizer training corpus**를 sample한다. Domain ratio는 raw-web distribution이 아니라 *final* pretraining mix와 맞아야 한다. 그렇지 않으면 80%+ web-dominated BPE가 code와 math를 drown한다.
2. SentencePiece로 byte-level BPE를 학습한다. Target vocab = 128K(code-heavy) 또는 151K(multilingual), `byte_fallback=True`를 켜서 어떤 byte도 representable하게 한다.
3. Special token block을 **미리 reserve**한다. `<|begin_of_text|>`, `<|end_of_text|>`, `<|im_start|>`, `<|im_end|>`와 future extension을 위한 200-500개의 unused-but-reserved id(`<|reserved_0|>` … `<|reserved_500|>`). 이것이 "leave space for `<think>`" decision이다. §5는 이를 생략하면 왜 full pretrain 비용이 드는지 설명한다.
4. Full pretrain tokenizing을 on-the-fly로 실행하거나 pre-tokenize and shard한다(§2가 tradeoff를 다룬다). Tokenizer를 freeze하고 BLAKE3 hash가 있는 `tokenizer.json`을 emit한다. §2의 모든 shard는 이 hash로 tag된다.

---

## 2. Shard와 streaming — Mosaic, WebDataset, tar vs parquet

현대 pretrain은 full epoch 동안 약 15 TB의 tokenized data를 읽는다. 1024개 H100이 각각 약 40 MB/s의 token을 소비하면 aggregate read bandwidth는 약 40 GB/s다. File-system layout은 이에 맞아야 한다.

**Production의 세 shard format.**

- **WebDataset (`.tar` shards)** — 원래 NVIDIA DALI에서 온 형식. 각 shard는 수천 개의 `.json` 또는 `.txt` document를 담은 plain tar file이며 synthetic name(`{split}-{shard:06d}.tar`)으로 address된다. Read는 streaming-sequential이다. Tar를 열고 순서대로 document를 꺼낸다. Random access는 `O(shard-size)`다. Shard 중간의 document로 seek하는 것은 비싸다. Pretraining에는 좋다(sequential sweep이 자연스럽다). Document별 lookup을 원하는 ablation run에는 나쁘다.
- **MosaicML Streaming (`.mds` shards)** — MDS = "Mosaic Dataset Shards." 각 shard는 shard 안에서 random access가 `O(log n)`이 되도록 명시적 in-file index를 가진 binary file이다. 핵심적으로 이 format은 **deterministic resumable streaming**을 지원한다. Dataloader state = (shard, offset, rng)가 몇 byte에 들어가고 clean하게 round-trip한다. 이것이 [[olmo-3]] model-flow를 가능하게 한다. 모든 stage(pretrain → mid-train → long-context)가 다른 mix에서 clean하게 resume된다.
- **Parquet** — columnar, compressed, row-group index가 있다. PyArrow / Polars를 통한 read throughput은 wide schema(text + 20 attribute column)에 탁월하다. [[fineweb]]을 만든 `datatrove` pipeline은 parquet을 native로 emit한다. 이것은 *curation* pipeline에 올바른 format이다. 각 row가 `(doc_id, text, lang_score, classifier_score, minhash_signature, dedup_cluster_id, ...)`이기 때문이다. 대부분의 frontier pretrain은 마지막 stage에서 pure streaming throughput을 위해 parquet을 MDS 또는 tar로 다시 pack한다.

**Read-throughput vs seek-cost.** ~40 GB/s를 읽는 1024-GPU pretrain에서 관련 metric은 random-access latency가 아니라 worker node별 sustained sequential read다. 병적인 shard layout, 예를 들어 corpus의 1%를 담은 shard 하나는 dataloader 하나가 peer보다 100배 느리게 끝나는 straggler를 만든다. Operational fix는 **uniform shard size**다. [[fineweb]]은 15T release에서 200 MB per shard가 sweet spot이었다고 보고한다(99% shard가 median의 1.5배 이내). Llama 3는 15.6T token에 대해 비슷한 ~500 MB shard size를 사용한다([[llama-3]]). 너무 작은 shard(10 MB)는 open/close overhead를 키우고, 너무 큰 shard(5 GB)는 어떤 read-sample도 long-tail event로 만든다.

**Tokenize-offline vs tokenize-on-the-fly.** 두 regime:

| Mode | When | Pros | Cons |
|---|---|---|---|
| 한 번 offline tokenize하고 `int32[]`로 저장 | Tokenizer가 frozen이고, multiple epoch 또는 ablation sweep을 실행할 계획일 때 | Dataloader 4배 빠름(re-tokenization 없음); deterministic replay | Tokenizer를 바꾸면 re-tokenization이 full shard rewrite(15T token에서 ~$10K, 128-node cluster 1주) |
| Text shard에서 on-the-fly tokenize | 초기 experimentation, tokenizer가 아직 flux에 있을 때 | Tokenizer change가 쌈; text가 human-readable로 유지 | Dataloader CPU-bound; 15-25% throughput penalty |

DeepSeek-V3([[deepseek-v3]])는 FP8 training과 극단적 GPU utilization 때문에 offline tokenize한다. Trainer는 20% CPU tax를 감당할 수 없다. OLMo 2 / 3는 offline tokenize하지만 raw-text shard도 release하므로 downstream researcher가 자신의 vocab으로 re-tokenize할 수 있다.

**Per-rank dataloader state.** 1024-way data-parallel dimension의 FSDP 아래에서 각 rank는 shard list의 *portion*을 가진다. Ch-06 silent-failure mode가 돌아온다. Resume across에서 per-rank shard assignment가 보존되지 않으면, 다른 rank가 이미 소비한 shard를 조용히 다시 학습한다. Mosaic Streaming은 dataloader state를 compact serializable `(rank_shard_list, within_shard_offset, rng_seed)` tuple로 만들어 이를 해결한다. [[olmo-3]]의 8× SFT throughput improvement("Open Instruct에서 Olmo Core로 SFT를 옮기자 throughput이 8배 개선되었다고 보고")는 여기에 기반한다.

---

## 3. Dataset lineage와 versioning — content-addressed hashing, doc-id tracking

**Problem statement.** Pretrain 6개월 뒤 regulator가 묻는다. "문서 `X`로 학습했나요?" 또는 dedup bug가 발견되어 document `Y`와 cluster된 400K document 전부를 식별하고 제거해야 한다. 또는 [[physics-of-lm-3]]가 high-quality data subset을 4배 반복하면 factual recall에 도움이 된다고 암시해서, classifier를 다시 실행하지 않고 current shard에서 "top-10% FineWeb-Edu classifier subset"을 다시 도출해야 한다. Lineage 없이는 어느 것도 tractable하지 않다.

**Content-addressed hashing.** 모든 raw crawled document는 **stable primary key** = original UTF-8 byte의 BLAKE3 hash를 받는다. 이것은 immutable하다. 내일 같은 URL을 다시 crawl하면 byte가 동일할 때만 같은 doc-id가 나온다. [[ccnet]]의 hash-based dedup은 이 pattern의 가장 이른 large-scale operational use였다. "Hash를 통한 exact dedup"은 hash가 16 byte에 들어가고 bloom filter가 stream을 처리할 수 있었기 때문에 Common Crawl까지 scale했다. [[dolma]]는 이를 일반화했다. 모든 filter stage가 `(doc_id, text)`를 읽고 `(doc_id, text, attribute_k)`를 emit한다. Filtered corpus는 단지 `SELECT doc_id WHERE attributes_satisfy(...)`다.

**Lineage substrate로서의 attribute file.** [[dolma]]의 operational pattern:

```
raw/cc-2024-30/doc-{doc_id}.json          ← raw text + URL + crawl date
attributes/lang-id/doc-{doc_id}.json      ← {lang: "en", score: 0.94}
attributes/quality/doc-{doc_id}.json      ← {gopher_quality: 0.82, c4_rules: [pass, pass, fail]}
attributes/pii/doc-{doc_id}.json          ← {emails: 2, phones: 0, ips: 1, removed_spans: [[44,63],...]}
attributes/classifier/doc-{doc_id}.json   ← {fineweb_edu: 4, toxic: 0.02}
attributes/dedup/doc-{doc_id}.json        ← {minhash_sig: [...], cluster_id: 17234}
```

Attribute file은 append-only다. 새 filter는 corpus를 rewrite하지 않는다. 새 attribute directory를 emit한다. Training mix를 만드는 것은 그다음 *query*다.

```python
docs = filter(
    lambda d: d.lang == "en"
          and d.quality.gopher_quality > 0.7
          and d.classifier.fineweb_edu >= 3
          and d.dedup.cluster_rank == 0
          and d.pii.removed_spans  # PII was scrubbed, not deleted
    , all_docs)
```

Query가 *mix specification*이다. 몇 달 뒤 mix를 재현한다 = 같은 attribute file에 대해 query를 다시 실행한다. Mix version = `hash(query_source + attribute_file_hashes)`.

**Cross-stage doc-id tracking.** Doc-id는 foreign key다. FineWeb([[fineweb]])이 per-snapshot MinHash를 실행하고 `cluster_id` attribute를 만들 때, 그 cluster는 original CC dump의 doc-id를 reference한다. Llama 3([[llama-3]])가 pretrain document에 condition된 rejection-sampled SFT response를 생성할 때도 SFT row는 여전히 `source_doc_id`를 가진다. 이것이 ["reproducibility graph"]다. 어떤 SFT sample의 어떤 token이든 그 pretrain origin까지 추적할 수 있다. [[olmo-3]] model-flow는 여섯 stage lineage graph다. `raw → Dolma 3 → Dolma 3 Mix → Dolmino → Longmino → Dolci`. 각 arrow는 filter이고, 각 filter는 attribute를 emit하며, graph가 corpus다.

**몇 달 뒤 mix 재현하기.** 살아남아야 하는 세 가지:

1. **Raw shards**(content-addressed). 비용: frontier scale에서 raw는 100+ TB. 싼 object storage.
2. **Attribute files.** 비용: filter당 million docs당 몇 MB. 무시 가능.
3. **Mix query**(training config와 함께 git에서 versioned). 비용: kilobytes.

거부해야 할 것: filtered, tokenized shard를 "corpus"로 저장하는 것. 그것은 *output*이지 *source*가 아니다. 다른 subset을 다시 도출하는 능력을 잃는다. [[fineweb]]은 의도적으로 둘 다 shipping한다(raw parquet for re-query + classifier score as separate column). [[dolma]]도 `.attributes.jsonl` emission으로 같은 일을 한다.

---

## 4. Operation으로서의 code-repo filtering과 PII removal

**Code repo는 다른 생물이다.** Code는 licensed되어 있고, license는 file별이 아니라 repo별이다. [[llama-3]]의 pretrain은 많은 repo의 code를 포함한다. Pipeline은 다음을 해야 한다.

1. **Repo-level license check.** 모든 file에 대해 source repo의 license를 조회한다. Permissive(MIT, Apache, BSD) → include. Restrictive(GPL, AGPL) → exclude(대부분의 lab은 copyleft contamination을 피하려고 exclude한다). Unknown → exclude. The Stack v2(DeepSeek-Coder, OLMo 3가 사용)는 GitHub API scrape에서 이 table을 만든다. Lookup이 느린 step이다.
2. **Line-length heuristic.** Machine-generated minified JS(한 line, 100K chars), 긴 base64 blob, SQL dump는 모두 code처럼 보이지만 learning signal이 없다. Filter: `max_line_length > 1000` 또는 `mean_line_length > 100`인 file을 drop한다([[dolma]]의 heuristic; 대부분의 pipeline이 이제 비슷한 threshold를 사용).
3. **Executability screen.** Code가 parse되는가? File 자체의 doctest가 실행되는가? Tree-sitter parse-check는 syntactically broken file을 filter한다. Llama 3의 code pipeline([[llama-3]])은 더 나아간다. "Code-exec-filtered code"는 file이 sandbox에서 실행되어 non-error output을 냈다는 뜻이다. Filter는 비싸지만(10K containers/s에서 file당 ~10s), syntactically-valid지만 semantically-broken인 code 약 5%를 잡는다.
4. **Secret detection.** API key, AWS credential, private SSH key. TruffleHog / detect-secrets regex suite. Match가 있는 file은 *file 전체를 drop*하거나 선택적으로 key를 scrub하고 나머지를 keep한다. [[dolma]] pipeline은 document를 drop한다. 더 공격적인 접근은 scrub하고 audit하는 것이다.

**PII removal은 policy가 아니라 operation이다.** Policy question("phone number를 제거해야 하는가?")은 pipeline upstream에서 해결된다. Operational question은 policy가 주어졌을 때 *15T token 전체에서 어떻게 신뢰성 있게 실행하는가*다.

[[dolma]]의 PII filter는 three-tier cascade다.

| Tier | Method | Precision | Recall | Latency |
|---|---|---|---|---|
| 1 | Regex (email, phone, IP) | standard format에서 ~99% | ~80%(obfuscated `john [at] gmail [dot] com` 누락) | ~1 µs/doc |
| 2 | Named-Entity-Recognition classifier (spaCy / fastText) | ~90% | ~92% | ~5 ms/doc |
| 3 | Flagged document에만 LLM classifier(예: Llama Guard 3, [[llama-3]]) | ~95% | ~96% | ~100 ms/doc |

Tier 1만 모든 document에서 실행된다. Tier 2는 Tier 1을 unflagged로 통과했지만 high-risk domain(forums, social media)에서 온 document에 실행된다. Tier 3는 audit을 위해 약 1% sample에 실행된다. 15T token(~3B documents)에서 Tier 3를 global하게 실행하는 것은 prohibitively expensive(~10M CPU-hours)이기 때문에 cascade가 필요하다.

**대개 문서화되지 않는 세 가지 operational decision.**

1. **Scrub vs delete.** Scrub(span을 `[EMAIL]`로 replace)는 document를 보존한다. Delete는 document를 drop한다. [[dolma]]는 email/IP/phone을 scrub하고, FineWeb은 redact한다. Tradeoff: delete는 더 안전하다(부분적 PII가 살아남을 가능성이 없음). Scrub은 surrounding context를 보존한다. 대부분의 modern pipeline은 scrub한다.
2. **Opt-out registry.** GDPR과 유사 regime은 "내 data를 제거하라"는 요청을 존중할 능력을 요구한다. Operational implementation: registry에 유지되는 doc-id(또는 URL, content hash) list. 모든 pipeline run은 current registry에 대해 filter한다. Frontier scale의 registry size: ~10K entries, 증가 중. 비용은 모든 shard read에서의 lookup이다. 40 GB/s에서는 database query가 아니라 bloom filter여야 한다.
3. **Removal event audit log.** 모든 `(doc_id, removed_span, reason, timestamp, pipeline_version)` tuple은 log shard에 append된다. Log cardinality는 pretrain run당 ~10M row다. Log가 *legal artifact*다. Regulator가 어떤 PII가 제거되었는지 묻는다면 log가 답이다. [[dolma]]의 attribute-file design은 이를 자연스럽게 만든다. `attributes/pii/` *자체가* audit log다.

---

## 5. Tokenizer extension pitfall — post-pretrain `<think>` trap

모든 modern chat/reasoning model은 pretraining *이후* special token을 추가한다. [[phi-4]]는 reasoning trace를 감싸는 `<think>` block을 갖고, Qwen 3([[qwen-3]])는 "thinking budget"을 노출하며, OLMo 2 Instruct([[olmo-2]])는 Tulu 3의 `<|im_start|>` / `<|im_end|>` chat template을 재사용한다. 이 token들은 pretrain tokenizer에 존재하지 않았다. 주의 없이 추가하는 것은 가장 흔한 SFT regression 중 하나다.

**문제.** Tokenizer를 `V`에서 `V + k` token으로 확장하면 input embedding matrix `E_in ∈ R^{V × d}`와 output projection `E_out ∈ R^{V × d}`를 `(V + k) × d`로 확장해야 한다. 새 row는 *uninitialized*다. 일반적인 library default는 이를 `N(0, 0.02)`로 초기화한다. Fresh embedding을 위해 pretrain에서 쓰는 것과 같은 scheme이다.

이는 거의 항상 틀렸다. Pretrain init에서는 *모든* embedding이 `N(0, 0.02)`에서 뽑히고, model은 역시 작은 값으로 초기화된 LayerNorm, bias, attention weight와 함께 co-train된다. 모든 것이 small-magnitude이기 때문에 model은 small-magnitude embedding을 다루는 법을 배운다. 15T token pretraining 이후 기존 token의 learned embedding은 일반적으로 L2 norm ~1.0-1.5에 settling되어 있다. 새 `N(0, 0.02)` embedding, 즉 L2 norm ~0.04를 주입하면 새 token의 embedding은 neighbor보다 30배 작다. `<think>`가 처음 model에 들어갈 때 attention은 사실상 그것을 무시한다. Query / key projection이 near-zero logit을 만들기 때문에 그 token을 향하거나 `<think>`에서 오는 post-softmax attention weight는 uniform distribution minus epsilon으로 collapse한다. Chat template은 첫 ~1000 SFT step 동안 special token이 없는 것처럼 조용히 행동한다. 그 뒤 embedding이 마침내 커지고 model이 `<think>`에 attend하기 시작하지만, 그 시점에는 *다른* weight들이 잘못된 가정 아래 부분적으로 update되어 reasoning benchmark에서 측정 가능한 downstream regression을 만든다.

**Recipe — mean-of-neighbors initialization.** 수정법은 [[olmo-2]]의 Tulu 3 application과 [[phi-4]] report의 "no chat-template regression" claim이 암묵적으로 인용하는 것이다.

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

새 token의 embedding은 이제 대략 올바른 L2 norm을 갖는다. 이미 올바른 norm을 가진 vector들의 평균이기 때문이다. Semantic content도 합리적인 prior다. `<think>`를 `think`, `thought`, `reason`의 평균에서 초기화하면 SFT 시작 전부터 이미 "올바른 neighborhood"에 있다. 그러면 SFT step은 구덩이에서 기어 올라오는 대신 좋은 initial state를 refine한다.

**Tokenizer-freezing discipline.** §1은 "200-500 unused token을 미리 reserve하라"고 말했다. 이것이 이유다. Future SFT phase가 `<think>`, `<tool_call>`, `<search>`, `<code>` 등을 추가할 것을 미리 안다면 pretrain time에 id를 reserve하라. Embedding slot은 pretrain에서 보이지 않으므로 near-zero mean 정도로 학습되지만, SFT time에는 matrix를 resize하지 않고 `<|reserved_7|>` slot을 `<think>`로 *rename*한다. New row가 없고, initialization question이 없으며, drift도 없다. Llama 3([[llama-3]])가 128K vocab의 256 reserved slot으로 하는 일이 이것이다.

**Diagnosis — drift가 발생할 때 감지하는 방법.** 값싼 log 두 가지:

- `||E_in[new_id]||` vs `median(||E_in[existing_id]||)`. SFT step 어디에서든 ratio < 0.3이면 alarm.
- Validation batch에 대해 layer별로 평균한, new token position에 할당된 attention probability mass. `1/seq_len`보다 유의하게 낮으면 alarm.

두 signal 모두 drift가 일어나는 그대로 보여 주며, downstream eval 전에 silent regression을 잡을 수 있다.

---

## 6. 하나의 graph로서의 full pipeline

§§1-5를 operator의 mental model로 결합하면 다음과 같다. Interactive version은 `figures/pipeline-ops.html`을 보라.

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

모든 edge는 attribute graph에 append되는 `(doc_id, attribute) → (doc_id, attribute')` transformation이다. 모든 edge에는 failure mode가 있다.

- raw → filter1: **new crawl date, different snapshot; content가 조용히 바뀌면 doc-id collision**.
- filter1 → PII: **obfuscated span에서 regex false-negative; audit log entry missing → regulator-facing risk**.
- PII → dedup: **PII 제거가 이전에는 없던 near-duplicate를 만든다. Dedup은 PII 전이 아니라 후에 실행되어야 한다**.
- dedup → tokenize: **pretrain과 SFT 사이 tokenizer version mismatch; BLAKE3 check가 이를 잡는다**.
- tokenize → shuffle: **non-deterministic shuffle; resume desync가 돌아옴**(ch-06 §5.1).
- shuffle → pack: **shard size imbalance; straggler worker가 step을 block**.
- pack → trainer: **dataloader state-dict에 `mix_pointer` 누락; §3 mix version lost**.

Interactive figure는 각 stage를 따라가며 15T-token [[fineweb]] / 14.8T-token [[deepseek-v3]] scale의 typical throughput을 보여 주고, 각 handoff에 반드시 존재해야 하는 lineage attribute를 나열한다.

---

## 연결과 다음 내용

- **[[dolma]] / [[fineweb]] / [[ccnet]]** — 이 장이 operationalize하는 세 pipeline. 앞 장들은 모두 이것들을 *design*으로 다루지만, 여기서는 *running system*으로 다룬다.
- **[[llama-3]] / [[olmo-2]] / [[olmo-3]] / [[qwen-3]] / [[deepseek-v3]] / [[phi-4]]** — 여섯 production tokenizer decision, 여섯 shard-layout choice, 여섯 lineage convention.
- **[[physics-of-lm-3]]** — small-model vocab-size 재고를 강제하는 capacity argument.
- **ch-06 (checkpointing)** — data-iterator state는 checkpoint에 산다. 나쁜 shard layout은 resume을 조용히 깨뜨린다.
- **ch-10 (curation pipelines)** — design. 이 장은 operation.
- **ch-12 (dedup)** — tokenize 이후 다음 stage. MinHash는 `doc_id → signature` lineage edge 위에 산다.
- **ch-17 (lab)** — minimal CCNet-style pipeline을 end-to-end로 구현할 것이다. 여기의 lineage와 tokenizer discipline이 interface다.

## 더 읽을거리

- Sennrich, Haddow & Birch 2015 — [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) — canonical BPE.
- Kudo & Richardson 2018 — [arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226) — SentencePiece; byte fallback.
- [[dolma]] — paper §3은 attribute-file design, §4는 PII cascade.
- [[fineweb]] — `datatrove` codebase; per-dump MinHash ablation은 shard-aware dedup에 대한 canonical reference.
- [[ccnet]] — scalability argument로서의 hash-based lineage.
- [[llama-3]] — 128K tokenizer, reserved-slot discipline, code-exec filter.
- [[olmo-3]] — model-flow worldview, OLMES + OlmoTrace + decontam tooling.
- [[qwen-3]] — 36T token, 119개 language에서 multilingual tokenizer sizing.
- [[deepseek-v3]] — Chinese-English tokenizer, FP8 training → shard throughput이 결정적.
- [[phi-4]] — textbook synthetic data, reasoning trace를 위한 special-token extension.

## 동반 시각화

**[figures/pipeline-ops.html](figures/pipeline-ops.html)** — interactive production-pipeline diagram. 어떤 stage(raw → filter1 → filter2 → dedup → tokenize → shuffle → pack → train)든 클릭하면 (a) 그 edge에서 emit되는 lineage attribute, (b) 15T-token pretrain scale의 typical throughput, (c) 해당 stage의 흔한 operational failure mode와 관측 가능한 symptom을 볼 수 있다. Lineage graph가 어떻게 축적되는지 internalize하는 데 사용하라. 모든 stage는 attribute를 append하고, 어느 stage도 doc-id를 rewrite하지 않으며, final trainer는 text의 filesystem이 아니라 compact `(doc_id → shard_offset)` map을 소비한다.
