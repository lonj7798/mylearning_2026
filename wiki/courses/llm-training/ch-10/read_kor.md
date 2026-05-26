<!-- chapter: ch-10
     track: data
     title: Open Curation Pipelines — CCNet, C4, Dolma, FineWeb
     kind: content
     deps: [ch-09]
     sources: [[ccnet]], [[c4]], [[dolma]], [[fineweb]], [[scaling-laws-data-quality]], [[rephrasing-the-web]], [[deduplicating-training-data]], [[minhash-lsh]]
     figures: figures/pipeline-compare.html
-->

# 10장 — Open Curation Pipeline: CCNet, C4, Dolma, FineWeb

> **핵심 통찰.** 네 개의 pipeline, 다섯 해, 하나의 개념적 골격. CCNet(2019)은 순서, 즉 **language ID → dedup → quality**를 고정했고, 이후 모든 open pipeline은 세 번째 step에 *어떤* quality signal이 들어가야 하는지에 대한 논쟁이었다. C4는 heuristic으로 답했고, Dolma는 heuristic + transparency + ablation으로 답했으며, FineWeb은 LLM-labeled classifier로 답했다. 2026년에 새 pipeline을 읽는 올바른 방법은 2019년에 읽던 방법과 같다. **어떤 filter를, 어떤 순서로, 어떤 threshold로, 어떻게 ablate했는가.**
>
> **가이드라인.** Web pipeline을 audit하거나 설계할 때는 (1) *quality signal*(KenLM perplexity / heuristic stack / classifier)을 식별하고, (2) dedup이 quality에 비해 어디에 있는지 찾고(CCNet은 앞에, Dolma는 둘레에, FineWeb은 extraction 뒤 classifier 앞에 둔다), (3) 고정된 model size에서 고정된 downstream eval에 대해 *하나의* stage만 움직이는 ablation table을 요구하고, (4) 그런 table이 없는 pipeline은 반증 불가능한 것으로 다루라.

---

## 이 장이 존재하는 이유

[[ch-09]]는 pretraining dataset의 *landscape*를 그렸다. 무엇이 존재하는지, 각 corpus가 무엇을 빠뜨리는지, 어떤 lab이 mix 공개를 멈췄는지 다뤘다. 이 장은 2019년부터 2026년까지 open pretraining을 형성한 네 pipeline, 즉 **[[ccnet]]**(template), **[[c4]]**(모두가 복사하는 heuristic baseline), **[[dolma]]**(transparency benchmark), **[[fineweb]]**(classifier-era replacement)로 한 층 더 내려간다. 끝나면 새 data-release paper를 열고 pipeline diagram을 훑은 뒤, 어떤 filter가 load-bearing인지, 어떤 것이 장식인지, 어떤 threshold에 이의를 제기할지 식별할 수 있어야 한다.

세 가지 역사적 흐름이 중요하다. 첫째, 고전적 signal은 **신뢰된 anchor corpus에 대한 perplexity**였다. CCNet은 Wikipedia로 KenLM을 학습하고 crawl을 그것으로 score한다. 둘째, heuristic 흐름, 즉 C4의 terminal-punctuation과 bad-word rule은 semantic targeting보다 speed와 legibility를 선택했다. 셋째, classifier 흐름, 즉 FineWeb-Edu의 Llama-3-labeled educational-value score는 작은 supervised classifier가 modern scale에서 손으로 만든 rule stack을 이긴다고 베팅했다. [[scaling-laws-data-quality]]는 이론적 짝을 제공한다. Quality는 scaling variable이므로 token count가 같은 두 corpus도 서로 다른 loss curve 위에 있을 수 있다. 여기서 공부하는 pipeline들은 그 주장의 empirical manifestation이다.

네 번째 흐름은 네 논문 모두에서 대부분 암묵적인 **ordering**이다. 같은 여섯 filter도 다른 순서로 적용하면 실질적으로 다른 corpus를 만든다. 각 filter는 이전 filter를 통과해 살아남은 distribution 위에서 작동하기 때문이다. CCNet → Dolma → FineWeb 진화는 무엇을 filter할지뿐 아니라 cascade의 *언제* filter할지에 대한 것이기도 하다. 이 pipeline들을 rule의 bag으로 읽어서는 안 된다. pipeline으로 읽어야 한다.

Companion interactive [figures/pipeline-compare.html](figures/pipeline-compare.html)은 네 pipeline을 click-to-inspect stage와 함께 나란히 배치한다. 아래 §1-§5를 읽는 동안 running reference로 사용하라.

---

## 1. CCNet — template (2019)

[[ccnet]]은 이후 pipeline들이 변형하는 개념적 골격이다. 세 stage가 있고, 각각 한 가지 일을 한다.

1. **Paragraph-level exact dedup, ~5 GB shard별.** Normalized paragraph를 hash한다(lowercase, digits → 0, punctuation stripped). ~1M doc의 shard 안에서 duplicate를 drop한다. *먼저* 수행한다. 싸고, language classifier를 bias할 boilerplate를 제거하기 때문이다.
2. **fastText language ID.** 각 document에 `lid.176`을 실행한다. score ≥ 0.5인 top-predicted language를 keep하고, document를 language별 downstream processing으로 보낸다. CCNet은 design상 multilingual이다. 이후 English-only recipe(C4, FineWeb)가 달라지는 step이 여기다.
3. **Wikipedia 대비 KenLM perplexity.** 각 language에 대해 해당 language의 Wikipedia로 5-gram KenLM을 학습한다. 각 document의 perplexity를 score한다. Perplexity percentile에 따라 **head / middle / tail**로 partition한다. *Drop하지 말고 label하라.* CCNet은 세 partition을 모두 내보내고 downstream user가 선택하게 한다.

이것이 template이 된 이유는 세 stage가 직교적이기 때문이다. 각각은 서로 다른 failure mode, 즉 duplicate-boilerplate noise, language confusion, off-distribution text를 고치며, 각각 *cheap* signal(hash, tiny classifier, n-gram LM)을 사용한다. Recipe는 CPU node 몇 개에서 돌아가고 사용할 수 있는 multilingual pretraining data를 만든다. 이후 모든 open pipeline은 이 skeleton을 유지하고 step three를 두고 논쟁했다.

**왜 Wikipedia를 anchor로 쓰는가?** 2019년에 사용 가능했던 유일한 multilingual corpus로서 (a) 사람이 정리했고, (b) 5-gram LM을 학습할 만큼 충분히 크며(언어당 최소 수백만 token), (c) pretraining language model이 대체로 모방하기 원하는 "encyclopedic explanation" register와 stylistically aligned되어 있었기 때문이다. 이 선택은 중립적이지 않다. Wikipedia는 자체 style bias(formal register, Western-encyclopedic framing, 특정 topic distribution)를 가지며 CCNet의 "head" partition은 그것을 상속한다. 하지만 2019년의 대안은 anchor가 전혀 없는 것이었고, 그 비교는 박빙이 아니다.

Dedup-before-langID ordering은 의도적이며 때때로 잘못 복사된다. [[dolma]]는 재정렬한다(dedup을 langID 앞이 아니라 quality 둘레에 둔다). Dolma의 dedup은 document-level이고, 비영어 document를 먼저 버리고 싶을 만큼 비싸기 때문이다. [[fineweb]]은 CCNet-style structural-first ordering(URL filter → extraction → langID → quality → dedup)으로 돌아가고, ablation을 통해 이것이 15T-token scale에서 옳다고 주장한다.

**"Head partition" convention.** 새 독자가 놓치는 미묘한 점이 있다. CCNet 자체는 "high quality" corpus를 ship하지 않는다. *Perplexity-partitioned* corpus를 ship한다. Head partition은 가장 낮은 perplexity third(가장 Wikipedia-like)이고, tail은 가장 높은 perplexity third(가장 colloquial / noisy)다. Downstream user들, 즉 RedPajama-V1, Llama-1의 training mix, 초기 open release들은 head partition을 "CCNet-filtered Common Crawl"로 골랐고 middle과 tail은 버렸다. 이것은 *use-site* decision이지 pipeline decision이 아니다. CCNet의 effective quality-filtering power 대부분이 실제로 여기에 있다. CCNet을 채택한다면 사실상 CCNet-head를 채택하는 것이다.

---

## 2. C4 — heuristic baseline (2019)

[[c4]]는 T5를 위해 만들어졌고, flat하고 reproducible한 English crawl이면 경쟁력 있는 model을 학습하기에 충분하다는 생각을 대중화했다. Pipeline은 하나의 Common Crawl snapshot(April 2019 WET)에 적용되는 약 8개의 rule sequence다.

- **Terminal-punctuation rule** — `.` `!` `?` `"`로 끝나는 line만 keep한다. Raw line의 약 50%를 drop한다. C4에서 가장 공격적인 단일 filter이자 가장 둔한 filter다. Menu item과 함께 code, bullet, title도 삭제한다.
- **Line length ≥ 5 words.** Navigation fragment와 caption을 target한다.
- **Bad-word blocklist** — "List of Dirty, Naughty, Obscene and Otherwise Bad Words"의 약 400개 word 중 하나라도 포함한 document를 drop한다. Line-level이 아니라 document-level이다.
- **JavaScript-line strip** — "javascript"라는 단어(case-insensitive)를 포함한 line을 제거한다. "Please enable JavaScript" boilerplate를 target한다.
- **Curly-brace strip** 및 **"lorem ipsum" strip** — template placeholder를 포함한 page를 삭제한다.
- **langdetect → English with P ≥ 0.99.** 엄격한 threshold다. C4는 명시적으로 English-only다.
- **Three-sentence line dedup** — 모든 contiguous 3-sentence span을 emit하고, 각 unique span의 copy 하나만 global하게 keep한다. Document-level dedup이 놓치는 templated boilerplate의 long tail을 잡는다.

C4는 web pretraining을 reproducible하게 만들었다(snapshot을 고정하고 rule을 돌리면 같은 corpus가 나온다). 그리고 빠르게 실행됐다. 또한 이후 audit(Dodge et al. 2021, [[c4]]에서 인용)이 기록한 bias를 내장했다. Bad-word blocklist는 LGBTQ identity, medical content, 일부 English dialect에 관한 document를 불균형하게 제거한다. P(English) ≥ 0.99 threshold는 African-American English와 code-switched prose를 떨어뜨린다. 이것들은 blocklist나 classifier의 버그가 아니다. *Feedback 없이 document-level heuristic filtering을 하는 피할 수 없는 비용*이다. C4에는 ablation table이 없다. 각 rule을 개별적으로 toggle해 T5를 다시 실행한 사람은 없다.

**C4를 recipe가 아니라 reference point로 읽어라.** 그 heuristic들은 이후 거의 모든 pipeline에 거의 그대로 다시 나타난다(Gopher가 빌렸고, Dolma가 쌓았으며, FineWeb이 classifier 전에 적용한다). Ablation의 부재는 Dolma가 나중에 메운 gap이다.

**Terminal-punctuation rule에 대한 한 가지 관찰.** 이것은 open-pretraining 문헌에서 가장 consequential한 heuristic이다. *암묵적으로* 복사되기 때문이다. Gopher-style "fraction of lines ending in terminal punctuation > 0.6" threshold를 C4의 rule에서 tuned한 모든 pipeline은 C4의 blind spot을 물려받는다. Line이 *structurally valid*하지만 prose-terminated가 아닌 document(bullet-heavy technical docs, code-heavy explainer, poetry, song lyric)는 지속적으로 덜 keep된다. Downstream model이 poetry나 markdown-heavy doc에 놀란다면 blame chain은 여기서 끝난다.

---

## 3. Dolma — transparency benchmark (2024)

[[dolma]]의 기여는 새로운 filter가 아니다. **Stage마다 ablation row가 있는 완전히 문서화된 six-stage cascade**다. AI2는 모든 config, 모든 threshold, pipeline을 JSONL shard 위의 streaming pass로 실행하는 toolkit(`dolma` CLI)을 공개했다. 각 filter는 document별 *attribute file*을 score와 함께 emit한다. 최종 keep/drop decision은 attribute file 위의 값싼 second pass다. 이 decoupling 자체가 design contribution이다. 다른 threshold로 pipeline을 다시 실행하는 비용을 거의 0으로 만든다.

Cascade(web lane):

1. **URL + Bloom dedup** across CC snapshot(FP ≤ 1e-6). Re-crawl이 CC volume을 지배한다. 먼저 URL dedup을 하면 가장 쉬운 near-duplicate를 싸게 제거한다.
2. **Document-level near dedup** via normalized text의 Bloom filter. 같은 content가 다른 URL에서 serving되는 경우다.
3. **fastText language ID → English.**
4. **Quality filter stack** adapted from Gopher + C4: symbol-to-word ratio < 0.1, mean line length > 5, fraction-of-lines-with-terminal-punctuation > 0.6, stopword ratio > 0.06, duplicate-line fraction < 0.3.
5. **Content filter** — Jigsaw Toxic Comments로 학습한 fastText toxicity/NSFW classifier. Threshold-drop.
6. **PII filter** — 세 category(email, IP, phone number)에 대해서만 high-precision regex. PII-density threshold를 넘는 document는 drop한다(in-place redaction이 아님).
7. **Paragraph-level dedup last** — Bloom exact-match. 대안인 `dolma-ngram`은 duplicated n-gram의 fraction이 threshold T(default T = 1.0)를 넘으면 paragraph duplicate로 mark한다.

**왜 paragraph dedup이 마지막인가.** 앞 stage들은 어떤 paragraph가 살아남는지를 바꾼다. Dedup은 raw distribution이 아니라 surviving distribution 위에서 작동해야 한다. Dolma paper의 ablation(arXiv version Table 5)은 stage-off corpus에서 150B token으로 학습한 고정 1B-parameter OLMo-style model로 각 stage를 toggle한다. Full-pipeline row가 downstream task average의 최상단에 있다. Document-dedup 제거가 가장 큰 비용을 낳고, paragraph-dedup 제거는 작지만 0이 아닌 비용을 낳으며, quality stack 제거는 full-pipeline delta의 대략 절반을 잃는다. Ablation이 pipeline의 나머지를 *falsifiable*하게 만든다.

**Dolma ordering의 두 가지 미묘한 점.** 첫째, document dedup *이후* language ID는 CCNet의 "dedup after langID"와 반대다. Bloom-based doc dedup이 full multilingual crawl에서 실행할 만큼 싸고, dedup 이후 non-English document를 drop해도 duplicate가 다시 도입되지 않기 때문에 정당화된다. 둘째, content filtering(toxicity, NSFW)은 PII stage *전에* 실행된다. 그렇지 않으면 NSFW classifier가 PII가 이미 context 밖으로 shift된 document를 score하게 된다. Content classifier를 raw text 가까이에 두면 calibration이 보존된다.

**Per-source lane.** 위 web lane은 Dolma pipeline 중 하나일 뿐이다. `peS2o`(scientific papers)는 다른 quality stack을 사용한다. Publication structure를 신뢰하므로 terminal-punctuation rule은 section-header와 LaTeX-noise heuristic으로 대체된다. `The Stack`(code)은 paragraph hash가 아니라 code token에 대한 MinHash near-dedup을 사용한다. Code reuse pattern은 long-range(여러 repository에 걸친 동일 file의 긴 verbatim copy)이기 때문이다. Social-media lane(Reddit)은 text classifier가 아니라 subreddit-level quality list로 filter한다. Books(Project Gutenberg)는 source가 이미 curated되어 있으므로 거의 filter하지 않는다. **올바른 filter는 source distribution에 달려 있다.** Dolma는 이를 명시한 첫 open pipeline이다.

**Design artifact로서의 `dolma` toolkit.** Toolkit은 YAML config를 받고, JSONL shard 위의 streaming pass로 filter를 실행하며, document별 attribute file(filter당 하나의 score)을 emit해서 최종 keep/drop decision을 별도의 값싼 pass로 만든다. 이 two-pass architecture, 즉 *모든 것을 score하고 나중에 결정하기*는 진짜 engineering contribution이다. 비싼 filter pass를 다시 실행하지 않고 threshold를 다시 sweep할 수 있다. Ablation도 싸게 만든다. Ablation table은 pipeline 재실행이 아니라 attribute file 위의 script다.

Filter × model-size × eval number가 있는 full ablation table은 [[excerpts/dolma]]를 보라.

---

## 4. FineWeb과 FineWeb-Edu — classifier era (2024)

[[fineweb]]은 heuristic-stacking question을 supervised-learning question으로 바꾼다. *LLM-labeled data로 학습한 작은 classifier가 어떤 regex stack도 할 수 없는 일을 할 수 있는가?* HF의 15T-token corpus가 낸 답은 yes다. Knowledge-and-reasoning benchmark에서 큰 차이로 그렇다.

**FineWeb base pipeline.**

1. WARC file에 대한 **URL blocklist filter**.
2. HTML-to-text extraction에 **Trafilatura** 사용. CCNet이 쓰던 WET/boilerpipe extractor를 대체한다. Trafilatura는 paragraph structure를 보존하고 chrome을 더 공격적으로 제거한다. [[fineweb]]의 ablation: WET을 Trafilatura로 바꾸면 *semantic filter가 실행되기 전*에도 측정 가능한 quality bump가 생긴다. 이것은 paper에서 가장 덜 인용되는 교훈이다. Extractor choice는 infrastructure가 아니라 filter다.
3. **fastText language ID → English**, P ≥ 0.65(C4의 0.99보다 느슨함. FineWeb은 나중에 quality로 조인다).
4. **Gopher + C4 heuristic cascade**, Dolma의 stage 4와 동일.
5. **Snapshot별 MinHash dedup**, global이 아님. 5-gram에 대한 MinHash-LSH, Jaccard threshold 0.8, 96개 CC snapshot 각각 안에서 적용.
6. **PII redaction**(email, phone) — document drop이 아니라 in-place replace.

**Per-snapshot MinHash** finding은 [[fineweb]]의 진짜 새로운 empirical result다. Standard [[deduplicating-training-data]] recipe인 global dedup을 시험했더니 downstream eval을 해치는 것으로 나타났다. Penedo et al.은 global dedup된 "evergreen" page, 즉 high-quality reference material이기 때문에 여러 snapshot에 나타나는 document야말로 여러 번 등장하는 것이 training signal을 운반한다고 주장한다. Snapshot 안에서만 dedup하면 그 signal을 보존하고, snapshot 사이 dedup은 그것을 파괴한다.

**FineWeb-Edu: educational-value classifier.**

- **Label source.** Llama-3-70B-Instruct가 450K web sample을 0-5 integer scale(0 = not educational, 5 = highly educational)로 annotate한다.
- **Classifier.** Frozen embedding model 위의 작은 classification head(public release: `HuggingFaceFW/fineweb-edu-classifier`).
- **Hold-out performance.** 46,867-sample Llama-3-annotated hold-out에서 score-≥-3 binary classification은 F1 = 82%에 도달한다.
- **Threshold.** Score ≥ 3이 shipped default다. 이것은 **FineWeb의 약 92%를 버리고**, 15T에서 1.3T token만 남긴다.
- **왜 3이고 4가 아닌가?** Threshold sweep은 threshold 4가 MMLU를 얻지만 HellaSwag을 잃는다는 것을 보인다. Threshold 3은 MMLU/ARC와 HellaSwag trade-off의 sweet spot이다.

MMLU와 ARC에서 classifier-vs-heuristic margin이 FineWeb-Edu의 headline이다. Full threshold × token-count × benchmark table은 [[excerpts/fineweb]]을 보라.

**왜 classifier가 2024+ scale에서 heuristic을 이기는가.** Heuristic은 punctuation, ratio, line length 같은 surface feature를 encode한다. LLM-labeled educational-value data로 학습한 classifier는 어떤 regex도 표현할 수 없는 semantic judgment를 encode한다. *이 document가 textbook, lecture, 잘 쓴 explainer의 느낌을 갖는가?* FineWeb paper는 Gopher + C4 cascade 이후 추가 heuristic을 쌓아도 MMLU에서 plateau한다고 보고한다. Rule을 더 추가해도 도움이 멈춘다. Classifier는 plateau를 깨는 첫 signal이다. 이것이 [[scaling-laws-data-quality]]의 quality-as-a-scaling-variable argument가 가진 empirical content다. Heuristic은 어떤 scale에서 token당 information을 다 써버리고, 계속 올라가려면 더 풍부한 signal이 필요하다.

**FineWeb-Edu를 복제하는 실제 비용.** Llama-3-70B-Instruct로 450K document에 label을 붙이는 것은 공짜가 아니다. Document당 ~1000 token이고 Llama-3의 2024 API rate를 기준으로 annotation pass는 대략 low-four-figure dollar cost와 cluster에서 multi-day wall time이 든다. Frozen embedding 위 classifier head 학습은 빠르다(single GPU에서 몇 시간). 15T → 1.3T filtering pass는 FineWeb 위의 streaming read 한 번이다. *비용 대부분은 anchor-LLM annotation pass에 있다.* 그래서 FineWeb-Edu는 specialization하기 쉽다. *다른* prompt("math reasoning value로 평가" / "code explanation quality로 평가")로 450K document를 annotate하면 같은 marginal cost로 task-specialized Edu-style corpus를 얻는다. HF의 Nemotron-CC와 이후 task-specific corpus들은 정확히 이 recipe를 따른다.

**DCLM-baseline comparison.** DCLM(DataComp-LM, Li et al. 2024, 이 장의 source list에는 없지만 언급할 가치가 있음)은 같은 DCLM benchmark harness에서 C4-style heuristic, CCNet-style perplexity, Dolma stack, FineWeb-Edu-style classifier를 head-to-head로 실행한다. DCLM leaderboard는 matched eval에서 이 pipeline들의 가장 명확한 public ranking이다. Classifier-based pipeline은 상단을 지배하고, heuristic-only pipeline은 중간에 모이며, raw Common Crawl은 바닥에 있다. 어떤 pipeline을 빌릴지 의심될 때 DCLM leaderboard가 현재의 empirical answer다. 이 장의 paper들은 ranking이 *왜* 그렇게 나오는지 이해하게 해 준다.

---

## 5. Filter별 비교

이것이 외울 table이다. Row는 filter *family*이고, column은 네 pipeline이다. Entry는 concrete signal + threshold + (보고된 경우) ablation delta다.

| Filter family | CCNet (2019) | C4 (2019) | Dolma web (2024) | FineWeb / -Edu (2024) |
|---|---|---|---|---|
| Source snapshots | Multiple CC WET, multilingual | 1 CC snapshot (Apr 2019) WET | Multiple CC, English slice | 96 CC snapshots, English |
| HTML→text | WET (boilerpipe) | WET (boilerpipe) | WET + per-source variants | **Trafilatura** (paper: WET 대비 측정 가능한 quality gain) |
| URL filter | — | — | URL + Bloom dedup (FP ≤ 1e-6) | URL blocklist + URL dedup |
| Language ID | fastText lid.176, score ≥ 0.5, multilingual | langdetect, P(English) ≥ 0.99 | fastText, English-only | fastText, P(English) ≥ 0.65 |
| Heuristic quality | — (perplexity 사용) | Terminal-punct + ≥5-word lines + JS/curly/lorem strip | Gopher + C4 stack: symbol-ratio < 0.1, mean-line > 5, term-punct > 0.6, stopword > 0.06, dup-line < 0.3 | Dolma와 같은 Gopher + C4 stack |
| Semantic quality | **KenLM perplexity vs Wikipedia**; head/mid/tail partition | — | — (heuristic only) | **FineWeb-Edu classifier, score ≥ 3** (Llama-3-labeled, F1 = 82%) |
| Blocklist / content | — | ~400-word bad-word list, doc-level drop | fastText toxicity/NSFW (Jigsaw) | stage 1의 URL blocklist only |
| PII | — | — | Regex: email/IP/phone, density 기준 doc-drop | Regex: email/phone, in-place redact |
| Exact dedup | Paragraph, shard별, langID 전 | 3-sentence line span, global | Doc-level Bloom + 마지막 paragraph Bloom | — |
| Near dedup | — | — | Optional `dolma-ngram` paragraph variant | **MinHash per snapshot** (global 아님; ablation: global hurts) |
| Ablation published? | No (partition 자체가 설명) | No | Yes — stage별 × downstream eval (Table 5) | Yes — dedup strategy + classifier threshold sweep |
| Output scale | ~2 T tokens (all languages, all partitions) | ~750 B tokens | ~3 T tokens | 15 T (base) → 1.3 T (Edu, score ≥ 3) |

두 row가 load-bearing이다. **Semantic-quality row**, 즉 각 pipeline의 "quality signal"이 사는 곳이 field가 실제로 움직인 축이다(perplexity → heuristics → classifier). **Ablation-published row**는 field가 *artisanal*이 아니라 *scientific*해진 축이다.

---

## 6. 새 pipeline을 읽고 비판하는 방법

새 data release(GPT-5-era open corpus, Qwen-3 data card, 어떤 lab의 다음 FineWeb clone)를 열면 이 네 질문 checklist를 실행하라.

1. **Quality signal은 무엇인가?** 어떤 anchor에 대한 perplexity인가? 어떤 heuristic과 어떤 threshold인가? 어떤 label로 학습한 classifier인가? Paper가 이를 한 paragraph로 답하지 못하면 pipeline은 reproducible하지 않다.
2. **Dedup과 langID에 비해 어떤 순서인가?** Pre-dedup quality는 duplicate를 score하느라 cycle을 낭비한다. Post-dedup quality는 classifier가 학습되지 않은 distribution을 score할 수 있다. 질문하라. 왜 *이* 순서인가?
3. **어떤 threshold이고, 어떻게 정당화되었는가?** Sweep 없는 threshold는 미신이다. FineWeb-Edu의 score ≥ 3은 MMLU + ARC + HellaSwag에 대한 0-5 sweep으로 방어된다. C4의 P(English) ≥ 0.99는 아무것으로도 방어되지 않는다. 이제 우리는 그것이 corpus에서 특정 English dialect를 비용으로 치렀다는 것을 안다.
4. **어떻게 ablate했는가?** 한 stage off, fixed model size, fixed downstream eval. Ablation이 없으면 pipeline을 scientific claim이 아니라 engineering artifact로 다루라. [[dolma]]가 현재 기준선이다. [[c4]]는 기준선 아래에 있으며, 가장 많이 복사된 recipe임에도 5년 동안 그래 왔다.

**시야에 둘 두 가지 직교 방향.** 첫째, [[scaling-laws-data-quality]]는 quality가 scaling variable이라고 주장한다. 같은 pipeline delta도 scale에 따라 다른 model-size jump의 가치가 있을 수 있다. 1B에서 도움이 되는 classifier가 70B에서 plateau할 수 있고, 그 반대도 가능하다. Model size와 downstream eval을 고정하지 않고 pipeline을 rank할 수 없다. 둘째, [[rephrasing-the-web]](WRAP)은 classifier-filtered web text조차 충분히 noisy해서 이를 더 깨끗한 style(Wikipedia-like, Q&A, terse)로 *rewriting*하면 추가 multiplicative gain이 생긴다고 주장한다. 즉 filtering에는 ceiling이 있고, synthetic rephrasing이 다음 move다. Deduplication([[ch-12]])과 synthetic data(Track 2) 장들이 각각의 thread를 이어받는다.

**새 pipeline에서 주의할 세 가지 failure mode.** (a) *잘못된 scale의 ablation* — 150M parameter에서 도움이 되는 filter가 7B에서 regress할 수 있다. Ablation model size와 token count를 요구하라. (b) *Eval leakage* — benchmark test set과 overlap되는 data로 학습한 classifier는 기적처럼 보인다. FineWeb-Edu는 MMLU/ARC/HellaSwag에 대해 명시적으로 decontaminate하지만, 많은 follow-up은 그렇지 않다. (c) *Benchmark에 filter를 overfit* — classifier의 training label이 "MMLU usefulness로 평가"라면 downstream MMLU win은 거의 tautological이다. 의미 있는 test는 classifier가 tuned되지 않은 held-out evaluation으로의 transfer다.

**Pipeline critique의 worked example.** 2026년 paper가 120개 CC snapshot에서 만든 20T-token web corpus "DeepWeb-Pure"를 발표하고 FineWeb-Edu 대비 +3.5 pp MMLU를 보고한다고 하자. Claim을 통과시킬 질문:

- Quality signal은 무엇인가? "GPT-5-labeled educational value로 학습한 classifier"라면 어떤 labelling prompt를 사용했는지, label이 public인지 묻는다. Unreleased label = unreproducible pipeline이다.
- Dedup은 어디인가? 120개 snapshot 전체에 global MinHash라면, [[fineweb]]의 per-dump finding은 gain을 classifier에 귀속하기 전에 per-snapshot MinHash로 재확인해야 함을 시사한다.
- 어떤 ablation인가? Table이 "full pipeline vs no pipeline"이면 +3.5 pp가 한 stage(extraction, dedup, classifier) 전체에 있을 수 있다. 알 수 없다. Stage별 row를 요구하라.
- Eval decontamination은? Paper가 말하지 않는다면 MMLU gain이 contaminated라고 가정하고 first-order correction으로 discount하라.

이 checklist를 새 pipeline에 적용하는 것이 논문 하나를 더 읽는 것보다 intuition을 더 빠르게 날카롭게 한다. 이 장의 네 pipeline은 vocabulary를 제공한다. Checklist는 그것을 사용하는 방법이다.

**미래 pipeline의 형태.** 2025-2026 release에서 세 trend가 보인다.

- *Multi-classifier cascade* — 하나의 educational-value classifier 대신, pipeline이 specialized classifier(math-value, code-quality, reasoning-density)를 쌓고 score를 blend한다.
- *Rephrasing hybrid* — [[rephrasing-the-web]] 스타일 pipeline은 filter 후 rewrite한다. Curation pipeline이 앞단에 quality gate를 둔 generation pipeline으로 변한다.
- *Domain-native per-source pipeline* — Dolma의 per-source lane은 proof of concept였다. 현대 frontier training mix는 8-12개의 per-source pipeline을 병렬로 실행하며 web pipeline은 그중 하나일 뿐이다.

이 각 방향은 CCNet skeleton을 유지하면서 step three에 추가 machinery를 붙인다. 어느 것도 위의 네 질문 checklist를 벗어나지 못한다.

Multi-classifier cascade도 single-classifier pipeline을 audit할 때와 같은 질문으로 audit한다. 어떤 signal, 어떤 order, 어떤 threshold, 어떻게 ablate했는가. Skeleton이 안정적이기 때문에 checklist가 load-bearing이다. Pipeline이 CCNet shape에 맞는 한 같은 네 질문이 여전히 올바른 질문이다.

---

## 7. 이것이 남기는 것

이 장이 끝나면 mental model은 다음과 같아야 한다.

- **CCNet은 여전히 skeleton이다.** Structural stage → dedup stage → quality stage. 이후 모든 pipeline은 이 shape을 다른 component와 ordering으로 구체화한 것이다.
- **C4는 recipe가 아니라 reference다.** Reproducibility를 위해 rule을 복사해야 한다면 복사하라. 하지만 어떤 rule이 일을 하는지(three-sentence dedup, langdetect), 어떤 rule이 legacy noise인지(JavaScript-line strip, lorem-ipsum strip) 이해하라.
- **Dolma는 scientific bar다.** Per-source pipeline, published threshold, ablation table. 다음 data release가 이 수준의 transparency에 맞지 않으면 reader는 평가할 수 없다.
- **FineWeb-Edu는 2024+ web-quality filtering의 baseline이다.** 하나의 classifier, 하나의 threshold, 그 threshold에 대한 하나의 ablation-driven choice. 또한 recipe이기도 하다. Anchor-LLM annotation prompt를 바꾸면 자신의 domain에 맞게 다시 실행할 수 있다.

Chapters [[ch-11]](tokenizers, shards, lineage, PII ops)와 [[ch-12]](deduplication)는 pipeline의 downstream half를 더 자세히 분해한다. 이 장은 pipeline이 cleaned text를 emit하는 지점에서 멈춘다. 다음 두 장은 그 text가 model에 닿기 전에 무엇을 하는지 다룬다.

---

## 연결과 다음 내용

- **[[ccnet]] / §1** — three-stage template, KenLM-vs-Wikipedia perplexity signal.
- **[[c4]] / §2** — heuristic baseline, bias case study.
- **[[dolma]] / §3** — transparency benchmark, per-source pipeline과 ablation table.
- **[[fineweb]] / §4** — classifier era, per-dump MinHash, FineWeb-Edu의 Llama-3-labeled quality model.
- **[[scaling-laws-data-quality]]** — 명시적인 scaling variable로서의 quality.
- **[[rephrasing-the-web]]** — "filter에는 ceiling이 있다"는 critique와 rephrase-the-web complement.
- **[[deduplicating-training-data]] / [[minhash-lsh]]** — §3-§4의 pipeline이 적용하는 dedup method.
- **ch-11 (data ops)** — *operation*으로서의 tokenizer, shard, lineage, PII. 이 장의 pipeline이 token을 emit한 *뒤* 만드는 것.
- **ch-12 (dedup)** — Lee 2021 foundation과 Dolma/FineWeb이 사용하는 near/semantic dedup method.

## 더 읽을거리

- [[ccnet]] — Wenzek et al. 2019, three-stage multilingual template.
- [[c4]] — Raffel et al. 2019(T5 paper); heuristic baseline과 Connections의 Dodge et al. 2021 bias audit.
- [[dolma]] — Soldaini et al. 2024; ablation table(Table 5)이 사진 찍을 대상이다.
- [[fineweb]] — Penedo et al. 2024; per-dump MinHash finding과 FineWeb-Edu classifier recipe.
- [[scaling-laws-data-quality]] — Subramanyam et al. 2025; scaling variable로서의 quality.
- [[rephrasing-the-web]] — Maini et al. 2024; filter-ceiling critique.

## 동반 시각화

**[figures/pipeline-compare.html](figures/pipeline-compare.html)** — 네 개의 vertical pipeline lane(CCNet, C4, Dolma, FineWeb-Edu). 어떤 stage든 클릭하면 정확한 filter signal, threshold, 그리고 paper가 보고한 경우 ablation delta를 볼 수 있다. 이 page는 §5의 comparison table에 대한 running reference다. Stage는 *kind*(structural, quality, content, dedup)별로 color되어 각 pipeline이 filter budget을 어디에 투자하는지 한눈에 볼 수 있다.
