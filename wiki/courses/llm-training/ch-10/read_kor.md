<!-- chapter: ch-10
     track: pretraining
     kind: content
     title: 휴리스틱 큐레이션 파이프라인: CCNet, C4, Dolma, FineWeb
     deps: [ch-09]
     sources: [[ccnet]], [[c4]], [[dolma]], [[fineweb]], [[minhash-lsh]], [[paloma]], [[whose-language-counts-quality-filter]], [[pretrainers-guide-training-data]]
     figures: figures/pipeline-compare.html
     revised: 2026-09 (generality revision)
-->

# 10장 — 휴리스틱 큐레이션 파이프라인: CCNet, C4, Dolma, FineWeb

> **핵심 통찰.** 휴리스틱 파이프라인은 Common Crawl snapshot의 대다수를 제거하며(Dolma의 CCNet stage만으로 byte의 84.2% 제거, [[dolma]] §5.1), 각 rule은 다른 domain 및 dialect를 서로 다른 비율로 제거한다. Dodge 등의 C4 재구축에서는 cleaning rule이 1.4T SpaCy token을 198B로 줄였고, 이어 word blocklist가 African American English로 분류된 문서의 42%를 제거한 반면 White-aligned English는 6.2%를 제거했다([[c4]], Dodge et al. 2021 §5.3). 1.2B 파라미터와 150B token에서 single-source filtered web corpus(C4, mC4-en, RefinedWeb)로 학습한 모델은 multi-source corpus 모델보다 다양한 Paloma domain에서 평균 perplexity가 높았다([[dolma]] §9.2). Dolma와 FineWeb이 유지한 rule set은 1.2B–1.71B 파라미터 data ablation으로 선택됐다. FineWeb은 8개 task commonsense·knowledge aggregate를 채점했고, Dolma는 filter ablation의 domain별 perplexity를 appendix figure로만 보인다. 따라서 특정 domain에서 filter 효과는 따로 검사해야 한다.
>
> **지침.** 휴리스틱 filter를 채택할 때는 rule의 granularity(line, paragraph, sentence, document), threshold, unit을 포함한 removal rate를 기록하라. “C4 filters”라는 이름의 rule도 T5 논문, Dodge 등, FineWeb이 사용한 datatrove default에서 threshold가 다르기 때문이다. Threshold를 고를 때는 고정 model size·token budget의 data ablation을 쓰고, code·LaTex·dialectal text에서 domain별 perplexity를 확인하라. FineWeb의 selection suite에는 이들이 없으며 Paloma의 C4-only baseline은 arXiv에서 perplexity 391,171에 도달했기 때문이다([[paloma]] §4.1). 많은 Common Crawl snapshot을 MinHash로 deduplicate할 때는 각 snapshot을 별도로 deduplicate하라. FineWeb의 1.71B-parameter, 350B-token ablation에서 global deduplication은 이득이 거의 없었고, 오래된 snapshot에서 유지한 data는 제거한 data보다 더 나쁜 모델을 학습시켰다([[fineweb]] §3.4). 단일 snapshot이면 두 절차는 같다.

## 일반 목적 모델에서 이 장이 중요한 이유

Dolma v1.6에서 filtered web text는 가장 큰 source다. Common Crawl은 Dolma v1.6의 3,059B token 중 2,479B를 공급한다([[dolma]] Table 1). ch-09는 source(web, code, book, academic text)를 비교했고, 이 장은 learned quality model을 적용하기 전 web source 안에서 일어나는 selection을 다룬다. Model-based quality classifier(FineWeb-Edu, DCLM, Nemotron-CC)는 ch-10a, deduplication algorithm의 심층 내용은 ch-12에서 다룬다.

Filtering rule은 모델이 절대 보지 못하는 text를 결정한다. Filter가 제거한 register, dialect, notation은 사전학습 data의 다른 source를 통해서만 모델에 도달할 수 있다(해석). 이 장은 휴리스틱 pipeline의 세 질문에 답한다. (1) 어떤 선택이 breadth를 유지하는가: stage order, language별 threshold, dropping 대신 labeling, snapshot별 deduplication. (2) 어떤 선택이 coverage를 좁히는가: word blocklist, code-removal rule, reference-anchored perplexity cutoff, 한 benchmark aggregate에 조정한 threshold. (3) 효과를 어떻게 측정하는가: domain별 perplexity, topic·dialect별 removal audit, model size·token budget·seed를 명시한 data ablation.

동반 그림 [figures/pipeline-compare.html](figures/pipeline-compare.html)은 reader가 네 pipeline의 각 stage를 눌러 rule, threshold, removal rate, locus를 보고, 서로 다른 bucket setting의 MinHash match probability를 계산하며, line·paragraph·document 수준 punctuation rule이 같은 document를 어떻게 다루는지 비교하게 한다.

## §1 휴리스틱 파이프라인 읽는 법

**정의.**
- **Common Crawl snapshot**은 crawl release 하나다. 각각은 WARC file(raw HTML 및 request metadata)과 WET file(Common Crawl이 추출한 text)로 배포된다([[fineweb]] §3.2). CCNet 시대 snapshot은 약 30억 page에서 온 비압축 text 20–30 TB를 담는다([[ccnet]] §3.1).
- **휴리스틱 filter**는 quality를 예측하도록 학습한 model이 아니라 text의 표면 통계(punctuation, word count, word list, repetition)로 계산한 rule이다. 이 장은 pipeline의 표준 stage이므로 language identification(fastText classifier)과 reference perplexity(5-gram model)도 다룬다.
- **Granularity**는 rule이 제거하는 단위다. Line, paragraph(Dolma에서 newline-separated span, [[dolma]] §4 footnote 3), sentence, 또는 document다.
- **Removal rate**는 명시한 unit에서 제거된 fraction이다. 출처는 서로 다른 unit을 쓴다. Character(CCNet), byte, document, UTF-8 character(Dolma), GPT-2 token(FineWeb), SpaCy token(Dodge 등)이다.
- **Data ablation**은 data 결정 하나만 제외하고 동일한 model을 학습하여 같은 evaluation에서 비교한다([[dolma]] §4.2; [[fineweb]] §3.1).

**읽기 템플릿.** 새 pipeline의 각 stage에서 여섯 field를 추출한다.
1. Stage order에서의 위치, order가 ablation되었는지 cost 때문에 선택됐는지.
2. Granularity.
3. Threshold 및 선택 방식(ablation, histogram inspection, token budget, 또는 미진술).
4. Unit을 포함한 removal rate.
5. Ablation design: model size, 실행당 token, seed 수.
6. Evaluation suite 및 측정하지 않는 capability.

§7은 네 pipeline 전체에 이 template을 표 하나로 적용한다.

## §2 CCNet: deduplication, language identification, perplexity bucket

**정의.** CCNet(Wenzek 등, arXiv 2019-11)은 각 Common Crawl snapshot의 WET text를 paragraph deduplication → fastText language identification → SentencePiece tokenization 및 5-gram perplexity → 언어와 perplexity bucket별 재그룹화 순서로 처리한다([[ccnet]] §3, Figure 1).

**문제.** 중복 paragraph는 snapshot text의 “70%를 차지한다”(§3.2). 논문은 navigation menu, cookie warning, contact information을 제거된 boilerplate로 들며, 이 step이 다른 언어 page에서 English content를 제거하여 다음 language identification step을 더 견고하게 만든다고 말한다(§3.2).

### §2.1 Paragraph deduplication, 그 다음 language identification

메커니즘([[ccnet]] §3.2–§3.3, §4.2):
1. 각 paragraph를 정규화한다. 소문자화하고, digit를 0으로 치환하며, Unicode punctuation과 accent mark를 제거한다.
2. 이를 hash(SHA-1 첫 64 bit)하고 hash를 5 GB shard별로 저장한다.
3. 비교 set에 그 hash가 있으면 paragraph를 제거한다. Shard 1개 전반에서는 shard character의 42%가 남고, shard 100개 전반에서는 28%가 남는다(Figure 4). 저자는 corpus의 약 3%인 shard 50개(1.5B hash, RAM 40 GB)와 비교한다.
4. fastText language identification(언어 176개)을 실행한다. Top language의 score가 0.5 초과이면 그 언어로 page를 유지하고, 아니면 폐기한다.

CCNet은 exact matching만 사용하며 MinHash 또는 다른 fuzzy step은 없다([[ccnet]] Technical Details).

**순서가 중요한 이유.** 저자들은 “cookie warning 같은 English boilerplate가 다른 언어의 page에 많이 존재하기” 때문에 language identification 전에 deduplicate한다(§4.1). February 2019 snapshot의 1%로 추정한 Figure 3은 “dedup then LID”와 “LID then dedup”의 언어별 document ratio를 그린다. 본문은 저자원 언어가 가장 큰 이득을 보며, 먼저 deduplicate하지 않으면 그 document가(일반적으로 English로) 오분류되거나 언어를 식별할 수 없어 폐기된다고 말한다(§4.1). 언어별 값은 figure에만 있다. **Result(단일 연구).**

구성한 예(fastText score는 계산하지 않음): Swahili page에 Swahili 400 character와 수천 page에 등장하는 English cookie notice·menu 600 character가 있다. Deduplication 전에는 English character가 다수여서 Swahili top-language score가 0.5 아래로 내려가 page는 폐기되거나 English로 label될 수 있다. Deduplication이 반복 English paragraph를 제거한 뒤 page에는 Swahili text만 남는다.

### §2.2 Wikipedia 대비 perplexity bucketing

CCNet은 언어 48개 각각에 대해 해당 언어 Wikipedia로 SentencePiece tokenizer와 5-gram Kneser–Ney model(KenLM)을 학습하고 각 paragraph의 perplexity를 계산한다([[ccnet]] §3.4). 논문은 식을 인쇄하지 않으며 표준 정의는 다음과 같다.

```
PPL(x) = exp( −(1/N) · Σ_{i=1..N} ln p(t_i | t_{i−4}, …, t_{i−1}) )
```

- `x`는 paragraph, `t_i`는 i번째 SentencePiece token, `N`은 token 수다.
- `p(t_i | …)`는 앞선 네 token이 주어졌을 때 5-gram model이 `t_i`에 주는 확률이다.

낮은 perplexity는 text가 reference domain에 더 가깝다는 뜻이다(§3.4). Paragraph score를 document score로 어떻게 결합하는지는 논문이 설명하지 않는다.

**계산 예.** 확률 0.5, 0.25, 0.5, 0.125인 4-token paragraph의 평균 log-probability는 (−0.693 − 1.386 − 0.693 − 2.079)/4 = −1.213이므로 PPL = e^1.213 ≈ 3.36이다(동치로 (2·4·2·8)^(1/4) = 128^(1/4)). 한 언어의 document 9개 perplexity가 12, 15, 18, 25, 31, 40, 55, 90, 300이라고 하자. 동등한 세 부분으로 나누면 head = {12, 15, 18}, middle = {25, 31, 40}, tail = {55, 90, 300}이다. CCNet은 perplexity 분포가 언어마다 다르므로 이 tercile threshold를 언어별로 따로 계산한다(§3.4, §5.2). 저자들은 차이를 각 LM 학습 Wikipedia의 크기에 돌린다(해석, §5.2). English LM은 “534M” text, Gujarati LM은 “12M” text로 학습했다(§5.2).

**CCNet은 label하고 drop하지 않는다.** “유효한 text임에도 Wikipedia와 vocabulary가 매우 다른 document는 tail에 들어간다. 여기에는 spoken-like text가 있는 blog comment 또는 특수 jargon이 있는 전문 forum이 포함된다. 우리는 LM score를 기준으로 content를 제거하지 않기로 했다”([[ccnet]] §5.2). Downstream user는 더 엄격한 선택을 했다. FineWeb §2는 CC-100이 “low perplexity에 할당된 text만” 유지한다고 설명한다([[ccnet]] Connections).

**근거.** English head·middle·tail로 학습한 word embedding(fastText, 300-d)은 analogy accuracy 77.9, 74.2, 62.0에 도달한다(Table 1). Next-sentence prediction 없는 BERT-BASE를 CC head로 학습하면 en/ru/zh/ur의 XNLI dev accuracy 평균은 Wikipedia 학습의 72.6 대비 75.9이며, Urdu는 57.3에서 64.3으로 오른다(Table 2). **한계:** snapshot 하나뿐이며, Table 2는 CC head를 Wikipedia와 비교할 뿐 unfiltered·middle·tail Common Crawl과 비교하지 않는다. Seed나 variance도 보고하지 않는다([[ccnet]] Findings).

**함의.** Reference-perplexity score는 reference corpus와의 유사도를 측정한다. [[whose-language-counts-quality-filter]]는 다른 scorer로 같은 설계 선택을 시험한다. Wikipedia, Books3, OpenWebText를 Common Crawl과 구별하도록 학습한 복제 GPT-3 quality classifier는 U.S. high-school newspaper article 910K개에서 더 부유하고 교육 수준이 높으며 urban한 ZIP code의 더 큰 school article을 더 높게 평가했다(§3.4, Table 3). Factual reliability가 높은 news source와 낮은 source 사이 score 차이는 없었다(§4.2, Kolmogorov–Smirnov p = 0.085). **Result(단일 연구).** CCNet bucket을 label로 유지하면 나중의 mixture가 tail text 사용량을 결정할 수 있다.

## §3 C4: line·page rule, span deduplication, blocklist가 제거한 것

**정의.** C4(Raffel 등, arXiv 2019-10)는 line 및 page rule과 langdetect English identification을 통과한 April 2019 Common Crawl WET text로, 약 750 GB text다([[c4]] §2.2).

### §3.1 진술된 rule과 code된 rule

[[c4]] §2.2는 다음을 열거한다. terminal punctuation mark(period, exclamation mark, question mark, end quotation mark)로 끝나는 line만 유지; sentence 3개 미만 page 폐기; word 5개 이상 line만 유지; “List of Dirty, Naughty, Obscene or Otherwise Bad Words”의 단어가 하나라도 있는 page 제거; “Javascript”가 있는 line 제거; “lorem ipsum”이 있는 page 제거; “많은 programming language에는 나타나지만 … natural text에는 나타나지” 않는 curly bracket “{”가 있는 page 제거; citation marker 제거; policy string이 있는 line 제거; “한 번보다 많이 나타나는 세-sentence span은 하나만 남기고 모두 폐기”; langdetect가 확률 0.99 이상 English로 분류한 page 유지.

TensorFlow Datasets release code(`tensorflow_datasets/text/c4_utils.py`, master, 2026-09-14 fetch, L248–270)는 각 rule의 granularity와 검사 순서를 보인다.

```python
    if not line.endswith(_END_MARKS) or line.endswith(_ELLIPSIS):
      counter_inc_fn("line-filtered:no_endmark")
      continue
    if len(line.split()) < min_words_per_line:
      counter_inc_fn("line-filtered:too_short")
      continue
    line_lower = line.lower()
    # Remove documents which contain lorem ipsum
    if "lorem ipsum" in line_lower:
      counter_inc_fn("filtered:loremipsum")
      return
    # Remove "javascript must be enabled" notices
    if "javascript" in line_lower:
      counter_inc_fn("line-filtered:javascript")
      continue
    # Remove docs which probably contain javascript code
    if "{" in line:
      counter_inc_fn("filtered:squigglybracket")
      return
    # Remove policy lines
    if any(p in line_lower for p in _POLICY_SUBSTRINGS):
      counter_inc_fn("line-filtered:policy")
      continue
```

`continue`는 line 하나를 제거하고 `return`은 page를 제거한다. 이 code에서 `{`는 이미 end-mark 및 5-word 검사를 통과한 line에 있을 때만 page를 제거하고, “...”로 끝나는 line은 제거된다. 같은 file은 line 수준에서 deduplicate한다(`remove_duplicate_text`, hashed line당 URL 하나 유지). 반면 §2.2는 three-sentence span을 기술한다. 두 artifact 어느 것도 2019 release를 만든 procedure가 무엇인지 진술하지 않는다.

### §3.2 근거: filtered 대 unfiltered C4

T5는 C4(745 GB) 및 “unfiltered C4”(langdetect만 적용한 6.1 TB)에서 2^35 ≈ 34B token으로 사전학습한 220M baseline을 비교한다([[c4]] §3.4.1, Table 8). C4는 일곱 열 모두 더 높다. GLUE 83.28 대 81.46, CNN/DM 19.24 대 19.14, SQuAD 80.88 대 78.78, SuperGLUE 71.36 대 68.04, EnDe 26.98 대 26.55, EnFr 39.82 대 39.34, EnRo 27.65 대 27.21. **Result(단일 연구).** 비교는 모든 rule을 한꺼번에 제거했으며 T5 논문은 개별 rule의 효과를 측정하지 않는다([[c4]] Findings).

### §3.3 Dodge 등(2021)이 측정한 부작용

Dodge 등(arXiv:2104.08758v2, [[c4]] Connections에 기록)은 세 version을 재구축했다. Langdetect만 적용한 C4.EN.NOCLEAN은 1.1B document와 1.4T SpaCy token; blocklist 외 모든 rule을 적용한 C4.EN.NOBLOCKLIST는 395M document와 198B token; C4.EN은 365M document와 156B token이다(Table 1).

**계산 예(Table 1에서 도출).**
- Blocklist 이외 rule은 token의 1 − 198/1,400 = 85.9%를 제거한다.
- Blocklist는 남은 document의 1 − 365/395 = 7.6%를 제거하지만, 남은 token의 1 − 156/198 = 21.2%를 제거한다.
- 따라서 제거된 document는 평균 42B/30M = 1,400 token이며, 유지 document의 156B/365M ≈ 427 token과 대조된다. 단어 하나라도 있으면 발동하는 page-level rule은 listed word를 포함할 기회가 많은 긴 document를 더 자주 제거한다(해석).

**Blocklist가 제거한 것**(§5): 50 cluster로 clustering한 excluded document 100,000개 중 16 cluster(제외 document의 31%)는 주로 sexual하다. 다른 cluster는 science, medicine, health, legal, political text를 포괄한다. Sexual orientation 언급은 가장 높은 제거 가능성을 가지며, “lesbian” 및 “gay”를 언급해 제외된 document를 각각 50개 sample하면 22%와 36%가 non-offensive 또는 non-sexual이다. African American English 및 Hispanic-aligned English로 분류된 document는 42% 및 32%가 제거되며, White-aligned는 6.2%, 다른 English는 7.2%다. C4.EN의 97.8%는 White-aligned, 0.07%는 AAE, 0.09%는 Hispanic-aligned다(§5.3). **Result(단일 연구).** 이 제거는 blocklist에 귀속된다. Langdetect 0.99 threshold가 dialect에 주는 효과는 Dodge 등이나 T5 논문에서 측정하지 않는다.

### §3.4 재구현은 threshold를 바꾼다

Rule 이름은 threshold를 고정하지 않는다. T5 §2.2 및 TFDS code는 line당 word 5개와 page당 sentence 3개를 쓴다. Dodge 등 §2는 “word 세 개 미만”과 “sentence 다섯 개 미만”을 기술한다([[c4]] Connections). Commit acd431b의 datatrove `C4QualityFilter`는 `min_num_sentences: int = 5`, `min_words_per_line: int = 3`을 설정한다(`src/datatrove/pipeline/filters/c4_filters.py` L66–67). FineWeb 공개 script는 `filter_no_terminal_punct=False`만 바꾼 채 이 class를 호출한다(§5.6).

## §4 Dolma: source별 pipeline과 보고된 filter ablation

**정의.** Dolma v1.6(Soldaini 등, arXiv 2024-01)는 Common Crawl, GitHub, Reddit, Semantic Scholar, Project Gutenberg, Wikipedia/Wikibooks에서 온 3,059B-token English corpus이며, 각 source에 자체 pipeline이 있다([[dolma]] Table 1). Web subset은 2020-05부터 2023-06까지 snapshot 25개에서 온 2.28T token이다(§5).

### §4.1 Web pipeline 순서와 removal rate

순서([[dolma]] §5.5): CCNet output → URL dedup → document dedup → quality filter → content filter → paragraph dedup.
1. **CCNet stage**(§5.1): fastText English score ≥ 0.5는 byte 기준 data의 61.7%를 제거한다. Snapshot 안 paragraph dedup은 paragraph 약 70%를 제거하며 CCNet 전체는 175.1 TB → 27.7 TB, 84.2%를 제거한다.
2. **Bloom filter를 이용한 deduplication**(§5.4): exact URL은 document 53.2%, exact document는 URL-deduplicated document 14.9%, exact paragraph는 paragraph 18.7%를 제거한다. URL·document dedup이 “효율을 높이기 위해” 먼저 실행되고, paragraph dedup은 “paragraph 제거가 content analysis를 방해할 위험” 때문에 마지막이다. 순서는 cost로 정당화되고 순서 ablation은 보고되지 않는다(§5.4).
3. **Quality**(§5.2): 모든 Gopher rule(“Gopher All”, character 15.23% tag)과 C4 rule 하나(“C4 NoPunc”, 22.73%)다. Gopher rule에는 word 50개 미만 또는 100K 초과, median word length 3 미만 또는 10 초과, symbol-to-word ratio 0.10 초과, bullet로 시작하는 line 0.90 초과, ellipsis로 끝나는 line 0.30 초과, duplicate line 0.30 초과가 포함된다(App. N.4).
4. **Toxicity**(§5.3): Jigsaw Toxic Comments로 학습한 fastText classifier 두 개가 sentence를 score하며, τ = 0.4 초과 sentence는 제거한다.
5. **PII**(§5.3): email address, IP address, phone number의 regular expression이다.

**Dolma 내부 granularity 충돌.** §5.2는 C4 NoPunc을 “punctuation으로 끝나지 않는 paragraph” 제거로 기술한다. Datasheet(App. N.4)는 같은 22.73% filter를 “line의 절반 초과가 '.', '?', '!', 또는 '"'로 끝나지 않는 document 제거”로 기술한다. 논문은 둘을 조정하지 않는다.

### §4.2 Filter ablation

설정: contamination을 피하려 선택한 8개 dataset, ARC-E, ARC-C, BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande를 zero-shot으로 평가한 150B-token 학습 1.2B-parameter OLMo-architecture model(§4.2, Table 2, App. D)이다. 결과는 training token에 따른 curve로 제시된다. Main text에는 HellaSwag, 다른 task 및 Paloma subset(예: Twitter AAE, 4chan, M2D2 S2ORC)은 App. O에 있다. 논문은 이 ablation의 numeric table을 주지 않는다.
- **Quality rule**(§5.2, Figure 1, HellaSwag; 다른 task는 App. O): “C4 NoPunc 단독은 perplexity와 downstream task 모두에서 C4 All 및 Gopher All보다 낫고”, “Gopher All + C4 NoPunc이 최선 성능을 제공한다.”
- **Toxicity threshold**(§5.3, Figure 2): τ = 0.4는 5.5–7.3%를 제거하지만, 29.1–34.9%를 제거하는 τ = 0.0004보다 “일반적으로 더 낮은 downstream 성능”을 준다. 저자는 “최소 token count 요구를 충족하기 위해” τ = 0.4를 채택했다.
- **Stacking**(§5.5, Figure 3): quality filter에 dedup 및 content filter를 더하면 HellaSwag에서 “긍정적 복합 효과”가 난다. Filter가 제거하는 것은 거의 겹치지 않는다(§5.3, App. J).
- **Heuristic 대 reference perplexity**(§5.2): CCNet KenLM bucket 비율(high 21.9%, medium 28.5%, low 49.6%)은 Dolma heuristic filter 뒤에도 바뀌지 않았다. 저자들은 두 signal이 직교한다고 읽는다(해석).

**계산 예(PII rule, §5.3).** 탐지 span 5개 document는 각 span을 `|||EMAIL_ADDRESS|||` 같은 token으로 치환해 text를 유지한다. Span 6개 document는 제거된다. 논문은 document 0.02% masking, 0.001% removal을 보고하며 replacement 대 removal의 측정 효과는 없다. Datasheet는 masking tag 0.05%, removal tag 0.11%를 열거한다(App. N.4).

### §4.3 Source별 pipeline

Code(§6): 생성자가 이미 MinHash·LSH deduplicate한 The Stack; JSON·CSV 같은 data file 제거; RedPajama v1 rule 및 StarCoder rule(RedPajama v1 rule 단독보다 더 낮은 code perplexity); detect-secrets match가 있는 document 제거다. Reddit(§7): 500 character 미만 comment, 400 character 미만 submission, vote 3개 미만 comment 제거; banned 또는 NSFW subreddit 26,123개 제외; comment와 submission을 별도 document로 취급하는 것이 thread linearization보다 점수가 높다(Figure 4). peS2o는 그대로 사용, Project Gutenberg는 exact title로 deduplicate, Wikipedia는 25 word 이하 document를 제거한다(§8). **한계**(Limitations): ablation은 1B 규모 dense architecture 하나를 쓰며, 이는 “더 큰 model size에는 관련되지 않는 design decision”을 낳을 수 있다. Release된 Dolma v1.6은 decontaminate되지 않았다(App. N.4).

## §5 FineWeb: 추출, ablation으로 고른 filter, snapshot별 MinHash

**정의.** FineWeb(Penedo 등, arXiv 2024-06)은 stage마다 data ablation으로 선택한 Common Crawl snapshot 96개 기반 15T GPT-2-token English corpus다([[fineweb]] §3).

### §5.1 Ablation protocol

1.71B-parameter model(Llama architecture, sequence length 2,048, global batch 약 2M token)은 filtering ablation에 약 28B token, 일부 deduplication·cumulative run에 350B token으로 학습한다. 각 data version은 서로 다른 data subset·seed로 두 번 실행해 평균한다([[fineweb]] §3.1). Benchmark는 CommonSense QA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC, MMLU이며 run 사이 낮은 variance, 학습 중 거의 monotonic한 개선, 이 규모에서 random 이상 score 때문에 골랐다(§3.1). 70개 초과 model을 학습했으며 추정 80,000 H100 GPU hour다(§3.1).

### §5.2 Extraction과 base filtering

WARC file에 적용한 Trafilatura는 language filtering만 적용한 28B-token ablation에서 Common Crawl WET text보다 나았다(§3.2, Figure 1). Base filtering(§3.3)은 adult content URL blocklist, fastText English score ≥ 0.65, 원래 threshold의 MassiveText(Gopher) quality·repetition filter를 적용하며 96 snapshot에서 약 36T token을 만든다.

### §5.3 MinHash parameter

**정의.** Document를 word 5-gram(shingle)으로 나눈다. Document A와 B의 Jaccard similarity는 `J(A, B) = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|`이며 `S(D)`는 D의 shingle set이다([[minhash-lsh]] §2). Shingle space의 균일한 random permutation 하나에서 S(A) ∪ S(B)의 최소 shingle이 양쪽에 있을 확률, 즉 두 document의 minimum이 같을 확률은 J(A, B)와 같다([[minhash-lsh]] §3, Theorem 1). MinHash 구현은 각 random permutation을 hash function으로 근사한다.

**계산 예(shingle 크기).** A = “the cat sat on the mat today”, B = “the cat sat on a mat today”라 하자. 단어 하나면 J = 6/7 = 0.86, bigram이면 4/8 = 0.50, trigram이면 2/8 = 0.25다. 한 단어만 다른 1,000-word document 둘은 각각 word 5-gram 996개, 그중 5개가 다르므로 J = 991/1,001 = 0.99다. 더 긴 document는 고정 shingle size에서 작은 edit을 허용한다.

**FineWeb 설정**([[fineweb]] §3.4, App. E.1): hash function 112개를 8개씩 14 bucket으로 나눈다. 한 bucket 이상에서 minhash 8개가 모두 같으면 두 document는 candidate duplicate다. Candidate는 transitive하게 cluster하고 cluster당 random document 하나를 유지한다. Match probability는 다음과 같다.

```
P(s) = 1 − (1 − s^8)^14
```

- `s`는 두 document의 n-gram(Jaccard) similarity다.
- `s^8`은 bucket 하나가 match할 확률, `(1 − s^8)^14`는 14 bucket 어느 것도 match하지 않을 확률이다.

s = 0.8이면 0.8^8 = 0.168, (1 − 0.168)^14 = 0.076, P = 0.92다. 논문은 s = 0.7, 0.75, 0.8, 0.85에서 P = 56%, 77%, 92%, 98.8%를 보고한다(App. E.1). s = 0.5에서 P = 0.053이다. RefinedWeb은 hash 9,000개를 20개씩 450 bucket에 사용해 더 많은 compute로 더 가파른 곡선을 만든다(App. E.1, Figure 13). Release script도 같은 값을 설정한다(`datatrove@acd431b examples/fineweb.py` L80–88: `num_buckets=14`, `hashes_per_bucket=8`, `n_grams=5`). [figures/pipeline-compare.html](figures/pipeline-compare.html)의 MinHash panel은 bucket·bucket당 hash를 바꾸고 P(s)를 읽게 한다.

### §5.4 Global 대 snapshot별 deduplication 및 제시된 이유

1. **Global run**(§3.4, Figure 3): 최신(2023-50)에서 오래된 snapshot 순으로 96개 모두에 MinHash를 적용하면 오래된 snapshot base-filtered data의 최대 90%를 제거하고 4T token을 남긴다. 350B-token run은 non-deduplicated data보다 “거의 개선”되지 않았다.
2. **진단**(§3.4, Figure 4): snapshot 2013-48에서 global dedup이 유지한 약 31B token은 제거된 약 460B token을 따로 deduplicate해 얻은 171B token보다 나쁜 model을 학습한다. 육안 점검에서 kept data는 “더 많은 ad, incoherent keyword list, 일반적으로 잘못 formatting된 text”를 포함한다. 논문은 kept share를 10%로 기술하나 31/(31 + 460) = 6.3%다(도출).
3. **Snapshot별 run**(§3.4, Figure 5): 각 snapshot을 독립 deduplicate하면 20T token을 주고 RefinedWeb과 일치한다.
4. **위에 올린 가벼운 global 방법**(App. E.3, Figure 15): URL dedup(제거 71.5%), line dedup(77.8%), minimum word를 둔 line dedup(85%), 3-line dedup(80.9%) 모두 snapshot별 MinHash 단독보다 낮은 score를 냈다.

**제시된 이유.** 저자들은 이득 대부분이 모든 crawl에 존재하는 큰 duplicate cluster 제거에서 오며, 약 100개( crawl 수)보다 적은 copy를 가진 cluster 제거는 “성능을 해칠 수 있다”고 가설을 세운다(해석, §3.4). Step 2에 대한 course 해석(논문 미검증)은 newest-to-oldest global dedup에서 2013-48 page가 살아남으려면 이후 snapshot 어느 것에도 near-duplicate가 없어야 한다는 것이다. 수년 re-crawl된 page는 old snapshot에서 제거되고 한 번 나타난 page는 유지된다. **측정 한계**(App. E.2, Figure 14): 동일한 200B-token snapshot 100개의 simulation에서 1B-token sample은 거의 모두 unique document이므로 1B-token ablation으로 dedup effect를 보일 수 없다. **Result(단일 연구);** ch-12는 이를 Llama 3 global dedup과 비교한다.

### §5.5 C4 rule 재-ablation과 새 rule 도출

Base-filtered, snapshot별 deduplicate한 2019-18 crawl에서([[fineweb]] §3.5, Figure 6), terminal-punctuation rule은 가장 큰 개별 HellaSwag 이득을 주었지만 token 약 30%를 제거했다. Curly-bracket 및 word-length rule은 2.8%와 4.3%를 제거했고, lorem-ipsum·javascript·policy rule은 각각 0.5% 미만을 제거했다. Terminal punctuation을 제외한 모든 rule은 약 7%를 제거하면서 terminal punctuation 단독보다 높은 score를 냈다. FineWeb은 terminal punctuation을 제외한 모든 C4 rule을 유지했다.

**새 rule**(§3.6): 저자들은 2013-48의 snapshot별 deduplicate version(더 높은 quality로 취급)과 global deduplicate version(더 낮은 quality로 취급)에서 50개 초과 statistic을 계산하고, 낮은-quality set이 더 높은 density를 보이는 위치에 threshold를 정했다. Candidate 16개를 28B token에서 시험하고(App. E.4, Table 2) 세 개를 유지했다. Punctuation으로 끝나는 line fraction이 ≤ 0.12(document token 10.14%), duplicated line 안 character fraction이 ≥ 0.1(12.47%), 30 character보다 짧은 line fraction이 ≥ 0.67(3.73%)인 document를 제거한다. 함께 token 약 22%를 제거하고 aggregate score를 약 1% 높였다. 여기서 “higher”와 “lower” quality label은 human judgment가 아니라 deduplication outcome에서 왔다.

**계산 예(granularity).** 10-line document에 punctuation으로 끝나는 line 3개가 있으며 각각 word 5개 이상이라고 하자.
- C4 line rule: line 3개는 유지하고 7개를 삭제한다(page는 그 3 line이 sentence 3개 이상이면 생존).
- Dolma datasheet rule: 10 line 중 7개가 punctuation이 없으므로 0.7 > 0.5이고 document 전체를 제거한다.
- FineWeb rule: 3/10 = 0.3 > 0.12이므로 document 전체를 바꾸지 않고 유지한다.

10개 중 punctuated line이 1개면 FineWeb ratio는 0.1 ≤ 0.12이고 document는 제거된다. [figures/pipeline-compare.html](figures/pipeline-compare.html)의 granularity panel은 punctuated line 수를 조절할 수 있는 document에 세 rule을 적용한다.

**Dolma와 FineWeb은 punctuation rule에서 불일치한다. Open question.** Dolma는 C4 NoPunc을 유지하고 다른 C4 rule은 버렸으며 FineWeb은 terminal punctuation을 버리고 다른 C4 rule은 유지했다. Dolma에서 C4 NoPunc 단독은 full C4 rule set 및 Gopher All보다 좋았고, FineWeb에서는 terminal punctuation이 시험한 C4 rule 중 가장 큰 개별 HellaSwag 이득을 냈다. 구성은 granularity(Dolma: paragraph 또는 document, FineWeb: line), baseline data(Dolma: CCNet-processed WET text, §5.1; FineWeb: base filtering·snapshot별 MinHash의 trafilatura extraction), model size(1.2B 대 1.71B), token budget이 다르며 FineWeb은 token removal을 이유로 제시했다. 어느 출처도 이 factor를 고정한 채 rule을 ablation하지 않았다.

### §5.6 최종 순서와 release code와의 충돌

논문의 최종 pipeline(§3.7)은 WARC extraction → base filtering → snapshot별 MinHash → C4 rule → custom rule이며, 이어 email 및 public IP address를 anonymize한다. Datasheet(App. A)는 custom filter 뒤에 MinHash를 열거한다([[fineweb]] Technical Details). Release script(`datatrove@acd431b examples/fineweb.py` L41–62)는 MinHash 전에 dump별 모든 filter를 실행한다.

```python
        URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
        Trafilatura(favour_precision=True),
        LanguageFilter(
        ...
        GopherRepetitionFilter(
        ...
        GopherQualityFilter(
        ...
        C4QualityFilter(
            filter_no_terminal_punct=False,
        ...
        FineWebQualityFilter(
```

같은 commit의 `FineWebQualityFilter` 기본값은 `line_punct_thr: float = 0.12`, `short_line_thr: float = 0.67`, `char_duplicates_ratio: float = 0.01`이다(`fineweb_quality_filter.py` L14–18). 마지막 값은 App. E.4 Table 2(≤ 0.01)와 일치하며 §3.6 text(≥ 0.1)와 다르다. Filter는 word당 newline이 0.3을 넘는 document도 제거한다(`new_line_ratio`, L19, L49–52). 논문은 이 rule을 기술하지 않는다. 상태: **충돌(conflict)**. 논문, datasheet, pinned code는 별도 fact이며 code commit(2024-08)은 논문 experiment보다 늦다.

## §6 분포 선택기로서의 filter: filter가 제거하는 것 측정하기

Removal rate는 filter가 제거하는 text 양만 말하며, 어느 domain이 text를 잃는지는 말하지 않는다. 두 측정이 이를 다룬다. Filtered data로 학습한 model의 domain별 perplexity와, 제거 document의 topic·dialect audit이다.

**Domain별 perplexity**([[paloma]] §3, §4.1). Evaluation document set N에 대해 `ℓ = Σ_{t∈N} Σ_i ln p(t_i | t_<i)`, `perplexity = exp(−ℓ / T(N))`이며, `t`는 document, `t_i`는 i번째 token, `T(N)`은 token count다. Domain set D의 macro average는 다음과 같다.

```
PPL_macro(D) = (1/|D|) · Σ_{d∈D} perplexity(d)
```

- `D`는 domain set(예: subreddit, programming language, S2ORC field)이며 `perplexity(d)`는 domain d 단독에서 계산한다.

**계산 예.** News perplexity 15, arXiv perplexity 60인 model을 두자. News가 evaluation token의 90%면 token-weighted perplexity는 exp(0.9·ln 15 + 0.1·ln 60) = exp(2.847) ≈ 17.2다. Macro average는 (15 + 60)/2 = 37.5다. Token-weighted 값 17.2는 news 값보다 2.2 높을 뿐인 반면 macro average 37.5는 arXiv 쪽으로 절반 이동하므로, macro average는 약한 domain을 보이고 token-weighted 값은 대체로 숨긴다.

**근거.**
- **Single-source web corpus**([[dolma]] §9.2, Figure 5; [[paloma]] §4.1): C4, mC4-en, RefinedWeb의 150B token으로 학습한 1.2B model은 Pile, Dolma, RedPajama model보다 평균 Paloma perplexity가 높다. C4 model은 RedPajama arXiv에서 perplexity 391,171, Dolma peS2o에서 14를 얻는다. RefinedWeb 및 mC4-en model은 Max programming-language domain에서 21,652 및 1,409를 얻는다. Paloma 저자는 이 domain이 LaTeX와 code를 포함한다고 지적하고, “하나의 data source에 적용된 하나의 cleaning filter set만 사용해 완전히 filter된 특정 language type에 노출되지 않은 것”을 제안한다(해석). Course 해석(미검증)은 code 제거를 위해 쓴 C4의 page-level `{` rule(§3.1)이 LaTeX page도 제거한다는 것이다.
- **이질적 source**([[pretrainers-guide-training-data]] §6): Pile에서 Common Crawl, Books, OpenWeb 중 하나를 제거하면 1.5B decoder model의 평균 downstream QA 성능이 가장 많이 저하됐다. Source와 metric은 다르지만 [[dolma]] §9.2와 방향이 같아 **Replicated**다.
- **Filter의 topic·domain shift**([[fineweb]] §4.1–§4.2): FineWeb 저자는 corpus별 document 50k를 embed하고 cluster해 cluster share를 비교한 뒤, decontamination 없이 Paloma perplexity를 비교한다. Classifier-filtered FineWeb-Edu에 적용하면 “Education, Learning, Teaching” +3.2%, “History, Culture, Politics” +2.2%이며 Business, Entertainment, Travel cluster는 down-sample된다. FineWeb은 Twitter AAE, Manosphere, Gab, 100 Subreddits, 4chan을 더 잘 맞추고 FineWeb-Edu는 Wikipedia, S2ORC, arXiv, programming language 100개를 더 잘 맞춘다. 같은 방법은 어느 heuristic filter에나 적용할 수 있으며 ch-10a가 classifier 자체를 다룬다.
- **Filter는 capability를 불균등하게 바꾼다**([[pretrainers-guide-training-data]] §5, Figures 5–7): blocklist 없이 deduplicate한 C4 version의 1.5B decoder model을 과제별 fine-tune한다. T = 0.3의 Perspective API toxicity filtering은 data 61%를 유지하고 평균 QA를 −2.7점(Wiki −3.8, Web −4.4) 바꾸며 toxicity identification도 낮춘다. Least toxic document를 제거하는 inverse filter(T = 0.06, 92% 유지)는 +1.7 및 최선 toxicity identification을 준다. T = 0.975 classifier quality filter(91% 유지)는 평균 +2.5이나 Books QA는 −2.2이며, “이득은 text characteristic에서 예측할 수 없다”(§1). **한계:** single-shot experiment, fine-tuned evaluation만(§8).

## §7 새 pipeline 읽기: 하나의 template에서 보는 네 pipeline

| 필드 | CCNet (2019년 2월) | C4 (2019년 4월) | Dolma v1.6 web | FineWeb (snapshot 96개) |
|---|---|---|---|---|
| Stage 순서 | paragraph dedup → LID → 5-gram PPL bucket | line/page rule, span dedup, langdetect (code상 순서: §3.1) | CCNet → URL → document dedup → quality → toxicity → paragraph dedup | WARC extraction → URL/LID/Gopher → snapshot별 MinHash → C4 rule → custom rule (논문; code는 다름, §5.6) |
| 순서 근거 | LID 전 dedup ablation (Fig. 3) | 없음 | 비용 논거; ablation하지 않음 | §3 순서의 반복 ablation |
| Granularity | paragraph (dedup); document (LID, bucket) | line 및 page | document, sentence (toxicity), paragraph (dedup), span (PII) | document (filter); duplicate cluster (MinHash); span (PII) |
| Threshold와 선택 | LID > 0.5; 언어별 tercile | 5 words, 3 sentences, langdetect ≥ 0.99; ablation하지 않음 | Gopher original; C4 NoPunc; token count를 위해 고른 τ = 0.4 | LID ≥ 0.65; histogram에서 custom threshold를 정한 뒤 28B ablation |
| 제거(단위) | dedup 뒤 char의 42%→28%가 남음(1→100 shard) | 1.4T → 156B SpaCy token (Dodge) | CCNet가 84.2% 제거(TB); 뒤 단계는 §4.1 참조 | 36T → 20T → 15T GPT-2 token |
| Ablation model | BERT-BASE, fastText embedding | T5 220M, 2^35 token | 1.2B, 150B token | 1.71B, 28B/350B token, seed 2개 |
| 평가 공백 | generative LM 없음; unfiltered-CC baseline 없음 | rule별 효과; dialect(후속 audit) | zero-shot task 8개; 숫자 없이 App. O figure로 제시한 Paloma subset | commonsense·knowledge 8-task aggregate; FineWeb 대 FineWeb-Edu에 대해서만 Paloma 점검 |

이 장에서 관측한 실패 각각에 연결한, 새 release를 위한 점검 항목은 다음과 같다.

1. 논문, datasheet, pinned script를 읽고 각 순서와 threshold를 따로 기록한다(FineWeb §5.6; C4 §3.1).
2. 단계를 비교하기 전에 모든 제거율을 한 단위로 변환한다(§3.3은 document의 7.6% = token의 21.2%임을 보인다).
3. threshold가 ablation, histogram, token budget 중 무엇으로 선택됐는지 묻는다(Dolma τ = 0.4; FineWeb §3.6).
4. Ablation token budget이 측정하려는 효과에 맞는지 점검한다. Deduplication 효과는 sample에 반복이 있을 때만 나타난다. FineWeb simulation에서는 반복이 약 100B token 부근에서 나타나기 시작한다(App. E.2).
5. Benchmark suite 바깥 domain에 대한 domain별 perplexity 또는 제거 audit을 찾는다(§6).

## Negative sample과 negative feedback

Heuristic pipeline은 style standard의 첫 번째 의미에서만 negative를 만든다. 즉 positive training target으로 사용했을 때 성능을 낮춘다고 판단된 document라는 **negative marginal value**다. 이 document들은 rule, word list, reference perplexity, hash collision, fastText classifier로 label되며, human 또는 verifier label은 관여하지 않는다. 현재 관행은 이를 버리는 것이지만, 정보를 유지하는 예외가 세 가지 있다. CCNet은 tail bucket을 제거하는 대신 release한다([[ccnet]] §5.2). Dolma는 span이 5개 이하인 document에서 PII span을 mask한다([[dolma]] §5.3). Dolma는 document 전체가 아니라 toxic sentence를 제거한다(§5.3). 어떤 stage도 제거한 text를 content, conditioning, gradient로 쓰지 않으므로, 이 stage에서는 likelihood-displacement 위험이 적용되지 않는다.

**잘못 제거된 document(보고됨).** Blocklist-excluded sample에서 “lesbian”을 언급한 document의 22%, “gay”를 언급한 document의 36%는 offensive도 sexual도 아니었다(Dodge et al. §5.2, [[c4]] 경유). CCNet은 tail에 유효한 spoken-like text 및 specialized forum이 있다고 보고한다(§5.2). FineWeb의 global deduplication은 자신이 남긴 data보다 더 좋은 model을 학습한 data를 제거했다(§3.4).

**제거한 text가 일부 capability에는 가치 있다는 근거.** 가장 toxic한 document를 유지하는 inverse toxicity filter는 toxicity identification에서 최고 성능과 평균 QA +1.7을 냈다([[pretrainers-guide-training-data]] §5). 두 연구는 서로 다른 방향을 가리킨다. Dolma에서 더 낮은 toxicity threshold는 sentence의 29.1–34.9%를 제거했고, 채택한 threshold가 제거한 5.5–7.3%보다 일반적으로 8-task suite에서 높은 score를 냈다([[dolma]] §5.3). Pretrainer's Guide에서 document의 39%를 제거한 T = 0.3 Perspective API filter는 unfiltered baseline에 비해 평균 QA를 2.7점 낮추고 toxicity identification도 낮췄다. Filter(sentence-level fastText 대 document-level Perspective API), model, evaluation이 다르므로, 제거 text의 가치는 측정하는 capability에 의존하며 capability별로 평가해야 한다.

**Control.** Filter score를 attribute로 저장하고 나중에 결정한다(CCNet bucket; Dolma toolkit은 `attributes/` 아래에 duplicate span을 쓴다, [[dolma]] Connections). 근거가 뒷받침하는 경우 더 작은 granularity를 선호한다(sentence-level toxicity removal, PII의 span masking). Filter별로 제거한 document의 random sample을 audit용으로 보관한다.

**Diagnostic.** Filter별 document 및 token 제거 count를 log한다. Topic별로 제거 document를 cluster한다(Dodge §5.1; [[fineweb]] §4.1). 제거 set과 유지 set에 dialect classifier를 실행한다(Dodge §5.3). Filter 적용 전후 domain별 perplexity를 비교한다([[paloma]]).

**Generality에 미치는 효과.** 제거는 제거한 domain의 coverage를 낮춘다(§6). Heuristic filter에 대해 이 장의 어떤 source도 calibration, hallucination, over-refusal에 미치는 효과를 보고하지 않는다.

## 레시피

아래 행은 data-pipeline 설정과 이를 선택한 ablation model이다. Style standard의 stage list에는 data-curation stage가 없으므로 pipeline 행에는 `pretrain-stable (data)`를 사용한다.

| Model(정확한 release) | 크기 | Stage | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| CCNet, 2019년 2월 snapshot | n/a (data) | pretrain-stable (data) | paragraph dedup 범위 | normalized paragraph의 SHA-1 (64 bits); shard 50개(corpus의 약 3%)에서 비교 | arXiv:1911.00359v2 §3.2, §4.2 | verified 2026-09-15 | Fig. 4–5: RAM 대비 남은 character(1 shard에서 42%, 100에서 28%) |
| CCNet | n/a (data) | pretrain-stable (data) | stage 순서 | LID 전에 dedup | §4.1, Fig. 3 | verified 2026-09-15 | Fig. 3: 2019년 2월의 1%에서 언어별 document 비율 |
| CCNet | n/a (data) | pretrain-stable (data) | LID threshold | fastText top-language score > 0.5 | §3.3 | verified 2026-09-15 | ablation 보고 없음 |
| CCNet | n/a (data) | pretrain-stable (data) | quality score | Wikipedia의 48개 언어로 학습한 SentencePiece + 5-gram KenLM; 언어별 tercile; 제거 없음 | §3.4, §5.2 | verified 2026-09-15 | Table 1(head > middle > tail analogy); Table 2(CC head 대 Wikipedia) |
| C4 (T5 논문) | n/a (data) | pretrain-stable (data) | line 및 page rule | line은 terminal punctuation으로 끝남; line당 ≥ 5 words; page당 ≥ 3 sentences; blocklist word, “lorem ipsum”, “{”가 있으면 page 제거 | arXiv:1910.10683v4 §2.2 | verified 2026-09-15 | Table 8: 모든 rule 대 rule 없음만 비교 |
| C4 (T5 논문) | n/a (data) | pretrain-stable (data) | language ID; dedup | langdetect English p ≥ 0.99; 반복된 세-sentence span은 하나 외 모두 제거 | §2.2 | verified 2026-09-15 | ablation 보고 없음 |
| C4 (TFDS code) | n/a (data) | pretrain-stable (data) | dedup 단위; 추가 rule | line-level dedup; “...”로 끝나는 line 제거; 1,000 character 초과 word는 해당 line 제거 | tensorflow_datasets/text/c4_utils.py (master, fetched 2026-09-14) L41–45, L244–248, L345–370 | conflict | 논문은 three-sentence span을 말함; code release date와 2019 build의 관계는 기재되지 않음 |
| T5 baseline | ~220M | pretrain-stable | data-ablation token | 2^35 ≈ 34B, 반복 없음 | §3.1.2, Table 8 | verified 2026-09-14 ([[c4]]) | SuperGLUE 71.36(C4) 대 68.04(unfiltered) |
| datatrove C4QualityFilter | n/a (data) | pretrain-stable (data) | default | min_num_sentences 5; min_words_per_line 3 | github.com/huggingface/datatrove@acd431b src/datatrove/pipeline/filters/c4_filters.py L66–67 | conflict | T5 §2.2와 다름; ablation 보고 없음 |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | LID | fastText English ≥ 0.5 (byte 61.7% 제거) | arXiv:2402.00159v2 §5.1 | verified 2026-09-14 ([[dolma]]) | ablation 보고 없음 |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | quality rule | Gopher All + C4 NoPunc | §5.2 | verified 2026-09-14 | Fig. 1: C4 NoPunc > C4 All, Gopher All; 조합이 최고(1.2B, HellaSwag 제시) |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | C4 NoPunc granularity | punctuation으로 끝나지 않는 paragraph(§5.2) / punctuation으로 끝나지 않는 line이 절반 초과인 document(App. N.4) | §5.2; App. N.4 | conflict | 양쪽 모두 character의 22.73% |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | toxicity threshold | τ = 0.4, sentence 제거 | §5.3 | verified 2026-09-14 | Fig. 2: τ = 0.0004가 일반적으로 더 좋음; 0.4는 token count를 충족하려고 선택 |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | PII rule | ≤ 5 span은 mask; ≥ 6 span이면 document 제거 | §5.3 | verified 2026-09-15 | 제거 대 replacement: 효과를 측정하지 않음 |
| Dolma v1.6 web | n/a (data) | pretrain-stable (data) | dedup 순서 | URL → document → (quality, content) → paragraph, Bloom filter | §5.4, §5.5 | verified 2026-09-14 | 효율성 논거; 순서를 ablation하지 않음 |
| Dolma data-ablation model | 1.2B | pretrain-stable | token; eval | 150B token; zero-shot task 8개 | §4.2; App. D.1 | verified 2026-09-14 | ablation 보고 없음 |
| FineWeb | n/a (data) | pretrain-stable (data) | extraction | WARC에 trafilatura | arXiv:2406.17557v2 §3.2 | verified 2026-09-15 | Fig. 1: 28B token에서 WARC+trafilatura > WET |
| FineWeb | n/a (data) | pretrain-stable (data) | LID; base quality | fastText English ≥ 0.65; MassiveText quality 및 repetition, original threshold; URL blocklist | §3.3 | verified 2026-09-15 | Fig. 2: base filtering > unfiltered WARC(조합, rule별 아님) |
| FineWeb | n/a (data) | pretrain-stable (data) | MinHash | word 5-gram; hash 112개 = bucket 14개 × 8; snapshot별 | §3.4, App. E.1; datatrove@acd431b examples/fineweb.py L80–88 | verified 2026-09-15 | Fig. 3–5, 15: global 및 가벼운 global 방법이 snapshot별보다 낮음 |
| FineWeb | n/a (data) | pretrain-stable (data) | C4 rule | terminal punctuation 외 모두 | §3.5 | verified 2026-09-15 | Fig. 6: punctuation 제외 모두(약 7% 제거) > terminal punct 단독(약 30% 제거) |
| FineWeb | n/a (data) | pretrain-stable (data) | custom rule | line-punct ratio ≤ 0.12; short-line(< 30 chars) ratio ≥ 0.67 | §3.6; fineweb_quality_filter.py L14–16 | verified 2026-09-15 | Fig. 7, App. E.4: rule 3개가 약 22% 제거, 28B에서 aggregate +약 1% |
| FineWeb | n/a (data) | pretrain-stable (data) | duplicated-line character ratio | ≥ 0.1(§3.6 text) / 0.01(App. E.4 Table 2; code L18) | §3.6; App. E.4; datatrove@acd431b | conflict | Table 2 행에서 token 12.47% 제거 |
| FineWeb data-ablation model | 1.71B | pretrain-stable | token; seed | 약 28B(filter), 350B(dedup, cumulative); data version당 run 2회 | §3.1 | verified 2026-09-14 ([[fineweb]]) | 28B는 1.71B에 약 Chinchilla-optimal이라고 기술 |

**작은 general-purpose run의 시작점.** 아래의 모든 값은 위에서 verified된 행과 그 source 조건에서 왔다. Snapshot이 하나보다 많은 English web text라면 WARC에서 trafilatura로 text를 extract하고, fastText English score ≥ 0.65인 document를 유지하며, original threshold에서 Gopher quality 및 repetition rule을 적용한다(FineWeb, snapshot 96개, parameter 1.71B 및 token 28B에서 선택). Word 5-gram과 8 hash의 bucket 14개로 각 snapshot을 따로 deduplicate한다(FineWeb, token 350B에서 비교). Terminal punctuation을 제외한 C4 rule, FineWeb line-punctuation(≤ 0.12) 및 short-line(≥ 0.67) rule을 적용한다. Duplicated-line threshold는 conflict 행이므로 자체 ablation이 필요하다. Multilingual corpus라면 language identification 전에 paragraph를 deduplicate하고 score가 0.5를 넘을 때만 page를 top language에 할당한다(CCNet, 2019년 2월 snapshot). Span이 5개 이하인 document에서는 PII span을 mask하고, 6개 이상인 document는 제거한다(Dolma, 1.2B에서 측정된 효과 없음). FineWeb 및 Dolma 값은 parameter 1.2–1.71B에서 commonsense 및 knowledge benchmark로 선택했다. Code, LaTeX, dialectal text를 포괄해야 하는 model에 이 값을 채택하기 전에 domain별 perplexity를 점검한다.

## 일반화 관점

**(a) Breadth를 늘리는 것.**
- Language identification 전에 반복 boilerplate를 deduplicate하면 low-resource language document를 더 많이 유지한다([[ccnet]] §4.1, Fig. 3; Result, 단일 연구).
- Perplexity bucket과 filter score를 label로 release하면 이후 mixture가 spoken-like comment 및 specialized forum 같은 tail text를 회복할 수 있다([[ccnet]] §5.2).
- Filtered web text를 curated non-web source와 결합하면 diverse domain에 대한 fit([[dolma]] §9.2)과 downstream QA([[pretrainers-guide-training-data]] §6)가 개선된다(방향이 **Replicated**됨).
- Snapshot별 MinHash는 20T token을 유지하고 RefinedWeb과 일치한 반면, global MinHash는 4T token을 유지하며 이득이 거의 없었다([[fineweb]] §3.4).
- 더 작은 제거 granularity는 document의 나머지를 유지한다. Dolma는 document를 버리는 대신 toxic sentence를 제거하고 PII span을 mask한다([[dolma]] §5.3).

**(b) Narrowing을 일으키는 것.**
- Word blocklist는 dialect 및 identity-related text를 불균등한 비율로 제거한다. AAE 42%, Hispanic-aligned English 32%, White-aligned 6.2%다([[c4]], Dodge et al. §5.3).
- Code-removal rule 및 single-source web data는 LaTeX와 code domain에 gap을 남긴다. C4 baseline perplexity는 RedPajama arXiv에서 391,171이다([[paloma]] §4.1; `{` rule과의 연결은 Interpretation).
- Reference-anchored score는 reference corpus의 author와 register를 선호한다([[ccnet]] §5.2; [[whose-language-counts-quality-filter]] §3.4).
- Toxicity filtering은 toxicity identification과 평균 QA를 낮춘다([[pretrainers-guide-training-data]] §5, Fig. 5, 7).
- English-only language identification은 non-English ability를 residual text에 의존하게 한다. Dolma 저자는 corpus가 NLP에서 English가 “default” language라는 기대를 “reinforces”한다고 말한다([[dolma]] Limitations).
- 100K word를 넘는 document를 제거하는 Gopher rule([[dolma]] App. N.4)은 가장 긴 document를 제거한다. 이 장의 어느 source도 long-context ability에 미치는 효과를 보고하지 않는다.
- HellaSwag-type benchmark에서 선택한 threshold는 그 benchmark가 보상하는 것을 위해 선택된다. FineWeb은 small scale에서 variance가 낮다는 이유로 benchmark를 골랐으며([[fineweb]] §3.1), 이는 coverage criterion이 아니라 statistical criterion이다(Interpretation).

**(c) 이 stage에서의 측정 방법.**
- Decontaminated training data로 domain별 macro perplexity를 측정한다([[paloma]] G1–G5). FineWeb §4.2 비교는 의도적으로 decontamination을 생략했으므로 domain-fit 수치에는 가능한 overlap이 포함된다.
- Removal audit: 제거 document의 topic cluster, identity-term PMI, filter 각각의 전후 dialect share(Dodge et al. §5, [[c4]] 경유).
- 효과에 맞춘 token budget의 data ablation: filter에는 token 28B, deduplication에는 350B([[fineweb]] §3.1, App. E.2), seed는 둘 이상.
- 알려진 measurement error: Paloma 단일 source 중 시험한 downstream task 8개 모두와 상관하는 것은 없다([[paloma]] App. A). 인접 checkpoint 간 ranking agreement 평균은 0.513이다([[paloma]] App. A). Dolma의 filter ablation은 curve로만 보고된다(main text의 HellaSwag, 다른 task와 Paloma subset은 App. O). FineWeb의 것은 aggregate score curve이므로 effect size는 figure에서 읽어야 한다.

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| CCNet을 LID → dedup → perplexity로 재구현 | CCNet 통계보다 low-resource language의 document가 적음 | 1% sample에서 두 순서를 모두 실행하고 언어별 document count를 비교(CCNet Fig. 3 방법) |
| 이름만 보고 “C4 filter”를 복사 | reference와 제거율이 다르며 short line 또는 page가 예기치 않게 유지·제거됨 | threshold를 출력; T5 §2.2(5 words, 3 sentences)와 datatrove default(3 words, 5 sentences)를 비교 |
| 서로 다른 단위의 제거율을 비교 | “7.6%”를 제거한 filter를 사소하다고 취급 | tokenizer 이름과 함께 document 및 token을 보고; Dodge Table 1은 document 7.6% = token 21.2%를 제시 |
| 많은 snapshot에 global MinHash | 오래된 snapshot이 약 10% 이하로 줄고 benchmark 이득이 작음 | 350B token에서 snapshot별 대 global ablation; 유지 sample에서 ad와 keyword list 점검 |
| 1B-token sample로 deduplication 측정 | Dedup variant 사이에 차이가 없음 | Duplicate-cluster size를 직접 count; FineWeb처럼 350B token에서 dedup variant 평가(§3.1; App. E.2 simulation) |
| 한 benchmark family에서 threshold 선택 | Benchmark 변화 없이 code, LaTeX, dialect domain fit이 악화 | Filter 전후 domain별 Paloma macro perplexity |
| Page-level word blocklist | AAE, Hispanic-aligned English, LGBTQ-related document가 불균형하게 제거 | 제거 대 유지 document에 dialect classifier와 identity-term PMI(Dodge §5) |
| Reference-perplexity cutoff로 document 제거 | Spoken-like text 및 specialized forum의 손실 | Tail document를 sample하여 읽고, bucket을 attribute로 유지(CCNet §5.2) |
| 논문의 stage 순서가 release 순서라고 가정 | 재현한 token count가 release와 다름 | 논문, datasheet, pinned code를 비교(FineWeb §3.7, App. A, examples/fineweb.py) |

## 이해도 점검

1. CCNet은 language identification 전에 paragraph를 deduplicate한다. 이 순서가 low-resource language에 할당되는 document 수를 어떻게 바꾸는지 설명하고, deduplication을 shard 50개가 아니라 하나로 제한하면 변화 방향을 예측하라.
2. Dodge et al.의 Table 1에서 blocklist는 document의 7.6%, token의 21.2%를 제거한다. Page-level word rule의 어떤 속성이 이 차이를 만들며, 이것이 long-document coverage에 무엇을 뜻하는지 설명하라.
3. FineWeb의 global MinHash는 2013-48에서 약 31B token을 유지했는데, 이는 제거한 data에서 다시 구성한 171B token보다 나쁜 model을 학습했다. Snapshot을 newest-to-oldest로 처리하는 것이 어떤 old page를 생존시킬지 어떻게 정하는지, 그리고 1B-token ablation이 왜 문제를 발견하지 못했을지 설명하라.
4. FineWeb의 hash 112개에서, 8개씩 14 bucket인 경우와 14개씩 8 bucket인 경우에 대해 P(s = 0.6)를 계산하라. 어느 설정이 더 많은 near-duplicate를 찾는지와, 낮은 similarity에서 false match가 늘어나는 비용을 설명하라.
5. Dolma는 C4 NoPunc만 유지했고 FineWeb은 terminal punctuation을 제외한 모든 C4 rule을 유지했다. 이를 모순된 result로 읽지 못하게 하는 setup 차이를 열거하고, 둘 중 무엇을 택할지 결정할 ablation 하나를 설계하라.
6. Filter가 parameter 1.7B에서 commonsense benchmark 8개의 평균을 높인다. 같은 filter가 arXiv 및 programming-language text에 대한 fit을 어떻게 낮출 수 있는지, 이를 발견할 §6의 measurement가 무엇인지 설명하라.
7. Pretrainer's Guide의 inverse toxicity filter는 toxicity identification을 개선했다. Negative marginal value의 정의를 사용하여, 이 result가 general-purpose model을 위한 negative로 제거 document에 label을 붙이는 일에 관해 무엇을 말하는지 설명하라.

## 연결

- **이전:** ch-09 — Pretraining Data Composition and Capability Coverage(source-level composition; 이 장은 web source 내부에서 선택한다).
- **다음:** ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection(FineWeb-Edu, DCLM, Nemotron-CC classifier; §6 measurement를 사용).
- **§4.1 PII rule 확장:** ch-11 — Tokenizers, Data Provenance, and PII Removal.
- **§2.1 및 §5.3–§5.4 확장:** ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic(MinHash parameter, Llama 3와 비교한 global 대 snapshot별 deduplication).
- **§2 language identification 확장:** ch-13a — Multilingual Coverage and Vocabulary as Capability Axes(FineWeb2 언어별 filter).
- **§6의 measurement 배경:** ch-00 — What General Capability Means and How It Is Measured(Paloma domain별 perplexity).
- **이 장을 적용하는 lab:** ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement.

## 출처

- [[ccnet]] — stage 순서 및 그 ablation(§4.1), paragraph dedup 범위(§3.2, §4.2), LID threshold(§3.3), KenLM bucket 및 no-removal 결정(§3.4, §5.2), Table 1–2.
- [[c4]] — §2.2 rule 목록, filtered 대 unfiltered Table 8, TFDS code pointer, 그리고 Connections에 기록된 Dodge et al. 2021 audit 수치(Table 1, §5).
- [[dolma]] — web pipeline 순서 및 제거율(§5), quality 및 toxicity ablation(Figure 1–3), PII rule, source별 pipeline, domain fit(§9.2), limitation.
- [[fineweb]] — ablation protocol(§3.1), extraction 및 base filtering, MinHash parameter와 dedup 연구(§3.4, App. E), C4 재-ablation 및 custom rule(§3.5–§3.6), 최종 순서(§3.7), topic 및 domain analysis(§4.1–§4.2).
- [[minhash-lsh]] — Jaccard resemblance, shingling, min-hash estimator(Theorem 1).
- [[paloma]] — perplexity 및 macro-average 정의, single-source baseline spike(§4.1), measurement caution(App. A).
- [[whose-language-counts-quality-filter]] — reference-anchored quality score와 그 demographic·factuality alignment(§3.2–§4.2); chapter excerpt, 2026-09-15 현재 library card 없음.
- [[pretrainers-guide-training-data]] — parameter 1.5B에서 toxicity 및 quality filter trade-off와 domain-removal effect(§5–§6, Figure 5–7); chapter excerpt, 2026-09-15 현재 library card 없음.
