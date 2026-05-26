<!-- chapter: ch-09
     track: data
     title: Training Data Landscape
     sources: [[the-pile]], [[c4]], [[ccnet]], [[dolma]], [[fineweb]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[qwen-3]], [[qwen-3-5]], [[deepseek-v3]], [[scaling-laws-data-quality]]
     figures: figures/data-landscape.html
-->

# 9장 — 학습 데이터 지형

> **핵심 통찰.** Pretraining corpus는 텍스트 더미가 아니다. 그것은 *composition*이다. 각자 고유한 license, domain, freshness, failure mode를 가진 source들의 집합이며, 명시적인 mixture policy 아래 조립된 것이다. 눈길을 끄는 숫자들(15T token, 36T token, 671B MoE param)은 모두 mixture decision 위에 놓여 있다. 2020년에는 모든 lab이 이를 공개했지만, 2026년에는 거의 아무도 공개하지 않는다. The Pile과 Llama 3 사이의 가장 중요한 변화는 scale이 아니다. *release가 데이터에 대해 무엇을 말해 주는가*다.
>
> **가이드라인.** 두 모델을 비교하기 전에 네 축으로 corpus를 비교하라. (1) token count, (2) source composition(web / code / math / books / encyclopedic / synthetic), (3) license regime(research-only / permissive / proprietary / opted-out), (4) disclosure granularity(full mix / stage-level / headline-only / none). Matched-composition ablation을 견디지 못하는 성능 차이는 algorithm이 아니라 data의 artifact다.

---

## 이 장이 존재하는 이유

1-8장은 trainer를 만들었다. optimizer moment, mixed precision, LR schedule, packed SFT, FSDP shard, checkpointing, silent-failure mode, 그리고 작동하는 minimal run이 그것이다. 이 모든 주제는 디스크 어딘가에 *어떤* pretraining corpus가 있다고 가정한다. 이 장은 그 corpus에 대한 것이다. 그것이 실제로 무엇인지, 2020-2026년 open work를 지배하는 corpus는 무엇인지, 왜 frontier lab들이 자신들의 corpus에 무엇이 들어 있는지 말하지 않게 되었는지, disclosure가 의도적으로 얇더라도 model report에서 data discipline을 어떻게 읽을지 다룬다.

Data track(ch-09..ch-17)은 여기서 아래로 내려가는 구조다. ch-10은 open-pipeline deep dive(CCNet, C4, Dolma, FineWeb을 cookbook으로 읽기)이고, ch-11은 tokenizer/shard/lineage operation이다. 이후 장들은 domain balance, decontamination, long-context mix, lab을 다룬다. 그 전에 지도가 필요하다.

세 가지 변화가 2020 → 2026 기간을 정의하며 track의 나머지를 framing한다.

1. **Scale.** Open corpus는 800 GiB([[the-pile]], 2020)에서 15T token([[fineweb]], 2024)으로 증가했다. 대략 두 자릿수 규모다. Frontier closed corpus는 ~1T([[c4]]/RedPajama era)에서 36T([[qwen-3]])로 갔다.
2. **Composition.** The Pile의 22개 hand-curated subset은 classifier-filtered web text가 지배적 ingredient가 되고 code + math + synthetic slice가 덧붙는 방식으로 대체되었다.
3. **Disclosure.** 2020-2022년에는 모든 data slice가 이름과 weight를 가졌다. 2024-2026년에는 [[llama-3]], [[qwen-3]], [[deepseek-v3]], [[qwen-3-5]]가 token total은 보고하지만 mix percentage는 보고하지 않는다. [[dolma]], [[fineweb]], OLMo line은 transparency counter-movement다.

이 장은 ch-10..ch-17의 vocabulary다. §2의 table을 이 장의 spine으로 다루라.

---

## 1. CommonCrawl과 Wikipedia — 원시 substrate

어떤 corpus보다 앞서, 다른 모든 것이 파생되는 두 가지 주요 upstream source가 있다. CommonCrawl과 Wikipedia다. 이들은 직접 학습하는 corpus가 아니다. 모든 public pipeline이 처리하는 *substrate*다.

**CommonCrawl (CC)**은 open web을 crawl하는 non-profit이며, 월별 "snapshot"("dump"라고도 부름)을 WARC format으로 공개한다. 각 snapshot은 raw로 약 300 TB이고 약 3B document를 포함한다. [[fineweb]]은 15T-token output을 위해 **96개 snapshot**을 처리한다. [[ccnet]]의 pipeline은 "CC dump를 English training text로 바꾸는" canonical recipe다. Text extraction(Trafilatura 또는 custom), language ID(fastText), deduplication, reference corpus에 대한 quality scoring으로 구성된다.

알아야 할 두 가지 operational fact:

- **CC는 단일한 것이 아니다.** 각 snapshot은 다른 crawl에서 나온다. Snapshot 간 near-duplicate가 흔하다(같은 URL이 다시 crawl됨). 그래서 [[fineweb]]의 놀라운 발견, 즉 *per-dump MinHash가 global MinHash를 이긴다*는 점이 중요하다. Global dedup은 "snapshot마다 한 번씩 다시 나타나는 high-quality content" signal(예: canonical reference page)을 지워 버린다.
- **CC는 여러분이 법적으로 접근할 수 있는 것이다.** 대안인 자체 web crawl은 robots.txt, copyright, anti-bot infrastructure 영역으로 들어간다. CC의 license(CC0에 가까우며 원 host를 존중)는 open pretraining이 존재할 수 있는 유일한 이유다.

**Wikipedia**는 다른 substrate다. CC에 비해 Wikipedia는 작지만(English 약 5B token), 세 가지 이유로 자기 weight보다 훨씬 큰 영향을 낸다. (i) CC-BY-SA licensed이고, 깨끗하며, 거의 canonical한 reference text다. (ii) *Quality anchor*다. [[ccnet]]은 Wikipedia-trained KenLM의 perplexity로 CC document를 score하고, 이후 pipeline도 Wikipedia-style text를 positive class로 사용한다. (iii) 모든 frontier model은 모든 Wikipedia article을 여러 번 읽었으므로, signal-rich ingredient인 동시에 over-memorization risk가 큰 ingredient다.

각각이 빠뜨리는 것:

- **CC가 빠뜨리는 것**: paywalled content, 많은 book, 대부분의 forum archive(rate-limited), code repository(GitHub가 primary source), academic PDF(부분적이고 extraction이 부족함), login 뒤의 모든 것, quality parity의 대부분 non-English content(English는 crawl design상 과대표집됨).
- **Wikipedia가 빠뜨리는 것**: current event(editorial lag), non-Anglophone depth(English Wikipedia는 다음으로 큰 German의 약 2.5배), conversational / colloquial text.

아래의 모든 open corpus는 CC + Wikipedia에서 시작한 뒤 *더하고 뺀다*. Addition(code, books, papers, forums, synthetic)과 subtraction(filters, dedup, content removals)이 recipe들을 구분한다.

---

## 2. 비교 표 — open corpus, 2020-2026

이 장의 중심 artifact다. 각 row는 released frontier 또는 frontier-adjacent model이 학습한 corpus다. Column은 Guideline의 네 축이다.

| Dataset | Year | Tokens (approx.) | Docs | Licence | Source mix (공개된 경우 token 기준 %) | Disclosure |
|---|---|---|---|---|---|---|
| **CommonCrawl (raw)** | 2008- | snapshot 연간 ~100-1000 T(uncleaned) | snapshot당 ~3 B | CC0-adjacent | 100% web | upstream only — corpus가 아님 |
| **Wikipedia (EN)** | continuous | ~5 B | ~7 M articles | CC-BY-SA 3.0 | 100% encyclopedic | full dump 공개 |
| **C4** ([[c4]]) | 2019 | ~180 B | ~365 M | CC에서 파생, terms는 CC를 따름 | heuristic filter 이후 100% web | full recipe 공개; document inspect 가능 |
| **The Pile** ([[the-pile]]) | 2020 | ~300 B (825 GiB) | 22 subsets | mixed; 일부 subset은 논쟁적(Books3) | 24% web (Pile-CC) · 13% PubMed Central · 9% arXiv · 7% GitHub · 5% FreeLaw · 4% StackExchange · 3% USPTO · 3% PubMed abstracts · 2.5% Books3 · 13개 더 작은 subset | 22-domain weight 전체 공개 |
| **RedPajama v1** | 2023 | ~1.2 T | ~930 M | source별 license를 따름 | 67% CC · 15% C4 · 4.5% GitHub · 4.5% books · 2.5% arXiv · 2% Wikipedia · 2% StackExchange | Together AI가 full table 공개 |
| **SlimPajama** | 2023 | ~627 B | ~590 M | RedPajama와 동일 | RedPajama에서 global near-dupe 제거(50% 작음) | 공개됨; dedup stats public |
| **Dolma v1.7** ([[dolma]]) | 2024 | ~3 T | ~2 B | source별, full `LICENSE` manifest | ~80% CC (web) · ~8% code (The Stack subset) · ~5% academic (peS2o) · ~3% Reddit · ~2% books (Gutenberg) · ~2% Wikipedia | 모든 filter threshold + ablation 공개; `dolma` CLI open source |
| **OLMo-Mix-1124** ([[olmo-2]]) | 2024 | ~3.9 T | n/a | DCLM + Dolma 1.7 + Starcoder + Proof Pile II에서 파생 | component별 공개, percentage는 Dolma보다 덜 세밀함 | stage 1 mix 공개; 50B Dolmino cooldown은 별도의 higher-quality slice |
| **FineWeb** ([[fineweb]]) | 2024 | ~15 T | ~22 B | CC-derived; `datatrove` recipe public | classifier 이후 100% web(96 CC dumps) | full pipeline + ablation + per-dump MinHash decision 공개 |
| **FineWeb-Edu** ([[fineweb]]) | 2024 | ~1.3 T | ~1.5 B | CC-derived | 100% web, score≥3 educational classifier(Llama-3-70B annotation) | classifier + 450K labelled sample 공개 |
| **Dolma 3** ([[olmo-3]]) | 2025 | ~9.3 T source / ~5.9 T mix | n/a | source별 | Dolma 3 Mix는 1.7보다 math/code를 강조; 100 B Dolmino mid-training; 50 B Longmino long-context | full *curriculum*을 stage별 공개 |
| **Llama 3 pretraining** ([[llama-3]]) | 2024 | 15.6 T | not disclosed | proprietary | *not disclosed* (paper는 "high-quality"라고 말하고 capability category를 나열) | token total만 공개, mix 없음 |
| **Qwen3 pretraining** ([[qwen-3]]) | 2025 | 36 T | not disclosed | proprietary | 세 stage 외에는 *not disclosed*: ~30T general → ~5T reasoning → long-context | stage budget만 공개, source mix 없음 |
| **Qwen3.5 pretraining** ([[qwen-3-5]]) | 2026 | not disclosed | not disclosed | proprietary | 전혀 disclosed 아님 — technical report 없음 | headline model만 있음 |
| **DeepSeek-V3 pretraining** ([[deepseek-v3]]) | 2024 | 14.8 T | not disclosed | proprietary | *not disclosed* | token total + capability claim만 있음 |

Table은 가로보다 세로로 먼저 읽어라. "Disclosure" column이 이 장의 실제 주제다. 모든 open row는 mixture를 공개했고, 모든 closed row는 공개하지 않았다. Table 위에서 아래로의 변화가 §3-§7의 나머지가 설명하는 변화다.

---

## 3. The Pile의 22개 subset — 무엇이 잘 늙었고, 무엇은 그렇지 않은가

[[the-pile]](Gao et al., 2020)은 *mixture argument*로 명성을 얻었다. Diversity가 scaling variable이라는 주장이다. Source에서:

> The Pile은 academic text, code, books, web text, forum을 포괄하는 22개의 다양한 high-quality subset으로 만든 825 GiB English text corpus다. 핵심 주장은 source diversity가 generic crawl baseline에 비해 cross-domain performance를 실질적으로 개선한다는 것이다.

대략적인 token share에 따른 22개 subset(EleutherAI가 공개한 weight):

```
Pile-CC            227 GB   (18.1%)  CC-derived web text
PubMed Central     96 GB    ( 7.6%)  biomed full-text paper
Books3             101 GB   ( 8.1%)  Bibliotik shadow-library book
OpenWebText2       63 GB    ( 5.0%)  Reddit에서 upvote된 web page
arXiv              56 GB    ( 4.5%)  math/physics/CS paper
GitHub             95 GB    ( 7.6%)  code
FreeLaw            51 GB    ( 4.1%)  미국 legal case
StackExchange      32 GB    ( 2.6%)  약 350개 site의 Q&A
USPTO              22 GB    ( 1.7%)  patent grant
PubMed Abstracts   19 GB    ( 1.5%)  biomed abstract
Gutenberg (PG-19)  11 GB    ( 0.9%)  public-domain book
OpenSubtitles      13 GB    ( 1.0%)  movie subtitle
Wikipedia (EN)     6  GB    ( 0.5%)  encyclopedic text
DM Mathematics     8  GB    ( 0.7%)  synthetic math problem
Ubuntu IRC         5  GB    ( 0.4%)  developer chat
BookCorpus2        6  GB    ( 0.5%)  novel (BookCorpus follow-up)
EuroParl           4  GB    ( 0.3%)  EU parliamentary proceeding
HackerNews         4  GB    ( 0.3%)  startup/tech forum
YouTube Subtitles  3  GB    ( 0.3%)  auto-generated caption
PhilPapers         2  GB    ( 0.2%)  philosophy paper
NIH ExPorter       2  GB    ( 0.1%)  grant abstract
Enron Emails       0.9 GB   ( 0.1%)  email corpus
```

(Percentage는 token-weighted다. The Pile은 일부 subset에 명시적 upsampling을 적용하므로 "training 중 weight" ≠ "raw corpus share"다.)

**잘 늙은 것:** arXiv, PubMed Central, GitHub, StackExchange, Wikipedia, FreeLaw, USPTO. 모든 2024-2026 open mix는 여전히 이들의 직접적인 analog를 포함한다. arXiv paper는 peS2o([[dolma]])의 일부가 되었고, GitHub는 The Stack(2024년 기준 3T token)으로 흡수되고 확장되었으며, StackExchange는 사실상 모든 chat/code pretraining에서 살아남았다. 공통 속성은 *structured, license-clean, long-document, domain-specialist text*다.

**잘 늙지 않은 것:**

- **Books3** — 2.5% books slice는 shadow library인 Bibliotik에서 왔다. 이것은 2023년 Rhode Island lawsuit와 여러 후속 소송의 중심 주제다. 2024년까지 모든 production derivative에서 제거되었다(RedPajama와 Dolma는 Project Gutenberg로 대체했고, commercial lab들은 아마 licensed deal을 가졌을 것이다). Books3의 부재는 open pretraining data에서 *가장* 큰 license-regime event다. 또한 2024 open corpus가 closed corpus에 비해 long-form literary text에서 체계적 약점을 갖는 이유이기도 하다.
- **OpenSubtitles** — machine-generated subtitle text는 timing artifact와 truncated line 같은 quality issue가 있다. 2020 recipe는 이를 허용했지만 2024 classifier는 걸러낼 것이다. Derivative에서 제거되었다.
- **Enron emails / Ubuntu IRC** — 특정 domain의 human chat data. 2024+ scale에서는 synthetic chat data보다 성능이 낮다.
- **YouTube Subtitles** — auto-caption quality issue. 대부분의 derivative에서 제거되었다.
- **DM Mathematics** — synthetic math *problem*(step-by-step reasoning 없음). Real human math(MATH, GSM8K)와 [[qwen-3]]의 Qwen2.5-Math synthetic chain-of-thought로 대체되었다.

교훈: **hand-curated diversity는 작동하지만, hand-curated source는 유지 가능해야 한다.** The Pile은 one-shot release였다. 그 subset들은 이제 5년 됐다. [[dolma]], [[fineweb]], OLMo line은 release가 아니라 *pipeline*이다. Fresh CC snapshot에서 다시 실행하고 mix를 다시 도출한다.

---

## 4. 2020 → 2026 변화 — raw-web maximalism에서 web+code+synthetic으로

Frontier pretraining mix의 composition은 세 개의 뚜렷한 phase로 진화했다.

**Phase 1 (2018-2021): C4, GPT-3 era.** Web text가 지배한다. Code는 작은 side channel이다. T5/GPT-3는 books와 Wikipedia의 약간의 boost를 제외하면 ~99% web-derived text로 학습했다. [[c4]]가 대표적이다. CC에 대한 aggressive heuristic filtering, code 없음, math 없음, synthetic 없음.

**Phase 2 (2020-2023): The Pile, RedPajama, Llama 1/2 era.** Hand-curated mixture가 표준이 된다. 명시적 domain subset(code ~5-10%, academic ~5%, books ~5%, web ~60-70%, forums/misc ~10%). Llama 1은 mix를 공개했다(CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, Gutenberg+Books3 4.5%, arXiv 2.5%, StackExchange 2%). Llama 2도 scale adjustment가 있는 같은 구조를 공개했다.

**Phase 3 (2024-2026): classifier-filtered web + heavy code + synthetic, mix undisclosed.** [[fineweb]]은 CC에 대한 단일 LLM-labeled quality classifier가 MMLU에서 모든 heuristic stack을 이긴다는 것을 보였다. 그리고 갑자기 web text가 돌아왔다. 다만 *filtered* web text다. [[qwen-3]]는 **36T token 중 ~5T가 Qwen2.5-Math와 Qwen2.5-Coder에서 온 synthetic math와 synthetic code를 포함하는 전용 "reasoning stage"**라고 공개한다. Source에서:

> Data expansion은 다음을 포함한다.
>   - Qwen2.5-VL을 사용한 대규모 PDF corpus의 OCR-style text extraction
>   - Qwen2.5-Math의 synthetic math data
>   - Qwen2.5-Coder 및 관련 model의 synthetic code/data variant
> Report는 data가 educational value, domain, safety에 대해 대규모로 annotate된 뒤 proxy-model ablation을 사용해 instance level에서 mixed된다고 말한다.

[[llama-3]]는 "high-quality" web, code, mathematical data로 15.6T token을 pretrain했다. Report는 source-mix table을 전혀 제공하지 않고 *capability*(multilinguality, coding, reasoning, tool use)를 설명한다. [[deepseek-v3]]는 mix percentage 없이 14.8T token만 공개한다. [[qwen-3-5]]는 token도 mix도 공개하지 않는다. 2026 frontier non-disclosure의 극단이다.

**Frontier lab들이 disclosure를 중단한 이유.** 솔직함의 순서로 네 가지 이유가 있다.

1. **Competitive moat.** Mixture는 model의 capability profile을 재현하는 데 가장 actionable한 signal이다. 공개하는 것은 competitor에게 recipe를 주는 것이다.
2. **Synthetic data attribution.** Pretraining mix의 ~30%가 *자신의 이전 model version이 생성한 것*이면, "mix를 공개한다"는 것은 generator의 output을 공개한다는 뜻이 되며, 그 output 자체가 proprietary다. 이 recursion은 transparency를 막는다.
3. **License liability surface.** Books3는 published-mix-percentage가 litigation의 target을 만든다는 사실을 lab들에게 가르쳤다. Percentage를 공개하지 않으면 discovery는 상대방의 문제가 된다.
4. **Opt-out register.** 2023-2025년에는 opt-out mechanism(NYT의 2023 robots.txt directive, CC의 "noai" convention, ai.txt proposal)이 등장했다. Mix를 공개하는 lab은 "X가 opt out했는데 왜 포함했나"라는 질문에 노출된다. 조용한 default가 더 안전하다.

비대칭에 주목하라. [[olmo-2]]와 [[olmo-3]]는 시간이 갈수록 더 *많은* detail을 공개한다(Dolma 3는 단순한 mix가 아니라 staged curriculum이다). 반면 Qwen과 DeepSeek는 점점 더 적게 공개한다. 업계는 둘로 갈라지고 있다.

---

## 5. License와 governance — copyright, opt-out, enterprise-data constraint

Data license는 깨끗한 engineering picture가 지저분한 legal reality와 만나는 곳이다. 세 가지 regime이 지배적이다.

**Regime A: CC-BY-SA / CC0 / public domain.** Wikipedia(CC-BY-SA 3.0), Project Gutenberg(public domain), Common Crawl(crawl time에 robots.txt를 존중함. downstream use는 별도 문제), arXiv(author license 다양, de facto scraped). 법적으로 가장 안전한 tier지만 가장 작기도 하다. Wikipedia는 ~5B token, Gutenberg는 ~11B token이다. 현실적 scale의 model은 Regime A를 주로 사용해 학습할 수 없다.

**Regime B: per-document mixed license, derivative risk.** Common Crawl output, GitHub scrape(MIT부터 GPL, unlicensed, "all rights reserved"까지 수백만 개의 서로 다른 repo license), Reddit(post는 Reddit user agreement 아래 CC-BY-4.0이지만 2023 API monetization과 이후 legal action이 downstream republication을 복잡하게 만든다). 대부분의 2020-2024 open corpus는 Regime B에서 작동하며 LLM training의 법적 근거로 *fair use*에 의존한다. Authors Guild, NYT, Getty, Universal Music lawsuit가 도전하는 것이 이것이다.

구체적인 Regime B artifact:

- **The Stack (BigCode)**는 GitHub를 license로 filter한다. Permissive license(MIT/Apache/BSD)가 있는 repo만 포함하고, project는 **opt-out register**를 추가했다(contributor가 exclusion을 요청할 수 있음). The Stack v1.2는 약 3 TB code이고, 2024년에 나온 The Stack v2는 약 67 TB로 확장되었다.
- **Books3**는 그 반대였다. License defense가 없는 shadow-library book이다. 2023 Rhode Island filing은 이것을 Regime D(아래 참조)였어야 할 Regime B inclusion의 대표 사례로 만들었다.
- **FineWeb / FineWeb-Edu**는 CC의 licensing posture를 상속한다. `datatrove` codebase에는 opt-out request를 구현하는 URL-filter list가 포함되어 있다.

**Regime C: licensed content.** Frontier lab들은 특정 corpus를 사거나 license한다. Reddit(Google은 2024년부터 연 60M달러에 license), news(OpenAI는 AP, Axel Springer, FT, News Corp와 deal을 맺었고, Google은 Reddit 및 Stack Overflow와 deal을 맺었다), image+caption pair(여러 사례). 이것은 disclosure gap의 일부를 설명한다. License는 종종 corpus를 재공개하거나 어떤 license를 썼는지 공개하는 것까지 금지한다.

**Regime D: enterprise / proprietary / customer data.** Pretraining에는 없다. 이것은 post-training과 RAG/fine-tuning layer다. 하지만 2024-2026년 "data moat" 논의가 실제로 일어나는 곳이므로 이름을 붙일 가치가 있다. Anthropic, OpenAI, Google은 모두 기본적으로 API customer data로 학습하지 않겠다고 약속한다. Enterprise contract는 이를 명시한다.

**Opt-out register** pattern은 2024-2025년의 governance innovation이다.

- `ai.txt`(C2PA group이 제안).
- `User-agent: GPTBot`, `User-agent: CCBot` line이 있는 `robots.txt` extension.
- **HaveIBeenTrained** pattern: creator가 opt out하고 responsible crawler가 이를 조회하는 public registry.
- **Spawning API**(Stable Diffusion용 opt-out register, 현재 LLM training으로 확장).

이 mechanism들은 non-signatory lab을 구속하지 않는다. 이 장의 목적상 다음처럼 가정하라. **model tech report가 opt-out register를 언급하지 않으면, 사용하지 않았다고 가정하라.** [[dolma]]는 URL-filter list를 언급하고, [[fineweb]]은 URL-blocklist를 언급한다. [[llama-3]]와 [[qwen-3]]는 opt-out mechanic을 전혀 언급하지 않는다.

---

## 6. Open vs closed disclosure — data section 읽기

Guideline의 four-axis framework를 2024-2026 주요 release 일곱 개에 적용하면 다음과 같다.

| Model / corpus | Tokens | Composition | Licence | Disclosure granularity |
|---|---|---|---|---|
| Dolma v1.7 | 3 T | 6 source, stage별 % | source별, full manifest | **모든 threshold + ablation 공개** |
| FineWeb | 15 T | 100% CC post-filter | CC-derived | **full pipeline + classifier weight 공개** |
| OLMo-Mix-1124 / OLMo 2 | 3.9 T + 50 B cooldown | DCLM + Dolma 1.7 + Starcoder + Proof Pile II | source별 | **component별 공개** |
| Dolma 3 / OLMo 3 | 9.3 T source / 5.9 T mix | full curriculum (Mix + Dolmino + Longmino) | source별 | **stage별 budget** |
| Llama 3 | 15.6 T | "high-quality web, code, math" | proprietary | **token total만 공개** |
| Qwen3 | 36 T | 세 stage(30T + 5T + LC) + "synthetic" | proprietary | **stage budget만 공개, source mix 없음** |
| Qwen3.5 | not disclosed | not disclosed | proprietary | **없음** |
| DeepSeek-V3 | 14.8 T | "diverse, high-quality" | proprietary | **token total만 공개** |

Disclosure column을 실무에서 읽는 방법:

1. **Report가 token-weighted percentage를 주면 방향성은 신뢰하되, "released docs"와 "training time" 사이에 ±20% drift가 있다고 예상하라.** Upsampling, decontamination, dedup adjustment 이후의 차이다.
2. **Report가 stage-level budget만 준다면(Qwen3의 30T + 5T + long-context), stage semantic을 informative content로 다루라.** Qwen3의 "reasoning stage is 5T of the 36T"는 source가 아니라 *effort*에 대한 진술이다. 5T는 같은 upstream pool에서 다른 sampling weight로 뽑힌다.
3. **Report가 total token count만 준다면(Llama 3의 15.6T, DeepSeek-V3의 14.8T), 흥미로운 data signal은 *capability* section에 있다.** Llama 3 paper는 long-context, multilingual, code capability를 길게 다룬다. *"MMLU improvement의 X%가 reasoning data expansion에서 왔다"* 같은 claim으로부터 rough mix proportion을 역추정할 수 있다. Noise가 있지만 signal이 0은 아니다.
4. **Report가 아무것도 주지 않는다면(Qwen3.5), headline model이 유일한 data다.** Model을 known-mix peer와 benchmark하고 delta를 "data+architecture+inference improvement, undecomposed"로 다루라.

Scaling-law 관점([[scaling-laws-data-quality]])은 이 연습을 고집해야 하는 이론적 정당화를 제공한다.

> Data quality는 anecdotal curation benefit일 뿐 아니라 명시적인 scaling variable로 다룰 수 있다. […] 같은 token count를 가진 두 corpus도 quality가 충분히 다르면 서로 다른 scaling curve 위에 있을 수 있다.

예컨대 Llama 3(15.6T undisclosed)와 FineWeb(15T fully-disclosed, CC-only, no code)을 matched-token-count로 비교하면 data-mix leverage를 *직접* 알려준다. Matched-compute에서 Llama 3가 FineWeb-trained baseline을 이긴다면, 그 delta는 closed corpus의 composition advantage다.

---

## 7. Data track 나머지의 그림

Ch-10은 CCNet, C4, Dolma, FineWeb을 cookbook으로 읽는다. 각 corpus가 §2에서 나온 filter pipeline의 concrete worked example이다. Ch-11은 operational surface를 다룬다. Tokenizer, shard, lineage tracking(OlmoTrace가 reference), PII handling이다. 이후 장들은 domain balance, eval set에 대한 decontamination, long-context mix, code-heavy mix, multilingual 등을 전문적으로 다룬다. Data-track lab(outline의 ch-17)은 filter pipeline을 만들고 그 composition choice를 방어할 것을 요구한다.

Through-line은 이 장이 세운 **composition + disclosure discipline**이다. Ch-17에 이르면 여러분은 어떤 새로운 model report도 10분 안에 four-axis framework로 분류하고, known-mix baseline과의 방어 가능한 비교를 쓸 수 있어야 한다. 그 능력이 data track의 목적이다.

---

## 연결과 다음 내용

- **[[the-pile]]** — diversity-as-scaling-variable thesis, 22-subset composition, hand-curated mixture design pattern.
- **[[c4]] / [[ccnet]]** — heuristic-pipeline ancestor, T5와 multilingual-CC recipe.
- **[[dolma]]** — 3T open corpus, six-stage cascade, "reproducibility as scientific contribution" framing.
- **[[fineweb]]** — 15T open corpus, FineWeb-Edu classifier, per-dump-vs-global dedup finding.
- **[[olmo-2]] / [[olmo-3]]** — open-lab disclosure counter-movement, staged curriculum으로서의 Dolma 3.
- **[[llama-3]] / [[qwen-3]] / [[qwen-3-5]] / [[deepseek-v3]]** — "token total + some detail"에서 "headline only"로 가는 closed-lab gradient.
- **[[scaling-laws-data-quality]]** — quality를 scaling variable로 다루는 formalism.
- **ch-10 (Open Curation Pipelines)** — CCNet/C4/Dolma/FineWeb 각각의 filter-cascade를 code와 함께 파고든다.
- **ch-11 (Data Operations)** — tokenizers, shards, lineage, PII. 3T token을 실제로 다루는 operational layer.
- **ch-17 (Data Lab)** — filter pipeline을 만들고 composition과 decontamination을 방어한다.

## 더 읽을거리

- [[the-pile]] — Gao et al. 2020; 22-subset table과 mixture-weight argument.
- [[c4]] — Raffel et al. 2019; T5 paper § data, heuristic-blocklist lineage.
- [[ccnet]] — Wenzek et al. 2019; language ID + dedup + Wikipedia-perplexity quality scoring.
- [[dolma]] — Soldaini et al. 2024 (ACL); canonical open-pipeline reference.
- [[fineweb]] — Penedo et al. 2024; 15T + FineWeb-Edu, per-dump MinHash ablation.
- [[llama-3]] — Grattafiori et al. 2024; data section에서 *말하지 않는 것*을 읽어라.
- [[olmo-2]] / [[olmo-3]] — open-counterpart discipline, OLMo-Mix-1124와 Dolma 3 staged curriculum.
- [[qwen-3]] — Qwen Team 2025; synthetic-in-pretraining pattern.
- [[scaling-laws-data-quality]] — Subramanyam et al. 2025; effective-sample-size framing.
- Secondary: RedPajama v1 (Together AI, 2023), SlimPajama (Cerebras, 2023), DCLM (Li et al., 2024).

## 동반 시각화

**[figures/data-landscape.html](figures/data-landscape.html)** — open pretraining corpus의 interactive landscape. Stacked-bar view는 Pile, RedPajama, SlimPajama, Dolma, FineWeb, FineWeb-Edu, OLMo-Mix, Dolma 3에 걸친 domain composition(web / code / math / books / encyclopedic / forums / synthetic)을 보여준다. **Token-count** view(absolute scale, "각각이 얼마나 많은가" 렌즈)와 **document-count** view("artifact가 몇 개인가" 렌즈. web은 token 기준으로 지배적이지만 doc당 상대적으로 homogeneous하고, book은 document 수는 적지만 큰 token block을 제공한다)를 전환할 수 있다. Segment에 hover하면 upstream source URL과 filter note([[ccnet]] heuristic vs [[fineweb]] classifier vs [[dolma]] six-stage vs Pile hand-curated)를 볼 수 있다. Figure의 핵심은 직관적으로 느끼게 하는 것이다. Pile의 22-color stack에서 FineWeb의 single-color 15T bar로의 이동 *그 자체가* 2020 → 2026 composition shift다.
