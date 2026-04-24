<!-- chapter: ch-09
     track: data
     title: Training Data Landscape
     sources: [[the-pile]], [[c4]], [[ccnet]], [[dolma]], [[fineweb]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[qwen-3]], [[qwen-3-5]], [[deepseek-v3]], [[scaling-laws-data-quality]]
     figures: figures/data-landscape.html
-->

# Chapter 9 — Training Data Landscape

> **Core insight.** A pretraining corpus is not a pile of text; it is a *composition* — a set of sources, each with its own licence, domain, freshness, and failure modes — assembled under an explicit mixture policy. The headline-grabbing numbers (15T tokens, 36T tokens, 671B MoE params) all sit on top of a mixture decision that, in 2020, every lab disclosed and, in 2026, almost none do. The single most important change between The Pile and Llama 3 is not scale; it is *what the release tells you about the data*.
>
> **Guideline.** Before comparing two models, compare their corpora along four axes — (1) token count, (2) source composition (web / code / math / books / encyclopedic / synthetic), (3) licence regime (research-only / permissive / proprietary / opted-out), (4) disclosure granularity (full mix / stage-level / headline-only / none). Any performance difference that does not survive a matched-composition ablation is an artifact of the data, not the algorithm.

---

## Why this chapter exists

Chapters 1–8 built the trainer: optimizer moments, mixed precision, LR schedules, packed SFT, FSDP shards, checkpointing, silent-failure modes, and a working minimal run. Every one of those topics assumes *some* pretraining corpus is sitting on disk. This chapter is about that corpus — what it actually is, which corpora dominate 2020–2026 open work, why the frontier labs stopped telling you what's in theirs, and how to read a model report for its data discipline even when the disclosure is deliberately thin.

The data track (ch-09..ch-17) is structured downwards from here: ch-10 is the open-pipeline deep dive (CCNet, C4, Dolma, FineWeb as cookbooks); ch-11 is tokenizer/shard/lineage operations; later chapters handle domain balance, decontamination, long-context mixes, and the lab. Before any of that, you need a map.

Three shifts define the 2020 → 2026 period and frame the rest of the track.

1. **Scale.** Open corpora went from 800 GiB ([[the-pile]], 2020) to 15T tokens ([[fineweb]], 2024) — roughly two orders of magnitude. Frontier closed corpora went from ~1T ([[c4]]/RedPajama era) to 36T ([[qwen-3]]).
2. **Composition.** The Pile's 22 hand-curated subsets gave way to classifier-filtered web text as the dominant ingredient, with code + math + synthetic slices bolted on.
3. **Disclosure.** In 2020–2022, every data slice was named and weighed. By 2024–2026, [[llama-3]], [[qwen-3]], [[deepseek-v3]], and [[qwen-3-5]] report token totals but not mix percentages; [[dolma]], [[fineweb]], and the OLMo line are the transparency counter-movement.

This chapter is the vocabulary for ch-10..ch-17; treat the table in §2 as the chapter's spine.

---

## 1. CommonCrawl and Wikipedia — the raw substrate

Before any corpus, there are two primary upstream sources that everything else derives from: CommonCrawl and Wikipedia. They are not corpora you train on directly — they are the *substrate* that every public pipeline processes.

**CommonCrawl (CC)** is a non-profit crawl of the open web, released as monthly "snapshots" (also called "dumps") in WARC format. Each snapshot is ~300 TB raw, covering ~3 B documents. [[fineweb]] processes **96 snapshots** for its 15T-token output. [[ccnet]]'s pipeline is the canonical "turn a CC dump into English training text" recipe: text extraction (Trafilatura or custom), language ID (fastText), deduplication, quality scoring against a reference corpus.

Two operational facts you need to know:

- **CC is not a single thing.** Each snapshot is drawn from a different crawl; near-duplicates across snapshots are common (same URL re-crawled), which is why [[fineweb]]'s surprising finding — *per-dump MinHash beats global MinHash* — matters: globally deduping erases the "high-quality content that re-appears once per snapshot" signal (e.g., canonical reference pages).
- **CC is what you legally have access to.** The alternative, crawling the web yourself, lands you in robots.txt, copyright, and anti-bot-infrastructure territory. CC's licence (CC0-adjacent, with respect for the original hosts) is the only reason open pretraining exists at all.

**Wikipedia** is the other substrate. Relative to CC, Wikipedia is tiny (~5 B English tokens) but it punches far above its weight for three reasons: (i) it's CC-BY-SA licensed, clean, and near-canonical reference text; (ii) it's a *quality anchor* — [[ccnet]] scores CC documents by perplexity under a Wikipedia-trained KenLM, and later pipelines still use Wikipedia-style text as a positive class; (iii) every frontier model has read every Wikipedia article many times, which makes it both a signal-rich and an over-memorization-risky ingredient.

What each omits:

- **CC omits** paywalled content, many books, most forum archives (rate-limited), code repositories (GitHub is the primary source), academic PDFs (partial, under-extracted), anything behind login, and most non-English content at quality-parity (English is over-represented by crawl design).
- **Wikipedia omits** current events (editorial lag), non-Anglophone depth (English Wikipedia is ~2.5× the size of German, the next-largest), and conversational / colloquial text.

Every open corpus below starts from CC + Wikipedia and then *adds and subtracts*. The additions (code, books, papers, forums, synthetic) and subtractions (filters, dedup, content removals) are what distinguish one recipe from the next.

---

## 2. The comparison table — open corpora, 2020–2026

The central artifact of this chapter. Every row is a corpus that a released frontier-or-frontier-adjacent model has trained on. Columns are the four axes from the Guideline.

| Dataset | Year | Tokens (approx.) | Docs | Licence | Source mix (% by tokens where disclosed) | Disclosure |
|---|---|---|---|---|---|---|
| **CommonCrawl (raw)** | 2008– | ~100–1000 T per year of snapshots (uncleaned) | ~3 B/snapshot | CC0-adjacent | 100% web | upstream only — not a corpus |
| **Wikipedia (EN)** | continuous | ~5 B | ~7 M articles | CC-BY-SA 3.0 | 100% encyclopedic | full dumps public |
| **C4** ([[c4]]) | 2019 | ~180 B | ~365 M | derived from CC; terms follow CC | 100% web after heuristic filter | full recipe published; docs inspectable |
| **The Pile** ([[the-pile]]) | 2020 | ~300 B (825 GiB) | 22 subsets | mixed; some subsets contentious (Books3) | 24% web (Pile-CC) · 13% PubMed Central · 9% arXiv · 7% GitHub · 5% FreeLaw · 4% StackExchange · 3% USPTO · 3% PubMed abstracts · 2.5% Books3 · plus 13 smaller subsets | full 22-domain weights published |
| **RedPajama v1** | 2023 | ~1.2 T | ~930 M | follows per-source licences | 67% CC · 15% C4 · 4.5% GitHub · 4.5% books · 2.5% arXiv · 2% Wikipedia · 2% StackExchange | full table published by Together AI |
| **SlimPajama** | 2023 | ~627 B | ~590 M | same as RedPajama | RedPajama minus global near-dupes (50% smaller) | published; dedup stats public |
| **Dolma v1.7** ([[dolma]]) | 2024 | ~3 T | ~2 B | per-source; full `LICENSE` manifest | ~80% CC (web) · ~8% code (The Stack subset) · ~5% academic (peS2o) · ~3% Reddit · ~2% books (Gutenberg) · ~2% Wikipedia | every filter threshold + ablation published; `dolma` CLI is open source |
| **OLMo-Mix-1124** ([[olmo-2]]) | 2024 | ~3.9 T | n/a | derived from DCLM + Dolma 1.7 + Starcoder + Proof Pile II | disclosed per-component but percentages less granular than Dolma | stage 1 mix disclosed; 50B Dolmino cooldown is a separate higher-quality slice |
| **FineWeb** ([[fineweb]]) | 2024 | ~15 T | ~22 B | CC-derived; `datatrove` recipe public | 100% web after classifier (96 CC dumps) | full pipeline + ablations + per-dump MinHash decision |
| **FineWeb-Edu** ([[fineweb]]) | 2024 | ~1.3 T | ~1.5 B | CC-derived | 100% web, score≥3 educational classifier (Llama-3-70B annotations) | classifier + 450K labelled samples released |
| **Dolma 3** ([[olmo-3]]) | 2025 | ~9.3 T source / ~5.9 T mix | n/a | per-source | Dolma 3 Mix emphasizes math/code over 1.7; 100 B Dolmino mid-training; 50 B Longmino long-context | full *curriculum* published stage-by-stage |
| **Llama 3 pretraining** ([[llama-3]]) | 2024 | 15.6 T | not disclosed | proprietary | *not disclosed* (paper says "high-quality" and lists capability categories) | token total only; no mix |
| **Qwen3 pretraining** ([[qwen-3]]) | 2025 | 36 T | not disclosed | proprietary | *not disclosed* beyond three stages: ~30T general → ~5T reasoning → long-context | stage budgets only; no source mix |
| **Qwen3.5 pretraining** ([[qwen-3-5]]) | 2026 | not disclosed | not disclosed | proprietary | not disclosed at all — no technical report | headline model only |
| **DeepSeek-V3 pretraining** ([[deepseek-v3]]) | 2024 | 14.8 T | not disclosed | proprietary | *not disclosed* | token total + capability claims only |

Read the table vertically before horizontally. The "disclosure" column is the chapter's actual subject. Every open row published the mixture; every closed row didn't. The shift from top to bottom of the table is the shift the rest of §3–§7 explains.

---

## 3. The Pile's 22 subsets — what aged, what didn't

[[the-pile]] (Gao et al., 2020) built its reputation on the *mixture argument*: diversity is a scaling variable. From the source:

> The Pile is an 825 GiB English text corpus built from 22 diverse high-quality subsets spanning academic text, code, books, web text, and forums. The main claim is that source diversity materially improves cross-domain performance relative to generic crawl baselines.

The 22 subsets by approximate token share (EleutherAI's published weights):

```
Pile-CC            227 GB   (18.1%)  web text, CC-derived
PubMed Central     96 GB    ( 7.6%)  biomed full-text papers
Books3             101 GB   ( 8.1%)  Bibliotik shadow-library books
OpenWebText2       63 GB    ( 5.0%)  Reddit-upvoted web pages
arXiv              56 GB    ( 4.5%)  math/physics/CS papers
GitHub             95 GB    ( 7.6%)  code
FreeLaw            51 GB    ( 4.1%)  US legal cases
StackExchange      32 GB    ( 2.6%)  Q&A across ~350 sites
USPTO              22 GB    ( 1.7%)  patent grants
PubMed Abstracts   19 GB    ( 1.5%)  biomed abstracts
Gutenberg (PG-19)  11 GB    ( 0.9%)  public-domain books
OpenSubtitles      13 GB    ( 1.0%)  movie subtitles
Wikipedia (EN)     6  GB    ( 0.5%)  encyclopedic
DM Mathematics     8  GB    ( 0.7%)  synthetic math problems
Ubuntu IRC         5  GB    ( 0.4%)  developer chat
BookCorpus2        6  GB    ( 0.5%)  novels (follow-up to BookCorpus)
EuroParl           4  GB    ( 0.3%)  EU parliamentary proceedings
HackerNews         4  GB    ( 0.3%)  startup/tech forum
YouTube Subtitles  3  GB    ( 0.3%)  auto-generated captions
PhilPapers         2  GB    ( 0.2%)  philosophy papers
NIH ExPorter       2  GB    ( 0.1%)  grant abstracts
Enron Emails       0.9 GB   ( 0.1%)  email corpus
```

(Percentages are token-weighted; The Pile applies explicit upsampling to some subsets, so "weight during training" ≠ "share of raw corpus.")

**Which aged well:** arXiv, PubMed Central, GitHub, StackExchange, Wikipedia, FreeLaw, USPTO. Every 2024–2026 open mix still includes direct analogs of these — arXiv papers became part of peS2o ([[dolma]]), GitHub was absorbed and extended into The Stack (3 T tokens by 2024), StackExchange survives in essentially every chat/code pretraining. The common property: *structured, license-clean, long-document, domain-specialist text*.

**Which did not age well:**

- **Books3** — the 2.5% books slice was sourced from Bibliotik, a shadow library. It is the central subject of the 2023 Rhode Island lawsuit and multiple follow-ups; by 2024 it is removed from every production derivative (RedPajama and Dolma swap in Project Gutenberg; commercial labs presumably had licensed deals). Books3's absence is *the* single biggest licence-regime event in open pretraining data. It also explains why 2024 open corpora have a systematic weakness on long-form literary text vs closed ones.
- **OpenSubtitles** — machine-generated subtitle text has quality issues (timing artefacts, truncated lines) that a 2020 recipe tolerated and a 2024 classifier would filter out. Removed from derivatives.
- **Enron emails / Ubuntu IRC** — human chat data from a specific domain; underperforms synthetic chat data at 2024+ scale.
- **YouTube Subtitles** — auto-caption quality issues; removed from most derivatives.
- **DM Mathematics** — synthetic math *problems* (no step-by-step reasoning); superseded by real human math (MATH, GSM8K) and by [[qwen-3]]'s Qwen2.5-Math synthetic chains-of-thought.

The lesson: **hand-curated diversity works, but hand-curated sources must be maintainable.** The Pile was a one-shot release; its subsets are now 5 years old. [[dolma]], [[fineweb]], and the OLMo line are *pipelines*, not releases — they re-run on fresh CC snapshots and re-derive the mix.

---

## 4. The 2020 → 2026 shift — from raw-web maximalism to web+code+synthetic

The composition of a frontier pretraining mix evolved in three distinct phases.

**Phase 1 (2018–2021): C4, GPT-3 era.** Web text dominates; code is a small side channel. T5/GPT-3 trained on ~99% web-derived text with minor boosts from books and Wikipedia. [[c4]] is representative: aggressive heuristic filtering of CC, no code, no math, no synthetic.

**Phase 2 (2020–2023): The Pile, RedPajama, Llama 1/2 era.** Hand-curated mixture becomes the standard. Explicit domain subsets (code ~5–10%, academic ~5%, books ~5%, web ~60–70%, forums/misc ~10%). Llama 1 published its mix (CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, Gutenberg+Books3 4.5%, arXiv 2.5%, StackExchange 2%). Llama 2 published the same structure with scale adjustments.

**Phase 3 (2024–2026): classifier-filtered web + heavy code + synthetic, mix undisclosed.** [[fineweb]] shows that a single LLM-labeled quality classifier on CC beats every heuristic stack for MMLU — and suddenly web text is back, but *filtered* web text. [[qwen-3]] discloses that **~5T of its 36T tokens are a dedicated "reasoning stage" including synthetic math and synthetic code** from Qwen2.5-Math and Qwen2.5-Coder. From the source:

> Data expansion includes:
>   - OCR-style text extraction from large PDF corpora using Qwen2.5-VL
>   - synthetic math data from Qwen2.5-Math
>   - synthetic code/data variants from Qwen2.5-Coder and related models
> The report says the data is annotated at large scale for educational value, domain, and safety, then mixed at the instance level using proxy-model ablations.

[[llama-3]] pretrained on 15.6 T tokens with "high-quality" web, code, and mathematical data — the report describes *capabilities* (multilinguality, coding, reasoning, tool use) without providing a source-mix table at all. [[deepseek-v3]] discloses 14.8T tokens with no mix percentages. [[qwen-3-5]] discloses neither tokens nor mix — the 2026 frontier non-disclosure extreme.

**Why the frontier labs stopped disclosing.** Four reasons, in order of candour:

1. **Competitive moat.** The mixture is the single most actionable signal for reproducing a model's capability profile. Disclosing it is giving competitors the recipe.
2. **Synthetic data attribution.** When ~30% of the pretraining mix is *generated by an earlier version of your own model*, "disclosing the mix" now means disclosing the generator's outputs, which are themselves proprietary. The recursion forecloses transparency.
3. **Licence liability surface.** Books3 taught labs that published-mix-percentages create a target for litigation. If you don't publish the percentage, discovery becomes the adversary's problem.
4. **Opt-out registers.** 2023–2025 saw the rise of opt-out mechanisms (NYT's 2023 robots.txt directives, CC's "noai" convention, ai.txt proposals). Labs that publish their mix expose themselves to "why did you include X when X opted out" questions. The silent default is safer.

Note the asymmetry: [[olmo-2]] and [[olmo-3]] publish progressively *more* detail over time (Dolma 3 is a staged curriculum, not just a mix), while Qwen and DeepSeek publish progressively less. The industry is bifurcating.

---

## 5. Licence and governance — copyright, opt-out, enterprise-data constraint

Data licences are where the clean engineering picture meets the messy legal reality. Three regimes dominate.

**Regime A: CC-BY-SA / CC0 / public domain.** Wikipedia (CC-BY-SA 3.0), Project Gutenberg (public domain), Common Crawl (respects robots.txt at crawl time; downstream use is a separate question), arXiv (author licences vary; de facto scraped). Legally the safest tier, but also the smallest — Wikipedia is ~5 B tokens, Gutenberg is ~11 B tokens. No realistic-scale model can train primarily on Regime A.

**Regime B: per-document mixed licence, derivative risk.** Common Crawl output, GitHub scrapes (millions of different repo licences, from MIT to GPL to unlicensed to "all rights reserved"), Reddit (posts are CC-BY-4.0 under Reddit's user agreement, but the 2023 API monetization and subsequent legal actions complicate downstream republication). Most 2020–2024 open corpora operate in Regime B and rely on *fair use* as the legal basis for LLM training. This is what the Authors Guild, NYT, Getty, and Universal Music lawsuits are challenging.

Concrete Regime B artefacts:

- **The Stack (BigCode)** filters GitHub by licence: only repos with a permissive licence (MIT/Apache/BSD) are included, and the project added an **opt-out register** (contributors can request exclusion). The Stack v1.2 is ~3 TB of code; The Stack v2 released in 2024 expanded to ~67 TB.
- **Books3** was the opposite: shadow-library books with no licence defence. The 2023 Rhode Island filing made it the exemplar of a Regime B inclusion that should have been Regime D (see below).
- **FineWeb / FineWeb-Edu** inherit CC's licensing posture. The `datatrove` codebase includes URL-filter lists that implement opt-out requests.

**Regime C: licensed content.** Frontier labs buy or licence specific corpora — Reddit (Google licensed it for $60M/year from 2024), news (OpenAI signed deals with AP, Axel Springer, FT, News Corp; Google with Reddit and Stack Overflow), image+caption pairs (various). This explains part of the disclosure gap: the licences often forbid re-publishing the corpus or even disclosing which licence was used.

**Regime D: enterprise / proprietary / customer data.** Not in pretraining — this is the post-training and RAG/fine-tuning layer — but worth naming because it's where 2024–2026 "data moat" conversations actually happen. Anthropic, OpenAI, and Google all pledge not to train on API customer data by default; enterprise contracts make this explicit.

The **opt-out register** pattern is the governance innovation of 2024–2025:

- `ai.txt` (proposed by the C2PA group).
- `robots.txt` extensions with `User-agent: GPTBot`, `User-agent: CCBot` lines.
- The **HaveIBeenTrained** pattern: a public registry where creators opt out, and responsible crawlers consult it.
- **Spawning API** (opt-out register for Stable Diffusion, now extended to LLM training).

These mechanisms do not bind non-signatory labs. For this chapter's purposes, assume: **if a model's tech report does not mention an opt-out register, assume it didn't use one.** [[dolma]] mentions its URL-filter list; [[fineweb]] mentions its URL-blocklist; [[llama-3]] and [[qwen-3]] do not mention opt-out mechanics at all.

---

## 6. Open vs closed disclosure — reading a data section

The four-axis framework from the Guideline, applied to the seven main 2024–2026 releases.

| Model / corpus | Tokens | Composition | Licence | Disclosure granularity |
|---|---|---|---|---|
| Dolma v1.7 | 3 T | 6 sources, per-stage % | per-source, full manifest | **every threshold + ablation published** |
| FineWeb | 15 T | 100% CC post-filter | CC-derived | **full pipeline + classifier weights** |
| OLMo-Mix-1124 / OLMo 2 | 3.9 T + 50 B cooldown | DCLM + Dolma 1.7 + Starcoder + Proof Pile II | per-source | **per-component disclosed** |
| Dolma 3 / OLMo 3 | 9.3 T source / 5.9 T mix | full curriculum (Mix + Dolmino + Longmino) | per-source | **stage-by-stage budgets** |
| Llama 3 | 15.6 T | "high-quality web, code, math" | proprietary | **token total only** |
| Qwen3 | 36 T | three stages (30T + 5T + LC) + "synthetic" | proprietary | **stage budgets, no source mix** |
| Qwen3.5 | not disclosed | not disclosed | proprietary | **none** |
| DeepSeek-V3 | 14.8 T | "diverse, high-quality" | proprietary | **token total only** |

How to read the disclosure column in practice:

1. **If a report gives token-weighted percentages, trust them directionally but expect ±20% drift** between "released docs" and "training time" (after upsampling, decontamination, and dedup adjustments).
2. **If a report gives stage-level budgets only (Qwen3's 30T + 5T + long-context), treat the stage semantics as the informative content.** Qwen3's "reasoning stage is 5T of the 36T" is a statement about *effort*, not source — the 5T is drawn from the same upstream pool with different sampling weights.
3. **If a report gives only a total token count (Llama 3's 15.6T, DeepSeek-V3's 14.8T), the interesting data signal is in the *capability* sections.** Llama 3's paper goes into long-context, multilingual, and code capability at length; you can reverse-engineer rough mix proportions from *"X% of our improvements on MMLU came from the reasoning data expansion"*-style claims. It is noisy but non-zero signal.
4. **If a report gives nothing (Qwen3.5), the headline model is the only data.** Benchmark the model against a known-mix peer and treat the delta as "data+architecture+inference improvements, undecomposed."

The scaling-law view ([[scaling-laws-data-quality]]) gives the theoretical justification for insisting on this exercise:

> Data quality can be treated as an explicit scaling variable, not just an anecdotal curation benefit. […] Two corpora with the same token count can sit on different scaling curves if quality differs enough.

A matched-token-count comparison between, say, Llama 3 (15.6T undisclosed) and FineWeb (15T fully-disclosed, CC-only, no code) tells you about data-mix leverage *directly*: if Llama 3 outperforms a FineWeb-trained baseline at matched-compute, the delta is the closed corpus's composition advantage.

---

## 7. The picture for the rest of the data track

Ch-10 reads CCNet, C4, Dolma, and FineWeb as cookbooks — each one a concrete worked example of the filter pipeline the corpora in §2 came out of. Ch-11 handles the operational surface: tokenizers, shards, lineage tracking (OlmoTrace is the reference), PII handling. Later chapters specialise — domain balance, decontamination against eval sets, long-context mixes, code-heavy mixes, multilingual — and the data-track lab (ch-17 in the outline) requires producing a filter pipeline and defending its composition choices.

The through-line is the **composition + disclosure discipline** this chapter established. By ch-17 you should be able to read any new model report, classify it on the four-axis framework in ten minutes, and write a defensible comparison against a known-mix baseline. That capability is the point of the data track.

---

## Connections and what's next

- **[[the-pile]]** — the diversity-as-scaling-variable thesis; the 22-subset composition; the hand-curated mixture design pattern.
- **[[c4]] / [[ccnet]]** — the heuristic-pipeline ancestors; the T5 and multilingual-CC recipes.
- **[[dolma]]** — 3 T open corpus; the six-stage cascade; the "reproducibility as scientific contribution" framing.
- **[[fineweb]]** — 15 T open corpus; FineWeb-Edu classifier; the per-dump-vs-global dedup finding.
- **[[olmo-2]] / [[olmo-3]]** — the open-lab disclosure counter-movement; Dolma 3 as staged curriculum.
- **[[llama-3]] / [[qwen-3]] / [[qwen-3-5]] / [[deepseek-v3]]** — the closed-lab gradient from "token total + some detail" to "headline only."
- **[[scaling-laws-data-quality]]** — the formalism for treating quality as a scaling variable.
- **ch-10 (Open Curation Pipelines)** — drills into the filter-cascade of each of CCNet/C4/Dolma/FineWeb with code.
- **ch-11 (Data Operations)** — tokenizers, shards, lineage, PII — the how-you-actually-handle-3T-tokens operational layer.
- **ch-17 (Data Lab)** — produce a filter pipeline; defend its composition and decontamination.

## Further reading

- [[the-pile]] — Gao et al. 2020; the 22-subset table and the mixture-weight argument.
- [[c4]] — Raffel et al. 2019; T5 paper § data, the heuristic-blocklist lineage.
- [[ccnet]] — Wenzek et al. 2019; language ID + dedup + Wikipedia-perplexity quality scoring.
- [[dolma]] — Soldaini et al. 2024 (ACL); the canonical open-pipeline reference.
- [[fineweb]] — Penedo et al. 2024; 15T + FineWeb-Edu; the per-dump MinHash ablation.
- [[llama-3]] — Grattafiori et al. 2024; read the data section for *what it does not say*.
- [[olmo-2]] / [[olmo-3]] — the open-counterpart discipline; OLMo-Mix-1124 and Dolma 3 staged curriculum.
- [[qwen-3]] — Qwen Team 2025; the synthetic-in-pretraining pattern.
- [[scaling-laws-data-quality]] — Subramanyam et al. 2025; effective-sample-size framing.
- Secondary: RedPajama v1 (Together AI, 2023), SlimPajama (Cerebras, 2023), DCLM (Li et al., 2024).

## Companion visualization

**[figures/data-landscape.html](figures/data-landscape.html)** — interactive landscape of open pretraining corpora. Stacked-bar view shows domain composition (web / code / math / books / encyclopedic / forums / synthetic) across Pile, RedPajama, SlimPajama, Dolma, FineWeb, FineWeb-Edu, OLMo-Mix, Dolma 3. Toggle between **token-count** view (absolute scale, the "how much of each" lens) and **document-count** view (the "how many artefacts" lens — web dominates by token but is relatively homogeneous per-doc, while books contribute huge token blocks but few documents). Hover any segment to see the upstream source URL and the filter notes ([[ccnet]] heuristic vs [[fineweb]] classifier vs [[dolma]] six-stage vs Pile hand-curated). The point of the figure is visceral: the shift from Pile's 22-colored stack to FineWeb's single-color 15T bar *is* the 2020 → 2026 composition shift.
