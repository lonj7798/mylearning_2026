<!-- chapter: ch-28
     track: synthetic
     title: Modality — Long-Context Synthesis
     sources: [[longalpaca]], [[longalign]], [[longchat]], [[longmit]], [[prolong]],
              [[ruler]], [[babilong]], [[needle-in-haystack-data]], [[longbench]],
              [[long-context-llama3]], [[long-context-data-engineering]],
              [[qwen-long-context-synth]], [[longrope-data]], [[pose-synthesis]],
              [[longembed-synth]], [[gemini-long-context-tricks]]
     figures: figures/context-extension.html
-->

# 28장 — 모달리티: Long-Context Synthesis

> **핵심 통찰.** Long-context capability는 short-context training을 single knob로 확장한 것이 아니다. 2024+ frontier release가 동시에 풀어야 했던 세 independent axis의 *co-design*이다. (i) **Position encoding**: RoPE의 base frequency는 rescale되어야 한다(Llama-3는 500K, Fu-2024는 200M, LongRoPE는 evolutionary search로 찾은 *per-dimension* factors λ_i를 사용). 그렇지 않으면 sinusoid가 pretraining range를 지나 alias된다. (ii) **Data**: 짧은 문서의 concatenation이 아니라 길고 *coherent*한 document가 필요하다. ProLong의 coherence filter는 20B tokens로 512K context를 얻는 것과 garbage의 차이다. (iii) **Evaluation**: single-needle NIAH pass는 long-context를 증명하지 않는다. RULER의 13-task generator와 BABILong의 reasoning-in-a-haystack은 "claimed context"와 "effective context"가 frontier model에서도 자주 2× 다르다는 것을 드러낸다(Llama-3.1-70B: claimed 128K, RULER 기준 effective 약 64K). Instruction data와 evaluation tasks의 synthesis가 세 axis 각각을 tractable하게 만든다.
>
> **가이드라인.** Long-context training을 three-stage stack으로 다루라. (1) staged RoPE-base rescaling을 통한 **position extension**(또는 data-poor라면 LongRoPE per-dim search), (2) cross-domain ratio를 보존하고 within-domain length upsampling을 적용한 modest token budget(Fu는 5B, ProLong은 20B)의 coherent long documents 위 **continued pretraining**, (3) synthesized long tasks를 쓰는 **SFT** — cross-span coverage에는 LongAlign-style 5-questions-pick-one, explicit retrieval training에는 multi-needle retrieval, dialog state에는 LongMIT-style multi-turn. RULER + BABILong + realistic LongBench로 평가하고, NIAH만으로 평가하지 말라.

---

## 이 장이 필요한 이유

23–27장은 short-context SFT를 위한 synthetic-data design pattern, 즉 generate → filter → dedup → verify → select → mix를 세웠다. Long-context는 이 loop의 각 step을 깨뜨린다. Teacher model은 유용한 question-answer pair를 만들기 전에 source document 100K tokens를 *읽어야* 하므로 teacher-context ceiling이 hard constraint가 된다([[longmit]]). Filter cost는 sequence length와 함께 scale되므로 32K-shingled docs의 MinHash는 4K보다 8× 비싸다. Verification이 가장 가파르다. 생성된 answer가 50K tokens 문서에 흩어진 evidence를 실제로 사용했는지 싸게 automatic check할 방법이 없다. LLM-judge + human spot-check가 전부다([[longalign]]은 94/100을 manual validate).

동시에 *architecture* 쪽도 바뀌었다. Context-extension은 단순히 "더 긴 sequence로 train"하는 것이 아니다. RoPE([[longrope-data]], [[long-context-data-engineering]])는 base θ에 묶인 fixed frequency spectrum을 가지며, rescale하지 않고 position index를 training range 밖으로 밀면 frequency aliasing이 생긴다. 따라서 position fix + long-doc CPT + synthetic long-SFT + synthetic long-eval이라는 stack이 Llama-3-128K, ProLong-512K, Qwen-2.5-1M을 서로 구분한다.

이 장은 lab들이 실제로 만드는 순서대로 stack을 걷는다. 먼저 synthetic evaluation(측정할 수 없는 capability를 위한 data를 설계할 수 없기 때문), 그다음 SFT-only recipes, continued-pretraining recipes, fine-tune data를 10× 줄이는 position-encoding lane, 마지막으로 1M-context frontier다.

각 section을 뒷받침하는 raw-data library는 `wiki/raw-data/llm-training/papers/`에 있다. Kamradt NIAH README부터 Qwen 2.5-1M technical report까지 16 source files다. 각 section은 `[[wikilink]]`로 cite한다. Paper별 가장 하중을 받는 claim의 deeper walk-through는 `excerpts/` 아래를 보라.

---

## 1. Synthetic-task family — NIAH → RULER → BABILong

Long-context *evaluation* 자체도 synthetic이다. 자연적으로 충분히 긴 benchmark가 없기 때문이다. Lineage는 blog post에서 시작한다.

### 1.1 Needle-in-a-Haystack (Kamradt, Nov 2023)

원래 [[needle-in-haystack-data]] test는 one-liner다. Paul Graham essays를 target length까지 padding하고 그 안의 programmatic depth에 *"The best thing to do in San Francisco is eat a sandwich at Dolores Park on a sunny day"*라는 sentence를 숨긴다. *"What is the best thing to do in San Francisco?"*라고 묻고 exact-substring으로 score한다. Output은 2D heatmap(depth × length, accuracy as colour)이다. Teacher도 없고 비용도 거의 0이며 visually legible하다. 마지막 속성 때문에 이것은 *de facto* long-context marketing metric이 되었다. 이제 모든 128K / 200K / 1M-context release는 NIAH heatmap을 낸다.

**NIAH가 놓치는 것.** 하나의 fact를 하나의 depth에서 하나의 query로 retrieval하는 것만 test한다. 실제 long-context behaviour에는 multiple facts retrieval, distractor 무시, span across aggregation, retrieved items 간 reasoning이 필요하다. 모델은 128K에서 NIAH를 통과하면서 32K multi-hop에서 collapse할 수 있다. 그래서 두 successor benchmark가 필요하다.

### 1.2 RULER와 BABILong — task-family table

| Family | Paper | Primitive | Complexity knobs | What it stresses |
|---|---|---|---|---|
| **S-NIAH** (single-needle) | [[needle-in-haystack-data]], [[ruler]] | 1 key-value injected in haystack | depth, length, needle type (word / 7-digit / UUID), haystack type (noise / essay) | baseline retrieval |
| **MK-NIAH** (multi-key) | [[ruler]] | N keys, query one | N ∈ {4, …, full-haystack-distractors} | distractor resistance |
| **MV-NIAH** (multi-value) | [[ruler]] | one key → k values, return all | k ∈ {2, 4, 8} | recall completeness |
| **MQ-NIAH** (multi-query) | [[ruler]] | multiple independent queries per haystack | number of queries | parallel retrieval |
| **VT** (variable tracing) | [[ruler]] | `X2 = X1`, `X3 = X2`, …; return equivalents | chain count, hops per chain | coreference / state |
| **CWE** (common-word extraction) | [[ruler]] | tokens from common + uncommon word distribution | common count, frequency ratio | aggregation |
| **FWE** (frequent-word extraction) | [[ruler]] | Zeta-distributed tokens, return top-K | Zeta α, K | aggregation tail |
| **QA** | [[ruler]] | SQuAD / HotpotQA + distractor paragraphs | paragraph count | realistic retrieval + reasoning |
| **bAbI-in-PG19** | [[babilong]] | 20 bAbI reasoning tasks inside natural PG19 prose | 0K → 10M length (50M reported), 20 task templates | retrieval + symbolic reasoning |

RULER의 세 methodology point는 조용히 하중을 받는다. (a) **context length와 task complexity를 독립적으로 vary**한다. 모델이 raw length 때문에 깨졌는지, distractor density 때문에 깨졌는지 구분할 수 있다. (b) 명시적 answer prefix가 chat template에 있는 **task/length당 500 examples**. (c) **effective context size**는 score가 Llama2-7B@4K baseline 85.6 위에 머무르는 가장 긴 length로 정의된다. BABILong의 보완적 선택은 reasoning structure를 *templated*로 유지해 contamination-resistant하게 만들고, distractor는 *real prose*로 두는 것이다(artificial filler가 아님).

**모든 training report에서 중요한 결과.** Advertised 128K window를 가진 많은 model이 distractor, multiple targets, aggregation이 도입되면 급격히 degrade된다. Claimed ≠ effective context. Llama-3.1-70B의 NIAH @ 128K는 약 99%지만 RULER effective context는 약 64K다([[long-context-llama3]]). Qwen-2.5-14B-1M의 NIAH @ 1M은 약 100%지만 RULER @ 1M은 약 85%다([[qwen-long-context-synth]]). 이 gap이 reasoning-in-a-haystack tax다.

---

## 2. Early SFT-only recipes — LongAlpaca와 LongAlign

20B-token continued pretraining을 위한 compute가 누구에게도 없던 시기의 질문은 이것이었다. *SFT만으로 short-context base를 long-context model처럼 행동하게 만들 수 있는가?*

### 2.1 LongAlpaca-12K ([[longalpaca]], Chen et al. 2023)

Recipe는 minimum viable product다. (i) **3,000 long documents** 수집 — 40% ArXiv CS papers, 30% public-domain books, 30% GitHub repos. (ii) 각 document에 대해 **ChatGPT 또는 Claude**(가장 긴 docs에는 Claude 선호, 당시에도 100K ceiling)를 full text와 task-type specification(summarize / QA / extract / analyze)과 함께 prompt한다. Document당 **3 QA pairs** 생성. (iii) Filter: document ≥ 8K tokens, answer ≥ 30 tokens, profanity check. (iv) Short-chat behaviour를 보존하기 위해 **3K random Alpaca samples**(≤2K tokens)를 mix. **LongLoRA's shifted-sparse attention**과 결합하면 API fee 약 \$5K로 Llama-2-7B를 32K/100K까지 확장한다.

Artifact `Yukang/LongAlpaca-12k`는 state-of-the-art였기 때문이 아니라 shipped되었기 때문에 이후 long-context dataset의 reference baseline이 되었다.

### 2.2 LongAlign-10k ([[longalign]], Bai et al. 2024)

LongAlign은 long-context SFT를 독립된 training problem으로 진지하게 다룬 첫 recipe다. 네 재료:

1. **9 source mixes 위 Self-Instruct-style synthesis** — ArXiv, Books3, C4, CLUECorpus2020, CommonCrawl, GitHub, StackExchange, Wikipedia, WuDaoCorpora; 90% EN / 10% ZH. Teacher는 **Claude 2.1**. Generation은 two-stage다. *Claude에게 문서 전체를 덮는 5 candidate questions를 묻고, 그중 하나를 random pick해 answer를 요청한다.* 이 pick-one trick은 cross-span coverage를 강제한다. 없으면 teacher는 locally-answerable questions를 고르고, student는 long reasoning이 아니라 long retrieval을 배운다.
2. **`flash_attn_varlen_func`와 `cu_seqlens`를 쓰는 Packing + block-diagonal mask.** 평균 pack은 약 12 sequences를 담는다. Batch-size 8 → global batch 96.
3. **Sequence-level loss weighting** — naive packed loss는 sequence가 적은(긴) pack을 과가중하고 target token이 많은 target을 과가중한다. Fix: 각 target token을 그 sequence의 target length `N`으로 `1/N` 가중한다. Training 중 `K/(M·N)`으로 scale한다. 여기서 `K` packs, `M` sequences. ChatGLM3-6B-64k에서 LongBench-Chat은 **5.76 → 6.21**, Llama-2-7B-64k에서 **5.89 → 6.10**으로 오른다.
4. **Pre-SFT context extension**: RoPE base를 **10,000 → 2,000,000**(200× rescale)으로 확장하고, SFT 전에 10B tokens로 64K까지 continually pretrain한다.

따라서 LongAlign은 post-extension recipe다. Position extrapolation을 해결한다고 주장하지 않는다. Data quality는 약 **10k long examples**에서 saturate한다. 그 이후에는 volume보다 diversity가 중요하며, LongAlign-10k는 multi-segment integration에서 더 큰 LongAlpaca-12k를 이긴다.

### 2.3 LongChat과 LongMIT — conversational variants

[[longchat]](LMSYS, June 2023)은 **ShareGPT의 long tail**을 mined한다. 실제 user conversations ≥ 8K tokens, ≥ 4 turns에서 18K long conversations를 얻고, **condensed rotary embedding**(position index `i`를 `i/c`로 바꿈, `c = 8` for 16K)으로 Vicuna를 fine-tune한다. Position trick으로서는 수명이 짧았다(NTK-aware → YaRN → LongRoPE에 밀림). 하지만 *data-sourcing signal*로는 오래 남았다. Real long conversations에는 synthesis가 흉내 내기 어려운 topic shifts, backtracking, cross-turn reference patterns가 들어 있다.

[[longmit]]는 이를 synthesized multi-turn long-context dialogs로 일반화한다. 각 5–10 turns가 document span을 reference하고, full context는 20K–100K다. Single-turn long-doc SFT 위에 multi-turn을 추가하면 model family 전반에서 LongBench-Chat이 **5–10 points** 오른다고 보고한다. Limiter는 teacher-context ceiling이다. 50K-token document에 조건화된 10-turn conversation을 coherent하게 생성할 수 있는 것은 frontier closed models뿐이다.

---

## 3. ProLong — document-coherence thesis

[[prolong]](Gao, Wettig, Yen, Chen; Princeton NLP 2024)은 "long-doc quality가 중요한가, 아니면 volume이면 충분한가?"에 답한 논문이다. 답은 quality가 중요하며, HELMET에서 10+ points 차이를 만든다는 것이다.

### 3.1 Coherence filter — 무엇이 "long document"인가

ProLong의 training mix는 length뿐 아니라 explicit *coherence* criterion에 맞춰 curated된다. Filter threshold는 **document당 ≥ 64K tokens의 coherent content**이며, coherence는 source-type별로 판단된다.

- **Code**: single file이 아니라 전체 *repository*(README → source → tests를 sensible order로 concatenate).
- **Books**: structural fidelity를 가지고 parse한 full-book PDFs.
- **Academic**: references를 *포함한* full papers.
- **Web**: **discarded** — 긴 web docs라도 대부분 long-range dependency가 약한 scraped listings다.

마지막 rule이 논문의 sting이다. "Long"과 "coherent"는 web data에서 같은 predicate가 아니다. Filter 후 30B-token mix는 source별로 re-weight된다. **code × 4, books × 2, academic × 2, forum × 1, web × 0.5**, 최종 distribution은 약 40% code, 25% books, 15% academic, 10% long forum threads, 10% misc web.

Ablation이 증명이다. Curated long documents를 동일 token budget의 *concatenated short documents*로 바꾸는 obvious shortcut은 HELMET에서 **10+ points**를 잃는다. Concatenation shortcut은 model에게 "long context" = "locally-coherent short segments의 sequence"라고 가르친다. 그 failure mode가 이후 RULER의 multi-hop tracing과 BABILong의 reasoning task에 나타난다.

### 3.2 Staged schedule

- **Stage 1 — CPT(20B tokens)**: RoPE base를 **500K → 128M**으로 rescale(Llama-3.1 NTK-aware style). 처음에는 64K context로 train하고, second half에서 512K로 expand. LR 1e-4 → 1e-5 cosine. **100% long coherent documents**, training sample당 document 하나, *no cross-document packing*.
- **Stage 2 — SFT(5B tokens)**: 70% long-instruction(LongAlign-style, Claude-3-generated) + 30% short-instruction(UltraChat) + retrieval을 명시적으로 가르치는 synthetic multi-needle NIAH training samples.

총 compute 약 200K H100-hours로 512K context의 ProLong-8B(Llama-3-8B base)를 만들며, HELMET에서 open 8B models 중 lead하고 InfiniteBench @ 128K에서 Llama-3.1-8B-Instruct와 Qwen2-7B-Instruct를 이긴다.

---

## 4. Production recipe — Llama 3와 Fu 2024

두 parallel 2024 report가 128K frontier recipe를 못 박는다.

### 4.1 Llama 3의 staged schedule ([[long-context-llama3]])

Meta는 Llama 3.1(405B / 70B / 8B)을 8K에서 128K로 확장할 때 **six-stage continued pretraining** schedule을 사용한다. 총 약 800B tokens:

| Stage | Context | Tokens | RoPE base | Data mix shift |
|---|---|---|---|---|
| A | 8K → 16K | ~100B | adjusted | short:long 80:20 |
| B | 16K → 32K | ~100B | adjusted | 70:30 |
| C | 32K → 64K | ~150B | adjusted | 60:40 |
| D | 64K → 128K | ~200B | **500K** (final) | 40:60 |

**핵심 formula change.** RoPE base frequency `θ`는 Llama-2의 `10K`에서 final 128K model의 **500K**로 rescale된다. RoPE의 per-dimension frequency는

$$
\theta_i \;=\; \theta^{-2i/d}
$$

이며 `d`는 head dimension이다. `θ`를 키우면 모든 frequency가 *shrinks*되고, 모든 wavelength가 *stretches*되며, sinusoid aliasing point가 position axis 더 바깥으로 밀린다. Llama 3는 `θ = 500K`로 ship된다. Fu 2024([[long-context-data-engineering]])는 Llama-2에서 128K를 위해 `θ = 200M`까지 민다. LongRoPE는 이를 per-dimension λ_i로 일반화한다.

**Post-training integration.** Long-context SFT는 **total SFT samples의 약 0.1%**(약 100M 중 약 100K)로 유지된다. Long-SFT fraction을 1% 이상으로 올리면 MMLU가 약 1 point 손실된다. Binding constraint는 long-context gain이 아니라 short-context regression이다. Teacher는 Llama 3 405B 자신(self-distillation)이다.

**Claimed-vs-effective gap.** Llama-3.1-405B NIAH@128K는 약 99%이고 RULER effective context는 약 96K다. Llama-3.1-70B의 effective RULER context는 128K support에도 약 64K다. Meta는 논문에서 이를 인정한다. NIAH를 RULER가 대체하는 순간 training-eval co-design gap이 드러난다.

### 4.2 Fu 2024 — 5B-token open recipe ([[long-context-data-engineering]])

보완적 thesis: **cross-domain ratio는 보존되어야 하며, within-domain length distribution만 바꾼다.** 7 sources(CC, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange)를 가진 SlimPajama에서 시작한다. 원래 proportions를 보존한다(CC ≈ 67%, Books ≈ 4%, …). *각 source 내부에서* length histogram을 계산하고 32K보다 긴 document가 **5× weight**를 갖도록 sampling을 reweight한다. RoPE base를 `10K → 200M` NTK-aware로 rescale하고 80K context window에서 5B tokens train한다.

Critical ablation: cross-domain ratio를 깨면(예: Books를 globally up-weight) long-context NIAH는 유지되지만 short-context MMLU가 **3–5 points** 떨어진다. Long-context gain이 short-context capability를 희생하면 안 된다. Within-domain-only rule이 사주는 것이 그것이다.

---

## 5. LongRoPE와 per-dim search — position-encoding lane

Data-centric papers가 거의 답하지 않는 질문이 있다. *Uniform NTK-aware나 YaRN보다 더 잘 작동하는 RoPE rescaling이 있는가?*

[[longrope-data]](Ding et al., MSRA 2024)는 evolutionary search로 답한다. RoPE는 dimension `i`에서 frequency `θ_i = θ^(2i/d)`로 rotation을 적용한다. Uniform rescaling은 single `λ`로 `θ_i → θ_i / λ`를 한다. LongRoPE는 이를 일반화한다.

$$
\theta_i' \;=\; \theta_i \,/\, \lambda_i
$$

각 **per-dimension factor** `λ_i`가 학습된다. `d = 128` RoPE dimensions이면 128-dimensional search space다.

**Search cost.** Population size **64**, **40 generations**, mutation rate **0.3**. 따라서 최악에는 약 **64 × 40 = 2560** fitness evaluations이며, 각각 long-context forward pass다. Initial population은 NTK-aware, YaRN, uniform rescaling에서 seed된다. 첫 generation이 garbage evaluations에 낭비되지 않게 하는 합리적 prior다. Fitness는 held-out corpus에서 long-context perplexity + NIAH retrieval accuracy의 weighted combination이다. Output은 target context(256K, 1M, 2M)별 optimized `λ_i` vector다.

**왜 이기는가.** Uniform rescale은 모든 RoPE dimension이 같은 effective range에 도달하게 만든다. 하지만 high-frequency dimensions(small `i`)는 low-frequency ones보다 훨씬 먼저 saturate한다. Per-dimension λ는 high-frequency dimensions는 그대로 두면서 low-frequency ones를 aggressive하게 compress할 수 있다. Payoff는 **fine-tune data 10× 감소**다. LongRoPE는 두 stage(256K에서 약 300M, 2M에서 약 600M)에 걸쳐 **< 1B tokens of fine-tune data**로 LLaMA-2-7B를 2M context까지 확장한다. Fu 2024의 128K 5B와 비교된다. Short-context MMLU와 GSM8K는 base에서 1 point 이내로 유지된다.

LongRoPE를 보완하는 것은 [[pose-synthesis]]의 **PoSE**다. 4K-token sample로 train하되 두 chunk 사이에 random position-ID gap `δ ~ Uniform[0, target_ctx − 4K]`를 inject한다. 모델이 short-context compute로 long-position attention을 학습하게 하는 것이다. Full-length 대비 compute가 4× 줄지만 retrieval-strong / reasoning-weak하다. PoSE는 position distribution을 simulate할 뿐 content distribution을 simulate하지 않기 때문이다.

---

## 6. Qwen 2.5-1M과 Qwen 3 — 1M inference stack

[[qwen-long-context-synth]]는 명시적인 *training-inference split*을 가진 three-leg pipeline으로 frontier를 128K에서 1M tokens로 민다. Reasonable cap에서 train하고 inference에서 extrapolate한다.

1. **Gradual continued pretraining** 32K → 128K → 256K(50B + 50B + 100B tokens). Stage 2는 code-repo heavy, stage 3은 topics 간 document-concatenation으로 multi-topic mixed-context sequences를 명시적으로 synthesize한다.
2. **Qwen-Max가 생성한 Synthetic SFT mix** — multi-needle retrieval(1–8 needles at varied positions), long-doc summarization(50K–200K → summary), RAG-QA(5–20 candidate passages requiring cross-passage fusion), long-code understanding, FILL-IN(masked-segment reconstruction). Shortcut learning을 피하기 위해 answer가 source document의 multiple positions를 reference해야 한다는 filter를 둔다.
3. **Dual-Chunk Attention (DCA) at inference** — 긴 query를 256K-sized chunks로 나눈다. Intra-chunk attention은 standard다. Inter-chunk는 RoPE를 smooth하게 extrapolate하는 low-rank formulation을 사용한다. 이를 통해 1M-token training 없이 1M-token serving이 가능하다.

**Qwen 2.5-1M의 1M tokens eval-curve.** Qwen-2.5-14B-1M은 **NIAH @ 1M ≈ 100%** 및 **RULER @ 1M ≈ 85%**에 도달한다. 15-point NIAH-to-RULER gap은 표준 diagnostic이 된 reasoning-in-a-haystack tax다. InfiniteBench도 강하다. Short-context MMLU / GSM8K는 base Qwen-2.5에서 1 point 이내다. Qwen 3는 hybrid-thinking training pipeline 안에서 이 recipe를 상속하고 refine한다.

Gemini([[gemini-long-context-tricks]])는 product-side observation을 추가한다. Million-token scale에서는 *prompt organization과 context caching*이 eval story의 일부가 된다. 더 많은 context가 uniform attention quality를 의미하지 않는다. Training-side implication: long-context SFT data를 합성할 때 answer-relevant evidence가 context의 어디에 있는지(front, middle, back)를 vary하는 것 자체가 diversity axis이지 nuisance variable이 아니다.

---

## 7. 기억할 것 — three-lane matrix

| Lane | Knobs | Representative paper | Numbers to memorize |
|---|---|---|---|
| **Position extension** | RoPE base θ; per-dim λ_i; PoSE δ offset | [[long-context-llama3]], [[longrope-data]] | Llama-3: θ = 500K; Fu: θ = 200M; LongRoPE: 2560 search evals → 10× FT data savings |
| **Data (CPT)** | document coherence filter; within-domain length upsample; domain weights | [[prolong]], [[long-context-data-engineering]] | ProLong 20B CPT + 5B SFT; Fu 5B CPT @ 80K; ProLong code × 4 / books × 2 / web × 0.5 |
| **Data (SFT)** | cross-span QA; multi-needle; multi-turn | [[longalign]], [[longmit]], [[longalpaca]] | LongAlign 10k; long-SFT kept at ~0.1% (Llama-3); +5–10 LongBench-Chat from multi-turn |
| **Evaluation** | NIAH heatmap; RULER 13 tasks; BABILong 20 bAbI-in-PG19 | [[needle-in-haystack-data]], [[ruler]], [[babilong]] | RULER 500 ex/task/length; Llama-3.1-70B claimed 128K / effective 64K; Qwen 1M: NIAH 100 / RULER 85 |

Unifying lesson: long-context capability number는 *length에 대한 claim*이며, 그런 claim은 그 숫자가 네 lane 중 어디에서 측정됐는지를 명시해야 한다. "128K context"라고만 하고 "NIAH / RULER / BABILong / real" 중 무엇인지 말하지 않으면 underspecified다.

**[figures/context-extension.html](figures/context-extension.html)**에서 axis를 직접 만져 보라. Interactive는 RoPE frequency-band visualization(uniform rescale vs LongRoPE per-dim)과 context scale에 따라 simulated needle-recovery를 보고하는 growing NIAH haystack을 결합한다. Right panel은 claimed-vs-effective gap을 명시적으로 보여 준다. Context를 8K에서 1M으로 올리면 retrieval은 거의 100%를 유지하지만 reasoning-weighted metric은 training-data cap 주변에서 collapse하는 것을 볼 수 있다.

---

## Connections and what's next

- **[[longalign]] / [[longalpaca]] / [[longchat]] / [[longmit]]** — early SFT-only recipes; LongAlign의 packed-loss correction과 LongMIT의 multi-turn supplement가 SFT lane을 덮는다.
- **[[prolong]] / [[long-context-llama3]] / [[long-context-data-engineering]]** — production CPT recipes. 각각 data-budget vs curation curve의 다른 지점이다.
- **[[longrope-data]] / [[pose-synthesis]]** — position-encoding lane. 둘 다 data recipe와 orthogonal하다.
- **[[qwen-long-context-synth]] / [[gemini-long-context-tricks]]** — 1M-context frontier; DCA는 따로 공부할 만한 training-inference trick이다.
- **[[ruler]] / [[babilong]] / [[needle-in-haystack-data]] / [[longbench]] / [[longembed-synth]]** — 나머지를 measurable하게 만드는 evaluation lanes.
- **ch-27** — 이 장이 long-context modality로 specialize하는 synthetic-data design pattern.
- **ch-29 (next)** — learner-authored pipeline. 약 5K instructions를 end-to-end로 생성한다. Long-context synthesis는 optional modality specializations 중 하나다.

## Further reading

- [[longalign]] — 먼저 읽을 논문. 가장 완성된 end-to-end SFT recipe이며 packed-loss derivation이 load-bearing이다.
- [[prolong]] — §4(data curation)와 HELMET ablation을 읽어라. Coherence filter vs concatenation ablation은 volume alone이 작동하지 않는다는 가장 명확한 evidence다.
- [[ruler]] — Table 5(13-task configuration)와 effective-context definition을 읽어라. Standing long-context eval로 사용하라.
- [[longrope-data]] — §3(evolutionary search)와 `θ_i' = θ_i / λ_i` equation을 읽어라. Position-encoding이 independent lane인 가장 깔끔한 예다.
- [[qwen-long-context-synth]] — frontier 1M recipe. DCA는 inference / serving chapters에서 따로 볼 만한 training-inference trick이다.

## Companion visualization

**[figures/context-extension.html](figures/context-extension.html)** — 세 interactive panel. **Left:** RoPE frequency-band chart. Max-context slider를 움직이고 uniform-rescale vs LongRoPE per-dim mode를 toggle하면 frequency bands `θ_i = θ^{-2i/d}`가 다시 그려지고, effective "usable-range" band가 shaded된다. **Middle:** context slider와 함께 커지는 NIAH haystack. Needle sprite가 선택한 depth에 머물고 simulated recovery-rate counter가 distance-from-training-cap 및 noise에 따라 update된다. **Right:** claimed-vs-effective curve. NIAH, RULER, BABILong-style reasoning metric을 context가 커지는 같은 axis에 plot해 세 curve가 갈라지는 모습을 볼 수 있다. §1의 eval taxonomy와 §5의 position-encoding math를 한 그림에서 합치는 데 사용하라.
