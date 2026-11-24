<!-- chapter: ch-09
     track: pretraining
     kind: content
     title: Pretraining Data Composition and Capability Coverage
     deps: [ch-08b]
     sources: [[the-pile]], [[pretrainers-guide-training-data]], [[to-code-or-not-to-code]], [[code-pretraining-task-performance]], [[data-constrained-scaling]], [[llama-3]], [[llama-3-recipe]], [[deepseek-v3]], [[dolma]], [[fineweb]], [[dclm]], [[nemotron-cc]], [[olmo-2]], [[weborganizer]], [[paloma]], [[datadecide]], [[doremi]], [[consent-in-crisis]], [[epoch-data-stock]]
     figures: figures/data-landscape.html
     revised: 2026-09 (generality revision)
-->

# 9장 — 사전학습 데이터 구성과 능력 커버리지

> **핵심 통찰.** Pretrained model이 fit하는 domain과 할 수 있는 task는 corpus source에서 나온다. Controlled 1.3B–1.5B run에서 22-source Pile로 학습한 model은 22개 Pile component 모두에서 CC-100-trained model보다 bytes당 bit가 낮았다([[the-pile]] Table 4). Pile에서 Common Crawl, Books, OpenWebText2를 제거하면 average QA score가 각각 4.8, 2.7, 1.4 낮아졌다([[pretrainers-guide-training-data]] Fig. 8). 25%(470M) 및 50%(4.2B)의 code share는 natural-language average를 낮추지 않았지만([[to-code-or-not-to-code]], [[data-constrained-scaling]]), code share가 커질수록 world-knowledge score는 낮아졌다. 25% code에서 relative 3.4%, 75%에서 31%였다([[to-code-or-not-to-code]] §3.3; 또한 [[code-pretraining-task-performance]] §5). Llama 3는 small-model scaling-law experiment로 선택한 4-category mix(약 general knowledge 50%, math/reasoning 25%, code 17%, multilingual 8%)만 공개한다([[llama-3]] §3.1.2). 따라서 composition의 controlled evidence는 open corpus에서 나온다.
>
> **지침.** Goal이 breadth라면 heterogeneous source type(web, books, encyclopedic, academic, code)을 모두 mixture에 유지하고 per-domain loss와 broad task suite로 score한 small-scale run으로 share를 정한다. Common Crawl, Books, OpenWebText2를 제거하면 average QA accuracy가 낮아졌고, best average는 모든 source 또는 거의 모든 source를 썼기 때문이다([[pretrainers-guide-training-data]] §6). 1B-1x에서 curation method 10개의 Core score는 7B-1x score와 Pearson r = 0.956으로 correlate했다([[dclm]] §3.2, Fig. 3). Code를 추가할 때 natural-language average가 낮지 않았던 tested range인 25%(470M, 200B token)~50%(4.2B, 84B token)로 다루고, 25% code에서 relative 3.4%, 75%에서 31% 하락한 world-knowledge task를 별도로 추적한다([[to-code-or-not-to-code]] §3.3). Quality/topic classifier로 web data를 고르면 그것이 남기는 topic mix를 측정한다. Llama 3 classifier는 arts/entertainment를 downsample하고([[llama-3]] §3.1.2), MMLU에 tuning한 mixture는 HellaSwag를 57.5에서 54.1로 낮췄다([[weborganizer]] Table 10). Web portion이 이미 model-based filter로 선택되었다면(DCLM-Baseline은 fastText score 기준 document top 10%를 남김), curated non-web source가 여전히 도움이 되는지 시험한다. 이를 더하자 1B scale에서 DCLM Core가 1.2 낮아졌기 때문이다([[dclm]] Table 6).

## 일반 목적 모델에서 이 장이 필요한 이유

Training pipeline에는 pretraining, mid-training, supervised fine-tuning(SFT), preference optimization, reinforcement learning(RL), evaluation의 여섯 stage가 있다. 이 장은 첫 stage를 다룬다. [[ch-08b]]는 broad text의 next-token prediction이 target하지 않은 ability를 왜 만드는지 설명했다. 이 장은 “broad text”의 어느 부분이 어느 ability를 공급하는지 묻는다.

측정 가능한 문제는 coverage다. [[ch-08a]]의 compute allocation이 정한 fixed token budget에서 corpus 두 개는 average loss는 비슷하지만 per-domain loss, domain-specific question accuracy, recent event knowledge는 서로 다른 model을 만들 수 있다. Later stage가 이 gap을 완전히 고치지는 않는다. [[pretrainers-guide-training-data]] §4에서 pretraining-data year와 evaluation year의 mismatch로 생긴 accuracy loss는 full year-matched training set으로 fine-tuning한 뒤에도 남았다.

이 장은 네 용어를 계속 쓴다. **Source**는 Common Crawl extraction, code host, book collection처럼 provenance 하나를 가진 collection이다. **Domain**은 analysis를 수행하는 사람이 정의한, 비슷한 content의 document/source group이다. **Mixture weight**는 source에서 가져온 training token 또는 byte share이며, source가 repeat/subsample되면 raw corpus share와 다르다. **Capability coverage**는 stated baseline보다 model이 높은 performance를 보이는 evaluation domain/task type의 set이다. Document를 제거하는 filter는 [[ch-10]], [[ch-10a]], mixture weight search algorithm은 [[ch-13]], multilingual coverage는 [[ch-13a]], repetition/unique-token limit는 [[ch-14]], late-stage annealing mixture는 [[ch-32]]에서 다룬다.

## §1 구성: source, mixture weight, 단위

**정의.** Pretraining composition은 run에 사용한 source list와 run 전체에서 각 source의 mixture weight다.

**문제.** Weight 없는 source list 또는 unit 없는 weight를 담은 report는 reader가 training distribution을 재현하거나 run 둘을 비교하게 하지 못한다.

**Mechanism.** The Pile에서 source weight는 두 선택으로 정해진다.

1. Source를 collect하고 raw size S_i를 측정한다.
2. Full corpus를 한 번 도는 동안 mixture가 source를 몇 pass(`e_i`, “epochs”) 할지 정한다. Pile은 “higher quality components”의 epoch를 올렸고, Wikipedia는 최대 3회였다. 이는 “Following Brown et al. (2020)”이었다([[the-pile]] §2).
3. Source의 effective size는 S_i · e_i이고, weight는 effective size를 total effective size로 나눈 값이다.

**수식.**

```
w_i = (S_i · e_i) / Σ_j (S_j · e_j)
```

- w_i: source i에서 나온 training byte의 share.
- S_i: up-/down-sampling 전 source i의 raw size(GiB).
- e_i: mixture를 한 번 도는 동안 source i를 도는 횟수.
- Sum은 mixture의 모든 source j에 대해 취한다.

**Worked example.** [[the-pile]] Table 1(arXiv:2101.00027v1)에서 가져온 row다.

```
Component         Raw Size     Weight   Epochs   Effective Size
Pile-CC           227.12 GiB   18.11%   1.0      227.12 GiB
PubMed Central     90.27 GiB   14.40%   2.0      180.55 GiB
Books3            100.96 GiB   12.07%   1.5      151.44 GiB
OpenWebText2       62.77 GiB   10.01%   2.0      125.54 GiB
ArXiv              56.21 GiB    8.96%   2.0      112.42 GiB
Github             95.16 GiB    7.59%   1.0       95.16 GiB
Wikipedia (en)      6.38 GiB    1.53%   3.0       19.13 GiB
The Pile          825.18 GiB                     1254.20 GiB
```

PubMed Central은 90.27 × 2.0 = 180.55 GiB이고, 180.55 / 1254.20 = 14.40%다. Raw corpus share는 90.27 / 825.18 = 10.9%다. Pile-CC는 반대 방향이다. Raw byte의 27.5%(227.12 / 825.18)이지만 training byte의 18.11%다. 다른 큰 source 대부분이 repeat되었기 때문이며, 1.0 epoch의 Github은 그렇지 않았다. Wikipedia는 raw byte 0.77%, training byte 1.53%다. Raw size를 effective total로 나누면(예: 90.27 / 1254.20 = 7.2%) 두 column을 섞게 되어 어느 quantity도 얻지 못한다. Table 1 caption은 “Weight is the percentage of bytes in the final dataset”이라고 말한다. Weight는 token share가 아니라 byte share다.

**Report 간 unit.** 같은 source의 byte share와 token share는 content별 bytes per token이 다르므로 다르다. Pile은 Github, ArXiv, Stack Exchange, DM Mathematics가 GPT-2 token당 byte가 가장 적은 component 중 하나라고 보고한다([[the-pile]] §5.1). 따라서 이 source들은 byte share보다 큰 token share를 갖는다. 이 장의 corpus마다 token count tokenizer도 다르다. Dolma는 Llama token([[dolma]] Table 1), FineWeb은 GPT-2 token([[fineweb]] §3), DCLM은 GPT-NeoX token([[dclm]] §3.1), Epoch stock estimate는 cl100k_base token([[epoch-data-stock]] §2)을 쓴다. 따라서 서로 다른 report의 token count는 tokenizer compression 비율 범위 안에서만 비교할 수 있다.

**조건과 한계.** Pile의 epoch choice는 ablate되지 않았다. 이후 [[doremi]]는 280M proxy model로 Pile domain weight를 search했고 default weight 대비 8B model average one-shot exact match를 20.03에서 26.56으로 올렸다(§3.2, Table 5). 따라서 default weight는 그 evaluation에 최적이 아니었다(**결과, 단일 연구**; 세부는 [[ch-13]]).

**함의.** Composition은 각 row가 source, raw size, repetition, unit을 명시할 때만 완전히 규정된다. Companion figure **[figures/data-landscape.html](figures/data-landscape.html)**는 reader가 Panel A를 모든 Pile component의 raw-size share와 training weight 사이에서 바꾸게 하므로 epoch column effect를 row별로 읽을 수 있다.

## §2 Source type과 그것이 공급하는 능력

**정의.** Bytes per byte(BPB)는 test set에서의 model average negative log-likelihood를 base 2로 변환하고 token 대신 UTF-8 byte로 normalize한 값이다. Pile은 tokenizer에 의존하지 않으므로 BPB를 선호한다([[the-pile]] §3.1).

**문제.** Web이 다수인 corpus의 held-out loss 하나는 minority domain의 높은 loss를 숨길 수 있다. Domain별 BPB가 이를 드러낸다.

```
BPB = (L_T / L_B) · ℓ / ln 2
```

- ℓ: test set에서 token당 평균 negative log-likelihood(nat).
- L_T: test set token length; L_B: UTF-8 byte length.
- ln 2는 nat을 bit로 변환한다.

**Worked example.** Pile은 Pile 전체에서 L_T / L_B = byte당 GPT-2 token 0.29335를 보고한다(§3.1). ℓ = token당 2.0 nat인 model의 BPB는 0.29335 × 2.0 / 0.6931 = 0.846이다.

**근거.** Pile 저자는 architecture가 동일한 1.3B model을 Pile, CC-100(English), raw Common Crawl에서 각각 학습했다. 세 corpus는 evaluation set against deduplicate되었고 약 40GB로 downsample되었다(§4.1; §4는 token count/step을 보고하지 않음). Table 3의 Pile test BPB는 Pile model 0.9433, CC-100 1.3293, raw CC 1.1275였다. LAMBADA accuracy는 50.1, 49.7, 43.8이었다. Table 4의 selected row(각 Pile test component BPB; 낮을수록 좋음)는 다음과 같다.

```
Evaluated on        trained on: The Pile   CC-100 (en)   Raw CC (en)
Pile-CC                          0.9989     1.0873        1.0287
ArXiv                            0.7945     1.8159        1.2642
Github                           0.5597     1.6509        0.9301
DM Mathematics                   1.5206     3.1774        2.6229
EuroParl                         1.1202     2.7141        1.4917
Books3                           1.0734     1.2264        1.1366
Wikipedia (en)                   0.8961     1.1807        1.0252
```

Pile model은 22개 row 모두에서 최고다. 세 corpus 모두 web text를 포함하는 Pile-CC gap은 0.09 bit이고, web page가 아닌 ArXiv, Github, DM Mathematics, EuroParl에서는 1.0 bit가 넘는다. Raw CC는 Pile BPB에서 CC-100보다 낫다. 저자는 이를 Wikipedia로 학습한 language model을 쓰고 perplexity가 너무 높거나 너무 낮은 document를 버리는 CC-100 perplexity filter 탓으로 돌린다. Wikipedia와 너무 비슷하거나 너무 다른 document가 제거되는 것이다(§4.2, 해석). **복제됨:** 150B token의 1.2B model에서 Pile/Dolma model은 diverse Paloma domain을 잘 fit한 반면 single-source web corpus(C4, mC4-en, RefinedWeb)는 average perplexity가 더 높았다([[dolma]] §9.2, Fig. 5). [[paloma]] §4.1에서 C4-only 1B baseline은 RedPajama arXiv에서 perplexity 391,171에 이르렀고, C4/mC4-en baseline은 약 20B~150B token 사이 각각 65/43 domain에서 더 나빠졌다.

**[figures/data-landscape.html](figures/data-landscape.html)**의 Panel B는 Table 4 전체 22 row를 plot하고 CC-100/Pile model gap으로 sort한다.

| Source type | 근거가 공급한다고 말하는 것 | Setting | Status |
|---|---|---|---|
| Filtered web crawl | Broad QA에서 가장 큰 single effect: deduplicated Pile의 26.9%인 Pile-CC 제거는 evaluation group 7개 모두를 낮추고 average −4.8([[pretrainers-guide-training-data]] Fig. 8) | 1.5B, Pile variant | 결과(단일 연구) |
| Link-curated web(OpenWebText2) | 제거 시 common-sense QA −5.8, average −1.4 | same | 결과(단일 연구) |
| Books | 제거 시 average −2.7, Web QA −6.3; book은 long document(Books3 mean 538.36 KiB 대 Pile-CC 4.33 KiB, [[the-pile]] Table 1)이며 이는 [[ch-32b]] long-context data에 중요 | same | 결과(단일 연구) |
| Wikipedia | 제거 시 Wiki QA −1.3, toxic generation +4.2(Table 3) | same | 결과(단일 연구) |
| Academic, biomedical | PubMed 제거는 Biomed QA −5.8; 4.9% reference share는 24.2%와 거의 같은 S2ORC perplexity([[dolma]] App. M) | 1.5B; 1B | 각각 결과(단일 연구) |
| Code | Code task는 code share와 함께 상승; 일부 reasoning/structured-output task도 상승(§4) | 374M-4.2B | 복제됨(§4) |
| Math | Pile model DM Mathematics BPB 1.5206 대 CC-100 3.1774; Llama 3는 token 25%를 math/reasoning에 할당([[llama-3]] §3.1.2) | 1.3B; report | 결과(단일 연구) |
| Multilingual | EuroParl BPB 1.1202 대 2.7141; Pile은 fastText 기준 97.4% English([[the-pile]] §5.2); coverage decision은 [[ch-13a]] | 1.3B | 결과(단일 연구) |

**조건과 한계.** Pile comparison은 Pile 자체 component에서 평가하므로 Pile model은 in-distribution, CC model은 out of distribution이다. Table은 각 corpus가 cover하는 domain을 측정하며 general quality를 측정하지 않는다. 저자는 Pile 크기의 약 1/3인 CC-100에 40GB size control이 유리하다고 지적한다(§4.1).

**함의.** §3 QA suite에서 Common Crawl은 total effect가 가장 컸고 books는 data percentage point당 effect가 가장 컸다. Non-web source는 academic paper, code, math, non-English text처럼 web-only corpus가 150B training token에서도 더 나쁘게 fit하는 domain을 공급한다.

## §3 Controlled removal, data age, filter

**Setting.** [[pretrainers-guide-training-data]](Longpre et al., arXiv v1 2023-05)는 T5X에서 1.5B parameter decoder-only model(“LM-XL”) 28개를 batch 4,096, sequence length 512, 88,064 step으로 pretrain했다(App. C, Table 5). 이는 4,096 × 512 × 88,064 ≈ 184.7B token이다(도출). 각 model은 Natural Questions로 fine-tune했고 domain별 27 QA dataset group에서 평가했다(§6, App. C.5). 실험 전 저자는 C4와 Pile을 deduplicate했으므로 아래 data share는 Pile Table 1과 다르다(§2.1).

**Source removal.** Full deduplicated Pile로 학습한 model 대비 Fig. 8 값이다. “% data”는 removal 뒤 남은 share(Table 3)다.

| Removed | % data left | Wiki | Web | Books | Biomed | Academic | Common Sense | Contrast Sets | Average |
|---|---|---|---|---|---|---|---|---|---|
| Social | 98.8 | −0.8 | −3.7 | 2.6 | 0.1 | 3.5 | −3.5 | 3.5 | 0.4 |
| Wikipedia | 97.9 | −1.3 | −5.3 | 3.0 | 0.2 | 0.9 | −4.4 | 7.2 | −0.3 |
| Books | 93.1 | −3.5 | −6.3 | 1.0 | 0.0 | −1.6 | −6.5 | −4.4 | −2.7 |
| OpenWeb | 93.1 | −2.0 | −4.1 | 0.1 | −1.0 | 0.6 | −5.8 | −2.9 | −1.4 |
| Legal | 91.0 | −2.7 | −2.9 | 3.8 | 0.4 | 0.8 | −2.6 | −0.4 | −0.6 |
| Academic | 87.1 | −0.3 | −2.5 | 0.3 | −0.9 | 2.2 | −1.1 | 4.3 | 0.2 |
| PubMed | 85.1 | −0.3 | −3.0 | 3.9 | −5.8 | −1.5 | −5.9 | 3.9 | −1.2 |
| Code | 80.9 | −0.5 | −3.1 | 2.9 | −1.2 | 1.2 | −5.8 | 4.4 | −0.1 |
| Common Crawl | 73.1 | −3.2 | −6.2 | −2.9 | −4.6 | −5.9 | −8.0 | −5.2 | −4.8 |

**Worked example: 제거한 data percent당 effect**(도출; linear effect와 configuration당 one run 가정). Books: −2.7 / 6.9 = percentage point당 −0.39. Common Crawl: −4.8 / 26.9 = −0.18. OpenWeb: −1.4 / 6.9 = −0.20. PubMed: −1.2 / 14.9 = −0.08. Code: −0.1 / 19.1 = −0.005. Data unit당 이 QA suite에서 book은 Common Crawl의 약 두 배 effect였고 code는 거의 없었다. 저자는 QA set이 coding skill을 요구하지 않는다고 지적한다(§6).

**Findings.** Common Crawl 제거는 Academic QA를 5.9 낮췄으며 Academic source 제거(+2.2)보다 컸다. 저자는 Common Crawl/OpenWeb/Books가 academic topic을 포함한 많은 topic을 cover한다고 가정한다(§6, 해석). Best average는 모든 source 또는 거의 모든 source에서 학습한 model이 냈고, 저자는 target task와 관련 없어 보이는 source도 넣으라고 권한다(§6). Toxicity metric은 QA와 다른 방향으로 움직인다. Books 제거는 toxic generation을 6.2 낮추고 toxicity identification도 1.3 낮췄다(Table 3).

**Data age.** 저자는 cutoff가 2013, 2016, 2019, 2022인 Common Crawl에서 C4를 다시 만들고, version당 LM-XL 하나를 pretrain하여 year-split task로 각각 fine-tune하고 모든 year에서 test했다(§4). Temporal degradation(TD)은 pretraining year/evaluation year distance 1년의 expected score change다. LM-XL의 task 5개 mean TD는 0.41이고 score와 year distance의 Pearson r은 0.61이었다. 20M LM-Small은 0.08이었고 저자는 이 size에서 pretraining effect가 significant하다고 찾지 못했다(Table 2, §4). Degradation은 두 방향 모두에서 보였고 evaluation data가 pretraining data보다 새로울 때 더 가팔랐다(Fig. 4). Worked example(도출; linear 가정): 3년 gap은 LM-XL score가 3 × 0.41 ≈ 1.2 낮아질 것을 예측한다.

**Composition change로서의 filter.** Threshold 0.975의 quality classifier(PaLM/GLaM classifier)는 C4 91%를 남기고 QA average를 2.5 올렸다. 0.7에서는 46%를 남기고 average를 0.7 올렸지만 Books QA는 6.7 낮아졌다(Fig. 6). 0.3 toxicity filter는 61%를 남기고 average를 2.7 낮췄다. Least toxic 8%를 제거하는 inverse toxicity filter는 이를 1.7 올렸다(Fig. 7). 저자는 “Quality filtering effects are not easily predicted by dataset characteristics”라고 결론낸다(§5). Filter mechanism은 [[ch-10]], [[ch-10a]]에 있다.

**조건과 한계.** Configuration당 one run, English dataset 두 개, prompted evaluation이 아닌 fine-tuned evaluation, black-box API의 toxicity score(§8)다. **[figures/data-landscape.html](figures/data-landscape.html)**의 Panel C는 Fig. 8 table을 “% data left” column과 함께 heat map으로 보인다.

## §4 Pretraining의 code

**정의.** **Competitive** mixture에서는 total token count가 fixed이므로 code token 하나가 natural-language token 하나를 대체한다. **Additive** mixture에서는 natural-language token count가 fixed이고 그 위에 code를 더한다([[code-pretraining-task-performance]] §3).

```
N_code = m · N_total,   N_lang = (1 − m) · N_total        (competitive: N_total fixed)
N_total = N_lang / (1 − m)                                 (additive: N_lang fixed)
```

- m: code training token fraction; N_total, N_code, N_lang: total, code, language token count.

**Worked example.** N_lang = 132B(Petty et al. §4.1 base volume)일 때, m = 0.5인 additive mixture는 N_total = 132 / 0.5 = 264B token이다. m = 0.2이면 165B token이고 그중 33B가 code다. m = 0.5 competitive run은 language token 66B다. Competitive setting에서 knowledge task가 하락하면 language token이 적어서일 수 있으며, additive setting이 이 effect를 분리한다.

**근거 1: [[to-code-or-not-to-code]]**(Aryabumi et al., Cohere, arXiv v1 2024-08). Text는 GitHub/StackExchange를 제외한 SlimPajama(503B token), code는 filtered Stack subset(139B token)과 markup 및 proprietary 3.2B-token synthetic Python set이다(§2.1). Table 2의 row다. Caption은 size를 말하지 않지만 value는 §3.1/§3.6의 470M result와 일치한다. Reasoning은 task 11개 average, knowledge는 TriviaQA/NQ average, code는 HumanEval/MBPP pass@1 average다.

```
Variant           Recipe                Text    Code   Reason.  Know.  Code
Text-only         Pre-training          400B    -      49.0     9.5    0.4
Balanced-only     Pre-training          200B    200B   51.8     8.1    9.0
Balanced → Text   Pre-training Init.    100B    100B   52.0     7.4    7.8
                  Continue Pre-train.   +180B   +20B   53.0     9.9    4.8
                  Cooldown              +32B    +8B    54.9     10.9   5.8
```

같은 400B token의 text-only 대비 balanced→text는 reasoning을 (53.0 − 49.0) / 49.0 = 8.2%, knowledge를 (9.9 − 9.5) / 9.5 = 4.2% 올리고, code는 4.8 / 0.4 = 12× 올렸다(§3.6). Proportion sweep(scratch에서 model 6개, 200B token)에서는 25% code가 reasoning/knowledge average 최상이었다. Knowledge는 25% code에서 no-code 대비 3.4% 낮고, 75%에서 31%, 100% code에서 86% 낮았다. Code score는 code share에 거의 선형으로 상승했다(§3.3). Code 20%의 40B-token cooldown은 no-cooldown 대비 reasoning 3.6%, knowledge 10.1%, code 20%를 올렸지만, code 없는 cooldown은 reasoning/code를 올리지 못했다(§3.5). Trend는 2.8B에서도 유지됐다(§3.2).

**근거 2: [[code-pretraining-task-performance]]**(Petty et al., arXiv v1 2024-09). 374M model, C4 text/Pile GitHub code, mixture별 seed 5개다(§4.1). Output이 formal structure일 때 code share는 compositional generalization을 개선했다. COGS-vf structural generalization에서 code share 대비 accuracy의 best-fit slope는 competitive 0.147, additive 0.165였고, 저자는 0.147을 0%→100% code의 predicted accuracy 14.7% 증가로 읽는다(§5). Multi-digit arithmetic은 개선됐으며 competitive setting에서는 40–50% code에서 peak였다(Fig. 4). English passivization 및 BigBench general knowledge, fantasy reasoning, implicatures, common morpheme accuracy는 두 setting 모두 code share와 함께 떨어졌다(Figs. 5-6). Competitive setting의 모든 BigBench multiple-choice task에서 task별 slope는 permutation보다 variance가 높고(p = 0.0002) upper quartile도 높았지만(p = 0.006), mean difference는 significant하지 않았다(p = 0.1038, Fig. 8). 그럼에도 저자의 discussion은 aggregate improvement를 기술한다(§6).

**근거 3.** 84B-token budget에서 4.2B parameter model은 token 최대 50%를 Python으로 대체해도 19-task natural-language average가 낮아지지 않았고, bAbI는 no-code 0.0에서 50% code 23.2가 되었다([[data-constrained-scaling]] §7, App. M Table 10, seed 5개). 1B에서 code를 0%→5%→15%로 올리면 bAbI in-context exact match는 0.0→8.8→10.1이었다([[dolma]] App. M Table 3).

**상태.** **복제됨:** moderate code share는 natural-language average를 낮추지 않고 state-tracking, arithmetic, structured-output task를 올린다([[to-code-or-not-to-code]], [[code-pretraining-task-performance]], [[data-constrained-scaling]], [[dolma]]). **복제됨:** code share가 커지면 world-knowledge task는 하락한다([[to-code-or-not-to-code]] §3.3; [[code-pretraining-task-performance]] §5). **결과(단일 연구):** purely linguistic task도 하락했다([[code-pretraining-task-performance]] §5). **해석:** Petty et al.은 absolute 또는 relative natural-language exposure 감소를 원인으로 돌린다(§6).

**조건과 한계.** 4.2B data-constrained run 외에는 모든 model이 2.8B 이하이고, cooldown을 포함한 모든 budget은 440B token 이하이다. Aryabumi et al./Petty et al. 어느 쪽도 code share가 SFT/RL과 상호작용하는 방식을 측정하지 않았고([[code-pretraining-task-performance]] §6), Aryabumi et al.은 safety를 연구하지 않았다(§6).

**함의.** General model에서 code share는 한쪽의 code/structure task와 다른 쪽의 token당 knowledge 사이 trade다. Knowledge cost는 natural-language average와 분리해 knowledge task를 보고할 때만 보인다.

## §5 Llama 3 mixture와 선택 방법

[[llama-3]](arXiv:2407.21783, v1 2024-07)는 §3.1.2에서 mixture 선택 방식을 설명한다. 일부를 생략한 passage는 다음과 같다.

> “Knowledge classification. We develop a classifier to categorize the types of information contained in our web data to more effectively determine a data mix. We use this classifier to downsample data categories that are over-represented on the web, for example, arts and entertainment. Scaling laws for data mix. To determine the best data mix, we perform scaling law experiments in which we train several small models on a data mix and use that to predict the performance of a large model on that mix (see Section 3.2.1). [...] Data mix summary. Our final data mix contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.”

**Mechanism, step by step.**

1. URL/document/line deduplication, heuristic, model-based quality classifier(Wikipedia가 reference한 text를 recognize하도록 학습한 fastText model, Llama 2 quality judgment로 학습한 DistilRoberta model)로 web data를 filter하고, code/math page를 extract하는 별도 classifier를 쓴다(§3.1.1).
2. Web document를 knowledge category로 label하고 over-represented category를 downsample한다(§3.1.2).
3. Candidate mix마다 small model 여러 개를 학습하고 scaling law를 fit해 large-model performance를 predict한다. Mix 여러 개에서 반복한다(§3.1.2, §3.2.1의 method 및 [[ch-08a]]).
4. 선택한 candidate로 larger model을 학습하고 key benchmark를 점검한다(§3.1.2).
5. 405B run 중 mixture를 바꾼다. Later stage에서 non-English data를 늘리고, math를 upsample하며, web data를 더 recent하게 만들고, lower-quality subset을 downsample한다. Percentage는 인쇄되지 않았다(§3.4.1, [[llama-3-recipe]]).
6. New small dataset은 50%-trained 8B model을 learning rate 0까지 40B token에 걸쳐 anneal하여 평가한다. New dataset weight 30%, default mix 70%다(§3.1.3).

**Unit error worked example.** 17% × 15.6T = 2.65T는 405B model이 본 code token 수가 아니다. Report가 든 이유는 둘이다. Training 중 mix가 바뀌었고(§3.4.1), category share는 final mix에 대해 “roughly” 주어진다. Derived number는 source가 없으므로 recipe table에 넣으면 안 된다.

**보고되지 않은 것.** Source별 share, “general knowledge” 정의, arts/entertainment 외 knowledge-classifier category, mix scaling-law experiment result, data 자체다. [[deepseek-v3]] §4.1은 14.8T-token corpus가 DeepSeek-V2 대비 math/programming sample ratio를 올리고 multilingual coverage를 넓힌다고만 말하며 mixture percentage는 주지 않는다.

**Topic reweighting이 breadth를 바꾼다는 열린 근거.** FineWeb-Edu educational classifier는 “Education, Learning, Teaching”(+3.2%), “History, Culture, Politics”(+2.2%)를 올리고 “Entertainment, Film, Theater”를 downsample했다([[fineweb]] §4.1). [[weborganizer]]에서 200B-token pool의 29B token으로 학습한 1B model은 MMLU/HellaSwag target topic × format mixture로 task 9개 average를 51.6에서 54.6으로 올렸다(Table 1). FineWeb-Edu selection은 HellaSwag를 random-selection baseline보다 1.5 낮췄고, MMLU만 tune한 mixture는 HellaSwag 57.5→54.1, PIQA 71.3→69.9를 만들었다(Table 10). Domain 안에서 FineWeb-Edu의 implicit topic × format mixture를 random하게 sample하면 average gain의 84%를 회복했다(Table 2). **해석:** Quality classifier는 부분적으로 topic reweighting으로 작동한다. Entertainment content를 downsample한 독립 report 둘(Llama 3, FineWeb-Edu)은 targeted benchmark 밖 task에서 측정해야 할 composition decision을 했다.

**Model-based web filtering 뒤 curated source.** [[dclm]] Table 6(1B-1x scale)에서 Llama 1/RedPajama ratio(67% Common Crawl, 33% Wikipedia/books/StackExchange/arXiv/GitHub)로 high-quality non-web source를 섞으면 C4 Core는 2.2, RedPajama-CC는 1.7, RefinedWeb은 1.4 올랐지만 DCLM-Baseline은 1.2 낮아졌다. DCLM Core는 small scale의 low-variance signal을 위해 선택한 ARC, HellaSwag, BoolQ, CommonsenseQA, BIG-bench task 6개 등을 포함한 task 22개다(App. G.1). Academic/code/book text의 per-domain loss가 없으므로 이 결과는 §2의 per-domain fit과 다르게 측정됐다. [[nemotron-cc]]는 모든 1T-token 8B comparison에서 fixed 27% block(books/patents 9%, papers 9%, code 5%, conversational 3%, Wikipedia 1%)을 유지하고 73% English Common Crawl 부분만 바꿨다(§3.1, App. D Table 12). **열린 질문:** Classifier-filtered web text 위에 curated non-web source가 breadth를 더하는지는 measurement suite에 의존하며, 위 source 중 어느 것도 같은 run에서 두 suite를 모두 시험하지 않았다.

## §6 Open corpus와 각각이 공개하는 것

| Corpus | Size와 unit | Source | 공개한 것 | 보고하지 않은 것 또는 한계 |
|---|---|---|---|---|
| The Pile([[the-pile]], 2020-12) | raw 825.18 GiB, effective 1254.20 GiB | component 22개 | data, component별 weight/epoch(Table 1), preprocessing code | component별 token; weight ablation |
| Dolma v1.6([[dolma]], 2024-01) | Llama token 3,059B | Common Crawl 2,479B; GitHub 411B; Reddit 89B; Semantic Scholar 70B; Gutenberg 6.0B; Wikipedia/Wikibooks 4.3B(Table 1) | data(ODC-By), toolkit, 1.2B filter ablation | OLMo-1B training에 쓴 sampling rate |
| FineWeb, FineWeb-Edu([[fineweb]], 2024-06) | GPT-2 token 15T, 1.3T | Common Crawl snapshot 96개; web only | data, datatrove code, classifier, annotation, ablation model | code가 prevalent하지 않을 가능성(App. A) |
| DCLM([[dclm]], 2024-06) | GPT-NeoX token pool 240T; DCLM-Baseline 3.8T | 2023 이전 Common Crawl 전체 | pool, baseline data, task 53개 evaluation, scale 5개 training recipe | run-to-run variation을 충분히 탐색하지 않음; tokenizer 하나(§6) |
| Nemotron-CC([[nemotron-cc]], 2024-12) | token 6.3T: deduplicated real 4.4T, synthetic 1.9T | 2013-20~2024-30 Common Crawl snapshot 99개 | Common Crawl Terms of Use 아래 data, quality classifier, NeMo Curator code | comparison은 다른 dataset의 fixed 27% non-crawl blend 사용(Table 12) |
| OLMo 2 Mix 1124([[olmo-2]], 2025-01) | token 3.90T | DCLM-Baseline 3.71T; StarCoder 83.0B; peS2o 58.6B; arXiv 20.8B; OpenWebMath 12.2B; Algebraic Stack 11.8B; Wikipedia 3.7B(Table 4) | data, source별 count, mid-training mix | pretraining mix ablation |
| Llama 3([[llama-3]], 2024-07) | token 15.6T(405B model) | category 4개 | category share, selection method | source별 share, data |
| DeepSeek-V3([[deepseek-v3]], 2024-12) | token 14.8T | 나열하지 않음 | V2에서 바뀐 방향 | mixture percentage |

**Unique token 대 total token.** Globally deduplicate하지 않았다면 corpus token count에는 repeated content가 포함된다. [[nemotron-cc]] Table 4는 global fuzzy deduplication 뒤 unique token을 estimate한다. DCLM은 total 3.8T/unique 1.0T, FineWeb-Edu는 1.3T/0.2T, Nemotron-CC는 total 6.3T, unique real 4.4T/synthetic 1.9T다. Worked example: DCLM만에서 뽑는 15T-token run은 unique token 1.0T를 약 15회 본다. Nemotron-CC real token만에서 뽑는 run은 약 15 / 4.4 ≈ 3.4회 본다(도출). [[data-constrained-scaling]]은 약 4 epoch 이후 repeated token당 value가 감소한다고 찾았다(세부는 [[ch-14]]). Nemotron-CC는 저자 own evaluation harness에서 MMLU 70.3을 기록한 8B model의 15T token 중 7.2T를 제공했다. 이는 Llama 3.1 8B의 65.3과 대비된다(§3.2, Table 6).

**Stock의 한계.** [[epoch-data-stock]](Villalobos et al., arXiv v1 2022-11, v2 2024-06)은 indexed web을 510T token(95% CI 130T–2,100T), Common Crawl을 130T(100T–260T)로 estimate한다(Table 1). Training set이 public human text effective stock 약 4e14 token에 도달하는 시점은 2026~2032년이며 median은 2028년으로 project한다(Abstract, §2.5). 이 estimate는 DCLM pool과 다른 tokenizer/counting rule을 쓰므로 240T와 130T는 직접 비교할 수 없다.

## §7 Opt-out과 consent restriction을 측정된 coverage 감소로 보기

**정의.** Consent restriction은 crawling 또는 AI use를 불허하는 website의 machine-readable(robots.txt) 또는 legal(Terms of Service) statement다. [[consent-in-crisis]](Longpre et al., arXiv v1 2024-07)는 7개 AI organization crawler 중 적어도 하나를 완전히 block한 web domain에서 corpus에 온 token을 “restricted token”으로 센다(§2).

**Setting.** C4, RefinedWeb, Dolma의 바탕인 web domain 14,000개다. Corpus별 token 기준 top 2,000 domain(union 뒤 3.95k, “HEAD”)과 random domain 10,000개다. 2016년 1월~2024년 4월 monthly Wayback Machine snapshot의 robots.txt/Terms page를 썼다(§2).

**근거.**

- Full corpus 전반에서 2023년 중반 C4/RefinedWeb/Dolma token 약 1%가 robots.txt-restricted였고, 2024년 4월에는 5–7%였다. Head domain에서는 under 3%에서 20–33%가 되었다(§3.1, Fig. 2a).
- Restriction은 domain type별로 집중된다. Head sample의 news-website token 거의 45%가 fully restricted였으며 2023년에는 3%였다. 저자는 robots.txt-respecting crawl이 news/social media/forum에서 organization/e-commerce site 쪽으로 이동할 것으로 예상한다(§3.1, 해석).
- Restriction은 crawler별로 다르다. C4 head sample에서 OpenAI crawler는 token의 25.9%, Anthropic/Common Crawl은 13.3%, Google AI crawler는 9.8%, Cohere는 4.9%, Meta는 4.1%에서 restricted였다(Fig. 1).
- Token의 45–55%에 해당하는 Terms of Service page에는 어떤 data-use restriction이 있지만, 대부분 crawler는 Terms page를 읽지 않는다(§3.1, Fig. 2c).
- 같은 taxonomy로 label한 WildChat conversation(저자가 cluster한 100개, GPT-4o가 label한 1,000개)에서 30% 초과가 fiction, role-play, poetry 같은 creative composition을 요청했다. 저자는 이것이 해당 corpus web domain에 잘 representation되지 않는다고 찾았다(§3.4).

**Worked example(도출).** 7개 organization 전체에 대해 robots.txt를 존중하고 C4 같은 domain distribution으로 web slice를 만드는 pipeline은 overall web token 약 5–7%를 잃는다. Composition target에 head domain의 news가 포함되면 그 domain 안 loss는 약 45%다. 남은 news share를 복원하려면 남은 것을 upweight해야 하며, 이는 이를 repeat한다.

**조건과 한계.** 저자는 restriction이 없다는 것이 consent를 뜻한다고 주장하지 않는다(§3.1). Fully restricted domain만 세므로 partial restriction은 더 높다(§3.1). WildChat sample은 proxy service 하나에서 왔고 다른 user를 represent하지 않을 수 있다(§3.4).

**함의.** Consent restriction은 domain 전반에 uniform하지 않은 composition change다. 따라서 더 엄격한 crawling policy 아래 rebuilt corpus는 old corpus의 작은 copy라고 가정하지 말고 domain별로 다시 측정해야 한다.

## §8 Composition change가 coverage를 넓혔는지 측정하기

1. **Domain별 loss.** Domain 전체의 macro-averaged perplexity 또는 BPB, `(1/|D|) Σ_{d∈D} ppl(d)`를 보고한다. D는 evaluation domain set, ppl(d)는 domain d의 perplexity다. Token-weighted(micro) average 하나를 쓰지 않는다. Micro average는 evaluation token을 가장 많이 기여하는 domain 쪽으로 weight되기 때문이다([[paloma]] §4.1). Evaluation text against training data를 decontaminate하고 document마다 따로 score한다. 같은 Pythia 1.4B checkpoint에서 concatenated input은 perplexity 92.23 ± 17.33, separate document는 42.57 ± 0.29였다(App. H, Table 17).
2. **Centered score를 가진 broad task suite.** DCLM Core는 task 22개, Extended는 53개를 average하며, task accuracy a는 random-guess accuracy r을 써 `(a − r) / (1 − r)`로 rescale한다([[dclm]] §3.5). Worked example: four-choice task 40% accuracy는 (0.40 − 0.25) / 0.75 = 0.20, 25%는 0이다. Centering이 없으면 chance의 two-choice task 50%는 four-choice task 45%(centered score (0.45 − 0.25) / 0.75 = 0.27)보다 더 크게 count된다.
3. **Transfer되는 small-scale decision.** [[dclm]] Fig. 3에서 400M-1x, 1B-1x, 3B-1x의 curation method 10개 Core score는 7B-1x score와 Pearson r = 0.838, 0.956, 0.982로 correlate했다. [[datadecide]]에서 150M model로 corpus 25개를 rank하면 pairwise comparison 약 80%에서 1B winner를 고른다. 그러나 SocialIQA/BoolQ decision은 시험한 scale에서 unreliable했다(§3.1). Proxy-scale run이 data를 선택할 때는 그 scale에서 candidate corpus를 분리하는 task로 rank한다. [[datadecide]]에서 다른 task ranking은 1B winner를 predict하지 못했기 때문이다.
4. **Contamination check.** DCLM-Baseline에서 detected MMLU/HellaSwag overlap을 제거해도 score는 낮아지지 않았다(7B-2x에서 MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4; [[dclm]] Table 7). Dolma에는 여러 test set이 완전히 들어 있고 그중 다수는 code subset에 있다([[dolma]] App. L). Method는 [[ch-48]]에 있다.

## Negative sample과 negative feedback

이 section은 course standard의 “negative” 네 의미를 composition에 적용한다. 전체 처리는 [[ch-31a]], [[ch-43a]]에 있다. Pretraining composition은 의미 1과 2만 쓴다. **Negative marginal value(1):** quality/toxicity filter가 제거한 document는 inclusion이 performance를 낮출 sample로 취급한다. §3 근거는 label이 task-dependent임을 보인다. C4 61%를 남긴 toxicity filter는 QA average 2.7과 toxicity identification을 낮췄고, least-toxic document를 제거한 inverse filter는 toxicity identification 최고를 냈다([[pretrainers-guide-training-data]] §5). **Content로서 negative(2):** corpus에 남긴 toxic/low-quality text는 일반 cross-entropy로 학습한다. 이러한 text를 recognize하는 ability를 더하고 toxic generation도 올린다(Table 3). 이 장 pretraining source는 meaning 3(conditioning token)이나 meaning 4(document likelihood를 낮추는 gradient)를 적용하지 않으므로 이 stage에는 likelihood displacement가 없다. Control은 recognition task를 포함한 suite에서 정한 threshold로 filter하고, domain별 removed share를 보고하며, 저자가 제안하듯 behavior suppression을 post-training으로 옮기는 것이다(§7). Diagnostic은 toxicity identification/toxic generation을 나란히 보고하고 filter threshold별 domain QA change를 기록한다.

## 레시피

| Model (정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama 3.1 (all) | 8B-405B | pretrain-stable | final data mix(token share) | general knowledge 약 50%, mathematical/reasoning 25%, code 17%, multilingual 8% | arXiv:2407.21783v3 §3.1.2 ([[llama-3-recipe]]) | verified 2026-09-14 | candidate mix별 small-model scaling law, candidate larger model; classifier가 over-represented web category downsample; table 없음 |
| Llama 3.1 405B | 405B | pretrain-stable | training 중 mix change | non-English 증가; math upsample; 후반 more recent web; lower-quality subset downsample; percentage 미인쇄 | arXiv:2407.21783v3 §3.4.1 | verified 2026-09-14(percentages not reported) | ablation 없음 |
| Llama 3 8B(experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50%-trained 8B; 40B token에 걸쳐 LR 0으로 linear anneal; new dataset 30%, default 70% | arXiv:2407.21783v3 §3.1.3 | verified 2026-09-14 | small dataset별 scaling-law run보다 efficient하다고 명시; 수치 없음 |
| The Pile(corpus) | n/a(data) | pretrain-stable | component byte share/epoch | Pile-CC 18.11%(1.0); PubMed 14.40%(2.0); Books3 12.07%(1.5); OWT2 10.01%(2.0); ArXiv 8.96%(2.0); Github 7.59%(1.0); FreeLaw 6.12%(1.5); StackExchange 5.13%(2.0); Wikipedia 1.53%(3.0); Table 1의 smaller 13개 | arXiv:2101.00027v1 Table 1 | verified 2026-09-15 | higher-quality component는 “Following Brown et al. (2020)”(§2); ablation 없음 |
| OLMo 2 7B/13B/32B | 7B,13B,32B | pretrain-stable | OLMo 2 Mix 1124(token) | DCLM 3.71T; StarCoder 83.0B; peS2o 58.6B; arXiv 20.8B; OWM 12.2B; Algebraic 11.8B; Wiki/Wikibooks 3.7B; total 3.90T | arXiv:2501.00656v3 §2.4.1 Table4 | verified 2026-09-15 | pretraining mix ablation 없음 |
| OLMo 2 7B/13B/32B | 7B,13B,32B | pretrain-stable | pretraining token | 3.90T(7B);5T(13B);6.06T(32B) | §2.3 Overall | verified 2026-09-15 | ablation 없음; Table9 caption은 같은 checkpoint에 4T/5T/7T(보고서 내 conflict) |
| DCLM-Baseline data | n/a(data) | pretrain-stable | model filter | fastText: positive OpenHermes2.5+r/ELI5, negative RefinedWeb reproduction; top 10% | arXiv:2406.11794v4 §4.4 | verified 2026-09-15 | Table5(7B-1x): Core 41.0(10%) vs39.8(15%) vs38.7(20%); Wikipedia positive 35.7 |
| DCLM mixing ablation | 1.4B(1B-1x,28.8B) | pretrain-stable | mixture | CC subset 67%, Wikipedia/books/StackExchange/arXiv/GitHub 33%(Llama1/RPJ ratio) | arXiv:2406.11794v4 §4.5 Table1 | verified 2026-09-15 | Table6 Core +2.2(C4),+1.7(RPJ),+1.4(RefinedWeb),−1.2(DCLM) |
| DCLM 7B(final) | 6.9B | pretrain-stable/decay/long-context | data/stage | dataset4.1T(DCLM3.8T+StarCoder+ProofPile2);2.5T; cooldown 200B/270B: 70% tight fastText+30% math, souped; context 2048→8192에100B | arXiv:2406.11794v4 §5 | verified 2026-09-15 | Table8(2.6T listed): Core57.1,MMLU63.7,Extended45.4; 70/30 ablation 없음 |
| Nemotron-CC comparison | 8B | pretrain-stable | 1T blend | English CC73%; books/patents9%; papers9%; code5%; conversational3%; Wiki1% | arXiv:2412.02595v2 §3.1 App.D Table12 | verified 2026-09-15 | Table5 MMLU59.0(HQ) vs53.4(DCLM) vs42.9(FineWeb-Edu); non-crawl27% 미변경 |
| Nemotron-CC long horizon | 8B | pretrain-stable,two phase | English CC phase share | phase1 9T의59%(5.31T); phase2 6T의31%(1.86T); total15T의47.8%(7.17T) | App.E(§3.2 rounds 7.2T) | verified 2026-09-15 | Table6 MMLU70.3 vs Llama3.1 8B65.3; phase share ablation 없음(curriculum Feng et al.) |
| To Code model | 470M | pretrain-stable | scratch code share | 200B token의 0/25/50/75/90/100% | arXiv:2408.10914v1 §3.3 | verified 2026-09-15 | 25% reasoning+knowledge 최고; knowledge −3.4%(25%),−31%(75%) |
| Data-constrained code | 4.2B | pretrain-stable | 84B Python share | 0–90%, 10% step(The Stack) | arXiv:2305.16264 §7 App.M Table10 ([[data-constrained-scaling]]) | verified 2026-09-14 | 50%까지 19-task average drop 없음(Fig6, seed5) |
| Dolma mixture ablation | 1B | pretrain-stable | web/code/reference/books | WebOnly100/0/0/0%; Reference+81.2/13.5/4.9/0.4%; Gopher68.4/5.4/24.2/2.0% | arXiv:2402.00159v2 App.M Table4 ([[dolma]]) | verified 2026-09-14 | Fig12 WebOnly가 HumanEval/S2ORC perplexity 높음; Reference+≈Gopher S2ORC |
| Pretrainer's Guide LM-XL | 1.5B | pretrain-stable | batch;seq;step | 4,096;512;88,064(≈184.7B, derived) | arXiv:2305.13169v2 App.C Table5 | verified 2026-09-15 | Wang et al. 2022 default; ablation 없음 |
| To Code balanced→text | 470M | pretrain-stable→decay | stage token | init100B text+100B code; continue+180B text/+20B code; cooldown+32B/+8B, linear LR anneal 1e-6 | arXiv:2408.10914v1 Table2 §3.5 | verified 2026-09-15 | cooldown 뒤 reasoning54.9/knowledge10.9, text-only cooldown54.1/11.1 |

**작은 범용 run의 시작점.** 84B–200B token으로 470M–4.2B from-scratch model을 학습하면 code 25–50%는 tested range 안이다. 470M/200B에서 25%는 reasoning+knowledge 최고였고, 4.2B/84B에서 Python 50%는 19-task average를 낮추지 않았다. 470M에서 25% code의 knowledge가 3.4% 낮았으므로 knowledge task를 따로 보고한다. Fixed non-crawl block을 유지한다. Verified size는 8B/1T Nemotron-CC의 27%(books/patents9%, papers9%, code5%, conversational3%, Wiki1%)와 1.4B DCLM mixing run의 33%다. DCLM run은 web portion이 DCLM-Baseline이면 이런 block이 Core를 낮출 수 있음을 보인다. DCLM mixing/filter row처럼 centered multi-task suite로 score한 1B-1x(28.8B) run으로 candidate web subset을 고르고, §8의 1B-1x→7B-1x transfer를 본다. Later-stage mix에서 작은 new source는 Llama3 절차, 즉 50% 학습한 8B에 대해 candidate 30%를 넣어 40B token anneal로 판단한다. 8B 미만 model의 multilingual/math share를 주는 verified row는 없다.

## 일반화 렌즈

**(a) 무엇이 breadth를 늘리는가.**
- Heterogeneous source: 1.3B에서 22-source Pile은 component 22개 모두에서 CC-100보다 낮은 BPB였고([[the-pile]] Table4), 1.2B Dolma Paloma domain([[dolma]] §9.2)과 1B Paloma([[paloma]] §4.1)에서 복제됐다.
- All 또는 거의 all source 유지: best QA average는 all/nearly all Pile source model에서 나왔다([[pretrainers-guide-training-data]] §6).
- 470M code 25%: no-code 대비 reasoning +3.4%, knowledge −3.4%; 75% code에서는 knowledge −31%였고([[to-code-or-not-to-code]] §3.3), 4.2B의 50%는 natural-language average를 유지했다([[data-constrained-scaling]] §7).
- Evaluation period와 시간상 가까운 data: 1.5B mean TD는 year당 0.41이고 fine-tuning으로 제거되지 않았다([[pretrainers-guide-training-data]] Table2).

**(b) 무엇이 narrowing/forgetting을 일으키는가.**
- Heterogeneous source 제거: QA average Common Crawl −4.8, Books −2.7([[pretrainers-guide-training-data]] Fig8).
- High code share: 75% code knowledge −31%,100% −86%; code share 증가와 linguistic/knowledge task accuracy 하락([[to-code-or-not-to-code]] §3.3; [[code-pretraining-task-performance]] §5).
- Benchmark 하나에 tune한 filter/mixture: MMLU-target mixture는 HellaSwag −3.4, PIQA −1.4([[weborganizer]] Table10); Wikipedia와 너무 비슷/다른 document를 버리는 perplexity filter의 CC-100은 raw CC보다 Pile BPB가 나빴다([[the-pile]] §4.2, 저자 해석); toxicity filter는 QA average/toxicity identification을 낮췄다([[pretrainers-guide-training-data]] §5).
- Domain별로 고르지 않은 consent source loss: 2024-04 head-domain news token 약45% restricted([[consent-in-crisis]] §3.1).

**(c) 이 stage에서 측정.** Decontaminated suite의 macro-averaged per-domain loss([[paloma]]), DCLM Core/Extended 같은 centered multi-task suite([[dclm]] §3.5), knowledge/reasoning/code/generation 분리 score([[to-code-or-not-to-code]] §2.2), data-age year-split task([[pretrainers-guide-training-data]] §4)를 보고한다. Known error: in-distribution evaluation은 component가 test set을 정의한 corpus에 유리([[the-pile]] Table4); proxy scale chance 근처 task는 ranking unreliable([[datadecide]] §3.1); 다른 lab의 model 재평가는 original number와 다를 수 있음([[nemotron-cc]] Table6 note); concatenated evaluation document는 variance inflate([[paloma]] App.H).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| Raw-size share를 mixture weight로 읽음 | Share 합이100%가 아니거나 report weight와 다름 | `w_i = S_i e_i / Σ S_j e_j` 재계산(Pile Table1) |
| Byte/token/document share 혼합 | Code/math가 source report보다 작거나 크게 보임 | 모든 share 옆 unit/tokenizer 기록; source별 byte/token으로 변환 |
| Final category share×total token | Report에 없는 “code tokens seen” 수 | training 중 mix change(Llama3 §3.4.1) 점검; not reported 표시 |
| Composition을 average 하나로 판단 | Average 상승, domain/task group 하락 | 전후 domain loss와 knowledge/reasoning/code group score |
| In-distribution domain만 평가 | Test component 정의 corpus가 항상 승리 | 비교 corpus 어느 것도 component가 아닌 evaluation domain 추가 |
| Filter를 quality-only로 취급 | Filter 뒤 entertainment/fiction/news topic share 하락 | kept/removed document에 topic/format classifier([[weborganizer]]) |
| Knowledge check 없이 code 추가 | Natural-language average flat, TriviaQA/NQ 하락 | code share 전반 world-knowledge task 별도 추적 |
| 모든 task에서 small-scale rank transfer 가정 | Target scale에서 proxy ranking flip | Proxy scale chance 이상 task, held-out scale pair rank correlation |
| New consent rule crawl 재구축 뒤 old mix weight 재사용 | 이전 corpus와 domain share 상이; news/forum eval 하락 | Training 전 rebuilt corpus domain별 token share 측정 |

## 이해를 점검하세요

1. Pile model은 Pile-CC에서 CC-100보다 0.09 bit, ArXiv에서 1.0 bit 초과로 이겼다. Component별 gap이 다른 이유와 이것이 Pile의 일반적 우월을 보이지 못하는 이유를 설명하라.
2. Academic source 제거는 Academic QA를 2.2 올렸고 Common Crawl 제거는 5.9 낮췄다. 저자 hypothesis와 일관된 causal account, 이를 시험할 experiment를 제시하라.
3. Competitive code mixture에서 code share 증가와 world-knowledge 하락을 만들 두 mechanism과 additive setting이 이를 어떻게 분리하는지 설명하라.
4. Llama3 classifier가 arts/entertainment를 downsample한다. FineWeb-Edu/WebOrganizer로 어떤 evaluation이 하락할 수 있는지, MMLU mix search가 왜 이를 못 찾는지 설명하라.
5. Curated source가 C4 Core는 올리고 DCLM-Baseline은 낮췄다. fastText가 이미 고르는 것/Core suite 측정에 관련한 explanation 둘을 제시하라.
6. Temporal degradation이 year-matched fine-tuning 뒤에도 남고 1.5B에서 20M보다 컸다. Pipeline에서 data freshness decision 순서에 주는 함의를 설명하라.
7. New robots restriction을 지키는 crawl이 overall 5–7%, head-news 약45% token을 잃는다. Remaining news upweight가 repetition/coverage를 모두 바꾸는 이유를 설명하라.

## 연결

- 순서상 의존/이전: ch-08b — Why Next-Token Pretraining Produces General Ability: In-Context Learning, Emergence, and Predictability.
- 다음: ch-10 — Heuristic Curation Pipelines: CCNet, C4, Dolma, FineWeb.
- ch-00 — What General Capability Means and How It Is Measured.
- ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability.
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection.
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic.
- ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale.
- ch-13a — Multilingual Coverage and Vocabulary as Capability Axes.
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination.
- ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures.
- ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement.
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL.
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression.
- ch-48 — Contamination Detection and Its Effect on Reported Scores.
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.

## 출처

- [[the-pile]] — Table1 component size/weight/epoch; BPB definition; Table3 size-controlled comparison; Table4 component BPB; English share. Library card에 수치가 없으므로 chapter excerpt 사용.
- [[pretrainers-guide-training-data]] — 1.5B source-removal(Fig8/Table3), data age(Table2/Fig4), quality/toxicity filter(Fig6-7), training setting(Table5).
- [[to-code-or-not-to-code]] — code proportion sweep, initialization variant(Table2), code cooldown, 2.8B scaling.
- [[code-pretraining-task-performance]] — competitive/additive mixture, compositional slope, arithmetic/knowledge/linguistic decline, permutation test.
- [[data-constrained-scaling]] — 4.2B Python 50%까지 natural-language loss 없음, repetition limit.
- [[llama-3]], [[llama-3-recipe]] — §3.1.2 knowledge classifier/mix scaling/final mix; §3.1.3 annealing evaluation; §3.4.1 mix change.
- [[deepseek-v3]] — closed report 비교: 14.8T token, mixture percentage 없음.
- [[dolma]] — v1.6 source size, Paloma domain fit, mixture/code-share ablation, contamination.
- [[fineweb]] — FineWeb/FineWeb-Edu size/release, educational classifier topic shift.
- [[dclm]] — Pool, fastText filter ablation, curated mixing, Core/Extended metric, cross-scale correlation, decontamination, final7B recipe.
- [[nemotron-cc]] — unique/total token,73/27 blend,15T 8B comparison.
- [[olmo-2]] — OLMo2 Mix1124 composition/stage token count(verified value는 chapter excerpt).
- [[weborganizer]] — topic×format mixture, quality-filter implicit mixture, MMLU-target narrowing.
- [[paloma]] — macro domain perplexity, CC-only baseline, evaluation guideline.
- [[datadecide]] — corpus25개 small-scale decision accuracy/task dependency.
- [[doremi]] — Pile default weight가 8B에서 optimal이 아니었다는 근거.
- [[consent-in-crisis]] — time/domain/crawler별 robots.txt/ToS restriction, WildChat mismatch.
- [[epoch-data-stock]] — Common Crawl/indexed web/effective public text stock estimate.
