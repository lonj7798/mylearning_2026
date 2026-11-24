<!-- chapter: ch-28
     track: synthetic
     kind: content
     title: Long-Context Data Synthesis and Synthetic Evaluation Task Families
     deps: [ch-27, ch-32c]
     sources: [[needle-in-haystack-data]], [[excerpts/ruler-task-family]], [[ruler]], [[babilong]], [[helmet]], [[nolima]], [[excerpts/longalign-pipeline]], [[longalign]], [[excerpts/longmit-mimg]], [[excerpts/llama3-staged-schedule]], [[llama-3]], [[llama-3-recipe]], [[qwen-long-context-synth]], [[prolong]], [[prolong-recipe]], [[excerpts/artificial-needles-kv-retrieval]], [[in2-film]], [[context-synthesis-short-to-long]], [[nextlong]], [[lost-in-the-middle]], [[pose-synthesis]], [[excerpts/primary-quotes-2026-09]]
     figures: figures/task-family-generator.html, figures/context-extension.html
     revised: 2026-09 (generality revision)
-->

# Chapter 28 — Long-Context Data Synthesis와 Synthetic Evaluation Task Family

> **핵심 통찰.** 같은 program-built generator(needle, key-value list, chunk에 걸친 multi-hop question)를 long-context training data와 long-context evaluation task 모두를 만드는 데 사용하며, synthetic task는 generator가 분리한 skill만 측정한다. 128K에서 instruction-tuned model 35개를 비교하면 NIAH와 ∞Bench QA의 Spearman `ρ = 0.63`인 반면, RAG task HotpotQA는 `0.88`이다([[helmet]] Fig. 4). Data 측면에서 synthetic long SFT share를 측정한 두 report는 서로 다르다. Long continued training을 40B token 수행하고 mix에 short data 37%를 넣은 ProLong base model은 short-only UltraChat SFT가 가장 좋았으며 SFT token의 1%가 synthetic long data일 때 1.6 point를 잃었다([[prolong]] Table 8). 반면 Llama 3.1은 short-only SFT가 long-context ability에 “significant regressions”를 일으켰고 synthetic long share 0.1%가 short/long benchmark를 최적화했다고 보고한다([[excerpts/llama3-staged-schedule]] §4.3.4). ProLong 저자들은 이 차이를 앞선 long training의 양과 short SFT set의 크기로 설명한다(Interpretation). 약 4K token test에서는 fact가 없는 synthetic retrieval data가 NIAH-style training data가 일으키는 knowledge-benchmark loss 없이 real retrieval로 transfer한다(Mistral-7B TriviaQA `+0.11` vs `−6.33`, [[excerpts/artificial-needles-kv-retrieval]] Table 2).
>
> **지침.** Long-context ability를 평가할 때는 RULER effective length와 함께 application suite(HELMET category) 하나 이상 및 short-context suite를 보고하라. 128K에서 frontier model 여섯 개 중 다섯 개의 NIAH는 100.0이며([[helmet]] Table 5), application performance 순위와 일치하지 않기 때문이다. Long continued-training을 할 때는 high-quality short data를 mix에 유지하라(ProLong의 5B-token ablation에서 가장 좋은 평균은 long 60% / short 40%). 100% long data는 perplexity를 개선했지만 SFT 뒤 downstream score는 낮췄으므로, SFT 뒤 측정한 long/short score로 mix를 고르라([[prolong]] §2.1, §3.2). Long continued-training stage가 짧거나 SFT set이 크면, SFT에 작은 synthetic long share를 넣어라(Llama 3.1: 0.1%; LongAlign: ShareGPT 76k에 long example 10k 추가). Llama 3.1과 SmolLM2에서 short-only SFT가 long context를 regression시켰으므로 ablation으로 검증하라([[excerpts/primary-quotes-2026-09]] smol-talk). 반대로 model이 이미 extensive long continued training을 거쳤으면 먼저 short-only SFT를 시험하라. ProLong에서는 1%, 3%, 10%, 50% synthetic long SFT token이 모두 0% arm보다 낮았기 때문이다([[prolong]] Table 8). Training에 synthetic retrieval data를 쓸 때는 factual content가 없는 data를 우선하고, real retrieval result와 닮은 distractor로 평가하라. Key-value data는 retrieved(relevant) distractor가 있는 MDQA를 개선하지 못했기 때문이다([[excerpts/artificial-needles-kv-retrieval]] §4).

## 학습한 버전에 대한 정정

1. “ProLong의 coherence filter는 20B token에서 512K context와 garbage를 가르는 차이다” → ProLong은 coherence filter를 설명하지 않는다. 40B token(64K에서 20B, 그 뒤 512K에서 20B)을 훈련하며, 핵심 data 결과는 long data를 high-quality short data와 섞어야 한다는 것이다([[prolong]], §3.2, Table 9).
2. “Fu-2024는 200M을 사용한다”(RoPE base, 또한 “10K → 200M NTK-aware”) → Fu et al.은 Xiong et al. (2023)처럼 RoPE base를 조정한다고만 말하며 값을 인쇄하지 않는다([[excerpts/primary-quotes-2026-09]], long-context-data-engineering, §1, §4).
3. 지침 “staged RoPE-base rescaling으로 position extension” 및 “마지막 128K model에서 RoPE base `θ`를 Llama-2의 10K에서 500K로 rescale” → Llama 3는 `θ = 500,000`을 pre-training 전체의 architecture hyperparameter로 설정한다. Long-context stage 여섯 개 사이에서 base를 바꾸었다는 설명은 없다([[excerpts/llama3-staged-schedule]], §3.2, §3.4.2).
4. 지침 “(3) synthesized long task로 SFT”를 default stage로 둔 것 → 앞선 stage에 따라 달라진다. ProLong의 final SFT는 모든 tested share에서 synthetic long data가 해로웠으므로 short-only UltraChat이고, Llama 3.1에는 synthetic long data 0.1%가 필요했다([[prolong]], §5, Table 8; [[excerpts/llama3-staged-schedule]], §4.3.4).
5. “Chapter 23–27이 synthetic-data design pattern을 세웠다” 및 “ch-27 — synthetic-data design pattern” → Design pattern은 ch-18이고, ch-27은 Agentic Trajectory Data다(outline.json).
6. “32K-shingled document에서 MinHash는 4K보다 8× 비싸다” → 이 수치를 report한 source가 없어 제거했다.
7. “Teacher-context ceiling은 hard constraint가 된다”(unverified `longmit` card 인용) → LongMIT은 truncated document segment에서 question을 생성한 뒤, 다른 document로 context를 fixed length까지 다시 채운다(App. C.2). 따라서 generator는 전체 training context를 읽지 않는다([[excerpts/longmit-mimg]], App. C.2).
8. NIAH needle “eat a sandwich at Dolores Park”, “exact-substring으로 score”, “blog post” → Default needle은 “eat a sandwich and sit in Dolores Park on a sunny day”다. Original harness는 GPT-4 judge의 1–10 rubric으로 score하며, artifact는 linked tweet thread가 있는 GitHub repository다([[needle-in-haystack-data]], L24, L360-390, Summary).
9. “모든 128K / 200K / 1M-context release가 이제 NIAH heatmap을 제공한다” → 뒷받침되지 않아 제거했다([[needle-in-haystack-data]], Verification).
10. BABILong “task template 20개” → Released evaluation set은 task 10개(QA1–QA10)를 다루며 Tables 2와 4는 QA1–QA5 평균을 낸다. 50M token은 ARMT에서만 도달한다([[babilong]], App. N.2, §3.3).
11. “Llama-3.1-70B의 RULER effective context는 약 64K”(unverified `long-context-llama3` card 인용) 및 “Meta가 paper에서 이를 인정한다” → Claimed 128K / effective 64K 결과는 RULER Table 3의 것이며 Llama 3 report는 RULER를 평가하지 않는다([[excerpts/ruler-task-family]], Table 3; [[excerpts/llama3-staged-schedule]], §5.2.6).
12. “Llama-3.1-405B NIAH@128K는 약 99%, RULER effective context는 약 96K” → 405B는 RULER Table 3에 없다. Llama 3 report는 405B의 NIAH 100%, Multi-needle 98.1을 제시한다([[excerpts/llama3-staged-schedule]], §5.2.6, Table 21).
13. “Qwen-2.5-14B-1M NIAH @ 1M은 약 100%, RULER @ 1M은 약 85%” → RULER는 128K까지만 보고한다(14B-Instruct-1M 92.2). 1M에서 Passkey Retrieval은 14B-Instruct-1M에서 perfect다([[qwen-long-context-synth]], Table 4, Fig. 1).
14. LongAlpaca “long document 3,000개 — ArXiv 40%, book 30%, GitHub 30%”, “ChatGPT 또는 Claude”, “document당 QA pair 3개”, length/profanity filter, “약 $5K” → LongLoRA App. B.6은 technical paper, science fiction, 기타 book의 self-collected QA, long QA 9k + Alpaca short QA 3k, epoch 5를 말한다. Generator, document count, code, filter, cost는 명시하지 않는다([[excerpts/primary-quotes-2026-09]], longalpaca).
15. “Data quality는 long example 약 10k에서 saturate” 및 “LongAlign-10k는 larger LongAlpaca-12k보다 multi-segment integration에서 낫다” → LongAlign은 0k에서 5k, 10k까지 일관된 개선을 보고하며 saturation은 없다. LongAlign-10k는 LongBench-Chat(6.28 vs 4.58)과 MT-Bench에서 높지만 LongAlpaca-12k는 LongBench S-Doc QA와 M-Doc QA에서 높다([[excerpts/longalign-pipeline]], §4.2, Table 3).
16. LongChat “ShareGPT의 long tail(≥ 8K token, ≥ 4 turn)을 mine하여 long conversation 18K개를 만든다” → LongChat은 16K로 truncate한 Vicuna ShareGPT conversation을 reuse하고, 7B는 conversation 80k개, 13B는 18k개로 fine-tune한다([[excerpts/primary-quotes-2026-09]], longchat, Step 2).
17. “LongMIT … synthesized multi-turn long-context dialog — 5–10 turn … LongBench-Chat을 5–10 point 올림” → LongMIT은 MIMG multi-agent framework(Chen et al. 2024)로 만든 multi-hop long-context instruction data다. LongBench-Chat이 아니라 LongBench QA subset을 보고한다([[excerpts/longmit-mimg]], §2, Table 1).
18. ProLong “coherent content ≥ 64K token”, “Web: discard”, “code × 4, book × 2, academic × 2, forum × 1, web × 0.5”, “code 약 40%, book 25%, academic 15% …” → Final mix는 code repository 30%, book 30%, textbook 3%, ShortMix 37%다. CommonCrawl은 long source로 test했다(Table 4)([[prolong]], Table 4, Table 9).
19. “Curated long document를 concatenated short document로 바꾸면 HELMET에서 10+ point를 잃는다” → Paper에 이런 ablation은 없다([[prolong]], Verification).
20. ProLong stage 1 “RoPE base rescaled 500K → 128M”, “LR 1e-4 → 1e-5”, “100% long coherent document … cross-document packing 없음” → RoPE base는 64K에서 `8×10^6`, 512K에서 `1.28×10^8`이다. LR은 stage마다 cosine decay로 `1e-5`에서 `1e-6`까지다. Short data는 pack하고 document 사이 attention은 mask한다([[prolong]], Table 9, App. B.2).
21. ProLong “Stage 2 — SFT(5B token): long-instruction 70%(Claude-3-generated) + UltraChat 30% + multi-needle NIAH” → SFT는 UltraChat만 1B token이다. Llama-3-8B-Instruct와 Llama-3-70B-Instruct의 synthetic long data를 test했지만 사용하지 않았다([[prolong]], §5, Table 9).
22. “약 200K H100-hour”, “(Llama-3-8B base)”, “128K에서 InfiniteBench로 Llama-3.1-8B-Instruct와 Qwen2-7B-Instruct를 이김” → 2.2K + 12.2K H100 hour를 썼다. Llama-3-8B-Instruct에서 initialize했다. HELMET 128K average는 Llama-3.1-8B-Instruct 46.5 대비 49.4, ∞Bench는 46.7 대비 48.0, RULER는 81.3 대비 71.9다([[prolong]], Tables 9, 10, 26).
23. Llama 3의 stage 네 개 token 약 100B, 약 100B, 약 150B, 약 200B와 short:long ratio `80:20 → 40:60` 표 → Report는 stage 여섯 개, 약 800B token을 인쇄하며 stage별 length/token/ratio는 없다([[llama-3-recipe]], v3 §3.4.2).
24. “Long-context SFT는 total SFT sample의 약 0.1%(약 100M 중 약 100K)로 유지한다. Long-SFT fraction을 1%보다 높이면 MMLU 약 1 point 비용” → Report는 0.1%만 말한다(Table 7: example 0.11%, average token 38,135.6). Total example count와 MMLU cost는 보고하지 않는다([[llama-3-recipe]], §4.3.4, Table 7).
25. “Teacher는 Llama 3 405B 자신(self-distillation)” → QA는 “an earlier version of the Llama 3 model”, chunk summary는 “our strongest Llama 3 8K context model”이 생성한다([[excerpts/llama3-staged-schedule]], §4.3.4).
26. Fu 2024 “32K보다 긴 document에 5× weight” 및 “Books를 globally up-weight하면 short-context MMLU가 3–5 point 하락” → 두 수치 모두 paper에 없다. Domain upsampling의 per-domain loss effect(Table 5)를 보고하며 document boundary와 무관하게 80K chunk로 pack한다([[excerpts/primary-quotes-2026-09]], long-context-data-engineering, §3, §4).
27. LongRoPE “`θ_i = θ^(2i/d)`” 및 “RoPE dimension `d = 128`이면 128-dimensional search space” → `θ_i = θ^(−2i/d)`다. Search에는 frequency마다 factor 하나(`d/2` value)와 interpolation 없이 유지할 initial token 수 `n̂`가 있다([[excerpts/primary-quotes-2026-09]], longrope-data, §2.1, §3.2).
28. LongRoPE “Fitness는 long-context perplexity + NIAH retrieval accuracy의 weighted combination” → Search는 PG19 validation sample 5개에서 perplexity를 minimize한다. `P = 64`, `T = 40`, `p = 0.3`은 맞다([[excerpts/primary-quotes-2026-09]], longrope-data, Algorithm 1, §4.1).
29. LongRoPE “token < 1B … (256K에서 약 300M, 2M에서 약 600M)”, “fine-tune data 10× 감소”, “MMLU/GSM8K는 1 point 이내” → Fine-tuning은 global batch 32로 128k segment에서 400 step, 256k에서 600 step을 더 하며 token count는 인쇄하지 않는다. 2048k는 두 번째 search로 도달한다. 10× 수치는 paper에 없다([[excerpts/primary-quotes-2026-09]], longrope-data, §4.1).
30. PoSE “4K-token sample로 train”, “compute 4× reduction”, “retrieval-strong / reasoning-weak” → PoSE는 2,048-token training window를 쓴다. Compute와 reasoning 주장은 paper에 없다([[pose-synthesis]], §4.1, Verification).
31. Qwen2.5-1M “32K → 128K → 256K(50B + 50B + 100B token)” → Stage는 `4,096 → 32,768 → 65,536 → 131,072 → 262,144`이고 RoPE base는 최대 10,000,000이다. Token count는 보고하지 않는다([[qwen-long-context-synth]], §3).
32. Qwen2.5-1M “Qwen-Max가 생성한 synthetic SFT mix”와 multi-needle, 50K–200K summarization, 5–20-passage RAG-QA, FILL-IN, multi-position filter → Query는 random segment에서 Qwen2.5가 작성하고 response는 Qwen-Agent framework가 작성한다. Fill-in-the-middle은 pre-training task다. 열거한 SFT task와 filter는 report에 없다([[qwen-long-context-synth]], §3, §4).
33. “DCA … inter-chunk는 low-rank formulation 사용”, “256K-size chunk” → DCA는 relative position을 intra-chunk, inter-chunk, successive-chunk position index로 remap하여 pre-training length보다 큰 distance가 없게 한다. Training-free다([[qwen-long-context-synth]], §5.1).
34. “Short-context MMLU / GSM8K가 base Qwen-2.5의 1 point 이내”, “Qwen 3가 이 recipe를 상속하고 refine” → Table 6은 섞인 변화를 보인다(14B GPQA `45.5 → 39.9`, 7B MMLU-Pro `56.3 → 54.3`, 14B GSM8K `94.8 → 94.8`). Report는 Qwen 3를 언급하지 않는다([[qwen-long-context-synth]], Table 6, Verification).
35. “ch-29(next) … long-context synthesis는 optional modality specialization 하나”, “DCA … inference/serving chapter에서 별도 연구할 가치” → ch-29 concept에는 long-context component가 없고 outline에는 inference/serving chapter가 없다. Long-document synthesis는 ch-29a, context extension은 ch-32b다(outline.json).

## 이 장이 범용 모델에 중요한 이유

범용 모델의 pipeline은 pre-training → mid-training → SFT → preference optimization → RL → evaluation이다. Long-context ability는 주로 mid-training(ch-32b)에서 추가하고, SFT에서 보존하거나 끌어내며, evaluation(ch-32c)에서 gate한다. Long-range dependency가 있는 natural data와 natural long evaluation task는 모두 부족하므로 lab은 이를 합성한다. 이 장은 evaluation에 쓰는 synthetic task family, synthetic long SFT data의 초기/production recipe, 그리고 synthetic long data를 얼마나 섞을지에 대한 증거를 다룬다.

측정 가능한 문제 세 가지가 이 장을 이끈다. 첫째, generator 하나로 만든 benchmark는 saturation될 수 있지만 application ability는 달라진다. 128K에서 frontier model 여섯 개 중 다섯 개의 NIAH는 100.0이지만, 같은 표에서 그 다섯 model의 HELMET average는 47.0(Llama-3.1-8B-Inst)부터 63.8(GPT-4o-08)까지다([[helmet]] Table 5). 둘째, long SFT data는 prior stage에 따라 short-context ability와 long-context ability를 서로 다른 방향으로 바꾼다(§4, §5). 셋째, synthetic long task 학습은 data가 담은 내용에 따라 transfer할 수도, transfer하지 못할 수도, 다른 ability를 없앨 수도 있다(§6). 이 모두는 generality question이다. Long-context gain은 generator가 만들지 않은 task에서도 유지되고 short-context ability를 없애지 않을 때만 general model에 유용하다.

## §1 Synthetic evaluation task family: NIAH, RULER, BABILong

**정의.** *Synthetic long-context task*는 program으로 task-relevant text(*needle* 또는 *fact*)를 긴 filler text(*haystack*) 안에 넣어 annotation 없이 answer를 알 수 있게 만든다. *Task family*는 needle type, distractor 수, hop 수, input length 같은 parameter를 지닌 generator로서 많은 instance를 만든다.

**Needle-in-a-Haystack(NIAH).** Kamradt harness는 Paul Graham essay에 sentence 하나를 선택한 depth에 삽입하고, 선택한 length로 context를 trim하며, needle이 답하는 question을 묻고, GPT-4 judge의 1–10 rubric으로 reply를 score한다([[needle-in-haystack-data]] L360-390). Released GPT-4-128K run에서는 `(length, depth)` cell 225개 중 207개가 10점을 받았다([[needle-in-haystack-data]] Findings). 이 test가 다루는 문제는 model이 주어진 position에서 fact 하나를 retrieve할 수 있는가다. Similar distractor를 무시할 수 있는가, item 여러 개를 반환할 수 있는가, fact를 결합할 수 있는가는 다루지 않는다.

**RULER.** RULER는 NIAH를 category 네 개의 task configuration 13개로 확장한다([[excerpts/ruler-task-family]] §3, Table 5).

1. Retrieval: single needle(S-NIAH), distractor needle이 있는 multi-key(MK-NIAH), multi-value(MV-NIAH), multi-query(MQ-NIAH).
2. Multi-hop tracing: variable tracking(VT), `X2 = X1`, `X3 = X2`, value 하나에 bound된 모든 name을 반환.
3. Aggregation: common-word/frequent-word extraction(CWE, FWE).
4. QA: distractor paragraph가 있는 SQuAD와 HotpotQA.

각 model은 model chat template 안에서 answer prefix와 recall-based accuracy를 사용해 4K, 8K, 16K, 32K, 64K, 128K에서 task당 generated example 500개로 평가한다([[excerpts/ruler-task-family]] §4).

**Effective length.** RULER는 effective length를 13-task average가 4K에서의 Llama2-7B score를 넘는 tested length 중 가장 큰 것으로 정의한다.

```
L_eff(model) = max { L ∈ {4K, 8K, 16K, 32K, 64K, 128K} : S(model, L) > 85.6 }
```

- `S(model, L)`은 input length `L`에서 13 task의 mean accuracy다.
- `85.6`은 4K에서의 Llama2-7B mean이다([[excerpts/ruler-task-family]] Table 3).

**계산 예시.** Llama 3.1 70B는 4K부터 128K까지 96.5, 95.8, 95.4, 94.8, 88.4, 66.6을 기록한다. Score는 64K의 88.4까지 85.6을 넘고 128K의 66.6에서 그 아래로 떨어지므로 claimed 128K의 `L_eff = 64K`다. Llama 3.1 8B는 32K에서 87.4, 64K에서 84.7이므로 `L_eff = 32K`다. GradientAI의 Llama-3 70B(claimed 1M)는 16K에서 90.8, 32K에서 85.4이므로 `L_eff = 16K`다([[excerpts/ruler-task-family]] Table 3). 32K에서 0.2 point 차이(85.4 vs 85.6)는 reported length를 두 배로 바꾼다. 따라서 `L_eff`는 step size가 큰 threshold statistic이며 full score row가 더 많은 정보를 담는다. Figure [figures/context-extension.html](figures/context-extension.html)은 이 Table 3 row를 threshold와 함께 그려 reader가 model별 effective length를 읽고 score가 85.6에 얼마나 가까운지 보게 한다.

**RULER failure analysis.** Yi-34B-200K에서 full-haystack setting에 distractor needle을 더하면 256K에서 accuracy가 약 40 point 낮아졌다. Query 수를 1에서 8로 늘리면 약 15 point 낮아졌고, 128K에서 CWE output의 80% 초과는 one-shot example을 복사했다([[excerpts/ruler-task-family]] §5). 이는 single-needle test가 드러내지 못하는 서로 다른 failure type 세 가지다.

**BABILong.** BABILong은 bAbI reasoning task(agent가 location 사이를 이동)의 fact를 PG19 book sentence 사이에 숨긴다. Task는 고정하고 length만 키운다([[babilong]] §2, App. N.3). Test한 LLM은 “effectively use only 10-20% of the context”한다. Background text가 있으면 test한 모든 LLM이 QA2(two supporting fact)에서 85% 미만이었지만, 여럿은 QA1(one fact)을 16K–64K까지 통과했다([[babilong]] Abstract, §3.1). RAG pipeline은 모든 length에서 QA1 약 60%에 도달했지만 QA2/QA3에서는 random보다 낮아졌다([[babilong]] §3.2).

**측정 한계.**

- *Application과의 correlation.* HELMET은 128K에서 instruction-tuned model 35개에 걸쳐 synthetic task와 application category의 Spearman `ρ`를 계산한다. NIAH vs ∞Bench QA는 0.63, RULER MK vs ∞Bench QA는 0.81, HotpotQA RAG vs ∞Bench QA는 0.88이다. 어떤 synthetic task도 downstream category에 대해 0.8을 넘는 평균은 없다([[helmet]] Figs. 3–4). 결과는 single study다.
- *Lexical overlap.* Question이 needle과 word를 거의 공유하지 않으면 GPT-4o는 32K에서 99.3에서 69.7로 낮아진다([[nolima]] Table 3). 저자들은 standard needle을 word matching으로 locate할 수 있다는 뜻으로 해석한다(Interpretation).
- *구별되는 fact.* bAbI sentence는 짧고 PG19 book prose(1919년 전 출판된 book)와 style이 다르며, 저자들은 fine-tuned model이 fact와 background를 분리하는 token을 학습할 수 있다고 말한다([[babilong]] Limitations).
- *Position.* Document 20개에서 GPT-3.5-Turbo는 answer document가 first일 때 75.8%, position 10일 때 53.8%, position 20일 때 63.2%를 기록한다([[lost-in-the-middle]] App. G Table 6). Depth 평균은 이 curve를 가린다.

**범용 모델에 대한 의미.** Synthetic family는 skill 하나를 controlled하게 probe한다. Diagnosis(어느 knob가 model을 망가뜨리는가)와 training-data design에 유용하지만 application breadth의 약한 proxy다. 상세 evaluation gate는 ch-32c에 있다. Figure [figures/task-family-generator.html](figures/task-family-generator.html)은 이 chapter에서 쓴 family 각각의 작은 instance를 생성한다. Reader는 generator knob(distractor, value, hop, depth)를 바꾸고, 각 instance 옆에서 family가 측정하는 것과 report된 limit을 볼 수 있다.

## §2 초기 long instruction data: LongAlpaca와 LongAlign

**정의.** *Long-context instruction data*는 prompt가 긴 document(LongAlign에서는 8K–64K token, LongAlpaca는 length를 보고하지 않음)를 포함하고 response가 그 document에 의존하는 SFT data다.

**문제.** Continued training으로 context를 extension한 뒤 model은 긴 input을 받아들일 수 있지만, 그 input 전반에서 instruction을 따르지 못할 수 있다. LongAlign은 이를 직접 측정한다. ShareGPT만으로 fine-tune한 ChatGLM3-6B-64k(LongAlign-0k)는 LongBench-Chat에서 3.73이었고, long example 10k개를 추가하면 6.28로 올랐다([[excerpts/longalign-pipeline]] Table 3).

**LongAlpaca (2023).** LongLoRA의 SFT set은 “technical papers, science fiction, and other books”의 long-context QA pair 9k개와 Alpaca에서 sample한 short QA pair 3k개를 포함하며, epoch 5로 훈련한다([[excerpts/primary-quotes-2026-09]] longalpaca). Paper는 generator를 이름으로 밝히지 않고 filter도 제공하지 않는다. RULER Table 3에서 LongAlpaca-13B(claimed 32K)의 effective length는 4K 미만이다([[excerpts/ruler-task-family]] Table 3).

**LongAlign (2024), 단계별 과정** ([[excerpts/longalign-pipeline]] §3.2, App. A):

1. Base model을 extension한다. RoPE base를 `10,000 → 2,000,000`으로 바꾸고 64K 미만 sequence에서 continued training을 10B token 수행한다.
2. Source 9개에서 64K token 미만 document를 sample하고 더 긴 document는 upsample한다.
3. Document마다 task prompt 네 개(general, summary, multi-hop reasoning, information extraction) 중 하나를 random으로 고르고, Claude 2.1에 text의 “cover all parts”를 하는 question 다섯 개를 요청한다.
4. Question 하나를 random으로 고르고 Claude 2.1에 answer를 요청한다.
5. Sample을 손으로 검사하여 quality를 확인한다. Answer 100개 중 correct 94개, incorrect 2개, incomplete 3개, irrelevant 1개다.
6. Long example 10k개를 ShareGPT example 76k개와 섞고 epoch 2로 훈련한다.

**Mixing 결과(ChatGLM3-6B-64k, Table 3).** LongBench-Chat은 long example 0k / 5k / 10k에서 3.73 / 5.99 / 6.28이다. Short task는 MMLU에서 1.1 point 이내(45.5 / 46.6 / 45.5), MT-Bench에서 0.17 이내(5.34 / 5.50 / 5.51)로 유지한다. LongAlpaca-12k는 LongBench-Chat 4.58, MT-Bench 4.93이지만 S-Doc QA(65.8)와 M-Doc QA(45.6)는 가장 높다. 결과는 이 표에 model 하나를 쓴 single study다. 저자들은 LongAlpaca의 LongBench advantage를 book-based source와 닮은 2WikiMQA/NarrativeQA 때문이라고 설명한다(Interpretation).

**Packing 아래의 loss weighting.** Long SFT data는 shape가 치우쳐 있다. LongAlign-10k target은 average 200 token이고 target/sequence token ratio는 0.015다. ShareGPT는 330 token, ratio는 19.3으로 인쇄되어 있다(unit을 인쇄하지 않으며 percentage 0.193일 때만 값이 일관됨)([[excerpts/longalign-pipeline]] App. A). Sequence를 pack하고 pack마다 loss를 token mean으로 내면, sequence가 동일한 weight를 받지 않는다. 의도한 loss는 sequence마다 weight `1/M`을 준다.

```
L = (1/M) Σ_i L_i / N_i                                  (per-sequence mean)
L' = (1/K) Σ_k ( Σ_{i∈k} L_i ) / ( Σ_{i∈k} N_i )          (per-pack token mean)
```

- `M`은 batch의 sequence 수, `K`는 pack 수다.
- `L_i`는 sequence `i`의 summed token loss, `N_i`는 그 target token 수다.
- `L'`에서 pack `k`의 sequence `i`는 mean loss `L_i/N_i`에 `N_i / (K · Σ_{j∈k} N_j)`의 weight를 받는다.

LongAlign은 sequence `i`의 각 target token을 `K / (N_i · M)`로 scale하고 pack에 걸쳐 sum하여 `L`을 복원한다([[excerpts/longalign-pipeline]] Eq. 4, App. B).

**계산 예시.** `K = 2` pack, `M = 4` sequence를 두자. Pack A에는 target token 200개인 sequence 하나가 있다. Pack B에는 target token 10, 20, 30개인 sequence 세 개가 있으며 합계는 60이다. `L'` 아래에서 pack-A sequence의 weight는 `200 / (2 · 200) = 0.50`이고, pack-B sequence의 weight는 `10/120 = 0.083`, `20/120 = 0.167`, `30/120 = 0.25`다. 의도한 weight는 각각 `1/4 = 0.25`다. 긴 single sequence는 의도한 weight의 두 배를 받고 10-token sequence는 1/3을 받는다. 이 fix를 적용하면 ChatGLM3-6B-64k LongBench-Chat은 5.76(packing)에서 6.21로, Llama-2-7B-64k는 5.89에서 6.10으로 오른다([[excerpts/longalign-pipeline]] Table 4). Packing과 sorted batching은 ChatGLM3-6B-64k training time을 45.4 h에서 각각 20.5 h와 19.1 h로 줄인다(Fig. 5). Packing mask는 ch-04에서 다룬다.

## §3 Multi-hop long instruction synthesis: LongMIT와 MIMG framework

**정의.** *Multi-hop* question은 context의 서로 다른 부분에 있는 evidence 두 개 이상을 요구한다. LongMIT은 MIMG(Multi-agent Interactive Multi-hop Generation)로 만든 multi-hop long-context instruction dataset이다([[excerpts/longmit-mimg]]).

**측정한 문제.** Qwen2-72B로 Self-Instruct-style generation을 하면 저자의 manual annotation에서 multi-hop sample은 35% 미만이고 poor-quality sample은 40% 초과였다([[excerpts/longmit-mimg]] Abstract).

**방법** ([[excerpts/longmit-mimg]] §2, App. C):

1. Single-hop question agent가 document segment에서 question을 생성한 뒤 answer한다(question-then-answer order).
2. Sampling step은 BGE embedding으로 question-similarity matrix를 만들고, 한 document에서 related question을 선택한다(더 높은 quality). 또는 여러 document에서 선택한다(더 높은 diversity).
3. Merging agent가 선택한 single-hop question을 rationale 없이 multi-hop question 하나로 결합한다.
4. Quality verification agent(InternLM2-20B, rationale 포함)가 여러 criterion으로 sample 각각을 score하며 quality score가 8.5를 초과하는 sample을 남긴다.
5. 다른 document로 context를 fixed length까지 다시 채운다.

**증거.** Human-labelled sample에서 Qwen-72B-Instruct를 backbone으로 사용했을 때 MIMG는 high-quality 94.8, diversity 88.2, multi-hop 94.8에 도달했다. Self-Instruct는 각각 61.3 / 53.4 / 33.1이었다([[excerpts/longmit-mimg]] Table 7). LongMIT으로 SFT하면 LongBench QA average(GPT-4o score)는 LLaMA3-8B에서 64.29이고, LongAlign은 52.92, LongAlpaca는 52.02였다([[excerpts/longmit-mimg]] Table 1). Verification finding은 다음과 같다. Scoring은 binary classification보다 낫다. LLM verifier는 long input에서 human과 낮은 kappa로 잘 합의하지 못하지만 high precision으로 select한다. Rationale을 제거하면 domain 전반에서 verifier precision이 8.6% 초과로 낮아진다(§3.1).

**조건과 한계.** LongMIT은 sample 64.40k, token 5.07B이고 LongAlign은 9.89k, 0.17B다(Table 3). 따라서 Table 1 비교는 token을 맞춘 비교가 아니다. Evaluation은 LongBench QA subset을 다루며 summarization과 code는 Table 1에 없다. Llama-3-8B-ProLong base의 short-context check에서 IFEval instruction-level loose accuracy는 0.1259에서 0.1583으로 올랐고, ArenaHard는 overlapping confidence interval과 함께 7.2에서 6.7로 움직였다(Tables 6, 8). 결과는 single study다.

**범용 모델에 대한 의미.** 여기서 측정할 수 있는 lever는 multi-hop rate와 verifier precision이다. 둘 다 source document가 아닌 generator와 filter의 property다. 범용 model에서는 같은 verifier logic이 모든 long SFT slice에 적용된다. Ch-29a는 long-document generator를 더 깊게 다룬다.

## §4 Production long-SFT generator: Llama 3.1과 Qwen2.5-1M

**Llama 3.1 long-context stage.** Llama 3 405B pre-training에서 context는 약 800B token에 걸친 stage 여섯 개에서 8K에서 128K로 늘어난다. Report는 8B와 70B의 schedule은 제공하지 않는다. Stage는 “model performance on short-context evaluations has recovered completely”하고 model이 해당 length까지 “perfectly solves 'needle in a haystack' tasks”할 때 끝난다([[excerpts/llama3-staged-schedule]] §3.4.2). 이 gate는 short-context criterion과 synthetic retrieval criterion을 결합한다. Retrieval criterion 하나만으로 application ability를 확립하지 못하는 이유는 §1에서 설명했다.

**Llama 3.1 long SFT generator** ([[excerpts/llama3-staged-schedule]] §4.3.4):

1. *QA:* Curated long document를 8K chunk로 나누고, random으로 고른 chunk에서 QA pair를 생성하며, whole document를 context로 하여 훈련한다.
2. *Summarization:* 8K chunk를 summarize한 뒤 summary를 다시 summarize한다. Model은 full document를 summarize하도록 훈련하고, global understanding이 필요한 QA를 summary에서 생성한다.
3. *Code reasoning:* Python import를 parse하고, 다른 file 다섯 개 이상이 참조하는 file을 선택해 하나를 제거하며, model에 어떤 file이 그것에 depend했는지와 missing code가 무엇인지 묻는다.
4. Sample을 16K, 32K, 64K, 128K bucket에 넣는다.

Generator는 8K chunk를 읽지만 training example은 whole document를 포함하므로 generator에 long context window가 필요하지 않다. Answer는 chunk 하나(QA) 또는 document의 여러 부분(summary QA, code)에 의존한다.

**Mixing 증거.** Short-only SFT는 “resulted in significant regressions in long-context capabilities”했으며, synthetic long share 0.1%는 “optimizes the performance across both short-context and long-context benchmarks”한다. SFT model이 long context에서 강할 때 short-only DPO는 long context를 해치지 않았다([[excerpts/llama3-staged-schedule]] §4.3.4). Ablation table은 인쇄하지 않았다. 결과는 수치 없는 single study다.

**계산 예: example share와 token share.** Table 7은 long-context data를 SFT example의 0.11%로 제시하며 average token은 38,135.6이다. 모든 SFT example에 걸친 average token은 846.1이다([[excerpts/llama3-staged-schedule]] Table 7). Long-context token share는 `0.0011 × 38,135.6 / 846.1 ≈ 0.050`, 즉 SFT token의 약 5%다(derived; report는 token share를 인쇄하지 않음). Example share를 token에 그대로 적용한 “long 0.1%” recipe는 Llama 3.1이 쓴 것보다 long token을 약 50배 적게 담게 된다.

**Qwen2.5-1M.** Pre-training은 stage 다섯 개에서 context를 extension한다(`4,096 → 32,768 → 65,536 → 131,072 → 262,144` token, RoPE base 최대 10,000,000). Stage 3–5에서는 sequence의 75%가 해당 stage의 maximum length이고 25%는 더 짧다([[qwen-long-context-synth]] §3). Natural long text는 “often exhibits weak long-distance associations”하므로 synthetic task 세 가지를 더한다. Fill-in-the-middle, keyword/position-based paragraph retrieval, paragraph reordering이다(§3). SFT에서 Qwen2.5는 random으로 추출한 segment에서 query를 작성하고, Qwen-Agent framework는 retrieval, chunk-by-chunk reading, step-by-step reasoning으로 full document에 대한 answer를 작성한다. SFT는 먼저 32,768 token까지 short data에서 돌리고, 그 뒤 262,144 token까지 short/long mix에서 돌린다(§4). Offline RL은 최대 8,192 token pair만 쓰지만 LongBench-Chat을 올린다(7B `7.32 → 8.08`, 14B `8.56 → 8.76`)(Table 3). Report는 이를 short-to-long transfer로 해석한다(Interpretation).

**조건과 한계.** 두 report 모두 long SFT share의 example count나 ablation table을 공개하지 않는다. 128K model에 대조한 Qwen2.5-1M의 short-context result는 섞여 있다(14B GPQA `45.5 → 39.9`, IFEval `81.0 → 84.3`; [[qwen-long-context-synth]] Table 6).

## §5 ProLong이 제시한 continued-training data와 SFT share

**설정.** ProLong은 long context용 Llama-3 8B continued training을 연구한다. 64K에서 token 5B씩 ablation하고, 각각 UltraChat SFT 뒤 HELMET subset과 short task 다섯 개에서 평가한다([[prolong]] §2–§3).

**수치를 포함한 결과.**

1. *Perplexity는 mix를 선택하지 못한다.* Long share가 100%까지 올라갈수록 PG19 perplexity는 계속 개선되지만, downstream long-context average는 떨어진다([[prolong]] §2.1, Fig. 1).
2. *SFT 뒤 평가한다.* SFT 전에는 recall/RAG가 높은 long share를 선호한다. SFT 뒤에는 long data가 많을수록 이들이 하락한다. Long share가 커질수록 short-task score는 단조롭게 하락한다. 가장 좋은 average는 long 60% / short 40%다([[prolong]] §3.2, Fig. 3). 저자들은 long-only model이 generic SFT에 나쁜 initialization이라고 가설을 세운다(Interpretation).
3. *Long source.* Long data 60%에서 book/repository 1:1 average는 54.6, book은 53.8, code repository는 52.3(highest recall, 99.2), CommonCrawl은 50.9다([[prolong]] Table 4). Repository file은 dependency ordering 없이 concatenate하며, 저자들은 이것이 “increase the distance between dependent files and reduce recency bias”할 것으로 기대한다(§3.1).
4. *Short source.* ShortMix는 long 54.6 / short 65.5이고, FineWeb-Edu는 53.0 / 63.0이다([[prolong]] Table 6).
5. *Document masking.* Document 사이 attention을 mask하면 54.6 / 65.5이고 mask가 없으면 53.6 / 64.9다([[prolong]] App. B.2, Table 20).
6. *평가 length보다 길게 훈련한다.* 64K checkpoint 20B token에서 시작하면, 둘 다 64K에서 평가했을 때 512K에서 token 4B를 더 훈련한 것이 64K에서 token 4B를 더 훈련한 것보다 낫다(recall 98.5 vs 95.0, re-rank 32.9 vs 28.0)([[prolong]] Table 7).

**평가 length보다 긴 sequence로 훈련하는 것이 도움이 되는 이유를 저자들이 model로 설명한 방식(가설).** Task가 정확히 `d` token에 걸친 dependency example을 필요로 하고, dependency가 모든 position에서 같은 확률로 생긴다고 하자. Length `L` sequence에는 length `d` span이 `L − d + 1`개 있다. Length `n·d` document 하나에는 그런 span이 `n·d − d + 1`개이고, length `d` document `n`개에는 `n`개 있다. 차이는 `(n − 1)(d − 1)`이다([[prolong]] §4). 저자들은 이 assumption이 실제에서는 성립하지 않는다고 말하며 Table 7 결과의 intuition으로 이 model을 제시한다. 계산 예: `d = 4`, `n = 3`이다. 12-token document 하나에는 `12 − 4 + 1 = 9` span이 있고, 4-token document 세 개에는 3개가 있다. 차이는 `6 = (3 − 1)(4 − 1)`이다. 같은 count는 document masking으로 unrelated short document를 pack해도 long span이 추가되지 않는 이유를 설명한다(Interpretation).

**Synthetic long SFT data.** Llama-3-8B-Instruct가 생성한 synthetic long instruction data는 random chunk QA 40%, chunk list에 대한 RAG 30%, recursive book summary 30%이며 token share로 UltraChat에 섞는다. ProLong의 5B-token data ablation과 달리, 이 SFT run은 512K까지 token 40B로 훈련한 ProLong base model에서 시작한다([[prolong]] §5, Table 8).

| Synthetic share(token) | 0% | 1% | 3% | 10% | 50% |
|---|---|---|---|---|---|
| Average(32K/64K average의 RAG, re-rank, ICL; 512K의 JsonKV, QA, summarization) | 55.7 | 54.1 | 53.5 | 53.9 | 43.3 |

Llama-3-70B-Instruct generator는 1 / 3 / 10 / 50%에서 54.2 / 55.4 / 53.6 / 49.7을 냈고, 이 역시 55.7보다 낮다([[prolong]] App. B.5, Table 23). Figure [figures/task-family-generator.html](figures/task-family-generator.html)은 이 table도 bar로 그린다.

**ProLong과 Llama 3.1/LongAlign을 조화시키기.** 저자들은 가설 두 개를 제시한다. Earlier work는 long continued training이 너무 적어 synthetic SFT data가 추가 long training처럼 작동했을 수 있다. 그리고 Llama 3.1의 short instruction set은 훨씬 커서 extensive short SFT 동안 long task의 degeneration을 막으려면 long share가 필요할 수 있다([[prolong]] §5). 둘 다 Interpretation이다. 두 번째를 뒷받침하는 관찰은 다음과 같다. Short sample만으로 fine-tune한 SmolLM2는 2,048 token 너머의 long-context ability를 잃었고, SmolTalk은 16K token 미만의 LongAlign sample을 추가했다([[excerpts/primary-quotes-2026-09]] smol-talk). Long continued-training budget과 SFT set size를 둘 다 바꾸는 연구는 없다는 것이 open question이다.

**조건과 한계.** ProLong result는 Llama-3 model의 10B scale에서 나온다([[prolong]] Limitations). SFT 뒤 ProLong은 short task 다섯 개에서 average 69.4이고 Llama-3-8B-Instruct는 69.8이지만, MMLU에서는 64.6 vs 67.0, GSM8K에서는 58.9 vs 68.5로 낮다. 저자들은 이를 Llama-3-8B-Instruct의 closed instruction data 탓으로 돌린다([[prolong]] App. B.7, Table 25).

## §6 Synthetic retrieval training과 real task로의 transfer

**질문.** Synthetic retrieval generator로 fine-tuning하면 real long-context task를 개선하는가? 다른 ability에는 어떤 비용이 드는가?

**Artificial Needles.** Data는 integer dictionary의 list다. Model은 key의 value와 그 key를 담은 dictionary를 보고한다. 더 어려운 variant에서는 subkey가 다른 key와 부분적으로 overlap하는 tuple key를 찾는다([[excerpts/artificial-needles-kv-retrieval]] §2). Mistral-7B-Instruct-v0.1은 약 3,900 token sample 350개로 epoch 2를 훈련하며 loss는 answer token에만 계산한다. GPT-3.5 Turbo는 multi-subkey sample 150개로 epoch 3을 훈련한다(§3.1). Prompt의 answer template는 formatting token에 계산하는 loss를 낮게 유지한다(§2, Fig. 4).

**증거.**

- **Transfer:** Document 20개, 약 4K token의 MDQA에서 U-shaped position curve가 평평해진다. Abstract는 GPT-3.5 Turbo가 position 10에서 10.5%를 얻었다고 보고한다. 비슷한 token count로 MDQA 자체를 fine-tune하면 MDQA accuracy가 더 낮았다([[excerpts/artificial-needles-kv-retrieval]] §3.2, Fig. 5).
- **Retention과 비교(Mistral-7B, 비슷한 training token, Table 2):**

| Fine-tuning data | MMLU | TriviaQA | NQ-Open |
|---|---|---|---|
| none | 53.42 | 47.63 | 11.61 |
| synthetic key-value, template | 53.44 | 47.74 | 11.98 |
| MultidocQA | 53.19 | 45.20 | 8.69 |
| IN2 | 53.49 | 45.44 | 9.80 |
| Needle-in-a-haystack | 52.83 | 41.30 | 4.88 |

저자들은 baseline의 TriviaQA/NQ-Open 하락을 factual content에 훈련한 탓으로 돌린다. 이는 hallucination을 조장할 수 있다(Interpretation, Gekhman et al. 인용).

**관련 synthetic design.** IN2는 Mistral-7B-Instruct-v0.2를 4K–32K context의 random position에 evidence를 둔 synthesized QA example 1.4M개와 short QA/general instruction으로 훈련한다. VAL Probing min-max position gap은 56.2에서 13.9로 낮아지고 MMLU는 59.3에서 59.2로 움직인다([[in2-film]] Table 1, Fig. 5). Distractor context로 훈련하는 것도 중요하다. GPT-4o-mini가 작성한 context(relevant 1개 + irrelevant 9개)가 있는 human-written QA pair 1.6k개로 훈련한 LLaMA3.1-8B는 LongBench average 38.57에 도달한다. Original document에서 instruction을 합성한 경우는 25.39, UltraChat만 쓴 경우는 23.35다([[context-synthesis-short-to-long]] Table 3). Continued training에서는 short document의 각 chunk를 가장 similar한 retrieved chunk와 interleave하면 128K에서 HELMET+RULER average 62.58에 도달한다. Random concatenation은 52.85다([[nextlong]] Table 1).

**조건과 한계.** Artificial Needles는 약 4K token context와 24K run 하나를 test하며, retrieved(relevant) distractor가 있는 MDQA에서는 “Models finetuned on our dataset will not improve”한다고 한다([[excerpts/artificial-needles-kv-retrieval]] §4, Fig. 10). BABILong의 QA1 fine-tuning은 반대 방향의 transfer risk를 보여 준다. QA1로 fine-tune한 GPT-3.5는 QA2–QA5에서도 개선됐지만, full fine-tune한 Mistral-7B는 그 task에서 degrade했다([[babilong]] App. I, Fig. 9). 결과는 single study들이다.

**범용 모델에 대한 의미.** Synthetic retrieval data는 stand-alone recipe가 아니라 mixture의 input이다. 범용 model에서 그 가치는 realistic distractor가 있는 held-out real task와 knowledge benchmark에서 측정한다. Training data와 같은 generator가 만든 task의 gain은 transfer를 보이지 않는다. Artificial Needles gain은 retrieved distractor가 있는 MDQA로 확장되지 않았기 때문이다(그 paper의 §4).

## Negative sample과 negative feedback

**이 stage에서 negative가 나타나는 곳.** (1) *Negative marginal value:* Filter가 reject한 sample이다. 예를 들어 InternLM2-20B verifier가 8.5 이하로 score한 LongMIT sample([[excerpts/longmit-mimg]] App. C.4.2), 또는 manual check에서 incorrect/incomplete/irrelevant로 판명된 LongAlign answer(100개 중 6개, paper는 full set에서 이런 sample을 제거했는지 설명하지 않음)가 있다. (2) *Content로서의 negative:* Input에 distractor를 두고 correct answer에 ordinary cross-entropy로 훈련하는 방식이다. MK-NIAH distractor needle, gold key와 subkey를 공유하는 multi-subkey key([[excerpts/artificial-needles-kv-retrieval]] Fig. 2), example 하나당 irrelevant context 9개([[context-synthesis-short-to-long]]), similar retrieved chunk([[nextlong]])가 있다. (3) *Gradient로서의 negative:* 이 chapter의 source에서는 Qwen2.5-1M의 DPO와 “similar to DPO”인 offline RL만 rejected response를 사용하며, pair는 최대 8,192 token이다([[qwen-long-context-synth]] §4). 여기의 어떤 source도 negative를 conditioning으로 쓰지 않는다.

**현재 관행이 하는 일.** Long-context pipeline은 주로 failure를 버리고(verifier threshold), distractor를 content로 사용한다. 이 source에서는 long input에 negative gradient를 사용했다고 보고하지 않는다. Llama 3.1과 Qwen2.5-1M은 둘 다 preference data를 short로 유지한다([[excerpts/llama3-staged-schedule]] §4.3.4).

**Gradient case의 mechanism.** Probability `p`를 갖는 logit `z`의 softmax에서 token `y`의 log-probability gradient는 다음과 같다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

- `1[j = y]`는 chosen token이면 1, 아니면 0이고 `p_j`는 token `j`의 model probability다.

`log p_y`를 낮추는 step은 `z_y`를 `(1 − p_y)`만큼 내리고 다른 모든 `z_j`를 `p_j`만큼 올린다. 계산 예를 보자. `p = (0.7, 0.2, 0.1)`이고 `y = token 3`(unlikely한 rejected token)이다. Logit change는 `(+0.7, +0.2, −0.9)`에 비례하므로 제거된 mass 대부분은 이미 가장 likely한 token 1로 간다. 따라서 unlikely rejected continuation을 내리면 distribution은 현재 mode 쪽으로 sharpen된다. 전체 설명은 ch-43a, negative의 supervised use는 ch-31a에 있다.

**증거와 control.** Content로서의 distractor는 인용한 study에서 robustness를 개선했다. Distractor가 test 때 등장하면 1k-essay distractor로 훈련한 NIAH pilot model은 32K에서 90%를 넘는 반면, distractor 없이 훈련한 model은 degrade했다([[context-synthesis-short-to-long]] §3.3, Fig. 2). Similarity로 고른 hard negative는 continued-training data에서 random negative보다 좋았다([[nextlong]] §5.4, Fig. 7, 값은 인쇄하지 않음). Filter의 control은 verifier에 rationale을 넣고 criterion 여러 개로 score하는 것이다(§3). 이 chapter의 어떤 source도 negative가 차지하는 improvement share를 측정하지 않는다.

**Diagnostic.** Source domain별, length bucket별 verifier rejection rate를 log한다(classification에서는 더 긴 input에서 verifier precision이 낮아짐; [[excerpts/longmit-mimg]] Fig. 3c). Distractor가 있는 instance와 없는 instance의 accuracy를 분리해 보고한다. Long input에서 DPO-style stage를 쓴다면 chosen/rejected log-probability를 따로 log한다.

**Generality에 미치는 효과.** Content로서의 distractor data는 application도 공유하는 failure, 즉 비슷하지만 틀린 item을 retrieve하는 failure를 겨냥한다. Filter rejection은 남은 data의 task/domain distribution을 바꾼다. Filter가 long slice에서 domain 하나를 제거하지 않도록 domain별 survival rate를 기록하라.

## 레시피

| Model(정확한 release) | 크기 | Stage | 설정 | 값 | Source 위치 | 상태 | 이 값의 증거 |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | long-context | stage; token | 8K에서 128K까지 stage 여섯 개; token 약 800B; stage별 length/token/mix는 인쇄하지 않음 | arXiv:2407.21783v3 §3.4.2 ([[llama-3-recipe]]) | verified 2026-09-15 | Stage gate: short-context eval 완전 회복 및 그 length에서 NIAH 해결 |
| Llama 3.1 | 8B/70B/405B | pretrain-stable | RoPE base | 500,000(pre-training 전체) | §3.2 | verified 2026-09-15 | 32,768 이하 length에는 Xiong et al. (2023)을 인용; ablation 보고 없음 |
| Llama 3.1 | 범위 미지정 | SFT | long-context share | synthetic long data 0.1%(§4.3.4); example 0.11%, example당 token 38,135.6(Table 7) | §4.3.4, Table 7 | verified 2026-09-15 | “careful ablations”; 표 없음 |
| Llama 3.1 | 범위 미지정 | SFT | length bucket | 16K, 32K, 64K, 128K | §4.3.4 | verified 2026-09-15 | ablation 보고 없음 |
| Llama 3.1 | 범위 미지정 | preference | DPO data length | short-context만 | §4.3.4 | verified 2026-09-15 | SFT model이 long context에 강할 때 long-context regression 없음; 표 없음 |
| ProLong-64k-Base (Llama-3-8B-ProLong-64k-Base) | 8B | long-context | stage 1 | 64K에서 token 20B; RoPE base `8×10^6`; batch 4M token; AdamW LR `1e-5`, warmup 10%, cosine to `1e-6` | arXiv:2410.02660v4 Table 9 ([[prolong-recipe]]) | verified 2026-09-14 | App. B.1 Table 18: base `8×10^6` → 54.6 vs `4×10^6` → 48.7 |
| ProLong-512k-Base (Llama-3-8B-ProLong-512k-Base) | 8B | long-context | stage 2 | 512K에서 token 20B; RoPE base `1.28×10^8`; batch 8M token; LR `1e-5`(Table 9) vs `5e-6`(released `train_512K.sh`) | Table 9; github.com/princeton-nlp/ProLong@499fa29 | conflict | Table 7: 512K에서 +4B token > 64K에서 +4B, 64K에서 평가 |
| ProLong-64k-Base | 8B | long-context | data mix(share type 미명시) | code repo 30%, book 30%, textbook 3%, ShortMix 37%(FineWeb-Edu 27%, FineWeb 27%, Wikipedia 11%, StackExchange 11%, Tulu-v2 8%, OpenWebMath 8%, ArXiv 8%) | Table 9 | verified 2026-09-15 | Fig. 3(5B-token ablation): long 60%가 최고 average; Tables 4, 6 |
| ProLong-64k-Base, ProLong-512k-Base | 8B | long-context | cross-document attention | masked | Table 9; App. B.2 | verified 2026-09-15 | Table 20: 54.6 / 65.5 vs 53.6 / 64.9 |
| ProLong-512k-Instruct (Llama-3-8B-ProLong-512k-Instruct) | 8B | SFT | data; token; LR; batch | UltraChat만; token 1B; LR `2e-5`, warmup 5%, cosine to `2e-6`; 4M token | Table 9 | verified 2026-09-15 | Table 8: synthetic 0% 55.7 vs 1% 54.1 |
| ChatGLM3-6B-64k, Llama-2-7B/13B-64k (LongAlign) | 6B, 7B, 13B | long-context | RoPE base; token | `10,000 → 2,000,000`; 64K 미만 token 10B | EMNLP 2024 Findings §4.1 ([[excerpts/longalign-pipeline]]) | verified 2026-09-15 | ablation 보고 없음 |
| LongAlign-6B-64k | 6B | SFT | mix; epoch; packing | ShareGPT 76k + LongAlign 10k; epoch 2(약 1500-2000 step); pack당 sequence 약 12개; global batch 96; loss weighting | §4.1, App. B | verified 2026-09-15 | Table 3: 0k / 5k / 10k에서 LongBench-Chat 3.73 / 5.99 / 6.28; Table 4 |
| LongAlpaca (LongLoRA) | 7B-70B(범위 미지정) | SFT | data; epoch | long QA 9k + Alpaca short QA 3k; epoch 5 | arXiv:2309.12307 App. B.6 ([[excerpts/primary-quotes-2026-09]]) | verified 2026-09-15 | ablation 보고 없음 |
| LLaMA3-8B + LongMIT | 8B | SFT | data; optimizer; batch; epoch | sample 64.40k, token 5.07B; Adam LR `3×10^-5`, `β1 0.9`, `β2 0.95`; packing; 4M token; epoch 1; max length 4K-128K | arXiv:2409.01893v2 Table 3, App. E.1 | verified 2026-09-15 | Table 1: 64.29 vs 52.92(LongAlign) |
| LongMIT data | — | SFT(data generation) | generator; verifier; threshold | Qwen2-72B-Instruct; InternLM2-20B; quality score `> 8.5` | App. C.3, C.4.2 | verified 2026-09-15 | §3.1, Figs. 3–4(verifier strategy) |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | stage; RoPE base; length mix | `4,096 → 32,768`(base `10,000 → 1,000,000`); `65,536 / 131,072 / 262,144`에서 `1M / 5M / 10M`; max length 75%, shorter 25%; token 미보고 | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | Table 2: RULER-128K 37.6 / 56.0 / 83.8 / 87.6(14B) |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | SFT | stage | short `≤ 32,768`, 이후 short + long `≤ 262,144`; ratio 미보고 | §4 | verified 2026-09-14 | ablation 보고 없음 |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | preference | pair length | DPO와 유사한 offline RL, pair `≤ 8,192` token; `β` 미보고 | §4 | verified 2026-09-14 | Table 3: LongBench-Chat `+0.75 / +0.20` |
| Mistral-7B-Instruct-v0.1 (Artificial Needles) | 7B | SFT | data; epoch; loss | key-value sample 350개, dictionary 85개, 약 3,900 token; epoch 2; answer token만 | arXiv:2406.19292v2 §3.1 ([[excerpts/artificial-needles-kv-retrieval]]) | verified 2026-09-15 | Fig. 5; Table 2 |

**작은 범용 run의 시작점.** ProLong과 견줄 long continued training(40B token, Llama-3 8B)을 이미 한 8B model이라면, ProLong-8B가 LR `2e-5`와 UltraChat token 1B로 했듯 short data만으로 SFT를 시작하고 synthetic long-data 1% arm을 ablation으로 돌려라. Table 8이 그 setting에서 short-only를 선택했기 때문이다. Continued-training stage 자체의 verified starting mix는 64K에서 RoPE base `8×10^6`, document masking, long 63%(code repo 30%, book 30%, textbook 3%)와 ShortMix 37%인 ProLong-64k-Base다(Table 9). 이는 Llama-3의 8B에서 token 20B에 썼다. 이를 선택한 5B-token ablation은 long 60% / short 40%가 가장 좋았다(Fig. 3). Long continued-training stage가 훨씬 짧거나 SFT set이 크면 Llama 3.1이 보고한 long share 0.1%에서 시작하라(§4.3.4; Table 7은 example 0.11%, model size 범위 미지정). 선택 전에 그것이 뜻하는 token share를 계산하라(Table 7에서 약 5%, §4에서 derived). LongAlign의 long example 10k와 ShareGPT 76k 혼합은(`10/86 ≈ 12%` example, derived) context extension token 10B 뒤 ChatGLM3-6B-64k에 대한 verified point다.

## 일반화 관점

**(a) 폭을 넓히는 것.**

- Long continued training 중 short data: long share가 오를수록 short-task average는 단조 하락하고, SFT 뒤 long average는 60/40에서 가장 좋았다([[prolong]] Fig. 3).
- Diverse long source: book/code repository 1:1은 long average에서 각각 단독보다 낫다(54.6 vs 53.8과 52.3). Book은 ICL/summarization에, repository는 recall에 도움이 된다([[prolong]] Table 4).
- Long SFT에 여러 task type: LongAlign의 prompt type 네 개와 Llama 3.1의 QA/summarization/code generator는 다른 application category를 겨냥한다. 128K에서 HELMET category correlation은 0.34까지 낮으므로 task type 하나가 나머지를 포괄하지 않는다([[helmet]] Fig. 5).
- Fact-free synthetic retrieval data: TriviaQA/NQ-Open은 그대로 두지만 NIAH-style data는 이를 6.33과 6.73 낮춘다([[excerpts/artificial-needles-kv-retrieval]] Table 2).

**(b) Narrowing 또는 forgetting을 일으키는 것.**

- Long continued training 100%: perplexity는 개선하지만 SFT 뒤 downstream long score는 하락([[prolong]] §2.1, §3.2).
- Extensive long continued training 뒤 synthetic long SFT 과다: synthetic 50%는 55.7 대비 43.3([[prolong]] Table 8).
- ProLong 규모의 long continued training 없이 short-only SFT: Llama 3.1의 regression([[excerpts/llama3-staged-schedule]] §4.3.4)과 SmolLM2의 2,048 token 너머 loss([[excerpts/primary-quotes-2026-09]] smol-talk).
- Factual long QA/needle data training: Mistral-7B의 TriviaQA/NQ-Open loss 1.81–6.73 point([[excerpts/artificial-needles-kv-retrieval]] Table 2).
- Templated family 하나로 훈련: BABILong QA1로 full fine-tune한 Mistral-7B는 QA2–QA5에서 degrade([[babilong]] App. I).

**(c) 이 단계에서 측정하는 방법.**

- Long side: RULER full score row/effective length, application suite 하나(HELMET category), position-resolved accuracy([[lost-in-the-middle]]).
- Short side: 각 long stage 전후에 같은 short suite(ProLong은 short task 다섯 개 사용, Llama 3.1은 short-context recovery로 stage gate).
- SFT 전이 아니라 SFT 뒤([[prolong]] §2.2, Fig. 2).
- Held-out family: Training data를 만든 generator와 다른 family로 평가한다. 예를 들어 key-value retrieval로 train하고 retrieved distractor가 있는 MDQA에서 test한다([[excerpts/artificial-needles-kv-retrieval]] §4). ProLong은 development subset 밖의 held-out HELMET task, NoCha, RULER, ∞Bench를 보고한다([[prolong]] Limitations). RULER score는 Llama-3.1-8B-Instruct의 81.3보다 낮은 71.9이지만 HELMET average는 46.5보다 높은 49.4이므로 conclusion은 suite에 의존한다([[prolong]] Tables 10, 26).
- 알려진 measurement error: NIAH score는 judge/prompt template에 의존([[needle-in-haystack-data]] Guideline), effective length는 threshold statistic(§1 worked example), LongBench-Chat은 query 50개([[excerpts/longalign-pipeline]] §3.4).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| Perplexity로 continued-training mix 선택 | PG19 또는 held-out long perplexity는 개선, SFT 뒤 HELMET recall/RAG는 하락 | Fixed short SFT run 뒤 candidate mix 각각을 평가([[prolong]] Fig. 1, Fig. 3) |
| NIAH를 long-context gate로 취급 | 모든 depth에서 NIAH 100, 같은 length에서 RULER MK-NIAH/VT/CWE는 낮음 | Length별 RULER row와 MK-NIAH/VT task를 따로 보고 |
| Example share를 token share로 적용 | Example 0.1%를 token 0.1%로 적용하면 long SFT slice token이 의도보다 약 50배 작음 | Token share = example share × long mean length / overall mean length를 계산(§4 worked example) |
| Extensive long continued training 뒤 synthetic long SFT 추가 | SFT 뒤 long average가 short-only SFT보다 하락 | Synthetic 0%/1% arm 실행([[prolong]] Table 8) |
| Long continued training이 제한된 model에 short-only SFT | SFT 뒤 long score가 pre-SFT checkpoint보다 낮음 | Base/SFT checkpoint를 같은 long suite에서 평가([[excerpts/llama3-staged-schedule]] §4.3.4) |
| Sequence별 loss weighting 없는 packed long SFT | Training이 long single-sequence pack을 선호, long-chat score가 naive batching보다 낮음 | LongBench-Chat에서 loss weighting 유무 packing을 비교([[excerpts/longalign-pipeline]] Table 4) |
| Needle-style factual data로 훈련 | Long-context fine-tuning 뒤 TriviaQA/NQ-Open 하락 | Short regression suite에 knowledge QA 포함([[excerpts/artificial-needles-kv-retrieval]] Table 2) |
| Generator가 multi-hop label이 붙은 single-hop question 생성 | QA score는 높고 2WikiMQA/MuSiQue/HotpotQA gain은 낮음 | Training 전 multi-hop rate를 hand-label sample로 확인([[excerpts/longmit-mimg]] Abstract) |
| Needle depth 평균 | Mean accuracy는 괜찮고 middle position은 낮음 | Position별 accuracy와 best-minus-worst gap plot/report([[lost-in-the-middle]]) |
| Real retrieval과 다른 evaluation distractor | Random-distractor MDQA에는 gain, retrieved-distractor MDQA에는 없음 | Retrieved distractor로 test([[excerpts/artificial-needles-kv-retrieval]] Fig. 10) |

## 이해를 확인해 보세요

1. RULER는 Llama 3.1 8B의 effective length를 32K, GradientAI의 Llama-3 70B를 16K로 보고한다. Table 3 row를 사용하여 두 번째 model의 32K score 85.4가 왜 reported effective length를 두 배 차이 나게 하는지, 그리고 reader가 single number 대신 무엇을 보고해야 하는지 설명하라.
2. HELMET은 NIAH와 ∞Bench QA의 correlation 0.63, HotpotQA RAG의 0.88을 찾았다. RULER의 Yi-34B failure analysis와 NoLiMa lexical-overlap result를 사용하여 NIAH generator의 어떤 property가 더 약한 predictor를 만드는지 설명하라.
3. ProLong은 short-only SFT가 최선이었지만 Llama 3.1은 short-only SFT가 regression을 일으킨다고 보았다. 저자의 두 가설과 SmolTalk observation을 사용하여 각 pipeline의 어떤 property가 반대 결과를 낳을 수 있는지 설명하고, 가설을 구별하는 가장 작은 experiment를 설계하라.
4. LongAlign worked example에서 pack 하나에 든 200-token sequence는 0.25가 아니라 0.50 weight를 받는다. Target/sequence ratio LongAlign-10k 0.015와 ShareGPT 19.3(percentage로 읽음)을 써서 long-context SFT data가 short chat data보다 이 bias를 크게 만드는 이유를 설명하라.
5. Llama 3.1 QA generator는 8K chunk를 읽지만 training example은 whole document를 담는다. Generator가 필요로 하지 않았던 무엇을 model은 이 example에서 학습해야 하는가? Chunk 내용이 document의 다른 위치에 duplicate되어 있을 때 이 design은 어떤 failure를 위험하게 하는가?
6. Artificial Needles data는 TriviaQA를 그대로 뒀지만 NIAH-style data는 6.33 point 낮췄다. 저자의 causal explanation, 그 증거의 status, 이를 시험할 control experiment를 설명하라.
7. ProLong dependency count는 length `n·d` document 하나에 length `d` document `n`개보다 `(n − 1)(d − 1)`개의 span이 더 있다고 말한다. 이 논증이 packed short data의 document masking을 지지하는 이유는 무엇인가? 왜 모순이 아닌가?
8. Qwen2.5-1M의 offline RL은 최대 8,192 token pair만 쓰고 LongBench-Chat을 개선했다. 이것이 long-context ability를 어디에서 train해야 하는가에 대해 보이는 것과 보이지 않는 것은 무엇인가? Long-context loss가 없다고 주장하려면 어떤 추가 measurement가 필요한가?

## 연결

- 이전: ch-27 — Agentic Trajectory Data.
- 다음: ch-29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification.
- 의존성: ch-27 — Agentic Trajectory Data; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation.
- 여기서 쓰는 design pattern: ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix.
- Context-extension method와 mixture: ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression.
- 더 깊은 long-data synthesis: ch-29a — Long-Document Synthesis for Continued Pretraining and Long-Context SFT; ch-29b — Long-Conversation and Accumulating-Context Synthesis.
- SFT mixture의 long-context share: ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares.
- Negative: ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- Long-context RL: ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL.

## 출처

- [[needle-in-haystack-data]] — NIAH harness default, GPT-4 judge scoring, original run count.
- [[excerpts/ruler-task-family]] — RULER task family, protocol, effective-length threshold, Table 3 row, Yi-34B failure analysis.
- [[ruler]] — RULER library card(task list).
- [[babilong]] — bAbI-in-PG19 construction, 10–20% context use, QA1–QA3 result, QA1 fine-tuning transfer, limitation.
- [[helmet]] — Synthetic/RAG task와 application category의 correlation, 128K의 NIAH saturation.
- [[nolima]] — question-needle lexical overlap을 제거했을 때 accuracy 하락.
- [[lost-in-the-middle]] — multi-document QA accuracy의 position dependence.
- [[excerpts/longalign-pipeline]] — LongAlign data construction, manual check, mixing table, loss weighting equation/result.
- [[longalign]] — LongAlign library card.
- [[excerpts/longmit-mimg]] — LongMIT/MIMG pipeline, verifier finding, dataset size, training setting, result.
- [[excerpts/llama3-staged-schedule]] — Llama 3 RoPE base, six-stage long-context gate, long SFT generator, 0.1% share, Table 7/Table 21.
- [[llama-3]], [[llama-3-recipe]] — verified Llama 3 card와 recipe ledger row.
- [[qwen-long-context-synth]] — Qwen2.5-1M stage, synthetic pre-training task, SFT generation, short-pair RL, DCA, short-context Table 6.
- [[prolong]], [[prolong-recipe]] — ProLong mix, short/long ratio, masking, training length, synthetic SFT table, final result.
- [[excerpts/artificial-needles-kv-retrieval]] — synthetic key-value retrieval data, transfer, retention/baseline comparison, limitation.
- [[in2-film]] — IN2 position-balanced synthetic QA와 position-gap result.
- [[context-synthesis-short-to-long]] — Human QA pair용 distractor가 있는 LLM-written context.
- [[nextlong]] — continued-training data용 hard-negative interleaving.
- [[pose-synthesis]] — PoSE training window, correction 30에 사용.
- [[excerpts/primary-quotes-2026-09]] — LongAlpaca, LongChat, Fu et al. 2024, LongRoPE, SmolTalk의 primary quote. Correction과 SmolLM2 observation에 사용.
