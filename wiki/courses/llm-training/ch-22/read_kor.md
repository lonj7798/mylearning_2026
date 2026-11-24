# 제22장 — 품질, 다양성, 그리고 gradient 기반 데이터 선택

> **핵심 통찰.** 각 sample을 한 가지 속성(LLM이 평가한 정확성, instruction-following 난이도, response 길이)으로 점수화하는 선택 방법은 50K–300K pool에서 1K–10K subset을 취할 때 judge 기반 instruction-following score를 개선한다. 그러나 같은 논문에서 MMLU 같은 knowledge benchmark는 하락하고, 어떤 task category는 더 높은 비율로 제거된다([[alpagasus]] Table 2, §6; [[cherry-llm]] Tables 1, 6–7). Pool이 5.8M sample로 커지고 evaluation이 7개 task를 포괄하면 IFD, perplexity, length selection의 평균은 random selection보다 낮다. Multi-task selection에서 7개 방법 중 3개만 balanced random에 도달하고, representation-similarity 방법 RDS+는 10K부터 2.5M까지 시험한 모든 budget에서 balanced random보다 높게 유지된다([[instruction-data-selection-at-scale]] Tables 1–2, Fig. 4). Diversity term은 worst-case performance를 높이고([[qdit-data-diversity]] §4.3), gradient-space diversity는 math와 NLI에서 out-of-distribution accuracy를 예측한다([[prismatic-synthesis]] Table 1).
>
> **지침.** 선택한 set이 여러 task를 지원해야 한다면, non-judge benchmark를 포함한 multi-task suite에서 모든 selector를 balanced random selection과 longest-response baseline에 비교한다. 5.8M Tülu 2 pool에서 326K sample을 선택했을 때 IFD/perplexity 기반 selection은 35.7–41.3으로 balanced random의 47.5에 못 미쳤기 때문이다([[instruction-data-selection-at-scale]] Table 2). Target capability가 알려져 있고 그 example이 몇 개 있으면 gradient-based selection을 사용하고([[less]]), 논문은 target만 평가하므로 non-target benchmark도 함께 보고한다. Quality score를 쓸 때는 diversity term 또는 source별 quota를 추가하고 category별 retention을 보고한다. Quality-only selection이 [[alpagasus]]에서 전체 82% 대비 coding example을 88% 제거했기 때문이다(§6). IFD threshold와 rater threshold는 pool-specific hyperparameter로 취급한다.

## 학습한 버전의 수정 사항

1. “[[prismatic-synthesis]]의 7B-beats-671B … 7B proxy의 diversity-targeted curation이 generator를 두 자릿수 규모로 키우는 것보다 낫다” → PrismMath-7B는 student다. 문제는 Qwen2.5-72B-Instruct가 생성하고 solution은 R1-Distill-Qwen-32B가 생성했다. 이 모델은 671B R1의 data로 학습한 R1-Distill-Qwen-7B보다 7개 benchmark 중 6개에서 높다. Gradient proxy는 Qwen2.5-0.5B-Instruct다([[prismatic-synthesis]], Abstract, §2.1, §3.1, Table 4).
2. “Synthetic pool의 대부분은 harmful 또는 neutral이다” → 인용한 source 중 이 비율을 측정한 것은 없다. 이 장은 이제 측정된 비교만 서술한다.
3. “StackExchange-only single-source SFT pool을 2K에서 32K로 키우면 preference improvement가 0이었다” → 이 ablation은 LLaMA 7B, ChatGPT가 1–6 helpfulness scale로 채점한 response quality, 지수적으로 증가하는 Stack Exchange set을 사용했다. 논문은 training set을 두 배로 늘려도 response quality가 개선되지 않았다고 보고한다. Human-preference result는 아니다([[lima]], arXiv:2305.11206 §5, Fig. 6).
4. AlpaGasus가 “모든 judged benchmark에서 full 52K set을 이겼다” → 7B와 13B에서 네 test set 모두 GPT-4-judged win이 loss보다 많았다(Fig. 5). 그러나 MMLU는 7B에서 40.86→38.78, 13B에서 47.89→46.12로 하락했다([[alpagasus]] Table 2).
5. AlpaGasus rubric이 “relevance, factual correctness, completeness, format을 각각 0–5로 매긴 뒤 average” 및 “accuracy와 helpfulness를 rating”한다 → 각 rating run은 하나의 named dimension에서 0–5 score 하나를 요구한다. Main run은 “accuracy”를 사용하고 helpfulness는 별도 experiment다([[alpagasus]] §2.3, Fig. 3, App. A.5).
6. “[[alpagasus]] sweep은 4.5 > 4.0 > 3.5를 보인다” → τ = 4.5(9,229 example)와 τ = 4.0(39k)만 training했고 τ = 3.5 run은 없다([[alpagasus]] §4.2, Fig. 7).
7. “IFD = PPL_cond(a|q) / PPL_uncond(a)” → Cherry LLM은 IFD를 mean token loss의 비 `s(A|Q)/s(A)`로 정의하고 script도 그 비를 계산한다. Superfiltering은 perplexity 비를 출력한다. 두 정의는 sample 순위를 다르게 매길 수 있다(아래 §3)([[cherry-llm]] Eq. 3–5; [[superfiltering]] Eq. 2).
8. “IFD ≈ 1이면 q는 무관하고 learning value가 낮다”, “<1 band 안에서 top 5–15%를 유지한다” → IFD가 높을수록 instruction이 answer prediction에 덜 도움이 되며 논문은 이를 더 어려운 sample로 읽는다. Cherry LLM은 IFD > 1을 제거하고 남은 것 중 가장 높은 값을 유지한다. Lowest-IFD subset training은 가장 낮은 winning score를 냈다. Test budget은 5–20%이고 논문은 top 10%를 권한다([[cherry-llm]] §2.2, §4.2.3, App. G.2).
9. “[[ifd]]는 random pool sample 약 1K의 1-epoch warmup을 명시한다 … 이를 건너뛰면 IFD는 template-mismatch noise가 지배한다” → 1,000 pre-experience sample은 100 K-means cluster에서 cluster당 10개다. 0 pre-experience sample은 시험한 count 중 결과가 가장 낮았지만 10% data에서 Alpaca는 여전히 이겼다. LLaMA2 run은 base model을 직접 사용했다([[cherry-llm]] §2.1, §4.3.1, §4.4).
10. Superfiltering “GPT-2-125M … 높은 Spearman ρ … 선택한 15%로 7B 학습” → proxy는 GPT-2(124M)다. LLaMA2-7B와 IFD rank correlation은 Alpaca 0.679, Alpaca-GPT4 0.788, WizardLM-70k 0.802이다. Top-15% overlap은 0.49, 0.51, 0.61이다. Filtering은 8분, LLaMA2-7B IFD는 161분이며, LLaMA2-7B IFD가 고른 data가 GPT-4 pairwise comparison에서 Superfiltering data를 이겼다([[superfiltering]] arXiv:2402.00530v2 Tables 1, 4).
11. “Family-mismatched proxy는 transfer를 깨므로 plausibly similar tokenizer의 proxy를 쓴다” → 논문 proxy GPT-2는 LLaMA2와 family/tokenizer가 다르고 family-mismatch experiment도 보고하지 않는다([[superfiltering]] §3).
12. “IFD, Superfiltering, LESS는 모두 random pool subset에서 짧은 LoRA/full-rank warmup이 필요하다” → Superfiltering은 training 없는 GPT-2를 쓰며(§3.3, §6.1), Cherry LLM set은 cluster-sampled이고, random subset warmup은 LESS만 한다([[superfiltering]]; [[cherry-llm]] §2.1; [[less]] §5.1).
13. DEITA complexity가 “13B LLM이 score”하고 mutation이 “constraint 추가, depth/breadth 증가”한다 → complexity/quality scorer는 2K seed sample의 evolved variant 6개를 ChatGPT ranking으로 학습한 LLaMA-1 7B이고, embedding은 LLaMA-1 13B다. Complexity mutation은 constraint 추가, deepening, concretizing, reasoning step 증가다([[deita]] §2.3–2.5).
14. DEITA가 sample을 “이미 선택한 모든 sample과 embedding distance가 threshold τ(≈ 0.9 cosine)를 넘을 때만” 들인다 → nearest selected sample과 cosine similarity가 0.9 미만일 때 들인다([[deita]] §2.5, released code).
15. “Diversity filter를 빼면 downstream score가 붕괴하고 complexity를 빼면 reasoning benchmark, quality를 빼면 format compliance가 약해진다” → DEITA는 leave-one-dimension-out ablation을 보고하지 않고 MT-Bench의 single-dimension study만 보고한다(Tables 2–4)([[deita]]).
16. “6K–10K DEITA-selected sample은 MT-Bench에서 Zephyr-7B-beta(약 200K SFT)와 match한다” → 6K SFT DEITA-Mistral-7B는 200K SFT-only zephyr-beta-sft의 5.32에 비해 7.22다. DPO 뒤 zephyr-beta는 7.34, 6K SFT+10K DPO pair DEITA는 7.55다([[deita]] Table 6).
17. DEITA는 “lexicographic이며 weighted sum이 아니다 … diversity 없는 pure combined score가 near-duplicate로 붕괴하기 때문이다” → Algorithm 1은 complexity × quality로 sort한 후 similarity로 filter한다. 논문은 weighted-sum objective와 비교하지 않는다([[deita]] §3.1).
18. LESS가 “full SFT budget의 약 4%로 target base model을 pool에 LoRA-train한다” → random 5% subset으로 4 epoch LoRA warmup한다([[less]] arXiv:2402.04333 §4.1, §5.1).
19. LESS가 “raw gradient component `g_{i,k}`를 `g_{i,k}/(√v̂_k + ε)`로 바꾼다” → warmup run moment를 써서 Adam update direction `m/(√v + ε)`를 사용하고, 4개 warmup checkpoint에서 learning-rate-weighted cosine similarity를 합한다([[less]] Def. 3.1, Eq. 2).
20. “고정 Gaussian random projection을 d ≈ 8K로 쓴다” → projection entry는 Rademacher(uniform on {−1, 1})이고 d = 8192다([[less]] §4.1).
21. “5% LESS-selected가 MMLU, BBH, TydiQA에서 100% random을 이긴다” → LLaMA-2-7B에서 LESS 5% 대 full 270K는 MMLU 50.2 대 51.6, TydiQA 56.2 대 54.0, BBH 41.5 대 43.2다. LESS는 5% random(46.5, 52.7, 38.9)을 세 task 모두에서 이긴다([[less]] Table 2).
22. “SGD-influence와 Adam-influence가 top 5%의 약 30%에서 불일치한다” → 이 수치는 논문에 없다. All-data warmup 비교의 average는 Adam 50.5, SGD 49.7, SignGD 47.8이다([[less]] App. D.2, Table 9).
23. G-Vendi가 “random-project … 약 8K dimension”, “ρ = K / tr(K)” → d = 1024이고 K = GGᵀ/|D|이며, G 각 row가 unit norm이면 K/tr(K)와 같다([[prismatic-synthesis]] §2.1, Eq. 3).
24. Prismatic pipeline이 “각 candidate의 G-Vendi marginal contribution score, greedy max-entropy selection, math answer-verifier” → proxy gradient K-means clustering을 하고 smallest cluster에 들어간 generated sample을 유지한다(k = pool size의 1%, math에서는 smallest k/2 cluster). Ground-truth answer 없이 majority-vote filtering한다([[prismatic-synthesis]] §3.1).
25. “실제로 OOD transfer를 예측하는 것은 diversity metric” → correlation은 quality를 고정한 math/NLI에서만 측정했고, 저자는 G-Vendi만으로 다른 generation process의 dataset을 비교하지 말라고 경고한다([[prismatic-synthesis]] Table 1, App. D).
26. “Tülu 3 data-selection stage는 DEITA-lineage recipe를 쓴다” → Tülu 3 SFT mix는 skill-specific submix, persona-driven synthesis, decontamination으로 만들었고 DEITA-style scorer는 없다([[tulu-3-sft-mix]]).
27. IFD compute가 “SFT 1 epoch 비용 정도” → 두 forward pass는 token당 최대 4N FLOPs(unconditioned pass가 더 짧음), training epoch은 6N이므로, scoring은 [[instruction-data-selection-at-scale]] §3.3의 2N/6N convention에서 pre-experience training을 더하기 전 최대 1 epoch의 2/3이다(derived).
28. Connections “Track 3(SFT-at-scale)”, “Track 4(RL)”, “ch-21(synthetic generation)” → 현재 outline의 id/title은 아래 Connections를 보라. 이전 figure의 두 번째 IFD axis(“query-reversed IFD”)는 cited paper에 정의돼 있지 않아 figure를 다시 썼다.

## 이 장이 범용 모델에서 중요한 이유

Synthetic generation(ch-19부터 ch-21)은 SFT budget보다 큰 pool을 만든다. Data selection은 training에 쓸 pool의 subset을 고르는 단계다. Pipeline `pre-training → mid-training → **synthetic generation → selection** → SFT → preference optimization → RL → evaluation`에서 generation과 SFT 사이에 있고, 같은 scoring idea가 RL prompt filtering(ch-16)에도 다시 나타난다.

측정 가능한 문제는 세 부분이다. 첫째, selector는 training set의 task distribution을 바꾸므로 하나의 benchmark는 올리고 다른 benchmark는 낮출 수 있다. 둘째, AlpaGasus, Cherry LLM, DEITA, QDIT, Long Is More는 main gain을 response length와 preference가 상관되는 LLM judge로 보고한다([[judge-llm-bias]]). 셋째, 50K–300K pool에서 10K sample을 선택해 얻은 결과는 1M-sample pool과 multi-task target에서 유지되지 않을 수 있다. 범용 모델에서 selector가 유용하려면 실제 사용할 scale에서 broad suite로 random selection을 이겨야 한다.

## §1 Baseline: random, balanced random, length

**정의.** *Pool* D는 candidate `(instruction, response)` pair의 집합이다. *Budget* n은 유지할 sample 수다. *Random selection*은 D에서 n sample을 uniform하게 뽑는다. *Balanced random selection*은 source 전반에서 uniform하게 뽑고 소진된 source의 budget을 재분배한다([[instruction-data-selection-at-scale]] §3). *Length selection*은 response가 가장 긴 n sample을 유지한다([[long-is-more-for-alignment]]).

**Baseline을 먼저 두는 이유.** Selector는 그 score가 sample의 training value와 상관한다고 주장한다. Random selection은 budget을 통제한다. Balanced random selection은 source imbalance를 통제한다. Filter하지 않은 Tülu 2 pool은 FLAN/Open Orca data가 지배하므로 이것이 중요하다([[instruction-data-selection-at-scale]] §3.2). Length selection은 긴 output을 선호하는 judge preference를 통제한다.

**Data quantity만으로 judged quality가 오르지 않는 근거.** LIMA ablation에서 LLaMA 7B를 지수적으로 늘리는 Stack Exchange set에 학습시키고 ChatGPT가 helpfulness를 1–6 scale로 채점했다. Training set을 두 배로 늘려도 score는 개선되지 않았지만, prompt diversity(Stack Exchange 대 wikiHow)와 response filtering(0.5-point 차이)은 개선됐다([[lima]] arXiv:2305.11206 §5, Figs. 5–6). **결과(single study).**

**이 근거의 한계.** LIMA ablation은 judge로 helpfulness를 측정한다. Objective benchmark에서는 fine-tuning example 수에 따라 GSM8K, multihop QA, coding, IFEval의 post-training performance가 power law로 오르고, style/format은 약 100 example 안에 saturation하지만 reasoning accuracy는 계속 개선된다([[revisiting-superficial-alignment-hypothesis]] §2, Fig. 1). Tülu 3 SFT mix에는 prompt 939,344개가 있다([[tulu-3-sft-mix]]). 따라서 “작은 curated set이면 충분하다”는 말은 evaluation과 capability에 달려 있다. Objective benchmark를 쓴 두 source([[revisiting-superficial-alignment-hypothesis]]; [[instruction-data-selection-at-scale]])가 이 주의를 **replicate**한다.

## §2 LLM-rated quality: AlpaGasus

**정의.** LLM rater G는 prompt `p_G`와 함께 instruction, input, response를 받아 score를 반환한다. 유지하는 set은 다음과 같다.

```
S = { x ∈ V : G(x, p_G) ≥ τ }        (AlpaGasus Eq. 1)
```

V는 instruction dataset, x는 하나의 `(instruction, input, response)` triplet, τ는 threshold다.

**다루는 문제.** Alpaca의 response 52,002개는 response-level check 없이 text-davinci-003가 생성했으므로, 일부는 incorrect하거나 instruction과 무관하다([[alpagasus]] Abstract).

**작동 방식.**

1. ChatGPT가 각 triplet을 하나의 named dimension(main run에서는 “accuracy”)에서 0부터 5까지 score하고, score를 첫 줄에 쓴다(Fig. 3).
2. Score histogram을 읽는다: <3: 172; 3: 1,550; 3.5: 10,811; 4: 30,240; 4.5: 9,218; 5: 11(Fig. 4).
3. Score histogram에 “according to”하여 τ = 4.5를 고른다. Example 9,229개, data의 17.75%를 유지한다(§2.3, §4.1).
4. Alpaca script로 유지한 set에 LLaMA를 학습한다.

**계산 예시.** Histogram에서 τ = 4.5는 9,218 + 11 = 9,229 example을 유지한다. τ를 4.0으로 내리면 score 4.0의 example 30,240개가 더해져 39,469개를 유지하며, 논문은 이것을 39k subset으로 보고한다. τ를 0.5 바꾸면 유지 set이 4.3배가 된다. Histogram mode인 score 4.0 example 30,240개가 τ = 4.0에서는 포함되고 τ = 4.5에서는 제외되기 때문이다.

**근거.** GPT-4를 judge로 한 Alpaca-52k 대비에서, 9k model은 7B의 네 test set 모두에서 win이 loss보다 많다(예: WizardLM W/T/L 83/65/70). 13B에서도 그렇다(Fig. 5). Random 9k subset도 이긴다(Fig. 6). 7B training time은 80분에서 14분으로 줄었다(Table 5). **결과(single study).**

**조건과 한계.**

- MMLU는 7B에서 40.86 → 38.78, 13B에서 47.89 → 46.12로 떨어졌다. BBH, DROP, HumanEval은 0.12–0.75 point 올랐다(Table 2).
- 전체 제거율 82.25%에 비해 coding example은 88.16%(718 → 85)가 제거됐다. Coding을 포함한 WizardLM skill 29개 중 7개에서 AlpaGasus-7B가 더 좋지 않았다(§6).
- Rater는 code를 실행하거나 math를 확인할 수 없다. 논문은 rater model version이나 temperature를 보고하지 않는다.

**범용 모델에 대한 함의.** Rater threshold는 category proportion을 바꾼다. 범용 SFT set에 rater threshold를 적용한다면, training 전에 category별 retention을 보고해야 한다. [[alpagasus]]에서 coding share가 전체 share보다 더 빨리 줄었기 때문이다(§6).

## §3 Instruction-Following Difficulty(IFD)

**정의.** Weight θ를 가진 model 아래에서 question Q(instruction과 input)와 N token의 answer A에 대해 다음과 같이 둔다.

```
s(A|Q) = −(1/N) Σ_{i=1..N} log P(w_i | Q, w_1 … w_{i−1}; θ)     (Cherry LLM Eq. 3)
s(A)   = −(1/N) Σ_{i=1..N} log P(w_i | w_1 … w_{i−1}; θ)        (Eq. 4)
IFD(Q, A) = s(A|Q) / s(A)                                         (Eq. 5)
```

`w_i`는 i번째 answer token이고 θ는 score를 매기는(“pre-experienced”) model의 weight다. [[superfiltering]](Eq. 2)은 `PPL(y|x)/PPL(y)`로 IFD를 쓰며, 여기서 `PPL = exp(mean token loss)`다. Card [[ifd]]도 perplexity form을 반복한다. 이 장은 Cherry LLM released script가 계산하는 loss-ratio form을 쓴다([[cherry-llm]], Cherry_LLM@01fac8d `cherry_seletion/data_by_IFD.py` L103–L120).

**다루는 문제.** External rater scoring은 비용이 크고 rater의 한계를 그대로 물려받는다. IFD는 model 자체의 loss만 사용한다.

**작동 방식.**

1. Instruction embedding의 100개 K-means cluster에서 cluster당 10개씩 고른 1,000 sample로 pre-experienced model을 1 epoch 학습한다(§2.1).
2. 모든 sample에서 `s(A|Q)`와 `s(A)`를 계산한다(two forward pass).
3. Instruction이 answer prediction을 더 어렵게 하는 IFD > 1 sample을 제거한다(§2.2).
4. 나머지를 sort하고 가장 높은 IFD value, 즉 instruction이 가장 덜 도움이 되는 sample을 유지한다(§1, §2.2). 논문은 top 10%를 권한다(App. G.2).
5. 유지한 set에 base model을 학습한다.

**계산 예시(논문).** “Give a brief description of the coronavirus.”는 `s(A) = 0.761`, `s(A|Q) = 0.696`이므로 `IFD = 0.696 / 0.761 = 0.915`다. Answer가 “Frustration”인 “What emotion is expressed in this tweet?”는 `IFD = 0.601 / 6.593 = 0.091`이다([[cherry-llm]] App. F, Fig. 9). Cherry LLM은 첫 sample을 두 번째보다 높게 rank한다. Instruction이 “Frustration”의 loss를 11배 줄이므로, 두 번째 sample은 instruction이 주어지면 쉽기 때문이다.

**계산 예시(두 정의가 불일치함).** Sample A는 `s(A|Q) = 0.2`, `s(A) = 0.4`다. Loss ratio는 0.50, perplexity ratio는 `exp(0.2 − 0.4) = 0.82`다. Sample B는 `s(A|Q) = 3.0`, `s(A) = 4.0`다. Loss ratio는 0.75, perplexity ratio는 `exp(−1.0) = 0.37`이다. Loss ratio는 B를 A보다 높게 rank하고 perplexity ratio는 A를 B보다 높게 rank한다. 재구현은 어느 form을 쓰는지 밝혀야 한다(derived).

**근거.** Alpaca의 5%(IFD 기준)로 학습한 LLaMA-7B는 Open LLM average 52.06, AlpacaEval 34.74를 얻는다. Official Alpaca의 50.21, 26.46보다 높다(Table 1). 5%에서 IFD selection average 52.06은 random 50.61, K-means diversity 49.48, low IFD 50.77보다 높다(App. B Table 5). **결과(single study).**

**조건과 한계.**

- LLaMA-7B의 MMLU는 5% Alpaca에서 41.73 → 36.51, 10% WizardLM에서 37.75 → 33.08로 떨어졌다(Tables 1, 3).
- 5% Alpaca model은 Math/Coding에서 official Alpaca보다 나쁘다. 10% WizardLM model은 Math, Code, Complex Format, Counterfactual에서 나쁘다(App. C, Tables 6–7).
- Selection은 open-ended generation에 집중된다(top 5%에 “write story” 119개). Editing task는 제거된다(bottom 5%에 “rewrite sentence” 155개)(Table 4).
- IFD는 주어진 answer의 loss만 쓴다. 논문은 유지한 answer가 correct한지 검사하지 않는다(Eq. 3–5).
- 5.8M-sample scale에서 IFD는 7 task 평균 35.8, random selection은 40.7이다. SQuAD는 46.6 대 83.6이다([[instruction-data-selection-at-scale]] Table 1).

**범용 모델에 대한 함의.** IFD는 instruction에서 answer를 예측하기 어려운 instruction을 선호하며, 이는 대부분 open-ended generation task다. 이 선호는 judge score를 올리고 extraction/knowledge score를 낮춘다. 이는 [[cherry-llm]](App. C)과 [[instruction-data-selection-at-scale]](§5; IFD는 FLAN data를 더 선택하고 AlpacaEval에서만 잘함)에서 **replicate**된다.

[figures/ifd-scatter.html](figures/ifd-scatter.html) figure는 독자가 `s(A|Q)`와 `s(A)`를 입력해 loss-ratio/perplexity-ratio IFD를 비교하고 Cherry LLM keep rule이 적용되는 모습을 보게 한다.

## §4 Weak-to-strong IFD: Superfiltering

**정의.** Superfiltering은 학습할 model 대신 작은 pretrained model GPT-2(124M)로 IFD를 계산하고, IFD가 1 미만인 sample 중 상위 k%를 유지한다([[superfiltering]] §3.3).

**다루는 문제.** LLaMA2-7B로 Alpaca에서 IFD를 계산하는 데는 161분이 걸리고 reward model은 1,400분이 걸렸다(Table 4).

**작동 방식.**

1. 추가 학습 없이 GPT-2로 모든 sample의 IFD를 계산한다(§3.3, §6.1).
2. IFD < 1인 sample 가운데 상위 5%, 10%, 또는 15%를 유지한다.
3. 유지 set으로 LLaMA2-7B 또는 13B를 학습한다.

**수식(rank agreement).** Spearman의 `ρ = 1 − 6 Σ d_i² / (n(n² − 1))`이며, `d_i`는 두 scorer 아래에서 한 sample이 차지한 rank의 차이, n은 sample 수다.

**계산 예시.** LLaMA2-7B가 5개 sample을 `(1, 2, 3, 4, 5)`, GPT-2가 `(2, 1, 3, 5, 4)`로 rank하면 `d² = 1, 1, 0, 1, 1`, `Σ = 4`이고 `ρ = 1 − 24/120 = 0.80`이다. `n = 5`, top-2 budget이면 두 top set은 `{1, 2}`와 `{2, 1}`이므로 overlap은 100%다. 큰 pool의 top 5%에서는 같은 ρ가 낮은 overlap과 함께 나올 수 있다. 그래서 논문은 overlap을 별도로 보고한다.

**근거.** GPT-2와 LLaMA2-7B IFD의 ρ는 Alpaca, Alpaca-GPT4, WizardLM-70k에서 각각 0.679, 0.788, 0.802이다. Top-15% overlap은 각각 0.49, 0.51, 0.61이다(Table 1). Alpaca 5%(sample 2,600개)로 학습한 LLaMA2-7B는 Open LLM average 55.67, AlpacaEval 33.04를 얻고, full data의 55.25, 27.75보다 높다. MMLU는 45.21 대 47.02다(Table 2). Filtering은 8분이 걸렸다(Table 4). **결과(single study).**

**조건과 한계.** LLaMA2-7B IFD가 고른 data는 Superfiltering data를 이겼다. 이에 대한 Superfiltering의 pairwise winning score는 5/10/15%에서 각각 0.853, 0.761, 0.927이다(Table 4). Proxy는 더 싸지만 target model보다 더 나쁜 data를 고른다.

**범용 모델에 대한 함의.** Pool이 너무 커서 target model로 score할 수 없을 때 cheap proxy는 적절하다. 하지만 proxy는 §3의 IFD가 만드는 모든 narrowing effect도 물려받는다.

## §5 Complexity, quality, diversity: DEITA와 QDIT

**DEITA 정의.** DEITA는 각 sample에 `s = c × q`로 score를 매긴다. c는 complexity score, q는 quality score이며, 각각 2K seed sample의 evolved variant 6개에 대한 ChatGPT ranking으로 학습한 LLaMA-1 7B model이 예측한다([[deita]] §2.3–2.4). 이어 s 내림차순으로 pool을 훑으며 sample x를 다음일 때만 들인다.

```
max_{y ∈ S} cos(e(x), e(y)) < τ,   τ = 0.9     (REPR FILTER, §2.5)
```

S는 selected set이고 `e(·)`는 LLaMA-1 13B sentence embedding이다. Selection은 m sample에서 멈춘다(Algorithm 1).

**QDIT 정의.** QDIT는 facility-location function으로 pool V의 subset A 다양성을 재고, 이를 quality와 선형 결합한다([[qdit-data-diversity]] Eq. 1, §3.2).

```
d(A) = Σ_{v ∈ V} max_{a ∈ A} sim(a, v)
f(a | A, α) = (1 − α) · d(a | A) + α · q(a)
```

sim은 instruction embedding의 cosine similarity, `d(a|A)`는 a를 더해서 생기는 d의 gain, `q(a)`는 quality score(ChatGPT 또는 reward model)이며, `α ∈ [0, 1]`이다. QDIT는 매 step에서 f가 가장 높은 sample을 더한다(Algorithm 1).

**계산 예시(두 diversity rule).** Pool `V = {v1, v2, v3}`에서 `sim(v1, v2) = 0.95`, `sim(v1, v3) = 0.10`, `sim(v2, v3) = 0.20`, quality `q = (0.9, 0.8, 0.3)`이라고 하자. `τ = 0.9`인 DEITA는 quality로 sort한다. v1은 들어가고, v2는 `0.95 ≥ 0.9`이므로 거절되며, v3은 들어간다. `A = {v1}`에서 시작하는 `α = 0.5` QDIT는 `d({v1}) = 1 + 0.95 + 0.10 = 2.05`다. v2를 더하면 `d = 1 + 1 + 0.20 = 2.20`, gain 0.15, `f = 0.5·0.15 + 0.5·0.8 = 0.475`다. v3를 더하면 `d = 1 + 0.95 + 1 = 2.95`, gain 0.90, `f = 0.5·0.90 + 0.5·0.3 = 0.60`이다. 따라서 QDIT는 v3를 고른다. `α = 0.9`이면 `f(v2) = 0.735`, `f(v3) = 0.36`이므로 QDIT는 v2를 고른다. DEITA threshold는 hard rule이고 QDIT의 α는 quality와 coverage 사이를 trade-off한다.

**근거(DEITA).** LLaMA-1-13B, 6K sample에서 DEITA는 MT-Bench 6.46, AlpacaEval 77.08을 얻는다. Random은 5.84 / 73.91, Alpagasus는 5.61 / 71.21이다(Table 5). 6K SFT sample의 DEITA-Mistral-7B는 7.22 / 80.78이고, random 10K는 5.89 / 56.90, 200K zephyr-beta-sft는 5.32 / 75.12다(Table 6). Open LLM Leaderboard에서 DEITA-LLaMA1-13B 10K average는 64.27, random 10K는 60.14다(Table 7). **결과(single study).**

**근거(QDIT).** LLaMA-1 7B로 5개 pool에 평균해 보면 quality-only selection은 5 setting 중 2개에서 random 대비 lowest-10% winning score를 낮췄다. QDIT는 quality-only selection보다 worst-case reward-model score를 5.2%, worst-case winning score를 6.26% 개선했다(§4.3). UltraChat 10K에서 quality-only selection은 DROP을 random 26.24에서 17.01로 낮췄지만 QDIT는 26.73이었다(Table 3). `α ∈ {0.5, 0.7, 0.9}`가 가장 좋았고 `α = 0.1`, `α = 1`은 더 나빴다(§4.4). **결과(single study).**

**조건과 한계.** DEITA metric study는 GPT-4-judged MT-Bench만 쓰고 seed를 보고하지 않는다. 저자는 AlpacaEval gain이 MT-Bench gain과 일관되지 않는다고 언급한다(§2.2, §3.3). DEITA selection에서 m이 커질수록 MT-Bench는 올랐다가 내려가며, 3K sample은 300K 전체와 match한다(Fig. 2). QDIT는 Claude 2 judge와 reward model, 그리고 Table 3의 benchmark 6개로 평가한다. 어느 논문도 1.3M sample을 넘는 pool은 시험하지 않는다.

**범용 모델에 대한 함의.** Diversity term은 average보다 task distribution의 tail(worst-case prompt)을 보호한다. Quality-only selection이 coverage를 좁힌다는 방향은 [[alpagasus]](§6), [[cherry-llm]](Table 4), [[qdit-data-diversity]](§4.3)에서 **replicate**된다.

## §6 Gradient-based targeted selection: LESS

**정의.** LESS(Low-rank gradiEnt Similarity Search)는 작은 target validation set의 loss gradient와 training update가 얼마나 align하는지로 training sample z를 score한다([[less]] arXiv:2402.04333).

**다루는 문제.** Text/embedding similarity는 같은 skill을 학습시키는 example이 아니라 target과 닮은 example(예: Bengali QA target에 대한 Bengali text)을 고른다(§6.2, Table 16).

**수식.** Adam에서 checkpoint θ에서 sample별 update direction은 element-wise `Γ(z, θ) = m / (√v + ε)`다. Validation example z′에 대한 z의 influence는 다음이다.

```
Inf_Adam(z, z′) = Σ_{i=1..N} η̄_i · cos( ∇ℓ(z′; θ_i), Γ(z, θ_i) )     (Def. 3.1)
```

N은 warmup checkpoint 수(epoch마다 하나), `η̄_i`는 epoch i의 average learning rate, `θ_i`는 epoch i 뒤 checkpoint, ℓ은 completion의 mean token cross-entropy다. m, v는 warmup run에서 취한 Adam first/second moment다. Subtask j가 있는 target task에서는 subtask 안에서 validation gradient를 average하고 sample score는 subtask들 가운데 maximum이다(Eq. 2, §4.2).

**작동 방식.**

1. Pool의 random 5%에서 base model을 `N = 4` epoch LoRA-train하고 epoch마다 저장한다(§4.1, §5.1).
2. 모든 pool sample/checkpoint에서 LoRA gradient를 계산하고 Γ를 만들며, entry가 `{−1, 1}`에서 uniform인 `d × P` matrix Π로 projection한다. `d = 8192`다(§4.1).
3. Projected feature를 저장한다. Gradient datastore는 약 270K sample에 17.7 GB, A100 48시간이 든다(Table 4).
4. Target마다 projected validation gradient를 계산하고 Eq. 2로 score해 top 5%를 유지한다(§5.1).
5. 유지 set으로 target model을 학습한다(논문에서는 LoRA).

**계산 예시.** Average LR가 `η̄ = (2e-5, 1e-5)`인 checkpoint 두 개를 생각하자. Sample z는 `θ_1`에서 validation gradient와 `cos = 0.30`, `θ_2`에서 0.10이므로 `Inf = 2e-5·0.30 + 1e-5·0.10 = 7e-6`다. 두 번째 training sample `z₂`는 cosine이 0.05와 0.40이므로 `Inf = 1e-6 + 4e-6 = 5e-6`다. 첫 checkpoint의 average learning rate가 더 커 cosine에 더 큰 weight가 가므로 z가 더 높은 rank다.

**Dot product가 아니라 cosine을 쓰는 이유.** Sequence gradient는 token gradient의 average이므로 completion length가 길수록 norm이 작아진다. 따라서 dot-product influence는 short training sequence에 더 큰 weight를 주며, 저자는 이것이 더 나쁜 performance를 낸다고 보고한다(§3.2, Table 13).

**근거.** LLaMA-2-7B, pool의 5%에서 LESS는 MMLU 50.2, TydiQA 56.2, BBH 41.5다. Random 5%는 46.5, 52.7, 38.9이고 full data는 51.6, 54.0, 43.2다(Table 2). LLaMA-2-7B datastore로 LLaMA-2-13B/Mistral-7B의 data를 선택한 LESS-T는 random selection을 이긴다(Table 2). 이 270K scale에서 BM25, DSIR, RDS selection은 “minimal improvement over random”을 보인다(Table 3). Warmup도 중요하다. Untrained base model gradient는 MMLU 46.7이고, 5% warmup 뒤에는 50.2다(Table 5). 5% 대신 full pool warmup에서는 Adam influence average가 50.5, SGD influence가 49.7, SignGD가 47.8이다(Table 9). 각 수치는 **결과(single study)**다.

**조건과 한계.**

- 논문은 자신이 선택하는 target 3개만 평가한다. Selection 후 non-target task performance는 보고하지 않는다.
- LLaMA-2-13B/Mistral-7B에서 LESS selection 후 MMLU(54.0, 61.8)는 untuned base model(55.3, 62.4)보다 낮다(App. D.3, Table 10).
- Duplicate는 같은 score를 받고 value를 두 번 더하는 것으로 취급된다(Limitations 4).
- 200K-pool, seven-task setting에서 LESS average는 44.2이고 RDS+는 46.4다. HumanEval은 random 27.0에 비해 19.6이다([[instruction-data-selection-at-scale]] Table 1). Compute cost 때문에 5.8M에서는 실행하지 않았다.

**범용 모델에 대한 함의.** LESS는 “어느 sample이 이 target에 도움이 되는가”에는 답하지만 “어느 sample이 다른 ability를 유지하는가”에는 답하지 않는다. 논문은 target-task score만 보고하므로, 범용 pipeline 안에서 LESS를 쓸 때는 training 뒤 non-target task의 held-out suite도 평가해야 한다([[ch-30a]] — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; [[ch-47]] — Evaluation Harness and Suite Design for General Capability).

## §7 Gradient-space diversity: G-Vendi와 Prismatic Synthesis

**정의.** G-Vendi는 작은 frozen proxy model의 normalized/projected loss gradient로 만든 similarity matrix eigenvalue의 exponentiated entropy로 dataset D의 diversity를 측정한다([[prismatic-synthesis]] §2.1).

**수식.**

```
g(x, y)  = −∇log P(y|x; θ) / ‖−∇log P(y|x; θ)‖,   g_proj = Πᵀ g,  Π_ij ~ U({−1, 1})   (Eq. 1–2)
G-Vendi(D) = exp( − Σ_i λ_i log λ_i ),   λ_i = eigenvalues of K = G Gᵀ / |D|      (Eq. 3)
```

θ는 proxy parameter(Qwen2.5-0.5B-Instruct), d = 1024는 projected dimension이며 `G ∈ ℝ^{|D|×d}`는 projected gradient를 쌓은 matrix다. K의 모든 off-diagonal entry가 0이면 G-Vendi는 |D|이고 모두 1이면 1이다(App. C).

**계산 예시.** Gradient cosine이 c인 sample 둘에서 `K = [[1, c], [c, 1]] / 2`의 eigenvalue는 `(1 + c)/2`, `(1 − c)/2`다. `c = 0`이면 `λ = (0.5, 0.5)`, entropy는 `ln 2`, G-Vendi는 2다. `c = 0.8`이면 `λ = (0.9, 0.1)`, entropy는 `−(0.9 ln 0.9 + 0.1 ln 0.1) = 0.325`, G-Vendi는 1.38이다. `c = 1`이면 G-Vendi는 1이다.

**근거.** 1.5M-sample synthetic pool에서 뽑은 300개 subset을 대상으로, Table 1의 100K subset scale에서 G-Vendi는 average OOD accuracy와 Spearman `ρ = 0.899`(math), `ρ = 0.893`(NLI)에 도달한다. Proxy보다 14배 큰 encoder를 쓴 Embedding Vendi의 math 0.754, Skill-Set Entropy 0.812보다 높다(Table 1). Diversity가 높은 10K subset은 OOD benchmark에서 diversity가 낮은 100K subset을 자주 이긴다(§2.3.1). PrismMath-7B average는 R1-Distill-Qwen-7B의 72.80보다 높은 75.25다(Table 4). **결과(single study).**

**Prismatic Synthesis 작동 방식.** (1) Proxy gradient를 계산하고 K-means를 실행한다. (2) Pool에서 few-shot prompting하여 새 sample을 생성한다. (3) Sparse cluster에 들어가는 generated sample을 유지한다(math에서는 `k = pool size의 1%`, smallest `k/2` cluster). (4) Sampled solution의 majority vote(math `N = 3`, `τ = 2`)로 filter하고 decontaminate한다(§3.1).

**조건과 한계.** In-distribution accuracy는 G-Vendi와 상관이 더 낮고(`ρ = 0.780`), dataset size에 더 의존한다(Table 1). Correlation은 quality를 control했을 때만 성립하므로, G-Vendi 하나로 서로 다른 generator의 dataset을 비교하면 안 된다(App. D). Coding, open-ended chat, agentic data는 시험하지 않았다.

**범용 모델에 대한 함의.** Gradient-space diversity는 두 task family에서 검증한 breadth의 candidate signal이다. Broad multi-domain suite의 held-out performance를 예측하는지는 **open question**이다.

## §8 Scale에서의 selection: 많은 방법이 random보다 낮다

**Setting.** [[instruction-data-selection-at-scale]]는 Tülu 2에 고려한 전체 data로 pool을 만든다(exact deduplication 뒤 5.8M sample). MMLU, GSM8K, BBH, TydiQA, HumanEval-Codex, SQuAD, AlpacaEval의 7개 evaluation을 위한 selection을 하고, Llama 2 7B를 2 epoch full fine-tune한다(§3.2–3.3). *RDS+*는 sample에 대한 base model last hidden state의 position-weighted mean pool과 query example에 대한 그것 사이 cosine similarity로 pool sample을 score하고, query example/task 전반에서 round-robin으로 선택한다(§3, §3.1).

**근거.**

- Single-task, 10K selected: 200K에서 RDS+ 46.4, LESS 44.2, Length 43.5, random 40.9, IFD 38.7이다. 5.8M에서는 RDS+ 50.5, Embed(GTR) 46.4, random 40.7, IFD 35.8, Top-PPL 30.4, Length 17.8이다(Table 1).
- 200K에서 5.8M pool로 옮기면 두 크기에서 실행한 모든 method average가 RDS+(46.4 → 50.5)와 Embed(GTR)(42.1 → 46.4)를 제외하고 하락한다. 예를 들어 Length는 43.5 → 17.8, IFD는 38.7 → 35.8이다(Table 1; 저자는 §4.1에서 PPL, random, IFD, Embed(NV)를 지목한다).
- Multi-task, 326K selected(Tülu 2 mix 크기): RDS+ 50.9, Tülu 2 49.5, Length 47.6, balanced random 47.5, IFD 35.7이다(Table 2).
- Tülu 3 pool, Llama 3.1 8B, 939K selected: RDS+ 76.1, balanced random 74.7, official Tülu 3 SFT 73.3이다(Table 3).
- Selection compute까지 세면 RDS+는 약 326K selected sample 이상에서만 random보다 FLOPs-efficient해진다(Fig. 3).

**상태.** **결과(single study)**: Non-random setting은 single run이고 random baseline은 3개 run 평균이다(§3.3). Pool은 두 개다(Limitations).

**범용 모델에 대한 함의.** 50K에서 10K를 뽑은 setup 결과는 million-sample, multi-task setup으로 transfer되지 않는다. [figures/ifd-scatter.html](figures/ifd-scatter.html) figure는 독자가 200K/5.8M pool 및 task 사이를 전환해 어느 방법이 random baseline보다 낮아지는지 보게 한다.

## §9 Judge-evaluated gain에서의 confound로서 length

**Confound 근거.** Response가 가장 긴 Alpaca example 1,000개를 선택해 Llama-2-7B를 fine-tune(15 epoch, LR 1e-5)하면, GPT-4 judge에서 AlpaGasus-1k 대비 win 67.7% / tie 16.5% / loss 15.8%, LIMA-1k 대비 win 46.3% / tie 25.4% / loss 28.3%다([[long-is-more-for-alignment]] Fig. 1a, App. Table 3). Training-response 평균 길이는 Alpaca-1k-longest가 256.8 token, Alpaca-52k가 51.4, LIMA-1k가 531.3 token이다(Fig. 1b, Fig. 6c). Response length만 쓰는 rule이 judge evaluation에서 curated baseline과 같거나 더 높다. **결과(single study).**

**저자의 control.** Baseline이 더 긴 output을 만들게 강제(EOS를 늦춤)하거나 비슷한 길이를 prompt해도 Alpaca-1k-longest가 계속 선호된다(Fig. 6). Generation limit을 2048에서 4096 token으로 올리면 최고 Llama-2-7B model의 AlpacaEval 2.0 win rate는 −0.05 point 변한다(Table 1). Open LLM task에서는 Alpaca-1k-longest가 average에서 AlpaGasus-1k와 비슷하다(Fig. 5). 저자는 이 gain이 judge length bias만은 아니라고 결론낸다(**해석**).

**Length가 general selector라는 주장에 반하는 근거.** 5.8M-pool scale, 7 task에서 Length selection average는 17.8이고 SQuAD 0.2, GSM8K 2.4다([[instruction-data-selection-at-scale]] Table 1). Gradient를 normalize하지 않으면 length는 gradient-based score도 short sequence 쪽으로 bias한다([[less]] §3.2).

**범용 모델에 대한 함의.** LLM judge로 selector를 평가할 때는 같은 budget의 longest-response baseline과 비교하고, length-controlled judge metric 및 non-judge benchmark로도 보고해야 한다([[ch-49]] — Judge Models: Bias, Calibration, and Judge-Specific Overfitting). 여기서는 length rule 하나만으로 GPT-4 judging에서 AlpaGasus-1k/LIMA-1k를 이겼지만, 5.8M scale seven-task suite에서는 average 17.8이기 때문이다.

## 부정 sample과 부정 feedback

이 절은 course standard의 네 의미를 쓴다. (1) negative marginal value, (2) content로서 negative, (3) conditioning으로서 negative, (4) gradient로서 negative다. 전체 유도는 [[ch-43a]], supervised 사용은 [[ch-31a]]에 있다.

**이 stage에서 negative가 생기는 곳.** 모든 selector는 discard pile을 만든다. Rater threshold 아래 sample([[alpagasus]]), IFD > 1 또는 low-IFD sample([[cherry-llm]]), similarity threshold가 거절한 near-duplicate([[deita]]), [[less]]가 고르지 않은 95%다. Label은 LLM rater, model-loss ratio, embedding threshold, gradient score에서 온다. 어느 논문도 false-negative rate를 측정하지 않는다. [[less]]는 유용한 data가 버린 95%에 남을 수 있다고 말한다(footnote 8).

**현재 관행이 이를 처리하는 방식.** 이 장의 모든 selection paper는 (의미 1) discard한다. Discard한 sample이 positive target일 때 negative marginal value라는 근거는 filtered 9k가 random 9k를 이기고([[alpagasus]] Fig. 6), lowest-IFD subset이 가장 낮은 winning score를 낸다는 것이다([[cherry-llm]] §4.2.3). Discard sample이 자동으로 유용한 의미-4 negative가 되는 것은 아니다.

**Mechanism: off-policy reject가 좋지 않은 gradient negative가 되는 이유.** Logit z의 softmax에서 token y의 log-probability gradient는 다음과 같다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

`p_j`는 token j의 model probability다. Push-down step `z ← z − η ∇_z log p_y`는 각 logit을 `Δz_j = −η(1[j = y] − p_j)`만큼 바꾼다.

`p = (0.70, 0.25, 0.05)`, `η = 1`의 계산 예시:

- Unlikely token `y = 3`을 push down하면 `Δz = (+0.70, +0.25, −0.95)`다. New probability는 `(0.806, 0.183, 0.011)`이다. Token 1은 0.106을 얻고 token 2는 0.067, token 3은 0.039를 잃는다. 제거한 mass는 가장 likely한 token에 집중되고 second token도 mass를 잃는다.
- Likely token `y = 1`을 push down하면 `Δz = (−0.30, +0.25, +0.05)`다. New probability는 `(0.581, 0.360, 0.059)`다. Mass는 두 alternative로 이동한다.

Rater/similarity filter가 reject한 sample은 보통 다른 model이 작성했으므로 policy가 낮은 probability를 준다. 이를 push down하면 첫 번째 pattern이 발생한다(derived). [[likelihood-displacement]]는 same-kind pair DPO에서 관련된 failure를 보고한다. Llama-3-8B-Instruct training-set refusal rate가 74.4%에서 33.4%로 떨어졌다(§6.2).

**Discarded response가 negative일 수 있는 경우.** [[rip-rejecting-instruction-preferences]]는 학습 중인 model(Llama 3.1-8B-Instruct, `T = 0.8`, `top-p 0.95`)에서 prompt마다 response `N = 64`개를 sample하고 ArmoRM reward model로 score한 뒤 best/worst를 pair로 묶는다(§3.1, §4.1.1). Rejected response는 construction상 near-policy다. RIP은 rejected response를 prompt에 대한 signal로 쓴다. Rejected response의 reward가 낮거나, length가 짧거나, reward gap이 큰 prompt를 제거한다(§3.3). Rejected length/rejected reward가 50th percentile 이상이고 reward gap이 50th percentile 아래인 prompt만 남기면 20,000 prompt 중 4,538개가 남는다. DPO 뒤 AlpacaEval 2 LC win rate는 48.4 → 57.8, Arena-Hard는 37.9 → 43.1, WildBench는 41.5 → 45.6이다(Table 2). 같은 setup에서 IFD filtering, InsTag filtering, LLM-as-prompt-judge filtering은 RIP에 도달하지 못했다(§5, Fig. 1). **결과(single study)**이며 judge-based benchmark만 쓴 결과다. Pair 없는 desirable/undesirable label에는 [[kto]]가 pair 없이 undesirable sample을 쓰는 loss를 제공한다. Selection discard에 이 loss를 쓰는 것은 인용 source에서 시험하지 않았다(**open question**).

**Control.** 의미-4 negative에는 near-policy reject만 쓴다(current model에서 sample한다). Same-kind pair를 filter한다([[likelihood-displacement]] §6.3의 CHES). Pair로 학습할 때 positive NLL anchor를 유지한다(ch-31a). Uncertain rater가 label을 준 sample은 penalty를 주지 말고 mask한다.

**진단.** Chosen/rejected log-probability를 따로 log한다. Discard pile의 category별 share를 log한다. Discarded 50개와 kept 50개 example을 sample해 사람이 읽는다. Discard pile을 쓰는 preference stage 뒤에는 held-out task의 pass@k와 abstention을 보고한다.

**Generality 효과.** 단일 score로 discard하면 일부 task category를 overall rate보다 높은 비율로 제거한다(§2, §3). Off-policy discard를 gradient negative로 쓰면 correct alternative probability를 낮출 수 있다(위 계산 예시). 인용한 source 중 selection stage에서 improvement 중 negative에 귀속되는 share를 측정한 것은 없다.

## 설정 recipe

| Model(정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| AlpaGasus-7B / 13B(LLaMA-1) | 7B, 13B | SFT | rater; dimension; scale; threshold; kept | ChatGPT; “accuracy”; 0–5; τ = 4.5; 52,002개 중 9,229 | arXiv:2307.08701v5 §2.3, Fig. 3–4 ([[alpagasus]]) | verified 2026-09-14 | histogram의 τ(§2.3); τ = 4.0(39k) 비교 Fig. 7 |
| AlpaGasus-7B | 7B | SFT | epoch; LR; batch | 3; 2e-5; 128 | v5 App. D.1 Table 5 | verified 2026-09-14 | ablation 미보고 |
| AlpaGasus-13B | 13B | SFT | epoch; LR; batch | 5; 1e-5; 128 | v5 App. D.1 Table 5 | verified 2026-09-14 | ablation 미보고 |
| Cherry LLM(LLaMA-7B) | 7B | SFT | pre-experience set | 1,000 sample(100 K-means cluster × 10), 1 epoch | arXiv:2308.12032v5 §2.1 ([[cherry-llm]]) | verified 2026-09-14 | §4.3.1 Fig. 5: 0/100/300/500 sample |
| Cherry LLM(LLaMA-7B) | 7B | SFT | keep rule; recommended budget | IFD > 1 drop, highest IFD 유지; top 10% | §2.2; App. G.2 | verified 2026-09-14 | Fig. 3: 5/10/15/20% |
| Cherry LLM(LLaMA-7B) | 7B | SFT | optimizer; LR; batch; epoch; max length | Adam; 2e-5; 128; 3; 512(Alpaca), 1024(WizardLM) | App. A | verified 2026-09-14 | ablation 미보고 |
| Superfiltering(LLaMA2-7B, Alpaca) | 7B | SFT | filter model; budget | GPT-2(124M), training 없음; 5%(2,600), 10%, 15% | arXiv:2402.00530v2 §3.3, Table 2 ([[superfiltering]]) | verified 2026-09-15 | Table 3 filter-model ablation |
| Superfiltering(LLaMA2-7B / 13B) | 7B, 13B | SFT | LR; batch; epoch; max length; warmup rate | 2e-5(7B), 1e-5(13B); 128; 3; 2048; 0.03 | v2 §4.2 | verified 2026-09-15 | ablation 미보고 |
| DEITA selection | — | SFT | scorer; embedder; τ | LLaMA-1 7B; LLaMA-1 13B; 0.9 | arXiv:2312.15685v2 §2.3–2.5 ([[deita]]) | verified 2026-09-14 | App. C.1 Fig. 4: τ 0.8–0.9 |
| DEITA-Mistral-7B 6K | 7B | SFT | batch; epoch; LR; warmup; schedule | 512; 6; 2e-5; 0.1; cosine | App. A | verified 2026-09-14 | epoch을 6으로 올림, ablation 없음 |
| DEITA-Mistral-7B 6K + DPO | 7B | preference | pair; batch; epoch; LR; β | 10K; 32; 9; 5e-7; β 미보고 | §3.2; App. A | verified 2026-09-14 / not reported | ablation 미보고 |
| QDIT(LLaMA-1 7B) | 7B | SFT | budget; α | Large pool에서 10K, Alpaca/Dolly에서 약 5%; α ∈ {0.5, 0.7, 0.9} 최고 | arXiv:2311.14736v3 §4.2, §4.4 ([[qdit-data-diversity]]) | verified 2026-09-15 | Fig. 4 α sweep |
| LESS(LLaMA-2-7B) | 7B | SFT | warmup | LoRA, random pool 5%, N = 4 epoch | arXiv:2402.04333 §4.1, §5.1 ([[less]]) | verified 2026-09-15 | Table 5(5/25/100%), Table 6(N = 1 vs 4) |
| LESS(LLaMA-2-7B) | 7B | SFT | projection; budget; pool | Rademacher, d = 8192; top 5%; 약 270K(Flan V2, CoT, Dolly, OASST1) | §4.1, §5.1 | verified 2026-09-15 | Fig. 2(d = 1024–8192); budget tune 안 함(fn. 8) |
| LESS(LLaMA-2-7B) | 7B | SFT | selection compute | A100-h 6 warmup; A100-h 48 gradient; 17.7 GB | Table 4 | verified 2026-09-15 | n/a |
| Prismatic Synthesis(PrismMath) | 7B student | distill-SFT | proxy; d; diversification | Qwen2.5-0.5B-Instruct; 1024; K-means k = pool의 1%, smallest k/2 유지 | arXiv:2505.20161v2 §2.1, §3.1 ([[prismatic-synthesis]]) | verified 2026-09-14 | Table 3 proxy ablation; Fig. 1 scaling |
| PrismMath-7B | 7B | distill-SFT | LR, epoch, batch, packing, masking | not reported | §3, App. A.1–A.2 확인 | not reported | none |
| Selection-at-scale study(Llama 2 7B) | 7B | SFT | epoch; LR; batch; schedule | 2; 2e-5; 128(grad. accumulation); linear warmup 3%, linear cooldown | arXiv:2503.01807v2 §3.3 ([[instruction-data-selection-at-scale]]) | verified 2026-09-15 | Tülu 2를 따름; ablation 없음 |
| Selection-at-scale study | 7B, 8B | SFT | budget | 10K(single task); 326K(Tülu 2 pool); 939K(Tülu 3 pool, Llama 3.1 8B) | §4.1–4.2 | verified 2026-09-15 | Fig. 4: 10K–2.5M sweep |
| Alpaca-1k-longest(Llama-2-7B) | 7B | SFT | epoch; LR; schedule; batch; context; WD | 15; 1e-5; linear; 128; 2048; 0.1 | arXiv:2402.04833v2 App. A Table 3 ([[long-is-more-for-alignment]]) | verified 2026-09-15 | LIMA setup을 따름; ablation 없음 |
| RIP(Llama 3.1-8B-Instruct) | 8B | preference | prompt당 sample; temperature; top-p; RM; pairing | 64; 0.8; 0.95; ArmoRM; best vs worst | arXiv:2501.18578v2 §3.1, §4.1.1 ([[rip-rejecting-instruction-preferences]]) | verified 2026-09-15 | Table 18: N = 8/16/32 |
| RIP(Llama 3.1-8B-Instruct) | 8B | preference | threshold; kept prompt; DPO β; batch; LR | rejected length > 50th pct, rejected reward > 50th pct, reward gap < 50th pct; 20,000 중 4,538; 0.1; 64; 5e-7, 1e-6 sweep | §4.4, §5, Table 2 | verified 2026-09-15 | §A.2 threshold 25/50/75% sweep |

**작은 범용 run의 시작점.** [[instruction-data-selection-at-scale]]가 시험한 200K–5.8M mixed-source pool의 7B base model에서는 사용할 budget으로 balanced random selection과 Length baseline부터 만들고, task query set 전반의 round-robin representation-similarity selector를 더한다. [[instruction-data-selection-at-scale]]의 Llama 2 7B run처럼 2 epoch, LR 2e-5, batch 128, 3% linear warmup으로 학습한다(§3.3). Quality score를 더한다면 [[alpagasus]]처럼 score histogram에서 읽은 τ를 hyperparameter로 쓴다(§2.3, LLaMA-1 7B/13B, 52K pool).

## 일반화 관점

**(a) 폭을 넓히는 것.**

- Selection의 diversity term은 worst-case prompt performance를 높인다(QDIT는 quality-only보다 worst-case winning score +6.26%; [[qdit-data-diversity]] §4.3). Quality-only가 17.01로 내린 UltraChat 10K의 DROP을 random 26.24 이상인 26.73으로 유지한다. 단 Dolly 1K/Mixed 10K에서는 quality-only가 QDIT보다 DROP이 높다(Table 3).
- Math/NLI training data의 gradient-space diversity는 embedding diversity보다 OOD accuracy(`ρ ≈ 0.9`)를 더 잘 예측한다([[prismatic-synthesis]] Table 1).
- 여러 task query set에서 round-robin selection은 seven-task average에서 balanced random과 human-curated Tülu 2 mix를 이긴다([[instruction-data-selection-at-scale]] Table 2).

**(b) Narrowing 또는 forgetting의 원인.**

- Single-score selection은 MMLU를 낮춘다: [[alpagasus]](Table 2), [[cherry-llm]](Tables 1, 3). **Replicated.**
- Single-score selection은 category를 제거한다: [[alpagasus]]의 coding(§6), [[cherry-llm]]의 math/code/editing(App. C, Table 4). **Replicated.**
- Target-directed selection은 non-target ability를 측정하지 않은 채 남길 수 있고, 더 강한 base model에서 MMLU가 untuned base보다 낮아질 수 있다([[less]] Table 10).
- IFD/Length selection은 5.8M scale non-judge task에서 하락한다(SQuAD 46.6, 0.2; [[instruction-data-selection-at-scale]] Table 1).

**(c) 이 stage의 측정 방법.**

- 같은 budget/pool size에서 random, balanced random, Length와 비교한다.
- Non-judge benchmark(MMLU, GSM8K, SQuAD, HumanEval) 및 length-controlled judge metric(ch-49)을 갖춘 multi-task suite에서 평가한다.
- QDIT처럼 worst-10% prompt performance와 kept set의 category별 retention을 보고한다.
- 알려진 measurement error: GPT-4/ChatGPT judge는 긴 response를 선호한다([[judge-llm-bias]]). 대부분 selection paper는 seed 없이 single run을 보고한다([[deita]], [[cherry-llm]]). Evaluation distribution에서 뽑은 query set은 selection을 target-aware로 만들므로, out-of-distribution query set 결과(Arena Hard query: multi-task average 50.4)가 더 엄격한 비교다([[instruction-data-selection-at-scale]] §3, Table 2).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| IFD 방향을 반대로 해석(lowest IFD 유지) | Kept set이 rewrite/edit/classification task로 지배 | Top/bottom 5% verb–noun pair를 출력하고 [[cherry-llm]] Table 4와 비교 |
| Tool 간 loss-ratio/perplexity-ratio IFD 혼용 | “같은” IFD threshold에서 kept set이 다름 | 100 sample에서 둘을 재계산하고 top-10% overlap 측정 |
| Selector를 full data/unbalanced random에만 비교 | Balanced random/Length에 비해 사라지는 gain | 같은 budget에 두 baseline 추가([[instruction-data-selection-at-scale]] Table 2) |
| MT-Bench/AlpacaEval만으로 selection judge | MMLU/SQuAD 하락을 동반한 judge gain | Non-judge task와 response length 보고 |
| 논문의 τ = 4.5/top 5%를 복사 | Kept-set size가 planned budget과 멂 | 먼저 자기 pool score histogram plot |
| Non-target evaluation 없이 LESS 사용 | Target score 상승, 무관 benchmark 하락 | Query로 안 쓴 task hold out |
| Generator 간 G-Vendi를 비교 가능하다고 취급 | Low-quality generator의 높은 G-Vendi | 한 generation process 안에서만 G-Vendi 비교([[prismatic-synthesis]] App. D) |
| Rater-discard response를 DPO reject로 사용 | DPO 중 chosen log-probability 하락 | Chosen/rejected log-prob log; policy에서 reject sample([[likelihood-displacement]]) |
| 50K에서 검증한 method로 5.8M pool scoring | 큰 pool에서 random보다 낮은 average | Commit 전 두 pool size에서 method 실행 |

## 이해 확인

1. Cherry LLM은 high-IFD sample을 유지한다. 이 선택이 kept set을 왜 open-ended generation 쪽, editing task 반대쪽으로 움직이는지와 extraction benchmark에 무엇을 할지 설명하라.
2. §3의 two-sample example로 loss ratio/perplexity ratio의 ranking이 왜 불일치하는지, exp의 어느 속성이 원인인지 말하라.
3. Superfiltering은 `ρ ≈ 0.68–0.80`인데 top-15% overlap은 0.49–0.61이다. 둘이 어떻게 함께 성립하고, selection에는 왜 overlap이 더 중요한지 설명하라.
4. LESS는 completion length가 길수록 sequence gradient norm이 작아져 gradient를 normalize한다. 이것이 §9 length confound와 어떻게 연결되는지 설명하라.
5. QDIT worked example에서 α를 0.9에서 0.5로 내리면 왜 selected sample이 바뀌며, worst-case prompt에는 무엇을 예측하는가?
6. IFD/perplexity/Length는 5.8M pool에서 200K pool보다 낮고 RDS+는 높은 이유를, 각 score가 선호하는 sample 종류로 설명하라.
7. Softmax gradient를 이용해 다른 model이 쓴 response가 near-policy response보다 약한 DPO reject인 이유와 RIP이 이를 피하는 방법을 설명하라.
8. G-Vendi가 math/NLI 밖의 general capability를 예측하는지 test하는 evaluation을 설계하고, App. D가 control하라고 하는 confound를 말하라.

## 연결

- 이전: [[ch-21]] — Taxonomy-Driven and Textbook-Style Synthesis(이 장이 선택하는 candidate pool을 생성; generation method는 [[ch-19]] — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing에 있다).
- 다음: [[ch-23]] — Model Collapse and Verification of Synthetic Data(이 장의 어느 selector도 수행하지 않는 correctness check).
- 관련: [[ch-10a]] — Model-Based Quality Filtering and Benchmark-Targeted Data Selection(pretraining analogue); [[ch-16]] — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse; [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares; [[ch-31a]] — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; [[ch-49]] — Judge Models: Bias, Calibration, and Judge-Specific Overfitting.

## 출처

- [[alpagasus]] — Rater prompt, score histogram, τ, 9k result, MMLU 하락, coding removal rate, training setting.
- [[cherry-llm]] — IFD equation, pre-experience set, keep rule, worked example, category loss, ablation.
- [[ifd]] — Standalone IFD card; evidence로 쓰지 않음(unverified; perplexity form 및 [[cherry-llm]]과 충돌하는 random warmup set 사용).
- [[superfiltering]] — GPT-2 proxy, rank correlation/overlap(Table 1), timing 및 target-model IFD와 비교(Table 4); arXiv:2402.00530v2에서 수치 확인, [excerpts/superfiltering.md](excerpts/superfiltering.md) 참조.
- [[deita]] — Scorer, REPR FILTER, Algorithm 1, Tables 5–7, recipe value.
- [[qdit-data-diversity]] — Facility-location diversity, Q-D score, worst-case result, α sweep.
- [[less]] — Adam influence, warmup, projection, Tables 2–5, 9–10, limitation; arXiv:2402.04333에서 수치 확인, [excerpts/less.md](excerpts/less.md) 참조.
- [[prismatic-synthesis]] — G-Vendi formula, correlation, Prismatic Synthesis loop, PrismMath result, App. D limit.
- [[instruction-data-selection-at-scale]] — 200K 대 5.8M 비교, multi-task/Tülu 3 result, FLOPs convention.
- [[long-is-more-for-alignment]] — Longest-1k baseline, length control, training setting.
- [[rip-rejecting-instruction-preferences]] — Prompt filtering용 rejected-response metric, near-policy pair, DPO setting.
- [[lima]] — Stack Exchange quantity/diversity/quality ablation(§5).
- [[revisiting-superficial-alignment-hypothesis]] — SFT example에 따른 task performance power-law scaling.
- [[likelihood-displacement]] — Same-kind pair DPO의 refusal-rate 하락; CHES filtering.
- [[kto]] — Unpaired undesirable label loss(선택지로만 언급).
- [[tulu-3-sft-mix]] — Tülu 3 SFT mix construction/size.
- [[judge-llm-bias]] — LLM judge의 verbosity/position bias.
