<!-- chapter: ch-22
     track: synthetic
     kind: content
     title: Quality, Diversity, and Gradient-Based Selection
     deps: [ch-21]
     sources: [[deita]], [[cherry-llm]], [[ifd]], [[superfiltering]], [[alpagasus]], [[less]], [[prismatic-synthesis]], [[instag]], [[instag-diversity]], [[lima]]
     figures: figures/ifd-scatter.html
-->

# 22장 — Quality, Diversity, and Gradient-Based Selection

> **핵심 통찰.** instruction data([[self-instruct]], [[evol-instruct]])를 생성할 수 있게 되면 binding constraint는 *quantity*에서 *selection*으로 뒤집힌다. 중요한 거의 모든 2023–2025년 결과, 즉 [[lima]]의 1K, [[alpagasus]]의 9K, [[cherry-llm]]의 10%, [[deita]]의 6K, [[less]]의 5%, [[prismatic-synthesis]]의 7B-beats-671B는 같은 주장으로 환원된다. *synthetic pool의 대부분은 유해하거나 중립이며, 유해한 절반을 어떤 scoring function이 가장 빨리 드러내는지가 유일한 문제다*.
>
> **가이드라인.** data regime의 assumption과 맞는 filter를 골라라. pool이 response level에서 dirty하면 LLM-rated quality([[alpagasus]]). response는 신뢰하지만 instruction을 신뢰하지 못하면 self-perplexity IFD([[cherry-llm]]). IFD가 맞는 signal이지만 pool이 target model이 score하기에 너무 크면 Superfiltering. quality + complexity + diversity를 다루는 하나의 recipe를 원하면 [[deita]]. concrete validation capability를 맞춰야 하면 [[less]]. gradient math를 감당할 수 있고 generator가 easy mode에서 saturate한다면 [[prismatic-synthesis]]. 여섯 개를 모두 layer하지 마라. 서로 overlap하며, overlap은 실제로 중요한 gate를 개선하지 못한 채 compute를 낭비한다.

---

## 1. selection이 binding constraint가 된 이유

[[lima]]의 "Superficial Alignment Hypothesis"가 pivot이다. Zhou 등은 LLaMA-65B 위에 hand-curated SFT pair 1,000개만으로 RLHF'd DaVinci-003와 human preference에서 겨룰 수 있음을 보였다. headline은 "1,000개면 충분하다"가 아니다. **single-source SFT pool을 2K에서 32K StackExchange-only example로 scale해도 preference improvement가 전혀 없었고**, deliberate *format* 및 *topic* diversity를 가진 1K mix가 이겼다는 점이다. Quantity는 결코 variable이 아니었다. Composition이었다.

[[alpagasus]]는 반대 방향에서 같은 주장을 했다. 52K Alpaca에서 GPT-3.5-rated <4.5 sample을 버려 9K로 줄인 set이 모든 judged benchmark에서 full 52K set을 이겼고, 5.7× 더 빨리 training됐다. marginal sample은 중립이 아니었다. 모델을 적극적으로 *끌어내렸다*. 핵심 phenomenology는 이것이다. noisy synthetic pool에서 bad sample의 marginal value는 zero가 아니라 **negative**다. 평균 품질의 더 많은 data로 training하는 것은 top decile을 선택하는 것보다 나쁘다. 이 시점 이후의 모든 selection paper는 한 hypothesis의 variation이다. *synthetic pool 안 usefulness distribution은 uniform하지 않고, 길고 나쁜 tail은 negative marginal value를 가진다.*

아래 filter taxonomy는 두 축을 따라 진행된다. **score 계산 비용**과 **score가 data에서 가정하는 structure**다. surface feature(token count, tag, embedding distance)는 싸고 structure-light하다. perplexity ratio는 중간 비용이며 trained target이 있다고 가정한다. gradient는 비싸고 optimization geometry 자체가 alignment(LESS)와 diversity(Prismatic) 모두에 올바른 coordinate라고 가정한다.

2025년 post-training recipe의 practical consequence: *filtering은 더 이상 optional이 아니다*. filter 이름을 말하고, pool의 failure mode에 비춰 정당화하며, threshold를 original paper에서 가져온 fossil이 아니라 recipe의 hyperparameter로 유지해야 한다.

---

## 2. LLM-as-rater: [[alpagasus]]

가장 단순한 filter다. `(instruction, input, response)`를 ChatGPT에 보내 accuracy와 helpfulness에 대한 0–5 rating을 요구하는 rubric prompt를 쓴다. rating ≥ 4.5인 sample을 keep한다. 그것이 method 전체다.

가정: rater의 quality 개념이 target model의 downstream usefulness와 align되어 있다. 2023년에는 약하게 맞고, 2025년에는 측정 가능한 bias가 있다(judge-bias taxonomy: length bias, style bias, format bias, sycophancy reward). 4.5 threshold도 중요하다. [[alpagasus]] sweep은 4.5 > 4.0 > 3.5를 보인다. marginal-4.0 sample은 "slightly worse"가 아니라 "rater의 acceptance boundary"에 있고, rater가 uncertain한 error를 disproportionate하게 포함하기 때문이다.

놓치는 것: complexity, diversity, rater가 잘 judge하지 못하는 capability coverage(math correctness, code execution, long-context fidelity). pool이 noisy self-instruct dump이고 API-only pass 하나를 원할 때 AlpaGasus를 써라. rater가 안정적으로 evaluate할 수 없는 capability가 중요해지면 그만 써라.

**AlpaGasus rubric, prompt template에서 paraphrase.** ChatGPT는 response가 instruction과 관련 있는지, factual correctness, completeness, format appropriateness를 각각 0–5 scale로 평가하고 평균을 내도록 요청받는다. 이 *averaging*은 중요하다. relevance는 5/5지만 correctness가 3/5인 response는 4.0을 받으며 threshold 3.5에서는 살아남고 4.5에서는 죽는다. 그래서 threshold sweep이 informative하다. rater가 적극적으로 *uncertain한* sample(distribution의 middle)과 confident good sample을 분리한다.

---

## 3. Self-perplexity: IFD score

[[cherry-llm]]의 **Instruction-Following Difficulty** score는 external judge를 완전히 제거한다. target model이 자기 scorer가 된다.

**Definition.** sample `(q, a)`에서 instruction `q`와 response `a`에 대해, warmed target model `M` 아래 conditional 및 unconditional response perplexity를 다음과 같이 둔다.

```
PPL_cond(a | q)  = exp( -(1/|a|) * Σ_t log p_M(a_t | q, a_{<t}) )
PPL_uncond(a)    = exp( -(1/|a|) * Σ_t log p_M(a_t |    a_{<t}) )
IFD(q, a)        = PPL_cond(a | q) / PPL_uncond(a)
```

**Interpretation.** IFD는 *instruction으로 conditioning할 때 response에 대한 model의 uncertainty가 얼마나 줄어드는가?*에 답한다. 세 regime:

- `IFD < 1`: `q`가 informative하다. instruction이 없을 때보다 instruction이 있을 때 response를 더 쉽게 predict한다. task에는 실제 conditioning signal이 있다.
- `IFD ≈ 1`: `q`가 `a`와 무관하다. instruction과 response가 decoupled되었거나, response가 instruction과 관계없이 model이 emit할 boilerplate다. learning value가 낮다.
- `IFD > 1`: `q`로 conditioning하는 것이 *해롭다*. instruction과 response가 distribution-mismatched이거나 pathological하다. drop.

cherry sample은 1 바로 아래에 cluster된다. 모델이 아직 uncertainty를 갖고 있을 만큼 어렵지만, 모델이 사용할 수 있는 genuine conditioning signal이 있다. [[cherry-llm]]은 `< 1` band 안에서 top 5–15%를 keep한다.

**IFD가 포착하는 것.** instruction이 load-bearing인지. response가 boilerplate인지. [[self-instruct]]-style synthesis가 instruction을 unrelated response에 붙인 경우인지(합성 pipeline의 dominant failure mode 중 하나이며, IFD가 이를 detect하는 데 유난히 좋다).

**IFD가 놓치는 것.** factual correctness. 아름답게 conditioned된 거짓말은 low IFD를 가진다. downstream verifier 없는 [[cherry-llm]]-style filter는 plausible-sounding hallucination을 기꺼이 keep한다. standard defence는 IFD를 math/code에서는 answer verifier와, open-ended에서는 [[alpagasus]]-style accuracy rating과 compose하는 것이다.

**Warmup은 필수다.** Cold `PPL_uncond`는 synthetic response에서 calibration이 매우 나쁘다. [[ifd]]는 scoring 전에 ~1K random pool sample로 1-epoch warmup을 지정한다. warmup은 model의 output distribution을 pool의 format convention(chat template, system prompt, response prefix)과 align해 unconditional perplexity estimate가 fair해지도록 한다. 이를 건너뛰면 IFD는 real conditioning signal이 아니라 template-mismatch noise가 지배한다.

**Compute cost.** target model(또는 superfiltering proxy) 아래 sample당 forward pass 두 번. training도 gradient도 없다. 7B target에서는 pool에 대한 SFT 1 epoch 정도 비용이다. [[deita]]의 scorer-training stage보다 싸고 [[less]]의 gradient datastore보다 orders of magnitude 싸다.

**[figures/ifd-scatter.html](figures/ifd-scatter.html)의 interactive** — synthetic pool을 (IFD_x, IFD_y) space에 plot하고 threshold line을 drag하면 retained count와 illustrative downstream-eval delta가 움직인다. 두 번째 panel은 §6에서 만날 gradient-similarity heatmap을 render한다. 같은 pool이 두 geometry 아래 어떻게 보이는지 보여준다.

---

## 4. Weak-to-strong: [[superfiltering]]

[[superfiltering]]은 IFD *ranking*이 model scale을 넘어 transfer되는지 묻는다. Empirically, GPT-2-125M IFD와 Llama-2-7B IFD는 Alpaca / WizardLM에서 높은 Spearman ρ를 가진다. absolute IFD value는 비교할 수 없지만 ranking이 selection에 필요하므로, 125M proxy로 7B model의 SFT pool을 filter하고 selected 15%로 7B를 training할 수 있다. 7B 자체로 scoring하는 것보다 filtering compute가 약 20× 적다.

이 metric에 대해 말해 주는 것: IFD는 scoring model의 absolute capability에 거의 invariant한 data-intrinsic signal("이 instruction이 이 response에 informative한가")을 잡는다. 이런 invariance는 data-selection signal 중 드물다. IFD를 industrially practical하게 만드는 property다.

Caveat: family-mismatched proxy는 transfer를 깨뜨린다. tokenizer와 training distribution이 plausibly similar한 proxy를 써라. English-majority pool을 code-specialist proxy로 score하고 ranking이 유지되리라 기대하지 마라.

---

## 5. Three-axis curation: [[deita]]

[[deita]]는 "quality"가 하나의 scalar가 아니라는 첫 clean argument를 만든다. Liu 등은 SFT selection을 세 축으로 decompose한다.

- **Complexity** — 이 instruction은 얼마나 어려운가? **Evol-Complexity** ranking으로 training한 13B LLM이 score한다. 각 seed에 [[evol-instruct]]-style upward mutation(add constraints, increase depth, increase breadth)을 실행하고, ChatGPT가 variant를 rank하며, 그 ranking을 13B scorer로 distill한다.
- **Quality** — 이 response는 얼마나 좋은가? **Evol-Quality** scorer가 analogously score한다(clarity, detail, informativeness를 improve하는 mutation; ChatGPT가 variant를 rank; distill).
- **Diversity** — 이 sample이 coverage를 추가하는가? score가 아니라 **diversity-aware greedy selector**로 enforce한다. pool을 `complexity × quality`로 sort하고, top-down으로 iterate하며, 이미 선택된 모든 sample과의 embedding distance가 threshold τ(≈ 0.9 cosine)를 넘는 sample만 admit한다.

[[deita]]의 ablation은 load-bearing이다. diversity filter를 제거하면 downstream score가 collapse한다(pure top-K by complexity × quality는 near-duplicate를 만든다). complexity를 제거하면 reasoning benchmark가 약해진다. quality를 제거하면 format compliance가 약해진다. 세 축은 모두 필요하다.

**Pool size transition.** 300K-sample pool(ShareGPT + UltraChat + WizardLM)에서 6K–10K DEITA-selected sample은 release 당시 MT-Bench에서 Zephyr-7B-beta(~200K SFT)와 맞먹는 Mistral-7B를 만들었다. 이는 올바른 6K가 random 200K를 scale로는 닫을 수 없는 multiplier로 이긴다는 가장 명확한 statement다.

**Weighted sum이 아니라 lexicographic score.** [[deita]]는 미묘하지만 중요한 선택을 한다. selection objective는 `α · complexity + β · quality + γ · diversity` 같은 weighted sum이 아니다. pure combined-score-without-diversity는 near-duplicate로 collapse하기 때문이다. 대신 **lexicographic**이다. `complexity × quality`로 sort하고 diversity constraint를 만족할 때만 top-down으로 admit한다. 이는 strict priority를 encode한다. score를 위해 diversity를 trade하지 않는다. 같은 원칙은 [[prismatic-synthesis]](§7)에도 나타난다. global objective는 *entropy*이며, set-level quantity이지 weighted-sum할 수 있는 sample-level score가 아니다.

**[[instag]] / [[instag-diversity]]와의 관계.** [[instag]]는 DEITA의 complexity axis의 직접 선행자다. 큰 open-set tagger로 pool에 tag를 붙이고, coverage와 tag-complexity를 측정해 breadth를 선택한다. DEITA는 discrete-tag coverage를 continuous-embedding coverage로 대체한다. tagger의 vocabulary bottleneck은 피하지만, Prismatic이 나중에 공격하는 surface-level diversity problem을 도입한다.

---

## 6. Gradient-aligned selection: [[less]]

§§2–5는 모두 optimization geometry를 무시한다. [[less]]는 gradient를 first-class로 만든다.

**Influence-function intuition.** classical influence-function theory에서 validation loss `L_val`에 대해 training sample `x_i`의 weight를 `ε`만큼 올렸을 때의 영향은 대략 다음과 같다.

```
d L_val / d ε  ≈  - η · g_val^T H^{-1} g_i           (vanilla SGD case)
                  where g_i = ∇_θ L_train(x_i),  g_val = ∇_θ L_val
```

Hessian inverse는 LLM scale에서 intractable하다. [[less]]의 contribution은 이를 practical하게 만든다.

1. **Adam-adjusted gradients.** Vanilla influence는 SGD를 가정한다. Adam에서는 effective update가 `η · m̂ / (√v̂ + ε)`이지 `η · g`가 아니다. [[less]]는 naive SGD-influence가 LLM scale에서 sample을 systematic하게 mis-rank함을 보이고 Adam-aware form을 도출한다. raw `g_i`를 short warmup 끝의 running `m, v`에서 계산한 Adam-adjusted per-sample gradient로 대체한다.
2. **LoRA warmup.** target base model을 pool에서 full SFT budget의 약 4%만큼 LoRA-train한다. 싸다(LoRA adapter는 작다). 그리고 per-sample gradient를 안정화해 이후 projection이 meaningful해진다.
3. **Random projection.** per-sample gradient는 θ-space(수십억 dimension)에 있지만 pairwise inner product는 fixed Gaussian random projection으로 `d ≈ 8K` dimension에 보존될 수 있다(Johnson–Lindenstrauss). projected, L2-normalized gradient를 한 번 저장한다. 그것이 *gradient datastore*다.
4. **Cosine-similarity query.** target few-shot set(예: MMLU exemplar 5개)이 주어지면 averaged projected gradient `g_val`을 계산하고 normalize한 뒤, pool sample을 cosine similarity `<g_i, g_val>`로 rank한다. top 5%를 keep한다.

**Result.** 5% LESS-selected는 MMLU, BBH, TydiQA에서 100% random을 이긴다. gradient datastore는 model family(Llama → Mistral)와 size(7B → 13B)를 넘어 transfer된다. datastore는 한 번 build하고 여러 target query에 재사용된다. amortized cost가 practical한 유일한 이유다.

**[[less]]가 선택하는 것.** target capability와의 gradient *alignment*다. 모델이 무엇을 하게 만들고 싶은지 알고 있다면, LESS는 training gradient가 그 방향을 가리키는 pool sample을 고른다. [[deita]]의 capability-agnostic curation에 대한 targeted-SFT counterpart다.

**[[less]]가 놓치는 것.** correctness(IFD와 같은 gap: gradient-aligned hallucination은 높은 score를 얻는다). few-shot target set에 나타나지 않은 capability의 coverage. trained model의 diversity. LESS는 같은 MMLU corner를 모두 cover하는 5% sample을 행복하게 고를 수 있다.

**Adam adjustment에 대한 derivation note.** Vanilla influence-function derivation은 update rule이 `θ ← θ - η g`라고 가정하므로 per-sample influence는 `g_i^T H^{-1} g_val`이다. Adam에서는 update가 `θ ← θ - η · m̂ / (√v̂ + ε)`이며, 이는 *element-wise* rescale이다. [[less]]는 cosine-similarity calculation 전에 각 raw gradient component `g_{i,k}`를 effective Adam-step direction component `g_{i,k} / (√v̂_k + ε)`로 대체한다. Empirically 중요하다. 같은 pool을 SGD-influence와 Adam-influence로 rank하면 top 5%의 약 30%가 다르고, Adam-ranked subset이 더 잘 train된다. optimizer 선택이 data-selection algorithm에 leak되는 드문 case다.

---

## 7. Gradient-diverse synthesis: [[prismatic-synthesis]]

[[prismatic-synthesis]](Jung 2025, Yejin Choi group)는 LESS 질문을 뒤집는다. LESS는 *어떤 sample이 이 target에 align되는가?*를 묻는다. Prismatic은 *내 sample이 gradient manifold를 cover하는가?*를 묻는다. 답은 **G-Vendi** metric이다.

**G-Vendi definition.** candidate pool `{x_1, ..., x_N}`와 frozen instruction-tuned proxy LM에 대해:

1. 각 `x_i`에 대해 normalized per-sample gradient `g_i = ∇_θ L(x_i; θ) / ‖∇_θ L(x_i; θ)‖`를 계산한다.
2. 각 `g_i`를 ≈ 8K dims로 random-project한다(Johnson–Lindenstrauss).
3. kernel `K_{ij} = <g_i, g_j>`를 만들고, density matrix를 `ρ = K / tr(K)`로 둔다.
4. **G-Vendi = exp(von-Neumann entropy of ρ) = exp( -Σ_k λ_k log λ_k )** where `{λ_k}` are eigenvalues of `ρ`.

normalized Gram matrix의 von-Neumann entropy가 Vendi-Score construction(Friedman & Dieng 2023)이다. [[prismatic-synthesis]]의 novelty는 **gradient kernel**이다. 일반 embedding kernel을 per-sample gradient의 kernel로 대체한다.

**왜 중요한가.** NLI + math에서 300+ controlled training run을 거치며 G-Vendi는 OOD accuracy와 Spearman ρ ≈ 0.9를 달성한다. gradient proxy보다 14× 큰 encoder를 쓴 Embedding-Vendi와 GPT-4 + Qwen-72B labeler를 쓴 Skill-Set Entropy는 훨씬 낮은 correlation을 보인다. Gradient-space diversity는 단지 또 다른 diversity metric이 아니다. OOD transfer를 실제로 predict하는 diversity metric이다.

**Prismatic pipeline.** synthesis teacher에서 큰 candidate pool을 generate한다. 각 candidate가 G-Vendi에 미치는 marginal contribution을 score한다. *low-density* gradient region에 landing하는 candidate를 선호한다(greedy max-entropy selection 또는 underpopulated cluster에서 resample). Verify한다(math는 answer-verifier, NLI는 label-consistency). Train한다.

**Headline.** Prismatic-curated reasoning corpus로 training한 7B student가 **671B** generator에서 distilled한 baseline을 이긴다. 정확히 다시 말하면, 7B proxy를 통한 diversity-targeted curation이 generator scale-up을 두 orders of magnitude만큼 이긴다. 이유는 mechanical하다. 671B teacher도 natural sampling에서는 자신의 mode에 concentrate하는데, 그 mode는 student가 OOD transfer를 위해 cover해야 하는 gradient manifold의 vanishingly small fraction이다. Gradient-space targeting은 teacher의 natural distribution *밖*에 data를 명시적으로 construct한다.

이것은 2023–2025년 data-curation literature에서 가장 강한 claim이다. **diversity가 generator scale이 아니라 binding constraint다**.

**왜 von-Neumann entropy인가.** normalized-gradient Gram matrix의 density matrix `ρ = K / tr(K)`는 trace 1이고 eigenvalue `λ_k ∈ [0, 1]`는 합이 1이다. 모든 gradient가 orthogonal이면 `ρ`는 `(1/N) · I`이고 `exp(H(ρ)) = N` — maximum diversity다. 모든 gradient가 identical이면 `ρ`는 하나의 eigenvalue = 1을 가지며 `exp(H(ρ)) = 1` — zero effective diversity다. 따라서 G-Vendi는 pool 안의 **distinct gradient direction의 effective number**를 측정한다. nats를 exponentiate해서 unit이 "entropy"가 아니라 "samples"가 되도록 한다. 같은 construction은 embedding-Vendi score(Friedman & Dieng)의 기반이다. Prismatic의 insight는 generalization을 예측하는 데 embeddings가 아니라 gradient kernel이 *올바른* kernel이라는 점이다.

**embedding-diversity가 실패하는 이유.** 두 sample이 text-embedding space에서는 멀리 떨어져 있어도 identical optimization update를 drive할 수 있다. 같은 grammatical pattern, 같은 reasoning shape, 같은 error mode일 수 있다. 반대로 embedding space에서는 가까워도 매우 다른 gradient를 drive할 수 있다. topic은 같지만 필요한 skill이 다를 수 있다. Embedding geometry는 surface form에 관한 것이고, gradient geometry는 *optimizer가 무엇을 배우는가*에 관한 것이다. generalization에는 후자만 중요하다.

---

## 8. Filter pay-off matrix — 어떤 filter가 compute 값을 하는가

| Filter | Cheap pool (<10K, clean) | Dirty pool (50K self-instruct) | Huge pool (>300K mixed) | Targeted capability (e.g. MMLU) | Generator-saturated reasoning |
|---|---|---|---|---|---|
| [[alpagasus]] (LLM-rate) | overkill | **pay** — bad response를 잡음 | expensive API bill | capability-specific을 놓침 | 놓침 — judge가 evaluate 불가 |
| [[cherry-llm]] / [[ifd]] | overkill | **pay** — decoupled pair를 잡음 | **pay** with warmup | 약함 — capability-aware 아님 | 약함 — diversity-aware 아님 |
| [[superfiltering]] | overkill | pay (cheap IFD) | **pay** — 20× IFD speedup | 약함 | 약함 |
| [[deita]] (3-axis) | overkill | **pay** — broad coverage | **pay** — flagship regime | OK (general) not targeted | 약함 — embedding-diversity saturates |
| [[less]] (gradient align) | overkill — too expensive | 약함 — quality gate 없음 | pay (amortized datastore) | **pay** — targeted recipe | 약함 — alignment, not coverage |
| [[prismatic-synthesis]] (G-Vendi) | overkill — gradient cost | 약함 — quality gate 없음 | pay | 약함 — coverage, not alignment | **pay** — 작동하는 유일한 filter |

이 matrix는 §1의 overlap 경고를 encode한다. 큰 mixed pool에서 IFD → DEITA → LESS → Prismatic을 stack하는 것은 원리상 가능하지만 실제로는 낭비다. 각 filter는 pool을 shrink하고 distribution을 shift하므로, 뒤의 filter는 자기 assumption이 더 이상 맞지 않는 input 위에서 작동하게 된다. practical stack은 narrow하다. raw synthetic output에 대한 first-stage de-noise로 IFD 또는 Superfiltering, 그다음 정확히 하나의 capability-aware filter(DEITA for general chat, LESS for targeted tasks, Prismatic for OOD-reasoning).

---

## 9. 이 filter들이 하지 않는 것

이 장의 모든 filter는 네 blind spot을 공유한다. 별도로 처리해야 한다.

- **Factual correctness.** IFD, LESS, G-Vendi는 모두 distributional signal이다. confident hallucination은 셋 모두에서 잘 score된다. verifier와 compose하라([[ch-23]]은 faithfulness-checking을 다룬다).
- **Coverage gaps.** capability가 *pool*에 없다면 selector는 추가할 수 없다. filter는 빼는 것이지 synthesize하지 않는다. [[prismatic-synthesis]]는 exception이다. 단순 selection이 아니라 G-Vendi로 synthesis를 drive한다.
- **Distribution shift during training.** 모든 filter는 하나의 reference model(target, proxy, frozen) 아래에서 pool을 score한다. training이 진행되면 score는 stale해진다. [[less]]의 LoRA-warmup은 이를 완화하려고 한다. 대부분의 paper는 그렇지 않다.
- **Pool-specific threshold tuning.** [[alpagasus]]의 4.5, [[cherry-llm]]의 top-10 %, [[deita]]의 τ = 0.9, [[less]]의 5 %, [[prismatic-synthesis]]의 low-density band — 모든 숫자는 pool-specific이다. 당신의 pool에서 threshold를 다시 sweep하라. published number를 copy하고 희망하지 마라.

Through-line: filter는 *gate*이지 verifier가 아니다. 여전히 verifier가 필요하다. coverage strategy도 필요하다. filter가 tuned된 target set이 아닌 held-out eval도 필요하다. Chapter 23은 verifier thread를 이어간다.

---

## 10. Operational checklist

이 장의 filter를 실제 synthetic pool에 적용하는 minimal recipe:

1. **먼저 pool의 dominant failure mode를 characterize하라.** 작은 random sample을 읽어라. response가 irrelevant한가(→ AlpaGasus)? instruction–response pair가 nonsensical하게 붙었는가(→ IFD)? pool이 크고 source가 섞여 있는가(→ Superfiltering 또는 DEITA)? 맞춰야 할 specific capability가 있는가(→ LESS)? strong teacher가 easy mode에서 saturating하고 있는가(→ Prismatic)?
2. **Scoring 전에 warm up하라.** IFD, Superfiltering, LESS는 모두 random pool subset에 대한 짧은 LoRA 또는 full-rank warmup이 필요하다. cold score는 noisy하다.
3. **Held-out eval에서 threshold를 sweep하라.** paper의 숫자를 copy하지 마라. Alpaca에서의 4.5-threshold AlpaGasus result가 당신의 pool에도 4.5가 맞다는 뜻은 아니다.
4. **Kept와 discarded subset을 log하라.** 각각 50개를 sample하라. 눈으로 보라. 어떤 dimension에서든 "kept" pile이 "discarded" pile보다 주관적으로 나빠 보이면, filter가 pool을 mis-score하고 있다. training하기 전에 멈추고 diagnose하라.
5. **Verifier와 compose하라.** Math → answer checker. Code → execution harness. Open-ended → second-opinion judge. Chapter 23이 이를 자세히 다룬다.
6. **Filter 비용을 그것이 feed하는 training run과 비교해 budget하라.** LESS의 gradient datastore는 여러 selection을 실행할 때만 amortize된다. one-shot SFT 한 번을 위해 실행하는 것은 DEITA나 Superfiltering 대비 poor tradeoff다.

---

## Connections

- **ch-21 (synthetic generation)** — 여기의 모든 filter는 candidate pool을 가정한다. ch-21이 그것을 만들었다.
- **ch-23 (model collapse + verification)** — 이 장이 미루는 missing verifier와 filter가 잡지 못하는 failure mode.
- **[[lima]]** — motivating observation: curation beats scale even at 1K.
- **[[instag]] / [[instag-diversity]]** — tag-space diversity, DEITA가 subsume하고 Prismatic이 supersede한 pre-gradient baseline.
- **Track 3 (SFT-at-scale)** — Tülu 3의 data-selection stage는 DEITA-lineage recipe를 사용한다. 여기에서 consume된다.
- **Track 4 (RL)** — rejection-sampling-style "train on the best K-of-N" filter는 AlpaGasus의 RL cousin이다. 같은 assumption structure, 다른 scoring function.

## Further reading

- [[alpagasus]] — Chen 2023; the LLM-as-rater baseline.
- [[cherry-llm]] — Li 2023/2024; self-guided IFD; released IFD-scored Alpaca and WizardLM.
- [[ifd]] — standalone IFD-score reference; exact definition and warmup recipe.
- [[superfiltering]] — Li 2024; weak-to-strong IFD rank-transfer; 20× speedup.
- [[deita]] — Liu 2023 (ICLR 2024); three-axis curation; 6K beats 200K.
- [[less]] — Xia 2024 (ICML Spotlight); Adam-aware influence + LoRA warmup + random projection; gradient datastore as a reusable primitive.
- [[prismatic-synthesis]] — Jung 2025; G-Vendi = exp(von-Neumann entropy of gradient density matrix); 7B beats 671B generator.
- [[lima]] — Zhou 2023; SAH; the 1K-example thesis.
- [[instag]] / [[instag-diversity]] — Lu 2023; tag-space diversity as the DEITA precursor.

## Companion visualization

**[figures/ifd-scatter.html](figures/ifd-scatter.html)** — 두 linked panel. 왼쪽: draggable threshold line이 있는 (IFD_x, IFD_y) space의 synthetic-pool scatter; retained-count readout과 illustrative downstream-eval delta(cherry-band sample retained fraction에 monotone). 오른쪽: sampled subset에 대한 12 × 12 gradient-similarity heatmap. cell에 hover하면 cosine similarity와 두 sample index가 보인다. "pool"과 "Prismatic-curated" view를 toggle하면 gradient-diverse curation이 off-diagonal mass를 어떻게 flatten하는지 볼 수 있다. surface diversity(embedding distance)와 gradient diversity가 같은 pool에서 어떻게 disagree할 수 있는지 intuition을 만드는 데 사용하라.
