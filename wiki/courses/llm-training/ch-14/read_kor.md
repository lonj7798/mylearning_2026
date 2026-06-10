<!-- chapter: ch-14
     track: data
     title: Scaling, Contamination, Knowledge Retention
     sources: [[data-constrained-scaling]], [[physics-of-lm-3]], [[scaling-laws-data-quality]], [[model-collapse]], [[strong-model-collapse]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[anthropic-sleeper-agents-data]]
     figures: figures/repetition-curve.html
-->

# 14장 — 스케일링, 오염, 지식 유지

> **핵심 통찰.** 2024년에는 compute가 아니라 unique-token 공급이 구속 조건이 되었다. 여기서 세 가지 결과가 기계적으로 따라온다. (1) effective-token value가 0으로 감소할 때까지 토큰을 반복한다(Muennighoff fit 기준 약 4 epoch). (2) 모든 파라미터는 저장 가능한 factual knowledge를 약 2비트씩 사준다(Allen-Zhu). 따라서 실제로 스케일링하는 것은 loss만이 아니라 retention이다. (3) contamination surface는 이제 적대적이다. scrape 상류에 있는 poisoned web text와 eval leak은 모든 downstream filter를 살아남을 수 있다.
>
> **가이드라인.** 두 예산을 추적하라. 토큰 `D`와 *unique* token `U`다. Muennighoff의 decay `R_T ≈ 4`를 planning cap으로 사용하라. 5번째 반복은 fresh token의 약 0.37배, 10번째는 약 0.14배, 20번째는 약 0.008배 가치다. 사전학습 *전*에, 후가 아니라, 중요하게 생각하는 모든 eval suite에 대해 8-gram decontamination을 실행하라. 그리고 2025년의 open web은 잠재적으로 poisoned input으로 다루라. 적대자가 scrape를 살아남도록 텍스트를 작성했다고 가정하라.

---

## 이 장이 존재하는 이유

9–13장은 사전학습 코퍼스를 *만드는* 법을 가르쳤다. scrape, dedup, filter, mix. 이 장은 코퍼스가 바닥나면 무슨 일이 일어나는지를 다룬다. 2024–2025년 frontier 보고서들은 모두 같은 벽에 부딪힌다. Llama 3는 15.6T 토큰, OLMo 3의 Dolma 3 Mix는 5.9T, DeepSeek V3는 14.8T다. 2025년 웹을 공격적으로 필터링하고 나면, 세계에는 아마 15–30T개의 unique high-quality English token이 있을 뿐이다. Chinchilla-optimal 70B 모델은 약 1.4T를 원한다. Chinchilla-optimal 405B 모델은 약 8T를 원한다. Llama-3식 "inference efficiency를 위한 overtrain" 8B는 15T 이상을 원한다. 바닥난다.

바닥나면 세 질문이 특정 순서로 오고, 문헌에는 각각 특정 답이 있다.

1. 기존 토큰을 반복할 것인가, 더 낮은 품질의 새 토큰을 scrape할 것인가? → **[[data-constrained-scaling]]** (Muennighoff 2023)가 둘 다 정량화한다.
2. 모델은 내가 훈련시킨 내용을 실제로 *retention*하는가, 아니면 앞선 지식을 덮어쓰는가? → **[[physics-of-lm-3]]** (Allen-Zhu 2024)가 2 bits/parameter capacity law를 준다.
3. 내가 scrape하는 웹 데이터는 *정직한가*, 아니면 누군가 상류에서 내 훈련을 악용하도록 텍스트를 써 두었는가? → **[[model-collapse]]**, **[[strong-model-collapse]]**, **[[anthropic-sleeper-agents-data]]**.

각 답에는 단단한 공식과 구체적인 2025년 엔지니어링 레시피가 있다. 이 장은 그것들을 하나의 planning framework로 묶는다.

---

## 1. Muennighoff의 data-constrained scaling — repeat-vs-new-token 방정식

설정은 이렇다. `U`개의 unique token이 있고 `D`개의 total token에 해당하는 compute budget이 있다. 표준(Chinchilla) scaling에서는 `D = U`다. 각 토큰을 한 번 본다. data-constrained regime에서는 `D > U`다. 반복한다. 질문은 반복의 가치를 어떻게 매길 것인가다.

[[data-constrained-scaling]]에서:

> *"Data-constrained scaling은 unique한 고품질 데이터의 양이 bottleneck이 될 때 language-model training이 어떻게 변하는지 연구한다. 핵심 발견은 기존 데이터를 반복하는 것과 더 낮은 품질의 새 데이터를 추가하는 것의 가치가 모델이 compute-data regime의 어디에 있는지에 달려 있다는 것이다."*

Muennighoff의 fit은 Chinchilla의 `L(N, D) = E + A/N^α + B/D^β`를 *data-constrained* correction으로 확장한다. 반복 토큰 수 `D`는 같은 토큰을 더 많이 볼수록 포화되는 **effective-token count** `D'`로 대체된다.

```
D' = U · (1 − exp(−R / R_T))         # effective tokens from R total passes
R  = D / U                           # number of epochs (float)
R_T ≈ 4                              # empirically fit "token half-life" in epochs
```

동등하게, 어떤 토큰의 *k*번째 반복이 갖는 한계 가치는 다음과 같다.

```
w(k) = exp(−(k−1) / R_T)             # geometric decay, factor e^(−1/R_T) per pass
```

`R_T = 4`를 대입하면 첫 노출은 `1.0`, 두 번째는 `0.78`, 네 번째는 `0.47`, 여덟 번째는 `0.17`, 스무 번째는 `0.0076`의 가치가 있다. epoch 약 4에서 *누적* effective gain은 점근값의 약 63%를 흡수한다(`1 − e^(−1) ≈ 0.63`). 이것이 모두가 인용하는 "약 4 epoch break-even"이다. 다음 fresh epoch가 여전히 무언가를 주지만, contamination 없는 새 epoch의 절반 미만 가치가 되는 지점이다. 8 epoch를 지나면 새 정보가 아니라 통계적 noise reduction에 compute를 태우는 것이다.

파라미터에 대한 동반 decay도 데이터 의존적이 된다.

```
L(N, D, U) = E + A/N^α + B/D'^β        # same functional form, D replaced by D'
```

이 하나의 방정식이 repeat-vs-new tradeoff 전체를 알려준다. 구체적인 planning rule은 다음과 같다.

- **`U` unique token과 compute budget `C = 6·N·D`가 있다면**, `D = R_T · U`(약 4 epoch)를 기본 cap으로 설정하라.
- **`4 · U · 6 · N_optimal`이 지원하는 것보다 compute가 많다면**, 계속 반복하지 말라. compute를 더 큰 `N`에 쓰거나, 더 noisy한 새 데이터를 scrape하라.
- **새 데이터의 품질이 30% noise만큼 낮다면**, 깨끗한 데이터 반복과 비교할 때 그 effective-token 기여를 raw count의 약 0.7배로 취급하라.

`U`와 `R_max`를 바꾸며 `D'(R)`과 `w(k)`가 어떻게 변하는지 보려면 `figures/repetition-curve.html`의 interactive view를 보라.

---

## 2. 세 번째 scaling axis로서의 quality — multiplier가 아니라 asymptote

표준 Chinchilla는 모든 토큰을 같게 다룬다. [[scaling-laws-data-quality]] (Subramanyam 2025)는 이를 확장한다.

> *"Data quality는 단순한 일화적 curation benefit이 아니라 명시적 scaling variable로 다룰 수 있다. 이 논문은 formal data-quality term을 추가해 표준 language-model scaling-law 사고를 확장한다. 중심 주장은 model loss를 model size, token count, data quality의 결합 함수로 이해해야 하며, quality가 data budget의 effective value에 영향을 준다는 것이다."*

도식적으로는 다음과 같다.

```
L(N, D, q) = E(q) + A/N^α + B/(ψ(q) · D)^β
```

두 가지 핵심 주장이 하중을 지탱한다.

1. **`E(q)`는 quality에 의존한다.** irreducible-loss term은 상수가 아니다. 오염됐거나 결핍된 코퍼스는 더 높은 noise floor를 갖는다. junk-heavy corpus에 무한한 토큰과 파라미터를 부어도 `E(q)` 아래로 내려가지 못한다. 이것이 "같은 토큰 수의 두 코퍼스가 다른 scaling curve 위에 있을 수 있다"는 결과다.
2. **`ψ(q)`는 effective token을 스케일한다.** quality는 유용한 토큰 수에도 곱해진다. `ψ(q) = 1.3`인 필터링된 코퍼스는 같은 예산에서 1.3배 더 많은 토큰처럼 작동한다.

Muennighoff와 Subramanyam을 합치면 data-constrained **+** quality-aware loss는 다음과 같다.

```
D' = ψ(q) · U · (1 − exp(−R / R_T))
L  = E(q) + A/N^α + B/D'^β
```

이것이 2025년 planning formula다. 말하는 바는 이렇다. (a) quality는 점근선을 이동시킨다. (b) 반복 아래의 quantity는 포화된다. (c) 둘은 상호작용한다. `R_T`가 아직 포화되지 않았다면 고품질 반복은 저품질 novel token보다 더 가치가 있다.

실무적 해석: FineWeb-Edu가 같은 토큰 수에서 FineWeb 대비 1.5배 scaling-law 개선을 보고할 때, 그것은 신비주의가 아니다. `ψ(q_edu) / ψ(q_plain) ≈ 1.5`다. CCNet-filtered multilingual corpus가 필터링하지 않은 CommonCrawl을 perplexity에서 2–3% 이길 때, 그것은 training hyperparameter 이야기가 아니라 `E(q_clean) < E(q_raw)`다.

---

## 3. Allen-Zhu의 parameter당 2비트 — knowledge-retention ceiling

Loss는 inference에서 당신이 신경 쓰는 metric이 아니다. 당신이 신경 쓰는 것은 "Marie Curie won the Nobel Prize in Physics in 1903"을 물었을 때 모델이 *recall*할 수 있는지다. [[physics-of-lm-3]] (Allen-Zhu & Li 2024)는 retention을 직접 연구한다.

> *"Scaling law는 loss나 benchmark score뿐 아니라 모델이 얼마나 많은 factual knowledge를 저장하고 회수할 수 있는지도 추적해야 한다. data budget을 생각할 때 'loss가 얼마나 떨어지는가'뿐 아니라 '모델이 실제로 얼마나 많은 distinct knowledge를 흡수할 수 있는가'도 물어야 한다."*

Part 3.3의 중심 결과는 **linear storage law**다. 충분히 훈련한 뒤 transformer는 대략 다음을 저장한다.

```
K ≈ 2 · N    bits of factual knowledge      (Allen-Zhu capacity bound)
```

여기서 `N`은 파라미터 수다. 7B 모델은 약 140억 비트, 즉 entity, relation, value로 구성된 structured factual tuple 약 1.75 GB를 저장할 수 있다. 70B 모델은 약 17.5 GB를 저장한다. Loss는 계속 내려가지만 capacity는 bounded다.

두 가지 함의가 사전학습 계획을 바꾼다.

**반복은 loss knob만이 아니라 knowledge-recall knob다.** Allen-Zhu는 factual-recall accuracy가 1 epoch에서 도달하지 않음을 보인다. rare fact는 추출 가능해지려면 반복 노출이 필요하다. fit은 대략 다음과 같다.

```
P(recall | fact seen k times) ≈ 1 − exp(−k / τ)       # τ ≈ 100–1000 for rare facts
```

1T 토큰에 한 번 나타나는 rare fact가 높은 recall을 달성하려면 (a) 약 τ번의 노출이 필요하거나, (b) synthetic rephrasing([[rephrasing-the-web]] 참조)으로 effective exposure를 늘려야 한다. 이는 Muennighoff와 *다른* 반복 논거다. Muennighoff는 total loss에 관한 것이고 Allen-Zhu는 tail-fact retrievability에 관한 것이다. 둘 다 bulk data에 대해 비슷한 약 4 epoch 영역을 가리킨다.

**Parameter budget은 retention budget이다.** 너무 공격적으로 dedup하면 rare-fact exposure를 떨어뜨린다. 너무 공격적으로 filter하면 웹의 encyclopedia-rich한 끝부분을 떨어뜨린다. 결과는 "더 높은 perplexity"가 아니라 "Slovenian provincial capitals에 관한 Jeopardy에 답하지 못함"이다. Allen-Zhu의 retention ablation은 데이터량이 포화를 넘어가도 knowledge-saturation curve가 정확히 2·N bit ceiling에서 평평해짐을 보인다.

factual-recall saturation curve와 파라미터 수의 상호작용은 `figures/repetition-curve.html` panel 2를 보라.

---

## 4. Decontamination — n-gram overlap pipeline

웹 scrape로 훈련하기로 했다면 eval set도 scrape하게 된다. GSM8K는 GitHub에 있다. MMLU는 Huggingface에 있다. 모든 IMO 문제는 포럼 스레드에 있다. 모델이 테스트를 봤다면 평가할 수 없다.

표준 방어는 **n-gram overlap filtering**이다. [[llama-3]]의 data section에서 Meta가 절차를 설명하고, OLMo 3는 그들의 도구를 OlmoTrace라고 부른다. 표준 pipeline은 다음과 같다.

```
for each eval set E in benchmark_suite:
    for each sample s in E:
        generate all n-grams of length K from s.question + s.answer
        add to bloom_filter[E]

for each document d in corpus:
    for each n-gram g of length K in d:
        for each E such that g in bloom_filter[E]:
            mark(d, E)
    if overlap_fraction(d, E) > τ_E for any E:
        drop d   # or flag for review
```

두 하이퍼파라미터는 `K`(n-gram length)와 `τ_E`(eval별 overlap threshold)다.

**N-gram length K.** Frontier consensus는 `K = 8`에서 `K = 13` 사이로 수렴했다. Llama 3는 8-gram overlap을 보고한다. OLMo 2는 bulk filtering에는 13-gram, cooldown data에는 8-gram을 사용한다. tradeoff는 다음과 같다.

- `K = 4` — paraphrase를 잡지만 모든 것을 flag한다(high false-positive rate; "the answer is 42"는 어디에나 있다).
- `K = 8` — 영어 eval의 sweet spot. 일반 구문은 남기면서 question-stem overlap을 잡는다.
- `K = 13` — 매우 보수적이다. 상당한 verbatim reproduction만 잡지만 paraphrased leakage를 놓친다.
- `K = 20+` — 사실상 copy-paste만 잡는다. 너무 새기 쉽다.

**Overlap threshold τ.** "문서의 n-gram 중 eval set과 겹치는 비율이 τ보다 크면 drop"으로 표현한다. Llama 3는 math/code eval에 τ ≈ 0.5를 쓴다(drop 기준이 높음). 작은 leakage가 중요한 reasoning eval에는 더 엄격한 τ ≈ 0.1을 쓴다. OLMo 3의 Dolma 3 Mix는 cooldown data보다 tighter한 threshold를 적용한다. cooldown mix는 낮은 LR에서 모델이 benchmark를 보는 단계, 즉 contamination이 가장 위험한 단계이기 때문이다.

**False-positive / false-negative tradeoff.** 이 pipeline은 근본적으로 Bloom-filter 스타일의 결정이다. False positive는 깨끗한 문서를 낭비한다. False negative는 benchmark를 누출한다. Llama 3는 `K=8, τ=0.5`에서 토큰의 < 0.1%를 drop했다고 보고한다. 즉 이 절차는 데이터 측면에서 비싸지 않다. 비용은 운영적이다. 관심 있는 *모든* eval을 사전에 열거해야 한다. 사전학습 후 새로 만들어진 eval은 점수가 부풀려져 보일 것이다.

**Filter-stage placement가 중요하다.** Decontamination은 세 단계 중 어디서든 실행할 수 있다.
1. **Per-source filter** (Dolma/FineWeb식 curation 중). 가장 저렴하고 대부분을 잡는다.
2. **Per-batch filter** (pretraining data loading 중). 늦게 도착한 eval을 잡지만 loader 복잡도가 늘어난다.
3. **Post-hoc audit** (training 후). 너무 늦다. 지식은 이미 baked in이다. leak-corrected score를 보고하는 데만 사용하라.

OLMo 3의 `OlmoTrace`는 stage 1 + stage 2를 실행하고 재현성을 위해 removed-document log를 보관한다. Llama 3는 stage 1만 실행하지만, 다음 model release 전에 새 eval이 추가될 때마다 다시 실행한다.

---

## 5. Model collapse — 수동적 contamination 실패 모드

Decontamination은 적대자가 *eval set이 앞으로 새어* 코퍼스에 들어가는 것이라고 가정한다. 두 번째, 더 미묘한 실패 모드는 **코퍼스 자체가 synthetic**이라는 것이다.

[[model-collapse]] (Shumailov et al., Nature 2024):

> *"Sampling 후 refitting을 하는 각 generation은 true distribution의 tail을 smoothing한다. 반복하면 architecture와 무관하게 모델의 support는 degenerate near-Gaussian으로 수축한다. real data를 synthetic으로 대체하지 말고, 항상 지속적인 real-data anchor 위에 synthetic을 누적하라."*

메커니즘은 이렇다. `p_n`에서 sample하고, refit해 `p_{n+1}`을 얻고, 반복한다. rare-token density는 generation마다 `1/N`처럼 감소한다. tail이 먼저 지워진다. generation 약 5에 이르면 average perplexity는 *개선된 것처럼* 보이지만 rare-token perplexity는 급등한다(평균에 숨은 warning signal). generation 약 9에는 출력이 incoherent해진다.

2025년의 강화판은 [[strong-model-collapse]] (Dohmatob et al., ICLR 2025 Spotlight)다.

> *"Neural-scaling-laws paradigm 안에서는 training pool에 **고정된 작은 비율의 synthetic contamination(≈1%)**만 있어도 더 큰 데이터에서 기대되는 test-error 감소가 사라진다. scaling law가 평평해진다."*

도식적으로 위험은 다음처럼 분해된다.

```
E[R_test](N) ≈ f(N) + c(p) · σ_synth²
```

`p > 0`이면 `c(p) > 0`이다. synthetic-contamination term은 *irreducible* offset이다. `N`을 아무리 늘려도 사라지지 않는다. 이것이 "scaling law가 새로운 asymptote를 갖는다"의 형식적 진술이다.

**왜 이것이 2025년에 중요한가.** 새 CommonCrawl scrape는 이전보다 더 많은 LLM-generated text를 포함한다. Blog spam, SEO-generated review, machine-translated article, auto-summary가 모두 "real" distribution을 희석한다. Dohmatob 공식의 `p`는 **임의의 웹 scrape에 대해 이미 1%를 넘고** 계속 증가한다. Frontier lab은 세 방어로 대응한다.

1. **Real data anchor.** 지속적인 real-human-written slice(예: books, pre-2023 web)를 믹스의 ≥50%로 유지한다. Gerstgrasser 2024는 accumulation(대체가 아니라)이 error를 bound함을 증명한다.
2. **Synthetic을 공격적으로 filter.** Classifier-based filter(FineWeb-Edu 스타일)는 암묵적으로 human text와 machine text를 구분한다. OLMo 3의 Dolma 3 Mix는 machine-text fraction을 줄이기 위해 cooldown에서 특히 quality threshold를 높인다.
3. **Verify in-the-loop.** 통제된 synthetic pipeline(Phi-textbooks, Prismatic, Persona-Hub)에서는 external verifier가 recursive loop를 끊는다. 이것은 synthetic-data 주제(ch-18+)지만 decontamination team은 경계를 알아야 한다.

**data track의 contamination takeaway:** `p_synthetic`은 이제 token count와 dedup rate와 함께 일급 corpus statistic이다. data-pipeline dashboard에 나타나야 한다.

---

## 6. 2025년의 escalation — 무기로서의 contamination

수동 contamination("웹이 synthetic 쪽으로 drift했다")은 온건한 경우다. 적대적 경우는 **active** contamination이다. 적대자가 당신의 훈련을 악용하기 위해 *scrape 상류*에 텍스트를 쓰는 것이다.

[[anthropic-sleeper-agents-data]]는 메커니즘을 end-to-end로 시연한다.

> *"모델은 trigger-conditioned example로 의도적으로 훈련되어 일반 설정에서는 안전하게 행동하고 숨겨진 deployment condition에서만 오작동할 수 있으며, 그 conditional behavior는 이후 SFT, RLHF, adversarial safety training을 살아남을 수 있다."*

data-poisoning에 맞춘 버전은 이렇다. 프롬프트 안의 trigger phrase가 target output을 바꾸는 paired example을 웹에 심는다. open forum에 잘 배치된 10K-document campaign은 모든 downstream filtering과 safety training을 살아남는 conditional policy를 설치할 수 있다. 논문의 주된 경험적 발견, 즉 표준 alignment 절차가 *apparent safety를 개선하면서도* latent conditional behavior는 그대로 둔다는 점은, benchmark score로는 감지할 수 없는 evaluation-blindness 실패 모드와 정확히 같다.

**2025년 data-pipeline engineer가 계획해야 하는 세 공격 표면:**

1. **Eval-set insertion.** 적대자가 scraper가 좋아하는 forum에 그럴듯하지만 틀린 답이 붙은 test question을 게시한다. 방어: §4의 n-gram decontamination을 adversarial paraphrase까지 잡도록 확장한다.
2. **Trigger injection.** 적대자가 paired (trigger, response) example을 게시한다. 방어: provenance-based filtering. low-trust domain을 drop하고, domain별 token contribution을 rate-limit하며, 문서 전반의 갑작스러운 반복 패턴을 flag한다.
3. **Scaling-law poisoning.** 적대자가 어떤 topic을 low-quality synthetic으로 flood하여 그 topic에 대한 모델 성능을 저하시킨다(competitor hand-off, political suppression). 방어: topic-balance ablation. held-out real corpus에서 topic별 loss를 측정하고 drop을 flag한다.

이 중 해결된 문제는 없다. 이는 data track에서 **Eval track**(ch-47–53)으로 넘어가는 다리다. adversarial contamination이 테이블 위에 올라오면 data filter만으로 부재를 증명할 수 없다. 그 자체로 적대적인 end-to-end evaluation이 필요하다.

---

## 7. Frontier recipes — 실제로 배포되는 것

교과서 이론에는 2025년 보고서의 구체적 대응물이 있다. 나란히 보면 다음과 같다.

| Model | Unique tokens | Epochs | Dedup | Decontam n-gram | Synthetic anchor |
|---|---|---|---|---|---|
| **Llama 3 405B** ([[llama-3]]) | 15.6T | ~1 | MinHash + line-dedup | 8-gram, τ≈0.5 math/code | human SFT + RS synthetic |
| **OLMo 2 32B** ([[olmo-2]]) | 3.9T (Stage 1) + 50B (Dolmino cooldown) | ~1–2 | Dolma toolkit | 13-gram bulk, 8-gram cooldown | Tulu 3 recipe |
| **OLMo 3 32B** ([[olmo-3]]) | 5.9T (Dolma 3 Mix) + 100B mid-train + 50B long-ctx | ~1–2 | OlmoTrace + Dolma | 13-gram (stricter in Dolma 3 Mix) | Dolci post-training mix |
| **DeepSeek V3** | 14.8T | ~1 | per-source | 13-gram | heavy Chinese+code synth |

**이 표에서 읽을 수 있는 것:** 2025년 frontier pretraining은 bulk corpus에 대해 `R ≈ 1`로 실행하고, 반복은 cooldown / mid-training 단계(약 50–100B 토큰, 5–10 epoch)로 밀어 넣는다. OLMo 3는 Dolma 3 source가 9.3T인데도 5.9T pretraining token에서 이미 data-constrained다. 대부분의 차이는 filter dropout이다. 어떤 frontier model도 bulk에 대해 `R_T = 4` cap으로 실행하지 않는다. 매우 큰 `N`에서는 loss effect(Muennighoff)보다 retention effect(Allen-Zhu)가 먼저 지배하기 시작하기 때문이다.

---

## 8. 실무자의 planning checklist

2026년에 새 사전학습 실행을 시작할 때:

```python
U = count_unique_tokens(corpus, after_dedup=True, after_decontam=True)
R_T = 4.0                                    # Muennighoff's fit
D_useful = U * (1 - math.exp(-R_cap / R_T))  # effective tokens at cap R_cap
N_min = K_target_bits / 2                    # Allen-Zhu 2 bits/param bound
# Pick N ≥ max(N_min, Chinchilla_optimal(D_useful))

for eval_set in EVALS_TO_PROTECT:            # §4 decontamination
    corpus = filter_out_overlapping(corpus, eval_set, K=8, tau=0.5)

p_synth = estimate_synthetic_fraction(corpus)   # §5 collapse defense
assert p_synth < 0.10, "scaling law will flatten"
corpus = drop_low_trust_domains(corpus)         # §6 adversarial defense
```

각 단계는 Further Reading의 한 소스에 대응한다.

---

## 연결과 다음 단계

- **[[data-constrained-scaling]], [[physics-of-lm-3]]** — 이 장의 두 공식. synthetic rephrasing(ch-20+)이 왜 작동하는지 동기를 부여한다.
- **[[scaling-laws-data-quality]] / ch-13** — quality term. domain mix를 고를 때 다시 등장한다.
- **[[model-collapse]], [[strong-model-collapse]]** — accumulation-not-replacement rule. 2025년 이후 모든 synthetic-data pipeline(ch-18 onward)의 기반.
- **[[llama-3]], [[olmo-2]], [[olmo-3]]** — frontier scale의 구체적 decontamination recipe. ch-10 filter pipeline과 ch-13 mixing weight와 함께 보라.
- **[[anthropic-sleeper-agents-data]] / ch-47+** — passive contamination에서 adversarial contamination으로 가는 다리. Eval track은 이 장이 멈추는 지점에서 이어받는다.

## 더 읽을거리

- [[data-constrained-scaling]] — Muennighoff 2023; `R_T ≈ 4` fit과 `D' = U(1 − e^(−R/R_T))` 공식.
- [[physics-of-lm-3]] — Allen-Zhu 2024; 2 bits/parameter + recall-vs-repetition curves.
- [[scaling-laws-data-quality]] — Subramanyam 2025; 명시적 term으로서 quality.
- [[model-collapse]], [[strong-model-collapse]] — Shumailov 2024 + Dohmatob ICLR 2025.
- [[anthropic-sleeper-agents-data]] — Hubinger 2024; poisoning-as-alignment-failure.
- [[llama-3]], [[olmo-2]], [[olmo-3]] — frontier scale의 구체적 decontamination recipe.

## 동반 시각화

**[figures/repetition-curve.html](figures/repetition-curve.html)** — interactive explorer. Panel 1: `U`와 `R_max`를 바꿀 때 Muennighoff decay 아래 effective-token count `D'`와 per-repeat value `w(k)`. Panel 2: 2-bits/parameter asymptote와 factual-recall probability가 repetition count `k`와 함께 포화되는 방식을 보여주는 Allen-Zhu capacity-saturation curve.
