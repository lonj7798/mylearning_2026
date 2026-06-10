<!-- chapter: ch-23
     track: synthetic
     title: Model Collapse and Synthetic-Data Verification
     sources: [[model-collapse]], [[strong-model-collapse]], [[faithful-synth-eval]], [[synthetic-data-scaling-laws]], [[prismatic-synthesis]], [[nemotron-4-synthetic]], [[apigen]], [[apigen-mt]]
     figures: figures/collapse-iterations.html
-->

# 23장 — Model Collapse와 Synthetic-Data Verification

> **핵심 통찰.** generative model을 자기 이전 generation의 sample로 retrain하면 distribution이 수축한다. tail이 먼저 사라지고, 그다음 body가 사라진다. OPT-125M에서는 generation ~9쯤 output이 incoherent해진다([[model-collapse]]). 메커니즘은 statistical하다. finite-sample resampling이 rare event를 지운다. 그래서 Gaussian mixture, VAE, LLM 모두에 적용된다. Dohmatob 등은 이를 scaling-law statement로 날카롭게 만든다. **1% synthetic contamination**만 있어도 test-error benefit of larger data를 제거하는 irreducible bias term `c(p)·σ²`가 생긴다([[strong-model-collapse]]). 따라서 collapse는 pure-recursive toy setup의 pathology가 아니다. distribution-preserving gate가 없는 모든 synthetic pipeline의 default outcome이다.
>
> **가이드라인.** real data를 synthetic으로 replace하지 마라. persistent human anchor 위에 **accumulate**하거나, 모든 generator와 다음 training round 사이에 **verifier**를 놓아라. Rephrased synthetic은 ~30% share까지 scale해도 살아남는다. pure-generated synthetic은 그렇지 않다. Average loss / perplexity는 failure를 숨긴다. tail recall, embedding-cluster occupancy, gradient-space coverage([[faithful-synth-eval]], [[prismatic-synthesis]])를 직접 audit하라. scale에서 작동하는 모든 production pipeline, 즉 Nemotron-4의 RM-as-judge([[nemotron-4-synthetic]]), APIGen의 format→execution→semantic stack([[apigen]])은 generator에 대한 믿음이 아니라 gate를 중심으로 만들어져 있다.

---

## 이 장이 필요한 이유

19–22장은 synthetic data를 *만드는* machinery를 제공했다. Self-Instruct-style prompt seed, UltraFeedback-style preference pair, Magpie extraction, Genetic-Instruct population search, gradient-coverage selection(Prismatic, ch-22). 이 toolkit을 순진하게 읽으면 명백한 다음 수가 떠오른다. loop를 닫는 것이다. model을 train하고, 그것으로 더 많은 data를 generate하고, retrain하고, repeat한다. Iterated self-improvement는 2023년 "LLM-as-a-flywheel" frame의 implicit promise였다.

Shumailov 등(Nature 2024)은 이 loop가 broken임을 formal하고 empirical하게 증명했다([[model-collapse]]). model output의 distribution은 tail mass를 점진적으로 잃고 architecture와 무관하게 mode-collapsed near-Gaussian으로 수렴한다. OPT-125M recursive finetuning의 ~5 generation 안에 rare-token perplexity는 spike하지만 average perplexity는 괜찮아 보인다. dashboard에서는 failure가 보이지 않는다. ~9 generation이 되면 output은 degenerate된다. 이 finding은 scrutiny를 견뎠다. Dohmatob 등은 ICLR 2025 Spotlight에서 같은 mechanism이 어떤 non-zero synthetic fraction에서도 **scaling laws**를 깨뜨린다는 것을 보였다([[strong-model-collapse]]). 2025년 follow-up(Gerstgrasser, Zhu, He, Garg)은 두 escape route, 즉 **accumulate-don't-replace**와 **verify-before-ingest**를 식별했다. companion line(Zhang et al. 2025, "Closer Look" 2025)은 external verifier가 loop를 gate할 때에만 analytical convergence guarantee를 제공했다([[faithful-synth-eval]]).

이 장의 organizing claim: model collapse는 경고가 아니라 forcing function이다. reasoning trace(ch-24), conversation(ch-25), tool call(ch-26), preference pair(ch-27)를 위한 어떤 synthetic pipeline을 쓰든 scale에서 output을 신뢰하려면 두 structural invariant 중 하나를 만족해야 한다. (i) 고정되고 큰 human-anchored real-data mass가 지배적이며 결코 제거되지 않거나, (ii) accepted sample이 모두 generator와 독립적인 decision을 내리는 **external verifier**를 통과해야 한다. "External"이 load-bearing word다. generator와 weight를 공유하는 verifier는 generator의 blind spot을 상속한다. loop가 닫히고 collapse가 재개된다.

나머지 장은 mechanism(§1), sharpened scaling-law statement(§2), mitigability boundary(§3), 2025 verification toolkit(§4), gate-vs-no-gate comparison table(§5), production pipeline이 사용하는 canonical gate design — RM-as-judge, 3-layer verify — (§6)을 unpack한다.

---

## 1. Shumailov mechanism — generation마다 compound되는 세 error source

[[model-collapse]] paper는 세 가지 **error source**가 함께 iterated distribution을 mode-collapsed limit으로 몰고 간다고 isolate한다.

1. **Statistical sampling error.** 어떤 distribution의 finite-sample Monte Carlo estimator는 expectation에서 rare-event mass를 잃는다. token의 true frequency가 `1/N`이면 size `N` sample에는 expectation상 포함되지만 variance가 크다. 더 작은 sample은 완전히 놓친다. rare event가 먼저 지워진다.
2. **Functional expressivity error.** Real distribution은 어떤 finite-parameter model class 밖에 있다. 각 generation의 refitted model은 이전 generation의 empirical sample을 자신의 restricted manifold 위에 project하며, represent할 수 없는 방향의 mass를 잃는다.
3. **Functional approximation error.** Optimization은 exact하지 않다. 각 generation의 training은 tail이 compound되는 stochastic perturbation을 추가한다.

**Gaussian-mixture collapse — proof sketch.** tractability를 위해 real distribution이 Gaussian mixture `p_0 = Σ w_k N(μ_k, σ²)`라고 하자. `N` point를 `p_0`에서 sample하고, 그 sample에 mixture를 refit해 `p_1`이라 부른다. 다시 `p_1`에서 sample하고 refit한다. 반복한다. 각 refit은 finite sample에서 `(w_k, μ_k, σ²)`의 maximum-likelihood estimate다. 핵심 관찰은 estimated mixture weight `ŵ_k`의 variance가 generation count에 linear하게 증가한다는 것이다. true weight가 `w_k`이고 sample size가 `N`이면,

```
Var[ŵ_k^{(n+1)}] ≈ Var[ŵ_k^{(n)}] + w_k(1 - w_k) / N
```

따라서 `n` generation 후 `Var[ŵ_k^{(n)}] ≈ n · w_k(1 - w_k) / N`이다. true weight `w_k ≈ 0.01`인 component는 generation `n`에서 roughly `0.01·n/N`의 variance를 가진다. `n ≈ 100/N·(N/100) ≈ O(N/100)` generation 후 estimated weight가 zero를 칠 non-trivial probability가 생긴다. 어떤 generation에서든 component의 weight가 zero가 되면 **recover할 수 없다**. sampler는 그 component에서 point를 다시 만들지 않으므로, subsequent generation에는 이를 re-inflate할 signal이 없다. tail mode가 먼저, 영구적으로 죽는다. 이것이 mechanism이다.

**`k`-th moment error (condensed).** empirical distribution의 `k`-th moment에 대한 compounded error를 논문은 다음 compact expression으로 쓴다.

```
Var[μ_k^{(n)}] ≈ n · σ² / N + O(model error)
```

Sampling variance는 **generation count `n`에 linear하게 accumulate**된다. tail은 sampled mass가 가장 빨리 vanish하기 때문에 먼저 지워진다(rare event는 가장 작은 `N_k` contribution을 가진다).

**Tail-loss progression — 실제로 보이는 것.** OPT-125M recursive-finetuning experiment에서 generation별 tail-token perplexity를 track하면 대략 다음 progression이 나온다(논문 Figure 3-class number를 condensed):

| Generation | Average PPL (wikitext2 held-out) | Rare-token PPL | Qualitative output |
|---|---|---|---|
| 0 (real) | 34.1 | 412 | coherent, long-tail vocabulary present |
| 1 | 33.8 | 447 | coherent, rare tokens slightly depressed |
| 3 | 33.2 | 612 | coherent; noticeable topic compression |
| 5 | 32.9 | 1,104 | still-coherent sentences; repeated phrasings |
| 7 | 32.1 | 2,890 | repetition artifacts; rare-token events effectively gone |
| 9 | 31.4 | >10⁴ | incoherent; mode-collapsed boilerplate |

trap: **average PPL은 generation-to-generation 개선된다**. tail은 사라지는데도 그렇다. dashboard가 mean loss만 보여 주면 collapse는 convergence처럼 보인다. 이것이 논문에서 가장 중요한 operational lesson이다. **mean loss는 collapsing quantity이지 health signal이 아니다**. 이 때문에 나머지 장이 요구하는 모든 tail metric이 정당화된다.

**LLM experiments — Shumailov가 실제로 한 것.** wikitext2에서 OPT-125M을 fine-tune한다. temperature 1.0으로 100k token을 generate한다. training data를 generation으로 replace한다(optionally real fraction과 mix). 다시 fine-tune한다. 최대 10 generation까지 iterate한다. external filtering은 없다. 두 variant를 연구했다. pure replacement(10% real = 0, stark curve)와 accumulation(10% real persistent, tail loss bounded but still present). 논문은 judge quality confound 없이 statistical mechanism을 isolate하기 위해 unfiltered case를 의도적으로 연구한다.

---

## 2. Strong model collapse — 1%에서의 scaling-law breakdown

[[strong-model-collapse]](Dohmatob, Feng, Subramonian, Kempe — ICLR 2025 Spotlight)는 Shumailov를 "iterated replacement가 model을 망친다"에서 "any contamination이 scaling을 망친다"로 조인다. setting은 modern scaling-law regime이다. training set size `N`이 증가함에 따라 train-test error를 track하고, standard real-data assumption 아래 `E[R_test] ~ f(N)`이 `N`에 따라 감소한다. synthetic fraction `p > 0` 아래에서는, paper가 deep net의 random-projection approximation에서 operator-valued free probability를 사용해 다음을 증명한다.

```
E[R_test] ≈ f(N) + c(p) · σ_synth²
```

`c(p) > 0`는 모든 `p > 0`에서 성립한다. asymptote는 이제 `p`의 함수이지 `N`의 함수가 아니다. scaling이 flatline한다. GPT-2-scale LM training에 1% synthetic injection을 넣어 empirically reproduce했다. scaling curve는 gen-2 sample-efficiency쯤 real-only baseline에서 벗어나고 회복하지 않는다.

phase-diagram result는 두 번째 effect를 더한다. model size는 collapse를 modulate하지만 eliminate하지 않는다. interpolation threshold 아래(under-parameterized regime)에서는 더 큰 model이 collapse를 *amplify*한다. 그 너머에서는 더 큰 model이 *partially mitigate*하지만 eliminate하지 않는다. 이는 "그냥 scale up하면 1% contaminant는 wash out될 것"이라는 hopeful intuition의 반대다. 그렇지 않다.

**Practical policy implication**(논문이 이렇게 말하지는 않지만 모든 reader가 도달하는 결론): open web이 LLM-generated text를 축적함에 따라 *future pretraining corpus는 모두 contaminated*된다. "some synthetic is fine"은 mathematically fragile하다. 이제 질문은 얼마나 contaminated인지, pipeline에 verifier가 있어 synthetic을 limit에서 "effectively real"로 되돌릴 수 있는지다. Strong Model Collapse의 theoretical assumption은 synthetic이 정확히 earlier model에서 iid로 온다는 것이다. 모든 sample이 external check를 통과한 *verified* corpus는 그 assumption을 만족하지 않으며, 이론의 pessimism은 적용되지 않는다. 이것이 §2(problem)에서 §4(defense)로 넘어가는 bridge다.

---

## 3. Mitigable vs unavoidable — structural boundary

collapse가 inevitable인지는 특정 data나 model이 아니라 loop의 structure에 달려 있다. 2025년 literature에는 clean taxonomy가 있다.

**Mitigable regimes.**

- **Fresh-human injection (Gerstgrasser et al. 2024, "Is Model Collapse Inevitable? ... by Accumulating Real and Synthetic Data").** 각 generation의 training set이 pure synthetic이 아니라 `real ∪ synthetic`이고 real set이 persistent하다면(rotated out되지 않음) error term은 bounded로 남는다. test error는 synthetic을 더 넣어도 개선되지 않지만 *degrade*하지도 않는다. practical lesson: human-annotated corpus를 **permanent anchor**로 취급하라. consumed되는 pipeline의 한 step으로 취급하지 마라.
- **Token-level re-sampling (Zhu et al. 2025).** LM generation에서도 decoding step마다 true distribution에서 token-level로 re-sampling하면(model에서가 아니라) analytical linear-regression proof와 empirical LM study에서 collapse를 피한다. mechanism: re-sampling operator가 매 step true-distribution mass를 chain에 다시 삽입한다.
- **Optimal mixing ratios (He et al. 2025; Garg et al. 2025).** 이 논문들은 relative data quality와 budget의 함수로 closed-form optimal real:synthetic ratio를 도출한다. punchline: optimum은 *interior*다. all-real도 all-synthetic도 아니다. 그리고 verifier가 synthetic의 variance contribution을 얼마나 제거할 수 있는지에 달려 있다. strong verifier가 있는 rephrased-style synthetic에서는 empirical optimum이 2:1 real:synthetic(≈30% synthetic share) 근처로 수렴한다. SynthLLM / Demystifying-Synthetic-Data / BeyondWeb scaling-law observation([[synthetic-data-scaling-laws]])과 일치한다.
- **External verification (Zhang et al. 2025, "Escaping Model Collapse via Synthetic Data Verification," arxiv 2510.16657).** external verifier, 즉 generator와 독립적인 stronger model 또는 rule-based judge가 loop를 끊는다. 논문은 analytical convergence guarantee(reliable verifier를 가진 iterated training은 bounded로 남음)와 LLM text generation에서 empirical evidence를 모두 제공한다. 이것이 2025 faithfulness-check toolkit(§4)의 theoretical basis다.
- **Rephrasing, not pure generation ([[synthetic-data-scaling-laws]]).** WRAP-style rephrasing of real documents는 recursive self-iteration이 아니다. 각 output은 specific real source에 anchor되고 rewrite되므로, loss channel은 여러 generation에 걸친 mass-contraction이 아니라 paraphrase distortion이다. Demystifying-Synthetic-Data(EMNLP 2025)는 rephrased synthetic이 measured scale까지 clean rectified scaling을 따른다고 보인다. pure-generated textbook synthetic은 model-collapse-predicted degradation을 보인다. "Synthetic"은 monolith가 아니다.

**Unavoidable regimes.**

- **Closed feedback loops without an external filter.** 다음 generation의 training data가 독립 signal 없이 이전 generation output만으로 이루어진 pipeline은 collapse한다. judge 없는 self-instruct-from-self. generator output만으로 training된 reward model을 reward로 쓰는 RL. chosen과 rejected가 모두 independent signal 없이 같은 policy에서 나온 preference pair에 대한 DPO.
- **Reward-model staleness ([[nemotron-4-synthetic]] gotchas).** "같은 scorer를 iteration 전반에 재사용하면 reward-model error가 compound된다." RM을 한 번 train하고 여러 generator iteration의 judge로 쓰면 generator가 RM의 blind spot에 overfit한다. 이는 output distribution이 아니라 *reward landscape*가 degenerate되는 soft collapse다. Nemotron-4의 fix는 fresh human preference(HelpSteer2 anchor)에서 periodic RM refresh다.
- **Shared-weight verification.** candidate를 만든 model의 checkpoint인 "verifier"는 external이 아니다. 공유하는 blind spot을 flag할 수 없다. Zhang et al. 2025 convergence guarantee는 verifier가 generator와 독립적일 것을 요구한다. judge가 generator가 볼 수 없는 failure를 볼 때만 gate가 닫힌다.

synthetic pipeline을 design하기 전에 물어야 할 structural question: **내 loop에는 generator와 독립적이며 generator drift를 detect할 수 있는 signal이 있는가?** yes라면 gate를 design하고 mitigable regime에 있다. no라면 unavoidable regime이며, collapse가 visible해지기까지 generation이 얼마나 남았는지가 유일한 질문이다.

---

## 4. Verification as defense — 2025 faithfulness-check protocol

[[faithful-synth-eval]]은 2024–2025년 verification literature를 네 complementary audit axis로 consolidate한다. 어느 하나만으로는 충분하지 않다. 함께 사용해야 production synthetic pipeline의 minimum gate가 된다.

**Axis 1 — Tail-mass measurements.** Shumailov의 OPT-125M table이 보이게 만든 failure(§1)는 mean PPL에서는 보이지 않고 tail PPL에서는 명백하다. concrete measurement:

- **Rare-token recall.** real reference corpus를 고정한다. frequency `< 10⁻⁴`인 token을 식별하고, synthetic corpus가 이 token들을 comparable frequency로 produce하는 fraction을 계산한다. collapse는 3–5 generation 후 <50% recall로 나타난다.
- **Rare n-gram overlap.** 같은 방식이지만 3-gram과 5-gram에 적용한다. rare phrase가 rare token보다 더 빨리 죽기 때문에 mode collapse에 더 날카로운 signal이다.
- **Rare-concept recall.** LLM-tagged categorical entity(named entity, scientific terminology, minority language)에서 long-tail이 preserved되는가? frontier-model eval이 신경 쓰는 semantic tail에 가장 가까운 proxy다.

**Axis 2 — External verification.** Zhang et al. 2025의 convergence-guaranteeing filter다. modality별 concrete form:

- **Math:** ground-truth answer matcher(OpenMathInstruct, rStar-Math). deterministic; 0% false-positive rate; temperature 1.0의 raw generation 중 ~60% reject.
- **Code:** unit-test execution(APIGen, CodeUltraFeedback). sandboxed runtime; test당 5-second timeout.
- **Tool / function calls:** [[apigen]] 3-layer stack — format → execution → semantic. ablation: 어느 layer를 제거해도 BFCL accuracy 6–18% 손실(§6).
- **Factual / RAG:** retrieval-grounded NLI entailment. claim은 retrieved evidence에 entail되어야 하며, unsupported claim은 reject된다.
- **Open-ended text:** reward-model-as-judge([[nemotron-4-synthetic]] Nemotron-4-340B-Reward). 가장 unreliable하다. RM blind spot이 dominant failure mode다.

**externality** requirement는 negotiable하지 않다. verifier의 model weight는 generator와 다른 training lineage(ideal하게는 다른 organization, 다른 base model, 다른 preference dataset)에서 와야 한다. rule-based verifier(answer matcher, unit test, schema validator)가 preferred다. maximally external하기 때문이다.

**Axis 3 — Coverage / diversity metrics.** synthetic corpus가 몇몇 high-density basin에 cluster되지 않고 proxy model의 *gradient space*를 채우는지 측정함으로써 tail recall을 보완한다.

- **G-Vendi ([[prismatic-synthesis]]).** 각 candidate `x_i`에 대해 작은 proxy LM에서 normalized gradient `g_i = ∇_θ L(x_i; θ)`를 계산하고, ~8K dims로 random-project하며, density matrix `K_{ij} = <g_i, g_j>`를 만들고, `exp(vN-entropy(K/tr(K)))`를 return한다. 300+ run에서 OOD accuracy와 Spearman ρ ≈ 0.9. embedding-Vendi(14× larger encoder)와 GPT-4-based Skill-Set Entropy를 **이긴다**.
- **Embedding-cluster occupancy.** reference encoder에서 k-means(typical `k` = 1000)로 synthetic corpus가 populate하는 distinct embedding cluster 수를 센다. corpus size가 늘어도 cluster count가 떨어지면 mode collapse다.
- **kNN diversity.** embedding space에서 average kNN distance. rare-concept loss와 함께 collapse한다.

**Axis 4 — Drift-over-iteration signals.** iteration이 unavoidable하다면 drift를 monitor한다.

- round `k`와 round `k+1` 사이 Δ(rare-token recall) > 5% → early warning.
- round 간 Δ(G-Vendi) > 10% → active collapse.
- 3 round 동안 Δ(embedding-cluster occupancy) > 20% → mode collapse in progress.

**Memorization↔generalization drift ("Closer Look at Model Collapse," 2025).** synthetic fraction이 증가하면 model은 surface metric이 놓치는 memorization-heavy regime으로 이동한다. operational test: memorization probe(training string의 exact-match recall)는 synthetic fraction과 함께 상승해서는 **안 된다**. 상승하면 model이 training data로 collapse하는 중이다. Gaussian-mixture mode contraction의 synthetic-fraction analog다.

---

## 5. Gate vs. no-gate — comparison

course의 나머지가 반복해서 참조할 single table이다. empirical number는 [[model-collapse]](OPT-125M, 10 generations), [[strong-model-collapse]](GPT-2 scale, scaling law), [[synthetic-data-scaling-laws]](8B rephrased vs textbook), [[nemotron-4-synthetic]](340B alignment), [[apigen]](7B function-calling), Zhang et al. 2025에서 aggregate했다.

| Pipeline | Real anchor | Synthetic fraction | Verifier gate | Observed collapse | Notes |
|---|---|---|---|---|---|
| Pure recursive replacement (Shumailov 2024 baseline) | 0% | 100% | none | gen-9까지 incoherent; tail-PPL >10⁴ | reference worst-case |
| Shumailov 10%-real-accumulation | 10%, persistent | 90%/gen | none | tails bounded; real-only보다 ~30% worse | accumulation alone bounds error |
| Open-web mix with 1% LLM contamination (Dohmatob empirical) | 99% | 1% | none | scaling-law flatline; 더 큰 `N`의 benefit 없음 | policy-relevant regime |
| WRAP / rephrased-synthetic pretraining (~30%) | 70% | ~30% rephrased | external paraphrase constraint | measured scale까지 clean rectified scaling; 5–10× speedup | rephrasing *is* a gate (real source에 anchor) |
| Pure-generated "textbook" synthetic at high fraction | varies | >50% textbook | none / weak | Shumailov 예측대로 collapse | Phi-cautionary regime |
| Nemotron-4 alignment (98% synthetic) | ~20K human (HelpSteer2) | 98% | Nemotron-4-340B-Reward + category-seeded prompts | no collapse; SOTA RewardBench | RM-as-judge, human anchor에서 periodic refresh |
| APIGen function-calling (60K) | 0% human | 100% synthetic | 3-layer (format → execution → semantic) | no collapse; xLAM-7B #1 BFCL <13B | external, rule-heavy gate; layer 하나라도 제거 → 6–18% loss |
| Zhang et al. 2025 verified loop | variable | variable | external verifier | bounded convergence (analytical + empirical) | verification suffices |
| Gradient-targeted synthesis (Prismatic) | 0% human | 100% synthetic | G-Vendi coverage + answer verifier | no collapse; 7B beats 671B teacher | teacher의 gradient manifold *밖*에서 generate; coverage is a gate |

pattern: **collapse 없이 scale되는 모든 pipeline에는 gate가 있다.** gate의 형태는 다르다. rule-based execution(APIGen), reward-model judgment(Nemotron-4), paraphrase anchoring(WRAP), gradient-coverage selection(Prismatic), fresh-human accumulation(Gerstgrasser). 그러나 gate는 항상 있다. gate 없는 pipeline은 model scale과 무관하게 3–10 generation timescale에서 collapse한다.

---

## 6. Canonical gate designs — 무엇을 build할 것인가

두 gate template이면 필요한 대부분을 cover한다. 둘 다 pattern이지 library가 아니다. implementation detail은 modality-specific chapter(24–27)에 있지만 invariant는 같다.

**Gate template A — RM-as-judge ([[nemotron-4-synthetic]]).** rule-based verification이 infeasible한 open-ended text용. Structure:

```
human_anchor:      20K human preferences (HelpSteer2 class)
                       │
                       ▼
reward_model = train(human_anchor)          # Nemotron-4-340B-Reward
                       │
generator_v0 ──► candidates  ──► RM.score(c) ≥ τ  ──► SFT set_v0
                       │
generator_v1 = finetune(generator_v0, SFT set_v0)
                       │
generator_v1 ──► preferences (chosen, rejected) chosen via RM
                       │
generator_v2 = DPO(generator_v1, preferences)  then RPO with RM-reweighting

# Every N iterations, retrain RM on fresh human preferences (anti-stale)
reward_model = train(human_anchor_fresh ∪ human_anchor)
```

Critical invariant:
- Human anchor는 consumed되지 않는다. 항상 additive다.
- RM은 cadence에 맞춰 refresh된다(Nemotron-4는 HelpSteer2 anchor에서 generation마다 refresh).
- RM은 자신의 training data를 score하지 않는다.
- Acceptance threshold `τ`는 held-out human preference agreement rate로 설정한다(Nemotron-4는 human judge와 ~80% agreement가 되도록 τ를 tune했다고 보고).

Failure mode: RM drift. 같은 RM checkpoint를 여러 generator generation에 재사용하면 generator가 RM을 Goodhart's-laws한다. 실제로 더 quality가 높지 않은데 high-RM-score output을 만드는 법을 배운다. refresh cadence가 이를 끊는다.

**Gate template B — 3-layer rule-heavy verification ([[apigen]]).** rule-based ground truth가 존재하는 modality(code, math, tool call, structured output)용:

| Layer | Check | Rejection rate | Cost |
|---|---|---|---|
| 1. FORMAT | parse JSON; required params; type/enum match schema | ~15–25% of raw | ~1ms |
| 2. EXECUTION | run in 5-sec sandbox; exception or timeout → reject | ~10–15% of L1-passed | ~5s |
| 3. SEMANTIC | LLM-judge (external): "does call fulfill query given result?" | ~10% of L2-passed | ~1 teacher call |

Overall acceptance ≈ raw generation의 60%.

APIGen의 ablation은 세 layer가 모두 중요하다는 load-bearing evidence다.
- format 제거: −18% BFCL-V1.
- execution 제거: −11% BFCL-V1.
- semantic 제거: −6% BFCL-V1.

어느 layer를 제거해도 특정 failure class가 통과한다. Format-only는 schema error를 잡지만 hallucinated argument를 통과시킨다. Format+execution은 runtime error를 잡지만 semantically-wrong-but-runnable call(예: conversion function에 wrong unit)을 통과시킨다. 세 layer를 모두 쓰면 xLAM-7B가 BFCL-V1에서 report한 <3% hallucination rate가 나온다.

**왜 layer가 multiplicatively stack되는가.** 세 layer는 대략 independent하다. sample은 execution을 시도하지 않으므로 format에는 fail하지만 execution에는 fail하지 않을 수 있다. correct call도 가끔 timeout되므로 execution에는 fail하지만 semantics에는 fail하지 않을 수 있다. 실행은 되었지만 semantic은 틀릴 수 있다. acceptance rate는 multiply한다. `0.80 × 0.85 × 0.90 ≈ 0.61`. 이것이 single-layer "LLM-as-judge" gate가 보기보다 약한 이유이기도 하다. 그것은 첫 두 layer 없는 third layer이며, 첫 두 layer가 raw generation의 roughly half를 제거한다.

**새 modality를 위한 minimal reference gate.** 새로운 modality(예: multi-turn tool-use trajectory — [[apigen-mt]] 참조)에 synthetic pipeline을 설계한다면 template은 다음과 같다.

```python
def accept_candidate(candidate, ground_truth=None, judge=None) -> bool:
    # Layer 1 — structural: parse + schema + type check
    try: parsed = parse_and_validate(candidate)
    except (JSONDecodeError, SchemaError): return False
    # Layer 2 — executable: sandbox with 5s timeout; match ground truth if any
    result = None
    if has_executable_ground_truth(parsed):
        try: result = sandbox_execute(parsed, timeout_s=5)
        except (Timeout, Exception): return False
        if ground_truth is not None and result != ground_truth: return False
    # Layer 3 — semantic: external judge independent of generator
    if judge is not None and judge.score(candidate, parsed, result) != "Yes":
        return False
    return True
```

~60% acceptance(APIGen ratio)를 target하고 generation마다 `len(accepted) / raw_generated`를 report하는 build loop로 감싸라. 그 band 바깥이면 gate가 misconfigured된 것이다. modality-specific `parse_and_validate` / `sandbox_execute` / `judge`와 함께 이 skeleton은 collapse하지 않는다고 publish된 모든 production synthetic pipeline의 shape다. 다음 세 장은 이 gate 위에 modality-specific piece를 채운다.

---

## 7. Dashboard가 보여야 하는 것

average를 믿을 수 없다. synthetic pipeline에서 iteration마다 log해야 할 minimum set:

| Metric | Cadence | Threshold for action |
|---|---|---|
| Raw acceptance rate (gate pass %) | per generation | previous gen 대비 Δ > 10% → gate 또는 generator 조사 |
| Rare-token recall vs real reference | per generation | <75% → early collapse signal |
| Rare 5-gram recall vs real reference | per generation | <50% → mode collapse starting |
| Embedding-cluster occupancy (k=1000) | per generation | 3 gens 동안 Δ > 20% → active collapse |
| G-Vendi on a 10k sample | per generation | Δ > 10% → gradient-space contraction |
| Mean loss on a fixed real held-out set | per generation | **auxiliary only** — primary signal로 삼지 말 것 |
| Memorization probe (exact-match recall of training strings) | per generation | synthetic fraction과 함께 상승 → collapse into memorization regime |
| Verifier disagreement rate on a held-out human-labeled set | per N iterations | rising → RM drift; refresh |

Mean PPL은 일부러 bottom에 있다. collapse 아래에서 *wrong direction*으로 움직인다. mode-collapsed output은 self-consistent하기 때문에 내려간다. 바로 이 signal이 Shumailov-era researcher에게 model이 improving한다고 믿게 했지만 실제로는 collapse 중이었다. log는 하되, 단독으로 믿지 마라.

---

## Connections and what's next

- **[[model-collapse]] / §1** — Shumailov Nature 2024: mechanism + OPT-125M. **[[strong-model-collapse]] / §2** — Dohmatob ICLR 2025: scaling flatline at 1%. **[[faithful-synth-eval]] / §4** — 2025 verification cluster(tail / external / coverage / drift axes).
- **[[synthetic-data-scaling-laws]] / §3, §5** — scaling 아래 rephrased vs pure-generated. **[[prismatic-synthesis]] / §4 Axis 3, §5** — G-Vendi; anti-collapse mechanism으로서 gradient-targeted generation.
- **[[nemotron-4-synthetic]] / §6 Template A** — RM gate + HelpSteer2 anchor가 98%-synthetic production pipeline을 구함. **[[apigen]] / §6 Template B** — 3-layer rule-heavy verification; 각 layer의 load-bearing은 ablation으로 확인됨. **[[apigen-mt]]** — 같은 gate의 multi-turn extension.
- **ch-22** — upstream filter(Quality / Diversity / Gradient-Based Selection). **ch-24–27** — modality application(reasoning traces, conversation, tool calls, preference pairs)이 모두 이 장의 gate template 위에 build된다.

## Further reading

- [[model-collapse]] — Nature 2024; OPT-125M figure와 Gaussian-mixture proof를 읽어라.
- [[strong-model-collapse]] — ICLR 2025 Spotlight; Theorem 1 + 1% empirical reproduction.
- [[faithful-synth-eval]] — Zhang 2510.16657 is the convergence-guarantee anchor.
- [[synthetic-data-scaling-laws]] — SynthLLM + Demystifying + BeyondWeb (2025).
- [[prismatic-synthesis]] — Jung 2025 (2505.20161); G-Vendi definition and 7B-beats-671B result.
- [[nemotron-4-synthetic]] / [[apigen]] — production templates for Gates A and B respectively.

## Companion visualization

**[figures/collapse-iterations.html](figures/collapse-iterations.html)** — recursive training generation 0→10에 걸친 Gaussian-mixture collapse interactive. Slider는 (a) fresh-human injection fraction(0–100% anchor persistence), (b) verifier pass-rate(off-distribution sample이 다음 generation training set에 들어가기 전에 reject될 probability), (c) generation당 sample size `N`을 control한다. 왼쪽 panel은 generation별 density를 dark(gen 0)에서 light(gen 10)로 color해 pure self-iteration 아래 tail collapse와 real-data anchoring 또는 external verifier gate 아래 tail preservation을 보여 준다. side panel은 generation별 두 quantity, **rare-mode recall**(low-weight component 중 non-zero estimated weight를 가진 fraction)과 **KL to the true distribution**을 track한다. default(0% anchor, 0% verifier)에서 시작해 Shumailov curve를 reproduce하고, verifier slider를 올려 Zhang et al. 2025 convergence guarantee가 작동하는 것을 관찰하라.
