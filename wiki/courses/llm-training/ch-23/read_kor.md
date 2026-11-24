<!-- chapter: ch-23
     track: synthetic
     kind: content
     title: Model Collapse and Verification of Synthetic Data
     deps: [ch-22]
     sources: [[model-collapse]], [[model-collapse-accumulation]], [[collapse-or-thrive]], [[strong-model-collapse]], [[beyond-model-collapse]], [[escaping-collapse-verification]], [[long-tail-knowledge]], [[rlhf-generalisation-diversity]], [[artificial-hivemind]], [[echo-chamber-rl-post-training]], [[bespoke-stratos]], [[openmathinstruct]], [[omegaprm]], [[deepswe]], [[qwen-2.5]], [[likelihood-displacement]], [[nemotron-4-synthetic]], [[apigen]], [[prismatic-synthesis]]
     figures: figures/collapse-iterations.html
     revised: 2026-09 (generality revision)
-->

# 제23장 — Model Collapse와 Synthetic Data의 Verification

> **핵심 통찰.** 각 model generation이 오직 이전 generation output으로 train되면 low-probability content가 먼저 사라지고 error가 커진다. OPT-125m을 이렇게 wikitext2에서 fine-tune하면 real test set perplexity가 generation마다 오른다([[model-collapse]] Fig. 1). Outcome은 data가 synthetic인가뿐 아니라 data를 어떻게 쓰는가에 달렸다. Real data를 유지하고 round별 synthetic data를 추가하면 filter 없이도 9M–126M language model과 Gemma 2 SFT에서 test error가 bounded로 유지됐다([[model-collapse-accumulation]] Table 2; [[collapse-or-thrive]] Fig. 3). 그러나 regression theory에서는 shifted data의 fixed non-vanishing share가 있으면 data가 늘어도 test error가 더 낮아지지 않는다([[strong-model-collapse]] Corollary 1). Wrong label keep rate가 correct label keep rate보다 낮은 verifier는 training이 성공하는 generator error rate를 높인다([[beyond-model-collapse]] Theorem 4.2). Oracle selector에서는 Llama-2-7B generator summary의 12.5%가 모든 human summary보다 나은 model을 train했다(Fig. 5). 많은 round 뒤 model은 truth가 아니라 verifier가 accept하는 쪽으로 움직인다([[escaping-collapse-verification]] Theorem 4.1).
>
> **지침.** Pipeline이 자기 output으로 두 round 이상 retrain하면 original real data와 earlier round를 replace하지 말고 training set에 유지하라. Temperature 1.0 sample에서 iteration 4의 TinyStories test cross-entropy는 replace에서 0.53→0.66으로 올랐지만 accumulate에서는 낮아졌다([[model-collapse-accumulation]] Table 2). Synthetic label을 training에 남긴다면 labeled sample에서 verifier의 correct/wrong label keep rate를 재고 breakdown point `p* = 1/(1 + ψ/ϕ)`를 계산하라. Summarization experiment에서 stronger model이 generator 자신보다 나쁜 selector였기 때문이다([[beyond-model-collapse]] §6.2). Verifier reject를 negative로 reuse하면 false-negative rate를 먼저 재라. Answer parser가 model judge가 accept한 correct math solution의 약 2/3를 버렸다([[bespoke-stratos]] “Data Curation”, derived). 그렇지 않으면 uncertain reject는 discard/mask하라. 범용 모델은 synthetic round 전후 fact frequency별 accuracy와 input당 output diversity를 보고하라. Average score는 tail loss를 보이지 않기 때문이다([[long-tail-knowledge]] §3; [[rlhf-generalisation-diversity]] §6.2).

## 학습한 버전에 대한 정정

1. “OPT-125M generation 약 9에서 output incoherent”, “rare-token PPL spike” → generation-9 text는 반복 phrase pattern을 보이고 original wikitext2 test의 mean PPL은 generation마다 오르며 rare-token PPL은 측정하지 않는다([[model-collapse]], Example 1, Fig. 1 caption).
2. “Average PPL은 generation 간 개선하면서 tail이 사라짐”, “collapse에서 mean PPL이 down” → real-test PPL은 증가한다. Generation-0 model이 generated sequence에 매긴 PPL은 더 낮아지고 high-PPL tail은 길어진다([[model-collapse]], Fig. 1 caption).
3. Tail-loss table의 34.1→31.4/412→`>10⁴` → 존재하지 않는다. Reported number는 real-data fine-tuning 뒤 mean PPL 34(zero-shot 115), original data 없이 20→28 PPL point loss다([[model-collapse]], “Model collapse in language models”).
4. “temperature 1.0으로 100k token, 10 generation” → 64-token block마다 64 token을 predict하는 five-way beam search, training-size generated set이며 generation 수는 미명시다([[model-collapse]], same section, Verification).
5. “pure replacement/accumulation” 및 “real-only보다 30% worse” → 두 번째 setting은 10 epoch와 generation마다 random original data 10% sampling이며 accumulation은 [[model-collapse-accumulation]]의 것이다. 30% figure는 없다([[model-collapse]], Fig. 1c).
6. Statistical sampling error/near-Gaussian 및 제시한 weight/mean recursion → 용어는 statistical approximation error, limit은 delta/low-variance point estimate, 1D Gaussian은 `Var(X_j^n) = σ²(1 + n/M) + O(2)`이며 weight recursion은 없다. `(100/N)·(N/100)=1`이다([[model-collapse]], Definition 2.1, Theorem 3.1, SI §3.1.1).
7. “1% contamination bias `c(p)·σ²`” → fixed label-shift synthetic distribution ridge regression의 floor는 small `d/n`에서 `p₂²c²`이고 `c²`는 label mismatch다. 1%는 abstract의 theory statement다([[strong-model-collapse]], arXiv v2 Corollary 1, Eq. 12).
8. “GPT-2-scale 1% injection scaling curve never recovers” → quality-filtered BabiStories synthetic data는 scaling law progression을 delay하며 authors는 plateau를 예상할 뿐 보여주지 않는다([[strong-model-collapse]] §4.2, Fig. 8).
9. “verified corpus는 iid earlier-model assumption을 충족하지 않음” → 여기 synthetic은 test distribution에서 벗어나는 모든 distribution이므로 non-zero label mismatch verified data도 theory 안에 있다([[strong-model-collapse]] §1.1). Iterated verified retraining은 verifier knowledge center로 converge한다([[escaping-collapse-verification]] Theorem 4.1).
10. “gate 없는 synthetic pipeline은 3–10 generation에 model scale과 무관하게 collapse” → accumulation은 verifier 없이 LM 9M–126M/linear regression에서 error를 bounded로 유지했고 Gemma 2 SFT에도 replicate됐다([[model-collapse-accumulation]] Fig. 2, Table 2, Theorem 2; [[collapse-or-thrive]] Fig. 3). Model size는 profile을 바꾼다([[strong-model-collapse]] Theorem 2, Fig. 8).
11. Zhu et al. token-level re-sampling → human data를 token-level edit해 semi-synthetic data를 만드는 제안이다(arXiv:2412.14689v3, Abstract; library card 없음).
12. “rephrased synthetic은 30%/2:1 optimum” → verified library source에 없다. 좁은 verified result는 Gemma 2 2B HelpSteer2 SFT에서 real example이 1024 이하일 때만 synthetic example이 test loss를 낮췄다는 것이다([[collapse-or-thrive]] §4, Fig. 5).
13. Zhang et al. reliable verifier guarantee → 저자는 Yi/Liu/Cheng/Xu이며 verified retraining은 verifier knowledge center로 converge하고 unbiased verifier가 아니면 early gain이 plateau/reverse한다([[escaping-collapse-verification]], Abstract, §4).
14. “Closer Look at Model Collapse” LM probe → own sample로 재훈련한 image diffusion model 연구다(arXiv:2509.16499v3, Abstract; library card 없음).
15. Nemotron-4 RM refresh/80% agreement quote → report에 없다. RM은 HelpSteer2 10K로 train하고 aligned 340B intermediate model이 다음 round data를 generate한다([[nemotron-4-synthetic]], §3.1, §3.2.4, Verification).
16. APIGen의 format/execution/semantic delta, rate/sandbox/hallucination/rank/acceptance → add-back ablation은 −4.06/−5.94(7B), −9.59/−12.17(1B), format arm 없음; generator별 pass rate 34.42%–84.15%, xLAM-7B rank 6th/85.65, timeout/hallucination rate 미보고다([[apigen]], Tables 1–2, Fig. 5, Verification).
17. Math matcher 0% FP/60% reject → answer matching은 flawed reasoning의 right-answer solution을 keep하며 OpenMathInstruct는 rare라 할 뿐 rate는 없다([[openmathinstruct]] §2.3). Parser도 false negative를 만든다([[bespoke-stratos]], “Data Curation”).
18. Nemotron/APIGen/Prismatic gate-no-collapse table 및 “scale without collapse는 모두 gate” → repeated self-training generation across error를 측정하지 않아 collapse evidence가 아니다([[nemotron-4-synthetic]], [[apigen]], [[prismatic-synthesis]]).
19. G-Vendi 8K dim/7B beats 671B teacher → projection dim 1024; R1-Distill-Qwen-32B에서 distilled PrismMath-7B는 R1-Distill-Qwen-7B보다 benchmark 7개 중 6개에서 낫다([[prismatic-synthesis]] §2.1, Table 4).
20. Drift threshold/four-axis audit/`faithful-synth-eval` card → source가 threshold를 주지 않으며 card에 verifiable primary source가 없어 제거했다.
21. Same-policy chosen/rejected DPO가 unavoidable collapse → cited source가 test하지 않았다. Verification 결과를 정하는 variable은 sample policy 공유가 아니라 label source다([[beyond-model-collapse]] §6.2; [[escaping-collapse-verification]] §4).
22. “ch-24–27 reasoning/conversation/tool/preference pair” → ch-27은 Agentic Trajectory Data이며 preference pair는 ch-39/41이 다룬다(outline).
23. Figure preset “10% accumulate”/true-component-distance verifier → 전자는 Shumailov per-generation 10% sampling을 mislabel하고 후자는 verifier success를 simulation에 심으므로 figure를 다시 썼다.

Old version section map: §1 → §1; §2 → §3; §3 → §2/§4; §4 → §4/§7; §5 removed(corrections 16–18); §6 → §4 및 “Negative samples and negative feedback”; §7 → §7. 이 version의 §5/§6은 새로 추가했다.

## 이 장이 범용 모델에 중요한 이유

Synthetic data는 pretraining 뒤 모든 stage에 들어간다. Mid-training의 rephrased document, SFT의 distilled instruction/trace, preference optimization의 model-ranked pair, RL의 self-generated rollout이다. Breadth에는 두 risk가 있다. Recursive collapse는 earlier model output을 여러 round train할 때 real-data error가 자라는 것이고([[model-collapse]]), single-round narrowing은 recursion 없이도 distillation/post-training 한 round가 output diversity/tail coverage를 낮추는 것이다([[rlhf-generalisation-diversity]]; [[artificial-hivemind]]). 두 cost는 training data에 드문 content, 즉 적은 document에만 있는 fact([[long-tail-knowledge]]), low-resource language(ch-13a), uncommon answer format([[echo-chamber-rl-post-training]])에 떨어진다. Nature paper 저자는 low-probability event가 marginalized group과 관련 있는 경우가 많다고 말한다([[model-collapse]] “Discussion”). 이 장은 risk 둘을 나누고 replace/accumulate/verify theory·experiment와 각각의 measurement를 제시한다.

## §1 Model output의 recursive training: Shumailov et al.이 측정한 것

**정의.** Model collapse는 generation 하나가 만든 data가 다음 generation training data를 pollute하여 later model이 original distribution을 잘못 지각하게 되는 degenerative process다([[model-collapse]] Definition 2.1). Early collapse에서는 distribution tail이 사라지고 late collapse에서는 original과 거의 닮지 않은, 흔히 variance가 훨씬 작은 distribution으로 converge한다.

**문제.** Generated data로 train한 model은 common content에서 generator를 match하면서도 low-probability content를 잃을 수 있다. 측정 가능한 질문은 generation에 따라 real test data error가 어떻게 바뀌는가다.

**방법.** Paper는 generation에 걸쳐 compound되는 error source 세 개를 든다.

1. Statistical approximation error: finite sample이 low-probability event를 under-represent한다. Primary source다.
2. Functional expressivity error: model class가 true distribution을 represent하지 못한다.
3. Functional approximation error: training이 class 안의 best model을 찾지 못한다.

Exact fit(type 2/3 error 없음)의 discrete model에서 매 generation은 current distribution에서 sample `M`개를 그리고 empirical frequency를 next distribution으로 둔다. Probability `q ≤ 1/M` state의 expected sample은 generation마다 1개 미만이다. State가 zero sample을 받으면 이후 모든 generation에서 probability zero이므로 chain은 probability 1로 delta function에 converge한다(“Discrete distributions with exact approximation”).

**수식.** Sample size `M`인 1D Gaussian recursive fitting(SI §3.1.1):

`Var(X_j^n) = σ² (1 + n/M) + O(2)`

- `X_j^n`: generation `n`의 sample, `σ²`: original distribution variance, `M`: generation당 sample, `O(2)`: `1/M`의 second-order term.

Fixed sample size의 multidimensional Gaussian에서 fitted covariance는 almost surely 0으로 converge하고 original data까지 expected squared Wasserstein-2 distance는 diverge한다(Theorem 3.1).

**계산 예시.** Discrete model에서 `M = 1,000`, state 하나 `q = 0.0005`면 expected count는 0.5이고 generation 하나에서 zero sample일 probability는 `(1 − 0.0005)^1000 = 0.606`(computed)이다. Probability 0.606으로 state는 generation 1과 이후 전부에 없다. Gaussian model에서 `σ² = 1`, `M = 100`, `n = 10`이면 generation-10 sample marginal variance는 `1 × (1 + 10/100) = 1.1`이다. Fitted parameter가 random walk를 따르기 때문이다([[model-collapse]] SI §3.1.1).

**증거(Result, single study).** Wikitext2에서 fine-tune한 OPT-125m은 zero-shot 115에서 mean PPL 34에 도달한다. 각 generation은 five-way beam search로 training set과 같은 size의 dataset을 생성한다(64-token block당 64 token). Next model은 그것으로 fine-tune하며 setting당 run 5회다. Original data 없이 epoch 5에서 저자들은 “losing some performance, from 20 to 28 perplexity points”를 보고한다(Fig. 1b). Generation마다 random original data 10%를 sample하고 epoch 10을 쓰면 degradation은 “only minor”다(Fig. 1c). Fig. 1 caption은 original test mean PPL이 generation마다 증가하고 later generation이 generation-0 model이 probable하다고 보는 sequence를 더 많이 만들며, generation-0이 절대 만들지 않을 sequence tail도 길어진다고 한다. Repetition penalty 2.0은 PPL을 두 배로 하고 effect를 없애지 못한다(“Ablation: Repetitions”). Gaussian mixture/VAE에서도 같은 qualitative loss가 보인다(SI §4).

**조건과 한계.** LLM experiment는 125M model 하나, pretraining이 아닌 fine-tuning, beam-search generation, data replacement를 쓴다. Learning rate/batch size/optimizer/generation 수는 보고하지 않는다(card Verification). Verifier/filter가 없고 larger model도 test하지 않는다.

**의미.** Real held-out data의 mean PPL은 이 collapse를 detect한다. Tail loss는 per-sequence score distribution에 보이므로 average generation score만 보고하는 loop는 typical output 쪽 shift를 놓칠 수 있다. [figures/collapse-iterations.html](figures/collapse-iterations.html)의 Panel A는 discrete model을 simulate하고 reader가 chosen `M`/generation 수에서 replace/accumulate 아래 surviving low-probability state 수를 비교하게 한다.

## §2 Replace, accumulate, fixed-size subset

**정의** ([[model-collapse-accumulation]] §2, Fig. 1). **Replace:** model `n`은 model `n−1`에서 sample한 data로만 train한다. **Accumulate:** model `n`은 original real data와 round 1부터 `n−1`까지의 모든 synthetic dataset으로 train한다. **Accumulate-subsample:** pool은 accumulate하지만 각 round는 fixed-size subsample로 train한다([[collapse-or-thrive]] §3).

**문제.** §1의 OPT-125m setting 1은 round 1 뒤 real data가 사라지는 replace다. Web corpus는 시간이 갈수록 커진다. Llama 1/2/3는 token 1.4T/2T/15T로 train됐다([[model-collapse-accumulation]] §1). Data가 accumulate할 때도 collapse하는지가 질문이다.

**방법(linear regression, Theorem 2).** Feature `x ~ N(0, I_d)`, label `y = x·w* + ε`, `ε ~ N(0, σ²)`다. Round 1은 real sample `T`개에 OLS를 fit한다. 이후 round마다 같은 input에 previous fit과 fresh noise로 relabel한 뒤 chosen data에 refit한다.

`E_test^Replace(ŵ_n) = σ²d/(T − d − 1) × n`

`E_test^Accum(ŵ_n) = σ²d/(T − d − 1) × Σ_{i=1..n} 1/i²  ≤  σ²d/(T − d − 1) × π²/6`

- `E_test`: true distribution expected squared error에서 noise floor를 뺀 것(Eq. 1), `σ²`: label-noise variance, `d`: input dimension, `T`: round마다 추가한 sample(`T ≥ d + 2`), `n`: round index.

Accumulate에서 round `i`는 data의 `1/i` share를 기여하므로 noise는 squared error에 `1/i²` weight로 들어가며 `1/i²` sum은 finite다(§3.2).

**계산 예시.** `σ² = 1`, `d = 10`, `T = 111`이면 `σ²d/(T − d − 1) = 10/100 = 0.1`이다. Round 10 뒤 replace는 `0.1 × 10 = 1.0`이다. Accumulate는 `0.1 × (1 + 1/4 + … + 1/100) = 0.1 × 1.550 = 0.155`이고 bound `0.1 × 1.645 = 0.164`보다 낮다(computed). [figures/collapse-iterations.html](figures/collapse-iterations.html)의 Panel B는 어떤 `σ²`, `d`, `T`에도 두 curve를 그린다.

**증거.** GPT-2(9M)와 Llama-2(12M, 42M, 126M)를 TinyStories(token 470M)에서 epoch 1로 scratch pretraining한다. 각 round는 TinyStories-size dataset을 temperature 1.0 또는 0.3으로 sample한다([[model-collapse-accumulation]] §2.1). Evaluation cross-entropy(Table 2):

| Model | Round 1 | Round 4, accumulate | Round 4, replace | Round 10, replace | Round 4, grown dataset으로 replace |
|---|---|---|---|---|---|
| GPT-2 9M | 1.82 | 1.74 | 2.39 | 2.91 | 2.18 |
| GPT-2 9M, temperature 0.3 | 1.82 | 1.75 | 5.82 | 9.85 | not run |
| Llama-2 12M | 2.06 | 1.94 | 2.72 | not run | not run |
| Llama-2 42M | 1.90 | 1.76 | 2.52 | not run | not run |
| Llama-2 126M | 1.71 | 1.59 | 2.23 | not run | not run |

마지막 column은 synthetic set이 accumulated set만큼 커도 replace가 degrade함을 보인다. 따라서 dataset size만으로 차이를 설명할 수 없다. **Replicated:** HelpSteer2(round당 example 약 12.5K)에서 Gemma 2 2B/9B/27B를 fine-tune하면 replace에서는 collapse하고 accumulate에서는 그렇지 않다([[collapse-or-thrive]] §2.3, Fig. 3). Accumulate-subsample은 setting 다섯 개에서 둘 사이에 있고 대체로 plateau한다([[collapse-or-thrive]] §3, Fig. 4).

**조건과 한계.** Language-model run은 최대 parameter 126M, filter 없음, 그리고 GPT-3.5/4-generated TinyStories를 쓴다. CelebA VAE는 accumulate에서도 replace보다 느리지만 degrade하고 glass 같은 minor attribute를 잃는다([[model-collapse-accumulation]] §2.3). “Bounded”는 “data를 더 넣으면 개선”이 아니다. Linear model에서 accumulate bound는 round-1 error의 최대 1.645배다. Fine-tuning에서는 real-data count가 중요하다. HelpSteer2의 Gemma 2 2B는 real example이 1024 이하일 때 synthetic example이 test loss를 낮췄고, 그보다 많으면 fixed real count에서 synthetic data가 늘수록 거의 항상 test loss가 올랐다([[collapse-or-thrive]] §4, Fig. 5). Curated web-scale pretraining pipeline이 model generation을 넘겨 관찰한 것은 이 source가 보고하지 않는다(Open question).

**의미.** Multi-round pipeline은 original human data와 earlier round를 pool에 유지해야 한다. 이는 divergence를 control한다. Synthetic data를 real data만큼 유용하게 만들지는 못하며 그것은 §3의 주제다.

## §3 Strong model collapse: scaling regime에서 shifted data의 fixed share

**정의.** Strong model collapse는 training data의 사라지지 않는 fraction이 test distribution과 다른 distribution에서 오면 total data size가 커져도 test error가 clean-data value로 계속 낮아지지 않는 결과다([[strong-model-collapse]] §1.1, §3.1). 이 paper에서 “synthetic”은 그런 shifted distribution의 모든 data를 뜻하며 recursive loop가 아닌 fixed mixture다.

**문제.** §2는 accumulation이 divergence를 막음을 보인다. Scaling-law pipeline에는 data가 커질수록 계속 낮아지는 error가 더 필요하다.

**방법.** Real sample은 `y = x·w₁* + ε`, synthetic sample은 같은 input distribution에서 `y = x·w₂* + ε`을 따른다. Label mismatch `δ = w₂* − w₁*`의 covariance는 `Δ`이고 synthetic data quality는 `c² = (1/d) tr ΣΔ`다(Definition 1). Ridge model을 union으로 train한다.

**수식**(Corollary 1, Eq. 12; isotropic feature, unregularized limit, small `d/n`; paper는 `d/n`을 `ϕ`로 쓰며 이는 §4의 keep rate `ϕ`와 다른 quantity다):

`E_test ≃ σ² d/n + p₂² c² + O(ϕ²)`

- `σ²`: label noise, `d`: dimension, `n`: total training sample, `p₂ = n₂/n`: synthetic fraction, `c²`: synthetic distribution의 label mismatch.

첫 term은 `n`이 커질수록 내려가지만 둘째 term은 `n`에 의존하지 않는다. “The scaling law plateaus unless `p₂ → 0+`”(§3.1).

**계산 예시.** `σ² = 1`, `d = 100`, `c² = 1`이다. Real data만 쓰면 `n = 10⁴, 10⁶, 10⁸`에서 0.01, 0.0001, 0.000001이다. `p₂ = 0.01`이면 0.0101, 0.0002, 0.000101이다. `n = 10⁸`에서 mixed model error는 real-only error의 약 101배이고 floor 0.0001은 `p₂²c²`가 정한다(computed). `p₂ = 0.1`이면 floor는 0.01이다. [figures/collapse-iterations.html](figures/collapse-iterations.html)의 Panel C는 어떤 `d`, `c²`에도 이 curve를 그린다.

**증거.** 다른 network의 synthetic label을 쓴 MNIST two-layer network는 `p₂`가 커질수록 scaling이 느려지고 plateau하며 `p₂ → 0`일 때만 가라앉는다(§4.1, Fig. 7). Language modeling에서 BabiStories/GPT-2-generated story(quality-filtered, temperature 1) mix로 train한 GPT-2-small(124M)은 “even moderate amounts of synthetic data delay the progression of the scaling laws”하며 저자는 “expect this to eventually lead to plateaus”라고 한다(§4.2, Fig. 8). LM run은 synthetic 1%에서 plateau를 보이지 않는다. Model size도 중요하다. `p₂ = 1`에서 18/24-layer model은 token `1×10¹⁰` 아래에서는 12-layer보다 lower loss, `3×10¹⁰` 너머에서는 higher loss다(§4.2). Fixed weight의 weighted single-step mixing은 floor를 없애지 않는다. Half real data의 iterative mixing은 scaling law를 recover하지만 real half만 train하는 것보다 못하다(§5, Corollaries 2–3, Fig. 10).

**조건과 한계.** Theory는 Gaussian input의 linear regression/random projection과 fixed-size label shift다. Filtering은 analysis 밖이다(§1.2). §2 accumulation과 여기 strong collapse는 서로 다른 질문, 즉 round across bounded error와 data 증가에 따른 continued improvement를 답한다(Interpretation).

**의미.** 범용 모델에서 label이 truth와 systematic하게 다른 synthetic data는 아무리 조금이어도 affected task에 더 많은 data가 없애지 못하는 error floor를 만든다. Verification(§4) 또는 real content rephrasing으로 `c²`를 낮추는 것과 `p₂`를 낮추는 것이 lever다.

## §4 Synthesis 아래 verification: breakdown point와 long-run convergence

**정의.** Verifier는 synthetic example마다 keep 여부를 정하는 procedure다. Answer checker, unit test, reward model, judge model, 사람일 수 있다.

**문제.** Generator는 자신이 select하는 것보다 좋은 answer를 만들 수 있다. Eigenvalue를 tolerance 1%로 predict하는 transformer에서 oracle이 고른 beam 50개 중 최고는 60.4% correct, model 자신이 lowest-perplexity로 고른 beam은 19.2%, greedy decoding은 20.2% correct다([[beyond-model-collapse]] §3, Table 1). Model likelihood는 correct output을 식별하지 못한다.

**방법과 수식(single round, [[beyond-model-collapse]] Theorem 4.2).** `p = P(synthetic label ≠ true label)`, `ϕ = P(keep | label correct)`, `ψ = P(keep | label wrong)`라 하자. Infinite-data limit에서 kept example로 train한 linear classifier는 다음을 따른다.

`p* = 1 / (1 + ψ/ϕ)`; `p < p*`면 accuracy → 100%, `p > p*`면 → 0%.

**계산 예시.** Filter가 없으면(`ϕ = ψ = 1`) `p* = 0.5`이므로 label 절반 초과가 wrong인 generator는 반대 rule을 가르친다. `ϕ = 0.9`, `ψ = 0.3` verifier면 `p* = 1/(1 + 1/3) = 0.75`다. Correct label 30%를 reject하고 wrong label 5%를 keep하는 verifier(`ϕ = 0.7`, `ψ = 0.05`)면 `p* = 0.933`이다(computed). High false-negative rate는 data를 잃게 하지만 `p*`는 `ψ/ϕ` ratio에 의존하므로 high false-positive rate보다 `p*`를 덜 움직인다. [figures/collapse-iterations.html](figures/collapse-iterations.html)의 Panel D는 어떤 rate에도 `p*`와 kept/rejected pool composition을 계산한다.

**증거(Result, single study).** Eigenvalue에서 verification 없음(`p* = 0.5`), synthetic example 10M개(generator training set의 50배)는 generator보다 나쁜 model을 train한다. Oracle verifier는 accuracy를 거의 두 배로 만들며 curve는 `p* ≈ 0.65` 근처에서 generator를 cross한다(§6.1, Fig. 4). XLSUM English 12.5%로 fine-tune한 Llama-2-7B generator의 news summarization에서 matched-size random selection은 generator 아래다. Oracle selection은 모든 setting에서 generator를 이기며 12.5%의 oracle selection은 human summary 100%로 train하는 것보다 낫다. Perplexity self-selection은 generator를 이기지만 generator보다 ROUGE-1이 높은 fine-tuned Llama-3 selection은 random에 가깝다(§6.2, Fig. 5). 저자들은 Llama-3 결과를 Llama-2 generated data와 Llama-3의 correlation이 더 낮기 때문, self-selection gain을 더 큰 article pool에서 input을 select했기 때문으로 해석한다(§6.2, §7; Interpretation). Finite-data simulation에서 weak verifier의 best data size는 example 약 10,000개이고 그 뒤에는 selected data가 많을수록 나빠진다(§5.2, Fig. 3).

**Many round([[escaping-collapse-verification]]).** Verifier는 true parameter `θ*`를 담는 center `θ_c` 주위 knowledge ball `B_r(θ_c)`로 model한다. `(x, y)`는 `|y − xᵀθ_c| ≤ r‖x‖ + σ_c`일 때 accept한다(§2, Eq. 1). Round 하나는 bias-variance trade-off로 error를 낮출 수 있다(Theorem 3.1). Round를 거치면 estimate는 `θ_c`로 converge하며 selectivity `r`은 rate를 바꾸되 limit은 바꾸지 않는다(Theorem 4.1). Unbiased verifier는 continued improvement, mildly biased verifier는 early gain 뒤 plateau/reverse, strongly biased verifier는 degradation을 준다(§4). Oracle ROUGE-1 verifier가 top 12.5%를 keep하는 XSUM의 SmolLM2-135M은 early round에서 개선한 뒤 reported round 15개 안에서 stabilize했고, unfiltered retraining은 initial score 주변에서 fluctuate했다(§5.3, Fig. 5).

**Production multi-round loop.** Nemotron-4-340B alignment는 synthetic SFT/preference data 98% 초과, HelpSteer2 example 10K로 train한 reward model judge, aligned 340B intermediate model이 next round data를 generate하는 round를 사용했다([[nemotron-4-synthetic]] §3.1, §3.2.4). Report는 round 간 diversity/tail coverage/real-data error를 측정하지 않으므로 pattern을 사용함은 보이지만 narrowing을 피함은 보이지 않는다.

**조건과 한계.** 두 theory는 single task의 linear model이다. Summarization/XSUM experiment는 reference에 대한 ROUGE-1을 oracle로 쓰는데 open-ended task에는 없다. Verifier를 싸게 쓸 수 있는 영역은 주로 compiler/reference solution/heuristic checker가 있는 code/math다([[beyond-model-collapse]] §2).

**의미.** 두 theory와 summarization experiment 모두 verification effect는 verifier 자신의 task score가 아니라 generator own output에서 correct/incorrect label을 얼마나 잘 분리하는지에 달렸다. Verifier가 싼 곳만 verify하는 loop는 data를 math/code/tool use로 shift한다(Interpretation). 다른 domain breadth에는 human 또는 calibrated-judge evaluation이 필요하다(ch-44b, ch-49).

## §5 별개의 risk 둘: recursive collapse와 single-round narrowing

**정의.** **Recursive collapse**는 model output으로 train을 반복한 round에 걸친 error growth다(§1–§2). **Single-round narrowing**은 recursion 없이 teacher 또는 model own output으로 train한 round 하나 뒤 tail content/output variety가 줄어드는 것이다.

**문제.** Risk 둘에는 다른 control이 필요하다. Accumulation(§2)은 첫 risk를 다룬다. Single distillation round에는 accumulate할 earlier synthetic round가 없으므로 둘째에는 아무것도 하지 못한다.

**Single-round narrowing의 mechanism(Interpretation).**

1. Fact에서 generator accuracy는 그 fact가 pretraining data에 나타난 횟수에 의존한다.
2. 따라서 rare fact synthetic data는 error가 더 많거나 fact를 omit한다.
3. Verifier는 error 일부를 reject해 rare-fact example count를 더 낮춘다.
4. Student는 head보다 tail의 example을 더 적고 덜 정확하게 본다.

**Step 1의 증거([[long-tail-knowledge]]).** BLOOM-176B TriviaQA accuracy는 question/answer entity를 담은 pretraining document 수가 `10¹`에서 `10⁴`로 늘면 25%에서 55% 초과로 오른다(§3.1). Sampled question의 relevant document를 제외한 C4에서 retrain한 4.8B model은 그 question accuracy를 잃으며 document가 더 많았던 question일수록 더 잃는다(§3.2, Fig. 5). Relevant document가 100개 미만인 Natural Questions item은 strong supervised baseline에 닿으려면 log-linear fit이 parameter `10¹⁸` 초과를 extrapolate한다(§4.2).

**계산 예(derived).** 176B generator가 rare entity question(relevant document 약 10개)은 25%, common entity question(약 `10⁴`)은 55%로 answer한다고 하자. 각 kind question 1,000개를 만들고 correct answer만 keep하는 answer checker를 쓴다. Kept set에는 rare 250개, common 550개가 있으므로 rare share는 prompt의 50%에서 kept example의 31%(250/800)로 낮아진다. Checker가 없으면 rare example 1,000개 중 750개는 wrong answer다. 어느 선택이든 correct rare-fact supervision은 common fact에 비해 줄어든다. Generator accuracy는 위 BLOOM-176B number이며 generation accuracy가 QA accuracy와 같다는 것은 assumption이다.

**Recursion 없는 narrowing의 증거.** Shumailov Fig. 1 histogram은 first generation 뒤 이미 high-probability sequence 쪽으로 shift한다([[model-collapse]] Fig. 1). Real example 1024개 초과의 Gemma 2 2B SFT는 fixed real count에 synthetic example을 더하면 test loss가 오른다([[collapse-or-thrive]] §4). [[beyond-model-collapse]] 저자들은 perplexity-based self-selection의 성공을 easy-to-learn sample preference로 해석한다(§6.2, Interpretation). Library source 중 distillation SFT round 하나 전후 fact frequency별 accuracy를 재는 것은 없다. Single round tail loss 크기는 Open question이다.

**측정.** [[long-tail-knowledge]]처럼 pretraining corpus의 relevant-document count(§2)로, 또는 entity popularity로 evaluation question을 stratify한다. PopQA(Mallen et al.) 같은 popularity-stratified QA는 [[long-tail-knowledge]] §5가 QA accuracy/pretraining frequency concurrent work로 이름을 들지만 library card가 없어 여기 number는 인용하지 않는다. Low-resource language(ch-13a)를 포함한 language별, answer format별 accuracy를 보고한다. Synthetic round마다 같은 slice에서 전/후 checkpoint를 비교한다(ch-50).

**의미.** 범용 모델은 recursion하지 않는 pipeline에서도 rare knowledge/language/format을 잃을 수 있다. Average benchmark score는 common item이 지배하므로 check는 slice-level이어야 한다.

## §6 Post-training에서의 output-diversity loss

**정의.** Per-input diversity는 같은 prompt에 대해 sample한 output 여러 개의 diversity고, cross-input diversity는 서로 다른 prompt에서 output 하나씩의 diversity다([[rlhf-generalisation-diversity]] §5.2, Eq. 2–3).

**수식.** Diversity metric `D`, input당 sample `K`개, input `N`개에 대해:

`PerInput_D(π) = (1/N) Σ_i D({y_1..y_K ~ π(·|x_i)})`   `CrossInput_D(π) = D({y^(1)(x_1), …, y^(1)(x_N)})`

- `π`: policy, `y^(1)(x_i)`: input `x_i`의 first sample. Sentence-BERT metric은 sentence embedding의 mean pairwise cosine similarity에 `D = 1 −`을 적용한다. EAD는 length correction이 있는 distinct n-gram을 세고, NLI diversity는 contradiction/entailment를 센다.

**계산 예시.** Prompt 하나의 output 셋이 pairwise cosine similarity 0.90, 0.80, 0.85이면 mean은 0.85이므로 Sentence-BERT diversity는 0.15다(computed).

**증거.** LLaMA 7B summarization policy(`K = 16`, `N = 500`, temperature 1)에서 RLHF(PPO, `β_KL = 0.05`)는 SFT보다 EAD/Sentence-BERT per-input diversity가 낮고 더 작은 margin으로 cross-input diversity도 낮다. NLI diversity는 model을 분리하지 못하며 OPT model size 다섯 개도 같은 trend를 보인다(§6.2, Figs. 5–6, App. J.4). Generalization에서 LLaMA 7B RLHF policy는 summarization OOD(TL;DR→CNN/DailyMail, Fig. 2)와 instruction following에서 SFT보다 좋다. Instruction following에서 모든 model은 AlpacaEval에는 같은 정도로 generalize하고 RLHF advantage는 더 어려운 Sequential Instructions shift에서 크다(§6.1, Fig. 3). Diversity는 summarization policy에서만 측정했다. 저자들은 이를 generalization/diversity trade-off라 부른다(Abstract). **Model 간([[artificial-hivemind]]).** Open-ended query 100개에서 top-p 0.9/temperature 1.0으로 model 하나의 response 50개를 sample하면 25 model 전체의 model–query pair 중 79%에서 average pairwise similarity가 0.8을 넘는다. Min-p sampling(min-p 0.1, temperature 2.0)도 response pair 61.2%가 0.8 초과다. Model이 다른 response 간 average similarity는 0.71–0.82다(§3, Figs. 4–6). Study는 observational이고 cause를 test하지 않으며 shared data pipeline/synthetic-data contamination을 possible explanation으로 든다(§3). **RL에서([[echo-chamber-rl-post-training]]).** Math document와 solution format이 다른 instruction dataset 세 개로 pretrain한 150M model은 첫 PPO epoch 안에 format 하나로 간다. KL coefficient 0.001에서는 late training에 pass@64가 떨어지고 0.01에서는 comparable pass@1과 함께 stable하다(§3.1, Fig. 3).

**조건과 한계.** Kirk et al.은 diversity metric이 long instruction-following output에서 model을 분리하지 못했으므로 summarization에 대해서만 diversity를 보고한다(§6.2). Echo-chamber model은 parameter 150M과 1B다.

**Synthetic data에 대한 의미(Interpretation).** Per-input diversity가 낮은 post-trained teacher는 prompt당 distinct answer가 적은 synthetic set을 만들고 prompt당 response를 더 sample해도 비슷한 text가 나온다. Model output similarity가 0.71–0.82로 share될 때 teacher 하나를 다른 teacher로 바꾼다고 variety가 보장되지 않는다. Synthetic set 자체에서 per-input/cross-input diversity를 재고 같은 prompt의 human data와 비교해야 한다(selection metric은 ch-22, RL control은 ch-43).

## §7 Synthetic-data loop에서 log할 것

위 source는 다음 measurement를 뒷받침한다. 어느 것도 action threshold를 주지 않는다. Threshold는 pipeline own baseline run과 그 variance에서 정해야 한다(ch-51).

| Measurement | 탐지하는 것 | Method의 source |
|---|---|---|
| Round별 fixed real held-out set의 loss/perplexity | Recursive collapse(error 상승) | [[model-collapse]] Fig. 1; [[model-collapse-accumulation]] Table 2 |
| Round-0 model 아래 generated data의 per-sequence likelihood histogram | Typical output 쪽 shift와 긴 improbable tail | [[model-collapse]] Fig. 1 |
| Training pool의 real/synthetic count와 fraction | Replace vs accumulate; synthetic share `p₂` | [[model-collapse-accumulation]] §2; [[strong-model-collapse]] §3 |
| Labeled sample에서 verifier keep rate `ϕ`, `ψ`; `p*` | Selection이 correct label을 recover할 수 있는가 | [[beyond-model-collapse]] §4, §6 |
| Round 전반에서 verifier와 ground truth의 agreement | Verifier bias 쪽 drift | [[escaping-collapse-verification]] §4 |
| Fact frequency/language/answer format별 accuracy | Single round의 tail loss | [[long-tail-knowledge]] §3 |
| Per-input/cross-input diversity | Output narrowing | [[rlhf-generalisation-diversity]] §5.2 |
| Synthetic set과 다른 model output의 similarity | Teacher 간 shared output | [[artificial-hivemind]] §3 |
| Large `k`의 pass@1/pass@k | Solution coverage narrowing | [[echo-chamber-rl-post-training]] §3.1 |

## Negative sample과 negative feedback

**여기서 쓰는 네 의미**(course standard): (1) negative marginal value는 positive target으로 쓸 때 performance를 낮추는 sample, (2) content로서 negative는 input/corrected target에 failure를 넣고 cross-entropy로 train, (3) conditioning으로서 negative는 control token 아래 failure를 train, (4) gradient로서 negative는 sample likelihood를 명시적으로 낮춘다. Probability mass를 sample에서 제거하는 것은 (4)뿐이다. 이 section은 verifier reject를 다룬다. Discard하면 type (1), rejected response/negative advantage로 reuse하면 type (4)다.

**1. Negative의 출처.** Verifier reject는 wrong final answer([[openmathinstruct]]는 6.6M개를 release하고 train하지 않음, §1), parser failure, failed unit test/execution, timeout, truncated response, judge/reward-model rejection이다. Label error는 양방향이다. False positive: answer matching은 flawed reasoning의 right-answer solution을 keep하며 OpenMathInstruct는 rate 없이 rare라고 한다(§2.3). Agent는 우연히 test를 pass하고 unrelated file을 edit할 수 있다([[deepswe]] §2.3). False negative: Sky-T1 regex+sympy parser는 gpt-4o-mini judge가 73% keep한 correct math solution 중 25%를 keep했으므로 parser는 judge가 accept한 solution의 약 66%를 버렸다(`1 − 25/73`, derived)([[bespoke-stratos]] “Data Curation”). Too-hard question의 Monte-Carlo step label도 false negative를 만든다([[omegaprm]] App. A).

**2. 현재 관행이 하는 일.** SFT filter는 reject를 discard한다(type 1): [[beyond-model-collapse]], [[openmathinstruct]], [[bespoke-stratos]]. Qwen2.5는 execution feedback/answer matching에 실패한 response를 offline DPO pair의 rejected side로 쓴다(type 4)([[qwen-2.5]] §4.2). DeepSWE는 failed test에 주는 zero reward로 train하지 않고 context limit/20-minute timeout/step limit에 닿은 trajectory의 loss를 mask한다([[deepswe]] §2.3). Group baseline에서 group mean 아래 zero reward는 negative advantage(type 4)다.

**3. Mechanism.** Next token logit `z`의 softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. Step size `η`로 `log p_y`를 낮추는 gradient step은 `z_y`를 `−η(1 − p_y)`로, 다른 모든 `z_j`를 `+η p_j`로 바꾸므로 removed mass는 이미 가장 likely했던 token으로 주로 간다. 계산 예: probability `(a, b, y) = (0.70, 0.20, 0.10)`, `η = 1`에서 logit change `(+0.70, +0.20, −0.90)`는 new probability `(0.832, 0.144, 0.024)`를 준다(computed). Token b는 penalize되지 않아도 mass를 잃고 token a는 0.132를 얻는다. Penalized response가 이미 unlikely(small `p_y`)해도 같은 concentration이 적용된다.

**Error asymmetry(계산 예, derived).** Generator가 answer 1,000개를 만들고 400개가 correct라고 하자. Verifier는 correct answer 30%를 reject하고 wrong answer 5%를 keep한다. Kept pool은 correct 280개 + wrong 30개이므로 SFT target 9.7%가 wrong(type-1 harm)이다. Generator error `p = 0.6`은 `p* = 0.933` 아래라 §4 infinite-data theory에서 kept pool training은 여전히 성공한다. Reject pool은 correct 120개 + wrong 570개이므로 reject 17.4%가 correct다. Reject를 discard하면 correct example 120개를 잃는다. Reject를 type-4 negative로 쓰면 이 correct answer 120개의 likelihood를 낮추고 위 mechanism은 mass를 most likely alternative, 즉 typical wrong answer일 수 있는 쪽으로 옮긴다. False negative는 discard에서는 lost data이고 negative gradient에서는 correct behavior에 대한 active push다. [figures/collapse-iterations.html](figures/collapse-iterations.html)의 Panel D는 default setting으로 이 pool을 재현한다.

**4. 수치를 포함한 증거.** 이 chapter source는 verifier reject를 negative로 reuse하는 benefit을 discard와 비교해 측정하지 않는다. Type-4 negative failure mode: on-policy refusal pair의 DPO는 preferred refusal에서 probability가 멀어져 Llama-3-8B-Instruct refusal rate를 74.4%에서 33.4%로 낮췄다([[likelihood-displacement]] §6.2). SFT term을 더하면 plain DPO가 낮춘 preferred response log probability(mean change Gemma-2B-IT `−59.2`, Llama-3-8B-Instruct `−48.1`)를 올린다(`+20.2`, `+28.6`)(Table 16). Synthetic-data verification에서 negative/positive가 각각 improvement에 기여한 share는 측정하지 않는다(RL setting은 ch-43a 참고).

**5. Control.** Reject를 negative로 쓰기 전에 labeled sample에서 reason code(parse, execution, timeout, truncation, judge)별 verifier false-negative rate를 잰다. Timeout/truncation 같은 uncertain failure는 penalize하지 말고 mask한다([[deepswe]] §2.3). Reject를 wrong으로 label하기 전 second checker를 더한다([[bespoke-stratos]]). Positive NLL term으로 preference loss를 anchor한다([[likelihood-displacement]] Table 16). Negative는 on-policy로 유지하고 first wrong step에 localize한다(ch-24, ch-31a).

**6. Diagnostic.** Chosen/rejected log probability를 따로 log하고, round별 reason code로 reject를 log한다. Large `k`에서 pass@1/pass@k, per-input diversity, unanswerable prompt의 abstention rate를 track한다.

**7. Generality에 미치는 효과(Interpretation).** Parser false negative는 parser가 처리하지 못하는 answer format에 concentrate하므로 이를 penalize하면 model을 그 format에서 밀어내고 format coverage를 좁힌다. Judge reject는 judge preference가 human과 다른 곳에 concentrate한다. [[artificial-hivemind]]는 annotator disagreement가 있는 response에서 reward model/judge calibration이 낮다고 보고한다(Abstract). Same-policy pair loop가 narrow/broaden하는지는 [[escaping-collapse-verification]] §4가 말하듯 label source에 달려 있다. Model은 label이 accept하는 쪽으로 움직인다.

## 레시피

| Model(정확한 release) | 크기 | Stage | 설정 | 값 | Source 위치 | 상태 | 이 값의 증거 |
|---|---|---|---|---|---|---|---|
| OPT-125m (Shumailov et al.) | 125M | SFT | data; generation | wikitext2; five-way beam search, 64-token block당 64 token; training set과 같은 size generated set; run 5회 | Nature 631:755, “Model collapse in language models” | verified 2026-09-14 ([[model-collapse]]) | ablation 보고 없음 |
| OPT-125m | 125M | SFT | setting 1; setting 2 | epoch 5/original data 없음; epoch 10/generation마다 original random 10% | same, Fig. 1b–c | verified 2026-09-14 | Fig. 1b vs 1c |
| OPT-125m | 125M | SFT | LR; batch; optimizer; generation 수 | 미보고 | main text/Fig. 1 caption/SI §4 확인 | not reported | — |
| GPT-2; Llama-2 (Gerstgrasser et al.) | 9M; 12M, 42M, 126M | pretrain-stable | data; epoch; round sample | TinyStories 470M token; 1; temperature 1.0(0.3도)에서 round마다 new TinyStories-size sample; round마다 new initialization | arXiv:2404.01413v2 §2.1 | verified 2026-09-15 | Table 2: accumulate/replace/grown-dataset replace; temp 0.3 row |
| Same | 9M–126M | pretrain-stable | LR; batch; sequence length | 미보고 | §2.1, App. C 확인 | not reported | — |
| Gemma 2 (Kazdan et al.) | 2B, 9B, 27B | SFT | real data; round당 data | HelpSteer2; replace 약 12.5K; accumulate 약 `12.5K × t` | arXiv:2410.16713v4 §2.3 | verified 2026-09-15 | Fig. 3 |
| Gemma 2 2B | 2B | SFT | synthetic pool; mix | completion 100k, `>512` token 제거, 55,000 keep; real 128–12,000, synthetic 0–40,000 | v4 §4, Fig. 5 | verified 2026-09-15 | synthetic은 real `≤1024`에서만 loss 감소 |
| GPT-2-small (Dohmatob et al.) | 124M; 166M(18 layer); 204M(24 layer) | pretrain-stable | data; synthetic fraction; generation | BabiStories 2.2M story; `p₂ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0}`; temperature 1, top-p 1, quality-filtered | arXiv:2410.04840v2 §4.2, Fig. 8, App. A.3 | verified 2026-09-15 | Fig. 8 left(`p₂`)/right(`p₂ = 1` depth) |
| GPT-2-small | 124M | pretrain-stable | LR; dropout; weight decay; warm-up; context | `5×10⁻³`; 0.05; 0.1; iteration 2,000; token 512 | v2 App. A.3 | verified 2026-09-15 | ablation 없음 |
| Llama-2-7B generator (Feng et al.) | 7B | SFT | data; LR; schedule; epoch; batch; block | XLSUM English train 12.5%(307,000); `5e-5`; cosine; 1; 32; 1024 | arXiv:2406.07515v2 §6.2, App. D | verified 2026-09-15 | ablation 없음 |
| Llama-2-7B on selected data | 7B | distill-SFT | selection rate; LR; schedule; decoding | 12.5%, 25%, 50%; `2e-5`; constant; greedy | v2 §6.2, App. D | verified 2026-09-15 | Fig. 5: oracle > self-selection > generator; Llama-3 selection ≈ random |
| SmolLM2-135M (Yi et al.) | 135M | distill-SFT | data; round; verifier | XSUM 12.5%, epoch 1; round 15; oracle가 ROUGE-1 top 12.5% keep; greedy | arXiv:2510.16657v3 §5.3 | verified 2026-09-15 | Fig. 5: filtered vs unfiltered |
| LLaMA 7B RLHF (Kirk et al.) | 7B | RL | algorithm; KL coefficient/placement | PPO; `β_KL = 0.05`, SFT policy KL을 reward에서 subtract | arXiv:2310.06452v3 §4, Eq. 1 | verified 2026-09-15 | “our own early experiments”(§4); number 미보고 |
| LLaMA 7B diversity evaluation | 7B | eval-gate | sample/input; input; temperature | 16; 500; 1 | v3 §5.2 | verified 2026-09-15 | ablation 없음 |
| OLMo-codebase model (Zhao et al.) | 150M | RL | KL coefficient | default `1e-3`; 0/0.01 비교 | arXiv:2504.07912v2 App. C | verified 2026-09-14 ([[echo-chamber-rl-post-training]]) | 0.01은 comparable pass@1에서 second format/stable pass@64 유지 |
| Bespoke-Stratos-17k curation | — | distill-SFT | math correctness filter | regex+sympy parser 대신 gpt-4o-mini judge vs ground truth | blog “Data Curation” | verified 2026-09-14 ([[bespoke-stratos]]) | retained correct solution 25%(parser) → 73%(judge) |

2026-09-15 row는 [[model-collapse-accumulation]], [[collapse-or-thrive]], [[beyond-model-collapse]], [[escaping-collapse-verification]], [[rlhf-generalisation-diversity]]의 library card가 없고 [[strong-model-collapse]] card가 unverified라 그 날짜 primary arXiv PDF를 읽었다. Chapter excerpt에는 checked extract가 있다.

**작은 범용 run의 시작점.** Small LM multi-round self-training은 TinyStories run처럼 original real data+모든 earlier synthetic round로 매 round train하라(9M–126M, round당 epoch 1, temperature 1.0). Accumulate는 round-4 cross-entropy를 round 1 아래로 유지했고 replace는 올렸다. 7B verifier SFT는 Feng et al. reference를 쓴다. Pretrained model을 constant LR `2e-5`, epoch 1, greedy generation으로 fine-tune하고 measured `p*`로 selector를 고르며 rate 12.5%/25%/50%를 test한다. 이는 summarization task 하나에서 selector/rate만 바꿨다. Real SFT example이 1,024개 초과면 synthetic addition을 real-only control과 test하라. HelpSteer2 Gemma 2 2B는 length-filter synthetic data가 그 count 이상에서 거의 항상 test loss를 올렸기 때문이다. RL output diversity가 중요하면 150M echo-chamber처럼 pass@k에서 KL 0.001/0.01을 비교한다.

## 일반화 관점

**(a) 폭을 넓히는 것.**

- 매 round real data 유지: 126M TinyStories cross-entropy는 round 4에 1.71→1.59([[model-collapse-accumulation]] Table 2); Gemma 2 SFT replicate([[collapse-or-thrive]] Fig. 3).
- High breakdown-point selection: Llama-2-7B에서 oracle-selected summary 12.5%가 human summary 100% training을 이김([[beyond-model-collapse]] Fig. 5).
- Larger input pool selection: self-selection gain은 shifted input distribution으로 해석([[beyond-model-collapse]] §7, Interpretation).
- Real data scarce일 때 small synthetic addition: real `≤1024`에서 lower test loss([[collapse-or-thrive]] Fig. 5).
- Rare fact retrieval: relevant document가 적은 question에서 가장 큰 gain([[long-tail-knowledge]] §4.3).
- GPT-4 preference로 측정한 큰 distribution shift에서는 SFT보다 RLHF([[rlhf-generalisation-diversity]] §6.1).

**(b) Narrowing/forgetting을 일으키는 것.**

- Round마다 data replace: GPT-2 9M cross-entropy 1.82→2.91(round 10)([[model-collapse-accumulation]] Table 2); OPT-125m PPL 상승([[model-collapse]] Fig. 1).
- Label-shift data fixed share: theory error floor `p₂²c²`; GPT-2 delayed scaling([[strong-model-collapse]] §3.1, §4.2).
- Many round biased verifier: verifier center로 converge([[escaping-collapse-verification]] Theorem 4.1).
- Correctness와 track하지 않는 own likelihood selection: best-of-50 19.2% vs oracle 60.4%([[beyond-model-collapse]] Table 1).
- Per-input diversity를 낮추는 post-training([[rlhf-generalisation-diversity]] §6.2), low-KL RL format collapse([[echo-chamber-rl-post-training]] Fig. 3).
- Model family 간 similar output teacher([[artificial-hivemind]] §3).
- High false-negative verifier의 negative(Negative samples section; derived example).

**(c) 이 stage에서 측정하는 방법.**

- Round별 real held-out loss/per-sequence likelihood histogram(§7).
- 매 round 전후 relevant-document count/entity popularity/language/format별 accuracy([[long-tail-knowledge]] §2–§3; ch-13a; ch-50).
- Labeled sample verifier keep rate와 `p*`([[beyond-model-collapse]] §4).
- EAD/Sentence-BERT는 분리했지만 NLI diversity는 못했으므로 multi-metric per-/cross-input diversity([[rlhf-generalisation-diversity]] §6.2).
- Large `k` pass@k와 pass@1([[echo-chamber-rl-post-training]] §3.1).
- 알려진 error: diversity metric은 length-sensitive이며 long instruction-following output을 분리하지 못했다([[rlhf-generalisation-diversity]] §6.2). Relevant-document pipeline은 TriviaQA 300-question sample에서 precision 약 60%([[long-tail-knowledge]] §2), GPT-4 preference judgment에는 judge bias가 있다(ch-49).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| Earlier data를 round synthetic data로 replace | Round마다 real held-out loss 상승 | Accumulate round 하나를 matched round에서 비교([[model-collapse-accumulation]] Table 2) |
| “Accumulate가 collapse 방지”를 “synthetic이 real처럼 scale”로 읽음 | Synthetic share가 커질수록 loss plateau | Real/synthetic count에 loss를 따로 fit([[strong-model-collapse]] Eq. 12; [[collapse-or-thrive]] §4) |
| Generated-data likelihood만 track | Generated text는 typical해지고 real loss는 상승 | Fixed real held-out set에서 매 round eval([[model-collapse]] Fig. 1) |
| Available model 중 strongest를 verifier로 선택 | Selected data가 random보다 나은 model을 못 train | Generator own output에서 `ϕ`,`ψ`,`p*` 측정([[beyond-model-collapse]] §6.2) |
| Correctness를 track하지 않는 generator likelihood로 select | Kept example accuracy가 greedy output보다 높지 않음 | Sample oracle check: best-of-k vs lowest-perplexity accuracy([[beyond-model-collapse]] Table 1) |
| Ground-truth audit 없이 verified round 실행 | Early gain이 stop/reverse | Verifier acceptance만이 아니라 held-out ground truth로 round별 score([[escaping-collapse-verification]] §4) |
| Parser reject를 negative gradient로 reuse | Unusual format correct answer의 likelihood 하락 | Labeled reject sample에서 reason-code FN rate([[bespoke-stratos]]) |
| Timeout/context-limit/step-limit trajectory를 failure로 score | Multi-turn agent RL training reward collapse(Qwen3-14B DeepSWE ablation) | 이 case를 mask하고 reward curve 비교([[deepswe]] §2.3, Fig. 6) |
| Distillation 뒤 average만 보고 | Rare fact/low-resource language regression을 나중에 발견 | 전후 frequency/language slice([[long-tail-knowledge]] §3) |
| Low-diversity teacher에서 prompt당 response를 더 sample | Near-duplicate response 다수 | Synthetic set per-input embedding similarity([[artificial-hivemind]] Fig. 4) |
| Single-round production pipeline을 verification이 collapse를 막는 증거로 인용 | Across-round measurement 없음 | 결론 전 round-by-round real-data evaluation 요구 |

## 이해를 확인해 보세요

1. Accumulation paper Table 2에서 accumulated size로 grown한 dataset replace도 degrade한다. 이것이 accumulation advantage의 어떤 설명을 배제하고 linear-regression proof는 대신 무엇을 주는가?
2. Accumulation은 error를 bounded로 유지하지만 Strong Model Collapse는 fixed synthetic share가 error 하락을 막는다고 한다. §2/§3의 두 formula를 써 둘 다 참인 이유를 설명하라.
3. `p* = 1/(1 + ψ/ϕ)`로 correct label을 많이 reject하는 verifier도 near-optimal training을 허용할 수 있지만 wrong label을 modest share keep하는 verifier는 실패하는 이유를 설명하라.
4. Generator보다 ROUGE-1이 높은 fine-tuned Llama-3가 Llama-2 generator data를 random과 비슷하게 select했다. 어떤 verifier property가 중요함을 보이며 training 전 어떻게 재는가?
5. Verified retraining knowledge-ball에서 selectivity가 convergence speed만 바꾸고 limit을 안 바꾸는 이유와 reward model 하나로 verify한 loop에 대한 limit의 뜻은?
6. Verifier false positive가 없어도 verified distillation round 하나가 rare fact correct supervision을 줄이는 이유를 long-tail accuracy number로 설명하라.
7. `∂ log p_y / ∂ z_j = 1[j = y] − p_j`를 써 verifier false negative를 negative gradient로 reuse하는 것이 discard보다 왜 나쁘고 removed probability가 어디로 가는지 설명하라.
8. RLHF는 SFT보다 OOD generalization이 좋지만 per-input diversity는 낮다. General-purpose student용 synthetic SFT set generator로 RLHF-tuned model을 쓸 때의 뜻은?

## 연결

- 이전: ch-22 — Quality, Diversity, and Gradient-Based Data Selection. 거기의 selection metric은 fixed pool에 작동하고 이 장은 model이 pool을 regenerate할 때를 다룬다.
- 다음: ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data. Answer checking/first-wrong-step negative가 §4/negative-sample section을 trace에 적용한다.
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix(verification 위치).
- ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining(long-tail knowledge); ch-13a — Multilingual Coverage and Vocabulary as Capability Axes(language slice).
- ch-29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification(§7 적용).
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation; ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity; ch-43 — Entropy, Output Diversity, and KL Control in RL(§6).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages(type-4 full derivation).
- ch-44 — Process Supervision and Verifiable Rewards; ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing(verifier coverage).
- ch-45 — Self-Improvement Loops and Multi-Stage Reasoning Pipelines(RL multi-round loop).
- ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting; ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing; ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.

## 출처

- [[model-collapse]] — definition/error source/discrete·Gaussian theory/OPT-125m setting/Fig. 1.
- [[model-collapse-accumulation]] — replace/accumulate definition, Table 2 cross-entropy, linear-regression Theorem 2.
- [[collapse-or-thrive]] — Gemma 2 SFT replication, accumulate-subsample, real-data cardinality result.
- [[strong-model-collapse]] — label-shift setting, Corollary 1 floor, MNIST/GPT-2 experiment, size/mixing result.
- [[beyond-model-collapse]] — generator best-of-k/self-selection, breakdown point `p*`, eigenvalue/summarization experiment.
- [[escaping-collapse-verification]] — verifier knowledge-ball model, verifier center convergence, XSUM round.
- [[long-tail-knowledge]] — relevant-document count별 accuracy, counterfactual retraining, retrieval.
- [[rlhf-generalisation-diversity]] — per-/cross-input diversity metric, RLHF generalization/diversity result.
- [[artificial-hivemind]] — open-ended output intra/inter-model similarity, judge calibration.
- [[echo-chamber-rl-post-training]] — RL format collapse와 KL-coefficient comparison.
- [[bespoke-stratos]] — parser false negative(retained correct solution 25% vs 73%).
- [[openmathinstruct]] — released wrong solution, rare로 설명한 flawed-reasoning positive.
- [[omegaprm]] — too-hard question의 false-negative step label.
- [[deepswe]] — timeout/context-limit/step-limit trajectory mask; chance test pass.
- [[qwen-2.5]] — failing response를 rejected DPO sample로 사용.
- [[likelihood-displacement]] — DPO 아래 preferred response displacement와 SFT term effect.
- [[nemotron-4-synthetic]] — multi-round weak-to-strong synthetic alignment; RM refresh claim 정정.
- [[apigen]] — ablation/pass-rate/ranking claim 정정.
- [[prismatic-synthesis]] — G-Vendi/distillation claim 정정.
