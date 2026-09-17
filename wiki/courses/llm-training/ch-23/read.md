<!-- chapter: ch-23
     track: synthetic
     kind: content
     title: Model Collapse and Verification of Synthetic Data
     deps: [ch-22]
     sources: [[model-collapse]], [[model-collapse-accumulation]], [[collapse-or-thrive]], [[strong-model-collapse]], [[beyond-model-collapse]], [[escaping-collapse-verification]], [[long-tail-knowledge]], [[rlhf-generalisation-diversity]], [[artificial-hivemind]], [[echo-chamber-rl-post-training]], [[bespoke-stratos]], [[openmathinstruct]], [[omegaprm]], [[deepswe]], [[qwen-2.5]], [[likelihood-displacement]], [[nemotron-4-synthetic]], [[apigen]], [[prismatic-synthesis]]
     figures: figures/collapse-iterations.html
     revised: 2026-09 (generality revision)
-->

# Chapter 23 — Model Collapse and Verification of Synthetic Data

> **Core insight.** When each model generation is trained only on the previous generation's outputs, low-probability content is lost first and error grows: OPT-125m fine-tuned this way on wikitext2 has rising perplexity on the real test set across generations ([[model-collapse]] Fig. 1). The outcome depends on how data are used, not only on whether data are synthetic: keeping the real data and adding each round's synthetic data kept test error bounded without any filter, for language models from 9M to 126M parameters and for Gemma 2 SFT ([[model-collapse-accumulation]] Table 2; [[collapse-or-thrive]] Fig. 3), but in a regression theory a fixed non-vanishing share of shifted data still stops test error from falling with more data ([[strong-model-collapse]] Corollary 1). A verifier whose keep rate for wrong labels is low relative to its keep rate for correct labels raises the generator error rate at which training still succeeds ([[beyond-model-collapse]] Theorem 4.2); with an oracle selector, 12.5% of a Llama-2-7B generator's summaries trained a better model than all human summaries (Fig. 5). Over many rounds the model moves toward what the verifier accepts, not toward the truth ([[escaping-collapse-verification]] Theorem 4.1).
>
> **Guideline.** When a pipeline retrains on its own outputs for more than one round, keep the original real data and earlier rounds in the training set instead of replacing them, because with samples at temperature 1.0, replace raised TinyStories test cross-entropy by 0.53 to 0.66 at iteration 4 while accumulate lowered it ([[model-collapse-accumulation]] Table 2). When synthetic labels are kept for training, measure the verifier's keep rates for correct and wrong labels on a labeled sample and compute the breakdown point `p* = 1/(1 + ψ/ϕ)`, because a stronger model was a worse selector than the generator itself in the summarization experiment ([[beyond-model-collapse]] §6.2). When the verifier's rejects are reused as negatives, measure its false-negative rate first, because an answer parser discarded about two thirds of correct math solutions that a model judge accepted ([[bespoke-stratos]] "Data Curation", derived). Otherwise, discard or mask uncertain rejects. For a general-purpose model, report accuracy by fact frequency and output diversity per input before and after each synthetic round, because average scores do not show tail loss ([[long-tail-knowledge]] §3; [[rlhf-generalisation-diversity]] §6.2).

## Corrections to the version you studied

1. "by generation ~9 on OPT-125M, outputs are incoherent" and "rare-token perplexity spikes while average perplexity looks fine" → the paper shows generation-9 text with a repeated phrase pattern and reports mean perplexity on the original wikitext2 test set rising across generations; rare-token perplexity is not measured ([[model-collapse]], Example 1, Fig. 1 caption).
2. "The trap: average PPL improves generation-to-generation while tails disappear" and "Mean PPL … moves the wrong way under collapse (down …)" → real-test perplexity increases; what moves toward lower values is the perplexity of generated sequences scored by the generation-0 model, together with a longer high-perplexity tail ([[model-collapse]], Fig. 1 caption).
3. The "Tail-loss progression" table (average PPL 34.1 → 31.4, rare-token PPL 412 → >10⁴, "Figure 3-class numbers") → no such table exists; the reported numbers are 34 mean perplexity after real-data fine-tuning (115 zero-shot) and a loss "from 20 to 28 perplexity points" without original data ([[model-collapse]], "Model collapse in language models").
4. "generate 100k tokens with temperature 1.0 … iterate up to 10 generations" → five-way beam search predicting 64 tokens for each 64-token block, with a generated set the size of the training set; the number of generations is not stated ([[model-collapse]], same section, Verification).
5. "pure replacement (10% real = 0) and accumulation (10% real persistent …)" and "~30% worse than real-only" → the second setting trains 10 epochs and samples a random 10% of the original data at each generation; it is not accumulation, which comes from [[model-collapse-accumulation]]; no "30% worse" figure is reported ([[model-collapse]], Fig. 1c).
6. "statistical sampling error", "converges to a mode-collapsed near-Gaussian", the weight recursion `Var[ŵ_k^(n)] ≈ n·w_k(1−w_k)/N` with "n ≈ 100/N·(N/100) ≈ O(N/100)", and "`Var[μ_k^(n)] ≈ n·σ²/N`" → the term is statistical approximation error; the limit is a delta function or a low-variance point estimate; the 1D Gaussian result is `Var(X_j^n) = σ²(1 + n/M) + O(2)`; the weight recursion is not in the paper, and (100/N)·(N/100) = 1 ([[model-collapse]], Definition 2.1, Theorem 3.1, SI §3.1.1).
7. "even 1% synthetic contamination introduces an irreducible bias term `c(p)·σ²`" → in ridge regression with a fixed label-shifted synthetic distribution, the floor is `p₂²c²` for small `d/n`, where `c²` measures label mismatch; the 1% figure is a theoretical statement in the abstract ([[strong-model-collapse]], arXiv v2 Corollary 1, Eq. 12).
8. "Empirically reproduced on GPT-2-scale LM training with 1% synthetic injection: the scaling curve … never recovers" → with quality-filtered synthetic BabiStories, synthetic data "delay the progression of the scaling laws"; the authors expect plateaus and do not show one ([[strong-model-collapse]] §4.2, Fig. 8).
9. "the theoretical assumption … is precisely that synthetic is iid from an earlier model; a verified corpus … does not satisfy that assumption" → synthetic data there means any data from a distribution that deviates from the test distribution, so verified data with non-zero label mismatch is inside the theory ([[strong-model-collapse]] §1.1); iterated verified retraining converges to the verifier's knowledge center ([[escaping-collapse-verification]] Theorem 4.1).
10. "Collapse is … the default outcome of any synthetic pipeline that lacks a distribution-preserving gate" and "Pipelines without gates collapse on the timescale of 3–10 generations regardless of model scale" → accumulation keeps error bounded without a verifier in language models (9M–126M) and linear regression ([[model-collapse-accumulation]] Fig. 2, Table 2, Theorem 2), replicated in Gemma 2 SFT ([[collapse-or-thrive]] Fig. 3); model size changes the collapse profile ([[strong-model-collapse]] Theorem 2, Fig. 8).
11. "Token-level re-sampling (Zhu et al. 2025) … from the true distribution … at every decoding step" → Zhu et al. propose token-level editing of human-produced data to obtain semi-synthetic data (arXiv:2412.14689v3, Abstract; no library card).
12. "Rephrased synthetic survives scaling up to ~30% share" and "the empirical optimum converges near 2:1 real:synthetic" → no verified source in the library states this (the synthetic-data-scaling-laws card has no verification record); the verified mixing result here is narrower: for Gemma 2 2B SFT on HelpSteer2, synthetic examples lowered test loss only when real examples numbered 1024 or fewer ([[collapse-or-thrive]] §4, Fig. 5).
13. "Zhang et al. 2025 … analytical convergence guarantee (iterated training with a reliable verifier stays bounded)" → the authors are Yi, Liu, Cheng, and Xu; iterated verified retraining converges to the verifier's knowledge center, and early gains plateau or reverse unless the verifier is unbiased ([[escaping-collapse-verification]], Abstract, §4).
14. "Memorization↔generalization drift ('Closer Look at Model Collapse,' 2025)" as a language-model probe → that paper studies image diffusion models retrained on their own samples (arXiv:2509.16499v3, Abstract; no library card).
15. Nemotron-4 "periodic RM refresh from fresh human preferences", "refreshes per-generation", "τ tuned to ~80% agreement", and the quoted "Reward-model errors compound …" → none is in the report; the reward model is trained on 10K HelpSteer2 examples, and each aligned 340B intermediate model generates the next round's data ([[nemotron-4-synthetic]], §3.1, §3.2.4, Verification).
16. APIGen "Remove format: −18% … execution −11% … semantic −6% BFCL-V1", the layer rejection rates, "5-sec sandbox", "<3% hallucination rate", "xLAM-7B #1 BFCL <13B", "Overall acceptance ≈ 60%" → an add-back ablation with deltas −4.06/−5.94 (7B) and −9.59/−12.17 (1B) and no format arm; pass rates 34.42%–84.15% by generator; xLAM-7B ranks 6th with 85.65; no timeout or hallucination rate is reported ([[apigen]], Tables 1–2, Fig. 5, Verification).
17. "Math: ground-truth answer matcher … 0% false-positive rate; rejects ~60% of raw generations" → answer matching keeps right-answer solutions with flawed reasoning, which OpenMathInstruct calls rare without giving a rate ([[openmathinstruct]] §2.3); answer parsers also produce false negatives ([[bespoke-stratos]], "Data Curation"); no 60% rejection rate is reported.
18. The gate-versus-no-gate table rows for Nemotron-4, APIGen, and Prismatic Synthesis ("no collapse") and "every pipeline that scales without collapsing has a gate" → these reports do not measure error across repeated self-training generations, so they are not evidence about collapse ([[nemotron-4-synthetic]], [[apigen]], [[prismatic-synthesis]]).
19. G-Vendi "random-project to ~8K dims" and "7B beats 671B teacher" → projection dimension 1024; PrismMath-7B, distilled from R1-Distill-Qwen-32B, beats R1-Distill-Qwen-7B on 6 of 7 benchmarks ([[prismatic-synthesis]] §2.1, Table 4).
20. Drift thresholds (rare-token recall "Δ > 5%", "<75%", rare 5-gram "<50%", G-Vendi "Δ > 10%", cluster occupancy "Δ > 20% over 3 gens", "k = 1000") and the four-axis audit attributed to the `faithful-synth-eval` card → no source gives these thresholds, and that card has no verifiable primary source; they are removed.
21. "DPO on preference pairs where both chosen and rejected come from the same policy with no independent signal" listed as unavoidable collapse → no cited source tests this case; in the verification results the variable that sets the outcome is the label source, not whether both samples share a policy ([[beyond-model-collapse]] §6.2; [[escaping-collapse-verification]] §4).
22. "ch-24–27 … reasoning traces, conversation, tool calls, preference pairs" → ch-27 is "Agentic Trajectory Data"; preference-pair construction is covered in ch-39 "Offline Preference Optimization: DPO and Its Variants" and ch-41 "Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization" (outline).
23. Figure presets "10% accumulate" and a "verifier" that rejects by distance to the true components → the first mislabels Shumailov's per-generation 10% sampling; the second builds the verifier's success into the simulation; the figure is rewritten.

Section map from the old version: §1 → §1; §2 → §3; §3 → §2 and §4; §4 → §4 and §7; §5 removed (see corrections 16–18); §6 → §4 and "Negative samples and negative feedback"; §7 → §7. §5 and §6 of this version are new.

## Why this chapter matters for a general-purpose model

Synthetic data enters every stage after pretraining: rephrased documents in mid-training, distilled instructions and traces in SFT, model-ranked pairs in preference optimization, and self-generated rollouts in RL. Two measurable risks follow for breadth. The first is recursive collapse: when a model trains on outputs of earlier models over several rounds, error on real data grows ([[model-collapse]]). The second is single-round narrowing: one round of distillation or post-training can lower output diversity or tail coverage without any recursion ([[rlhf-generalisation-diversity]]; [[artificial-hivemind]]). Both costs fall on content that is rare in the training data: facts that appear in few documents ([[long-tail-knowledge]]), low-resource languages (ch-13a), and uncommon answer formats ([[echo-chamber-rl-post-training]]). The authors of the Nature paper note that low-probability events are often relevant to marginalized groups ([[model-collapse]] "Discussion"). This chapter separates the two risks, states what the theory and experiments show about replacing, accumulating, and verifying data, and gives measurements for each.

## §1 Recursive training on model outputs: what Shumailov et al. measured

**Definition.** Model collapse is a degenerative process in which data generated by one model generation pollute the training data of the next, so that later models misperceive the original distribution ([[model-collapse]] Definition 2.1). In early collapse the tails of the distribution are lost; in late collapse the model converges to a distribution with little resemblance to the original, often with much lower variance.

**Problem.** A model trained on generated data can match the generator on common content and still lose low-probability content. The measurable question is how error on real test data changes across generations.

**Mechanism.** The paper names three error sources that compound across generations:
1. Statistical approximation error: a finite sample under-represents low-probability events. This is the primary source.
2. Functional expressivity error: the model class cannot represent the true distribution.
3. Functional approximation error: training does not find the best model in the class.

In the paper's discrete model with an exact fit (no errors of type 2 or 3), each generation draws M samples from the current distribution and sets the next distribution to the empirical frequencies. A state with probability `q ≤ 1/M` has fewer than one expected sample per generation. Once a state receives zero samples it has probability zero in every later generation, so the chain converges to a delta function with probability 1 ("Discrete distributions with exact approximation").

**Formulas.** For recursive fitting of a 1D Gaussian with sample size M (SI §3.1.1):

`Var(X_j^n) = σ² (1 + n/M) + O(2)`

- `X_j^n`: a sample at generation n; `σ²`: variance of the original distribution; `M`: samples per generation; `O(2)`: terms of second order in 1/M.

For multidimensional Gaussians with a fixed sample size, the fitted covariance converges to 0 almost surely and the expected squared Wasserstein-2 distance to the original data diverges (Theorem 3.1).

**Worked example.** Discrete model, M = 1,000, one state with q = 0.0005. The expected count is 0.5, and the probability of zero samples in one generation is `(1 − 0.0005)^1000 = 0.606` (computed). With probability 0.606 that state is absent from generation 1 and from all later generations. Gaussian model, σ² = 1, M = 100, n = 10: the marginal variance of a generation-10 sample is `1 × (1 + 10/100) = 1.1`, because the fitted parameters follow a random walk ([[model-collapse]] SI §3.1.1).

**Evidence (Result, single study).** OPT-125m fine-tuned on wikitext2 reaches 34 mean perplexity from a zero-shot 115. Each generation generates a dataset of the same size by five-way beam search (64 tokens per 64-token block) and the next model is fine-tuned on it; 5 runs per setting. With 5 epochs and no original data, the authors report "losing some performance, from 20 to 28 perplexity points" (Fig. 1b). With 10 epochs and a random 10% of the original data sampled each generation, degradation is "only minor" (Fig. 1c). The Fig. 1 caption states that mean perplexity on the original test set increases across generations, that later generations produce more sequences the generation-0 model finds probable, and that a longer tail of sequences it would never produce appears. A repetition penalty of 2.0 doubles perplexity and does not remove the effect ("Ablation: Repetitions"). The same qualitative loss appears in Gaussian mixtures and VAEs (SI §4).

**Conditions and limits.** The LLM experiment uses one 125M model, fine-tuning rather than pretraining, beam-search generation, and replacement of data. Learning rate, batch size, optimizer, and the number of generations are not reported (card Verification). No verifier or filter is applied, and no larger model is tested.

**Implication.** Mean perplexity on real held-out data does detect this form of collapse. Tail loss is visible in the distribution of per-sequence scores, so a loop that reports only an average generation score can miss the shift toward typical outputs. Panel A of [figures/collapse-iterations.html](figures/collapse-iterations.html) simulates the discrete model and lets the reader compare how many low-probability states survive under replace and accumulate for chosen M and number of generations.

## §2 Replace, accumulate, and fixed-size subsets

**Definitions** ([[model-collapse-accumulation]] §2, Fig. 1). **Replace**: model n trains only on data sampled from model n−1. **Accumulate**: model n trains on the original real data plus every synthetic dataset from rounds 1 to n−1. **Accumulate-subsample**: the pool accumulates, but each round trains on a fixed-size subsample ([[collapse-or-thrive]] §3).

**Problem.** The OPT-125m setting 1 in §1 uses replace, in which real data disappear after round 1. Web corpora grow over time instead: Llama 1, 2, and 3 were trained on 1.4T, 2T, and 15T tokens ([[model-collapse-accumulation]] §1). The question is whether collapse still occurs when data accumulate.

**Mechanism (linear regression, Theorem 2).** Features x ~ N(0, I_d), labels y = x·w* + ε with ε ~ N(0, σ²). Round 1 fits OLS on T real samples. Each later round relabels the same inputs with the previous fit plus fresh noise, then refits on the chosen data.

`E_test^Replace(ŵ_n) = σ²d/(T − d − 1) × n`

`E_test^Accum(ŵ_n) = σ²d/(T − d − 1) × Σ_{i=1..n} 1/i²  ≤  σ²d/(T − d − 1) × π²/6`

- `E_test`: expected squared error on the true distribution minus the noise floor (Eq. 1); `σ²`: label noise variance; `d`: input dimension; `T`: samples added per round (T ≥ d + 2); `n`: round index.

Under accumulate, round i contributes a 1/i share of the data, so its noise enters the squared error with weight 1/i², and the sum of 1/i² is finite (§3.2).

**Worked example.** σ² = 1, d = 10, T = 111 gives `σ²d/(T − d − 1) = 10/100 = 0.1`. After 10 rounds, replace gives 0.1 × 10 = 1.0. Accumulate gives 0.1 × (1 + 1/4 + … + 1/100) = 0.1 × 1.550 = 0.155, below the bound 0.1 × 1.645 = 0.164 (computed). Panel B of [figures/collapse-iterations.html](figures/collapse-iterations.html) plots both curves for any σ², d, and T.

**Evidence.** GPT-2 (9M) and Llama-2 (12M, 42M, 126M) are pretrained from scratch for one epoch on TinyStories (470M tokens); each round samples a TinyStories-sized dataset at temperature 1.0 or 0.3 ([[model-collapse-accumulation]] §2.1). Evaluation cross-entropy (Table 2):

| Model | Round 1 | Round 4, accumulate | Round 4, replace | Round 10, replace | Round 4, replace with grown dataset |
|---|---|---|---|---|---|
| GPT-2 9M | 1.82 | 1.74 | 2.39 | 2.91 | 2.18 |
| GPT-2 9M, temperature 0.3 | 1.82 | 1.75 | 5.82 | 9.85 | not run |
| Llama-2 12M | 2.06 | 1.94 | 2.72 | not run | not run |
| Llama-2 42M | 1.90 | 1.76 | 2.52 | not run | not run |
| Llama-2 126M | 1.71 | 1.59 | 2.23 | not run | not run |

The last column shows that replace degrades even when its synthetic set is as large as the accumulated set, so dataset size alone does not explain the difference. **Replicated:** Gemma 2 2B, 9B, and 27B fine-tuned on HelpSteer2 (about 12.5K examples per round) collapse under replace and do not under accumulate ([[collapse-or-thrive]] §2.3, Fig. 3). Accumulate-subsample lies between the two in five settings and typically plateaus ([[collapse-or-thrive]] §3, Fig. 4).

**Conditions and limits.** The language-model runs use models up to 126M parameters, no filtering, and TinyStories, which is itself GPT-3.5/4-generated. A VAE on CelebA still degrades under accumulate, more slowly than under replace, and loses minor attributes such as glasses ([[model-collapse-accumulation]] §2.3). "Bounded" is not "improving with more data": the accumulate bound is at most 1.645 times the round-1 error in the linear model. For fine-tuning, the real-data count matters: with Gemma 2 2B on HelpSteer2, synthetic examples lowered test loss when real examples numbered 1024 or fewer, and above that count more synthetic data almost always raised it at a fixed real count ([[collapse-or-thrive]] §4, Fig. 5). What curated web-scale pretraining pipelines have observed across model generations is not reported in these sources (Open question).

**Implication.** A multi-round pipeline should keep its original human data and earlier rounds in the pool. This controls divergence. It does not by itself make synthetic data as useful as real data, which is the subject of §3.

## §3 Strong model collapse: a fixed share of shifted data in a scaling regime

**Definition.** Strong model collapse is the result that, when a non-vanishing fraction of training data comes from a distribution that differs from the test distribution, test error stops decreasing toward the clean-data value as the total data size grows ([[strong-model-collapse]] §1.1, §3.1). "Synthetic" in this paper means any data from such a shifted distribution; it is a fixed mixture, not a recursive loop.

**Problem.** §2 shows that accumulation prevents divergence. A scaling-law pipeline needs more: error that keeps falling as data grow.

**Mechanism.** Real samples follow y = x·w₁* + ε; synthetic samples follow y = x·w₂* + ε with the same input distribution. The label mismatch δ = w₂* − w₁* has covariance Δ, and the quality of the synthetic data is `c² = (1/d) tr ΣΔ` (Definition 1). A ridge model is trained on the union.

**Formula** (Corollary 1, Eq. 12; isotropic features, unregularized limit, small d/n; the paper writes d/n as ϕ, a different quantity from the keep rate ϕ in §4):

`E_test ≃ σ² d/n + p₂² c² + O(ϕ²)`

- `σ²`: label noise; `d`: dimension; `n`: total training samples; `p₂ = n₂/n`: synthetic fraction; `c²`: label mismatch of the synthetic distribution.

The first term falls as n grows; the second does not depend on n. "The scaling law plateaus unless p₂ → 0+" (§3.1).

**Worked example.** σ² = 1, d = 100, c² = 1. Real data only: n = 10⁴, 10⁶, 10⁸ gives 0.01, 0.0001, 0.000001. With p₂ = 0.01: 0.0101, 0.0002, 0.000101. At n = 10⁸ the mixed model's error is about 101 times the real-only error, and the floor 0.0001 is set by p₂²c² (computed). With p₂ = 0.1 the floor is 0.01. Panel C of [figures/collapse-iterations.html](figures/collapse-iterations.html) draws these curves for any d and c².

**Evidence.** Two-layer networks on MNIST with synthetic labels from another network show scaling that slows and plateaus as p₂ grows, subsiding only as p₂ → 0 (§4.1, Fig. 7). In language modeling, GPT-2-small (124M) trained on mixes of BabiStories and GPT-2-generated stories (quality-filtered, temperature 1) shows that "even moderate amounts of synthetic data delay the progression of the scaling laws"; the authors "expect this to eventually lead to plateaus" (§4.2, Fig. 8). A plateau at 1% synthetic data is not shown in the language-model runs. Model size matters: with p₂ = 1, 18- and 24-layer models have lower loss than the 12-layer model below 1×10¹⁰ tokens and higher loss beyond 3×10¹⁰ tokens (§4.2). Weighted single-step mixing does not remove the floor for a fixed weight; iterative mixing with half real data recovers a scaling law but underperforms training on the real half alone (§5, Corollaries 2–3, Fig. 10).

**Conditions and limits.** The theory is linear regression and random projections with Gaussian inputs and a label shift of fixed size. Filtering is outside the analysis (§1.2). Accumulation in §2 and strong collapse here answer different questions: bounded error across rounds versus continued improvement with more data (Interpretation).

**Implication.** For a general-purpose model, synthetic data whose labels differ systematically from the truth, even slightly, sets an error floor on the affected tasks that more data does not remove. Reducing c², by verification (§4) or by rephrasing real content, is the lever; reducing p₂ is the other.

## §4 Verification under synthesis: breakdown point and long-run convergence

**Definition.** A verifier is a procedure that decides, per synthetic example, whether to keep it: an answer checker, unit tests, a reward model, a judge model, or a person.

**Problem.** A generator can produce better answers than it selects. For a transformer predicting eigenvalues at 1% tolerance, the best of 50 beams chosen by an oracle is correct 60.4% of the time, the model's own lowest-perplexity beam 19.2%, and greedy decoding 20.2% ([[beyond-model-collapse]] §3, Table 1). The model's likelihood does not identify its correct outputs.

**Mechanism and formula (single round, [[beyond-model-collapse]] Theorem 4.2).** Let `p = P(synthetic label ≠ true label)`, `ϕ = P(keep | label correct)`, and `ψ = P(keep | label wrong)`. For a linear classifier trained on the kept examples in the infinite-data limit:

`p* = 1 / (1 + ψ/ϕ)`;  accuracy → 100% if p < p*, and → 0% if p > p*.

**Worked example.** No filter (ϕ = ψ = 1): p* = 0.5, so a generator wrong on more than half of its labels teaches the opposite rule. A verifier with ϕ = 0.9 and ψ = 0.3: p* = 1/(1 + 1/3) = 0.75. A verifier that rejects 30% of correct labels and keeps 5% of wrong ones (ϕ = 0.7, ψ = 0.05): p* = 0.933 (computed). A high false-negative rate costs data, but it moves p* less than a high false-positive rate does, because p* depends on the ratio ψ/ϕ. Panel D of [figures/collapse-iterations.html](figures/collapse-iterations.html) computes p* and the composition of kept and rejected pools for any rates.

**Evidence (Result, single study).** Eigenvalues: without verification (p* = 0.5), 10M synthetic examples, 50 times the generator's training set, train a model worse than the generator; the oracle verifier nearly doubles accuracy; the curve crosses the generator near p* ≈ 0.65 (§6.1, Fig. 4). News summarization with a Llama-2-7B generator fine-tuned on 12.5% of XLSUM English: random selection at matched size scores below the generator; oracle selection beats the generator in every setting, and oracle selection of 12.5% beats training on 100% of the human summaries; self-selection by perplexity beats the generator; selection by a fine-tuned Llama-3 with higher ROUGE-1 than the generator is close to random (§6.2, Fig. 5). The authors attribute the Llama-3 result to Llama-2's generated data being less correlated with Llama-3, and the self-selection gain to selecting inputs from a larger pool of articles (§6.2, §7; Interpretation). In a finite-data simulation, a weak verifier has a best data size near 10,000 examples, beyond which more selected data is worse (§5.2, Fig. 3).

**Many rounds ([[escaping-collapse-verification]]).** The verifier is modeled as a knowledge ball `B_r(θ_c)` around a center `θ_c` that contains the true parameter θ*; it accepts (x, y) when `|y − xᵀθ_c| ≤ r‖x‖ + σ_c` (§2, Eq. 1). One round can lower error through a bias-variance trade-off (Theorem 3.1). Over rounds the estimate converges to `θ_c`, with selectivity r changing the rate but not the limit (Theorem 4.1). An unbiased verifier gives continued improvement; a mildly biased one gives early gains that plateau or reverse; a strongly biased one degrades (§4). SmolLM2-135M on XSUM with an oracle ROUGE-1 verifier keeping the top 12.5% improves in early rounds and then stabilizes within the 15 reported rounds, while unfiltered retraining fluctuates around its initial score (§5.3, Fig. 5).

**A production multi-round loop.** Nemotron-4-340B alignment used over 98% synthetic SFT and preference data, a reward model trained on 10K HelpSteer2 examples as judge, and rounds in which each aligned 340B intermediate model generated the next round's data ([[nemotron-4-synthetic]] §3.1, §3.2.4). The report gives no across-round measurement of diversity, tail coverage, or real-data error, so it shows the pattern is used, not that it avoids narrowing.

**Conditions and limits.** Both theories are linear models with a single task. The summarization and XSUM experiments use ROUGE-1 against references as the oracle, which is not available for open-ended tasks. Verifiers are available cheaply mainly for code and mathematics, where compilers, reference solutions, and heuristic checkers exist ([[beyond-model-collapse]] §2).

**Implication.** In both theories and in the summarization experiment, the effect of verification depends on how well the verifier separates correct from incorrect labels on the generator's own outputs, not on the verifier's own task score. A loop that verifies only where verifiers are cheap shifts its data toward math, code, and tool use (Interpretation); breadth in other domains then needs human or calibrated-judge evaluation (ch-44b, ch-49).

## §5 Two separate risks: recursive collapse and single-round narrowing

**Definitions.** **Recursive collapse** is error growth across repeated rounds of training on model outputs (§1–§2). **Single-round narrowing** is a loss of tail content or output variety after one round of training on a teacher's or the model's own outputs, with no recursion.

**Problem.** The two risks need different controls. Accumulation (§2) addresses the first. It does nothing for the second, because a single distillation round has no earlier synthetic rounds to accumulate.

**Mechanism for single-round narrowing (Interpretation).**
1. A generator's accuracy on a fact depends on how often the fact appeared in its pretraining data.
2. Synthetic data about rare facts therefore contain more errors or omit those facts.
3. A verifier rejects some errors, which lowers the count of rare-fact examples further.
4. The student sees fewer and less accurate examples of the tail than of the head.

**Evidence for step 1 ([[long-tail-knowledge]]).** BLOOM-176B's TriviaQA accuracy rises from 25% to above 55% as the number of pretraining documents containing the question and answer entities rises from 10¹ to 10⁴ (§3.1). A 4.8B model retrained on C4 without the relevant documents for sampled questions loses accuracy on them, more for questions that had more documents (§3.2, Fig. 5). For Natural Questions items with fewer than 100 relevant documents, a log-linear fit extrapolates to over 10¹⁸ parameters to reach a strong supervised baseline (§4.2).

**Worked example (derived).** Suppose a 176B generator answers rare-entity questions (about 10 relevant documents) at 25% and common-entity questions (about 10⁴) at 55%, and 1,000 questions of each kind are generated with an answer checker that keeps only correct answers. The kept set has 250 rare and 550 common examples, so the rare share falls from 50% of prompts to 31% of kept examples (250/800). Without a checker, 750 of the 1,000 rare examples carry wrong answers. Either choice reduces correct rare-fact supervision relative to common facts. The generator accuracies are the BLOOM-176B numbers above; that a model's generation accuracy equals its QA accuracy is an assumption.

**Evidence for narrowing without recursion.** Shumailov's Fig. 1 histograms already shift toward high-probability sequences after the first generation ([[model-collapse]] Fig. 1). For Gemma 2 2B SFT with more than 1024 real examples, adding synthetic examples raised test loss at a fixed real count ([[collapse-or-thrive]] §4). The authors of [[beyond-model-collapse]] interpret the success of perplexity-based self-selection as a preference for easy-to-learn samples (§6.2, Interpretation). No source in the library measures accuracy by fact frequency before and after one round of distillation SFT; the size of tail loss from a single round is an Open question.

**Measurement.** Stratify evaluation questions by relevant-document count in the pretraining corpus, as [[long-tail-knowledge]] does (§2), or by entity popularity, as in popularity-stratified QA such as PopQA (Mallen et al., named in [[long-tail-knowledge]] §5 as concurrent work on QA accuracy and pretraining frequency; no library card, so no numbers are cited here). Report accuracy per language, including low-resource languages (ch-13a), and per answer format. Compare the checkpoint before and after each synthetic round on the same slices (ch-50).

**Implication.** A general-purpose model can lose rare knowledge, languages, and formats in a pipeline that never recurses. Average benchmark scores are dominated by common items, so the check is slice-level.

## §6 Output-diversity loss from post-training

**Definition.** Per-input diversity is the diversity of several outputs sampled for the same prompt; cross-input diversity is the diversity of one output each across different prompts ([[rlhf-generalisation-diversity]] §5.2, Eq. 2–3).

**Formulas.** For a diversity metric D, K samples per input, and N inputs:

`PerInput_D(π) = (1/N) Σ_i D({y_1..y_K ~ π(·|x_i)})`   `CrossInput_D(π) = D({y^(1)(x_1), …, y^(1)(x_N)})`

- `π`: the policy; `y^(1)(x_i)`: the first sample for input x_i. The Sentence-BERT metric is `D = 1 − mean pairwise cosine similarity` of sentence embeddings; EAD counts distinct n-grams with a length correction; NLI diversity counts contradictions and entailments.

**Worked example.** Three outputs for one prompt with pairwise cosine similarities 0.90, 0.80, and 0.85 have mean 0.85, so Sentence-BERT diversity is 0.15 (computed).

**Evidence.** LLaMA 7B summarization policies (K = 16, N = 500, temperature 1): RLHF (PPO, β_KL = 0.05) has lower per-input diversity than SFT on EAD and Sentence-BERT, and lower cross-input diversity by a smaller margin; NLI diversity does not separate the models; OPT models of five sizes show the same trends (§6.2, Figs. 5–6, App. J.4). For generalization, LLaMA 7B RLHF policies outperform SFT out of distribution on summarization (TL;DR to CNN/DailyMail, Fig. 2) and on instruction following; in instruction following, all models generalize equally well to AlpacaEval, and RLHF has the larger advantage on the harder Sequential Instructions shift (§6.1, Fig. 3). Diversity was measured only for the summarization policies. The authors describe this as a trade-off between generalization and diversity (Abstract). **Across models ([[artificial-hivemind]]).** On 100 open-ended queries with top-p 0.9 and temperature 1.0, the average pairwise similarity of 50 responses from one model exceeds 0.8 in 79% of cases (model–query pairs across 25 models); min-p sampling (min-p 0.1, temperature 2.0) still leaves 61.2% of response pairs above 0.8; average similarity between different models' responses ranges from 0.71 to 0.82 (§3, Figs. 4–6). The study is observational and does not test causes; the authors list shared data pipelines and synthetic-data contamination as possible explanations (§3). **In RL ([[echo-chamber-rl-post-training]]).** A 150M model pretrained on math documents plus three instruction datasets with different solution formats moves to one format within the first PPO epoch; pass@64 declines late in training at KL coefficient 0.001 and stays stable at 0.01 with comparable pass@1 (§3.1, Fig. 3).

**Conditions and limits.** Kirk et al. report diversity only for summarization because the metrics did not separate models on long instruction-following outputs (§6.2). The echo-chamber models have 150M and 1B parameters.

**Implication for synthetic data (Interpretation).** A post-trained teacher with low per-input diversity yields synthetic sets with fewer distinct answers per prompt, and sampling more responses per prompt returns similar text. Replacing one teacher with another does not guarantee variety when models share outputs at 0.71–0.82 similarity. Diversity should be measured on the synthetic set itself, per input and across inputs, and compared with human data for the same prompts (ch-22 for selection metrics, ch-43 for RL controls).

## §7 What to log in a synthetic-data loop

The sources above support the following measurements. None of them gives an action threshold; thresholds must be set from the pipeline's own baseline runs and their variance (ch-51).

| Measurement | What it detects | Source of the method |
|---|---|---|
| Loss or perplexity on a fixed real held-out set, per round | recursive collapse (rising error) | [[model-collapse]] Fig. 1; [[model-collapse-accumulation]] Table 2 |
| Histogram of per-sequence likelihood of generated data under the round-0 model | shift toward typical outputs plus a long improbable tail | [[model-collapse]] Fig. 1 |
| Real and synthetic counts and fractions in the training pool | replace versus accumulate; synthetic share p₂ | [[model-collapse-accumulation]] §2; [[strong-model-collapse]] §3 |
| Verifier keep rates ϕ and ψ on a labeled sample; p* | whether selection can recover correct labels | [[beyond-model-collapse]] §4, §6 |
| Verifier agreement with ground truth across rounds | drift toward the verifier's bias | [[escaping-collapse-verification]] §4 |
| Accuracy by fact frequency, language, and answer format | tail loss in a single round | [[long-tail-knowledge]] §3 |
| Per-input and cross-input diversity | output narrowing | [[rlhf-generalisation-diversity]] §5.2 |
| Similarity between the synthetic set and other models' outputs | shared outputs across teachers | [[artificial-hivemind]] §3 |
| pass@1 and pass@k at large k | narrowing of solution coverage | [[echo-chamber-rl-post-training]] §3.1 |

## Negative samples and negative feedback

**Four meanings used here** (course standard): (1) negative marginal value, a sample that lowers performance when used as a positive target; (2) negative as content, a failure placed in the input or in a corrected target and trained with cross-entropy; (3) negative as conditioning, a failure trained under a control token; (4) negative as gradient, an explicit decrease of the sample's likelihood. Only (4) removes probability mass from the sample. This section concerns verifier rejects, which are type (1) when discarded and type (4) when reused as rejected responses or negative advantages.

**1. Where negatives come from.** Verifier rejects: wrong final answers ([[openmathinstruct]] releases 6.6M of them and does not train on them, §1), parser failures, failed unit tests or execution, timeouts, truncated responses, and judge or reward-model rejections. Label errors run both ways. False positives: answer matching keeps right-answer solutions with flawed reasoning, which OpenMathInstruct calls rare without a rate (§2.3); an agent can pass tests by chance and then edit unrelated files ([[deepswe]] §2.3). False negatives: Sky-T1's regex+sympy parser kept 25% of correct math solutions where a gpt-4o-mini judge kept 73%, so the parser discarded about 66% of the solutions the judge accepted (`1 − 25/73`, derived) ([[bespoke-stratos]] "Data Curation"); Monte-Carlo step labels from too-hard questions produce false negatives ([[omegaprm]] App. A).

**2. What current practice does.** SFT filters discard rejects (type 1): [[beyond-model-collapse]], [[openmathinstruct]], [[bespoke-stratos]]. Qwen2.5 uses responses that fail execution feedback or answer matching as the rejected side of offline DPO pairs (type 4) ([[qwen-2.5]] §4.2). DeepSWE masks the loss for trajectories that hit the context limit, time out (20 minutes), or reach the step limit, instead of training on them with the zero reward given to failed tests ([[deepswe]] §2.3); with a group baseline, a zero reward below the group mean is a negative advantage (type 4).

**3. Mechanism.** For a softmax over next tokens with logits z, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. A gradient step that lowers `log p_y` with step size η changes `z_y` by `−η(1 − p_y)` and every other `z_j` by `+η p_j`, so the removed mass goes mostly to the tokens that were already most likely. Worked example: probabilities (a, b, y) = (0.70, 0.20, 0.10), η = 1: logit changes (+0.70, +0.20, −0.90) give new probabilities (0.832, 0.144, 0.024) (computed). Token b was not penalized and still loses mass; token a gains 0.132. When the penalized response is already unlikely (small p_y), the same concentration applies.

**Error asymmetry (worked example, derived).** A generator produces 1,000 answers, 400 correct. The verifier rejects 30% of correct answers and keeps 5% of wrong ones. Kept pool: 280 correct + 30 wrong, so 9.7% of SFT targets are wrong (type-1 harm). The generator error p = 0.6 is below p* = 0.933, so in the §4 infinite-data theory training on the kept pool still succeeds. Reject pool: 120 correct + 570 wrong, so 17.4% of the rejects are correct. Discarding the rejects loses 120 correct examples. Using the rejects as type-4 negatives lowers the likelihood of those 120 correct answers, and the mechanism above moves their mass toward the most likely alternatives, which may be the typical wrong answers. A false negative is lost data under discard and an active push against correct behavior under negative gradient. Panel D of [figures/collapse-iterations.html](figures/collapse-iterations.html) reproduces these pools with its default settings.

**4. Evidence with numbers.** No source in this chapter measures the benefit of reusing verifier rejects as negatives against discarding them. Failure mode of type-4 negatives: DPO on on-policy refusal pairs lowered the refusal rate of Llama-3-8B-Instruct from 74.4% to 33.4% because probability moved away from the preferred refusals ([[likelihood-displacement]] §6.2). Adding an SFT term raised the preferred responses' log probability (+20.2 Gemma-2B-IT, +28.6 Llama-3-8B-Instruct, mean change) where plain DPO lowered it (−59.2, −48.1) (Table 16). The share of improvement due to negatives versus positives is not measured for synthetic-data verification (see ch-43a for RL settings).

**5. Controls.** Measure the verifier's false-negative rate by reason code (parse, execution, timeout, truncation, judge) on a labeled sample before any reject is used as a negative. Mask uncertain failures such as timeouts and truncation instead of penalizing them ([[deepswe]] §2.3). Add a second checker for rejects before labeling them wrong ([[bespoke-stratos]]). Anchor preference losses with a positive NLL term ([[likelihood-displacement]] Table 16). Keep negatives on-policy and localized to the first wrong step (ch-24, ch-31a).

**6. Diagnostics.** Log chosen and rejected log probabilities separately; log rejects by reason code per round; track pass@1 and pass@k at large k, per-input diversity, and the abstention rate on unanswerable prompts.

**7. Effect on generality (Interpretation).** Parser false negatives concentrate on answer formats the parser does not handle, so penalizing them pushes the model away from those formats and narrows format coverage. Judge rejects concentrate where the judge's preferences differ from humans'; [[artificial-hivemind]] reports that reward models and judges are less calibrated on responses where annotators disagree (Abstract). Whether a loop over same-policy pairs narrows or broadens depends on the label source, following [[escaping-collapse-verification]] §4: the model moves toward what the labels accept.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OPT-125m (Shumailov et al.) | 125M | SFT | data; generation | wikitext2; five-way beam search, 64 tokens per 64-token block; generated set same size as training set; 5 runs | Nature 631:755, "Model collapse in language models" | verified 2026-09-14 ([[model-collapse]]) | no ablation reported |
| OPT-125m | 125M | SFT | setting 1; setting 2 | 5 epochs, no original data; 10 epochs, random 10% of original data each generation | same, Fig. 1b–c | verified 2026-09-14 | Fig. 1b vs 1c |
| OPT-125m | 125M | SFT | LR; batch; optimizer; number of generations | not reported | checked main text, Fig. 1 caption, SI §4 | not reported | — |
| GPT-2; Llama-2 (Gerstgrasser et al.) | 9M; 12M, 42M, 126M | pretrain-stable | data; epochs; per-round sample | TinyStories 470M tokens; 1 epoch; new TinyStories-sized sample per round at temperature 1.0 (also 0.3); new initialization each round | arXiv:2404.01413v2 §2.1 | verified 2026-09-15 | Table 2: accumulate vs replace vs replace with grown dataset; temperature 0.3 row |
| Same | 9M–126M | pretrain-stable | LR; batch; sequence length | not reported | checked §2.1, App. C | not reported | — |
| Gemma 2 (Kazdan et al.) | 2B, 9B, 27B | SFT | real data; data per round | HelpSteer2; replace ~12.5K; accumulate ~12.5K × t | arXiv:2410.16713v4 §2.3 | verified 2026-09-15 | Fig. 3 |
| Gemma 2 2B | 2B | SFT | synthetic pool; mixes | 100k completions, >512 tokens removed, 55,000 kept; real counts 128–12,000, synthetic counts 0–40,000 | v4 §4, Fig. 5 | verified 2026-09-15 | Fig. 5: synthetic lowered loss only at ≤1024 real examples |
| GPT-2-small (Dohmatob et al.) | 124M; 166M (18 layers); 204M (24 layers) | pretrain-stable | data; synthetic fraction; generation | BabiStories 2.2M stories; p₂ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0}; temperature 1, top-p 1, quality-filtered | arXiv:2410.04840v2 §4.2, Fig. 8, App. A.3 | verified 2026-09-15 | Fig. 8 left (p₂) and right (depth at p₂ = 1) |
| GPT-2-small | 124M | pretrain-stable | LR; dropout; weight decay; warm-up; context | 5×10⁻³; 0.05; 0.1; 2,000 iterations; 512 tokens | v2 App. A.3 | verified 2026-09-15 | no ablation reported |
| Llama-2-7B generator (Feng et al.) | 7B | SFT | data; LR; schedule; epochs; batch; block | 12.5% of XLSUM English train (307,000); 5e-5; cosine; 1; 32; 1024 | arXiv:2406.07515v2 §6.2, App. D | verified 2026-09-15 | no ablation reported |
| Llama-2-7B on selected data | 7B | distill-SFT | selection rate; LR; schedule; decoding | 12.5%, 25%, 50%; 2e-5; constant; greedy | v2 §6.2, App. D | verified 2026-09-15 | Fig. 5: oracle > self-selection > generator; Llama-3 selection ≈ random |
| SmolLM2-135M (Yi et al.) | 135M | distill-SFT | data; rounds; verifier | 12.5% of XSUM, 1 epoch; 15 rounds; oracle keeps top 12.5% by ROUGE-1; greedy | arXiv:2510.16657v3 §5.3 | verified 2026-09-15 | Fig. 5: filtered vs unfiltered |
| LLaMA 7B RLHF (Kirk et al.) | 7B | RL | algorithm; KL coefficient and placement | PPO; β_KL = 0.05, KL to the SFT policy subtracted from the reward | arXiv:2310.06452v3 §4, Eq. 1 | verified 2026-09-15 | "our own early experiments" (§4); numbers not reported |
| LLaMA 7B diversity evaluation | 7B | eval-gate | samples per input; inputs; temperature | 16; 500; 1 | v3 §5.2 | verified 2026-09-15 | no ablation reported |
| OLMo-codebase model (Zhao et al.) | 150M | RL | KL coefficient | 1e-3 default; 0 and 0.01 compared | arXiv:2504.07912v2 App. C | verified 2026-09-14 ([[echo-chamber-rl-post-training]]) | Fig. 3: 0.01 keeps a second format and stable pass@64 at comparable pass@1 |
| Bespoke-Stratos-17k curation | — | distill-SFT | math correctness filter | gpt-4o-mini judge vs ground truth (replacing regex+sympy parser) | blog "Data Curation" | verified 2026-09-14 ([[bespoke-stratos]]) | retained correct solutions 25% (parser) → 73% (judge) |

Rows marked 2026-09-15 were read in the primary arXiv PDFs on that date because no library card exists for [[model-collapse-accumulation]], [[collapse-or-thrive]], [[beyond-model-collapse]], [[escaping-collapse-verification]], or [[rlhf-generalisation-diversity]], and the [[strong-model-collapse]] card has not been verified; the chapter excerpts hold the checked extracts.

**Starting point for a small general-purpose run.** For a multi-round self-training loop on a small language model, train each round on the original real data plus all earlier synthetic rounds, as in the TinyStories runs (9M–126M parameters, one epoch per round, samples at temperature 1.0), where accumulate kept round-4 cross-entropy below round 1 and replace raised it. When a verifier selects SFT data for a 7B model, the verified reference is Feng et al.: fine-tune the pre-trained model for 1 epoch at a constant LR of 2e-5 with greedy generation, choosing among candidate selectors by measured p* and testing selection rates of 12.5%, 25%, and 50%; only the selector and rate were varied, on one summarization task. When a pipeline has more than 1,024 real SFT examples, test synthetic additions against a real-only control, because for Gemma 2 2B on HelpSteer2 synthetic data filtered only by length almost always raised test loss above that count. When RL output diversity matters, compare KL coefficients 0.001 and 0.01 on pass@k, as the 150M echo-chamber runs did.

## Generalization lens

**(a) What increases breadth.**
- Keeping real data in every round: accumulate lowered TinyStories cross-entropy from 1.71 to 1.59 at 126M by round 4 ([[model-collapse-accumulation]] Table 2); replicated in Gemma 2 SFT ([[collapse-or-thrive]] Fig. 3).
- Selection with a high breakdown point: oracle-selected 12.5% of summaries beat training on 100% of the human summaries for Llama-2-7B ([[beyond-model-collapse]] Fig. 5).
- Selection over a larger pool of inputs: self-selection gains are attributed to a shifted input distribution ([[beyond-model-collapse]] §7, Interpretation).
- Small synthetic additions when real data are scarce: lower test loss at ≤1024 real examples ([[collapse-or-thrive]] Fig. 5).
- Retrieval for rare facts: largest gains on questions with few relevant documents ([[long-tail-knowledge]] §4.3).
- RLHF over SFT for larger distribution shifts, measured by GPT-4 preference ([[rlhf-generalisation-diversity]] §6.1).

**(b) What causes narrowing or forgetting.**
- Replacing data across rounds: GPT-2 9M cross-entropy 1.82 → 2.91 by round 10 ([[model-collapse-accumulation]] Table 2); OPT-125m perplexity rises ([[model-collapse]] Fig. 1).
- A fixed share of label-shifted data: error floor p₂²c² in theory; delayed scaling in GPT-2 ([[strong-model-collapse]] §3.1, §4.2).
- Biased verifiers over many rounds: convergence to the verifier's center ([[escaping-collapse-verification]] Theorem 4.1).
- Selecting by the model's own likelihood where likelihood does not track correctness: 19.2% vs 60.4% oracle best-of-50 ([[beyond-model-collapse]] Table 1).
- Post-training that lowers per-input diversity ([[rlhf-generalisation-diversity]] §6.2) and format collapse in RL at low KL ([[echo-chamber-rl-post-training]] Fig. 3).
- Teachers that produce similar outputs across model families ([[artificial-hivemind]] §3).
- Negatives from a verifier with a high false-negative rate (Negative samples section; derived example).

**(c) How to measure it for this stage.**
- Real held-out loss per round and per-sequence likelihood histograms (§7).
- Accuracy by relevant-document count, entity popularity, language, and format, before and after each round ([[long-tail-knowledge]] §2–§3; ch-13a; ch-50).
- Verifier keep rates on a labeled sample, reported with p* ([[beyond-model-collapse]] §4).
- Per-input and cross-input diversity with several metrics, since NLI diversity did not separate models where EAD and Sentence-BERT did ([[rlhf-generalisation-diversity]] §6.2).
- pass@k at large k alongside pass@1 ([[echo-chamber-rl-post-training]] §3.1).
- Known measurement errors: diversity metrics are length-sensitive and failed to separate long instruction-following outputs ([[rlhf-generalisation-diversity]] §6.2); the relevant-document pipeline has about 60% precision on a 300-question TriviaQA sample ([[long-tail-knowledge]] §2); GPT-4 preference judgments carry judge bias (ch-49).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Replacing earlier data with each round's synthetic data | real held-out loss rises across rounds | train one round with accumulate and compare at matched rounds ([[model-collapse-accumulation]] Table 2) |
| Reading "accumulate avoids collapse" as "synthetic data scale like real data" | loss plateaus while the synthetic share grows | fit loss against real and synthetic counts separately ([[strong-model-collapse]] Eq. 12; [[collapse-or-thrive]] §4) |
| Tracking only generated-data likelihood | generated text becomes more typical while real-data loss rises | evaluate every round on a fixed real held-out set ([[model-collapse]] Fig. 1) |
| Choosing the strongest available model as the verifier | selected data trains a model no better than random selection | measure ϕ, ψ, and p* on the generator's own outputs ([[beyond-model-collapse]] §6.2) |
| Selecting by the generator's own likelihood where it does not track correctness | kept examples are not more accurate than greedy outputs | oracle check on a sample: best-of-k vs lowest-perplexity accuracy ([[beyond-model-collapse]] Table 1) |
| Running verified rounds without a ground-truth audit | early gains stop or reverse | score each round against held-out ground truth, not only verifier acceptance ([[escaping-collapse-verification]] §4) |
| Reusing parser rejects as negative gradients | correct answers in unusual formats become less likely | false-negative rate by reason code on a labeled reject sample ([[bespoke-stratos]]) |
| Scoring timeouts, context-limit, and step-limit trajectories as failures | training reward collapses in multi-turn agent RL (DeepSWE ablation on Qwen3-14B) | mask these cases and compare reward curves ([[deepswe]] §2.3, Fig. 6) |
| Reporting only averages after distillation | rare-fact and low-resource-language regressions found later | frequency and language slices before and after ([[long-tail-knowledge]] §3) |
| Sampling more responses per prompt to gain diversity from a low-diversity teacher | many near-duplicate responses | per-input embedding similarity of the synthetic set ([[artificial-hivemind]] Fig. 4) |
| Citing single-round production pipelines as proof that verification prevents collapse | no across-round measurement exists | require round-by-round real-data evaluation before drawing the conclusion |

## Check your understanding

1. In Table 2 of the accumulation paper, replace with a dataset grown to the accumulated size still degrades. Which explanation of accumulate's advantage does this rule out, and which one does the linear-regression proof give instead?
2. Accumulation keeps error bounded, yet Strong Model Collapse says a fixed synthetic share stops error from falling. Explain why both statements can be true, using the two formulas in §2 and §3.
3. Using `p* = 1/(1 + ψ/ϕ)`, explain why a verifier that rejects many correct labels can still allow near-optimal training, while one that keeps a modest share of wrong labels can fail.
4. A fine-tuned Llama-3 with higher ROUGE-1 than the Llama-2 generator selected data no better than random. What property of a verifier does this show matters, and how would you measure it before training?
5. In the knowledge-ball model of verified retraining, why does verifier selectivity change the speed of convergence but not its limit, and what does the limit imply for a loop verified by one reward model?
6. Why can one round of verified distillation reduce correct supervision about rare facts, even when the verifier has no false positives? Use the long-tail accuracy numbers.
7. Using `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, explain why reusing verifier false negatives as negative gradients is worse than discarding them, and where the removed probability goes.
8. RLHF generalized better out of distribution than SFT but lowered per-input diversity. What does this imply for using an RLHF-tuned model as the generator of a synthetic SFT set for a general-purpose student?

## Connections

- Previous: ch-22 — Quality, Diversity, and Gradient-Based Data Selection. Selection metrics there operate on a fixed pool; this chapter covers what happens when the pool is regenerated by models.
- Next: ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data. Answer checking and first-wrong-step negatives apply §4 and the negative-samples section to traces.
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix (where verification sits in the pipeline).
- ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining (long-tail knowledge in detail); ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (language slices).
- ch-29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification (applies §7).
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation; ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity; ch-43 — Entropy, Output Diversity, and KL Control in RL (§6).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages (full derivations for type-4 negatives).
- ch-44 — Process Supervision and Verifiable Rewards; ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing (verifier coverage).
- ch-45 — Self-Improvement Loops and Multi-Stage Reasoning Pipelines (multi-round loops in RL).
- ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting; ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing; ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.

## Sources

- [[model-collapse]] — definition, error sources, discrete and Gaussian theory, OPT-125m settings and Fig. 1 results.
- [[model-collapse-accumulation]] — replace vs accumulate definitions, Table 2 cross-entropy, linear-regression Theorem 2.
- [[collapse-or-thrive]] — Gemma 2 SFT replication, accumulate-subsample, real-data cardinality result.
- [[strong-model-collapse]] — label-shift setting, Corollary 1 floor, MNIST and GPT-2 experiments, model-size and mixing results.
- [[beyond-model-collapse]] — generator best-of-k vs self-selection, breakdown point p*, eigenvalue and summarization experiments.
- [[escaping-collapse-verification]] — verifier knowledge-ball model, convergence to the verifier center, XSUM rounds.
- [[long-tail-knowledge]] — accuracy by relevant-document count, counterfactual retraining, retrieval.
- [[rlhf-generalisation-diversity]] — per-input and cross-input diversity metrics; RLHF generalization and diversity results.
- [[artificial-hivemind]] — intra-model and inter-model similarity of open-ended outputs; judge calibration.
- [[echo-chamber-rl-post-training]] — format collapse in RL and the KL-coefficient comparison.
- [[bespoke-stratos]] — parser false negatives (25% vs 73% retained correct solutions).
- [[openmathinstruct]] — released wrong solutions; flawed-reasoning positives described as rare.
- [[omegaprm]] — false-negative step labels from too-hard questions.
- [[deepswe]] — masking of timeouts, context-limit, and step-limit trajectories; tests passed by chance.
- [[qwen-2.5]] — failing responses used as rejected DPO samples.
- [[likelihood-displacement]] — displacement of preferred responses under DPO and the effect of an SFT term.
- [[nemotron-4-synthetic]] — multi-round weak-to-strong synthetic alignment; corrections to reward-model refresh claims.
- [[apigen]] — corrections to ablation, pass-rate, and ranking claims.
- [[prismatic-synthesis]] — corrections to G-Vendi and distillation claims.
