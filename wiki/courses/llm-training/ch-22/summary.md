# Ch-22 Summary — Quality, Diversity, and Gradient-Based Selection

## 1. Core thesis

Core insight what I get from this chapter is 'How to select the best sample from the dataset pool.'
until ch-21, we talked about data perspective. but this chapter, we slightly change our view point from data to model. 
Bad samples have negative impact to the model performance (negative marginal value, not just zero.) Selection became binding constraint, not quantity. 

## 2. The 6 methods

### 2.1 AlpaGasus
Mechanism (1줄): for each data sample, we use llm to rate the quality of the data sample with 1 to 5 score with a specific rubric. (sample the data ≥4.5)
Key empirical (1줄): Average qulaity training on more data is worse than selecting for the top decile, because usefulness of each sample is not uniform. 
핵심 limit (1줄): Judge bias inheritance (length, style, sycophancy bias), cannot evaluate the correctness.

### 2.2 Cherry-LLM / IFD

IFD score conssist with two dferent thing: PPL_cond(a | q) / PPL_uncond(a). 
Both PPL_cond and PPL_uncond are perplexity of a(answer)
Remember PPL(perplexity) = 1/P (1/probability)

- IFD < 1: sample can provide good training signal. the response is easier to predict under given q. 
- IFD > 1: sample is noise. q-a mismatched.
- IFD ~= 1: sample is neutral (boilerplate)

also IFD can filter out 'Instruction-response mismatch' 
Cherry-LLM suggest thrshold just below 1.0 provide the sweet spot(IFD 0.7-0.95) of data selection.
if IFD is too small like IFD=0.1, it is trivial sample. 
if IFD is too close to 1, it is boilperate (bare learning signal). 
this method require warm-up for calculating IFD score. (becuase cold model is already dominated by format/template gradient. so need to warm-up for content gradient.) 

### 2.3 Superfiltering
(this is weak-to-strong)
The method of here is same as IFD. BUT! instead of original model we want to train, we use the small model to measrue IFD. 
and this only work with the same family model, because it requires similar distribution (behavior). for example, Qwen Family, Deepseek Family. 
so, inside of the family models, they share the similar distribution and behavior. so we can use the small model to measure IFD score. (save money)

### 2.4 DEITA
- Complexity (instruction의 difficulty) — Evol-Instruct mutation ranking으로 distilled scorer
- Quality (response의 quality) — Evol-Quality mutation ranking으로 distilled scorer
- Diversity — embedding distance 기반 greedy selector

Set-level vs sample-level distinction: Diversity는 set-level (previously selected와 비교)
Lexicographic priority (NOT weighted sum) — diversity is not traded off with quality
after data selection with DEITA, 6K-10K samples beats 200K samples
Embedding distance, τ=0.9 threshold

### 2.5 LESS
run LoRA fine-tunning with a small amount of data sample. and then measure the gradient of 'when sample-trainning' and 'when verification-data-training'. if the direction is same, then the data sample is a good sample to train, because both direction shows the same direction and we can know that sampe will improve the model in that verification dataset. if it has opposite dirrection (cosine < 0), then the data sample may provide harmful training signal to iprove that sepcific domain-verification dataset (increase the validation loss). 
and also, applied Adam optimizer's parameter during warm-up stage (small LoRA) to correct the sensitivity of parameter dimension for each data sample. 

g_val DOT g_i > 0: positive direction = same direction = good data
g_val DOT g_i < 0: negative direction = opposite direction = bad data
apply diagonal Hessian inverse to g_val and g_i: g_val^T H^{-1} g_i (use Adam's v as approximate diagonal Hessian inverse. use LoRA warm-up's byproduct, v)
applied JL projection to reduce the computational complexity (7B -> 8k dimension.)

but the problem of this method is, targeting to the speicific task which means trageting the human-selected domain (target-aware). like ch-21, it can make blind spot. but this is usefull method to improve the model in that specific domain. 

### 2.6 Prismatic

The main focus of G-Vendi is 'diversity'. except Prismatic, all of the selection ask the question; 'How to improve the density of the dataset?' and Prismatic ask 'How to improve the coverage of the dataset?' 
compare with encoder-based method, G-Vendi collects g_i (gradient of sample i) and normalize it (by applying L2-normalization, minimize the gradient magnitude. which means ignore the gradient magnitude, but consider the diversity-keep the agnle). 
Ulike IFD or LESS, G-Vendi doesn't need to warm-up to the model, just extract the agnle (8k dimension). and actually this G-Vendi score works well with OOD generalization selection. (of course increase the model performance, as G-Vendi filtered dataset cover larger area.)

Per-sample normalized gradient → Gram matrix K → Density matrix ρ = K/tr(K) → Eigenvalues {λ_k} → Von-Neumann entropy H(ρ) = -Σ λ_k log λ_k → G-Vendi = exp(H(ρ)) = effective number of distinct gradient directions

as a result of Prismatic, 7B Prismatic shows better performance than 671B brute distillation. 
which means, Diversity, not generator scale, is the binding constraint (core insight of Prismatic)

also let me add the difference between Embeddding vs Gradient
Embedding kernel just check the diversity of the dataset (surface level)
but Gradient can check the diversity from the model's training.


## 3. Framework extension — Ch-21's 4-axis → Ch-22's 6-axis

- Distribution shape of rater scores — generator-specific
- Augmentation × Selection interaction — DEITA가 explicit instance
- Signal portability — data-intrinsic vs model-specific
- Three reference frames — absolute / pool-relative
- Cost scaling profile — O(N) ~ O(N³)
- Selection intent — quality / informativeness / diversity / kind-target / OOD 

## 4. Selection intent table — Method classification

| Method | Intent | Mechanism | 해상도 | Cost order |
|---|---|---|---|---|
| AlpaGasus | Quality | teacher LLM to rate the data sample.  | sample-level | high (if you use api) |
| IFD | Informativeness | evaluate the perplexity with instruction and without instruction(condition) | sample-level | medium |
| Superfiltering | Informativeness | use IFD with small family model | sample-level | low |
| DEITA | Quality+Diversity+Complexity | Distillation scorer for quality and difficulty, embedding for diversity. | sample-level | medium |
| LESS | Kind / Target-match | gradient alignment with validation set | sample-level | high (datastore) but amortizable |
| Prismatic | Diversity | use G-Vendi score | sample-level | high (O(N^3)) |

## 5. What none of these filters do (§9 limits)

There are 4 common blind spots for these filters
1. Factual correctness
2. Coverage gap of *pool*
3. Distribution shift during training
4. Pool-specific threshold tuning

I think biggest limitation is 'Factual Correctness' because if we are filtering the data not related to coding or matth, it is hard to evaluate the correctness of the data if LLM-as-Judge hallucinate and produce the incorrect answer.

## 6. Open questions / parked items
From LESS, Can we believe Adam-diagonal hessian approximately works as a weight?
How can we mix the data selection strategy effectively? like LESS + Prismatic? 
If we want to cover diverse ODD, we need Prismatic, but how can we generate the diverse data sample?

## 7. Connections

### Backward — Ch-19, Ch-20, Ch-21
- ch-20 axis 5 quirks inheritance가 AlpaGasus + IFD에서 어떻게 manifest
- ch-20 pass@k self-improvement와 IFD의 self-referential selection isomorphism
- ch-21 3-layer architecture의 Layer 3 (selection) 본격 전개
- ch-21 4-axis predictor가 ch-22 6-axis로 확장

### Forward — Ch-23, Ch-24, Ch-44
- ch-23 (verification): filter는 gate, verifier 아님 — factual correctness 등 ch-22 limit이 ch-23으로 forward
- ch-24 (process reward): step-level reward = selection의 RL 대응
- ch-44 (RLVR): "diversity > scale" thesis (Prismatic)와 RLVR (verifier-grounded scale)의 대비