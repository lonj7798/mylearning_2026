<!-- chapter: ch-13
     track: pretraining
     kind: content
     title: Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale
     deps: [ch-12a]
     sources: [[doremi]], [[data-mixing-laws]], [[regmix]], [[aioli-data-mixing]], [[olmix]], [[llama-3]], [[llama-3-recipe]], [[smollm2]], [[olmo-2]], [[olmo-3]], [[long-context-data-engineering]], [[prolong]], [[prolong-recipe]], [[qwen-long-context-synth]], [[skyladder]], [[to-code-or-not-to-code]], [[data-constrained-scaling]], [[weborganizer]], [[paloma]], [[datadecide]]
     figures: figures/doremi-reweighting.html
     revised: 2026-09 (generality revision)
-->

# Chapter 13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale

> **Core insight.** A domain mixture is a probability vector over data groups. The offline methods in this chapter (DoReMi, Data Mixing Laws, RegMix, Olmix) choose it with small proxy models and transfer the decision to a larger run. DoReMi weights found with a 280M proxy raised an 8B model's average one-shot exact match on The Pile from 20.03 to 26.56 and lowered log-perplexity on all 22 domains ([[doremi]] Table 5, Table 3a), but a 1B proxy produced different weights, and in 160M-model re-implementations no existing method, DoReMi included, beat stratified sampling on all six data settings ([[aioli-data-mixing]] Table 2). Regression over many proxy runs transfers rankings when the proxy is large enough (Spearman 0.896 between 30M proxies and 1B targets, 0.73 for 1M proxies; [[olmix]] §3.3.2), and the best mixture differs by stage and by scale: annealing on GSM8K and MATH training sets raised Llama 3 8B validation scores by 24.0% and 6.4% but had a negligible effect at 405B ([[llama-3]] §3.1.3).
>
> **Guideline.** When choosing pretraining weights over many domains, fit one regression model per evaluation task on a swarm of at least 3(m + 1) proxy runs of at least 15M parameters, enforce a repetition cap inside the optimization, and compare the proposed mix with the natural distribution and with stratified sampling at the target scale, because each of these choices gave lower average bits per byte for 1B targets than the alternatives tested in [[olmix]] §3.3 and several published methods lost to stratified sampling in [[aioli-data-mixing]] §6.1. When a stage has a narrower objective (annealing, long-context extension, SFT), choose its mixture separately and keep a share of general short-context data, because 34% long plus 66% short data cost 0.8 points on a short-context suite against 2.5 points for 66% long ([[olmo-3]] §3.6.3). When benchmark training sets are candidates for an annealing mix, exclude them or report a held-out suite beside them, because Llama 3 excluded them in order to measure out-of-domain generalization ([[llama-3]] §3.1.3).

## Why this chapter matters for a general-purpose model

The training pipeline runs pretraining → mid-training → SFT → preference optimization → RL → evaluation. A mixture decision is made at every stage: the token shares of pretraining sources, the upsampled sources of an annealing phase, the long-to-short ratio of a context-extension phase, and the category shares of an SFT set. This chapter covers the pretraining decision in depth and the later-stage decisions as far as they are mixture decisions. The composition evidence that motivates keeping heterogeneous sources is in ch-09; model-based selection inside a source is in ch-10a; repetition limits are in ch-14.

The measurable problem is that a fixed token budget can be spent in many ways that give different models. In [[regmix]] Table 3, 64 models of 1B parameters trained on 25B tokens of The Pile, each with a different mixture, ranged from 43.7 to 47.9 in average accuracy over 13 tasks, and from 18.9 to 33.5 on Lambada. Two generality questions follow. First, a mixture tuned to one evaluation suite can lower scores outside it: a mixture chosen for MMLU lowered HellaSwag from 57.5 to 54.1 ([[weborganizer]] Table 10). Second, a mixture chosen with small models must still be right for the large model, and the evidence in §4 shows that this transfer holds only under stated conditions.

## §1 Mixture weights, units, and the repetition they imply

**Definition.** Let D_1, ..., D_m be domains, where domain D_i holds N_i available tokens. A mixture is a vector p on the probability simplex (p_i ≥ 0, Σ p_i = 1); training on R tokens takes p_i · R tokens from D_i ([[olmix]] §3.1.1). A domain can be a provenance source (GitHub, Wikipedia) or a topic partition of a source ([[olmix]] §3.1.1; topic domains are built in [[weborganizer]]). Two reference mixtures recur below. The **natural distribution** sets p_i ∝ N_i. **Stratified sampling** sets p_i = 1/m ([[aioli-data-mixing]] §6).

**Units.** DoReMi defines the training distribution as P_α = Σ_i α_i · unif(D_i), a distribution over examples, and computes its weights from token counts under a specific tokenizer ([[doremi]] §2, §1 footnote 1). Llama 3 reports shares "of tokens" for pretraining and shares "of examples" for SFT ([[llama-3-recipe]]). A report that does not say which unit it uses cannot be reproduced.

**Problem.** A weight above a domain's supply forces repetition. Repetition is not visible in the mixture vector; it appears only when the vector is combined with R and N_i.

**Formula.**

```
e_i = p_i · R / N_i            (epochs of domain i)
p_i ≤ k · N_i / R              (repetition cap)
```

e_i is the number of passes over domain i; p_i the mixture weight; R the requested training tokens; N_i the available tokens in domain i; k the maximum number of passes allowed. The cap is the constraint used by [[olmix]] §3.3.4.

**Worked example (toy numbers).** R = 1,000B tokens. Web N = 2,000B, code N = 100B, math N = 20B. The mix p = (0.80, 0.15, 0.05) takes 800B, 150B, and 50B tokens, so e = (0.4, 1.5, 2.5). With k = 4, the caps are code ≤ 4 · 100 / 1,000 = 0.40 and math ≤ 4 · 20 / 1,000 = 0.08. A search that proposes math = 0.10 asks for 100B math tokens, which is 5 epochs. Using the fitted law of [[data-constrained-scaling]] (Eq. 5, R*_D = 15.39), 2.5 epochs are worth about 97% of the same number of fresh tokens and 5 epochs about 90% (derived). That law was fit on whole-dataset repetition, not repetition of one domain inside a mix (App. Q), so the percentages are an extrapolation.

**Evidence.**
- In [[olmix]] Table 4 (1B targets, k = 4), a swarm restricted to feasible mixes with an unconstrained optimizer proposed a mix that repeated one domain 5 times; constraining the optimizer met the cap and gave the best average BPB (0.7647 vs 0.7855 for constraining both).
- OLMo 3 avoided repeating any domain more than about 4-7 times for its 6T-token pretraining mix ([[olmo-3]] §3.4). SmolLM2 capped StarCoderData at 10% of stage 1 "to ensure approximately 4 epochs over 11T tokens" ([[smollm2]] §4.2).
- The Pile's default weights already include repetition: DoReMi's baseline weights are example counts multiplied by the epochs chosen by the Pile authors, then normalized ([[doremi]] App. C).

**Implication.** Every mixture row in a recipe needs R and N_i beside it. A search that ignores N_i can move weight to a small, high-utility domain until it is repeated more than k times, and the resulting loss of value is not in the regression model that proposed the weight.

## §2 DoReMi: a worst-group objective over domains

**Definition.** DoReMi (Domain Reweighting with Minimax Optimization; Xie et al., arXiv 2023-05) chooses domain weights by training a small proxy model with Group DRO (group distributionally robust optimization), which minimizes the loss of the worst-off domain, and returns the averaged weights instead of the robust model ([[doremi]] §1-§2).

**Problem.** Choosing weights with downstream tasks needs many training runs and, in the authors' words, "risks overfitting to the particular set of downstream tasks" ([[doremi]] §1). A generality criterion that uses no task data is the worst-case loss over all domains.

**Mechanism.**
1. Train a reference model p_ref on reference weights α_ref. For The Pile, α_ref is the Pile's default weights; for GLaM it is uniform ([[doremi]] §3.1).
2. Train a proxy model of the same size with Group DRO. Each minibatch is sampled with **uniform** domain weights; the current α rescales each domain's loss (Algorithm 1). Minibatches are not sampled from α.
3. Per-token excess loss ℓ_θ − ℓ_ref is clipped at 0, averaged over each domain's tokens in the batch, and used in an exponentiated-gradient step on α, followed by smoothing toward uniform.
4. Return ᾱ, the average of α_t over all T steps, and train the large model on data resampled from ᾱ.

**Formula.** The objective (§2, eq. 1) and the per-step update (Algorithm 1):

```
min_θ max_{α ∈ Δ^k}  Σ_i α_i · [ (1 / Σ_{x∈D_i} |x|) · Σ_{x∈D_i} ( ℓ_θ(x) − ℓ_ref(x) ) ]

λ_t[i] = (1 / Σ_{x∈B∩D_i} |x|) · Σ_{x∈B∩D_i} Σ_j max{ ℓ_{θ_{t−1},j}(x) − ℓ_{ref,j}(x), 0 }
α'_t   = α_{t−1} ⊙ exp(η · λ_t)
α_t    = (1 − c) · α'_t / Σ_i α'_t[i] + c · u
ᾱ      = (1/T) Σ_t α_t
```

θ are proxy parameters; ℓ_θ(x) and ℓ_ref(x) the negative log-likelihoods of example x under proxy and reference; |x| its token count; Δ^k the simplex over k domains; B the minibatch; j a token index; η the step size; c the smoothing weight; u the uniform vector 1/k; ⊙ elementwise product. The log-space form is log α'_t = log α_{t−1} + η λ_t. The paper uses η = 1 and c = 1e-3 in all experiments and "did not extensively tune" them; the optimizer is Adafactor (§2).

**Worked example (toy losses).** Three domains, α_{t−1} = (1/3, 1/3, 1/3), η = 1, c = 0.001. Per-token excess losses in the batch: domain A (0.5, 0.1), domain B (−0.4, +0.2), domain C (−0.3, −0.1). After clipping, λ = (0.30, 0.10, 0.00). Then α' = (1/3)(e^0.30, e^0.10, e^0) = (0.44995, 0.36839, 0.33333); normalizing gives (0.39069, 0.31987, 0.28943); smoothing gives α_t = (0.39064, 0.31989, 0.28948). A second step with the same λ gives (0.45045, 0.30211, 0.24744). Without clipping, λ = (0.30, −0.10, −0.20) and the step gives α_t = (0.43910, 0.29445, 0.26646): B falls further behind A, although B has a token on which the proxy still trails the reference. In a real run λ changes each step because upweighting a domain scales up the proxy's updates on it and lowers its excess loss.

The figure `figures/doremi-reweighting.html` lets the reader repeat this step with other losses, η, and c, convert a mixture into epochs per domain under a repetition cap, and compare the published Pile weights from a 280M and a 1B proxy.

**Evidence** (8B main models, 200k steps, batch 512 × 1,024 tokens; [[doremi]] §3.1, App. C).
- The Pile, 280M proxy: average one-shot exact match over five generative tasks 20.03 → 26.56; baseline accuracy reached at 75k of 200k steps; worst-case domain log-perplexity 1.71 → 1.46 and average 1.64 → 1.40; all 22 domains improved (Table 5, Table 3a). Result (single study).
- The weights moved toward diverse web text: Pile-CC 0.1121 → 0.6057, OpenWebText2 0.1247 → 0.1019, Wikipedia 0.0919 → 0.0699, ArXiv 0.1052 → 0.0036, PubMed Central 0.1071 → 0.0046 (Table 1). ArXiv log-perplexity still fell from 1.64 to 1.38 (Table 4).
- Proxy versus main model at 1B: the 1B proxy itself beat the 1B baseline on 0 of 22 domains, yet its weights let a 1B main model reach baseline accuracy more than 2x faster; the authors attribute the gap to loss reweighting in the proxy versus resampling for the main model (§4, Table 3b).
- Proxy size, 8B main model: 70M / 150M / 280M / 1B proxies gave 21.11 / 22.51 / 26.56 / 22.35 average accuracy; all four improved 22 of 22 domains (Table 5, Table 3a). The 1B proxy put 0.3289 on OpenWebText2 and 0.1199 on Pile-CC; the authors suggest multiple local minima (Table 8, §3.2).
- GLaM, 8 domains: round 1 matched the uniform baseline; iterated DoReMi, which reuses the previous round's ᾱ as α_ref, matched downstream-tuned weights at round 2 and converged in 3 rounds (§3.2, Table 2, Figure 3b).
- Objective ablation at 280M: weighting by proxy loss alone ("hardest") or by negative reference loss alone ("easiest") gave average log-perplexity 2.62 and 4.18, against 2.32 for the baseline and 2.13 for DoReMi (Table 7).
- Compute: reference plus proxy at 280M cost 8% of the 8B model's training FLOPs (§3.1).

**Conditions and limits.** Downstream evaluation is five one-shot QA and cloze tasks; there are no code, math, or instruction-following evaluations and no seed variance (§3.1; card "Not reported"). In same-size runs at 510M, 760M, and 1B, DoReMi improved 15, 17, and 19 of 22 domains, not all (Table 6). Domains are defined by provenance, and the method was more effective with 22 domains than with 8 (§6). Why weights transfer across scale is left open (§6). In [[aioli-data-mixing]] Table 2 (160M models, reference trained with stratified sampling, 5K or 40K steps), DoReMi was worse than stratified sampling by 5.303, 6.898, and 0.703 test perplexity on GitHub/C4, CommonCrawl/GitHub/Wikipedia, and full SlimPajama, and better on the other three settings; the settings differ from the DoReMi paper in model size, steps, and reference weights.

**Implication.** DoReMi's objective is a formal breadth criterion: it spends tokens where the proxy trails a reference model on any domain. The evidence supports its weights for one corpus and one main-model size, and shows that the returned weights depend on proxy size, so they are a candidate to validate (§4), not a transferable constant.

## §3 Mixture laws and regression over proxy runs

**Definition.** An offline mixing method trains a swarm of small proxy models on different mixtures, fits a regression model from mixture to performance, and proposes the mixture that optimizes the prediction ([[olmix]] §3.1.2). A **mixing law** is the functional form of that regression.

**Problem.** Training one large model per candidate mixture is unaffordable. The regression must interpolate between observed mixtures and must be fit at a scale where rankings still hold at the target scale.

**Data Mixing Laws** (Ye et al., arXiv 2024-03). The loss on validation domain i is modeled as ([[data-mixing-laws]] Eq. 7):

```
L_i(r_1..M) = c_i + k_i · exp( Σ_{j=1..M} t_ij · r_j )
```

r_j is the proportion of training domain j; c_i the loss not reducible by changing the mixture; k_i > 0 a scale; t_ij the interaction between training domain j and validation domain i, where a negative t_ij means that data from j lowers loss on i (§3.2). The overall validation loss is Σ_i s_i L_i, with s_i the validation-set share of domain i, learned when the composition is unknown (§3.3). With 24 fitting and 8 validation mixtures over GitHub, Books3, and Pile-CC, this form had validation mean absolute errors of 0.0365, 0.0074, and 0.0078, against 0.8758, 0.1331, and 0.1045 for a random guess between the observed extremes (Table 1). To reach target scale, the authors fit a power law in training steps and one in model size per mixture, then fit the mixing law on the extrapolated losses (Algorithm 1).

**Worked example (toy parameters; c is assumed known to allow a hand fit).** Two domains with share r for domain 1. Observations on domain 1: L_1(0.25) = 2.6065, L_1(0.50) = 2.3679, L_1(0.75) = 2.2231 with c_1 = 2.0. Then ln(L_1 − c_1) = −0.50, −1.00, −1.50, a line with slope t = −2 and intercept 0, so k_1 = 1 and L_1(0.60) is predicted as 2 + e^(−1.2) = 2.3012. Let domain 2 follow L_2 = 1.0 + 1.5 · exp(−3(1 − r)) and the validation set be 50/50. The total L(r) = 0.5 L_1 + 0.5 L_2 has dL/dr = −e^(−2r) + 2.25 e^(−3+3r) = 0 at r* = (3 − ln 2.25) / 5 = 0.438, where L = 1.847, against 1.882 at r = 0.25, 1.851 at r = 0.50, and 1.966 at r = 0.75. The optimum is interior because each domain's loss falls with its own share and rises as the other share grows.

**Evidence for Data Mixing Laws.** Proxies of 70M, 160M, 305M, and 410M parameters were trained for 30B tokens on 20 mixtures of RedPajama; the predicted mixture was used for a 1B model on 100B tokens, validated on The Pile's validation set. It reached the default mixture's final performance at 0.73 of the steps and was estimated to match the default trained on 1.48 times more tokens (Figure 8), with lower loss than DoGE and DoReMi mixtures at the same budget (Figure 9). For continued pretraining of Pythia-70M on The Pile plus Python, the fitted law predicted the Python share that kept Pile loss unchanged (§5, Figure 10). Result (single study).

**RegMix** (Liu et al., arXiv 2024-07). RegMix samples mixtures from a Dirichlet distribution centered on token counts, trains 512 proxies of 1M parameters on 1B tokens each, fits LightGBM (gradient-boosted trees) from weights to Pile-CC validation loss, and averages the top 100 predicted mixtures ([[regmix]] §3). Spearman rank correlation between predicted and actual rankings on unseen mixtures was 98.45 (1M models), 98.64 (60M, 1B tokens), and 97.12 (1B, 25B tokens) for LightGBM, against 90.08, 89.26, and 88.01 for linear regression (Table 2). At 1B parameters and 25B tokens, the RegMix mixture averaged 47.3 over 14 tasks, against 45.1 for the Pile authors' weights, 46.8 for DoReMi's published weights renormalized to the 17 available domains (which the authors note may understate DoReMi), and 46.8 for training on Pile-CC alone; finding the mixture cost 3.5e18 FLOPs against 3.7e19 for DoReMi (Table 4). More proxy runs raised ρ more than more tokens per run beyond about 0.25B tokens (§4.2, Figure 4).

**Aioli** (Chen, Hu, et al., arXiv 2024-11). Aioli shows that DoReMi, DoGE, Skill-It, and Data Mixing Laws all minimize average loss under a mixing law of the form L_{t+1}(p_t) = σ(A_t p_t) with σ equal to identity or exp, and differ in how they set the matrix A_t ([[aioli-data-mixing]] §1, §3). Both parameterizations fit the observed loss-proportion relationship (average MSE 0.0005, R² 0.969), but the online methods' A_t values were not consistently accurate, and the similarity of each method's A_t to the fitted one had a moderate positive correlation with its improvement over stratified sampling in the 2- and 3-group settings (R² = 0.491; §4.3, Figure 3). With 160M models trained on SlimPajama subsets (batch 8 sequences of 2,048 tokens; 5K steps for 2-3 groups, 40K for 7), Aioli, which estimates A_t from the current run, beat stratified sampling on 6 of 6 settings by an average of 0.274 perplexity; every other method lost to stratified sampling on at least one setting, by up to 6.9 (Table 2). Grid search and DML each used 10 extra full runs and still lost on 2 of 6 (Table 2). A caveat for generality: across the SlimPajama models, the correlation between test perplexity and the macro-average of 8 downstream tasks was 0.529, so lower perplexity predicted worse downstream scores, and the DML mixture, which omitted three of seven groups, had the best downstream average (App. F.1).

**Olmix** (Chen, Murray, et al., Ai2, arXiv 2026-02). Olmix studies seven configuration choices of the offline schema with 1B targets trained on 100B tokens and 52 tasks scored in bits per byte (BPB) ([[olmix]] §3.3.1).
- Proxy size: proxies of at least 15M parameters had rank correlation above 0.89 with 1B targets; 1M proxies had 0.73; 30M proxies on 3B tokens had 0.896 (§3.3.2, Figure 3). The authors found that RegMix's public "1M" implementation is closer to 15M (footnote 1).
- Swarm size: the required number of runs grows linearly with the number of domains m, and K ≥ 3(m + 1) runs gave close to zero error relative to the best mix found (Figure 4).
- Regression: the log-linear form adapted from Data Mixing Laws had the best fit at K ≥ 75 and the best downstream BPB at K = 128; LightGBM needed more than 118 runs (Figure 6). One model per task gave BPB 0.765 against 0.774 for one model on the average (Table 3).
- Optimization: an exact solver minimized predicted BPB but a KL penalty λ = 0.05 toward the natural distribution gave the best target BPB, 0.7647 against 0.7772 (Figure 8).
- Evolving domains: over five domain-set updates ending at 64 domains, reusing the ratios of unaffected domains reached +11.6% over the natural distribution with 216 proxy runs, against +12.2% for recomputing from scratch with 832 runs (§5.1.2). The best mix reached the natural distribution's final BPB in about 20,000 steps instead of 61,000.

**Conditions and limits.** All four studies optimize loss or BPB on a chosen validation set, not accuracy on an unseen task suite. Data Mixing Laws note that mixture rankings can change with model size and token count (footnote 7), an **open question** for any single transfer. RegMix's targets are one validation domain (Pile-CC) and tasks averaged from 0 to 5 shots; Aioli's models are 160M with batch 8; Olmix covers only the offline schema and its reuse theory assumes the log-linear form (§6).

**Implication.** Regression methods replace a guess with a fitted surface, but the surface is only as general as its targets. Per-task targets, a regularizer toward the natural distribution, and a repetition cap each moved the target model toward lower average BPB in [[olmix]]; a single-domain or single-benchmark target does not measure breadth.

## §4 Validating a mixture across scale

**Definition.** Rank invariance is the assumption that the ordering of mixtures by performance is the same for the proxy and for the target model ([[regmix]] §1).

**Problem.** A mixture that wins at 30M or 1B can lose at the target scale, and the error is discovered only after the expensive run.

**Mechanism: a validation protocol assembled from the sources.**
1. Measure proxy-to-target rank correlation on the metric used for the decision, at two proxy sizes, before trusting a swarm ([[olmix]] Figure 3; [[regmix]] Table 2).
2. Use evaluation tasks that separate recipes at the proxy scale; in [[datadecide]] (25 data recipes, most of them filtering choices rather than mixture weights), 150M rankings predicted the 1B winner in about 80% of pairs, but SocialIQA and BoolQ decisions were unreliable at the scales tested (Abstract; §3.1).
3. Train one intermediate model on the candidate and evaluate "several key benchmarks" before committing, as Llama 3 did (§3.1.2).
4. Report a held-out suite that was not used to fit or select the mixture (examples below).

**Evidence on transfer.**
- For: RegMix LightGBM ρ = 97.12 from 1M proxies to 1B models on 25B tokens, and RegMix outperformed human-selected weights for most tasks in 7B models trained up to 100B tokens (Table 2, Figure 1). Olmix ρ = 0.896 from 30M to 1B (Figure 3).
- Against: DoReMi proxy sizes gave non-monotonic 8B accuracy (1B proxy worse than 280M) and different weights (Table 5, Table 8). Olmix found 1M proxies unreliable (ρ = 0.73). Data Mixing Laws cite studies in which rankings change with size and tokens (footnote 7). Status: rank transfer from proxies of roughly 15M parameters or more is reported by two studies, RegMix (if Olmix's reading of its implementation is correct) and Olmix; the threshold itself is a **Result (single study)**.

**The Llama 3 process** ([[llama-3]] §3.1.2-§3.1.3, §3.4; Llama Team, arXiv 2024-07). (1) A knowledge classifier labels the type of information in web data and is used to downsample categories over-represented on the web, "for example, arts and entertainment". (2) Scaling-law experiments train several small models per candidate mix and predict a large model's performance; this is repeated to select a candidate. (3) A larger model is trained on the candidate and evaluated on key benchmarks. (4) The final mix is roughly 50% general knowledge, 25% mathematical and reasoning, 17% code, and 8% multilingual tokens. (5) During 405B training the mix was adjusted: more non-English data, upsampled math, more recent web data late, and downsampled lower-quality subsets; percentages are not printed (§3.4.1). (6) Annealing on GSM8K and MATH training sets improved a pre-trained 8B model's validation scores by 24.0% and 6.4% but gave negligible gains at 405B, and benchmark training sets were excluded from annealing data "to assess the true few-shot learning capabilities and out-of-domain generalization" (§3.1.3). (7) To value a small new dataset, a 50%-trained 8B model is annealed linearly to LR 0 over 40B tokens with 30% weight on the dataset and 70% on the default mix (§3.1.3).

**Conditions and limits.** Llama 3 prints no per-candidate results, no classifier accuracy, and no held-out comparison of mixes. The 8B-versus-405B annealing result is one pair of models on two math benchmarks, and the report's explanation (strong in-context learning at 405B) is an **Interpretation**.

**Held-out suites used by open reports.** SmolLM2 reports MMLU-Pro, Natural Questions, and TriviaQA as benchmarks "not monitored during training" ([[smollm2]] §4.7, Table 4). OLMo 2 separates development benchmarks from held-out evaluations in its mid-training table and used only 200 of 1,319 GSM8K examples for development ([[olmo-2]] Table 9, footnote 6). OLMo 3 used RULER to develop its long-context recipe and kept HELMET as an unseen suite, noting that the two suites overlap, so HELMET is "not a perfect held-out suite" ([[olmo-3]] §3.6).

**Implication.** Scale transfer is a measured property of a proxy size, a metric, and a task set. A mixture recipe should carry that measurement, and a gain concentrated on the targeted benchmark at small scale, as in the Llama 3 annealing experiment, is a reason to test the claim at the target scale on held-out tasks.

## §5 Stage-specific mixtures

**Definition.** A stage mixture is the mixture used for one training stage with its own token budget and objective. The stages below differ in what the mixture is chosen to change.

| Stage | What the mixture is chosen to change | Disclosed example | Measured trade-off |
|---|---|---|---|
| Pretraining, stable LR | broad coverage over all domains | OLMo 3 6T mix: Common Crawl 76.1%, olmOCR PDFs 13.6%, Stack-Edu 6.89%, FineMath 2.56%, arXiv 0.86%, Wikipedia 0.04% ([[olmo-3]] Table 4) | mixing over web topics improved 1B BPB by 0.056 on average; 13 of 54 tasks degraded, none by more than 0.035 (§3.4) |
| Multi-stage pretraining | capability gaps seen at checkpoints | SmolLM2 1.7B, four stages (table below) | math 3.21 → 22.07 and code 8.87 → 23.21 across stages ([[smollm2]] Table 3) |
| Annealing / mid-training | targeted skills during LR decay | OLMo 2 Dolmino 50B; OLMo 3 Dolmino 100B | skewed mixes trade math/code against QA ([[olmo-3]] Table 7) |
| Long-context extension | long-range use without short-context loss | §6 | 66% long: −2.5; 34% long: −0.8 ([[olmo-3]] §3.6.3) |
| SFT | instruction format and capability coverage | Llama 3.1 Table 7 shares of examples | adjusted each round "to tune performance across a wide range of benchmarks" ([[llama-3-recipe]]) |

**SmolLM2 1.7B stage mixtures** ([[smollm2]] §4.2-§4.6, Figure 2; Hugging Face, arXiv 2025-02). Stage 1 (0-6T tokens): English web 90% (FineWeb-Edu 60 / DCLM 40), StarCoderData 10%, no math. Stage 2 (6-8T): web 75%, code 20%, OpenWebMath 5%. Stage 3 (8-10T): web 74% (FineWeb-Edu 40 / DCLM 60), code 16% (Stack-Edu), math 10%. Stage 4, the decay phase (10-11T): web 58%, Stack-Edu 24%, math 14% including 0.02% AugGSM8K, Cosmopedia v2 4%. The authors describe the approach as "online": they monitored benchmarks after each stage and changed the mixture, rather than running several from-scratch schedules, because pretraining cost about $250,000 of GPU compute (§4). This is a sequence of single decisions without a control run per stage, so per-stage gains cannot be separated from continued training and LR decay.

**OLMo 2 mid-training** ([[olmo-2]] §4; Ai2, arXiv 2025-01). The 50B Dolmino mix for OLMo 2 7B is 47.2% filtered DCLM, 16.6% decontaminated FLAN, 2.45% StackExchange Q&A, 5.85% peS2o, 7.11% Wikipedia/Wikibooks, and 20.8% math (Table 13); the math portion includes the GSM8K train split (Table 5). In 50B-token candidate runs from a 4T-token 7B checkpoint, LR decay on the pretraining mix alone raised OLMES from 69.6 to 74.0 and left the GSM8K development score at 27.0 (from 28.5); adding math and instruction data gave OLMES 75.7, OLMES-Gen 70.2, and GSM* 46.5 (Table 11). Microanneals are short annealing runs from the pretrained 7B checkpoint. In the first one, a math mixture of about 200M tokens (TuluMath, DolminoSynthMath, MetaMath, CodeSearchNet, and the GSM8K train split) mixed with DCLM gave GSM* 63.5 at 35% math and 61.0 at 10% math, against 28.5 before annealing; the authors read this as "not strictly necessary to have a large proportion" of domain-specific data (§4.4, Table 12).

**OLMo 3 mid-training trade-offs** ([[olmo-3]] §3.5.4, Table 7; arXiv 2025-12). In 100B-token exploratory anneals, a Gen-QA mix without math, code, and thinking data scored Math 27.5 and Code 11.9 against 57.3 and 31.2 for the final mix, with GenQA 72.5 against 73.1. A math-code-thinking mix scored Math 60.8 and Code 35.6 but MC Non-STEM 69.6 against 77.4. The final mix was highest on MC STEM (66.4) and GenQA (73.1), 0.7 below the Gen-QA mix on MC Non-STEM, and between the two skewed mixes on Math, Code, and FIM.

**Code share depends on the stage** ([[to-code-or-not-to-code]]; Cohere, arXiv 2024-08). The authors recommend "a balanced mixture of code and text data during pre-training from scratch", "a relatively lower code percentage during continual pre-training", and code data in the cooldown mixture (§3.6). Their cooldown with 20% code over 40B tokens gave 3.6%, 10.1%, and 20% relative gains in reasoning, knowledge, and code over no cooldown, while cooldown without code gave no reasoning or code gain (§3.5). The code-proportion sweep behind the first recommendation is in ch-09.

**Benchmark training sets in stage mixes.** Llama 3 excluded them from annealing (§3.1.3). OLMo 2 included the GSM8K train split and reported GSM8K on the 1,119 test examples not used for development ([[olmo-2]] Table 5, footnote 6). SmolLM2 put 0.02% AugGSM8K, an augmented GSM8K training set, in stage 4 ([[smollm2]] §4.5). A GSM8K gain from these mixes measures a stage that saw GSM8K-format training data; a held-out math suite is needed to read it as a general gain.

**SFT mixtures.** Llama 3.1's SFT data is 52.66% general English, 14.89% code, 3.01% multilingual, 8.14% exam-like, 21.19% reasoning and tools, and 0.11% long context, as shares of examples, with some sources epoched multiple times ([[llama-3-recipe]], Table 7). The SFT mixture is a separate decision with interference effects of its own; it is the subject of ch-30b. Domain mixing for RL prompts is in ch-44b.

## §6 The long-context stage mixture: length upsampling and long:short ratios

**Definition.** A long-context stage continues training a pretrained model on longer sequences. Its mixture sets two quantities: the share of long documents, and the domain composition of both the long and the short parts.

**Problem.** Document length and source domain are confounded, because long documents come mostly from books, code repositories, and papers ([[long-context-data-engineering]] §3). Raising the long share by upsampling long documents also shifts the domain mixture, and a shifted mixture can raise loss on short web text.

**Mechanism: four data strategies compared by Fu et al.** ([[long-context-data-engineering]] §3; arXiv 2024-02; LLaMA-2 7B on SlimPajama, 5B tokens packed to 80K, constant LR 2e-5, batch 4M tokens).
1. **Cut at 4K / cut at 128K:** keep the original mixture; about 30% of SlimPajama documents are naturally longer than 4K (Figure 2).
2. **Global upsampling:** upsample long documents regardless of source, which shifts the domain mixture.
3. **Upsample Arxiv / Book / Github:** raise the long domains, which shifts both length and domain.
4. **Per-source upsampling:** keep SlimPajama's domain shares (67% CommonCrawl, 15% C4, 4.5% GitHub, 4.5% Wikipedia, 4.5% books, 2.5% ArXiv, 2.0% StackExchange) and raise the share of sequences longer than 4K within each source from about 30% to about 70% (§5.3).

**Worked example (toy numbers).** Web is 90% of tokens with 20% of its tokens in long documents; books are 10% with 90% long. Overall long share is 0.9 · 0.2 + 0.1 · 0.9 = 27%. Multiplying the sampling weight of every long-document token by 3 (global upsampling) gives web 54 + 72 = 126 and books 27 + 1 = 28 units, so books rise from 10% to 28/154 = 18.2% of tokens and the long share becomes 52.6%. Applying the same factor within each source and then restoring the 90/10 split (per-source upsampling) gives long shares of 42.9% in web and 96.4% in books, 48.2% overall, with books still at 10%.

**Evidence.**
- Fu et al., Table 5 (loss change against the original mixture, 0-4K context): per-source upsampling changed C4 / CommonCrawl / StackExchange loss by +0.002 / +0.008 / −0.001; global upsampling by +0.008 / +0.010 / +0.015; upsampling books by +0.010 / +0.016 / +0.021 and raised GitHub loss by +0.029; upsampling code raised book loss by +0.030. The 7B per-source model scored 88.0 on Needle-in-a-Haystack and 43.3 on MMLU (Table 3). The original mixture reached nearly the same validation loss as the per-source mixture but lower retrieval accuracy (Figure 4).
- ProLong ([[prolong]]; Princeton, arXiv 2024-10; Llama-3-8B, 5B-token ablations at 64K evaluated after SFT): as the long share rose to 100%, PG19 perplexity kept improving while the downstream long-context average fell, and the short-task average fell monotonically with the long share; 60% long plus 40% short data had the best average (§2.1, Figure 1; §3.2, Figure 3). Books and code repositories at 1:1 were the best long source (54.6), and a curated ShortMix was the best short source (Table 4, Table 6). Fine-tuning Llama-3-8B on Fu et al.'s SlimPajama long mix lowered MMLU from 66.5 to 63.1 and GSM8K from 44.7 to 40.6 (Table 2), and in a matched reproduction Fu et al.'s mix scored 51.8 long / 65.4 short against 54.6 / 67.5 for ProLong's (App. B.6, Table 24).
- Released ratios: ProLong-64k-Base used 30% code repositories, 30% books, 3% textbooks, and 37% ShortMix over 20B tokens ([[prolong-recipe]], Table 9). Qwen2.5-1M stages at 65,536 to 262,144 tokens used 75% sequences at the current maximum length and 25% shorter ([[qwen-long-context-synth]] §3). OLMo 3 mixed 34% long documents from its Longmino pool with 66% Dolmino mid-training data for 50B (7B) and 100B (32B) tokens; in a 10B-token test, 66% long cost 2.5 points on a subset of OlmoBaseEval against 0.8 for 34% long ([[olmo-3]] §3.6, §3.6.3). SmolLM2 extended from 2K to 8K with 40% documents of at least 8K tokens (DCLM 10%, FineWeb-Edu 10%, Dolma books 20%) and 60% of the stage-4 mixture, from a checkpoint before the final 75B tokens, and reports "next to no degradation" after extension ([[smollm2]] §4.6-§4.7).
- Llama 3.1 405B increased context from 8K to 128K in six stages over about 800B tokens, and advanced a stage only when short-context evaluations had "recovered completely" and needle-in-a-haystack was solved at that length ([[llama-3]] §3.4.2). The report prints no long/short ratio.
- SFT after the long stage: Llama 3 found that SFT with only short-context data regressed long-context ability and used a 0.1% synthetic long-context share ([[llama-3]] §4.3.4); ProLong found in its SFT ablations that short UltraChat data alone gave a long-context average of 55.7, against 54.1 with 1% and 43.3 with 50% synthetic long instruction data (§5, Table 8). The two results come from different amounts of long-context training and different evaluations.

**Context window scheduling in pretraining.** SkyLadder ([[skyladder]]; arXiv 2025-03) changes the attention window, not the data mixture: at a fixed 100B-token budget, 1B models pretrained with shorter windows scored higher on nine standard benchmarks, and growing the window linearly from 32 tokens to 8K raised the standard average from 46.3 to 50.0 on CommonCrawl (Table 1). The authors argue that ordering data by document length would introduce domain bias, because long documents cluster in domains such as books (§2, §4.6; Interpretation). This is the same confound that Fu et al. measured in Table 5.

**Conditions and limits.** Fu et al. evaluate with per-domain loss, needle retrieval, BookQA, and MMLU, without SFT. ProLong's ablations are at 8B on Llama-3. OLMo 3's ratio test is 10B tokens on a subset of its suite. None of the reports ablates the ratio at more than one model size.

**Implication.** For breadth, the long-context stage is a mixture problem with two constraints: keep the domain shares of general data and keep enough short data that short-context evaluations do not fall. Evaluate the stage on a short-context suite and on long-context tasks after SFT, because perplexity on long text and needle retrieval did not rank the mixtures the same way as downstream long-context scores (ProLong §2.1; Fu et al. Figure 4).

## §7 Cross-domain transfer from composition

The sections above show that a mixture decision for one domain changes performance on others. The measured cases, with their status:

- **Diverse web text supports many tasks.** DoReMi lowered Wikipedia's weight (0.0919 → 0.0699) while Wikipedia-derived TriviaQA rose from 24.55 to 34.86 ([[doremi]] Table 1, Table 5). RegMix found Pile-CC validation loss the most correlated with downstream performance among Pile domains, and training on Pile-CC alone matched DoReMi's weights at 46.8 ([[regmix]] §5.2, Table 4). **Replicated** across two studies on The Pile; the authors' explanation for DoReMi (medium-entropy domains transfer to all domains, App. D) is an **Interpretation**.
- **Code helps reasoning up to a share, then costs knowledge.** Python up to 50% of 84B tokens did not lower a 19-task natural-language average at 4.2B and raised bAbI from 0.0 to 23.2 ([[data-constrained-scaling]] §7, App. M). In the paper's 470M-scale ablations (§3.3 does not restate the size), 25% code gave the best reasoning score and 75% code lowered world knowledge by 31% relative to no code ([[to-code-or-not-to-code]] §3.3). **Replicated** for the reasoning direction; the knowledge cost is from one study.
- **Targeted mixtures transfer partly.** In 1B models on 100B tokens, a QA-targeted mixture also lowered math and code BPB relative to the natural distribution (0.643 vs 0.719; 0.535 vs 0.592), and each single-target mixture had the lowest BPB on its own target; the multi-objective mixture was lowest on no column but below the natural distribution on all three ([[olmo-3]] Table 38). A mixture targeted at MMLU lowered HellaSwag and PIQA ([[weborganizer]] Table 10). Result (single study each).
- **Mid-training additions change other skills.** Adding math data to OLMo 2's high-quality web mid-training mix raised OLMES-Gen from 63.8 to 69.7 and lowered MMLU from 63.1 to 62.3 ([[olmo-2]] Table 11). Upsampling one long domain raised short-context loss on other domains ([[long-context-data-engineering]] Table 5).
- **Mixing laws encode interaction.** A negative t_ij in the Data Mixing Laws form means that training data from domain j lowers validation loss on domain i. Among five coarse Pile domains, most pairs had little relationship, with facilitation in some pairs (dialogue data for internet text) and conflict in others (symbolic data for prose) ([[data-mixing-laws]] §3.2, Figure 4).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DoReMi proxy | 280M | pretrain | DRO step size η; smoothing c; optimizer | 1; 1e-3; Adafactor | arXiv:2305.10429v4 §2, Algorithm 1 ([[doremi]]) | verified 2026-09-14 | "did not extensively tune" (§2) |
| DoReMi reference and proxy | 280M | pretrain | reference weights; steps; batch | Pile default weights (uniform for GLaM); 200k steps (300k for GLaM); 512 sequences × 1,024 tokens (104.9B tokens, derived: 200,000 × 512 × 1,024) | §3.1, App. C | verified 2026-09-14 / derived | Table 5: 280M proxy 26.56 avg vs 21.11 (70M), 22.51 (150M), 22.35 (1B) |
| DoReMi (280M) → 8B, The Pile | 8B main | pretrain | mixture (token-based sampling weight), top 6 | Pile-CC 0.6057; OpenWebText2 0.1019; Wikipedia (en) 0.0699; YoutubeSubtitles 0.0502; PhilPapers 0.0274; Books3 0.0224 (22 values in Table 1) | Table 1 | verified 2026-09-14 | Table 5, Table 3a vs default weights |
| RegMix proxies | 1M (public code "closer to 15M" per [[olmix]] fn. 1) | pretrain (proxy) | runs; tokens; regression; target; aggregation | 512 runs; 1B tokens each; LightGBM; Pile-CC validation loss; mean of top 100 predicted mixtures | arXiv:2407.01492v2 §3-§4 ([[regmix]] excerpt) | verified 2026-09-15 | Table 2: ρ 97.12 at 1B; Table 4: 47.3 vs 45.1 (Pile weights) |
| Data Mixing Laws pipeline | 70M, 160M, 305M, 410M → 1B | pretrain (proxy) | proxy tokens; mixtures; batch; schedule | 30B tokens; 20 mixtures (chosen from 40 on 70M); 1M tokens; cosine, 2k warmup, decay to 0.1 of max at step 100k | arXiv:2403.16952v2 §4.2 ([[data-mixing-laws]] excerpt) | verified 2026-09-15 | Figure 8: default performance at 0.73 of steps |
| OlmixBase (paper setting) | 30M proxies → 1B target | pretrain (proxy) | proxy tokens; swarm size; regression; solver; repetition cap | 3B tokens (5× Chinchilla); K ≥ 3(m + 1); log-linear per task; exact solver + KL 0.05 to natural; k = 4 in experiments | arXiv:2602.12237v1 §3.3-§3.4 ([[olmix]] excerpt) | verified 2026-09-15 | Figures 3, 4, 6, 8; Table 3; Table 4 |
| Olmo 3 Base (Dolma 3 Mix) | 7B, 32B | pretrain-stable | swarm; cap; final mix of 5.93T tokens (7B trained one epoch; 32B schedule truncated at 5.5T, Figures 3-4) | 30M proxies on 3B tokens; swarm 5× number of domains; ≤ about 4-7 repeats; CC 76.1%, olmOCR PDFs 13.6%, Stack-Edu 6.89%, FineMath 3+ 2.56%, arXiv 0.86%, Wikipedia 0.04% | arXiv:2512.13961v2 §3.4, Table 4 ([[olmo-3]] excerpt) | verified 2026-09-15 | §3.4: +0.056 avg BPB at 1B vs natural; Table 38 |
| Llama 3.1 (all) | 8B-405B | pretrain-stable | final mix (token share) | roughly 50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual | arXiv:2407.21783v3 §3.1.2 ([[llama-3-recipe]]) | verified 2026-09-14 | scaling-law experiments per candidate; no table |
| Llama 3 8B (experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50%-trained model; LR linear to 0 over 40B tokens; 30% new data / 70% default mix | §3.1.3 | verified 2026-09-14 | no numbers reported |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | annealing | final 40M tokens; LR linear to 0 at 128K context; high-quality sources upsampled (shares not printed); no benchmark training sets; Polyak averaging | §3.4.3, §3.1.3 | verified 2026-09-14 | no ablation of this mix reported; related §3.1.3 test: annealing on GSM8K/MATH train sets gave +24.0% / +6.4% at 8B, negligible at 405B |
| Llama 3.1 405B | 405B | long-context | stages; tokens; gate | six stages 8K → 128K; approximately 800B tokens; long/short ratio not printed | §3.4.2 | verified 2026-09-14 (ratio not reported) | gate: short-context evals fully recovered and NIAH solved |
| Llama 3.1 | not scoped | SFT | data composition (% of examples) | general English 52.66; code 14.89; multilingual 3.01; exam-like 8.14; reasoning and tools 21.19; long context 0.11 | v3 Table 7, §4.3.4 | verified 2026-09-14 | short-only SFT regressed long context; no table |
| SmolLM2 1.7B | 1.7B | pretrain-stable, stage 1 | tokens; mix | 0-6T; web 90% (FineWeb-Edu 60 / DCLM 40), StarCoderData 10% | arXiv:2502.02737v1 §4.2, Figure 2 ([[smollm2]] excerpt) | verified 2026-09-15 | 10% chosen to "ensure approximately 4 epochs over 11T tokens"; no ablation |
| SmolLM2 1.7B | 1.7B | pretrain-stable, stage 2 | tokens; mix | 6-8T; web 75%, code 20%, OpenWebMath 5% | §4.3, Figure 2 | verified 2026-09-15 | Table 3 after stage 1: math 3.21, code 8.87 |
| SmolLM2 1.7B | 1.7B | pretrain-stable, stage 3 | tokens; mix | 8-10T; web 74% (FineWeb-Edu 40 / DCLM 60), Stack-Edu 16%, math 10% | §4.4, Figure 2 | verified 2026-09-15 | annealing ablations: more DCLM raised MMLU MCF (§4.3) |
| SmolLM2 1.7B | 1.7B | pretrain-decay/anneal | tokens; mix; LR | 10-11T; web 58%, Stack-Edu 24%, math 14% (incl. OWM 0.08%, AugGSM8K 0.02%), Cosmopedia v2 4%; LR linear to 0 over 10% of training | §4.5, Figure 2 | verified 2026-09-15 | Table 3: math 7.27 → 22.07, code 16.75 → 23.21 |
| SmolLM2 1.7B | 1.7B | long-context | lengths; mix; RoPE | 2K → 8K; 40% docs ≥ 8K tokens (DCLM 10%, FineWeb-Edu 10%, Dolma books 20%) + 60% stage-4 mix; RoPE 130k; from checkpoint before final 75B tokens | §4.6 | verified 2026-09-15 | "next to no degradation" (§4.7); no ratio ablation |
| OLMo 2 7B | 7B | mid-train | Dolmino 50B mix (% of mix) | filtered DCLM 47.2; FLAN (decontam.) 16.6; StackExchange Q&A 2.45; peS2o 5.85; Wikipedia/Wikibooks 7.11; Dolmino Math 20.8 (includes GSM8K train) | arXiv:2501.00656v3 Table 13, Table 5 ([[olmo-2]] excerpt) | verified 2026-09-15 | Table 11: Web + Math + Ins 75.7 OLMES vs PT Mix 74.0; three 50B runs averaged (Table 9) |
| Olmo 3 7B / 32B | 7B, 32B | long-context | context length; long:short; tokens | 8,192 → 65,536 tokens; 34% Longmino long documents / 66% Dolmino mix; 50B (7B), 100B (32B) | arXiv:2512.13961v2 §3.6, §3.6.3, Table 11 | verified 2026-09-15 | 10B-token test: −0.8 (34% long) vs −2.5 (66% long) on OlmoBaseEval subset |
| ProLong-64k-Base | 8B | long-context | mix; tokens; length | 30% code repos, 30% books, 3% textbooks, 37% ShortMix; 20B tokens; 64K | arXiv:2410.02660v4 Table 9 ([[prolong-recipe]]) | verified 2026-09-14 | Fig. 3: 60% long best; Table 4: books/repos 1:1 54.6 |
| Qwen2.5-1M | 7B, 14B | long-context | length mix, stages 3-5 | 75% at current max length, 25% shorter; tokens per stage not reported | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | no ablation reported |
| LLaMA-2 7B long-context (Fu et al.) | 7B | long-context | strategy; tokens; LR; batch; length | per-source length upsampling (>4K share about 30% → about 70% per source); 5B tokens; constant 2e-5; 4M tokens; packed to 80K | arXiv:2402.10171v1 §4, §5.3 ([[long-context-data-engineering]] excerpt) | verified 2026-09-15 | Table 5 per-domain loss; RoPE base changed "as in Xiong et al." (value not printed) |
| To Code balanced → text | 470M | pretrain-decay/anneal | cooldown code share; tokens; LR | 20% code; 40B tokens (10% of pretraining budget); linear anneal to 1e-6 | arXiv:2408.10914v1 §3.5 ([[to-code-or-not-to-code]] excerpt) | verified 2026-09-15 | +3.6% reasoning, +10.1% knowledge, +20% code vs no cooldown |

**Starting point for a small general-purpose run.** For a 1B-parameter model trained on 100B tokens over tens of domains, the only fully specified mixture-search configuration in the table is OlmixBase: 30M proxies on 3B tokens, at least 3(m + 1) proxy runs, one log-linear model per task, an exact solver with KL 0.05 toward the natural distribution, and a repetition cap of 4 (1B targets, 100B tokens, 52 BPB tasks). The proposed mix is then validated by training the target on it and on the natural distribution; when m is small, a stratified-sampling run is a third baseline, because several methods lost to it in [[aioli-data-mixing]] Table 2. If a from-scratch run is staged, SmolLM2's four stage mixtures are the verified sequence for a 1.7B model on 11T tokens; its stage-4 decay mix (web 58%, code 24%, math 14%, synthetic textbooks 4%) is a candidate annealing mix, with the AugGSM8K component removed if GSM8K is part of the held-out suite. For a context-extension stage, the verified ratios are 34% long / 66% short (Olmo 3 7B, 50B tokens, 8K → 65K) and 40% / 60% (SmolLM2 1.7B, 2K → 8K). When a context-extension stage is run, advancing it only after short-context evaluations recover is the gate Llama 3.1 used at 405B.

## Generalization lens

**(a) What increases breadth.**
- A worst-domain objective without task data: DoReMi's weights improved all 22 Pile domains and the worst-case domain log-perplexity at 8B ([[doremi]] Table 3a).
- Per-task regression targets and regularization toward the natural distribution: BPB 0.765 vs 0.774 (per-task vs aggregated) and 0.7647 vs 0.7772 (KL 0.05 vs exact) at 1B ([[olmix]] Table 3, Figure 8).
- A balanced mid-training mix: the final OLMo 3 mix scored 73.1 on GenQA against 72.5 for the GenQA-skewed mix while keeping Math at 57.3 against 27.5 ([[olmo-3]] Table 7).
- Keeping general short data and domain shares in long-context training ([[prolong]] Figure 3; [[olmo-3]] §3.6.3; [[long-context-data-engineering]] Table 5).

**(b) What causes narrowing or forgetting.**
- Single-target mixtures: MMLU-targeted mixing lowered HellaSwag 57.5 → 54.1 and PIQA 71.3 → 69.9 ([[weborganizer]] Table 10); skewed mid-training mixes lost 7.8 points of MC Non-STEM or 29.8 points of Math ([[olmo-3]] Table 7).
- Benchmark-format training data in a stage mix: +24.0% GSM8K at 8B that did not appear at 405B ([[llama-3]] §3.1.3).
- Long-only continued training: short-task scores fell monotonically with the long share, and 100% long lowered downstream long-context scores after SFT ([[prolong]] §3.2).
- A perplexity target read as a capability target: lower test perplexity predicted worse downstream averages across Aioli's SlimPajama models (correlation 0.529; [[aioli-data-mixing]] App. F.1).
- Unconstrained upweighting of small domains, which produced 5 repeats against a cap of 4 ([[olmix]] Table 4).

**(c) How to measure it at this stage.**
- Macro-averaged per-domain perplexity or BPB with the worst domain reported, on decontaminated evaluation text ([[paloma]] §3-§4; [[doremi]] Table 3a).
- A task suite split into development tasks used to fit the mixture and held-out tasks never used for selection ([[smollm2]] Table 4; [[olmo-2]] Table 9; HELMET kept unseen, with task overlap with RULER, in [[olmo-3]] §3.6).
- Proxy-to-target rank correlation for the metric used, and a target-scale comparison against natural and stratified mixtures ([[olmix]] Figure 3; [[aioli-data-mixing]] Table 2).
- For long-context stages, a short-context regression gate and long-context tasks scored after SFT ([[llama-3]] §3.4.2; [[prolong]] §2.2).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Sampling minibatches from α when reimplementing DoReMi | A rerun on The Pile does not reproduce the published Table 1 weights | Sample uniform domain weights; apply α to the loss (Algorithm 1) |
| Omitting per-token clipping or smoothing in DoReMi | A domain where the proxy beats the reference on some tokens loses weight | Clip token-level excess at 0; set c = 1e-3 |
| Using the last α instead of the average ᾱ | Returned weights change with the stopping step | Return (1/T) Σ α_t |
| Transferring DoReMi weights from one proxy size to all targets | A different proxy size gives a different top domain | Compare weights from two proxy sizes (Table 8) and validate at target scale |
| Ignoring domain supply when optimizing | Proposed weights imply more than 4-5 epochs for a small domain | Compute e_i = p_i R / N_i for every row; add p_i ≤ k N_i / R to the optimizer |
| Using 1M-parameter proxies for decisions | Rankings disagree with a 1B check run | Measure proxy-target Spearman ρ; Olmix found 0.73 at 1M |
| Optimizing one validation domain or one benchmark | Target improves while other tasks fall | Fit per-task models; report non-target tasks and a held-out suite |
| Treating lower perplexity as better downstream | Mixture with best perplexity has a lower task average | Report both; Aioli App. F.1 found them anti-correlated |
| Upsampling long documents globally | Book or code share rises; web short-context loss increases | Compare domain shares before and after upsampling; per-domain loss at 0-4K |
| Leaving benchmark training sets in annealing data without a held-out check | Large gain on that benchmark only | Remove them, or compare with a held-out suite in the same domain |
| Reporting a stage mix without tokens, unit, and stage | Mix cannot be reproduced; token vs example shares confused | Record R, N_i, share unit, stage, and model size per row |

## Check your understanding

1. DoReMi samples minibatches uniformly and applies α to the loss, while the final model is trained on data resampled from ᾱ. Explain why the 1B proxy could be worse than its baseline on all 22 domains while its weights still sped up the 1B main model, using the authors' explanation.
2. In the worked example, clipping raised domain B's excess loss from −0.10 to 0.10. Explain what the clip does to the objective and why Group DRO needs a non-negative loss.
3. Pile-CC received 0.6057 of DoReMi's weight at 280M but 0.1199 at 1B, while OpenWebText2 went the other way. Explain why both results can be consistent with a worst-group objective, and what experiment would decide which weights to use for a 70B model.
4. RegMix reports ρ = 97.12 from 1M proxies to 1B targets, and Olmix reports 0.73 for 1M proxies. Give two differences in setup that could produce the disagreement, including the footnote about RegMix's implementation.
5. Aioli finds that lower test perplexity predicted worse downstream averages. Explain how a mixture method that minimizes average perplexity per group could still narrow a model's capabilities.
6. Llama 3 found a 24.0% GSM8K gain from annealing on GSM8K training data at 8B and a negligible gain at 405B. Explain why this result is evidence about generality measurement rather than about math data quality.
7. Using Fu et al. Table 5 and the per-source worked example, explain why global length upsampling raised short-context web loss while per-source upsampling did not, and why ProLong still found a short-context cost for Fu et al.'s mix on Llama-3-8B.
8. OLMo 3's math-code-thinking mid-training mix raised Math from 57.3 to 60.8 but lowered MC Non-STEM from 77.4 to 69.6. Propose a mixture-search objective that would have found the final mix, and state what held-out evidence would show it generalizes.

## Connections

- Previous and dependency: ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining (how exposure frequency within a mixture affects what is retained).
- Next: ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (the multilingual share as a mixture decision).
- ch-09 — Pretraining Data Composition and Capability Coverage (source removal and code-proportion evidence behind §7).
- ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability (the step and size power laws nested in §3).
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (topic quotas and benchmark-targeted selection).
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic (unique tokens N_i before mixing).
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (value of repeated tokens behind §1's cap).
- ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (full recipe rows).
- ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement (applies §4 and the generalization lens).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (depends on this chapter; annealing mixtures in depth).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (continues §6).
- ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (SFT mixtures).
- ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing (mixtures of RL prompts).
- ch-48 — Contamination Detection and Its Effect on Reported Scores (benchmark data in stage mixes).

## Sources

- [[doremi]] — objective, Algorithm 1, η and c, Pile weights (Tables 1, 8), 8B results, proxy-size and same-size ablations, GLaM iterated DoReMi, limits.
- [[data-mixing-laws]] — Eq. 7 mixing law, fitting errors (Table 1), nested scaling-law pipeline, 1B/100B RedPajama result, continued-pretraining use (chapter excerpt; no library card).
- [[regmix]] — 512 × 1M proxy regression, rank correlations (Table 2), mixture spread (Table 3), comparison with Pile weights and DoReMi (Table 4) (chapter excerpt).
- [[aioli-data-mixing]] — LMO framework, stratified-sampling comparison (Table 2), restricted setting, perplexity-downstream correlation (App. F.1) (chapter excerpt).
- [[olmix]] — proxy size, swarm size, regression form, per-task targets, repetition constraints, KL solver, mixture reuse (chapter excerpt).
- [[llama-3]], [[llama-3-recipe]] — knowledge classifier and scaling-law mix search, final mix, annealing experiments, long-context stages and gate, SFT composition.
- [[smollm2]] — four-stage pretraining mixtures, per-stage results, context-extension mix, held-out benchmarks (chapter excerpt).
- [[olmo-2]] — Dolmino mix compositions (Tables 5, 13), mid-training candidate mixes (Table 11), microanneals, held-out split (use the chapter excerpt for verified values).
- [[olmo-3]] — Dolma 3 Mix (Table 4), constrained and conditional mixing, Dolmino trade-offs (Table 7), targeted mixes (Table 38), Longmino long:short ratio (use the chapter excerpt).
- [[long-context-data-engineering]] — four length/domain strategies, per-domain loss changes (Table 5), training settings (use the chapter excerpt; the library card has unverified values).
- [[prolong]], [[prolong-recipe]] — long/short ratio ablation, long and short sources, comparison with Fu et al., released mix.
- [[qwen-long-context-synth]] — 75% / 25% length mix in Qwen2.5-1M stages.
- [[skyladder]] — context-window scheduling and the length-domain confound.
- [[to-code-or-not-to-code]] — stage-dependent code-share recommendation and cooldown with code (chapter excerpt).
- [[data-constrained-scaling]] — effective value of repeated tokens; code up to 50% at 4.2B.
- [[weborganizer]] — MMLU-targeted mixture and its effect on non-target tasks.
- [[paloma]] — macro-averaged per-domain perplexity for measuring breadth.
- [[datadecide]] — small-scale decision accuracy and its task dependence.
