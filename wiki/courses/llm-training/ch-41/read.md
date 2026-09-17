<!-- chapter: ch-41
     track: preference
     kind: content
     title: Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization
     deps: [ch-15]
     sources: [[bradley-terry-rm]], [[reward-model-overoptimization]], [[reward-ensembling]], [[helping-or-herding]], [[warm-weight-averaged-reward-models]], [[pairrm]], [[generative-reward-models]], [[rlaif-scaling]], [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]], [[rewardbench]], [[rm-bench]], [[ppe-reward-model-eval]], [[deepseek-grm]], [[policy-coverage-loss]], [[training-verifiers-to-solve-math-word-problems]], [[v-star]], [[prm800k]], [[west-of-n]], [[dpo]], [[llama-2-recipe]], [[llama-3-recipe]], [[rlhf-instructgpt]], [[rubrics-as-rewards]]
     figures: figures/rm-overopt.html
     revised: 2026-09 (generality revision)
-->

# Chapter 41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization

> **Core insight.** A reward model (RM) trained with the Bradley–Terry loss learns score differences on the response distribution it was trained on, and a policy optimized against it moves to responses outside that distribution. In a synthetic setup with a 6B "gold" RM, the gold score first rises and then falls as √KL grows, and larger proxy RMs move the peak outward without removing it ([[reward-model-overoptimization]] §3.2, Fig. 1, Fig. 3). Accuracy on a static RM benchmark is a weak predictor of the policy that the RM produces: in a controlled DPO experiment, Nemotron-4-340B-Reward (RewardBench 92.0) produced a policy with Arena score 1172, within the confidence interval of the untrained base model (1178), while Athene-RM-8B produced 1209 ([[ppe-reward-model-eval]] Table 3; [[nemotron-4-synthetic]] Table 4).
>
> **Guideline.** When an RM will be optimized against with RL or best-of-n, measure it on the policy's own samples and on held-out domains (PPE-style correctness sets, RM-Bench style-controlled pairs), and track a separate evaluator as a function of √KL, because benchmark accuracy on fixed pairs does not predict the downstream result ([[ppe-reward-model-eval]] §7; [[rm-bench]] §5.2). When several fine-tuned RMs from one pretrained checkpoint are affordable, use a conservative ensemble (worst-case or uncertainty-weighted) or a weight average, because both reduced over-optimization in their studies ([[reward-ensembling]] §5; [[warm-weight-averaged-reward-models]] §5.2); when the RMs share a pretraining checkpoint, expect shared errors that no aggregation removes ([[helping-or-herding]] Abstract). When the task has a checkable answer, train the verifier on the current policy's own correct and incorrect samples and refresh it as the policy changes ([[training-verifiers-to-solve-math-word-problems]] §4.2; [[v-star]] §3).

---

## Why this chapter matters for a general-purpose model

The pipeline position of this chapter is: preference data (ch-15) → reward model (this chapter) → policy optimization (ch-37 to ch-40). An RM converts pairwise human or AI judgments into a scalar that an optimizer can maximize. For a general-purpose model the RM has to score responses in every domain the prompt distribution covers, including domains with no checkable answer, and it has to keep scoring correctly after the policy has changed.

Three measurable problems follow from this position:

1. **Distribution shift.** The RM is trained on responses from earlier models, and RL moves the policy to new responses. Gao et al. call the resulting failure extremal Goodhart and expect it to cause most of the decline in gold score ([[reward-model-overoptimization]] §4.2.2, Interpretation).
2. **Uneven domain accuracy.** An RM can be accurate on chat and near random on math or code. On RM-Bench, the highest-scoring RM has 28.4% hard accuracy on math and 30.7% on code ([[rm-bench]] §4.1, Tables 14-15). A policy trained with such an RM receives a wrong signal in exactly those domains.
3. **Measurement.** RM quality is usually reported as accuracy on fixed preference pairs, but the quantity that matters is the quality of the policy trained with the RM. These two are only partly correlated ([[ppe-reward-model-eval]] §7; [[rm-bench]] §5.2).

Each section below addresses one of these problems: the loss and what it identifies (§1), over-optimization (§2), robustness methods (§3), out-of-distribution evaluation and generalist RMs (§4), and policy coverage as the condition under which an imperfect RM still transfers (§5).

---

## §1 The Bradley–Terry reward model

### §1.1 Definition and the problem it solves

A Bradley–Terry (BT) reward model is a network that outputs one scalar score per (prompt, response) and is trained so that the difference of two scores predicts which response a labeler preferred. It addresses the problem that humans give reliable relative judgments but not calibrated absolute scores, while RL needs a scalar per response.

### §1.2 Mechanism

1. Collect triples `(x, y_w, y_l)`: prompt, preferred response, dispreferred response (ch-15 covers protocols and agreement).
2. Build the scorer: a pretrained or SFT language model with the unembedding layer removed and a linear head that outputs a scalar. InstructGPT removes the final unembedding layer of the SFT model ([[rlhf-instructgpt]] §3.5); Stiennon et al. add a randomly initialized linear head ([[bradley-terry-rm]], Stiennon §3.4).
3. Score both responses in a pair and apply the logistic loss to the score difference.
4. Before RL, shift the scores by a constant so that a reference set has mean 0 (§1.4).

### §1.3 Formula

```
p*(y1 ≻ y2 | x) = exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2))) = σ(r*(x,y1) − r*(x,y2))        (DPO §3 Eq. 1)
L_R(r_φ) = − E_{(x, y_w, y_l) ~ D} [ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]                          (DPO §3 Eq. 2)
```

`x` is the prompt; `y1, y2` are two responses; `r*` is the latent reward; `r_φ` is the RM with parameters `φ`; `y_w` and `y_l` are the preferred and dispreferred responses; `D` is the comparison dataset; `σ(z) = 1/(1+e^{−z})`. The loss is binary cross-entropy with label 1 and logit `r_φ(x,y_w) − r_φ(x,y_l)` (derived in [[bradley-terry-rm]]).

The derivative with respect to each score is:

```
∂L/∂r_φ(x,y_w) = −(1 − σ(Δ)),     ∂L/∂r_φ(x,y_l) = +(1 − σ(Δ)),     Δ = r_φ(x,y_w) − r_φ(x,y_l)
```

### §1.4 Worked example and identifiability

Take `r(x,y_w) = 1.2`, `r(x,y_l) = 0.4`. Then `Δ = 0.8`, `σ(0.8) = 0.690`, the loss is `−ln 0.690 = 0.371`, and each score receives a gradient of magnitude `1 − 0.690 = 0.310`. Add 5 to both scores: `Δ` is still 0.8, so the loss and gradients are unchanged.

This is the identifiability property. Reward functions that differ by any function `f(x)` of the prompt induce the same BT preference distribution (DPO §5.1 Definition 1, Lemma 1, via [[bradley-terry-rm]]). Two consequences follow.

- **Scores are not comparable across prompts.** A score of 3.0 on one prompt and 1.0 on another does not say the first response is better. Stiennon et al. normalize so that reference summaries have mean score 0; InstructGPT adds a bias so labeler demonstrations have mean 0 before RL, "since the RM loss is invariant to shifts in reward" ([[bradley-terry-rm]], Stiennon §3.4; InstructGPT §3.5).
- **Order statistics across RMs need a shared offset.** If RM A and RM B choose different offsets `C(x)`, the minimum or median of their scores is not meaningful. Eisenstein et al. add a term `η · E[(r(x,y+) + r(x,y−))²]` with small η > 0 that regularizes the sum of the two rewards in each pair towards zero, which fixes the offset before ensembling ([[helping-or-herding]] §2.1 Eq. 2).

### §1.5 K-way rankings: Plackett–Luce versus all-pairs

When a labeler ranks K responses, two different objectives are used.

**Plackett–Luce** models the whole ranking τ as a sequence of choices from the remaining items:

```
p*(τ | y1..yK, x) = ∏_{k=1}^{K}  exp(r*(x, y_τ(k))) / Σ_{j=k}^{K} exp(r*(x, y_τ(j)))            (DPO App. A.3 Eq. 18)
```

`τ(k)` is the index of the response ranked k-th. Each factor is a softmax over the items not yet chosen, so the log-likelihood is a sum of log-softmax terms, not a sum of `log σ` terms. For K = 2 the product reduces to BT ([[dpo]] App. A.3).

**All-pairs BT (InstructGPT)** expands a ranking into its C(K,2) pairwise comparisons and averages the BT loss over them: `loss(θ) = −(1/C(K,2)) E[log σ(r_θ(x,y_w) − r_θ(x,y_l))]` ([[rlhf-instructgpt]] §3.5 Eq. 1).

Worked example with three responses ranked A ≻ B ≻ C and scores (2, 1, 0):

| Objective | Computation | Negative log-likelihood |
|---|---|---|
| Plackett–Luce | first choice `e²/(e²+e¹+e⁰) = 7.389/11.107 = 0.665`; second choice `e¹/(e¹+e⁰) = 0.731`; product 0.486 | `0.408 + 0.313 = 0.721` |
| All-pairs BT, averaged | pairs (A,B): `−ln σ(1) = 0.313`; (A,C): `−ln σ(2) = 0.127`; (B,C): `0.313` | mean `0.251` |

The two objectives weight pairs differently. Plackett–Luce couples the top choice to all other items at once; all-pairs BT treats the three comparisons as separate terms that share a forward pass. InstructGPT reports that shuffling all comparisons into one dataset made the RM overfit within one pass, and that putting all C(K,2) comparisons of a prompt in one batch element fixed this ([[rlhf-instructgpt]] §3.5).

### §1.6 Evidence

- **Scale of RM and data (Result, single study).** Stiennon et al. trained 7 RMs from 160M to 13B parameters on 8k to 64k comparisons; doubling data raised validation accuracy by about 1.1% and doubling model size by about 1.8% ([[bradley-terry-rm]], Stiennon §4.3, Fig. 6).
- **Transfer (Result, single study).** TL;DR-trained RMs agree with labelers on CNN/DM 62.4% (1.3B) and 66.5% (6.7B) of the time, against 66.9% inter-labeler agreement (Stiennon §4.3).
- **Preference strength.** Llama 2's helpfulness RM has 79.1% accuracy on "significantly better" pairs and 54.5% on "negligibly better / unsure" pairs ([[bradley-terry-rm]], Llama 2 App. A.3.3 Table 28). A margin term `m(r)` that grows with rating strength changed average accuracy from 62.5 (no margin) to 63.0 (small margin) ([[llama-2-recipe]], Table 28). Llama 3 removed the margin, citing diminishing improvements after data scaling ([[llama-3-recipe]], §4.1.2).

### §1.7 Conditions and limits

- **Non-transitive preferences.** A single score per response cannot represent them; a pairwise preference model can (NLHF §3.1 via [[bradley-terry-rm]]).
- **Deterministic labels.** If `p*(y ≻ y′) = 1`, BT requires `r(y) − r(y′) → +∞`, and the optimal KL-regularized policy sets `π*(y′) = 0` for any KL strength (IPO §4.2 via [[bradley-terry-rm]]). This is a reason not to train an RM to zero loss on small datasets.
- **Length.** Stiennon's 6.7B RM prefers shortening edits 62.6% of the time, against 76.4% for humans (Stiennon §4.3), so the RM under-rewards shorter improvements relative to humans in that setting.

**Implication for a general-purpose model.** The BT loss constrains only score differences within a prompt and within the training distribution. Nothing in the loss constrains scores on responses the RM never saw, which is where RL sends the policy. The rest of the chapter is about that gap.

---

## §2 Over-optimization: proxy score versus gold score

### §2.1 Definition and measurable problem

Over-optimization is the regime in which further optimization raises the RM (proxy) score while lowering the true quality of the responses. Measuring it needs a second, trusted scorer. Gao et al. replace humans with a fixed 6B "gold" RM, label 100,000 comparisons with it (10% held out), train proxy RMs of 3M to 3B parameters on those labels, optimize a 1.2B policy against each proxy, and score the resulting samples with the gold RM ([[reward-model-overoptimization]] §2.1).

### §2.2 Mechanism and formulas

The x-axis is the square root of the KL divergence from the initial policy:

```
d := √( D_KL(π ‖ π_init) )
R_bon(d) = d · (α_bon − β_bon · d)            best-of-n
R_RL(d)  = d · (α_RL  − β_RL · log d)          PPO                                  (Gao §1)
KL_bon(n) = log n − (n − 1)/n                                                        (Gao §2, from Stiennon App. G.3)
```

`π` is the optimized policy; `π_init` the initial SFT policy; `R` the gold score, recentred so `R(0) = 0`; `α`, `β` are fitted coefficients; `n` is the number of samples in best-of-n. The RL form has infinite slope at `d = 0` and likely does not hold near the origin (footnote 1).

Setting `dR/dd = 0` gives the peak (derived in [[reward-model-overoptimization]], not printed as a formula in the paper):

```
d*_bon = α_bon / (2 β_bon)            d*_RL = exp(α_RL / β_RL − 1)
```

### §2.3 Worked example

**KL of best-of-n.** For n = 1,000: `KL = ln 1000 − 999/1000 = 6.908 − 0.999 = 5.909` nats, so `d = 2.43`. For n = 60,000: `KL = 11.002 − 1.000 = 10.00` nats, `d = 3.16`. These match the paper's "KL ≈ 6 nats" and "KL ≈ 10 nats" for the fitting and validation runs (§3.1). The KL of best-of-n grows with `log n`, so each further unit of `d` costs exponentially more samples.

**Peak location (illustrative coefficients, not values from the paper).** The paper plots but does not print α and β. Its Fig. 3 shows α_bon increasing and β_bon and β_RL decreasing as proxy RM size grows, with α_RL held constant across sizes (§3.2, Fig. 3). Take two illustrative proxy RMs:

| Proxy RM (illustrative) | α_bon | β_bon | d* = α/(2β) | KL at peak = d*² | R at peak = d*(α − βd*) |
|---|---|---|---|---|---|
| smaller | 0.50 | 0.12 | 2.08 | 4.34 nats | 0.52 |
| larger | 0.65 | 0.09 | 3.61 | 13.04 nats | 1.17 |

With the RL form and α_RL = 0.5 held fixed, lowering β_RL from 0.15 to 0.12 moves `d*_RL = exp(α/β − 1)` from 10.3 to 23.7. The RL peak is exponentially sensitive to α/β, which is one reason RL peaks are harder to predict from a short run than best-of-n peaks.

[figures/rm-overopt.html](figures/rm-overopt.html) lets the reader set α and β for either functional form, see where the peak falls, and convert a best-of-n sample count to `d`; its second panel shows the BT loss and gradient used in §1 and in the negatives section.

### §2.4 Evidence

- **Proxy RM size (Result, single study).** Coefficients change smoothly with proxy RM size; larger RMs have smaller β, which the authors interpret as improved robustness (§3.2, §4.2.2, Fig. 3).
- **RM data (Result, single study).** In the data sweep (RM size held at 12M), the authors report that for all RM sizes, data amounts below about 2,000 comparisons give little improvement over near-chance loss (§3.3, Fig. 6). Holding the number of SGD steps fixed, four epochs instead of one gave no change in gold score, while one epoch of four times as much data performed better; the footnote does not state the two data amounts (§3.3, footnote 7).
- **Policy size (Result, single study).** A 6B policy gains less from optimization than a 1.2B policy but peaks at almost the same KL and shows almost the same proxy–gold gap (§3.4, Fig. 7). The paper does not report that larger policies exploit the proxy more.
- **KL penalty (Result, single study; authors call it hyperparameter-sensitive).** A nonzero KL penalty behaved like early stopping and did not raise the gold-score-versus-KL frontier (§3.6, Fig. 9).
- **RL versus best-of-n.** The authors report that RL consumes far more KL than best-of-n for both optimization and over-optimization, without giving a ratio, and conclude that KL distance cannot compare the amount of optimization across the two methods (§3.5).
- **Test-time search against a verifier (Result, single study).** With a 6B GSM8K verifier, solve rate improves as the number of ranked completions grows up to 400 and then decreases; the authors attribute the decrease to search finding "adversarial solutions that fool the verifier" ([[training-verifiers-to-solve-math-word-problems]] §5.1, Fig. 7a).

### §2.5 Interpretation of the two terms

Gao et al. interpret α as regressional Goodhart (the proxy equals the true score plus independent noise, so selection partly selects noise) and β as extremal Goodhart (optimized samples leave the RM's training distribution, where the proxy–gold relation weakens) (§4.2.1-4.2.2, Interpretation). Under their simplifying assumptions, splitting optimization into k rounds with retrained RMs adds `β_RL · d · log k` to the gold score and leaves the α term unchanged (§4.3, Interpretation).

### §2.6 Conditions and limits

Only the InstructGPT instruction environment was tested. The gold RM is itself a model, so the setup does not capture the gap between labels and human intent, and adversarial Goodhart is not modelled (§4.2.4, §4.5). No satisfactory fit was found for the proxy score (§3.1).

**Implication for a general-purpose model.** The peak is a property of the proxy RM, its data, and the optimizer. A general-purpose policy trained on many domains reaches the peak of different domains at different `d`, so a single stop point chosen on an aggregate evaluator can be past the peak in the domains where the RM is weakest.

---

## §3 Methods that make the reward signal more robust

### §3.1 Prediction ensembles with conservative aggregation

**Definition.** Train k RMs and combine their scores per response. Coste et al. study three combinations ([[reward-ensembling]] §3, arXiv:2310.02743v2):

```
R_mean(q,a) = (1/k) Σ_i R_i(q,a)                                                   (Eq. 3)
R_WCO(q,a)  = min_i R_i(q,a)                                                        (Eq. 4)
R_UWO(q,a)  = (1/k) Σ_i R_i(q,a) − λ · (1/k) Σ_i ( R_i(q,a) − (1/k) Σ_i R_i(q,a) )²     (Eq. 5)
```

`q` is the prompt, `a` the response, `R_i` the i-th RM, `λ` the weight on the intra-ensemble variance. WCO is worst-case optimization; UWO is uncertainty-weighted optimization. The authors note that mean optimization is not conservative: one member that overestimates can still be exploited (§3).

**Worked example (λ = 0.5).** Scores (1.0, 1.2, 2.6): mean 1.60, variance 0.507, UWO `1.60 − 0.25 = 1.35`, WCO 1.0. Scores (1.0, 1.1, 1.2): mean 1.10, variance 0.0067, UWO 1.097, WCO 1.0. The response on which one member is far higher loses 0.25 under UWO; the response on which members agree loses 0.003.

**Setup.** Pythia 1.4B policy; proxy RMs of 7M, 44M, and 1.3B; AlpacaFarm 7B human-preference RM as gold; RM training set fixed at 46k samples (preference pairs built from AlpacaFarm instructions, two responses each); RMs differ only in random seed (head initialization and data order); optional 25% label noise; ensembles of five (§4.2-4.3).

**Evidence (Result, single study).** For best-of-n, ensembles improve final gold performance by up to about 30% without label noise and up to about 75% with 25% label noise over the average single RM, and WCO and UWO show no over-optimization up to n = 12,500 (≈ 8.4 nats), while mean optimization does over-optimize with noisy labels (§5, §5.1, Fig. 3). For PPO, WCO and UWO reduce but do not eliminate over-optimization without a KL penalty; with a KL penalty of 0.01 they prevent it without a notable performance loss, while a single RM needs a penalty of 0.2 and loses performance (§5.2, Figs. 4-6). Performance is similar for four and five members, with a gap from three to four (§5.4, Fig. 11).

**Limits.** One environment, offline RLHF without RM refresh, and synthetic gold labels (§6).

### §3.2 Shared errors across ensemble members

Eisenstein et al. train T5 RMs from five pretraining seeds × five fine-tuning seeds (25 RMs per task and scale) on TL;DR, Anthropic Helpfulness, and XSum/NLI ([[helping-or-herding]] §2.3). They report (Abstract; Result, single study):

1. RMs with similar in-distribution accuracy give different rewards on aligned-policy outputs (underspecification).
2. Aligning to one RM does not improve reward as measured by another RM trained on the same data.
3. Ensembles whose members differ in pretraining seed generalize better than ensembles that differ only in fine-tuning seed, and both beat single RMs.
4. Even pretrain ensembles do not eliminate reward hacking when all members share an error pattern: summarization policies became too short when tuned for factuality and too verbose when tuned for summary quality, and assistant policies overused formulaic answer formats when tuned for helpfulness (§1).

Coste et al.'s ensembles differ only in fine-tuning seed ([[reward-ensembling]] §4.3), so their results correspond to the weaker ensemble type in Eisenstein et al.

### §3.3 Weight-averaged reward models (WARM)

**Definition.** Fine-tune M RMs from the same SFT checkpoint with different hyperparameters and data orders, then average their weights: `φ_WARM = (1/M) Σ_i φ_i` ([[warm-weight-averaged-reward-models]] §3.1). Inference costs one RM.

**Mechanism.** Weights fine-tuned from a shared pretraining stay linearly mode connected, so the averaged model is at least as accurate on OOD pairs as the interpolation of the members' accuracies (Observation 1, Eq. 2). In the paper's bag-of-features model, prediction ensembling weights feature j by `p_j` and weight averaging by `p_j²`, where `p_j` is the probability that one run learns the feature (§4.3, Eqs. 4-5). A feature learned in 90% of runs keeps weight 0.81; a feature learned in 10% keeps 0.01 (derived in the card). Run-specific features, which include memorized label noise, lose relative weight.

**Evidence (Result, single study).** On TL;DR with PaLM-XXS RMs and AI labels, a policy trained with WARM (M = 6) has a 79.4% oracle win rate against a policy trained with the best single RM (§5.2, Fig. 9(c)). With 25% swapped labels, weight averaging fits fewer corrupted training pairs than prediction ensembling and is better on OOD test pairs (Observation 3, Figs. 4-5).

**Limits.** If every member relies on the same spurious cue, such as length, WARM keeps it (§6). One task and AI labels only (§5). Weight averaging is not an uncertainty estimate; prediction ensembles remain the option when disagreement is needed as a signal (§6).

### §3.4 Pairwise rankers (PairRM)

**Definition.** A pairwise ranker encodes the prompt and both candidates in one sequence and predicts which is better, instead of scoring each response alone ([[pairrm]] §3.1-3.2).

**Mechanism.** The input is `<s><source> x </s> <candidate1> y_i </s> <candidate2> y_j </s>`; MLP heads on the special-token embeddings give `s^i` and `s^j`, and the pair score is `s_ij = s^i − s^j` (§3.3, App. A). Candidate order within a pair is shuffled during training. Aggregating all pairs (MaxLogits) needs O(N²) comparisons; one bubble-sort pass needs N − 1 (§3.3).

**Evidence.** On MixInstruct (N = 11 candidates), PairRanker's selections have average GPT-Rank 3.20 against 3.90 for the best single LLM and 3.66 for the pointwise SummaReranker (Table 2). The released 0.4B PairRM checkpoint scores 59.05 on Auto-J pairwise against 59.85 for UltraRM-13B, and 84.62 on HHH-Alignment against 83.71 (model card).

**Limits.** A pairwise ranker does not give a pointwise reward, so it cannot be used directly as a PPO reward; it fits reranking and pair construction.

### §3.5 Generative reward models and direct LLM scoring

**GenRM (Mahan et al.).** An LLM is trained to output the preferred-answer token, optionally after a rationale ([[generative-reward-models]] §4). In STaR-DPO, the chosen output is a rationale that reaches the correct verdict and the rejected output is a rationale that reaches the wrong verdict (Eq. 9). With Llama-3.1-8B-Instruct trained on UltraFeedback, STaR-DPO reaches 73.9% in-distribution (BT, PairRM, and GenRM "around 73–74%") and 81.9% on RewardBench, against 78.9% for GenRM without reasoning (§5.1). When trained on UltraInteract, explicit RMs reach about 94% in-distribution, but the BT RM falls below random on RewardBench Reasoning while STaR-DPO reaches 87.2% (§5.2). The paper does not use the judge as an RL reward (§7).

**d-RLAIF (Lee et al.).** An off-the-shelf LLM is prompted to rate a response from 1 to 10; the likelihood of each score token is normalized to a distribution, the reward is the expected score `s(y|x) = Σ_{i=1}^{10} i · P(i | y, x)`, rescaled to [−1, 1], and used directly in RL with no trained RM ([[rlaif-scaling]] §2.2.2). On summarization, human raters preferred d-RLAIF over SFT 74% of the time, against 68% for same-size RLAIF with a distilled RM, and preferred d-RLAIF over same-size RLAIF 60% of the time (Table 1). The authors estimate AI labeling at over 10× cheaper than human annotation (§4.1, App. L). Because the scorer is not trained on a fixed set of responses, d-RLAIF avoids RM staleness, but its errors become the errors of a prompted judge (ch-42, ch-49).

### §3.6 Multi-attribute regression reward models

Nemotron-4-340B-Reward replaces the final softmax layer of Nemotron-4-340B-Base with a linear projection to five HelpSteer attributes (Helpfulness, Correctness, Coherence, Complexity, Verbosity), trained on 10K HelpSteer2 examples and combined by a weighted sum at inference ([[nemotron-4-synthetic]] §3.1). The report states that multi-attribute regression separates helpfulness from artifacts such as length better than pairwise ranking models, without an ablation (Interpretation). The attribute weights, learning rate, epochs, and loss are not given in the report ([[nemotron-4-synthetic-recipe]], §3.1). Used as a judge for preference pairs, it reached 0.87 Chat-Hard accuracy against 0.54 for LLM-as-judge (§3.2.3).

RM-Bench tests whether the correctness and verbosity heads are separated: chosen and rejected responses separate on correctness in the safety domain, but overlap in math and code ([[rm-bench]] §4.3, Fig. 3).

**Implication for a general-purpose model.** Each method in §3 reduces one error source: ensembles and WARM reduce seed-specific and label-noise errors; pairwise and generative RMs reduce errors from scoring responses in isolation; multi-attribute heads expose trade-offs. None of them removes errors shared by all members, which come from the shared pretraining and the shared preference data.

---

## §4 Evaluating reward models out of distribution

### §4.1 RewardBench

**Definition.** A benchmark of prompt–chosen–rejected trios on which an RM is correct when it scores the chosen response higher ([[rewardbench]] §4.2). Sections and sizes: Chat 358, Chat Hard 456 (MT-Bench close ratings and LLMBar adversarial pairs), Safety 740, Reasoning 1,431 (PRM800K human versus buggy LLM answers; HumanEvalPack correct versus buggy code in six languages), and Prior Sets 17.2k (Anthropic Helpful, HHH, SHP, Summarize) weighted at 0.5 in the final score (Table 1; §4.2 footnote 4).

**Limits.** Many chosen and rejected responses come from different models, and prior-set test data have accuracy ceilings of 60–70% from inter-annotator disagreement (§1). RM-Bench and PPE both argue that pairs from models of different strength let an RM succeed on style and source cues ([[rm-bench]] §1; [[ppe-reward-model-eval]] §3).

### §4.2 RM-Bench: subtle errors and style control

**Mechanism.**
1. Generate chosen and rejected responses with the same model (gpt-4o), injecting factual errors into chat answers with a many-shot jailbreak prompt and selecting correct and incorrect samples for code and math by unit tests and answers ([[rm-bench]] §3.1-3.2).
2. Rewrite every response in three styles: concise `y∅`, detailed plain text `y^L`, detailed with Markdown `y^{L,M}` (§3.4).
3. Score all 3 × 3 chosen-style × rejected-style combinations. Easy accuracy is the mean of the lower triangle (chosen has more style), Normal the diagonal, Hard the upper triangle (rejected has more style) (§3.5).

**Worked example.** The Style-Substance matrix printed for FsfairX-LLaMA3-RM-v0.1 on chat (Fig. 2) has rows for chosen style (∅, L, L+M) and columns for rejected style:

```
            y_r∅      y_rL     y_rL,M
y_c∅       83.61      3.83      2.19
y_cL       99.45     66.12     49.73
y_cL,M    100.00     80.33     66.67
```

Hard = (3.83 + 2.19 + 49.73)/3 = 18.58%. Normal = (83.61 + 66.12 + 66.67)/3 = 72.13%. Easy = (99.45 + 100.00 + 80.33)/3 = 93.26%. When the incorrect answer is longer and formatted, this RM prefers it in more than 80% of chat pairs.

**Evidence.** The top RM (Skywork-Reward-Llama-3.1-8B) has 70.1% average and 46.6% hard accuracy; Nemotron-340B-Reward has 69.5% average and 56.1% hard (Table 3). Using four Tülu-v2.5 RMs and their PPO policies, RM-Bench scores correlate with normalized downstream policy performance at Pearson r = 0.55 (p = 0.07), against r = 0.21 (p = 0.51) for RewardBench (§5.2, App. F). Four RMs × three task families is a small sample (Open question on stability).

### §4.3 Preference Proxy Evaluations (PPE): accuracy versus the trained policy

**Mechanism** ([[ppe-reward-model-eval]] §4-6).
1. Human-preference set: 16,038 Chatbot Arena pairs from 20 models, over 121 languages, 6,120 voters (§4).
2. Correctness set: benchmark prompts (MMLU-Pro, MATH, GPQA, MBPP-Plus, IFEval), 32 samples per prompt from each of four models, labelled by the benchmark's verifier; prompts with fewer than 10% or more than 90% correct samples are removed (§5.1).
3. Downstream test: nine RMs each label 8,000 prompts × 16 samples from Llama-3.1-8B-Instruct (chosen = highest RM score, rejected = a uniformly sampled rank); each dataset trains Llama-3.1-8B-Instruct with DPO; the nine policies are ranked by 12,190 blind Arena votes (§6.1-6.2).

**Worked comparison.** Post-DPO Arena scores (Table 3) with RewardBench scores reported elsewhere:

| RM used to build DPO pairs | RewardBench overall | Post-DPO Arena score (95% CI) |
|---|---|---|
| Athene-RM-8B | not reported in the sources cited here | 1209 (1199–1219) |
| Skywork-Reward-Gemma-2-27B | 94.1 ([[deepseek-grm]] Table 2) | 1173 (1163–1182) |
| Nemotron-4-340B-Reward | 92.0 ([[nemotron-4-synthetic]] Table 4) | 1172 (1163–1180) |
| none (Llama-3.1-8B-Instruct base) | — | 1178 (1168–1187) |

The two RMs with RewardBench scores above 92 produced policies whose intervals overlap the untrained base model.

**Evidence (Result, single study).** Pairwise accuracy on the human-preference set is the best single predictor of post-DPO Arena score; ranking-correlation metrics (Spearman, Kendall) have nearly zero correlation (§7). Aggregating category scores at a low quantile (closer to the minimum) raises correlation, with accuracy peaking at 0.80 (Fig. 5); the authors read this as "any domain weakness in a reward model can be exploited by the LLM during training" (§7, Interpretation). Among top models, RewardBench score correlates negatively with the downstream result (§2.2, Fig. 4).

**Limits.** Nine RMs, one base model, offline DPO rather than online RL (§8.2). Over-optimization under PPO may change which metrics matter.

### §4.4 Generalist generative reward models (DeepSeek-GRM)

**Definition.** A pointwise generative RM that, given a query and one or more responses, writes principles, then critiques, then a score per response ([[deepseek-grm]] §3).

**Mechanism: Self-Principled Critique Tuning (SPCT).**
1. Rejective fine-tuning (cold start): sample principle-and-critique trajectories N_RFT = 3 times per RM example with DeepSeek-V2.5; reject trajectories whose scores do not rank the ground-truth best response highest, and reject examples where all three trajectories are correct as too easy (§3.2 Eq. 10; App. C.1). Optionally append "The best response is: Response j" to the prompt (hinted sampling).
2. Rule-based online RL with GRPO: reward `+1` if the extracted scores rank the ground-truth best response strictly highest (or match the single-response label), `−1` otherwise; no format reward, a larger KL coefficient instead (§3.2 Eq. 11).
3. Inference-time scaling: sample k trajectories and sum the pointwise scores; optionally a meta RM scores each trajectory and only the top k_meta vote (§4, Eq. 14).

**Evidence (Result, single study).** Gemma-2-27B-based models, same training data for all baselines (§5.1, App. C.2). Overall score across RewardBench, PPE Preference, PPE Correctness, and RMB (Table 2):

| Model | RewardBench | PPE Pref. | PPE Correct. | RMB | Overall |
|---|---|---|---|---|---|
| DeepSeek-BTRM-27B (scalar BT) | 81.7 | 68.3 | 66.7 | 57.9 | 68.6 |
| DeepSeek-PairRM-27B | 87.1 | 65.8 | 64.8 | 58.2 | 69.0 |
| DeepSeek-GRM-27B (greedy) | 86.0 | 64.7 | 59.8 | 69.0 | 69.9 |
| DeepSeek-GRM-27B, meta-RM voting @32 | 90.4 | 67.2 | 63.2 | 70.3 | 72.8 |

The scalar BT RM is the strongest on PPE Correctness (verifiable tasks) and the weakest on RMB; the authors describe scalar RMs as showing domain biases (§5.2). Without rejective-sampling data in the cold start, the model scores 66.1 before RL and 68.7 after RL, against 69.9 for the full method; removing general instruction data lowers the RFT model from 68.8 to 63.3 (Table 4, §5.2). DeepSeek-R1 on a 300-sample RewardBench subset scored below the 236B RFT model (§5.2).

**Limits.** The GRM is evaluated as a judge on benchmarks, not as the reward inside policy RL; the authors list that integration as future work (§7). Rubric and checklist rewards used during multi-domain RL are covered in ch-44b ([[rubrics-as-rewards]]).

**Implication for a general-purpose model.** An RM benchmark score is an average over pairs. The PPE low-quantile result and the RM-Bench math and code results both point to the minimum over domains as the quantity to monitor for a generalist policy.

---

## §5 Policy coverage: when an imperfect reward model still transfers

### §5.1 Definition and problem

Policy coverage measures how much probability a data-collection policy puts on the responses an optimal policy would produce. Huang et al. define the coverage coefficient ([[policy-coverage-loss]] Def. 2.2):

```
Cov_{π̃|π} = E_{s~ρ, a~π̃} [ π̃(a|s) / π(a|s) ]
```

`ρ` is the prompt distribution, `π̃` the target policy, `π` the policy that collects data, `a` a response. A large coefficient means `π` rarely samples responses that `π̃` needs.

### §5.2 Worked example

Two responses to one prompt. The target puts `π̃ = (0.5, 0.5)`.
- Collector `π = (0.9, 0.1)`: `Cov = 0.5 · (0.5/0.9) + 0.5 · (0.5/0.1) = 0.278 + 2.5 = 2.78`.
- Collector `π = (0.99, 0.01)`: `Cov = 0.5 · 0.505 + 0.5 · 50 = 25.3`.

A 9-point change in probability on the second response increases the coefficient about nine times. An RM that is accurate on average still cannot improve a policy on a response the policy almost never samples.

### §5.3 Result and conditions

Under the KL-regularized objective `J_β(π; r) = E[r] − β KL(π ‖ π_ref)`, a policy's coverage of the optimal policy is bounded by `1 + κ(e^{2R/β}) · value gap / β` for policies induced by rewards in [0, R] (Lemma 3.1). With `β = 0` a near-optimal deterministic policy can have infinite coverage coefficient, so the bound depends on regularization (§3.1). The authors therefore select, per iteration, among policies induced by several imperfect RMs by estimated win rate against the current policy, and keep the current policy as a candidate (§5, Alg. 3). On XSum with a T5-small (80M) policy and a Llama-3-8B-based simulated human reward, this reaches 54.0 ± 1.2% win rate against iterative DPO without transfer after three iterations, and 49.1–50.6% against always using the best source (Table 1; Result, single study). In iteration 3 the selection returns to ordinary online learning because no source wins (§6, Fig. 2).

Related evidence on coverage from verifier training: Cobbe et al. generate verifier training samples from a generator trained for only 2 epochs, because test@100 performance "degrades much more sharply than test@1" with more epochs and the diversity of solutions collapses ([[training-verifiers-to-solve-math-word-problems]] §4.1-4.2, Fig. 3).

**Limits.** One summarization task, T5 policies, simulated preferences, no held-out-task evaluation ([[policy-coverage-loss]] §7).

**Implication for a general-purpose model.** Reward-model accuracy and policy coverage are separate conditions. An RM transfers to a policy only on responses that policy samples with non-negligible probability. This connects the RM to the diversity controls of ch-43 and to the SFT-versus-RL coverage discussion of ch-38a.

---

## Negative samples and negative feedback

This section applies the course definitions of "negative" (ch-43a holds the derivations) to reward models and verifiers.

**1. Where negatives come from and how they are labelled.**
- Human pairwise labels: the dispreferred response `y_l`. Labels are noisy; Coste et al. simulate 25% flipped labels to match reported agreement rates ([[reward-ensembling]] §4.3).
- Outcome checks on the policy's own samples: a solution is negative if its final answer is wrong. Cobbe et al. note that some solutions reach the correct answer with flawed reasoning, which gives false positives in the positive class ([[training-verifiers-to-solve-math-word-problems]] §4.2). In PRM800K, 14.2% of labelled solutions reach a correct answer and 73.1% of labelled steps are correct ([[prm800k]] App. B Table 3), so an outcome label on a wrong solution does not say which step is wrong.
- RM-labelled pools: West-of-N pairs the highest- and lowest-scored of N policy samples under a base preference model ([[west-of-n]] §4, arXiv:2401.12086v2).
- Model-generated rationales judged by the preference label: in GenRM, a rationale reaching the wrong verdict ([[generative-reward-models]] §4).

**2. What current practice does with them.**
- BT RM training uses negatives as gradient on the RM: the loss lowers `r(y_l)` relative to `r(y_w)`, and only the difference is constrained (§1.3).
- V-STaR uses them as gradient on a language model: a verifier is trained with DPO on (correct, incorrect) pairs from the Cartesian product of the generator's solutions, which lowers the verifier's likelihood of incorrect solutions ([[v-star]] §3.1 Eq. 2).
- STaR-style self-training and STaR-SFT discard them.
- PRM800K uses them to select what to label: the current PRM's highest-scoring wrong-answer solutions go to human labelers ([[prm800k]] App. B).

**3. Mechanism.**
For the RM, the gradient on the rejected score is `∂L/∂r(y_l) = σ(r(y_l) − r(y_w))`. An easy negative with `Δ = 3` gives `σ(−3) = 0.047`; a near-miss negative with `Δ = 0.1` gives `σ(−0.1) = 0.475`, ten times larger. Pairs the RM already separates contribute almost no gradient; near-miss pairs carry most of the update and most of the label risk.

For a verifier or policy trained on token likelihoods, the softmax gradient is `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. With logits giving `p = (0.665, 0.245, 0.090)`, one gradient step of size η that decreases `log p_3` (an unlikelihood step on the unlikely token 3) changes the logits by `Δz_3 = −0.910η`, `Δz_1 = +0.665η`, `Δz_2 = +0.245η`. Most of the removed probability goes to the most likely alternative, token 1. When token 1 is itself wrong, a push-down on an already-unlikely negative concentrates mass on that wrong answer (ch-43a).

**4. Evidence with numbers.**
- *Near-miss negatives are where RMs fail.* Llama 2 helpfulness RM accuracy is 79.1% on "significantly better" pairs and 54.5% on "negligibly better" pairs ([[bradley-terry-rm]], Table 28). RM-Bench hard accuracy, where the rejected answer contains a subtle error but has more style, is 46.6% for its top RM (§4.2).
- *Easy negatives are label-accurate but less on-policy.* In West-of-N, pseudo-preference accuracy increases with N and exceeds high-confidence human data; N = 2 (label all pairs) harms the RM. The likelihood of the best and worst samples under the policy decreases as N grows, which the authors note can make responses out of distribution (§5.2, Fig. 4). One West-of-N round raised RM accuracy by up to about 2.3%, against about 1% from doubling human data (§5.1).
- *Negatives from the current policy.* Cobbe et al. train the verifier on 100 samples per training problem from the current generator ([[training-verifiers-to-solve-math-word-problems]] §4.2). V-STaR collects verifier data iteratively from better generators instead of a fixed generator and reports 4% to 17% test accuracy improvement over prior self-improvement and verification methods on GSM8K and MBPP with Llama 2 7B and 13B ([[v-star]] Abstract, §3).
- *Mining convincing wrong answers.* Active learning that labels convincing wrong-answer solutions is estimated at about 2.6× more data efficient than uniform labelling ([[prm800k]] §4.2, Fig. 4a).
- *Keeping versus discarding negatives in a judge.* STaR-SFT discards wrong-verdict rationales and reaches 67.4% on UltraFeedback; STaR-DPO uses them as rejected outputs and reaches 73.9%, and 81.9% on RewardBench ([[generative-reward-models]] §5.1). The loss also changes between the two, so this is not a clean measurement of the negatives' share.

**5. Controls.**
- Keep negatives on-policy: regenerate verifier and RM training samples from the current policy each round (V-STaR, West-of-N).
- Filter or weight uncertain labels: West-of-N retains only pairs with high base-model confidence `P_θ(y+ ≻ y− | x)`, which gave further gains (§5.2, Fig. 5). PPE drops prompts with fewer than 10% or more than 90% correct samples ([[ppe-reward-model-eval]] §5.1).
- Mask rather than penalize when the label source cannot distinguish a wrong answer from an unverifiable one (for example, a correct answer in an unexpected format). The hardest mined negatives are the ones most likely to be mislabelled positives (Interpretation; no source in this chapter measures the false-negative rate of mined negatives).
- Localize: label up to the first incorrect step instead of the whole solution ([[prm800k]] §2.6; ch-44).
- Keep a fixed regression set of known negatives and known positives (the "negative anchor" set from the learner's ch-20/21 extension) and report the verifier's false-accept and false-reject rates on it after each refresh.

**6. Diagnostics.** Log RM scores of chosen and rejected responses separately; report accuracy by preference strength and by style condition (Easy/Normal/Hard); for verifiers, report false-accept rate on known-wrong solutions and false-reject rate on known-correct ones; plot best-of-k selected correctness against k and watch for a decline at large k ([[training-verifiers-to-solve-math-word-problems]] Fig. 7a; [[ppe-reward-model-eval]] §5.2.1).

**7. Effect on generality.** An RM that fails on near-miss and style-controlled negatives rewards confident, well-formatted wrong answers, which raises hallucination under RL. An RM trained on negatives from one domain's policy samples does not cover negatives in other domains. No source cited here measures the share of RM improvement attributable to negatives separately from positives.

---

## Recipe

All rows are reward-model or verifier training settings as printed. Card rows were verified 2026-09-14; rows marked 2026-09-15 were checked in the primary text for this revision.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| InstructGPT RM | 6B | reward-model | initialization; head | §3.5: SFT model with unembedding removed, scalar output; App. C.2: final RM initialized from GPT-3 6B fine-tuned on public NLP datasets | arXiv:2203.02155v1 §3.5, App. C.2 [[rlhf-instructgpt]] | conflict (§3.5 vs App. C.2; App. C.2 describes the final RM) | App. C.2: similar results from GPT-3 or SFT init; no table |
| InstructGPT RM | 6B | reward-model | training prompts; responses ranked per prompt | 6,623 labeler + 26,584 customer = 33,207 prompts; K = 4 to 9; ties dropped | Table 6; §3.5; App. C.2 | derived (sum of Table 6 rows) | — |
| InstructGPT RM | 6B | reward-model | LR; schedule; batch; epochs | 9e-6; cosine to 10% of initial; 64 prompts (≤ 2,304 comparisons); 1 epoch | App. C.2 | verified 2026-09-15 | App. C.2: ±50% LR similar; multiple epochs overfit; no table |
| Llama 2 Helpfulness RM, Safety RM | 70B / smaller | reward-model | max LR; schedule; epochs; batch | 5e-6 (70B), 1e-5 (others); cosine to 10%, warmup 3%; 1 epoch; 512 pairs | arXiv:2307.09288v2 §3.2.2 [[llama-2-recipe]] | verified 2026-09-14 | "training longer can lead to over-fitting"; no numbers |
| Llama 2 Helpfulness RM | per size | reward-model | margin m(r) | small {1, 2/3, 1/3, 0} or large {3, 2, 1, 0}; variant used in final RMs not stated | v2 Eq. 2, Table 27 | verified 2026-09-14 | Table 28: 62.5 / 63.0 / 62.9 avg accuracy |
| Llama 3.1 405B RM | 405B | reward-model | objective; ranking data | Llama 2 loss without margin; edited > chosen > rejected | arXiv:2407.21783 §4.1.2 [[llama-3-recipe]] | verified 2026-09-14 | margin removed for diminishing improvements; no numbers |
| Nemotron-4-340B-Reward | 340B | reward-model | data; head | HelpSteer2 10K; linear projection to 5 attributes, weighted sum | arXiv:2406.11704v2 §3.1 [[nemotron-4-synthetic-recipe]] | verified 2026-09-14 | RewardBench 92.0 (Table 4) |
| Nemotron-4-340B-Reward | 340B | reward-model | LR, epochs, loss, attribute weights | not reported | §3.1 checked | not reported | — |
| Gao proxy RMs | 3M–3B | reward-model | comparisons; batch | 100,000 (10% held out); 64 | arXiv:2210.10760v1 §2.1, App. C Table 1 [[reward-model-overoptimization]] | verified 2026-09-14 | Fig. 4: more data, higher gold score |
| Coste proxy RMs (Pythia) | 7M / 44M / 1.3B | reward-model | LR; epochs; batch; RM training samples | 1e-5; 5; 32; 46k samples | arXiv:2310.02743v2 §4.3, App. D.1 Table 2 [[reward-ensembling]] | verified 2026-09-15 | no ablation reported |
| Coste ensembles | 44M members | reward-model | members; diversity | 5; random seed only (head init, data order) | §4.3 | verified 2026-09-15 | Fig. 11: 4 ≈ 5 members, gap from 3 |
| Coste PPO with WCO/UWO | Pythia 1.4B policy | RL | KL penalty | 0.01 (single RM needed 0.2) | §5.2, Figs. 4, 6 | verified 2026-09-15 | Fig. 7 across penalties |
| WARM RMs (PaLM-XXS) | not reported | reward-model | steps; batch; LR set; members | 10k; 128; {1e-5, 4e-5, 1e-4}; M = 6 reference | arXiv:2401.12187v1 App. B.3, §5.2 [[warm-weight-averaged-reward-models]] | verified 2026-09-14 | Fig. 9: M = 6 vs best single RM |
| GenRM STaR-DPO | 8B | reward-model | LR; DPO β; STaR iterations | 1.0e-6; 1.0; 3 | arXiv:2410.12832v1 App. A.2.1 Table 2, App. A.3 [[generative-reward-models]] | verified 2026-09-14 | Table 3 per iteration |
| DeepSeek-GRM-27B | 27B | reward-model (RFT) | data; LR; batch; steps | 1256K (1070K general instruction + 186K rejective-sampled); 5e-6; 1024; 900 | arXiv:2504.02495v3 App. C.1 [[deepseek-grm]] | verified 2026-09-15 | Table 4: w/o general instruction data 63.3 |
| DeepSeek-GRM-27B | 27B | reward-model (RL) | algorithm; data; LR; batch; steps; KL β; group size | GRPO with ±1 rule reward; 237K; 4e-7; 512; 900; 0.08; G = 4 | App. C.1 | verified 2026-09-15 | β grid {0.00, 0.01, 0.02, 0.08}: smaller β collapsed on some subsets |
| West-of-N self-trained RM | T5-XXL 11B | reward-model | N; data mix; sampling temperature | N = 64; 1:1 base and West-of-N pairs; 0.7 | arXiv:2401.12086v2 §5 "Methods" [[west-of-n]] | verified 2026-09-15 | Fig. 4a: gains grow with N |
| GSM8K verifier | 6B, 175B | reward-model | generator epochs; samples per problem; verifier epochs; aux loss | 2; 100; 1; joint LM objective | arXiv:2110.14168v2 §4.2 [[training-verifiers-to-solve-math-word-problems]] | verified 2026-09-15 | Fig. 3 (coverage); Fig. 6b (LM objective) |
| PPE downstream DPO | Llama-3.1-8B-Instruct | eval-gate | pairs; β (τ); batch; LR | 8,000 prompts × 16 samples; 0.1; 64; printed as "2.00 × 10−0.6" | arXiv:2410.14872v2 §6.1, App. A.3 [[ppe-reward-model-eval]] | verified 2026-09-15; LR conflict (printed exponent not usable) | Table 3 |

**Starting point for a small general-purpose run.** For a pairwise scalar RM initialized from a chat or SFT checkpoint at 6B–13B scale, the verified rows support one epoch over the preference data (InstructGPT 6B, Llama 2 all sizes, both reporting overfitting with more epochs), a peak learning rate of 1e-5 with cosine decay to 10% (Llama 2 sizes below 70B) or 9e-6 (InstructGPT 6B), and grouping all comparisons from one ranked prompt in one batch element when K > 2 (InstructGPT). These values come from runs with 33K ranked prompts (InstructGPT) and about 2.9M comparisons (Llama 2); they were not ablated for smaller data. When the RM will be optimized against with RL, the Coste rows support five fine-tuning seeds with a worst-case or uncertainty-weighted combination and a small KL penalty (0.01 in a Pythia 1.4B setup). Before using the RM, run a PPE-style downstream check or at least report accuracy per domain and the minimum across domains.

---

## Generalization lens

**(a) What increases breadth.**
- Diverse pretraining among RM ensemble members: pretrain-seed ensembles generalize better than fine-tune-seed ensembles ([[helping-or-herding]] Abstract).
- Weight averaging of RMs from one checkpoint, which favours features learned in most runs ([[warm-weight-averaged-reward-models]] §4.3, §5.2).
- General instruction data in generative-RM training: removing it lowered DeepSeek-GRM-27B-RFT from 68.8 to 63.3 overall ([[deepseek-grm]] Table 4).
- On-policy RM and verifier data: West-of-N pairs are orders of magnitude more likely under the policy than human-dataset pairs ([[west-of-n]] §5.2, Fig. 4c); V-STaR's iterative verifier data ([[v-star]] §3).
- Rationales sampled from the judge itself rather than from a stronger model: Llama 3.1 70B rationales lowered RewardBench accuracy in GenRM ([[generative-reward-models]] §5.3, Table 1).

**(b) What causes narrowing.**
- Optimizing past the gold-score peak ([[reward-model-overoptimization]] Fig. 1).
- Shared RM errors on style: summaries too short under a factuality RM, too verbose under a quality RM, formulaic formats under a helpfulness RM ([[helping-or-herding]] §1).
- Domain-biased scalar RMs: a BT RM below random on RewardBench Reasoning after UltraInteract training ([[generative-reward-models]] §5.2); scalar BT strongest on PPE Correctness and weakest on RMB ([[deepseek-grm]] §5.2).
- Low policy coverage: responses the policy rarely samples are not improved by any RM ([[policy-coverage-loss]] §3.1); over-trained generators lose test@100 ([[training-verifiers-to-solve-math-word-problems]] Fig. 3).

**(c) How to measure it at this stage.**
- Downstream-linked accuracy on a human-preference set plus correctness sets built from policy samples, aggregated at a low quantile across categories ([[ppe-reward-model-eval]] §7, Fig. 5).
- Style-controlled hard accuracy per domain ([[rm-bench]] §3.5).
- Gold or independent-judge score versus √KL during optimization, with the proxy plotted on the same axis ([[reward-model-overoptimization]] Fig. 8).
- Intra-ensemble variance during RL as a drift signal: under PPO with mean optimization it rose almost three times without label noise and about 2.5 times with 25% noise, against about 20% under UWO (λ = 0.1) with noise ([[reward-ensembling]] §5.5, Fig. 10).
- Known measurement errors: benchmark leakage ([[ppe-reward-model-eval]] §8.1); chosen and rejected responses from different models, which lets style decide ([[rm-bench]] §1); small numbers of RMs in correlation studies (4 in RM-Bench §5, 9 in PPE §6).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Selecting an RM by RewardBench overall score alone | Policy trained with the RM does not beat the base model on a blind comparison | PPE-style downstream test or per-domain minimum accuracy ([[ppe-reward-model-eval]] Table 3) |
| Comparing RM scores across prompts | Filtering or curriculum by raw score favours easy prompts | Normalize per prompt or against a reference response; BT scores are identified only up to `f(x)` (§1.4) |
| Reporting proxy score as training progress | Proxy keeps rising while an independent judge flattens or falls | Plot both against √KL on one axis ([[reward-model-overoptimization]] Fig. 8) |
| Min or median over RMs without fixing offsets | Aggregate is dominated by whichever RM chose the lowest `C(x)` | Offset regularizer or per-prompt centring before aggregation ([[helping-or-herding]] Eq. 2) |
| Treating seed ensembles as independent | Ensemble and members make the same length or format error | Pretrain-diverse members; inspect top-scored outputs of the final policy by style features |
| Training a verifier on a fixed early generator | Verifier accuracy is high offline but best-of-k gains shrink after policy updates | Regenerate verifier data from the current policy each round ([[v-star]] §3) |
| Mining only easy (best-versus-worst) negatives | RM accuracy rises on its own pairs but hard accuracy on subtle errors stays near random | Report RM-Bench-style Hard accuracy and accuracy by preference strength |
| Unbounded best-of-k against a verifier | Selected correctness decreases at large k | Best-of-k curve over k ([[training-verifiers-to-solve-math-word-problems]] Fig. 7a) |
| Using a K-way ranking as independent shuffled pairs | RM overfits within one pass | Group all C(K,2) comparisons of a prompt in one batch element ([[rlhf-instructgpt]] §3.5) |

---

## Check your understanding

1. The BT loss is unchanged when a constant is added to both scores of a pair. Explain why this makes the minimum over an ensemble of RMs undefined without an extra constraint, and how Eisenstein et al.'s regularizer removes the problem.
2. For three responses with scores (2, 1, 0), Plackett–Luce and averaged all-pairs BT give different losses (0.721 and 0.251). Which pair comparisons does each objective weight more, and what would change for a K = 9 ranking?
3. Gao et al. find that α_RL can be held constant while β_RL decreases with proxy RM size. Using `d*_RL = exp(α_RL/β_RL − 1)`, explain why a small error in estimating β_RL from a short run gives a large error in the predicted stop point.
4. Coste et al. find that mean optimization over-optimizes with 25% label noise while WCO and UWO do not. Explain the mechanism in terms of one ensemble member overestimating, and explain why this does not address the failures Eisenstein et al. report.
5. In PPE, two RMs with RewardBench scores above 92 produced policies no better than the base model. Give two causal explanations that are consistent with RM-Bench's style-controlled results and PPE's low-quantile result.
6. A verifier trained on a 2-epoch generator's samples works well, but after RL the policy produces a new class of wrong answers. Using the coverage coefficient and the V-STaR data loop, explain why the verifier's false-accept rate on the new class is not measured by its original validation set.
7. Explain why a near-miss negative (`Δ = 0.1`) produces ten times the RM gradient of an easy negative (`Δ = 3`), and why the same property raises the cost of a mislabelled near-miss negative.

---

## Connections

- Previous: **ch-15** — Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage. Supplies the pairwise labels, agreement rates, and prompt coverage that set the ceiling for §1.
- Next: **ch-37** — Policy-Gradient Foundations for Language Models. Uses the RM score as the return in policy-gradient estimators.
- **ch-38** — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax. The KL term whose interaction with over-optimization is measured in §2.4.
- **ch-38a** — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity. Coverage and on-policy data from §5.
- **ch-39** — Offline Preference Optimization: DPO and Its Variants. Replaces the explicit RM with the implicit reward `β log π/π_ref` derived from the BT and Plackett–Luce models in §1.
- **ch-42** — Reward Hacking and Judge Design. Failure modes of optimized policies and of LLM judges used as rewards.
- **ch-43a** — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Full derivations behind the negatives section.
- **ch-44** — Process Supervision and Verifiable Rewards. Step-level labels and verifiers.
- **ch-44b** — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. Rubric and checklist rewards used during RL.
- **ch-30c** — Weight Averaging and Model Merging for Generalist Models. Weight averaging applied to policies rather than RMs.
- **ch-49** — Judge Models: Bias, Calibration, and Judge-Specific Overfitting. Evaluation-side treatment of the generative judges in §3.5 and §4.4.

---

## Sources

- [[bradley-terry-rm]] — BT model and loss, identifiability, Stiennon RM scaling and transfer, Llama 2 accuracy by preference strength, IPO deterministic-label limit.
- [[dpo]] — Plackett–Luce model (App. A.3 Eq. 18) and its reduction to BT.
- [[rlhf-instructgpt]] — K-way all-pairs RM loss, batching per prompt, RM hyperparameters (App. C.2), dataset sizes (Table 6).
- [[reward-model-overoptimization]] — functional forms, coefficient scaling, data and policy-size results, KL penalty result, Goodhart interpretation.
- [[reward-ensembling]] — mean, WCO, UWO aggregation; best-of-n and PPO results with label noise; ensemble size; hyperparameters (cited at loci in arXiv:2310.02743v2).
- [[helping-or-herding]] — pretrain versus fine-tune ensembles, offset regularizer, reward hacks shared across members.
- [[warm-weight-averaged-reward-models]] — weight averaging, toy `p_j²` analysis, 79.4% win rate, limits.
- [[pairrm]] — joint pairwise encoding, aggregation cost, MixInstruct and model-card results.
- [[generative-reward-models]] — STaR-DPO judge, RewardBench and UltraInteract generalization results, negatives kept versus discarded.
- [[rlaif-scaling]] — d-RLAIF scoring mechanism, win rates, cost estimate (cited at loci in arXiv:2309.00267v3).
- [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]] — multi-attribute RM architecture, RewardBench 92.0, RM-as-judge result, unreported RM settings.
- [[rewardbench]] — benchmark construction and scoring.
- [[rm-bench]] — style-substance matrix, hard accuracy, correlation with policy performance.
- [[ppe-reward-model-eval]] — downstream DPO experiment, Arena scores, metric correlations, low-quantile aggregation.
- [[deepseek-grm]] — SPCT method, generalist RM results, ablations, training settings.
- [[policy-coverage-loss]] — coverage coefficient, Lemma 3.1, empirical TPO results.
- [[training-verifiers-to-solve-math-word-problems]] — verifier recipe, generator coverage, test-time decline at large k (cited at loci in arXiv:2110.14168v2).
- [[v-star]] — DPO verifier on the policy's correct and incorrect samples, iterative data (cited at loci in arXiv:2402.06457v2).
- [[prm800k]] — convincing wrong-answer active learning, label statistics, first-incorrect-step labels.
- [[west-of-n]] — best-versus-worst pseudo-preferences, N and filtering ablations (cited at loci in arXiv:2401.12086v2).
- [[llama-2-recipe]], [[llama-3-recipe]] — RM training rows.
- [[rubrics-as-rewards]] — pointer for rubric rewards covered in ch-44b.
