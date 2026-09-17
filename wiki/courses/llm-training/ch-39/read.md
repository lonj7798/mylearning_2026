<!-- chapter: ch-39
     track: preference
     kind: content
     title: Offline Preference Optimization: DPO and Its Variants
     deps: [ch-38a]
     sources: [[dpo]], [[ipo]], [[kto]], [[simpo]], [[orpo]], [[rpo]], [[likelihood-displacement]], [[learning-dynamics-llm-finetuning]], [[dpo-positive]], [[noise-contrastive-alignment]], [[d2o-negating-negatives]], [[unpacking-dpo-ppo]], [[preference-ranking-accuracy]], [[daa-overoptimization-scaling]], [[likelihood-overoptimisation-daa]], [[on-policy-suboptimal-preference-data]], [[length-controlled-alpacaeval]], [[openrlhf-dpo]], [[openrlhf-dpo-recipe]], [[open-instruct-allenai-recipes-recipe]], [[tulu-3]], [[llama-3-recipe]], [[on-off-policy-rlhf]], [[trl-online-dpo]], [[ultrafeedback]], [[hh-rlhf]], [[pairrm]], [[rlhf-instructgpt]], [[longpo]], [[eto-trial-and-error]]
     figures: figures/dpo-landscape.html, figures/negative-gradient-squeeze.html
     revised: 2026-09 (generality revision)
-->

# Chapter 39 — Offline Preference Optimization: DPO and Its Variants

> **Core insight.** For any reward function, the maximizer of the KL-regularized RLHF objective is
> `π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x)`. Inverting that identity expresses the reward as
> `β log π*/π_ref` plus a prompt-only term that cancels inside a Bradley-Terry difference, which turns
> preference learning into a binary classification loss over pairs with no reward model and no sampling
> from the policy ([[dpo]] §4, Eq. 4–7). The resulting loss depends on a pair only through the margin
> `h = log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)`: it constrains a difference, not the two
> likelihoods separately. Every variant in this chapter changes what else the objective constrains, and
> the family's measured failures — falling chosen likelihood, likelihood displacement, ranking accuracy
> below 60%, over-optimization inside one epoch — are consequences of that one property.
>
> **Guideline.** When a fixed preference set exists and the goal is chat quality, instruction following
> and truthfulness, use DPO with a reference model equal to the SFT checkpoint that produced or scored the
> data, one epoch, and early stopping on a held-out win rate, because in the [[unpacking-dpo-ppo]] 13B
> study preference training moved instruction following and truthfulness by more than 8 points while
> leaving factuality within about 1 point (§3.1, Table 1), and because in [[daa-overoptimization-scaling]]
> the best checkpoints on TL;DR appear after roughly 25% of one epoch under wide KL budgets (§3.1, Fig. 2).
> When the chosen responses are correct answers that must stay reachable by sampling — mathematics, code,
> tool calls — add a positive term on the chosen response (`α · NLL` as in [[rpo]], the ORPO SFT term, the
> DPOP penalty of [[dpo-positive]], or the NCA form of [[noise-contrastive-alignment]]), because in those
> settings plain DPO lowered sampled accuracy relative to the same data with the anchor (GSM8K 61.8% vs
> 73.1% after one iteration, [[rpo]] Table 1). When feedback arrives as single thumbs rather than pairs,
> use KTO ([[kto]]). When the pairs can be regenerated from the current policy, prefer that over a static
> set ([[on-policy-suboptimal-preference-data]] §5.3; [[on-off-policy-rlhf]] §2.1), and treat the offline
> objectives in this chapter as the loss you apply to those samples rather than as a replacement for RL.

---

## Why this chapter matters for a general-purpose model

The pipeline position is: pre-training → mid-training → SFT → **preference optimization** → RL →
evaluation. This is the first stage whose training signal is comparative rather than a target sequence,
and the first in which a term of the loss explicitly *decreases* the probability of text the model
produced or could produce. Two consequences matter for breadth of capability.

First, the measured gains are concentrated. In the controlled 13B study of [[unpacking-dpo-ppo]], DPO on
14 different preference sets moved instruction following by up to +8.6 points and truthfulness by up to
+12.7 points over the SFT model, while every dataset left MMLU inside a 1-point band (55.4 for the SFT
model, 54.7–55.7 after training, §3.1 Table 1). This is not the stage that teaches facts or reasoning; it reshapes how existing
ability is expressed, and it can remove ability when run too far.

Second, the negative term is not free. The rejected response enters the gradient as an explicit push-down
(negative as gradient, category 4 in the course's negative-feedback taxonomy). Pushing down a response
the model already finds unlikely moves most of the recovered probability mass to whatever token was
already most likely ([[learning-dynamics-llm-finetuning]] §3.3), which is how a run can raise its training
margin, lower its loss, and still lower the probability of the preferred response
([[likelihood-displacement]] §2.2) or its refusal rate on safety prompts (74.4% → 33.4% for
Llama-3-8B-Instruct, §6.2).

---

## §1 What the offline setting is

**Definition.** Offline preference optimization trains a policy `π_θ` on a fixed set of triples
`{(x, y_w, y_l)}` — prompt, preferred response, dispreferred response — or, for [[kto]], on unary examples
`{(x, y, b)}` with `b ∈ {desirable, undesirable}`. No response is sampled from `π_θ` during training.

**The problem it addresses.** The RLHF pipeline of [[rlhf-instructgpt]] requires a reward model plus
online sampling: each update needs fresh generations scored by a separate network. Offline preference
optimization removes both, at the cost of training on responses drawn from some other distribution.

**Where the pairs come from.** Human comparisons ([[hh-rlhf]]), model-scored multi-model completions
([[ultrafeedback]]), or responses regenerated from the starting checkpoint and ranked by a reward model —
the "Instruct" setup of [[simpo]] samples 5 responses per UltraFeedback prompt at temperature 0.8 from the
SFT model and takes the best and worst under PairRM ([[pairrm]]), which the authors describe as closer to
an on-policy setting (§3). **Interpretation.** In the comparisons collected in §9, where the pairs come
from changes the outcome more than which of the losses below is applied.

---

## §2 DPO

### 2.1 The objective and its closed-form optimum

The KL-regularized RLHF objective ([[dpo]] §3, Eq. 3):

```
max_π  E_{x ~ D, y ~ π(·|x)} [ r(x, y) ]  −  β · D_KL( π(·|x) ‖ π_ref(·|x) )            (1)
```

`x` prompt; `y` response; `r` reward; `π_ref` the reference policy, in practice the SFT checkpoint;
`β > 0` the strength of the KL constraint. For a fixed `x` this is an optimization over a distribution on
responses. Its maximizer is ([[dpo]] Eq. 4, App. A.1):

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x, y) / β ),   Z(x) = Σ_{y'} π_ref(y'|x) exp( r(x, y')/β )   (2)
```

`Z(x)` is the partition function: a sum over all responses, not computable in practice.

### 2.2 Inverting for the reward and cancelling `Z(x)`

Taking logarithms of (2) and solving for `r` ([[dpo]] Eq. 5):

```
r(x, y) = β · log[ π*(y|x) / π_ref(y|x) ]  +  β · log Z(x)                              (3)
```

Under the Bradley-Terry model `P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l))` (the same likelihood that
ch-41's reward models are fit under), the term `β log Z(x)` appears with both signs and cancels, because
both responses share the prompt. Substituting the difference into the Bradley-Terry negative
log-likelihood and treating `π*` as the trainable `π_θ` gives the DPO loss ([[dpo]] Eq. 7):

```
L_DPO(π_θ; π_ref) = − E_{(x,y_w,y_l) ~ D} [ log σ( β·log[π_θ(y_w|x)/π_ref(y_w|x)]
                                                 − β·log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]   (4)
```

`σ` is the logistic function; `π_θ(y|x)` is the product of token probabilities of the response, so
`log π_θ(y|x)` is the **sum** of token log-probabilities over the response (this is what the reference
implementations compute; see §10). Define the **implicit reward** ([[dpo]] §4):

```
r̂_θ(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ]                                            (5)
```

and the **margin** `h = log[π_θ(y_w|x)/π_ref(y_w|x)] − log[π_θ(y_l|x)/π_ref(y_l|x)]`, so that
`L = −log σ(β h)` per pair.

### 2.3 Gradient

[[dpo]] §4 (App. A.4):

```
∇_θ L_DPO = − β · E[ σ( r̂_θ(x,y_l) − r̂_θ(x,y_w) ) · ( ∇_θ log π_θ(y_w|x) − ∇_θ log π_θ(y_l|x) ) ]   (6)
```

The scalar weight `σ(r̂_l − r̂_w)` lies in (0, 1) and is larger when the implicit reward currently ranks
the dispreferred response above the preferred one. The paper labels the two vector terms "increase
likelihood of `y_w`" and "decrease likelihood of `y_l`".

**Worked example.** Take `β = 0.1`. Suppose for one pair the policy and reference assign summed
log-probabilities `log π_θ(y_w) = −40`, `log π_ref(y_w) = −42`, `log π_θ(y_l) = −55`,
`log π_ref(y_l) = −50`. Then `r̂_w = 0.1 × 2 = 0.2`, `r̂_l = 0.1 × (−5) = −0.5`, `h = 7`, `βh = 0.7`,
loss `−log σ(0.7) = 0.403`, and the gradient weight is `σ(−0.7) = 0.332`. At initialization
(`π_θ = π_ref`) the margin is 0, the loss is `log 2 = 0.693` and the weight is 0.5; a pair the model
currently ranks the wrong way has a weight above 0.5, approaching 1.

**The invariance that matters.** Now take a second pair with `log π_θ(y_w) = −45` (three nats *below* the
reference) and `log π_θ(y_l) = −60`. Again `h = 7`, the loss is again 0.403, and the gradient weight is
identical. The objective cannot distinguish "the chosen response became 7 nats more likely relative to
reference" from "the chosen response became 3 nats less likely and the rejected 10 nats less likely". §8
and the negative-feedback section show what the optimizer does with that freedom.

[figures/dpo-landscape.html](figures/dpo-landscape.html) plots per-pair loss and gradient weight against
the margin for DPO, IPO and SimPO with adjustable `β`, `τ` and `γ`, and marks the loss value a pair must
reach before its ranking flips under Theorem 4.1 of [[preference-ranking-accuracy]] (§8.2).

### 2.4 What `β` does

`β` multiplies the margin inside the sigmoid. A larger `β` makes the same log-ratio difference produce a
larger preference probability, so less movement of `π_θ` is needed to reduce the loss; a smaller `β`
permits more movement. In [[dpo]] the sweep was `β ∈ {0.05, 0.1, 1, 5}` on IMDb (§6.1, Fig. 2 left) and
the reported default is 0.1, with the statement "we did not meaningfully tune DPO's β" (§6.2). `β` is not
a KL penalty computed at training time: the run's realized KL is an outcome. Across DPO, IPO and SLiC,
[[daa-overoptimization-scaling]] reports "clear dependencies between the β parameter and the
corresponding KL achieved at the end of training" (§3.1), not a fixed budget.

---

## §3 IPO: bounded target instead of a logistic link

**The measured problem.** [[ipo]] observes that if a pair is deterministic in the data
(`p*(y_w ≻ y_l) = 1`), the Bradley-Terry fit requires the reward difference to diverge, and the optimal
policy then sets `π*(y_l) = 0` "irrespective of what constant τ is used for the KL-regularisation"
(§4, the analysis of DPO and RLHF that motivates a bounded `Ψ`). With finite data the empirical preference
probability is 0 or 1 for most pairs, so this is the normal case rather than an edge case.

**Mechanism.** `ΨPO` generalizes the objective to
`max_π E[Ψ(p*(y_w ≻ y_l|x))] − τ D_KL(π ‖ π_ref)` for non-decreasing `Ψ`. `Ψ(p) = log(p/(1−p))` recovers
DPO; `Ψ(p) = p` gives IPO, whose sampled loss is ([[ipo]] Eq. 17):

```
L_IPO = E_{(x,y_w,y_l) ~ D} [ ( h_π(y_w, y_l, x) − 1/(2τ) )^2 ]                          (7)
h_π(y, y', x) = log[ π_θ(y|x) π_ref(y'|x) / ( π_θ(y'|x) π_ref(y|x) ) ]
```

`τ > 0` is the regularization strength, and `h_π` is the same margin as in §2.2. The optimum is the
finite value `h* = 1/(2τ)`: overshooting the target is penalized like undershooting it.

**Worked example.** With `τ = 0.1` the target is `1/(2τ) = 5`. A pair at `h = 7` has loss `(7−5)² = 4`
and gradient `2(h − 5) = +4`, which *reduces* the margin. Under DPO the same pair has loss 0.403 and a
gradient that continues to increase the margin.

**Evidence and limits.** The IPO paper's experiments are three-action bandit problems with a uniform
reference policy trained for 18,000 Adam steps, not language models (§5.3–5.4). Language-model evidence
comes from later comparisons: in [[simpo]] Table 4, IPO reaches AlpacaEval 2 length-controlled 11.8 on
Mistral-Base and 35.6 on Llama-3-Instruct versus 15.1 and 40.3 for DPO; in
[[daa-overoptimization-scaling]] §3.1 IPO reaches lower KL under the same constraint and shows less
over-optimization than DPO and SLiC on TL;DR at 1B–6.9B. **Result (single study)** in each case, and the
two disagree about whether IPO is better, because they measure different things (win rate at a tuned
setting versus degradation across KL budgets).

---

## §4 KTO: unary labels and a batch reference point

**The problem.** Production feedback is usually one signal per response (accept, thumbs-down, escalation),
not a pair. Building pairs from it discards data.

**Mechanism** ([[kto]] §4.1, Eq. 8). With `r_θ(x, y) = log[π_θ(y|x)/π_ref(y|x)]` (no `β` inside):

```
z_0 = KL( π_θ(y'|x) ‖ π_ref(y'|x) )                                                      (8)
v(x, y) = λ_D · σ( β ( r_θ(x,y) − z_0 ) )   if y is desirable
        = λ_U · σ( β ( z_0 − r_θ(x,y) ) )   if y is undesirable
L_KTO = E_{(x,y) ~ D} [ λ_y − v(x, y) ]
```

`λ_D`, `λ_U` weight the desirable and undesirable sides; `β` scales the argument; `z_0` is the reference
point, estimated in a batch by pairing each prompt with another example's response,
`ẑ_0 = max(0, (1/m) Σ_i log[π_θ(y_j|x_i)/π_ref(y_j|x_i)])` with `j = (i mod m) + 1`, and no gradient flows
through it (§4.1, "KL Estimate"). The mismatched pairing is deliberate: the matched response is often a
deliberately good or bad output and would give unrepresentative values.

**Worked example.** `β = 0.1`, `z_0 = 0.5`, `λ_D = λ_U = 1`. A desirable response with `r_θ = 2` has
`v = σ(0.1 × 1.5) = 0.537`, loss `1 − 0.537 = 0.463`. An undesirable response with `r_θ = −3` has
`v = σ(0.1 × 3.5) = 0.587`, loss 0.413. Raising `r_θ` on the desirable side while `z_0` rises by the same
amount leaves the loss unchanged, which is the paper's stated intent: the reward may increase only if the
KL term stays flat (§4.1).

**Class imbalance.** The recommendation is `λ_D n_D / (λ_U n_U) ∈ [1, 3/2]`, with `n_D`, `n_U` the counts
of desirable and undesirable examples (§4.2, Eq. 9). The paper's worked case is a 1:10 ratio with
`λ_U = 1`, for which it sets `λ_D = 13.33`; discarding 90% of the desirable data for Llama-7B and
rebalancing this way still outperformed DPO in their evaluation (§4.3, Fig. 5).

**Evidence.** Aligning Zephyr-β-SFT on UltraFeedback for one epoch: GSM8K 39.0 (SFT) → 40.0 (DPO) → 53.5
(KTO); BBH 46.3 → 44.1 → 52.6; MMLU 57.2 → 58.2 → 58.6 (Table 2). Training on one response per prompt
(`one-y-per-x`) gives GSM8K 50.0 with half the data. The recommended learning rate is 5e-6 with AdamW,
"2x to 10x the optimal learning rate for DPO", and the paper states the model "is more sensitive to the
learning rate than any other" hyperparameter (§4.2). **Result (single study).**

---

## §5 SimPO and length-normalized DPO: changing the units of the reward

**The measured problem.** The DPO implicit reward is a sum over tokens, so a uniform per-token increase
in log-probability changes it in proportion to response length. [[simpo]] reports that only about 50% of
training triples satisfy `p_θ(y_w|x) > p_θ(y_l|x)` in average log-likelihood after DPO training, where
`p_θ` is the length-normalized likelihood used for ranking (§2.2, Fig. 4b): the quantity optimized during
training and the quantity that decides generation disagree on half the data.

**Mechanism.** Replace the log-ratio with the average token log-probability and drop the reference
([[simpo]] Eq. 4, Eq. 6):

```
r_SimPO(x, y) = (β / |y|) · Σ_{t=1..|y|} log π_θ(y_t | x, y_<t)                          (9)
L_SimPO = − E[ log σ( (β/|y_w|) log π_θ(y_w|x) − (β/|y_l|) log π_θ(y_l|x) − γ ) ]        (10)
```

`|y|` is the token count; `γ ≥ 0` is a target margin the difference must exceed before the loss
saturates. Because the reward is now per token, `β` takes larger values than in DPO: the paper's settings are
`β = 2.0, γ = 1.6` (Mistral-Base), `2.5 / 0.3` (Mistral-Instruct), `2.0 / 1.0` (Llama-3-Base),
`2.5 / 1.4` (Llama-3-Instruct), with batch 128 and one epoch (App. B, Table 8). The general guidance in the
same paper is `β ∈ [2.0, 2.5]` and `γ ∈ [0.5, 1.5]` (§3); the later Llama-3-Instruct v0.2 setting, whose
pairs are labeled by the ArmoRM reward model rather than PairRM, departs from it with `β = 10`, `γ = 3`
(App. H).

**Worked example.** `y_w` has 100 tokens at an average log-probability of −1.0 (sum −100); `y_l` has 50
tokens at −1.2 (sum −60). Summed log-probabilities rank `y_l` above `y_w`; per-token values rank `y_w`
above `y_l`. With `β = 2.5, γ = 1.4`: rewards `−2.5` and `−3.0`, argument `0.5 − 1.4 = −0.9`, loss
`−log σ(−0.9) = 1.241`, so the pair still contributes a large gradient. Under DPO the same pair would be
scored on the summed quantities, where the length difference dominates.

**Evidence.** AlpacaEval 2 length-controlled win rate, Llama-3-Instruct v0.1 setup: SFT 26.0, DPO 40.3,
SimPO 44.7; Mistral-Instruct: 17.1 / 26.8 / 32.1 (Table 4). Under the v0.2 labels the ordering changes:
DPO 48.2, SimPO v0.1 44.7, SimPO v0.2 53.7 (Table 12), so the margin over DPO depends on the annotator as
well as on the loss. Removing length normalization costs the most of the
ablated components (Table 5 row "w/o LN": 11.9 versus 21.5 on Mistral-Base).

**The cost, stated by the same paper.** On the Open LLM Leaderboard tasks, GSM8K falls for almost every
preference method: Llama-3-Base SFT 46.32 → DPO 38.67 → SimPO 31.54 (Table 9). Continuing from
Llama-3-8B-Instruct with three learning rates (Table 16), as AlpacaEval 2 LC / ZeroEval GSM / ZeroEval
MMLU: starting model 26.0 / 78.5 / 61.7; LR 4e-7 → 38.8 / 77.9 / 62.6; 5e-7 → 44.6 / 77.0 / 62.3; 1e-6,
the released v0.2 checkpoint, → 53.7 / 57.4 / 54.9. The authors describe the last row as catastrophic
forgetting on GSM8K and MMLU: one hyperparameter trades the chat metric against the capability metrics.

**Length-normalized DPO.** Tülu 3 keeps the reference model and normalizes each log-ratio by its own
length ([[tulu-3]] §5.2, Eq. 6): `L = −log σ( (β/|y_c|)·log[π_θ(y_c)/π_ref(y_c)] −
(β/|y_r|)·log[π_θ(y_r)/π_ref(y_r)] )`. In their algorithm ablation on an early SFT checkpoint with
UltraFeedback (§5.4.1, Table 18), average score: SFT base 55.7; SimPO 51.8 and 52.9; DPO (β = 0.1, 3
epochs) 55.2; PPO 54.5 and 55.5; length-normalized DPO 46.8–57.3 depending on `β` and epochs, best at
`β = 5`, 1 epoch, LR 5e-7 (57.3). Only the length-normalized variant beat the starting checkpoint, which
is why the released Tülu 3 models use it ([[open-instruct-allenai-recipes-recipe]]).

---

## §6 ORPO: one stage, odds ratio, no reference model

**The measured problem.** Cross-entropy on chosen responses gives no term that lowers rejected
responses. Training OPT-350M on the chosen side of [[hh-rlhf]], [[orpo]] Fig. 3 shows the log-probability
of rejected responses rising along with the chosen ones, and states that rejected responses sometimes end
up more probable than chosen ones (§3).

**Mechanism.** [[orpo]] defines the sequence likelihood as a **length-normalized** quantity (Eq. 3):
`log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t | x, y_<t)`, with `m` the token count. On top of that:

```
odds_θ(y|x) = P_θ(y|x) / (1 − P_θ(y|x))                                                  (11)
OR_θ(y_w, y_l) = odds_θ(y_w|x) / odds_θ(y_l|x)                                           (12)
L_ORPO = E[ L_SFT(y_w|x) + λ · L_OR ],   L_OR = −log σ( log OR_θ(y_w, y_l) )              (13)
```

`L_SFT` is the causal-LM negative log-likelihood of the chosen response and `λ` weights the odds-ratio
term. The length normalization in Eq. 3 is what makes the odds meaningful: if `P_θ` were the raw sequence
probability, it would be the product of hundreds of token probabilities, so `1 − P_θ ≈ 1` and
`log OR` would collapse to the plain log probability ratio.

**Worked example.** Chosen response with mean token log-probability −0.5: `P_θ = 0.6065`,
`odds = 1.541`. Rejected with mean −1.0: `P_θ = 0.3679`, `odds = 0.582`. Then `log OR = 0.974` and
`L_OR = −log σ(0.974) = 0.320`. The corresponding log probability ratio is `−0.5 − (−1.0) = 0.5`, giving
0.474. For this pair the odds-ratio margin is the larger of the two, so its loss and its gradient are the
smaller. That direction matches the caption of the paper's Fig. 6 ("log OR has a wider range given the
same input probability pairs"), but not the body of §7.1, which states that "the probability ratio leads
to more extreme discrimination of the disfavored responses than the odds ratio" and uses that to justify
the odds ratio when the preference term is combined with SFT. The two statements in the source point in
opposite directions; the computed example above matches the caption. **Open question** which comparison
the authors intended (the §7.1 plot scales the probability ratio by `β` and the caption does not).

**Evidence.** Mistral-7B trained with ORPO on UltraFeedback alone, with no separate SFT stage:
AlpacaEval 2.0 12.20% (Mistral-ORPO-β), IFEval instruction-level loose 66.19%, MT-Bench 7.32 (Table 1,
Table 6). Reward-model win rate of ORPO against DPO on OPT-1.3B: 70.9% on HH-RLHF, 57.8% on
UltraFeedback (Tables 2–3); at OPT-125M ORPO loses to DPO (41.7% / 48.8%). The `λ` ablation on Mistral-7B
(App. E.1, Fig. 9) is the useful part for this chapter: at `λ = 0.1` the rejected log-probabilities do
not fall; at `λ = 0.5` chosen rise and rejected fall; at `λ = 1.0` both fall while the margin grows.
Training hyperparameters are SFT-scale (peak LR 8e-6) rather than DPO-scale (App. C).

---

## §7 Iterative RPO: DPO plus an NLL anchor, refreshed each round

**The measured problem.** On reasoning data with verifier-labeled pairs, plain DPO lowered GSM8K accuracy
relative to methods that keep a positive term: 61.8% (DPO from Llama-2-70B-Chat) and 60.3% (DPO from an
SFT model) against 63.5% for SFT on gold chains of thought and 73.1% for one iteration of the method
below ([[rpo]] Table 1).

**Mechanism** ([[rpo]] Eq. 1). Per pair, with `c` the chain of thought and `y` the answer:

```
L_DPO+NLL = −log σ( β·log[M_θ(c_w,y_w|x)/M_t(c_w,y_w|x)] − β·log[M_θ(c_l,y_l|x)/M_t(c_l,y_l|x)] )
            − α · log M_θ(c_w, y_w | x) / ( |c_w| + |y_w| )                              (14)
```

`M_t` is the previous iteration's model, used both as the initialization and as the reference; the NLL
term is normalized by the total length of the chosen sequence; `α` balances the two terms. Each iteration
samples `N = 30` solutions per problem, labels them by checking the final answer against the gold answer,
builds `K = 10` pairs per problem (around 55–60K pairs), and trains the next model.

**Evidence.** Llama-2-70B-Chat on GSM8K training prompts only: zero-shot CoT 55.6% → 73.1% → 78.0% →
81.1% → 81.6% over four iterations (88.7% with majority voting over 32 samples); MATH 12.5 → 20.8;
ARC-Challenge 77.8 → 86.7 (Abstract, Table 1). Doubling the data in iteration 1 gives 74.8%, less than
running a second iteration (78.0%), so the gain is not only a data-quantity effect (§4). Settings:
`α = 1` (tuned over {0.25, 0.5, 1, 2}), `β = 0.1` (tuned over {0.05, 0.1, 0.5, 1.0}), LR 7e-7, batch 16.
Figure 3 of the paper shows the chosen-sequence log-probability falling over training for DPO without the
NLL term and not falling with it.

**Industrial version.** Llama 3.1 405B uses DPO with an NLL term at coefficient 0.2 on chosen sequences,
`β = 0.1`, LR 1e-5, masking of header and termination tokens in both responses, and preference batches
drawn primarily from the most recent round (arXiv:2407.21783v3 §4.1.4); post-training runs for six
rounds of annotate-then-train (§4.1.6) ([[llama-3-recipe]]). The report states the NLL term stabilizes training and
prevents the chosen log-probability from decreasing, citing [[rpo]] and [[dpo-positive]]; it prints no
ablation.

---

## §8 What this family improves, and what it does not

### 8.1 Capability profile

[[unpacking-dpo-ppo]] holds the base policy (Tülu 2 13B SFT), the evaluation suite and the data size
fixed and varies one factor at a time. Relative to the SFT model (factuality 55.4, reasoning 47.8, coding
45.1, truthfulness 56.6, safety 91.8, instruction following 44.2), DPO on the best dataset
(per-aspect-scored UltraFeedback, 60,908 pairs) gives 55.3 / 50.9 / 45.9 / 69.3 / 91.9 / 52.8. Across all
14 datasets factuality stays inside 54.7–55.7 and reasoning inside 46.0–50.9. The same paper's ordering
of factors is: preference data first, learning algorithm second, reward-model quality third, extra
unlabeled prompts last (Abstract). Averaged over five datasets, PPO minus DPO is +1.3 reasoning, +2.9
coding, +2.3 safety, −2.5 truthfulness, +0.7 overall (Table 2). **Result (single study)**, one base model.

Reading this together with [[simpo]] Table 9 and Table 16, and with [[noise-contrastive-alignment]]
Table 3 (Mixtral-8×7B-SFT on UltraInteract: average over eight reasoning benchmarks 56.3 → 50.1 with DPO,
→ 57.9 with NCA), the pattern that replicates across three independent studies is: chat and
instruction-following metrics improve; mathematics and code are the first things to degrade; factuality
moves least in either direction. **Replicated.**

### 8.2 Ranking accuracy is not what is being learned

Define ranking accuracy as `R = E 1[π_θ(y_w|x) ≥ π_θ(y_l|x)]` ([[preference-ranking-accuracy]] Def. 2.3).
It is not the implicit-reward accuracy that trainers log (that compares `r̂_w` with `r̂_l`, which involves
the reference model). Measured on the AlpacaFarm validation split, length-normalized / raw: Zephyr-7B-DPO
54% / 42%, Tülu-2-DPO-7B 53% / 42%, Llama-2-7B-Chat-HF 53% / 40%, while the idealized ranking accuracy
computed from the same reference models (its median over a range of `β`) is 97–98% length-normalized and
99% raw for those three models (Table 1; the fourth model in the table, Gemma-7B-IT, has a
length-normalized median of 73%).

The mechanism is Theorem 4.1: for a pair whose reference log-ratio is
`log[π_ref(y_l|x)/π_ref(y_w|x)] = c`, the trained model ranks that pair correctly if and only if the
pair's DPO loss satisfies `L ≤ −log σ(βc)`.

**Worked example.** `β = 0.1`. A pair the reference model gets wrong by `c = 5` nats requires the per-pair
loss to fall from `log 2 = 0.693` to `−log σ(0.5) = 0.474`. One the reference gets wrong by `c = 30` nats
(easy when the two responses differ in length by tens of tokens) requires `−log σ(3) = 0.049`. Training
that reduces the average loss to 0.3 flips the first kind of pair and not the second. Empirically, at the
point of lowest validation loss on Anthropic HH with Pythia-2.8B, fewer than 10% of the initially
mis-ranked training pairs have been flipped (§4.1, Fig. 2).

**Implication.** Rising `rewards/accuracies` in a trainer log is compatible with a policy that still
assigns higher likelihood to the rejected response on most pairs. The quantity that correlates with win
rate is the distance from the reference: ranking accuracy and win rate move together early and become
anti-correlated once the policy has moved away from `π_ref` (§5).

---

## §9 Offline pairs, on-policy pairs, and over-optimization

**Where the reward peak sits.** [[on-policy-suboptimal-preference-data]] classifies methods by on-policy
sampling, sample reuse and negative gradient, and ties the benefit of the last two to geometry: both help
when the high-reward region lies where `π_ref` puts little mass, and neither is needed when the
high-reward region is already likely under `π_ref` (§5, Fig. 1 right). The evidence is bandit problems,
synthetic length-control tasks, and AlpacaFarm/UltraFeedback runs.

**Same loss, different sampling.** [[on-off-policy-rlhf]] holds the IPO loss and hyper-parameters fixed
and varies only whether the pairs are sampled from the current policy: on-policy sampling reaches a
higher peak win rate on all four datasets tested. Used as pairwise classifiers, offline policies peak at
about 70% accuracy during training while a proxy preference model of the same size reaches 70–90%, and
the offline policies generate worse responses (§2.1 Fig. 1; §5.3.1 Fig. 8).

**Over-optimization without an external reward model.** [[daa-overoptimization-scaling]] trains DPO, IPO
and SLiC on TL;DR (92K pairs) at Pythia 1B, 2.8B and 6.9B with seven regularization settings. GPT-4 win
rate against the reference summary is hump-shaped in the achieved KL for every objective, and under wide
KL budgets the best checkpoint appears after about 25% of a single epoch (§3.1, Figs. 1–2). Within a
model size, neither the implicit-reward classification accuracy nor the training loss predicts win rate
(§3.2, Fig. 4). The smallest model degrades fastest; IPO degrades least.

**Falling chosen likelihood is partly expected.** When `π_ref` is an SFT model trained on the preferred
responses, `E_{p_D(y_w|x)}[log π_θ(y_w|x)/π_ref(y_w|x)] ≈ −D_KL(π_ref ‖ π_θ)`, which is at most zero and
falls as the policy moves ([[daa-overoptimization-scaling]] §3.5, Eq. 7). Falling chosen log-probability
is therefore not by itself evidence of a broken run; what matters is whether sampled quality falls with
it.

**Higher likelihood is not better either.** [[likelihood-overoptimisation-daa]] trains Command R 7B and
35B with Hinge, DPO and IPO on binarised UltraFeedback and an internal set. Win probability as a function
of mean chosen log-likelihood is fitted better by a quadratic than by a line (RMSE lower by 24.4% at 7B,
25.8% at 35B), the best checkpoints sit in the middle of the chosen/rejected likelihood heatmap rather
than at its Pareto corner, and larger margins after about 1,000 steps come with falling win probability
(§4.2, Figs. 1–3). Their proposed stopping signals are a reversal in top-k entropy and a shrinking top-k
probability mass (§4.3).

---

## §10 What the implementation computes

[[openrlhf-dpo]] at commit 64c1cc4, `openrlhf/models/loss.py` L246–281:

```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios

if self.ipo:
    losses = (logits - 1 / (2 * self.beta)) ** 2   # Eq. 17 of arXiv:2310.12036v2
else:
    losses = (
        -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        - F.logsigmoid(-self.beta * logits) * self.label_smoothing
    )
loss = losses.mean()
chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

`logits` is the margin `h` of §2.2; DPO, cDPO (label smoothing) and IPO differ in the last expression
only. Three implementation facts follow from the surrounding code and are worth checking in any trainer:

1. `_get_batch_logps` (dpo_trainer.py L362–388) asserts `average_log_prob == False` and returns the
   **sum** of token log-probabilities with prompt tokens masked out. A trainer that returns the mean is
   computing length-normalized DPO, which needs a different `β` (Tülu 3 uses 5, [[tulu-3]] Table 20).
2. The optional NLL term is the mean token log-probability of the chosen responses, added as
   `nll_loss * args.model.nll_loss_coef` (dpo_trainer.py L157, L172–180). The CLI default coefficient is 0
   ([[openrlhf-dpo-recipe]]), so the anchor of §7 is off unless it is set.
3. The logged quantities are `acc = (chosen_reward > reject_reward).mean()`, `chosen_reward` and
   `reject_reward` (dpo_trainer.py L184–197). `chosen_reward` is `β(log π_θ(y_w) − log π_ref(y_w))`, so
   with a frozen reference its trend is the trend of the chosen log-probability. `acc` is the
   implicit-reward accuracy of §8.2, not ranking accuracy.

For the online variant of the same loss — regenerate both responses each step and label them with a
judge — see [[trl-online-dpo]]; the judge that produces those labels is the subject of ch-42 (Reward
Hacking and Judge Design).

---

## §11 Beyond single-turn chat: long context and agent trajectories

The pair construction, not the loss, is what changes in these settings.

- **Long context.** Llama 3.1 ran DPO on short-context preference data only and reports that long-context
  performance did not regress, conditioned on the SFT model already being strong at long context
  ([[llama-3-recipe]] §4.3.4). [[longpo]] instead builds pairs inside the long-context setting: the chosen
  response is the model's own answer given only the relevant chunk, the rejected one is its answer given
  the full document, the reference distribution is conditioned on the chunk, and an NLL term is kept on
  the chosen long sequence. Mistral-7B-Instruct-v0.2 moves from InfiniteBench 13.82 to 39.27 while MMLU
  moves 59.15 → 59.99; DPO on the same pairs reaches 25.56 and degrades short-context scores.
- **Agent trajectories.** [[eto-trial-and-error]] uses whole trajectories as the two sides of the pair:
  expert trajectory as chosen, the SFT agent's own failed trajectory for the same task as rejected, with
  DPO at `β = 0.1` (0.5 on ALFWorld), LR 1e-6, 3 epochs. Llama-2-7B-Chat reaches average reward 67.4 on
  WebShop versus 63.1 for the SFT agent, with declines after the third explore-train iteration. The
  rejected side is now a long sequence whose failure is localized in a few steps while the gradient treats
  all of its tokens identically. ch-44 (Process Supervision and Verifiable Rewards) covers step-level
  credit assignment as the alternative, and ch-43a the behaviour of the negative gradient itself.

---

## Negative samples and negative feedback

**Which kind of negative.** The rejected response in every objective of this chapter is a **negative used
as gradient** (category 4): the loss contains a term whose effect is to decrease `log π_θ(y_l|x)`. ORPO
and RPO additionally keep a positive cross-entropy term on the chosen response, and [[d2o-negating-negatives]]
replaces the positive side with a distribution of self-samples.

**Where the negatives come from and how reliable they are.** Human comparisons ([[hh-rlhf]]), an LLM
judge with per-aspect scores ([[ultrafeedback]]), a verifier ([[rpo]]: final-answer check), or a reward
model over the policy's own samples ([[simpo]] Instruct setup, [[pairrm]]). Label reliability is part of
the input: [[d2o-negating-negatives]] Fig. 1 reports that 33.88% of the *preferred* responses in
PKU-SafeRLHF and 8.03% in HH are marked toxic by an off-the-shelf classifier, and
[[likelihood-displacement]] §6.2 reports that in over 70% of SORRY-Bench prompts both sampled responses
were refusals, so the "rejected" label separates two responses of the same kind.

**Mechanism, at the level of one softmax.** For a single next-token distribution `p` over the vocabulary,
`∂ log p_y / ∂ z_j = 1[j = y] − p_j`. A step that decreases `log p_y` moves the logits by
`Δz = −η(e_y − p)`: the target logit falls by `η(1 − p_y)` and every other logit rises by `η p_j`, in
proportion to how likely that token already was. After renormalization the mass concentrates on the
argmax.

**Worked example.** Take `p = (0.70, 0.20, 0.08, 0.02)` and push down the last token with `η = 1`. The
new distribution is `(0.806, 0.140, 0.050, 0.004)`: the target drops from 0.02 to 0.004, the most likely
token gains +0.106, and the two middle tokens *also lose* mass (−0.060, −0.030). With `η = 3` the result
is `(0.924, 0.059, 0.016, 0.0002)`. Pushing down a token that is already unlikely moves almost all the
recovered mass to the current argmax. This is the squeezing effect proved for logistic regression in
[[learning-dynamics-llm-finetuning]] App. E and stated as: the mass is squeezed into
`y* = argmax_{i ≠ y⁻} π(i)`, the effect is stronger the peakier the distribution and the smaller
`π(y⁻)`. [figures/negative-gradient-squeeze.html](figures/negative-gradient-squeeze.html) lets the reader
vary the distribution, the step size, and whether a positive term on another token is applied at the same
time.

**Why a positive term does not automatically fix it.** In the same example, applying a positive step on
token 2 alone (`η = 1`) gives `(0.392, 0.502, 0.083, 0.022)`; applying the positive step on token 2 and
the negative step on token 4 together gives `(0.526, 0.409, 0.060, 0.006)`. Relative to the positive step
alone, adding the push-down moves mass to the already-dominant token 1 (0.392 → 0.526) and takes it away
from the intended token 2 (0.502 → 0.409): the negative term's share of the redistribution goes mostly to
the argmax, not to the response the positive term is raising. This is the token-level counterpart of the sequence-level result of
[[dpo-positive]] §3: for pairs that differ at one position, the DPO gradient with respect to the logits
at later positions is `s_j^{(y_l^{<k})} − s_j^{(y_w^{<k})}`, which lowers the logit of the correct token
at every position after the difference. Measured on 900 MetaMath samples, the average log-probability of
tokens after the differing index in the preferred completion is −0.37 for the reference model, −0.26 for
DPOP and −1.82 for DPO (§5, Fig. 4).

**Evidence for benefit.** [[on-policy-suboptimal-preference-data]] §5.2 finds that offline objectives
containing a negative gradient converge faster and to better solutions than supervised objectives on the
same data when the reward peak is unlikely under `π_ref`; [[orpo]] Fig. 3 versus Fig. 7 shows that without
a negative term the rejected log-probability rises during SFT on chosen responses, and that with
`λ = 1.0` it falls.

**Evidence for harm.** [[likelihood-displacement]] trains DPO on on-policy refusal pairs from
SORRY-Bench and reports the training-set refusal rate falling from 74.4% to 33.4% (Llama-3-8B-Instruct)
and 80.5% to 54.8% (Gemma-2B-IT), with mean change in preferred log-probability −48.1 ± 22.1 and
−59.2 ± 5.3 (§6.2, Table 16). [[rpo]] Table 1 reports GSM8K 61.8% for DPO against 73.1% for the same data
with the NLL term. [[noise-contrastive-alignment]] Table 3 reports the eight-benchmark average falling
56.3 → 50.1 under DPO on UltraInteract at Mixtral-8×7B.

**Controls.**

| Control | Form | Evidence |
|---|---|---|
| Positive NLL anchor on chosen | `+ α·NLL(y_w)`, `α = 1` per-token ([[rpo]] Eq. 1); coefficient 0.2 at 405B ([[llama-3-recipe]] §4.1.4) | GSM8K 61.8 → 73.1 at one iteration ([[rpo]] Table 1); chosen log-probability stops falling (Fig. 3) |
| SFT term inside the preference loss | ORPO `L_SFT + λ·L_OR` ([[orpo]] Eq. 6) | λ = 0.5 lowers rejected while chosen rises; λ = 1.0 lowers both (App. E.1, Fig. 9) |
| Penalty when chosen falls below reference | DPOP `−λ·max(0, log[π_ref(y_w)/π_θ(y_w)])` inside the sigmoid, `λ = 50`, `β = 0.3` ([[dpo-positive]] Eq. 3) | Only method to improve over the base model on low-edit-distance MetaMath (Fig. 2) |
| Absolute rather than relative target | NCA: `−log σ(r_w) − ½log σ(−r_w) − ½log σ(−r_l)` ([[noise-contrastive-alignment]] Table 1) | Reasoning average 50.1 (DPO) → 57.9 (NCA) at 8×7B (Table 3) |
| Replace the positive side with self-samples | D2O: `(β/K)Σ_i log[π_θ(y_i)/π_{r-}(y_i)] − α log[π_θ(y_l)/π_{r+}(y_l)]` ([[d2o-negating-negatives]] Eq. 3) | Harmfulness −4.27 and win rate 61.82 versus −1.02 / 32.43 for DPO and 1.21 / 20.13 for plain gradient ascent on negatives (Table 1) |
| Curate the pairs | Drop pairs with high length-normalized CHES ([[likelihood-displacement]] §6.3) | 5% lowest-CHES subset restored refusal rates, more than adding an SFT term |
| Make the rejected side less unlikely first | Include `y⁻` in the SFT stage ([[learning-dynamics-llm-finetuning]] §4.3) | Win rate against the baseline pipeline 0.65 / 0.69 / 0.67 at DPO epochs 2 / 4 / 6 (ChatGPT judge, Table 1) |

**Worked check on one control.** The NCA form has a fixed point the relative losses lack. For the chosen
term, `d/dr[−log σ(r) − ½ log σ(−r)] = −(1 − σ(r)) + ½σ(r) = 0` at `σ(r) = 2/3`, that is `r_w = ln 2`.
The chosen implicit reward is driven toward a positive constant instead of toward `+∞`, which is the
paper's stated reason for the likelihood behaviour in its Fig. 5.

**Diagnostics.** Log `chosen_reward` and `reject_reward` separately, never only their difference
(§10); the failure mode is both falling together. Log the greedy-decode log-probability or top-k entropy
([[likelihood-overoptimisation-daa]] §4.3; [[learning-dynamics-llm-finetuning]] §4.2 tracks the
greedy-decoded sequence, whose log-probability rises from −113 to −63 in 8 epochs while everything
observed falls). For reasoning data, log sampled pass@1 and pass@k, not the loss: within a model size the
loss does not predict quality ([[daa-overoptimization-scaling]] §3.2). For safety data, log the refusal
rate on held-out prompts ([[likelihood-displacement]] §6).

**Effect on generality.** The negative term is what moves probability mass out of the region the SFT
model occupies; it is also what removes mass from responses that were never labeled. The measured
consequences are reduced sampled accuracy on tasks where the correct response is rare ([[rpo]],
[[noise-contrastive-alignment]]), reduced refusal on safety prompts when both sides of the pair are of
the same kind ([[likelihood-displacement]]), and reduced output diversity, whose reversal point coincides
with the win-rate turning point ([[likelihood-overoptimisation-daa]]).

---

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DPO paper runs (GPT-2-large, Pythia-2.8B, GPT-J) | ≤ 6B | preference | β | 0.1 (TL;DR run: 0.5) | arXiv:2305.18290v3 App. B | verified 2026-09-14 ([[dpo]]) | "we did not meaningfully tune DPO's β" (§6.2); IMDb sweep {0.05, 0.1, 1, 5} |
| DPO paper runs | ≤ 6B | preference | optimizer; LR; warmup; batch | RMSprop; 1e-6; linear over 150 steps; 64 | App. B | verified 2026-09-14 ([[dpo]]) | no ablation reported |
| Tülu 2 13B + DPO | 13B | preference | β; LR; epochs | 0.01; 5e-7; 3 | arXiv:2406.09279v2 App. F.1 | verified 2026-09-15 ([[unpacking-dpo-ppo]]) | pilot sweep β {0.1, 0.01, 0.001}, LR {5e-6, 5e-7, 5e-8} on HH-RLHF and UltraFeedback |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | loss; β; LR; batch (pairs); epochs; max length | length-normalized DPO; 5; 5e-7; 128; 1; 2,048 | open-instruct `docs/tulu3.md` L144–159; report §5.4.1 | verified 2026-09-14 ([[open-instruct-allenai-recipes-recipe]]) | Table 18: SFT base 55.7, DPO (β 0.1) 55.2, SimPO 51.8/52.9, DPO-norm best 57.3 |
| Llama 3.1 405B | 405B | preference | β; LR; NLL coefficient on chosen; token masking | 0.1; 1e-5; 0.2; header and termination tokens masked | arXiv:2407.21783v3 §4.1.4 (rounds: §4.1.6, 6 rounds) | verified 2026-09-14 ([[llama-3-recipe]]) | no ablation printed; NLL term cited to [[rpo]] and [[dpo-positive]] |
| Llama-3-Instruct SimPO v0.1 (8B) | 8B | preference | β; γ; LR; batch; epochs; max length | 2.5; 1.4; 1e-6; 128; 1; 2,048 | arXiv:2405.14734v3 App. B, Table 8 | verified 2026-09-15 ([[simpo]]) | Table 12: AlpacaEval 2 LC 44.7 against DPO 48.2 and SFT 26.0 in the same setting |
| Llama-3-Instruct SimPO v0.2 (8B, preferences labeled by ArmoRM) | 8B | preference | β; γ; other settings | 10; 3; as v0.1 | arXiv:2405.14734v3 App. H | verified 2026-09-15 ([[simpo]]) | Table 12: LC 53.7, the highest in the table; Table 16 LR comparison 4e-7 / 5e-7 / 1e-6 → LC 38.8 / 44.6 / 53.7 with ZeroEval GSM 77.9 / 77.0 / 57.4 |
| ORPO runs (OPT-125M/350M/1.3B, Phi-2 2.7B, Llama-2 7B, Mistral 7B) | 0.125–7B | preference | peak LR; epochs; max length | 8e-6; 10 (best checkpoint by evaluation loss); 2,048 for UltraFeedback, 1,024 for HH-RLHF | arXiv:2403.07691v2 App. C | verified 2026-09-15 ([[orpo]]) | no ablation reported for LR |
| Mistral-ORPO-α, -β | 7B | preference | λ | 0.1 | arXiv:2403.07691v2 §6.1 | verified 2026-09-15 ([[orpo]]) | App. E.1 Fig. 9: λ ∈ {0.1, 0.5, 1.0} changes whether rejected log-probabilities fall; λ = 0.25 used for Phi-2 and 0.2 for Llama-2 (§6.1) |
| Zephyr-β-SFT + KTO | 7B | preference | β; λ_D, λ_U; effective batch; epochs | 0.1; 1.0, 1.0; 32; 1 | arXiv:2402.01306 §4.2 (batch), Table 1 caption (λ), Table 2 (run) | verified 2026-09-15 ([[kto]]) | Table 2: GSM8K 39.0 (SFT) / 40.0 (DPO) / 53.5 (KTO) |
| Llama-3 8B, SFT+KTO | 8B | preference | LR; β | 5e-6; 0.05 | Table 1 | verified 2026-09-15 ([[kto]]) | Table 1: AlpacaEval LC 10.59, BBH 65.15, GSM8K 60.20; the paper states the model is more sensitive to the learning rate than any other hyperparameter (§4.2) |
| Iterative RPO from Llama-2-70B-Chat | 70B | preference | β; α; LR; batch; samples per problem; pairs per problem; iterations | 0.1; 1.0; 7e-7; 16; N = 30; K = 10; 4 | arXiv:2404.19733v3 "Experimental setup" | verified 2026-09-15 ([[rpo]]) | α tuned over {0.25, 0.5, 1, 2}, β over {0.05, 0.1, 0.5, 1.0}; Table 1: 73.1 → 78.0 → 81.1 → 81.6 |
| Mistral-7B-based model + DPOP | 7B | preference | β; λ | 0.3; 50 | arXiv:2402.13228v2 §5 | verified 2026-09-15 ([[dpo-positive]]) | λ ∈ {5, 50, 500} changed results little; β ∈ {0.1, 0.3, 1.0} did not prevent the DPO failure (Fig. 3) |
| Mixtral-8×7B-SFT + NCA | 8×7B (MoE) | preference | β; LR; epochs; schedule | 0.1; 5e-7; 1; cosine, warmup ratio 0.1 | arXiv:2402.05369v3 App. C | verified 2026-09-15 ([[noise-contrastive-alignment]]) | Table 3: average 56.3 (SFT) / 50.1 (DPO) / 57.9 (NCA) |
| Pythia 1B / 2.8B / 6.9B on TL;DR | 1–6.9B | preference | data; batch; optimizer; LR; epochs | 92K pairs; 128; RMSProp; 5e-7 with 150 warmup steps; 1 | arXiv:2406.02900v2 App. "Experimental details" | verified 2026-09-15 ([[daa-overoptimization-scaling]]) | Figs. 1–2: hump-shaped win rate versus KL; best point near 25% of an epoch |
| Command R 7B / 35B | 7B, 35B | preference | batch; LR; epochs; monitoring | 32; 5e-6 or 1e-5; 1; evaluate every 50 steps with early stopping | arXiv:2410.11677v2 §4.1 | verified 2026-09-15 ([[likelihood-overoptimisation-daa]]) | Figs. 1–4: quadratic fit of win probability against chosen log-likelihood |
| OpenRLHF `train_dpo.py` CLI defaults | any | preference | β; label smoothing; NLL coefficient; epochs; LR | 0.1; 0; 0; 1; 1e-5 | train_dpo.py L236–246, L255 | verified 2026-09-14 ([[openrlhf-dpo-recipe]]) | no ablation reported; class defaults differ (β 0.01, 2 epochs) |
| OpenRLHF `train_dpo_llama.sh` example | 8B | preference | β; LR; epochs; max length | 0.1; 5e-7; 1; 8,192 | train_dpo_llama.sh L13–17 | verified 2026-09-14 ([[openrlhf-dpo-recipe]]) | no ablation reported |

**Starting point for a small general-purpose run.** For an 8B SFT checkpoint and a preference set of the
size of binarised UltraFeedback (about 61K pairs), the configuration with the most supporting evidence in
this table is: summed-log-probability DPO with `β = 0.1`, LR 5e-7, batch 128 pairs, one epoch, max length
2,048, reference model frozen at the SFT checkpoint. The learning rate 5e-7 is the value in both the
OpenRLHF 8B example script and Tülu 3 8B; the batch of 128 pairs and the max length of 2,048 are the
Tülu 3 8B values (the OpenRLHF example uses global batch 256 and max length 8,192)
([[openrlhf-dpo-recipe]], [[open-instruct-allenai-recipes-recipe]]). If
the trainer returns mean rather than summed log-probabilities, the corresponding setting is
length-normalized DPO with `β = 5` (Tülu 3 8B, Table 18: 57.3 against 55.7 for the starting checkpoint).
Add `α = 0.2`–`1.0` NLL on chosen when the data contains verifiable answers ([[llama-3-recipe]], [[rpo]]).
Evaluate every 50–100 steps rather than at the end of the epoch, because the TL;DR runs in
[[daa-overoptimization-scaling]] peaked near 25% of an epoch and the Command R runs in
[[likelihood-overoptimisation-daa]] were monitored every 50 steps with early stopping.

---

## Generalization lens

**(a) What increases breadth.** Better preference data first: in [[unpacking-dpo-ppo]] the spread across
14 datasets at fixed algorithm (average 51.5 to 61.0) is larger than the DPO-versus-PPO gap (0.7), and
per-aspect-scored synthetic data was the best of the 14. Pairs whose two sides are semantically distinct:
[[on-policy-suboptimal-preference-data]] Fig. 17 shows chosen likelihood *rising* for Mistral-7B on
UltraFeedback (responses from different generator models) while it falls for Pythia-1.4B on AlpacaFarm.
Pairs regenerated from the current checkpoint: the on-policy Instruct setup of [[simpo]], the on-policy
preference construction of Tülu 3 ([[tulu-3]] §5.2.1), and the controlled comparison in
[[on-off-policy-rlhf]]. A positive term on the chosen response when correctness matters ([[rpo]],
[[noise-contrastive-alignment]], [[dpo-positive]]).

**(b) What causes narrowing.** Running past the win-rate peak: hump-shaped win rate against KL at every
size and objective tested ([[daa-overoptimization-scaling]] §3.1). Large learning rates on an already
instruction-tuned checkpoint: ZeroEval GSM 78.5 → 57.4 and MMLU 61.7 → 54.9 for the released
Llama-3-8B-Instruct SimPO checkpoint at LR 1e-6 ([[simpo]] Table 16). Reasoning data with near-identical
pairs: the low-edit-distance MetaMath failure ([[dpo-positive]] Fig. 2) and the UltraInteract regression
([[noise-contrastive-alignment]] Table 3). Pairs whose two sides are the same kind of response:
refusal-rate collapse from 74.4% to 33.4% ([[likelihood-displacement]] §6.2). Diversity: entropy over
top-k tokens reverses direction, and the reversal coincides with the win-rate turning point
([[likelihood-overoptimisation-daa]] §4.3).

**(c) How to measure it at this stage.** A held-out win rate with length control
([[length-controlled-alpacaeval]]: the length-controlled score raised Spearman correlation with Chatbot
Arena from 0.94 to 0.98, and verbosity prompting moves the raw score of one model from 22.9% to 64.3%
versus 41.9% to 51.6% after control). A fixed capability suite that the preference data does not target
(MMLU, GSM8K, HumanEval, IFEval) evaluated at the same checkpoints — this is what made the SimPO
regression visible. Sampled metrics rather than loss: neither training loss nor implicit-reward accuracy
predicts win rate within a model size ([[daa-overoptimization-scaling]] §3.2). A safety slice with
refusal rate if any safety data is in the mixture. Known measurement errors: implicit-reward accuracy is
not ranking accuracy ([[preference-ranking-accuracy]] Remark 2.5); ranking accuracy itself becomes
anti-correlated with win rate once the policy has moved from the reference (§5); judges prefer longer
answers, which is what length control removes.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reading `rewards/accuracies` as evidence that the model ranks pairs correctly | Accuracy above 0.7 while sampled outputs do not improve | Compute `1[π_θ(y_w) ≥ π_θ(y_l)]` directly on held-out pairs; expect below 60% ([[preference-ranking-accuracy]] Table 1) |
| Logging only the margin | Loss falls, margin rises, quality falls | Log `chosen_reward` and `reject_reward` separately (§10); a margin that grows because both fall is the displacement regime |
| Treating a falling chosen log-probability as a bug by itself | A healthy run is stopped, or an anchor term is added without need | Compare against the forward-KL identity ([[daa-overoptimization-scaling]] §3.5) and decide with sampled accuracy or win rate |
| Using a `β` from a different loss family | Loss barely moves, or the run diverges in a few hundred steps | Match `β` to the reward units: 0.1 for summed-log-ratio DPO, 5 for length-normalized DPO ([[tulu-3]] Table 20), 2.0–2.5 for SimPO with `γ` 0.5–1.5 ([[simpo]] §3 and Table 8; the later v0.2 setting uses β = 10, γ = 3, App. H) |
| Reference model that is not the model the data came from | Ranking accuracy stuck; large `c` in Theorem 4.1 | Set `π_ref` to the SFT checkpoint that produced or scored the responses ([[dpo]] §4; [[preference-ranking-accuracy]] Theorem 4.1) |
| Training a full epoch or more without intermediate evaluation | Final checkpoint worse than one at 25% of the epoch | Evaluate every 50–100 steps ([[daa-overoptimization-scaling]] Fig. 2; [[likelihood-overoptimisation-daa]] §4.1) |
| Preference pairs whose two sides are the same kind of response | Refusal rate or task accuracy falls while the loss looks healthy | Score pair similarity (length-normalized CHES, [[likelihood-displacement]] §5) and drop the highest scores |
| Reporting only a judge win rate after training | Chat score up, capability scores down and unreported | Report length-controlled win rate ([[length-controlled-alpacaeval]]) next to a fixed capability suite ([[simpo]] Table 9, Table 16) |
| Running DPO on near-identical mathematical pairs | Sampled accuracy falls while the training margin grows | Measure the edit distance of the pairs; add the DPOP penalty or an NLL anchor ([[dpo-positive]] §5; [[rpo]] Table 1) |

---

## Check your understanding

1. Two pairs have the same margin `h = 7` but differ in that the first raised the chosen log-probability
   by 2 nats and the second lowered it by 3 nats. Explain why the DPO loss and gradient weight are
   identical, and name the term in the gradient that is responsible.
2. A colleague reports that `rewards/accuracies` reached 0.85 and concludes the model now ranks preferred
   responses above dispreferred ones. Using Theorem 4.1 of [[preference-ranking-accuracy]], state the
   condition under which that conclusion fails and compute the per-pair loss needed to flip a pair whose
   reference log-ratio is 20 nats at `β = 0.1`.
3. Why does ORPO's odds ratio behave differently from a probability ratio, and what would go wrong if
   `P_θ(y|x)` in Eq. 11 were the raw sequence probability instead of the length-normalized quantity of
   [[orpo]] Eq. 3?
4. The squeezing example in the negative-feedback section moves mass to the argmax token. Explain why
   pushing down a response that the *current* policy already finds unlikely is worse in this respect than
   pushing down one it finds likely, and connect this to the on-policy result in
   [[on-policy-suboptimal-preference-data]].
5. SimPO improves AlpacaEval 2 LC from 26.0 to 53.7 on Llama-3-8B-Instruct and moves ZeroEval GSM from
   78.5 to 57.4. Give a causal account of both effects in terms of the objective and the learning rate,
   and say what you would measure to decide which learning rate to release.
6. Llama 3 adds an NLL term at coefficient 0.2 and masks header and termination tokens. Explain what each
   of those two choices prevents, and why the second one is specific to the fact that the loss is summed
   over tokens.
7. [[daa-overoptimization-scaling]] finds no relation between the training loss and the win rate within a
   model size, while [[likelihood-overoptimisation-daa]] finds that a higher chosen likelihood is not
   monotonically better. Are these the same finding? Justify the answer in terms of what each quantity
   measures.
8. You have 60K preference pairs from a judge and an 8B SFT model, and the requirement is that GSM8K and
   MMLU must not fall by more than 1 point. Choose an objective and a stopping rule, and state the
   evidence for each choice.

---

## Connections

- **Previous: ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output
  Diversity.** Supplies the on-policy-versus-offline framing and the KL-to-reference argument that Eq. (1)
  formalizes here.
- **Next: ch-42 — Reward Hacking and Judge Design.** The judge that labels the pairs used in this chapter
  is itself an optimization target; length control here is one instance of the general problem treated there.
- **ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization.** Supplies
  the Bradley-Terry likelihood used in §2.2 and the over-optimization baseline that
  [[daa-overoptimization-scaling]] compares against.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.** The on-policy family whose negative
  advantage plays the role of the rejected term here.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative
  Advantages.** Holds the full derivations summarized in this chapter's negative-feedback section.
- **ch-45a — Preference-Optimization and RL Stage Recipes Side by Side.** Places the recipe table here
  next to the RL-stage tables.

---

## Sources

- [[dpo]] — derivation (Eq. 3–7), implicit reward, gradient, App. B hyperparameters, TL;DR and CNN/DailyMail results.
- [[ipo]] — ΨPO family, the deterministic-preference argument, the squared loss with target `1/(2τ)`.
- [[kto]] — unary-label objective, the batch reference point `z_0`, the `λ_D n_D / λ_U n_U` rule, Table 1 and Table 2 results.
- [[simpo]] — length-normalized reference-free reward, `γ`, Table 4 win rates, Table 9 and Table 16 capability regressions.
- [[orpo]] — length-normalized odds, the joint SFT plus odds-ratio loss, the `λ` ablation on rejected log-probabilities.
- [[rpo]] — DPO+NLL loss (Eq. 1), iteration protocol, GSM8K/MATH/ARC numbers, chosen-log-probability curves.
- [[likelihood-displacement]] — definition of displacement, CHES score, refusal-rate collapse and its mitigation.
- [[learning-dynamics-llm-finetuning]] — the squeezing effect and its proof setting, greedy-response log-probability, the SFT "extend" mitigation.
- [[dpo-positive]] — the low-edit-distance failure mode, the DPOP penalty, the token-level measurement.
- [[noise-contrastive-alignment]] — DPO as a special case of InfoNCA, the absolute-likelihood NCA form, UltraInteract results.
- [[d2o-negating-negatives]] — alignment from negatives only against a distribution of self-samples, and the failure of plain gradient ascent on negatives.
- [[unpacking-dpo-ppo]] — the controlled 13B comparison of data, algorithm, reward model and prompts.
- [[preference-ranking-accuracy]] — ranking accuracy definition and measurements, Theorem 4.1, relation to win rate.
- [[daa-overoptimization-scaling]] — KL budgets, intra-epoch degradation, loss and implicit-reward accuracy as non-predictors.
- [[likelihood-overoptimisation-daa]] — quadratic relation between chosen likelihood and win probability; entropy and top-k stopping signals.
- [[on-policy-suboptimal-preference-data]] — the on-policy / sample-reuse / negative-gradient taxonomy and the geometric condition.
- [[on-off-policy-rlhf]] — same-loss online versus offline comparison and the classification-versus-generation gap.
- [[length-controlled-alpacaeval]] — the GLM, the length-controlled win rate, and the verbosity-prompt test.
- [[openrlhf-dpo]], [[openrlhf-dpo-recipe]] — the reference implementation of the loss, the NLL option, the logged quantities, and the defaults.
- [[open-instruct-allenai-recipes-recipe]], [[tulu-3]] — length-normalized DPO, Table 18 algorithm ablation, and the released 8B/70B/405B settings.
- [[llama-3-recipe]] — the 405B DPO settings, NLL coefficient, token masking, and round structure.
- [[trl-online-dpo]] — the online variant of the same loss, used in ch-42.
- [[ultrafeedback]], [[hh-rlhf]], [[pairrm]] — the preference datasets and the ranking model used to build on-policy pairs.
- [[longpo]], [[eto-trial-and-error]] — pair construction for long context and for agent trajectories.
- [[rlhf-instructgpt]] — the reward-model-plus-PPO pipeline whose objective this chapter optimizes without RL.
