<!-- chapter: ch-41
     track: rl
     kind: content
     title: Reward Modeling
     deps: [ch-40]
     sources: [[bradley-terry-rm]], [[reward-model-overoptimization]], [[reward-ensembling]], [[pairrm]], [[generative-reward-models]], [[direct-judgement-preference]], [[nemotron-4-synthetic]], [[west-of-n]], [[rlaif-scaling]]
     figures: figures/rm-overopt.html
-->

# Chapter 41 — Reward Modeling

> **Core insight.** A reward model is a learned proxy for human preference, and the moment you start optimizing against it you are in Goodhart territory: proxy reward grows monotonically, gold reward follows an inverted-U in `d = sqrt(KL(π‖π_ref))`, and the peak is a property of the RM — bigger policies do not rescue a weak RM. The engineering question is not "how do I train the best RM" but "how do I keep the proxy-vs-gold gap narrow over the KL budget I actually plan to spend in RL."
>
> **Guideline.** Choose the smallest RM mechanism that fits the failure mode you expect: a Bradley-Terry scalar for stable, homogenous preferences; PairRM when the reranking task needs A-vs-B attention; a generative RM when you need rubric steerability or calibrated uncertainty; a multi-attribute head when composition at RL time is the goal; a judge-LLM when you have no budget to train and the base LM is strong enough to read. Then monitor `R_gold(d)` — not `R_proxy(t)` — and stop RL before the predicted peak.

---

## §1 Bradley-Terry from first principles

The BT model is the scaffold underneath every preference-based post-training method from InstructGPT through DPO. Derive it once, believe it for life.

**Setup.** Each response `y` to a prompt `x` has a latent scalar quality `r(x, y)`. Humans shown a pair `(y_1, y_2)` pick `y_1` with probability depending only on the score *difference*. Bradley & Terry (1952) posited the logistic form:

```
P(y_1 ≻ y_2 | x) = exp(r(x, y_1)) / [exp(r(x, y_1)) + exp(r(x, y_2))]
                 = σ(r(x, y_1) − r(x, y_2))
```

This is a plain logistic regression on score differences — nothing more. The `σ` comes from an *assumption* (logit link) about how latent scores translate to discrete choices; it is not derived, and the whole enterprise of "your RM is well-calibrated" rests on it.

**The RM loss.** Given a dataset `D = {(x, y_w, y_l)}` of triples where `y_w` was chosen over `y_l`, the negative log-likelihood is the InstructGPT RM loss [[bradley-terry-rm]]:

```
L_RM(θ) = − E_{(x, y_w, y_l) ~ D} [ log σ(r_θ(x, y_w) − r_θ(x, y_l)) ]
```

Observe: this is exactly binary cross-entropy with label 1 and logit `r_θ(x, y_w) − r_θ(x, y_l)`. BT = BCE on score differences. Nothing exotic.

**Parameterization.** `r_θ(x, y)` is the scalar output of a linear head `w ∈ ℝ^d` sitting on the last-token hidden state of a pretrained LM. Attested across InstructGPT, Stiennon 2020, Tülu, DeepSeek.

**Identifiability.** Scores are only identified *up to an additive constant per prompt* — adding a constant to both `r(x, y_w)` and `r(x, y_l)` leaves the loss unchanged. The scalar head absorbs this: in practice people either subtract the per-prompt mean before PPO or let the KL penalty anchor the policy. This is why you cannot meaningfully compare `r` values across prompts.

**Plackett-Luce.** For K-way rankings, generalize to `P(y_1 ≻ y_2 ≻ … ≻ y_K) = ∏_i exp(r_i) / Σ_{j≥i} exp(r_j)`. Used when humans rank > 2 items; the loss is a sum of `log σ` terms along the chain.

**Scaling.** Stiennon 2020 Fig. 2 shows BT-trained RM accuracy improves monotonically with RM parameters on TL;DR — at 6B, RM agreement with held-out humans reaches the inter-annotator ceiling. This is the *within-distribution* scaling law; the story changes out-of-distribution, which is §2.

**Failure modes attested in [[bradley-terry-rm]].**
- *Length bias* — scalar `r` eats length as a free feature because longer responses are on average preferred in crowd labels; patched by regressing out length or training on length-matched pairs.
- *Transitivity violation* — humans are not transitive; BT is. IPO and generalized preference objectives sidestep this, at the cost of giving up closed-form DPO.
- *Context-dependence* — BT assumes `r(x, y)` is prompt-local; cross-prompt calibration is not identified and should not be trusted.

---

## §2 Gao 2022 — the inverted-U law

Gao, Schulman, Hilton 2022 constructed the canonical synthetic-preference overoptimization experiment [[reward-model-overoptimization]]. Fix a 6B "gold" RM as ground truth, train smaller "proxy" RMs on its preferences, optimize a policy against the proxy, and plot gold reward vs `d = sqrt(KL(π‖π_SFT))`.

**Why sqrt(KL).** KL is a squared distance locally, so `d = sqrt(KL)` linearizes many relationships. Plot against `d`, not against step or `β`. Both PPO's β-sweep and early-stopping trace out the same `(d, R_gold)` curve — β is not a separate knob, it is a reparameterization of the KL budget.

**Best-of-N curve.** For best-of-N reranking (N ≤ ~10⁴), the fitted form is:

```
R_gold(d) ≈ d · (α_bon − β_bon · d)
```

A quadratic in `d`. Maximum at `d* = α_bon / (2 β_bon)`. Past that, gold reward falls — the proxy continues picking winners, but those winners are decreasingly good by gold. `KL_bon(n) = log n − (n−1)/n` is the analytically derived KL of the BoN distribution, so `d = sqrt(log n − (n−1)/n)`.

**PPO curve.** For RL:

```
R_gold(d) ≈ d · (α_RL − β_RL · log d)
```

The log gives PPO a slower decay than BoN's quadratic — overoptimization accumulates logarithmically, not quadratically. PPO can ride a little further out in `d` before collapse, but it *does* collapse.

**Scaling with RM.** Both α and β shrink smoothly with RM parameters; a 10× larger RM roughly halves the overoptimization slope but does not remove the peak. Larger RMs give you *more* KL budget, not infinite budget.

**Policy size barely matters.** Bigger policies reach the same proxy-vs-gold peak faster — they are better at exploiting the proxy, but the exploit is an RM property, not a policy property.

**Empirical number to memorize.** At RM ≈ 3M params, gold peaks near `d ≈ 3` nats^0.5 and has lost most of the gain by `d ≈ 8`. Larger RMs push the peak right. Modern 7B-scale RMs push it further still, but never to infinity.

**What this demands operationally.**
1. Plot `R_gold(d)` during RL, not `R_proxy(step)`.
2. Stop before the predicted peak.
3. KL penalty β and early stopping are the same knob — don't tune both independently.

See [figures/rm-overopt.html](figures/rm-overopt.html) for an interactive sweep over RM quality and ensemble size — slide the quality down and watch the peak close in on `d = 0`.

---

## §3 Ensembling as Goodhart insurance

Coste 2023 [[reward-ensembling]] asked the natural question: if the proxy has bounded generalization, average K proxies and push the peak right. Their paper replicates Gao's benchmark with an ensemble defense.

**Aggregators.** Let `r_1, …, r_K` be K independently-trained BT RMs on different seeds or data shards.

- **Mean:** `r̄(x,y) = (1/K) Σ_k r_k(x,y)` — simple average; unbiased under iid RM error.
- **LCB (lower confidence bound):** `r̄ − λ · std_k r_k` — pessimistic under disagreement; λ around 1 attested.
- **Min:** `min_k r_k` — maximally pessimistic; robust to one confidently-wrong RM.
- **UWO (uncertainty-weighted objective):** reward minus a penalty on `std_k r_k` — treats disagreement as anomaly.

**Result.** All ensemble strategies delay overoptimization. The peak shifts from `d ≈ 3` (single-RM baseline) to `d ≈ 5–8` with K = 3–5. Mean gives the highest peak; LCB and min are safer but cap lower. The returns are diminishing past K = 5.

**Variance reduction intuition.** For K iid RMs each with error variance `σ²`, mean has variance `σ²/K`. The proxy-vs-gold gap at fixed `d` scales with RM error, so halving variance pushes the peak right by roughly a constant on the `d` axis. This is the whole theoretical justification — cheap, clean, familiar.

**When it fails — shared blind spots.** If all K RMs share a systematic miscalibration (e.g. all trained on the same length-biased dataset), ensembling does not help. Coste demonstrates this on adversarial prompts where all RMs agree on a wrong answer. Ensemble diversity must come from *data shards*, not just random seeds.

**Overhead.** K reward-forward-passes per RL step. Usually affordable because RMs are smaller than the policy, but for 70B-scale RMs this becomes real compute.

**Disagreement as anomaly signal.** `std_k r_k` is a calibrated OOD flag: high std correlates with the policy drifting into novel territory where the RM has thin data. Surface it during training as a diagnostic even when you are not using LCB as the reward.

---

## §4 PairRM and generative RMs — non-scalar alternatives

The BT scalar is not the only architecture.

**PairRM** [[pairrm]]. Joint-encode `(x, y_A, y_B)` and emit a single preference logit:
```
f(x, y_A, y_B) → logit;   P(A ≻ B) = σ(logit)
```
Cross-attention between the two candidates captures relative differences that scalar RMs miss — especially when absolute quality is ambiguous but relative quality is obvious (length-matched, near-tie, paraphrase-similar cases). A PairRM-0.4B DeBERTa matches scalar RMs at Llama-2-7B on MixInstruct and MT-Bench reranking.

Two essential tricks:
- **Swap augmentation:** evaluate both `(y_A, y_B)` and `(y_B, y_A)`, average the logits — cancels position bias at train and inference time. Ablations in [[pairrm]] show 2–3 pp from swap alone.
- **Tournament Best-of-N:** `O(N log N)` pairwise comparisons with bracketed advancement; retains near-optimal selection without the `O(N²)` pass.

PairRM's production role today is not reward *shaping* in PPO — it is *pair filtering* for DPO: keep `(y_w, y_l)` where `PairRM(y_w, y_l) > τ`. Lifts DPO ~2 pp on held-out evals.

**Generative RMs (GenRM)** [[generative-reward-models]]. Replace `<LM + linear head>` with an LM that emits a rationale then a verdict. Read the reward off the log-prob:
```
r(x, y_A, y_B) = log P_RM("A is better" | x, y_A, y_B, rubric)
```
or a pointwise variant that scores `(x, y)` on a rubric with a 1–10 verdict and takes a log-prob-weighted expectation.

Key properties:
- *Critique-then-verdict* adds 3–10 pp on RewardBench over direct verdict — CoT is not free but it's real.
- *Rubric steerability.* The rubric prompt *is* the reward specification; `"longer is not better, flag sycophancy"` generalizes to unseen prompts. Scalar RMs cannot do this at inference.
- *Calibration.* Verdict-token probabilities track ground-truth agreement; use them as LCB inputs when ensembling.
- *Cost.* Slower per query (must generate critique tokens). But reuses the base-LM inference stack — no separate scalar-RM infrastructure.

**d-RLAIF — skip the RM entirely.** Lee 2023 [[rlaif-scaling]] shows that for summarization / helpful / harmless tasks, you can use a strong labeler LM's log-prob of `"Response 1 is better"` as the reward *every step*, with no trained RM. Outperforms classical RLAIF on human-eval win rates. ~100× cheaper preference labels than crowd-source, and the signal refreshes as the policy drifts — no stale-RM problem.

**Self-taught / J1 line** [[direct-judgement-preference]]. Train a judge via DPO on judgment pairs; iterate (judge → prefs → new policy → new judge prefs). Self-Taught Evaluators crosses GPT-4-as-judge in ~3 iterations with ~40K synthetic prefs. Risk: *judge-as-weapon* — using the same judge at training time and at benchmark-evaluation time creates silent leakage.

**West-of-N** [[west-of-n]]. When human prefs are scarce but the current RM is decent, sample N responses, pair `argmax_RM` with `argmin_RM`, and use those extremum pairs to retrain the next-iter RM. One iteration ≈ doubling the human preference set. N = 16 near-optimal. Crucially, the gain comes from *argmax-vs-argmin* — argmax-vs-random loses 4 pp.

---

## §5 Multi-attribute RMs — HelpSteer2 and compositional RL

Nemotron-4-340B [[nemotron-4-synthetic]] ships a reward model with a **5-dimensional head** trained on the HelpSteer2 dataset. The attributes (attested labels):

| Dimension | Meaning |
|-----------|---------|
| Helpfulness | Does the response address the request? |
| Correctness | Is the content factually right? |
| Coherence | Is the response well-structured and clear? |
| Complexity | Is the response appropriately detailed (not too simple, not over-stuffed)? |
| Verbosity | Is the response too long, too short, or right-sized? |

Each dimension has its own scalar output head on the same LM backbone; labels are 0–4 Likert ratings from a small human-annotated set (~10K pairs in HelpSteer2, within the 20K total human-anchor budget Nemotron spent).

**Why multi-attribute.** A scalar BT RM collapses all preferences into one number, and that number absorbs whatever trade-offs the labeling instructions implied. With per-attribute heads, you can *compose* at RL time:

```
r_compose(x, y) = Σ_k w_k · r_k(x, y)       (Helpfulness, Correctness, …)
```

The weights `w_k` are a policy knob set *after* training, not inside the RM. Want safety-first? Upweight a safety head. Want concise answers? Downweight Verbosity's penalty. Nemotron uses this to disentangle verbosity from helpfulness — critical because unconstrained BT RMs reliably learn "longer = better" and the scalar aggregation hides it.

**HelpSteer2 in the Nemotron pipeline.** The 5-dim RM is used as (a) a filter on synthetic responses during alignment, (b) a judge for DPO pair construction, and (c) the scorer inside RPO (reward-aware preference optimization). 98% of Nemotron-4's post-training data is synthetic; only ~20K human pairs ground the pipeline. The composability of the 5-dim head is what lets the same RM serve all three roles at different dimension weightings.

**Open problem.** The attribute set is *ontological* — it encodes a specific theory of "what preference is." HelpSteer2's 5 dimensions are not universal; other projects have used safety, factuality, harmlessness, reasoning quality as separate heads. The dimensions you pick *are* the reward specification.

---

## §6 Decision framework — which RM for which job

The practical question this chapter exists to answer.

| Situation | Pick | Why |
|-----------|------|-----|
| Homogeneous domain, large stable pref dataset (>50K pairs), scalar reward sufficient for PPO | **Scalar BT RM** [[bradley-terry-rm]] | Cheapest to train, cheapest at inference, well-understood failure surface. Default choice. |
| Reranking N candidates inline (DPO pair filtering, inference-time BoN, prompt-wise selection) | **PairRM** [[pairrm]] | Joint A-vs-B attention beats scalar subtraction at small model size; tournament gives `O(N log N)` selection. |
| Need rubric steerability, calibrated uncertainty, or safety-critical deployment where reasoning must be auditable | **Generative RM** [[generative-reward-models]] | Rubric = reward spec at inference; verdict log-prob is calibrated; critique provides audit trail. |
| Multiple preference axes that must be composed differently at RL time (helpfulness vs verbosity vs safety) | **Multi-attribute RM** (HelpSteer2 / Nemotron-style) [[nemotron-4-synthetic]] | Per-attribute heads let you reweight without retraining. |
| Strong base LM available, budget does not support RM training, or RM drift is a known problem | **Judge-LLM** (RLAIF / d-RLAIF) [[rlaif-scaling]] | 100× cheaper per label; d-RLAIF skips the RM entirely; fresh signal every step. |
| An RM trained on a previous-gen policy exists and the new policy lives in the same distribution | **Reuse** | Cheapest path; audit by plotting `R_gold(d)` on new policy samples — if the curve flattens early, the RM has drifted OOD and you need a retrain. |
| Preference data is scarce and expensive; existing RM is decent | **West-of-N** augmentation [[west-of-n]] | One iter ≈ doubling human data; argmax/argmin pairing is the whole trick. |
| Overoptimization predicted (long RL horizon, strong policy, small RM) | **Ensemble** [[reward-ensembling]] on top of any choice above | K = 3–5 shifts peak from `d ≈ 3` to `d ≈ 5–8`; LCB/min for safety-critical. |

**Rule of thumb.** Start with the mechanism one row up from what you think you need, ensemble it, and budget KL against the predicted peak. Upgrading the RM class is expensive; ensembling is cheap.

**Anti-patterns this framework kills.**
- "We'll tune β to avoid overoptimization" — β and early-stopping are the same knob; tune one, monitor `d`.
- "Bigger policy will fix the RM" — Gao 2022: policy size barely matters, peak is an RM property.
- "K = 10 is safer than K = 5" — diminishing returns after K = 5; spend that compute on data-shard diversity instead of more seeds.
- "Retrain the RM from scratch each iteration" — unless the policy has drifted OOD (visible as `R_gold(d)` flattening early), reuse with a fresh pref top-up.

---

## Connections

- **ch-40** (KL control in RLHF) is the mechanism that makes `d = sqrt(KL)` the natural axis and β the budget knob [[reward-model-overoptimization]].
- **ch-42** (reward hacking taxonomy) catalogs the *failure modes* whose risk the RM choices above trade off against.
- **ch-43** (entropy dynamics) covers what happens to the policy's output distribution as `d` grows past the RM's peak.
- **DPO chapters** (ch-44+) exploit BT's closed-form — the loss in §1 is the same loss, just rearranged via the KL-regularized optimal-policy identity.
- **Synthetic data chapters** (ch-23, ch-29) are where the preference pairs for everything above come from; [[west-of-n]], [[nemotron-4-synthetic]], [[direct-judgement-preference]] all assume that stack.

## Further reading

- [[bradley-terry-rm]] — the loss, scaling, failure modes.
- [[reward-model-overoptimization]] — Gao 2022 scaling laws; Figs. 1, 2, 5, 7 to memorize.
- [[reward-ensembling]] — Coste 2023 aggregators; shared-blind-spot counterexample.
- [[pairrm]] — joint-encoder mechanics; swap augmentation; tournament BoN.
- [[generative-reward-models]] — verdict-token reward; critique-then-verdict; rubric steerability.
- [[direct-judgement-preference]] — Con-J, Self-Taught Evaluators, J1; judge-as-weapon risk.
- [[nemotron-4-synthetic]] — HelpSteer2 5-dim head; 98% synthetic with 20K human anchor.
- [[rlaif-scaling]] — d-RLAIF; CoT preference prompts; same-size labeler still helps.
- [[west-of-n]] — best/worst-of-N extremum pairing.

## Companion visualization

**[figures/rm-overopt.html](figures/rm-overopt.html)** — interactive two-panel figure. Panel A: slide RM quality and ensemble size, watch the inverted-U `R_gold(d)` peak slide. Panel B: calibration bars for a scalar PairRM vs a 5-dim HelpSteer2 RM across the five dimensions. Use panel A to build the reflex of reading `d` on the x-axis and `R_gold` on the y-axis; use panel B to see why "one scalar" is a compression choice, not a physical constant.
