<!-- chapter: ch-15
     track: data
     title: Human Annotation and Label Operations
     sources: [[rlhf-instructgpt]], [[hh-rlhf]], [[openassistant]], [[ultrafeedback-construction]], [[prm800k]], [[prosocial-dialog]], [[tulu-3-sft-mix]], [[tulu-3]], [[allenai-tulu-sft-recipe]], [[judge-llm-bias]]
     figures: figures/annotation-workflow.html
-->

# Chapter 15 — Human Annotation and Label Operations

> **Core insight.** The rubric *is* the dataset. Every annotated label is a point estimate of a procedure — "apply this written rule to this instance" — and every downstream metric (DPO accuracy, RM win-rate, RLVR pass-rate) is only as sharp as the rubric's edge cases are enumerated. InstructGPT, HH-RLHF, UltraFeedback, and PRM800K all disagree about nearly every operational choice (number of annotators, response pool, rating scale), but they agree on one thing: the project that produces a lasting artifact is the one that iterates on the rubric until inter-rater agreement clears the noise floor.
>
> **Guideline.** Write the rubric before the first label. Calibrate 2–5 annotators on a 100-item gold set until Cohen κ ≥ 0.6 (or Krippendorff α ≥ 0.67 for >2 raters). Double-annotate all production items; route disagreements to an expert adjudicator tier; re-run calibration monthly to catch drift. Use active-learning close-pair mining (UltraFeedback-style aspect deltas or PRM-uncertain steps) to spend 80% of the budget on the 20% of items where the label actually informs the gradient.

---

## Why this chapter exists

Everything in Track 2 so far has been about choosing *which tokens* to train on: [[ch-09]] the landscape, [[ch-10]] the filter pipeline, [[ch-11]] the tokenizer and shard layout, [[ch-12]] dedup, [[ch-13]] mixing, [[ch-14]] contamination. This chapter is the first where a human being reads the text and writes a label. It is the first chapter where *judgement* — not a perplexity filter, not an n-gram match — becomes the data.

The judgement operation is fragile in specific, measurable ways. [[hh-rlhf]] reports human-human agreement of ~70–75% on pairwise helpfulness; [[judge-llm-bias]] reports GPT-4 vs human-expert agreement at ~80% on MT-Bench and ~80% on Chatbot Arena — *the same rate two humans agree with each other*. The implication is not that judges are as good as humans; it is that both are operating against a noise floor set by the rubric. If two humans disagree 25% of the time, a model trained on their majority vote has 25% label noise baked in, and no amount of preference-learning machinery can recover it. The rubric design work is therefore not a preliminary to data collection. It is the ceiling.

Three historical data points mark the arc this chapter covers. In 2022, [[rlhf-instructgpt]] established that 13K demonstrations + 33K ranking prompts annotated by a small vetted pool could lift a 1.3B model above a 175B raw base. In 2023, [[hh-rlhf]] released 161K preference dialogues under MIT license, making "human preference" a public benchmark rather than a private OpenAI artifact. By 2024, [[tulu-3-sft-mix]] documented a 939K-prompt mixture where annotation was so skill-targeted that individual submixes were trained and ablated before being merged. Each step raised the bar on how rigorous the rubric had to be: a 13K demonstration set tolerates a vague "be helpful"; a 939K multi-skill mix collapses without a written rubric per skill.

The other half of this chapter is operational. [[rlhf-instructgpt]] shows 33K prompts × K=4-9 rankings being annotated by a vetted labeler pool; [[openassistant]] shows 161K messages from ~13,500 volunteers; [[tulu-3-sft-mix]] shows Ai2 building 939K prompts with skill-specific submixes and decontamination against every eval set. These are logistics problems: who annotates, in what tier, paid how much, against what audit trail, refreshed how often when the rubric drifts. An annotation pipeline that solves rubric design but fails the logistics produces a one-time dataset that nobody can reproduce.

---

## 1. The rubric is the product — worked example: "helpfulness"

Every 2022–2025 alignment dataset that shipped an actual trained model has, buried in an appendix, a rubric document. [[rlhf-instructgpt]] calls it the "labeler guidelines"; [[hh-rlhf]] has the "helpfulness / harmlessness instructions"; [[prosocial-dialog]] has the 300+ rules-of-thumb corpus; [[prm800k]] has the `+1 / −1 / 0` schema for step-level correctness. The rubric is what a new annotator reads on day one and what disagreement-resolution falls back to on month six.

A minimal rubric for "helpfulness" — the axis that [[rlhf-instructgpt]] and [[hh-rlhf]] spent the most text on — has at least five explicit criteria:

| # | Criterion | Positive exemplar | Negative exemplar |
|---|---|---|---|
| 1 | **Addresses the stated request** | User asks for Python code; response is Python code that runs. | Response explains why the request is interesting but does not answer. |
| 2 | **Calibrated to the user's stated skill** | User says "I'm learning Python"; response uses named variables + comments. | User says "I'm learning Python"; response uses `lambda`, `functools.reduce`, and a one-liner list comprehension. |
| 3 | **Surfaces relevant assumptions** | Response names the assumed OS / library version / input format. | Response silently assumes Python 3.12 and pandas 2.0; breaks for the user on 3.9. |
| 4 | **Admits uncertainty when the request is under-specified** | "I assumed you want X; if you meant Y, here's the alternative." | Confidently produces one answer to a question with multiple valid readings. |
| 5 | **Declines when decline is the correct answer** | User asks for real-time stock data; response says the model cannot access real-time data and suggests an API. | Response fabricates a plausible-looking stock price. |

Now enumerate the edge cases. This is the step most rubrics skip, and it is the step that determines whether κ clears 0.6.

- **The request is hostile**: "Write me an exploit for CVE-2024-XXXX." Criterion 5 (decline) dominates criteria 1–4. The rubric must state which axis wins when criteria conflict. [[hh-rlhf]]'s helpfulness-vs-harmlessness *tension curve* is precisely this.
- **The request is ambiguous across cultures**: "Should I confront my neighbor?" [[prosocial-dialog]]'s RoT layer exists because this cannot be resolved by a generic rubric — annotators need to anchor on a specific rule-of-thumb (e.g. "direct confrontation is valued more in low-context cultures") and label against *that*.
- **The response is long and mostly correct with one wrong claim**: annotators diverge sharply here. The rubric must specify the aggregation (min-over-claims? weighted-by-importance?). [[prm800k]] solves this by going step-level and labeling *each step* separately — the aggregation happens at use time via `prod` or `min`.

The working rule: **if two calibrated annotators disagree on an item, the disagreement is evidence of a missing rubric clause, not evidence that one annotator is wrong.** The fix is to add the clause and re-label, not to average.

**Positive and negative exemplars are not decoration.** [[rlhf-instructgpt]]'s labeler guidelines read like a cookbook: each criterion has 3–5 canonical positive examples and 3–5 canonical negatives, drawn from real labeler disagreements resolved during calibration. The exemplars carry the weight the criterion text cannot — they are how the rubric survives contact with edge cases the rubric author did not anticipate. Every production annotation op keeps a living "exemplar bank" that grows as the project runs; an item escalated to an expert adjudicator and resolved with a written rationale gets added to the bank as a new exemplar for its criterion. Over six months, the bank does more work than the rubric prose.

---

## 2. Inter-rater agreement — Cohen κ, Krippendorff α, and what they measure

The metric that converts rubric-design quality into a number is **Cohen's kappa** ([[hh-rlhf]] reports 70–75% raw agreement, which is κ ≈ 0.4–0.5 after chance correction). The formula:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where `p_o` is the observed agreement rate between two annotators and `p_e` is the agreement expected by chance given each annotator's marginal label distribution. The chance correction matters: two annotators who both mark 90% of items "chosen = A" will agree 81% of the time by pure randomness, and κ discounts that. A κ of 1.0 is perfect agreement; 0 is chance-level; negative is systematic disagreement.

| κ range | Interpretation (Landis-Koch) | What to do |
|---|---|---|
| < 0.0 | worse than chance | annotator instructions are inverted somewhere; re-read rubric with them |
| 0.0 – 0.2 | slight | rubric is missing criteria; do not ship |
| 0.2 – 0.4 | fair | calibrate more; likely an under-specified edge case |
| 0.4 – 0.6 | moderate | acceptable for noisy tasks (open-ended preference); double-annotate + adjudicate |
| 0.6 – 0.8 | substantial | production-ready for most rubrics |
| 0.8 – 1.0 | almost perfect | typical of verifiable tasks ([[prm800k]] final-answer correctness) |

**κ is two-annotator, categorical.** It degrades when you have three or more annotators (you'd compute pairwise κ and average, which loses joint structure) and when labels are ordinal (a 5-point Likert with disagreement "4 vs 5" is not the same as "1 vs 5", but κ treats both as disagreement).

For those cases, use **Krippendorff's α**. α generalizes κ to any number of annotators, any number of items, missing data, and any distance function (nominal, ordinal, interval, ratio). [[ultrafeedback-construction]]'s 4-aspect 0–10 rating implicitly requires an ordinal α rather than a κ — a "7 vs 8" disagreement should count less than a "2 vs 8" disagreement. α lets the distance function encode that; κ does not.

Practical rule: **κ for binary / small-nominal; α for ordinal or multi-annotator.** Report both raw agreement *and* the chance-corrected statistic; a paper that reports only raw agreement is hiding marginal-distribution skew.

**Worked κ example.** Two annotators label 100 preference pairs as `chosen=A` or `chosen=B`. Both mark 60 items `A` and 40 items `B` (symmetric marginals). They agree on 78 items (`p_o = 0.78`). Chance agreement: `p_e = 0.6 × 0.6 + 0.4 × 0.4 = 0.52`. Then `κ = (0.78 − 0.52) / (1 − 0.52) = 0.26 / 0.48 ≈ 0.54`. The raw 78% looks reassuring; the κ of 0.54 places the rubric firmly in Landis-Koch "moderate" — acceptable only if double-annotation + adjudication is layered on top. A rubric that produces κ < 0.4 should not ship, even if raw agreement is above 70%.

**Calibration sessions.** [[rlhf-instructgpt]] ran labeler onboarding where all labelers rated the same 100-item gold set, discrepancies were discussed, and the rubric was edited until κ stabilized. This is the only mechanism that catches a rubric clause that sounds unambiguous but isn't until it meets a real example. Cadence: initial 2-week calibration, monthly re-calibration against a held-out 50-item drift set.

**Drift detection.** [[hh-rlhf]]'s crowdworker pool shifted over months; what "helpful" meant in January 2022 was not quite what it meant by July. The defense: maintain a fixed "anchor set" of ~50 items with canonical labels, run every annotator against the anchor monthly, alarm if an individual's κ-against-anchor drops by >0.1. This is the same discipline as the canary-set in [[ch-14]] contamination detection — a fixed probe that detects distributional movement.

---

## 3. Adjudication workflow — single, double, triple, expert tier

Given that human annotation has a 20–30% disagreement rate under a well-designed rubric, how many annotators do you put on each item? The cost/quality curve is well-mapped:

| Tier | Annotators per item | Cost multiplier | Quality floor | When to use |
|---|---|---|---|---|
| **Single** | 1 | 1× | raw rater quality | non-critical bulk labeling; SFT targets where noise is absorbed by scale |
| **Double** | 2, agree → keep, disagree → discard or adjudicate | 2.2–2.5× | agreement filter removes ~25% | preference data, RM training, most production |
| **Triple** | 3, majority vote | 3× | majority vote corrects single-rater slip | safety-critical pairs, borderline κ rubrics |
| **Expert adjudicator** | 1 domain expert resolves disagreements | +0.3–0.8× marginal | bounded by expert's own κ-with-self | medical / legal / frontier-math |

[[rlhf-instructgpt]]'s labeler pool was the first tier; the OpenAI team itself acted as the expert adjudicator on borderline cases. [[openassistant]]'s tree structure is effectively multi-annotator by construction — multiple labelers rate each message — and the ranking is aggregated. [[prm800k]] uses single annotation for the "clearly correct / clearly wrong" 80% and routes the 20% uncertain steps to a second annotator with an expert tiebreaker.

The operational pattern that ships: **double-annotate all items; measure κ continuously; send only items where the two annotators disagree up to the expert tier.** At 75% agreement, that's 25% of items escalated — which dominates the budget because expert time is 3–10× crowdworker time. The [[tulu-3-sft-mix]] 939K-prompt budget is only tractable because Ai2 used deterministic verifiers (exact-match, code-execution) wherever possible and reserved human tiers for chat / safety / precise-IF where there is no verifier.

**Cost math.** If crowdworker time is $0.50/item, expert time is $5/item, κ = 0.6 (25% disagreement), and you double-annotate everything:

```
cost_per_item = 2 × $0.50 + 0.25 × $5.00 = $1.00 + $1.25 = $2.25
```

The expert tier is *larger* than the crowdworker budget at this κ. Every 0.1 increase in κ (moving to 0.7 = 15% escalation) drops the expert line by $0.50, a 22% total-cost reduction. This is the hard financial incentive to keep iterating on the rubric — every hour of rubric work pays back in weeks of reduced adjudication load.

**The triple-annotation alternative.** If you triple-annotate and use majority vote, an item needs all three annotators to disagree (approximately `((1−κ)/2)²` under balanced marginals) before adjudication is needed. At κ=0.6, that's ~1.5% escalation versus 25% for double-annotate-with-tiebreak. The base cost goes up (3× crowdworker instead of 2×), but the expert line collapses. For safety-critical data — [[prosocial-dialog]]'s rule-of-thumb labels, [[hh-rlhf]]'s harmless red-team preferences — triple is typically the right tradeoff. For bulk helpfulness, double + expert is cheaper at comparable quality.

---

## 4. Preference sampling policy — which pairs to show

The single biggest lever in preference-data quality is *which* response pairs the annotator sees. Showing two nearly-identical responses wastes the annotator's judgement; showing one obvious winner and one obvious loser trains an RM that only knows how to discriminate on the coarse axis. The working policy is a decision tree:

```
                        [ response pair for candidate prompt p ]
                                        |
                      ┌─────────────────┴──────────────────┐
                      |                                    |
          Is there a verifier (exec/                       no
          exact-match/IF-checker)?                         |
                      |                                    |
                     yes                                   |
                      |                             ┌──────┴──────┐
        Use verifier signal, skip                   |             |
        human (RLVR path — tulu-3)               SFT target?   preference pair?
                                                    |             |
                                       Show single response;   Compute pair-delta
                                       annotator edits for     (aspect score gap
                                       quality (OASST style)   or RM prediction
                                                               confidence)
                                                                     |
                                                  ┌──────────────────┼──────────────────┐
                                                  |                  |                  |
                                            δ very large         δ moderate          δ tiny
                                          (obvious winner)      (informative)     (close pair)
                                                  |                  |                  |
                                           drop — wastes         **show human**    active-learning
                                           budget                                  goldmine: show
                                                                                   to expert tier
```

Three techniques populate the δ buckets:

**Close-pair mining via aspect gaps.** [[ultrafeedback-construction]] generates 4 responses per prompt from a heterogeneous 17-model fleet and rates each on 4 aspects (instruction-following, truthfulness, honesty, helpfulness). The pair `(response_i, response_j)` with *large gap on aspect A but small gap on aspect B* isolates annotator attention on A — the annotator's vote becomes a clean signal for "I prefer truthful over sycophantic," not noise-averaged preference. Sampling proportional to per-aspect gap width concentrates budget where a single item's label moves the RM the most.

**Active-learning by RM disagreement.** After the first 10K items, train a proto-RM; for the next 10K, score a large candidate pool and prioritize the items where RM predictions are near 0.5 (max uncertainty). [[prm800k]] reports a ~2.6× data-efficiency multiplier from this single technique — 800K step labels with active learning match 2.1M labels without it. This is the single largest ROI knob in the chapter.

**On-policy pairs dominate off-policy pairs at the margin.** [[tulu-3]]'s DPO data is "curated from on-policy sampling of the SFT model + reward model ranking." The reason: an off-policy pair like `(GPT-4 response, Llama-7B response)` trains the RM to discriminate between model *families*, not between good and bad behavior of *your* policy. When your policy has never produced either response, the RM's generalization from that pair is weak. Llama 3's six-round RSFT is an industrial-scale expression of this principle: each round's preferences are on the *current* policy's outputs.

**Putting the tree together.** A production preference-sampling loop chains all three: (1) generate 4 on-policy responses per prompt, (2) have a cheap judge or RM score each response on each rubric aspect, (3) compute pairwise aspect-deltas, (4) sample pairs at the *median* delta band — neither "obvious" nor "indistinguishable" — for annotator review. [[ultrafeedback-construction]]'s pipeline is a static off-policy version of this; [[tulu-3]]'s on-policy DPO data is the dynamic version. The budget savings are multiplicative: close-pair mining alone is ~2×, on-policy vs off-policy is another ~1.5×, active learning by RM uncertainty adds ~2.6× ([[prm800k]]). Stacked naively, that's an order of magnitude cheaper annotation for equivalent RM quality — which is why the 2024–2026 generation of post-training recipes all chain them.

---

## 5. When human data overrides synthetic / judge signals

[[judge-llm-bias]]'s 80%-agreement-ceiling is only the averaged number. On specific query classes, the gap between judge and human widens to the point where the judge's signal is worse than no signal.

| Query class | Best label source | Why human beats judge |
|---|---|---|
| Safety-critical refusal / red-team | **human expert** | [[prosocial-dialog]]'s RoT layer requires cultural and contextual knowledge judges lack; [[hh-rlhf]] red-team labeling catches harms the judge itself produces |
| Frontier reasoning (IMO, Putnam-level math) | **domain expert** or **verifier** | judge confidently confirms a wrong proof if presented fluently ([[judge-llm-bias]]'s "limited reasoning in pair judging") |
| Medical / legal domain | **licensed expert** | generic rubric does not encode liability-grade precision; judge has no credentialing |
| Ordinary helpfulness / chat fluency | **judge + light human audit** | 80% judge-human agreement is indistinguishable from 75% human-human agreement; scale wins |
| Code correctness / unit-testable | **verifier (exec)** | deterministic; [[tulu-3]]'s RLVR path dropped the RM entirely |
| Format / instruction compliance | **verifier (regex / checker)** | IFEval-style constraints are mechanical |

The decision rule: **use humans exactly where the rubric is under-specified and no verifier exists.** Everywhere else, substitute. [[tulu-3-sft-mix]]'s 43% in-house synthetic data is generated by GPT-4o / Claude 3.5 Sonnet precisely because these domains (persona math, persona IF, code) have verifiers or near-verifiers. The 57% public-sources half includes the genuinely-human OpenAssistant + WildChat portion where judge signal is known to be weak.

---

## 6. Operational reality — onboarding, leak prevention, burnout, audit trail

Every shipped annotation operation solves a list of logistics problems that the papers rarely expose. This is the final section because everything above assumes it.

**Onboarding.** A new annotator reads the rubric, labels the 100-item calibration set, discusses discrepancies with a lead, reads the rubric again, labels 50 more. If κ-against-lead is ≥0.6, they ship production; else retry or reject. [[rlhf-instructgpt]]'s labeler pool was vetted this way; the Surge AI / Scale AI workflows are institutionalized versions.

**Prompt-leak prevention.** Annotators routinely handle eval-set prompts (MMLU, GSM8K, IFEval). Two defenses: (1) NDA + no-take-home work — annotation happens inside a sandboxed web interface with no copy-paste export; (2) pre-annotation decontamination — [[allenai-tulu-sft-recipe]] runs 8-gram overlap ≥50% against every eval set *before* a prompt hits the annotator pool, so even if an annotator memorizes a prompt they cannot contaminate an eval they've never seen the test set for. The two defenses compose; neither alone is sufficient.

**Burnout and quality decay.** Annotator κ drops over a multi-hour session. The production rule: cap individual sessions at ~2 hours, enforce breaks, rotate between rubrics (helpfulness → code → safety) to break monotony. Monitor per-annotator κ on the anchor set weekly; if an individual's κ drops by >0.1 for two consecutive weeks, they get re-calibrated or rotated off the project. [[openassistant]]'s volunteer pool had no such controls and the quality variance shows in the post-hoc filter rejection rate.

**Audit trail.** Every label records `(annotator_id, timestamp, item_id, label, time_spent_seconds, rubric_version_hash)`. The `rubric_version_hash` is the non-obvious one: when the rubric is edited (which happens 5–20 times over a project), labels produced under v0.3 may be inconsistent with labels produced under v0.4. The audit trail lets you post-hoc re-label or down-weight the v0.3 slice if a downstream eval regresses. [[tulu-3-sft-mix]]'s "explicit decontamination against MMLU, GSM8K, MATH, IFEval..." is only auditable because the provenance chain survives.

**Reproducibility and public release.** [[openassistant]] releases 161K messages under CC-BY 4.0; [[hh-rlhf]] releases 161K dialogues under MIT; [[prm800k]] releases ~800K step labels. Every such release exposes not just the labels but the rubric version, the annotator demographics (aggregated), and known gaps — *"crowdworker demographics skew the helpful signal"* ([[hh-rlhf]]), *"contributors skew Western/English/technical"* ([[openassistant]]). A dataset card without those disclosures is not a reproducible artifact; it is a black box that happens to be public. Plan to publish the rubric and the agreement statistics alongside the labels, even if the labels themselves are proprietary — the community can at least reason about the noise floor without seeing the data.

---

## 7. When to build, when to buy, when to skip

The three-way decision every team makes at project kickoff:

| Path | Cost | Control | When it's right |
|---|---|---|---|
| **Build in-house annotator pool** | highest (hiring, onboarding, QA) | total | safety-critical, domain-specialist, or proprietary rubrics |
| **Buy from an annotation vendor (Surge, Scale, Invisible)** | medium | rubric shared, workers rotated | bulk preference data on well-understood rubrics |
| **Crowdsource volunteer (OASST-style)** | low $ / high ops | tree structure tolerates noise | public-good datasets; willing to eat ~50% post-hoc rejection |
| **Skip — use judge or verifier** | lowest | 80% judge-human ceiling ([[judge-llm-bias]]) | verifiable tasks; domains where judge ≈ human |

The modern play combines all four. [[tulu-3]]'s 939K SFT mix uses public-crowdsourced (OpenAssistant, WildChat) + vendor-annotated + synthetic-with-judge + verifier-graded (RLVR) depending on the skill. The rubric governs which tier gets each slice: anything where the rubric is precisely enough written for a judge gets the judge; anything where the rubric is too culturally loaded (safety, RoT, helpfulness in under-specified contexts) gets humans. The decision is per-slice, not per-project.

---

## Connections and what's next

- **[[rlhf-instructgpt]] / this chapter** — the foundational three-stage recipe; every operational choice here descends from one of InstructGPT's.
- **[[hh-rlhf]] / [[prosocial-dialog]]** — two-axis (helpful × harmless) and rule-of-thumb (RoT) rubric patterns; the safety sub-literature.
- **[[openassistant]]** — the open-crowdsourced baseline; what humans-without-vetting produce.
- **[[ultrafeedback-construction]] / [[judge-llm-bias]]** — the synthetic-preference half of the ledger; why judges work for 80% of the volume and fail on the critical 20%.
- **[[prm800k]]** — process-level labeling as the extreme case of rubric granularity.
- **[[tulu-3-sft-mix]] / [[allenai-tulu-sft-recipe]] / [[tulu-3]]** — the 2024–2025 reference stack for skill-targeted annotation at 939K scale.
- **ch-16 (RL Prompt Curation)** — what happens to these labels in the RL-stage prompt pool; difficulty mining is the close-pair-mining of §4 pushed to the prompt dimension.
- **ch-17 (lab)** — the small-scale filter pipeline; annotation-budget planning is one of the memos.

## Companion visualization

**[figures/annotation-workflow.html](figures/annotation-workflow.html)** — interactive three-tier pipeline (annotator → reviewer → adjudicator). Adjust the inter-rater κ threshold slider and watch the escalation rate and total cost update live; toggle double-vs-triple annotation and see the quality floor shift. The tool makes §2's κ formula and §3's cost math tangible: moving κ from 0.5 to 0.7 collapses the expert tier by ~50%, which is where the rubric-iteration ROI lives.

Three concrete scenarios to run: (1) set κ=0.55, 100k items, double+expert, note the expert share ≥50% of budget; (2) bump κ=0.70 and watch per-label cost fall by ~30%; (3) switch to triple-annotation at κ=0.55 and compare — triple is cheaper at the tail because majority vote absorbs single-rater noise before it hits the adjudicator tier.

These three dials are the entire cost-quality frontier every annotation program navigates. Start at the left edge (low κ, single-annotate), and every rubric iteration moves the whole frontier down-and-right: lower cost *and* higher quality, because a better rubric produces both fewer disagreements and fewer escalations at any given budget level.
