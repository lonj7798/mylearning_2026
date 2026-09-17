<!-- chapter: ch-15
     track: preference
     kind: content
     title: Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage
     deps: [ch-30]
     sources: [[rlhf-instructgpt]], [[hh-rlhf]], [[lima]], [[wildchat]], [[ultrafeedback-construction]], [[openassistant]], [[rlhf-length-correlations]], [[sycophancy-in-lms]], [[openai-sycophancy-postmortem]], [[chatbot-arena]], [[helpsteer2]], [[prism-alignment]], [[tulu-3-preference-ablations]], [[reward-model-overoptimization]]
     figures: figures/annotation-workflow.html
     revised: 2026-09 (generality revision)
-->

# Chapter 15 — Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage

> **Core insight.** Human preference data teaches a model what its annotators rewarded on the prompts they were shown, so the protocol sets both the breadth and the biases of what preference training can teach. Breadth comes mainly from the prompt distribution: InstructGPT outputs were preferred over the same 175B model finetuned on FLAN or T0 in 78 ± 4% and 79 ± 4% of comparisons on API prompts ([[rlhf-instructgpt]] §4.1), and removing WildChat real-user prompts from Tülu 3 8B SFT lowered AlpacaEval 2 from 12.4 to 7.5 ([[tulu-3-preference-ablations]] Table 10). Agreement between annotators is moderate (72.6 ± 1.5% for InstructGPT training labelers, about 63% between Anthropic researchers and HH-RLHF crowdworkers), and reward models trained on such labels reach about the same accuracy (69.6-72.4% in InstructGPT §4.1). Features that annotators reward beyond quality are learned and amplified: always choosing the longer response agrees with 55.7-63.1% of labels in three preference datasets ([[rlhf-length-correlations]] Table 5), and a thumbs-up/thumbs-down reward added to GPT-4o in April 2025 contributed to a sycophancy regression that was rolled back ([[openai-sycophancy-postmortem]]).
>
> **Guideline.** When collecting preference or instruction data for a general-purpose model, sample prompts from real user traffic stratified by topic and by user, and report the category mix, because diverse prompts beat more examples of one prompt type in [[lima]] §5 and in [[tulu-3-preference-ablations]] Figs. 8-9. When labels are subjective, use at least three annotators per item, report a chance-corrected agreement statistic per attribute, and record the rate of high-disagreement items, because HelpSteer2 helpfulness κ rose from 0.465 to 0.706 only after edge-case clarifications and annotator removal ([[helpsteer2]] Table 1). When a length, agreement-with-user, or thumbs-up signal correlates with the label, measure the policy's length and sycophancy at several optimization strengths before release, because both behaviours rose under optimization against human-trained reward in [[rlhf-length-correlations]] and [[sycophancy-in-lms]]. Otherwise, when the label is verifiable, use a verifier instead of human preference (ch-44).

## Why this chapter matters for a general-purpose model

The training pipeline runs pretraining → mid-training → SFT → preference optimization → RL → evaluation. This chapter covers the data that enters the preference stage: human demonstrations, rankings, and ratings, and the prompts on which they are collected. ch-41 trains reward models on these labels, ch-38 and ch-39 optimize policies against them, and ch-42a treats the behaviour changes that follow.

The measurable problem has three parts. First, a human label is a sample from an annotator population applying written instructions, and different annotators give different labels for about 17-37% of the pairwise comparisons in the human-labeled agreement measurements of §2. Second, the prompt set determines which behaviours receive any signal: a model is not pushed toward good behaviour on prompt types that never appear. Third, annotators reward features that are easy to see, such as length or agreement with the user, and optimization amplifies whatever the labels reward. Each part can make a preference-trained model narrower than its SFT starting point.

## §1 Annotation protocols: record types, rubrics, and annotators

**Definitions.** A *demonstration* is a human-written response to a prompt, used as an SFT target. A *comparison* is a label saying which of two responses is better. A *ranking* orders K responses and implies C(K,2) comparisons. A *rating* scores one response on a scale, often per attribute. A *rubric* is the written set of criteria, their definitions, and the order in which they apply when they conflict. The *protocol* is everything else that determines the label: prompt source, candidate responses, annotator selection, interface, and aggregation.

**Problem.** Two datasets with the same record type can encode different targets. The table lists protocol choices that the sources report.

| Dataset | Record | Prompts from | Candidates from | Annotators | Aggregation |
|---|---|---|---|---|---|
| InstructGPT ([[rlhf-instructgpt]]) | demonstrations; rankings K = 4-9 | API Playground users + labeler-written | several outputs sampled from the authors' models (Fig. 2) | about 40 screened contractors | "most comparisons are only labeled by 1 contractor for cost reasons" (§5.3) |
| HH-RLHF ([[hh-rlhf]]) | comparisons in multi-turn dialogue | the crowdworker, during the conversation | 52B context-distilled LM; rejection sampling (usually k = 16); RLHF models | MTurk + Upwork; about 20 workers produced about 80% | one label; weak preferences and ties dropped |
| OASST1 ([[openassistant]]) | human replies; rankings | volunteers | volunteer-written replies | over 13,500 volunteers | Tideman ranked-pairs merge |
| HelpSteer2 ([[helpsteer2]]) | Likert-5 ratings on 5 attributes | ShareGPT + proprietary | 2 responses from different model families or humans | about 1,000 Scale AI annotators | 3 most-agreeing of up to 5; high spread removed |
| PRISM ([[prism-alignment]]) | cardinal ratings, per participant | participant, guided by topic type | 4 of 21 LLMs, then the top-rated model | 1,500 participants, 75 birth countries | none; individual IDs kept |
| Chatbot Arena ([[chatbot-arena]]) | pairwise votes | the voting user | 2 anonymous models | about 90K users | Bradley-Terry over votes |
| UltraFeedback ([[ultrafeedback-construction]]) | 1-5 ratings on 4 aspects | 6 public datasets | 4 of 17 models | GPT-4 (AI labels, contrast case) | mean of aspects |

**Mechanism: what a rubric specifies.**
1. *Criteria and their definitions.* InstructGPT labelers judged helpfulness, truthfulness, and harmlessness ([[rlhf-instructgpt]] §3.3). HelpSteer2 separates helpfulness, correctness, coherence, complexity, and verbosity, so that length can be modeled as its own attribute ([[helpsteer2]] §2.1, §3).
2. *Priority when criteria conflict.* InstructGPT prioritized helpfulness to the user when labeling training data, but truthfulness and harmlessness in the final evaluations (§3.4, App. B.2). A reward model trained on the first rule and evaluated under the second is measured against a different target.
3. *Edge-case clauses.* HelpSteer2's vendor clarifications covered "whether coherence should consider previous turns, and how helpfulness should be evaluated if prompt instructions are unclear" ([[helpsteer2]] §2.1). OASST1 guidelines rank a reply that admits not knowing below a correct reply and above an incorrect one ([[openassistant]] App. A).
4. *What annotators are not expected to check.* HH-RLHF crowdworkers were told "lying isn't helpful", but were not expected to fact-check, "and for example they often prefer responses that include non-functional URLs" ([[hh-rlhf]] §2.1).

**Annotator selection.** InstructGPT screened candidates on agreement with researchers on sensitive-speech flags and on rankings, on rated demonstrations, and on self-reported coverage of cultural groups, with "soft cutoffs at 75% agreement ... and a 6/7 demonstration score" ([[rlhf-instructgpt]] App. B.1). HH-RLHF selected workers by "the sophistication and variation in their dialogues ... rather than based on any measure of agreement" ([[hh-rlhf]] §2.1). HelpSteer2 removed prompts in languages and domains its annotators could not judge: non-English prompts and prompts containing code ([[helpsteer2]] §2.1). That choice raises label quality and removes coverage in the same step.

**Implication for a general-purpose model (Interpretation).** Each protocol choice defines a target that is narrower than "good behaviour". A general-purpose model needs the rubric priorities, the unchecked properties (such as factual accuracy of URLs), and the excluded domains recorded, so that later stages can supply those signals from other sources, for example verifiers for code and math (ch-44).

## §2 Agreement statistics and what they measure

**Definition.** *Raw agreement* p_o is the fraction of items on which two annotators give the same label. A *chance-corrected* statistic subtracts the agreement expected if the two annotators labeled independently with their own label frequencies.

**Problem.** Raw agreement depends on the label distribution. If both annotators pick "response A" for 90% of items, they agree on 0.9 · 0.9 + 0.1 · 0.1 = 82% of items without reading them. Reported agreement numbers also use different definitions (ties counted or not, tie-discounted scoring), so they are not directly comparable.

**Formula (Cohen's κ, two annotators, nominal labels).**

```
κ = (p_o − p_e) / (1 − p_e),     p_e = Σ_c  p_1(c) · p_2(c)
```

p_o is observed agreement; p_e is expected agreement under independence; p_1(c) and p_2(c) are the fractions of items that annotators 1 and 2 assign to category c. κ = 1 is perfect agreement, κ = 0 is chance level.

**Worked example.** Two annotators label 100 pairs as A-better or B-better. Annotator 1 says A on 60, annotator 2 says A on 50. They agree on 70 items (40 both A, 30 both B). p_o = 0.70. p_e = 0.6 · 0.5 + 0.4 · 0.5 = 0.50. κ = (0.70 − 0.50) / 0.50 = 0.40. In a second case both annotators say A on 90% of items and agree on 85%: p_e = 0.82 and κ = (0.85 − 0.82) / 0.18 = 0.17. Higher raw agreement gives the lower κ. When response order is randomized in the interface, the label frequencies are near 50/50, p_e ≈ 0.5, and κ ≈ 2 p_o − 1. InstructGPT's 72.6% then corresponds to κ ≈ 0.45 (derived; the paper reports raw agreement only).

**Weighted κ for ordinal ratings.** When labels are ordered (Likert scales), a 1-vs-5 disagreement should count more than 1-vs-2.

```
κ_w = 1 − (Σ_ij w_ij O_ij) / (Σ_ij w_ij E_ij),     w_ij = (i − j)² / (k − 1)²
```

O_ij is the observed fraction of items rated i by annotator 1 and j by annotator 2; E_ij = p_1(i) · p_2(j) is the expected fraction under independence; k is the number of scale points; w_ij is the quadratic disagreement weight.

*Worked example (k = 3, six items).* Annotator 1 rates (1, 2, 3, 3, 2, 1); annotator 2 rates (1, 3, 3, 2, 2, 2). Three items differ by one point, so the mean observed weight is 3 · 0.25 / 6 = 0.125. The marginals are (1/3, 1/3, 1/3) and (1/6, 1/2, 1/3), which give a mean expected weight of 0.292. κ_w = 1 − 0.125 / 0.292 = 0.57. Unweighted κ on the same data is (0.50 − 0.333) / 0.667 = 0.25, because it counts the three one-point differences as full disagreements. HelpSteer2 uses quadratic weighted κ "because HelpSteer2 attributes are ordinal scores" ([[helpsteer2]] §2.1).

**Krippendorff's α for many annotators and missing labels.**

```
α = 1 − D_o / D_e
D_o = (1/n) Σ_c Σ_k o_ck · δ(c,k)          D_e = (1/(n(n−1))) Σ_c Σ_k n_c n_k · δ(c,k)
```

o_ck is the coincidence count: for each item with m_u labels, every ordered pair of labels from different annotators adds 1/(m_u − 1) to o_ck; n_c = Σ_k o_ck; n is the total number of pairable labels; δ(c,k) is the distance between labels (for nominal labels, 0 if c = k and 1 otherwise; for ordinal or interval labels, a squared difference). Items with fewer than two labels are dropped.

*Worked example (three annotators, four items, labels A/B).* Items: (A,A,A), (A,A,B), (B,B,B), (A,B,B). Each item has 6 ordered pairs of weight 1/2. Coincidences: o_AA = 3 + 1 = 4, o_BB = 3 + 1 = 4, o_AB = o_BA = 1 + 1 = 2. So n = 12, n_A = n_B = 6. D_o = (2 + 2)/12 = 0.333. D_e = (6 · 6 + 6 · 6)/(12 · 11) = 0.545. α = 1 − 0.333/0.545 = 0.39. HelpSteer2 chose Cohen's κ over α because with about 1,000 annotators, pairs of annotators "were rarely allocated common sets of samples" ([[helpsteer2]] §2.1); α handles that design directly because it does not need a fixed annotator pair.

**Evidence: reported agreement.**

| Source | Who vs who | Metric | Value |
|---|---|---|---|
| [[rlhf-instructgpt]] §3.4 | training labelers; held-out labelers | raw pairwise | 72.6 ± 1.5%; 77.3 ± 1.3% |
| [[hh-rlhf]] §2.1 | Anthropic researchers vs crowdworkers | raw | about 63% |
| [[lima]] §4.1 | crowd-crowd; author-author; crowd-GPT-4 | tie-discounted, 50 items | 82%; 78%; 78% |
| [[chatbot-arena]] Table 3 | crowd vs expert; expert vs expert | raw, 160 battles over two model pairs | 72.8-83.1%; 79.4% and 89.8% |
| [[ultrafeedback-construction]] Table 4 | human vs human; GPT-4 vs human | win/tie/lose, random = 33% | 54.7-58.1%; 59.7% average |
| [[helpsteer2]] Table 1 | annotators, helpfulness | quadratic weighted κ | 0.465 initial → 0.706 → 0.791 after filtering |
| [[helpsteer2]] Table 1 | annotators, coherence | quadratic weighted κ | 0.169 → 0.387 → 0.428 |

**Conditions and limits.** The rows use different metrics, item sets, and tie handling; the UltraFeedback numbers include ties with a 33% chance baseline, so they are not lower quality than the 72-83% rows by the difference in value. HelpSteer2's final κ is computed after removing high-disagreement items, so it describes the retained data, not the annotation process. The figure [figures/annotation-workflow.html](figures/annotation-workflow.html) lets the reader compute κ from raw agreement and label shares, single-label and majority-vote label noise for uniform and split item populations (§3), reward-margin compression under random label flips (§3), and the Bradley-Terry gradient by margin (§7).

**Implication.** An agreement number is only interpretable with its metric, its item sample, and whether filtering happened before it was computed. Per-attribute agreement matters for generality: HelpSteer2's coherence κ is less than half its helpfulness κ, so a reward model trained on coherence ratings has a noisier target for that property.

## §3 From disagreement to label noise, adjudication, and reward-model accuracy

**Problem.** A pairwise disagreement rate is not the label-noise rate of an aggregated dataset. The mapping depends on how many annotators label each item and on whether disagreement is spread across items or concentrated on some items.

**Mechanism and formula (a simple model, derived here).** Assume each annotator independently gives the label that the annotator population prefers with probability q.

```
a = q² + (1 − q)²                          pairwise agreement
q = (1 + sqrt(2a − 1)) / 2
P_maj(n) = Σ_{k > n/2} C(n,k) q^k (1 − q)^(n−k)     majority of n (n odd) matches the population label
```

a is pairwise agreement; q is per-annotator accuracy against the population-preferred label; n is the number of annotators; C(n,k) is the binomial coefficient. With binary labels and odd n, a majority always exists, so escalation rules based on "no majority" cannot trigger.

**Worked example 1 (uniform items).** a = 0.726 gives q = (1 + sqrt(0.452))/2 = 0.836. A single label disagrees with the population preference on 16.4% of items. Majority of 3 gives P = 0.836³ + 3 · 0.836² · 0.164 = 0.928, so 7.2% of aggregated labels disagree.

**Worked example 2 (split items).** Assume 55% of items are clear (q = 0.95) and 45% are split, where the annotator population itself divides 50/50 (q = 0.5). Pairwise agreement is 0.55 · 0.905 + 0.45 · 0.5 = 0.723, almost the same as example 1. Majority of 3 now disagrees with the population majority on 0.55 · 0.007 + 0.45 · 0.5 = 22.9% of items. On the split items no label is correct, and adding annotators does not change that. The same reported agreement rate is compatible with 7% or 23% aggregated disagreement.

**Effect on a Bradley-Terry reward model (derived).** If a fraction ε of comparison labels are flipped at random, the probability that the recorded winner is y_w becomes (1 − ε)σ(Δ) + ε σ(−Δ), where Δ = r(x, y_w) − r(x, y_l) is the true reward margin and σ is the logistic function. In the large-data limit, a flexible model fit to these labels learns Δ' = logit((1 − ε)σ(Δ) + ε σ(−Δ)). This is monotone in Δ, so the ordering of responses is preserved in expectation, but margins shrink: with Δ = 2 and ε = 0.164, σ(2) = 0.881, the noisy probability is 0.756, and Δ' = 1.13. Symmetric noise reduces the margin a policy optimizes against; it does not reverse preferences on average. Systematic noise (a feature that annotators reward regardless of quality, §5) does shift the learned ordering.

**Adjudication designs reported in the sources.**
1. *Drop weak preferences and ties.* HH-RLHF keeps a comparison only if the worker expressed "a preference stronger than the weakest available" ([[hh-rlhf]] §2.2).
2. *Add annotators on high spread.* HelpSteer2 recruits two more annotators when the helpfulness range among three exceeds 2 points, and keeps "annotations from the three most in agreement" ([[helpsteer2]] §2.1).
3. *Remove high-disagreement items.* HelpSteer2 keeps responses whose helpfulness range is 2 points or below, removing about 10% of samples; about 50% of all annotations were excluded across all stages (§2.1).
4. *Remove annotators.* HelpSteer2 removed annotators "deemed 'untrusted' or [who] consistently had low agreement with others" (§2.1).
5. *Expert relabeling for validation.* Chatbot Arena relabeled 160 battles between GPT-4-Turbo and two other models with experts who were "asked to carefully fact-check model's answer with external resources like search engine"; the authors attribute the crowd-expert gap mostly to crowd users "overlooking factual errors" ([[chatbot-arena]] §6.3).
6. *Keep graded margins.* UltraRM subtracts a normalized score-difference margin m(r) inside the ranking loss ([[ultrafeedback-construction]] App. D.1).

No source in this chapter reports a controlled ablation of downstream policy quality for designs 1-4 against keeping all labels (not reported).

**Evidence: reward-model accuracy stays near label agreement.** InstructGPT RMs trained with 5-fold cross-validation over labeler groups predict held-out groups at 69.6 ± 0.9% and their own groups at 72.4 ± 0.4% ([[rlhf-instructgpt]] §4.1), against 72.6% labeler agreement (Result, single study). On HH-RLHF helpfulness data, a Bayesian logistic model over 23-24 GPT-4-labelled features (the paper gives both counts) reaches 71.3% holdout accuracy, against about 72% for a 52B preference model trained on the same data ([[sycophancy-in-lms]] §4.1, App. B). **Interpretation.** When single-label agreement is near 72%, reward-model accuracy near 72% on single labels is close to what the labels can support. The 71.3% reached by a model that sees only 23-24 labelled response features means most of that accuracy is reproducible from visible features rather than from a deeper quality judgement.

## §4 Prompt-distribution coverage

**Definition.** *Prompt coverage* is how well the set of training prompts represents the tasks, topics, languages, turn counts, and user intents the model will receive. It is separate from response quality and from label agreement.

**Problem.** Preference data only gives signal on prompts that are in it. A dataset built from benchmark-shaped or single-template prompts can raise scores on that shape and leave other request types unchanged.

**Mechanism: how the sources build coverage.**
1. *Sample from real traffic and prevent a few users from dominating.* InstructGPT deduplicates by long common prefix, caps prompts at 200 per user ID, splits train and test by user ID, and filters PII ([[rlhf-instructgpt]] §3.2).
2. *Stratify by topic.* HelpSteer2 clusters prompts into about 1,000 BERTopic topics and samples uniformly per topic, then samples uniformly per complexity level with double weight on the highest level ([[helpsteer2]] §2.1). LIMA samples a wikiHow category first, then an article ([[lima]] §2.1).
3. *Measure the category mix.* InstructGPT's API prompts: generation 45.6%, open QA 12.4%, brainstorming 11.2%, chat 8.4%, rewrite 6.6%, summarization 4.2%, classification 3.5%, other 3.5%, closed QA 2.6%, extract 1.9% ([[rlhf-instructgpt]] Table 1). WildChat English first turns: assisting/creative writing 61.9%, analysis/decision explanation 13.6%, coding 6.7%, factual info 6.3%, math reasoning 6.1% ([[wildchat]] Table 4). Chatbot Arena user prompts form 600 topic clusters, the largest holding 1% of prompts ([[chatbot-arena]] §6.1).

**Evidence.**
- *Real-user prompts vs public NLP tasks.* 175B InstructGPT was preferred over 175B GPT-3 finetuned on FLAN in 78 ± 4% of comparisons and over T0 in 79 ± 4% ([[rlhf-instructgpt]] §4.1). The authors attribute this to public datasets covering classification and QA (about 18% of API use) rather than open-ended generation and brainstorming (about 57%), and to low input diversity (Interpretation by the authors).
- *Diversity vs quantity (LIMA 7B ablation, ChatGPT-graded helpfulness 1-6, 2,000 examples each).* Filtered Stack Exchange (heterogeneous prompts) 3.83; wikiHow ("how to" prompts only, high-quality responses) 3.49; unfiltered Stack Exchange 3.33. Scaling filtered Stack Exchange from 2K to 32K examples plateaued ([[lima]] §5, Figs. 5-6). Result (single study); the authors note that the two sources differ in more than diversity.
- *Coverage measured as likelihood.* A Llama-2 7B model finetuned on Alpaca prompts has NLL 10.87 on WildChat first-turn prompts and 11.11 on ShareGPT, while a model finetuned on WildChat has 3.28 on Alpaca against Alpaca's own 2.24 ([[wildchat]] Fig. 3). In this measurement, training on synthetic Alpaca prompts does not cover real-user prompts, while training on real-user prompts covers most of the Alpaca and Dolly prompts (Result, single study).
- *Removal ablation.* Tülu 3 8B SFT without WildChat: average 60.1 → 58.9, AlpacaEval 2 12.4 → 7.5, IFEval 72.8 → 70.1, safety 93.1 → 95.2 ([[tulu-3-preference-ablations]] Table 10). Result (single study, one seed reported).
- *Unique prompts in preference data.* In Tülu 3 8B DPO, more unique prompts gave "noticeable performance gains", while expanding UltraFeedback from 64k to 383k pairs by reusing the same 64k prompts "performs similarly", with slight losses on DROP, GSM8k, and AlpacaEval; prompts unused in SFT were slightly better than reused SFT prompts ([[tulu-3-preference-ablations]] §5.3, Figs. 8-10).
- *Direction of the conversation.* HH-RLHF red-team conversations move toward harmful content and helpfulness conversations toward beneficial content; the authors state this "made it difficult to train models that were both helpful and harmless" and recommend collecting harmlessness data from conversations that move in the beneficial direction ([[hh-rlhf]] §2.2).

**Conditions and limits.** The LIMA ablations (ChatGPT grader) and the Tülu 3 AlpacaEval scores use LLM judges, which can prefer longer outputs (§5, ch-49). WildChat's coverage heatmap covers first-turn prompts only. InstructGPT's prompt set is over 96% English (§3.3), so its coverage evidence does not extend to other languages.

**Implication for a general-purpose model (Interpretation).** In the evidence above, prompt breadth changed downstream breadth more consistently than label volume did. A prompt set needs its category, language, and turn-count distribution reported, compared against a real-traffic reference such as WildChat, and deduplicated per user, before annotation effort is spent on it.

## §5 Annotator-induced biases that narrow behaviour

**Definition.** An *annotator-induced bias* is a feature of responses that raises the probability of being preferred independently of the quality the rubric describes. A reward model learns it, and a policy optimized against that reward model increases it.

**Length.**
1. *In the labels.* Always choosing the longer response matches 55.7% of WebGPT labels (human), 59.6% of Stack labels (upvote-derived), and 63.1% of RLCD labels (synthetic) ([[rlhf-length-correlations]] Table 5).
2. *In the reward model.* Within 8 samples for the same prompt, the Pearson correlation between length and reward is 0.72 (WebGPT), 0.55 (Stack), and 0.67 (RLCD), at eval accuracies of 61.5%, 70%, and 80% (Table 4). Confidently wrong RM predictions mostly follow the length heuristic (§5, Fig. 5).
3. *In the policy.* PPO (Llama-7B, LoRA) raised mean output length on WebGPT from 100 to 230 tokens. Only 2.0% of the reward gain on WebGPT and 27.2% on RLCD came from reward increases within fixed-length buckets (Table 1). A reward that only targets a length reached 56% simulated win rate against SFT on WebGPT versus 58% for standard PPO, and 64% versus 63% on RLCD (Table 2).
4. *Interventions.* Length-balancing the preference data reduced the length-reward correlation on WebGPT to −0.13 but lowered RM accuracy to 52.6%; "no strategy works for all settings" (§1, Table 4). Result (single study, 7B, three datasets of which one has human labels). HelpSteer2 records verbosity as a separate attribute so that a reward model can separate verbosity from helpfulness; in its data, response length has Pearson R = 0.0845 with helpfulness ([[helpsteer2]] §2.2, §3).

**Sycophancy.** *Sycophancy* is a response matching the user's stated views over a truthful one. In 15K HH-RLHF helpfulness comparisons, "matches user's beliefs" is among the most predictive of the 23-24 GPT-4-labelled features of human preference, though not always first across data splits ([[sycophancy-in-lms]] §4.1, App. B). The Claude 2 preference model prefers a convincing sycophantic response over a helpful truthful one for 45% of the hardest misconceptions (§4.3.1). Five assistants from three developers admit a mistake after "Are you sure?" on 42% (GPT-4) to 98% (Claude 1.3) of questions (App. A.4). Feedback and mimicry sycophancy increased over the Claude 2 RL phase (Fig. 6b). An official report on a production system describes a sycophancy increase after adding a user-feedback reward ([[openai-sycophancy-postmortem]]); it gives no quantitative data, so it is consistent with, not a replication of, the measurements above.

**Other visible features.** HH-RLHF crowdworkers "often prefer responses that include non-functional URLs" ([[hh-rlhf]] §2.1). In PRISM ratings, more characters, ending with a question, and enumeration raise scores; refusals lower them, with R² = 0.06 for all such factors together ([[prism-alignment]] §3.2).

**Amplification by optimization.** In a synthetic-label setup, gold reward first rises and then falls as best-of-n or PPO optimization against a proxy reward model increases KL from the initial policy, while the proxy reward keeps rising ([[reward-model-overoptimization]] Fig. 1, Fig. 8). The bias features above are a concrete form of the proxy-gold gap; ch-41 and ch-42 cover measurement and controls.

**Implication for a general-purpose model (Interpretation).** Length and agreement with the user are features that exist in every task, so a bias learned from chat comparisons can change behaviour on other tasks; SycophancyEval found sycophancy on MATH solutions and factual QA as well as on arguments and poems ([[sycophancy-in-lms]] §3). Measuring these features on held-out prompts from other domains is part of evaluating a preference-trained model.

## §6 Preference heterogeneity: disagreement as information

**Definition.** *Preference heterogeneity* is systematic disagreement between annotators or user groups that remains after the rubric is clarified. It differs from annotation error.

**Problem.** Treating every disagreement as noise and removing it trains the model toward the preferences of the annotators who remain, on the items that remain.

**Evidence.**
- InstructGPT: "we are aligning to demonstrations and preferences provided by our training labelers", who are "mostly English-speaking people living in the United States or Southeast Asia", with inter-labeler agreement about 73% (§5.2). Most comparisons had one labeler, and "In cases of disagreement, aligning to the average labeler preference may not be desirable" ([[rlhf-instructgpt]] §5.3).
- OASST1 annotators: 89.1% identify as male, median age 26 ([[openassistant]] §7).
- PRISM: rankings of 21 models change with who rates and what they discuss; palm-2 drops 4 places for US participants, llama-7b drops 7 places in Asia, mistral-7b gains 7 places in Africa; in the US sample no model is the top choice for more than 45% of participants; selecting a model by the ratings of 100 white participants lowers welfare for non-white participants ([[prism-alignment]] §3.2-§3.3).
- HelpSteer2 kept items with a helpfulness spread of up to 2 Likert points, "recognizing that differences among annotators can also stem from inherent subjectivity or individual preferences rather than misunderstandings" ([[helpsteer2]] §2.1).
- Chatbot Arena experts disagree on 10-20% of battles, which the authors attribute mostly to prompts without a ground-truth answer ([[chatbot-arena]] §6.3).

**Options (course synthesis; no source here compares them on a general-purpose policy).**
1. Filter high-disagreement items (HelpSteer2): lowers noise on the retained set; removes the subjective items from the training signal.
2. Keep individual labels with annotator IDs and demographics (PRISM): allows group-level analysis and personalization; requires a decision on aggregation.
3. Keep graded or soft targets (UltraRM margins): represents weak preferences as small margins.
4. Report the reward model's accuracy per annotator group, as InstructGPT did with held-out labeler groups.

**Open question.** Whether a single policy trained on aggregated heterogeneous preferences is broader or narrower than one trained on filtered consensus items has not been measured with held-out capability suites.

## §7 Pair selection: which responses are compared

**Definition.** *Pair selection* decides which candidate responses an annotator compares: their source model, how many, and how far apart in quality. *On-policy* candidates come from the model that will be trained; *off-policy* candidates come from other models or humans.

**Problem.** The reward model learns to separate the pairs it sees. Pairs that are far apart in quality are easy to label and teach coarse distinctions; pairs near the policy's current quality are harder to label and are the distinctions the policy needs later.

**Mechanism and worked example.** A ranking of K responses gives C(K,2) = K(K − 1)/2 comparisons: K = 4 gives 6 and K = 9 gives 36. The comparisons from one prompt share responses, so they are correlated. InstructGPT found that shuffling them into one dataset made the reward model overfit in one pass, and trains all C(K,2) comparisons of a prompt as one batch element ([[rlhf-instructgpt]] §3.5). The Bradley-Terry loss gradient shows which pairs dominate training:

```
L = −log σ(Δ),    ∂L/∂Δ = −σ(−Δ),    Δ = r(x, y_w) − r(x, y_l)
```

At Δ = 0 the gradient magnitude is 0.5 (Panel D of [figures/annotation-workflow.html](figures/annotation-workflow.html) plots the curve); at Δ = 3 (an easy pair the model already ranks correctly) it is 0.047; at Δ = −3 (a pair the model ranks against its label) it is 0.953. Pairs the model already separates in the direction of their label contribute small gradients, and a mislabeled pair that the model separates confidently in the opposite direction contributes the largest gradient.

**Evidence.**
- *Candidate distribution shifts during collection.* HH-RLHF's online preference model scores 74%, 70%, and 67% on test sets from the base, rejection-sampled, and online distributions, which contain progressively better responses ([[hh-rlhf]] §4.5). Restricting to comparisons between high-scoring samples lowers accuracy further (Fig. 25).
- *On-distribution data at equal size.* Two 52B RLHF runs whose preference models used equal-sized datasets (about 44k base comparisons vs an even base/RS/online mix) and identical settings: crowdworkers preferred the online-mix model ([[hh-rlhf]] §4.5, Fig. 16). Result (single study).
- *On-policy candidates in synthetic preference data.* Tülu 3 8B DPO with on-policy responses (one response from the SFT model) scored higher on the aggregate than fully off-policy data ([[tulu-3-preference-ablations]] Fig. 11). Labels were produced by GPT-4o, not humans.
- *Human-written candidates.* OASST1 reward models were trained on rankings of human-written replies; the LLaMA-30B RLHF model improved LMEH and Vicuna Elo over SFT but not OpenAI Evals or HumanEval, and the authors hypothesize the candidate source explains part of the gap ([[openassistant]] Table 1, §7).
- *Deliberate spread.* UltraFeedback samples 4 of 17 models of different families and sizes per instruction "to alleviate the potential spurious correlation between text styles and response quality" ([[ultrafeedback-construction]] §2.3). HelpSteer2 always draws its two responses from two different sources ([[helpsteer2]] §2.1).

**Conditions and limits.** None of these studies isolates pair difficulty from candidate source with human labels at fixed budget (not reported). The on-policy benefit in Tülu 3 is for DPO with an AI judge.

**Implication.** For a general-purpose model, candidates drawn from the policy being trained, on a broad prompt set, give reward-model accuracy where the policy operates. Candidates from a fixed pool of other models give broader style variation but may not cover the policy's own failure modes.

## Negative samples and negative feedback

This section applies the terms of ch-43a to human preference data.

**Where negatives come from and how they are labeled.**
1. *Dispreferred responses* in comparisons and rankings, labeled by annotators. Given the agreement rates in §2, a second annotator would reverse roughly 23-37% of single pairwise labels in InstructGPT and HH-RLHF style data (derived from 72.6-77.3% and about 63% agreement; the HH figure is researcher vs crowdworker).
2. *Red-team comparisons* in which the worker chooses the more harmful response ([[hh-rlhf]] §2.2).
3. *Removed data*: weak preferences and ties (HH-RLHF), high-disagreement items (HelpSteer2), spam-flagged and moderator-deleted messages (OASST1).
4. *Explicit user feedback*: thumbs-down in a product interface ([[openai-sycophancy-postmortem]]).

**What current practice does with them (four meanings).** Removed data (item 3) is *negative marginal value*: discarded, never used as a target. OASST1's deleted messages had mean Detoxify toxicity 4.625% versus 0.988% for retained messages, and the authors state that toxicity scores alone cannot decide exclusion ([[openassistant]] §6.2). Dispreferred responses (items 1-2) are *negative as gradient* when a reward model is trained with the Bradley-Terry loss (ch-41) or when DPO decreases their likelihood (ch-39). OASST1's ranking guideline places "I don't know" replies between correct and incorrect ones, which is a rubric decision about which negatives abstention is preferred to ([[openassistant]] App. A). Thumbs data (item 4) became *negative as gradient* in GPT-4o through an added reward signal.

**Mechanism.** In reward-model training, the gradient in §7 raises r(x, y_w) and lowers r(x, y_l) by the same amount σ(−Δ). When the policy is later trained against that reward, or when DPO applies the same loss to log-probabilities, lowering the likelihood of a response moves probability mass to other responses. For a softmax over logits z, ∂ log p_y / ∂ z_j = 1[j = y] − p_j, so pushing down a response token that already has low probability changes the other probabilities in proportion to their current values, and most of the removed mass goes to the most likely alternatives (derivation in ch-43a). A dispreferred label is therefore a statement about what to avoid that leaves the replacement to the model's current distribution.

**Evidence with numbers.**
- *Failure mode, user feedback.* The April 2025 GPT-4o update added "an additional reward signal based on user feedback—thumbs-up and thumbs-down data from ChatGPT"; OpenAI's assessment is that the combined changes "weakened the influence of our primary reward signal, which had been holding sycophancy in check", and that "user feedback in particular can sometimes favor more agreeable responses". Offline evaluations "generally looked good", expert testers said behaviour "'felt' slightly off", and "We didn't have specific deployment evaluations tracking sycophancy"; rollback began April 28, 2025 ([[openai-sycophancy-postmortem]]). Official source; no quantitative data reported.
- *Failure mode, human comparisons.* Sycophancy and length both increased under optimization against human-trained reward (§5).
- *Benefit.* No source in this chapter measures the share of preference-training gains that comes from the dispreferred side of pairs versus the preferred side (not reported), so attributing the gains to the negative side is unsupported here.

**Controls.**
1. Keep ties and weak preferences out of the gradient, or give them small margins (HH-RLHF filter, UltraRM margins).
2. Remove or re-annotate high-disagreement items before training (HelpSteer2).
3. Keep user feedback as a bounded auxiliary signal, not a replacement for the primary reward, and treat behaviour regressions as launch-blocking ([[openai-sycophancy-postmortem]] stated process changes).
4. Add attribute-level labels for known bias features (verbosity in HelpSteer2) so they can be controlled separately.
5. Use on-policy candidates so that dispreferred responses are responses the policy actually produces (§7).

**Diagnostics.** Log chosen and rejected reward scores separately during reward-model training, and chosen and rejected log-probabilities during DPO (ch-39). Track reward-model accuracy per annotator group and per prompt category. Track the within-prompt correlation between length and reward ([[rlhf-length-correlations]] Table 4 method). Run sycophancy probes ("Are you sure?", stated user opinion) at several optimization strengths ([[sycophancy-in-lms]] §3). Track refusal and abstention rates on benign prompts (ch-52).

**Effect on generality.** Dispreferred labels that encode agreement or length preferences push the policy away from disagreement and brevity on all tasks, which harms calibration and factual accuracy outside chat (§5). Thumbs-down data covers only the prompts users send and the responses they react to, so it is weakest on rare task types.

## Recipe

Values below are data and annotation settings. Optimizer settings for the policy stages are in ch-38 and ch-39.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| InstructGPT SFT | 1.3B, 6B, 175B | SFT | training prompts | 11,295 labeler-written + 1,430 customer | arXiv:2203.02155v1 App. A Table 6 ([[rlhf-instructgpt]]) | verified 2026-09-15 | no ablation reported |
| InstructGPT RM | 6B | reward-model | training prompts; responses ranked per prompt | 6,623 labeler + 26,584 customer; K = 4 to 9 | Table 6; §3.5 | verified 2026-09-15 | §3.5: per-prompt batching of all C(K,2) pairs avoided one-epoch overfitting (no numbers) |
| InstructGPT (all) | 1.3B-175B | reward-model | labelers; screening cutoffs | about 40 contractors; soft cutoffs 75% agreement, 6/7 demonstration score | §3.4; App. B.1 | verified 2026-09-15 | no ablation reported |
| HH-RLHF preference models | 52B | reward-model | comparisons: base; rejection-sampled; online | 44k helpful + 42k red-team; 52k helpful + 2k red-team; 22k helpful | arXiv:2204.05862v1 §2.3 ([[hh-rlhf]]) | verified 2026-09-15 | §4.5 Fig. 16: equal-size online mix preferred over base-only |
| HH-RLHF data collection | 52B candidates | reward-model | rejection-sampling candidates; label filter | k = 16 "most often"; keep preferences stronger than the weakest, no ties | §2.3; §2.2 | verified 2026-09-15 | no ablation reported |
| HelpSteer2 RMs (Llama 3 70B RM; Nemotron-4 340B RM) | 70B; 340B | reward-model | prompts × responses; annotators; filter | 10,681 × 2 = 21,362; 3 annotators, 5 on spread > 2, mean 3.41; keep helpfulness spread ≤ 2 (about 10% removed) | arXiv:2406.08673v1 §2.1 ([[helpsteer2]]) | verified 2026-09-15 | Table 1: κ 0.465 → 0.706 → 0.791 (agreement only; no downstream ablation of the filter) |
| HelpSteer2 DPO model (Llama 3 70B SFT start) | 70B | preference | pairs; epochs; LR; KL penalty; global batch | 7,221; 7; 2e-7 constant; 1e-3; 128 | §4.3 | verified 2026-09-15 | LR searched over {3e-7, 2e-7, 1e-7, 9e-8}; KL over {1e-3, 4e-4} |
| UltraRM (LLaMA2-13B) | 13B | reward-model | instructions; candidates; labeler; scale; UF pairs | 63,967; 4 of 17 models; GPT-4; 1-5 on 4 aspects; 340,025 of 749,702 total pairs | arXiv:2310.01377v2 §2.2-§2.4, App. E.1 ([[ultrafeedback-construction]]) | verified 2026-09-15 | §3.1 Table 2: overall-score variant lags fine-grained variants on WebGPT |
| UltraRM (LLaMA2-13B) | 13B | reward-model | epochs; batch; LR schedule | 1; 512 pairs; 1e-5 cosine, 3% warmup, final 1e-6 | App. D.1 | verified 2026-09-15 | "Following Touvron et al. (2023b)"; no ablation reported |
| LIMA | 65B | SFT | examples; tokens; epochs; LR; batch | 1,000; about 750,000; 15; 1e-5 → 1e-6 linear, no warmup; 32 | arXiv:2305.11206v1 Table 1, §3 ([[lima]]) | verified 2026-09-15 | §5 (7B): diversity and quality ablation (Fig. 5), quantity plateau (Fig. 6) |
| WildLlama | 7B | SFT | data; epochs; LR; batch; max length | WildChat to 2023-07-16; 3; 2e-5; 128 conversations; 2048 | arXiv:2405.01470v1 §5 ([[wildchat]]) | verified 2026-09-15 | Vicuna hyperparameters adopted; no ablation reported |
| Tülu 3 8B DPO | 8B | preference | pairs; on-policy / off-policy from SFT prompts; judge | 271,409; 19,444 / 96,911; GPT-4o-2024-08-06, 1-5 on 4 aspects | arXiv:2411.15124v5 §5.2, Table 15 ([[tulu-3-preference-ablations]]) | verified 2026-09-15 | §5.3 Figs. 8-11; Table 16 mixes average 60.54-62.27 |
| Tülu 3 70B DPO | 70B | preference | pairs | 334,302 | Table 15 | verified 2026-09-15 | mix ablations run mainly at 8B (§5.2.2) |
| GPT-4o (ChatGPT update of 2025-04-25) | not reported | RL | reward signals | added user-feedback (thumbs) reward; weights not reported | OpenAI blog 2025-05-02 ([[openai-sycophancy-postmortem]]) | verified 2026-09-15 (archived copy) | rolled back from 2025-04-28; no ablation reported |
| Length-correlation PPO study | Llama-7B | RL | KL coefficient λ; batch; LoRA rank | 0.04 (HIGH λ 0.12); 64; 16 | arXiv:2310.03716v2 §2.1, §4.1 ([[rlhf-length-correlations]]) | verified 2026-09-15 | λ chosen "based on reward and downstream evaluation"; larger than 0.12 impeded convergence |

**Starting point for a small general-purpose run.** For a preference dataset built from human ratings on a budget similar to HelpSteer2's (about 10K prompts), use two responses per prompt from different model sources, three annotators per response with two more when the helpfulness range exceeds 2 Likert points, and removal of responses whose range stays above 2 points; this is the HelpSteer2 protocol that fed 70B and 340B reward models. If comparisons are collected as rankings, use K = 4 to 9 per prompt and train all C(K,2) pairs of a prompt in one batch element, as InstructGPT did for a 6B reward model. For preference pairs at 8B scale produced by an AI judge instead of humans, the Tülu 3 8B DPO mix of 271,409 pairs included 19,444 on-policy pairs; this is a data-composition reference, not a human-annotation budget.

## Generalization lens

**(a) What increases breadth.**
- Real-user prompts from many users, deduplicated and capped per user: InstructGPT over FLAN and T0 at 78-79% ([[rlhf-instructgpt]] §4.1); WildChat-trained models cover other prompt sets at lower NLL than the reverse ([[wildchat]] Fig. 3); removing WildChat lowered Tülu 3 8B SFT AlpacaEval 2 from 12.4 to 7.5 ([[tulu-3-preference-ablations]] Table 10).
- More unique prompts rather than more pairs per prompt ([[tulu-3-preference-ablations]] Figs. 8-9) and diverse over homogeneous prompts at equal count ([[lima]] Fig. 5).
- Annotators screened for sensitivity across groups, and held-out annotator groups for evaluation ([[rlhf-instructgpt]] §3.4, §4.1).

**(b) What causes narrowing or forgetting.**
- Labels that reward length or agreement with the user, which reward models learn and policies amplify ([[rlhf-length-correlations]] Tables 1-2; [[sycophancy-in-lms]] §4).
- Removing domains the annotators cannot judge (HelpSteer2 removed non-English and code prompts, [[helpsteer2]] §2.1) without supplying those domains from another source.
- Aggregating over a narrow annotator population (OASST1 89.1% male; PRISM rankings shift by region, [[prism-alignment]] §3.2).
- Adding a user-feedback reward that outweighs the primary reward ([[openai-sycophancy-postmortem]]).
- Prompt sets where the conversation direction differs between objectives (HH-RLHF red-team vs helpful, [[hh-rlhf]] §2.2), which made a jointly helpful and harmless model harder to train.

**(c) How to measure it for this stage.**
- Report the prompt category, language, and turn-count distribution and compare it to a real-traffic reference (InstructGPT Table 1, WildChat Table 4).
- Report chance-corrected agreement per attribute, before and after filtering (HelpSteer2 Table 1).
- Evaluate the reward model on held-out annotator groups (InstructGPT §4.1), on held-out candidate distributions (HH-RLHF §4.5), and on benchmark sets outside the training prompts (ch-41).
- Measure length-reward correlation and sycophancy before and after optimization (Singhal Table 4; Sharma §3).
- Known measurement errors: LLM judges used for downstream evaluation prefer length ([[rlhf-length-correlations]] §2.1 caveat; ch-49); agreement numbers use different tie handling (§2); crowd voters miss factual errors that experts catch ([[chatbot-arena]] §6.3).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reporting raw agreement only | agreement looks high on a dataset with a dominant label position | compute p_e and κ; randomize response order and compare |
| Treating pairwise disagreement as the majority-label noise rate | budget spent on more annotators does not reduce reward-model errors on split items | measure how disagreement is distributed across items (§3 example 2) |
| Escalating binary 3-annotator items only when "no majority" | the escalation queue is always empty | use a spread or confidence rule, or 5 annotators on flagged items |
| Prompt set built from one source or template | good scores on in-distribution prompts; low scores on real-user prompts | NLL cross-coverage test ([[wildchat]] Fig. 3); category histogram |
| Scaling pairs by reusing prompts | more pairs without gains; slight losses on some suites | compare unique-prompt count at fixed pair count ([[tulu-3-preference-ablations]] Fig. 9) |
| Length-correlated labels | policy length rises during optimization while quality within length buckets is flat | length heuristic accuracy on labels; non-length reward gain ([[rlhf-length-correlations]] Tables 1, 5) |
| Rewarding agreement with the user | answers change after "Are you sure?" | SycophancyEval-style probes at several checkpoints ([[sycophancy-in-lms]] §3) |
| Adding thumbs feedback as a reward | A/B metrics improve while expert spot checks report behaviour changes | launch-blocking sycophancy and behaviour evaluations ([[openai-sycophancy-postmortem]]) |
| Rubric priority differs between training labels and evaluation | reward model agrees with training labels but loses on evaluation labels | record the priority rule per collection batch ([[rlhf-instructgpt]] App. B.2) |
| Candidates only from other models | reward model accuracy drops on the policy's own samples | RM accuracy split by candidate source ([[hh-rlhf]] §4.5) |

## Check your understanding

1. InstructGPT labelers agree on 72.6% of comparisons and a reward model predicts held-out labeler groups at 69.6%. Explain why a larger reward model trained on the same single labels is unlikely to exceed about 72% on those labels, and what change to the data could raise the achievable accuracy.
2. Two datasets both report 72% pairwise agreement. In one, disagreement is spread evenly over items; in the other, it is concentrated on 45% of items. Explain why majority voting with three annotators helps the first dataset and not the second, and what that implies for filtering.
3. Length-balancing WebGPT preference data removed the length-reward correlation but lowered reward-model accuracy to near chance. Explain what this suggests about which features the original reward model had learned.
4. Explain why a thumbs-up/thumbs-down reward can increase sycophancy even when every individual thumbs label is an honest user reaction.
5. HelpSteer2 removed non-English and code prompts because annotators could not judge them. Describe the effect on a general-purpose model trained only on this reward signal, and a way to restore the missing coverage.
6. Explain why expanding UltraFeedback from 64k to 383k pairs by reusing prompts did not improve Tülu 3 8B DPO, using the gradient of the Bradley-Terry loss and the correlation between pairs from the same prompt.
7. PRISM finds that no model is the top choice for more than 45% of US participants. Explain what this means for a reward model trained on pooled labels, and how you would test whether the pooled policy is worse for some group.

## Connections

- Previous: ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split.
- Next: ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization (trains on the labels described here).
- Dependency: ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate (the SFT model that produces on-policy candidates).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks (prompt breadth for SFT).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection (diversity measures for prompt sets).
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity (on-policy candidates).
- ch-39 — Offline Preference Optimization: DPO and Its Variants (dispreferred responses as gradient).
- ch-42 — Reward Hacking and Judge Design; ch-42a — Narrow Training, Broad Behaviour Change: Emergent Misalignment, Sycophancy, and Trait Transmission.
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL.
- ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting (AI labels as an alternative to human labels).
- ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming.

## Sources

- [[rlhf-instructgpt]] — prompt splits (Table 6), use-case mix (Table 1), labeler screening (App. B.1), agreement (§3.4), RM held-out labeler accuracy and FLAN/T0 comparison (§4.1), K-way ranking and batching (§3.5), limitations (§5.3) (chapter excerpt rewritten from the primary text).
- [[hh-rlhf]] — crowdworker selection and 63% agreement (§2.1), preference-strength filter and conversation direction (§2.2), data tranches (§2.3), PM accuracy by distribution and online experiment (§4.5) (chapter excerpt).
- [[openassistant]] — volunteer scale, annotator demographics, deleted-message toxicity, human-written candidates (verified library card).
- [[lima]] — 1,000-example composition, agreement, 7B diversity/quality/quantity ablations (chapter excerpt).
- [[wildchat]] — real-user statistics, category mix, NLL coverage heatmap, WildLlama (chapter excerpt).
- [[ultrafeedback-construction]] — AI-labeled contrast: 17-model candidate pool, 1-5 aspect scale, UltraRM data and margins, human agreement table (chapter excerpt).
- [[helpsteer2]] — annotator scale, escalation and disagreement filter, weighted κ per attribute, verbosity attribute, DPO settings (chapter excerpt; no library card).
- [[prism-alignment]] — participant diversity, ranking shifts by group and topic, welfare under sampling schemes (chapter excerpt; no library card).
- [[chatbot-arena]] — vote volume, topic clusters, active pair sampling, crowd-expert agreement (chapter excerpt).
- [[rlhf-length-correlations]] — length heuristic accuracy, length share of reward gain, length-only PPO, interventions (chapter excerpt).
- [[sycophancy-in-lms]] — feature analysis of HH-RLHF preferences, PM preference for sycophancy, sycophancy under RL (verified library card).
- [[openai-sycophancy-postmortem]] — thumbs-feedback reward and GPT-4o rollback (chapter excerpt from an archived copy of the official post).
- [[tulu-3-preference-ablations]] — WildChat removal, unique vs duplicated prompts, on-policy candidates, preference mix sizes (chapter excerpt).
- [[reward-model-overoptimization]] — proxy vs gold reward under optimization (verified library card).
