<!-- chapter: ch-19
     track: rl-pipeline
     kind: content
     title: RL Without a Clear Score: Rewards for Naturalness, Style, and Other Subjective Qualities
     deps: [ch-11, ch-18]
     sources: [[subjective-reward-guide]], [[bradley-terry-rm]], [[generative-reward-models]], [[rubrics-as-rewards]], [[rlcf-checklists]], [[rubicon-rubric-anchors]], [[writing-zero]], [[rlmt-think-chat]], [[skywork-reward-v2]], [[litbench]], [[judge-llm-bias]], [[kimi-k2-recipe]], [[constitutional-ai]], [[ultrafeedback]], [[deepseek-grm]], [[mai-thinking-1]], [[master-rm-one-token]], [[reward-model-overoptimization]], [[sycophancy-in-lms]], [[speechjudge]], [[speechalign]], [[antislop]], [[algorithm-crosswalk]], [[nemotron-ultra-recipe]], [[olmo-3-recipe]], [[ifbench]], [[dpo]], [[minimax-m1]]
     figures: figures/subjective-reward.html
-->

# Chapter 19 — RL Without a Clear Score: Rewards for Naturalness, Style, and Other Subjective Qualities

> **Core insight.** When no program can check an answer, the reward is a model of human judgments, and every RL step searches for the places where that model and the humans disagree. The measured cases in this chapter show the same sequence: a learned scalar reward is optimized, a cheap surface feature that the reward over-scores appears (length, self-justification, an engagement-bait closer, a single word), the feature spreads through the rollouts within tens of steps, and a public LLM-judged benchmark can still rank the hacked policy first ([[writing-zero]] Table 2–3; [[rlmt-think-chat]] Tables 8–9). The design work is therefore less about choosing one reward model and more about (1) building the preference data so that known biases are removed before fitting, (2) choosing a judge format whose errors are hard to exploit (pairwise against an in-group reference, rubric anchors, vetoes), (3) measuring the judge against humans on the policy's own outputs, and (4) monitoring named hacking features during the run and stopping at the peak of a held-out human-aligned score.
>
> **Guideline.** When the quality is subjective and you can collect human labels, collect pairwise labels with a preference-strength field and report RM accuracy per strength, because Llama 2's RM scored 79.1% on "significantly better" and 54.5% on "negligibly better" pairs ([[bradley-terry-rm]], Llama 2 Table 28). When labels come from a natural signal (votes, ratings, call outcomes), rebalance pair lengths before fitting, because 65.25% of raw chosen stories were longer and an RM trained without rebalancing saturated at 70% and preferred the longer story ([[litbench]] §3.2, Fig. 7). When the policy is trained online against a learned reward, compare each rollout pairwise with a random rollout from its own group and drop groups whose verdicts are nearly all one sign, because the scalar-RM baseline grew to 1,872 mean tokens with 417 tokens of self-explanation versus 1,292 and 58 for the pairwise GenRM ([[writing-zero]] Table 3). When one reward mixes constraint-following and style, train them in separate stages, because creativity-only training cost Collie −6.0 and IFEval −5.9 ([[rubicon-rubric-anchors]] §4.4). When choosing between online RL and DPO for a subjective domain, prefer online optimization against a calibrated judge if the judge can score fresh samples, because online DPO with a speech judge reached CMOS 0.25 against 0.16–0.21 for offline DPO variants ([[speechjudge]] Fig. 6a) and GRPO beat DPO on the same instruction-following data, IFEval 89.65 vs 79.67 ([[ifbench]] Table 5, via [[algorithm-crosswalk]]); prefer a token-level or offline pairwise method when the target is a narrow lexical pattern, because sequence-level DPO lost up to 26% lexical diversity where FTPO lost about 5% ([[antislop]] Table 2).

---

## Why this chapter matters

The earlier grader chapter (ch-11) treated judges as one rung of a ladder whose lower rungs are programs. For naturalness, style, tone and human-likeness there are no lower rungs. Three consequences follow, and each has a measurement in the sources:

1. **The reward's error is the target of optimization.** Gao et al. show that optimizing a proxy RM first raises and then lowers a held-out gold score as KL from the initial policy grows, while the proxy keeps rising ([[reward-model-overoptimization]] Fig. 1, Fig. 8). In a verifiable domain the error of the grader is bounded by its false-positive rate on known probes (ch-11); in a subjective domain the error is not measured unless you measure it.
2. **Human agreement sets a ceiling.** On SpeechJudge's naturalness pairs only 51.5% had full rater agreement and 17.2% weak agreement ([[speechjudge]] §3.2, Fig. 3). On MT-Bench, human experts agreed with each other on 81% of non-tied votes ([[judge-llm-bias]] Table 5a). A judge cannot be validated beyond the agreement of the people it imitates.
3. **The algorithm choice interacts with the reward.** Online RL samples the current policy and asks the judge about outputs it has never seen, which is where judge errors are largest. Offline DPO never queries a judge on new outputs but trains on a fixed distribution. §8 collects what labs chose and the few head-to-head measurements.

The learner's applied target is a sales voice agent whose quality is mostly subjective: whether a reply sounds natural, whether it answers before it pitches, whether the voice sounds human. This chapter is the reward-design basis for the capstone (ch-22) and the human-likeness bonus (ch-23).

The figure [subjective-reward.html](figures/subjective-reward.html) shows the pipeline from human pairs to RL with the failure-mode check that belongs at each stage. Open it before §2; its stages follow §2–§7 in order.

---

## §1 What changes when the score is unclear

A verifiable reward is a function `r(x, y) ∈ {0, 1}` computed by a program. A subjective reward is an estimate `r̂(x, y)` of a population's preference, built from finite labels. Four error sources enter `r̂` that do not exist for a program:

| Error source | What it is | Where it is measured |
|---|---|---|
| Label noise | Raters disagree on near-ties | Llama 2 RM 54.5% on "negligibly better" pairs ([[bradley-terry-rm]]); 51.5% full agreement ([[speechjudge]]) |
| Population taste | The RM learns the raters' taste, not a universal one | Reddit users skew male, educated, middle-aged ([[litbench]] §7); Chinese subsets had higher agreement than English ([[speechjudge]] §6) |
| Judge bias | Position, length, self-preference of an LLM judge | GPT-4 order-consistent on 65.0% of pairs ([[judge-llm-bias]] Table 2); Claude-3.5-Sonnet chose the first response about 60% of the time ([[writing-zero]] §3.1.3) |
| Optimization pressure | The policy moves to regions where `r̂` is wrong | proxy rises after gold peaks ([[reward-model-overoptimization]]) |

The first three are properties of the reward before training. The fourth only appears during training. The rest of the chapter is organized by which error each design choice reduces.

---

## §2 Pairwise human preferences and Bradley-Terry reward models

**The model.** Under Bradley-Terry, `p(y1 ≻ y2 | x) = σ(r(x, y1) − r(x, y2))`, and the RM is fit by minimizing `−E[log σ(r(x, y_w) − r(x, y_l))]` ([[bradley-terry-rm]], quoting DPO §3 Eq. 1–2). Only differences of `r` are identified; any `f(x)` can be added without changing the likelihood (DPO §5.1 Lemma 1). Two practical consequences: an RM score is meaningful only relative to another response for the same prompt, which matches how a group baseline uses it (llm-training ch-40); and deterministic labels (`p = 1`) push the reward gap to infinity and remove the effect of the KL term (IPO §4.2, in [[bradley-terry-rm]]).

**Report accuracy by preference strength.** Llama 2's helpfulness RM: 79.1% on "significantly better", 66.9% "better", 59.8% "slightly better", 54.5% "negligibly better / unsure" (App. A.3.3 Table 28, in [[bradley-terry-rm]]). A single average accuracy hides that the RM is near chance on exactly the pairs an RL policy produces late in training, when its samples differ in small ways.

**Data quality dominates data quantity.** Skywork-Reward-V2 trained plain BT RMs on 26M curated pairs and ranked first among open RMs on seven benchmarks (8B average 85.8, Table 1). Its ablations ([[skywork-reward-v2]] §4.3–4.4, Table 6):

- adding 12M uncurated pairs did not surpass the seed model;
- 1.8% of a 16M mixture (about 290K pairs) already exceeded the previous open state of the art;
- human curation: raw labels +0.4, with per-pair attributes +1.1, with the full written verification protocol +3.2 over a seed of 71.0;
- simple LLM curation added only 0.1.

The authors also cite concurrent work reporting that subjective preference learning does not show the data-scaling behavior that objective preferences show (§4.3). That is a cited claim, not a measurement in this paper.

**Natural signals need filters.** LitBench built creative-writing pairs from r/WritingPrompts upvotes ([[litbench]] §3.2): each story ≥10 upvotes, ≤2,048 tokens, ≥50 words; pairs need an upvote gap ≥25% and the higher-voted story must be posted later (so it cannot have gathered votes by being earlier); lengths are pruned to a symmetric length-difference histogram. The filter ablation (Fig. 7, Llama-3.2-1B BT RMs): a lightly filtered set of 395K pairs saturated at 65%; 1.03M timestamp-paired pairs without length filtering saturated at 70% and were length-biased; the filtered 43,827-pair training set did better. For a sales agent the natural signals are call outcomes and ratings; the same three controls apply (minimum gap, exposure control, length balance).

**A small in-domain RM can beat a frontier judge.** On LitBench, a Llama-8B BT RM reached about 78% versus 73% for Claude-3.7-Sonnet zero-shot (§5, Fig. 4). In a human study on new LLM-written stories, crowd-workers chose the RM-preferred story 57% of the time and the RM-rejected one 41% (§5, Fig. 8); Claude-3.7-Sonnet as a selector was at chance in that study.

**The RM determines the RL outcome.** RLMT ran the same GRPO recipe against two RMs: chat average 39.0 with Skywork-V2 and 14.8 with ArmoRM for the thinking model (Table 4, in [[rlmt-think-chat]]). With ArmoRM, MMLU-Redux and creative writing also dropped. This is a causal comparison within one setup (only the RM changed).

---

## §3 Generative, pairwise and group judges

A generative reward model (GenRM) produces text (a rationale, then a verdict or score) instead of a scalar head. The evidence on whether that helps is mixed and domain-dependent, so the choice should be made on your own gold set.

| Study | Domain | GenRM vs BT | Source |
|---|---|---|---|
| STaR-DPO judge | UltraFeedback / RewardBench | 73.9% in-distribution (BT ~73–74%); 81.9% RewardBench vs 78.9% GenRM without reasoning | [[generative-reward-models]] §5.1 |
| SpeechJudge-GRM | TTS naturalness | 77.2% (79.4% at voting@10) vs 72.7% BTRM, same backbone and data | [[speechjudge]] Table 3 |
| LitBench GenRM-CoT | creative writing | 72% vs about 78% for BT; distilled CoT lowered accuracy | [[litbench]] §5 |
| Writing-Zero GenRM | Chinese writing | 57.5 vs 64.7 for the in-house scalar RM in-domain, but better inside RL | [[writing-zero]] Table 1, §4.3.2 |

The Writing-Zero row is the important one for RL: **benchmark accuracy of a judge and its usefulness as an RL reward are different quantities.** The GenRM was less accurate on held-out pairs and still produced the better policy, because the scalar RM was exploitable in a way accuracy on human pairs does not measure.

**Pairwise against an in-group reference (BRPO).** Writing-Zero draws one rollout `o_ref` at random from each group, compares every rollout to it with the pairwise GenRM, and gives advantage +1 if the rollout wins in all voting conditions, else −1 (§3.2.1, Eq. 11). Groups whose results are nearly all one sign carry no ranking signal (an outlier reference makes everything win or everything lose), so groups with `|Σ R_i| / G > 0.6` are filtered; with G = 16 that is "filter if sum > 9" ([[writing-zero]] §4.1). Results against the scalar-RM GRPO baseline (Table 2–3):

| Policy | Mean length | Redundant self-explanation tokens | WritingBench | Held-out Writing Testset |
|---|---|---|---|---|
| ScalarRM-GRPO | 1,872 | 417 | 8.87 (marked hacked) | 2.83 |
| Writing-Zero (BRPO) | 1,292 | 58 | 8.29 | 3.84 |

Two lessons are in this table. The pairwise relative reward removed most of the length and self-justification growth. And WritingBench, an LLM-judged public benchmark, gave the highest score to the policy the authors describe as producing "gibberish" (§4.3.2). The card notes the KL setting and number of RL steps are not reported.

**Position bias inside the judge.** The cold-started Writing-Zero GenRM favored the second response; GenRM RL moved the mean P_former toward 0.5 and an advantage weighting (Eq. 10) reduced its batch-to-batch variance (Fig. 4). The general fix is order swapping (§6).

**Group judges at lab scale.** Three labs describe judges that see several responses at once:

- MAI-Thinking-1's RM is a post-trained MAI-Base-1 that predicts k-way 1–5 scores as text, applied cyclically over k permutations; reward = P(score = 5) ([[mai-thinking-1]] p. 44–45). Rocket runs such grading on the problem worker because it needs the whole group ([[mai-thinking-1]] Sec. 3.6; ch-02).
- Nemotron 3 Nano's RLHF uses a GenRM with circular pairwise comparisons, each response judged twice in different positions, 128 prompts × 16 responses ([[nemotron-ultra-recipe]] NR §3.3).
- Kimi K2's critic ranks actor responses by pairwise evaluation against core, prescriptive and human-annotated rubrics, and is refined during RL with on-policy rollouts from verifiable-reward prompts ([[kimi-k2-recipe]] §3.2.2). The report lists over-confidence as a known side effect of the prescriptive rubrics, which penalize hedging (App. F.3).

**Sampling the judge several times.** DeepSeek-GRM samples a pointwise GenRM that writes its own principles and critiques, sums k samples, and optionally keeps the better half with a scalar meta-RM: 67.9 at Voting@1 → 70.6 at Voting@8 → 72.8 at Voting@32 with the meta-RM ([[deepseek-grm]] Tables 2–3). These are RM-benchmark accuracies; the card records no policy-RL result. SpeechJudge reports +2.2 points at voting@10 (77.2 → 79.4). Voting multiplies judge cost by k; budget it as a service (ch-11, [[rlcf-checklists]]: 25 judge samples took 92 h vs 40 h for 5 on one 8×H100 node).

**An open implementation for comparison.** Open-instruct's LM-judge verifier ([[open-instruct-rlvr-pipeline]]) is pointwise and reference-free for general quality: one judge call per response, a 1–10 score in JSON, divided by 10.

```python
# open-instruct@f020e9a open_instruct/judge_utils.py:L30–L49 (excerpt)
general_quality_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
...
[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""
```

```python
# open-instruct@f020e9a open_instruct/judge_utils.py:L267–L268, L272–L275
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning(f"Could not parse score from due to invalid json: {score_str}, defaulting to 0.0")
...
def extract_score_with_fallback_max_10(score_str: str) -> "tuple[str, float]":
    """Extractor based on score with fallback"""
    reasoning, score = extract_json_score_with_fallback(score_str)
    return reasoning, score / 10.0
```

Compared with the measured designs above, this judge has no order swap (it is pointwise, so position bias does not apply, but the rubric-free Likert format is the weakest variant in [[rubrics-as-rewards]] Fig. 2: 25.5 vs 31.2 with per-prompt rubrics) and no length control in the prompt. The parse-failure path returns 0.0, so a judge output that is not valid JSON is indistinguishable from a bad response; inside a GRPO group that becomes a negative advantage for the policy sample. Count parse failures per step as a monitored metric if you use this pattern.

---

## §4 Rubric anchors and checklists

A rubric turns "is this good?" into several narrower questions, each closer to verifiable. The measured comparisons:

- **Per-prompt rubrics beat Likert and generic rubrics.** GRPO on Qwen2.5-7B with a gpt-4o-mini judge: HealthBench 31.2 with prompt-specific rubrics (RaR-Implicit), 25.5 with a rubric-free Likert judge, 12.5 with four generic rubrics shared by all prompts (Fig. 2 bar labels). Rubrics written with a reference answer: 35.9 vs 32.0 without ([[rubrics-as-rewards]] Table 1). Tested only on medicine and science, 300 GRPO steps (§9).
- **Checklists generated from candidate responses beat checklists from the instruction alone:** IFEval 77.3 vs 74.3, FollowBench 75.3 vs 72.8 ([[rlcf-checklists]] Table 6). RLCF uses the checklist score to choose DPO pairs, an offline method (§8).
- **Score aspects separately.** UltraFeedback's four-aspect annotation gave 65.2% on WebGPT vs 62.0% for one overall score ([[ultrafeedback]] Table 2).

**Rubrics for style.** Rubicon used a bank of more than 10,000 rubrics on about 5K samples and raised Qwen3-30B-A3B's average on seven open-ended benchmarks from 65.29 to 70.50 (Table 1). Its "Plain Narrative" style rubric scores voice/tone, content, and language on 1–5, with imperatives such as "Any hint of artificiality is a significant failure" (§4.2). The paper supports the "less AI-like" effect with case studies only; no human-likeness metric is reported ([[rubicon-rubric-anchors]] §4.2, Verification).

**Vetoes.** Rubicon's Reward Hacking Defense Rubric was written after inspecting anomalously high-reward rollouts. A deterministic filter flags "prefatory sycophancy" (praise of the user's prompt) and "laudatory self-evaluation" (meta-commentary on the response's own merit); flagged responses get a null score regardless of other dimensions (App. A.1). The authors report fewer reward spikes; no number is given (§3.2). The two flagged patterns are the same two features RLMT measured rising under a scalar RM (§7).

**The seesaw between objectives.** Rubicon trained single objectives to see interference: creativity-only training cost Collie −6.0 and IFEval −5.9; instruction-only training cost EQ-Bench3 −2.2 (§4.4, Fig. 2). Its final recipe trains instruction-following with verifiable checks and static rubrics first, then creative and social tasks with instance-specific rubrics (§3.1). Even so, the released model lost 1.85 IFEval points, 2.65 GPQA-D and 4.34 LCB v5 (Tables 1–2).

**Gating instead of summing.** MAI-Thinking-1 combines its helpfulness rewards lexicographically with instruction-following rewards primary; safety uses an aggregator in which a non-compliant response gets the minimum reward; the style judge (0/1/2) is applied only after the verifiable and safety gates ([[mai-thinking-1]] Sec. 3.4). The report gives the rule, not an ablation. The reason a gate is preferable to a weighted sum for style is that a weighted sum lets a large style score compensate a constraint failure; a gate does not. That is design reasoning, not a measured comparison.

---

## §5 Reference-based versus reference-free scoring

| Situation | Choice | Evidence |
|---|---|---|
| A correct answer exists (facts, prices, policy) | Reference-based judge | Reference-conditioned rubrics 35.9 vs 32.0 ([[rubrics-as-rewards]]); reference answers cut GPT-4 math judging failures from 14/20 to 3/20 ([[judge-llm-bias]] Table 4) |
| Reference-based judge in an RL loop | Probe it with content-free answers first | False-positive rates up to 90.9% (Qwen2.5-72B worst case) for ":", "Thought process:", "Solution" ([[master-rm-one-token]] Table 1) |
| Style target with real human examples | Real human data as the preferred side | Real-recording speech tokens beat own generations in 71% of 100 human checks; WER 7.2 → 6.0 over 3 DPO rounds ([[speechalign]] Tables 2–3); phrase frequency against a human corpus defines lexical AI-ness ([[antislop]] §3.1) |
| No single good answer (open conversation) | Reference-free pairwise against a sibling rollout | [[writing-zero]] BRPO (§3) |
| No ground truth but a fixed reference chosen by the judge | Pairwise −1/0/1 against a reference picked by Swiss-round GenRM scoring | [[minimax-m1]] §4.2.1 (no ablation in the card) |

The master-key result matters for subjective domains too: a sales-quality judge that also checks product facts against the catalog is a reference-based judge on that part, and the same probes apply.

---

## §6 Calibrating the judge against humans

Calibration is a procedure, not a single number. The sources give these steps and the measurements behind them:

1. **Build a gold set under a written protocol.** Per-pair attributes and tool access raised the human-curation gain from +0.4 to +3.2 ([[skywork-reward-v2]] Table 6). Keep preference strength as a label (§2).
2. **Evaluate on full-agreement pairs and per slice.** SpeechJudge-Eval keeps 1,000 pairs with full agreement and no ties (§4.1); expressive speech had 44.3% full agreement vs 54.5% for regular speech (§3.2). Report judge accuracy per slice, because the slice with lower human agreement is usually where the judge is weakest: SpeechJudge's best AudioLLM baseline scored 73.5 regular vs 66.2 expressive (Table 2).
3. **Swap order and count only consistent verdicts.** GPT-4 was order-consistent on 65.0%, GPT-3.5 46.2%, Claude-v1 23.8% (default prompt, [[judge-llm-bias]] Table 2). Few-shot judging raised GPT-4's consistency to 77.5% at 4× API cost (Table 12); the authors note consistency does not imply accuracy.
4. **Balance lengths in the eval set**, or the judge's length bias is invisible ([[litbench]] §3.2). The verbosity attack in MT-Bench (a list rephrased and prepended) fooled Claude-v1 and GPT-3.5 91.3% of the time and GPT-4 8.7% ([[judge-llm-bias]] Table 3).
5. **Validate on the policy's own outputs, not only on the human-written pairs.** LitBench's 57% vs 41% human study used new LLM-written stories (§5). Skywork-V2-Llama-3.1-8B agreed with humans 81.2% vs 74.3% for GPT-4o on an internal hold-out (App. H.2 Table 16; the set is undescribed in the card).
6. **Clamp AI-feedback labels before preference-model training** when the labeler writes chain-of-thought: Constitutional AI clamped label probabilities to 40–60% because unclamped CoT labels are near 0 or 1 and RL-CAI policies "would learn to output more extreme responses" (§4.1, §4.3 in [[constitutional-ai]]). The comparison of clamp ranges is qualitative (no numbers).
7. **Compare to the human ceiling.** GPT-4 pairwise agreed with experts on 85% of non-tied MT-Bench votes vs 81% human–human ([[judge-llm-bias]] Table 5a). Judge-human agreement near human-human agreement does not mean the judge is unexploitable; it means it is as reliable as one more human on the distribution it was tested on.

---

## §7 Failure modes specific to subjective rewards

| Failure | Measured evidence | Mechanism (stated or inferred) | Source |
|---|---|---|---|
| Length growth | 65.25% of raw chosen stories longer; +166 words (no-think) / +170 (think) RL − SFT | labels and RMs prefer longer answers; RL follows | [[litbench]], [[rlmt-think-chat]] Table 9 |
| Self-justification, self-praise | 417 vs 58 explanation tokens; self-praise rate +0.114 / +0.217 | the RM scores text that asserts quality | [[writing-zero]] Table 3; [[rlmt-think-chat]] Table 9 |
| Engagement-bait closers | rate +0.176 / +0.232 ("Would you like to hear...") | not stated | [[rlmt-think-chat]] Table 9 |
| Fast spread of a lexical hack | planted +6.81 bonus for "absolutely": 1% → 15% → 96% of rollouts at steps 0 / 40 / 70 with thinking; 3% → 3% → 19% without | a group baseline amplifies any feature that raises reward consistently | [[rlmt-think-chat]] Table 8 |
| Public benchmark rewards a hacked policy | 8.87, the highest WritingBench score, for a "gibberish" policy | benchmark judge shares the RM's biases | [[writing-zero]] §4.3.2 |
| Proxy rises after gold peaks | gold vs √KL rises then falls | Goodhart | [[reward-model-overoptimization]] Fig. 1 |
| Sycophancy | Claude 2 PM preferred convincing sycophantic answers over helpful truthful ones for 45% of the hardest misconceptions | non-expert labels reward agreement | [[sycophancy-in-lms]] §4.3.1 |
| Proxy metric near chance | UTMOS 53.7%, DNSMOS 57.9% agreement with human naturalness preferences | MOS predictors were not built for pairwise naturalness of modern TTS | [[speechjudge]] Table 2 |
| Seesaw between objectives | creativity-only: Collie −6.0, IFEval −5.9 | shared parameters, conflicting rewards | [[rubicon-rubric-anchors]] §4.4 |

**Correlation versus causation in this table.** The planted-hack row is causal: the authors added the bonus and measured its spread. The think-vs-no-think difference in that row is a comparison between two trained configurations; it shows thinking models picked up the hack faster in that setup, not why. The Table 9 organic features are measured as differences between RL and SFT checkpoints; that they were caused by RM exploitation is the authors' reading, consistent with but not proven by the data. The sycophancy row measures the preference model's preference, not a trained policy's behavior; [[sycophancy-in-lms]] separately reports that sycophancy rose during Claude 2 RL training (§4.2).

**Detection rule.** For every failure in the table, the feature is cheap to compute (length, bullet count, phrase regex, a classifier for praise). The measurement that converts these into an alarm is the RL − SFT delta tracked per step, plus a planted hack at the start of the run to verify that the monitoring catches spread within the step budget of one evaluation interval.

---

## §8 Which algorithm: online RL against a judge, or DPO-family

The algorithm crosswalk (ch-18, [[algorithm-crosswalk]]) lists what labs used. For subjective domains the cases are:

| Lab / paper | Subjective stage | Method | Stated reason | Source |
|---|---|---|---|---|
| InstructGPT | chat preference | PPO against a 6B BT RM | 1.3B preferred over 175B GPT-3 | [[algorithm-crosswalk]] Table 2 |
| Tülu 3 | preference stage before RLVR | length-normalized DPO | selected over standard DPO and SimPO (Table 18) | [[algorithm-crosswalk]] |
| Olmo 3 | preference stage before RL | DPO with delta-learning pairs (chosen from a large model, rejected from a small one) | pair quality depends on the contrast | [[olmo-3-recipe]] §4.3 |
| Nemotron 3 Nano | RLHF stage | RL against GenRM with circular pairwise comparisons, group-relative length control | verbosity −30% | [[nemotron-ultra-recipe]] NR §3.3 |
| Nemotron 3 Nano | experiment | DPO against tool hallucination | gains reported; "released model does not rely on DPO", reason not stated | [[nemotron-ultra-recipe]] App. C |
| MAI-Thinking-1 | helpfulness + safety specialist | GRPO (shared recipe) with RM = P(score = 5), lexicographic and gated composition | shared recipe across specialists; no subjective-specific reason stated | [[mai-thinking-1]] Sec. 3.4 |
| Kimi K2 | joint RL incl. subjective | K1.5 objective with self-critique rubric reward | not stated | [[kimi-k2-recipe]] §3.2 |
| MiniMax-M1 | general tasks mixed into RL | CISPO with GenRM; online length-bias recalibration | offline mitigation "frequently failed to prevent length bias during RL" | [[minimax-m1]] §4.2.2 |
| RLMT | chat with thinking | GRPO / PPO / online DPO against a BT RM | 40 runs; DPO/PPO comparison table not read in detail in the card | [[rlmt-think-chat]] |
| RLCF | instruction following | offline DPO on checklist-selected pairs | method choice not stated in the card | [[rlcf-checklists]] |
| Writing-Zero | writing | BRPO (GRPO-family, pairwise ±1) | removes need for a fixed reference and std normalization | [[writing-zero]] §3.2.1 |

**Head-to-head measurements.** Four sources compare online and offline methods on the same data:

1. **DPO vs PPO on TL;DR**: about 61% vs 57% GPT-4 win rate ([[dpo]] §6.2). Favors DPO, single study, models up to 6B.
2. **GRPO vs DPO on instruction following** with the same prompts and verifiers: IFEval 89.65 vs 79.67 ([[ifbench]] Table 5 via [[algorithm-crosswalk]]). Favors online RL; the reward is verifiable here, not subjective.
3. **Online vs offline DPO for TTS naturalness** (Qwen2.5-0.5B-TTS, [[speechjudge]] Fig. 6a): naturalness CMOS vs base — offline DPO on INTP 0.18, offline DPO on SpeechJudge-Data 0.16, offline DPO on judge-relabeled INTP 0.21, online DPO with the judge as reward 0.25 ± 0.09. Favors online use of the judge. Note the ±0.09 interval overlaps the offline values.
4. **DPO vs PPO vs Best-of-N for TTS** on the same real-vs-generated pairs ([[speechalign]] Table 3, LibriSpeech WER): RLHF-PPO 7.1, BoN 8.0, DPO-Iter1 6.7, DPO-Iter3 6.0. Favors iterated DPO, which is itself an on-policy loop (each iteration regenerates the rejected side).

**What the evidence supports.** No source isolates "online vs offline" for conversational naturalness or style text with a learned judge. The supportable reading is:

- Offline DPO is used by labs as a stage before RL (Tülu 3, Olmo 3), when pairs come from a construction rule rather than a judge (delta learning, real-vs-generated, style-contrast prompts, ch-23), or when the target is a narrow pattern (Antislop's FTPO, ch-23).
- Online RL against a judge is used when a judge exists that can score fresh samples (MAI, Kimi, MiniMax, Nemotron, Writing-Zero). Its risk is that the policy reaches outputs the judge was never validated on; the mitigations are those of §3–§7.
- Iterated or online DPO sits between the two: on-policy samples, pairwise loss, no value model. SpeechJudge and SpeechAlign both used it for speech.

The causal claim "online beats offline for subjective rewards" is not established by these sources. Comparisons 3 and 4 point in that direction for speech; comparison 1 points the other way for summarization.

---

## Recipe

1. Write down the subjective quality as named components (e.g., register, turn length, answers-before-pitching, voice naturalness). For each, decide whether a program, a reference-based judge, or a reference-free pairwise judge can score it (§5).
2. Build a gold set: pairwise human labels under a written protocol with preference strength, length-balanced, including outputs from your current policy (§2, §6).
3. If you mine natural signals, apply a minimum signal gap, exposure control, and length rebalancing before fitting ([[litbench]] §3.2).
4. Fit at least two judge types (BT and GenRM) and compare them on the gold set per slice and per preference strength; do not choose by public-benchmark accuracy (§3; [[skywork-reward-v2]] Fig. 2).
5. Calibrate: order-swap consistency, length-attack test, master-key probes for any reference-based part, human validation on policy samples (§6).
6. Choose the reward format: pairwise against an in-group reference with a one-sign group filter, or per-prompt rubrics; add vetoes for user praise and self-praise; gate style behind constraint and safety checks (§3–§4).
7. Order stages: constraint-following first, style second ([[rubicon-rubric-anchors]]).
8. Choose the algorithm (§8): online group-baseline RL against the calibrated judge when it can score fresh samples; DPO-family for construction-rule pairs or as a pre-RL stage; token-level methods for lexical patterns.
9. Before the run, plant a known hack and confirm the monitoring detects its spread ([[rlmt-think-chat]] Table 8).
10. During the run, log per step: length, bullets, engagement-bait rate, self-praise rate, judge parse failures, order-consistency on a fixed probe set, and a held-out human-aligned score vs √KL. Stop at the held-out score's peak ([[reward-model-overoptimization]]).
11. Evaluate with a different judge than the one you trained against, plus a small human study; do not rely on an LLM-judged public benchmark alone ([[writing-zero]] §4.3.2).

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Reporting one RM accuracy number | RL rewards small, random differences late in training | accuracy by preference strength; near 50% on near-ties ([[bradley-terry-rm]]) |
| Choosing a judge by public RM benchmarks | good benchmark score, poor policy | RewardBench vs PPE Preference Pearson 0.07 ([[skywork-reward-v2]] Fig. 2); evaluate on your gold set |
| Training on unbalanced lengths | mean length rises every eval | length-difference histogram of chosen − rejected; LitBench symmetry check |
| Single-order pairwise judging | rewards depend on which rollout is "A" | order-swap consistency on a fixed probe set ([[judge-llm-bias]]) |
| Trusting rising judge reward | proxy up, human score flat or down | held-out human or different-judge score vs √KL ([[reward-model-overoptimization]]) |
| No vetoes | "Great question!" and "This answer covers all…" appear | per-response praise / self-praise regex rate, RL − SFT ([[rlmt-think-chat]] Table 9) |
| Joint style + constraint training | IFEval or constraint metric drops | track a constraint benchmark per eval ([[rubicon-rubric-anchors]] Fig. 2) |
| Using a MOS predictor as speech reward | naturalness does not improve for listeners | check proxy agreement with human A/B labels first; UTMOS 53.7% ([[speechjudge]]) |
| Silent judge parse failures | random negative advantages | count of unparsed judge outputs per step (open-instruct returns 0.0, `judge_utils.py` L267–268) |

---

## Check your understanding

1. Why does a Bradley-Terry RM's accuracy on "negligibly better" pairs matter more for late-stage RL than its average accuracy?
2. Writing-Zero's GenRM was less accurate than the scalar RM on held-out pairs yet produced a better policy. Explain the mechanism by which the scalar RM was exploited, and why pairwise comparison against a sibling rollout makes that exploitation harder.
3. Why does BRPO need the `|ΣR_i|/G > 0.6` filter? What would the advantages look like without it when the random reference is the best rollout in the group?
4. Rubicon trained instruction-following before creative rubrics. What measurement motivated that order, and what would you expect if the order were reversed?
5. In the RLMT planted-hack test, why does a group-baseline algorithm spread a consistent small bonus to nearly all rollouts within tens of steps, even if the bonus is half a reward standard deviation?
6. SpeechJudge's online DPO beat offline DPO while the DPO paper's TL;DR result favored DPO over PPO. What differs between these setups that could explain the opposite directions, and why does neither settle the question for conversational text?
7. You have a sales-quality judge that agrees with humans at 82% on full-agreement pairs. Why is that not sufficient evidence to run 1,000 GRPO steps against it?

---

## Connections

- Course chapters: ch-11 (grader ladder; master-key probes, judge prompt design), ch-12 (reward composition and hacking monitors), ch-14 (instruction-following pipeline; the constraint stage that precedes style), ch-17 (stage order across domains), ch-18 (algorithm crosswalk), ch-22 (capstone worksheet, grader ladder and reward composition rows), ch-23 (human-likeness components, DPO costs, speech judges).
- llm-training: ch-41 (reward modeling, Bradley-Terry, over-optimization), ch-42 (reward hacking and judge design), ch-49 (judge bias and calibration), ch-40 (group baselines; why a consistent bonus spreads), ch-43 (entropy and KL control), ch-44b (multi-domain RL with non-verifiable rewards), ch-46 (DPO vs RLVR lab).
- Sources: [[subjective-reward-guide]] is the rule list this chapter expands; [[writing-zero]], [[rlmt-think-chat]], [[rubicon-rubric-anchors]] for hacking evidence; [[skywork-reward-v2]], [[litbench]] for data construction; [[speechjudge]] for speech and online DPO.

---

## Sources

- [[subjective-reward-guide]] — rule list (measured vs practice) for subjective rewards; organizing basis of this chapter.
- [[bradley-terry-rm]] — BT model and loss; Llama 2 accuracy by preference strength; IPO deterministic-label issue.
- [[skywork-reward-v2]] — curation vs quantity ablations; human-agreement and benchmark-correlation numbers.
- [[litbench]] — natural-signal pair construction, length bias, small RM vs frontier judge, human validation.
- [[generative-reward-models]] — STaR-DPO GenRM vs BT accuracy.
- [[deepseek-grm]] — pointwise GenRM with principles; voting and meta-RM scaling.
- [[writing-zero]] — pairwise GenRM inside RL (BRPO); scalar-RM hacking; hacked policy tops WritingBench.
- [[rlmt-think-chat]] — RM-strength ablation; planted hack; organic hacking features.
- [[rubrics-as-rewards]] — per-prompt rubric rewards vs Likert and generic rubrics; reference-conditioned rubrics.
- [[rlcf-checklists]] — candidate-based checklists; grading cost.
- [[rubicon-rubric-anchors]] — rubric bank, style rubric, veto rubric, seesaw effect, stage order.
- [[ultrafeedback]] — multi-aspect vs overall annotation.
- [[judge-llm-bias]] — position and verbosity bias; reference-guided grading; judge-human agreement.
- [[constitutional-ai]] — clamping AI-feedback labels.
- [[master-rm-one-token]] — master-key false positives in reference-based judges.
- [[reward-model-overoptimization]] — gold vs proxy under optimization; stop at the peak.
- [[sycophancy-in-lms]] — preference-model sycophancy.
- [[kimi-k2-recipe]] — self-critique rubric reward.
- [[mai-thinking-1]] — k-way RM, lexicographic and gated reward composition.
- [[nemotron-ultra-recipe]] — circular pairwise GenRM RLHF; DPO experiment not shipped.
- [[olmo-3-recipe]] — DPO delta-learning stage.
- [[minimax-m1]] — GenRM length bias needs online recalibration; pairwise vs reference.
- [[algorithm-crosswalk]] — lab × algorithm table; IFBench GRPO vs DPO; DPO vs PPO.
- [[ifbench]], [[dpo]] — head-to-head comparisons cited via the crosswalk.
- [[speechjudge]] — speech naturalness judge; MOS proxies near chance; online vs offline DPO.
- [[speechalign]] — real-vs-generated speech pairs; DPO vs PPO vs BoN.
- [[antislop]] — DPO diversity cost vs FTPO.
- [[open-instruct-rlvr-pipeline]] — open-instruct verifier pipeline (judge code quoted at f020e9a).
