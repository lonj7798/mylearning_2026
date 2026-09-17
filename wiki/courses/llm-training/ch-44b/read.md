<!-- chapter: ch-44b
     track: rl
     kind: content
     title: Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing
     deps: [ch-44a, ch-38a]
     sources: [[deepseek-r1]], [[deepseek-r1-recipe]], [[qwen-3]], [[qwen-3-general-rl]], [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[kimi-k2]], [[kimi-k2-recipe]], [[nemotron-cascade]], [[rubrics-as-rewards]], [[rlcf-checklists]], [[deepseek-grm]], [[guru-cross-domain-rl]], [[general-reasoner]], [[nemotron-crossthink]], [[multi-domain-rlvr-data-centric]], [[scaling-reasoning-losing-control-mathif]], [[qwen3-2507-instruct-thinking-split]], [[deliberative-alignment]], [[openai-safe-completions]], [[ifbench]], [[grpo]]
     figures: figures/cascade-stage-tracker.html
     revised: 2026-09 (generality revision)
-->

# Chapter 44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing

> **Core insight.** RL over many domains raises the average across those domains, while the highest score
> in any single domain usually belongs to a run trained on that domain alone: in [[nemotron-crossthink]]
> the best mixed blend reaches 58.12 average over seven benchmarks against 57.82 for the math-only blend,
> but 65.0 against 70.0 on AMC 23 (Table 3); in [[multi-domain-rlvr-data-centric]] the three-domain run has
> the highest overall average (56.57) and a lower puzzle average (49.73) than puzzle-only training (61.98)
> (Table 9). Frontier recipes accept that trade: after general RL, Qwen3-32B gains 12.0 points of IFEval and
> loses 2.4 points of AIME'24 relative to the reasoning-RL checkpoint ([[qwen-3-general-rl]], Table 22).
> The part of the recipe that makes general RL possible is the reward: rubric, checklist, generative-reward-model
> and safety-specification judges replace the verifier where no verifier exists.
>
> **Guideline.** When a stage must cover tasks with no programmatic checker, use an instance-specific rubric
> or checklist scored by a judge rather than a single Likert score from the same judge, because RaR-Implicit
> reaches 31.2 on HealthBench against 25.5 for a rubric-free Likert reward at 7B ([[rubrics-as-rewards]], Fig. 2).
> When a rule-based verifier exists but response quality also matters, combine the verifier with a
> reward-model score inside one reward instead of using the verifier alone, because verifier-only
> instruction-following RL lowered AlpacaEval 2 from 33.5 to 21.3 ([[ifbench]], Table 3) and degraded human
> alignment in [[nemotron-cascade]] (§4.4.2). When domains differ in verification latency and response
> length, sequential domain-wise RL is a reported option under which the largest later-stage
> loss on an earlier stage's benchmark is 3.8 points of ArenaHard, 93.1 after RLHF against 89.3 after Math RL
> on the 14B-Thinking model ([[nemotron-cascade]], Tables 4-8); when the goal is one balanced model and the domains are
> already similar in cost, mixed-domain RL is the reported option with the highest cross-domain average
> ([[guru-cross-domain-rl]], Table 3). No source compares the two designs at equal data and compute.

## Why this chapter matters for a general-purpose model

The previous chapters of this track train one reward channel at a time: a verifier for math and code
([[ch-44]]), a length control ([[ch-44a]]), a Bradley-Terry reward model for chat ([[ch-41]]). A deployed
model is asked for all of it in the same session, so the last part of post-training has to hold several
objectives at once. This chapter covers the stage that sits between reasoning RL and the self-improvement
loops of [[ch-45]]: pre-training → mid-training → SFT → preference optimization → **reasoning RL → general,
multi-domain RL** → evaluation.

Two questions decide whether this stage broadens or narrows the model. The first is where the reward comes
from for tasks that have no checker, because a wrong answer to that question produces a model that optimizes
the judge rather than the task ([[ch-42]]). The second is how prompts from different domains are ordered or
mixed, because RL updates on one domain change behaviour on the others, in both directions, by measured
amounts of up to 22.56 points ([[multi-domain-rlvr-data-centric]], Table 9).

## §1 What frontier recipes do in the final RL stage

**Definition.** *General RL* (the term used by [[qwen-3]] and [[glm-5]]) is an RL stage whose prompt set spans
many task types, including ones with no programmatic verifier, and whose reward comes from a mixture of rule
checks, reward models, and judge models. *Multi-domain RLVR* is the narrower case where every domain still
has a verifier.

Two structures appear in the reports. In a **mixed** structure, one stage draws prompts from all domains and
routes each prompt to its own reward function. In a **sequential** (cascaded) structure, each domain gets its
own stage, run one after another on the same checkpoint.

| Recipe | Structure of the final RL stages | Reward sources | What the report shows about interference |
|---|---|---|---|
| DeepSeek-R1 ([[deepseek-r1]]) | Mixed: rule rewards for reasoning prompts plus preference rewards for general prompts in one 1,700-step stage; general data and preference rewards only in the final 400 steps | rule + helpful RM (66k pairs) + safety RM (106k prompts) + format + language consistency (Eq. 8-10) | With the helpful RM, reward rises while Codeforces performance falls (B.5, Fig. 6); the 400-step window is the stated mitigation (§3.2.2) |
| Qwen3 ([[qwen-3]], [[qwen-3-general-rl]]) | Mixed: one General RL stage over "over 20 distinct tasks" after reasoning RL and thinking-mode fusion | rule; model-based with reference answer (Qwen2.5-72B-Instruct); reward model without reference (§4.4) | Table 22: general and instruction metrics rise, AIME'24 and LiveCodeBench fall in thinking mode; the authors state they accept the trade for versatility |
| GLM-4.5 ([[glm-4-5]]) | Expert models per domain (reasoning, agent, general chat), each cold-started and RL-trained, then distilled into one model; general RL runs last in four tracks | RM trained on human preferences + AI rubric scoring; rule + RM + critique model for instruction following; step-wise rules for function calling (§3.4) | Pathologies occur in "often less than 1% of outputs", which the report calls sample-inefficient to fix inside general RL (§3.4) |
| GLM-5 ([[glm-5]]) | Sequential: reasoning RL → agentic RL → general RL, then on-policy cross-stage distillation | rule-based functions + outcome reward models + generative reward models, plus human-written responses as style anchors (§3.4) | The distillation stage exists because "sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities" (§3.5) |
| Kimi K2 ([[kimi-k2]]) | Joint: a verifiable-reward gym and a self-critique rubric reward trained together | verifiers (math, STEM, logic, code, SWE sandbox, IF with a hack-check layer) + K2 critic ranking against core, prescriptive and human rubrics (§3.2) | A PTX loss (an auxiliary next-token-prediction loss on hand-selected high-quality samples, added to the RL objective) is used against forgetting and overfitting to the RL task set (§3.2.3); no ablation |
| Nemotron-Cascade ([[nemotron-cascade]]) | Sequential: RLHF → IF-RL → Math RL → Code RL → SWE RL, 2,232 RL steps in total for the 14B run | 72B Bradley-Terry RM; rule-based IF verifier combined with the same RM; symbolic math verifier; unit tests; SWE tests | Per-stage tables for every benchmark; later stages "rarely degrade the benchmark performance attained in earlier domains" (Abstract) |

The table describes two design families rather than a ranking. Mixed stages need one reward router and one
length budget for prompts whose response lengths differ by an order of magnitude; sequential stages let each
domain keep its own maximum response length, temperature and verifier latency, which is the stated motivation
in [[nemotron-cascade]] (§1), where math verification is "orders of magnitude faster" than code verification.

## §2 Rewards when there is no verifier

### 2.1 The four reward sources

Two reports name overlapping sets with their stated trade-offs: [[qwen-3-general-rl]] (§4.4) lists types 1-3
below, and [[glm-5]] (§3.4) lists types 1, 3 and 4.

1. **Rule-based reward** — a program decides. Precise, cheap, hard to hack; limited to what a rule can express.
2. **Model-based reward with a reference answer** — a judge compares the response to a gold answer. Qwen3 uses
   Qwen2.5-72B-Instruct here and states the purpose as "avoiding false negatives that can occur with purely
   rule-based rewards".
3. **Model-based reward without a reference** — a scalar reward model trained on human preferences. GLM-5
   calls these outcome reward models: low variance and efficient, "more susceptible to reward hacking".
4. **Generative reward model** — a language model that writes criteria or critiques and then scores. GLM-5:
   "more robust to such exploitation, but tend to exhibit higher variance".

The rest of §2 is about what (2) and (4) look like when built properly.

### 2.2 Rubrics as rewards

**Definition.** A rubric reward replaces one quality score with k weighted criteria written for that single
prompt, each checked by a judge ([[rubrics-as-rewards]], §2.2).

**The problem it addresses.** A single Likert score from a judge gives one number per response with no
statement of what was wrong, and judges of different sizes disagree about it (§5, Fig. 3).

**Mechanism.** An LLM writes 7-20 self-contained items per prompt, conditioned on the dataset's reference
answer; each item carries a category (Essential, Important, Optional, Pitfall) and a weight (§3.2). At
training time, either the judge returns one score given all criteria (implicit) or each criterion is checked
and the scores are combined (explicit, Eq. 1):

```
r(x, ŷ) = Σ_j w_j · c_j(x, ŷ) / Σ_j w_j
```

x is the prompt; ŷ a sampled response; k the number of criteria; w_j the weight of criterion j;
c_j(x, ŷ) ∈ {0, 1} the judge's decision that ŷ satisfies criterion j. The denominator makes rewards
comparable across prompts with different numbers of criteria.

**Worked example.** RaR-Explicit maps Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9 (§4.4). Take
a rubric with one Essential, one Important and one Pitfall item, so the denominator is
1.0 + 0.7 + 0.9 = 2.6. A response that satisfies the Essential and Pitfall items and misses the Important one scores
(1.0 + 0.9) / 2.6 = 1.9 / 2.6 = 0.73. A response that instead misses the Essential item and satisfies the
other two scores (0.7 + 0.9) / 2.6 = 1.6 / 2.6 = 0.62. Because the aggregation is linear, the cost of a miss
is its own weight divided by the total: 1.0 / 2.6 = 0.38 for the Essential item against 0.7 / 2.6 = 0.27 for
the Important one, a ratio of 1.43 = 1.0 / 0.7.

**Evidence.** GRPO on Qwen2.5-7B with a gpt-4o-mini judge: HealthBench 31.2 (RaR-Implicit), 29.7
(RaR-Explicit), 28.9 (reference-Likert), 25.5 (direct Likert), 12.5 (four generic rubrics shared by all
prompts), 22.7 (Qwen2.5-7B-Instruct off the shelf); GPQA-Diamond 37.6 / 36.9 / 36.5 / 34.8 / 31.7 / 35.0
(Fig. 2 bar labels). Rubrics written with the reference answer beat rubrics written without it: 35.9 against
32.0 on HealthBench-1k (Table 1). **Result (single study).**

**Conditions and limits.** One policy size (7B), 300 GRPO steps, medicine and science only; dialogue, tool
use and agentic tasks are not tested (§9). The same judge model scores training rewards and evaluation
(Fig. 2 caption), so part of the measured gain may be agreement with that judge rather than task quality.

### 2.3 Checklist feedback

[[rlcf-checklists]] builds the same idea for instruction following and adds programs. Qwen2.5-72B-Instruct
writes a yes/no checklist per instruction, conditioned on candidate responses from four small models; each
item gets an importance weight out of 100. Each item is graded by the mean of 25 judge samples in 0-100 and,
where the judge is confident it can check the requirement exactly, by a generated verification program; the
two scores are averaged. Weighted item scores are averaged into one score per response, the 40% of pairs with
the largest per-criterion difference become DPO pairs, and DPO trains the policy.

Two details matter for generality. First, the authors added two "universal requirements" (directness and
context-appropriate tone, weight 100/100) to every checklist after they observed responses that opened with
long preambles, which they read as reward hacking (§2). Second, the method is preference-based: policy-gradient
use of checklist feedback is listed as future work (§7).

**Evidence.** On Qwen2.5-7B-Instruct: FollowBench average hard satisfaction rate 71.4 → 75.3 with RLCF,
69.5 with DPO via Skywork-Reward-Gemma-2-27B, 70.4 via ArmoRM; InFoBench overall 78.1 → 84.1 / 82.0 / 83.5;
Arena-Hard vanilla 51.3 → 54.6 / 55.1 / 50.8. The reward models help on some benchmarks and regress on others,
while RLCF is positive on all five (Tables 2-4). **Result (single study).**

**Narrowing that the paper measures itself.** RLCF prompts come from WildChat, which is weighted toward daily
assistance and writing. On tasks outside that distribution, scores move down: XSTest unsafe 83.0 → 81.0,
GSM8K 83.2 → 82.2, TruthfulQA MC1 43.5 → 42.0 (Table 9). This is the recurring pattern in this chapter: a
reward that is well designed for its prompt distribution still narrows the model outside that distribution.

### 2.4 Generalist generative reward models

[[deepseek-grm]] trains the judge itself with RL. The model generates principles, then critiques, then a
pointwise score per response, and Self-Principled Critique Tuning trains that behaviour: rejective fine-tuning
keeps only trajectories whose scores rank the ground-truth best response strictly highest (N_RFT = 3), then
rule-based RL gives +1 when the score ordering is correct and −1 otherwise, with a larger KL coefficient to
hold the output format (§3.2). At inference, k sampled trajectories with shuffled response order are summed,
optionally filtered by a meta reward model.

**Evidence.** Overall accuracy across RewardBench, PPE Preference, PPE Correctness and RMB: 69.9 greedy,
71.0 with Voting@32, 72.8 with meta-RM Voting@32, against 70.5 for Nemotron-4-340B-Reward and 66.9 for
Skywork-Reward-Gemma-2-27B (Table 2). Removing principle generation costs 2.4 points (69.9 → 67.5, Table 4).
Scalar reward models score highest on PPE Correctness and fall behind elsewhere, which the authors describe
as domain bias (§5.2). **Result (single study).**

**Limit that matters here.** The paper evaluates the GRM as a judge. Using it as the reward inside online RL
is listed as future work (§7), so the transfer from judge accuracy to policy quality is not measured. The
practical version of that transfer is in [[glm-5]] §3.4, where generative reward models are one of three
reward types in General RL, with no ablation isolating their contribution.

### 2.5 Self-critique rubric reward

[[kimi-k2]] runs the judge inside the policy. The K2 critic is initialized from preference data during SFT,
then ranks actor responses by pairwise comparison against core rubrics, prescriptive rubrics intended to
remove reward hacking, and human-annotated rubrics (§3.2.2). During RL the critic is updated on on-policy
rollouts from verifiable-reward prompts, which the report describes as distilling RLVR signals into the
evaluation model.

The prescriptive rubrics are specific behaviour bans: no opening compliments, no sentences explaining why the
response is good (App. F.2). The report also states the side effect in App. F.3: the rules penalize hedging and
disclaimers, so "the model may occasionally overstate certainty" where nuance is appropriate. A behaviour
reward that forbids a surface pattern also removes the legitimate uses of that pattern.

### 2.6 Combining a verifier with a reward model

A verifier that returns 0 or 1 for constraint satisfaction says nothing about response quality. Two sources
measure the consequence. [[ifbench]] reports that verifier-only instruction-following RLVR from a Tülu-3-8B-DPO
policy moved AlpacaEval 2 from 33.5 to 21.3 and GPT-4.1 judge quality of the same responses with the constraint
removed from 7 to 6.4 out of 10 (Table 3, §5), and that adding a reward-model term recovers AlpacaEval 2 to
31.6 at a cost of IF scores (App. E). [[nemotron-cascade]] reports that a rule-based IF verifier alone
"degraded human alignment results" and gives the combined reward it used for the thinking models (§4.4.2):

```
r_i = R_IF(o_i) + sigmoid(R̂_RM(o_i))   if R_IF(o_i) = 1
r_i = 0                                 otherwise
R̂_RM(o_i) = (R_RM(o_i) − mean({R_RM(o_j)}_{j=1..G})) / std({R_RM(o_j)}_{j=1..G})
```

o_i is the i-th sampled response in a group of G; R_IF ∈ {0, 1} is the verifier output; R_RM is the scalar
reward-model score; R̂_RM is that score standardized inside the group; sigmoid maps it into (0, 1) so it can
never overturn the verifier decision.

**Worked example.** Take G = 4 responses to one constrained prompt. Two satisfy every constraint and get
standardized RM scores 0.8 and −1.2; two violate a constraint. Their rewards are
1 + sigmoid(0.8) = 1.69, 1 + sigmoid(−1.2) = 1.23, 0, 0. Group mean 0.73, standard deviation 0.748, so the
GRPO advantages are +1.28, +0.67, −0.98, −0.98. Every compliant response keeps a positive advantage and the
better-written one gets about twice the push of the weaker one, which is the intended ordering: satisfy the
constraint first, write well second.

The 8B unified model in the same report takes the other route to the same problem: RLHF is run in both
thinking and non-thinking modes, and IF-RL only in non-thinking mode, on the stated hypothesis that IF-RL
applied to an already-aligned model in the cheaper mode is less prone to reward-hacking the rule-based verifier
(§4.4.2). Reversing the order of RLHF and IF-RL gave "much worse results"; no numbers are printed.
**Result (single study), qualitative.**

## §3 Cross-domain transfer and interference

**Definition.** *Transfer* is the change on domain B caused by RL on domain A; it is *interference* when the
change is negative. Every study below measures it the same way: train on one prompt set, evaluate on a fixed
suite covering all domains.

### 3.1 Transfer is asymmetric by domain and by task difficulty

[[guru-cross-domain-rl]] trains Qwen2.5-7B-Base and Qwen2.5-32B-Base on 3K-sample subsets of Math, Code,
Science, Logic, Simulation, Tabular, and on their 18K mixture, and evaluates all six. The stated result:
Math, Code and Science gain from RL on other domains, while Logic, Simulation and Tabular need in-domain data.
Within the gaining domains, easy tasks (MATH500, AMC, HumanEval, MBPP) gain more from out-of-domain RL than
hard ones (AIME24, LiveCodeBench); the lowest-scoring tasks (ARC-AGI, CodeI/O) show "marginal to negligible"
cross-domain gains (§3.1). The authors' explanation is exposure during pre-training (**Interpretation**).

The trained models give the in-domain version of the same point. GURU-7B averages 43.29 over 17 benchmarks
against 35.42 for Open-Reasoner-Zero-7B and 33.76 for General-Reasoner-7B; GURU-32B averages 54.24 against
47.53 (Table 3). The exception is instruction following, which no domain in the corpus targets: GURU-7B scores
35.81 on IFEval against 39.56 for General-Reasoner-7B and 36.69 for SimpleRL-7B. Breadth across reasoning
domains did not produce breadth on instruction following in this comparison.

### 3.2 Difficulty filtering is an in-domain optimization that can cost transfer

Filtering a domain's prompts down to hard items is standard practice for reasoning RL. [[guru-cross-domain-rl]]
measures its cross-domain price (Table 2, Qwen2.5-7B-Base, 200 steps, best validation accuracy):

| Training data | MATH500 | AMC | AIME24 | HumanEval | LiveCodeBench | HiTab | Multihiertt |
|---|---|---|---|---|---|---|---|
| Unfiltered math | 75.8 | 52.1 | 15.8 | 82.3 | 11.1 | 56.5 | 32.0 |
| Difficulty-filtered math | 78.6 | 58.4 | 21.7 | 73.1 | 10.7 | 53.5 | 35.5 |

In-domain gains of +2.8, +6.3 and +5.9 come with −9.2 on HumanEval and −3.0 on HiTab, the two easiest
cross-domain tasks in the row, and the authors report accuracy collapse on both during training on the
filtered data. The two hardest cross-domain tasks move in the other direction or not at all: LiveCodeBench
falls 0.4 points and Multihiertt rises 3.5. **Result (single study).**

### 3.3 Mixtures, and what the mixture buys

[[nemotron-crossthink]] compares blends on Qwen2.5-7B with GRPO. Its best blend, a 2:1 ratio of
general-purpose reasoning data to math data, averages 58.12 over seven benchmarks against 44.75 untrained and
57.82 for the math-only blend; general-purpose data alone averages 53.30. The per-benchmark split shows where
the average difference comes from: the mixed blend is +3.56 on MMLU-Pro and +2.32 on AGIEval over math-only, and −5.0 on
AMC 23 (Table 3). The same report measures response length: correct answers from the mixed-blend model use
28% fewer tokens on average than from the math-only model, and the mixed model varies its length by task type
(622 tokens on math against 385 on general prompts, a 62% increase, against 14% for the math-only model)
(§6, Table 12). **Result (single study).**

[[general-reasoner]] runs the same comparison with a different corpus: 230K web-sourced questions with a 1.5B
generative verifier for answers that rule-based matching cannot check. At 7B, math-only training is 0.6 points
higher on the math-related average (49.1 against 48.5) and lower on all three general benchmarks, by 2.0
(MMLU-Pro 56.9 against 58.9), 1.5 (GPQA 32.8 against 34.3) and 4.4 points (SuperGPQA 29.8 against 34.2); at
14B the full-data model is higher in every column (Table 4). Its verifier ablation is a separate result:
with identical conditions on Qwen3-4B-Base for 120 steps, the model-based verifier beats the rule-based one
on all four evaluation sets (MMLU-Pro 60.1 against 58.1), and the rule-based run plateaus near step 60
(Table 5, Fig. 4). The authors read this as rule-based matching marking valid answers wrong and so removing
usable reward signal (**Interpretation**).

Taken together with §3.1: **Replicated** — mixing domains raises the multi-domain average over any
single-domain run at the same scale ([[nemotron-crossthink]] Table 3, [[general-reasoner]] Table 4,
[[multi-domain-rlvr-data-centric]] Table 9). **Replicated** — the peak of a single domain usually belongs to a
run specialized on it or on data closer to it ([[nemotron-crossthink]] AMC 23,
[[multi-domain-rlvr-data-centric]] puzzles, [[guru-cross-domain-rl]] Table 2 for in-domain difficulty
filtering and Table 3 for IFEval). [[guru-cross-domain-rl]] does not print the per-cell single-domain versus
mixed numbers behind its transfer study (Fig. 3), so its contribution here is the direction of transfer, not a
mixed-versus-single average.

### 3.4 Interference is measured, and the averaging convention can hide it

[[multi-domain-rlvr-data-centric]] runs all seven combinations of Math, Code and Puzzle on Qwen2.5-7B-Base
(Table 9):

| Training data | Math avg | Code avg | Puzzle avg | All avg |
|---|---|---|---|---|
| Base | 22.48 | 67.46 | 9.07 | 31.50 |
| Math | 47.48 | 64.23 | 22.42 | 45.11 |
| Puzzle | 29.47 | 71.35 | 61.98 | 50.72 |
| Code | 19.17 | 73.95 | 22.55 | 35.78 |
| Math + Puzzle | 49.72 | 44.90 | 49.78 | 48.36 |
| Puzzle + Code | 32.06 | 74.88 | 55.15 | 50.89 |
| Math + Code | 47.22 | 75.06 | 25.34 | 48.92 |
| Math + Puzzle + Code | 49.75 | 73.63 | 49.73 | 56.57 |

Math + Puzzle is 22.56 points below the base model on code, worse than either single-domain run; adding code
data to that pair restores the code average to 73.63 and leaves the puzzle average unchanged (49.78 → 49.73).
Measured against the best pair instead, the triple combination costs 5.42 points of puzzle accuracy
(55.15 → 49.73) and gains 17.69 on math. The triple combination has the best overall average and collapses on
no domain, which is the argument for mixing.

**Worked example on the averaging convention.** "All avg" in that table is the mean of the seven individual
benchmark scores, not the mean of the three domain averages. For the base model, the seven scores
(MATH500 56.40, CountDown 1.05, AIME24 10.00, HumanEval 70.12, MBPP 64.80, KK 17.86, Zebra 0.27) average to
220.50 / 7 = 31.50, while the three domain averages give (22.48 + 67.46 + 9.07) / 3 = 33.00. The two
conventions can reorder runs: Puzzle + Code (50.89) is above Puzzle-only (50.72) under the task mean, and
below it under the domain mean ((32.06 + 74.88 + 55.15)/3 = 54.03 against (29.47 + 71.35 + 61.98)/3 = 54.27).
A domain with more benchmarks in the suite gets more weight in the first convention. When mixtures are
compared by an average, state which convention the average uses and print the per-domain numbers next to it,
because the two conventions reorder two of the eight runs in this table.

A second warning from the same paper concerns measurement rather than training: the same checkpoint scores
0.00 or 20.79 on CountDown depending on which chat template the evaluation uses, and the instruct model drops
to 1.80 on MATH500 under a mismatched template (Table 11). Cross-domain comparisons made with one template
across base and instruct models are not comparable.

## §4 Reasoning capability versus general capability

Three independent measurements say that reasoning-oriented training reduces instruction compliance, and that
general RL reduces peak reasoning.

**Reasoning training lowers constraint following.** [[scaling-reasoning-losing-control-mathif]] builds MathIF:
420 math queries with 1-3 Python-verifiable output constraints. Across 23 reasoning models, the best hard
accuracy is 50.71% (Qwen3-14B). In the paper's controlled runs on four Qwen2.5 bases, long-CoT SFT and GRPO
lowered hard and soft constraint accuracy in 15 of 16 trained variants while raising average math accuracy in
15 of 16 (Table 4). Raising the RL maximum rollout length from 1k to 8k tokens raised math accuracy from 28.73
to 39.82 and lowered hard accuracy from 19.05 to 14.29 (Table 5). **Result (single study)**, with a
mechanism the authors test only through the constraint-repetition intervention.

**General RL lowers peak reasoning.** [[qwen-3-general-rl]] Table 22 tracks Qwen3-32B through reasoning RL,
thinking-mode fusion and general RL: IFEval strict prompt 73.0 → 78.4 → 85.0, Multi-IF 61.4 → 64.6 → 73.0,
ToolUse 63.3 → 70.4 → 85.5, against AIME'24 83.8 → 81.9 → 81.4 and LiveCodeBench v5 68.4 → 67.2 → 65.7. The
report states the trade directly: the degradation is conjectured to come from training on a broader range of
general tasks, and the team chose to accept it for overall versatility. One score per stage, no variance
reported.

**Cascade puts numbers on the same trade at the stage level.** In [[nemotron-cascade]], RLHF raises ArenaHard
from 76.9 to 93.1 on the 14B-Thinking model and lowers IFEval from 69.8 to 56.2; IF-RL then raises IFEval to
81.3 and IFBench from 25.6 to 40.4, and lowers ArenaHard to 90.2. ArenaHard never returns to its post-RLHF
value in the later stages. The report also notes that IF-RL reduces entropy and shortens reasoning traces
(§4.4.3), which connects this stage to the diversity controls in [[ch-43]].

**Hybrid versus separate models.** Qwen released a hybrid thinking / non-thinking model and then split it:
"we decided to stop using hybrid thinking mode. Instead, we'll train Instruct and Thinking models separately
so we can get the best quality possible" ([[qwen3-2507-instruct-thinking-split]]). The card's benchmark tables
show Thinking-2507 above the hybrid checkpoint's thinking mode (AIME25 81.5 → 92.3, Arena-Hard v2 61.5 → 79.7),
but the 2507 models changed in more than the mode split, so this is not a controlled comparison. The opposite
claim is made by [[nemotron-cascade]], whose 8B unified model is trained on parallel thinking and non-thinking
responses for the same prompts and has RLHF batches split evenly between modes: it ends at 89.5 AIME 2024 and
37.2 SWE-bench Verified against 88.8 and 38.5 for the dedicated 8B-Thinking model, with higher IFEval
(90.2 against 83.7) (Table 8); the unified model's IFEval is measured in non-thinking mode and its other
benchmarks in thinking mode (Table 8 caption). The report's ablation for the mode split (§6.1, Fig. 9) finds the half-and-half
batch better than either single-mode RLHF on ArenaHard, math and code. **Open question** — whether one model
can hold both modes at frontier scale without a reasoning cost is argued in both directions with no shared
benchmark protocol.

## §5 Stage order, forgetting, and cross-stage distillation

[Interactive figure: scores after each stage of sequential domain-wise RL](figures/cascade-stage-tracker.html)
— select a model and a comparison stage to see which benchmarks move at each stage of the Nemotron-Cascade
pipeline, and by how much.

### 5.1 What sequential RL does to earlier domains

[[nemotron-cascade]] is the only source here that publishes a full benchmark sweep after every stage. For the
14B-Thinking model: math and code scores from earlier stages survive the later ones within about a point
(AIME 2025 81.8 after RLHF, 82.3, 83.3, 83.5, 83.3; LiveCodeBench v6 72.3, 72.7, 72.4, 74.8, 74.6), while the
largest single-stage movements are all in alignment metrics (IFEval −13.6 at RLHF and +25.1 at IF-RL,
ArenaHard +16.2 at RLHF, IFBench +14.8 at IF-RL).
Two of Table 4's change arrows for this model disagree with the printed scores: the IFEval fall is marked
↓12.4 while Table 2's 69.8 and Table 4's 56.2 differ by 13.6, and the ArenaHard rise is marked ↑19.2 while
76.9 and 93.1 differ by 16.2. The arrows for the 8B models agree with their printed scores, so this chapter
uses the printed scores throughout.
SWE-bench Verified gains 4.3 points in RLHF (34.5 → 38.8), moves by at most 1.3 points across IF-RL, Math RL
and Code RL, and gains 3.5 points in its own stage (39.6 → 43.1). **Result (single study)**, one seed.

The report's explanation of why sequential RL forgets less than sequential SFT has four parts (§4.1.1,
**Interpretation**): RL data are generated by the current policy, so still-rewarded old behaviours keep being
sampled; RL optimizes expected reward rather than fitting token-level targets; forgetting is expected when a
new domain's reward conflicts with an old one, particularly for semantically similar prompts; and the pipeline
minimizes prompt overlap, ordering stages from general (RLHF, IF-RL) to specialized (math, code, SWE). This
connects to the on-policy argument in [[ch-38a]].

Two decisions in the same report are supported by measurements rather than argument. Math and competitive
programming prompts are removed from the RLHF prompt set; leaving math prompts in cost about 2% on AIME25
(§4.3.1). And the reward model's size changes what the policy learns. In a separate study whose policy is
AceReason-Nemotron-1.0-7B rather than a Cascade checkpoint, the 7B RM raised reward by producing longer
outputs and its ArenaHard gain shrank once style control was applied, while the 72B RM produced stable
response lengths and about 3% higher AIME25 (§6.2, Fig. 10). A reward model that is out of distribution for
the policy's outputs is a source of instability, which is the reward-model-size instance of
[[reward-model-overoptimization]].

### 5.2 Mixed stages need their own containment

The mixed structure has no stage boundary to protect earlier domains, so the containment is inside the stage.
[[deepseek-r1]] restricts general instruction data and preference rewards to the final 400 of 1,700 steps
because longer exposure to the model-based preference reward led to reward hacking, with reward rising while
Codeforces performance fell (§3.2.2, B.5, Fig. 6). [[kimi-k2]] adds a PTX loss on hand-selected high-quality
samples to the RL objective against forgetting (§3.2.3). [[glm-4-5]] runs general RL as the final stage in
four tracks and reports that fixing rare pathologies (language mixing, repetition, formatting) inside general
RL is sample-inefficient because they appear in "often less than 1% of outputs" (§3.4).

### 5.3 Cross-stage distillation as recovery

**Definition.** *On-policy distillation* trains the student on its own samples, scoring each token by a
teacher's log-probability rather than by an environment reward.

[[glm-5]] applies it across stages of its own pipeline. The report places the stage in two different ways:
§1 says the distillation is used "throughout" the sequential pipeline, while §3.5 runs it "as the final
stage"; the text does not reconcile the two. In the §3.5 description, the final checkpoints of preceding
stages become teachers, prompts are drawn from those teachers' RL training sets, and the GRPO advantage is
replaced by

```
Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ]
```

y_{i,t} is token t of sample i, x the prompt, π_teacher^infer the teacher's probability from the inference
engine, π_θ^train the current policy's probability under the training engine, and sg the stop-gradient
operator. Group size is set to 1 and batch size to 1,024, because the advantage no longer needs a group of
samples to estimate a baseline (§3.5).

**Worked example.** For a token the teacher assigns probability 0.60 and the student 0.20, the advantage is
log(0.60 / 0.20) = log 3 = +1.10, and the update raises the student's probability for that token. For a token
the teacher assigns 0.10 and the student 0.50, the advantage is log(0.2) = −1.61 and the update lowers it.
Tokens where the two agree contribute close to zero. The signal is dense (one value per token) and signed in
both directions, which is why a single sample per prompt carries enough information.

**Evidence for the mechanism, not for GLM-5's use of it.** [[glm-5]] reports no number isolating the
distillation stage. The closest measured comparison is [[qwen-3-general-rl]] Table 21 on Qwen3-8B: starting
from the same off-policy distilled checkpoint, RL reaches AIME'24 67.6 for 17,920 GPU hours, and on-policy
distillation reaches 74.4 for 1,800 GPU hours, with MMLU-Redux 86.9 against 88.3 and GPQA-Diamond 61.3 against
63.3. **Result (single study)** in one setting (math and code queries, one student size). [[glm-4-5]] uses the
same family of moves differently: three domain experts are trained separately and distilled into one model
through an SFT set of millions of expert samples (§3.1), and agentic RL alternates RL with self-distillation
(§3.3.2).

The design point for a general-purpose model: a sequential pipeline can treat each earlier stage's checkpoint
as a teacher for a cheap recovery pass, rather than re-running that stage's RL. What no public source gives is
the size of the recovery.

## §6 Safety and helpfulness in the same RL stage

Safety data belong to this chapter because they are another domain in the same mixture, with their own reward
model and their own interference pattern. DeepSeek-R1 trains a separate safety reward model on 106,000 prompts
with safe/unsafe labels and a point-wise loss next to the 66,000-pair helpful RM (§3.1), and adds 12,000
harmlessness questions to the general RL prompt set (B.3.1).

Two OpenAI reports describe the reward design in more detail.

**Deliberative alignment** ([[deliberative-alignment]]) teaches the model the text of the safety specification
and rewards reasoning over it. SFT data are produced by context distillation: the specification is placed in
the system prompt, completions are generated and filtered by a judge model that also has the specification,
and the specification is then stripped from the prompt, so the trained model must recall it. In the RL stage
the same judge supplies reward for safety prompts, and the chain of thought is hidden from the judge, which
the authors state avoids applying optimization pressure to the CoT and so avoids training deceptive CoTs.
Reported effects: StrongREJECT goodness@0.1 0.88 for o1 against 0.37 for GPT-4o, with XSTest non-overrefusal
0.93 against 0.88 (Table 1); removing safety data from either SFT or RL gives intermediate results (§4.1);
policy-retrieval accuracy in the CoT is 0.75 / 0.91 / 0.54 for hard refusal / safe completion / compliance
against 0.27 / 0.21 / 0.09 for an untrained baseline (Table 2). Generalization is the strongest claim: a model
whose safety data contain no non-English and no encoded examples scores 0.97 ± 0.02 on encoded jailbreaks and
0.69 ± 0.01 on multilingual jailbreaks, against 0.95 ± 0.03 and 0.68 ± 0.01 for the full-data model and
0.65 ± 0.06 and 0.44 ± 0.01 with no safety training (Table 3).

**Safe-completions** ([[openai-safe-completions]]) changes the reward shape from a refusal decision to a
product of two reward models:

```
r_i = h_i · s_i
```

s_i ∈ [0, 1] is policy adherence of the output (1 compliant, 0 a severe violation, intermediate values for
borderline low-severity cases); h_i ∈ [0, 1] is helpfulness, from one reward model that scores direct
helpfulness (fulfilling the stated task) and indirect helpfulness (alternatives, risk framing, a transparent
refusal) and is high if either is high.

**Worked example.** A detailed answer that violates the policy scores h = 0.9, s = 0 → r = 0. A high-level
answer that stays inside the policy scores h = 0.6, s = 1 → r = 0.6. A borderline answer scores
h = 0.9, s = 0.3 → r = 0.27. The ordering rewards the second response most, which is the mechanism the report
gives for replacing hard refusals: under a low s_i the only way to raise reward is indirect helpfulness.

Reported effects: in the controlled ablation (same architecture and post-training recipe, only the safety
paradigm differs), safety improves on dual-use prompts and helpfulness given a safe output rises by more than
1.0 point on a 1-4 scale on malicious prompts; in the production pair, GPT-5 Thinking improves safety over o3
by 9 and 10 percentage points on dual-use and malicious prompts. Among unsafe responses on 620 biorisk
prompts, the share graded high or moderate harm is 14.7% against 42.7% (§3.3). **Result (single study,
official)**; the underlying data, reward-model sizes and RL settings are not published.

The general lesson for reward design in this chapter: a multiplicative gate (safety times helpfulness) makes
one objective a hard constraint on the other without adding a separate penalty term, and it keeps a usable
gradient on the constrained side, unlike a binary refusal target.

## Negative samples and negative feedback

Multi-domain RL produces negatives in all four senses of the term ([[ch-43a]] holds the derivations).

**Where the negatives come from.** A verifier returns 0; a reward model returns a low scalar; a judge marks a
rubric item unsatisfied; a safety judge returns s = 0; a format or tool-call check halts the trace with reward
0 ([[glm-4-5]] §3.3.2). False-negative rates are reported by none of these sources; [[general-reasoner]] Table 5
is the closest measurement, showing a model-based verifier beating a rule-based one on every evaluation set,
which is indirect evidence that rule-based verifiers mark correct answers wrong outside mathematics.

**What practice does with them.**

1. *Negative marginal value* — discard. Rejection sampling in [[deepseek-r1]] keeps only correct responses for
   the 800K SFT set (B.3.3); [[glm-4-5]] drops trajectories that judge agents do not mark completed (§3.1).
2. *Negative as content* — keep and mask. [[glm-5]] retains erroneous segments inside otherwise good agent
   trajectories and masks them in the loss, "allowing the model to learn error correction behaviors without
   reinforcing incorrect actions" (§3.1).
3. *Negative as conditioning* — not used by the sources in this chapter.
4. *Negative as gradient* — the group-normalized advantage. In GRPO ([[grpo]]) every response below the group
   mean receives a negative advantage, so in a mixed stage the negative gradient is applied by whichever
   reward channel scored that prompt.

**Mechanism.** For a negative advantage on sampled token y, the update follows the softmax logit gradient
`∂ log p_y / ∂ z_j = 1[j = y] − p_j`: the probability removed from y is redistributed to the other tokens in
proportion to their current probabilities, so the most probable alternative absorbs most of it. Two
consequences appear in the recipes. First, penalizing a behaviour that is already rare is inefficient:
[[glm-4-5]] states that pathologies appear in less than 1% of outputs, so general RL rarely samples them
(§3.4). Second, the size of the push-down matters. [[nemotron-cascade]] sets the reward of a code-switched
response to the batch minimum minus 10 in RLHF, which guarantees the most negative advantage in the group
(§4.3.2); in Code RL the same team found that a −1 penalty for code switching hurt coding performance, and
used 0 instead, because with the penalty "the model produces incorrect answers without code-switching" when
every rollout in the group is wrong or mixed (§4.6.2). The same negative signal helped in one stage and hurt
in another.

**Controls used by these recipes.** Bound the penalty (0 rather than −1 for code switching); gate rather than
subtract (r = h · s in [[openai-safe-completions]], where s = 0 removes reward without pushing down a specific
token pattern); anchor with a positive term ([[kimi-k2]]'s PTX loss); keep the negatives on-policy (strictly
on-policy GRPO in [[nemotron-cascade]] §4.1.2); make sure the negative cannot dominate the positive signal
(sigmoid-bounded RM term under the verifier in §4.4.2); select pairs by margin rather than by score threshold
([[rlcf-checklists]] keeps the 40% of pairs with the largest per-criterion difference).

**Diagnostics.** Log reward and pass rate per domain, not only in aggregate; log the fraction of groups with
zero variance (no gradient); log chosen and rejected scores separately when the stage is preference-based;
track entropy and pass@k per domain ([[ch-43]]); track over-refusal and abstention on a safety suite
([[xstest]]), since a safety reward that only penalizes unsafe outputs moves the model toward refusal.

**Honesty about the size of the effect.** None of the multi-domain sources in this chapter measures what share
of the improvement comes from the negative-gradient half of the update in this setting. The chapter's claims
about negatives are therefore about failure modes and controls, not about attribution. **Open question.**

## Recipe

Values are quoted from the loci given; a value not printed by a source is not filled in from another one.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1 | 671B MoE | RL | mixed-stage steps; preference-reward window | 1,700 steps; general data and preference rewards only in the final 400 | arXiv:2501.12948v2 §3.2.2 | verified 2026-09-14 ([[deepseek-r1-recipe]]) | B.5 Fig. 6: with the helpful RM, reward rises while Codeforces pass@1 falls |
| DeepSeek-R1 | 671B MoE | RL | reward composition | Reward = Reward_reasoning (rule) + Reward_general (RM + format) + Reward_language | v2 §3.2.2 Eq. 8-10 | verified 2026-09-15 | no ablation reported |
| DeepSeek-R1 | 671B MoE | RL | general prompts | 66k helpfulness questions; 12,000 harmlessness questions | v2 B.3.1, Table 4 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 safety RM | R1 + reward head | reward-model | data; loss | 106,000 prompts labeled safe/unsafe; point-wise | v2 §3.1 | verified 2026-09-14 | no ablation reported |
| Qwen3 (32B and 235B-A22B) | 32B, 235B MoE | RL | General RL task coverage | "over 20 distinct tasks"; rule, reference-based judge (Qwen2.5-72B-Instruct), preference RM | arXiv:2505.09388v1 §4.4 | verified 2026-09-15 ([[qwen-3-general-rl]]) | §4.7 Table 22: stage-by-stage scores for Qwen3-32B |
| Qwen3-8B | 8B | distill-SFT | on-policy distillation vs RL, from the same checkpoint | 74.4 vs 67.6 AIME'24; 1,800 vs 17,920 GPU hours | §4.7 Table 21 | verified 2026-09-15 | Table 21 is the ablation |
| GLM-5 | 744B total / 40B active | RL | reasoning-RL algorithm settings | GRPO + IcePop, KL removed; β = 2; ε_low 0.2; ε_high 0.28; group 32; batch 32; fully on-policy | arXiv:2602.15763v2 §3.2 | verified 2026-09-15 ([[glm-5]]) | no ablation reported |
| GLM-5 | 744B / 40B act. | RL | general-RL reward types | rule-based functions + outcome RMs + generative RMs; human-written responses as style anchors; weights not printed | §3.4 | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B act. | distill-SFT | cross-stage on-policy distillation | teacher = preceding stages' final checkpoints; advantage = sg[log(π_teacher/π_θ)]; group size 1; batch 1,024 | §3.5 Eq. 2 | verified 2026-09-15 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | RL | holistic general-RL prompts | about 5,000 prompts over 7 primary, 33 secondary, 139 tertiary categories; RM trained on human preferences plus AI rubric scoring | arXiv:2508.06471v1 §3.4 | verified 2026-09-14 ([[glm-4-5-recipe]]) | no ablation reported |
| GLM-4.5 | 355B / 32B act. | RL | instruction-following RL | 7 major and 151 minor constraint types; rules + RM + critique model | §3.4, Fig. 9 | verified 2026-09-14 | Fig. 9: no clear reward hacking up to about 1,000 steps |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | non-verifiable reward | self-critique rubric reward: K2 critic ranks pairs against core, prescriptive and human rubrics; critic updated on verifiable-reward rollouts | arXiv:2507.20534v2 §3.2.2 | verified 2026-09-14 ([[kimi-k2-recipe]]) | App. F.3 lists over-confidence as a side effect; no ablation |
| Nemotron-Cascade 8B / 14B | 8B, 14B | reward-model | RM base; data; training | Qwen2.5-72B-Instruct + linear head, Bradley-Terry; 82K pairs (HelpSteer2 10K, HelpSteer3 36K after filtering, plus generated pairs); batch 256; LR 2e-6; 1 epoch | arXiv:2512.13607v2 §4.2 | verified 2026-09-15 ([[nemotron-cascade]]) | §6.2: 72B RM gives about 3% higher AIME25 than a 7B RM and stabler response length |
| Nemotron-Cascade 8B / 14B | 8B, 14B | RL | RLHF settings | max response 12K, no overlong filtering; 8 rollouts; temperature 0.6; top-p 0.95; LR 2e-6; entropy and KL coefficients 0; about 800 steps | §4.3.2 | verified 2026-09-15 | §4.3.3, Table 4: gains on every benchmark except IFEval |
| Nemotron-Cascade 8B / 14B | 8B, 14B | RL | RLHF batch size; steps | body text: batch 128, about 800 steps; App. Table 15: batch 256, 800 (8B) and 900 (14B) steps | §4.3.2 vs Table 15 | conflict | the report does not say which values produced the released checkpoints |
| Nemotron-Cascade 14B-Thinking | 14B | RL | IF-RL reward | r_i = R_IF + sigmoid(R̂_RM) if R_IF = 1, else 0, with R̂_RM group-normalized | §4.4.2 | verified 2026-09-15 | §4.4.2: a rule-based verifier alone degraded human-alignment results (no numbers) |
| Nemotron-Cascade 14B-Thinking | 14B | RL | Math RL stages | 28K → 40K max response; LR 2.5e-6; temperature 1.2 → 1.1; steps 0-120 and 120-220; 8 rollouts; batch 128 | App. Table 18 | verified 2026-09-15 | Table 6: AIME 2024 89.2 → 90.4 |
| Nemotron-Cascade 14B-Thinking | 14B | RL | Code RL | 56K max response; LR 4e-6; 64 steps; temperature 1.0; code-switch reward 0 (not −1) | App. Table 19; §4.6.2 | verified 2026-09-15 | §4.6.2: a −1 code-switch penalty lowered coding performance |
| GURU-7B / GURU-32B | 7B, 32B | RL | mixed-domain GRPO | LR 1e-6, 10-step warm-up; 512 prompts per step; 16 samples per prompt at temperature 1.0; mini-batch 64; 4K prompt / 8K generation; no KL, no entropy loss; ε = 0.2 | arXiv:2506.14965v1 §4.1 | verified 2026-09-15 ([[guru-cross-domain-rl]]) | Table 3: 43.29 and 54.24 average over 17 benchmarks |
| Nemotron-CrossThink-7B | 7B | RL | domain blend | 2:1 general-purpose reasoning to math | arXiv:2504.13941v3 §5, Table 3 | verified 2026-09-15 ([[nemotron-crossthink]]) | Table 3: 58.12 average against 57.82 (math-only) and 53.30 (general-only) |
| RaR policies from Qwen2.5-7B | 7B | RL | rubric reward | GRPO; 16 samples per prompt; LR 5e-6; 300 steps; judge gpt-4o-mini; weights Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9 | arXiv:2507.17746v2 §4.2, §4.4, Table 10 | verified 2026-09-14 ([[rubrics-as-rewards]]) | Fig. 2: HealthBench 31.2 vs 25.5 for a Likert reward |
| RLCF from Qwen2.5-7B-Instruct | 7B | preference | checklist feedback | Qwen2.5-72B-Instruct checklists; mean of 25 judge samples per item; keep top 40% of pairs by per-criterion difference; DPO 2 epochs, batch 1024, cosine LR 3e-6 → 2e-6 | arXiv:2507.18624v2 §3, §4.1 | verified 2026-09-15 ([[rlcf-checklists]]) | Tables 2-4: positive on all five benchmarks; reward-model baselines are mixed |
| DeepSeek-GRM-27B | 27B | reward-model | SPCT | RFT LR 5e-6, batch 1024; RL LR 4e-7, batch 512; 900 steps each; KL β = 0.08; N_RFT = 3 | arXiv:2504.02495v3 App. C.1 | verified 2026-09-15 ([[deepseek-grm]]) | App. D: β grid {0, 0.01, 0.02, 0.08}; smaller values collapse on some subsets |

**Starting point for a small general-purpose run.** Every number here comes from a `verified` row above, with
the conditions under which it was used. For a 7B-class policy trained on mixed reasoning domains, the
[[guru-cross-domain-rl]] configuration is the one with a published multi-domain result: GRPO, LR 1e-6, 512
prompts per step, 16 samples per prompt at temperature 1.0, 8K generation limit, no KL and no entropy term,
ε = 0.2, on 20 nodes of 8 Hopper GPUs for about 3 days at 7B. For a general (non-verifiable) stage on top of
it at the same size, [[rubrics-as-rewards]] used GRPO with 16 samples per prompt, LR 5e-6, 300 steps and a
gpt-4o-mini judge on one 8×H100 node, with rubrics generated against reference answers. Neither configuration
was tuned for the other's data, and no source reports a joint configuration at this scale.

## Generalization lens

**(a) What increases breadth.**
- Prompts from domains the model will be asked about. Underrepresented domains gain from in-domain data and
  not from cross-domain data ([[guru-cross-domain-rl]] §3.1); adding code data to a math-plus-puzzle mixture
  removes a 22.56-point code collapse ([[multi-domain-rlvr-data-centric]] Table 9).
- A verifier that accepts correct answers in unusual formats. The model-based verifier beats rule matching on
  all four evaluation sets and does not plateau at step 60 ([[general-reasoner]] Table 5, Fig. 4).
- Instance-specific rubrics or checklists instead of one global quality score ([[rubrics-as-rewards]] Fig. 2;
  [[rlcf-checklists]] Tables 2-4).
- A quality term next to a verifier term, so that constraint satisfaction does not become the only objective
  ([[ifbench]] App. E; [[nemotron-cascade]] §4.4.2).
- Training safety as reasoning over a written specification, which transferred to jailbreak formats absent
  from the safety training data ([[deliberative-alignment]] Table 3).
- Mode coverage inside one model: RLHF batches split evenly between thinking and non-thinking prompts scored
  highest on ArenaHard, math and code in the report's own comparison ([[nemotron-cascade]] §6.1).

**(b) What causes narrowing or forgetting.**
- Optimizing a preference reward model for many steps: reward rises while a held-out capability falls
  ([[deepseek-r1]] B.5, Fig. 6), which is why the preference window is 400 of 1,700 steps.
- A rule-only verifier on a narrow constraint taxonomy: AlpacaEval 2 33.5 → 21.3 ([[ifbench]] Table 3).
- Aggressive difficulty filtering inside one domain: −9.2 on HumanEval, −3.0 on HiTab with accuracy collapse
  during training ([[guru-cross-domain-rl]] Table 2).
- Single-domain RL: math-only training lowers code from 67.46 to 64.23, and CountDown-only training to 29.59
  ([[multi-domain-rlvr-data-centric]] §3.1).
- Reasoning-oriented SFT and RL reducing constraint following in 15 of 16 controlled variants
  ([[scaling-reasoning-losing-control-mathif]] Table 4), and general RL reducing peak reasoning
  ([[qwen-3-general-rl]] Table 22).
- A prompt distribution that ignores a capability: RLCF on assistant and writing prompts moves GSM8K 83.2 →
  82.2 and TruthfulQA MC1 43.5 → 42.0 ([[rlcf-checklists]] Table 9); GURU-7B is below two narrower baselines
  on IFEval ([[guru-cross-domain-rl]] Table 3).
- A rubric that bans a surface pattern also bans its legitimate uses: hedging penalties leading to overstated
  certainty ([[kimi-k2]] App. F.3).

**(c) How to measure it for this stage.**
- Per-domain reward and per-domain accuracy during training, not only the aggregate reward.
- A held-out suite covering domains that are *not* in the prompt mixture. [[guru-cross-domain-rl]] evaluates
  IFEval and LiveBench as unseen domains; [[rlcf-checklists]] evaluates XSTest, GSM8K and TruthfulQA.
- A fixed evaluation cadence and an explicit checkpoint rule: 13 online tasks every 10 steps, with the
  best-average checkpoint selected ([[guru-cross-domain-rl]] §3.1). A stage-level sweep across every benchmark
  after every stage is the sequential-pipeline version ([[nemotron-cascade]], Tables 2-8).
- Judge independence: [[rubrics-as-rewards]] uses gpt-4o-mini both as the training reward and as the
  evaluation judge, which leaves judge agreement and task quality confounded. When a judge supplies the
  training reward, report the stage's evaluation with a different judge or with a programmatic metric.
- A fixed averaging convention and per-domain numbers next to it (§3.4), and a fixed chat template across
  compared checkpoints ([[multi-domain-rlvr-data-centric]] Table 11).
- Contamination control for the benchmarks that decide the mixture: 9-gram filtering of math RL data and
  evaluation only on problems released after the training cutoff ([[nemotron-cascade]] §4.5.1, §5).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Using one aggregate reward curve for a mixed-domain stage | Total reward rises while one domain's accuracy falls; the cause is invisible in the curve | Log reward and pass rate per data source; compare per-domain eval before and after the stage |
| Verifier-only reward for instruction following | Constraint metrics rise, open-ended quality metrics fall (AlpacaEval 2 33.5 → 21.3 in [[ifbench]]) | Add a general-quality eval to the in-loop suite; inspect responses with the constraint removed |
| A reward model far out of distribution for the policy's outputs | Response length grows with reward; style-controlled scores lag uncontrolled ones ([[nemotron-cascade]] §6.2) | Track mean response length against reward; report style-controlled ArenaHard; use a larger RM |
| Preference rewards applied for the whole stage | Reward rises steadily while a verifiable benchmark falls ([[deepseek-r1]] Fig. 6) | Cap the preference-reward window; alternate with verifiable prompts; watch a code or math eval |
| Comparing mixtures by one average | Two mixtures swap ranks under a different averaging convention (§3.4) | Fix the convention, publish per-domain numbers, and count benchmarks per domain |
| Evaluating checkpoints under different chat templates | Large unexplained drops on a subset of tasks ([[multi-domain-rlvr-data-centric]] Table 11) | Pin one template per model family; run a template-matched control before drawing conclusions |
| Difficulty-filtering every domain by default | In-domain gains with collapse on easy out-of-domain tasks ([[guru-cross-domain-rl]] §3.3) | Keep a difficulty spread, or add cross-domain prompts; watch easy held-out tasks during training |
| Penalizing a rare pathology inside general RL | No measurable change; the behaviour appears in under 1% of rollouts ([[glm-4-5]] §3.4) | Measure the pathology rate first; if it is rare, target it with selected prompts or with SFT data |
| A large negative reward for a surface property | Accuracy falls in the affected domain while the property disappears ([[nemotron-cascade]] §4.6.2) | Compare penalty magnitudes (0 against −1) on a held-out split before adopting one |
| Training reward judge = evaluation judge | Gains appear on the judge's metric and not on independent ones ([[rubrics-as-rewards]] Fig. 2 caption) | Evaluate with a different judge or a programmatic metric |

## Check your understanding

1. In [[nemotron-crossthink]] the best mixed blend beats the math-only blend by 0.30 points on average but
   loses 5.0 points on AMC 23. Explain why a lab building a general-purpose model would still take the mixed
   blend, and name the evaluation result that would change that decision.
2. [[nemotron-cascade]] removes math and competitive-programming prompts from its RLHF stage. Derive the
   failure it is preventing from the properties of the reward model, and predict what would happen to the
   AIME25 score if those prompts were left in.
3. The IF-RL reward is `R_IF + sigmoid(R̂_RM)` when the verifier passes, and 0 otherwise. Explain why the
   sigmoid and the group normalization are both necessary, and what would break if the RM score were added
   unbounded.
4. [[guru-cross-domain-rl]] finds that difficulty filtering helps in-domain and hurts easy out-of-domain
   tasks. Give a mechanism for that asymmetry in terms of what the policy samples during training, and say
   what you would log to test it.
5. GLM-5 replaces the group-normalized advantage with `sg[log(π_teacher/π_θ)]` in its final stage. Explain why
   this allows a group size of 1, and why the same substitution would be a poor choice for the reasoning RL
   stage itself.
6. Safe-completions multiply helpfulness by safety instead of subtracting a penalty for unsafe content. Work
   through what the policy is pushed toward under each of the two designs when a response is highly helpful
   and moderately unsafe.
7. [[rubrics-as-rewards]] uses the same judge for the training reward and for the evaluation. Describe an
   experiment that separates "the policy got better at the task" from "the policy got better at agreeing with
   this judge".
8. Qwen3 accepted a 2.4-point AIME'24 loss in exchange for general-RL gains, while Qwen later split the hybrid
   model into separate Instruct and Thinking checkpoints. State what evidence would be needed to decide
   whether the split was the right response to that trade.

## Connections

- Previous chapter: [[ch-44a]] — Length in RL: Overlong Responses, Length Control, and Long-Context RL. The
  length budgets and overlong-filtering choices from that chapter are what a multi-domain stage has to set
  once per domain.
- Dependency: [[ch-38a]] — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output
  Diversity. The forgetting argument in §5.1 is the multi-domain instance of that chapter's result.
- Next chapter: [[ch-45]] — Self-Improvement Loops and Multi-Stage Reasoning Pipelines. The expert-model and
  self-distillation moves in §5.3 are the entry point to those loops.
- [[ch-41]] — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization: the reward
  models used as one channel here.
- [[ch-42]] — Reward Hacking and Judge Design: the failure mode that rubric, checklist and critic rewards are
  built against.
- [[ch-43]] — Entropy, Output Diversity, and KL Control in RL: entropy and pass@k are the diversity
  diagnostics named in this chapter's negative-feedback section.
- [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative
  Advantages: the derivations behind this chapter's negative-feedback section.
- [[ch-46]] — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention: the
  held-out retention measurement designed here is run there.

## Sources

- [[deepseek-r1]], [[deepseek-r1-recipe]] — the mixed final RL stage: reward composition (Eq. 8-10), the
  400-step preference window, the helpful and safety reward models, and the reward-hacking evidence.
- [[qwen-3]], [[qwen-3-general-rl]] — General RL over 20+ tasks, the three reward types, the stage-by-stage
  table for Qwen3-32B, and the on-policy distillation comparison.
- [[glm-4-5]], [[glm-4-5-recipe]] — expert-model iteration and distillation, holistic and instruction-following
  general RL, the rarity of pathologies.
- [[glm-5]] — sequential reasoning / agentic / general RL, the three-part hybrid reward system, and the
  cross-stage on-policy distillation objective.
- [[kimi-k2]], [[kimi-k2-recipe]] — the self-critique rubric reward, the closed-loop critic update, the PTX
  loss, and the stated over-confidence side effect.
- [[nemotron-cascade]] — the cascaded pipeline, per-stage benchmark tables, the combined IF and RM reward, the
  RM-size study, and the mode-split ablation.
- [[rubrics-as-rewards]] — rubric rewards: aggregation formulas, weights, HealthBench and GPQA results, and
  the judge-alignment study.
- [[rlcf-checklists]] — checklist feedback: generation, judge-plus-program scoring, DPO pairs, benchmark
  results, and the non-target-task regressions.
- [[deepseek-grm]] — SPCT, inference-time scaling of a generalist generative reward model, and its limits as
  an online RL reward.
- [[guru-cross-domain-rl]] — cross-domain transfer asymmetry, the difficulty-filtering ablation, the
  17-benchmark results, and the pass@k analysis.
- [[general-reasoner]] — diverse versus math-only training data, and the model-based versus rule-based
  verifier ablation.
- [[nemotron-crossthink]] — blend ratios, the per-benchmark split against math-only training, and the
  token-efficiency measurement.
- [[multi-domain-rlvr-data-centric]] — all seven domain combinations on one base model, and the template
  sensitivity table.
- [[scaling-reasoning-losing-control-mathif]] — MathIF and the controlled runs showing reasoning training
  lowering constraint following.
- [[qwen3-2507-instruct-thinking-split]] — the official statement ending hybrid thinking mode, and the
  model-card tables.
- [[deliberative-alignment]] — specification-based safety reward, stage ablation, policy retrieval, and
  out-of-distribution generalization.
- [[openai-safe-completions]] — the r = h · s reward, the controlled and production comparisons, and the harm
  severity analysis.
- [[ifbench]] — verifiable instruction following, constraint-generalization ablations, and the reward-model
  mixing result.
- [[grpo]] — the group-normalized advantage used by almost every recipe in this chapter.
- [[reward-model-overoptimization]], [[xstest]] — background for the RM-size and over-refusal diagnostics.
