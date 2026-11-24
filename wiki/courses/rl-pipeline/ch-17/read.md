<!-- chapter: ch-17
     track: rl-pipeline
     kind: content
     title: Multi-Domain RL Recipes: Stage Order and Mixture
     deps: [ch-14, ch-15, ch-16]
     sources: [[mai-thinking-1]], [[glm-4-5-recipe]], [[glm-4-5]], [[qwen-3-recipe]], [[qwen-3]], [[kimi-k2-recipe]], [[kimi-k2]], [[deepseek-v3-2]], [[deepseek-v3.1]], [[olmo-3-recipe]], [[olmo-3-rl-pipeline]], [[minimax-m1]], [[nemotron-ultra-recipe]], [[mai-rl-infrastructure]], [[recipe-crosswalk]], [[rlvr-tulu3]], [[tulu-3]], [[magistral]], [[open-instruct-rlvr-pipeline]], [[agentrl]], [[scalerl]], [[scaling-reasoning-losing-control-mathif]], [[ifbench]], [[general-reasoner-verifier]]
     figures: figures/stage-recipes.html
-->

# Chapter 17 — Multi-Domain RL Recipes: Stage Order and Mixture

> **Core insight.** None of the eight lab recipes carded here orders RL as instruction following → reasoning → search/agentic → general alignment. The observed orders fall into three patterns. (1) **Reasoning first, general last:** reasoning RL comes first, then a general stage that holds IF, preference alignment and often agent tasks (Qwen3, MiniMax-M1; GLM-4.5 runs reasoning, agent and general as separate experts). (2) **Parallel specialists, then merge:** domain specialists are each trained with RL, merged by SFT or distillation, and finished with a short or mixed RL stage (MAI Thinking 1, DeepSeek-V3.2, GLM-4.5). (3) **One joint stage:** all verifiable domains train together in one RL run (Kimi K2, Nemotron 3 Nano's RLVR, Olmo 3's OlmoRL). IF is never a separate first stage. It sits with general or alignment RL, or inside the joint mixture. The labs that state reasons cite forgetting: DeepSeek-V3.2 merges domains to avoid "catastrophic forgetting issues commonly associated with multi-stage training paradigms" ([[deepseek-v3.1]] §3), and Nemotron trains all environments together because single-environment training "often results in un-recoverable degradation of other benchmarks" ([[nemotron-ultra-recipe]] NR §3.2). Neither prints a staged-versus-joint ablation, and no card does ([[recipe-crosswalk]] Guideline).
>
> **Guideline.** When choosing a stage order, treat the interference measurements as the evidence and the lab orders as practice. Keep each domain's data in every later stage that could erode it, because MAI keeps "a small proportion of STEM and coding data" in its final RL since reasoning otherwise degrades ([[mai-thinking-1]] Sec. 3.5), and MathIF measured reasoning RL lowering constraint compliance in 15 of 16 variants ([[scaling-reasoning-losing-control-mathif]] Table 4). When domains with different lengths share one RL loop, normalize advantages per task (65.0 vs 59.4 without, [[agentrl]] Table 6) and set mixture weights in tokens, not prompts, because MAI's consolidation mixture is 56% STEM + coding by sample but 89% by token ([[mai-thinking-1]] Table 10). When the framework allows it, make mixture weights runtime configuration (Rocket applies them at step or sampling boundaries, [[mai-rl-infrastructure]] p. 12–13), because no recipe prints a mixture-weight ablation, so the weights will need adjustment during the run.

---

## Why this chapter matters

ch-14 to ch-16 built one pipeline per domain. A general model needs all of them, plus preference alignment, in one set of weights. Two decisions remain: the **order** of domains across stages, and the **mixture** inside a stage. Both decide which capabilities survive to the release.

The outline asks how labs order IF → reasoning → search/agentic → alignment. §1 answers with each lab's stage order as reported. §2 collects what was measured about interference and retention. §3 to §5 cover the mechanics that make joint training possible: per-domain graders in one loop, mixture weights as configuration, and duration differences between domains. §6 covers consolidation by distillation, which several labs use as the stage boundary.

The applied target of this course needs several of these domains at once: IF (format and policy constraints), multi-turn tool use (CRM lookups, scheduling), and a subjective style reward (ch-19, ch-23). The capstone (ch-22) chooses a stage order for it using this chapter's evidence.

---

## §1 Each lab's stage order

The figure [figures/stage-recipes.html](figures/stage-recipes.html) shows these orders as side-by-side timelines. Filter by domain to see where IF, reasoning, agentic and alignment training appear in each recipe.

| Lab | Reported order (post-training) | Where IF sits | Where search / agentic sits | Where preference alignment sits | Source |
|---|---|---|---|---|---|
| **MAI Thinking 1** | three specialist climbs with one shared recipe: STEM + competitive code (longest); agentic coding + tool use, run jointly with STEM; helpfulness + safety → consolidation SFT → short final RL at 128k keeping some STEM/code | helpfulness + safety climb (IF primary in that climb's lexicographic reward) | agentic climb (SWE, tool use); no web-search climb | helpfulness + safety climb | [[mai-thinking-1]] Fig. 12, Sec. 3.2–3.5 |
| **GLM-4.5** | per-expert cold-start SFT → RL for Reasoning, Agent and General-chat experts → distill experts into one model by SFT; general RL is described as the final stage | general RL (7 major / 151 minor constraint types) | agent expert RL (web search, SWE) | general RL (holistic RL, about 5,000 prompts) | [[glm-4-5]] §3, §3.4; [[recipe-crosswalk]] |
| **Qwen3 (flagship)** | long-CoT cold start → Reasoning RL (3,995 query–verifier pairs, 170 steps for 235B) → Thinking Mode Fusion (SFT) → General RL (>20 tasks) | general RL | general RL (agent with real execution feedback, RAG-style tasks) | general RL (preference alignment) | [[qwen-3]] §4.1–4.4 |
| **Kimi K2** | SFT → one joint RL stage | joint | joint (coding, SWE, agentic tool use) | joint (self-critique rubric reward) | [[kimi-k2]] §3.2 |
| **DeepSeek-V3.2** | specialists, each with large-scale RL (math, programming, logical reasoning, general agentic, agentic coding, agentic search, writing, general QA) → distillation → one mixed RL stage ("thousands of steps") | mixed RL (general tasks with rubric GenRM) | specialists, then mixed RL | mixed RL | [[deepseek-v3.1]] §3 |
| **Olmo 3 Think** | SFT → DPO (delta-learning pairs) → OlmoRL (math, code, IF, chat in one mix); Olmo 3.1 Think 32B continues the best RL run 21 more days on 224 GPUs | inside OlmoRL | none | DPO before RL; chat inside OlmoRL with an LM judge | [[olmo-3-rl-pipeline]] §4.4; [[olmo-3-recipe]] |
| **MiniMax-M1** | CPT → long-CoT SFT → RL run 1 (40K output): rule-verified reasoning first, general tasks mixed in gradually → RL run 2: output window 40K → 80K | general tasks, mixed in later | SWE inside the reasoning set from the start | general tasks with GenRM | [[minimax-m1]] §4.3, §5 |
| **Nemotron 3 Nano** | SFT → RLVR (all environments at once) → RLHF (GenRM) → RLVR | both RLVR stages (46K IF + 3K multi-turn IF) | both RLVR stages (Workplace Assistant, banking agent) | RLHF between the two RLVR stages | [[nemotron-ultra-recipe]] NR §3.2–3.3 |
| **Tülu 3** (reference) | SFT → length-normalized DPO → RLVR (PPO, math + IF) | last stage (RLVR) | none | DPO before RLVR | [[rlvr-tulu3]]; [[tulu-3]] |
| **Magistral** (reference) | Medium: RL only, from Mistral Medium 3 Instruct. Small: SFT on Medium traces → RL | not trained in RL (internal IFEval 86.8 → 87.4 after math+code RL) | none | none | [[magistral]] §5, §7.3 |

**Ambiguity noted.** For GLM-4.5, the cards record both a General-chat expert trained with RL in stage 1 and a statement that "general RL" is "the final stage of post-training" ([[glm-4-5]] §3, §3.4). The cards do not settle whether the final general RL runs on the unified model or is the General-chat expert's RL. The figure shows the two readings.

**Answer to the outline question.** No lab trains IF before reasoning. Where stages are sequential, the order is reasoning first and general (IF, preference, often agentic) after: Qwen3, MiniMax-M1, Tülu 3 (where the "reasoning" is math RLVR together with IF in the last stage), and Nemotron, where RLHF follows the first RLVR stage. Where specialists run in parallel (MAI, DeepSeek-V3.2, GLM-4.5), the order is set by the merge step, not by a domain sequence. Search appears as its own specialist only in DeepSeek-V3.2 and GLM-4.5's agent expert. MAI has no web-search climb, Olmo 3 has no search RL, and Qwen3 places "RAG-style" tasks inside general RL.

---

## §2 What was measured about interference and retention

Statements in the recipes are of three kinds, and this section keeps them separate.

### §2.1 Stated reasons without numbers

| Lab | Statement | Location |
|---|---|---|
| DeepSeek-V3.2 | merging domains into one RL stage balances them while "circumventing the catastrophic forgetting issues commonly associated with multi-stage training paradigms"; no ablation printed | [[deepseek-v3.1]] §3 |
| Nemotron 3 Nano | single-environment training "often results in un-recoverable degradation of other benchmarks"; no numbers | [[nemotron-ultra-recipe]] NR §3.2 |
| MAI Thinking 1 | joint agentic + STEM "helps to stabilize the RL climb"; agentic tasks transfer "neither positively nor negatively" to STEM | [[mai-thinking-1]] p. 40 |
| MAI Thinking 1 | final RL keeps "a small proportion of STEM and coding data" because reasoning otherwise degrades | Sec. 3.5, p. 49 |
| Kimi K2 | PTX loss on hand-selected high-quality samples to prevent forgetting and reduce overfitting to the RL task set; the authors state it improves generalization; no ablation | [[kimi-k2]] §3.2.3 |
| GLM-4.5 | RL on web-search and SWE "leads to generalized performance improvements" on general tool use and Terminal-Bench; no numbers | [[glm-4-5]] §3.3.2 |
| Olmo 3 Instruct | a warm-up on math, then mixed data without math, was "suboptimal" | [[olmo-3-rl-pipeline]] §5.4, footnote 56 |

### §2.2 Measured, within one lab's pipeline

- **Qwen3-32B, Table 22.** Thinking Mode Fusion and General RL did not improve knowledge, STEM, math or coding relative to the Reasoning RL checkpoint ([[qwen-3]] §5.3 point 3). The later stages added general and IF capability without raising reasoning scores. The card does not record a loss on reasoning, only no gain.
- **Olmo 3 7B Think, Table 24.** IFEval / IFBench: SFT 77.9 / 30.0, DPO 75.9 / 28.3, RL 88.2 / 41.6. The RL stage held IF prompts next to math and code, and IF rose ([[olmo-3-rl-pipeline]] Table 24). There is no RL run without IF to compare against.
- **Magistral, Table 5 (cross-domain).** From one 24B checkpoint at AIME'24 32.2 / LiveCodeBench v5 22.7: math-only RL reached 62.5 / 38.3, and code-only RL reached 49.7 / 42.7 ([[magistral]] §6.1). Each single-domain run improved the other domain, which is positive transfer between math and code.
- **ScaleRL, Fig. 11.** Joint math + code training gave parallel per-domain trends ([[scalerl]] §5). No interference is visible in the fitted curves.

### §2.3 Measured, in controlled studies outside lab recipes

- **Reasoning RL erodes IF.** Long-CoT SFT and GRPO on math lowered constraint compliance in 15 of 16 variants ([[scaling-reasoning-losing-control-mathif]] Table 4).
- **IF RL erodes general quality.** IF-RLVR from Tülu-DPO lowered AlpacaEval 2 from 33.5 to 21.3 ([[ifbench]] Table 3).
- **Mixed domains beat math-only off-domain.** SuperGPQA 34.2 vs 29.8 at 7B, 39.5 vs 35.6 at 14B ([[general-reasoner-verifier]] Table 4).
- **Multi-environment agents.** One policy trained on five environments averaged 67.7 against 67.8 for the best single-task policies, while single-task policies fail elsewhere (DB policy 0.2 on ALFWorld) ([[agentrl]] Table 4).

### §2.4 Reading the evidence

Interference is measured between domains that look far apart: math and IF, IF and chat quality, one agent environment and another. Between math and code the measured transfer is positive (Magistral) or neutral (ScaleRL). That pattern fits the lab practice of grouping math and code in one reasoning stage and keeping IF and alignment together. No study varies stage order while holding data and compute fixed, so "staged vs joint" is undecided by measurement. The stated preference for joint training (DeepSeek, Nemotron, Kimi) and the practice of retention data in late stages (MAI) both address the same measured effect: a domain absent from a stage loses ground in that stage.

---

## §3 Per-domain graders under one loop

A joint stage needs every row to carry its grader.

**open-instruct.** Each row's `dataset` tag selects a verifier. `build_all_verifiers` registers every `VerifierFunction` subclass under its lower-cased name, and adds a stdio variant of the code verifier (`ground_truth_utils.py` L1239–1259). At scoring time the dispatch loop looks the tag up per row and sums weighted scores:

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L1323-L1336
for gt, ds in zip(ground_truth_list, dataset_list):
    reward_func = reward_fn_mapping.get(ds.lower())
    if reward_func is None:
        logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
        continue

    task = reward_func.async_call(
        tokenized_prediction=tok_prediction,
        prediction=prediction,
        label=gt,
        query=query,
        rollout_state=rollout_state,
    )
    async_tasks.append(task)
```

An unknown tag logs a warning and skips the reward, so that row's reward stays 0 (`response_rewards = [0] * len(responses)`, L1345). A typo in a dataset tag makes a whole domain silently score 0. Monitor reward per tag, not only the overall mean. The Olmo 3 scripts set all verifier weights to 1.0, and no per-domain reward weight was found ([[open-instruct-rlvr-pipeline]] Verification).

**Rocket.** Grading is split by scope. Per-rollout graders run on rollout workers. Group graders (for example a pairwise LLM comparison) run on problem workers, which also compute GRPO advantages ([[mai-rl-infrastructure]] Fig. 1; [[mai-thinking-1]] Sec. 3.6). A pairwise judge that needs the whole group therefore runs after the group completes, which ties grading latency to the slowest rollout of the group.

**Scale normalization across graders.** Domains whose rewards have different scales or variances get different update sizes. open-instruct multiplies every verifier score by `verification_reward` (default 10.0, L1371). Judge scores are divided by 10 (ch-14 §5). MAI combines several signals per domain by lexicographic priority or gating ([[mai-thinking-1]] Sec. 3.4.1). AgentRL normalizes advantages per task, and removing that lowered the average from 65.0 to 59.4 ([[agentrl]] Table 6).

---

## §4 Mixture weights as runtime configuration

**Rocket.** The controller selects each problem "from the curriculum" (Fig. 1, step 1). Runtime settings are typed and validated, and they include inference capacity and concurrency, admission, staleness limits, batch packing and mixture weights. They are applied at step or sampling boundaries ([[mai-rl-infrastructure]] p. 12–13). Table 1 of the same post counts 123 runtime-elasticity events in the measured portion of training, each estimated to save 2 hours (246 h, 9.88% of 2,490 h). That count covers all runtime changes, not only mixture changes; the post does not break it down.

**Open alternatives.**

| System | How the mixture is set | Can it change mid-run? |
|---|---|---|
| open-instruct (Olmo 3) | `--dataset_mixer_list` proportions or counts; Olmo 3 passes one pre-mixed dataset at 1.0 (`7b_think_rl.sh` L28) | no (restart with a new mix) |
| Nemotron 3 Nano | fixed domain ratio per batch (ratios not printed); per-domain Gaussian target pass-rate distribution whose mean falls linearly; re-profiling with the best checkpoint when progress plateaus | the curriculum moves the pass-rate target within each domain; the domain ratio is fixed ([[nemotron-ultra-recipe]] NR §3.2.2) |
| MiniMax-M1 | rule-verified reasoning first, then general tasks mixed in "gradually" with dynamic weighting | yes (mechanism not described) ([[minimax-m1]] §4.3) |
| MiniMax-M1 run 2 | data re-filtered by the 40K model's pass rates; harder math/code upweighted; synthetic reasoning downsampled because its outputs "become repetitive and homogenous" and destabilize long-context RL | between runs ([[minimax-m1]] §5) |

**Weights in prompts vs tokens.** MAI's consolidation SFT mixture is 56 / 11 / 33% by sample and 89 / 9 / 2% by token for STEM + coding / agentic / helpfulness + safety ([[mai-thinking-1]] Table 10). Under a token-level loss over the global batch (MAI's RL objective, Sec. 3.1.1), a domain's share of the gradient is its token share. A helpfulness prompt with a 500-token answer and a STEM prompt with a 30,000-token answer differ by 60× in gradient weight at equal prompt count (illustrative lengths). State mixture weights in both units.

**No ablation.** No recipe carded here prints an ablation of mixture weights ([[recipe-crosswalk]] Guideline; [[olmo-3-rl-pipeline]] Verification: "Not reported: per-domain reward weights in the final mix"). Weights are set by judgment and adjusted. That is the practical reason to make them runtime-changeable.

---

## §5 Duration differences between domains in one loop

| Domain (Rocket production mix) | Median duration | Median turns |
|---|---|---|
| instruction following | 2 min | 1 |
| math & science | 7 min | 1 |
| tool use | 12 min | 13 |
| competitive coding | 34 min | 1 |
| web research | 51 min | 89 |
| software engineering | 93 min | 249 |

Source: [[mai-rl-infrastructure]] Fig. 2.

A joint stage mixes rollouts whose medians differ by 46× (2 min vs 93 min). Three consequences follow:

1. **Composition drift within a step.** If the learner trains on whatever has finished, short domains are over-represented in early batches and long domains arrive later, from older weights. The realized mixture per step then differs from the configured mixture. Rocket's priority by rollout start time and its packing of training segments (ch-04) act on this. A synchronous loop (Nemotron's "synchronous GRPO", NR §3.2.5) avoids drift by waiting, at the cost of idle time.
2. **Staleness by domain.** Long-domain samples are staler on average. MAI drops rollouts more than 8 updates (40 steps) stale ([[mai-thinking-1]] hyperparameter table). Under such a cutoff the longest domain loses the most samples, which lowers its realized weight.
3. **Infra placement.** GLM-4.5 uses synchronous colocated slime for general and reasoning RL, and asynchronous disaggregated slime for agentic SWE ([[recipe-crosswalk]]; [[glm-4-5]] §3.5). Splitting the domains into different stages lets each use its own scheduler. This is a systems reason for staging, separate from the forgetting arguments in §2.

Log the realized domain share per training step in prompts and tokens, and the staleness distribution per domain, to see whether the configured mixture is the one being trained.

---

## §6 Consolidation and distillation as stage boundaries

Three of the recipes join specialists by SFT or distillation and not by RL:

- **MAI.** Consolidation SFT on the specialists' traces (4 epochs, LR max 1e−5 decayed 2×), then a short final RL ([[mai-thinking-1]] Sec. 3.5). Self-distillation also restarts a climb after a collapse or a base-model update. Findings: O(1M) traces suffice to match the teacher; training on incorrect traces performs similarly to successful-only; traces from several later checkpoints beat the final checkpoint only; prompt diversity beats more traces per prompt; random selection beats shortest-trace and heuristic selection; mid-training data must be mixed in to keep long-context behaviour. Resuming from a pre-collapse checkpoint "is often not viable" because instabilities are embedded "many steps before the actual collapse" (Sec. 3.1.4).
- **DeepSeek-V3.2.** Models trained on the distilled specialist data score slightly below the specialists, and the subsequent mixed RL removes the gap. No numbers are printed ([[deepseek-v3.1]] §3).
- **GLM-4.5.** Experts are distilled into one model with "millions of samples". In agentic RL, iterative distillation replaces the cold-start data with RL-model responses after RL plateaus, repeats SFT, and continues RL on harder tasks ([[glm-4-5]] §3.1, §3.3.2).

Qwen3's Thinking Mode Fusion is also an SFT stage between two RL stages, built by rejection sampling from the Stage-2 model ([[qwen-3]] §4.3).

**What the boundary does.** Distillation into one model converts several RL-trained distributions into one SFT distribution. The joint RL that follows starts from a checkpoint that already has each specialist's behaviour, so it begins near each domain's specialist ceiling, not from the base. This is the parallel-specialist answer to interference: domains never compete inside one long RL run, only in the short final one. In MAI's case even that final RL keeps STEM data for retention.

---

## Recipe

1. **List domains and their graders.** For each: grader type, rollout duration, turns, reward scale.
2. **Decide the structure.** Use parallel specialists → merge → short joint RL when domains need different schedulers or rollout shapes (SWE vs IF), or when one domain's climb is much longer (MAI's STEM). Use one joint stage when the domains share a rollout shape and a grader infrastructure (Olmo 3's math/code/IF/chat). Use reasoning first, general last when reasoning data is the main investment and general behaviour must be shaped last (Qwen3, MiniMax-M1).
3. **Never drop a domain in a later stage without retention data.** Keep a small share of every earlier domain (MAI keeps STEM/code in final RL; Olmo's math-free stage was suboptimal).
4. **Wire graders by tag**, and check that every tag resolves. Alert on any domain whose mean reward is exactly 0 for a step.
5. **Normalize per domain.** Per-task advantage normalization, or rescale rewards to a common variance. Record each grader's reward scale.
6. **Set weights in tokens.** Compute token share per domain at the stage's length limits. Adjust prompt counts to reach the intended token share.
7. **Make weights runtime-changeable** if the framework allows it, and log every change with its step number so later score changes can be attributed.
8. **Track realized mixture per step** (prompts and tokens) and per-domain staleness and drop rates.
9. **Evaluate every domain at every evaluation step**, including those not in the current stage: IF (HAcc on held-out types), reasoning, agentic, and a chat-quality judge.
10. **At stage boundaries, consider distillation.** If specialists exist, merge by SFT on traces from several late checkpoints with diverse prompts (MAI findings), then run a short joint RL with retention data.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Dropping an earlier domain in a later stage | that domain's eval falls during the later stage | evaluate all domains at every eval step; keep retention data |
| Treating lab stage orders as measured evidence | an order is copied without testing; results differ | label each choice "stated reason" or "measured"; most are stated only |
| Mixing by prompt count | short-output domains regress; long-output domains dominate | log token share per domain per batch |
| A dataset tag with no registered verifier | a domain's reward is exactly 0 everywhere | per-tag reward histogram; fail on unknown tags instead of warning |
| Unnormalized reward scales across graders | one domain's advantages dominate updates | log advantage magnitude per domain; normalize per task |
| Asynchronous loop with mixed durations | realized mixture drifts toward short domains; long-domain samples are stale or dropped | log realized vs configured mixture and per-domain staleness |
| Changing weights mid-run without logging | score changes cannot be attributed | log every runtime config change with its step |
| Assuming joint RL removes all interference | IF compliance falls as reasoning length grows inside the joint stage | track HAcc with length (MathIF Table 5) |
| Resuming from a pre-collapse checkpoint | the same collapse recurs | restart with self-distillation instead (MAI Sec. 3.1.4) |

---

## Check your understanding

1. DeepSeek-V3.2 and Nemotron both justify joint RL by forgetting, and neither prints an ablation. Design the smallest experiment that would test "staged vs joint" for two domains with data and compute held fixed, and name the confound you would control first.
2. Magistral's math-only and code-only RL each improved the other domain, while MathIF shows math RL lowering IF compliance. Propose a property of the domains that would explain positive transfer in one case and interference in the other, and say how you would measure it.
3. MAI keeps some STEM data in the final RL. Using the policy-gradient view (no gradient on absent domains), explain why a domain that is absent from a stage can still degrade, and why a small retention share can be enough.
4. In an asynchronous joint stage with Rocket's Fig. 2 durations, explain why the realized domain mixture of a training step can differ from the configured one, and how a staleness cutoff changes the realized weight of SWE.
5. GLM-4.5 runs general and reasoning RL synchronously and agentic SWE RL asynchronously. Explain how this infrastructure choice pushes the recipe toward separate stages, independent of forgetting.
6. MAI finds that self-distillation traces from several later checkpoints beat traces from the final checkpoint only. Give one mechanism that could explain this, and an experiment that would test it.
7. Explain why mixture weights stated in prompts and in tokens can differ by more than an order of magnitude for one domain, using MAI's Table 10, and which one controls the gradient under a token-level loss.

---

## Connections

- **ch-14, ch-15, ch-16** — the per-domain pipelines combined here.
- **ch-13** — domain and length axes; mixture weights in tokens.
- **ch-03, ch-04** — staleness and the levers that act on mixed-duration batches.
- **ch-18** — algorithm choice per stage and domain.
- **ch-19** — subjective rewards for the alignment stages.
- **ch-21** — elasticity: runtime config changes, including mixture weights.
- **ch-22** — capstone: choosing a stage order for the learner's setting.
- **llm-training ch-44b** (multi-domain RL for general capability: non-verifiable rewards and domain mixing) — the general treatment of domain mixing.
- **llm-training ch-45a** (preference-optimization and RL stage recipes side by side) — earlier stage-recipe comparisons.
- **llm-training ch-16** (RL prompt distribution) — difficulty and domain breadth inside one stage.
- **llm-training ch-40** (group baselines) — per-task advantage normalization in context.

---

## Sources

- [[recipe-crosswalk]] — the library crosswalk of eight lab recipes; the "no staged-vs-joint ablation" finding.
- [[mai-thinking-1]] — specialist climbs, consolidation mixture, final RL with retention data, self-distillation findings, grading placement.
- [[mai-rl-infrastructure]] — curriculum controller, runtime mixture weights, Table 1 elasticity events, per-domain rollout durations.
- [[glm-4-5]], [[glm-4-5-recipe]] — expert iteration, agentic RL, general RL, iterative distillation, sync/async split.
- [[qwen-3]], [[qwen-3-recipe]] — four-stage flagship pipeline; Table 22.
- [[kimi-k2]], [[kimi-k2-recipe]] — joint RL with verifiable gym and rubric reward; PTX loss.
- [[deepseek-v3.1]], [[deepseek-v3-2]] — specialists → distillation → mixed RL; the forgetting rationale.
- [[olmo-3-rl-pipeline]], [[olmo-3-recipe]] — OlmoRL mix, IF by stage, Instruct footnote on math-free stage.
- [[open-instruct-rlvr-pipeline]] — per-row verifier dispatch code.
- [[minimax-m1]] — reasoning-first then gradual general mixing; run-2 re-filtering.
- [[nemotron-ultra-recipe]] — RLVR → RLHF → RLVR; single-environment degradation statement; curriculum.
- [[rlvr-tulu3]], [[tulu-3]] — SFT → DPO → RLVR reference order.
- [[magistral]] — math/code cross-domain transfer; IF and function calling after text RL.
- [[scalerl]] — math + code joint training trends.
- [[scaling-reasoning-losing-control-mathif]], [[ifbench]], [[general-reasoner-verifier]], [[agentrl]] — controlled interference and multi-domain results.
