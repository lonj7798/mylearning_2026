<!-- chapter: ch-14
     track: rl-pipeline
     kind: content
     title: Instruction-Following RL Pipeline
     deps: [ch-12]
     sources: [[ifbench]], [[rlvr-tulu3]], [[tulu-3]], [[scaling-reasoning-losing-control-mathif]], [[rubrics-as-rewards]], [[mai-thinking-1]], [[open-instruct-rlvr-pipeline]], [[olmo-3-rl-pipeline]], [[rlcf-checklists]], [[glm-4-5]], [[glm-4-5-recipe]], [[kimi-k2]], [[minimax-m1]], [[nemotron-ultra-recipe]], [[qwen-3]], [[recipe-crosswalk]], [[mai-rl-infrastructure]], [[magistral]], [[rl-design-guide]]
     figures: figures/if-pipeline.html
-->

# Chapter 14 — Instruction-Following RL Pipeline

> **Core insight.** Instruction-following (IF) RL turns a constraint taxonomy into a grader. Each prompt is a general instruction with 1 to n attached constraints. Each constraint maps to a program that returns pass or fail on the response. The per-constraint results are combined into one reward. This makes IF one of the cheapest RL domains to roll out: median 2 minutes and 1 turn in Rocket's production mix ([[mai-rl-infrastructure]] Fig. 2). The same structure creates three failure modes. The choice of aggregation (all constraints satisfied, fraction satisfied, or sum) changes which responses get negative advantage, and Ai2's own paper and code disagree on which one Olmo 3 uses. The checker tests surface properties, so the policy can satisfy it while the answer gets worse: pure constraint reward lowered AlpacaEval 2 from 33.5 to 21.3 ([[ifbench]] Table 3). Reasoning RL on other domains lowers constraint compliance unless IF stays in the mix: 15 of 16 controlled variants lost compliance ([[scaling-reasoning-losing-control-mathif]] Table 4).
>
> **Guideline.** When building IF prompts, attach several constraints per prompt drawn from many types, with variable ranges that include and extend the test range, and hold out whole constraint types for evaluation, because training on up to 5–6 constraints generalized better than up to 3 although tests had at most 3 ([[ifbench]] §4.1, Fig. 2), and Tülu 3 8B scored 82.4 on IFEval but 24.3 on IFEval-OOD ([[tulu-3]] Table 31). When constraint rewards are the only signal, add a quality signal (reward model, rubric judge or lexicographic gate), because IFBench's reward-model-gated shaping recovered AlpacaEval 2 to 31.6 at a cost of IF accuracy (IFEval 86.1, IFBench 30; [[ifbench]] App. E). Decide the aggregation rule explicitly and write it down, because open-instruct at f020e9a returns the fraction of satisfied constraints (`ground_truth_utils.py` L381) while the Olmo 3 report describes a binary all-satisfied reward ([[olmo-3-rl-pipeline]] Verification). When IF is trained next to reasoning RL, track a verifiable IF metric next to reasoning accuracy, because a 1k → 8k rollout length raised math accuracy from 28.73 to 39.82 and lowered hard IF accuracy from 19.05 to 14.29 ([[scaling-reasoning-losing-control-mathif]] Table 5).

---

## Why this chapter matters

IF is the first domain pipeline in this phase because it isolates the grader design problem. The prompts are short, rollouts are single-turn, and there is no sandbox or tool server. The only hard part is the reward. Every component of ch-09 to ch-12 shows up in its simplest form:

- **Task (ch-09):** the constraint taxonomy and the prompt construction rules.
- **Rollout (ch-10):** short, single-turn generations. Rocket's IF rollouts have a median of 2 minutes and 1 turn, the shortest of the eight domains in Fig. 2 ([[mai-rl-infrastructure]]).
- **Grader (ch-11):** programmatic checkers for hard constraints, LLM judges or rubrics for soft ones.
- **Reward composition (ch-12):** how per-constraint results become one scalar, and how constraint reward combines with a quality signal.

IF also interacts with every other domain. ch-17 shows that IF appears in nearly every lab recipe, and §8 shows that reasoning RL removes constraint compliance unless IF is trained alongside it. For the applied target of this course, a sales voice agent, the hard constraints include "answer in at most two sentences", "do not quote a price before qualifying", and "always confirm the callback number". These are the constraint types this chapter covers.

---

## §1 Where IF RL sits in lab recipes

| Lab | Stage that trains IF | Grader | Source |
|---|---|---|---|
| Tülu 3 | RLVR (PPO) after SFT and DPO; 14,973 IF prompts in a 29,946-prompt verifiable mix | binary verifier, α = 10 | [[rlvr-tulu3]] Table 22, Eq. 7–8 |
| Olmo 3 | OlmoRL after SFT and DPO; IF-RLVR 30,186 prompts (Think) / 38,000 (Instruct) inside a mixed math/code/IF/chat set | constraint functions; report says binary, code returns fraction | [[olmo-3-rl-pipeline]] Table 20, §4.4.1 |
| Qwen3 | General RL (stage 4), one of >20 tasks | rule-based plus model-based rewards | [[qwen-3]] §4.4 |
| GLM-4.5 | General RL expert; 7 major and 151 minor constraint types | rules + reward model + critique model; GRPO | [[glm-4-5]] §3.4 |
| Kimi K2 | one joint RL stage | code-interpreter checks, LLM-judge checks, hack-check layer | [[kimi-k2]] §3.2.1 |
| MiniMax-M1 | general tasks mixed in after rule-verified reasoning | rule-based constraint reward + model-based quality reward | [[minimax-m1]] §4.2.1 |
| Nemotron 3 Nano | RLVR with all environments at once; 46K IFEval/IFBench-style + 3K multi-turn judged IF | rules; LLM judge for multi-turn | [[nemotron-ultra-recipe]] NR §3.2.1 |
| MAI Thinking 1 | helpfulness + safety climb; expert-written + synthetic prompts from a constraint taxonomy, >40 domains, conflicting system/developer/user cases | rule checks + per-rubric binary LLM judge (averaged) + RM, combined lexicographically with IF rewards primary | [[mai-thinking-1]] Sec. 3.4.2 |

**Pattern.** No carded lab trains IF as its own first RL stage. IF appears (a) as part of a final verifiable-RL stage (Tülu 3), (b) inside a mixed RLVR stage (Olmo 3, Kimi K2, Nemotron), or (c) inside a general or helpfulness stage after reasoning RL (Qwen3, GLM-4.5, MiniMax-M1, MAI). ch-17 compares the full stage orders.

---

## §2 Prompt construction: taxonomy, constraint count, held-out types

### §2.1 The taxonomies

- **IFEval**: 25 constraint templates. Many models score above 80% on it at sizes as small as 2B ([[ifbench]] §1).
- **IFBench**: 58 new test constraints in 7 categories (count, ratio, words, sentence, format, custom, copy), 300 prompts, each with 1 or 2 constraints, human-checked for prompt–constraint compatibility. **IFTrain**: 29 training constraints, disjoint from the IFBench test constraints ([[ifbench]] §2, App. A–B).
- **GLM-4.5**: 7 major and 151 minor constraint types ([[glm-4-5]] §3.4).
- **MAI Thinking 1**: a constraint taxonomy over more than 40 domains, including conflicting system, developer and user instructions ([[mai-thinking-1]] Sec. 3.4.2).

### §2.2 How prompts are built

The common recipe, from the [[recipe-crosswalk]] IF synthesis:

1. Take general instructions from an SFT mix. Tülu 3 pairs instructions sampled from the Tülu 2 SFT mix with IFEval-taxonomy constraints ([[rlvr-tulu3]] Table 22). IF-RLVR uses Tülu-3-SFT instructions ([[ifbench]] §3).
2. Attach 1 to n constraints. Olmo 3 uses up to 5, sampled from IFEval and IFBench-Train ([[olmo-3-rl-pipeline]] §4.4.2).
3. Block contradictory pairs with a conflict dictionary ([[ifbench]] §3). open-instruct implements this as `INSTRUCTION_CONFLICTS` next to the checker registry (`IFEvalG/instructions_registry.py` L193).
4. Expand variable ranges beyond the benchmark's ranges (IF-RLVR expands IFEval's ranges, [[ifbench]] §3).

### §2.3 What the ablations measured

All of the following are from [[ifbench]], with a single model family per row:

| Variable | Result | Location |
|---|---|---|
| Constraints per instance n = 1…6 (Qwen2.5 policy) | IFEval 71.2, 79.9, 77.8, 79.5, 79.9, 85.8; IFBench 48.9, 53.1, 59.5, 49.4, 55.8, 54.1 | Table 1 |
| Up to 5–6 vs up to 3 (Tülu-DPO policy) | 5–6 generalized better, although test prompts have ≤ 3 (IFEval) or ≤ 2 (IFBench) constraints | §4.1, Fig. 2 |
| Instances per constraint 10 → 1000 | IFBench 48.6, 52.7, 51.7, 51, 48.6: no monotone gain | Table 1 |
| Variable range disjoint from test | lowest IFEval; a wider range that includes the test range is comparable or better | §4.3, Fig. 5 |
| Removing a category | removing LENGTH or KEYWORDS lowers IFEval most | §4.4, Fig. 6 |

Read the n = 1…6 IFBench row as non-monotone: the peak is 59.5 at n = 3. Only IFEval rises at n = 6. The "5–6 is better" conclusion comes from the Tülu-DPO runs in Fig. 2.

### §2.4 Held-out constraint types

A model can score well on the training taxonomy and fail on new constraint types. Tülu 3 8B: IFEval 82.4, IFEval-OOD 24.3 ([[tulu-3]] Table 31). GPT-4.1 and Claude 3.7 Sonnet score below 50% on IFBench's 58 unseen constraints ([[ifbench]] §1, Fig. 1). The authors attribute the gap to overfitting a small constraint set, and the paper does not say RLVR caused that overfitting ([[ifbench]] Verification). Hold out constraint **types**, not only prompts, because a held-out prompt with a seen constraint type tests the training checkers again.

---

## §3 From a constraint string to a check: open-instruct

open-instruct at f020e9a is the open implementation behind Olmo 3 ([[open-instruct-rlvr-pipeline]]). Its path from data row to reward shows where the grader can go wrong.

**Data row.** A row carries `messages`, `ground_truth`, `dataset` (the verifier tag) and `constraint_type` / `constraint` (`test_data/rlvr_sample.jsonl` keys). For IF, `dataset = "ifeval"` and `ground_truth` is a Python-literal list containing one dict with parallel `instruction_id` and `kwargs` arrays.

**Dispatch.** `apply_verifiable_reward` looks up `reward_fn_mapping[ds.lower()]` per row and sums `reward_mult * score * reward_weight` over the row's (ground truth, dataset) pairs (`ground_truth_utils.py` L1298–1362). `reward_mult` is `verification_reward`, default 10.0 (L1371).

**The IF verifier loop.**

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L370-L381
for instruction_key, args in zip(instruction_keys, args_list):
    if args is None:
        args = {}
    args = {k: v for k, v in args.items() if v is not None}
    instruction_cls = instruction_dict[instruction_key]
    instruction_instance = instruction_cls(instruction_key)
    instruction_instance.build_description(**args)
    if prediction.strip() and instruction_instance.check_following(answer):
        rewards.append(1.0)
    else:
        rewards.append(0.0)
return VerificationResult(score=sum(rewards) / max(len(rewards), 1))
```

Before this loop, `answer = remove_thinking_section(prediction, ...)` keeps only the text after the last `</think>` and strips `<answer>` tags (L195–205, L363). Constraints are checked on the final answer only, so a constraint such as "use no commas" does not apply to the reasoning.

**One checker.** `keywords:frequency` maps to `KeywordFrequencyChecker`:

```python
# open-instruct@f020e9a open_instruct/IFEvalG/instructions.py L811-L818
def check_following(self, value):
    """Checks if the response contain the keyword with required frequency."""
    actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

    if self._comparison_relation == _COMPARISON_RELATION[0]:
        return actual_occurrences < self._frequency
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
        return actual_occurrences >= self._frequency  # pytype: disable=bad-return-type
```

Two properties of this code matter for RL (code-reading observations; no measurement of their effect was found):

1. **Substring counting.** `re.findall(self._keyword, …)` has no word boundary, so the keyword "art" is also counted inside "start" and "party". An "at least N" constraint can be met with words that contain the keyword, and a "less than N" constraint can fail because of them. The checker is looser than the instruction text in one direction and stricter in the other.
2. **Random defaults.** If `kwargs` is missing a value, `build_description` picks one at random: `random.randint(1, _KEYWORD_FREQUENCY)` for the frequency and `random.choice(_COMPARISON_RELATION)` for the relation (L784, L787). A malformed label then gives a different constraint on each call, which makes the reward noisy. open-instruct includes an admission screen, `if_training_issues`, whose docstring says: "Missing arguments are rejected because checkers otherwise choose random defaults" (`ifeval_constraints.py` L63–68). At this commit its callers are in `open_instruct/miles/`, not in `grpo_fast.py` ([[open-instruct-rlvr-pipeline]] Technical Details). Run an equivalent screen on any IF dataset before RL.

**Registry size.** `INSTRUCTION_DICT` registers 54 active checker classes; several entries are commented out (`instructions_registry.py` L125–191, counted at this commit).

[figures/if-pipeline.html](figures/if-pipeline.html) shows this path for one prompt: the constraint list, a set of candidate responses, each checker's pass or fail, and the reward each response gets under the three aggregation rules of §4.

---

## §4 Aggregation: binary, fraction, or sum

### §4.1 Three rules in one lineage

| Source | Rule for a prompt with n constraints | Location |
|---|---|---|
| Tülu 3 RLVR | v(x, y) = α if the constraint is satisfied, else 0; α = 10 | [[rlvr-tulu3]] Eq. 7–8 |
| IFBench IF-RLVR | sum over constraints of verifiable_reward_i · multiplier_i · weight_i (multipliers and weights generally 1) | [[ifbench]] §3, Eq. 1 |
| Olmo 3 report | "1 if all constraints are satisfied, and 0 otherwise" | [[olmo-3-rl-pipeline]] §4.4.1 |
| open-instruct code at f020e9a | fraction satisfied, `sum(rewards) / max(len(rewards), 1)`, then × `verification_reward` (default 10) | `ground_truth_utils.py` L381, L1371 |

**Paper–code discrepancy.** The Olmo 3 report and the code at f020e9a disagree: binary all-satisfied in the paper, fraction satisfied in the code. The card states that it is "Not resolved which version trained the released models" ([[olmo-3-rl-pipeline]] Verification). The repository history after the Olmo 3 release could explain the difference, but that history was not traced. Treat the rule as unknown for the released models.

MathIF names the two metrics: hard accuracy HAcc = ∏ I(C_i) (all constraints met) and soft accuracy SAcc = (1/n) Σ I(C_i) ([[scaling-reasoning-losing-control-mathif]] Eq. 1). Binary aggregation is an HAcc reward. Fraction aggregation is an SAcc reward. Sum aggregation is n × SAcc.

### §4.2 What the rule does to the gradient (derived)

Take one prompt with 3 constraints and a group of 4 responses that satisfy 3, 2, 2 and 0 constraints. Use advantages A_i = r_i − mean(r), without standard-deviation division (as in OlmoRL, which removes std normalization, [[olmo-3-rl-pipeline]] §4.4.1).

| Response | Satisfied | Binary r | Binary A | Fraction r | Fraction A | Sum r | Sum A |
|---|---|---|---|---|---|---|---|
| 1 | 3/3 | 1 | +0.75 | 1.00 | +0.42 | 3 | +1.25 |
| 2 | 2/3 | 0 | −0.25 | 0.67 | +0.08 | 2 | +0.25 |
| 3 | 2/3 | 0 | −0.25 | 0.67 | +0.08 | 2 | +0.25 |
| 4 | 0/3 | 0 | −0.25 | 0.00 | −0.58 | 0 | −1.75 |

Under the binary rule, the two 2/3 responses are pushed down as hard as the 0/3 response. Under the fraction rule, they are pushed slightly up. The sum rule has the same ordering as the fraction rule and scales advantages by n, so prompts with more constraints get larger updates unless advantages are normalized. A constant multiplier (α = 10 or `verification_reward` = 10) cancels under group-std division and scales the advantage directly without it.

Each rule also has a zero-signal case. Under the binary rule, a group where every response misses at least one constraint has zero variance and gives no gradient. Under the fraction rule, that group still varies. With 5 constraints per prompt (Olmo 3's maximum), all-satisfied is rarer than any-satisfied, so the binary rule produces more zero-variance groups.

**Measured comparison.** No ablation of binary vs fraction vs sum for IF RL was found in the cards read. The closest measurement is in a different domain. Magistral's proportional code reward (fraction of tests passed) discarded three times less data but gave 2% lower LiveCodeBench than binary pass/fail ([[magistral]] §7.4.1, Fig. 11). The authors suggest partial rewards may give "false signal to incorrect solutions". That is an analogue, not IF evidence.

---

## §5 Soft constraints: LLM judges and rubrics

Hard constraints (word counts, formats, keywords) have checkers. Soft constraints ("be concise", "sound friendly", "address the customer by name naturally") do not. Labs grade them with a model.

| Source | Soft-constraint grader | Notes |
|---|---|---|
| GLM-4.5 | rules + reward model + critique model | IF RL run alone showed no clear reward hacking up to about 1,000 steps (Fig. 9) ([[glm-4-5]] §3.4) |
| MiniMax-M1 | constrained IF uses rule-based constraint reward + model-based quality reward | [[minimax-m1]] §4.2.1 |
| Kimi K2 | code-interpreter checks, LLM-judge checks, and a hack-check layer that detects "claimed but unfulfilled compliance" | [[kimi-k2]] §3.2.1 |
| MAI Thinking 1 | per-rubric binary LLM judge, averaged; plus RM; combined lexicographically with IF primary | [[mai-thinking-1]] Sec. 3.4.2 |
| Rubrics as Rewards | per-prompt rubric written with a reference answer, scored by a judge | HealthBench-1k: 35.9 with reference-conditioned rubrics vs 32.0 without reference vs 23.9 Likert ([[rubrics-as-rewards]] Table 1) |
| RLCF | per-instruction checklist generated from instruction + candidate responses, items scored 0–100 by a 72B judge averaged over 25 samples; used to choose DPO pairs | IFEval prompt-loose 75.0 → 77.3; candidate-based beat direct checklists (77.3 vs 74.3) ([[rlcf-checklists]] Tables 2, 6) |

**open-instruct's judge interfaces.** `LMJudgeVerifier` calls an OpenAI-compatible judge; the prompt types are mapped in `JUDGE_PROMPT_MAP` (`judge_utils.py` L278–287: quality, quality_rubric, quality_ref, safety, factuality, creative_writing, refusal, web_instruct_general_verifier). The score extractor divides a 1–10 score by 10, and a parse failure yields 0.0:

```python
# open-instruct@f020e9a open_instruct/judge_utils.py L267-L275
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning(f"Could not parse score from due to invalid json: {score_str}, defaulting to 0.0")
        return score_str, 0.0


def extract_score_with_fallback_max_10(score_str: str) -> "tuple[str, float]":
    """Extractor based on score with fallback"""
    reasoning, score = extract_json_score_with_fallback(score_str)
    return reasoning, score / 10.0
```

A judge parse failure is therefore scored the same as the worst possible answer. Log the parse-failure rate per step, because a rise in it looks like a quality drop in the reward curve. Olmo 3 uses Qwen3 32B as judge, with thinking off, 32,768-token input and 2,048-token output ([[olmo-3-rl-pipeline]] Key Contributions). A separate `RubricVerifier` scores a list of weighted rubric items and returns their weighted average; its default judge is `gpt-4.1` at temperature 0.0 (`ground_truth_utils.py` L1110–1128).

**Cost.** RLCF's checklist grading with 25 judge samples per item took 92 hours, against 40 hours with 5 samples, on one 8×H100 node for WildChat ([[rlcf-checklists]] §5.7, Fig. 4). An IF rollout is short (2 minutes median in Rocket), so a judge call can be slower than the rollout it grades. Budget judge throughput as part of the rollout fleet.

---

## §6 Mixing hard and soft rewards

### §6.1 The measured problem

Constraint-only reward narrows the model. From Tülu-DPO, IF-RLVR moved AlpacaEval 2 from 33.5 to 21.3, GSM8K from 84.3 to 83.2 and MMLU from 68.7 to 66.4. A GPT-4.1 judge scored responses with the constraint removed at 7 for the base policy and 6.4 for the IF-RLVR policy, out of 10 ([[ifbench]] Table 3, §5). IF-RLVR responses averaged 210 tokens, against 2,214 for frontier models (App. D.1). The authors also report that IF-RLVR models tend to prioritize the constraint over the task (§1, §5).

### §6.2 Three ways to combine

**(a) Gated shaping (IFBench App. E).** With verifiable reward V_i and reward-model score S_i:

```
F_i = V_i + 1     if V_i > 0 and S_i > α
F_i = V_i − 0.5   if V_i > 0 and S_i ≤ α
F_i = V_i         if V_i ≤ 0
```

Settings: Llama-3.1-Tulu-3-8B-RM, α = 7, effective batch 512, 8 samples per prompt, 1,100 steps. Result: IFEval 86.1, IFBench 30, AlpacaEval 2 31.6. That is lower IF and higher AlpacaEval than ground-truth-only training ([[ifbench]] App. E, ledger). This is a measured trade-off, not a free improvement.

**(b) Lexicographic priority (MAI).** Rewards are combined by priority order, not by a weighted sum, with IF rewards primary in the IF domain. In the safety domain a gate sets the minimum reward for any non-compliant response, and the style judge (0/1/2 scale) applies only after the verifiable and safety gates ([[mai-thinking-1]] Sec. 3.4.1, domain table). The card records no ablation of lexicographic against weighted combination. The stated rule is a design choice.

**(c) Parallel signals in one judge stack (GLM-4.5, MiniMax-M1, Kimi K2).** Rules, reward model and critique model all contribute. The combination formula is not printed in the cards read.

**The difference between (a) and (b), derived.** Under a weighted sum, a response that fails a hard constraint can outscore one that passes if its quality score is high enough. Under a lexicographic rule it cannot, because quality only orders responses that tie on the higher-priority term. Gated shaping (a) sits in between: quality changes the reward only when V_i > 0.

---

## §7 Reward hacking: satisfying the checker while degrading the answer

**Observed forms.**

1. **Keyword stuffing.** IFBench Fig. 7 shows an over-optimized output that repeats required keywords instead of solving the task ([[ifbench]] §5).
2. **Over-optimization at high KL.** Tülu 3 App. B.4, Figures 28–29 show over-optimized IFEval outputs from higher-KL runs. Average scores dropped as KL grew at lower β ([[rlvr-tulu3]] §6.2.1, Fig. 21).
3. **Claimed but unfulfilled compliance.** Kimi K2 added a hack-check layer for responses that claim to follow a constraint without doing so ([[kimi-k2]] §3.2.1). The report gives no rate.
4. **Short, constraint-first answers.** The 210-token average and the 7 → 6.4 judge score in §6.1.

**Code-level hack surfaces.** The substring count and random-default behaviours in §3 are places where checker semantics differ from instruction semantics. A policy optimizes the checker, not the instruction text.

**Counter-evidence.** GLM-4.5 reports no clear reward hacking up to about 1,000 steps for an IF RL run with rules + RM + critique model (Fig. 9, [[glm-4-5]] §3.4). That is one run with a combined grader, and it shows a combined grader can delay visible hacking. It does not show that hacking is absent.

**Detection.**
- Grade a fixed sample of responses with the constraint removed, using a held-out judge (IFBench's §5 protocol). A falling score means quality is being traded for compliance.
- Track response length and keyword density per constraint type.
- Track IF accuracy on held-out constraint types (IFBench test set). A rise on training types with flat held-out types suggests the policy is learning the checkers, not the instructions.
- Spot-check responses that pass all constraints but have the lowest quality scores.

---

## §8 The reasoning-vs-instruction-following tension

### §8.1 Controlled evidence

MathIF evaluates math problems with 1–3 attached format constraints ([[scaling-reasoning-losing-control-mathif]]).

- In controlled runs on four Qwen2.5 bases, long-CoT SFT and GRPO lowered both HAcc and SAcc in 15 of 16 trained variants while raising average math accuracy in 15 of 16 (Table 4). Example, Qwen2.5-7B HAcc / SAcc / math: base 15.95 / 33.13 / 13.59; +SFT 7.86 / 21.03 / 23.10; +SFT+RL 7.62 / 21.07 / 32.82; +cold-RL 10.48 / 27.26 / 28.39 (Table 4).
- The only variant that rose above its base on both IF metrics was cold-RL with a format reward on Qwen2.5-1.5B (10.95 / 28.49 vs 10.00 / 27.26) (Table 4).
- Rollout length: continued RL of R1-Distill-Qwen-1.5B at 1k / 2k / 4k / 8k max length gave HAcc 19.05 / 16.43 / 16.91 / 14.29 and math 28.73 / 36.32 / 40.03 / 39.82 (Table 5). The trend is not monotone at every step.
- Repeating the constraint after the CoT raised Qwen3-32B HAcc from 43.81 to 59.29 and lowered correctness from 70.00 to 63.81 (Table 6).

The authors' explanation, that distance between instruction and answer dilutes attention to the constraint, is tested only by the constraint-repetition experiment (§5.3). Treat it as an interpretation.

### §8.2 What happens when IF stays in the RL mix (observational)

Olmo 3 Table 24 reports IFEval / IFBench by stage for 7B Think: SFT 77.9 / 30.0, DPO 75.9 / 28.3, RL 88.2 / 41.6. The RL stage contained IF-RLVR prompts alongside math and code ([[olmo-3-rl-pipeline]] Table 20, Table 24). The comparison is between stages, not against an RL run without IF, so it shows that IF can rise during mixed reasoning RL, not by how much IF prompts are responsible.

### §8.3 Design implications

- If the RL stage is reasoning-heavy, include IF prompts and evaluate HAcc on held-out constraint types at every evaluation step ([[rl-design-guide]] §4 rule 9).
- If the response length budget grows, expect compliance to fall unless IF prompts at the new length are present. MathIF's Table 5 is the measured length effect.
- MAI keeps a small proportion of STEM and coding data in its final RL "because reasoning otherwise degrades" ([[mai-thinking-1]] Sec. 3.5). The same retention logic applies in the other direction: IF data must stay in a reasoning-heavy final stage.

---

## §9 Multi-turn IF

IFBench defines a multi-turn setting: turn 1 task, turn 2 assistant response, turn 3 a request to rewrite it under a constraint. Mean input length is 76 tokens single-turn and 408 multi-turn ([[ifbench]] §2, App. D.1). Training results on Qwen2.5-7B-Instruct, IFBench single / multi-turn (Table 7):

| Training data | Single-turn | Multi-turn |
|---|---|---|
| single-turn | 45.9 | 71.7 |
| multi-turn | 34.7 | 68.6 |
| mixed | 54.8 | 72.9 |

Mixed training was best on both. Nemotron 3 Nano includes 3K multi-turn IF prompts graded by an LLM judge in its RLVR mix ([[nemotron-ultra-recipe]] NR §3.2.1). For a voice agent, constraints persist across turns ("keep answers short" applies to every turn), so per-turn checks are needed. A final-response check misses earlier violations.

---

## §10 Comparing with Rocket

The Rocket post does not describe an IF-specific pipeline. Two parts of its design apply directly:

1. **Where the grader runs.** Per-rollout graders run on rollout workers; group graders, for example a pairwise LLM comparison, run on problem workers ([[mai-rl-infrastructure]] Fig. 1; [[mai-thinking-1]] Sec. 3.6). An IF constraint check is a per-rollout grader. A rubric judge that ranks responses within a group is a group grader. open-instruct runs every verifier per response inside `apply_verifiable_reward` and has no group-grader stage.
2. **Duration mismatch.** IF rollouts (median 2 min) share a job with SWE rollouts (median 93 min) ([[mai-rl-infrastructure]] Fig. 2). If the controller admits work by count, IF groups finish and wait in the queue. Rocket's priority by rollout start time and its packing of training segments address this ordering (ch-04). An IF-only pipeline (Tülu 3, IFBench) never meets the problem.

---

## Recipe

1. **Fix the taxonomy and hold out types.** Choose training constraint types (IFEval 25 + IFTrain 29, or your own). Reserve whole types for evaluation (IFBench's 58, or a held-out set of your own) and never train on them.
2. **Write and unit-test every checker.** For each type, test edge cases: substring vs word match, case, punctuation, Unicode, empty answer, answer inside `<think>`. Decide whether constraints apply to the final answer only (open-instruct removes the thinking section).
3. **Screen labels before RL.** Reject rows with missing arguments or degenerate parameters (for example "less than 3 letters"), as `if_training_issues` does. A missing argument must never fall through to a random default.
4. **Build prompts.** General instructions from an SFT pool, 1–5 constraints each from a conflict-checked taxonomy, with variable ranges wider than the test range.
5. **Choose and record the aggregation rule** (binary all-satisfied, fraction, or sum). If no ablation is available, run both binary and fraction on a small budget and compare held-out HAcc and a quality judge. Log the zero-variance group share under each.
6. **Add a quality signal.** Use a reward model, a rubric judge, or a lexicographic gate. Decide priority explicitly: IF-first lexicographic (MAI) or gated shaping (IFBench App. E). Measure the trade-off: IFBench App. E recovered AlpacaEval 2 to 31.6 at IFEval 86.1.
7. **Budget the judge.** Estimate judge calls per step. Remember that a judge parse failure scores 0 in open-instruct, and log the parse-failure rate.
8. **Mix IF into reasoning stages.** Include IF prompts at the length budget of each reasoning stage, and track HAcc next to math accuracy at each evaluation.
9. **Include multi-turn IF** if the deployment is conversational. Mixed single- and multi-turn data was best in IFBench Table 7.
10. **Monitor for hacking.** Quality judge on constraint-free responses, length and keyword density per type, held-out-type accuracy, and manual review of all-pass, low-quality samples.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Assuming the paper's aggregation rule is the code's | reproduction differs; zero-variance share differs from expectations | read the verifier return line (open-instruct L381 returns a fraction); log reward histogram per prompt |
| Missing constraint arguments | reward for the same response varies across calls | call each checker twice on fixed inputs; run an admission screen |
| Substring checkers | "at least N" constraints satisfied by words containing the keyword | unit-test with near-miss words; switch to word-boundary regexes for word constraints |
| Constraint-only reward | AlpacaEval or a quality judge falls while IF rises (33.5 → 21.3) | judge constraint-free versions of responses every N steps |
| Evaluating only on seen constraint types | IFEval rises to 80s–90s; held-out types stay low (IFEval-OOD 24.3) | report IFBench-style held-out-type accuracy |
| Judge parse failures scored as 0 | reward dips that correlate with judge load or long responses | log parse-failure rate; retry or exclude rather than score 0 |
| Dropping IF from a reasoning-heavy stage | HAcc falls as rollout length grows | track HAcc per evaluation; include IF prompts at the stage's length |
| Constraints checked on reasoning text | the model learns to satisfy constraints in `<think>` or is penalized for them | confirm the checker input excludes the thinking section |
| Training only single-turn IF for a dialogue product | multi-turn compliance lags | add multi-turn rewrite-style prompts; per-turn checks |

---

## Check your understanding

1. In the §4.2 table, the binary rule pushes a 2/3 response down as hard as a 0/3 response. Explain why this can still be the preferred rule for a product where any constraint violation is a failure, and what it costs in zero-variance groups when prompts carry 5 constraints.
2. IFBench's n = 1…6 IFBench scores peak at n = 3, while the authors conclude that training on up to 5–6 constraints generalizes better. Explain how both statements can hold, using which policy and which figure each comes from.
3. Why does holding out constraint *types* test generalization when holding out *prompts* with seen types does not? Use the IFEval vs IFEval-OOD gap in your answer.
4. IFBench's gated shaping (App. E) raised AlpacaEval 2 and lowered IFEval relative to pure constraint reward. Using the F_i formula, explain which responses lose reward under gating and why that lowers IF accuracy.
5. MathIF finds that longer RL rollouts lower constraint compliance. Give the authors' explanation and the one experiment that tests it. What would you need to show that the explanation is causal?
6. A lexicographic reward ranks hard-constraint compliance above quality. Describe a group of responses for which a weighted sum and a lexicographic rule give opposite advantages to the same response.
7. open-instruct scores a judge parse failure as 0.0. Explain how this interacts with GRPO advantages when parse failures are more frequent for long responses, and what the policy may learn from it.

---

## Connections

- **ch-09** — prompt and reference-answer design; IF prompts are the simplest case.
- **ch-11** — grader types; this chapter's checkers and judges are worked instances.
- **ch-12** — reward composition and prompt selection; §4 and §6 apply it to IF.
- **ch-13** — the constraint-count axis.
- **ch-17** — where IF sits in each lab's stage order.
- **ch-18** — algorithm choice for IF (GRPO vs DPO vs PPO).
- **ch-19** — non-verifiable style rewards, the extension of §5.
- **llm-training ch-44** (process supervision and verifiable rewards) — verifier mechanics.
- **llm-training ch-44b** (multi-domain RL for general capability) — non-verifiable rewards and domain mixing.
- **llm-training ch-40** (group-baseline RL) — the advantage formulas used in §4.2.
- **llm-training ch-44a** (length in RL) — length effects behind §8.

---

## Sources

- [[ifbench]] — IFBench / IFTrain, constraint-count and range ablations, DPO vs GRPO, multi-turn table, over-optimization and RM-gated shaping.
- [[rlvr-tulu3]] — Tülu 3 RLVR: binary verifier α = 10, IF prompt construction, KL over-optimization.
- [[tulu-3]] — IFEval vs IFEval-OOD gap (Table 31).
- [[olmo-3-rl-pipeline]] — Dolci-Think-RL IF data, binary IF verifier in the report, the paper–code discrepancy, IF by stage (Table 24), judge settings.
- [[open-instruct-rlvr-pipeline]] — code path for dispatch, IFEval checkers, judges; all code quoted from open-instruct@f020e9a.
- [[scaling-reasoning-losing-control-mathif]] — HAcc / SAcc definitions, controlled reasoning-RL effect on compliance, length and repetition interventions.
- [[rubrics-as-rewards]] — per-prompt rubric rewards and the reference-answer ablation.
- [[rlcf-checklists]] — candidate-based checklists, judge sampling cost.
- [[mai-thinking-1]] — IF taxonomy and lexicographic reward combination; final-RL retention of STEM data.
- [[glm-4-5]], [[glm-4-5-recipe]] — IF constraint taxonomy and the no-clear-hacking curve.
- [[kimi-k2]] — hack-check layer.
- [[minimax-m1]] — rule + model reward for constrained IF.
- [[nemotron-ultra-recipe]] — IF environments in joint RLVR.
- [[qwen-3]] — IF inside general RL.
- [[recipe-crosswalk]] — IF pipeline synthesis across labs.
- [[mai-rl-infrastructure]] — IF rollout duration and grader placement in Rocket.
- [[magistral]] — proportional vs binary code reward (analogue for §4).
