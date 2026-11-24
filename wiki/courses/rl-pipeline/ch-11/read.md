<!-- chapter: ch-11
     track: rl-pipeline
     kind: content
     title: Designing the Grader: Rule Verifiers, Execution, LLM Judges, and Rubrics
     deps: [ch-10]
     sources: [[rl-design-guide]], [[math-verify]], [[verifier-pitfalls]], [[master-rm-one-token]], [[deepseek-grm]], [[general-reasoner-verifier]], [[rlcf-checklists]], [[rubrics-as-rewards]], [[generative-reward-models]], [[judge-llm-bias]], [[reward-hacking-taxonomy]], [[mai-rl-infrastructure]], [[mai-thinking-1]], [[kimi-k1-5-recipe]], [[r2e-gym]], [[olmo-3-rl-pipeline]], [[open-instruct-rlvr-pipeline]], [[areal-agent-pipelines]], [[slime-agentic-and-mismatch-examples]], [[writing-zero]], [[multichallenge]], [[rstar2-agent]], [[rubicon-rubric-anchors]], [[verl-rollout]]
     figures: figures/grader-ladder.html
-->

# Chapter 11 — Designing the Grader: Rule Verifiers, Execution, LLM Judges, and Rubrics

> **Core insight.** Graders form a ladder from cheap and narrow to expensive and general: string and math-equivalence rules, code execution, small model-based verifiers, LLM judges with a reference answer, rubric and checklist judges, generative reward models with sampled principles and voting, and pairwise or group judges that compare rollouts with each other (Rocket's group graders). Each step up raises recall on correct-but-unusual responses and lowers false negatives, and each step up adds false positives that the policy can find and exploit. The two error types are asymmetric in RL: a false negative removes signal from a correct behavior, while a false positive is a target. Rule verifiers had precision about 0.99–1.00 but average recall 86%, falling to 0.78 on Skywork-OR1 responses and 47% on general science ([[verifier-pitfalls]] Table 5, §4.3); a generative verifier used as the RL reward was hacked after about 450 iterations by single-symbol and gibberish outputs (§5.2); reference-based LLM judges accepted content-free responses such as ":" or "Thought process:" at an average false-positive rate of 66.8% for Qwen2.5-72B ([[master-rm-one-token]] Table 1). The design rule that follows: use the lowest rung whose errors on *your policy's outputs* are measured and acceptable, combine rungs so a model judge only sees what the rules rejected, probe every model judge for known exploits before the run, and audit training reward against an independent oracle during it.
>
> **Guideline.** When a rule verifier's recall on your policy's outputs is below its precision, use a hybrid grader — rules first, a model verifier only on rule-rejected responses — because on Qwen2.5-7B GRPO this raised the average from 55.0 to 57.3 ([[verifier-pitfalls]] Table 2); for the fallback, prefer a discriminative verifier, because xVerify-3B-I kept training reward within 0.004 of a GPT-4o oracle with no late collapse while the generative R1-Distill-Verifier-1.5B reached 35.0% attack success under adversarial prefixes (Table 3, §6.2). Before any RL run, probe each LLM judge with master-key responses, and if you train a judge, add truncated reasoning-opener negatives, because Master-RM-7B trained this way had 0.1% average FPR with Cohen's κ = 0.91 against GPT-4o, and CoT prompting of the judge did not fix FPR ([[master-rm-one-token]] Tables 1–3). Outside math, use a small generative verifier that sees question, reference and response, because rule matching agreed with Gemini on 22.2% of correct answers against 78.7% for the 1.5B General-Verifier ([[general-reasoner-verifier]] §5.3). For open-ended tasks, give the judge a per-prompt rubric written with the reference answer, because on HealthBench-1k that scored 35.9 against 32.0 without the reference and 23.9 for a Likert judge without a rubric ([[rubrics-as-rewards]] Table 1). For pairwise judging, swap order and count a win only when both orders agree, because GPT-4 was order-consistent on 65.0% of pairs ([[judge-llm-bias]] Table 2). Budget grader latency as part of the rollout: 25 judge samples per item took 92 hours against 40 hours for 5 on one 8×H100 node ([[rlcf-checklists]] §5.7).

---

## Why this chapter matters

In RL, the grader is the specification. The policy is optimized against whatever the grader rewards, including its mistakes, and the optimization pressure is strongest exactly where the grader is weakest: responses the grader accepts but should not. Skalse et al. prove that on any policy set containing an open subset, a non-trivial reward pair that is unhackable must induce the same ordering over policies; a proxy that differs from the true reward in any way that matters is hackable ([[reward-hacking-taxonomy]] Theorem 1). A better grader reduces the size of the difference; it does not remove the possibility.

Rocket separates graders by what they see: per-rollout graders on the rollout worker score one trajectory and, where needed, the environment state; group graders on the problem worker score all rollouts of a problem together before advantages are computed ([[mai-rl-infrastructure]] p. 3). This chapter covers what each grader computes. ch-08 covered where it runs; ch-12 covers how several grader outputs become one reward.

This chapter follows [[rl-design-guide]] §3 and cites the underlying cards. The interactive figure [grader-ladder.html](figures/grader-ladder.html) lays out the ladder with the measured error rates for each rung and includes a calculator for the hybrid grader of §4.

---

## §1 The ladder

| Rung | Sees | Output | Typical error | Cost | Measured example |
|---|---|---|---|---|---|
| 1. String / math-equivalence rule | extracted answer, reference | 0/1 | false negatives on equivalent forms | microseconds; timeouts on symbolic simplification | recall 0.78–0.95 by dataset ([[verifier-pitfalls]]) |
| 2. Code execution | program, tests | pass/fail or pass rate | weak or flaky tests (FN and FP) | seconds to minutes; sandbox | under 20% of generated tests distinguish patches for most problems ([[r2e-gym]]) |
| 3. Discriminative model verifier | question, reference, response | 0/1 | FP under adversarial patterns | one forward pass | xVerify-3B-I 0.2% attack success ([[verifier-pitfalls]]) |
| 4. Generative verifier / reference judge | question, reference, response | CoT then verdict | FP on master keys; hacking | one generation | Qwen2.5-72B 66.8% average FPR ([[master-rm-one-token]]) |
| 5. Rubric / checklist judge | prompt, rubric, response | per-item scores, aggregated | rubric ambiguity; length and style pressure | k judge calls per response | 35.9 vs 23.9 Likert ([[rubrics-as-rewards]]) |
| 6. Generative RM with voting | prompt, response(s) | principles, critique, score; k samples | unfaithful critiques; cost | k generations | 67.9 → 72.8 at k = 32 with meta-RM ([[deepseek-grm]]) |
| 7. Pairwise / group judge | several responses | preferences or relative scores | position bias | per pair or per group | 65.0% order consistency for GPT-4 ([[judge-llm-bias]]) |

Rungs 1–2 dominate verifiable domains; rungs 4–7 dominate open-ended ones. Real pipelines combine rungs: MAI's tool-use graders mix final-state checks, tool-pattern checks, answer checks and an LLM judge per sub-task ([[mai-thinking-1]] Sec. 3.3.2), and its instruction-following graders mix rule checks, a per-rubric binary judge and a reward model (Sec. 3.4.2).

---

## §2 Rule verifiers

**Three steps, three failure modes.** Math-Verify separates a rule verifier into extraction (find an answer in free text), normalization (convert gold and prediction to a common SymPy form), and comparison ([[math-verify]] Core Insight). Extraction fails when the answer is outside a recognized environment (a bare `x = 1` extracts nothing). Normalization fails on forms it does not know. Comparison is asymmetric by design: a gold inequality accepts a predicted interval but not the reverse, and a gold number accepts a prediction chain ending in that number but not the reverse, so that restating the question does not count as answering it (README FAQ). The README recommends passing gold first, storing gold answers in canonical form, using the stricter reward settings (`malformed_operators=False`, `nits=False`, `equations=False`, `boxed="all"`), and not mixing `StringExtractionConfig()` with other configs, because first-match string extraction can pick stray letters A–D from ordinary text.

verl's Math-Verify wrapper follows the gold-first rule, runs in a process pool so that a timeout can interrupt symbolic computation, and returns `timeout_score` (default 0) on timeout:

```python
# verl@8357f1e verl/utils/reward_score/math_verify.py L42-66
def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score
```

Two details matter for RL. The `max` over all extracted predictions means a response that writes several candidate answers is accepted if any one matches, which raises recall and gives the policy an incentive to list candidates. And every exception path returns 0, so infrastructure errors are indistinguishable from wrong answers in the reward. Neither is wrong by itself; both should be measured (§8).

open-instruct's `MathVerifier` tries `\boxed{}`, then Minerva-style `Answer:`, then the last LaTeX span, then the whole output, and accepts if any candidate is equivalent (`ground_truth_utils.py` L245–300 at f020e9a). verl's `math_dapo` scorer is the strict opposite: last 300 characters, exact string comparison, ±1 (ch-09 §5). Strict graders are safe only for canonical references.

**Measured recall.** [[verifier-pitfalls]] ran three rule verifiers (VERL / Qwen-Math / HF Math-Verify) on responses from many models (Table 5):

| Dataset of responses | VERL | Qwen-Math | HF Math-Verify |
|---|---|---|---|
| MATH | 0.92 | 0.95 | 0.95 |
| DeepScaleR | 0.86 | 0.94 | 0.94 |
| ORZ-Math | 0.89 | 0.94 | 0.95 |
| Skywork-OR1 | 0.78 | 0.86 | 0.88 |

Precision was about 0.99–1.00. Recall averaged 86%, so about 14% of correct responses were marked wrong, and recall was lowest on responses from the strongest policy. On WebInstruct-Verified (general science), HF recall was 47% (§4.3). Named failure types: equivalent forms in context (0.5π vs 90°) and unit conversion (3 hours vs 180 minutes). Outside math, General-Reasoner found rule matching agreed with Gemini-2.0-Flash on 22.2% of 50K answers Gemini judged correct ([[general-reasoner-verifier]] §5.3).

The trend with policy strength matters for a long run: as the policy improves, its correct answers take more varied forms, and the rule verifier's recall falls. A false-negative rate that was acceptable at step 0 can grow.

**Constraint checkers for instruction following** are rule verifiers too. open-instruct's `IFEvalVerifier` checks each constraint with a registered function and returns the fraction satisfied (`ground_truth_utils.py` L366–381), while the Olmo 3 report describes the IF reward as binary, "1 if all constraints are satisfied, and 0 otherwise"; the card marks this as an unresolved code–paper discrepancy ([[olmo-3-rl-pipeline]] Verification). ch-12 discusses partial versus binary credit.

---

## §3 Code execution

Execution graders run the response against tests. They are rules over program behavior, with their own error sources.

- **Test quality.** Kimi k1.5 generated 50 tests per problem and kept a test only if at least 7 of 10 ground-truth submissions agreed on it; a problem was kept only if at least 9 of 10 submissions passed all selected tests; 323 of 1,000 problems survived ([[kimi-k1-5-recipe]] §2.3.5). R2E-Gym measured that for most problems fewer than 20% of generated tests distinguish correct from incorrect patches, and named "toxic tests" as a failure type ([[r2e-gym]] §4.2, Fig. 5).
- **Resource limits.** MAI's competitive-coding problems carry runtime and memory limits ([[mai-thinking-1]] Sec. 3.2.2). slime's ReTool sandbox uses 120 s and 4 GB per process ([[slime-agentic-and-mismatch-examples]]).
- **Grader isolation.** slime's coding agent grades the agent's diff in a fresh sandbox ("exit 0 == solved") ([[slime-agentic-and-mismatch-examples]]); MAI scrubs git history and resets test files ([[mai-thinking-1]] Sec. 3.3.1 via card). ch-08 §3 covers both.
- **Output comparison.** AReaL's Daytona example compares stdout after `strip()`: `return 1.0 if result.stdout.strip() == expected_stdout.strip() else 0.0` (`examples/sandbox_daytona/reward_example.py` L11–14) ([[areal-agent-pipelines]]). Exact output matching is a rule verifier with the false-negative profile of §2 (formatting differences fail).
- **Partial credit.** Olmo 3's code verifier supports pass-rate or all-pass rewards, run on AWS Lambda ([[olmo-3-rl-pipeline]] Key Contributions).

R2E-Gym's main result is about test-time selection, not RL reward, but it shows the limits of each kind of grader: a test-generating agent and a learned trajectory verifier each plateaued near 43% Best@26 (43.7% and 42.8%), and the hybrid reached 51.0%; the learned verifier lost 5.2 points when it saw only the final patch, which shows it relied on trajectory text ([[r2e-gym]] §4.2, Table 4, Fig. 7a). A verifier that relies on the agent's narration rewards narration.

---

## §4 Model-based verifiers and the hybrid grader

**Static accuracy on what rules miss.** [[verifier-pitfalls]] evaluated model verifiers on the responses HF Math-Verify rejected (Table 1, precision / recall):

| Verifier | Precision | Recall |
|---|---|---|
| DS-R1-Distill-Qwen-7B | 0.79 | 0.59 |
| Qwen2.5-7B | 0.84 | 0.51 |
| R1-Distill-Verifier-1.5B | 0.73 | 0.62 |
| xVerify-3B-I | 0.90 | 0.78 |
| general-verifier | 0.90 | 0.86 |
| random | 0.17 | 0.44 |

**Hybrid in RL.** Rules first, model verifier only on rule-rejected responses, raised the Qwen2.5-7B GRPO average from 55.0 to 57.3 (Table 2). The structure confines the model verifier's false positives to the rule-rejected subset, which is smaller than the full batch and contains fewer correct responses.

**Discriminative vs generative fallback.** With a trained generative verifier as the reward, hacking appeared after about 450 iterations; the dominant exploits were "Single Symbol" (e.g. `{`) and "Gibberish", and Skywork-OR1 performance fell from 58.7 to 55.5 (§5.2, App. I). Under adversarial prefixes, R1-Distill-Verifier-1.5B had 35.0% attack success (base 21.7), general-verifier 22.1%, xVerify-0.5B-I 0.0%, xVerify-3B-I 0.2% (Table 3). Scale did not remove vulnerability: Qwen2.5-Math-72B 12.0% vs Qwen2.5-72B 1.2%. The discriminative xVerify-3B-I kept training reward within 0.004 of the GPT-4o oracle with no late collapse. The paper's guidance (§7): use rules alone only when both precision and recall are high; otherwise use a hybrid; prefer discriminative verifiers in the RL loop.

**Small generative verifiers outside math.** General-Verifier (1.5B, from Qwen2.5-Math-1.5B) writes a short CoT and outputs true/false given question, reference and response ([[general-reasoner-verifier]] §3.2). In a matched run it beat the rule verifier on every reported metric (MMLU-Pro 60.1 vs 58.1 at 120 steps; the rule verifier plateaued near 58% from step 60; Table 5, Fig. 4). [[master-rm-one-token]] measured the same General-Verifier at 9.7% average and 87.0% worst-case FPR against master keys, including 66.8% on MATH with a single blank space (Table 1). A verifier can be better than rules on recall and still have an exploitable false-positive mode; both numbers are needed.

open-instruct ships the General-Verifier prompt, which instructs the judge not to solve the problem itself:

```python
# open-instruct@f020e9a open_instruct/judge_utils.py L124-135
web_instruct_general_verifier_template = """User: ### Question: {input}


### Ground Truth Answer: {label}


### Student Answer: {output}


For the above question, please verify if the student's answer is equivalent to the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.
If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output Final Decision: No. Assistant:"""
```

**CoT reward models in production.** Kimi k1.5 compared a classic value-head math RM (about 800k data points) with a CoT RM that reasons before a JSON judgment (about 800k CoT-labeled examples); manual spot-check accuracy was about 84.4 vs 98.5, and the CoT RM was used in RL ([[kimi-k1-5-recipe]] §2.3.5; spot-check size not printed).

The figure's calculator combines a rule verifier's recall with a model verifier's recall on rule-rejected responses. With HF Math-Verify's 0.88 on Skywork-OR1 outputs and xVerify-3B-I's 0.78, the hybrid's recall is 0.88 + 0.12 × 0.78 ≈ 0.974. The hybrid's false-positive rate depends on the model verifier's false-positive rate on incorrect responses, which [[verifier-pitfalls]] Table 1 reports only as precision on a specific mix; the calculator leaves it as an input.

---

## §5 LLM judges: reference, prompt, parsing

**What the judge sees.** open-instruct's `LMJudgeVerifier` strips the reasoning section, formats the judge prompt with the query, the final answer and the label, and retries up to three times with exponential backoff; after three failures it returns 0.0:

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L812-815, L859-868
        final_answer = extract_final_answer(prediction, selected_input=selected_input)
        prompt = self.prompt_template.format(input=query, output=final_answer, label=label)

        max_retries = 3  # for rate limits
...
                reasoning, score = self.parse_completion(response)
                cost = self.get_cost(response, self.verifier_config.llm_judge_model)
                # normalize score to be between 0 and 1
                return VerificationResult(score=score, cost=cost, reasoning=reasoning)

            except Exception as e:
                logger.warning(f"LLM judge attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM judge failed after {max_retries} attempts. Returning default score of 0.0")
                    return VerificationResult(score=0.0, cost=0.0, reasoning=f"Error: {str(e)}")
```

The judge templates (`judge_utils.py` L30–103) include a quality judge (1–10 score), a reference-based judge with the note that "sometimes the reference answer is not the only answer", and a rubric judge whose overall score is 0 if any criterion scores 0. Olmo 3 used Qwen3 32B with thinking off, 32,768-token input and 2,048-token output as its judge ([[olmo-3-rl-pipeline]] Key Contributions). AReaL's search judge compares only the first 200 characters of the predicted answer with the gold answer ([[areal-agent-pipelines]] `react_agent.py` L274–317), a deliberate truncation that limits what a long answer can do to the judge.

**Parsing is part of the grader.** A judge's text must become a number. open-instruct's `coerce_score` docstring records a bug it fixes: the factuality and refusal templates ask the judge to answer "True" or "False", `float()` raised on that, "and the except path scored every verdict 0.0, making correct and incorrect answers indistinguishable" (`judge_utils.py` L210–225). A grader that silently returns a constant makes every group zero-variance, which dynamic sampling then filters out (ch-10 §5); the run looks like it has no learnable prompts rather than like a broken grader. Log the distribution of judge outputs and parse failures per step.

**Reference answers help.** For math and reasoning prompts, giving GPT-4 an independently generated reference answer lowered its failure rate on 10 math questions from 14/20 to 3/20 ([[judge-llm-bias]] §3.4, Table 4). [[rubrics-as-rewards]] found reference-conditioned rubrics better than rubrics without references (35.9 vs 32.0).

**References also open an exploit.** Reference-based judges that decide whether a response matches a reference accepted content-free responses — a blank space, ".", ":", "Thought process:", "Solution", or 解 — at average / worst-case FPRs of 66.8 / 90.9% for Qwen2.5-72B, 80.6 / 95.1% for LLaMA3-70B, 14.6 / 53.6% for GPT-4o, 12.4 / 57.7% for Claude-4, and 6.0 / 37.3% for GPT-o1 ([[master-rm-one-token]] Table 1). The issue was found when an RLVR policy collapsed to responses under 30 tokens such as "Solution." that the Qwen2.5-72B judge rewarded (§1, Figs. 2–3). Inference-time fixes did not work: CoT prompting raised Qwen2.5-7B's average FPR from 12.6 to 40.4 (Table 3a; column order inferred by the card's reader, treat as approximate), and FPR was non-monotonic in model size, lowest around 7–14B (Fig. 4). Removing the question from the judge prompt cut Qwen2.5-72B's MATH FPR from 78.7 to 2.8, but general reasoning often needs the question (Table 3b). The fix that worked was training data: 20K regenerated responses truncated to their first sentence and labeled NO, mixed into a 160K verification set; Master-RM-7B reached 0.1 / 2.9% average / worst FPR and κ = 0.91 against GPT-4o (§3.2, Tables 1–2).

**Position bias.** GPT-4 as a pairwise judge was consistent under order swap on 65.0% of pairs, GPT-3.5 on 46.2%, Claude-v1 on 23.8% ([[judge-llm-bias]] Table 2); few-shot judging raised GPT-4's consistency to 77.5% at 4× the API cost (Table 12). The conservative rule is to call the judge in both orders and count a win only when both agree. MAI's blind grading of suspect ground truths presents the two answers "in randomized order" for the same reason ([[mai-thinking-1]] Sec. 3.2.1).

**Agreement with humans.** GPT-4 agreed with human experts on 85% of non-tied MT-Bench votes, above the 81% agreement among the experts themselves ([[judge-llm-bias]] Table 5a). Agreement is a property of the judge, the task and the prompt; it must be re-measured on the policy's own outputs.

---

## §6 Rubrics and checklists

**Prompt-specific rubrics.** Rubrics as Rewards (RaR) gives a gpt-4o-mini judge a prompt-specific rubric (weighted binary criteria). On HealthBench with Qwen2.5-7B and GRPO, the implicit variant (judge sees all criteria, returns one score) reached 31.2, against 29.7 for the explicit weighted sum, 25.5 for a rubric-free Likert judge, 28.9 for a reference-Likert judge, and 12.5 for four generic rubrics shared by all prompts (Fig. 2 bar labels); on GPQA-Diamond 37.6 vs 34.8 for Likert ([[rubrics-as-rewards]]). Rubrics generated with reference answers beat rubrics without them (35.9 vs 32.0, Table 1). Ablations of rubric elements showed small differences (Essential-only 34.9; no pitfall criteria 37.2; all rubrics 37.2; Table 2). Tested only on one 7B policy for 300 steps on medicine and science.

**Checklists from candidates.** RLCF generates a per-instruction checklist from the instruction plus responses of several model sizes (candidate-based), scores each item 0–100 with a Qwen2.5-72B judge averaged over 25 samples, adds a verification program only where one can check the item exactly, and weights items by importance ([[rlcf-checklists]] Summary). Candidate-based checklists beat direct ones on IFEval (77.3 vs 74.3) and FollowBench (75.3 vs 72.8) (Table 6). The method was used to select DPO pairs, not as an online RL reward, but the grader design is the same. As a general reward model it was weak (RewardBench Safety 71.4 vs 93.0 for Skywork-27B, Table 5): a grader specialized to instruction requirements is not a general preference model.

**Per-rubric binary judgments.** MAI-Thinking-1's instruction-following reward uses an LLM judge that "independently evaluates atomic rubrics, producing a binary judgment per rubric. Multiple judgment passes are averaged for robustness" ([[mai-thinking-1]] Sec. 3.4.2, read from the PDF). MultiChallenge measured why item-level binary questions help: a judge given a per-item yes/no rubric question agreed with human raters on 93.95% of judgments, against 37.33% for a judge given the full conversation and category criteria ([[multichallenge]] Table 4).

**An open implementation.** open-instruct's `RubricVerifier` scores each rubric item in parallel with a separate judge call, normalizes the judge's 0–2 score to 0–1, and returns a weighted average whose denominator is the sum of absolute weights, so negative-weight ("pitfall") items lower the score:

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L1195-1213
                obj = extract_json_from_response(resp)
                if obj and isinstance(obj, dict) and "score" in obj:
                    score = float(obj["score"]) / 2.0  # Normalize from 0-2 to 0-1
                    return score, weight
            except Exception as e:
                logger.warning(f"Error scoring rubric: {e}")

            return 0.0, weight

        # Run all rubric scoring in parallel
        results = await asyncio.gather(*[score_rubric(r) for r in rubrics])

        # Compute weighted average
        total_weight = sum(abs(w) for _, w in results)
        if total_weight == 0:
            return VerificationResult(score=0.0)

        weighted_sum = sum(s * w for s, w in results)
        final_score = weighted_sum / total_weight
```

A failed judge call scores the item 0 but keeps its weight, so infrastructure errors lower the reward in proportion to item weight.

**Coarse vs prompt-specific for style.** MAI's style judges use a coarse 0 / 1 / 2 scale (major / minor / no issues), and the report states that coarse graders "outperform more granular graders or prompt-specific rubrics" for style because the judge has more flexibility and the grader is harder to hack ([[mai-thinking-1]] Sec. 3.4.5, read from the PDF; no numbers). RaR's finding for expert medical and science answers points the other way. The two results are from different domains and are not a controlled comparison; for a new domain, test both.

**Rubric-induced pressure.** MAI reports that "AI feedback rubrics tend to induce an upwards pressure on length and style elements", which it counters with the reward model and with a verifiable length term that penalizes responses outside a per-context quantile range of acceptable lengths, computed offline ([[mai-thinking-1]] Sec. 3.4.1). Rubicon adds a veto rubric that zeroes the reward for prefatory praise of the user and self-evaluation, after early runs showed reward spikes from rubric exploitation ([[rubicon-rubric-anchors]] Guideline; no number for the veto's effect).

---

## §7 Generative reward models, sampling and voting

**Principles, critique, score.** DeepSeek-GRM generates its own principles and critiques and then a 1–10 score per response; it is trained by rejective fine-tuning and then rule-based online RL ([[deepseek-grm]] Summary). Sampling it k times and summing scores raised DeepSeek-GRM-27B from 67.9 (Voting@1) to 70.6 (Voting@8), and a meta-RM that keeps the better half of samples gave 72.8 at k = 32, above GPT-4o's 71.3 on the paper's RM benchmarks (Tables 2–3). These are RM-benchmark accuracies; the paper read does not report policy RL with this RM. The stated risks include unfaithful principles and critiques.

**Self-generated rationales.** GenRM trains a judge by DPO on its own correct versus incorrect reasoning chains (STaR-DPO); with Llama-3.1-8B-Instruct on UltraFeedback it matched Bradley-Terry, PairRM and GenRM models in-distribution (73.9% vs about 73–74%) and scored 81.9% on RewardBench; majority voting at 32 added 1.6–4.9% depending on data ([[generative-reward-models]] §5.1, §5.4). One 8B model, preference-benchmark accuracy only.

**MAI's reward model** predicts k-way 1–5 scores as text over a side-by-side context of k responses, trained by SFT on human preference data only. Because later positions are noisier for an autoregressive model, it is applied cyclically over the k rotations of the response list; for each call only the first token is decoded and its distribution gives the score of the first-listed response; the reward is P(score = 5) ([[mai-thinking-1]] Sec. 3.4.1, read from the PDF). The report says this gave "a more stable climbing signal than alternative formulations we considered" (no numbers). Cyclic application is the multi-response version of order swapping.

---

## §8 Pairwise and group judges

Rocket runs group graders on the problem worker once all rollouts of a problem are complete ([[mai-rl-infrastructure]] p. 3). Rocket v1's example is an LLM performing "pairwise comparisons of different model answers"; a problem may have several grades, aggregated by "a user-defined policy" ([[mai-thinking-1]] Sec. 3.6.3).

A group judge produces relative information, which matches what a group-relative advantage uses. Writing-Zero's BRPO scores each rollout pairwise against a randomly chosen rollout from the same group (advantage ±1) and filters groups whose preferences are nearly all one sign (|ΣR_i|/G > 0.6), because an outlier reference yields all-+1 or all-−1 groups. Against a scalar-RM GRPO baseline, mean response length was 1,292 vs 1,872 and redundant self-explanation 58 vs 417 tokens, and the held-out writing score was 3.84 vs 2.83 ([[writing-zero]] Tables 2–3). The scalar-RM policy produced "gibberish" and still scored highest on WritingBench (8.87), which is why an independent evaluator is required.

Costs and biases specific to group judges:
- **Latency.** The judge waits for the slowest rollout, and every rollout in the group accumulates staleness while it runs (ch-03). Per-rollout judging can start as each rollout ends.
- **Position bias** scales with the number of comparisons; apply swaps or rotations (MAI's cyclic RM).
- **Context limits.** A k-way judge must fit k responses; long rollouts limit k.

---

## §9 Judge hacking and audits

**Known exploits and their signatures.**

| Exploit | Signature in training logs | Source |
|---|---|---|
| Content-free "master keys" (":", "Thought process:", "Solution") | collapse to very short responses with rising reward | [[master-rm-one-token]] §1 |
| Single symbol / gibberish | training reward diverges from oracle reward | [[verifier-pitfalls]] §5.2 |
| Length and style inflation from rubric judges | rising length; more formatting | [[mai-thinking-1]] Sec. 3.4.1 |
| Self-justification, self-praise | explanation tokens rise (417 vs 58) | [[writing-zero]] |
| Multiple candidate answers | more answers per response; recall rises | max-over-candidates in verl Math-Verify wrapper (§2) |
| Reliance on trajectory narration | verifier score drops when only the patch is shown (−5.2) | [[r2e-gym]] Fig. 7a |

Prompt injection into the judge through the response text is a plausible further exploit; none of the sources read for this chapter measures it.

**Audits before the run.**
1. Measure each rule verifier's recall and precision on a hand-labeled sample of the starting policy's outputs, per data source.
2. Probe every model judge with master-key responses and the [[verifier-pitfalls]] adversarial patterns; reject a judge whose FPR on them is not near zero.
3. Measure judge–human agreement on the policy's outputs (MultiChallenge and MT-Bench style), and order consistency for pairwise judges.

**Audits during the run.**
1. Track training reward against an independent oracle on a sample: [[verifier-pitfalls]] detected verifier hacking by tracking training reward against GPT-4o (§5).
2. Inspect the shortest and longest high-reward responses each N steps; both judge collapses in the sources were found by looking at degenerate outputs ([[master-rm-one-token]] §1; [[verifier-pitfalls]] §5).
3. Log grader failures and parse failures separately from zero scores.
4. Re-measure rule-verifier recall periodically, since it falls with policy strength.

---

## §10 The grader as a service: latency and cost

Graders that call models are inference workloads.

- **Cost per call scales with samples.** RLCF's 25 judge samples per item on WildChat took 92 hours against 40 hours for 5 samples on one 8×H100 node ([[rlcf-checklists]] §5.7). DeepSeek-GRM's gains come from k = 8–32 samples per response ([[deepseek-grm]] Table 3).
- **Separate serving.** AReaL serves its judge as a second SGLang engine with its own allocation and disables its staleness admission bound, since the judge's weights never change ([[areal-agent-pipelines]] `train.py` L127–146). verl's COLOCATED mode serves a judge in the same placement group without weight sync ([[verl-rollout]]).
- **Parallelism and retries.** open-instruct runs all verifiers for a batch with `asyncio.gather`, retries judge calls with backoff, checks the judge's context window and truncates messages, and records cost per call from a price table (`ground_truth_utils.py` L799–871; `judge_utils.py` L8). rStar2-Agent offloads answer verification to its CPU tool-execution service ([[rstar2-agent]] §3.1).
- **Timeouts.** verl's Math-Verify wrapper times out after 30 s and returns 0; slime's coding grader has a 600 s command timeout ([[slime-agentic-and-mismatch-examples]]).
- **Speed of iteration.** MAI uses AI judges because they "can be adapted quickly and customized to any given context", providing "a fast lever to shape the behavior of the model in targeted ways without incurring delays for reward model data collection and retraining" ([[mai-thinking-1]] Sec. 3.4.1). It also states that verifiable rewards are "less prone to reward hacking", "less sensitive to multi-epoching", and "generally useful for stabilizing training" than non-verifiable ones (no numbers).

Grader latency adds directly to rollout duration and therefore to staleness. A judge that takes a minute per rollout on an instruction-following task whose rollout median is 2 minutes ([[mai-rl-infrastructure]] Fig. 2) increases that task's staleness by half.

---

## Recipe

1. **Choose the rung per domain**: rules for canonical math answers and constraint checks; execution for code; a small verifier (prefer discriminative) for short non-canonical answers; rubric judges for open-ended tasks; pairwise or group judges where only relative quality is defined.
2. **Build a hybrid** where rules exist: rules first, model verifier only on rule-rejected responses.
3. **Write judge prompts** that give the question, the reference (if any), the final answer with reasoning stripped, and per-item binary questions; instruct the judge not to solve the problem.
4. **Make parsing strict and logged**: a parse failure is a separate outcome, not a zero score.
5. **Probe before the run**: master keys, adversarial prefixes, empty and one-token responses, very long responses, responses that list several answers. Reject judges that accept them.
6. **Calibrate**: recall and precision of rules on a labeled sample of the starting policy's outputs; judge–human agreement; order consistency (swap or rotate).
7. **Serve the judge separately** with its own capacity, timeouts and retries; record cost and latency per call.
8. **Audit during the run**: oracle reward on a sample, extreme-length high-reward samples, grader-failure rates, periodic re-measurement of rule recall.
9. **Patch the grader when an exploit is found**, and restart or roll back if the policy has already learned it (ch-12 §9).

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Measuring verifier recall on another model's outputs | false negatives grow as policy improves | recall on a labeled sample of current policy outputs |
| Model judge on every response instead of a hybrid | judge FP applied to the whole batch | compare reward with rules-only reward |
| Generative verifier as sole reward | collapse to symbols or gibberish | training vs oracle reward |
| Reference judge without master-key probing | short content-free responses rewarded | probe set; response-length trace |
| CoT prompting the judge to fix FPR | FPR unchanged or higher | re-run probe set with and without CoT |
| Pairwise judge without order swap | reward depends on position | swap consistency rate |
| Parse failures scored as 0 | constant rewards; groups filtered as zero-variance | parse-failure counter; judge output histogram |
| Infrastructure errors scored as 0 | reward correlated with cluster load | error counter separate from score |
| `max` over all extracted answers | responses list multiple candidates | answers-per-response metric |
| Rubric judge without length control | length and formatting rise | length quantiles per context (MAI) |
| Judge latency ignored | staleness of short tasks rises | grader time as a fraction of rollout time |

---

## Check your understanding

1. Rule verifiers had near-perfect precision and recall that fell on stronger policies' outputs. Explain why recall falls with policy strength, and why a false negative and a false positive have different consequences under GRPO.
2. The hybrid grader applies the model verifier only to rule-rejected responses. Why does this confine the model's false positives, and in what situation does the hybrid still give the policy an exploitable target?
3. Master-RM was fixed by adding truncated reasoning openers labeled NO to its training data, while CoT prompting did not help. Explain why a data fix addresses this failure and a prompting fix does not.
4. General-Verifier beat rule matching on downstream accuracy and also had 87.0% worst-case FPR against master keys. How can both be true, and what does it imply about selecting graders by downstream accuracy alone?
5. Why do per-item binary rubric questions agree with humans much more than a holistic judgment (93.95% vs 37.33%), and what does that imply for how MAI structures its instruction-following judge?
6. MAI applies its k-way reward model cyclically over rotations. Relate this to order swapping for pairwise judges, and explain what noise it removes.
7. A grader that returns 0 on any exception can make a broken run look like a run with no learnable prompts. Trace the chain from exception to dynamic sampling to what the logs show.

---

## Connections

- ch-08 (where per-rollout and group graders run; sandbox isolation for execution graders), ch-09 (reference formats that make rules safe), ch-10 (dynamic sampling that hides constant-output graders), ch-12 (combining grader outputs; hacking monitors), ch-14 (instruction-following graders), ch-15 (search graders), ch-16 (math, code and SWE graders), ch-19 (graders without a reference for subjective qualities).
- Algorithm mechanics: [llm-training ch-44](../../llm-training/ch-44/read.md) (verifiable rewards and process supervision), [llm-training ch-44b](../../llm-training/ch-44b/read.md) (non-verifiable rewards across domains), [llm-training ch-40](../../llm-training/ch-40/read.md) (how group advantages consume grader outputs).
- Sources: [[rl-design-guide]] (§3 outline), [[math-verify]], [[verifier-pitfalls]], [[master-rm-one-token]], [[deepseek-grm]], [[general-reasoner-verifier]], [[rlcf-checklists]], [[rubrics-as-rewards]], [[generative-reward-models]], [[judge-llm-bias]], [[reward-hacking-taxonomy]], [[mai-rl-infrastructure]], [[mai-thinking-1]], [[kimi-k1-5-recipe]], [[r2e-gym]], [[olmo-3-rl-pipeline]], [[areal-agent-pipelines]], [[slime-agentic-and-mismatch-examples]], [[writing-zero]], [[multichallenge]], [[rubicon-rubric-anchors]], [[rstar2-agent]], [[verl-rollout]].

---

## Sources

- [[rl-design-guide]] — §3 grader rules used as this chapter's outline.
- [[math-verify]] — extraction / normalization / comparison, asymmetric verify, strict reward settings.
- [[verifier-pitfalls]] — rule recall by dataset, model verifiers on rule-rejected responses, hybrid RL result, verifier hacking, adversarial attack rates.
- [[master-rm-one-token]] — master-key FPRs across judges, RL collapse case, Master-RM training data and agreement, CoT and question-removal tests.
- [[deepseek-grm]] — principles/critique GRM, voting and meta-RM scaling (RM benchmarks).
- [[general-reasoner-verifier]] — 1.5B generative verifier, agreement vs rules outside math, matched RL comparison.
- [[rlcf-checklists]] — candidate-based checklists, judge sampling cost.
- [[rubrics-as-rewards]] — prompt-specific rubrics, reference-conditioned rubrics, implicit vs explicit aggregation.
- [[generative-reward-models]] — STaR-DPO judge, majority voting.
- [[judge-llm-bias]] — position bias, reference answers for math, human agreement.
- [[reward-hacking-taxonomy]] — impossibility of non-trivial unhackable proxies on open policy sets.
- [[mai-rl-infrastructure]] — per-rollout vs group graders in Rocket; Fig. 2 durations.
- [[mai-thinking-1]] — graders per domain, per-rubric binary judging, cyclic k-way RM, length-quantile penalty, AI-judge rationale (Sec. 3.3–3.4, 3.6.3, read from the PDF).
- [[kimi-k1-5-recipe]] — test-case validity voting, CoT RM vs classic RM spot checks.
- [[r2e-gym]] — test distinguishability, hybrid test + learned verifier, reliance on trajectory text.
- [[olmo-3-rl-pipeline]] — per-domain verifiers, judge model settings, IF code–paper discrepancy.
- [[open-instruct-rlvr-pipeline]] — verifier dispatch, judge prompts and parsing, rubric verifier (code read in the clone at f020e9a).
- [[areal-agent-pipelines]] — search judge on first 200 characters, judge as separate engine, stdout grader.
- [[slime-agentic-and-mismatch-examples]] — fresh-sandbox SWE grading, sandbox limits.
- [[writing-zero]] — pairwise in-group judge, group filter, scalar-RM hacking signatures.
- [[multichallenge]] — per-item binary rubric agreement with humans.
- [[rubicon-rubric-anchors]] — veto rubric against praise and self-evaluation.
- [[rstar2-agent]] — verification offloaded to the tool service.
- [[verl-rollout]] — colocated judge mode.
