<!-- chapter: ch-16
     track: rl-pipeline
     kind: content
     title: Math, Code, and SWE RL Pipelines
     deps: [ch-12]
     sources: [[swe-rl]], [[r2e-gym]], [[deepswe]], [[rstar2-agent]], [[deepseek-v3-2]], [[deepseek-v3.1]], [[swe-smith]], [[slime-agentic-and-mismatch-examples]], [[areal-agent-pipelines]], [[minimax-m1]], [[mai-thinking-1]], [[mai-rl-infrastructure]], [[math-verify]], [[verifier-pitfalls]], [[open-instruct-rlvr-pipeline]], [[olmo-3-rl-pipeline]], [[magistral]], [[dapo]], [[big-math]], [[kimi-k2]], [[glm-4-5]], [[demystifying-agentic-rl]]
     figures: none
-->

# Chapter 16 — Math, Code, and SWE RL Pipelines

> **Core insight.** Math, competitive code and SWE are the three verifiable domains, and they differ in what the reward needs. Math needs a string-to-expression comparison, which runs in-process and is limited by recall: rule verifiers miss 14% of correct answers on average, and on general science recall falls to 47% ([[verifier-pitfalls]] Table 5, §4.3). Competitive code needs program execution against tests in a sandbox, and the decisions are how many tests to run and whether partial passes count. Magistral's fraction-of-tests reward gave 2% lower LiveCodeBench than binary pass/fail ([[magistral]] §7.4.1). SWE needs a whole repository environment per task, built through a filtering funnel: MAI started from 102M public PRs and kept 265,617 verified environments ([[mai-thinking-1]] p. 42–43). SWE also needs a multi-turn agent loop and a grading step that only the model's diff can influence. The three domains sit at different points of the duration distribution in one production job: median 7 minutes and 1 turn for math & science, 34 minutes and 1 turn for competitive coding, 93 minutes and 249 turns for SWE ([[mai-rl-infrastructure]] Fig. 2). For SWE the decisive design question is what to do with a rollout that fails or times out, and the frameworks and labs make four different choices.
>
> **Guideline.** For math, measure the rule verifier's recall on your own policy's outputs before training, and add a model verifier only on rule-rejected responses, because the hybrid raised a Qwen2.5-7B GRPO average from 55.0 to 57.3 ([[verifier-pitfalls]] Table 2). Use a discriminative verifier for that fallback, because a generative verifier was hacked after about 450 iterations (§5.2). For code, run all tests and use a pass/fail reward (Olmo 3's scripts set `--code_pass_rate_reward_threshold 0.99`, `7b_think_rl.sh` L63), and isolate execution in a real sandbox rather than a pattern filter. For SWE, grade in a fresh container that receives only the model's diff, scrub history and reset test files (MAI p. 42–43; slime `swe.py` L290–298), and choose the failure policy explicitly. Mask trajectories that hit context, step or time limits if you can tolerate the bias, because DeepSWE's ablation shows this prevents or delays reward collapse ([[deepswe]] §2.3, Fig. 6; curve only). Drop them if throughput matters more (Rocket, p. 7). Train sandbox-terminated trajectories with reward 0 if they contain real actions (slime `generate.py` L369–392).

---

## Why this chapter matters

ch-14 and ch-15 covered domains whose graders are a checker or a judge. This chapter covers the domains whose graders are programs that must run correctly at scale. The comparison shows how much of an RL pipeline is not the RL algorithm. For SWE, MAI's report spends its detail on the environment funnel, not on the objective ([[mai-thinking-1]] p. 42–43). DeepSWE's blog reports that thousands of containers crashed the Docker daemon, and the authors moved scheduling to Kubernetes ([[deepswe]] §2.2).

The three domains also run together in production. Rocket's job mixes math, coding and SWE rollouts whose durations differ by more than 10× (Fig. 2). MAI runs its agentic climb jointly with STEM, stating that this stabilizes the climb ([[mai-thinking-1]] p. 40). ch-17 covers the mixing. This chapter gives each domain's requirements separately, so that the mixing decisions have known inputs.

---

## §1 The three domains side by side

| Property | Math | Competitive code | SWE |
|---|---|---|---|
| Rollout shape | 1 turn; long CoT ending in a final answer | 1 turn; long CoT ending in a program | many turns; tool calls in a repository container; ends with a patch |
| Rocket median duration / turns | 7 min / 1 (math & science) | 34 min / 1 | 93 min / 249 |
| Grader | extract answer → normalize → compare to reference | compile and run against tests with time and memory limits | apply diff in a clean container → run fail-to-pass (F2P) and pass-to-pass (P2P) tests |
| Grader runs where | in-process (CPU) | sandbox service (CPU) | container fleet (CPU, disk-heavy) |
| Main grader error | false negatives (recall) | weak or wrong tests; partial-pass ambiguity | tests that can be gamed; flaky tests; environment build failures |
| Reference answer | closed-form answer, ideally canonical (DAPO converts all to integers) | reference solution plus tests | gold patch (for F2P validation) plus hidden tests |
| Failure cases to handle | parser timeouts | sandbox errors, timeouts | container crash, agent timeout, context exhaustion |

Rocket durations: [[mai-rl-infrastructure]] Fig. 2, from 1.13M rollouts in one day of a production run. DAPO answer conversion: [[dapo]] §3.5.

---

## §2 Math: answer-matching graders

### §2.1 What a rule verifier does

[[math-verify]] separates three steps, each with its own failure modes: extract an answer from text (LaTeX environment, plain expression, or string), convert gold and prediction to a common SymPy form, and compare them. Its `verify(gold, pred)` is asymmetric on purpose. A gold inequality accepts a predicted interval but not the reverse, and a gold number accepts a prediction chain ending in that number but not the reverse, so restating the question does not count as an answer (README FAQ). Recommended reward settings are `malformed_operators=False`, `nits=False`, `equations=False`, `boxed="all"`. The README also warns not to mix `StringExtractionConfig()` with other extractors, because first-match string extraction can pick up stray A–D letters.

### §2.2 Recall is the bottleneck

| Measurement | Value | Location |
|---|---|---|
| Rule verifier precision | about 0.99–1.00 | [[verifier-pitfalls]] Table 5 |
| Recall, Skywork-OR1 responses, VERL / Qwen-Math / HF Math-Verify | 0.78 / 0.86 / 0.88 | Table 5 |
| Recall, MATH responses | 0.92 / 0.95 / 0.95 | Table 5 |
| HF Math-Verify recall on WebInstruct-Verified (general science) | 47% | §4.3 |
| Hybrid grader (rules, then model verifier on rejects), Qwen2.5-7B GRPO average | 55.0 → 57.3 | Table 2 |
| Generative verifier as RL reward | hacked after about 450 iterations by single-symbol and gibberish outputs | §5.2 |

Recall falls as the generator gets stronger: the lowest numbers are on Skywork-OR1 responses. The paper names two failure types: equivalent forms in context (0.5π vs 90°) and unit conversion (3 hours vs 180 minutes) ([[verifier-pitfalls]] Technical Details). A false negative gives a correct response a negative advantage, which is the worst error for a policy-gradient method.

### §2.3 open-instruct's math verifier

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L267-L293
# Attempt extraction from \boxed{}.
boxed_answer = last_boxed_only_string(raw_answer)
if boxed_answer is not None:
    try:
        boxed_answer = remove_boxed(boxed_answer)
    except AssertionError:
        boxed_answer = None
if boxed_answer is not None:
    all_answers.append(boxed_answer)

# Attempt extraction via Minerva format.
minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
if minerva_answer is not None and minerva_answer != "[invalidanswer]":
    all_answers.append(minerva_answer)

# Attempt extraction from the last LaTeX-formatted answer.
if not all_answers:
    dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
    if len(dollars) > 1:
        answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
        all_answers.append(answer)

# Fallback to the full output.
if not all_answers:
    all_answers.append(normalize_final_answer(prediction))
    # also provide original string in case normalization fails
    all_answers.append(prediction)
```

The compare loop (L296–300, after this excerpt) returns 1.0 if **any** candidate matches. Two consequences follow from reading the code (no measurement of their effect was found):

1. **Two answers, one credit.** A response whose last `\boxed{}` holds one value and whose Minerva-format "Final Answer" line holds another is scored correct if either matches. A policy can learn to state two answers. `StrictMathVerifier` (L303) uses only the Minerva extraction and closes this path.
2. **Thinking text is not removed.** Unlike `IFEvalVerifier`, `MathVerifier` does not call `remove_thinking_section`. `last_boxed_only_string` uses `rfind`, so the last box anywhere in the output counts, including one inside `<think>` if the final answer has none.

This is the precision side of the recall problem in §2.2. Relaxing extraction to raise recall opens hedging paths. The asymmetric `verify(gold, pred)` of Math-Verify is one way to keep recall without accepting restatements.

### §2.4 Reference answers

- MAI's STEM set comes from textbooks, academic PDFs, forums, competition archives and vendors. MCQ and proof items are rewritten to open-ended form (3× plus consensus), 4 solver tiers rate difficulty, and blind grading drops suspect ground truths. Grading uses a SymPy-style verifier, an AI judge or tests ([[mai-thinking-1]] p. 37–38, domain table).
- MiniMax-M1 dedups, separates SFT from RL data, decontaminates, drops proofs, multi-part and binary questions, converts MCQ to open-ended, and keeps items with pass@10 in (0, 0.9), giving about 50K problems ([[minimax-m1]] §4.1).
- Big-Math found 117,018 multiple-choice matches in a 643K pool, items a policy can guess ([[big-math]] Table 2).

### §2.5 Math with a tool

When math is solved with a Python tool, the rollout becomes multi-turn and the grader gains a dependency on a sandbox.

- **rStar2-Agent.** Under a binary answer-only reward, correct trajectories that contain failed tool calls are still reinforced. With plain GRPO, the tool-error ratio inside positive trajectories plateaued near 10% for Qwen3-14B ([[rstar2-agent]] §2.2.2, Fig. 4). GRPO-RoC samples 2G = 32 rollouts and keeps 16: negatives uniformly, positives with probability inversely proportional to a tool-error-plus-format penalty. The model reached 80.6% AIME24 / 69.8% AIME25 in 510 steps (§2.2.3, Table 3). The environment service ran up to 45K tool calls per step at 0.3 s mean latency, with 1,024 execution workers per worker node. The rollout scheduler assigns requests by free KV-cache capacity because SGLang evicts half of in-progress rollouts on KV overflow (§3.1–3.2). The authors rejected step-level rewards and tool-error penalties as prone to reward hacking, without an experiment (§2.2.3).
- **slime ReTool example.** Up to 16 turns; the interpreter runs with a 120 s timeout and a 4 GB memory limit per process (`tool_sandbox.py` L23–28). The reward adds a tool-use bonus to wrong answers: `(num_turns − 2) / 2 * 0.1`, capped so a wrong answer scores between −1.0 and −0.6 (`generate_with_retool.py` L422–428). This shaping rewards more tool calls on failures. Whether it changes behaviour on correct answers is not measured in the example.
- **Isolation.** ReTool's sandbox screens code with a regex denylist before running it in a subprocess (`tool_sandbox.py` L112–135). For example, `r"import\s+os"` blocks `import os` but not `from os import path`, because that string contains `import path`, not `import os`. A denylist is a pattern filter, not isolation. rStar2's design runs code in an isolated execution service ([[rstar2-agent]] §3.1).
- **Cold start for tool use.** SFT on 3K real end-to-end tool-use trajectories gave AIME2025 average@32 of 29.79%, against 3.65% for stitched synthetic trajectories (Qwen3-4B-Instruct-2507) ([[demystifying-agentic-rl]] Table 1).

---

## §3 Competitive code: execution against tests

### §3.1 Graders in practice

| Source | Execution | Reward | Location |
|---|---|---|---|
| open-instruct (Olmo 3) | HTTP service, `/test_program` and `/test_program_stdio`; Olmo 3 points it at an AWS endpoint | pass rate, set to 0 below `code_pass_rate_reward_threshold`; Olmo 3 sets the threshold to 0.99 | `ground_truth_utils.py` L1015–1018; `7b_think_rl.sh` L62–63 |
| Magistral | 20 random tests, 4 s and 300 MB each, C++20 compile timeout 10 s | format 0.1 + 0.9 if all pass | [[magistral]] §2.2 |
| MAI Thinking 1 | test cases with runtime and memory limits; 160k problems, 17 languages, verified reference solutions | not stated beyond tests | [[mai-thinking-1]] Sec. 3.2.2 |
| MiniMax-M1 | missing tests generated by MiniMax-Text-01; pass-rate filtering → 30K problems | test execution | [[minimax-m1]] §4.1 |
| AReaL sandbox example | one Daytona sandbox kept warm across calls; `code_run` with a 5 s timeout | 1.0 iff stdout matches | `examples/sandbox_daytona/reward_example.py` L11–14 |

```python
# open-instruct@f020e9a open_instruct/ground_truth_utils.py L1015-L1018
result = await asyncio.to_thread(make_request)
passes = result["results"]
pass_rate = sum(passes) / len(passes) if passes else 0.0
score = 0.0 if pass_rate < self.pass_rate_reward_threshold else pass_rate
```

With the threshold at 0.99, any failed test sets the reward to 0, so the code reward is effectively binary. An optional performance penalty multiplies each pass by the remaining fraction of the time limit (L1019–1028). Any exception in the request path also returns 0.0 (L1030–1032), so a sandbox outage looks like a failed program to the learner. The same pattern appears in the judge path of ch-14: log grader errors separately from wrong answers.

### §3.2 Partial credit

Magistral tested a proportional reward (fraction of tests passed). It discarded three times less data, because fewer groups had zero variance, but gave 2% lower LiveCodeBench and slower length growth (24B, 250 steps). The authors suggest partial rewards may give "false signal to incorrect solutions" ([[magistral]] §7.4.1, Fig. 11). This is one study at one scale. It is the only measured partial-vs-binary comparison for code in the cards read.

### §3.3 Test quality

Tests are the reference answer for code, and they can be wrong.

- Olmo 3's code pipeline rewrites the problem, generates solutions and tests, keeps solutions that pass more than 80% of tests, and drops failing tests ([[olmo-3-rl-pipeline]] §4.4.2).
- Magistral discards tests with insufficient agreement across solutions, rewrites tests that have agreement but no passing solution to the most common output, and generates missing tests ([[magistral]] §4.2).
- R2E-Gym finds that LLM-generated tests separate correct from incorrect patches poorly: for the majority of problems, under 20% of tests do ([[r2e-gym]] §4.2, Fig. 5). This was measured on SWE tasks, and it is a warning for generated tests in any domain.

---

## §4 SWE: repository environments

### §4.1 Building environments

The cost of an SWE pipeline is in building environments.

**MAI funnel** ([[mai-thinking-1]] p. 42–43): 102M public PRs → 4.87M with linked issues → 2.08M built (42.8%) → 745,452 with an F2P signal (15.3%) → 265,617 verified (5.5%), across 94,044 repositories. Synthetic tasks in BugPilot, SWE-Smith and SWE-Mirror styles are added.

**SWE-smith** ([[swe-smith]] §2.2, Tables 1–2) inverts the direction. It starts from a working repository, synthesizes a bug, and keeps it only if it breaks at least one passing test. This gave 50,137 instances from 128 repositories for about $1,360 and 295 GB of environments, where the authors estimate 50–150 TB for a SWE-bench-style build of the same count. Strategy yields: LM Modify 56.0%, LM Rewrite 35.0%, Procedural 40.2%, Combine 96.9%, PR Mirror 33.8%. In SFT ablations at 1,000 instances, PR-mirror, procedural and LM-rewrite bugs gave 9.2 / 8.6 / 8.8% on SWE-bench Verified, against 5.7% for LM-modify (Table 4). The reported spread is ±1.5–1.8, so the first three are within noise of each other.

**R2E-Gym** builds environments from commits without human-written issues, using thresholds of at most 5 non-test files, 100 edited lines, 2,000 characters of patch, and an LLM-as-judge filter, with more than 8.1K tasks ([[r2e-gym]] §2, App. A).

**DeepSeek-V3.2** counts a code-agent environment as built only if the gold patch gives F2P > 0 and P2F = 0, across Python, Java, JavaScript, TypeScript, C, C++, Go and PHP, giving 24,667 code-agent tasks ([[deepseek-v3.1]] §3.2.3, Table 1).

**SWE without execution.** SWE-RL trains Llama-3.3-70B-Instruct with GRPO on 273k PR seeds, using a reward of −1 for malformed output and otherwise `difflib.SequenceMatcher` similarity to the oracle patch. It reaches 41.0% on SWE-bench Verified without running tests during training ([[swe-rl]] §2.1, Table 1). A discrete exact-match reward stayed near zero because human patches are diverse (§3.6). No execution-reward baseline was run, so the paper does not show that similarity beats execution.

### §4.2 Rewards

| Source | Reward | Location |
|---|---|---|
| MAI Thinking 1 | F2P + P2P hidden tests inside the container | p. 42–43 |
| DeepSWE | 1 if the patch passes a selected sample of P2P and F2P tests within the time limit, else 0; training limit 5 min, official evaluation limit 30 min | [[deepswe]] §2.2 |
| MiniMax-M1 | pass all tests → positive; compile/runtime error or regression → zero or negative | [[minimax-m1]] §4.1 |
| GLM-4.5 | test results; a wrong tool-call format halts the trace with reward 0 | [[glm-4-5]] §3.3.2 |
| slime coding_agent_rl | "exit 0 == solved", reward 1.0 iff solved | `swe.py` L303–307 |

### §4.3 Making the grader unhackable

An SWE agent has write access to the repository, including the tests. MAI's anti-hacking measures are network control, git history scrubbed past the base commit, test files reset, and an LLM monitor ([[mai-thinking-1]] p. 42–43). slime grades in a separate sandbox:

```python
# slime@79aa41a examples/coding_agent_rl/swe.py L290-L298
async def run_evaluation(md: dict, *, diff_text: str, timeout_sec: int, trace=None) -> EvalResult:
    """Uniform entry point: dispatch to the protocol's grader.

    No-test-cheating guarantee (both grading protocols): the eval sandbox is built from
    the same image but starts CLEAN, so only the model-produced diff affects
    reward."""
    if md.get("protocol") == PROTOCOL_SWEBENCH:
        return await _grade_swebench(md, diff_text, timeout_sec, trace=trace)
    return await _grade_scaleswe(md, diff_text, timeout_sec, trace=trace)
```

The clean sandbox removes one class of hack: state left behind in the rollout container. It does not remove diffs that edit test files, unless the grader resets them after applying the diff, which is MAI's "test files reset". Git history scrubbing removes a different hack: reading the fix from future commits. DeepSWE gives a third failure: an agent can pass all tests by chance, "for example by answering correctly in the first 10 steps and editing random files later" ([[deepswe]] §2.3).

### §4.4 Timeouts and failures: four policies

A long SWE rollout can end without a clean result: the context fills, the step limit is reached, the agent or grader times out, or the sandbox dies. The sources handle these cases in different ways:

| Policy | Who | What happens to the trajectory | Stated reason | Bias it introduces |
|---|---|---|---|---|
| **Mask** (no gradient) | DeepSWE "compact filtering": max context, max steps, or 20-minute generation timeout | kept in the group for nothing; loss masked | prevents or delays reward collapse (Fig. 6, curve only); tests can pass by chance | the policy is never penalized for running out of time or steps |
| **Zero reward** | slime coding_agent_rl, on `SandboxTerminatedError` after actions were produced | trained with reward 0 | "A terminal sandbox failure is an observed trajectory outcome" (`generate.py` L369–372) | infrastructure failures become negative advantage for whatever the agent did |
| **Drop** | Rocket: "drop failed rollouts" | removed; no retry | avoids retry gaps that add staleness | "noise or bias in the GRPO baseline", because failures "may share features that never make their way to the learner" ([[mai-rl-infrastructure]] p. 7) |
| **Discard on timeout / abort** | AReaL SWE agent: `asyncio.wait_for(…, timeout)` re-raises and discards (default 1,800 s; the YAML sets 3,600); slime aborts with loss mask `[0]` when no generation happened (`_abort_result`) | removed | not stated | same as drop |
| **Retry** | MAI Thinking 1's Rocket v1 problem worker retried failed rollouts | rerun | not stated; the Base 2 post later drops them | adds wall-clock delay and staleness |

Sources: [[deepswe]] §2.3; [[slime-agentic-and-mismatch-examples]], `generate.py` L369–392, L446–455; [[mai-rl-infrastructure]] p. 7; [[areal-agent-pipelines]], `agent.py` L186–205; [[mai-thinking-1]] Sec. 3.6.

```python
# slime@79aa41a examples/coding_agent_rl/generate.py L369-L372
except SandboxTerminatedError as error:
    # A terminal sandbox failure is an observed trajectory outcome. Keep
    # its sampled actions and train them with zero reward instead of
    # discarding the trajectory and retrying until execution succeeds.
```

No source compares these policies on the same task. The choice depends on what the failure mostly reflects. If failures come from the agent's behaviour (looping, running out of context), masking them removes the only signal against that behaviour. rStar2 makes the same argument about overlong filtering: it raised the overlong ratio because repetitive outputs lost their only negative signal ([[rstar2-agent]] §4.3.1). If failures come from the infrastructure, zero reward punishes the agent for the cluster's errors. Record the failure cause per trajectory, so that the choice can be made per cause.

### §4.5 The sandbox fleet

| Source | Fleet |
|---|---|
| DeepSWE | 512 Docker containers per RL iteration (batch 64, 8 passes); Docker daemon crashes at thousands of containers led to Kubernetes; about 200 CPU cores and over 6 TB local NVMe with preloaded images per worker node; scales beyond 1,000 cores; autoscaler removes nodes idle for about 20 minutes; 64 H100 for six days ([[deepswe]] §2.2) |
| Kimi K2 | Kubernetes sandbox with over 10,000 concurrent instances ([[kimi-k2]] §3.2.1) |
| GLM-4.5 | asynchronous disaggregated slime for agentic (SWE) RL with a Docker runtime ([[recipe-crosswalk]]; [[glm-4-5]] §3.5) |
| slime coding_agent_rl | default 3 training, 3 inference and 2 sandbox nodes; 168 concurrent agent trajectories; agent / grader / whole-rollout timeouts 3,600 / 600 / 7,200 s; context 131,072 tokens with compaction at 70% at most twice per trajectory (`README.md` L10–24) |
| SWE-smith | one environment per repository: 295 GB for 50,137 tasks ([[swe-smith]] Table 2) |

The DeepSWE numbers give a sizing rule: containers per iteration = prompts per batch × rollouts per prompt (64 × 8 = 512), each running for up to the agent timeout. Image storage is the other constraint. SWE-smith's one-environment-per-repository design cuts storage by sharing images, and DeepSWE preloads images on local NVMe.

---

## §5 What each domain demands from the system

| Component | Math | Competitive code | SWE |
|---|---|---|---|
| **Rollout worker** | one generate call; long outputs (MAI doubles max length 8k → 128k) | one generate call; long outputs | agent loop with tool parser, context compaction, step limit, timeouts; token-in/token-out across turns |
| **Sandbox fleet** | none, unless tool-integrated (rStar2: up to 45K calls per step) | CPU execution service with per-test time and memory limits; high call rate, short calls | one container per rollout for up to an hour or more; image storage; Kubernetes-scale scheduling |
| **Grader** | in-process, fast; recall is the limit; a model fallback adds GPU load | in the sandbox; test quality is the limit; binary vs partial decision | a second clean container; history scrub and test reset; F2P + P2P; flaky-test handling |
| **Failure handling** | parser timeouts → treat as wrong (log separately) | sandbox errors → do not score as wrong silently | explicit policy per cause (§4.4) |
| **Staleness** | low: short rollouts | medium | high: 93-minute median; groups span many learner steps (Rocket Fig. 4) |

**Rocket's comparison point.** Rocket places per-rollout graders on rollout workers and group graders on problem workers. Rollout workers call environments, which include "SEE" and external tool services (Fig. 1, steps 3–5) ([[mai-rl-infrastructure]]). The post does not describe SEE further. The SWE environment pipeline is described in the MAI Thinking 1 report (§4.1), not in the infrastructure post.

---

## §6 Algorithm and recipe notes per domain

- **Math/code.** GRPO with DAPO-style changes is the common base (ch-18). MiniMax-M1 uses CISPO for all its RL domains, including math, logic, competitive programming and SWE ([[minimax-m1]] §3.1).
- **SWE.** DeepSWE's GRPO++ combines clip-high, no KL loss, no std normalization, length normalization by max context length, a leave-one-out advantage, compact filtering and no entropy loss ([[deepswe]] §2.3). Pass@1 rose from 23% to 42% over about 200 RL steps (Fig. 2). Three things did not help: RL after SFT on Claude trajectories, RL on SWE-Smith or SWE-Gym data, and non-thinking mode (§5–6). No card states an SWE-specific reason for choosing the base algorithm ([[algorithm-crosswalk]]).
- **Tool-integrated math.** rStar2 keeps the reward answer-only and moves quality control into rollout selection (RoC), stating that tool-error penalties invite hacking ([[rstar2-agent]] §2.2.3).
- **Joint training.** MAI's agentic climb (SWE + tool use) runs jointly with STEM tasks ([[mai-thinking-1]] p. 40). MiniMax-M1 mixes SWE with math, logic and code from the start of RL run 1 ([[minimax-m1]] §4.3). ch-17 compares these.

---

## Recipe

1. **Math reference answers.** Keep closed-form, uniquely verifiable items. Rewrite or drop MCQ, proofs and multi-part items. Store gold answers in one canonical form.
2. **Math grader.** Use Math-Verify-style asymmetric comparison, gold first. Measure recall on 200–500 hand-checked responses from your current policy. If recall is below precision by more than a few points, add a discriminative model verifier on rule rejects only. Prevent "two answers, one credit" by scoring one designated answer location.
3. **Code tests.** Validate tests against reference solutions. Drop or repair tests that disagree. Run all tests per rollout with per-test time and memory limits.
4. **Code reward.** Start with binary all-pass (or a threshold of 0.99, as in Olmo 3). If zero-variance groups dominate, measure the effect of partial credit on a held-out benchmark before adopting it (Magistral −2%).
5. **Execution isolation.** Run model code in containers or VMs, not behind a regex filter. Return execution errors to the model as observations and log them separately from wrong answers.
6. **SWE environments.** Build per-repository images. Keep only tasks whose gold patch gives F2P > 0 and P2F = 0. Add synthetic bugs (SWE-smith strategies) to raise volume. Budget storage per repository, not per task.
7. **SWE grader isolation.** Grade in a fresh container that receives only the diff. Reset test files after applying the diff. Scrub git history past the base commit. Disable network access. Sample-review trajectories with an LLM monitor.
8. **Timeouts.** Set agent, grader and whole-rollout limits separately (slime: 3,600 / 600 / 7,200 s). Record the cause of every non-clean ending.
9. **Failure policy per cause.** Agent-caused (context or step exhaustion, loops): keep as a negative or mask, and decide by measuring whether the failure rate grows. Infrastructure-caused (sandbox crash): drop or mask, rather than score 0, unless the agent caused the crash. Never retry silently without logging the staleness added.
10. **Fleet sizing.** Containers in flight = concurrent rollouts. Size CPU, NVMe and image cache for that number at the agent timeout. Use Kubernetes-class scheduling above a few hundred containers (DeepSWE).
11. **Mixing.** If math, code and SWE share one job, account for the duration spread (7 / 34 / 93 min medians) in admission and staleness control (ch-04, ch-17).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Unmeasured rule-verifier recall | correct-looking responses with reward 0; reward plateaus early | hand-label a sample of rejected responses each N steps |
| "Any candidate matches" extraction | responses state two final answers | count answer locations per response; score one designated location |
| Generative verifier in the loop without an oracle check | training reward rises while oracle reward falls; short or gibberish responses rewarded | compare training reward with a held-out stronger grader ([[verifier-pitfalls]] §5) |
| Sandbox exceptions scored as 0 | reward dips that track infrastructure incidents | log grader-error rate separately from failures |
| Regex denylist as sandbox | model code touches the filesystem or network | run code in a container; test with `from os import …` patterns |
| Rollout container reused for grading | patches that pass only with rollout-time state | grade in a clean container from the same image with only the diff |
| Test files editable by the agent | rising pass rate with edits to test files | reset test files after applying the diff; diff-scan for test paths |
| Masking all timeouts | timeouts and step exhaustion become more frequent | plot the timeout rate over training; if it rises, the policy is not penalized for it |
| Zero reward for infrastructure failures | negative advantage concentrated in one node or image | tag failures by cause and by host |
| Unsized container fleet | Docker daemon crashes; rollouts stall at high concurrency | load-test at the target concurrency before training (DeepSWE's daemon failure) |

---

## Check your understanding

1. Rule verifiers have precision near 1.0 and recall that falls as the policy improves. Explain why this asymmetry harms a group-baseline RL run more as training progresses, using the advantage of a correct response that is marked wrong.
2. open-instruct's `MathVerifier` returns 1.0 if any extracted candidate matches. Construct a response that exploits this, and explain why `StrictMathVerifier` and Math-Verify's asymmetric comparison each block it.
3. Magistral's partial code reward reduced zero-variance groups threefold and still lowered LiveCodeBench by 2%. Give the authors' explanation, and describe one experiment that would separate "false signal" from "fewer hard negatives".
4. slime grades SWE patches in a fresh sandbox from the same image. List two hacks this design prevents and one it does not, and say which of MAI's anti-hacking measures covers the remaining one.
5. Compare DeepSWE's masking of timed-out trajectories with rStar2's finding that overlong filtering raised the overlong ratio. Under what conditions does masking remove a signal the policy needs?
6. Rocket drops failed rollouts and states that this adds bias to the GRPO baseline. Explain the bias concretely for a group in which two of eight SWE rollouts fail because they hit the same flaky test.
7. DeepSWE runs 512 containers per iteration. Derive this number from its batch settings, and explain how the container count scales when the group size doubles and when the agent timeout doubles.

---

## Connections

- **ch-08** — agentic rollouts, tools, sandboxes and graders: the general frame this chapter instantiates.
- **ch-09, ch-11** — reference answers and grader design; §2 and §3 apply them.
- **ch-03, ch-04** — staleness; SWE durations make it largest here.
- **ch-15** — the other long-horizon agentic domain; compare failure and timeout handling.
- **ch-17** — mixing math, code and SWE with other domains.
- **ch-18** — algorithm choices per domain.
- **ch-20** — fault tolerance; §4.4's failure policies at the job level.
- **llm-training ch-44** (process supervision and verifiable rewards) — verifier mechanics.
- **llm-training ch-45b** (multi-turn agentic RL) — masking and credit assignment for SWE agents.
- **llm-training ch-40** (group baselines) — leave-one-out and std-free advantages used by DeepSWE.
- **llm-training ch-44a** (length in RL) — length schedules for math and code.

---

## Sources

- [[verifier-pitfalls]] — rule-verifier recall, hybrid grading, generative verifier hacking.
- [[math-verify]] — extraction / normalization / comparison steps and recommended reward settings.
- [[open-instruct-rlvr-pipeline]] — open-instruct@f020e9a math and code verifiers.
- [[olmo-3-rl-pipeline]] — code data processing and verifier settings in Olmo 3.
- [[magistral]] — code grader limits; proportional vs binary code reward; test repair.
- [[dapo]], [[big-math]] — canonical integer answers; MCQ contamination in math pools.
- [[mai-thinking-1]] — STEM reference-answer pipeline; competitive code set; SWE funnel and anti-hacking; Rocket v1 retry behaviour.
- [[mai-rl-infrastructure]] — per-domain rollout durations; drop-failed policy and its stated bias.
- [[minimax-m1]] — math, code and SWE data and rewards; CISPO across domains.
- [[rstar2-agent]] — tool-integrated math: RoC, execution service scale, KV-aware scheduling, overlong-filtering result.
- [[demystifying-agentic-rl]] — real vs stitched tool trajectories for cold start.
- [[slime-agentic-and-mismatch-examples]] — slime@79aa41a ReTool sandbox and reward; coding-agent grader isolation and failure handling.
- [[areal-agent-pipelines]] — AReaL@bcef492 SWE timeout handling and Daytona code grading.
- [[swe-smith]] — synthetic bug strategies, environment sharing, cost.
- [[r2e-gym]] — SWE-Gen environment construction; weak separability of generated tests.
- [[deepswe]] — GRPO++, compact filtering, Kubernetes fleet, reward and time limits.
- [[swe-rl]] — execution-free similarity reward for SWE.
- [[deepseek-v3-2]], [[deepseek-v3.1]] — code-agent environment criteria and task counts.
- [[kimi-k2]] — 10,000-instance sandbox.
- [[glm-4-5]] — SWE reward and asynchronous infrastructure.
