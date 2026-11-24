<!-- chapter: ch-09
     track: rl-pipeline
     kind: content
     title: Designing the Task: Prompts, Reference Answers, and Verifiability
     deps: [ch-08]
     sources: [[rl-design-guide]], [[numina-math]], [[big-math]], [[deepmath-103k]], [[reasoning-gym]], [[swe-smith]], [[r2e-gym]], [[olmo-3-rl-pipeline]], [[verifier-pitfalls]], [[mai-thinking-1]], [[dapo]], [[kimi-k1-5-recipe]], [[general-reasoner-verifier]], [[polaris]], [[skywork-or1]], [[tau2-bench]], [[agentscaler]], [[open-instruct-rlvr-pipeline]]
     figures: figures/task-construction.html
-->

# Chapter 9 — Designing the Task: Prompts, Reference Answers, and Verifiability

> **Core insight.** An RL task is a prompt plus a way to decide, by program, whether a response is right, plus metadata (domain, source, difficulty). Its usefulness depends on three properties of the item, not on where it came from: the answer is uniquely verifiable, the question is open-ended so the answer cannot be guessed, and the answer has a closed form a program can check ([[big-math]] §1). Every task pipeline in the sources is a sequence of filters that enforce those properties and each filter exists because of a measured or stated failure: 117,018 multiple-choice matches in a 643,374-problem pool ([[big-math]] Table 2); near-duplicates of 90% of AIME24 and AMC23 in a 2,869K-question raw web pool ([[deepmath-103k]] §3); rule-verifier recall falling as the policy gets stronger ([[verifier-pitfalls]] Table 5). Reference answers come from four sources — cleaned existing labels, agreement among independent strong-model solutions, execution-verified tests, and procedural generators — and each needs its own check that the reference is correct. The last step is to measure difficulty as the pass rate of a reference policy, so the RL run can select prompts by pass rate later (ch-12).
>
> **Guideline.** When assembling a math or science RL set from existing corpora, remove or rewrite multiple-choice, yes/no, true/false, multi-part and proof items, because guessable or uncheckable items give reward without reasoning; Big-Math found 117,018 multiple-choice and 40,509 proof matches in its pool (Table 2), and MAI-Thinking-1 rewrites multiple-choice and proof questions to open-ended form three times and drops any item that does not reach consensus ([[mai-thinking-1]] Sec. 3.2.1). When a reference answer comes from a web or forum source, keep the item only if independent solutions agree on the final answer, because agreement is the cheapest signal of a wrong or unverifiable reference; DeepMath keeps a problem only if three R1 solutions and the source solution give identical answers ([[deepmath-103k]] Stage 4). When the pool is mined from the web, decontaminate with embedding retrieval plus an LLM paraphrase judge, not n-grams alone, because exact matching misses reworded competition problems ([[numina-math]] §3.4; [[deepmath-103k]] §3). When a program can generate the task, use a generator with a difficulty parameter and advance difficulty on policy accuracy, because Reasoning Gym's curriculum (advance after 70% over 20 steps) beat fixed difficulty at every reported level, e.g. Spell Backwards length 4 from 30.00 to 70.67 ([[reasoning-gym]] Table 5). Store each item's pass rate from N rollouts of a reference policy at build time ([[big-math]] §3.1: 64 rollouts of Llama-3.1-8B).

---

## Why this chapter matters

The grader can only be as correct as the reference it compares against, and the optimizer will exploit any item where reward is reachable without the intended behavior. Three examples from the sources:

- A multiple-choice item rewards a guess with probability 1/k. Kimi k1.5 removes a prompt if the model, without chain of thought, guesses the correct answer within N = 8 attempts, stating that this "can remove the majority easy-to-hack prompts" ([[kimi-k1-5-recipe]] §2.1; no numbers given).
- A wrong reference answer gives a correct response reward 0 and a matching wrong response reward 1. Big-Math's solvability filter exists to remove such items ([[big-math]] §2.2.2), and MAI-Thinking-1 runs a blind-grading stage specifically to "guard against faulty ground truths" (Sec. 3.2.1).
- A reference in an unmatched format makes a rule verifier reject correct responses. DAPO rewrote every answer in its prompt set as an integer "because varied answer formats made rule-based checking error-prone" ([[dapo]] §3.5).

This chapter is the first of four in the RL-design phase. It follows [[rl-design-guide]] §1 as its outline and cites the underlying cards for every number. ch-10 covers rollout settings, ch-11 graders, ch-12 reward composition and prompt selection.

---

## §1 What makes an item usable for RL

Big-Math defines three desiderata ([[big-math]] §1):

1. **Uniquely verifiable.** There is one correct answer (up to equivalence), so a checker can decide correctness.
2. **Open-ended.** The question cannot be answered by choosing among offered options or guessing yes/no.
3. **Closed-form.** The answer is a value, expression, set or other object that a program can compare.

The consequences for item types:

| Item type | Problem | Treatment in the sources |
|---|---|---|
| Multiple choice | guessable | removed (Big-Math, Kimi k1.5) or rewritten open-ended (Big-Math-Reformulated 47,010 items; MAI 3× + consensus) |
| Yes/no, true/false | guessable | removed ([[big-math]] §2.2.2; [[kimi-k1-5-recipe]] §2.1) |
| Proof | not checkable without a strong judge | removed (Big-Math, Kimi) or rewritten (MAI: proofs "are hard to verify directly without relying on a stronger AI judge") |
| Multi-part | one score cannot express partial correctness; ambiguous extraction | removed ([[big-math]] §2.2.2) |
| Answer contained in the question | trivially solvable | dropped by MAI's answer-leakage stage ([[mai-thinking-1]] Sec. 3.2.1) |
| Open forum discussion with no checkable answer | unverifiable | dropped by MAI's verifiability classifier |

MAI-Thinking-1 gives the reason for rewriting multiple-choice items directly: they "can often be solved by guessing, providing an unreliable reward signal" (Sec. 3.2.1). Fig. 16 of the report shows STEM Mix by *original* problem format: open-ended 56.1%, proof 33.3%, MCQ 10.6%. Its caption states that multiple-choice and proof problems "are converted to open-ended form during ingestion (and dropped where conversion is infeasible)", and that MAI retains "a small fraction of multiple-choice problems so the model remains familiar with the format" ([[mai-thinking-1]] Fig. 16 caption, read from the PDF). Removal is therefore a default, not an absolute rule; the retained fraction is not stated.

---

## §2 Sourcing prompts

**Existing datasets.** NuminaMath-CoT has 859,608 rows across sources: cn_k12 276,591; synthetic_math 167,895; orca_math 153,334; olympiads 150,581; synthetic_amc 62,111; aops_forum 30,201; math 7,478; gsm8k 7,345; amc_aime 4,072 ([[numina-math]] Technical Details). Big-Math starts from NuminaMath, HARP, Omni-MATH, MATH, GSM8K and others, 643,374 problems before filtering, 463,426 after source-specific cleaning, 251,122 at the end ([[big-math]] Table 1).

**Web and forums.** DeepMath-103K collects Math StackExchange subsets of MMIQC and WebInstructSub plus NuminaMath-CoT, 2,869K questions after basic filtering, ending with about 95K hard problems plus 8K level 3–5 problems from SimpleRL ([[deepmath-103k]] Summary). General-Reasoner re-crawls the source pages of about 5M WebInstruct instructions, drops items without clearly human-written answers, extracts about 1M questions with verifiable short answers using Gemini-1.5-Pro, and ends with about 230K after difficulty filtering ([[general-reasoner-verifier]] §3.1).

**Documents (textbooks, academic PDFs).** MAI-Thinking-1's STEM pipeline is built of independent asynchronous stages grouped into four phases ([[mai-thinking-1]] Sec. 3.2.1, Fig. 17, read from the PDF):
1. *Hierarchical parsing*: OCR by a vision-language model or OCR service; chunking; discarding pages without STEM content; boilerplate removal; repairing cross-references, numbering and page-split artifacts; an LLM marks question and answer spans.
2. *QA pairing*: for exercises whose answers sit elsewhere (end of chapter, appendix), candidate answers are retrieved by structural and semantic similarity and an LLM selects and validates the pairing, over multiple rounds.
3. *Curation*: LLM classifiers for verifiability, question type, topic taxonomy, PII and answer leakage; conversion of MCQ and proofs to open-ended (3× plus consensus); cleanup of non-mathematical text and external references.
4. *Scoring*: each problem solved k times by each of four model tiers (rated by AIME 2025 performance), pass rates bucketed into difficulty brackets, and blind grading to drop suspect ground truths (§4 below).

"Stages that are sensitive to noise or hallucinations are run multiple times with consensus voting." The result is STEM Mix with more than 5M samples, the hardest portion more than 550k (q, a) pairs.

**Targeted and vendor sources.** For competitive programming, unstructured documents rarely include comprehensive tests, so MAI used targeted sources and vendor data: 160k problems with verified reference solutions, runtime and memory limits, 17 languages ([[mai-thinking-1]] Sec. 3.2.2).

**A multi-domain mixture.** Olmo 3's Dolci-Think-RL mixes domains each with its own verifier (prompts in Think RL / Instruct RL): precise IF 30,186 / 38,000; math sources totalling 30,186 / 64,000 (Open-Reasoner-Zero, DAPO-Math, AceReason-Math, Polaris, KlearReasoner-MathSub, OMEGA); code 23,110 / 20,000; chat 21,387 / 49,950; totals 104,869 / 171,950 ([[olmo-3-rl-pipeline]] Table 20; the math, code and chat subtotals are my sums of the card's per-dataset rows). The report describes the mix as "roughly equal amounts of data … for each domain (with slightly more focus on math and instruction following)".

In open-instruct, the domain is part of each row: a row carries `messages`, `ground_truth`, and a `dataset` field that names the verifier (`open_instruct/dataset_transformation.py` L1012–1014 at f020e9a: `GROUND_TRUTHS_KEY = "ground_truth"`, `VERIFIER_SOURCE_KEY = "dataset"`). The reward function dispatches on that field (ch-11). verl does the same with `data_source`, mapping, for example, `"math_dapo"` and any name starting with `"aime"` to the `math_dapo` scorer and the six `numina_*` sources to `prime_math` (`verl/utils/reward_score/__init__.py` L50–83 at 8357f1e). The verifier choice is fixed when the task is created.

---

## §3 Creating reference answers

Four methods appear in the sources. Each has a typical failure and a corresponding check.

### §3.1 Existing labels, cleaned

NuminaMath's per-source processing shows what "existing label" means in practice ([[numina-math]] §3.1):
- MATH and GSM8K reference solutions reformatted to CoT by GPT-4.
- Orca-Math: answers located by regular expression and wrapped in `\boxed{}`; solutions not rewritten.
- AMC/AIME from the AoPS wiki: the first community solution containing `\boxed{}` is kept; about 6,500 problems, about 4,300 retained after embedding decontamination.
- AoPS forum: among replies with `\boxed{}` or ■, the one with the most LaTeX symbols becomes the reference and is rewritten by GPT-4o.
- Chinese K-12 exams: OCR, regex segmentation, GPT-4o translation and realignment.

These heuristics choose a reference without verifying it. Big-Math's answer filter then drops items with no extractable answer (no `\boxed{}`) ([[big-math]] §2.2.2). Failure mode: the extracted answer is wrong, partial, or in a form the grader cannot match.

### §3.2 Strong-model generation with agreement

- **DeepMath-103K**: GPT-4o rewrites problems toward one numerical or symbolic answer; three R1 solutions are generated; the problem is kept only if all extracted final answers, including the source solution's when present, are identical ([[deepmath-103k]] Stage 4). The paper reports a total cost of $138,000 in GPT-4o API fees and 127,000 H20 GPU hours for the dataset (§3).
- **NuminaMath-TIR**: tool-integrated solutions sampled from GPT-4o and kept by answer match against the reference ([[numina-math]] §3.3).
- **General-Reasoner**: Gemini-1.5-Pro extracts short answers from human-written solutions; Gemini-2.0-Flash generates 8 candidate solutions and items where all 8 fail or all 8 pass are dropped ([[general-reasoner-verifier]] §3.1). The 8-sample filter is a difficulty filter, not an agreement check on the reference.
- **Kimi k1.5 reward model data** and **MAI-Thinking-1 consensus stages** use multiple LLM passes and voting for the same purpose ([[mai-thinking-1]] Sec. 3.2.1).

Failure mode: models that share training data can agree on the same wrong answer. Agreement lowers the error rate; it does not certify the answer. No source in the library measures the residual error rate after agreement filtering.

### §3.3 Execution-verified tests

For code and repository tasks the reference is a test suite, and the reference solution's role is to certify the tests.

- **Kimi k1.5 test generation** ([[kimi-k1-5-recipe]] §2.3.5): generate 50 test cases per problem; run 10 ground-truth submissions; a test is valid if at least 7 of 10 submissions agree on it; a problem is kept if at least 9 of 10 submissions pass all selected tests. Of 1,000 problems, about 614 needed no special judge, 463 generators produced at least 40 valid tests, and 323 problems were kept.
- **Olmo 3 code** ([[olmo-3-rl-pipeline]] §4.4.2): problem rewriting → solution generation → test generation; keep solutions that pass more than 80% of tests and drop the failing tests.
- **MAI competitive coding**: reference solutions verified to pass all associated tests, with runtime and memory constraints per problem ([[mai-thinking-1]] Sec. 3.2.2).
- **SWE tasks** use fail-to-pass (F2P) tests: tests that fail on the pre-fix code and pass after the fix, plus pass-to-pass (P2P) tests that must keep passing. MAI's SWE funnel: 102M public PRs → 4.87M with linked issues → 2.08M built (42.8%) → 745,452 with an F2P signal (15.3%) → 265,617 verified (5.5%) across 94,044 repositories ([[mai-thinking-1]] p. 42–43 via card). R2E-Gym builds environments from commits, uses existing tests as F2P where available and generates F2P tests from an Agentless-like method with the ground-truth patch as context where not, and back-translates the commit into an issue text without revealing the solution ([[r2e-gym]] §2, App. A).

Failure modes: tests that do not distinguish correct from incorrect patches, and tests that fail for unrelated reasons. R2E-Gym measured that for the majority of problems fewer than 20% of generated tests separate correct from incorrect patches (§4.2, Fig. 5), and it names "toxic tests" as a failure type. That measurement is about verifiers for test-time selection, but the same tests would be the RL reward.

### §3.4 Procedural generators

When a program can produce both the problem and its answer, the reference problem disappears.

- **Reasoning Gym** provides 102 datasets in 10 categories with algorithmic verifiers (principle P1) and difficulty parameters (P3), plus structural and stylistic parameters ([[reasoning-gym]] §2, Table 6).
- **τ²-bench** composes tasks from atomic subtasks, each with initialization, solution and assertion functions; 15 subtask groups over 3 intents give 2,285 compositions, of which 114 were subsampled ([[tau2-bench]] §3.2, Table 3). The number of actions becomes a controlled difficulty variable, and pass^1 falls toward zero beyond 7 required actions (Fig. 5).
- **AgentScaler** groups over 30,000 APIs by community detection into more than 1,000 domains, generates a database schema and Python tool implementations per domain, samples tool sequences as walks on a tool-dependency graph from a sampled initial state, and accepts a trajectory only if the final database state matches the gold state ([[agentscaler]] §2, §3.1).
- **MAI-Thinking-1's tool-use environments** are generated from plain-English descriptions in three phases: environment bootstrapping (tool descriptions, function implementations, seeded databases), task creation (sampling likely tool-call trajectories, creating entities, formulating requests), and verification and refinement (executing the generated actions, removing overly similar tasks), with critique-and-refine loops at each phase ([[mai-thinking-1]] Sec. 3.3.2).

Failure modes: generator bugs give wrong references for a whole family at once, and generated items cover only the distribution the generator encodes. Reasoning Gym reports uneven transfer across categories: RG-Algorithmic training raised Algebra from 23.83 to 52.89, while RG-Algebra training lowered ARC from 6.49 to 4.18 (Table 2); all results are from one 3B model.

| Method | Example | Check used | Remaining risk |
|---|---|---|---|
| Existing labels | NuminaMath, Big-Math | answer filter, solvability filter | wrong or partial labels |
| Model agreement | DeepMath (3 R1 + source) | identical final answers | correlated errors |
| Execution tests | Kimi k1.5, Olmo 3, SWE F2P | reference passes; test validity votes | low-distinguishing or flaky tests |
| Procedural | Reasoning Gym, τ², AgentScaler | answer computed by program | generator bugs; narrow coverage |

> Interactive: [figures/task-construction.html](figures/task-construction.html) — filtering funnels with sourced counts (Big-Math, Big-Math-Reformulated, DeepMath, Kimi k1.5 tests, SWE-smith, MAI SWE, General-Reasoner), and a clickable decision tree for choosing a reference-answer method with the checks each path needs.

---

## §4 Checking that answers are correct and uniquely verifiable

**Solvability filter.** Big-Math keeps an item only if Llama-3.1-8B (64 rollouts) or Llama-3.1-405B (5–8 rollouts) produces the reference answer at least once; HARP, Omni-MATH, MATH and GSM8K were exempt ([[big-math]] §2.2.2). The filter removes items whose reference is probably wrong and items whose format cannot be matched. It also removes correct items that are too hard for those models. The exemption of curated sources is how Big-Math limits that loss.

**Blind grading.** MAI-Thinking-1 resolves the same ambiguity differently. For items the strongest solver tier solves at a low pass rate, it shows a judge that tier's consensus answer and the ground truth "in randomized order". If the judge prefers the consensus answer, the item is dropped "as having a suspect ground truth"; if it prefers the ground truth, the item is kept "as a genuinely hard problem" ([[mai-thinking-1]] Sec. 3.2.1, read from the PDF). Randomized order addresses the judge's position bias (ch-11).

**Tuning the filters.** Big-Math's model-based filters were tuned in human-in-the-loop rounds, each inspecting 100 positive and 100 negative examples, to at least 90% precision and recall; the multiple-choice filter took 5–8 rounds. Final per-filter precision and recall are not reported ([[big-math]] §2.2).

**Rubric validation for non-math tasks.** For instruction-following tasks with rubrics, MAI validates each rubric "for self-containment, unambiguity, and alignment with their stated constraints" and screens prompts against safety policies ([[mai-thinking-1]] Sec. 3.4.2). A rubric item that the judge cannot decide consistently is an unverifiable item.

**The grader's own recall is part of verifiability.** An item is only verifiable relative to a grader. [[verifier-pitfalls]] measured rule-verifier recall on responses from many models: averaged 86%, and on Skywork-OR1 responses 0.78 / 0.86 / 0.88 for the VERL / Qwen-Math / HF Math-Verify verifiers; on WebInstruct-Verified (general science), HF recall was 47% (Table 5, §4.3). Named failure cases are equivalent forms in context (0.5π vs 90°) and unit conversions (3 hours vs 180 minutes). An item whose answer has many equivalent forms is less verifiable under a rule grader than one with a single canonical form. §5 covers what the task builder can do about that; ch-11 covers the grader side.

---

## §5 Normalizing answer formats

The task builder controls the reference's format; the grader has to match the response's. The simplest way to raise grader recall is to choose references that leave little room for equivalent forms.

**DAPO's integer rewrite.** Every answer in DAPO-Math-17K is an integer. For an answer of the form (a + √b)/c, the question is reworded so the expected answer is a + b + c ([[dapo]] §3.5). The grader can then compare strings.

**What that buys at the grader.** verl's `math_dapo` scorer, used for `data_source` values such as `math_dapo` and `aime*`, keeps only the last 300 characters, tries a Minerva-style `Answer:` extraction, falls back to the last `\boxed{}` in the last 100 characters, and compares strings exactly:

```python
# verl@8357f1e verl/utils/reward_score/math_dapo.py L209-220 (is_correct_strict_box body), L270-275
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred
...
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution (Minerva first, then `\boxed{}` fallback unless strict_box_verify)
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
```

Exact string comparison is safe for integer references and unsafe for references like `\frac{1}{2}` that a response may write as `0.5`. The format of the reference determines which grader is safe.

**Normalizing the gold side for symbolic graders.** Math-Verify's documentation advises passing the gold answer first and storing gold answers "already in the preferred canonical format"; its `verify(gold, pred)` is asymmetric by design (a gold inequality accepts a predicted interval but not the reverse) ([[math-verify]] Guideline). open-instruct's `MathVerifier` tries `\boxed{}`, then Minerva format, then the last LaTeX span, then the whole output, and accepts if any candidate is equivalent to the label (`ground_truth_utils.py` L245–300 at f020e9a). Multiple extraction paths raise recall and also raise the chance of accepting an answer the response did not commit to; ch-11 discusses that trade-off.

**Normalization by rewriting.** DeepMath asks GPT-4o to rewrite problems "toward one numerical or symbolic answer" ([[deepmath-103k]] Stage 4); MAI's cleanup stage removes "extraneous, non-mathematical text and external references not relevant to the question" ([[mai-thinking-1]] Sec. 3.2.1). Rewriting can change the problem; MAI runs its conversion stage three times and drops items without consensus for that reason.

---

## §6 Removing ambiguous or unverifiable items

The filters in the sources, collected:

| Filter | Removes | Source |
|---|---|---|
| Answer extractability | items without a boxed or extractable answer | [[big-math]] §2.2.2 |
| Item-type classifiers (regex + Llama-3.1-70B) | MC, multi-part, proof, yes/no/TF | [[big-math]] §2.2.2 |
| Verifiability classifier | items without a checkable answer | [[mai-thinking-1]] Sec. 3.2.1 |
| Answer leakage | answer contained in the question | [[mai-thinking-1]] Sec. 3.2.1 |
| Human-written-answer check | items whose answers are not clearly human-written | [[general-reasoner-verifier]] §3.1 |
| Figure dependence | HARP Asymptote-figure problems (625) | [[big-math]] §2.2.1 |
| Easy-to-hack | prompts guessed within 8 attempts without CoT | [[kimi-k1-5-recipe]] §2.1 |
| Rubric validation | ambiguous or non-self-contained rubric items | [[mai-thinking-1]] Sec. 3.4.2 |
| Evaluability check | SWE tasks missing image/workdir or not evaluable | slime `coding_agent_rl/generate.py` L263–267 |

The last row is an online check: slime's coding-agent rollout aborts a task before spending a sandbox if the metadata lacks an image or working directory, or if `swe.evaluability_check` returns a reason. Pushing such checks into the task build saves rollout capacity.

---

## §7 Contamination checks

**Measured contamination of a raw pool.** Before decontamination, DeepMath's 2,869K-question pool contained near-duplicates of 90% of AIME24 and AMC23, 76.6% of MATH500 and 33.6% of OlympiadBench ([[deepmath-103k]] §3, Stage 2). These are contamination rates; the paper does not measure how much they would inflate RL evaluation.

**Methods.**
- DeepMath: top-5 embedding retrieval with paraphrase-multilingual-MiniLM-L12-v2, then Llama-3.3-70B-Instruct judges duplicate or paraphrase; any match is discarded; 14 benchmarks ([[deepmath-103k]] Stage 2).
- NuminaMath: 10-gram exact match plus Mistral-embedding distance < 0.15, motivated by MATH test problems that are reworded AIME/AMC problems ([[numina-math]] §3.4).
- MAI-Thinking-1: exact deduplication by SHA-256 question hashes, then character n-gram shingling with MinHash LSH, then embedding cosine similarity, each with a threshold "tuned … to retain as much data as possible while ensuring strict exclusion of any benchmark leakage", applied against reported benchmarks and internal evaluations ([[mai-thinking-1]] Sec. 3.2.3, read from the PDF; thresholds not given).
- Big-Math removed 4,229 items in decontamination ([[big-math]] §2.2.2).

**Why it matters for RL specifically.** RL reinforces behaviors that reach reward. A contaminated item lets the policy reach reward by recalling an answer, and the evaluation on the same benchmark then measures recall. The spurious-reward results (ch-12) show a related effect: on Qwen2.5-Math-7B, random rewards raised MATH-500 by 21.4 points, but on AIME2025, written after the tested models' knowledge cutoffs, ground-truth reward had a clear advantage and other rewards gave −0.4 to +4.5 ([[spurious-rewards-rlvr]] §2.2, App. D). Evaluation on post-cutoff problems is the check that decontamination cannot provide.

---

## §8 Difficulty measurement by policy pass rate

Difficulty is defined relative to a model: the fraction of N sampled responses that the grader accepts.

| Source | Reference policy | N | Use |
|---|---|---|---|
| [[big-math]] §3.1 | Llama-3.1-8B | 64 | quintiles: 28.64%, 12.16%, 10.26%, 12.44%, 36.50% from easiest to hardest; hardest bin is success < 20% |
| [[polaris]] | the policy to be trained | 8 at T = 0.6 | remove 8/8-solved; shape a mirrored-J distribution |
| [[olmo-3-rl-pipeline]] §4.4.2 | starting checkpoint (7B only) | 8 | remove pass rate > 62.5%; 32B relies on active sampling instead |
| [[kimi-k1-5-recipe]] §2.1 | SFT model | 10 at "relatively high" temperature | pass rate as difficulty for sampling |
| [[skywork-or1]] §3.1 | base model | not stated in card | drop correctness 0 or 1 offline |
| [[general-reasoner-verifier]] §3.1 | Gemini-2.0-Flash | 8 | drop all-pass and all-fail |
| [[mai-thinking-1]] Sec. 3.2.1 | four solver tiers, rated by AIME 2025 | k per tier | difficulty brackets; trigger blind grading |
| [[deepmath-103k]] Stage 3 | GPT-4o rating (not pass rate) | 6 ratings | keep AoPS-style level ≥ 5 |

Two properties of pass-rate difficulty matter for the run.

**It drifts.** The policy improves, so a prompt's pass rate under the training policy changes during the run. Build-time pass rates select the initial window; online filters (DAPO dynamic sampling, MAI's early-exit, Skywork's per-stage re-filtering, ScaleRL's No-Positive-Resampling) keep the window current. ch-12 covers them.

**It is policy-specific.** A pass rate measured with one model does not transfer exactly to another. MAI uses four tiers so that brackets reflect a range of abilities; Big-Math uses an 8B model and reports that 36.50% of its final set is below 20% success, which for a stronger policy would be a different number.

**Difficulty labels without pass rates are weaker evidence.** SWE-smith trained a 7B student on tasks at difficulty levels 2 / 4 / 6 / 8 and got 12.4 / 10.8 / 13.6 / 12.2% on SWE-bench Verified with no clear trend ([[swe-smith]] §4.1). The difficulty there was an estimate, the training was SFT on teacher trajectories, and the reported spread is about ±1.5–1.8, so this does not show that difficulty is irrelevant for RL; it shows that the estimate did not predict SFT value.

---

## §9 Agentic and SWE tasks

**Synthesizing bugs inside one environment per repository.** SWE-smith installs a repository once, generates candidate bugs with several strategies, runs the test suite with a 2-minute cap, and keeps a candidate only if at least one passing test starts failing. The reference answer is the reverse of the bug patch; the failing tests are the grader ([[swe-smith]] Summary):

| Strategy | Yield | Instances | Cost per candidate |
|---|---|---|---|
| LM Modify | 56.0% | 17,887 | 0.38¢ |
| LM Rewrite | 35.0% | 4,173 | 3.93¢ |
| Procedural (13 AST transforms) | 40.2% | 15,641 | 0.00¢ |
| Combine | 96.9% | 10,092 | 0.00¢ |
| PR Mirror | 33.8% | 2,344 | 5.53¢ |
| Total | 50.1% | 50,137 | 2.32¢ |

Total cost was about $1,360 and 295 GB of environments, against an estimated 50–150 TB for a SWE-bench-style build of the same count (§2.2, Tables 1–2). In an SFT ablation at 1,000 instances each, PR-mirror, procedural and LM-rewrite bugs gave 9.2 / 8.6 / 8.8% against 5.7% for LM-modify (Table 4); the paper reports a spread of about ±1.5–1.8, so the first three are within noise of each other. These are SFT results; using them to choose RL tasks is an inference.

**Issue text.** SWE-smith's issue-text ablation on 600 PR-mirror instances gave 7.8% (original), 7.7% (LM-generated), 7.3% (F2P test as issue), 6.4% (fixed template), and test-based issues led the model to skip writing reproduction scripts (Table 5). R2E-Gym's back-translation prompt tells the LLM to describe the failure and expected behavior without revealing the solution (App. A). An issue text that leaks the fix is the agentic version of answer leakage.

**Composed tool-use tasks.** τ²-bench's atomic subtasks and AgentScaler's tool-graph walks (§3.4) make horizon length a parameter. MAI adds tasks that offer tools but do not require them "to mitigate over-eager tool-calling behavior", and environment-specific personas to diversify tasks ([[mai-thinking-1]] Sec. 3.3.2). A task set where every task needs tools teaches the policy that tools are always needed.

**Environment-level anti-hacking belongs to the task.** MAI's SWE tasks scrub git history past the base commit, reset test files, and control network access ([[mai-thinking-1]] card table). An environment where the fix can be read from `git log` is a task with its answer leaked.

---

## §10 Applied example: tasks for a sales agent

For a sales voice agent on the learner's own endpoints, the four reference methods map as follows. Booking and quoting tasks can be procedurally generated from a seeded catalog and calendar (τ²-style atomic subtasks: look up product, quote price, check availability, book), with references computed by program and graded by final state. Factual product questions can use existing catalog data as labels, with answers normalized to canonical forms (price as an integer number of cents, dates in ISO format) so a rule grader is safe. Objection-handling turns have no unique answer; they are not RL-verifiable items in the sense of §1 and need a rubric judge (ch-11) and a gated combination (ch-12). Tasks should include customer requests that need no tool call, following MAI's over-eager-tool mitigation.

---

## Recipe

1. **Define the item schema**: prompt (messages), reference (answer, tests, rubric, or gold state), verifier name, source, domain, and a slot for pass rate.
2. **Collect** from existing datasets, web/forum pools, documents, vendors or generators. Record the source of every item.
3. **Type-filter**: remove or rewrite MC, yes/no, TF, proof and multi-part items. If rewriting, run the rewrite several times and keep only consensus outputs.
4. **Create or confirm the reference**: cleaned label; agreement of ≥ 3 independent strong-model solutions plus the source; execution-verified tests with validity votes; or a generator.
5. **Normalize the reference format** to the canonical form your grader matches (integers if the grader compares strings).
6. **Remove unverifiable and leaky items**: verifiability classifier, answer-leakage check, figure-dependent items, issue texts or environments that reveal the solution.
7. **Decontaminate** against every evaluation set with hash, MinHash and embedding retrieval plus an LLM paraphrase judge; keep post-cutoff evaluation sets for final checks.
8. **Run a solvability or blind-grading check** on items that strong models rarely solve; drop items where the strong-model consensus beats the reference in an order-randomized judgment.
9. **Measure pass rate** with N ≥ 8 rollouts of the starting policy (or several tiers); store it per item.
10. **Hand off to ch-12** for windowing by pass rate and online re-filtering.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Keeping multiple-choice items | reward rises without longer reasoning; answers are option letters | item-type classifier; Kimi's no-CoT guessing test |
| Trusting one model's reference answer | correct responses get reward 0 | agreement check; blind grading on low-pass items |
| Reference in a non-canonical format | grader recall low on some sources | per-source accuracy of grader on a hand-labeled sample |
| Solvability filter applied to curated hard sets | hard correct items removed | exempt curated sources (Big-Math) or use blind grading |
| n-gram-only decontamination | inflated benchmark scores | embedding + LLM judge recheck; post-cutoff evaluation |
| Tests that do not distinguish patches | reward for wrong patches | fraction of tests separating reference from mutated patches |
| Issue text or repository history revealing the fix | very fast solves on SWE | scrub history; inspect solved trajectories for reads of `git log` |
| Every agentic task requires a tool | tool calls on tasks that need none | include no-tool tasks; measure unnecessary call rate |
| Pass rate measured once with a different model | windows wrong for the training policy | re-measure with the starting policy; online filters |

---

## Check your understanding

1. Big-Math's solvability filter and MAI's blind grading address the same ambiguity — an item that strong models rarely solve. Explain what each does when the reference is correct but the item is very hard, and why MAI randomizes the order in which the judge sees the two answers.
2. DAPO rewrote all answers as integers. Why does that change which grader is safe to use, and what does it cost in terms of the problem distribution?
3. Three R1 solutions plus the source solution agreeing does not prove the answer correct. Under what condition does agreement fail, and what additional check from the sources would catch that case?
4. Why is a contaminated item more damaging in RL than in SFT, considering what the RL objective reinforces? What evaluation design detects the damage that decontamination missed?
5. Pass rate is defined relative to a model. Explain why a build-time pass rate is necessary but not sufficient for prompt selection during a long run.
6. SWE-smith's difficulty-level ablation showed no trend. Give two reasons from the setup why this result does not transfer directly to RL task selection.
7. For a procedurally generated tool-use task, what replaces the "reference answer", and what kind of error in the generator would make an entire task family unverifiable?

---

## Connections

- ch-08 (environments and graders that consume these tasks), ch-10 (rollout settings that depend on task length and difficulty), ch-11 (grader recall and false positives that decide which reference formats are safe), ch-12 (prompt selection by pass-rate window), ch-13 (diversity axes of the task set), ch-14 (IF constraint tasks), ch-16 (math, code and SWE task pipelines end to end).
- [llm-training ch-16](../../llm-training/ch-16/read.md) for the RL prompt distribution, difficulty filtering and prompt reuse; [llm-training ch-44](../../llm-training/ch-44/read.md) for verifiable rewards; [llm-training ch-44b](../../llm-training/ch-44b/read.md) for multi-domain tasks with non-verifiable components.
- Sources: [[rl-design-guide]] (outline of this chapter, §1), [[big-math]], [[deepmath-103k]], [[numina-math]], [[reasoning-gym]], [[swe-smith]], [[r2e-gym]], [[olmo-3-rl-pipeline]], [[verifier-pitfalls]], [[mai-thinking-1]], [[dapo]], [[kimi-k1-5-recipe]], [[general-reasoner-verifier]], [[polaris]], [[skywork-or1]], [[tau2-bench]], [[agentscaler]], [[math-verify]], [[spurious-rewards-rlvr]], [[open-instruct-rlvr-pipeline]].

---

## Sources

- [[rl-design-guide]] — §1 task and reference-answer rules used as this chapter's outline.
- [[big-math]] — the three desiderata, filter pipeline and counts, solvability filter, MC reformulation, 64-rollout difficulty quintiles.
- [[deepmath-103k]] — raw-pool contamination rates, embedding + LLM-judge decontamination, 3-way R1 agreement, dataset cost.
- [[numina-math]] — per-source label extraction, TIR answer-match filter, n-gram + embedding decontamination, source counts.
- [[reasoning-gym]] — procedural generators, parametric difficulty, curriculum vs fixed difficulty, cross-domain transfer.
- [[swe-smith]] — bug-synthesis strategies, yields and costs, environment sharing, issue-text and difficulty ablations (SFT).
- [[r2e-gym]] — commit-based environments, F2P test generation, back-translated issues, test distinguishability.
- [[olmo-3-rl-pipeline]] — Dolci-Think-RL composition, per-domain processing, offline difficulty filter.
- [[verifier-pitfalls]] — rule-verifier recall by dataset and policy strength.
- [[mai-thinking-1]] — STEM (q, a) extraction pipeline, conversion with consensus, blind grading, competitive-coding and SWE funnels, tool-use environment generation, three-stage deduplication (Sec. 3.2–3.4, read from the PDF).
- [[dapo]] — integer answer rewrite.
- [[kimi-k1-5-recipe]] — excluded item types, easy-to-hack filter, test-case generation and validity voting.
- [[general-reasoner-verifier]] — WebInstruct re-crawl and short-answer extraction, 8-sample filter.
- [[polaris]], [[skywork-or1]] — build-time pass-rate filtering.
- [[tau2-bench]], [[agentscaler]] — compositional and graph-walk task generation with state-based references.
- [[math-verify]] — canonical gold format and asymmetric verification.
- [[spurious-rewards-rlvr]] — post-cutoff evaluation as a check.
- [[open-instruct-rlvr-pipeline]] — row schema with verifier tag.
