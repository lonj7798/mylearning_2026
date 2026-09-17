<!-- chapter: ch-35a
     track: sft
     kind: content
     title: Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters
     deps: [ch-35, ch-22]
     sources: [[open-thoughts]], [[openmathreasoning]], [[opencodereasoning]], [[qwen-3]], [[light-r1]], [[limo]], [[s1]], [[magistral]], [[magistral-recipe]], [[glm-4-5]], [[llama-4]], [[kimi-k1-5]], [[acereason-nemotron-1-1]], [[distillation-source-matters]], [[small-models-learnability-gap]], [[capacity-gap-law-distillation]], [[deepdistill]], [[naturalthoughts]], [[phi-4]], [[bespoke-stratos]], [[openr1-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[deepseek-v3]], [[llama-nemotron]], [[hermes-4]]
     figures: figures/difficulty-filter-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 35a — Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters

> **Core insight.** In the open distillation studies with controlled ablations, the question source and the teacher changed student scores by 13-17 points: 17.2 points between the best and worst code sources for a Qwen2.5-7B-Instruct student ([[open-thoughts]] Table 3), and 13.4 AIME2024 points between three teachers on one set of 1.89M queries for Qwen2.5-32B students ([[distillation-source-matters]] Table 1). Answer filtering did not beat keeping all answers for 7B-14B students: on the OpenThoughts math sets, random selection of 31,600 answers scored a higher math average than every answer filter at equal size (64.8 vs 61.4 for GPT verification, [[open-thoughts]] Table 7), and 151,251 solutions that fail all unit tests trained a better 14B student than 151,251 passing ones (LiveCodeBench 52.3 vs 47.0), which the authors trace to the failing solutions coming from harder questions ([[opencodereasoning]] Table 3). The higher-scoring model is not always the better teacher: QwQ-32B traces beat DeepSeek-R1 traces for 7B students (44.2 vs 41.6 average with a math question source, [[open-thoughts]] Table 8). Every difficulty label in these pipelines is a statistic of a named reference model, so a filter is reproducible only with that model, the number of samples, and the kept range.
>
> **Guideline.** When selecting prompts, record the reference model, samples per prompt, and kept range of correct samples, because published rules such as "drop if solved within 4 attempts", "keep 1-3 of 32", and "drop pass rate ≥ 0.75 of 8" keep different prompts ([[limo]] §3.1.1; [[llama-nemotron]] §5.1). When more distinct prompts can still be collected, add prompts before adding responses per prompt, because the AceReason-Nemotron 1.1 regression gives prompts a larger coefficient (4.831) than responses per prompt (2.635) ([[acereason-nemotron-1-1]] §4.4.2); when the pool is exhausted, sample more responses per prompt (16 in [[open-thoughts]] §4.4) and train to the epoch plateau (epochs 5-6 in [[acereason-nemotron-1-1]] Fig. 6). When choosing a teacher, train a small student on each candidate's traces instead of ranking teachers by benchmark score, because QwQ-32B, the lower-scoring model, was the better teacher for 7B students ([[open-thoughts]] Table 8). When the student has 3B parameters or fewer, mix short-CoT traces into long-CoT data, because Qwen2.5-3B scored 40.3 after long-CoT SFT, 43.4 after short-CoT SFT, and 45.9 after a 1:4 long-to-short mix ([[small-models-learnability-gap]] Tables 1, 3). When traces seed a cold start or an RL prompt set, verify answers strictly, because a wrong reference produces a wrong reward (Interpretation); otherwise, for bulk SFT data, run an equal-size, difficulty-matched ablation before adopting an answer filter.

## Why this chapter matters for a general-purpose model

Distillation SFT trains a student model on text written by a stronger model, the teacher. [[ch-35]] mapped where labs place teacher data: pretraining with teacher logits, mid-training, cold start, rejection-sampling SFT after RL, small-model distillation, and on-policy distillation. This chapter covers the decisions inside one distillation dataset: which prompts are sent, how the teacher is prompted and sampled, which outputs are kept, how the data is decontaminated, and which SFT settings the student receives.

Pipeline position: SFT and distillation SFT, after pretraining and mid-training and before preference optimization and RL. Three RL prompt rules (Kimi k1.5, Magistral, and LN-Ultra) are included because they use the same difficulty statistics; RL prompt curricula are the topic of [[ch-16]]. Data selection by quality and diversity scores is the topic of [[ch-22]].

These decisions affect generality in three measurable ways.
1. **Domain share.** Open distillation sets are dominated by verifiable math and code. In the Llama-Nemotron SFT data, math is 66.8% and code 30.6% of 33,011,757 samples, while chat is 0.12%, instruction following 0.17%, and safety 0.10% ([[llama-nemotron]] Table 2).
2. **Loss of ability outside the target.** After SFT stage 2 on about 3k mostly math questions, Light-R1-32B's GPQA score fell from 64.3 to 60.6 ([[light-r1]] Table 2). LN-Super-SFT scored 81.9 on IFEval in reasoning-on mode (83.0 off), against 92.1 for Llama-3.3-70B-Instruct, the model it is derived from, and the report states that "reasoning-focused SFT causes a noticeable drop in IFEval scores" ([[llama-nemotron]] Table 4). OpenThinker3-7B's HarmBench harmfulness rate rose from 14.5 to 55.5 against its Qwen2.5-7B-Instruct base; no safety data was used ([[open-thoughts]] App. L).
3. **Measurement gaps.** [[distillation-source-matters]] trains on a query set that is 41.8% general chat by count but reports only math and code results (Fig. 2; Tables 1-2).

## §1 Question sourcing

A **question source** is a pool of prompts, such as a forum archive or a competition collection, from which distillation prompts are drawn. Sources differ in how much they teach at equal size: across code sources, each used to build a 31,600-example set with DeepSeek-R1 answers for a Qwen2.5-7B-Instruct student, the average scores over all 8 development benchmarks of the best and worst code sources differ by 17.2 points (38.8 for StackExchange-CodeGolf vs 21.6, [[open-thoughts]] §4.1, Table 3).

The OpenThoughts sourcing ablation proceeds in three steps ([[open-thoughts]] §4):
1. Build one 31,600-example set per candidate source (27 code, 21 math, 14 science sources).
2. Fine-tune Qwen2.5-7B-Instruct on each set and average 8 development benchmarks.
3. Keep the best option and move to the next pipeline step: mixing, question filtering, answers per question, answer filtering, teacher.

Mixing the top 2 code sources averaged 41.3 and mixing the top 16 averaged 36.4 (Table 4); at most two sources per domain was best in all three domains (§4.2). The selected sources were OpenMath-2-Math (math), StackExchange CodeGolf and OpenCodeReasoning (code), and StackExchange Physics and OrganicChemistry-PDFs (science). Status: Result (single study), one student model, selection by an average over reasoning benchmarks.

**Forum-mined math.** OpenMathReasoning builds its pool from Art of Problem Solving (AoPS) forums, with Qwen2.5-32B-Instruct performing every step ([[openmathreasoning]] §2.1, Table 1):
1. Extract problems from first posts: 620K discussions give 580K problems.
2. Classify each problem as proof or not, multiple choice or not, binary (yes/no) or not, valid or not; remove multiple-choice, binary, and invalid problems (550K remain).
3. Convert proofs into answer-based questions and extract final answers for the rest.
4. Remove problems that an LLM judges close to benchmark items (540K remain).

The final pool has 260K converted proofs, 190K problems with an extracted answer, and 90K without one (Table 2). The "Middle School Math" forum was excluded as "too elementary and unhelpful for training in our preliminary experiments", with no numbers given. Llama-Nemotron reuses this pipeline but removes proof problems instead of converting them ([[llama-nemotron]] §3.1.1), so the same forum yields a different pool.

**Competitive-programming aggregation.** OpenCodeReasoning merges TACO, APPS, CodeContests, and the OpenR1 CodeForces set, removes exact duplicates, and keeps 28,904 questions from 10 platforms; CodeForces contributes 10,403 questions and 388,405 of the 736,712 samples ([[opencodereasoning]] §2.1, Table 1). Scaling from 25k to 736k samples did not plateau, and the largest gains came from adding "unique, varied, and difficult questions" (§2.4, Fig. 3). Status: Replicated for the direction that more distinct prompts help, by [[acereason-nemotron-1-1]] §4.4.2 in math and code at 7B.

**A lab reference point.** DeepSeek-R1's 804,745-sample SFT set has 395,285 math, 211,129 code, 10,124 STEM, 10,395 logic, and 177,812 general samples, with mean lengths from 1,419.8 tokens (general) to 7,435.7 tokens (code) ([[deepseek-r1-recipe]], v2 B.3.3 Table 5). General samples are 22.1% of the set (177,812 / 804,745, derived). All six DeepSeek-R1-Distill students were trained on this set (B.4.3).

Implication for a general-purpose model: selecting sources by an average of math, code, and science benchmarks optimizes for those domains. The OpenThoughts authors note that selecting by the overall average assumes cross-domain transfer (§6). A source ablation for a general student needs held-out chat, instruction-following, and safety suites inside the selection score.

## §2 Difficulty relative to a named reference model

A **difficulty label** is a statistic of a reference model's outputs on a prompt: its pass rate over n samples, whether it answers without reasoning, the length of its response, or whether two models agree. The problem they address is that easy prompts may teach less than hard ones, which LIMO measured: it trained Qwen2.5-32B-Instruct on three 500-problem sets of increasing difficulty (MATH levels 1-2, MATH levels 3-5, AIME), all with DeepSeek-R1 solutions, and reports that "modifying problem selection alone leads to 16% accuracy improvement on AIME2024, reaching 51.5%" ([[limo]] arXiv:2502.03387v3 §6.3.2, Fig. 4).

| Pipeline (stage) | Reference model | Statistic | Keep rule and resulting count | Source |
|---|---|---|---|---|
| Qwen3 cold start (SFT) | Qwen2.5-72B-Instruct | correct answer without chain of thought (CoT) | drop prompts answered correctly without CoT; also drop multi-part and free-text-generation queries | [[qwen-3]] §4.1 |
| s1K (distill-SFT) | Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct, graded by Claude 3.5 Sonnet | solved or not | drop if either model solves: 59,029 collected → 54,116 after API errors → 51,581 after format checks → 24,496 after the difficulty filter → 1,000 after domain-stratified sampling (384 of the 1,000 taken directly from sources judged high-quality) | [[s1]] §2.1-2.2 |
| LIMO (distill-SFT) | Qwen2.5-Math-7B-Instruct, then DeepSeek-R1-Distill-Qwen-32B | correct in 4 attempts; correct count in 32 | 0 of 4, then 1-3 of 32: 2,125 problems | [[limo]] §3.1.1 |
| Light-R1 (distill-SFT) | DeepScaleR-1.5B-Preview, then DeepSeek-R1 (§3.1.3; §1 names DeepSeek-R1-Distill-Qwen-32B for the second stage) | pass rate (n not printed) | from about 1000k seed questions, pass rate < α (α not printed): about 76k; then < α and not uniformly correct or wrong: about 3k | [[light-r1]] §1, §3.1.1, §3.1.3 |
| Phi-4-reasoning (distill-SFT) | plurality answer of a strong model as proxy truth; Phi-4 or GPT-4o as weaker models | agreement rate of weaker models with proxy truth; rubric LLM counts reasoning steps | keep seeds with "a meaningful gap" (threshold not printed) | [[phi-4]] arXiv:2504.21318v1 §2.1 |
| OpenMathReasoning, round 2 (distill-SFT) | Qwen2.5-Math-72B-Instruct with tools | pass rate over 32 | ≤ 0.3, Olympiad forums only, solutions ≥ 5000 tokens: 2.2M samples | [[openmathreasoning]] §5.1 |
| Magistral math (RL data) | Mistral Large 2, then a 24B RL-trained grader | 16 samples per pass | drop never-solved and high-success problems, and problems whose majority answer contradicts the reference: 699k → 501k after format filtering → 38k | [[magistral]] §4.1, Table 1 |
| GLM-4.5 (SFT) | not stated | response length | drop the bottom 50% of prompts | [[glm-4-5]] §3.1 |
| OpenThoughts math and science (distill-SFT) | GPT-4.1-mini | response length | keep the longest | [[open-thoughts]] §4.3 |
| AceReason-Nemotron 1.1 (distill-SFT) | DeepSeek-R1 | response length | randomly downsample prompts with responses around or below 2,000 tokens | [[acereason-nemotron-1-1]] §3.1.1 |
| Llama 4 Maverick (SFT) | Llama-model judges | "easy" tag | remove more than 50% of data tagged easy; Behemoth: 95% of SFT data pruned, "as opposed to 50% for smaller models" | [[llama-4]] post |
| Kimi k1.5 (RL prompts, general question answering) | a model without CoT | correct guesses in N = 8 | drop if guessed within 8 attempts | [[kimi-k1-5]] §2.1 |
| LN-Ultra (RL data) | LN-Super | pass rate over 8 | drop pass rate ≥ 0.75 | [[llama-nemotron]] §5.1 |

**Formula.** When a filter samples the reference model n times on a prompt and keeps the prompt if the number of correct samples k lies in a set S, the probability of keeping it is

P(keep | p) = Σ_{k ∈ S} C(n, k) · p^k · (1 − p)^(n − k)

where p is the reference model's per-sample probability of solving that prompt, n is the number of samples, k is the number of correct samples, S is the kept set of k values, and C(n, k) is the binomial coefficient. The formula assumes independent samples with one fixed p per prompt; the papers do not report p.

**Worked example.** LIMO stage 1 keeps S = {0} with n = 4, so P(keep) = (1 − p)^4. A problem the 7B model solves with p = 0.1 survives with probability 0.9^4 = 0.656; with p = 0.3, 0.7^4 = 0.240; with p = 0.5, 0.5^4 = 0.0625. LIMO stage 2 keeps S = {1, 2, 3} with n = 32: P(keep) is 0.275 at p = 0.01, 0.732 at p = 0.05, 0.566 at p = 0.1, and 0.092 at p = 0.2, with its maximum of 0.740 near p = 0.057. The LN-Ultra rule keeps k ≤ 5 of 8: a prompt LN-Super solves with p = 0.75 is still kept with probability 0.32, and with p = 0.9, 0.038. The Kimi guessability rule keeps k = 0 of 8: a prompt guessable with p = 0.1 survives with probability 0.9^8 = 0.430. Small n makes the kept set noisy near the threshold.

The companion figure [difficulty-filter-explorer.html](figures/difficulty-filter-explorer.html) draws P(keep | p) for these rules, compares two rules on one chart, and shows the published selection stages of OpenMathReasoning, s1K, LIMO, Magistral, and Light-R1 on a log scale.

**Evidence of effect.**
- s1 at 32B, AIME24: 1,000 random questions 36.7; 1,000 longest traces 33.3; s1K (quality, difficulty, diversity) 50.0; all 59K questions 53.3, with a 95% bootstrap interval for the 59K-minus-s1K difference of [−13.3%, 20.0%], which includes zero ([[s1]] Table 2). Training on 59K took 394 H100 GPU hours against 7 for s1K (§5.1).
- GLM-4.5: dropping the bottom 50% of prompts by response length gave "a 2%-4% improvement in math and science tasks, despite training with only half the data" ([[glm-4-5]] §3.1). Model size, benchmarks, and a table are not given.
- Light-R1-32B: stage 2 on about 3k harder examples raised AIME24 from 69.0 to 73.0 and AIME25 from 57.4 to 64.3 while GPQA fell from 64.3 to 60.6 ([[light-r1]] Table 2).
- NaturalThoughts, Llama-3.1-8B-Instruct student, GPQA-Diamond: at 10k examples, prompts where DeepSeek-R1 and Llama-3.3-70B disagree scored 39.7 against 37.6 for random; at 500k, 45.2 against 48.3 for random ([[naturalthoughts]] Table 1).

Status: Replicated that selected prompts beat random or easier prompts at 500 to 10k examples: s1K combines quality, difficulty, and diversity ([[s1]]), LIMO compares question difficulty levels ([[limo]]), and NaturalThoughts selects by model disagreement ([[naturalthoughts]]). Difficulty alone is not sufficient: in s1, the 1,000 longest traces scored 33.3 on AIME24, below 36.7 for random ([[s1]] Table 2). Open question whether the advantage persists at hundreds of thousands of examples; the only 500k comparison above reverses it for one criterion.

**Conditions and limits.**
- The kept set depends on the reference model. Magistral used a second, stronger grader because a single pass with Mistral Large 2 "would likely have caused it to discard many genuinely difficult problems by incorrectly classifying them as unsolvable" ([[magistral]] §4.1).
- s1 filters with Qwen2.5-32B-Instruct, which is also its student base; the authors note the filtering "may be optimized for our setup" ([[s1]] §2).
- A pass rate of 0 mixes hard prompts with broken ones. In Light-R1's RL prompt set, "over half" of pass-rate-0 prompts had unverifiable or incorrect reference answers ([[light-r1]] §4).
- Response length conflates difficulty with verbosity; [[ch-22]] covers length as a confound in selection.

Implication for a general-purpose model: pass-rate filters need a reference answer, so they do not apply to open-ended chat prompts. Phi-4-reasoning's plurality proxy answer and rubric-based step counting are the documented alternatives for domains without verifiable answers ([[phi-4]] §2.1).

## §3 What is sent to the teacher: system prompts, mode control, sampling

A **teacher request** consists of a system prompt, an optional mode flag, the user prompt, and sampling settings: temperature T, the nucleus threshold top-p (sampling only from the smallest token set whose probability sum reaches top-p), and a maximum number of new tokens. **Mode control** is an instruction that selects reasoning or direct answering; training on both modes lets one student keep both behaviors.

**Qwen3 flags.** Thinking-mode fusion trains the Stage-2 reasoning model on thinking data from rejection sampling with the Stage-2 model on Stage-1 queries, plus non-thinking data covering coding, mathematics, instruction following, multilingual tasks, creative writing, question answering, and role-play, checked with automatically generated checklists ([[qwen-3]] §4.3). The chat template (Table 9):

```
<|im_start|>user                     <|im_start|>user
{query} /think<|im_end|>             {query} /no_think<|im_end|>
<|im_start|>assistant                <|im_start|>assistant
<think>                              <think>
{thinking content}                   
</think>                             </think>
{response}<|im_end|>                 {response}<|im_end|>
```

Non-thinking samples keep an empty think block. In multi-turn data, several flags are inserted and the response follows the last one. A thinking budget is enforced at inference by inserting "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"; the report states this ability "is not explicitly trained" (§4.3). Distillation into Qwen3-0.6B to 14B and Qwen3-30B-A3B first uses teacher outputs in both modes (off-policy), then minimizes the KL divergence between student and teacher logits on student-generated responses (on-policy) (§4.5).

**Llama-Nemotron system instruction.** Reasoning samples carry "detailed thinking on" and non-reasoning samples "detailed thinking off". Pairs are built by sampling prompts from the reasoning data and generating non-reasoning responses with Llama-3.1-Nemotron-70B-Instruct (general prompts) or Llama-3.3-70B-Instruct (others), then filtering by ground truth or reward models ([[llama-nemotron]] §3.2). In math, reasoning-on samples number 2,225,427 and reasoning-off samples 19,840,970 (Table 2). LN-Nano scores 61.3 on AIME24 with reasoning on and 3.0 with it off (Table 3). On IFEval, LN-Nano-SFT scores 69.9 in both modes; the final LN-Nano scores 79.29 on and 82.1 off after RPO stages that "mainly targeted IFEval accuracy improvement" (Table 3, §7.2).

**DeepSeek-V3 expert data.** Each domain expert is trained on two sample types, <problem, original response> and <system prompt, problem, R1 response>, where the system prompt asks for reflection and verification; RL with high-temperature sampling teaches the expert to produce these patterns without the system prompt, and the experts then generate SFT data by rejection sampling ([[deepseek-v3]] §5.1). On DeepSeek-V2.5, this data raised MATH-500 from 74.6 to 83.2 while mean response length grew from 769 to 1510 tokens (Table 9).

**DeepSeek-R1 cold-start style.** The authors prefer a thinking process "that begins with comprehending the problem, followed by detailed reasoning that incorporates reflection and verification", written in the first person and in the language of the question ([[deepseek-r1]] v2 B.3.2). DeepSeek-R1-Zero samples at temperature 1.0 are filtered for correct answers (sympy for math) and readability (repetition detection, a language-mixing filter); DeepSeek-V3 refines them with the instruction "Translate the thinking process to the same language as the question."; human annotators convert traces to a conversational style, an LLM rewrites more data from those examples, and a second human pass verifies the output ([[deepseek-r1-recipe]] B.3.2). The summary prompt (Listing 1) begins:

```
Based on the above thought process, provide a clear, easy-to-follow, and well-formatted
solution to the question. Use the same language as the question.
The solution must strictly follow these requirements:
- Stay faithful and consistent with the given thought process. Do not add new reasoning
steps or conclusions not shown in the original.
```

**Magistral and Phi-4-reasoning system prompts.** Magistral's system prompt contains "Write both your thoughts and summary in the same language as the task posed by the user", and the authors report that RL is "quite sensitive" to the system prompt ([[magistral]] §2.2.4). Phi-4-reasoning trains with one fixed reasoning system message. Replacing it with generic variants during training made the model more robust to random system messages at inference but produced "greater variability in benchmarks scores and a slight decrease in average benchmark performance" under the original message ([[phi-4]] arXiv:2504.21318v1 §3.1). Safety responses were generated with detailed guidelines in the teacher prompt, and the guidelines were removed from the training prompt "to incentivize the model to implicitly learn the expected behavior" (§2.2).

| Pipeline | Teacher | Temperature | top-p | Max new tokens | Source |
|---|---|---|---|---|---|
| OpenMathReasoning CoT | DeepSeek-R1, QwQ-32B | 0.7 | 0.95 | 16,384 | [[openmathreasoning]] §2.3 |
| OpenCodeReasoning | DeepSeek-R1 | 0.6 | 0.95 | 16k | [[opencodereasoning]] §2.2 |
| Llama-Nemotron code | DeepSeek-R1 | 0.6 | 0.95 | not reported | [[llama-nemotron]] §3.1.2 |
| DeepSeek-R1 cold start | DeepSeek-R1-Zero | 1.0 | not reported | not reported | [[deepseek-r1-recipe]] B.3.2 |
| OpenR1-Math-220k | DeepSeek-R1 | "model card's recommended parameters", values not printed | not printed | 16k | [[openr1-recipe]] Update #2 |
| Bespoke-Stratos-17k, released script | DeepSeek-R1 API | 0.0 (`"temp": 0.0`) | not reported | not reported | [[bespoke-stratos]] `generate_numina_data.py` L78 |
| OpenThoughts3 | QwQ-32B | not reported | not reported | not reported | [[open-thoughts]] App. D-E |
| Phi-4-reasoning | o3-mini, high reasoning effort | not reported | not reported | not reported | [[phi-4]] §3.2 |

None of these sources reports an ablation of teacher temperature for SFT data. The OpenCodeReasoning sweep over temperatures 0.0, 0.2, 0.6, 0.7, and 1.0 is for student inference, where 0.6 and 0.7 were best ([[opencodereasoning]] §3). Status: Open question for teacher sampling temperature.

**Length limits in teacher output.** In OpenR1-Math-220k, only 75% of problems were solvable under 8k tokens and most others needed 16k ([[openr1-recipe]] Update #2). Hermes 4 data contains thinking traces up to 16 thousand tokens, yet the Qwen3-14B student reached its 40,960-token limit in 60% of LiveCodeBench generations; a second SFT stage inserted `</think>` at 30,000 tokens into the model's own generations and trained only `</think>` and `<eos>`, which moved LiveCodeBench v6 from 28.6 to 44.2 and AIME'25 from 48.7 to 42.5 ([[hermes-4]] §3.1, Table 2). Length control in RL is covered in [[ch-44a]].

Implication for a general-purpose model: mode-paired data is the documented method for keeping a direct-answer mode in a reasoning student, and the task coverage of the non-thinking half determines which non-reasoning behaviors are trained at all.

## §4 Teacher choice and student capacity

**Teacher choice** is the selection of the model that writes the training traces. The measurable problem is that a teacher's own benchmark score does not predict the score of the student trained on its outputs.

**Evidence.**
- OpenThoughts, Qwen2.5-7B-Instruct students, 31,600 examples: QwQ-32B traces averaged 44.2 against 42.3 for DeepSeek-R1 with a code question source, and 44.2 against 41.6 with a math source, although DeepSeek-R1 itself scores higher on CodeElo (53.7 vs 44.3), GPQA-Diamond (73.7 vs 65.0), and JEEBench (92.3 vs 69.9) ([[open-thoughts]] §4.6, Tables 8 and 29). The appendix tables for the same comparison print lower DeepSeek-R1 averages (38.0 for code, 40.6 for math) with the same QwQ-32B value of 44.2 (Tables 47-48), so the direction of the result does not depend on which table is used. Claude 3.7 annotations were slightly worse than DeepSeek-R1 for code and science (Tables 19-20).
- [[distillation-source-matters]]: 1.89M shared queries; each teacher regenerates until its response scores at least 0.9 on the category verifier; identical SFT of Qwen2.5-32B base (LR 8e-5, 2 epochs, batch 64, 32k tokens). Students of AM-Thinking-v1 / Qwen3-235B-A22B / DeepSeek-R1 score 84.3 / 79.4 / 70.9 on AIME2024, 72.2 / 62.2 / 52.8 on AIME2025, and 65.9 / 59.6 / 57.0 on LiveCodeBench (Table 1). Mean perplexity of the three training sets is 2.5 / 3.0 / 2.9 (§2.3, Fig. 5). One run per teacher; AM-Thinking-v1 is the authors' own model.
- Phi-4-reasoning: o3-mini at medium effort had "similar effect to DeepSeek-R1" as a teacher and was more token-efficient; high effort was "a stronger teacher than medium-effort consistently across tasks" with longer responses ([[phi-4]] §3.2). No numbers are printed.

Status: Replicated that the choice of teacher changes the student score at fixed prompts and fixed SFT settings ([[open-thoughts]] Table 8; [[distillation-source-matters]] Table 1). Result (single study) that a lower-scoring teacher can be the better teacher: [[open-thoughts]] reports the teachers' own scores (Table 29), while [[distillation-source-matters]] does not report the three teachers' scores in its text and so does not test this point.

**Learnability gap.** [[small-models-learnability-gap]] defines

∆Long = P_Long − P_Short,  ∆Large = P_Large − P_Small

where P_Long and P_Short are a student's average score over five math benchmarks after SFT on long CoT (from QwQ-32B-Preview) or short CoT (from Qwen2.5-32B-Instruct), and P_Large and P_Small are the scores after SFT on short CoT from Qwen2.5-72B-Instruct or Qwen2.5-3B-Instruct, whose responses average 432.98 and 440.70 tokens. For Qwen2.5-3B-Instruct, ∆Long = 40.3 − 43.4 = −3.1; for Qwen2.5-32B-Instruct, ∆Long = 73.0 − 59.3 = +13.7 (Table 1). ∆Long is negative for Qwen2.5 0.5B, 1.5B, 3B and Llama-3.2 1B, 3B, and positive from 7B upward. Mixing long and short CoT at a 1:4 ratio gave Qwen2.5-3B an average of 45.9, above both pure sets (Table 3). Setting: 7,500 MATH prompts, greedy teacher decoding, two epochs, LoRA above 14B, math benchmarks only. Status: Result (single study).

**Capacity-gap law.** [[capacity-gap-law-distillation]] fits the teacher size that gives the best student:

T* ≈ 2.498 · S − 11.498  (R² = 0.9957)

where S is the student size and T* the best teacher size, both in millions of parameters. For S = 300M, T* ≈ 2.498 × 300 − 11.498 = 737.9M; for S = 3,000M, T* ≈ 7,482.5M, which matches the authors' choice of 7B-8B teachers for 3B students (§4). Conditions: students are pruned from the teacher and trained with a token-level loss that is half teacher distribution and half ground truth, on pretraining-style data (§3.2, Eq. 4). The authors state that for distillation from generated text the concern "might be rather waived along the reduction of teacher knowledge from informative distributed probabilities to one-hot labels" (§2.1). The law is therefore evidence for logit distillation; for trace SFT the direct evidence is the teacher comparisons and the learnability-gap study above (Interpretation).

**Upper bound from SFT.** Llama-Nemotron states that "distillation inherently sets an upper bound on the student's performance" and that LN-Ultra-SFT "can approach the performance of DeepSeek-R1 but not exceed it" ([[llama-nemotron]] §5). LN-Ultra-SFT scores 66.4 on GPQA-Diamond, DeepSeek-R1 71.5, and LN-Ultra after RL 76.0 (Table 5). Other SFT-versus-RL comparisons:
- DeepSeek-R1-Distill-Qwen-32B reaches 72.6 on AIME2024 against 47.0 for RL from Qwen2.5-32B base for over 10K steps ([[deepseek-r1-recipe]] F.1, Table 16).
- Magistral Small (24B), AIME'24 pass@1: SFT on Magistral Medium traces 65.4, RL only 65.8, SFT then RL 70.7; maj@64 is 90.0, 86.7, 83.3 ([[magistral]] Table 3).
- Qwen3-8B from the same off-policy distilled checkpoint (55.0 AIME'24, pass@64 90.0): RL reaches 67.6 with pass@64 90.0 in 17,920 GPU hours; on-policy distillation reaches 74.4 with pass@64 93.3 in 1,800 GPU hours ([[qwen-3]] §4.7, Table 21).
- A student can exceed its teacher on a single benchmark: OpenThinker3-7B scores 72.4 on JEEBench against 69.9 for QwQ-32B ([[open-thoughts]] App. K).

Implication for a general-purpose model: off-policy traces show the student only teacher text, never its own errors; on-policy distillation and RL train on student samples ([[ch-35]], [[ch-38a]]).

## §5 More prompts, more responses per prompt, or more epochs

Let x be the number of distinct prompts, y the responses per prompt, and e the epochs; the SFT set has x·y examples and training sees x·y·e example passes.

**AceReason-Nemotron 1.1 regression.** Seven SFT sets v1-v7 (36K to 2.2M samples, Qwen2.5-Math-7B student, DeepSeek-R1 teacher) were fitted with ([[acereason-nemotron-1-1]] §4.4.2)

z = a · x̃ + b · ỹ + c

where z is the average accuracy on AIME24, AIME25, LiveCodeBench v5 and v6; x̃ and ỹ are log2 x and log2 y standardized to zero mean and unit variance across the seven sets; a and b are the fitted coefficients and c the intercept. The fit gives a = 4.831, b = 2.635, R² = 0.989.

Worked example: two sets that differ by one standard deviation of log2 prompts are predicted to differ by 4.831 accuracy points; one standard deviation of log2 responses per prompt, by 2.635 points, a ratio of 1.83. The standard deviations are not printed, so per-doubling effects cannot be computed from the paper. With 7 points and 3 parameters, 4 degrees of freedom remain, and v1-v4 all have one response per prompt, so x and y were not varied independently. Status: Result (single study).

Other evidence from the same paper: v7 keeps a similar number of prompts as v6 with "nearly twice as many responses per prompt", and AIME25 rose from 41.3 to 49.3 (§4.4.1, §4.4.3). For v6 and v7, accuracy rose from epoch 1 to 5 and plateaued around epochs 5-6; the authors attribute the benefit of "a certain degree of 'overfitting'" to exposure bias (Fig. 6; Interpretation).

**Answers per question in OpenThoughts.** The final pipeline uses 16 answers per question in all domains (§4.4). Sixteen answers with exact deduplication was the best science option (average 36.2, Table 43); in code, no deduplication with 16 answers tied exact deduplication with 4 answers at 41.3 (Table 41); in math, exact deduplication with one answer scored 41.7 against 40.1 with 16 answers, and the authors chose 16 answers "as it provides better scalability" (§4.4 and App. S.4, Table 42; the main text names 4 answers as the math winner, the appendix table shows 1). Worked example: the OpenThoughts3 dataset card lists filtering to 180k math, 60k code, and 60k science questions (300k), deduplication, downsampling to 75k questions, and 16 QwQ-32B annotations each: 75,000 × 16 = 1,200,000 examples, the final dataset size. With one answer per question, the same filtered pool could supply at most 300,000 examples (derived).

**Other multiplicity data.**
- OpenMathReasoning generates up to 32 candidates per problem, more for harder problems; 5.2M generated and 3.2M kept, a kept share of 61.5% (DeepSeek-R1 2.7M of 4.2M, 64.3%; QwQ-32B 0.5M of 1.0M, 50%; derived from [[openmathreasoning]] Table 5).
- GLM-4.5: four responses per hard prompt gave "an additional 1%-2% improvement" ([[glm-4-5]] §3.1).
- NaturalThoughts, random selection, Llama-3.1-8B-Instruct: GPQA-Diamond 37.1 at 1k, 42.5 at 100k, 48.3 at 500k ([[naturalthoughts]] Table 1).
- Hermes 4 includes "multiple unique trajectories to the same verified result" following OpenThoughts ([[hermes-4]] §2.2).

**Epochs.** Curated sets of 800 to 10k examples are trained for 5 to 15 epochs: LIMO trains 800 examples for 15 epochs ([[limo]] §4), s1 trains 1,000 for 5 epochs ([[s1]] App. D), NaturalThoughts uses 10 epochs for 1k and 6 for 10k ([[naturalthoughts]] §4). Llama-Nemotron reports that "extended training over multiple epochs improves performance, particularly for smaller models"; LN-Super was trained for one epoch at 5e-6 because of compute limits, while smaller runs "suggested that performance improves up to 3-4 epochs with larger learning rates (5e-5)" ([[llama-nemotron]] §4.1-4.2). OpenMathReasoning observes that "smaller models need to be trained for longer" ([[openmathreasoning]] Fig. 4).

**Stronger SFT before RL.** RL from the AceReason SFT v5 and v7 models narrowed their AIME24 gap from 6.6% to 1.6% ([[acereason-nemotron-1-1]] Fig. 7). LN-Ultra's RL was started from an earlier SFT checkpoint although later ones scored higher, "to improve final RL outcomes" ([[llama-nemotron]] §7.4). Status: Open question how long to train SFT before RL; the two reports point in different directions and neither varies SFT length in a controlled way.

Implication for a general-purpose model: none of these sources reports non-target suites as a function of epochs or responses per prompt. That measurement belongs to the forgetting analysis in [[ch-30a]] and the [[ch-36]] lab.

## §6 Student SFT settings: learning rate, length, and units

**Token loss averaging.** SFT loss is the mean cross-entropy over target tokens, and the averaging unit changes the gradient weight of each token. Worked example: a batch holds one 20,000-token trace and one 200-token answer. If the loss averages over all target tokens in the batch, every token has weight 1/20,200 and the short answer carries 200/20,200 = 1.0% of the total. If the loss averages within each sequence and then across sequences, each trace token has weight 1/(2 × 20,000) = 2.5 × 10⁻⁵ and each answer token 1/(2 × 200) = 2.5 × 10⁻³, a factor of 100. Llama-Nemotron reports that "models require higher learning rates to effectively learn from long reasoning traces, especially due to sequence-length-dependent token loss averaging" ([[llama-nemotron]] §4.1); the report does not state which averaging convention it uses, so this link is an Interpretation. Loss averaging is covered in [[ch-30]].

**Learning-rate evidence.**
- DeepDistill, Qwen2.5-72B base, about 5M samples: LR 8e-5 gave 79.2 on AIME2024 and 63.8 on LiveCodeBench; 8e-6 gave 72.5 and 60.2 ([[deepdistill]] §4.4.2).
- OpenCodeReasoning, Qwen2.5 7B-32B base and instruct: a grid over 1e-5, 3e-5, 5e-5, 8e-5, 1e-4 selected 5e-5 "across model sizes" ([[opencodereasoning]] §3).
- LN-Ultra: 5e-5 "generally improve[d] outcomes" but caused gradient explosions; the final run warmed up to 1e-5 and decayed to 1e-6, and still needed a restart with reinitialized optimizer states after the first epoch ([[llama-nemotron]] §4.2).
- Phi-4-reasoning, starting from the post-trained Phi-4: a grid over [1e-6, 2e-5] selected 1e-5; "higher learning rates result in lower training loss, but saturation and/or degradation across various downstream evaluations" ([[phi-4]] §3.1).
- OlympicCoder, SFT of an instruct model (Qwen2.5 Coder Instruct) on reasoning data: "the difference across each doubling of the learning rate amounted to almost 10 points improvement on LiveCodeBench", read from a figure; 4e-5 was used ([[openr1-recipe]] Update #3).

Conditions and limits: the results favoring 4e-5 or higher use tens of thousands to millions of examples and mostly base models (OpenCodeReasoning and OlympicCoder also train instruct models); Phi-4-reasoning starts from a post-trained model with 1.4M pairs and found 1e-5 best; LIMO (5.0e-6) and Light-R1-DS (5.0e-6) start from instruction-tuned or already distilled models with 800 or about 3k examples. Starting checkpoint, data size, and learning rate are confounded across these reports, and no study varies them jointly. Status: Open question.

**Units.** Hermes 4 prints 9,000 steps at a global batch of 384 samples and a 16,384-token context, and 56B training tokens per model ([[hermes-4]] §3, Table 1): 9,000 × 384 × 16,384 = 56.6B token positions (derived), consistent with 56B for packed batches. Phi-4-reasoning prints "roughly 16K steps" at batch 32 and context 32K, and 16B training tokens: 16,000 × 32 × 32,000 = 16.4B (derived from the rounded values) ([[phi-4]] §3, §3.2). OpenThoughts prints batch 512 without a unit ([[open-thoughts]] App. D.2). A batch value is comparable across reports only after converting to tokens per step.

## §7 Answer-side and trace-side quality control

**Answer-side quality control** compares the final answer or program with a reference: a rule verifier, a model judge, or execution. **Trace-side quality control** checks the reasoning text: language, format, repetition, and internal consistency.

**Answer-side mechanisms.**
1. Rule verifier with a model-judge fallback. For OpenR1-Math-220k, 55% of problems pass Math-Verify, and a Llama-3.3-70B-Instruct judge applied to a subset of rejected problems recovers 28,000 problems ([[openr1-recipe]] Update #2).
2. Model judge in place of a parser. Replacing Sky-T1's regex and sympy parser with a gpt-4o-mini judge raised the share of correct math solutions retained from 25% to 73% ([[bespoke-stratos]] "Data Curation"). Worked example: of 10,000 correct teacher solutions, the parser keeps 2,500 and discards 7,500 correct ones; the judge keeps 7,300 (derived).
3. Equivalence judge. Qwen2.5-32B-Instruct judges whether predicted and expected answers are equivalent "in the context of the problem" ([[openmathreasoning]] §2.3; [[llama-nemotron]] §3.1.1).
4. Structured judge. DeepSeek-R1's rejection-sampling SFT data uses DeepSeek-V3 with a prompt that classifies an answer as correct or incorrect and returns JSON with an 'analysis' key ([[deepseek-r1]] v2 B.3.3, Listing 4).
5. Consensus labels. When no answer can be extracted, the most common answer across candidates becomes the reference ([[openmathreasoning]] §2.3); Llama-Nemotron uses majority voting for science questions without ground truth (§3.1.3).
6. Thresholds and regeneration. DeepDistill keeps responses above 0.99 verifier score for math, code, and instruction following, above 4.99 on a 0-5 scale for science, and above 0.7 for multi-turn and other data ([[deepdistill]] §2.4.2); [[distillation-source-matters]] regenerates each response until it scores at least 0.9 (§2.2).
7. Many task verifiers. Hermes 4 rejection-samples against "roughly a thousand task-specific verifiers", and its DataForge judge revises failed answers until they pass or a maximum iteration count is reached ([[hermes-4]] §2.1-2.2).

**Trace-side mechanisms.**
- DeepSeek-R1 800K set: chains of thought with mixed languages, long paragraphs, or code blocks are removed ([[deepseek-r1-recipe]] B.3.3).
- Qwen3 cold start removes responses that (1) give incorrect final answers, (2) contain substantial repetition, (3) show guesswork without adequate reasoning, (4) have inconsistent thinking and summary, (5) mix languages or shift style inappropriately, or (6) are suspected of being overly similar to validation items; when QwQ-32B consistently fails, human annotators assess the responses ([[qwen-3]] §4.1).
- OpenCodeReasoning keeps responses with `<think>` tags and a code block in the solution, removes responses whose reasoning contains code blocks, and checks solution syntax with Tree Sitter ([[opencodereasoning]] §2.3).
- DeepDistill removes samples with perplexity above 20, 20-token strings repeated more than 20 times, odd turn counts, or missing think or answer segments ([[deepdistill]] §2.5).
- GLM-4.5 removes repetitive, short, truncated, or badly formatted samples, uses reward models for subjective responses, and removes tool trajectories that violate the call protocol or miss the expected terminal state ([[glm-4-5]] §3.1).
- LIMO scores each candidate chain (sampled from DeepSeek-R1, DeepSeek-R1-Distill-Qwen-32B, and QwQ-32B) with weights 30% length, 20% frequency of verification words ("check", "verify"), 25% frequency of tentative words ("perhaps", "might"), and 25% frequency of connectives ("therefore", "since"), with keyword counts normalized by text length; the best chain per problem is kept and the top 800 of 2,125 problems form the dataset ([[limo]] §3.1.2).

**Counter-evidence.**
- OpenThoughts, 7B, math questions, math average: from 63,200 answers, random selection of 31,600 scored 64.8; GPT verification 61.4; training on all 63,200 scored 65.6 at twice the data ([[open-thoughts]] §4.5, Tables 7, 45). For code questions at equal size, GPT verification (40.7) and majority consensus (41.3) did beat random selection (39.8) on the overall average, but no filter exceeded training on all answers (42.2) by more than 0.1 (Table 44). The authors' takeaway: "no filtering strategy outperformed the baseline, which uses all the answers" (§4.5).
- OpenThoughts, verified versus unverified OpenThoughts-114K (sizes not matched): the 7B average fell from 45.0 to 41.9 and the 32B average rose from 62.1 to 64.5 (App. H.1.1, Table 15). LLM-generated unit-test filtering at an equal 16,000 code examples gave LiveCodeBench 36.0 against 38.5 for a random unfiltered sample (App. H.1.4, Table 18).
- OpenCodeReasoning, Qwen2.5-14B-Instruct: no filtering (445,618) 54.1 on LiveCodeBench; passing solutions only (151,251) 47.0; solutions failing all tests (151,251) 52.3 ([[opencodereasoning]] Table 3).
- s1: the grader judged 53.6% of s1K traces correct, and the s1K-trained model reached 50.0 on AIME24 ([[s1]] §2, Table 2).

Status: Replicated that correctness filtering at equal size gave no gain or a loss for 7B-14B students in math ([[open-thoughts]] Table 45) and code ([[opencodereasoning]] Table 3; [[open-thoughts]] Table 18). Two results limit this: OpenThoughts code filters at equal size beat random selection (Table 44), and verification helped at 32B in a comparison that did not match sizes ([[open-thoughts]] App. H.1.1).

Mechanism (Interpretation, stated by the OpenCodeReasoning authors): teachers fail more often on hard questions, so keeping only verified answers at a fixed count shifts the set toward easier questions; "incorrect solutions span questions that are more challenging than the ones associated with the correct solutions" (§4.1, Fig. 4). An answer-filter ablation measures correctness only when the kept and discarded sets are matched on difficulty.

**Where strict verification is used.** Cold-start data is filtered for correctness and readability and verified by humans ([[deepseek-r1-recipe]] B.3.2). RL prompt sets are audited for wrong references: Magistral removes problems whose majority answer contradicts the reference ([[magistral]] §4.1), Light-R1 re-checks pass-rate-0 prompts with a model verifier ([[light-r1]] §4), and AceReason notes that "incorrect test cases can lead to false negative rewards" ([[acereason-nemotron-1-1]] §3.2.2). In these stages a wrong reference answer produces a wrong reward signal (Interpretation).

Implication for a general-purpose model: keeping unverified answers in bulk SFT trains wrong content as a positive target. None of the sources above measures the effect on factuality, calibration, or hallucination, so an unfiltered-versus-filtered comparison needs those suites in addition to reasoning benchmarks.

## §8 Decontamination, deduplication, and a distillation data card

**Decontamination** removes training prompts that match evaluation items. An **n-gram filter** flags a training item that shares any run of n consecutive tokens with a benchmark item; an **embedding filter** flags items whose embedding similarity to a benchmark item exceeds a threshold, often followed by an LLM or human review.

| Pipeline | Method | Benchmarks | Audit or note | Source |
|---|---|---|---|---|
| s1K | 8-gram overlap | MATH500, GPQA Diamond, AIME24 | — | [[s1]] App. C.5 |
| Open R1 | 8-gram match plus deduplication, following s1 | benchmark datasets | `scripts/decontaminate.py` | [[openr1-recipe]] README |
| AceReason-Nemotron 1.1 | 9-gram overlap | math and coding benchmarks | — | [[acereason-nemotron-1-1]] §3.1.1 |
| DeepSeek-V3 and R1 | 10-gram match on pre- and post-training data | evaluation questions and reference solutions | about six million potential math pretraining texts removed; paraphrases not caught | [[deepseek-r1-recipe]] v2 D.1 |
| OpenThoughts | character-level Indel similarity ≥ 75% or any shared 13-gram (Qwen2-7B-Instruct tokenizer) | evaluation set | misses 12 of 3,092 planted items; drops 1.4% of 3,000 clean items | [[open-thoughts]] App. F |
| Light-R1 | exact match ignoring digits, plus 32-gram | AIME24, AIME25, MATH-500, GPQA | audit of public sets (below) | [[light-r1]] §3.1.2, App. C |
| OpenCodeReasoning | nearest benchmark neighbor by cosine similarity (threshold 0.7), two LLM judges, manual review | LiveCodeBench, CodeContests, HumanEval, MBPP | 90 flagged items (≤ 0.3%) judged not paraphrases | [[opencodereasoning]] §2.1 |
| OpenMathReasoning, Llama-Nemotron | LLM-based comparison following Yang et al. (2023) | popular math benchmarks; GPQA, MMLU, MMLU-Pro for science | — | [[openmathreasoning]] §2.1; [[llama-nemotron]] §3.1.3 |
| DeepDistill, distillation-source-matters | exact match and bge-m3 similarity > 0.9 | evaluation set, AIME2024 named | — | [[deepdistill]] §2.2; [[distillation-source-matters]] §2.1 |
| Phi-4-reasoning | Phi-4 decontamination process | a printed list including AIME-2024, MATH, GPQA, LiveCodeBench, Codeforces, OmniMATH, SWE-Bench Verified, ArenaHard, MT-Bench | AIME-2025 released after data were finalized | [[phi-4]] §2.2 |
| Hermes 4 | not reported | — | — | [[hermes-4]] |

**Audit of public sets.** Light-R1 counts training prompts that match MATH-500 in public distillation sets: Open-Reasoner-Zero 325, data_ablation_full59K 244, DeepScaleR-Preview-Dataset 196, Bespoke-Stratos-17k 125, OpenThoughts-114k 100, OpenR1-Math-220k 10, s1K-1.1 3, LIMO 0; AIME24 and AIME25 matches are 0 for all listed sets ([[light-r1]] App. C, Table 7). The table counts matched training prompts, not distinct benchmark items, so the share of MATH-500 items affected cannot be computed from it; the text states that MATH-500 "contains tens of compromised questions that are either identical or differ only in numerical values" (§3.1.2).

**Worked example.** A 30-token benchmark question has a variant in the training pool that changes only the numbers at token positions 10 and 20. The shared token runs are positions 1-9 (9 tokens), 11-19 (9 tokens), and 21-30 (10 tokens). An 8-gram or 9-gram filter flags the variant because a 9-token run contains a shared 9-gram. A 13-gram or 32-gram filter does not flag it. OpenThoughts' Indel rule is similarity = LCS length / max(|s1|, |s2|), where LCS is the longest common subsequence and |s| is the length of string s (App. F, Eq. 1). The paper applies it to characters; applied to tokens for this example, the longest common subsequence has 28 tokens, so similarity = 28 / 30 = 0.93, above 0.75, and the rule flags the variant. Light-R1 covers this case with exact matching that ignores digits.

**Scope limit.** These filters compare training prompts with benchmark items. None of them tests whether a teacher's trace reproduces a benchmark solution that the teacher memorized. Date-filtered evaluation is the complementary check: AIME 2025 was released after the Phi-4-reasoning data were finalized ([[phi-4]] §2.2), and AceReason-Nemotron 1.1 reports AIME25 and LiveCodeBench v6 as benchmarks "that carry a lower risk of contamination" ([[acereason-nemotron-1-1]] §4.3). Detection methods are in [[ch-48]].

**Deduplication.** OpenThoughts chose exact deduplication for math and science and no deduplication for code, each with 16 answers per question; the authors report that "there does not seem to be a clear trend in types of deduplication that improve performance" (§4.4, Table 6 caption; Tables 41-43). OpenCodeReasoning deduplicates questions by exact match, and AceReason-Nemotron 1.1 deduplicates so that each prompt is unique (method not stated). No source here compares near-duplicate or semantic deduplication of distillation prompts; methods are in [[ch-12]].

**Distillation data card.** The table lists the fields that make a distillation set reproducible and checkable; the example column fills it from the OpenMathReasoning CoT subset with values as printed.

| Field | What to record | Example: OpenMathReasoning CoT subset ([[openmathreasoning]]) |
|---|---|---|
| Stage and students | stage name; student bases and sizes | distill-SFT; Qwen2.5-Base 14B and 32B, Qwen2.5-Math 1.5B and 7B with RoPE base 500K (§5.1) |
| Prompt sources and stage counts | each source; count after each filter | AoPS: 620K → 580K → 550K → 540K (Table 1) |
| Domain and task shares | shares by count and by tokens | math only; 260K converted proofs, 190K with answer, 90K without (Table 2) |
| Difficulty statistic | reference model; n; kept range | Qwen2.5-72B-Math-Instruct pass rate over 32; all kept, more samples for harder problems (§2.3) |
| Teacher and prompt | teacher models; system prompt; mode flags | DeepSeek-R1, QwQ-32B; system prompt not printed |
| Sampling | T; top-p; max new tokens; samples per prompt | 0.7; 0.95; 16,384; up to 32 (§2.3) |
| Answer-side QC | verifier; judge; recall audit | Qwen2.5-32B-Instruct equivalence judge; majority answer when missing; recall not reported |
| Trace-side QC | rules and thresholds | not reported for the CoT subset |
| Kept share | kept / generated | 3.2M / 5.2M = 61.5% (Table 5; derived) |
| Decontamination and dedup | method; n or threshold; benchmark list | LLM-based comparison (§2.1); n-gram length not applicable |
| Student SFT | LR; schedule; batch with unit; epochs; max length; packing; loss mask | 1e-4 (14B, 32B), cosine, 10% warmup, 1024 samples, 6 epochs, max length not reported, packing, loss mask not reported; final model is the average of 4 equally spaced checkpoints (§5.1) |
| Evaluation | held-out and date-filtered sets; non-target suites | Comp-Math-24-25 from 2024-2025 competitions; non-target suites not reported (§2.2) |
| Not-reported audit | every field the source does not give | max SFT length; loss mask; trace-side rules; verifier recall |

## Negative samples and negative feedback

This section uses the four meanings of "negative" from [[ch-31a]]: (1) negative marginal value, a sample filtered out because it would lower performance as a positive target; (2) negative as content; (3) negative as conditioning; (4) negative as gradient.

1. **Where negatives come from.** Answer-level labels from rule verifiers (Math-Verify, sympy), model judges (DeepSeek-V3, Qwen2.5-32B-Instruct, gpt-4o-mini, Llama-3.3-70B-Instruct), execution against unit tests, reward models (Llama-3.1-Nemotron-70B for general data in [[llama-nemotron]] §3.1.4), and majority-vote labels. Trace-level labels from format and language rules. Prompt-level labels from reference-model statistics: too easy, unsolved, guessable, or with a reference answer the model majority contradicts. Reported false negatives: the regex and sympy parser retained 25% of correct math solutions against 73% for a gpt-4o-mini judge ([[bespoke-stratos]]); the Llama judge recovered 28,000 problems that Math-Verify rejected ([[openr1-recipe]]); over half of Light-R1's pass-rate-0 RL prompts had unverifiable or incorrect references ([[light-r1]] §4).
2. **What current practice does.** OpenMathReasoning, Llama-Nemotron math, DeepSeek-R1, Qwen3, Light-R1, and OpenR1-Math-220k discard samples whose answers fail verification (meaning 1); [[distillation-source-matters]] (§2.2) and the Hermes 4 DataForge judge loop ([[hermes-4]] §2.1) regenerate or revise until a sample passes. Some sets keep unverified traces as positive targets: s1K, with 53.6% graded correct, and the OpenCodeReasoning release, whose ablation shows failing solutions still teach ([[s1]] §2; [[opencodereasoning]] §4.1). Light-R1 uses negatives as gradient (meaning 4) in a DPO stage with the NCA loss: rejected responses are verified-incorrect samples from the SFT-stage-2 model and chosen responses are verified-correct DeepSeek-R1 answers ([[light-r1]] §3.2).
3. **Mechanism.** In SFT on a kept trace, each target token y changes the logits z by ∂ log p_y / ∂z_j = 1[j = y] − p_j, where p_j is the model's probability of token j; every non-target token loses logit in proportion to its probability. Discarding a wrong trace removes its positive update and pushes nothing down. A negative-gradient term reverses the sign on the rejected token. Worked example: next-token probabilities are 0.7, 0.2, 0.1, and one gradient step of size 1 decreases log p of the 0.1 token. The logits change by −(1[j = neg] − p_j): −0.9 for the rejected token, +0.7 and +0.2 for the others. The new probabilities are 0.832, 0.144, 0.024. The rejected token loses 0.076 and the second token also loses 0.056, while the most likely token gains 0.132: all removed mass goes to the token that was already most likely. Full derivations are in [[ch-31a]] §2 and [[ch-43a]].
4. **Evidence with numbers.** Discarding wrong answers at equal size: no gain or a loss at 7B-14B in math ([[open-thoughts]] Table 7) and code ([[opencodereasoning]] Table 3). Verified versus unverified OpenThoughts-114K without matched sizes: a loss at 7B (45.0 → 41.9) and a gain at 32B (62.1 → 64.5) ([[open-thoughts]] Table 15). Negative-gradient stage: Light-R1-32B DPO moved AIME24 from 73.0 to 75.8 and AIME25 from 64.3 to 63.4 (Table 2); the authors report that for hard math, chosen responses from "significantly stronger models yielded better results" than fully on-policy DPO, without numbers (§3.2). No source here measures the share of a gain attributable to negatives.
5. **Controls.** Audit verifier recall on a labeled sample before filtering; add a judge fallback for parser failures; compare filters at matched difficulty; verify strictly where data seeds a cold start or an RL prompt set; for DPO-type stages on distilled traces, keep a positive log-likelihood term or bound the negative term, as in [[ch-31a]] §5.
6. **Diagnostics.** Kept share per verifier and per difficulty bin; length distributions of kept and discarded samples; correct and incorrect counts by difficulty level ([[opencodereasoning]] Fig. 4); for DPO stages, chosen and rejected log-probabilities logged separately; pass@k at large k before and after each filter change.
7. **Effect on generality.** Coverage: filtering by correctness at fixed count removes hard prompts (Interpretation from [[opencodereasoning]] §4.1), and Qwen3's on-policy distillation raised pass@64 while RL did not ([[qwen-3]] Table 21). Calibration and hallucination: not measured by these sources. Over-refusal: OpenThinker3-7B's XSTest over-refusal rose from 4.4 to 5.6 with no safety data in training ([[open-thoughts]] App. L), a data-coverage effect rather than a filtering effect. Forgetting: not measured as a function of filtering.

## Recipe

Units: "samples" and "examples" are prompt-response pairs; batch values are sequences unless stated; LR is peak learning rate.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenMath-Nemotron (OpenMathReasoning CoT data) | 1.5B-32B | distill-SFT | teachers; samples per problem; temperature; top-p; max tokens | DeepSeek-R1, QwQ-32B; up to 32, more for harder problems; 0.7; 0.95; 16,384 | arXiv:2504.16891v1 §2.3 | verified 2026-09-15 | no ablation reported |
| OCR-Qwen (OpenCodeReasoning data) | 7B-32B | distill-SFT | teacher; temperature; top-p; max output | DeepSeek-R1; 0.6; 0.95; 16k | arXiv:2504.01943v2 §2.2 | verified 2026-09-15 | no ablation reported |
| LN-Nano, LN-Super, LN-Ultra (math data) | 8B, 49B, 253B | distill-SFT | generations per problem | 16 DeepSeek-R1 (reasoning); 64 Qwen2.5-Math-7B-Instruct (non-reasoning) | arXiv:2505.00949v5 §3.1.1 | verified 2026-09-15 | no ablation reported |
| LN-Nano, LN-Super, LN-Ultra (code data) | 8B, 49B, 253B | distill-SFT | temperature; top-p | 0.6; 0.95 | arXiv:2505.00949v5 §3.1.2 | verified 2026-09-15 | no ablation reported |
| OpenThinker3-7B | 7B | distill-SFT | teacher; answers per question; teacher temperature, top-p, max tokens | QwQ-32B; 16; not reported | arXiv:2506.04178v2 §4.4, §4.6; checked App. D, E, R.3, dataset card | verified 2026-09-14; sampling not reported | Table 8 (teacher, 31,600-example runs); Tables 41-43 (16 answers best for science, tied for code, second for math; chosen for scalability) |
| DeepSeek-R1 (Dev1 cold start) | 671B MoE | SFT | generator; temperature | DeepSeek-R1-Zero; 1.0 | arXiv:2501.12948v2 B.3.2 | verified 2026-09-14 | no ablation reported |
| OpenR1-Math-220k | — | distill-SFT | max generation; samples per problem; temperature | 16k; 2, "in some cases, four"; not printed | HF blog Open R1 Update #2 | verified 2026-09-14; temperature not reported | no ablation reported |
| Bespoke-Stratos-17k curation script | — | distill-SFT | generation_params for DeepSeek-R1 (Numina subsets) | `{"temp": 0.0}` | curator main `generate_numina_data.py` L78 | verified 2026-09-14 (released code, not confirmed as the dataset run) | no ablation reported |
| DeepSeek-R1-Distill-Qwen-7B | 7B | distill-SFT | base; data; epochs; initial LR; schedule; context; batch | Qwen2.5-Math-7B; 804,745-sample B.3.3 set (Table 5); 2-3; 8×10⁻⁵; cosine to 1/10; 32,768; 64 | arXiv:2501.12948v2 B.4.3, Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-32B | 32B | distill-SFT | base; initial LR (other settings as the 7B row) | Qwen2.5-32B; 6×10⁻⁵ | arXiv:2501.12948v2 B.4.3, Table 6 | verified 2026-09-14 | no ablation reported |
| OpenThinker3-7B | 7B | distill-SFT | base; examples; LR; batch; epochs; packing; max length | Qwen2.5-7B-Instruct; 1.2M; 8e-5; 512 (unit not stated); 5; yes; not reported | arXiv:2506.04178v2 App. D.2 Table 9 "Large" | derived (1.2M falls in the > 31.6K range) | per-scale sweeps mentioned, results not reported |
| LN-Nano | 8B | distill-SFT | stage-1 data; LR; epochs; global batch; packed length | reasoning data only; 1e-4; 4; 256; 32k | arXiv:2505.00949v5 §4.2 | verified 2026-09-15 | stated reason: prevents repetitive completions; no table |
| LN-Super | 49B | distill-SFT | epochs; LR; sequence length; global batch | 1; 5e-6 fixed; 16k; 256 | arXiv:2505.00949v5 §4.2 | verified 2026-09-15 | smaller runs improved up to 3-4 epochs at 5e-5; no table |
| LN-Ultra | 253B | distill-SFT | LR schedule; packed length; global batch; epochs | warmup to 1e-5 (10%), cosine to 1e-6; 24k; 256; not reported | arXiv:2505.00949v5 §4.2 | verified 2026-09-15; epochs not reported | 5e-5 better in ablations but unstable; no table |
| AceReason-Nemotron-1.1 SFT-7B | 7B | distill-SFT | base; samples (v7); epochs; LR; batch; max length | Qwen2.5-Math-7B, rope_theta 1,000,000; 2.2M; not reported; not reported; not reported; not reported | arXiv:2506.13284v1 §3.1.2, §4.4.3 | verified 2026-09-15; epochs and optimizer values not reported (checked §3.1, §4.4) | Fig. 6: accuracy of v6 and v7 plateaus around epochs 5-6; §4.4.2 regression |
| OpenMath-Nemotron-14B, -32B | 14B, 32B | distill-SFT | data; epochs; peak LR; final LR; batch; warmup; weight decay | 5.5M (3.2M CoT, 1.7M TIR, 566K GenSelect); 6; 1e-4; 1000× smaller than peak; 1024 samples; 10% linear; 0.01 | arXiv:2504.16891v1 §5.1 | verified 2026-09-15 | Fig. 4 (accuracy over epochs, 1.5B and 14B) |
| OCR-Qwen-7B/14B/32B(-Instruct) | 7B-32B | distill-SFT | epochs; batch; max length; LR; warmup; packing | 3; 256; 32,768; 5e-5; ratio 0.1; yes | arXiv:2504.01943v2 §3 | verified 2026-09-15 | LR grid 1e-5 to 1e-4 |
| Light-R1-32B | 32B | distill-SFT | stage 1 LR / batch / length; stage 2 LR / batch / length; epochs | 5.0e-5 / 96 / 20k; 1.0e-5 / 32 / 20k; not reported | arXiv:2503.10460v4 App. D Table 8 | verified 2026-09-15; epochs not reported | Table 2 (stage-wise scores) |
| Light-R1-7B-DS | 7B | distill-SFT | start; examples; LR; batch; length | DeepSeek-R1-Distill-Qwen-7B; about 3k; 5.0e-6; 32; 20k | arXiv:2503.10460v4 §3.4, Table 8 | verified 2026-09-15 | Table 3 |
| Magistral Small | 24B | distill-SFT | start; data; epochs; checkpoint rule; LR, batch | Mistral Small 3 Instruct; Magistral Medium RL traces + responses to OpenThoughts and OpenR1-code prompts + 10% general instruction data; 4; best AIME'24; not reported | arXiv:2506.10910v1 §5.3 | verified 2026-09-14; LR, batch not reported | Table 3 (SFT vs RL vs SFT + RL) |
| Phi-4-reasoning | 14B | distill-SFT | examples; unique tokens; steps; batch; context; LR; warmup; weight decay | 1.4M pairs; 8.3B; ~16K; 32; 32K; 1e-5; 450 steps; 1e-4 | arXiv:2504.21318v1 §3 | verified 2026-09-15 | §3.1: LR grid [1e-6, 2e-5]; weight decay 0 vs 1e-4 within variance |
| Hermes 4 14B / 70B / 405B | 14B, 70B, 405B | distill-SFT | base; tokens; LR; steps; warmup; batch; context; epochs | Qwen3 14B / Llama 3.1 70B / Llama 3.1 405B; 56B each; 5e-5 / 1e-5 / 5e-6; 9,000, cosine; 300; 384 samples; 16,384; not reported | Hermes 4 Technical Report §3, Table 1 | verified 2026-09-15; epochs not reported | no ablation reported |
| LIMO (Qwen2.5-32B-Instruct) | 32B | distill-SFT | examples; LR; schedule; warmup; epochs; batch | 800; 5.0e-6; cosine; none; 15; 64 | arXiv:2502.03387v3 §4 | verified 2026-09-15 | no ablation reported |
| s1-32B | 32B | distill-SFT | examples; epochs; batch; steps; LR; warmup; betas; weight decay; loss | 1,000; 5; 16; 315; 1e-5; 5% (16 steps), cosine to 0; (0.9, 0.95); 1e-4; reasoning and solution tokens only | arXiv:2501.19393v3 App. D | verified 2026-09-15 | App. D.1 (sequence length) |
| Qwen2.5-72B base student (DeepDistill Stage I) | 72B | distill-SFT | samples; LR; epochs; packing; max length; batch | 5M; 8e-5; 1; yes; 32k; 64 | arXiv:2504.17565v3 §3.2, §4.1 | verified 2026-09-14 | §4.4.2: 8e-6 gave AIME2024 72.5 vs 79.2 |
| OpenR1-Distill-7B | 7B | distill-SFT | data; epochs; LR; schedule; batch; max length; packing; grad clip | Mixture-of-Thoughts 349,317 rows; 5; 4.0e-5; cosine to 10%, warmup 0.03; 128 (2 per device × 8 accumulation × 8 devices); 32,768; false; 0.2 | `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`@1416fa0; model card | verified 2026-09-14 | model card: all three domains > math + code |
| NaturalThoughts students | 8B, 7B | distill-SFT | LR; epochs (1k / 10k / 100k-500k); batch; max response length | 2e-5 constant; 10 / 6 / 8; about 400k tokens; 16,384 | arXiv:2507.01921v1 §4 | verified 2026-09-15 | no ablation reported |

**Starting point for a small general-purpose run.** For a 7B base student trained on 349,317 DeepSeek-R1 traces from math, code, and science on 8 devices with a 32,768-token context, the OpenR1-Distill-7B row is a documented configuration: 5 epochs, LR 4.0e-5 with cosine decay to 10% and 3% warmup, 128 sequences per step, no packing, gradient clipping at 0.2. For a 14B student starting from a post-trained model, Phi-4-reasoning used LR 1e-5, batch 32, 32K context, 450 warmup steps, and weight decay 1e-4 over 1.4M pairs. For the data, 16 teacher answers per question was the OpenThoughts choice at 7B, and the documented teacher sampling settings are temperature 0.7, top-p 0.95, 16,384 tokens for math (OpenMathReasoning) and temperature 0.6, top-p 0.95, 16k tokens for code (OpenCodeReasoning). None of these configurations was tested for retention of chat, instruction-following, or safety behavior. For a general-purpose student, mode-paired non-reasoning data as in Llama-Nemotron (§3.2) and scores on those suites are the documented additions.

## Generalization lens

**(a) What increases breadth.**
- Mode-paired data gives one model a reasoning mode and a direct-answer mode: LN-Nano scores 61.3 on AIME24 with reasoning on and 3.0 with it off ([[llama-nemotron]] Table 3). SFT with mode-paired data did not preserve instruction following: LN-Nano-SFT IFEval was 69.9 in both modes against 81.8 for Llama-3.1-8B-Instruct, and the RPO stages raised it to 82.1 off (Table 3, §7.2). Result (single study); Qwen3 uses the same design ([[qwen-3]] §4.3) but reports no paired ablation.
- Broad question pools: NaturalThoughts, drawn from 2.8M questions across domains, raised MMLU-Pro for Llama-3.1-8B-Instruct from 47.7 to 59.8 at 100k random examples and 61.9 at 500k; 100k OpenThoughts3 examples (math, code, science) gave 59.0 on MMLU-Pro but 82.2 on MATH-500 against 67.5 for 100k NaturalThoughts ([[naturalthoughts]] Table 1). Result (single study).
- Mixed-domain reasoning data with safety and alignment prompts: Phi-4-reasoning's IFEval strict rose from 62.3 (Phi-4) to 83.4, and ArenaHard from 68.1 to 73.3, with Phi-4 evaluated at temperature 0.0 and Phi-4-reasoning at 0.8 ([[phi-4]] §5.2, Table 2).
- Science data added to code data: GPQA-Diamond 52.7 for both mixes with science against 47.3 and 36.7 for code-only sets ([[open-thoughts]] App. H.5, Table 24).

**(b) What causes narrowing or forgetting.**
- Math-focused SFT stages: Light-R1-32B GPQA 64.3 → 60.6 at stage 2 ([[light-r1]] Table 2). The same 3k stage-2 data gave Light-R1-7B-DS no out-of-domain gain, "improvements confined solely to in-domain tasks" (GPQA 49.1 → 49.4), while Light-R1-32B-DS GPQA rose from 62.1 to 68.0 (§3.4, Table 3).
- Reasoning SFT and instruction following: LN-Super-SFT IFEval 81.9 against 92.1 for Llama-3.3-70B-Instruct ([[llama-nemotron]] Table 4). The Phi-4-reasoning result in (a) goes the other way; the two differ in data (Phi-4-reasoning adds alignment data), start model, and evaluation temperature, so they are not a controlled comparison.
- Missing safety data: OpenThinker3-7B HarmBench 14.5 → 55.5 ([[open-thoughts]] App. L).
- Parametric recall: Phi-4-reasoning's no-context Kitab recall fell from 8.2 to 4.9 while precision rose from 19.3 to 23.2 ([[phi-4]] Table 2).
- Teacher-student mismatch for small students: negative ∆Long for Qwen2.5 0.5B-3B ([[small-models-learnability-gap]] Table 1).
- Robustness: on six structure-preserving variants of an Alice-in-Wonderland problem, the authors report "strong performance fluctuations" for all tested distilled reasoning models, for example OpenThinker-32B near 1 on variations 2 and 3 and near 0 on variation 1 ([[open-thoughts]] App. N, Fig. 13).

**(c) How to measure it for this stage.**
- Non-target suites next to reasoning scores: IFEval, BFCL, Arena-Hard as in [[llama-nemotron]] Tables 3-5; HarmBench and XSTest as in [[open-thoughts]] App. L.
- Date-filtered benchmarks: AIME 2025 for Phi-4-reasoning; AIME25 and LiveCodeBench v6 for AceReason ([[acereason-nemotron-1-1]] §4.3).
- Enough samples per item: AIME2024 pass@1 standard deviation is 1.8 at avg@16, 1.2 at avg@32, and 0.7 at avg@64 ([[acereason-nemotron-1-1]] §4.1); Light-R1 observed deviations above 3 points with 16 or fewer samples ([[light-r1]] §2). Interval methods are in [[ch-51]].
- Bootstrap intervals for data ablations, as in [[s1]] Table 2.
- Contamination audit counts per public set, as in [[light-r1]] Table 7.
- pass@k at large k: pass@64 in [[qwen-3]] Table 21.
- Known measurement errors in these sources: selection by the same benchmarks later reported (OpenThoughts avoids this for AIME25, HMMT, HLE, LCB 06/24-01/25, [[open-thoughts]] §4); baselines evaluated once while the students average 64 runs ([[opencodereasoning]] Table 2); an evaluation claim of chat coverage without chat benchmarks ([[distillation-source-matters]] §3.2).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Difficulty filter described without reference model, n, and kept range | A rerun keeps a different prompt set or a different count | Log reference model, n, S, and kept counts per stage; recompute P(keep \| p) for borderline prompts |
| Pass-rate-0 prompts kept as "hard" without an answer audit | Items with unverifiable or wrong reference answers found on a manual check; RL groups where every sample gets zero reward | Re-check a sample of pass-rate-0 prompts with a judge, as Light-R1 did |
| Teacher chosen by its own benchmark score | Student underperforms a student of a lower-scoring teacher | Train small students on each teacher's traces on the same prompts |
| Long-CoT-only data for a ≤3B student | Score below short-CoT SFT on the same prompts | Compare long, short, and 1:4 mixed data |
| Answer filter adopted without a matched ablation | Hard-problem accuracy falls; kept set shorter than the discarded set | Compare filtered vs unfiltered at equal count within difficulty bins |
| Parser-only verifier | Low kept share on problems the teacher usually solves | Hand-label a random sample of rejected outputs and report the false-rejection rate; add a judge fallback |
| Batch size compared across reports without units | Learning rate copied from a report gives unstable or slow training | Convert batch to tokens per step before copying LR |
| n-gram length chosen without testing numeric variants | MATH-500 or similar score higher than date-filtered scores | Add digit-insensitive exact match or similarity filters; report per-set match counts |
| Reasoning-only SFT for a general assistant | IFEval, BFCL, or safety scores fall; no direct-answer mode | Add mode-paired non-reasoning data; evaluate both modes |
| Epochs set from a small-data recipe on a large set | Target scores plateau while non-target suites drop | Track target and non-target suites per epoch |

## Check your understanding

1. LIMO's stage-2 rule keeps 1-3 successes out of 32. Using P(keep | p), explain why a problem with p = 0.2 is rarely kept while one with p = 0.05 is usually kept, and what this implies for how the kept set changes if DeepSeek-R1-Distill-Qwen-32B is replaced with a stronger model.
2. In OpenCodeReasoning, failing solutions trained a better student than passing solutions at equal count. Give a causal account that uses the difficulty distribution, and design an ablation that would isolate the effect of correctness.
3. QwQ-32B traces produced better 7B students than DeepSeek-R1 traces although DeepSeek-R1 scores higher on several benchmarks. Which properties of a trace set, measured in [[distillation-source-matters]], could explain this, and why can verification pass rate not detect them?
4. The AceReason coefficients are 4.831 for prompts and 2.635 for responses per prompt. Why can these values not be converted into points per doubling from the paper, and why does the design of v1-v4 weaken the comparison?
5. The capacity-gap law was fitted on logit distillation. Explain why its authors expect a weaker effect for training on generated text, and which evidence in this chapter addresses teacher-student mismatch for trace SFT.
6. DeepDistill found 8e-5 better than 8e-6 at 72B, while Phi-4-reasoning found higher learning rates degraded downstream scores. List the differences in setting that make both results compatible, and explain how token loss averaging enters the argument.
7. A 30-token question and a training variant differ only in two numbers. Explain which of the 8-, 13-, and 32-gram filters flag it and why Light-R1 adds a digit-insensitive exact match.
8. A general-purpose 8B student is distilled on 95% math and code reasoning traces. Name three measurements from this chapter that would detect narrowing first, and the source result that motivates each.

## Connections

- Previous: [[ch-35]] — Distillation in Practice A: Where Labs Insert Teacher Data. That chapter places teacher data in the pipeline; this chapter builds one distillation dataset.
- Next: [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split. The data card and non-target measurements of §8 feed that lab.
- Dependency: [[ch-22]] — Quality, Diversity, and Gradient-Based Data Selection (selection scores and the length confound).
- Related: [[ch-20]] — Distillation as Data: Explanation Traces and the R1-Distill Lineage; [[ch-24]] — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data; [[ch-30]] — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate; [[ch-30a]] — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; [[ch-31a]] — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; [[ch-12]] — Deduplication: Exact, Near-Duplicate, and Semantic.
- Later: [[ch-16]] — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse; [[ch-38a]] — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity; [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; [[ch-44a]] — Length in RL: Overlong Responses, Length Control, and Long-Context RL; [[ch-48]] — Contamination Detection and Its Effect on Reported Scores; [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.

## Sources

- [[open-thoughts]] — source, mixing, question-filter, answers-per-question, answer-filter, and teacher ablations; decontamination rule; safety and robustness results.
- [[openmathreasoning]] — AoPS pool stages, teacher sampling settings, consensus labels, kept shares, student SFT settings (excerpt; no library card).
- [[opencodereasoning]] — competitive-programming aggregation, embedding plus LLM decontamination, scaling to 736k, execution-filtering ablation (excerpt; no library card).
- [[qwen-3]] — cold-start query filter, six response criteria, /think and /no_think template, on-policy distillation table. The library card is not re-verified; loci are arXiv:2505.09388v1.
- [[light-r1]] — two-stage pass-rate filter, 32-gram and digit-insensitive decontamination, audit of public sets, DPO negatives, GPQA drop (excerpt; no library card).
- [[limo]] — two-stage difficulty filter, rule-based chain score, 15-epoch SFT, question-difficulty ablation. The library card is not re-verified; loci are arXiv:2502.03387v3.
- [[s1]] — two-model difficulty filter, selection ablation with bootstrap intervals, 53.6% correct traces, 8-gram decontamination, SFT settings. The library card is not re-verified; loci are arXiv:2501.19393v3.
- [[magistral]] and [[magistral-recipe]] — two-pass difficulty grading, majority-versus-reference filter, system prompt, Magistral Small SFT and Table 3.
- [[glm-4-5]] — response-length prompt cut, four responses per hard prompt, SFT rejection filters.
- [[llama-4]] — removal of easy-tagged SFT data and 95% pruning for Behemoth.
- [[kimi-k1-5]] — guessability rule with N = 8 and excluded question formats.
- [[acereason-nemotron-1-1]] — prompts-versus-responses regression, epoch plateau, 9-gram decontamination, SFT-then-RL gap (excerpt; no library card).
- [[distillation-source-matters]] — controlled three-teacher comparison on 1.89M queries.
- [[small-models-learnability-gap]] — ∆Long and ∆Large by student size; Mix Distillation (excerpt; no library card).
- [[capacity-gap-law-distillation]] — T* ≈ 2.498·S − 11.498 and its scope (excerpt; no library card).
- [[deepdistill]] — verifier thresholds, trace filters, 72B learning-rate ablation.
- [[naturalthoughts]] — selection strategies at 1k to 500k, scaling with random selection, mixed System-1/System-2 distillation (excerpt; no library card).
- [[phi-4]] — Phi-4-reasoning seed selection, fixed system message, teacher effort, LR grid, decontamination list, general benchmarks. The library card is not re-verified; loci are arXiv:2504.21318v1.
- [[bespoke-stratos]] — parser-versus-judge retention (25% to 73%), released generation parameters, student settings.
- [[openr1-recipe]] — Math-Verify pass rate and judge recovery, 16k generations, 8-gram decontamination, OpenR1-Distill-7B configuration, OlympicCoder learning-rate observation.
- [[deepseek-r1]] and [[deepseek-r1-recipe]] — cold-start style and filters, V3 judge prompt, 800K domain table, R1-Distill settings, Table 16, 10-gram decontamination.
- [[deepseek-v3]] — expert data with a reflection-and-verification system prompt; Table 9.
- [[llama-nemotron]] — "detailed thinking on/off" pairs, data shares, SFT settings by model, distillation upper bound, IFEval drop, RL prompt filter (excerpt; no library card).
- [[hermes-4]] — task verifiers and judge loop, training units, 30k `</think>` length-control stage (excerpt; no library card).
