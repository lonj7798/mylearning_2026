<!-- chapter: ch-24
     track: synthetic
     kind: content
     title: Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data
     deps: [ch-23]
     sources: [[openmathinstruct]], [[openmathinstruct-2]], [[openmathinstruct-2-recipe]], [[metamath]], [[mathscale]], [[mammoth]], [[mammoth-2]], [[rstar]], [[rstar-math]], [[s1]], [[limo]], [[step-dpo]], [[lema-learning-from-mistakes]], [[omegaprm]], [[math-shepherd]], [[transferability-of-llm-reasoning]], [[scaling-reasoning-losing-control-mathif]], [[gsm-symbolic]], [[longer-context-deeper-thinking]], [[open-thoughts]], [[front-loading-reasoning]], [[likelihood-displacement]], [[numina-math]], [[quiet-star]]
     figures: figures/rstar-mcts.html
     revised: 2026-09 (generality revision)
-->

# Chapter 24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data

> **Core insight.** Reasoning-trace data is produced in three ways: sampling multiple solutions per problem from a teacher and keeping those whose final answer matches, searching over individual steps, or selecting a small set of long traces. In the sources, the largest measured data lever is on the problem side: at a fixed 256K pairs, raising unique questions from 1K to 6.5K added 10.5 MATH-validation points, while judge and reward-model filtering of solutions gave no gain ([[openmathinstruct-2]], §1, Table 3, Fig. 6). Long chain-of-thought SFT on 800–1,000 examples reaches competition-math scores on a strong base but not on a weaker one of the same size (LIMO data on Qwen2.5-32B-Instruct: 63.3 AIME24; on Qwen1.5-32B-Chat: 9.2), and math-only SFT of Qwen3-14B-Base on Qwen3-32B traces lowered IFEval from 69.2 to 42.3 while GRPO on the same problems gave 70.0 ([[limo]] §6.3.3; [[transferability-of-llm-reasoning]] Table 1).
>
> **Guideline.** When the example budget is fixed, add distinct problems before adding solutions per problem, because in the OpenMathInstruct-2 ablation 6.5K questions scored 10.5 points above 1K questions at the same 256K pairs (one 8B student, math only). Otherwise, when no new problems can be collected or written, add solutions per problem and expect smaller gains: after 80K answer-augmented samples, 20K more added 0.1 GSM8K points for LLaMA-2-7B ([[metamath]] §4.5). When a wrong trace is used as a negative, locate the first wrong step and apply the negative gradient to that step only, because Step-DPO scored 55.8 vs 55.0 MATH for whole-answer DPO at 5K pairs on Qwen2-7B-SFT and 64.1 vs 62.5 on Qwen2-72B-SFT (single study, no run counts; [[step-dpo]] Table 3); otherwise use the failure as content (a correction target) or discard it. When traces are longer than the SFT sequence length, raise the length or drop the example, because a 4,096-token s1 run cut 74% of samples and scored 30.0 vs 50.0 AIME24 ([[s1]] App. D.1). When the model must stay general, measure instruction following, a non-math reasoning suite, and perturbed math problems before and after trace SFT, because the regressions above do not appear on math benchmarks.

## Corrections to the version you studied

1. "the *verifier* matters more than the *teacher* … Get the verifier wrong and more data makes the student *worse*" → OpenMathInstruct-2 finds SFT robust to up to 20% wrong-answer solutions at ≥256K pairs, no gain from judge or reward-model filtering, and question diversity as the driver of data-scaling gains; no source here shows unchecked data making the student worse ([[openmathinstruct-2]], Abstract, §2.2.3–2.2.4, Table 3, Figs. 5–6).
2. "Chapters 21-23 argued the general case" and "the one modality-specific drill-down" → the general synthetic-data chapters are ch-18 to ch-23, and ch-25 to ch-28 are also modality chapters (outline).
3. "Track 4's RL chapters (ch-48 onwards)", "ch-48 (RLVR), ch-52 (PRM-weighted reward)" → the RL phase runs from ch-40 to ch-46a in outline order; process rewards and verifiable rewards are ch-44; ch-48 is contamination detection and ch-52 is safety evaluation (outline).
4. "ch-22, ch-23 (synthetic data at scale …)" → ch-22 is "Quality, Diversity, and Gradient-Based Data Selection" (outline).
5. "sample K=32-64 solutions per problem from Mixtral-8x7B-Instruct" → Mixtral-base with 5-shot prompts; GSM8K 128 samples per prompt (256 total), MATH 224 per prompt (896 total) ([[openmathinstruct]], §2.1, Table 2).
6. "averaging ~120 solutions per GSM8K problem and ~100 per MATH problem", "~500K GPU-hours", "Filter by SymPy-canonical equivalence", "Apache-2.0" → not in the paper; grading details and compute are not reported; the dataset card gives an NVIDIA license that permits commercial use ([[openmathinstruct]], Verification).
7. "the 15K seed grows to ~600K via … Paraphrase … Novel question conditioned on the topic tag" → new questions are written from 5 few-shot pairs of an original and a "similar" question, with no topic or difficulty instruction; 607.3K unique questions, 592K of them synthesized ([[openmathinstruct-2]], §3, Table 5).
8. "at 405B scale, text-CoT outperforms TIR" → the format ablation compares two text formats, OpenMath CoT 44.5 vs Llama CoT 40.6; the paper has no tool-integrated comparison ([[openmathinstruct-2]], Table 1).
9. "~650K H100-hours", "served BF16 via vLLM", "students … do not acquire backtracking" → not reported by the paper ([[openmathinstruct-2]], Verification).
10. "Llama-3.1-405B at 1M samples beats Mixtral at 10M samples" → the only teacher ablation is Llama3.1-405B-Instruct vs Llama3.1-8B-Base at matched coverage, 37.9 vs 30.1 MATH validation ([[openmathinstruct-2]], Table 2).
11. "Self-Verification (SV) — Given Q and candidate answer A', is A' correct? If not, fix." → SV rewrites the question and answer as a declarative statement, masks one number as x, and asks for x ([[metamath]], §3.3, App. A.2).
12. The FOBAR example "Jane has 3 apples …" and "AnsAug ~+4, Rephrasing +3, SV +2, FOBAR +3 — additive" → the example is invented (the paper's Example 3.4 masks "x packs of beef", answer 110); Table 3 (LLaMA-2-7B, GSM8K) gives 41.6 original SFT, 59.6 AnsAug, 60.6 +Rephrasing, 64.4 +SV and FOBAR together ([[metamath]], Table 3).
13. "topics (~2K) and knowledge points (~5K)", "Sampling rare (topic, concept) edges", "~5% of 'gold' answers are wrong" → 2,018 topics and 8,892 knowledge points; the walk probability is proportional to co-occurrence count, which favors frequent pairs; GPT-4 marked 26% of a 5K sample of solutions incorrect, and correcting them did not improve a 5K run ([[mathscale]], §4.1, Eqs. 1–2, §5.3, Table 7).
14. "ratio PoT:CoT ≈ 57:43", "CoT-only drops MATH by 8 points; PoT-only drops AQuA by 15", the `def solution()` template → Table 1 rows give about 73% CoT (derived); CoT-only MATH 9.9 vs hybrid 31.5, PoT-only AQuA 28.6 vs 44.5; the template is not from the paper ([[mammoth]], Tables 1, 6).
15. "[[mammoth-2]] … Recall → Extract → Refine with Mixtral, trading verifier strength (LLM-judge …)" → Qwen-72B extracts, Mixtral-22B×8 and Qwen-72B refine, and no judge-score filter is reported ([[mammoth-2]], §2.2–2.3).
16. s1 "three filters in sequence — difficulty, diversity, quality (manual trace-style check)" → the order is quality (API errors and formatting patterns, automated), difficulty (drop questions solved by Qwen2.5-7B-Instruct or Qwen2.5-32B-Instruct, graded by Claude 3.5 Sonnet), then diversity (MSC domains, sampling toward longer traces) ([[s1]], §2.2).
17. The comparison table ("base 0 | ~17 | ~84", "OpenMathInstruct-2 SFT 14M | ~40 | ~90", "Random 1K from 59K | ~24 | ~86") and its figure copy → no source has these rows; s1 reports 1K-random 36.7 / 90.6 and s1K 50.0 / 93.0 AIME24 / MATH500 under budget forcing (Table 2) and Qwen2.5-32B-Instruct 26.7 / 84.0 (Table 1); neither s1 nor LIMO trains on OpenMathInstruct-2 ([[s1]], Tables 1–2).
18. "LIMO … 817 long-CoT samples … 63.3 AIME24, 95.6 MATH500" → version mix: arXiv v1 reports 817 samples with 57.1 / 94.8; v3 (COLM 2025) reports 800 samples with 63.3 / 95.6 ([[limo]], v1 and v3 Abstracts).
19. LIMO "hand-filter removes traces …" and pool "GSM8K-hard, physics olympiad" → v3 keeps the highest rule-scored chain per question (length 30%, verification words 20%, tentative words 25%, connectives 25%) from DeepSeek R1, R1-Distill-Qwen-32B and QwQ-32B samples; the pool is NuminaMath-CoT, DeepScaleR, pre-2024 AIME, MATH, and Chinese exam problems ([[limo]], v3 §3.1).
20. "substitute 817 short-CoT traces and the result is closer to the random-1K baseline" → neither paper runs this; LIMO compares chain-quality levels L1–L5 (Fig. 3) and NuminaMath-100k short CoT (average 32.3 vs base 49.9, Table 1) ([[limo]]).
21. rStar actions "A2: decompose into subquestions; A3: directly answer a subquestion, then verify; A4: rephrase; A5: propose a new intermediate subquestion" → A1 one-step thought; A2 remaining thought steps; A3 next sub-question with its answer; A4 re-answer that sub-question with few-shot CoT; A5 rephrase the question by listing its conditions ([[rstar]], §3.2).
22. "a *separately-prompted* discriminator (same base model, different prompt)" and "12.5 → 63.1 (+50.6)" → the discriminator is Phi3-mini-4k (3.8B) for every generator (self-discrimination only when Phi3 generates); LLaMA2-7B GSM8K is 12.51 → 63.91 ([[rstar]], §4.1, Table 2).
23. rStar-Math pseudocode "c.Q + c_puct * c.P * sqrt(node.N) / (1 + c.N)" and "r = int(extract_boxed(leaf) == gold_answer)" → UCT(s) = Q(s) + c·sqrt(ln N_parent(s) / N(s)) with Q = q/N and c = 2; terminal nodes score +1 or −1 ([[rstar-math]], Eqs. 1–2, App. A.1).
24. "Round 0: Qwen2.5-Math-7B-Instruct as bootstrap generator … 747K problems from [[numina-math]] + olympiad + AIME archives" → round 1 uses DeepSeek-Coder-V2-Instruct (236B) with 8 rollouts; the 747K problems come mainly from NuminaMath (competition level only) and MetaMath, plus GPT-4 problems seeded from MATH and AMC-AIME ([[rstar-math]], §3.4, Table 2).
25. "siblings with Q-gap > δ form pairs … authors argue pairwise avoids the Goodhart pathology" and "top-K trajectories by PPM score" → per step, the two highest-Q candidates leading to correct answers are positives and the two lowest-Q candidates leading to wrong answers are negatives; the stated reason is that Q-values are too imprecise as regression labels; SFT uses the top-2 correct trajectories by average Q, fine-tuned from the base model each round ([[rstar-math]], §3.3–3.4, App. A.1).
26. "MATH climbs 58 → 78 → 85 → 88 → 90"; "90.0 MATH, 53.3 AIME24, 58.5 Olympiad" → with 8 PPM-guided trajectories: 58.8 (base), 75.2, 86.6, 87.0, 89.4 (Table 6); the policy alone: 69.6, 73.6, 75.8, 78.4 (Table 3); 90.0 / 53.3 / 65.6 OlympiadBench need 64 trajectories and the 7B PPM (Table 5).
27. OmegaPRM "K=16 and L=10 … 4× saving", "1.5M step labels over ~80K problems → PRM regressed via MSE", "weighted best-of-N (PRM × policy log-prob)", "~7% rate on OpenMathInstruct-2 audits" → k = 8 rollouts and 16 pieces per solution; 1.5M annotations from 12K MATH training questions; classification loss on soft labels; PRM-weighted majority voting with the product of step scores; the 7% figure is in no source ([[omegaprm]], §3.2–§4; [[openmathinstruct]] §2.3 calls flawed-reasoning positives "rare").
28. Step-DPO "Strong teacher (GPT-4 / Qwen2-72B) identifies the first erroneous step … Teacher generates a corrected step" → the first wrong step is found manually or by GPT-4; the chosen step is sampled from the reference model itself; GPT-4-corrected steps gave 55.1 vs 55.8 MATH ([[step-dpo]], §3.2, Table 4).
29. "10K pairs beat full-trajectory DPO on 100K pairs (MATH 58.6 vs 54.3 on Qwen2-7B)" and "the KL denominator washes out the signal" → at 5K pairs each, Qwen2-7B-SFT 54.8 → 55.0 (DPO) vs 55.8 (Step-DPO); the paper's reason is that rejecting a whole answer also rejects its correct earlier steps ([[step-dpo]], §3.1, Table 3).
30. "Both methods depend on gold answers and a stronger teacher" and "Synthetic generation from non-reasoning teachers will not produce backtracking" → OmegaPRM labels steps with Monte-Carlo rollouts and uses no human or stronger-model step labels; Step-DPO needs GPT-4 or a person only to locate errors; rStar-Math shows backtracking in search outputs although no self-reflection data or prompt was used (Fig. 4, App. A.2) ([[omegaprm]] §4; [[step-dpo]] §3.2; [[rstar-math]] §5).
31. "Track-4 will revisit [Quiet-STaR]" → no chapter in the outline lists [[quiet-star]]; its section is removed because it is continued pretraining, not trace synthesis.
32. Figure panel "Attested anchors" and "Curation beats volume" → the rows and the predicted curve had no source; the figure is rewritten.

Section map from the old version: §1 → §1; §2 → §2; §3 → §3; §4 → §4; §5 → §5; §6 → §6 and "Negative samples and negative feedback"; §7 removed; §8 → Guideline and Recipe. §7 and §8 of this version are new.

## Why this chapter matters for a general-purpose model

A reasoning trace is the text a model writes between a problem and its answer. Trace data enters the pipeline at four points: distillation SFT on teacher traces (ch-20), process reward models and step preferences (ch-44), cold-start SFT before RL (ch-31), and reasoning data in pretraining ([[front-loading-reasoning]], at 8B: in pretraining a 336B-token diverse reasoning corpus scored above a 1.2M-example long-trace corpus, 64.09 vs 54.98 average, Table 1; in SFT the long-trace corpus scored above the diverse one, 44.99 vs 31.54, Table 5). Most trace datasets in this chapter are math, where a final answer can be checked by string or symbolic match; even the multi-domain OpenThoughts3 mixture is 850,000 math, 250,000 code, and 100,000 science examples ([[open-thoughts]] §5). For a general-purpose model this creates three measurable risks. First, gains may reflect memorized benchmark patterns: for 21 of 25 models, the score on the original GSM8K questions lies more than one standard deviation from the mean over templated variants, usually above it ([[gsm-symbolic]], §4.1). Second, trace SFT may lower abilities outside math (§7). Third, long traces interact with sequence length and instruction following (§7, §8). This chapter covers how traces are generated and filtered, and how to measure those three risks.

## §1 Terms and design choices

- **Chain-of-thought (CoT) trace**: intermediate natural-language steps before the answer, typically hundreds of tokens in the math SFT sets below (OpenMathInstruct-2 drops solutions over 1,024 tokens; [[openmathinstruct-2]] App. A.2).
- **Long CoT**: traces with self-checks and restarts; s1-32B averages 6,984 thinking tokens on AIME24 ([[s1]] Table 8), and LIMO training responses are under 16,384 tokens ([[limo]] §4).
- **Code-interpreter (tool-integrated) solution**: text interleaved with code whose executed output is inserted before generation resumes ([[openmathinstruct]] §2.1).
- **Step-level data**: a label or preference attached to one step given its prefix, rather than to a whole answer.
- **Verifier**: the check that accepts a trace; in these sources it is final-answer match, code execution, rollouts from a prefix, a second model, or a trained reward model.

| Design choice | Options in this chapter's sources | Where the evidence is |
|---|---|---|
| Problem source | train sets (GSM8K, MATH); rewritten problems; concept-graph problems; web Q-A; competition pools | §2, §3 |
| Solutions per problem | 1 to 896 samples | §2 |
| Trace generator | teacher model; the model being trained; tree search | §2, §4, §5 |
| Trace style | short CoT; CoT plus programs; long CoT | §3, §4 |
| Acceptance check | final answer; code runs; Monte-Carlo value; second-model agreement | §2, §5, §6 |
| Use of wrong traces | discard; correction target; step-level negative | Negative samples section |

## §2 Sampling solutions at scale: OpenMathInstruct-1 and -2

**Definition.** Rejection-sampling distillation samples many solutions per training problem from a generator and keeps those whose final answer matches the reference. **Problem addressed.** When OpenMathInstruct-1 appeared (February 2024), large math SFT sets such as MetaMathQA and MAmmoTH were built from closed models with commercially restrictive licenses ([[openmathinstruct]], Abstract).

**Mechanism in OpenMathInstruct-1** ([[openmathinstruct]], NVIDIA, arXiv 2024-02):
1. Prompt Mixtral-base with an instruction and 5 examples in code-interpreter format (§2.1).
2. Sample at temperature 1.0, top-p 0.95; at most 3 code blocks; stop at the first execution error (§2.1).
3. Put the reference solution in the prompt with intermediate numbers masked (Mask-Text); this raised MATH training-set coverage from 80.1% to 85.9% at 224 samples (Table 2).
4. Keep solutions that reach the correct answer. Totals: MATH 896 samples per problem, 6,978 of 7,500 problems covered; GSM8K 256 samples, 7,469 of 7,473 covered (Table 2). The released set has 1.8M pairs; training uses about 1.02M after fair downsampling (GSM8K) and code-preferential selection (MATH) (§4, footnote 10).

OpenMath-Mistral-7B reaches 80.2 GSM8K and 44.5 MATH; on GSM-Hard, a numerically harder variant, it reaches 63.7, which the authors cite as limited robustness to perturbation (Table 3, Limitations). Correct-answer solutions with flawed reasoning are not filtered; the authors call them rare, anecdotally (§2.3).

**OpenMathInstruct-2** ([[openmathinstruct-2]], arXiv 2024-10) changes the generator to Llama3.1-405B-Instruct, adds 592K new questions, and runs SFT ablations with a Llama3.1-8B-Base student on a held-out 1K MATH validation split (§2.2):

| Ablation (MATH validation) | Result | Locus |
|---|---|---|
| Solution format | OpenMath CoT 44.5 vs Llama CoT 40.6; mean length 237.0 vs 331.3 tokens | Table 1 |
| Teacher at matched coverage | 405B-Instruct 37.9 ± 0.6 vs 8B-Base 30.1 ± 0.6 | Table 2 |
| Filtering low-quality solutions (128K) | 43.0–43.8 filtered vs 43.6 ± 1.7 unfiltered | Table 3 |
| Added wrong-answer or mispaired solutions | little to no loss up to 20% at ≥256K pairs | Fig. 5 |
| Unique questions at 256K pairs | 1K → 6.5K questions: +10.5 | §1, Fig. 6 |

**Worked example.** At 256K pairs, 1K unique questions means 256,000 / 1,000 = 256 solutions per question; 6.5K questions means 256,000 / 6,500 ≈ 39.4. The second set has fewer solutions per question and scores 10.5 points higher. At 20% noise, 51,200 of the 256K solutions are wrong, and accuracy changes little (Fig. 5). **Evidence status:** Result (single study): one student (Llama3.1-8B-Base), one domain, four runs per point (§2.2). **Limits.** The paper evaluates only math; the 70B model improves on a subset of benchmarks, which the authors attribute to choices tuned on 8B validation (§4, Interpretation). New questions get the majority vote of 32 solutions as their answer, with no minimum agreement (App. C.1, Table 9). **Implication.** Noise tolerance at ≤20% does not mean noise is harmless for behaviors outside the benchmark; it means that for math accuracy at this scale, problem coverage mattered more than solution filtering.

## §3 Problem-side diversity: MetaMath, MathScale, MAmmoTH, and perturbation tests

**Definition.** Problem-side augmentation creates new or rewritten problems instead of more solutions to the same problems. **Problem addressed.** More solutions to a fixed problem set stop helping: for LLaMA-2-7B, answer augmentation saturated near 80K samples at 59.6% GSM8K ([[metamath]] §4.5).

**MetaMath** ([[metamath]], arXiv 2023-09) generates with GPT-3.5-Turbo at temperature 0.7 and keeps paths whose answer equals the ground truth (§4.1, Eqs. 1–4):
1. **AnsAug**: new solutions to the original question.
2. **Rephrasing**: an 8-example prompt rewrites the question, which GPT-3.5-Turbo then solves.
3. **Self-Verification (SV)**: the question and answer become a declarative statement; one number is replaced by x; the model is asked for x.
4. **FOBAR**: one number is replaced by x and "If we know the answer to the above question is {a*}, what is the value of unknown variable x?" is appended.

Result: at an extra 20K samples, AnsAug added 0.1 GSM8K points while Rephrasing, FOBAR, and SV added 0.4, 2.3, and 2.6; diversity gain and accuracy gain have Pearson correlation 0.972 (§4.5). Training only on the GSM8K-derived subset raised MATH from 3.0 to 5.7, and MetaMath-7B scored 37.1 vs WizardMath-7B 31.5 on DROP numeric questions (Table 3; App. E.4 Table 10).

**MathScale** ([[mathscale]], arXiv 2024-03) extracts 2,018 topics and 8,892 knowledge points from about 20K seed questions and samples concept combinations by random walk:

`f_co(u, v) = log(w_uv + ε)`,  `p_uv = exp(f_co(u, v)) / Σ_{v′∈N(u)} exp(f_co(u, v′))` (Eqs. 1–2)

where `w_uv` is the number of seed questions containing both concepts u and v, `ε = 1e-5`, and `N(u)` is the set of neighbors of u. Worked example: if u has two neighbors with counts 3 and 1, the walk picks them with probability 3/4 and 1/4 (ε ignored). Halving knowledge points lowered the 25K-example macro average by 8.6% relative, halving seed questions by 2.9% (Table 6). No solution check is applied: GPT-4 judged 26% of 5K solutions wrong, and replacing them did not help (10.6/11.5 vs 10.2/11.1 micro/macro; Table 7).

**MAmmoTH** ([[mammoth]], arXiv 2023-09) mixes 13 sources with CoT and program-of-thought (PoT) rationales, a PoT rationale being a Python program whose printed output is the answer. On Llama-2 7B the 9-set average is 32.0 CoT-only, 41.0 PoT-only, 47.9 hybrid (Table 6). Its generality evidence is the contrast with dataset-specific SFT: WizardMath-70B scores 20.0 on AQuA vs 40.9 for Llama-2-70B (Table 3).

**Perturbation test.** When problem-side augmentation is used to reduce reliance on memorized surface forms, evaluate the trained model on perturbed problems, because a score on the fixed public test set does not separate the two. GSM-Symbolic ([[gsm-symbolic]], arXiv 2024-10) turns 100 GSM8K test questions into templates and samples 50 sets of 100 variants (§3.2). Gemma2-9b-it scores 87.0 on the originals and 79.1 ± 3.0 on variants, a gap of 7.9 points or 2.6 standard deviations (Fig. 2). Adding one irrelevant clause (GSM-NoOp) gives a reported accuracy drop of 17.5% for o1-preview and over 65% for Phi-3-mini (Fig. 8a). None of the augmentation papers above reports GSM-Symbolic results; this is an Open question for their data.

## §4 Long chain-of-thought from small curated sets: s1 and LIMO

**Definition.** Small-set long-CoT SFT fine-tunes an instruction-tuned model on about 1K long traces written by a reasoning model. Both papers below are SFT results, not RL results.

**s1** ([[s1]], arXiv 2025-01, v3): 59,029 questions from 16 sources get traces from Gemini 2.0 Flash Thinking Experimental (§2.1). Selection (§2.2):
1. Quality: remove API errors (54,116 left) and formatting problems (51,581 left).
2. Difficulty: remove questions that Qwen2.5-7B-Instruct or Qwen2.5-32B-Instruct solves, as graded by Claude 3.5 Sonnet (24,496 left).
3. Diversity: pick a Mathematics Subject Classification domain uniformly, then a question with probability favoring longer traces, until 1,000 questions span 50 domains.

The grader judges only 53.6% of s1K traces correct; incorrect traces are kept (§2.2). s1-32B scores 56.7 AIME24, 93.0 MATH500, 59.6 GPQA-Diamond with budget forcing (Table 1). **Budget forcing** is a decoding rule: to extend thinking, suppress the end-of-thinking delimiter and append "Wait"; to cap it, insert the delimiter (§3.1). Budget forcing raises AIME24 from 50.0 to 56.7 (Table 1), and the gain flattens out at six forced continuations (§4.2). Ablations under a ≈30,000-token cap (Table 2): 1K-random 36.7, 1K-diverse 26.7, 1K-longest 33.3, 59K-full 53.3, s1K 50.0 AIME24; the 59K run used 394 H100 GPU-hours vs 7.

**LIMO** ([[limo]], arXiv 2025-02; v3 2025-07, COLM 2025): tens of millions of problems are filtered by Qwen2.5-Math-7B-Instruct (drop if solved within 4 attempts) and DeepSeek-R1-Distill-Qwen-32B (keep if solved 1–3 times in 32), giving 2,125 problems (§3.1.1). Chains from three reasoning models are ranked by a rule-based score, and the top 800 form the set (§3.1.2). Results on Qwen2.5-32B-Instruct (Table 1):

| Benchmark (pass@1) | Base | NuminaMath-100k SFT | OpenThoughts-114k SFT | LIMO (800) |
|---|---|---|---|---|
| AIME24 | 16.5 | 6.5 | 50.2 | 63.3 |
| MATH500 | 79.4 | 59.2 | 80.6 | 95.6 |
| GPQA | 48.0 | 25.8 | 42.9 | 70.7 |
| 10-benchmark average | 49.9 | 32.3 | 58.3 | 78.1 |

**Base-model dependence.** The same 800 examples give 9.2 AIME24 on Qwen1.5-32B-Chat vs 63.3 on Qwen2.5-32B-Instruct, and 2.5 at 3B vs 68.3 at 72B in the Qwen2.5-Instruct series (§6.3.3–6.3.4). **Evaluation differences.** s1 reports Qwen2.5-32B-Instruct at 26.7 AIME24 (greedy) and LIMO reports 16.5 (4 samples at temperature 0.6) (s1 §4.1; LIMO §5): the same model differs by 10.2 points across protocols. **Status.** Replicated: two independent groups report that about 1K long traces raise Qwen2.5-32B-Instruct's AIME24 by more than 20 points under their own protocols (s1: 26.7 → 50.0 without budget forcing; LIMO: 16.5 → 63.3); the size of the gain depends on the data and the protocol.

**Distillation settings in this chapter's sources.** Stage: distill-SFT throughout.

| Source | Student | Teacher | Prompts | Sampling | Quality control | Size |
|---|---|---|---|---|---|---|
| [[openmathinstruct]] | Mistral-7B to CodeLlama-70B | Mixtral-base, 5-shot | GSM8K, MATH train | T 1.0, top-p 0.95; 256 / 896 per problem | final answer | 1.8M (1.02M used) |
| [[openmathinstruct-2]] | Llama3.1-8B/70B-Base | Llama3.1-405B-Instruct | train + 592K new | new questions: 32 at T 0.7 | final answer (train questions); majority answer (new questions); LLM decontamination | 14M |
| [[metamath]] | LLaMA-2 7B–70B | GPT-3.5-Turbo | GSM8K, MATH rewrites | T 0.7 | ground-truth answer | 395K |
| [[s1]] | Qwen2.5-32B-Instruct | Gemini 2.0 Flash Thinking | 59K pool | not reported | none on correctness (53.6% correct) | 1K |
| [[limo]] v3 | Qwen2.5-32B-Instruct | R1, R1-Distill-Qwen-32B, QwQ-32B | 2,125 hard problems | not reported | rule-based chain score | 800 |
| [[transferability-of-llm-reasoning]] | Qwen3-14B-Base | Qwen3-32B | 47K math problems | not reported | correct answer | not reported |

**Implication.** Both small-set results are math-heavy and depend on a base that already solves part of the task. OpenThoughts ablations show why the trace style matters: removing self-reflection phrases cut average trace length from 11,593 to 328 tokens and the average score from 51.4 to 26.3 ([[open-thoughts]], App. H.3, Table 22).

## §5 Search-based trace generation: rStar and rStar-Math

**rStar** ([[rstar]], arXiv 2024-08) is inference-time search without fine-tuning. A generator runs Monte Carlo Tree Search (MCTS), a search that repeatedly selects a path, expands a node, simulates to an answer, and back-propagates the result. Each step is produced under one of five actions (§3.2): A1 one-step thought; A2 the remaining steps; A3 the next sub-question with its answer; A4 re-answer the sub-question with few-shot CoT; A5 rephrase the question by listing its conditions. The terminal node's reward is the confidence of self-consistency majority voting, added to every node on the path; no self-rewarding of intermediate nodes, external tools, or trained value models are used (§3.2). A discriminator, Phi3-mini-4k (3.8B), receives the first 20–80% of a trajectory and completes it; trajectories whose answers agree are kept (§3.3, §4.1). With 32 rollouts, LLaMA2-7B goes from 12.51 (few-shot CoT) to 63.91 GSM8K and from 58.82 to 67.25 on StrategyQA, a commonsense task (Table 2). On 200 GSM8K questions for LLaMA3-8B, A3 alone gives 70.5 and all five actions 75.0 (Table 1).

**rStar-Math** ([[rstar-math]], arXiv 2025-01) turns search into training data for a 7B policy and a 7B process preference model (PPM):
1. Each step is a natural-language comment plus Python; only candidates whose code executes, together with all previous steps, become nodes (§3.2).
2. Rounds 1–2 use terminal-guided annotation: `q(s_i)^k = q(s_i)^{k−1} + q(s_d)^k` (Eq. 2), where `q(s_i)^k` is the value of step `s_i` after rollout k, `q(s_d)^k` is +1 if the terminal answer matches the ground truth and −1 otherwise, and `q(s_i)^0 = 0`.
3. Selection uses `UCT(s) = Q(s) + c·sqrt(ln N_parent(s) / N(s))`, `Q(s) = q(s)/N(s)` (Eq. 1), where `N(s)` is the visit count of s, `N_parent(s)` that of its parent, and `c = 2` (App. A.1).
4. SFT data: the top-2 correct trajectories per problem by average Q; the policy is re-trained from the base model each round (§3.4.1, App. A.1).
5. PPM data: at each step, the two highest-Q candidates that lead to correct answers vs the two lowest-Q candidates that lead to wrong answers, trained with `L_ppm = −(1/(2×2)) E[log σ(r_θ(x, y_i^pos) − r_θ(x, y_i^neg))]` (Eq. 4), where `r_θ` is the PPM score, `x` the problem, and `y_i` the trajectory up to step i.
6. Round 1 uses DeepSeek-Coder-V2-Instruct (236B) with 8 rollouts; rounds 2–4 use the 7B models with 16 rollouts; from round 3 the PPM supplies initial q values (§3.2, §3.4.2).

**Worked example.** A step visited in 4 rollouts with outcomes +1, +1, −1, +1 has q = 2 and Q = 0.5. With `N_parent = 10` and `c = 2`, UCT = 0.5 + 2·sqrt(ln 10 / 4) = 0.5 + 2·0.759 = 2.017. [figures/rstar-mcts.html](figures/rstar-mcts.html) Panel A runs this annotation on a 16-leaf tree, so the reader can step through rollouts and see which trajectories become SFT data and which step-1 candidates form PPM pairs.

**Evidence.** Qwen2.5-Math-7B fine-tuned on round-4 data reaches 78.4 MATH vs 73.4 for rejection sampling with an ORM and 72.4 for random self-samples from the same policy (Table 7). Problem coverage rises from 60.17% to 90.25% of 747K problems over four rounds (Table 2). The 90.0 MATH / 53.3 AIME24 reported in §4.2 requires 64 trajectories scored by the PPM; the greedy policy scores 78.4 / 26.7, below Qwen2.5-Math-7B-Instruct on MATH (82.6) (Tables 5, 10). A PQM trained with MSE on Q-values scores 88.2 vs 89.4 for the PPM (Table 8). **Limits.** Word problems only; code or general tasks would need test cases, human labels, or mutual verification (§5). Of 20 unsolved problems sampled after round 4, 19 had wrong reference answers (§3.4.2).

## §6 Step-level data: Monte-Carlo labels and step preferences

**Monte-Carlo step label.** For question q and prefix `x_{1:t}`, a completer samples k rollouts: `c_t = (correct rollouts from step t) / (total rollouts from step t)` ([[omegaprm]] Eq. 1), where a rollout is correct if its final answer equals the reference. Math-Shepherd's hard label is `1[c_t > 0]` ([[math-shepherd]] §3.3).

**OmegaPRM** ([[omegaprm]], arXiv 2024-06, v2) locates the first error by binary search: roll out from the midpoint m; if `c_m > 0`, the error is in the second half, else in the first; repeat until one step remains (§3.2). With M steps, the cost is O(k log M) policy calls instead of O(kM). Worked example with the paper's settings, k = 8 and 16 pieces per solution: per-step estimation needs 8 × 16 = 128 rollouts per solution, binary search 8 × log₂16 = 32 (derived). The PRM trains on 1.5M soft-label annotations from 12K MATH training questions; with PRM-weighted majority voting over 64 samples, Gemini Pro reaches 69.4 on MATH500 vs 67.2 for majority voting and 67.6 with a PRM800K-trained PRM (Table 1).

**Step-DPO** ([[step-dpo]], arXiv 2024-06) builds pairs at the first wrong step (§3.2):
1. Sample step-formatted answers ("Step i:") from the reference model; keep those with a wrong final answer.
2. Find the first wrong step `s_k` manually or with GPT-4.
3. Sample continuations from the reference model given `s_{1:k−1}`; keep those reaching the gold answer; take their first step as `s_win`.

`L(θ) = −E[log σ(β log(π_θ(s_win | x; s_{1:k−1}) / π_ref(s_win | x; s_{1:k−1})) − β log(π_θ(s_lose | x; s_{1:k−1}) / π_ref(s_lose | x; s_{1:k−1})))]` (Eq. 2)

where `π_θ` is the trained policy, `π_ref` the frozen reference, `σ` the sigmoid, `s_lose = s_k`, and `β` sets the distance from the reference (0.4, or 0.5 at 72B; §4.1). On 10K pairs, gains over SFT models are +1.0 (Qwen2-7B) to +3.0 (Qwen2-72B) MATH (Table 1).

## §7 Transfer beyond math: SFT versus RL on the same data, and instruction following

**Problem.** Most trace datasets in this chapter are math, while a general-purpose model is also evaluated on code, science, instruction following, and conversational question answering. **Evidence 1** ([[transferability-of-llm-reasoning]], arXiv 2025-07, v2): Qwen3-14B-Base trained on 47K math problems with SFT on Qwen3-32B traces or with GRPO on answer correctness (App. A.3):

| Qwen3-14B-Base variant | Math avg | Other reasoning avg | Non-reasoning avg | IFEval |
|---|---|---|---|---|
| Base | 27.7 | 30.2 | 45.7 | 69.2 |
| SFT, thinking traces | 49.8 | 45.3 | 21.1 | 42.3 |
| SFT, non-thinking traces | 32.3 | 45.2 | 29.0 | 41.4 |
| RL (GRPO) | 53.8 | 60.0 | 53.2 | 70.0 |

Table 1. The ablation on Qwen3-8B-Base separates the sampling distribution: off-policy SFT on teacher traces gives a non-reasoning transferability index of −40.5, on-policy SFT (rejection sampling from the model itself) +30.2, and on-policy RL +32.4 (Table 4). **Limits.** The SFT learning rate is 5×10⁻⁵ and the RL learning rate 1×10⁻⁶ (App. A.3.1), so update size is not matched; the authors do not ablate it. **Interpretation.** Self-generated traces (rStar-Math rounds 2–4, Step-DPO's in-distribution chosen steps) are closer to on-policy data than teacher traces; neither paper measures transfer outside math.

**Evidence 2** ([[scaling-reasoning-losing-control-mathif]], arXiv 2025-05): MathIF measures hard accuracy (all constraints met) and soft accuracy (share met) on 420 math problems with Python-checked constraints. Worked example: a query with 3 constraints of which 2 are met has hard accuracy 0 and soft accuracy 2/3 (Eq. 1). In controlled runs on four Qwen2.5 bases, long-CoT SFT and GRPO lowered both scores in 15 of 16 variants while raising math accuracy in 15 of 16 (Table 4, counted from the table); Qwen2.5-7B +SFT went from 15.95 to 7.86 hard accuracy and from 13.59 to 23.10 math accuracy. Among released models, s1-32B has 58.04 IFEval prompt-level strict vs 80.96 for Qwen2.5-32B-Instruct (App. G, Table 13; not a controlled comparison). Raising the RL rollout cap from 1k to 8k tokens raised math accuracy from 28.73 to 39.82 and lowered hard accuracy from 19.05 to 14.29 (Table 5). **Status.** Replicated in direction for SFT-induced instruction-following loss (Huan et al. Table 1; Fu et al., the MathIF authors, Table 4), with different bases and data. For RL the two studies disagree: GRPO kept IFEval at 70.0 vs 69.2 in Huan et al., while cold-start GRPO lowered MathIF hard accuracy on all four Qwen2.5 bases (for example 15.95 → 10.48 on Qwen2.5-7B, Table 4); benchmarks, bases, and training data differ, so the cause is an Open question.

**Evidence 3** ([[open-thoughts]], App. L, Table 30): OpenThinker3-7B, trained on 1.2M math, code, and science traces without safety data, has a HarmBench harmfulness rate of 55.5 vs 14.5 for Qwen2.5-7B-Instruct.

## §8 Trace length versus the SFT sequence length

**Problem.** Long-CoT targets can be longer than the SFT maximum sequence length. The tokens removed by truncation are the final ones: the answer, the end-of-answer delimiter, and the stop token.

**Mechanism.**
1. With prompt length `L_p`, target length `L_t`, and maximum length `L_max`, the fraction of target tokens trained is `min(1, (L_max − L_p) / L_t)`.
2. If `L_p + L_t > L_max`, no training token teaches the model to emit the answer or stop after this trace.
3. The s1 authors report the consequence: with a shorter training length, "the answer section of the training sample is more commonly cut off", the answer gets fewer gradient updates, and reasoning at test time becomes longer ([[s1]] App. D.1).

**Worked example.** A 300-token prompt and a 12,000-token trace at `L_max = 4,096`: (4,096 − 300) / 12,000 = 31.6% of target tokens are trained; the last 8,204, including the answer and stop token, are removed.

**Evidence.** s1 on Qwen2.5-32B-Instruct (Table 8): training length 4,096 cut 74% of samples and gave AIME24 30.0% with 20,721 average thinking tokens; 32,768 cut 0% and gave 50.0% with 6,984 tokens (2.97× shorter). LIMO sets its length limit so that all responses, all under 16,384 tokens, fit ([[limo]] §4). MathIF's controlled SFT drops traces over 8,192 tokens instead ([[scaling-reasoning-losing-control-mathif]] §5.2). The s1 authors state that the context window limits further scaling with budget forcing; in a separate run where the model was prompted to use up to 512 steps, 12 of 30 AIME24 responses exceeded the context window ([[s1]] §6.2).

**Long-context ability before reasoning SFT** ([[longer-context-deeper-thinking]], arXiv 2025-05): LLaMA3-8B-Instruct variants differing only in RoPE θ scale were fine-tuned on identical OpenR1-Math-220K splits (≤8K and 8K–16K tokens). θ×16, which had the best 32K needle-in-a-haystack score (77.05 vs 0.00 at ×1), gave MATH500 59.36 vs 54.80 after SFT (average of the short and long runs; Tables 2–3), GPQA 41.92 vs 37.27 after science SFT, and MMLU-STEM 74.27 vs 71.06 (Tables 4, 6). For Qwen2.5-Math-7B-Instruct (4K context), long-trace SFT scored below short-trace SFT (83.80 vs 86.28) until the context was extended (89.12 vs 88.28) (Table 7). **Limits.** SFT only, 7B–14B, number of runs not stated (§5; recipe ledger). **Implication.** The two results differ: on LLaMA3-8B-Instruct the 8K–16K split scored above the ≤8K split at every θ factor, including ×1, whose effective length the paper estimates at 9K without describing the method (Tables 2–3), while on Qwen2.5-Math-7B-Instruct the long split scored below the short one until the context was extended (Table 7). When the planned traces are longer than the model's measured effective context, extend the context before long-trace SFT and compare the long and short splits, because in both studies the extended models scored higher after SFT (Tables 3, 7). Otherwise, choose trace length by the same split comparison (ch-32b, ch-32c).

## Negative samples and negative feedback

**Four meanings used here** (course standard): (1) negative marginal value, a sample that hurts when used as a positive target; (2) negative as content, a failure placed in the input or in a corrected target and trained with cross-entropy; (3) negative as conditioning, a failure trained under a control token; (4) negative as gradient, an explicit decrease of the sample's likelihood. Only (4) removes probability mass from the sample.

**1. Where negatives come from in trace synthesis.** Wrong final answers from rejection sampling ([[openmathinstruct]] releases 6.6M of them for verifier training and does not use them, §1); the first wrong step located manually or by GPT-4 ([[step-dpo]] §3.2); Monte-Carlo labels with `c_t = 0` ([[omegaprm]] §3.2); low-Q search candidates leading to wrong answers ([[rstar-math]] §3.3); GPT-4 error explanations ([[lema-learning-from-mistakes]] §2.1). Label noise: OmegaPRM notes that too-hard questions create false negatives and filters questions with 0 correct or 0 wrong answers in 32 rollouts, stating that the effect of the remaining noise is uncertain (App. A, §5). Math-Shepherd hard labels are 86% accurate against 160 human-labeled steps at N = 4 and less accurate at larger N because of false positives ([[math-shepherd]] §5.2).

**2. What current practice does.** Discard (type 1): OpenMathInstruct, MetaMath, and MAmmoTH; s1 applies no correctness filter. Content (type 2): LEMA trains on (question + wrong path) → (wrong step, explanation, corrected solution), with loss on the correction only (App. B.2). Conditioning (type 3): no source in this chapter; see ch-31a. Gradient (type 4): Step-DPO's rejected step. Reward-model labels: OmegaPRM and the rStar-Math PPM use negatives as classifier or ranking labels, not as policy gradients.

**3. Mechanism.** For a softmax over next tokens with logits z, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. Gradient descent on `log p_y` with step η changes `z_y` by `−η(1 − p_y)` and every other `z_j` by `+η p_j`, so the most probable alternative receives the largest logit increase. Worked example: probabilities (a, b, y) = (0.80, 0.15, 0.05) and η = 1 give logit changes (+0.80, +0.15, −0.95) and new probabilities (0.902, 0.088, 0.010): token b, which was not penalized, also loses mass, and a gains 0.102 (computed). Panel C of [figures/rstar-mcts.html](figures/rstar-mcts.html) recomputes this for any inputs and counts the penalized steps for any answer length and first-error position. **Credit assignment.** When the chosen and rejected answers are sampled independently, every step of the rejected answer appears only in the rejected term; a 10-step answer whose first error is step 7 contributes 6 correct steps, the wrong step, and 3 steps that follow from it to the negative gradient. If whole-answer DPO used a chosen and a rejected answer that share the prefix `s_{1:k−1}`, the prefix log-ratio terms would be identical in both and cancel in the margin (derived); Step-DPO places the prefix in the conditioning and drops the continuation after `s_k`, so only `s_k` receives the negative gradient. Step-DPO's stated reason matches this: rejecting a whole answer "may also discard preceding correct reasoning steps" (§3.1). **Similarity risk.** Chosen and rejected steps conditioned on the same prefix are expected to have similar hidden embeddings (Interpretation; not measured for Step-DPO). In [[likelihood-displacement]], pairs with a higher centered hidden-embedding similarity (CHES) score lowered the chosen response's log probability more (§4.2.2 theory, §5 Fig. 2). That study used chat and refusal pairs, not reasoning steps, so the risk for Step-DPO pairs is an Open question; Step-DPO does not report chosen-step log probabilities.

**4. Evidence with numbers.** Benefit: Step-DPO vs DPO at 5K pairs, 55.8 vs 55.0 (Qwen2-7B-SFT) and 64.1 vs 62.5 (Qwen2-72B-SFT) (Table 3). In-distribution chosen steps beat GPT-4 corrections, 55.8 vs 55.1 (Table 4); the authors attribute this to the low reference log probability of out-of-distribution text (§3.2, Interpretation). LEMA adds +1.5 to +2.9 GSM8K across five models (Table 1), keeps a gain at matched size for 4 of 5 models (not LLaMA-2-7B) and at matched tokens (LLaMA-2-70B 82.1 → 83.5) (Fig. 4, Table 2). On LLaMA-2-7B, SFT on 7,473 GPT-3.5-Turbo paths with wrong final answers gave 43.6 GSM8K, vs 41.6 for the 7,473 original training solutions and 52.2 for 7,473 correct-answer paths ([[metamath]] §4.7, Table 4); the authors hypothesize that some intermediate steps in these paths are correct (Interpretation). Step-DPO and LEMA report single numbers without run counts or variance. **Size of effect.** No source here isolates the share of improvement due to negatives vs positives for trace data; Huan et al. name negative gradients as one of several factors but ablate them jointly with credit assignment ([[transferability-of-llm-reasoning]] §5.2).

**5. Controls.** Localize to the first wrong step (Step-DPO, OmegaPRM). Take chosen steps from the model's own samples (Step-DPO Table 4). Drop questions whose Monte-Carlo outcomes are all correct or all wrong (OmegaPRM App. A), and drop synthetic problems whose trajectories are below 50% accurate, which rStar-Math uses against wrong reference answers (App. A.1). Train only on the pairs with the lowest length-normalized CHES score, as in [[likelihood-displacement]] §6.3, which kept 5% of pairs (tested on refusal data only). Use failures as correction content when a gradient is not needed (LEMA).

**6. Diagnostics.** Log chosen and rejected log probabilities separately; Step-DPO's Fig. 2 tracks the reward margin, which plateaus for DPO. Report the fraction of steps from correct solutions that receive `c_t = 0`; Panel B of the figure computes the false-negative probability `(1 − p)^k` for a completer that reaches the answer with probability p per rollout (p = 0.10, k = 8: 0.430; k = 32: 0.034). Track pass@k: rStar-Math reports pass@N curves for its policies (Figs. 6–7).

**7. Effect on generality.** LEMA's gains extend to SVAMP and ASDiv, which the authors treat as out-of-distribution because training used GSM8K (§4.1); CSQA also improves (+1.1 at 70B), but with CSQA's own correction data (§3.2), so it is not a transfer result. No source in this chapter reports instruction-following, calibration, or refusal changes from step-level negatives; this is an Open question (see ch-43a for negative gradients in RL).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| s1-32B (Qwen2.5-32B-Instruct) | 32B | distill-SFT | examples; trace source | 1,000; Gemini 2.0 Flash Thinking Experimental | arXiv:2501.19393v3 §2.1–2.2 | verified 2026-09-15 | Table 2: s1K vs 1K-random/diverse/longest and 59K-full, 95% bootstrap intervals |
| s1-32B | 32B | distill-SFT | epochs; batch; steps; peak LR; warmup; decay | 5; 16; 315; 1e-5; linear 5% (16 steps); cosine to 0 | v3 App. D | verified 2026-09-15 | no ablation reported |
| s1-32B | 32B | distill-SFT | optimizer; weight decay; precision; loss masking | AdamW β1 0.9, β2 0.95; 1e-4; bfloat16; loss on reasoning traces and solutions only | v3 App. D | verified 2026-09-15 | no ablation reported |
| s1-32B | 32B | distill-SFT | training sequence length | "large enough to avoid cutting off any samples" | v3 App. D, D.1 | verified 2026-09-15 | Table 8: 4,096 (74% cut) AIME24 30.0 vs 32,768 (0% cut) 50.0; runs per setting not stated |
| s1-32B | 32B | distill-SFT | compute | 16 H100, 26 minutes (7 H100 GPU-hours) | v3 §4.1, §5.1 | verified 2026-09-15 | 59K-full: 394 H100 GPU-hours |
| s1-32B | 32B | eval-gate | decoding; forcing | temperature 0; "Wait" appended at attempted end of thinking | v3 §3.1, §4.1 | verified 2026-09-15 | Table 4: 2× "Wait" 53.3 vs no extension 50.0 AIME24 |
| LIMO (Qwen2.5-32B-Instruct), arXiv v3 | 32B | distill-SFT | examples; LR; schedule; warmup; epochs; batch | 800; 5.0e-6; cosine; none; 15; 64 | arXiv:2502.03387v3 §3.1.2, §4 | verified 2026-09-15 | §6.3.5: 400/800/1.2k/1.6k/2k subsets; 800 → 1.2k +0.9 AIME24 |
| LIMO v3 | 32B | distill-SFT | sequence length | limit set so all responses fit; responses under 16,384 tokens | v3 §4; v1 §4.1 ("16,384 tokens") | verified 2026-09-15 | no ablation reported |
| LIMO v1 vs v3 | 32B | distill-SFT | examples; AIME24; MATH500 | v1: 817, 57.1, 94.8; v3: 800, 63.3, 95.6 | v1 Abstract; v3 Abstract, Table 1 | conflict (v3 is the COLM 2025 version) | no ablation reported |
| OpenMath2-Llama3.1-8B | 8B | distill-SFT | data; epochs; batch; LR; optimizer | 13.97M pairs; 2; 512 (unit not printed); constant 2e-5; AdamW, weight decay 1e-2 | arXiv:2410.01560v2 §4; [[openmathinstruct-2-recipe]] | verified 2026-09-14 | Fig. 1: gains at 1M, 2M, 5M, full |
| OpenMathInstruct-2 ablation student (Llama3.1-8B-Base) | 8B | distill-SFT | epochs; batch; LR; runs | 4; 256; constant 5e-6; 4 runs averaged | v2 §2.2 | verified 2026-09-14 | no ablation reported |
| OpenMathInstruct-2 new questions (Llama3.1-405B-Instruct) | 405B | distill-SFT | solutions per question; temperature; answer rule | 32; 0.7; majority vote, minimum vote threshold 0 | v2 §3, App. C.1 | verified 2026-09-14 | Table 9: thresholds 0/8/16/24 → 50.1/49.2/44.4/42.0 (sizes not matched) |
| OpenMathInstruct-2 | — | distill-SFT | length filter | drop solutions > 1,024 Llama3.1 tokens or < 200 characters | v2 App. A.2 | verified 2026-09-14 | no ablation reported |
| MetaMath-7B (LLaMA-2-7B) | 7B | distill-SFT | epochs; batch; LR; warmup | 3; 128; 2e-5; 3% | arXiv:2309.12284v4 App. B | verified 2026-09-14 | no ablation reported (Table 3 compares data operators, not these settings) |
| MAmmoTH-7B (Llama-2 7B) | 7B | SFT | LR; batch; schedule; epochs | 2e-5; 128; cosine, 3% warmup; 3 | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | no ablation reported (Table 6 compares rationale formats, not these settings) |
| rStar-Math policy (Qwen2.5-Math-7B) | 7B | SFT | data rule; epochs; sequence length; batch; LR | top-2 correct trajectories by average Q; 2; 4,096; 128; 7e-6, linear, AdamW; re-trained from base each round | arXiv:2501.04519v1 §3.4.1, App. A.1 | verified 2026-09-15 | data rule: Table 7, 78.4 vs rejection sampling 73.4 MATH; re-training from base: "based on our extensive experiments" (App. A.1), numbers not reported; other settings: no ablation reported |
| rStar-Math PPM | 7B | reward-model | pairs; loss; epochs; batch; LR | 2 positive × 2 negative per step; pairwise Bradley–Terry; 1; 512; 7e-6 | v1 §3.3, App. A.1 | verified 2026-09-15 | Table 8: PPM 89.4 vs PQM (MSE) 88.2 vs ORM 82.6 MATH |
| rStar-Math data, rounds 2–4 | 7B policy | SFT | rollouts; depth; candidates per step; c | 16 per problem (+16 for hard; round 4 up to 128); 16; 8 (round 4: 16); 2 | v1 §3.4, App. A.1 | verified 2026-09-15 | Table 2: coverage 60.17% → 90.25% |
| Step-DPO SFT stage (Qwen2-7B) | 7B | SFT | data; epochs; batch; LR; schedule | 299K; 3; 256; 5e-6; linear decay, warmup ratio 0.03 | arXiv:2406.18629v1 §4.1 | verified 2026-09-15 | no ablation reported |
| Step-DPO (Qwen2-7B-SFT; >30B models) | 7B; >30B | preference | pairs; epochs; batch; LR; β; schedule | ~10K; 8 (7B), 4 (>30B); 128; 5e-7; 0.4 (0.5 at 72B); cosine, warmup 0.1 | v1 §4.1 | verified 2026-09-15 | no ablation of β, LR, or epochs reported; Table 3 compares the loss: 55.8 vs DPO 55.0 at 5K pairs |
| OmegaPRM data (Gemini Pro, Gemma2 27B policies) | — | reward-model | k; search limit; α, β, L; c_puct; question filter | 8; 100; 0.5, 0.9, 500; 0.125; 32 rollouts, drop if 0 correct or 0 wrong | arXiv:2406.06592v2 §4, App. A | verified 2026-09-14 | Table 2: soft labels 70.1% vs hard 63.3% step accuracy |
| LEMA (LLaMA-2-70B) | 70B | SFT (correction content) | data; method; LR; batch; steps; checkpoint rule | ~32K CoT + 12,523 GSM8K corrections; QLoRA r 64, dropout 0.05; 1e-4; 96; 2,000; best of checkpoints every 100 steps on the test set | arXiv:2310.20689v4 §3.2–3.3 | verified 2026-09-15 | data: Fig. 4 and Table 2 compare with CoT-only at matched size and tokens; LR, batch, steps: no ablation reported |
| UniReason-Qwen3-14B (SFT) | 14B | distill-SFT | data; LR; batch; epochs | 47K math problems, Qwen3-32B traces; 5e-5; 512; 1.5 | arXiv:2507.00432v2 App. A.3 | verified 2026-09-15 | no ablation reported; 1.5 epochs chosen "to align with our RL settings" (App. A.3.1) |
| UniReason-Qwen3-14B (RL) | 14B | RL | algorithm; LR; batch; rollouts per prompt; max length; clip; KL; steps | GRPO (verl); 1e-6; 512; 16; 16k tokens; "between 0.22 and 0.28"; 0 (entropy 0); 140 | v2 App. A.3.1 | verified 2026-09-15 | no ablation reported; KL on vs off compared only on Qwen3-8B (Table 4) |
| LLaMA3-8B-Instruct RoPE θ×16 | 8B | long-context, then distill-SFT | extension; SFT data; batch; LR; epochs | θ×16; OpenR1-Math-220K ≤8K or 8K–16K splits, 20K each; 32; 1.0e-5; 3 | arXiv:2505.17315v2 §3.1–3.2 | verified 2026-09-14 | Table 3: MATH500 avg 59.36 (×16) vs 54.80 (×1) |

Status "verified 2026-09-15" rows were read in the primary arXiv PDF on that date because the library cards for [[s1]], [[limo]], [[rstar-math]], [[step-dpo]], and [[transferability-of-llm-reasoning]] had not been verified; the chapter excerpts hold the checked extracts.

**Starting point for a small general-purpose run.** For long-CoT distillation SFT of an instruction-tuned model on about 1K traces, the verified reference is s1-32B: 5 epochs, batch 16, peak LR 1e-5 with 5% linear warmup and cosine decay to 0, AdamW (0.9, 0.95), weight decay 1e-4, loss on traces and answers only, and a sequence length that cuts no sample (32,768 in its ablation). Those values were used for Qwen2.5-32B-Instruct on 16 H100 GPUs, and only the sequence length was ablated. s1 reports no instruction-following evaluation (its non-math check is GPQA Diamond), and s1-32B has 58.04 IFEval vs 80.96 for its base in MathIF's measurement, so the same run needs the regression checks in the Generalization lens.

## Generalization lens

**(a) What increases breadth.** The first three items measure accuracy within math (several math benchmarks or held-out math sets); items marked "outside math" measure other domains.
- More distinct problems at fixed size: +10.5 MATH validation from 1K to 6.5K questions ([[openmathinstruct-2]] Fig. 6); knowledge-point coverage −8.6% relative when halved ([[mathscale]] Table 6); diversity gain correlated 0.972 with accuracy gain ([[metamath]] §4.5).
- Several sources and rationale formats: MAmmoTH hybrid 47.9 vs 32.0 CoT-only on a 9-set math average including 5 out-of-domain math sets ([[mammoth]] Table 6).
- Self-generated chosen steps: in-distribution 55.8 vs GPT-4-corrected 55.1 MATH ([[step-dpo]] Table 4; math only).
- Science data in a mixture (outside math): a 31K science set gives 48.8 GPQA-Diamond and two 31K code sets give 47.3 and 36.7; each code set mixed with the science set (62K) gives 52.7, so the code-to-science transfer difference disappears once in-domain data is added ([[open-thoughts]] App. H.5, Table 24).
- On-policy data (outside math): on-policy SFT non-reasoning transferability index +30.2 vs off-policy −40.5 on Qwen3-8B-Base ([[transferability-of-llm-reasoning]] Table 4).
- Effective long context before long-trace SFT (outside math): GPQA 41.92 vs 37.27 after science SFT and MMLU-STEM 74.27 vs 71.06 ([[longer-context-deeper-thinking]] Tables 4, 6).
- Reasoning data in pretraining (outside math in part): 8B models with reasoning data in pretraining stayed ahead after SFT, 35.92 vs 26.62 average over math, science, code, and instruction-following evaluations ([[front-loading-reasoning]] Table 2).

**(b) What causes narrowing or forgetting.**
- Math-only SFT on teacher traces: IFEval 69.2 → 42.3 and non-reasoning average 45.7 → 21.1 on Qwen3-14B-Base ([[transferability-of-llm-reasoning]] Table 1).
- Long-CoT SFT and reasoning RL: lower constraint compliance in 15 of 16 variants ([[scaling-reasoning-losing-control-mathif]] Table 4); longer rollouts lower hard accuracy (Table 5).
- Dataset-specific math SFT: WizardMath-70B AQuA 20.0 vs 40.9 for its base ([[mammoth]] Table 3).
- Safety behavior not covered by trace data: HarmBench 14.5 → 55.5 ([[open-thoughts]] Table 30).
- Tool coupling: OpenMathInstruct-1 models are evaluated with code execution in the loop ([[openmathinstruct]] §3.2); deployment without the executor was not evaluated.

**(c) How to measure it for this stage.**
- Perturbed problems: templated variants and irrelevant clauses ([[gsm-symbolic]] Figs. 2, 8); report the distribution over sets, not one score.
- Post-cutoff problems: s1's Table 5 lists s1 without budget forcing at 50.0 on AIME 2024 and 26.7 on AIME 2025, with AIME 2025 values attributed to Ye et al. (2025b) in the caption ([[s1]] App. A); decontaminate synthetic questions against test sets with paraphrase checks ([[openmathinstruct-2]] §3.1).
- A before/after suite outside math: IFEval or MathIF, a non-math reasoning set (GPQA-Diamond, LiveCodeBench), and non-reasoning sets (CoQA, HaluEval), because these are where [[transferability-of-llm-reasoning]] and [[scaling-reasoning-losing-control-mathif]] found regressions; add a safety set when trace data has no safety content.
- Protocol control: the same model differs by 10.2 AIME24 points between greedy and 4-sample evaluation ([[s1]] Table 1; [[limo]] Table 1); vLLM batch size changes greedy results ([[s1]] App. B). Reporting the best checkpoint on the test set, as LEMA does for both arms (§3.3), can inflate reported scores (Interpretation).
- Search-time vs policy scores: report the policy's greedy score next to any score that uses search or a reward model ([[rstar-math]] Tables 5, 10).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| SFT sequence length shorter than traces | longer test-time traces and lower accuracy (s1 at 4,096: 20,721 vs 6,984 thinking tokens, 30.0 vs 50.0 AIME24) | share of examples with `L_p + L_t > L_max`; count training targets that still end with the stop token after truncation ([[s1]] Table 8) |
| Spending the budget on more solutions per problem | accuracy stops rising while pair count grows | unique-question count per 100K pairs; ablate questions at fixed pairs ([[openmathinstruct-2]] Fig. 6) |
| Treating final-answer match as step correctness | flawed intermediate steps in accepted traces | audit a sample by step; compare with code-verified or PRM-scored steps ([[rstar-math]] §3.2) |
| Whole-answer DPO on independently sampled reasoning pairs | reward margin plateaus; correct prefixes penalized | log chosen and rejected log probabilities; compare with first-wrong-step pairs ([[step-dpo]] Fig. 2) |
| Teacher-written corrections as chosen steps | low reference log probability of chosen steps | histogram of `log π_ref(s_win)` for teacher vs self-sampled steps ([[step-dpo]] Table 4) |
| Monte-Carlo labels from a weak completer with few rollouts | `c_t = 0` labels on steps taken from verified-correct solutions | false-negative rate on steps from verified solutions; question filter ([[omegaprm]] App. A) |
| Reporting only math benchmarks after trace SFT | instruction-following or non-math regressions found later | before/after IFEval, GPQA, non-reasoning set ([[transferability-of-llm-reasoning]] Table 1) |
| Crediting a small curated set without a base-model control | recipe fails on another base | run the same data on a weaker base ([[limo]] §6.3.3) |
| Quoting search-time scores as model scores | deployed greedy model scores lower | report greedy policy score with trajectory count and reward model ([[rstar-math]] Table 10) |
| Evaluating only on fixed public test sets | score above the templated-variant distribution | GSM-Symbolic-style variants, post-cutoff sets ([[gsm-symbolic]] §4.1) |
| Extending thinking past the context window | repetition loops after repeated forcing; responses that exceed the context window | track forced continuations and the count of responses over the window ([[s1]] §4.2, §6.2) |

## Check your understanding

1. At a fixed 256K pairs, why can 6.5K questions with 39 solutions each train a better student than 1K questions with 256 each, even though the second set has more verified solutions per problem?
2. OpenMathInstruct-2 finds little loss from 20% wrong solutions, yet Step-DPO gains from penalizing single wrong steps. Explain why both results can hold, using the difference between negative marginal value and negative as gradient.
3. Using `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, explain why pushing down a rejected step whose first token is already unlikely mainly raises the most likely alternative, and what that implies for similar chosen and rejected steps.
4. Why does truncating 12,000-token traces at 4,096 tokens make the trained model think longer at test time, according to the s1 authors' mechanism?
5. In the Qwen3-14B study, SFT and RL used the same 47K problems but SFT lowered IFEval. Which differences between the two runs could cause this, and which one does the Qwen3-8B ablation isolate?
6. rStar-Math's 90.0 MATH and its policy's 78.4 MATH come from the same training. Which components produce the difference, and which number should be compared with an SFT-only model?
7. Why do Monte-Carlo hard labels gain false negatives when the completer is weak and false positives when the number of rollouts grows?

## Connections

- Previous: ch-23 — Model Collapse and Verification of Synthetic Data. The verification and self-training risks there apply to self-generated traces in §5 and on-policy data in §7.
- Next: ch-25 — Multi-Turn Conversation Synthesis.
- ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage (teacher-trace distillation beyond math).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection (the diversity measures used in §3).
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation; ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation (§8).
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity (§7).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; ch-44 — Process Supervision and Verifiable Rewards; ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores.

## Sources

- [[openmathinstruct]] — sampling counts, Mask-Text coverage, GSM-Hard robustness note, released wrong solutions.
- [[openmathinstruct-2]] — format, teacher, filtering, noise, and question-diversity ablations; decontamination.
- [[openmathinstruct-2-recipe]] — SFT settings of OpenMath2-Llama3.1-8B.
- [[metamath]] — operator definitions, data-scaling gains, wrong-answer paths as SFT targets, DROP transfer.
- [[mathscale]] — concept-graph walk equations, knowledge-point ablation, unvalidated solutions.
- [[mammoth]] — hybrid CoT/PoT ablation and dataset-specific SFT regression.
- [[mammoth-2]] — web Q-A extraction and refinement models (correction only).
- [[rstar]] — five actions, Phi3-mini discriminator, GSM8K and StrategyQA results.
- [[rstar-math]] — code-augmented search, Q annotation, PPM loss, rounds, policy vs search scores.
- [[s1]] — selection pipeline, budget forcing, data ablations, sequence-length ablation, training settings.
- [[limo]] — v1/v3 differences, selection, base-model and size dependence, OOD table, training settings.
- [[step-dpo]] — pair construction, loss, DPO comparison, in- vs out-of-distribution chosen steps.
- [[lema-learning-from-mistakes]] — correction data as content, gains, matched-size and matched-token controls.
- [[omegaprm]] — Monte-Carlo labels, binary search, soft labels, question filter.
- [[math-shepherd]] — hard labels and label accuracy vs rollouts.
- [[transferability-of-llm-reasoning]] — SFT vs RL on the same math data; on-policy ablation.
- [[scaling-reasoning-losing-control-mathif]] — instruction-following loss in long-CoT SFT and RL; length effects.
- [[gsm-symbolic]] — templated and irrelevant-clause perturbation results.
- [[longer-context-deeper-thinking]] — long-context ability before reasoning SFT.
- [[open-thoughts]] — reflection-phrase ablation, cross-domain GPQA effect, safety regression.
- [[front-loading-reasoning]] — reasoning data in pretraining vs SFT.
- [[likelihood-displacement]] — similarity-driven displacement in DPO-style losses.
- [[numina-math]] — problem source for rStar-Math (named in a correction).
- [[quiet-star]] — removed section (named in a correction).
