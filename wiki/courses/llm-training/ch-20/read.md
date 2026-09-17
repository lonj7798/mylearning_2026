<!-- chapter: ch-20
     track: synthetic
     kind: content
     title: Distillation as Data: Explanation Traces and the R1-Distill Lineage
     deps: [ch-19]
     sources: [[orca]], [[orca-2]], [[distilling-step-by-step]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[deepseek-r1-distill-synth]], [[bespoke-stratos]], [[sky-t1]], [[openr1]], [[openr1-recipe]], [[open-thoughts]], [[scaling-reasoning-losing-control-mathif]], [[scaling-reasoning-losing-control-mathif-recipe]], [[transferability-of-llm-reasoning]], [[gemma-2]], [[qwen-3]], [[qwen-qwq-traces]], [[dolphin]], [[qwen-2.5]], [[anthropic-distillation-attacks-report]], [[teacher-output-licenses]]
     figures: figures/distill-lineage.html
     revised: 2026-09 (generality revision)
-->

# Chapter 20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage

> **Core insight.** Supervised fine-tuning (SFT) on filtered teacher outputs transfers the teacher's behavior on the prompt distribution that was sampled, including its output format and reasoning style. Inside that distribution the transfer can exceed what RL achieves from the same base: DeepSeek-R1-Distill-Qwen-32B, trained with SFT on 804,745 R1-derived samples, reaches AIME 2024 pass@1 72.6, while RL from Qwen2.5-32B-Base for over 10K steps reaches 47.0 ([[deepseek-r1]], Table 16). Outside that distribution the transfer is weaker and can be negative: in a controlled Qwen3-14B study, SFT on teacher traces for 47K math problems lowered the non-reasoning average from 45.7 to 21.1, while RL on the same problems raised it to 53.2 ([[transferability-of-llm-reasoning]], Table 1).
>
> **Guideline.** When the goal is reasoning ability in a small model and a stronger teacher with a license that permits training on its outputs exists, use SFT on verified teacher traces, because at 32B this beat RL from the base on all reported benchmarks (R1 Table 16). When the student must remain a general model, put non-reasoning data into the distillation set and measure instruction following, safety, and non-math tasks before and after, because math-only trace SFT lowered IFEval from 69.2 to 42.3 in the controlled study (Table 1) and R1's own set holds 177,812 general samples (Table 5). When choosing a teacher, train small students on each candidate's outputs, because QwQ-32B produced better 7B students than the higher-scoring DeepSeek-R1 ([[open-thoughts]], Table 8). When compute allows more than about 17K examples, scale the question pool and answers per question, because OpenThoughts3 scores rise with dataset size up to the 1M scale (Fig. 1; App. G, Fig. 8) and the 16,710-example Bespoke-Stratos-7B trails R1-Distill-Qwen-7B on AIME 2024 by 23.3 points in the authors' own evaluation ([[bespoke-stratos]]).

## Corrections to the version you studied

The version at git commit 4a72e54 contained the following errors. Each line gives the old claim, the corrected statement, and the source locus.

1. "Distillation in 2023–2025 is not logit-matching" → Gemma 2 2B and 9B are pre-trained with a loss on the teacher's full next-token distribution, and Qwen3 small models are fine-tuned by aligning student logits with a teacher's ([[gemma-2]], §3.2; [[qwen-3]], arXiv:2505.09388 §4.5).
2. Guideline "keep ~10–20K correct traces for a strong student — not ~1M" → OpenThoughts3 scores rise with data size up to the 1M scale ([[open-thoughts]], Fig. 1; App. G, Fig. 8); the 16,710-example Bespoke-Stratos-32B scores AIME 2024 63.3 vs 72.6 reported for R1-Distill-Qwen-32B ([[bespoke-stratos]], blog 32B table).
3. "QwQ-32B beats R1 ... even though R1 wins every headline eval" → DeepSeek-R1 scores higher on CodeElo, GPQA-D, and JEEBench, not on every benchmark ([[open-thoughts]], §4.6).
4. "Chapter 19 covered rejection-sampling fine-tuning on one's own rollouts" → ch-19 covers generation methods; rejection sampling on a model's own samples is taught in ch-31.
5. The 16-row Orca system-message table "reconstructed from Appendix A" → the list is Table 2 of the paper and differs from the old table; for example no message asks for "a precise step-by-step plan, then executing the plan" or "Think like you are a world-class expert" (§1 below quotes Table 2) ([[orca]], Table 2).
6. Orca queried "FLAN-v2 / Big-Bench / Chain-of-Thought / GSM-style" tasks → queries come from four FLAN-v2 sub-collections (CoT, NiV2, T0, Flan 2021); Big-Bench tasks are excluded because BBH is an evaluation set ([[orca]], §3.1.2).
7. "BigBench-Hard zero-shot from Vicuna-13B's 44.0 to Orca-13B's 49.7" → Vicuna-13B scores 23.3 on BBH; Orca-13B 49.7; ChatGPT 48.9 ([[orca]], Table 11).
8. "A student trained on trace-rich supervision tends to always emit a long trace ('2+2 = ?' → 400 tokens)", presented as what Orca 2 fixes → neither paper reports this; Orca 2 states that the best solution strategy for a small model may differ from the teacher's ([[orca-2]], Abstract, §3).
9. Orca 2 "five target-behavior modes (direct, step-by-step, explain-then-answer, recall-then-answer, extract-then-answer)" → the paper names step-by-step, recall-then-generate, recall-reason-generate, extract-generate, and direct-answer "and others", and chooses a strategy per task "guided by the performance of Orca" ([[orca-2]], §1, §3).
10. Distilling Step-by-Step: "770M T5 beats 540B PaLM few-shot on 4 benchmarks using only 80% of the labeled data" → the 80% result is for ANLI only; with 100% data the smallest winning student is 220M (e-SNLI), 770M (ANLI, SVAMP), 11B (CQA) ([[distilling-step-by-step]], Abstract, §4.2–4.3).
11. "3–8 in-context CoT exemplars" and "both heads share the T5 encoder", in a block labeled "verbatim from the paper" → about 10-shot prompts; one text-to-text model with `[label]` and `[rationale]` input prefixes, no separate heads; the block was not verbatim ([[distilling-step-by-step]], §3.2, Limitations).
12. "First paper that explicitly argues rationale supervision carries a better learning signal per token" → the paper relates its results to earlier rationale-training work by Magister et al. (2022) and Ho et al. (2022) ([[distilling-step-by-step]], §4.4).
13. R1 "a V3 judge filters them for readability + correctness" → rule-based answer checks, plus a DeepSeek-V3 judge against the reference answer for part of the data (Listing 4), removal of CoT with mixed languages, long paragraphs, or code blocks, and multiple samples per prompt with only correct ones kept ([[deepseek-r1]], Supp. B.3.3).
14. The 800K slice table (math ~200–300K, code ~200–300K, logic/science ~50–100K, non-reasoning ~200K, with SymPy and unit-test verifiers) → Table 5: Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812; total 804,745 ([[deepseek-r1-recipe]], Supp. B.3.3 Table 5).
15. R1-Distill students "Qwen2.5-Math 1.5B/7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B" → Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct ([[deepseek-r1-recipe]], Supp. B.4.3 Table 6). The card [[deepseek-r1-distill-synth]] repeats the old error.
16. "SFT-on-R1-traces for one epoch" → 2–3 epochs, per-model initial LR from 1×10⁻⁴ to 2×10⁻⁵, cosine decay to one tenth, 32,768-token context, batch 64 ([[deepseek-r1-recipe]], B.4.3).
17. "The public DeepSeek docs give ... not a per-source breakdown ... not the judge prompt" → arXiv v2 gives the per-domain Table 5 and the judge prompt (Listing 4); samples per prompt and yield are still not reported ([[deepseek-r1]], B.3.3).
18. "Stage-2 Alignment RL with helpfulness + harmlessness preference rewards" → rule, reward-model, format, and language rewards combined; 1,700 steps with preference rewards only in the final 400; the ~800K SFT restarts from DeepSeek-V3-Base ([[deepseek-r1]], §3.2.2, Fig. 2).
19. "ε = 10 intentionally loose — tight clipping destroys exploration" → the stated reason is that a lower ε truncates gradients for many tokens and a higher ε may cause instability ([[deepseek-r1]], §3.2.1).
20. The open reproductions "collapse the 800K corpus to 17K–440K ... within 2–3 points of the official distill" → corpus sizes and gaps are in §4 below; Stratos-32B trails R1-Distill-Qwen-32B by 9.3 AIME 2024 points against the reported score ([[bespoke-stratos]]).
21. Bespoke-Stratos prompt pool "7K math + 5K code + 5K science", "T = 0.6, up to 3 retries", "SymPy", "GPT-4o judge", "MinHash", "~30–50% reject rate", "median ~3K tokens", "~$4K training" → 10k math (NuminaMATH AIME/MATH/Olympiads), 5k code (APPs, TACO), 1k STILL-2; math filtered by a gpt-4o-mini judge; code by execution; no science verifier in the released script; $800 (dataset card); 8×H100 for 27 hours; the other values are not in the sources ([[bespoke-stratos]], blog, cards, `combine_data.py`).
22. Stratos-32B "LCB ~57%" and ablations "removing code-verification halves LiveCodeBench gain" → LiveCodeBench v2 All 71.1 (57.9 is Sky-T1-32B); the blog reports no ablations ([[bespoke-stratos]], blog 32B table).
23. OpenR1 team "Tunstall, Beeching, Lambert, ..." → Nathan Lambert is not listed on any checked page ([[openr1]], Verification).
24. OpenR1-Math "220K × 2 = ~440K", "T = 0.6", "~20% reject", "~5K median", "~$10K", "HF H100 + API mix" → 400k NuminaMath 1.5 problems × 2 (some 4) = 800k traces, 16k-token limit, 512 H100s locally, released `default` 93,733 and `extended` 131,396 rows; temperature, reject rate, length, and cost not printed ([[openr1]], Update #2, dataset card).
25. OpenR1-Qwen-7B "MATH ~80%, AIME24 ~40% (+3–5 AIME from follow-up GRPO)" on "Qwen2.5-7B-Instruct" → Qwen2.5-Math-7B-Instruct; MATH-500 90.6, AIME24 36.7, AIME25 40.0; no GRPO result appears in the checked sources ([[openr1]], Update #2 table).
26. Sky-T1 "10K math + 5K code + 2K science", "T = 0.7, max 8K", "SymPy", "GPT-4o-mini LLM-judge" for science, "Rewrite via GPT-4o", "~5% format failures" → 10k math, 5k code, 1k STILL-2 science and puzzle; exact matching for math and unit tests for code; no judge; traces rewritten by GPT-4o-mini; temperature, length, and failure rate not stated ([[sky-t1]], NovaSky blog "Data Curation Process").
27. Sky-T1 "~20 pts behind on AIME (teacher ceiling: QwQ < R1)" → 43.3 vs 72.6 is 29.3 points; the blog does not test the teacher as the cause ([[sky-t1]], blog results; [[deepseek-r1]], Table 15).
28. §4.1 "the marginal 783K traces buy you ~2 points on AIME" and §4.2 "the verifier determines the upper bound ... Stratos has the widest stack and the strongest student" → neither is supported; Stratos had no multi-verifier stack, and no source compares students by verifier coverage.
29. §4.3 explanations for QwQ > R1 ("closer distribution", "median 3K vs 5K", "Chinese leaks confuse small students") and "Sky-T1 chose QwQ for cost" → OpenThoughts reports the result without testing these causes; the Sky-T1 blog describes QwQ-32B-Preview as open and comparable to o1-preview and gives no cost reason ([[open-thoughts]], §4.6; [[sky-t1]], blog).
30. §5.3 "Open-R1 flagged ... gains on AIME25 are smaller than gains on AIME24 because the teacher is less saturated" and §5.5 "[[openr1]] calls this out explicitly: wrong-question-correctly" → neither appears in the Open R1 materials ([[openr1]], Verification).
31. §5.5 "process reward (ch-24) is the only defense" → process supervision is ch-44; R1 lists model-based PRMs as unsuccessful, and its V3 judge prompt already checks the reasoning process against the reference ([[deepseek-r1]], G.2, Listing 4).
32. §5.4 Dolphin five-step recipe (regex + classifier refusal detection) → the card names removed categories but gives no method or counts ([[dolphin]], Technical Details).
33. §6 "No answer-side filter beats keeping everything" → no answer filter beat the random baseline; the no-filtering run used 63,200 vs 31,600 examples and is not compute-controlled ([[open-thoughts]], §4.5 Table 7).
34. §7 Llama-3 license row "outputs used to train a non-Llama model are permitted only up to 700M MAU" → the Llama 3.1 model card states that its license allows using outputs "to improve other models including synthetic data generation and distillation"; the old row mixed an output-use rule with a user-count clause ([[teacher-output-licenses]], §2).
35. §7 "R1 and QwQ outputs are the only ones that can be redistributed under permissive licenses at all" → Qwen2.5 models are Apache 2.0 except 3B and 72B, and Llama 3.1 permits output use for distillation ([[teacher-output-licenses]], §2, §4).
36. Connections "ch-24 (RLVR at scale) covers the R1-Zero → R1 RL path" → ch-24 is reasoning-trace synthesis; group-baseline RL is ch-40 and verifiable rewards are ch-44.

## Why this chapter matters for a general-purpose model

**Distillation as data** is the use of a teacher model's sampled outputs as SFT targets for a student model. For a prompt x, the teacher samples y ~ p_T(·|x), a filter keeps or drops y, and the student minimizes −log p_S(y|x) over the kept pairs. This is sequence-level distillation: the student sees one sampled sequence, not the teacher's probabilities. §9 covers the token-level and on-policy variants.

In the post-training pipeline (pre-training → mid-training → SFT → preference optimization → RL → evaluation), distillation data enters mostly at SFT. ch-35 maps the stages where labs insert teacher data. For a general model the question is not only whether the student gains on the teacher's target benchmarks. The measured questions are:

1. Breadth: which abilities outside the sampled prompts improve? (§8, Generalization lens)
2. Narrowing: which abilities fall, such as instruction following, safety behavior, and non-math tasks? (§5, §8)
3. Measurement: how to separate transfer from teacher memorization and from benchmark contamination. (§5.3)

## §1 Orca and Orca 2 — explanation traces elicited with system messages

**Definition.** Explanation tuning trains a student on triples ⟨system message, user query, teacher response⟩, where the system message asks the teacher for explanations or step-by-step answers ([[orca]], §3.1).

**Problem.** Earlier imitation models were trained on short answers from small, homogeneous data. Measured with GPT-4 as judge, Vicuna-13B reaches 92% of ChatGPT; measured on exams it reaches 64%, and on BBH 48% ([[orca]], §1, Figs. 1–3). The concurrent critique by Gudibande et al. reports that imitation models "close little to none of the gap from the base LM to ChatGPT on tasks that are not heavily supported in the imitation data" (quoted in [[orca]], Connections).

**Mechanism.**
1. Sample zero-shot queries from FLAN-v2 sub-collections: CoT 150K, NiV2 440K, Flan 2021 2.5M, T0 2M; Dialog 0 ([[orca]], Table 3).
2. Attach one of 16 hand-written system messages. Messages #8 and #10 are used only for multiple-choice questions (§3.1.1).
3. Stage 1: ChatGPT answers all 5M queries (FLAN-5M). Stage 2: GPT-4 answers a random 1M subset (FLAN-1M). The paper gives two reasons for the ChatGPT stage: the capacity gap between a 13B student and GPT-4, and cost and rate limits (§3.1.3).
4. Train LLaMA-13B for 4 epochs per stage with loss on teacher tokens only (§3.2).

The 16 system messages, quoted from Table 2 ([[orca]]):

```
1  <empty system message>
2  You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.
3  You are an AI assistant. You will be given a task. You must generate a detailed and long answer.
4  You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.
5  You are an AI assistant that follows instruction extremely well. Help as much as you can.
6  You are an AI assistant that helps people find information. Provide a detailed answer so user don't need to search outside to understand the answer.
7  You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.
8  You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.
9  Explain how you used the definition to come up with the answer.
10 You are an AI assistant. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question.
11 You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer.
12 User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.
13 You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.
14 You are an AI assistant, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines that it provides. You solve the task and show how you used the guidelines to solve the task.
15 Given a definition of a task and a sample input, break the definition into small parts. [...] Part #: a key part of the definition. Usage: Sample response that meets the criteria from the key part. Explain why you think it meets the criteria.
16 You are an AI assistant that helps people find information.
```

The paper states that the messages are "designed to preserve the ability of the model to generate both short and long answers" (Table 2 caption).

**Evidence.** Orca-13B, evaluated zero-shot without CoT and with the empty system message: BBH 49.7 (ChatGPT 48.9, GPT-4 67.4, Vicuna-13B 23.3); AGIEval 41.7 (ChatGPT 47.2) ([[orca]], Tables 8, 11). Two stages vs FLAN-1M alone: AGIEval 41.7 vs 37.18 (Table 10). **Result (single study).**

**Conditions and limits.** Table 10 changes two variables at once (the ChatGPT stage and 5× more data), so the teacher-assistant effect is not isolated. No run compares explanation targets with answer-only targets on the same queries ([[orca]], Findings). The authors state that training data simulate zero-shot prompts and that performance "is likely to correlate strongly with the distribution of the tuning data" (§8).

**Orca 2.** [[orca-2]] keeps explanation tuning but selects a solution strategy per task. The procedure (§3): start from diverse tasks; decide, "guided by the performance of Orca", which tasks need which strategy (direct-answer, step-by-step, explain-then-answer, and others); write task-specific system instructions for the teacher; at training time replace them with a generic system message. The last step is **Prompt Erasing**: the student never sees the strategy instruction, so it must infer the strategy from the task. The data are ~817K instances: FLAN ~602K from 1,448 tasks grouped into 126 sub-categories with one instruction each, few-shot 55K, math ~160K, and 2,000 synthetic doctor–patient conversations (§4.1). Orca-2-13B averages 64.49 on six zero-shot reasoning benchmarks vs 56.85 for LLaMA-2-Chat-70B (Fig. 4). Outside that group it is weaker: MT-Bench 6.15 vs 6.64 for LLaMA-2-Chat-13B, attributed to the absence of conversations in training data (Table 3, §6.4); the cautious system message raises its hallucination rate from 10.97 to 29.00 (Table 11). No ablation isolates Prompt Erasing.

**Implication for a general-purpose model.** Diverse system messages widen the output formats a student learns, but the student's strengths still follow the query distribution: Orca 2's conversational score fell below its instruct-model baseline because conversations were absent.

## §2 Distilling Step-by-Step — rationales as a second training task

**Definition.** Distilling Step-by-Step (DSBS) extracts rationales and labels from PaLM-540B by few-shot CoT prompting and trains a small model on two tasks: predict the label, and generate the rationale ([[distilling-step-by-step]], §3).

**Problem.** Task-specific small models need large labeled or distilled datasets to match few-shot LLMs (Abstract).

**Mechanism.**
1. Prompt PaLM-540B with about 10 exemplars of (input, rationale, label) and let it continue for each training input, producing r̂_i and ŷ_i (§3.1, Limitations).
2. Build two training inputs per example: `[label] x_i` → ŷ_i and `[rationale] x_i` → r̂_i, in one T5 model (§3.2).
3. At test time generate only with the `[label]` prefix, so inference cost equals a label-only model (§3.2).

**Formula** (Eqs. 1, 3–4):

```
L = L_label + λ · L_rationale
L_label     = (1/N) Σ_i ℓ(f(x_i), ŷ_i)
L_rationale = (1/N) Σ_i ℓ(f(x_i), r̂_i)
```

N is the number of training examples; f is the student; ℓ is token cross-entropy; ŷ_i is the human label (fine-tuning setting) or the LLM label (distillation setting); r̂_i is the LLM rationale; λ weights the rationale loss. The paper does not print λ. The released code uses `loss = α·label + (1−α)·rationale` with α = 0.5 "recommended" ([[distilling-step-by-step]], Recipe ledger).

**Worked example.** One example; the label is one token that the student predicts with probability 0.5, so L_label = −ln 0.5 = 0.693. The rationale has 4 tokens, each predicted with probability 0.8, so the mean token loss is −ln 0.8 = 0.223. With λ = 1, L = 0.693 + 0.223 = 0.916. With the code's α = 0.5, L = 0.5·0.693 + 0.5·0.223 = 0.458, which is the λ = 1 loss scaled by 0.5. With λ = 1 and α = 0.5, the two conventions differ only by the constant factor 0.5 on the loss.

**Evidence.** T5 220M, 100% data, accuracy on e-SNLI / ANLI / CQA / SVAMP: standard fine-tuning 88.38 / 43.58 / 62.19 / 62.63; DSBS with PaLM rationales 89.51 / 49.58 / 63.29 / 65.50; with GPT-NeoX-20B rationales 89.12 / 48.15 / 63.25 / 63.00 (Table 1). A single-task target that concatenates rationale and label scores 88.88 / 43.50 / 61.37 / 63.00, below standard fine-tuning on ANLI and CQA (Table 2). On ANLI, a 770M student beats few-shot PaLM-540B with 80% of the labeled data (§4.3). **Result (single study),** 4 random runs (App. A.1).

**Conditions and limits.** Every student is trained and tested on one dataset; no held-out-task evaluation exists, so DSBS says nothing about general capability ([[distilling-step-by-step]], Findings). The paper does not report whether rationales with wrong teacher labels were removed.

**Implication for a general-purpose model.** The DSBS result that the training format matters (multi-task > concatenated target at 220M) is a caution for later trace distillation, which uses the concatenated form `<think> rationale </think> answer`. That comparison has not been repeated at LLM scale in the sources of this chapter.

## §3 DeepSeek-R1 distillation data — what the teacher generated

**Definition.** The R1-Distill models are six dense models trained with SFT only on the ~800K samples of DeepSeek-R1's second SFT stage ([[deepseek-r1]], Supp. F).

**Where the data come from.** The R1 pipeline is cold-start SFT on "thousands" of examples → reasoning RL → rejection-sampling SFT restarted from DeepSeek-V3-Base → mixed-reward RL ([[deepseek-r1]], §3, Fig. 2). The ~800K set is built in the third stage from the checkpoint of the first RL stage. ch-31 covers cold start and SFT–RL alternation; ch-35 places this set among other labs' stages.

The paper's description of the reasoning half, quoted from Supp. B.3.3:

> "In the previous stage, we only included data that could be evaluated using rule-based rewards. However, in this stage, we expand the dataset by incorporating additional data, some of which uses a generative reward model by feeding the ground-truth and model predictions into DeepSeek-V3 for judgment [...] we have filtered out chain-of-thought with mixed languages, long paragraphs, and code blocks. For each prompt, we sample multiple responses and retain only the correct ones. In total, we collect about 600k reasoning-related training samples."

The judge prompt (Listing 4, excerpt) defines the two classes:

```
1. **correct**: The answer fully aligns with the reference answer in both reasoning
   process and final conclusion, and address the question without any errors or omissions.
2. **incorrect**: The answer contains major errors in key reasoning steps or the final
   conclusion, or completely deviates from the core of the question.
Output the following content in **JSON** format, including two key:
1. 'analysis': analysis of the answer's correctness;
2. 'correctness': correct/incorrect
```

For the non-reasoning half (~200k), DeepSeek reuses portions of the DeepSeek-V3 SFT data (writing, factual QA, self-cognition, translation), adds software-engineering data (program repair, front-end development), and prompts V3 to write a CoT for some tasks but not for simple queries such as "hello" (B.3.3).

**Composition** ([[deepseek-r1-recipe]], Table 5), with token totals derived as samples × average tokens:

| Domain | Samples | Avg tokens | Tokens (derived) | Share of samples | Share of tokens |
|---|---|---|---|---|---|
| Math | 395,285 | 6,094.2 | 2.409B | 49.1% | 55.9% |
| Code | 211,129 | 7,435.7 | 1.570B | 26.2% | 36.4% |
| STEM | 10,124 | 4,928.8 | 0.050B | 1.3% | 1.2% |
| Logic | 10,395 | 2,739.0 | 0.028B | 1.3% | 0.7% |
| General | 177,812 | 1,419.8 | 0.252B | 22.1% | 5.9% |
| Total | 804,745 | 5,355.3 | 4.310B | 100% | 100% |

**Worked example.** General data are 177,812 / 804,745 = 22.1% of samples but 0.252B / 4.310B = 5.9% of tokens, because a general sample averages 1,419.8 tokens and a math sample 6,094.2. With a token-averaged loss, 94% of the gradient signal comes from reasoning domains. At 2–3 epochs the students see about 8.6B–12.9B tokens (derived; the paper reports epochs, not tokens).

**Students** (Table 6; B.4.3): Qwen2.5-Math-1.5B (initial LR 1×10⁻⁴), Qwen2.5-Math-7B (8×10⁻⁵), Qwen2.5-14B (7×10⁻⁵), Qwen2.5-32B (6×10⁻⁵), Llama-3.1-8B (5×10⁻⁵), Llama-3.3-70B-Instruct (2×10⁻⁵); 2–3 epochs, cosine decay to one tenth, 32,768-token context, batch 64.

**Evidence** (Table 15, pass@1): Distill-Qwen-1.5B AIME 2024 28.9, MATH-500 83.9 (GPT-4o-0513: 9.3, 74.6); Distill-Qwen-32B 72.6, 94.3, GPQA Diamond 62.1, LiveCodeBench 57.2; Distill-Llama-70B 70.0, 94.5, 65.2, 57.5. Distillation vs RL at 32B (Table 16): Qwen2.5-32B-Zero, RL for over 10K steps, scores 47.0 / 91.6 / 55.0 / 40.2 on AIME 2024 / MATH-500 / GPQA-D / LiveCodeBench. **Result (single study).** The authors conclude that distillation is economical, but that "advancing beyond the boundaries of human intelligence may still require more powerful base models and larger-scale reinforcement learning" (F.1, **Interpretation**). They also state that adding RL to the distilled models "could substantially boost model performance" and left it out (Supp. F).

**Conditions and limits.** Samples per prompt, the teacher sampling temperature for rejection sampling, and the yield ratio are not reported. Most samples are single-turn (average rounds 1.0), which the authors say may limit multi-turn ability (B.3.3). All Table 15 benchmarks are math, code, and STEM.

**Implication for a general-purpose model.** R1's own set is not reasoning-only: adding non-reasoning and software-engineering SFT data (Dev2 → Dev3) raised AlpacaEval 2.0 from 55.8 to 62.1 and Aider-Polyglot from 25.6 to 44.8 ([[deepseek-r1]], §4). The 22% general share by samples is the only non-reasoning reference point with a disclosed count in this lineage.

## §4 Open reproductions of R1-style distillation

Three January–February 2025 projects released their data and training settings. The comparison below uses only values printed in the cited sources. [figures/distill-lineage.html](figures/distill-lineage.html) shows the teacher → dataset → student lineage from §1–§6; click a node to see its data, filter, training settings, and reported results with source loci.

| Attribute | Sky-T1-data-17K | Bespoke-Stratos-17k | OpenR1-Math-220k |
|---|---|---|---|
| Source | [[sky-t1]] NovaSky blog | [[bespoke-stratos]] blog, cards, script | [[openr1]] Update #2, dataset card |
| Teacher | QwQ-32B-Preview | DeepSeek-R1 | DeepSeek-R1 |
| Prompts | 10k math (NuminaMATH AIME, MATH, Olympiads), 5k code (APPs, TACO), 1k STILL-2 science and puzzle | same mix as Sky-T1: 10k / 5k / 1k | 400k NuminaMath 1.5 problems |
| Samples per prompt | not reported | not reported | 2, some 4 (800k traces) |
| Max generation | not reported | not reported | 16k tokens |
| Math filter | exact match with ground truth | gpt-4o-mini judge vs ground truth | Math-Verify; Llama-3.3-70B-Instruct judge on a subset of rejected problems |
| Code filter | dataset unit tests | execution (Ray cluster) | none (math only) |
| Formatting | QwQ traces rewritten by GPT-4o-mini | R1 traces not reformatted | R1 output with `\boxed{}` suffix prompt |
| Released size | 17K | 16,710 rows | `default` 93,733; `extended` 131,396 rows |
| Student | Qwen2.5-32B-Instruct, 3 epochs, LR 1e-5, batch 96 | Qwen2.5-32B-Instruct, 3 epochs, LR 1e-5, batch 96 | Qwen2.5-Math-7B-Instruct, 3 epochs, LR 5e-5 |
| Compute / cost | 19 h on 8×H100 (~$450) | generation $800; training 27 h on 8×H100 | generation on 512 H100s; cost not reported |
| Reported results | AIME24 43.3, MATH500 82.4, GPQA-D 56.8 (o1-preview 40.0, 81.4, 75.2) | AIME24 63.3, MATH500 93.0, GPQA-D 58.1, LCB v2 All 71.1 | MATH-500 90.6, AIME24 36.7, AIME25 40.0 |
| Same-size R1-Distill | Qwen-32B: 72.6 / 94.3 / 62.1 (reported) | Qwen-32B: 66.7 / 89.8 / 61.1 / 72.2 (authors' eval) | Qwen-7B: 91.6 / 43.3 / 40 (Update #2 table) |

**§4.1 Verifier false negatives.** A false negative is a correct teacher solution that the filter rejects. Bespoke Labs report that Sky-T1's regex-plus-sympy parser "often fails to extract the right answer"; replacing it with a gpt-4o-mini judge raised the share of retained correct math solutions from 25% to 73% ([[bespoke-stratos]], blog "Data Curation"). Worked example: of 1,000 correct R1 math solutions, the parser keeps 250 and the judge keeps 730, so the judge adds 480 correct training examples without any new teacher calls. Sky-T1 reports the same class of problem for code: without reformatting, taking the last code block gave QwQ ~25% accuracy on APPs; after GPT-4o-mini reformatting, accuracy exceeded 90% ([[sky-t1]], blog). OpenR1's Math-Verify found at least one correct answer for 55% of problems, and the Llama judge recovered 28,000 more problems ([[openr1]], Update #2). **Replicated** across three projects: answer extraction rejected correct teacher traces. Only Bespoke Labs measured the size of the loss (25% vs 73% retention).

**§4.2 Filter strictness.** Open R1 trained Qwen 7B Instruct for one epoch on six filtered versions of a random 200k pool: no filter (200k), Llama judge (124k), Math-Verify (88.7k), re-parsed Math-Verify (101k), union (154k), intersection (71.2k). In the first 40 steps MATH-500 was 0.61 without filtering and 0.72 with the intersection; the gap shrank later and the unfiltered set stayed competitive ([[openr1]], Update #3). The authors recommend the union. **Result (single study),** no final-step table printed.

**§4.3 Data mixture and student size in Sky-T1.** Training the 32B model on 3–4K Numina math problems raised AIME24 from 16.7 to 43.3; adding APPs code data dropped it to 36.7; enriching the mix with harder NuminaMath and TACO problems restored 43.3 while improving code ([[sky-t1]], blog "Data mixture matters"). At 7B and 14B the team saw "only modest improvements" (Qwen2.5-14B-Coder-Instruct on APPs: LiveCodeBench 42.6 → 46.3) and frequent repetitive outputs. The authors hypothesize that math and code need different reasoning styles (**Interpretation**).

**§4.4 Small data at 7B.** Bespoke-Stratos-7B (Qwen2.5-7B-Instruct, 16,710 examples) scores AIME2024 20.0, MATH500 82.0, GPQA-D 37.8, against 10.0 / 74.2 / 33.3 for its instruct base and 43.3 / 89.4 / 44.9 for R1-Distill-Qwen-7B in the authors' evaluation (55.5 / 92.8 / 49.1 reported). LCB v2 Hard falls below the base (1.6 vs 3.3) ([[bespoke-stratos]], 7B table). The model card suggests 17k vs 800k examples as a possible cause of the gap (**Interpretation**).

**Conditions and limits.** All three projects evaluate only math, code, and GPQA-Diamond. Bespoke Labs state that "benchmarks convey only one side to the story" ([[bespoke-stratos]], "Thoughts and future work"). None reports instruction following, safety, or chat evaluations.

## §5 What the student inherits from the teacher

**§5.1 Format and reasoning style.** The student learns the teacher's trace structure together with the answers. OpenThoughts removed self-reflection phrases from its traces: average trace length fell from 11,593 to 328 tokens and the average score from 51.4 to 26.3; dropping traces over 2,048 tokens gave 34.2 ([[open-thoughts]], App. H.3, Table 22). Open R1 found that without a `<think>` prefill, out-of-domain prompts reverted to the base instruct model's behavior, and recommends enforcing the prefill in the chat template ([[openr1]], Update #3 Lesson 4). The QwQ-32B-Preview card lists "Language Mixing and Code-Switching" and "Recursive Reasoning Loops" as known limitations ([[teacher-output-licenses]], §3); a trace filter that checks only the final answer does not remove them. R1 removes mixed-language CoT explicitly (B.3.3).

**§5.2 Style versus capability.** Imitation can match the teacher's style without matching its ability. Orca's own motivating measurement is that GPT-4 judging rates Vicuna-13B at 92% of ChatGPT while BBH rates it at 48% ([[orca]], §1). OpenThoughts reports that all tested distilled reasoning models, including OpenThinker3-7B, show accuracy changes across structure-preserving variants of the same problems; the paper describes the changes as large and gives them only in a figure ([[open-thoughts]], App. N, Fig. 13). **Replicated** as a measurement caution: judge scores and fixed benchmark items overstate imitation models.

**§5.3 Contamination and time-split evaluation.** AIME and MATH items are public, so a teacher may have seen them. Protocols in this lineage: R1 reports AIME 2025 (after its training data) as 11.3/15 vs 12.0/15 for o1-1217 and notes that 10-gram decontamination cannot catch paraphrases ([[deepseek-r1]], E.2, D.1). OpenThoughts keeps AIME25, HMMT 02/25, HLE, and LiveCodeBench 06/24–01/25 out of pipeline decisions and decontaminates with Indel similarity and 13-gram matching ([[open-thoughts]], §4, App. F). Open R1 uses 8-gram decontamination following s1 ([[openr1]], README).

**§5.4 Safety and refusal behavior.** A student inherits the teacher's safety behavior only on prompts that appear in the data. OpenThinker3-7B, trained without safety data, has a HarmBench harmfulness rate of 55.5 vs 14.5 for its base Qwen2.5-7B-Instruct; XSTest over-refusal moves from 4.4 to 5.6 ([[open-thoughts]], App. L, Table 30). Orca-2-13B, with no RLHF safety training, has a 13.47% violent-content defect rate vs 0.17% for LLaMA-2-Chat-13B ([[orca-2]], Tables 4–5). The Dolphin dataset removed "instances of alignment, refusal, avoidance, and bias" from Orca-style GPT-4 and GPT-3.5 completions, but its card gives no method, counts, or evaluation ([[dolphin]]).

**§5.5 Correct answer, wrong reasoning.** An outcome filter compares only the final answer, so a trace with a correct answer and an invalid derivation passes. None of the open-reproduction sources measures how often this happens. R1's generative judge asks whether the answer "fully aligns with the reference answer in both reasoning process and final conclusion" (Listing 4), which targets this case for the subset it judges. For code, the reverse error also occurs: seven R1 CodeForces solutions that passed all public tests failed the full test set, so public-test filtering admits false positives ([[openr1]], Update #3). Step-level verification is taught in ch-44; R1 lists PRMs under unsuccessful attempts and states that a model-based PRM "inevitably leads to reward hacking" (G.2).

## §6 OpenThoughts — controlled experiments on the data recipe

**Protocol.** Each candidate strategy produces 31,600 examples and fine-tunes Qwen2.5-7B-Instruct; the strategy with the best average over 8 development benchmarks is kept for the next pipeline step; DeepSeek-R1 is the default teacher ([[open-thoughts]], §4). More than 1,000 such experiments were run (Abstract).

**Findings with numbers** (§4.1–4.6):
1. Question sources: the best and worst code sources differ by 17.2 points (Table 3). Top-2 code sources average 41.3 vs 36.4 for top-16 (Table 4).
2. Question filters: GPT-4o-mini difficulty ratings win for code (43.0); longest-response selection with GPT-4.1-mini wins for math (41.9) and science; best filters add 4% (math) and 6% (code) over random (Table 5). LLM-based filters beat embedding and fastText filters (§1).
3. Answers per question: 16 in all domains; exact deduplication for math and science, none for code (Table 6).
4. Answer filtering: for math, random filtering (64.8) beats every other filter, for example GPT verification 61.4 (Table 7).
5. Teacher: QwQ-32B vs DeepSeek-R1 average 44.2 vs 42.3 (code) and 44.2 vs 41.6 (math) (Table 8); science 39.1 vs 35.9 (Table 49).
6. Scale: OpenThinker3-7B on 1.2M examples (850K math, 250K code, 100K science) scores AIME25 53.3, LiveCodeBench 06/24–01/25 51.7, GPQA-D 53.7, 12.4 points above R1-Distill-Qwen-7B on the 12-task average (Table 1).

**Conditions and limits.** Pipeline choices were selected at 31,600 examples with a 7B student; the verification result flips with student size (7B 45.0 → 41.9 with verification; 32B 62.1 → 64.5) ([[open-thoughts]], App. H.1.1, Table 15). The mixture ratio was copied from OpenThoughts2-1M, not ablated (§5). Teacher sampling temperature and length are not reported (Verification). On HLE MCQ, OpenThinker3-7B scores 10.2 vs 12.7 for its instruct base (Table 1). The authors note that selecting by overall average assumes cross-domain transfer (§6).

**Implication for a general-purpose model.** The recipe was optimized for math, code, and science averages. Code-only sets reach GPQA-D 47.3 or 36.7 depending on the source, while mixes with science data reach 52.7 ([[open-thoughts]], App. H.5), so transfer between reasoning domains depends on the data source, and transfer to non-reasoning tasks was not a selection criterion.

## §7 Licensing — whether teacher outputs may be used for training

A distillation dataset is releasable only when the teacher's license or terms of service permit training other models on its outputs. The statements below are quoted in [[teacher-output-licenses]]; they are not legal advice and must be re-checked at generation time.

| Teacher | License | Output use for training other models | Source |
|---|---|---|---|
| DeepSeek-R1 | MIT (code and weights) | "allow for any modifications and derivative works, including, but not limited to, distillation for training other LLMs" | R1 README §7 |
| R1-Distill-Qwen models | inherit Apache 2.0 from Qwen2.5 | per base license | R1 README §7 |
| R1-Distill-Llama-8B / -70B | Llama 3.1 / Llama 3.3 licenses | per base license | R1 README §7 |
| Llama 3.1 | Llama 3.1 Community License | the license "allows" using outputs "to improve other models including synthetic data generation and distillation" | Llama 3.1 model card |
| QwQ-32B-Preview, QwQ-32B | Apache 2.0 | Apache 2.0 terms | model card metadata |
| Qwen2.5 | Apache 2.0 except 3B and 72B | per model license | [[qwen-2.5]], Table 1 |
| Closed API models | terms of service | Anthropic reports distillation campaigns that violated its terms of service | [[anthropic-distillation-attacks-report]] |

The released datasets carry their own licenses: Bespoke-Stratos-17k and OpenR1-Math-220k are Apache 2.0; Dolphin is Apache 2.0 while "each model follows the license of its base model" ([[teacher-output-licenses]], §6). A dataset license does not override the terms under which its outputs were generated.

## §8 Effect of reasoning distillation on non-reasoning ability

**§8.1 Instruction following (MathIF).** **Definition.** MathIF adds 1–3 Python-verifiable constraints (length, lexical, format, affix) to 420 math problems ([[scaling-reasoning-losing-control-mathif]], §3). **Formula** (Eq. 1): for a query with n constraints and I(C_i) = 1 when constraint i is met,

```
HAcc = Π_{i=1..n} I(C_i)          SAcc = (1/n) Σ_{i=1..n} I(C_i)
```

HAcc (hard accuracy) is 1 only when all constraints are met; SAcc (soft accuracy) is the share met; both are averaged over queries. **Worked example:** a triple-constraint query with 2 of 3 constraints met has HAcc = 0 and SAcc = 2/3 = 0.67.

**Evidence.** The best of 23 reasoning models, Qwen3-14B, reaches HAcc 50.71 (Table 3). DeepSeek-R1-Distill-Llama-70B (41.43) scores below Qwen3-4B (44.05). The largest relative correctness drops when constraints are added are for the SFT-only distilled models R1-Distill-Qwen-1.5B (−40.09%) and R1-Distill-Llama-8B (−39.04%) (Table 3). In controlled runs on Qwen2.5-7B with 18k QwQ-32B traces (wrong answers and CoT over 8,192 tokens removed), SFT moved HAcc / SAcc / math accuracy from 15.95 / 33.13 / 13.59 to 7.86 / 21.03 / 23.10 (Table 4). Across 16 trained variants on four bases, 15 lost HAcc and SAcc while 15 gained math accuracy. On IFEval (prompt-level strict), Llama-3.3-70B-Instruct scores 90.38 and DeepSeek-R1-Distill-Llama-70B 78.74; Qwen2.5-32B-Instruct 80.96 and s1-32B 58.04 (App. G, Table 13; released models, not a controlled comparison). **Result (single study)** for the controlled runs.

**§8.2 Transfer to other domains (Transferability Index).** **Definition.** Huan et al. define a Transferability Index (TI) that expresses a model's gain on a group of non-math benchmarks as a percentage of its gain on math benchmarks ([[transferability-of-llm-reasoning]], arXiv:2507.00432v2 §2.1). **Formula:** for group g and benchmark b,

```
ΔR_b = R_b(model) − R_b(base);   σ_g = Std{ΔR_b : b ∈ B_g};   δ_b = ΔR_b / σ_g
s_b  = sign(δ_b)·|δ_b|^(1/2);    w_b = 100 − R_b(base);       ŵ_b = w_b / Σ_{u∈B_g} w_u
DI_g = Σ_b ŵ_b·s_b;              TI_g(%) = 100 · DI_g / DI_math,   g ∈ {other, non}
```

R_b is accuracy in percent; σ_g normalizes gains within a group; the square root limits the effect of extreme gains; w_b gives harder benchmarks (lower base score) more weight; DI_g is the group's domain index.

**Worked example** (two benchmarks per group; Std taken as the population standard deviation, which the paper does not specify). Math: A goes 20 → 40, B goes 60 → 70, so ΔR = (20, 10), σ = 5, δ = (4, 2), s = (2, 1.414), w = (80, 40), ŵ = (0.667, 0.333), DI_math = 1.805. Non-reasoning: C goes 50 → 45, D goes 70 → 72, so ΔR = (−5, 2), σ = 3.5, δ = (−1.429, 0.571), s = (−1.195, 0.756), w = (50, 30), ŵ = (0.625, 0.375), DI_non = −0.464. TI_non = 100 · (−0.464 / 1.805) = −25.7%: the model lost ground outside math while gaining in math.

**Evidence.** Controlled study on Qwen3-14B-Base with 47K math problems (DeepScaler plus SimpleRL levels 3–5); SFT targets are Qwen3-32B traces kept by rejection sampling; RL uses GRPO with answer correctness (§2.2, App. A.3.2). Table 1:

| Model | Math avg | Other reasoning avg | Non-reasoning avg | IFEval | TI_other | TI_non |
|---|---|---|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 | 45.7 | 69.2 | – | – |
| SFT on thinking-mode traces | 49.8 | 45.3 | 21.1 | 42.3 | +52.2 | −104.1 |
| SFT on non-thinking traces | 32.3 | 45.2 | 29.0 | 41.4 | +165.4 | −278.9 |
| RL (GRPO) | 53.8 | 60.0 | 53.2 | 70.0 | +82.3 | +52.2 |

In an ablation on Qwen3-8B-Base (Table 4), off-policy SFT on Qwen3-32B traces gives non-reasoning average 26.6 (base 33.6, TI_non −40.5); on-policy SFT on the student's own filtered samples gives 35.0 (TI_non +30.2); off-policy RL gives 31.7 (+4.5). The authors conclude that the sampling distribution is the most important factor (§6, **Interpretation**) and report larger hidden-state shifts after SFT than after RL (Table 2). **Result (single study)**, one base family, math-only prompts.

**Conditions and limits.** Both studies distill from teachers with math-only prompts; R1's own set has 22% general samples, which neither study reproduces. The controlled SFT data in §8.2 contain no non-reasoning samples, so the result does not show that distillation with a mixed set loses general ability. ch-38a covers the SFT-versus-RL evidence and its counter-studies; ch-30a covers forgetting measurement.

**Implication for a general-purpose model.** When the distilled student must remain general, evaluate it on instruction-following (IFEval, MathIF-style constraints), non-reasoning QA, and safety sets in addition to the teacher's target benchmarks, because math-only trace SFT lowered IFEval and non-reasoning averages in both studies above. When the distillation set is math-only, plan non-reasoning data or a later stage for breadth; no source in this chapter shows that a math-only set preserves it.

## §9 Token-level and on-policy distillation

**Token-level (logit) distillation.** The student matches the teacher's full next-token distribution instead of one sampled token. Gemma 2's objective ([[gemma-2]], §3.2):

```
min_{P_S}  Σ_x  −P_T(x | x_c) · log P_S(x | x_c)
```

x is a vocabulary token, x_c its context, P_T the teacher's probability, and P_S the student's. **Worked example:** vocabulary of 3 tokens, teacher P_T = (0.7, 0.2, 0.1), student P_S = (0.5, 0.3, 0.2). The soft cross-entropy is −(0.7 ln 0.5 + 0.2 ln 0.3 + 0.1 ln 0.2) = 0.887. The teacher entropy is 0.802, so the KL divergence is 0.887 − 0.802 = 0.085. Sequence-level SFT on a sampled token 1 would use only −ln 0.5 = 0.693 and ignore the teacher's 0.2 and 0.1 on the other tokens. **Evidence:** a 2B model trained on 500B tokens averages 60.3 on 3 benchmarks from scratch and 67.7 when distilled from a 7B teacher (Table 6); the released Gemma 2 2B and 9B are pre-trained this way on 2T and 8T tokens, while 27B is trained from scratch (§1). Token-level distillation needs the teacher's probabilities at every position, so it requires teacher weights or logit access and a shared tokenizer; API text outputs are not enough.

**On-policy distillation.** The student generates the sequences, and the teacher scores every token. Qwen3 uses two phases for its 0.6B–14B and 30B-A3B models ([[qwen-3]], arXiv:2505.09388 §4.5): (1) off-policy distillation on teacher outputs in `/think` and `/no_think` modes; (2) on-policy distillation, where "the student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence". Table 21 (§4.7), starting from the same off-policy distilled Qwen3-8B, math and code queries only:

| Method | AIME'24 (pass@64) | AIME'25 (pass@64) | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-D | GPU hours |
|---|---|---|---|---|---|---|---|
| Off-policy distillation | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | – |
| + RL | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy distillation | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

**Result (single study).** The report attributes the pass@64 gain to the teacher's logits expanding the student's exploration (§4.7, **Interpretation**). ch-35 places this stage among other labs' pipelines and links the objectives to the separate on-policy-distillation course (ch-04 and ch-05).

## Negative samples and negative feedback

**Where negatives come from.** In distillation-as-data, negatives are teacher samples that fail a check: an answer mismatch (Sky-T1 exact match; Math-Verify), a model judge verdict (gpt-4o-mini; Llama-3.3-70B-Instruct; DeepSeek-V3 with Listing 4), a unit-test failure, or a readability rule (mixed languages, long paragraphs, code blocks in R1). Reported error rates of the labelers: the Sky-T1 parser kept 25% of correct math solutions vs 73% for a judge ([[bespoke-stratos]]); public CodeForces tests passed seven solutions that failed the full tests ([[openr1]]).

**What current practice does with them.** All sources in this chapter use type (1), negative marginal value: failing samples are discarded and never enter training. None uses them as content, conditioning, or gradient. R1's RL stages do use negative advantages, but the distillation set does not ([[deepseek-r1]], Findings).

**Mechanism: SFT only pushes up.** For logits z and softmax p, the gradient of the log-probability of target token y is

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

where 1[j = y] is 1 for the target token and 0 otherwise. Worked example: p = (0.7, 0.2, 0.1) and target y = token 2 (p = 0.2). The gradient is (−0.7, +0.8, −0.1); a gradient step on −log p_y raises z_2 and lowers the other logits in proportion to their probabilities. If the same sample were used as a negative gradient, the signs flip to (+0.7, −0.8, +0.1): the removed probability goes mostly to token 1, the already most likely token. This is why negative gradients on low-probability samples concentrate mass on the current mode (ch-43a derives it in full). Discarding avoids this effect but also discards the information in the failure.

**Evidence with numbers.**
1. Benefit of discarding depends on student size: verification lowered the 7B average (45.0 → 41.9) and raised the 32B average (62.1 → 64.5) ([[open-thoughts]], Table 15). LLM-generated unit-test filtering on 16,000 code examples gave LCB 36.0 vs 38.5 unfiltered (Table 18).
2. Early vs late training: strict filtering helped at 40 steps (MATH-500 0.72 vs 0.61) and the gap closed later ([[openr1]], Update #3).
3. Negative gradients and transfer: in the Qwen3-8B ablation, off-policy RL (the GRPO objective with KL on n = 8 Qwen3-32B responses per query, using the highest-reward response for the gradient, §5.2) gave TI_non +4.5 vs −40.5 for off-policy SFT on Qwen3-32B traces ([[transferability-of-llm-reasoning]], Table 4). The authors head the result "Credit assignments and negative examples matter" (§5.2, **Interpretation**); the ablation does not isolate a negative-gradient term, so it is not evidence about negatives alone.

**Controls.** When a parser-based filter is used, a judge fallback or answer re-parsing recovers false negatives before discarding ([[bespoke-stratos]]; [[openr1]], Update #3). When correct final answers can come from invalid reasoning, a reference-answer judge that also checks the reasoning targets that case (Listing 4). When failures are used as training signal, the options in ch-31a (corrections as content, failure conditioning, unlikelihood with an NLL anchor) bound the negative term, whereas an unbounded negative term added to a distillation SFT loss has the concentration effect derived above.

**Diagnostics.** Filter retention per domain and per source; a hand-checked sample of rejected traces for false negatives; students trained with and without the filter at the target student size; pass@1 and pass@k for the student (ch-00, ch-43).

**Effect on generality.** An answer filter with false negatives can remove problems whose answers are hard to parse and so shift the question distribution; no source in this chapter measures that shift (**Interpretation**). OpenThoughts measured no answer-filter gain for math at 31,600 examples (Table 7). OpenThinker3-7B, trained without safety data, has a higher HarmBench harmfulness rate than its base (§5.4); the authors did not test which data change caused it.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1 (Dev3 SFT set) | 671B MoE teacher | distill-SFT data | samples (avg tokens) by domain | Math 395,285 (6,094.2); Code 211,129 (7,435.7); STEM 10,124 (4,928.8); Logic 10,395 (2,739.0); General 177,812 (1,419.8); total 804,745 (5,355.3) | arXiv:2501.12948v2 Supp. B.3.3 Table 5 ([[deepseek-r1-recipe]]) | verified 2026-09-14 | §4 Table 3: Dev2 → Dev3 AlpacaEval 2.0 55.8 → 62.1 |
| DeepSeek-R1 (Dev3 SFT set) | — | distill-SFT data | reasoning filter | multiple samples per prompt, keep correct; V3 judge vs reference for part (Listing 4); drop CoT with mixed languages, long paragraphs, code blocks | v2 B.3.3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev3 SFT set) | — | distill-SFT data | samples per prompt; sampling temperature; yield | not reported | checked v2 §3, B.3.3, B.4 | not reported | — |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | distill-SFT | base; initial LR | Qwen2.5-Math-1.5B; 1×10⁻⁴ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-7B | 7B | distill-SFT | base; initial LR | Qwen2.5-Math-7B; 8×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-14B | 14B | distill-SFT | base; initial LR | Qwen2.5-14B; 7×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-32B | 32B | distill-SFT | base; initial LR | Qwen2.5-32B; 6×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | Table 16: 72.6 AIME 2024 vs 47.0 for RL from Qwen2.5-32B-Base |
| DeepSeek-R1-Distill-Llama-8B | 8B | distill-SFT | base; initial LR | Llama-3.1-8B; 5×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Llama-70B | 70B | distill-SFT | base; initial LR | Llama-3.3-70B-Instruct; 2×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | v1 §2.4: Llama-3.3 reasoning "slightly better" than Llama-3.1 |
| All six R1-Distill models | 1.5B–70B | distill-SFT | epochs; schedule; context; batch | 2–3; cosine to 1/10 of initial LR; 32,768 tokens; 64 (unit not stated) | v2 B.4.3 | verified 2026-09-14 | no ablation reported |
| Bespoke-Stratos-32B | 32B | distill-SFT | examples; teacher; math filter | 16,710; DeepSeek-R1; gpt-4o-mini judge vs ground truth | HF dataset card; blog "Data Curation" ([[bespoke-stratos]]) | verified 2026-09-14 | retained correct solutions 25% (parser) → 73% (judge) |
| Bespoke-Stratos-32B | 32B | distill-SFT | base; LR; schedule; warmup; batch; epochs | Qwen2.5-32B-Instruct; 1e-5; cosine; ratio 0.1; 96 sequences; 3 | HF model card "Training hyperparameters" | verified 2026-09-14 | no ablation reported |
| Bespoke-Stratos-32B | 32B | distill-SFT | compute; teacher temperature | 8×H100, 27 h; temperature not reported (script passes `temp: 0.0`, not confirmed as the run) | model card; `generate_numina_data.py` L78 | verified 2026-09-14 / not reported | — |
| Sky-T1-32B-Preview | 32B | distill-SFT | teacher; data | QwQ-32B-Preview; 5k code (APPs, TACO), 10k math (NuminaMATH AIME, MATH, Olympiads), 1k STILL-2; traces rewritten by GPT-4o-mini | NovaSky blog "Data Curation Process" ([[sky-t1]]) | verified 2026-09-15 (blog text) | blog "Data mixture matters": AIME24 43.3 with balanced mix vs 36.7 after adding APPs code |
| Sky-T1-32B-Preview | 32B | distill-SFT | base; epochs; LR; batch; compute | Qwen2.5-32B-Instruct; 3; 1e-5; 96; 19 h on 8×H100, DeepSpeed ZeRO-3 offload | blog "Training" | verified 2026-09-15 (blog text) | no ablation reported |
| Sky-T1-32B-Preview | 32B | distill-SFT | teacher temperature; max tokens; max sequence length | not reported | checked blog | not reported | — |
| OpenR1-Math-220k | — | distill-SFT data | prompts; samples; max length; temperature | 400k NuminaMath 1.5; 2 (some 4); 16k tokens; not printed | HF Update #2; dataset card ([[openr1-recipe]]) | verified 2026-09-14; temperature not reported | only 75% of problems solved within 8k tokens |
| OpenR1-Qwen-7B | 7B | distill-SFT | base; data; epochs; LR; schedule | Qwen2.5-Math-7B-Instruct; `default` 93,733 rows; 3; 5e-5; linear, 10% warmup | Update #2 | verified 2026-09-14 | dataset card: `default` > `extended` after SFT |
| OpenR1-Distill-7B | 7B | distill-SFT | data; epochs; LR; batch; max length; packing; hardware | Mixture-of-Thoughts, 349,317 rows (math, code, science); 5; 4.0e-5; 128 sequences; 32,768; off; 8×H100, DeepSpeed ZeRO-3 | `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`@1416fa0; Mixture-of-Thoughts card; model card | verified 2026-09-14 | Update #3: packing worse on all ablated datasets |
| OpenThinker3-7B | 7B | distill-SFT | examples; teacher; answers per question | 1.2M (850K math, 250K code, 100K science); QwQ-32B; 16 | arXiv:2506.04178v2 §5, §4.4, §4.6 ([[open-thoughts]]) | verified 2026-09-14 | Tables 6 and 8 (31,600-example runs) |
| OpenThinker3-7B | 7B | distill-SFT | LR; batch; epochs; packing | 8e-5; 512 (unit not stated); 5; yes | App. D.2 Table 9 "Large" | derived (1.2M is in the > 31.6K range) | per-scale sweep results not reported |
| Orca-13B | 13B | distill-SFT | data; epochs; length; loss | FLAN-5M ChatGPT then FLAN-1M GPT-4; 4 per stage; 2,048 tokens, packing 2.7 examples/sequence; loss on teacher tokens | arXiv:2306.02707v1 §3.1.3, §3.2 ([[orca]]) | verified 2026-09-14 | Table 10: 41.7 vs 37.18 AGIEval (FLAN-1M only) |
| Orca-13B | 13B | distill-SFT | optimizer; LR; batch | not reported | checked §3.2, §4, footnotes | not reported | — |
| Orca-2-13B | 13B | distill-SFT | stages; epochs; length | FLAN-v2 (1) → 5M ChatGPT (3) → 1M GPT-4 + ~817K Orca 2 (4); 4,096 packed | arXiv:2311.11045v2 §4.2 ([[orca-2]]) | verified 2026-09-14 | no ablation reported |
| T5-Large student (DSBS) | 770M | distill-SFT | LR; batch; max input; steps; λ | 5×10⁻⁵; 64; 1,024; 10,000; λ not reported (code α = 0.5) | arXiv:2305.02301v2 App. A.1; repo README ([[distilling-step-by-step]]) | verified 2026-09-14 / λ not reported | Table 2 (220M): multi-task > single-task on 4 datasets |
| UniReason-Qwen3-14B (SFT) | 14B | distill-SFT | data; teacher | 47K math problems; Qwen3-32B traces kept by rejection sampling | arXiv:2507.00432v2 §2.2, App. A.3.2 ([[transferability-of-llm-reasoning]]) | verified 2026-09-15 (paper text) | Table 1: non-reasoning avg 45.7 → 21.1 (SFT) vs 53.2 (RL) |
| MathIF §5.2 SFT runs | 1.5B, 7B | distill-SFT | teacher; filter; examples; max length; LR | QwQ-32B; wrong answers and CoT > 8,192 tokens dropped; 18k; 8,192; 1e-6 | arXiv:2505.14810v2 §5.2, Table 7 ([[scaling-reasoning-losing-control-mathif-recipe]]) | verified 2026-09-14 | Table 4 |
| Qwen3-8B | 8B | distill-SFT (on-policy) | teacher; objective; compute | Qwen3-32B or Qwen3-235B-A22B; student logits aligned to teacher, KL minimized; 1,800 GPU hours (RL: 17,920) | arXiv:2505.09388v1 §4.5; §4.7 Table 21 ([[qwen-3]]) | verified 2026-09-15 (report text) | Table 21 |
| Gemma 2 2B | 2B | pretrain-stable | objective; tokens | cross-entropy to teacher next-token distribution; 2T tokens | arXiv:2408.00118v3 §3.1, §3.2 ([[gemma-2]]) | verified 2026-09-14 | Table 6: 67.7 vs 60.3 from scratch (500B-token ablation) |

**Starting point for a small general-purpose run.** For a 7B student with SFT on R1-style traces, the verified rows give two tested configurations: OpenR1-Distill-7B uses 5 epochs, LR 4.0e-5, 128 sequences per step, 32,768-token sequences, and no packing on 349,317 math, code, and science rows on 8×H100; R1-Distill-Qwen-7B uses 2–3 epochs, initial LR 8×10⁻⁵ decayed to one tenth, batch 64, and 32,768-token context on 804,745 samples. Both sources evaluate only math, code, and science, so neither configuration was selected for general ability. The data side for a general student has one disclosed reference point: R1's 177,812 general samples in 804,745 (Table 5). When the student must stay general, the evaluation used to choose epochs should include IFEval and a safety set, because neither source's selection included them.

## Generalization lens

**(a) What increases breadth.**
- Mixing domains in the distillation set: Open R1's Mixture-of-Thoughts found training on all domains together best ([[openr1]]); science data raised GPQA-D to 52.7 from code-only 36.7–47.3 ([[open-thoughts]], App. H.5). R1's non-reasoning and SWE data raised AlpacaEval 2.0 and Aider-Polyglot (§3).
- On-policy distillation after off-policy distillation: Qwen3-8B improved on all six reported benchmarks, including MMLU-Redux 86.4 → 88.3, and pass@64 on AIME (§9).
- Diverse output formats requested from the teacher (Orca's 16 system messages) give the student short and long answer modes (§1); no source measures the effect separately.

**(b) What causes narrowing or forgetting.**
- Math-only trace SFT: non-reasoning average 45.7 → 21.1 and IFEval 69.2 → 42.3 on Qwen3-14B-Base ([[transferability-of-llm-reasoning]], Table 1); HAcc and SAcc fell in all four MathIF SFT-only runs, and in 15 of 16 trained variants including RL ([[scaling-reasoning-losing-control-mathif]], Table 4). **Replicated** across two independent studies, both with math-only prompts.
- No safety data: HarmBench harmfulness 14.5 → 55.5 for OpenThinker3-7B (§5.4).
- Missing conversation data: Orca-2-13B MT-Bench 6.15 vs 6.64 for LLaMA-2-Chat-13B (§1); R1's set is single-turn on average (§3).
- Mixing without balance: adding APPs code to Sky-T1's math data lowered AIME24 from 43.3 to 36.7 until harder math and TACO code were added (§4.3).

**(c) How to measure it for this stage.**
- Benchmarks dated after the teacher's data: AIME 2025, HMMT 02/25, LiveCodeBench windows after the teacher release ([[open-thoughts]], §4; [[deepseek-r1]], E.2).
- Non-reasoning and constraint suites: IFEval, MathIF HAcc/SAcc, CoQA, HaluEval; summarize with TI_other and TI_non (§8).
- Safety and over-refusal: HarmBench and XSTest before and after distillation (§5.4).
- Perturbed problems to separate memorized answers from reasoning ([[open-thoughts]], App. N).
- Known measurement errors: GPT-4-as-judge overestimates imitation models (§5.2); AIME 2024 has 30 problems, so Open R1 averages pass@1 over 64 samples ([[openr1]]); 10-gram decontamination misses paraphrases ([[deepseek-r1]], D.1).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Answer parser rejects correct traces | retention rate below the teacher's measured accuracy; hard or unusually formatted problems under-represented | hand-label 100 rejected traces; compare retention with a judge fallback (25% vs 73% in [[bespoke-stratos]]) |
| Evaluating the student only on the teacher's target benchmarks | math and code gains reported; no IFEval, chat, or safety numbers | add IFEval, a non-reasoning QA set, and HarmBench/XSTest; compute TI_non |
| Math-only distillation set for a general model | IFEval and non-reasoning scores fall after SFT | run the same eval on the base and the student; add general samples (R1: 22% of samples) |
| Missing `<think>` prefill in the chat template | out-of-domain prompts get no reasoning trace | test prompts outside the training domains with and without the prefill ([[openr1]], Lesson 4) |
| Choosing the teacher by benchmark score alone | a weaker-scoring teacher gives better students | train small students on 31.6K-scale samples from each candidate ([[open-thoughts]], Table 8) |
| Selecting the recipe at one student size and applying it at another | filter or data choices reverse at the new size | repeat the key ablation at the target size (verification: 7B −3.1, 32B +2.4 in [[open-thoughts]] App. H.1.1 Table 15) |
| Using public tests as the only code filter | high training-set pass rate; failures on hidden tests | re-run a sample on full test sets ([[openr1]], Update #3) |
| Contaminated evaluation | large gains on pre-teacher benchmarks, small on newer ones | report benchmarks dated after the teacher's data; run n-gram and similarity decontamination |
| Generating data from a teacher whose terms forbid it | dataset cannot be released or used commercially | read the license file or terms at generation time ([[teacher-output-licenses]]) |

## Check your understanding

1. R1's general data are 22.1% of samples but 5.9% of tokens. Explain how the loss normalization (per sequence vs per token) changes the influence of general data on the student, and what this implies for choosing the general share.
2. R1-Distill-Qwen-32B beats RL from Qwen2.5-32B-Base on every Table 16 benchmark, while in Huan et al. RL beats SFT on non-reasoning tasks. Explain why both results can hold, using the prompt distributions and evaluation suites of each study.
3. A gpt-4o-mini judge raised retained correct math solutions from 25% to 73%. Explain which problems the parser most likely dropped and how that drop changes the difficulty and format distribution of the SFT set.
4. In the TI worked example, change benchmark C to 50 → 55. Recompute DI_non and TI_non and explain why the square root and difficulty weights change the result compared with averaging raw gains.
5. Explain why token-level distillation (Gemma 2) cannot be applied with a closed API teacher and why sequence-level SFT can, and state what information the student loses in the sequence-level case.
6. OpenThoughts' verification filter hurt a 7B student and helped a 32B student. Propose a causal explanation that could be tested, and describe the experiment.
7. Explain why discarding failed teacher traces avoids the probability-concentration effect of negative gradients, and what information is lost by discarding.
8. Orca 2 uses Prompt Erasing. Explain what the student must learn when the strategy instruction is removed, and why its MT-Bench score could still fall below an instruct baseline.

## Connections

- Previous: **ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing.** Covers prompt-side generation and the imitation-versus-capability critique; this chapter covers the response side when a stronger teacher writes the targets.
- Next: **ch-21 — Taxonomy-Driven and Textbook-Style Synthesis.** Controls coverage through a taxonomy instead of a teacher's reasoning traces.
- **ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data.** Trace generation methods and step-level data.
- **ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation.** Rejection sampling on a model's own samples and R1's cold start.
- **ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.** Uses of the traces this chapter discards.
- **ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control.** Measurement of the losses in §8.
- **ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data.** Stage map across labs, including logit and on-policy distillation.
- **ch-35a — Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters.** Side-by-side distillation-SFT settings and filters.
- **ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** The broader SFT-versus-RL evidence behind §8.2.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.** Full derivation of the gradient example.
- **ch-44 — Process Supervision and Verifiable Rewards.** Step-level checks for §5.5.

## Sources

- [[orca]] — Table 2 system messages, FLAN-v2 sampling, two-stage teacher schedule, BBH/AGIEval results, GPT-4-judge overestimation, Gudibande quote.
- [[orca-2]] — strategy selection, Prompt Erasing, dataset composition, reasoning average, MT-Bench, hallucination and safety rates.
- [[distilling-step-by-step]] — multi-task loss, prefixes, Tables 1–2, ANLI 80% result, training settings.
- [[deepseek-r1]] — pipeline, B.3.3 data description and Listing 4, Tables 15–16, generality evidence across stages, PRM attempt.
- [[deepseek-r1-recipe]] — Table 5 composition and Table 6 student settings.
- [[deepseek-r1-distill-synth]] — pointer to the R1 repository; its student-base list is outdated (Correction 15).
- [[bespoke-stratos]] — Stratos-17k composition, gpt-4o-mini filter, 32B and 7B results, training settings.
- [[sky-t1]] — NovaSky blog: data curation, GPT-4o-mini rewrite, training settings, data-mixture and model-size findings.
- [[openr1]] — OpenR1-Math-220k generation and filtering, verification-filter ablation, prefill lesson, CodeForces false positives.
- [[openr1-recipe]] — OpenR1-Qwen-7B and OpenR1-Distill-7B settings.
- [[open-thoughts]] — pipeline ablations, teacher comparison, scaling, reflection-removal, safety, verification by student size.
- [[scaling-reasoning-losing-control-mathif]] — HAcc/SAcc, 23-model results, controlled SFT/RL runs, IFEval comparisons.
- [[scaling-reasoning-losing-control-mathif-recipe]] — settings of the MathIF SFT runs.
- [[transferability-of-llm-reasoning]] — Transferability Index, Qwen3-14B controlled study, Qwen3-8B on/off-policy ablation (read from arXiv v2 text; the card has no numbers).
- [[gemma-2]] — logit-distillation objective and Table 6.
- [[qwen-3]] — strong-to-weak distillation phases and Table 21 (read from arXiv:2505.09388v1 text).
- [[qwen-qwq-traces]] — QwQ-32B as a teacher model, used by OpenThoughts and MathIF.
- [[dolphin]] — Orca-style dataset with refusal filtering and no stated method.
- [[qwen-2.5]] — Qwen2.5 license per model size.
- [[anthropic-distillation-attacks-report]] — terms-of-service violations in distillation from a closed API.
- [[teacher-output-licenses]] — quoted license statements for R1, Llama 3.1, QwQ, and dataset licenses.
