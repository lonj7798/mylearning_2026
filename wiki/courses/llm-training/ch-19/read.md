<!-- chapter: ch-19
     track: synthetic
     kind: content
     title: Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing
     deps: [ch-18]
     sources: [[self-instruct]], [[alpaca]], [[evol-instruct]], [[code-evol-instruct]], [[wizardcoder]], [[wizardmath]], [[magpie]], [[persona-hub]], [[prismatic-synthesis]], [[rephrasing-the-web]], [[humpback]], [[tulu-1-how-far-can-camels-go]], [[mammoth]], [[false-promise-imitating-proprietary-llms]], [[length-controlled-alpacaeval]], [[wildchat]], [[lmsys-chat-1m]]
     figures: figures/generation-methods.html
     revised: 2026-09 (generality revision)
-->

# Chapter 19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing

> **Core insight.** The generation methods in this chapter differ mainly in where the prompts come from: a 175-task seed pool (Self-Instruct), LLM rewrites of an existing set (Evol-Instruct), the aligned model's own distribution over user turns (Magpie), a persona collection built from web text (Persona Hub), or real documents (WRAP, Humpback). Several of them report gains with LLM-judged win rates (Evol-Instruct, Magpie, Humpback) or with benchmarks close to the target skill (WizardCoder, WizardMath). Broad imitation data raised crowdworker ratings but lowered Natural Questions accuracy of LLaMA 7B from 17 to 10 ([[false-promise-imitating-proprietary-llms]] Table 1), AlpacaEval win rates of one model ranged from 22.9% to 64.3% when only a verbosity instruction changed ([[length-controlled-alpacaeval]] §4.1), and a model trained on Alpaca prompts had an average NLL of 10.87 on WildChat user prompts versus 5.18 for a model trained on WildChat ([[wildchat]] Fig. 3). The prompt distribution a method samples from therefore has to be measured against real usage and against held-out capabilities, not only against a judge.
>
> **Guideline.** When the target is general instruction following and an aligned open-weight model is available, use Magpie-style extraction with filtering, because Magpie-Pro-300K-Filtered reached 25.08 AlpacaEval 2 length-controlled (LC) win rate on Llama-3-8B versus 22.92 for Llama-3-8B-Instruct ([[magpie]] Table 1); add targeted reasoning data, because the same SFT model scored 47.92 versus 71.72 on GSM8K (Table 3). When harder instructions for one domain are needed, evolve them and choose the number of rounds on a held-out dev set, because WizardCoder pass@1 peaked after three rounds ([[code-evol-instruct]] §3.2, Fig. 3). When the goal is pretraining efficiency on noisy web text, rephrase and keep real text in a 1:1 mix, because synthetic-only data raised Pile perplexity on many sub-domains ([[rephrasing-the-web]] §7 RQ2). In every case, report a length-controlled or benchmark-based score, a before/after suite outside the target skill, and a coverage check against held-out real user prompts, because judge scores reward length and style that imitation data transfers without the underlying capability.

## Corrections to the version you studied

1. "If yes, instances use an *input-first* template ... If no, *output-first*." → Reversed. Classification tasks use output-first generation (class labels first, then an input for each label) because input-first generation produced inputs biased toward one label; non-classification tasks use input-first ([[self-instruct]] arXiv:2212.10560v2 §2.2, Tables 7–8; [[excerpts/self-instruct]]).
2. "From ~252K raw generations the filter accepts ~52K instructions paired with ~82K instances" → the paper reports 52,445 instructions (11,584 classification) and 82,439 instances after filtering and no raw count; 252 is the size of the user-oriented human-evaluation set ([[self-instruct]] Table 1, §4.4).
3. "Applied to text-davinci-001-era GPT-3" → applied to vanilla GPT-3 ("davinci", 175B); InstructGPT001 is the comparison. On 119 unseen SuperNI tasks ROUGE-L rose from 6.8 to 39.9 (+33.1) versus 40.8 for InstructGPT001 ([[self-instruct]] §3, Table 3).
4. "Larger seeds accelerate the ROUGE filter's rejection rate" and the figure's "After ~50K accepted examples, ROUGE rejection rate spikes above 80%" → no seed-size or rejection-rate analysis is reported; the size analysis shows human-evaluation gains plateau after about 16K instructions ([[self-instruct]] §4.5, Fig. 7).
5. "Drop instances where input == output, outputs are too short/long" → identical instances and instances with the same input but different outputs are removed; heuristics remove instructions that are too long or too short and outputs that repeat the input ([[self-instruct]] §2.2).
6. Alpaca "52K accepted instructions (no new filter)" → the released code filters instructions by length, blacklisted words, format, and ROUGE-L F-measure > 0.7, does not check outputs, generates one instance per instruction, and has no classification split ([[alpaca]], generate_instruction.py @761dc5b L52–101, L196).
7. "Ablating the teacher to GPT-3.5 barely changes downstream skills; ablating to GPT-4 introduces GPT-4's refusal patterns" → Alpaca reports no teacher ablation, and text-davinci-003 is itself a GPT-3.5-series model; the blog states that Alpaca reflects the dataset's style and gives shorter answers than ChatGPT ([[alpaca]], blog).
8. Evol-Instruct elimination "same-or-similar / sorry / punctuation-only / copies the input verbatim" → (1) no information gain, judged by ChatGPT; (2) contains "sorry" and is shorter than 80 words; (3) only punctuation and stop words; (4) copies words from the evolving prompt such as "given prompt"; failed instructions return to the pool ([[evol-instruct]] §3.2).
9. "Mutation — ... a rarer domain, same overall topic" → in-breadth evolving writes a new instruction in the same domain that is "even more rare", with similar length and difficulty ([[evol-instruct]] Example 3.3).
10. "~250K evolved instructions after filtering. SFT on LLaMA-7B/13B/70B" and "long-tail complexity histogram in ... Figure 3" → 250k instructions in total over 4 rounds, 70k sampled for SFT of LLaMA 13B (v3; 7B in v1); 65B and 70B are ablations; the paper reports mean difficulty per round (Fig. 5a, App. I Table 3), not a histogram ([[evol-instruct]] §4.2, Table 2).
11. "GPT-4 will not produce a problem it cannot solve" → WizardLM and WizardCoder used gpt-3.5-turbo as the evolver, and no cited paper measures a teacher-ceiling effect; in WizardCoder a GPT-4 evolver gave 62.2 versus 59.8 HumanEval pass@1 for GPT-3.5 at 15B ([[evol-instruct]] §4.2; [[code-evol-instruct]] Table 4).
12. WizardMath operators "reduce constraints, shorten the chain, make arithmetic easier; compose with another concept, require multiple solution steps" → downward evolution revises a question to lower difficulty or writes an easier question on a different topic; upward evolution adds constraints, concretizes, or increases reasoning; GPT-4 runs 5 rounds (2 downward, 3 upward) ([[wizardmath]] arXiv:2308.09583v3 §3.1; [[excerpts/wizardmath]]).
13. "downward-only, upward-only, and bidirectional evolution all underperform the bidirectional blend" and the claim that a broader difficulty spectrum makes the model recognize problem level → Table 6 (Mistral-7B SFT): original data 59.7 GSM8K / 15.1 MATH; two downward rounds 74.5 / 34.7; three upward rounds 78.6 / 42.5; all five rounds 81.2 / 46.2. The paper does not give a difficulty-recognition explanation ([[wizardmath]] Table 6).
14. "the IRM × PRM reward product that ch-26 will unpack" → the product reward is defined in WizardMath §3.3; process reward models are taught in ch-44; ch-26 is tool and function-calling data.
15. WizardCoder "4. Deepen problem complexity (time/space bounds, edge cases, misleading wordings). 5. Require a specific language or library." → the five heuristics are: add constraints (about 10 words); replace a common requirement with a less common, more specific one; add reasoning steps; provide erroneous code as misdirection; propose higher time or space complexity requirements ([[code-evol-instruct]] §3.1).
16. "Yield = ~78K evolved pairs after three rounds and the standard elimination filter", "HumanEval+ 50.6", "operator-by-operator HumanEval ablation" → 78k is the cumulative seed plus three merged rounds; no elimination filter and no per-heuristic ablation are reported; the EvalPlus greedy result for 15B is 59.8 versus Claude-Plus 53.0 and Bard 44.5 (the sentence does not say whether HumanEval or HumanEval+ is meant), and 50.6 does not appear ([[code-evol-instruct]] v1 §4.5, v2 §4.3).
17. "Magpie (2025)" and "each method was invented in reaction to the previous one's limit" → Magpie (arXiv 2024-06, ICLR 2025) and Persona Hub (arXiv 2024-06) appeared in the same month, and Humpback (2023-08) predates WRAP (2024-01) ([[magpie]]; [[persona-hub]]; [[humpback]]).
18. "The 3M → 300K curation is where the method lives or dies" and "sharp, low-entropy distribution on the pre-query prefix" → for Magpie-Air, 3M-Raw scored 22.96 AlpacaEval 2 LC and 300K-Filtered 22.66; for Magpie-Pro, 300K-Filtered 25.08 versus 1M-Raw 24.16 ([[magpie]] Table 12); the paper hypothesizes implicit memorization of user turns (§2.1 Remark) and does not measure entropy.
19. "Persona-to-Persona ... list 10 people who might interact with them professionally / personally / antagonistically" → the prompt is "Who is in close relationship with the given persona?", run for six iterations ([[persona-hub]] §2.2).
20. "Two personas that are 80% similar produce problems that are only 40–50% similar", "does not saturate at 1M; headroom comparable to raw data scaling", "dedup-threshold ablation", "full 1B gated" → the text states that at persona similarity 0.9 most problem similarities are 0.6–0.75 (Fig. 10c); the scaling trend "generally aligns with the scaling law" with no raw-data comparison; no dedup ablation is reported (a 0.5 threshold is suggested for more diversity); the initial release is 200,000 personas and "gated" is not stated ([[persona-hub]] §1, §2.3, §4.1.2).
21. "That is a 40× data-efficiency multiplier and a 4× parameter-efficiency multiplier" → 15% of C4 is 1/0.15 ≈ 6.7× fewer real tokens, and 1.3B/350M ≈ 3.7× (derived from [[rephrasing-the-web]] §1). The ">50% perplexity" figure matches the ACL 2024 abstract and body; the arXiv v1 abstract says "more than 10%" (conflict between versions; [[excerpts/rephrasing-the-web]]).
22. WRAP "rephrases longer than 300 tokens drop information", "Q/A style is denser in extractable facts per token", "reproducible against held-out Mistral rephrases" → rephrasing inputs longer than 300 tokens often lost information; the authors attribute the gains to style that reflects downstream evaluation style and to higher quality; the leakage analysis compares SimCSE cosine similarity of real–synthetic pairs with baselines ([[rephrasing-the-web]] §3.1, §7 RQ3).
23. Humpback "self-curation beats external curation ... better-calibrated on its own outputs than a fresh reward model" and "prompt an aligned seed model" → on a 250-example dev set, curation precision/recall was 0.44/0.09 for the seed model M0, 0.52/0.44 for M1, and 0.88/0.92 for GPT-4; no reward model was compared; instructions come from a backward model fine-tuned on 3,200 seed pairs ([[humpback]] §2.2, App. B Table 9).
24. The §9 comparison table (costs per 1K, "Diversity (1 − avg cos)" 0.42–0.72) and "Persona-Hub (0.72) is the biggest single-method diversity gain" → no source reports these values or a common diversity measurement. Reported costs: Self-Instruct about $600 for generation (App. A.2); Alpaca less than $500 ([[alpaca]] blog); Magpie $0.12 (Air) and $1.1 (Pro) per 1,000 instances ([[magpie]] §3.4); WRAP about 25K A100 hours for 85B tokens ([[rephrasing-the-web]] §9.1). Embedding-based diversity can rank datasets opposite to out-of-distribution accuracy ([[prismatic-synthesis]] Table 2).
25. "Ch-18 set up the generator–verifier stack: who calls the teacher, how rollouts are framed" → ch-18 teaches the six-stage generate, filter, deduplicate, verify, select, mix loop.

## Why this chapter matters for a general-purpose model

The SFT datasets compared in this chapter (Alpaca, WizardLM, WizardCoder, WizardMath, Magpie, Persona Hub math, Humpback) were produced by the methods below, and WRAP applies the same idea to pretraining text. A generation method fixes two distributions: the distribution of prompts (which tasks, topics, formats, languages, and difficulty levels appear) and the distribution of responses (whose style and knowledge the student imitates). A general-purpose model needs prompts that cover what users ask and responses that transfer capability rather than only style. This chapter sits at the generate stage of the ch-18 loop, before quality and diversity selection (ch-22) and before preference optimization and RL. For each method it gives the mechanism, the evaluation its authors used, and the evidence on breadth and narrowing. Sections §2–§8 keep the method order of the version you studied; the unsourced comparison table is replaced by §9 (imitation versus capability), §10 (judge length bias), §11 (coverage of real user prompts), and §12 (an evidence table for composing methods).

## §1 Terms

- **SuperNI**: Super-NaturalInstructions, a collection of NLP tasks with human-written instructions; Self-Instruct evaluates on 119 of its tasks that were not used for training.
- **NLL**: negative log-likelihood, −log p(x), averaged over tokens or examples; lower values mean the model assigns higher probability to the data.
- **AE2 LC**: AlpacaEval 2 length-controlled win rate (§10). **RtA**: refuse-to-answer rate, the share of prompts the model declines (higher is safer for jailbreak prompts).
- **Seed set**: human-written examples that condition generation (175 tasks in Self-Instruct, 52k Alpaca instructions for Evol-Instruct).
- **Generator / evolver**: the model that writes new prompts; **response generator**: the model that writes the training targets. They can differ (Magpie App. F.5 swaps only the response generator).
- **Prompt distribution**: the probability of each kind of prompt appearing in the dataset. **Coverage** is how much probability the trained model assigns to prompts from a reference distribution; WildChat measures it as the NLL of held-out user prompts (§11).
- **Imitation data**: responses written by a stronger model and used as SFT targets for a weaker one ([[false-promise-imitating-proprietary-llms]] Abstract).
- **Judge win rate**: the probability that an LLM judge prefers the model's response over a baseline response, averaged over a fixed prompt set; **LC win rate** removes the part explained by the length difference (§10).

| Method (arXiv v1) | Prompt source | Response source | Main evaluation in the source |
|---|---|---|---|
| Self-Instruct (2022-12) | 175 seeds + GPT-3 davinci | GPT-3 davinci | SuperNI unseen tasks; 252 novel instructions, human-rated |
| Alpaca (blog 2023-03) | 175 seeds + text-davinci-003 | text-davinci-003 | 5 authors, blind pairwise vs text-davinci-003 |
| Evol-Instruct / WizardLM (2023-04) | Alpaca 52k rewritten by gpt-3.5-turbo | gpt-3.5-turbo | 9 benchmarks incl. AlpacaEval, MT-Bench, WizardEval (GPT-4 judged) |
| WizardCoder (2023-06) | Code Alpaca 20k rewritten by gpt-3.5-turbo | gpt-3.5-turbo | HumanEval, HumanEval+, MBPP, DS-1000, MultiPL-E |
| WizardMath (2023-08) | GSM8k + MATH train rewritten by GPT-4 | GPT-4-0613 | GSM8k, MATH |
| Humpback (2023-08) | backward model over ClueWeb text | the web text itself | AlpacaEval, human preference, MMLU |
| WRAP (2024-01) | C4 documents | Mistral-7B-Instruct rephrase | Pile perplexity; 13 zero-shot tasks |
| Magpie (2024-06) | Llama-3-Instruct given only the user header | same model family | AlpacaEval 2 LC, Arena-Hard; Open LLM Leaderboard tasks |
| Persona Hub (2024-06) | GPT-4 prompted with a persona | gpt-4o solutions (math) | MATH (out of distribution), synthetic test |

Panel D of [figures/generation-methods.html](figures/generation-methods.html) lists the pipeline steps and the main reported numbers for each method in this table, so methods can be compared step by step while reading §2–§8.

## §2 Bootstrap: Self-Instruct

**Definition.** Self-Instruct generates instruction data from a language model that is conditioned on a small pool of tasks and adds accepted outputs back to that pool ([[self-instruct]]).

**Problem.** Human-written instruction sets are "limited in quantity, diversity, and creativity" (Abstract); the measurable target was zero-shot accuracy on unseen tasks without private annotation.

**Mechanism** (§2.2):
1. Start the pool with 175 tasks (1 instruction and 1 instance each).
2. Sample 8 instructions as in-context examples, 6 human-written and 2 model-generated, and let the model continue "Task 9:".
3. Ask the model whether the new instruction is a classification task (a task with a small, limited label space).
4. Generate instances. For classification tasks, generate the class labels first and then an input for each label (output-first), because input-first generation "can generate inputs biased toward one label" (for grammar error detection it "usually generates grammatical input"). For other tasks, generate the input first.
5. Filter: add an instruction only if its ROUGE-L similarity with every existing instruction is below 0.7; drop instructions with keywords such as image, picture, graph; drop identical instances and same-input/different-output instances; drop instructions that are too long or short and outputs that repeat the input.

**Formula.** ROUGE-L uses the longest common subsequence (LCS) of two token sequences a (candidate) and b (reference):

```
P = LCS(a, b) / |a|     R = LCS(a, b) / |b|     F = 2 P R / (P + R)
```

`|a|`, `|b|` are token counts; P is precision, R recall, F the F-measure. The Alpaca code compares F against 0.7 (`generate_instruction.py` L196); the Self-Instruct paper says "ROUGE-L similarity" without naming the variant.

**Worked example.** Candidate "write a short poem about the ocean" (7 tokens) against pool instruction "write a poem about the sea" (6 tokens): LCS = "write a poem about the" = 5, P = 5/7 = 0.714, R = 5/6 = 0.833, F = 0.769 > 0.7, so the candidate is rejected. Candidate "explain why the sea is salty" (6 tokens): LCS = "the sea" = 2, F = 0.333, accepted. The filter rejects lexical near-copies; it does not reject a new instruction that asks for the same skill in different words. Panel A of [figures/generation-methods.html](figures/generation-methods.html) computes this F-measure for any two instructions and shows the accept or reject decision.

**Evidence.** GPT-3 davinci (175B), fine-tuned through the OpenAI API for 2 epochs on 52,445 instructions and 82,439 instances: SuperNI ROUGE-L 6.8 → 39.9, versus 40.8 for InstructGPT001 (Table 3). On 252 newly written user-oriented instructions rated by their authors, the model is 5% behind InstructGPT001 when "acceptable with minor imperfections" counts as valid (§4.4). An audit of 200 samples found 92% valid instructions but only 54% with all fields valid (Table 2). Human-evaluation gains "almost plateau after 16K" instructions, and SuperNI gains plateau "at around hundreds" (§4.5). Cost: about $600 for generation and $338 for fine-tuning (App. A.2–A.3). **Result (single study).**

**Conditions and limits.** Instance quality is not verified: 46% of audited examples had at least one invalid field. The seed pool has average ROUGE-L 0.21 to its closest SuperNI test instruction and 0.34 to the user-oriented set, with one identical instruction (App. A.1). When the same released data trained LLaMA 13B in [[tulu-1-how-far-can-camels-go]], all five capability benchmarks fell below the base model: MMLU 42.3 → 30.4, GSM 14.5 → 11.0, BBH 39.3 → 30.7, TyDiQA 43.2 → 41.3, Codex-Eval P@10 28.6 → 12.5 (Table 3).

**Implication.** The 252-instruction human evaluation on novel tasks is closer to a generality measurement than a judge win rate, and the Tülu 1 result shows that data which helped a 175B base can lower accuracy for a different 13B base.

## §3 Alpaca as a low-cost replication

[[alpaca]] (blog 2023-03) changed four parts of Self-Instruct: text-davinci-003 as generator, a new prompt requesting 20 instructions per call, one instance per instruction, and no classification split (README "Data Generation Process"). The released code rejects instructions with ≤ 3 or > 150 words, blacklisted words, non-ASCII starts, or ROUGE-L F > 0.7, and does not check outputs (`generate_instruction.py` @761dc5b). LLaMA 7B was fine-tuned on 52K examples for 3 epochs at LR 2e-5 with prompt tokens masked; data cost less than $500 and training less than $100 (3 hours on 8 × 80GB A100) (blog; README). The only evaluation was 5 student authors comparing outputs blind against text-davinci-003 on the Self-Instruct evaluation set: 90 wins versus 89 (blog). The authors state that the evaluation "may be limited in scale and diversity" and that hallucination is common "even compared to text-davinci-003".

**Breadth evidence.** At 13B in Tülu 1, Alpaca data changed MMLU 42.3 → 45.0, GSM 14.5 → 9.5, BBH 39.3 → 36.6, TyDiQA 43.2 → 31.1, and Codex-Eval 28.6 → 29.9 ([[tulu-1-how-far-can-camels-go]] Table 3): MMLU rose 2.7 and Codex-Eval 1.3, the other three fell, and multilingual QA lost 12.1 points. Alpaca's 52,002 prompts have 1 language and 19.67 user tokens on average, versus 68 languages and 295.58 tokens in WildChat ([[wildchat]] Table 1). **Implication.** A low-cost replication inherits the teacher's response style and the seed pool's prompt distribution; a pairwise judgment against the teacher does not test either.

## §4 Evolution: Evol-Instruct, WizardCoder, WizardMath

### §4.1 Evol-Instruct

**Definition.** Evol-Instruct uses an LLM to rewrite each instruction into a more complex or rarer one, round by round, and trains on all rounds ([[evol-instruct]], WizardLM paper).

**Problem.** Human-written sets contain few high-difficulty instructions; mean difficulty (1–10, ChatGPT judge) is 3.00 for Alpaca and 4.63 for ShareGPT (App. I Table 3).

**Mechanism** (§3.2, §4.2):
1. For each instruction in each round, sample one of six prompts with equal probability.
2. In-depth (five prompts): add constraints, deepening, concretizing, increased reasoning steps, complicating input (the last uses in-context XML, SQL, Python, HTML, shell, and JSON examples). Each rewrite may add only 10 to 20 words. The add-constraints prompt reads: "Please add one more constraints/requirements into #Given Prompt#" (Example 3.1).
3. In-breadth (one prompt): "This new prompt should belong to the same domain as the #Given Prompt# but be even more rare" (Example 3.3).
4. Generate the response with gpt-3.5-turbo from the instruction alone.
5. Eliminate failures (no information gain judged by ChatGPT; "sorry" and fewer than 80 words; only punctuation and stop words; copied prompt words); failed instructions return to the pool.
6. After M = 4 rounds on 52k Alpaca seeds (250k instructions, 624k API requests), sample 70k from all rounds for SFT.

**Evidence.** LLaMA 13B on the 70k sample averages 58.96 over nine benchmarks versus 54.60 for Vicuna-13b (70k ShareGPT) and 43.44 for a retrained Alpaca-13b (Table 1). Mean difficulty rises from 5.48 (round 1) to 7.08 (round 4) (Table 3). With the evolver replaced by Llama-2-70B-Chat the average is 56.27 (Table 2). **Result (single study).**

**Conditions and limits.** The seed changes which skills improve: a ShareGPT seed raised the nine-benchmark average to 61.87 but lowered GSM8k from 37.15 to 31.46, which the authors attribute to 4.3% math instructions in ShareGPT versus 11.8% in Alpaca (§4.5, Table 2). Vicuna-13b scored higher on TruthfulQA (52.68 vs 50.55) (Table 1). No contamination check is reported (card Verification). Three of the nine metrics are GPT-4 judged.

### §4.2 WizardCoder: code heuristics and round selection

[[code-evol-instruct]] (WizardCoder paper, arXiv 2023-06) replaces the six prompts with one template, "Please increase the difficulty of the given programming test question a bit", and five heuristics printed in §3.1:

```
(1) Add new constraints and requirements to the original problem, adding approximately 10 additional words.
(2) Replace a commonly used requirement in the programming task with a less common and more specific one.
(3) If the original problem can be solved with only a few logical steps, please add more reasoning steps.
(4) Provide a piece of erroneous code as a reference to increase misdirection.
(5) Propose higher time or space complexity requirements, but please refrain from doing so frequently.
```

Deepening, complicating input, and in-breadth evolving were removed (v1 §3.1). After each round, all evolved data is merged with the Code Alpaca seed (about 20k); cumulative sizes are 38k, 58k, 78k, 98k after rounds 1–4, and the 78k set is final (v1 §4.5). **Complexity versus quantity** (Table 5, HumanEval pass@1, each round trained alone): at matched sample counts, seed 20.0k → 45.7, round 1 18.8k → 56.1, round 2 → 53.0, round 3 → 54.3, round 4 → 51.2; at a matched 2.3M tokens, 44.5, 51.8, 52.4, 50.0, 49.4. **Worked reading.** Round 1 at 94% of the seed's sample count (18.8/20.0) scores 10.4 points higher, so the gain is not explained by more examples; the decline after round 2 at matched tokens shows that more rounds are not monotonically better.

**Held-out breadth.** WizardCoder 34B is above CodeLlama-Instruct-34B in all 8 MultiPL-E languages (Table 2); WizardCoder 15B has HumanEval 57.3 and MBPP 51.8 (Table 1). **Measurement error.** v1 chose the number of rounds and the final model by HumanEval pass@1, a reported test benchmark; v2 describes an external dev set (MBPP-400) instead (v1 §3.2; v2 §3.2, §5). Decontamination retrieves the top 5 training samples per test item with gte-large embeddings and lets GPT-4 decide matches (App. C). Effects on non-code tasks are not reported.

### §4.3 WizardMath: downward and upward evolution

[[wizardmath]] (arXiv 2023-08; v3 is the ICLR 2025 version) adds a downward direction: "revising high difficulty questions to lower difficulty, or ii) producing a new and easier question with another different topic", alongside upward evolution by adding constraints, concretizing, and increasing reasoning. GPT-4 evolves each GSM8k and MATH training question for 5 rounds (2 downward, 3 upward), producing 448k unique instructions, of which 30k were removed as possible contamination, leaving 418k with GPT-4-0613 step-by-step answers (v3 §3.1, §4.1; [[excerpts/wizardmath]]).

**Worked example** (Table 6, Mistral-7B SFT, pass@1). Original 7.5k data: 59.7 GSM8K, 15.1 MATH. Adding two downward rounds: +14.8 and +19.6. Adding three upward rounds: +18.9 and +27.4. Adding all five: +21.5 and +31.1 (81.2 and 46.2; the §4.3 text prints 46.5). The combined gain is larger than either direction alone but smaller than their sum (14.8 + 18.9 = 33.7 > 21.5), which is consistent with the two directions partly teaching the same thing (Interpretation). RL with a process reward model and an instruction reward model then raises Mistral-7B from 82.8 / 48.1 to 90.7 / 55.4 (Table 3); both reward models are trained on GPT-4-0613 labels (§4.1), and process supervision is covered in ch-44.

**Narrowing.** For the 2023 release, [[mammoth]] reports WizardMath-70B at 20.0 on AQuA versus 40.9 for Llama-2-70B, and 13.2 on SAT-Math versus 51.3 (Tables 3–4), while GSM8K is 81.6 (Table 3). **Implication.** Domain evolution raises in-domain accuracy, and out-of-domain tasks in the same broad area can fall; both need to be in the evaluation.

## §5 Extraction: Magpie

**Definition.** Magpie generates a user instruction by sending an aligned model only the part of its chat template that precedes the user message, then generates the response with the full template ([[magpie]] §2).

**Problem.** Open data-creation methods need human seeds or hand-written prompts, which limits scale and scope (Abstract).

**Mechanism** (§2.1, App. E.1):

```
x = T_pre-query ⊕ q ⊕ T_post-query
T_pre-query  = <|start_header_id|>user<|end_header_id|>
T_post-query = <|eot_id|><|start_header_id|>assistant<|end_header_id|>
Step 1: input T_pre-query only; sample until end-of-sequence  → instruction q
Step 2: input T_pre-query ⊕ q ⊕ T_post-query; greedy decoding → response
```

`q` is the generated user query and ⊕ is concatenation. Instruction sampling mixes temperature {1.0, 1.1, 1.2, 1.25} and top-p {1.00, 0.995, 0.990}; higher values lowered quality slightly and raised difficulty and diversity (App. D.3). The authors hypothesize that the model memorized user turns implicitly even when instruction loss was masked during alignment (§2.1 Remark; Interpretation). Eight filter metrics are defined: input length, output length, task category, input quality, input difficulty, minimum neighbor distance (all-mpnet-base-v2 + FAISS), reward r* from FsfairX-LLaMA3-RM-v0.1, and reward difference r* − r_base, where r_base is the reward of a base-model response (App. C, Table 5). The output-length filter keeps the longest responses.

**Evidence** (Llama-3-8B-Base SFT, 2 epochs; Magpie sets have 300K conversations, baseline sizes as listed). AlpacaEval 2 LC: Magpie-Pro-300K-Filtered 25.08, Magpie-Air-300K-Filtered 22.66, WildChat 652K 14.62, Evol Instruct 143K 8.52, Self-Instruct (Llama-3) 100K 7.21, Llama-3-8B-Instruct 22.92 (Table 1). Arena-Hard: 18.9, 14.9, 8.7, 5.1, 4.0, 20.6. Compute: Air 206 GPU hours, Pro 614 GPU hours on 4 × A100; $0.12 and $1.1 per 1,000 instances (§1, §3.4). **Result (single study).**

**Conditions and limits.**
- Reasoning: Open LLM Leaderboard average 61.58 versus 66.13 for Llama-3-8B-Instruct; GSM8K 47.92 versus 71.72 (Table 3). Adding 150K math, code, and reasoning data gives GSM8K 63.08.
- Filtering versus scale: Air 3M-Raw 22.96 LC versus 300K-Filtered 22.66; Pro 300K-Filtered 25.08 versus 1M-Raw 24.16 (Table 12). No single filter configuration was best on all benchmarks (Table 13).
- Response generator: Qwen2-7B-Instruct responses on the same instructions lowered LC from 22.66 to 15.01 (Table 14), so part of the gain belongs to the response model.
- Inherited prompt prior: "over half" of Magpie-Pro instructions are information seeking (§3.2), and 99.128% (Air) and 99.347% (Pro) are labeled safe by Llama-Guard-2 (Table 6). The coverage analysis compares Magpie-Pro with Alpaca, Evol Instruct, and UltraChat by t-SNE (§3.1), all synthetic; no comparison with real user prompts is reported (§11).
- Safety behavior: the SFT model refuses jailbreaks less often than Llama-3-8B-Instruct (RtA 0.80 vs 0.93, Table 15).

**Implication.** Extraction reproduces the aligned model's distribution over user turns. The resulting SFT model does well on instruction-following judges and trails on math reasoning, which the authors attribute to a small share of reasoning instructions (§4.2, Interpretation). Unsafe prompts are under 1% of Magpie data by Llama-Guard-2, while 5% of LMSYS-Chat-1M conversations are flagged by the OpenAI moderation API (§11); the classifiers differ, so this is a lead to check rather than a measured gap.

## §6 Persona conditioning: Persona Hub

**Definition.** Persona-driven synthesis inserts a persona description into a data-synthesis prompt so that the LLM writes data from that persona's perspective ([[persona-hub]] Abstract).

**Mechanism** (§2.1–§2.3):
1. Text-to-Persona: ask an LLM "Who is likely to [read|write|like|dislike|...] the text?" for RedPajama v2 documents.
2. Persona-to-Persona: ask "Who is in close relationship with the given persona?" for six iterations.
3. Deduplicate with 1-gram MinHash (signature 128, threshold 0.9), then remove personas with embedding cosine similarity above 0.9 (0.5 is suggested when diversity matters more than count), then apply heuristic quality filters. Result: 1,015,863,523 personas.
4. Prompt a model with a persona plus a task (zero-shot, few-shot, or persona-enhanced few-shot).

**Evidence.** GPT-4 wrote 1.09M math problems from 1.09M personas with no MATH instances; gpt-4o wrote solutions; Qwen2-7B fine-tuned on 1.07M reaches 64.9% on MATH with greedy decoding, versus 64.5% for gpt-4-turbo-0125-preview (Table 2). Two math experts judged 7 of 200 challenging problems invalid (96.5% valid). Similarity study (100 persona pairs per level, temperature 0, text-embedding-3-small): problem similarity "tends to be correlated with but lower than" persona similarity, rises when the prompt adds a topic constraint, and at persona similarity 0.9 mostly falls between 0.6 and 0.75 (§4.1.2, Fig. 10). **Result (single study).** SFT hyperparameters are not reported.

**Conditions and limits.** Lower embedding similarity is not the same as coverage of different skills. In [[prismatic-synthesis]] (10k math samples from Qwen2.5-72B-Instruct), a set built from 0.1k seeds and 10k personas had higher embedding Vendi score (24.49 vs 22.03; the Vendi score is the exponentiated entropy of the eigenvalues of a similarity matrix, and equals the number of items when all items are dissimilar) but lower out-of-distribution accuracy (38.15 vs 54.77) than a set from 10k seeds and 0.1k personas (Table 2). Persona Hub's own authors state that the in-distribution synthetic-test result "should be taken as a reference only" (§4.1.2). **Implication.** Persona conditioning varies surface context; the evidence above shows that varying the underlying problems (seeds) mattered more for transfer in math. The gradient-based diversity measure behind this comparison is taught in ch-22.

## §7 Rephrasing: WRAP

**Definition.** Web Rephrase Augmented Pretraining (WRAP) prompts an instruction-tuned model to paraphrase web documents in specified styles and pretrains on real and rephrased text together ([[rephrasing-the-web]]; [[excerpts/rephrasing-the-web]]).

**Mechanism** (§3.1–§3.2):
1. Split C4 documents into chunks of about 300 tokens with the NLTK sentence splitter; longer inputs "often led to a loss of information".
2. Rephrase each chunk with frozen Mistral-7B-Instruct in one of four styles: Easy, Medium (Wikipedia-like), Hard (terse), Q/A (conversation question-answer format).
3. Sample real and synthetic data 1:1, so each document is seen once raw and once rephrased.
4. Pretrain decoder-only models (128M, 350M, 1.3B) at sequence length 1024.

**Evidence.** The ACL abstract reports a speed-up of about 3×, Pile perplexity improved by more than 50% on average, and more than 2% higher zero-shot accuracy across 13 tasks; the arXiv v1 abstract says "more than 10%" for perplexity while its body says 50% (**conflict between versions**). A 350M model trained on real plus synthetic rephrases of 15% of C4 outperforms a 1.3B model trained on all of C4 (§1).

**Worked example: reading an average over 13 tasks** (Table 1, 1.3B, WRAP rows averaged over 3 runs; §7 RQ2 refers to the Table 1 synthetic data as generated with the QA prompt). Synthetic+C4 with 85B real tokens averages 52.3 on 8 general tasks versus 50.3 for Full C4 with 170B. Per-task differences: ARC-E +2.8, BoolQ +8.0, Winogrande −0.1, PIQA −0.1, HellaSwag −0.4, TruthfulQA +7.1, OBQA −1.5, LogiQA +0.2. Two tasks contribute (8.0 + 7.1)/8 = 1.9 of the 2.0-point gap, and 4 of 8 tasks are lower. On 5 specialized-knowledge tasks the average is 44.0 versus 42.4, and TinyLlama trained on 1T tokens scores 44.6. The authors conclude that synthetic data "can not impart 'new knowledge'. It can only help pre-train faster" (§5.2).

**Conditions and limits.**
- Style matched to the evaluation: the authors attribute gains to style "that closely reflects downstream evaluation style" (Abstract); Fig. 1 (b, c) states Q/A-style rephrasing, §7 RQ2 describes the Table 1 synthetic data as QA-prompt data, and the zero-shot tasks are question answering. On 1.3B / 150B-token runs, the Medium style alone averaged 47.9 and Q/A alone 51.0 (Table 2). This is a benchmark-format confound for generality (Interpretation).
- Real data: synthetic-only training degraded Pile perplexity on many sub-domains, which the authors attribute to clean synthetic text lacking special characters and structure of real web text (§7 RQ2, Fig. 4).
- Rephraser: Qwen-1.8B and Mistral-7B rephrases gave lower perplexity than Vicuna-13B; every rephraser beat real C4 alone (RQ1, Fig. 3).
- Cost: about 25K A100 hours for 85B tokens at 3M tokens per hour (§9.1).

**Implication.** Rephrasing changes style and not information content; measure breadth with per-task results and with perplexity on domains unlike the rephrase style. Larger-scale textbook-style synthesis is covered in ch-21.

## §8 Backtranslation: Humpback

[[humpback]] (arXiv 2023-08) fine-tunes LLaMA on 3,200 Open Assistant seed pairs in the reverse direction to get a backward model p(x|y), generates one candidate instruction for each of 502k ClueWeb segments, and lets a seed-trained model rate each pair on a 5-point prompt; after two iterations the top-score set A5 holds 41,821 pairs (§2.2–§2.3, Table 1). LLaMA 65B trained on seed plus curated data reaches 83.71% AlpacaEval win rate over text-davinci-003 (Table 3). Uncurated augmented data did not raise win rate as its size grew (§3.3, Fig. 2). On an author-labeled 250-example dev set with 20% positives, curation precision/recall was 0.44/0.09 for M0, 0.52/0.44 for M1, and 0.88/0.92 for GPT-4 (App. B, Table 9). **Implication.** Responses come from human web text rather than a teacher, which avoids transferring a teacher's style; the self-curator discards most good pairs at iteration 0 (recall 0.09), and evaluation is judge-based.

## §9 Imitation versus capability

[[false-promise-imitating-proprietary-llms]] (arXiv 2023-05) fine-tuned GPT-2 1.5B and LLaMA 7B and 13B on 0.3M to 150M tokens of ChatGPT outputs (ShareGPT-Mix: about 50K ShareGPT, 27K HC3, 10k Discord examples) with loss on outputs only ([[excerpts/false-promise-imitating-proprietary-llms]]). Crowdworkers rated about 70% of imitation outputs as equal to or better than ChatGPT's (Fig. 1). On 5-shot MMLU, 3-shot Natural Questions, and 0-shot HumanEval, broad imitation models "do not improve (or even decline) in accuracy as compared to the base model, even when adding additional imitation data" (§4.3, Fig. 4 top).

**Worked example** (Table 1, NQ accuracy). LLaMA 7B: base 17, ShareGPT-Mix 10 (−7), NQ-synthetic 22 (+5). LLaMA 13B: 20, 15 (−5), 27 (+7). ChatGPT: 31. The targeted set has 6,000 examples generated from 10 seed QA pairs; the broad models were trained on up to 150M tokens. At 150M tokens, the imitation model uses a list when ChatGPT does in 81% of cases (13% for the base) and its length correlation with ChatGPT rises from −0.11 to 0.62 (Table 2). Increasing base size raised accuracy more than increasing imitation data (Fig. 4 bottom).

**Interpretation and limits.** The authors conclude that fine-tuning on imitation data transfers style, not knowledge, and that regressions may come from "distribution shift and tension between the conversational-style fine-tuning data and the downstream benchmarks" (§4.3). GPT-4 as evaluator showed the same trends as crowdworkers (§4.4), so a GPT-4 judge does not detect the gap. [[tulu-1-how-far-can-camels-go]] reports a complementary result at 13B: GPT4-Alpaca data gave 63.1% AlpacaEval but TyDiQA 23.5 versus 43.2 for the base, and win rate correlates with the average number of unique response tokens at r = 0.96 (§5.4, Table 3). The authors of Tülu 1 argue that imitation data helps when it covers diverse skills (§6; Interpretation). ch-29e treats this result for instruction tuning in general.

## §10 Judge-score length bias as a confound

**Definition.** AlpacaEval compares a model's responses with a baseline's on 805 fixed instructions using a GPT-4-turbo judge; the win rate is the mean preference probability ([[length-controlled-alpacaeval]] §2).

**Problem.** Prompting gpt4_1106_preview to be verbose or concise moved its win rate from 22.9% to 64.3% (§4.1). Two methods in this chapter select or produce long responses (Magpie keeps the longest responses; imitation data raises length correlation with ChatGPT), so a raw win-rate gain can come from length.

**Formula** (§3, Eq. 1–2):

```
q(y=1) = logistic( θ_m − θ_b + φ_{m,b} · tanh( Δlen / std(Δlen) ) + (ψ_m − ψ_b) γ_x )
winrate_LC = 100 · E_x[ logistic( θ_m − θ_b + (ψ_m − ψ_b) γ_x ) ]
```

θ_m, θ_b are model terms; Δlen = len(z_m) − len(z_b) is the length difference between the model's and baseline's responses; φ_{m,b} is the fitted length coefficient; γ_x is the difficulty of instruction x and ψ its per-model weight; the LC win rate sets the length term to zero.

**Worked example** (illustrative parameter values, not fitted to any model). Let θ_m − θ_b = 0, the instruction term be 0, φ = 1.0, and responses one standard deviation longer than the baseline. Raw predicted preference = logistic(tanh(1)) = logistic(0.762) = 0.682, so the raw win rate is 68.2%; LC = logistic(0) = 50%. With θ_m − θ_b = 0.5, raw = logistic(1.262) = 77.9% and LC = 62.2%. Panel C of [figures/generation-methods.html](figures/generation-methods.html) recomputes both rates for chosen θ, φ, and length difference.

**Evidence.** Length control lowered the normalized standard deviation across verbosity prompts from 25% to 10% and raised Spearman correlation with Chatbot Arena from 0.94 to 0.98 (38 models; bootstrap p = 0.07 versus AlpacaEval) (§4.1–§4.2, Table 1). The largest rank losses under length control were open models trained with RLHF (§4.2). A truncation attack raised a GPT-4 LC score from 3.7 to 25.9 without regularization and to 12.2 with it (§4.3).

**Limits.** LC removes one mediator (length); list use, tone, and factual errors remain, and Chatbot Arena itself may reward surface features (§2). **Implication.** Magpie reports LC win rates; Evol-Instruct reports an AlpacaEval score without stating whether it is length-controlled ([[evol-instruct]] Table 1). Comparisons across these papers need the same metric, and any judge result needs a benchmark-based check beside it (ch-49).

## §11 Coverage against real user prompts

**Definition.** A generated prompt set covers a real distribution when a model trained on it assigns high likelihood to held-out prompts from that distribution. WildChat measures this as the average NLL of first-turn user prompts under Llama-2 7B fine-tuned on each dataset (70% train, 30% validation) ([[wildchat]] §3, Fig. 3; [[excerpts/wildchat]]).

**Real distributions.** WildChat: 1,039,785 opt-in ChatGPT conversations, 204,736 users by IP, 68 languages, 2.54 turns (Table 1); English first-turn categories: assisting or creative writing 61.9%, analysis or decision explanation 13.6%, coding 6.7%, factual information 6.3%, math reasoning 6.1% (Table 4). [[lmsys-chat-1m]]: 1,000,000 conversations with 25 models from 210K IP addresses, April to August 2023 (Abstract, §3.1); in 20 k-means clusters of 100K English prompts, the coding clusters sum to 29.64% and the three unsafe clusters to 12.28% (derived from Fig. 3), 5% of conversations are flagged by the OpenAI moderation API (§3.3), and two clusters contain script-generated template prompts, so the authors state the figure "might not reflect the real-world topic distributions" ([[excerpts/lmsys-chat-1m]]).

**Worked example** (Fig. 3 NLL; unit as printed). Evaluated on WildChat prompts: trained on WildChat 5.18, Open Assistant 6.57, ShareGPT 7.61, Dolly 8.77, Alpaca 10.87. The Alpaca-trained model is 10.87 − 5.18 = 5.69 worse than the in-distribution model. Evaluated on Alpaca prompts: trained on Alpaca 2.24, WildChat 3.28 (1.04 worse). The asymmetry (5.69 versus 1.04) is the coverage gap. It is consistent with the WildChat prompt distribution containing the kinds of prompts found in Alpaca, while the Alpaca distribution lacks kinds of prompts found in WildChat (Interpretation). Panel B of [figures/generation-methods.html](figures/generation-methods.html) shows the full matrix and computes this gap for any pair.

**What each method measured.**
- Self-Instruct: seed overlap with test sets (App. A.1) and a novel-instruction human evaluation; no real-user comparison.
- Evol-Instruct: t-SNE of BERT embeddings with 20 k-means clusters, qualitative (App. J); WizardEval has 218 author-built real-world instructions (§4.4).
- Magpie: t-SNE against three synthetic datasets (§3.1); SFT on WildChat is a baseline, not a coverage reference.
- Persona Hub: similarity between generated items; the instruction use case uses WildChat demonstrations with inferred personas (§4.3).
- WRAP: perplexity on 21 Pile domains, which is a coverage measurement for text rather than prompts.

**Limits.** Category shares from Magpie (information seeking), WildChat, and LMSYS use different taxonomies and classifiers, so they are not a controlled comparison (Open question). WildChat reflects users of a free ChatGPT service who opted in, LMSYS reflects users of a public demo and arena, and both include toxic prompts; LMSYS also includes script-generated prompts. The logs can also serve directly as SFT data: WildLlama (Llama-2 7B on WildChat) scores MT-bench 6.35 versus 6.13 for Vicuna 7B (Table 9), and LMSYS HighQuality-7B (45K conversations with OpenAI and Anthropic responses, 33M tokens) scores 47.7 MMLU and 6.03 MT-Bench versus 49.8 and 6.17 for Vicuna-7B-v1.5 (370M tokens) ([[lmsys-chat-1m]] Table 6). Prompt-distribution coverage for human-written data is also discussed in ch-15.

## §12 Composing methods: an evidence table

| Method | What it varies | Breadth evidence (source) | Narrowing evidence (source) |
|---|---|---|---|
| Self-Instruct | new tasks from a seed pool | +33.1 SuperNI ROUGE-L on GPT-3; novel-task human eval ([[self-instruct]] Table 3, §4.4) | on LLaMA 13B all 5 benchmarks fall ([[tulu-1-how-far-can-camels-go]] Table 3) |
| Alpaca | same, stronger generator | none beyond author pairwise eval ([[alpaca]]) | TyDiQA −12.1, GSM −5.0 at 13B (Tülu 1 Table 3) |
| Evol-Instruct | difficulty and rarity of existing prompts | 9-benchmark avg 58.96 vs 54.60 ([[evol-instruct]] Table 1) | seed choice moves GSM8k 37.15 ↔ 31.46 (Table 2) |
| WizardCoder | code difficulty | 8/8 MultiPL-E languages above CodeLlama-Instruct-34B ([[code-evol-instruct]] Table 2) | non-code not reported; v1 round selection on test set |
| WizardMath | difficulty in both directions | +21.5 GSM8K, +31.1 MATH over original ([[wizardmath]] Table 6) | AQuA 20.0 vs 40.9 base at 70B ([[mammoth]] Table 3) |
| Magpie | the aligned model's user-turn prior | AE2 LC 25.08 vs 22.92 official ([[magpie]] Table 1) | GSM8K 47.92 vs 71.72 (Table 3) |
| Persona Hub | persona context | MATH 64.9% with no MATH data ([[persona-hub]] Table 2) | persona-diverse set lower OOD than seed-diverse ([[prismatic-synthesis]] Table 2) |
| WRAP | style of real documents | 8 general-task avg 52.3 vs 50.3 at half the real tokens ([[rephrasing-the-web]] Table 1) | synthetic-only Pile perplexity worse (Fig. 4); no new knowledge (§5.2) |
| Humpback | instructions for real text | 65B AlpacaEval 83.71% ([[humpback]] Table 3) | self-curation recall 0.09 at M0 (Table 9) |

No source in this chapter trains on a controlled composition of several methods and measures breadth, so the table does not rank methods (Open question). The methods vary different parts of the data: which tasks exist (seeds, real prompts), how hard they are (evolution), whose phrasing is used (persona, rephrasing), and whose responses are imitated (the response generator). A composition covers more of these parts only when each component's contribution is checked with the measurements in the Generalization lens. The lab in ch-29 builds and filters one such set.

## Negative samples and negative feedback

**Four meanings** (course standard): (1) negative marginal value, a sample that hurts when used as a positive target; (2) negative as content, a failure placed in the input or in a corrected target; (3) negative as conditioning, a failure trained under a control token; (4) negative as gradient, an explicit decrease of the sample's likelihood. Only (4) removes probability mass from the sample.

**1. Where negatives come from.** Heuristic filters (Self-Instruct ROUGE-L, keywords, format; Alpaca length and blacklist), LLM judgments (Evol-Instruct no-information-gain check by ChatGPT), learned scores (Magpie reward and reward difference; Humpback self-ratings), and reward-ranked samples (Magpie-DPO rejected responses). Label error: Humpback's M0 curator had precision 0.44 and recall 0.09 (App. B, Table 9), so 91% of the good pairs in the dev set were discarded (false negatives). Self-Instruct's filters let through data in which 46% of audited examples had an invalid field (Table 2), which are false positives of the filter.

**2. What practice does.** Discard (meaning 1) in every generation pipeline above; no count of discarded samples is reported for Self-Instruct, Alpaca, or Evol-Instruct. Gradient (meaning 4) appears in Magpie's DPO extension: for each selected instruction, k = 5 responses at temperature 0.8 are scored by ArmoRM-Llama3-8B-v0.1, and the lowest-reward response becomes the rejected sample (§2.2, §4.1). Meanings 2 and 3 do not appear in these sources; see ch-31a.

**3. Mechanism for the gradient case.** For a softmax over next tokens with logits z, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, where y is the rejected token and p_j the probability of token j. A descent step of size η on `log p_y` changes z_y by `−η(1 − p_y)` and each other z_j by `+η p_j`. **Worked example.** p = (0.60, 0.30, 0.10) for tokens (a, b, y), η = 1: logit changes (+0.60, +0.30, −0.90), new probabilities (0.710, 0.263, 0.026). Token b, which was not penalized, loses 0.037, and the most likely token a gains 0.110 (computed). Magpie's rejected responses are sampled from the generating aligned model, not from the Llama-3-8B SFT model being trained (§4.1), so their tokens can have low probability under the policy; by the gradient above, pushing such tokens down moves mass mainly toward the policy's most likely alternatives (Interpretation; Magpie does not measure policy log-probabilities of rejected responses). The DPO derivation is in ch-39 and the displacement analysis in ch-43a.

**4. Evidence.** Benefit: Llama-3-8B with Magpie-Pro SFT then 100K Magpie-Pro-DPO pairs reaches AE2 LC 50.10 and Arena-Hard 35.7, versus 25.08 and 18.9 after SFT (Table 1). The gain is not split between the chosen and rejected terms, and DPO β is not reported. Discard: uncurated Humpback data did not improve win rate with size (§3.3, Fig. 2), and Magpie-Air filtering (3M → 300K) did not raise LC (22.96 vs 22.66, Table 12). **Size of effect.** No source here measures the share of improvement due to negatives.

**5. Controls.** Measure filter precision and recall on a labeled dev set before trusting a curator (Humpback Table 9). Draw rejected responses from the policy being trained when possible, or keep a positive NLL term. Keep the "sorry" rule conditional on short length, as Evol-Instruct does (fewer than 80 words), so that full answers containing an apology are not discarded.

**6. Diagnostics.** Report discard counts per filter; log chosen and rejected log-probabilities separately during DPO; track refusal rate and response length before and after filtering, since length filters and refusal filters change both.

**7. Effect on generality.** Filters that discard refusals (Evol-Instruct's "sorry" rule) or keep the longest responses (Magpie's output-length filter) change how often refusals appear as SFT targets; no source here measures the effect on refusal behavior. The Magpie SFT model has jailbreak RtA 0.80 versus 0.93 for Llama-3-8B-Instruct (Table 15), and the paper does not attribute this to a specific filter (Open question). Effects of the DPO rejected term on calibration and pass@k are not reported.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT3-SELF-INST (GPT-3 davinci) | 175B | SFT | data | 52,445 instructions; 82,439 instances | arXiv:2212.10560v2 Table 1 | verified 2026-09-15 | §4.5 Fig. 7: human-eval gains plateau after 16K instructions |
| GPT3-SELF-INST | 175B | SFT | epochs; prompt loss weight; other settings | 2; 0; OpenAI API defaults (undisclosed) | v2 App. A.3 | verified 2026-09-15 | "we find this works better in our case" (loss weight, no numbers); 2 epochs "to avoid overfitting the training tasks" |
| Self-Instruct data generation | 175B generator | SFT (data gen) | in-context examples; ROUGE-L rule | 8 (6 human, 2 generated); add if < 0.7 to all pool instructions | v2 §2.2 | verified 2026-09-15 | no ablation reported |
| Self-Instruct data generation | 175B generator | SFT (data gen) | instruction decoding; instance decoding | temp 0.7, top_p 0.5, presence penalty 2, max 1024; temp 0, presence penalty 1.5, max 300 | v2 App. A.2 Table 4 | verified 2026-09-15 | no ablation reported |
| Alpaca 7B (LLaMA 7B) | 7B | distill-SFT | examples; epochs; LR; batch; max length; weight decay | 52K; 3; 2e-5; 128; 512; 0 | README @761dc5b "Fine-tuning" | verified 2026-09-14 | no ablation reported |
| Alpaca data generation | — | distill-SFT | teacher; seeds; instructions per request; code defaults | text-davinci-003; 175; 20; temperature 1.0, top_p 1.0, ROUGE-L F reject > 0.7 | blog; README; generate_instruction.py L117–118, L196 | verified 2026-09-14 (code defaults; run arguments not reported) | no ablation reported |
| WizardLM-13b (LLaMA 13B) | 13B | SFT (data gen) | evolver; seeds; rounds; total | gpt-3.5-turbo; Alpaca 52k; 4; 250k | arXiv:2304.12244v3 §4.2 | verified 2026-09-14 | Table 2: Llama-2-70B-Chat evolver avg 56.27 vs 58.96 |
| WizardLM-13b | 13B | SFT | examples; optimizer; LR; batch; epochs; max tokens | 70k sampled from 250k; Adam; 2e-5; 4 per GPU on 8 V100; 3; 2048 | v3 §4.2 | verified 2026-09-14 | Table 2: 250k variant 60.30; App. L Table 4: 2.5/2.75/3 epochs 57.92/58.24/58.96 |
| WizardCoder 15B (StarCoder 15B) | 15B | SFT | data; evolver | about 78k (seed + 3 rounds); gpt3.5-turbo | arXiv:2306.08568v2 §4.2; v1 §4.5 | verified 2026-09-14 | Fig. 3: pass@1 peaks after 3 rounds |
| WizardCoder 15B | 15B | SFT | batch (unit not stated); seq length; steps; warmup; peak LR; schedule | 512; 2048; 200; 30 steps; 2e-5; cosine, fp16 | v2 §4.2 | verified 2026-09-14 | no ablation reported |
| WizardMath-Mistral-7B | 7B | SFT | data; evolver; rounds | 418k (448k unique, 30k removed for contamination); GPT-4, 5 rounds (2 down, 3 up), 6 evolutions per round, temperature 0.7; answers by GPT-4-0613 | arXiv:2308.09583v3 §3.1, §4.1 | verified 2026-09-15 | Table 6: all rounds 81.2 / 46.2 vs original 59.7 / 15.1 |
| WizardMath (Llama 2 7B/13B; 70B; Mistral-7B) | 7B–70B | SFT | epochs; LR; batch; seq length | 3; 2e-5 / 1e-5 / 5e-6; 512; 2048 | v3 §4.1 | verified 2026-09-15 | no ablation reported |
| Llama-3-8B + Magpie-Pro-300K-Filtered | 8B | SFT | examples; LR; schedule; warmup; epochs | 300K; 2e-5; cosine; 100 steps; 2 | arXiv:2406.08464v2 §4.1, App. E.2 Table 8 | verified 2026-09-14 | Table 12: 300K-Filtered 25.08 vs 1M-Raw 24.16 LC (data only) |
| same | 8B | SFT | effective batch; optimizer; max length | 32 (4 × 1 × 8); AdamW β (0.9, 0.999), ε 1e-8; 8192 | App. E.2 Table 8; §4.1 | verified 2026-09-14 | no ablation reported |
| Magpie-Air generation (Llama-3-8B-Instruct) | 8B generator | SFT (data gen) | instruction decoding; response decoding | temperature {1.0, 1.1, 1.2} × top-p {1.00, 0.995, 0.990} at 300K each + 1.25 × same at 100K each; greedy | App. E.1 Table 7 | verified 2026-09-14 | App. D.3 Fig. 11: higher values raise difficulty and diversity |
| Llama-3-8B + Magpie-Pro-DPO | 8B | preference | pairs; sampling; LR; epochs; batch; β | 100K; k = 5, T = 0.8, ArmoRM max/min; 5e-7 cosine, 10% warmup; 1; 128; β not reported | §4.1; App. E.2 Table 9 | verified 2026-09-14 (β not reported) | Table 1: LC 50.10 vs SFT 25.08 |
| Qwen2-7B on Persona Hub math | 7B | distill-SFT | examples; problem generator; solution generator; LR, epochs, batch | 1.07M; GPT-4 zero-shot persona prompt; gpt-4o; not reported | arXiv:2406.20094v3 §4.1.2 | verified 2026-09-14 / not reported (hyperparameters) | Fig. 9: accuracy vs instance count (no numbers in text) |
| WRAP 1.3B (C4 + Q/A rephrases) | 1.3B | pretrain | steps × batch; seq length; peak/min LR; schedule | 300k × 1M tokens (300B tokens, derived); 1024; 2e-4 / 1e-5; cosine, 1% warmup | ACL 2024 §3.2; Fig. 2 caption | verified 2026-09-15 | no ablation reported |
| WRAP 1.3B | 1.3B | pretrain | optimizer; weight decay; clip; real:synthetic; chunk | Adam (0.9, 0.999); 0.01; 1.0; 1:1; about 300 tokens | §3.1–§3.2 | verified 2026-09-15 | Table 2: Med+C4 49.3 vs Med only 47.9; chunk size: "empirical observation", no numbers |
| WRAP rephrase generation | 7B rephraser | pretrain (data gen) | model; throughput; compute for 85B tokens | Mistral-7B-Instruct; 3M tokens/hour per A100; about 25K GPU hours | §3.1, §9.1 | verified 2026-09-15 | RQ1 Fig. 3: Qwen-1.8B comparable, 3× throughput |
| Humpback 65B (LLaMA 65B) | 65B | SFT | data; curation; LR; batch | 3,200 seed + about 42k curated (45k total); k = 5 (A5 defined as score ≥ 4.5 in §3.3), 2 iterations; 1e-5 → 9e-6 linear ("most models"); 32 | arXiv:2308.06259v3 §2.3, §3.1, Table 3 | verified 2026-09-14 | Fig. 2: A5 and A4 vs no curation (7B) |
| WildLlama (Llama-2 7B) | 7B | SFT | data; batch; LR; seq length; epochs | WildChat to 2023-07-16; 128 conversations; 2e-5; 2048; 3 | arXiv:2405.01470v1 §5 | verified 2026-09-15 | no ablation reported (Vicuna settings reused) |
| Imitation model (LLaMA 7B/13B, ShareGPT-Mix) | 7B, 13B | distill-SFT | block; epochs; optimizer; LR; warmup; batch | 2048 tokens; 1; AdamW with gradients rescaled by weight magnitude; 2e-3; 1000 steps; 32 | arXiv:2305.15717v1 §4.1 | verified 2026-09-15 | Fig. 4 top: more imitation data does not raise MMLU/NQ/HumanEval |

Rows marked "verified 2026-09-15" were read in the primary PDFs on that date because the library cards for [[self-instruct]], [[wizardmath]], [[rephrasing-the-web]], and [[wildchat]] have no Verification section and no cards exist yet for [[false-promise-imitating-proprietary-llms]], [[length-controlled-alpacaeval]], and [[lmsys-chat-1m]]; the chapter excerpts hold the checked extracts.

**Starting point for a small general-purpose run.** For extraction-based instruction SFT of an 8B base model, the verified reference is the Magpie run: 300K filtered Magpie-Pro conversations generated by Llama-3-70B-Instruct, 2 epochs, peak LR 2e-5 with cosine decay and 100 warmup steps, AdamW (0.9, 0.999), effective batch 32 sequences, maximum length 8192, on 4 devices. Only the data quantity and filtering were ablated (Table 12); the optimization settings were not. The same run trails the official instruct model on GSM8K (47.92 vs 71.72) and on jailbreak refusal (0.80 vs 0.93), so it needs the reasoning data and safety checks listed in the Generalization lens.

## Generalization lens

**(a) What increases breadth.**
- Prompts drawn from real users: a WildChat-trained model is 1.04 NLL above the best model on Alpaca prompts, while an Alpaca-trained model is 5.69 above the best on WildChat prompts ([[wildchat]] Fig. 3).
- Seed diversity over surface diversity: 10k seeds with 0.1k personas scored 54.77 out-of-distribution accuracy versus 38.15 for 0.1k seeds with 10k personas at 10k samples ([[prismatic-synthesis]] Table 2).
- Merging evolved rounds with the seed data: single-round code data at matched size scored 51.2–56.1 versus 45.7 for the seed ([[code-evol-instruct]] Table 5); both difficulty directions in math (+21.5 GSM8K, +31.1 MATH; [[wizardmath]] Table 6).
- Keeping real text beside rephrases: Med+C4 49.3 versus Med-only 47.9 on general tasks ([[rephrasing-the-web]] Table 2).
- A stronger base model: larger bases raised imitation-model accuracy where more imitation data did not ([[false-promise-imitating-proprietary-llms]] Fig. 4 bottom); on the same Tülu mix, LLaMA-2 7B averaged 45.7 versus 38.3 for LLaMA 7B ([[tulu-1-how-far-can-camels-go]] Table 4).
- Adding targeted reasoning data to extracted chat data: GSM8K 47.92 → 63.08 ([[magpie]] Table 3).

**(b) What causes narrowing or forgetting.**
- Broad imitation data: NQ 17 → 10 (7B) and 20 → 15 (13B) ([[false-promise-imitating-proprietary-llms]] Table 1).
- Base-GPT-3 Self-Instruct data on LLaMA 13B: MMLU −11.9, Codex-Eval −16.1 ([[tulu-1-how-far-can-camels-go]] Table 3).
- Single-turn teacher-generated data: Alpaca and GPT4-Alpaca lowered TyDiQA to 31.1 and 23.5 from 43.2; the authors attribute such losses to little multilingual data in those sets (Tülu 1 §5.1, Table 3; Interpretation).
- Domain evolution: WizardMath-70B AQuA 20.0 and SAT-Math 13.2 versus 40.9 and 51.3 for Llama-2-70B ([[mammoth]] Tables 3–4).
- The generator's prompt prior: Magpie SFT trails on GSM8K and on jailbreak refusal ([[magpie]] Tables 3, 15).
- Seed composition: ShareGPT seed lowered GSM8k 37.15 → 31.46 ([[evol-instruct]] Table 2).
- Synthetic-only pretraining text: higher Pile perplexity on many sub-domains ([[rephrasing-the-web]] Fig. 4).

**(c) How to measure it for this stage.**
- A held-out novel-task human evaluation written without reference to the seeds, with a seed-to-test overlap check ([[self-instruct]] §4.4, App. A.1).
- Length-controlled judge scores beside capability benchmarks; report response length ([[length-controlled-alpacaeval]] §4; [[tulu-1-how-far-can-camels-go]] §5.4).
- A before/after suite covering knowledge, reasoning, multilingual, coding, and safety relative to the base model ([[tulu-1-how-far-can-camels-go]] Table 3).
- Coverage NLL on held-out WildChat or LMSYS-Chat-1M prompts, with per-category shares and removal of template clusters ([[wildchat]] Fig. 3; [[lmsys-chat-1m]] Fig. 3).
- Per-task results, not only averages (WRAP worked example in §7).
- Round or checkpoint selection on a dev set separate from reported benchmarks, and decontamination of generated prompts ([[code-evol-instruct]] §3.2, App. C; [[wizardmath]] §4.1).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Using input-first generation for classification tasks | one label dominates generated instances | label histogram per classification task; switch to output-first ([[self-instruct]] §2.2) |
| Treating a ROUGE-L filter as a diversity guarantee | accepted prompts ask for the same skill in new words | cluster by embedding or by task category after filtering; compare category shares with WildChat |
| Reporting raw AlpacaEval gains after selecting long responses | win rate rises with mean response length | report LC win rate and length; rerun with a concise-answer instruction ([[length-controlled-alpacaeval]] §4.1) |
| Reading judge wins as capability gains | benchmark accuracy flat or lower while judge scores rise | before/after MMLU, NQ, HumanEval, TyDiQA against the base ([[false-promise-imitating-proprietary-llms]] Fig. 4) |
| Choosing evolution rounds on a test benchmark | best round differs on a separate dev set | hold out a dev set such as MBPP-400 ([[code-evol-instruct]] §5) |
| Assuming extraction data covers real users | high NLL on real prompts; few math, coding, or unsafe prompts | coverage NLL on held-out WildChat prompts; category shares ([[wildchat]] Fig. 3, Table 4) |
| Measuring diversity only with embedding similarity | high embedding diversity, low out-of-distribution accuracy | compare with a gradient-based measure or held-out accuracy ([[prismatic-synthesis]] Table 2) |
| Crediting the prompt method for a response-model effect | gain disappears with a different response generator | swap only the response generator ([[magpie]] Table 14) |
| Dropping real text when rephrasing | QA scores rise while perplexity on code, forum, or book domains worsens | per-domain Pile perplexity with and without real data ([[rephrasing-the-web]] Fig. 4) |
| Trusting a self-curator without measuring it | small curated set; good pairs discarded | precision and recall on a labeled dev set ([[humpback]] Table 9) |
| Averaging over tasks that match the rephrase style | one or two tasks produce most of the average gain | per-task differences and sign counts (§7 worked example) |

## Check your understanding

1. Self-Instruct generates the label first for classification tasks. Explain what goes wrong with input-first generation for a grammar-error detection task and why generating labels first fixes it.
2. The ROUGE-L filter accepted "explain why the sea is salty" next to "write a poem about the sea". Explain why a set that passes this filter can still cover few skills, and which measurement in this chapter would detect it.
3. Imitation models were rated as good as ChatGPT in about 70% of cases while their NQ accuracy fell. Using §9 and §10, explain which properties of the responses a crowdworker or GPT-4 judge rewards and why these transfer through SFT when knowledge does not.
4. In the WildChat coverage matrix, the Alpaca-trained model is 5.69 NLL worse on WildChat prompts, while the WildChat-trained model is only 1.04 worse on Alpaca prompts. Explain the asymmetry in terms of the two prompt distributions.
5. Magpie SFT data beats the official instruct model on AlpacaEval 2 LC but trails it by 23.8 points on GSM8K. Explain how the pre-query extraction mechanism could produce both results, and what data change the authors tested.
6. In WizardMath's Table 6 the combined gain of downward and upward rounds (+21.5 GSM8K) is less than the sum of the separate gains (+33.7). What does this say about the information the two directions add, and why does it still matter to include both?
7. WRAP's Q/A-style model improves the 8-task average by 2.0 points, and two tasks account for 1.9 of it. Why does this matter for a claim of general improvement, and what additional evaluation would separate a style match from a capability gain?
8. A persona-conditioned dataset has lower pairwise embedding similarity than a seed-conditioned dataset but transfers worse. Explain which kind of variation each dataset contains.

## Connections

- Previous: ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix. This chapter covers the generate stage and the filters attached to it.
- Next: ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage (response-side imitation of stronger models).
- ch-09 — Pretraining Data Composition and Capability Coverage (WRAP as pretraining data, §7).
- ch-15 — Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage (real prompt distributions, §11).
- ch-21 — Taxonomy-Driven and Textbook-Style Synthesis (top-down prompt design and textbook-style text).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection (G-Vendi and selection after generation, §6).
- ch-25 — Multi-Turn Conversation Synthesis (WildChat and LMSYS-Chat-1M as multi-turn references).
- ch-29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification.
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks (imitation and format versus capability, §9).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-39 — Offline Preference Optimization: DPO and Its Variants; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-44 — Process Supervision and Verifiable Rewards (WizardMath's reward models, §4.3).
- ch-47 — Evaluation Harness and Suite Design for General Capability; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting (§10).

## Sources

- [[self-instruct]] — pipeline, output-first branch, filters, data statistics, audit, SuperNI and novel-task results, size plateau, costs (read in the primary PDF; checked extract in [[excerpts/self-instruct]]).
- [[alpaca]] — pipeline changes, released filters, SFT settings, evaluation scope, cost.
- [[evol-instruct]] — in-depth and in-breadth prompts, elimination rules, scale, seed and evolver ablations, difficulty per round.
- [[code-evol-instruct]] — WizardCoder heuristics, round merging and sizes, Table 5 complexity-versus-quantity, stop rule, decontamination, MultiPL-E.
- [[wizardcoder]] — model-results card for the same paper; its operator list is not verified and is not used here.
- [[wizardmath]] — downward and upward evolution, data counts, Table 6, reward-model ablation (primary PDF; extract in [[excerpts/wizardmath]]).
- [[mammoth]] — out-of-domain regression of WizardMath-70B.
- [[magpie]] — template, decoding, filters, SFT/DPO results, reasoning gap, filtering and response-generator ablations, safety labels, costs.
- [[persona-hub]] — persona construction and deduplication, math result, similarity study, stated limits.
- [[prismatic-synthesis]] — seed-diverse versus persona-diverse sets and embedding versus gradient diversity.
- [[rephrasing-the-web]] — WRAP styles, mixing, training settings, Table 1–2, version conflict, cost (primary; extract in [[excerpts/rephrasing-the-web]]).
- [[humpback]] — backtranslation, self-curation, curation precision/recall.
- [[tulu-1-how-far-can-camels-go]] — per-dataset regressions at 13B and the correlation of AlpacaEval win rate with unique response tokens.
- [[false-promise-imitating-proprietary-llms]] — imitation data setup, NQ and style tables, base-size effect (extract).
- [[length-controlled-alpacaeval]] — LC regression, gameability, Arena correlation, truncation attack (extract).
- [[wildchat]] — dataset statistics, prompt categories, coverage NLL matrix, WildLlama (extract).
- [[lmsys-chat-1m]] — dataset statistics, topic clusters and caveats, unsafe share, SFT subsets (extract).
