<!-- chapter: ch-30b
     track: sft
     kind: content
     title: Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares
     deps: [ch-30a, ch-18]
     sources: [[sft-data-composition-dmt]], [[tulu-1-how-far-can-camels-go]], [[tulu-3]], [[tulu-3-sft-mix]], [[llama-3]], [[llama-3-recipe]], [[long-context-llama3]], [[glm-4-5]], [[qwen-3]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[agenttuning]], [[agent-flan]], [[agent-data-protocol]], [[limi]], [[llama-nemotron]], [[prolong]], [[prolong-recipe]], [[qdit-data-diversity]], [[deita]], [[instag]], [[mix-data-or-merge-models]], [[am-reasoning-answers-for-non-reasoning]], [[nemotron-ultra]]
     figures: figures/mixture-share-calculator.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares

> **Core insight.** No single SFT dataset, and no single mixture ratio, is best for every skill. In controlled LLaMA 7B-33B experiments, mixing math, code, and chat data raises each skill when each source is small (with LLaMA-7B code as the authors' stated outlier) and lowers each skill when each source is used in full ([[sft-data-composition-dmt]]), and removing one subset from the Tülu 3 8B mixture changes scores on skills that the subset does not target ([[tulu-3]]). The reports that print agentic and long-context shares use η = 0.2 as the agent sampling ratio (AgentTuning) and 0.11% of examples for long-context data (Llama 3.1), and the long-context evidence conflicts: Llama 3 reports that short-only SFT regressed long-context ability, while in ProLong every tested synthetic long SFT share (1%, 3%, 10%, and 50% of tokens) lowered its long-context average.
>
> **Guideline.** When a specialist skill has a large dataset (110,142 GSM8K RFT examples in DMT), train it before the general stage and replay a fraction of it (1/256 in DMT) in the final stage, because at LLaMA-7B the DMT schedule with k = 1/256 scored 41.92 GSM8K and 6.08 MT-Bench against 32.60 and 6.02 without replay. When a single-stage mixture is edited, re-evaluate every skill and at least one unseen evaluation per skill, because removing the Persona data from Tülu 3 8B lowered development IFEval by 19.2 points while IFEval-OOD rose by 0.4. When agent trajectories are added, keep general instruction data in the mix and select the agent share on held-out agent tasks, because agent-only training of AgentLM-7B scored 0.09 on held-out agent tasks against 0.67 for η = 0.2. When a mixture share is reported or copied, state its unit, because 0.11% of Llama 3.1 SFT examples is 4.95% of its tokens.

## Why this chapter matters for a general-purpose model

Supervised fine-tuning (SFT) trains a pre-trained or mid-trained model with cross-entropy on target responses. A general-purpose model needs one SFT stage that covers chat, instruction following, math, code, tool use, multilingual text, safety, long inputs, and sometimes two response modes. The SFT mixture is the set of datasets and the rule that decides how often each one is sampled.

Three problems follow, and each is measurable.

1. **Interference.** Adding data for one skill can lower the score of another skill, measured as the difference from a run that trains the other skill without the addition.
2. **Units.** A share quoted as "% of examples" can differ by a factor of 45 from the same share in tokens (this chapter's §1), so a share copied from one report into a recipe that counts a different unit changes the mixture.
3. **Narrowing.** A mixture tuned on development benchmarks can raise those benchmarks without raising unseen evaluations of the same skill (this chapter's §3).

In the pipeline pre-training → mid-training → SFT → preference optimization → RL → evaluation, the SFT mixture sets the capability profile that later stages start from. Later stages can still change that profile: Llama 3 reports that short-context direct preference optimization (DPO, ch-39) did not undo long-context SFT and suspects the reason is that its DPO recipe has fewer optimizer steps than SFT (this chapter's §6), while Qwen3 reports thinking-mode declines after mode-fusion SFT and general RL (this chapter's §4). ch-30 covers per-example design choices (masking, packing, templates, epochs). ch-30a covers how to measure forgetting. This chapter covers what goes into the mixture and in what proportion. ch-30c covers the alternative of training separate models and merging weights.

## §1 Units of a mixture: example share, token share, and loss weight

**Definition.** A mixture share is the fraction of the training signal assigned to one domain d. It can be counted in examples, in all tokens, or in loss tokens (the response tokens that receive cross-entropy after masking).

**Problem.** Reports print shares in different units, and the unit that sets each domain's gradient weight depends on how the loss is averaged. A domain's token share exceeds its example share by the ratio of its average example length to the mixture's average length, which is 45 for Llama 3.1 long-context data.

**Mechanism.**
1. Each dataset d has n_d raw examples and a sampling multiplier w_d, so m_d = n_d · w_d examples are drawn per epoch.
2. Each example of d has L_d tokens on average and ℓ_d loss tokens on average.
3. With a per-sequence mean loss, each example's loss is averaged over its own tokens before examples are averaged, so every example has equal weight.
4. With a token mean over the batch, or a token sum, every loss token has equal weight, so long-response domains receive more weight.

**Formula.**

```
example share_d = m_d / Σ_j m_j
token share_d   = m_d · L_d / Σ_j m_j · L_j
loss share_d    = m_d · ℓ_d / Σ_j m_j · ℓ_j
```

m_d is the number of sampled examples of domain d; L_d is the average number of tokens per example of d; ℓ_d is the average number of loss tokens per example of d; the sums run over all domains j in the mixture.

**Worked example 1 (Llama 3.1, [[llama-3]] Table 7).** The table prints "% of examples", average tokens per example, and average tokens in the final response.

| Domain | % of examples | Avg tokens | Avg final-response tokens |
|---|---|---|---|
| General English | 52.66 | 974.0 | 317.1 |
| Code | 14.89 | 753.3 | 374.5 |
| Multilingual | 3.01 | 520.5 | 289.7 |
| Exam-like | 8.14 | 297.8 | 173.4 |
| Reasoning and tools | 21.19 | 661.6 | 301.9 |
| Long context | 0.11 | 38,135.6 | 740.5 |

- Long-context tokens per average example: 0.0011 × 38,135.6 = 41.95.
- All six rows: 0.5266 × 974.0 + 0.1489 × 753.3 + 0.0301 × 520.5 + 0.0814 × 297.8 + 0.2119 × 661.6 + 41.95 = 847.1 tokens (the printed total is 846.1 because the percentages are rounded).
- Long-context token share: 41.95 / 847.1 = 4.95% (derived). This is 45 times the example share.
- Long-context final-response share: 0.0011 × 740.5 = 0.81 of 310.4 tokens = 0.26% (derived).

Which of the three numbers sets the gradient weight depends on the loss mask and aggregation, which Table 7 does not state.

**Worked example 2 (DeepSeek-R1 Dev3 SFT, [[deepseek-r1-recipe]], arXiv:2501.12948v2 Table 5).** General has 177,812 of 804,745 samples, which is 22.10% of samples. Its tokens are 177,812 × 1,419.8 = 252.5M of 804,745 × 5,355.3 = 4,309.7M, which is 5.86% of tokens (derived). Math is 49.12% of samples and 55.90% of tokens.

**Evidence that aggregation matters.** ProLong changed its SFT loss to "the average over all valid training tokens" because per-sequence averaging "skews the optimization and also our control over the domain proportions" ([[prolong]], App. A.3). Llama-Nemotron reports that its models "require higher learning rates to effectively learn from long reasoning traces, especially due to sequence-length-dependent token loss averaging" ([[llama-nemotron]], §4.1). Neither of these two reports prints an ablation of the aggregation choice. Tülu 3 switched its SFT to a sum loss, which "effectively weights all tokens equally", after tracing a gap between training setups to loss averaging, and its Fig. 5 compares mean and sum loss across learning rates for Llama 3.0 on the Tülu 2 mixture, with sum loss at 5e-6 best ([[tulu-3]] §4.3.2; figure only, no table). ch-30 treats sum versus mean aggregation in detail.

[Mixture share calculator](figures/mixture-share-calculator.html): enter counts, multipliers, and average sizes (presets for Llama 3.1 Table 7, DeepSeek-R1 Table 5, and the ADP agentic mixture) to see example, token, and loss-token shares and which one sets the loss weight under each aggregation rule.

**Conditions and limits.** Table 7 gives averages, not distributions; a domain whose examples vary widely in length has a token share that the average describes but a per-batch share that varies from batch to batch. Rounded percentages limit precision to about two significant digits for small shares.

**Implication.** A mixture share copied from a report is reproducible only with its unit, its loss mask, and its aggregation rule.

## §2 Interference and transfer as functions of data amount, ratio, and order

**Definition.** Negative transfer (interference) from domain B to skill A is a lower A score after training on A + B than after training on A alone with the same A data. Positive transfer is a higher A score.

**Problem.** A generalist mixture combines many domains, and a score change on skill A can come from the amount of B, the ratio of B to A, or the order in which A and B are trained.

**First-order explanatory model (Interpretation; none of the cited sources measures gradients).** Let g_A and g_B be the loss gradients of skills A and B. One step on B with learning rate η changes the loss on A by approximately

```
ΔL_A ≈ −η · (g_A · g_B)
```

where g_A · g_B is the inner product of the two gradient vectors. A negative inner product raises L_A (interference); a positive one lowers it (transfer). Example: g_A = (1, 2), g_B = (−2, 0.5), η = 0.1. Then g_A · g_B = −2 + 1 = −1 and ΔL_A ≈ +0.1. With g_B = (1, 0), g_A · g_B = 1 and ΔL_A ≈ −0.1. This model describes one step; it does not predict how effects accumulate over epochs.

**Evidence: DMT ([[sft-data-composition-dmt]], LLaMA 7B/13B/33B; GSM8K RFT 110,142, Code Alpaca 20,022, ShareGPT 86,060 examples; 3 epochs, LR 2e-5).** GSM8K RFT is 7.5K GSM8K training questions with multiple reasoning paths collected by rejection sampling (App. A). DMT (dual-stage mixed fine-tuning) is the paper's proposed schedule, described in item 4.

1. **Amount (§3.2, Fig. 2).** GSM8K rises with data amount at all three sizes; HumanEval follows an irregular scaling curve at 7B and 13B and an approximately log-linear one at 33B. MT-Bench "emerges with only around 1k data samples" and then rises slowly.
2. **Mixed versus single source (§3.3, Fig. 3).** Mixing the three sources at the same fraction raises each score over the single-source run at 1/256 and lowers it at the full amount; the authors list LLaMA-7B code at 1/256 as an outlier. For 7B the crossover lies between 1/64 and 1/16. The authors state that the low-resource gain for math and general ability grows with model size (Result, single study).
3. **Ratio (Fig. 4).** With one source fixed and the other scaled, the math score changes little with the ratio. HumanEval fluctuates, which the authors attribute to code data inside ShareGPT (Interpretation). Removing code and math examples from ShareGPT (86K → 63K), identified with InsTag (an LLM-based tagger that assigns open-set intent tags to instructions, [[instag]]), reduces the high-resource conflict "to some extent" (§4.2, Fig. 6).
4. **Order (Table 1).** Sequential training (code → math → general) forgets the earlier skills. DMT trains code and math first, then trains on ShareGPT plus a fraction k of the code and math data.

**Worked example (Table 1, GSM8K / HumanEval / MT-Bench).**

| LLaMA-7B run | GSM8K | HumanEval | MT-Bench |
|---|---|---|---|
| Single source for the column's skill (math only, code only, general only) | 49.10 | 18.40 | 5.88 |
| Multi-task learning (one mixed stage) | 47.53 | 14.63 | 5.76 |
| Sequential training | 31.39 | 15.85 | 5.72 |
| Mixed sequential training (code+math, then general) | 32.60 | 15.24 | 6.02 |
| DMT, k = 1/256 | 41.92 | 17.68 | 6.08 |

- Multi-task minus single source: −1.57 GSM8K, −3.77 HumanEval, −0.12 MT-Bench. At 33B the same differences are −1.22, −7.92, and −0.56.
- Sequential minus math only: 31.39 − 49.10 = −17.71 GSM8K. This is forgetting from later stages.
- DMT minus mixed sequential: +9.32 GSM8K, +2.44 HumanEval, +0.06 MT-Bench. At k = 1/256 the second stage adds 430 GSM8K RFT and 78 Code Alpaca examples (Table 2) to 86,060 ShareGPT examples, which is 508 / 86,568 = 0.59% of stage-2 examples (derived).
- DMT still trails the math-only run by 7.18 GSM8K points at 7B; at 33B the gap is 1.55 (57.91 vs 56.36).
- Raising k from 1/256 toward 1/4 trades general score against specialist score, and k from 1/4 to 1 lowers the general score (§4.3, Fig. 5).

**Conditions and limits.** LLaMA-1 models, one benchmark per ability, and MT-Bench scored by GPT-4. Training seeds and repeated training runs are not reported. The paper prints two batch sizes (16 in §3.1, 128 in App. C). The paper averages three MT-Bench runs because the scores "will fluctuate" (App. C) but does not report the fluctuation, so the −0.12 MT-Bench difference cannot be compared with run-to-run variation.

**Implication.** Mixture effects depend on amount, not only on ratio, so a ratio chosen at one data size is not evidence for a different data size. Training order creates forgetting that a replay fraction of 1/256 reduced in DMT; ch-30a covers replay as a forgetting control.

## §3 No single dataset is best: Tülu 1 comparisons and Tülu 3 removal ablations

**Evidence 1: Tülu 1 ([[tulu-1-how-far-can-camels-go]], LLaMA 13B, 12 datasets, one recipe, Table 3).**
- The Human+GPT mixture has the best average (45.2) but is best on only 2 of 6 capability evaluations.
- ShareGPT has the highest AlpacaEval win rate (70.5%) and lowers TyDiQA F1 from 43.2 (base) to 30.5.
- 6 of 12 datasets lower GSM and 8 of 12 lower TyDiQA below the base model.
- CoT data raises GSM from 14.5 to 40.0; SuperNI lowers BBH from 39.3 to 4.5.
- The authors attribute GSM and TyDiQA losses to the absence of CoT and multilingual data in those datasets (Interpretation, §5.1).

**Evidence 2: Tülu 3 8B SFT removal ablations ([[tulu-3]], arXiv:2411.15124v5 Table 10).** The final mixture has 939,344 prompts (Table 7). Each row removes one subset and retrains.

| Run | Avg | BBH | DROP | GSM | MATH | IFEval | AE 2 | TQA | Safety |
|---|---|---|---|---|---|---|---|---|---|
| Tülu 3 8B SFT | 60.1 | 67.9 | 61.3 | 76.2 | 31.5 | 72.8 | 12.4 | 46.8 | 93.1 |
| without WildChat | 58.9 | 65.6 | 59.3 | 75.8 | 31.8 | 70.1 | 7.5 | 45.2 | 95.2 |
| without Safety | 58.0 | 68.3 | 59.4 | 76.9 | 32.6 | 71.0 | 12.4 | 45.5 | 74.7 |
| without Persona | 58.6 | 68.3 | 62.2 | 76.8 | 30.1 | 53.6 | 13.5 | 48.9 | 93.9 |
| without Math | 58.2 | 68.9 | 60.9 | 64.1 | 23.5 | 70.6 | 12.0 | 47.1 | 93.5 |

- Removing WildChat (real user conversations) changes 7 of 12 benchmarks by at least 1 point, including AlpacaEval 2 (−4.9) and BBH (−2.3), and raises Safety by 2.1.
- Removing the Safety subsets lowers Safety by 18.4; the authors describe safety as "generally orthogonal" to other skills, yet HumanEval (−1.7), DROP (−1.9), and IFEval (−1.8) also fall.
- Removing Persona data (synthetic math, code, and instruction-following prompts) lowers IFEval by 19.2 and raises TruthfulQA by 2.1.
- Removing Math data lowers GSM8K by 12.1 and MATH by 8.0 and raises BBH by 1.0.

**How large is noise?** Table 14 trains Tülu 3 8B SFT with five seeds; the averages range from 59.8 to 60.1, a 0.3-point spread. The removal deltas on the average (−1.2 to −2.1) are four to seven times that spread. Per-benchmark seed variance is not reported, so single-benchmark changes near 1 point (BBH +1.0 without Math) cannot be separated from seed noise with the published data.

**Development versus unseen evaluations ([[tulu-3]] §7.4, Table 32).** IFEval-OOD is an unseen instruction-following test with 52 verifiable constraints that are not among IFEval's 25 (§7.3.1). IFEval-OOD scores are 17.6 for the full mixture, 18.0 without Persona, and 20.8 without WildChat. The Persona subset raised development IFEval by 19.2 points and did not raise IFEval-OOD. The authors write that "our choices overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning." The unseen averages still favor the final mixture (29.9 against 27.4 to 29.7 for the ablations).

**How the mixture was built (§4.1.2).** "We first built skill-specific data mixtures and models, keeping the mixtures that led to the best performance on individual skills, ignoring other evaluations," then combined them into a preview mix and iterated by adding or removing datasets, decontaminating, and downsampling large datasets. [[tulu-3-sft-mix]] lists the released components; its component list omits OpenMathInstruct 2 (50,000 prompts in Table 7) and sums to 889,344.

**Status.** Replicated: Tülu 1, Tülu 3, and DMT agree that no single source or ratio is best for all skills, in different models and data.

**Implication.** Every subset has a vector of effects across skills, some positive and some negative. Mixture decisions made on development benchmarks need a paired unseen evaluation per skill (ch-47) to detect subsets that raise only the development score.

## §4 Capability experts combined into one generalist

**Definition.** A capability expert is a model trained mainly for one capability (code, multilingual, reasoning, agent, chat). Expert-to-generalist designs move expert ability into one model through data (annotation, rejection sampling, distillation SFT) or through weights (merging, ch-30c).

**Problem.** A single mixture from the start forces all capabilities to share one training run, so one capability's data and RL cannot be tuned without affecting the others.

**Three published designs.**

| Report | Experts | How experts reach the generalist | Measured cost to specialist skills |
|---|---|---|---|
| Llama 3 ([[llama-3]] §4.3.1, §4.3.2) | Code expert: branch of pre-training continued on 1T tokens of >85% code, long-context fine-tuning to 16K, then post-trained. Multilingual expert: continued on 90% multilingual tokens, then post-trained | The code expert is used to collect human annotations and to rejection-sample responses for coding prompts; the multilingual expert is used to collect non-English annotations until pre-training finished | not reported |
| GLM-4.5 ([[glm-4-5]] §3, §3.1) | Reasoning, Agent, and General-chat experts, each with cold-start SFT then RL | "Overall SFT": millions of expert-generated samples, including long-context tasks, at up to 128K tokens; data with and without explicit thinking "meticulously balanced" | not reported |
| Qwen3 ([[qwen-3]] arXiv:2505.09388v1 §4.3, §4.7) | One lineage: long-CoT cold start → reasoning RL (Stage 2) | Stage 3 "Thinking Mode Fusion": continual SFT on the Stage 2 model with thinking data rejection-sampled by that model and curated non-thinking data; Stage 4 general RL | Table 22, below |

**Worked example (Qwen3-32B, Table 22).** Stage 2 → Stage 3 → Stage 4, thinking mode:

| Benchmark | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|
| AIME'24 | 83.8 | 81.9 | 81.4 |
| LiveCodeBench v5 | 68.4 | 67.2 | 65.7 |
| MMLU-Redux | 91.4 | 91.0 | 90.9 |
| IFEval strict prompt | 73.0 | 78.4 | 85.0 |
| ToolUse (in-house) | 63.3 | 70.4 | 85.5 |
| ThinkFollow (in-house mode switching) | not applicable | 88.7 | 98.9 |

- AIME'24 falls 2.4 points and LiveCodeBench falls 2.7 points across the two stages, while IFEval rises 12.0 and ToolUse rises 22.2.
- AIME'24 has 30 questions, each sampled 64 times (§4.6). One question is 3.33 points of average accuracy, so the 2.4-point decline equals 0.72 of one question's accuracy (derived). No confidence interval is reported.
- The report states: "We conjecture this degradation is due to the model being trained on a broader range of general tasks, which may compromise its specialized capabilities in handling complex problems. During the development of Qwen3, we choose to accept this performance trade-off to enhance the model's overall versatility" (§4.7) (Interpretation).

**Conditions and limits.** None of the three reports compares the expert route against one model trained on a direct mixture of the same data, so the benefit of experts is not measured. GLM-4.5 and Llama 3 do not print per-expert scores next to the generalist. In a separate 8B study, merging a general-only and a safety-only DPO checkpoint of a pre-release Aya 23 8B model with SLERP (spherical linear interpolation of the two weight vectors) scored 78.0% win rate and −57.8% relative harm against 71.0% and −54.7% for one model trained on the 15% safety mixture ([[mix-data-or-merge-models]], Table 1); ch-30c covers that comparison.

**Implication.** Expert routes allow per-domain RL before combination, at the cost that the combination step is itself an SFT mixture with the interference problems of this chapter's §2-§3. ch-35 maps where labs insert expert data.

## §5 The agentic share of the mixture

**Definition.** Agentic SFT data are multi-turn trajectories in which the target includes tool calls or actions and the context includes environment observations.

**Problem.** Agent trajectories differ from chat in format and length. An agent-only mixture can overfit the format and lower general and held-out agent ability, while a mixture without agent data leaves held-in agent tasks low (AgentLM-7B general-only held-in score 0.38 against 1.96 mixed, [[agenttuning]] Table 5).

**AgentTuning ([[agenttuning]], Llama-2-chat 7B/13B/70B).** The mixture objective is

```
J(θ) = η · E_(x,y)~D_agent [log π_θ(y | x)] + (1 − η) · E_(x,y)~D_general [log π_θ(y | x)]
```

π_θ(y | x) is the model's probability of response y given instruction and history x; D_agent is 1,866 filtered trajectories from six AgentBench tasks, generated mainly by GPT-4; D_general is ShareGPT; η is the agent sampling ratio (§2.2.2, Eq. 1; App. A). η was scanned from 0 to 1 in steps of 0.1 at 7B, and η = 0.2 was best on held-out agent tasks. Table 6 prints a batch size of 64 without a unit; read as examples, η = 0.2 gives an expected 12.8 agent examples per batch (derived).

Table 5 (held-in agent / held-out agent / general, normalized scores):

| Size | Mixed η = 0.2 | General only | Agent only |
|---|---|---|---|
| 7B | 1.96 / 0.67 / 0.63 | 0.38 / 0.64 / 0.61 | 1.34 / 0.09 / 0.22 |
| 70B | 2.55 / 1.40 / 0.96 | 0.99 / 0.98 / 1.00 | 2.47 / 0.87 / 0.83 |

Agent-only data raises held-in agent tasks and lowers held-out agent tasks and general ability. At 7B the mixture's held-out score is close to general-only (0.67 vs 0.64); the held-out advantage of mixing appears at 70B (1.40 vs 0.98).

**Agent-FLAN ([[agent-flan]], Llama2 7B ablations).** T-Eval is a held-out tool-utilization benchmark in this paper's evaluation (§4). Three changes to the agentic data: ReAct-format turns are rewritten as multi-turn chat (T-Eval 61.8 → 64.9), data are decomposed by capability and reweighted, and negative samples are added (see the negative-samples section below). Halving reasoning data lowers T-Eval to 63.8 and halving understanding data lowers it to 64.6, while halving retrieval (65.3) or instruction-following data (65.9) does not lower it (Table 2). The weighted mixture uses reasoning : retrieval : understanding = 1 : 0.25 : 0.75 and 18.1M tokens instead of 37.3M, and scores T-Eval 66.3 (Table 2; App. A). ShareGPT is mixed 1:1 with the agent corpus, unit not stated (App. A). General scores at 7B move from MMLU 50.0 to 49.7, GSM8K 21.9 to 22.1, and HumanEval 15.1 to 15.5 (Table 4).

**Agent Data Protocol ([[agent-data-protocol]], Qwen2.5-Coder-Instruct 7B-32B, Qwen2.5-7B-Instruct, and Qwen3-8B).** Each dataset is resampled with m_d = ⌈w_d · n_d⌉ (App. C). Worked example with the non-web subset used for OpenHands and SWE-Agent (Table 1 counts, Table 9 multipliers). App. C.1 excludes Mind2Web, Go-Browse, NNetNav, and Synatra but does not say whether the Mind2Web and WebShop subsets inside AgentInstruct are kept; this example keeps all of AgentInstruct:
- Orca AgentInstruct: 1,046.1K raw trajectories × 0.001 = 1.05K sampled. Its share falls from 91.71% of raw non-web trajectories to 3.74% of sampled examples (derived).
- Code-Feedback: 66.4K × 0.1 = 6.64K. Nebius SWE trajectories: 13.4K × 0.2 = 2.68K. SWE-Gym: 0.5K × 3 = 1.5K. AgentInstruct: 1.9K × 2 = 3.8K. CodeActInstruct 7.1K, SWE-smith 5.0K, and openhands-feedback 0.2K keep w = 1.
- Total: 27.97K sampled examples (derived from counts rounded to 0.1K); App. C.1 states "around 30K".
- Web datasets were excluded from this subset "to avoid potential interference from web-specific interaction patterns" (App. C.1). No ablation of the multipliers is reported.

Results (Table 6; App. E.1 Table 10): on SWE-bench Verified with OpenHands, Qwen3-8B scores 0.2% after CodeActInstruct + Code-Feedback, 11.0% after SWE-smith only, and 16.6% after the ADP mixture; at a matched scale of about 30K samples, SWE-smith up-sampled still scores 11.0%. On GAIA, Qwen2.5-7B-Instruct scores 0.6% after AgentInstruct only and 9.1% after ADP.

**LIMI ([[limi]], GLM-4.5 355B).** 78 successful trajectories collected with GPT-5 and human collaborators, averaging 42.4k tokens (maximum 152k), about 3.3M tokens in total (derived). Table 2 and Table 3 compare fine-tuning GLM-4.5 on four sets. AgencyBench scores 10 multi-round vibe-coding and research tasks; tau2 airline and retail are τ²-bench tool-use tasks scored as Pass^4, the fraction of four independent runs that succeed (§3.4):

| Training set | Samples | AgencyBench avg | tau2 airline | tau2 retail |
|---|---|---|---|---|
| none (GLM-4.5) | none | 45.1 | 28.0 | 36.8 |
| CC-Bench trajectories | 260 | 29.2 | 38.0 | 39.6 |
| AFM-WebAgent-SFT | 7,610 | 36.7 | 18.0 | 13.2 |
| AFM-CodeAgent-SFT | 10,000 | 47.8 | 20.0 | 16.7 |
| LIMI | 78 | 73.5 | 34.0 | 45.6 |

- The 7,610-sample and 10,000-sample single-domain sets lower tau2 retail from 36.8 to 13.2 and 16.7, and tau2 airline from 28.0 to 18.0 and 20.0, in one run each. This course reads it as narrowing on a tool-use task that the sets do not target (Interpretation).
- The abstract's "53.7% improvement" is relative: (73.5 − 47.8) / 47.8 = 53.8%; the difference is 25.7 points (derived). §4.3 calls it a "percentage point" improvement.
- The printed Table 3 average for LIMI (57.2) cannot be reproduced as an unweighted mean of the printed columns (45.6 without AgencyBench, 49.1 with it; derived).
- AgencyBench has 10 tasks from the authors' group, in the same two domains as the training queries. Samples are not token-matched, no seeds are reported, and the only non-agentic benchmarks are coding and scientific-computing sets (EvalPlus, DS-1000, SciCode); knowledge, chat, and safety benchmarks are not reported. Without CLI tools, GLM-4.5 scores higher than LIMI on tau2 retail (52.6 vs 49.1, Table 4).

**Llama 3.** "Reasoning and tools" is 21.19% of SFT examples (Table 7), and tool data include queries that need no tool so that the model does not call tools unnecessarily ([[llama-3]] §4.3.5). The tool-only share is not reported.

| Source | Agentic share as printed | Unit | Non-agentic ability reported |
|---|---|---|---|
| AgentTuning | η = 0.2 | sampling ratio | yes (MMLU, HumanEval, GSM8K, MT-Bench) |
| Agent-FLAN | ShareGPT : agent 1:1 | not stated | yes (MMLU, GSM8K, HumanEval) |
| ADP | multipliers w_d; about 30K non-web samples | examples per epoch | no |
| LIMI | 78 agentic samples only | examples | coding and scientific computing only (EvalPlus, DS-1000, SciCode) |
| Llama 3.1 | Reasoning and tools 21.19% | % of examples | yes (full report) |

**Implication.** The agentic evidence supports three conditional rules: mix agent data with general data (AgentTuning), balance agent sources and capabilities rather than raw counts (Agent-FLAN, ADP), and treat single-domain agent sets (7,610 and 10,000 samples in LIMI's comparison) as a narrowing risk for other agent tasks (LIMI, ADP). ADP reports no non-agentic benchmark and LIMI reports only coding and scientific-computing benchmarks, so the effect of their mixtures on knowledge, chat, and safety is an Open question. ch-45d compares agentic recipes across stages.

## §6 The long-context share of the mixture

**Definition.** Long-context SFT data are examples whose context uses the extended context window of the model. Llama 3.1 long-context examples average 38,135.6 tokens against 846.1 for its whole SFT mix, a factor of 45 (Table 7), and are bucketed at 16K, 32K, 64K, and 128K tokens.

**Problem.** In both reports compared below, the context window is extended by continued pre-training before SFT (ch-32b). The measurable question is whether the long-context benchmark score after SFT is lower with a short-only SFT mixture than with a mixture that includes synthetic long examples.

**Evidence A: Llama 3 ([[llama-3]] §4.3.4; Table 7).** "Naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training." The long data are synthetic QA over 8K chunks with the full document as context, hierarchical summaries, and repository code reasoning, generated by earlier Llama 3 models. "Through careful ablations, we observe that mixing 0.1% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks." No ablation table is printed. Short-context DPO did not hurt long-context performance when the SFT model was strong on long context; the authors suspect this is because DPO uses fewer optimizer steps than SFT.

**Evidence B: ProLong ([[prolong]] §5, Table 8; the ProLong base model, Llama-3-8B-Instruct after 40B tokens of long-context continued training up to 512K, then 1B tokens of UltraChat SFT).** Synthetic long data (40% QA, 30% RAG, 30% summarization, generated by Llama-3-8B-Instruct) mixed with UltraChat, share measured in tokens:

| Synthetic share (tokens) | JsonKV | RAG | Re-rank | ICL | QA | Summ. | Avg |
|---|---|---|---|---|---|---|---|
| 0% | 65.7 | 58.1 | 38.5 | 80.3 | 49.7 | 42.1 | 55.7 |
| 1% | 61.5 | 57.0 | 38.3 | 80.8 | 45.3 | 41.5 | 54.1 |
| 3% | 62.0 | 56.4 | 37.9 | 80.6 | 44.8 | 39.5 | 53.5 |
| 10% | 70.3 | 55.5 | 36.1 | 80.6 | 41.7 | 39.4 | 53.9 |
| 50% | 45.8 | 48.8 | 18.8 | 70.5 | 42.3 | 33.3 | 43.3 |

The columns are HELMET tasks (§2.1): JsonKV retrieves a value for a key from a long JSON file; RAG answers questions over retrieved Wikipedia passages; Re-rank orders shuffled documents; ICL learns classification from many in-context examples; QA answers questions about a full book; Summ. summarizes long legal documents. JsonKV, QA, and Summ. are evaluated at 512K; the other columns average 32K and 64K. A Llama-3-70B-Instruct generator gave 54.2, 55.4, 53.6, and 49.7 at 1%, 3%, 10%, and 50% (App. B.5, Table 23), all below 55.7.

**The conflict, stated with conditions.** Llama 3's 0.11% example share is 4.95% of tokens (this chapter's §1), which lies between ProLong's 3% and 10% token shares; both averaged below the 0% run in ProLong (53.5 and 53.9 against 55.7). The units do not resolve the disagreement.

| Condition | Llama 3.1 | ProLong |
|---|---|---|
| Model | 8B, 70B, 405B (size for the 0.1% ablation not stated) | Llama-3-8B-Instruct initialization |
| Long-context continued training | about 800B tokens in six stages, 8K → 128K (stated for 405B, §3.4.2) | 40B tokens, 64K then 512K |
| SFT data | internal, several rounds, rejection-sampled and synthetic | UltraChat, 1B tokens |
| Long SFT data | QA, summarization, repository code; earlier Llama 3 models | QA, RAG, summarization; Llama-3-8B or 70B-Instruct |
| Evaluation | not printed | HELMET subset at 32K and 64K, some tasks at 512K |

ProLong's authors give two hypotheses: prior work may have had insufficient long-context training, so synthetic data acted as extra long-context training; and "it is possible that when using an extensive short instruction dataset, mixing in synthetic long data avoids the model from degenerating on long-context tasks" (§5). **Status: Open question.** A resolving experiment would vary the short SFT size and the long SFT share in one model and report long and short benchmarks with seeds.

GLM-4.5 includes long-context understanding tasks in its unified SFT at up to 128K tokens without printing their share ([[glm-4-5]] §3.1). The library card [[long-context-llama3]] has not been re-verified; the numbers in this section are quoted from the Llama 3 report through [[llama-3]] and [[llama-3-recipe]].

**Implication.** When the base model has had long-context continued training and the SFT set is about 1B tokens of short open chat data, start from a short-only SFT mixture, because that setting matches ProLong. When the SFT set is multi-round with rejection-sampled and synthetic data, include a long-context share near 0.1% of examples and measure, because that setting matches Llama 3. In both cases the long-context benchmark is a required gate for every mixture change (ch-32c).

## §7 Reasoning and non-reasoning modes in one mixture

**Definition.** A reasoning mode produces an explicit reasoning trace before the answer; a non-reasoning mode answers directly. A mode control (system prompt or flag) selects the mode at inference.

**Problem.** Samples in reasoning-heavy domains are long: in DeepSeek-R1 Dev3 SFT, the Math, Code, STEM, and Logic domains are 77.90% of samples and 94.14% of tokens (derived from Table 5, §1). Reasoning-focused SFT also lowered IFEval in Llama-Nemotron (below), so a two-mode mixture has to be checked for instruction-following loss in both modes.

**Llama-Nemotron ([[llama-nemotron]], arXiv:2505.00949v5).** Reasoning samples carry the system prompt "detailed thinking on" and non-reasoning samples "detailed thinking off". Paired data take prompts randomly sampled from the reasoning dataset and add a non-reasoning response generated by Llama-3.1-Nemotron-70B-Instruct (general domain) or Llama-3.3-70B-Instruct (other domains) (§3.2). Table 2 lists 33,011,757 SFT samples; the reasoning-on rows sum to 3,934,627 (11.9%, derived). Math reasoning-off alone is 60.1% of samples.
- LN-Nano (8B) trains in three stages: reasoning data only at LR 1e-4 for four epochs, which "prevents failure modes such as repetitive completions"; then reasoning plus non-reasoning data; then a smaller chat, instruction-following, and tool-calling blend (§4.2).
- LN-Nano scores AIME24 61.3 with reasoning on and 3.0 with reasoning off, while BFCL V2 Live (Berkeley Function-Calling Leaderboard, live split) is 63.9 and 63.6 (Table 3). The toggle changes AIME24 by 58.3 points and function calling by 0.3 points.
- LN-Super-SFT (49B) scores IFEval 81.9 (on) and 83.0 (off) against 92.1 for Llama-3.3-70B-Instruct, the model from which LN-Super was derived by architecture search and 40B tokens of knowledge distillation (§2.1-§2.2); the report states that "reasoning-focused SFT causes a noticeable drop in IFEval scores" and adds a dedicated IFEval RL run (§7.3). The released LN-Super, after that run, online-RPO RLHF, and model merging, scores 89.2 and 89.0 (§6, §7.3, Table 4).

**DeepSeek-R1 ([[deepseek-r1]], [[deepseek-r1-recipe]]).** The Dev3 SFT set combines about 600k reasoning samples with about 200k non-reasoning samples (writing, factual QA, self-cognition, translation, and software engineering data). For "simpler queries, such as hello," no chain of thought is provided (Supp. B.3.3). From Dev2 to Dev3 (Table 3), AlpacaEval 2.0 rises 55.8 → 62.1, IF-Eval 72.0 → 78.1, Aider-Polyglot 25.6 → 44.8, and AIME 2024 74.0 → 78.1, while SimpleQA falls 28.2 → 24.9 and C-Eval 91.9 → 86.4. Dev3 restarts SFT from DeepSeek-V3-Base with new data, so these differences are not a controlled ablation of the non-reasoning share.

**Qwen3 ([[qwen-3]] §4.3, Table 9).** Non-thinking samples use a `/no_think` flag and keep an empty think block:

```
<|im_start|>user
{query} /no_think<|im_end|>
<|im_start|>assistant
<think>

</think>

{response}<|im_end|>
```

Thinking is the default, so some thinking samples omit the `/think` flag; in multi-turn data several flags are inserted and the response follows the last flag.

**Related result.** For a non-reasoning Qwen2.5-32B student, SFT on the answer segment of DeepSeek-R1 outputs scored IFEval 76.9 (73.4 when trained on the original community responses), while prepending a summary of the thinking scored 61.2 ([[am-reasoning-answers-for-non-reasoning]], Table 1).

**Implication.** A two-mode mixture needs per-mode evaluation (both modes on the same suite), a mode-control metric such as ThinkFollow, and a token-unit account of each share. In DeepSeek-R1 Dev3 the General domain is 22.10% of samples and 5.86% of tokens; Table 5 is split by domain, not by reasoning and non-reasoning data, so the token share of the about 200k non-reasoning samples is not reported.

## §8 Data diversity measures used to balance a mix, and their limits

**Definition.** A diversity measure assigns a number to how widely a set of examples covers some space: embedding space, tag space, or task labels.

**Problem.** Quality-only selection can pick near-duplicate high-scoring examples, which leaves parts of the instruction distribution uncovered.

**Facility location (QDIT, [[qdit-data-diversity]] §3).**

```
d(A) = Σ_{v∈V} max_{a∈A} sim(a, v)
f(a | A, α) = (1 − α) · d(a | A) + α · q(a)
```

V is the candidate pool; A is the selected subset; sim is the cosine similarity of instruction embeddings (all-mpnet-base-v2); d(a | A) = d(A ∪ {a}) − d(A) is the marginal diversity gain; q(a) is a quality score from a reward model or ChatGPT rating; α ∈ [0, 1] sets the trade-off. Selection adds the example with the highest f at each step.

**Worked example (four prompts, select two).** Similarities: sim(v1, v2) = 0.9 (two near-duplicate math prompts), sim(v1, v3) = 0.1, sim(v1, v4) = 0.2, sim(v2, v3) = 0.1, sim(v2, v4) = 0.2, sim(v3, v4) = 0.3, and sim(v, v) = 1. Quality: q = 0.9, 0.8, 0.5, 0.6.
1. Step 1, α = 0.5: d({v1}) = 1 + 0.9 + 0.1 + 0.2 = 2.2, so f(v1) = 0.5 × 2.2 + 0.5 × 0.9 = 1.55. f(v2) = 1.50, f(v3) = 1.00, f(v4) = 1.15. Select v1.
2. Step 2: d({v1, v2}) = 1 + 1 + 0.1 + 0.2 = 2.3, gain 0.1, f = 0.05 + 0.40 = 0.45. d({v1, v3}) = 1 + 0.9 + 1 + 0.3 = 3.2, gain 1.0, f = 0.50 + 0.25 = 0.75. d({v1, v4}) = 3.2, gain 1.0, f = 0.50 + 0.30 = 0.80. Select v4.
3. Quality-only selection (α = 1) picks {v1, v2}: mean quality 0.85, diversity 2.3. QDIT at α = 0.5 picks {v1, v4}: mean quality 0.75, diversity 3.2.

The paper does not state how d and q are brought to a common scale; the example uses raw values.

**Evidence.** On Ultrachat 1.3M → 10K with LLaMA-1 7B, the lowest-10% reward-model score is 2.620 (random), 3.405 (quality-only), and 3.497 (QDIT), with means 6.219, 6.961, and 6.993 (Table 1). In the main results, QDIT improves worst-case reward score by 5.2% and worst-case winning score against Alpaca 52K by 6.26% over quality-only selection (§4.3). α ∈ {0.5, 0.7, 0.9} is typically best; α = 0.1 and α = 1 both perform worse (§4.4).

**Tags and embedding thresholds.** InsTag ([[instag]]) assigns open-set intent tags to instructions. DMT used the tags to remove code and math examples from ShareGPT (this chapter's §2). Llama 3 used the number of InsTag intentions as a difficulty score and ran semantic deduplication inside RoBERTa clusters with a cosine threshold that is not printed ([[llama-3-recipe]], §4.2.3). DEITA's embedding-distance filter (threshold 0.9) scored MT-Bench 6.17 against 6.10 for InsTag diversity and 5.82 for random selection when selecting 6K samples from a 300K pool at LLaMA-1 13B ([[deita]], Table 4). QDIT reports that threshold de-duplication and cluster-balanced selection give lower worst-case performance than facility location (§4.4, Fig. 8a).

**Limits.**
- All four measures describe coverage of the candidate pool, not coverage of the deployment distribution or of skills absent from the pool.
- Evaluations are judge- or reward-model-based at 7B-13B, with short maximum lengths (512 tokens in QDIT).
- None of the studies selects across skill domains such as math, agentic, and long-context data; balancing domains still requires the per-skill evaluations of this chapter's §2-§3.
- ADP attributes its matched-scale gain to "the greater diversity and unified structure" of its corpus (App. E.1; Interpretation), but that comparison controls sample count, not domain composition. Agent-FLAN concludes from its data-scaling curve that more diversity or quality, not more volume, is the next step (§5.1.1; Interpretation); neither paper measures diversity with a number.

**Implication.** A diversity measure can balance selection within one pool when it is paired with a quality score (QDIT's α ∈ {0.5, 0.7, 0.9}). The share of each skill domain in a general-purpose mixture is not set by these measures and still has to be chosen with per-skill and unseen evaluations.

## Negative samples and negative feedback

This chapter uses "negative" in the style-guide §6.1 sense. SFT mixtures contain types 1 and 2; the chapter's sources use no negative gradients (type 4).

1. **Where negatives come from.** Execution and task success (AgentTuning keeps trajectories with reward r = 1, and r ≥ 2/3 for Mind2Web; Table 1 keeps 1,866 of 35,341), judge agents (GLM-4.5 keeps agent trajectories that multiple judge agents mark as completed, §3.1), reward models (Llama 3 top-quartile quality, §4.2.3), and removal ablations that reveal a subset with negative marginal value for one skill (Tülu 3 WildChat for the safety score, this chapter's §3). No false-negative rates are reported.
2. **Type 1, negative marginal value, discarded.** AgentTuning trained on unfiltered trajectories scores held-in 1.34 and held-out 0.47 at 7B, against 1.96 and 0.65 filtered (Table 2). In a DMT ablation, removing InsTag-tagged code and math examples from ShareGPT reduced the high-resource conflict "to some extent" (§4.2). A subset can have negative value for one skill and positive value for another, so "discard" decisions need all skills.
3. **Type 2, negative as content.** Agent-FLAN adds prompts that invite a tool call but whose correct target is a plain-text reply (no tools given, or only irrelevant tools given), built from 761 ToolBench queries (App. C). At 7B they lower ReAct-keyword hallucination from 15.6 to 9.9 and general-keyword hallucination from 13.5 to 11.9, with T-Eval 66.3 → 66.0 ([[agent-flan]] Table 3). Llama 3 adds queries that need no tool (§4.3.5) and knowledge-probe refusals (§4.3.6). Tülu 3 adds CoCoNot contrastive prompts, which the authors report helped prevent over-refusal of safe prompts (§4.2; no numbers).
4. **Mechanism.** Type 2 samples are trained with ordinary cross-entropy on the corrected target. The gradient of log p_y with respect to logit z_j is 1[j = y] − p_j, so raising the plain-text target token lowers the probability of the competing tool-call token only through the shared softmax; no sample's likelihood is pushed down directly (ch-30 covers this implicit suppression).
5. **Controls.** Agent-FLAN builds its negative-as-content samples from 761 source queries against 24,703 agent samples, and their effect on the positive capability is checked (T-Eval change −0.3).
6. **Diagnostics.** Tool-call rate on prompts that need no tool, refusal rate on safe prompts (CoCoNot-style), and per-skill deltas after each filter change.
7. **Effect on generality.** Type 2 samples target over-triggering (tool calls, refusals), which is a narrowing failure; their effect on unseen tool-use tasks is not reported.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DMT runs (LLaMA-1) | 7B, 13B, 33B | SFT | epochs; peak LR; warmup | 3; 2e-5; 3% | arXiv:2310.05492v4 §3.1, App. C | verified 2026-09-15 | no ablation reported |
| DMT runs (LLaMA-1) | 7B, 13B, 33B | SFT | batch size | 16 (§3.1); 128 (App. C) | §3.1; App. C | conflict | the paper does not state which value produced Table 1 |
| DMT runs (LLaMA-1) | 7B, 13B, 33B | SFT | full dataset sizes | GSM8K RFT 110,142; Code Alpaca 20,022; ShareGPT 86,060 examples | App. A Table 2 | verified 2026-09-15 | not applicable |
| DMT (k = 1/256) | 7B, 13B, 33B | SFT | stage-2 replay | ShareGPT plus 1/256 of GSM8K RFT and Code Alpaca (430 + 78 examples) | Table 1; Table 2 | verified 2026-09-15 (k); derived (counts) | Table 1: 7B GSM8K 41.92 vs 32.60 mixed sequential; Fig. 5 k sweep |
| Tülu 1 Human+GPT mix | 7B-65B | SFT | mixing rule; instances | 7 datasets concatenated with no sampling weights; 490,694 | arXiv:2306.04751v2 §3.3, Table 1 | verified 2026-09-14 ([[tulu-1-how-far-can-camels-go]]); total derived | Table 3: mix average 45.2 vs 42.0 for best single dataset (13B) |
| Tülu 3 8B SFT | 8B | SFT | prompts | 939,344 | arXiv:2411.15124v5 Table 7 | verified 2026-09-15 | Table 10 removal ablations |
| Tülu 3 8B SFT | 8B | SFT | peak LR; schedule; warmup ratio; epochs; effective batch; max length; loss aggregation | 5e-6; linear; 0.03; 2; 128; 4,096; sum loss | Table 11; §4.3; §4.3.2 | verified 2026-09-15 | §4.3.2 Figs. 5-6 (Llama 3.0 on the Tülu 2 mixture, not the final mix): sum loss at 5e-6 best across LRs and loss types; 2 epochs best; figures only, seeds not stated |
| Tülu 3 70B SFT | 70B | SFT | peak LR (other settings as 8B) | 2e-6 | Table 11 | verified 2026-09-15 | same as above |
| Tülu 3 8B SFT | 8B | SFT | compute | 32 GPUs for 6 hours (8×H100 nodes) | §4.3 | verified 2026-09-15 | not applicable |
| Tülu 3 8B / 70B SFT | 8B, 70B | eval-gate | checkpoint selection | best single seed run, not a soup | §4.3.1, Table 14 | verified 2026-09-15 | Table 14: 8B seeds 59.8-60.1, best soup 60.2 |
| Llama 3.1 | not scoped | SFT | mixture | General English 52.66%, Code 14.89%, Multilingual 3.01%, Exam-like 8.14%, Reasoning and tools 21.19%, Long context 0.11% of examples | arXiv:2407.21783v3 Table 7 | verified 2026-09-14 ([[llama-3-recipe]]) | mix adjusted each round "to tune performance across a wide range of benchmarks" (§4.2.2); no table |
| Llama 3.1 | not scoped | SFT | long-context share | 0.1% synthetic long-context data; buckets 16K, 32K, 64K, 128K | §4.3.4 | verified 2026-09-14 ([[llama-3-recipe]]) | "careful ablations"; no table |
| Llama 3.1 | not scoped | SFT | long-context token share | 4.95% of tokens (0.0011 × 38,135.6 ÷ 847.1) | Table 7 | derived | not applicable |
| Llama 3.1 405B | 405B | SFT | LR; steps | 1e-5; 8.5K-9K steps ("our largest models") | §4.1.3 | verified 2026-09-14 ([[llama-3-recipe]]) | "work well across different rounds and data mixes"; no numbers |
| AgentLM-7B/13B/70B | 7B, 13B, 70B | SFT | agent sampling ratio η; general data | 0.2; ShareGPT with GPT-4 : GPT-3.5 = 0.2 : 0.8 | arXiv:2310.12823v2 §2.2.1-§2.2.2; App. A | verified 2026-09-14 ([[agenttuning]]) | η scanned 0-1 in 0.1 steps at 7B, best held-out (per-η scores not tabulated); Table 5 endpoints |
| AgentLM-7B / 13B; AgentLM-70B | 7B, 13B; 70B | SFT | peak LR | 5e-5; 1e-5 | Table 6 | verified 2026-09-14 ([[agenttuning]]) | no ablation reported |
| Agent-FLAN (Llama2) | 7B, 13B, 70B | SFT | ShareGPT : agent; format mix; capability weights | 1:1 (unit not stated); 10% ReAct, 90% conversation; reasoning : retrieval : understanding = 1 : 0.25 : 0.75 | arXiv:2403.12881v1 App. A | verified 2026-09-14 ([[agent-flan]]) | Table 2 half-data ablation at 7B; the 1:1 ratio conflicts with AgentTuning's η = 0.2 that App. A says it follows |
| Agent-FLAN (Llama2) | 7B, 13B, 70B | SFT | epochs; peak LR | 1; 2e-5 | App. A Table 5 | verified 2026-09-14 ([[agent-flan]]) | no ablation reported |
| ADP (OpenHands, SWE-Agent runs) | 7B-32B | SFT | per-dataset multipliers w_d | orca agentinstruct 0.001; synatra 0.01; code feedback 0.1; nebius SWE 0.2; agenttuning subsets 2; swe-gym openhands 3; others 1 | arXiv:2510.24702v2 App. C Table 9 | verified 2026-09-15 | no ablation reported |
| ADP (OpenHands, SWE-Agent runs); ADP (AgentLab runs) | 7B-32B | SFT | subset; sampled size | non-web only, around 30K; web only, around 20K | App. C.1 | verified 2026-09-15 | Table 10: 16.6 vs 11.0 SWE-bench Verified at matched scale (Qwen3-8B) |
| ADP | 7B-32B | SFT | LR; epochs; sequence length | not reported (LLaMA-Factory pipeline) | checked §5.1, App. C-E | not reported | not applicable |
| LIMI (from GLM-4.5) | 355B | SFT | samples; tokens per sample | 78; mean 42.4k, max 152k | arXiv:2509.17567v2 §3.2, Fig. 4 | verified 2026-09-15 | Table 2 vs 260 / 7,610 / 10,000-sample sets, one run each |
| LIMI | 355B | SFT | LR; epochs; batch | not reported (slime framework) | checked §4.1 | not reported | not applicable |
| LN-Nano | 8B | SFT | stages; stage-1 LR and epochs; batch; packed length | 3 stages; 1e-4, 4 epochs; 256; 32k | arXiv:2505.00949v5 §4.2 | verified 2026-09-15 | stage 1 "prevents failure modes such as repetitive completions"; no table |
| LN-Super | 49B | SFT | LR; epochs; length; batch | fixed 5e-6; 1; 16k; 256 | §4.2 | verified 2026-09-15 | smaller runs suggested gains up to 3-4 epochs at 5e-5; no table |
| LN-Ultra | 253B | SFT | LR schedule; packed length; batch | warmup to 1e-5 over 10%, cosine to 1e-6; 24k; 256 | §4.2 | verified 2026-09-15 | initial ablations: LRs such as 5e-5 generally improved outcomes, but consistently high LRs caused instability including gradient explosions; no table |
| Llama-Nemotron SFT data | 8B-253B | SFT | samples; reasoning-on share | 33,011,757; 3,934,627 reasoning-on (11.9%) | Table 2 | verified 2026-09-15; share derived | no ablation reported |
| DeepSeek-R1 Dev3 | 671B MoE | SFT | samples by domain | Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812; total 804,745 | arXiv:2501.12948v2 Supp. B.3.3 Table 5 | verified 2026-09-14 ([[deepseek-r1-recipe]]) | Table 3 Dev2 → Dev3, not a controlled ablation |
| DeepSeek-R1 Dev3 | 671B MoE | SFT | General token share | 5.86% (177,812 × 1,419.8 ÷ 804,745 × 5,355.3) | Table 5 | derived | not applicable |
| DeepSeek-R1 Dev3 | 671B MoE | SFT | epochs | 2-3 (v2 B.4.2); two (v1 §2.3.3) | v2 B.4.2; v1 §2.3.3 | conflict | neither version states which count produced the released checkpoint |
| Qwen3-32B Stage 3 | 32B | SFT | data size; thinking : non-thinking ratio | not reported | checked arXiv:2505.09388v1 §4.3, Table 9, §4.7 | not reported | Table 22 stage-by-stage scores |
| GLM-4.5 unified SFT | 355B | SFT | samples; maximum context | "millions of samples"; 128K tokens | arXiv:2508.06471v1 §3.1 | verified 2026-09-14 ([[glm-4-5]]) | no ablation reported; thinking balance ratio not reported |
| ProLong-512k-Instruct | 8B | SFT | data; synthetic long share; peak LR; global batch | UltraChat 1B tokens; 0% (token share); 2e-5; 4M tokens | arXiv:2410.02660v4 Table 9; Table 8 | verified 2026-09-14 ([[prolong-recipe]]) | Table 8: 0% avg 55.7 vs 54.1, 53.5, 53.9, 43.3 at 1%, 3%, 10%, 50% |
| QDIT selection (LLaMA-1) | 7B | SFT | α; selected size; LR; epochs; batch; max length | per-dataset α (App. Table 5), {0.5, 0.7, 0.9} typically best; 10K; 2e-5; 3; 128; 512 | arXiv:2311.14736v3 §4.2, §4.4, App. D Table 4 | verified 2026-09-15 | Fig. 4 α sweep |

**Starting point for a small general-purpose run.** The verified rows come from different setups and were never tested together. For a 7B-8B model with open data, the Tülu 3 8B SFT settings are a documented baseline: 939,344 prompts, sum loss with peak LR 5e-6 (the LR was selected for the sum loss), linear schedule, warmup ratio 0.03, 2 epochs, effective batch 128, maximum length 4,096, on Llama 3.1 8B with 32 GPUs for 6 hours. If a specialist skill is trained in an earlier stage, DMT's final-stage replay fraction k = 1/256 was tested on LLaMA-1 7B-33B with 3 epochs at LR 2e-5. If agent trajectories are added to general chat data, AgentTuning's η = 0.2 was selected at Llama-2-chat-7B on held-out agent tasks with ShareGPT as the general data. For long-context data, the two verified values disagree (0.1% of examples in Llama 3.1 with no model size stated; 0% in ProLong at 8B after 40B tokens of long-context training), so the share is a hyperparameter to measure, not a default.

## Generalization lens

**(a) What increases breadth.**
- Mixing sources at low data per skill: at 1/256 of each dataset, mixed sources score above single sources, with LLaMA-7B code listed by the authors as an outlier ([[sft-data-composition-dmt]] §3.3, Fig. 3).
- Real user conversations: removing WildChat from Tülu 3 8B lowers AlpacaEval 2 by 4.9 and BBH by 2.3 ([[tulu-3]] Table 10).
- General data alongside agent data: AgentLM-70B held-out agent score 1.40 with η = 0.2 against 0.87 agent-only ([[agenttuning]] Table 5).
- Cross-domain agentic mixtures over single-domain sets: SWE-bench Verified 16.6% vs 11.0% at matched sample count ([[agent-data-protocol]] Table 10).
- Diversity-aware selection for worst-case prompts: +5.2% worst-case reward score over quality-only selection ([[qdit-data-diversity]] §4.3).

**(b) What causes narrowing or forgetting.**
- Full-size mixing at high resource: HumanEval −7.92 at 33B for multi-task against code-only ([[sft-data-composition-dmt]] Table 1).
- Sequential skill stages without replay: GSM8K 31.39 vs 49.10 at 7B (same table).
- Single-domain agent sets: tau2 retail 36.8 → 13.2 (web set) and 16.7 (code set) for GLM-4.5 ([[limi]] Table 3); agent-only AgentLM-7B general score 0.22 vs 0.63 mixed ([[agenttuning]] Table 5).
- Broader general SFT and RL after reasoning RL: Qwen3-32B thinking AIME'24 83.8 → 81.4 and LiveCodeBench 68.4 → 65.7 ([[qwen-3]] Table 22).
- Reasoning-focused SFT: LN-Super-SFT IFEval 81.9-83.0 vs 92.1 for Llama-3.3-70B-Instruct, the model it was derived from; the comparison also includes the architecture search and distillation to 49B, and the report attributes the drop to reasoning-focused SFT ([[llama-nemotron]] §7.3, Table 4).
- Development-benchmark tuning: Persona data +19.2 IFEval and −0.4 IFEval-OOD in Tülu 3 8B ([[tulu-3]] Table 32).
- At the RL stage, the Nemotron 3 white paper cites DeepSeek-V3.2-Exp for the observation that staged single-task approaches often degrade some capabilities, and trains all environments together; no staged-versus-simultaneous numbers are printed ([[nemotron-ultra]] §2.6).

**(c) How to measure it for this stage.**
- For every added or removed subset, report every skill's delta against the unchanged mixture, with seeds; Tülu 3's 8B seed spread on the average is 0.3 points (Table 14), and per-benchmark spread must be measured locally.
- Pair each development benchmark with an unseen benchmark of the same skill (Tülu 3 §7.4 pairs IFEval with IFEval-OOD, HumanEval with BigCodeBench).
- For agentic data, report held-in agent tasks, held-out agent tasks, and non-agentic general benchmarks (AgentTuning's three groups); ADP omits the third group, and LIMI reports only coding and scientific-computing benchmarks for it.
- For long-context data, report long-context tasks after SFT, not before (ProLong §2.2), and short-context tasks in the same run.
- For two-mode models, report both modes on the same suite and a mode-switching metric (Qwen3 ThinkFollow).
- Report mixture shares in examples and tokens, and state the loss aggregation (this chapter's §1).
- Known measurement errors: GPT-4-judged MT-Bench in DMT; relative-versus-point wording and non-reproducible averages in LIMI; AIME'24 with 30 questions where 3.33 points is one question (Qwen3 §4.6).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Copying a share without its unit | A "0.1% long-context" mixture copied as a token share has one forty-fifth of the source's long-context tokens | Compute example, token, and loss-token shares with the calculator before training |
| Changing loss aggregation while keeping the mixture | Long-response domains gain or lose weight; per-skill scores move without a data change | Log per-domain loss-token counts per step; compare per-sequence and token-mean runs |
| Judging a mixture change on the average only | Average moves 1 point while one skill moves 19 points | Per-skill delta table for every change (Tülu 3 Table 10 format) |
| Treating a 1-point single-benchmark change as a result | Deltas flip sign between reruns | Run at least two seeds per mixture and report the spread per benchmark |
| Tuning a subset on its own development benchmark | Development score rises; paired unseen benchmark flat or lower | Pair each development benchmark with an unseen one of the same skill |
| Training skills in sequence without replay | Earlier skill drops after the later stage (GSM8K −17.71 in DMT) | Evaluate all skills after each stage; add a replay fraction (1/256 in DMT) |
| Agent-only or single-domain agent SFT | Held-in agent tasks rise; held-out agent tasks and chat benchmarks fall | Evaluate held-out agent tasks and non-agentic benchmarks together |
| Balancing agent sources by raw counts | One synthetic source dominates sampled examples (Orca at 91.71% of raw non-web ADP rows) | Apply per-dataset multipliers and print sampled shares |
| Assuming long-context ability survives short SFT, or assuming synthetic long SFT helps | Long-context benchmark drops after SFT, or drops after adding synthetic long data | Measure long-context tasks after SFT for 0% and for a 0.1%-of-examples long share in the same setup |
| Evaluating a two-mode model in one mode | Reasoning-off or chat regressions go unreported | Report both modes and mode-switch accuracy |
| Quality-only data selection | Near-duplicate selections; worst-case prompts fail | Measure pool coverage (facility location) and bottom-10% scores |

## Check your understanding

1. DMT finds that mixing helps each skill at 1/256 of each dataset and hurts it at the full size. Give a mechanism consistent with both results, and name a measurement that would separate your mechanism from the authors' "data from other sources could be viewed as noise" interpretation.
2. Llama 3.1's long-context examples are 0.11% of examples, 4.95% of tokens, and 0.26% of final-response tokens. Under which loss mask and aggregation rule does each number describe the gradient weight, and why does the answer change the comparison with ProLong?
3. Removing Persona data from Tülu 3 8B lowers IFEval by 19.2 points and changes IFEval-OOD by +0.4. Explain what this implies about what the Persona instruction-following data taught, and decide how you would test whether to keep it.
4. AgentTuning's agent-only model scores 1.34 on held-in agent tasks and 0.09 on held-out agent tasks, against 1.96 and 0.67 for the mixed model. Explain why removing general chat data can lower performance on unseen agent tasks, and why the mixing advantage over general-only data appears only at 70B.
5. List three differences between the Llama 3 and ProLong settings that could explain their opposite long-context SFT results, and design one experiment that would test the most likely explanation.
6. LIMI's 78 samples outperform a 10,000-sample set on AgencyBench. State which variables differ besides sample count, and explain why the comparison does not establish that curation is more effective than scale for general agentic ability.
7. Qwen3-32B loses 2.4 AIME'24 points in thinking mode after mode fusion and general RL. Using the evaluation protocol in the Qwen3 report's §4.6, explain whether this change is distinguishable from sampling noise and what data the report would need to print to decide.
8. In the facility-location example, quality-only selection picks two near-duplicate prompts. Explain why this can lower worst-case instruction-following performance even when average quality is higher, and when quality-only selection would be the better choice.

## Connections

- Dependencies: ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix (the Mix step whose evidence is taught here); ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control (forgetting measurement and replay).
- Previous chapter: ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control.
- Next chapter: ch-30c — Weight Averaging and Model Merging for Generalist Models (merging separately trained models as the alternative to mixing data).
- Related: ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate (loss aggregation, implicit negatives); ch-29e — Instruction Tuning and Generalization to Unseen Tasks (task diversity); ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation; ch-34 — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports (Qwen3 mode fusion in context); ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data (expert-to-unified placement); ch-35a — Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters (thinking on/off pairs); ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL; ch-47 — Evaluation Harness and Suite Design for General Capability; ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting.

## Sources

- [[sft-data-composition-dmt]] — DMT amount, ratio, and order experiments; Table 1 strategies; replay fraction k (chapter excerpt from arXiv:2310.05492v4).
- [[tulu-1-how-far-can-camels-go]] — 12-dataset comparison at LLaMA 13B; no single dataset best.
- [[tulu-3]] — SFT mixture construction, Table 10 removal ablations, Table 14 seeds, Table 32 unseen evaluations, Table 11 settings, §4.3.2 sum loss (values read at arXiv:2411.15124v5 on 2026-09-15).
- [[tulu-3-sft-mix]] — released component list of the SFT mixture; component counts in this chapter are taken from the paper's Table 7.
- [[llama-3]] — Table 7 SFT composition, code and multilingual experts, 0.1% long-context share, tool-use queries without tool need.
- [[llama-3-recipe]] — verified SFT ledger rows and data-pruning signals (InsTag difficulty, semantic deduplication).
- [[long-context-llama3]] — owner card for Llama 3 long-context material; not used for numbers because it has not been re-verified.
- [[glm-4-5]] — Reasoning, Agent, and General-chat experts distilled into one hybrid model.
- [[qwen-3]] — Thinking Mode Fusion template and Table 22 stage-by-stage scores (values read at arXiv:2505.09388v1 on 2026-09-15).
- [[deepseek-r1]] — Dev3 reasoning and non-reasoning SFT data and Table 3 stage results.
- [[deepseek-r1-recipe]] — Table 5 domain counts and the SFT epoch conflict.
- [[agenttuning]] — η-weighted mixture objective and agent-only versus general-only versus mixed ablation.
- [[agent-flan]] — chat-format alignment, capability weights, and negative-as-content samples.
- [[agent-data-protocol]] — per-dataset sampling multipliers, web exclusion, and diverse versus single-domain results (chapter excerpt).
- [[limi]] — 78-sample agentic SFT, single-domain narrowing on tau2, and measurement caveats (chapter excerpt).
- [[llama-nemotron]] — reasoning on/off paired data, Table 2 split, staged SFT, IFEval drop and recovery (chapter excerpt).
- [[prolong]] — Table 8 synthetic long SFT share, hypotheses for the conflict, token-averaged loss.
- [[prolong-recipe]] — verified ProLong SFT settings.
- [[qdit-data-diversity]] — facility-location diversity, quality-diversity trade-off, worst-case evaluation (chapter excerpt).
- [[deita]] — embedding-threshold diversity filter compared with InsTag diversity.
- [[instag]] — intent-tagging method used by DMT and Llama 3.
- [[mix-data-or-merge-models]] — one mix-versus-merge comparison, as a pointer to ch-30c.
- [[am-reasoning-answers-for-non-reasoning]] — answer-only versus summary targets for a non-reasoning student.
- [[nemotron-ultra]] — qualitative statement on staged versus simultaneous multi-capability RL.
